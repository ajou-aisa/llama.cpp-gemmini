#pragma once

#include "direct-types.hpp"
#include "../rmd/rmd-types.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace ggml::gemmini::residual {

inline rmd::RmdStatus validate_direct_payload(const DirectStripePayload &payload) {
    if (payload.row_count == 0 || payload.logical_k == 0 || payload.logical_j == 0 ||
        payload.events.empty() ||
        payload.row_count > std::numeric_limits<size_t>::max() - payload.row_begin) {
        return rmd::RmdStatus::invalid_packet;
    }
    for (size_t i = 0; i < payload.events.size(); ++i) {
        const ResidualEvent &event = payload.events[i];
        if (event.local_row >= payload.row_count || event.original_k >= payload.logical_k ||
            event.residual == 0) {
            return rmd::RmdStatus::invalid_packet;
        }
        if (i != 0) {
            const ResidualEvent &previous = payload.events[i - 1];
            if (std::tie(event.local_row, event.original_k) <=
                std::tie(previous.local_row, previous.original_k)) {
                return rmd::RmdStatus::invalid_packet;
            }
        }
    }
    return rmd::RmdStatus::success;
}

inline rmd::RmdStatus expand_direct_payloads_to_plane(
        const std::vector<DirectStripePayloadHandle> & payloads,
        size_t global_row_begin,
        size_t global_row_end,
        size_t logical_k,
        size_t logical_j,
        size_t col_count,
        std::vector<int32_t> & plane) {
    if (global_row_begin >= global_row_end || logical_k == 0 || logical_j == 0 ||
        col_count == 0 || col_count > logical_k) {
        return rmd::RmdStatus::invalid_arguments;
    }
    const size_t row_count = global_row_end - global_row_begin;
    if (row_count > std::numeric_limits<size_t>::max() / col_count) {
        return rmd::RmdStatus::overflow;
    }
    const size_t value_count = row_count * col_count;

    std::vector<DirectStripePayloadHandle> ordered;
    std::vector<int32_t> staged;
    std::vector<uint8_t> seen;
    try {
        ordered = payloads;
        staged.assign(value_count, 0);
        seen.assign(value_count, 0);
    } catch (const std::bad_alloc &) {
        return rmd::RmdStatus::allocation_failure;
    } catch (const std::length_error &) {
        return rmd::RmdStatus::allocation_failure;
    }

    for (size_t index = 0; index < ordered.size(); ++index) {
        const DirectStripePayloadHandle & handle = ordered[index];
        if (!handle ||
            validate_direct_payload(*handle) != rmd::RmdStatus::success ||
            handle->logical_k != logical_k ||
            handle->logical_j != logical_j) {
            return rmd::RmdStatus::invalid_packet;
        }
        for (size_t previous = 0; previous < index; ++previous) {
            if (ordered[previous] && ordered[previous]->stripe_id == handle->stripe_id) {
                return rmd::RmdStatus::invalid_packet;
            }
        }
    }
    std::sort(
        ordered.begin(), ordered.end(),
        [](const DirectStripePayloadHandle & lhs,
           const DirectStripePayloadHandle & rhs) {
            return std::tie(lhs->row_begin, lhs->stripe_id) <
                std::tie(rhs->row_begin, rhs->stripe_id);
        });
    for (size_t index = 1; index < ordered.size(); ++index) {
        const DirectStripePayload & previous = *ordered[index - 1];
        const DirectStripePayload & current = *ordered[index];
        if (previous.row_begin + previous.row_count > current.row_begin) {
            return rmd::RmdStatus::invalid_packet;
        }
    }

    for (const DirectStripePayloadHandle & handle : ordered) {
        for (const ResidualEvent & event : handle->events) {
            const size_t global_row = handle->row_begin + event.local_row;
            if (global_row < global_row_begin || global_row >= global_row_end ||
                event.original_k >= col_count) {
                continue;
            }
            const size_t local_row = global_row - global_row_begin;
            const size_t offset = local_row * col_count + event.original_k;
            if (seen[offset] != 0) {
                return rmd::RmdStatus::invalid_packet;
            }
            seen[offset] = 1;
            staged[offset] = event.residual;
        }
    }
    plane.swap(staged);
    return rmd::RmdStatus::success;
}

class DirectStripeBuilder {
public:
    void reset(size_t stripe_id, size_t row_begin, size_t row_count,
               size_t logical_k, size_t logical_j) {
        status_ = rmd::RmdStatus::success;
        stripe_id_ = stripe_id;
        row_begin_ = row_begin;
        row_count_ = row_count;
        logical_k_ = logical_k;
        logical_j_ = logical_j;
        events_.clear();
        if (row_count == 0 || logical_k == 0 || logical_j == 0) {
            status_ = rmd::RmdStatus::invalid_arguments;
        } else if (row_count > std::numeric_limits<size_t>::max() - row_begin) {
            status_ = rmd::RmdStatus::overflow;
        }
    }

    bool add_residual(size_t local_row, size_t original_k, int32_t residual) {
        if (status_ != rmd::RmdStatus::success) {
            return false;
        }
        if (local_row >= row_count_ || original_k >= logical_k_) {
            status_ = rmd::RmdStatus::invalid_arguments;
            return false;
        }
        if (residual == 0) {
            return true;
        }
        try {
            events_.push_back({local_row, original_k, residual});
        } catch (const std::bad_alloc &) {
            status_ = rmd::RmdStatus::allocation_failure;
            return false;
        }
        return true;
    }

    bool empty() const { return events_.empty(); }
    rmd::RmdStatus status() const { return status_; }

    DirectStripePayloadHandle finish() {
        if (status_ != rmd::RmdStatus::success || events_.empty()) {
            return nullptr;
        }
        try {
            auto payload = std::make_shared<DirectStripePayload>();
            payload->stripe_id = stripe_id_;
            payload->row_begin = row_begin_;
            payload->row_count = row_count_;
            payload->logical_k = logical_k_;
            payload->logical_j = logical_j_;
            payload->events = events_;
            std::sort(payload->events.begin(), payload->events.end(),
                      [](const ResidualEvent &lhs, const ResidualEvent &rhs) {
                          return std::tie(lhs.local_row, lhs.original_k) <
                              std::tie(rhs.local_row, rhs.original_k);
                      });
            const rmd::RmdStatus validation = validate_direct_payload(*payload);
            if (validation != rmd::RmdStatus::success) {
                status_ = validation;
                return nullptr;
            }
            return payload;
        } catch (const std::bad_alloc &) {
            status_ = rmd::RmdStatus::allocation_failure;
            return nullptr;
        }
    }

private:
    rmd::RmdStatus status_ = rmd::RmdStatus::success;
    size_t stripe_id_ = 0;
    size_t row_begin_ = 0;
    size_t row_count_ = 0;
    size_t logical_k_ = 0;
    size_t logical_j_ = 0;
    std::vector<ResidualEvent> events_;
};

inline DirectStripePayloadHandle slice_direct_payloads(
        const std::vector<DirectStripePayloadHandle> &payloads,
        size_t row_begin, size_t row_end, size_t stripe_id, rmd::RmdStatus &status) {
    status = rmd::RmdStatus::success;
    if (row_begin >= row_end) {
        status = rmd::RmdStatus::invalid_arguments;
        return nullptr;
    }

    DirectStripePayloadHandle exact;
    size_t overlaps = 0;
    size_t logical_k = 0;
    size_t logical_j = 0;
    for (const auto &handle : payloads) {
        if (!handle) continue;
        if (handle->row_count > std::numeric_limits<size_t>::max() - handle->row_begin) {
            status = rmd::RmdStatus::invalid_packet;
            return nullptr;
        }
        const size_t end = handle->row_begin + handle->row_count;
        if (handle->row_begin >= row_end || end <= row_begin) continue;
        if (validate_direct_payload(*handle) != rmd::RmdStatus::success) {
            status = rmd::RmdStatus::invalid_packet;
            return nullptr;
        }
        if (overlaps == 0) {
            logical_k = handle->logical_k;
            logical_j = handle->logical_j;
        } else if (handle->logical_k != logical_k || handle->logical_j != logical_j) {
            status = rmd::RmdStatus::invalid_packet;
            return nullptr;
        }
        ++overlaps;
        if (handle->row_begin == row_begin && end == row_end &&
            handle->stripe_id == stripe_id) exact = handle;
    }
    if (overlaps == 1 && exact) {
        status = validate_direct_payload(*exact);
        return status == rmd::RmdStatus::success ? exact : nullptr;
    }
    if (overlaps == 0) return nullptr;

    DirectStripeBuilder builder;
    builder.reset(stripe_id, row_begin, row_end - row_begin, logical_k, logical_j);
    if (builder.status() != rmd::RmdStatus::success) {
        status = builder.status();
        return nullptr;
    }
    for (const auto &handle : payloads) {
        if (!handle) continue;
        const size_t handle_end = handle->row_begin + handle->row_count;
        if (handle->row_begin >= row_end || handle_end <= row_begin) continue;
        for (const ResidualEvent &event : handle->events) {
            const size_t global_row = handle->row_begin + event.local_row;
            if (global_row < row_begin || global_row >= row_end) continue;
            if (!builder.add_residual(global_row - row_begin, event.original_k, event.residual)) {
                status = builder.status();
                return nullptr;
            }
        }
    }
    DirectStripePayloadHandle result = builder.finish();
    status = builder.status();
    return result;
}

}
