#include "direct-executor.hpp"

#include "direct-builder.hpp"
#include "../../ggml-gemmini-args.h"
#include "../../quants/common/weight_reader.hpp"
#include "../../quants/common/weight_route.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace ggml::gemmini::residual {

namespace {

namespace wreader = quants::wreader;
namespace wroute = quants::wroute;
using WeightFormat = ggml_gemmini_args_t::im2p_weight_format_t;

constexpr size_t kJTile = 16;
constexpr __int128 kInt64Min = static_cast<__int128>(std::numeric_limits<int64_t>::min());
constexpr __int128 kInt64Max = static_cast<__int128>(std::numeric_limits<int64_t>::max());

bool checked_add(int64_t lhs, int64_t rhs, int64_t & result) {
    const __int128 sum = static_cast<__int128>(lhs) + rhs;
    if (sum < kInt64Min || sum > kInt64Max) return false;
    result = static_cast<int64_t>(sum);
    return true;
}

bool checked_multiply(int64_t lhs, int64_t rhs, int64_t & result) {
    const __int128 product = static_cast<__int128>(lhs) * rhs;
    if (product < kInt64Min || product > kInt64Max) return false;
    result = static_cast<int64_t>(product);
    return true;
}

bool checked_size_product(size_t lhs, size_t rhs, size_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) return false;
    result = lhs * rhs;
    return true;
}

bool dense_route_is_addressable(const wroute::WeightRoutePlan & plan,
                                size_t k_count, size_t j_count) {
    if (plan.native_weight_blocks) return true;
    if (plan.weight_stride == 0) return false;
    const bool column_major = plan.layout == wroute::WeightLayout::JxK_ColMajor;
    const size_t major_count = column_major ? j_count : k_count;
    const size_t minor_count = column_major ? k_count : j_count;
    if (plan.weight_stride < minor_count) return false;
    size_t major_offset = 0;
    return checked_size_product(major_count - 1, plan.weight_stride, major_offset) &&
        minor_count - 1 <= std::numeric_limits<size_t>::max() - major_offset;
}

bool uses_reader_scale(const wroute::WeightRoutePlan & plan) {
    return plan.route == wroute::WeightRouteKind::H0 ||
        plan.route == wroute::WeightRouteKind::H1 ||
        plan.route == wroute::WeightRouteKind::HP1;
}

const int8_t * native_q8_codes(const ggml_gemmini_args_t & args,
                               size_t j, size_t block_id) {
    switch (args.weight_format) {
        case WeightFormat::q8_h1: {
            const block_q8_h1 * block = args.q8_h1_block(j, block_id);
            return block == nullptr ? nullptr : block->qs;
        }
        case WeightFormat::q8_hp1: {
            const block_q8_hp1 * block = args.q8_hp1_block(j, block_id);
            return block == nullptr ? nullptr : block->qs;
        }
        default:
            return nullptr;
    }
}

bool finite_double(double value) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double representation");
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

rmd::RmdStatus reader_failure(wreader::WeightReaderStatus status) {
    return status == wreader::WeightReaderStatus::ScaleOverflow ?
        rmd::RmdStatus::overflow : rmd::RmdStatus::execution_failed;
}

}

rmd::RmdStatus execute_direct_stripe(const ggml_gemmini_args_t & args,
                                     const DirectStripePayload & payload,
                                     rmd::DirectOutput & correction,
                                     DirectExecutionMetrics * metrics) {
    if (validate_direct_payload(payload) != rmd::RmdStatus::success)
        return rmd::RmdStatus::invalid_packet;
    if (args.K != payload.logical_k || args.J != payload.logical_j)
        return rmd::RmdStatus::invalid_arguments;

    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid) {
        if (wreader::validate(args, plan) == wreader::WeightReaderStatus::ScaleOverflow)
            return rmd::RmdStatus::overflow;
        return rmd::RmdStatus::unsupported_route;
    }
    if (wroute::weight_route_status(plan, wroute::WeightExecutionPath::CpuDirect) !=
        wroute::WeightRouteStatus::Success) {
        return rmd::RmdStatus::unsupported_route;
    }

    const bool floating_block =
        plan.scale_domain == wroute::WeightScaleDomain::FloatingBlock;
    const bool integer_block =
        plan.scale_domain == wroute::WeightScaleDomain::IntegerBlockTimesColumn;
    if ((floating_block && plan.route != wroute::WeightRouteKind::H0) ||
        (!floating_block && (!integer_block ||
         !wroute::route_supports_integer_block_scale(plan)))) {
        return rmd::RmdStatus::unsupported_route;
    }
    if (!wroute::route_covers_k(plan, payload.logical_k) ||
        !dense_route_is_addressable(plan, payload.logical_k, payload.logical_j)) {
        return rmd::RmdStatus::unsupported_route;
    }

    size_t output_count = 0;
    if (!checked_size_product(payload.row_count, payload.logical_j, output_count))
        return rmd::RmdStatus::overflow;

    std::vector<rmd::OutputValue> staged_integer;
    std::vector<double> staged_floating;
    if ((integer_block && output_count > staged_integer.max_size()) ||
        (floating_block && output_count > staged_floating.max_size())) {
        return rmd::RmdStatus::allocation_failure;
    }
    try {
        if (floating_block) {
            staged_floating.assign(output_count, 0.0);
        } else {
            staged_integer.assign(output_count, rmd::OutputValue{0});
        }
    } catch (const std::bad_alloc &) {
        return rmd::RmdStatus::allocation_failure;
    }

    const size_t j_tile_count =
        (payload.logical_j + kJTile - 1) / kJTile;
    std::vector<rmd::RmdStatus> tile_status;
    std::vector<size_t> tile_native_q8_values;
    try {
        tile_status.assign(j_tile_count, rmd::RmdStatus::success);
        tile_native_q8_values.assign(j_tile_count, 0);
    } catch (const std::bad_alloc &) {
        return rmd::RmdStatus::allocation_failure;
    }

    // Events are canonical row/K order. For each J tile, consume contiguous
    // row/block/K spans, then apply that block's scale exactly once.
    auto execute_j_tile = [&](size_t tile_index) {
        const size_t j_begin = tile_index * kJTile;
        const size_t tile_j = std::min(kJTile, payload.logical_j - j_begin);
        size_t & native_q8_values = tile_native_q8_values[tile_index];
        size_t event_index = 0;
        while (event_index < payload.events.size()) {
            const ResidualEvent & first = payload.events[event_index];
            const size_t row = first.local_row;
            const size_t block_id = first.original_k / rmd::kBlockSize;
            size_t span_end = event_index + 1;
            while (span_end < payload.events.size() &&
                   payload.events[span_end].local_row == row &&
                   payload.events[span_end].original_k / rmd::kBlockSize == block_id) {
                ++span_end;
            }

            std::array<int64_t, kJTile> block_sum{};
            for (size_t local_j = 0; local_j < tile_j; ++local_j) {
                const size_t j = j_begin + local_j;
                const int8_t * codes = native_q8_codes(args, j, block_id);
                if (codes != nullptr) {
                    for (size_t index = event_index; index < span_end; ++index) {
                        const ResidualEvent & event = payload.events[index];
                        block_sum[local_j] +=
                            static_cast<int64_t>(event.residual) *
                            codes[event.original_k % rmd::kBlockSize];
                    }
                    native_q8_values += span_end - event_index;
                    continue;
                }
                for (size_t index = event_index; index < span_end; ++index) {
                    const ResidualEvent & event = payload.events[index];
                    const wreader::WeightCodeResult code = wreader::read_code_validated(
                        args, plan, j, event.original_k);
                    if (!code.ok()) return reader_failure(code.status);
                    // A block has at most 32 signed INT16 codes and INT32
                    // residuals, so its complete dot product fits in INT64.
                    block_sum[local_j] +=
                        static_cast<int64_t>(event.residual) * code.value;
                }
            }

            for (size_t local_j = 0; local_j < tile_j; ++local_j) {
                const size_t j = j_begin + local_j;
                wreader::WeightScaleResult scale{};
                if (uses_reader_scale(plan)) {
                    scale = wreader::read_scale_validated(args, plan, j, block_id);
                    if (!scale.ok()) return reader_failure(scale.status);
                } else {
                    scale.status = wreader::WeightReaderStatus::Success;
                    scale.domain = plan.scale_domain;
                    scale.integer_block_scale =
                        wroute::route_block_scale(plan, args, j, block_id);
                }
                if (scale.domain != plan.scale_domain)
                    return rmd::RmdStatus::execution_failed;

                const size_t output_index = row * payload.logical_j + j;
                if (floating_block) {
                    const double scaled = static_cast<double>(block_sum[local_j]) *
                        static_cast<double>(scale.floating_block_scale);
                    const double sum = staged_floating[output_index] + scaled;
                    if (!finite_double(scaled) || !finite_double(sum))
                        return rmd::RmdStatus::overflow;
                    staged_floating[output_index] = sum;
                } else {
                    if (scale.integer_block_scale >
                        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                        return rmd::RmdStatus::overflow;
                    }
                    int64_t scaled = 0;
                    if (!checked_multiply(
                            block_sum[local_j],
                            static_cast<int64_t>(scale.integer_block_scale), scaled) ||
                        !checked_add(staged_integer[output_index], scaled,
                                     staged_integer[output_index])) {
                        return rmd::RmdStatus::overflow;
                    }
                }
            }
            event_index = span_end;
        }
        return rmd::RmdStatus::success;
    };

#if defined(GGML_GEMMINI_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(j_tile_count > 1)
#endif
    for (std::ptrdiff_t tile_index = 0;
         tile_index < static_cast<std::ptrdiff_t>(j_tile_count);
         ++tile_index) {
        tile_status[static_cast<size_t>(tile_index)] =
            execute_j_tile(static_cast<size_t>(tile_index));
    }
    size_t native_q8_values = 0;
    for (size_t tile_index = 0; tile_index < j_tile_count; ++tile_index) {
        if (tile_status[tile_index] != rmd::RmdStatus::success) {
            return tile_status[tile_index];
        }
        native_q8_values += tile_native_q8_values[tile_index];
    }

    rmd::DirectOutput staged_output = floating_block ?
        rmd::DirectOutput(rmd::PreScaledFloat64Correction{std::move(staged_floating)}) :
        rmd::DirectOutput(rmd::BlockScaledInt64Correction{std::move(staged_integer)});
    correction.swap(staged_output);
    if (metrics != nullptr) {
        metrics->event_count = payload.events.size();
        metrics->call_count = 1;
        metrics->native_q8_values = native_q8_values;
        metrics->j_tile_count = j_tile_count;
    }
    return rmd::RmdStatus::success;
}

}
