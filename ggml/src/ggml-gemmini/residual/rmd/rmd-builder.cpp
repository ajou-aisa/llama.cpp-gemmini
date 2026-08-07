#include "rmd-builder.hpp"

#include <algorithm>
#include <limits>
#include <utility>
#include <new>

namespace ggml::gemmini::rmd {

namespace {

bool checked_mul(size_t lhs, size_t rhs, size_t & out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool checked_add(size_t lhs, size_t rhs, size_t & out) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        return false;
    }
    out = lhs + rhs;
    return true;
}

}

const char * rmd_status_message(RmdStatus status) {
    switch (status) {
        case RmdStatus::success:            return "success";
        case RmdStatus::invalid_arguments:  return "rmd: invalid arguments";
        case RmdStatus::invalid_packet:     return "rmd: invalid packet";
        case RmdStatus::residual_too_wide:  return "rmd: residual exceeds four balanced radix-256 digits";
        case RmdStatus::unsupported_route:  return "rmd: weight route has no integer block scale";
        case RmdStatus::overflow:           return "rmd: integer overflow";
        case RmdStatus::allocation_failure: return "rmd: allocation failed";
        case RmdStatus::execution_failed:   return "rmd: execution failed";
    }
    return "rmd: unknown status";
}

bool decompose_balanced_radix256(int32_t residual, BalancedDigits & out) {
    out = BalancedDigits{};
    int64_t rest = residual;
    for (size_t lane = 0; lane < kMaxLanes; ++lane) {
        // Balanced digit in [-128, 127] without any signed shift.
        int32_t digit = static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(rest)) & 0xFFu);
        if (digit > 127) {
            digit -= 256;
        }
        out.digits[lane] = static_cast<int8_t>(digit);
        if (digit != 0) {
            out.lane_mask |= static_cast<uint8_t>(1u << lane);
        }
        rest = (rest - digit) / 256;
    }
    if (rest != 0) {
        out = BalancedDigits{};
        return false;
    }
    return true;
}

int64_t compose_balanced_radix256(const BalancedDigits & digits) {
    int64_t place = 1;
    int64_t value = 0;
    for (size_t lane = 0; lane < kMaxLanes; ++lane) {
        value += static_cast<int64_t>(digits.digits[lane]) * place;
        place *= 256;
    }
    return value;
}

void RmdStripeBuilder::reset(size_t stripe_id, size_t row_begin, size_t row_count,
                             size_t logical_k, size_t logical_j) {
    status_ = RmdStatus::success;
    stripe_id_ = stripe_id;
    row_begin_ = row_begin;
    row_count_ = row_count;
    logical_k_ = logical_k;
    logical_j_ = logical_j;
    entries_.clear();
    blocks_.clear();
    if (row_count == 0 || logical_k == 0 || logical_j == 0 ||
        logical_k > std::numeric_limits<uint32_t>::max() ||
        row_count > std::numeric_limits<uint16_t>::max()) {
        status_ = RmdStatus::invalid_arguments;
    }
}

bool RmdStripeBuilder::add_residual(size_t local_row, size_t original_k, int32_t residual) {
    if (status_ != RmdStatus::success) {
        return false;
    }
    if (local_row >= row_count_ || original_k >= logical_k_) {
        status_ = RmdStatus::invalid_arguments;
        return false;
    }
    if (residual == 0) {
        return true;
    }

    BalancedDigits digits{};
    if (!decompose_balanced_radix256(residual, digits)) {
        status_ = RmdStatus::residual_too_wide;
        return false;
    }

    const size_t block_id = original_k / kBlockSize;
    const size_t block_local_k = original_k % kBlockSize;
    if (block_id > std::numeric_limits<uint32_t>::max()) {
        status_ = RmdStatus::invalid_arguments;
        return false;
    }

    try {
        BlockAccum & accum = blocks_[static_cast<uint32_t>(block_id)];
        accum.k.insert(static_cast<uint16_t>(block_local_k));
        accum.lane_mask |= digits.lane_mask;
        for (size_t lane = 0; lane < kMaxLanes; ++lane) {
            if (digits.digits[lane] == 0) {
                continue;
            }
            entries_.push_back({
                static_cast<uint32_t>(block_id),
                static_cast<uint32_t>(local_row),
                static_cast<uint16_t>(block_local_k),
                static_cast<uint8_t>(lane),
                digits.digits[lane],
            });
        }
    } catch (const std::bad_alloc &) {
        status_ = RmdStatus::allocation_failure;
        return false;
    }
    return true;
}

StripePacketHandle RmdStripeBuilder::finish() {
    if (status_ != RmdStatus::success || entries_.empty()) {
        return nullptr;
    }

    try {
        auto packet = std::make_shared<StripePacket>();
        packet->version = kPacketVersion;
        packet->stripe_id = stripe_id_;
        packet->row_begin = row_begin_;
        packet->row_count = row_count_;
        packet->logical_k = logical_k_;
        packet->logical_j = logical_j_;
        packet->j_padded = align_up(logical_j_, kArrayDim);
        if (packet->j_padded == 0) {
            status_ = RmdStatus::overflow;
            return nullptr;
        }
        packet->block_size = kBlockSize;
        packet->array_dim = kArrayDim;

        const size_t rows_padded = align_up(row_count_, kArrayDim);
        if (rows_padded == 0) {
            status_ = RmdStatus::overflow;
            return nullptr;
        }
        if (rows_padded > std::numeric_limits<uint16_t>::max()) {
            status_ = RmdStatus::invalid_arguments;
            return nullptr;
        }

        size_t lane_stride_values = 0;
        if (!checked_mul(rows_padded, packet->j_padded, lane_stride_values) ||
            lane_stride_values > std::numeric_limits<uint32_t>::max()) {
            status_ = RmdStatus::overflow;
            return nullptr;
        }

        // Blocks are emitted in ascending original block id; std::map keeps that order.
        packet->blocks.reserve(blocks_.size());
        size_t k_cursor = 0;
        size_t activation_cursor = 0;
        size_t output_cursor = 0;
        std::map<uint32_t, size_t> block_index_by_id;
        // block id -> (block-local k -> compact index)
        std::map<uint32_t, std::vector<uint16_t>> compact_index_by_block;

        for (const auto & [block_id, accum] : blocks_) {
            BlockDescriptor descriptor{};
            descriptor.block_id = block_id;
            size_t global_k_begin = 0;
            if (!checked_mul(block_id, kBlockSize, global_k_begin) ||
                global_k_begin > std::numeric_limits<uint32_t>::max()) {
                status_ = RmdStatus::overflow;
                return nullptr;
            }
            descriptor.global_k_begin = static_cast<uint32_t>(global_k_begin);
            descriptor.compact_k_count = static_cast<uint16_t>(accum.k.size());
            const size_t padded_k = align_up(accum.k.size(), kArrayDim);
            if (padded_k == 0) {
                status_ = RmdStatus::overflow;
                return nullptr;
            }
            if (padded_k > std::numeric_limits<uint16_t>::max()) {
                status_ = RmdStatus::overflow;
                return nullptr;
            }
            descriptor.padded_k_count = static_cast<uint16_t>(padded_k);
            descriptor.active_lane_mask = accum.lane_mask;
            descriptor.rows_padded = static_cast<uint16_t>(rows_padded);
            descriptor.lane_stride_values = static_cast<uint32_t>(lane_stride_values);

            uint8_t lane_count = 0;
            for (uint8_t lane = 0; lane < kMaxLanes; ++lane) {
                if ((accum.lane_mask & static_cast<uint8_t>(1u << lane)) != 0) {
                    descriptor.lane_ids[lane_count] = lane;
                    ++lane_count;
                }
            }
            descriptor.active_lane_count = lane_count;

            if (k_cursor > std::numeric_limits<uint32_t>::max() ||
                activation_cursor > std::numeric_limits<uint32_t>::max() ||
                output_cursor > std::numeric_limits<uint32_t>::max()) {
                status_ = RmdStatus::overflow;
                return nullptr;
            }
            descriptor.k_index_offset = static_cast<uint32_t>(k_cursor);
            descriptor.activation_offset = static_cast<uint32_t>(activation_cursor);
            descriptor.output_value_offset = static_cast<uint32_t>(output_cursor);

            // Compact K index table for this block: ascending, deduplicated, block local.
            std::vector<uint16_t> compact(kBlockSize, std::numeric_limits<uint16_t>::max());
            uint16_t compact_index = 0;
            for (const uint16_t local_k : accum.k) {
                compact[local_k] = compact_index++;
                packet->k_indices.push_back(local_k);
            }
            compact_index_by_block.emplace(block_id, std::move(compact));

            size_t lane_activation = 0;
            size_t block_activation = 0;
            size_t block_output = 0;
            if (!checked_mul(rows_padded, padded_k, lane_activation) ||
                !checked_mul(lane_activation, lane_count, block_activation) ||
                !checked_mul(lane_stride_values, lane_count, block_output) ||
                !checked_add(k_cursor, accum.k.size(), k_cursor) ||
                !checked_add(activation_cursor, block_activation, activation_cursor) ||
                !checked_add(output_cursor, block_output, output_cursor)) {
                status_ = RmdStatus::overflow;
                return nullptr;
            }

            block_index_by_id.emplace(block_id, packet->blocks.size());
            packet->blocks.push_back(descriptor);
        }

        packet->total_output_values = output_cursor;
        // Padded rows, padded K slots and inactive lanes stay zero by construction.
        packet->stacked_activation.assign(activation_cursor, 0);

        for (const DigitEntry & entry : entries_) {
            const BlockDescriptor & descriptor = packet->blocks[block_index_by_id[entry.block_id]];
            const uint16_t compact_k =
                compact_index_by_block[entry.block_id][entry.block_local_k];
            uint8_t lane_position = kMaxLanes;
            for (uint8_t position = 0; position < descriptor.active_lane_count; ++position) {
                if (descriptor.lane_ids[position] == entry.lane) {
                    lane_position = position;
                    break;
                }
            }
            if (lane_position == kMaxLanes || compact_k >= descriptor.compact_k_count) {
                status_ = RmdStatus::invalid_packet;
                return nullptr;
            }
            const size_t index = descriptor.activation_offset +
                static_cast<size_t>(lane_position) * descriptor.rows_padded * descriptor.padded_k_count +
                static_cast<size_t>(entry.local_row) * descriptor.padded_k_count +
                compact_k;
            packet->stacked_activation[index] = entry.digit;
        }

        const RmdStatus validation = validate_packet(*packet);
        if (validation != RmdStatus::success) {
            status_ = validation;
            return nullptr;
        }
        return packet;
    } catch (const std::bad_alloc &) {
        status_ = RmdStatus::allocation_failure;
        return nullptr;
    }
}

StripePacketHandle slice_packets(const std::vector<StripePacketHandle> & packets,
                                 size_t row_begin,
                                 size_t row_end,
                                 size_t stripe_id,
                                 RmdStatus & status) {
    status = RmdStatus::success;
    if (row_begin >= row_end) {
        status = RmdStatus::invalid_arguments;
        return nullptr;
    }

    size_t logical_k = 0;
    size_t logical_j = 0;
    // (local row, original K) -> residual, rebuilt from the balanced digits.
    std::map<std::pair<uint32_t, uint32_t>, int64_t> residuals;
    for (const StripePacketHandle & handle : packets) {
        if (!handle) {
            continue;
        }
        const StripePacket & packet = *handle;
        logical_k = packet.logical_k;
        logical_j = packet.logical_j;
        if (packet.row_begin >= row_end || packet.row_begin + packet.row_count <= row_begin) {
            continue;
        }
        for (const BlockDescriptor & block : packet.blocks) {
            for (uint8_t position = 0; position < block.active_lane_count; ++position) {
                const uint8_t lane_id = block.lane_ids[position];
                int64_t place = 1;
                for (uint8_t step = 0; step < lane_id; ++step) {
                    place *= 256;
                }
                const size_t lane_base = block.activation_offset +
                    static_cast<size_t>(position) * block.rows_padded * block.padded_k_count;
                for (size_t row = 0; row < packet.row_count; ++row) {
                    const size_t global_row = packet.row_begin + row;
                    if (global_row < row_begin || global_row >= row_end) {
                        continue;
                    }
                    const int8_t * source =
                        packet.stacked_activation.data() + lane_base + row * block.padded_k_count;
                    for (size_t k = 0; k < block.compact_k_count; ++k) {
                        if (source[k] == 0) {
                            continue;
                        }
                        const uint32_t column = block.global_k_begin +
                            packet.k_indices[block.k_index_offset + k];
                        residuals[{static_cast<uint32_t>(global_row - row_begin), column}] +=
                            static_cast<int64_t>(source[k]) * place;
                    }
                }
            }
        }
    }

    if (residuals.empty()) {
        return nullptr;
    }

    RmdStripeBuilder builder;
    builder.reset(stripe_id, row_begin, row_end - row_begin, logical_k, logical_j);
    for (const auto & [key, value] : residuals) {
        if (value > std::numeric_limits<int32_t>::max() ||
            value < std::numeric_limits<int32_t>::min()) {
            status = RmdStatus::overflow;
            return nullptr;
        }
        if (!builder.add_residual(key.first, key.second, static_cast<int32_t>(value))) {
            status = builder.status();
            return nullptr;
        }
    }
    StripePacketHandle packet = builder.finish();
    status = builder.status();
    return packet;
}

RmdStatus validate_packet(const StripePacket & packet) {
    if (packet.version != kPacketVersion ||
        packet.block_size != kBlockSize || packet.array_dim != kArrayDim) {
        return RmdStatus::invalid_packet;
    }
    if (packet.row_count == 0 || packet.logical_j == 0 || packet.logical_k == 0 ||
        packet.blocks.empty()) {
        return RmdStatus::invalid_packet;
    }
    if (packet.j_padded != align_up(packet.logical_j, kArrayDim)) {
        return RmdStatus::invalid_packet;
    }

    const size_t rows_padded = align_up(packet.row_count, kArrayDim);
    size_t expected_k_cursor = 0;
    size_t expected_activation = 0;
    size_t expected_output = 0;
    uint64_t previous_block_id = 0;
    bool has_previous = false;

    for (const BlockDescriptor & block : packet.blocks) {
        if (has_previous && block.block_id <= previous_block_id) {
            return RmdStatus::invalid_packet; // strictly ascending original block ids
        }
        previous_block_id = block.block_id;
        has_previous = true;

        if (static_cast<size_t>(block.global_k_begin) != static_cast<size_t>(block.block_id) * kBlockSize) {
            return RmdStatus::invalid_packet;
        }
        if (block.compact_k_count == 0 || block.compact_k_count > kBlockSize) {
            return RmdStatus::invalid_packet;
        }
        if (block.padded_k_count != align_up(block.compact_k_count, kArrayDim) ||
            block.padded_k_count % kArrayDim != 0) {
            return RmdStatus::invalid_packet;
        }
        if (block.rows_padded != rows_padded || block.rows_padded % kArrayDim != 0) {
            return RmdStatus::invalid_packet;
        }
        if (block.lane_stride_values != rows_padded * packet.j_padded) {
            return RmdStatus::invalid_packet;
        }
        if (block.active_lane_mask == 0 || block.active_lane_count == 0 ||
            block.active_lane_count > kMaxLanes) {
            return RmdStatus::invalid_packet;
        }

        // lane mask must agree with lane_ids: same population, ascending, unique, 0..3.
        uint8_t rebuilt_mask = 0;
        for (uint8_t position = 0; position < block.active_lane_count; ++position) {
            const uint8_t lane_id = block.lane_ids[position];
            if (lane_id >= kMaxLanes) {
                return RmdStatus::invalid_packet;
            }
            const uint8_t bit = static_cast<uint8_t>(1u << lane_id);
            if ((rebuilt_mask & bit) != 0) {
                return RmdStatus::invalid_packet; // duplicate lane id
            }
            if (position != 0 && lane_id <= block.lane_ids[position - 1]) {
                return RmdStatus::invalid_packet;
            }
            rebuilt_mask |= bit;
        }
        if (rebuilt_mask != block.active_lane_mask) {
            return RmdStatus::invalid_packet;
        }

        if (block.k_index_offset != expected_k_cursor ||
            block.activation_offset != expected_activation ||
            block.output_value_offset != expected_output) {
            return RmdStatus::invalid_packet; // offsets must tile the buffers without gaps
        }

        // Selected K must be ascending, unique and inside the block.
        for (size_t i = 0; i < block.compact_k_count; ++i) {
            const size_t index = block.k_index_offset + i;
            if (index >= packet.k_indices.size()) {
                return RmdStatus::invalid_packet;
            }
            const uint16_t local_k = packet.k_indices[index];
            if (local_k >= kBlockSize) {
                return RmdStatus::invalid_packet;
            }
            if (i != 0 && local_k <= packet.k_indices[index - 1]) {
                return RmdStatus::invalid_packet;
            }
            if (static_cast<size_t>(block.global_k_begin) + local_k >= packet.logical_k) {
                return RmdStatus::invalid_packet;
            }
        }

        expected_k_cursor += block.compact_k_count;
        expected_activation += static_cast<size_t>(block.active_lane_count) *
            block.rows_padded * block.padded_k_count;
        expected_output += static_cast<size_t>(block.active_lane_count) * block.lane_stride_values;
    }

    if (packet.k_indices.size() != expected_k_cursor ||
        packet.stacked_activation.size() != expected_activation ||
        packet.total_output_values != expected_output) {
        return RmdStatus::invalid_packet;
    }

    // Padded rows and padded K slots must be zero.
    for (const BlockDescriptor & block : packet.blocks) {
        for (uint8_t position = 0; position < block.active_lane_count; ++position) {
            const size_t lane_base = block.activation_offset +
                static_cast<size_t>(position) * block.rows_padded * block.padded_k_count;
            for (size_t row = 0; row < block.rows_padded; ++row) {
                const size_t row_base = lane_base + row * block.padded_k_count;
                const size_t valid_k = row < packet.row_count ? block.compact_k_count : 0;
                for (size_t k = valid_k; k < block.padded_k_count; ++k) {
                    if (packet.stacked_activation[row_base + k] != 0) {
                        return RmdStatus::invalid_packet;
                    }
                }
            }
        }
    }

    return RmdStatus::success;
}

}
