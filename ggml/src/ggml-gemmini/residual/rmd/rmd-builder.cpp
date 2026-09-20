#include "rmd-builder.hpp"

#include "../../quants/common/weight_reader.hpp"

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

bool checked_activation_sizes(uint8_t digit_bits,
                              size_t packed_rows,
                              size_t padded_k_count,
                              size_t & value_count,
                              size_t & byte_count) {
    const size_t group_rows = align_up(packed_rows, kArrayDim);
    if (group_rows == 0 ||
        !checked_mul(group_rows, padded_k_count, value_count)) {
        return false;
    }
    if (digit_bits == 4) {
        if (padded_k_count % 2 != 0) {
            return false;
        }
        byte_count = value_count / 2;
        return true;
    }
    if (digit_bits == 8) {
        byte_count = value_count;
        return true;
    }
    return digit_bits == 16 && checked_mul(value_count, sizeof(int16_t), byte_count);
}

void choose_lane_partition(
    const std::array<uint32_t, kMaxNativeRadixLanes> & lane_support,
    uint8_t lane_count,
    size_t row_count,
    uint8_t lane_position,
    uint8_t group_count,
    std::array<uint8_t, kMaxNativeRadixLanes> & assignment,
    std::array<uint8_t, kMaxNativeRadixLanes> & best_assignment,
    size_t baseline_calls,
    size_t & best_i_tiles) {
    if (lane_position == lane_count) {
        std::array<uint32_t, kMaxNativeRadixLanes> group_support{};
        std::array<size_t, kMaxNativeRadixLanes> group_lanes{};
        for (uint8_t lane = 0; lane < lane_count; ++lane) {
            group_support[assignment[lane]] |= lane_support[lane];
            ++group_lanes[assignment[lane]];
        }
        size_t calls = 0;
        size_t i_tiles = 0;
        for (uint8_t group = 0; group < group_count; ++group) {
            const size_t k_tiles =
                (static_cast<size_t>(__builtin_popcount(group_support[group])) +
                 kArrayDim - 1) / kArrayDim;
            calls += k_tiles;
            i_tiles += k_tiles * (align_up(group_lanes[group] * row_count, kArrayDim) / kArrayDim);
        }
        // Reduce lane-row tile work without increasing K-tile calls per J tile.
        if (calls <= baseline_calls && i_tiles < best_i_tiles) {
            best_i_tiles = i_tiles;
            best_assignment = assignment;
        }
        return;
    }

    // Every live group costs at least one K tile; extra groups cannot beat the baseline.
    for (uint8_t group = 0; group <= group_count && group < baseline_calls; ++group) {
        assignment[lane_position] = group;
        choose_lane_partition(lane_support, lane_count, row_count,
                              static_cast<uint8_t>(lane_position + 1),
                              static_cast<uint8_t>(group_count + (group == group_count)),
                              assignment, best_assignment, baseline_calls, best_i_tiles);
    }
}

Int4Packing int4_packing_for_bits(uint8_t digit_bits) {
    return digit_bits == 4 ? Int4Packing::adjacent_low_nibble_first : Int4Packing::none;
}

RmdStatus write_packet_digit(StripePacket & packet,
                             const LaneGroupDescriptor & group,
                             size_t lane_row,
                             size_t k,
                             int32_t digit) {
    const BalancedRadixContract contract = balanced_radix_contract(packet.digit_bits);
    if (contract.radix == 0 || lane_row >= group.row_ids.size() ||
        k >= group.padded_k_count ||
        digit < contract.digit_min || digit > contract.digit_max) {
        return RmdStatus::invalid_arguments;
    }

    size_t row_offset = 0;

    if (packet.digit_storage == DigitStorage::packed_signed_int4) {
        const size_t row_bytes = group.padded_k_count / 2;
        size_t packed_index = 0;
        uint8_t shift = 0;
        if (!quants::wreader::native_mvin_q4_position(
                group.padded_k_count, k, packed_index, shift) ||
            !checked_mul(lane_row, row_bytes, row_offset) ||
            !checked_add(group.activation_byte_offset, row_offset, row_offset) ||
            !checked_add(row_offset, packed_index, row_offset) ||
            row_offset >= packet.stacked_activation.packed_int4.size()) {
            return RmdStatus::invalid_packet;
        }
        const uint8_t nibble = static_cast<uint8_t>(digit) & 0x0fu;
        const uint8_t mask = static_cast<uint8_t>(0x0fu << shift);
        uint8_t & packed = packet.stacked_activation.packed_int4[row_offset];
        packed = static_cast<uint8_t>((packed & static_cast<uint8_t>(~mask)) |
                                      static_cast<uint8_t>(nibble << shift));
        return RmdStatus::success;
    }

    if (!checked_mul(lane_row, group.padded_k_count, row_offset) ||
        !checked_add(group.activation_offset, row_offset, row_offset) ||
        !checked_add(row_offset, k, row_offset)) {
        return RmdStatus::overflow;
    }
    if (packet.digit_storage == DigitStorage::signed_int8) {
        if (row_offset >= packet.stacked_activation.signed_int8.size()) {
            return RmdStatus::invalid_packet;
        }
        packet.stacked_activation.signed_int8[row_offset] = static_cast<int8_t>(digit);
        return RmdStatus::success;
    }
    if (packet.digit_storage == DigitStorage::signed_int16) {
        if (row_offset >= packet.stacked_activation.signed_int16.size()) {
            return RmdStatus::invalid_packet;
        }
        packet.stacked_activation.signed_int16[row_offset] = static_cast<int16_t>(digit);
        return RmdStatus::success;
    }
    return RmdStatus::invalid_packet;
}

}

BalancedRadixContract balanced_radix_contract(uint8_t operand_bits) {
    switch (operand_bits) {
        case 4:  return {16, 9, -8, 7};
        case 8:  return {256, 5, -128, 127};
        case 16: return {65536, 3, -32768, 32767};
        default: return {};
    }
}

RmdStatus decompose_balanced_radix(int32_t residual,
                                   uint8_t operand_bits,
                                   NativeBalancedDigits & out) {
    const BalancedRadixContract contract = balanced_radix_contract(operand_bits);
    if (contract.radix == 0) {
        return RmdStatus::invalid_arguments;
    }
    NativeBalancedDigits staged{};
    staged.radix = contract.radix;
    staged.lane_capacity = contract.lane_capacity;
    // Repeat the half-radix bit in every w-bit lane: 0x8, 0x80, or 0x8000.
    // Unsigned addition propagates balanced-radix carries; XOR removes the per-lane bias.
    // Example (w=8): residual 255 becomes digits [-1, 1], since -1 + 256 = 255.
    // Sign-extend before unsigned arithmetic so negative INT32 values stay negative.
    // The 64-bit intermediate preserves a carry past bit 31 as one ordinary extra
    // lane; e.g. INT32_MAX at w=8 becomes [-1, 0, 0, -128, 1].
    const uint64_t high_bits = operand_bits == 4 ? 0x8888888888888888ull :
                               operand_bits == 8 ? 0x8080808080808080ull :
                                                   0x8000800080008000ull;
    uint64_t packed = (static_cast<uint64_t>(static_cast<int64_t>(residual)) +
                       high_bits) ^ high_bits;
    const uint32_t digit_mask = contract.radix - 1;
    const uint32_t sign_bit = contract.radix / 2;
    // Do not stop on a zero digit: higher lanes may still be live.
    for (uint8_t lane = 0; lane < contract.lane_capacity && packed != 0; ++lane) {
        const uint32_t raw = static_cast<uint32_t>(packed & digit_mask);
        // Sign-extend the w-bit digit without shifting a negative signed integer.
        const int32_t digit = static_cast<int32_t>(raw ^ sign_bit) -
                              static_cast<int32_t>(sign_bit);
        staged.digits[lane] = digit;
        if (digit != 0) {
            // Highest used lane plus one, not the number of nonzero digits.
            staged.active_lane_count = static_cast<uint8_t>(lane + 1);
        }
        packed >>= operand_bits;
    }
    out = staged;
    return RmdStatus::success;
}

RmdStatus compose_balanced_radix(const NativeBalancedDigits & digits,
                                 int64_t & out) {
    const uint8_t operand_bits = digits.radix == 16 ? 4 :
        digits.radix == 256 ? 8 : digits.radix == 65536 ? 16 : 0;
    const BalancedRadixContract contract = balanced_radix_contract(operand_bits);
    if (contract.radix != digits.radix ||
        contract.lane_capacity != digits.lane_capacity ||
        digits.active_lane_count > digits.lane_capacity) {
        return RmdStatus::invalid_arguments;
    }

    int64_t staged = 0;
    int64_t place = 1;
    uint8_t expected_active_lane_count = 0;
    for (uint8_t lane = 0; lane < digits.lane_capacity; ++lane) {
        const int32_t digit = digits.digits[lane];
        if (digit < contract.digit_min || digit > contract.digit_max) {
            return RmdStatus::invalid_arguments;
        }
        if (digit != 0) {
            expected_active_lane_count = static_cast<uint8_t>(lane + 1);
        }
        staged += static_cast<int64_t>(digit) * place;
        if (lane + 1 < digits.lane_capacity) {
            place *= contract.radix;
        }
    }
    if (expected_active_lane_count != digits.active_lane_count) {
        return RmdStatus::invalid_arguments;
    }
    for (size_t lane = digits.lane_capacity; lane < digits.digits.size(); ++lane) {
        if (digits.digits[lane] != 0) {
            return RmdStatus::invalid_arguments;
        }
    }
    if (staged < std::numeric_limits<int32_t>::min() ||
        staged > std::numeric_limits<int32_t>::max()) {
        return RmdStatus::residual_too_wide;
    }
    out = staged;
    return RmdStatus::success;
}

const char * rmd_status_message(RmdStatus status) {
    switch (status) {
        case RmdStatus::success:            return "success";
        case RmdStatus::invalid_arguments:  return "rmd: invalid arguments";
        case RmdStatus::invalid_packet:     return "rmd: invalid packet";
        case RmdStatus::residual_too_wide:  return "rmd: reconstructed residual exceeds INT32 range";
        case RmdStatus::unsupported_route:  return "rmd: route is unsupported by the exact result contract";
        case RmdStatus::overflow:           return "rmd: integer overflow";
        case RmdStatus::allocation_failure: return "rmd: allocation failed";
        case RmdStatus::execution_failed:   return "rmd: execution failed";
    }
    return "rmd: unknown status";
}

static RmdStatus read_group_digit(const StripePacket & packet,
                            const BlockDescriptor & block,
                            const LaneGroupDescriptor & group,
                            uint8_t lane_position,
                            size_t row,
                            size_t k,
                            int32_t & digit) {
    const BalancedRadixContract contract = balanced_radix_contract(packet.digit_bits);
    if (lane_position >= group.lane_positions.size() || row >= block.rows_padded ||
        k >= group.padded_k_count) {
        return RmdStatus::invalid_arguments;
    }
    if (block.active_lane_count == 0 || block.active_lane_count > contract.lane_capacity ||
        packet.row_count == 0 || block.rows_padded != align_up(packet.row_count, kArrayDim) ||
        group.lane_positions.size() > contract.lane_capacity ||
        group.lane_positions[lane_position] >= block.active_lane_count ||
        block.lane_ids[group.lane_positions[lane_position]] >= contract.lane_capacity ||
        (block.active_lane_mask & static_cast<uint16_t>(
             1u << block.lane_ids[group.lane_positions[lane_position]])) == 0) {
        return RmdStatus::invalid_packet;
    }

    size_t expected_values = 0;
    size_t expected_bytes = 0;
    size_t expected_value_end = 0;
    size_t expected_byte_end = 0;
    size_t mapped_byte_offset = 0;
    size_t packet_byte_count = 0;
    if (!checked_activation_sizes(packet.digit_bits, group.row_ids.size(),
                                  group.padded_k_count,
                                  expected_values, expected_bytes) ||
        group.activation_byte_count != expected_bytes ||
        !checked_add(group.activation_offset, expected_values, expected_value_end) ||
        !checked_add(group.activation_byte_offset, expected_bytes, expected_byte_end) ||
        expected_value_end > packet.activation_value_count) {
        return RmdStatus::invalid_packet;
    }
    if (packet.digit_storage == DigitStorage::packed_signed_int4) {
        if (group.activation_offset % 2 != 0 ||
            packet.activation_value_count % 2 != 0) {
            return RmdStatus::invalid_packet;
        }
        mapped_byte_offset = group.activation_offset / 2;
        packet_byte_count = packet.activation_value_count / 2;
    } else if (packet.digit_storage == DigitStorage::signed_int8) {
        mapped_byte_offset = group.activation_offset;
        packet_byte_count = packet.activation_value_count;
    } else if (!checked_mul(group.activation_offset, sizeof(int16_t),
                            mapped_byte_offset) ||
               !checked_mul(packet.activation_value_count, sizeof(int16_t),
                            packet_byte_count)) {
        return RmdStatus::invalid_packet;
    }
    if (group.activation_byte_offset != mapped_byte_offset) {
        return RmdStatus::invalid_packet;
    }

    // Public reads use original row coordinates, including removed zero rows.
    // Check metadata and payload extents even when returning an implicit zero.
    if (group.row_offsets.front() != 0 ||
        group.row_offsets[group.lane_positions.size()] != group.row_ids.size()) {
        return RmdStatus::invalid_packet;
    }
    const size_t first = group.row_offsets[lane_position];
    const size_t last = group.row_offsets[lane_position + 1];
    if (first >= last || last > group.row_ids.size()) return RmdStatus::invalid_packet;
    const auto begin = group.row_ids.begin() + first;
    const auto end = group.row_ids.begin() + last;
    const auto found = std::lower_bound(begin, end, row);
    const bool absent_row = row >= packet.row_count || found == end || *found != row;
    const size_t lane_row = absent_row ? first : static_cast<size_t>(found - group.row_ids.begin());
    size_t row_offset = 0;
    int32_t staged = 0;

    if (packet.digit_storage == DigitStorage::packed_signed_int4) {
        const size_t row_bytes = group.padded_k_count / 2;
        int8_t decoded = 0;
        if (!checked_mul(lane_row, row_bytes, row_offset) ||
            !checked_add(group.activation_byte_offset, row_offset, row_offset) ||
            row_offset >= expected_byte_end ||
            expected_byte_end > packet.stacked_activation.packed_int4.size() ||
            packet.stacked_activation.packed_int4.size() != packet_byte_count ||
            !packet.stacked_activation.signed_int8.empty() ||
            !packet.stacked_activation.signed_int16.empty() ||
            !quants::wreader::decode_native_mvin_q4(
                packet.stacked_activation.packed_int4.data() + row_offset,
                row_bytes, group.padded_k_count, k, decoded)) {
            return RmdStatus::invalid_packet;
        }
        staged = decoded;
    } else {
        if (!checked_mul(lane_row, group.padded_k_count, row_offset) ||
            !checked_add(group.activation_offset, row_offset, row_offset) ||
            !checked_add(row_offset, k, row_offset)) {
            return RmdStatus::invalid_packet;
        }
        if (row_offset >= expected_value_end) {
            return RmdStatus::invalid_packet;
        }
        if (packet.digit_storage == DigitStorage::signed_int8) {
            if (expected_value_end > packet.stacked_activation.signed_int8.size() ||
                packet.stacked_activation.signed_int8.size() !=
                    packet.activation_value_count ||
                !packet.stacked_activation.packed_int4.empty() ||
                !packet.stacked_activation.signed_int16.empty()) {
                return RmdStatus::invalid_packet;
            }
            staged = packet.stacked_activation.signed_int8[row_offset];
        } else if (packet.digit_storage == DigitStorage::signed_int16) {
            if (group.activation_byte_offset % alignof(int16_t) != 0 ||
                group.activation_byte_count % sizeof(int16_t) != 0 ||
                expected_byte_end > packet_byte_count ||
                expected_value_end > packet.stacked_activation.signed_int16.size() ||
                packet.stacked_activation.signed_int16.size() !=
                    packet.activation_value_count ||
                !packet.stacked_activation.packed_int4.empty() ||
                !packet.stacked_activation.signed_int8.empty()) {
                return RmdStatus::invalid_packet;
            }
            staged = packet.stacked_activation.signed_int16[row_offset];
        } else {
            return RmdStatus::invalid_packet;
        }
    }

    if (staged < contract.digit_min || staged > contract.digit_max) {
        return RmdStatus::invalid_packet;
    }
    digit = absent_row ? 0 : staged;
    return RmdStatus::success;
}

RmdStatus read_packet_digit(const StripePacket & packet,
                            const BlockDescriptor & block,
                            uint8_t lane_position, size_t row, size_t k,
                            int32_t & digit) {
    const BalancedRadixContract contract = balanced_radix_contract(packet.digit_bits);
    if (packet.version != kPacketVersion || contract.radix == 0 ||
        packet.lane_capacity != contract.lane_capacity ||
        packet.digit_storage != digit_storage_for_bits(packet.digit_bits) ||
        packet.int4_packing != int4_packing_for_bits(packet.digit_bits) ||
        packet.block_size != kBlockSize || packet.array_dim != kArrayDim) {
        return RmdStatus::invalid_packet;
    }
    if (lane_position >= block.active_lane_count || row >= block.rows_padded ||
        k >= block.padded_k_count) {
        return RmdStatus::invalid_arguments;
    }
    const LaneGroupDescriptor * selected = nullptr;
    size_t group_lane = 0;
    for (const LaneGroupDescriptor & group : block.groups) {
        for (size_t lane = 0; lane < group.lane_positions.size(); ++lane) {
            if (group.lane_positions[lane] == lane_position) {
                if (selected != nullptr) return RmdStatus::invalid_packet;
                selected = &group;
                group_lane = lane;
            }
        }
    }
    if (selected == nullptr) return RmdStatus::invalid_packet;
    size_t group_k = 0;
    bool absent = k >= block.compact_k_count;
    if (!absent) {
        const size_t index = static_cast<size_t>(block.k_index_offset) + k;
        if (index >= packet.k_indices.size() || packet.k_indices[index] >= kBlockSize) {
            return RmdStatus::invalid_packet;
        }
        // The public K index is block-compact; group masks use original local K.
        const uint32_t bit = uint32_t{1} << packet.k_indices[index];
        absent = (selected->k_mask & bit) == 0;
        // Selected bits below this K give its rank in the group's packed row.
        group_k = static_cast<size_t>(__builtin_popcount(selected->k_mask & (bit - 1)));
    }
    int32_t staged = 0;
    const RmdStatus status = read_group_digit(packet, block, *selected,
        static_cast<uint8_t>(group_lane), row, absent ? 0 : group_k, staged);
    if (status != RmdStatus::success) return status;
    digit = absent ? 0 : staged;
    return RmdStatus::success;
}

void RmdStripeBuilder::reset(size_t stripe_id, size_t row_begin, size_t row_count,
                             size_t logical_k, size_t logical_j) {
    reset(stripe_id, row_begin, row_count, logical_k, logical_j,
          GGML_GEMMINI_ACTIVATION_BITS);
}

void RmdStripeBuilder::reset(size_t stripe_id, size_t row_begin, size_t row_count,
                             size_t logical_k, size_t logical_j,
                             uint8_t digit_bits) {
    status_ = RmdStatus::success;
    stripe_id_ = stripe_id;
    row_begin_ = row_begin;
    row_count_ = row_count;
    logical_k_ = logical_k;
    logical_j_ = logical_j;
    digit_bits_ = digit_bits;
    residual_event_count_ = 0;
    residual_min_ = residual_max_ = 0;
    required_planes_ = 0;
    entries_.clear();
    blocks_.clear();
    if (row_count == 0 || logical_k == 0 || logical_j == 0 ||
        logical_k > std::numeric_limits<uint32_t>::max() ||
        row_count > std::numeric_limits<uint16_t>::max() ||
        balanced_radix_contract(digit_bits).radix == 0) {
        status_ = RmdStatus::invalid_arguments;
    } else if (row_begin > std::numeric_limits<size_t>::max() - row_count) {
        status_ = RmdStatus::overflow;
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

    NativeBalancedDigits digits{};
    const RmdStatus decomposition =
        decompose_balanced_radix(residual, digit_bits_, digits);
    if (decomposition != RmdStatus::success) {
        status_ = decomposition;
        return false;
    }

    ++residual_event_count_;
    residual_min_ = residual_event_count_ == 1 ? residual : std::min(residual_min_, residual);
    residual_max_ = residual_event_count_ == 1 ? residual : std::max(residual_max_, residual);
    required_planes_ = std::max(required_planes_, digits.active_lane_count);
    const size_t block_id = original_k / kBlockSize;
    const size_t block_local_k = original_k % kBlockSize;
    if (block_id > std::numeric_limits<uint32_t>::max()) {
        status_ = RmdStatus::invalid_arguments;
        return false;
    }

    try {
        BlockAccum & accum = blocks_[static_cast<uint32_t>(block_id)];
        if (accum.row_lane_masks.empty()) accum.row_lane_masks.resize(row_count_, 0);
        // K bits retain original block-local coordinates; OR unions this stripe's rows.
        accum.k_mask |= uint32_t{1} << block_local_k;
        uint16_t active_lanes = 0;
        for (uint8_t lane = 0; lane < digits.lane_capacity; ++lane) {
            if (digits.digits[lane] == 0) {
                continue;
            }
            // Keep original lane IDs so pruning cannot change the radix exponent.
            active_lanes |= static_cast<uint16_t>(1u << lane);
            accum.lane_k_masks[lane] |= uint32_t{1} << block_local_k;
            entries_.push_back({
                static_cast<uint32_t>(block_id),
                static_cast<uint32_t>(local_row),
                static_cast<uint16_t>(block_local_k),
                lane,
                digits.digits[lane],
            });
        }
        accum.lane_mask |= active_lanes;
        accum.row_lane_masks[local_row] |= active_lanes;
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
        const BalancedRadixContract contract = balanced_radix_contract(digit_bits_);
        if (contract.radix == 0) {
            status_ = RmdStatus::invalid_arguments;
            return nullptr;
        }

        auto packet = std::make_shared<StripePacket>();
        packet->version = kPacketVersion;
        packet->digit_bits = digit_bits_;
        packet->lane_capacity = contract.lane_capacity;
        packet->digit_storage = digit_storage_for_bits(digit_bits_);
        packet->int4_packing = int4_packing_for_bits(digit_bits_);
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
        packet->residual_event_count = residual_event_count_;
        packet->residual_min = residual_min_;
        packet->residual_max = residual_max_;
        packet->required_planes = required_planes_;
        packet->digit_nnz = entries_.size();
        packet->residual_observations_valid = true;

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
        size_t activation_value_cursor = 0;
        size_t activation_byte_cursor = 0;
        size_t output_cursor = 0;
        struct BlockPacking {
            size_t block_index = 0;
            std::array<uint8_t, kMaxNativeRadixLanes> group_ids{};
            std::vector<uint32_t> row_indices; // transient original (lane,row) -> group row
        };
        std::map<uint32_t, BlockPacking> packing_by_block;

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
            const size_t compact_k_count = static_cast<size_t>(__builtin_popcount(accum.k_mask));
            descriptor.compact_k_count = static_cast<uint16_t>(compact_k_count);
            const size_t padded_k = align_up(compact_k_count, kArrayDim);
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
            descriptor.lane_k_masks = accum.lane_k_masks;
            descriptor.rows_padded = static_cast<uint16_t>(rows_padded);
            descriptor.lane_stride_values = static_cast<uint32_t>(lane_stride_values);

            uint8_t lane_count = 0;
            for (uint8_t lane = 0; lane < contract.lane_capacity; ++lane) {
                if ((accum.lane_mask & static_cast<uint16_t>(1u << lane)) != 0) {
                    descriptor.lane_ids[lane_count++] = lane;
                }
            }
            descriptor.active_lane_count = lane_count;

            // Fix the final groups before allocating payload; each entry is packed once.
            std::array<uint32_t, kMaxNativeRadixLanes> lane_support{};
            for (uint8_t lane = 0; lane < lane_count; ++lane) {
                lane_support[lane] = descriptor.lane_k_masks[descriptor.lane_ids[lane]];
            }
            std::array<uint8_t, kMaxNativeRadixLanes> assignment{};
            std::array<uint8_t, kMaxNativeRadixLanes> best_assignment{};
            const size_t baseline_calls = padded_k / kArrayDim;
            size_t best_i_tiles = baseline_calls * (align_up(lane_count * row_count_, kArrayDim) / kArrayDim);
            if (baseline_calls > 1) {
                choose_lane_partition(lane_support, lane_count, row_count_,
                    1, 1, assignment, best_assignment, baseline_calls, best_i_tiles);
            }
            const size_t group_count = *std::max_element(best_assignment.begin(),
                best_assignment.begin() + lane_count) + 1;
            descriptor.groups.resize(group_count);
            BlockPacking packing{};
            packing.block_index = packet->blocks.size();
            packing.row_indices.resize(contract.lane_capacity * row_count_);
            for (uint8_t lane = 0; lane < lane_count; ++lane) {
                LaneGroupDescriptor & group = descriptor.groups[best_assignment[lane]];
                packing.group_ids[descriptor.lane_ids[lane]] = best_assignment[lane];
                group.lane_positions.push_back(lane);
                group.k_mask |= lane_support[lane];
            }
            size_t block_activation_values = 0;
            size_t block_activation_bytes = 0;
            for (LaneGroupDescriptor & group : descriptor.groups) {
                uint16_t group_mask = 0;
                for (uint8_t lane : group.lane_positions) group_mask |= uint16_t{1} << descriptor.lane_ids[lane];
                size_t packed_rows = 0;
                for (uint16_t mask : accum.row_lane_masks) packed_rows += __builtin_popcount(mask & group_mask);
                group.row_ids.reserve(packed_rows);
                for (size_t group_lane = 0; group_lane < group.lane_positions.size(); ++group_lane) {
                    const uint8_t lane_id = descriptor.lane_ids[group.lane_positions[group_lane]];
                    for (size_t row = 0; row < row_count_; ++row) {
                        if ((accum.row_lane_masks[row] & (uint16_t{1} << lane_id)) == 0) continue;
                        packing.row_indices[lane_id * row_count_ + row] =
                            static_cast<uint32_t>(group.row_ids.size());
                        group.row_ids.push_back(static_cast<uint16_t>(row));
                    }
                    group.row_offsets[group_lane + 1] = static_cast<uint32_t>(group.row_ids.size());
                }
                group.padded_k_count = static_cast<uint16_t>(align_up(
                    static_cast<size_t>(__builtin_popcount(group.k_mask)), kArrayDim));
                size_t values = 0, bytes = 0;
                size_t value_offset = 0, byte_offset = 0;
                if (!checked_activation_sizes(digit_bits_, group.row_ids.size(),
                        group.padded_k_count, values, bytes) ||
                    !checked_add(activation_value_cursor, block_activation_values, value_offset) ||
                    !checked_add(activation_byte_cursor, block_activation_bytes, byte_offset) ||
                    value_offset > std::numeric_limits<uint32_t>::max() ||
                    byte_offset > std::numeric_limits<uint32_t>::max() ||
                    bytes > std::numeric_limits<uint32_t>::max() ||
                    !checked_add(block_activation_values, values, block_activation_values) ||
                    !checked_add(block_activation_bytes, bytes, block_activation_bytes)) {
                    status_ = RmdStatus::overflow;
                    return nullptr;
                }
                group.activation_offset = static_cast<uint32_t>(value_offset);
                group.activation_byte_offset = static_cast<uint32_t>(byte_offset);
                group.activation_byte_count = static_cast<uint32_t>(bytes);
            }
            size_t block_output = 0;
            if (lane_count == 0 ||
                !checked_mul(lane_stride_values, lane_count, block_output) ||
                k_cursor > std::numeric_limits<uint32_t>::max() ||
                activation_value_cursor > std::numeric_limits<uint32_t>::max() ||
                activation_byte_cursor > std::numeric_limits<uint32_t>::max() ||
                block_activation_bytes > std::numeric_limits<uint32_t>::max() ||
                output_cursor > std::numeric_limits<uint32_t>::max()) {
                status_ = RmdStatus::overflow;
                return nullptr;
            }
            descriptor.k_index_offset = static_cast<uint32_t>(k_cursor);
            descriptor.activation_offset = static_cast<uint32_t>(activation_value_cursor);
            descriptor.activation_byte_offset = static_cast<uint32_t>(activation_byte_cursor);
            descriptor.activation_byte_count = static_cast<uint32_t>(block_activation_bytes);
            descriptor.output_value_offset = static_cast<uint32_t>(output_cursor);

            // Compact K index table for this block: ascending, deduplicated, block local.
            // ctz finds the lowest set bit; x & (x - 1) removes it. Never call ctz(0).
            for (uint32_t remaining = accum.k_mask; remaining != 0; remaining &= remaining - 1) {
                const uint16_t local_k = static_cast<uint16_t>(__builtin_ctz(remaining));
                packet->k_indices.push_back(local_k);
            }

            if (!checked_add(k_cursor, compact_k_count, k_cursor) ||
                !checked_add(activation_value_cursor, block_activation_values,
                             activation_value_cursor) ||
                !checked_add(activation_byte_cursor, block_activation_bytes,
                             activation_byte_cursor) ||
                !checked_add(output_cursor, block_output, output_cursor) ||
                k_cursor > std::numeric_limits<uint32_t>::max() ||
                activation_value_cursor > std::numeric_limits<uint32_t>::max() ||
                activation_byte_cursor > std::numeric_limits<uint32_t>::max() ||
                output_cursor > std::numeric_limits<uint32_t>::max()) {
                status_ = RmdStatus::overflow;
                return nullptr;
            }

            packing_by_block.emplace(block_id, std::move(packing));
            packet->blocks.push_back(std::move(descriptor));
        }

        packet->activation_value_count = activation_value_cursor;
        packet->total_output_values = output_cursor;
        // Signed two's-complement Q4 and scalar padding both encode numeric zero.
        if (packet->digit_storage == DigitStorage::packed_signed_int4) {
            packet->stacked_activation.packed_int4.assign(activation_byte_cursor, 0x00u);
        } else if (packet->digit_storage == DigitStorage::signed_int8) {
            packet->stacked_activation.signed_int8.assign(activation_value_cursor, 0);
        } else {
            packet->stacked_activation.signed_int16.assign(activation_value_cursor, 0);
        }

#if LOG_CYCLE
        std::vector<uint64_t> active_rows((row_count_ + 63) / 64, 0);
#endif
        for (const DigitEntry & entry : entries_) {
#if LOG_CYCLE
            active_rows[entry.local_row / 64] |= uint64_t{1} << (entry.local_row % 64);
#endif
            const BlockPacking & packing = packing_by_block[entry.block_id];
            const BlockDescriptor & descriptor = packet->blocks[packing.block_index];
            const LaneGroupDescriptor & group = descriptor.groups[packing.group_ids[entry.lane]];
            const uint32_t bit = uint32_t{1} << entry.block_local_k;
            // bit - 1 selects lower K positions; their popcount is the packed index.
            const size_t group_k = static_cast<size_t>(
                __builtin_popcount(group.k_mask & (bit - 1)));
            const RmdStatus write = write_packet_digit(*packet, group,
                packing.row_indices[entry.lane * row_count_ + entry.local_row], group_k, entry.digit);
            if (write != RmdStatus::success) {
                status_ = write;
                return nullptr;
            }
        }

#if LOG_CYCLE
        for (uint64_t rows : active_rows) {
            packet->active_original_rows += static_cast<size_t>(__builtin_popcountll(rows));
        }
        packet->active_original_rows_valid = true;
#endif

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
    if (row_begin >= row_end ||
        row_end - row_begin > std::numeric_limits<uint16_t>::max()) {
        status = RmdStatus::invalid_arguments;
        return nullptr;
    }

    StripePacketHandle exact_match;
    size_t overlapping_packets = 0;
    size_t logical_k = 0;
    size_t logical_j = 0;
    uint8_t digit_bits = 0;
    bool have_metadata = false;
    for (const StripePacketHandle & handle : packets) {
        if (!handle) {
            continue;
        }
        const StripePacket & packet = *handle;
        if (packet.row_count > std::numeric_limits<size_t>::max() - packet.row_begin) {
            status = RmdStatus::invalid_packet;
            return nullptr;
        }
        const size_t packet_row_end = packet.row_begin + packet.row_count;
        if (packet.row_begin >= row_end || packet_row_end <= row_begin) {
            continue;
        }
        status = validate_packet(packet);
        if (status != RmdStatus::success) {
            return nullptr;
        }
        if (!have_metadata) {
            logical_k = packet.logical_k;
            logical_j = packet.logical_j;
            digit_bits = packet.digit_bits;
            have_metadata = true;
        } else if (packet.logical_k != logical_k || packet.logical_j != logical_j ||
                   packet.digit_bits != digit_bits) {
            status = RmdStatus::invalid_packet;
            return nullptr;
        }
        ++overlapping_packets;
        if (packet.row_begin == row_begin && packet_row_end == row_end &&
            packet.stripe_id == stripe_id) {
            exact_match = handle;
        }
    }
    if (overlapping_packets == 1 && exact_match) {
        return exact_match;
    }

    // (local row, original K) -> residual, rebuilt from the native balanced digits.
    std::map<std::pair<uint32_t, uint32_t>, int64_t> residuals;
    for (const StripePacketHandle & handle : packets) {
        if (!handle) {
            continue;
        }
        const StripePacket & packet = *handle;
        const size_t packet_row_end = packet.row_begin + packet.row_count;
        if (packet.row_begin >= row_end || packet_row_end <= row_begin) {
            continue;
        }
        const BalancedRadixContract contract = balanced_radix_contract(packet.digit_bits);
        for (const BlockDescriptor & block : packet.blocks) {
            for (uint8_t position = 0; position < block.active_lane_count; ++position) {
                int64_t place = 1;
                for (uint8_t step = 0; step < block.lane_ids[position]; ++step) {
                    if (place > std::numeric_limits<int64_t>::max() / contract.radix) {
                        status = RmdStatus::overflow;
                        return nullptr;
                    }
                    place *= contract.radix;
                }
                for (size_t row = 0; row < packet.row_count; ++row) {
                    const size_t global_row = packet.row_begin + row;
                    if (global_row < row_begin || global_row >= row_end) {
                        continue;
                    }
                    for (size_t k = 0; k < block.compact_k_count; ++k) {
                        int32_t digit = 0;
                        status = read_packet_digit(packet, block, position, row, k, digit);
                        if (status != RmdStatus::success) {
                            return nullptr;
                        }
                        if (digit == 0) {
                            continue;
                        }
                        const uint32_t column = block.global_k_begin +
                            packet.k_indices[block.k_index_offset + k];
                        const auto key = std::make_pair(
                            static_cast<uint32_t>(global_row - row_begin), column);
                        const int64_t contribution = static_cast<int64_t>(digit) * place;
                        int64_t & residual = residuals[key];
                        if ((contribution > 0 && residual >
                             std::numeric_limits<int64_t>::max() - contribution) ||
                            (contribution < 0 && residual <
                             std::numeric_limits<int64_t>::min() - contribution)) {
                            status = RmdStatus::overflow;
                            return nullptr;
                        }
                        residual += contribution;
                    }
                }
            }
        }
    }

    if (residuals.empty()) {
        return nullptr;
    }

    RmdStripeBuilder builder;
    builder.reset(stripe_id, row_begin, row_end - row_begin,
                  logical_k, logical_j, digit_bits);
    if (builder.status() != RmdStatus::success) {
        status = builder.status();
        return nullptr;
    }
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
    const BalancedRadixContract contract = balanced_radix_contract(packet.digit_bits);
    if (packet.version != kPacketVersion || contract.radix == 0 ||
        packet.lane_capacity != contract.lane_capacity ||
        packet.digit_storage != digit_storage_for_bits(packet.digit_bits) ||
        packet.int4_packing != int4_packing_for_bits(packet.digit_bits) ||
        packet.block_size != kBlockSize || packet.array_dim != kArrayDim) {
        return RmdStatus::invalid_packet;
    }
    if (packet.row_count == 0 || packet.logical_j == 0 || packet.logical_k == 0 ||
        packet.logical_k > std::numeric_limits<uint32_t>::max() ||
        packet.blocks.empty() ||
        packet.row_begin > std::numeric_limits<size_t>::max() - packet.row_count) {
        return RmdStatus::invalid_packet;
    }

    const size_t expected_j_padded = align_up(packet.logical_j, kArrayDim);
    const size_t rows_padded = align_up(packet.row_count, kArrayDim);
    size_t expected_lane_stride = 0;
    if (expected_j_padded == 0 || rows_padded == 0 ||
        packet.j_padded != expected_j_padded ||
        rows_padded > std::numeric_limits<uint16_t>::max() ||
        !checked_mul(rows_padded, expected_j_padded, expected_lane_stride) ||
        expected_lane_stride > std::numeric_limits<uint32_t>::max()) {
        return RmdStatus::invalid_packet;
    }

    size_t expected_k_cursor = 0;
    size_t expected_activation_values = 0;
    size_t expected_activation_bytes = 0;
    size_t expected_output = 0;
    uint32_t previous_block_id = 0;
    bool has_previous = false;

    for (const BlockDescriptor & block : packet.blocks) {
        if (has_previous && block.block_id <= previous_block_id) {
            return RmdStatus::invalid_packet;
        }
        previous_block_id = block.block_id;
        has_previous = true;

        size_t expected_global_k_begin = 0;
        if (!checked_mul(block.block_id, kBlockSize, expected_global_k_begin) ||
            expected_global_k_begin != block.global_k_begin ||
            expected_global_k_begin >= packet.logical_k ||
            block.compact_k_count == 0 || block.compact_k_count > kBlockSize) {
            return RmdStatus::invalid_packet;
        }
        const size_t expected_padded_k = align_up(block.compact_k_count, kArrayDim);
        if (expected_padded_k == 0 || block.padded_k_count != expected_padded_k ||
            block.padded_k_count % kArrayDim != 0 ||
            block.rows_padded != rows_padded || block.rows_padded % kArrayDim != 0 ||
            block.lane_stride_values != expected_lane_stride ||
            block.active_lane_mask == 0 || block.active_lane_count == 0 ||
            block.active_lane_count > contract.lane_capacity) {
            return RmdStatus::invalid_packet;
        }

        uint16_t rebuilt_mask = 0;
        for (uint8_t position = 0; position < block.active_lane_count; ++position) {
            const uint8_t lane_id = block.lane_ids[position];
            if (lane_id >= contract.lane_capacity ||
                (position != 0 && lane_id <= block.lane_ids[position - 1])) {
                return RmdStatus::invalid_packet;
            }
            const uint16_t bit = static_cast<uint16_t>(1u << lane_id);
            rebuilt_mask |= bit;
        }
        if (rebuilt_mask != block.active_lane_mask) {
            return RmdStatus::invalid_packet;
        }
        for (size_t position = block.active_lane_count;
             position < block.lane_ids.size(); ++position) {
            if (block.lane_ids[position] != 0) {
                return RmdStatus::invalid_packet;
            }
        }

        size_t block_activation_values = 0;
        size_t block_activation_bytes = 0;
        size_t block_output = 0;
        uint16_t grouped_lanes = 0;
        if (block.groups.empty() || block.groups.size() > block.active_lane_count) {
            return RmdStatus::invalid_packet;
        }
        for (const LaneGroupDescriptor & group : block.groups) {
            if (group.lane_positions.empty() || group.lane_positions.size() > block.active_lane_count ||
                group.row_offsets.front() != 0 ||
                group.row_offsets[group.lane_positions.size()] != group.row_ids.size() || group.k_mask == 0 ||
                group.padded_k_count != align_up(
                    static_cast<size_t>(__builtin_popcount(group.k_mask)), kArrayDim)) {
                return RmdStatus::invalid_packet;
            }
            uint32_t support = 0;
            for (size_t lane = 0; lane < group.lane_positions.size(); ++lane) {
                const uint8_t position = group.lane_positions[lane];
                if (position >= block.active_lane_count ||
                    (lane != 0 && position <= group.lane_positions[lane - 1]) ||
                    (grouped_lanes & (uint16_t{1} << position)) != 0) {
                    return RmdStatus::invalid_packet;
                }
                grouped_lanes |= uint16_t{1} << position;
                support |= block.lane_k_masks[block.lane_ids[position]];
                const size_t first = group.row_offsets[lane];
                const size_t last = group.row_offsets[lane + 1];
                if (first >= last || last > group.row_ids.size()) return RmdStatus::invalid_packet;
                for (size_t row = first; row < last; ++row) {
                    if (group.row_ids[row] >= packet.row_count ||
                        (row != first && group.row_ids[row] <= group.row_ids[row - 1])) {
                        return RmdStatus::invalid_packet;
                    }
                }
            }
            size_t values = 0, bytes = 0;
            if (support != group.k_mask ||
                !checked_activation_sizes(packet.digit_bits, group.row_ids.size(),
                    group.padded_k_count, values, bytes) ||
                group.activation_offset != expected_activation_values + block_activation_values ||
                group.activation_byte_offset != expected_activation_bytes + block_activation_bytes ||
                group.activation_byte_count != bytes ||
                !checked_add(block_activation_values, values, block_activation_values) ||
                !checked_add(block_activation_bytes, bytes, block_activation_bytes)) {
                return RmdStatus::invalid_packet;
            }
        }
        if (grouped_lanes != (uint16_t{1} << block.active_lane_count) - 1 ||
            !checked_mul(block.active_lane_count, block.lane_stride_values,
                         block_output) ||
            block.activation_byte_count != block_activation_bytes ||
            block.k_index_offset != expected_k_cursor ||
            block.activation_offset != expected_activation_values ||
            block.activation_byte_offset != expected_activation_bytes ||
            block.output_value_offset != expected_output ||
            (packet.digit_storage == DigitStorage::signed_int16 &&
             (block.activation_byte_offset % alignof(int16_t) != 0 ||
              block.activation_byte_count % sizeof(int16_t) != 0))) {
            return RmdStatus::invalid_packet;
        }

        // Selected K indices are ascending, unique, block-local, and in logical K.
        for (size_t i = 0; i < block.compact_k_count; ++i) {
            size_t index = 0;
            if (!checked_add(block.k_index_offset, i, index) ||
                index >= packet.k_indices.size()) {
                return RmdStatus::invalid_packet;
            }
            const uint16_t local_k = packet.k_indices[index];
            size_t global_k = 0;
            if (local_k >= kBlockSize ||
                (i != 0 && local_k <= packet.k_indices[index - 1]) ||
                !checked_add(block.global_k_begin, local_k, global_k) ||
                global_k >= packet.logical_k) {
                return RmdStatus::invalid_packet;
            }
        }

        if (!checked_add(expected_k_cursor, block.compact_k_count,
                         expected_k_cursor) ||
            !checked_add(expected_activation_values, block_activation_values,
                         expected_activation_values) ||
            !checked_add(expected_activation_bytes, block_activation_bytes,
                         expected_activation_bytes) ||
            !checked_add(expected_output, block_output, expected_output) ||
            expected_k_cursor > std::numeric_limits<uint32_t>::max() ||
            expected_activation_values > std::numeric_limits<uint32_t>::max() ||
            expected_activation_bytes > std::numeric_limits<uint32_t>::max() ||
            expected_output > std::numeric_limits<uint32_t>::max()) {
            return RmdStatus::invalid_packet;
        }
    }

    const bool q4_payload =
        packet.stacked_activation.packed_int4.size() == expected_activation_bytes &&
        packet.stacked_activation.signed_int8.empty() &&
        packet.stacked_activation.signed_int16.empty();
    const bool q8_payload =
        packet.stacked_activation.packed_int4.empty() &&
        packet.stacked_activation.signed_int8.size() == expected_activation_values &&
        packet.stacked_activation.signed_int16.empty() &&
        expected_activation_bytes == expected_activation_values;
    size_t expected_int16_bytes = 0;
    const bool int16_size_ok =
        checked_mul(expected_activation_values, sizeof(int16_t), expected_int16_bytes);
    const bool q16_payload =
        packet.stacked_activation.packed_int4.empty() &&
        packet.stacked_activation.signed_int8.empty() &&
        packet.stacked_activation.signed_int16.size() == expected_activation_values &&
        int16_size_ok && expected_activation_bytes == expected_int16_bytes;
    if (packet.k_indices.size() != expected_k_cursor ||
        packet.activation_value_count != expected_activation_values ||
        packet.total_output_values != expected_output ||
        (packet.digit_storage == DigitStorage::packed_signed_int4 && !q4_payload) ||
        (packet.digit_storage == DigitStorage::signed_int8 && !q8_payload) ||
        (packet.digit_storage == DigitStorage::signed_int16 && !q16_payload)) {
        return RmdStatus::invalid_packet;
    }

    // Every stored lane is active, while padded rows and K slots decode to zero.
    size_t rebuilt_residual_event_count = 0;
    for (const BlockDescriptor & block : packet.blocks) {
        std::array<uint32_t, kMaxNativeRadixLanes> rebuilt_lane_k_masks{};
        std::array<std::array<uint32_t, kBlockSize>, kMaxNativeRadixLanes> group_k_bits{};
        std::array<std::array<uint32_t, kMaxNativeRadixLanes>, kMaxNativeRadixLanes> row_cursors{};
        for (size_t group_index = 0; group_index < block.groups.size(); ++group_index) {
            std::copy_n(block.groups[group_index].row_offsets.begin(),
                        block.groups[group_index].lane_positions.size(), row_cursors[group_index].begin());
            size_t k_count = 0;
            for (uint32_t remaining = block.groups[group_index].k_mask;
                 remaining != 0; remaining &= remaining - 1) {
                // Unsigned negation isolates the lowest original-K bit, including K=31.
                group_k_bits[group_index][k_count++] = remaining & (0u - remaining);
            }
        }
        for (size_t row = 0; row < packet.row_count; ++row) {
            uint32_t row_k_mask = 0;
            for (size_t group_index = 0; group_index < block.groups.size(); ++group_index) {
                const LaneGroupDescriptor & group = block.groups[group_index];
                const size_t k_count = static_cast<size_t>(__builtin_popcount(group.k_mask));
                for (size_t lane = 0; lane < group.lane_positions.size(); ++lane) {
                    const uint8_t lane_id = block.lane_ids[group.lane_positions[lane]];
                    auto & cursor = row_cursors[group_index][lane];
                    if (cursor == group.row_offsets[lane + 1] || group.row_ids[cursor] != row) continue;
                    const size_t lane_row = cursor++;
                    bool row_nonzero = false;
                    const auto record_digit = [&](size_t k, bool nonzero) {
                        if (!nonzero) return true;
                        row_nonzero = true;
                        if (k >= k_count) return false;
                        const uint32_t bit = group_k_bits[group_index][k];
                        rebuilt_lane_k_masks[lane_id] |= bit;
                        row_k_mask |= bit;
                        return true;
                    };
                    // Extents are validated above; every native bit pattern is a valid digit.
                    if (packet.digit_storage == DigitStorage::packed_signed_int4) {
                        const uint8_t * values = packet.stacked_activation.packed_int4.data() +
                            group.activation_byte_offset + lane_row * (group.padded_k_count / 2);
                        for (size_t k = 0; k < group.padded_k_count; ++k) {
                            if (!record_digit(k, ((values[k / 2] >> (4 * (k % 2))) & 0x0f) != 0)) {
                                return RmdStatus::invalid_packet;
                            }
                        }
                    } else if (packet.digit_storage == DigitStorage::signed_int8) {
                        const int8_t * values = packet.stacked_activation.signed_int8.data() +
                            group.activation_offset + lane_row * group.padded_k_count;
                        for (size_t k = 0; k < group.padded_k_count; ++k) {
                            if (!record_digit(k, values[k] != 0)) return RmdStatus::invalid_packet;
                        }
                    } else {
                        const int16_t * values = packet.stacked_activation.signed_int16.data() +
                            group.activation_offset + lane_row * group.padded_k_count;
                        for (size_t k = 0; k < group.padded_k_count; ++k) {
                            if (!record_digit(k, values[k] != 0)) return RmdStatus::invalid_packet;
                        }
                    }
                    if (!row_nonzero) return RmdStatus::invalid_packet;
                }
            }
            rebuilt_residual_event_count += static_cast<size_t>(__builtin_popcount(row_k_mask));
        }
        // Real lane rows are adjacent; only the tail of the complete group is padding.
        for (const LaneGroupDescriptor & group : block.groups) {
            const size_t real_values = group.row_ids.size() * group.padded_k_count;
            const auto nonzero = [](auto value) { return value != 0; };
            if (packet.digit_storage == DigitStorage::packed_signed_int4) {
                const auto begin = packet.stacked_activation.packed_int4.begin() +
                    group.activation_byte_offset;
                if (std::any_of(begin + real_values / 2,
                                begin + group.activation_byte_count, nonzero)) {
                    return RmdStatus::invalid_packet;
                }
            } else if (packet.digit_storage == DigitStorage::signed_int8) {
                const auto begin = packet.stacked_activation.signed_int8.begin() +
                    group.activation_offset;
                if (std::any_of(begin + real_values,
                                begin + group.activation_byte_count, nonzero)) {
                    return RmdStatus::invalid_packet;
                }
            } else {
                const auto begin = packet.stacked_activation.signed_int16.begin() +
                    group.activation_offset;
                if (std::any_of(begin + real_values,
                                begin + group.activation_byte_count / sizeof(int16_t), nonzero)) {
                    return RmdStatus::invalid_packet;
                }
            }
        }
        for (uint8_t position = 0; position < block.active_lane_count; ++position) {
            if (rebuilt_lane_k_masks[block.lane_ids[position]] == 0) return RmdStatus::invalid_packet;
        }
        uint32_t selected_k_mask = 0, rebuilt_k_mask = 0;
        for (size_t k = 0; k < block.compact_k_count; ++k) {
            selected_k_mask |= uint32_t{1} << packet.k_indices[block.k_index_offset + k];
        }
        for (uint32_t support : rebuilt_lane_k_masks) rebuilt_k_mask |= support;
        if (selected_k_mask != rebuilt_k_mask) return RmdStatus::invalid_packet;
        if (rebuilt_lane_k_masks != block.lane_k_masks) {
            return RmdStatus::invalid_packet;
        }
    }
    if (packet.residual_event_count == 0 ||
        packet.residual_event_count != rebuilt_residual_event_count) {
        return RmdStatus::invalid_packet;
    }

    return RmdStatus::success;
}

}
