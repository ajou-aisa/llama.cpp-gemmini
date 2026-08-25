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
                              size_t lane_count,
                              size_t rows_padded,
                              size_t padded_k_count,
                              size_t & value_count,
                              size_t & byte_count) {
    size_t row_values = 0;
    if (!checked_mul(rows_padded, padded_k_count, row_values) ||
        !checked_mul(row_values, lane_count, value_count)) {
        return false;
    }
    if (digit_bits == 4) {
        if (padded_k_count % 2 != 0) {
            return false;
        }
        size_t row_bytes = padded_k_count / 2;
        return checked_mul(rows_padded, row_bytes, row_bytes) &&
            checked_mul(row_bytes, lane_count, byte_count);
    }
    if (digit_bits == 8) {
        byte_count = value_count;
        return true;
    }
    return digit_bits == 16 && checked_mul(value_count, sizeof(int16_t), byte_count);
}

Int4Packing int4_packing_for_bits(uint8_t digit_bits) {
    return digit_bits == 4 ? Int4Packing::adjacent_low_nibble_first : Int4Packing::none;
}

RmdStatus write_packet_digit(StripePacket & packet,
                             const BlockDescriptor & block,
                             uint8_t lane_position,
                             size_t row,
                             size_t k,
                             int32_t digit) {
    const BalancedRadixContract contract = balanced_radix_contract(packet.digit_bits);
    if (contract.radix == 0 || lane_position >= block.active_lane_count ||
        row >= block.rows_padded || k >= block.padded_k_count ||
        digit < contract.digit_min || digit > contract.digit_max) {
        return RmdStatus::invalid_arguments;
    }

    size_t lane_row = 0;
    size_t row_offset = 0;
    if (!checked_mul(lane_position, block.rows_padded, lane_row) ||
        !checked_add(lane_row, row, lane_row)) {
        return RmdStatus::overflow;
    }

    if (packet.digit_storage == DigitStorage::packed_signed_int4) {
        const size_t row_bytes = block.padded_k_count / 2;
        size_t packed_index = 0;
        uint8_t shift = 0;
        if (!quants::wreader::native_mvin_q4_position(
                block.padded_k_count, k, packed_index, shift) ||
            !checked_mul(lane_row, row_bytes, row_offset) ||
            !checked_add(block.activation_byte_offset, row_offset, row_offset) ||
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

    if (!checked_mul(lane_row, block.padded_k_count, row_offset) ||
        !checked_add(block.activation_offset, row_offset, row_offset) ||
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
        case 4:  return {16, 8, -8, 7};
        case 8:  return {256, 4, -128, 127};
        case 16: return {65536, 2, -32768, 32767};
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
    if (residual < kSigned21Min || residual > kSigned21Max) {
        return RmdStatus::residual_too_wide;
    }

    NativeBalancedDigits staged{};
    staged.radix = contract.radix;
    staged.lane_capacity = contract.lane_capacity;
    int64_t rest = residual;
    for (uint8_t lane = 0; lane < contract.lane_capacity && rest != 0; ++lane) {
        int32_t digit = static_cast<int32_t>(
            static_cast<uint64_t>(rest) & (contract.radix - 1));
        if (digit > contract.digit_max) {
            digit -= static_cast<int32_t>(contract.radix);
        }
        staged.digits[lane] = digit;
        if (digit != 0) {
            staged.active_lane_count = static_cast<uint8_t>(lane + 1);
        }
        rest = (rest - digit) / contract.radix;
    }
    if (rest != 0) {
        return RmdStatus::residual_too_wide;
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
    if (staged < kSigned21Min || staged > kSigned21Max) {
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
        case RmdStatus::residual_too_wide:  return "rmd: residual exceeds the width-native lane capacity";
        case RmdStatus::unsupported_route:  return "rmd: route is unsupported by the exact result contract";
        case RmdStatus::overflow:           return "rmd: integer overflow";
        case RmdStatus::allocation_failure: return "rmd: allocation failed";
        case RmdStatus::execution_failed:   return "rmd: execution failed";
    }
    return "rmd: unknown status";
}

bool decompose_balanced_radix256(int32_t residual, BalancedDigits & out) {
    out = BalancedDigits{};
    int64_t rest = residual;
    for (size_t lane = 0; lane < kLegacyRadix256Lanes; ++lane) {
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
    for (size_t lane = 0; lane < kLegacyRadix256Lanes; ++lane) {
        value += static_cast<int64_t>(digits.digits[lane]) * place;
        place *= 256;
    }
    return value;
}

RmdStatus read_packet_digit(const StripePacket & packet,
                            const BlockDescriptor & block,
                            uint8_t lane_position,
                            size_t row,
                            size_t k,
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
    if (block.active_lane_count == 0 ||
        block.active_lane_count > contract.lane_capacity ||
        block.lane_ids[lane_position] >= contract.lane_capacity ||
        (block.active_lane_mask & static_cast<uint8_t>(
             1u << block.lane_ids[lane_position])) == 0) {
        return RmdStatus::invalid_packet;
    }

    size_t expected_values = 0;
    size_t expected_bytes = 0;
    size_t expected_value_end = 0;
    size_t expected_byte_end = 0;
    size_t mapped_byte_offset = 0;
    size_t packet_byte_count = 0;
    if (!checked_activation_sizes(packet.digit_bits, block.active_lane_count,
                                  block.rows_padded, block.padded_k_count,
                                  expected_values, expected_bytes) ||
        block.activation_byte_count != expected_bytes ||
        !checked_add(block.activation_offset, expected_values, expected_value_end) ||
        !checked_add(block.activation_byte_offset, expected_bytes, expected_byte_end) ||
        expected_value_end > packet.activation_value_count) {
        return RmdStatus::invalid_packet;
    }
    if (packet.digit_storage == DigitStorage::packed_signed_int4) {
        if (block.activation_offset % 2 != 0 ||
            packet.activation_value_count % 2 != 0) {
            return RmdStatus::invalid_packet;
        }
        mapped_byte_offset = block.activation_offset / 2;
        packet_byte_count = packet.activation_value_count / 2;
    } else if (packet.digit_storage == DigitStorage::signed_int8) {
        mapped_byte_offset = block.activation_offset;
        packet_byte_count = packet.activation_value_count;
    } else if (!checked_mul(block.activation_offset, sizeof(int16_t),
                            mapped_byte_offset) ||
               !checked_mul(packet.activation_value_count, sizeof(int16_t),
                            packet_byte_count)) {
        return RmdStatus::invalid_packet;
    }
    if (block.activation_byte_offset != mapped_byte_offset) {
        return RmdStatus::invalid_packet;
    }

    size_t lane_row = 0;
    size_t row_offset = 0;
    int32_t staged = 0;
    if (!checked_mul(lane_position, block.rows_padded, lane_row) ||
        !checked_add(lane_row, row, lane_row)) {
        return RmdStatus::invalid_packet;
    }

    if (packet.digit_storage == DigitStorage::packed_signed_int4) {
        const size_t row_bytes = block.padded_k_count / 2;
        int8_t decoded = 0;
        if (!checked_mul(lane_row, row_bytes, row_offset) ||
            !checked_add(block.activation_byte_offset, row_offset, row_offset) ||
            row_offset >= expected_byte_end ||
            expected_byte_end > packet.stacked_activation.packed_int4.size() ||
            packet.stacked_activation.packed_int4.size() != packet_byte_count ||
            !packet.stacked_activation.signed_int8.empty() ||
            !packet.stacked_activation.signed_int16.empty() ||
            !quants::wreader::decode_native_mvin_q4(
                packet.stacked_activation.packed_int4.data() + row_offset,
                row_bytes, block.padded_k_count, k, decoded)) {
            return RmdStatus::invalid_packet;
        }
        staged = decoded;
    } else {
        if (!checked_mul(lane_row, block.padded_k_count, row_offset) ||
            !checked_add(block.activation_offset, row_offset, row_offset) ||
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
            if (block.activation_byte_offset % alignof(int16_t) != 0 ||
                block.activation_byte_count % sizeof(int16_t) != 0 ||
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
    digit = staged;
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
    const size_t block_id = original_k / kBlockSize;
    const size_t block_local_k = original_k % kBlockSize;
    if (block_id > std::numeric_limits<uint32_t>::max()) {
        status_ = RmdStatus::invalid_arguments;
        return false;
    }

    try {
        BlockAccum & accum = blocks_[static_cast<uint32_t>(block_id)];
        accum.k.insert(static_cast<uint16_t>(block_local_k));
        for (uint8_t lane = 0; lane < digits.lane_capacity; ++lane) {
            if (digits.digits[lane] == 0) {
                continue;
            }
            accum.lane_mask |= static_cast<uint8_t>(1u << lane);
            entries_.push_back({
                static_cast<uint32_t>(block_id),
                static_cast<uint32_t>(local_row),
                static_cast<uint16_t>(block_local_k),
                lane,
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
            for (uint8_t lane = 0; lane < contract.lane_capacity; ++lane) {
                if ((accum.lane_mask & static_cast<uint8_t>(1u << lane)) != 0) {
                    descriptor.lane_ids[lane_count++] = lane;
                }
            }
            descriptor.active_lane_count = lane_count;

            size_t block_activation_values = 0;
            size_t block_activation_bytes = 0;
            size_t block_output = 0;
            if (lane_count == 0 ||
                !checked_activation_sizes(digit_bits_, lane_count, rows_padded,
                                          padded_k, block_activation_values,
                                          block_activation_bytes) ||
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
            std::vector<uint16_t> compact(kBlockSize, std::numeric_limits<uint16_t>::max());
            uint16_t compact_index = 0;
            for (const uint16_t local_k : accum.k) {
                compact[local_k] = compact_index++;
                packet->k_indices.push_back(local_k);
            }
            compact_index_by_block.emplace(block_id, std::move(compact));

            if (!checked_add(k_cursor, accum.k.size(), k_cursor) ||
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

            block_index_by_id.emplace(block_id, packet->blocks.size());
            packet->blocks.push_back(descriptor);
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

        for (const DigitEntry & entry : entries_) {
            const BlockDescriptor & descriptor =
                packet->blocks[block_index_by_id[entry.block_id]];
            const uint16_t compact_k =
                compact_index_by_block[entry.block_id][entry.block_local_k];
            uint8_t lane_position = kMaxNativeRadixLanes;
            for (uint8_t position = 0; position < descriptor.active_lane_count; ++position) {
                if (descriptor.lane_ids[position] == entry.lane) {
                    lane_position = position;
                    break;
                }
            }
            if (lane_position == kMaxNativeRadixLanes ||
                compact_k >= descriptor.compact_k_count) {
                status_ = RmdStatus::invalid_packet;
                return nullptr;
            }
            const RmdStatus write = write_packet_digit(
                *packet, descriptor, lane_position, entry.local_row, compact_k, entry.digit);
            if (write != RmdStatus::success) {
                status_ = write;
                return nullptr;
            }
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

        uint8_t rebuilt_mask = 0;
        for (uint8_t position = 0; position < block.active_lane_count; ++position) {
            const uint8_t lane_id = block.lane_ids[position];
            if (lane_id >= contract.lane_capacity ||
                (position != 0 && lane_id <= block.lane_ids[position - 1])) {
                return RmdStatus::invalid_packet;
            }
            const uint8_t bit = static_cast<uint8_t>(1u << lane_id);
            if ((rebuilt_mask & bit) != 0) {
                return RmdStatus::invalid_packet;
            }
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
        if (!checked_activation_sizes(packet.digit_bits, block.active_lane_count,
                                      block.rows_padded, block.padded_k_count,
                                      block_activation_values, block_activation_bytes) ||
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
        for (uint8_t position = 0; position < block.active_lane_count; ++position) {
            bool lane_has_value = false;
            for (size_t row = 0; row < block.rows_padded; ++row) {
                for (size_t k = 0; k < block.padded_k_count; ++k) {
                    int32_t digit = 0;
                    if (read_packet_digit(packet, block, position, row, k, digit) !=
                        RmdStatus::success) {
                        return RmdStatus::invalid_packet;
                    }
                    const bool logical =
                        row < packet.row_count && k < block.compact_k_count;
                    if (!logical && digit != 0) {
                        return RmdStatus::invalid_packet;
                    }
                    lane_has_value = lane_has_value || (logical && digit != 0);
                }
            }
            if (!lane_has_value) {
                return RmdStatus::invalid_packet;
            }
        }
        for (size_t row = 0; row < packet.row_count; ++row) {
            for (size_t k = 0; k < block.compact_k_count; ++k) {
                bool residual_nonzero = false;
                for (uint8_t position = 0;
                     position < block.active_lane_count; ++position) {
                    int32_t digit = 0;
                    if (read_packet_digit(packet, block, position, row, k,
                                          digit) != RmdStatus::success) {
                        return RmdStatus::invalid_packet;
                    }
                    residual_nonzero = residual_nonzero || digit != 0;
                }
                rebuilt_residual_event_count += residual_nonzero;
            }
        }
    }
    if (packet.residual_event_count == 0 ||
        packet.residual_event_count != rebuilt_residual_event_count) {
        return RmdStatus::invalid_packet;
    }

    return RmdStatus::success;
}

}
