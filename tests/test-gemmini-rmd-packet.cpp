#include "../ggml/src/ggml-gemmini/quants/common/weight_reader.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

namespace rmd = ggml::gemmini::rmd;
namespace wreader = ggml::gemmini::quants::wreader;

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

template <typename T, size_t N>
rmd::StripePacketHandle build_packet(uint8_t digit_bits,
                                     const std::array<T, N> & digits) {
    rmd::RmdStripeBuilder builder;
    builder.reset(17, 5, 1, N, 3, digit_bits);
    for (size_t k = 0; k < digits.size(); ++k) {
        if (!builder.add_residual(0, k, static_cast<int32_t>(digits[k]))) {
            return nullptr;
        }
    }
    return builder.finish();
}

bool descriptor_equals(const rmd::BlockDescriptor & left,
                       const rmd::BlockDescriptor & right) {
    if (left.groups.size() != right.groups.size()) return false;
    for (size_t index = 0; index < left.groups.size(); ++index) {
        const auto & a = left.groups[index];
        const auto & b = right.groups[index];
        if (a.lane_positions != b.lane_positions || a.k_mask != b.k_mask ||
            a.padded_k_count != b.padded_k_count ||
            a.activation_offset != b.activation_offset ||
            a.activation_byte_offset != b.activation_byte_offset ||
            a.activation_byte_count != b.activation_byte_count) return false;
    }
    return left.block_id == right.block_id &&
        left.global_k_begin == right.global_k_begin &&
        left.compact_k_count == right.compact_k_count &&
        left.padded_k_count == right.padded_k_count &&
        left.active_lane_mask == right.active_lane_mask &&
        left.active_lane_count == right.active_lane_count &&
        left.lane_ids == right.lane_ids &&
        left.lane_k_masks == right.lane_k_masks &&
        left.k_index_offset == right.k_index_offset &&
        left.activation_offset == right.activation_offset &&
        left.activation_byte_offset == right.activation_byte_offset &&
        left.activation_byte_count == right.activation_byte_count &&
        left.output_value_offset == right.output_value_offset &&
        left.rows_padded == right.rows_padded &&
        left.lane_stride_values == right.lane_stride_values;
}

bool packet_equals(const rmd::StripePacket & left,
                   const rmd::StripePacket & right) {
    if (left.version != right.version ||
        left.digit_bits != right.digit_bits ||
        left.lane_capacity != right.lane_capacity ||
        left.digit_storage != right.digit_storage ||
        left.int4_packing != right.int4_packing ||
        left.stripe_id != right.stripe_id ||
        left.row_begin != right.row_begin ||
        left.row_count != right.row_count ||
        left.logical_k != right.logical_k ||
        left.logical_j != right.logical_j ||
        left.j_padded != right.j_padded ||
        left.block_size != right.block_size ||
        left.array_dim != right.array_dim ||
        left.activation_value_count != right.activation_value_count ||
        left.residual_event_count != right.residual_event_count ||
        left.total_output_values != right.total_output_values ||
        left.k_indices != right.k_indices ||
        !(left.stacked_activation == right.stacked_activation) ||
        left.blocks.size() != right.blocks.size()) {
        return false;
    }
    for (size_t i = 0; i < left.blocks.size(); ++i) {
        if (!descriptor_equals(left.blocks[i], right.blocks[i])) {
            return false;
        }
    }
    return true;
}

bool rejects_without_mutation(rmd::StripePacket packet, const char * message) {
    const rmd::StripePacket before = packet;
    return check(rmd::validate_packet(packet) == rmd::RmdStatus::invalid_packet &&
                     packet_equals(packet, before),
                 message);
}

bool check_round_trip(const rmd::StripePacket & packet,
                      const std::vector<int32_t> & expected,
                      const char * message) {
    if (packet.blocks.size() != 1) {
        return check(false, message);
    }
    const rmd::BlockDescriptor & block = packet.blocks.front();
    for (size_t k = 0; k < expected.size(); ++k) {
        int32_t digit = 0;
        if (rmd::read_packet_digit(packet, block, 0, 0, k, digit) !=
                rmd::RmdStatus::success ||
            digit != expected[k]) {
            return check(false, message);
        }
    }
    for (size_t k = expected.size(); k < block.padded_k_count; ++k) {
        int32_t digit = 1;
        if (rmd::read_packet_digit(packet, block, 0, 0, k, digit) !=
                rmd::RmdStatus::success ||
            digit != 0) {
            return check(false, message);
        }
    }
    return true;
}

bool test_width_native_round_trip() {
    constexpr std::array<int8_t, 16> w4 = {
        -8, -7, -6, -5, -4, -3, -2, -1,
         1,  2,  3,  4,  5,  6,  7, -8,
    };
    constexpr std::array<int16_t, 16> w8 = {
        -128, -127, -64, -2, -1, 1, 2, 3,
        4, 5, 6, 7, 63, 64, 126, 127,
    };
    constexpr std::array<int32_t, 16> w16 = {
        -32768, -32767, -4096, -2, -1, 1, 2, 3,
        4, 5, 6, 7, 4096, 16384, 32766, 32767,
    };

    const rmd::StripePacketHandle p4 = build_packet(4, w4);
    const rmd::StripePacketHandle p8 = build_packet(8, w8);
    const rmd::StripePacketHandle p16 = build_packet(16, w16);
    if (!check(p4 != nullptr && p8 != nullptr && p16 != nullptr,
               "width-native packet fixtures build")) {
        return false;
    }

    bool ok = true;
    ok = check(p4->version == rmd::kPacketVersion &&
                   p4->digit_bits == 4 && p4->lane_capacity == 9 &&
                   p4->digit_storage == rmd::DigitStorage::packed_signed_int4 &&
                   p4->int4_packing == rmd::Int4Packing::adjacent_low_nibble_first &&
                   p4->blocks.front().active_lane_count == 1,
               "W4 packet metadata and active-lane trimming") && ok;
    ok = check(p8->digit_bits == 8 && p8->lane_capacity == 5 &&
                   p8->digit_storage == rmd::DigitStorage::signed_int8 &&
                   p8->int4_packing == rmd::Int4Packing::none,
               "W8 packet metadata") && ok;
    ok = check(p16->digit_bits == 16 && p16->lane_capacity == 3 &&
                   p16->digit_storage == rmd::DigitStorage::signed_int16 &&
                   p16->int4_packing == rmd::Int4Packing::none,
               "W16 packet metadata") && ok;

    std::vector<uint8_t> expected_w4(
        p4->blocks.front().activation_byte_count, uint8_t{0x00});
    constexpr std::array<uint8_t, 8> w4_literal = {
        0x98, 0xba, 0xdc, 0xfe, 0x21, 0x43, 0x65, 0x87,
    };
    std::copy(w4_literal.begin(), w4_literal.end(), expected_w4.begin());
    ok = check(p4->stacked_activation.packed_int4 == expected_w4,
               "W4 payload matches adjacent native-transport nibble literals") && ok;

    std::vector<int8_t> expected_w8(p8->activation_value_count, int8_t{0});
    std::copy(w8.begin(), w8.end(), expected_w8.begin());
    ok = check(p8->stacked_activation.signed_int8 == expected_w8,
               "W8 payload stores scalar signed bytes") && ok;

    std::vector<int16_t> expected_w16(p16->activation_value_count, int16_t{0});
    std::transform(w16.begin(), w16.end(), expected_w16.begin(),
                   [](int32_t value) { return static_cast<int16_t>(value); });
    ok = check(p16->stacked_activation.signed_int16 == expected_w16 &&
                   p16->blocks.front().activation_byte_offset % alignof(int16_t) == 0 &&
                   reinterpret_cast<uintptr_t>(
                       p16->stacked_activation.signed_int16.data()) %
                       alignof(int16_t) == 0,
               "W16 payload stores aligned scalar signed elements") && ok;

    std::vector<int32_t> expected4(w4.begin(), w4.end());
    std::vector<int32_t> expected8(w8.begin(), w8.end());
    std::vector<int32_t> expected16(w16.begin(), w16.end());
    ok = check_round_trip(*p4, expected4, "W4 payload round-trips signed digits") && ok;
    ok = check_round_trip(*p8, expected8, "W8 payload round-trips signed digits") && ok;
    ok = check_round_trip(*p16, expected16, "W16 payload round-trips signed digits") && ok;
    ok = check(rmd::validate_packet(*p4) == rmd::RmdStatus::success &&
                   rmd::validate_packet(*p8) == rmd::RmdStatus::success &&
                   rmd::validate_packet(*p16) == rmd::RmdStatus::success,
               "all width-native packets validate") && ok;

    int8_t decoded = 0;
    constexpr std::array<uint8_t, 2> rtl_port_vectors = {0x10, 0xf8};
    ok = check(wreader::decode_native_mvin_q4(
                   rtl_port_vectors.data(), rtl_port_vectors.size(), 4, 0, decoded) &&
                   decoded == 0 &&
                   wreader::decode_native_mvin_q4(
                       rtl_port_vectors.data(), rtl_port_vectors.size(), 4, 1, decoded) &&
                   decoded == 1 &&
                   wreader::decode_native_mvin_q4(
                       rtl_port_vectors.data(), rtl_port_vectors.size(), 4, 2, decoded) &&
                   decoded == -8 &&
                   wreader::decode_native_mvin_q4(
                       rtl_port_vectors.data(), rtl_port_vectors.size(), 4, 3, decoded) &&
                   decoded == -1,
               "native transport bytes 10/f8 decode as adjacent [0,1]/[-8,-1]") && ok;
    ok = check(wreader::decode_native_mvin_q4(
                   w4_literal.data(), w4_literal.size(), 16, 0, decoded) &&
                   decoded == -8 &&
                   wreader::decode_native_mvin_q4(
                       w4_literal.data(), w4_literal.size(), 16, 1, decoded) &&
                   decoded == -7 &&
                   wreader::decode_native_mvin_q4(
                       w4_literal.data(), w4_literal.size(), 16, 15, decoded) &&
                   decoded == -8,
               "packet literals round-trip through signed INT4 native packing") && ok;
    decoded = 42;
    ok = check(!wreader::decode_native_mvin_q4(
                   w4_literal.data(), w4_literal.size() - 1, 16, 15, decoded) &&
                   decoded == 42,
               "truncated native Q4 decode leaves output unchanged") && ok;

    std::printf("W4 bytes=");
    for (size_t i = 0; i < w4_literal.size(); ++i) {
        std::printf("%s%02x", i == 0 ? "" : " ",
                    static_cast<unsigned>(w4_literal[i]));
    }
    std::printf(" payload_bytes=%zu\n", p4->stacked_activation.packed_int4.size());

    std::printf("W8 bytes=");
    for (size_t i = 0; i < 4; ++i) {
        std::printf("%s%02x", i == 0 ? "" : " ",
                    static_cast<unsigned>(static_cast<uint8_t>(
                        p8->stacked_activation.signed_int8[i])));
    }
    std::printf(" elements=%zu\n", p8->stacked_activation.signed_int8.size());

    std::printf("W16 elements=%d %d %d %d elements=%zu\n",
                p16->stacked_activation.signed_int16[0],
                p16->stacked_activation.signed_int16[4],
                p16->stacked_activation.signed_int16[5],
                p16->stacked_activation.signed_int16[15],
                p16->stacked_activation.signed_int16.size());
    return ok;
}

bool test_lane_capacity_and_trimming() {
    struct LaneCase {
        uint8_t bits;
        int32_t residual;
        uint8_t capacity;
        uint8_t low_lane;
        uint8_t high_lane;
    };
    constexpr std::array<LaneCase, 3> cases = {{
        {4,  std::numeric_limits<int32_t>::min() + 1, 9, 0, 7},
        {8,  (int32_t{1} << 24) + 1, 5, 0, 3},
        {16, (int32_t{1} << 16) + 1, 3, 0, 1},
    }};

    bool ok = true;
    for (const LaneCase & test : cases) {
        rmd::RmdStripeBuilder builder;
        builder.reset(19, 0, 1, 1, 1, test.bits);
        const bool added = builder.add_residual(0, 0, test.residual);
        const rmd::StripePacketHandle packet = builder.finish();
        ok = check(added && packet != nullptr &&
                       packet->lane_capacity == test.capacity &&
                       packet->blocks.front().active_lane_count == 2 &&
                       packet->blocks.front().lane_ids[0] == test.low_lane &&
                       packet->blocks.front().lane_ids[1] == test.high_lane,
                   "width-native packet trims inactive middle lanes") && ok;

    }
    return ok;
}

bool test_int32_packet_round_trip() {
    constexpr std::array<int32_t, 7> residuals = {
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max(),
        -(int32_t{1} << 20) - 1,
        int32_t{1} << 20,
        0,
        1,
        0x77777777,
    };
    bool ok = true;
    for (uint8_t bits : {4, 8, 16}) {
        rmd::RmdStripeBuilder builder;
        builder.reset(20, 5, 2, residuals.size(), 1, bits);
        for (size_t row = 0; row < 2; ++row) {
            for (size_t k = 0; k < residuals.size(); ++k) {
                if (!builder.add_residual(row, k, residuals[k])) {
                    return check(false, "full INT32 residuals are accepted");
                }
            }
        }
        const auto packet = builder.finish();
        if (!check(packet != nullptr &&
                       rmd::validate_packet(*packet) == rmd::RmdStatus::success,
                   "full INT32 packet validates")) return false;

        rmd::RmdStatus status;
        const auto sliced = rmd::slice_packets({packet}, 6, 7, 21, status);
        if (!check(sliced != nullptr && status == rmd::RmdStatus::success &&
                       sliced->row_begin == 6 && sliced->row_count == 1 &&
                       rmd::validate_packet(*sliced) == rmd::RmdStatus::success,
                   "slicing rebuilds full INT32 residuals without narrowing")) return false;

        for (const auto & candidate : {packet, sliced}) {
            const auto & block = candidate->blocks.front();
            const uint8_t carry_lane = 32 / bits;
            ok = check(block.active_lane_count == candidate->lane_capacity &&
                           block.lane_ids[block.active_lane_count - 1] == carry_lane &&
                           (block.active_lane_mask & (uint16_t{1} << carry_lane)) != 0 &&
                           block.lane_k_masks[carry_lane] == (uint32_t{1} << 1),
                       "all 9/5/3 lanes and the carry lane retain original IDs and K masks") && ok;
            for (size_t row = 0; row < candidate->row_count; ++row) {
                std::array<int64_t, residuals.size()> restored{};
                for (size_t k = 0; k < block.compact_k_count; ++k) {
                    const size_t original_k = candidate->k_indices[block.k_index_offset + k];
                    for (size_t lane = 0; lane < block.active_lane_count; ++lane) {
                        int32_t digit = 0;
                        if (!check(rmd::read_packet_digit(*candidate, block, lane, row, k, digit) ==
                                       rmd::RmdStatus::success,
                                   "full INT32 packet digit decodes")) return false;
                        restored[original_k] += static_cast<int64_t>(digit) *
                            (int64_t{1} << (bits * block.lane_ids[lane]));
                    }
                }
                ok = check(std::equal(restored.begin(), restored.end(), residuals.begin()),
                           "original and sliced packets reconstruct INT32 endpoints exactly") && ok;
            }
        }
    }
    return ok;
}

bool test_multiblock_offsets_and_output_layout() {
    std::array<rmd::StripePacketHandle, 3> packets{};
    constexpr std::array<uint8_t, 3> widths = {4, 8, 16};
    for (size_t i = 0; i < widths.size(); ++i) {
        rmd::RmdStripeBuilder builder;
        builder.reset(29, 7, 1, 64, 3, widths[i]);
        if (!builder.add_residual(0, 1, 1) ||
            !builder.add_residual(0, 33, -1)) {
            return check(false, "two-block packet residuals build");
        }
        packets[i] = builder.finish();
        if (!check(packets[i] != nullptr, "two-block packet finishes")) {
            return false;
        }
    }

    bool ok = true;
    constexpr size_t block_values = rmd::kArrayDim * rmd::kArrayDim;
    constexpr std::array<size_t, 3> block_bytes = {
        block_values / 2, block_values, block_values * sizeof(int16_t)};
    for (size_t i = 0; i < packets.size(); ++i) {
        const rmd::StripePacket & packet = *packets[i];
        if (!check(packet.blocks.size() == 2, "two-block packet retains both blocks")) {
            return false;
        }
        const rmd::BlockDescriptor & first = packet.blocks[0];
        const rmd::BlockDescriptor & second = packet.blocks[1];
        ok = check(packet.array_dim == rmd::kArrayDim &&
                       packet.block_size == rmd::kBlockSize &&
                       packet.j_padded == rmd::kArrayDim &&
                       first.rows_padded == rmd::kArrayDim &&
                       first.padded_k_count == rmd::kArrayDim,
                   "DIM geometry is width-independent") && ok;
        ok = check(packet.blocks.size() == 2 &&
                       first.block_id == 0 && first.global_k_begin == 0 &&
                       second.block_id == 1 &&
                       second.global_k_begin == rmd::kBlockSize &&
                       first.k_index_offset == 0 && second.k_index_offset == 1,
                   "original block identity and compact K offsets are preserved") && ok;
        ok = check(first.activation_offset == 0 &&
                       second.activation_offset == block_values &&
                       first.activation_byte_offset == 0 &&
                       first.activation_byte_count == block_bytes[i] &&
                       second.activation_byte_offset == block_bytes[i] &&
                       second.activation_byte_count == block_bytes[i],
                   "native payload byte and value offsets tile exactly") && ok;
        ok = check(first.output_value_offset == 0 &&
                       second.output_value_offset == block_values &&
                       first.lane_stride_values == block_values &&
                       second.lane_stride_values == block_values &&
                       packet.total_output_values == 2 * block_values,
                   "canonical output layout is width-independent") && ok;
        ok = check(rmd::validate_packet(packet) == rmd::RmdStatus::success,
                   "two-block native packet validates") && ok;
    }
    ok = check(packets[2]->blocks[1].activation_byte_offset % alignof(int16_t) == 0,
               "every W16 block offset is aligned") && ok;
    return ok;
}

bool test_malformed_packets_reject_atomically() {
    constexpr std::array<int8_t, 16> w4 = {
        -8, -7, -6, -5, -4, -3, -2, -1,
         1,  2,  3,  4,  5,  6,  7, -8,
    };
    constexpr std::array<int16_t, 16> w8 = {
        -128, -127, -64, -2, -1, 1, 2, 3,
        4, 5, 6, 7, 63, 64, 126, 127,
    };
    constexpr std::array<int32_t, 16> w16 = {
        -32768, -32767, -4096, -2, -1, 1, 2, 3,
        4, 5, 6, 7, 4096, 16384, 32766, 32767,
    };
    const rmd::StripePacketHandle p4 = build_packet(4, w4);
    const rmd::StripePacketHandle p8 = build_packet(8, w8);
    const rmd::StripePacketHandle p16 = build_packet(16, w16);
    if (!check(p4 != nullptr && p8 != nullptr && p16 != nullptr,
               "malformed packet fixtures build")) {
        return false;
    }

    bool ok = true;
    rmd::StripePacket malformed = *p4;
    malformed.version = rmd::kPacketVersion - 1;
    ok = rejects_without_mutation(malformed, "stale packet version rejects atomically") && ok;

    malformed = *p4;
    malformed.int4_packing = rmd::Int4Packing::none;
    ok = rejects_without_mutation(malformed, "missing Q4 nibble metadata rejects atomically") && ok;

    malformed = *p4;
    malformed.stacked_activation.packed_int4.pop_back();
    ok = rejects_without_mutation(malformed, "truncated W4 payload rejects atomically") && ok;

    malformed = *p8;
    malformed.stacked_activation.signed_int8.pop_back();
    ok = rejects_without_mutation(malformed, "truncated W8 payload rejects atomically") && ok;

    malformed = *p8;
    ++malformed.residual_event_count;
    ok = rejects_without_mutation(
             malformed, "forged source residual event count rejects atomically") && ok;

    malformed = *p16;
    malformed.stacked_activation.signed_int16.pop_back();
    ok = rejects_without_mutation(malformed, "truncated W16 payload rejects atomically") && ok;

    malformed = *p16;
    malformed.blocks.front().activation_byte_offset = 1;
    ok = rejects_without_mutation(malformed, "misaligned W16 offset rejects atomically") && ok;

    malformed = *p16;
    malformed.blocks.front().activation_byte_count -= 2;
    ok = rejects_without_mutation(malformed, "short W16 block extent rejects atomically") && ok;

    malformed = *p8;
    malformed.digit_storage = rmd::DigitStorage::signed_int16;
    ok = rejects_without_mutation(malformed, "mismatched digit storage rejects atomically") && ok;

    malformed = *p16;
    malformed.blocks.front().lane_ids[0] = malformed.lane_capacity;
    ok = rejects_without_mutation(malformed, "excess lane id rejects atomically") && ok;

    malformed = *p4;
    malformed.lane_capacity = 7;
    ok = rejects_without_mutation(malformed, "wrong lane capacity rejects atomically") && ok;

    malformed = *p4;
    malformed.blocks.front().active_lane_mask ^= 0x02;
    ok = rejects_without_mutation(malformed, "lane mask/count mismatch rejects atomically") && ok;

    malformed = *p8;
    ++malformed.blocks.front().active_lane_count;
    ok = rejects_without_mutation(malformed, "forged active lane count rejects atomically") && ok;

    malformed = *p16;
    malformed.blocks.front().lane_ids[malformed.blocks.front().active_lane_count] = 1;
    ok = rejects_without_mutation(malformed, "nonzero inactive lane tail rejects atomically") && ok;

    malformed = *p8;
    ++malformed.blocks.front().activation_offset;
    ok = rejects_without_mutation(malformed, "gapped activation extent rejects atomically") && ok;

    malformed = *p8;
    ++malformed.blocks.front().output_value_offset;
    ok = rejects_without_mutation(malformed, "gapped output extent rejects atomically") && ok;

    malformed = *p8;
    malformed.stacked_activation.signed_int8.push_back(0);
    ok = rejects_without_mutation(malformed, "oversized payload rejects atomically") && ok;

    malformed = *p16;
    malformed.stacked_activation.signed_int8.assign(1, 1);
    ok = rejects_without_mutation(malformed, "one-byte A16 payload rejects atomically") && ok;

    malformed = *p8;
    malformed.array_dim += 1;
    ok = rejects_without_mutation(malformed, "wrong packet DIM rejects atomically") && ok;

    malformed = *p16;
    malformed.logical_j = std::numeric_limits<size_t>::max() -
        (std::numeric_limits<size_t>::max() % rmd::kArrayDim);
    malformed.j_padded = malformed.logical_j;
    ok = rejects_without_mutation(malformed, "overflowing packet geometry rejects atomically") && ok;

    int32_t sentinel = 0x13579;
    ok = check(rmd::read_packet_digit(*p16, p16->blocks.front(), 0,
                                     p16->blocks.front().rows_padded, 0,
                                     sentinel) == rmd::RmdStatus::invalid_arguments &&
                   sentinel == 0x13579,
               "out-of-range digit read leaves caller output unchanged") && ok;

    malformed = *p16;
    malformed.stacked_activation.signed_int16.pop_back();
    sentinel = 0x2468a;
    ok = check(rmd::read_packet_digit(malformed, malformed.blocks.front(),
                                     0, 0, 15, sentinel) ==
                       rmd::RmdStatus::invalid_packet &&
                   sentinel == 0x2468a,
               "truncated digit read leaves caller output unchanged") && ok;
    return ok;
}

bool test_final_group_payload() {
    bool ok = true;
    const size_t rows = 9;
    for (uint8_t bits : {4, 8, 16}) {
        rmd::RmdStripeBuilder builder;
        builder.reset(0, 0, rows, 32, 7, bits);
        for (size_t k = 0; k < 32; ++k) {
            if (!builder.add_residual(k % rows, k, k < 16 ? 1 : int32_t{1} << bits)) {
                return check(false, "disjoint lane support builds");
            }
        }
        const auto packet = builder.finish();
        const size_t expected_groups = rmd::kArrayDim < rmd::kBlockSize ? 2 : 1;
        if (!check(packet != nullptr && packet->blocks[0].groups.size() == expected_groups,
                   "disjoint support groups stay within the baseline K calls")) return false;
        const auto & block = packet->blocks[0];
        ok = check((expected_groups == 2 ?
                       block.groups[0].k_mask == 0x0000ffffu &&
                       block.groups[1].k_mask == 0xffff0000u &&
                       block.groups[0].lane_positions == std::vector<uint8_t>{0} &&
                       block.groups[1].lane_positions == std::vector<uint8_t>{1} :
                       block.groups[0].k_mask == 0xffffffffu &&
                       block.groups[0].lane_positions == std::vector<uint8_t>({0, 1})) &&
                   packet->activation_value_count == expected_groups * rmd::kArrayDim * rmd::kArrayDim,
                   "final groups allocate only their native padded K extents") && ok;
        for (size_t row = 0; row < rows; ++row) {
            for (size_t k = 0; k < 32; ++k) {
                for (uint8_t lane = 0; lane < 2; ++lane) {
                    int32_t digit = -1;
                    ok = check(rmd::read_packet_digit(*packet, block, lane, row, k, digit) ==
                                   rmd::RmdStatus::success &&
                               digit == static_cast<int32_t>(row == k % rows && lane == (k >= 16)),
                               "group decoding restores original compact K and absent-lane zeros") && ok;
                }
            }
        }
        rmd::RmdStatus status;
        const auto sliced = rmd::slice_packets({packet}, 1, 3, 1, status);
        ok = check(sliced != nullptr && status == rmd::RmdStatus::success &&
                   sliced->row_begin == 1 && sliced->row_count == 2 &&
                   rmd::validate_packet(*sliced) == rmd::RmdStatus::success,
                   "slicing rebuilds final groups for the selected rows") && ok;
        auto malformed = *packet;
        malformed.blocks[0].groups.back().lane_positions.clear();
        ok = rejects_without_mutation(malformed, "empty group rejects") && ok;
        malformed = *packet;
        malformed.blocks[0].groups.back().lane_positions.back() = 0;
        ok = rejects_without_mutation(malformed, "duplicate group lane ownership rejects") && ok;
        malformed = *packet;
        malformed.blocks[0].active_lane_count = 255;
        malformed.blocks[0].groups[0].lane_positions[0] = 254;
        int32_t sentinel = 12345;
        ok = check(rmd::read_packet_digit(malformed, malformed.blocks[0], 254, 0, 0,
                                         sentinel) == rmd::RmdStatus::invalid_packet &&
                   sentinel == 12345,
                   "malformed group read rejects without indexing beyond lane IDs") && ok;
        malformed = *packet;
        malformed.blocks[0].groups.back().k_mask &= ~(uint32_t{1} << 31);
        ok = rejects_without_mutation(malformed, "missing group K rejects") && ok;
        malformed = *packet;
        ++malformed.blocks[0].groups.back().activation_offset;
        ok = rejects_without_mutation(malformed, "misaligned group extent rejects") && ok;
        malformed = *packet;
        const size_t row_padding = block.groups[0].lane_positions.size() *
            packet->row_count * block.groups[0].padded_k_count;
        if (bits == 4) malformed.stacked_activation.packed_int4[row_padding / 2] = 1;
        if (bits == 8) malformed.stacked_activation.signed_int8[row_padding] = 1;
        if (bits == 16) malformed.stacked_activation.signed_int16[row_padding] = 1;
        ok = rejects_without_mutation(malformed, "nonzero group row padding rejects") && ok;
    }
    return ok;
}

bool test_partition_uses_packed_group_rows() {
    rmd::RmdStripeBuilder builder;
    builder.reset(17, 5, 6, 64, 1, 8);
    for (size_t row = 0; row < 6; ++row) {
        for (size_t k = 0; k < 20; ++k) {
            if (!builder.add_residual(row, 32 + k, int32_t{1} << (8 * (k / 5)))) return false;
        }
    }
    const auto packet = builder.finish();
    if (!check(packet != nullptr && packet->blocks.size() == 1 &&
                   rmd::validate_packet(*packet) == rmd::RmdStatus::success,
               "four disjoint five-K lanes build a valid packet")) return false;
    const auto & block = packet->blocks[0];
    size_t calls = 0;
    for (const auto & group : block.groups) calls += group.padded_k_count / rmd::kArrayDim;
    std::printf("packed-group regression DIM%zu: groups=%zu calls=%zu tiles=%zu\n",
                rmd::kArrayDim, block.groups.size(), calls,
                packet->activation_value_count / (rmd::kArrayDim * rmd::kArrayDim));
    bool ok = check(block.block_id == 1 && block.global_k_begin == 32 &&
                        block.active_lane_count == 4 && block.lane_ids[0] == 0 &&
                        block.lane_ids[1] == 1 && block.lane_ids[2] == 2 &&
                        block.lane_ids[3] == 3 && packet->row_begin == 5,
                    "partition preserves original block, row and lane coordinates");
    ok = check(calls == block.padded_k_count / rmd::kArrayDim,
               "partition keeps the baseline K-call budget") && ok;
    if (rmd::kArrayDim == 16) {
        ok = check(block.groups.size() == 2 &&
                       block.groups[0].lane_positions == std::vector<uint8_t>({0, 1}) &&
                       block.groups[1].lane_positions == std::vector<uint8_t>({2, 3}) &&
                       block.groups[0].k_mask == 0x3ffu &&
                       block.groups[1].k_mask == 0xffc00u &&
                       packet->activation_value_count == 512,
                   "six rows use the two-tile 2+2 split instead of the three-tile 3+1 split") && ok;
    } else {
        ok = check(block.groups.size() == 1 && block.groups[0].k_mask == 0xfffffu &&
                       packet->activation_value_count == rmd::kArrayDim * rmd::kArrayDim,
                   "one-call baseline keeps all four lanes in one group") && ok;
    }
    for (size_t row = 0; row < 6; ++row) {
        int64_t output = 0;
        for (size_t k = 0; k < 20; ++k) {
            int64_t residual = 0;
            for (uint8_t lane = 0; lane < block.active_lane_count; ++lane) {
                int32_t digit = 0;
                if (!check(rmd::read_packet_digit(*packet, block, lane, row, k, digit) ==
                               rmd::RmdStatus::success,
                           "partitioned digit decodes")) return false;
                residual += digit * (int64_t{1} << (8 * block.lane_ids[lane]));
            }
            ok = check(packet->k_indices[k] == k &&
                           residual == (int64_t{1} << (8 * (k / 5))),
                       "partitioned payload reconstructs every original residual and K") && ok;
            output += residual * (block.global_k_begin + packet->k_indices[k] + 1);
        }
        ok = check(output == 175 + 200 * int64_t{256} + 225 * int64_t{65536} +
                               250 * int64_t{16777216},
                   "partitioned payload reproduces the independent weighted output") && ok;
    }
    return ok;
}

bool test_original_k_masks() {
    rmd::RmdStripeBuilder builder;
    builder.reset(0, 0, 2, 64, 3, 8);
    if (!builder.add_residual(0, 0, 1) ||
        !builder.add_residual(0, 17, 65536) ||
        !builder.add_residual(1, 31, -65536) ||
        !builder.add_residual(1, 32, 256)) {
        return check(false, "sparse original-K mask input builds");
    }
    const auto packet = builder.finish();
    if (!check(packet != nullptr && packet->blocks.size() == 2,
               "sparse original-K mask packet builds")) {
        return false;
    }
    const auto & first = packet->blocks[0];
    bool ok = check(first.lane_k_masks[0] == 1 &&
                    first.lane_k_masks[1] == 0 &&
                    first.lane_k_masks[2] == ((uint32_t{1} << 17) | (uint32_t{1} << 31)) &&
                    packet->blocks[1].lane_k_masks[1] == 1 &&
                    packet->k_indices == std::vector<uint16_t>({0, 17, 31, 0}),
                    "lane masks preserve original K and lane positions across blocks");
    auto malformed = *packet;
    malformed.blocks[0].lane_k_masks[2] &= ~(uint32_t{1} << 31);
    ok = rejects_without_mutation(malformed, "missing K support rejects atomically") && ok;
    malformed = *packet;
    malformed.blocks[0].lane_k_masks[7] = 1;
    ok = rejects_without_mutation(malformed, "inactive lane K support rejects atomically") && ok;
    builder.reset(1, 0, 1, 64, 3, 8);
    if (!builder.add_residual(0, 31, 1)) return false;
    const auto reset_packet = builder.finish();
    return check(reset_packet != nullptr && reset_packet->blocks.size() == 1 &&
                 reset_packet->blocks[0].lane_k_masks[0] == (uint32_t{1} << 31) &&
                 reset_packet->blocks[0].lane_k_masks[2] == 0,
                 "builder reset clears lane K masks") && ok;
}

bool test_overlapping_lane_k_content() {
    bool ok = true;
    for (uint8_t bits : {4, 8, 16}) {
        rmd::RmdStripeBuilder builder;
        builder.reset(0, 0, 3, 64, 1, bits);
        const int32_t overlap = (int32_t{1} << bits) + 1;
        if (!builder.add_residual(0, 1, overlap) ||
            !builder.add_residual(0, 31, std::numeric_limits<int32_t>::max()) ||
            !builder.add_residual(1, 1, overlap) ||
            !builder.add_residual(1, 17, -overlap) ||
            !builder.add_residual(1, 31, std::numeric_limits<int32_t>::max()) ||
            !builder.add_residual(2, 63, std::numeric_limits<int32_t>::max())) return false;
        const auto packet = builder.finish();
        if (!check(packet != nullptr && packet->residual_event_count == 6 &&
                       packet->blocks.size() == 2 && packet->blocks[0].groups.size() == 1 &&
                       rmd::validate_packet(*packet) == rmd::RmdStatus::success,
                   "overlapping lanes count each original row/K event once across blocks")) return false;
        const auto & block = packet->blocks[0];
        const uint8_t carry_lane = 32 / bits;
        ok = check(block.lane_k_masks[carry_lane] == (uint32_t{1} << 31) &&
                       packet->blocks[1].lane_k_masks[carry_lane] == (uint32_t{1} << 31),
                   "bit-32 carry preserves sparse original K bit 31 across blocks") && ok;
        int32_t carry_digit = 0;
        ok = check(rmd::read_packet_digit(*packet, block, block.active_lane_count - 1,
                                         0, 2, carry_digit) == rmd::RmdStatus::success &&
                       carry_digit == 1,
                   "sparse carry packs at its group K rank") && ok;

        const auto set_digit = [&](rmd::StripePacket & target, size_t lane,
                                   size_t row, size_t k, uint8_t value) {
            const size_t index = (lane * packet->row_count + row) *
                block.groups[0].padded_k_count + k;
            if (bits == 4) {
                const uint8_t shift = 4 * (index % 2);
                uint8_t & packed = target.stacked_activation.packed_int4[index / 2];
                packed = static_cast<uint8_t>((packed & ~(0x0fu << shift)) | (value << shift));
            } else if (bits == 8) {
                target.stacked_activation.signed_int8[index] = static_cast<int8_t>(value);
            } else {
                target.stacked_activation.signed_int16[index] = value;
            }
        };
        auto malformed = *packet;
        set_digit(malformed, 0, 0, block.compact_k_count, 1);
        ok = rejects_without_mutation(malformed, "nonzero group K padding rejects atomically") && ok;
        malformed = *packet;
        set_digit(malformed, 1, 0, 0, 0);
        set_digit(malformed, 1, 1, 0, 0);
        ok = rejects_without_mutation(malformed,
            "missing actual lane K support rejects even when another lane keeps the events") && ok;
        malformed = *packet;
        malformed.blocks[0].lane_k_masks[1] = 0;
        for (size_t row = 0; row < packet->row_count; ++row) {
            for (size_t k = 0; k < block.compact_k_count; ++k) set_digit(malformed, 1, row, k, 0);
        }
        ok = rejects_without_mutation(malformed, "empty stored lane rejects with consistent support") && ok;
        malformed = *packet;
        malformed.k_indices[0] = 0;
        ok = rejects_without_mutation(malformed, "selected K must equal actual payload support") && ok;
        malformed = *packet;
        --malformed.residual_event_count;
        ok = rejects_without_mutation(malformed, "undercounted row/K events reject atomically") && ok;
    }
    return ok;
}

bool test_group_row_padding() {
    bool ok = true;
    for (uint8_t bits : {4, 8, 16}) {
        const uint8_t upper_lane = bits == 16 ? 1 : 2;
        const int32_t place = int32_t{1} << (bits * upper_lane);
        for (size_t rows : {size_t{1}, rmd::kArrayDim - 1,
                            rmd::kArrayDim, rmd::kArrayDim + 1}) {
            rmd::RmdStripeBuilder builder;
            builder.reset(0, 0, rows, 32, 3, bits);
            for (size_t row = 0; row < rows; ++row) {
                const int32_t low = static_cast<int32_t>(row % 7) - 3;
                const int32_t high = row % 2 == 0 ? 1 : -1;
                if (!builder.add_residual(row, 5, low + high * place)) return false;
            }
            const auto packet = builder.finish();
            if (!check(packet != nullptr && packet->blocks.size() == 1 &&
                           packet->blocks[0].groups.size() == 1,
                       "group row boundary fixture builds")) return false;
            const auto & block = packet->blocks[0];
            const auto & group = block.groups[0];
            const size_t stored_rows = rmd::align_up(2 * rows, rmd::kArrayDim);
            const size_t values = stored_rows * rmd::kArrayDim;
            const size_t bytes = values * bits / 8;
            ok = check(packet->version == 5 && block.active_lane_count == 2 &&
                           block.lane_ids[0] == 0 && block.lane_ids[1] == upper_lane &&
                           group.lane_positions == std::vector<uint8_t>({0, 1}) &&
                           packet->activation_value_count == values &&
                           group.activation_byte_count == bytes &&
                           block.activation_byte_count == bytes &&
                           packet->total_output_values == 2 * block.lane_stride_values &&
                           rmd::validate_packet(*packet) == rmd::RmdStatus::success,
                       "group pads once while retaining original lane IDs and output layout") && ok;

            std::vector<uint8_t> expected4(bits == 4 ? bytes : 0, 0);
            std::vector<int8_t> expected8(bits == 8 ? values : 0, 0);
            std::vector<int16_t> expected16(bits == 16 ? values : 0, 0);
            for (uint8_t lane = 0; lane < 2; ++lane) {
                for (size_t row = 0; row < block.rows_padded; ++row) {
                    const int32_t expected = row >= rows ? 0 : lane == 0 ?
                        static_cast<int32_t>(row % 7) - 3 : row % 2 == 0 ? 1 : -1;
                    int32_t digit = 99;
                    ok = check(rmd::read_packet_digit(*packet, block, lane, row, 0, digit) ==
                                   rmd::RmdStatus::success && digit == expected,
                               "dense lane rows decode with virtual per-lane zero padding") && ok;
                    if (row >= rows) continue;
                    const size_t index = (lane * rows + row) * rmd::kArrayDim;
                    if (bits == 4) expected4[index / 2] = static_cast<uint8_t>(expected) & 0x0f;
                    if (bits == 8) expected8[index] = static_cast<int8_t>(expected);
                    if (bits == 16) expected16[index] = static_cast<int16_t>(expected);
                }
            }
            ok = check(packet->stacked_activation.packed_int4 == expected4 &&
                           packet->stacked_activation.signed_int8 == expected8 &&
                           packet->stacked_activation.signed_int16 == expected16,
                       "native bytes contain adjacent real lane rows and one zero group tail") && ok;
            if (stored_rows > 2 * rows) {
                auto malformed = *packet;
                const size_t index = 2 * rows * rmd::kArrayDim;
                if (bits == 4) malformed.stacked_activation.packed_int4[index / 2] = 0x10;
                if (bits == 8) malformed.stacked_activation.signed_int8[index] = 1;
                if (bits == 16) malformed.stacked_activation.signed_int16[index] = 1;
                ok = rejects_without_mutation(malformed, "nonzero physical group tail rejects") && ok;
            }
            if (rows < block.rows_padded) {
                auto malformed = *packet;
                if (bits == 4) malformed.stacked_activation.packed_int4.pop_back();
                if (bits == 8) malformed.stacked_activation.signed_int8.pop_back();
                if (bits == 16) malformed.stacked_activation.signed_int16.pop_back();
                int32_t sentinel = 123;
                ok = check(rmd::read_packet_digit(malformed, malformed.blocks[0], 1,
                                                  rows, 0, sentinel) ==
                               rmd::RmdStatus::invalid_packet && sentinel == 123,
                           "virtual padding reads still reject truncated native payload") && ok;
            }
        }
    }
    return ok;
}

}

int main() {
    const bool ok = test_width_native_round_trip() &&
        test_lane_capacity_and_trimming() &&
        test_int32_packet_round_trip() &&
        test_multiblock_offsets_and_output_layout() &&
        test_malformed_packets_reject_atomically() &&
        test_original_k_masks() &&
        test_overlapping_lane_k_content() &&
        test_final_group_payload() &&
        test_partition_uses_packed_group_rows() &&
        test_group_row_padding();
    if (ok) {
        std::puts("PASS: width-native SRMD packet contract");
    }
    return ok ? 0 : 1;
}
