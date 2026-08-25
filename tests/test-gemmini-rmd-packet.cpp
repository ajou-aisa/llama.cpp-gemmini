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
    return left.block_id == right.block_id &&
        left.global_k_begin == right.global_k_begin &&
        left.compact_k_count == right.compact_k_count &&
        left.padded_k_count == right.padded_k_count &&
        left.active_lane_mask == right.active_lane_mask &&
        left.active_lane_count == right.active_lane_count &&
        left.lane_ids == right.lane_ids &&
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
                   p4->digit_bits == 4 && p4->lane_capacity == 8 &&
                   p4->digit_storage == rmd::DigitStorage::packed_signed_int4 &&
                   p4->int4_packing == rmd::Int4Packing::adjacent_low_nibble_first &&
                   p4->blocks.front().active_lane_count == 1,
               "W4 packet metadata and active-lane trimming") && ok;
    ok = check(p8->digit_bits == 8 && p8->lane_capacity == 4 &&
                   p8->digit_storage == rmd::DigitStorage::signed_int8 &&
                   p8->int4_packing == rmd::Int4Packing::none,
               "W8 packet metadata") && ok;
    ok = check(p16->digit_bits == 16 && p16->lane_capacity == 2 &&
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

    ok = check(p4->stacked_activation.packed_int4.size() >= w4_literal.size() &&
                   std::equal(w4_literal.begin(), w4_literal.end(),
                              p4->stacked_activation.packed_int4.begin()),
               "signed INT4 values map to adjacent native packet bytes") && ok;

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
        {4,  rmd::kSigned21Min + 1, 8, 0, 5},
        {8,  (int32_t{1} << 16) + 1, 4, 0, 2},
        {16, (int32_t{1} << 16) + 1, 2, 0, 1},
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

        rmd::RmdStripeBuilder envelope_builder;
        envelope_builder.reset(20, 0, 1, 1, 1, test.bits);
        const bool envelope_added =
            envelope_builder.add_residual(0, 0, rmd::kSigned21Max);
        const rmd::StripePacketHandle envelope = envelope_builder.finish();
        const uint8_t expected_span = test.bits == 4 ? 6 : test.bits == 8 ? 3 : 2;
        ok = check(envelope_added && envelope != nullptr &&
                       envelope->blocks.front().active_lane_count != 0 &&
                       envelope->blocks.front().lane_ids[
                           envelope->blocks.front().active_lane_count - 1] + 1 ==
                           expected_span,
                   "signed-21 envelope spans the expected width-native radix lanes") && ok;
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

    rmd::RmdStripeBuilder overflow_builder;
    overflow_builder.reset(23, 0, 1, 1, 1, 4);
    ok = check(!overflow_builder.add_residual(0, 0, rmd::kSigned21Max + 1) &&
                   overflow_builder.status() == rmd::RmdStatus::residual_too_wide &&
                   overflow_builder.finish() == nullptr,
               "residual envelope overflow emits no packet") && ok;

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

}

int main() {
    const bool ok = test_width_native_round_trip() &&
        test_lane_capacity_and_trimming() &&
        test_multiblock_offsets_and_output_layout() &&
        test_malformed_packets_reject_atomically();
    if (ok) {
        std::puts("PASS: width-native SRMD packet contract");
    }
    return ok ? 0 : 1;
}
