#include "../ggml/src/ggml-gemmini/residual/direct/direct-executor.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-reference.hpp"
#include "../ggml/src/ggml-gemmini/residual/residual-capture.hpp"
#include "../ggml/src/ggml-gemmini/quants/common/weight_reader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace ggml::gemmini::rmd;

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

int64_t independently_compose(const NativeBalancedDigits & digits) {
    int64_t value = 0;
    int64_t place = 1;
    for (uint8_t lane = 0; lane < digits.active_lane_count; ++lane) {
        value += static_cast<int64_t>(digits.digits[lane]) * place;
        place *= digits.radix;
    }
    return value;
}

bool matches_division_oracle(int32_t residual, const NativeBalancedDigits & digits) {
    int64_t remaining = residual;
    for (size_t lane = 0; lane < digits.digits.size(); ++lane) {
        int64_t digit = remaining % digits.radix;
        if (digit >= static_cast<int64_t>(digits.radix / 2)) digit -= digits.radix;
        if (digit < -static_cast<int64_t>(digits.radix / 2)) digit += digits.radix;
        if (digits.digits[lane] != digit) return false;
        remaining = (remaining - digit) / digits.radix;
    }
    return remaining == 0 && independently_compose(digits) == residual;
}

bool test_width_native_radix_happy_boundaries() {
    struct WidthCase {
        uint8_t bits;
        uint32_t radix;
        uint8_t capacity;
        int32_t digit_min;
        int32_t digit_max;
    };
    constexpr std::array<WidthCase, 3> widths = {{
        {4, 16, 9, -8, 7},
        {8, 256, 5, -128, 127},
        {16, 65536, 3, -32768, 32767},
    }};
    constexpr std::array<int32_t, 28> values = {
        std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min() + 1,
        -16777217, -(int32_t{1} << 20) - 1, -(int32_t{1} << 20),
        -65537, -32769, -129, -128, -9, -8, -1, 0, 1, 7, 8,
        127, 128, 255, 256, 32768,
        (int32_t{1} << 20) - 1, int32_t{1} << 20, 16777216, 2139062143, 2139062144,
        std::numeric_limits<int32_t>::max() - 1, std::numeric_limits<int32_t>::max(),
    };

    NativeBalancedDigits zero{};
    zero.digits.fill(1);
    zero.active_lane_count = balanced_radix_contract(8).lane_capacity;
    bool ok = check(decompose_balanced_radix(0, 8, zero) == RmdStatus::success &&
                        zero.active_lane_count == 0 &&
                        zero.digits == std::array<int32_t, kMaxNativeRadixLanes>{},
                    "zero decomposition clears all reused digits and the active lane count");
    for (const WidthCase & width : widths) {
        const BalancedRadixContract contract = balanced_radix_contract(width.bits);
        ok = check(contract.radix == width.radix && contract.lane_capacity == width.capacity &&
                       contract.digit_min == width.digit_min &&
                       contract.digit_max == width.digit_max,
                   "width selects native radix, digit range, and lane capacity") && ok;
        for (const int32_t value : values) {
            NativeBalancedDigits digits{};
            const RmdStatus status = decompose_balanced_radix(value, width.bits, digits);
            bool value_ok = check(status == RmdStatus::success,
                                  "full int32 boundary decomposes");
            value_ok = check(digits.radix == width.radix &&
                                 digits.lane_capacity == width.capacity &&
                                 digits.active_lane_count <= width.capacity,
                             "decomposition records the selected width contract") && value_ok;
            for (uint8_t lane = 0; lane < width.capacity; ++lane) {
                value_ok = check(digits.digits[lane] >= width.digit_min &&
                                     digits.digits[lane] <= width.digit_max,
                                 "every native digit stays in its signed range") && value_ok;
            }
            for (uint8_t lane = digits.active_lane_count; lane < width.capacity; ++lane) {
                value_ok = check(digits.digits[lane] == 0,
                                 "inactive trailing lanes are zero") && value_ok;
            }
            uint8_t expected_active = 0;
            for (uint8_t lane = 0; lane < width.capacity; ++lane) {
                if (digits.digits[lane] != 0) {
                    expected_active = static_cast<uint8_t>(lane + 1);
                }
            }
            value_ok = check(digits.active_lane_count == expected_active,
                             "active lanes trim only trailing zero places") && value_ok;
            value_ok = check(matches_division_oracle(value, digits),
                             "independent division oracle agrees on every signed digit") && value_ok;
            int64_t composed = std::numeric_limits<int64_t>::min();
            value_ok = check(compose_balanced_radix(digits, composed) == RmdStatus::success &&
                                 composed == value,
                             "production recomposition matches the boundary") && value_ok;
            ok = value_ok && ok;
        }

        int64_t place = 1;
        for (uint8_t lane = 0; lane + 1 < width.capacity; ++lane) {
            const int64_t boundary = static_cast<int64_t>(width.radix / 2) * place;
            if (boundary > std::numeric_limits<int32_t>::max()) {
                break;
            }
            for (const int64_t candidate : {
                     boundary - 1, boundary, boundary + 1,
                     -boundary + 1, -boundary, -boundary - 1,
                 }) {
                if (candidate < std::numeric_limits<int32_t>::min() ||
                    candidate > std::numeric_limits<int32_t>::max()) {
                    continue;
                }
                NativeBalancedDigits digits{};
                ok = check(decompose_balanced_radix(static_cast<int32_t>(candidate),
                                                    width.bits, digits) == RmdStatus::success &&
                               matches_division_oracle(static_cast<int32_t>(candidate), digits),
                           "every native digit boundary carries and borrows exactly") && ok;
            }

            NativeBalancedDigits carry{};
            ok = check(decompose_balanced_radix(static_cast<int32_t>(boundary),
                                                width.bits, carry) == RmdStatus::success &&
                           carry.active_lane_count == lane + 2 &&
                           carry.digits[lane] == width.digit_min &&
                           carry.digits[lane + 1] == 1,
                       "positive half-radix carries into the next native lane") && ok;
            place *= width.radix;
        }

        int64_t without_top_carry_max = 0;
        place = 1;
        for (uint8_t lane = 0; lane + 1 < width.capacity; ++lane) {
            without_top_carry_max += width.digit_max * place;
            place *= width.radix;
        }
        for (int64_t candidate : {without_top_carry_max - 1, without_top_carry_max,
                                  without_top_carry_max + 1, without_top_carry_max + 2}) {
            NativeBalancedDigits digits{};
            ok = check(decompose_balanced_radix(static_cast<int32_t>(candidate), width.bits,
                                                digits) == RmdStatus::success &&
                           matches_division_oracle(static_cast<int32_t>(candidate), digits) &&
                           digits.digits[width.capacity - 1] ==
                               (candidate > without_top_carry_max ? 1 : 0),
                       "first top carry uses the extra ordinary radix lane") && ok;
        }

        uint32_t state = 0x12345678u;
        for (size_t sample = 0; sample < 100000; ++sample) {
            state = state * 1664525u + 1013904223u;
            const int64_t signed_value = state <= uint32_t{INT32_MAX} ?
                static_cast<int64_t>(state) : static_cast<int64_t>(state) - (int64_t{1} << 32);
            NativeBalancedDigits digits{};
            if (!check(decompose_balanced_radix(static_cast<int32_t>(signed_value), width.bits,
                                                digits) == RmdStatus::success &&
                           matches_division_oracle(static_cast<int32_t>(signed_value), digits),
                       "full-range samples match independent division without signed overflow")) {
                return false;
            }
        }
    }
    return ok;
}

bool test_width_native_compose_and_expand() {
    struct WidthCase {
        uint8_t bits;
        int32_t residual;
    };
    constexpr std::array<WidthCase, 10> cases = {{
        {4, std::numeric_limits<int32_t>::min()},
        {4, std::numeric_limits<int32_t>::max()},
        {4, 16},
        {8, std::numeric_limits<int32_t>::min()},
        {8, std::numeric_limits<int32_t>::max()},
        {8, 65537},
        {8, 256},
        {16, std::numeric_limits<int32_t>::min()},
        {16, std::numeric_limits<int32_t>::max()},
        {16, 65536},
    }};
    constexpr std::array<int64_t, 2> weights = {3, -257};
    constexpr int64_t block_scale = 5;

    bool ok = true;
    for (const WidthCase & test : cases) {
        RmdStripeBuilder builder;
        builder.reset(31, 0, 1, 1, weights.size(), test.bits);
        if (!check(builder.add_residual(0, 0, test.residual),
                   "width-native compose residual accepted")) {
            return false;
        }
        const StripePacketHandle packet = builder.finish();
        if (!check(packet != nullptr && packet->blocks.size() == 1,
                   "width-native compose packet built")) {
            return false;
        }

        CompressedOutput compressed;
        compressed.j_padded = packet->j_padded;
        compressed.values.assign(packet->total_output_values, 0);
        const BlockDescriptor & block = packet->blocks.front();
        for (uint8_t lane_position = 0;
             lane_position < block.active_lane_count; ++lane_position) {
            int32_t digit = 0;
            if (!check(read_packet_digit(*packet, block, lane_position, 0, 0,
                                         digit) == RmdStatus::success,
                       "typed packet digit feeds compose oracle")) {
                return false;
            }
            const size_t lane_base = block.output_value_offset +
                static_cast<size_t>(lane_position) * block.lane_stride_values;
            for (size_t j = 0; j < weights.size(); ++j) {
                compressed.values[lane_base + j] =
                    static_cast<int64_t>(digit) * weights[j] * block_scale;
            }
        }

        Correction correction = BlockScaledInt64Correction{{71, 73, 79}};
        const RmdStatus status = compose_rmd_output(*packet, compressed, correction);
        const auto * values = std::get_if<BlockScaledInt64Correction>(&correction);
        ok = check(status == RmdStatus::success && values != nullptr &&
                       values->values.size() == weights.size() &&
                       values->values[0] ==
                           static_cast<int64_t>(test.residual) * weights[0] * block_scale &&
                       values->values[1] ==
                           static_cast<int64_t>(test.residual) * weights[1] * block_scale,
                   "checked radix composition matches independent residual oracle") && ok;

        std::vector<int32_t> plane;
        expand_packets_to_plane({packet}, 1, 1, plane);
        ok = check(plane.size() == 1 && plane[0] == test.residual,
                   "typed packet expansion reconstructs the original residual") && ok;

        StripePacket malformed_packet = *packet;
        if (malformed_packet.digit_storage == DigitStorage::packed_signed_int4) {
            malformed_packet.stacked_activation.packed_int4.pop_back();
        } else if (malformed_packet.digit_storage == DigitStorage::signed_int8) {
            malformed_packet.stacked_activation.signed_int8.pop_back();
        } else {
            malformed_packet.stacked_activation.signed_int16.pop_back();
        }
        plane = {121, 123};
        expand_packets_to_plane(
            {std::make_shared<const StripePacket>(std::move(malformed_packet))},
            1, 1, plane);
        ok = check(plane == std::vector<int32_t>({121, 123}),
                   "malformed typed expansion leaves caller plane unchanged") && ok;

        CompressedOutput malformed = compressed;
        malformed.values.pop_back();
        const Correction sentinel = BlockScaledInt64Correction{{91, 92, 93}};
        correction = sentinel;
        const auto malformed_status = compose_rmd_output(*packet, malformed, correction);
        const auto * malformed_values =
            std::get_if<BlockScaledInt64Correction>(&correction);
        ok = check(malformed_status == RmdStatus::invalid_arguments &&
                       malformed_values != nullptr &&
                       malformed_values->values ==
                           std::get<BlockScaledInt64Correction>(sentinel).values,
                   "malformed compressed geometry leaves correction unchanged") && ok;
    }

    for (const uint8_t bits : {uint8_t{4}, uint8_t{8}, uint8_t{16}}) {
        const BalancedRadixContract contract = balanced_radix_contract(bits);
        RmdStripeBuilder builder;
        builder.reset(37, 0, 1, 1, 1, bits);
        builder.add_residual(0, 0, static_cast<int32_t>(contract.radix / 2));
        const StripePacketHandle packet = builder.finish();
        if (!check(packet != nullptr && packet->blocks.front().active_lane_count == 2,
                   "overflow compose packet spans two native lanes")) {
            return false;
        }
        CompressedOutput compressed;
        compressed.j_padded = packet->j_padded;
        compressed.values.assign(packet->total_output_values, 0);
        const BlockDescriptor & block = packet->blocks.front();
        compressed.values[block.output_value_offset] =
            std::numeric_limits<int64_t>::min() + 1;
        compressed.values[block.output_value_offset + block.lane_stride_values] =
            int64_t{1} << (63 - bits);
        Correction correction = BlockScaledInt64Correction{{101, 103}};
        const auto cancellation_status = compose_rmd_output(*packet, compressed, correction);
        const auto * cancellation_values =
            std::get_if<BlockScaledInt64Correction>(&correction);
        ok = check(cancellation_status == RmdStatus::success &&
                       cancellation_values != nullptr &&
                       cancellation_values->values == std::vector<int64_t>({1}),
                   "wide radix contribution cancels before final int64 narrowing") && ok;

        for (uint8_t lane_position = 0; lane_position < 2; ++lane_position) {
            compressed.values[block.output_value_offset +
                static_cast<size_t>(lane_position) * block.lane_stride_values] =
                    std::numeric_limits<int64_t>::max();
        }
        const Correction sentinel = BlockScaledInt64Correction{{101, 103}};
        correction = sentinel;
        const auto overflow_status = compose_rmd_output(*packet, compressed, correction);
        const auto * overflow_values =
            std::get_if<BlockScaledInt64Correction>(&correction);
        ok = check(overflow_status == RmdStatus::overflow &&
                       overflow_values != nullptr &&
                       overflow_values->values ==
                           std::get<BlockScaledInt64Correction>(sentinel).values,
                   "native-radix overflow leaves correction unchanged") && ok;

        builder.reset(38, 0, 1, kBlockSize + 1, 1, bits);
        if (!check(builder.add_residual(0, 0, static_cast<int32_t>(contract.radix)) &&
                       builder.add_residual(0, kBlockSize,
                                            -static_cast<int32_t>(contract.radix)),
                   "opposite higher-lane residuals accepted across blocks")) {
            return false;
        }
        const StripePacketHandle cancellation_packet = builder.finish();
        if (!check(cancellation_packet != nullptr && cancellation_packet->blocks.size() == 2,
                   "cross-block cancellation packet built")) {
            return false;
        }
        compressed.j_padded = cancellation_packet->j_padded;
        compressed.values.assign(cancellation_packet->total_output_values, 0);
        compressed.values[cancellation_packet->blocks[0].output_value_offset] = int64_t{1} << 62;
        compressed.values[cancellation_packet->blocks[1].output_value_offset] = -(int64_t{1} << 62);
        correction = sentinel;
        const auto block_cancellation_status =
            compose_rmd_output(*cancellation_packet, compressed, correction);
        const auto * block_cancellation_values =
            std::get_if<BlockScaledInt64Correction>(&correction);
        ok = check(block_cancellation_status == RmdStatus::success &&
                       block_cancellation_values != nullptr &&
                       block_cancellation_values->values == std::vector<int64_t>({0}),
                   "wide block contributions cancel before final int64 narrowing") && ok;
    }
    return ok;
}

bool test_width_native_out_of_int32_compose_is_atomic() {
    constexpr std::array<uint8_t, 3> widths = {4, 8, 16};
    bool ok = true;
    for (const uint8_t bits : widths) {
        const auto contract = balanced_radix_contract(bits);
        for (const int32_t top_digit : {-1, 1}) {
            NativeBalancedDigits digits{};
            digits.radix = contract.radix;
            digits.lane_capacity = digits.active_lane_count = contract.lane_capacity;
            digits.digits[contract.lane_capacity - 1] = top_digit;
            int64_t output = 123456789;
            ok = check(compose_balanced_radix(digits, output) ==
                           RmdStatus::residual_too_wide &&
                           output == 123456789,
                       "digits representing plus or minus 2^32 reject atomically") && ok;
        }
    }
    return ok;
}

bool test_width_native_malformed_compose_is_atomic() {
    constexpr std::array<uint8_t, 3> widths = {4, 8, 16};
    bool ok = true;
    auto rejects_without_output_mutation = [&](const NativeBalancedDigits & malformed,
                                               const char * message) {
        constexpr int64_t sentinel = INT64_C(0x123456789abcdef);
        int64_t output = sentinel;
        return check(compose_balanced_radix(malformed, output) ==
                         RmdStatus::invalid_arguments &&
                         output == sentinel,
                     message);
    };

    for (const uint8_t bits : widths) {
        NativeBalancedDigits canonical{};
        if (!check(decompose_balanced_radix(1, bits, canonical) == RmdStatus::success,
                   "malformed-compose fixture decomposes")) {
            return false;
        }

        NativeBalancedDigits untrimmed = canonical;
        ++untrimmed.active_lane_count;
        ok = rejects_without_output_mutation(
                 untrimmed, "untrimmed active-lane metadata rejects atomically") && ok;

        NativeBalancedDigits missing_active = canonical;
        missing_active.active_lane_count = 0;
        ok = rejects_without_output_mutation(
                 missing_active, "nonzero logical digit requires active-lane metadata") && ok;

        NativeBalancedDigits zero_untrimmed{};
        if (!check(decompose_balanced_radix(0, bits, zero_untrimmed) == RmdStatus::success,
                   "zero malformed-compose fixture decomposes")) {
            return false;
        }
        zero_untrimmed.active_lane_count = 1;
        ok = rejects_without_output_mutation(
                 zero_untrimmed, "zero decomposition cannot claim an active lane") && ok;

        if (canonical.lane_capacity < canonical.digits.size()) {
            NativeBalancedDigits physical_tail = canonical;
            physical_tail.digits[canonical.lane_capacity] = 1;
            ok = rejects_without_output_mutation(
                     physical_tail,
                     "nonzero physical tail beyond logical capacity rejects atomically") && ok;
        }
    }
    return ok;
}

bool test_top_carry_lane_preserves_sparse_mask() {
    bool ok = true;
    for (uint8_t bits : {uint8_t{4}, uint8_t{8}, uint8_t{16}}) {
        RmdStripeBuilder builder;
        builder.reset(0, 0, 1, kBlockSize, 1, bits);
        if (!check(builder.add_residual(0, 31, std::numeric_limits<int32_t>::max()),
                   "builder accepts full int32 residual at K31")) return false;
        const auto packet = builder.finish();
        if (!check(packet != nullptr && packet->blocks.size() == 1,
                   "top carry packet builds")) return false;
        const auto & block = packet->blocks.front();
        const uint8_t top_lane = 32 / bits;
        const uint16_t expected_mask = (1u << 0) | (1u << (top_lane - 1)) | (1u << top_lane);
        ok = check(block.active_lane_mask == expected_mask && block.active_lane_count == 3 &&
                       block.lane_ids[0] == 0 && block.lane_ids[1] == top_lane - 1 &&
                       block.lane_ids[2] == top_lane &&
                       block.lane_k_masks[top_lane] == (uint32_t{1} << 31) &&
                       validate_packet(*packet) == RmdStatus::success,
                   "sparse mask retains original top lane and original K31") && ok;
    }
    return ok;
}

bool test_EXPLICIT_BLOCK_ID_DIM_PADDING() {
    RmdStripeBuilder builder;
    builder.reset(7, 9, 3, 64, 17, 8);
    if (!builder.add_residual(0, 31, 1) ||
        !builder.add_residual(1, 32, 65536) ||
        !builder.add_residual(2, 33, 1)) {
        return check(false, "boundary residuals accepted");
    }
    const StripePacketHandle packet = builder.finish();
    if (!check(packet != nullptr, "boundary packet built") ||
        !check(packet->blocks.size() == 2, "block boundary stays explicit")) {
        return false;
    }

    const BlockDescriptor & first = packet->blocks[0];
    const BlockDescriptor & second = packet->blocks[1];
    bool ok = true;
    ok = check(first.block_id == 0 && first.global_k_begin == 0, "first block id") && ok;
    ok = check(second.block_id == 1 && second.global_k_begin == kBlockSize, "second block id") && ok;
    ok = check(second.active_lane_mask == 0x05 && second.active_lane_count == 2,
               "lane gap remains sparse") && ok;
    ok = check(second.lane_ids[0] == 0 && second.lane_ids[1] == 2,
               "lane ids preserve radix places") && ok;
    ok = check(first.rows_padded == kArrayDim && second.rows_padded == kArrayDim,
               "rows use DIM padding") && ok;
    ok = check(first.padded_k_count == kArrayDim && second.padded_k_count == kArrayDim,
               "each block uses independent DIM padding") && ok;
    ok = check(packet->j_padded == align_up(packet->logical_j, kArrayDim),
               "output columns use DIM padding") && ok;
    ok = check(validate_packet(*packet) == RmdStatus::success, "built packet validates") && ok;

    StripePacket malformed = *packet;
    malformed.stacked_activation.signed_int8[
        first.activation_offset + first.compact_k_count] = 1;
    ok = check(validate_packet(malformed) == RmdStatus::invalid_packet,
               "nonzero K padding rejected") && ok;
    malformed = *packet;
    malformed.stacked_activation.signed_int8[
        first.activation_offset + packet->row_count * first.padded_k_count] = 1;
    return check(validate_packet(malformed) == RmdStatus::invalid_packet,
                 "nonzero row padding rejected") && ok;
}

bool test_empty_residual_is_empty_success() {
    RmdStripeBuilder builder;
    builder.reset(0, 0, 1, 1, 1);
    return check(builder.add_residual(0, 0, 0), "zero residual accepted") &&
        check(builder.empty(), "zero residual emits no entries") &&
        check(builder.finish() == nullptr, "empty residual emits no packet") &&
        check(builder.status() == RmdStatus::success, "empty residual is successful");
}

bool test_padding_overflow_fails() {
    RmdStripeBuilder builder;
    builder.reset(0, 0, 1, 1, std::numeric_limits<size_t>::max());
    if (!builder.add_residual(0, 0, 1)) {
        return check(false, "overflow fixture residual accepted");
    }
    return check(builder.finish() == nullptr, "padding overflow emits no packet") &&
        check(builder.status() == RmdStatus::overflow, "padding overflow is explicit");
}

enum class WeightFamily : uint8_t {
    H0,
    H1,
    HP1,
};

struct WeightCapabilityFixture {
    ggml_gemmini_args_t args{};
    block_q4_h0 q4_h0{};
    block_q4_h1 q4_h1{};
    block_q4_hp1 q4_hp1{};
    block_q8_0 q8_h0{};
    block_q8_h1 q8_h1{};
    block_q8_hp1 q8_hp1{};
    block_q16_h0 q16_h0{};
    block_q16_h1 q16_h1{};
    block_q16_hp1 q16_hp1{};

    WeightCapabilityFixture(uint8_t bits, WeightFamily family) {
        using Format = ggml_gemmini_args_t::im2p_weight_format_t;
        args.J = 1;
        args.K = 32;
        args.block_size_k = 32;
        args.native_block_count = 1;
        args.native_blocks_per_row = 1;

        std::fill(std::begin(q4_h0.qs), std::end(q4_h0.qs), uint8_t{0x88});
        std::fill(std::begin(q4_h1.qs), std::end(q4_h1.qs), uint8_t{0x88});
        std::fill(std::begin(q4_hp1.qs), std::end(q4_hp1.qs), uint8_t{0x88});
        q4_h0.d = ggml_fp32_to_fp16(0.5f);
        q8_h0.d = ggml_fp32_to_fp16(0.5f);
        q16_h0.d = ggml_fp32_to_fp16(0.5f);
        q4_h1.c_b = q8_h1.c_b = q16_h1.c_b = 2;
        q4_h1.R = q8_h1.R = q16_h1.R = 3;
        q4_h1.s_rf = q8_h1.s_rf = q16_h1.s_rf = 0.25f;
        q4_hp1.m = q8_hp1.m = q16_hp1.m = 2;
        q4_hp1.channel_scale = q8_hp1.channel_scale = q16_hp1.channel_scale = 0.25f;

        if (bits == 4) {
            if (family == WeightFamily::H0) {
                args.weight_format = Format::q4_h0;
                args.q4_h0_blocks = &q4_h0;
                args.native_weight_bytes = sizeof(q4_h0);
            } else if (family == WeightFamily::H1) {
                args.weight_format = Format::q4_h1;
                args.q4_h1_blocks = &q4_h1;
                args.native_weight_bytes = sizeof(q4_h1);
            } else {
                args.weight_format = Format::q4_hp1;
                args.q4_hp1_blocks = &q4_hp1;
                args.native_weight_bytes = sizeof(q4_hp1);
            }
        } else if (bits == 8) {
            if (family == WeightFamily::H0) {
                args.weight_format = Format::q8_h0;
                args.B_blocks = &q8_h0;
                args.blocks_J = 1;
                args.blocks_K = 1;
                args.native_weight_bytes = sizeof(q8_h0);
            } else if (family == WeightFamily::H1) {
                args.weight_format = Format::q8_h1;
                args.q8_h1_blocks = &q8_h1;
                args.q8_h1_block_count = 1;
                args.q8_h1_rows = 1;
                args.blocks_per_row = 1;
                args.native_weight_bytes = sizeof(q8_h1);
            } else {
                args.weight_format = Format::q8_hp1;
                args.q8_hp1_blocks = &q8_hp1;
                args.q8_hp1_block_count = 1;
                args.q8_hp1_blocks_per_row = 1;
                args.native_weight_bytes = sizeof(q8_hp1);
            }
        } else if (family == WeightFamily::H0) {
            args.weight_format = Format::q16_h0;
            args.q16_h0_blocks = &q16_h0;
            args.native_weight_bytes = sizeof(q16_h0);
        } else if (family == WeightFamily::H1) {
            args.weight_format = Format::q16_h1;
            args.q16_h1_blocks = &q16_h1;
            args.native_weight_bytes = sizeof(q16_h1);
        } else {
            args.weight_format = Format::q16_hp1;
            args.q16_hp1_blocks = &q16_hp1;
            args.native_weight_bytes = sizeof(q16_hp1);
        }
    }
};

bool test_weight_capability_happy_table() {
    namespace wreader = ggml::gemmini::quants::wreader;
    namespace wroute = ggml::gemmini::quants::wroute;
    struct CapabilityCase {
        uint8_t bits;
        WeightFamily family;
        wroute::WeightRouteKind route;
        wroute::WeightScaleDomain domain;
        bool compact;
    };
    constexpr std::array<CapabilityCase, 9> cases = {{
        {4,  WeightFamily::H0,  wroute::WeightRouteKind::H0,  wroute::WeightScaleDomain::FloatingBlock, false},
        {4,  WeightFamily::H1,  wroute::WeightRouteKind::H1,  wroute::WeightScaleDomain::IntegerBlockTimesColumn, true},
        {4,  WeightFamily::HP1, wroute::WeightRouteKind::HP1, wroute::WeightScaleDomain::IntegerBlockTimesColumn, true},
        {8,  WeightFamily::H0,  wroute::WeightRouteKind::H0,  wroute::WeightScaleDomain::FloatingBlock, false},
        {8,  WeightFamily::H1,  wroute::WeightRouteKind::H1,  wroute::WeightScaleDomain::IntegerBlockTimesColumn, true},
        {8,  WeightFamily::HP1, wroute::WeightRouteKind::HP1, wroute::WeightScaleDomain::IntegerBlockTimesColumn, true},
        {16, WeightFamily::H0,  wroute::WeightRouteKind::H0,  wroute::WeightScaleDomain::FloatingBlock, false},
        {16, WeightFamily::H1,  wroute::WeightRouteKind::H1,  wroute::WeightScaleDomain::IntegerBlockTimesColumn, true},
        {16, WeightFamily::HP1, wroute::WeightRouteKind::HP1, wroute::WeightScaleDomain::IntegerBlockTimesColumn, true},
    }};

    bool ok = true;
    for (const CapabilityCase & test : cases) {
        WeightCapabilityFixture fixture(test.bits, test.family);
        const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
            fixture.args, wroute::WeightScaleInfoMode::Residual);
        const wreader::WeightCodeResult zero = wreader::read_code(fixture.args, plan, 0, 16);
        ok = check(plan.valid && plan.route == test.route &&
                       plan.weight_bits == test.bits && plan.scale_domain == test.domain,
                   "residual route capability resolves by family and width") && ok;
        ok = check(zero.status == wreader::WeightReaderStatus::Success && zero.value == 0,
                   "RMD capability uses the shared signed-code reader") && ok;
        ok = check(wroute::weight_route_status(plan, wroute::WeightExecutionPath::CpuDirect) ==
                       wroute::WeightRouteStatus::Success,
                   "all supported residual families allow CPU-direct") && ok;
        ok = check(
            wroute::weight_route_status(plan, wroute::WeightExecutionPath::Compact) ==
                (test.compact ? wroute::WeightRouteStatus::Success :
                                wroute::WeightRouteStatus::UnsupportedExecution),
            "compact capability follows the scale domain") && ok;
    }
    return ok;
}

bool test_weight_capability_failure_table() {
    namespace wroute = ggml::gemmini::quants::wroute;
    bool ok = true;
    for (uint8_t bits : {uint8_t{4}, uint8_t{8}, uint8_t{16}}) {
        WeightCapabilityFixture fixture(bits, WeightFamily::H0);
        const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
            fixture.args, wroute::WeightScaleInfoMode::Residual);
        ok = check(wroute::weight_route_status(plan, wroute::WeightExecutionPath::Compact) ==
                       wroute::WeightRouteStatus::UnsupportedExecution,
                   "H0 compact request is rejected before packet work") && ok;
    }

    wroute::WeightRoutePlan invalid{};
    invalid.status = wroute::WeightRouteStatus::InvalidMetadata;
    ok = check(wroute::weight_route_status(invalid, wroute::WeightExecutionPath::CpuDirect) ==
                   wroute::WeightRouteStatus::InvalidMetadata,
               "invalid metadata remains a typed route failure") && ok;
    return ok;
}

namespace residual = ggml::gemmini::residual;
namespace rmd = ggml::gemmini::rmd;

const char * weight_family_name(WeightFamily family) {
    switch (family) {
        case WeightFamily::H0:  return "H0";
        case WeightFamily::H1:  return "H1";
        case WeightFamily::HP1: return "HP1";
    }
    return "unknown";
}

struct DirectOracleFixture {
    ggml_gemmini_args_t args{};
    std::array<block_q4_h0, 4> q4_h0{};
    std::array<block_q4_h1, 4> q4_h1{};
    std::array<block_q4_hp1, 4> q4_hp1{};
    std::array<block_q8_0, 4> q8_h0{};
    std::array<block_q8_h1, 4> q8_h1{};
    std::array<block_q8_hp1, 4> q8_hp1{};
    std::array<block_q16_h0, 4> q16_h0{};
    std::array<block_q16_h1, 4> q16_h1{};
    std::array<block_q16_hp1, 4> q16_hp1{};
    residual::DirectStripePayloadHandle payload;

    DirectOracleFixture(uint8_t bits, WeightFamily family) {
        using Format = ggml_gemmini_args_t::im2p_weight_format_t;
        args.I = 3;
        args.J = 2;
        args.K = 64;
        args.block_size_k = 32;
        args.native_block_count = 4;
        args.native_blocks_per_row = 2;

        auto populate_q4 = [](auto & blocks) {
            for (auto & block : blocks) {
                std::fill(std::begin(block.qs), std::end(block.qs), uint8_t{0x88});
            }
            // Block order is [j0b0, j0b1, j1b0, j1b1]. Low nibbles
            // encode K[0..15], high nibbles K[16..31].
            blocks[0].qs[0] = 0xf0; blocks[0].qs[15] = 0x78; // [-8, +7, -1]
            blocks[1].qs[0] = 0x8f; blocks[1].qs[15] = 0xe0; // [+7, -8, +6]
            blocks[2].qs[0] = 0x0f; blocks[2].qs[15] = 0xb8; // [+7, -8, +3]
            blocks[3].qs[0] = 0x80; blocks[3].qs[15] = 0x4f; // [-8, +7, -4]
        };
        populate_q4(q4_h0);
        populate_q4(q4_h1);
        populate_q4(q4_hp1);

        auto populate_q8 = [](auto & blocks) {
            blocks[0].qs[0] = -128; blocks[0].qs[16] = 127; blocks[0].qs[31] = -17;
            blocks[1].qs[0] = 127; blocks[1].qs[15] = -128; blocks[1].qs[31] = 99;
            blocks[2].qs[0] = 127; blocks[2].qs[16] = -128; blocks[2].qs[31] = 31;
            blocks[3].qs[0] = -128; blocks[3].qs[15] = 127; blocks[3].qs[31] = -64;
        };
        populate_q8(q8_h0);
        populate_q8(q8_h1);
        populate_q8(q8_hp1);

        auto populate_q16 = [](auto & blocks) {
            blocks[0].qs[0] = -32768; blocks[0].qs[16] = 32767; blocks[0].qs[31] = -257;
            blocks[1].qs[0] = 32767; blocks[1].qs[15] = -32768; blocks[1].qs[31] = 12345;
            blocks[2].qs[0] = 32767; blocks[2].qs[16] = -32768; blocks[2].qs[31] = 511;
            blocks[3].qs[0] = -32768; blocks[3].qs[15] = 32767; blocks[3].qs[31] = -16384;
        };
        populate_q16(q16_h0);
        populate_q16(q16_h1);
        populate_q16(q16_hp1);

        constexpr std::array<float, 4> h0_scales = {0.5f, 0.0f, -1.25f, 2.0f};
        constexpr std::array<uint8_t, 4> h1_codes = {1, 0, 1, 2};
        constexpr std::array<uint16_t, 4> h1_offsets = {1, 0, 2, 3};
        constexpr std::array<float, 4> h1_columns = {0.25f, 0.25f, -0.5f, -0.5f};
        constexpr std::array<int16_t, 4> hp1_exponents = {
            1, std::numeric_limits<int16_t>::min(), 2, 3,
        };
        constexpr std::array<float, 4> hp1_columns = {0.125f, 0.125f, -0.75f, -0.75f};
        for (size_t i = 0; i < 4; ++i) {
            q4_h0[i].d = ggml_fp32_to_fp16(h0_scales[i]);
            q8_h0[i].d = ggml_fp32_to_fp16(h0_scales[i]);
            q16_h0[i].d = ggml_fp32_to_fp16(h0_scales[i]);

            q4_h1[i].c_b = q8_h1[i].c_b = q16_h1[i].c_b = h1_codes[i];
            q4_h1[i].R = q8_h1[i].R = q16_h1[i].R = h1_offsets[i];
            q4_h1[i].s_rf = q8_h1[i].s_rf = q16_h1[i].s_rf = h1_columns[i];

            q4_hp1[i].m = q8_hp1[i].m = q16_hp1[i].m = hp1_exponents[i];
            q4_hp1[i].channel_scale = q8_hp1[i].channel_scale =
                q16_hp1[i].channel_scale = hp1_columns[i];
        }

        if (bits == 4) {
            if (family == WeightFamily::H0) {
                args.weight_format = Format::q4_h0;
                args.q4_h0_blocks = q4_h0.data();
                args.native_weight_bytes = sizeof(q4_h0);
            } else if (family == WeightFamily::H1) {
                args.weight_format = Format::q4_h1;
                args.q4_h1_blocks = q4_h1.data();
                args.native_weight_bytes = sizeof(q4_h1);
            } else {
                args.weight_format = Format::q4_hp1;
                args.q4_hp1_blocks = q4_hp1.data();
                args.native_weight_bytes = sizeof(q4_hp1);
            }
        } else if (bits == 8) {
            if (family == WeightFamily::H0) {
                args.weight_format = Format::q8_h0;
                args.B_blocks = q8_h0.data();
                args.blocks_J = 2;
                args.blocks_K = 2;
                args.native_weight_bytes = sizeof(q8_h0);
            } else if (family == WeightFamily::H1) {
                args.weight_format = Format::q8_h1;
                args.q8_h1_blocks = q8_h1.data();
                args.q8_h1_block_count = q8_h1.size();
                args.q8_h1_rows = 2;
                args.blocks_per_row = 2;
                args.native_weight_bytes = sizeof(q8_h1);
            } else {
                args.weight_format = Format::q8_hp1;
                args.q8_hp1_blocks = q8_hp1.data();
                args.q8_hp1_block_count = q8_hp1.size();
                args.q8_hp1_blocks_per_row = 2;
                args.native_weight_bytes = sizeof(q8_hp1);
            }
        } else if (family == WeightFamily::H0) {
            args.weight_format = Format::q16_h0;
            args.q16_h0_blocks = q16_h0.data();
            args.native_weight_bytes = sizeof(q16_h0);
        } else if (family == WeightFamily::H1) {
            args.weight_format = Format::q16_h1;
            args.q16_h1_blocks = q16_h1.data();
            args.native_weight_bytes = sizeof(q16_h1);
        } else {
            args.weight_format = Format::q16_hp1;
            args.q16_hp1_blocks = q16_hp1.data();
            args.native_weight_bytes = sizeof(q16_hp1);
        }

        residual::DirectStripeBuilder builder;
        builder.reset(19, 7, 3, 64, 2);
        constexpr std::array<residual::ResidualEvent, 12> events = {{
            {0, 0, 3}, {0, 16, -2}, {0, 31, 5},
            {0, 32, -4}, {0, 47, 7}, {0, 63, -6},
            {2, 0, -9}, {2, 16, 4}, {2, 31, -3},
            {2, 32, 8}, {2, 47, -5}, {2, 63, 2},
        }};
        for (const residual::ResidualEvent & event : events) {
            if (!builder.add_residual(event.local_row, event.original_k, event.residual)) {
                return;
            }
        }
        payload = builder.finish();
    }
};

struct DirectOracleCase {
    uint8_t bits;
    WeightFamily family;
    std::array<int64_t, 6> integer_expected{};
    std::array<double, 6> floating_expected{};
};

constexpr std::array<DirectOracleCase, 9> kDirectOracleCases = {{
    {4, WeightFamily::H0,  {}, {-21.5, 145.0, 0.0, 0.0, 51.5, -84.0}},
    {4, WeightFamily::H1,  {-86, 681, 0, 0, 206, -847}, {}},
    {4, WeightFamily::HP1, {-86, 1048, 0, 0, 206, -1272}, {}},
    {8, WeightFamily::H0,  {}, {-361.5, 2580.0, 0.0, 0.0, 855.5, -1389.0}},
    {8, WeightFamily::H1,  {-1446, 11301, 0, 0, 3422, -14179}, {}},
    {8, WeightFamily::HP1, {-1446, 17448, 0, 0, 3422, -21288}, {}},
    {16, WeightFamily::H0, {}, {-82561.5, 709500.0, 0.0, 0.0, 213375.5, -383109.0}},
    {16, WeightFamily::H1, {-330246, 2792901, 0, 0, 853502, -3576259}, {}},
    {16, WeightFamily::HP1, {-330246, 4335528, 0, 0, 853502, -5380008}, {}},
}};

template <typename T, size_t N>
bool values_match(const std::vector<T> & actual, const std::array<T, N> & expected) {
    return actual.size() == expected.size() &&
        std::equal(actual.begin(), actual.end(), expected.begin());
}

bool direct_outputs_match(const rmd::DirectOutput & lhs, const rmd::DirectOutput & rhs) {
    if (lhs.index() != rhs.index()) return false;
    if (const auto * left = std::get_if<rmd::BlockScaledInt64Correction>(&lhs)) {
        const auto * right = std::get_if<rmd::BlockScaledInt64Correction>(&rhs);
        return right != nullptr && left->values == right->values;
    }
    if (const auto * left = std::get_if<rmd::PreScaledFloat64Correction>(&lhs)) {
        const auto * right = std::get_if<rmd::PreScaledFloat64Correction>(&rhs);
        return right != nullptr && left->values == right->values;
    }
    const auto * left = std::get_if<rmd::FullyScaledFloat64Correction>(&lhs);
    const auto * right = std::get_if<rmd::FullyScaledFloat64Correction>(&rhs);
    return left != nullptr && right != nullptr && left->values == right->values;
}

bool test_block_direct_weight_families() {
    bool ok = true;
    for (WeightFamily family : {WeightFamily::H0, WeightFamily::H1, WeightFamily::HP1}) {
        DirectOracleFixture fixture(8, family);
        if (!check(fixture.payload != nullptr, "BLOCK direct family fixture builds")) {
            return false;
        }
        auto & metadata = fixture.args.act_quant.storage().emplace<
            ggml::gemmini::quants::act::block::Meta>();
        metadata.rows = fixture.payload->row_begin + fixture.payload->row_count;
        metadata.cols = fixture.args.K;
        metadata.scales.assign(metadata.rows * 2, 1.0f);
        metadata.scales[(fixture.payload->row_begin + 0) * 2 + 0] = 0.5f;
        metadata.scales[(fixture.payload->row_begin + 0) * 2 + 1] = 8.0f;
        metadata.scales[(fixture.payload->row_begin + 2) * 2 + 0] = 2.0f;
        metadata.scales[(fixture.payload->row_begin + 2) * 2 + 1] = 0.25f;

        std::vector<double> expected(fixture.payload->row_count * fixture.args.J, 0.0);
        for (const residual::ResidualEvent & event : fixture.payload->events) {
            const size_t block = event.original_k / kBlockSize;
            const size_t global_row = fixture.payload->row_begin + event.local_row;
            const double activation_scale = metadata.scales[global_row * 2 + block];
            for (size_t j = 0; j < fixture.args.J; ++j) {
                const size_t index = j * 2 + block;
                const size_t k = event.original_k % kBlockSize;
                int32_t code;
                double scale;
                if (family == WeightFamily::H0) {
                    code = fixture.q8_h0[index].qs[k];
                    scale = ggml_fp16_to_fp32(fixture.q8_h0[index].d);
                } else if (family == WeightFamily::H1) {
                    const auto & weight = fixture.q8_h1[index];
                    code = weight.qs[k];
                    scale = static_cast<double>(weight.c_b + weight.R) * weight.s_rf;
                } else {
                    const auto & weight = fixture.q8_hp1[index];
                    code = weight.qs[k];
                    scale = weight.m == std::numeric_limits<int16_t>::min() ? 0.0 :
                        std::ldexp(static_cast<double>(weight.channel_scale), weight.m);
                }
                expected[event.local_row * fixture.args.J + j] +=
                    static_cast<double>(event.residual) *
                    static_cast<double>(code) * scale * activation_scale;
            }
        }

        Correction actual = BlockScaledInt64Correction{{777}};
        const RmdStatus status = residual::execute_direct_stripe(
            fixture.args, *fixture.payload, actual);
        const auto * fully_scaled = std::get_if<FullyScaledFloat64Correction>(&actual);
        ok = check(status == RmdStatus::success && fully_scaled != nullptr &&
                       fully_scaled->values == expected,
                   "BLOCK H0/H1/HP1 direct matches independent per-block reference") && ok;
    }
    if (ok) std::puts("BLOCK_DIRECT_FAMILIES H0=pass H1=pass HP1=pass scales=differ");
    return ok;
}

bool test_direct_oracle_happy_matrix() {
    bool ok = true;
    for (const DirectOracleCase & test : kDirectOracleCases) {
        DirectOracleFixture fixture(test.bits, test.family);
        rmd::DirectOutput actual = rmd::PreScaledFloat64Correction{{91.5, -27.25}};
        residual::DirectExecutionMetrics metrics{};
        metrics.event_count = 71;
        metrics.call_count = 73;
        const rmd::RmdStatus status = fixture.payload == nullptr ? rmd::RmdStatus::invalid_packet :
            residual::execute_direct_stripe(fixture.args, *fixture.payload, actual, &metrics);
        bool case_ok = check(status == rmd::RmdStatus::success,
                             "matched-width direct oracle executes");
        if (test.family == WeightFamily::H0) {
            const auto * output = std::get_if<rmd::PreScaledFloat64Correction>(&actual);
            case_ok = check(output != nullptr &&
                                values_match(output->values, test.floating_expected),
                            "H0 direct output matches literal pre-scaled oracle") && case_ok;
        } else {
            const auto * output = std::get_if<rmd::BlockScaledInt64Correction>(&actual);
            case_ok = check(output != nullptr &&
                                values_match(output->values, test.integer_expected),
                            "H1/HP1 direct output matches literal integer-domain oracle") && case_ok;
            if (output != nullptr) {
                std::vector<rmd::ReferenceResidual> residuals;
                residuals.reserve(fixture.payload->events.size());
                for (const residual::ResidualEvent & event : fixture.payload->events) {
                    residuals.push_back({
                        static_cast<uint32_t>(event.local_row),
                        static_cast<uint32_t>(event.original_k), event.residual});
                }
                std::vector<rmd::OutputValue> direct_reference = {901, 902};
                std::vector<rmd::OutputValue> radix_reference = {903, 904};
                const rmd::RmdStatus direct_status = rmd::reference_direct_correction(
                    fixture.args, fixture.payload->row_count, residuals,
                    direct_reference);
                const rmd::RmdStatus radix_status = rmd::reference_rmd_correction(
                    fixture.args, fixture.payload->row_count, residuals,
                    radix_reference);
                case_ok = check(direct_status == rmd::RmdStatus::success &&
                                    radix_status == rmd::RmdStatus::success &&
                                    direct_reference == output->values &&
                                    radix_reference == output->values,
                                "direct, radix reference, and executor agree by width") &&
                    case_ok;
            }
            if (test.bits == 8 && output != nullptr) {
                rmd::DirectOutput tagged = rmd::BlockScaledInt64Correction{{901, 902}};
                const rmd::RmdStatus tagged_status = residual::execute_direct_stripe(
                    fixture.args, *fixture.payload, tagged, nullptr);
                const auto * tagged_integer =
                    std::get_if<rmd::BlockScaledInt64Correction>(&tagged);
                case_ok = check(
                    tagged_status == rmd::RmdStatus::success && tagged_integer != nullptr &&
                        values_match(tagged_integer->values, test.integer_expected),
                    "Q8 integer direct output retains its explicit domain tag") && case_ok;
            }
        }
        case_ok = check(metrics.event_count == fixture.payload->events.size() &&
                            metrics.call_count == 1,
                        "direct metrics commit only after complete success") && case_ok;
        std::printf("DIRECT_ORACLE width=%u family=%s status=%s values=6\n",
                    static_cast<unsigned>(test.bits), weight_family_name(test.family),
                    rmd::rmd_status_message(status));
        ok = case_ok && ok;
    }
    return ok;
}

bool test_a16_allocation_products() {
    ggml::gemmini::quants::act::QuantizedActivationBuffer buffer;
    if (!check(buffer.allocate(1, 2, 16), "A16 allocation sentinel initializes")) {
        return false;
    }
    buffer.bytes->at(0) = 0x5a;
    const auto original_bytes = buffer.bytes;
    const size_t original_rows = buffer.rows;
    const size_t original_cols = buffer.cols;
    const size_t original_stride = buffer.row_stride_bytes;
    bool ok = check(!buffer.allocate(1, std::numeric_limits<size_t>::max(), 16),
                    "A16 column byte product overflow rejects before allocation");
    ok = check(!buffer.allocate(std::numeric_limits<size_t>::max() / 4 + 1, 2, 16),
               "A16 row byte product overflow rejects before allocation") && ok;
    return check(buffer.bytes == original_bytes && buffer.bytes->at(0) == 0x5a &&
                     buffer.rows == original_rows && buffer.cols == original_cols &&
                     buffer.row_stride_bytes == original_stride,
                 "failed A16 allocation leaves prior buffer byte-identical") && ok;
}

bool test_direct_failure_matrix() {
    const rmd::DirectOutput sentinel = rmd::PreScaledFloat64Correction{{13.25, -9.5, 4.0}};
    auto fails_atomically = [&](const ggml_gemmini_args_t & args,
                                const residual::DirectStripePayload & payload,
                                rmd::RmdStatus expected,
                                const char * message) {
        rmd::DirectOutput output = sentinel;
        const auto * before = std::get_if<rmd::PreScaledFloat64Correction>(&output);
        const double * const before_data = before->values.data();
        const size_t before_capacity = before->values.capacity();
        residual::DirectExecutionMetrics metrics{};
        metrics.event_count = 79;
        metrics.call_count = 83;
        const rmd::RmdStatus status =
            residual::execute_direct_stripe(args, payload, output, &metrics);
        const auto * after = std::get_if<rmd::PreScaledFloat64Correction>(&output);
        return check(status == expected && direct_outputs_match(output, sentinel) &&
                         after != nullptr && after->values.data() == before_data &&
                         after->values.capacity() == before_capacity &&
                         std::memcmp(after->values.data(),
                                     std::get<rmd::PreScaledFloat64Correction>(sentinel).values.data(),
                                     after->values.size() * sizeof(double)) == 0 &&
                         metrics.event_count == 79 && metrics.call_count == 83,
                     message);
    };

    bool ok = test_a16_allocation_products();

    DirectOracleFixture malformed_payload_fixture(16, WeightFamily::H1);
    residual::DirectStripePayload malformed_payload = *malformed_payload_fixture.payload;
    std::swap(malformed_payload.events[0], malformed_payload.events[1]);
    ok = fails_atomically(malformed_payload_fixture.args, malformed_payload,
                          rmd::RmdStatus::invalid_packet,
                          "malformed payload leaves direct output unchanged") && ok;

    DirectOracleFixture missing_extent(8, WeightFamily::H1);
    missing_extent.args.native_weight_bytes = 0;
    ok = fails_atomically(missing_extent.args, *missing_extent.payload,
                          rmd::RmdStatus::unsupported_route,
                          "zero native byte extent rejects atomically") && ok;

    DirectOracleFixture malformed_storage(4, WeightFamily::H1);
    malformed_storage.args.native_weight_bytes = sizeof(malformed_storage.q4_h1) - 1;
    ok = fails_atomically(malformed_storage.args, *malformed_storage.payload,
                          rmd::RmdStatus::unsupported_route,
                          "truncated Q4 storage leaves direct output unchanged") && ok;

    DirectOracleFixture reference_fixture(16, WeightFamily::HP1);
    std::vector<rmd::ReferenceResidual> malformed_reference = {
        {0, static_cast<uint32_t>(reference_fixture.args.K), 1},
    };
    std::vector<rmd::OutputValue> reference_sentinel = {107, 109, 113};
    const std::vector<rmd::OutputValue> reference_before = reference_sentinel;
    ok = check(rmd::reference_rmd_correction(
                   reference_fixture.args, 1, malformed_reference,
                   reference_sentinel) == rmd::RmdStatus::invalid_arguments &&
                   reference_sentinel == reference_before,
               "malformed width-native reference input rejects atomically") && ok;

    DirectOracleFixture invalid_h0(16, WeightFamily::H0);
    const uint16_t quiet_nan = 0x7e00u;
    std::memcpy(&invalid_h0.q16_h0[0].d, &quiet_nan, sizeof(quiet_nan));
    ok = fails_atomically(invalid_h0.args, *invalid_h0.payload,
                          rmd::RmdStatus::unsupported_route,
                          "non-finite H0 block scale leaves direct output unchanged") && ok;

    DirectOracleFixture shape_overflow(16, WeightFamily::H1);
    residual::DirectStripePayload oversized = *shape_overflow.payload;
    oversized.row_begin = 0;
    oversized.row_count = std::numeric_limits<size_t>::max() / 2 + 1;
    ok = fails_atomically(shape_overflow.args, oversized, rmd::RmdStatus::overflow,
                          "direct result size overflow leaves output unchanged") && ok;

    DirectOracleFixture scale_overflow(16, WeightFamily::HP1);
    scale_overflow.q16_hp1[0].m = 62;
    residual::DirectStripeBuilder scale_builder;
    scale_builder.reset(23, 0, 1, 64, 2);
    scale_builder.add_residual(0, 0, std::numeric_limits<int32_t>::max());
    const auto scale_payload = scale_builder.finish();
    ok = check(scale_payload != nullptr, "integer scale overflow payload builds") && ok;
    if (scale_payload != nullptr) {
        ok = fails_atomically(scale_overflow.args, *scale_payload, rmd::RmdStatus::overflow,
                              "int64 block-scale product overflow is atomic") && ok;
    }

    std::array<block_q16_h1, 2> add_weights{};
    for (block_q16_h1 & block : add_weights) {
        std::fill(std::begin(block.qs), std::end(block.qs), int16_t{32767});
        block.c_b = 1;
        block.R = 2999;
        block.s_rf = 0.25f;
    }
    ggml_gemmini_args_t add_args{};
    add_args.I = add_args.J = 1;
    add_args.K = 64;
    add_args.block_size_k = 32;
    add_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q16_h1;
    add_args.q16_h1_blocks = add_weights.data();
    add_args.native_block_count = add_weights.size();
    add_args.native_blocks_per_row = add_weights.size();
    add_args.native_weight_bytes = sizeof(add_weights);
    residual::DirectStripeBuilder add_builder;
    add_builder.reset(29, 0, 1, 64, 1);
    for (size_t k = 0; k < 64; ++k) {
        add_builder.add_residual(0, k, std::numeric_limits<int32_t>::max());
    }
    const auto add_payload = add_builder.finish();
    ok = check(add_payload != nullptr, "integer add overflow payload builds") && ok;
    if (add_payload != nullptr) {
        ok = fails_atomically(add_args, *add_payload, rmd::RmdStatus::overflow,
                              "int64 cross-block add overflow is atomic") && ok;
    }

    if (ok) {
        std::puts(
            "DIRECT_FAILURE_MATRIX malformed=1 storage=1 a16=1 scale=1 add=1 "
            "reference_malformed=1 atomic=1");
    }
    return ok;
}

struct CompactOracleResidual {
    residual::ResidualEvent event{};
    uint8_t lane_id = 0;
    int8_t digit = 0;
};

struct CompactOracleFixture {
    static constexpr size_t rows = kArrayDim + 1;
    static constexpr size_t columns = 3;
    static constexpr size_t logical_k = 2 * kBlockSize;
    static constexpr size_t blocks_per_row = logical_k / kBlockSize;
    static constexpr size_t block_count = columns * blocks_per_row;

    ggml_gemmini_args_t args{};
    std::array<block_q4_h1, block_count> q4_h1{};
    std::array<block_q4_hp1, block_count> q4_hp1{};
    std::array<block_q8_h1, block_count> q8_h1{};
    std::array<block_q8_hp1, block_count> q8_hp1{};
    std::array<block_q16_h1, block_count> q16_h1{};
    std::array<block_q16_hp1, block_count> q16_hp1{};
    std::vector<int32_t> codes;
    std::array<uint64_t, block_count> integer_scales{};
    std::vector<CompactOracleResidual> residuals;
    StripePacketHandle packet;
    uint8_t active_lane_count = 0;
    bool valid = false;

    CompactOracleFixture(uint8_t bits, WeightFamily family) {
        using Format = ggml_gemmini_args_t::im2p_weight_format_t;
        args.I = rows;
        args.J = columns;
        args.K = logical_k;
        args.block_size_k = kBlockSize;
        args.native_block_count = block_count;
        args.native_blocks_per_row = blocks_per_row;
        if (!args.A.allocate(rows, logical_k, bits)) {
            return;
        }

        for (auto & block : q4_h1) {
            std::fill(std::begin(block.qs), std::end(block.qs), uint8_t{0x88});
        }
        for (auto & block : q4_hp1) {
            std::fill(std::begin(block.qs), std::end(block.qs), uint8_t{0x88});
        }
        codes.assign(columns * logical_k, 0);

        auto set_q4 = [](auto & block, size_t local_k, int32_t value) {
            const size_t byte = local_k % (kBlockSize / 2);
            const uint8_t nibble = static_cast<uint8_t>(value + 8);
            if (local_k < kBlockSize / 2) {
                block.qs[byte] = static_cast<uint8_t>((block.qs[byte] & 0xf0u) | nibble);
            } else {
                block.qs[byte] = static_cast<uint8_t>((block.qs[byte] & 0x0fu) |
                                                      (nibble << 4));
            }
        };
        for (size_t j = 0; j < columns; ++j) {
            for (size_t block_id = 0; block_id < blocks_per_row; ++block_id) {
                const size_t block_index = j * blocks_per_row + block_id;
                for (size_t local_k = 0; local_k < kBlockSize; ++local_k) {
                    const size_t seed = 1 + j * 97 + block_id * 43 + local_k * 19;
                    int32_t code = bits == 4 ? static_cast<int32_t>(seed % 16) - 8 :
                        bits == 8 ? static_cast<int32_t>((seed * 37) % 256) - 128 :
                                    static_cast<int32_t>((seed * 7919) % 65536) - 32768;
                    if (local_k == 0) {
                        code = -(int32_t{1} << (bits - 1));
                    } else if (local_k == 1) {
                        code = (int32_t{1} << (bits - 1)) - 1;
                    } else if (local_k == 2) {
                        code = 0;
                    }
                    codes[j * logical_k + block_id * kBlockSize + local_k] = code;
                    if (bits == 4) {
                        set_q4(q4_h1[block_index], local_k, code);
                        set_q4(q4_hp1[block_index], local_k, code);
                    } else if (bits == 8) {
                        q8_h1[block_index].qs[local_k] = static_cast<int8_t>(code);
                        q8_hp1[block_index].qs[local_k] = static_cast<int8_t>(code);
                    } else {
                        q16_h1[block_index].qs[local_k] = static_cast<int16_t>(code);
                        q16_hp1[block_index].qs[local_k] = static_cast<int16_t>(code);
                    }
                }
            }
        }

        constexpr std::array<uint64_t, block_count> h1_scales = {
            0, 1, 257, 65535, 3, 1024,
        };
        constexpr std::array<int16_t, block_count> hp1_exponents = {
            std::numeric_limits<int16_t>::min(), 0, 1, 8, 20, 2,
        };
        for (size_t index = 0; index < block_count; ++index) {
            const size_t column = index / blocks_per_row;
            const uint64_t h1_scale = h1_scales[index];
            const uint8_t code = static_cast<uint8_t>(std::min<uint64_t>(h1_scale, 255));
            const uint16_t offset = static_cast<uint16_t>(h1_scale - code);
            q4_h1[index].c_b = q8_h1[index].c_b = q16_h1[index].c_b = code;
            q4_h1[index].R = q8_h1[index].R = q16_h1[index].R = offset;
            q4_h1[index].s_rf = q8_h1[index].s_rf = q16_h1[index].s_rf =
                static_cast<float>(column + 1) * 0.125f;

            q4_hp1[index].m = q8_hp1[index].m = q16_hp1[index].m =
                hp1_exponents[index];
            q4_hp1[index].channel_scale = q8_hp1[index].channel_scale =
                q16_hp1[index].channel_scale =
                    static_cast<float>(column + 1) * 0.0625f;
            integer_scales[index] = family == WeightFamily::H1 ? h1_scale :
                hp1_exponents[index] == std::numeric_limits<int16_t>::min() ? 0 :
                uint64_t{1} << static_cast<unsigned>(hp1_exponents[index]);
        }

        if (bits == 4) {
            if (family == WeightFamily::H1) {
                args.weight_format = Format::q4_h1;
                args.q4_h1_blocks = q4_h1.data();
                args.native_weight_bytes = sizeof(q4_h1);
            } else {
                args.weight_format = Format::q4_hp1;
                args.q4_hp1_blocks = q4_hp1.data();
                args.native_weight_bytes = sizeof(q4_hp1);
            }
        } else if (bits == 8) {
            if (family == WeightFamily::H1) {
                args.weight_format = Format::q8_h1;
                args.q8_h1_blocks = q8_h1.data();
                args.q8_h1_block_count = q8_h1.size();
                args.q8_h1_rows = columns;
                args.blocks_per_row = blocks_per_row;
                args.native_weight_bytes = sizeof(q8_h1);
            } else {
                args.weight_format = Format::q8_hp1;
                args.q8_hp1_blocks = q8_hp1.data();
                args.q8_hp1_block_count = q8_hp1.size();
                args.q8_hp1_blocks_per_row = blocks_per_row;
                args.native_weight_bytes = sizeof(q8_hp1);
            }
        } else if (family == WeightFamily::H1) {
            args.weight_format = Format::q16_h1;
            args.q16_h1_blocks = q16_h1.data();
            args.native_weight_bytes = sizeof(q16_h1);
        } else {
            args.weight_format = Format::q16_hp1;
            args.q16_hp1_blocks = q16_hp1.data();
            args.native_weight_bytes = sizeof(q16_hp1);
        }

        const BalancedRadixContract radix = balanced_radix_contract(bits);
        // Keep this fixture's original small magnitudes: its scale-overflow tests
        // target block scaling, while full-int32 execution is exercised separately.
        constexpr int32_t fixture_magnitude = int32_t{1} << 20;
        std::array<int64_t, kMaxNativeRadixLanes> places{};
        places[0] = 1;
        active_lane_count = 1;
        while (active_lane_count < radix.lane_capacity &&
               places[active_lane_count - 1] <=
                   static_cast<int64_t>(fixture_magnitude) / radix.radix) {
            places[active_lane_count] =
                places[active_lane_count - 1] * radix.radix;
            ++active_lane_count;
        }
        auto digit_for = [&](size_t ordinal, uint8_t lane) {
            return places[lane] >= fixture_magnitude || ordinal % 2 != 0 ?
                int8_t{-1} : int8_t{1};
        };
        for (size_t local_k = 0; local_k < 19; ++local_k) {
            const uint8_t lane = static_cast<uint8_t>(local_k % active_lane_count);
            const int8_t digit = digit_for(local_k, lane);
            residuals.push_back({
                {local_k % rows, local_k, static_cast<int32_t>(digit * places[lane])},
                lane,
                digit,
            });
        }
        constexpr std::array<size_t, 6> second_block_k = {0, 1, 2, 15, 16, 31};
        for (size_t index = 0; index < second_block_k.size(); ++index) {
            const uint8_t lane =
                static_cast<uint8_t>((index + 1) % active_lane_count);
            const int8_t digit = digit_for(index, lane);
            residuals.push_back({
                {(index * 5) % rows, kBlockSize + second_block_k[index],
                 static_cast<int32_t>(digit * places[lane])},
                lane,
                digit,
            });
        }

        RmdStripeBuilder builder;
        builder.reset(41, 9, rows, logical_k, columns, bits);
        for (const CompactOracleResidual & residual : residuals) {
            if (!builder.add_residual(residual.event.local_row,
                                      residual.event.original_k,
                                      residual.event.residual)) {
                return;
            }
        }
        packet = builder.finish();
        valid = packet != nullptr && builder.status() == RmdStatus::success;
    }

    int32_t code(size_t j, size_t k) const {
        return codes[j * logical_k + k];
    }
};

bool compact_oracle_output(const CompactOracleFixture & fixture,
                           std::vector<OutputValue> & expected) {
    if (!fixture.valid) {
        return false;
    }
    expected.assign(fixture.packet->total_output_values, OutputValue{0});
    for (const BlockDescriptor & block : fixture.packet->blocks) {
        for (size_t lane_position = 0; lane_position < block.active_lane_count;
             ++lane_position) {
            const uint8_t lane_id = block.lane_ids[lane_position];
            const size_t lane_base = block.output_value_offset +
                lane_position * block.lane_stride_values;
            for (size_t row = 0; row < CompactOracleFixture::rows; ++row) {
                for (size_t j = 0; j < CompactOracleFixture::columns; ++j) {
                    __int128 raw = 0;
                    for (const CompactOracleResidual & residual : fixture.residuals) {
                        if (residual.event.local_row != row || residual.lane_id != lane_id ||
                            residual.event.original_k / kBlockSize != block.block_id) {
                            continue;
                        }
                        raw += static_cast<__int128>(residual.digit) *
                            fixture.code(j, residual.event.original_k);
                    }
                    const size_t scale_index = j * CompactOracleFixture::blocks_per_row +
                        block.block_id;
                    const __int128 scaled = raw * fixture.integer_scales[scale_index];
                    if (scaled < std::numeric_limits<int64_t>::min() ||
                        scaled > std::numeric_limits<int64_t>::max()) {
                        return false;
                    }
                    expected[lane_base + row * fixture.packet->j_padded + j] =
                        static_cast<int64_t>(scaled);
                }
            }
        }
    }
    return true;
}

bool compressed_outputs_match(const CompressedOutput & lhs,
                              const CompressedOutput & rhs) {
    return lhs.domain == rhs.domain && lhs.j_padded == rhs.j_padded &&
        lhs.values == rhs.values;
}

bool test_block_scale_before_sum() {
    CompactOracleFixture fixture(8, WeightFamily::H1);
    if (!check(fixture.valid, "BLOCK compact scale fixture builds")) return false;

    RmdStripeBuilder builder;
    builder.reset(79, 0, 1, 2 * kBlockSize, 1, 8);
    builder.add_residual(0, 0, 1);
    builder.add_residual(0, kBlockSize, 1);
    const StripePacketHandle packet = builder.finish();
    if (!check(packet != nullptr && packet->blocks.size() == 2,
               "BLOCK compact packet keeps both original K blocks")) return false;

    CompressedOutput compressed;
    compressed.j_padded = packet->j_padded;
    compressed.values.assign(packet->total_output_values, 0);
    compressed.values[packet->blocks[0].output_value_offset] = 32;
    compressed.values[packet->blocks[1].output_value_offset] = -32;

    fixture.args.I = 1;
    fixture.args.J = 1;
    fixture.q8_h1[0].qs[0] = 32;
    fixture.q8_h1[1].qs[0] = -32;
    fixture.q8_h1[0].c_b = 1;
    fixture.q8_h1[1].c_b = 1;
    fixture.q8_h1[0].R = 0;
    fixture.q8_h1[1].R = 0;
    fixture.q8_h1[0].s_rf = 1.0f;
    fixture.q8_h1[1].s_rf = 1.0f;
    auto & metadata =
        fixture.args.act_quant.storage().emplace<act::block::Meta>();
    metadata.rows = 1;
    metadata.cols = 2 * kBlockSize;
    metadata.scales = {1.0f, 10.0f};

    Correction correction = BlockScaledInt64Correction{{777}};
    std::array<float, 1> destination = {123.0f};
    fixture.args.f_out = destination.data();
    const RmdStatus compose_status =
        compose_block_rmd_output(fixture.args, *packet, compressed, correction);
    size_t nonzero_count = 0;
    const RmdStatus merge_status = compose_status == RmdStatus::success ?
        merge_rmd_correction(fixture.args, *packet, correction, &nonzero_count) : compose_status;
    Correction compact = BlockScaledInt64Correction{{777}};
    Correction direct = BlockScaledInt64Correction{{779}};
    residual::DirectStripeBuilder direct_builder;
    direct_builder.reset(79, 0, 1, 2 * kBlockSize, 1);
    direct_builder.add_residual(0, 0, 1);
    direct_builder.add_residual(0, kBlockSize, 1);
    const auto direct_payload = direct_builder.finish();
    const RmdStatus compact_status =
        execute_rmd_stripe_reference(fixture.args, *packet, compact);
    residual::DirectExecutionMetrics metrics{};
    const RmdStatus direct_status = direct_payload == nullptr ? RmdStatus::invalid_packet :
        residual::execute_direct_stripe(fixture.args, *direct_payload, direct, &metrics);
    bool ok = check(merge_status == RmdStatus::success && destination[0] == -165.0f &&
                              nonzero_count == 1 && metrics.event_count == 2 && metrics.call_count == 1 &&
                              compact_status == RmdStatus::success &&
                              direct_status == RmdStatus::success &&
                              direct_outputs_match(compact, direct) &&
                              direct_outputs_match(compact, correction),
                          "BLOCK direct and compact apply each activation scale before the K sum");
    correction = FullyScaledFloat64Correction{{3000000000.0}};
    destination[0] = 0.0f;
    ok = check(merge_rmd_correction(fixture.args, 0, 1, correction) == RmdStatus::success &&
                   destination[0] == static_cast<float>(3000000000.0),
               "fully-scaled correction is not clamped to INT32") && ok;
    if (ok) std::puts("BLOCK_DIRECT_COMPACT correction=-288 merged=-165 unclamped=3e9 tag=fully_scaled");
    return ok;
}

bool test_block_correction_failure_atomicity() {
    CompactOracleFixture fixture(8, WeightFamily::H1);
    if (!check(fixture.valid, "BLOCK correction failure fixture builds")) return false;

    auto & metadata =
        fixture.args.act_quant.storage().emplace<act::block::Meta>();
    metadata.rows = fixture.args.I + fixture.packet->row_begin;
    metadata.cols = fixture.args.K;
    metadata.scales.assign(metadata.rows * CompactOracleFixture::blocks_per_row, 1.0f);

    Correction correction = FullyScaledFloat64Correction{{13.25, -9.5}};
    const Correction correction_before = correction;
    RmdExecutionMetrics metrics{};
    metrics.packet_call_count = 17;
    metrics.matmul_call_count = 19;
#if !defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) && \
    !defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART) && !defined(__riscv)
    bool ok = check(execute_rmd_stripe_ws(
                        fixture.args, *fixture.packet, correction, &metrics) ==
                            RmdStatus::unsupported_route &&
                        direct_outputs_match(correction, correction_before) &&
                        metrics.packet_call_count == 17 && metrics.matmul_call_count == 19,
                    "production host BLOCK compact rejection is failure-atomic");
#else
    bool ok = true;
#endif

    std::vector<OutputValue> values;
    if (!compact_oracle_output(fixture, values)) return false;
    CompressedOutput compressed;
    compressed.j_padded = fixture.packet->j_padded;
    compressed.values = std::move(values);
    metadata.scales[0] = std::numeric_limits<float>::quiet_NaN();
    correction = correction_before;
    ok = check(compose_block_rmd_output(
                   fixture.args, *fixture.packet, compressed, correction) ==
                       RmdStatus::invalid_arguments &&
                   direct_outputs_match(correction, correction_before),
               "malformed BLOCK scale preserves compact correction") && ok;

    metadata.scales[0] = 1.0f;
#if GGML_GEMMINI_WEIGHT_BITS == 8
    auto channel_args = fixture.args;
    std::vector<elem_t> channel_codes(channel_args.J * channel_args.K, 1);
    std::vector<float> channel_scales(channel_args.J, 1.0f);
    channel_args.B = channel_codes.data();
    channel_args.sB = channel_args.K;
    channel_args.weight_format =
        ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
    channel_args.weight_channel_scales = channel_scales.data();
    channel_args.weight_channel_scale_count = channel_scales.size();
    ok = check(channel_args.has_q8_channel_dense_sidecar_contract() &&
                   compose_block_rmd_output(channel_args, *fixture.packet, compressed,
                                            correction) == RmdStatus::unsupported_route &&
                   direct_outputs_match(correction, correction_before) &&
                   execute_rmd_stripe_reference(channel_args, *fixture.packet, correction,
                                                &metrics) == RmdStatus::unsupported_route &&
                   direct_outputs_match(correction, correction_before) &&
                   metrics.packet_call_count == 17 && metrics.matmul_call_count == 19,
               "unsupported BLOCK channel correction preserves output and metrics") && ok;
#endif
    std::vector<float> destination(fixture.args.I * fixture.args.J, 23.0f);
    const auto destination_before = destination;
    fixture.args.f_out = destination.data();
    Correction huge = FullyScaledFloat64Correction{
        std::vector<double>(fixture.args.J, std::numeric_limits<double>::max())};
    size_t nonzero_count = 29;
    ok = check(merge_rmd_correction(
                   fixture.args, 0, 1, huge, &nonzero_count) == RmdStatus::overflow &&
                   destination == destination_before && nonzero_count == 29,
               "fully-scaled overflow preserves destination and metrics") && ok;

    fixture.args.act_quant.storage().emplace<act::tensor::Meta>().scale = 1.0f;
    correction = FullyScaledFloat64Correction{
        std::vector<double>(fixture.args.J, 1.0)};
    destination.assign(destination.size(), 31.0f);
    const auto domain_before = destination;
    nonzero_count = 37;
    const bool domain_ok = check(merge_rmd_correction(
                     fixture.args, 0, 1, correction, &nonzero_count) ==
                         RmdStatus::unsupported_route &&
                     destination == domain_before && nonzero_count == 37,
                 "fully-scaled domain mismatch preserves destination and metrics") && ok;
    if (domain_ok) {
        std::puts("BLOCK_CORRECTION_FAILURE malformed=atomic overflow=atomic domain=atomic host_fallback=rejected");
    }
    return domain_ok;
}

RmdStatus stream_compressed_reverse(const StripePacket & packet,
                                    const CompressedOutput & compressed,
                                    Correction & correction) {
    RmdOutputAssembler assembler;
    RmdStatus status = assembler.begin(packet, correction);
    if (status != RmdStatus::success) return status;
    for (size_t block_index = packet.blocks.size(); block_index-- > 0;) {
        const BlockDescriptor & block = packet.blocks[block_index];
        for (size_t lane = block.active_lane_count; lane-- > 0;) {
            for (size_t m = (packet.row_count + kArrayDim - 1) / kArrayDim; m-- > 0;) {
                for (size_t j = packet.j_padded / kArrayDim; j-- > 0;) {
                    std::array<OutputValue, kArrayDim * kArrayDim> values{};
                    const size_t rows = std::min(kArrayDim, packet.row_count - m * kArrayDim);
                    const size_t cols = std::min(kArrayDim, packet.logical_j - j * kArrayDim);
                    for (size_t row = 0; row < rows; ++row) {
                        const size_t source = block.output_value_offset +
                            lane * block.lane_stride_values +
                            (m * kArrayDim + row) * packet.j_padded + j * kArrayDim;
                        std::copy_n(compressed.values.data() + source, cols,
                                    values.data() + row * kArrayDim);
                    }
                    PhysicalTile tile{};
                    tile.packet_block_index = static_cast<uint32_t>(block_index);
                    tile.lane_position = static_cast<uint8_t>(lane);
                    tile.lane_id = block.lane_ids[lane];
                    tile.m_tile = static_cast<uint32_t>(m);
                    tile.j_tile = static_cast<uint32_t>(j);
                    tile.valid_rows = static_cast<uint16_t>(rows);
                    tile.valid_cols = static_cast<uint16_t>(cols);
                    tile.values = values.data();
                    status = assembler.submit(tile);
                    if (status != RmdStatus::success) return status;
                }
            }
        }
    }
    return assembler.finish();
}

bool test_streaming_assembler_boundaries() {
    const Correction sentinel = PreScaledFloat64Correction{{13.25, -9.5}};
    bool ok = true;
    for (uint8_t bits : {uint8_t{4}, uint8_t{8}, uint8_t{16}}) {
        RmdStripeBuilder builder;
        builder.reset(53, 0, 1, 3 * kBlockSize, 1, bits);
        builder.add_residual(0, 0, int32_t{1} << bits);
        builder.add_residual(0, kBlockSize, int32_t{1} << bits);
        builder.add_residual(0, 2 * kBlockSize, 1);
        const StripePacketHandle packet = builder.finish();
        if (!check(packet != nullptr && packet->blocks.size() == 3,
                   "streaming cancellation packet builds")) return false;
        CompressedOutput compressed;
        compressed.j_padded = packet->j_padded;
        compressed.values.assign(packet->total_output_values, 0);
        compressed.values[packet->blocks[0].output_value_offset] =
            std::numeric_limits<int64_t>::max();
        compressed.values[packet->blocks[1].output_value_offset] =
            -std::numeric_limits<int64_t>::max();
        compressed.values[packet->blocks[2].output_value_offset] = 7;
        Correction composed = sentinel;
        Correction streamed = sentinel;
        ok = check(compose_rmd_output(*packet, compressed, composed) == RmdStatus::success &&
                       stream_compressed_reverse(*packet, compressed, streamed) ==
                           RmdStatus::success &&
                       direct_outputs_match(composed, streamed) &&
                       std::get<BlockScaledInt64Correction>(streamed).values ==
                           std::vector<int64_t>{7},
                   "streaming original lane shifts preserve wide cross-block cancellation") && ok;

        compressed.values[packet->blocks[1].output_value_offset] = 0;
        composed = streamed = sentinel;
        ok = check(compose_rmd_output(*packet, compressed, composed) == RmdStatus::overflow &&
                       stream_compressed_reverse(*packet, compressed, streamed) ==
                           RmdStatus::overflow &&
                       direct_outputs_match(composed, sentinel) &&
                       direct_outputs_match(streamed, sentinel),
                   "streaming final narrowing overflow leaves correction unchanged") && ok;

        std::array<OutputValue, kArrayDim * kArrayDim> values{};
        values[0] = 1;
        PhysicalTile tile{};
        tile.lane_id = packet->blocks[0].lane_ids[0];
        tile.valid_rows = tile.valid_cols = 1;
        tile.values = values.data();
        RmdOutputAssembler assembler;
        streamed = sentinel;
        ok = check(assembler.begin(*packet, streamed) == RmdStatus::success &&
                       direct_outputs_match(streamed, sentinel) &&
                       assembler.submit(tile) == RmdStatus::success &&
                       direct_outputs_match(streamed, sentinel) &&
                       assembler.finish() == RmdStatus::invalid_packet &&
                       direct_outputs_match(streamed, sentinel),
                   "streaming missing tiles do not publish a partial correction") && ok;
        ok = check(assembler.begin(*packet, streamed) == RmdStatus::success &&
                       assembler.submit(tile) == RmdStatus::success &&
                       assembler.submit(tile) == RmdStatus::invalid_arguments &&
                       direct_outputs_match(streamed, sentinel),
                   "streaming duplicate tile preserves correction") && ok;
        ++tile.lane_id;
        ok = check(assembler.begin(*packet, streamed) == RmdStatus::success &&
                       assembler.submit(tile) == RmdStatus::invalid_arguments &&
                       direct_outputs_match(streamed, sentinel),
                   "streaming mismatched lane tag preserves correction") && ok;
    }
    return ok;
}

bool test_compact_oracle_happy_matrix() {
    struct CompactCase {
        uint8_t bits;
        WeightFamily family;
    };
    constexpr std::array<CompactCase, 6> cases = {{
        {4, WeightFamily::H1}, {4, WeightFamily::HP1},
        {8, WeightFamily::H1}, {8, WeightFamily::HP1},
        {16, WeightFamily::H1}, {16, WeightFamily::HP1},
    }};

    bool ok = true;
    std::array<bool, 3> have_geometry{};
    std::array<std::array<size_t, 7>, 3> invariant_geometry{};
    for (const CompactCase & test : cases) {
        CompactOracleFixture fixture(test.bits, test.family);
#if defined(GGML_GEMMINI_TESTING)
        constexpr std::array<uint16_t, 6> edge_k = {0, 1, 2, 31, 15, 16};
        std::array<int32_t, kArrayDim * kArrayDim> wide_tile{};
        wide_tile.fill(123456789);
        RmdExecutionMetrics gather_metrics{};
        ggml::gemmini::quants::wreader::test_reset_weight_reader_counters();
        const RmdStatus gather_status = fixture.valid ?
            rmd::gather_wide_weight_tile_for_test(
                fixture.args, 1, edge_k.data(), edge_k.size(), 1,
                CompactOracleFixture::columns - 1, wide_tile.data(), kArrayDim,
                &gather_metrics) :
            RmdStatus::invalid_packet;
        bool gather_ok = check(gather_status == RmdStatus::success,
                               "matched-width WeightGather accepts signed edge codes");
        gather_ok = check(gather_metrics.weight_values_gathered == edge_k.size() * 2 &&
                              gather_metrics.weight_baseline_address_resolutions == edge_k.size() * 2 &&
                              gather_metrics.weight_address_resolutions == 2 &&
                              ggml::gemmini::quants::wreader::test_weight_reader_code_address_resolutions() == 2,
                          "native gather resolves each selected column block exactly once") && gather_ok;
        for (size_t k = 0; k < kArrayDim; ++k) {
            for (size_t j = 0; j < kArrayDim; ++j) {
                const int32_t expected_code = k < edge_k.size() && j < 2 ?
                    fixture.code(j + 1, kBlockSize + edge_k[k]) : 123456789;
                gather_ok = check(wide_tile[k * kArrayDim + j] == expected_code,
                                  "native gather preserves original K order, column stride, and padding") && gather_ok;
            }
        }
        const auto valid_tile = wide_tile;
        auto invalid_k = edge_k;
        invalid_k.back() = kBlockSize;
        ggml::gemmini::quants::wreader::test_reset_weight_reader_counters();
        gather_ok = check(rmd::gather_wide_weight_tile_for_test(
                              fixture.args, 1, invalid_k.data(), invalid_k.size(), 1, 2,
                              wide_tile.data(), kArrayDim, &gather_metrics) == RmdStatus::execution_failed &&
                              wide_tile == valid_tile && gather_metrics.weight_address_resolutions == 2 &&
                              gather_metrics.weight_values_gathered == edge_k.size() * 2 &&
                              ggml::gemmini::quants::wreader::test_weight_reader_code_address_resolutions() == 0,
                          "late invalid K rejects before block resolution without publishing a tile") && gather_ok;
        uint16_t single_k = 31;
        wide_tile.fill(123456789);
        auto single_expected = wide_tile;
        single_expected[0] = fixture.code(1, kBlockSize + single_k);
        single_expected[1] = fixture.code(2, kBlockSize + single_k);
        gather_metrics = {};
        ggml::gemmini::quants::wreader::test_reset_weight_reader_counters();
        gather_ok = check(rmd::gather_wide_weight_tile_for_test(
                              fixture.args, 1, &single_k, 1, 1, 2, wide_tile.data(), kArrayDim,
                              &gather_metrics) == RmdStatus::success && wide_tile == single_expected &&
                              gather_metrics.weight_values_gathered == 2 &&
                              gather_metrics.weight_baseline_address_resolutions == 2 &&
                              gather_metrics.weight_address_resolutions == 2 &&
                              ggml::gemmini::quants::wreader::test_weight_reader_code_address_resolutions() == 2,
                          "single-K native gather preserves K31, padding, and observed address counts") && gather_ok;
        single_k = kBlockSize;
        ggml::gemmini::quants::wreader::test_reset_weight_reader_counters();
        gather_ok = check(rmd::gather_wide_weight_tile_for_test(
                              fixture.args, 1, &single_k, 1, 1, 2, wide_tile.data(), kArrayDim,
                              &gather_metrics) == RmdStatus::execution_failed && wide_tile == single_expected &&
                              gather_metrics.weight_values_gathered == 2 &&
                              gather_metrics.weight_address_resolutions == 2 &&
                              ggml::gemmini::quants::wreader::test_weight_reader_code_address_resolutions() == 0,
                          "invalid single local K preserves tile and metrics before any read") && gather_ok;
#else
        bool gather_ok = check(false, "compact oracle requires GGML_GEMMINI_TESTING");
#endif

        std::vector<OutputValue> expected;
        CompressedOutput actual;
        actual.j_padded = 7;
        actual.values = {91, 92, 93};
        RmdExecutionMetrics metrics{};
        const RmdStatus status = fixture.valid && compact_oracle_output(fixture, expected) ?
            execute_rmd_stripe_reference(fixture.args, *fixture.packet, actual, &metrics) :
            RmdStatus::invalid_packet;
        bool case_ok = gather_ok;
        case_ok = check(status == RmdStatus::success,
                        "matched-width compact oracle executes") && case_ok;
        case_ok = check(actual.domain == CompressedOutput::Domain::block_scaled_int64 &&
                            actual.values == expected,
                        "compact output matches independent block/lane oracle") && case_ok;
        case_ok = check(metrics.packet_call_count == 1 && metrics.ws_call_count == 0,
                        "software compact executes one packet and no hardware dispatch") && case_ok;
        if (status == RmdStatus::success) {
            Correction composed = PreScaledFloat64Correction{{13.25}};
            Correction streamed = composed;
            RmdExecutionMetrics streaming_metrics{};
            case_ok = check(compose_rmd_output(*fixture.packet, actual, composed) ==
                                RmdStatus::success &&
                                execute_rmd_stripe_reference(fixture.args, *fixture.packet,
                                                            streamed, &streaming_metrics) ==
                                RmdStatus::success &&
                                direct_outputs_match(streamed, composed),
                            "streaming executor equals compact execution plus compose") && case_ok;
            case_ok = check(streaming_metrics.packet_call_count == metrics.packet_call_count &&
                                streaming_metrics.matmul_call_count == metrics.matmul_call_count &&
                                streaming_metrics.physical_tile_count == metrics.physical_tile_count &&
                                streaming_metrics.lane_group_count == metrics.lane_group_count,
                            "streaming preserves execution geometry metrics") && case_ok;
            streamed = PreScaledFloat64Correction{{-29.5}};
            case_ok = check(stream_compressed_reverse(*fixture.packet, actual, streamed) ==
                                RmdStatus::success && direct_outputs_match(streamed, composed),
                            "reverse physical tile order preserves all logical rows and columns") &&
                case_ok;
        }
        const size_t packet_block_count =
            fixture.packet == nullptr ? 0 : fixture.packet->blocks.size();
        case_ok = check(packet_block_count == 2 &&
                            fixture.packet->blocks[0].compact_k_count == 19 &&
                            fixture.packet->blocks[0].padded_k_count ==
                                align_up(19, kArrayDim) &&
                            fixture.packet->blocks[1].compact_k_count == 6,
                        "compact packet preserves K0>1 and final partial K fragment") && case_ok;
        const bool partition_expected = kArrayDim < kBlockSize;
        case_ok = check(metrics.active_lanes ==
                                2 * fixture.active_lane_count &&
                            metrics.lane_group_count >= packet_block_count &&
                            (!partition_expected ||
                             metrics.lane_group_count > packet_block_count) &&
                            metrics.matmul_call_count >= metrics.lane_group_count,
                        "compact packet exercises DIM-aware lane groups") && case_ok;

        const std::array<size_t, 7> geometry = {
            metrics.packet_call_count,
            metrics.active_blocks,
            metrics.active_lanes,
            metrics.compact_k_count,
            metrics.padded_k_count,
            metrics.matmul_call_count,
            metrics.lane_group_count,
        };
        const size_t geometry_index = test.bits == 4 ? 0 : test.bits == 8 ? 1 : 2;
        if (!have_geometry[geometry_index]) {
            invariant_geometry[geometry_index] = geometry;
            have_geometry[geometry_index] = true;
        } else {
            case_ok = check(geometry == invariant_geometry[geometry_index],
                            "compact packet counts are independent of weight family") &&
                case_ok;
        }
        std::printf(
            "COMPACT_ORACLE width=%u family=%s domain=block_scaled_int64 "
            "packets=%zu k_fragments=%zu lane_groups=%zu\n",
            static_cast<unsigned>(test.bits), weight_family_name(test.family),
            metrics.packet_call_count, metrics.matmul_call_count,
            metrics.lane_group_count);
        ok = case_ok && ok;
    }
    return ok;
}

bool test_full_int32_compact_direct_agreement(size_t rows) {
    namespace act = ggml::gemmini::quants::act;
    bool ok = true;
    for (uint8_t bits : {uint8_t{4}, uint8_t{8}, uint8_t{16}}) {
        for (WeightFamily family : {WeightFamily::H1, WeightFamily::HP1}) {
            WeightCapabilityFixture fixture(bits, family);
            const auto contract = balanced_radix_contract(bits);
            int64_t carry_boundary = 0;
            int64_t place = 1;
            for (uint8_t lane = 0; lane + 1 < contract.lane_capacity; ++lane) {
                carry_boundary += contract.digit_max * place;
                place *= contract.radix;
            }
            const std::array<int32_t, 8> samples = {
                std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(),
                static_cast<int32_t>(carry_boundary), static_cast<int32_t>(carry_boundary + 1),
                -(int32_t{1} << 20) - 1, int32_t{1} << 20, 0, 1,
            };
            std::vector<int32_t> values(rows);
            for (size_t row = 0; row < rows; ++row) {
                values[row] = samples[row % samples.size()];
            }
            fixture.args.I = values.size();
            if (!check(fixture.args.A.allocate(values.size(), kBlockSize, bits),
                       "full int32 fixture activation storage allocates")) return false;
            fixture.args.act_quant.storage().emplace<act::tensor::Meta>().scale = 0.5f;
            fixture.q4_h1.qs[0] = fixture.q4_hp1.qs[0] = 0x85;
            fixture.q4_h1.qs[15] = fixture.q4_hp1.qs[15] = 0xd8;
            fixture.q8_h1.qs[0] = fixture.q8_hp1.qs[0] = -3;
            fixture.q8_h1.qs[31] = fixture.q8_hp1.qs[31] = 5;
            fixture.q16_h1.qs[0] = fixture.q16_hp1.qs[0] = -3;
            fixture.q16_h1.qs[31] = fixture.q16_hp1.qs[31] = 5;

            RmdStripeBuilder compact_builder;
            residual::DirectStripeBuilder direct_builder;
            compact_builder.reset(83, 0, values.size(), kBlockSize, 1, bits);
            direct_builder.reset(83, 0, values.size(), kBlockSize, 1);
            for (size_t row = 0; row < values.size(); ++row) {
                if (!compact_builder.add_residual(row, 0, values[row]) ||
                    !compact_builder.add_residual(row, 31, 1) ||
                    !direct_builder.add_residual(row, 0, values[row]) ||
                    !direct_builder.add_residual(row, 31, 1)) {
                    return check(false, "full int32 compact and direct builders accept all events");
                }
            }
            const auto packet = compact_builder.finish();
            const auto direct_payload = direct_builder.finish();
            if (!check(packet != nullptr && direct_payload != nullptr,
                       "full int32 compact and direct payloads build")) return false;

            Correction streamed = PreScaledFloat64Correction{{-7.5}};
            Correction direct = streamed;
            Correction composed = streamed;
            CompressedOutput compressed;
            RmdExecutionMetrics metrics{};
            if (!check(execute_rmd_stripe_reference(fixture.args, *packet, streamed, &metrics) ==
                           RmdStatus::success &&
                           execute_rmd_stripe_reference(fixture.args, *packet, compressed) ==
                           RmdStatus::success &&
                           compose_rmd_output(*packet, compressed, composed) == RmdStatus::success &&
                           residual::execute_direct_stripe(fixture.args, *direct_payload, direct) ==
                           RmdStatus::success,
                       "full int32 executes through streaming, compose and CPU direct")) return false;
            const int64_t scale = family == WeightFamily::H1 ? 5 : 4;
            std::vector<int64_t> expected;
            for (int32_t value : values) {
                expected.push_back((static_cast<int64_t>(value) * -3 + 5) * scale);
            }
            ok = check(std::get<BlockScaledInt64Correction>(streamed).values == expected &&
                           direct_outputs_match(streamed, direct) &&
                           direct_outputs_match(streamed, composed),
                       "top carry correction matches literal int64 and independent direct output") && ok;

            const auto & block = packet->blocks.front();
            const size_t group_rows = align_up(block.active_lane_count * rows, kArrayDim);
            ok = check(packet->blocks.size() == 1 && block.groups.size() == 1 &&
                           metrics.stacked_i_tile_count == group_rows / kArrayDim &&
                           metrics.physical_tile_count ==
                               block.active_lane_count * align_up(rows, kArrayDim) / kArrayDim &&
                           metrics.block_padding_zeros == group_rows * (kArrayDim - 2) &&
                           metrics.row_padding_zeros ==
                               (group_rows - block.active_lane_count * rows) * kArrayDim &&
                           packet->activation_value_count == group_rows * kArrayDim,
                       "group row padding drives payload size and dispatched tile metrics") && ok;
            for (size_t lane = 0; lane < block.active_lane_count; ++lane) {
                for (size_t row = 0; row < block.rows_padded; ++row) {
                    for (size_t col = 0; col < packet->j_padded; ++col) {
                        if (row >= rows || col >= packet->logical_j) {
                            const size_t index = block.output_value_offset +
                                lane * block.lane_stride_values + row * packet->j_padded + col;
                            ok = check(compressed.values[index] == 0,
                                       "dense input groups preserve zero compressed output padding") && ok;
                        }
                    }
                }
            }

            std::vector<float> output(values.size(), 7.0f);
            fixture.args.f_out = output.data();
            size_t nonzero_count = 0;
            ok = check(merge_rmd_correction(fixture.args, *packet, streamed, &nonzero_count) ==
                           RmdStatus::success && nonzero_count == values.size(),
                       "full int32 correction merges with final nonzero count") && ok;
            for (size_t row = 0; row < values.size(); ++row) {
                const float expected_output = 7.0f + static_cast<float>(
                    static_cast<double>(expected[row]) * 0.25 * 0.5);
                ok = check(output[row] == expected_output,
                           "full int32 merge matches explicit column and activation scales") && ok;
            }
        }
    }
    return ok;
}

bool test_shared_weight_preparation() {
    namespace wreader = ggml::gemmini::quants::wreader;
    namespace act = ggml::gemmini::quants::act;
    WeightCapabilityFixture fixture(GGML_GEMMINI_WEIGHT_BITS, WeightFamily::HP1);
    auto & args = fixture.args;
#if GGML_GEMMINI_WEIGHT_BITS == 4
    std::array<block_q4_hp1, 3> blocks{};
    args.q4_hp1_blocks = blocks.data();
#elif GGML_GEMMINI_WEIGHT_BITS == 16
    std::array<block_q16_hp1, 3> blocks{};
    args.q16_hp1_blocks = blocks.data();
#else
    std::array<block_q8_hp1, 3> blocks{};
    args.q8_hp1_blocks = blocks.data();
#endif
    args.I = 3;
    args.K = kBlockSize * blocks.size();
    args.q8_hp1_block_count = args.native_block_count = blocks.size();
    args.q8_hp1_blocks_per_row = args.native_blocks_per_row = blocks.size();
    args.native_weight_bytes = sizeof(blocks);
    for (size_t block = 0; block < blocks.size(); ++block) {
#if GGML_GEMMINI_WEIGHT_BITS == 4
        std::fill(std::begin(blocks[block].qs), std::end(blocks[block].qs), uint8_t{0x88});
        blocks[block].qs[0] = 0x8d;
#else
        blocks[block].qs[0] = 5;
#endif
        blocks[block].m = static_cast<int16_t>(2 + block);
        blocks[block].channel_scale = block == 2 ? 0.5f : 0.25f;
    }
    const std::array<int32_t, 3> residuals{129, -257, 3};
    std::array<StripePacketHandle, 3> packets;
    std::array<ggml_gemmini_args_t, 3> stripe_args{args, args, args};
    for (size_t stripe = 0; stripe < packets.size(); ++stripe) {
        stripe_args[stripe].act_quant.storage().emplace<act::tensor::Meta>().scale =
            0.5f * static_cast<float>(stripe + 1);
        RmdStripeBuilder builder;
        builder.reset(stripe, stripe, 1, args.K, args.J, GGML_GEMMINI_ACTIVATION_BITS);
        if (!builder.add_residual(0, stripe == 0 ? 0 : kBlockSize, residuals[stripe])) return false;
        packets[stripe] = builder.finish();
        if (!packets[stripe]) return false;
    }
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
    Correction rejected = PreScaledFloat64Correction{{19.0}};
    const Correction rejected_sentinel = rejected;
    RmdExecutionMetrics rejected_metrics{};
    rejected_metrics.packet_call_count = 73;
    rmd::detail::RmdWeightPreparation rejected_weights;
    return check(rmd::detail::execute_rmd_stripe_ws_with_weights(
                     stripe_args[0], *packets[0], rejected, rejected_weights,
                     &rejected_metrics) == RmdStatus::unsupported_route &&
                     direct_outputs_match(rejected, rejected_sentinel) &&
                     rejected_metrics.packet_call_count == 73,
                 "FPGA UART rejects shared WS execution without CPU fallback");
#endif
    std::array<Correction, 3> corrections;
    std::array<RmdStatus, 3> statuses{};
    std::array<float, 3> shared_output{7, 7, 7};
    std::array<float, 3> fresh_output = shared_output;
    rmd::detail::RmdWeightPreparation shared;
    wreader::test_reset_weight_reader_counters();
    std::array<std::thread, 3> workers;
    for (size_t stripe = 0; stripe < packets.size(); ++stripe) {
        workers[stripe] = std::thread([&, stripe] {
            statuses[stripe] = rmd::detail::execute_rmd_stripe_ws_with_weights(
                stripe_args[stripe], *packets[stripe], corrections[stripe], shared);
            if (statuses[stripe] == RmdStatus::success) {
                statuses[stripe] = rmd::detail::merge_rmd_correction_with_weights(
                    stripe_args[stripe], shared_output.data(), *packets[stripe], corrections[stripe], shared);
            }
        });
    }
    for (auto & worker : workers) worker.join();
    const size_t shared_validations = wreader::test_weight_reader_storage_validations();
    bool ok = check(shared_validations == 1 && shared.column_preparations() == 1 &&
                        shared.selected_block_preparations() == 1,
                    "concurrent stripes prepare immutable weights and selected columns once");
    wreader::test_reset_weight_reader_counters();
    for (size_t stripe = 0; stripe < packets.size(); ++stripe) {
        rmd::detail::RmdWeightPreparation fresh_weights;
        Correction fresh;
        const int64_t expected = static_cast<int64_t>(residuals[stripe]) * 5 * (stripe == 0 ? 4 : 8);
        ok = check(statuses[stripe] == RmdStatus::success &&
                       std::get<BlockScaledInt64Correction>(corrections[stripe]).values ==
                           std::vector<int64_t>{expected} &&
                       rmd::detail::execute_rmd_stripe_ws_with_weights(
                           stripe_args[stripe], *packets[stripe], fresh, fresh_weights) == RmdStatus::success &&
                       direct_outputs_match(fresh, corrections[stripe]) &&
                       rmd::detail::merge_rmd_correction_with_weights(stripe_args[stripe], fresh_output.data(),
                           *packets[stripe], fresh, fresh_weights) == RmdStatus::success &&
                       shared_output[stripe] == 7.0f + static_cast<float>(
                           static_cast<double>(expected) * 0.25 * 0.5 * (stripe + 1)),
                   "shared stripes match fresh execution and literal correction with current row scales") && ok;
    }
    const size_t fresh_validations = wreader::test_weight_reader_storage_validations();
    ok = check(fresh_validations == 3 && shared_output == fresh_output,
               "shared preparation reduces three validations to one with exact FP32 output") && ok;

    const auto output_before = shared_output;
    Correction sentinel = PreScaledFloat64Correction{{19.0}};
    Correction correction = sentinel;
    RmdExecutionMetrics metrics;
    metrics.packet_call_count = 73;
    auto malformed = *packets[0];
    ++malformed.version;
    size_t count = 91;
    ok = check(rmd::detail::execute_rmd_stripe_ws_with_weights(
                   stripe_args[0], malformed, correction, shared, &metrics) == RmdStatus::invalid_packet &&
                   direct_outputs_match(correction, sentinel) && metrics.packet_call_count == 73,
               "reused preparation retains packet validation and transactional failures") && ok;
    malformed = *packets[0];
    ++malformed.logical_j;
    ok = check(rmd::detail::merge_rmd_correction_with_weights(stripe_args[0], shared_output.data(),
                   malformed, corrections[0], shared, &count) == RmdStatus::invalid_arguments &&
                   shared_output == output_before && count == 91,
               "shared merge retains per-stripe shape checks without partial writes") && ok;
    RmdStripeBuilder selected;
    selected.reset(9, 0, 1, args.K, args.J, GGML_GEMMINI_ACTIVATION_BITS);
    if (!selected.add_residual(0, 2 * kBlockSize, 1)) return false;
    const auto mismatched = selected.finish();
    if (!mismatched) return false;
    ok = check(rmd::detail::execute_rmd_stripe_ws_with_weights(
                   stripe_args[0], *mismatched, correction, shared) == RmdStatus::success &&
                   rmd::detail::merge_rmd_correction_with_weights(stripe_args[0], shared_output.data(),
                       *mismatched, correction, shared, &count) == RmdStatus::unsupported_route &&
                   shared_output == output_before && count == 91 &&
                   rmd::detail::merge_rmd_correction_with_weights(stripe_args[0], shared_output.data(),
                       *packets[0], corrections[0], shared) == RmdStatus::success,
               "only a selected mismatched block rejects, without poisoning disjoint stripes") && ok;
    std::array<float, 3> public_output{7, 7, 7};
    ok = check(merge_rmd_correction_to(stripe_args[0], public_output.data(),
                   *packets[0], corrections[0]) == RmdStatus::success,
               "public packet merge checks column scales only for selected blocks") && ok;
    const auto public_before = public_output;
    ok = check(merge_rmd_correction_to(stripe_args[0], public_output.data(),
                   0, 1, corrections[0]) == RmdStatus::unsupported_route && public_output == public_before,
               "public row-range merge checks all block scales without partial writes") && ok;
    blocks[0].channel_scale = 0.125f;
    rmd::detail::RmdWeightPreparation next_job;
    fresh_output.fill(7.0f);
    ok = check(rmd::detail::execute_rmd_stripe_ws_with_weights(
                   stripe_args[0], *packets[0], correction, next_job) == RmdStatus::success &&
                   rmd::detail::merge_rmd_correction_with_weights(stripe_args[0], fresh_output.data(),
                       *packets[0], correction, next_job) == RmdStatus::success && fresh_output[0] == 168.25f,
               "a fresh job reads changed metadata at the same storage address") && ok;
    blocks[0].m = 63;
    rmd::detail::RmdWeightPreparation overflow_job;
    correction = sentinel;
    ok = check(rmd::detail::execute_rmd_stripe_ws_with_weights(
                   stripe_args[0], *packets[0], correction, overflow_job, &metrics) == RmdStatus::overflow &&
                   direct_outputs_match(correction, sentinel) && metrics.packet_call_count == 73,
               "a fresh job retains HP1 exponent overflow and unchanged output metrics") && ok;
    ok = check(execute_rmd_stripe_ws(stripe_args[0], *packets[0], correction) == RmdStatus::overflow &&
                   direct_outputs_match(correction, sentinel),
               "public execution revalidates changed weight metadata") && ok;
    ok = check(merge_rmd_correction_to(stripe_args[0], public_output.data(),
                   *packets[0], corrections[0]) == RmdStatus::unsupported_route &&
                   public_output == public_before,
               "public merge revalidates changed scales without partial writes") && ok;
    std::printf("SHARED_PREPARATION stripes=3 shared_validations=%zu fresh_validations=%zu columns=%zu selected_blocks=%zu\n",
                shared_validations, fresh_validations, shared.column_preparations(),
                shared.selected_block_preparations());
    return ok;
}

bool test_compact_failure_matrix() {
    const CompressedOutput sentinel = {
        CompressedOutput::Domain::block_scaled_int64, 7, {13, -9, 4},
    };
    auto fails_atomically = [&](const ggml_gemmini_args_t & args,
                                const StripePacket & packet,
                                RmdStatus expected,
                                const char * message) {
        CompressedOutput output = sentinel;
        RmdExecutionMetrics metrics{};
        const RmdStatus status = execute_rmd_stripe_ws(args, packet, output, &metrics);
        const Correction correction_sentinel = PreScaledFloat64Correction{{13.25, -9.5}};
        Correction correction = correction_sentinel;
        RmdExecutionMetrics streaming_metrics{};
        streaming_metrics.packet_call_count = 17;
        streaming_metrics.matmul_call_count = 19;
        streaming_metrics.im2p_stats.fields[0] = 23;
        const RmdStatus streaming_status =
            execute_rmd_stripe_ws(args, packet, correction, &streaming_metrics);
        return check(status == expected && compressed_outputs_match(output, sentinel) &&
                         metrics.packet_call_count == 0 && metrics.ws_call_count == 0 &&
                         metrics.matmul_call_count == 0 && streaming_status == expected &&
                         direct_outputs_match(correction, correction_sentinel) &&
                         streaming_metrics.packet_call_count == 17 &&
                         streaming_metrics.matmul_call_count == 19 &&
                         streaming_metrics.im2p_stats.fields[0] == 23,
                     message);
    };

    RmdStripeBuilder single_builder;
    single_builder.reset(43, 0, 1, kBlockSize, 1);
    single_builder.add_residual(0, 0, 1);
    const StripePacketHandle single_packet = single_builder.finish();
    bool ok = check(single_packet != nullptr, "compact failure packet builds");
    if (single_packet == nullptr) {
        return false;
    }

    for (uint8_t bits : {uint8_t{4}, uint8_t{8}, uint8_t{16}}) {
        WeightCapabilityFixture h0(bits, WeightFamily::H0);
        const ggml::gemmini::quants::wroute::WeightRoutePlan h0_plan =
            ggml::gemmini::quants::wroute::resolve_weight_route_plan(
                h0.args,
                ggml::gemmini::quants::wroute::WeightScaleInfoMode::Residual);
        if (ggml::gemmini::quants::wroute::weight_route_status(
                h0_plan,
                ggml::gemmini::quants::wroute::WeightExecutionPath::Compact) ==
            ggml::gemmini::quants::wroute::WeightRouteStatus::Success) {
            continue;
        }
        h0.args.A.allocate(1, kBlockSize, bits);
        ok = fails_atomically(h0.args, *single_packet, RmdStatus::unsupported_route,
                              "non-A8 H0 compact rejects before packet dispatch") && ok;
    }

    block_q8_h2 h2{};
    h2.channel_scale = 1.0f;
    ggml_gemmini_args_t h2_args{};
    h2_args.I = h2_args.J = 1;
    h2_args.K = kBlockSize;
    h2_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h2;
    h2_args.q8_h2_blocks = &h2;
    h2_args.q8_h2_block_count = 1;
    h2_args.q8_h2_blocks_per_row = 1;
    ok = fails_atomically(h2_args, *single_packet, RmdStatus::unsupported_route,
                          "H2 compact rejects before packet dispatch") && ok;

    block_q8_hp2 hp2{};
    hp2.channel_scale = 1.0f;
    ggml_gemmini_args_t hp2_args{};
    hp2_args.I = hp2_args.J = 1;
    hp2_args.K = kBlockSize;
    hp2_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2;
    hp2_args.q8_hp2_blocks = &hp2;
    hp2_args.q8_hp2_block_count = 1;
    hp2_args.q8_hp2_blocks_per_row = 1;
    ok = fails_atomically(hp2_args, *single_packet, RmdStatus::unsupported_route,
                          "HP2 compact rejects before packet dispatch") && ok;

    CompactOracleFixture forged_mask(8, WeightFamily::HP1);
    ok = check(forged_mask.valid, "forged K-mask fixture builds") && ok;
    if (forged_mask.valid) {
        StripePacket malformed = *forged_mask.packet;
        malformed.blocks.front().lane_k_masks.fill(0);
        ok = fails_atomically(forged_mask.args, malformed, RmdStatus::invalid_packet,
                              "forged K-mask rejects before packet dispatch") && ok;
    }

    CompactOracleFixture mixed(16, WeightFamily::H1);
    mixed.args.A.allocate(CompactOracleFixture::rows,
                          CompactOracleFixture::logical_k, 8);
    ok = check(mixed.valid, "mixed artifact compact fixture builds") && ok;
    if (mixed.valid) {
        ok = fails_atomically(mixed.args, *mixed.packet, RmdStatus::unsupported_route,
                              "mixed activation/weight identity rejects before dispatch") && ok;
    }

#if defined(GGML_GEMMINI_TESTING)
    constexpr uint8_t mismatch_bits =
        GGML_GEMMINI_ACTIVATION_BITS == 16 ? uint8_t{8} : uint8_t{16};
    CompactOracleFixture mismatched_native(mismatch_bits, WeightFamily::H1);
    ok = check(mismatched_native.valid,
               "packet/build mismatch compact fixture builds") && ok;
    if (mismatched_native.valid) {
        CompressedOutput output = sentinel;
        RmdExecutionMetrics metrics{};
        const RmdStatus status = rmd::execute_rmd_stripe_gemmini_for_test(
            mismatched_native.args, *mismatched_native.packet, output, &metrics);
        ok = check(status == RmdStatus::unsupported_route &&
                       compressed_outputs_match(output, sentinel) &&
                       metrics.packet_call_count == 0 && metrics.ws_call_count == 0 &&
                       metrics.matmul_call_count == 0,
                   "packet/build width mismatch rejects before Gemmini dispatch") && ok;
    }
    CompactOracleFixture truncated_native(GGML_GEMMINI_WEIGHT_BITS, WeightFamily::H1);
    --truncated_native.args.native_weight_bytes;
    if (truncated_native.valid) {
        CompressedOutput output = sentinel;
        RmdExecutionMetrics metrics{};
        ggml::gemmini::quants::wreader::test_reset_weight_reader_counters();
        ok = check(rmd::execute_rmd_stripe_gemmini_for_test(
                       truncated_native.args, *truncated_native.packet, output, &metrics) ==
                       RmdStatus::unsupported_route && compressed_outputs_match(output, sentinel) &&
                       metrics.packet_call_count == 0 && metrics.ws_call_count == 0 &&
                       ggml::gemmini::quants::wreader::test_weight_reader_code_address_resolutions() == 0,
                   "truncated native weights reject before any gather or Gemmini dispatch") && ok;
    } else {
        ok = check(false, "truncated native fixture builds") && ok;
    }
    if (sizeof(elem_t) < sizeof(int16_t)) {
        CompactOracleFixture wide_source(16, WeightFamily::HP1);
        const std::array<uint16_t, 2> local_k = {2, 0};
        std::array<elem_t, kArrayDim * kArrayDim> tile{};
        tile.fill(71);
        const auto before = tile;
        RmdExecutionMetrics metrics{};
        metrics.weight_values_gathered = 19;
        metrics.weight_address_resolutions = 23;
        ok = check(wide_source.valid && gather_weight_tile_for_test(
                       wide_source.args, 1, local_k.data(), local_k.size(), 1, 2,
                       tile.data(), kArrayDim, &metrics) == RmdStatus::overflow &&
                       tile == before && metrics.weight_values_gathered == 19 &&
                       metrics.weight_address_resolutions == 23,
                   "late code narrowing overflow preserves the whole tile and metrics") && ok;
    }
#else
    ok = check(false, "compact failure matrix requires GGML_GEMMINI_TESTING") && ok;
#endif

    CompactOracleFixture scale_overflow(16, WeightFamily::HP1);
    scale_overflow.q16_hp1[0].m = 62;
    std::fill(std::begin(scale_overflow.q16_hp1[0].qs),
              std::end(scale_overflow.q16_hp1[0].qs), int16_t{32767});
    ok = check(scale_overflow.valid, "scale overflow compact fixture builds") && ok;
    if (scale_overflow.valid) {
#if defined(GGML_GEMMINI_TESTING)
        CompressedOutput output = sentinel;
        RmdExecutionMetrics metrics{};
        const RmdStatus status = execute_rmd_stripe_reference(
            scale_overflow.args, *scale_overflow.packet, output, &metrics);
        ok = check(status == RmdStatus::overflow &&
                       compressed_outputs_match(output, sentinel) &&
                       metrics.packet_call_count == 0 && metrics.ws_call_count == 0 &&
                       metrics.matmul_call_count == 0,
                   "checked int64 block-scale overflow is failure-atomic") && ok;
        const Correction correction_sentinel = PreScaledFloat64Correction{{13.25, -9.5}};
        Correction correction = correction_sentinel;
        metrics.packet_call_count = 17;
        metrics.matmul_call_count = 19;
        metrics.im2p_stats.fields[0] = 23;
        ok = check(execute_rmd_stripe_reference(scale_overflow.args, *scale_overflow.packet,
                                                correction, &metrics) == RmdStatus::overflow &&
                       direct_outputs_match(correction, correction_sentinel) &&
                       metrics.packet_call_count == 17 && metrics.matmul_call_count == 19 &&
                       metrics.im2p_stats.fields[0] == 23,
                   "streaming scale overflow preserves correction and existing metrics") && ok;
#else
        ok = check(false, "scale overflow oracle requires GGML_GEMMINI_TESTING") && ok;
#endif
    }

    CompactOracleFixture malformed(GGML_GEMMINI_ACTIVATION_BITS,
                                   WeightFamily::H1);
    ok = check(malformed.valid, "malformed compact fixture builds") && ok;
    if (malformed.valid) {
        StripePacket packet = *malformed.packet;
        ++packet.activation_value_count;
        ok = fails_atomically(malformed.args, packet, RmdStatus::invalid_packet,
                              "malformed compact packet is failure-atomic") && ok;
    }

    ggml_gemmini_args_t dense;
    std::array<elem_t, kBlockSize> codes{};
    dense.B = codes.data();
    dense.J = 1;
    dense.K = kBlockSize;
    dense.sB = std::numeric_limits<size_t>::max();
    dense.weight_i8_scale_active = true;
    dense.weight_scale = 1.0f;
    const uint16_t local_k = 0;
    std::array<int32_t, kArrayDim * kArrayDim> tile{};
    tile.fill(71);
    ok = check(gather_wide_weight_tile_for_test(dense, 0, &local_k, 1, 0, 1,
                   tile.data(), kArrayDim) == RmdStatus::execution_failed && tile[0] == 71,
               "nonhierarchical gather still validates storage extent before reading") && ok;

    if (ok) {
        std::puts(
            "COMPACT_FAILURE_MATRIX h0=3 h2=1 hp2=1 mixed_identity=1 "
            "packet_build_mismatch=1 scale_overflow=1 malformed=1 dispatches=0 "
            "output=unchanged");
    }
    return ok;
}

}

static bool test_cpu_capture_is_canonical_and_packet_free() {
    using namespace ggml::gemmini::residual;
    TimedResidualCapture capture(ResidualRoute::cpu_direct);
    capture.reset(3, 7, 2, 64, 17);
    if (!capture.add_residual(1, 33, -129) ||
        !capture.add_residual(0, 31, 256) ||
        !capture.add_residual(0, 2, 1)) {
        return check(false, "CPU residual events accepted");
    }
    const ResidualStripePayload payload = capture.finish();
    if (!check(payload.direct != nullptr, "CPU creates a direct payload") ||
        !check(payload.packet == nullptr, "CPU creates no packet") ||
        !check(payload.direct->events.size() == 3, "CPU retains every original event")) {
        return false;
    }
    const auto &events = payload.direct->events;
    return check(events[0] == ResidualEvent{0, 2, 1}, "CPU event order row 0 k 2") &&
        check(events[1] == ResidualEvent{0, 31, 256}, "CPU event order row 0 k 31") &&
        check(events[2] == ResidualEvent{1, 33, -129}, "CPU event order row 1 k 33");
}

static bool test_ws_capture_preserves_packet_contract() {
    using namespace ggml::gemmini::residual;
    RmdStripeBuilder legacy;
    legacy.reset(3, 7, 2, 64, 17);
    legacy.add_residual(1, 33, -129);
    legacy.add_residual(0, 31, 256);
    legacy.add_residual(0, 2, 1);
    const StripePacketHandle expected = legacy.finish();

    TimedResidualCapture capture(ResidualRoute::ws_packet);
    capture.reset(3, 7, 2, 64, 17);
    capture.add_residual(1, 33, -129);
    capture.add_residual(0, 31, 256);
    capture.add_residual(0, 2, 1);
    const ResidualStripePayload payload = capture.finish();
    if (!check(payload.direct == nullptr, "WS creates no event payload") ||
        !check(payload.packet != nullptr, "WS creates a packet") ||
        !check(expected != nullptr, "legacy packet fixture built")) {
        return false;
    }
    return check(payload.packet->k_indices == expected->k_indices,
                 "WS K-index packet bytes preserved") &&
        check(payload.packet->stacked_activation == expected->stacked_activation,
              "WS activation packet bytes preserved");
}

static bool test_empty_capture_and_single_sink_selection() {
    using namespace ggml::gemmini::residual;
    TimedResidualCapture cpu(ResidualRoute::cpu_direct);
    cpu.reset(0, 0, 1, 1, 1);
    const ResidualStripePayload cpu_empty = cpu.finish();
    TimedResidualCapture ws(ResidualRoute::ws_packet);
    ws.reset(0, 0, 1, 1, 1);
    const ResidualStripePayload ws_empty = ws.finish();
    return check(cpu.holds_cpu_sink() && !cpu.holds_ws_sink(),
                 "CPU selection instantiates only CPU sink") &&
        check(ws.holds_ws_sink() && !ws.holds_cpu_sink(),
              "WS selection instantiates only WS sink") &&
        check(cpu_empty.empty() && ws_empty.empty(), "empty stripes produce no work") &&
        check(cpu_empty.capture_ns == 0 && ws_empty.capture_ns == 0,
              "empty stripes skip timed finish work");
}

static bool test_direct_payload_slicing_and_ownership() {
    using namespace ggml::gemmini::residual;
    static_assert(std::is_same_v<DirectStripePayloadHandle,
                                std::shared_ptr<const DirectStripePayload>>,
                  "direct payload ownership must be immutable");
    TimedResidualCapture first(ResidualRoute::cpu_direct);
    first.reset(4, 10, 2, 64, 17);
    first.add_residual(0, 4, 11);
    first.add_residual(1, 5, 12);
    TimedResidualCapture second(ResidualRoute::cpu_direct);
    second.reset(5, 12, 2, 64, 17);
    second.add_residual(0, 6, 13);
    second.add_residual(1, 7, 14);
    std::vector<DirectStripePayloadHandle> inputs{
        first.finish().direct, second.finish().direct};

    RmdStatus status = RmdStatus::success;
    DirectStripePayloadHandle identity = slice_direct_payloads(inputs, 10, 12, 4, status);
    if (!check(status == RmdStatus::success && identity == inputs[0],
               "exact direct slice preserves payload identity")) {
        return false;
    }
    DirectStripePayloadHandle slice = slice_direct_payloads(inputs, 11, 13, 9, status);
    if (!check(status == RmdStatus::success && slice != nullptr,
               "direct payload slice succeeds") ||
        !check(slice->row_begin == 11 && slice->row_count == 2,
               "direct payload slice identity")) {
        return false;
    }
    return check(slice->events.size() == 2, "slice keeps matching events") &&
        check(slice->events[0] == ResidualEvent{0, 5, 12}, "slice renormalizes first local row") &&
        check(slice->events[1] == ResidualEvent{1, 6, 13}, "slice renormalizes second local row") &&
        check(inputs[0]->events[1] == ResidualEvent{1, 5, 12},
              "slice does not mutate source ownership");
}


static bool test_direct_payload_validation_rejects_malformed_contracts() {
    using namespace ggml::gemmini::residual;
    auto valid = [] {
        DirectStripePayload payload;
        payload.stripe_id = 1;
        payload.row_begin = 4;
        payload.row_count = 2;
        payload.logical_k = 8;
        payload.logical_j = 3;
        payload.events = {{0, 1, 7}, {1, 2, -9}};
        return payload;
    };

    DirectStripePayload malformed = valid();
    malformed.row_count = 0;
    bool ok = check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
                    "direct validator rejects zero dimensions");
    malformed = valid();
    malformed.row_begin = std::numeric_limits<size_t>::max();
    ok = check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
               "direct validator rejects row interval overflow") && ok;
    malformed = valid();
    malformed.events[0].local_row = malformed.row_count;
    ok = check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
               "direct validator rejects out-of-range row") && ok;
    malformed = valid();
    malformed.events[0].original_k = malformed.logical_k;
    ok = check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
               "direct validator rejects out-of-range K") && ok;
    malformed = valid();
    malformed.events[0].residual = 0;
    ok = check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
               "direct validator rejects zero residual") && ok;
    malformed = valid();
    std::swap(malformed.events[0], malformed.events[1]);
    ok = check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
               "direct validator rejects unsorted keys") && ok;
    malformed = valid();
    malformed.events[1] = malformed.events[0];
    return check(validate_direct_payload(malformed) == RmdStatus::invalid_packet,
                 "direct validator rejects duplicate keys") && ok;
}

static bool test_exact_slice_validates_payload_and_dimensions() {
    using namespace ggml::gemmini::residual;
    auto malformed = std::make_shared<DirectStripePayload>();
    malformed->stripe_id = 7;
    malformed->row_begin = 10;
    malformed->row_count = 2;
    malformed->logical_k = 8;
    malformed->logical_j = 3;
    malformed->events = {{1, 2, 4}, {0, 1, 3}};
    RmdStatus status = RmdStatus::success;
    const auto exact = slice_direct_payloads({malformed}, 10, 12, 7, status);
    if (!check(exact == nullptr && status == RmdStatus::invalid_packet,
               "exact slice validates before returning identity")) {
        return false;
    }

    auto first = std::make_shared<DirectStripePayload>();
    first->stripe_id = 1;
    first->row_begin = 0;
    first->row_count = 2;
    first->logical_k = 8;
    first->logical_j = 3;
    first->events = {{0, 1, 2}};
    auto second = std::make_shared<DirectStripePayload>();
    second->stripe_id = 2;
    second->row_begin = 2;
    second->row_count = 2;
    second->logical_k = 16;
    second->logical_j = 3;
    second->events = {{0, 2, 3}};
    const auto mixed = slice_direct_payloads({first, second}, 1, 3, 9, status);
    return check(mixed == nullptr && status == RmdStatus::invalid_packet,
                 "slice rejects overlapping payloads with mixed dimensions");
}

static bool test_direct_builder_rejects_row_interval_overflow() {
    using namespace ggml::gemmini::residual;
    DirectStripeBuilder builder;
    builder.reset(0, std::numeric_limits<size_t>::max(), 2, 8, 3);
    return check(builder.status() == RmdStatus::overflow,
                 "direct builder reset rejects row interval overflow") &&
        check(builder.finish() == nullptr, "overflow builder cannot finish");
}

enum class TestSelection {
    all,
    happy_table,
    failure_table,
    direct_happy,
    direct_failure,
    compact_happy,
    compact_failure,
    radix_happy,
    radix_failure,
    invalid,
};

static TestSelection parse_selection(int argc, char ** argv) {
    if (argc == 1) return TestSelection::all;
    if (argc != 2) return TestSelection::invalid;
    if (std::strcmp(argv[1], "--case=happy-table") == 0) return TestSelection::happy_table;
    if (std::strcmp(argv[1], "--case=failure-table") == 0) return TestSelection::failure_table;
    if (std::strcmp(argv[1], "--case=direct-happy") == 0) return TestSelection::direct_happy;
    if (std::strcmp(argv[1], "--case=direct-failure") == 0) return TestSelection::direct_failure;
    if (std::strcmp(argv[1], "--case=compact-happy") == 0) return TestSelection::compact_happy;
    if (std::strcmp(argv[1], "--case=compact-failure") == 0) return TestSelection::compact_failure;
    if (std::strcmp(argv[1], "--case=radix-happy") == 0) return TestSelection::radix_happy;
    if (std::strcmp(argv[1], "--case=radix-failure") == 0) return TestSelection::radix_failure;
    return TestSelection::invalid;
}

int main(int argc, char ** argv) {
    const TestSelection selection = parse_selection(argc, argv);
    if (selection == TestSelection::invalid) {
        std::fputs(
            "usage: test-gemmini-rmd [--case=happy-table|--case=failure-table|"
            "--case=direct-happy|--case=direct-failure|--case=compact-happy|"
            "--case=compact-failure|--case=radix-happy|--case=radix-failure]\n",
            stderr);
        return 2;
    }

    // Native H0 readers use the FP16 tables initialized by ggml_init.
    ggml_context * context = ggml_init({0, nullptr, true});
    if (!check(context != nullptr, "GGML initialization succeeds")) return 1;
    ggml_free(context);

    bool ok = true;
    if (selection == TestSelection::all || selection == TestSelection::happy_table) {
        ok = test_width_native_radix_happy_boundaries() && ok;
        ok = test_width_native_compose_and_expand() && ok;
        ok = test_top_carry_lane_preserves_sparse_mask() && ok;
        ok = test_EXPLICIT_BLOCK_ID_DIM_PADDING() && ok;
        ok = test_empty_residual_is_empty_success() && ok;
        ok = test_cpu_capture_is_canonical_and_packet_free() && ok;
        ok = test_ws_capture_preserves_packet_contract() && ok;
        ok = test_empty_capture_and_single_sink_selection() && ok;
        ok = test_direct_payload_slicing_and_ownership() && ok;
        ok = test_weight_capability_happy_table() && ok;
    }
    if (selection == TestSelection::all || selection == TestSelection::failure_table) {
        ok = test_width_native_out_of_int32_compose_is_atomic() && ok;
        ok = test_width_native_malformed_compose_is_atomic() && ok;
        ok = test_padding_overflow_fails() && ok;
        ok = test_direct_payload_validation_rejects_malformed_contracts() && ok;
        ok = test_exact_slice_validates_payload_and_dimensions() && ok;
        ok = test_direct_builder_rejects_row_interval_overflow() && ok;
        ok = test_weight_capability_failure_table() && ok;
    }
    if (selection == TestSelection::all || selection == TestSelection::direct_happy) {
        ok = test_block_direct_weight_families() && ok;
        ok = test_direct_oracle_happy_matrix() && ok;
    }
    if (selection == TestSelection::all || selection == TestSelection::direct_failure) {
        ok = test_direct_failure_matrix() && ok;
    }
    if (selection == TestSelection::all || selection == TestSelection::compact_happy) {
        ok = test_block_scale_before_sum() && ok;
        ok = test_compact_oracle_happy_matrix() && ok;
        for (size_t rows : {size_t{1}, size_t{2}, size_t{8},
                            kArrayDim - 1, kArrayDim, kArrayDim + 1}) {
            ok = test_full_int32_compact_direct_agreement(rows) && ok;
        }
        ok = test_shared_weight_preparation() && ok;
    }
    if (selection == TestSelection::all || selection == TestSelection::compact_failure) {
        ok = test_block_correction_failure_atomicity() && ok;
        ok = test_compact_failure_matrix() && ok;
        ok = test_streaming_assembler_boundaries() && ok;
    }
    if (selection == TestSelection::radix_happy) {
        ok = test_width_native_radix_happy_boundaries() && ok;
        ok = test_width_native_compose_and_expand() && ok;
        ok = test_top_carry_lane_preserves_sparse_mask() && ok;
    }
    if (selection == TestSelection::radix_failure) {
        ok = test_width_native_out_of_int32_compose_is_atomic() && ok;
        ok = test_width_native_malformed_compose_is_atomic() && ok;
    }
    if (ok) {
        const char * message =
            selection == TestSelection::direct_happy ?
                "PASS: matched-width CPU-direct oracle matrix" :
            selection == TestSelection::direct_failure ?
                "PASS: matched-width CPU-direct failure matrix" :
            selection == TestSelection::compact_happy ?
                "PASS: matched-width compact software oracle matrix" :
            selection == TestSelection::compact_failure ?
                "PASS: matched-width compact failure matrix" :
            selection == TestSelection::radix_happy ?
                "PASS: width-native radix full int32 boundaries" :
            selection == TestSelection::radix_failure ?
                "PASS: width-native radix rejection and malformed metadata atomicity" :
            selection == TestSelection::failure_table ?
                "PASS: RMD residual weight failure table" :
            selection == TestSelection::happy_table ?
                "PASS: RMD residual weight happy table" :
                "PASS: RMD residual weight plus matched-width direct and compact matrices";
        std::puts(message);
    }
    return ok ? 0 : 1;
}
