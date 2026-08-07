#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

using namespace ggml::gemmini::rmd;

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

bool test_balanced_radix_decomposition() {
    constexpr std::array<int32_t, 13> values = {
        std::numeric_limits<int32_t>::min(), -16777217, -129, -128, -1, 0, 1,
        127, 128, 255, 256, 16777216, 2139062143,
    };
    for (const int32_t value : values) {
        BalancedDigits digits{};
        if (!check(decompose_balanced_radix256(value, digits), "balanced value decomposes") ||
            !check(compose_balanced_radix256(digits) == value, "balanced digits round-trip")) {
            return false;
        }
    }
    return true;
}

bool test_q4_nonzero_fails_explicitly() {
    BalancedDigits digits{};
    digits.digits.fill(1);
    digits.lane_mask = 0x0f;
    const bool decomposed = decompose_balanced_radix256(std::numeric_limits<int32_t>::max(), digits);

    RmdStripeBuilder builder;
    builder.reset(0, 0, 1, 1, 1);
    const bool added = builder.add_residual(0, 0, std::numeric_limits<int32_t>::max());
    return check(!decomposed, "q4 != 0 rejects decomposition") &&
        check(digits.lane_mask == 0, "failed decomposition clears output") &&
        check(!added && builder.status() == RmdStatus::residual_too_wide,
              "builder exposes q4 failure status");
}

bool test_EXPLICIT_BLOCK_ID_DIM_PADDING() {
    RmdStripeBuilder builder;
    builder.reset(7, 9, 3, 64, 17);
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
    ok = check(packet->j_padded == 2 * kArrayDim, "output columns use DIM padding") && ok;
    ok = check(validate_packet(*packet) == RmdStatus::success, "built packet validates") && ok;

    StripePacket malformed = *packet;
    malformed.stacked_activation[first.activation_offset + first.compact_k_count] = 1;
    ok = check(validate_packet(malformed) == RmdStatus::invalid_packet,
               "nonzero K padding rejected") && ok;
    malformed = *packet;
    malformed.stacked_activation[first.activation_offset +
        packet->row_count * first.padded_k_count] = 1;
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

}

int main() {
    const bool ok = test_balanced_radix_decomposition() &&
        test_q4_nonzero_fails_explicitly() &&
        test_EXPLICIT_BLOCK_ID_DIM_PADDING() &&
        test_empty_residual_is_empty_success() &&
        test_padding_overflow_fails();
    if (ok) {
        std::puts("PASS: balanced radix, q4 failure, block boundaries, lane gaps, padding, empty residual, overflow");
    }
    return ok ? 0 : 1;
}
