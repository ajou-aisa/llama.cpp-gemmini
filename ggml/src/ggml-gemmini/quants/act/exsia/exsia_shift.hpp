#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ggml::gemmini::quants::act::exsia::detail {
static inline int16_t exp_to_theta(int16_t e, int16_t rho) {
    const int16_t neg_inf = std::numeric_limits<int16_t>::min();

    if (e == neg_inf)
        return neg_inf;

    return static_cast<int16_t>(static_cast<int>(e) - static_cast<int>(rho));
}

static int32_t round_shift_right_i32(int32_t q, int shift) {
    if (shift <= 0)
        return q;
    if (shift >= 32)
        return 0;

    const int64_t magnitude = q < 0 ? -static_cast<int64_t>(q) : q;
    const int64_t quotient = magnitude >> shift;
    const int64_t remainder = magnitude & ((int64_t{1} << shift) - 1);
    const int64_t half = int64_t{1} << (shift - 1);
    const int64_t rounded = quotient +
        int(remainder > half || (remainder == half && (quotient & 1) != 0));
    return static_cast<int32_t>(q < 0 ? -rounded : rounded);
}

static inline int32_t shift_q_i32(int32_t q, int16_t delta_theta) {
    if (delta_theta > 0) {
        const int      shift    = std::min<int>(delta_theta, 31);
        const bool     negative = q < 0;
        const uint64_t magnitude =
            negative ? static_cast<uint64_t>(-static_cast<int64_t>(q)) : static_cast<uint64_t>(q);
        const uint64_t shifted        = magnitude << shift;
        const uint64_t negative_limit = uint64_t{1} << 31;
        if (!negative)
            return shifted > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())
                       ? std::numeric_limits<int32_t>::max()
                       : static_cast<int32_t>(shifted);
        if (shifted >= negative_limit)
            return std::numeric_limits<int32_t>::min();
        return -static_cast<int32_t>(shifted);
    }

    if (delta_theta < 0) {
        const int shift = -static_cast<int>(delta_theta);
        return round_shift_right_i32(q, shift);
    }

    return q;
}
} // namespace ggml::gemmini::quants::act::exsia::detail
