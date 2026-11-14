#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>

namespace ggml::gemmini::quants::act::exsia::detail {

static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559 &&
              std::numeric_limits<float>::digits == 24, "ExSIA requires IEEE binary32 inputs");

inline uint32_t f32_bits(float value) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline int16_t unbiased_exp_f32(float value) noexcept {
    const uint32_t magnitude = f32_bits(value) & 0x7fffffff;
    const uint32_t exponent = magnitude >> 23;
    if (magnitude == 0 || exponent == 255)
        return std::numeric_limits<int16_t>::min();
    if (exponent != 0)
        return static_cast<int16_t>(static_cast<int>(exponent) - 127);
    int16_t result = -149;
    for (uint32_t remaining = magnitude >> 1; remaining != 0; remaining >>= 1)
        ++result;
    return result;
}

inline int32_t quantize_f32_pot(float value, int16_t theta) noexcept {
    const uint32_t bits = f32_bits(value);
    const uint32_t exponent = (bits >> 23) & 255;
    if (theta == std::numeric_limits<int16_t>::min() || exponent == 255)
        return 0;
    const bool negative = (bits >> 31) != 0;
    const uint32_t mantissa = (bits & 0x7fffff) | (exponent != 0 ? 0x800000 : 0);
    if (mantissa == 0)
        return 0;
    const int shift = (exponent != 0 ? int(exponent) - 150 : -149) - int(theta);
    uint64_t magnitude;
    if (shift >= 0) {
        if (shift > 31)
            return negative ? std::numeric_limits<int32_t>::min() : std::numeric_limits<int32_t>::max();
        magnitude = uint64_t(mantissa) << shift;
    } else {
        const unsigned right = unsigned(-shift);
        if (right > 24) {
            magnitude = 0;
        } else {
            const uint32_t integer = mantissa >> right;
            const uint32_t fraction = mantissa & ((uint32_t{1} << right) - 1);
            const uint32_t half = uint32_t{1} << (right - 1);
            magnitude = integer + uint32_t(fraction > half || (fraction == half && (integer & 1)));
        }
    }
    const uint64_t limit = negative ? uint64_t{1} << 31 : std::numeric_limits<int32_t>::max();
    if (magnitude >= limit)
        return negative ? std::numeric_limits<int32_t>::min() : std::numeric_limits<int32_t>::max();
    return negative ? -int32_t(magnitude) : int32_t(magnitude);
}

// Convert original F32 once at the final stripe scale. The reachable exponent
// range is exact in double; explicit ties-even rounding preserves caller fenv.
inline int32_t quantize_f32_stripe_rn(float value, int16_t theta) noexcept {
    if (theta == std::numeric_limits<int16_t>::min() ||
        (f32_bits(value) & 0x7f800000u) == 0x7f800000u)
        return 0;
    const double scaled = std::ldexp(static_cast<double>(value), -int(theta));
    if (scaled >= double(std::numeric_limits<int32_t>::max()))
        return std::numeric_limits<int32_t>::max();
    if (scaled <= double(std::numeric_limits<int32_t>::min()))
        return std::numeric_limits<int32_t>::min();
    const double magnitude = std::fabs(scaled);
    const double base = std::floor(magnitude);
    const double fraction = magnitude - base;
    const int64_t integer = static_cast<int64_t>(base);
    const int64_t rounded = integer +
        (fraction > 0.5 || (fraction == 0.5 && (integer & 1)));
    return static_cast<int32_t>(scaled < 0 ? -rounded : rounded);
}

} // namespace ggml::gemmini::quants::act::exsia::detail
