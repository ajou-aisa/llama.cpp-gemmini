#pragma once

// Copied from ggml-gemmini/quants/common/math.hpp.

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>

namespace ggml::potal {

// Signed low digit for the supported 4/8/16-bit operands; unsigned conversion preserves low bits.
constexpr int32_t lowdigit(uint64_t value, unsigned bits) {
    const uint32_t sign = uint32_t{1} << (bits - 1);
    return static_cast<int32_t>((value & ((sign << 1) - 1)) ^ sign) - static_cast<int32_t>(sign);
}

inline bool checked_mul_size(size_t lhs, size_t rhs, size_t & out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        return false;

    out = lhs * rhs;
    return true;
}

inline float apply_activation_exponent(float value, int16_t e_t, int16_t m) {
    if (e_t == std::numeric_limits<int16_t>::min())
        return value;

    const int shift = static_cast<int>(e_t) - static_cast<int>(m);
    if (shift == 0)
        return value;

    return std::scalbn(value, shift);
}

} // namespace ggml::potal
