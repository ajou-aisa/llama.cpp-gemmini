#pragma once

#include <cstdint>

namespace ggml::gemmini::quants {

// The caller clips to the int32 range first. Truncation and exact fractional
// comparison avoid both caller-fenv dependence and abs(x) + 0.5 double rounding.
inline int32_t round_nearest_even_i32(double value) {
    const int32_t truncated = static_cast<int32_t>(value);
    const double fraction = value - static_cast<double>(truncated);
    const bool odd = truncated % 2 != 0;
    return truncated + int(fraction > 0.5 || (fraction == 0.5 && odd))
                     - int(fraction < -0.5 || (fraction == -0.5 && odd));
}

} // namespace ggml::gemmini::quants
