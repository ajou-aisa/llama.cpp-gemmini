#pragma once

#include <cstdint>
#include <cmath>
#include <limits>

namespace ggml::gemmini {

inline float apply_activation_exponent(float value, int16_t e_t, int16_t m) {
    if (e_t == std::numeric_limits<int16_t>::min())
        return value;

    const int shift = static_cast<int>(e_t) - static_cast<int>(m);
    if (shift == 0)
        return value;

    return std::scalbn(value, shift);
}

}
