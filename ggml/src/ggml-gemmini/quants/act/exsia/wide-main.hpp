#pragma once

#include <cstdint>

namespace ggml::gemmini::quants::act::exsia {

template <int Bits>
bool split_wide_main(int32_t value, int8_t & low, int8_t & high) {
    static_assert(Bits == 4 || Bits == 8);
    constexpr int32_t radix = int32_t{1} << Bits;
    constexpr int32_t half = radix / 2;
    constexpr int32_t storage_min = -(int32_t{1} << (2 * Bits - 1));
    constexpr int32_t digit_max = (half - 1) * (radix + 1);
    // Intersect the storage range with the two signed-digit range before narrowing.
    if (value < storage_min || value > digit_max) return false;
    const int32_t lo = int32_t((uint32_t(value) + uint32_t(half)) & uint32_t(radix - 1)) - half;
    low = int8_t(lo);
    high = int8_t((value - lo) / radix);
    return true;
}

} // namespace ggml::gemmini::quants::act::exsia
