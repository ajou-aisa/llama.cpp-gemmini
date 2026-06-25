#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ggml::gemmini::quants::act::exsia::detail
{
    static int32_t round_shift_right_i32(
        int32_t q,
        int shift)
    {
        if (shift <= 0)
            return q;

        const int64_t x = q;
        const int64_t offset =
            int64_t{1} << (shift - 1);

        if (x >= 0)
            return static_cast<int32_t>(
                (x + offset) >> shift);

        return static_cast<int32_t>(
            -(((-x) + offset) >> shift));
    }

    static int32_t shift_q_i32(int32_t q, int16_t delta_theta)
    {
        if (delta_theta > 0)
        {
            const int shift = std::min<int>(delta_theta, 31);
            const int64_t shifted = static_cast<int64_t>(q) << shift;
            return static_cast<int32_t>(std::clamp<int64_t>(
                shifted,
                std::numeric_limits<int32_t>::min(),
                std::numeric_limits<int32_t>::max()));
        }

        if (delta_theta < 0)
        {
            const int shift = std::min<int>(-delta_theta, 31);
            return round_shift_right_i32(q, shift);
        }

        return q;
    }
}
