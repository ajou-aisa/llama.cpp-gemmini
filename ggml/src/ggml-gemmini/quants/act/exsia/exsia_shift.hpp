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
            const bool negative = q < 0;
            const uint64_t magnitude =
                negative ? static_cast<uint64_t>(-static_cast<int64_t>(q))
                         : static_cast<uint64_t>(q);
            const uint64_t shifted = magnitude << shift;
            const uint64_t negative_limit = uint64_t{1} << 31;
            if (!negative)
              return shifted > static_cast<uint64_t>(
                                   std::numeric_limits<int32_t>::max())
                         ? std::numeric_limits<int32_t>::max()
                         : static_cast<int32_t>(shifted);
            if (shifted >= negative_limit)
              return std::numeric_limits<int32_t>::min();
            return -static_cast<int32_t>(shifted);
        }

        if (delta_theta < 0)
        {
            const int shift = std::min<int>(-delta_theta, 31);
            return round_shift_right_i32(q, shift);
        }

        return q;
    }
}
