// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Orca Contributors
//
// FP16 conversion utilities (ggml-free)

#include "fp16_util.hpp"

#include <cstring>

namespace ggml::gemmini::quants { // IEEE 754 fp16 format:
// - Sign: 1 bit
// - Exponent: 5 bits (bias 15)
// - Mantissa: 10 bits
//
// Special values:
// - Exponent 0x00: zero or subnormal
// - Exponent 0x1F: infinity or NaN

float fp16_to_fp32(uint16_t h)
{
    // Extract fields
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp16 = (h >> 10) & 0x1F;
    uint32_t frac16 = h & 0x3FF;

    uint32_t f32_bits;

    if (exp16 == 0)
    {
        // Zero or subnormal
        if (frac16 == 0)
        {
            // Zero
            f32_bits = (sign << 31);
        }
        else
        {
            // Subnormal: convert to normalized fp32
            // Normalize mantissa and adjust exponent
            int e = -14;
            uint32_t m = frac16;
            while ((m & 0x400) == 0)
            {
                m <<= 1;
                e--;
            }
            m &= 0x3FF; // Remove leading 1
            uint32_t exp32 = (e + 127) & 0xFF;
            f32_bits = (sign << 31) | (exp32 << 23) | (m << 13);
        }
    }
    else if (exp16 == 0x1F)
    {
        // Infinity or NaN
        f32_bits = (sign << 31) | (0xFF << 23) | (frac16 << 13);
    }
    else
    {
        // Normalized number
        uint32_t exp32 = exp16 - 15 + 127; // Rebias exponent
        f32_bits = (sign << 31) | (exp32 << 23) | (frac16 << 13);
    }

    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

uint16_t fp32_to_fp16(float f)
{
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &f, sizeof(float));

    uint32_t sign = (f32_bits >> 31) & 0x1;
    uint32_t exp32 = (f32_bits >> 23) & 0xFF;
    uint32_t frac32 = f32_bits & 0x7FFFFF;

    uint16_t h;

    if (exp32 == 0)
    {
        // Zero or subnormal fp32 -> zero fp16
        h = static_cast<uint16_t>(sign << 15);
    }
    else if (exp32 == 0xFF)
    {
        // Infinity or NaN
        h = static_cast<uint16_t>((sign << 15) | (0x1F << 10) | (frac32 >> 13));
    }
    else
    {
        // Normalized number
        int exp16 = static_cast<int>(exp32) - 127 + 15;

        if (exp16 <= 0)
        {
            // Underflow -> zero
            h = static_cast<uint16_t>(sign << 15);
        }
        else if (exp16 >= 0x1F)
        {
            // Overflow -> infinity
            h = static_cast<uint16_t>((sign << 15) | (0x1F << 10));
        }
        else
        {
            // Normal conversion
            uint16_t frac16 = static_cast<uint16_t>(frac32 >> 13);
            h = static_cast<uint16_t>((sign << 15) | (exp16 << 10) | frac16);
        }
    }

    return h;
    }
} // namespace ggml::gemmini::quants
