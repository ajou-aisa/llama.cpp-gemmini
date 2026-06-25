#pragma once

#include <cstdint>

namespace ggml::gemmini::quants
{
    // Convert IEEE 754 fp16 (raw bits) to fp32
    float fp16_to_fp32(uint16_t h);

    // Convert fp32 to IEEE 754 fp16 (raw bits)
    uint16_t fp32_to_fp16(float f);
} // namespace ggml::gemmini::quants
