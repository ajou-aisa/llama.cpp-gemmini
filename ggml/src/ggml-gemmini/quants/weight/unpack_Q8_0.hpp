// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Orca Contributors
//
// Q8_0 block layout shared by Q8_H1 reprocessing.

#pragma once

#include <cstdint>

namespace ggml::gemmini::quants
{
/// Q8_0 block structure: 32 int8 quantized values + 1 fp16 scale
/// Layout matches ggml's block_q8_0 (verified by adapter layer)
struct BlockQ8_0 {
    uint16_t d;       ///< Scale factor (IEEE 754 fp16 raw bits)
    int8_t qs[32];    ///< Quantized values (32 elements per block)
};

static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block size must be 34 bytes");
static_assert(alignof(BlockQ8_0) <= 2, "Q8_0 block alignment must be <= 2");

} // namespace ggml::gemmini::quants
