// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Orca Contributors
//
// Q8_H1 weight quantization - Pure algorithm layer (ggml-free)

#pragma once

#include "unpack_Q8_0.hpp"

#include <cstdint>

namespace ggml::gemmini::quants
{
using block_q8_0 = BlockQ8_0;

/// Row-wise Q8_H1 storage view.
///
/// c_b and qs are caller-owned arrays sized for the row:
/// - c_b: blocks_per_row uint8 scale codes
/// - qs: blocks_per_row * 32 int8 quantized values
struct BlockQ8_H1 {
    float s_rf = 0.0f;
    uint16_t R = 0;
    uint8_t *c_b = nullptr;
    int8_t *qs = nullptr;
};

/// Convert one row of Q8_0 blocks to Q8_H1 row metadata and payload.
///
/// The input Q8_0 fp16 scales are double-quantized per row:
///   s_rf = (max_scale - min_scale) / 256
///   R = round(min_scale / s_rf)
///   c_b[i] = clamp(round(s_b[i] / s_rf) - R, 0, 255)
///
/// Scale recovery is performed with recover_block_scale().
///
/// @return true on success, false on validation failure.
bool quantize_row_q8_h1(
    const block_q8_0 *src_blocks,
    int blocks_per_row,
    BlockQ8_H1 *dst
);

/// Recover the fp32 scale for a block in a Q8_H1 row.
float recover_block_scale(const BlockQ8_H1 *block, int block_idx);
} // namespace ggml::gemmini::quants
