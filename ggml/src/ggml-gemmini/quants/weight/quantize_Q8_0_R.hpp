// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Orca Contributors
//
// Q8_0_R weight quantization - Pure algorithm layer (ggml-free)

#pragma once

#include "unpack_Q8_0.hpp"

#include <cstdint>

namespace ggml::gemmini::quants
{
using block_q8_0 = BlockQ8_0;

/// Row-wise Q8_0_R storage view.
///
/// c_b and qs are caller-owned arrays sized for the row:
/// - c_b: blocks_per_row uint8 scale codes
/// - qs: blocks_per_row * 32 int8 quantized values
struct BlockQ8_0_R {
    float s_rf = 0.0f;
    uint16_t R = 0;
    uint8_t *c_b = nullptr;
    int8_t *qs = nullptr;
};

/// Convert one row of Q8_0 blocks to Q8_0_R row metadata and payload.
///
/// The input Q8_0 fp16 scales are double-quantized per row:
///   s_rf = (max_scale - min_scale) / 256
///   R = round(min_scale / s_rf)
///   c_b[i] = clamp(round(s_b[i] / s_rf) - R, 0, 255)
///
/// Scale recovery is performed with recover_block_scale().
///
/// @return true on success, false on validation failure.
bool quantize_row_q8_0_r(
    const block_q8_0 *src_blocks,
    int blocks_per_row,
    BlockQ8_0_R *dst
);

/// Convert a panel of Q8_0 rows to Q8_0_R rows using one shared panel scale range.
///
/// The input Q8_0 fp16 scales are double-quantized across the full panel:
///   s_rf = (max_panel_scale - min_panel_scale) / 255
///   R = round(min_panel_scale / s_rf)
///   c_b[row, block] = clamp(round(s_b[row, block] / s_rf) - R, 0, 255)
///
/// dst contains num_rows caller-owned row views. dst_s_rf_panel and dst_R_panel
/// receive the shared panel metadata.
///
/// @return true on success, false on validation failure.
bool quantize_panel_q8_0_r(
    const block_q8_0 *src_blocks,
    int blocks_per_row,
    int num_rows,
    BlockQ8_0_R *dst,
    float *dst_s_rf_panel,
    uint16_t *dst_R_panel
);

/// Recover the fp32 scale for a block in a Q8_0_R row.
float recover_block_scale(const BlockQ8_0_R *block, int block_idx);
} // namespace ggml::gemmini::quants
