// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Orca Contributors
//
// Q8_0 weight unpacking - Pure algorithm layer (ggml-free)
//
// This module provides ggml-independent Q8_0 block unpacking functionality.
// It operates on plain pointers and strides, with no dependencies on ggml types.

#pragma once

#include "../common/fp16_util.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_tensor;
struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants
{
// ============================================================================
// Q8_0 Block Layout (ggml-independent redefinition)
// ============================================================================

/// Q8_0 block structure: 32 int8 quantized values + 1 fp16 scale
/// Layout matches ggml's block_q8_0 (verified by adapter layer)
struct BlockQ8_0 {
    uint16_t d;       ///< Scale factor (IEEE 754 fp16 raw bits)
    int8_t qs[32];    ///< Quantized values (32 elements per block)
};

static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block size must be 34 bytes");
static_assert(alignof(BlockQ8_0) <= 2, "Q8_0 block alignment must be <= 2");

// ============================================================================
// Unpacking Result Metadata
// ============================================================================

/// Metadata returned by unpack_q8_0()
struct UnpackQ80Result {
    size_t blocks_K = 0;        ///< Number of blocks along K dimension (dim_k / block_size)
    size_t logical_cols = 0;    ///< Total rows (dim_j * dim_z * dim_w) [legacy name: logical_rows]
    size_t block_size = 32;     ///< Block size (always 32 for Q8_0)

    // Layout documentation:
    // - Scale table: [logical_rows][blocks_K] (row-major, J-major)
    //   Access: dst_scales[row_idx * blocks_K + blk]
    //
    // - Packed matrix: [logical_rows][dim_k] (row-major, JxK; no transpose)
    //   Access: dst_jxk[row_idx * dst_stride_elems + k]
};

enum class Q80OutputLayout : uint8_t {
    JXK_ROW_MAJOR = 0, ///< dst[row=j][col=k], stride >= K
    KXJ_ROW_MAJOR = 1, ///< dst[row=k][col=j], stride >= J
};

// ============================================================================
// Main Unpacking Function
// ============================================================================

/// Unpack Q8_0 tensor to int8 matrix + scale table.
///
/// This function performs the following:
/// 1. Validates dimensions (dim_k must be divisible by block_size)
/// 2. Computes logical_rows = dim_j * dim_z * dim_w (legacy name: logical_cols)
/// 3. Unpacks quantized int8 values from Q8_0 blocks into requested matrix layout
/// 4. Extracts fp16 scales from blocks and converts to fp32
///
/// @param src Input ggml Q8_0 tensor
/// @param args Gemmini args used for base-pointer extraction helpers
/// @param block_size Block size (must be 32 for Q8_0)
/// @param dst_jxk Output int8 matrix buffer
/// @param dst_stride_elems Row stride in elements (depends on @p layout)
/// @param dst_scales Output scale table [logical_rows x blocks_K] (row-major)
/// @param layout Output matrix layout (JxK or KxJ row-major)
/// @param out_meta Optional output metadata (can be nullptr)
/// @return true on success, false on validation failure
bool unpack_q8_0(
    const ggml_tensor *src,
    ggml_gemmini_args_t &args,
    size_t block_size,
    int8_t *dst_jxk,
    size_t dst_stride_elems,
    float *dst_scales,
    Q80OutputLayout layout,
    UnpackQ80Result *out_meta = nullptr
);

bool unpack_q80_r_weight(
    const ggml_tensor *src_q80,
    ggml_gemmini_args_t &args,
    std::vector<int8_t> &dst_qs,
    std::vector<uint8_t> &dst_c_b,
    std::vector<float> &dst_s_rf,
    std::vector<uint16_t> &dst_R,
    std::vector<float> *dst_s_rf_panel,
    std::vector<uint16_t> *dst_R_panel);
} // namespace ggml::gemmini::quants
