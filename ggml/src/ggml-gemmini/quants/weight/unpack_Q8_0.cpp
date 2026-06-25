// Q8_0 weight unpacking implementation (ggml-free)

#include "unpack_Q8_0.hpp"
#include "../../ggml-gemmini-config.hpp"
#include "../../ggml-gemmini-args.h"
#include "../common/tensor_util.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace ggml::gemmini::quants { namespace {

constexpr float q8_0_r_scale_bins = 255.0f;

uint8_t quantize_scale_code(float s_b, float s_rf, uint16_t R) {
    if (!std::isfinite(s_b) || !std::isfinite(s_rf) || s_rf <= 0.0f)
        return 0;

    const double c_eff = std::round(static_cast<double>(s_b) / static_cast<double>(s_rf));
    const double shifted_code = c_eff - static_cast<double>(R);
    const double clamped = std::max(0.0, std::min(255.0, shifted_code));
    return static_cast<uint8_t>(clamped);
}

void set_constant_scale(float s_b, size_t blocks_per_row, uint8_t *dst_c_b, float &dst_s_rf, uint16_t &dst_R) {
    if (s_b > 0.0f && std::isfinite(s_b)) {
        dst_s_rf = s_b;
        dst_R = 1;
    } else {
        dst_s_rf = 0.0f;
        dst_R = 0;
    }

    std::memset(dst_c_b, 0, blocks_per_row * sizeof(uint8_t));
}

bool quantize_row_scales_q80_r(
    const float *row_scales,
    size_t blocks_per_row,
    uint8_t *dst_c_b,
    float &dst_s_rf,
    uint16_t &dst_R) {
    float min_s = std::numeric_limits<float>::max();
    float max_s = std::numeric_limits<float>::lowest();

    for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const float s_b = row_scales[block_idx];
        if (!std::isfinite(s_b))
            return false;

        min_s = std::min(min_s, s_b);
        max_s = std::max(max_s, s_b);
    }

    const float scale_range = max_s - min_s;
    if (!std::isfinite(scale_range) || scale_range <= 0.0f) {
        set_constant_scale(min_s, blocks_per_row, dst_c_b, dst_s_rf, dst_R);
        return true;
    }

    dst_s_rf = scale_range / q8_0_r_scale_bins;
    if (!std::isfinite(dst_s_rf) || dst_s_rf <= 0.0f) {
        set_constant_scale(min_s, blocks_per_row, dst_c_b, dst_s_rf, dst_R);
        return true;
    }

    const double r_val = std::round(static_cast<double>(min_s) / static_cast<double>(dst_s_rf));
    dst_R = static_cast<uint16_t>(std::min(65535.0, std::max(0.0, r_val)));

    for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx)
        dst_c_b[block_idx] = quantize_scale_code(row_scales[block_idx], dst_s_rf, dst_R);

    return true;
}

bool quantize_stripe_scales_q80_r(
    const float *stripe_scales,
    size_t blocks_per_row,
    size_t num_rows,
    uint8_t *dst_c_b,
    float *dst_s_rf,
    uint16_t *dst_R,
    float &dst_s_rf_stripe,
    uint16_t &dst_R_stripe) {
    float min_s = std::numeric_limits<float>::max();
    float max_s = std::numeric_limits<float>::lowest();

    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        const float *row_scales = stripe_scales + row_idx * blocks_per_row;
        for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const float s_b = row_scales[block_idx];
            if (!std::isfinite(s_b))
                return false;

            min_s = std::min(min_s, s_b);
            max_s = std::max(max_s, s_b);
        }
    }

    const float scale_range = max_s - min_s;
    if (!std::isfinite(scale_range) || scale_range <= 0.0f) {
        for (size_t row_idx = 0; row_idx < num_rows; ++row_idx)
            set_constant_scale(min_s, blocks_per_row, dst_c_b + row_idx * blocks_per_row, dst_s_rf[row_idx], dst_R[row_idx]);
        dst_s_rf_stripe = dst_s_rf[0];
        dst_R_stripe = dst_R[0];
        return true;
    }

    const float s_rf = scale_range / q8_0_r_scale_bins;
    if (!std::isfinite(s_rf) || s_rf <= 0.0f) {
        for (size_t row_idx = 0; row_idx < num_rows; ++row_idx)
            set_constant_scale(min_s, blocks_per_row, dst_c_b + row_idx * blocks_per_row, dst_s_rf[row_idx], dst_R[row_idx]);
        dst_s_rf_stripe = dst_s_rf[0];
        dst_R_stripe = dst_R[0];
        return true;
    }

    const double r_val = std::round(static_cast<double>(min_s) / static_cast<double>(s_rf));
    const uint16_t R = static_cast<uint16_t>(std::min(65535.0, std::max(0.0, r_val)));

    dst_s_rf_stripe = s_rf;
    dst_R_stripe = R;

    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        const float *row_scales = stripe_scales + row_idx * blocks_per_row;
        uint8_t *row_codes = dst_c_b + row_idx * blocks_per_row;
        dst_s_rf[row_idx] = s_rf;
        dst_R[row_idx] = R;

        for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx)
            row_codes[block_idx] = quantize_scale_code(row_scales[block_idx], s_rf, R);
    }

    return true;
}

}

// ============================================================================
// Q8_0 Unpacking Implementation
// ============================================================================

bool unpack_q8_0(
    const ggml_tensor *src,
    ggml_gemmini_args_t &args,
    size_t block_size,
    int8_t *dst_jxk,
    size_t dst_stride_elems,
    float *dst_scales,
    Q80OutputLayout layout,
    UnpackQ80Result *out_meta
) {
    // Validate inputs
    if (!src || src->type != GGML_TYPE_Q8_0 || !dst_jxk || !dst_scales)
        return false;

#if GGML_GEMMINI_COMPUTE_TYPE == 0 && GGML_GEMMINI_ACTIVATION_QUANT == 0
    static_assert(GGML_GEMMINI_BLOCK_SIZE == 32, "GGML_GEMMINI_BLOCK_SIZE must be 32 for INT + EXSIA Q8_0_R weight format");
#endif
    if (block_size != 32)
        return false;

    // Normalize dimensions (treat 0 as 1)
    const block_q8_0 *block_base = ggml::gemmini::weight_block_base(src);
    if (!block_base)
        return false;

    const size_t nb0 = src->nb[0];
    const size_t nb1 = src->nb[1];
    const size_t nb2 = src->nb[2];
    const size_t nb3 = src->nb[3];
    const int64_t dim_k = src->ne[0];
    const int64_t dim_j = (src->ne[1] > 0) ? src->ne[1] : 1;
    const int64_t dim_z = (src->ne[2] > 0) ? src->ne[2] : 1;
    const int64_t dim_w = (src->ne[3] > 0) ? src->ne[3] : 1;

    // Validate K dimension
    if (dim_k <= 0 || (dim_k % static_cast<int64_t>(block_size)) != 0)
        return false;

    // Compute metadata (avoid overflow on logical_rows)
    size_t blocks_K = static_cast<size_t>(dim_k) / block_size;
    const __int128 logical_rows_128 =
        static_cast<__int128>(dim_j) * static_cast<__int128>(dim_z) * static_cast<__int128>(dim_w);
    if (logical_rows_128 > static_cast<__int128>(std::numeric_limits<size_t>::max()))
        return false;
    size_t logical_rows = static_cast<size_t>(logical_rows_128);

    // Validate stride
    if (layout == Q80OutputLayout::JXK_ROW_MAJOR &&
        dst_stride_elems < static_cast<size_t>(dim_k)) {
        return false;
    }
    if (layout == Q80OutputLayout::KXJ_ROW_MAJOR &&
        dst_stride_elems < logical_rows) {
        return false;
    }
    if (layout != Q80OutputLayout::JXK_ROW_MAJOR &&
        layout != Q80OutputLayout::KXJ_ROW_MAJOR) {
        return false;
    }
    if (nb0 == 0 || nb0 < sizeof(BlockQ8_0))
        return false;

    const char *effective_base = reinterpret_cast<const char *>(block_base);

    // ========================================================================
    // Main unpacking loop
    // ========================================================================
    //
    // Iteration order: (iw, iz, iy) -> maps to row index
    // For each row:
    //   - Get row pointer to Q8_0 blocks
    //   - Unpack all K elements (iterating over blocks)
    //   - Extract scales from each block
    //
    // Output layout:
    //   dst_jxk[row_idx][k] = quantized value at (row_idx, k)
    //   dst_scales[row_idx][blk] = scale for block at (row_idx, blk)

    size_t row_idx = 0;

    for (int64_t iw = 0; iw < dim_w; ++iw) {
        for (int64_t iz = 0; iz < dim_z; ++iz) {
            for (int64_t iy = 0; iy < dim_j; ++iy) {
                // Compute row pointer for this (iy, iz, iw) coordinate
                // Equivalent to: base + offs + iw*nb3 + iz*nb2 + iy*nb1
                const char *row_ptr_bytes = effective_base
                    + iw * nb3
                    + iz * nb2
                    + iy * nb1;

#ifndef NDEBUG
                // Q8_0 blocks require 2-byte alignment for safe fp16 reads.
                assert((reinterpret_cast<uintptr_t>(row_ptr_bytes) & 0x1u) == 0);
                // ggml Q8_0 blocks should be tightly packed.
                assert(nb0 >= sizeof(BlockQ8_0));
#endif

                // Unpack all K elements for this row
                float *row_scales = dst_scales + row_idx * blocks_K;
                for (size_t blk = 0; blk < blocks_K; ++blk) {
                    // Use byte-wise access + memcpy to avoid strict-aliasing/unaligned UB.
                    const char *blk_ptr = row_ptr_bytes + blk * nb0;

                    // Extract and convert scale (memcpy avoids unaligned read UB)
                    uint16_t d_raw = 0;
                    std::memcpy(&d_raw, blk_ptr + offsetof(BlockQ8_0, d), sizeof(d_raw));
                    row_scales[blk] = fp16_to_fp32(d_raw);

                    if (layout == Q80OutputLayout::JXK_ROW_MAJOR) {
                        int8_t *row_dst = dst_jxk + row_idx * dst_stride_elems;
                        // Unpack quantized values (32 elements per block)
                        std::memcpy(row_dst + blk * block_size,
                                    blk_ptr + offsetof(BlockQ8_0, qs),
                                    block_size);
                    } else {
                        // Write transposed KxJ row-major directly.
                        const int8_t *src_q = reinterpret_cast<const int8_t *>(blk_ptr + offsetof(BlockQ8_0, qs));
                        for (size_t t = 0; t < block_size; ++t) {
                            const size_t k_idx = blk * block_size + t;
                            dst_jxk[k_idx * dst_stride_elems + row_idx] = src_q[t];
                        }
                    }
                }

                ++row_idx;
            }
        }
    }

    // Store metadata if requested
    if (out_meta) {
        out_meta->blocks_K = blocks_K;
        out_meta->logical_cols = logical_rows;
        out_meta->block_size = block_size;
    }

    return true;
}

bool unpack_q80_r_weight(
    const ggml_tensor *src_q80,
    ggml_gemmini_args_t &args,
    std::vector<int8_t> &dst_qs,
    std::vector<uint8_t> &dst_c_b,
    std::vector<float> &dst_s_rf,
    std::vector<uint16_t> &dst_R,
    std::vector<float> *dst_s_rf_stripe,
    std::vector<uint16_t> *dst_R_stripe) {
    if (!src_q80 || src_q80->type != GGML_TYPE_Q8_0)
        return false;

    const block_q8_0 *block_base = ggml::gemmini::weight_block_base(src_q80);
    if (!block_base || src_q80->ne[0] <= 0 || src_q80->ne[0] % QK8_0 != 0)
        return false;

    const int64_t dim_k = src_q80->ne[0];
    const int64_t dim_j = src_q80->ne[1] > 0 ? src_q80->ne[1] : 1;
    const int64_t dim_z = src_q80->ne[2] > 0 ? src_q80->ne[2] : 1;
    const int64_t dim_w = src_q80->ne[3] > 0 ? src_q80->ne[3] : 1;
    const __int128 logical_rows_128 =
        static_cast<__int128>(dim_j) * static_cast<__int128>(dim_z) * static_cast<__int128>(dim_w);
    if (logical_rows_128 <= 0 || logical_rows_128 > static_cast<__int128>(std::numeric_limits<size_t>::max()))
        return false;

    const size_t logical_rows = static_cast<size_t>(logical_rows_128);
    const size_t k_elems = static_cast<size_t>(dim_k);
    const size_t blocks_per_row = k_elems / static_cast<size_t>(QK8_0);
    if (blocks_per_row == 0 || logical_rows > std::numeric_limits<size_t>::max() / k_elems)
        return false;

    dst_qs.assign(logical_rows * k_elems, 0);
    std::vector<float> block_scales(logical_rows * blocks_per_row, 0.0f);
    UnpackQ80Result meta {};
    const bool unpacked = unpack_q8_0(
        src_q80,
        args,
        QK8_0,
        dst_qs.data(),
        k_elems,
        block_scales.data(),
        Q80OutputLayout::JXK_ROW_MAJOR,
        &meta);
    if (!unpacked)
        return false;

    dst_c_b.assign(meta.logical_cols * meta.blocks_K, 0);
    dst_s_rf.assign(meta.logical_cols, 0.0f);
    dst_R.assign(meta.logical_cols, 0u);

    const size_t stripe_J = args.stripe_J > 1 ? args.stripe_J : 1;
    const bool use_stripe_metadata = stripe_J > 1;
    if (use_stripe_metadata) {
        if (!dst_s_rf_stripe || !dst_R_stripe)
            return false;

        const size_t num_stripes = (meta.logical_cols + stripe_J - 1) / stripe_J;
        dst_s_rf_stripe->assign(num_stripes, 0.0f);
        dst_R_stripe->assign(num_stripes, 0u);

        for (size_t stripe_idx = 0; stripe_idx < num_stripes; ++stripe_idx) {
            const size_t row_begin = stripe_idx * stripe_J;
            const size_t stripe_rows = std::min(stripe_J, meta.logical_cols - row_begin);
            if (!quantize_stripe_scales_q80_r(
                block_scales.data() + row_begin * meta.blocks_K,
                meta.blocks_K,
                stripe_rows,
                dst_c_b.data() + row_begin * meta.blocks_K,
                dst_s_rf.data() + row_begin,
                dst_R.data() + row_begin,
                (*dst_s_rf_stripe)[stripe_idx],
                (*dst_R_stripe)[stripe_idx])) {
                return false;
            }
        }
    } else {
        if (dst_s_rf_stripe)
            dst_s_rf_stripe->clear();
        if (dst_R_stripe)
            dst_R_stripe->clear();

        for (size_t row_idx = 0; row_idx < meta.logical_cols; ++row_idx) {
            if (!quantize_row_scales_q80_r(
                block_scales.data() + row_idx * meta.blocks_K,
                meta.blocks_K,
                dst_c_b.data() + row_idx * meta.blocks_K,
                dst_s_rf[row_idx],
                dst_R[row_idx])) {
                return false;
            }
        }
    }

    args.B = reinterpret_cast<elem_t *>(dst_qs.data());
    args.sB = k_elems;
    args.B_blocks = block_base;
    args.B_scales = nullptr;
    args.c_b = dst_c_b.data();
    args.s_rf = dst_s_rf.data();
    args.R = dst_R.data();
    args.blocks_per_row = meta.blocks_K;
    args.blocks_K = meta.blocks_K;
    args.blocks_J = meta.logical_cols;
    args.blocks_I = meta.logical_cols;
    args.block_size_k = QK8_0;


    return true;
}
} // namespace ggml::gemmini::quants
