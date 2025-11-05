#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>

#include "ggml.h"

#ifndef GGML_COMMON_DECL
#define GGML_GEMMINI_ARGS_DEFINE_GGML_COMMON
#define GGML_COMMON_DECL_CPP
#endif
#include "../ggml-common.h"
#ifdef GGML_GEMMINI_ARGS_DEFINE_GGML_COMMON
#undef GGML_COMMON_DECL_CPP
#undef GGML_GEMMINI_ARGS_DEFINE_GGML_COMMON
#endif
#include "include/gemmini_params.h"

// Forward declaration to avoid including full gemmini.h (breaks include cycles)
enum tiled_matmul_type_t : int;

/*  
Gemmini 호출 인자를 한 데 모은 구조체 + Q8_0 전처리 헬퍼
기존에는 GemminiTensor가 ggml 텐서를 INT8 버퍼로 변환했으나, 정확도 측정을 위한 Q8_0 지원을 위해
변환된 버퍼와 블록별 스케일을 명시적으로 관리할 필요 */
typedef struct ggml_gemmini_args_t {
    // tiled_matmul_auto args
    size_t I = 0;
    size_t J = 0;
    size_t K = 0;

    elem_t *A = nullptr;
    elem_t *B = nullptr;
    void *C = nullptr;
    const void *D = nullptr;

    size_t sA = 0;
    size_t sB = 0;
    size_t sC = 0;
    size_t sD = 0;

    scale_t scale_A = 1.f;
    scale_t scale_B = 1.f;
    scale_acc_t scale_D = 1;
    int act = 0; // default NO_ACTIVATION
    acc_scale_t scale = 1.0f;
    acc_scale_t bert_scale = 1.0f;

    bool repeating_bias = false;
    bool transpose_A = false;
    bool transpose_B = false;
    bool full_C = false;
    bool low_D = false;

    uint8_t weightA = 0;
    tiled_matmul_type_t tiled_matmul_type = static_cast<tiled_matmul_type_t>(0);

    // metadata extracted from Q8_0 tensors
    const block_q8_0 *B_blocks = nullptr;
    const float *B_scales = nullptr;

    size_t blocks_K = 0;      // number of Q8_0 blocks along the K dimension
    size_t blocks_J = 0;      // number of logical columns covered by scale table
    size_t blocks_I = 0;      // optional helper for rows (when needed)

    uint32_t block_size_k = QK8_0;

    float scale_out = 1.0f;

    // origin output
    float* f_out = nullptr;
    size_t stride_f_out = 0;

    // logging & profiling helpers
    const char *layer_name = "";
    const char *tag = "";
    bool measure_cycles = true;

} ggml_gemmini_args_t;

/*
기존 dequantize_weight.h의 get_q80_row_ptr()/q80_to_T_* 흐름을 참고하여, Q8_0 블록에서
qs만 추출해 Gemmini가 요구하는 INT8 행렬을 구성하고 블록 스케일까지 계산하는 헬퍼. 
*/

// ggml 텐서에서 (iy, iz, iw) 좌표에 해당하는 Q8_0 행의 시작 블록을 반환
inline const block_q8_0 *ggml_gemmini_get_q80_row_ptr(const ggml_tensor *tensor,
                                                      int64_t iy,
                                                      int64_t iz,
                                                      int64_t iw) {
    const char *base = (const char *)(tensor->view_src ? tensor->view_src->data : tensor->data);
    const size_t offs = tensor->view_src ? tensor->view_offs : 0;
    return reinterpret_cast<const block_q8_0 *>(base + offs + iw * tensor->nb[3] + iz * tensor->nb[2] + iy * tensor->nb[1]);
}

// 텐서 데이터의 첫 Q8_0 블록 주소를 얻는다. view가 있으면 view offset까지 반영.
inline const block_q8_0 *ggml_gemmini_args_block_base(const ggml_tensor *tensor) {
    const char *base = (const char *)(tensor->view_src ? tensor->view_src->data : tensor->data);
    const size_t offs = tensor->view_src ? tensor->view_offs : 0;
    return reinterpret_cast<const block_q8_0 *>(base + offs);
}

// 각 Q8_0 블록의 스케일(d)을 float 배열로 추출한다.
inline void ggml_gemmini_extract_q80_scales(const ggml_tensor *src,
                                            float *dst_scales,
                                            size_t blocks_K,
                                            size_t logical_cols) {
    if (dst_scales == nullptr) {
        return;
    }

    const int64_t ny = src->ne[1] ? src->ne[1] : 1;  // logical columns (J)
    const int64_t nz = src->ne[2] ? src->ne[2] : 1;
    const int64_t nw = src->ne[3] ? src->ne[3] : 1;

    size_t col_idx = 0;
    for (int64_t iw = 0; iw < nw; ++iw) {
        for (int64_t iz = 0; iz < nz; ++iz) {
            for (int64_t iy = 0; iy < ny; ++iy) {
                const block_q8_0 *row_blocks = ggml_gemmini_get_q80_row_ptr(src, iy, iz, iw);
                for (size_t blk = 0; blk < blocks_K; ++blk) {
                    dst_scales[blk * logical_cols + col_idx] = ggml_fp16_to_fp32(row_blocks[blk].d);
                }
                ++col_idx;
            }
        }
    }

    GGML_ASSERT(col_idx == logical_cols);
}

// q80_to_T_rowwise()/q80_to_T_transposed 를 참고한 Q8_0 전처리 래퍼.
//  - INT8 버퍼(dst_base)에 qs를 채움
//  - block scale(d)을 float 배열로 저장
//  - ggml_gemmini_args_t에 관련 메타데이터를 기록
// 이후 Gemmini와 DEC가 동일 버퍼/스케일을 공유할 수 있도록 한 번만 호출
inline void ggml_gemmini_pack_q80(const ggml_tensor *src,
                                   bool transpose,
                                   elem_t *dst_base,
                                   size_t dst_stride_elems,
                                   float *dst_scales,
                                   ggml_gemmini_args_t &args) {
    GGML_ASSERT(src);
    GGML_ASSERT(src->type == GGML_TYPE_Q8_0);

    const int64_t dim_k = src->ne[0];
    const int64_t dim_j = src->ne[1] ? src->ne[1] : 1;
    const int64_t dim_z = src->ne[2] ? src->ne[2] : 1;
    const int64_t dim_w = src->ne[3] ? src->ne[3] : 1;

    GGML_ASSERT(dim_k % QK8_0 == 0);
    const size_t blocks_K = static_cast<size_t>(dim_k) / QK8_0;
    const size_t logical_cols = static_cast<size_t>(dim_j * dim_z * dim_w);

    if (dst_base) {
        if (transpose) {
            // Produce a (K x logical_cols) row-major buffer so that Gemmini consumes a true K×J matrix.
            for (int64_t k = 0; k < dim_k; ++k) {
                elem_t *dst_row = dst_base + static_cast<size_t>(k) * dst_stride_elems;
                size_t col_idx = 0;
                for (int64_t iw = 0; iw < dim_w; ++iw) {
                    for (int64_t iz = 0; iz < dim_z; ++iz) {
                        for (int64_t iy = 0; iy < dim_j; ++iy, ++col_idx) {
                            const block_q8_0 *row_blocks = ggml_gemmini_get_q80_row_ptr(src, iy, iz, iw);
                            const size_t blk = static_cast<size_t>(k) / QK8_0;
                            const int off = static_cast<int>(k % QK8_0);
                            dst_row[col_idx] = row_blocks[blk].qs[off];
                        }
                    }
                }
                GGML_ASSERT(col_idx == logical_cols);
            }
        } else {
            // Keep original (logical_cols x K) row-major layout
            size_t row_idx = 0;
            for (int64_t iw = 0; iw < dim_w; ++iw) {
                for (int64_t iz = 0; iz < dim_z; ++iz) {
                    for (int64_t iy = 0; iy < dim_j; ++iy, ++row_idx) {
                        const block_q8_0 *row_blocks = ggml_gemmini_get_q80_row_ptr(src, iy, iz, iw);
                        elem_t *dst_row = dst_base + row_idx * dst_stride_elems;
                        for (size_t blk = 0; blk < blocks_K; ++blk) {
                            std::memcpy(dst_row + blk * QK8_0, row_blocks[blk].qs, QK8_0 * sizeof(elem_t));
                        }
                    }
                }
            }
            GGML_ASSERT(row_idx == logical_cols);
        }
    }

    ggml_gemmini_extract_q80_scales(src, dst_scales, blocks_K, logical_cols);

    args.B_blocks = ggml_gemmini_args_block_base(src);
    args.B_scales = dst_scales;
    args.blocks_K = blocks_K;
    args.blocks_J = logical_cols;
    args.blocks_I = static_cast<size_t>(dim_k);
    args.block_size_k = QK8_0;
}

// Activation tensor quantization helper shared across Gemmini helpers.
inline void ggml_gemmini_quantize_activation(const ggml_tensor *src,
                                             ggml_gemmini_args_t &args,
                                             int8_t *dst)
{
    if (src == nullptr || dst == nullptr)
        return;

    if (src->type != GGML_TYPE_F32)
        return;

    GGML_ASSERT(src->data != nullptr);

    const float *srcf = static_cast<const float *>(src->data);
    const size_t total = args.I * args.K;
    if (total == 0)
        return;

    // 통계
    float max_abs = 0.0f;
    float min_val = srcf[0];
    float max_val = srcf[0];
    double sum = 0.0;
    double sum_sq = 0.0;
    size_t near_zero = 0;
    constexpr float zero_eps = 1e-6f;

    for (size_t i = 0; i < total; ++i)
    {
        const float v = srcf[i];
        max_abs = std::max(max_abs, std::fabs(v));
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        sum += static_cast<double>(v);
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        if (std::fabs(v) < zero_eps)
        {
            ++near_zero;
        }
    }

    // scale 결정
    constexpr float eps = 1e-8f;
    const float denom = std::max(max_abs, eps);
    float scale = denom / 127.0f;
    if (!std::isfinite(scale) || scale < eps)
        scale = 1.0f;

    args.scale_A = scale;

    const float inv_scale = 1.0f / scale;
    const char *layer_name = args.layer_name ? args.layer_name : "";
    const double mean = sum / static_cast<double>(total);
    const double variance = std::max(0.0, (sum_sq / static_cast<double>(total)) - mean * mean);
    const double stddev = std::sqrt(variance);
    const double zero_ratio = static_cast<double>(near_zero) / static_cast<double>(total);

    // 양자화 + 복원오차 지표(MAE/RMSE/SNR)
    size_t sat_pos = 0, sat_neg = 0;
    long double err_abs_sum = 0.0L, err_sq_sum = 0.0L, x_sq_sum = 0.0L;
    float max_abs_err = 0.0f;

    for (size_t i = 0; i < total; ++i)
    {
        const float x = srcf[i];
        float scaled = x * inv_scale;
        int qx = static_cast<int>(std::lrintf(scaled));
        if (qx > 127)
        {
            qx = 127;
            ++sat_pos;
        }
        else if (qx < -127)
        {
            qx = -127;
            ++sat_neg;
        }
        dst[i] = static_cast<int8_t>(qx);

        const float deq = static_cast<float>(qx) * scale;
        const float err = deq - x;
        const float aerr = std::fabs(err);
        err_abs_sum += aerr;
        err_sq_sum += static_cast<long double>(err) * static_cast<long double>(err);
        x_sq_sum += static_cast<long double>(x) * static_cast<long double>(x);
        max_abs_err = std::max(max_abs_err, aerr);
    }

    const double mae = static_cast<double>(err_abs_sum) / static_cast<double>(total);
    const double rmse = std::sqrt(static_cast<double>(err_sq_sum) / static_cast<double>(total));
    const double snr_db = (err_sq_sum > 0.0L && x_sq_sum > 0.0L)
                              ? 10.0 * std::log10(static_cast<double>(x_sq_sum / err_sq_sum))
                              : INFINITY;
    const double sat_ratio = (static_cast<double>(sat_pos + sat_neg) * 100.0) / static_cast<double>(total);

    // 요약 로그
    printf("[layer=%s][qact] N=%zu scale_A=%.6g sat=%.3f%% min=%g max=%g mean=%.6g std=%.6g mae=%.6g rmse=%.6g max|err|=%.6g snr=%.2f dB near0=%.2f%%\n",
           layer_name, total, scale, sat_ratio, min_val, max_val, mean, stddev, mae, rmse, max_abs_err, snr_db, zero_ratio * 100.0);

    // 포화되었을 때만 경고
    if (sat_pos || sat_neg)
    {
        printf("[layer=%s][qact.warn] saturation pos=%zu neg=%zu (%.4f%%)\n",
               layer_name, sat_pos, sat_neg, sat_ratio);
    }
}
