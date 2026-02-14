// ggml-gemmini/ggml-gemmini-args.h
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <limits>

struct ggml_gemmini_qact_outlier
{
    int row = 0;       // activation row index (i)
    int col = 0;       // activation channel index (k)
    float original = 0.f;
    float saturated = 0.f;
};

enum ggml_gemmini_group_scope_t : uint8_t {
    GGML_GEMMINI_GROUP_BLOCK = 0,
    GGML_GEMMINI_GROUP_TILE = 1,
    GGML_GEMMINI_GROUP_ROW = 2,
    GGML_GEMMINI_GROUP_TENSOR = 3,
};

#include "ggml.h"
#include "ggml-quants.h"
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

    //elements
    elem_t *A = nullptr;
    elem_t *B = nullptr;
    void *C = nullptr;
    const void *D = nullptr;

    size_t sA = 0;
    size_t sB = 0;
    size_t sC = 0;
    size_t sD = 0;

    //scales, gemmini input val. 
    scale_t scale_A = 1.f;
    scale_t scale_B = 1.f;
    scale_acc_t scale_D = 1;
    int act = 0; // default NO_ACTIVATION
    acc_scale_t scale = 1.0f;
    acc_scale_t bert_scale = 1.0f;

    //for block scaling ////////////////////////////////////
    const block_q8_0 * A_blocks = nullptr;
    const float * A_scales = nullptr;
    size_t A_scale_rows = 0; // number of activation rows with Q8_0 block scales
    size_t A_scale_cols = 0; // number of Q8_0 blocks per row (K / QK8_0)


    //setiing flags 
    bool repeating_bias = false;
    bool transpose_A = false;
    bool transpose_B = false;
    bool full_C = true;
    bool low_D = false;

    // activation quantization metadata
    bool activation_block_scaled = false;
    std::vector<ggml_gemmini_qact_outlier> activation_outliers; // per-activation saturation records
    int activation_shamt = 0; // reserved (legacy field)

    //for weight checking   
    uint8_t weightA = 0;
    tiled_matmul_type_t tiled_matmul_type = static_cast<tiled_matmul_type_t>(0);

    // metadata extracted from Q8_0 tensors
    struct unpacked_weight {
        std::vector<int8_t> q;
        std::vector<float> scales;
        const block_q8_0 *blocks = nullptr;

        int64_t dim_k = 0;
        int64_t dim_j = 0;
        int64_t dim_z = 0;
        int64_t dim_w = 0;

        size_t logical_cols = 0;
        size_t blocks_K = 0;
        size_t blocks_J = 0;
        size_t blocks_I = 0;
        uint32_t block_size_k = QK8_0;
        size_t stride = 0;
        bool transpose_b = true;

        bool matches(const block_q8_0 *base,
                int64_t k,
                int64_t j,
                int64_t z,
                int64_t w,
                size_t stride_elems,
                size_t blocks_k,
                size_t logical_cols_,
                bool transpose_b_layout) const {
            if (blocks != base) {
                return false;
            }
            if (dim_k != k || dim_j != j || dim_z != z || dim_w != w) {
                return false;
            }
            if (stride != stride_elems) {
                return false;
            }
            if (blocks_K != blocks_k || blocks_J != logical_cols_) {
                return false;
            }
            if (block_size_k != QK8_0 || logical_cols != logical_cols_) {
                return false;
            }
            if (transpose_b != transpose_b_layout) {
                return false;
            }
            if (q.size() != static_cast<size_t>(k) * logical_cols_) {
                return false;
            }
            if (scales.size() != blocks_k * logical_cols_) {
                return false;
            }
            return true;
        }
    };

    const block_q8_0 *B_blocks = nullptr;
    const float *B_scales = nullptr;

    size_t blocks_K = 0;      // number of Q8_0 blocks along the K dimension
    size_t blocks_J = 0;      // number of logical columns covered by scale table
    size_t blocks_I = 0;      // optional helper for rows (when needed)

    // quant-group metadata resolved by orca quant logic
    ggml_gemmini_group_scope_t group_scope = GGML_GEMMINI_GROUP_BLOCK;
    size_t effective_group_size = 0;           // group element count (scope-aware)
    size_t effective_group_size_k = 0;         // K-axis group span used for Gemmini split
    size_t effective_group_size_aligned = 0;   // DIM(16)-aligned K-axis group span

    // Activation-scale layout metadata (data-driven polymorphism for dequant)
    bool activation_scale_table_enabled = false;
    size_t activation_scale_group_size_k = 0;  // K span per activation scale group
    size_t activation_scale_row_stride = 0;    // scale index += row * row_stride
    size_t activation_scale_group_stride = 0;  // scale index += k_group * group_stride

    size_t gemmini_call_k_logical = 0;         // logical K used by latest Gemmini call
    size_t gemmini_call_k_aligned = 0;         // padded K used by latest Gemmini call
    size_t gemmini_call_tile_k_elems = 0;      // tile_k * DIM used by latest call

    // Weight block scale granularity (Q8_0 scale table axis).
    uint32_t block_size_k = QK8_0;

    float scale_out = 1.0f;

    // origin output
    float* f_out = nullptr;
    size_t stride_f_out = 0;      // row stride in elements
    size_t col_stride_f_out = 0;  // column stride in elements

    // logging & profiling helpers
    const char *layer_name = "";
    const char *tag = "";
    bool measure_cycles = true;

} ggml_gemmini_args_t;

// TODO: Consider moving cache management to orca::ggml layer
inline const block_q8_0 *ggml_gemmini_args_block_base(const ggml_tensor *tensor) {
    const char *base = (const char *)(tensor->view_src ? tensor->view_src->data : tensor->data);
    const size_t offs = tensor->view_src ? tensor->view_offs : 0;
    return reinterpret_cast<const block_q8_0 *>(base + offs);
}
