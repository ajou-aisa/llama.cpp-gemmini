// ggml-gemmini/ggml-gemmini-args.h
#pragma once

#include <algorithm>
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

namespace ggml::gemmini::types {
enum class LayerType : uint8_t;
}

#include <ggml.h>
#ifndef GGML_COMMON_DECL
#define GGML_GEMMINI_ARGS_DEFINE_GGML_COMMON
#define GGML_COMMON_DECL_CPP
#endif
#include "../ggml-common.h"
#ifdef GGML_GEMMINI_ARGS_DEFINE_GGML_COMMON
#undef GGML_COMMON_DECL_CPP
#undef GGML_GEMMINI_ARGS_DEFINE_GGML_COMMON
#endif
#include <gemmini_params.h>

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

    // scales, gemmini input val.
    scale_t scale_B = 1.f;
    scale_acc_t scale_D = 1;
    int act = 0; // default NO_ACTIVATION
    acc_scale_t scale = 1.0f;
    acc_scale_t bert_scale = 1.0f;

    //setiing flags 
    bool repeating_bias = false;
    bool transpose_A = false;
    bool transpose_B = false;
    bool full_C = true;
    bool low_D = false;

    // activation quantization metadata
    const ggml_tensor *activation_src = nullptr; // source tensor for tile-row re-quantization in Gemmini loops
    int16_t activation_e_t = std::numeric_limits<int16_t>::min();
    int16_t activation_m = 0;
    std::vector<int16_t> activation_e_t_per_tile; // one exponent per logical tile_I x K row panel
    std::vector<ggml_gemmini_qact_outlier> activation_outliers; // per-activation saturation records

    //for weight checking   
    uint8_t weightA = 0;
    tiled_matmul_type_t tiled_matmul_type = static_cast<tiled_matmul_type_t>(0);

    // metadata extracted from Q8_0 tensors
    struct unpacked_weight {
        // Q8_0_R path (default): row-wise double-quantized planar weights
        std::vector<int8_t> q_qs;          // [logical_rows * K] dense int8 weights
        std::vector<uint8_t> c_b;         // [logical_rows][blocks_per_row]
        std::vector<float> s_rf;           // [logical_rows]
        std::vector<uint16_t> R;            // [logical_rows]

        // Panel-wise scale metadata (panel_J logical output columns per shared panel)
        std::vector<float> s_rf_panel;      // [num_panels_J] per-panel float scale
        std::vector<uint16_t> R_panel;      // [num_panels_J] per-panel offset
        size_t panel_J = 0;                 // producer panel width used for unpacked panel metadata (0 or 1 = row-wise)
        size_t logical_panel_J = 1;         // active cache contract width on the logical J axis (row-wise = 1)

        // Legacy Q8_0 path (preserved, not used by default)
        std::vector<int8_t> q;
        std::vector<float> scales; // [logical_rows][blocks_K] row-major
        const block_q8_0 *blocks = nullptr;

        int64_t dim_k = 0;
        int64_t dim_j = 0;
        int64_t dim_z = 0;
        int64_t dim_w = 0;

        size_t logical_cols = 0; // logical rows (J * Z * W) [legacy name]
        size_t blocks_K = 0;
        size_t blocks_J = 0;     // logical rows (J * Z * W) for scale rows
        size_t blocks_I = 0;     // legacy alias for logical rows (keep for ABI)
        uint32_t block_size_k = QK8_0;
        size_t stride = 0;
        bool transpose_b = true;

        bool matches(const block_q8_0 *base,
                int64_t k,
                int64_t j,
                int64_t z,
                int64_t w,
                size_t /*stride_elems*/,
                size_t blocks_k,
                size_t logical_cols_,
                bool /*transpose_b_layout*/,
                size_t logical_panel_J_ = 1,
                size_t panel_J_ = 0) const {
            if (blocks != base) {
                return false;
            }
            if (dim_k != k || dim_j != j || dim_z != z || dim_w != w) {
                return false;
            }
            if (blocks_K != blocks_k || blocks_J != logical_cols_) {
                return false;
            }
            if (block_size_k != QK8_0 || logical_cols != logical_cols_) {
                return false;
            }
            // Q8_0_R planar validity check
            if (q_qs.size() != logical_cols_ * static_cast<size_t>(k)) {
                return false;
            }
            if (c_b.size() != blocks_k * logical_cols_) {
                return false;
            }
            if (s_rf.size() != logical_cols_) {
                return false;
            }
            if (R.size() != logical_cols_) {
                return false;
            }
            if (logical_panel_J != logical_panel_J_) {
                return false;
            }
            if (panel_J != panel_J_) {
                return false;
            }
            if (panel_J_ > 1) {
                const size_t num_panels = (logical_cols_ + panel_J_ - 1) / panel_J_;
                if (s_rf_panel.size() != num_panels) {
                    return false;
                }
                if (R_panel.size() != num_panels) {
                    return false;
                }
            }
            return true;
        }
    };

    const block_q8_0 *B_blocks = nullptr;
    const float *B_scales = nullptr; // [blocks_J][blocks_K] row-major (row = J*Z*W)

    // Q8_0_R weight fields (default path, no mode flag needed)
    const uint8_t  *c_b = nullptr;       // [J * blocks_per_row] per-block effective code
    const float    *s_rf = nullptr;       // [J] per-row float scale
    const uint16_t *R = nullptr;           // [J] per-row offset
    size_t blocks_per_row = 0;            // K / 32

    size_t panel_J = 0;                  // logical output-column element count per shared scale panel
    const float *s_rf_panel = nullptr;   // [num_panels_J] per-panel float scale
    const uint16_t *R_panel = nullptr;    // [num_panels_J] per-panel offset

    size_t blocks_K = 0;      // number of Q8_0 blocks along the K dimension
    size_t blocks_J = 0;      // number of logical rows covered by scale table (J * Z * W)
    size_t blocks_I = 0;      // legacy alias for rows (kept for ABI)

    // quant-group metadata resolved by ggml::gemmini quant logic
    ggml_gemmini_group_scope_t group_scope = GGML_GEMMINI_GROUP_BLOCK;
    size_t effective_group_size = 0;           // group element count (scope-aware)
    size_t effective_group_size_k = 0;         // K-axis group span used for Gemmini split
    size_t effective_group_size_aligned = 0;   // DIM(16)-aligned K-axis group span

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
    ggml::gemmini::types::LayerType layer_type = static_cast<ggml::gemmini::types::LayerType>(0);
    const char *model_arch = "";
    const char *tag = "";
    bool measure_cycles = true;

    // Gemmini auto-tiling counts in DIM units. Use tile_*_elems() for logical element spans.
    size_t tile_I = 0;
    size_t tile_J = 0;
    size_t tile_K = 0;

    inline size_t tile_I_elems() const { return tile_I * static_cast<size_t>(DIM); }
    inline size_t tile_J_elems() const { return tile_J * static_cast<size_t>(DIM); }
    inline size_t tile_K_elems() const { return tile_K * static_cast<size_t>(DIM); }
    inline size_t panel_J_or_rowwise_elems() const { return panel_J > 0 ? panel_J : 1; }
    inline int16_t resolve_tile_row_activation_e_t(int tile_row) const {
        if (tile_row >= 0 && !activation_e_t_per_tile.empty()) {
            const size_t panel_idx = static_cast<size_t>(tile_row);
            if (panel_idx < activation_e_t_per_tile.size()) {
                return activation_e_t_per_tile[panel_idx];
            }
        }
        return activation_e_t;
    }

    inline void prepare_group_meta() {
        const size_t dim_k = K;
        const size_t weight_block_k = block_size_k > 0
                                          ? static_cast<size_t>(block_size_k)
                                          : static_cast<size_t>(QK8_0);
        const auto ceil_div_size = [](size_t a, size_t b) {
            return (a + b - 1) / b;
        };

        if (group_scope == GGML_GEMMINI_GROUP_BLOCK) {
            const size_t group_k = std::max<size_t>(1, weight_block_k);
            const size_t aligned =
                std::max<size_t>(16, ceil_div_size(group_k, static_cast<size_t>(16)) * static_cast<size_t>(16));

            effective_group_size = group_k;
            effective_group_size_k = group_k;
            effective_group_size_aligned = aligned;
            return;
        }

        size_t group = effective_group_size;
        if (group == 0) {
            group = block_size_k > 0 ? block_size_k : QK8_0;
        }

        group = std::max<size_t>(1, group);
        size_t group_k = effective_group_size_k;
        if (group_k == 0) {
            if (dim_k > 0 && group > dim_k) {
                group_k = dim_k;
            } else {
                group_k = group;
            }
        }
        group_k = std::max<size_t>(1, group_k);

        const size_t aligned =
            std::max<size_t>(16, ceil_div_size(group_k, static_cast<size_t>(16)) * static_cast<size_t>(16));

        effective_group_size = group;
        effective_group_size_k = group_k;
        effective_group_size_aligned = aligned;
    }

    // ethos parameter
    bool ethos_override_enabled = false;
    int ethos_q = 0;
    int ethos_delta = 0;
    bool ethos_l2_enabled = false;
    int ethos_l2_c = 1;
    int ethos_l2_d = 1;

} ggml_gemmini_args_t;


