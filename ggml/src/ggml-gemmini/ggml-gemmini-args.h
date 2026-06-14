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

struct ggml_gemmini_activation_ethos_meta_t
{
    int16_t e_s = std::numeric_limits<int16_t>::min();
    int16_t m = 0;
    std::vector<int16_t> e_s_per_stripe_i; // one exponent per stripe (tile_I x K)

    inline int16_t resolve_stripe_e_s(int stripe_idx) const {
        if (stripe_idx >= 0 && !e_s_per_stripe_i.empty()) {
            const size_t s = static_cast<size_t>(stripe_idx);
            if (s < e_s_per_stripe_i.size()) {
                return e_s_per_stripe_i[s];
            }
        }
        return e_s;
    }

    inline void reset() {
        e_s = std::numeric_limits<int16_t>::min();
        m = 0;
        e_s_per_stripe_i.clear();
    }
};

struct ggml_gemmini_activation_tensor_meta_t
{
    float scale = 1.0f;

    inline void reset() {
        scale = 1.0f;
    }
};

struct ggml_gemmini_activation_quant_meta_t
{
    std::vector<ggml_gemmini_qact_outlier> outliers; // per-activation saturation records
    ggml_gemmini_activation_ethos_meta_t ethos{};
    ggml_gemmini_activation_tensor_meta_t tensor{};

    inline void reset() {
        outliers.clear();
        ethos.reset();
        tensor.reset();
    }
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
    ggml_gemmini_activation_quant_meta_t act_quant{};

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

        // Stripe-wise scale metadata (stripe_J logical output columns per shared stripe)
        std::vector<float> s_rf_stripe;      // [num_stripes_J] per-stripe float scale
        std::vector<uint16_t> R_stripe;      // [num_stripes_J] per-stripe offset
        size_t stripe_J = 0;                 // producer stripe width used for unpacked stripe metadata (0 or 1 = row-wise)
        size_t logical_stripe_J = 1;         // active cache contract width on the logical J axis (row-wise = 1)

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
        uint32_t block_size_k = GGML_GEMMINI_BLOCK_SIZE;
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
                size_t logical_stripe_J_ = 1,
                size_t stripe_J_ = 0) const {
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
            if (logical_stripe_J != logical_stripe_J_) {
                return false;
            }
            if (stripe_J != stripe_J_) {
                return false;
            }
            if (stripe_J_ > 1) {
                const size_t num_stripes = (logical_cols_ + stripe_J_ - 1) / stripe_J_;
                if (s_rf_stripe.size() != num_stripes) {
                    return false;
                }
                if (R_stripe.size() != num_stripes) {
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

    size_t stripe_J = 0;                  // logical output-column element count per shared scale stripe
    const float *s_rf_stripe = nullptr;   // [num_stripes_J] per-stripe float scale
    const uint16_t *R_stripe = nullptr;    // [num_stripes_J] per-stripe offset

    size_t blocks_K = 0;      // number of Q8_0 blocks along the K dimension
    size_t blocks_J = 0;      // number of logical rows covered by scale table (J * Z * W)
    size_t blocks_I = 0;      // legacy alias for rows (kept for ABI)

    uint32_t block_size_k = GGML_GEMMINI_BLOCK_SIZE;

    // output buffer (float, dequantized)
    float *f_out = nullptr;
    size_t col_stride_f_out = 0;
    size_t stride_f_out = 0;

    // layer/model metadata
    ggml::gemmini::types::LayerType layer_type{};
    const char *model_arch = nullptr;

    // Gemmini auto-tiling counts in DIM units (multiply by DIM to get element counts).
    size_t tile_I = 0;
    size_t tile_J = 0;
    size_t tile_K = 0;

    inline size_t stripe_J_or_rowwise_elems() const { return stripe_J > 0 ? stripe_J : 1; }
    inline bool stripe_mode_matches_tile_j(size_t tile_J_elems) const {
        return stripe_J <= 1 || (tile_J_elems > 0 && stripe_J == tile_J_elems);
    }
    inline int16_t resolve_stripe_activation_e_s(int stripe_idx) const {
        return act_quant.ethos.resolve_stripe_e_s(stripe_idx);
    }

    // Gemmini call metadata (for debugging/validation)
    size_t gemmini_call_k_logical = 0;
    size_t gemmini_call_k_aligned = 0;
    size_t gemmini_call_tile_k_elems = 0;

    // ethos parameter
    bool ethos_override_enabled = false;
    int ethos_q = 0;
    int ethos_delta = 0;
    bool ethos_l2_enabled = false;
    int ethos_l2_c = 1;
    int ethos_l2_d = 1;

} ggml_gemmini_args_t;

