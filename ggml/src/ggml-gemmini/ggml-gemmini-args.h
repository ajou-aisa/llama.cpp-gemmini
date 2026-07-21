// ggml-gemmini/ggml-gemmini-args.h
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>

#include "quants/act/meta.hpp"
#include "quants/act/types.hpp"

namespace act = ggml::gemmini::quants::act;

using ggml_gemmini_qact_outlier = act::ggml_gemmini_qact_outlier;

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

// Bit-level ldexp replacement for the HP1/HP2 hot loop.
// Valid for positive normalized float x and exp range where the result is also normal.
// ponytail: skips ldexpf libc call; falls back only on denormal/inf/nan inputs (rejected by contract upstream).
// Upgrade: if denormals ever become legal upstream, this branch must be revisited.
static inline float gemmini_ldexp_fast_pos(float x, int m) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    const int32_t exp = (int32_t)((u >> 23) & 0xFFu);
    if (exp == 0 || exp == 0xFF) return std::ldexp(x, m);
    const int32_t new_exp = exp + m;
    if (new_exp <= 0) return 0.0f;
    if (new_exp >= 0xFF) return INFINITY;
    const uint32_t out = (u & ~(0xFFu << 23)) | ((uint32_t)new_exp << 23);
    float r;
    std::memcpy(&r, &out, sizeof(r));
    return r;
}

/*  
    Gemmini 호출 인자를 한 데 모은 구조체 + Q8_0 전처리 헬퍼
    기존에는 GemminiTensor가 ggml 텐서를 INT8 버퍼로 변환했으나, 정확도 측정을 위한 Q8_0 지원을 위해
    변환된 버퍼와 블록별 스케일을 명시적으로 관리할 필요 */
typedef struct ggml_gemmini_args_t {
    enum class im2p_weight_format_t : uint8_t {
        q8_0_unpacked_to_h1 = 0,
        q8_h0 = 1,
        q8_h2 = 2,
        q8_h1 = 3,
        q8_hp1 = 4,
        q8_hp2 = 5,
    };

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
    act::Meta act_quant{};

    //for weight checking   
    uint8_t weightA = 0;
    tiled_matmul_type_t tiled_matmul_type = static_cast<tiled_matmul_type_t>(0);

    // metadata extracted from Q8_0 tensors
    struct unpacked_weight {
        // Q8_H1 path (default): row-wise double-quantized planar weights
        std::vector<int8_t> q_qs;          // [logical_rows * K] dense int8 weights
        std::vector<uint8_t> c_b;         // [logical_rows][blocks_per_row]
        std::vector<float> s_rf;           // [logical_rows]
        std::vector<uint16_t> R;            // [logical_rows]

        // Stripe-wise scale metadata (stripe_J logical output columns per shared stripe)
        std::vector<float> s_rf_stripe;      // [num_stripes_J] per-stripe float scale
        std::vector<uint16_t> R_stripe;      // [num_stripes_J] per-stripe offset
        size_t stripe_J = 0;                 // producer stripe width used for unpacked stripe metadata (0 or 1 = row-wise)
        size_t logical_stripe_J = 1;

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

    };

    unpacked_weight unpacked;

    const block_q8_0 *B_blocks = nullptr;
    const float *B_scales = nullptr; // [blocks_J][blocks_K] row-major (row = J*Z*W)

    bool weight_i8_scale_active = false;
    float weight_scale = 1.0f;
    im2p_weight_format_t weight_format = im2p_weight_format_t::q8_0_unpacked_to_h1;

    const block_q8_h1 *q8_h1_blocks = nullptr;
    size_t q8_h1_block_count = 0;
    size_t q8_h1_rows = 0;

    const block_q8_h2 *q8_h2_blocks = nullptr;
    size_t q8_h2_block_count = 0;
    size_t q8_h2_blocks_per_row = 0;

    const block_q8_hp1 *q8_hp1_blocks = nullptr;
    size_t q8_hp1_block_count = 0;
    size_t q8_hp1_blocks_per_row = 0;

    const block_q8_hp2 *q8_hp2_blocks = nullptr;
    size_t q8_hp2_block_count = 0;
    size_t q8_hp2_blocks_per_row = 0;

    // Q8_H1 weight fields (default path, no mode flag needed)
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
    // Gemmini call metadata (for debugging/validation)
    size_t gemmini_call_k_logical = 0;
    size_t gemmini_call_k_aligned = 0;
    size_t gemmini_call_tile_k_elems = 0;

    inline const block_q8_h2 *q8_h2_block(size_t row, size_t block) const {
        if (q8_h2_blocks == nullptr || q8_h2_blocks_per_row == 0 ||
            row >= J || block >= q8_h2_blocks_per_row ||
            row > std::numeric_limits<size_t>::max() / q8_h2_blocks_per_row) {
            return nullptr;
        }

        const size_t row_offset = row * q8_h2_blocks_per_row;
        if (block > std::numeric_limits<size_t>::max() - row_offset) {
            return nullptr;
        }

        const size_t offset = row_offset + block;
        return offset < q8_h2_block_count ? q8_h2_blocks + offset : nullptr;
    }

    inline bool has_q8_h2_im2p_contract() const {
        if (weight_format != im2p_weight_format_t::q8_h2 ||
            q8_h2_blocks == nullptr || J == 0 || K == 0 ||
            K % QK8_H2 != 0 || q8_h2_blocks_per_row != K / QK8_H2 ||
            reinterpret_cast<uintptr_t>(q8_h2_blocks) % alignof(block_q8_h2) != 0 ||
            J > std::numeric_limits<size_t>::max() / q8_h2_blocks_per_row ||
            q8_h2_block_count != J * q8_h2_blocks_per_row) {
            return false;
        }

        for (size_t i = 0; i < q8_h2_block_count; ++i) {
            uint32_t scale_bits = 0;
            std::memcpy(&scale_bits, &q8_h2_blocks[i].channel_scale, sizeof(scale_bits));
            if ((scale_bits & 0x7f800000u) == 0x7f800000u) {
                return false;
            }
        }

        return true;
    }

    inline bool has_no_q8_h1_metadata() const {
        return B == nullptr && B_blocks == nullptr && B_scales == nullptr &&
               c_b == nullptr && s_rf == nullptr && R == nullptr &&
               s_rf_stripe == nullptr && R_stripe == nullptr &&
               q8_h1_blocks == nullptr && q8_h1_block_count == 0 &&
               q8_h1_rows == 0 && unpacked.q_qs.empty() && unpacked.c_b.empty() &&
               unpacked.s_rf.empty() && unpacked.R.empty() && unpacked.s_rf_stripe.empty() &&
               unpacked.R_stripe.empty() && unpacked.q.empty() && unpacked.scales.empty() &&
               unpacked.blocks == nullptr;
    }

    inline const block_q8_hp1 *q8_hp1_block(size_t row, size_t block) const {
        if (q8_hp1_blocks == nullptr || q8_hp1_blocks_per_row == 0 ||
            row >= J || block >= q8_hp1_blocks_per_row ||
            row > std::numeric_limits<size_t>::max() / q8_hp1_blocks_per_row) {
            return nullptr;
        }

        const size_t row_offset = row * q8_hp1_blocks_per_row;
        if (block > std::numeric_limits<size_t>::max() - row_offset) {
            return nullptr;
        }

        const size_t offset = row_offset + block;
        return offset < q8_hp1_block_count ? q8_hp1_blocks + offset : nullptr;
    }

    inline bool has_q8_hp1_im2p_contract() const {
        if (weight_format != im2p_weight_format_t::q8_hp1 ||
            q8_hp1_blocks == nullptr || J == 0 || K == 0 ||
            K % QK8_HP != 0 || q8_hp1_blocks_per_row != K / QK8_HP ||
            reinterpret_cast<uintptr_t>(q8_hp1_blocks) % alignof(block_q8_hp1) != 0 ||
            J > std::numeric_limits<size_t>::max() / q8_hp1_blocks_per_row ||
            q8_hp1_block_count != J * q8_hp1_blocks_per_row ||
            q8_hp1_block_count > std::numeric_limits<size_t>::max() / sizeof(block_q8_hp1) ||
            q8_hp2_blocks != nullptr || q8_hp2_block_count != 0 ||
            q8_hp2_blocks_per_row != 0 || !has_no_q8_h1_metadata()) {
            return false;
        }

        // Payload validity is guaranteed at quantize time (llama-quant.cpp) and, if requested,
        // once at load (check_tensors). Weights are immutable during inference and the HP kernel
        // is robust to malformed data (ldexp_fast_pos fallbacks, m==INT16_MIN handled), so we do
        // not re-scan the whole tensor via ggml_validate_row_data on every matmul call.
        return true;
    }

    inline const block_q8_hp2 *q8_hp2_block(size_t row, size_t block) const {
        if (q8_hp2_blocks == nullptr || q8_hp2_blocks_per_row == 0 ||
            row >= J || block >= q8_hp2_blocks_per_row ||
            row > std::numeric_limits<size_t>::max() / q8_hp2_blocks_per_row) {
            return nullptr;
        }

        const size_t row_offset = row * q8_hp2_blocks_per_row;
        if (block > std::numeric_limits<size_t>::max() - row_offset) {
            return nullptr;
        }

        const size_t offset = row_offset + block;
        return offset < q8_hp2_block_count ? q8_hp2_blocks + offset : nullptr;
    }

    inline bool has_q8_hp2_im2p_contract() const {
        if (weight_format != im2p_weight_format_t::q8_hp2 ||
            q8_hp2_blocks == nullptr || J == 0 || K == 0 ||
            K % QK8_HP != 0 || q8_hp2_blocks_per_row != K / QK8_HP ||
            reinterpret_cast<uintptr_t>(q8_hp2_blocks) % alignof(block_q8_hp2) != 0 ||
            J > std::numeric_limits<size_t>::max() / q8_hp2_blocks_per_row ||
            q8_hp2_block_count != J * q8_hp2_blocks_per_row ||
            q8_hp2_block_count > std::numeric_limits<size_t>::max() / sizeof(block_q8_hp2) ||
            q8_hp1_blocks != nullptr || q8_hp1_block_count != 0 ||
            q8_hp1_blocks_per_row != 0 || !has_no_q8_h1_metadata()) {
            return false;
        }

        // See has_q8_hp1_im2p_contract: payload is validated at quantize/load time, not per matmul.
        return true;
    }

    inline const block_q8_h1 *q8_h1_block(size_t row, size_t block) const {
        if (q8_h1_blocks == nullptr || blocks_per_row == 0 ||
            row >= q8_h1_rows || block >= blocks_per_row ||
            row > std::numeric_limits<size_t>::max() / blocks_per_row) {
            return nullptr;
        }

        const size_t row_offset = row * blocks_per_row;
        if (block > std::numeric_limits<size_t>::max() - row_offset) {
            return nullptr;
        }

        const size_t offset = row_offset + block;
        if (offset >= q8_h1_block_count) {
            return nullptr;
        }

        return q8_h1_blocks + offset;
    }

    inline bool has_q8_h1_im2p_contract() const {
        if (weight_format != im2p_weight_format_t::q8_h1 ||
            q8_h1_blocks == nullptr || J == 0 || K == 0 ||
            blocks_per_row == 0 || q8_h1_block_count == 0 || q8_h1_rows < J ||
            reinterpret_cast<uintptr_t>(q8_h1_blocks) % alignof(block_q8_h1) != 0 ||
            K > std::numeric_limits<size_t>::max() - (QK8_0 - 1) ||
            blocks_per_row != (K + QK8_0 - 1) / QK8_0 ||
            J > std::numeric_limits<size_t>::max() / blocks_per_row) {
            return false;
        }

        if (q8_h1_block_count < J * blocks_per_row) {
            return false;
        }

        for (size_t row = 0; row < J; ++row) {
            const block_q8_h1 *first = q8_h1_block(row, 0);
            if (first == nullptr || !std::isfinite(first->s_rf) || first->s_rf < 0.0f ||
                (first->s_rf == 0.0f && (first->R != 0 || first->c_b != 0))) {
                return false;
            }

            for (size_t block = 1; block < blocks_per_row; ++block) {
                const block_q8_h1 *current = q8_h1_block(row, block);
                if (current == nullptr || current->s_rf != first->s_rf || current->R != first->R ||
                    (first->s_rf == 0.0f && current->c_b != 0)) {
                    return false;
                }
            }
        }

        return true;
    }

} ggml_gemmini_args_t;

#if defined(GGML_GEMMINI_TEST_OBSERVER)
namespace ggml::gemmini {
using test_i_observer_t = void (*)(const char * consumer, size_t I, void * user_data);

GGML_API void set_test_i_observer(test_i_observer_t observer, void * user_data);
}
#endif
