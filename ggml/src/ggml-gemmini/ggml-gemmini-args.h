// ggml-gemmini/ggml-gemmini-args.h
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "ggml-gemmini-config.hpp"
#include <gemmini/evaluation_metrics.hpp>
#if CYCLE_SIM
#include <gemmini/cycle_sim_log.hpp>
#endif
#include "ggml-gemmini-geometry.hpp"
#include "quants/act/buffer.hpp"
#include "quants/act/meta.hpp"
#include "quants/act/types.hpp"

namespace act = ggml::gemmini::quants::act;
namespace ggml::gemmini::optrace {
struct Context;
}

namespace ggml::gemmini::quants::act::exsia {
struct StripeReadySink;
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
#if defined(GGML_GEMMINI_CONFIGURED_DIM)
static_assert(DIM == GGML_GEMMINI_CONFIGURED_DIM,
              "Gemmini parameter header DIM does not match configured DIM");
#endif

static_assert(sizeof(elem_t) == GGML_GEMMINI_ACTIVATION_STORAGE_BYTES,
              "elem_t must match configured activation transport storage");
static_assert(sizeof(elem_t) == GGML_GEMMINI_WEIGHT_STORAGE_BYTES,
              "elem_t must match configured weight transport storage");
static_assert(GGML_GEMMINI_ACTIVATION_BITS == 16 ? std::is_same_v<elem_t, int16_t>
                                                 : std::is_same_v<elem_t, int8_t>,
              "elem_t must be int16_t only for A16/W16, otherwise int8_t");

// Forward declaration to avoid including full gemmini.h (breaks include cycles)
enum tiled_matmul_type_t : int;

// Bit-level ldexp replacement for the HP1/HP2 hot loop.
// Valid for positive normalized float x and exp range where the result is also normal.
// ponytail: skips ldexpf libc call; falls back only on denormal/inf/nan inputs (rejected by
// contract upstream). Upgrade: if denormals ever become legal upstream, this branch must be
// revisited.
static inline float gemmini_ldexp_fast_pos(float x, int m) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    const int32_t exp = (int32_t)((u >> 23) & 0xFFu);
    if (exp == 0 || exp == 0xFF)
        return std::ldexp(x, m);
    const int32_t new_exp = exp + m;
    if (new_exp <= 0)
        return 0.0f;
    if (new_exp >= 0xFF)
        return std::numeric_limits<float>::max();
    const uint32_t out = (u & ~(0xFFu << 23)) | ((uint32_t)new_exp << 23);
    float          r;
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
        q8_h0               = 1,
        q8_h2               = 2,
        // 3 is retired.
        q8_hp1                   = 4,
        q8_hp2                   = 5,
        q8_channel               = 6,
        q8_channel_dense_sidecar = 7,
        q4_h0                    = 8,
        // 9 is retired.
        q4_hp1  = 10,
        q16_h0  = 11,
        q16_h1  = 12,
        q16_hp1 = 13,
    };

    // Geometry and format-dependent strides remain in their ABI field order.
    size_t I = 0;
    size_t J = 0;
    size_t K = 0;

    // A shares its backing with copies/slices; other raw input/output views borrow storage.
    act::QuantizedActivationBuffer A;
    elem_t *                       B      = nullptr;
    const float *                  A_fp32 = nullptr;
    const float *                  B_fp32 = nullptr;
    void *                         C      = nullptr;
    const void *                   D      = nullptr;

    size_t sA = 0;
    size_t sB = 0;
    size_t sC = 0;
    size_t sD = 0;

    size_t activation_row_offset      = 0;
    size_t activation_rows_per_stripe = 0;

    // scales, gemmini input val.
    scale_t     scale_B    = 1.f;
    scale_acc_t scale_D    = 1;
    int         act        = 0; // default NO_ACTIVATION
    acc_scale_t scale      = 1.0f;
    acc_scale_t bert_scale = 1.0f;

    // setiing flags
    bool repeating_bias = false;
    bool transpose_A    = false;
    bool transpose_B    = false;
    bool full_C         = true;
    bool low_D          = false;

    // Metadata values own their containers; residual handles share payloads and the sink is
    // borrowed.
    act::Meta          act_quant{};
    act::ResidualRoute residual_route = act::ResidualRoute::ws_packet;
    const ggml::gemmini::quants::act::exsia::StripeReadySink * exsia_stripe_ready_sink = nullptr;

    // for weight checking
    uint8_t             weightA           = 0;
    tiled_matmul_type_t tiled_matmul_type = static_cast<tiled_matmul_type_t>(0);

    // Unpacked vectors own backing; descriptor copies do not rebase raw aliases into those vectors.
    struct unpacked_weight {
        // Dense affine path (default): row-wise double-quantized planar weights
        std::vector<int8_t>   q_qs; // [logical_rows * K] dense int8 weights
        std::vector<uint8_t>  c_b;  // [logical_rows][blocks_per_row]
        std::vector<float>    s_rf; // [logical_rows]
        std::vector<uint16_t> R;    // [logical_rows]

        // Stripe-wise scale metadata (stripe_J logical output columns per shared stripe)
        std::vector<float>    s_rf_stripe; // [num_stripes_J] per-stripe float scale
        std::vector<uint16_t> R_stripe;    // [num_stripes_J] per-stripe offset
        size_t                stripe_J =
            0; // producer stripe width used for unpacked stripe metadata (0 or 1 = row-wise)
        size_t logical_stripe_J = 1;

        // Legacy Q8_0 path (preserved, not used by default)
        std::vector<int8_t> q;
        std::vector<float>  scales; // [logical_rows][blocks_K] row-major
        const block_q8_0 *  blocks = nullptr;

        int64_t dim_k = 0;
        int64_t dim_j = 0;
        int64_t dim_z = 0;
        int64_t dim_w = 0;

        size_t   logical_cols = 0; // logical rows (J * Z * W) [legacy name]
        size_t   blocks_K     = 0;
        size_t   blocks_J     = 0; // logical rows (J * Z * W) for scale rows
        size_t   blocks_I     = 0; // legacy alias for logical rows (keep for ABI)
        uint32_t block_size_k = GGML_GEMMINI_BLOCK_SIZE;
        size_t   stride       = 0;
        bool     transpose_b  = true;
    };

    unpacked_weight unpacked;

    // Borrowed weight blocks/scales must outlive every consumer of this descriptor.
    const block_q8_0 * B_blocks = nullptr;
    const float *      B_scales = nullptr; // [blocks_J][blocks_K] row-major (row = J*Z*W)
    const float *      weight_channel_scales      = nullptr;
    size_t             weight_channel_scale_count = 0;

    const uint8_t * q8_channel_row_base   = nullptr;
    size_t          q8_channel_row_stride = 0;
    size_t          q8_channel_row_count  = 0;

    bool                 weight_i8_scale_active = false;
    float                weight_scale           = 1.0f;
    im2p_weight_format_t weight_format          = im2p_weight_format_t::q8_0_unpacked_to_h1;

    const block_q8_h2 * q8_h2_blocks         = nullptr;
    size_t              q8_h2_block_count    = 0;
    size_t              q8_h2_blocks_per_row = 0;

    const block_q8_hp1 * q8_hp1_blocks         = nullptr;
    size_t               q8_hp1_block_count    = 0;
    size_t               q8_hp1_blocks_per_row = 0;

    const block_q8_hp2 * q8_hp2_blocks         = nullptr;
    size_t               q8_hp2_block_count    = 0;
    size_t               q8_hp2_blocks_per_row = 0;

    // Native matched-width FULL-provider formats. Each block spans 32 K
    // elements; Q4 callbacks unpack signed nibbles and Q16 callbacks preserve
    // the int16 codes for the canonical typed provider ABI.
    const block_q4_h0 *   q4_h0_blocks          = nullptr;
    const block_q4_hp1 *  q4_hp1_blocks         = nullptr;
    const block_q16_h0 *  q16_h0_blocks         = nullptr;
    const block_q16_h1 *  q16_h1_blocks         = nullptr;
    const block_q16_hp1 * q16_hp1_blocks        = nullptr;
    size_t                native_block_count    = 0;
    size_t                native_blocks_per_row = 0;
    // Checked available backing extent for native block readers. Native
    // dispatch requires a nonzero extent covering every declared block.
    size_t native_weight_bytes = 0;

    // Dense affine weight fields (default path, no mode flag needed)
    const uint8_t *  c_b            = nullptr; // [J * blocks_per_row] per-block effective code
    const float *    s_rf           = nullptr; // [J] per-row float scale
    const uint16_t * R              = nullptr; // [J] per-row offset
    size_t           blocks_per_row = 0;       // K / 32

    size_t           stripe_J    = 0; // logical output-column element count per shared scale stripe
    const float *    s_rf_stripe = nullptr; // [num_stripes_J] per-stripe float scale
    const uint16_t * R_stripe    = nullptr; // [num_stripes_J] per-stripe offset

    size_t blocks_K = 0; // number of Q8_0 blocks along the K dimension
    size_t blocks_J = 0; // number of logical rows covered by scale table (J * Z * W)
    size_t blocks_I = 0; // legacy alias for rows (kept for ABI)

    uint32_t block_size_k = GGML_GEMMINI_BLOCK_SIZE;

    // Borrowed destination; execution may redirect this view to an owned stage, then restore it.
    float * f_out            = nullptr;
    size_t  col_stride_f_out = 0;
    size_t  stride_f_out     = 0;

    // model_arch is borrowed; the matmul_layer string below owns its identity.
    uint8_t      reserved_layer_metadata = 0;
    const char * model_arch              = nullptr;

    // Gemmini auto-tiling counts in DIM units (multiply by DIM to get element counts).
    size_t tile_I = 0;
    size_t tile_J = 0;
    size_t tile_K = 0;

    inline ggml::gemmini::GemminiGeometryResult activation_geometry() const {
        return ggml::gemmini::make_gemmini_geometry({{I, J, K}, {tile_I, tile_J, tile_K}, DIM});
    }

    inline bool activation_geometry_matches(ggml::gemmini::GemminiGeometry & geometry) const {
        const auto result = activation_geometry();
        if (!result.ok() || activation_rows_per_stripe != result.geometry.stripe_rows)
            return false;
        geometry = result.geometry;
        return true;
    }

    inline ggml::gemmini::GemminiGeometryResult activation_quant_geometry() const {
        const size_t array_dim = DIM;
        if (array_dim == 0)
            return activation_geometry();
        const auto tile_or_full_extent = [](size_t tile, size_t extent, size_t dimension) {
            if (tile != 0)
                return tile;
            return extent / dimension + static_cast<size_t>(extent % dimension != 0);
        };
        return ggml::gemmini::make_gemmini_geometry({{I, J, K},
                                                     {tile_or_full_extent(tile_I, I, array_dim),
                                                      tile_or_full_extent(tile_J, J, array_dim),
                                                      tile_or_full_extent(tile_K, K, array_dim)},
                                                     array_dim});
    }

    inline bool activation_quant_geometry_matches(ggml::gemmini::GemminiGeometry & geometry) const {
        const auto result = activation_quant_geometry();
        if (!result.ok() || (activation_rows_per_stripe != 0 &&
                             activation_rows_per_stripe != result.geometry.stripe_rows))
            return false;
        geometry = result.geometry;
        return true;
    }

    inline size_t stripe_J_or_rowwise_elems() const {
        return stripe_J > 0 ? stripe_J : 1;
    }
    inline bool stripe_mode_matches_tile_j(size_t tile_J_elems) const {
        return stripe_J <= 1 || (tile_J_elems > 0 && stripe_J == tile_J_elems);
    }
    // Gemmini call metadata (for debugging/validation)
    size_t gemmini_call_k_logical    = 0;
    size_t gemmini_call_k_aligned    = 0;
    size_t gemmini_call_tile_k_elems = 0;

    // Owned identity and shared immutable observation contexts survive asynchronous consumers.
    std::string matmul_layer;
    // Optional immutable driver provenance, explicitly owned by asynchronous
    // frontend/RMD work. No process-global phase and no allocation when off.
    std::shared_ptr<const ggml::gemmini::optrace::Context> optrace_context;
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS || GGML_GEMMINI_SCALE_METRICS
    std::shared_ptr<ggml::gemmini::evaluation::Invocation> evaluation_context;
    // Evaluation only: the terminal lm_head of a metric-only session. Every observation runs; the
    // numerical GEMMs, their reconstruction and the output (the logits) do not.
    bool metric_terminal_only = false;
#endif
#if CYCLE_SIM
    ggml::gemmini::cycle_sim::Context cycle_sim_context;
    std::vector<uint64_t>             cycle_sim_host_dependencies;
#endif

    inline const uint8_t * q8_channel_row(size_t row) const {
        if (q8_channel_row_base == nullptr || q8_channel_row_stride == 0 ||
            row >= q8_channel_row_count ||
            row > std::numeric_limits<size_t>::max() / q8_channel_row_stride) {
            return nullptr;
        }

        return q8_channel_row_base + row * q8_channel_row_stride;
    }

    inline const elem_t * q8_channel_payload(size_t row) const {
        const uint8_t * row_base = q8_channel_row(row);
        return row_base == nullptr ? nullptr
                                   : reinterpret_cast<const elem_t *>(row_base + sizeof(float));
    }

    inline float q8_channel_scale(size_t row) const {
        const uint8_t * row_base = q8_channel_row(row);
        if (row_base == nullptr) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        float scale = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(&scale, row_base, sizeof(scale));
        return scale;
    }

    inline bool has_q8_channel_row_metadata() const {
        return q8_channel_row_base != nullptr || q8_channel_row_stride != 0 ||
               q8_channel_row_count != 0;
    }

    inline size_t q8_channel_scale_source_count() const {
        return static_cast<size_t>(has_q8_channel_row_metadata()) +
               static_cast<size_t>(weight_channel_scales != nullptr) +
               static_cast<size_t>(B_scales != nullptr) +
               static_cast<size_t>(weight_i8_scale_active);
    }

    inline bool has_q8_channel_direct_read_contract() const {
        static_assert(GGML_GEMMINI_WEIGHT_BITS != 8 || sizeof(elem_t) == 1,
                      "W8 Q8_CHANNEL direct-read requires one-byte elem_t");
        if constexpr (GGML_GEMMINI_WEIGHT_BITS != 8 || GGML_GEMMINI_WEIGHT_STORAGE_BYTES != 1) {
            return false;
        }

        if (weight_format != im2p_weight_format_t::q8_channel || q8_channel_row_base == nullptr ||
            J == 0 || K == 0 || q8_channel_row_count != J || q8_channel_row_stride == 0 ||
            K > std::numeric_limits<size_t>::max() - sizeof(float) ||
            q8_channel_row_stride != sizeof(float) + K || sB != q8_channel_row_stride ||
            B == nullptr ||
            q8_channel_row_count > std::numeric_limits<size_t>::max() / q8_channel_row_stride ||
            B_scales != nullptr || weight_channel_scales != nullptr ||
            weight_channel_scale_count != 0 || weight_i8_scale_active ||
            q8_channel_scale_source_count() != 1 || q8_channel_payload(0) != B) {
            return false;
        }

        for (size_t row = 0; row < q8_channel_row_count; ++row) {
            if (!std::isfinite(q8_channel_scale(row))) {
                return false;
            }
        }

        return true;
    }

    inline bool has_q8_channel_dense_sidecar_contract() const {
        static_assert(GGML_GEMMINI_WEIGHT_BITS != 8 || sizeof(elem_t) == 1,
                      "W8 Q8_CHANNEL dense-sidecar requires one-byte elem_t");
        if constexpr (GGML_GEMMINI_WEIGHT_BITS != 8 || GGML_GEMMINI_WEIGHT_STORAGE_BYTES != 1) {
            return false;
        }

        if (weight_format != im2p_weight_format_t::q8_channel_dense_sidecar || B == nullptr ||
            J == 0 || K == 0 || sB != K || J > std::numeric_limits<size_t>::max() / K ||
            weight_channel_scales == nullptr || weight_channel_scale_count != J ||
            has_q8_channel_row_metadata() || B_scales != nullptr || weight_i8_scale_active ||
            q8_channel_scale_source_count() != 1) {
            return false;
        }

        for (size_t row = 0; row < J; ++row) {
            if (!std::isfinite(weight_channel_scales[row])) {
                return false;
            }
        }

        return true;
    }

    inline const block_q8_h2 * q8_h2_block(size_t row, size_t block) const {
        if (q8_h2_blocks == nullptr || q8_h2_blocks_per_row == 0 || row >= J ||
            block >= q8_h2_blocks_per_row ||
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
        if (weight_format != im2p_weight_format_t::q8_h2 || q8_h2_blocks == nullptr || J == 0 ||
            K == 0 || K % QK8_H2 != 0 || q8_h2_blocks_per_row != K / QK8_H2 ||
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

    inline bool has_no_affine_metadata() const {
        return B == nullptr && B_blocks == nullptr && B_scales == nullptr && c_b == nullptr &&
               s_rf == nullptr && R == nullptr && s_rf_stripe == nullptr && R_stripe == nullptr &&
               unpacked.q_qs.empty() && unpacked.c_b.empty() && unpacked.s_rf.empty() &&
               unpacked.R.empty() && unpacked.s_rf_stripe.empty() && unpacked.R_stripe.empty() &&
               unpacked.q.empty() && unpacked.scales.empty() && unpacked.blocks == nullptr;
    }

    inline const block_q8_hp1 * q8_hp1_block(size_t row, size_t block) const {
        if (q8_hp1_blocks == nullptr || q8_hp1_blocks_per_row == 0 || row >= J ||
            block >= q8_hp1_blocks_per_row ||
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
        if (weight_format != im2p_weight_format_t::q8_hp1 || q8_hp1_blocks == nullptr || J == 0 ||
            K == 0 || K % QK8_HP != 0 || q8_hp1_blocks_per_row != K / QK8_HP ||
            reinterpret_cast<uintptr_t>(q8_hp1_blocks) % alignof(block_q8_hp1) != 0 ||
            J > std::numeric_limits<size_t>::max() / q8_hp1_blocks_per_row ||
            q8_hp1_block_count != J * q8_hp1_blocks_per_row ||
            q8_hp1_block_count > std::numeric_limits<size_t>::max() / sizeof(block_q8_hp1) ||
            native_weight_bytes < q8_hp1_block_count * sizeof(block_q8_hp1) ||
            q8_hp2_blocks != nullptr || q8_hp2_block_count != 0 || q8_hp2_blocks_per_row != 0 ||
            !has_no_affine_metadata()) {
            return false;
        }

        // Payload validity is guaranteed at quantize time (llama-quant.cpp) and, if requested,
        // once at load (check_tensors). Weights are immutable during inference and the HP kernel
        // is robust to malformed data (ldexp_fast_pos fallbacks, m==INT16_MIN handled), so we do
        // not re-scan the whole tensor via ggml_validate_row_data on every matmul call.
        return true;
    }

    inline const block_q8_hp2 * q8_hp2_block(size_t row, size_t block) const {
        if (q8_hp2_blocks == nullptr || q8_hp2_blocks_per_row == 0 || row >= J ||
            block >= q8_hp2_blocks_per_row ||
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
        if (weight_format != im2p_weight_format_t::q8_hp2 || q8_hp2_blocks == nullptr || J == 0 ||
            K == 0 || K % QK8_HP != 0 || q8_hp2_blocks_per_row != K / QK8_HP ||
            reinterpret_cast<uintptr_t>(q8_hp2_blocks) % alignof(block_q8_hp2) != 0 ||
            J > std::numeric_limits<size_t>::max() / q8_hp2_blocks_per_row ||
            q8_hp2_block_count != J * q8_hp2_blocks_per_row ||
            q8_hp2_block_count > std::numeric_limits<size_t>::max() / sizeof(block_q8_hp2) ||
            q8_hp1_blocks != nullptr || q8_hp1_block_count != 0 || q8_hp1_blocks_per_row != 0 ||
            !has_no_affine_metadata()) {
            return false;
        }

        // See has_q8_hp1_im2p_contract: payload is validated at quantize/load time, not per matmul.
        return true;
    }

    inline bool has_native_matched_width_contract() const {
        if (J == 0 || K == 0 || K % 32 != 0 || native_blocks_per_row != K / 32 ||
            J > std::numeric_limits<size_t>::max() / native_blocks_per_row ||
            native_block_count != J * native_blocks_per_row) {
            return false;
        }

        const void * blocks      = nullptr;
        size_t       alignment   = 1;
        size_t       block_bytes = 0;
        switch (weight_format) {
        case im2p_weight_format_t::q4_h0:
            blocks      = q4_h0_blocks;
            alignment   = alignof(block_q4_h0);
            block_bytes = sizeof(block_q4_h0);
            break;
        case im2p_weight_format_t::q4_hp1:
            blocks      = q4_hp1_blocks;
            alignment   = alignof(block_q4_hp1);
            block_bytes = sizeof(block_q4_hp1);
            break;
        case im2p_weight_format_t::q16_h0:
            blocks      = q16_h0_blocks;
            alignment   = alignof(block_q16_h0);
            block_bytes = sizeof(block_q16_h0);
            break;
        case im2p_weight_format_t::q16_h1:
            blocks      = q16_h1_blocks;
            alignment   = alignof(block_q16_h1);
            block_bytes = sizeof(block_q16_h1);
            break;
        case im2p_weight_format_t::q16_hp1:
            blocks      = q16_hp1_blocks;
            alignment   = alignof(block_q16_hp1);
            block_bytes = sizeof(block_q16_hp1);
            break;
        default:
            return false;
        }
        return blocks != nullptr &&
               native_block_count <= std::numeric_limits<size_t>::max() / block_bytes &&
               native_weight_bytes >= native_block_count * block_bytes &&
               reinterpret_cast<uintptr_t>(blocks) % alignment == 0;
    }

    // GGUF Q8_0 blocks used as stored (H0): fp16 d is the floating block scale.
    inline bool has_q8_h0_contract() const {
        return weight_format == im2p_weight_format_t::q8_h0 && B_blocks != nullptr && J != 0 &&
               K != 0 && K % QK8_0 == 0 && blocks_K == K / QK8_0 && blocks_J == J &&
               J <= std::numeric_limits<size_t>::max() / blocks_K &&
               J * blocks_K <= std::numeric_limits<size_t>::max() / sizeof(block_q8_0) &&
               native_weight_bytes >= J * blocks_K * sizeof(block_q8_0) &&
               reinterpret_cast<uintptr_t>(B_blocks) % alignof(block_q8_0) == 0;
    }

} ggml_gemmini_args_t;

#if defined(GGML_GEMMINI_TEST_OBSERVER)
namespace ggml::gemmini {
using test_i_observer_t = void (*)(const char * consumer, size_t I, void * user_data);

GGML_API void set_test_i_observer(test_i_observer_t observer, void * user_data);
} // namespace ggml::gemmini
#endif

#if defined(GGML_GEMMINI_TESTING) || defined(GGML_GEMMINI_TEST_OBSERVER)
namespace ggml::gemmini {
enum class TestSemanticLayerSite : uint8_t {
    fp_facade,
    physical_auto_fp,
    physical_set_tile_ws,
    physical_im2p_impl,
    physical_auto_im2p,
    physical_baseline_dense,
};
using test_semantic_layer_observer_t = bool (*)(TestSemanticLayerSite site,
                                                const char *          layer,
                                                void *                user_data);

GGML_API void set_test_semantic_layer_observer(test_semantic_layer_observer_t observer,
                                               void *                         user_data);
GGML_API bool test_observe_semantic_layer(TestSemanticLayerSite site, const char * layer);
GGML_API bool test_probe_physical_layer_sites(const ggml_gemmini_args_t & args);
GGML_API bool test_probe_physical_null_args();
GGML_API bool test_probe_fp_facade_layer(const std::string & layer);
} // namespace ggml::gemmini
#endif

#if defined(GGML_GEMMINI_TESTING)
namespace ggml::gemmini {
GGML_API bool test_hp1_native_weight_admission_contract();
GGML_API std::string test_resolve_backend_matmul_layer(std::string_view model_arch,
                                                       std::string_view weight_name,
                                                       std::string_view input_name,
                                                       std::string_view consumer_name);
GGML_API void        test_reset_unclassified_matmul_diagnostics();
GGML_API size_t      test_unclassified_matmul_diagnostic_count();
} // namespace ggml::gemmini
#endif
