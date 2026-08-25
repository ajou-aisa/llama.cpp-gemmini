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
#include "ggml-gemmini-geometry.hpp"
#include "quants/act/meta.hpp"
#include "quants/act/types.hpp"

namespace act = ggml::gemmini::quants::act;

namespace ggml::gemmini::quants::act {

struct QuantizedActivationBuffer {
    std::shared_ptr<std::vector<uint8_t>> bytes;
    uint8_t bits = 8;
    size_t rows = 0;
    size_t cols = 0;
    size_t row_stride_bytes = 0;
    size_t row_offset = 0;

    bool allocate(size_t r, size_t c, uint8_t b) {
        if (r == 0 || c == 0 || (b != 4 && b != 8 && b != 16)) return false;

        size_t staged_row_stride = c;
        if (b == 16) {
            if (c > std::numeric_limits<size_t>::max() / sizeof(int16_t)) return false;
            staged_row_stride = c * sizeof(int16_t);
        }
        if (r > std::numeric_limits<size_t>::max() / staged_row_stride) return false;
        const size_t byte_count = r * staged_row_stride;
        if (byte_count > std::vector<uint8_t>{}.max_size()) return false;

        std::shared_ptr<std::vector<uint8_t>> staged_bytes;
        try {
            staged_bytes = std::make_shared<std::vector<uint8_t>>(byte_count, 0);
        } catch (const std::bad_alloc &) {
            return false;
        } catch (const std::length_error &) {
            return false;
        }

        bytes = std::move(staged_bytes);
        bits = b;
        rows = r;
        cols = c;
        row_stride_bytes = staged_row_stride;
        row_offset = 0;
        return true;
    }

    bool valid() const {
        return bytes != nullptr && !bytes->empty() && bits != 0 && rows != 0 && cols != 0;
    }

    int32_t get(size_t row, size_t col) const {
        if (!valid() || row >= rows || col >= cols) return 0;
        const size_t actual_row = row_offset + row;
        const size_t byte_offset = actual_row * row_stride_bytes;
        if (bits == 4 || bits == 8) {
            const int8_t v = static_cast<int8_t>((*bytes)[byte_offset + col]);
            return static_cast<int32_t>(v);
        } else { // 16-bit
            const size_t idx = byte_offset + col * 2;
            int16_t v = 0;
            std::memcpy(&v, &(*bytes)[idx], sizeof(v));
            return static_cast<int32_t>(v);
        }
    }

    bool set(size_t row, size_t col, int32_t value) {
        if (!valid() || row >= rows || col >= cols) return false;
        int32_t qmin = -(int32_t{1} << (bits - 1));
        int32_t qmax = (int32_t{1} << (bits - 1)) - 1;
        if (value < qmin || value > qmax) return false;
        const size_t actual_row = row_offset + row;
        const size_t byte_offset = actual_row * row_stride_bytes;
        if (bits == 4 || bits == 8) {
            (*bytes)[byte_offset + col] = static_cast<uint8_t>(value);
        } else { // 16-bit
            const size_t idx = byte_offset + col * 2;
            int16_t v = static_cast<int16_t>(value);
            std::memcpy(&(*bytes)[idx], &v, sizeof(v));
        }
        return true;
    }

    QuantizedActivationBuffer slice_rows(size_t begin, size_t count) const {
        QuantizedActivationBuffer s;
        s.bytes = bytes;
        s.bits = bits;
        s.rows = count;
        s.cols = cols;
        s.row_stride_bytes = row_stride_bytes;
        s.row_offset = row_offset + begin;
        return s;
    }

    void zero_fill() {
        if (bytes) std::fill(bytes->begin(), bytes->end(), 0);
    }

    const uint8_t *raw_data() const {
      if (!bytes || (row_stride_bytes != 0 &&
                     row_offset > std::numeric_limits<size_t>::max() /
                                      row_stride_bytes)) {
        return nullptr;
      }
      const size_t offset = row_offset * row_stride_bytes;
      return offset <= bytes->size() ? bytes->data() + offset : nullptr;
    }

    size_t raw_size() const {
      if (!bytes || (row_stride_bytes != 0 &&
                     row_offset > std::numeric_limits<size_t>::max() /
                                      row_stride_bytes)) {
        return 0;
      }
      const size_t offset = row_offset * row_stride_bytes;
      return offset <= bytes->size() ? bytes->size() - offset : 0;
    }

    // Backward-compatible conversion for 8bit hardware path.
    // Only valid when bits == 8; returns nullptr otherwise.
    operator elem_t*() {
      return bits == 8
                 ? reinterpret_cast<elem_t *>(const_cast<uint8_t *>(raw_data()))
                 : nullptr;
    }
    operator const elem_t*() const {
      return bits == 8 ? reinterpret_cast<const elem_t *>(raw_data()) : nullptr;
    }
};

} // namespace ggml::gemmini::quants::act

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
static_assert(GGML_GEMMINI_ACTIVATION_BITS == 16
                  ? std::is_same_v<elem_t, int16_t>
                  : std::is_same_v<elem_t, int8_t>,
              "elem_t must be int16_t only for A16/W16, otherwise int8_t");

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
    if (new_exp >= 0xFF) return std::numeric_limits<float>::max();
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
        q8_channel = 6,
        q8_channel_dense_sidecar = 7,
        q4_h0 = 8,
        q4_h1 = 9,
        q4_hp1 = 10,
        q16_h0 = 11,
        q16_h1 = 12,
        q16_hp1 = 13,
    };

    // tiled_matmul_auto args
    size_t I = 0;
    size_t J = 0;
    size_t K = 0;

    //elements
    act::QuantizedActivationBuffer A;
    elem_t *B = nullptr;
    const float *A_fp32 = nullptr;
    const float *B_fp32 = nullptr;
    void *C = nullptr;
    const void *D = nullptr;

    size_t sA = 0;
    size_t sB = 0;
    size_t sC = 0;
    size_t sD = 0;

    size_t activation_row_offset = 0;
    size_t activation_rows_per_stripe = 0;

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
    act::ResidualRoute residual_route = act::ResidualRoute::ws_packet;
    const ggml::gemmini::quants::act::exsia::StripeReadySink *exsia_stripe_ready_sink = nullptr;

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
    const float *weight_channel_scales = nullptr;
    size_t weight_channel_scale_count = 0;

    const uint8_t *q8_channel_row_base = nullptr;
    size_t q8_channel_row_stride = 0;
    size_t q8_channel_row_count = 0;

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

    // Native matched-width FULL-provider formats. Each block spans 32 K
    // elements; Q4 callbacks unpack signed nibbles and Q16 callbacks preserve
    // the int16 codes for the canonical typed provider ABI.
    const block_q4_h0 *q4_h0_blocks = nullptr;
    const block_q4_h1 *q4_h1_blocks = nullptr;
    const block_q4_hp1 *q4_hp1_blocks = nullptr;
    const block_q16_h0 *q16_h0_blocks = nullptr;
    const block_q16_h1 *q16_h1_blocks = nullptr;
    const block_q16_hp1 *q16_hp1_blocks = nullptr;
    size_t native_block_count = 0;
    size_t native_blocks_per_row = 0;
    // Checked available backing extent for native block readers. Native
    // dispatch requires a nonzero extent covering every declared block.
    size_t native_weight_bytes = 0;

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
    uint8_t reserved_layer_metadata = 0;
    const char *model_arch = nullptr;

    // Gemmini auto-tiling counts in DIM units (multiply by DIM to get element counts).
    size_t tile_I = 0;
    size_t tile_J = 0;
    size_t tile_K = 0;

    inline ggml::gemmini::GemminiGeometryResult activation_geometry() const {
        return ggml::gemmini::make_gemmini_geometry(
            {{I, J, K}, {tile_I, tile_J, tile_K}, DIM});
    }

    inline bool activation_geometry_matches(
        ggml::gemmini::GemminiGeometry &geometry) const {
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
        return ggml::gemmini::make_gemmini_geometry(
            {{I, J, K},
             {tile_or_full_extent(tile_I, I, array_dim),
              tile_or_full_extent(tile_J, J, array_dim),
              tile_or_full_extent(tile_K, K, array_dim)},
             array_dim});
    }

    inline bool activation_quant_geometry_matches(
        ggml::gemmini::GemminiGeometry &geometry) const {
        const auto result = activation_quant_geometry();
        if (!result.ok() ||
            (activation_rows_per_stripe != 0 &&
             activation_rows_per_stripe != result.geometry.stripe_rows))
            return false;
        geometry = result.geometry;
        return true;
    }

    inline size_t stripe_J_or_rowwise_elems() const { return stripe_J > 0 ? stripe_J : 1; }
    inline bool stripe_mode_matches_tile_j(size_t tile_J_elems) const {
        return stripe_J <= 1 || (tile_J_elems > 0 && stripe_J == tile_J_elems);
    }
    // Gemmini call metadata (for debugging/validation)
    size_t gemmini_call_k_logical = 0;
    size_t gemmini_call_k_aligned = 0;
    size_t gemmini_call_tile_k_elems = 0;

    std::string matmul_layer;

    inline const uint8_t *q8_channel_row(size_t row) const {
        if (q8_channel_row_base == nullptr || q8_channel_row_stride == 0 ||
            row >= q8_channel_row_count ||
            row > std::numeric_limits<size_t>::max() / q8_channel_row_stride) {
            return nullptr;
        }

        return q8_channel_row_base + row * q8_channel_row_stride;
    }

    inline const elem_t *q8_channel_payload(size_t row) const {
        const uint8_t *row_base = q8_channel_row(row);
        return row_base == nullptr ? nullptr :
            reinterpret_cast<const elem_t *>(row_base + sizeof(float));
    }

    inline float q8_channel_scale(size_t row) const {
        const uint8_t *row_base = q8_channel_row(row);
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
        if constexpr (GGML_GEMMINI_WEIGHT_BITS != 8 ||
                      GGML_GEMMINI_WEIGHT_STORAGE_BYTES != 1) {
            return false;
        }

        if (weight_format != im2p_weight_format_t::q8_channel ||
            q8_channel_row_base == nullptr || J == 0 || K == 0 ||
            q8_channel_row_count != J || q8_channel_row_stride == 0 ||
            K > std::numeric_limits<size_t>::max() - sizeof(float) ||
            q8_channel_row_stride != sizeof(float) + K ||
            sB != q8_channel_row_stride || B == nullptr ||
            q8_channel_row_count > std::numeric_limits<size_t>::max() / q8_channel_row_stride ||
            B_scales != nullptr || weight_channel_scales != nullptr ||
            weight_channel_scale_count != 0 || weight_i8_scale_active ||
            q8_channel_scale_source_count() != 1 ||
            q8_channel_payload(0) != B) {
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
        if constexpr (GGML_GEMMINI_WEIGHT_BITS != 8 ||
                      GGML_GEMMINI_WEIGHT_STORAGE_BYTES != 1) {
            return false;
        }

        if (weight_format != im2p_weight_format_t::q8_channel_dense_sidecar ||
            B == nullptr || J == 0 || K == 0 || sB != K ||
            J > std::numeric_limits<size_t>::max() / K ||
            weight_channel_scales == nullptr || weight_channel_scale_count != J ||
            has_q8_channel_row_metadata() || B_scales != nullptr ||
            weight_i8_scale_active || q8_channel_scale_source_count() != 1) {
            return false;
        }

        for (size_t row = 0; row < J; ++row) {
            if (!std::isfinite(weight_channel_scales[row])) {
                return false;
            }
        }

        return true;
    }

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
            native_weight_bytes < q8_hp1_block_count * sizeof(block_q8_hp1) ||
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

    inline bool has_native_matched_width_contract() const {
        if (J == 0 || K == 0 || K % 32 != 0 ||
            native_blocks_per_row != K / 32 ||
            J > std::numeric_limits<size_t>::max() / native_blocks_per_row ||
            native_block_count != J * native_blocks_per_row) {
            return false;
        }

        const void *blocks = nullptr;
        size_t alignment = 1;
        size_t block_bytes = 0;
        switch (weight_format) {
        case im2p_weight_format_t::q4_h0:
            blocks = q4_h0_blocks;
            alignment = alignof(block_q4_h0);
            block_bytes = sizeof(block_q4_h0);
            break;
        case im2p_weight_format_t::q4_h1:
            blocks = q4_h1_blocks;
            alignment = alignof(block_q4_h1);
            block_bytes = sizeof(block_q4_h1);
            break;
        case im2p_weight_format_t::q4_hp1:
            blocks = q4_hp1_blocks;
            alignment = alignof(block_q4_hp1);
            block_bytes = sizeof(block_q4_hp1);
            break;
        case im2p_weight_format_t::q16_h0:
            blocks = q16_h0_blocks;
            alignment = alignof(block_q16_h0);
            block_bytes = sizeof(block_q16_h0);
            break;
        case im2p_weight_format_t::q16_h1:
            blocks = q16_h1_blocks;
            alignment = alignof(block_q16_h1);
            block_bytes = sizeof(block_q16_h1);
            break;
        case im2p_weight_format_t::q16_hp1:
            blocks = q16_hp1_blocks;
            alignment = alignof(block_q16_hp1);
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

    inline bool has_q8_h1_im2p_contract() const {
        if (weight_format != im2p_weight_format_t::q8_h1 ||
            q8_h1_blocks == nullptr || J == 0 || K == 0 ||
            blocks_per_row == 0 || q8_h1_block_count == 0 || q8_h1_rows < J ||
            reinterpret_cast<uintptr_t>(q8_h1_blocks) % alignof(block_q8_h1) != 0 ||
            K > std::numeric_limits<size_t>::max() - (QK8_0 - 1) ||
            blocks_per_row != (K + QK8_0 - 1) / QK8_0 ||
            J > std::numeric_limits<size_t>::max() / blocks_per_row ||
            q8_h1_block_count > std::numeric_limits<size_t>::max() / sizeof(block_q8_h1) ||
            native_weight_bytes < q8_h1_block_count * sizeof(block_q8_h1)) {
            return false;
        }

        if (q8_h1_block_count < J * blocks_per_row) {
            return false;
        }

        // EXPERIMENT (contract-scan parity with Q8_HP): skip the per-row s_rf/R uniformity
        // scan so H1 pays the same O(1) per-call contract as HP1, isolating pure format cost.
        return true;
    }

} ggml_gemmini_args_t;

#if defined(GGML_GEMMINI_TEST_OBSERVER)
namespace ggml::gemmini {
using test_i_observer_t = void (*)(const char * consumer, size_t I, void * user_data);

GGML_API void set_test_i_observer(test_i_observer_t observer, void * user_data);
}
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
using test_semantic_layer_observer_t = bool (*)(
    TestSemanticLayerSite site, const char * layer, void * user_data);

GGML_API void set_test_semantic_layer_observer(
    test_semantic_layer_observer_t observer, void * user_data);
GGML_API bool test_observe_semantic_layer(
    TestSemanticLayerSite site, const char * layer);
GGML_API bool test_probe_physical_layer_sites(const ggml_gemmini_args_t & args);
GGML_API bool test_probe_physical_null_args();
GGML_API bool test_probe_fp_facade_layer(const std::string & layer);
}
#endif

#if defined(GGML_GEMMINI_TESTING)
namespace ggml::gemmini {
GGML_API std::string test_resolve_backend_matmul_layer(
    std::string_view model_arch, std::string_view weight_name,
    std::string_view input_name, std::string_view consumer_name);
GGML_API void test_reset_unclassified_matmul_diagnostics();
GGML_API size_t test_unclassified_matmul_diagnostic_count();
}
#endif
