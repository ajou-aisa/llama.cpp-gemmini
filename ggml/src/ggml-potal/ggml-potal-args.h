// ggml-potal/ggml-potal-args.h
//
// Copied from ggml-gemmini/ggml-gemmini-args.h and reduced to what the PoTal
// path uses: geometry, the quantized activation buffer and its metadata, the
// Q8_HP1 weight view, the F32 output view and the tiling result. Field names,
// order and helper semantics are kept; Gemmini-only formats, residual/RMD,
// stripe, optrace, evaluation and IM2P fields are removed. No gemmini.h or
// gemmini_params.h: PoTal is A8W8 (elem_t = int8_t) by its parameter contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "quants/act/buffer.hpp"
#include "quants/act/meta.hpp"

#include <ggml.h>
#ifndef GGML_COMMON_DECL
#define GGML_POTAL_ARGS_DEFINE_GGML_COMMON
#define GGML_COMMON_DECL_CPP
#endif
#include "../ggml-common.h"
#ifdef GGML_POTAL_ARGS_DEFINE_GGML_COMMON
#undef GGML_COMMON_DECL_CPP
#undef GGML_POTAL_ARGS_DEFINE_GGML_COMMON
#endif

namespace act = ggml::potal::quants::act;

typedef struct ggml_potal_args_t {
    // Same value as ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1.
    enum class weight_format_t : uint8_t {
        q8_hp1 = 4,
    };

    size_t I = 0;
    size_t J = 0;
    size_t K = 0;

    // A owns its backing (row stride K); C is a borrowed view.
    act::QuantizedActivationBuffer A;
    void *                         C = nullptr;

    size_t sA = 0;
    size_t sB = 0;
    size_t sC = 0;

    act::Meta act_quant{};

    weight_format_t weight_format = weight_format_t::q8_hp1;

    // Borrowed Q8_HP1 blocks: row j (output column) holds q8_hp1_blocks_per_row blocks.
    const block_q8_hp1 * q8_hp1_blocks         = nullptr;
    size_t               q8_hp1_block_count    = 0;
    size_t               q8_hp1_blocks_per_row = 0;
    // Checked available backing extent; must cover every declared block.
    size_t native_weight_bytes = 0;

    uint32_t block_size_k = QK8_HP;

    // Borrowed F32 destination.
    float * f_out            = nullptr;
    size_t  col_stride_f_out = 0;
    size_t  stride_f_out     = 0;

    // model_arch is borrowed; matmul_layer owns its identity.
    const char * model_arch = nullptr;
    std::string  matmul_layer;

    // Tiling counts in DIM units (potal_tiled_matmul_auto result).
    size_t tile_I = 0;
    size_t tile_J = 0;
    size_t tile_K = 0;

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

    // ggml_gemmini_args_t::has_q8_hp1_im2p_contract without the other formats.
    inline bool has_q8_hp1_contract() const {
        return weight_format == weight_format_t::q8_hp1 && q8_hp1_blocks != nullptr && J != 0 &&
               K != 0 && K % QK8_HP == 0 && q8_hp1_blocks_per_row == K / QK8_HP &&
               reinterpret_cast<uintptr_t>(q8_hp1_blocks) % alignof(block_q8_hp1) == 0 &&
               J <= std::numeric_limits<size_t>::max() / q8_hp1_blocks_per_row &&
               q8_hp1_block_count == J * q8_hp1_blocks_per_row &&
               q8_hp1_block_count <= std::numeric_limits<size_t>::max() / sizeof(block_q8_hp1) &&
               native_weight_bytes >= q8_hp1_block_count * sizeof(block_q8_hp1);
    }
} ggml_potal_args_t;
