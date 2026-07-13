#pragma once

#include "../../ggml-gemmini-args.h"

#include <cstddef>

namespace ggml::gemmini::quants::dec
{
    enum class WeightLayout
    {
        KxJ_RowMajor,
        JxK_ColMajor,
    };

    inline bool is_q8_h1_native_args(const ggml_gemmini_args_t &args)
    {
        return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h1_native &&
               args.q8_h1_native_blocks != nullptr;
    }

    inline bool is_q8_h2_args(const ggml_gemmini_args_t &args)
    {
        return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2 &&
               args.q8_h2_blocks != nullptr &&
               args.q8_h2_blocks_per_row > 0;
    }

    inline bool is_q8_h1_weight_args(const ggml_gemmini_args_t &args)
    {
        return is_q8_h1_native_args(args) ||
               (args.B &&
               !args.B_scales &&
               args.c_b &&
               ((args.stripe_J > 1) || (args.s_rf && args.R)) &&
               args.blocks_per_row > 0);
    }

    inline WeightLayout resolve_weight_layout(const ggml_gemmini_args_t &args)
    {
        if (is_q8_h1_weight_args(args) || args.transpose_B)
            return WeightLayout::JxK_ColMajor;

        return WeightLayout::KxJ_RowMajor;
    }

    inline size_t resolve_weight_stride_elems(const ggml_gemmini_args_t &args)
    {
        if (is_q8_h1_weight_args(args))
            return args.K;

        const size_t fallback_stride = args.transpose_B ? args.K : args.J;
        return args.sB ? args.sB : fallback_stride;
    }
}
