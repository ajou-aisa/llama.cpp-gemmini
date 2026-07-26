#pragma once

#include "../../ggml-gemmini-args.h"

#include <cstddef>

namespace ggml::gemmini::quants::dec
{
    struct WeightScaleInfo
    {
        const float *data = nullptr;
        size_t rows = 0;
        size_t cols = 0;
        size_t block_size = 0;
        float scalar = 1.0f;
        bool scalar_mode = false;
        bool row_header_mode = false;
        bool channel_mode = false;
        bool supported = true;
    };

    enum class WeightScaleInfoMode
    {
        CommonOutput,
        Dec,
    };

    WeightScaleInfo build_weight_scale_info(
        const ggml_gemmini_args_t &args,
        WeightScaleInfoMode mode);

    enum class WeightLayout
    {
        KxJ_RowMajor,
        JxK_ColMajor,
    };

    inline bool is_q8_h1_args(const ggml_gemmini_args_t &args)
    {
        return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h1 &&
               args.q8_h1_blocks != nullptr;
    }

    inline bool is_q8_h2_args(const ggml_gemmini_args_t &args)
    {
        return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2 &&
               args.q8_h2_blocks != nullptr &&
               args.q8_h2_blocks_per_row > 0;
    }

    inline bool is_q8_hp1_args(const ggml_gemmini_args_t &args)
    {
        return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1 &&
               args.q8_hp1_blocks != nullptr &&
               args.q8_hp1_block_count > 0 &&
               args.q8_hp1_blocks_per_row > 0;
    }

    inline bool is_q8_hp2_args(const ggml_gemmini_args_t &args)
    {
        return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2 &&
               args.q8_hp2_blocks != nullptr &&
               args.q8_hp2_block_count > 0 &&
               args.q8_hp2_blocks_per_row > 0;
    }

    inline bool is_q8_channel_direct_read_args(const ggml_gemmini_args_t &args)
    {
        return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel ||
               args.has_q8_channel_row_metadata();
    }

    inline bool is_q8_channel_dense_sidecar_args(const ggml_gemmini_args_t &args)
    {
        return args.has_q8_channel_dense_sidecar_contract();
    }

    inline bool has_q8_hp1_native_dec_contract(const ggml_gemmini_args_t &args)
    {
        return args.B == nullptr &&
               !args.weight_i8_scale_active &&
               is_q8_hp1_args(args) &&
               args.has_q8_hp1_im2p_contract();
    }

    inline bool has_q8_hp2_native_dec_contract(const ggml_gemmini_args_t &args)
    {
        return args.B == nullptr &&
               !args.weight_i8_scale_active &&
               is_q8_hp2_args(args) &&
               args.has_q8_hp2_im2p_contract();
    }

    inline bool is_q8_h1_weight_args(const ggml_gemmini_args_t &args)
    {
        return is_q8_h1_args(args) ||
               (args.B &&
               !args.B_scales &&
               args.c_b &&
               ((args.stripe_J > 1) || (args.s_rf && args.R)) &&
               args.blocks_per_row > 0);
    }

    inline WeightLayout resolve_weight_layout(const ggml_gemmini_args_t &args)
    {
        if (is_q8_channel_direct_read_args(args) || is_q8_channel_dense_sidecar_args(args))
            return WeightLayout::JxK_ColMajor;

        if (is_q8_hp1_args(args) || is_q8_hp2_args(args) ||
            is_q8_h1_weight_args(args) || args.transpose_B)
            return WeightLayout::JxK_ColMajor;

        return WeightLayout::KxJ_RowMajor;
    }

    inline size_t resolve_weight_stride_elems(const ggml_gemmini_args_t &args)
    {
        if (is_q8_hp1_args(args) || is_q8_hp2_args(args))
            return args.K;

        if (is_q8_h1_weight_args(args))
            return args.K;

        const size_t fallback_stride = args.transpose_B ? args.K : args.J;
        return args.sB ? args.sB : fallback_stride;
    }
}
