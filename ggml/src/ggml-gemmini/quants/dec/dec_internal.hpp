#pragma once

#include "../../ggml-gemmini-args.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml::gemmini::quants::dec
{
    inline constexpr size_t kDecGroupSizeK = 32;

    struct ResidualGroupEntry
    {
        uint32_t row;
        uint32_t k;
        int32_t residual;
    };

    struct ActiveRowGroup
    {
        uint32_t row;
        uint32_t k_group;
        size_t entry_begin;
        size_t entry_end;
    };

    inline void build_active_row_groups(
        std::vector<ResidualGroupEntry> &entries,
        std::vector<ActiveRowGroup> &groups)
    {
        std::sort(entries.begin(), entries.end(), [](const ResidualGroupEntry &lhs, const ResidualGroupEntry &rhs)
        {
            if (lhs.row != rhs.row)
                return lhs.row < rhs.row;
            const uint32_t lhs_group = lhs.k / kDecGroupSizeK;
            const uint32_t rhs_group = rhs.k / kDecGroupSizeK;
            if (lhs_group != rhs_group)
                return lhs_group < rhs_group;
            if (lhs.k != rhs.k)
                return lhs.k < rhs.k;
            return lhs.residual < rhs.residual;
        });

        groups.clear();
        for (size_t begin = 0; begin < entries.size();)
        {
            const uint32_t row = entries[begin].row;
            const uint32_t k_group = entries[begin].k / kDecGroupSizeK;
            size_t end = begin + 1;
            while (end < entries.size() && entries[end].row == row &&
                   entries[end].k / kDecGroupSizeK == k_group)
                ++end;
            groups.push_back({row, k_group, begin, end});
            begin = end;
        }
    }

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

    enum class DecWeightRoute
    {
        Unsupported,
        Dense,
        Q8ChannelDirect,
        Q8ChannelSidecar,
        Q8H1,
        Q8H2,
        Q8HP1,
        Q8HP2,
    };

    struct DecRoutePlan
    {
        DecWeightRoute route = DecWeightRoute::Unsupported;
        WeightLayout layout = WeightLayout::KxJ_RowMajor;
        WeightScaleInfo scales{};
        size_t weight_stride = 0;
        const char *reject_reason = "unsupported weight format";
        bool native_weight_blocks = false;
        bool valid = false;
    };

    DecRoutePlan resolve_dec_route_plan(
        const ggml_gemmini_args_t &args,
        WeightScaleInfoMode mode);

    const char *dec_route_name(const DecRoutePlan &plan);

    const char *dec_scale_mode_name(const DecRoutePlan &plan);

    bool dec_route_covers_k(const DecRoutePlan &plan, size_t k_count);

    bool dec_route_block_for_range(
        const DecRoutePlan &plan,
        size_t k_offset,
        size_t block_k,
        size_t &block_index);

    float dec_route_weight_scale(
        const DecRoutePlan &plan,
        const ggml_gemmini_args_t &args,
        size_t j,
        size_t block_index);

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
