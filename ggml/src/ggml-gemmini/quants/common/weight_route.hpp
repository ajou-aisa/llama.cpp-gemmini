#pragma once

#include "../../ggml-gemmini-args.h"

#include <cstddef>
#include <cstdint>

namespace ggml::gemmini::quants::wroute
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
        bool on_demand_mode = false;
        bool supported = true;
    };

    enum class WeightScaleInfoMode
    {
        CommonOutput,
        Residual,
    };

    WeightScaleInfo build_weight_scale_info(
        const ggml_gemmini_args_t &args,
        WeightScaleInfoMode mode);

    enum class WeightLayout
    {
        KxJ_RowMajor,
        JxK_ColMajor,
    };

    enum class WeightRouteKind
    {
        Unsupported,
        Dense,
        Q8ChannelDirect,
        Q8ChannelSidecar,
        H0,
        H1,
        HP1,
    };

    enum class WeightScaleDomain
    {
        None,
        FloatingBlock,
        IntegerBlockTimesColumn,
    };

    enum class WeightExecutionPath
    {
        CpuDirect,
        Compact,
    };

    enum class WeightRouteStatus
    {
        Success,
        UnsupportedFormat,
        InvalidMetadata,
        UnsupportedExecution,
    };

    struct WeightRoutePlan
    {
        WeightRouteKind route = WeightRouteKind::Unsupported;
        WeightLayout layout = WeightLayout::KxJ_RowMajor;
        WeightScaleInfo scales{};
        WeightScaleDomain scale_domain = WeightScaleDomain::None;
        WeightRouteStatus status = WeightRouteStatus::UnsupportedFormat;
        size_t weight_stride = 0;
        const char *reject_reason = "unsupported weight format";
        uint8_t weight_bits = 0;
        bool native_weight_blocks = false;
        bool cpu_direct_capable = false;
        bool compact_capable = false;
        bool valid = false;
    };

    WeightRoutePlan resolve_weight_route_plan(
        const ggml_gemmini_args_t &args,
        WeightScaleInfoMode mode);

    WeightRouteStatus weight_route_status(
        const WeightRoutePlan &plan,
        WeightExecutionPath path);

    const char *weight_route_status_name(WeightRouteStatus status);

    const char *weight_route_kind_name(const WeightRoutePlan &plan);

    const char *weight_scale_mode_name(const WeightRoutePlan &plan);

    bool route_covers_k(const WeightRoutePlan &plan, size_t k_count);

    bool route_block_for_range(
        const WeightRoutePlan &plan,
        size_t k_offset,
        size_t block_k,
        size_t &block_index);

    float route_weight_scale(
        const WeightRoutePlan &plan,
        const ggml_gemmini_args_t &args,
        size_t j,
        size_t block_index);

    // Integer-domain residual executors require the scale to factor as:
    //
    //     weight_scale(j, block) == integer_block(j, block) * column_float(j)
    //
    // H1 and HP1 satisfy this for every supported width. H0 deliberately does
    // not: its arbitrary floating block scale belongs to the CPU-direct path.
    bool route_supports_integer_block_scale(const WeightRoutePlan &plan);

    uint64_t route_block_scale(
        const WeightRoutePlan &plan,
        const ggml_gemmini_args_t &args,
        size_t j,
        size_t block_index);

    float route_column_scale(
        const WeightRoutePlan &plan,
        const ggml_gemmini_args_t &args,
        size_t j);

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

    inline bool has_q8_hp1_native_contract(const ggml_gemmini_args_t &args)
    {
        return args.B == nullptr &&
               !args.weight_i8_scale_active &&
               is_q8_hp1_args(args) &&
               args.has_q8_hp1_im2p_contract();
    }

    inline bool has_q8_hp2_native_contract(const ggml_gemmini_args_t &args)
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

    inline bool is_native_matched_width_format(const ggml_gemmini_args_t &args)
    {
        using Format = ggml_gemmini_args_t::im2p_weight_format_t;
        switch (args.weight_format) {
            case Format::q4_h0:
            case Format::q4_h1:
            case Format::q4_hp1:
            case Format::q16_h0:
            case Format::q16_h1:
            case Format::q16_hp1:
                return true;
            default:
                return false;
        }
    }

    inline WeightLayout resolve_weight_layout(const ggml_gemmini_args_t &args)
    {
        if (is_q8_channel_direct_read_args(args) || is_q8_channel_dense_sidecar_args(args))
            return WeightLayout::JxK_ColMajor;

        if (is_native_matched_width_format(args) || is_q8_hp1_args(args) ||
            is_q8_hp2_args(args) || is_q8_h1_weight_args(args) || args.transpose_B)
            return WeightLayout::JxK_ColMajor;

        return WeightLayout::KxJ_RowMajor;
    }

    inline size_t resolve_weight_stride_elems(const ggml_gemmini_args_t &args)
    {
        if (is_native_matched_width_format(args) || is_q8_hp1_args(args) ||
            is_q8_hp2_args(args) || is_q8_h1_weight_args(args))
            return args.K;

        const size_t fallback_stride = args.transpose_B ? args.K : args.J;
        return args.sB ? args.sB : fallback_stride;
    }
}
