#include "dequant.hpp"

#include "../../ggml-gemmini-args.h"
#include "../act/dispatch.hpp"
#include "weight_reader.hpp"
#include "weight_route.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace ggml::gemmini {

namespace {

namespace wreader = quants::wreader;
namespace wroute = quants::wroute;
using Format = ggml_gemmini_args_t::im2p_weight_format_t;

bool checked_mul_size(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        return false;

    out = lhs * rhs;
    return true;
}

bool finite_float_metadata(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

bool is_reader_family_format(const ggml_gemmini_args_t &args)
{
    switch (args.weight_format) {
        case Format::q4_h0:
        case Format::q4_h1:
        case Format::q4_hp1:
        case Format::q8_h0:
        case Format::q8_h1:
        case Format::q8_hp1:
        case Format::q16_h0:
        case Format::q16_h1:
        case Format::q16_hp1:
            return true;
        case Format::q8_0_unpacked_to_h1:
            return wroute::is_q8_h1_weight_args(args);
        default:
            return false;
    }
}

size_t reader_family_blocks_per_row(const ggml_gemmini_args_t &args)
{
    switch (args.weight_format) {
        case Format::q4_h0:
        case Format::q4_h1:
        case Format::q4_hp1:
        case Format::q16_h0:
        case Format::q16_h1:
        case Format::q16_hp1:
            return args.native_blocks_per_row;
        case Format::q8_h0:
            return args.blocks_K;
        case Format::q8_h1:
        case Format::q8_0_unpacked_to_h1:
            return args.blocks_per_row;
        case Format::q8_hp1:
            return args.q8_hp1_blocks_per_row;
        default:
            return 0;
    }
}

wroute::WeightScaleInfo build_weight_scale_info_impl(
    const ggml_gemmini_args_t &args,
    wroute::WeightScaleInfoMode mode)
{
    wroute::WeightScaleInfo result{};
    if (wroute::is_q8_channel_direct_read_args(args)) {
        if (!args.has_q8_channel_direct_read_contract()) {
            result.supported = false;
            return result;
        }

        result.rows = args.J;
        result.cols = 1;
        result.block_size = 1;
        result.row_header_mode = true;
        return result;
    }

    if (args.weight_format == Format::q8_channel_dense_sidecar) {
        if (!wroute::is_q8_channel_dense_sidecar_args(args)) {
            result.supported = false;
            return result;
        }

        result.data = args.weight_channel_scales;
        result.rows = args.J;
        result.cols = 1;
        result.block_size = 1;
        result.channel_mode = true;
        return result;
    }

    if (args.weight_i8_scale_active) {
        if (!args.B)
            return result;

        result.rows = args.J;
        result.cols = 1;
        result.block_size = 1;
        result.scalar = args.weight_scale;
        result.scalar_mode = true;
        return result;
    }

    if (is_reader_family_format(args)) {
        result.rows = args.J;
        result.cols = reader_family_blocks_per_row(args);
        result.block_size = 32;
        result.on_demand_mode = true;
        return result;
    }

    if (mode == wroute::WeightScaleInfoMode::CommonOutput &&
        (args.weight_format == Format::q8_h2 || args.weight_format == Format::q8_hp2)) {
        const size_t rows = args.J;
        const size_t cols = args.weight_format == Format::q8_h2 ?
            args.q8_h2_blocks_per_row : args.q8_hp2_blocks_per_row;
        size_t scale_count = 0;
        if (rows == 0 || cols == 0 || !checked_mul_size(rows, cols, scale_count))
            return result;

        static thread_local std::vector<float> weight_scales;
        weight_scales.resize(scale_count);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block_index = 0; block_index < cols; ++block_index) {
                if (args.weight_format == Format::q8_h2) {
                    const block_q8_h2 *block = args.q8_h2_block(row, block_index);
                    if (block == nullptr) {
                        result.supported = false;
                        return result;
                    }
                    weight_scales[row * cols + block_index] =
                        block->channel_scale * block->m / 255.0f;
                } else {
                    const block_q8_hp2 *block = args.q8_hp2_block(row, block_index);
                    if (block == nullptr) {
                        result.supported = false;
                        return result;
                    }
                    weight_scales[row * cols + block_index] = block->m == INT16_MIN ?
                        0.0f : gemmini_ldexp_fast_pos(
                            block->channel_scale, static_cast<int>(block->m));
                }
            }
        }
        result.data = weight_scales.data();
        result.rows = rows;
        result.cols = cols;
        result.block_size = 32;
        return result;
    }

    if (!args.B_scales)
        return result;

    result.data = args.B_scales;
    result.rows = mode == wroute::WeightScaleInfoMode::Residual ?
        args.blocks_J : (args.blocks_J ? args.blocks_J : args.J);
    result.cols = args.blocks_K;
    result.block_size = args.block_size_k ? args.block_size_k : QK8_0;
    return result;
}

bool output_offset(size_t row, size_t col, size_t row_stride, size_t col_stride, size_t &offset)
{
    size_t row_offset = 0;
    size_t col_offset = 0;
    if (!checked_mul_size(row, row_stride, row_offset) ||
        !checked_mul_size(col, col_stride, col_offset) ||
        row_offset > std::numeric_limits<size_t>::max() - col_offset) {
        return false;
    }

    offset = row_offset + col_offset;
    return true;
}

bool checked_k_end(size_t k_offset, size_t block_k, size_t &k_end)
{
    if (block_k == 0 || k_offset > std::numeric_limits<size_t>::max() - (block_k - 1))
        return false;

    k_end = k_offset + block_k - 1;
    return true;
}

bool has_complete_scale_metadata(
    const wroute::WeightScaleInfo &scales,
    const ggml_gemmini_args_t &args)
{
    if (!scales.supported)
        return false;
    if (scales.scalar_mode)
        return std::isfinite(scales.scalar);
    if (scales.row_header_mode)
        return scales.rows >= args.J;
    if (scales.channel_mode)
        return scales.data != nullptr && scales.rows >= args.J;
    return (scales.data != nullptr || scales.on_demand_mode) &&
        scales.rows >= args.J && scales.cols > 0 && scales.block_size > 0;
}

wroute::WeightRouteKind classify_route(
    const ggml_gemmini_args_t &args,
    wroute::WeightScaleInfoMode mode,
    wroute::WeightRoutePlan &plan)
{
    using Route = wroute::WeightRouteKind;
    using Status = wroute::WeightRouteStatus;
    auto accept_family = [&](Route route, uint8_t bits, bool native_blocks) {
        const bool integer_domain = route == Route::H1 ||
            (route == Route::HP1 && mode == wroute::WeightScaleInfoMode::Residual);
        plan.status = Status::Success;
        plan.weight_bits = bits;
        plan.native_weight_blocks = native_blocks;
        plan.cpu_direct_capable = true;
        plan.compact_capable = integer_domain;
        plan.scale_domain = integer_domain ?
            wroute::WeightScaleDomain::IntegerBlockTimesColumn :
            wroute::WeightScaleDomain::FloatingBlock;
        return route;
    };
    auto invalid = [&](const char *reason) {
        plan.status = Status::InvalidMetadata;
        plan.reject_reason = reason;
        return Route::Unsupported;
    };

    if (args.weight_format != Format::q8_h0 &&
        wroute::is_q8_channel_direct_read_args(args)) {
        if (!args.has_q8_channel_direct_read_contract())
            return invalid("invalid q8_channel contract");
        plan.status = Status::Success;
        plan.weight_bits = 8;
        plan.cpu_direct_capable = true;
        plan.compact_capable = true;
        plan.scale_domain = wroute::WeightScaleDomain::IntegerBlockTimesColumn;
        return Route::Q8ChannelDirect;
    }

    switch (args.weight_format) {
        case Format::q4_h0:
            return args.has_native_matched_width_contract() ?
                accept_family(Route::H0, 4, true) : invalid("invalid q4_h0 contract");
        case Format::q4_h1:
            return args.has_native_matched_width_contract() ?
                accept_family(Route::H1, 4, true) : invalid("invalid q4_h1 contract");
        case Format::q4_hp1:
            return args.has_native_matched_width_contract() ?
                accept_family(Route::HP1, 4, true) : invalid("invalid q4_hp1 contract");
        case Format::q8_h0:
            return accept_family(Route::H0, 8, args.B_blocks != nullptr);
        case Format::q8_h1:
            return args.has_q8_h1_im2p_contract() ?
                accept_family(Route::H1, 8, true) : invalid("invalid q8_h1 contract");
        case Format::q8_hp1:
            return wroute::has_q8_hp1_native_contract(args) ?
                accept_family(Route::HP1, 8, true) : invalid("invalid q8_hp1 contract");
        case Format::q16_h0:
            return args.has_native_matched_width_contract() ?
                accept_family(Route::H0, 16, true) : invalid("invalid q16_h0 contract");
        case Format::q16_h1:
            return args.has_native_matched_width_contract() ?
                accept_family(Route::H1, 16, true) : invalid("invalid q16_h1 contract");
        case Format::q16_hp1:
            return args.has_native_matched_width_contract() ?
                accept_family(Route::HP1, 16, true) : invalid("invalid q16_hp1 contract");
        case Format::q8_h2:
            if (mode == wroute::WeightScaleInfoMode::Residual) {
                plan.reject_reason = "H2 residual weights are unsupported";
                return Route::Unsupported;
            }
            if (!args.has_q8_h2_im2p_contract())
                return invalid("invalid q8_h2 contract");
            plan.status = Status::Success;
            plan.weight_bits = 8;
            plan.cpu_direct_capable = true;
            plan.scale_domain = wroute::WeightScaleDomain::FloatingBlock;
            return Route::Dense;
        case Format::q8_hp2:
            if (mode == wroute::WeightScaleInfoMode::Residual) {
                plan.reject_reason = "HP2 residual weights are unsupported";
                return Route::Unsupported;
            }
            if (!wroute::has_q8_hp2_native_contract(args))
                return invalid("invalid q8_hp2 contract");
            for (size_t i = 0; i < args.q8_hp2_block_count; ++i) {
                const block_q8_hp2 &block = args.q8_hp2_blocks[i];
                if (block.padding[0] != 0 || block.padding[1] != 0 ||
                    !finite_float_metadata(block.channel_scale)) {
                    return invalid("invalid q8_hp2 block metadata");
                }
            }
            plan.status = Status::Success;
            plan.weight_bits = 8;
            plan.cpu_direct_capable = true;
            plan.scale_domain = wroute::WeightScaleDomain::FloatingBlock;
            return Route::Dense;
        case Format::q8_channel:
            return invalid("invalid q8_channel contract");
        case Format::q8_channel_dense_sidecar:
            if (!args.has_q8_channel_dense_sidecar_contract())
                return invalid("invalid q8_channel_dense_sidecar contract");
            plan.status = Status::Success;
            plan.weight_bits = 8;
            plan.cpu_direct_capable = true;
            plan.compact_capable = true;
            plan.scale_domain = wroute::WeightScaleDomain::IntegerBlockTimesColumn;
            return Route::Q8ChannelSidecar;
        case Format::q8_0_unpacked_to_h1:
            if (wroute::is_q8_h1_weight_args(args))
                return accept_family(Route::H1, 8, wroute::is_q8_h1_args(args));
            plan.status = Status::Success;
            plan.weight_bits = 8;
            plan.cpu_direct_capable = true;
            return Route::Dense;
    }

    plan.reject_reason = "unsupported weight format";
    return Route::Unsupported;
}

wroute::WeightRouteStatus route_status_from_reader(wreader::WeightReaderStatus status)
{
    switch (status) {
        case wreader::WeightReaderStatus::Success:
            return wroute::WeightRouteStatus::Success;
        case wreader::WeightReaderStatus::InvalidArguments:
        case wreader::WeightReaderStatus::InvalidMetadata:
        case wreader::WeightReaderStatus::ScaleOverflow:
            return wroute::WeightRouteStatus::InvalidMetadata;
        case wreader::WeightReaderStatus::UnsupportedFormat:
            return wroute::WeightRouteStatus::UnsupportedFormat;
    }
    return wroute::WeightRouteStatus::UnsupportedFormat;
}

}

namespace quants::wroute {

WeightScaleInfo build_weight_scale_info(
    const ggml_gemmini_args_t &args,
    WeightScaleInfoMode mode)
{
    return build_weight_scale_info_impl(args, mode);
}

WeightRoutePlan resolve_weight_route_plan(
    const ggml_gemmini_args_t &args,
    WeightScaleInfoMode mode)
{
    WeightRoutePlan plan{};
    plan.route = classify_route(args, mode, plan);
    if (plan.route == WeightRouteKind::Unsupported)
        return plan;

    plan.layout = plan.route == WeightRouteKind::Dense ?
        (args.transpose_B ? WeightLayout::JxK_ColMajor : WeightLayout::KxJ_RowMajor) :
        WeightLayout::JxK_ColMajor;
    plan.weight_stride = plan.native_weight_blocks ||
        plan.route == WeightRouteKind::H0 ||
        plan.route == WeightRouteKind::H1 ||
        plan.route == WeightRouteKind::HP1 ? args.K :
        (args.sB ? args.sB : (plan.layout == WeightLayout::JxK_ColMajor ? args.K : args.J));
    plan.scales = build_weight_scale_info_impl(args, mode);
    if (!has_complete_scale_metadata(plan.scales, args)) {
        plan.status = WeightRouteStatus::InvalidMetadata;
        plan.reject_reason = "unsupported or incomplete weight scale metadata";
        return plan;
    }

    const bool block_scales = !plan.scales.scalar_mode &&
        !plan.scales.row_header_mode && !plan.scales.channel_mode;
    if ((plan.route == WeightRouteKind::H0 ||
         plan.route == WeightRouteKind::H1 ||
         plan.route == WeightRouteKind::HP1) && !block_scales) {
        plan.status = WeightRouteStatus::InvalidMetadata;
        plan.reject_reason = "hierarchical weight route requires block scales";
        return plan;
    }
    if (mode == WeightScaleInfoMode::Residual && block_scales) {
        const size_t required_scale_cols = args.K == 0 ? 0 :
            1 + (args.K - 1) / plan.scales.block_size;
        if (plan.scales.cols != required_scale_cols) {
            plan.status = WeightRouteStatus::InvalidMetadata;
            plan.reject_reason = "weight scale metadata must exactly cover K";
            return plan;
        }
    }

    if (plan.route == WeightRouteKind::H0 ||
        plan.route == WeightRouteKind::H1 ||
        plan.route == WeightRouteKind::HP1) {
        const wreader::WeightReaderStatus reader_status = wreader::validate(args, plan);
        if (reader_status != wreader::WeightReaderStatus::Success) {
            plan.status = route_status_from_reader(reader_status);
            plan.reject_reason = "invalid residual weight reader metadata";
            return plan;
        }
    } else {
        plan.scale_domain = block_scales ? WeightScaleDomain::FloatingBlock :
            WeightScaleDomain::IntegerBlockTimesColumn;
        plan.cpu_direct_capable = true;
        plan.compact_capable = plan.scale_domain == WeightScaleDomain::IntegerBlockTimesColumn;
    }

    plan.status = WeightRouteStatus::Success;
    plan.valid = true;
    return plan;
}

WeightRouteStatus weight_route_status(
    const WeightRoutePlan &plan,
    WeightExecutionPath path)
{
    if (!plan.valid)
        return plan.status;
    const bool capable = path == WeightExecutionPath::CpuDirect ?
        plan.cpu_direct_capable : plan.compact_capable;
    return capable ? WeightRouteStatus::Success : WeightRouteStatus::UnsupportedExecution;
}

const char *weight_route_status_name(WeightRouteStatus status)
{
    switch (status) {
        case WeightRouteStatus::Success:              return "success";
        case WeightRouteStatus::UnsupportedFormat:    return "unsupported-format";
        case WeightRouteStatus::InvalidMetadata:      return "invalid-metadata";
        case WeightRouteStatus::UnsupportedExecution: return "unsupported-execution";
    }
    return "unsupported-format";
}

const char *weight_route_kind_name(const WeightRoutePlan &plan)
{
    switch (plan.route) {
        case WeightRouteKind::Dense:
            return plan.scales.scalar_mode ? "tensor-scalar" : "dense-block";
        case WeightRouteKind::Q8ChannelDirect:  return "q8-channel-direct";
        case WeightRouteKind::Q8ChannelSidecar: return "q8-channel-sidecar";
        case WeightRouteKind::H0:               return "h0";
        case WeightRouteKind::H1:               return "h1";
        case WeightRouteKind::HP1:              return "hp1";
        case WeightRouteKind::Unsupported:      return "unsupported";
    }
    return "unsupported";
}

const char *weight_scale_mode_name(const WeightRoutePlan &plan)
{
    if (plan.scales.scalar_mode)
        return "scalar";
    if (plan.scales.row_header_mode)
        return "row-header";
    if (plan.scales.channel_mode)
        return "channel";
    if (plan.scale_domain == WeightScaleDomain::FloatingBlock)
        return "floating-block";
    if (plan.scale_domain == WeightScaleDomain::IntegerBlockTimesColumn)
        return "integer-block-times-column";
    return "none";
}

bool route_covers_k(const WeightRoutePlan &plan, size_t k_count)
{
    if (!plan.valid || plan.scales.scalar_mode ||
        plan.scales.row_header_mode || plan.scales.channel_mode)
        return plan.valid;
    return k_count == 0 ||
        1 + (k_count - 1) / plan.scales.block_size <= plan.scales.cols;
}

bool route_block_for_range(
    const WeightRoutePlan &plan,
    size_t k_offset,
    size_t block_k,
    size_t &block_index)
{
    block_index = 0;
    if (!plan.valid || block_k == 0)
        return false;
    if (plan.scales.scalar_mode || plan.scales.row_header_mode || plan.scales.channel_mode)
        return true;

    size_t k_end = 0;
    if (!checked_k_end(k_offset, block_k, k_end))
        return false;
    block_index = k_offset / plan.scales.block_size;
    return block_index == k_end / plan.scales.block_size &&
        block_index < plan.scales.cols;
}

float route_weight_scale(
    const WeightRoutePlan &plan,
    const ggml_gemmini_args_t &args,
    size_t j,
    size_t block_index)
{
    if (plan.scales.row_header_mode)
        return args.q8_channel_scale(j);
    if (plan.scales.scalar_mode)
        return plan.scales.scalar;
    if (plan.scales.channel_mode)
        return plan.scales.data[j];
    if (plan.route == WeightRouteKind::H0 ||
        plan.route == WeightRouteKind::H1 ||
        plan.route == WeightRouteKind::HP1) {
        const wreader::WeightScaleResult scale =
            wreader::read_scale(args, plan, j, block_index);
        if (!scale.ok())
            return std::numeric_limits<float>::quiet_NaN();
        if (scale.domain == WeightScaleDomain::FloatingBlock)
            return scale.floating_block_scale;
        return static_cast<float>(
            static_cast<double>(scale.integer_block_scale) *
            static_cast<double>(scale.column_scale));
    }
    return plan.scales.data[j * plan.scales.cols + block_index];
}

bool route_supports_integer_block_scale(const WeightRoutePlan &plan)
{
    return plan.valid &&
        plan.scale_domain == WeightScaleDomain::IntegerBlockTimesColumn;
}

uint64_t route_block_scale(
    const WeightRoutePlan &plan,
    const ggml_gemmini_args_t &args,
    size_t j,
    size_t block_index)
{
    if (!route_supports_integer_block_scale(plan))
        return 0;
    if (plan.scales.row_header_mode || plan.scales.scalar_mode || plan.scales.channel_mode)
        return 1;
    if (plan.route == WeightRouteKind::H1 || plan.route == WeightRouteKind::HP1) {
        const wreader::WeightScaleResult scale =
            wreader::read_scale(args, plan, j, block_index);
        return scale.ok() ? scale.integer_block_scale : 0;
    }
    return 0;
}

float route_column_scale(
    const WeightRoutePlan &plan,
    const ggml_gemmini_args_t &args,
    size_t j)
{
    if (plan.scales.row_header_mode)
        return args.q8_channel_scale(j);
    if (plan.scales.scalar_mode)
        return plan.scales.scalar;
    if (plan.scales.channel_mode)
        return plan.scales.data[j];
    if (plan.route == WeightRouteKind::H1 || plan.route == WeightRouteKind::HP1) {
        const wreader::WeightScaleResult scale = wreader::read_scale(args, plan, j, 0);
        return scale.ok() ? scale.column_scale : 0.0f;
    }
    return 0.0f;
}

}

void dequantize(
    const ggml_gemmini_args_t &args,
    size_t k_offset,
    size_t block_k,
    const int32_t *acc32,
    size_t acc_stride)
{
    if (!args.f_out || !acc32 || args.I == 0 || args.J == 0 ||
        block_k == 0 || acc_stride == 0)
        return;

    if (acc_stride < args.J)
        return;

    const quants::wroute::WeightRoutePlan plan =
        quants::wroute::resolve_weight_route_plan(
            args, quants::wroute::WeightScaleInfoMode::CommonOutput);
    if (!plan.valid)
        return;

    size_t first_block = 0;
    if (!quants::wroute::route_block_for_range(plan, k_offset, block_k, first_block))
        return;

    const size_t row_stride = args.stride_f_out ? args.stride_f_out : args.J;
    const size_t col_stride = args.col_stride_f_out ? args.col_stride_f_out : 1;
    const std::vector<float> activation_scales =
        quants::act::activation_scales(args, args.I);

    for (size_t i = 0; i < args.I; ++i) {
        const float activation_scale =
            i < activation_scales.size() ? activation_scales[i] : 1.0f;
        const int32_t *row_acc32 = acc32 + i * acc_stride;

        for (size_t j = 0; j < args.J; ++j) {
            size_t dst_offset = 0;
            if (!output_offset(i, j, row_stride, col_stride, dst_offset))
                return;

            const float weight_scale =
                quants::wroute::route_weight_scale(plan, args, j, first_block);
            const double scaled = static_cast<double>(row_acc32[j]) *
                                  static_cast<double>(weight_scale) *
                                  static_cast<double>(activation_scale);
            args.f_out[dst_offset] += static_cast<float>(scaled);
        }
    }
}

}
