#include "dec.hpp"
#include "dec_internal.hpp"
#include "dec_kernel.hpp"
#include "../../ggml-gemmini-args.h"
#include "../act/dispatch.hpp"
#include <gemmini/log.hpp>
#include <gemmini/cycle_reader.hpp>

#ifndef DEC_VALIDATION
#define DEC_VALIDATION 0
#endif

#if LOG_DEBUG
#include "../../ggml-gemmini-config.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#if LOG_DEBUG || DEC_VALIDATION
#include <cmath>
#endif

namespace ggml::gemmini::quants::dec
{
namespace
{

    struct ActivationDECScratch
    {
        std::vector<ResidualGroupEntry> residual_entries;
        std::vector<ActiveRowGroup> active_row_groups;
        std::vector<size_t> group_offsets;
        std::vector<size_t> group_row_group_indices;
        GroupKCSCPlan group_k_csc_plan;
        std::vector<uint32_t> unique_k;
        std::vector<size_t> active_rows_per_k;
        std::vector<uint32_t> active_groups_global;
        std::vector<float> Y_com;

        void resize_for_dims(size_t I, size_t J)
        {
            residual_entries.clear();
            active_row_groups.clear();
            group_offsets.clear();
            group_row_group_indices.clear();
            group_k_csc_plan.clear();
            unique_k.clear();
            active_rows_per_k.clear();
            active_groups_global.clear();
            if (Y_com.size() != I * J)
                Y_com.resize(I * J);
            std::fill(Y_com.begin(), Y_com.end(), 0.f);
        }
    };

    ActivationDECScratch &get_dec_scratch()
    {
        static thread_local ActivationDECScratch scratch;
        return scratch;
    }

    size_t saturating_add_size(size_t lhs, size_t rhs)
    {
        return rhs > std::numeric_limits<size_t>::max() - lhs ?
            std::numeric_limits<size_t>::max() : lhs + rhs;
    }

    size_t saturating_mul_size(size_t lhs, size_t rhs)
    {
        if (lhs == 0 || rhs == 0)
            return 0;
        return lhs > std::numeric_limits<size_t>::max() / rhs ?
            std::numeric_limits<size_t>::max() : lhs * rhs;
    }

    void log_dec_reject(const char *layer, const char *reason, const ggml_gemmini_args_t &args)
    {
#if LOG_DEBUG
        ggml::gemmini::log::debug(
            layer,
            "[dec.reject] reason=%s I=%zu K=%zu J=%zu format=%u",
            reason,
            args.I,
            args.K,
            args.J,
            static_cast<unsigned>(args.weight_format));
#else
        (void) layer;
        (void) reason;
        (void) args;
#endif
    }

#if LOG_DEBUG
    const char *requested_activation_name()
    {
        switch (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT)
        {
            case ggml::gemmini::config::ActivationQuantAlgo::EXSIA:  return "exsia";
            case ggml::gemmini::config::ActivationQuantAlgo::TENSOR: return "tensor";
            case ggml::gemmini::config::ActivationQuantAlgo::TOKEN:  return "token";
            case ggml::gemmini::config::ActivationQuantAlgo::STRIPE: return "stripe";
        }
        return "unsupported";
    }
#endif

#if DEC_VALIDATION
    bool checked_add_i64(int64_t lhs, int64_t rhs, int64_t &out)
    {
        if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
            (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
            return false;

        out = lhs + rhs;
        return true;
    }

    bool checked_mul_i64(int64_t lhs, int64_t rhs, int64_t &out)
    {
        if (lhs == 0 || rhs == 0)
        {
            out = 0;
            return true;
        }

        if ((lhs == -1 && rhs == std::numeric_limits<int64_t>::min()) ||
            (rhs == -1 && lhs == std::numeric_limits<int64_t>::min()))
            return false;

        if ((lhs > 0 && rhs > 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) ||
            (lhs > 0 && rhs < 0 && rhs < std::numeric_limits<int64_t>::min() / lhs) ||
            (lhs < 0 && rhs > 0 && lhs < std::numeric_limits<int64_t>::min() / rhs) ||
            (lhs < 0 && rhs < 0 && lhs < std::numeric_limits<int64_t>::max() / rhs))
            return false;

        out = lhs * rhs;
        return true;
    }

    bool reference_weight_code(
        size_t k,
        size_t j,
        const ggml_gemmini_args_t &args,
        const DecRoutePlan &plan,
        int8_t &code)
    {
        if (plan.route == DecWeightRoute::Q8HP1)
        {
            const block_q8_hp1 *block = args.q8_hp1_block(j, k / QK8_HP);
            if (!block)
                return false;
            code = block->qs[k % QK8_HP];
            return true;
        }
        if (plan.route == DecWeightRoute::Q8HP2)
        {
            const block_q8_hp2 *block = args.q8_hp2_block(j, k / QK8_HP);
            if (!block)
                return false;
            code = block->qs[k % QK8_HP];
            return true;
        }
        if (plan.route == DecWeightRoute::Q8H2)
        {
            const block_q8_h2 *block = args.q8_h2_block(j, k / QK8_H2);
            if (!block)
                return false;
            code = block->qs[k % QK8_H2];
            return true;
        }
        if (plan.route == DecWeightRoute::Q8H1 && plan.native_weight_blocks)
        {
            const block_q8_h1 *block = args.q8_h1_block(j, k / QK8_0);
            if (!block)
                return false;
            code = block->qs[k % QK8_0];
            return true;
        }

        const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
        if (!weights)
            return false;

        code = plan.layout == WeightLayout::KxJ_RowMajor ?
            weights[k * plan.weight_stride + j] : weights[j * plan.weight_stride + k];
        return true;
    }

    bool accumulate_scalar_reference(
        const ggml_gemmini_args_t &args,
        const DecRoutePlan &plan,
        const float *activation_scales,
        const std::vector<ResidualGroupEntry> &entries,
        float *reference_ycom)
    {
        const bool per_block_scale = !plan.scales.scalar_mode && !plan.scales.row_header_mode &&
            !plan.scales.channel_mode;
        const size_t route_count = per_block_scale ? plan.scales.cols : 1;
        std::vector<int64_t> accumulator(args.I * args.J, 0);

        for (size_t route = 0; route < route_count; ++route)
        {
            std::fill(accumulator.begin(), accumulator.end(), int64_t {0});
            for (const ResidualGroupEntry &entry : entries)
            {
                const size_t k_sz = entry.k;
                if (k_sz >= args.K ||
                    (per_block_scale && k_sz / plan.scales.block_size != route))
                    continue;

                for (size_t j = 0; j < args.J; ++j)
                {
                    int8_t weight_code = 0;
                    if (!reference_weight_code(k_sz, j, args, plan, weight_code))
                        return false;

                    if (entry.row >= args.I)
                        return false;
                    int64_t product = 0;
                    int64_t updated = 0;
                    const size_t accumulator_index = static_cast<size_t>(entry.row) * args.J + j;
                    if (!checked_mul_i64(entry.residual, weight_code, product) ||
                        !checked_add_i64(accumulator[accumulator_index], product, updated))
                        return false;
                    accumulator[accumulator_index] = updated;
                }
            }

            for (size_t r = 0; r < args.I; ++r)
            {
                const float activation_scale = activation_scales ? activation_scales[r] : 1.0f;
                for (size_t j = 0; j < args.J; ++j)
                {
                    const size_t index = r * args.J + j;
                    if (plan.route == DecWeightRoute::Q8H1)
                    {
                        const H1ScaleParams params = h1_scale_params(args, plan, j, route);
                        reference_ycom[index] += apply_h1_scale_ordered(
                            accumulator[index], params.c_eff, params.s_rf, activation_scale);
                    }
                    else
                    {
                        reference_ycom[index] += static_cast<float>(
                            static_cast<double>(accumulator[index]) * activation_scale *
                            dec_route_weight_scale(plan, args, j, route));
                    }
                }
            }
        }

        return true;
    }

    bool output_offset(const ggml_gemmini_args_t &args, size_t r, size_t j, size_t &offset)
    {
        const size_t row_stride = args.stride_f_out ? args.stride_f_out : args.J;
        const size_t col_stride = args.col_stride_f_out ? args.col_stride_f_out : 1;
        if ((r != 0 && row_stride > std::numeric_limits<size_t>::max() / r) ||
            (j != 0 && col_stride > std::numeric_limits<size_t>::max() / j))
            return false;

        const size_t row_offset = r * row_stride;
        const size_t col_offset = j * col_stride;
        if (row_offset > std::numeric_limits<size_t>::max() - col_offset)
            return false;

        offset = row_offset + col_offset;
        return true;
    }

    bool capture_output(const ggml_gemmini_args_t &args, std::vector<float> &output)
    {
        output.resize(args.I * args.J);
        for (size_t r = 0; r < args.I; ++r)
        {
            for (size_t j = 0; j < args.J; ++j)
            {
                size_t offset = 0;
                if (!output_offset(args, r, j, offset))
                    return false;
                output[r * args.J + j] = args.f_out[offset];
            }
        }
        return true;
    }
#endif

    size_t stage_from_outliers(
        const std::vector<QactOutlier> &outliers,
        size_t I,
        size_t K,
        ActivationDECScratch &scratch)
    {
        if (scratch.residual_entries.capacity() < outliers.size())
            scratch.residual_entries.reserve(outliers.size());

        if (outliers.empty() || I == 0 || K == 0)
            return 0;

        size_t staged = 0;
        for (const auto &outlier : outliers)
        {
            const int k = outlier.col;
            const int r_idx = outlier.row;
            if (k < 0 || r_idx < 0)
                continue;

            const size_t r = static_cast<size_t>(r_idx);
            const size_t k_sz = static_cast<size_t>(k);
            if (r >= I || k_sz >= K)
                continue;

            scratch.residual_entries.push_back({
                static_cast<uint32_t>(r), static_cast<uint32_t>(k_sz), outlier.residual});
            ++staged;
        }

#if LOG_DEBUG
        if (staged > 0)
        {
            double residual_sum = 0.0;
            double residual_sq_sum = 0.0;
            double residual_max = 0.0;
            const size_t start_idx = scratch.residual_entries.size() - staged;
            for (size_t i = start_idx; i < scratch.residual_entries.size(); ++i)
            {
                const double abs_res = std::fabs(static_cast<double>(scratch.residual_entries[i].residual));
                residual_sum += abs_res;
                residual_sq_sum += abs_res * abs_res;
                residual_max = std::max(residual_max, abs_res);
            }

            const double residual_mean = residual_sum / staged;
            const double residual_std = std::sqrt((residual_sq_sum / staged) - (residual_mean * residual_mean));
            ggml::gemmini::log::debug(
                nullptr,
                "[dec.select.outlier] total_outliers=%zu staged=%zu mean_residual=%.6g std_residual=%.6g max_residual=%.6g",
                outliers.size(),
                staged,
                residual_mean,
                residual_std,
                residual_max);
        }
#endif

        return staged;
    }

}

ActivationDECResult compensate_activation_dec(
    const std::vector<ggml::gemmini::quants::QactOutlier> &outliers,
    ggml_gemmini_args_t &args,
    const char *layer)
{
    uint64_t start = ggml::gemmini::cycle::read();

    ActivationDECResult result{};
    const DecRoutePlan plan = resolve_dec_route_plan(args, WeightScaleInfoMode::Dec);
    if (!plan.valid)
    {
        log_dec_reject(layer, plan.reject_reason, args);
        return result;
    }

    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if ((!weights && !plan.native_weight_blocks) || !args.f_out)
    {
        log_dec_reject(layer, "missing weight or output buffer", args);
        return result;
    }

    const size_t I = args.I;
    const size_t K = args.K;
    const size_t J = args.J;

    if (I == 0 || K == 0 || J == 0)
    {
        log_dec_reject(layer, "zero dimension", args);
        return result;
    }

    if (I > std::numeric_limits<uint32_t>::max() || K > std::numeric_limits<uint32_t>::max())
    {
        log_dec_reject(layer, "DEC residual dimensions exceed uint32", args);
        return result;
    }

    if (plan.weight_stride == 0)
    {
        log_dec_reject(layer, "zero weight stride", args);
        return result;
    }

    const size_t minimum_weight_stride = plan.layout == WeightLayout::KxJ_RowMajor ? J : K;
    if (weights && !plan.native_weight_blocks && plan.weight_stride < minimum_weight_stride)
    {
        log_dec_reject(layer, "weight stride is shorter than logical row", args);
        return result;
    }

    auto activation_scales_vec = act::activation_scales(args, I);
    const float *activation_scales = activation_scales_vec.data();
    if (!dec_route_covers_k(plan, K))
    {
        log_dec_reject(layer, "unsupported or incomplete weight scale metadata", args);
        return result;
    }

    ActivationDECScratch &scr = get_dec_scratch();

    uint64_t end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Ready compensation", start, end);

    start = ggml::gemmini::cycle::read();

    scr.resize_for_dims(I, J);

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Initialize scratch", start, end);

    start = ggml::gemmini::cycle::read();

    size_t total_staged = 0;
    if (!outliers.empty())
        total_staged += stage_from_outliers(outliers, I, K, scr);

    result.total_selected = total_staged;
    if (total_staged == 0)
        return result;

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Stage outlier residuals", start, end);

    start = ggml::gemmini::cycle::read();

    build_active_row_groups(scr.residual_entries, scr.active_row_groups);
    result.nnz = scr.residual_entries.size();
    scr.unique_k.reserve(result.nnz);
    for (const ResidualGroupEntry &entry : scr.residual_entries)
        scr.unique_k.push_back(entry.k);
    std::sort(scr.unique_k.begin(), scr.unique_k.end());
    scr.unique_k.erase(std::unique(scr.unique_k.begin(), scr.unique_k.end()), scr.unique_k.end());
    result.unique_k_count = scr.unique_k.size();

    scr.active_groups_global.reserve(scr.active_row_groups.size());
    for (const ActiveRowGroup &group : scr.active_row_groups)
        scr.active_groups_global.push_back(group.k_group);
    std::sort(scr.active_groups_global.begin(), scr.active_groups_global.end());
    scr.active_groups_global.erase(
        std::unique(scr.active_groups_global.begin(), scr.active_groups_global.end()),
        scr.active_groups_global.end());

    if (result.unique_k_count == 0)
        return result;

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Build active row-group plan", start, end);

    start = ggml::gemmini::cycle::read();

    const size_t group_count = K / kDecGroupSizeK + (K % kDecGroupSizeK != 0);
    build_group_major_index(
        scr.active_row_groups, group_count, scr.group_offsets, scr.group_row_group_indices);

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Build group-major index", start, end);

    // Test/replay opt-in; production remains row-direct.
    const char *force_group_k_csc = std::getenv("DEC_GROUP_K_CSC_FORCE");
    if (force_group_k_csc && force_group_k_csc[0] == '1' && force_group_k_csc[1] == '\0')
    {
        start = ggml::gemmini::cycle::read();

        if (!build_group_k_csc_plan(
                scr.residual_entries, scr.active_row_groups, scr.group_offsets,
                scr.group_row_group_indices, group_count, scr.group_k_csc_plan))
        {
            log_dec_reject(layer, "unable to represent group-K CSC plan", args);
            return ActivationDECResult{};
        }

        end = ggml::gemmini::cycle::read();
        ggml::gemmini::log::cycle(layer, "[dec] cpu.Build group-K CSC plan", start, end);
    }

    start = ggml::gemmini::cycle::read();

    const bool use_int64_scalar = plan.route == DecWeightRoute::Dense && plan.scales.scalar_mode;
    const bool use_int64_channel_direct = plan.route == DecWeightRoute::Q8ChannelDirect;
    const bool use_int64_channel_sidecar = plan.route == DecWeightRoute::Q8ChannelSidecar;
    const bool use_int64_h1 = plan.route == DecWeightRoute::Q8H1;
    const bool use_int64_block = !use_int64_h1 && !plan.scales.scalar_mode && !plan.scales.row_header_mode &&
        !plan.scales.channel_mode && (plan.route == DecWeightRoute::Dense ||
            plan.route == DecWeightRoute::Q8H1 || plan.route == DecWeightRoute::Q8H2 ||
            plan.route == DecWeightRoute::Q8HP1 || plan.route == DecWeightRoute::Q8HP2);
    const bool whole_k_grouped = use_int64_scalar || use_int64_channel_direct || use_int64_channel_sidecar;
    scr.active_rows_per_k.assign(result.unique_k_count, 0);
    size_t active_rows = 0;
    bool first_entry = true;
    uint32_t previous_row = 0;
    uint32_t previous_k = 0;
    for (const ResidualGroupEntry &entry : scr.residual_entries)
    {
        const bool new_row = first_entry || entry.row != previous_row;
        if (new_row)
            active_rows = saturating_add_size(active_rows, 1);
        if (new_row || entry.k != previous_k)
        {
            result.active_row_k_pairs = saturating_add_size(result.active_row_k_pairs, 1);
            const auto active_k = std::lower_bound(
                scr.unique_k.begin(), scr.unique_k.end(), entry.k);
            const size_t active_k_index = static_cast<size_t>(active_k - scr.unique_k.begin());
            const size_t row_count = saturating_add_size(scr.active_rows_per_k[active_k_index], 1);
            scr.active_rows_per_k[active_k_index] = row_count;
            result.rows_per_active_k_max = std::max(result.rows_per_active_k_max, row_count);
        }
        first_entry = false;
        previous_row = entry.row;
        previous_k = entry.k;
    }

    result.int_mac_count = saturating_mul_size(result.nnz, J);
    result.logical_weight_reference_count = result.int_mac_count;
    result.weight_scalar_load_count = result.int_mac_count;
    result.estimated_weight_bytes_read = saturating_mul_size(
        result.weight_scalar_load_count, sizeof(int8_t));

    const bool scaled_route = !plan.scales.scalar_mode && !plan.scales.row_header_mode && !plan.scales.channel_mode;
    const size_t scale_group_size = use_int64_h1 ? kDecGroupSizeK : plan.scales.block_size;
    size_t active_row_scale_groups = 0;
    if (scaled_route)
        for (const ActiveRowGroup &group : scr.active_row_groups)
        {
            size_t previous_scale_group = std::numeric_limits<size_t>::max();
            for (size_t position = group.entry_begin; position < group.entry_end; ++position)
            {
                const size_t scale_group = scr.residual_entries[position].k / scale_group_size;
                if (scale_group != previous_scale_group)
                {
                    active_row_scale_groups = saturating_add_size(active_row_scale_groups, 1);
                    previous_scale_group = scale_group;
                }
            }
        }

    const size_t y_com_update_groups = scaled_route ? active_row_scale_groups : active_rows;
    result.ycom_global_write_count = saturating_mul_size(y_com_update_groups, J);
    size_t plan_bytes = 0;
    plan_bytes = saturating_add_size(plan_bytes, saturating_mul_size(
        scr.residual_entries.size(), sizeof(ResidualGroupEntry)));
    plan_bytes = saturating_add_size(plan_bytes, saturating_mul_size(
        scr.active_row_groups.size(), sizeof(ActiveRowGroup)));
    plan_bytes = saturating_add_size(plan_bytes, saturating_mul_size(
        scr.group_offsets.size(), sizeof(size_t)));
    plan_bytes = saturating_add_size(plan_bytes, saturating_mul_size(
        scr.group_row_group_indices.size(), sizeof(size_t)));
    plan_bytes = saturating_add_size(plan_bytes, saturating_mul_size(
        scr.unique_k.size(), sizeof(uint32_t)));
    result.current_sparse_plan_bytes = saturating_add_size(plan_bytes, saturating_mul_size(
        scr.active_groups_global.size(), sizeof(uint32_t)));
    result.group_k_csc_plan_bytes = group_k_csc_plan_logical_bytes(scr.group_k_csc_plan);
    const size_t thread_accumulator_rows = whole_k_grouped ? I : 1;
    result.thread_scratch_bytes = saturating_mul_size(
        saturating_mul_size(thread_accumulator_rows, std::min(J, kDecInt64JTileWidth)),
        sizeof(int64_t));
#if LOG_DEBUG
    const size_t j_tiles = dec_int64_j_tile_count(J);
    const int dec_threads = resolve_dec_threads(j_tiles);
    const char *layout = plan.layout == WeightLayout::KxJ_RowMajor ? "kxj-row-major" : "jxk-col-major";
    const char *accessor = use_int64_channel_direct ? "q8-channel-row" :
        (use_int64_channel_sidecar ? "q8-channel-sidecar" :
            (plan.native_weight_blocks ? dec_route_name(plan) : "dense-int8"));
    const char *reducer = use_int64_scalar ? "tensor" :
        (use_int64_channel_direct || use_int64_channel_sidecar ? "channel" :
            (use_int64_h1 ? "h1" :
                (plan.route == DecWeightRoute::Q8H2 ? "h2" :
                (plan.route == DecWeightRoute::Q8HP1 ? "hp1" :
                    (plan.route == DecWeightRoute::Q8HP2 ? "hp2" : "block")))));
    ggml::gemmini::log::debug(
        layer,
        "[dec.route] algorithm=grouped sparse_kernel=row-direct traversal=row-major j_inner_nr=1 row_panel=full weight_vector=direct accumulator=int64 output_panel=off group_size_k=%zu activation=%s weight=%s accessor=%s reducer=%s j_partition=contiguous-range microtile=%zu threads=%d residual_format=common weight_layout=%s scale_mode=%s I=%zu J=%zu K=%zu",
        kDecGroupSizeK,
        requested_activation_name(),
        dec_route_name(plan),
        accessor,
        reducer,
        kDecInt64JTileWidth,
        dec_threads,
        layout,
        dec_scale_mode_name(plan),
        I,
        J,
        K);
    const size_t route_accumulate_count = saturating_mul_size(
        scaled_route ? active_row_scale_groups : scr.active_row_groups.size(), J);
    const size_t scale_apply_count = saturating_mul_size(
        scaled_route ? active_row_scale_groups : active_rows, J);
    const size_t scale_eval_count = scaled_route && scale_group_size == kDecGroupSizeK ?
        saturating_mul_size(scr.active_groups_global.size(), J) : scale_apply_count;
    size_t row_groups_per_group_max = 0;
    for (size_t group = 0; group < group_count; ++group)
        row_groups_per_group_max = std::max(
            row_groups_per_group_max, scr.group_offsets[group + 1] - scr.group_offsets[group]);
    const double row_groups_per_group_mean = scr.active_groups_global.empty() ? 0.0 :
        static_cast<double>(scr.active_row_groups.size()) / scr.active_groups_global.size();
    const double rows_per_active_k_mean = result.unique_k_count == 0 ? 0.0 :
        static_cast<double>(result.active_row_k_pairs) / result.unique_k_count;
    const double weight_reuse_ratio = result.weight_scalar_load_count == 0 ? 0.0 :
        static_cast<double>(result.logical_weight_reference_count) / result.weight_scalar_load_count;
    ggml::gemmini::log::debug(
        layer,
        "[dec.work] nnz=%zu active_rows=%zu active_groups_global=%zu active_row_groups=%zu row_groups_per_group_mean=%.6g row_groups_per_group_max=%zu int_mac_count=%zu logical_weight_reference_count=%zu weight_scalar_load_count=%zu weight_vector_load_count=%zu estimated_weight_bytes_read=%zu weight_reuse_ratio=%.6g active_row_k_pairs=%zu rows_per_active_k_mean=%.6g rows_per_active_k_max=%zu ycom_global_write_count=%zu current_sparse_plan_bytes=%zu group_k_csc_plan_bytes=%zu thread_scratch_bytes=%zu route_accumulate_count=%zu scale_apply_count=%zu scale_eval_count=%zu active_k=%zu j_tiles=%zu threads=%d",
        result.nnz,
        active_rows,
        scr.active_groups_global.size(),
        scr.active_row_groups.size(),
        row_groups_per_group_mean,
        row_groups_per_group_max,
        result.int_mac_count,
        result.logical_weight_reference_count,
        result.weight_scalar_load_count,
        result.weight_vector_load_count,
        result.estimated_weight_bytes_read,
        weight_reuse_ratio,
        result.active_row_k_pairs,
        rows_per_active_k_mean,
        result.rows_per_active_k_max,
        result.ycom_global_write_count,
        result.current_sparse_plan_bytes,
        result.group_k_csc_plan_bytes,
        result.thread_scratch_bytes,
        route_accumulate_count,
        scale_apply_count,
        scale_eval_count,
        result.unique_k_count,
        j_tiles,
        dec_threads);
#endif

    if (use_int64_scalar)
        accumulate_to_ycom_int64_scalar(
            args, plan, I, J, activation_scales, scr.residual_entries, scr.active_row_groups,
            scr.group_offsets, scr.group_row_group_indices,
            scr.Y_com.data());
    else if (use_int64_channel_direct)
        accumulate_to_ycom_int64_channel_direct(
            args, plan, I, J, activation_scales, scr.residual_entries, scr.active_row_groups,
            scr.group_offsets, scr.group_row_group_indices,
            scr.Y_com.data());
    else if (use_int64_channel_sidecar)
        accumulate_to_ycom_int64_channel_sidecar(
            args, plan, I, J, activation_scales, scr.residual_entries, scr.active_row_groups,
            scr.group_offsets, scr.group_row_group_indices,
            scr.Y_com.data());
    else if (use_int64_h1)
        accumulate_to_ycom_int64_h1(
            args, plan, I, J, activation_scales, scr.residual_entries, scr.active_row_groups,
            scr.group_offsets, scr.group_row_group_indices,
            scr.Y_com.data());
    else if (use_int64_block)
        accumulate_to_ycom_int64_block(
            args, plan, I, J, activation_scales, scr.residual_entries, scr.active_row_groups,
            scr.group_offsets, scr.group_row_group_indices,
            scr.Y_com.data());
    else
    {
        log_dec_reject(layer, "valid route has no integer kernel", args);
        return result;
    }

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Compute and accumulate compensation", start, end);

#if DEC_VALIDATION
    std::vector<float> validation_output_before;
    std::vector<float> reference_ycom(I * J, 0.0f);
    const bool validation_ready = capture_output(args, validation_output_before) &&
        accumulate_scalar_reference(
            args,
            plan,
            activation_scales,
            scr.residual_entries,
            reference_ycom.data());
#endif

    start = ggml::gemmini::cycle::read();
    apply_ycom_to_output(scr.Y_com.data(), I, J, args);
    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Apply Y_com to output", start, end);

#if DEC_VALIDATION
    if (!validation_ready)
    {
        ggml::gemmini::log::debug(layer, "[dec.validation] skipped: reference setup failed");
    }
    else
    {
        float max_abs_error = 0.0f;
        float max_relative_error = 0.0f;
        bool finite = true;
        for (size_t r = 0; r < I; ++r)
        {
            for (size_t j = 0; j < J; ++j)
            {
                size_t offset = 0;
                if (!output_offset(args, r, j, offset))
                {
                    finite = false;
                    continue;
                }

                const size_t index = r * J + j;
                const float expected = validation_output_before[index] + reference_ycom[index];
                const float observed = args.f_out[offset];
                if (!std::isfinite(expected) || !std::isfinite(observed))
                {
                    finite = false;
                    continue;
                }

                const float absolute_error = std::fabs(observed - expected);
                const float relative_error = absolute_error /
                    std::max(std::fabs(expected), std::numeric_limits<float>::min());
                max_abs_error = std::max(max_abs_error, absolute_error);
                max_relative_error = std::max(max_relative_error, relative_error);
            }
        }
        ggml::gemmini::log::debug(
            layer,
            "[dec.validation] finite=%d max_abs_error=%.9g max_relative_error=%.9g",
            finite ? 1 : 0,
            max_abs_error,
            max_relative_error);
    }
#endif

#if LOG_DEBUG
    {
        ggml::gemmini::log::debug(
            layer,
            "[dec] I=%zu K=%zu J=%zu staged=%zu nnz=%zu unique_k=%zu",
            I,
            K,
            J,
            result.total_selected,
            result.nnz,
            result.unique_k_count);

        if (result.nnz > 0)
        {
            double total_compensation = 0.0;
            for (const ResidualGroupEntry &entry : scr.residual_entries)
                total_compensation += std::fabs(static_cast<double>(entry.residual));

            const double avg_compensation_per_entry = total_compensation / result.nnz;
            const double sparsity = 100.0 * result.nnz / (I * K);
            const double comp_density = total_compensation / (I * J);

            ggml::gemmini::log::debug(
                layer,
                "[dec.summary] total_comp=%.6g avg_comp_per_entry=%.6g sparsity=%.6f%% comp_density=%.6g",
                total_compensation,
                avg_compensation_per_entry,
                sparsity,
                comp_density);
        }
    }
#endif
    return result;
}
} // namespace ggml::gemmini::quants::dec
