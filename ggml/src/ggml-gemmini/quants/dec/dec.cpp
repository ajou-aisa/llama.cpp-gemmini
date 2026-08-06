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
#include <memory>
#include <numeric>
#include <new>
#include <vector>

#if LOG_DEBUG || DEC_VALIDATION
#include <cmath>
#endif

namespace ggml::gemmini::quants::dec
{
namespace
{
    constexpr bool kH1SmallGroupFastpathEnabled =
        GGML_GEMMINI_DEC_H1_SMALL_GROUP_FASTPATH != 0;

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

    bool checked_mul_size(size_t lhs, size_t rhs, size_t &out)
    {
        if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs)
            return false;
        out = lhs * rhs;
        return true;
    }

    bool checked_add_size(size_t lhs, size_t rhs, size_t &out)
    {
        if (lhs > std::numeric_limits<size_t>::max() - rhs)
            return false;
        out = lhs + rhs;
        return true;
    }

    PreparedDecSelectedRoute select_prepared_dec_route(
        bool use_group_k_csc,
        bool use_int64_h1,
        bool native_weight_blocks,
        size_t nnz)
    {
        (void) native_weight_blocks;
        if (use_group_k_csc)
            return PreparedDecSelectedRoute::group_k_csc;
        if (kH1SmallGroupFastpathEnabled && use_int64_h1)
        {
            if (nnz == 1)
                return PreparedDecSelectedRoute::h1_small_group_single;
            if (nnz >= 2 && nnz <= 4)
                return PreparedDecSelectedRoute::h1_small_group_2_to_4;
        }
        return PreparedDecSelectedRoute::row_direct;
    }

    size_t count_active_rows(const std::vector<ActiveRowGroup> &groups)
    {
        size_t active_rows = 0;
        uint32_t previous_row = std::numeric_limits<uint32_t>::max();
        for (const ActiveRowGroup &group : groups)
        {
            if (group.row == previous_row)
                continue;
            previous_row = group.row;
            ++active_rows;
        }
        return active_rows;
    }

    size_t count_active_row_scale_groups(
        const std::vector<ResidualGroupEntry> &entries,
        const std::vector<ActiveRowGroup> &groups,
        size_t scale_group_size)
    {
        if (scale_group_size == 0)
            return 0;

        size_t active_row_scale_groups = 0;
        for (const ActiveRowGroup &group : groups)
        {
            size_t previous_scale_group = std::numeric_limits<size_t>::max();
            for (size_t position = group.entry_begin; position < group.entry_end; ++position)
            {
                const size_t scale_group = entries[position].k / scale_group_size;
                if (scale_group == previous_scale_group)
                    continue;
                ++active_row_scale_groups;
                previous_scale_group = scale_group;
            }
        }
        return active_row_scale_groups;
    }

    size_t estimate_group_k_csc_vector_load_count(size_t nr, size_t active_k_count, size_t J)
    {
        if (nr <= 1)
            return 0;

        size_t vector_loads_per_k = 0;
        for (size_t jb = 0; jb < J; jb += kDecInt64JTileWidth)
        {
            const size_t width = std::min(kDecInt64JTileWidth, J - jb);
            vector_loads_per_k = saturating_add_size(
                vector_loads_per_k, width / nr + (width % nr != 0));
        }
        return saturating_mul_size(active_k_count, vector_loads_per_k);
    }

    PreparedDecWorkloadHistogram make_prepared_workload_histogram(
        const std::vector<ResidualGroupEntry> &entries,
        const std::vector<ActiveRowGroup> &active_row_groups,
        const DecRoutePlan &route,
        size_t I,
        size_t J,
        size_t K,
        size_t unique_k_count,
        size_t active_row_k_pairs,
        size_t rows_per_active_k_max,
        bool use_group_k_csc,
        bool use_int64_h1,
        size_t nr)
    {
        PreparedDecWorkloadHistogram histogram{};
        histogram.residual_nnz = entries.size();
        histogram.residual_density =
            I == 0 || K == 0 ? 0.0 :
            static_cast<double>(entries.size()) / static_cast<double>(I * K);
        histogram.active_rows = count_active_rows(active_row_groups);
        histogram.active_row_groups = active_row_groups.size();
        histogram.active_k = unique_k_count;
        for (const ActiveRowGroup &group : active_row_groups)
        {
            const size_t group_nnz = group.entry_end - group.entry_begin;
            if (group_nnz == 1)
                ++histogram.bin_1;
            else if (group_nnz <= 4)
                ++histogram.bin_2_to_4;
            else if (group_nnz <= 8)
                ++histogram.bin_5_to_8;
            else
                ++histogram.bin_over_8;
        }
        histogram.rows_per_active_k_mean = unique_k_count == 0 ? 0.0 :
            static_cast<double>(active_row_k_pairs) / static_cast<double>(unique_k_count);
        histogram.rows_per_active_k_max = rows_per_active_k_max;
        histogram.estimated_int_mac_count = saturating_mul_size(entries.size(), J);
        histogram.selected_route = select_prepared_dec_route(
            use_group_k_csc, use_int64_h1, route.native_weight_blocks, entries.size());

        const bool scaled_route = !route.scales.scalar_mode && !route.scales.row_header_mode &&
            !route.scales.channel_mode;
        const size_t active_rows = histogram.active_rows;
        const size_t scale_group_size = use_int64_h1 ? kDecGroupSizeK : route.scales.block_size;
        const size_t active_row_scale_groups = scaled_route ?
            count_active_row_scale_groups(entries, active_row_groups, scale_group_size) : 0;
        histogram.ycom_write_count = saturating_mul_size(
            scaled_route ? active_row_scale_groups : active_rows, J);
        histogram.weight_scalar_load_count = use_group_k_csc ?
            saturating_mul_size(unique_k_count, J) : histogram.estimated_int_mac_count;
        histogram.weight_vector_load_count = use_group_k_csc ?
            estimate_group_k_csc_vector_load_count(nr, unique_k_count, J) : 0;
        return histogram;
    }

    size_t estimate_group_k_csc_plan_bytes(size_t group_count, size_t entry_count, size_t active_row_group_count)
    {
        size_t bytes = 0;
        const size_t group_column_offset_count =
            saturating_mul_size(group_count, kDecGroupSizeK + 1);
        bytes = saturating_add_size(
            bytes, saturating_mul_size(group_column_offset_count, sizeof(uint32_t)));
        bytes = saturating_add_size(bytes, saturating_mul_size(entry_count, sizeof(uint32_t)));
        bytes = saturating_add_size(
            bytes, saturating_mul_size(saturating_add_size(group_count, 1), sizeof(uint32_t)));
        bytes = saturating_add_size(
            bytes, saturating_mul_size(active_row_group_count, sizeof(uint32_t)));
        return bytes;
    }

    size_t estimate_group_k_csc_saved_weight_bytes(
        size_t logical_weight_reference_count, size_t unique_k_count, size_t J)
    {
        const size_t grouped_weight_bytes = saturating_mul_size(unique_k_count, J);
        return logical_weight_reference_count > grouped_weight_bytes ?
            logical_weight_reference_count - grouped_weight_bytes : 0;
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

struct PreparedDecSlice::Data
{
    ggml_gemmini_args_t args;
    DecRoutePlan route;
    std::vector<float> activation_scales;
    std::vector<ResidualGroupEntry> residual_entries;
    std::vector<ActiveRowGroup> active_row_groups;
    std::vector<size_t> group_offsets;
    std::vector<size_t> group_row_group_indices;
    GroupKCSCPlan group_k_csc_plan;
    std::vector<DecColumnRange> ranges;
    ActivationDECResult result{};
    PreparedDecWorkloadHistogram histogram{};
    DecPreparationCounters counters{};
    size_t nr = 1;
    bool use_group_k_csc = false;
    bool use_int64_scalar = false;
    bool use_int64_channel_direct = false;
    bool use_int64_channel_sidecar = false;
    bool use_int64_h1 = false;
    bool use_int64_block = false;
};

PreparedDecSlice::PreparedDecSlice(std::shared_ptr<const Data> data) : data_(std::move(data)) {}

size_t PreparedDecSlice::shard_count() const
{
    return data_ ? data_->ranges.size() : 0;
}

DecColumnRange PreparedDecSlice::shard_range(size_t shard_index) const
{
    return data_ && shard_index < data_->ranges.size() ? data_->ranges[shard_index] : DecColumnRange{};
}

size_t PreparedDecSlice::nr() const
{
    return data_ ? data_->nr : 1;
}

bool PreparedDecSlice::uses_group_k_csc() const
{
    return data_ && data_->use_group_k_csc;
}

const ActivationDECResult & PreparedDecSlice::result() const
{
    static const ActivationDECResult empty;
    return data_ ? data_->result : empty;
}

const PreparedDecWorkloadHistogram & PreparedDecSlice::workload_histogram() const
{
    static const PreparedDecWorkloadHistogram empty;
    return data_ ? data_->histogram : empty;
}

DecPreparationCounters PreparedDecSlice::preparation_counters() const
{
    return data_ ? data_->counters : DecPreparationCounters{};
}

bool h1_small_group_fastpath_enabled()
{
    return kH1SmallGroupFastpathEnabled;
}

namespace
{
DecStatus slice_dec_columns(
    const ggml_gemmini_args_t &source, size_t col_begin, size_t col_end,
    ggml_gemmini_args_t &local_args)
{
    if (col_begin >= col_end || col_end > source.J || source.f_out == nullptr)
        return DecStatus::invalid_arguments;

    const size_t output_col_stride = source.col_stride_f_out ? source.col_stride_f_out : 1;
    size_t offset = 0;
    if (!checked_mul_size(col_begin, output_col_stride, offset))
        return DecStatus::invalid_arguments;

    const size_t width = col_end - col_begin;
    local_args = source;
    local_args.J = width;
    if (local_args.stride_f_out == 0)
        local_args.stride_f_out = source.J;
    local_args.f_out += offset;

    const bool channel_direct = source.weight_format ==
        ggml_gemmini_args_t::im2p_weight_format_t::q8_channel;
    const bool jxk_layout = channel_direct ||
        source.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar ||
        source.transpose_B || source.q8_h1_blocks != nullptr || source.q8_h2_blocks != nullptr ||
        source.q8_hp1_blocks != nullptr || source.q8_hp2_blocks != nullptr || source.c_b != nullptr;

    if (source.B != nullptr) {
        const size_t weight_stride = channel_direct ? source.q8_channel_row_stride :
            (jxk_layout ? (source.sB ? source.sB : source.K) : 1);
        if (!checked_mul_size(col_begin, weight_stride, offset))
            return DecStatus::invalid_arguments;
        if (channel_direct)
            local_args.B = reinterpret_cast<elem_t *>(reinterpret_cast<uint8_t *>(source.B) +
                offset);
        else if (jxk_layout)
            local_args.B = reinterpret_cast<elem_t *>(reinterpret_cast<int8_t *>(source.B) +
                offset);
        else
            local_args.B = reinterpret_cast<elem_t *>(reinterpret_cast<int8_t *>(source.B) + col_begin);
    }
    if (source.B_blocks != nullptr && source.blocks_K != 0) {
        if (!checked_mul_size(col_begin, source.blocks_K, offset))
            return DecStatus::invalid_arguments;
        local_args.B_blocks = source.B_blocks + offset;
    }
    if (source.B_scales != nullptr && source.blocks_K != 0) {
        if (!checked_mul_size(col_begin, source.blocks_K, offset))
            return DecStatus::invalid_arguments;
        local_args.B_scales = source.B_scales + offset;
    }
    if (source.weight_channel_scales != nullptr)
        local_args.weight_channel_scales = source.weight_channel_scales + col_begin;
    local_args.weight_channel_scale_count = width;
    if (source.q8_channel_row_base != nullptr) {
        if (!checked_mul_size(col_begin, source.q8_channel_row_stride, offset))
            return DecStatus::invalid_arguments;
        local_args.q8_channel_row_base = source.q8_channel_row_base +
            offset;
        local_args.q8_channel_row_count = width;
    }
    if (source.q8_h1_blocks != nullptr) {
        if (!checked_mul_size(col_begin, source.blocks_per_row, offset) ||
            !checked_mul_size(width, source.blocks_per_row, local_args.q8_h1_block_count))
            return DecStatus::invalid_arguments;
        local_args.q8_h1_blocks = source.q8_h1_blocks + offset;
        local_args.q8_h1_rows = width;
    }
    if (source.q8_h2_blocks != nullptr) {
        if (!checked_mul_size(col_begin, source.q8_h2_blocks_per_row, offset) ||
            !checked_mul_size(width, source.q8_h2_blocks_per_row, local_args.q8_h2_block_count))
            return DecStatus::invalid_arguments;
        local_args.q8_h2_blocks = source.q8_h2_blocks + offset;
    }
    if (source.q8_hp1_blocks != nullptr) {
        if (!checked_mul_size(col_begin, source.q8_hp1_blocks_per_row, offset) ||
            !checked_mul_size(width, source.q8_hp1_blocks_per_row, local_args.q8_hp1_block_count))
            return DecStatus::invalid_arguments;
        local_args.q8_hp1_blocks = source.q8_hp1_blocks + offset;
    }
    if (source.q8_hp2_blocks != nullptr) {
        if (!checked_mul_size(col_begin, source.q8_hp2_blocks_per_row, offset) ||
            !checked_mul_size(width, source.q8_hp2_blocks_per_row, local_args.q8_hp2_block_count))
            return DecStatus::invalid_arguments;
        local_args.q8_hp2_blocks = source.q8_hp2_blocks + offset;
    }
    if (source.c_b != nullptr && source.blocks_per_row != 0) {
        if (!checked_mul_size(col_begin, source.blocks_per_row, offset))
            return DecStatus::invalid_arguments;
        local_args.c_b = source.c_b + offset;
    }
    if (source.s_rf != nullptr)
        local_args.s_rf = source.s_rf + col_begin;
    if (source.R != nullptr)
        local_args.R = source.R + col_begin;
    if (source.stripe_J > 1) {
        if (source.s_rf_stripe == nullptr || source.R_stripe == nullptr ||
            col_begin % source.stripe_J != 0)
            return DecStatus::unsupported;
        local_args.s_rf_stripe = source.s_rf_stripe + col_begin / source.stripe_J;
        local_args.R_stripe = source.R_stripe + col_begin / source.stripe_J;
    }
    return DecStatus::success;
}

class ExternalDecShardGuard
{
public:
    ExternalDecShardGuard() { enter_external_dec_shard(); }
    ~ExternalDecShardGuard() { leave_external_dec_shard(); }
};
}

DecStatus prepare_activation_dec_slice(
    const std::vector<QactOutlier> &outliers,
    const ggml_gemmini_args_t &args,
    size_t row_begin,
    size_t row_end,
    size_t requested_shards,
    DispatchOverride dispatch_override,
    std::shared_ptr<const PreparedDecSlice> &prepared)
{
    prepared.reset();
    if (requested_shards == 0 || row_begin >= row_end || row_end > args.I ||
        args.K == 0 || args.J == 0 || args.f_out == nullptr ||
        row_end - row_begin > std::numeric_limits<uint32_t>::max() ||
        args.K > std::numeric_limits<uint32_t>::max())
        return DecStatus::invalid_arguments;

    try {
        auto data = std::make_shared<PreparedDecSlice::Data>();
        data->args = args;
        const size_t input_stride = args.sA ? args.sA : args.K;
        const size_t output_stride = args.stride_f_out ? args.stride_f_out : args.J;
        if (row_begin > std::numeric_limits<size_t>::max() / std::max(input_stride, output_stride))
            return DecStatus::invalid_arguments;
        if (row_begin > std::numeric_limits<size_t>::max() - data->args.activation_row_offset)
            return DecStatus::invalid_arguments;
        data->args.I = row_end - row_begin;
        if (data->args.A != nullptr)
            data->args.A += row_begin * input_stride;
        data->args.f_out += row_begin * output_stride;
        data->args.activation_row_offset += row_begin;

        data->route = resolve_dec_route_plan(data->args, WeightScaleInfoMode::Dec);
        data->counters.route_plan_build_count = 1;
        if (!data->route.valid)
            return DecStatus::unsupported;
        const int8_t *weights = reinterpret_cast<const int8_t *>(data->args.B);
        if ((!weights && !data->route.native_weight_blocks) || data->route.weight_stride == 0 ||
            !dec_route_covers_k(data->route, data->args.K))
            return DecStatus::unsupported;
        const size_t minimum_weight_stride = data->route.layout == WeightLayout::KxJ_RowMajor ?
            data->args.J : data->args.K;
        if (weights && !data->route.native_weight_blocks &&
            data->route.weight_stride < minimum_weight_stride)
            return DecStatus::invalid_arguments;

        std::vector<float> all_activation_scales = act::activation_scales(args, args.I);
        if (!all_activation_scales.empty())
            data->activation_scales.assign(
                all_activation_scales.begin() + row_begin, all_activation_scales.begin() + row_end);
        data->residual_entries.reserve(outliers.size());
        for (const QactOutlier &outlier : outliers) {
            if (outlier.row < 0 || outlier.col < 0)
                continue;
            const size_t row = static_cast<size_t>(outlier.row);
            const size_t k = static_cast<size_t>(outlier.col);
            if (row >= row_begin && row < row_end && k < data->args.K)
                data->residual_entries.push_back({ static_cast<uint32_t>(row - row_begin),
                                                   static_cast<uint32_t>(k), outlier.residual });
        }
        data->result.total_selected = data->residual_entries.size();
        build_active_row_groups(data->residual_entries, data->active_row_groups);
        data->counters.active_row_plan_build_count = 1;
        data->result.nnz = data->residual_entries.size();
        std::vector<uint32_t> unique_k;
        unique_k.reserve(data->residual_entries.size());
        for (const ResidualGroupEntry &entry : data->residual_entries)
            unique_k.push_back(entry.k);
        std::sort(unique_k.begin(), unique_k.end());
        unique_k.erase(std::unique(unique_k.begin(), unique_k.end()), unique_k.end());
        data->result.unique_k_count = unique_k.size();

        const size_t group_count = data->args.K / kDecGroupSizeK +
            (data->args.K % kDecGroupSizeK != 0);
        build_group_major_index(data->active_row_groups, group_count,
            data->group_offsets, data->group_row_group_indices);
        data->use_int64_scalar = data->route.route == DecWeightRoute::Dense &&
            data->route.scales.scalar_mode;
        data->use_int64_channel_direct = data->route.route == DecWeightRoute::Q8ChannelDirect;
        data->use_int64_channel_sidecar = data->route.route == DecWeightRoute::Q8ChannelSidecar;
        data->use_int64_h1 = data->route.route == DecWeightRoute::Q8H1;
        data->use_int64_block = !data->use_int64_h1 && !data->route.scales.scalar_mode &&
            !data->route.scales.row_header_mode && !data->route.scales.channel_mode &&
            (data->route.route == DecWeightRoute::Dense || data->route.route == DecWeightRoute::Q8H2 ||
             data->route.route == DecWeightRoute::Q8HP1 || data->route.route == DecWeightRoute::Q8HP2);
        if (!data->use_int64_scalar && !data->use_int64_channel_direct &&
            !data->use_int64_channel_sidecar && !data->use_int64_h1 && !data->use_int64_block)
            return DecStatus::unsupported;

        const bool group_supported = data->use_int64_scalar || data->use_int64_h1;
        size_t active_row_k_pairs = 0;
        std::vector<size_t> rows_per_k(unique_k.size(), 0);
        bool first = true;
        uint32_t previous_row = 0;
        uint32_t previous_k = 0;
        for (const ResidualGroupEntry &entry : data->residual_entries) {
            if (first || entry.row != previous_row || entry.k != previous_k)
            {
                ++active_row_k_pairs;
                const auto it = std::lower_bound(unique_k.begin(), unique_k.end(), entry.k);
                if (it != unique_k.end() && *it == entry.k)
                    ++rows_per_k[static_cast<size_t>(it - unique_k.begin())];
            }
            first = false;
            previous_row = entry.row;
            previous_k = entry.k;
        }
        data->result.active_row_k_pairs = active_row_k_pairs;
        for (size_t row_count : rows_per_k)
            data->result.rows_per_active_k_max =
                std::max(data->result.rows_per_active_k_max, row_count);
        const double rows_per_active_k_mean = unique_k.empty() ? 0.0 :
            static_cast<double>(active_row_k_pairs) / unique_k.size();
        const size_t estimated_plan_bytes = estimate_group_k_csc_plan_bytes(
            group_count, data->residual_entries.size(), data->active_row_groups.size());
        const size_t saved_weight_bytes = estimate_group_k_csc_saved_weight_bytes(
            data->result.logical_weight_reference_count, unique_k.size(), data->args.J);
        const bool common = (data->use_int64_scalar || data->use_int64_h1) &&
            data->args.I > 1 && data->args.J >= 8 && rows_per_active_k_mean >= 16.0 &&
            saved_weight_bytes > estimated_plan_bytes;
        const char *disable_group_k_csc = std::getenv("DEC_GROUP_K_CSC_DISABLE");
        const bool disabled = dispatch_override == DispatchOverride::row_direct ||
            (disable_group_k_csc && disable_group_k_csc[0] == '1' &&
             disable_group_k_csc[1] == '\0');
        const char *enable_group_k_csc = std::getenv("DEC_GROUP_K_CSC_ENABLE");
        const char *force_group_k_csc = std::getenv("DEC_GROUP_K_CSC_FORCE");
        const bool overridden = data->args.I > 1 &&
            (dispatch_override == DispatchOverride::group_k_csc ||
             (enable_group_k_csc && enable_group_k_csc[0] == '1' &&
              enable_group_k_csc[1] == '\0') ||
             (force_group_k_csc && force_group_k_csc[0] == '1' &&
              force_group_k_csc[1] == '\0'));
        data->use_group_k_csc = !data->residual_entries.empty() && !disabled &&
            group_supported && (overridden || common);
        data->nr = data->use_group_k_csc ? (data->args.J < 8 ? 4 : 8) : 1;
        if (data->use_group_k_csc) {
            if (!build_group_k_csc_plan(data->residual_entries, data->active_row_groups,
                    data->group_offsets, data->group_row_group_indices, group_count,
                    data->group_k_csc_plan))
                return DecStatus::unsupported;
            data->counters.group_k_csc_plan_build_count = 1;
            data->result.group_k_csc_plan_bytes = group_k_csc_plan_logical_bytes(
                data->group_k_csc_plan);
        }

        const size_t native_scale_group = data->use_int64_h1 && !data->route.native_weight_blocks &&
            data->args.stripe_J > 1 ? data->args.stripe_J : 1;
        const size_t gcd = std::gcd(data->nr, native_scale_group);
        if (data->nr > std::numeric_limits<size_t>::max() / (native_scale_group / gcd))
            return DecStatus::invalid_arguments;
        const size_t alignment = data->nr * (native_scale_group / gcd);
        const size_t units = data->args.J / alignment + (data->args.J % alignment != 0);
        const size_t shard_count = std::min(requested_shards, units);
        data->ranges.reserve(shard_count);
        const size_t base_units = units / shard_count;
        const size_t extra_units = units % shard_count;
        size_t begin = 0;
        size_t end_unit = 0;
        for (size_t shard = 0; shard < shard_count; ++shard) {
            end_unit += base_units + (shard < extra_units ? 1 : 0);
            size_t end = data->args.J;
            if (shard + 1 != shard_count) {
                if (end_unit > std::numeric_limits<size_t>::max() / alignment)
                    return DecStatus::invalid_arguments;
                end = end_unit * alignment;
            }
            data->ranges.push_back({ begin, end });
            begin = end;
        }
        data->histogram = make_prepared_workload_histogram(
            data->residual_entries,
            data->active_row_groups,
            data->route,
            data->args.I,
            data->args.J,
            data->args.K,
            data->result.unique_k_count,
            data->result.active_row_k_pairs,
            data->result.rows_per_active_k_max,
            data->use_group_k_csc,
            data->use_int64_h1,
            data->nr);
        data->result.int_mac_count = data->histogram.estimated_int_mac_count;
        data->result.logical_weight_reference_count = data->result.int_mac_count;
        data->result.weight_scalar_load_count = data->histogram.weight_scalar_load_count;
        data->result.weight_vector_load_count = data->histogram.weight_vector_load_count;
        data->result.ycom_global_write_count = data->histogram.ycom_write_count;
        prepared = std::shared_ptr<const PreparedDecSlice>(new PreparedDecSlice(std::move(data)));
        return DecStatus::success;
    } catch (const std::bad_alloc &) {
        return DecStatus::allocation_failure;
    }
}

DecStatus execute_prepared_dec_shard(
    const PreparedDecSlice &prepared,
    size_t shard_index,
    DecShardScratch &scratch,
    float *ycom_output,
    size_t ycom_stride)
{
    if (!prepared.data_)
        return DecStatus::invalid_arguments;
    const PreparedDecSlice::Data &data = *prepared.data_;
    if (shard_index >= data.ranges.size())
        return DecStatus::invalid_shard;
    const DecColumnRange range = data.ranges[shard_index];
    const size_t width = range.end - range.begin;
    if (ycom_output != nullptr && ycom_stride != 0 && ycom_stride < data.args.J)
        return DecStatus::invalid_arguments;

    try {
        ggml_gemmini_args_t local_args;
        const DecStatus slice_status = slice_dec_columns(data.args, range.begin, range.end, local_args);
        if (slice_status != DecStatus::success)
            return slice_status;
        DecRoutePlan local_route = data.route;
        local_route.scales.rows = width;
        size_t scale_offset = 0;
        if (local_route.scales.data != nullptr) {
            if (!checked_mul_size(range.begin, local_route.scales.cols, scale_offset))
                return DecStatus::invalid_arguments;
            local_route.scales.data += scale_offset;
        }
        if (data.args.I > std::numeric_limits<size_t>::max() / width)
            return DecStatus::invalid_arguments;
        scratch.ycom.assign(data.args.I * width, 0.0f);
        scratch.result = data.result;
        scratch.result.int_mac_count = saturating_mul_size(data.result.nnz, width);
        scratch.result.logical_weight_reference_count = scratch.result.int_mac_count;
        scratch.internal_thread_count = 1;
        const float *activation_scales = data.activation_scales.empty() ? nullptr :
            data.activation_scales.data();
        GroupKCSCScalarStats group_stats;
        bool accumulated = true;
        {
            ExternalDecShardGuard guard;
            if (data.use_group_k_csc) {
                accumulated = data.use_int64_h1 ?
                    (data.nr == 4 ? accumulate_to_ycom_int64_h1_group_k_csc_nr4(
                        local_args, local_route, data.args.I, width, activation_scales,
                        data.residual_entries, data.group_k_csc_plan, scratch.ycom.data(), group_stats) :
                        accumulate_to_ycom_int64_h1_group_k_csc_nr8(
                        local_args, local_route, data.args.I, width, activation_scales,
                        data.residual_entries, data.group_k_csc_plan, scratch.ycom.data(), group_stats)) :
                    (data.nr == 4 ? accumulate_to_ycom_int32_mixed_group_k_csc_nr4(
                        local_args, local_route, data.args.I, width, activation_scales,
                        data.residual_entries, data.group_k_csc_plan, scratch.ycom.data(), group_stats) :
                        accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
                        local_args, local_route, data.args.I, width, activation_scales,
                        data.residual_entries, data.group_k_csc_plan, scratch.ycom.data(), group_stats));
            } else if (!data.residual_entries.empty()) {
                if (data.use_int64_scalar)
                    accumulate_to_ycom_int64_scalar(local_args, local_route, data.args.I, width,
                        activation_scales, data.residual_entries, data.active_row_groups,
                        data.group_offsets, data.group_row_group_indices, scratch.ycom.data());
                else if (data.use_int64_channel_direct)
                    accumulate_to_ycom_int64_channel_direct(local_args, local_route, data.args.I, width,
                        activation_scales, data.residual_entries, data.active_row_groups,
                        data.group_offsets, data.group_row_group_indices, scratch.ycom.data());
                else if (data.use_int64_channel_sidecar)
                    accumulate_to_ycom_int64_channel_sidecar(local_args, local_route, data.args.I, width,
                        activation_scales, data.residual_entries, data.active_row_groups,
                        data.group_offsets, data.group_row_group_indices, scratch.ycom.data());
                else if (data.use_int64_h1)
                    switch (data.histogram.selected_route)
                    {
                        case PreparedDecSelectedRoute::h1_small_group_single:
                            accumulate_to_ycom_int64_h1_small_group_single(
                                local_args, local_route, data.args.I, width,
                                activation_scales, data.residual_entries, data.active_row_groups,
                                data.group_offsets, data.group_row_group_indices,
                                scratch.ycom.data());
                            break;
                        case PreparedDecSelectedRoute::h1_small_group_2_to_4:
                            accumulate_to_ycom_int64_h1_small_group_2_to_4(
                                local_args, local_route, data.args.I, width,
                                activation_scales, data.residual_entries, data.active_row_groups,
                                data.group_offsets, data.group_row_group_indices,
                                scratch.ycom.data());
                            break;
                        default:
                            accumulate_to_ycom_int64_h1(local_args, local_route, data.args.I, width,
                                activation_scales, data.residual_entries, data.active_row_groups,
                                data.group_offsets, data.group_row_group_indices, scratch.ycom.data());
                            break;
                    }
                else if (data.use_int64_block)
                    accumulate_to_ycom_int64_block(local_args, local_route, data.args.I, width,
                        activation_scales, data.residual_entries, data.active_row_groups,
                        data.group_offsets, data.group_row_group_indices, scratch.ycom.data());
            }
        }
        if (!accumulated)
            return DecStatus::execution_failed;
        if (data.use_group_k_csc) {
            scratch.result.weight_scalar_load_count = group_stats.weight_scalar_load_count;
            scratch.result.weight_vector_load_count = group_stats.weight_vector_load_count;
            scratch.result.thread_scratch_bytes = group_stats.thread_scratch_bytes;
        }
        if (ycom_output != nullptr) {
            const size_t stride = ycom_stride == 0 ? data.args.J : ycom_stride;
            size_t last_row_offset = 0;
            size_t output_end = 0;
            if (data.args.I != 0 &&
                (!checked_mul_size(data.args.I - 1, stride, last_row_offset) ||
                 !checked_add_size(last_row_offset, range.begin, output_end) ||
                 !checked_add_size(output_end, width, output_end)))
                return DecStatus::invalid_arguments;
            for (size_t row = 0; row < data.args.I; ++row)
                std::copy_n(scratch.ycom.data() + row * width, width,
                    ycom_output + row * stride + range.begin);
        } else {
            apply_ycom_to_output(scratch.ycom.data(), data.args.I, width, local_args);
        }
        return DecStatus::success;
    } catch (const std::bad_alloc &) {
        return DecStatus::allocation_failure;
    }
}

ActivationDECResult compensate_activation_dec(
    const std::vector<ggml::gemmini::quants::QactOutlier> &outliers,
    ggml_gemmini_args_t &args,
    const char *layer,
    DispatchOverride dispatch_override,
    float * ycom_output,
    size_t ycom_stride)
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

    const char *disable_group_k_csc = std::getenv("DEC_GROUP_K_CSC_DISABLE");
    const bool group_k_csc_disabled = dispatch_override == DispatchOverride::row_direct ||
        (disable_group_k_csc &&
         disable_group_k_csc[0] == '1' && disable_group_k_csc[1] == '\0');
    const char *enable_group_k_csc = std::getenv("DEC_GROUP_K_CSC_ENABLE");
    const bool group_k_csc_enabled = enable_group_k_csc &&
        enable_group_k_csc[0] == '1' && enable_group_k_csc[1] == '\0';
    const char *force_group_k_csc = std::getenv("DEC_GROUP_K_CSC_FORCE");
    const bool group_k_csc_forced = dispatch_override == DispatchOverride::group_k_csc ||
        (force_group_k_csc &&
         force_group_k_csc[0] == '1' && force_group_k_csc[1] == '\0');
    const size_t group_count = K / kDecGroupSizeK + (K % kDecGroupSizeK != 0);
    result.int_mac_count = saturating_mul_size(result.nnz, J);
    result.logical_weight_reference_count = result.int_mac_count;
    result.weight_scalar_load_count = result.int_mac_count;
    result.estimated_weight_bytes_read = saturating_mul_size(
        result.weight_scalar_load_count, sizeof(int8_t));
    const double rows_per_active_k_mean = result.unique_k_count == 0 ? 0.0 :
        static_cast<double>(result.active_row_k_pairs) / result.unique_k_count;
    const size_t estimated_group_k_csc_plan_bytes = estimate_group_k_csc_plan_bytes(
        group_count, scr.residual_entries.size(), scr.active_row_groups.size());
    const size_t estimated_group_k_csc_saved_weight_bytes = estimate_group_k_csc_saved_weight_bytes(
        result.logical_weight_reference_count, result.unique_k_count, J);
    const bool group_k_csc_common = (use_int64_scalar || use_int64_h1) &&
        I > 1 &&
        J >= 8 &&
        rows_per_active_k_mean >= 16.0 &&
        estimated_group_k_csc_saved_weight_bytes > estimated_group_k_csc_plan_bytes;
    const bool group_k_csc_supported_route = use_int64_scalar || use_int64_h1;
    const bool group_k_csc_override = (group_k_csc_forced || group_k_csc_enabled) && I > 1;
    bool use_group_k_csc = !group_k_csc_disabled && group_k_csc_supported_route &&
        (group_k_csc_override || group_k_csc_common);
    const size_t group_k_csc_nr = J < 8 ? 4 : 8;

    start = ggml::gemmini::cycle::read();
    build_group_major_index(
        scr.active_row_groups, group_count, scr.group_offsets, scr.group_row_group_indices);

    end = ggml::gemmini::cycle::read();
    ggml::gemmini::log::cycle(layer, "[dec] cpu.Build group-major index", start, end);

    if (use_group_k_csc)
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

    GroupKCSCScalarStats group_k_csc_stats;
    if (use_group_k_csc)
    {
        const bool accumulated = use_int64_h1 ?
            (group_k_csc_nr == 4 ?
                accumulate_to_ycom_int64_h1_group_k_csc_nr4(
                    args, plan, I, J, activation_scales, scr.residual_entries,
                    scr.group_k_csc_plan, scr.Y_com.data(), group_k_csc_stats) :
                accumulate_to_ycom_int64_h1_group_k_csc_nr8(
                    args, plan, I, J, activation_scales, scr.residual_entries,
                    scr.group_k_csc_plan, scr.Y_com.data(), group_k_csc_stats)) :
            (group_k_csc_nr == 4 ?
                accumulate_to_ycom_int32_mixed_group_k_csc_nr4(
                    args, plan, I, J, activation_scales, scr.residual_entries,
                    scr.group_k_csc_plan, scr.Y_com.data(), group_k_csc_stats) :
                accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
                    args, plan, I, J, activation_scales, scr.residual_entries,
                    scr.group_k_csc_plan, scr.Y_com.data(), group_k_csc_stats));
        use_group_k_csc = accumulated;
    }

    if (use_group_k_csc)
    {
        result.logical_weight_reference_count = group_k_csc_stats.logical_weight_reference_count;
        result.weight_scalar_load_count = group_k_csc_stats.weight_scalar_load_count;
        result.weight_vector_load_count = group_k_csc_stats.weight_vector_load_count;
        result.estimated_weight_bytes_read = saturating_mul_size(
            result.weight_scalar_load_count, sizeof(int8_t));
    }

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
    result.thread_scratch_bytes = use_group_k_csc ? group_k_csc_stats.thread_scratch_bytes :
        saturating_mul_size(
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
        "[dec.route] algorithm=grouped sparse_kernel=%s traversal=%s j_inner_nr=%zu row_panel=full weight_vector=%s accumulator=%s output_panel=off group_size_k=%zu activation=%s weight=%s accessor=%s reducer=%s j_partition=contiguous-range microtile=%zu threads=%d residual_format=common weight_layout=%s scale_mode=%s I=%zu J=%zu K=%zu",
        use_group_k_csc ? "group-k-csc" : "row-direct",
        use_group_k_csc ? "group-k-major" : "row-major",
        use_group_k_csc ? group_k_csc_nr : size_t {1},
        use_group_k_csc ? "vector" : "direct",
        use_group_k_csc ? (use_int64_h1 ? "int64" : "mixed-int32-int64") : "int64",
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
    if (use_group_k_csc)
    {
        const char *width_path = group_k_csc_stats.width_path == GroupKCSCWidthPath::AllInt32 ?
            "all-int32" : group_k_csc_stats.width_path == GroupKCSCWidthPath::AllInt64 ?
            "all-int64" : "mixed";
        ggml::gemmini::log::debug(
            layer,
            "[dec.group-k-csc] width_path=%s classify=%zu scratch_init=%zu sparse_update=%zu merge=%zu safe_updates=%zu fallback_updates=%zu branch_entry_classify=%zu classify_cycles=%llu scratch_init_cycles=%llu sparse_update_cycles=%llu merge_cycles=%llu",
            width_path,
            group_k_csc_stats.classification_work_count,
            group_k_csc_stats.scratch_init_count,
            group_k_csc_stats.sparse_update_count,
            group_k_csc_stats.merge_count,
            group_k_csc_stats.safe_update_count,
            group_k_csc_stats.fallback_update_count,
            group_k_csc_stats.branch_entry_classification_count,
            static_cast<unsigned long long>(group_k_csc_stats.classification_cycles),
            static_cast<unsigned long long>(group_k_csc_stats.scratch_init_cycles),
            static_cast<unsigned long long>(group_k_csc_stats.sparse_update_cycles),
            static_cast<unsigned long long>(group_k_csc_stats.merge_cycles));
    }
#endif

    if (!use_group_k_csc)
    {
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
    if (ycom_output != nullptr) {
        const size_t output_stride = ycom_stride == 0 ? J : ycom_stride;
        for (size_t row = 0; row < I; ++row)
            std::copy_n(scr.Y_com.data() + row * J, J, ycom_output + row * output_stride);
    } else {
        apply_ycom_to_output(scr.Y_com.data(), I, J, args);
    }
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

ActivationDECRowSliceStatus compensate_activation_dec_rows(
    const std::vector<QactOutlier> &outliers,
    ggml_gemmini_args_t &args,
    size_t row_begin,
    size_t row_end,
    const char *layer,
    DispatchOverride dispatch_override)
{
    const auto &storage = args.act_quant.storage();
    if (!std::holds_alternative<act::NoneMeta>(storage) &&
        !std::holds_alternative<act::tensor::Meta>(storage)) {
        return ActivationDECRowSliceStatus::unsupported;
    }
    if (row_begin >= row_end || row_end > args.I || args.A == nullptr || args.f_out == nullptr) {
        return ActivationDECRowSliceStatus::invalid_arguments;
    }

    const size_t input_stride = args.sA ? args.sA : args.K;
    const size_t output_stride = args.stride_f_out ? args.stride_f_out : args.J;
    if ((input_stride != 0 && row_begin > std::numeric_limits<size_t>::max() / input_stride) ||
        (output_stride != 0 && row_begin > std::numeric_limits<size_t>::max() / output_stride)) {
        return ActivationDECRowSliceStatus::invalid_arguments;
    }

    std::vector<QactOutlier> local_outliers;
    for (const QactOutlier &outlier : outliers) {
        if (outlier.row >= 0 && static_cast<size_t>(outlier.row) >= row_begin &&
            static_cast<size_t>(outlier.row) < row_end) {
            local_outliers.push_back({
                static_cast<int>(static_cast<size_t>(outlier.row) - row_begin),
                outlier.col,
                outlier.residual,
            });
        }
    }

    ggml_gemmini_args_t local_args = args;
    local_args.I = row_end - row_begin;
    local_args.A += row_begin * input_stride;
    local_args.f_out += row_begin * output_stride;
    compensate_activation_dec(local_outliers, local_args, layer, dispatch_override);
    return ActivationDECRowSliceStatus::success;
}

ActivationDECRowSliceStatus compensate_activation_dec_rows_columns(
    const std::vector<QactOutlier> &outliers,
    ggml_gemmini_args_t &args,
    size_t row_begin,
    size_t row_end,
    size_t col_begin,
    size_t col_end,
    const char *layer,
    DispatchOverride dispatch_override,
    float * ycom_output,
    size_t ycom_stride)
{
    if (row_begin >= row_end || row_end > args.I || col_begin >= col_end || col_end > args.J)
        return ActivationDECRowSliceStatus::invalid_arguments;
    if (args.A == nullptr || args.f_out == nullptr)
        return ActivationDECRowSliceStatus::invalid_arguments;

    const size_t input_stride = args.sA ? args.sA : args.K;
    const size_t output_row_stride = args.stride_f_out ? args.stride_f_out : args.J;
    const size_t output_col_stride = args.col_stride_f_out ? args.col_stride_f_out : 1;
    size_t row_offset = 0;
    size_t col_offset = 0;
    size_t output_offset = 0;
    size_t offset = 0;
    if (!checked_mul_size(row_begin, input_stride, row_offset) ||
        !checked_mul_size(row_begin, output_row_stride, offset) ||
        !checked_mul_size(col_begin, output_col_stride, col_offset) ||
        !checked_add_size(offset, col_offset, output_offset))
        return ActivationDECRowSliceStatus::invalid_arguments;

    const size_t width = col_end - col_begin;
    ggml_gemmini_args_t local_args = args;
    size_t global_rows_per_tile = 0;
    if (args.tile_I != 0) {
        if (!checked_mul_size(args.tile_I, DIM, global_rows_per_tile))
            return ActivationDECRowSliceStatus::invalid_arguments;
    } else {
        if (!checked_add_size(args.I, DIM - 1, offset) ||
            !checked_mul_size(offset / DIM, DIM, global_rows_per_tile))
            return ActivationDECRowSliceStatus::invalid_arguments;
    }
    if (!checked_add_size(args.activation_row_offset, row_begin, offset))
        return ActivationDECRowSliceStatus::invalid_arguments;
    local_args.tile_I = global_rows_per_tile / DIM;
    local_args.I = row_end - row_begin;
    local_args.J = width;
    local_args.A += row_offset;
    local_args.f_out += output_offset;
    local_args.activation_row_offset = offset;

    const bool channel_direct = args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel;
    const bool jxk_layout = channel_direct ||
        args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar ||
        args.transpose_B ||
        args.q8_h1_blocks != nullptr || args.q8_h2_blocks != nullptr ||
        args.q8_hp1_blocks != nullptr || args.q8_hp2_blocks != nullptr || args.c_b != nullptr;

    if (args.B != nullptr) {
        const size_t weight_stride = channel_direct ? args.q8_channel_row_stride :
            (jxk_layout ? (args.sB ? args.sB : args.K) : 1);
        if (!checked_mul_size(col_begin, weight_stride, offset))
            return ActivationDECRowSliceStatus::invalid_arguments;
        if (channel_direct)
            local_args.B = reinterpret_cast<elem_t *>(reinterpret_cast<uint8_t *>(args.B) +
                offset);
        else if (jxk_layout)
            local_args.B = reinterpret_cast<elem_t *>(reinterpret_cast<int8_t *>(args.B) +
                offset);
        else
            local_args.B = reinterpret_cast<elem_t *>(reinterpret_cast<int8_t *>(args.B) + col_begin);
    }
    if (args.B_blocks != nullptr && args.blocks_K != 0) {
        if (!checked_mul_size(col_begin, args.blocks_K, offset))
            return ActivationDECRowSliceStatus::invalid_arguments;
        local_args.B_blocks = args.B_blocks + offset;
    }
    if (args.B_scales != nullptr && args.blocks_K != 0) {
        if (!checked_mul_size(col_begin, args.blocks_K, offset))
            return ActivationDECRowSliceStatus::invalid_arguments;
        local_args.B_scales = args.B_scales + offset;
    }
    if (args.weight_channel_scales != nullptr)
        local_args.weight_channel_scales = args.weight_channel_scales + col_begin;
    local_args.weight_channel_scale_count = width;

    if (args.q8_channel_row_base != nullptr) {
        if (!checked_mul_size(col_begin, args.q8_channel_row_stride, offset))
            return ActivationDECRowSliceStatus::invalid_arguments;
        local_args.q8_channel_row_base = args.q8_channel_row_base + offset;
        local_args.q8_channel_row_count = width;
    }
    if (args.q8_h1_blocks != nullptr) {
        if (!checked_mul_size(col_begin, args.blocks_per_row, offset) ||
            !checked_mul_size(width, args.blocks_per_row, local_args.q8_h1_block_count))
            return ActivationDECRowSliceStatus::invalid_arguments;
        local_args.q8_h1_blocks = args.q8_h1_blocks + offset;
        local_args.q8_h1_rows = width;
    }
    if (args.q8_h2_blocks != nullptr) {
        if (!checked_mul_size(col_begin, args.q8_h2_blocks_per_row, offset) ||
            !checked_mul_size(width, args.q8_h2_blocks_per_row, local_args.q8_h2_block_count))
            return ActivationDECRowSliceStatus::invalid_arguments;
        local_args.q8_h2_blocks = args.q8_h2_blocks + offset;
        local_args.q8_h2_blocks_per_row = args.q8_h2_blocks_per_row;
    }
    if (args.q8_hp1_blocks != nullptr) {
        if (!checked_mul_size(col_begin, args.q8_hp1_blocks_per_row, offset) ||
            !checked_mul_size(width, args.q8_hp1_blocks_per_row, local_args.q8_hp1_block_count))
            return ActivationDECRowSliceStatus::invalid_arguments;
        local_args.q8_hp1_blocks = args.q8_hp1_blocks + offset;
        local_args.q8_hp1_blocks_per_row = args.q8_hp1_blocks_per_row;
    }
    if (args.q8_hp2_blocks != nullptr) {
        if (!checked_mul_size(col_begin, args.q8_hp2_blocks_per_row, offset) ||
            !checked_mul_size(width, args.q8_hp2_blocks_per_row, local_args.q8_hp2_block_count))
            return ActivationDECRowSliceStatus::invalid_arguments;
        local_args.q8_hp2_blocks = args.q8_hp2_blocks + offset;
        local_args.q8_hp2_blocks_per_row = args.q8_hp2_blocks_per_row;
    }
    if (args.c_b != nullptr && args.blocks_per_row != 0) {
        if (!checked_mul_size(col_begin, args.blocks_per_row, offset))
            return ActivationDECRowSliceStatus::invalid_arguments;
        local_args.c_b = args.c_b + offset;
    }
    if (args.s_rf != nullptr)
        local_args.s_rf = args.s_rf + col_begin;
    if (args.R != nullptr)
        local_args.R = args.R + col_begin;
    if (args.stripe_J > 1) {
        if (args.s_rf_stripe == nullptr || args.R_stripe == nullptr || col_begin % args.stripe_J != 0)
            return ActivationDECRowSliceStatus::unsupported;
        local_args.s_rf_stripe = args.s_rf_stripe + col_begin / args.stripe_J;
        local_args.R_stripe = args.R_stripe + col_begin / args.stripe_J;
    }

    std::vector<QactOutlier> local_outliers;
    local_outliers.reserve(outliers.size());
    for (const QactOutlier &outlier : outliers) {
        if (outlier.row >= 0 && static_cast<size_t>(outlier.row) >= row_begin &&
            static_cast<size_t>(outlier.row) < row_end)
            local_outliers.push_back({static_cast<int>(static_cast<size_t>(outlier.row) - row_begin),
                                      outlier.col, outlier.residual});
    }
    if (local_outliers.empty())
        return ActivationDECRowSliceStatus::success;
    const ActivationDECResult result = compensate_activation_dec(
        local_outliers, local_args, layer, dispatch_override, ycom_output, ycom_stride);
    return ycom_output != nullptr && result.int_mac_count == 0 ?
        ActivationDECRowSliceStatus::unsupported : ActivationDECRowSliceStatus::success;
}
} // namespace ggml::gemmini::quants::dec
