#include "dec_kernel.hpp"
#include "dec_internal.hpp"
#include "../../ggml-gemmini-args.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <vector>

#ifndef DEC_VALIDATION
#define DEC_VALIDATION 0
#endif

#if defined(GGML_GEMMINI_HAS_OPENMP)
#include <omp.h>
#endif

namespace ggml::gemmini::quants::dec
{
int resolve_dec_threads(size_t task_count, int omp_max_threads)
{
    const int task_limit = task_count > static_cast<size_t>(std::numeric_limits<int>::max()) ?
        std::numeric_limits<int>::max() : std::max(1, static_cast<int>(task_count));
    const int worker_limit = std::max(1, omp_max_threads);
    int dec_threads = std::min(task_limit, worker_limit);

    if (const char *env = std::getenv("DEC_THREADS"))
    {
        char *end = nullptr;
        errno = 0;
        const long long parsed = std::strtoll(env, &end, 10);
        if (errno == 0 && end != env && end != nullptr && *end == '\0' &&
            parsed > 0 && parsed <= std::numeric_limits<int>::max())
            dec_threads = static_cast<int>(parsed);
    }

    return std::min(std::max(1, dec_threads), std::min(task_limit, worker_limit));
}

int resolve_dec_threads(size_t task_count)
{
#if defined(GGML_GEMMINI_HAS_OPENMP)
    return resolve_dec_threads(task_count, omp_get_max_threads());
#else
    return resolve_dec_threads(task_count, 1);
#endif
}

namespace
{
    size_t resolve_out_stride_row(const ggml_gemmini_args_t &args)
    {
        return args.stride_f_out ? args.stride_f_out : args.J;
    }

    size_t resolve_out_stride_col(const ggml_gemmini_args_t &args)
    {
        return args.col_stride_f_out ? args.col_stride_f_out : 1;
    }

    template <typename TileWork>
    void for_each_grouped_j_tile(size_t scratch_rows, size_t J, TileWork tile_work)
    {
        if (scratch_rows == 0 || J == 0)
            return;

        const size_t tile_capacity = std::min(J, kDecInt64JTileWidth);
        const auto run_range = [&](size_t j_begin, size_t j_end)
        {
            std::vector<int64_t> accumulator(scratch_rows * tile_capacity);
            for (size_t jb = j_begin; jb < j_end; jb += kDecInt64JTileWidth)
                tile_work(jb, std::min(kDecInt64JTileWidth, j_end - jb), accumulator);
        };

#if defined(GGML_GEMMINI_HAS_OPENMP)
        const size_t tile_count = dec_int64_j_tile_count(J);
        const int dec_threads = resolve_dec_threads(tile_count);
        if (dec_threads == 1 || omp_in_parallel() != 0)
        {
            run_range(0, J);
            return;
        }
#pragma omp parallel num_threads(dec_threads)
        {
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
            const size_t threads = static_cast<size_t>(omp_get_num_threads());
            const size_t first_tile = tile_count * tid / threads;
            const size_t last_tile = tile_count * (tid + 1) / threads;
            run_range(first_tile * kDecInt64JTileWidth,
                      std::min(J, last_tile * kDecInt64JTileWidth));
        }
#else
        run_range(0, J);
#endif
    }

    template <typename CodeFor>
    void add_group_dot(
        const std::vector<ResidualGroupEntry> &entries,
        size_t begin,
        size_t end,
        size_t jb,
        size_t width,
        CodeFor code_for,
        int64_t *accumulator)
    {
        for (size_t p = begin; p < end; ++p)
        {
            const ResidualGroupEntry &entry = entries[p];
            const int64_t residual = entry.residual;
            for (size_t t = 0; t < width; ++t)
            {
                const int64_t term = residual * code_for(jb + t, entry.k);
#if DEC_VALIDATION
                const __int128 checked_sum = static_cast<__int128>(accumulator[t]) + term;
                if (checked_sum < std::numeric_limits<int64_t>::min() ||
                    checked_sum > std::numeric_limits<int64_t>::max())
                    std::abort();
#endif
                accumulator[t] += term;
            }
        }
    }

    template <typename CodeFor, typename ScaleForColumn>
    void run_whole_k_grouped_dec(
        size_t I,
        size_t J,
        const float *activation_scales,
        const std::vector<ResidualGroupEntry> &entries,
        const std::vector<ActiveRowGroup> &groups,
        const std::vector<size_t> &group_offsets,
        const std::vector<size_t> &group_row_group_indices,
        CodeFor code_for,
        ScaleForColumn scale_for_column,
        float *Y_com)
    {
        if (!Y_com || entries.empty() || groups.empty() || group_offsets.size() < 2 ||
            group_row_group_indices.size() != groups.size() || I == 0 || J == 0)
            return;

        for_each_grouped_j_tile(I, J, [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
        {
            std::fill(accumulator.begin(), accumulator.begin() + I * width, int64_t {0});
            for (size_t k_group = 0; k_group + 1 < group_offsets.size(); ++k_group)
                for (size_t position = group_offsets[k_group]; position < group_offsets[k_group + 1]; ++position)
                {
                    const ActiveRowGroup &group = groups[group_row_group_indices[position]];
                    add_group_dot(entries, group.entry_begin, group.entry_end, jb, width, code_for,
                                  accumulator.data() + static_cast<size_t>(group.row) * width);
                }

            uint32_t previous_row = std::numeric_limits<uint32_t>::max();
            for (const ActiveRowGroup &group : groups)
            {
                if (group.row == previous_row)
                    continue;
                previous_row = group.row;
                const size_t row_index = group.row;
                const float activation_scale = activation_scales ? activation_scales[row_index] : 1.0f;
                const int64_t *row = accumulator.data() + row_index * width;
                float *output = Y_com + row_index * J + jb;
                for (size_t t = 0; t < width; ++t)
                    output[t] += static_cast<float>(
                        static_cast<double>(row[t]) * activation_scale * scale_for_column(jb + t));
            }
        });
    }

    template <typename CodeFor, typename ScaleFor>
    void run_scaled_grouped_dec(
        size_t I,
        size_t J,
        size_t scale_group_size,
        const float *activation_scales,
        const std::vector<ResidualGroupEntry> &entries,
        const std::vector<ActiveRowGroup> &groups,
        const std::vector<size_t> &group_offsets,
        const std::vector<size_t> &group_row_group_indices,
        CodeFor code_for,
        ScaleFor scale_for,
        float *Y_com)
    {
        if (!Y_com || entries.empty() || groups.empty() || group_offsets.size() < 2 ||
            group_row_group_indices.size() != groups.size() || I == 0 || J == 0 || scale_group_size == 0)
            return;

        for_each_grouped_j_tile(1, J, [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
        {
            alignas(64) std::array<float, kDecInt64JTileWidth> scale_tile{};
            for (size_t k_group = 0; k_group + 1 < group_offsets.size(); ++k_group)
            {
                if (group_offsets[k_group] == group_offsets[k_group + 1])
                    continue;

                if (scale_group_size == kDecGroupSizeK)
                    for (size_t t = 0; t < width; ++t)
                        scale_tile[t] = scale_for(jb + t, k_group);

                for (size_t position = group_offsets[k_group]; position < group_offsets[k_group + 1]; ++position)
                {
                    const ActiveRowGroup &group = groups[group_row_group_indices[position]];
                    for (size_t begin = group.entry_begin; begin < group.entry_end;)
                    {
                        const size_t scale_block = entries[begin].k / scale_group_size;
                        size_t end = begin + 1;
                        while (end < group.entry_end && entries[end].k / scale_group_size == scale_block)
                            ++end;

                        std::fill(accumulator.begin(), accumulator.begin() + width, int64_t {0});
                        add_group_dot(entries, begin, end, jb, width, code_for, accumulator.data());
                        const size_t row = group.row;
                        const float activation_scale = activation_scales ? activation_scales[row] : 1.0f;
                        float *output = Y_com + row * J + jb;
                        for (size_t t = 0; t < width; ++t)
                        {
                            const float weight_scale = scale_group_size == kDecGroupSizeK ?
                                scale_tile[t] : scale_for(jb + t, scale_block);
                            output[t] += static_cast<float>(
                                static_cast<double>(accumulator[t]) * activation_scale * weight_scale);
                        }
                        begin = end;
                    }
                }
            }
        });
    }

    template <typename CodeFor>
    void run_block_route(
        const ggml_gemmini_args_t &args,
        const DecRoutePlan &plan,
        size_t I,
        size_t J,
        const float *activation_scales,
        const std::vector<ResidualGroupEntry> &entries,
        const std::vector<ActiveRowGroup> &groups,
        const std::vector<size_t> &group_offsets,
        const std::vector<size_t> &group_row_group_indices,
        CodeFor code_for,
        float *Y_com)
    {
        run_scaled_grouped_dec(I, J, plan.scales.block_size, activation_scales, entries, groups,
            group_offsets, group_row_group_indices,
            code_for,
            [&args, &plan](size_t j, size_t block) {
                return dec_route_weight_scale(plan, args, j, block);
            }, Y_com);
    }
}

void accumulate_to_ycom_int64_scalar(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups, const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    run_whole_k_grouped_dec(I, J, activation_scales, entries, groups,
        group_offsets, group_row_group_indices,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, [scale = plan.scales.scalar](size_t) { return scale; }, Y_com);
}

void accumulate_to_ycom_int64_channel_direct(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups, const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    run_whole_k_grouped_dec(I, J, activation_scales, entries, groups,
        group_offsets, group_row_group_indices,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, [&args](size_t j) { return args.q8_channel_scale(j); }, Y_com);
}

void accumulate_to_ycom_int64_channel_sidecar(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups, const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    run_whole_k_grouped_dec(I, J, activation_scales, entries, groups,
        group_offsets, group_row_group_indices,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, [scales = plan.scales.data](size_t j) { return scales[j]; }, Y_com);
}

void accumulate_to_ycom_int64_block(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups, const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices, float *Y_com)
{
    if (plan.route == DecWeightRoute::Q8HP1)
        return run_block_route(args, plan, I, J, activation_scales, entries, groups,
            group_offsets, group_row_group_indices,
            [&args](size_t j, size_t k) { return args.q8_hp1_block(j, k / QK8_HP)->qs[k % QK8_HP]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8HP2)
        return run_block_route(args, plan, I, J, activation_scales, entries, groups,
            group_offsets, group_row_group_indices,
            [&args](size_t j, size_t k) { return args.q8_hp2_block(j, k / QK8_HP)->qs[k % QK8_HP]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8H2)
        return run_block_route(args, plan, I, J, activation_scales, entries, groups,
            group_offsets, group_row_group_indices,
            [&args](size_t j, size_t k) { return args.q8_h2_block(j, k / QK8_H2)->qs[k % QK8_H2]; }, Y_com);

    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    run_block_route(args, plan, I, J, activation_scales, entries, groups,
        group_offsets, group_row_group_indices,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, Y_com);
}

void accumulate_to_ycom_int64_h1(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups, const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    run_scaled_grouped_dec(I, J, kDecGroupSizeK, activation_scales, entries, groups,
        group_offsets, group_row_group_indices,
        [&args, &plan, weights](size_t j, size_t k) {
            const block_q8_h1 *native = plan.native_weight_blocks ? args.q8_h1_block(j, k / QK8_0) : nullptr;
            return native ? native->qs[k % QK8_0] : weights[j * plan.weight_stride + k];
        },
        [&args, &plan](size_t j, size_t block) {
            return dec_route_weight_scale(plan, args, j, block);
        }, Y_com);
}

void apply_ycom_to_output(
    const float *Y_com,
    size_t I,
    size_t J,
    const ggml_gemmini_args_t &args)
{
    float *out_data = args.f_out;
    if (!Y_com || !out_data || I == 0 || J == 0)
        return;

    const size_t stride_row = resolve_out_stride_row(args);
    const size_t stride_col = resolve_out_stride_col(args);
    for (size_t r = 0; r < I; ++r)
    {
        const float *src = Y_com + r * J;
        float *dst = out_data + r * stride_row;
        if (stride_col == 1)
            for (size_t j = 0; j < J; ++j)
                dst[j] += src[j];
        else
            for (size_t j = 0; j < J; ++j)
                dst[j * stride_col] += src[j];
    }
}
} // namespace ggml::gemmini::quants::dec
