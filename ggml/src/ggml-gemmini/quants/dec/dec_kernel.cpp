#include "dec_kernel.hpp"
#include "dec_internal.hpp"
#include "../../ggml-gemmini-args.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <limits>

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
    void for_each_int64_j_tile(size_t I, size_t J, TileWork tile_work)
    {
        if (I == 0 || J == 0)
            return;

        const size_t tile_count = dec_int64_j_tile_count(J);
        const size_t tile_capacity = std::min(J, kDecInt64JTileWidth);

        const auto run_serial = [&]()
        {
            std::vector<int64_t> accumulator(I * tile_capacity);
            for (size_t tile_index = 0; tile_index < tile_count; ++tile_index)
            {
                const size_t jb = tile_index * kDecInt64JTileWidth;
                tile_work(jb, std::min(kDecInt64JTileWidth, J - jb), accumulator);
            }
        };

#if defined(GGML_GEMMINI_HAS_OPENMP)
        const int dec_threads = resolve_dec_threads(tile_count);
        if (dec_threads == 1 || omp_in_parallel() != 0)
        {
            run_serial();
            return;
        }
#pragma omp parallel num_threads(dec_threads)
        {
            std::vector<int64_t> accumulator(I * tile_capacity);
#pragma omp for schedule(static)
            for (ptrdiff_t tile_index = 0; tile_index < static_cast<ptrdiff_t>(tile_count); ++tile_index)
            {
                const size_t jb = static_cast<size_t>(tile_index) * kDecInt64JTileWidth;
                tile_work(jb, std::min(kDecInt64JTileWidth, J - jb), accumulator);
            }
        }
#else
        run_serial();
#endif
    }

    std::vector<size_t> h1_active_blocks(
        const ggml_gemmini_args_t &args,
        const DecRoutePlan &plan,
        const std::vector<int> &unique_k)
    {
        std::vector<size_t> blocks;
        blocks.reserve(std::min(unique_k.size(), plan.scales.cols));
        for (int k : unique_k)
        {
            if (k < 0 || static_cast<size_t>(k) >= args.K)
                continue;
            const size_t block = static_cast<size_t>(k) / plan.scales.block_size;
            if (block < plan.scales.cols)
                blocks.push_back(block);
        }
        std::sort(blocks.begin(), blocks.end());
        blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
        return blocks;
    }

    template <typename CodeFor, typename ScaleForColumn>
    void accumulate_to_ycom_int64_impl(
        const ggml_gemmini_args_t &args,
        size_t I,
        size_t J,
        const float *activation_scales,
        const std::vector<int> &unique_k,
        const std::vector<size_t> &rk_offs,
        const std::pair<int, int32_t> *rk_pairs,
        CodeFor code_for,
        ScaleForColumn scale_for_column,
        float *Y_com)
    {
        if (!rk_pairs || !Y_com || I == 0 || J == 0)
            return;

        for_each_int64_j_tile(I, J, [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
        {
            std::fill(accumulator.begin(), accumulator.begin() + I * width, int64_t {0});
            for (int k : unique_k)
            {
                if (k < 0)
                    continue;
                const size_t k_sz = static_cast<size_t>(k);
                if (k_sz >= args.K || k_sz + 1 >= rk_offs.size())
                    continue;

                for (size_t p = rk_offs[k_sz]; p < rk_offs[k_sz + 1]; ++p)
                {
                    const int r = rk_pairs[p].first;
                    if (r < 0 || static_cast<size_t>(r) >= I)
                        continue;

                    int64_t *row = accumulator.data() + static_cast<size_t>(r) * width;
                    const int64_t residual = rk_pairs[p].second;
                    for (size_t t = 0; t < width; ++t)
                        row[t] += residual * code_for(jb + t, k_sz);
                }
            }

            for (size_t r = 0; r < I; ++r)
            {
                const float activation_scale = activation_scales ? activation_scales[r] : 1.0f;
                const int64_t *row = accumulator.data() + r * width;
                float *output = Y_com + r * J + jb;
                for (size_t t = 0; t < width; ++t)
                    output[t] += static_cast<float>(
                        static_cast<double>(row[t]) * activation_scale * scale_for_column(jb + t));
            }
        });
    }

    template <typename CodeFor, typename ScaleForColumn>
    void accumulate_single_row_to_ycom_int64_impl(
        const ggml_gemmini_args_t &args,
        size_t J,
        const float *activation_scales,
        const std::vector<int> &unique_k,
        const std::vector<int64_t> &delta_by_k,
        CodeFor code_for,
        ScaleForColumn scale_for_column,
        float *Y_com)
    {
        if (!Y_com || J == 0)
            return;

        for_each_int64_j_tile(1, J, [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
        {
            std::fill(accumulator.begin(), accumulator.begin() + width, int64_t {0});
            for (int k : unique_k)
            {
                if (k < 0)
                    continue;
                const size_t k_sz = static_cast<size_t>(k);
                if (k_sz >= args.K || k_sz >= delta_by_k.size())
                    continue;

                const int64_t residual = delta_by_k[k_sz];
                for (size_t t = 0; t < width; ++t)
                    accumulator[t] += residual * code_for(jb + t, k_sz);
            }

            const float activation_scale = activation_scales ? activation_scales[0] : 1.0f;
            for (size_t t = 0; t < width; ++t)
                Y_com[jb + t] += static_cast<float>(
                    static_cast<double>(accumulator[t]) * activation_scale * scale_for_column(jb + t));
        });
    }

    template <typename CodeFor>
    void accumulate_to_ycom_int64_block_impl(
        const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
        const float *activation_scales, const std::vector<int> &unique_k,
        const std::vector<size_t> &rk_offs, const std::pair<int, int32_t> *rk_pairs,
        CodeFor code_for, float *Y_com)
    {
        if (!rk_pairs || !Y_com || I == 0 || J == 0)
            return;

        for_each_int64_j_tile(I, J, [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
        {
            for (size_t block = 0; block < plan.scales.cols; ++block)
            {
                std::fill(accumulator.begin(), accumulator.begin() + I * width, int64_t {0});
                for (int k : unique_k)
                {
                    if (k < 0)
                        continue;
                    const size_t k_sz = static_cast<size_t>(k);
                    if (k_sz >= args.K || k_sz / plan.scales.block_size != block || k_sz + 1 >= rk_offs.size())
                        continue;
                    for (size_t p = rk_offs[k_sz]; p < rk_offs[k_sz + 1]; ++p)
                    {
                        const int r = rk_pairs[p].first;
                        if (r < 0 || static_cast<size_t>(r) >= I)
                            continue;
                        int64_t *row = accumulator.data() + static_cast<size_t>(r) * width;
                        const int64_t residual = rk_pairs[p].second;
                        for (size_t t = 0; t < width; ++t)
                            row[t] += residual * code_for(jb + t, k_sz);
                    }
                }
                for (size_t r = 0; r < I; ++r)
                {
                    const float activation_scale = activation_scales ? activation_scales[r] : 1.0f;
                    const int64_t *row = accumulator.data() + r * width;
                    float *output = Y_com + r * J + jb;
                    for (size_t t = 0; t < width; ++t)
                        output[t] += static_cast<float>(static_cast<double>(row[t]) * activation_scale *
                            dec_route_weight_scale(plan, args, jb + t, block));
                }
            }
        });
    }

    template <typename CodeFor>
    void accumulate_single_row_to_ycom_int64_block_impl(
        const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t J,
        const float *activation_scales, const std::vector<int> &unique_k,
        const std::vector<int64_t> &delta_by_k,
        CodeFor code_for, float *Y_com)
    {
        if (!Y_com || J == 0)
            return;

        for_each_int64_j_tile(1, J, [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
        {
            for (size_t block = 0; block < plan.scales.cols; ++block)
            {
                std::fill(accumulator.begin(), accumulator.begin() + width, int64_t {0});
                for (int k : unique_k)
                {
                    if (k < 0)
                        continue;
                    const size_t k_sz = static_cast<size_t>(k);
                    if (k_sz >= args.K || k_sz >= delta_by_k.size() || k_sz / plan.scales.block_size != block)
                        continue;
                    const int64_t residual = delta_by_k[k_sz];
                    for (size_t t = 0; t < width; ++t)
                        accumulator[t] += residual * code_for(jb + t, k_sz);
                }
                const float activation_scale = activation_scales ? activation_scales[0] : 1.0f;
                for (size_t t = 0; t < width; ++t)
                    Y_com[jb + t] += static_cast<float>(static_cast<double>(accumulator[t]) * activation_scale *
                        dec_route_weight_scale(plan, args, jb + t, block));
            }
        });
    }
}

void accumulate_to_ycom_int64_scalar(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs, const std::pair<int, int32_t> *rk_pairs,
    float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    accumulate_to_ycom_int64_impl(args, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, [scale = plan.scales.scalar](size_t) { return scale; }, Y_com);
}

void accumulate_single_row_to_ycom_int64_scalar(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    accumulate_single_row_to_ycom_int64_impl(args, J, activation_scales, unique_k, delta_by_k,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, [scale = plan.scales.scalar](size_t) { return scale; }, Y_com);
}

void accumulate_to_ycom_int64_channel_direct(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs, const std::pair<int, int32_t> *rk_pairs,
    float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    accumulate_to_ycom_int64_impl(args, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, [&args](size_t j) { return args.q8_channel_scale(j); }, Y_com);
}

void accumulate_single_row_to_ycom_int64_channel_direct(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    accumulate_single_row_to_ycom_int64_impl(args, J, activation_scales, unique_k, delta_by_k,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, [&args](size_t j) { return args.q8_channel_scale(j); }, Y_com);
}

void accumulate_to_ycom_int64_channel_sidecar(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs, const std::pair<int, int32_t> *rk_pairs,
    float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    accumulate_to_ycom_int64_impl(args, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, [scales = plan.scales.data](size_t j) { return scales[j]; }, Y_com);
}

void accumulate_single_row_to_ycom_int64_channel_sidecar(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights)
        return;
    accumulate_single_row_to_ycom_int64_impl(args, J, activation_scales, unique_k, delta_by_k,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, [scales = plan.scales.data](size_t j) { return scales[j]; }, Y_com);
}

void accumulate_to_ycom_int64_block(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs, const std::pair<int, int32_t> *rk_pairs,
    float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (plan.route == DecWeightRoute::Q8HP1)
        return accumulate_to_ycom_int64_block_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
            [&args](size_t j, size_t k) { return args.q8_hp1_block(j, k / QK8_HP)->qs[k % QK8_HP]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8HP2)
        return accumulate_to_ycom_int64_block_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
            [&args](size_t j, size_t k) { return args.q8_hp2_block(j, k / QK8_HP)->qs[k % QK8_HP]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8H2)
        return accumulate_to_ycom_int64_block_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
            [&args](size_t j, size_t k) { return args.q8_h2_block(j, k / QK8_H2)->qs[k % QK8_H2]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8H1 && plan.native_weight_blocks)
        return accumulate_to_ycom_int64_block_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
            [&args](size_t j, size_t k) { return args.q8_h1_block(j, k / QK8_0)->qs[k % QK8_0]; }, Y_com);
    if (!weights)
        return;
    accumulate_to_ycom_int64_block_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, Y_com);
}

void accumulate_single_row_to_ycom_int64_block(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (plan.route == DecWeightRoute::Q8HP1)
        return accumulate_single_row_to_ycom_int64_block_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
            [&args](size_t j, size_t k) { return args.q8_hp1_block(j, k / QK8_HP)->qs[k % QK8_HP]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8HP2)
        return accumulate_single_row_to_ycom_int64_block_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
            [&args](size_t j, size_t k) { return args.q8_hp2_block(j, k / QK8_HP)->qs[k % QK8_HP]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8H2)
        return accumulate_single_row_to_ycom_int64_block_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
            [&args](size_t j, size_t k) { return args.q8_h2_block(j, k / QK8_H2)->qs[k % QK8_H2]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8H1 && plan.native_weight_blocks)
        return accumulate_single_row_to_ycom_int64_block_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
            [&args](size_t j, size_t k) { return args.q8_h1_block(j, k / QK8_0)->qs[k % QK8_0]; }, Y_com);
    if (!weights)
        return;
    accumulate_single_row_to_ycom_int64_block_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
        [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, Y_com);
}

void accumulate_to_ycom_int64_h1(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs, const std::pair<int, int32_t> *pairs,
    float *Y_com)
{
    if (!pairs || !Y_com || I == 0 || J == 0)
        return;

    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    const std::vector<size_t> active_blocks = h1_active_blocks(args, plan, unique_k);
    for_each_int64_j_tile(I, J, [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
    {
        for (size_t block : active_blocks)
        {
            std::fill(accumulator.begin(), accumulator.begin() + I * width, int64_t {0});
            for (int k : unique_k)
            {
                if (k < 0)
                    continue;
                const size_t k_index = static_cast<size_t>(k);
                if (k_index >= args.K || k_index / QK8_0 != block || k_index + 1 >= rk_offs.size())
                    continue;

                for (size_t p = rk_offs[k_index]; p < rk_offs[k_index + 1]; ++p)
                {
                    const int row = pairs[p].first;
                    if (row < 0 || static_cast<size_t>(row) >= I)
                        continue;

                    int64_t *output_acc = accumulator.data() + static_cast<size_t>(row) * width;
                    for (size_t t = 0; t < width; ++t)
                    {
                        const size_t j = jb + t;
                        const block_q8_h1 *native = plan.native_weight_blocks ? args.q8_h1_block(j, block) : nullptr;
                        const int8_t code = native ? native->qs[k_index % QK8_0] :
                            weights[j * plan.weight_stride + k_index];
                        const int64_t term = static_cast<int64_t>(pairs[p].second) * code;
#if DEC_VALIDATION
                        const __int128 checked_sum = static_cast<__int128>(output_acc[t]) + term;
                        if (checked_sum < std::numeric_limits<int64_t>::min() ||
                            checked_sum > std::numeric_limits<int64_t>::max())
                            std::abort();
#endif
                        output_acc[t] += term;
                    }
                }
            }

            for (size_t row = 0; row < I; ++row)
            {
                const float activation_scale = activation_scales ? activation_scales[row] : 1.0f;
                const int64_t *row_acc = accumulator.data() + row * width;
                float *output = Y_com + row * J + jb;
                for (size_t t = 0; t < width; ++t)
                {
                    const size_t j = jb + t;
                    const block_q8_h1 *native = plan.native_weight_blocks ? args.q8_h1_block(j, block) : nullptr;
                    const uint64_t offset = native ? native->R :
                        (args.stripe_J > 1 ? args.R_stripe[j / args.stripe_J] : args.R[j]);
                    const uint64_t c_eff = (native ? native->c_b : args.c_b[j * args.blocks_per_row + block]) + offset;
                    const float s_rf = native ? native->s_rf :
                        (args.stripe_J > 1 ? args.s_rf_stripe[j / args.stripe_J] : args.s_rf[j]);
                    output[t] += static_cast<float>(static_cast<double>(row_acc[t]) * c_eff * s_rf * activation_scale);
                }
            }
        }
    });
}

void accumulate_single_row_to_ycom_int64_h1(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta, float *Y_com)
{
    if (!Y_com || J == 0)
        return;

    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    const std::vector<size_t> active_blocks = h1_active_blocks(args, plan, unique_k);
    for_each_int64_j_tile(1, J, [&](size_t jb, size_t width, std::vector<int64_t> &accumulator)
    {
        for (size_t block : active_blocks)
        {
            std::fill(accumulator.begin(), accumulator.begin() + width, int64_t {0});
            for (int k : unique_k)
            {
                if (k < 0)
                    continue;
                const size_t k_index = static_cast<size_t>(k);
                if (k_index >= args.K || k_index >= delta.size() || k_index / QK8_0 != block)
                    continue;

                for (size_t t = 0; t < width; ++t)
                {
                    const size_t j = jb + t;
                    const block_q8_h1 *native = plan.native_weight_blocks ? args.q8_h1_block(j, block) : nullptr;
                    const int8_t code = native ? native->qs[k_index % QK8_0] :
                        weights[j * plan.weight_stride + k_index];
#if DEC_VALIDATION
                    const __int128 term = static_cast<__int128>(delta[k_index]) * code;
                    const __int128 checked_sum = static_cast<__int128>(accumulator[t]) + term;
                    if (term < std::numeric_limits<int64_t>::min() ||
                        term > std::numeric_limits<int64_t>::max() ||
                        checked_sum < std::numeric_limits<int64_t>::min() ||
                        checked_sum > std::numeric_limits<int64_t>::max())
                        std::abort();
#endif
                    accumulator[t] += delta[k_index] * code;
                }
            }

            const float activation_scale = activation_scales ? activation_scales[0] : 1.0f;
            for (size_t t = 0; t < width; ++t)
            {
                const size_t j = jb + t;
                const block_q8_h1 *native = plan.native_weight_blocks ? args.q8_h1_block(j, block) : nullptr;
                const uint64_t offset = native ? native->R :
                    (args.stripe_J > 1 ? args.R_stripe[j / args.stripe_J] : args.R[j]);
                const uint64_t c_eff = (native ? native->c_b : args.c_b[j * args.blocks_per_row + block]) + offset;
                const float s_rf = native ? native->s_rf :
                    (args.stripe_J > 1 ? args.s_rf_stripe[j / args.stripe_J] : args.s_rf[j]);
                Y_com[j] += static_cast<float>(static_cast<double>(accumulator[t]) * c_eff * s_rf * activation_scale);
            }
        }
    });
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
        {
            for (size_t j = 0; j < J; ++j)
                dst[j] += src[j];
        }
        else
        {
            for (size_t j = 0; j < J; ++j)
                dst[j * stride_col] += src[j];
        }
    }
}
} // namespace ggml::gemmini::quants::dec
