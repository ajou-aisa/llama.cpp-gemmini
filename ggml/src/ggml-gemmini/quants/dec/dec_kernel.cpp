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

namespace
{
    constexpr size_t kBlockedJWidth = 128;
    constexpr size_t kDecodeJWidth = 64;

    size_t resolve_out_stride_row(const ggml_gemmini_args_t &args)
    {
        return args.stride_f_out ? args.stride_f_out : args.J;
    }

    size_t resolve_out_stride_col(const ggml_gemmini_args_t &args)
    {
        return args.col_stride_f_out ? args.col_stride_f_out : 1;
    }

    inline void accumulate_row_unrolled(float *dst, const float *Wk_f, float delta, size_t J)
    {
        size_t j = 0;
        for (; j + 7 < J; j += 8)
        {
            dst[j + 0] += delta * Wk_f[j + 0];
            dst[j + 1] += delta * Wk_f[j + 1];
            dst[j + 2] += delta * Wk_f[j + 2];
            dst[j + 3] += delta * Wk_f[j + 3];
            dst[j + 4] += delta * Wk_f[j + 4];
            dst[j + 5] += delta * Wk_f[j + 5];
            dst[j + 6] += delta * Wk_f[j + 6];
            dst[j + 7] += delta * Wk_f[j + 7];
        }

        for (; j < J; ++j)
            dst[j] += delta * Wk_f[j];
    }

    inline void flush_y_block(const float *block, size_t I, size_t block_width, size_t J, size_t jb, float *Y_com)
    {
        for (size_t r = 0; r < I; ++r)
        {
            const float *src = block + r * block_width;
            float *dst = Y_com + r * J + jb;
            for (size_t t = 0; t < block_width; ++t)
                dst[t] += src[t];
        }
    }

    inline void accumulate_j_block(
        size_t jb,
        const ggml_gemmini_args_t &args,
        const DecRoutePlan &plan,
        size_t I,
        size_t J,
        const float *activation_scales,
        const std::vector<int> &unique_k,
        const std::vector<size_t> &rk_offs,
        const std::pair<int, int32_t> *rk_pairs,
        float *Y_com,
        std::vector<float> &y_block)
    {
        const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
        if (!weights)
            return;

        const size_t block_width = std::min(kBlockedJWidth, J - jb);
        std::fill(y_block.begin(), y_block.begin() + I * block_width, 0.0f);

        for (size_t t = 0; t < block_width; ++t)
        {
            const size_t j = jb + t;
            const int8_t *weight_row = weights + j * plan.weight_stride;

            for (int k : unique_k)
            {
                if (k < 0)
                    continue;

                const size_t k_sz = static_cast<size_t>(k);
                if (k_sz >= args.K || k_sz + 1 >= rk_offs.size())
                    continue;

                const size_t beg = rk_offs[k_sz];
                const size_t end = rk_offs[k_sz + 1];
                if (beg >= end)
                    continue;

                float weight = static_cast<float>(weight_row[k_sz]);
                weight *= dec_route_weight_scale(plan, args, j, k_sz / plan.scales.block_size);

                for (size_t p = beg; p < end; ++p)
                {
                    const int r = rk_pairs[p].first;
                    if (r < 0 || static_cast<size_t>(r) >= I)
                        continue;

                    const float activation_scale = activation_scales ? activation_scales[r] : 1.0f;
                    const float delta = static_cast<float>(rk_pairs[p].second) * activation_scale;
                    y_block[static_cast<size_t>(r) * block_width + t] += delta * weight;
                }
            }
        }

        flush_y_block(y_block.data(), I, block_width, J, jb, Y_com);
    }

    inline void accumulate_single_row_j_block(
        size_t jb,
        const ggml_gemmini_args_t &args,
        const DecRoutePlan &plan,
        size_t J,
        const float *activation_scales,
        const std::vector<int> &unique_k,
        const std::vector<int64_t> &delta_by_k,
        float *Y_com,
        std::vector<float> &y_block)
    {
        const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
        if (!weights)
            return;

        const size_t block_width = std::min(kDecodeJWidth, J - jb);
        std::fill(y_block.begin(), y_block.begin() + block_width, 0.0f);

        for (size_t t = 0; t < block_width; ++t)
        {
            const size_t j = jb + t;
            const int8_t *weight_row = weights + j * plan.weight_stride;

            for (int k : unique_k)
            {
                if (k < 0)
                    continue;

                const size_t k_sz = static_cast<size_t>(k);
                if (k_sz >= args.K || k_sz >= delta_by_k.size())
                    continue;

                const int64_t delta_i64 = delta_by_k[k_sz];
                if (delta_i64 == 0)
                    continue;

                const float activation_scale = activation_scales ? activation_scales[0] : 1.0f;
                const float delta = static_cast<float>(delta_i64) * activation_scale;

                float weight = static_cast<float>(weight_row[k_sz]);
                weight *= dec_route_weight_scale(plan, args, j, k_sz / plan.scales.block_size);

                y_block[t] += delta * weight;
            }
        }

        float *dst = Y_com + jb;
        for (size_t t = 0; t < block_width; ++t)
            dst[t] += y_block[t];
    }

    template <typename ScaleForColumn>
    void accumulate_to_ycom_int64_impl(
        const ggml_gemmini_args_t &args,
        const DecRoutePlan &plan,
        size_t I,
        size_t J,
        const float *activation_scales,
        const std::vector<int> &unique_k,
        const std::vector<size_t> &rk_offs,
        const std::pair<int, int32_t> *rk_pairs,
        std::vector<int64_t> &accumulator,
        ScaleForColumn scale_for_column,
        float *Y_com)
    {
        const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
        if (!weights || !rk_pairs || !Y_com || I == 0 || J == 0)
            return;

        accumulator.assign(I * J, int64_t {0});
        for (int k : unique_k)
        {
            if (k < 0)
                continue;
            const size_t k_sz = static_cast<size_t>(k);
            if (k_sz >= args.K || k_sz + 1 >= rk_offs.size())
                continue;

            const size_t begin = rk_offs[k_sz];
            const size_t end = rk_offs[k_sz + 1];
            const int8_t *row = plan.layout == WeightLayout::KxJ_RowMajor ?
                weights + k_sz * plan.weight_stride : nullptr;
            for (size_t p = begin; p < end; ++p)
            {
                const int r = rk_pairs[p].first;
                if (r < 0 || static_cast<size_t>(r) >= I)
                    continue;

                int64_t *accumulator_row = accumulator.data() + static_cast<size_t>(r) * J;
                const int64_t residual = rk_pairs[p].second;
                // int32 residual * int8 code is <= 2^38; INT64 supports at least 2^25 terms per output.
                if (row)
                {
                    for (size_t j = 0; j < J; ++j)
                        accumulator_row[j] += residual * row[j];
                }
                else
                {
                    for (size_t j = 0; j < J; ++j)
                        accumulator_row[j] += residual * weights[j * plan.weight_stride + k_sz];
                }
            }
        }

        for (size_t r = 0; r < I; ++r)
        {
            const float activation_scale = activation_scales ? activation_scales[r] : 1.0f;
            const int64_t *accumulator_row = accumulator.data() + r * J;
            float *output_row = Y_com + r * J;
            for (size_t j = 0; j < J; ++j)
                output_row[j] += static_cast<float>(
                    static_cast<double>(accumulator_row[j]) * activation_scale * scale_for_column(j));
        }
    }

    template <typename ScaleForColumn>
    void accumulate_single_row_to_ycom_int64_impl(
        const ggml_gemmini_args_t &args,
        const DecRoutePlan &plan,
        size_t J,
        const float *activation_scales,
        const std::vector<int> &unique_k,
        const std::vector<int64_t> &delta_by_k,
        std::vector<int64_t> &accumulator,
        ScaleForColumn scale_for_column,
        float *Y_com)
    {
        const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
        if (!weights || !Y_com || J == 0)
            return;

        accumulator.assign(J, int64_t {0});
        for (int k : unique_k)
        {
            if (k < 0)
                continue;
            const size_t k_sz = static_cast<size_t>(k);
            if (k_sz >= args.K || k_sz >= delta_by_k.size())
                continue;

            const int64_t residual = delta_by_k[k_sz];
            const int8_t *row = plan.layout == WeightLayout::KxJ_RowMajor ?
                weights + k_sz * plan.weight_stride : nullptr;
            if (row)
            {
                for (size_t j = 0; j < J; ++j)
                    accumulator[j] += residual * row[j];
            }
            else
            {
                for (size_t j = 0; j < J; ++j)
                    accumulator[j] += residual * weights[j * plan.weight_stride + k_sz];
            }
        }

        const float activation_scale = activation_scales ? activation_scales[0] : 1.0f;
        for (size_t j = 0; j < J; ++j)
            Y_com[j] += static_cast<float>(
                static_cast<double>(accumulator[j]) * activation_scale * scale_for_column(j));
    }

    template <typename CodeFor>
    void accumulate_to_ycom_int64_block_impl(
        const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
        const float *activation_scales, const std::vector<int> &unique_k,
        const std::vector<size_t> &rk_offs, const std::pair<int, int32_t> *rk_pairs,
        std::vector<int64_t> &accumulator, CodeFor code_for, float *Y_com)
    {
        if (!rk_pairs || !Y_com || I == 0 || J == 0)
            return;
        accumulator.assign(I * J, int64_t {0});
        for (size_t block = 0; block < plan.scales.cols; ++block)
        {
            std::fill(accumulator.begin(), accumulator.end(), int64_t {0});
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
                    int64_t *row = accumulator.data() + static_cast<size_t>(r) * J;
                    const int64_t residual = rk_pairs[p].second;
                    for (size_t j = 0; j < J; ++j)
                        row[j] += residual * code_for(j, k_sz);
                }
            }
            for (size_t r = 0; r < I; ++r)
            {
                const float activation_scale = activation_scales ? activation_scales[r] : 1.0f;
                const int64_t *row = accumulator.data() + r * J;
                float *output = Y_com + r * J;
                for (size_t j = 0; j < J; ++j)
                    output[j] += static_cast<float>(static_cast<double>(row[j]) * activation_scale *
                        dec_route_weight_scale(plan, args, j, block));
            }
        }
    }

    template <typename CodeFor>
    void accumulate_single_row_to_ycom_int64_block_impl(
        const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t J,
        const float *activation_scales, const std::vector<int> &unique_k,
        const std::vector<int64_t> &delta_by_k, std::vector<int64_t> &accumulator,
        CodeFor code_for, float *Y_com)
    {
        if (!Y_com || J == 0)
            return;
        accumulator.assign(J, int64_t {0});
        for (size_t block = 0; block < plan.scales.cols; ++block)
        {
            std::fill(accumulator.begin(), accumulator.end(), int64_t {0});
            for (int k : unique_k)
            {
                if (k < 0)
                    continue;
                const size_t k_sz = static_cast<size_t>(k);
                if (k_sz >= args.K || k_sz >= delta_by_k.size() || k_sz / plan.scales.block_size != block)
                    continue;
                const int64_t residual = delta_by_k[k_sz];
                for (size_t j = 0; j < J; ++j)
                    accumulator[j] += residual * code_for(j, k_sz);
            }
            const float activation_scale = activation_scales ? activation_scales[0] : 1.0f;
            for (size_t j = 0; j < J; ++j)
                Y_com[j] += static_cast<float>(static_cast<double>(accumulator[j]) * activation_scale *
                    dec_route_weight_scale(plan, args, j, block));
        }
    }
}

void accumulate_to_ycom_int64_scalar(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs, const std::pair<int, int32_t> *rk_pairs,
    std::vector<int64_t> &accumulator, float *Y_com)
{
    accumulate_to_ycom_int64_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
        accumulator, [scale = plan.scales.scalar](size_t) { return scale; }, Y_com);
}

void accumulate_single_row_to_ycom_int64_scalar(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k, std::vector<int64_t> &accumulator, float *Y_com)
{
    accumulate_single_row_to_ycom_int64_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
        accumulator, [scale = plan.scales.scalar](size_t) { return scale; }, Y_com);
}

void accumulate_to_ycom_int64_channel_direct(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs, const std::pair<int, int32_t> *rk_pairs,
    std::vector<int64_t> &accumulator, float *Y_com)
{
    accumulate_to_ycom_int64_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
        accumulator, [&args](size_t j) { return args.q8_channel_scale(j); }, Y_com);
}

void accumulate_single_row_to_ycom_int64_channel_direct(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k, std::vector<int64_t> &accumulator, float *Y_com)
{
    accumulate_single_row_to_ycom_int64_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
        accumulator, [&args](size_t j) { return args.q8_channel_scale(j); }, Y_com);
}

void accumulate_to_ycom_int64_channel_sidecar(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs, const std::pair<int, int32_t> *rk_pairs,
    std::vector<int64_t> &accumulator, float *Y_com)
{
    accumulate_to_ycom_int64_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
        accumulator, [scales = plan.scales.data](size_t j) { return scales[j]; }, Y_com);
}

void accumulate_single_row_to_ycom_int64_channel_sidecar(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k, std::vector<int64_t> &accumulator, float *Y_com)
{
    accumulate_single_row_to_ycom_int64_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
        accumulator, [scales = plan.scales.data](size_t j) { return scales[j]; }, Y_com);
}

void accumulate_to_ycom_int64_block(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs, const std::pair<int, int32_t> *rk_pairs,
    std::vector<int64_t> &accumulator, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (plan.route == DecWeightRoute::Q8HP1)
        return accumulate_to_ycom_int64_block_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
            accumulator, [&args](size_t j, size_t k) { return args.q8_hp1_block(j, k / QK8_HP)->qs[k % QK8_HP]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8HP2)
        return accumulate_to_ycom_int64_block_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
            accumulator, [&args](size_t j, size_t k) { return args.q8_hp2_block(j, k / QK8_HP)->qs[k % QK8_HP]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8H2)
        return accumulate_to_ycom_int64_block_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
            accumulator, [&args](size_t j, size_t k) { return args.q8_h2_block(j, k / QK8_H2)->qs[k % QK8_H2]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8H1 && plan.native_weight_blocks)
        return accumulate_to_ycom_int64_block_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
            accumulator, [&args](size_t j, size_t k) { return args.q8_h1_block(j, k / QK8_0)->qs[k % QK8_0]; }, Y_com);
    if (!weights)
        return;
    accumulate_to_ycom_int64_block_impl(args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs,
        accumulator, [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, Y_com);
}

void accumulate_single_row_to_ycom_int64_block(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k, std::vector<int64_t> &accumulator, float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (plan.route == DecWeightRoute::Q8HP1)
        return accumulate_single_row_to_ycom_int64_block_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
            accumulator, [&args](size_t j, size_t k) { return args.q8_hp1_block(j, k / QK8_HP)->qs[k % QK8_HP]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8HP2)
        return accumulate_single_row_to_ycom_int64_block_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
            accumulator, [&args](size_t j, size_t k) { return args.q8_hp2_block(j, k / QK8_HP)->qs[k % QK8_HP]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8H2)
        return accumulate_single_row_to_ycom_int64_block_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
            accumulator, [&args](size_t j, size_t k) { return args.q8_h2_block(j, k / QK8_H2)->qs[k % QK8_H2]; }, Y_com);
    if (plan.route == DecWeightRoute::Q8H1 && plan.native_weight_blocks)
        return accumulate_single_row_to_ycom_int64_block_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
            accumulator, [&args](size_t j, size_t k) { return args.q8_h1_block(j, k / QK8_0)->qs[k % QK8_0]; }, Y_com);
    if (!weights)
        return;
    accumulate_single_row_to_ycom_int64_block_impl(args, plan, J, activation_scales, unique_k, delta_by_k,
        accumulator, [weights, &plan](size_t j, size_t k) {
            return plan.layout == WeightLayout::KxJ_RowMajor ? weights[k * plan.weight_stride + j] :
                weights[j * plan.weight_stride + k];
        }, Y_com);
}

void accumulate_to_ycom_int64_h1(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t I, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs, const std::pair<int, int32_t> *pairs,
    std::vector<int64_t> &accumulator, float *Y_com)
{
    if (!pairs || !Y_com || I == 0 || J == 0)
        return;

    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    accumulator.resize(I * J);
    for (size_t block = 0; block < plan.scales.cols; ++block)
    {
        std::fill(accumulator.begin(), accumulator.end(), int64_t {0});
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

                int64_t *output_acc = accumulator.data() + static_cast<size_t>(row) * J;
                for (size_t j = 0; j < J; ++j)
                {
                    const block_q8_h1 *native = plan.native_weight_blocks ? args.q8_h1_block(j, block) : nullptr;
                    const int8_t code = native ? native->qs[k_index % QK8_0] :
                        weights[j * plan.weight_stride + k_index];
                    const uint64_t offset = native ? native->R :
                        (args.stripe_J > 1 ? args.R_stripe[j / args.stripe_J] : args.R[j]);
                    const uint64_t c_eff = (native ? native->c_b : args.c_b[j * args.blocks_per_row + block]) + offset;
                    const int64_t term = static_cast<int64_t>(pairs[p].second) * code * static_cast<int64_t>(c_eff);
#if DEC_VALIDATION
                    // One term is bounded by |INT32_MIN| * 128 * (UINT16_MAX + UINT8_MAX).
                    const __int128 checked_sum = static_cast<__int128>(output_acc[j]) + term;
                    if (checked_sum < std::numeric_limits<int64_t>::min() ||
                        checked_sum > std::numeric_limits<int64_t>::max())
                        std::abort();
#endif
                    output_acc[j] += term;
                }
            }
        }

        for (size_t row = 0; row < I; ++row)
        {
            const float activation_scale = activation_scales ? activation_scales[row] : 1.0f;
            for (size_t j = 0; j < J; ++j)
            {
                const block_q8_h1 *native = plan.native_weight_blocks ? args.q8_h1_block(j, block) : nullptr;
                const float s_rf = native ? native->s_rf :
                    (args.stripe_J > 1 ? args.s_rf_stripe[j / args.stripe_J] : args.s_rf[j]);
                Y_com[row * J + j] += static_cast<float>(
                    static_cast<double>(accumulator[row * J + j]) * s_rf * activation_scale);
            }
        }
    }
}

void accumulate_single_row_to_ycom_int64_h1(
    const ggml_gemmini_args_t &args, const DecRoutePlan &plan, size_t J,
    const float *activation_scales, const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta, std::vector<int64_t> &accumulator, float *Y_com)
{
    if (!Y_com || J == 0)
        return;

    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    accumulator.resize(J);
    for (size_t block = 0; block < plan.scales.cols; ++block)
    {
        std::fill(accumulator.begin(), accumulator.end(), int64_t {0});
        for (int k : unique_k)
        {
            if (k < 0)
                continue;
            const size_t k_index = static_cast<size_t>(k);
            if (k_index >= args.K || k_index >= delta.size() || k_index / QK8_0 != block)
                continue;

            for (size_t j = 0; j < J; ++j)
            {
                const block_q8_h1 *native = plan.native_weight_blocks ? args.q8_h1_block(j, block) : nullptr;
                const int8_t code = native ? native->qs[k_index % QK8_0] :
                    weights[j * plan.weight_stride + k_index];
                const uint64_t offset = native ? native->R :
                    (args.stripe_J > 1 ? args.R_stripe[j / args.stripe_J] : args.R[j]);
                const uint64_t c_eff = (native ? native->c_b : args.c_b[j * args.blocks_per_row + block]) + offset;
                const __int128 term = static_cast<__int128>(delta[k_index]) * code * c_eff;
#if DEC_VALIDATION
                const __int128 checked_sum = static_cast<__int128>(accumulator[j]) + term;
                if (term < std::numeric_limits<int64_t>::min() ||
                    term > std::numeric_limits<int64_t>::max() ||
                    checked_sum < std::numeric_limits<int64_t>::min() ||
                    checked_sum > std::numeric_limits<int64_t>::max())
                    std::abort();
#endif
                accumulator[j] += static_cast<int64_t>(term);
            }
        }

        const float activation_scale = activation_scales ? activation_scales[0] : 1.0f;
        for (size_t j = 0; j < J; ++j)
        {
            const block_q8_h1 *native = plan.native_weight_blocks ? args.q8_h1_block(j, block) : nullptr;
            const float s_rf = native ? native->s_rf :
                (args.stripe_J > 1 ? args.s_rf_stripe[j / args.stripe_J] : args.s_rf[j]);
            Y_com[j] += static_cast<float>(static_cast<double>(accumulator[j]) * s_rf * activation_scale);
        }
    }
}

void accumulate_to_ycom_jmajor_blocked(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs,
    const std::pair<int, int32_t> *rk_pairs,
    float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights || !rk_pairs || !Y_com || I == 0 || J == 0)
        return;

    if (plan.layout != WeightLayout::JxK_ColMajor || plan.weight_stride < args.K)
        return;

    const size_t block_count = (J + kBlockedJWidth - 1) / kBlockedJWidth;

#if defined(GGML_GEMMINI_HAS_OPENMP)
    const int dec_threads = resolve_dec_threads(block_count, omp_get_max_threads());
#pragma omp parallel num_threads(dec_threads)
    {
        std::vector<float> y_block(I * kBlockedJWidth, 0.0f);
#pragma omp for schedule(static)
        for (ptrdiff_t jb_idx = 0; jb_idx < static_cast<ptrdiff_t>(block_count); ++jb_idx)
        {
            const size_t jb = static_cast<size_t>(jb_idx) * kBlockedJWidth;
            accumulate_j_block(jb, args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs, Y_com, y_block);
        }
    }
#else
    std::vector<float> y_block(I * kBlockedJWidth, 0.0f);
    for (size_t jb_idx = 0; jb_idx < block_count; ++jb_idx)
    {
        const size_t jb = jb_idx * kBlockedJWidth;
        accumulate_j_block(jb, args, plan, I, J, activation_scales, unique_k, rk_offs, rk_pairs, Y_com, y_block);
    }
#endif
}

void accumulate_single_row_to_ycom_jmajor_blocked(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k,
    float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights || !Y_com || J == 0)
        return;

    if (plan.layout != WeightLayout::JxK_ColMajor || plan.weight_stride < args.K)
        return;

    const size_t block_count = (J + kDecodeJWidth - 1) / kDecodeJWidth;

    std::vector<float> y_block(kDecodeJWidth, 0.0f);
    for (size_t jb_idx = 0; jb_idx < block_count; ++jb_idx)
    {
        const size_t jb = jb_idx * kDecodeJWidth;
        accumulate_single_row_j_block(jb, args, plan, J, activation_scales, unique_k, delta_by_k, Y_com, y_block);
    }
}

void accumulate_to_ycom(
    const float *Wk_f,
    size_t J,
    size_t rk_beg,
    size_t rk_end,
    const std::pair<int, int32_t> *rk_pairs,
    const float *activation_scales,
    float *Y_com)
{
    if (!Wk_f || !rk_pairs || !Y_com || J == 0)
        return;

    for (size_t t = rk_beg; t < rk_end; ++t)
    {
        const int r = rk_pairs[t].first;
        if (r < 0)
            continue;

        const float activation_scale = activation_scales ? activation_scales[r] : 1.0f;
        const float d = static_cast<float>(rk_pairs[t].second) * activation_scale;

        float *Yr = Y_com + static_cast<size_t>(r) * J;
        accumulate_row_unrolled(Yr, Wk_f, d, J);
    }
}

void accumulate_single_row_delta_to_ycom(
    const float *Wk_f,
    size_t J,
    int64_t delta_i64,
    const float *activation_scales,
    float *Y_com)
{
    if (!Wk_f || !Y_com || J == 0 || delta_i64 == 0)
        return;

    const float activation_scale = activation_scales ? activation_scales[0] : 1.0f;
    const float d = static_cast<float>(delta_i64) * activation_scale;

    accumulate_row_unrolled(Y_com, Wk_f, d, J);
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
