#include "dec_kernel.hpp"
#include "../../ggml-gemmini-args.h"

#include <algorithm>
#include <cstdlib>

#if defined(GGML_GEMMINI_HAS_OPENMP)
#include <omp.h>
#endif

namespace ggml::gemmini::quants::dec { namespace
{
    constexpr size_t kBlockedJWidth = 128;
    constexpr size_t kDecodeJWidth = 64;

    bool is_q80_r_weight_args(const ggml_gemmini_args_t &args)
    {
        return args.B &&
               !args.B_scales &&
               args.c_b &&
               ((args.stripe_J > 1) || (args.s_rf && args.R)) &&
               args.blocks_per_row > 0;
    }

    WeightLayout resolve_weight_layout(const ggml_gemmini_args_t &args)
    {
        if (is_q80_r_weight_args(args) || args.transpose_B)
        {
            return WeightLayout::JxK_ColMajor;
        }

        return WeightLayout::KxJ_RowMajor;
    }

    size_t resolve_weight_stride_elems(const ggml_gemmini_args_t &args)
    {
        if (is_q80_r_weight_args(args))
        {
            return args.K;
        }

        const size_t fallback_stride = args.transpose_B ? args.K : args.J;
        return args.sB ? args.sB : fallback_stride;
    }

    size_t resolve_out_stride_row(const ggml_gemmini_args_t &args)
    {
        return args.stride_f_out ? args.stride_f_out : args.J;
    }

    size_t resolve_out_stride_col(const ggml_gemmini_args_t &args)
    {
        return args.col_stride_f_out ? args.col_stride_f_out : 1;
    }

    inline int resolve_dec_threads(size_t block_count)
    {
        int dec_threads = std::max(1, static_cast<int>(block_count));

#if defined(GGML_GEMMINI_HAS_OPENMP)
        dec_threads = std::min(dec_threads, omp_get_max_threads());
#endif

        if (const char *env = std::getenv("DEC_THREADS"))
        {
            char *end = nullptr;
            const long parsed = std::strtol(env, &end, 10);
            if (end != env && end && *end == '\0' && parsed > 0)
            {
                dec_threads = static_cast<int>(parsed);
            }
        }

        return std::max(1, dec_threads);
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
        {
            dst[j] += delta * Wk_f[j];
        }
    }

    inline void accumulate_row_simple(float *dst, const float *Wk_f, float delta, size_t J)
    {
        for (size_t j = 0; j < J; ++j)
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
        const float *scale_base,
        size_t block_size_k,
        size_t blocks_k,
        size_t scale_rows,
        size_t I,
        size_t J,
        const std::vector<int> &unique_k,
        const std::vector<size_t> &rk_offs,
        const std::pair<int, float> *rk_pairs,
        float *Y_com,
        std::vector<float> &y_block)
    {
        const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
        if (!weights)
            return;

        const size_t weight_stride = resolve_weight_stride_elems(args);
        const size_t block_width = std::min(kBlockedJWidth, J - jb);
        std::fill(y_block.begin(), y_block.begin() + I * block_width, 0.0f);

        for (size_t t = 0; t < block_width; ++t)
        {
            const size_t j = jb + t;
            const int8_t *weight_row = weights + j * weight_stride;
            const float *scale_row = (scale_base && j < scale_rows) ? scale_base + j * blocks_k : nullptr;

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
                if (scale_row && block_size_k > 0)
                {
                    const size_t blk = k_sz / block_size_k;
                    if (blk < blocks_k)
                        weight *= scale_row[blk];
                }

                for (size_t p = beg; p < end; ++p)
                {
                    const int r = rk_pairs[p].first;
                    if (r < 0 || static_cast<size_t>(r) >= I)
                        continue;

                    y_block[static_cast<size_t>(r) * block_width + t] += rk_pairs[p].second * weight;
                }
            }
        }

        flush_y_block(y_block.data(), I, block_width, J, jb, Y_com);
    }

    inline void accumulate_single_row_j_block(
        size_t jb,
        const ggml_gemmini_args_t &args,
        const float *scale_base,
        size_t block_size_k,
        size_t blocks_k,
        size_t scale_rows,
        size_t J,
        const std::vector<int> &unique_k,
        const std::vector<float> &delta_by_k,
        float *Y_com,
        std::vector<float> &y_block)
    {
        const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
        if (!weights)
            return;

        const size_t weight_stride = resolve_weight_stride_elems(args);
        const size_t block_width = std::min(kDecodeJWidth, J - jb);
        std::fill(y_block.begin(), y_block.begin() + block_width, 0.0f);

        for (size_t t = 0; t < block_width; ++t)
        {
            const size_t j = jb + t;
            const int8_t *weight_row = weights + j * weight_stride;
            const float *scale_row = (scale_base && j < scale_rows) ? scale_base + j * blocks_k : nullptr;

            for (int k : unique_k)
            {
                if (k < 0)
                    continue;

                const size_t k_sz = static_cast<size_t>(k);
                if (k_sz >= args.K || k_sz >= delta_by_k.size())
                    continue;

                const float delta = delta_by_k[k_sz];
                if (delta == 0.0f)
                    continue;

                float weight = static_cast<float>(weight_row[k_sz]);
                if (scale_row && block_size_k > 0)
                {
                    const size_t blk = k_sz / block_size_k;
                    if (blk < blocks_k)
                        weight *= scale_row[blk];
                }

                y_block[t] += delta * weight;
            }
        }

        float *dst = Y_com + jb;
        for (size_t t = 0; t < block_width; ++t)
            dst[t] += y_block[t];
    }
}

void accumulate_to_ycom_jmajor_blocked(
    const ggml_gemmini_args_t &args,
    const float *weight_scales,
    size_t scale_rows,
    size_t blocks_k,
    size_t block_size_k,
    size_t I,
    size_t J,
    const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs,
    const std::pair<int, float> *rk_pairs,
    float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights || !rk_pairs || !Y_com || I == 0 || J == 0)
        return;

    const size_t weight_stride = resolve_weight_stride_elems(args);
    if (resolve_weight_layout(args) != WeightLayout::JxK_ColMajor || weight_stride < args.K)
        return;

    const size_t block_count = (J + kBlockedJWidth - 1) / kBlockedJWidth;

#if defined(GGML_GEMMINI_HAS_OPENMP)
    const int dec_threads = resolve_dec_threads(block_count);
#pragma omp parallel num_threads(dec_threads)
    {
        std::vector<float> y_block(I * kBlockedJWidth, 0.0f);
#pragma omp for schedule(static)
        for (ptrdiff_t jb_idx = 0; jb_idx < static_cast<ptrdiff_t>(block_count); ++jb_idx)
        {
            const size_t jb = static_cast<size_t>(jb_idx) * kBlockedJWidth;
            accumulate_j_block(jb, args, weight_scales, block_size_k, blocks_k, scale_rows, I, J, unique_k, rk_offs, rk_pairs, Y_com, y_block);
        }
    }
#else
    std::vector<float> y_block(I * kBlockedJWidth, 0.0f);
    for (size_t jb_idx = 0; jb_idx < block_count; ++jb_idx)
    {
        const size_t jb = jb_idx * kBlockedJWidth;
        accumulate_j_block(jb, args, weight_scales, block_size_k, blocks_k, scale_rows, I, J, unique_k, rk_offs, rk_pairs, Y_com, y_block);
    }
#endif
}

void accumulate_single_row_to_ycom_jmajor_blocked(
    const ggml_gemmini_args_t &args,
    const float *weight_scales,
    size_t scale_rows,
    size_t blocks_k,
    size_t block_size_k,
    size_t J,
    const std::vector<int> &unique_k,
    const std::vector<float> &delta_by_k,
    float *Y_com)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    if (!weights || !Y_com || J == 0)
        return;

    const size_t weight_stride = resolve_weight_stride_elems(args);
    if (resolve_weight_layout(args) != WeightLayout::JxK_ColMajor || weight_stride < args.K)
        return;

    const size_t block_count = (J + kDecodeJWidth - 1) / kDecodeJWidth;

    std::vector<float> y_block(kDecodeJWidth, 0.0f);
    for (size_t jb_idx = 0; jb_idx < block_count; ++jb_idx)
    {
        const size_t jb = jb_idx * kDecodeJWidth;
        accumulate_single_row_j_block(jb, args, weight_scales, block_size_k, blocks_k, scale_rows, J, unique_k, delta_by_k, Y_com, y_block);
    }
}

void accumulate_to_output(
    const float *Wk_f,
    size_t J,
    size_t rk_beg,
    size_t rk_end,
    const std::pair<int, float> *rk_pairs,
    const ggml_gemmini_args_t &args,
    bool unroll8)
{
    float *out_data = args.f_out;
    if (!Wk_f || !rk_pairs || !out_data || J == 0)
        return;

    const size_t stride_row = resolve_out_stride_row(args);
    const size_t stride_col = resolve_out_stride_col(args);

    for (size_t t = rk_beg; t < rk_end; ++t)
    {
        const int r = rk_pairs[t].first;
        const float d = rk_pairs[t].second;
        if (r < 0 || static_cast<size_t>(r) >= args.I)
            continue;

        if (stride_col == 1)
        {
            float *Yr = out_data + static_cast<size_t>(r) * stride_row;
            if (unroll8)
                accumulate_row_unrolled(Yr, Wk_f, d, J);
            else
                accumulate_row_simple(Yr, Wk_f, d, J);
        }
        else
        {
            float *row_base = out_data + static_cast<size_t>(r) * stride_row;
            for (size_t j = 0; j < J; ++j)
                row_base[j * stride_col] += d * Wk_f[j];
        }
    }
}

void accumulate_to_ycom(
    const float *Wk_f,
    size_t J,
    size_t rk_beg,
    size_t rk_end,
    const std::pair<int, float> *rk_pairs,
    float *Y_com,
    bool unroll8)
{
    if (!Wk_f || !rk_pairs || !Y_com || J == 0)
        return;

    for (size_t t = rk_beg; t < rk_end; ++t)
    {
        const int r = rk_pairs[t].first;
        const float d = rk_pairs[t].second;
        if (r < 0)
            continue;

        float *Yr = Y_com + static_cast<size_t>(r) * J;
        if (unroll8)
            accumulate_row_unrolled(Yr, Wk_f, d, J);
        else
            accumulate_row_simple(Yr, Wk_f, d, J);
    }
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
