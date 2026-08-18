#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

#include "../../../ggml-gemmini-args.h"
#include "../../../residual/rmd/rmd-builder.hpp"
#include "../../../residual/rmd/rmd-compose.hpp"
#include "../../common/tensor_util.hpp"
#include "stripe.hpp"

#include <gemmini/layer.hpp>
#include <gemmini/log.hpp>

#ifndef GGML_GEMMINI_EXSIA_SIGMA
#define GGML_GEMMINI_EXSIA_SIGMA 2
#endif

namespace ggml::gemmini::quants::act::stripe
{
namespace
{
    bool checked_mul_size(size_t lhs, size_t rhs, size_t &out)
    {
        if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
            return false;

        out = lhs * rhs;
        return true;
    }

    struct BitMask
    {
        size_t rows = 0;
        size_t cols = 0;
        std::vector<uint64_t> words;

        bool resize(size_t row_count, size_t col_count)
        {
            size_t bit_count = 0;
            if (!checked_mul_size(row_count, col_count, bit_count))
            {
                rows = 0;
                cols = 0;
                words.clear();
                return false;
            }

            rows = row_count;
            cols = col_count;
            words.assign((bit_count + 63) / 64, 0);
            return true;
        }

        void mark_outlier(size_t row, size_t col)
        {
            if (row >= rows || col >= cols) return;

            const size_t idx = row * cols + col;
            words[idx / 64] |= uint64_t(1) << (idx % 64);
        }

        bool is_marked(size_t row, size_t col) const
        {
            if (row >= rows || col >= cols) return false;

            const size_t idx = row * cols + col;
            return (words[idx / 64] & (uint64_t(1) << (idx % 64))) != 0;
        }
    };

    struct StripeStats
    {
        double mean = 0.0;
        double sigma = 0.0;
        double max_abs = 0.0;
        size_t count = 0;
    };

    bool compute_stripe_stats(
        const float *src_data,
        const ggml_gemmini_args_t &args,
        size_t row_start,
        size_t row_end,
        StripeStats &stats)
    {
        if (!src_data || row_start >= row_end || row_end > args.I)
            return false;

        double sum = 0.0;
        double sum_sq = 0.0;

        for (size_t row = row_start; row < row_end; ++row)
        {
            for (size_t col = 0; col < args.K; ++col)
            {
                const float value = src_data[row * args.K + col];
                if (!std::isfinite(value)) continue;

                const double x = static_cast<double>(value);
                sum += x;
                sum_sq += x * x;
                stats.max_abs = std::max(stats.max_abs, std::fabs(x));
                ++stats.count;
            }
        }

        if (stats.count == 0) return true;

        stats.mean = sum / static_cast<double>(stats.count);
        const double variance = std::max(0.0, sum_sq / static_cast<double>(stats.count) - stats.mean * stats.mean);
        stats.sigma = std::sqrt(variance);
        return true;
    }

    bool mark_outliers_sigma(
        const float *src_data,
        const ggml_gemmini_args_t &args,
        size_t row_start,
        size_t row_end,
        const StripeStats &stats,
        BitMask &mask,
        StripeStats &inlier_stats)
    {
        if (!src_data || row_start >= row_end || row_end > args.I || mask.rows < args.I || mask.cols < args.K)
            return false;

        for (size_t row = row_start; row < row_end; ++row)
        {
            for (size_t col = 0; col < args.K; ++col)
            {
                const float value = src_data[row * args.K + col];
                if (!std::isfinite(value))
                {
                    mask.mark_outlier(row, col);
                    continue;
                }

                const double x = static_cast<double>(value);
                const double z_score = stats.sigma == 0.0 ? 0.0 : (x - stats.mean) / stats.sigma;
                if (std::fabs(z_score) > GGML_GEMMINI_EXSIA_SIGMA)
                {
                    mask.mark_outlier(row, col);
                    continue;
                }

                inlier_stats.max_abs = std::max(inlier_stats.max_abs, std::fabs(x));
                ++inlier_stats.count;
            }
        }

        return true;
    }

    int32_t quantize_to_i32(float value, float scale)
    {
        if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0f)
            return 0;

        const double scaled =
            static_cast<double>(value) / static_cast<double>(scale);
        if (!std::isfinite(scaled))
            return scaled < 0.0 ? std::numeric_limits<int32_t>::min()
                                : std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();

        return static_cast<int32_t>(std::lrint(scaled));
    }

    int8_t clip_to_i8(int32_t value)
    {
        const int32_t clipped = value > 127 ? 127 : (value < -128 ? -128 : value);
        return static_cast<int8_t>(clipped);
    }

    bool set_scale(size_t stripe, const StripeStats &stats, Meta &meta)
    {
        if (stripe >= meta.scales.size()) return false;
        if (stats.count == 0 || stats.max_abs == 0.0)
        {
            meta.scales[stripe] = 1.0f;
            return true;
        }

        const float scale = static_cast<float>(stats.max_abs / 127.0);
        meta.scales[stripe] = std::isfinite(scale) && scale > 0.0f ? scale : 1.0f;
        return std::isfinite(meta.scales[stripe]) && meta.scales[stripe] > 0.0f;
    }
}

void set_config(Meta &meta)
{
    (void)meta;
}

bool quantize(const ggml_tensor *src, ggml_gemmini_args_t &args)
{
    int8_t *dst = reinterpret_cast<int8_t *>(args.A);
    if (!src || src->type != GGML_TYPE_F32 || !dst || args.I == 0 || args.K == 0)
        return false;

    auto *meta = std::get_if<Meta>(&args.act_quant.storage());
    if (!meta)
        return false;

    const float *src_data = ggml::gemmini::activation_data(src);
    if (!src_data)
        return false;

    //cpu fallback이면 args.I, else determined by gemmini tile
    size_t rows_per_stripe = args.I;
    if (args.tile_I > 0 && !checked_mul_size(args.tile_I, DIM, rows_per_stripe))
        return false;
    if (rows_per_stripe == 0)
        return false;
    const size_t num_stripes = (args.I - 1) / rows_per_stripe + 1;

    meta->scales.assign(num_stripes, 1.0f);
    meta->rmd_packets.clear();
    meta->direct_residuals.clear();

#if GGML_GEMMINI_ENABLE_RMD
    ggml::gemmini::residual::TimedResidualCapture residual_capture(args.residual_route);
    residual_capture.reset(0, 0, args.I, args.K, args.J);
    BitMask outliers;
    if (!outliers.resize(args.I, args.K))
        return false;
#endif

    for (size_t stripe = 0; stripe < num_stripes; stripe++)
    {
        const size_t row_start = stripe * rows_per_stripe;
        const size_t row_end = std::min(row_start + rows_per_stripe, args.I);

        StripeStats stripe_stats{};
        if (!compute_stripe_stats(src_data, args, row_start, row_end, stripe_stats))
            return false;

#if GGML_GEMMINI_ENABLE_RMD
        StripeStats inlier_stats{};
        if (!mark_outliers_sigma(src_data, args, row_start, row_end, stripe_stats, outliers, inlier_stats))
            return false;
        const StripeStats &scale_stats = inlier_stats.count != 0 ? inlier_stats : stripe_stats;
#else
        const StripeStats &scale_stats = stripe_stats;
#endif

        if (!set_scale(stripe, scale_stats, *meta))
            return false;
        const float scale = meta->scales[stripe];

        for (size_t row = row_start; row < row_end; ++row)
        {
            for (size_t col = 0; col < args.K; ++col)
            {
                const size_t idx = row * args.K + col;
                const int32_t q32 = quantize_to_i32(src_data[idx], scale);
                const int8_t q8 = clip_to_i8(q32);
                dst[idx] = q8;

#if GGML_GEMMINI_ENABLE_RMD
                const int64_t wide_residual = static_cast<int64_t>(q32) - static_cast<int64_t>(q8);
                if (wide_residual < std::numeric_limits<int32_t>::min() ||
                    wide_residual > std::numeric_limits<int32_t>::max())
                    return false;
                const int32_t residual = static_cast<int32_t>(wide_residual);
                if (outliers.is_marked(row, col) && residual != 0 &&
                    !residual_capture.add_residual(row, col, residual))
                    return false;
#endif

            }
        }
    }

#if GGML_GEMMINI_ENABLE_RMD
    const auto payload = residual_capture.finish();
    if (payload.packet)
        meta->rmd_packets.push_back(payload.packet);
    if (payload.direct)
        meta->direct_residuals.push_back(payload.direct);
    if (residual_capture.status() != ggml::gemmini::rmd::RmdStatus::success)
        return false;
#endif

    return true;
}

bool dequantize_activation(
    float *dst,
    size_t dst_row_stride,
    size_t dst_col_stride,
    size_t rows,
    size_t cols,
    const ggml_gemmini_args_t &args)
{
    const int8_t *src = reinterpret_cast<const int8_t *>(args.A);
    if (!src || !dst || args.I == 0 || args.K == 0 || dst_row_stride == 0 || dst_col_stride == 0 || rows == 0 || cols == 0)
        return false;

    const auto *meta_ptr = std::get_if<Meta>(&args.act_quant.storage());
    if (!meta_ptr)
        return false;
    const Meta &meta = *meta_ptr;

    if (args.sA != 0 && args.sA != args.K)
        return false;

    size_t rows_per_stripe = args.I;
    if (args.tile_I > 0 && !checked_mul_size(args.tile_I, DIM, rows_per_stripe))
        return false;
    if (rows_per_stripe == 0)
        return false;

    const size_t expected_stripes = (args.I - 1) / rows_per_stripe + 1;
    if (meta.scales.size() != expected_stripes)
        return false;

    const size_t row_count = std::min(rows, args.I);
    const size_t col_count = std::min(cols, args.K);
    const size_t max_size = std::numeric_limits<size_t>::max();
    const size_t src_row_stride = args.K;

#if GGML_GEMMINI_ENABLE_RMD
    size_t residual_count = 0;
    if (!checked_mul_size(row_count, col_count, residual_count))
        return false;
    (void) residual_count;

    std::vector<int32_t> residuals;
    ggml::gemmini::rmd::expand_packets_to_plane(
        meta.rmd_packets, row_count, col_count, residuals);
#endif

    for (size_t row = 0; row < row_count; ++row)
    {
        const size_t stripe_idx = row / rows_per_stripe;
        const float scale = meta.scales[stripe_idx];
        if (!std::isfinite(scale) || scale <= 0.0f) return false;

        for (size_t col = 0; col < col_count; ++col)
        {
            if ((row != 0 && src_row_stride > max_size / row) ||
                (row != 0 && dst_row_stride > max_size / row) ||
                (col != 0 && dst_col_stride > max_size / col))
                return false;

            const size_t src_row_offset = row * src_row_stride;
            if (src_row_offset > max_size - col)
                return false;

            const size_t dst_row_offset = row * dst_row_stride;
            const size_t dst_col_offset = col * dst_col_stride;
            if (dst_row_offset > max_size - dst_col_offset)
                return false;

            const size_t src_idx = src_row_offset + col;
            const size_t dst_idx = dst_row_offset + dst_col_offset;

            int32_t q = static_cast<int32_t>(src[src_idx]);
#if GGML_GEMMINI_ENABLE_RMD
            q += residuals[row * col_count + col];
#endif
            dst[dst_idx] = static_cast<float>(q) * scale;
        }
    }

    return true;
}

}
