#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

#include "../../../ggml-gemmini-args.h"
#include "../../common/tensor_util.hpp"
#include "stripe.hpp"

#include <gemmini/layer.hpp>
#include <gemmini/log.hpp>

namespace ggml::gemmini::quants::act::stripe
{

namespace {

bool checked_mul_size(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }

    out = lhs * rhs;
    return true;
}

bool checked_add_size(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        return false;
    }

    out = lhs + rhs;
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
        if (!checked_mul_size(row_count, col_count, bit_count)) {
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
        if (row >= rows || col >= cols) {
            return;
        }

        const size_t idx = row * cols + col;
        words[idx / 64] |= uint64_t(1) << (idx % 64);
    }

    bool is_marked(size_t row, size_t col) const
    {
        if (row >= rows || col >= cols) {
            return false;
        }

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
    if (!src_data || row_start >= row_end || row_end > args.I) {
        return false;
    }

    double sum = 0.0;
    double sum_sq = 0.0;

    for (size_t row = row_start; row < row_end; ++row) {
        for (size_t col = 0; col < args.K; ++col) {
            const float value = src_data[row * args.K + col];
            if (!std::isfinite(value)) {
                continue;
            }

            const double x = static_cast<double>(value);
            sum += x;
            sum_sq += x * x;
            stats.max_abs = std::max(stats.max_abs, std::fabs(x));
            ++stats.count;
        }
    }

    if (stats.count == 0) {
        return true;
    }

    stats.mean = sum / static_cast<double>(stats.count);
    const double variance = std::max(0.0, sum_sq / static_cast<double>(stats.count) - stats.mean * stats.mean);
    stats.sigma = std::sqrt(variance);
    return true;
}

bool mark_outliers_3sigma(
    const float *src_data,
    const ggml_gemmini_args_t &args,
    size_t row_start,
    size_t row_end,
    const StripeStats &stats,
    BitMask &mask,
    StripeStats &inlier_stats)
{
    if (!src_data || row_start >= row_end || row_end > args.I || mask.rows < args.I || mask.cols < args.K) {
        return false;
    }

    for (size_t row = row_start; row < row_end; ++row) {
        for (size_t col = 0; col < args.K; ++col) {
            const float value = src_data[row * args.K + col];
            if (!std::isfinite(value)) {
                mask.mark_outlier(row, col);
                continue;
            }

            const double x = static_cast<double>(value);
            const double z_score = stats.sigma == 0.0 ? 0.0 : (x - stats.mean) / stats.sigma;
            if (std::fabs(z_score) > 3.0) {
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
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0f) {
        return 0;
    }

    const double scaled = std::round(static_cast<double>(value) / static_cast<double>(scale));
    if (!std::isfinite(scaled)) {
        return scaled < 0.0
                   ? std::numeric_limits<int32_t>::min()
                   : std::numeric_limits<int32_t>::max();
    }

    const double min_i32 = static_cast<double>(std::numeric_limits<int32_t>::min());
    const double max_i32 = static_cast<double>(std::numeric_limits<int32_t>::max());
    if (scaled <= min_i32) {
        return std::numeric_limits<int32_t>::min();
    }
    if (scaled >= max_i32) {
        return std::numeric_limits<int32_t>::max();
    }

    return static_cast<int32_t>(scaled);
}

int8_t clip_to_i8(int32_t value)
{
    const int32_t clipped = value > 127 ? 127 : (value < -128 ? -128 : value);
    return static_cast<int8_t>(clipped);
}

bool set_scale(size_t stripe, const StripeStats &stats, Meta &meta)
{
    if (stripe >= meta.scales.size()) {
        return false;
    }
    if (stats.count == 0 || stats.max_abs == 0.0) {
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
    if (!src || src->type != GGML_TYPE_F32 || !dst || args.I == 0 || args.K == 0) {
        return false;
    }

    auto *meta = std::get_if<Meta>(&args.act_quant.storage());
    if (!meta) {
        return false;
    }

    const float *src_data = ggml::gemmini::activation_data(src);
    if (!src_data) {
        return false;
    }

    size_t rows_per_stripe = 0;
    size_t num_stripes = 0;
    if (!resolve_rows_per_stripe(args, rows_per_stripe) ||
        !resolve_num_stripes(args, rows_per_stripe, num_stripes)) {
        return false;
    }

    meta->scales.assign(num_stripes, 1.0f);
    meta->outliers.clear();

    (void)clip_to_i8;
    (void)compute_stripe_stats;
    (void)mark_outliers_3sigma;
    (void)quantize_to_i32;
    (void)set_scale;

    return false;
}

bool dequantize_activation(
    float *dst,
    size_t dst_row_stride,
    size_t dst_col_stride,
    size_t rows,
    size_t cols,
    const ggml_gemmini_args_t &args)
{
    (void)dst;
    (void)dst_row_stride;
    (void)dst_col_stride;
    (void)rows;
    (void)cols;
    (void)args;
    return false;
}

}
