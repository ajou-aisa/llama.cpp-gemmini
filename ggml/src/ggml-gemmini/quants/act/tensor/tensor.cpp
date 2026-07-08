#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

#include "../../../ggml-gemmini-args.h"
#include "../../common/tensor_util.hpp"
#include "tensor.hpp"

#include <gemmini/layer.hpp>
#include <gemmini/log.hpp>

namespace ggml::gemmini::quants::act::tensor
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
};

struct TensorStats
{
    double mean = 0.0;
    double sigma = 0.0;
    double max_abs = 0.0;
    size_t count = 0;
};

bool compute_tensor_stats(const float *src_data, const ggml_gemmini_args_t &args, TensorStats &stats)
{
    if (!src_data) {
        return false;
    }

    double t_sum = 0.0;
    double t_sum_sq = 0.0;
    size_t t_count = 0;
    for (size_t row = 0; row < args.I; ++row) {
        for (size_t col = 0; col < args.K; ++col) {
            const float value = src_data[row * args.K + col];
            if (!std::isfinite(value)) {
                continue;
            }

            const double x = static_cast<double>(value);
            t_sum += x;
            t_sum_sq += x * x;
            stats.max_abs = std::max(stats.max_abs, std::fabs(x));
            ++t_count;
        }
    }

    if (t_count == 0) {
        return false;
    }

    stats.count = t_count;
    stats.mean = t_sum / static_cast<double>(t_count);

    const double variance = std::max(0.0, t_sum_sq / static_cast<double>(t_count) - stats.mean * stats.mean);
    stats.sigma = std::sqrt(variance);
    return true;
}

bool mark_outliers_3sigma(
    const float *src_data,
    const ggml_gemmini_args_t &args,
    const TensorStats &stats,
    BitMask &mask,
    TensorStats &inlier_stats)
{
    if (!src_data || !mask.resize(args.I, args.K)) {
        return false;
    }

    for (size_t row = 0; row < args.I; ++row) {
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

    return static_cast<int32_t>(std::round(value / scale));
}

int8_t clip_to_i8(int32_t value)
{
    const int32_t clipped = value > 127 ? 127 : (value < -128 ? -128 : value);
    return static_cast<int8_t>(clipped);
}

bool set_scale(const TensorStats &stats, Meta &meta)
{
    if (stats.count == 0 || stats.max_abs == 0.0) {
        meta.scale = 1.0f;
        return true;
    }

    meta.scale = static_cast<float>(stats.max_abs / 127.0);
    return std::isfinite(meta.scale) && meta.scale > 0.0f;
}

}

// per-tensor에서는 의미 없음.
void set_config(Meta &meta)
{
    (void)meta;
}

bool quantize(const ggml_tensor *src, ggml_gemmini_args_t &args)
{
    // TODO: per-tensor quantization. 
    // Outlier는 옵션에 따라 처리. 
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

    const char *layer = ggml::gemmini::types::to_string(args.layer_type);

    #if ERROR_COMPENSATION
        TensorStats stats;
        TensorStats inlier_stats;
        BitMask outliers;
        if (!compute_tensor_stats(src_data, args, stats)) {
            return false;
        }
        if (!mark_outliers_3sigma(src_data, args, stats, outliers, inlier_stats)) {
            return false;
        }
        const TensorStats &scale_stats = inlier_stats.count != 0 ? inlier_stats : stats;
        if (!set_scale(scale_stats, *meta)) {
            return false;
        }
        meta->outliers.clear();
    #else
        TensorStats stats;
        if (!compute_tensor_stats(src_data, args, stats)) {
            return false;
        }
        if (!set_scale(stats, *meta)) {
            return false;
        }
    #endif

    for (size_t row = 0; row < args.I; ++row) {
        for (size_t col = 0; col < args.K; ++col) {
            const size_t idx = row * args.K + col;
            const int32_t q32 = quantize_to_i32(src_data[idx], meta->scale);
            const int8_t q8 = clip_to_i8(q32);
            dst[idx] = q8;

            #if ERROR_COMPENSATION
                const int32_t residual = q32 - static_cast<int32_t>(q8);
                if (residual != 0) {
                    meta->outliers.push_back({
                        static_cast<int>(row),
                        static_cast<int>(col),
                        residual,
                    });
                }
            #endif
        }
    }

    #if ERROR_COMPENSATION
        ggml::gemmini::log::debug(layer,
            "[quantize_tensor] I=%zu K=%zu scale=%.9g mean=%.6g sigma=%.6g max_abs=%.6g count=%zu "
            "inlier_max_abs=%.6g inlier_count=%zu residual_outliers=%zu",
            args.I, args.K,
            static_cast<double>(meta->scale),
            stats.mean, stats.sigma, stats.max_abs, stats.count,
            inlier_stats.max_abs, inlier_stats.count,
            meta->outliers.size());
    #else
        ggml::gemmini::log::debug(layer,
            "[quantize_tensor] I=%zu K=%zu scale=%.9g mean=%.6g sigma=%.6g max_abs=%.6g count=%zu",
            args.I, args.K,
            static_cast<double>(meta->scale),
            stats.mean, stats.sigma, stats.max_abs, stats.count);
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
    if (!src || !dst || args.I == 0 || args.K == 0 ||
        dst_row_stride == 0 || dst_col_stride == 0 ||
        rows == 0 || cols == 0) {
        return false;
    }

    const auto *meta_ptr = std::get_if<Meta>(&args.act_quant.storage());
    if (!meta_ptr) {
        return false;
    }
    const Meta &meta = *meta_ptr;

    if (args.sA != 0 && args.sA != args.K) {
        return false;
    }

    const size_t src_row_stride = args.K;
    const size_t row_count = std::min(rows, args.I);
    const size_t col_count = std::min(cols, args.K);
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (row_count != 0 && col_count > max_size / row_count) {
        return false;
    }

    std::vector<int32_t> residuals(row_count * col_count, 0);

    #if ERROR_COMPENSATION
        for (const auto &outlier : meta.outliers) {
            if (outlier.row < 0 || outlier.col < 0) {
                continue;
            }

            const size_t row = static_cast<size_t>(outlier.row);
            const size_t col = static_cast<size_t>(outlier.col);
            if (row < row_count && col < col_count) {
                residuals[row * col_count + col] += outlier.residual;
            }
        }
    #endif

    for (size_t row = 0; row < row_count; ++row) {
        for (size_t col = 0; col < col_count; ++col) {
            if ((row != 0 && src_row_stride > max_size / row) ||
                (row != 0 && dst_row_stride > max_size / row) ||
                (col != 0 && dst_col_stride > max_size / col)) {
                return false;
            }

            const size_t src_row_offset = row * src_row_stride;
            if (src_row_offset > max_size - col) {
                return false;
            }

            const size_t dst_row_offset = row * dst_row_stride;
            const size_t dst_col_offset = col * dst_col_stride;
            if (dst_row_offset > max_size - dst_col_offset) {
                return false;
            }

            const int32_t q =
                static_cast<int32_t>(src[src_row_offset + col]) +
                residuals[row * col_count + col];
            dst[dst_row_offset + dst_col_offset] =
                static_cast<float>(q) * meta.scale;
        }
    }

    return true;
}

}
