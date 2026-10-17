// Copied from ggml-gemmini/quants/act/tensor/tensor.cpp with GGML_GEMMINI_ENABLE_RMD=0:
// the outlier mask, residual capture and residual dequantization are removed
// (error compensation is not part of ggml-potal). The scale, rounding and
// clipping are unchanged.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

#include "../../../ggml-potal-args.h"
#include "../../../matmul/options.hpp"
#include "../../common/math.hpp"
#include "../../common/tensor_util.hpp"
#include "tensor.hpp"

namespace ggml::potal::quants::act::tensor {
namespace {

struct TensorStats {
    double mean    = 0.0;
    double sigma   = 0.0;
    double max_abs = 0.0;
    size_t count   = 0;
};

bool compute_tensor_stats(const float *             src_data,
                          const ggml_potal_args_t & args,
                          TensorStats &             stats) {
    if (!src_data)
        return false;

    double sum    = 0.0;
    double sum_sq = 0.0;
    for (size_t row = 0; row < args.I; ++row) {
        for (size_t col = 0; col < args.K; ++col) {
            const float value = src_data[row * args.K + col];
            if (!std::isfinite(value))
                continue;

            const double x = static_cast<double>(value);
            sum += x;
            sum_sq += x * x;
            stats.max_abs = std::max(stats.max_abs, std::fabs(x));
            ++stats.count;
        }
    }

    if (stats.count == 0)
        return false;

    stats.mean = sum / static_cast<double>(stats.count);
    const double variance =
        std::max(0.0, sum_sq / static_cast<double>(stats.count) - stats.mean * stats.mean);
    stats.sigma = std::sqrt(variance);
    return true;
}

int32_t quantize_to_i32(float value, float scale) {
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0f)
        return 0;

    const double scaled = static_cast<double>(value) / static_cast<double>(scale);
    if (!std::isfinite(scaled))
        return scaled < 0.0 ? std::numeric_limits<int32_t>::min()
                            : std::numeric_limits<int32_t>::max();
    if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
        return std::numeric_limits<int32_t>::min();
    if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
        return std::numeric_limits<int32_t>::max();

    return static_cast<int32_t>(std::lrint(scaled));
}

int32_t clip_quantized(int32_t value) {
    const int32_t qmax    = ggml::potal::options::activation_qmax;
    const int32_t qmin    = ggml::potal::options::activation_qmin;
    const int32_t clipped = value > qmax ? qmax : (value < qmin ? qmin : value);
    return clipped;
}

bool set_scale(const TensorStats & stats, Meta & meta) {
    if (stats.count == 0 || stats.max_abs == 0.0) {
        meta.scale = 1.0f;
        return true;
    }

    meta.scale = static_cast<float>(
        stats.max_abs / static_cast<float>(ggml::potal::options::activation_qmax));
    return std::isfinite(meta.scale) && meta.scale > 0.0f;
}

} // namespace

void set_config(Meta & meta) {
    (void)meta;
}

bool quantize(const ggml_tensor * src, ggml_potal_args_t & args) {
    if (!src || src->type != GGML_TYPE_F32 || !args.A.valid() || args.I == 0 || args.K == 0)
        return false;

    auto * meta = std::get_if<Meta>(&args.act_quant.storage());
    if (!meta)
        return false;

    const float * src_data = ggml::potal::activation_data(src);
    if (!src_data)
        return false;

    TensorStats stats{};
    if (!compute_tensor_stats(src_data, args, stats))
        return false;

    if (!set_scale(stats, *meta))
        return false;

    for (size_t row = 0; row < args.I; ++row) {
        for (size_t col = 0; col < args.K; ++col) {
            const size_t  idx = row * args.K + col;
            const int32_t q32 = quantize_to_i32(src_data[idx], meta->scale);
            const int32_t q   = clip_quantized(q32);
            if (!args.A.set(row, col, q))
                return false;
        }
    }

    return true;
}

bool dequantize_activation(float *                   dst,
                           size_t                    dst_row_stride,
                           size_t                    dst_col_stride,
                           size_t                    rows,
                           size_t                    cols,
                           const ggml_potal_args_t & args) {
    if (!args.A.valid() || !dst || args.I == 0 || args.K == 0 || dst_row_stride == 0 ||
        dst_col_stride == 0 || rows == 0 || cols == 0)
        return false;

    const auto * meta_ptr = std::get_if<Meta>(&args.act_quant.storage());
    if (!meta_ptr)
        return false;
    const Meta & meta = *meta_ptr;

    if (args.sA != 0 && args.sA != args.K)
        return false;

    const size_t row_count      = std::min(rows, args.I);
    const size_t col_count      = std::min(cols, args.K);
    const size_t max_size       = std::numeric_limits<size_t>::max();
    const size_t src_row_stride = args.K;

    for (size_t row = 0; row < row_count; ++row) {
        for (size_t col = 0; col < col_count; ++col) {
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

            const int32_t q = args.A.get(row, col);
            dst[dst_row_offset + dst_col_offset] = static_cast<float>(q) * meta.scale;
        }
    }

    return true;
}

} // namespace ggml::potal::quants::act::tensor
