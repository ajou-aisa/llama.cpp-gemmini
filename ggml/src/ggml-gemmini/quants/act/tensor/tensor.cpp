#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

#include "tensor.hpp"

#include "../../../ggml-gemmini-args.h"
#include "../../common/tensor_util.hpp"

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

bool mark_outliers_3sigma(const float *src_data, const ggml_gemmini_args_t &args, const TensorStats &stats, BitMask &mask)
{
    if (!src_data || !mask.resize(args.I, args.K)) {
        return false;
    }

    if (stats.sigma == 0.0) {
        return true;
    }

    for (size_t row = 0; row < args.I; ++row) {
        for (size_t col = 0; col < args.K; ++col) {
            const float value = src_data[row * args.K + col];
            if (!std::isfinite(value)) {
                mask.mark_outlier(row, col);
                continue;
            }

            const double z_score = (static_cast<double>(value) - stats.mean) / stats.sigma;
            if (std::fabs(z_score) > 3.0) {
                mask.mark_outlier(row, col);
            }
        }
    }

    return true;
}

}

    // per-tensor에서는 의미 없음. 
void set_config(Meta &meta)
{
    (void)meta;
}

void set_scale(const ggml_gemmini_args_t &args, Meta &meta)
{
    // TODO: per-tensor scale 결정. outlier 리스트에 해당하는 값은 scale 계산에 포함하지 않도록 해야 함
    (void)args;
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

    #if ERROR_COMPENSATION
        const float *src_data = ggml::gemmini::activation_data(src);
        if (!src_data) {
            return false;
        }

        TensorStats stats;
        BitMask outliers;
        if (!compute_tensor_stats(src_data, args, stats)) {
            return false;
        }
        if (!mark_outliers_3sigma(src_data, args, stats, outliers)) {
            return false;
        }
        (void)outliers;
    #endif
    // TODO: outlier selection하지 않고 모든 값을 inlier로 간주.

    // TODO: int32_t 타입으로 양자화 후 [-128, 127]을 초과하는 값의 clipping된 residual을 index와 함께 outlier로 저장.
    // TODO: quantization 완료 시 true 반환, 실패 시 false 반환
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

    const size_t src_row_stride = args.sA != 0 ? args.sA : args.K;
    const size_t row_count = std::min(rows, args.I);
    const size_t col_count = std::min(cols, args.K);
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t col = 0; col < col_count; ++col) {
            const int8_t q = src[row * src_row_stride + col];
            dst[row * dst_row_stride + col * dst_col_stride] =
                static_cast<float>(q) * meta.scale;
        }
    }

    return true;
}

}
