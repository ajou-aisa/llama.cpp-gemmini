#include "dispatch.hpp"
#include "exsia/exsia.hpp"
#include "tensor/tensor.hpp"
#include "token/token.hpp"
#include "../../ggml-gemmini-args.h"
#include "../../ggml-gemmini-config.hpp"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <variant>

namespace ggml::gemmini::quants::act {

namespace {

bool checked_mul_size(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        return false;

    out = lhs * rhs;
    return true;
}

void reset_quantize_failure(ggml_gemmini_args_t &args)
{
    args.act_quant.reset();

    int8_t *dst = reinterpret_cast<int8_t *>(args.A);
    size_t elem_count = 0;
    if (dst == nullptr || !checked_mul_size(args.I, args.K, elem_count))
        return;

    if (elem_count == 0)
        elem_count = args.sA != 0 ? args.sA : args.K;

    std::fill_n(dst, elem_count, int8_t{0});
}

}

bool quantize(const ggml_tensor *src, ggml_gemmini_args_t &args)
{
    switch (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT) {
    case ggml::gemmini::config::ActivationQuantAlgo::EXSIA:
    default:
    {
        auto &meta = args.act_quant.storage().emplace<exsia::Meta>();
        exsia::ExSIA exsia;
        if (!exsia.run(meta, src, args)) {
            reset_quantize_failure(args);
            return false;
        }
        return true;
    }
    case ggml::gemmini::config::ActivationQuantAlgo::TENSOR:
    {
        args.act_quant.storage().emplace<tensor::Meta>();
        if (!tensor::quantize(src, args)) {
            reset_quantize_failure(args);
            return false;
        }
        return true;
    }
    case ggml::gemmini::config::ActivationQuantAlgo::TOKEN:
    {
        args.act_quant.storage().emplace<token::Meta>();
        if (!token::quantize(src, args)) {
            reset_quantize_failure(args);
            return false;
        }
        return true;
    }
    }
}

bool dequantize_activation(float *dst,
                           size_t dst_row_stride,
                           size_t dst_col_stride,
                           size_t rows,
                           size_t cols,
                           const ggml_gemmini_args_t &args)
{
    switch (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT) {
    case ggml::gemmini::config::ActivationQuantAlgo::EXSIA:
    default:
        return exsia::dequantize_activation(dst, dst_row_stride, dst_col_stride, rows, cols, args);
    case ggml::gemmini::config::ActivationQuantAlgo::TENSOR:
        return tensor::dequantize_activation(dst, dst_row_stride, dst_col_stride, rows, cols, args);
    case ggml::gemmini::config::ActivationQuantAlgo::TOKEN:
        if (const auto *meta = std::get_if<token::Meta>(&args.act_quant.storage());
            meta != nullptr && meta->scales.size() != args.I) {
            return false;
        }
        return token::dequantize_activation(dst, dst_row_stride, dst_col_stride, rows, cols, args);
    }
}

std::vector<QactOutlier> outliers(const ggml_gemmini_args_t &args)
{
    const auto &storage = args.act_quant.storage();
    if (const auto *meta = std::get_if<exsia::Meta>(&storage)) {
        return meta->outliers;
    }
    if (const auto *meta = std::get_if<tensor::Meta>(&storage)) {
        return meta->outliers;
    }
    if (const auto *meta = std::get_if<token::Meta>(&storage)) {
        return meta->outliers;
    }

    return {};
}

std::vector<float> activation_scales(const ggml_gemmini_args_t &args, size_t row_count)
{
    std::vector<float> scales(row_count, 1.0f);
    const auto &storage = args.act_quant.storage();

    if (const auto *meta = std::get_if<exsia::Meta>(&storage)) {
        const int16_t invalid_theta = std::numeric_limits<int16_t>::min();
        size_t rows_per_stripe = args.I;
        if (args.tile_I > 0 && !checked_mul_size(args.tile_I, DIM, rows_per_stripe)) {
            return scales;
        }
        if (rows_per_stripe == 0) {
            return scales;
        }

        for (size_t row = 0; row < row_count; ++row) {
            const size_t stripe_idx = row / rows_per_stripe;
            const int16_t theta = meta->resolve_stripe_theta(static_cast<int>(stripe_idx));
            scales[row] = theta == invalid_theta ? 1.0f : std::ldexp(1.0f, theta);
        }
        return scales;
    }

    if (const auto *meta = std::get_if<tensor::Meta>(&storage)) {
        std::fill(scales.begin(), scales.end(), meta->scale);
        return scales;
    }

    if (const auto *meta = std::get_if<token::Meta>(&storage)) {
        if (meta->scales.size() != args.I) {
            GGML_ASSERT(false && "TOKEN activation scale cardinality must equal args.I");
        }
        const size_t count = std::min(row_count, meta->scales.size());
        for (size_t row = 0; row < count; ++row) {
            const float scale = meta->scales[row];
            scales[row] = std::isfinite(scale) && scale > 0.0f ? scale : 1.0f;
        }
    }

    return scales;
}

} // namespace ggml::gemmini::quants::act
