#include "dispatch.hpp"
#include "exsia/exsia.hpp"
#include "stripe/stripe.hpp"
#include "tensor/tensor.hpp"
#include "token/token.hpp"
#include "../../ggml-gemmini-args.h"
#include "../../ggml-gemmini-config.hpp"
#include "gemmini/log.hpp"

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
    args.A.zero_fill();
}

}

ActivationMetadataView::ActivationMetadataView(const ggml_gemmini_args_t &source,
                                               size_t global_row_begin,
                                               size_t global_row_end)
    : source_(&source), global_row_begin_(global_row_begin), global_row_end_(global_row_end)
{
    if (global_row_begin > global_row_end)
        return;

    rows_per_stripe_ = source.I;
    if (source.tile_I > 0 && !checked_mul_size(source.tile_I, DIM, rows_per_stripe_))
        return;
    if (rows_per_stripe_ == 0 ||
        global_row_end > std::numeric_limits<size_t>::max() - (rows_per_stripe_ - 1))
        return;

    global_stripe_begin_ = global_row_begin / rows_per_stripe_;
    global_stripe_end_ = (global_row_end + rows_per_stripe_ - 1) / rows_per_stripe_;
    const auto &storage = source.act_quant.storage();
    if (const auto *meta = std::get_if<exsia::Meta>(&storage))
        valid_ = global_stripe_end_ <= meta->theta.size();
    else if (const auto *meta = std::get_if<token::Meta>(&storage))
        valid_ = global_row_end_ <= meta->scales.size();
    else if (const auto *meta = std::get_if<block::Meta>(&storage))
        valid_ = global_row_end_ <= meta->scales.size();
    else if (const auto *meta = std::get_if<stripe::Meta>(&storage))
        valid_ = global_stripe_end_ <= meta->scales.size();
    else
        valid_ = true;
}

bool ActivationMetadataView::valid() const
{
    return valid_;
}

size_t ActivationMetadataView::row_count() const
{
    return valid_ ? global_row_end_ - global_row_begin_ : 0;
}

size_t ActivationMetadataView::stripe_count() const
{
    return valid_ ? global_stripe_end_ - global_stripe_begin_ : 0;
}

bool ActivationMetadataView::global_row(size_t local_row, size_t &global_row) const
{
    if (!valid_ || local_row >= row_count())
        return false;
    global_row = global_row_begin_ + local_row;
    return true;
}

bool ActivationMetadataView::global_stripe(size_t local_stripe, size_t &global_stripe) const
{
    if (!valid_ || local_stripe >= stripe_count())
        return false;
    global_stripe = global_stripe_begin_ + local_stripe;
    return true;
}

bool ActivationMetadataView::scale(size_t local_row, float &scale) const
{
    size_t row = 0;
    if (!global_row(local_row, row))
        return false;

    const auto &storage = source_->act_quant.storage();
    if (const auto *meta = std::get_if<exsia::Meta>(&storage)) {
        const int16_t value = meta->resolve_stripe_theta(static_cast<int>(row / rows_per_stripe_));
        if (value == std::numeric_limits<int16_t>::min())
            return false;
        scale = std::ldexp(1.0f, value);
    } else if (const auto *meta = std::get_if<tensor::Meta>(&storage)) {
        scale = meta->scale;
    } else if (const auto *meta = std::get_if<token::Meta>(&storage)) {
        scale = meta->scales[row];
    } else if (const auto *meta = std::get_if<block::Meta>(&storage)) {
        scale = meta->scales[row];
    } else if (const auto *meta = std::get_if<stripe::Meta>(&storage)) {
        scale = meta->scales[row / rows_per_stripe_];
    } else {
        return false;
    }
    return std::isfinite(scale) && scale > 0.0f;
}

bool ActivationMetadataView::theta(size_t local_stripe, int16_t &theta) const
{
    size_t stripe = 0;
    if (!global_stripe(local_stripe, stripe))
        return false;
    const auto *meta = std::get_if<exsia::Meta>(&source_->act_quant.storage());
    if (meta == nullptr)
        return false;
    theta = meta->resolve_stripe_theta(static_cast<int>(stripe));
    return theta != std::numeric_limits<int16_t>::min();
}

bool quantize(const ggml_tensor *src, ggml_gemmini_args_t &args)
{
    switch (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT) {
    case ggml::gemmini::config::ActivationQuantAlgo::EXSIA:
    default:
    {
        auto &meta = args.act_quant.storage().emplace<exsia::Meta>();
        exsia::ExSIA exsia;
        if (!exsia.run(meta, src, args, args.exsia_stripe_ready_sink)) {
            ggml::gemmini::log::debug(
                ggml::gemmini::types::to_string(args.layer_type),
                "[exsia] quantization failed failure_code=%d failure_stripe=%zu",
                static_cast<int>(exsia.state().failure_code),
                exsia.state().failure_stripe);
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
    case ggml::gemmini::config::ActivationQuantAlgo::STRIPE:
    {
        args.act_quant.storage().emplace<stripe::Meta>();
        if (!stripe::quantize(src, args)) {
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
    case ggml::gemmini::config::ActivationQuantAlgo::STRIPE:
        return stripe::dequantize_activation(dst, dst_row_stride, dst_col_stride, rows, cols, args);
    }
}

const RmdPacketList &rmd_packets(const ggml_gemmini_args_t &args)
{
    static const RmdPacketList empty;
    const auto &storage = args.act_quant.storage();
    if (const auto *meta = std::get_if<exsia::Meta>(&storage)) return meta->rmd_packets;
    if (const auto *meta = std::get_if<tensor::Meta>(&storage)) return meta->rmd_packets;
    if (const auto *meta = std::get_if<token::Meta>(&storage)) return meta->rmd_packets;
    if (const auto *meta = std::get_if<stripe::Meta>(&storage)) return meta->rmd_packets;
    if (const auto *meta = std::get_if<block::Meta>(&storage)) return meta->rmd_packets;
    return empty;
}


const DirectResidualList &direct_residuals(const ggml_gemmini_args_t &args)
{
    static const DirectResidualList empty;
    const auto &storage = args.act_quant.storage();
    if (const auto *meta = std::get_if<exsia::Meta>(&storage)) return meta->direct_residuals;
    if (const auto *meta = std::get_if<tensor::Meta>(&storage)) return meta->direct_residuals;
    if (const auto *meta = std::get_if<token::Meta>(&storage)) return meta->direct_residuals;
    if (const auto *meta = std::get_if<stripe::Meta>(&storage)) return meta->direct_residuals;
    if (const auto *meta = std::get_if<block::Meta>(&storage)) return meta->direct_residuals;
    return empty;
}

std::vector<float> activation_scales(const ggml_gemmini_args_t &args, size_t row_count)
{
    std::vector<float> scales(row_count);
    if (row_count > std::numeric_limits<size_t>::max() - args.activation_row_offset)
        return {};
    const ActivationMetadataView view(
        args, args.activation_row_offset, args.activation_row_offset + row_count);
    if (!view.valid())
        return {};
    for (size_t row = 0; row < row_count; ++row) {
        if (!view.scale(row, scales[row]))
            return {};
    }
    return scales;
}

} // namespace ggml::gemmini::quants::act
