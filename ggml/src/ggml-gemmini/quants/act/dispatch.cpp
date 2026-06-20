#include "dispatch.hpp"
#include "exsia/exsia.hpp"
#include "tensor/tensor.hpp"
#include "../../ggml-gemmini-args.h"
#include "../../ggml-gemmini-config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <variant>

namespace ggml::gemmini::quants::act {

void quantize(const ggml_tensor *src, ggml_gemmini_args_t &args)
{
    switch (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT) {
    case ggml::gemmini::config::ActivationQuantAlgo::EXSIA:
    default:
    {
        auto &meta = args.act_quant.storage().emplace<exsia::Meta>();
        exsia::ExSIA exsia;
        exsia.run(meta, src, args);
        break;
    }
    case ggml::gemmini::config::ActivationQuantAlgo::TENSOR:
    {
        args.act_quant.storage().emplace<tensor::Meta>();
        tensor::quantize(src, args);
        break;
    }
    }
}

void dequantize(const ggml_gemmini_args_t &args,
                size_t k_offset, size_t block_k,
                const int32_t *acc32, size_t acc_stride)
{
    switch (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT) {
    case ggml::gemmini::config::ActivationQuantAlgo::EXSIA:
    default:
        exsia::dequantize(args, k_offset, block_k, acc32, acc_stride);
        break;
    case ggml::gemmini::config::ActivationQuantAlgo::TENSOR:
        tensor::dequantize(args, acc32, acc_stride);
        break;
    }
}

std::vector<QactOutlier> outliers(const ggml_gemmini_args_t &args)
{
    const auto &storage = args.act_quant.storage();
    if (const auto *meta = std::get_if<exsia::Meta>(&storage)) {
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
        const size_t rows_per_stripe = args.tile_I > 0 ? args.tile_I * DIM : args.I;
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
    }

    return scales;
}

} // namespace ggml::gemmini::quants::act
