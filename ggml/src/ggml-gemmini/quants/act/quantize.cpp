#include "quantize.hpp"
#include "ethos/ethos.hpp"
#include "ethos/types.hpp"
#include "tensor/tensor.hpp"
#include "../../ggml-gemmini-args.h"
#include "../common/tensor_util.hpp"

#include <gemmini/log.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace ggml::gemmini::quants { namespace
{
    size_t clamp_tile_extent(size_t total, size_t offset, size_t tile_span)
    {
        if (offset >= total)
            return 0;

        return std::min(tile_span, total - offset);
    }

    size_t align_up(size_t value, size_t alignment)
    {
        if (alignment == 0)
            return value;

        return ((value + alignment - 1) / alignment) * alignment;
    }

}

size_t qact_outlier_count(const ActivationQuantResult &result)
{
    return result.outliers.size();
}

const QactOutlier *qact_outliers(const ActivationQuantResult &result)
{
    if (result.outliers.empty())
        return nullptr;

    return result.outliers.data();
}

ActivationQuantResult quantize_activation_f32(
    const ggml_tensor *src,
    ggml_gemmini_args_t &args,
    int8_t *dst,
    ActivationQuantConfig &cfg)
{
    ActivationQuantConfig internal_cfg = cfg;
    internal_cfg.result.outliers.clear();
    internal_cfg.result.e_s = std::numeric_limits<int16_t>::min();
    internal_cfg.result.e_s_per_stripe.clear();

    if (!src || src->type != GGML_TYPE_F32 || !dst || args.I == 0 || args.K == 0)
        return internal_cfg.result;

    const float *src_data = ggml::gemmini::activation_data(src);
    if (!src_data)
        return internal_cfg.result;

    const size_t stride_k_bytes = src->nb[0] ? src->nb[0] : sizeof(float);
    const size_t stride_i_bytes = src->nb[1] ? src->nb[1] : args.K * stride_k_bytes;

    if (internal_cfg.block_size == 0 || (args.K % internal_cfg.block_size) != 0)
        return internal_cfg.result;

    switch (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT) {
    case ggml::gemmini::config::ActivationQuantAlgo::ETHOS:
    default:
    {
        ActivationQuantResult aggregated;
        aggregated.e_s = std::numeric_limits<int16_t>::min();

        const size_t stripe_I = args.tile_I > 0 ? args.tile_I * DIM : args.I;
        const size_t num_stripes = (args.I + stripe_I - 1) / stripe_I;
        size_t current_row_offset = 0;

        for (size_t s = 0; s < num_stripes; ++s) {
            const size_t rows_this_stripe = std::min(stripe_I, args.I - current_row_offset);

            internal_cfg.result.outliers.clear();
            internal_cfg.result.e_s = std::numeric_limits<int16_t>::min();
            internal_cfg.result.e_s_per_stripe.clear();
            internal_cfg.num_real_blocks = 0;

            act::ethos::Ethos ethos;
            if (!ethos.run(
                    internal_cfg,
                    reinterpret_cast<const char *>(src_data) + current_row_offset * stride_i_bytes,
                    stride_i_bytes,
                    stride_k_bytes,
                    rows_this_stripe,
                    args.K,
                    current_row_offset,
                    0,
                    dst + current_row_offset * args.K))
            {
                return internal_cfg.result;
            }

            aggregated.e_s_per_stripe.push_back(internal_cfg.result.e_s);
            if (aggregated.e_s == std::numeric_limits<int16_t>::min())
                aggregated.e_s = internal_cfg.result.e_s;
            aggregated.outliers.insert(
                aggregated.outliers.end(),
                internal_cfg.result.outliers.begin(),
                internal_cfg.result.outliers.end());

            current_row_offset += rows_this_stripe;
        }

        return aggregated;
    }
    case ggml::gemmini::config::ActivationQuantAlgo::TENSOR:
        return internal_cfg.result;
    }

    return internal_cfg.result;
}

void reset_activation_quant_state(ggml_gemmini_args_t &args) {
    args.act_quant.reset();
    args.gemmini_call_k_logical = 0;
    args.gemmini_call_k_aligned = 0;
    args.gemmini_call_tile_k_elems = 0;
}

ActivationQuantConfig make_activation_quant_config(const ggml_gemmini_args_t &args) {
    ActivationQuantConfig cfg {};
    cfg.block_size = static_cast<size_t>(GGML_GEMMINI_BLOCK_SIZE);
    if (args.layer_type != ggml::gemmini::types::LayerType::unknown)
        cfg.preset.second = args.layer_type;

    switch (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT) {
    case ggml::gemmini::config::ActivationQuantAlgo::ETHOS:
    default:
        act::ethos::set_config(cfg);
        break;
    case ggml::gemmini::config::ActivationQuantAlgo::TENSOR:
        act::tensor::set_config(cfg);
        break;
    }
    if (args.ethos_override_enabled) {
        const int q = args.ethos_q;
        if (q >= 2 && q <= 8) {
            cfg.m = static_cast<int16_t>(q - 2);
            cfg.qmax = static_cast<int8_t>((1 << (q - 1)) - 1);
        }
        cfg.delta = args.ethos_delta;
        cfg.l2_on = args.ethos_l2_enabled;
        cfg.l2 = {args.ethos_l2_c, args.ethos_l2_d};
    }

    return cfg;
}

void capture_activation_quant_result(
    ggml_gemmini_args_t &args,
    const ActivationQuantConfig &cfg,
    const ActivationQuantResult &res) {
    args.act_quant.ethos.e_s = res.e_s;
    args.act_quant.ethos.m = cfg.m;
    args.act_quant.ethos.e_s_per_stripe_i = res.e_s_per_stripe;

    const size_t outlier_count = qact_outlier_count(res);
    if (outlier_count == 0)
        return;

    const auto *outliers = qact_outliers(res);
    args.act_quant.outliers.clear();
    args.act_quant.outliers.reserve(outlier_count);
    for (size_t i = 0; i < outlier_count; ++i) {
        const auto &outlier = outliers[i];
        args.act_quant.outliers.push_back({outlier.row, outlier.col, outlier.original, outlier.saturated});
    }
}
} // namespace ggml::gemmini::quants
