#include "quantize.hpp"
#include "exsia/exsia.hpp"
#include "tensor/types.hpp"
#include "../../ggml-gemmini-args.h"
#include "../../ggml-gemmini-config.hpp"
#include "../common/tensor_util.hpp"

#include <vector>
#include <variant>

namespace ggml::gemmini::quants { namespace
{
    void configure_exsia_meta(const ggml_gemmini_args_t &args, act::exsia::Meta &meta)
    {
        if (args.layer_type != ggml::gemmini::types::LayerType::unknown) {
            (void)args.layer_type;
        }
        if (args.exsia_override_enabled) {
            const int q = args.exsia_q;
            if (q >= 2 && q <= 8) {
                meta.rho = static_cast<int16_t>(q - 2);
            }
        }
    }

}

void quantize_activation(const ggml_tensor *src, ggml_gemmini_args_t &args)
{
    reset_activation_quant_state(args);

    args.gemmini_call_k_logical = args.K;
    args.gemmini_call_k_aligned = args.K;
    args.gemmini_call_tile_k_elems = args.tile_K > 0 ? args.tile_K * DIM : args.K;

    int8_t *dst = reinterpret_cast<int8_t *>(args.A);
    if (!src || src->type != GGML_TYPE_F32 || !dst || args.I == 0 || args.K == 0) {
        return;
    }

    if (!ggml::gemmini::activation_data(src)) {
        return;
    }

    switch (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT) {
    case ggml::gemmini::config::ActivationQuantAlgo::EXSIA:
    default:
    {
        if (args.K % static_cast<size_t>(GGML_GEMMINI_BLOCK_SIZE) != 0) {
            return;
        }
        init_exsia_meta(args);
        configure_exsia_meta(args, get_exsia_meta_mut(args));
        act::exsia::ExSIA exsia;
        exsia.run(get_exsia_meta_mut(args), src, args);
        break;
    }
    case ggml::gemmini::config::ActivationQuantAlgo::TENSOR:
        args.act_quant.storage() = act::tensor::Meta{};
        break;
    }
}

void reset_activation_quant_state(ggml_gemmini_args_t &args) {
    args.act_quant.reset();
    args.gemmini_call_k_logical = 0;
    args.gemmini_call_k_aligned = 0;
    args.gemmini_call_tile_k_elems = 0;
}

void init_exsia_meta(ggml_gemmini_args_t &args)
{
    args.act_quant.storage() = act::exsia::Meta{};
}

const act::exsia::Meta &get_exsia_meta(const ggml_gemmini_args_t &args)
{
    return std::get<act::exsia::Meta>(args.act_quant.storage());
}

act::exsia::Meta &get_exsia_meta_mut(ggml_gemmini_args_t &args)
{
    return std::get<act::exsia::Meta>(args.act_quant.storage());
}

std::vector<QactOutlier> activation_outliers(const ggml_gemmini_args_t &args)
{
    const auto &storage = args.act_quant.storage();
    if (const auto *meta = std::get_if<act::exsia::Meta>(&storage)) {
        return meta->outliers;
    }

    return {};
}
} // namespace ggml::gemmini::quants
