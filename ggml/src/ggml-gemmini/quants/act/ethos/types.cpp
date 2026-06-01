#include "types.hpp"

namespace ggml::gemmini::quants::act::ethos {

void set_config(Config &cfg)
{
    switch (cfg.preset.first)
    {
    case ggml::gemmini::types::ModelType::gpt2:
        switch (cfg.preset.second)
        {
        case ggml::gemmini::types::LayerType::attn_norm:
            cfg.m = 6;
            cfg.qmax = (1 << (cfg.m + 1)) - 1;
            cfg.delta = 0;
            cfg.l2_on = false;
            cfg.l2 = {1, 1};
            break;
        case ggml::gemmini::types::LayerType::ffn_norm:
            cfg.m = 6;
            cfg.qmax = (1 << (cfg.m + 1)) - 1;
            cfg.delta = 0;
            cfg.l2_on = false;
            cfg.l2 = {1, 1};
            break;
        case ggml::gemmini::types::LayerType::ffn_gelu:
            cfg.m = 6;
            cfg.qmax = (1 << (cfg.m + 1)) - 1;
            cfg.delta = 0;
            cfg.l2_on = false;
            cfg.l2 = {1, 1};
            break;
        case ggml::gemmini::types::LayerType::kqv_out:
            cfg.m = 6;
            cfg.qmax = (1 << (cfg.m + 1)) - 1;
            cfg.delta = 0;
            cfg.l2_on = false;
            cfg.l2 = {1, 1};
            break;
        case ggml::gemmini::types::LayerType::result_norm:
            cfg.m = 6;
            cfg.qmax = (1 << (cfg.m + 1)) - 1;
            cfg.delta = 0;
            cfg.l2_on = true;
            cfg.l2 = {1, 1};
            break;
        default:
            break;
        }
        break;

    case ggml::gemmini::types::ModelType::llama_3_2_1b:
        switch (cfg.preset.second)
        {
        case ggml::gemmini::types::LayerType::attn_norm:
            cfg.m = 6;
            cfg.qmax = (1 << (cfg.m + 1)) - 1;
            cfg.delta = 0;
            cfg.l2_on = false;
            cfg.l2 = {1, 1};
            break;
        case ggml::gemmini::types::LayerType::ffn_norm:
            cfg.m = 6;
            cfg.qmax = (1 << (cfg.m + 1)) - 1;
            cfg.delta = 0;
            cfg.l2_on = false;
            cfg.l2 = {1, 1};
            break;
        case ggml::gemmini::types::LayerType::ffn_gate_par:
            cfg.m = 6;
            cfg.qmax = (1 << (cfg.m + 1)) - 1;
            cfg.delta = 0;
            cfg.l2_on = false;
            cfg.l2 = {1, 1};
            break;
        case ggml::gemmini::types::LayerType::kqv_out:
            cfg.m = 6;
            cfg.qmax = (1 << (cfg.m + 1)) - 1;
            cfg.delta = 0;
            cfg.l2_on = false;
            cfg.l2 = {1, 1};
            break;
        case ggml::gemmini::types::LayerType::result_norm:
            cfg.m = 6;
            cfg.qmax = (1 << (cfg.m + 1)) - 1;
            cfg.delta = 0;
            cfg.l2_on = true;
            cfg.l2 = {1, 1};
            break;
        default:
            break;
        }
        break;

    case ggml::gemmini::types::ModelType::gemma:
        break;

    default:
        break;
    }
}

} // namespace ggml::gemmini::quants::act::ethos
