#pragma once

#include "../../view/act_view.hpp"
#include <gemmini/layer.hpp>
#include "../../../types/model.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#ifndef GGML_GEMMINI_ETHOS_C_DEFAULT
#define GGML_GEMMINI_ETHOS_C_DEFAULT 1
#endif

#ifndef GGML_GEMMINI_ETHOS_D_DEFAULT
#define GGML_GEMMINI_ETHOS_D_DEFAULT 1
#endif

#ifndef GGML_GEMMINI_ETHOS_Q_DEFAULT
#define GGML_GEMMINI_ETHOS_Q_DEFAULT 8
#endif

#ifndef GGML_GEMMINI_ETHOS_DELTA_DEFAULT
#define GGML_GEMMINI_ETHOS_DELTA_DEFAULT 0
#endif

namespace ggml::gemmini::quants::act::ethos
{
struct L2Config
{
    int c = GGML_GEMMINI_ETHOS_C_DEFAULT;
    int d = GGML_GEMMINI_ETHOS_D_DEFAULT;
};

struct Result
{
    std::vector<::ggml::gemmini::quants::QactOutlier> outliers;
    int16_t e_t = std::numeric_limits<int16_t>::min(); // tile exponent

    std::vector<int16_t> e_t_per_tile;
    size_t num_padding_blocks = 0;
};

struct Config
{
    std::pair<ggml::gemmini::types::ModelType, ggml::gemmini::types::LayerType> preset = {
        ggml::gemmini::types::ModelType::gpt2,
        ggml::gemmini::types::LayerType::ffn_norm,
    };

    size_t block_size = 32;
    int16_t m = GGML_GEMMINI_ETHOS_Q_DEFAULT - 2;
    int8_t qmax = (1 << (GGML_GEMMINI_ETHOS_Q_DEFAULT - 1)) - 1;
    int delta = 0;
    bool l2_on = false;
    L2Config l2{};
    Result result{};
    size_t num_real_blocks = 0;
};

void set_config(Config &cfg);

} // namespace ggml::gemmini::quants::act::ethos
