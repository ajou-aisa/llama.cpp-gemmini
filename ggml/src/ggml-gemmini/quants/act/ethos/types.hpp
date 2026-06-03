#pragma once

#include "../types.hpp"
#include "../../view/act_view.hpp"
#include <gemmini/layer.hpp>
#include "../../../types/model.hpp"

#include <utility>

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
using Config = ::ggml::gemmini::quants::act::Config;
using Result = ::ggml::gemmini::quants::act::Result;
using L2Config = ::ggml::gemmini::quants::act::L2Config;

inline void apply_ethos_defaults(Config &cfg) {
    cfg.m = GGML_GEMMINI_ETHOS_Q_DEFAULT - 2;
    cfg.qmax = static_cast<int8_t>((1 << (GGML_GEMMINI_ETHOS_Q_DEFAULT - 1)) - 1);
}

void set_config(Config &cfg);

} // namespace ggml::gemmini::quants::act::ethos
