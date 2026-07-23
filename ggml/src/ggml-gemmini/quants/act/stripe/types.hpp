#pragma once

#include "../types.hpp"

#include <vector>

namespace ggml::gemmini::quants::act::stripe
{

struct Meta
{
    std::vector<float> scales;
    std::vector<ggml_gemmini_qact_outlier> outliers;

    inline void reset() {
        scales.clear();
        outliers.clear();
    }
};

}