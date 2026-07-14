#pragma once

#include "../types.hpp"

#include <vector>

namespace ggml::gemmini::quants::act::tensor
{

struct Meta
{
    float scale = 1.0f;
    std::vector<ggml_gemmini_qact_outlier> outliers;

    inline void reset() {
        scale = 1.0f;
        outliers.clear();
    }
};

}
