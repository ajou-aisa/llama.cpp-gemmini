#pragma once

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

namespace ggml::gemmini::quants::act
{
struct ggml_gemmini_qact_outlier
{
    int row = 0;
    int col = 0;
    float original = 0.f;
    float saturated = 0.f;
};

}

namespace ggml::gemmini::quants
{
using QactOutlier = act::ggml_gemmini_qact_outlier;
}
