#pragma once
#include <cstdint>

#ifndef GGML_GEMMINI_BLOCK_SIZE
#define GGML_GEMMINI_BLOCK_SIZE 32
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE GGML_GEMMINI_BLOCK_SIZE
#endif

namespace ggml::gemmini::quants::act
{
struct ggml_gemmini_qact_outlier
{
    int row = 0;
    int col = 0;
    int32_t residual = 0;
};

}

namespace ggml::gemmini::quants
{
using QactOutlier = act::ggml_gemmini_qact_outlier;
}
