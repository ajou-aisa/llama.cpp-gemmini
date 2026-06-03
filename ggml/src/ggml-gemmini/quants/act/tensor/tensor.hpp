#pragma once
#include "types.hpp"

struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::act::tensor
{
    void set_config(Config &cfg);

    void dequantize(
        const ggml_gemmini_args_t &args,
        size_t k_offset,
        size_t block_k,
        const int32_t *acc32,
        size_t acc_stride);
}
