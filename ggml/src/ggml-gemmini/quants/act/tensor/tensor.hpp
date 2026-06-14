#pragma once
#include "types.hpp"

struct ggml_gemmini_args_t;
struct ggml_tensor;

namespace ggml::gemmini::quants::act::tensor
{
    void set_config(Config &cfg);

    void set_scale(const ggml_gemmini_args_t &args, Config &cfg);

    bool quantize(
        Config &cfg,
        const ggml_tensor *src,
        const ggml_gemmini_args_t &args,
        int8_t *dst);

    void dequantize(
        const ggml_gemmini_args_t &args,
        const int32_t *acc32,
        size_t acc_stride);
}
