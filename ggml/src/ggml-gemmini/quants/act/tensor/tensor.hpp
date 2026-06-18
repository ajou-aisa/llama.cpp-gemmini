#pragma once
#include "types.hpp"

#include <cstddef>
#include <cstdint>

struct ggml_gemmini_args_t;
struct ggml_tensor;

namespace ggml::gemmini::quants::act::tensor
{
    void set_config(Meta &meta);

    void set_scale(const ggml_gemmini_args_t &args, Meta &meta);

    bool quantize(
        Meta &meta,
        const ggml_tensor *src,
        const ggml_gemmini_args_t &args,
        int8_t *dst);

    void dequantize(
        const ggml_gemmini_args_t &args,
        const int32_t *acc32,
        size_t acc_stride);
}
