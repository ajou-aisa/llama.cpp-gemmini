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

    bool quantize(const ggml_tensor *src, ggml_gemmini_args_t &args);

    bool dequantize_activation(
        float *dst,
        size_t dst_row_stride,
        size_t dst_col_stride,
        size_t rows,
        size_t cols,
        const ggml_gemmini_args_t &args);

    void dequantize(
        const ggml_gemmini_args_t &args,
        const int32_t *acc32,
        size_t acc_stride);
}
