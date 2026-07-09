#pragma once

#include "types.hpp"

#include <cstddef>

struct ggml_tensor;
struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::act::token
{
    void set_config(Meta &meta);

    bool quantize(const ggml_tensor *src, ggml_gemmini_args_t &args);

    bool dequantize_activation(
        float *dst,
        size_t dst_row_stride,
        size_t dst_col_stride,
        size_t rows,
        size_t cols,
        const ggml_gemmini_args_t &args);

}
