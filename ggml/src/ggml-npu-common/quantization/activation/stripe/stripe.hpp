#pragma once

#include "types.hpp"

#include <cstddef>

struct ggml_tensor;
struct npu_matmul_args;

namespace ggml::npu::quants::act::stripe {
void set_config(Meta & meta);

bool quantize(const ggml_tensor * src, npu_matmul_args & args);

bool dequantize_activation(float *                     dst,
                           size_t                      dst_row_stride,
                           size_t                      dst_col_stride,
                           size_t                      rows,
                           size_t                      cols,
                           const npu_matmul_args & args);

} // namespace ggml::npu::quants::act::stripe
