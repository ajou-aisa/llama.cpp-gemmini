#pragma once

// Copied from ggml-gemmini/quants/act/tensor/tensor.hpp (ActivationQuantAlgo::TENSOR).
//
//   scale = float(max_abs(finite x) / float(qmax))      one scalar for the I x K tensor
//   q     = clip(lrint(double(x) / double(scale)))      [qmin, qmax]
//
// Defined edge cases (Gemmini semantics, RMD off):
//   non-finite x          excluded from max_abs, quantized to code 0
//   no finite value       quantize() fails
//   all values zero       scale 1.0, every code 0

#include "types.hpp"

#include <cstddef>

struct ggml_tensor;
struct ggml_potal_args_t;

namespace ggml::potal::quants::act::tensor {
void set_config(Meta & meta);

bool quantize(const ggml_tensor * src, ggml_potal_args_t & args);

bool dequantize_activation(float *                   dst,
                           size_t                    dst_row_stride,
                           size_t                    dst_col_stride,
                           size_t                    rows,
                           size_t                    cols,
                           const ggml_potal_args_t & args);

} // namespace ggml::potal::quants::act::tensor
