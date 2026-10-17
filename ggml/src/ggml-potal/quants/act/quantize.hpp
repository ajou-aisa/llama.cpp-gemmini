#pragma once

// Copied from ggml-gemmini/quants/act/quantize.hpp. The algorithm comes from
// the existing build option GGML_GEMMINI_ACTIVATION_QUANT; ggml-potal
// implements TENSOR only (checked in matmul/options.hpp and CMake). No
// residual accessors: error compensation is not part of ggml-potal.

#include <cstddef>

struct ggml_tensor;
struct ggml_potal_args_t;

namespace ggml::potal::quants {
bool quantize_activation(const ggml_tensor * src, ggml_potal_args_t & args);

bool dequantize_activation(float *                   dst,
                           size_t                    dst_row_stride,
                           size_t                    dst_col_stride,
                           size_t                    rows,
                           size_t                    cols,
                           const ggml_potal_args_t & args);

void reset_activation_quant_state(ggml_potal_args_t & args);

} // namespace ggml::potal::quants
