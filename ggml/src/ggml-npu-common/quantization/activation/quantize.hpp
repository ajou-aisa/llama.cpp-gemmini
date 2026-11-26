#pragma once

#include "exsia/types.hpp"
#include "types.hpp"

#include <cstddef>
#include <vector>

struct ggml_tensor;
struct npu_matmul_args;

namespace ggml::npu::quants {
bool quantize_activation(const ggml_tensor * src, npu_matmul_args & args);

bool dequantize_activation(float *                     dst,
                           size_t                      dst_row_stride,
                           size_t                      dst_col_stride,
                           size_t                      rows,
                           size_t                      cols,
                           const npu_matmul_args & args);

void reset_activation_quant_state(npu_matmul_args & args);

const act::RmdPacketList &      activation_rmd_packets(const npu_matmul_args & args);
const act::DirectResidualList & activation_direct_residuals(const npu_matmul_args & args);

} // namespace ggml::npu::quants
