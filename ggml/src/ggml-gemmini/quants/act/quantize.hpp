#pragma once

#include "exsia/types.hpp"
#include "types.hpp"

#include <cstddef>
#include <vector>

struct ggml_tensor;
struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants
{
bool quantize_activation(const ggml_tensor *src, ggml_gemmini_args_t &args);

bool dequantize_activation(float *dst,
                           size_t dst_row_stride,
                           size_t dst_col_stride,
                           size_t rows,
                           size_t cols,
                           const ggml_gemmini_args_t &args);

void reset_activation_quant_state(ggml_gemmini_args_t &args);

void init_exsia_meta(ggml_gemmini_args_t &args);
const act::exsia::Meta &get_exsia_meta(const ggml_gemmini_args_t &args);
act::exsia::Meta &get_exsia_meta_mut(ggml_gemmini_args_t &args);
const act::RmdPacketList &activation_rmd_packets(const ggml_gemmini_args_t &args);

} // namespace ggml::gemmini::quants
