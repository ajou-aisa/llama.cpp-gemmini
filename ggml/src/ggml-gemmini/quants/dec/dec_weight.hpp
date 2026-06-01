#pragma once

#include <cstddef>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::dec
{
void load_weight_row_scaled(
    size_t k,
    const ggml_gemmini_args_t &args,
    const float *weight_scales,
    size_t scale_rows,
    size_t blocks_k,
    size_t block_size_k,
    float *Wk_f);
} // namespace ggml::gemmini::quants::dec
