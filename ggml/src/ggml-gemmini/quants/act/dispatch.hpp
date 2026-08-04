#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_tensor;
struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::act {

bool quantize(const ggml_tensor *src, ggml_gemmini_args_t &args);

bool dequantize_activation(float *dst,
                           size_t dst_row_stride,
                           size_t dst_col_stride,
                           size_t rows,
                           size_t cols,
                           const ggml_gemmini_args_t &args);

std::vector<QactOutlier> outliers(const ggml_gemmini_args_t &args);
const std::vector<QactOutlier> &outliers_view(const ggml_gemmini_args_t &args);

std::vector<float> activation_scales(const ggml_gemmini_args_t &args, size_t row_count);

} // namespace ggml::gemmini::quants::act
