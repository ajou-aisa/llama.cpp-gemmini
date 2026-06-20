#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_tensor;
struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::act {

void quantize(const ggml_tensor *src, ggml_gemmini_args_t &args);

void dequantize(const ggml_gemmini_args_t &args,
                size_t k_offset, size_t block_k,
                const int32_t *acc32, size_t acc_stride);

std::vector<QactOutlier> outliers(const ggml_gemmini_args_t &args);

std::vector<float> activation_scales(const ggml_gemmini_args_t &args, size_t row_count);

} // namespace ggml::gemmini::quants::act
