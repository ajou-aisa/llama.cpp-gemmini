#pragma once

#include "../act/types.hpp"

#include <cstddef>
#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::dec
{
struct ActivationDECResult
{
    size_t total_selected = 0;
    size_t nnz = 0;
    size_t unique_k_count = 0;
};

ActivationDECResult compensate_activation_dec(
    const std::vector<QactOutlier> &outliers,
    ggml_gemmini_args_t &args,
    const char *layer);
} // namespace ggml::gemmini::quants::dec
