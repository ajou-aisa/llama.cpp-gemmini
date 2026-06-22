#pragma once

#include "../act/types.hpp"

#include <cstddef>
#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::dec
{
struct ActivationDECConfig
{
    bool record_stats = false;
    const char *layer = "";
};

struct ActivationDECResult
{
    size_t total_selected = 0;
    size_t nnz = 0;
    size_t unique_k_count = 0;
    bool success = false;
};

ActivationDECResult compensate_activation_dec(
    const std::vector<QactOutlier> &outliers,
    ggml_gemmini_args_t &args,
    const ActivationDECConfig &cfg);

bool should_apply_dec(const ggml_gemmini_args_t &args);

void append_activation_outliers(
    const ggml_gemmini_args_t &args,
    std::vector<QactOutlier> &outliers);
} // namespace ggml::gemmini::quants::dec
