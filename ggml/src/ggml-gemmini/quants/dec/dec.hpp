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
    size_t int_mac_count = 0;
    size_t logical_weight_reference_count = 0;
    size_t weight_scalar_load_count = 0;
    size_t weight_vector_load_count = 0;
    size_t estimated_weight_bytes_read = 0;
    size_t active_row_k_pairs = 0;
    size_t rows_per_active_k_max = 0;
    size_t ycom_global_write_count = 0;
    size_t current_sparse_plan_bytes = 0;
    size_t group_k_csc_plan_bytes = 0;
    size_t thread_scratch_bytes = 0;
};

ActivationDECResult compensate_activation_dec(
    const std::vector<QactOutlier> &outliers,
    ggml_gemmini_args_t &args,
    const char *layer);
} // namespace ggml::gemmini::quants::dec
