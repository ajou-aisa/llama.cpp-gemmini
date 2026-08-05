#pragma once

#include "../act/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct ggml_gemmini_args_t;

#ifndef GGML_GEMMINI_DEC_H1_SMALL_GROUP_FASTPATH
#define GGML_GEMMINI_DEC_H1_SMALL_GROUP_FASTPATH 1
#endif

namespace ggml::gemmini::quants::dec
{
enum class DispatchOverride : uint8_t
{
    automatic,
    row_direct,
    group_k_csc,
};

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

enum class DecStatus : uint8_t
{
    success,
    unsupported,
    invalid_arguments,
    allocation_failure,
    invalid_shard,
    execution_failed,
};

struct DecColumnRange
{
    size_t begin = 0;
    size_t end = 0;
};

struct DecPreparationCounters
{
    size_t route_plan_build_count = 0;
    size_t active_row_plan_build_count = 0;
    size_t group_k_csc_plan_build_count = 0;
};

enum class PreparedDecSelectedRoute : uint8_t
{
    row_direct,
    group_k_csc,
    h1_small_group_single,
    h1_small_group_2_to_4,
};

struct PreparedDecWorkloadHistogram
{
    size_t residual_nnz = 0;
    double residual_density = 0.0;
    size_t active_row_groups = 0;
    size_t active_k = 0;
    size_t bin_1 = 0;
    size_t bin_2_to_4 = 0;
    size_t bin_5_to_8 = 0;
    size_t bin_over_8 = 0;
    double rows_per_active_k_mean = 0.0;
    size_t rows_per_active_k_max = 0;
    size_t estimated_int_mac_count = 0;
    size_t ycom_write_count = 0;
    size_t weight_scalar_load_count = 0;
    size_t weight_vector_load_count = 0;
    PreparedDecSelectedRoute selected_route = PreparedDecSelectedRoute::row_direct;
};

struct DecShardScratch
{
    std::vector<float> ycom;
    ActivationDECResult result{};
    size_t internal_thread_count = 0;
};

class PreparedDecSlice
{
public:
    size_t shard_count() const;
    DecColumnRange shard_range(size_t shard_index) const;
    size_t nr() const;
    bool uses_group_k_csc() const;
    const ActivationDECResult & result() const;
    const PreparedDecWorkloadHistogram & workload_histogram() const;
    DecPreparationCounters preparation_counters() const;

private:
    struct Data;
    explicit PreparedDecSlice(std::shared_ptr<const Data> data);

    std::shared_ptr<const Data> data_;

    friend DecStatus prepare_activation_dec_slice(
        const std::vector<QactOutlier> &, const ggml_gemmini_args_t &, size_t, size_t,
        size_t, DispatchOverride, std::shared_ptr<const PreparedDecSlice> &);
    friend DecStatus execute_prepared_dec_shard(
        const PreparedDecSlice &, size_t, DecShardScratch &, float *, size_t);
};

bool h1_small_group_fastpath_enabled();

DecStatus prepare_activation_dec_slice(
    const std::vector<QactOutlier> &outliers,
    const ggml_gemmini_args_t &args,
    size_t row_begin,
    size_t row_end,
    size_t requested_shards,
    DispatchOverride dispatch_override,
    std::shared_ptr<const PreparedDecSlice> &prepared);

DecStatus execute_prepared_dec_shard(
    const PreparedDecSlice &prepared,
    size_t shard_index,
    DecShardScratch &scratch,
    float *ycom_output = nullptr,
    size_t ycom_stride = 0);

enum class ActivationDECRowSliceStatus {
    success,
    unsupported,
    invalid_arguments,
};

ActivationDECResult compensate_activation_dec(
    const std::vector<QactOutlier> &outliers,
    ggml_gemmini_args_t &args,
    const char *layer,
    DispatchOverride override = DispatchOverride::automatic,
    float * ycom_output = nullptr,
    size_t ycom_stride = 0);

ActivationDECRowSliceStatus compensate_activation_dec_rows(
    const std::vector<QactOutlier> &outliers,
    ggml_gemmini_args_t &args,
    size_t row_begin,
    size_t row_end,
    const char *layer,
    DispatchOverride override = DispatchOverride::automatic);

ActivationDECRowSliceStatus compensate_activation_dec_rows_columns(
    const std::vector<QactOutlier> &outliers,
    ggml_gemmini_args_t &args,
    size_t row_begin,
    size_t row_end,
    size_t col_begin,
    size_t col_end,
    const char *layer,
    DispatchOverride override = DispatchOverride::automatic,
    float * ycom_output = nullptr,
    size_t ycom_stride = 0);
} // namespace ggml::gemmini::quants::dec
