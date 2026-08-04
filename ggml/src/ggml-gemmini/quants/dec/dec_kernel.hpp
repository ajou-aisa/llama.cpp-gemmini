#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::dec
{
struct DecRoutePlan;
struct GroupKCSCPlan;
struct ResidualGroupEntry;
struct ActiveRowGroup;

struct H1ScaleParams
{
    uint64_t c_eff;
    float s_rf;
};

enum class GroupKCSCWidthPath : uint8_t
{
    Mixed,
    AllInt32,
    AllInt64,
};

struct GroupKCSCScalarStats
{
    size_t logical_weight_reference_count = 0;
    size_t weight_scalar_load_count = 0;
    size_t weight_vector_load_count = 0;
    size_t classification_work_count = 0;
    size_t scratch_init_count = 0;
    size_t sparse_update_count = 0;
    size_t merge_count = 0;
    size_t safe_update_count = 0;
    size_t fallback_update_count = 0;
    size_t branch_entry_classification_count = 0;
    size_t thread_scratch_bytes = 0;
    size_t int32_row_count = 0;
    size_t int64_fallback_row_count = 0;
    uint64_t classification_cycles = 0;
    uint64_t scratch_init_cycles = 0;
    uint64_t sparse_update_cycles = 0;
    uint64_t merge_cycles = 0;
    GroupKCSCWidthPath width_path = GroupKCSCWidthPath::Mixed;
};

inline constexpr size_t kDecInt64JTileWidth = 128;

inline size_t dec_int64_j_tile_count(size_t columns)
{
    return columns / kDecInt64JTileWidth + (columns % kDecInt64JTileWidth != 0);
}

int resolve_dec_threads(size_t task_count, int omp_max_threads);
int resolve_dec_threads(size_t task_count);

float apply_h1_scale_ordered(
    int64_t accumulator,
    uint64_t c_eff,
    float s_rf,
    float activation_scale);

H1ScaleParams h1_scale_params(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t j,
    size_t block);

void accumulate_to_ycom_int64_scalar(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups,
    const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices,
    float *Y_com);

bool accumulate_to_ycom_int64_scalar_group_k_csc(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan,
    float *Y_com,
    GroupKCSCScalarStats &stats);

bool accumulate_to_ycom_int64_scalar_group_k_csc_nr8(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan,
    float *Y_com,
    GroupKCSCScalarStats &stats);

bool accumulate_to_ycom_int64_scalar_group_k_csc_nr4(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan,
    float *Y_com,
    GroupKCSCScalarStats &stats);

bool accumulate_to_ycom_int64_h1_group_k_csc_nr8(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan,
    float *Y_com,
    GroupKCSCScalarStats &stats);

bool accumulate_to_ycom_int64_h1_group_k_csc_nr4(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan,
    float *Y_com,
    GroupKCSCScalarStats &stats);

bool accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan,
    float *Y_com,
    GroupKCSCScalarStats &stats);

bool accumulate_to_ycom_int32_mixed_group_k_csc_nr4(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<ResidualGroupEntry> &entries,
    const GroupKCSCPlan &group_k_csc_plan,
    float *Y_com,
    GroupKCSCScalarStats &stats);

void accumulate_to_ycom_int64_channel_direct(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups,
    const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices,
    float *Y_com);

void accumulate_to_ycom_int64_channel_sidecar(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups,
    const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices,
    float *Y_com);

void accumulate_to_ycom_int64_block(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<ResidualGroupEntry> &entries,
    const std::vector<ActiveRowGroup> &groups,
    const std::vector<size_t> &group_offsets,
    const std::vector<size_t> &group_row_group_indices,
    float *Y_com);

void accumulate_to_ycom_int64_h1(const ggml_gemmini_args_t &, const DecRoutePlan &, size_t, size_t, const float *, const std::vector<ResidualGroupEntry> &, const std::vector<ActiveRowGroup> &, const std::vector<size_t> &, const std::vector<size_t> &, float *);

void apply_ycom_to_output(
    const float *Y_com,
    size_t I,
    size_t J,
    const ggml_gemmini_args_t &args);
} // namespace ggml::gemmini::quants::dec
