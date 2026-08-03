#pragma once

#include <cstddef>
#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::dec
{
struct DecRoutePlan;
struct ResidualGroupEntry;
struct ActiveRowGroup;

inline constexpr size_t kDecInt64JTileWidth = 128;

inline size_t dec_int64_j_tile_count(size_t columns)
{
    return columns / kDecInt64JTileWidth + (columns % kDecInt64JTileWidth != 0);
}

int resolve_dec_threads(size_t task_count, int omp_max_threads);
int resolve_dec_threads(size_t task_count);

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
