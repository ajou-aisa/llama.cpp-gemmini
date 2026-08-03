#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::dec
{
struct DecRoutePlan;

inline constexpr size_t kDecInt64JTileWidth = 128;

inline size_t dec_int64_j_tile_count(size_t columns)
{
    return columns / kDecInt64JTileWidth + (columns % kDecInt64JTileWidth != 0);
}

int resolve_dec_threads(size_t task_count, int omp_max_threads);
int resolve_dec_threads(size_t task_count);

void accumulate_to_ycom_jmajor_blocked(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs,
    const std::pair<int, int32_t> *rk_pairs,
    float *Y_com);

void accumulate_single_row_to_ycom_jmajor_blocked(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k,
    float *Y_com);

void accumulate_to_ycom_int64_scalar(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs,
    const std::pair<int, int32_t> *rk_pairs,
    float *Y_com);

void accumulate_single_row_to_ycom_int64_scalar(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k,
    float *Y_com);

void accumulate_to_ycom_int64_channel_direct(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs,
    const std::pair<int, int32_t> *rk_pairs,
    float *Y_com);

void accumulate_single_row_to_ycom_int64_channel_direct(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k,
    float *Y_com);

void accumulate_to_ycom_int64_channel_sidecar(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs,
    const std::pair<int, int32_t> *rk_pairs,
    float *Y_com);

void accumulate_single_row_to_ycom_int64_channel_sidecar(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k,
    float *Y_com);

void accumulate_to_ycom_int64_block(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs,
    const std::pair<int, int32_t> *rk_pairs,
    float *Y_com);

void accumulate_single_row_to_ycom_int64_block(
    const ggml_gemmini_args_t &args,
    const DecRoutePlan &plan,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k,
    float *Y_com);

void accumulate_to_ycom_int64_h1(const ggml_gemmini_args_t &, const DecRoutePlan &, size_t, size_t, const float *, const std::vector<int> &, const std::vector<size_t> &, const std::pair<int, int32_t> *, float *);
void accumulate_single_row_to_ycom_int64_h1(const ggml_gemmini_args_t &, const DecRoutePlan &, size_t, const float *, const std::vector<int> &, const std::vector<int64_t> &, float *);

void accumulate_to_ycom(
    const float *Wk_f,
    size_t J,
    size_t rk_beg,
    size_t rk_end,
    const std::pair<int, int32_t> *rk_pairs,
    const float *activation_scales,
    float *Y_com);

void accumulate_single_row_delta_to_ycom(
    const float *Wk_f,
    size_t J,
    int64_t delta_i64,
    const float *activation_scales,
    float *Y_com);

void apply_ycom_to_output(
    const float *Y_com,
    size_t I,
    size_t J,
    const ggml_gemmini_args_t &args);
} // namespace ggml::gemmini::quants::dec
