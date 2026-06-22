#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::dec
{
void accumulate_to_ycom_jmajor_blocked(
    const ggml_gemmini_args_t &args,
    const float *weight_scales,
    size_t scale_rows,
    size_t blocks_k,
    size_t block_size_k,
    size_t I,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<size_t> &rk_offs,
    const std::pair<int, int32_t> *rk_pairs,
    float *Y_com);

void accumulate_single_row_to_ycom_jmajor_blocked(
    const ggml_gemmini_args_t &args,
    const float *weight_scales,
    size_t scale_rows,
    size_t blocks_k,
    size_t block_size_k,
    size_t J,
    const float *activation_scales,
    const std::vector<int> &unique_k,
    const std::vector<int64_t> &delta_by_k,
    float *Y_com);

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
