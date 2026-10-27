#pragma once
#include "ggml-cuda-cpu-exact.h"

// Opt-in Gemmini arithmetic. This does not change quantization or attention.
GGML_API bool ggml_hp1_scu_enabled();
GGML_API bool ggml_cuda_hp1_scu_main(const int32_t * a, const void * weights, unsigned bits,
    size_t rows, size_t columns, size_t k, int32_t * output);
GGML_API ggml_cuda_residual_status ggml_cuda_hp1_scu_residual(const void * weights,
    size_t bytes, unsigned bits, size_t rows, size_t columns, size_t k,
    const uint32_t * offsets, const ggml_cuda_residual_event * events, size_t event_count,
    int64_t * output);
struct ggml_hp1_scu_stats {
    uint64_t main_launches, residual_launches;
    uint64_t main_fragments, residual_fragments;
    uint64_t main_shift_saturations, main_acc_saturations, residual_shift_saturations, residual_acc_saturations;
};
GGML_API ggml_hp1_scu_stats ggml_cuda_hp1_scu_stats();
GGML_API const char * ggml_cuda_hp1_scu_error();
