// Derived from Gemmini SDK develop fcfcf5ce; see matmul/README.md and LICENSE.gemmini*.
#pragma once
#include <cstddef>
#include "gemmini_params.h"
namespace ggml::gemmini {
void matmul_cpu_fp(bool transA, bool transB,
                          size_t DIM_I, size_t DIM_J, size_t DIM_K,
                          const float* A, const float* B, const float* D,
                          float* C,
                          size_t stride_A, size_t stride_B,
                          size_t stride_D, size_t stride_C);
void matmul_cpu_int32(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
                               const elem_t *A, const elem_t *B, const acc_t *D,
                               void *C,
                               size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                               scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                               int act, acc_scale_t scale, acc_scale_t bert_scale, bool repeating_bias,
                               bool full_C);
} // namespace ggml::gemmini
