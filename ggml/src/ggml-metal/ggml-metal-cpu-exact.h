#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// A and B contain contiguous FP32 rows of length K; output contains M rows of N.
// This contract follows the Apple Clang optimized CPU FLOAT helper for K32 inputs.
bool ggml_metal_cpu_exact_float(size_t m, size_t n, size_t k,
                              const float * a, const float * b, float * output);
const char * ggml_metal_cpu_exact_last_error(void);
uint64_t ggml_metal_cpu_exact_gpu_calls(void);
uint64_t ggml_metal_cpu_exact_software_calls(void);
#ifdef __cplusplus
}
#endif
