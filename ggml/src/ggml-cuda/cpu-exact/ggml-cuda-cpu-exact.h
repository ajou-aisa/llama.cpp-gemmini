#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>

constexpr size_t GGML_CUDA_CPU_EXACT_MAX_COLUMNS = 512;
struct ggml_cuda_residual_event { uint32_t k; int32_t value; };
enum class ggml_cuda_residual_status { success, invalid_input, overflow, gpu_failure };

GGML_API bool ggml_cuda_cpu_exact_int_enabled();
GGML_API bool ggml_cuda_cpu_exact_int_dot(const int32_t * a, const int32_t * w,
    size_t rows, size_t columns, size_t k, size_t block_size, int32_t * dots);
GGML_API bool ggml_cuda_cpu_exact_hp1_dot(const int32_t * a, const void * w, unsigned bits,
    size_t rows, size_t columns, size_t k, size_t block_size, int32_t * dots);
GGML_API ggml_cuda_residual_status ggml_cuda_cpu_exact_hp1_residual(
    const void * weights, size_t weight_bytes, unsigned bits, size_t rows, size_t columns, size_t k,
    const uint32_t * offsets, const ggml_cuda_residual_event * events, size_t event_count, int64_t * output);
GGML_API bool ggml_cuda_cpu_exact_attention_supported(const ggml_tensor * op);
GGML_API bool ggml_cuda_cpu_exact_attention(ggml_tensor * op);
GGML_API const char * ggml_cuda_cpu_exact_last_error();
GGML_API uint64_t ggml_cuda_cpu_exact_int_launches();
GGML_API uint64_t ggml_cuda_cpu_exact_residual_launches();
GGML_API uint64_t ggml_cuda_cpu_exact_attention_calls();
