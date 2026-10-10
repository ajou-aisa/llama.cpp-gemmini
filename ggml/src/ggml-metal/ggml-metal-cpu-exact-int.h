#pragma once

#include <stddef.h>
#include <stdint.h>
#include "ggml.h"

// Inputs are signed, unpacked A4/A8 codes. Output order is [K block][row][column].
constexpr size_t GGML_METAL_CPU_EXACT_MAX_COLUMNS = 512;
GGML_API bool ggml_metal_cpu_exact_int_enabled();
GGML_API bool ggml_metal_cpu_exact_int_dot(const int32_t * a, const int32_t * w,
    size_t rows, size_t columns, size_t k, size_t block_size, int32_t * dots);
GGML_API uint64_t ggml_metal_cpu_exact_int_launches();
GGML_API bool ggml_metal_cpu_exact_hp1_dot(const int32_t * a, const void * packed_weight, unsigned bits,
    size_t rows, size_t columns, size_t k, size_t block_size, int32_t * dots);
GGML_API bool ggml_metal_cpu_exact_block_dot(const int32_t * a, const void * packed_weight, unsigned bits,
    size_t rows, size_t columns, size_t k, size_t block_size, int32_t * dots);
GGML_API bool ggml_metal_cpu_exact_q6_dot(const int32_t * a, const void * packed_weight,
    size_t rows, size_t columns, size_t k, int32_t * dots);

struct ggml_metal_residual_event { uint32_t k; int32_t value; };
enum class ggml_metal_residual_status { success, invalid_input, overflow, gpu_failure };
GGML_API ggml_metal_residual_status ggml_metal_cpu_exact_hp1_residual(
    const void * weights, size_t weight_bytes, unsigned bits, size_t rows, size_t columns, size_t k,
    const uint32_t * row_offsets, const ggml_metal_residual_event * events, size_t event_count, int64_t * output);
GGML_API uint64_t ggml_metal_cpu_exact_residual_launches();
