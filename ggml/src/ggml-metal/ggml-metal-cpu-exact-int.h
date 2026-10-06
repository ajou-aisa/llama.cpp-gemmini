#pragma once

#include <stddef.h>
#include <stdint.h>
#include "ggml.h"

// Inputs are signed, unpacked A4/A8 codes. Output order is [K block][row][column].
GGML_API bool ggml_metal_cpu_exact_int_enabled();
GGML_API bool ggml_metal_cpu_exact_int_dot(const int32_t * a, const int32_t * w,
    size_t rows, size_t columns, size_t k, size_t block_size, int32_t * dots);
GGML_API uint64_t ggml_metal_cpu_exact_int_launches();
