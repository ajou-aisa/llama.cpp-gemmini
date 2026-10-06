#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal-quantized-producer.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_quantized_trace {
    // [m,n,ceil(k/fragment)], fragment=32 for BLOCK, min(dim,32) for HP1.
    int32_t * raw_dots;
    int32_t * scu_values;
    size_t fragment_capacity;
    int32_t * dense_integer;
    int64_t * correction;
    size_t output_capacity;
    // Requests concatenated in input order, each [request.m,request.n].
    int32_t * lane_integer;
    size_t lane_capacity;
};

struct ggml_metal_quantized_stats {
    uint64_t block_calls;
    uint64_t hp1_calls;
    uint64_t dense_launches;
    uint64_t residual_launches;
    uint64_t merge_launches;
    uint64_t failed_calls;
    uint64_t fallback_calls;
    double producer_seconds;
    double transfer_seconds;
    double dense_gpu_seconds;
    double residual_gpu_seconds;
    double merge_gpu_seconds;
    double total_seconds;
};

GGML_BACKEND_API bool ggml_metal_quantized_enabled(void);
GGML_BACKEND_API bool ggml_metal_quantized_supports_op(const struct ggml_tensor * op);
GGML_BACKEND_API enum ggml_status ggml_metal_quantized_compute(struct ggml_tensor * op);
GGML_BACKEND_API const char * ggml_metal_quantized_last_error(void);
GGML_BACKEND_API struct ggml_metal_quantized_stats ggml_metal_quantized_get_stats(void);
GGML_BACKEND_API void ggml_metal_quantized_reset_stats(void);

// Synchronous Metal execution, including completion and error checks. Output and
// trace arrays are published only on success. Fixtures may use non-K32 tails;
// model admission remains restricted to valid original GGUF layouts.
GGML_BACKEND_API bool ggml_metal_quantized_execute(
    const struct ggml_metal_quantized_view * view,
    const struct ggml_metal_quantized_request * const * requests,
    float * output,
    struct ggml_metal_quantized_trace * trace);

#ifdef __cplusplus
}
#endif
