#pragma once
#include "ggml.h"
#include <cstddef>
#include <cstdint>

struct ggml_cuda_potal_attention_weights;
struct ggml_cuda_potal_attention_fragment {
    uint32_t compact_begin, count, original_block;
    uint32_t original_k[32];
};
struct ggml_cuda_potal_attention_stats {
    uint64_t main_dot_calls, residual_dot_calls, gemm_batches, fragments;
    uint64_t main_fragments, residual_fragments;
};
GGML_API bool ggml_cuda_potal_attention_enabled();
GGML_API ggml_cuda_potal_attention_weights * ggml_cuda_potal_attention_create(
    const int32_t * low, const int32_t * high, const uint32_t * carriers,
    size_t columns, size_t padded_k, unsigned bits);
GGML_API void ggml_cuda_potal_attention_destroy(ggml_cuda_potal_attention_weights * weights);
// Null fragments selects all original K32 blocks in order. Explicit fragments
// preserve compact residual runs, their original K positions and SCU carriers.
GGML_API bool ggml_cuda_potal_attention_dot(const ggml_cuda_potal_attention_weights * weights,
    unsigned pass, const int8_t * activations, size_t rows, size_t row_stride,
    size_t column_begin, size_t columns, const ggml_cuda_potal_attention_fragment * fragments,
    size_t fragment_count, bool residual, int32_t * output);
GGML_API ggml_cuda_potal_attention_stats ggml_cuda_potal_attention_get_stats();
GGML_API const char * ggml_cuda_potal_attention_error();
