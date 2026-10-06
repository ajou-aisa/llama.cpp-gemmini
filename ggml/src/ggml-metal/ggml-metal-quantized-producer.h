#pragma once

#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_metal_quantized_payload;

enum ggml_metal_quantized_mode {
    GGML_METAL_QUANTIZED_BLOCK = 0,
    GGML_METAL_QUANTIZED_HP1_EXSIA = 1,
};

struct ggml_metal_quantized_profile {
    enum ggml_metal_quantized_mode mode;
    uint32_t bits;
    uint32_t dim;
    bool residual_enabled;
};

struct ggml_metal_quantized_run {
    uint32_t original_block_id;
    uint32_t original_global_k_begin;
    uint32_t union_k_mask;
    size_t compact_k_begin;
    size_t compact_k_count;
    const uint16_t * original_local_k;
};

struct ggml_metal_quantized_row {
    uint32_t original_lane_id;
    // Local row within the request's source stripe; add source_row_begin.
    uint32_t source_row;
};

struct ggml_metal_quantized_request {
    size_t m, n, k;
    size_t source_row_begin, source_row_count;
    size_t tile_i, tile_j, tile_k;
    size_t run_count;
    const struct ggml_metal_quantized_run * runs;
    const struct ggml_metal_quantized_row * rows; // [m], original lane order
    const int8_t * activations;                 // [m,k]
    const int32_t * weights;                    // [k,n], exact original codes
    const uint32_t * carriers;                  // [run_count,n]
};

struct ggml_metal_quantized_view {
    struct ggml_metal_quantized_profile profile;
    size_t m, n, k;
    size_t activation_rows_per_stripe;
    const int8_t * activations;      // [m,k], signed scalar codes, including A4
    const int8_t * weights;          // [n,k], exact original signed codes
    const float * activation_scales; // BLOCK [m,k/32]; HP1 [m]
    const float * weight_scales;     // BLOCK [n,k/32]; NULL for HP1
    const int32_t * block_residual;  // BLOCK [m,k] or NULL; selected residual only
    const uint32_t * carriers;       // HP1 [k/32,n]; NULL for BLOCK
    const float * column_scales;     // HP1 [n]; NULL for BLOCK
    const int16_t * stripe_theta;    // HP1 [stripe_count]; NULL for BLOCK
    size_t stripe_count;
    size_t request_count;            // HP1 ordered packet requests
    const void * original_weights;  // Borrowed immutable original GGUF storage
    size_t original_weight_bytes;
    size_t original_weight_row_stride;
    int original_weight_type;
};

GGML_BACKEND_API struct ggml_metal_quantized_profile ggml_metal_quantized_get_profile(void);
GGML_BACKEND_API bool ggml_metal_quantized_can_prepare(const struct ggml_tensor * weight,
                                     const struct ggml_tensor * activation);

// Tensor data must be host-accessible and ready. All owned payload pointers stay
// valid until free; original_weights remains borrowed until the caller releases it.
// On failure *out is unchanged and error receives a diagnostic when provided.
GGML_BACKEND_API bool ggml_metal_quantized_prepare(const struct ggml_tensor * weight,
                                 const struct ggml_tensor * activation,
                                 struct ggml_metal_quantized_payload ** out,
                                 char * error, size_t error_capacity);
GGML_BACKEND_API const struct ggml_metal_quantized_view * ggml_metal_quantized_get_view(
    const struct ggml_metal_quantized_payload * payload);
GGML_BACKEND_API const struct ggml_metal_quantized_request * ggml_metal_quantized_get_request(
    const struct ggml_metal_quantized_payload * payload, size_t index);
GGML_BACKEND_API void ggml_metal_quantized_free(struct ggml_metal_quantized_payload * payload);

// Releases cached weight ownership; live payloads remain valid. Exact original
// packed bytes are compared on every reuse. Activations are never cached.
GGML_BACKEND_API void ggml_metal_quantized_clear_weight_cache(void);
// Defaults to 2 GiB, clamps larger limits to 2 GiB, and disables reuse at zero.
// At most 256 weight entries are retained. New entries that do not fit are used
// uncached; reducing the limit evicts least recently used entries first.
// The limit includes packed snapshots and decoded vector capacity; live payloads
// can retain evicted entries beyond this cache ownership limit.
GGML_BACKEND_API void ggml_metal_quantized_set_weight_cache_limit(size_t bytes);

#ifdef __cplusplus
}
#endif
