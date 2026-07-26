#pragma once

#include "llama.h"
#include <algorithm>
#include <cstddef>

static inline ggml_type llama_quantize_default_type_for_ftype(llama_ftype ftype) {
    return ftype == LLAMA_FTYPE_MOSTLY_Q8_CHANNEL ? GGML_TYPE_Q8_CHANNEL : GGML_TYPE_COUNT;
}

static inline size_t llama_quantize_work_size(ggml_type type, int64_t nelements, int64_t n_per_row) {
    const size_t f32_size = (size_t) nelements * sizeof(float);
    const size_t row_size = ggml_row_size(type, n_per_row);
    const size_t quantized_size = ((size_t) nelements / (size_t) n_per_row) * row_size;
    return std::max(f32_size, quantized_size);
}
