#pragma once
#include "ggml.h"
#include <cstdint>

// Host graph tensors; immutable model weights are retained on the CUDA device.
GGML_API bool ggml_cuda_rtn_enabled();
GGML_API int ggml_cuda_rtn_activation_bits();
GGML_API bool ggml_cuda_rtn_supported(const ggml_tensor * op);
GGML_API bool ggml_cuda_rtn_compute(ggml_tensor * op);
GGML_API const char * ggml_cuda_rtn_error();
GGML_API uint64_t ggml_cuda_rtn_dense_calls();
GGML_API uint64_t ggml_cuda_rtn_attention_calls();
GGML_API void ggml_cuda_rtn_clear();
