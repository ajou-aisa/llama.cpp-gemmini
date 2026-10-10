#pragma once

#if defined(GGML_GEMMINI_CUDA_CPU_EXACT)
#include "../ggml-cuda/cpu-exact/ggml-cuda-cpu-exact.h"
namespace gemmini_gpu {
constexpr auto enabled = ggml_cuda_cpu_exact_int_enabled;
constexpr auto int_dot = ggml_cuda_cpu_exact_int_dot;
constexpr auto hp1_dot = ggml_cuda_cpu_exact_hp1_dot;
constexpr auto hp1_residual = ggml_cuda_cpu_exact_hp1_residual;
constexpr auto int_launches = ggml_cuda_cpu_exact_int_launches;
constexpr auto residual_launches = ggml_cuda_cpu_exact_residual_launches;
constexpr auto attention_supported = ggml_cuda_cpu_exact_attention_supported;
constexpr auto attention = ggml_cuda_cpu_exact_attention;
constexpr auto attention_calls = ggml_cuda_cpu_exact_attention_calls;
constexpr auto last_error = ggml_cuda_cpu_exact_last_error;
constexpr size_t max_columns = GGML_CUDA_CPU_EXACT_MAX_COLUMNS;
using residual_event = ggml_cuda_residual_event;
using residual_status = ggml_cuda_residual_status;
inline uint64_t float_calls() { return 0; }
inline bool activation_fp16_enabled() { return false; }
constexpr const char * backend = "cuda";
constexpr const char * proof_prefix = "CUDA_CPU_EXACT_PROOF";
constexpr const char * enable_env = "GGML_GEMMINI_CUDA_CPU_EXACT";
constexpr const char * head_env = "GGML_GEMMINI_CUDA_CPU_EXACT_Q6_HEAD";
}
#elif defined(GGML_GEMMINI_METAL_CPU_EXACT)
#include "../ggml-metal/ggml-metal-cpu-exact.h"
#include "../ggml-metal/ggml-metal-cpu-exact-int.h"
namespace gemmini_gpu {
constexpr auto enabled = ggml_metal_cpu_exact_int_enabled;
constexpr auto int_dot = ggml_metal_cpu_exact_int_dot;
constexpr auto hp1_dot = ggml_metal_cpu_exact_hp1_dot;
constexpr auto hp1_residual = ggml_metal_cpu_exact_hp1_residual;
constexpr auto int_launches = ggml_metal_cpu_exact_int_launches;
constexpr auto residual_launches = ggml_metal_cpu_exact_residual_launches;
constexpr auto attention_supported = ggml_metal_cpu_exact_attention_supported;
constexpr auto attention = ggml_metal_cpu_exact_attention;
constexpr auto attention_calls = ggml_metal_cpu_exact_attention_calls;
constexpr auto last_error = ggml_metal_cpu_exact_last_error;
constexpr auto float_calls = ggml_metal_cpu_exact_gpu_calls;
constexpr auto activation_fp16_enabled = ggml_metal_cpu_exact_activation_fp16_enabled;
constexpr size_t max_columns = GGML_METAL_CPU_EXACT_MAX_COLUMNS;
using residual_event = ggml_metal_residual_event;
using residual_status = ggml_metal_residual_status;
constexpr const char * backend = "metal";
constexpr const char * proof_prefix = "METAL_CPU_EXACT_PROOF";
constexpr const char * enable_env = "GGML_GEMMINI_METAL_CPU_EXACT";
constexpr const char * head_env = "GGML_GEMMINI_METAL_CPU_EXACT_Q6_HEAD";
}
#endif
