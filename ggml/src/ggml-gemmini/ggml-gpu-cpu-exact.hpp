#pragma once

#if defined(GGML_GEMMINI_CUDA_CPU_EXACT)
#include "ggml-gemmini.h"
#include "../ggml-cuda/cpu-exact/ggml-cuda-cpu-exact.h"
#include "../ggml-cuda/cpu-exact/ggml-cuda-hp1-scu.h"
#if defined(GGML_GEMMINI_CUDA_RTN_ATTENTION)
#include "../ggml-cuda/cpu-exact/ggml-cuda-rtn.h"
#endif
namespace gemmini_gpu {
constexpr auto enabled = ggml_cuda_cpu_exact_int_enabled;
constexpr auto int_dot = ggml_cuda_cpu_exact_int_dot;
constexpr auto hp1_dot = ggml_cuda_cpu_exact_hp1_dot;
constexpr auto hp1_residual = ggml_cuda_cpu_exact_hp1_residual;
inline uint64_t int_launches() { return ggml_cuda_cpu_exact_int_launches() + ggml_cuda_hp1_scu_stats().main_launches; }
inline uint64_t residual_launches() { return ggml_cuda_cpu_exact_residual_launches() + ggml_cuda_hp1_scu_stats().residual_launches; }
inline bool attention_is_rtn() {
#if defined(GGML_GEMMINI_CUDA_RTN_ATTENTION)
    return ggml_cuda_rtn_enabled();
#else
    return false;
#endif
}
inline bool attention_supported(const ggml_tensor * op) {
    // Explicit PoTal attention keeps develop's quantized attention executor.
    if (op && ggml_gemmini_get_attention_role(op) != GGML_GEMMINI_ATTENTION_NONE) return false;
#if defined(GGML_GEMMINI_CUDA_RTN_ATTENTION)
    // Keep the baseline attention on CUDA even when HP1 main GEMM is the CPU reference.
    if (attention_is_rtn()) return op && op->src[1] &&
        (op->src[1]->ne[2] > 1 || op->src[1]->ne[3] > 1) && ggml_cuda_rtn_supported(op);
#endif
    return ggml_cuda_cpu_exact_attention_supported(op);
}
inline bool attention(ggml_tensor * op) {
#if defined(GGML_GEMMINI_CUDA_RTN_ATTENTION)
    if (attention_is_rtn()) return attention_supported(op) && ggml_cuda_rtn_compute(op);
#endif
    return ggml_cuda_cpu_exact_attention(op);
}
inline uint64_t attention_calls() {
#if defined(GGML_GEMMINI_CUDA_RTN_ATTENTION)
    if (attention_is_rtn()) return ggml_cuda_rtn_attention_calls();
#endif
    return ggml_cuda_cpu_exact_attention_calls();
}
inline const char * attention_error() {
#if defined(GGML_GEMMINI_CUDA_RTN_ATTENTION)
    if (attention_is_rtn()) return ggml_cuda_rtn_error();
#endif
    return ggml_cuda_cpu_exact_last_error();
}
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
#endif
