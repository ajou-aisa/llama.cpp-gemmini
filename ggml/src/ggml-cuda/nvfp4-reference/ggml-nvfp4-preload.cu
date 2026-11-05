// Opt-in numerical NVFP4 emulation on pre-Blackwell GPUs. No native FP4 GEMM claim.
#include "nvfp4-numeric.h"
#include "ggml-cuda-rtn.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace nvfp4_impl {
thread_local std::string error;
std::atomic<uint64_t> dense_calls{0};
template<class T> T original(const char * symbol) {
    auto p = reinterpret_cast<T>(dlsym(RTLD_NEXT, symbol));
    if (!p) throw std::runtime_error(std::string("NVFP4 missing RTN symbol: ") + symbol);
    return p;
}
bool enabled() {
    const char * value = std::getenv("GGML_CUDA_NVFP4");
    return value && !std::strcmp(value, "1");
}
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void check(cublasStatus_t e) { if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("NVFP4 cuBLAS status " + std::to_string(e)); }
struct buffer {
    void * p = nullptr;
    size_t bytes = 0;
    ~buffer() { if (p) cudaFree(p); }
    void reserve(size_t n) {
        if (n <= bytes) return;
        if (p) { check(cudaFree(p)); p = nullptr; bytes = 0; }
        check(cudaMalloc(&p, n)); bytes = n;
    }
};
struct statistics { float amax; int invalid; };
struct weight { buffer data; int64_t k, n; ggml_type type; };
struct runtime {
    std::mutex mutex;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    buffer source, activation, output, stats;
    std::unordered_map<const void *, std::unique_ptr<weight>> cache;
    uint64_t attempted = 0, quantized_weights = 0, quantized_weight_elements = 0;
    bool failed = false, emitted = false;
    void init() {
        if (stream) return;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cublasCreate(&blas)); check(cublasSetStream(blas, stream));
        check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
        stats.reserve(sizeof(statistics));
        int device; cudaDeviceProp prop{};
        check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&prop, device));
        std::fprintf(stderr, "NVFP4_DEVICE name=%s sm=%d%d implementation=fake_dequant_f16 block=16 global=dynamic_tensor_amax rounding=RNE\n", prop.name, prop.major, prop.minor);
    }
    ~runtime() { if (blas) cublasDestroy(blas); if (stream) cudaStreamDestroy(stream); }
};
runtime & state() { static runtime r; return r; }
template<class T> __device__ float read_value(const T * x, size_t i) { return float(x[i]); }
template<> __device__ float read_value(const __half * x, size_t i) { return __half2float(x[i]); }
template<class T> __global__ void tensor_amax(const T * x, size_t count, statistics * stats) {
    __shared__ float maxima[256];
    float m = 0;
    for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < count; i += size_t(gridDim.x) * 256) {
        float v = read_value(x, i);
        if (!isfinite(v)) atomicExch(&stats->invalid, 1);
        else m = fmaxf(m, fabsf(v));
    }
    maxima[threadIdx.x] = m; __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
        __syncthreads();
    }
    if (!threadIdx.x) atomicMax(reinterpret_cast<unsigned *>(&stats->amax), __float_as_uint(maxima[0]));
}
template<class T> __global__ void fake_quantize(const T * x, __half * out, size_t count, statistics * stats) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned mask = __activemask();
    float value = read_value(x, i), m = fabsf(value);
    for (int d = 8; d; d /= 2) m = fmaxf(m, __shfl_xor_sync(mask, m, d, 16));
    // All-zero tensors have an explicit zero-preserving extension of the reference.
    float global = stats->amax == 0 ? 1.0f : nvfp4_div(stats->amax, 2688.0f);
    if (!(global > 0) || !isfinite(global)) { atomicExch(&stats->invalid, 2); return; }
    float scale = 0;
    if (!(threadIdx.x & 15)) scale = nvfp4_block_scale(m, global);
    scale = __shfl_sync(mask, scale, (threadIdx.x % 32) & ~15, 32);
    float decode = scale * global;
    float q = nvfp4_value(nvfp4_code(nvfp4_div(value, decode)));
    out[i] = __float2half_rn(q * decode);
}
template<class T> void quantize(runtime & r, const T * source, __half * dest, size_t count) {
    check(cudaMemsetAsync(r.stats.p, 0, sizeof(statistics), r.stream));
    auto stats = static_cast<statistics *>(r.stats.p);
    tensor_amax<<<std::min<size_t>((count + 255) / 256, 4096), 256, 0, r.stream>>>(source, count, stats);
    fake_quantize<<<(count + 255) / 256, 256, 0, r.stream>>>(source, dest, count, stats);
    check(cudaGetLastError());
}
void validate(runtime & r, const statistics & stats) {
    if (stats.invalid) { r.failed = true; throw std::runtime_error("NVFP4 rejects non-finite inputs or unrepresentable global scale"); }
}
const void * weights(runtime & r, const ggml_tensor * w) {
    auto it = r.cache.find(w->data);
    if (it != r.cache.end()) {
        if (it->second->k != w->ne[0] || it->second->n != w->ne[1] || it->second->type != w->type)
            throw std::runtime_error("NVFP4 cached weight identity changed");
        return it->second->data.p;
    }
    auto item = std::make_unique<weight>(); item->k = w->ne[0]; item->n = w->ne[1]; item->type = w->type;
    size_t count = ggml_nelements(w);
    item->data.reserve(count * sizeof(__half)); r.source.reserve(ggml_nbytes(w));
    check(cudaMemcpyAsync(r.source.p, w->data, ggml_nbytes(w), cudaMemcpyHostToDevice, r.stream));
    if (w->type == GGML_TYPE_F16) quantize(r, static_cast<const __half *>(r.source.p), static_cast<__half *>(item->data.p), count);
    else quantize(r, static_cast<const float *>(r.source.p), static_cast<__half *>(item->data.p), count);
    statistics stats{}; check(cudaMemcpyAsync(&stats, r.stats.p, sizeof(stats), cudaMemcpyDeviceToHost, r.stream));
    check(cudaStreamSynchronize(r.stream)); validate(r, stats);
    std::fprintf(stderr, "NVFP4_WEIGHT name=%s elements=%zu amax=%.9g tensor_scale=%.9g\n", w->name, count, stats.amax, stats.amax == 0 ? 1.0f : stats.amax / 2688.0f);
    ++r.quantized_weights; r.quantized_weight_elements += count;
    const void * result = item->data.p; r.cache.emplace(w->data, std::move(item)); return result;
}
}

bool ggml_cuda_rtn_compute(ggml_tensor * op) {
    using namespace nvfp4_impl;
    static auto base = original<bool (*)(ggml_tensor *)>("_Z21ggml_cuda_rtn_computeP11ggml_tensor");
    if (!enabled() || (op && op->src[1] && (op->src[1]->ne[2] > 1 || op->src[1]->ne[3] > 1))) return base(op);
    error.clear(); auto & r = state(); std::lock_guard<std::mutex> lock(r.mutex); ++r.attempted;
    try {
        if (!op || op->op != GGML_OP_MUL_MAT || !op->src[0] || !op->src[1] || !op->data)
            throw std::runtime_error("NVFP4 invalid dense operation");
        const auto * w = op->src[0]; const auto * x = op->src[1];
        if ((w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_F32) || x->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
            !w->data || !x->data || !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(op) ||
            w->ne[0] != x->ne[0] || w->ne[0] % 16 || w->ne[2] != 1 || w->ne[3] != 1)
            throw std::runtime_error("NVFP4 requires contiguous FP16/FP32 model weights and FP32 dense inputs, K divisible by 16");
        r.init(); int k = w->ne[0], n = w->ne[1], m = x->ne[1];
        const void * weight = weights(r, w);
        size_t count = size_t(k) * m;
        r.source.reserve(count * sizeof(float)); r.activation.reserve(count * sizeof(__half)); r.output.reserve(ggml_nbytes(op));
        check(cudaMemcpyAsync(r.source.p, x->data, count * sizeof(float), cudaMemcpyHostToDevice, r.stream));
        quantize(r, static_cast<const float *>(r.source.p), static_cast<__half *>(r.activation.p), count);
        const float alpha = 1, beta = 0;
        check(cublasGemmEx(r.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, weight, CUDA_R_16F, k,
                          r.activation.p, CUDA_R_16F, k, &beta, r.output.p, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        statistics stats{};
        check(cudaMemcpyAsync(op->data, r.output.p, ggml_nbytes(op), cudaMemcpyDeviceToHost, r.stream));
        check(cudaMemcpyAsync(&stats, r.stats.p, sizeof(stats), cudaMemcpyDeviceToHost, r.stream));
        check(cudaStreamSynchronize(r.stream)); validate(r, stats); ++dense_calls; return true;
    } catch (const std::exception & e) { r.failed = true; if (r.stream) cudaStreamSynchronize(r.stream); error = e.what(); return false; }
}
uint64_t ggml_cuda_rtn_dense_calls() {
    static auto base = nvfp4_impl::original<uint64_t (*)()>("_Z25ggml_cuda_rtn_dense_callsv");
    return base() + nvfp4_impl::dense_calls.load();
}
const char * ggml_cuda_rtn_error() {
    static auto base = nvfp4_impl::original<const char * (*)()>("_Z19ggml_cuda_rtn_errorv");
    return nvfp4_impl::error.empty() ? base() : nvfp4_impl::error.c_str();
}
void ggml_cuda_rtn_clear() {
    using namespace nvfp4_impl;
    static auto base = original<void (*)()>("_Z19ggml_cuda_rtn_clearv");
    if (enabled()) {
        auto & r = state(); std::lock_guard<std::mutex> lock(r.mutex);
        if (r.stream) check(cudaStreamSynchronize(r.stream));
        uint64_t calls = dense_calls.load();
        if (r.attempted && !r.emitted) {
            std::fprintf(stderr,
                "NVFP4_PROOF {\"schema\":\"nvfp4-fake-v1\",\"complete\":%s,\"format\":\"NVFP4\",\"weight_bits\":4,\"activation_bits\":4,\"attention_bits\":16,\"block_size\":16,\"scale_format\":\"E4M3FN\",\"tensor_scale\":\"FP32\",\"rounding\":\"RNE\",\"global_scaling\":\"dynamic_tensor_amax\",\"weight_scaling\":\"tensor_amax_once\",\"native_fp4\":false,\"gemm\":\"dequant_f16_fp32_acc\",\"dense_gpu_calls\":%llu,\"verified_dense_calls\":%llu,\"attempted_dense_calls\":%llu,\"attention_gpu_calls\":%llu,\"quantized_weight_tensors\":%llu,\"quantized_weight_elements\":%llu,\"embedding\":\"original_fp16\",\"lm_head\":\"NVFP4\",\"zero_tensor\":\"zero_preserving_extension\",\"nonfinite\":\"reject\"}\n",
                !r.failed && calls && calls == r.attempted ? "true" : "false", (unsigned long long)calls, (unsigned long long)calls,
                (unsigned long long)r.attempted, (unsigned long long)ggml_cuda_rtn_attention_calls(),
                (unsigned long long)r.quantized_weights, (unsigned long long)r.quantized_weight_elements);
            r.emitted = true;
        }
        r.cache.clear();
    }
    base();
}

// Test-only entry point; no model or GPU is touched unless explicitly invoked.
extern "C" int nvfp4_gpu_reference(const float * input, size_t count, float * out) {
    using namespace nvfp4_impl;
    if (!input || !out || !count || count % 16) return 1;
    try {
        runtime r; r.init(); buffer source, dest;
        source.reserve(count * sizeof(float)); dest.reserve(count * sizeof(__half));
        check(cudaMemcpyAsync(source.p, input, count * sizeof(float), cudaMemcpyHostToDevice, r.stream));
        quantize(r, static_cast<const float *>(source.p), static_cast<__half *>(dest.p), count);
        std::unique_ptr<__half[]> half(new __half[count]); statistics stats{};
        check(cudaMemcpyAsync(half.get(), dest.p, count * sizeof(__half), cudaMemcpyDeviceToHost, r.stream));
        check(cudaMemcpyAsync(&stats, r.stats.p, sizeof(stats), cudaMemcpyDeviceToHost, r.stream));
        check(cudaStreamSynchronize(r.stream)); validate(r, stats);
        for (size_t i = 0; i < count; ++i) out[i] = __half2float(half[i]);
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "NVFP4 reference test: %s\n", e.what()); return 2; }
}
