#include "ggml-cuda-potal-attention.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error_text;
void require(bool valid, const char * message) { if (!valid) throw std::invalid_argument(message); }
void check(cudaError_t result) { if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result)); }
void check(cublasStatus_t result) {
    if (result != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("PoTal attention cuBLAS status " + std::to_string(result));
}
struct buffer {
    void * data = nullptr;
    size_t bytes = 0;
    ~buffer() { if (data) cudaFree(data); }
    void reserve(size_t wanted) {
        if (wanted <= bytes) return;
        void * next = nullptr;
        check(cudaMalloc(&next, wanted));
        if (data) cudaFree(data);
        data = next; bytes = wanted;
    }
    void upload(const void * source, size_t wanted) {
        reserve(wanted);
        if (wanted) check(cudaMemcpy(data, source, wanted, cudaMemcpyHostToDevice));
    }
};
struct runtime {
    std::mutex mutex;
    buffer source, activations, maps, packed_a, packed_w, dots, output, error;
    cublasHandle_t blas = nullptr;
    ggml_cuda_potal_attention_stats stats{};
    ~runtime() { if (blas) cublasDestroy(blas); }
};
runtime & state() { static runtime value; return value; }

__global__ void prepare_weights(const int32_t * input, const uint32_t * carriers,
    int8_t * linear, int8_t * packed, unsigned columns, unsigned padded_columns,
    unsigned k, unsigned bits, unsigned * error) {
    const size_t index = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index >= size_t(padded_columns)*k) return;
    const unsigned lane = index%32, column = (index/32)%padded_columns;
    const unsigned block = index/(32*padded_columns);
    int value = 0;
    if (column < columns) {
        const size_t source = size_t(block*32+lane)*columns+column;
        value = input[source];
        const int bound = 1 << (bits-1);
        if (value < -bound || value >= bound) atomicExch(error, 1u);
        linear[source] = int8_t(value);
        const unsigned carrier = carriers[size_t(block)*columns+column];
        if (!lane && carrier > 32767u && carrier != 0x80000000u) atomicExch(error, 1u);
    }
    packed[index] = int8_t(value);
}

__global__ void pack_activations(const int8_t * input, int8_t * packed,
    const ggml_cuda_potal_attention_fragment * maps, unsigned rows, unsigned padded_rows,
    unsigned stride, unsigned fragments) {
    const size_t index = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index >= size_t(fragments)*padded_rows*32) return;
    const unsigned lane = index%32, row = (index/32)%padded_rows, fragment = index/(32*padded_rows);
    const unsigned count = maps ? maps[fragment].count : 32;
    const unsigned begin = maps ? maps[fragment].compact_begin : fragment*32;
    packed[index] = row < rows && lane < count ? input[size_t(row)*stride+begin+lane] : 0;
}

__global__ void gather_weights(const int8_t * input, int8_t * packed,
    const ggml_cuda_potal_attention_fragment * maps, unsigned columns, unsigned padded_columns,
    unsigned original_columns, unsigned column_begin, unsigned fragments) {
    const size_t index = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index >= size_t(fragments)*padded_columns*32) return;
    const unsigned lane = index%32, column = (index/32)%padded_columns;
    const unsigned fragment = index/(32*padded_columns);
    const unsigned count = maps ? maps[fragment].count : 32;
    const unsigned k = maps ? maps[fragment].original_k[lane] : fragment*32+lane;
    packed[index] = column < columns && lane < count
        ? input[size_t(k)*original_columns+column_begin+column] : 0;
}

__device__ int sat32(long long value) {
    return value > INT32_MAX ? INT32_MAX : value < INT32_MIN ? INT32_MIN : int(value);
}
__device__ int shift_fragment(int value, unsigned carrier) {
    if (!value || carrier == 0x80000000u) return 0;
    if (carrier >= 32) return value < 0 ? INT32_MIN : INT32_MAX;
    return sat32(static_cast<long long>(value)*(1LL << carrier));
}
__global__ void accumulate(const int32_t * dots, const uint32_t * carriers,
    const ggml_cuda_potal_attention_fragment * maps, int32_t * output,
    unsigned rows, unsigned columns, unsigned padded_rows, unsigned padded_columns,
    unsigned original_columns, unsigned column_begin, unsigned fragments) {
    const size_t index = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (index >= size_t(rows)*columns) return;
    const unsigned row = index/columns, column = index%columns;
    int value = 0;
    // A full-K GEMM changes saturation. Keep one exact GEMM per K32 fragment,
    // then perform the same shift and saturating accumulation in original order.
    for (unsigned f = 0; f < fragments; ++f) {
        const unsigned original_block = maps ? maps[f].original_block : f;
        const unsigned carrier = carriers[size_t(original_block)*original_columns+column_begin+column];
        const int raw = dots[(size_t(f)*padded_rows+row)*padded_columns+column];
        value = sat32(static_cast<long long>(value)+shift_fragment(raw, carrier));
    }
    output[index] = value;
}
} // namespace

struct ggml_cuda_potal_attention_weights {
    buffer low, high, packed_low, packed_high, carriers;
    unsigned columns, padded_columns, k;
    bool has_high;
};

bool ggml_cuda_potal_attention_enabled() {
    const char * value = std::getenv("GGML_GEMMINI_CUDA_POTAL_ATTENTION");
    return value && std::strcmp(value, "1") == 0;
}

ggml_cuda_potal_attention_weights * ggml_cuda_potal_attention_create(
    const int32_t * low, const int32_t * high, const uint32_t * carriers,
    size_t columns, size_t k, unsigned bits) {
    try {
        require(low && carriers && columns && columns <= INT_MAX-16 && k && k <= INT_MAX && k%32 == 0 &&
                (bits == 4 || bits == 8), "Invalid CUDA attention weights");
        auto & s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        auto weights = std::make_unique<ggml_cuda_potal_attention_weights>();
        weights->columns = unsigned(columns); weights->padded_columns = unsigned((columns+15)/16*16);
        weights->k = unsigned(k); weights->has_high = high != nullptr;
        weights->carriers.upload(carriers, k/32*columns*sizeof(uint32_t));
        s.error.reserve(sizeof(unsigned));
        check(cudaMemset(s.error.data, 0, sizeof(unsigned)));
        auto prepare = [&](const int32_t * input, buffer & linear, buffer & packed) {
            s.source.upload(input, k*columns*sizeof(int32_t));
            linear.reserve(k*columns);
            packed.reserve(k*weights->padded_columns);
            prepare_weights<<<unsigned((k*weights->padded_columns+255)/256),256>>>(
                static_cast<const int32_t *>(s.source.data), static_cast<const uint32_t *>(weights->carriers.data),
                static_cast<int8_t *>(linear.data), static_cast<int8_t *>(packed.data), unsigned(columns),
                weights->padded_columns, unsigned(k), bits, static_cast<unsigned *>(s.error.data));
            check(cudaGetLastError());
        };
        prepare(low, weights->low, weights->packed_low);
        if (high) prepare(high, weights->high, weights->packed_high);
        unsigned error = 0;
        check(cudaMemcpy(&error, s.error.data, sizeof(error), cudaMemcpyDeviceToHost));
        require(!error, "Invalid signed attention codes or SCU carriers");
        error_text.clear();
        return weights.release();
    } catch (const std::exception & error) { error_text = error.what(); return nullptr; }
}

void ggml_cuda_potal_attention_destroy(ggml_cuda_potal_attention_weights * weights) { delete weights; }

bool ggml_cuda_potal_attention_dot(const ggml_cuda_potal_attention_weights * w, unsigned pass,
    const int8_t * activations, size_t rows, size_t stride, size_t column_begin, size_t columns,
    const ggml_cuda_potal_attention_fragment * maps, size_t fragments, bool residual, int32_t * output) {
    try {
        require(w && activations && output && rows && rows <= INT_MAX-16 && columns &&
                column_begin < w->columns && columns <= w->columns-column_begin && stride <= INT_MAX &&
                pass < (w->has_high ? 2u : 1u), "Invalid CUDA attention dot");
        if (!maps) { require(stride >= w->k, "Short attention activation stride"); fragments = w->k/32; }
        require(fragments && fragments <= INT_MAX, "Invalid attention fragments");
        if (maps) for (size_t f = 0; f < fragments; ++f) {
            require(maps[f].count && maps[f].count <= 32 && maps[f].compact_begin <= stride &&
                    maps[f].count <= stride-maps[f].compact_begin && maps[f].original_block < w->k/32,
                    "Invalid compact attention fragment");
            for (unsigned lane = 0; lane < maps[f].count; ++lane)
                require(maps[f].original_k[lane] < w->k && maps[f].original_k[lane]/32 == maps[f].original_block,
                        "Invalid original attention K index");
        }
        auto & s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!s.blas) check(cublasCreate(&s.blas));
        const unsigned pr = unsigned((rows+15)/16*16), pc = unsigned((columns+15)/16*16);
        s.activations.upload(activations, rows*stride);
        const ggml_cuda_potal_attention_fragment * device_maps = nullptr;
        if (maps) {
            s.maps.upload(maps, fragments*sizeof(*maps));
            device_maps = static_cast<const ggml_cuda_potal_attention_fragment *>(s.maps.data);
        }
        s.packed_a.reserve(fragments*pr*32);
        s.dots.reserve(fragments*pr*pc*sizeof(int32_t));
        s.output.reserve(rows*columns*sizeof(int32_t));
        pack_activations<<<unsigned((fragments*pr*32+255)/256),256>>>(
            static_cast<const int8_t *>(s.activations.data), static_cast<int8_t *>(s.packed_a.data),
            device_maps, unsigned(rows), pr, unsigned(stride), unsigned(fragments));
        const auto * packed_weights = static_cast<const int8_t *>(pass ? w->packed_high.data : w->packed_low.data);
        long long weight_batch_stride = static_cast<long long>(w->padded_columns)*32;
        if (!maps && column_begin+pc <= w->padded_columns) {
            // The stationary K32 layout already contains this column window.
            // Padded output columns may read neighboring weights, but are discarded.
            // Keep the original fragment stride instead of gathering every stripe.
            packed_weights += column_begin*32;
        } else {
            s.packed_w.reserve(fragments*pc*32);
            gather_weights<<<unsigned((fragments*pc*32+255)/256),256>>>(
                static_cast<const int8_t *>(pass ? w->high.data : w->low.data),
                static_cast<int8_t *>(s.packed_w.data), device_maps, unsigned(columns), pc,
                w->columns, unsigned(column_begin), unsigned(fragments));
            packed_weights = static_cast<const int8_t *>(s.packed_w.data);
            weight_batch_stride = static_cast<long long>(pc)*32;
        }
        check(cudaGetLastError());
        const int32_t alpha = 1, beta = 0;
        check(cublasGemmStridedBatchedEx(s.blas, CUBLAS_OP_T, CUBLAS_OP_N, pc, pr, 32, &alpha,
            packed_weights, CUDA_R_8I, 32, weight_batch_stride,
            s.packed_a.data, CUDA_R_8I, 32, static_cast<long long>(pr)*32, &beta,
            s.dots.data, CUDA_R_32I, pc, static_cast<long long>(pc)*pr, int(fragments),
            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        accumulate<<<unsigned((rows*columns+255)/256),256>>>(
            static_cast<const int32_t *>(s.dots.data), static_cast<const uint32_t *>(w->carriers.data),
            device_maps, static_cast<int32_t *>(s.output.data), unsigned(rows), unsigned(columns), pr, pc,
            w->columns, unsigned(column_begin), unsigned(fragments));
        check(cudaGetLastError());
        check(cudaMemcpy(output, s.output.data, rows*columns*sizeof(int32_t), cudaMemcpyDeviceToHost));
        ++(residual ? s.stats.residual_dot_calls : s.stats.main_dot_calls);
        s.stats.gemm_batches += fragments;
        s.stats.fragments += rows*columns*fragments;
        (residual ? s.stats.residual_fragments : s.stats.main_fragments) += rows*columns*fragments;
        error_text.clear();
        return true;
    } catch (const std::exception & error) { error_text = error.what(); return false; }
}

ggml_cuda_potal_attention_stats ggml_cuda_potal_attention_get_stats() {
    auto & s = state(); std::lock_guard<std::mutex> lock(s.mutex); return s.stats;
}
const char * ggml_cuda_potal_attention_error() { return error_text.c_str(); }
