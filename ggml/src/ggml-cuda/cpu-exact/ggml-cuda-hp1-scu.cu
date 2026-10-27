#include "ggml-cuda-hp1-scu.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error_text;
void check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void check(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("HP1 cuBLAS status " + std::to_string(s));
}
void require(bool b, const char * s) { if (!b) throw std::invalid_argument(s); }
struct buffer {
    void * p = nullptr; size_t size = 0;
    ~buffer() { if (p) cudaFree(p); }
    void reserve(size_t n) {
        if (n <= size) return;
        void * q = nullptr; check(cudaMalloc(&q, n));
        if (p) cudaFree(p);
        p = q; size = n;
    }
    void upload(const void * q, size_t n) {
        reserve(n); if (n) check(cudaMemcpy(p, q, n, cudaMemcpyHostToDevice));
    }
};
struct device_stats { unsigned long long fragments, shifts, adds; unsigned error; };
struct runtime {
    std::mutex mutex;
    buffer a, w, offsets, events, c, counters;
    buffer packed_a, packed_w, exponents, fragments;
    std::map<uintptr_t,std::unique_ptr<buffer>> weights;
    size_t weight_bytes = 0;
    uint64_t weight_hits = 0;
    cublasHandle_t blas = nullptr;
    ggml_hp1_scu_stats totals{};
    ~runtime() { if (blas) cublasDestroy(blas); }
    void prepare_gemm() { if (!blas) check(cublasCreate(&blas)); }
};
runtime & state() { static runtime s; return s; }
const void * cached_weights(runtime & s, const void * source, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(source);
    auto entry = s.weights.upper_bound(address);
    if (entry == s.weights.begin()) return nullptr;
    --entry;
    const size_t offset = address-entry->first;
    if (offset > entry->second->size || bytes > entry->second->size-offset) return nullptr;
    ++s.weight_hits;
    return static_cast<const uint8_t *>(entry->second->p)+offset;
}
const void * device_weights(runtime & s, const void * source, size_t bytes) {
    if (const auto * p = cached_weights(s,source,bytes)) return p;
    s.w.upload(source,bytes);
    return s.w.p;
}
__device__ int code(const uint8_t * b, unsigned bits, unsigned k) {
    return bits == 4 ? int((b[k % 16] >> (k < 16 ? 0 : 4)) & 15) - 8 : int(int8_t(b[k]));
}
__device__ int exponent(const uint8_t * b, unsigned bits, device_stats * stats) {
    const unsigned p = bits == 4 ? 16 : 32;
    const int e = int16_t(unsigned(b[p]) | (unsigned(b[p + 1]) << 8));
    const unsigned s = unsigned(b[p+4]) | (unsigned(b[p+5]) << 8) |
        (unsigned(b[p+6]) << 16) | (unsigned(b[p+7]) << 24);
    if (b[p+2] || b[p+3] || (s & 0x7f800000u) == 0x7f800000u || (e < 0 && e != INT16_MIN))
        atomicExch(&stats->error, 1u);
    return e;
}
__device__ int sat(long long x, unsigned long long & hits) {
    if (x > INT32_MAX) { ++hits; return INT32_MAX; }
    if (x < INT32_MIN) { ++hits; return INT32_MIN; }
    return int(x);
}
__device__ int scu(int raw, int e, unsigned long long & hits) {
    if (!raw || e == INT16_MIN || e < 0) return 0;
    if (e >= 32) { ++hits; return raw < 0 ? INT32_MIN : INT32_MAX; }
    return sat(static_cast<long long>(raw) * (1LL << e), hits);
}

__global__ void pack_activation_blocks(const int32_t * a, int8_t * packed,
    unsigned rows, unsigned padded_rows, unsigned k) {
    const size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= size_t(padded_rows)*k) return;
    const unsigned t = i%32, r = (i/32)%padded_rows, b = i/(32*padded_rows);
    packed[i] = r < rows ? int8_t(a[size_t(r)*k+b*32+t]) : 0;
}

__global__ void pack_weight_blocks(const uint8_t * w, int8_t * packed, int * exponents,
    unsigned cols, unsigned padded_cols, unsigned k, unsigned bits, device_stats * stats, bool validate) {
    const size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= size_t(padded_cols)*k) return;
    const unsigned t = i%32, c = (i/32)%padded_cols, b = i/(32*padded_cols);
    if (c < cols) {
        const auto * block = w+(size_t(c)*(k/32)+b)*(bits == 4 ? 24 : 40);
        packed[i] = int8_t(code(block,bits,t));
        if (!t) exponents[size_t(b)*padded_cols+c] = validate ? exponent(block,bits,stats) : 0;
    } else {
        packed[i] = 0;
        if (!t) exponents[size_t(b)*padded_cols+c] = INT16_MIN;
    }
}

// GEMM computes exact 32-element integer fragments. SCU still saturates each
// fragment and accumulates in the original K order; a single full-K GEMM cannot.
__global__ void accumulate_fragments(const int32_t * dots, const int * exponents, int32_t * out,
    unsigned rows, unsigned cols, unsigned padded_rows, unsigned padded_cols,
    unsigned blocks, device_stats * stats) {
    const size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= size_t(rows)*cols) return;
    const unsigned r = i/cols, c = i%cols;
    int acc = 0;
    unsigned long long shifts = 0, adds = 0;
    for (unsigned b = 0; b < blocks; ++b) {
        const int raw = dots[(size_t(b)*padded_rows+r)*padded_cols+c];
        acc = sat(static_cast<long long>(acc)+scu(raw,exponents[size_t(b)*padded_cols+c],shifts),adds);
    }
    out[i] = acc;
    atomicAdd(&stats->fragments,static_cast<unsigned long long>(blocks));
    if (shifts) atomicAdd(&stats->shifts,shifts);
    if (adds) atomicAdd(&stats->adds,adds);
}

void gemm_main(runtime & s, const void * weights, unsigned rows, unsigned cols, unsigned k, unsigned bits) {
    s.prepare_gemm();
    const unsigned pr = (rows+15)/16*16, pc = (cols+15)/16*16, blocks = k/32;
    s.packed_a.reserve(size_t(pr)*k);
    s.packed_w.reserve(size_t(pc)*k);
    s.exponents.reserve(size_t(pc)*blocks*sizeof(int));
    s.fragments.reserve(size_t(pr)*pc*blocks*sizeof(int32_t));
    pack_activation_blocks<<<unsigned((size_t(pr)*k+255)/256),256>>>(
        static_cast<const int32_t *>(s.a.p),static_cast<int8_t *>(s.packed_a.p),rows,pr,k);
    pack_weight_blocks<<<unsigned((size_t(pc)*k+255)/256),256>>>(
        static_cast<const uint8_t *>(weights),static_cast<int8_t *>(s.packed_w.p),
        static_cast<int *>(s.exponents.p),cols,pc,k,bits,static_cast<device_stats *>(s.counters.p),true);
    check(cudaGetLastError());
    const int32_t alpha = 1, beta = 0;
    check(cublasGemmStridedBatchedEx(s.blas,CUBLAS_OP_T,CUBLAS_OP_N,pc,pr,32,&alpha,
        s.packed_w.p,CUDA_R_8I,32,static_cast<long long>(pc)*32,
        s.packed_a.p,CUDA_R_8I,32,static_cast<long long>(pr)*32,&beta,
        s.fragments.p,CUDA_R_32I,pc,static_cast<long long>(pc)*pr,blocks,
        CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    accumulate_fragments<<<unsigned((size_t(rows)*cols+255)/256),256>>>(
        static_cast<const int32_t *>(s.fragments.p),static_cast<const int *>(s.exponents.p),
        static_cast<int32_t *>(s.c.p),rows,cols,pr,pc,blocks,static_cast<device_stats *>(s.counters.p));
}

// DIM=B_K=32: each dp4a reduction is one hardware fragment. Preserve K order.
__global__ void main_kernel(const int32_t * a, const uint8_t * w, int32_t * out,
    unsigned rows, unsigned cols, unsigned k, unsigned bits, device_stats * stats) {
    __shared__ int av[8][8], wv[16][8];
    const unsigned t = threadIdx.y * 16 + threadIdx.x;
    const unsigned r = blockIdx.y * 8 + threadIdx.y, c = blockIdx.x * 16 + threadIdx.x;
    const unsigned stride = bits == 4 ? 24 : 40;
    int acc = 0;
    unsigned long long shifts = 0, adds = 0;
    for (unsigned b = 0; b < k / 32; ++b) {
        for (unsigned item = t; item < 24 * 8; item += 128) {
            const unsigned local = item / 8, group = item % 8;
            unsigned packed = 0;
            for (unsigned lane = 0; lane < 4; ++lane) {
                const unsigned pos = b * 32 + group * 4 + lane;
                int v = 0;
                if (local < 8) {
                    const unsigned row = blockIdx.y * 8 + local;
                    if (row < rows) v = a[size_t(row) * k + pos];
                } else {
                    const unsigned col = blockIdx.x * 16 + local - 8;
                    if (col < cols) v = code(w + (size_t(col) * (k / 32) + b) * stride, bits, pos % 32);
                }
                packed |= unsigned(uint8_t(v)) << (lane * 8);
            }
            if (local < 8) av[local][group] = int(packed); else wv[local-8][group] = int(packed);
        }
        __syncthreads();
        int raw = 0;
#pragma unroll
        for (unsigned g = 0; g < 8; ++g) raw = __dp4a(av[threadIdx.y][g], wv[threadIdx.x][g], raw);
        if (r < rows && c < cols) {
            const int e = exponent(w + (size_t(c) * (k/32) + b) * stride, bits, stats);
            const int scaled = scu(raw, e, shifts);
            acc = sat(static_cast<long long>(acc) + scaled, adds);
        }
        __syncthreads();
    }
    if (r < rows && c < cols) {
        out[size_t(r)*cols+c] = acc;
        atomicAdd(&stats->fragments, static_cast<unsigned long long>(k/32));
        if (shifts) atomicAdd(&stats->shifts, shifts);
        if (adds) atomicAdd(&stats->adds, adds);
    }
}

// Native run-aware execution keeps one INT32 accumulator per lane across K.
// At DIM32 the compact union has <=32 positions, hence one fragment per block/lane.
// Accumulate reconstruction in 128 bits (two limbs) before checking INT64, so
// temporary overflow followed by cancellation agrees with the native oracle.
__global__ void residual_kernel(const uint8_t * w, const uint32_t * offsets,
    const ggml_cuda_residual_event * events, int64_t * out, unsigned rows, unsigned cols,
    unsigned k, unsigned bits, device_stats * stats) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(rows)*cols) return;
    const unsigned r = index/cols, c = index%cols, stride = bits == 4 ? 24 : 40;
    unsigned long long low = 0, shifts = 0, adds = 0, fragments = 0;
    long long high = 0;
    int accumulators[9] = {};
    for (unsigned begin = offsets[r]; begin < offsets[r+1];) {
        const unsigned b = events[begin].k/32;
        const auto * wb = w + (size_t(c)*(k/32)+b)*stride;
        int raw[9] = {};
        unsigned used = 0, end = begin;
        while (end < offsets[r+1] && events[end].k/32 == b) {
            long long value = events[end].value;
            const int weight = code(wb,bits,events[end].k%32);
            unsigned lane = 0;
            while (value) {
                const int digit = int((static_cast<unsigned long long>(value) + (1u << (bits-1))) & ((1u << bits)-1)) - (1 << (bits-1));
                raw[lane] += digit*weight;
                if (digit) used |= 1u << lane;
                // Balanced digits make the numerator exactly divisible by 2^bits.
                value = (value-digit) >> bits;
                ++lane;
            }
            ++end;
        }
        const int e = exponent(wb,bits,stats);
        for (unsigned lane=0; lane<=32/bits; ++lane) if (used & (1u << lane)) {
            const int scaled = scu(raw[lane],e,shifts);
            accumulators[lane] = sat(static_cast<long long>(accumulators[lane]) + scaled, adds);
            ++fragments;
        }
        begin = end;
    }
    for (unsigned lane=0; lane<=32/bits; ++lane) {
        const long long term = static_cast<long long>(accumulators[lane])*(1LL << (lane*bits));
        const unsigned long long old = low;
        low += static_cast<unsigned long long>(term);
        high += (term < 0 ? -1 : 0) + (low < old ? 1 : 0);
    }
    if (!((high == 0 && low <= INT64_MAX) || (high == -1 && low >= (1ULL << 63))))
        atomicExch(&stats->error, 2u);
    out[index] = static_cast<int64_t>(low);
    if (fragments) atomicAdd(&stats->fragments,fragments);
    if (shifts) atomicAdd(&stats->shifts,shifts);
    if (adds) atomicAdd(&stats->adds,adds);
}
void shape(unsigned bits,size_t rows,size_t cols,size_t k) {
    require((bits==4 || bits==8) && rows && cols && k && k%32==0 &&
        rows<=128 && cols<=UINT32_MAX && k<=UINT32_MAX && cols<=SIZE_MAX/k &&
        rows<=SIZE_MAX/cols && cols*(k/32)<=SIZE_MAX/(bits==4?24:40) &&
        rows*cols<=SIZE_MAX/sizeof(int64_t) && rows*k<=SIZE_MAX/sizeof(int32_t) &&
        rows*cols <= size_t(INT_MAX)*128,"Invalid HP1 SCU shape");
}
device_stats finish(runtime & s) {
    check(cudaGetLastError());
    device_stats stats{};
    check(cudaMemcpy(&stats,s.counters.p,sizeof(stats),cudaMemcpyDeviceToHost));
    return stats;
}
}
bool ggml_hp1_scu_enabled() {
    const char * v=std::getenv("GGML_GEMMINI_HP1_SCU"); return v && std::strcmp(v,"1")==0;
}
bool ggml_cuda_hp1_scu_cache_weights(const void * weights, size_t bytes) {
    auto & s = state(); std::lock_guard<std::mutex> lock(s.mutex);
    if (!weights || !bytes) return false;
    if (cached_weights(s,weights,bytes)) return true;
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free,&total) != cudaSuccess) return false;
    if (bytes > (size_t{2} << 30)-s.weight_bytes || bytes > free/2) return false;
    try {
        auto copy = std::make_unique<buffer>();
        copy->upload(weights,bytes);
        const auto inserted = s.weights.emplace(reinterpret_cast<uintptr_t>(weights),std::move(copy));
        if (inserted.second) s.weight_bytes += bytes;
        return inserted.second;
    } catch (const std::exception &) {
        cudaGetLastError();
        return false;
    }
}
void ggml_cuda_hp1_scu_clear_weights() {
    auto & s = state(); std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.weights.empty()) std::fprintf(stderr,"HP1_SCU_WEIGHT_CACHE bytes=%zu tensors=%zu hits=%llu\n",
        s.weight_bytes,s.weights.size(),static_cast<unsigned long long>(s.weight_hits));
    s.weights.clear(); s.weight_bytes = 0; s.weight_hits = 0;
}
const char * ggml_cuda_hp1_scu_error() { return error_text.c_str(); }
ggml_hp1_scu_stats ggml_cuda_hp1_scu_stats() {
    auto & s=state(); std::lock_guard<std::mutex> lock(s.mutex); return s.totals;
}
bool ggml_cuda_hp1_scu_main(const int32_t * a,const void * w,unsigned bits,
    size_t rows,size_t cols,size_t k,int32_t * out) {
    auto & s=state(); std::lock_guard<std::mutex> lock(s.mutex); error_text.clear();
    try {
        shape(bits,rows,cols,k);
        require(a && w && out && cols<=GGML_CUDA_CPU_EXACT_MAX_COLUMNS,"Invalid HP1 SCU pointers/tile");
        const int limit=1 << (bits-1);
        for (size_t i=0;i<rows*k;++i) require(a[i]>=-limit && a[i]<limit,"Main code exceeds operand width");
        s.a.upload(a,rows*k*sizeof(*a));
        const void * weights = device_weights(s,w,cols*(k/32)*(bits==4?24:40));
        s.c.reserve(rows*cols*sizeof(*out)); s.counters.reserve(sizeof(device_stats));
        check(cudaMemset(s.counters.p,0,sizeof(device_stats)));
        if (rows >= 8 && cols >= 16) {
            gemm_main(s,weights,rows,cols,k,bits);
        } else {
            main_kernel<<<dim3((cols+15)/16,(rows+7)/8),dim3(16,8)>>>(
                static_cast<int32_t *>(s.a.p),static_cast<const uint8_t *>(weights),static_cast<int32_t *>(s.c.p),
                rows,cols,k,bits,static_cast<device_stats *>(s.counters.p));
        }
        const auto st=finish(s);
        require(!st.error,"Invalid HP1 carrier metadata");
        check(cudaMemcpy(out,s.c.p,rows*cols*sizeof(*out),cudaMemcpyDeviceToHost));
        ++s.totals.main_launches; s.totals.main_fragments+=st.fragments;
        s.totals.main_shift_saturations+=st.shifts; s.totals.main_acc_saturations+=st.adds;
        return true;
    } catch (const std::exception & e) { error_text=e.what(); return false; }
}
ggml_cuda_residual_status ggml_cuda_hp1_scu_residual(const void * w,size_t bytes,unsigned bits,
    size_t rows,size_t cols,size_t k,const uint32_t * offsets,const ggml_cuda_residual_event * events,
    size_t count,int64_t * out) {
    auto & s=state(); std::lock_guard<std::mutex> lock(s.mutex); error_text.clear();
    try {
        shape(bits,rows,cols,k);
        require(w && offsets && out && (events || !count) && count<=UINT32_MAX &&
            bytes==cols*(k/32)*(bits==4?24:40) && offsets[0]==0 && offsets[rows]==count,"Invalid SCU residual storage");
        for (size_t row=0;row<rows;++row) {
            require(offsets[row]<=offsets[row+1] && offsets[row+1]<=count,"Invalid residual offsets");
            for (size_t e=offsets[row];e<offsets[row+1];++e)
                require(events[e].k<k && (e==offsets[row] || events[e-1].k<events[e].k),"Noncanonical residual events");
        }
        const void * weights = device_weights(s,w,bytes);
        s.offsets.upload(offsets,(rows+1)*sizeof(*offsets));
        s.events.upload(events,count*sizeof(*events)); s.c.reserve(rows*cols*sizeof(*out));
        s.counters.reserve(sizeof(device_stats)); check(cudaMemset(s.counters.p,0,sizeof(device_stats)));
        residual_kernel<<<unsigned((rows*cols+127)/128),128>>>(static_cast<const uint8_t *>(weights),
            static_cast<uint32_t *>(s.offsets.p),static_cast<ggml_cuda_residual_event *>(s.events.p),
            static_cast<int64_t *>(s.c.p),rows,cols,k,bits,static_cast<device_stats *>(s.counters.p));
        const auto st=finish(s);
        if (st.error) return st.error==2 ? ggml_cuda_residual_status::overflow : ggml_cuda_residual_status::invalid_input;
        check(cudaMemcpy(out,s.c.p,rows*cols*sizeof(*out),cudaMemcpyDeviceToHost));
        ++s.totals.residual_launches; s.totals.residual_fragments+=st.fragments;
        s.totals.residual_shift_saturations+=st.shifts; s.totals.residual_acc_saturations+=st.adds;
        return ggml_cuda_residual_status::success;
    } catch (const std::invalid_argument & e) {
        error_text=e.what(); return ggml_cuda_residual_status::invalid_input;
    } catch (const std::exception & e) {
        error_text=e.what(); return ggml_cuda_residual_status::gpu_failure;
    }
}
