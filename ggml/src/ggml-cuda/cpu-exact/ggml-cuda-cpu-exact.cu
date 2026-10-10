#include "ggml-cuda-cpu-exact.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error_text;
std::atomic<uint64_t> integer_calls{0}, residual_calls{0}, attention_calls{0};
constexpr size_t max_elements = 16 * 1024 * 1024;
struct residual_result { int64_t value; uint32_t event; uint32_t error; };

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
size_t product(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b) throw std::runtime_error("CUDA matrix size overflow");
    return a * b;
}
void require(bool condition, const char * message) {
    if (!condition) throw std::invalid_argument(message);
}
struct buffer {
    void * data = nullptr;
    size_t capacity = 0;
    buffer() = default;
    buffer(const buffer &) = delete;
    buffer & operator=(const buffer &) = delete;
    ~buffer() { if (data) cudaFree(data); }
    void reserve(size_t bytes) {
        if (bytes <= capacity) return;
        void * next = nullptr;
        check(cudaMalloc(&next, bytes));
        if (data) cudaFree(data);
        data = next;
        capacity = bytes;
    }
    void upload(const void * source, size_t bytes, cudaStream_t stream) {
        reserve(bytes);
        if (bytes) check(cudaMemcpyAsync(data, source, bytes, cudaMemcpyHostToDevice, stream));
    }
};
struct runtime {
    std::mutex mutex;
    cudaStream_t stream = nullptr;
    buffer a, w, c, offsets, events, residual_weights;
    std::vector<uint8_t> weight_source;
    std::vector<int32_t> dots;
    std::vector<ggml_fp16_t> x_half, w_half;
    std::vector<float> attention_output;
    std::vector<residual_result> residual_output;
    void prepare() {
        if (stream) return;
        int device = 0;
        check(cudaGetDevice(&device));
        cudaDeviceProp info{};
        check(cudaGetDeviceProperties(&info, device));
        require(info.major * 10 + info.minor >= 53, "CUDA exact attention requires SM53 or newer");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::fprintf(stderr, "CUDA_CPU_EXACT_DEVICE name=%s sm=%d%d shared_memory=%zu\n",
            info.name, info.major, info.minor, size_t(info.sharedMemPerBlock));
    }
    void finish() { check(cudaGetLastError()); check(cudaStreamSynchronize(stream)); }
    ~runtime() { if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); } }
};
runtime & state() { static runtime instance; return instance; }
struct drain_on_exit {
    runtime & ctx;
    ~drain_on_exit() { if (ctx.stream) cudaStreamSynchronize(ctx.stream); }
};

__device__ int weight_code(const uint8_t * block, unsigned bits, unsigned k) {
    return bits == 4 ? int((block[k % 16] >> (k < 16 ? 0 : 4)) & 15) - 8
                     : int(static_cast<int8_t>(block[k]));
}

// Each fragment is independent; the CPU retains the original ordered scale restoration.
__global__ void integer_tiles(const int32_t * a, const void * weights, int32_t * output,
                             unsigned rows, unsigned columns, unsigned k, unsigned fragment, unsigned bits) {
    __shared__ int av[8][8];
    __shared__ int wv[16][8];
    const unsigned t = threadIdx.y * 16 + threadIdx.x, start = blockIdx.z * fragment;
    const unsigned groups = (fragment + 3) / 4;
    for (unsigned item = t; item < 24 * groups; item += 128) {
        const unsigned local_row = item / groups, group = item % groups;
        unsigned packed = 0;
        for (unsigned lane = 0; lane < 4; ++lane) {
            const unsigned offset = group * 4 + lane, pos = start + offset;
            int value = 0;
            if (offset < fragment && pos < k) {
                if (local_row < 8) {
                    const unsigned row = blockIdx.y * 8 + local_row;
                    if (row < rows) value = a[size_t(row) * k + pos];
                } else {
                    const unsigned col = blockIdx.x * 16 + local_row - 8;
                    if (col < columns) {
                        if (!bits) value = static_cast<const int32_t *>(weights)[size_t(col) * k + pos];
                        else {
                            const auto * block = static_cast<const uint8_t *>(weights) +
                                (size_t(col) * (k / 32) + pos / 32) * (bits == 4 ? 24 : 40);
                            value = weight_code(block, bits, pos % 32);
                        }
                    }
                }
            }
            packed |= unsigned(uint8_t(value)) << (8 * lane);
        }
        if (local_row < 8) av[local_row][group] = int(packed);
        else wv[local_row - 8][group] = int(packed);
    }
    __syncthreads();
    const unsigned row = blockIdx.y * 8 + threadIdx.y, col = blockIdx.x * 16 + threadIdx.x;
    if (row >= rows || col >= columns) return;
    int dot = 0;
    for (unsigned group = 0; group < groups; ++group) {
        const int ap = av[threadIdx.y][group], wp = wv[threadIdx.x][group];
#if __CUDA_ARCH__ >= 610
        dot = __dp4a(ap, wp, dot);
#else
        for (unsigned lane = 0; lane < 4; ++lane)
            dot += int(int8_t(unsigned(ap) >> (lane * 8))) * int(int8_t(unsigned(wp) >> (lane * 8)));
#endif
    }
    output[(size_t(blockIdx.z) * rows + row) * columns + col] = dot;
}

__global__ void residual_kernel(const uint8_t * weights, const uint32_t * offsets,
    const ggml_cuda_residual_event * events, residual_result * output,
    size_t count, unsigned columns, unsigned k, unsigned bits) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned row = index / columns, col = index % columns, carrier = bits == 4 ? 16 : 32;
    int64_t acc = 0;
    for (unsigned begin = offsets[row]; begin < offsets[row + 1];) {
        const unsigned block_id = events[begin].k / 32;
        const uint8_t * w = weights + (size_t(col) * (k / 32) + block_id) * (bits == 4 ? 24 : 40);
        int64_t raw = 0;
        unsigned end = begin;
        while (end < offsets[row + 1] && events[end].k / 32 == block_id) {
            raw += int64_t(events[end].value) * weight_code(w, bits, events[end].k % 32);
            ++end;
        }
        const int exponent = int16_t(unsigned(w[carrier]) | (unsigned(w[carrier + 1]) << 8));
        const unsigned scale = unsigned(w[carrier + 4]) | (unsigned(w[carrier + 5]) << 8) |
            (unsigned(w[carrier + 6]) << 16) | (unsigned(w[carrier + 7]) << 24);
        if (w[carrier + 2] || w[carrier + 3] || (scale & 0x7f800000u) == 0x7f800000u ||
            (exponent < 0 && exponent != INT16_MIN)) { output[index] = {0, begin, 1}; return; }
        if (exponent >= 63) { output[index] = {0, begin, 2}; return; }
        int64_t scaled = 0;
        if (exponent != INT16_MIN) {
            const bool negative = raw < 0;
            const uint64_t magnitude = negative ? uint64_t(0) - uint64_t(raw) : uint64_t(raw);
            const uint64_t limit = negative ? (uint64_t(1) << 63) : uint64_t(INT64_MAX);
            if (magnitude > (limit >> exponent)) { output[index] = {0, begin, 2}; return; }
            const uint64_t shifted = magnitude << exponent;
            scaled = int64_t(negative ? uint64_t(0) - shifted : shifted);
        }
        if ((scaled > 0 && acc > INT64_MAX - scaled) || (scaled < 0 && acc < INT64_MIN - scaled)) {
            output[index] = {0, begin, 2}; return;
        }
        acc += scaled;
        begin = end;
    }
    output[index] = {acc, 0, 0};
}

struct attention_shape { size_t m, n, k, wh, xh, wb, xb; };
__global__ void attention_kernel(const __half2 * x, const __half2 * w, float * output, attention_shape s) {
    __shared__ __half2 xv[8][16];
    __shared__ __half2 wv[16][16];
    const size_t col = size_t(blockIdx.x) * 16 + threadIdx.x;
    const size_t row = size_t(blockIdx.y) * 8 + threadIdx.y, plane = blockIdx.z;
    const size_t wh = (plane % s.xh) / (s.xh / s.wh) + (plane / s.xh) / (s.xb / s.wb) * s.wh;
    const unsigned thread = threadIdx.y * 16 + threadIdx.x;
    __half2 lanes[16];
#pragma unroll
    for (int lane = 0; lane < 16; ++lane) lanes[lane] = __float2half2_rn(0.0f);
    for (size_t k = 0; k < s.k; k += 32) {
        for (unsigned item = thread; item < 24 * 16; item += 128) {
            const unsigned local = item / 16, pair = item % 16;
            __half2 value = __float2half2_rn(0.0f);
            if (local < 8) {
                const size_t r = size_t(blockIdx.y) * 8 + local;
                if (r < s.m) value = x[((plane * s.m + r) * s.k + k) / 2 + pair];
                xv[local][pair] = value;
            } else {
                const size_t c = size_t(blockIdx.x) * 16 + local - 8;
                if (c < s.n) value = w[((wh * s.n + c) * s.k + k) / 2 + pair];
                wv[local - 8][pair] = value;
            }
        }
        __syncthreads();
#pragma unroll
        for (int lane = 0; lane < 16; ++lane)
            lanes[lane] = __hfma2(xv[threadIdx.y][lane], wv[threadIdx.x][lane], lanes[lane]);
        __syncthreads();
    }
    if (row >= s.m || col >= s.n) return;
    float sum[4];
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
        const __half2 low = __hadd2(__hadd2(lanes[pair], lanes[pair + 8]),
                                   __hadd2(lanes[pair + 4], lanes[pair + 12]));
        const __half2 high = __hadd2(__hadd2(lanes[pair + 2], lanes[pair + 10]),
                                    __hadd2(lanes[pair + 6], lanes[pair + 14]));
        const float2 lo = __half22float2(low), hi = __half22float2(high);
        sum[pair * 2] = __fadd_rn(lo.x, hi.x);
        sum[pair * 2 + 1] = __fadd_rn(lo.y, hi.y);
    }
    output[(plane * s.m + row) * s.n + col] = __fadd_rn(__fadd_rn(sum[0], sum[1]), __fadd_rn(sum[2], sum[3]));
}

bool int_dot(const int32_t * a, const void * w, unsigned bits, size_t rows, size_t cols,
             size_t k, size_t fragment, int32_t * output) {
    error_text.clear();
    auto & ctx = state();
    std::lock_guard<std::mutex> lock(ctx.mutex);
    drain_on_exit drain{ctx};
    try {
        require(a && w && output && rows && cols && k && fragment && fragment <= 32 &&
            rows <= 128 && cols <= GGML_CUDA_CPU_EXACT_MAX_COLUMNS && k <= UINT32_MAX,
            "Invalid CUDA integer dot shape");
        require(!bits || ((bits == 4 || bits == 8) && k % 32 == 0 && (fragment == 16 || fragment == 32)),
            "Invalid native HP1 dot profile");
        const size_t an = product(rows, k), wn = product(cols, k), fragments = (k + fragment - 1) / fragment;
        const size_t cn = product(product(rows, cols), fragments);
        require(an <= max_elements && wn <= max_elements && cn <= max_elements && fragments <= 65535,
            "CUDA integer dot exceeds tile limits");
        for (size_t i = 0; i < an; ++i) require(a[i] >= -128 && a[i] <= 127, "Activation code outside INT8 range");
        if (!bits) for (size_t i = 0; i < wn; ++i) {
            const int value = static_cast<const int32_t *>(w)[i];
            require(value >= -128 && value <= 127, "Weight code outside INT8 range");
        }
        ctx.prepare();
        ctx.a.upload(a, an * sizeof(int32_t), ctx.stream);
        ctx.w.upload(w, bits ? wn / 32 * (bits == 4 ? 24 : 40) : wn * sizeof(int32_t), ctx.stream);
        ctx.c.reserve(cn * sizeof(int32_t));
        ctx.dots.resize(cn);
        integer_tiles<<<dim3((cols + 15) / 16, (rows + 7) / 8, fragments), dim3(16, 8), 0, ctx.stream>>>(
            static_cast<const int32_t *>(ctx.a.data), ctx.w.data, static_cast<int32_t *>(ctx.c.data),
            rows, cols, k, fragment, bits);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(ctx.dots.data(), ctx.c.data, cn * sizeof(int32_t), cudaMemcpyDeviceToHost, ctx.stream));
        ctx.finish();
        std::memcpy(output, ctx.dots.data(), cn * sizeof(int32_t));
        ++integer_calls;
        return true;
    } catch (const std::exception & error) { error_text = error.what(); return false; }
}
}

bool ggml_cuda_cpu_exact_int_enabled() {
    const char * value = std::getenv("GGML_GEMMINI_CUDA_CPU_EXACT");
    return value && std::strcmp(value, "1") == 0;
}
bool ggml_cuda_cpu_exact_int_dot(const int32_t * a, const int32_t * w,
    size_t rows, size_t cols, size_t k, size_t fragment, int32_t * output) {
    return int_dot(a, w, 0, rows, cols, k, fragment, output);
}
bool ggml_cuda_cpu_exact_hp1_dot(const int32_t * a, const void * w, unsigned bits,
    size_t rows, size_t cols, size_t k, size_t fragment, int32_t * output) {
    if (bits != 4 && bits != 8) { error_text = "HP1 requires 4-bit or 8-bit weights"; return false; }
    return int_dot(a, w, bits, rows, cols, k, fragment, output);
}
const char * ggml_cuda_cpu_exact_last_error() { return error_text.c_str(); }
uint64_t ggml_cuda_cpu_exact_int_launches() { return integer_calls.load(); }
uint64_t ggml_cuda_cpu_exact_residual_launches() { return residual_calls.load(); }
uint64_t ggml_cuda_cpu_exact_attention_calls() { return attention_calls.load(); }

ggml_cuda_residual_status ggml_cuda_cpu_exact_hp1_residual(const void * weights, size_t bytes,
    unsigned bits, size_t rows, size_t cols, size_t k, const uint32_t * offsets,
    const ggml_cuda_residual_event * events, size_t event_count, int64_t * output) {
    error_text.clear();
    auto & ctx = state();
    std::lock_guard<std::mutex> lock(ctx.mutex);
    drain_on_exit drain{ctx};
    try {
        require(weights && offsets && output && (events || !event_count) && rows && cols && k &&
            rows < UINT32_MAX && cols <= UINT32_MAX && k <= UINT32_MAX && event_count <= UINT32_MAX &&
            k % 32 == 0 && (bits == 4 || bits == 8), "Invalid HP1 residual shape");
        require(bytes == product(product(cols, k / 32), bits == 4 ? 24 : 40), "Invalid HP1 residual weight size");
        require(offsets[0] == 0 && offsets[rows] == event_count, "Invalid HP1 residual offsets");
        for (size_t row = 0; row < rows; ++row) {
            require(offsets[row] <= offsets[row + 1] && offsets[row + 1] <= event_count, "Invalid residual row offsets");
            for (size_t e = offsets[row]; e < offsets[row + 1]; ++e)
                require(events[e].k < k && (e == offsets[row] || events[e - 1].k < events[e].k), "Residual K indices are not canonical");
        }
        const size_t count = product(rows, cols), result_bytes = product(count, sizeof(residual_result));
        require(count <= size_t(INT_MAX) * 128, "Residual CUDA grid exceeds device limits");
        ctx.residual_output.resize(count);
        auto & result = ctx.residual_output;
        ctx.prepare();
        if (ctx.weight_source.size() != bytes || std::memcmp(ctx.weight_source.data(), weights, bytes)) {
            ctx.weight_source.clear();
            ctx.residual_weights.upload(weights, bytes, ctx.stream);
            ctx.finish();
            const auto * source = static_cast<const uint8_t *>(weights);
            ctx.weight_source.assign(source, source + bytes);
        }
        ctx.offsets.upload(offsets, (rows + 1) * sizeof(uint32_t), ctx.stream);
        ctx.events.upload(events, event_count * sizeof(*events), ctx.stream);
        ctx.c.reserve(result_bytes);
        residual_kernel<<<unsigned((count + 127) / 128), 128, 0, ctx.stream>>>(
            static_cast<const uint8_t *>(ctx.residual_weights.data), static_cast<const uint32_t *>(ctx.offsets.data),
            static_cast<const ggml_cuda_residual_event *>(ctx.events.data), static_cast<residual_result *>(ctx.c.data),
            count, cols, k, bits);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(result.data(), ctx.c.data, result_bytes, cudaMemcpyDeviceToHost, ctx.stream));
        ctx.finish();
        ++residual_calls;
        size_t first_tile = SIZE_MAX, first_event = SIZE_MAX, first_col = SIZE_MAX;
        uint32_t error = 0;
        for (size_t i = 0; i < count; ++i) if (result[i].error) {
            const size_t col = i % cols, tile = col / 16, event = result[i].event;
            if (tile < first_tile || (tile == first_tile && (event < first_event ||
                (event == first_event && col < first_col)))) {
                first_tile = tile; first_event = event; first_col = col; error = result[i].error;
            }
        }
        if (error) return error == 2 ? ggml_cuda_residual_status::overflow : ggml_cuda_residual_status::invalid_input;
        for (size_t i = 0; i < count; ++i) output[i] = result[i].value;
        return ggml_cuda_residual_status::success;
    } catch (const std::invalid_argument & error) {
        error_text = error.what(); return ggml_cuda_residual_status::invalid_input;
    } catch (const std::exception & error) {
        error_text = error.what(); return ggml_cuda_residual_status::gpu_failure;
    }
}

bool ggml_cuda_cpu_exact_attention_supported(const ggml_tensor * op) {
    if (!ggml_cuda_cpu_exact_int_enabled() || !op || op->op != GGML_OP_MUL_MAT ||
        op->type != GGML_TYPE_F32 || !op->src[0] || !op->src[1] || !ggml_is_contiguous(op)) return false;
    const auto * w = op->src[0]; const auto * x = op->src[1];
    if (w->type != GGML_TYPE_F16 || x->type != GGML_TYPE_F32 || x->ne[2] <= 1 ||
        w->nb[0] != sizeof(ggml_fp16_t) || x->nb[0] != sizeof(float) || w->ne[0] % 32 || w->ne[0] != x->ne[0]) return false;
    for (int d = 0; d < 4; ++d) if (w->ne[d] <= 0 || x->ne[d] <= 0 || w->ne[d] > UINT32_MAX || x->ne[d] > UINT32_MAX) return false;
    return x->ne[2] % w->ne[2] == 0 && x->ne[3] % w->ne[3] == 0 && op->ne[0] == w->ne[1] &&
        op->ne[1] == x->ne[1] && op->ne[2] == x->ne[2] && op->ne[3] == x->ne[3];
}
bool ggml_cuda_cpu_exact_attention(ggml_tensor * op) {
    error_text.clear();
    auto & ctx = state();
    std::lock_guard<std::mutex> lock(ctx.mutex);
    drain_on_exit drain{ctx};
    try {
        require(ggml_cuda_cpu_exact_attention_supported(op) && op->data && op->src[0]->data && op->src[1]->data,
            "Unsupported CUDA attention layout");
        const auto * w = op->src[0]; const auto * x = op->src[1];
        const attention_shape s{size_t(x->ne[1]), size_t(w->ne[1]), size_t(w->ne[0]),
            size_t(w->ne[2]), size_t(x->ne[2]), size_t(w->ne[3]), size_t(x->ne[3])};
        const size_t count = product(product(s.m, s.n), product(s.xh, s.xb));
        require((s.m + 7) / 8 <= 65535 && product(s.xh, s.xb) <= 65535 && (s.n + 15) / 16 <= INT_MAX,
            "Attention CUDA grid exceeds device limits");
        ctx.x_half.resize(product(product(s.m, s.k), product(s.xh, s.xb)));
        ctx.w_half.resize(product(product(s.n, s.k), product(s.wh, s.wb)));
        ctx.attention_output.resize(count);
        for (int input = 0; input < 2; ++input) {
            const auto * tensor = input ? w : x;
            auto * dest = input ? ctx.w_half.data() : ctx.x_half.data();
            for (int64_t b = 0; b < tensor->ne[3]; ++b) for (int64_t h = 0; h < tensor->ne[2]; ++h)
                for (int64_t r = 0; r < tensor->ne[1]; ++r) {
                    const auto * src = static_cast<const char *>(tensor->data) + b*tensor->nb[3] + h*tensor->nb[2] + r*tensor->nb[1];
                    if (input) std::memcpy(dest, src, s.k * sizeof(*dest));
                    else ggml_fp32_to_fp16_row(reinterpret_cast<const float *>(src), dest, s.k);
                    dest += s.k;
                }
        }
        ctx.prepare();
        ctx.a.upload(ctx.x_half.data(), ctx.x_half.size() * sizeof(ggml_fp16_t), ctx.stream);
        ctx.w.upload(ctx.w_half.data(), ctx.w_half.size() * sizeof(ggml_fp16_t), ctx.stream);
        ctx.c.reserve(product(count, sizeof(float)));
        attention_kernel<<<dim3((s.n + 15) / 16, (s.m + 7) / 8, s.xh * s.xb), dim3(16, 8), 0, ctx.stream>>>(
            static_cast<const __half2 *>(ctx.a.data), static_cast<const __half2 *>(ctx.w.data),
            static_cast<float *>(ctx.c.data), s);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(ctx.attention_output.data(), ctx.c.data, count * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream));
        ctx.finish();
        std::memcpy(op->data, ctx.attention_output.data(), count * sizeof(float));
        if (attention_calls++ == 0) std::fprintf(stderr, "CUDA_CPU_EXACT_ATTENTION FP16_dot=GPU reduction_order=preserved\n");
        return true;
    } catch (const std::exception & error) { error_text = error.what(); return false; }
}
