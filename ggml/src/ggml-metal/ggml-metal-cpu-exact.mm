#include "ggml-metal-cpu-exact.h"
#include "ggml.h"
#include "ggml-metal-cpu-exact-source.inc"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error_text;
std::mutex execution_mutex;
uint64_t calls = 0;
uint64_t software_calls = 0;
uint64_t attention_calls = 0;
struct shape { uint64_t m, n, k, software; };
struct attention_shape { uint64_t m, n, k, w_heads, x_heads, w_batches, x_batches; };

struct buffer {
    id<MTLBuffer> value = nil;
    ~buffer() { [value release]; }
    buffer() = default;
    buffer(const buffer &) = delete;
    buffer & operator=(const buffer &) = delete;
    void prepare(id<MTLDevice> device, size_t bytes, const void * input) {
        if (bytes > device.maxBufferLength) { throw std::runtime_error("Metal buffer exceeds device limit"); }
        if (!value || value.length < bytes) {
            id<MTLBuffer> replacement = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!replacement) { throw std::runtime_error("Metal buffer allocation failed"); }
            [value release];
            value = replacement;
        }
        if (input) { std::memcpy(value.contents,input,bytes); }
    }
};

int minimum_bit_exponent(const float * values, size_t count, bool & subnormal) {
    int minimum = 1024;
    for (size_t i=0; i<count; ++i) {
        uint32_t bits;
        std::memcpy(&bits,values+i,sizeof(bits));
        const uint32_t exponent=(bits>>23)&255, fraction=bits&0x7fffff;
        if (exponent == 255) { throw std::runtime_error("CPU FLOAT Metal requires finite inputs"); }
        if (!exponent && !fraction) { continue; }
        subnormal |= !exponent;
        const uint32_t significand=fraction | (exponent ? 0x800000u : 0u);
        const int bit=(exponent ? int(exponent)-150 : -149)+__builtin_ctz(significand);
        if (bit < minimum) { minimum = bit; }
    }
    return minimum;
}

struct context {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLComputePipelineState> attention_pipeline = nil;
    buffer av, bv, cv;
    // Keep only the largest matrix; caching the entire FP32 model would multiply memory use.
    buffer cached_weight;
    std::vector<uint8_t> cached_source;
    ggml_type cached_type = GGML_TYPE_COUNT;
    size_t cached_n = 0, cached_k = 0;
    int cached_exponent = 1024;
    bool cached_subnormal = false;
    ~context() { [attention_pipeline release]; [pipeline release]; [library release]; [queue release]; [device release]; }
    void prepare() {
        if (pipeline) { return; }
        if (!device) { device = MTLCreateSystemDefaultDevice(); }
        if (!device) { throw std::runtime_error("No Metal device"); }
        if (!queue) { queue = [device newCommandQueue]; }
        if (!queue) { throw std::runtime_error("Metal command queue creation failed"); }
        NSError * error = nil;
        if (!library) {
            MTLCompileOptions * options = [[MTLCompileOptions alloc] init];
            options.fastMathEnabled = NO;
            library = [device newLibraryWithSource:[NSString stringWithUTF8String:ggml_metal_cpu_exact_source]
                                           options:options error:&error];
            [options release];
            if (!library) { throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal source compilation failed"); }
        }
        id<MTLFunction> function = [library newFunctionWithName:@"cpu_exact_float"];
        if (!function) { throw std::runtime_error("Metal CPU FLOAT function missing"); }
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        if (!pipeline) { throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal pipeline creation failed"); }
    }
};
size_t bytes_for(size_t rows, size_t cols) {
    if (rows > std::numeric_limits<size_t>::max()/sizeof(float)/cols) {
        throw std::runtime_error("Matrix byte size overflow");
    }
    return rows*cols*sizeof(float);
}
}

bool ggml_metal_cpu_exact_attention_supported(const ggml_tensor * op) {
    const char * enabled=std::getenv("GGML_GEMMINI_METAL_CPU_EXACT");
    if (!enabled || std::strcmp(enabled,"1") || !op || op->op != GGML_OP_MUL_MAT ||
        op->type != GGML_TYPE_F32 || !op->src[0] || !op->src[1] || !ggml_is_contiguous(op)) return false;
    const auto * w=op->src[0]; const auto * x=op->src[1];
    if (w->type != GGML_TYPE_F16 || x->type != GGML_TYPE_F32 || x->ne[2] <= 1 ||
        w->nb[0] != sizeof(ggml_fp16_t) || x->nb[0] != sizeof(float) || w->ne[0]%32 ||
        w->ne[0] != x->ne[0]) return false;
    for (int d=0;d<4;++d) if (w->ne[d]<=0 || x->ne[d]<=0 || w->ne[d]>UINT32_MAX || x->ne[d]>UINT32_MAX) return false;
    return x->ne[2]%w->ne[2]==0 && x->ne[3]%w->ne[3]==0 &&
        op->ne[0]==w->ne[1] && op->ne[1]==x->ne[1] && op->ne[2]==x->ne[2] && op->ne[3]==x->ne[3];
}

uint64_t ggml_metal_cpu_exact_attention_calls() {
    std::lock_guard<std::mutex> lock(execution_mutex);
    return attention_calls;
}

bool ggml_metal_cpu_exact_attention(ggml_tensor * op) {
    error_text.clear();
    @autoreleasepool {
        try {
            if (!ggml_metal_cpu_exact_attention_supported(op) || !op->data || !op->src[0]->data || !op->src[1]->data)
                throw std::runtime_error("Unsupported CPU-equivalent attention layout");
            const auto * w=op->src[0]; const auto * x=op->src[1];
            const attention_shape s{uint64_t(x->ne[1]),uint64_t(w->ne[1]),uint64_t(w->ne[0]),
                uint64_t(w->ne[2]),uint64_t(x->ne[2]),uint64_t(w->ne[3]),uint64_t(x->ne[3])};
            std::lock_guard<std::mutex> lock(execution_mutex);
            static context ctx;
            ctx.prepare();
            if (!ctx.attention_pipeline) {
                NSError * error=nil;
                id<MTLFunction> function=[ctx.library newFunctionWithName:@"cpu_exact_attention"];
                if (!function) throw std::runtime_error("Metal attention function missing");
                ctx.attention_pipeline=[ctx.device newComputePipelineStateWithFunction:function error:&error];
                [function release];
                if (!ctx.attention_pipeline) throw std::runtime_error(error.localizedDescription.UTF8String ?: "Metal attention pipeline failed");
            }
            ctx.av.prepare(ctx.device,bytes_for(ggml_nrows(x),s.k)/2,nullptr);
            ctx.bv.prepare(ctx.device,bytes_for(ggml_nrows(w),s.k)/2,nullptr);
            ctx.cv.prepare(ctx.device,ggml_nbytes(op),nullptr);
            for (int input=0;input<2;++input) {
                const auto * tensor=input ? w : x;
                auto * dest=static_cast<ggml_fp16_t *>((input ? ctx.bv : ctx.av).value.contents);
                for (int64_t b=0;b<tensor->ne[3];++b) for (int64_t h=0;h<tensor->ne[2];++h)
                    for (int64_t r=0;r<tensor->ne[1];++r) {
                        const auto * src=static_cast<const char *>(tensor->data)+b*tensor->nb[3]+h*tensor->nb[2]+r*tensor->nb[1];
                        if (input) std::memcpy(dest,src,s.k*sizeof(*dest));
                        else ggml_fp32_to_fp16_row(reinterpret_cast<const float *>(src),dest,s.k);
                        dest+=s.k;
                    }
            }
            id<MTLCommandBuffer> command=[ctx.queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            if (!command || !encoder) throw std::runtime_error("Metal attention command allocation failed");
            [encoder setComputePipelineState:ctx.attention_pipeline];
            [encoder setBuffer:ctx.av.value offset:0 atIndex:0];
            [encoder setBuffer:ctx.bv.value offset:0 atIndex:1];
            [encoder setBuffer:ctx.cv.value offset:0 atIndex:2];
            [encoder setBytes:&s length:sizeof(s) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(s.n,s.m,s.x_heads*s.x_batches)
                threadsPerThreadgroup:MTLSizeMake(ctx.attention_pipeline.threadExecutionWidth,1,1)];
            [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                throw std::runtime_error(command.error.localizedDescription.UTF8String ?: "Metal attention command failed");
            std::memcpy(op->data,ctx.cv.value.contents,ggml_nbytes(op));
            if (attention_calls++ == 0) std::fprintf(stderr,"METAL_CPU_EXACT_ATTENTION: FP16_dot=GPU CPU_reduction_order=preserved\n");
            return true;
        } catch (const std::exception & error) { error_text=error.what(); return false; }
    }
}

static bool float_matmul(size_t m, size_t n, size_t k,
                         const float * a, const float * b, float * output, const ggml_tensor * native) {
    error_text.clear();
    @autoreleasepool {
        try {
            if (!m || !n || !k || k%32 || !a || (!b && !native) || !output ||
                m > UINT32_MAX || n > UINT32_MAX) {
                throw std::runtime_error("CPU FLOAT Metal requires nonempty K32 matrices and valid pointers");
            }
            const size_t ab = bytes_for(m,k), bb = bytes_for(n,k), cb = bytes_for(m,n);
            std::vector<float> activation;
            if (ggml_metal_cpu_exact_activation_fp16_enabled()) {
                activation.resize(m*k);
                for (size_t i = 0; i < activation.size(); ++i) {
                    activation[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(a[i]));
                }
                a = activation.data();
            }
            bool subnormal = false;
            const int amin = minimum_bit_exponent(a,m*k,subnormal);
            std::lock_guard<std::mutex> lock(execution_mutex);
            static context ctx;
            ctx.prepare();
            int bmin;
            id<MTLBuffer> weight_buffer;
            if (native) {
                if (!native->data || !ggml_is_contiguous(native) || native->ne[0] != int64_t(k) ||
                    ggml_nelements(native) != int64_t(n*k) ||
                    (native->type != GGML_TYPE_Q4_0 && native->type != GGML_TYPE_Q8_0 && native->type != GGML_TYPE_Q6_K) ||
                    k % ggml_blck_size(native->type)) {
                    throw std::runtime_error("Invalid native FLOAT weight tensor");
                }
                const size_t source_bytes = ggml_nbytes(native);
                const bool cache = !ctx.cached_weight.value || bb >= ctx.cached_weight.value.length;
                const bool hit = cache && ctx.cached_type == native->type && ctx.cached_n == n && ctx.cached_k == k &&
                    ctx.cached_source.size() == source_bytes &&
                    std::memcmp(ctx.cached_source.data(),native->data,source_bytes) == 0;
                buffer & target = cache ? ctx.cached_weight : ctx.bv;
                if (!hit) {
                    if (cache) ctx.cached_type = GGML_TYPE_COUNT;
                    target.prepare(ctx.device,bb,nullptr);
                    ggml_get_type_traits(native->type)->to_float(native->data,static_cast<float *>(target.value.contents),n*k);
                    bool weight_subnormal = false;
                    bmin = minimum_bit_exponent(static_cast<const float *>(target.value.contents),n*k,weight_subnormal);
                    subnormal |= weight_subnormal;
                    if (cache) {
                        ctx.cached_source.assign(static_cast<const uint8_t *>(native->data),static_cast<const uint8_t *>(native->data)+source_bytes);
                        ctx.cached_type = native->type; ctx.cached_n = n; ctx.cached_k = k;
                        ctx.cached_exponent = bmin; ctx.cached_subnormal = weight_subnormal;
                    }
                } else {
                    bmin = ctx.cached_exponent;
                    subnormal |= ctx.cached_subnormal;
                }
                weight_buffer = target.value;
            } else {
                bmin = minimum_bit_exponent(b,n*k,subnormal);
                ctx.bv.prepare(ctx.device,bb,b);
                weight_buffer = ctx.bv.value;
            }
            ctx.av.prepare(ctx.device,ab,a);
            ctx.cv.prepare(ctx.device,cb,nullptr);
            id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
            if (!command) { throw std::runtime_error("Metal command allocation failed"); }
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) { throw std::runtime_error("Metal encoder allocation failed"); }
            const shape s{m,n,k,subnormal || amin+bmin < -126};
            [encoder setComputePipelineState:ctx.pipeline];
            [encoder setBuffer:ctx.av.value offset:0 atIndex:0];
            [encoder setBuffer:weight_buffer offset:0 atIndex:1];
            [encoder setBuffer:ctx.cv.value offset:0 atIndex:2];
            [encoder setBytes:&s length:sizeof(s) atIndex:3];
            const NSUInteger width = ctx.pipeline.threadExecutionWidth;
            [encoder dispatchThreads:MTLSizeMake(n,m,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                throw std::runtime_error(command.error.localizedDescription.UTF8String ?: "Metal command failed");
            }
            const auto * result = static_cast<const uint32_t *>(ctx.cv.value.contents);
            for (size_t i=0; i<m*n; ++i) {
                if ((result[i]&0x7f800000u) == 0x7f800000u) {
                    throw std::runtime_error("CPU FLOAT Metal produced a nonfinite result");
                }
            }
            std::memcpy(output,ctx.cv.value.contents,cb);
            ++calls;
            static bool fp16_reported = false;
            if (ggml_metal_cpu_exact_activation_fp16_enabled() && !fp16_reported) {
                std::fprintf(stderr, "METAL_RTN_FP16: GPU matmul completed with FP16-rounded activation and original dequantized weight\n");
                fp16_reported = true;
            }
            software_calls += s.software != 0;
            return true;
        } catch (const std::exception & error) {
            error_text = error.what();
            return false;
        }
    }
}
bool ggml_metal_cpu_exact_float(size_t m, size_t n, size_t k,
                              const float * a, const float * b, float * output) {
    return float_matmul(m,n,k,a,b,output,nullptr);
}
bool ggml_metal_cpu_exact_float_quantized(size_t m, size_t n, size_t k,
                                        const float * a, const ggml_tensor * b, float * output) {
    if (!b) { error_text = "Missing native FLOAT weight tensor"; return false; }
    return float_matmul(m,n,k,a,nullptr,output,b);
}
const char * ggml_metal_cpu_exact_last_error(void) { return error_text.c_str(); }
bool ggml_metal_cpu_exact_activation_fp16_enabled(void) {
    const char * value = std::getenv("GGML_GEMMINI_METAL_ACTIVATION_FP16");
    return value && std::strcmp(value, "1") == 0;
}
uint64_t ggml_metal_cpu_exact_gpu_calls(void) {
    std::lock_guard<std::mutex> lock(execution_mutex);
    return calls;
}
uint64_t ggml_metal_cpu_exact_software_calls(void) {
    std::lock_guard<std::mutex> lock(execution_mutex);
    return software_calls;
}
