#include "ggml-metal-cpu-exact.h"
#include "ggml-metal-cpu-exact-source.inc"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error_text;
std::mutex execution_mutex;
uint64_t calls = 0;
uint64_t software_calls = 0;
struct shape { uint64_t m, n, k, software; };

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
    ~context() { [pipeline release]; [library release]; [queue release]; [device release]; }
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
struct buffer {
    id<MTLBuffer> value;
    buffer(id<MTLDevice> device, size_t bytes, const void * input) {
        if (bytes > device.maxBufferLength) { throw std::runtime_error("Metal buffer exceeds device limit"); }
        value = input ? [device newBufferWithBytes:input length:bytes options:MTLResourceStorageModeShared]
                      : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!value) { throw std::runtime_error("Metal buffer allocation failed"); }
    }
    ~buffer() { [value release]; }
    buffer(const buffer &) = delete;
    buffer & operator=(const buffer &) = delete;
};
size_t bytes_for(size_t rows, size_t cols) {
    if (rows > std::numeric_limits<size_t>::max()/sizeof(float)/cols) {
        throw std::runtime_error("Matrix byte size overflow");
    }
    return rows*cols*sizeof(float);
}
}

bool ggml_metal_cpu_exact_float(size_t m, size_t n, size_t k,
                              const float * a, const float * b, float * output) {
    error_text.clear();
    @autoreleasepool {
        try {
            if (!m || !n || !k || k%32 || !a || !b || !output ||
                m > UINT32_MAX || n > UINT32_MAX) {
                throw std::runtime_error("CPU FLOAT Metal requires nonempty K32 matrices and valid pointers");
            }
            const size_t ab = bytes_for(m,k), bb = bytes_for(n,k), cb = bytes_for(m,n);
            bool subnormal = false;
            const int amin = minimum_bit_exponent(a,m*k,subnormal);
            const int bmin = minimum_bit_exponent(b,n*k,subnormal);
            std::lock_guard<std::mutex> lock(execution_mutex);
            static context ctx;
            ctx.prepare();
            buffer av(ctx.device,ab,a), bv(ctx.device,bb,b), cv(ctx.device,cb,nullptr);
            id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
            if (!command) { throw std::runtime_error("Metal command allocation failed"); }
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            if (!encoder) { throw std::runtime_error("Metal encoder allocation failed"); }
            const shape s{m,n,k,subnormal || amin+bmin < -126};
            [encoder setComputePipelineState:ctx.pipeline];
            [encoder setBuffer:av.value offset:0 atIndex:0];
            [encoder setBuffer:bv.value offset:0 atIndex:1];
            [encoder setBuffer:cv.value offset:0 atIndex:2];
            [encoder setBytes:&s length:sizeof(s) atIndex:3];
            const NSUInteger width = ctx.pipeline.threadExecutionWidth;
            [encoder dispatchThreads:MTLSizeMake(n,m,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                throw std::runtime_error(command.error.localizedDescription.UTF8String ?: "Metal command failed");
            }
            const auto * result = static_cast<const uint32_t *>(cv.value.contents);
            for (size_t i=0; i<m*n; ++i) {
                if ((result[i]&0x7f800000u) == 0x7f800000u) {
                    throw std::runtime_error("CPU FLOAT Metal produced a nonfinite result");
                }
            }
            std::memcpy(output,cv.value.contents,cb);
            ++calls;
            software_calls += s.software != 0;
            return true;
        } catch (const std::exception & error) {
            error_text = error.what();
            return false;
        }
    }
}
const char * ggml_metal_cpu_exact_last_error(void) { return error_text.c_str(); }
uint64_t ggml_metal_cpu_exact_gpu_calls(void) {
    std::lock_guard<std::mutex> lock(execution_mutex);
    return calls;
}
uint64_t ggml_metal_cpu_exact_software_calls(void) {
    std::lock_guard<std::mutex> lock(execution_mutex);
    return software_calls;
}
