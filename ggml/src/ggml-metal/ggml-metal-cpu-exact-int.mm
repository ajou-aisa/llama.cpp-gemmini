#include "ggml-metal-cpu-exact-int.h"
#include "ggml-metal-cpu-exact-int-source.inc"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {
std::atomic<uint64_t> launches{0};
struct Runtime {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    std::mutex mutex;
    Runtime() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) return;
        NSError * error = nil;
        NSString * source = [NSString stringWithUTF8String:ggml_metal_cpu_exact_int_source];
        MTLCompileOptions * options = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"cpu_exact_int_blocks"];
        if (function) pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) { fprintf(stderr, "CPU_EXACT_INT: %s\n", error.localizedDescription.UTF8String); return; }
        queue = [device newCommandQueue];
    }
};
}

bool ggml_metal_cpu_exact_int_enabled() {
    const char * value = std::getenv("GGML_GEMMINI_METAL_CPU_EXACT");
    return value && std::strcmp(value, "1") == 0;
}

uint64_t ggml_metal_cpu_exact_int_launches() { return launches.load(); }

bool ggml_metal_cpu_exact_int_dot(const int32_t * a, const int32_t * w,
    size_t rows, size_t columns, size_t k, size_t block_size, int32_t * dots) {
    constexpr size_t max_elements = 16 * 1024 * 1024;
    if (!a || !w || !dots || !rows || !columns || !k || !block_size || block_size > 32 ||
        rows > 128 || columns > 128 || k > max_elements / rows || k > max_elements / columns) return false;
    const size_t blocks = (k + block_size - 1) / block_size;
    if (blocks > max_elements / rows / columns) return false;
    // This proves every <=32-term partial fits signed INT32 without saturation.
    for (size_t i = 0; i < rows * k; ++i) if (a[i] < -128 || a[i] > 127) return false;
    for (size_t i = 0; i < columns * k; ++i) if (w[i] < -128 || w[i] > 127) return false;
    @autoreleasepool {
        static Runtime runtime;
        std::lock_guard<std::mutex> lock(runtime.mutex);
        if (!runtime.pipeline || !runtime.queue) return false;
        const size_t count = blocks * rows * columns;
        id<MTLBuffer> a_buffer = [runtime.device newBufferWithBytes:a length:rows*k*sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> w_buffer = [runtime.device newBufferWithBytes:w length:columns*k*sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = [runtime.device newBufferWithLength:count*sizeof(int32_t) options:MTLResourceStorageModeShared];
        if (!a_buffer || !w_buffer || !output) return false;
        id<MTLCommandBuffer> command = [runtime.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return false;
        const uint32_t shape[] = {uint32_t(rows), uint32_t(columns), uint32_t(k), uint32_t(block_size)};
        [encoder setComputePipelineState:runtime.pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:w_buffer offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        [encoder setBytes:shape length:sizeof(shape) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(runtime.pipeline.threadExecutionWidth,1,1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "CPU_EXACT_INT: command failed: %s\n", command.error.localizedDescription.UTF8String);
            return false;
        }
        memcpy(dots, output.contents, count * sizeof(int32_t));
        if (launches.fetch_add(1) == 0) fprintf(stderr, "METAL_CPU_EXACT_INT: raw_integer_dot=GPU scale_and_residual=CPU\n");
        return true;
    }
}
