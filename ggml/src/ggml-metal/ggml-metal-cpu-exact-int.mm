#include "ggml-metal-cpu-exact-int.h"
#include "ggml-metal-cpu-exact-int-source.inc"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <algorithm>
#include <limits>

namespace {
std::atomic<uint64_t> launches{0};
std::atomic<uint64_t> residual_launches{0};
struct Runtime {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLComputePipelineState> residual_pipeline = nil;
    id<MTLComputePipelineState> hp1_pipeline = nil;
    id<MTLBuffer> a_buffer = nil, w_buffer = nil, output = nil;
    id<MTLBuffer> native_weight = nil, residual_rows = nil, residual_events = nil, residual_output = nil;
    size_t native_bytes = 0;
    std::mutex mutex;
    bool prepare_buffer(id<MTLBuffer> __strong & buffer, size_t bytes) {
        if (bytes > device.maxBufferLength) return false;
        if (!buffer || buffer.length < bytes) {
            id<MTLBuffer> replacement = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!replacement) return false;
            buffer = replacement;
        }
        return true;
    }
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
        function = [library newFunctionWithName:@"cpu_exact_hp1_residual"];
        if (function) residual_pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!residual_pipeline) fprintf(stderr,"CPU_EXACT_RESIDUAL: %s\n",error.localizedDescription.UTF8String);
        function = [library newFunctionWithName:@"cpu_exact_hp1_blocks"];
        if (function) hp1_pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!hp1_pipeline) fprintf(stderr,"CPU_EXACT_HP1: %s\n",error.localizedDescription.UTF8String);
    }
};
}

bool ggml_metal_cpu_exact_int_enabled() {
    const char * value = std::getenv("GGML_GEMMINI_METAL_CPU_EXACT");
    return value && std::strcmp(value, "1") == 0;
}

uint64_t ggml_metal_cpu_exact_int_launches() { return launches.load(); }
uint64_t ggml_metal_cpu_exact_residual_launches() { return residual_launches.load(); }

ggml_metal_residual_status ggml_metal_cpu_exact_hp1_residual(
    const void * weights, size_t weight_bytes, unsigned bits, size_t rows, size_t columns, size_t k,
    const uint32_t * row_offsets, const ggml_metal_residual_event * events, size_t event_count, int64_t * result) {
    using Status = ggml_metal_residual_status;
    struct Result { int64_t value; uint32_t event, error; };
    static_assert(sizeof(Result) == 16 && sizeof(ggml_metal_residual_event) == 8,"Residual device ABI");
    if (!weights || !row_offsets || !result || (event_count && !events) || (bits != 4 && bits != 8) ||
        !rows || !columns || !k || k % 32 || rows >= UINT32_MAX || columns > UINT32_MAX / rows ||
        k > UINT32_MAX || event_count > UINT32_MAX || row_offsets[0] || row_offsets[rows] != event_count)
        return Status::invalid_input;
    const size_t block_bytes = bits == 4 ? 24 : 40;
    if (columns > SIZE_MAX / block_bytes / (k / 32) || weight_bytes != columns * (k / 32) * block_bytes)
        return Status::invalid_input;
    for (size_t row = 0; row < rows; ++row) {
        if (row_offsets[row] > row_offsets[row + 1] || row_offsets[row + 1] > event_count) return Status::invalid_input;
        for (size_t e = row_offsets[row]; e < row_offsets[row + 1]; ++e)
            if (events[e].k >= k || (e > row_offsets[row] && events[e-1].k >= events[e].k)) return Status::invalid_input;
    }
    @autoreleasepool {
        static Runtime runtime;
        std::lock_guard<std::mutex> lock(runtime.mutex);
        const size_t count = rows * columns;
        if (!runtime.residual_pipeline || !runtime.queue || count > SIZE_MAX / sizeof(Result) ||
            !runtime.prepare_buffer(runtime.native_weight,weight_bytes) ||
            !runtime.prepare_buffer(runtime.residual_rows,(rows+1)*sizeof(uint32_t)) ||
            !runtime.prepare_buffer(runtime.residual_events,std::max(size_t(1),event_count)*sizeof(*events)) ||
            !runtime.prepare_buffer(runtime.residual_output,count*sizeof(Result))) return Status::gpu_failure;
        if (runtime.native_bytes != weight_bytes || memcmp(runtime.native_weight.contents,weights,weight_bytes)) {
            memcpy(runtime.native_weight.contents,weights,weight_bytes);
            runtime.native_bytes = weight_bytes;
        }
        memcpy(runtime.residual_rows.contents,row_offsets,(rows+1)*sizeof(uint32_t));
        if (event_count) memcpy(runtime.residual_events.contents,events,event_count*sizeof(*events));
        id<MTLCommandBuffer> command = [runtime.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return Status::gpu_failure;
        const uint32_t shape[] = {uint32_t(rows),uint32_t(columns),uint32_t(k),bits};
        [encoder setComputePipelineState:runtime.residual_pipeline];
        [encoder setBuffer:runtime.native_weight offset:0 atIndex:0];
        [encoder setBuffer:runtime.residual_rows offset:0 atIndex:1];
        [encoder setBuffer:runtime.residual_events offset:0 atIndex:2];
        [encoder setBuffer:runtime.residual_output offset:0 atIndex:3];
        [encoder setBytes:shape length:sizeof(shape) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(count,1,1)
            threadsPerThreadgroup:MTLSizeMake(runtime.residual_pipeline.threadExecutionWidth,1,1)];
        [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return Status::gpu_failure;
        const auto * values = static_cast<const Result *>(runtime.residual_output.contents);
        // Match the CPU executor's first failure: J tile, event span, then column.
        for (size_t column = 0; column < columns; column += 16) {
            uint32_t first_event = UINT32_MAX, error = 0;
            size_t first_column = columns;
            for (size_t row = 0; row < rows; ++row) for (size_t j = column; j < std::min(column+16,columns); ++j) {
                const auto & value = values[row*columns+j];
                if (value.error && (value.event < first_event || (value.event == first_event && j < first_column))) {
                    first_event = value.event; first_column = j; error = value.error;
                }
            }
            if (error) return error == 2 ? Status::overflow : Status::invalid_input;
        }
        for (size_t i = 0; i < count; ++i) result[i] = values[i].value;
        if (residual_launches.fetch_add(1) == 0)
            fprintf(stderr,"METAL_CPU_EXACT_RESIDUAL: original_packed_weight=GPU selected_indices=GPU checked_int64_shift=GPU\n");
        return Status::success;
    }
}

static bool integer_dot(const int32_t * a, const void * w, unsigned native_bits,
    size_t rows, size_t columns, size_t k, size_t block_size, int32_t * dots) {
    constexpr size_t max_elements = 16 * 1024 * 1024;
    if (!a || !w || !dots || !rows || !columns || !k || !block_size || block_size > 32 ||
        rows > 128 || columns > GGML_METAL_CPU_EXACT_MAX_COLUMNS ||
        k > max_elements / rows || k > max_elements / columns) return false;
    if (native_bits && ((native_bits != 4 && native_bits != 8) || k%32 || (block_size != 16 && block_size != 32))) return false;
    const size_t blocks = (k + block_size - 1) / block_size;
    if (blocks > max_elements / rows / columns) return false;
    // This proves every <=32-term partial fits signed INT32 without saturation.
    for (size_t i = 0; i < rows * k; ++i) if (a[i] < -128 || a[i] > 127) return false;
    if (!native_bits) for (size_t i = 0; i < columns * k; ++i)
        if (static_cast<const int32_t *>(w)[i] < -128 || static_cast<const int32_t *>(w)[i] > 127) return false;
    @autoreleasepool {
        static Runtime runtime;
        std::lock_guard<std::mutex> lock(runtime.mutex);
        id<MTLComputePipelineState> pipeline=native_bits ? runtime.hp1_pipeline : runtime.pipeline;
        if (!pipeline || !runtime.queue) return false;
        const size_t count = blocks * rows * columns;
        const size_t weight_bytes=native_bits ? columns*(k/32)*(native_bits==4 ? 24 : 40) : columns*k*sizeof(int32_t);
        if (!runtime.prepare_buffer(runtime.a_buffer,rows*k*sizeof(int32_t)) ||
            !runtime.prepare_buffer(runtime.w_buffer,weight_bytes) ||
            !runtime.prepare_buffer(runtime.output,count*sizeof(int32_t))) return false;
        memcpy(runtime.a_buffer.contents,a,rows*k*sizeof(int32_t));
        memcpy(runtime.w_buffer.contents,w,weight_bytes);
        id<MTLCommandBuffer> command = [runtime.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) return false;
        const uint32_t shape[] = {uint32_t(rows), uint32_t(columns), uint32_t(k), uint32_t(block_size)};
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:runtime.a_buffer offset:0 atIndex:0];
        [encoder setBuffer:runtime.w_buffer offset:0 atIndex:1];
        [encoder setBuffer:runtime.output offset:0 atIndex:2];
        [encoder setBytes:shape length:sizeof(shape) atIndex:3];
        if (native_bits) [encoder setBytes:&native_bits length:sizeof(native_bits) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "CPU_EXACT_INT: command failed: %s\n", command.error.localizedDescription.UTF8String);
            return false;
        }
        memcpy(dots, runtime.output.contents, count * sizeof(int32_t));
        if (launches.fetch_add(1) == 0) fprintf(stderr, "METAL_CPU_EXACT_INT: raw_integer_dot=GPU dense_float_scale=CPU\n");
        return true;
    }
}

bool ggml_metal_cpu_exact_int_dot(const int32_t * a, const int32_t * w,
    size_t rows, size_t columns, size_t k, size_t block_size, int32_t * dots) {
    return integer_dot(a,w,0,rows,columns,k,block_size,dots);
}

bool ggml_metal_cpu_exact_hp1_dot(const int32_t * a, const void * w, unsigned bits,
    size_t rows, size_t columns, size_t k, size_t block_size, int32_t * dots) {
    if (bits != 4 && bits != 8) return false;
    return integer_dot(a,w,bits,rows,columns,k,block_size,dots);
}
