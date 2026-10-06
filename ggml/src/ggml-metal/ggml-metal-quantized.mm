#include "ggml-metal-quantized.h"
#include "ggml-metal-quantized-source.inc"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
double elapsed(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}
thread_local std::string last_error;
std::mutex execution_mutex;
ggml_metal_quantized_stats statistics{};

struct total_scope {
    clock_type::time_point start = clock_type::now();
    bool enabled;
    explicit total_scope(bool account) : enabled(account) {}
    ~total_scope() {
        if (enabled) {
            std::lock_guard<std::mutex> lock(execution_mutex);
            statistics.total_seconds += elapsed(start);
        }
    }
};

struct parameters {
    uint64_t m, n, k, blocks, fragments;
    uint32_t bits, dim, mode, trace, has_residual, reserved;
};
struct request_parameters {
    uint64_t m, n, k, runs, source_begin, source_count;
    uint32_t bits, dim;
};
struct run { uint64_t begin, count; };
struct row { uint32_t lane, source; };
static_assert(sizeof(parameters) == 64, "Metal parameter ABI");
static_assert(sizeof(request_parameters) == 56, "Metal request ABI");

struct buffer {
    id<MTLBuffer> value = nil;
    buffer(id<MTLDevice> device, const void * bytes, size_t count) {
        if (count > device.maxBufferLength) { throw std::runtime_error("Quantized buffer exceeds Metal device maxBufferLength"); }
        // Metal validates inactive device long* bindings against one full element.
        value = bytes && count ? [device newBufferWithBytes:bytes length:count options:MTLResourceStorageModeShared]
                               : [device newBufferWithLength:(count ? count : sizeof(uint64_t)) options:MTLResourceStorageModeShared];
        if (!value) { throw std::runtime_error("Metal shared-buffer allocation failed"); }
    }
    ~buffer() { [value release]; }
    buffer(const buffer &) = delete;
    buffer & operator=(const buffer &) = delete;
};

struct context {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> dense = nil, residual = nil, merge = nil;
    ~context() {
        [merge release]; [residual release]; [dense release];
        [library release]; [queue release]; [device release];
    }
    void initialize() {
        if (dense && residual && merge) { return; }
        if (!device) { device = MTLCreateSystemDefaultDevice(); }
        if (!device) { throw std::runtime_error("No Metal device available; CPU fallback is disabled"); }
        if (!queue) { queue = [device newCommandQueue]; }
        if (!queue) { throw std::runtime_error("Metal command queue creation failed"); }
        if (!library) {
            MTLCompileOptions * options = [[MTLCompileOptions alloc] init];
            options.fastMathEnabled = NO;
            options.languageVersion = MTLLanguageVersion3_1;
            NSError * error = nil;
            NSString * source = [[NSString alloc] initWithBytes:ggml_metal_quantized_source
                length:sizeof(ggml_metal_quantized_source) - 1 encoding:NSUTF8StringEncoding];
            library = [device newLibraryWithSource:source options:options error:&error];
            [source release]; [options release];
            if (!library) { throw std::runtime_error(std::string("Metal quantized shader compile failed: ") + [[error localizedDescription] UTF8String]); }
        }
        auto pipeline = [&](const char * name, id<MTLComputePipelineState> & target) {
            if (target) { return; }
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            NSError * error = nil;
            target = [device newComputePipelineStateWithFunction:function error:&error];
            [function release];
            if (!target) { throw std::runtime_error(std::string("Metal quantized pipeline failed: ") + name); }
        };
        pipeline("mq_dense", dense); pipeline("mq_residual", residual); pipeline("mq_merge", merge);
    }
};
context metal;

void require(bool condition, const char * diagnostic) {
    if (!condition) { throw std::runtime_error(diagnostic); }
}
size_t product(size_t a, size_t b) {
    require(!b || a <= SIZE_MAX / b, "Quantized payload extent overflow");
    return a * b;
}
bool carrier_valid(uint32_t carrier) { return carrier <= 32767 || carrier == 0x80000000u; }
void finite_scales(const float * scales, size_t count) {
    require(scales, "Missing quantized scale plane");
    for (size_t i = 0; i < count; ++i) { require(std::isfinite(scales[i]), "Nonfinite quantized scale"); }
}
void codes_valid(const int8_t * codes, size_t count, uint32_t bits) {
    require(codes, "Missing signed code plane");
    if (bits == 4) {
        for (size_t i = 0; i < count; ++i) { require(codes[i] >= -8 && codes[i] <= 7, "Code exceeds signed 4-bit range"); }
    }
}

size_t validate(const ggml_metal_quantized_view & v,
                const ggml_metal_quantized_request * const * requests,
                const ggml_metal_quantized_trace * trace) {
    require(v.profile.bits == 4 || v.profile.bits == 8, "Only A4W4 and A8W8 are supported");
    require(v.profile.dim == 16 || v.profile.dim == 32 || v.profile.dim == 64, "DIM must be 16, 32, or 64");
    require(v.profile.mode == GGML_METAL_QUANTIZED_BLOCK || v.profile.mode == GGML_METAL_QUANTIZED_HP1_EXSIA, "Unknown quantized arithmetic mode");
    require(v.m && v.n && v.k, "Empty quantized matrix");
    const size_t outputs = product(v.m, v.n);
    require(outputs <= UINT32_MAX && v.k <= UINT32_MAX, "Quantized Metal grid exceeds 32-bit dispatch extent");
    codes_valid(v.activations, product(v.m, v.k), v.profile.bits);
    codes_valid(v.weights, product(v.n, v.k), v.profile.bits);
    const size_t blocks = (v.k + 31) / 32;
    const size_t fragment = v.profile.mode == GGML_METAL_QUANTIZED_BLOCK ? 32 : std::min(v.profile.dim, 32u);
    const size_t fragments = (v.k + fragment - 1) / fragment;
    if (trace) {
        require(!(trace->raw_dots || trace->scu_values) || trace->fragment_capacity >= product(outputs, fragments), "Fragment trace buffer too small");
        require(!(trace->dense_integer || trace->correction) || trace->output_capacity >= outputs, "Output trace buffer too small");
    }
    if (v.profile.mode == GGML_METAL_QUANTIZED_BLOCK) {
        finite_scales(v.activation_scales, product(v.m, blocks));
        finite_scales(v.weight_scales, product(v.n, blocks));
        require(v.request_count == 0, "BLOCK input cannot contain HP1 requests");
        require(!v.block_residual || v.profile.residual_enabled, "BLOCK residual supplied to disabled profile");
        require(!v.block_residual || v.k <= size_t(INT64_MAX / (INT64_C(1) << 38)),
                "BLOCK residual dot extent exceeds proven signed64 range");
        return 0;
    }
    finite_scales(v.activation_scales, v.m);
    finite_scales(v.column_scales, v.n);
    require(v.carriers, "Missing HP1 carrier plane");
    for (size_t i = 0; i < product(blocks, v.n); ++i) { require(carrier_valid(v.carriers[i]), "Invalid HP1 carrier"); }
    require(!v.block_residual, "HP1 input cannot contain BLOCK residual plane");
    require(v.request_count == 0 || (requests && v.profile.residual_enabled), "Missing or disabled HP1 residual requests");
    size_t lane_values = 0, previous_source_end = 0;
    for (size_t r = 0; r < v.request_count; ++r) {
        require(requests[r], "Null HP1 residual request");
        const auto & q = *requests[r];
        require(q.n == v.n && q.m && q.k && q.run_count && q.source_row_count, "Invalid HP1 residual shape");
        require(q.source_row_begin >= previous_source_end && q.source_row_begin <= v.m && q.source_row_count <= v.m - q.source_row_begin, "Overlapping or out-of-range residual stripes");
        previous_source_end = q.source_row_begin + q.source_row_count;
        require(product(q.m, q.n) <= UINT32_MAX, "Residual grid exceeds 32-bit dispatch extent");
        require(q.runs && q.rows && q.weights && q.carriers, "Incomplete HP1 residual metadata");
        codes_valid(q.activations, product(q.m, q.k), v.profile.bits);
        size_t compact_end = 0;
        uint32_t previous_block = 0;
        for (size_t b = 0; b < q.run_count; ++b) {
            const auto & run = q.runs[b];
            require(run.compact_k_begin == compact_end && run.compact_k_count && run.compact_k_count <= 32 && run.compact_k_count <= q.k - compact_end, "Invalid compact residual run");
            require(run.original_block_id < blocks && (b == 0 || run.original_block_id > previous_block), "Residual runs must retain increasing original block order");
            require(run.original_global_k_begin == size_t(run.original_block_id) * 32 && run.original_local_k, "Invalid original K map");
            uint32_t mask = 0;
            for (size_t k = 0; k < run.compact_k_count; ++k) {
                const uint32_t local = run.original_local_k[k];
                require(local < 32 && (k == 0 || local > run.original_local_k[k - 1]) && size_t(run.original_global_k_begin) + local < v.k, "Invalid residual selected K order");
                mask |= uint32_t(1) << local;
                for (size_t n = 0; n < v.n; ++n) {
                    require(q.weights[(compact_end + k) * q.n + n] == v.weights[n * v.k + run.original_global_k_begin + local], "Residual weight code does not match original K map");
                }
            }
            require(mask == run.union_k_mask, "Residual K union mask mismatch");
            for (size_t n = 0; n < v.n; ++n) { require(q.carriers[b * q.n + n] == v.carriers[size_t(run.original_block_id) * v.n + n], "Residual carrier does not match original block"); }
            compact_end += run.compact_k_count;
            previous_block = run.original_block_id;
        }
        require(compact_end == q.k, "Residual compact K extent mismatch");
        uint64_t previous_row = 0;
        for (size_t m = 0; m < q.m; ++m) {
            const auto & row = q.rows[m];
            require(row.source_row < q.source_row_count && row.original_lane_id <= 32 / v.profile.bits, "Invalid residual lane or source row");
            uint64_t key = uint64_t(row.original_lane_id) * q.source_row_count + row.source_row;
            require(m == 0 || key > previous_row, "Residual rows must be unique and ordered by original lane and source row");
            previous_row = key;
        }
        require(lane_values <= SIZE_MAX - product(q.m, q.n), "Residual trace extent overflow");
        lane_values += q.m * q.n;
    }
    require(!trace || !trace->lane_integer || trace->lane_capacity >= lane_values, "Lane trace buffer too small");
    return lane_values;
}

void bind(id<MTLComputeCommandEncoder> encoder, size_t index, const buffer & b) {
    [encoder setBuffer:b.value offset:0 atIndex:index];
}
double launch(id<MTLComputePipelineState> pipeline, size_t count,
              const std::function<void(id<MTLComputeCommandEncoder>)> & configure) {
    id<MTLCommandBuffer> command = [metal.queue commandBuffer];
    require(command, "Metal command buffer creation failed");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder, "Metal compute encoder creation failed");
    [encoder setComputePipelineState:pipeline];
    configure(encoder);
    const NSUInteger width = std::min<NSUInteger>(128, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error(std::string("Metal quantized execution failed: ") + [[command.error localizedDescription] UTF8String]);
    }
    return command.GPUEndTime - command.GPUStartTime;
}

bool execute_locked(const ggml_metal_quantized_view & v,
                    const ggml_metal_quantized_request * const * requests,
                    float * output, ggml_metal_quantized_trace * trace) {
    require(output, "Missing output buffer");
    const size_t lane_count = validate(v, requests, trace);
    metal.initialize();
    const size_t outputs = v.m * v.n, blocks = (v.k + 31) / 32;
    const size_t fragment = v.profile.mode == GGML_METAL_QUANTIZED_BLOCK ? 32 : std::min(v.profile.dim, 32u);
    const size_t fragments = (v.k + fragment - 1) / fragment;
    uint32_t trace_flags = trace ? ((trace->raw_dots ? 1 : 0) | (trace->scu_values ? 2 : 0) | (trace->dense_integer ? 4 : 0) | (trace->correction ? 8 : 0)) : 0;
    parameters p{v.m, v.n, v.k, blocks, fragments, v.profile.bits, v.profile.dim,
                 uint32_t(v.profile.mode), trace_flags, v.block_residual != nullptr, 0};
    const auto transfer_start = clock_type::now();
    buffer a(metal.device, v.activations, product(v.m, v.k));
    buffer w(metal.device, v.weights, product(v.n, v.k));
    buffer as(metal.device, v.activation_scales, product(v.profile.mode == GGML_METAL_QUANTIZED_BLOCK ? product(v.m, blocks) : v.m, sizeof(float)));
    buffer ws(metal.device, v.weight_scales, v.weight_scales ? product(product(v.n, blocks), sizeof(float)) : 0);
    buffer carriers(metal.device, v.carriers, v.carriers ? product(product(blocks, v.n), sizeof(uint32_t)) : 0);
    buffer columns(metal.device, v.column_scales, v.column_scales ? product(v.n, sizeof(float)) : 0);
    buffer residual(metal.device, v.block_residual, v.block_residual ? product(product(v.m, v.k), sizeof(int32_t)) : 0);
    buffer staged(metal.device, nullptr, product(outputs, sizeof(float)));
    buffer raw(metal.device, nullptr, (trace_flags & 1) ? product(product(outputs, fragments), sizeof(int32_t)) : 0);
    buffer scu(metal.device, nullptr, (trace_flags & 2) ? product(product(outputs, fragments), sizeof(int32_t)) : 0);
    buffer integer(metal.device, nullptr, (trace_flags & 4) ? product(outputs, sizeof(int32_t)) : 0);
    buffer correction(metal.device, nullptr, (trace_flags & 8) ? product(outputs, sizeof(int64_t)) : 0);
    uint32_t zero = 0;
    buffer status(metal.device, &zero, sizeof(zero));
    statistics.transfer_seconds += elapsed(transfer_start);
    ++statistics.dense_launches;
    statistics.dense_gpu_seconds += launch(metal.dense, outputs, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBytes:&p length:sizeof(p) atIndex:0];
        bind(encoder, 1, a); bind(encoder, 2, w); bind(encoder, 3, as); bind(encoder, 4, ws);
        bind(encoder, 5, carriers); bind(encoder, 6, columns); bind(encoder, 7, residual);
        bind(encoder, 8, staged); bind(encoder, 9, raw); bind(encoder, 10, scu);
        bind(encoder, 11, integer); bind(encoder, 12, correction); bind(encoder, 13, status);
    });
    require(*static_cast<uint32_t *>(status.value.contents) == 0, "Nonfinite dense result or BLOCK correction");
    std::vector<int32_t> lane_trace(trace && trace->lane_integer ? lane_count : 0);
    size_t lane_offset = 0;
    for (size_t index = 0; index < v.request_count; ++index) {
        const auto & q = *requests[index];
        request_parameters rp{q.m, q.n, q.k, q.run_count, q.source_row_begin, q.source_row_count, v.profile.bits, v.profile.dim};
        const auto transfer_run_start = clock_type::now();
        std::vector<run> runs(q.run_count);
        for (size_t i = 0; i < q.run_count; ++i) { runs[i] = {q.runs[i].compact_k_begin, q.runs[i].compact_k_count}; }
        std::vector<row> rows(q.m);
        for (size_t i = 0; i < q.m; ++i) { rows[i] = {q.rows[i].original_lane_id, q.rows[i].source_row}; }
        buffer qa(metal.device, q.activations, product(q.m, q.k));
        buffer qw(metal.device, q.weights, product(product(q.k, q.n), sizeof(int32_t)));
        buffer qc(metal.device, q.carriers, product(product(q.run_count, q.n), sizeof(uint32_t)));
        buffer qr(metal.device, runs.data(), product(runs.size(), sizeof(run)));
        buffer qm(metal.device, rows.data(), product(rows.size(), sizeof(row)));
        buffer lanes(metal.device, nullptr, product(product(q.m, q.n), sizeof(int32_t)));
        statistics.transfer_seconds += elapsed(transfer_run_start);
        ++statistics.residual_launches;
        statistics.residual_gpu_seconds += launch(metal.residual, q.m * q.n, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBytes:&rp length:sizeof(rp) atIndex:0];
            bind(encoder, 1, qa); bind(encoder, 2, qw); bind(encoder, 3, qc);
            bind(encoder, 4, qr); bind(encoder, 5, lanes);
        });
        ++statistics.merge_launches;
        statistics.merge_gpu_seconds += launch(metal.merge, q.source_row_count * q.n, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBytes:&rp length:sizeof(rp) atIndex:0];
            bind(encoder, 1, lanes); bind(encoder, 2, qm); bind(encoder, 3, columns);
            bind(encoder, 4, as); bind(encoder, 5, staged); bind(encoder, 6, correction); bind(encoder, 7, status);
            [encoder setBytes:&trace_flags length:sizeof(trace_flags) atIndex:8];
        });
        const uint32_t code = *static_cast<uint32_t *>(status.value.contents);
        require(code != 2, "HP1 radix correction exceeds signed64");
        require(code == 0, "Nonfinite HP1 residual restoration or merge");
        if (!lane_trace.empty()) { std::memcpy(lane_trace.data() + lane_offset, lanes.value.contents, q.m * q.n * sizeof(int32_t)); }
        lane_offset += q.m * q.n;
    }
    const auto publish_start = clock_type::now();
    std::memcpy(output, staged.value.contents, outputs * sizeof(float));
    if (trace_flags & 1) { std::memcpy(trace->raw_dots, raw.value.contents, outputs * fragments * sizeof(int32_t)); }
    if (trace_flags & 2) { std::memcpy(trace->scu_values, scu.value.contents, outputs * fragments * sizeof(int32_t)); }
    if (trace_flags & 4) { std::memcpy(trace->dense_integer, integer.value.contents, outputs * sizeof(int32_t)); }
    if (trace_flags & 8) { std::memcpy(trace->correction, correction.value.contents, outputs * sizeof(int64_t)); }
    if (!lane_trace.empty()) { std::memcpy(trace->lane_integer, lane_trace.data(), lane_trace.size() * sizeof(int32_t)); }
    statistics.transfer_seconds += elapsed(publish_start);
    if (v.profile.mode == GGML_METAL_QUANTIZED_BLOCK) { ++statistics.block_calls; } else { ++statistics.hp1_calls; }
    return true;
}
} // namespace

bool ggml_metal_quantized_enabled(void) { return true; }
const char * ggml_metal_quantized_last_error(void) { return last_error.c_str(); }
ggml_metal_quantized_stats ggml_metal_quantized_get_stats(void) {
    std::lock_guard<std::mutex> lock(execution_mutex); return statistics;
}
void ggml_metal_quantized_reset_stats(void) {
    std::lock_guard<std::mutex> lock(execution_mutex); statistics = {};
}
bool ggml_metal_quantized_supports_op(const ggml_tensor * op) {
    return op && op->op == GGML_OP_MUL_MAT && op->type == GGML_TYPE_F32 &&
        op->ne[2] == 1 && op->ne[3] == 1 && op->nb[0] == sizeof(float) &&
        op->nb[1] >= size_t(op->ne[0]) * sizeof(float) &&
        ggml_metal_quantized_can_prepare(op->src[0], op->src[1]);
}
static bool execute_internal(const ggml_metal_quantized_view * view,
        const ggml_metal_quantized_request * const * requests, float * output,
        ggml_metal_quantized_trace * trace, bool account_total) {
    total_scope total(account_total);
    std::lock_guard<std::mutex> lock(execution_mutex);
    last_error.clear();
    bool success = false;
    @autoreleasepool {
        try {
            require(view, "Missing quantized payload view");
            success = execute_locked(*view, requests, output, trace);
        } catch (const std::exception & error) {
            last_error = error.what(); ++statistics.failed_calls;
        }
    }
    return success;
}
bool ggml_metal_quantized_execute(const ggml_metal_quantized_view * view,
        const ggml_metal_quantized_request * const * requests, float * output,
        ggml_metal_quantized_trace * trace) {
    return execute_internal(view, requests, output, trace, true);
}
enum ggml_status ggml_metal_quantized_compute(ggml_tensor * op) {
    total_scope total(true);
    last_error.clear();
    try {
        require(ggml_metal_quantized_supports_op(op), "Unsupported quantized Metal operation");
        require(op->data, "Quantized Metal output is not host accessible");
        ggml_metal_quantized_payload * raw_payload = nullptr;
        char error[512]{};
        const auto producer_start = clock_type::now();
        const bool prepared = ggml_metal_quantized_prepare(op->src[0], op->src[1], &raw_payload, error, sizeof(error));
        const double producer_time = elapsed(producer_start);
        { std::lock_guard<std::mutex> lock(execution_mutex); statistics.producer_seconds += producer_time; }
        require(prepared, error[0] ? error : "CPU quantized producer failed");
        std::unique_ptr<ggml_metal_quantized_payload, decltype(&ggml_metal_quantized_free)> payload(raw_payload, ggml_metal_quantized_free);
        const auto * view = ggml_metal_quantized_get_view(payload.get());
        require(view && view->n == size_t(op->ne[0]) && view->m == size_t(op->ne[1]), "Quantized output tensor shape mismatch");
        std::vector<const ggml_metal_quantized_request *> requests(view->request_count);
        for (size_t i = 0; i < requests.size(); ++i) { requests[i] = ggml_metal_quantized_get_request(payload.get(), i); }
        std::vector<float> staged(product(view->m, view->n));
        if (!execute_internal(view, requests.data(), staged.data(), nullptr, false)) { return GGML_STATUS_FAILED; }
        for (size_t m = 0; m < view->m; ++m) {
            std::memcpy(static_cast<char *>(op->data) + m * op->nb[1], staged.data() + m * view->n, view->n * sizeof(float));
        }
        return GGML_STATUS_SUCCESS;
    } catch (const std::bad_alloc & error) {
        last_error = error.what();
        std::lock_guard<std::mutex> lock(execution_mutex); ++statistics.failed_calls;
        return GGML_STATUS_ALLOC_FAILED;
    } catch (const std::exception & error) {
        last_error = error.what();
        std::lock_guard<std::mutex> lock(execution_mutex); ++statistics.failed_calls;
        return GGML_STATUS_FAILED;
    }
}
