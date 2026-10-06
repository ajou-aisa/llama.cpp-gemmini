#pragma once

#include "../ggml-gemmini-args.h"
#include "options.hpp"
#include "../ggml-gemmini-telemetry.hpp"
#include "../quants/act/exsia/exsia-event.hpp"
#include "../residual/rmd/rmd-compose.hpp"
#include "../residual/rmd/rmd-executor.hpp"

#include <array>
#if defined(__linux__)
#include <sched.h>
#endif
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gemmini/cycle_reader.hpp>
#include <gemmini/host-timing.hpp>
#include <gemmini/cpu_log_context.hpp>

#if !defined(GGML_GEMMINI_CONFIG_HAS_ACTIVATION_QUANT)
namespace ggml::gemmini::config {
inline constexpr int ACTIVATION_QUANT = static_cast<int>(CURRENT_ACTIVATION_QUANT);
}
#endif

namespace ggml::gemmini {

namespace detail {

enum class ActivationRoute : uint8_t { unknown, fp32, exsia, tensor, token, block, stripe };
enum class WeightRoute : uint8_t {
    unknown,
    fp32,
    tensor_i8,
    channel_i8,
    block_i8,
    affine,
    q8_hp1,
    q8_h2,
    q8_hp2,
    q8_channel_direct,
    q8_channel_sidecar,
    q8_h0
};
enum class BackendRoute : uint8_t { cpu, gemmini_ws, gemmini_os, ws_sim };

struct RouteKey {
    ActivationRoute activation = ActivationRoute::unknown;
    WeightRoute     weight     = WeightRoute::unknown;
    BackendRoute    backend    = BackendRoute::cpu;
};

struct RouteCapabilities {
    bool legacy_full             = false;
    bool full                    = false;
    bool sliced_dense            = false;
    bool sliced_compensation     = false;
    bool live_stripe_producer    = false;
    bool internal_parallel_dense = false;
    bool deprecated              = false;
};

RouteKey          normalize_route(const ggml_gemmini_args_t & args);
RouteCapabilities route_capabilities(const ggml_gemmini_args_t & args);
const char *      activation_route_name(ActivationRoute route);
const char *      weight_route_name(WeightRoute route);
const char *      backend_route_name(BackendRoute route);

} // namespace detail

enum class MatMulStatus {
    success,
    empty_stripes,
    malformed_stripe,
    duplicate_stripe,
    overlapping_stripe,
    missing_stripes,
    unsupported,
    invalid_contract,
    invalid_state,
    invalid_arguments,
};

enum class MatMulCapability {
    supported,
    unsupported,
};

enum class MatMulState {
    idle,
    accepting_stripes,
    completed,
};

struct MatMulStripe {
    size_t row_begin;
    size_t row_end;
};

struct MatMulResult {
    MatMulStatus     status;
    MatMulCapability capability;
};

enum class MatmulStatusCode : uint8_t {
    success,
    invalid_argument,
    invalid_contract,
    unsupported_route,
    unsupported_backend,
    unsupported_invocation,
    invalid_state,
    out_of_memory,
    execution_failure,
    cancelled,
};

struct MatmulStatus {
    MatmulStatusCode code       = MatmulStatusCode::success;
    const char *     message    = "success";
    MatMulCapability capability = MatMulCapability::supported;

    bool ok() const {
        return code == MatmulStatusCode::success;
    }

    explicit operator bool() const {
        return ok();
    }
};

#if defined(GGML_GEMMINI_TESTING)
struct MatmulTestCounters {
    uint64_t execution_constructions   = 0;
    uint64_t allocation_attempts       = 0;
    uint64_t dense_dispatches          = 0;
    uint64_t residual_dispatches       = 0;
    uint64_t hardware_dispatches       = 0;
    uint64_t fallback_dispatches       = 0;
    uint64_t native_integer_block_dots = 0;
    uint64_t native_post_dot_scales    = 0;
};

void               test_reset_matmul_counters();
MatmulTestCounters test_matmul_counters();
void               test_inject_output_stage_allocation_failure();
#endif

struct MatmulStageMetrics {
    uint64_t nanoseconds = 0;
    size_t   count       = 0;
};

// Scoped CPU transport: absent collection is not a measured zero. Identity and
// algorithm success belong to the caller's record, not to PMU validity.
struct MatmulCpuInterval {
    std::optional<uint64_t> cycles;
    std::string             reason = "not_collected";
    std::string             sample_reason;
    uint64_t                count                = 1;
    uint64_t                valid_count          = 0;
    uint64_t                not_applicable_count = 0;

    static MatmulCpuInterval measured(uint64_t value) {
        return {value, {}, {}, 1, 1, 0};
    }
    static MatmulCpuInterval unavailable(const char * reason) {
        return {{}, reason, {}, 1, 0, std::string_view(reason) == "not_applicable" ? 1U : 0U};
    }
};

struct MatmulCpuSample {
    uint64_t                  value     = 0;
    bool                      collected = false;
    log::CpuExclusionSnapshot exclusion{};
    log::CpuCorrelation       correlation{};
#if defined(__linux__) && defined(__aarch64__)
    cycle::NativeCycleSample native;
#endif
    uint64_t              ns               = 0;
    uint64_t              tid              = 0;
    uint64_t              thread_cpu_ns    = 0;
    bool                  thread_cpu_valid = false;
    gemmini_trace_context trace{};
    // Observed Linux CPU (sched_getcpu) right after the counter read; -1 when unavailable.
    int32_t cpu_core = -1;
};

inline MatmulCpuSample read_matmul_cpu_sample() {
    MatmulCpuSample result;
    result.exclusion = log::capture_cpu_exclusion();
#if LOG_CYCLE || CYCLE_SIM
    result.correlation = log::current_cpu_correlation();
#endif
#if LOG_CYCLE
    result.collected = true;
    result.trace     = gemmini_trace_capture();
#if defined(__linux__) && defined(__aarch64__)
    result.native = cycle::read_sample();
    result.value  = result.native.value;
#else
    result.value = cycle::read();
#endif
#if defined(__linux__)
    result.cpu_core = sched_getcpu();
#endif
#if CYCLE_DETAIL
    const auto host         = cycle::read_host_sample();
    result.ns               = host.ns;
    result.tid              = host.tid;
    result.thread_cpu_ns    = host.thread_cpu_ns;
    result.thread_cpu_valid = host.thread_cpu_valid;
#else
    result.ns  = cycle::timeline_now_ns();
    result.tid = cycle::host_thread_id();
#endif
#endif
    return result;
}

inline MatmulCpuInterval evaluate_matmul_cpu_interval(const MatmulCpuSample & start,
                                                      const MatmulCpuSample & end,
                                                      bool                    same_task = true) {
    if (!start.collected || !end.collected)
        return {};
    if (const char * excluded = log::cpu_exclusion_since(start.exclusion))
        return MatmulCpuInterval::unavailable(excluded);
    if (start.trace.task_id && end.trace.task_id && start.trace.task_id != end.trace.task_id)
        return MatmulCpuInterval::unavailable("structurally_cross_task");
#if defined(__linux__) && defined(__aarch64__)
    const auto delta = cycle::evaluate_interval(start.native, end.native, same_task);
    if (!delta.valid) {
        auto result = MatmulCpuInterval::unavailable(cycle::reason_name(delta.reason));
        if (delta.sample_reason != cycle::NativeCycleReason::none)
            result.sample_reason = cycle::reason_name(delta.sample_reason);
        return result;
    }
    return MatmulCpuInterval::measured(delta.value);
#else
    if (!same_task)
        return MatmulCpuInterval::unavailable("structurally_cross_task");
    if (end.value < start.value)
        return MatmulCpuInterval::unavailable("counter_regression");
    return MatmulCpuInterval::measured(end.value - start.value);
#endif
}

inline std::optional<uint64_t> matmul_cpu_run_id(const ggml_gemmini_args_t & args) {
    if (const auto * meta = std::get_if<quants::act::block::Meta>(&args.act_quant.storage()))
        return meta->run_id;
    const auto * meta = std::get_if<quants::act::exsia::Meta>(&args.act_quant.storage());
    return meta != nullptr ? meta->run_id : std::nullopt;
}

// ExSIA publishes run/slot identity with its optional metadata snapshot. This
// also covers by-value executions whose metadata predates the producer run.
inline std::optional<uint64_t> matmul_cpu_run_id(const quants::act::exsia::StripeReadyEvent & event,
                                                 std::optional<uint64_t> invocation_run_id = {}) {
    return event.activation_metadata.has_value() ? std::optional<uint64_t>(event.run_id)
                                                 : invocation_run_id;
}

inline bool matmul_telemetry_hash_enabled() {
    const char * value = std::getenv("GGML_GEMMINI_TELEMETRY_HASH");
    return value != nullptr && std::string_view(value) == "1";
}

struct MatmulJobMetrics {
    uint32_t                 cpu_identity_mask = 0;
    MatmulCpuInterval        cpu_dense;
    MatmulCpuInterval        cpu_prep;
    MatmulCpuInterval        cpu_backend;
    MatmulCpuInterval        cpu_merge;
    MatmulCpuInterval        cpu_residual_total;
    MatmulCpuSample          cpu_residual_start;
    bool                     telemetry_hash_enabled = false;
    uint64_t                 run_id                 = 0;
    size_t                   stripe_id              = 0;
    size_t                   slot                   = 0;
    size_t                   row_begin              = 0;
    size_t                   row_end                = 0;
    rmd::RmdExecutionMetrics rmd{};
    uint64_t                 la3_ns               = 0;
    uint64_t                 sf1_ns               = 0;
    uint64_t                 sf_mask_start_ns     = 0;
    uint64_t                 sf_mask_end_ns       = 0;
    uint64_t                 sf_exponent_start_ns = 0;
    uint64_t                 sf_exponent_end_ns   = 0;
    uint64_t                 sf_folding_start_ns  = 0;
    uint64_t                 sf_folding_end_ns    = 0;
    uint64_t                 sf_commit_ns         = 0;
    MatmulStageMetrics       la;
    MatmulStageMetrics       sf;
    MatmulStageMetrics       handoff;
    MatmulStageMetrics       capture_copy;
    MatmulStageMetrics       producer_wait;
    MatmulStageMetrics       queue_insert;
    MatmulStageMetrics       sf_handoff;
    MatmulStageMetrics       ws_queue;
    MatmulStageMetrics       ws_service;
    MatmulStageMetrics       ws;
    MatmulStageMetrics       rmd_decompose;
    MatmulStageMetrics       rmd_index;
    MatmulStageMetrics       rmd_pack;
    MatmulStageMetrics       rmd_queue;
    MatmulStageMetrics       rmd_execute;
    MatmulStageMetrics       rmd_finalize;
    uint64_t                 producer_wait_start_ns   = 0;
    uint64_t                 producer_wait_end_ns     = 0;
    uint64_t                 capture_queue_enqueue_ns = 0;
    uint64_t                 capture_queue_dequeue_ns = 0;
    uint64_t                 queue_enqueue_tid        = 0;
    uint64_t                 queue_dequeue_tid        = 0;
    uint64_t                 ws_start_ns              = 0;
    uint64_t                 ws_end_ns                = 0;
    uint64_t                 ws_start_tid             = 0;
    uint64_t                 ws_end_tid               = 0;
    uint64_t                 rmd_enqueue_ns           = 0;
    uint64_t                 rmd_start_ns             = 0;
    uint64_t                 rmd_end_ns               = 0;
    uint64_t                 backend_start_ns         = 0;
    uint64_t                 backend_end_ns           = 0;
    uint64_t                 backend_start_tid        = 0;
    uint64_t                 backend_end_tid          = 0;
    uint64_t                 merge_start_ns           = 0;
    uint64_t                 merge_end_ns             = 0;
    uint64_t                 merge_start_tid          = 0;
    uint64_t                 merge_end_tid            = 0;
    uint64_t                 finalize_start_ns        = 0;
    uint64_t                 finalize_end_ns          = 0;
    uint64_t                 finalize_start_tid       = 0;
    uint64_t                 finalize_end_tid         = 0;
    uint64_t                 telemetry_queue_tick     = 0;
    uint64_t                 telemetry_dense_start    = 0;
    uint64_t                 telemetry_dense_end      = 0;
    uint64_t                 telemetry_residual_start = 0;
    uint64_t                 telemetry_backend_start  = 0;
    uint64_t                 telemetry_backend_end    = 0;
    uint64_t                 telemetry_merge_start    = 0;
    uint64_t                 telemetry_merge_end      = 0;
    uint64_t                 telemetry_residual_end   = 0;
    std::string              telemetry_input_hash;
    std::string              telemetry_correction_hash;
    uint64_t                 telemetry_correction_nonzero_count = 0;
    std::string              telemetry_output_hash;
};

namespace detail {

struct MatmulCaptureTiming {
    MatmulStageMetrics capture_copy;
    MatmulStageMetrics producer_wait;
    MatmulStageMetrics queue_insert;
    MatmulStageMetrics rmd_pack;
    uint64_t           producer_wait_start_ns = 0;
    uint64_t           producer_wait_end_ns   = 0;
    uint64_t           queued_ns              = 0;
    uint64_t           dequeued_ns            = 0;
    uint64_t           enqueue_tid            = 0;
    uint64_t           dequeue_tid            = 0;
    uint64_t           telemetry_queued_tick  = 0;
};

struct MatmulCapturedStripe {
    gemmini_trace_context                                     trace_origin{};
    uint32_t                                                  cpu_identity_mask = 0;
    uint64_t                                                  run_id            = 0;
    size_t                                                    stripe_id         = 0;
    size_t                                                    slot              = 0;
    size_t                                                    row_begin         = 0;
    size_t                                                    row_end           = 0;
    std::optional<quants::act::exsia::StripeMetadataSnapshot> activation_metadata;
    rmd::StripePacketHandle                                   rmd_packet;
    residual::DirectStripePayloadHandle                       direct_residual;
    uint64_t                                                  la3_ns               = 0;
    uint64_t                                                  sf1_ns               = 0;
    uint64_t                                                  sf_mask_start_ns     = 0;
    uint64_t                                                  sf_mask_end_ns       = 0;
    uint64_t                                                  sf_exponent_start_ns = 0;
    uint64_t                                                  sf_exponent_end_ns   = 0;
    uint64_t                                                  sf_folding_start_ns  = 0;
    uint64_t                                                  sf_folding_end_ns    = 0;
    uint64_t                                                  sf_commit_ns         = 0;
    MatmulCaptureTiming                                       timing;
};

} // namespace detail

struct MatmulCollectorSnapshot {
    MatmulStatus status;
    size_t       capacity  = 0;
    size_t       pending   = 0;
    size_t       in_flight = 0;
    bool         running   = false;
};

enum class MatmulDenseState : uint8_t;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
enum class MatmulCollectorThread : uint8_t {
    worker,
};

enum class MatmulCollectorThreadFailure : uint8_t {
    exception,
    out_of_memory,
};
#endif

enum class MatmulExecutionState {
    empty,
    prepared,
    running,
    finishing,
    completed,
    failed,
};

inline constexpr const char * kRmdTelemetrySchema  = kCycleTelemetrySchema;
inline constexpr uint32_t     kRmdTelemetryVersion = kCycleTelemetryVersion;

struct RmdTelemetryCounters {
    uint64_t direct_events = 0;
    uint64_t direct_calls  = 0;
    uint64_t packet_calls  = 0;
    uint64_t ws_calls      = 0;
};

struct RmdTelemetryGeometry {
    uint64_t packet_count        = 0;
    uint64_t active_blocks       = 0;
    uint64_t compact_k_count     = 0;
    uint64_t padded_k_count      = 0;
    uint64_t physical_tile_count = 0;
};

struct RmdTelemetryTiming {
    MatmulCpuInterval prep;
    MatmulCpuInterval backend_service;
    MatmulCpuInterval merge;
    MatmulCpuInterval residual_total;
    MatmulCpuInterval queue;
    uint64_t          dense_end      = 0;
    uint64_t          residual_start = 0;
};

struct RmdTelemetryStripe {
    size_t stripe_id = 0;
    size_t row_begin = 0;
    size_t row_end   = 0;
    // prep, backend, merge, and proof start/end. Proof is deliberately last,
    // outside all measured service regions.
    std::array<uint64_t, 8> ordered_ticks{};
    std::string             input_hash;
    std::string             correction_hash;
    std::string             output_hash;
    uint64_t                correction_nonzero_count = 0;
    bool                    hash_enabled             = false;
    MatmulCpuInterval       dense{};
};

struct RmdTelemetryRecord {
    std::string                     schema  = kRmdTelemetrySchema;
    uint32_t                        version = kRmdTelemetryVersion;
    std::string                     runtime_bundle_id;
    std::string                     model_id;
    std::string                     layer;
    uint64_t                        run_id  = 0;
    RmdBackend                      backend = RmdBackend::cpu_direct;
    MatmulOptionSource              source  = MatmulOptionSource::build_default;
    std::string                     units;
    bool                            work = false;
    MatmulCpuInterval               invocation_total;
    RmdTelemetryCounters            counters;
    RmdTelemetryGeometry            geometry;
    RmdTelemetryTiming              timing;
    std::vector<RmdTelemetryStripe> stripes;
};

enum class RmdTelemetryCheckCode : uint8_t {
    ok,
    malformed_schema,
    unsupported_version,
    wrong_units,
    zero_work,
    route_not_exclusive,
    invalid_timing,
    ordering_violation,
    missing_detail,
    input_hash_mismatch,
    correction_hash_mismatch,
    correction_nonzero_count_mismatch,
    output_hash_mismatch,
};

struct RmdTelemetryCheckResult {
    RmdTelemetryCheckCode code    = RmdTelemetryCheckCode::ok;
    const char *          message = "ok";
    bool                  ok() const {
        return code == RmdTelemetryCheckCode::ok;
    }
};

enum class MatmulDenseState : uint8_t {
    idle,
    running,
    complete,
    failed,
    cancelled,
};

enum class MatmulResidualState : uint8_t {
    idle,
    ready,
    running,
    complete,
    failed,
    cancelled,
};

struct MatmulStripeJobSnapshot {
    MatmulStatus        status;
    MatmulJobMetrics    metrics;
    MatmulDenseState    dense     = MatmulDenseState::idle;
    MatmulResidualState residual  = MatmulResidualState::idle;
    bool                captured  = false;
    bool                finalized = false;
    bool                released  = false;
};

} // namespace ggml::gemmini
