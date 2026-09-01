#pragma once

#include "ggml-gemmini-args.h"
#include "ggml-gemmini-geometry.hpp"
#include "ggml-gemmini-matmul-config.hpp"
#include "ggml-gemmini-telemetry.hpp"
#include "quants/act/exsia/exsia.hpp"
#include "residual/rmd/rmd-compose.hpp"
#include "residual/rmd/rmd-executor.hpp"

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
#include <gemmini/cycle_reader.hpp>
#endif

#if !defined(GGML_GEMMINI_CONFIG_HAS_ACTIVATION_QUANT)
namespace ggml::gemmini::config {
inline constexpr int ACTIVATION_QUANT = static_cast<int>(CURRENT_ACTIVATION_QUANT);
}
#endif

namespace ggml::gemmini {

struct MatmulJobMetrics;

namespace detail {

enum class ActivationRoute : uint8_t { unknown, fp32, exsia, tensor, token, block, stripe };
enum class WeightRoute : uint8_t {
    unknown, fp32, tensor_i8, channel_i8, block_i8, q8_h1, q8_hp1, q8_h2, q8_hp2,
    q8_channel_direct, q8_channel_sidecar, q8_h0
};
enum class BackendRoute : uint8_t { cpu, gemmini_ws, gemmini_os, ws_sim };

struct RouteKey {
    ActivationRoute activation = ActivationRoute::unknown;
    WeightRoute weight = WeightRoute::unknown;
    BackendRoute backend = BackendRoute::cpu;
};

struct RouteCapabilities {
    bool legacy_full = false;
    bool full = false;
    bool sliced_dense = false;
    bool sliced_compensation = false;
    bool live_stripe_producer = false;
    bool internal_parallel_dense = false;
    bool deprecated = false;
};

RouteKey normalize_route(const ggml_gemmini_args_t & args);
RouteCapabilities route_capabilities(const ggml_gemmini_args_t & args);
const char * activation_route_name(ActivationRoute route);
const char * weight_route_name(WeightRoute route);
const char * backend_route_name(BackendRoute route);
PipelineStripeTelemetry pipeline_stripe_telemetry(
    const char * layer, const MatmulJobMetrics & profile);

}

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
    MatMulStatus status;
    MatMulCapability capability;
};

class MatmulStripeJob;
class MatmulExecution;
class MatmulStripeInput;
struct MatmulStatus;

class MatMul {
public:
    explicit MatMul(ggml_gemmini_args_t args);
    explicit MatMul(ggml_gemmini_args_t * args);
    MatMul(MatMul && other) noexcept;
    MatMul & operator=(MatMul && other) noexcept;
    ~MatMul();
    MatMul(const MatMul &) = delete;
    MatMul & operator=(const MatMul &) = delete;

    MatMulResult run_dense();
    MatMulResult run_full();
    MatMulStatus begin_stripes();
    MatMulStatus run_stripe(MatMulStripe stripe);
    MatMulStatus finish_stripes();

    static MatMulCapability stripe_capability(const ggml_gemmini_args_t & args);
    MatMulState state() const;

private:
    friend class MatmulExecution;
    friend class MatmulStripeCollector;
    friend MatmulStatus execute_full(MatmulExecution &);
    friend MatmulStatus finish_execution(MatmulExecution &);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput, rmd::StripePacketHandle);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput,
                                          residual::DirectStripePayloadHandle,
                                          rmd::StripePacketHandle);
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus accept_external_dense_completion(MatmulStripeJob &);
    friend MatmulStatus execute_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus compose_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus finalize_stripe(MatmulStripeJob &);

    MatMulResult run_dense(bool transactional);
    MatMulStatus run_stripe(MatMulStripe stripe, size_t stripe_id);
    MatMulStatus run_staged_stripe(
        MatMulStripe stripe, size_t stripe_id,
        const quants::act::Meta & activation_metadata);
    MatMulStatus begin_output_transaction();
    void commit_output_transaction();
    void discard_output_transaction();

    ggml_gemmini_args_t & args();
    const ggml_gemmini_args_t & args() const;

    ggml_gemmini_args_t owned_args_{};
    ggml_gemmini_args_t * args_ptr_ = nullptr;
    size_t first_row_ = 0;
    size_t last_row_begin_ = 0;
    size_t last_row_end_ = 0;
    size_t covered_rows_ = 0;
    bool has_stripes_ = false;
    MatMulState state_ = MatMulState::idle;
    float * output_destination_ = nullptr;
    size_t output_row_stride_ = 0;
    size_t output_col_stride_ = 0;
    std::vector<float> output_stage_;
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
    MatmulStatusCode code = MatmulStatusCode::success;
    const char * message = "success";
    MatMulCapability capability = MatMulCapability::supported;

    bool ok() const {
        return code == MatmulStatusCode::success;
    }

    explicit operator bool() const { return ok(); }
};

#if defined(GGML_GEMMINI_TESTING)
struct MatmulTestCounters {
    uint64_t execution_constructions = 0;
    uint64_t allocation_attempts = 0;
    uint64_t dense_dispatches = 0;
    uint64_t residual_dispatches = 0;
    uint64_t hardware_dispatches = 0;
    uint64_t fallback_dispatches = 0;
};

void test_reset_matmul_counters();
MatmulTestCounters test_matmul_counters();
void test_inject_output_stage_allocation_failure();
#endif

struct MatmulStageMetrics {
    uint64_t nanoseconds = 0;
    size_t count = 0;
};

struct MatmulJobMetrics {
    uint64_t run_id = 0;
    size_t stripe_id = 0;
    size_t slot = 0;
    size_t row_begin = 0;
    size_t row_end = 0;
    rmd::RmdExecutionMetrics rmd{};
    uint64_t la3_ns = 0;
    uint64_t sf1_ns = 0;
    uint64_t sf_mask_start_ns = 0;
    uint64_t sf_mask_end_ns = 0;
    uint64_t sf_exponent_start_ns = 0;
    uint64_t sf_exponent_end_ns = 0;
    uint64_t sf_folding_start_ns = 0;
    uint64_t sf_folding_end_ns = 0;
    uint64_t sf_commit_ns = 0;
    MatmulStageMetrics la;
    MatmulStageMetrics sf;
    MatmulStageMetrics handoff;
    MatmulStageMetrics capture_copy;
    MatmulStageMetrics producer_wait;
    MatmulStageMetrics queue_insert;
    MatmulStageMetrics sf_handoff;
    MatmulStageMetrics ws_queue;
    MatmulStageMetrics ws_service;
    MatmulStageMetrics ws;
    MatmulStageMetrics rmd_decompose;
    MatmulStageMetrics rmd_index;
    MatmulStageMetrics rmd_pack;
    MatmulStageMetrics rmd_queue;
    MatmulStageMetrics rmd_execute;
    MatmulStageMetrics rmd_output_read;
    MatmulStageMetrics rmd_compose;
    MatmulStageMetrics rmd_finalize;
    uint64_t producer_wait_start_ns = 0;
    uint64_t producer_wait_end_ns = 0;
    uint64_t capture_queue_enqueue_ns = 0;
    uint64_t ws_start_ns = 0;
    uint64_t ws_end_ns = 0;
    uint64_t rmd_enqueue_ns = 0;
    uint64_t rmd_start_ns = 0;
    uint64_t rmd_end_ns = 0;
    uint64_t compose_start_ns = 0;
    uint64_t compose_end_ns = 0;
    uint64_t merge_start_ns = 0;
    uint64_t merge_end_ns = 0;
    uint64_t finalize_start_ns = 0;
    uint64_t finalize_end_ns = 0;
    uint64_t telemetry_queue_tick = 0;
    uint64_t telemetry_dense_start = 0;
    uint64_t telemetry_dense_end = 0;
    uint64_t telemetry_residual_start = 0;
    uint64_t telemetry_backend_start = 0;
    uint64_t telemetry_backend_end = 0;
    uint64_t telemetry_merge_start = 0;
    uint64_t telemetry_merge_end = 0;
    uint64_t telemetry_residual_end = 0;
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
    cycle::NativeCycleSample telemetry_compose_start_sample;
    cycle::NativeCycleSample telemetry_compose_end_sample;
    cycle::NativeCycleSample telemetry_finalize_start_sample;
    cycle::NativeCycleSample telemetry_finalize_end_sample;
#endif
    std::string telemetry_input_hash;
    std::string telemetry_correction_hash;
    uint64_t telemetry_correction_nonzero_count = 0;
    std::string telemetry_output_hash;
};

namespace detail {

struct MatmulCaptureTiming {
    MatmulStageMetrics capture_copy;
    MatmulStageMetrics producer_wait;
    MatmulStageMetrics queue_insert;
    MatmulStageMetrics rmd_pack;
    uint64_t producer_wait_start_ns = 0;
    uint64_t producer_wait_end_ns = 0;
    uint64_t queued_ns = 0;
    uint64_t telemetry_queued_tick = 0;
};

struct MatmulCapturedStripe {
    uint64_t run_id = 0;
    size_t stripe_id = 0;
    size_t slot = 0;
    size_t row_begin = 0;
    size_t row_end = 0;
    std::optional<quants::act::exsia::StripeMetadataSnapshot> activation_metadata;
    rmd::StripePacketHandle rmd_packet;
    residual::DirectStripePayloadHandle direct_residual;
    uint64_t la3_ns = 0;
    uint64_t sf1_ns = 0;
    uint64_t sf_mask_start_ns = 0;
    uint64_t sf_mask_end_ns = 0;
    uint64_t sf_exponent_start_ns = 0;
    uint64_t sf_exponent_end_ns = 0;
    uint64_t sf_folding_start_ns = 0;
    uint64_t sf_folding_end_ns = 0;
    uint64_t sf_commit_ns = 0;
    MatmulCaptureTiming timing;
};

} // namespace detail

struct MatmulCollectorSnapshot {
    MatmulStatus status;
    size_t capacity = 0;
    size_t pending = 0;
    size_t in_flight = 0;
    bool running = false;
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

class MatmulStripeCollector {
public:
    explicit MatmulStripeCollector(size_t capacity);
    ~MatmulStripeCollector();
    bool start(MatmulExecution & execution);
    MatmulStatus cancel();
    MatmulStatus finish();
    const quants::act::exsia::StripeReadySink * sink() const;
    MatmulStatus status() const;
    MatmulCollectorSnapshot snapshot() const;
    std::vector<MatmulJobMetrics> profiles() const;
    rmd::StripePacketHandle captured_packet(size_t stripe) const;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    void test_inject_residual_failure(MatmulStatus failure);
    void test_inject_thread_start_failure(size_t attempt = 1);
    void test_inject_thread_exception(
        MatmulCollectorThread thread,
        MatmulCollectorThreadFailure failure = MatmulCollectorThreadFailure::exception);
    void test_pause_dense_before_execute();
    void test_pause_startup_after_attachment();
    void test_resume_startup();
    void test_wait_for_residual_failure();
    size_t test_in_flight() const;
    MatmulDenseState test_dense_state_at_release() const;
#endif

private:
    using CapturedStripe = detail::MatmulCapturedStripe;
    static bool on_ready(void *, const quants::act::exsia::StripeReadyEvent &);
    friend MatmulStatus execute_post_fold_pipeline(const ggml_gemmini_args_t &, MatmulStripeCollector &);
    void fail(MatmulStatus status);
    void release_in_flight_once(const std::shared_ptr<MatmulStripeJob> & job);
    void worker_loop();
    bool worker_started_ = false;
    bool startup_in_progress_ = false;
    bool stop_requested_ = false;
    bool dense_done_ = false;
    std::thread worker_;
    // Borrowed for the active pipeline; finish the collector before destroying the execution.
    MatmulExecution * execution_ = nullptr;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<CapturedStripe> pending_;
    std::vector<std::weak_ptr<MatmulStripeJob>> jobs_;
    size_t capacity_;
    size_t in_flight_ = 0;
    std::vector<CapturedStripe> stripes_;
    std::vector<MatmulJobMetrics> profiles_;
    MatmulStatus status_;
    quants::act::exsia::StripeReadySink sink_;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    bool test_pause_dense_ = false;
    bool test_pause_startup_ = false;
    bool test_residual_failure_observed_ = false;
    size_t test_fail_thread_start_attempt_ = 0;
    size_t test_thread_start_attempts_ = 0;
    std::optional<MatmulCollectorThread> test_thread_exception_;
    MatmulCollectorThreadFailure test_thread_exception_failure_ =
        MatmulCollectorThreadFailure::exception;
#endif
};

enum class MatmulInvocationMode {
    full,
    stripe_pipeline,
};

enum class MatmulExecutionState {
    empty,
    prepared,
    running,
    finishing,
    completed,
    failed,
};

enum class RmdBackend : uint8_t {
    cpu_direct,
    gemmini_ws_compact,
};

enum class MatmulOptionSource : uint8_t {
    build_default,
    environment,
    explicit_override,
};

inline constexpr const char * kRmdTelemetrySchema = kCycleTelemetrySchema;
inline constexpr uint32_t kRmdTelemetryVersion = kCycleTelemetryVersion;

struct RmdTelemetryCounters {
    uint64_t direct_events = 0;
    uint64_t direct_calls = 0;
    uint64_t packet_calls = 0;
    uint64_t ws_calls = 0;
};

struct RmdTelemetryGeometry {
    uint64_t packet_count = 0;
    uint64_t active_blocks = 0;
    uint64_t compact_k_count = 0;
    uint64_t padded_k_count = 0;
    uint64_t physical_tile_count = 0;
};

struct RmdTelemetryTiming {
    uint64_t prep = 0;
    uint64_t backend_service = 0;
    uint64_t merge = 0;
    uint64_t residual_total = 0;
    uint64_t queue = 0;
    uint64_t dense_end = 0;
    uint64_t residual_start = 0;
};

struct RmdTelemetryStripe {
    size_t stripe_id = 0;
    size_t row_begin = 0;
    size_t row_end = 0;
    // prep, backend, merge, and proof start/end. Proof is deliberately last,
    // outside all measured service regions.
    std::array<uint64_t, 8> ordered_ticks{};
    std::string input_hash;
    std::string correction_hash;
    std::string output_hash;
    uint64_t correction_nonzero_count = 0;
};

struct RmdTelemetryRecord {
    std::string schema = kRmdTelemetrySchema;
    uint32_t version = kRmdTelemetryVersion;
    std::string runtime_bundle_id;
    std::string model_id;
    std::string layer;
    uint64_t run_id = 0;
    RmdBackend backend = RmdBackend::cpu_direct;
    MatmulOptionSource source = MatmulOptionSource::build_default;
    std::string units;
    bool work = false;
    uint64_t invocation_total = 0;
    RmdTelemetryCounters counters;
    RmdTelemetryGeometry geometry;
    RmdTelemetryTiming timing;
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
    RmdTelemetryCheckCode code = RmdTelemetryCheckCode::ok;
    const char * message = "ok";
    bool ok() const { return code == RmdTelemetryCheckCode::ok; }
};

// Stable pure seams: Todo 10 can consume records without parsing log prose.
std::string serialize_rmd_telemetry(const RmdTelemetryRecord & record);
RmdTelemetryCheckResult check_rmd_telemetry(const RmdTelemetryRecord & record,
                                             std::string_view expected_units,
                                             bool comparison_mode);
RmdTelemetryCheckResult compare_rmd_telemetry_proofs(
    const RmdTelemetryRecord & lhs, const RmdTelemetryRecord & rhs);
std::string rmd_input_hash(const residual::DirectStripePayload & payload);
std::string rmd_input_hash(const rmd::StripePacket & packet);
std::string rmd_correction_hash(const rmd::Correction & correction);
std::string rmd_output_hash(const ggml_gemmini_args_t & args,
                            size_t row_begin, size_t row_end);
std::string resolve_rmd_model_id(const char * environment_model_id,
                                 std::string_view model_arch);
RmdTelemetryRecord make_rmd_telemetry_record(
    RmdBackend backend, MatmulOptionSource source,
    std::string runtime_bundle_id, std::string model_id, std::string layer,
    uint64_t run_id,
    uint64_t invocation_total, const std::vector<MatmulJobMetrics> & profiles);

struct ResolvedMatmulOptions {
    MatmulInvocationMode mode = static_cast<MatmulInvocationMode>(config::DEFAULT_MATMUL_MODE);
    size_t dense_threads = 0;
    bool validation = false;
    bool profiling = false;
    size_t job_capacity = config::DEFAULT_STRIPE_JOB_CAPACITY;
    RmdBackend rmd_backend = static_cast<RmdBackend>(config::DEFAULT_RMD_BACKEND);

    ResolvedMatmulOptions() {}
};

struct MatmulOptionOverrides {
    std::optional<MatmulInvocationMode> mode;
    size_t dense_threads = 0;
    bool validation = false;
    bool profiling = false;
    std::optional<size_t> job_capacity;
    std::optional<RmdBackend> rmd_backend;
};

#ifdef GGML_GEMMINI_MATMUL_IMPLEMENTATION
using MatmulOptions = ResolvedMatmulOptions;
#else
using MatmulOptions = MatmulOptionOverrides;
#endif

enum class MatmulOptionsError : uint8_t {
    none,
    invalid_mode,
    invalid_job_capacity,
    invalid_rmd_backend,
    runtime_override_disabled,
    disabled_mode,
};

struct MatmulOptionsResolution {
    ResolvedMatmulOptions options;
    MatmulOptionsError error = MatmulOptionsError::none;
    MatmulOptionSource rmd_backend_source = MatmulOptionSource::build_default;

    bool ok() const { return error == MatmulOptionsError::none; }
};

inline bool parse_positive_size(std::string_view text, size_t & value) {
    if (text.empty()) {
        return false;
    }
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size() && value > 0;
}

inline MatmulOptionsResolution resolve_matmul_options(const MatmulOptionOverrides & explicit_options = {}) {
    MatmulOptionsResolution result;
    if (!config::ALLOW_RUNTIME_MATMUL_OVERRIDE && !explicit_options.rmd_backend &&
        std::getenv("GEMMINI_RMD_BACKEND") != nullptr) {
        result.error = MatmulOptionsError::runtime_override_disabled;
        return result;
    }
    if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE) {
        if (!explicit_options.mode) if (const char * value = std::getenv("GEMMINI_MATMUL_MODE")) {
            const std::string_view mode(value);
            if (mode == "FULL") {
                result.options.mode = MatmulInvocationMode::full;
            } else if (mode == "STRIPE_PIPELINE") {
                result.options.mode = MatmulInvocationMode::stripe_pipeline;
            } else {
                result.error = MatmulOptionsError::invalid_mode;
                return result;
            }
        }
        if (!explicit_options.job_capacity) if (const char * value = std::getenv("GEMMINI_STRIPE_JOB_CAPACITY")) {
            if (!parse_positive_size(value, result.options.job_capacity)) {
                result.error = MatmulOptionsError::invalid_job_capacity;
                return result;
            }
        }
        if (!explicit_options.rmd_backend) if (const char * value = std::getenv("GEMMINI_RMD_BACKEND")) {
            const std::string_view backend(value);
            if (backend == "CPU") {
                result.options.rmd_backend = RmdBackend::cpu_direct;
            } else if (backend == "WS") {
                result.options.rmd_backend = RmdBackend::gemmini_ws_compact;
            } else {
                result.error = MatmulOptionsError::invalid_rmd_backend;
                return result;
            }
            result.rmd_backend_source = MatmulOptionSource::environment;
        }
    }

    if (explicit_options.mode) result.options.mode = *explicit_options.mode;
    if (explicit_options.job_capacity) result.options.job_capacity = *explicit_options.job_capacity;
    if (explicit_options.rmd_backend) {
        result.options.rmd_backend = *explicit_options.rmd_backend;
        result.rmd_backend_source = MatmulOptionSource::explicit_override;
    }
    result.options.dense_threads = explicit_options.dense_threads;
    result.options.validation = explicit_options.validation;
    result.options.profiling = explicit_options.profiling;

    if (explicit_options.job_capacity && *explicit_options.job_capacity == 0) {
        result.error = MatmulOptionsError::invalid_job_capacity;
        return result;
    }
    if (result.options.rmd_backend != RmdBackend::cpu_direct && result.options.rmd_backend != RmdBackend::gemmini_ws_compact) {
        result.error = MatmulOptionsError::invalid_rmd_backend;
        return result;
    }
    if (result.options.mode != MatmulInvocationMode::full &&
        result.options.mode != MatmulInvocationMode::stripe_pipeline) {
        result.error = MatmulOptionsError::invalid_mode;
        return result;
    }
    if (result.options.mode == MatmulInvocationMode::stripe_pipeline &&
        (!config::ENABLE_STRIPE_MATMUL || !config::ENABLE_STRIPE_PIPELINE)) {
        result.error = MatmulOptionsError::disabled_mode;
    }
    return result;
}

class MatmulStripeInput {
public:
    MatmulStripeInput(size_t row_begin, size_t row_end);
    MatmulStripeInput(size_t row_begin, size_t row_end, size_t stripe_id,
                      const int32_t * residual = nullptr, size_t residual_count = 0);
    MatmulStripeInput(const MatmulStripeInput &) = default;
    MatmulStripeInput & operator=(const MatmulStripeInput &) = default;
    MatmulStripeInput(MatmulStripeInput &&) noexcept = default;
    MatmulStripeInput & operator=(MatmulStripeInput &&) noexcept = default;

    size_t row_begin() const;
    size_t row_end() const;
    size_t stripe_id() const;
    const int32_t * residual() const;
    size_t residual_count() const;

private:
    size_t row_begin_;
    size_t row_end_;
    size_t stripe_id_;
    const int32_t * residual_;
    size_t residual_count_;
};

class MatmulExecution {
public:
    MatmulExecution();
    explicit MatmulExecution(MatmulStatus status);
    MatmulExecution(const MatmulExecution &) = delete;
    MatmulExecution & operator=(const MatmulExecution &) = delete;
    MatmulExecution(MatmulExecution &&) noexcept;
    MatmulExecution & operator=(MatmulExecution &&) noexcept;
    ~MatmulExecution();

    MatmulInvocationMode mode() const;
    MatmulExecutionState state() const;
    MatmulStatus status() const;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    bool test_pipeline_attached() const;
#endif

private:
    friend MatmulExecution prepare_execution(const ggml_gemmini_args_t &, ResolvedMatmulOptions);
    friend MatmulExecution prepare_execution(ggml_gemmini_args_t *, ResolvedMatmulOptions);
    friend MatmulStatus prepare_execution(ggml_gemmini_args_t &, const ResolvedMatmulOptions &, MatmulExecution &);
    friend MatmulStatus execute_full(MatmulExecution &);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput, rmd::StripePacketHandle);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput,
                                          residual::DirectStripePayloadHandle,
                                          rmd::StripePacketHandle);
    friend MatmulStatus capture_stripe(MatmulExecution &, const MatmulStripeInput &, MatmulStripeJob &);
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus accept_external_dense_completion(MatmulStripeJob &);
    friend MatmulStatus execute_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus compose_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus finalize_stripe(MatmulStripeJob &);
    friend class MatmulStripeCollector;
    friend class MatmulStripeJob;
    friend MatmulStatus finish_execution(MatmulExecution &);

    MatmulExecution(ggml_gemmini_args_t args, ResolvedMatmulOptions options);
    MatmulExecution(ggml_gemmini_args_t * args, ResolvedMatmulOptions options);
    void assert_pipeline_detached() const;

    size_t total_rows_;
    MatMul facade_;
    ResolvedMatmulOptions options_;
    MatmulStatus status_;
    MatmulExecutionState state_ = MatmulExecutionState::empty;
    std::shared_ptr<std::mutex> state_mutex_ = std::make_shared<std::mutex>();
    size_t active_jobs_ = 0;
    size_t captured_rows_ = 0;
    size_t finalized_rows_ = 0;
    size_t first_row_ = 0;
    size_t last_row_begin_ = 0;
    size_t last_row_end_ = 0;
    bool has_captures_ = false;
    std::unordered_set<size_t> captured_stripe_ids_;
    bool pipeline_attached_ = false;
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
    MatmulStatus status;
    MatmulJobMetrics metrics;
    MatmulDenseState dense = MatmulDenseState::idle;
    MatmulResidualState residual = MatmulResidualState::idle;
    bool captured = false;
    bool finalized = false;
    bool released = false;
};

class MatmulStripeJob {
public:
    MatmulStripeJob();
    MatmulStripeJob(const MatmulStripeJob &) = delete;
    MatmulStripeJob & operator=(const MatmulStripeJob &) = delete;
    MatmulStripeJob(MatmulStripeJob && other) noexcept;
    MatmulStripeJob & operator=(MatmulStripeJob && other) noexcept;
    ~MatmulStripeJob();

    MatmulStatus status() const;
    MatmulJobMetrics metrics() const;
    MatmulStripeJobSnapshot snapshot() const;

private:
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput, rmd::StripePacketHandle);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput,
                                          residual::DirectStripePayloadHandle,
                                          rmd::StripePacketHandle);
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus accept_external_dense_completion(MatmulStripeJob &);
    friend MatmulStatus execute_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus compose_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus finalize_stripe(MatmulStripeJob &);
    friend MatmulStatus execute_post_fold_pipeline(
        const ggml_gemmini_args_t &, MatmulStripeCollector &);
    friend class MatmulStripeCollector;

    MatmulStripeJob(MatmulExecution * execution, MatmulStripeInput input, MatmulStatus status,
                    residual::DirectStripePayloadHandle direct_residual = nullptr,
                    rmd::StripePacketHandle rmd_packet = nullptr);
    void cancel(MatmulStatus status);
    void release_slot();
    void record_failure(MatmulStatus status, bool dense_branch);

    MatmulExecution * execution_;
    MatmulStripeInput input_;
    MatmulStatus status_;
    MatmulJobMetrics metrics_;
    std::unique_ptr<quants::act::Meta> staged_activation_meta_;
    bool owns_slot_ = false;
    bool released_ = false;
    bool collector_slot_released_ = false;
    uint64_t rmd_queued_ns_ = 0;
    std::shared_ptr<std::mutex> job_mutex_ = std::make_shared<std::mutex>();
    std::condition_variable lifecycle_condition_;
    residual::DirectStripePayloadHandle direct_residual_;
    rmd::StripePacketHandle rmd_packet_;
    rmd::CompressedOutput rmd_output_;
    rmd::Correction rmd_correction_ = rmd::BlockScaledInt64Correction{};
    MatmulDenseState dense_state_ = MatmulDenseState::idle;
    MatmulResidualState residual_state_ = MatmulResidualState::idle;
    bool captured_ = true;
    bool finalized_ = false;
};

MatmulExecution prepare_execution(const ggml_gemmini_args_t & args, ResolvedMatmulOptions options);
MatmulExecution prepare_execution(ggml_gemmini_args_t * args, ResolvedMatmulOptions options);
MatmulStatus prepare_execution(ggml_gemmini_args_t & args, const ResolvedMatmulOptions & options,
                               MatmulExecution & execution);
MatmulStatus execute_full(MatmulExecution & execution);
MatmulStatus capture_stripe(MatmulExecution & execution, const MatmulStripeInput & input,
                            MatmulStripeJob & job);
MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input);
MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input,
                               rmd::StripePacketHandle rmd_packet);
MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input,
                               residual::DirectStripePayloadHandle direct_residual,
                               rmd::StripePacketHandle rmd_packet);
MatmulStatus capture_stripe(MatmulExecution & execution, const MatmulStripeInput & input,
                            MatmulStripeJob & job);
MatmulStatus execute_dense_stripe(MatmulStripeJob & job);
MatmulStatus accept_external_dense_completion(MatmulStripeJob &job);
MatmulStatus execute_rmd_stripe(MatmulStripeJob & job);
MatmulStatus compose_rmd_stripe(MatmulStripeJob & job);
MatmulStatus finalize_stripe(MatmulStripeJob & job);
MatmulStatus finish_execution(MatmulExecution & execution);
MatmulStatus matmul(ggml_gemmini_args_t & args, ResolvedMatmulOptions options);
MatmulStatus matmul(const ggml_gemmini_args_t & args, ResolvedMatmulOptions options);
MatmulStatus execute_post_fold_pipeline(const ggml_gemmini_args_t & args, MatmulStripeCollector & collector);

#ifndef GGML_GEMMINI_MATMUL_IMPLEMENTATION
inline MatmulStatus resolution_status(MatmulOptionsError error) {
    switch (error) {
        case MatmulOptionsError::disabled_mode:
            return { MatmulStatusCode::unsupported_invocation,
                     "requested matmul mode is disabled in this build",
                     MatMulCapability::unsupported };
        case MatmulOptionsError::none:
            return {};
        case MatmulOptionsError::invalid_mode:
        case MatmulOptionsError::invalid_job_capacity:
        case MatmulOptionsError::invalid_rmd_backend:
        case MatmulOptionsError::runtime_override_disabled:
            return { MatmulStatusCode::invalid_argument, "invalid matmul options" };
    }
    return { MatmulStatusCode::invalid_argument, "invalid matmul options" };
}

inline MatmulExecution prepare_execution(const ggml_gemmini_args_t & args, MatmulOptions options = {}) {
    const auto resolution = resolve_matmul_options(options);
    return resolution.ok() ? prepare_execution(args, resolution.options)
                           : MatmulExecution(resolution_status(resolution.error));
}

inline MatmulExecution prepare_execution(ggml_gemmini_args_t * args, MatmulOptions options = {}) {
    const auto resolution = resolve_matmul_options(options);
    return resolution.ok() ? prepare_execution(args, resolution.options)
                           : MatmulExecution(resolution_status(resolution.error));
}

inline MatmulStatus prepare_execution(ggml_gemmini_args_t & args, const MatmulOptions & options,
                                      MatmulExecution & execution) {
    const auto resolution = resolve_matmul_options(options);
    if (!resolution.ok()) {
        return resolution_status(resolution.error);
    }
    return prepare_execution(args, resolution.options, execution);
}

inline MatmulStatus matmul(ggml_gemmini_args_t & args, MatmulOptions options = {}) {
    const auto resolution = resolve_matmul_options(options);
    return resolution.ok() ? matmul(args, resolution.options)
                           : resolution_status(resolution.error);
}

inline MatmulStatus matmul(const ggml_gemmini_args_t & args, MatmulOptions options = {}) {
    const auto resolution = resolve_matmul_options(options);
    return resolution.ok() ? matmul(args, resolution.options)
                           : resolution_status(resolution.error);
}
#endif

}
