#pragma once

#include "ggml-gemmini-args.h"
#include "ggml-gemmini-matmul-config.hpp"
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
std::string pipeline_stripe_summary_json(const char * layer,
                                         size_t I,
                                         size_t J,
                                         size_t K,
                                         const char * backend_route,
                                         const char * schedule,
                                         const MatmulJobMetrics & profile);
bool append_pipeline_stripe_summary_jsonl(const std::string & json_record);

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
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus execute_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus compose_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus finalize_stripe(MatmulStripeJob &);

    MatMulStatus run_stripe(MatMulStripe stripe, size_t stripe_id);
    MatMulStatus run_staged_stripe(MatMulStripe stripe, size_t stripe_id);

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
    uint64_t la_cycles = 0;
    uint64_t la3_cycles = 0;
    uint64_t sf_cycles = 0;
    uint64_t la3_ns = 0;
    uint64_t sf1_ns = 0;
    std::array<uint64_t, 3> la_worker_start_ns{};
    std::array<uint64_t, 3> la_worker_end_ns{};
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
    uint64_t merge_start_ns = 0;
    uint64_t merge_end_ns = 0;
};

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
    struct CapturedStripe {
        uint64_t run_id = 0;
        size_t stripe_id;
        size_t slot = 0;
        size_t row_begin;
        size_t row_end;
        rmd::StripePacketHandle rmd_packet;
        uint64_t la_cycles = 0;
        uint64_t la3_cycles = 0;
        uint64_t sf_cycles = 0;
        uint64_t la3_ns = 0;
        uint64_t sf1_ns = 0;
        std::array<uint64_t, 3> la_worker_start_ns{};
        std::array<uint64_t, 3> la_worker_end_ns{};
        uint64_t sf_mask_start_ns = 0;
        uint64_t sf_mask_end_ns = 0;
        uint64_t sf_exponent_start_ns = 0;
        uint64_t sf_exponent_end_ns = 0;
        uint64_t sf_folding_start_ns = 0;
        uint64_t sf_folding_end_ns = 0;
        uint64_t sf_commit_ns = 0;
        MatmulStageMetrics capture_copy;
        MatmulStageMetrics producer_wait;
        MatmulStageMetrics queue_insert;
        MatmulStageMetrics rmd_pack;
        uint64_t producer_wait_start_ns = 0;
        uint64_t producer_wait_end_ns = 0;
        uint64_t queued_ns = 0;
    };
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
    stripe_sequential,
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

struct ResolvedMatmulOptions {
    MatmulInvocationMode mode = static_cast<MatmulInvocationMode>(config::DEFAULT_MATMUL_MODE);
    size_t stripe_rows = config::DEFAULT_STRIPE_ROWS.value_or(1);
    bool stripe_rows_auto = !config::DEFAULT_STRIPE_ROWS.has_value();
    size_t dense_threads = 0;
    bool validation = false;
    bool profiling = false;
    size_t job_capacity = config::DEFAULT_STRIPE_JOB_CAPACITY;

    ResolvedMatmulOptions() {}
};

struct MatmulOptionOverrides {
    std::optional<MatmulInvocationMode> mode;
    std::optional<size_t> stripe_rows;
    size_t dense_threads = 0;
    bool validation = false;
    bool profiling = false;
    std::optional<size_t> job_capacity;
};

#ifdef GGML_GEMMINI_MATMUL_IMPLEMENTATION
using MatmulOptions = ResolvedMatmulOptions;
#else
using MatmulOptions = MatmulOptionOverrides;
#endif

enum class MatmulOptionsError : uint8_t {
    none,
    invalid_mode,
    invalid_stripe_rows,
    invalid_job_capacity,
    disabled_mode,
};

struct MatmulOptionsResolution {
    ResolvedMatmulOptions options;
    MatmulOptionsError error = MatmulOptionsError::none;

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
    if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE) {
        if (!explicit_options.mode) if (const char * value = std::getenv("GEMMINI_MATMUL_MODE")) {
            const std::string_view mode(value);
            if (mode == "FULL") {
                result.options.mode = MatmulInvocationMode::full;
            } else if (mode == "STRIPE_SEQUENTIAL") {
                result.options.mode = MatmulInvocationMode::stripe_sequential;
            } else if (mode == "STRIPE_PIPELINE") {
                result.options.mode = MatmulInvocationMode::stripe_pipeline;
            } else {
                result.error = MatmulOptionsError::invalid_mode;
                return result;
            }
        }
        if (!explicit_options.stripe_rows) if (const char * value = std::getenv("GEMMINI_STRIPE_ROWS")) {
            if (std::string_view(value) == "AUTO") {
                result.options.stripe_rows_auto = true;
            } else if (size_t rows; parse_positive_size(value, rows)) {
                result.options.stripe_rows = rows;
                result.options.stripe_rows_auto = false;
            } else {
                result.error = MatmulOptionsError::invalid_stripe_rows;
                return result;
            }
        }
        if (!explicit_options.job_capacity) if (const char * value = std::getenv("GEMMINI_STRIPE_JOB_CAPACITY")) {
            if (!parse_positive_size(value, result.options.job_capacity)) {
                result.error = MatmulOptionsError::invalid_job_capacity;
                return result;
            }
        }
    }

    if (explicit_options.mode) result.options.mode = *explicit_options.mode;
    if (explicit_options.stripe_rows) {
        result.options.stripe_rows = *explicit_options.stripe_rows;
        result.options.stripe_rows_auto = false;
    }
    if (explicit_options.job_capacity) result.options.job_capacity = *explicit_options.job_capacity;
    result.options.dense_threads = explicit_options.dense_threads;
    result.options.validation = explicit_options.validation;
    result.options.profiling = explicit_options.profiling;

    if ((explicit_options.stripe_rows && *explicit_options.stripe_rows == 0) ||
        (explicit_options.job_capacity && *explicit_options.job_capacity == 0)) {
        result.error = explicit_options.stripe_rows && *explicit_options.stripe_rows == 0
            ? MatmulOptionsError::invalid_stripe_rows
            : MatmulOptionsError::invalid_job_capacity;
        return result;
    }
    if ((result.options.mode != MatmulInvocationMode::full && !config::ENABLE_STRIPE_MATMUL) ||
        (result.options.mode == MatmulInvocationMode::stripe_pipeline && !config::ENABLE_STRIPE_PIPELINE)) {
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
    friend MatmulStatus capture_stripe(MatmulExecution &, const MatmulStripeInput &, MatmulStripeJob &);
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
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
    std::unique_ptr<MatMul> staged_facade_;
    bool staged_metadata_active_ = false;
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
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus execute_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus compose_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus finalize_stripe(MatmulStripeJob &);
    friend class MatmulStripeCollector;

    MatmulStripeJob(MatmulExecution * execution, MatmulStripeInput input, MatmulStatus status,
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
    rmd::StripePacketHandle rmd_packet_;
    rmd::CompressedOutput rmd_output_;
    std::vector<rmd::OutputValue> rmd_correction_;
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
MatmulStatus capture_stripe(MatmulExecution & execution, const MatmulStripeInput & input,
                            MatmulStripeJob & job);
MatmulStatus execute_dense_stripe(MatmulStripeJob & job);
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
        case MatmulOptionsError::invalid_stripe_rows:
        case MatmulOptionsError::invalid_job_capacity:
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
