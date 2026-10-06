#include <gemmini/trace-context.hpp>
#include "execution.hpp"
#include "detail.hpp"
#include "../ggml-gemmini-geometry.hpp"
#include <gemmini/log.hpp>
#include <gemmini/performance.hpp>

#include "../quants/act/quantize.hpp"
#include "../quants/act/dispatch.hpp"
#include "../residual/rmd/rmd-builder.hpp"
#include "../residual/direct/direct-builder.hpp"
#include "../residual/direct/direct-executor.hpp"

#include <gemmini.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <new>
#include <sstream>
#include <cstring>
#include <tuple>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ggml::gemmini {

using detail::MatmulCycleDrain;
using detail::record_matmul_cpu_wall;
using detail::emit_matmul_cpu_interval;
using detail::emit_rmd_stripe_metrics;

namespace {

GemminiGeometryResult resolve_geometry(ggml_gemmini_args_t & args) {
    if (args.tile_I == 0 || args.tile_J == 0 || args.tile_K == 0) {
        gemmini_set_tile_ws(&args);
    }
    return make_gemmini_geometry(
        {{args.I, args.J, args.K}, {args.tile_I, args.tile_J, args.tile_K}, DIM});
}

MatmulStatus to_public_status(MatMulStatus                status,
                              MatMulCapability            capability,
                              const ggml_gemmini_args_t * args = nullptr) {
    MatmulStatusCode code    = MatmulStatusCode::invalid_state;
    const char *     message = "invalid state";
    switch (status) {
    case MatMulStatus::success:
        code    = MatmulStatusCode::success;
        message = "success";
        break;
    case MatMulStatus::empty_stripes:
        code    = MatmulStatusCode::invalid_contract;
        message = "missing stripes";
        break;
    case MatMulStatus::malformed_stripe:
        code    = MatmulStatusCode::invalid_argument;
        message = "invalid stripe bounds";
        break;
    case MatMulStatus::duplicate_stripe:
        code    = MatmulStatusCode::invalid_contract;
        message = "duplicate stripe";
        break;
    case MatMulStatus::overlapping_stripe:
        code    = MatmulStatusCode::invalid_contract;
        message = "overlapping stripe";
        break;
    case MatMulStatus::missing_stripes:
        code    = MatmulStatusCode::invalid_contract;
        message = "missing stripes";
        break;
    case MatMulStatus::unsupported:
        if (args != nullptr) {
            const auto backend = detail::normalize_route(*args).backend;
            if (backend == detail::BackendRoute::gemmini_os ||
                backend == detail::BackendRoute::ws_sim) {
                code    = MatmulStatusCode::unsupported_backend;
                message = "unsupported Gemmini backend";
                break;
            }
        }
        code    = MatmulStatusCode::unsupported_route;
        message = "unsupported route";
        break;
    case MatMulStatus::invalid_contract:
        code    = MatmulStatusCode::invalid_contract;
        message = "invalid route contract";
        break;
    case MatMulStatus::invalid_state:
        break;
    case MatMulStatus::invalid_arguments:
        code    = MatmulStatusCode::invalid_argument;
        message = "invalid argument";
        break;
    }
    return {code, message, capability};
}

MatmulStatus make_status(MatmulStatusCode code,
                         const char *     message,
                         MatMulCapability capability = MatMulCapability::supported) {
    return {code, message, capability};
}

MatmulStatus invalid_state(const char * message = "invalid state") {
    return make_status(MatmulStatusCode::invalid_state, message);
}

MatmulStatus invalid_contract(const char * message) {
    return make_status(MatmulStatusCode::invalid_contract, message);
}

MatmulStatus unsupported_backend(const char * message) {
    return make_status(
        MatmulStatusCode::unsupported_backend, message, MatMulCapability::unsupported);
}

bool residual_backend_available(RmdBackend backend) {
#if defined(__riscv) || defined(GGML_GEMMINI_TESTING)
    (void)backend;
    return true;
#else
    constexpr bool im2p_build =
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
        true;
#else
        false;
#endif
    return backend == RmdBackend::cpu_direct ||
           (backend == RmdBackend::gemmini_ws_compact &&
            rmd::compact_rmd_backend_available(false, im2p_build));
#endif
}

residual::ResidualRoute residual_route_for(RmdBackend backend) {
    return backend == RmdBackend::cpu_direct ? residual::ResidualRoute::cpu_direct
                                             : residual::ResidualRoute::ws_packet;
}

MatmulStatus validate_exsia_residual_route(const ggml_gemmini_args_t &   args,
                                           const ResolvedMatmulOptions & options) {
    if (detail::normalize_route(args).activation != detail::ActivationRoute::exsia) {
        return {};
    }
    using Format      = ggml_gemmini_args_t::im2p_weight_format_t;
    const auto format = args.weight_format;
    if (format == Format::q8_h2 || format == Format::q8_hp2) {
        return make_status(MatmulStatusCode::unsupported_route,
                           "H2/HP2 ExSIA residual formats are unsupported",
                           MatMulCapability::unsupported);
    }
    const bool h0 = format == Format::q4_h0 || format == Format::q8_h0 || format == Format::q16_h0;
    if (h0 && options.rmd_backend == RmdBackend::gemmini_ws_compact) {
        return make_status(MatmulStatusCode::unsupported_route,
                           "H0 ExSIA requires CPU-direct residual execution",
                           MatMulCapability::unsupported);
    }
    const std::uint8_t weight_bits =
        format == Format::q4_h0 || format == Format::q4_hp1                                 ? 4
        : format == Format::q16_h0 || format == Format::q16_h1 || format == Format::q16_hp1 ? 16
                                                                                            : 8;
    if (args.A.valid() && args.A.bits != weight_bits) {
        return make_status(MatmulStatusCode::unsupported_route,
                           "ExSIA requires matched activation and weight widths",
                           MatMulCapability::unsupported);
    }
    return {};
}

MatmulStatus from_rmd_status(rmd::RmdStatus status) {
    switch (status) {
    case rmd::RmdStatus::success:
        return {};
    case rmd::RmdStatus::unsupported_route:
        return unsupported_backend(rmd::rmd_status_message(status));
    case rmd::RmdStatus::invalid_arguments:
    case rmd::RmdStatus::invalid_packet:
        return make_status(MatmulStatusCode::invalid_argument,
                           rmd::rmd_status_message(status),
                           MatMulCapability::unsupported);
    case rmd::RmdStatus::allocation_failure:
        return make_status(MatmulStatusCode::out_of_memory, rmd::rmd_status_message(status));
    case rmd::RmdStatus::residual_too_wide:
    case rmd::RmdStatus::overflow:
    case rmd::RmdStatus::execution_failed:
        return make_status(MatmulStatusCode::execution_failure, rmd::rmd_status_message(status));
    }
    return make_status(MatmulStatusCode::execution_failure, "rmd: unknown status");
}

struct Clock {
    using time_point = std::chrono::steady_clock::time_point;
    static time_point now() {
#if CYCLE_DETAIL
        return time_point(std::chrono::nanoseconds(cycle::timestamp_ns()));
#else
        return time_point{};
#endif
    }
};

void record_metric(MatmulStageMetrics & metric, bool enabled, Clock::time_point start) {
    if (!enabled) {
        return;
    }
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    metric.nanoseconds += static_cast<uint64_t>(std::max<int64_t>(1, elapsed));
    ++metric.count;
}

uint64_t now_ns() {
#if CYCLE_DETAIL
    return cycle::timestamp_ns();
#else
    return 0;
#endif
}

} // namespace

namespace detail {

MatmulCapturedStripe capture_collector_event(const quants::act::exsia::StripeReadyEvent & event,
                                             MatmulCaptureTiming                          timing,
                                             std::optional<uint64_t>                      run_id) {
    run_id = matmul_cpu_run_id(event, run_id);
    MatmulCapturedStripe captured{};
    captured.trace_origin      = gemmini_trace_capture();
    captured.cpu_identity_mask = GEMMINI_CYCLE_HAS_STRIPE_ID;
    if (run_id.has_value())
        captured.cpu_identity_mask |= GEMMINI_CYCLE_HAS_RUN_ID | GEMMINI_CYCLE_HAS_SLOT;
    captured.run_id              = run_id.value_or(event.run_id);
    captured.stripe_id           = event.stripe_id;
    captured.slot                = event.slot;
    captured.row_begin           = event.row_begin;
    captured.row_end             = event.row_end;
    captured.activation_metadata = event.activation_metadata;
#if GGML_GEMMINI_ENABLE_RMD
    captured.rmd_packet      = event.rmd_packet;
    captured.direct_residual = event.direct_residual;
    if (event.rmd_packet != nullptr || event.direct_residual != nullptr) {
        timing.rmd_pack.nanoseconds = event.rmd_pack_ns;
        timing.rmd_pack.count       = 1;
    }
#endif
    captured.la3_ns =
        event.local_end_ns >= event.local_start_ns ? event.local_end_ns - event.local_start_ns : 0;
    captured.sf1_ns               = event.folding_end_ns >= event.folding_start_ns
                                        ? event.folding_end_ns - event.folding_start_ns
                                        : 0;
    captured.sf_mask_start_ns     = event.mask_assembly_start_ns;
    captured.sf_mask_end_ns       = event.mask_assembly_end_ns;
    captured.sf_exponent_start_ns = event.exponent_reduction_start_ns;
    captured.sf_exponent_end_ns   = event.exponent_reduction_end_ns;
    captured.sf_folding_start_ns  = event.folding_start_ns;
    captured.sf_folding_end_ns    = event.folding_end_ns;
    captured.sf_commit_ns         = event.folding_commit_ns;
    captured.timing               = std::move(timing);
    return captured;
}

void apply_captured_stripe(const MatmulCapturedStripe & captured, MatmulJobMetrics & profile) {
    profile.cpu_identity_mask        = captured.cpu_identity_mask;
    profile.run_id                   = captured.run_id;
    profile.stripe_id                = captured.stripe_id;
    profile.slot                     = captured.slot;
    profile.row_begin                = captured.row_begin;
    profile.row_end                  = captured.row_end;
    profile.la3_ns                   = captured.la3_ns;
    profile.sf1_ns                   = captured.sf1_ns;
    profile.sf_mask_start_ns         = captured.sf_mask_start_ns;
    profile.sf_mask_end_ns           = captured.sf_mask_end_ns;
    profile.sf_exponent_start_ns     = captured.sf_exponent_start_ns;
    profile.sf_exponent_end_ns       = captured.sf_exponent_end_ns;
    profile.sf_folding_start_ns      = captured.sf_folding_start_ns;
    profile.sf_folding_end_ns        = captured.sf_folding_end_ns;
    profile.sf_commit_ns             = captured.sf_commit_ns;
    profile.la                       = {captured.la3_ns, captured.la3_ns != 0 ? 1U : 0U};
    profile.sf                       = {captured.sf1_ns, captured.sf1_ns != 0 ? 1U : 0U};
    profile.capture_copy             = captured.timing.capture_copy;
    profile.producer_wait            = captured.timing.producer_wait;
    profile.queue_insert             = captured.timing.queue_insert;
    profile.rmd_pack                 = captured.timing.rmd_pack;
    profile.producer_wait_start_ns   = captured.timing.producer_wait_start_ns;
    profile.producer_wait_end_ns     = captured.timing.producer_wait_end_ns;
    profile.capture_queue_enqueue_ns = captured.timing.queued_ns;
    profile.capture_queue_dequeue_ns = captured.timing.dequeued_ns;
    profile.queue_enqueue_tid        = captured.timing.enqueue_tid;
    profile.queue_dequeue_tid        = captured.timing.dequeue_tid;
    profile.telemetry_queue_tick     = captured.timing.telemetry_queued_tick;
    profile.sf_handoff.nanoseconds   = captured.sf1_ns + profile.handoff.nanoseconds;
    profile.sf_handoff.count         = 1;
}

} // namespace detail

MatmulStripeInput::MatmulStripeInput(size_t row_begin, size_t row_end)
    : row_begin_(row_begin), row_end_(row_end), stripe_id_(row_begin), residual_(nullptr),
      residual_count_(0) {}

MatmulStripeInput::MatmulStripeInput(size_t          row_begin,
                                     size_t          row_end,
                                     size_t          stripe_id,
                                     const int32_t * residual,
                                     size_t          residual_count)
    : row_begin_(row_begin), row_end_(row_end), stripe_id_(stripe_id), residual_(residual),
      residual_count_(residual_count) {}

size_t MatmulStripeInput::row_begin() const {
    return row_begin_;
}

size_t MatmulStripeInput::row_end() const {
    return row_end_;
}

size_t MatmulStripeInput::stripe_id() const {
    return stripe_id_;
}

const int32_t * MatmulStripeInput::residual() const {
    return residual_;
}

size_t MatmulStripeInput::residual_count() const {
    return residual_count_;
}

MatmulExecution::MatmulExecution(ggml_gemmini_args_t args, ResolvedMatmulOptions options)
    : total_rows_(args.I), facade_(std::move(args)), options_(options) {
    initialize();
}

MatmulExecution::MatmulExecution()
    : total_rows_(0), facade_(static_cast<ggml_gemmini_args_t *>(nullptr)) {
    status_ = invalid_state("execution is not prepared");
}

MatmulExecution::MatmulExecution(MatmulStatus status)
    : total_rows_(0), facade_(static_cast<ggml_gemmini_args_t *>(nullptr)), status_(status),
      state_(status.ok() ? MatmulExecutionState::prepared : MatmulExecutionState::failed) {}

MatmulExecution::MatmulExecution(MatmulExecution && other) noexcept : MatmulExecution() {
    *this = std::move(other);
}

MatmulExecution::MatmulExecution(ggml_gemmini_args_t * args, ResolvedMatmulOptions options)
    : total_rows_(args != nullptr ? args->I : 0), facade_(args), options_(options) {
    initialize();
}

void MatmulExecution::initialize() {
    test_detail::observe_execution_construction();
    state_ = MatmulExecutionState::prepared;
    if (facade_.args_ptr_ == nullptr) {
        status_ = make_status(MatmulStatusCode::invalid_argument, "null execution args");
        state_  = MatmulExecutionState::failed;
        return;
    }
    if (!resolve_geometry(facade_.args()).ok()) {
        status_ = invalid_contract("invalid Gemmini geometry");
        state_  = MatmulExecutionState::failed;
        return;
    }
    status_ = validate_exsia_residual_route(facade_.args(), options_);
    if (!status_.ok()) {
        state_ = MatmulExecutionState::failed;
        return;
    }
    facade_.args().residual_route = residual_route_for(options_.rmd_backend);
    if (!residual_backend_available(options_.rmd_backend)) {
        status_ = unsupported_backend("RMD WS backend is unavailable on this host");
        state_  = MatmulExecutionState::failed;
        return;
    }
    if (options_.dense_threads > 1) {
        status_ = make_status(MatmulStatusCode::unsupported_invocation,
                              "dense stripe execution has one owner lane");
        state_  = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode != MatmulInvocationMode::full && options_.job_capacity == 0) {
        status_ = make_status(MatmulStatusCode::invalid_argument, "job capacity must be nonzero");
        state_  = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode == MatmulInvocationMode::stripe_pipeline &&
        !std::holds_alternative<quants::act::NoneMeta>(facade_.args().act_quant.storage()) &&
        !detail::route_capabilities(facade_.args()).live_stripe_producer) {
        status_ = make_status(MatmulStatusCode::unsupported_invocation,
                              "stripe pipeline requires an ExSIA live producer route");
        state_  = MatmulExecutionState::failed;
        return;
    }
    const bool defer_pipeline_route_validation =
        options_.mode == MatmulInvocationMode::stripe_pipeline &&
        std::holds_alternative<quants::act::NoneMeta>(facade_.args().act_quant.storage());
    if (options_.mode == MatmulInvocationMode::stripe_pipeline &&
        !defer_pipeline_route_validation) {
        const MatMulStatus status = facade_.begin_stripes();
        status_ =
            to_public_status(status,
                             status == MatMulStatus::unsupported ? MatMulCapability::unsupported
                                                                 : MatMulCapability::supported,
                             &facade_.args());
        if (!status_.ok()) {
            state_ = MatmulExecutionState::failed;
        }
    }
}

MatmulExecution & MatmulExecution::operator=(MatmulExecution && other) noexcept {
    if (this == &other) {
        return *this;
    }
    assert_pipeline_detached();
    other.assert_pipeline_detached();
    total_rows_              = other.total_rows_;
    facade_                  = std::move(other.facade_);
    options_                 = other.options_;
    status_                  = other.status_;
    state_                   = other.state_;
    state_mutex_             = std::move(other.state_mutex_);
    rmd_weights_             = std::move(other.rmd_weights_);
    active_jobs_             = other.active_jobs_;
    captured_rows_           = other.captured_rows_;
    finalized_rows_          = other.finalized_rows_;
    first_row_               = other.first_row_;
    last_row_begin_          = other.last_row_begin_;
    last_row_end_            = other.last_row_end_;
    has_captures_            = other.has_captures_;
    captured_stripe_ids_     = std::move(other.captured_stripe_ids_);
    pipeline_attached_       = other.pipeline_attached_;
    other.total_rows_        = 0;
    other.active_jobs_       = 0;
    other.captured_rows_     = 0;
    other.finalized_rows_    = 0;
    other.first_row_         = 0;
    other.last_row_begin_    = 0;
    other.last_row_end_      = 0;
    other.has_captures_      = false;
    other.pipeline_attached_ = false;
    return *this;
}

MatmulExecution::~MatmulExecution() {
    assert_pipeline_detached();
}

MatmulInvocationMode MatmulExecution::mode() const {
    return options_.mode;
}

MatmulExecutionState MatmulExecution::state() const {
    std::lock_guard<std::mutex> lock(*state_mutex_);
    return state_;
}

MatmulStatus MatmulExecution::status() const {
    std::lock_guard<std::mutex> lock(*state_mutex_);
    return status_;
}

void MatmulExecution::assert_pipeline_detached() const {
    if (!state_mutex_) {
        return;
    }
    std::lock_guard<std::mutex> lock(*state_mutex_);
    GGML_ASSERT(!pipeline_attached_);
    GGML_ASSERT(active_jobs_ == 0);
}

#if defined(GGML_GEMMINI_TEST_OBSERVER)
bool MatmulExecution::test_pipeline_attached() const {
    std::lock_guard<std::mutex> lock(*state_mutex_);
    return pipeline_attached_;
}
#endif

namespace {
#if defined(GGML_GEMMINI_TEST_OBSERVER)
MatmulStripeCollector * test_residual_failure_collector = nullptr;
MatmulStatus            test_residual_failure;
MatmulStripeCollector * test_dense_observer_collector   = nullptr;
MatmulDenseState        observed_dense_state_at_release = MatmulDenseState::idle;
#endif

bool dense_state_is_terminal(MatmulDenseState state) {
    return state == MatmulDenseState::complete || state == MatmulDenseState::failed ||
           state == MatmulDenseState::cancelled;
}
} // namespace

MatmulStripeCollector::MatmulStripeCollector(size_t capacity)
    : capacity_(capacity), sink_{this, &MatmulStripeCollector::on_ready} {
    if (capacity == 0) {
        status_ =
            make_status(MatmulStatusCode::invalid_argument, "collector capacity must be nonzero");
    }
}

MatmulStripeCollector::~MatmulStripeCollector() {
    finish();
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    std::lock_guard<std::mutex> lock(mutex_);
    if (test_residual_failure_collector == this) {
        test_residual_failure_collector = nullptr;
        test_residual_failure           = {};
    }
    if (test_dense_observer_collector == this) {
        test_dense_observer_collector   = nullptr;
        observed_dense_state_at_release = MatmulDenseState::idle;
    }
#endif
}

bool MatmulStripeCollector::start(MatmulExecution & execution) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!status_ || worker_started_ || startup_in_progress_ ||
            execution.mode() != MatmulInvocationMode::stripe_pipeline) {
            return false;
        }
        worker_started_      = true;
        startup_in_progress_ = true;
    }
    {
        std::lock_guard<std::mutex> execution_lock(*execution.state_mutex_);
        if (execution.pipeline_attached_) {
            std::lock_guard<std::mutex> lock(mutex_);
            status_              = invalid_state("execution already has a live stripe collector");
            worker_started_      = false;
            startup_in_progress_ = false;
            condition_.notify_all();
            return false;
        }
        execution.pipeline_attached_ = true;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        execution_      = &execution;
        dense_done_     = false;
        stop_requested_ = false;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        test_thread_start_attempts_ = 0;
#endif
    }
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (test_pause_startup_) {
            condition_.notify_all();
            condition_.wait(lock, [this] { return !test_pause_startup_; });
        }
    }
#endif
    const auto fail_start = [&](MatmulStatus failure) {
        std::vector<std::shared_ptr<MatmulStripeJob>> jobs;
        MatmulExecution *                             attached_execution = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            status_         = failure;
            stop_requested_ = true;
            dense_done_     = true;
            pending_.clear();
            attached_execution = execution_;
            for (const auto & weak_job : jobs_) {
                if (auto job = weak_job.lock()) {
                    jobs.push_back(std::move(job));
                }
            }
            execution_           = nullptr;
            worker_started_      = false;
            startup_in_progress_ = false;
        }
        for (const auto & job : jobs) {
            job->cancel(failure);
            release_in_flight_once(job);
        }
        condition_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        if (attached_execution != nullptr) {
            std::lock_guard<std::mutex> execution_lock(*attached_execution->state_mutex_);
            attached_execution->facade_.discard_output_transaction();
            MatMul & facade = attached_execution->facade_;
            if (facade.args_ptr_ != &facade.owned_args_) {
                facade.owned_args_ = {};
            }
            attached_execution->status_            = failure;
            attached_execution->state_             = MatmulExecutionState::failed;
            attached_execution->pipeline_attached_ = false;
        }
        condition_.notify_all();
        return false;
    };
    try {
        {
            std::lock_guard<std::mutex> lock(*execution.state_mutex_);
            MatMul &                    facade = execution.facade_;
            if (facade.args_ptr_ != nullptr && facade.args_ptr_ != &facade.owned_args_) {
                facade.owned_args_ = ggml_gemmini_args_t(facade.args());
            }
        }
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (test_fail_thread_start_attempt_ != 0 &&
                ++test_thread_start_attempts_ == test_fail_thread_start_attempt_) {
                throw std::system_error(
                    std::make_error_code(std::errc::resource_unavailable_try_again),
                    "injected thread start failure");
            }
        }
#endif
        const auto submitted_context = gemmini_trace_capture();
        worker_                      = std::thread([this, submitted_context] {
            trace::ScopedContext submission_scope(submitted_context);
            worker_loop();
        });
    } catch (const std::bad_alloc &) {
        return fail_start(
            make_status(MatmulStatusCode::out_of_memory, "collector startup allocation failed"));
    } catch (const std::system_error &) {
        return fail_start(make_status(MatmulStatusCode::execution_failure,
                                      "collector worker thread creation failed"));
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        startup_in_progress_ = false;
    }
    condition_.notify_all();
    return true;
}

MatmulStatus MatmulStripeCollector::cancel() {
    std::vector<std::shared_ptr<MatmulStripeJob>> jobs;
    MatmulStatus cancelled = make_status(MatmulStatusCode::cancelled, "stripe pipeline cancelled");
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !startup_in_progress_; });
        if (!worker_started_) {
            return make_status(MatmulStatusCode::invalid_state, "stripe pipeline is not running");
        }
        if (status_.ok()) {
            status_ = cancelled;
        } else {
            cancelled = status_;
        }
        stop_requested_ = true;
        pending_.clear();
        for (const auto & weak_job : jobs_) {
            if (auto job = weak_job.lock()) {
                jobs.push_back(std::move(job));
            }
        }
    }
    for (const auto & job : jobs) {
        job->cancel(cancelled);
        release_in_flight_once(job);
    }
    condition_.notify_all();
    return cancelled;
}

void MatmulStripeCollector::fail(MatmulStatus failure) {
    std::vector<std::shared_ptr<MatmulStripeJob>> jobs;
    MatmulExecution *                             execution = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (status_.ok()) {
            status_ = failure;
        } else {
            failure = status_;
        }
        stop_requested_ = true;
        pending_.clear();
        for (const auto & weak_job : jobs_) {
            if (auto job = weak_job.lock()) {
                jobs.push_back(std::move(job));
            }
        }
        execution = execution_;
    }
    for (const auto & job : jobs) {
        job->cancel(failure);
        release_in_flight_once(job);
    }
    if (execution != nullptr) {
        std::lock_guard<std::mutex> execution_lock(*execution->state_mutex_);
        execution->status_ = failure;
        execution->state_  = MatmulExecutionState::failed;
    }
    condition_.notify_all();
}

void MatmulStripeCollector::release_in_flight_once(const std::shared_ptr<MatmulStripeJob> & job) {
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    const MatmulDenseState dense_state = job->snapshot().dense;
#endif
    MatmulJobMetrics context;
    std::string      layer;
    {
        std::lock_guard<std::mutex> job_lock(*job->job_mutex_);
        context.cpu_identity_mask = job->metrics_.cpu_identity_mask;
        context.run_id            = job->metrics_.run_id;
        context.stripe_id         = job->metrics_.stripe_id;
        context.slot              = job->metrics_.slot;
        layer                     = job->execution_->facade_.args().matmul_layer;
    }
    const auto release_start = read_matmul_cpu_sample();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (job->collector_slot_released_) {
            return;
        }
        job->collector_slot_released_ = true;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        if (test_dense_observer_collector == this) {
            observed_dense_state_at_release = dense_state;
        }
#endif
        --in_flight_;
    }
    const auto release_end = read_matmul_cpu_sample();
    record_matmul_cpu_wall(release_start, release_end);
    emit_matmul_cpu_interval(
        layer.c_str(), "collector_capacity_release", release_start, release_end, true, &context);
}

MatmulStatus MatmulStripeCollector::finish() {
    const auto              drain_start = read_matmul_cpu_sample();
    std::string             layer;
    std::optional<uint64_t> run_id;
    bool                    drained = false;
    const auto              result  = [&]() -> MatmulStatus {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return !startup_in_progress_; });
            if (!worker_started_) {
                return status_;
            }
            layer           = execution_->facade_.args().matmul_layer;
            run_id          = matmul_cpu_run_id(execution_->facade_.args());
            drained         = true;
            stop_requested_ = true;
        }
        condition_.notify_all();
        if (worker_.joinable()) {
            const auto join_start = read_matmul_cpu_sample();
            worker_.join();
            const auto join_end = read_matmul_cpu_sample();
            // This is the caller's wait interval, not the worker's execution cost.
            emit_matmul_cpu_interval(layer.empty() ? nullptr : layer.c_str(),
                                     "worker_join_wait",
                                     join_start,
                                     join_end,
                                     true,
                                     nullptr,
                                     nullptr,
                                     run_id);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            dense_done_ = true;
        }
        condition_.notify_all();
        MatmulExecution * execution = nullptr;
        MatmulStatus      status    = {};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            execution       = execution_;
            status          = status_;
            execution_      = nullptr;
            worker_started_ = false;
        }
        if (execution != nullptr) {
            std::lock_guard<std::mutex> execution_lock(*execution->state_mutex_);
            if (!status) {
                execution->facade_.discard_output_transaction();
                execution->status_ = status;
                execution->state_  = MatmulExecutionState::failed;
            }
            execution->pipeline_attached_ = false;
        }
        return status;
    }();
    const auto drain_end = read_matmul_cpu_sample();
    if (drained)
        emit_matmul_cpu_interval(layer.empty() ? nullptr : layer.c_str(),
                                 "pipeline_drain_and_join",
                                 drain_start,
                                 drain_end,
                                 result.ok(),
                                 nullptr,
                                 nullptr,
                                 run_id);
    return result;
}

// One NPU stream: dense WS and RMD run back to back in this worker, in that order.
void MatmulStripeCollector::worker_loop() {
#if LOG_CYCLE
    cycle::WorkerCpuTiming worker_cpu;
    cycle::WorkerCpuTiming::observe(&worker_cpu, true);
#endif
    try {
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (test_thread_exception_ == MatmulCollectorThread::worker) {
                const auto failure = test_thread_exception_failure_;
                test_thread_exception_.reset();
                if (failure == MatmulCollectorThreadFailure::out_of_memory) {
                    throw std::bad_alloc();
                }
                throw std::runtime_error("injected worker thread failure");
            }
        }
#endif
        for (;;) {
            CapturedStripe          captured{};
            MatmulExecution *       execution = nullptr;
            MatmulCpuSample         queue_wait_start, queue_wait_end;
            const char *            wait_layer = nullptr;
            std::optional<uint64_t> wait_run_id;
            bool                    queue_drained = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
#if LOG_CYCLE
                if (execution_ != nullptr) {
                    // The execution owns this string until finish() joins us.
                    wait_layer  = execution_->facade_.args().matmul_layer.c_str();
                    wait_run_id = matmul_cpu_run_id(execution_->facade_.args());
                }
#endif
                queue_wait_start = read_matmul_cpu_sample();
                condition_.wait(lock, [this] {
                    return stop_requested_ || (!pending_.empty() && in_flight_ < capacity_);
                });
                queue_wait_end = read_matmul_cpu_sample();
                queue_drained  = pending_.empty();
                if (!queue_drained) {
                    execution = execution_;
                    captured  = std::move(pending_.front());
                    pending_.pop_front();
                    captured.timing.dequeued_ns = now_ns();
#if LOG_CYCLE
                    captured.timing.dequeue_tid = cycle::host_thread_id();
                    if ((captured.cpu_identity_mask & GEMMINI_CYCLE_HAS_RUN_ID) != 0)
                        wait_run_id = captured.run_id;
#endif
                    ++in_flight_;
                    condition_.notify_all();
                }
            }
            // Publish after releasing the queue mutex; do not classify a wait
            // as CPU-work wall time or subtract across the producer/worker.
            emit_matmul_cpu_interval(wait_layer,
                                     "worker_queue_wait",
                                     queue_wait_start,
                                     queue_wait_end,
                                     true,
                                     nullptr,
                                     nullptr,
                                     wait_run_id);
            if (queue_drained)
                break;

            trace::ScopedContext stripe_task(captured.trace_origin, true);
            trace::CpuStage      stripe_lifetime(execution->facade_.args().matmul_layer.c_str(),
                                                 "task.host_work",
                                                 trace::CpuStage::Scope::envelope);
            std::shared_ptr<MatmulStripeJob> job;
            const auto                       preparation_start = read_matmul_cpu_sample();
            try {
                job = std::make_shared<MatmulStripeJob>(capture_stripe(
                    *execution,
                    MatmulStripeInput(captured.row_begin, captured.row_end, captured.stripe_id),
                    std::move(captured.direct_residual),
                    std::move(captured.rmd_packet)));
                if (job->status().ok() && captured.activation_metadata.has_value()) {
                    job->staged_activation_meta_ = std::make_unique<quants::act::Meta>();
                    auto & local =
                        job->staged_activation_meta_->storage().emplace<quants::act::exsia::Meta>();
                    if ((captured.cpu_identity_mask & GEMMINI_CYCLE_HAS_RUN_ID) != 0)
                        local.run_id = captured.run_id;
                    local.e_s   = captured.activation_metadata->e_s;
                    local.rho   = captured.activation_metadata->rho;
                    local.sigma = captured.activation_metadata->sigma;
                    local.theta = {captured.activation_metadata->theta};
                }
            } catch (const std::exception &) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    --in_flight_;
                }
                condition_.notify_all();
                throw;
            }
            const auto preparation_end = read_matmul_cpu_sample();
            record_matmul_cpu_wall(preparation_start, preparation_end);
            {
                std::lock_guard<std::mutex> job_lock(*job->job_mutex_);
                detail::apply_captured_stripe(captured, job->metrics_);
            }
            emit_matmul_cpu_interval(execution->facade_.args().matmul_layer.c_str(),
                                     "stripe_job_preparation",
                                     preparation_start,
                                     preparation_end,
                                     job->status().ok(),
                                     &job->metrics_);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                jobs_.push_back(job);
            }
            MatmulStatus status = job->status();
            if (!status) {
                release_in_flight_once(job);
                fail(status);
                break;
            }

            {
                std::lock_guard<std::mutex> job_lock(*job->job_mutex_);
                job->rmd_queued_ns_          = captured.timing.queued_ns;
                job->metrics_.rmd_enqueue_ns = captured.timing.queued_ns;
                if (job->execution_->options_.profiling) {
                    job->metrics_.ws_queue.nanoseconds =
                        captured.timing.dequeued_ns - captured.timing.queued_ns;
                    job->metrics_.ws_queue.count = 1;
                }
            }
#if defined(GGML_GEMMINI_TEST_OBSERVER)
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return !test_pause_dense_ || stop_requested_; });
            }
#endif
            MatmulStatus collector_status = this->status();
            if (!collector_status) {
                job->cancel(collector_status);
                release_in_flight_once(job);
                break;
            }

            status = execute_dense_stripe(*job);
#if defined(GGML_GEMMINI_TEST_OBSERVER)
            if (status && test_residual_failure_collector == this && !test_residual_failure) {
                status = test_residual_failure;
                job->record_failure(status, false);
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    test_residual_failure_observed_ = true;
                }
                condition_.notify_all();
            }
#endif
            if (status)
                status = execute_rmd_stripe(*job);
#if GGML_GEMMINI_ENABLE_RMD
            if (status)
                status = compose_rmd_stripe(*job);
#endif
            if (status)
                status = finalize_stripe(*job);
            if (status) {
                const MatmulJobMetrics      profile = job->metrics();
                std::lock_guard<std::mutex> lock(mutex_);
                profiles_.push_back(profile);
            }
            release_in_flight_once(job);
            condition_.notify_all();
            if (!status) {
                fail(status);
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            dense_done_ = true;
        }
        condition_.notify_all();
    } catch (const std::bad_alloc &) {
        fail(make_status(MatmulStatusCode::out_of_memory,
                         "collector worker thread allocation failed"));
    } catch (const std::exception &) {
        fail(make_status(MatmulStatusCode::execution_failure, "collector worker thread failed"));
    }
#if LOG_CYCLE
    cycle::WorkerCpuTiming::observe(&worker_cpu, false);
    const auto & args = execution_->facade_.args();
    worker_cpu.emit(
        args.matmul_layer.c_str(), "gemmini.matmul_worker", matmul_cpu_run_id(args), status().ok());
#endif
}

const quants::act::exsia::StripeReadySink * MatmulStripeCollector::sink() const {
    return &sink_;
}

MatmulStatus MatmulStripeCollector::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

MatmulCollectorSnapshot MatmulStripeCollector::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {status_, capacity_, pending_.size(), in_flight_, worker_started_};
}

std::vector<MatmulJobMetrics> MatmulStripeCollector::profiles() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return profiles_;
}

rmd::StripePacketHandle MatmulStripeCollector::captured_packet(size_t stripe) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stripes_.at(stripe).rmd_packet;
}

#if defined(GGML_GEMMINI_TEST_OBSERVER)
void MatmulStripeCollector::test_inject_residual_failure(MatmulStatus failure) {
    std::lock_guard<std::mutex> lock(mutex_);
    test_residual_failure_collector = this;
    test_residual_failure           = failure;
    test_dense_observer_collector   = this;
    observed_dense_state_at_release = MatmulDenseState::idle;
}

void MatmulStripeCollector::test_inject_thread_start_failure(size_t attempt) {
    std::lock_guard<std::mutex> lock(mutex_);
    test_fail_thread_start_attempt_ = attempt;
}

void MatmulStripeCollector::test_inject_thread_exception(MatmulCollectorThread        thread,
                                                         MatmulCollectorThreadFailure failure) {
    std::lock_guard<std::mutex> lock(mutex_);
    test_thread_exception_         = thread;
    test_thread_exception_failure_ = failure;
}

void MatmulStripeCollector::test_pause_dense_before_execute() {
    std::lock_guard<std::mutex> lock(mutex_);
    test_pause_dense_ = true;
}

void MatmulStripeCollector::test_pause_startup_after_attachment() {
    std::lock_guard<std::mutex> lock(mutex_);
    test_pause_startup_ = true;
}

void MatmulStripeCollector::test_resume_startup() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        test_pause_startup_ = false;
    }
    condition_.notify_all();
}

void MatmulStripeCollector::test_wait_for_residual_failure() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return test_residual_failure_observed_; });
}

size_t MatmulStripeCollector::test_in_flight() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return in_flight_;
}

MatmulDenseState MatmulStripeCollector::test_dense_state_at_release() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return test_dense_observer_collector == this ? observed_dense_state_at_release
                                                 : MatmulDenseState::idle;
}

#endif

bool MatmulStripeCollector::on_ready(void *                                       user_data,
                                     const quants::act::exsia::StripeReadyEvent & event) {
    if (event.collect_submission_timing)
        event.submission_wait_ns = 0;
    auto & collector = *static_cast<MatmulStripeCollector *>(user_data);
    if (event.row_begin >= event.row_end ||
        (event.activation_metadata.has_value() &&
         event.activation_metadata->theta == std::numeric_limits<int16_t>::min())) {
        {
            std::lock_guard<std::mutex> lock(collector.mutex_);
            collector.status_ =
                make_status(MatmulStatusCode::invalid_argument, "invalid stripe event");
            collector.stop_requested_ = true;
        }
        collector.condition_.notify_all();
        return false;
    }
    const auto make_captured = [&event, &collector](MatmulStageMetrics capture_copy,
                                                    MatmulStageMetrics producer_wait,
                                                    uint64_t           producer_wait_start_ns,
                                                    uint64_t           producer_wait_end_ns) {
        detail::MatmulCaptureTiming timing{};
        timing.capture_copy           = capture_copy;
        timing.producer_wait          = producer_wait;
        timing.producer_wait_start_ns = producer_wait_start_ns;
        timing.producer_wait_end_ns   = producer_wait_end_ns;
        const auto copy_start         = Clock::now();
        const auto capture_start      = read_matmul_cpu_sample();
        auto       captured           = detail::capture_collector_event(
            event,
            std::move(timing),
            collector.execution_ != nullptr
                ? matmul_cpu_run_id(collector.execution_->facade_.args())
                : std::nullopt);
        const auto capture_end = read_matmul_cpu_sample();
        record_matmul_cpu_wall(capture_start, capture_end);
        record_metric(captured.timing.capture_copy, true, copy_start);
        MatmulJobMetrics context;
        context.cpu_identity_mask = captured.cpu_identity_mask;
        context.run_id            = captured.run_id;
        context.stripe_id         = event.stripe_id;
        context.slot              = event.slot;
        emit_matmul_cpu_interval(collector.execution_ != nullptr
                                     ? collector.execution_->facade_.args().matmul_layer.c_str()
                                     : nullptr,
                                 "stripe_input_capture",
                                 capture_start,
                                 capture_end,
                                 true,
                                 &context);
        return captured;
    };
    try {
        MatmulStageMetrics           capture_copy;
        std::unique_lock<std::mutex> lock(collector.mutex_);
        if (!collector.status_ || collector.stop_requested_) {
            return false;
        }
        if (collector.worker_started_) {
            MatmulStageMetrics producer_wait;
            uint64_t           producer_wait_start_ns = 0;
            uint64_t           producer_wait_end_ns   = 0;
            if (collector.pending_.size() + collector.in_flight_ >= collector.capacity_) {
                const auto wait_start  = read_matmul_cpu_sample();
                producer_wait_start_ns = now_ns();
                collector.condition_.wait(lock, [&collector] {
                    return collector.stop_requested_ || !collector.status_ ||
                           collector.pending_.size() + collector.in_flight_ < collector.capacity_;
                });
                producer_wait_end_ns      = now_ns();
                const auto wait_end       = read_matmul_cpu_sample();
                producer_wait.nanoseconds = producer_wait_end_ns - producer_wait_start_ns;
                producer_wait.count       = 1;
                if (event.collect_submission_timing)
                    event.submission_wait_ns = producer_wait.nanoseconds;
#if LOG_CYCLE
                MatmulJobMetrics wait_context;
                wait_context.cpu_identity_mask = GEMMINI_CYCLE_HAS_STRIPE_ID;
                wait_context.stripe_id         = event.stripe_id;
                const auto run_id =
                    matmul_cpu_run_id(event,
                                      collector.execution_ != nullptr
                                          ? matmul_cpu_run_id(collector.execution_->facade_.args())
                                          : std::nullopt);
                if (run_id.has_value()) {
                    wait_context.cpu_identity_mask |=
                        GEMMINI_CYCLE_HAS_RUN_ID | GEMMINI_CYCLE_HAS_SLOT;
                    wait_context.run_id = *run_id;
                    wait_context.slot   = event.slot;
                }
                // Retain the caller's completed sample even when cancellation
                // prevents the pending stripe from being admitted afterward.
                emit_matmul_cpu_interval(
                    collector.execution_ != nullptr
                        ? collector.execution_->facade_.args().matmul_layer.c_str()
                        : nullptr,
                    "producer_capacity_wait",
                    wait_start,
                    wait_end,
                    collector.status_.ok() && !collector.stop_requested_,
                    &wait_context);
#else
                (void)wait_start;
                (void)wait_end;
#endif
            }
            if (collector.stop_requested_ || !collector.status_) {
                return false;
            }
            CapturedStripe captured = make_captured(
                capture_copy, producer_wait, producer_wait_start_ns, producer_wait_end_ns);
            const auto insert_start = Clock::now();
            collector.pending_.push_back(std::move(captured));
            record_metric(collector.pending_.back().timing.queue_insert, true, insert_start);
            collector.pending_.back().timing.queued_ns = now_ns();
#if LOG_CYCLE
            collector.pending_.back().timing.enqueue_tid           = cycle::host_thread_id();
            collector.pending_.back().timing.telemetry_queued_tick = cycle::read();
#endif
            lock.unlock();
            collector.condition_.notify_all();
            return true;
        }
        if (collector.stripes_.size() >= collector.capacity_) {
            collector.status_ =
                make_status(MatmulStatusCode::out_of_memory, "collector capacity exhausted");
            collector.stop_requested_ = true;
            lock.unlock();
            collector.condition_.notify_all();
            return false;
        }
        CapturedStripe captured     = make_captured(capture_copy, {}, 0, 0);
        const auto     insert_start = Clock::now();
        collector.stripes_.push_back(std::move(captured));
        record_metric(collector.stripes_.back().timing.queue_insert, true, insert_start);
        collector.stripes_.back().timing.queued_ns = now_ns();
#if LOG_CYCLE
        collector.stripes_.back().timing.enqueue_tid           = cycle::host_thread_id();
        collector.stripes_.back().timing.telemetry_queued_tick = cycle::read();
#endif
    } catch (const std::bad_alloc &) {
        {
            std::lock_guard<std::mutex> lock(collector.mutex_);
            collector.status_ =
                make_status(MatmulStatusCode::out_of_memory, "stripe capture allocation failed");
            collector.stop_requested_ = true;
        }
        collector.condition_.notify_all();
        return false;
    }
    return true;
}

MatmulStripeJob::MatmulStripeJob(MatmulExecution *                   execution,
                                 MatmulStripeInput                   input,
                                 MatmulStatus                        status,
                                 residual::DirectStripePayloadHandle direct_residual,
                                 rmd::StripePacketHandle             rmd_packet)
    : execution_(execution), input_(std::move(input)), status_(status),
      direct_residual_(std::move(direct_residual)), rmd_packet_(std::move(rmd_packet)),
      captured_(status.ok()) {
    metrics_.cpu_identity_mask = GEMMINI_CYCLE_HAS_STRIPE_ID;
    metrics_.stripe_id         = input_.stripe_id();
    metrics_.row_begin         = input_.row_begin();
    metrics_.row_end           = input_.row_end();
}

MatmulStripeJob::MatmulStripeJob()
    : execution_(nullptr), input_(0, 0), status_(invalid_state("job is not captured")),
      captured_(false) {}

MatmulStripeJob::MatmulStripeJob(MatmulStripeJob && other) noexcept
    : execution_(other.execution_), input_(std::move(other.input_)), status_(other.status_),
      metrics_(other.metrics_), staged_activation_meta_(std::move(other.staged_activation_meta_)),
      owns_slot_(other.owns_slot_), released_(other.released_),
      collector_slot_released_(other.collector_slot_released_),
      rmd_queued_ns_(other.rmd_queued_ns_), job_mutex_(std::move(other.job_mutex_)),
      direct_residual_(std::move(other.direct_residual_)),
      rmd_packet_(std::move(other.rmd_packet_)), rmd_correction_(std::move(other.rmd_correction_)),
      rmd_correction_ready_(other.rmd_correction_ready_), dense_state_(other.dense_state_),
      residual_state_(other.residual_state_), captured_(other.captured_),
      finalized_(other.finalized_) {
    other.execution_               = nullptr;
    other.owns_slot_               = false;
    other.released_                = true;
    other.collector_slot_released_ = true;
}

MatmulStripeJob & MatmulStripeJob::operator=(MatmulStripeJob && other) noexcept {
    if (this != &other) {
        release_slot();
        execution_                     = other.execution_;
        input_                         = std::move(other.input_);
        status_                        = other.status_;
        metrics_                       = other.metrics_;
        staged_activation_meta_        = std::move(other.staged_activation_meta_);
        direct_residual_               = std::move(other.direct_residual_);
        rmd_packet_                    = std::move(other.rmd_packet_);
        rmd_correction_                = std::move(other.rmd_correction_);
        rmd_correction_ready_          = other.rmd_correction_ready_;
        owns_slot_                     = other.owns_slot_;
        released_                      = other.released_;
        collector_slot_released_       = other.collector_slot_released_;
        rmd_queued_ns_                 = other.rmd_queued_ns_;
        dense_state_                   = other.dense_state_;
        residual_state_                = other.residual_state_;
        captured_                      = other.captured_;
        finalized_                     = other.finalized_;
        job_mutex_                     = std::move(other.job_mutex_);
        other.execution_               = nullptr;
        other.owns_slot_               = false;
        other.released_                = true;
        other.collector_slot_released_ = true;
    }
    return *this;
}

MatmulStripeJob::~MatmulStripeJob() {
    release_slot();
}

void MatmulStripeJob::release_slot() {
    if (!owns_slot_ || released_ || execution_ == nullptr || job_mutex_ == nullptr) {
        return;
    }
    MatmulExecution * execution = nullptr;
    {
        std::lock_guard<std::mutex> lock(*job_mutex_);
        if (!owns_slot_ || released_ || execution_ == nullptr) {
            return;
        }
        owns_slot_ = false;
        released_  = true;
        execution  = execution_;
    }
    std::lock_guard<std::mutex> state_lock(*execution->state_mutex_);
    if (execution->active_jobs_ != 0) {
        --execution->active_jobs_;
    }
}

void MatmulStripeJob::cancel(MatmulStatus status) {
    {
        std::lock_guard<std::mutex> lock(*job_mutex_);
        if (finalized_) {
            return;
        }
        if (status_.ok()) {
            status_ = status;
        }
        if (dense_state_ != MatmulDenseState::complete &&
            dense_state_ != MatmulDenseState::failed) {
            dense_state_ = MatmulDenseState::cancelled;
        }
        if (residual_state_ != MatmulResidualState::complete &&
            residual_state_ != MatmulResidualState::failed) {
            residual_state_ = MatmulResidualState::cancelled;
        }
    }
    lifecycle_condition_.notify_all();
}

void MatmulStripeJob::record_failure(MatmulStatus status, bool dense_branch) {
    {
        std::lock_guard<std::mutex> lock(*job_mutex_);
        if (status_.ok()) {
            status_ = status;
            if (dense_branch) {
                dense_state_ = MatmulDenseState::failed;
            } else {
                residual_state_ = MatmulResidualState::failed;
            }
        }
    }
    lifecycle_condition_.notify_all();
}

MatmulStatus MatmulStripeJob::status() const {
    std::lock_guard<std::mutex> lock(*job_mutex_);
    return status_;
}

MatmulJobMetrics MatmulStripeJob::metrics() const {
    std::lock_guard<std::mutex> lock(*job_mutex_);
    return metrics_;
}

MatmulStripeJobSnapshot MatmulStripeJob::snapshot() const {
    std::lock_guard<std::mutex> lock(*job_mutex_);
    return {status_, metrics_, dense_state_, residual_state_, captured_, finalized_, released_};
}

MatmulExecution prepare_execution(const ggml_gemmini_args_t & args, ResolvedMatmulOptions options) {
    return MatmulExecution(args, options);
}

MatmulExecution prepare_execution(ggml_gemmini_args_t * args, ResolvedMatmulOptions options) {
    return MatmulExecution(args, options);
}

MatmulStatus prepare_execution(ggml_gemmini_args_t &         args,
                               const ResolvedMatmulOptions & options,
                               MatmulExecution &             execution) {
    execution = MatmulExecution(&args, options);
    return execution.status();
}

MatmulStatus execute_full(MatmulExecution & execution) {
    if (execution.options_.mode != MatmulInvocationMode::full) {
        execution.status_ = make_status(MatmulStatusCode::unsupported_invocation,
                                        "full execution requires full mode");
        execution.state_  = MatmulExecutionState::failed;
        return execution.status_;
    }
    const MatMulResult result = execution.facade_.run_full();
    execution.status_ =
        to_public_status(result.status, result.capability, &execution.facade_.args());
    execution.state_ =
        execution.status_.ok() ? MatmulExecutionState::completed : MatmulExecutionState::failed;
    return execution.status_;
}

MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input) {
    if (input.residual_count() != 0 && input.residual() == nullptr) {
        return MatmulStripeJob(
            &execution,
            std::move(input),
            make_status(MatmulStatusCode::invalid_argument, "null stripe capture payload"));
    }
    if (input.residual_count() != 0) {
        const size_t rows =
            input.row_end() > input.row_begin() ? input.row_end() - input.row_begin() : 0;
        if (rows == 0 || execution.facade_.args().K == 0 ||
            rows > std::numeric_limits<size_t>::max() / execution.facade_.args().K ||
            input.residual_count() != rows * execution.facade_.args().K) {
            return MatmulStripeJob(&execution,
                                   std::move(input),
                                   make_status(MatmulStatusCode::invalid_argument,
                                               "invalid dense residual cardinality"));
        }
        return MatmulStripeJob(&execution,
                               std::move(input),
                               make_status(MatmulStatusCode::unsupported_route,
                                           "raw residual stripe payload is unsupported",
                                           MatMulCapability::unsupported));
    }
#if GGML_GEMMINI_ENABLE_RMD
    // No live packet was handed over (sequential stripe mode): rebuild one for exactly
    // this row range out of the packets the quantizer produced.
    const size_t                        row_begin    = input.row_begin();
    const size_t                        row_end      = input.row_end();
    const size_t                        stripe_id    = input.stripe_id();
    rmd::RmdStatus                      slice_status = rmd::RmdStatus::success;
    residual::DirectStripePayloadHandle direct;
    rmd::StripePacketHandle             packet;
    try {
        test_detail::observe_allocation_attempt();
        if (execution.options_.rmd_backend == RmdBackend::cpu_direct) {
            direct = residual::slice_direct_payloads(
                quants::activation_direct_residuals(execution.facade_.args()),
                row_begin,
                row_end,
                stripe_id,
                slice_status);
        } else {
            packet = rmd::slice_packets(quants::activation_rmd_packets(execution.facade_.args()),
                                        row_begin,
                                        row_end,
                                        stripe_id,
                                        slice_status);
        }
    } catch (const std::bad_alloc &) {
        return MatmulStripeJob(
            &execution,
            std::move(input),
            make_status(MatmulStatusCode::out_of_memory, "stripe capture allocation failed"));
    }
    if (slice_status != rmd::RmdStatus::success) {
        return MatmulStripeJob(&execution, std::move(input), from_rmd_status(slice_status));
    }
    return capture_stripe(execution, std::move(input), std::move(direct), std::move(packet));
#else
    return capture_stripe(execution, std::move(input), nullptr);
#endif
}

MatmulStripeJob capture_stripe(MatmulExecution &       execution,
                               MatmulStripeInput       input,
                               rmd::StripePacketHandle rmd_packet) {
    return capture_stripe(execution, std::move(input), nullptr, std::move(rmd_packet));
}

MatmulStripeJob capture_stripe(MatmulExecution &                   execution,
                               MatmulStripeInput                   input,
                               residual::DirectStripePayloadHandle direct_residual,
                               rmd::StripePacketHandle             rmd_packet) {
#if !GGML_GEMMINI_ENABLE_RMD
    direct_residual.reset();
    rmd_packet.reset();
#endif
    const auto                  start = Clock::now();
    MatmulStatus                status{};
    std::lock_guard<std::mutex> state_lock(*execution.state_mutex_);
    if (!execution.status_.ok()) {
        status = execution.status_;
    } else if (execution.options_.mode == MatmulInvocationMode::full) {
        status = make_status(MatmulStatusCode::unsupported_invocation,
                             "stripe capture requires stripe mode");
    } else if (execution.options_.mode == MatmulInvocationMode::stripe_pipeline &&
               execution.facade_.state() == MatMulState::idle) {
        const MatMulStatus begin_status = execution.facade_.begin_stripes();
        if (begin_status != MatMulStatus::success) {
            status = to_public_status(begin_status,
                                      begin_status == MatMulStatus::unsupported
                                          ? MatMulCapability::unsupported
                                          : MatMulCapability::supported);
        }
    } else if (execution.facade_.state() != MatMulState::accepting_stripes) {
        status = invalid_state("execution is not accepting stripes");
    } else if (input.row_begin() >= input.row_end() || input.row_end() > execution.total_rows_ ||
               input.stripe_id() >= execution.total_rows_ ||
               ((input.residual() == nullptr) != (input.residual_count() == 0))) {
        status = make_status(MatmulStatusCode::invalid_argument, "invalid stripe input or id");
    } else if (execution.captured_stripe_ids_.find(input.stripe_id()) !=
               execution.captured_stripe_ids_.end()) {
        status = invalid_contract("duplicate stripe id");
    } else if (execution.active_jobs_ >= execution.options_.job_capacity) {
        status = make_status(MatmulStatusCode::out_of_memory, "job capacity exhausted");
    } else if (execution.has_captures_ && input.row_begin() == execution.last_row_begin_ &&
               input.row_end() == execution.last_row_end_) {
        status = invalid_contract("duplicate stripe");
    } else if (execution.has_captures_ && input.row_begin() < execution.last_row_end_) {
        status = invalid_contract("overlapping stripe");
    } else if ((execution.options_.rmd_backend == RmdBackend::cpu_direct &&
                rmd_packet != nullptr) ||
               (execution.options_.rmd_backend == RmdBackend::gemmini_ws_compact &&
                direct_residual != nullptr) ||
               (direct_residual != nullptr && rmd_packet != nullptr)) {
        status = invalid_contract("residual payload does not match selected backend");
    } else if (direct_residual != nullptr &&
               (direct_residual->row_begin != input.row_begin() ||
                direct_residual->row_count != input.row_end() - input.row_begin())) {
        status = invalid_contract("direct residual does not cover the stripe rows");
    } else if (rmd_packet != nullptr &&
               (rmd_packet->row_begin != input.row_begin() ||
                rmd_packet->row_count != input.row_end() - input.row_begin())) {
        status = invalid_contract("rmd packet does not cover the stripe rows");
    }

    MatmulStripeJob job(
        &execution, std::move(input), status, std::move(direct_residual), std::move(rmd_packet));
    if (job.status_.ok()) {
        if (const auto run_id = matmul_cpu_run_id(execution.facade_.args())) {
            job.metrics_.cpu_identity_mask |= GEMMINI_CYCLE_HAS_RUN_ID;
            job.metrics_.run_id = *run_id;
        }
        if (!execution.has_captures_) {
            execution.first_row_ = job.input_.row_begin();
        }
        execution.last_row_begin_ = job.input_.row_begin();
        execution.last_row_end_   = job.input_.row_end();
        execution.captured_rows_ += job.input_.row_end() - job.input_.row_begin();
        execution.captured_stripe_ids_.insert(job.input_.stripe_id());
        execution.has_captures_ = true;
        ++execution.active_jobs_;
        job.owns_slot_ = true;
        if (execution.state_ == MatmulExecutionState::prepared) {
            execution.state_ = MatmulExecutionState::running;
        }
        record_metric(job.metrics_.handoff, execution.options_.profiling, start);
    }
    return job;
}

MatmulStatus capture_stripe(MatmulExecution &         execution,
                            const MatmulStripeInput & input,
                            MatmulStripeJob &         job) {
    MatmulStripeJob captured = capture_stripe(execution,
                                              MatmulStripeInput(input.row_begin(),
                                                                input.row_end(),
                                                                input.stripe_id(),
                                                                input.residual(),
                                                                input.residual_count()));
    job                      = std::move(captured);
    return job.status();
}

MatmulStatus execute_dense_stripe(MatmulStripeJob & job) {
    const quants::act::Meta * staged_activation_meta = nullptr;
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ ||
            job.dense_state_ != MatmulDenseState::idle) {
            return invalid_state("dense execution requires captured Dense idle state");
        }
        job.dense_state_       = MatmulDenseState::running;
        staged_activation_meta = job.staged_activation_meta_.get();
    }
    const auto         start       = Clock::now();
    const auto         dense_start = read_matmul_cpu_sample();
    const MatMulStatus status =
        staged_activation_meta != nullptr
            ? job.execution_->facade_.run_staged_stripe(
                  {job.input_.row_begin(), job.input_.row_end()},
                  job.input_.stripe_id(),
                  *staged_activation_meta)
            : job.execution_->facade_.run_stripe({job.input_.row_begin(), job.input_.row_end()},
                                                 job.input_.stripe_id());
    const auto dense_end = read_matmul_cpu_sample();
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        job.metrics_.ws_start_ns           = dense_start.ns;
        job.metrics_.ws_end_ns             = dense_end.ns;
        job.metrics_.ws_start_tid          = dense_start.tid;
        job.metrics_.ws_end_tid            = dense_end.tid;
        job.metrics_.telemetry_dense_start = dense_start.value;
        job.metrics_.telemetry_dense_end   = dense_end.value;
        job.metrics_.cpu_dense             = evaluate_matmul_cpu_interval(dense_start, dense_end);
    }
    emit_matmul_cpu_interval(job.execution_->facade_.args().matmul_layer.c_str(),
                             "dense_backend_host_call",
                             dense_start,
                             dense_end,
                             status == MatMulStatus::success,
                             &job.metrics_);
    const MatmulStatus dense_status =
        to_public_status(status,
                         status == MatMulStatus::unsupported ? MatMulCapability::unsupported
                                                             : MatMulCapability::supported,
                         &job.execution_->facade_.args());
    if (!dense_status) {
        job.record_failure(dense_status, true);
        return dense_status;
    }
    MatmulStatus result;
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        job.dense_state_ = MatmulDenseState::complete;
        record_metric(job.metrics_.ws, job.execution_->options_.profiling, start);
        job.metrics_.ws_service = job.metrics_.ws;
        result                  = dense_status;
    }
    job.lifecycle_condition_.notify_all();
    return result;
}

MatmulStatus accept_external_dense_completion(MatmulStripeJob & job) {
    if (job.job_mutex_ == nullptr) {
        return invalid_state("dense state unavailable");
    }
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ || !job.status_ ||
            job.dense_state_ != MatmulDenseState::idle) {
            return invalid_state("external dense completion requires captured Dense idle state");
        }
        MatMul &           facade = job.execution_->facade_;
        const MatMulStripe stripe{job.input_.row_begin(), job.input_.row_end()};
        MatMulStatus       accepted = MatMulStatus::success;
        if (facade.state_ != MatMulState::accepting_stripes) {
            accepted = MatMulStatus::invalid_state;
        } else if (stripe.row_begin >= stripe.row_end || stripe.row_end > facade.args().I) {
            accepted = MatMulStatus::malformed_stripe;
        } else if (facade.has_stripes_ && facade.last_row_begin_ == stripe.row_begin &&
                   facade.last_row_end_ == stripe.row_end) {
            accepted = MatMulStatus::duplicate_stripe;
        } else if (facade.has_stripes_ && stripe.row_begin < facade.last_row_end_) {
            accepted = MatMulStatus::overlapping_stripe;
        }
        if (accepted != MatMulStatus::success) {
            return to_public_status(accepted, MatMulCapability::supported, &facade.args());
        }
        if (!facade.has_stripes_) {
            facade.first_row_ = stripe.row_begin;
        }
        facade.last_row_begin_ = stripe.row_begin;
        facade.last_row_end_   = stripe.row_end;
        facade.covered_rows_ += stripe.row_end - stripe.row_begin;
        facade.has_stripes_    = true;
        job.dense_state_       = MatmulDenseState::complete;
        job.metrics_.cpu_dense = MatmulCpuInterval::unavailable("external_completion");
        emit_matmul_cpu_interval(facade.args().matmul_layer.c_str(),
                                 "dense_backend_host_call",
                                 {},
                                 {},
                                 true,
                                 &job.metrics_,
                                 &job.metrics_.cpu_dense);
    }
    job.lifecycle_condition_.notify_all();
    return {};
}

MatmulStatus execute_rmd_stripe(MatmulStripeJob & job) {
    if (job.job_mutex_ == nullptr) {
        return invalid_state("residual state unavailable");
    }
    residual::DirectStripePayloadHandle direct;
    rmd::StripePacketHandle             packet;
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ || !job.status_ ||
            job.residual_state_ != MatmulResidualState::idle) {
            return invalid_state("residual execution requires captured idle state");
        }
        direct                    = job.direct_residual_;
        packet                    = job.rmd_packet_;
        job.residual_state_       = MatmulResidualState::running;
        job.metrics_.stripe_id    = job.input_.stripe_id();
        job.metrics_.row_begin    = job.input_.row_begin();
        job.metrics_.row_end      = job.input_.row_end();
        job.metrics_.rmd_start_ns = now_ns();
        if (job.execution_->options_.profiling && job.rmd_queued_ns_ != 0) {
            job.metrics_.rmd_queue.nanoseconds = job.metrics_.rmd_start_ns - job.rmd_queued_ns_;
            job.metrics_.rmd_queue.count       = 1;
        }
    }
    if (direct == nullptr && packet == nullptr) {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (!job.status_)
            return job.status_;
        job.rmd_correction_                          = rmd::BlockScaledInt64Correction{};
        job.rmd_correction_ready_                    = true;
        job.metrics_.rmd.residual_observations_valid = true;
        job.metrics_.rmd.active_original_rows_valid  = true;
        job.metrics_.rmd.digit_bits                  = GGML_GEMMINI_ACTIVATION_BITS;
        job.metrics_.rmd.original_rows = job.input_.row_end() - job.input_.row_begin();
        job.metrics_.rmd.logical_k     = job.execution_->facade_.args().K;
        job.metrics_.rmd.logical_j     = job.execution_->facade_.args().J;
        job.metrics_.rmd.array_dim     = rmd::kArrayDim;
        job.metrics_.rmd.original_rows_after_pruning = job.metrics_.rmd.original_rows;
        job.metrics_.cpu_prep           = MatmulCpuInterval::unavailable("not_applicable");
        job.metrics_.cpu_backend        = MatmulCpuInterval::unavailable("not_applicable");
        job.metrics_.cpu_merge          = MatmulCpuInterval::unavailable("not_applicable");
        job.metrics_.cpu_residual_total = MatmulCpuInterval::unavailable("not_applicable");
        job.residual_state_             = MatmulResidualState::complete;
        job.lifecycle_condition_.notify_all();
        return {};
    }

    const auto               start          = Clock::now();
    const auto               residual_start = read_matmul_cpu_sample();
    rmd::Correction          correction     = rmd::BlockScaledInt64Correction{};
    rmd::RmdExecutionMetrics metrics{};
    metrics.timing_identity = {
        {job.execution_->facade_.args().matmul_layer.c_str(), nullptr, 0, 0, nullptr, 0, nullptr},
        job.metrics_.cpu_identity_mask | GEMMINI_CYCLE_HAS_STRIPE_ID,
        job.metrics_.run_id,
        job.input_.stripe_id(),
        job.metrics_.slot,
        0,
        0};
    rmd::detail::RmdWeightPreparation * weights = nullptr;
    if (packet != nullptr) {
        try {
            std::lock_guard<std::mutex> state_lock(*job.execution_->state_mutex_);
            if (job.execution_->rmd_weights_ == nullptr) {
                job.execution_->rmd_weights_ =
                    std::make_unique<rmd::detail::RmdWeightPreparation>();
            }
            weights = job.execution_->rmd_weights_.get();
        } catch (const std::bad_alloc &) {
            const MatmulStatus failure = from_rmd_status(rmd::RmdStatus::allocation_failure);
            job.record_failure(failure, false);
            return failure;
        }
    }
    residual::DirectExecutionMetrics direct_metrics{};
    if ((job.metrics_.cpu_identity_mask & GEMMINI_CYCLE_HAS_RUN_ID) != 0)
        direct_metrics.run_id = job.metrics_.run_id;
    test_detail::observe_residual_dispatch();
    test_detail::observe_backend_dispatch(direct != nullptr);
    const auto           backend_start = read_matmul_cpu_sample();
    const rmd::RmdStatus status =
        direct != nullptr
            ? residual::execute_direct_stripe(
                  job.execution_->facade_.args(), *direct, correction, &direct_metrics)
            : rmd::detail::execute_rmd_stripe_ws_with_weights(
                  job.execution_->facade_.args(), *packet, correction, *weights, &metrics);
    const auto backend_end = read_matmul_cpu_sample();
#if LOG_CYCLE
    if (direct != nullptr && status == rmd::RmdStatus::success) {
        const auto metrics_start = read_matmul_cpu_sample();
        rmd::collect_direct_metrics(*direct, metrics);
        const auto metrics_end = read_matmul_cpu_sample();
        emit_matmul_cpu_interval(job.execution_->facade_.args().matmul_layer.c_str(),
                                 "residual_workload_observation",
                                 metrics_start,
                                 metrics_end,
                                 true,
                                 &job.metrics_);
    }
#endif
    record_matmul_cpu_wall(residual_start, backend_start);
    if (direct != nullptr) {
        record_matmul_cpu_wall(backend_start, backend_end);
    } else {
#if LOG_CYCLE
        performance::incomplete_cpu_wall("accelerator_cpu_stage_coverage_incomplete");
#endif
    }
    metrics.direct_event_count = direct_metrics.event_count;
    metrics.direct_call_count  = direct_metrics.call_count;
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        job.metrics_.cpu_residual_start = residual_start;
        job.metrics_.cpu_prep    = evaluate_matmul_cpu_interval(residual_start, backend_start);
        job.metrics_.cpu_backend = evaluate_matmul_cpu_interval(backend_start, backend_end);
        job.metrics_.telemetry_residual_start = residual_start.value;
        job.metrics_.telemetry_backend_start  = backend_start.value;
        job.metrics_.telemetry_backend_end    = backend_end.value;
        job.metrics_.backend_start_ns         = backend_start.ns;
        job.metrics_.backend_end_ns           = backend_end.ns;
        job.metrics_.backend_start_tid        = backend_start.tid;
        job.metrics_.backend_end_tid          = backend_end.tid;
    }
    emit_matmul_cpu_interval(job.execution_->facade_.args().matmul_layer.c_str(),
                             "residual_backend_host_call",
                             backend_start,
                             backend_end,
                             status == rmd::RmdStatus::success,
                             &job.metrics_);
    if (status != rmd::RmdStatus::success) {
        emit_rmd_stripe_metrics(job.execution_->facade_.args().matmul_layer,
                                job.metrics_,
                                job.execution_->options_.rmd_backend,
                                false,
                                rmd::rmd_status_message(status),
                                nullptr);
        const MatmulStatus failure = from_rmd_status(status);
        job.record_failure(failure, false);
        return failure;
    }
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (!job.status_)
            return job.status_;
        job.rmd_correction_       = std::move(correction);
        job.rmd_correction_ready_ = true;
        job.metrics_.rmd          = metrics;
        record_metric(job.metrics_.rmd_execute, job.execution_->options_.profiling, start);
    }
    job.lifecycle_condition_.notify_all();
    return {};
}

// Completes the residual lifecycle after the executor has staged the correction.
MatmulStatus compose_rmd_stripe(MatmulStripeJob & job) {
    if (job.job_mutex_ == nullptr) {
        return invalid_state("residual state unavailable");
    }
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ || !job.status_ ||
            !job.rmd_correction_ready_ ||
            (job.residual_state_ != MatmulResidualState::running &&
             job.residual_state_ != MatmulResidualState::complete)) {
            return invalid_state("compose requires an executed residual stripe");
        }
        job.residual_state_     = MatmulResidualState::complete;
        job.metrics_.rmd_end_ns = now_ns();
    }
    job.lifecycle_condition_.notify_all();
    return {};
}

MatmulStatus finalize_stripe(MatmulStripeJob & job) {
    const auto       start = Clock::now();
    MatmulStatus     merge_failure{};
    MatmulCpuSample  merge_start, merge_end, stats_start, stats_end, hash_start, hash_end;
    bool             merged = false;
    MatmulJobMetrics completion_context;
    std::string      layer;
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_) {
            return invalid_state("stripe is not finalizable");
        }
        if (!job.status_) {
            return job.status_;
        }
        if (job.dense_state_ != MatmulDenseState::complete ||
            job.residual_state_ != MatmulResidualState::complete) {
            return invalid_state("finalize requires dense and residual completion");
        }
        completion_context.cpu_identity_mask = job.metrics_.cpu_identity_mask;
        completion_context.run_id            = job.metrics_.run_id;
        completion_context.stripe_id         = job.input_.stripe_id();
        completion_context.slot              = job.metrics_.slot;
        layer                                = job.execution_->facade_.args().matmul_layer;
        job.finalized_                       = true;
        const auto finalize_start            = read_matmul_cpu_sample();
        job.metrics_.finalize_start_ns       = finalize_start.ns;
        job.metrics_.finalize_start_tid      = finalize_start.tid;
        job.metrics_.stripe_id               = job.input_.stripe_id();
        job.metrics_.row_begin               = job.input_.row_begin();
        job.metrics_.row_end                 = job.input_.row_end();
        if (!rmd::correction_empty(job.rmd_correction_)) {
            merged                          = true;
            size_t correction_nonzero_count = 0;
            merge_start                     = read_matmul_cpu_sample();
            const rmd::RmdStatus status     = [&] {
                if (job.staged_activation_meta_ != nullptr) {
                    const auto & args = job.execution_->facade_.args();
                    return job.direct_residual_ != nullptr
                               ? rmd::detail::merge_rmd_correction_with_metadata(
                                     args,
                                     job.input_.row_begin(),
                                     job.input_.row_end(),
                                     job.rmd_correction_,
                                     *job.staged_activation_meta_,
                                     &correction_nonzero_count,
                                     &job.metrics_.rmd)
                               : rmd::detail::merge_rmd_correction_with_weights(
                                     args,
                                     args.f_out,
                                     *job.rmd_packet_,
                                     job.rmd_correction_,
                                     *job.execution_->rmd_weights_,
                                     &correction_nonzero_count,
                                     &job.metrics_.rmd,
                                     job.staged_activation_meta_.get());
                }
                return job.direct_residual_ != nullptr
                           ? rmd::merge_rmd_correction(job.execution_->facade_.args(),
                                                       job.input_.row_begin(),
                                                       job.input_.row_end(),
                                                       job.rmd_correction_,
                                                       &correction_nonzero_count,
                                                       &job.metrics_.rmd)
                           : rmd::detail::merge_rmd_correction_with_weights(
                                 job.execution_->facade_.args(),
                                 job.execution_->facade_.args().f_out,
                                 *job.rmd_packet_,
                                 job.rmd_correction_,
                                 *job.execution_->rmd_weights_,
                                 &correction_nonzero_count,
                                 &job.metrics_.rmd);
            }();
            merge_end                           = read_matmul_cpu_sample();
            job.metrics_.merge_start_ns         = merge_start.ns;
            job.metrics_.merge_end_ns           = merge_end.ns;
            job.metrics_.merge_start_tid        = merge_start.tid;
            job.metrics_.merge_end_tid          = merge_end.tid;
            job.metrics_.telemetry_merge_start  = merge_start.value;
            job.metrics_.telemetry_merge_end    = merge_end.value;
            job.metrics_.telemetry_residual_end = merge_end.value;
            job.metrics_.cpu_merge = evaluate_matmul_cpu_interval(merge_start, merge_end);
            job.metrics_.cpu_residual_total =
                evaluate_matmul_cpu_interval(job.metrics_.cpu_residual_start, merge_end);
            stats_start = read_matmul_cpu_sample();
            job.metrics_.telemetry_correction_nonzero_count =
                status == rmd::RmdStatus::success
                    ? correction_nonzero_count
                    : std::visit(
                          [](const auto & typed) {
                              return static_cast<uint64_t>(std::count_if(
                                  typed.values.begin(), typed.values.end(), [](const auto value) {
                                      return value != 0;
                                  }));
                          },
                          job.rmd_correction_);
            stats_end = read_matmul_cpu_sample();
#if CYCLE_DETAIL
            job.metrics_.telemetry_hash_enabled = matmul_telemetry_hash_enabled();
            if (job.metrics_.telemetry_hash_enabled) {
                hash_start                             = read_matmul_cpu_sample();
                job.metrics_.telemetry_input_hash      = job.direct_residual_ != nullptr
                                                             ? rmd_input_hash(*job.direct_residual_)
                                                             : rmd_input_hash(*job.rmd_packet_);
                job.metrics_.telemetry_correction_hash = rmd_correction_hash(job.rmd_correction_);
                job.metrics_.telemetry_output_hash     = rmd_output_hash(
                    job.execution_->facade_.args(), job.input_.row_begin(), job.input_.row_end());
                hash_end = read_matmul_cpu_sample();
            }
#endif
            if (status != rmd::RmdStatus::success) {
                merge_failure = from_rmd_status(status);
            }
        }
        if (!merged && job.metrics_.cpu_residual_start.collected) {
            const auto residual_end             = read_matmul_cpu_sample();
            job.metrics_.telemetry_residual_end = residual_end.value;
            job.metrics_.cpu_residual_total =
                evaluate_matmul_cpu_interval(job.metrics_.cpu_residual_start, residual_end);
            job.metrics_.cpu_merge = MatmulCpuInterval::unavailable("not_applicable");
        }
        record_metric(job.metrics_.rmd_finalize, job.execution_->options_.profiling, start);
        const auto finalize_end       = read_matmul_cpu_sample();
        job.metrics_.finalize_end_ns  = finalize_end.ns;
        job.metrics_.finalize_end_tid = finalize_end.tid;
        emit_matmul_cpu_interval(layer.c_str(),
                                 job.direct_residual_ != nullptr
                                     ? "rmd_cpu_direct_finalize_cycles"
                                     : (job.rmd_packet_ != nullptr ? "rmd_packet_finalize_cycles"
                                                                   : "dense_finalize_cycles"),
                                 finalize_start,
                                 finalize_end,
                                 merge_failure.ok(),
                                 &completion_context);
    }
    // Publish children after the legacy inclusive end, without summing nested work.
    if (merged) {
        record_matmul_cpu_wall(merge_start, merge_end);
        record_matmul_cpu_wall(stats_start, stats_end);
        emit_matmul_cpu_interval(layer.c_str(),
                                 "output_correction_apply",
                                 merge_start,
                                 merge_end,
                                 merge_failure.ok(),
                                 &completion_context);
        emit_matmul_cpu_interval(layer.c_str(),
                                 "telemetry_stats_compute",
                                 stats_start,
                                 stats_end,
                                 true,
                                 &completion_context);
        if (hash_start.collected) {
            record_matmul_cpu_wall(hash_start, hash_end);
            emit_matmul_cpu_interval(layer.c_str(),
                                     "telemetry_hash_compute",
                                     hash_start,
                                     hash_end,
                                     true,
                                     &completion_context);
        }
    }
    emit_rmd_stripe_metrics(layer,
                            job.metrics_,
                            job.execution_->options_.rmd_backend,
                            merge_failure.ok(),
                            merge_failure.ok() ? "none" : "correction_merge_failed",
                            &job.metrics_.rmd);
    if (!merge_failure.ok()) {
        job.record_failure(merge_failure, false);
        return merge_failure;
    }
    const auto completion_start = read_matmul_cpu_sample();
    {
        std::lock_guard<std::mutex> state_lock(*job.execution_->state_mutex_);
        job.execution_->finalized_rows_ += job.input_.row_end() - job.input_.row_begin();
    }
    job.release_slot();
    job.lifecycle_condition_.notify_all();
    const auto completion_end = read_matmul_cpu_sample();
    record_matmul_cpu_wall(completion_start, completion_end);
    emit_matmul_cpu_interval(layer.c_str(),
                             "stripe_completion_bookkeeping",
                             completion_start,
                             completion_end,
                             true,
                             &completion_context);
    return {};
}

MatmulStatus finish_execution(MatmulExecution & execution) {
    const MatmulCycleDrain drain;
    const auto             finish_start = read_matmul_cpu_sample();
    const auto             result       = [&]() -> MatmulStatus {
        std::lock_guard<std::mutex> state_lock(*execution.state_mutex_);
        if (!execution.pipeline_attached_ && execution.active_jobs_ == 0) {
            execution.rmd_weights_.reset();
        }
        if (!execution.status_.ok()) {
            execution.facade_.discard_output_transaction();
            execution.state_ = MatmulExecutionState::failed;
            return execution.status_;
        }
        if (execution.options_.mode == MatmulInvocationMode::full) {
            return execution.facade_.state() == MatMulState::completed ? execution.status_
                                                                       : invalid_state();
        }
        if (execution.pipeline_attached_) {
            return invalid_state("cannot finish while stripe collector is attached");
        }
        execution.state_ = MatmulExecutionState::finishing;
        if (execution.active_jobs_ != 0) {
            execution.state_ = MatmulExecutionState::running;
            return invalid_state("cannot finish with live jobs");
        }
        if (!execution.has_captures_ || execution.first_row_ != 0 ||
            execution.last_row_end_ != execution.total_rows_ ||
            execution.captured_rows_ != execution.total_rows_ ||
            execution.finalized_rows_ != execution.total_rows_) {
            execution.facade_.discard_output_transaction();
            execution.state_ = MatmulExecutionState::failed;
            return invalid_contract("missing stripes");
        }
        const MatMulStatus status = execution.facade_.finish_stripes();
        execution.status_         = to_public_status(status, MatMulCapability::supported);
        execution.state_ =
            execution.status_.ok() ? MatmulExecutionState::completed : MatmulExecutionState::failed;
        return execution.status_;
    }();
    const auto finish_end = read_matmul_cpu_sample();
    emit_matmul_cpu_interval(execution.facade_.args().matmul_layer.c_str(),
                             "matmul_execution_finish",
                             finish_start,
                             finish_end,
                             result.ok(),
                             nullptr,
                             nullptr,
                             matmul_cpu_run_id(execution.facade_.args()));
    return result;
}

static MatmulStatus matmul_impl(MatmulExecution execution, ResolvedMatmulOptions options) {
    if (!execution.status()) {
        return execution.status();
    }
    if (options.mode == MatmulInvocationMode::full) {
        return execute_full(execution);
    }
    return make_status(MatmulStatusCode::unsupported_invocation,
                       "pipeline mode requires externally staged stripes");
}

MatmulStatus matmul(ggml_gemmini_args_t & args, ResolvedMatmulOptions options) {
    return matmul_impl(prepare_execution(&args, options), options);
}

MatmulStatus matmul(const ggml_gemmini_args_t & args, ResolvedMatmulOptions options) {
    return matmul_impl(prepare_execution(args, options), options);
}

MatmulStatus execute_post_fold_pipeline(const ggml_gemmini_args_t & args,
                                        MatmulStripeCollector &     collector) {
    if (!collector.status_) {
        return collector.status_;
    }
    ResolvedMatmulOptions options{};
    options.mode              = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity      = 1;
    options.rmd_backend       = args.residual_route == residual::ResidualRoute::cpu_direct
                                    ? RmdBackend::cpu_direct
                                    : RmdBackend::gemmini_ws_compact;
    MatmulExecution execution = prepare_execution(args, options);
    if (!execution.status()) {
        return execution.status();
    }
    for (auto & captured : collector.stripes_) {
        if (const auto run_id = matmul_cpu_run_id(args)) {
            captured.run_id = *run_id;
            captured.cpu_identity_mask |= GEMMINI_CYCLE_HAS_RUN_ID | GEMMINI_CYCLE_HAS_SLOT;
        }
        MatmulStripeJob job = capture_stripe(
            execution,
            MatmulStripeInput(captured.row_begin, captured.row_end, captured.stripe_id),
            std::move(captured.direct_residual),
            std::move(captured.rmd_packet));
        detail::apply_captured_stripe(captured, job.metrics_);
        MatmulStatus status = job.status();
        if (status)
            status = execute_dense_stripe(job);
        if (status)
            status = execute_rmd_stripe(job);
#if GGML_GEMMINI_ENABLE_RMD
        if (status)
            status = compose_rmd_stripe(job);
#endif
        if (status)
            status = finalize_stripe(job);
        if (!status) {
            return status;
        }
    }
    return finish_execution(execution);
}

} // namespace ggml::gemmini
