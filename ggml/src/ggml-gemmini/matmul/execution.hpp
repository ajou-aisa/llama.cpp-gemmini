#pragma once

#include "dense.hpp"
#include "gemmini.h"
#include "../../ggml-npu-common/matmul/options.hpp"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace ggml::gemmini {

class MatmulStripeCollector {
  public:
    explicit MatmulStripeCollector(size_t capacity);
    ~MatmulStripeCollector();
    bool                                        start(MatmulExecution & execution);
    MatmulStatus                                cancel();
    MatmulStatus                                finish();
    const quants::act::exsia::StripeReadySink * sink() const;
    MatmulStatus                                status() const;
    MatmulCollectorSnapshot                     snapshot() const;
    std::vector<MatmulJobMetrics>               profiles() const;
    rmd::StripePacketHandle                     captured_packet(size_t stripe) const;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    void test_inject_residual_failure(MatmulStatus failure);
    void test_inject_thread_start_failure(size_t attempt = 1);
    void test_inject_thread_exception(
        MatmulCollectorThread        thread,
        MatmulCollectorThreadFailure failure = MatmulCollectorThreadFailure::exception);
    void             test_pause_dense_before_execute();
    void             test_pause_startup_after_attachment();
    void             test_resume_startup();
    void             test_wait_for_residual_failure();
    size_t           test_in_flight() const;
    MatmulDenseState test_dense_state_at_release() const;
#endif

  private:
    using CapturedStripe = detail::MatmulCapturedStripe;
    static bool         on_ready(void *, const quants::act::exsia::StripeReadyEvent &);
    friend MatmulStatus execute_post_fold_pipeline(const ggml_gemmini_args_t &,
                                                   MatmulStripeCollector &);
    void                fail(MatmulStatus status);
    void                release_in_flight_once(const std::shared_ptr<MatmulStripeJob> & job);
    void                worker_loop();
    bool                worker_started_      = false;
    bool                startup_in_progress_ = false;
    bool                stop_requested_      = false;
    bool                dense_done_          = false;
    std::thread         worker_;
    // Borrowed for the active pipeline; finish the collector before destroying the execution.
    MatmulExecution *                           execution_ = nullptr;
    mutable std::mutex                          mutex_;
    std::condition_variable                     condition_;
    std::deque<CapturedStripe>                  pending_;
    std::vector<std::weak_ptr<MatmulStripeJob>> jobs_;
    size_t                                      capacity_;
    size_t                                      in_flight_ = 0;
    std::vector<CapturedStripe>                 stripes_;
    std::vector<MatmulJobMetrics>               profiles_;
    MatmulStatus                                status_;
    quants::act::exsia::StripeReadySink         sink_;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    bool                                 test_pause_dense_               = false;
    bool                                 test_pause_startup_             = false;
    bool                                 test_residual_failure_observed_ = false;
    size_t                               test_fail_thread_start_attempt_ = 0;
    size_t                               test_thread_start_attempts_     = 0;
    std::optional<MatmulCollectorThread> test_thread_exception_;
    MatmulCollectorThreadFailure         test_thread_exception_failure_ =
        MatmulCollectorThreadFailure::exception;
#endif
};

class MatmulStripeInput {
  public:
    MatmulStripeInput(size_t row_begin, size_t row_end);
    MatmulStripeInput(size_t          row_begin,
                      size_t          row_end,
                      size_t          stripe_id,
                      const int32_t * residual       = nullptr,
                      size_t          residual_count = 0);
    MatmulStripeInput(const MatmulStripeInput &)                 = default;
    MatmulStripeInput & operator=(const MatmulStripeInput &)     = default;
    MatmulStripeInput(MatmulStripeInput &&) noexcept             = default;
    MatmulStripeInput & operator=(MatmulStripeInput &&) noexcept = default;

    size_t          row_begin() const;
    size_t          row_end() const;
    size_t          stripe_id() const;
    const int32_t * residual() const;
    size_t          residual_count() const;

  private:
    size_t          row_begin_;
    size_t          row_end_;
    size_t          stripe_id_;
    const int32_t * residual_;
    size_t          residual_count_;
};

class MatmulExecution {
  public:
    MatmulExecution();
    explicit MatmulExecution(MatmulStatus status);
    MatmulExecution(const MatmulExecution &)             = delete;
    MatmulExecution & operator=(const MatmulExecution &) = delete;
    MatmulExecution(MatmulExecution &&) noexcept;
    MatmulExecution & operator=(MatmulExecution &&) noexcept;
    ~MatmulExecution();

    MatmulInvocationMode mode() const;
    MatmulExecutionState state() const;
    MatmulStatus         status() const;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    bool test_pipeline_attached() const;
#endif

  private:
    friend MatmulExecution prepare_execution(const ggml_gemmini_args_t &, ResolvedMatmulOptions);
    friend MatmulExecution prepare_execution(ggml_gemmini_args_t *, ResolvedMatmulOptions);
    friend MatmulStatus
    prepare_execution(ggml_gemmini_args_t &, const ResolvedMatmulOptions &, MatmulExecution &);
    friend MatmulStatus    execute_full(MatmulExecution &);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStripeJob
    capture_stripe(MatmulExecution &, MatmulStripeInput, rmd::StripePacketHandle);
    friend MatmulStripeJob capture_stripe(MatmulExecution &,
                                          MatmulStripeInput,
                                          residual::DirectStripePayloadHandle,
                                          rmd::StripePacketHandle);
    friend MatmulStatus
    capture_stripe(MatmulExecution &, const MatmulStripeInput &, MatmulStripeJob &);
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
    void initialize();
    void assert_pipeline_detached() const;

    size_t                      total_rows_;
    MatMul                      facade_;
    ResolvedMatmulOptions       options_;
    MatmulStatus                status_;
    MatmulExecutionState        state_       = MatmulExecutionState::empty;
    std::shared_ptr<std::mutex> state_mutex_ = std::make_shared<std::mutex>();
    std::unique_ptr<rmd::detail::RmdWeightPreparation> rmd_weights_;
    size_t                                             active_jobs_    = 0;
    size_t                                             captured_rows_  = 0;
    size_t                                             finalized_rows_ = 0;
    size_t                                             first_row_      = 0;
    size_t                                             last_row_begin_ = 0;
    size_t                                             last_row_end_   = 0;
    bool                                               has_captures_   = false;
    std::unordered_set<size_t>                         captured_stripe_ids_;
    bool                                               pipeline_attached_ = false;
};

class MatmulStripeJob {
  public:
    MatmulStripeJob();
    MatmulStripeJob(const MatmulStripeJob &)             = delete;
    MatmulStripeJob & operator=(const MatmulStripeJob &) = delete;
    MatmulStripeJob(MatmulStripeJob && other) noexcept;
    MatmulStripeJob & operator=(MatmulStripeJob && other) noexcept;
    ~MatmulStripeJob();

    MatmulStatus            status() const;
    MatmulJobMetrics        metrics() const;
    MatmulStripeJobSnapshot snapshot() const;

  private:
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStripeJob
    capture_stripe(MatmulExecution &, MatmulStripeInput, rmd::StripePacketHandle);
    friend MatmulStripeJob capture_stripe(MatmulExecution &,
                                          MatmulStripeInput,
                                          residual::DirectStripePayloadHandle,
                                          rmd::StripePacketHandle);
    friend MatmulStatus    execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus    accept_external_dense_completion(MatmulStripeJob &);
    friend MatmulStatus    execute_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus    compose_rmd_stripe(MatmulStripeJob &);
    friend MatmulStatus    finalize_stripe(MatmulStripeJob &);
    friend MatmulStatus    execute_post_fold_pipeline(const ggml_gemmini_args_t &,
                                                      MatmulStripeCollector &);
    friend class MatmulStripeCollector;

    MatmulStripeJob(MatmulExecution *                   execution,
                    MatmulStripeInput                   input,
                    MatmulStatus                        status,
                    residual::DirectStripePayloadHandle direct_residual = nullptr,
                    rmd::StripePacketHandle             rmd_packet      = nullptr);
    void cancel(MatmulStatus status);
    void release_slot();
    void record_failure(MatmulStatus status, bool dense_branch);

    MatmulExecution *                   execution_;
    MatmulStripeInput                   input_;
    MatmulStatus                        status_;
    MatmulJobMetrics                    metrics_;
    std::unique_ptr<quants::act::Meta>  staged_activation_meta_;
    bool                                owns_slot_               = false;
    bool                                released_                = false;
    bool                                collector_slot_released_ = false;
    uint64_t                            rmd_queued_ns_           = 0;
    std::shared_ptr<std::mutex>         job_mutex_               = std::make_shared<std::mutex>();
    std::condition_variable             lifecycle_condition_;
    residual::DirectStripePayloadHandle direct_residual_;
    rmd::StripePacketHandle             rmd_packet_;
    rmd::Correction                     rmd_correction_       = rmd::BlockScaledInt64Correction{};
    bool                                rmd_correction_ready_ = false;
    MatmulDenseState                    dense_state_          = MatmulDenseState::idle;
    MatmulResidualState                 residual_state_       = MatmulResidualState::idle;
    bool                                captured_             = true;
    bool                                finalized_            = false;
};

MatmulExecution prepare_execution(const ggml_gemmini_args_t & args, ResolvedMatmulOptions options);
MatmulExecution prepare_execution(ggml_gemmini_args_t * args, ResolvedMatmulOptions options);
MatmulStatus    prepare_execution(ggml_gemmini_args_t &         args,
                                  const ResolvedMatmulOptions & options,
                                  MatmulExecution &             execution);
MatmulStatus    execute_full(MatmulExecution & execution);
MatmulStatus
capture_stripe(MatmulExecution & execution, const MatmulStripeInput & input, MatmulStripeJob & job);
MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input);
MatmulStripeJob capture_stripe(MatmulExecution &       execution,
                               MatmulStripeInput       input,
                               rmd::StripePacketHandle rmd_packet);
MatmulStripeJob capture_stripe(MatmulExecution &                   execution,
                               MatmulStripeInput                   input,
                               residual::DirectStripePayloadHandle direct_residual,
                               rmd::StripePacketHandle             rmd_packet);
MatmulStatus
capture_stripe(MatmulExecution & execution, const MatmulStripeInput & input, MatmulStripeJob & job);
MatmulStatus execute_dense_stripe(MatmulStripeJob & job);
MatmulStatus accept_external_dense_completion(MatmulStripeJob & job);
MatmulStatus execute_rmd_stripe(MatmulStripeJob & job);
MatmulStatus compose_rmd_stripe(MatmulStripeJob & job);
MatmulStatus finalize_stripe(MatmulStripeJob & job);
MatmulStatus finish_execution(MatmulExecution & execution);
MatmulStatus matmul(ggml_gemmini_args_t & args, ResolvedMatmulOptions options);
MatmulStatus matmul(const ggml_gemmini_args_t & args, ResolvedMatmulOptions options);
MatmulStatus execute_post_fold_pipeline(const ggml_gemmini_args_t & args,
                                        MatmulStripeCollector &     collector);

inline MatmulStatus resolution_status(MatmulOptionsError error) {
    switch (error) {
    case MatmulOptionsError::disabled_mode:
        return {MatmulStatusCode::unsupported_invocation,
                "requested matmul mode is disabled in this build",
                MatMulCapability::unsupported};
    case MatmulOptionsError::none:
        return {};
    case MatmulOptionsError::invalid_mode:
    case MatmulOptionsError::invalid_job_capacity:
    case MatmulOptionsError::invalid_rmd_backend:
    case MatmulOptionsError::runtime_override_disabled:
        return {MatmulStatusCode::invalid_argument, "invalid matmul options"};
    }
    return {MatmulStatusCode::invalid_argument, "invalid matmul options"};
}

inline MatmulExecution prepare_execution(const ggml_gemmini_args_t & args,
                                         MatmulOptionOverrides       options = {}) {
    const auto resolution = resolve_matmul_options(options);
    return resolution.ok() ? prepare_execution(args, resolution.options)
                           : MatmulExecution(resolution_status(resolution.error));
}

inline MatmulExecution prepare_execution(ggml_gemmini_args_t * args,
                                         MatmulOptionOverrides options = {}) {
    const auto resolution = resolve_matmul_options(options);
    return resolution.ok() ? prepare_execution(args, resolution.options)
                           : MatmulExecution(resolution_status(resolution.error));
}

inline MatmulStatus prepare_execution(ggml_gemmini_args_t &         args,
                                      const MatmulOptionOverrides & options,
                                      MatmulExecution &             execution) {
    const auto resolution = resolve_matmul_options(options);
    if (!resolution.ok()) {
        return resolution_status(resolution.error);
    }
    return prepare_execution(args, resolution.options, execution);
}

inline MatmulStatus matmul(ggml_gemmini_args_t & args, MatmulOptionOverrides options = {}) {
    const auto resolution = resolve_matmul_options(options);
    return resolution.ok() ? matmul(args, resolution.options) : resolution_status(resolution.error);
}

inline MatmulStatus matmul(const ggml_gemmini_args_t & args, MatmulOptionOverrides options = {}) {
    const auto resolution = resolve_matmul_options(options);
    return resolution.ok() ? matmul(args, resolution.options) : resolution_status(resolution.error);
}

} // namespace ggml::gemmini

namespace ggml::gemmini {
void require_command(const char *what, gemmini_status_t status);
void tiled_matmul(const gemmini_transport_t &transport, size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type);
void tiled_matmul_auto(const gemmini_transport_t &transport, size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type);
void gemmini_set_tile_ws(struct ggml_gemmini_args_t *args);
} // namespace ggml::gemmini
