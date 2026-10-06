#pragma once

#include "im2p/route.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::im2p_adapter {

[[nodiscard]] Completion run_full(const ggml_gemmini_args_t & args) noexcept;
[[nodiscard]] Completion run_stripe_pipeline(const ggml_gemmini_args_t & args) noexcept;
struct ExsiaFullExecutionStart;

class ExsiaFullExecution {
  public:
    ExsiaFullExecution(ExsiaFullExecution &&) noexcept;
    ExsiaFullExecution & operator=(ExsiaFullExecution &&) noexcept;
    ~ExsiaFullExecution();

    ExsiaFullExecution(const ExsiaFullExecution &)             = delete;
    ExsiaFullExecution & operator=(const ExsiaFullExecution &) = delete;

    [[nodiscard]] Result     install_sink() noexcept;
    [[nodiscard]] Completion finish(bool quantization_succeeded) noexcept;

  private:
    class Impl;
    explicit ExsiaFullExecution(std::unique_ptr<Impl>) noexcept;
    std::unique_ptr<Impl> impl_;

    friend ExsiaFullExecutionStart start_exsia_full_execution(ggml_gemmini_args_t &) noexcept;
};

struct ExsiaFullExecutionStart {
    Result                              result{};
    std::unique_ptr<ExsiaFullExecution> execution;
};

[[nodiscard]] ExsiaFullExecutionStart
start_exsia_full_execution(ggml_gemmini_args_t & args) noexcept;

struct ExsiaStripePipelineStart;

class ExsiaStripePipeline {
  public:
    ExsiaStripePipeline(ExsiaStripePipeline &&) noexcept;
    ExsiaStripePipeline & operator=(ExsiaStripePipeline &&) noexcept;
    ~ExsiaStripePipeline();

    ExsiaStripePipeline(const ExsiaStripePipeline &)             = delete;
    ExsiaStripePipeline & operator=(const ExsiaStripePipeline &) = delete;

    [[nodiscard]] Result     install_sink() noexcept;
    [[nodiscard]] Completion finish(bool quantization_succeeded) noexcept;

  private:
    class Impl;
    explicit ExsiaStripePipeline(std::unique_ptr<Impl>) noexcept;
    std::unique_ptr<Impl> impl_;

    friend ExsiaStripePipelineStart start_exsia_stripe_pipeline(ggml_gemmini_args_t &) noexcept;
};

struct ExsiaStripePipelineStart {
    Result                               result{};
    std::unique_ptr<ExsiaStripePipeline> pipeline;
};

[[nodiscard]] ExsiaStripePipelineStart
start_exsia_stripe_pipeline(ggml_gemmini_args_t & args) noexcept;

// Validates and emits successful PIPELINE semantic/RMD rows. Dense raw timing
// and residual-simulator timing remain separate, non-additive clock domains.
[[nodiscard]] Result emit_residual_stripe_timings(const ::im2p::gemmini::FenceResult & result,
                                                  const ggml_gemmini_args_t &          args,
                                                  std::uint64_t expected_run_id) noexcept;

void install_rtl_debug_sink() noexcept;
void log_failure(const char * operation, const Result & result) noexcept;
void log_stats(const char *                mode,
               const Stats &               stats,
               std::uint64_t               run_id,
               const ggml_gemmini_args_t & args) noexcept;
void log_rmd_stats(const Completion & completion, const ggml_gemmini_args_t & args) noexcept;

#if defined(GGML_GEMMINI_TESTING)
constexpr std::size_t kTestStripeTraceCapacity = 32;

enum class TestFailure : std::uint8_t {
    none,
    malformed_contract,
    execute,
    quantization,
    provider,
    provider_read,
    provider_watchdog,
    provider_k_overflow,
    provider_block_overflow,
    provider_cancel_between_dots,
    progress,
    poll,
    fence,
    malformed_completion,
    incomplete_publication,
    blocked_submit,
    rmd,
    dense,
    residual_execute,
    compose,
    output_authorization,
    output_copy,
    simulator_create,
    collector_allocation,
    collector_capture,
};

enum class TestRuntimeArgsSite : std::uint8_t {
    simple_full_before_execute,
    simple_pipeline_before_execute,
    exsia_full_before_execute,
    exsia_pipeline_before_execute,
};

using TestRuntimeArgsObserver = void (*)(TestRuntimeArgsSite site,
                                         const char *        layer,
                                         void *              user_data);

struct TestCounters {
    std::uint64_t                             activation_allocations               = 0;
    std::uint64_t                             worker_starts                        = 0;
    std::uint64_t                             full                                 = 0;
    std::uint64_t                             pipeline                             = 0;
    std::uint64_t                             fence                                = 0;
    std::uint64_t                             stripe                               = 0;
    std::uint64_t                             accepted_stripes                     = 0;
    std::uint64_t                             max_outstanding                      = 0;
    std::uint64_t                             rmd_calls                            = 0;
    std::uint64_t                             rmd_events                           = 0;
    std::uint64_t                             rmd_packets                          = 0;
    std::uint64_t                             rmd_dot_calls                        = 0;
    std::uint64_t                             provider_dot_attempts                = 0;
    std::uint64_t                             dense_completions                    = 0;
    std::uint64_t                             dense_completions_at_first_residual  = 0;
    std::uint64_t                             residual_executions                  = 0;
    std::uint64_t                             compositions                         = 0;
    std::uint64_t                             authorize                            = 0;
    std::uint64_t                             commit                               = 0;
    std::uint64_t                             commit_event                         = 0;
    std::uint64_t                             collector_events                     = 0;
    std::uint64_t                             collector_handles                    = 0;
    std::uint64_t                             hardware                             = 0;
    std::uint64_t                             fallback                             = 0;
    std::uint64_t                             live_runs                            = 0;
    std::uint64_t                             residual_simulator_creates           = 0;
    std::uint64_t                             live_residual_simulators             = 0;
    std::uint64_t                             blocked_producers                    = 0;
    std::uint64_t                             quantization_failures                = 0;
    std::uint64_t                             progress_failures                    = 0;
    std::uint64_t                             poll_failures                        = 0;
    std::uint64_t                             first_publish_cycle                  = 0;
    std::uint64_t                             first_activation_read_cycle          = 0;
    std::uint64_t                             order_event_sequence                 = 0;
    std::uint64_t                             rmd_terminal_event                   = 0;
    std::uint64_t                             authorize_success_event              = 0;
    bool                                      blocked_submit_saw_execution_failure = false;
    bool                                      fence_saw_execution_failure          = false;
    Error                                     production_error                     = Error::success;
    WeightFamily                              observed_weight_family = WeightFamily::unsupported;
    std::uint64_t                             weight_family_observations = 0;
    std::size_t                               stripe_trace_size          = 0;
    std::array<int, kTestStripeTraceCapacity> stripe_ids{};
    std::array<int, kTestStripeTraceCapacity> slot_ids{};
    std::array<std::size_t, kTestStripeTraceCapacity>  stripe_row_begin{};
    std::array<std::size_t, kTestStripeTraceCapacity>  stripe_row_end{};
    std::array<std::size_t, kTestStripeTraceCapacity>  collector_row_begin{};
    std::array<std::size_t, kTestStripeTraceCapacity>  collector_row_end{};
    std::array<std::int16_t, kTestStripeTraceCapacity> collector_theta{};
    std::array<std::uint8_t, kTestStripeTraceCapacity> pipeline_callback_stripes{};
    std::array<std::size_t, kTestStripeTraceCapacity>  pipeline_merge_row_begin{};
    std::array<std::size_t, kTestStripeTraceCapacity>  pipeline_merge_row_end{};
};

void test_reset() noexcept;
void test_set_runtime_args_observer(TestRuntimeArgsObserver observer, void * user_data) noexcept;
void test_inject_failure(TestFailure failure) noexcept;
[[nodiscard]] bool         test_wait_for_blocked_producer() noexcept;
void                       test_release_blocked_producer_with_error() noexcept;
[[nodiscard]] TestCounters test_counters() noexcept;
[[nodiscard]] bool         test_production_failed() noexcept;
[[nodiscard]] bool         test_should_fail_quantization() noexcept;
void test_record_production_failure(Error error = Error::execution_failure) noexcept;
void test_observe_activation_allocation() noexcept;
void test_observe_weight_family(WeightFamily family) noexcept;
void test_observe_stripe_dispatch() noexcept;
void test_observe_hardware_dispatch() noexcept;
#endif

} // namespace ggml::gemmini::im2p_adapter
