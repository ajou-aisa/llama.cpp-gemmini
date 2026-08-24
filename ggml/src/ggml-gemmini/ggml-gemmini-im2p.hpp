#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

struct ggml_gemmini_args_t;

namespace im2p::gemmini {
enum class Mode : std::uint8_t;
struct Status;
struct FenceResult;
} // namespace im2p::gemmini

namespace ggml::gemmini::im2p_adapter {

enum class Error : std::uint8_t {
  success,
  invalid_argument,
  invalid_contract,
  unsupported_route,
  invalid_state,
  backpressure,
  out_of_memory,
  execution_failure,
};

struct Result {
  Error error = Error::success;
  const char *message = "success";
  bool native_contract = false;

  [[nodiscard]] bool ok() const noexcept { return error == Error::success; }
};

struct Stats {
  // Complete raw 64-bit RTL statistics in provider ABI order. Wait and
  // overlap counters are independent observations, not additive totals.
  std::uint64_t rtl_work_total_cycles = 0;
  std::uint64_t rtl_activation_read_requests = 0;
  std::uint64_t rtl_weight_read_requests = 0;
  std::uint64_t rtl_scale_read_requests = 0;
  std::uint64_t rtl_output_write_requests = 0;
  std::uint64_t rtl_output_write_responses = 0;
  std::uint64_t rtl_activation_wait_cycles = 0;
  std::uint64_t rtl_weight_wait_cycles = 0;
  std::uint64_t rtl_scale_wait_cycles = 0;
  std::uint64_t rtl_output_wait_cycles = 0;
  std::uint64_t rtl_stripe_host_wait_cycles = 0;
  std::uint64_t rtl_drain_cycles = 0;
  std::uint64_t rtl_weight_preload_cycles = 0;
  std::uint64_t rtl_same_block_scale_hits = 0;
  std::uint64_t rtl_next_scale_hits = 0;
  std::uint64_t rtl_scale_demand_misses = 0;
  std::uint64_t rtl_compute_cycles = 0;
  std::uint64_t rtl_overlap_cycles = 0;
  std::uint64_t rtl_activation_overlap_cycles = 0;
  std::uint64_t rtl_weight_overlap_cycles = 0;
  std::uint64_t rtl_scale_overlap_cycles = 0;
  std::uint64_t rtl_completed_fragments = 0;
  std::uint64_t rtl_completed_output_works = 0;
  std::uint64_t rtl_scheduler_groups_completed = 0;
  std::uint64_t rtl_stripes_published = 0;
  std::uint64_t rtl_stripe_rows_published = 0;
  std::uint64_t rtl_weight_bank_activations = 0;
  std::uint64_t rtl_cross_stripe_overlap_cycles = 0;
  std::uint64_t rtl_lookahead_prepared = 0;
  std::uint64_t rtl_first_publish_cycle = 0;
  std::uint64_t rtl_first_activation_read_cycle = 0;
  std::uint64_t rtl_first_weight_read_cycle = 0;
  std::uint64_t rtl_weight_preload_cycle = 0;
  std::uint64_t rtl_lookahead_weight_requests = 0;
  std::uint64_t rtl_lookahead_weight_reuse_hits = 0;
  std::uint64_t rtl_first_scale_read_cycle = 0;
  std::uint64_t rtl_lookahead_scale_requests = 0;
  std::uint64_t rtl_lookahead_scale_reuses = 0;
  std::uint64_t rtl_current_scheduler_group_completion_cycle = 0;
  std::uint64_t rtl_lookahead_ready_cycle = 0;
  std::uint64_t rtl_lookahead_start_cycle = 0;
};

struct Completion {
  Result result{};
  Stats stats{};
  std::uint64_t run_id = 0;
};

enum class PublicMode : std::uint8_t {
  full,
  stripe_pipeline,
};

enum class WeightFamily : std::uint8_t {
  h0,
  h1,
  hp1,
  h2,
  hp2,
  unsupported,
};

enum class ResidualBackend : std::uint8_t {
  cpu_direct,
  compact_ws,
};

enum class BuildIdentity : std::uint8_t {
  im2p_sim_ws,
  hardware_ws,
  hardware_cpu,
  hardware_os,
  unsupported,
};

struct ExsiaRouteRequest {
  bool exsia = true;
  std::uint8_t activation_bits = 0;
  std::uint8_t weight_bits = 0;
  std::uint8_t artifact_activation_bits = 0;
  std::uint8_t artifact_weight_bits = 0;
  bool rmd_enabled = false;
  PublicMode mode = PublicMode::full;
  WeightFamily family = WeightFamily::unsupported;
  ResidualBackend residual_backend = ResidualBackend::cpu_direct;
  BuildIdentity build_identity = BuildIdentity::unsupported;
};

[[nodiscard]] Result translate(const ::im2p::gemmini::Status &status) noexcept;
[[nodiscard]] Completion translate(
    const ::im2p::gemmini::FenceResult &result,
    ::im2p::gemmini::Mode mode,
    std::uint64_t expected_publications,
    std::uint64_t expected_published_rows) noexcept;
[[nodiscard]] Completion run_full(const ggml_gemmini_args_t &args) noexcept;
[[nodiscard]] Completion
run_stripe_pipeline(const ggml_gemmini_args_t &args) noexcept;
[[nodiscard]] Result gate_route(const ExsiaRouteRequest &request) noexcept;
[[nodiscard]] Result gate_route(bool exsia, std::uint8_t activation_bits,
                                bool rmd_enabled, bool cpu_direct_rmd,
                                std::uint8_t weight_bits = 8) noexcept;

struct ExsiaFullExecutionStart;

class ExsiaFullExecution {
public:
  ExsiaFullExecution(ExsiaFullExecution &&) noexcept;
  ExsiaFullExecution &operator=(ExsiaFullExecution &&) noexcept;
  ~ExsiaFullExecution();

  ExsiaFullExecution(const ExsiaFullExecution &) = delete;
  ExsiaFullExecution &operator=(const ExsiaFullExecution &) = delete;

  [[nodiscard]] Result install_sink() noexcept;
  [[nodiscard]] Completion finish(bool quantization_succeeded) noexcept;

private:
  class Impl;
  explicit ExsiaFullExecution(std::unique_ptr<Impl>) noexcept;
  std::unique_ptr<Impl> impl_;

  friend ExsiaFullExecutionStart
  start_exsia_full_execution(ggml_gemmini_args_t &) noexcept;
};

struct ExsiaFullExecutionStart {
  Result result{};
  std::unique_ptr<ExsiaFullExecution> execution;
};

[[nodiscard]] ExsiaFullExecutionStart
start_exsia_full_execution(ggml_gemmini_args_t &args) noexcept;

struct ExsiaStripePipelineStart;

class ExsiaStripePipeline {
public:
  ExsiaStripePipeline(ExsiaStripePipeline &&) noexcept;
  ExsiaStripePipeline &operator=(ExsiaStripePipeline &&) noexcept;
  ~ExsiaStripePipeline();

  ExsiaStripePipeline(const ExsiaStripePipeline &) = delete;
  ExsiaStripePipeline &operator=(const ExsiaStripePipeline &) = delete;

  [[nodiscard]] Result install_sink() noexcept;
  [[nodiscard]] Completion finish(bool quantization_succeeded) noexcept;

private:
  class Impl;
  explicit ExsiaStripePipeline(std::unique_ptr<Impl>) noexcept;
  std::unique_ptr<Impl> impl_;

  friend ExsiaStripePipelineStart
  start_exsia_stripe_pipeline(ggml_gemmini_args_t &) noexcept;
};

struct ExsiaStripePipelineStart {
  Result result{};
  std::unique_ptr<ExsiaStripePipeline> pipeline;
};

[[nodiscard]] ExsiaStripePipelineStart
start_exsia_stripe_pipeline(ggml_gemmini_args_t &args) noexcept;

void log_failure(const char *operation, const Result &result) noexcept;
void log_stats(const char * mode, const Stats & stats,
               std::uint64_t run_id,
               const ggml_gemmini_args_t & args) noexcept;

#if defined(GGML_GEMMINI_TESTING)
constexpr std::size_t kTestStripeTraceCapacity = 32;

enum class TestFailure : std::uint8_t {
  none,
  malformed_contract,
  execute,
  quantization,
  provider,
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
                                         const char *layer,
                                         void *user_data);

struct TestCounters {
  std::uint64_t activation_allocations = 0;
  std::uint64_t worker_starts = 0;
  std::uint64_t full = 0;
  std::uint64_t pipeline = 0;
  std::uint64_t fence = 0;
  std::uint64_t stripe = 0;
  std::uint64_t accepted_stripes = 0;
  std::uint64_t max_outstanding = 0;
  std::uint64_t rmd_calls = 0;
  std::uint64_t rmd_events = 0;
  std::uint64_t rmd_packets = 0;
  std::uint64_t dense_completions = 0;
  std::uint64_t residual_executions = 0;
  std::uint64_t compositions = 0;
  std::uint64_t authorize = 0;
  std::uint64_t commit = 0;
  std::uint64_t commit_event = 0;
  std::uint64_t collector_events = 0;
  std::uint64_t collector_handles = 0;
  std::uint64_t hardware = 0;
  std::uint64_t fallback = 0;
  std::uint64_t live_runs = 0;
  std::uint64_t blocked_producers = 0;
  std::uint64_t quantization_failures = 0;
  std::uint64_t progress_failures = 0;
  std::uint64_t poll_failures = 0;
  std::uint64_t first_publish_cycle = 0;
  std::uint64_t first_activation_read_cycle = 0;
  std::uint64_t order_event_sequence = 0;
  std::uint64_t rmd_terminal_event = 0;
  std::uint64_t authorize_success_event = 0;
  bool blocked_submit_saw_execution_failure = false;
  bool fence_saw_execution_failure = false;
  Error production_error = Error::success;
  WeightFamily observed_weight_family = WeightFamily::unsupported;
  std::uint64_t weight_family_observations = 0;
  std::size_t stripe_trace_size = 0;
  std::array<int, kTestStripeTraceCapacity> stripe_ids{};
  std::array<int, kTestStripeTraceCapacity> slot_ids{};
  std::array<std::size_t, kTestStripeTraceCapacity> stripe_row_begin{};
  std::array<std::size_t, kTestStripeTraceCapacity> stripe_row_end{};
  std::array<std::size_t, kTestStripeTraceCapacity> collector_row_begin{};
  std::array<std::size_t, kTestStripeTraceCapacity> collector_row_end{};
  std::array<std::int16_t, kTestStripeTraceCapacity> collector_theta{};
};

void test_reset() noexcept;
void test_set_runtime_args_observer(TestRuntimeArgsObserver observer,
                                    void *user_data) noexcept;
void test_inject_failure(TestFailure failure) noexcept;
[[nodiscard]] bool test_wait_for_blocked_producer() noexcept;
void test_release_blocked_producer_with_error() noexcept;
[[nodiscard]] TestCounters test_counters() noexcept;
[[nodiscard]] bool test_production_failed() noexcept;
[[nodiscard]] bool test_should_fail_quantization() noexcept;
void test_record_production_failure(
    Error error = Error::execution_failure) noexcept;
void test_observe_activation_allocation() noexcept;
void test_observe_weight_family(WeightFamily family) noexcept;
void test_observe_stripe_dispatch() noexcept;
void test_observe_hardware_dispatch() noexcept;
#endif

} // namespace ggml::gemmini::im2p_adapter
