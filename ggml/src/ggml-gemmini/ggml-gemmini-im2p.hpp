#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

struct ggml_gemmini_args_t;

namespace im2p::gemmini {
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
  std::uint64_t total_cycles = 0;
  std::uint64_t compute_cycles = 0;
  std::uint64_t overlap_cycles = 0;
  std::uint64_t completed_output_tiles = 0;
  std::uint64_t completed_stripes = 0;
  std::uint64_t stripes_published = 0;
  std::uint64_t first_publish_cycle = 0;
  std::uint64_t first_activation_read_cycle = 0;
};

struct Completion {
  Result result{};
  Stats stats{};
};

[[nodiscard]] Result translate(const ::im2p::gemmini::Status &status) noexcept;
[[nodiscard]] Completion
translate(const ::im2p::gemmini::FenceResult &result) noexcept;
[[nodiscard]] Completion run_full(const ggml_gemmini_args_t &args) noexcept;
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
void log_stats(const char *operation, const Stats &stats) noexcept;

#if defined(GGML_GEMMINI_TESTING)
constexpr std::size_t kTestStripeTraceCapacity = 8;

enum class TestFailure : std::uint8_t {
  none,
  malformed_contract,
  execute,
  quantization,
  progress,
  poll,
  fence,
  blocked_submit,
  rmd,
  collector_allocation,
  collector_capture,
};

struct TestCounters {
  std::uint64_t full = 0;
  std::uint64_t pipeline = 0;
  std::uint64_t fence = 0;
  std::uint64_t stripe = 0;
  std::uint64_t accepted_stripes = 0;
  std::uint64_t max_outstanding = 0;
  std::uint64_t rmd_calls = 0;
  std::uint64_t rmd_events = 0;
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
  std::size_t stripe_trace_size = 0;
  std::array<int, kTestStripeTraceCapacity> stripe_ids{};
  std::array<int, kTestStripeTraceCapacity> slot_ids{};
  std::array<std::size_t, kTestStripeTraceCapacity> collector_row_begin{};
  std::array<std::size_t, kTestStripeTraceCapacity> collector_row_end{};
  std::array<std::int16_t, kTestStripeTraceCapacity> collector_theta{};
};

void test_reset() noexcept;
void test_inject_failure(TestFailure failure) noexcept;
[[nodiscard]] bool test_wait_for_blocked_producer() noexcept;
void test_release_blocked_producer_with_error() noexcept;
[[nodiscard]] TestCounters test_counters() noexcept;
[[nodiscard]] bool test_production_failed() noexcept;
[[nodiscard]] bool test_should_fail_quantization() noexcept;
void test_record_production_failure(
    Error error = Error::execution_failure) noexcept;
void test_observe_stripe_dispatch() noexcept;
void test_observe_hardware_dispatch() noexcept;
#endif

} // namespace ggml::gemmini::im2p_adapter
