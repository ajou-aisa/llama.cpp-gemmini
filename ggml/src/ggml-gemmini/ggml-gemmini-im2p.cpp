#include "ggml-gemmini-im2p.hpp"

#include "ggml-gemmini-args.h"
#include "ggml-gemmini-matmul.hpp"
#include "ggml-impl.h"
#include "im2p_gemmini_frontend.hpp"
#include "quants/act/exsia/exsia.hpp"
#include "quants/act/exsia/types.hpp"

#if defined(GGML_GEMMINI_TESTING)
#include "im2p_gemmini_frontend_testing.hpp"
#endif

#include <algorithm>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#if defined(GGML_GEMMINI_TESTING)
#include <chrono>
#include <condition_variable>
#endif

namespace ggml::gemmini::im2p_adapter {
namespace {

Result from_matmul_status(const MatmulStatus &status) noexcept {
  if (status.ok()) {
    return {};
  }
  Error error = Error::execution_failure;
  switch (status.code) {
  case MatmulStatusCode::invalid_argument:
    error = Error::invalid_argument;
    break;
  case MatmulStatusCode::unsupported_invocation:
    error = Error::unsupported_route;
    break;
  case MatmulStatusCode::out_of_memory:
    error = Error::out_of_memory;
    break;
  case MatmulStatusCode::invalid_state:
    error = Error::invalid_state;
    break;
  default:
    break;
  }
  return {error, status.message, false};
}

bool checked_output_extent(const ggml_gemmini_args_t &args,
                           size_t &extent) noexcept {
  if (args.I == 0 || args.J == 0 || args.f_out == nullptr) {
    return false;
  }
  const size_t row_stride = args.stride_f_out == 0 ? args.J : args.stride_f_out;
  const size_t col_stride =
      args.col_stride_f_out == 0 ? 1 : args.col_stride_f_out;
  if (args.I - 1 > std::numeric_limits<size_t>::max() / row_stride ||
      args.J - 1 > std::numeric_limits<size_t>::max() / col_stride) {
    return false;
  }
  const size_t row_offset = (args.I - 1) * row_stride;
  const size_t col_offset = (args.J - 1) * col_stride;
  if (col_offset == std::numeric_limits<size_t>::max() ||
      row_offset > std::numeric_limits<size_t>::max() - col_offset - 1) {
    return false;
  }
  extent = row_offset + col_offset + 1;
  return true;
}

#if defined(GGML_GEMMINI_TESTING)
std::mutex test_mutex;
std::condition_variable test_changed;
TestCounters counters;
TestFailure injected_failure = TestFailure::none;
bool production_failed = false;
::im2p::gemmini::Run *active_blocked_run = nullptr;
#endif

} // namespace

Result translate(const ::im2p::gemmini::Status &status) noexcept {
  using Source = ::im2p::gemmini::StatusCode;

  Error error = Error::execution_failure;
  switch (status.code) {
  case Source::success:
    error = Error::success;
    break;
  case Source::invalid_argument:
    error = Error::invalid_argument;
    break;
  case Source::invalid_contract:
    error = Error::invalid_contract;
    break;
  case Source::unsupported_route:
    error = Error::unsupported_route;
    break;
  case Source::invalid_state:
    error = Error::invalid_state;
    break;
  case Source::backpressure:
    error = Error::backpressure;
    break;
  case Source::out_of_memory:
    error = Error::out_of_memory;
    break;
  case Source::execution_failure:
    error = Error::execution_failure;
    break;
  }
  return {error, status.message, status.native_contract};
}

Completion translate(const ::im2p::gemmini::FenceResult &result) noexcept {
  const auto &stats = result.stats.base;
  return {
      translate(result.status),
      {
          stats.work_total_cycles,
          stats.compute_cycles,
          stats.overlap_cycles,
          stats.completed_output_tiles,
          stats.completed_stripes,
          stats.stripes_published,
          result.stats.lookahead_publish_cycle,
          result.stats.lookahead_first_activation_cycle,
      },
  };
}

Result gate_route(bool exsia, std::uint8_t activation_bits, bool rmd_enabled,
                  bool cpu_direct_rmd, std::uint8_t weight_bits) noexcept {
  // Production ExSIA is A8/Q8 only. Matched A4/Q4 and A16/Q16 remain TODO;
  // mixed precision is unsupported. H0/H1/HP1 retain the existing Q8
  // direction, while H2/HP2 gain no new route here. This gate runs before
  // activation allocation, simulator/worker start, fence, RMD, commit, or
  // fallback.
  if (weight_bits != 8) {
    return {Error::unsupported_route, "IM2P only supports Q8 weight routes",
            false};
  }
  if (activation_bits != 4 && activation_bits != 8 && activation_bits != 16) {
    return {Error::unsupported_route, "unsupported IM2P activation width",
            false};
  }
  if (exsia) {
    if (activation_bits == 8 && rmd_enabled && cpu_direct_rmd) {
      return {};
    }
    return {Error::unsupported_route,
            "ExSIA IM2P requires the production A8/Q8 cpu_direct RMD route",
            false};
  }
  if (!rmd_enabled) {
    return {};
  }
  return {Error::unsupported_route,
          "non-ExSIA IM2P execution does not support RMD", false};
}

Completion run_full(const ggml_gemmini_args_t &args) noexcept {
  // The ggml orchestration always materializes an all-zero repeating bias.
  // IM2P's provider contract represents that identity bias by absence.
  ggml_gemmini_args_t runtime_args = args;
  runtime_args.D = nullptr;
  runtime_args.repeating_bias = false;
#if defined(GGML_GEMMINI_TESTING)
  TestFailure failure = TestFailure::none;
  {
    std::lock_guard lock(test_mutex);
    ++counters.full;
    failure = injected_failure;
    if (failure == TestFailure::execute) {
      production_failed = true;
      return {
          {Error::execution_failure, "injected IM2P execute failure", false},
          {}};
    }
  }
  if (failure == TestFailure::malformed_contract) {
    runtime_args.A = {};
  }
  std::vector<float> injected_output;
  if (failure == TestFailure::fence) {
    size_t output_extent = 0;
    if (!checked_output_extent(args, output_extent)) {
      std::lock_guard lock(test_mutex);
      production_failed = true;
      return {{Error::invalid_contract,
               "invalid output layout for fence injection", false},
              {}};
    }
    try {
      injected_output.assign(output_extent, 0.0f);
    } catch (...) {
      std::lock_guard lock(test_mutex);
      production_failed = true;
      return {{Error::out_of_memory, "failed to stage fence injection output",
               false},
              {}};
    }
    runtime_args.f_out = injected_output.data();
  }
#endif

  auto started =
      ::im2p::gemmini::execute(&runtime_args, ::im2p::gemmini::Mode::full,
                               ::im2p::gemmini::Options{65536});
  if (!started.status.ok()) {
#if defined(GGML_GEMMINI_TESTING)
    std::lock_guard lock(test_mutex);
    production_failed = true;
#endif
    return {translate(started.status), {}};
  }
  if (!started.run) {
#if defined(GGML_GEMMINI_TESTING)
    std::lock_guard lock(test_mutex);
    production_failed = true;
#endif
    return {{Error::invalid_state, "IM2P execute returned no run", false}, {}};
  }
#if defined(GGML_GEMMINI_TESTING)
  {
    std::lock_guard lock(test_mutex);
    ++counters.fence;
  }
#endif
  Completion completion = translate(::im2p::gemmini::fence(*started.run));
#if defined(GGML_GEMMINI_TESTING)
  if (failure == TestFailure::fence) {
    std::lock_guard lock(test_mutex);
    production_failed = true;
    return {{Error::execution_failure, "injected IM2P fence failure", false},
            completion.stats};
  }
  if (!completion.result.ok()) {
    std::lock_guard lock(test_mutex);
    production_failed = true;
  }
#endif
  return completion;
}

struct CapturedExsiaStripe {
  quants::act::exsia::StripeReadyEvent event;
  std::int16_t theta = 0;
};

static Result
apply_captured_rmd(const ggml_gemmini_args_t &runtime_args, float *output_data,
                   size_t output_elements,
                   const std::vector<CapturedExsiaStripe> &captured) noexcept {
  std::unique_ptr<ggml_gemmini_args_t> rmd_args;
  std::vector<const CapturedExsiaStripe *> ordered;
  try {
    rmd_args = std::make_unique<ggml_gemmini_args_t>(runtime_args);
    rmd_args->f_out = output_data;
    size_t output_extent = 0;
    if (!checked_output_extent(*rmd_args, output_extent) ||
        output_extent > output_elements) {
      return {Error::invalid_contract,
              "frontend RMD staging does not cover the output layout", false};
    }
    ordered.reserve(captured.size());
    for (const auto &stripe : captured) {
      ordered.push_back(&stripe);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto *lhs, const auto *rhs) {
                return lhs->event.row_begin < rhs->event.row_begin;
              });
    auto &captured_meta =
        rmd_args->act_quant.storage().emplace<quants::act::exsia::Meta>();
    captured_meta.theta.assign(captured.size(),
                               std::numeric_limits<std::int16_t>::min());
    size_t next_row = 0;
    for (size_t index = 0; index < ordered.size(); ++index) {
      const auto &stripe = *ordered[index];
      if (stripe.event.stripe_id != index ||
          stripe.event.row_begin != next_row ||
          stripe.event.row_begin >= stripe.event.row_end ||
          stripe.event.row_end > runtime_args.I) {
        return {Error::invalid_contract,
                "captured ExSIA stripes are not contiguous in row order",
                false};
      }
      captured_meta.theta[index] = stripe.theta;
      next_row = stripe.event.row_end;
    }
    if (ordered.empty() || next_row != runtime_args.I) {
      return {Error::invalid_contract,
              "captured ExSIA stripes do not cover the output rows", false};
    }
  } catch (const std::bad_alloc &) {
    return {Error::out_of_memory, "failed to stage captured ExSIA metadata",
            false};
  } catch (...) {
    return {Error::execution_failure, "failed to stage captured ExSIA metadata",
            false};
  }

  ResolvedMatmulOptions options;
  options.mode = MatmulInvocationMode::stripe_pipeline;
  options.stripe_rows = runtime_args.activation_rows_per_stripe;
  options.stripe_rows_auto = false;
  options.job_capacity = std::max<size_t>(2, ordered.size());
  options.rmd_backend = RmdBackend::cpu_direct;
  MatmulExecution execution = prepare_execution(rmd_args.get(), options);
  if (!execution.status().ok()) {
    return from_matmul_status(execution.status());
  }

  std::vector<MatmulStripeJob> jobs;
  try {
    jobs.reserve(ordered.size());
    for (const auto *stripe : ordered) {
      jobs.push_back(capture_stripe(
          execution,
          MatmulStripeInput(stripe->event.row_begin, stripe->event.row_end,
                            stripe->event.stripe_id),
          stripe->event.direct_residual, stripe->event.rmd_packet));
      if (!jobs.back().status().ok()) {
        return from_matmul_status(jobs.back().status());
      }
    }
  } catch (const std::bad_alloc &) {
    return {Error::out_of_memory, "failed to stage ExSIA RMD jobs", false};
  } catch (...) {
    return {Error::execution_failure, "failed to stage ExSIA RMD jobs", false};
  }

  for (size_t index = 0; index < jobs.size(); ++index) {
    MatmulStatus status = accept_external_dense_completion(jobs[index]);
    if (status.ok())
      status = execute_rmd_stripe(jobs[index]);
    if (status.ok())
      status = compose_rmd_stripe(jobs[index]);
    if (!status.ok())
      return from_matmul_status(status);
#if defined(GGML_GEMMINI_TESTING)
    {
      std::lock_guard lock(test_mutex);
      ++counters.rmd_calls;
      const auto &direct = ordered[index]->event.direct_residual;
      if (direct)
        counters.rmd_events += direct->events.size();
    }
#endif
  }
#if defined(GGML_GEMMINI_TESTING)
  {
    std::lock_guard lock(test_mutex);
    if (injected_failure == TestFailure::rmd) {
      return {Error::execution_failure, "injected RMD failure", false};
    }
  }
#endif
  for (auto &job : jobs) {
    const MatmulStatus status = finalize_stripe(job);
    if (!status.ok())
      return from_matmul_status(status);
  }
  return from_matmul_status(finish_execution(execution));
}

class ExsiaStripePipeline::Impl {
public:
  using PublishedStripe = CapturedExsiaStripe;

  explicit Impl(ggml_gemmini_args_t &source)
      : args(source), runtime_args(source), sink{this, &Impl::on_ready} {
    runtime_args.D = nullptr;
    runtime_args.repeating_bias = false;
  }

  ~Impl() {
    args.exsia_stripe_ready_sink = nullptr;
    if (run && !fenced) {
      (void)::im2p::gemmini::fence(*run);
      fenced = true;
    }
#if defined(GGML_GEMMINI_TESTING)
    unregister_run();
#endif
  }

  static bool
  on_ready(void *opaque,
           const quants::act::exsia::StripeReadyEvent &event) noexcept {
    return static_cast<Impl *>(opaque)->publish(event);
  }

  bool publish(const quants::act::exsia::StripeReadyEvent &event) noexcept {
#if defined(GGML_GEMMINI_TESTING)
    {
      std::lock_guard lock(test_mutex);
      ++counters.stripe;
      if (counters.stripe_trace_size < kTestStripeTraceCapacity) {
        const size_t index = counters.stripe_trace_size++;
        counters.stripe_ids[index] = static_cast<int>(event.stripe_id);
        counters.slot_ids[index] = static_cast<int>(event.slot);
      }
    }
#endif
    if (!sink_result.ok() || !run) {
      return false;
    }
    const auto *metadata =
        std::get_if<quants::act::exsia::Meta>(&args.act_quant.storage());
    const std::int16_t theta =
        metadata == nullptr
            ? std::numeric_limits<std::int16_t>::min()
            : metadata->resolve_stripe_theta(static_cast<int>(event.stripe_id));
    if (theta == std::numeric_limits<std::int16_t>::min()) {
      sink_result = {Error::invalid_contract,
                     "published ExSIA stripe has no committed theta", false};
      return false;
    }
    try {
      published.push_back({event, theta});
    } catch (const std::bad_alloc &) {
      sink_result = {Error::out_of_memory,
                     "failed to retain published ExSIA stripe", false};
      return false;
    } catch (...) {
      sink_result = {Error::execution_failure,
                     "failed to retain published ExSIA stripe", false};
      return false;
    }
    const auto status =
        ::im2p::gemmini::submit_stripe(*run, event, {true, theta});
    if (!status.ok()) {
      published.pop_back();
      sink_result = translate(status);
#if defined(GGML_GEMMINI_TESTING)
      std::lock_guard lock(test_mutex);
      if (injected_failure == TestFailure::blocked_submit &&
          status.code == ::im2p::gemmini::StatusCode::execution_failure) {
        counters.blocked_submit_saw_execution_failure = true;
      }
#endif
      return false;
    }
#if defined(GGML_GEMMINI_TESTING)
    const auto snapshot = ::im2p::gemmini::RunTestAccess::inspect(*run);
    {
      std::lock_guard lock(test_mutex);
      ++counters.accepted_stripes;
      counters.max_outstanding = std::max<std::uint64_t>(
          counters.max_outstanding, snapshot.outstanding);
    }
#endif
    return true;
  }

  Result authorize(bool rmd_succeeded) noexcept {
    if (rmd_succeeded && !rmd_terminal_succeeded) {
      return {Error::invalid_state,
              "output authorization requires terminal RMD success", false};
    }
#if defined(GGML_GEMMINI_TESTING)
    {
      std::lock_guard lock(test_mutex);
      ++counters.authorize;
      if (rmd_succeeded) {
        counters.authorize_success_event = ++counters.order_event_sequence;
      }
    }
#endif
    return translate(
        ::im2p::gemmini::authorize_output_commit(*run, rmd_succeeded));
  }

  void mark_rmd_terminal_success() noexcept {
    rmd_terminal_succeeded = true;
#if defined(GGML_GEMMINI_TESTING)
    std::lock_guard lock(test_mutex);
    counters.rmd_terminal_event = ++counters.order_event_sequence;
#endif
  }

  void copy_staged_output() noexcept {
    float *destination = args.f_out;
    const size_t row_stride =
        args.stride_f_out == 0 ? args.J : args.stride_f_out;
    const size_t col_stride =
        args.col_stride_f_out == 0 ? 1 : args.col_stride_f_out;
    for (size_t row = 0; row < args.I; ++row) {
      for (size_t col = 0; col < args.J; ++col) {
        const size_t offset = row * row_stride + col * col_stride;
        destination[offset] = staged_output[offset];
      }
    }
#if defined(GGML_GEMMINI_TESTING)
    std::lock_guard lock(test_mutex);
    ++counters.commit;
    counters.commit_event = ++counters.order_event_sequence;
#endif
  }

#if defined(GGML_GEMMINI_TESTING)
  void register_run() noexcept {
    std::lock_guard lock(test_mutex);
    registered = true;
    ++counters.live_runs;
    if (injected_failure == TestFailure::blocked_submit) {
      active_blocked_run = run.get();
      test_changed.notify_all();
    }
  }

  void unregister_run() noexcept {
    std::lock_guard lock(test_mutex);
    if (!registered) {
      return;
    }
    if (active_blocked_run == run.get()) {
      active_blocked_run = nullptr;
    }
    registered = false;
    --counters.live_runs;
    test_changed.notify_all();
  }
#endif

  ggml_gemmini_args_t &args;
  ggml_gemmini_args_t runtime_args;
  std::vector<float> staged_output;
  std::vector<PublishedStripe> published;
  std::unique_ptr<::im2p::gemmini::Run> run;
  quants::act::exsia::StripeReadySink sink;
  Result sink_result{};
  bool sink_installed = false;
  bool fenced = false;
  bool finished = false;
  bool rmd_terminal_succeeded = false;
#if defined(GGML_GEMMINI_TESTING)
  bool registered = false;
#endif
};

class ExsiaFullExecution::Impl {
public:
  explicit Impl(ggml_gemmini_args_t &source)
      : args(source), runtime_args(source), sink{this, &Impl::on_ready} {}

  ~Impl() {
    if (args.exsia_stripe_ready_sink == &sink) {
      args.exsia_stripe_ready_sink = nullptr;
    }
    if (run && !fenced) {
      (void)::im2p::gemmini::fence(*run);
    }
  }

  static bool
  on_ready(void *opaque,
           const quants::act::exsia::StripeReadyEvent &event) noexcept {
    return static_cast<Impl *>(opaque)->collect(event);
  }

  bool collect(const quants::act::exsia::StripeReadyEvent &event) noexcept {
    if (!collector_result.ok() || finished || !sink_installed)
      return false;
#if defined(GGML_GEMMINI_TESTING)
    {
      std::lock_guard lock(test_mutex);
      if (injected_failure == TestFailure::collector_capture) {
        collector_result = {Error::invalid_contract,
                            "injected ExSIA FULL collector capture failure",
                            false};
        return false;
      }
    }
#endif
    const auto *metadata =
        std::get_if<quants::act::exsia::Meta>(&args.act_quant.storage());
    const std::int16_t theta =
        metadata == nullptr
            ? std::numeric_limits<std::int16_t>::min()
            : metadata->resolve_stripe_theta(static_cast<int>(event.stripe_id));
    if (theta == std::numeric_limits<std::int16_t>::min()) {
      collector_result = {Error::invalid_contract,
                          "collected ExSIA stripe has no committed theta",
                          false};
      return false;
    }
    try {
      captured.push_back({event, theta});
    } catch (const std::bad_alloc &) {
      collector_result = {Error::out_of_memory,
                          "failed to retain collected ExSIA stripe", false};
      return false;
    } catch (...) {
      collector_result = {Error::execution_failure,
                          "failed to retain collected ExSIA stripe", false};
      return false;
    }
#if defined(GGML_GEMMINI_TESTING)
    {
      std::lock_guard lock(test_mutex);
      const size_t index = counters.collector_events++;
      if (event.direct_residual || event.rmd_packet)
        ++counters.collector_handles;
      if (index < kTestStripeTraceCapacity) {
        counters.collector_row_begin[index] = event.row_begin;
        counters.collector_row_end[index] = event.row_end;
        counters.collector_theta[index] = theta;
      }
    }
#endif
    return true;
  }

  void mark_rmd_terminal_success() noexcept {
#if defined(GGML_GEMMINI_TESTING)
    std::lock_guard lock(test_mutex);
    counters.rmd_terminal_event = ++counters.order_event_sequence;
#endif
  }

  void copy_staged_output() noexcept {
    float *destination = args.f_out;
    const size_t row_stride =
        args.stride_f_out == 0 ? args.J : args.stride_f_out;
    const size_t col_stride =
        args.col_stride_f_out == 0 ? 1 : args.col_stride_f_out;
    for (size_t row = 0; row < args.I; ++row) {
      for (size_t col = 0; col < args.J; ++col) {
        const size_t offset = row * row_stride + col * col_stride;
        destination[offset] = staged_output[offset];
      }
    }
#if defined(GGML_GEMMINI_TESTING)
    std::lock_guard lock(test_mutex);
    ++counters.commit;
    counters.commit_event = ++counters.order_event_sequence;
#endif
  }

  ggml_gemmini_args_t &args;
  ggml_gemmini_args_t runtime_args;
  std::vector<float> staged_output;
  std::vector<CapturedExsiaStripe> captured;
  std::unique_ptr<::im2p::gemmini::Run> run;
  quants::act::exsia::StripeReadySink sink;
  Result collector_result{};
  bool sink_installed = false;
  bool fenced = false;
  bool finished = false;
};

ExsiaFullExecution::ExsiaFullExecution(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
ExsiaFullExecution::ExsiaFullExecution(ExsiaFullExecution &&) noexcept =
    default;
ExsiaFullExecution &
ExsiaFullExecution::operator=(ExsiaFullExecution &&) noexcept = default;
ExsiaFullExecution::~ExsiaFullExecution() = default;

Result ExsiaFullExecution::install_sink() noexcept {
  if (!impl_ || impl_->sink_installed || impl_->finished) {
    return {Error::invalid_state, "IM2P ExSIA FULL sink is not installable",
            false};
  }
  impl_->args.exsia_stripe_ready_sink = &impl_->sink;
  impl_->sink_installed = true;
  return {};
}

ExsiaFullExecutionStart
start_exsia_full_execution(ggml_gemmini_args_t &args) noexcept {
  size_t output_extent = 0;
  if (!checked_output_extent(args, output_extent) ||
      args.activation_rows_per_stripe == 0) {
    return {{Error::invalid_contract,
             "invalid IM2P ExSIA FULL output or stripe layout", false},
            {}};
  }
#if defined(GGML_GEMMINI_TESTING)
  {
    std::lock_guard lock(test_mutex);
    if (injected_failure == TestFailure::collector_allocation) {
      return {{Error::out_of_memory,
               "injected ExSIA FULL collector allocation failure", false},
              {}};
    }
  }
#endif
  std::unique_ptr<ExsiaFullExecution::Impl> impl;
  try {
    impl = std::make_unique<ExsiaFullExecution::Impl>(args);
    impl->staged_output.assign(output_extent, 0.0f);
    const size_t stripe_count =
        (args.I - 1) / args.activation_rows_per_stripe + 1;
    impl->captured.reserve(stripe_count);
  } catch (const std::bad_alloc &) {
    return {{Error::out_of_memory,
             "failed to allocate IM2P ExSIA FULL transaction", false},
            {}};
  } catch (...) {
    return {{Error::execution_failure,
             "failed to initialize IM2P ExSIA FULL transaction", false},
            {}};
  }
  std::unique_ptr<ExsiaFullExecution> execution(
      new (std::nothrow) ExsiaFullExecution(std::move(impl)));
  if (!execution) {
    return {{Error::out_of_memory, "failed to allocate IM2P ExSIA FULL handle",
             false},
            {}};
  }
  return {{}, std::move(execution)};
}

Completion ExsiaFullExecution::finish(bool quantization_succeeded) noexcept {
  if (!impl_ || impl_->finished || !impl_->sink_installed) {
    return {{Error::invalid_state,
             "IM2P ExSIA FULL transaction is not finishable", false},
            {}};
  }
  impl_->finished = true;
  if (impl_->args.exsia_stripe_ready_sink == &impl_->sink) {
    impl_->args.exsia_stripe_ready_sink = nullptr;
  }
  if (!impl_->collector_result.ok())
    return {impl_->collector_result, {}};
  if (!quantization_succeeded) {
    return {{Error::execution_failure, "ExSIA quantization failed", false}, {}};
  }

  impl_->runtime_args = impl_->args;
  impl_->runtime_args.D = nullptr;
  impl_->runtime_args.repeating_bias = false;
  impl_->runtime_args.f_out = impl_->staged_output.data();
  impl_->runtime_args.exsia_stripe_ready_sink = nullptr;
#if defined(GGML_GEMMINI_TESTING)
  TestFailure failure;
  {
    std::lock_guard lock(test_mutex);
    ++counters.full;
    failure = injected_failure;
    if (failure == TestFailure::execute) {
      production_failed = true;
      return {
          {Error::execution_failure, "injected IM2P execute failure", false},
          {}};
    }
  }
#endif
  auto started = ::im2p::gemmini::execute(&impl_->runtime_args,
                                          ::im2p::gemmini::Mode::full,
                                          ::im2p::gemmini::Options{65536});
  if (!started.status.ok())
    return {translate(started.status), {}};
  if (!started.run) {
    return {{Error::invalid_state, "IM2P execute returned no FULL run", false},
            {}};
  }
  impl_->run = std::move(started.run);
#if defined(GGML_GEMMINI_TESTING)
  {
    std::lock_guard lock(test_mutex);
    ++counters.fence;
  }
#endif
  Completion completion = translate(::im2p::gemmini::fence(*impl_->run));
  impl_->fenced = true;
#if defined(GGML_GEMMINI_TESTING)
  if (failure == TestFailure::fence) {
    return {{Error::execution_failure, "injected IM2P fence failure", false},
            completion.stats};
  }
#endif
  if (!completion.result.ok())
    return completion;

  const Result rmd =
      apply_captured_rmd(impl_->runtime_args, impl_->staged_output.data(),
                         impl_->staged_output.size(), impl_->captured);
  if (!rmd.ok())
    return {rmd, completion.stats};
  impl_->mark_rmd_terminal_success();
  impl_->copy_staged_output();
  return completion;
}

ExsiaStripePipeline::ExsiaStripePipeline(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ExsiaStripePipeline::ExsiaStripePipeline(ExsiaStripePipeline &&) noexcept =
    default;
ExsiaStripePipeline &
ExsiaStripePipeline::operator=(ExsiaStripePipeline &&) noexcept = default;
ExsiaStripePipeline::~ExsiaStripePipeline() = default;

Result ExsiaStripePipeline::install_sink() noexcept {
  if (!impl_ || !impl_->run || impl_->sink_installed || impl_->finished) {
    return {Error::invalid_state, "IM2P ExSIA sink is not installable", false};
  }
  impl_->args.exsia_stripe_ready_sink = &impl_->sink;
  impl_->sink_installed = true;
  return {};
}

ExsiaStripePipelineStart
start_exsia_stripe_pipeline(ggml_gemmini_args_t &args) noexcept {
#if defined(GGML_GEMMINI_TESTING)
  TestFailure failure = TestFailure::none;
  {
    std::lock_guard lock(test_mutex);
    ++counters.pipeline;
    failure = injected_failure;
    if (failure == TestFailure::execute) {
      production_failed = true;
      counters.production_error = Error::execution_failure;
      return {
          {Error::execution_failure, "injected IM2P execute failure", false},
          {}};
    }
  }
#endif
  size_t output_extent = 0;
  if (!checked_output_extent(args, output_extent) ||
      args.activation_rows_per_stripe == 0) {
    return {{Error::invalid_contract,
             "invalid IM2P ExSIA output or stripe layout", false},
            {}};
  }

  std::unique_ptr<ExsiaStripePipeline::Impl> impl;
  try {
    impl = std::make_unique<ExsiaStripePipeline::Impl>(args);
    impl->staged_output.assign(output_extent, 0.0f);
    const size_t stripe_count =
        (args.I - 1) / args.activation_rows_per_stripe + 1;
    impl->published.reserve(stripe_count);
  } catch (const std::bad_alloc &) {
    return {
        {Error::out_of_memory, "failed to allocate IM2P ExSIA pipeline", false},
        {}};
  } catch (...) {
    return {{Error::execution_failure,
             "failed to initialize IM2P ExSIA pipeline", false},
            {}};
  }

  impl->runtime_args.f_out = impl->staged_output.data();
  impl->runtime_args.exsia_stripe_ready_sink = nullptr;
  auto started = ::im2p::gemmini::execute(
      &impl->runtime_args, ::im2p::gemmini::Mode::stripe_pipeline,
      ::im2p::gemmini::Options{65536});
  if (!started.status.ok()) {
    return {translate(started.status), {}};
  }
  if (!started.run) {
    return {
        {Error::invalid_state, "IM2P execute returned no pipeline run", false},
        {}};
  }
  impl->run = std::move(started.run);
#if defined(GGML_GEMMINI_TESTING)
  if (failure == TestFailure::blocked_submit) {
    ::im2p::gemmini::RunTestAccess::hold_progress(*impl->run);
  } else if (failure == TestFailure::progress) {
    ::im2p::gemmini::RunTestAccess::inject_progress_failure(*impl->run);
    std::lock_guard lock(test_mutex);
    ++counters.progress_failures;
  } else if (failure == TestFailure::poll) {
    ::im2p::gemmini::RunTestAccess::inject_poll_failure(*impl->run);
    std::lock_guard lock(test_mutex);
    ++counters.poll_failures;
  }
  impl->register_run();
#endif

  std::unique_ptr<ExsiaStripePipeline> pipeline(
      new (std::nothrow) ExsiaStripePipeline(std::move(impl)));
  if (!pipeline) {
    return {{Error::out_of_memory,
             "failed to allocate IM2P ExSIA pipeline handle", false},
            {}};
  }
  return {{}, std::move(pipeline)};
}

Completion ExsiaStripePipeline::finish(bool quantization_succeeded) noexcept {
  if (!impl_ || !impl_->run || impl_->finished) {
    return {
        {Error::invalid_state, "IM2P ExSIA pipeline is not finishable", false},
        {}};
  }
  impl_->finished = true;
  impl_->args.exsia_stripe_ready_sink = nullptr;
#if defined(GGML_GEMMINI_TESTING)
  {
    std::lock_guard lock(test_mutex);
    ++counters.fence;
  }
#endif
  Completion completion = translate(::im2p::gemmini::fence(*impl_->run));
  impl_->fenced = true;
#if defined(GGML_GEMMINI_TESTING)
  {
    std::lock_guard lock(test_mutex);
    counters.first_publish_cycle = completion.stats.first_publish_cycle;
    counters.first_activation_read_cycle =
        completion.stats.first_activation_read_cycle;
    if (injected_failure == TestFailure::blocked_submit &&
        completion.result.error == Error::execution_failure) {
      counters.fence_saw_execution_failure = true;
    }
  }
  impl_->unregister_run();
  TestFailure failure = TestFailure::none;
  {
    std::lock_guard lock(test_mutex);
    failure = injected_failure;
  }
#endif

  if (!quantization_succeeded || !impl_->sink_result.ok() ||
      !completion.result.ok()
#if defined(GGML_GEMMINI_TESTING)
      || failure == TestFailure::fence
#endif
  ) {
    (void)impl_->authorize(false);
#if defined(GGML_GEMMINI_TESTING)
    if (failure == TestFailure::fence && completion.result.ok()) {
      return {{Error::execution_failure, "injected IM2P fence failure", false},
              completion.stats};
    }
#endif
    if (!impl_->sink_result.ok()) {
      return {impl_->sink_result, completion.stats};
    }
    if (!quantization_succeeded) {
      return {{Error::execution_failure, "ExSIA quantization failed", false},
              completion.stats};
    }
    return completion;
  }

  const auto output_stage =
      ::im2p::gemmini::acquire_pipeline_output_stage(*impl_->run);
  if (!output_stage.status.ok() || output_stage.data == nullptr) {
    (void)impl_->authorize(false);
    return {translate(output_stage.status), completion.stats};
  }

  const Result rmd =
      apply_captured_rmd(impl_->runtime_args, output_stage.data,
                         output_stage.element_count, impl_->published);
  if (!rmd.ok()) {
    (void)impl_->authorize(false);
    return {rmd, completion.stats};
  }
  impl_->mark_rmd_terminal_success();
  const Result authorized = impl_->authorize(true);
  if (!authorized.ok()) {
    return {authorized, completion.stats};
  }
  impl_->copy_staged_output();
  return completion;
}

#if defined(GGML_GEMMINI_TESTING)
void test_reset() noexcept {
  std::lock_guard lock(test_mutex);
  GGML_ASSERT(active_blocked_run == nullptr);
  counters = {};
  injected_failure = TestFailure::none;
  production_failed = false;
}

void test_inject_failure(TestFailure failure) noexcept {
  std::lock_guard lock(test_mutex);
  injected_failure = failure;
}

bool test_wait_for_blocked_producer() noexcept {
  ::im2p::gemmini::Run *run = nullptr;
  {
    std::unique_lock lock(test_mutex);
    if (!test_changed.wait_for(lock, std::chrono::seconds(5),
                               [] { return active_blocked_run != nullptr; })) {
      return false;
    }
    run = active_blocked_run;
  }
  const bool blocked =
      ::im2p::gemmini::RunTestAccess::wait_for_blocked_submit(*run, 1);
  if (blocked) {
    std::lock_guard lock(test_mutex);
    ++counters.blocked_producers;
  }
  return blocked;
}

void test_release_blocked_producer_with_error() noexcept {
  ::im2p::gemmini::Run *run = nullptr;
  {
    std::lock_guard lock(test_mutex);
    run = active_blocked_run;
  }
  if (run != nullptr) {
    ::im2p::gemmini::RunTestAccess::inject_execution_failure(*run);
  }
}

TestCounters test_counters() noexcept {
  std::lock_guard lock(test_mutex);
  return counters;
}

bool test_production_failed() noexcept {
  std::lock_guard lock(test_mutex);
  return production_failed;
}

bool test_should_fail_quantization() noexcept {
  std::lock_guard lock(test_mutex);
  if (injected_failure != TestFailure::quantization) {
    return false;
  }
  ++counters.quantization_failures;
  return true;
}

void test_record_production_failure(Error error) noexcept {
  std::lock_guard lock(test_mutex);
  production_failed = true;
  counters.production_error = error;
}

void test_observe_stripe_dispatch() noexcept {
  std::lock_guard lock(test_mutex);
  ++counters.stripe;
}

void test_observe_hardware_dispatch() noexcept {
  std::lock_guard lock(test_mutex);
  ++counters.hardware;
}

#endif

void log_failure(const char *operation, const Result &result) noexcept {
  if (result.ok()) {
    return;
  }
  GGML_LOG_ERROR("IM2P %s failed: %s (error=%u, native_contract=%u)\n",
                 operation ? operation : "operation",
                 result.message ? result.message : "unknown error",
                 static_cast<unsigned>(result.error),
                 result.native_contract ? 1U : 0U);
}

void log_stats(const char *operation, const Stats &stats) noexcept {
  GGML_LOG_INFO("IM2P %s: cycles=%llu compute=%llu overlap=%llu tiles=%llu "
                "stripes=%llu/%llu\n",
                operation ? operation : "operation",
                static_cast<unsigned long long>(stats.total_cycles),
                static_cast<unsigned long long>(stats.compute_cycles),
                static_cast<unsigned long long>(stats.overlap_cycles),
                static_cast<unsigned long long>(stats.completed_output_tiles),
                static_cast<unsigned long long>(stats.completed_stripes),
                static_cast<unsigned long long>(stats.stripes_published));
}

} // namespace ggml::gemmini::im2p_adapter
