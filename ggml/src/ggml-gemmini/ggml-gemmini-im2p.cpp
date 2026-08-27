#include "ggml-gemmini-im2p.hpp"

#include "ggml-gemmini-args.h"
#include "ggml-gemmini-matmul.hpp"
#include "ggml-gemmini-telemetry.hpp"
#include "ggml-impl.h"
#include "im2p_gemmini_frontend.hpp"
#include "quants/act/exsia/exsia.hpp"
#include "quants/act/exsia/types.hpp"
#include "residual/direct/direct-executor.hpp"
#include "residual/rmd/rmd-compose.hpp"
#include "residual/rmd/rmd-im2p-executor.hpp"
#include <im2p_sim.h>

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

::im2p::gemmini::Status to_frontend_status(const Result &result) noexcept {
  using Code = ::im2p::gemmini::StatusCode;
  Code code = Code::execution_failure;
  switch (result.error) {
  case Error::success: code = Code::success; break;
  case Error::invalid_argument: code = Code::invalid_argument; break;
  case Error::invalid_contract: code = Code::invalid_contract; break;
  case Error::unsupported_route: code = Code::unsupported_route; break;
  case Error::invalid_state: code = Code::invalid_state; break;
  case Error::backpressure: code = Code::backpressure; break;
  case Error::out_of_memory: code = Code::out_of_memory; break;
  case Error::execution_failure: break;
  }
  return {code, ::im2p::gemmini::Route::unknown, result.native_contract,
          result.message};
}

Result from_rmd_status(rmd::RmdStatus status) noexcept {
  using Source = rmd::RmdStatus;
  Error error = Error::execution_failure;
  switch (status) {
  case Source::success:
    return {};
  case Source::invalid_arguments:
  case Source::invalid_packet:
    error = Error::invalid_argument;
    break;
  case Source::unsupported_route:
  case Source::residual_too_wide:
    error = Error::unsupported_route;
    break;
  case Source::allocation_failure:
    error = Error::out_of_memory;
    break;
  case Source::overflow:
  case Source::execution_failed:
    break;
  }
  return {error, rmd::rmd_status_message(status), false};
}

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

void copy_staged_output(const ggml_gemmini_args_t &args,
                        const std::vector<float> &staged) noexcept {
  const size_t row_stride = args.stride_f_out == 0 ? args.J : args.stride_f_out;
  const size_t col_stride =
      args.col_stride_f_out == 0 ? 1 : args.col_stride_f_out;
  for (size_t row = 0; row < args.I; ++row) {
    for (size_t column = 0; column < args.J; ++column) {
      const size_t offset = row * row_stride + column * col_stride;
      args.f_out[offset] = staged[offset];
    }
  }
}

#if defined(GGML_GEMMINI_TESTING)
std::mutex test_mutex;
std::condition_variable test_changed;
TestCounters counters;
TestFailure injected_failure = TestFailure::none;
bool production_failed = false;
TestRuntimeArgsObserver runtime_args_observer = nullptr;
void *runtime_args_observer_data = nullptr;
::im2p::gemmini::Run *active_blocked_run = nullptr;

rmd::Im2pProviderTestFault provider_fault(TestFailure failure) noexcept {
  switch (failure) {
  case TestFailure::provider:
    return rmd::Im2pProviderTestFault::write_failure;
  case TestFailure::provider_read:
    return rmd::Im2pProviderTestFault::read_failure;
  case TestFailure::provider_watchdog:
    return rmd::Im2pProviderTestFault::watchdog;
  case TestFailure::provider_k_overflow:
    return rmd::Im2pProviderTestFault::k_accumulation_overflow;
  case TestFailure::provider_block_overflow:
    return rmd::Im2pProviderTestFault::block_scale_overflow;
  case TestFailure::provider_cancel_between_dots:
    return rmd::Im2pProviderTestFault::cancel_after_first_dot;
  default:
    return rmd::Im2pProviderTestFault::none;
  }
}

void observe_runtime_args(TestRuntimeArgsSite site,
                          const ggml_gemmini_args_t &args) noexcept {
  TestRuntimeArgsObserver observer = nullptr;
  void *user_data = nullptr;
  {
    std::lock_guard lock(test_mutex);
    observer = runtime_args_observer;
    user_data = runtime_args_observer_data;
  }
  if (observer != nullptr) {
    observer(site, args.matmul_layer.c_str(), user_data);
  }
}
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

static Stats translate_stats(
    const im2p_work_stats_extended_t &source) noexcept {
  const auto &base = source.base;
  return Stats{
      base.work_total_cycles,
      base.activation_read_requests,
      base.weight_read_requests,
      base.scale_read_requests,
      base.output_write_requests,
      base.output_write_responses,
      base.activation_wait_cycles,
      base.weight_wait_cycles,
      base.scale_wait_cycles,
      base.output_wait_cycles,
      base.stripe_host_wait_cycles,
      base.drain_cycles,
      base.weight_preload_cycles,
      base.same_block_scale_hits,
      base.next_scale_hits,
      base.scale_demand_misses,
      base.compute_cycles,
      base.overlap_cycles,
      base.activation_overlap_cycles,
      base.weight_overlap_cycles,
      base.scale_overlap_cycles,
      base.completed_fragments,
      base.completed_output_tiles,
      base.completed_stripes,
      base.stripes_published,
      base.stripe_rows_published,
      base.weight_bank_activations,
      source.cross_stripe_overlap_cycles,
      source.lookahead_prepared,
      source.lookahead_publish_cycle,
      source.lookahead_first_activation_cycle,
      source.lookahead_first_weight_cycle,
      source.lookahead_weight_preload_cycle,
      source.lookahead_weight_requests,
      source.lookahead_weight_reuse_hits,
      source.lookahead_scale_cycle,
      source.lookahead_scale_requests,
      source.lookahead_scale_reuses,
      source.current_stripe_completion_cycle,
      source.lookahead_ready_cycle,
      source.lookahead_start_cycle,
  };
}

Completion translate(const ::im2p::gemmini::FenceResult &result,
                     ::im2p::gemmini::Mode mode,
                     std::uint64_t expected_publications,
                     std::uint64_t expected_published_rows) noexcept {
  const auto &base = result.stats.base;
  Stats stats = translate_stats(result.stats);
  Result translated = translate(result.status);
  if (!translated.ok()) {
    return {translated, stats};
  }

  if (mode == ::im2p::gemmini::Mode::full) {
    if (expected_publications != 0 || expected_published_rows != 0 ||
        base.stripes_published != 0 || base.stripe_rows_published != 0) {
      return {{Error::invalid_contract,
               "FULL IM2P statistics must publish zero stripes and rows",
               false},
              stats};
    }
  } else if (expected_publications == 0 || expected_published_rows == 0 ||
             base.stripes_published != expected_publications ||
             base.stripe_rows_published != expected_published_rows) {
    return {{Error::invalid_contract,
             "PIPELINE IM2P publication statistics do not match canonical geometry",
             false},
            stats};
  }
  Completion completion{translated, stats};
  completion.semantic_completion_count = result.semantic_completion_count;
  completion.rmd_dot_calls = result.rmd_dot_calls;
  completion.rmd_stats = translate_stats(result.rmd_stats);
  return completion;
}

static Result validate_stripe_timings(
    const ::im2p::gemmini::StripeRtlTimingView &timings,
    const ggml_gemmini_args_t &args, const Stats &stats,
    std::uint64_t expected_run_id) noexcept {
  if (args.activation_rows_per_stripe == 0) {
    return {Error::invalid_contract,
            "PIPELINE stripe timing geometry has zero rows per stripe", false};
  }
  const std::uint64_t expected_count =
      1 + (static_cast<std::uint64_t>(args.I) - 1) /
              static_cast<std::uint64_t>(args.activation_rows_per_stripe);
  if (timings.data == nullptr || timings.size != expected_count ||
      stats.rtl_stripes_published != expected_count ||
      stats.rtl_stripe_rows_published != args.I) {
    return {Error::invalid_contract,
            "PIPELINE stripe timing count does not match publication statistics",
            false};
  }
  std::size_t expected_row_begin = 0;
  for (std::size_t index = 0; index < timings.size; ++index) {
    const auto &timing = timings[index];
    const std::size_t expected_row_end =
        std::min(args.I, expected_row_begin + args.activation_rows_per_stripe);
    if (timing.run_id != expected_run_id || timing.stripe_id != index ||
        timing.slot != index % 2 || timing.row_begin != expected_row_begin ||
        timing.row_end != expected_row_end ||
        timing.publish_to_completion_cycles !=
            timing.completion_cycle - timing.publish_cycle) {
      return {Error::invalid_contract,
              "PIPELINE stripe timing metadata is malformed or out of order",
              false};
    }
    expected_row_begin = expected_row_end;
  }
  if (expected_row_begin != args.I) {
    return {Error::invalid_contract,
            "PIPELINE stripe timings do not partition the output rows", false};
  }
  return {};
}

static void emit_stripe_timings(
    const ::im2p::gemmini::StripeRtlTimingView &timings,
    const ggml_gemmini_args_t &args) noexcept {
  for (const auto &timing : timings) {
    Im2pStripeTelemetry record{};
    record.layer = args.matmul_layer;
    record.run_id = timing.run_id;
    record.stripe_id = timing.stripe_id;
    record.slot = timing.slot;
    record.row_begin = timing.row_begin;
    record.row_end = timing.row_end;
    record.publish_cycle = timing.publish_cycle;
    record.completion_cycle = timing.completion_cycle;
    emit_cycle_telemetry(record);
  }
}

static Result validate_residual_stripe_timings(
    const ::im2p::gemmini::FenceResult &result,
    std::uint64_t expected_run_id) noexcept {
  const Result status = translate(result.status);
  if (!status.ok())
    return status;

  const auto count = result.semantic_completion_count;
  if (result.semantic_stripes.size != count ||
      result.residual_stripe_timings.size != count ||
      (count != 0 && (result.semantic_stripes.data == nullptr ||
                      result.residual_stripe_timings.data == nullptr))) {
    return {Error::invalid_contract,
            "semantic and RMD telemetry counts do not match", false};
  }

  std::uint64_t summed_calls = 0;
  std::uint64_t summed_work = 0;
  for (std::size_t index = 0; index < count; ++index) {
    const auto &semantic = result.semantic_stripes[index];
    const auto &timing = result.residual_stripe_timings[index];
    if (semantic.run_id != expected_run_id || timing.run_id != expected_run_id ||
        semantic.stripe_id != index || timing.stripe_id != index ||
        semantic.slot != timing.slot || semantic.row_begin != timing.row_begin ||
        semantic.row_end != timing.row_end || semantic.row_end < semantic.row_begin ||
        timing.rmd_dot_calls >
            std::numeric_limits<std::uint64_t>::max() - summed_calls ||
        timing.rmd_stats.base.work_total_cycles >
            std::numeric_limits<std::uint64_t>::max() - summed_work) {
      return {Error::invalid_contract,
              "semantic or RMD stripe telemetry is malformed", false};
    }
    summed_calls += timing.rmd_dot_calls;
    summed_work += timing.rmd_stats.base.work_total_cycles;
  }
  if (summed_calls != result.rmd_dot_calls ||
      summed_work != result.rmd_stats.base.work_total_cycles) {
    return {Error::invalid_contract,
            "RMD aggregate telemetry does not match stripe durations", false};
  }

  return {};
}

Result emit_residual_stripe_timings(
    const ::im2p::gemmini::FenceResult &result,
    const ggml_gemmini_args_t &args,
    std::uint64_t expected_run_id) noexcept {
  const Result status =
      validate_residual_stripe_timings(result, expected_run_id);
  if (!status.ok())
    return status;
  for (const auto &timing : result.residual_stripe_timings) {
    Im2pExecutionTelemetry record{};
    record.residual_domain = true;
    record.layer = args.matmul_layer;
    record.run_id = timing.run_id;
    record.stripe_id = timing.stripe_id;
    record.slot = timing.slot;
    record.row_begin = timing.row_begin;
    record.row_end = timing.row_end;
    record.rmd_dot_calls = timing.rmd_dot_calls;
    record.rtl_work_total_cycles = timing.rmd_stats.base.work_total_cycles;
    emit_cycle_telemetry(record);
  }
  return {};
}

Result gate_route(const ExsiaRouteRequest &request) noexcept {
  const auto supported_width = [](std::uint8_t bits) {
    return bits == 4 || bits == 8 || bits == 16;
  };
  if (!supported_width(request.activation_bits)) {
    return {Error::unsupported_route, "unsupported IM2P activation width", false};
  }
  if (!supported_width(request.weight_bits)) {
    return {Error::unsupported_route, "unsupported IM2P weight width", false};
  }
  if (request.artifact_activation_bits != request.activation_bits ||
      request.artifact_weight_bits != request.weight_bits) {
    return {Error::invalid_contract,
            "IM2P artifact identity does not match the requested route", false};
  }
  if (request.mode != PublicMode::full &&
      request.mode != PublicMode::stripe_pipeline) {
    return {Error::unsupported_route, "unsupported public matmul mode", false};
  }
  if (request.residual_backend != ResidualBackend::cpu_direct &&
      request.residual_backend != ResidualBackend::compact_ws) {
    return {Error::unsupported_route, "unsupported residual backend", false};
  }
  if (!request.exsia) {
    if (!request.rmd_enabled &&
        request.activation_bits == request.weight_bits) {
      return {};
    }
    return {Error::unsupported_route,
            request.rmd_enabled
                ? "non-ExSIA IM2P execution does not support RMD"
                : "IM2P routes require matched activation and weight widths",
            false};
  }
  if (request.build_identity != BuildIdentity::im2p_sim_ws) {
    return {Error::unsupported_route,
            "ExSIA IM2P requires the WS+IM2P_SIM build identity", false};
  }
  if (!request.rmd_enabled) {
    return {Error::unsupported_route, "ExSIA IM2P requires RMD", false};
  }
  if (request.activation_bits != request.weight_bits) {
    return {Error::unsupported_route,
            "ExSIA IM2P requires matched activation and weight widths", false};
  }
  switch (request.family) {
    case WeightFamily::h0:
      if (request.residual_backend != ResidualBackend::cpu_direct) {
        return {Error::unsupported_route,
                "H0 ExSIA requires CPU-direct residual execution", false};
      }
      return {};
    case WeightFamily::h1:
    case WeightFamily::hp1:
      return {};
    case WeightFamily::h2:
    case WeightFamily::hp2:
      return {Error::unsupported_route,
              "H2/HP2 ExSIA residual formats are unsupported", false};
    case WeightFamily::unsupported:
      return {Error::unsupported_route,
              "unsupported ExSIA residual weight family", false};
  }
  return {Error::unsupported_route, "unsupported ExSIA route", false};
}

Result gate_route(bool exsia, std::uint8_t activation_bits, bool rmd_enabled,
                  bool cpu_direct_rmd, std::uint8_t weight_bits) noexcept {
  return gate_route({exsia,
                     activation_bits,
                     weight_bits,
                     activation_bits,
                     weight_bits,
                     rmd_enabled,
                     PublicMode::full,
                     WeightFamily::h1,
                     cpu_direct_rmd ? ResidualBackend::cpu_direct
                                    : ResidualBackend::compact_ws,
                     BuildIdentity::im2p_sim_ws});
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
#endif

  size_t output_extent = 0;
  if (!checked_output_extent(args, output_extent)) {
    return {{Error::invalid_contract, "invalid IM2P FULL output layout", false},
            {}};
  }
  std::vector<float> staged_output;
  try {
    staged_output.assign(output_extent, 0.0f);
  } catch (const std::bad_alloc &) {
    return {{Error::out_of_memory, "failed to stage IM2P FULL output", false},
            {}};
  } catch (...) {
    return {{Error::execution_failure,
             "failed to initialize IM2P FULL output staging", false},
            {}};
  }
  runtime_args.f_out = staged_output.data();
#if defined(GGML_GEMMINI_TESTING)
  observe_runtime_args(TestRuntimeArgsSite::simple_full_before_execute,
                       runtime_args);
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
  const auto fenced = ::im2p::gemmini::fence(*started.run);
  Completion completion =
      translate(fenced, ::im2p::gemmini::Mode::full, 0, 0);
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
  if (completion.result.ok()) {
    copy_staged_output(args, staged_output);
  }
  return completion;
}

Completion run_stripe_pipeline(const ggml_gemmini_args_t &args) noexcept {
  ggml_gemmini_args_t runtime_args = args;
  runtime_args.D = nullptr;
  runtime_args.repeating_bias = false;
  GemminiGeometry geometry;
  if (!runtime_args.activation_geometry_matches(geometry)) {
    return {{Error::invalid_contract,
             "IM2P stripe pipeline activation geometry mismatch", false},
            {}};
  }
#if defined(GGML_GEMMINI_TESTING)
  TestFailure failure = TestFailure::none;
  {
    std::lock_guard lock(test_mutex);
    ++counters.stripe;
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
#endif
#if defined(GGML_GEMMINI_TESTING)
  observe_runtime_args(TestRuntimeArgsSite::simple_pipeline_before_execute,
                       runtime_args);
#endif
  auto started = ::im2p::gemmini::execute(
      &runtime_args, ::im2p::gemmini::Mode::stripe_pipeline,
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
  const uint64_t run_id = quants::act::exsia::next_exsia_run_id();
  size_t stripe_id = 0;
  for (size_t row_begin = 0; row_begin < runtime_args.I;
       row_begin += runtime_args.activation_rows_per_stripe, ++stripe_id) {
    quants::act::exsia::StripeReadyEvent event{};
    event.run_id = run_id;
    event.stripe_id = stripe_id;
    event.slot = stripe_id % 2;
    event.row_begin = row_begin;
    event.row_end =
        std::min(runtime_args.I,
                 row_begin + runtime_args.activation_rows_per_stripe);
    const auto status = ::im2p::gemmini::submit_stripe(*started.run, event);
    if (!status.ok()) {
#if defined(GGML_GEMMINI_TESTING)
      std::lock_guard lock(test_mutex);
      production_failed = true;
#endif
      return {translate(status), {}};
    }
#if defined(GGML_GEMMINI_TESTING)
    {
      std::lock_guard lock(test_mutex);
      const size_t index = static_cast<size_t>(counters.accepted_stripes++);
      if (index < kTestStripeTraceCapacity) {
        counters.stripe_trace_size = index + 1;
        counters.stripe_ids[index] = static_cast<int>(event.stripe_id);
        counters.slot_ids[index] = static_cast<int>(event.slot);
        counters.stripe_row_begin[index] = event.row_begin;
        counters.stripe_row_end[index] = event.row_end;
      }
    }
#endif
  }
#if defined(GGML_GEMMINI_TESTING)
  {
    std::lock_guard lock(test_mutex);
    ++counters.fence;
  }
#endif
  const auto fenced = ::im2p::gemmini::fence(*started.run);
  Completion completion = translate(
      fenced, ::im2p::gemmini::Mode::stripe_pipeline,
      static_cast<std::uint64_t>(stripe_id),
      static_cast<std::uint64_t>(runtime_args.I));
#if defined(GGML_GEMMINI_TESTING)
  if (failure == TestFailure::fence) {
    std::lock_guard lock(test_mutex);
    production_failed = true;
    return {{Error::execution_failure, "injected IM2P fence failure", false},
            completion.stats};
  }
#endif
  if (!completion.result.ok()) {
#if defined(GGML_GEMMINI_TESTING)
    std::lock_guard lock(test_mutex);
    production_failed = true;
#endif
    return completion;
  }
  const Result timing_status = validate_stripe_timings(
      fenced.stripe_rtl_timings, runtime_args, completion.stats, run_id);
  const Result residual_timing_status =
      validate_residual_stripe_timings(fenced, run_id);
  if (!timing_status.ok() || !residual_timing_status.ok()) {
#if defined(GGML_GEMMINI_TESTING)
    std::lock_guard lock(test_mutex);
    production_failed = true;
#endif
    return {timing_status.ok() ? residual_timing_status : timing_status,
            completion.stats};
  }
  const auto committed =
      ::im2p::gemmini::authorize_output_commit(*started.run, true);
  if (!committed.ok()) {
#if defined(GGML_GEMMINI_TESTING)
    std::lock_guard lock(test_mutex);
    production_failed = true;
#endif
    return {translate(committed), completion.stats};
  }
  completion.run_id = run_id;
  emit_stripe_timings(fenced.stripe_rtl_timings, args);
  const Result emitted = emit_residual_stripe_timings(fenced, args, run_id);
  return emitted.ok() ? completion : Completion{emitted, completion.stats};
}

struct CapturedExsiaStripe {
  quants::act::exsia::StripeReadyEvent event;
  std::int16_t theta = 0;
};

static void emit_quantization_timings(
    const std::vector<CapturedExsiaStripe> &stripes,
    const ggml_gemmini_args_t &args) noexcept {
  for (const auto &stripe : stripes) {
    QuantizationStripeTelemetry record{};
    record.layer = args.matmul_layer;
    record.run_id = stripe.event.run_id;
    record.stripe_id = stripe.event.stripe_id;
    record.slot = stripe.event.slot;
    record.row_begin = stripe.event.row_begin;
    record.row_end = stripe.event.row_end;
    record.start = stripe.event.quantization_start;
    record.end = stripe.event.quantization_end;
    emit_cycle_telemetry(record);
  }
}

static bool has_immediate_theta_prefix(
    const quants::act::exsia::Meta &metadata,
    const quants::act::exsia::StripeReadyEvent &event) noexcept {
  const auto invalid = std::numeric_limits<std::int16_t>::min();
  size_t committed = 0;
  for (const std::int16_t theta : metadata.theta) {
    committed += theta != invalid;
  }
  return event.stripe_id < metadata.theta.size() &&
         committed == event.stripe_id + 1 &&
         metadata.resolve_stripe_theta(static_cast<int>(event.stripe_id)) !=
             invalid &&
         (event.stripe_id + 1 == metadata.theta.size() ||
          metadata.resolve_stripe_theta(
              static_cast<int>(event.stripe_id + 1)) == invalid);
}

#if defined(GGML_GEMMINI_TESTING)
static WeightFamily concrete_weight_family(
    ggml_gemmini_args_t::im2p_weight_format_t format) noexcept {
  using Format = ggml_gemmini_args_t::im2p_weight_format_t;
  switch (format) {
  case Format::q4_h0:
  case Format::q8_h0:
  case Format::q16_h0:
    return WeightFamily::h0;
  case Format::q8_0_unpacked_to_h1:
  case Format::q4_h1:
  case Format::q8_h1:
  case Format::q16_h1:
    return WeightFamily::h1;
  case Format::q4_hp1:
  case Format::q8_hp1:
  case Format::q16_hp1:
    return WeightFamily::hp1;
  default:
    return WeightFamily::unsupported;
  }
}
#endif

// FULL owns this synchronous post-fence path. Unlike the compatibility matmul
// facade, it keeps one compact simulator alive across every canonical packet.
static Result apply_captured_rmd_full(
    const ggml_gemmini_args_t &runtime_args, float *output_data,
    size_t output_elements,
    const std::vector<CapturedExsiaStripe> &captured,
    ::im2p::gemmini::ResidualStripeStats &result_stats) noexcept {
#if defined(GGML_GEMMINI_TESTING)
  TestFailure failure;
  {
    std::lock_guard lock(test_mutex);
    failure = injected_failure;
  }
  const bool force_malformed_order =
      failure == TestFailure::malformed_completion;
#else
  constexpr bool force_malformed_order = false;
#endif
  std::unique_ptr<ggml_gemmini_args_t> rmd_args;
  std::vector<const CapturedExsiaStripe *> ordered;
  try {
    rmd_args = std::make_unique<ggml_gemmini_args_t>(runtime_args);
    rmd_args->f_out = output_data;
    size_t output_extent = 0;
    if (!checked_output_extent(*rmd_args, output_extent) ||
        output_extent > output_elements) {
      return {Error::invalid_contract,
              "FULL RMD staging does not cover the output layout", false};
    }
    ordered.reserve(captured.size());
    for (const auto &stripe : captured)
      ordered.push_back(&stripe);
    std::sort(ordered.begin(), ordered.end(), [](const auto *lhs, const auto *rhs) {
      return lhs->event.row_begin < rhs->event.row_begin;
    });
    auto &metadata =
        rmd_args->act_quant.storage().emplace<quants::act::exsia::Meta>();
    metadata.theta.assign(captured.size(),
                          std::numeric_limits<std::int16_t>::min());
    size_t next_row = 0;
    for (size_t index = 0; index < ordered.size(); ++index) {
      const auto &stripe = *ordered[index];
      if ((force_malformed_order && index == 1) ||
          stripe.event.stripe_id != index ||
          stripe.event.row_begin != next_row ||
          stripe.event.row_begin >= stripe.event.row_end ||
          stripe.event.row_end > runtime_args.I) {
        return {Error::invalid_contract,
                "captured FULL stripes are not canonical", false};
      }
      metadata.theta[index] = stripe.theta;
      next_row = stripe.event.row_end;
    }
    if (ordered.empty() || next_row != runtime_args.I) {
      return {Error::invalid_contract,
              "captured FULL stripes do not cover the output rows", false};
    }
  } catch (const std::bad_alloc &) {
    return {Error::out_of_memory, "failed to stage FULL residual metadata",
            false};
  } catch (...) {
    return {Error::execution_failure,
            "failed to initialize FULL residual staging", false};
  }

#if defined(GGML_GEMMINI_TESTING)
  test_observe_weight_family(concrete_weight_family(rmd_args->weight_format));
#endif

  const bool has_packet = std::any_of(
      ordered.begin(), ordered.end(), [](const CapturedExsiaStripe *stripe) {
        return stripe->event.rmd_packet != nullptr;
      });
  struct SimulatorDeleter {
    void operator()(im2p_sim_t *sim) const noexcept {
      if (sim != nullptr)
        im2p_sim_destroy(sim);
#if defined(GGML_GEMMINI_TESTING)
      if (sim != nullptr) {
        std::lock_guard lock(test_mutex);
        --counters.live_residual_simulators;
      }
#endif
    }
  };
  std::unique_ptr<im2p_sim_t, SimulatorDeleter> simulator;
  if (has_packet) {
#if defined(GGML_GEMMINI_TESTING)
    if (failure == TestFailure::simulator_create) {
      return {Error::out_of_memory,
              "injected FULL residual simulator creation failure", false};
    }
#endif
    simulator.reset(im2p_sim_create());
    if (!simulator) {
      return {Error::out_of_memory,
              "failed to create FULL residual simulator", false};
    }
#if defined(GGML_GEMMINI_TESTING)
    {
      std::lock_guard lock(test_mutex);
      ++counters.residual_simulator_creates;
      ++counters.live_residual_simulators;
    }
#endif
  }

  ::im2p::gemmini::ResidualStripeStats staged_stats{};
  rmd::RmdProviderStats staged_provider_stats{};
  for (const auto *captured_stripe : ordered) {
    const auto &event = captured_stripe->event;
    rmd::Correction correction = rmd::BlockScaledInt64Correction{};
    rmd::CompressedOutput compressed;
    rmd::RmdExecutionMetrics metrics{};
    residual::DirectExecutionMetrics direct_metrics{};
    rmd::RmdStatus status = rmd::RmdStatus::success;
    const bool no_residual =
        event.direct_residual == nullptr && event.rmd_packet == nullptr;
    if (event.direct_residual != nullptr) {
      status = residual::execute_direct_stripe(
          *rmd_args, *event.direct_residual, correction, &direct_metrics);
    } else if (event.rmd_packet != nullptr) {
#if defined(GGML_GEMMINI_TESTING)
      if (provider_fault(failure) != rmd::Im2pProviderTestFault::none) {
        status = rmd::execute_rmd_stripe_im2p_for_test(
            simulator.get(), *rmd_args, *event.rmd_packet, compressed, &metrics,
            provider_fault(failure));
      } else
#endif
      {
        status = rmd::execute_rmd_stripe_im2p(
            simulator.get(), *rmd_args, *event.rmd_packet, compressed, &metrics);
      }
    }
    if (status != rmd::RmdStatus::success)
      return from_rmd_status(status);
    if (metrics.im2p_dot_calls >
            std::numeric_limits<std::uint64_t>::max() -
                staged_stats.rmd_dot_calls ||
        rmd::checked_accumulate_provider_stats(
            staged_provider_stats, metrics.im2p_stats) !=
            rmd::RmdStatus::success) {
      return {Error::execution_failure,
              "FULL RMD provider statistics overflow", false};
    }
    staged_stats.rmd_dot_calls += metrics.im2p_dot_calls;

#if defined(GGML_GEMMINI_TESTING)
    {
      std::lock_guard lock(test_mutex);
      ++counters.residual_executions;
      counters.rmd_dot_calls += metrics.im2p_dot_calls;
    }
    if (failure == TestFailure::compose) {
      return {Error::execution_failure,
              "injected FULL RMD compose failure", false};
    }
#endif
    if (event.rmd_packet != nullptr) {
      status = rmd::compose_rmd_output(*event.rmd_packet, compressed, correction);
    }
    if (status == rmd::RmdStatus::success && !no_residual) {
      status = rmd::merge_rmd_correction_to(
          *rmd_args, output_data, event.row_begin, event.row_end, correction);
    }
    if (status != rmd::RmdStatus::success)
      return from_rmd_status(status);
#if defined(GGML_GEMMINI_TESTING)
    {
      std::lock_guard lock(test_mutex);
      ++counters.compositions;
      ++counters.rmd_calls;
      counters.rmd_events += direct_metrics.event_count;
      counters.rmd_packets += event.rmd_packet != nullptr;
    }
#endif
  }
#if defined(GGML_GEMMINI_TESTING)
  if (failure == TestFailure::rmd) {
    return {Error::execution_failure, "injected RMD failure", false};
  }
#endif
  rmd::detail::expand_im2p_provider_stats(staged_provider_stats,
                                          staged_stats.rmd_stats);
  result_stats = staged_stats;
  return {};
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

  static ::im2p::gemmini::Status residual_stage(
      void *opaque, im2p_sim_t *simulator,
      const quants::act::exsia::StripeReadyEvent &event,
      ::im2p::gemmini::ResidualStageView stage,
      ::im2p::gemmini::ResidualStripeStats &stats) noexcept {
    return static_cast<Impl *>(opaque)->apply_residual(simulator, event, stage,
                                                       stats);
  }

  ::im2p::gemmini::Status apply_residual(
      im2p_sim_t *simulator,
      const quants::act::exsia::StripeReadyEvent &event,
      ::im2p::gemmini::ResidualStageView stage,
      ::im2p::gemmini::ResidualStripeStats &stats) noexcept {
    if (!event.activation_metadata.has_value() ||
        event.activation_metadata->theta ==
            std::numeric_limits<std::int16_t>::min() ||
        event.row_begin >= event.row_end || event.row_end > runtime_args.I ||
        stage.data == nullptr ||
        (event.direct_residual != nullptr && event.rmd_packet != nullptr) ||
        (residual_mode == ::im2p::gemmini::ResidualStageMode::host_direct
             ? simulator != nullptr || event.rmd_packet != nullptr
             : simulator == nullptr || event.direct_residual != nullptr)) {
      return to_frontend_status(
          {Error::invalid_contract, "invalid PIPELINE residual callback event",
           false});
    }
    size_t output_extent = 0;
    if (!checked_output_extent(runtime_args, output_extent) ||
        output_extent > stage.element_count) {
      return to_frontend_status(
          {Error::invalid_contract,
           "PIPELINE residual stage does not cover the output layout", false});
    }

    ggml_gemmini_args_t stripe_args;
    try {
      stripe_args = runtime_args;
      auto &metadata = stripe_args.act_quant.storage()
                           .emplace<quants::act::exsia::Meta>();
      metadata.e_s = event.activation_metadata->e_s;
      metadata.rho = event.activation_metadata->rho;
      metadata.sigma = event.activation_metadata->sigma;
      metadata.theta.assign(event.stripe_id + 1,
                            std::numeric_limits<std::int16_t>::min());
      metadata.theta[event.stripe_id] = event.activation_metadata->theta;
    } catch (const std::bad_alloc &) {
      return to_frontend_status(
          {Error::out_of_memory,
           "failed to stage PIPELINE residual callback metadata", false});
    } catch (...) {
      return to_frontend_status(
          {Error::execution_failure,
           "failed to initialize PIPELINE residual callback", false});
    }

#if defined(GGML_GEMMINI_TESTING)
    TestFailure failure;
    {
      std::lock_guard lock(test_mutex);
      failure = injected_failure;
      ++counters.dense_completions;
      if (counters.residual_executions == 0)
        counters.dense_completions_at_first_residual =
            counters.dense_completions;
    }
    if (failure == TestFailure::dense ||
        failure == TestFailure::residual_execute) {
      const char *message = failure == TestFailure::dense
                                ? "injected dense completion failure"
                                : "injected residual execute failure";
      return to_frontend_status(
          {Error::execution_failure, message, false});
    }
#endif

    rmd::Correction correction = rmd::BlockScaledInt64Correction{};
    rmd::CompressedOutput compressed;
    rmd::RmdExecutionMetrics metrics{};
    residual::DirectExecutionMetrics direct_metrics{};
    rmd::RmdStatus status = rmd::RmdStatus::success;
    const bool no_residual =
        event.direct_residual == nullptr && event.rmd_packet == nullptr;
    if (event.direct_residual != nullptr) {
      status = residual::execute_direct_stripe(
          stripe_args, *event.direct_residual, correction, &direct_metrics);
    } else if (event.rmd_packet != nullptr) {
#if defined(GGML_GEMMINI_TESTING)
      if (provider_fault(failure) != rmd::Im2pProviderTestFault::none) {
        status = rmd::execute_rmd_stripe_im2p_for_test(
            simulator, stripe_args, *event.rmd_packet, compressed, &metrics,
            provider_fault(failure));
      } else
#endif
      {
        status = rmd::execute_rmd_stripe_im2p(
            simulator, stripe_args, *event.rmd_packet, compressed, &metrics);
      }
    }
    if (status != rmd::RmdStatus::success)
      return to_frontend_status(from_rmd_status(status));

#if defined(GGML_GEMMINI_TESTING)
    {
      std::lock_guard lock(test_mutex);
      ++counters.residual_executions;
      counters.rmd_dot_calls += metrics.im2p_dot_calls;
    }
    if (failure == TestFailure::compose) {
      return to_frontend_status(
          {Error::execution_failure, "injected RMD compose failure", false});
    }
#endif
    if (event.rmd_packet != nullptr)
      status = rmd::compose_rmd_output(*event.rmd_packet, compressed, correction);
    if (status == rmd::RmdStatus::success && !no_residual) {
      status = rmd::merge_rmd_correction_to(
          stripe_args, stage.data, event.row_begin, event.row_end, correction);
    }
    if (status != rmd::RmdStatus::success)
      return to_frontend_status(from_rmd_status(status));
#if defined(GGML_GEMMINI_TESTING)
    if (failure == TestFailure::rmd) {
      return to_frontend_status(
          {Error::execution_failure, "injected RMD failure", false});
    }
#endif

    stats.rmd_dot_calls = metrics.im2p_dot_calls;
    rmd::detail::expand_im2p_provider_stats(metrics.im2p_stats,
                                            stats.rmd_stats);
#if defined(GGML_GEMMINI_TESTING)
    {
      std::lock_guard lock(test_mutex);
      const size_t index = static_cast<size_t>(counters.compositions);
      ++counters.compositions;
      ++counters.rmd_calls;
      counters.rmd_events += direct_metrics.event_count;
      counters.rmd_packets += event.rmd_packet != nullptr;
      if (index < kTestStripeTraceCapacity) {
        counters.pipeline_callback_stripes[index] =
            static_cast<std::uint8_t>(event.stripe_id);
        counters.pipeline_merge_row_begin[index] = event.row_begin;
        counters.pipeline_merge_row_end[index] = event.row_end;
      }
    }
#endif
    return {};
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
        counters.stripe_row_begin[index] = event.row_begin;
        counters.stripe_row_end[index] = event.row_end;
      }
    }
#endif
    if (!sink_result.ok() || !run) {
      return false;
    }
#if defined(GGML_GEMMINI_TESTING)
    {
      std::lock_guard lock(test_mutex);
      if (injected_failure == TestFailure::incomplete_publication &&
          event.row_end == runtime_args.I) {
        sink_result = {Error::execution_failure,
                       "injected incomplete stripe publication", false};
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
    if (metadata == nullptr || !has_immediate_theta_prefix(*metadata, event)) {
      sink_result = {Error::invalid_contract,
                     "published ExSIA stripe is not at the immediate theta boundary",
                     false};
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
      if (rmd_succeeded &&
          injected_failure == TestFailure::output_authorization) {
        return {Error::execution_failure,
                "injected output authorization failure", false};
      }
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
  ::im2p::gemmini::ResidualStageMode residual_mode =
      ::im2p::gemmini::ResidualStageMode::none;
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
    if (metadata == nullptr || !has_immediate_theta_prefix(*metadata, event)) {
      collector_result = {Error::invalid_contract,
                          "collected ExSIA stripe is not at the immediate theta boundary",
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
  GemminiGeometry geometry;
  size_t output_extent = 0;
  if (!args.activation_geometry_matches(geometry) ||
      !checked_output_extent(args, output_extent)) {
    return {{Error::invalid_contract,
             "invalid IM2P ExSIA FULL output or stripe geometry", false},
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
    impl->captured.reserve(geometry.stripe_count);
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
  observe_runtime_args(TestRuntimeArgsSite::exsia_full_before_execute,
                       impl_->runtime_args);
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
  const auto fenced = ::im2p::gemmini::fence(*impl_->run);
  Completion completion =
      translate(fenced, ::im2p::gemmini::Mode::full, 0, 0);
  impl_->fenced = true;
#if defined(GGML_GEMMINI_TESTING)
  if (failure == TestFailure::fence) {
    return {{Error::execution_failure, "injected IM2P fence failure", false},
            completion.stats};
  }
#endif
  if (!completion.result.ok())
    return completion;

  ::im2p::gemmini::ResidualStripeStats full_rmd_stats{};
  const Result rmd = apply_captured_rmd_full(
      impl_->runtime_args, impl_->staged_output.data(),
      impl_->staged_output.size(), impl_->captured, full_rmd_stats);
  if (!rmd.ok())
    return {rmd, completion.stats};
  completion.rmd_dot_calls = full_rmd_stats.rmd_dot_calls;
  completion.rmd_stats = translate_stats(full_rmd_stats.rmd_stats);
  completion.semantic_completion_count = impl_->captured.size();
  if (!impl_->captured.empty())
    completion.run_id = impl_->captured.front().event.run_id;
  impl_->mark_rmd_terminal_success();
#if defined(GGML_GEMMINI_TESTING)
  {
    std::lock_guard lock(test_mutex);
    if (failure == TestFailure::output_copy) {
      return {{Error::execution_failure,
               "injected FULL staged-output copy failure", false},
              completion.stats};
    }
  }
#endif
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
  GemminiGeometry geometry;
  if (!args.activation_geometry_matches(geometry)) {
    return {{Error::invalid_contract,
             "invalid IM2P ExSIA pipeline activation geometry", false},
            {}};
  }
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
  if (!checked_output_extent(args, output_extent)) {
    return {{Error::invalid_contract,
             "invalid IM2P ExSIA output layout", false},
            {}};
  }

  std::unique_ptr<ExsiaStripePipeline::Impl> impl;
  try {
    impl = std::make_unique<ExsiaStripePipeline::Impl>(args);
    impl->staged_output.assign(output_extent, 0.0f);
    impl->published.reserve(geometry.stripe_count);
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
  impl->residual_mode =
      args.residual_route == residual::ResidualRoute::ws_packet
          ? ::im2p::gemmini::ResidualStageMode::im2p_compact
          : ::im2p::gemmini::ResidualStageMode::host_direct;
#if defined(GGML_GEMMINI_TESTING)
  test_observe_weight_family(concrete_weight_family(args.weight_format));
  observe_runtime_args(TestRuntimeArgsSite::exsia_pipeline_before_execute,
                       impl->runtime_args);
#endif
  const ::im2p::gemmini::Options frontend_options{
      65536, impl->residual_mode, impl.get(),
      &ExsiaStripePipeline::Impl::residual_stage};
  auto started = ::im2p::gemmini::execute(
      &impl->runtime_args, ::im2p::gemmini::Mode::stripe_pipeline,
      frontend_options);
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
  {
    std::lock_guard lock(test_mutex);
    ++counters.worker_starts;
  }
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
  } else if (failure == TestFailure::malformed_completion) {
    ::im2p::gemmini::RunTestAccess::invalidate_timing_capacity(*impl->run);
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
  TestFailure failure = TestFailure::none;
  {
    std::lock_guard lock(test_mutex);
    ++counters.fence;
    failure = injected_failure;
  }
#endif
  const auto fenced = ::im2p::gemmini::fence(*impl_->run);
  const std::uint64_t expected_publications = static_cast<std::uint64_t>(
      (impl_->runtime_args.I - 1) /
          impl_->runtime_args.activation_rows_per_stripe +
      1);
  Completion completion = translate(
      fenced, ::im2p::gemmini::Mode::stripe_pipeline, expected_publications,
      static_cast<std::uint64_t>(impl_->runtime_args.I));
  impl_->fenced = true;
#if defined(GGML_GEMMINI_TESTING)
  {
    std::lock_guard lock(test_mutex);
    counters.first_publish_cycle = completion.stats.rtl_first_publish_cycle;
    counters.first_activation_read_cycle =
        completion.stats.rtl_first_activation_read_cycle;
    if (injected_failure == TestFailure::blocked_submit &&
        completion.result.error == Error::execution_failure) {
      counters.fence_saw_execution_failure = true;
    }
  }
  impl_->unregister_run();
#endif

  if (completion.result.error == Error::invalid_contract &&
      quantization_succeeded && impl_->sink_result.ok()) {
    return completion;
  }
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

  if (impl_->published.empty()) {
    (void)impl_->authorize(false);
    return {{Error::invalid_contract,
             "successful ExSIA pipeline has no published stripe", false},
            completion.stats};
  }
  const std::uint64_t run_id = impl_->published.front().event.run_id;
  const Result timing_status = validate_stripe_timings(
      fenced.stripe_rtl_timings, impl_->runtime_args, completion.stats, run_id);
  const Result residual_timing_status =
      validate_residual_stripe_timings(fenced, run_id);
  if (!timing_status.ok() || !residual_timing_status.ok() ||
      impl_->published.size() != fenced.stripe_rtl_timings.size) {
    (void)impl_->authorize(false);
    return {!timing_status.ok()
                ? timing_status
                : (!residual_timing_status.ok()
                       ? residual_timing_status
                       : Result{Error::invalid_contract,
                                "ExSIA publications do not match stripe timings",
                                false}),
            completion.stats};
  }

  impl_->mark_rmd_terminal_success();
  const Result authorized = impl_->authorize(true);
  if (!authorized.ok()) {
    completion.result = authorized;
    return completion;
  }
#if defined(GGML_GEMMINI_TESTING)
  {
    std::lock_guard lock(test_mutex);
    if (failure == TestFailure::output_copy) {
      return {{Error::execution_failure,
               "injected staged-output copy failure", false},
              completion.stats};
    }
  }
#endif
  impl_->copy_staged_output();
  completion.run_id = run_id;
  emit_quantization_timings(impl_->published, impl_->args);
  emit_stripe_timings(fenced.stripe_rtl_timings, impl_->args);
  const Result emitted =
      emit_residual_stripe_timings(fenced, impl_->args, run_id);
  return emitted.ok() ? completion : Completion{emitted, completion.stats};
}

#if defined(GGML_GEMMINI_TESTING)
void test_reset() noexcept {
  std::lock_guard lock(test_mutex);
  GGML_ASSERT(active_blocked_run == nullptr);
  counters = {};
  injected_failure = TestFailure::none;
  production_failed = false;
  rmd::reset_im2p_provider_dot_attempts_for_test();
}

void test_set_runtime_args_observer(TestRuntimeArgsObserver observer,
                                    void *user_data) noexcept {
  std::lock_guard lock(test_mutex);
  runtime_args_observer = observer;
  runtime_args_observer_data = user_data;
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
  TestCounters snapshot = counters;
  snapshot.provider_dot_attempts =
      rmd::im2p_provider_dot_attempts_for_test();
  return snapshot;
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

void test_observe_activation_allocation() noexcept {
  std::lock_guard lock(test_mutex);
  ++counters.activation_allocations;
}

void test_observe_weight_family(WeightFamily family) noexcept {
  std::lock_guard lock(test_mutex);
  ++counters.weight_family_observations;
  if (counters.observed_weight_family == WeightFamily::unsupported) {
    counters.observed_weight_family = family;
  }
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

void log_rmd_stats(const Completion &completion,
                   const ggml_gemmini_args_t &args) noexcept {
#if LOG_CYCLE
  if (completion.rmd_dot_calls == 0 &&
      completion.rmd_stats.rtl_work_total_cycles == 0)
    return;
  Im2pExecutionTelemetry record{};
  record.residual_domain = true;
  record.residual_aggregate = true;
  record.layer = args.matmul_layer;
  record.run_id = completion.run_id;
  record.rmd_dot_calls = completion.rmd_dot_calls;
  record.rtl_work_total_cycles =
      completion.rmd_stats.rtl_work_total_cycles;
  emit_cycle_telemetry(record);
#else
  (void)completion;
  (void)args;
#endif
}

void log_stats(const char * mode, const Stats & stats,
               std::uint64_t run_id,
               const ggml_gemmini_args_t & args) noexcept {
#if LOG_CYCLE || LOG_DEBUG
  Im2pExecutionTelemetry record{};
  record.layer = args.matmul_layer;
  record.run_id = run_id;
  record.mode = mode ? mode : "unknown";
  record.activation_bits = GGML_GEMMINI_ACTIVATION_BITS;
  record.weight_bits = GGML_GEMMINI_WEIGHT_BITS;
  record.dim = DIM;
  record.problem_i = args.I; record.problem_j = args.J; record.problem_k = args.K;
  record.tile_i = args.tile_I; record.tile_j = args.tile_J; record.tile_k = args.tile_K;
  record.rtl_work_total_cycles = stats.rtl_work_total_cycles;
  record.rtl_compute_cycles = stats.rtl_compute_cycles;
  record.rtl_drain_cycles = stats.rtl_drain_cycles;
  record.rtl_activation_wait_cycles = stats.rtl_activation_wait_cycles;
  record.rtl_weight_wait_cycles = stats.rtl_weight_wait_cycles;
  record.rtl_scale_wait_cycles = stats.rtl_scale_wait_cycles;
  record.rtl_output_wait_cycles = stats.rtl_output_wait_cycles;
  record.rtl_overlap_cycles = stats.rtl_overlap_cycles;
  record.rtl_activation_overlap_cycles = stats.rtl_activation_overlap_cycles;
  record.rtl_weight_overlap_cycles = stats.rtl_weight_overlap_cycles;
  record.rtl_scale_overlap_cycles = stats.rtl_scale_overlap_cycles;
  record.rtl_completed_output_works = stats.rtl_completed_output_works;
  record.rtl_completed_fragments = stats.rtl_completed_fragments;
  record.rtl_scheduler_groups_completed = stats.rtl_scheduler_groups_completed;
  record.rtl_stripes_published = stats.rtl_stripes_published;
  record.rtl_stripe_rows_published = stats.rtl_stripe_rows_published;
  emit_cycle_telemetry(record);
#else
  (void) mode; (void) stats; (void) run_id; (void) args;
#endif
}

} // namespace ggml::gemmini::im2p_adapter
