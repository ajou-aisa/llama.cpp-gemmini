#include <ggml-backend.h>
#include <ggml-gemmini.h>
#include <ggml-quants.h>
#include <ggml.h>

#include "ggml-gemmini-args.h"
#include "ggml-gemmini-im2p.hpp"
#include "quants/act/exsia/exsia_shift.hpp"
#include "quants/act/quantize.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>

namespace {

#if GGML_GEMMINI_ACTIVATION_QUANT == 0
constexpr bool compiled_exsia = true;
constexpr int64_t K = 64;
constexpr int64_t I = 2 * GGML_GEMMINI_TEST_IM2P_DIM + 1;
#else
constexpr bool compiled_exsia = false;
constexpr int64_t K = 32;
constexpr int64_t I = 3;
#endif
constexpr int64_t J = 2;
constexpr float sentinel = 12345.0f;

bool check(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
  }
  return condition;
}

bool run_graph_overhead_regression() {
  for (const size_t size :
       {size_t{1}, size_t{17}, size_t{GGML_DEFAULT_GRAPH_SIZE}}) {
    for (const bool grads : {false, true}) {
      const size_t overhead = ggml_graph_overhead_custom(size, grads);
      ggml_init_params params{overhead, nullptr, true};
      ggml_context *context = ggml_init(params);
      if (!check(context != nullptr,
                 "exact graph overhead initializes a context")) {
        return false;
      }
      ggml_cgraph *graph = ggml_new_graph_custom(context, size, grads);
      const bool exact = graph != nullptr && ggml_used_mem(context) == overhead;
      ggml_free(context);
      if (!check(exact,
                 "graph byte count exactly covers aligned graph storage")) {
        return false;
      }
    }
  }
  return true;
}

int32_t shift_q_oracle(int32_t q, int16_t delta_theta) {
  if (delta_theta > 0) {
    const int shift = std::min<int>(delta_theta, 31);
    const int64_t shifted = static_cast<int64_t>(q) * (int64_t{1} << shift);
    return static_cast<int32_t>(
        std::clamp<int64_t>(shifted, std::numeric_limits<int32_t>::min(),
                            std::numeric_limits<int32_t>::max()));
  }
  if (delta_theta < 0) {
    const int shift = std::min<int>(-delta_theta, 31);
    const int64_t value = q;
    const int64_t offset = int64_t{1} << (shift - 1);
    return static_cast<int32_t>(value >= 0 ? (value + offset) >> shift
                                           : -(((-value) + offset) >> shift));
  }
  return q;
}

bool check_shift_value(int32_t q, int16_t delta_theta) {
  const int32_t actual =
      ggml::gemmini::quants::act::exsia::detail::shift_q_i32(q, delta_theta);
  const int32_t expected = shift_q_oracle(q, delta_theta);
  uint32_t actual_bits = 0;
  uint32_t expected_bits = 0;
  std::memcpy(&actual_bits, &actual, sizeof(actual_bits));
  std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
  return actual == expected && actual_bits == expected_bits;
}

bool run_exsia_shift_regression() {
  for (int32_t q = std::numeric_limits<int16_t>::min(); q < 0; ++q) {
    for (int delta = -31; delta <= 31; ++delta) {
      if (!check_shift_value(q, static_cast<int16_t>(delta))) {
        return check(
            false,
            "every negative int16 shift preserves numeric and bit output");
      }
    }
  }
  constexpr int32_t extrema[] = {
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::min() + 1,
      -1,
      0,
      1,
      std::numeric_limits<int32_t>::max() - 1,
      std::numeric_limits<int32_t>::max(),
  };
  for (const int32_t q : extrema) {
    for (int delta = -31; delta <= 31; ++delta) {
      if (!check_shift_value(q, static_cast<int16_t>(delta))) {
        return check(false,
                     "int32 extrema shifts preserve numeric and bit output");
      }
    }
  }
  return true;
}

const char *compiled_route() {
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
  return "exsia";
#elif GGML_GEMMINI_ACTIVATION_QUANT == 1
  return "tensor";
#elif GGML_GEMMINI_ACTIVATION_QUANT == 2
  return "token";
#elif GGML_GEMMINI_ACTIVATION_QUANT == 3
  return "block";
#elif GGML_GEMMINI_ACTIVATION_QUANT == 4
  return "stripe";
#else
#error                                                                         \
    "test-gemmini-im2p-routing requires EXSIA, TENSOR, TOKEN, BLOCK, or STRIPE"
#endif
}

std::vector<float> make_activations() {
  std::vector<float> values(I * K);
  for (int64_t i = 0; i < I; ++i) {
    for (int64_t k = 0; k < K; ++k) {
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
      values[i * K + k] = 0.25f * static_cast<float>((i + 3 * k) % 15 - 7);
      if (i % GGML_GEMMINI_TEST_IM2P_DIM == 0 && k == 0) {
        values[i * K + k] = 4096.0f + static_cast<float>(i);
      }
#else
      values[i * K + k] = 0.125f * static_cast<float>((i + 2 * k) % 11 - 5);
#endif
    }
  }
  return values;
}

std::vector<uint8_t> make_weights() {
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
  constexpr size_t blocks_per_row = K / QK8_0;
  std::vector<uint8_t> encoded(J * blocks_per_row * sizeof(block_q8_0));
  auto *blocks = reinterpret_cast<block_q8_0 *>(encoded.data());
  for (int64_t j = 0; j < J; ++j) {
    for (size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
      block_q8_0 &block = blocks[j * blocks_per_row + block_index];
      block.d = ggml_fp32_to_fp16(0.125f * static_cast<float>(j + 1));
      for (int k = 0; k < QK8_0; ++k) {
        block.qs[k] = static_cast<int8_t>(
            (5 * j + static_cast<int>(block_index) + k) % 13 - 6);
      }
    }
  }
  return encoded;
#else
  const size_t row_size = sizeof(float) + K;
  std::vector<uint8_t> encoded(J * row_size);
  for (int64_t j = 0; j < J; ++j) {
    const float scale = 0.25f * static_cast<float>(j + 1);
    uint8_t *row = encoded.data() + j * row_size;
    std::memcpy(row, &scale, sizeof(scale));
    for (int64_t k = 0; k < K; ++k) {
      row[sizeof(float) + k] =
          static_cast<uint8_t>(static_cast<int8_t>((3 * j + k) % 9 - 4));
    }
  }
  return encoded;
#endif
}

bool scalar_oracle(const std::vector<float> &activations,
                   const std::vector<uint8_t> &weights,
                   std::vector<float> &expected) {
  ggml_init_params params = {
      ggml_tensor_overhead() * 4 + static_cast<size_t>(I * K) * sizeof(float) +
          1024,
      nullptr,
      false,
  };
  ggml_context *context = ggml_init(params);
  if (context == nullptr) {
    return false;
  }
  ggml_tensor *activation = ggml_new_tensor_2d(context, GGML_TYPE_F32, K, I);
  std::memcpy(activation->data, activations.data(),
              activations.size() * sizeof(float));

  ggml_gemmini_args_t args{};
  args.I = I;
  args.J = J;
  args.K = K;
  args.sA = K;
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
  args.tile_I = 1;
  args.activation_rows_per_stripe = GGML_GEMMINI_TEST_IM2P_DIM;
  args.residual_route = ggml::gemmini::residual::ResidualRoute::cpu_direct;
#endif
  const bool allocated = args.A.allocate(I, K, GGML_GEMMINI_ACTIVATION_BITS);
  const bool quantized =
      allocated && ggml::gemmini::quants::quantize_activation(activation, args);
  std::vector<float> decoded(I * K);
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
  const auto *exsia_meta = std::get_if<ggml::gemmini::quants::act::exsia::Meta>(
      &args.act_quant.storage());
  std::vector<int32_t> residuals(I * K, 0);
  size_t residual_event_count = 0;
  if (exsia_meta != nullptr) {
    for (const auto &payload : exsia_meta->direct_residuals) {
      if (!payload) {
        continue;
      }
      residual_event_count += payload->events.size();
      for (const auto &event : payload->events) {
        const size_t row = payload->row_begin + event.local_row;
        if (row >= static_cast<size_t>(I) ||
            event.original_k >= static_cast<size_t>(K)) {
          ggml_free(context);
          return false;
        }
        residuals[row * K + event.original_k] = event.residual;
      }
    }
  }
  bool decoded_ok = quantized && exsia_meta != nullptr &&
                    residual_event_count != 0 && exsia_meta->theta.size() == 3;
  if (decoded_ok) {
    for (int64_t i = 0; i < I; ++i) {
      const int stripe = static_cast<int>(i / GGML_GEMMINI_TEST_IM2P_DIM);
      const int16_t theta = exsia_meta->resolve_stripe_theta(stripe);
      if (theta == std::numeric_limits<int16_t>::min()) {
        decoded_ok = false;
        break;
      }
      for (int64_t k = 0; k < K; ++k) {
        const int32_t value = args.A.get(i, k) + residuals[i * K + k];
        decoded[i * K + k] = std::ldexp(static_cast<float>(value), theta);
      }
    }
  }
#else
  const bool decoded_ok =
      quantized && ggml::gemmini::quants::dequantize_activation(
                       decoded.data(), K, 1, I, K, args);
#if GGML_GEMMINI_ACTIVATION_QUANT == 3
  const auto *block_meta = std::get_if<ggml::gemmini::quants::act::block::Meta>(
      &args.act_quant.storage());
  const bool valid_block_meta =
      block_meta != nullptr && block_meta->scales.size() == I &&
      block_meta->rmd_packets.empty() && block_meta->direct_residuals.empty() &&
      std::all_of(
          block_meta->scales.begin() + 1, block_meta->scales.end(),
          [&](float scale) { return scale == block_meta->scales.front(); });
#else
  const bool valid_block_meta = true;
#endif
#endif
  ggml_free(context);
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
  if (!decoded_ok) {
#else
  if (!decoded_ok || !valid_block_meta) {
#endif
    return false;
  }

  expected.assign(I * J, 0.0f);
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
  constexpr size_t blocks_per_row = K / QK8_0;
  const auto *blocks = reinterpret_cast<const block_q8_0 *>(weights.data());
  for (int64_t i = 0; i < I; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      for (int64_t k = 0; k < K; ++k) {
        const block_q8_0 &block =
            blocks[j * blocks_per_row + static_cast<size_t>(k) / QK8_0];
        const float weight =
            ggml_fp16_to_fp32(block.d) *
            static_cast<float>(block.qs[static_cast<size_t>(k) % QK8_0]);
        expected[i * J + j] += decoded[i * K + k] * weight;
      }
    }
  }
#else
  const size_t row_size = sizeof(float) + K;
  for (int64_t i = 0; i < I; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      const uint8_t *row = weights.data() + j * row_size;
      float scale = 0.0f;
      std::memcpy(&scale, row, sizeof(scale));
      for (int64_t k = 0; k < K; ++k) {
        expected[i * J + j] +=
            decoded[i * K + k] *
            static_cast<float>(static_cast<int8_t>(row[sizeof(float) + k])) *
            scale;
      }
    }
  }
#endif
  return true;
}

struct GraphCase {
  ggml_backend_t backend = nullptr;
  ggml_context *context = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *output = nullptr;
  std::vector<float> activations = make_activations();
  std::vector<uint8_t> weights = make_weights();

  bool initialize() {
    backend = ggml_backend_gemmini_init();
    if (backend == nullptr) {
      return false;
    }
    ggml_init_params params = {
        ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        nullptr,
        true,
    };
    context = ggml_init(params);
    if (context == nullptr) {
      return false;
    }
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
    ggml_tensor *weight = ggml_new_tensor_2d(context, GGML_TYPE_Q8_0, K, J);
#else
    ggml_tensor *weight =
        ggml_new_tensor_2d(context, GGML_TYPE_Q8_CHANNEL, K, J);
#endif
    ggml_tensor *activation = ggml_new_tensor_2d(context, GGML_TYPE_F32, K, I);
    output = ggml_mul_mat(context, weight, activation);
    buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
      return false;
    }
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(activation, activations.data(), 0,
                            activations.size() * sizeof(float));
    std::vector<float> initial(I * J, sentinel);
    ggml_backend_tensor_set(output, initial.data(), 0,
                            initial.size() * sizeof(float));
    graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    return ggml_backend_supports_op(backend, output);
  }

  ~GraphCase() {
    if (buffer != nullptr) {
      ggml_backend_buffer_free(buffer);
    }
    if (context != nullptr) {
      ggml_free(context);
    }
    if (backend != nullptr) {
      ggml_backend_free(backend);
    }
  }

  std::vector<float> read_output() const {
    std::vector<float> values(I * J);
    ggml_backend_tensor_get(output, values.data(), 0,
                            values.size() * sizeof(float));
    return values;
  }
};

bool all_sentinel(const std::vector<float> &values) {
  return std::all_of(values.begin(), values.end(),
                     [](float value) { return value == sentinel; });
}

#if GGML_GEMMINI_ACTIVATION_QUANT == 0
bool check_three_stripe_trace(
    const ggml::gemmini::im2p_adapter::TestCounters &counters) {
  return check(counters.stripe_trace_size == 3,
               "three stripe publications are traced") &&
         check(counters.stripe_ids[0] == 0 && counters.stripe_ids[1] == 1 &&
                   counters.stripe_ids[2] == 2,
               "stripe publications are ordered 0,1,2") &&
         check(counters.slot_ids[0] == 0 && counters.slot_ids[1] == 1 &&
                   counters.slot_ids[2] == 0,
               "ExSIA slots are reused in order 0,1,0");
}

bool run_exsia_success() {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
  setenv("GEMMINI_RMD_BACKEND", "CPU", 1);
  setenv("GEMMINI_STRIPE_JOB_CAPACITY", "2", 1);
  GraphCase test_case;
  if (!check(test_case.initialize(), "initialize real ExSIA Gemmini graph")) {
    return false;
  }
  std::vector<float> expected;
  if (!check(scalar_oracle(test_case.activations, test_case.weights, expected),
             "build ExSIA plus RMD scalar oracle")) {
    return false;
  }

  test_reset();
  const ggml_status status =
      ggml_backend_graph_compute(test_case.backend, test_case.graph);
  const auto counters = test_counters();
  const auto actual = test_case.read_output();
  bool ok =
      check(status == GGML_STATUS_SUCCESS, "ExSIA production graph succeeds") &&
      check(!test_production_failed(),
            "ExSIA production graph records no failure") &&
      check(
          counters.full == 0 && counters.pipeline == 1,
          "ExSIA dispatch executes exactly one pipeline run and no full run") &&
      check(counters.fence == 1, "ExSIA dispatch fences exactly once") &&
      check(counters.stripe == 3 && counters.accepted_stripes == 3,
            "all three post-fold stripes reach the frontend") &&
      check(counters.max_outstanding > 0 && counters.max_outstanding <= 2,
            "frontend keeps at most two stripes outstanding") &&
      check(counters.rmd_calls == 3 && counters.rmd_events > 0,
            "unchanged cpu_direct RMD consumes every stripe and real residual "
            "events") &&
      check(counters.authorize == 1 && counters.rmd_terminal_event != 0 &&
                counters.authorize_success_event > counters.rmd_terminal_event,
            "RMD reaches terminal success before output authorization") &&
      check(counters.commit == 1, "staged output commits exactly once") &&
      check(counters.hardware == 0 && counters.fallback == 0,
            "ExSIA route enters no hardware or fallback path") &&
      check(counters.live_runs == 0, "all frontend workers are joined") &&
      check(counters.first_activation_read_cycle >=
                    counters.first_publish_cycle &&
                counters.first_activation_read_cycle != 0,
            "activation reads do not precede post-fold publication") &&
      check_three_stripe_trace(counters);
  for (size_t index = 0; index < actual.size(); ++index) {
    const float tolerance = 1e-4f * std::max(1.0f, std::fabs(expected[index]));
    ok = check(std::isfinite(actual[index]) &&
                   std::fabs(actual[index] - expected[index]) <= tolerance,
               "ExSIA production output matches scalar RMD oracle") &&
         ok;
  }
  if (ok) {
    std::printf(
        "route=exsia mode=stripe_pipeline bits=8 dim=%d stripes=3 slots=0,1,0 "
        "capacity=2 rmd=cpu_direct rmd_terminal_event=%llu "
        "authorize_event=%llu "
        "full=0 hardware=0 fallback=0\n",
        GGML_GEMMINI_TEST_IM2P_DIM,
        static_cast<unsigned long long>(counters.rmd_terminal_event),
        static_cast<unsigned long long>(counters.authorize_success_event));
  }
  return ok;
}

bool run_exsia_start_failure() {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
  setenv("GEMMINI_RMD_BACKEND", "CPU", 1);
  GraphCase test_case;
  if (!check(test_case.initialize(), "initialize start-failure ExSIA graph")) {
    return false;
  }
  test_reset();
  test_inject_failure(TestFailure::execute);
  const ggml_status status =
      ggml_backend_graph_compute(test_case.backend, test_case.graph);
  const auto counters = test_counters();
  const bool ok =
      check(status != GGML_STATUS_SUCCESS,
            "ExSIA start failure reaches graph status") &&
      check(test_production_failed() &&
                counters.production_error == Error::execution_failure,
            "ExSIA start failure returns a typed execution failure") &&
      check(all_sentinel(test_case.read_output()),
            "ExSIA start failure preserves destination sentinel") &&
      check(
          counters.pipeline == 1 && counters.full == 0 &&
              counters.stripe == 0 && counters.accepted_stripes == 0 &&
              counters.fence == 0 && counters.rmd_calls == 0 &&
              counters.authorize == 0 && counters.commit == 0,
          "ExSIA start failure precedes publication, fence, RMD, and commit") &&
      check(counters.live_runs == 0 && counters.hardware == 0 &&
                counters.fallback == 0,
            "ExSIA start failure leaves no workers or fallback dispatch");
  if (ok) {
    std::printf(
        "failure=execute pipeline=1 stripes=0 fence=0 rmd=0 authorize=0 "
        "commit=0 live_runs=0 hardware=0 fallback=0\n");
  }
  return ok;
}

bool run_exsia_boundary_failure(
    ggml::gemmini::im2p_adapter::TestFailure failure) {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
  setenv("GEMMINI_RMD_BACKEND", "CPU", 1);
  GraphCase test_case;
  if (!check(test_case.initialize(),
             "initialize boundary-failure ExSIA graph")) {
    return false;
  }
  test_reset();
  test_inject_failure(failure);
  const ggml_status status =
      ggml_backend_graph_compute(test_case.backend, test_case.graph);
  const auto counters = test_counters();
  const bool quantization = failure == TestFailure::quantization;
  const bool progress = failure == TestFailure::progress;
  const bool poll = failure == TestFailure::poll;
  const char *failure_name = quantization ? "quantization"
                             : progress   ? "progress"
                                          : "poll";
  const bool ok =
      check(status != GGML_STATUS_SUCCESS,
            "production boundary failure reaches graph status") &&
      check(test_production_failed() &&
                counters.production_error == Error::execution_failure,
            "production boundary returns a typed execution failure") &&
      check(all_sentinel(test_case.read_output()),
            "production boundary failure preserves destination sentinel") &&
      check(counters.pipeline == 1 && counters.full == 0 &&
                counters.fence == 1 && counters.rmd_calls == 0 &&
                counters.authorize == 1 && counters.commit == 0,
            "production boundary failure fences and rejects without RMD or "
            "commit") &&
      check(counters.quantization_failures == (quantization ? 1U : 0U) &&
                counters.progress_failures == (progress ? 1U : 0U) &&
                counters.poll_failures == (poll ? 1U : 0U),
            "one distinct production seam injects the selected failure") &&
      check(counters.hardware == 0 && counters.fallback == 0 &&
                counters.live_runs == 0,
            "production boundary failure joins workers without fallback");
  if (ok) {
    std::printf("failure=%s error=execution_failure sentinel=preserved full=0 "
                "hardware=0 fallback=0 live_runs=0\n",
                failure_name);
  }
  return ok;
}

bool run_exsia_staged_failure(
    ggml::gemmini::im2p_adapter::TestFailure failure) {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
  setenv("GEMMINI_RMD_BACKEND", "CPU", 1);
  GraphCase test_case;
  if (!check(test_case.initialize(), "initialize staged-failure ExSIA graph")) {
    return false;
  }
  test_reset();
  test_inject_failure(failure);
  const ggml_status status =
      ggml_backend_graph_compute(test_case.backend, test_case.graph);
  const auto counters = test_counters();
  const bool fence_failure = failure == TestFailure::fence;
  const bool ok =
      check(status != GGML_STATUS_SUCCESS,
            "staged ExSIA failure reaches graph status") &&
      check(test_production_failed() &&
                counters.production_error == Error::execution_failure,
            "staged ExSIA failure returns a typed execution failure") &&
      check(all_sentinel(test_case.read_output()),
            "fence or RMD failure preserves destination sentinel") &&
      check(counters.pipeline == 1 && counters.full == 0 && counters.fence == 1,
            "failure path executes one pipeline and one fence") &&
      check(counters.stripe == 3 && counters.accepted_stripes == 3,
            "failure path publishes all three stripes") &&
      check(counters.rmd_calls == (fence_failure ? 0U : 3U),
            "RMD runs only after a successful fence") &&
      check(counters.authorize == 1 && counters.commit == 0,
            "failed transaction is explicitly rejected and never committed") &&
      check(counters.hardware == 0 && counters.fallback == 0 &&
                counters.live_runs == 0,
            "failed transaction has no fallback and joins every worker");
  if (ok) {
    std::printf("failure=%s error=execution_failure sentinel=preserved full=0 "
                "hardware=0 fallback=0 live_runs=0\n",
                fence_failure ? "fence" : "rmd");
  }
  return ok;
}

bool run_exsia_blocked_submit_failure() {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
  setenv("GEMMINI_RMD_BACKEND", "CPU", 1);
  GraphCase test_case;
  if (!check(test_case.initialize(),
             "initialize blocked-producer ExSIA graph")) {
    return false;
  }
  test_reset();
  test_inject_failure(TestFailure::blocked_submit);
  std::atomic<int> graph_status{-1};
  std::thread compute([&] {
    graph_status.store(static_cast<int>(ggml_backend_graph_compute(
                           test_case.backend, test_case.graph)),
                       std::memory_order_release);
  });
  const bool blocked = test_wait_for_blocked_producer();
  test_release_blocked_producer_with_error();
  compute.join();

  const auto counters = test_counters();
  return check(blocked, "third producer blocks on the capacity-two frontend") &&
         check(graph_status.load(std::memory_order_acquire) !=
                   GGML_STATUS_SUCCESS,
               "blocked-producer failure reaches graph status") &&
         check(all_sentinel(test_case.read_output()),
               "blocked-producer failure preserves destination sentinel") &&
         check(counters.pipeline == 1 && counters.fence == 1,
               "blocked failure still fences its sole pipeline run") &&
         check(counters.accepted_stripes == 2 &&
                   counters.max_outstanding == 2 &&
                   counters.blocked_producers == 1,
               "capacity-two backpressure blocks exactly the third producer") &&
         check(counters.blocked_submit_saw_execution_failure &&
                   counters.fence_saw_execution_failure,
               "blocked producer and fence observe the same sticky execution "
               "error") &&
         check(counters.rmd_calls == 0 && counters.authorize == 1 &&
                   counters.commit == 0,
               "blocked failure runs no RMD and commits no output") &&
         check(counters.hardware == 0 && counters.fallback == 0 &&
                   counters.live_runs == 0,
               "blocked failure has no fallback and joins every worker") &&
         check_three_stripe_trace(counters);
}

bool run_exsia_prestart_rejection(bool include_cpu) {
  using namespace ggml::gemmini::im2p_adapter;
  bool ok = true;
  for (const char *backend : {"CPU", "WS"}) {
    if (!include_cpu && std::string_view(backend) == "CPU") {
      continue;
    }
    setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
    setenv("GEMMINI_RMD_BACKEND", backend, 1);
    GraphCase test_case;
    if (!check(test_case.initialize(),
               "initialize pre-start rejection graph")) {
      return false;
    }
    test_reset();
    const ggml_status status =
        ggml_backend_graph_compute(test_case.backend, test_case.graph);
    const auto counters = test_counters();
    ok = check(status != GGML_STATUS_SUCCESS,
               "unsupported ExSIA route reaches failed graph status") &&
         ok;
    ok = check(all_sentinel(test_case.read_output()),
               "pre-start rejection preserves destination sentinel") &&
         ok;
    ok = check(counters.full == 0 && counters.pipeline == 0 &&
                   counters.fence == 0 && counters.stripe == 0 &&
                   counters.rmd_calls == 0 && counters.authorize == 0 &&
                   counters.commit == 0 && counters.hardware == 0 &&
                   counters.fallback == 0 && counters.live_runs == 0,
               "unsupported ExSIA route rejects before run, worker, or "
               "fallback") &&
         ok;
  }
  return ok;
}
#endif

bool run_success_mode(const char *mode) {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", mode, 1);
  GraphCase test_case;
  if (!check(test_case.initialize(), "initialize real Gemmini graph")) {
    return false;
  }
  std::vector<float> expected;
  if (!check(scalar_oracle(test_case.activations, test_case.weights, expected),
             "build scalar quantized oracle")) {
    return false;
  }

  test_reset();
  const ggml_status status =
      ggml_backend_graph_compute(test_case.backend, test_case.graph);
  const auto counters = test_counters();
  const auto actual = test_case.read_output();
  bool ok =
      check(status == GGML_STATUS_SUCCESS,
            "production graph compute succeeds") &&
      check(counters.full == 1,
            "production dispatch executes one IM2P full run") &&
      check(counters.fence == 1, "production dispatch fences once") &&
      check(counters.stripe == 0, "production dispatch submits no stripes") &&
      check(counters.hardware == 0,
            "production dispatch enters no hardware path");
  for (size_t index = 0; index < actual.size(); ++index) {
    ok = check(std::isfinite(actual[index]) &&
                   std::fabs(actual[index] - expected[index]) < 1e-4f,
               "production output matches scalar quantized oracle") &&
         ok;
  }
  if (ok) {
    std::printf("route=%s mode=%s bits=%d dim=%d full=1 stripe=0 hardware=0\n",
                compiled_route(), mode, GGML_GEMMINI_ACTIVATION_BITS,
                GGML_GEMMINI_TEST_IM2P_DIM);
  }
  return ok;
}

bool run_malformed_contract() {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
  GraphCase test_case;
  if (!check(test_case.initialize(), "initialize malformed-contract graph")) {
    return false;
  }
  test_reset();
  test_inject_failure(TestFailure::malformed_contract);
  const ggml_status status =
      ggml_backend_graph_compute(test_case.backend, test_case.graph);
  const auto counters = test_counters();
  return check(status != GGML_STATUS_SUCCESS,
               "malformed native contract reaches graph status") &&
         check(all_sentinel(test_case.read_output()),
               "malformed contract preserves destination sentinel") &&
         check(counters.full == 1 && counters.fence == 0 &&
                   counters.stripe == 0 && counters.hardware == 0,
               "malformed contract rejects before fence or fallback");
}

bool run_fence_failure() {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
  GraphCase test_case;
  if (!check(test_case.initialize(), "initialize failure graph")) {
    return false;
  }
  test_reset();
  test_inject_failure(TestFailure::fence);
  const ggml_status status =
      ggml_backend_graph_compute(test_case.backend, test_case.graph);
  const auto counters = test_counters();
  return check(status != GGML_STATUS_SUCCESS,
               "fence failure reaches graph status") &&
         check(all_sentinel(test_case.read_output()),
               "fence failure preserves destination sentinel") &&
         check(counters.full == 1 && counters.fence == 1 &&
                   counters.stripe == 0 && counters.hardware == 0,
               "fence failure has no stripe, hardware, or fallback dispatch");
}

bool run_rmd_rejection() {
  using namespace ggml::gemmini::im2p_adapter;
  GraphCase test_case;
  if (!check(test_case.initialize(), "initialize RMD rejection graph")) {
    return false;
  }
  test_reset();
  const ggml_status status =
      ggml_backend_graph_compute(test_case.backend, test_case.graph);
  const auto counters = test_counters();
  return check(status != GGML_STATUS_SUCCESS,
               "non-ExSIA RMD reaches failed graph status") &&
         check(all_sentinel(test_case.read_output()),
               "RMD rejection preserves destination sentinel") &&
         check(counters.full == 0 && counters.fence == 0 &&
                   counters.stripe == 0 && counters.hardware == 0,
               "RMD rejects before execute or fallback");
}

bool run_invalid_mode_child() {
  setenv("GEMMINI_MATMUL_MODE", "garbage", 1);
  GraphCase test_case;
  if (!test_case.initialize()) {
    return false;
  }
  return ggml_backend_graph_compute(test_case.backend, test_case.graph) ==
         GGML_STATUS_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--invalid-mode-child") {
    return run_invalid_mode_child() ? 0 : 1;
  }
  if (!run_graph_overhead_regression() || !run_exsia_shift_regression()) {
    return 1;
  }
  if (argc == 3 && std::string_view(argv[1]) == "--case") {
    const std::string_view selected(argv[2]);
#if GGML_GEMMINI_ACTIVATION_QUANT == 0 && GGML_GEMMINI_ACTIVATION_BITS == 8 && \
    GGML_GEMMINI_ENABLE_RMD
    bool selected_ok = false;
    if (selected == "execute") {
      selected_ok = run_exsia_start_failure();
    } else if (selected == "quantization") {
      selected_ok = run_exsia_boundary_failure(
          ggml::gemmini::im2p_adapter::TestFailure::quantization);
    } else if (selected == "progress") {
      selected_ok = run_exsia_boundary_failure(
          ggml::gemmini::im2p_adapter::TestFailure::progress);
    } else if (selected == "poll") {
      selected_ok = run_exsia_boundary_failure(
          ggml::gemmini::im2p_adapter::TestFailure::poll);
    } else if (selected == "fence") {
      selected_ok = run_exsia_staged_failure(
          ggml::gemmini::im2p_adapter::TestFailure::fence);
    } else if (selected == "rmd") {
      selected_ok = run_exsia_staged_failure(
          ggml::gemmini::im2p_adapter::TestFailure::rmd);
    } else if (selected == "blocked-producer-fence-failure") {
      selected_ok = run_exsia_blocked_submit_failure();
    } else {
      std::fprintf(stderr, "unsupported test case: %s\n", argv[2]);
      return 2;
    }
    unsetenv("GEMMINI_MATMUL_MODE");
    unsetenv("GEMMINI_RMD_BACKEND");
    unsetenv("GEMMINI_STRIPE_JOB_CAPACITY");
    return selected_ok ? 0 : 1;
#else
    std::fprintf(stderr, "unsupported test case: %s\n", argv[2]);
    return 2;
#endif
  }

  std::string_view requested_route = compiled_route();
  int requested_bits = GGML_GEMMINI_ACTIVATION_BITS;
  int requested_dim = GGML_GEMMINI_TEST_IM2P_DIM;
  int requested_stripes = 3;
  int requested_queue_capacity = 2;
  std::string_view requested_rmd = "cpu_direct";
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--route" && index + 1 < argc) {
      requested_route = argv[++index];
    } else if (argument == "--bits" && index + 1 < argc) {
      requested_bits = std::atoi(argv[++index]);
    } else if (argument == "--dim" && index + 1 < argc) {
      requested_dim = std::atoi(argv[++index]);
    } else if (argument == "--stripes" && index + 1 < argc) {
      requested_stripes = std::atoi(argv[++index]);
    } else if (argument == "--queue-capacity" && index + 1 < argc) {
      requested_queue_capacity = std::atoi(argv[++index]);
    } else if (argument == "--rmd" && index + 1 < argc) {
      requested_rmd = argv[++index];
    } else {
      std::fprintf(stderr, "invalid argument: %s\n", argv[index]);
      return 2;
    }
  }
  if (requested_route != compiled_route() ||
      requested_bits != GGML_GEMMINI_ACTIVATION_BITS ||
      requested_dim != GGML_GEMMINI_TEST_IM2P_DIM ||
      (compiled_exsia &&
       (requested_stripes != 3 || requested_queue_capacity != 2 ||
        requested_rmd != "cpu_direct"))) {
    std::fprintf(stderr,
                 "requested route/build contract does not match (%s/%d/%d)\n",
                 compiled_route(), GGML_GEMMINI_ACTIVATION_BITS,
                 GGML_GEMMINI_TEST_IM2P_DIM);
    return 2;
  }

#if GGML_GEMMINI_ACTIVATION_QUANT == 0
  bool ok = true;
#if GGML_GEMMINI_ACTIVATION_BITS == 8 && GGML_GEMMINI_ENABLE_RMD
  ok = run_exsia_success() && ok;
  ok = run_exsia_start_failure() && ok;
  ok = run_exsia_boundary_failure(
           ggml::gemmini::im2p_adapter::TestFailure::quantization) &&
       ok;
  ok = run_exsia_boundary_failure(
           ggml::gemmini::im2p_adapter::TestFailure::progress) &&
       ok;
  ok = run_exsia_boundary_failure(
           ggml::gemmini::im2p_adapter::TestFailure::poll) &&
       ok;
  ok = run_exsia_blocked_submit_failure() && ok;
  ok = run_exsia_staged_failure(
           ggml::gemmini::im2p_adapter::TestFailure::fence) &&
       ok;
  ok =
      run_exsia_staged_failure(ggml::gemmini::im2p_adapter::TestFailure::rmd) &&
      ok;
  ok = run_exsia_prestart_rejection(false) && ok;
#else
  ok = run_exsia_prestart_rejection(true) && ok;
#endif
  unsetenv("GEMMINI_MATMUL_MODE");
  unsetenv("GEMMINI_RMD_BACKEND");
  unsetenv("GEMMINI_STRIPE_JOB_CAPACITY");
  return ok ? 0 : 1;
#else
#if GGML_GEMMINI_ENABLE_RMD
  return run_rmd_rejection() ? 0 : 1;
#endif

  bool ok = true;
  for (const char *mode : {"FULL", "STRIPE_SEQUENTIAL", "STRIPE_PIPELINE"}) {
    ok = run_success_mode(mode) && ok;
  }
  ok = run_malformed_contract() && ok;
  ok = run_fence_failure() && ok;

  std::string command = "ulimit -c 0; \"";
  command += argv[0];
  command += "\" --invalid-mode-child >/dev/null 2>&1";
  ok = check(std::system(command.c_str()) != 0,
             "garbage mode is rejected through production dispatch") &&
       ok;
  unsetenv("GEMMINI_MATMUL_MODE");
  return ok ? 0 : 1;
#endif
}
