#include <ggml-backend.h>
#include <ggml-gemmini.h>
#include <ggml-quants.h>
#include <ggml.h>

#include "ggml-gemmini-args.h"
#include "ggml-gemmini-geometry.hpp"
#include "ggml-gemmini-im2p.hpp"
#include "im2p_gemmini_frontend.hpp"
#include "quants/act/exsia/exsia.hpp"
#include "quants/act/exsia/exsia_shift.hpp"
#include "quants/act/quantize.hpp"
#include <gemmini/cycle_reader.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>
#include <unistd.h>

namespace {

#if GGML_GEMMINI_ACTIVATION_QUANT == 0
constexpr bool compiled_exsia = true;
constexpr int64_t K = 64;
constexpr int64_t I = 16 * GGML_GEMMINI_TEST_IM2P_DIM;
#else
constexpr bool compiled_exsia = false;
constexpr int64_t K = 32;
constexpr int64_t I = 3;
#endif
constexpr int64_t J = 2;
constexpr size_t graph_publications = 1;
constexpr float sentinel = 12345.0f;
const char *routing_program = nullptr;

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

const char *compiled_weight_route() {
#if GGML_GEMMINI_WEIGHT_BITS == 4
  return "q4_h1";
#elif GGML_GEMMINI_WEIGHT_BITS == 16
  return "q16_h1";
#else
  return compiled_exsia ? "q8_0" : "q8_channel";
#endif
}

const char *compiled_mismatch_support_result() {
  return GGML_GEMMINI_WEIGHT_BITS == 4 || GGML_GEMMINI_WEIGHT_BITS == 16
             ? "no"
             : "n/a";
}

bool run_matched_weight_gate_contract() {
  using ggml::gemmini::im2p_adapter::gate_route;

  const auto q4 = gate_route(false, 4, false, false, 4);
  const auto q16 = gate_route(false, 16, false, false, 16);
  const auto legacy_a4_q8 = gate_route(false, 4, false, false, 8);
  const auto legacy_a16_q8 = gate_route(false, 16, false, false, 8);
  const auto unsupported_a8_q4 = gate_route(false, 8, false, false, 4);
  const auto unsupported_a8_q16 = gate_route(false, 8, false, false, 16);
  const auto exsia_q4 = gate_route(true, 4, true, true, 4);
  const auto exsia_q16 = gate_route(true, 16, true, true, 16);

  return check(q4.ok(), "non-RMD A4/Q4 matched route is accepted") &&
         check(q16.ok(), "non-RMD A16/Q16 matched route is accepted") &&
         check(!legacy_a4_q8.ok(), "non-RMD A4/Q8 is rejected as mixed width") &&
         check(!legacy_a16_q8.ok(), "non-RMD A16/Q8 is rejected as mixed width") &&
         check(!unsupported_a8_q4.ok(), "A8/Q4 has no native artifact") &&
         check(!unsupported_a8_q16.ok(), "A8/Q16 has no native artifact") &&
         check(exsia_q4.ok(), "ExSIA A4/Q4 matched route is accepted") &&
         check(exsia_q16.ok(), "ExSIA A16/Q16 matched route is accepted");
}

std::vector<float> make_activations(int64_t rows = I) {
  std::vector<float> values(rows * K);
  for (int64_t i = 0; i < rows; ++i) {
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
#if GGML_GEMMINI_WEIGHT_BITS == 4
  constexpr size_t blocks_per_row = K / QK4_0;
  std::vector<uint8_t> encoded(J * blocks_per_row * sizeof(block_q4_h1));
  auto *blocks = reinterpret_cast<block_q4_h1 *>(encoded.data());
  for (int64_t j = 0; j < J; ++j) {
    for (size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
      block_q4_h1 &block = blocks[j * blocks_per_row + block_index];
      block.s_rf = 0.125f * static_cast<float>(j + 1);
      block.c_b = static_cast<uint8_t>(block_index + 1);
      block.R = 1;
      for (size_t lane = 0; lane < QK4_0; ++lane) {
        const int8_t code = static_cast<int8_t>(
            (5 * j + static_cast<int>(block_index) + static_cast<int>(lane)) % 16 - 8);
        const uint8_t nibble = static_cast<uint8_t>(code + 8);
        uint8_t &byte = block.qs[lane % (QK4_0 / 2)];
        byte = lane < QK4_0 / 2
                   ? static_cast<uint8_t>((byte & 0xf0) | nibble)
                   : static_cast<uint8_t>((byte & 0x0f) | (nibble << 4));
      }
    }
  }
  return encoded;
#elif GGML_GEMMINI_WEIGHT_BITS == 16
  constexpr size_t blocks_per_row = K / QK16_0;
  std::vector<uint8_t> encoded(J * blocks_per_row * sizeof(block_q16_h1));
  auto *blocks = reinterpret_cast<block_q16_h1 *>(encoded.data());
  for (int64_t j = 0; j < J; ++j) {
    for (size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
      block_q16_h1 &block = blocks[j * blocks_per_row + block_index];
      block.s_rf = 0.125f * static_cast<float>(j + 1);
      block.c_b = static_cast<uint8_t>(block_index + 1);
      block.R = 1;
      for (size_t lane = 0; lane < QK16_0; ++lane)
        block.qs[lane] = static_cast<int16_t>(
            (19 * j + 7 * static_cast<int>(lane) + static_cast<int>(block_index)) % 47 - 23);
    }
  }
  return encoded;
#elif GGML_GEMMINI_ACTIVATION_QUANT == 0
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
  const int64_t rows = static_cast<int64_t>(activations.size()) / K;
  ggml_init_params params = {
      ggml_tensor_overhead() * 4 +
          static_cast<size_t>(rows * K) * sizeof(float) + 1024,
      nullptr,
      false,
  };
  ggml_context *context = ggml_init(params);
  if (context == nullptr) {
    return false;
  }
  ggml_tensor *activation =
      ggml_new_tensor_2d(context, GGML_TYPE_F32, K, rows);
  std::memcpy(activation->data, activations.data(),
              activations.size() * sizeof(float));

  ggml_gemmini_args_t args{};
  args.I = rows;
  args.J = J;
  args.K = K;
  args.sA = K;
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
  args.tile_I = 1;
  args.tile_J = 1;
  args.tile_K = 1;
  args.activation_rows_per_stripe = GGML_GEMMINI_TEST_IM2P_DIM;
  args.residual_route = ggml::gemmini::residual::ResidualRoute::cpu_direct;
#endif
  const bool allocated =
      args.A.allocate(rows, K, GGML_GEMMINI_ACTIVATION_BITS);
  const bool quantized =
      allocated && ggml::gemmini::quants::quantize_activation(activation, args);
  std::vector<float> decoded(rows * K);
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
        if (row >= static_cast<size_t>(rows) ||
            event.original_k >= static_cast<size_t>(K)) {
          ggml_free(context);
          return false;
        }
        residuals[row * K + event.original_k] = event.residual;
      }
    }
  }
  const size_t stripe_count =
      (static_cast<size_t>(rows) + GGML_GEMMINI_TEST_IM2P_DIM - 1) /
      GGML_GEMMINI_TEST_IM2P_DIM;
  bool decoded_ok = quantized && exsia_meta != nullptr &&
                    residual_event_count != 0 &&
                    exsia_meta->theta.size() == stripe_count;
  if (decoded_ok) {
    for (int64_t i = 0; i < rows; ++i) {
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
                       decoded.data(), K, 1, rows, K, args);
#if GGML_GEMMINI_ACTIVATION_QUANT == 3
  const auto *block_meta = std::get_if<ggml::gemmini::quants::act::block::Meta>(
      &args.act_quant.storage());
  const bool valid_block_meta =
      block_meta != nullptr &&
      block_meta->scales.size() == static_cast<size_t>(rows) &&
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

  expected.assign(rows * J, 0.0f);
#if GGML_GEMMINI_WEIGHT_BITS == 4
  constexpr size_t blocks_per_row = K / QK4_0;
  const auto *blocks = reinterpret_cast<const block_q4_h1 *>(weights.data());
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      for (int64_t k = 0; k < K; ++k) {
        const block_q4_h1 &block =
            blocks[j * blocks_per_row + static_cast<size_t>(k) / QK4_0];
        const size_t lane = static_cast<size_t>(k) % QK4_0;
        const uint8_t byte = block.qs[lane % (QK4_0 / 2)];
        const int code = int(lane < QK4_0 / 2 ? byte & 0x0f : byte >> 4) - 8;
        const float factor = block.s_rf * (block.c_b + block.R);
        expected[i * J + j] += decoded[i * K + k] * code * factor;
      }
    }
  }
#elif GGML_GEMMINI_WEIGHT_BITS == 16
  constexpr size_t blocks_per_row = K / QK16_0;
  const auto *blocks = reinterpret_cast<const block_q16_h1 *>(weights.data());
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < J; ++j) {
      for (int64_t k = 0; k < K; ++k) {
        const block_q16_h1 &block =
            blocks[j * blocks_per_row + static_cast<size_t>(k) / QK16_0];
        const float factor = block.s_rf * (block.c_b + block.R);
        expected[i * J + j] += decoded[i * K + k] *
                               block.qs[static_cast<size_t>(k) % QK16_0] * factor;
      }
    }
  }
#elif GGML_GEMMINI_ACTIVATION_QUANT == 0
  constexpr size_t blocks_per_row = K / QK8_0;
  const auto *blocks = reinterpret_cast<const block_q8_0 *>(weights.data());
  for (int64_t i = 0; i < rows; ++i) {
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
  for (int64_t i = 0; i < rows; ++i) {
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
  explicit GraphCase(int64_t row_count = I)
      : rows(row_count), activations(make_activations(row_count)) {}

  int64_t rows = I;
  ggml_backend_t backend = nullptr;
  ggml_context *context = nullptr;
  ggml_backend_buffer_t buffer = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *output = nullptr;
  std::vector<float> activations;
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
#if GGML_GEMMINI_WEIGHT_BITS == 4
    ggml_tensor *weight = ggml_new_tensor_2d(context, GGML_TYPE_Q4_H1, K, J);
    ggml_tensor *mismatched_weight =
        ggml_new_tensor_2d(context, GGML_TYPE_Q16_H1, K, J);
#elif GGML_GEMMINI_WEIGHT_BITS == 16
    ggml_tensor *weight = ggml_new_tensor_2d(context, GGML_TYPE_Q16_H1, K, J);
    ggml_tensor *mismatched_weight =
        ggml_new_tensor_2d(context, GGML_TYPE_Q4_H1, K, J);
#elif GGML_GEMMINI_ACTIVATION_QUANT == 0
    ggml_tensor *weight = ggml_new_tensor_2d(context, GGML_TYPE_Q8_0, K, J);
#else
    ggml_tensor *weight =
        ggml_new_tensor_2d(context, GGML_TYPE_Q8_CHANNEL, K, J);
#endif
    ggml_tensor *activation =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, K, rows);
    output = ggml_mul_mat(context, weight, activation);
#if GGML_GEMMINI_WEIGHT_BITS == 4 || GGML_GEMMINI_WEIGHT_BITS == 16
    ggml_tensor *mismatched_output =
        ggml_mul_mat(context, mismatched_weight, activation);
#endif
    buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
      return false;
    }
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(activation, activations.data(), 0,
                            activations.size() * sizeof(float));
    std::vector<float> initial(rows * J, sentinel);
    ggml_backend_tensor_set(output, initial.data(), 0,
                            initial.size() * sizeof(float));
    graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    const bool selected_supported = ggml_backend_supports_op(backend, output);
#if GGML_GEMMINI_WEIGHT_BITS == 4 || GGML_GEMMINI_WEIGHT_BITS == 16
    const bool mismatched_rejected =
        !ggml_backend_supports_op(backend, mismatched_output);
    return selected_supported && mismatched_rejected;
#else
    return selected_supported;
#endif
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
    std::vector<float> values(rows * J);
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
using TestFailure = ggml::gemmini::im2p_adapter::TestFailure;

struct ExsiaPublicationTrace {
  const ggml::gemmini::quants::act::exsia::Meta *meta = nullptr;
  bool quantization_complete = false;
  bool callback_before_completion = true;
  bool committed_prefix_exact = true;
  bool next_theta_uncommitted = true;
  size_t nonzero_order_checks = 0;
  std::array<ggml::gemmini::quants::act::exsia::StripeReadyEvent, 4> events{};
  size_t event_count = 0;
};

bool trace_exsia_publication(
    void *opaque,
    const ggml::gemmini::quants::act::exsia::StripeReadyEvent &event) {
  auto &trace = *static_cast<ExsiaPublicationTrace *>(opaque);
  trace.callback_before_completion =
      trace.callback_before_completion && !trace.quantization_complete;
  if (trace.event_count >= trace.events.size()) {
    return false;
  }
  size_t committed = 0;
  for (const int16_t theta : trace.meta->theta) {
    committed += theta != std::numeric_limits<int16_t>::min();
  }
  trace.committed_prefix_exact =
      trace.committed_prefix_exact && committed == event.stripe_id + 1;
  if (event.stripe_id + 1 < trace.meta->theta.size()) {
    trace.next_theta_uncommitted =
        trace.next_theta_uncommitted &&
        trace.meta->resolve_stripe_theta(static_cast<int>(event.stripe_id + 1)) ==
            std::numeric_limits<int16_t>::min();
    ++trace.nonzero_order_checks;
  }
  trace.events[trace.event_count++] = event;
  return trace.committed_prefix_exact && trace.next_theta_uncommitted &&
         trace.meta->resolve_stripe_theta(static_cast<int>(event.stripe_id)) !=
             std::numeric_limits<int16_t>::min();
}

bool run_exsia_publication_boundary() {
  using namespace ggml::gemmini::quants::act::exsia;
  constexpr int64_t publication_rows = 2 * GGML_GEMMINI_TEST_IM2P_DIM + 1;
  ggml_init_params params{ggml_tensor_overhead() * 2 +
                               static_cast<size_t>(publication_rows * K) * sizeof(float) +
                               1024,
                           nullptr, false};
  ggml_context *context = ggml_init(params);
  if (!check(context != nullptr, "initialize ExSIA publication boundary input")) {
    return false;
  }
  ggml_tensor *activation =
      ggml_new_tensor_2d(context, GGML_TYPE_F32, K, publication_rows);
  const auto values = make_activations(publication_rows);
  std::memcpy(activation->data, values.data(), values.size() * sizeof(float));

  ggml_gemmini_args_t args{};
  args.I = publication_rows;
  args.J = J;
  args.K = K;
  args.sA = K;
  args.tile_I = 1;
  args.tile_J = 1;
  args.tile_K = 1;
  args.activation_rows_per_stripe = GGML_GEMMINI_TEST_IM2P_DIM;
  args.residual_route = ggml::gemmini::residual::ResidualRoute::cpu_direct;
  const bool allocated =
      args.A.allocate(publication_rows, K, GGML_GEMMINI_ACTIVATION_BITS);
  Meta meta{};
  ExsiaPublicationTrace trace{&meta};
  StripeReadySink sink{&trace, trace_exsia_publication};
  ExSIA exsia;
  exsia.set_execution_mode(ExSIAState::ExecutionMode::Sequential);
  ggml::gemmini::cycle::reset_read_count_for_test();
  const bool quantized = allocated && exsia.run(meta, activation, args, &sink);
  trace.quantization_complete = true;
  ggml_free(context);

#if GGML_GEMMINI_ENABLE_RMD
  const bool residual_handle_contract = std::all_of(
      trace.events.begin(), trace.events.begin() + trace.event_count,
      [](const StripeReadyEvent &event) {
        return event.rmd_packet || event.direct_residual;
      });
  const char *residual_handle_message =
      "residual sealing precedes each publication callback";
#else
  const bool residual_handle_contract = std::all_of(
      trace.events.begin(), trace.events.begin() + trace.event_count,
      [](const StripeReadyEvent &event) {
        return !event.rmd_packet && !event.direct_residual;
      });
  const char *residual_handle_message =
      "RMD-disabled publication carries no residual handles";
#endif

  return check(quantized, "three-stripe ExSIA quantization succeeds") &&
         check(trace.event_count == 3,
               "one callback is emitted for each sealed stripe") &&
         check(trace.callback_before_completion,
               "callback returns before whole ExSIA quantization completes") &&
         check(trace.committed_prefix_exact && trace.next_theta_uncommitted &&
                   trace.nonzero_order_checks == 2,
               "each callback observes only its committed theta prefix and an uncommitted next stripe") &&
         check(residual_handle_contract, residual_handle_message) &&
#if LOG_CYCLE
         check(trace.events[0].folding_commit_ns != 0 &&
                   trace.events[0].folding_commit_ns <= trace.events[1].folding_commit_ns &&
                   trace.events[1].folding_commit_ns <= trace.events[2].folding_commit_ns &&
                   trace.events[0].quantization_end >= trace.events[0].quantization_start &&
                   trace.events[1].quantization_end >= trace.events[1].quantization_start &&
                   trace.events[2].quantization_end >= trace.events[2].quantization_start &&
                   trace.events[2].quantization_end - trace.events[2].quantization_start > 0 &&
                   ggml::gemmini::cycle::read_count_for_test() != 0,
               "enabled per-stripe quantization intervals and folding commits are instrumented") &&
#else
         check(trace.events[0].folding_commit_ns == 0 &&
                   trace.events[1].folding_commit_ns == 0 &&
                   trace.events[2].folding_commit_ns == 0 &&
                   trace.events[0].quantization_start == 0 &&
                   trace.events[0].quantization_end == 0 &&
                   trace.events[1].quantization_start == 0 &&
                   trace.events[1].quantization_end == 0 &&
                   trace.events[2].quantization_start == 0 &&
                   trace.events[2].quantization_end == 0 &&
                   ggml::gemmini::cycle::read_count_for_test() == 0,
               "disabled quantization timing is deterministic and reads no timer") &&
#endif
         check(trace.events[0].slot == 0 && trace.events[1].slot == 1 &&
                   trace.events[2].slot == 0,
               "publication boundary retains the two-slot 0,1,0 cycle");
}

using namespace ggml::gemmini::im2p_adapter;

enum class LifecycleFamily { h0, h1, hp1 };

enum class LifecycleBackend { cpu_direct, compact_ws };

const char *lifecycle_family_name(LifecycleFamily family) {
  switch (family) {
  case LifecycleFamily::h0: return "H0";
  case LifecycleFamily::h1: return "H1";
  case LifecycleFamily::hp1: return "HP1";
  }
  return "unknown";
}

const char *lifecycle_backend_name(LifecycleBackend backend) {
  return backend == LifecycleBackend::cpu_direct ? "cpu_direct"
                                                  : "compact_ws_software";
}

struct LifecycleWeightStorage {
  std::vector<block_q4_h0> q4_h0;
  std::vector<block_q4_h1> q4_h1;
  std::vector<block_q4_hp1> q4_hp1;
  std::vector<block_q8_h1> q8_h1;
  std::vector<block_q8_hp1> q8_hp1;
  std::vector<block_q16_h0> q16_h0;
  std::vector<block_q16_h1> q16_h1;
  std::vector<block_q16_hp1> q16_hp1;
  size_t blocks_per_row = 0;

  LifecycleWeightStorage(size_t columns, size_t k)
      : q4_h0(columns * (k / 32)), q4_h1(columns * (k / 32)),
        q4_hp1(columns * (k / 32)), q8_h1(columns * (k / 32)),
        q8_hp1(columns * (k / 32)), q16_h0(columns * (k / 32)),
        q16_h1(columns * (k / 32)), q16_hp1(columns * (k / 32)),
        blocks_per_row(k / 32) {
    for (size_t index = 0; index < columns * blocks_per_row; ++index) {
      std::fill(std::begin(q4_h0[index].qs), std::end(q4_h0[index].qs),
                uint8_t{0x99});
      std::fill(std::begin(q4_h1[index].qs), std::end(q4_h1[index].qs),
                uint8_t{0x99});
      std::fill(std::begin(q4_hp1[index].qs), std::end(q4_hp1[index].qs),
                uint8_t{0x99});
      std::fill(std::begin(q8_h1[index].qs), std::end(q8_h1[index].qs),
                int8_t{1});
      std::fill(std::begin(q8_hp1[index].qs), std::end(q8_hp1[index].qs),
                int8_t{1});
      std::fill(std::begin(q16_h0[index].qs), std::end(q16_h0[index].qs),
                int16_t{1});
      std::fill(std::begin(q16_h1[index].qs), std::end(q16_h1[index].qs),
                int16_t{1});
      std::fill(std::begin(q16_hp1[index].qs), std::end(q16_hp1[index].qs),
                int16_t{1});
      q4_h0[index].d = q16_h0[index].d = ggml_fp32_to_fp16(0.25f);
      q4_h1[index].s_rf = q8_h1[index].s_rf = q16_h1[index].s_rf = 0.25f;
      q4_h1[index].c_b = q8_h1[index].c_b = q16_h1[index].c_b = 1;
      q4_h1[index].R = q8_h1[index].R = q16_h1[index].R = 1;
      q4_hp1[index].channel_scale = q8_hp1[index].channel_scale =
          q16_hp1[index].channel_scale = 0.25f;
      q4_hp1[index].m = q8_hp1[index].m = q16_hp1[index].m = 0;
    }
  }

  bool configure(ggml_gemmini_args_t &args, LifecycleFamily family) {
    using Format = ggml_gemmini_args_t::im2p_weight_format_t;
    const size_t count = args.J * blocks_per_row;
    args.native_block_count = count;
    args.native_blocks_per_row = blocks_per_row;
#if GGML_GEMMINI_WEIGHT_BITS == 4
    if (family == LifecycleFamily::h0) {
      args.weight_format = Format::q4_h0;
      args.q4_h0_blocks = q4_h0.data();
      args.native_weight_bytes = q4_h0.size() * sizeof(block_q4_h0);
    } else if (family == LifecycleFamily::h1) {
      args.weight_format = Format::q4_h1;
      args.q4_h1_blocks = q4_h1.data();
      args.native_weight_bytes = q4_h1.size() * sizeof(block_q4_h1);
    } else {
      args.weight_format = Format::q4_hp1;
      args.q4_hp1_blocks = q4_hp1.data();
      args.native_weight_bytes = q4_hp1.size() * sizeof(block_q4_hp1);
    }
#elif GGML_GEMMINI_WEIGHT_BITS == 8
    if (family == LifecycleFamily::h0) {
      return false;
    } else if (family == LifecycleFamily::h1) {
      args.weight_format = Format::q8_h1;
      args.q8_h1_blocks = q8_h1.data();
      args.q8_h1_block_count = q8_h1.size();
      args.q8_h1_rows = args.J;
      args.blocks_per_row = blocks_per_row;
      args.native_weight_bytes = q8_h1.size() * sizeof(block_q8_h1);
    } else {
      args.weight_format = Format::q8_hp1;
      args.q8_hp1_blocks = q8_hp1.data();
      args.q8_hp1_block_count = q8_hp1.size();
      args.q8_hp1_blocks_per_row = blocks_per_row;
      args.native_weight_bytes = q8_hp1.size() * sizeof(block_q8_hp1);
    }
#else
    if (family == LifecycleFamily::h0) {
      args.weight_format = Format::q16_h0;
      args.q16_h0_blocks = q16_h0.data();
      args.native_weight_bytes = q16_h0.size() * sizeof(block_q16_h0);
    } else if (family == LifecycleFamily::h1) {
      args.weight_format = Format::q16_h1;
      args.q16_h1_blocks = q16_h1.data();
      args.native_weight_bytes = q16_h1.size() * sizeof(block_q16_h1);
    } else {
      args.weight_format = Format::q16_hp1;
      args.q16_hp1_blocks = q16_hp1.data();
      args.native_weight_bytes = q16_hp1.size() * sizeof(block_q16_hp1);
    }
#endif
    return true;
  }
};

struct IntegratedLifecycleResult {
  bool ok = false;
  Completion completion{};
  TestCounters counters{};
  std::vector<float> output;
  bool semantic_layer_observed = false;
};

struct RuntimeArgsObservation {
  std::string expected;
  std::array<size_t, 4> counts{};
  bool exact = true;
};

void observe_runtime_args(TestRuntimeArgsSite site, const char *layer,
                          void *opaque) {
  auto &observation = *static_cast<RuntimeArgsObservation *>(opaque);
  const size_t index = static_cast<size_t>(site);
  if (index < observation.counts.size()) {
    ++observation.counts[index];
  }
  observation.exact = observation.exact && layer != nullptr &&
                      observation.expected == layer;
}

IntegratedLifecycleResult run_integrated_exsia_lifecycle(
    size_t rows, size_t tile_i, LifecycleFamily family,
    LifecycleBackend backend, PublicMode mode) {
  using namespace ggml::gemmini::quants::act::exsia;
  IntegratedLifecycleResult result{};
  const auto values = make_activations(static_cast<int64_t>(rows));
  ggml_init_params params{ggml_tensor_overhead() * 2 +
                              values.size() * sizeof(float) + 1024,
                          nullptr, false};
  ggml_context *context = ggml_init(params);
  if (context == nullptr) return result;
  ggml_tensor *activation = ggml_new_tensor_2d(context, GGML_TYPE_F32, K, rows);
  std::memcpy(activation->data, values.data(), values.size() * sizeof(float));

  ggml_gemmini_args_t args{};
  const std::string semantic_layer =
      "blk.15.mlp.down_proj.im2p-lifetime-sentinel-beyond-sso";
  {
    std::string source = semantic_layer;
    args.matmul_layer = source;
  }
  args.I = rows;
  args.J = J;
  args.K = K;
  args.sA = K;
  args.tiled_matmul_type = static_cast<tiled_matmul_type_t>(1);
  args.tile_I = tile_i;
  args.tile_J = 1;
  args.tile_K = 1;
  args.activation_rows_per_stripe = tile_i * GGML_GEMMINI_TEST_IM2P_DIM;
  args.residual_route = backend == LifecycleBackend::cpu_direct
                            ? ggml::gemmini::residual::ResidualRoute::cpu_direct
                            : ggml::gemmini::residual::ResidualRoute::ws_packet;
  result.output.assign(rows * J, sentinel);
  args.f_out = result.output.data();
  args.stride_f_out = J;
  args.col_stride_f_out = 1;
  LifecycleWeightStorage weights(J, K);
  if (!weights.configure(args, family) ||
      !args.A.allocate(rows, K, GGML_GEMMINI_ACTIVATION_BITS)) {
    ggml_free(context);
    return result;
  }
  auto &meta = args.act_quant.storage().emplace<Meta>();
  ExSIA exsia;
  exsia.set_execution_mode(ExSIAState::ExecutionMode::Sequential);
  RuntimeArgsObservation observation{semantic_layer};
  test_reset();
  test_set_runtime_args_observer(observe_runtime_args, &observation);
  if (mode == PublicMode::full) {
    auto started = start_exsia_full_execution(args);
    if (!started.result.ok() || !started.execution ||
        !started.execution->install_sink().ok()) {
      test_set_runtime_args_observer(nullptr, nullptr);
      ggml_free(context);
      return result;
    }
    const bool quantized = exsia.run(meta, activation, args,
                                    args.exsia_stripe_ready_sink);
    result.completion = started.execution->finish(quantized);
  } else {
    auto started = start_exsia_stripe_pipeline(args);
    if (!started.result.ok() || !started.pipeline ||
        !started.pipeline->install_sink().ok()) {
      test_set_runtime_args_observer(nullptr, nullptr);
      ggml_free(context);
      return result;
    }
    const bool quantized = exsia.run(meta, activation, args,
                                    args.exsia_stripe_ready_sink);
    result.completion = started.pipeline->finish(quantized);
  }
  test_set_runtime_args_observer(nullptr, nullptr);
  result.counters = test_counters();
  result.ok = result.completion.result.ok();
  const size_t expected_site = static_cast<size_t>(
      mode == PublicMode::full ? TestRuntimeArgsSite::exsia_full_before_execute
                               : TestRuntimeArgsSite::exsia_pipeline_before_execute);
  const size_t observations = std::accumulate(
      observation.counts.begin(), observation.counts.end(), size_t{0});
  result.semantic_layer_observed = args.matmul_layer == semantic_layer &&
      observation.exact && observation.counts[expected_site] == 1 &&
      observations == 1;
  if (!result.ok) {
    std::fprintf(stderr,
                 "integrated lifecycle failed family=%s backend=%s mode=%s: %s\n",
                 lifecycle_family_name(family), lifecycle_backend_name(backend),
                 mode == PublicMode::full ? "FULL" : "STRIPE_PIPELINE",
                 result.completion.result.message);
  }
  ggml_free(context);
  return result;
}

#if !GGML_GEMMINI_ENABLE_RMD
bool run_rmd_disabled_adapter_dense_only() {
  const size_t rows = GGML_GEMMINI_TEST_IM2P_DIM + 1;
  const auto full = run_integrated_exsia_lifecycle(
      rows, 1, LifecycleFamily::hp1, LifecycleBackend::cpu_direct,
      PublicMode::full);
  const auto pipeline = run_integrated_exsia_lifecycle(
      rows, 1, LifecycleFamily::hp1, LifecycleBackend::cpu_direct,
      PublicMode::stripe_pipeline);
  const auto no_residual_work = [](const IntegratedLifecycleResult &result) {
    return result.counters.residual_executions == 0 &&
        result.counters.compositions == 0 &&
        result.counters.rmd_calls == 0 &&
        result.counters.rmd_events == 0 &&
        result.counters.rmd_packets == 0;
  };
  return check(full.ok && pipeline.ok,
               "RMD-disabled FULL/PIPELINE adapter executions succeed") &&
      check(full.output == pipeline.output,
            "RMD-disabled FULL/PIPELINE outputs remain dense-only and equal") &&
      check(no_residual_work(full) && no_residual_work(pipeline),
            "RMD-disabled adapters report zero residual work") &&
      check(full.counters.commit == 1 && pipeline.counters.commit == 1,
            "RMD-disabled adapters commit dense output exactly once");
}
#endif

bool run_simple_runtime_args_observer_contract() {
  const std::string semantic_layer =
      "blk.15.mlp.down_proj.simple-runtime-copy-beyond-sso";
  auto observe_route = [&](bool pipeline) {
    ggml_gemmini_args_t args{};
    {
      std::string source = semantic_layer;
      args.matmul_layer = source;
    }
    args.I = pipeline ? 32 : 1;
    args.J = 1;
    args.K = pipeline ? 32 : 1;
    args.tile_I = 1;
    args.tile_J = 1;
    args.tile_K = 1;
    args.activation_rows_per_stripe =
        pipeline ? GGML_GEMMINI_TEST_IM2P_DIM : 1;
    std::vector<float> output(args.I, sentinel);
    args.f_out = output.data();
    args.stride_f_out = 1;
    args.col_stride_f_out = 1;
    if (!args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS)) {
      return false;
    }
    RuntimeArgsObservation observation{semantic_layer};
    test_set_runtime_args_observer(observe_runtime_args, &observation);
    const Completion completion = pipeline ? run_stripe_pipeline(args)
                                           : run_full(args);
    test_set_runtime_args_observer(nullptr, nullptr);
    const size_t expected_site = static_cast<size_t>(
        pipeline ? TestRuntimeArgsSite::simple_pipeline_before_execute
                 : TestRuntimeArgsSite::simple_full_before_execute);
    const size_t observations = std::accumulate(
        observation.counts.begin(), observation.counts.end(), size_t{0});
    const bool observed = observation.exact &&
        observation.counts[expected_site] == 1 && observations == 1;
    const bool exact_contract =
        completion.result.error == Error::invalid_contract &&
        std::strcmp(completion.result.message,
                    "invalid native Gemmini route contract") == 0;
    return check(observed,
                 pipeline ? "simple PIPELINE runtime copy is observed exactly once"
                          : "simple FULL runtime copy is observed exactly once") &&
           check(exact_contract,
                 pipeline ? "simple PIPELINE keeps exact invalid native route contract"
                          : "simple FULL keeps exact invalid native route contract");
  };
  return observe_route(false) && observe_route(true);
}

bool run_im2p_semantic_logging_contract() {
  const std::string semantic_layer =
      "blk.15.mlp.down_proj.im2p-lifetime-sentinel-beyond-sso";
  ggml_gemmini_args_t args{};
  {
    std::string source = semantic_layer;
    args.matmul_layer = source;
  }
  args.I = 65;
  args.J = J;
  args.K = K;
  args.tile_I = 1;
  args.tile_J = 1;
  args.tile_K = 1;

  int capture[2]{};
  const int saved_stderr = dup(STDERR_FILENO);
  if (!check(saved_stderr >= 0 && pipe(capture) == 0,
             "open IM2P semantic telemetry capture")) {
    if (saved_stderr >= 0) close(saved_stderr);
    return false;
  }
  std::fflush(stderr);
  dup2(capture[1], STDERR_FILENO);
  close(capture[1]);
  Stats full{};
  full.rtl_work_total_cycles = 117;
  log_stats("full", full, 41, args);
  Stats pipeline{};
  pipeline.rtl_work_total_cycles = 217;
  pipeline.rtl_stripes_published = 3;
  pipeline.rtl_stripe_rows_published = 65;
  log_stats("stripe_pipeline", pipeline, 42, args);
  std::fflush(stderr);
  dup2(saved_stderr, STDERR_FILENO);
  close(saved_stderr);
  std::string output;
  char chunk[1024];
  ssize_t count = 0;
  while ((count = read(capture[0], chunk, sizeof(chunk))) > 0) {
    output.append(chunk, static_cast<size_t>(count));
  }
  close(capture[0]);

  const std::string layer_json = "\"layer\":\"" + semantic_layer + "\"";
  const std::string cycle_type =
      "\"record_type\":\"IM2P_EXECUTION_TELEMETRY\"";
  const auto first = output.find(cycle_type);
#if LOG_CYCLE
  const auto second = first == std::string::npos
                          ? std::string::npos
                          : output.find(cycle_type, first + cycle_type.size());
  const bool cycle_layers = first != std::string::npos &&
                            second != std::string::npos &&
                            output.find(layer_json) != std::string::npos;
#else
  const bool cycle_layers = first == std::string::npos;
#endif
#if LOG_DEBUG
  const bool debug_modes =
      output.find("IM2P_EXECUTION_TELEMETRY_DETAIL mode=full") !=
          std::string::npos &&
      output.find("IM2P_EXECUTION_TELEMETRY_DETAIL mode=stripe_pipeline") !=
          std::string::npos;
#else
  const bool debug_modes =
      output.find("IM2P_EXECUTION_TELEMETRY_DETAIL") == std::string::npos;
#endif
  return check(cycle_layers,
               "IM2P cycle records follow the configured cycle sink state") &&
         check(debug_modes,
               "IM2P debug detail follows the configured debug sink state");
}

bool check_graph_stripe_trace(
    const ggml::gemmini::im2p_adapter::TestCounters &counters) {
  if (!check(counters.stripe_trace_size == graph_publications,
             "all graph stripe publications are traced")) {
    return false;
  }
  for (size_t index = 0; index < graph_publications; ++index) {
    if (!check(counters.stripe_ids[index] == static_cast<int>(index) &&
                   counters.slot_ids[index] == static_cast<int>(index % 2),
               "graph stripe ids and slots are deterministic")) {
      return false;
    }
  }
  return true;
}

bool check_graph_collector_rows(
    const ggml::gemmini::im2p_adapter::TestCounters &counters) {
  for (size_t index = 0; index < graph_publications; ++index) {
    if (!check(counters.collector_row_begin[index] ==
                       index * static_cast<size_t>(I) / graph_publications &&
                   counters.collector_row_end[index] ==
                       (index + 1) * static_cast<size_t>(I) / graph_publications,
               "FULL collector preserves canonical selected row bounds")) {
      return false;
    }
  }
  return true;
}

bool run_exsia_full_success() {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
  setenv("GEMMINI_RMD_BACKEND", "CPU", 1);
  GraphCase test_case;
  if (!check(test_case.initialize(), "initialize ExSIA FULL adapter graph")) {
    return false;
  }
  std::vector<float> expected;
  if (!check(scalar_oracle(test_case.activations, test_case.weights, expected),
             "build FULL ExSIA plus RMD scalar oracle")) {
    return false;
  }

  test_reset();
  const ggml_status status =
      ggml_backend_graph_compute(test_case.backend, test_case.graph);
  const auto counters = test_counters();
  const auto actual = test_case.read_output();
  if (status != GGML_STATUS_SUCCESS || counters.collector_events != graph_publications ||
      counters.rmd_calls != graph_publications) {
    std::fprintf(stderr,
                 "FULL counters status=%d collector=%llu rmd=%llu fence=%llu commit=%llu\n",
                 static_cast<int>(status),
                 static_cast<unsigned long long>(counters.collector_events),
                 static_cast<unsigned long long>(counters.rmd_calls),
                 static_cast<unsigned long long>(counters.fence),
                 static_cast<unsigned long long>(counters.commit));
  }
  bool ok =
      check(status == GGML_STATUS_SUCCESS, "ExSIA FULL adapter succeeds") &&
      check(!test_production_failed(), "FULL records no production failure") &&
      check(counters.full == 1 && counters.pipeline == 0,
            "FULL executes once without a pipeline run") &&
      check(counters.fence == 1 && counters.stripe == 0 &&
                counters.accepted_stripes == 0,
            "FULL fences once and submits zero RTL stripes") &&
      check(counters.collector_events == graph_publications &&
                counters.rmd_calls == graph_publications &&
                counters.rmd_events > 0,
            "FULL collector owns and applies all RMD stripes") &&
      check(counters.collector_handles == graph_publications &&
                counters.collector_theta[0] !=
                    std::numeric_limits<std::int16_t>::min(),
            "FULL collector retains theta and owned RMD handles") &&
      check_graph_collector_rows(counters) &&
      check(counters.rmd_terminal_event != 0 &&
                counters.commit_event > counters.rmd_terminal_event &&
                counters.commit == 1,
            "FULL reaches terminal RMD before one caller commit") &&
      check(counters.authorize == 0 && counters.live_runs == 0 &&
                counters.hardware == 0 && counters.fallback == 0,
            "FULL uses no pipeline authorization, leaked run, or fallback");
  for (size_t index = 0; index < actual.size(); ++index) {
    const float tolerance = 1e-4f * std::max(1.0f, std::fabs(expected[index]));
    ok = check(std::isfinite(actual[index]) &&
                   std::fabs(actual[index] - expected[index]) <= tolerance,
               "FULL output matches scalar ExSIA plus RMD oracle") &&
         ok;
  }
  if (ok) {
    std::printf(
        "events=collector[0:0-256,theta=committed,handle=owned]>"
        "full>fence>rmd[0]>terminal>commit "
        "mode=full full=1 pipeline=0 stripes=0 fence=1 rmd=1 commit=1\n");
  }
  return ok;
}

bool run_exsia_cross_mode_parity() {
  using namespace ggml::gemmini::im2p_adapter;
  std::vector<float> oracle;
  std::vector<float> full_output;
  std::vector<float> pipeline_output;
  TestCounters full_counters{};
  TestCounters pipeline_counters{};

  auto execute_mode = [&](const char *mode, std::vector<float> &output,
                          TestCounters &counters) {
    setenv("GEMMINI_MATMUL_MODE", mode, 1);
    setenv("GEMMINI_RMD_BACKEND", "CPU", 1);
    setenv("GEMMINI_STRIPE_JOB_CAPACITY", "2", 1);
    GraphCase test_case;
    if (!test_case.initialize()) {
      return false;
    }
    if (oracle.empty() &&
        !scalar_oracle(test_case.activations, test_case.weights, oracle)) {
      return false;
    }
    test_reset();
    const ggml_status status =
        ggml_backend_graph_compute(test_case.backend, test_case.graph);
    counters = test_counters();
    output = test_case.read_output();
    return status == GGML_STATUS_SUCCESS && !test_production_failed();
  };

  if (!check(execute_mode("FULL", full_output, full_counters),
             "cross-mode FULL completes") ||
      !check(execute_mode("STRIPE_PIPELINE", pipeline_output,
                          pipeline_counters),
             "cross-mode PIPELINE completes")) {
    return false;
  }
  bool ok =
      check(full_counters.full == 1 && full_counters.pipeline == 0 &&
                full_counters.stripe == 0 && full_counters.commit == 1 &&
                full_counters.fallback == 0,
            "cross-mode FULL commits once with zero fallback") &&
      check(pipeline_counters.full == 0 && pipeline_counters.pipeline == 1 &&
                pipeline_counters.stripe == graph_publications &&
                pipeline_counters.accepted_stripes == graph_publications &&
                pipeline_counters.max_outstanding > 0 &&
                pipeline_counters.max_outstanding <= 2 &&
                pipeline_counters.rmd_calls == graph_publications &&
                pipeline_counters.commit == 1 &&
                pipeline_counters.fallback == 0,
            "cross-mode PIPELINE publishes canonical capacity-two RMD stripes and commits once") &&
      check(full_counters.rmd_calls == graph_publications &&
                full_counters.rmd_terminal_event != 0 &&
                full_counters.commit_event > full_counters.rmd_terminal_event &&
                pipeline_counters.rmd_terminal_event != 0 &&
                pipeline_counters.commit_event >
                    pipeline_counters.rmd_terminal_event,
            "both modes reach terminal RMD before commit");
  float max_oracle_delta = 0.0f;
  float max_mode_delta = 0.0f;
  for (size_t index = 0; index < oracle.size(); ++index) {
    const float tolerance = 1e-4f * std::max(1.0f, std::fabs(oracle[index]));
    const float full_delta = std::fabs(full_output[index] - oracle[index]);
    const float pipeline_delta =
        std::fabs(pipeline_output[index] - oracle[index]);
    const float mode_delta =
        std::fabs(full_output[index] - pipeline_output[index]);
    max_oracle_delta =
        std::max(max_oracle_delta, std::max(full_delta, pipeline_delta));
    max_mode_delta = std::max(max_mode_delta, mode_delta);
    ok = check(full_delta <= tolerance && pipeline_delta <= tolerance &&
                   mode_delta <= tolerance,
               "FULL and PIPELINE match the independent ExSIA plus RMD oracle") &&
         ok;
  }
  if (ok) {
    std::printf("CROSS_MODE_QA full_output=[%.9g,%.9g] "
                "pipeline_output=[%.9g,%.9g] oracle_delta=%.9g "
                "mode_delta=%.9g events=terminal_rmd>commit "
                "full_commit=1 pipeline_commit=1 capacity=2 fallback=0\n",
                full_output[0], full_output[1], pipeline_output[0],
                pipeline_output[1], max_oracle_delta, max_mode_delta);
  }
  return ok;
}

bool run_exsia_full_failure(TestFailure failure) {
  using namespace ggml::gemmini::im2p_adapter;
  GraphCase test_case;
  if (!check(test_case.initialize(), "initialize ExSIA FULL failure graph")) {
    return false;
  }
  setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
  setenv("GEMMINI_RMD_BACKEND", "CPU", 1);
  test_reset();
  test_inject_failure(failure);
  const ggml_status status =
      ggml_backend_graph_compute(test_case.backend, test_case.graph);
  const auto counters = test_counters();
  const bool prequant = failure == TestFailure::collector_allocation;
  const bool quantization = failure == TestFailure::quantization;
  const bool execute = failure == TestFailure::execute;
  const bool fence = failure == TestFailure::fence;
  const bool rmd = failure == TestFailure::rmd;
  const bool ok =
      check(status != GGML_STATUS_SUCCESS, "FULL failure reaches graph status") &&
      check(all_sentinel(test_case.read_output()),
            "FULL failure preserves the caller sentinel") &&
      check(counters.pipeline == 0 && counters.stripe == 0 &&
                counters.accepted_stripes == 0 && counters.commit == 0,
            "FULL failure starts no pipeline, submits no stripe, and commits nothing") &&
      check(counters.full == ((!prequant && !quantization) ? 1U : 0U) &&
                counters.fence == ((fence || rmd) ? 1U : 0U) &&
                counters.rmd_calls == (rmd ? graph_publications : 0U),
            "FULL failure occurs at its injected lifecycle boundary") &&
      check(counters.live_runs == 0 && counters.hardware == 0 &&
                counters.fallback == 0,
            "FULL failure leaks no run and enters no fallback");
  if (ok) {
    const char *name = prequant      ? "collector-allocation"
                       : quantization ? "quantization"
                       : execute      ? "execute"
                       : fence        ? "fence"
                                      : "rmd";
    std::printf("full-failure=%s full=%llu fence=%llu rmd=%llu commit=0 "
                "sentinel=preserved live_runs=0\n",
                name, static_cast<unsigned long long>(counters.full),
                static_cast<unsigned long long>(counters.fence),
                static_cast<unsigned long long>(counters.rmd_calls));
  }
  return ok;
}

bool run_exsia_full_collector_capture_failure() {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
  setenv("GEMMINI_RMD_BACKEND", "CPU", 1);

  GraphCase failed_case;
  if (!check(failed_case.initialize(),
             "initialize callback-originated collector failure graph")) {
    return false;
  }
  test_reset();
  test_inject_failure(TestFailure::collector_capture);
  const ggml_status failed_status =
      ggml_backend_graph_compute(failed_case.backend, failed_case.graph);
  const TestCounters failed = test_counters();
  bool ok =
      check(failed_status != GGML_STATUS_SUCCESS,
            "callback-originated collector failure reaches graph status") &&
      check(failed.production_error == Error::invalid_contract,
            "callback-originated collector failure preserves its typed result") &&
      check(all_sentinel(failed_case.read_output()),
            "callback-originated collector failure preserves caller sentinel") &&
      check(failed.full == 0 && failed.pipeline == 0 && failed.fence == 0 &&
                failed.stripe == 0 && failed.accepted_stripes == 0 &&
                failed.rmd_calls == 0 && failed.authorize == 0 &&
                failed.commit == 0 && failed.hardware == 0 &&
                failed.fallback == 0 && failed.live_runs == 0,
            "callback-originated collector failure starts no execution or fallback") &&
      check(failed.quantization_failures == 0,
            "collector callback failure is distinct from injected quantization failure");

  GraphCase reused_case;
  if (!check(reused_case.initialize(),
             "initialize FULL transaction after collector failure")) {
    return false;
  }
  test_reset();
  const ggml_status reused_status =
      ggml_backend_graph_compute(reused_case.backend, reused_case.graph);
  const TestCounters reused = test_counters();
  ok = check(reused_status == GGML_STATUS_SUCCESS &&
                 !test_production_failed() && reused.full == 1 &&
                 reused.pipeline == 0 && reused.fence == 1 &&
                 reused.stripe == 0 &&
                 reused.rmd_calls == graph_publications &&
                 reused.commit == 1 && reused.fallback == 0 &&
                 reused.live_runs == 0,
             "FULL lifecycle is reusable after collector callback failure") &&
       ok;
  if (ok) {
    std::printf("full-failure=collector-capture error=invalid_contract "
                "full=0 pipeline=0 fence=0 rmd=0 commit=0 fallback=0 "
                "sentinel=preserved live_runs=0 reuse=pass\n");
  }
  return ok;
}

bool run_exsia_success() {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "STRIPE_PIPELINE", 1);
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
      check(counters.stripe == graph_publications &&
                counters.accepted_stripes == graph_publications,
            "all canonical post-fold stripes reach the frontend") &&
      check(counters.max_outstanding > 0 && counters.max_outstanding <= 2,
            "frontend keeps at most two stripes outstanding") &&
      check(counters.rmd_calls == graph_publications &&
                counters.rmd_events > 0,
            "unchanged cpu_direct RMD consumes every stripe and real residual "
            "events") &&
      check(counters.authorize == 1 && counters.rmd_terminal_event != 0 &&
                counters.authorize_success_event > counters.rmd_terminal_event,
            "RMD reaches terminal success before output authorization") &&
      check(counters.commit == 1, "staged output commits exactly once") &&
      check(counters.hardware == 0 && counters.fallback == 0,
            "ExSIA route enters no hardware or fallback path") &&
      check(counters.live_runs == 0, "all frontend workers are joined") &&
      check_graph_stripe_trace(counters);
  for (size_t index = 0; index < actual.size(); ++index) {
    const float tolerance = 1e-4f * std::max(1.0f, std::fabs(expected[index]));
    ok = check(std::isfinite(actual[index]) &&
                   std::fabs(actual[index] - expected[index]) <= tolerance,
               "ExSIA production output matches scalar RMD oracle") &&
         ok;
  }
  if (ok) {
    std::printf(
        "route=exsia mode=stripe_pipeline bits=%d dim=%d stripes=1 slots=0 "
        "capacity=2 events=seal>fold_commit>callback>rtl_publish>"
        "activation_read>quantization_complete rmd=cpu_direct "
        "publish_cycle=%llu activation_read_cycle=%llu "
        "rmd_terminal_event=%llu authorize_event=%llu "
        "full=0 pipeline=1 stripes=1 fence=1 rmd=1 commit=1 "
        "hardware=0 fallback=0\n",
        GGML_GEMMINI_ACTIVATION_BITS, GGML_GEMMINI_TEST_IM2P_DIM,
        static_cast<unsigned long long>(counters.first_publish_cycle),
        static_cast<unsigned long long>(counters.first_activation_read_cycle),
        static_cast<unsigned long long>(counters.rmd_terminal_event),
        static_cast<unsigned long long>(counters.authorize_success_event));
  }
  return ok;
}

bool run_integrated_geometry_oracle() {
  const auto a_full = run_integrated_exsia_lifecycle(
      2 * GGML_GEMMINI_TEST_IM2P_DIM + 1, 1, LifecycleFamily::h1,
      LifecycleBackend::cpu_direct, PublicMode::full);
  const auto a_pipeline = run_integrated_exsia_lifecycle(
      2 * GGML_GEMMINI_TEST_IM2P_DIM + 1, 1, LifecycleFamily::h1,
      LifecycleBackend::cpu_direct, PublicMode::stripe_pipeline);
  const auto b_full = run_integrated_exsia_lifecycle(
      16 * GGML_GEMMINI_TEST_IM2P_DIM, 5, LifecycleFamily::h1,
      LifecycleBackend::cpu_direct, PublicMode::full);
  const auto b_pipeline = run_integrated_exsia_lifecycle(
      16 * GGML_GEMMINI_TEST_IM2P_DIM, 5, LifecycleFamily::h1,
      LifecycleBackend::cpu_direct, PublicMode::stripe_pipeline);
  const auto rows = [](const TestCounters &counters, size_t index) {
    return counters.stripe_row_end[index] - counters.stripe_row_begin[index];
  };
  bool ok =
      check(a_full.ok && a_pipeline.ok && b_full.ok && b_pipeline.ok,
            "integrated canonical ExSIA transactions complete") &&
      check(a_full.semantic_layer_observed &&
                a_pipeline.semantic_layer_observed &&
                b_full.semantic_layer_observed &&
                b_pipeline.semantic_layer_observed,
            "FULL and PIPELINE runtime copies retain one semantic layer after source destruction") &&
      check(a_full.counters.collector_events == 3 &&
                a_full.counters.collector_handles == 3 &&
                a_full.counters.stripe == 0 &&
                a_full.completion.stats.rtl_stripes_published == 0 &&
                a_full.completion.stats.rtl_stripe_rows_published == 0 &&
                b_full.counters.collector_events == 4 &&
                b_full.counters.collector_handles == 4 &&
                b_full.counters.stripe == 0 &&
                b_full.completion.stats.rtl_stripes_published == 0 &&
                b_full.completion.stats.rtl_stripe_rows_published == 0,
            "FULL collects all theta/RMD ownership and publishes zero") &&
      check(a_pipeline.counters.stripe_trace_size == 3 &&
                rows(a_pipeline.counters, 0) == GGML_GEMMINI_TEST_IM2P_DIM &&
                rows(a_pipeline.counters, 1) == GGML_GEMMINI_TEST_IM2P_DIM &&
                rows(a_pipeline.counters, 2) == 1 &&
                a_pipeline.counters.rmd_calls == 3 &&
                a_pipeline.counters.commit == 1 &&
                a_pipeline.completion.stats.rtl_stripes_published == 3 &&
                a_pipeline.completion.stats.rtl_stripe_rows_published ==
                    2 * GGML_GEMMINI_TEST_IM2P_DIM + 1,
            "fixture A folds, seals, publishes, fences, applies RMD, and commits [DIM,DIM,1]") &&
      check(b_pipeline.counters.stripe_trace_size == 4 &&
                rows(b_pipeline.counters, 0) == 5 * GGML_GEMMINI_TEST_IM2P_DIM &&
                rows(b_pipeline.counters, 1) == 5 * GGML_GEMMINI_TEST_IM2P_DIM &&
                rows(b_pipeline.counters, 2) == 5 * GGML_GEMMINI_TEST_IM2P_DIM &&
                rows(b_pipeline.counters, 3) == GGML_GEMMINI_TEST_IM2P_DIM &&
                b_pipeline.counters.rmd_calls == 4 &&
                b_pipeline.counters.commit == 1 &&
                b_pipeline.completion.stats.rtl_stripes_published == 4 &&
                b_pipeline.completion.stats.rtl_stripe_rows_published ==
                    16 * GGML_GEMMINI_TEST_IM2P_DIM,
            "fixture B folds, seals, publishes, fences, applies RMD, and commits [80,80,80,16]") &&
      check(a_full.output == a_pipeline.output &&
                b_full.output == b_pipeline.output,
            "FULL and PIPELINE integrated outputs are identical");
  if (ok) {
    std::printf("INTEGRATED_EXSIA_GEOMETRY A=[%d,%d,1] B=[%d,%d,%d,%d] "
                "full=collect_theta_rmd/publish0 pipeline=fold>seal>publish>fence>rmd>authorize>commit\n",
                GGML_GEMMINI_TEST_IM2P_DIM, GGML_GEMMINI_TEST_IM2P_DIM,
                5 * GGML_GEMMINI_TEST_IM2P_DIM,
                5 * GGML_GEMMINI_TEST_IM2P_DIM,
                5 * GGML_GEMMINI_TEST_IM2P_DIM,
                GGML_GEMMINI_TEST_IM2P_DIM);
  }
  return ok;
}

bool run_route_lifecycle_table() {
  struct Identity {
    LifecycleFamily family;
    LifecycleBackend backend;
  };
  constexpr std::array<Identity, 5> identities{{
      {LifecycleFamily::h0, LifecycleBackend::cpu_direct},
      {LifecycleFamily::h1, LifecycleBackend::cpu_direct},
      {LifecycleFamily::h1, LifecycleBackend::compact_ws},
      {LifecycleFamily::hp1, LifecycleBackend::cpu_direct},
      {LifecycleFamily::hp1, LifecycleBackend::compact_ws},
  }};
  constexpr std::array<PublicMode, 2> modes{{PublicMode::full,
                                             PublicMode::stripe_pipeline}};
  size_t terminal = 0;
  for (const PublicMode mode : modes) {
    for (const Identity identity : identities) {
      bool case_ok = false;
      bool semantic_layer_observed = true;
      TestCounters counters{};
      Completion completion{};
#if GGML_GEMMINI_WEIGHT_BITS == 8
      if (identity.family == LifecycleFamily::h0) {
        case_ok = mode == PublicMode::full ? run_exsia_full_success()
                                           : run_exsia_success();
        counters = test_counters();
        completion.result = case_ok ? Result{} : Result{Error::execution_failure,
                                                        "H0 graph failed", false};
        completion.stats.rtl_stripes_published =
            mode == PublicMode::full ? 0 : graph_publications;
        completion.stats.rtl_stripe_rows_published =
            mode == PublicMode::full ? 0 : I;
      } else
#endif
      {
        const auto result = run_integrated_exsia_lifecycle(
            2 * GGML_GEMMINI_TEST_IM2P_DIM + 1, 1, identity.family,
            identity.backend, mode);
        case_ok = result.ok;
        semantic_layer_observed = result.semantic_layer_observed;
        counters = result.counters;
        completion = result.completion;
      }
      const uint64_t route_stripes =
#if GGML_GEMMINI_WEIGHT_BITS == 8
          identity.family == LifecycleFamily::h0 ? graph_publications : 3;
#else
          3;
#endif
      const uint64_t expected_publications =
          mode == PublicMode::full ? 0 : route_stripes;
      LifecycleFamily observed_family = LifecycleFamily::h0;
      bool family_observed = counters.weight_family_observations != 0;
      switch (counters.observed_weight_family) {
      case WeightFamily::h0:
        observed_family = LifecycleFamily::h0;
        break;
      case WeightFamily::h1:
        observed_family = LifecycleFamily::h1;
        break;
      case WeightFamily::hp1:
        observed_family = LifecycleFamily::hp1;
        break;
      default:
        family_observed = false;
        break;
      }
      family_observed = family_observed && observed_family == identity.family;
      const bool backend_observed =
          identity.backend == LifecycleBackend::cpu_direct
              ? counters.rmd_events > 0 && counters.rmd_packets == 0
              : counters.rmd_packets == route_stripes &&
                    counters.rmd_events == 0;
      case_ok = check(case_ok && semantic_layer_observed && family_observed &&
                          counters.fence == 1 &&
                          counters.rmd_calls == route_stripes &&
                          counters.commit == 1 && counters.live_runs == 0 &&
                          counters.fallback == 0 && backend_observed &&
                          completion.stats.rtl_stripes_published ==
                              expected_publications &&
                          (mode == PublicMode::full ||
                           completion.stats.rtl_stripe_rows_published != 0),
                      "legal route identity reaches one matching terminal lifecycle") &&
                case_ok;
      if (!case_ok) return false;
      ++terminal;
      std::printf("ROUTE_LIFECYCLE bits=%d mode=%s requested_family=%s "
                  "observed_family=%s requested_backend=%s observed_backend=%s "
                  "publications=%llu terminal=1 commit=1 fallback=0\n",
                  GGML_GEMMINI_ACTIVATION_BITS,
                  mode == PublicMode::full ? "FULL" : "STRIPE_PIPELINE",
                  lifecycle_family_name(identity.family),
                  lifecycle_family_name(observed_family),
                  lifecycle_backend_name(identity.backend),
                  lifecycle_backend_name(identity.backend),
                  static_cast<unsigned long long>(expected_publications));
    }
  }
  return check(terminal == 10,
               "each width executes ten legal family/backend/mode identities");
}

bool run_exsia_one_row_pipeline() {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "STRIPE_PIPELINE", 1);
  setenv("GEMMINI_RMD_BACKEND", "CPU", 1);
  GraphCase test_case(1);
  if (!check(test_case.initialize(), "initialize one-row ExSIA graph")) {
    return false;
  }

  test_reset();
  test_inject_failure(TestFailure::execute);
  const ggml_status status =
      ggml_backend_graph_compute(test_case.backend, test_case.graph);
  const auto counters = test_counters();
  const bool ok =
      check(status != GGML_STATUS_SUCCESS,
            "one-row PIPELINE reaches its injected start failure") &&
      check(all_sentinel(test_case.read_output()),
            "one-row PIPELINE failure preserves destination sentinel") &&
      check(counters.full == 0 && counters.pipeline == 1 &&
                counters.stripe == 0 && counters.fence == 0 &&
                counters.rmd_calls == 0 && counters.authorize == 0 &&
                counters.commit == 0,
            "one-row request remains pipeline without a full shortcut") &&
      check(counters.hardware == 0 && counters.fallback == 0 &&
                counters.live_runs == 0,
            "one-row PIPELINE has no fallback or leaked run");
  if (ok) {
    std::printf("route=exsia mode=stripe_pipeline rows=1 full=0 pipeline=1 "
                "stripes=0 commit=0 fallback=0 sentinel=preserved\n");
  }
  return ok;
}

bool run_exsia_start_failure() {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "STRIPE_PIPELINE", 1);
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
  setenv("GEMMINI_MATMUL_MODE", "STRIPE_PIPELINE", 1);
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
  const char *failure_name =
      quantization ? "quantization"
      : failure == TestFailure::provider ? "provider"
      : progress ? "progress"
      : poll ? "poll"
      : failure == TestFailure::malformed_completion ? "malformed-completion"
                                                     : "incomplete-publication";
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
  setenv("GEMMINI_MATMUL_MODE", "STRIPE_PIPELINE", 1);
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
  const bool before_composition =
      fence_failure || failure == TestFailure::dense ||
      failure == TestFailure::residual_execute || failure == TestFailure::compose;
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
      check(counters.stripe == graph_publications &&
                counters.accepted_stripes == graph_publications,
            "failure path publishes all canonical stripes") &&
      check(counters.rmd_calls ==
                (before_composition ? 0U : graph_publications),
            "completed RMD composition count matches the injected boundary") &&
      check(counters.authorize == 1 && counters.commit == 0,
            "failed transaction is explicitly rejected and never committed") &&
      check(counters.hardware == 0 && counters.fallback == 0 &&
                counters.live_runs == 0,
            "failed transaction has no fallback and joins every worker");
  if (ok) {
    const char *name = fence_failure ? "fence"
                       : failure == TestFailure::dense ? "dense"
                       : failure == TestFailure::residual_execute ? "residual-execute"
                       : failure == TestFailure::compose ? "compose"
                       : failure == TestFailure::output_authorization
                           ? "output-authorization"
                       : failure == TestFailure::output_copy ? "output-copy"
                                                            : "rmd";
    std::printf("failure=%s error=execution_failure sentinel=preserved full=0 "
                "hardware=0 fallback=0 live_runs=0\n", name);
  }
  return ok;
}

bool run_exsia_blocked_submit_failure() {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", "STRIPE_PIPELINE", 1);
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
  const bool ok =
      check(blocked, "third producer blocks on the capacity-two frontend") &&
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
         check(counters.stripe_trace_size == 3 &&
                   counters.stripe_ids[0] == 0 && counters.stripe_ids[1] == 1 &&
                   counters.stripe_ids[2] == 2 && counters.slot_ids[0] == 0 &&
                   counters.slot_ids[1] == 1 && counters.slot_ids[2] == 0,
               "blocked producer observes deterministic first-three order");
  if (ok) {
    std::printf("failure=blocked_submit producer=fence=execution_failure "
                "sentinel=preserved capacity=2 slots=0,1,0,1\n");
  }
  return ok;
}

bool run_exsia_unsupported_mode() {
  std::string command = "ulimit -c 0; \"";
  command += routing_program;
  command += "\" --unsupported-mode-child >/dev/null 2>&1";
  const bool ok = check(std::system(command.c_str()) != 0,
                        "unsupported ExSIA mode fails closed before dispatch");
  if (ok) {
    std::printf("removed_mode=STRIPE_SEQUENTIAL error=invalid_mode full=0 pipeline=0 "
                "stripes=0 commit=0 fallback=0\n");
  }
  return ok;
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

bool run_stats_translation_contract() {
  using namespace ggml::gemmini::im2p_adapter;

  ::im2p::gemmini::FenceResult source{};
  source.status = {};
  std::uint64_t value = 101;
#define SET_RAW(field) source.stats.field = value++
  SET_RAW(base.work_total_cycles);
  SET_RAW(base.activation_read_requests);
  SET_RAW(base.weight_read_requests);
  SET_RAW(base.scale_read_requests);
  SET_RAW(base.output_write_requests);
  SET_RAW(base.output_write_responses);
  SET_RAW(base.activation_wait_cycles);
  SET_RAW(base.weight_wait_cycles);
  SET_RAW(base.scale_wait_cycles);
  SET_RAW(base.output_wait_cycles);
  SET_RAW(base.stripe_host_wait_cycles);
  SET_RAW(base.drain_cycles);
  SET_RAW(base.weight_preload_cycles);
  SET_RAW(base.same_block_scale_hits);
  SET_RAW(base.next_scale_hits);
  SET_RAW(base.scale_demand_misses);
  SET_RAW(base.compute_cycles);
  SET_RAW(base.overlap_cycles);
  SET_RAW(base.activation_overlap_cycles);
  SET_RAW(base.weight_overlap_cycles);
  SET_RAW(base.scale_overlap_cycles);
  SET_RAW(base.completed_fragments);
  SET_RAW(base.completed_output_tiles);
  SET_RAW(base.completed_stripes);
  SET_RAW(base.stripes_published);
  SET_RAW(base.stripe_rows_published);
  SET_RAW(base.weight_bank_activations);
  SET_RAW(cross_stripe_overlap_cycles);
  SET_RAW(lookahead_prepared);
  SET_RAW(lookahead_publish_cycle);
  SET_RAW(lookahead_first_activation_cycle);
  SET_RAW(lookahead_first_weight_cycle);
  SET_RAW(lookahead_weight_preload_cycle);
  SET_RAW(lookahead_weight_requests);
  SET_RAW(lookahead_weight_reuse_hits);
  SET_RAW(lookahead_scale_cycle);
  SET_RAW(lookahead_scale_requests);
  SET_RAW(lookahead_scale_reuses);
  SET_RAW(current_stripe_completion_cycle);
  SET_RAW(lookahead_ready_cycle);
  SET_RAW(lookahead_start_cycle);
#undef SET_RAW

  const Completion translated = translate(
      source, ::im2p::gemmini::Mode::stripe_pipeline,
      source.stats.base.stripes_published,
      source.stats.base.stripe_rows_published);
  const std::array<std::uint64_t, 41> actual = {
      translated.stats.rtl_work_total_cycles,
      translated.stats.rtl_activation_read_requests,
      translated.stats.rtl_weight_read_requests,
      translated.stats.rtl_scale_read_requests,
      translated.stats.rtl_output_write_requests,
      translated.stats.rtl_output_write_responses,
      translated.stats.rtl_activation_wait_cycles,
      translated.stats.rtl_weight_wait_cycles,
      translated.stats.rtl_scale_wait_cycles,
      translated.stats.rtl_output_wait_cycles,
      translated.stats.rtl_stripe_host_wait_cycles,
      translated.stats.rtl_drain_cycles,
      translated.stats.rtl_weight_preload_cycles,
      translated.stats.rtl_same_block_scale_hits,
      translated.stats.rtl_next_scale_hits,
      translated.stats.rtl_scale_demand_misses,
      translated.stats.rtl_compute_cycles,
      translated.stats.rtl_overlap_cycles,
      translated.stats.rtl_activation_overlap_cycles,
      translated.stats.rtl_weight_overlap_cycles,
      translated.stats.rtl_scale_overlap_cycles,
      translated.stats.rtl_completed_fragments,
      translated.stats.rtl_completed_output_works,
      translated.stats.rtl_scheduler_groups_completed,
      translated.stats.rtl_stripes_published,
      translated.stats.rtl_stripe_rows_published,
      translated.stats.rtl_weight_bank_activations,
      translated.stats.rtl_cross_stripe_overlap_cycles,
      translated.stats.rtl_lookahead_prepared,
      translated.stats.rtl_first_publish_cycle,
      translated.stats.rtl_first_activation_read_cycle,
      translated.stats.rtl_first_weight_read_cycle,
      translated.stats.rtl_weight_preload_cycle,
      translated.stats.rtl_lookahead_weight_requests,
      translated.stats.rtl_lookahead_weight_reuse_hits,
      translated.stats.rtl_first_scale_read_cycle,
      translated.stats.rtl_lookahead_scale_requests,
      translated.stats.rtl_lookahead_scale_reuses,
      translated.stats.rtl_current_scheduler_group_completion_cycle,
      translated.stats.rtl_lookahead_ready_cycle,
      translated.stats.rtl_lookahead_start_cycle,
  };
  constexpr std::array<const char *, 41> names = {
      "rtl_work_total_cycles",
      "rtl_activation_read_requests",
      "rtl_weight_read_requests",
      "rtl_scale_read_requests",
      "rtl_output_write_requests",
      "rtl_output_write_responses",
      "rtl_activation_wait_cycles",
      "rtl_weight_wait_cycles",
      "rtl_scale_wait_cycles",
      "rtl_output_wait_cycles",
      "rtl_stripe_host_wait_cycles",
      "rtl_drain_cycles",
      "rtl_weight_preload_cycles",
      "rtl_same_block_scale_hits",
      "rtl_next_scale_hits",
      "rtl_scale_demand_misses",
      "rtl_compute_cycles",
      "rtl_overlap_cycles",
      "rtl_activation_overlap_cycles",
      "rtl_weight_overlap_cycles",
      "rtl_scale_overlap_cycles",
      "rtl_completed_fragments",
      "rtl_completed_output_works",
      "rtl_scheduler_groups_completed",
      "rtl_stripes_published",
      "rtl_stripe_rows_published",
      "rtl_weight_bank_activations",
      "rtl_cross_stripe_overlap_cycles",
      "rtl_lookahead_prepared",
      "rtl_first_publish_cycle",
      "rtl_first_activation_read_cycle",
      "rtl_first_weight_read_cycle",
      "rtl_weight_preload_cycle",
      "rtl_lookahead_weight_requests",
      "rtl_lookahead_weight_reuse_hits",
      "rtl_first_scale_read_cycle",
      "rtl_lookahead_scale_requests",
      "rtl_lookahead_scale_reuses",
      "rtl_current_scheduler_group_completion_cycle",
      "rtl_lookahead_ready_cycle",
      "rtl_lookahead_start_cycle",
  };
  bool ok = check(translated.result.ok(),
                  "sentinel statistics satisfy PIPELINE geometry");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    ok = check(actual[index] == 101 + index,
               "each unique raw sentinel maps to its semantic field exactly once") &&
         ok;
    for (std::size_t other = index + 1; other < actual.size(); ++other) {
      ok = check(actual[index] != actual[other],
                 "translated sentinel values remain pairwise unique") &&
           ok;
    }
  }

  auto full = source;
  full.stats.base.stripes_published = 0;
  full.stats.base.stripe_rows_published = 0;
  const Completion full_translated =
      translate(full, ::im2p::gemmini::Mode::full, 0, 0);
  ok = check(full_translated.result.ok() &&
                 full_translated.stats.rtl_compute_cycles == 117 &&
                 full_translated.stats.rtl_completed_output_works == 123 &&
                 full_translated.stats.rtl_scheduler_groups_completed == 124 &&
                 full_translated.stats.rtl_stripes_published == 0 &&
                 full_translated.stats.rtl_stripe_rows_published == 0,
             "FULL keeps non-publication RTL statistics but publishes no rows") &&
       ok;
  auto canonical_pipeline_source = source;
  canonical_pipeline_source.stats.base.stripes_published = 3;
  canonical_pipeline_source.stats.base.stripe_rows_published = 33;
  const Completion canonical_pipeline = translate(
      canonical_pipeline_source, ::im2p::gemmini::Mode::stripe_pipeline, 3,
      33);
  ok = check(canonical_pipeline.result.ok(),
             "PIPELINE accepts the canonical three-stripe 33-row geometry") &&
       ok;
  const Completion invalid_full =
      translate(source, ::im2p::gemmini::Mode::full, 0, 0);
  const Completion invalid_pipeline_count = translate(
      source, ::im2p::gemmini::Mode::stripe_pipeline,
      source.stats.base.stripes_published + 1,
      source.stats.base.stripe_rows_published);
  const Completion invalid_pipeline_rows = translate(
      source, ::im2p::gemmini::Mode::stripe_pipeline,
      source.stats.base.stripes_published,
      source.stats.base.stripe_rows_published + 1);
  ok = check(invalid_full.result.error == Error::invalid_contract &&
                 invalid_pipeline_count.result.error == Error::invalid_contract &&
                 invalid_pipeline_rows.result.error == Error::invalid_contract,
             "impossible FULL and PIPELINE publication geometry fails closed") &&
       ok;
  if (ok) {
    std::printf("IM2P_STATS_SENTINELS");
    for (std::size_t index = 0; index < actual.size(); ++index) {
      std::printf(" %s=%llu", names[index],
                  static_cast<unsigned long long>(actual[index]));
    }
    std::printf("\nIM2P_STATS_MODES pipeline_publications=%llu "
                "pipeline_rows=%llu canonical_publications=3 "
                "canonical_rows=33 full_compute=%llu "
                "full_publications=0 full_rows=0\n",
                static_cast<unsigned long long>(
                    translated.stats.rtl_stripes_published),
                static_cast<unsigned long long>(
                    translated.stats.rtl_stripe_rows_published),
                static_cast<unsigned long long>(
                    full_translated.stats.rtl_compute_cycles));
  }
  return ok;
}

bool run_routing_geometry_contract() {
  using namespace ggml::gemmini::im2p_adapter;
  const auto geometry = ggml::gemmini::make_gemmini_geometry(
      {{static_cast<size_t>(I), static_cast<size_t>(J), static_cast<size_t>(K)},
       {2, 1, 1}, GGML_GEMMINI_TEST_IM2P_DIM});
  const size_t expected_stripes = compiled_exsia ? 8 : 1;
  const size_t expected_final_rows = compiled_exsia ? 32 : 3;
  bool ok = check(geometry.ok(), "routing geometry is valid") &&
            check(geometry.geometry.stripe_rows == 32 &&
                      geometry.geometry.stripe_count == expected_stripes &&
                      geometry.geometry.final_rows == expected_final_rows,
                  "routing rows follow literal nontrivial geometry");
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
  ggml_gemmini_args_t mismatch{};
  mismatch.I = I; mismatch.J = J; mismatch.K = K;
  mismatch.tile_I = 2; mismatch.tile_J = 1; mismatch.tile_K = 1;
  mismatch.activation_rows_per_stripe = 16;
  test_reset();
  auto start = start_exsia_stripe_pipeline(mismatch);
  const auto counters = test_counters();
  ok = check(start.result.error == Error::invalid_contract && !start.pipeline,
             "IM2P rejects mismatched activation geometry with typed status") &&
       check(counters.pipeline == 0 && counters.stripe == 0 &&
                 counters.accepted_stripes == 0 && counters.commit == 0 &&
                 counters.live_runs == 0,
             "IM2P mismatch has zero allocation/publication side effects") && ok;
#endif
  if (ok) {
    std::printf("ROUTING_GEOMETRY stripe_rows=32 stripe_count=%zu final_rows=%zu mismatch=invalid_contract allocations=0 publications=0\n",
                geometry.geometry.stripe_count, geometry.geometry.final_rows);
  }
  return ok;
}

bool run_success_mode(const char *mode) {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", mode, 1);
  const bool pipeline = std::string_view(mode) == "STRIPE_PIPELINE";
  const auto geometry = ggml::gemmini::make_gemmini_geometry(
      {{static_cast<size_t>(I), static_cast<size_t>(J), static_cast<size_t>(K)},
       {1, 1, 1}, GGML_GEMMINI_TEST_IM2P_DIM});
  if (!check(geometry.ok(), "derive routing fixture geometry")) {
    return false;
  }
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
      check(counters.full == (pipeline ? 0u : 1u),
            "production dispatch selects the requested IM2P execution mode") &&
      check(counters.fence == 1, "production dispatch fences once") &&
      check(counters.stripe == (pipeline ? 1u : 0u),
            "production PIPELINE dispatch publishes through the stripe path") &&
      check(counters.accepted_stripes ==
                (pipeline ? static_cast<uint64_t>(geometry.geometry.stripe_count)
                          : uint64_t{0}),
            "production PIPELINE follows canonical geometry stripes") &&
      check(counters.hardware == 0,
            "production dispatch enters no hardware path");
  for (size_t index = 0; index < actual.size(); ++index) {
    ok = check(std::isfinite(actual[index]) &&
                   std::fabs(actual[index] - expected[index]) < 1e-4f,
               "production output matches scalar quantized oracle") &&
         ok;
  }
  if (ok) {
    std::printf("route=%s weight=%s mode=%s bits=%d weight_bits=%d dim=%d "
                "full=%llu stripe=%llu hardware=0 supports_selected=yes "
                "supports_mismatch=%s\n",
                compiled_route(), compiled_weight_route(), mode,
                GGML_GEMMINI_ACTIVATION_BITS, GGML_GEMMINI_WEIGHT_BITS,
                GGML_GEMMINI_TEST_IM2P_DIM,
                static_cast<unsigned long long>(counters.full),
                static_cast<unsigned long long>(counters.stripe),
                compiled_mismatch_support_result());
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

bool run_fence_failure(const char *mode) {
  using namespace ggml::gemmini::im2p_adapter;
  setenv("GEMMINI_MATMUL_MODE", mode, 1);
  const bool pipeline = std::string_view(mode) == "STRIPE_PIPELINE";
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
         check(counters.full == (pipeline ? 0u : 1u) &&
                   counters.fence == 1 &&
                   counters.stripe == (pipeline ? 1u : 0u) &&
                   counters.hardware == 0,
               "fence failure preserves requested dispatch without fallback");
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

bool run_unsupported_mode_child() {
  setenv("GEMMINI_MATMUL_MODE", "STRIPE_SEQUENTIAL", 1);
  setenv("GEMMINI_RMD_BACKEND", "CPU", 1);
  GraphCase test_case;
  if (!test_case.initialize()) {
    return false;
  }
  return ggml_backend_graph_compute(test_case.backend, test_case.graph) ==
         GGML_STATUS_SUCCESS;
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
  routing_program = argv[0];
  if (argc == 3 && std::string_view(argv[1]) == "--case" &&
      std::string_view(argv[2]) == "all") {
    argc = 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--route-matrix") {
    return run_matched_weight_gate_contract() ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--geometry-contract") {
    return run_routing_geometry_contract() ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--stats-translation") {
    return run_stats_translation_contract() ? 0 : 1;
  }
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
  if (argc == 2 && std::string_view(argv[1]) == "--publication-boundary") {
    return run_exsia_publication_boundary() ? 0 : 1;
  }
#endif
  if (argc == 2 && std::string_view(argv[1]) == "--unsupported-mode-child") {
    return run_unsupported_mode_child() ? 0 : 1;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--invalid-mode-child") {
    return run_invalid_mode_child() ? 0 : 1;
  }
  if (!run_matched_weight_gate_contract() || !run_graph_overhead_regression() ||
      !run_exsia_shift_regression() ||
      !run_simple_runtime_args_observer_contract() ||
      !run_im2p_semantic_logging_contract()
#if GGML_GEMMINI_ACTIVATION_QUANT == 0
      || !run_exsia_publication_boundary()
#endif
  ) {
    return 1;
  }
  if (argc == 3 && std::string_view(argv[1]) == "--case") {
    const std::string_view selected(argv[2]);
#if GGML_GEMMINI_ACTIVATION_QUANT == 0 && GGML_GEMMINI_ENABLE_RMD
    bool selected_ok = false;
    if (selected == "full") {
      selected_ok = run_exsia_full_success();
    } else if (selected == "integrated-geometry") {
      selected_ok = run_integrated_geometry_oracle();
    } else if (selected == "route-lifecycle") {
      selected_ok = run_route_lifecycle_table();
    } else if (selected == "cross-mode-oracle") {
      selected_ok = run_exsia_cross_mode_parity();
    } else if (selected == "pipeline") {
      selected_ok = run_exsia_success();
    } else if (selected == "full-collector-allocation") {
      selected_ok = run_exsia_full_failure(TestFailure::collector_allocation);
    } else if (selected == "full-collector-capture") {
      selected_ok = run_exsia_full_collector_capture_failure();
    } else if (selected == "full-quantization") {
      selected_ok = run_exsia_full_failure(TestFailure::quantization);
    } else if (selected == "full-execute") {
      selected_ok = run_exsia_full_failure(TestFailure::execute);
    } else if (selected == "full-fence") {
      selected_ok = run_exsia_full_failure(TestFailure::fence);
    } else if (selected == "full-rmd") {
      selected_ok = run_exsia_full_failure(TestFailure::rmd);
    } else if (selected == "one-row-pipeline") {
      selected_ok = run_exsia_one_row_pipeline();
    } else if (selected == "unsupported-mode") {
      selected_ok = run_exsia_unsupported_mode();
    } else if (selected == "execute") {
      selected_ok = run_exsia_start_failure();
    } else if (selected == "quantization") {
      selected_ok = run_exsia_boundary_failure(
          ggml::gemmini::im2p_adapter::TestFailure::quantization);
    } else if (selected == "provider") {
      selected_ok = run_exsia_boundary_failure(
          ggml::gemmini::im2p_adapter::TestFailure::provider);
    } else if (selected == "progress") {
      selected_ok = run_exsia_boundary_failure(
          ggml::gemmini::im2p_adapter::TestFailure::progress);
    } else if (selected == "poll") {
      selected_ok = run_exsia_boundary_failure(
          ggml::gemmini::im2p_adapter::TestFailure::poll);
    } else if (selected == "malformed-completion") {
      selected_ok = run_exsia_boundary_failure(
          ggml::gemmini::im2p_adapter::TestFailure::malformed_completion);
    } else if (selected == "incomplete-publication") {
      selected_ok = run_exsia_boundary_failure(
          ggml::gemmini::im2p_adapter::TestFailure::incomplete_publication);
    } else if (selected == "fence") {
      selected_ok = run_exsia_staged_failure(
          ggml::gemmini::im2p_adapter::TestFailure::fence);
    } else if (selected == "rmd") {
      selected_ok = run_exsia_staged_failure(TestFailure::rmd);
    } else if (selected == "dense") {
      selected_ok = run_exsia_staged_failure(TestFailure::dense);
    } else if (selected == "residual-execute") {
      selected_ok = run_exsia_staged_failure(TestFailure::residual_execute);
    } else if (selected == "compose") {
      selected_ok = run_exsia_staged_failure(TestFailure::compose);
    } else if (selected == "output-authorization") {
      selected_ok = run_exsia_staged_failure(TestFailure::output_authorization);
    } else if (selected == "output-copy") {
      selected_ok = run_exsia_staged_failure(TestFailure::output_copy);
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
  int requested_stripes = compiled_exsia ? graph_publications : 3;
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
       (requested_stripes != static_cast<int>(graph_publications) ||
        requested_queue_capacity != 2 ||
        requested_rmd != "cpu_direct"))) {
    std::fprintf(stderr,
                 "requested route/build contract does not match (%s/%d/%d)\n",
                 compiled_route(), GGML_GEMMINI_ACTIVATION_BITS,
                 GGML_GEMMINI_TEST_IM2P_DIM);
    return 2;
  }

#if GGML_GEMMINI_ACTIVATION_QUANT == 0
  bool ok = true;
#if GGML_GEMMINI_ENABLE_RMD
  ok = run_integrated_geometry_oracle() && ok;
  ok = run_route_lifecycle_table() && ok;
  ok = run_exsia_full_success() && ok;
  ok = run_exsia_cross_mode_parity() && ok;
  ok = run_exsia_full_failure(TestFailure::collector_allocation) && ok;
  ok = run_exsia_full_collector_capture_failure() && ok;
  ok = run_exsia_full_failure(TestFailure::quantization) && ok;
  ok = run_exsia_full_failure(TestFailure::execute) && ok;
  ok = run_exsia_full_failure(TestFailure::fence) && ok;
  ok = run_exsia_full_failure(TestFailure::rmd) && ok;
  ok = run_exsia_success() && ok;
  ok = run_exsia_one_row_pipeline() && ok;
  ok = run_exsia_unsupported_mode() && ok;
  ok = run_exsia_start_failure() && ok;
  ok = run_exsia_boundary_failure(
           ggml::gemmini::im2p_adapter::TestFailure::quantization) &&
       ok;
  ok = run_exsia_boundary_failure(
           ggml::gemmini::im2p_adapter::TestFailure::provider) &&
       ok;
  ok = run_exsia_boundary_failure(
           ggml::gemmini::im2p_adapter::TestFailure::progress) &&
       ok;
  ok = run_exsia_boundary_failure(
           ggml::gemmini::im2p_adapter::TestFailure::poll) &&
       ok;
  ok = run_exsia_boundary_failure(
           ggml::gemmini::im2p_adapter::TestFailure::malformed_completion) &&
       ok;
  ok = run_exsia_boundary_failure(
           ggml::gemmini::im2p_adapter::TestFailure::incomplete_publication) &&
       ok;
  ok = run_exsia_staged_failure(
           ggml::gemmini::im2p_adapter::TestFailure::fence) &&
       ok;
  ok = run_exsia_staged_failure(TestFailure::rmd) && ok;
  ok = run_exsia_staged_failure(TestFailure::dense) && ok;
  ok = run_exsia_staged_failure(TestFailure::residual_execute) && ok;
  ok = run_exsia_staged_failure(TestFailure::compose) && ok;
  ok = run_exsia_staged_failure(TestFailure::output_authorization) && ok;
  ok = run_exsia_staged_failure(TestFailure::output_copy) && ok;
#if GGML_GEMMINI_ACTIVATION_BITS == 8
  ok = run_exsia_prestart_rejection(false) && ok;
#endif
#else
  ok = run_rmd_disabled_adapter_dense_only() && ok;
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
  for (const char *mode : {"FULL", "STRIPE_PIPELINE"}) {
    ok = run_success_mode(mode) && ok;
  }
  ok = run_malformed_contract() && ok;
  ok = run_fence_failure("FULL") && ok;
  ok = run_fence_failure("STRIPE_PIPELINE") && ok;

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
