#define GGML_GEMMINI_TEST_OBSERVER 1

#include "ggml-gemmini-args.h"
#include "ggml-gemmini-im2p.hpp"
#include "ggml-gemmini-matmul.hpp"

#include <cstdio>
#include <cstdlib>
#include <string_view>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace {

using namespace ggml::gemmini;

bool lifecycle_check(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
  }
  return condition;
}

size_t resident_bytes() {
#if defined(__APPLE__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
    return static_cast<size_t>(info.resident_size);
  }
#endif
  return 0;
}

ggml_gemmini_args_t lifecycle_args(std::vector<elem_t> &activation,
                                   std::vector<elem_t> &weights,
                                   std::vector<float> &output) {
  ggml_gemmini_args_t args{};
  args.I = 3;
  args.J = 2;
  args.K = 2;
  if (!args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS)) {
    std::abort();
  }
  for (size_t row = 0; row < args.I; ++row) {
    for (size_t column = 0; column < args.K; ++column) {
      if (!args.A.set(row, column, activation[row * args.K + column])) {
        std::abort();
      }
    }
  }
  args.B = weights.data();
  args.sA = args.K;
  args.sB = args.J;
  args.f_out = output.data();
  args.col_stride_f_out = 1;
  args.stride_f_out = args.J;
  args.weight_i8_scale_active = true;
  args.weight_scale = 1.0f;
  args.tiled_matmul_type = static_cast<tiled_matmul_type_t>(2);
  args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = {0, 0,
                                                                        0};
  return args;
}

bool test_stage_state_failures() {
  std::vector<elem_t> activation = {1, 2, 3, 4, 5, 6};
  std::vector<elem_t> weights = {1, -1, 2, 3};
  std::vector<float> output(6, 9876.0f);
  const auto sentinel = output;
  auto args = lifecycle_args(activation, weights, output);

  MatmulOptions invalid_options{};
  invalid_options.mode = MatmulInvocationMode::stripe_pipeline;
  invalid_options.job_capacity = 0;
  auto invalid_begin = prepare_execution(&args, invalid_options);
  bool ok = lifecycle_check(
      invalid_begin.status().code == MatmulStatusCode::invalid_argument,
      "begin/reset rejects zero capacity with a typed error");

  auto malformed = args;
  malformed.A = {};
  MatmulOptions full_options{};
  full_options.mode = MatmulInvocationMode::full;
  const auto activation_stage = matmul(malformed, full_options);
  ok = lifecycle_check(!activation_stage.ok(),
                       "activation staging rejects an absent buffer") &&
       ok;

  malformed = args;
  malformed.B = nullptr;
  const auto weight_stage = matmul(malformed, full_options);
  ok = lifecycle_check(!weight_stage.ok(),
                       "weight staging rejects an absent buffer") &&
       ok;

  MatmulOptions pipeline_options{};
  pipeline_options.mode = MatmulInvocationMode::stripe_pipeline;
  pipeline_options.job_capacity = 2;
  auto execution = prepare_execution(&args, pipeline_options);
  ok = lifecycle_check(execution.status().ok(),
                       "valid execution reaches the live lifecycle") &&
       ok;

  auto malformed_job = capture_stripe(execution, MatmulStripeInput(1, 1, 0));
  ok = lifecycle_check(malformed_job.status().code ==
                           MatmulStatusCode::invalid_argument,
                       "activation publication rejects empty bounds") &&
       ok;

  auto job = capture_stripe(execution, MatmulStripeInput(0, 1, 0));
  ok = lifecycle_check(job.status().ok(), "valid stripe stages") && ok;
  ok = lifecycle_check(compose_rmd_stripe(job).code ==
                           MatmulStatusCode::invalid_state,
                       "RMD compose rejects before execute") &&
       ok;
  ok = lifecycle_check(execute_rmd_stripe(job).ok(),
                       "RMD execute accepts one captured stripe") &&
       ok;
  ok = lifecycle_check(execute_rmd_stripe(job).code ==
                           MatmulStatusCode::invalid_state,
                       "RMD execute rejects duplicate execution") &&
       ok;
  ok = lifecycle_check(accept_external_dense_completion(job).ok(),
                       "external dense completion is accepted once") &&
       ok;
  ok = lifecycle_check(compose_rmd_stripe(job).ok(),
                       "RMD compose accepts executed residual state") &&
       ok;
  ok = lifecycle_check(finalize_stripe(job).ok(),
                       "RMD finalize accepts completed branches") &&
       ok;
  ok = lifecycle_check(compose_rmd_stripe(job).code ==
                           MatmulStatusCode::invalid_state,
                       "RMD compose rejects after finalize") &&
       ok;
  ok = lifecycle_check(finalize_stripe(job).code ==
                           MatmulStatusCode::invalid_state,
                       "RMD finalize rejects duplicate finalize") &&
       ok;
  ok = lifecycle_check(finish_execution(execution).code ==
                           MatmulStatusCode::invalid_contract,
                       "finish rejects a missing stripe range") &&
       ok;
  ok = lifecycle_check(output == sentinel,
                       "all stage failures preserve the caller sentinel") &&
       ok;
  return ok;
}

bool test_capability_matrix() {
  using namespace ggml::gemmini::im2p_adapter;
  struct CapabilityCase {
    const char *name;
    bool exsia;
    std::uint8_t activation_bits;
    std::uint8_t weight_bits;
    bool rmd_enabled;
    bool cpu_direct_rmd;
    Error expected;
  };
  constexpr CapabilityCase cases[] = {
      {"a8-q8-exsia", true, 8, 8, true, true, Error::success},
      {"a4-q8-exsia", true, 4, 8, true, true,
       Error::unsupported_route},
      {"a16-q8-exsia", true, 16, 8, true, true,
       Error::unsupported_route},
      {"a8-q4-exsia", true, 8, 4, true, true,
       Error::unsupported_route},
      {"a8-q16-exsia", true, 8, 16, true, true,
       Error::unsupported_route},
      {"a8-q12-exsia", true, 8, 12, true, true,
       Error::unsupported_route},
      {"a4-q8-non-exsia", false, 4, 8, false, false, Error::success},
      {"a16-q8-non-exsia", false, 16, 8, false, false, Error::success},
      {"a8-q4-non-exsia", false, 8, 4, false, false,
       Error::unsupported_route},
      {"a8-q16-non-exsia", false, 8, 16, false, false,
       Error::unsupported_route},
  };

  std::vector<float> output(6, 9876.0f);
  const auto sentinel = output;
  test_reset();
  for (const auto &test : cases) {
    const Result result = gate_route(test.exsia, test.activation_bits,
                                     test.rmd_enabled, test.cpu_direct_rmd,
                                     test.weight_bits);
    if (!lifecycle_check(result.error == test.expected, test.name)) {
      return false;
    }
  }
  const TestCounters counters = test_counters();
  return lifecycle_check(
             output == sentinel,
             "capability rejections preserve the caller sentinel") &&
         lifecycle_check(
             counters.full == 0 && counters.pipeline == 0 &&
                 counters.fence == 0 && counters.stripe == 0 &&
                 counters.rmd_calls == 0 && counters.authorize == 0 &&
                 counters.commit == 0 && counters.hardware == 0 &&
                 counters.fallback == 0 && counters.live_runs == 0,
             "capability gate rejects before allocation, execution, RMD, or fallback");
}

bool test_full_handle_state() {
  using namespace ggml::gemmini::im2p_adapter;
  std::vector<elem_t> activation = {1, 2, 3, 4, 5, 6};
  std::vector<elem_t> weights = {1, -1, 2, 3};
  std::vector<float> output(6, 9876.0f);
  const auto sentinel = output;
  auto args = lifecycle_args(activation, weights, output);
  args.activation_rows_per_stripe = 1;

  test_reset();
  auto started = start_exsia_full_execution(args);
  bool ok = lifecycle_check(started.result.ok() && started.execution,
                            "FULL transaction allocates without a worker");
  ok = lifecycle_check(started.execution->install_sink().ok() &&
                           args.exsia_stripe_ready_sink != nullptr,
                       "FULL collector installs before quantization") && ok;
  ok = lifecycle_check(started.execution->install_sink().error ==
                           Error::invalid_state,
                       "FULL collector rejects duplicate installation") && ok;
  ok = lifecycle_check(!started.execution->finish(false).result.ok() &&
                           args.exsia_stripe_ready_sink == nullptr,
                       "FULL cancellation clears its borrowed sink") && ok;
  ok = lifecycle_check(started.execution->finish(false).result.error ==
                           Error::invalid_state,
                       "FULL transaction rejects stale repeated finish") && ok;
  started.execution.reset();

  auto abandoned = start_exsia_full_execution(args);
  ok = lifecycle_check(abandoned.result.ok() && abandoned.execution &&
                           abandoned.execution->install_sink().ok(),
                       "FULL ownership can be reacquired after cancellation") && ok;
  abandoned.execution.reset();
  const auto counters = test_counters();
  return lifecycle_check(args.exsia_stripe_ready_sink == nullptr,
                         "FULL destructor releases only its sink ownership") &&
         lifecycle_check(output == sentinel,
                         "FULL cancellation leaves partial output hidden") &&
         lifecycle_check(counters.full == 0 && counters.pipeline == 0 &&
                             counters.fence == 0 && counters.commit == 0,
                         "FULL cancellation creates no simulator run") && ok;
}

bool test_invalid_reuse() {
  using namespace ggml::gemmini::im2p_adapter;
  test_reset();
  const size_t before = resident_bytes();
  for (size_t iteration = 0; iteration < 1000; ++iteration) {
    const Result rejected = gate_route(true, 16, true, true, 8);
    if (!lifecycle_check(
            rejected.error == Error::unsupported_route,
            "invalid backend route returns one stable typed error")) {
      return false;
    }
  }
  const size_t after = resident_bytes();
  const TestCounters counters = test_counters();
  return lifecycle_check(
             counters.full == 0 && counters.pipeline == 0 &&
                 counters.fence == 0 && counters.stripe == 0 &&
                 counters.rmd_calls == 0 && counters.hardware == 0 &&
                 counters.fallback == 0 && counters.live_runs == 0,
             "1000 early rejections allocate no run or fallback resource") &&
         lifecycle_check(before == 0 || after <= before + 8 * 1024 * 1024,
                         "1000 early rejections keep resident memory bounded");
}

bool run_routing_suite() {
#if defined(GGML_GEMMINI_ROUTING_BINARY)
  const std::string command =
      std::string("\"") + GGML_GEMMINI_ROUTING_BINARY + "\"";
  return lifecycle_check(std::system(command.c_str()) == 0,
                         "real routing child exits successfully");
#else
  return lifecycle_check(false, "routing binary path is unavailable");
#endif
}

void print_help(const char *program) {
  std::printf(
      "usage: %s [--case all|routing|stage-failures|capabilities|full-state|invalid-reuse]\n",
      program);
}

} // namespace

int main(int argc, char **argv) {
  std::string_view selected = "all";
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    print_help(argv[0]);
    return 0;
  }
  if (argc == 3 && std::string_view(argv[1]) == "--case") {
    selected = argv[2];
  } else if (argc != 1) {
    print_help(argv[0]);
    return 2;
  }

  bool ok = false;
  if (selected == "all") {
    ok = test_stage_state_failures() && test_capability_matrix() &&
         test_full_handle_state() && test_invalid_reuse() && run_routing_suite();
  } else if (selected == "routing") {
    ok = run_routing_suite();
  } else if (selected == "stage-failures") {
    ok = test_stage_state_failures();
  } else if (selected == "capabilities") {
    ok = test_capability_matrix();
  } else if (selected == "full-state") {
    ok = test_full_handle_state();
  } else if (selected == "invalid-reuse") {
    ok = test_invalid_reuse();
  } else {
    std::fprintf(stderr, "unknown lifecycle case: %.*s\n",
                 static_cast<int>(selected.size()), selected.data());
    print_help(argv[0]);
    return 2;
  }
  if (ok) {
    std::printf("IM2P lifecycle case %.*s: PASS\n",
                static_cast<int>(selected.size()), selected.data());
  }
  return ok ? 0 : 1;
}
