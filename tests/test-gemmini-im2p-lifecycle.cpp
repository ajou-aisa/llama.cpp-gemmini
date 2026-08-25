#define GGML_GEMMINI_TEST_OBSERVER 1

#include "ggml-gemmini-args.h"
#include "ggml-gemmini-im2p.hpp"
#include "ggml-gemmini-matmul.hpp"
#include "im2p_gemmini_frontend.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

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
  (void) weights;
  args.I = 3;
  args.J = 2;
  args.K = 32;
  if (!args.A.allocate(args.I, args.K, GGML_GEMMINI_ACTIVATION_BITS)) {
    std::abort();
  }
  for (size_t row = 0; row < args.I; ++row) {
    for (size_t column = 0; column < args.K; ++column) {
      if (!args.A.set(row, column,
                      activation[(row * args.K + column) % activation.size()])) {
        std::abort();
      }
    }
  }
#if GGML_GEMMINI_ACTIVATION_BITS == 4
  static std::vector<block_q4_h1> native_weights(2);
  args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_h1;
  args.q4_h1_blocks = native_weights.data();
  args.native_block_count = native_weights.size();
  args.native_blocks_per_row = 1;
  args.native_weight_bytes = native_weights.size() * sizeof(block_q4_h1);
#elif GGML_GEMMINI_ACTIVATION_BITS == 16
  static std::vector<block_q16_h1> native_weights(2);
  args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q16_h1;
  args.q16_h1_blocks = native_weights.data();
  args.native_block_count = native_weights.size();
  args.native_blocks_per_row = 1;
  args.native_weight_bytes = native_weights.size() * sizeof(block_q16_h1);
#else
  args.B = weights.data();
#endif
  args.sA = args.K;
  args.sB = args.J;
  args.tile_I = 1;
  args.tile_J = 1;
  args.tile_K = 1;
  args.activation_rows_per_stripe = DIM;
  args.f_out = output.data();
  args.col_stride_f_out = 1;
  args.stride_f_out = args.J;
  args.weight_i8_scale_active = true;
  args.weight_scale = 1.0f;
  args.tiled_matmul_type = static_cast<tiled_matmul_type_t>(2);
  args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = {0};
  return args;
}

bool test_transactional_stripe_log_failures() {
#if defined(GGML_GEMMINI_ROUTING_BINARY)
  constexpr std::array<const char *, 12> selectors{{
      "full", "quantization", "provider", "progress", "poll", "fence",
      "rmd", "dense", "output-authorization", "malformed-completion",
      "incomplete-publication", "output-copy"}};
  for (const char *selector : selectors) {
    std::error_code error;
    std::filesystem::remove_all("output", error);
    if (error) {
      return lifecycle_check(false, "clear task-owned cycle log root");
    }
    const std::string command = std::string("\"") +
        GGML_GEMMINI_ROUTING_BINARY + "\" --case " + selector;
    if (!lifecycle_check(std::system(command.c_str()) == 0,
                         "production-linked failure selector passes")) {
      return false;
    }
    std::ifstream input("output/log/cycle-log.jsonl", std::ios::binary);
    const std::string log((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
    if (!lifecycle_check(
            log.find("\"record_type\":\"IM2P_STRIPE_TELEMETRY\"") ==
                    std::string::npos &&
                log.find("\"record_type\":\"QUANTIZATION_STRIPE_TELEMETRY\"") ==
                    std::string::npos,
            "FULL and failed transactions emit zero RTL or quantization stripe rows")) {
      return false;
    }
  }
  std::error_code error;
  std::filesystem::remove_all("output", error);
  return lifecycle_check(!error, "clean task-owned failure cycle logs");
#else
  return lifecycle_check(false, "routing binary path is unavailable");
#endif
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
  invalid_options.rmd_backend = RmdBackend::cpu_direct;
  auto invalid_begin = prepare_execution(&args, invalid_options);
  bool ok = lifecycle_check(
      invalid_begin.status().code == MatmulStatusCode::invalid_argument,
      "begin/reset rejects zero capacity with a typed error");

  auto malformed = args;
  malformed.A = {};
  MatmulOptions full_options{};
  full_options.mode = MatmulInvocationMode::full;
  full_options.rmd_backend = RmdBackend::cpu_direct;
  const auto activation_stage = matmul(malformed, full_options);
  ok = lifecycle_check(!activation_stage.ok(),
                       "activation staging rejects an absent buffer") &&
       ok;

  malformed = args;
#if GGML_GEMMINI_ACTIVATION_BITS == 4
  malformed.q4_h1_blocks = nullptr;
#elif GGML_GEMMINI_ACTIVATION_BITS == 16
  malformed.q16_h1_blocks = nullptr;
#else
  malformed.B = nullptr;
#endif
  const auto weight_stage = matmul(malformed, full_options);
  ok = lifecycle_check(!weight_stage.ok(),
                       "weight staging rejects an absent buffer") &&
       ok;

  MatmulOptions pipeline_options{};
  pipeline_options.mode = MatmulInvocationMode::stripe_pipeline;
  pipeline_options.job_capacity = 2;
  pipeline_options.rmd_backend = RmdBackend::cpu_direct;
  auto execution = prepare_execution(&args, pipeline_options);
  if (!execution.status().ok()) {
    std::fprintf(stderr, "stage fixture prepare failed: %s\n",
                 execution.status().message);
  }
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
  return test_transactional_stripe_log_failures() && ok;
}

bool test_capability_matrix() {
  using namespace ggml::gemmini::im2p_adapter;
  constexpr std::array<std::uint8_t, 3> widths{{4, 8, 16}};
  constexpr std::array<PublicMode, 2> modes{{PublicMode::full,
                                             PublicMode::stripe_pipeline}};
  constexpr std::array<WeightFamily, 5> families{{
      WeightFamily::h0, WeightFamily::h1, WeightFamily::hp1,
      WeightFamily::h2, WeightFamily::hp2}};
  constexpr std::array<ResidualBackend, 2> backends{{
      ResidualBackend::cpu_direct, ResidualBackend::compact_ws}};

  std::vector<float> output(6, 9876.0f);
  const auto sentinel = output;
  std::size_t accepted = 0;
  test_reset();
  for (const auto activation_bits : widths) {
    for (const auto weight_bits : widths) {
      for (const auto mode : modes) {
        for (const auto family : families) {
          for (const auto backend : backends) {
            ExsiaRouteRequest request{true, activation_bits, weight_bits,
                                      activation_bits, weight_bits, true, mode,
                                      family, backend,
                                      BuildIdentity::im2p_sim_ws};
            const bool matched = activation_bits == weight_bits;
            const bool supported_family = family == WeightFamily::h0 ||
                                          family == WeightFamily::h1 ||
                                          family == WeightFamily::hp1;
            const bool supported_backend = family != WeightFamily::h0 ||
                                           backend == ResidualBackend::cpu_direct;
            const bool expected = matched && supported_family && supported_backend;
            const Result result = gate_route(request);
            if (!lifecycle_check(result.ok() == expected,
                                 "exhaustive capability identity")) {
              return false;
            }
            if (result.ok()) ++accepted;
          }
        }
      }
    }
  }
  ExsiaRouteRequest mismatch{true, 8, 8, 4, 8, true, PublicMode::full,
                             WeightFamily::h1, ResidualBackend::cpu_direct,
                             BuildIdentity::im2p_sim_ws};
  const Result artifact_rejected = gate_route(mismatch);
  mismatch.artifact_activation_bits = 8;
  mismatch.build_identity = BuildIdentity::hardware_os;
  const Result os_rejected = gate_route(mismatch);

  const TestCounters counters = test_counters();
  const bool ok =
      lifecycle_check(accepted == 30,
                      "capability matrix accepts exactly 30 routes") &&
      lifecycle_check(artifact_rejected.error == Error::invalid_contract &&
                          os_rejected.error == Error::unsupported_route,
                      "artifact and OS identities fail closed") &&
      lifecycle_check(output == sentinel,
                      "capability rejections preserve the caller sentinel") &&
      lifecycle_check(
          counters.activation_allocations == 0 && counters.worker_starts == 0 &&
              counters.full == 0 && counters.pipeline == 0 &&
              counters.fence == 0 && counters.stripe == 0 &&
              counters.accepted_stripes == 0 && counters.rmd_calls == 0 &&
              counters.rmd_events == 0 && counters.authorize == 0 &&
              counters.commit == 0 && counters.hardware == 0 &&
              counters.fallback == 0 && counters.live_runs == 0 &&
              counters.blocked_producers == 0,
          "all capability rejections expose zero side-effect counters");
  if (ok) {
    std::printf("ROUTE_MATRIX accepted=%zu allocation=%llu worker_start=%llu "
                "fence=%llu rmd=%llu commit=%llu fallback=%llu hardware=%llu "
                "live_runs=%llu\n",
                accepted,
                static_cast<unsigned long long>(counters.activation_allocations),
                static_cast<unsigned long long>(counters.worker_starts),
                static_cast<unsigned long long>(counters.fence),
                static_cast<unsigned long long>(counters.rmd_calls),
                static_cast<unsigned long long>(counters.commit),
                static_cast<unsigned long long>(counters.fallback),
                static_cast<unsigned long long>(counters.hardware),
                static_cast<unsigned long long>(counters.live_runs));
  }
  return ok;
}

bool test_full_handle_state() {
  using namespace ggml::gemmini::im2p_adapter;
  std::vector<elem_t> activation = {1, 2, 3, 4, 5, 6};
  std::vector<elem_t> weights = {1, -1, 2, 3};
  std::vector<float> output(6, 9876.0f);
  const auto sentinel = output;
  auto args = lifecycle_args(activation, weights, output);

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

bool test_publication_geometry() {
  using namespace ggml::gemmini::im2p_adapter;

  struct FixtureResult {
    Completion full;
    Completion pipeline;
    TestCounters counters;
  };
  const auto execute = [](size_t rows, size_t tile_i, size_t k) {
    std::vector<elem_t> activation(rows * k, elem_t{1});
    const size_t blocks_per_row = (k + 31) / 32;
#if GGML_GEMMINI_ACTIVATION_BITS == 4
    std::vector<block_q4_h1> weights(DIM * blocks_per_row);
#elif GGML_GEMMINI_ACTIVATION_BITS == 16
    std::vector<block_q16_h1> weights(DIM * blocks_per_row);
#else
    std::vector<block_q8_h1> weights(DIM * blocks_per_row);
#endif
    for (auto &weight : weights) {
      weight.s_rf = 1.0f;
      weight.c_b = 1;
      weight.R = 0;
#if GGML_GEMMINI_ACTIVATION_BITS == 4
      std::fill(std::begin(weight.qs), std::end(weight.qs), uint8_t{0x99});
#elif GGML_GEMMINI_ACTIVATION_BITS == 16
      std::fill(std::begin(weight.qs), std::end(weight.qs), int16_t{1});
#else
      std::fill(std::begin(weight.qs), std::end(weight.qs), int8_t{1});
#endif
    }
    std::vector<float> output(rows * DIM, 4321.0f);
    ggml_gemmini_args_t args{};
    args.I = rows;
    args.J = DIM;
    args.K = k;
    if (!args.A.allocate(rows, k, GGML_GEMMINI_ACTIVATION_BITS)) {
      std::abort();
    }
    for (size_t row = 0; row < rows; ++row) {
      for (size_t column = 0; column < k; ++column) {
        if (!args.A.set(row, column, activation[row * k + column])) {
          std::abort();
        }
      }
    }
    args.sA = k;
    args.sB = DIM;
    args.f_out = output.data();
    args.stride_f_out = DIM;
    args.col_stride_f_out = 1;
    args.tile_I = tile_i;
    args.tile_J = 1;
    args.tile_K = 1;
    args.activation_rows_per_stripe = tile_i * DIM;
#if GGML_GEMMINI_ACTIVATION_BITS == 4
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_h1;
    args.q4_h1_blocks = weights.data();
#elif GGML_GEMMINI_ACTIVATION_BITS == 16
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q16_h1;
    args.q16_h1_blocks = weights.data();
#else
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = weights.data();
    args.q8_h1_block_count = weights.size();
    args.q8_h1_rows = DIM;
    args.blocks_per_row = blocks_per_row;
#endif
    args.native_block_count = weights.size();
    args.native_blocks_per_row = blocks_per_row;
    args.native_weight_bytes = weights.size() * sizeof(weights[0]);
    args.act_quant.storage()
        .emplace<quants::act::tensor::Meta>()
        .scale = 1.0f;

    test_reset();
    FixtureResult result{};
    result.full = run_full(args);
    result.pipeline = run_stripe_pipeline(args);
    result.counters = test_counters();
    return result;
  };

  const auto a_k1 = execute(2 * DIM + 1, 1, 2 * DIM);
  const auto a_k3 = execute(2 * DIM + 1, 1, 4 * DIM);
  const auto b_k1 = execute(16 * DIM, 5, 2 * DIM);
  const auto b_k3 = execute(16 * DIM, 5, 4 * DIM);
  const auto rows = [](const TestCounters &counters, size_t index) {
    return counters.stripe_row_end[index] - counters.stripe_row_begin[index];
  };
  const auto deterministic = [&](const TestCounters &counters,
                                 size_t publications) {
    if (counters.stripe_trace_size != publications)
      return false;
    for (size_t index = 0; index < publications; ++index) {
      if (counters.stripe_ids[index] != static_cast<int>(index) ||
          counters.slot_ids[index] != static_cast<int>(index % 2))
        return false;
    }
    return true;
  };

  if (!a_k1.full.result.ok() || !a_k1.pipeline.result.ok() ||
      !b_k1.full.result.ok() || !b_k1.pipeline.result.ok()) {
    std::fprintf(stderr,
                 "publication fixture failure A=%s/%s B=%s/%s\n",
                 a_k1.full.result.message, a_k1.pipeline.result.message,
                 b_k1.full.result.message, b_k1.pipeline.result.message);
  }
  bool ok =
      lifecycle_check(a_k1.full.result.ok() && a_k1.pipeline.result.ok() &&
                          b_k1.full.result.ok() && b_k1.pipeline.result.ok(),
                      "canonical FULL and PIPELINE fixtures complete") &&
      lifecycle_check(a_k1.full.stats.rtl_stripes_published == 0 &&
                          a_k1.full.stats.rtl_stripe_rows_published == 0 &&
                          b_k1.full.stats.rtl_stripes_published == 0 &&
                          b_k1.full.stats.rtl_stripe_rows_published == 0,
                      "FULL publishes exactly zero stripes and rows") &&
      lifecycle_check(a_k1.pipeline.stats.rtl_stripes_published == 3 &&
                          a_k1.pipeline.stats.rtl_stripe_rows_published ==
                              2 * DIM + 1 &&
                          rows(a_k1.counters, 0) == DIM &&
                          rows(a_k1.counters, 1) == DIM &&
                          rows(a_k1.counters, 2) == 1,
                      "fixture A publishes literal rows [DIM,DIM,1]") &&
      lifecycle_check(b_k1.pipeline.stats.rtl_stripes_published == 4 &&
                          b_k1.pipeline.stats.rtl_stripe_rows_published ==
                              16 * DIM &&
                          rows(b_k1.counters, 0) == 5 * DIM &&
                          rows(b_k1.counters, 1) == 5 * DIM &&
                          rows(b_k1.counters, 2) == 5 * DIM &&
                          rows(b_k1.counters, 3) == DIM,
                      "fixture B publishes literal rows [5*DIM,5*DIM,5*DIM,DIM]") &&
      lifecycle_check(deterministic(a_k1.counters, 3) &&
                          deterministic(b_k1.counters, 4),
                      "publication ids and two-slot order are deterministic") &&
      lifecycle_check(a_k3.pipeline.stats.rtl_stripes_published == 3 &&
                          b_k3.pipeline.stats.rtl_stripes_published == 4 &&
                          a_k3.pipeline.stats.rtl_completed_output_works == 3 &&
                          b_k3.pipeline.stats.rtl_completed_output_works == 16 &&
                          a_k3.pipeline.stats.rtl_completed_fragments == 12 &&
                          b_k3.pipeline.stats.rtl_completed_fragments == 64 &&
                          a_k3.pipeline.stats.rtl_scheduler_groups_completed == 3 &&
                          b_k3.pipeline.stats.rtl_scheduler_groups_completed == 4,
                      "works, scheduler groups, fragments, and publications remain distinct") &&
      lifecycle_check(a_k1.pipeline.stats.rtl_completed_fragments <
                              a_k3.pipeline.stats.rtl_completed_fragments &&
                          b_k1.pipeline.stats.rtl_completed_fragments <
                              b_k3.pipeline.stats.rtl_completed_fragments,
                      "K fragments change fragment stats but never publication count");
  if (ok) {
    std::printf("PUBLICATION_GEOMETRY A_rows=[%d,%d,1] B_rows=[%d,%d,%d,%d] "
                "A_works/groups/fragments=3/3/12 "
                "B_works/groups/fragments=16/4/64 full_publications=0 "
                "slots=A[0,1,0],B[0,1,0,1]\n",
                DIM, DIM, 5 * DIM, 5 * DIM, 5 * DIM, DIM);
  }
  return ok;
}

bool test_stats_contract_failures() {
  using namespace ggml::gemmini::im2p_adapter;
  std::vector<float> output(6, 9876.0f);
  const auto sentinel = output;
  test_reset();

  ::im2p::gemmini::FenceResult full_source{};
  full_source.stats.base.compute_cycles = 71;
  full_source.stats.base.completed_output_tiles = 72;
  full_source.stats.base.completed_fragments = 73;
  full_source.stats.base.completed_stripes = 74;
  full_source.stats.base.stripes_published = 1;
  full_source.stats.base.stripe_rows_published = 3;
  const Completion full =
      translate(full_source, ::im2p::gemmini::Mode::full, 0, 0);

  auto pipeline_count_source = full_source;
  pipeline_count_source.stats.base.stripes_published = 2;
  const Completion pipeline_count = translate(
      pipeline_count_source, ::im2p::gemmini::Mode::stripe_pipeline, 3, 3);

  auto pipeline_rows_source = full_source;
  pipeline_rows_source.stats.base.stripes_published = 3;
  pipeline_rows_source.stats.base.stripe_rows_published = 2;
  const Completion pipeline_rows = translate(
      pipeline_rows_source, ::im2p::gemmini::Mode::stripe_pipeline, 3, 3);

  const TestCounters counters = test_counters();
  const bool ok =
      lifecycle_check(full.result.error == Error::invalid_contract,
                      "FULL rejects nonzero publication statistics") &&
      lifecycle_check(pipeline_count.result.error == Error::invalid_contract,
                      "PIPELINE rejects a noncanonical publication count") &&
      lifecycle_check(pipeline_rows.result.error == Error::invalid_contract,
                      "PIPELINE rejects a noncanonical published-row count") &&
      lifecycle_check(output == sentinel,
                      "impossible statistics preserve the output sentinel") &&
      lifecycle_check(counters.authorize == 0 && counters.commit == 0,
                      "statistics reject before authorization and commit");
  if (ok) {
    std::printf("IM2P_STATS_FAILURE full_count=invalid_contract "
                "pipeline_count=invalid_contract pipeline_rows=invalid_contract "
                "other_stats=71,72,73,74 sentinel=preserved "
                "authorize=0 commit=0\n");
  }
  return ok;
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
      "usage: %s [--case all|routing|stage-failures|capabilities|full-state|publication-geometry|stats-invariants|invalid-reuse]\n",
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
         test_full_handle_state() && test_publication_geometry() &&
         test_stats_contract_failures() &&
         test_invalid_reuse() && run_routing_suite();
  } else if (selected == "routing") {
    ok = run_routing_suite();
  } else if (selected == "stage-failures") {
    ok = test_stage_state_failures();
  } else if (selected == "capabilities") {
    ok = test_capability_matrix();
  } else if (selected == "full-state") {
    ok = test_full_handle_state();
  } else if (selected == "publication-geometry") {
    ok = test_publication_geometry();
  } else if (selected == "stats-invariants") {
    ok = test_stats_contract_failures();
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
