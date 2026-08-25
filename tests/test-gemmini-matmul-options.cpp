#include "ggml-gemmini-matmul.hpp"
#include "ggml-gemmini-geometry.hpp"
#include "quants/act/exsia/exsia.hpp"

#include <ggml.h>

#include <gemmini.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

using namespace ggml::gemmini;

static_assert(std::is_same_v<decltype(ggml_gemmini_args_t::matmul_layer), std::string>,
              "matmul_layer must be owned string storage");

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

struct SemanticLayerObservations {
    std::array<size_t, 6> counts{};
    std::array<std::string, 6> layers{};
};

int semantic_layer_mutation_site = -1;

bool observe_semantic_layer(TestSemanticLayerSite site, const char * layer, void * user_data) {
    auto & observations = *static_cast<SemanticLayerObservations *>(user_data);
    const size_t index = static_cast<size_t>(site);
    if (index < observations.counts.size()) {
        ++observations.counts[index];
        observations.layers[index] = semantic_layer_mutation_site == static_cast<int>(index)
            ? "wrong.constant" : (layer != nullptr ? layer : "");
    }
    return site != TestSemanticLayerSite::fp_facade;
}

void clear_environment() {
    unsetenv("GEMMINI_MATMUL_MODE");
    unsetenv("GEMMINI_STRIPE_JOB_CAPACITY");
    unsetenv("GEMMINI_RMD_BACKEND");
}

bool test_owned_matmul_layer_lifetime() {
    const std::string expected = "blk.15.mlp.down_proj.semantic-layer-owned-beyond-sso";
    ggml_gemmini_args_t copied;
    ggml_gemmini_args_t moved;
    {
        ggml_gemmini_args_t source;
        {
            std::string semantic_layer = expected;
            source.matmul_layer = semantic_layer;
        }
        copied = source;
        moved = std::move(source);
    }

    const auto pipeline = ggml::gemmini::detail::pipeline_stripe_telemetry(
        copied.matmul_layer.c_str(), {});
    return check(copied.matmul_layer == expected,
                 "copied args own semantic layer after source destruction") &&
        check(moved.matmul_layer == expected,
              "moved args own semantic layer after source destruction") &&
        check(pipeline.layer == expected,
              "pipeline summary owns byte-identical semantic layer");
}

bool test_quantization_and_exsia_semantic_layer_seam() {
    ggml_gemmini_args_t args{};
    args.matmul_layer = "blk.15.mlp.down_proj";
    int capture[2]{};
    const int saved_stderr = dup(STDERR_FILENO);
    if (!check(saved_stderr >= 0 && pipe(capture) == 0,
               "quantization semantic seam capture opens")) {
        if (saved_stderr >= 0) close(saved_stderr);
        return false;
    }
    std::fflush(stderr);
    dup2(capture[1], STDERR_FILENO);
    close(capture[1]);
    const bool quantized = ggml::gemmini::quants::act::quantize(nullptr, args);
    std::fflush(stderr);
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);
    char captured[2048]{};
    const ssize_t count = read(capture[0], captured, sizeof(captured) - 1);
    close(capture[0]);
    if (count > 0) captured[count] = '\0';

    return check(!quantized, "malformed quantization input keeps failure behavior") &&
        check(args.matmul_layer == "blk.15.mlp.down_proj",
              "ExSIA keeps owned semantic layer unchanged") &&
        check(std::strstr(captured, "\"layer\":\"blk.15.mlp.down_proj\"") != nullptr,
              "quantization failure emits byte-identical semantic layer");
}

bool test_backend_semantic_resolution_and_dedupe() {
    ggml::gemmini::test_reset_unclassified_matmul_diagnostics();
    constexpr size_t thread_count = 16;
    std::vector<std::string> labels(thread_count);
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (size_t thread = 0; thread < thread_count; ++thread) {
        threads.emplace_back([&, thread] {
            labels[thread] = ggml::gemmini::test_resolve_backend_matmul_layer(
                "unknown", "bad weight/", "input", "consumer");
        });
    }
    for (std::thread & thread : threads) thread.join();

    bool ok = true;
    for (const std::string & label : labels) {
        ok = check(label == "unclassified.bad_weight_",
                   "concurrent malformed setup keeps bounded fallback") && ok;
    }
    ok = check(ggml::gemmini::test_unclassified_matmul_diagnostic_count() == 1,
               "concurrent duplicate fallback emits one diagnostic") && ok;
    const std::string canonical = ggml::gemmini::test_resolve_backend_matmul_layer(
        "llama", "blk.15.ffn_down.weight", "ffn", "consumer");
    ok = check(canonical == "blk.15.mlp.down_proj",
               "backend setup resolves trusted semantic tuple once") && ok;
    ok = check(ggml::gemmini::test_unclassified_matmul_diagnostic_count() == 1,
               "canonical setup emits no fallback diagnostic") && ok;
    (void) ggml::gemmini::test_resolve_backend_matmul_layer(
        "unknown", "bad weight/", "input", "second-consumer");
    ok = check(ggml::gemmini::test_unclassified_matmul_diagnostic_count() == 2,
               "different consumer emits one additional diagnostic") && ok;

    ggml::gemmini::test_reset_unclassified_matmul_diagnostics();
    for (size_t index = 0; index < 96; ++index) {
        (void) ggml::gemmini::test_resolve_backend_matmul_layer(
            "unknown", "bad weight/" + std::to_string(index), "input", "consumer");
    }
    ok = check(ggml::gemmini::test_unclassified_matmul_diagnostic_count() == 96,
                "every unique fallback tuple emits one diagnostic") && ok;
    for (size_t index = 0; index < 96; ++index) {
        (void) ggml::gemmini::test_resolve_backend_matmul_layer(
            "unknown", "bad weight/" + std::to_string(index), "input", "consumer");
    }
    ok = check(ggml::gemmini::test_unclassified_matmul_diagnostic_count() == 96,
                "duplicate fallback tuples emit no second diagnostic") && ok;
    ggml::gemmini::test_reset_unclassified_matmul_diagnostics();
    return ok;
}

bool test_args_layout_extension() {
    ggml_gemmini_args_t args;
    const auto * base = reinterpret_cast<const uint8_t *>(&args);
    const auto offset = [base](const auto * member) {
        return static_cast<size_t>(reinterpret_cast<const uint8_t *>(member) - base);
    };

    return check(sizeof(args) > 1032, "owned semantic layer increases args size") &&
        check(offset(&args.native_weight_bytes) == 848,
              "native_weight_bytes offset remains unchanged") &&
        check(offset(&args.col_stride_f_out) == 952,
              "col_stride_f_out offset remains unchanged") &&
        check(offset(&args.stride_f_out) == 960,
              "stride_f_out offset remains unchanged") &&
        check(offset(&args.tile_I) == 984, "tile_I offset remains unchanged");
}

ggml_gemmini_args_t make_args(std::vector<elem_t> & activation,
                              std::vector<elem_t> & weights,
                              std::vector<float> & output) {
    ggml_gemmini_args_t args{};
    args.I = 3;
    args.J = 2;
    args.K = 2;
    args.A.allocate(args.I, args.K, 8);
    for (size_t i = 0; i < args.I * args.K; ++i) {
        args.A.set(i / args.K, i % args.K, activation[i]);
    }
    args.B = weights.data();
    args.sA = args.K;
    args.sB = args.J;
    args.f_out = output.data();
    args.col_stride_f_out = 1;
    args.stride_f_out = args.J;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    args.tiled_matmul_type = CPU;
    args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    return args;
}

bool test_staged_exsia_host_pipeline_semantic_layer() {
    constexpr size_t stripe_rows = DIM;
    constexpr size_t rows = 2 * stripe_rows + 1;
    constexpr size_t columns = 2;
    constexpr size_t depth = 2 * DIM;
    const std::string semantic_layer =
        "blk.15.mlp.down_proj.host-pipeline-lifetime-beyond-sso";
    std::vector<float> activations(rows * depth);
    std::fill(activations.begin(), activations.end(), 1.0f);
    std::vector<elem_t> weights(depth * columns, elem_t{1});
    std::vector<float> output(rows * columns, 0.0f);

    ggml_init_params params{
        ggml_tensor_overhead() * 2 + activations.size() * sizeof(float) + 1024,
        nullptr,
        false,
    };
    ggml_context * context = ggml_init(params);
    if (!check(context != nullptr, "host pipeline activation context initializes")) {
        return false;
    }
    ggml_tensor * activation =
        ggml_new_tensor_2d(context, GGML_TYPE_F32, depth, rows);
    std::memcpy(activation->data, activations.data(),
                activations.size() * sizeof(float));

    ggml_gemmini_args_t args{};
    {
        std::string source = semantic_layer;
        args.matmul_layer = source;
    }
    args.I = rows;
    args.J = columns;
    args.K = depth;
    args.sA = depth;
    args.sB = columns;
    args.B = weights.data();
    args.f_out = output.data();
    args.col_stride_f_out = 1;
    args.stride_f_out = columns;
    args.tiled_matmul_type = CPU;
    args.tile_I = 1;
    args.tile_J = 1;
    args.tile_K = 2;
    args.activation_rows_per_stripe = stripe_rows;
    args.residual_route = residual::ResidualRoute::cpu_direct;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    if (!args.A.allocate(rows, depth, 8)) {
        ggml_free(context);
        return check(false, "host pipeline activation storage allocates");
    }
    args.act_quant.storage().emplace<quants::act::exsia::Meta>();

    ResolvedMatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 3;
    options.rmd_backend = RmdBackend::cpu_direct;
    options.profiling = true;
    auto execution = prepare_execution(
        static_cast<const ggml_gemmini_args_t &>(args), options);
    MatmulStripeCollector collector(3);
    if (!check(execution.status().ok() && collector.start(execution),
               "real staged ExSIA host pipeline starts")) {
        ggml_free(context);
        return false;
    }

    ggml_gemmini_args_t quant_args = args;
    args.matmul_layer = "mutated.after.staged-execution-copy";
    auto & meta = std::get<quants::act::exsia::Meta>(
        quant_args.act_quant.storage());
    quants::act::exsia::ExSIA exsia;
    exsia.set_execution_mode(
        quants::act::exsia::ExSIAState::ExecutionMode::Sequential);
    SemanticLayerObservations observations;
    set_test_semantic_layer_observer(observe_semantic_layer, &observations);

    FILE * capture = std::tmpfile();
    const int saved_stderr = dup(STDERR_FILENO);
    bool capture_ok = capture != nullptr && saved_stderr >= 0;
    if (capture_ok) {
        std::fflush(stderr);
        capture_ok = dup2(fileno(capture), STDERR_FILENO) >= 0;
    }
    const bool quantized = capture_ok && exsia.run(
        meta, activation, quant_args, collector.sink());
    const MatmulStatus collected = collector.finish();
    const MatmulStatus completed = finish_execution(execution);
    std::fflush(stderr);
    if (saved_stderr >= 0) {
        dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);
    }
    set_test_semantic_layer_observer(nullptr, nullptr);

    std::string debug_output;
    if (capture != nullptr) {
        std::rewind(capture);
        char chunk[1024];
        while (const size_t count =
                   std::fread(chunk, 1, sizeof(chunk), capture)) {
            debug_output.append(chunk, count);
        }
        std::fclose(capture);
    }
    ggml_free(context);

    const auto profiles = collector.profiles();
    if (!quantized || !collected.ok() || !completed.ok()) {
        std::fprintf(stderr,
                     "host pipeline capture: %s quantized=%d collector=%u/%s completed=%u/%s profiles=%zu\n",
                     debug_output.c_str(), quantized ? 1 : 0,
                     static_cast<unsigned>(collected.code), collected.message,
                     static_cast<unsigned>(completed.code),
                     completed.message, profiles.size());
    }
    std::vector<PipelineStripeTelemetry> summaries;
    summaries.reserve(profiles.size());
    for (const auto & profile : profiles) {
        summaries.push_back(detail::pipeline_stripe_telemetry(
            semantic_layer.c_str(), profile));
    }
    const bool canonical_ranges = profiles.size() == 3 &&
        profiles[0].row_begin == 0 &&
        profiles[0].row_end == stripe_rows &&
        profiles[1].row_begin == stripe_rows &&
        profiles[1].row_end == 2 * stripe_rows &&
        profiles[2].row_begin == 2 * stripe_rows &&
        profiles[2].row_end == rows;
    const size_t exact_summaries = static_cast<size_t>(std::count_if(
        summaries.begin(), summaries.end(), [&](const auto & summary) {
            return summary.layer == semantic_layer;
        }));
    const size_t physical_site =
        static_cast<size_t>(TestSemanticLayerSite::physical_baseline_dense);
    return check(capture_ok && quantized,
                 "real ExSIA quantization publishes staged host stripes") &&
        check(collected.ok() && completed.ok(),
              "real staged ExSIA host pipeline completes") &&
        check(profiles.size() == 3 &&
                  observations.counts[physical_site] == 3 &&
                  summaries.size() == 3,
              "host pipeline completes exactly three profiles, physical observations, and summaries") &&
        check(canonical_ranges,
              "host pipeline profile rows follow configured DIM") &&
        check(observations.layers[physical_site] == semantic_layer,
              "host pipeline physical dispatch keeps byte-identical layer") &&
        check(debug_output.find("\"layer\":\"" + semantic_layer + "\"") !=
                  std::string::npos,
              "host pipeline quantization log keeps byte-identical layer") &&
        check(exact_summaries == 3,
              "all three actual host pipeline summaries keep byte-identical layer");
}

bool test_non_exsia_pipeline_rejection() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    args.matmul_layer =
        "blk.15.mlp.down_proj.non-exsia-rejection-beyond-sso";
    ResolvedMatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 2;
    options.rmd_backend = RmdBackend::cpu_direct;
    auto execution = prepare_execution(
        static_cast<const ggml_gemmini_args_t &>(args), options);
    return check(execution.status().code ==
                     MatmulStatusCode::unsupported_invocation,
                 "non-ExSIA stripe pipeline keeps unsupported-route status") &&
        check(std::strcmp(execution.status().message,
                          "stripe pipeline requires an ExSIA live producer route") == 0,
              "non-ExSIA stripe pipeline keeps its rejection detail");
}

bool test_owned_route_object_lifetimes() {
    const std::string semantic_layer =
        "blk.15.mlp.down_proj.route-object-lifetime-beyond-sso";
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);

    auto facade_args = make_args(activation, weights, output);
    {
        std::string source = semantic_layer;
        facade_args.matmul_layer = source;
    }
    MatMul facade(facade_args);
    facade_args.matmul_layer = "mutated.after.facade.copy";
    SemanticLayerObservations facade_observations;
    set_test_semantic_layer_observer(observe_semantic_layer, &facade_observations);
    const auto facade_result = facade.run_full();
    set_test_semantic_layer_observer(nullptr, nullptr);

    std::fill(output.begin(), output.end(), 0.0f);
    auto execution_args = make_args(activation, weights, output);
    {
        std::string source = semantic_layer;
        execution_args.matmul_layer = source;
    }
    ResolvedMatmulOptions options{};
    options.mode = MatmulInvocationMode::full;
    options.rmd_backend = RmdBackend::cpu_direct;
    auto execution = prepare_execution(
        static_cast<const ggml_gemmini_args_t &>(execution_args), options);
    execution_args.matmul_layer = "mutated.after.execution.copy";
    SemanticLayerObservations execution_observations;
    set_test_semantic_layer_observer(observe_semantic_layer,
                                     &execution_observations);
    const auto execution_status = execute_full(execution);
    set_test_semantic_layer_observer(nullptr, nullptr);

    return check(facade_result.status == MatMulStatus::success,
                 "owned MatMul completes after source mutation") &&
        check(facade_observations.layers[5] == semantic_layer,
              "MatMul owns the non-SSO layer through physical completion") &&
        check(execution_status.ok(),
              "value-owned MatmulExecution completes after source mutation") &&
        check(execution_observations.layers[5] == semantic_layer,
              "MatmulExecution owns the non-SSO layer through physical completion");
}

bool test_all_physical_semantic_layer_sites() {
    const std::string semantic_layer =
        "blk.15.mlp.down_proj.semantic-layer-observer-beyond-sso";
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    args.matmul_layer = semantic_layer;

    SemanticLayerObservations observations;
    set_test_semantic_layer_observer(observe_semantic_layer, &observations);
    const bool probed = test_probe_physical_layer_sites(args);
    set_test_semantic_layer_observer(nullptr, nullptr);

    bool ok = check(probed, "all physical semantic layer sites are exercised");
    for (size_t site = 1; site < observations.counts.size(); ++site) {
        ok = check(observations.counts[site] == 1,
                   "physical semantic site is observed exactly once") && ok;
        ok = check(observations.layers[site] == semantic_layer,
                   "physical semantic site receives byte-identical owned layer") && ok;
    }
    return check(observations.counts[0] == 0,
                 "physical-only probe does not exercise FP facade seam") && ok;
}

bool test_physical_null_args_contract() {
    return check(test_probe_physical_null_args(),
                 "physical helpers preserve null-argument tolerance");
}

bool test_fp_facade_semantic_layer_forwarding() {
    const std::string semantic_layer =
        "blk.15.attn.q_proj.fp-facade-forwarding-beyond-sso";
    SemanticLayerObservations observations;
    set_test_semantic_layer_observer(observe_semantic_layer, &observations);
    const bool probed = test_probe_fp_facade_layer(semantic_layer);
    set_test_semantic_layer_observer(nullptr, nullptr);

    return check(probed, "FP facade probe succeeds") &&
        check(observations.counts[0] == 1,
              "FP facade-owned args are observed exactly once") &&
        check(observations.layers[0] == semantic_layer,
              "caller semantic layer reaches FP facade physical observation byte-identically");
}

bool test_physical_semantic_layer_seam() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    args.matmul_layer = "blk.15.mlp.down_proj";

    int capture[2]{};
    const int saved_stderr = dup(STDERR_FILENO);
    if (!check(saved_stderr >= 0 && pipe(capture) == 0,
               "physical semantic seam capture opens")) {
        if (saved_stderr >= 0) close(saved_stderr);
        return false;
    }
    std::fflush(stderr);
    dup2(capture[1], STDERR_FILENO);
    close(capture[1]);
    ggml::gemmini::MatMul facade(args);
    const auto result = facade.run_full();
    std::fflush(stderr);
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);
    char captured[2048]{};
    const ssize_t count = read(capture[0], captured, sizeof(captured) - 1);
    close(capture[0]);
    if (count > 0) captured[count] = '\0';

    const std::vector<float> expected = { -1.0f, 8.0f, -1.0f, 18.0f, -1.0f, 28.0f };
    return check(result.status == ggml::gemmini::MatMulStatus::success,
                 "physical semantic seam numeric route succeeds") &&
        check(output == expected, "physical semantic seam preserves quantized numeric output") &&
        check(std::strstr(captured, "\"layer\":\"blk.15.mlp.down_proj\"") != nullptr,
              "physical setup receives byte-identical semantic layer");
}

bool test_generated_config_contract() {
    const bool valid_default_mode =
        config::DEFAULT_MATMUL_MODE == static_cast<int>(MatmulInvocationMode::full) ||
        config::DEFAULT_MATMUL_MODE == static_cast<int>(MatmulInvocationMode::stripe_pipeline);
    const bool stripe_default_requires_support =
        config::DEFAULT_MATMUL_MODE == static_cast<int>(MatmulInvocationMode::full) ||
        config::ENABLE_STRIPE_MATMUL;
    const bool pipeline_default_requires_support =
        config::DEFAULT_MATMUL_MODE != static_cast<int>(MatmulInvocationMode::stripe_pipeline) ||
        config::ENABLE_STRIPE_PIPELINE;
    const auto defaults = resolve_matmul_options();

    return check(valid_default_mode, "generated default matmul mode value") &&
        check(config::DEFAULT_STRIPE_JOB_CAPACITY > 0, "generated job capacity") &&
        check(stripe_default_requires_support, "stripe default requires stripe support") &&
        check(pipeline_default_requires_support, "pipeline default requires pipeline support") &&
        check(defaults.ok(), "generated defaults resolve") &&
        check(defaults.options.mode == static_cast<MatmulInvocationMode>(config::DEFAULT_MATMUL_MODE),
              "resolved default mode matches generated config") &&
        check(defaults.options.job_capacity == config::DEFAULT_STRIPE_JOB_CAPACITY,
              "resolved job capacity matches generated config") &&
        check(config::DEFAULT_RMD_BACKEND == static_cast<int>(RmdBackend::cpu_direct) ||
                  config::DEFAULT_RMD_BACKEND == static_cast<int>(RmdBackend::gemmini_ws_compact),
              "generated RMD backend default is valid") &&
        check(defaults.options.rmd_backend == static_cast<RmdBackend>(config::DEFAULT_RMD_BACKEND),
              "resolved RMD backend matches generated default") &&
        check(defaults.rmd_backend_source == MatmulOptionSource::build_default,
              "default RMD backend source is build default");
}

bool test_checked_geometry_contract() {
    constexpr GemminiTileFactors tiles{5, 5, 48};
    constexpr size_t array_dim = 16;
    struct Fixture {
        GemminiLogicalShape shape;
        GemminiOuterCounts outer;
        size_t ws_inner_calls;
    };
    constexpr Fixture gpt2[] = {
        {{256, 2304, 768}, {4, 29, 1}, 116},
        {{256, 768, 768},  {4, 10, 1}, 40},
        {{256, 3072, 768}, {4, 39, 1}, 156},
        {{256, 768, 3072}, {4, 10, 4}, 160},
    };

    for (const auto & fixture : gpt2) {
        const auto result = make_gemmini_geometry({fixture.shape, tiles, array_dim});
        if (!check(result.ok(), "GPT-2 geometry accepted") ||
            !check(result.geometry.tiles.i == 5 && result.geometry.tiles.j == 5 &&
                       result.geometry.tiles.k == 48,
                   "tile factors remain DIM-block counts") ||
            !check(result.geometry.outer.i == fixture.outer.i &&
                       result.geometry.outer.j == fixture.outer.j &&
                       result.geometry.outer.k == fixture.outer.k,
                   "GPT-2 outer counts are pinned") ||
            !check(result.geometry.stripe_rows == 80 &&
                       result.geometry.stripe_count == 4 &&
                       result.geometry.final_rows == 16,
                   "logical stripes are 80,80,80,16") ||
            !check(result.geometry.ws_inner_calls == fixture.ws_inner_calls,
                   "WS inner-call product is pinned")) {
            return false;
        }
    }

    const auto zero_tile = make_gemmini_geometry({{256, 768, 768}, {0, 5, 48}, array_dim});
    const auto tile_product_overflow = make_gemmini_geometry(
        {{256, 768, 768}, {std::numeric_limits<size_t>::max(), 5, 48}, array_dim});
    const auto call_product_overflow = make_gemmini_geometry(
        {{std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(),
          std::numeric_limits<size_t>::max()}, {1, 1, 1}, 1});
    return check(zero_tile.error == GemminiGeometryError::zero_tile_factor,
                 "zero tile factor has typed rejection") &&
        check(tile_product_overflow.error == GemminiGeometryError::overflow,
              "tile-row product overflow has typed rejection") &&
        check(call_product_overflow.error == GemminiGeometryError::overflow,
              "WS inner-call product overflow has typed rejection");
}

bool test_precedence() {
    clear_environment();
    const auto defaults = resolve_matmul_options();
    if (!check(defaults.ok() && defaults.options.job_capacity == config::DEFAULT_STRIPE_JOB_CAPACITY,
               "generated build defaults")) {
        return false;
    }

    setenv("GEMMINI_MATMUL_MODE", "FULL", 1);
    setenv("GEMMINI_STRIPE_JOB_CAPACITY", "5", 1);
    setenv("GEMMINI_RMD_BACKEND", "WS", 1);
    const auto environment = resolve_matmul_options();
    if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE &&
        !check(environment.ok() && environment.options.mode == MatmulInvocationMode::full &&
                   environment.options.job_capacity == 5 &&
                   environment.options.rmd_backend == RmdBackend::gemmini_ws_compact &&
                   environment.rmd_backend_source == MatmulOptionSource::environment,
               "environment overrides build defaults")) {
        return false;
    }

    MatmulOptionOverrides explicit_options{};
    explicit_options.mode = config::ENABLE_STRIPE_MATMUL && config::ENABLE_STRIPE_PIPELINE
        ? MatmulInvocationMode::stripe_pipeline : MatmulInvocationMode::full;
    explicit_options.job_capacity = 6;
    explicit_options.rmd_backend = RmdBackend::cpu_direct;
    setenv("GEMMINI_RMD_BACKEND", "DEC", 1);
    const auto explicit_result = resolve_matmul_options(explicit_options);
    clear_environment();
    return check(explicit_result.ok() && explicit_result.options.mode == *explicit_options.mode &&
                     explicit_result.options.job_capacity == 6 &&
                     explicit_result.options.rmd_backend == RmdBackend::cpu_direct &&
                     explicit_result.rmd_backend_source == MatmulOptionSource::explicit_override,
                 "explicit options override environment");
}

bool test_invalid_environment() {
    struct InvalidEnvironment {
        const char * name;
        const char * value;
        MatmulOptionsError error;
    };
    const InvalidEnvironment cases[] = {
        { "GEMMINI_MATMUL_MODE", "AUTO", MatmulOptionsError::invalid_mode },
        { "GEMMINI_MATMUL_MODE", "full", MatmulOptionsError::invalid_mode },
        { "GEMMINI_MATMUL_MODE", "STRIPE_SEQUENTIAL", MatmulOptionsError::invalid_mode },
        { "GEMMINI_STRIPE_JOB_CAPACITY", "+2", MatmulOptionsError::invalid_job_capacity },
        { "GEMMINI_STRIPE_JOB_CAPACITY", "0", MatmulOptionsError::invalid_job_capacity },
        { "GEMMINI_RMD_BACKEND", "", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "cpu", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", " CPU", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "WS ", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "0", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "AUTO", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "INHERIT", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "OS", MatmulOptionsError::invalid_rmd_backend },
        { "GEMMINI_RMD_BACKEND", "DEC", MatmulOptionsError::invalid_rmd_backend },
    };
    for (const auto & invalid : cases) {
        clear_environment();
        setenv(invalid.name, invalid.value, 1);
        const auto result = resolve_matmul_options();
        if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE) {
            if (!check(!result.ok() && result.error == invalid.error, "invalid environment rejected")) {
                return false;
            }
        } else if (std::strcmp(invalid.name, "GEMMINI_RMD_BACKEND") == 0) {
            if (!check(!result.ok() && result.error == MatmulOptionsError::runtime_override_disabled,
                       "disabled RMD runtime override rejected")) {
                return false;
            }
        } else if (!check(result.ok(), "disabled runtime override ignores unrelated environment")) {
            return false;
        }
    }
    clear_environment();
    return true;
}

bool test_invalid_explicit_rmd_backend() {
    MatmulOptionOverrides options{};
    options.rmd_backend = static_cast<RmdBackend>(2);
    const auto result = resolve_matmul_options(options);
    return check(!result.ok() && result.error == MatmulOptionsError::invalid_rmd_backend,
                 "invalid explicit RMD backend rejected");
}

bool test_invalid_explicit_mode() {
    MatmulOptionOverrides options{};
    options.mode = static_cast<MatmulInvocationMode>(2);
    const auto result = resolve_matmul_options(options);
    return check(!result.ok() && result.error == MatmulOptionsError::invalid_mode,
                 "invalid explicit matmul mode rejected");
}

#if defined(GGML_GEMMINI_OPTIONS_TEST_BACKEND)
bool test_execution_route_propagation() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    const tiled_matmul_type_t main_route = args.tiled_matmul_type;
    ResolvedMatmulOptions options{};
    options.mode = MatmulInvocationMode::full;
    options.rmd_backend = RmdBackend::cpu_direct;
    auto execution = prepare_execution(&args, options);
    const bool cpu = check(execution.status().ok() &&
                               args.residual_route == residual::ResidualRoute::cpu_direct,
                           "resolved CPU backend reaches execution residual sink") &&
        check(args.tiled_matmul_type == main_route,
              "CPU residual selection preserves main backend");

    options.rmd_backend = RmdBackend::gemmini_ws_compact;
    auto ws_execution = prepare_execution(&args, options);
#if defined(__riscv) || defined(GGML_GEMMINI_TESTING)
    const bool ws = check(ws_execution.status().ok(),
                          "WS backend is available on target or testing host");
#else
    const bool ws = check(ws_execution.status().code == MatmulStatusCode::unsupported_backend,
                          "WS backend preflights as unavailable on production host");
#endif
    return cpu && ws &&
        check(args.residual_route == residual::ResidualRoute::ws_packet,
              "resolved WS backend reaches execution residual sink") &&
        check(args.tiled_matmul_type == main_route,
              "WS residual selection preserves main backend");
}
#endif

bool test_disabled_runtime_rmd_environment() {
    if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE) {
        return true;
    }
    clear_environment();
    setenv("GEMMINI_RMD_BACKEND", "WS", 1);
    const auto result = resolve_matmul_options();
    clear_environment();
    return check(!result.ok() && result.error == MatmulOptionsError::runtime_override_disabled &&
                     result.rmd_backend_source == MatmulOptionSource::build_default,
                 "disabled runtime override rejects valid RMD backend environment");
}

#if defined(GGML_GEMMINI_OPTIONS_TEST_BACKEND)
bool test_disabled_mode_status_contract() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(12, 0.0f);
    bool passed = true;

    if (!config::ENABLE_STRIPE_MATMUL || !config::ENABLE_STRIPE_PIPELINE) {
        auto args = make_args(activation, weights, output);
        args.I = 6;
        args.K = 1;
        args.sA = 1;
        activation.resize(args.I * args.K, 1);
        args.A.allocate(args.I, args.K, 8);
        for (size_t i = 0; i < args.I * args.K; ++i) {
            args.A.set(i / args.K, i % args.K, activation[i]);
        }
        auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
        meta.theta.assign(args.I, 0);

        MatmulOptionOverrides options{};
        options.mode = MatmulInvocationMode::stripe_pipeline;
        const auto resolution = resolve_matmul_options(options);
        const auto execution = prepare_execution(args, options);
        passed = check(!resolution.ok() && resolution.error == MatmulOptionsError::disabled_mode,
                       "disabled stripe pipeline resolves as disabled mode") && passed;
        passed = check(execution.status().code == MatmulStatusCode::unsupported_invocation,
                       "disabled stripe pipeline maps to unsupported invocation") && passed;
    }

    return passed;
}
#endif

}

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--semantic-layer-wrong-constant") == 0) {
        char * end = nullptr;
        const long site = std::strtol(argv[2], &end, 10);
        if (end == argv[2] || *end != '\0' || site < 0 || site > 5) return 2;
        semantic_layer_mutation_site = static_cast<int>(site);
        const bool unexpectedly_passed = site == 0
            ? test_fp_facade_semantic_layer_forwarding()
            : test_all_physical_semantic_layer_sites();
        semantic_layer_mutation_site = -1;
        return unexpectedly_passed ? 0 : 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--geometry-fixture") == 0) {
        if (!test_checked_geometry_contract()) return 1;
        const auto result = make_gemmini_geometry({{256, 2304, 768}, {5, 5, 48}, 16});
        std::printf("GEOMETRY stripe_rows=%zu stripe_count=%zu final_rows=%zu outer=%zu/%zu/%zu ws_inner_calls=%zu rows=80,80,80,16\n",
                    result.geometry.stripe_rows, result.geometry.stripe_count,
                    result.geometry.final_rows, result.geometry.outer.i,
                    result.geometry.outer.j, result.geometry.outer.k,
                    result.geometry.ws_inner_calls);
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--invalid-geometry-probe") == 0) {
        const auto zero = make_gemmini_geometry({{256, 768, 768}, {0, 5, 48}, 16});
        const auto overflow = make_gemmini_geometry(
            {{256, 768, 768}, {std::numeric_limits<size_t>::max(), 5, 48}, 16});
        const bool ok = zero.error == GemminiGeometryError::zero_tile_factor &&
            overflow.error == GemminiGeometryError::overflow;
        std::printf("INVALID_GEOMETRY zero_tile=%d overflow=%d allocations=0\n",
                    static_cast<int>(zero.error), static_cast<int>(overflow.error));
        return ok ? 0 : 1;
    }
    const bool ok = test_checked_geometry_contract() &&
        test_staged_exsia_host_pipeline_semantic_layer() &&
        test_owned_matmul_layer_lifetime() &&
        test_non_exsia_pipeline_rejection() &&
        test_owned_route_object_lifetimes() &&
        test_all_physical_semantic_layer_sites() &&
        test_physical_null_args_contract() &&
        test_fp_facade_semantic_layer_forwarding() &&
        test_physical_semantic_layer_seam() &&
        test_quantization_and_exsia_semantic_layer_seam() &&
        test_backend_semantic_resolution_and_dedupe() &&
        test_args_layout_extension() &&
        test_generated_config_contract() &&
        test_precedence() &&
        test_invalid_environment() &&
        test_invalid_explicit_rmd_backend() &&
        test_invalid_explicit_mode() &&
        test_disabled_runtime_rmd_environment()
#if defined(GGML_GEMMINI_OPTIONS_TEST_BACKEND)
        && test_execution_route_propagation()
#endif
#if defined(GGML_GEMMINI_OPTIONS_TEST_BACKEND)
        && test_disabled_mode_status_contract()
#endif
        ;
    return ok ? 0 : 1;
}
