#define GGML_GEMMINI_TESTING 1
#define GGML_GEMMINI_TEST_OBSERVER 1

#include "../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace ggml::gemmini;

bool expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

ggml_gemmini_args_t make_args(
        std::vector<elem_t> & activation,
        std::vector<elem_t> & weights,
        std::vector<float> & output) {
    ggml_gemmini_args_t args{};
    args.I = 3;
    args.J = 2;
    args.K = 2;
    if (!args.A.allocate(args.I, args.K, 8)) {
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
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0, 0 };
    return args;
}

bool same_output(const std::vector<float> & left, const std::vector<float> & right) {
    return left.size() == right.size() &&
        std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

bool counters_zero(const MatmulTestCounters & counters) {
    return counters.execution_constructions == 0 && counters.allocation_attempts == 0 &&
        counters.dense_dispatches == 0 && counters.residual_dispatches == 0 &&
        counters.hardware_dispatches == 0 && counters.fallback_dispatches == 0;
}

bool test_removed_sequential_rejects_before_work() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    const std::vector<float> environment_sentinel(6, 73.0f);
    std::vector<float> environment_output = environment_sentinel;
    auto environment_args = make_args(activation, weights, environment_output);

    unsetenv("GEMMINI_MATMUL_MODE");
    test_reset_matmul_counters();
    setenv("GEMMINI_MATMUL_MODE", "STRIPE_SEQUENTIAL", 1);
    const MatmulStatus environment_status = matmul(environment_args);
    const MatmulTestCounters environment_counters = test_matmul_counters();
    unsetenv("GEMMINI_MATMUL_MODE");

    return expect(environment_status.code == MatmulStatusCode::invalid_argument,
                  "removed sequential mode has typed rejection") &&
        expect(environment_output == environment_sentinel,
               "removed sequential mode preserves nonzero output sentinel") &&
        expect(counters_zero(environment_counters),
               "removed sequential mode performs zero construction/allocation/dispatch");
}

bool test_invalid_geometry_rejects_before_allocation() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 79.0f);
    auto args = make_args(activation, weights, output);
    args.I = std::numeric_limits<size_t>::max();
    args.J = std::numeric_limits<size_t>::max();
    args.K = std::numeric_limits<size_t>::max();
    args.tile_I = 1;
    args.tile_J = 1;
    args.tile_K = 1;
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;

    test_reset_matmul_counters();
    const auto execution = prepare_execution(&args, options);
    const auto counters = test_matmul_counters();
    return expect(execution.status().code == MatmulStatusCode::invalid_contract,
                  "overflowing geometry has typed invalid-contract status") &&
        expect(counters.execution_constructions == 1 && counters.allocation_attempts == 0,
               "invalid geometry is rejected before allocation");
}

bool test_output_parity() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> full_output(6, 0.0f);
    std::vector<float> pipeline_output(6, 0.0f);

    auto full_args = make_args(activation, weights, full_output);
    MatmulOptions full_options{};
    full_options.mode = MatmulInvocationMode::full;
    full_options.rmd_backend = RmdBackend::cpu_direct;
    const MatmulStatus full = matmul(full_args, full_options);

    auto pipeline_args = make_args(activation, weights, pipeline_output);
    MatmulOptions pipeline_options{};
    pipeline_options.mode = MatmulInvocationMode::stripe_pipeline;
    pipeline_options.job_capacity = 2;
    pipeline_options.rmd_backend = RmdBackend::cpu_direct;
    auto execution = prepare_execution(&pipeline_args, pipeline_options);
    MatmulStripeCollector collector(2);
    if (!expect(execution.status().ok() && collector.start(execution), "pipeline starts")) {
        std::fprintf(
            stderr, "pipeline status=%u message=%s\n",
            static_cast<unsigned>(execution.status().code),
            execution.status().message);
        return false;
    }
    const auto * sink = collector.sink();
    quants::act::exsia::StripeReadyEvent first{};
    first.stripe_id = 0;
    first.row_end = 1;
    quants::act::exsia::StripeReadyEvent second{};
    second.stripe_id = 1;
    second.slot = 1;
    second.row_begin = 1;
    second.row_end = 3;
    const bool accepted = sink->on_ready(sink->user_data, first) &&
        sink->on_ready(sink->user_data, second);
    const MatmulStatus collected = collector.finish();
    const MatmulStatus pipeline = finish_execution(execution);

    return expect(full.ok(), "FULL succeeds") &&
        expect(accepted && collected.ok() && pipeline.ok(), "PIPELINE succeeds") &&
        expect(same_output(full_output, pipeline_output), "FULL/PIPELINE parity");
}

bool test_single_row_pipeline() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> full_output(2, 0.0f);
    std::vector<float> pointer_output(2, 0.0f);
    std::vector<float> value_output(2, 0.0f);

    auto full_args = make_args(activation, weights, full_output);
    full_args.I = 1;
    MatmulOptions full_options{};
    full_options.mode = MatmulInvocationMode::full;
    full_options.rmd_backend = RmdBackend::cpu_direct;
    const MatmulStatus full = matmul(full_args, full_options);

    auto pipeline_args = make_args(activation, weights, pointer_output);
    pipeline_args.I = 1;
    MatmulOptions pipeline_options{};
    pipeline_options.mode = MatmulInvocationMode::stripe_pipeline;
    pipeline_options.job_capacity = 1;
    pipeline_options.rmd_backend = RmdBackend::cpu_direct;
    auto execution = prepare_execution(&pipeline_args, pipeline_options);
    MatmulStripeCollector collector(1);
    if (!expect(execution.status().ok() && collector.start(execution),
                "single-row pipeline starts")) {
        return false;
    }
    quants::act::exsia::StripeReadyEvent only{};
    only.stripe_id = 0;
    only.row_end = 1;
    const auto * sink = collector.sink();
    const bool accepted = sink->on_ready(sink->user_data, only);
    const MatmulStatus collected = collector.finish();
    const MatmulStatus pipeline = finish_execution(execution);

    auto value_args = make_args(activation, weights, value_output);
    value_args.I = 1;
    auto value_execution = prepare_execution(
        static_cast<const ggml_gemmini_args_t &>(value_args), pipeline_options);
    MatmulStripeCollector value_collector(1);
    if (!expect(value_execution.status().ok() &&
                    value_collector.start(value_execution),
                "single-row by-value pipeline starts")) {
        return false;
    }
    const auto * value_sink = value_collector.sink();
    const bool value_accepted =
        value_sink->on_ready(value_sink->user_data, only);
    const MatmulStatus value_collected = value_collector.finish();
    const MatmulStatus value_pipeline = finish_execution(value_execution);

    return expect(full.ok(), "single-row FULL succeeds") &&
        expect(accepted && collected.ok() && pipeline.ok(),
               "single-row pointer PIPELINE succeeds") &&
        expect(value_accepted && value_collected.ok() && value_pipeline.ok(),
               "single-row by-value PIPELINE succeeds") &&
        expect(same_output(full_output, pointer_output) &&
                   same_output(full_output, value_output),
               "single-row FULL/pointer/value PIPELINE parity");
}

residual::DirectStripePayloadHandle make_direct_payload(
        size_t stripe_id, size_t row_begin, size_t row_count, int32_t residual_value) {
    residual::DirectStripeBuilder builder;
    builder.reset(stripe_id, row_begin, row_count, 2, 2);
    if (residual_value != 0 && !builder.add_residual(0, 0, residual_value)) return {};
    return builder.finish();
}

void install_direct_payloads(ggml_gemmini_args_t & args) {
    auto & meta = std::get<quants::act::exsia::Meta>(args.act_quant.storage());
    meta.direct_residuals = {
        make_direct_payload(0, 0, 1, 256),
        make_direct_payload(1, 1, 2, -128),
    };
}

bool test_counter_hooks_connected() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> fallback_output(6, 0.0f);
    auto fallback_args = make_args(activation, weights, fallback_output);
    install_direct_payloads(fallback_args);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::full;
    options.rmd_backend = RmdBackend::cpu_direct;

    test_reset_matmul_counters();
    const MatmulStatus fallback_status = matmul(fallback_args, options);
    const MatmulTestCounters fallback = test_matmul_counters();

    std::vector<float> hardware_output(6, 0.0f);
    auto hardware_args = make_args(activation, weights, hardware_output);
    hardware_args.tiled_matmul_type = static_cast<tiled_matmul_type_t>(1);
    test_reset_matmul_counters();
    const MatmulStatus hardware_status = matmul(hardware_args, options);
    const MatmulTestCounters hardware = test_matmul_counters();

    const bool residual_counter_ok =
#if GGML_GEMMINI_ENABLE_RMD
        fallback.residual_dispatches > 0;
#else
        fallback.residual_dispatches == 0;
#endif
    return expect(fallback_status.ok(), "counter fallback control succeeds") &&
        expect(fallback.execution_constructions > 0 && fallback.allocation_attempts > 0 &&
                   fallback.dense_dispatches > 0 && residual_counter_ok &&
                   fallback.fallback_dispatches > 0,
               "construction/allocation/dense/residual/fallback hooks observe real work") &&
        expect(hardware_status.ok() && hardware.dense_dispatches > 0 &&
                   hardware.hardware_dispatches > 0,
               "hardware hook observes real dispatch");
}

bool test_cpu_direct_lifecycle_parity() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> full_output(6, 0.0f), pipeline_output(6, 0.0f);
    MatmulOptions options{};
    options.rmd_backend = RmdBackend::cpu_direct;

    auto full_args = make_args(activation, weights, full_output);
    install_direct_payloads(full_args);
    const auto main_route = full_args.tiled_matmul_type;
    options.mode = MatmulInvocationMode::full;
    const MatmulStatus full = matmul(full_args, options);
    const std::vector<float> first_full_output = full_output;
    std::fill(full_output.begin(), full_output.end(), 0.0f);
    const MatmulStatus repeated_full = matmul(full_args, options);

    auto pipeline_args = make_args(activation, weights, pipeline_output);
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 2;
    auto execution = prepare_execution(&pipeline_args, options);
    MatmulStripeCollector collector(2);
    if (!expect(execution.status().ok() && collector.start(execution), "direct pipeline starts")) return false;
    quants::act::exsia::StripeReadyEvent first{};
    first.stripe_id = 0; first.row_end = 1;
    first.direct_residual = make_direct_payload(0, 0, 1, 256);
    quants::act::exsia::StripeReadyEvent second{};
    second.stripe_id = 1; second.slot = 1; second.row_begin = 1; second.row_end = 3;
    second.direct_residual = make_direct_payload(1, 1, 2, -128);
    const auto * sink = collector.sink();
    const bool accepted = sink->on_ready(sink->user_data, first) && sink->on_ready(sink->user_data, second);
    const MatmulStatus collected = collector.finish();
    const MatmulStatus pipeline = finish_execution(execution);

    return expect(full.ok() && repeated_full.ok() && accepted && collected.ok() && pipeline.ok(),
                  "direct backend succeeds in FULL/pipeline and repeated FULL") &&
        expect(same_output(first_full_output, full_output),
               "repeated direct invocation has no stale residual state") &&
        expect(same_output(full_output, pipeline_output),
               "direct backend FULL/pipeline final-output parity") &&
        expect(full_output[0] != 5.0f || full_output[1] != 5.0f,
               "direct residual is merged after dense output") &&
        expect(full_args.tiled_matmul_type == main_route &&
                   pipeline_args.tiled_matmul_type == main_route,
               "backend selector preserves the main matmul route");
}

bool test_rmd_disabled_pipeline_skips_correction() {
#if !GGML_GEMMINI_ENABLE_RMD
    std::vector<elem_t> activation = {1, 2, 3, 4, 5, 6};
    std::vector<elem_t> weights = {1, -1, 2, 3};
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);

    MatmulOptions options{};
    std::vector<float> dense_output(6, 0.0f);
    auto dense_args = make_args(activation, weights, dense_output);
    options.mode = MatmulInvocationMode::full;
    options.rmd_backend = RmdBackend::cpu_direct;
    const MatmulStatus dense_status = matmul(dense_args, options);

    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.rmd_backend = RmdBackend::cpu_direct;
    options.job_capacity = 2;
    auto execution = prepare_execution(&args, options);
    MatmulStripeCollector collector(2);
    if (!expect(execution.status().ok() && collector.start(execution),
                "RMD-disabled pipeline starts")) {
        return false;
    }

    quants::act::exsia::StripeReadyEvent first{};
    first.stripe_id = 0;
    first.row_end = 1;
    first.direct_residual = make_direct_payload(0, 0, 1, 256);
    quants::act::exsia::StripeReadyEvent second{};
    second.stripe_id = 1;
    second.slot = 1;
    second.row_begin = 1;
    second.row_end = 3;
    second.direct_residual = make_direct_payload(1, 1, 2, -128);

    test_reset_matmul_counters();
    const auto * sink = collector.sink();
    const bool accepted =
        sink->on_ready(sink->user_data, first) &&
        sink->on_ready(sink->user_data, second);
    const MatmulStatus collected = collector.finish();
    const MatmulStatus pipeline = finish_execution(execution);
    const MatmulTestCounters counters = test_matmul_counters();
    return expect(dense_status.ok() && accepted && collected.ok() && pipeline.ok(),
                  "RMD-disabled pipeline completes") &&
        expect(output == dense_output,
               "RMD-disabled pipeline preserves dense-only output") &&
        expect(counters.residual_dispatches == 0,
               "RMD-disabled pipeline dispatches no residual work");
#else
    return true;
#endif
}

bool test_dense_rejects_residual_metadata() {
    std::vector<elem_t> activation = {1, 2, 3, 4, 5, 6};
    std::vector<elem_t> weights = {1, -1, 2, 3};
    std::vector<float> output(6, -23.0f);
    auto args = make_args(activation, weights, output);
    install_direct_payloads(args);
    MatMul facade(&args);
    const MatMulResult result = facade.run_dense();
    return expect(
        result.status == MatMulStatus::invalid_contract,
        "public dense execution rejects uncompensated residual metadata") &&
        expect(
            output == std::vector<float>(6, -23.0f),
            "rejected dense execution preserves destination");
}

bool test_correction_domain_composition() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> original = { 5, 5, 11, 9, 17, 13 };
    auto args = make_args(activation, weights, original);
    args.I = 1;
    args.K = rmd::kBlockSize;

    block_q8_h1 h1[2]{};
    for (block_q8_h1 & block : h1) {
        block.c_b = 1;
        block.s_rf = 0.25f;
    }
    args.weight_i8_scale_active = false;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = h1;
    args.q8_h1_block_count = 2;
    args.q8_h1_rows = 2;
    args.blocks_per_row = 1;
    args.native_weight_bytes = sizeof(h1);

    std::vector<float> full_destination = { 1.0f, 1.0f };
    const rmd::Correction h1_correction =
        rmd::BlockScaledInt64Correction{{8, -8}};
    bool ok = expect(
        rmd::merge_rmd_correction_to(
            args, full_destination.data(), 0, 1, h1_correction) ==
            rmd::RmdStatus::success &&
            full_destination == std::vector<float>({3.0f, -1.0f}),
        "FULL H1 correction applies column scale exactly once");

    constexpr int64_t wide_positive =
        static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 4096;
    constexpr int64_t wide_negative =
        static_cast<int64_t>(std::numeric_limits<int32_t>::min()) - 4096;
    std::vector<float> wide_output = { 0.0f, 0.0f };
    const rmd::Correction wide = rmd::BlockScaledInt64Correction{{
        wide_positive, wide_negative,
    }};
    ok = expect(
        rmd::merge_rmd_correction_to(args, wide_output.data(), 0, 1, wide) ==
                rmd::RmdStatus::success &&
            wide_output[0] == static_cast<float>(
                static_cast<double>(wide_positive) * 0.25) &&
            wide_output[1] == static_cast<float>(
                static_cast<double>(wide_negative) * 0.25),
        "H1 CPU correction retains int64 domain until column scaling") && ok;

    block_q8_0 h0[2]{};
    h0[0].d = ggml_fp32_to_fp16(0.5f);
    h0[1].d = ggml_fp32_to_fp16(4.0f);
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h0;
    args.B_blocks = h0;
    args.blocks_J = 2;
    args.blocks_K = 1;
    args.native_weight_bytes = sizeof(h0);
    std::vector<float> stripe_destination = { 1.0f, 1.0f };
    const rmd::Correction h0_correction =
        rmd::PreScaledFloat64Correction{{2.5, -4.0}};
    ok = expect(
        rmd::merge_rmd_correction_to(
            args, stripe_destination.data(), 0, 1, h0_correction) ==
                rmd::RmdStatus::success &&
            stripe_destination == std::vector<float>({3.5f, -3.0f}),
        "STRIPE H0 applies pre-scaled values directly without column scale") && ok;

    auto remains_unchanged = [&](const rmd::Correction & correction,
                                 rmd::RmdStatus expected,
                                 const char * message) {
        std::vector<float> destination = { 17.0f, -23.0f };
        const std::vector<float> before = destination;
        return expect(rmd::merge_rmd_correction_to(
                          args, destination.data(), 0, 1, correction) == expected &&
                          same_output(destination, before),
                      message);
    };
    ok = remains_unchanged(h1_correction, rmd::RmdStatus::unsupported_route,
                           "wrong integer domain for H0 is failure-atomic") && ok;
    ok = remains_unchanged(
             rmd::PreScaledFloat64Correction{{
                 std::numeric_limits<double>::quiet_NaN(), 1.0,
             }},
             rmd::RmdStatus::overflow,
             "NaN H0 correction is failure-atomic") && ok;

    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = h1;
    args.native_weight_bytes = sizeof(h1);
    ok = remains_unchanged(h0_correction, rmd::RmdStatus::unsupported_route,
                           "wrong floating domain for H1 is failure-atomic") && ok;
    rmd::RmdStripeBuilder builder;
    builder.reset(9, 0, 1, rmd::kBlockSize, 2);
    builder.add_residual(0, 0, 257);
    const rmd::StripePacketHandle packet = builder.finish();
    rmd::CompressedOutput compressed;
    compressed.j_padded = packet->j_padded;
    compressed.values.assign(packet->total_output_values,
                             std::numeric_limits<int64_t>::max());
    rmd::Correction compose_sentinel =
        rmd::PreScaledFloat64Correction{{7.25, -3.5}};
    const auto * compose_before =
        std::get_if<rmd::PreScaledFloat64Correction>(&compose_sentinel);
    const double * const compose_data = compose_before->values.data();
    ok = expect(
        rmd::compose_rmd_output(*packet, compressed, compose_sentinel) ==
                rmd::RmdStatus::overflow &&
            std::get_if<rmd::PreScaledFloat64Correction>(&compose_sentinel) != nullptr &&
            std::get<rmd::PreScaledFloat64Correction>(compose_sentinel).values ==
                std::vector<double>({7.25, -3.5}) &&
            std::get<rmd::PreScaledFloat64Correction>(compose_sentinel).values.data() ==
                compose_data,
        "checked radix add overflow leaves correction variant byte-identical") && ok;

    std::vector<float> compact_destination = { 31.0f, -41.0f };
    const std::vector<float> compact_before = compact_destination;
    ok = expect(
        rmd::merge_rmd_correction_to(
            args, compact_destination.data(), *packet, h0_correction) ==
                rmd::RmdStatus::unsupported_route &&
            same_output(compact_destination, compact_before),
        "H0 pre-scaled correction never enters compact composition") && ok;
    return ok;
}

bool test_cpu_direct_failure_commits_no_partial_correction() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    auto install_failure = [](ggml_gemmini_args_t & args) {
        auto & meta = std::get<quants::act::exsia::Meta>(args.act_quant.storage());
        meta.direct_residuals = {
            make_direct_payload(0, 0, 1, 256),
            make_direct_payload(3, 3, 1, -128),
        };
    };

    std::vector<float> direct_output(6, 91.0f);
    const std::vector<float> direct_sentinel = direct_output;
    auto direct_args = make_args(activation, weights, direct_output);
    direct_args.residual_route = residual::ResidualRoute::cpu_direct;
    install_failure(direct_args);
    MatMul facade(&direct_args);
    const MatMulResult direct = facade.run_full();

    std::vector<float> execution_output(6, 93.0f);
    const std::vector<float> execution_sentinel = execution_output;
    auto execution_args = make_args(activation, weights, execution_output);
    install_failure(execution_args);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::full;
    options.rmd_backend = RmdBackend::cpu_direct;
    const MatmulStatus execution = matmul(execution_args, options);
    return expect(direct.status != MatMulStatus::success && !execution.ok(),
                  "direct MatMul and MatmulExecution merge failures propagate") &&
        expect(same_output(direct_output, direct_sentinel) &&
                   same_output(execution_output, execution_sentinel) &&
                   direct_args.f_out == direct_output.data() &&
                   execution_args.f_out == execution_output.data(),
               "FULL failures restore destination ownership byte-identically");
}

bool test_stripe_residual_failure_is_transaction_atomic() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 83.0f);
    const std::vector<float> sentinel = output;
    auto args = make_args(activation, weights, output);

    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    options.rmd_backend = RmdBackend::cpu_direct;
    auto execution = prepare_execution(&args, options);
    MatmulStripeCollector collector(1);
    collector.test_inject_residual_failure({
        MatmulStatusCode::execution_failure,
        "injected exact-event residual failure",
    });
    if (!expect(execution.status().ok() && collector.start(execution),
                "atomic STRIPE probe starts")) {
        return false;
    }

    quants::act::exsia::StripeReadyEvent event{};
    event.stripe_id = 0;
    event.row_end = 1;
    event.direct_residual = make_direct_payload(0, 0, 1, 256);
    const auto * sink = collector.sink();
    const bool accepted = sink->on_ready(sink->user_data, event);
    collector.test_wait_for_residual_failure();
    const MatmulStatus collected = collector.finish();
    const MatmulStatus finished = finish_execution(execution);
    return expect(accepted, "atomic STRIPE probe accepts one partial-row stripe") &&
        expect(collected.code == MatmulStatusCode::execution_failure &&
                   finished.code == MatmulStatusCode::execution_failure,
               "exact-event STRIPE residual failure propagates") &&
        expect(same_output(output, sentinel) && args.f_out == output.data(),
               "STRIPE failure restores destination ownership byte-identically");
}

bool test_output_transaction_strides_overflow_and_cancel() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> strided(12, 67.0f);
    auto args = make_args(activation, weights, strided);
    args.stride_f_out = 4;
    MatmulOptions full{};
    full.mode = MatmulInvocationMode::full;
    full.rmd_backend = RmdBackend::cpu_direct;
    const MatmulStatus full_status = matmul(args, full);
    bool ok = expect(full_status.ok() && args.f_out == strided.data(),
                     "strided FULL output transaction commits and restores ownership");
    ok = expect(strided[2] == 67.0f && strided[3] == 67.0f &&
                    strided[6] == 67.0f && strided[7] == 67.0f &&
                    strided[10] == 67.0f && strided[11] == 67.0f,
                "transaction commits logical strided elements only") && ok;

    std::vector<float> overflow_output(6, 71.0f);
    const std::vector<float> overflow_sentinel = overflow_output;
    auto overflow_args = make_args(activation, weights, overflow_output);
    overflow_args.stride_f_out = std::numeric_limits<size_t>::max();
    const MatmulStatus overflow = matmul(overflow_args, full);
    ok = expect(!overflow.ok() && same_output(overflow_output, overflow_sentinel),
                "output transaction layout overflow rejects before dense mutation") && ok;

    std::vector<float> allocation_output(6, 73.0f);
    const std::vector<float> allocation_sentinel = allocation_output;
    auto allocation_args = make_args(activation, weights, allocation_output);
    test_reset_matmul_counters();
    test_inject_output_stage_allocation_failure();
    const MatmulStatus allocation = matmul(allocation_args, full);
    const MatmulTestCounters allocation_counters = test_matmul_counters();
    ok = expect(!allocation.ok() &&
                    same_output(allocation_output, allocation_sentinel) &&
                    allocation_counters.allocation_attempts == 1 &&
                    allocation_counters.dense_dispatches == 0,
                "output stage allocation failure rejects before dense dispatch") && ok;

    std::vector<float> cancel_output(6, 79.0f);
    const std::vector<float> cancel_sentinel = cancel_output;
    auto cancel_args = make_args(activation, weights, cancel_output);
    MatmulOptions stripe{};
    stripe.mode = MatmulInvocationMode::stripe_pipeline;
    stripe.job_capacity = 1;
    stripe.rmd_backend = RmdBackend::cpu_direct;
    auto execution = prepare_execution(&cancel_args, stripe);
    MatmulStripeCollector collector(1);
    if (!expect(execution.status().ok() && collector.start(execution),
                "transaction cancellation probe starts")) {
        return false;
    }
    const MatmulStatus cancelled = collector.cancel();
    const MatmulStatus collected = collector.finish();
    const MatmulStatus finished = finish_execution(execution);
    return expect(cancelled.code == MatmulStatusCode::cancelled &&
                      collected.code == MatmulStatusCode::cancelled &&
                      finished.code == MatmulStatusCode::cancelled,
                  "transaction cancellation propagates") &&
        expect(same_output(cancel_output, cancel_sentinel) &&
                   cancel_args.f_out == cancel_output.data(),
               "cancelled transaction restores destination ownership byte-identically") && ok;
}

bool test_rmd_failure_atomicity_suite() {
#if GGML_GEMMINI_ENABLE_RMD
    return test_cpu_direct_failure_commits_no_partial_correction() &&
        test_stripe_residual_failure_is_transaction_atomic() &&
        test_output_transaction_strides_overflow_and_cancel();
#else
    return true;
#endif
}

bool test_ws_preflight_is_atomic_on_host() {
#if defined(__riscv)
    return true;
#else
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 77.0f);
    const std::vector<float> before = output;
    auto args = make_args(activation, weights, output);
    const auto route = args.tiled_matmul_type;
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::full;
    options.rmd_backend = RmdBackend::gemmini_ws_compact;
    test_reset_matmul_counters();
    const MatmulStatus status = matmul(args, options);
    const MatmulTestCounters counters = test_matmul_counters();
#if defined(GGML_GEMMINI_TESTING)
    (void) counters;
    return expect(status.ok(),
                  "testing host allows the software compact lifecycle") &&
        expect(std::all_of(output.begin(), output.end(),
                           [](float value) { return std::isfinite(value); }),
               "testing compact output remains finite") &&
        expect(!same_output(output, before),
               "testing compact commits only its successful staged output") &&
        expect(args.tiled_matmul_type == route,
               "testing compact preserves the main route");
#else
    return expect(status.code == MatmulStatusCode::unsupported_backend,
                  "non-testing host rejects compact with unsupported_backend") &&
        expect(same_output(output, before), "WS preflight does not mutate output") &&
        expect(args.tiled_matmul_type == route, "WS preflight preserves main route");
#endif
#endif
}

bool test_bad_routes() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);

    auto malformed_args = make_args(activation, weights, output);
    malformed_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    MatmulOptions full_options{};
    full_options.mode = MatmulInvocationMode::full;
    full_options.rmd_backend = RmdBackend::cpu_direct;
    const MatmulStatus malformed = matmul(malformed_args, full_options);

    auto unsupported_args = make_args(activation, weights, output);
    unsupported_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h0;
    const MatmulStatus unsupported = matmul(unsupported_args, full_options);

    MatmulOptions invalid_options{};
    invalid_options.mode = MatmulInvocationMode::stripe_pipeline;
    invalid_options.job_capacity = 0;
    const MatmulStatus invalid = matmul(unsupported_args, invalid_options);

    return expect(malformed.code == MatmulStatusCode::invalid_contract, "malformed route rejected") &&
        expect(unsupported.ok(), "opened H0 route executes") &&
        expect(invalid.code == MatmulStatusCode::invalid_argument,
               "invalid options rejected");
}

bool test_cancel_and_failure() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    options.rmd_backend = RmdBackend::cpu_direct;

    auto cancel_args = make_args(activation, weights, output);
    auto cancel_execution = prepare_execution(&cancel_args, options);
    MatmulStripeCollector cancelled(1);
    if (!expect(cancel_execution.status().ok() && cancelled.start(cancel_execution), "cancel setup")) {
        return false;
    }
    const MatmulStatus cancel = cancelled.cancel();
    const MatmulStatus cancel_finish = cancelled.finish();
    const MatmulStatus cancelled_execution = finish_execution(cancel_execution);

    auto failure_args = make_args(activation, weights, output);
    auto failure_execution = prepare_execution(&failure_args, options);
    MatmulStripeCollector failed(1);
    failed.test_inject_thread_start_failure();
    const bool started = failed.start(failure_execution);

    return expect(cancel.code == MatmulStatusCode::cancelled &&
                      cancel_finish.code == MatmulStatusCode::cancelled &&
                      cancelled_execution.code == MatmulStatusCode::cancelled,
                  "cancellation propagates") &&
        expect(!started && failed.status().code == MatmulStatusCode::execution_failure &&
                   failure_execution.status().code == MatmulStatusCode::execution_failure,
               "startup failure propagates");
}

bool test_dense_no_output_is_failure() {
#if GGML_GEMMINI_ACTIVATION_BITS == 8 && GGML_GEMMINI_WEIGHT_BITS == 8
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 83.0f);
    const std::vector<float> before = output;
    auto args = make_args(activation, weights, output);
    args.B = nullptr;
    args.weight_i8_scale_active = false;
    args.weight_format =
        ggml_gemmini_args_t::im2p_weight_format_t::q8_h0;

    MatMul facade(args);
    const MatMulResult result = facade.run_full();
    return expect(result.status != MatMulStatus::success,
                  "dense no-output path reports failure") &&
        expect(output == before,
               "dense no-output path preserves destination transaction");
#else
    return true;
#endif
}

float float_from_bits(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool test_nonfinite_output_is_atomic() {
    const std::array<uint32_t, 4> nonfinite = {
        UINT32_C(0x7f800000),
        UINT32_C(0xff800000),
        UINT32_C(0x7fc00000),
        UINT32_C(0x7f800001),
    };
    bool ok = true;
    for (const uint32_t bits : nonfinite) {
        const float activation = float_from_bits(bits);
        const float weight = 1.0f;
        float output = 89.0f;
        ggml_gemmini_args_t args{};
        args.I = 1;
        args.J = 1;
        args.K = 1;
        args.A_fp32 = &activation;
        args.B_fp32 = &weight;
        args.f_out = &output;
        args.sA = 1;
        args.sB = 1;
        args.tiled_matmul_type = static_cast<tiled_matmul_type_t>(2);

        MatMul facade(args);
        const MatMulResult result = facade.run_full();
        if (result.status != MatMulStatus::invalid_contract || output != 89.0f) {
            uint32_t output_bits = 0;
            std::memcpy(&output_bits, &output, sizeof(output_bits));
            std::fprintf(stderr,
                         "finite-output input_bits=0x%08x status=%u output_bits=0x%08x\n",
                         bits, static_cast<unsigned>(result.status), output_bits);
        }
        ok = expect(result.status == MatMulStatus::invalid_contract,
                    "nonfinite dense output has typed rejection") && ok;
        ok = expect(output == 89.0f,
                    "nonfinite dense output preserves destination transaction") && ok;
    }
    return ok;
}

bool test_native_q4_hp1_cpu_dense_output() {
#if GGML_GEMMINI_ACTIVATION_BITS == 4 && GGML_GEMMINI_WEIGHT_BITS == 4
    ggml_gemmini_args_t args{};
    args.I = 3;
    args.J = 2;
    args.K = 32;
    args.sA = args.K;
    args.tiled_matmul_type = static_cast<tiled_matmul_type_t>(2);
    args.tile_I = 1;
    args.tile_J = 1;
    args.tile_K = 1;
    args.activation_rows_per_stripe = DIM;
    args.transpose_B = true;
    if (!args.A.allocate(args.I, args.K, 4)) {
        return expect(false, "Q4 activation allocation succeeds");
    }
    const std::array<size_t, 5> activation_k = {0, 1, 15, 16, 31};
    const std::array<int32_t, 5> activation_q = {-8, 7, -7, -1, 0};
    for (size_t index = 0; index < activation_k.size(); ++index) {
        if (!args.A.set(2, activation_k[index], activation_q[index])) {
            return expect(false, "asymmetric Q4 activation initialization succeeds");
        }
    }
    auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = {-1};

    std::array<block_q4_hp1, 2> weights{};
    for (block_q4_hp1 & weight : weights) {
        std::memset(weight.qs, 0x88, sizeof(weight.qs));
        weight.qs[0] = 0xf0;
        weight.qs[1] = 0x8a;
        weight.qs[15] = 0x71;
        weight.m = 2;
    }
    weights[0].channel_scale = 0.25f;
    weights[1].channel_scale = 0.5f;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1;
    args.q4_hp1_blocks = weights.data();
    args.native_block_count = weights.size();
    args.native_blocks_per_row = 1;
    args.native_weight_bytes = sizeof(weights);
    args.blocks_per_row = 1;
    args.blocks_K = 1;
    args.blocks_J = 2;
    args.blocks_I = 1;
    args.block_size_k = 32;

    residual::DirectStripeBuilder residual_builder;
    residual_builder.reset(0, 2, 1, args.K, args.J);
    const std::array<size_t, 4> residual_k = {0, 15, 16, 31};
    const std::array<int32_t, 4> residual_q = {1, -2, 3, 4};
    for (size_t index = 0; index < residual_k.size(); ++index) {
        if (!residual_builder.add_residual(0, residual_k[index], residual_q[index])) {
            return expect(false, "asymmetric Q4 residual initialization succeeds");
        }
    }
    meta.direct_residuals = {residual_builder.finish()};

    constexpr float sentinel = -777.0f;
    std::vector<float> output(17, sentinel);
    args.f_out = output.data();
    args.stride_f_out = 7;
    args.col_stride_f_out = 2;

    MatmulOptions options{};
    options.mode = MatmulInvocationMode::full;
    options.rmd_backend = RmdBackend::cpu_direct;
    const MatmulStatus status = matmul(args, options);
    if (!status.ok()) {
        std::fprintf(stderr, "native Q4_HP1 status=%u message=%s\n",
                     static_cast<unsigned>(status.code), status.message);
    }
    const float first_expected =
#if GGML_GEMMINI_ENABLE_RMD
        71.5f;
#else
        60.0f;
#endif
    const float second_expected =
#if GGML_GEMMINI_ENABLE_RMD
        143.0f;
#else
        120.0f;
#endif
    bool holes_preserved = true;
    const std::array<size_t, 6> logical = {0, 2, 7, 9, 14, 16};
    for (size_t index = 0; index < output.size(); ++index) {
        if (std::find(logical.begin(), logical.end(), index) == logical.end() &&
            output[index] != sentinel) {
            holes_preserved = false;
        }
    }
    return expect(status.ok(), "native Q4_HP1 CPU FULL succeeds") &&
        expect(output[0] == 0.0f && output[2] == 0.0f &&
                   output[7] == 0.0f && output[9] == 0.0f,
               "zero activation rows remain zero") &&
        expect(output[14] == first_expected && output[16] == second_expected,
               "raw Q4 split-half dense and final-float RMD oracle matches") &&
        expect(holes_preserved, "strided Q4 output preserves nonlogical destinations");
#else
    return true;
#endif
}

bool test_native_q4_repeating_bias() {
#if GGML_GEMMINI_ACTIVATION_BITS == 4 && GGML_GEMMINI_WEIGHT_BITS == 4
    ggml_gemmini_args_t args{};
    args.I = 2;
    args.J = 2;
    args.K = 32;
    args.sA = args.K;
    args.tiled_matmul_type = static_cast<tiled_matmul_type_t>(2);
    args.tile_I = 1;
    args.tile_J = 1;
    args.tile_K = 1;
    args.activation_rows_per_stripe = DIM;
    args.transpose_B = true;
    if (!args.A.allocate(args.I, args.K, 4)) {
        return expect(false, "Q4 bias activation allocation succeeds");
    }
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = {0};

    std::array<block_q4_hp1, 2> weights{};
    for (block_q4_hp1 & weight : weights) {
        std::memset(weight.qs, 0x88, sizeof(weight.qs));
        weight.m = 0;
        weight.channel_scale = 1.0f;
    }
    args.weight_format =
        ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1;
    args.q4_hp1_blocks = weights.data();
    args.native_block_count = weights.size();
    args.native_blocks_per_row = 1;
    args.native_weight_bytes = sizeof(weights);
    args.blocks_per_row = 1;
    args.blocks_K = 1;
    args.blocks_J = 2;
    args.blocks_I = 1;
    args.block_size_k = 32;

    const std::array<acc_t, 2> bias = {3, -5};
    args.D = bias.data();
    args.sD = args.J;
    args.repeating_bias = true;
    args.scale_D = 1;
    std::array<float, 4> output = {-9.0f, -9.0f, -9.0f, -9.0f};
    args.f_out = output.data();
    args.stride_f_out = args.J;
    args.col_stride_f_out = 1;

    MatmulOptions options{};
    options.mode = MatmulInvocationMode::full;
    options.rmd_backend = RmdBackend::cpu_direct;
    const MatmulStatus status = matmul(args, options);
    if (!status.ok()) {
        std::fprintf(stderr, "native Q4 bias status=%u message=%s\n",
                     static_cast<unsigned>(status.code), status.message);
    }
    return expect(status.ok(), "native Q4 repeating bias succeeds") &&
        expect(output == std::array<float, 4>{3.0f, -5.0f, 3.0f, -5.0f},
               "native Q4 repeating bias matches Gemmini epilogue");
#else
    return true;
#endif
}

bool test_native_q4_multiblock_final_float_oracle() {
#if GGML_GEMMINI_ACTIVATION_BITS == 4 && GGML_GEMMINI_WEIGHT_BITS == 4
    constexpr size_t rows = 2;
    constexpr size_t columns = 3;
    constexpr size_t depth = 64;
    constexpr size_t blocks_per_row = depth / 32;
    constexpr int16_t theta = -1;

    ggml_gemmini_args_t args{};
    args.I = rows;
    args.J = columns;
    args.K = depth;
    args.sA = depth;
    args.tiled_matmul_type = static_cast<tiled_matmul_type_t>(2);
    args.tile_I = 1;
    args.tile_J = 1;
    args.tile_K = 1;
    args.activation_rows_per_stripe = DIM;
    args.transpose_B = true;
    if (!args.A.allocate(rows, depth, 4)) {
        return expect(false, "multiblock Q4 activation allocation succeeds");
    }

    std::array<int32_t, rows * depth> activation_codes{};
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < depth; ++k) {
            const int32_t code =
                static_cast<int32_t>((i * 5 + k * 3) % 16) - 8;
            activation_codes[i * depth + k] = code;
            if (!args.A.set(i, k, code)) {
                return expect(false, "multiblock Q4 activation initialization succeeds");
            }
        }
    }

    auto & meta =
        args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = {theta};
    std::array<int32_t, rows * depth> residual_codes{};
    residual::DirectStripeBuilder residual_builder;
    residual_builder.reset(0, 0, rows, depth, columns);
    const std::array<std::tuple<size_t, size_t, int32_t>, 5> residuals = {{
        {0, 0, 2},
        {0, 32, -3},
        {1, 15, 4},
        {1, 31, -2},
        {1, 63, 3},
    }};
    for (const auto & [row, k, value] : residuals) {
        residual_codes[row * depth + k] = value;
        if (!residual_builder.add_residual(row, k, value)) {
            return expect(false, "multiblock Q4 residual initialization succeeds");
        }
    }
    meta.direct_residuals = {residual_builder.finish()};

    std::array<block_q4_hp1, columns * blocks_per_row> weights{};
    for (size_t j = 0; j < columns; ++j) {
        for (size_t block = 0; block < blocks_per_row; ++block) {
            block_q4_hp1 & weight = weights[j * blocks_per_row + block];
            weight.channel_scale = 0.25f * static_cast<float>(j + 1);
            weight.m = static_cast<int16_t>(block);
            for (size_t packed = 0; packed < 16; ++packed) {
                const int32_t low =
                    static_cast<int32_t>((j * 7 + block * 5 + packed * 3) % 16) - 8;
                const int32_t high =
                    static_cast<int32_t>((j * 11 + block * 3 + packed * 5 + 1) % 16) - 8;
                weight.qs[packed] = static_cast<uint8_t>(
                    (low + 8) | ((high + 8) << 4));
            }
        }
    }
    args.weight_format =
        ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1;
    args.q4_hp1_blocks = weights.data();
    args.native_block_count = weights.size();
    args.native_blocks_per_row = blocks_per_row;
    args.native_weight_bytes = sizeof(weights);
    args.blocks_per_row = blocks_per_row;
    args.blocks_K = blocks_per_row;
    args.blocks_J = columns;
    args.blocks_I = 1;
    args.block_size_k = 32;

    constexpr float sentinel = -321.0f;
    std::array<float, 12> output{};
    output.fill(sentinel);
    args.f_out = output.data();
    args.stride_f_out = 5;
    args.col_stride_f_out = 2;

    MatmulOptions options{};
    options.mode = MatmulInvocationMode::full;
    options.rmd_backend = RmdBackend::cpu_direct;
    test_reset_matmul_counters();
    const MatmulStatus status = matmul(args, options);
    const MatmulTestCounters counters = test_matmul_counters();
    if (!status.ok()) {
        std::fprintf(stderr, "multiblock Q4 status=%u message=%s\n",
                     static_cast<unsigned>(status.code), status.message);
        return false;
    }
    const bool uses_integer_blocks = expect(
        counters.native_integer_block_dots ==
                rows * columns * blocks_per_row &&
            counters.native_post_dot_scales ==
                rows * columns * blocks_per_row,
        "INT A4/Q4 uses integer block-dot then block-scale execution");

    bool values_match = true;
    bool holes_preserved = true;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j) {
            float expected = 0.0f;
            for (size_t k = 0; k < depth; ++k) {
                const size_t block = k / 32;
                const size_t local = k % 32;
                const block_q4_hp1 & weight =
                    weights[j * blocks_per_row + block];
                const uint8_t packed = weight.qs[local % 16];
                const int32_t weight_code =
                    static_cast<int32_t>(
                        local < 16 ? packed & 0x0f : packed >> 4) - 8;
                const float weight_value =
                    static_cast<float>(weight_code) *
                    std::ldexp(weight.channel_scale, weight.m);
                int32_t activation_code = activation_codes[i * depth + k];
#if GGML_GEMMINI_ENABLE_RMD
                activation_code += residual_codes[i * depth + k];
#endif
                const float activation_value =
                    std::ldexp(static_cast<float>(activation_code), theta);
                expected += activation_value * weight_value;
            }
            const float actual = output[i * 5 + j * 2];
            const float tolerance =
                1e-5f * std::max(1.0f, std::fabs(expected));
            values_match =
                std::fabs(actual - expected) <= tolerance && values_match;
        }
    }
    const std::array<size_t, rows * columns> logical = {0, 2, 4, 5, 7, 9};
    for (size_t index = 0; index < output.size(); ++index) {
        if (std::find(logical.begin(), logical.end(), index) == logical.end() &&
            output[index] != sentinel) {
            holes_preserved = false;
        }
    }
    return uses_integer_blocks && expect(values_match,
                  "multiblock Q4 final-float scalar oracle matches") &&
        expect(holes_preserved,
               "multiblock Q4 oracle preserves strided output holes");
#else
    return true;
#endif
}

bool test_native_q16_hp1_cpu_dense_output() {
#if GGML_GEMMINI_ACTIVATION_BITS == 16 && GGML_GEMMINI_WEIGHT_BITS == 16
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = 32;
    args.sA = args.K;
    args.tiled_matmul_type = static_cast<tiled_matmul_type_t>(2);
    args.tile_I = 1;
    args.tile_J = 1;
    args.tile_K = 1;
    args.activation_rows_per_stripe = DIM;
    args.transpose_B = true;
    if (!args.A.allocate(args.I, args.K, 16)) {
        return expect(false, "Q16 activation allocation succeeds");
    }
    for (size_t k = 0; k < args.K; ++k) {
        if (!args.A.set(0, k, 3)) {
            return expect(false, "Q16 activation initialization succeeds");
        }
    }
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = {-1};

    block_q16_hp1 weight{};
    std::fill(std::begin(weight.qs), std::end(weight.qs), int16_t{1});
    weight.m = 0;
    weight.channel_scale = 2.0f;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q16_hp1;
    args.q16_hp1_blocks = &weight;
    args.native_block_count = 1;
    args.native_blocks_per_row = 1;
    args.blocks_per_row = 1;
    args.blocks_K = 1;
    args.blocks_J = 1;
    args.blocks_I = 1;
    args.block_size_k = 32;
    args.native_weight_bytes = sizeof(weight);

    float output = -7.0f;
    args.f_out = &output;
    args.stride_f_out = 1;
    args.col_stride_f_out = 1;
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::full;
    options.rmd_backend = RmdBackend::cpu_direct;
    test_reset_matmul_counters();
    const MatmulStatus status = matmul(args, options);
    const MatmulTestCounters counters = test_matmul_counters();
    return expect(status.ok(), "native Q16_HP1 INT FULL succeeds") &&
        expect(counters.native_integer_block_dots == 1 &&
                   counters.native_post_dot_scales == 1,
               "INT A16/Q16 uses integer block-dot then block-scale execution") &&
        expect(output == 96.0f,
               "native Q16_HP1 INT FULL computes scaled dot product");
#else
    return true;
#endif
}

}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--invalid-geometry-probe") {
        const bool ok = test_invalid_geometry_rejects_before_allocation();
        const auto counters = test_matmul_counters();
        std::printf("INVALID_GEOMETRY status=%d constructions=%llu allocations=%llu\n",
                    static_cast<int>(MatmulStatusCode::invalid_contract),
                    static_cast<unsigned long long>(counters.execution_constructions),
                    static_cast<unsigned long long>(counters.allocation_attempts));
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--probe-removed-sequential") {
        const bool ok = test_removed_sequential_rejects_before_work();
        if (ok) {
            std::puts("PASS: removed sequential mode rejected; counters zero; sentinel preserved");
        }
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--case=correction-domain") {
        const bool ok = test_correction_domain_composition();
        if (ok) {
            std::puts("PASS: FULL/STRIPE correction domains, atomic failures, final saturation");
        }
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--case=transaction-atomicity") {
        const bool ok = test_rmd_failure_atomicity_suite();
        if (ok) {
            std::puts("PASS: FULL/STRIPE output transactions commit once or discard");
        }
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--case=native-q4-cpu") {
        const bool ok = test_native_q4_hp1_cpu_dense_output() &&
            test_native_q4_repeating_bias() &&
            test_native_q4_multiblock_final_float_oracle();
        if (ok) {
            std::puts("PASS: native Q4_HP1 INT dense output");
        }
        return ok ? 0 : 1;
    }
    if (argc == 2 &&
        std::string(argv[1]) == "--case=native-q4-multiblock") {
        const bool ok = test_native_q4_multiblock_final_float_oracle();
        if (ok) {
            std::puts("PASS: native Q4 multiblock final-float oracle");
        }
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--case=native-q16-cpu") {
        const bool ok = test_native_q16_hp1_cpu_dense_output();
        if (ok) {
            std::puts("PASS: native Q16_HP1 INT dense output");
        }
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--case=residual-guards") {
        const bool ok = test_rmd_disabled_pipeline_skips_correction() &&
            test_dense_rejects_residual_metadata();
        if (ok) {
            std::puts("PASS: residual disable and dense-only guards");
        }
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--case=dense-status") {
        const bool ok = test_dense_no_output_is_failure();
        if (ok) {
            std::puts("PASS: dense no-output path reports failure atomically");
        }
        return ok ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--case=finite-output") {
        const bool ok = test_nonfinite_output_is_atomic();
        if (ok) {
            std::puts("PASS: nonfinite dense output rejected atomically");
        }
        return ok ? 0 : 1;
    }
#if GGML_GEMMINI_ACTIVATION_BITS == 16 && GGML_GEMMINI_WEIGHT_BITS == 16
    if (!test_native_q16_hp1_cpu_dense_output()) {
        return 1;
    }
#else
    if (!test_removed_sequential_rejects_before_work() ||
        !test_invalid_geometry_rejects_before_allocation() ||
        !test_output_parity() || !test_single_row_pipeline() ||
        !test_native_q4_hp1_cpu_dense_output() ||
        !test_native_q4_repeating_bias() ||
        !test_native_q4_multiblock_final_float_oracle() ||
        !test_counter_hooks_connected() ||
        !test_cpu_direct_lifecycle_parity() ||
        !test_rmd_disabled_pipeline_skips_correction() ||
        !test_dense_rejects_residual_metadata() ||
        !test_correction_domain_composition() ||
        !test_rmd_failure_atomicity_suite() ||
        !test_ws_preflight_is_atomic_on_host() ||
        !test_dense_no_output_is_failure() ||
        !test_nonfinite_output_is_atomic() ||
        !test_bad_routes() || !test_cancel_and_failure()) {
        return 1;
    }
#endif
    std::puts("PASS: public mode contract/parity; malformed/unsupported/cancel/failure coverage");
    return 0;
}
