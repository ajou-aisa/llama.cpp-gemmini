#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp"

#include <gemmini.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

namespace {

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

ggml_gemmini_args_t make_args(std::vector<elem_t> & activation,
                              std::vector<elem_t> & weights,
                              std::vector<float> & output) {
    ggml_gemmini_args_t args{};
    args.I = 3;
    args.J = 2;
    args.K = 2;
    args.A = activation.data();
    args.B = weights.data();
    args.sA = args.K;
    args.sB = args.J;
    args.f_out = output.data();
    args.col_stride_f_out = 1;
    args.stride_f_out = args.J;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    args.tiled_matmul_type = CPU;
    return args;
}

bool same_output(const std::vector<float> & actual, const std::vector<float> & expected) {
    return actual.size() == expected.size() &&
        std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0;
}

bool test_full_facade_status_and_output_match_legacy() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> legacy_output(6, 0.0f);
    std::vector<float> facade_output(6, 0.0f);
    ggml_gemmini_args_t legacy_args = make_args(activation, weights, legacy_output);
    ggml_gemmini_args_t facade_args = make_args(activation, weights, facade_output);

    ggml::gemmini::tiled_matmul_auto_im2p(&legacy_args);
    ggml::gemmini::MatMul facade(facade_args);
    const auto result = facade.run_full();

    return check(result.status == ggml::gemmini::MatMulStatus::success, "full facade status") &&
        check(result.capability == ggml::gemmini::MatMulCapability::supported, "full facade capability") &&
        check(same_output(facade_output, legacy_output), "full facade output differs from legacy matmul");
}

bool test_full_and_stripe_sequential_outputs_match() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> full_output(6, 0.0f);
    std::vector<float> stripe_output(6, 0.0f);
    auto full_args = make_args(activation, weights, full_output);
    auto stripe_args = make_args(activation, weights, stripe_output);
    MatmulOptions stripe_options{};
    stripe_options.mode = MatmulInvocationMode::stripe_sequential;
    stripe_options.stripe_rows = 1;
    stripe_options.rc_shards = 2;
    return check(matmul(full_args).ok(), "full public matmul") &&
        check(matmul(stripe_args, stripe_options).ok(), "stripe sequential public matmul") &&
        check(same_output(full_output, stripe_output), "full and stripe sequential output differs");
}

bool test_empty_tail_and_malformed_stripe_status() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    ggml::gemmini::MatMul facade(make_args(activation, weights, output));

    return check(facade.begin_stripes() == ggml::gemmini::MatMulStatus::success, "begin empty stripes") &&
        check(facade.finish_stripes() == ggml::gemmini::MatMulStatus::empty_stripes, "empty stripes") &&
        check(facade.begin_stripes() == ggml::gemmini::MatMulStatus::success, "restart after empty stripes") &&
        check(facade.run_stripe({ 2, 3 }) == ggml::gemmini::MatMulStatus::success, "tail stripe") &&
        check(facade.run_stripe({ 2, 1 }) == ggml::gemmini::MatMulStatus::malformed_stripe, "reversed stripe") &&
        check(facade.run_stripe({ 0, 4 }) == ggml::gemmini::MatMulStatus::malformed_stripe, "out-of-range stripe");
}

bool test_duplicate_and_overlap_stripe_status() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    ggml::gemmini::MatMul facade(make_args(activation, weights, output));

    return check(facade.begin_stripes() == ggml::gemmini::MatMulStatus::success, "begin duplicate stripes") &&
        check(facade.run_stripe({ 0, 2 }) == ggml::gemmini::MatMulStatus::success, "first stripe") &&
        check(facade.run_stripe({ 0, 2 }) == ggml::gemmini::MatMulStatus::duplicate_stripe, "duplicate stripe") &&
        check(facade.run_stripe({ 1, 3 }) == ggml::gemmini::MatMulStatus::overlapping_stripe, "overlapping stripe");
}

bool test_h2_and_hp2_stripe_capability_is_explicitly_unsupported() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    ggml_gemmini_args_t h2_args = make_args(activation, weights, output);
    h2_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h2;
    ggml_gemmini_args_t hp2_args = h2_args;
    hp2_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2;

    return check(ggml::gemmini::MatMul::stripe_capability(h2_args) ==
                     ggml::gemmini::MatMulCapability::unsupported,
                 "H2 stripe capability") &&
        check(ggml::gemmini::MatMul::stripe_capability(hp2_args) ==
                     ggml::gemmini::MatMulCapability::unsupported,
                 "HP2 stripe capability");
}

bool test_stripe_state_lifecycle() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    ggml::gemmini::MatMul facade(make_args(activation, weights, output));

    return check(facade.state() == ggml::gemmini::MatMulState::idle, "initial stripe state") &&
        check(facade.begin_stripes() == ggml::gemmini::MatMulStatus::success, "begin stripe state") &&
        check(facade.state() == ggml::gemmini::MatMulState::accepting_stripes, "accepting stripe state") &&
        check(facade.run_stripe({ 0, 2 }) == ggml::gemmini::MatMulStatus::success, "first lifecycle stripe") &&
        check(facade.run_stripe({ 2, 3 }) == ggml::gemmini::MatMulStatus::success, "tail lifecycle stripe") &&
        check(facade.finish_stripes() == ggml::gemmini::MatMulStatus::success, "finish stripes") &&
        check(facade.state() == ggml::gemmini::MatMulState::completed, "completed stripe state") &&
        check(facade.run_stripe({ 0, 1 }) == ggml::gemmini::MatMulStatus::invalid_state, "stripe after completion");
}

bool run_staged_job(ggml::gemmini::MatmulStripeJob & job) {
    using namespace ggml::gemmini;
    return check(prepare_compensation(job).ok(), "prepare compensation") &&
        check(execute_dense_stripe(job).ok(), "dense stripe") &&
        check(execute_compensation_shard(job).ok(), "compensation shard") &&
        check(finalize_stripe(job).ok(), "finalize stripe");
}

bool test_public_contract_shape() {
    using namespace ggml::gemmini;
    const MatmulOptions defaults{};
    const MatmulStatus statuses[] = {
        { MatmulStatusCode::success, "success" },
        { MatmulStatusCode::invalid_argument, "invalid argument" },
        { MatmulStatusCode::invalid_contract, "invalid contract" },
        { MatmulStatusCode::unsupported_route, "unsupported route" },
        { MatmulStatusCode::unsupported_backend, "unsupported backend" },
        { MatmulStatusCode::unsupported_invocation, "unsupported invocation" },
        { MatmulStatusCode::invalid_state, "invalid state" },
        { MatmulStatusCode::out_of_memory, "out of memory" },
        { MatmulStatusCode::execution_failure, "execution failure" },
        { MatmulStatusCode::cancelled, "cancelled" },
    };
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.dense_threads = 2;
    options.rc_shards = 3;
    options.validation = true;
    options.profiling = true;
    options.force = true;
    options.job_capacity = 2;

    const int32_t residual[] = { 4, 5 };
    MatmulStripeInput input(1, 3, 7, residual, 2);
    MatmulJobMetrics metrics{};
    return check(defaults.job_capacity == 4, "default job capacity") &&
        check(statuses[0].ok(), "success status ok") &&
        check(!statuses[1].ok(), "failure status not ok") &&
        check(std::string_view(statuses[2].message) == "invalid contract", "status message") &&
        check(options.mode == MatmulInvocationMode::stripe_pipeline && options.dense_threads == 2 &&
                  options.rc_shards == 3 && options.validation && options.profiling && options.force &&
                  options.job_capacity == 2,
              "matmul options") &&
        check(input.row_begin() == 1 && input.row_end() == 3 && input.stripe_id() == 7 &&
                  input.residual() == residual && input.residual_count() == 2,
              "stripe input metadata") &&
        check(metrics.la.count == 0 && metrics.sf.count == 0 && metrics.handoff.count == 0 &&
                  metrics.ws.count == 0 && metrics.rc_prepare.count == 0 &&
                  metrics.rc_compute.count == 0 && metrics.rc_finalize.count == 0,
              "job metric storage");
}

bool test_route_capability_table() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    const auto key = detail::normalize_route(args);
    const auto caps = detail::route_capabilities(args);
    if (!check(key.activation == detail::ActivationRoute::fp32, "route activation normalization") ||
        !check(key.weight == detail::WeightRoute::tensor_i8, "route weight normalization") ||
        !check(caps.full && caps.sliced_dense && caps.sliced_compensation && caps.external_rc_shards,
               "route capability support")) {
        return false;
    }
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h0;
    if (!check(!detail::route_capabilities(args).full, "Q8_H0 explicit capability reject")) {
        return false;
    }
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h2;
    const auto deprecated = detail::route_capabilities(args);
    if (!check(detail::normalize_route(args).weight == detail::WeightRoute::q8_h2,
               "Q8_H2 route normalization") ||
        !check(deprecated.full && !deprecated.sliced_dense && deprecated.deprecated,
               "Q8_H2 full-only deprecated capability")) {
        return false;
    }
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    if (!check(detail::normalize_route(args).weight == detail::WeightRoute::q8_h1,
               "Q8_H1 route normalization") ||
        !check(detail::route_capabilities(args).full, "Q8_H1 full capability")) {
        return false;
    }
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
    if (!check(detail::normalize_route(args).weight == detail::WeightRoute::q8_hp1,
               "Q8_HP1 route normalization") ||
        !check(detail::route_capabilities(args).full, "Q8_HP1 full capability")) {
        return false;
    }
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2;
    if (!check(detail::normalize_route(args).weight == detail::WeightRoute::q8_hp2,
               "Q8_HP2 route normalization") ||
        !check(detail::route_capabilities(args).deprecated &&
                   !detail::route_capabilities(args).sliced_compensation,
               "Q8_HP2 full-only deprecated capability")) {
        return false;
    }
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel;
    if (!check(detail::normalize_route(args).weight == detail::WeightRoute::q8_channel_direct,
               "Q8_CHANNEL direct normalization")) {
        return false;
    }
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
    if (!check(detail::normalize_route(args).weight == detail::WeightRoute::q8_channel_sidecar,
               "Q8_CHANNEL sidecar normalization")) {
        return false;
    }
    args.tiled_matmul_type = WS;
    if (!check(detail::normalize_route(args).backend == detail::BackendRoute::gemmini_ws &&
                   detail::route_capabilities(args).full,
               "Gemmini WS backend capability")) {
        return false;
    }
    args.tiled_matmul_type = OS;
    return check(detail::normalize_route(args).backend == detail::BackendRoute::gemmini_os &&
                     !detail::route_capabilities(args).full,
                 "Gemmini OS explicit unsupported capability");
}

bool test_malformed_route_contract_rejected() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    ggml::gemmini::MatMul facade(args);
    const auto result = facade.run_full();
    return check(result.status == ggml::gemmini::MatMulStatus::invalid_contract,
                 "malformed native weight contract rejected");
}

bool test_bounded_pipeline_slots_and_reuse() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 2;
    options.rc_shards = 4;
    options.profiling = true;
    auto execution = prepare_execution(make_args(activation, weights, output), options);
    auto first = capture_stripe(execution, { 0, 1 });
    auto second = capture_stripe(execution, { 1, 2 });
    auto blocked = capture_stripe(execution, { 2, 3 });

    if (!check(first.status().ok() && second.status().ok(), "pipeline captures") ||
        !check(blocked.status().code == MatmulStatusCode::out_of_memory, "bounded backpressure") ||
        !check(finish_execution(execution).code == MatmulStatusCode::invalid_state, "finish with live jobs") ||
        !run_staged_job(first)) {
        return false;
    }
    auto tail = capture_stripe(execution, { 2, 3 });
    const bool passed = run_staged_job(second) && run_staged_job(tail) &&
        check(finish_execution(execution).ok(), "pipeline finish") &&
        check(first.metrics().handoff.count == 1 && first.metrics().ws.count == 1 &&
                  first.metrics().rc_prepare.count == 1 && first.metrics().rc_compute.count == 1 &&
                  first.metrics().rc_finalize.count == 1,
              "pipeline metric counters");
    if (passed) {
        std::puts("PASS edge: pipeline=externally-staged capacity=2 backpressure=out_of_memory slot_reuse=yes");
    }
    return passed;
}

bool test_staged_contract_errors() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;

    auto order_execution = prepare_execution(make_args(activation, weights, output), options);
    auto early = capture_stripe(order_execution, { 0, 1 });
    if (!check(finalize_stripe(early).code == MatmulStatusCode::invalid_state,
               "finalize before compensation")) {
        return false;
    }

    auto contract_execution = prepare_execution(make_args(activation, weights, output), options);
    auto malformed = capture_stripe(contract_execution, { 1, 1 });
    auto first = capture_stripe(contract_execution, { 0, 1 });
    auto duplicate = capture_stripe(contract_execution, { 0, 1 });
    auto overlap = capture_stripe(contract_execution, { 0, 2 });
    if (!check(malformed.status().code == MatmulStatusCode::invalid_argument, "malformed stripe") ||
        !check(duplicate.status().code == MatmulStatusCode::invalid_contract, "duplicate stripe") ||
        !check(overlap.status().code == MatmulStatusCode::invalid_contract, "overlapping stripe") ||
        !run_staged_job(first)) {
        return false;
    }
    auto tail = capture_stripe(contract_execution, { 2, 3 });
    const bool passed = run_staged_job(tail) &&
        check(finish_execution(contract_execution).code == MatmulStatusCode::invalid_contract,
              "missing stripe at finish") &&
        check(matmul(make_args(activation, weights, output),
                     { MatmulInvocationMode::stripe_pipeline }).code ==
                  MatmulStatusCode::unsupported_invocation,
              "automatic pipeline is explicit unsupported invocation");
    if (passed) {
        std::puts("PASS edge: duplicate=invalid_contract overlap=invalid_contract missing=invalid_contract "
                  "early_finalize=invalid_state automatic_pipeline=unsupported_invocation");
    }
    return passed;
}

bool test_live_pipeline_worker() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 2;
    auto execution = prepare_execution(make_args(activation, weights, output), options);
    MatmulStripeCollector collector(2);
    if (!check(execution.status().ok() && collector.start(execution), "live worker start")) {
        return false;
    }
    const auto * sink = collector.sink();
    if (!check(sink->on_ready(sink->user_data, { 0, 0, 2, nullptr, 0, 10, 20, 30, 50, 40, 60 }), "live worker capture") ||
        !check(sink->on_ready(sink->user_data, { 1, 2, 3, nullptr, 0, 11, 21, 31, 51, 41, 61 }), "live worker tail capture") ||
        !check(collector.finish().ok(), "live worker finish") ||
        !check(collector.profiles().size() == 2, "live worker stripe profiles") ||
        !check(collector.profiles()[0].la_cycles == 10 && collector.profiles()[0].la3_cycles == 20 &&
                   collector.profiles()[0].sf_cycles == 20,
               "live worker producer profile") ||
        !check(collector.profiles()[0].ws_start_ns < collector.profiles()[0].ws_end_ns &&
                   collector.profiles()[0].rc_start_ns < collector.profiles()[0].rc_end_ns,
               "live worker stage intervals") ||
        !check(finish_execution(execution).ok(), "live worker execution finish")) {
        return false;
    }
    std::puts("PASS edge: pipeline=live-worker capture->dense->rc->finish");
    return true;
}

bool run_captured_compensation(size_t shard_count, std::vector<float> & output) {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;
    options.rc_shards = shard_count;
    auto execution = prepare_execution(make_args(activation, weights, output), options);
    auto job = capture_stripe(execution, { 0, 3, 0 }, std::vector<quants::QactOutlier>{{ 0, 0, 2 }});
    if (!job.status().ok() || !prepare_compensation(job).ok() ||
        !execute_dense_stripe(job).ok())
        return false;
    const size_t actual_shards = std::max<size_t>(1, std::min(shard_count, size_t {2}));
    for (size_t shard = 0; shard < actual_shards; ++shard)
        if (!execute_compensation_shard(job, shard, actual_shards).ok())
            return false;
    return finalize_stripe(job).ok() && finish_execution(execution).ok();
}

bool test_compensation_shard_output_is_bitwise_stable() {
    std::vector<float> one(6, 0.0f);
    std::vector<float> four(6, 0.0f);
    const bool one_ok = run_captured_compensation(1, one);
    const bool four_ok = run_captured_compensation(4, four);
    if (one_ok && four_ok && !same_output(one, four)) {
        std::fprintf(stderr, "one=%g,%g,%g,%g,%g,%g four=%g,%g,%g,%g,%g,%g\n",
                     one[0], one[1], one[2], one[3], one[4], one[5],
                     four[0], four[1], four[2], four[3], four[4], four[5]);
    }
    return check(one_ok, "single compensation shard") &&
        check(four_ok, "multi compensation shards") &&
        check(same_output(one, four), "compensation shard output differs");
}

bool run_live_worker_compensation(size_t shard_count, std::vector<float> & output) {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 2;
    options.rc_shards = shard_count;
    auto execution = prepare_execution(make_args(activation, weights, output), options);
    MatmulStripeCollector collector(2);
    if (!execution.status().ok() || !collector.start(execution)) {
        return false;
    }
    quants::QactOutlier outlier{ 0, 0, 2 };
    const auto * sink = collector.sink();
    const bool captured = sink->on_ready(
        sink->user_data, { 0, 0, 3, &outlier, 1, 10, 20, 30, 50 });
    return captured && collector.finish().ok() && finish_execution(execution).ok();
}

bool test_live_worker_parallel_compensation_is_bitwise_stable() {
    std::vector<float> one(6, 0.0f);
    std::vector<float> four(6, 0.0f);
    const bool one_ok = run_live_worker_compensation(1, one);
    const bool four_ok = run_live_worker_compensation(4, four);
    return check(one_ok, "live single-shard compensation") &&
        check(four_ok, "live parallel-shard compensation") &&
        check(same_output(one, four), "live parallel compensation output differs");
}

}

int main(int argc, char ** argv) {
    const bool edge_only = argc == 2 && std::string_view(argv[1]) == "--edge";
    const bool edge = test_public_contract_shape() && test_route_capability_table() &&
        test_malformed_route_contract_rejected() &&
        test_bounded_pipeline_slots_and_reuse() &&
        test_live_pipeline_worker() && test_compensation_shard_output_is_bitwise_stable() &&
        test_live_worker_parallel_compensation_is_bitwise_stable() &&
        test_staged_contract_errors();
    if (edge_only) {
        return edge ? 0 : 1;
    }
    return edge && test_full_facade_status_and_output_match_legacy() &&
            test_full_and_stripe_sequential_outputs_match() &&
            test_empty_tail_and_malformed_stripe_status() &&
            test_duplicate_and_overlap_stripe_status() &&
            test_h2_and_hp2_stripe_capability_is_explicitly_unsupported() &&
            test_stripe_state_lifecycle()
        ? 0
        : 1;
}
