#define GGML_GEMMINI_TEST_OBSERVER 1

#include "../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp"

#include <cstdio>
#include <cstring>
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
    args.A = activation.data();
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

bool test_output_parity() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> full_output(6, 0.0f);
    std::vector<float> sequential_output(6, 0.0f);
    std::vector<float> pipeline_output(6, 0.0f);

    auto full_args = make_args(activation, weights, full_output);
    MatmulOptions full_options{};
    full_options.mode = MatmulInvocationMode::full;
    const MatmulStatus full = matmul(full_args, full_options);

    auto sequential_args = make_args(activation, weights, sequential_output);
    MatmulOptions sequential_options{};
    sequential_options.mode = MatmulInvocationMode::stripe_sequential;
    sequential_options.stripe_rows = 1;
    const MatmulStatus sequential = matmul(sequential_args, sequential_options);

    auto pipeline_args = make_args(activation, weights, pipeline_output);
    MatmulOptions pipeline_options{};
    pipeline_options.mode = MatmulInvocationMode::stripe_pipeline;
    pipeline_options.job_capacity = 2;
    auto execution = prepare_execution(&pipeline_args, pipeline_options);
    MatmulStripeCollector collector(2);
    if (!expect(execution.status().ok() && collector.start(execution), "pipeline starts")) {
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
        expect(sequential.ok(), "STRIPE_SEQUENTIAL succeeds") &&
        expect(accepted && collected.ok() && pipeline.ok(), "PIPELINE succeeds") &&
        expect(same_output(full_output, sequential_output), "FULL/STRIPE_SEQUENTIAL parity") &&
        expect(same_output(full_output, pipeline_output), "FULL/PIPELINE parity");
}

bool test_bad_routes() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);

    auto malformed_args = make_args(activation, weights, output);
    malformed_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    MatmulOptions full_options{};
    full_options.mode = MatmulInvocationMode::full;
    const MatmulStatus malformed = matmul(malformed_args, full_options);

    auto unsupported_args = make_args(activation, weights, output);
    unsupported_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h0;
    MatmulOptions stripe_options{};
    stripe_options.mode = MatmulInvocationMode::stripe_sequential;
    stripe_options.stripe_rows = 1;
    const MatmulStatus unsupported = matmul(unsupported_args, stripe_options);

    MatmulOptions invalid_options{};
    invalid_options.mode = MatmulInvocationMode::stripe_sequential;
    invalid_options.stripe_rows = 0;
    const MatmulStatus invalid = matmul(unsupported_args, invalid_options);

    return expect(malformed.code == MatmulStatusCode::invalid_contract, "malformed route rejected") &&
        expect(unsupported.code == MatmulStatusCode::unsupported_route, "unsupported route rejected") &&
        expect(invalid.code == MatmulStatusCode::invalid_argument, "invalid options rejected");
}

bool test_cancel_and_failure() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;

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

}

int main() {
    if (!test_output_parity() || !test_bad_routes() || !test_cancel_and_failure()) {
        return 1;
    }
    std::puts("PASS: FULL/STRIPE_SEQUENTIAL/PIPELINE parity; malformed/unsupported/cancel/failure coverage");
    return 0;
}
