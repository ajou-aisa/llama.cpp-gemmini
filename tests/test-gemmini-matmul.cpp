#define GGML_GEMMINI_TEST_OBSERVER 1

#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/dispatch.hpp"

#include <gemmini.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <future>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

bool extract_u64_field(const std::string &json, const char *key, uint64_t &value) {
    const std::string needle = std::string("\"") + key + "\":";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    size_t cursor = pos + needle.size();
    size_t end = cursor;
    while (end < json.size() && json[end] >= '0' && json[end] <= '9') {
        ++end;
    }
    if (end == cursor) {
        return false;
    }
    value = std::strtoull(json.c_str() + cursor, nullptr, 10);
    return true;
}

bool extract_double_field(const std::string &json, const char *key, double &value) {
    const std::string needle = std::string("\"") + key + "\":";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    char * end = nullptr;
    value = std::strtod(json.c_str() + pos + needle.size(), &end);
    return end != json.c_str() + pos + needle.size();
}

bool extract_string_field(const std::string &json, const char *key, std::string &value) {
    const std::string needle = std::string("\"") + key + "\":\"";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    const size_t start = pos + needle.size();
    const size_t end = json.find('"', start);
    if (end == std::string::npos) {
        return false;
    }
    value = json.substr(start, end - start);
    return true;
}

bool has_array_field(const std::string &json, const char *key) {
    return json.find(std::string("\"") + key + "\":[") != std::string::npos;
}

bool has_object_field(const std::string &json, const char *key) {
    return json.find(std::string("\"") + key + "\":{") != std::string::npos;
}

bool has_null_field(const std::string &json, const char *key) {
    return json.find(std::string("\"") + key + "\":null") != std::string::npos;
}

ggml::gemmini::quants::act::exsia::StripeReadyEvent make_ready_event(
    size_t stripe_id,
    size_t row_begin,
    size_t row_end,
    const ggml::gemmini::quants::QactOutlier *outliers = nullptr,
    size_t outlier_count = 0,
    uint64_t local_start_cycle = 0,
    uint64_t local_end_cycle = 0,
    uint64_t folding_start_cycle = 0,
    uint64_t folding_end_cycle = 0,
    uint64_t local_group3_start_cycle = 0,
    uint64_t local_group3_end_cycle = 0) {
    ggml::gemmini::quants::act::exsia::StripeReadyEvent event{};
    event.stripe_id = stripe_id;
    event.slot = stripe_id % 2;
    event.row_begin = row_begin;
    event.row_end = row_end;
    event.outliers = outliers;
    event.outlier_count = outlier_count;
    event.local_start_cycle = local_start_cycle;
    event.local_end_cycle = local_end_cycle;
    event.folding_start_cycle = folding_start_cycle;
    event.folding_end_cycle = folding_end_cycle;
    event.local_group3_start_cycle = local_group3_start_cycle;
    event.local_group3_end_cycle = local_group3_end_cycle;
    return event;
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
    args.act_quant.storage().emplace<ggml::gemmini::quants::act::tensor::Meta>().scale = 1.0f;
    return args;
}

bool same_output(const std::vector<float> & actual, const std::vector<float> & expected) {
    return actual.size() == expected.size() &&
        std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0;
}

size_t expected_persistent_rc_workers(size_t requested_shards, size_t columns) {
    const size_t hw = std::thread::hardware_concurrency();
    const size_t budget = hw == 0 ? 1 : (hw > 2 ? hw - 2 : size_t {1});
    return std::min({
        std::max(size_t {1}, requested_shards),
        std::max(size_t {1}, columns),
        budget,
    });
}

struct ScopedEnvVar {
    explicit ScopedEnvVar(const char * name) : name_(name) {
        if (const char * value = std::getenv(name_)) {
            old_value_ = value;
            had_value_ = true;
        }
    }

    ~ScopedEnvVar() {
        if (had_value_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

    void set(const char * value) const {
        setenv(name_, value, 1);
    }

    void clear() const {
        unsetenv(name_);
    }

private:
    const char * name_;
    std::string old_value_;
    bool had_value_ = false;
};

bool test_full_facade_status_and_output_match_legacy() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> legacy_output(6, 0.0f);
    std::vector<float> facade_output(6, 0.0f);
    ggml_gemmini_args_t legacy_args = make_args(activation, weights, legacy_output);
    ggml_gemmini_args_t facade_args = make_args(activation, weights, facade_output);

    ggml::gemmini::tiled_matmul_auto_baseline(
        &legacy_args,
        ggml::gemmini::baseline_activation_quant_t::TENSOR,
        ggml::gemmini::baseline_weight_quant_t::TENSOR);
    ggml::gemmini::MatMul facade(facade_args);
    const auto result = facade.run_full();

    return check(result.status == ggml::gemmini::MatMulStatus::success, "full facade status") &&
        check(result.capability == ggml::gemmini::MatMulCapability::supported, "full facade capability") &&
        check(same_output(facade_output, legacy_output), "full facade output differs from legacy matmul");
}

bool test_fp32_full_facade_matches_legacy() {
    const std::vector<float> activation = { 1.0f, -2.0f, 0.5f, 3.0f,
                                            2.0f, 1.5f, -1.0f, 4.0f };
    const std::vector<float> weights = { 0.25f, 2.0f, -1.0f, 0.5f,
                                         1.0f, -0.5f, 3.0f, 2.0f,
                                         -2.0f, 1.0f, 0.25f, -1.5f };
    std::vector<float> legacy_output(6, 0.0f);
    std::vector<float> facade_output(6, 0.0f);
    std::vector<float> stripe_output(6, 0.0f);

    ggml::gemmini::matmul_cpu_fp(false, true, 2, 3, 4, activation.data(), weights.data(), nullptr,
                                 legacy_output.data(), 4, 4, 0, 3);

    ggml_gemmini_args_t args{};
    args.I = 2;
    args.J = 3;
    args.K = 4;
    args.A_fp32 = activation.data();
    args.B_fp32 = weights.data();
    args.sA = 4;
    args.sB = 4;
    args.f_out = facade_output.data();
    args.stride_f_out = 3;
    args.tiled_matmul_type = CPU;
    ggml::gemmini::MatMul facade(args);
    const auto result = facade.run_full();

    ggml_gemmini_args_t stripe_args = args;
    stripe_args.f_out = stripe_output.data();
    ggml::gemmini::MatmulOptions stripe_options{};
    stripe_options.mode = ggml::gemmini::MatmulInvocationMode::stripe_sequential;
    stripe_options.stripe_rows = 1;
    const auto stripe_result = ggml::gemmini::matmul(stripe_args, stripe_options);

    const auto route = ggml::gemmini::detail::normalize_route(args);
    const auto capabilities = ggml::gemmini::detail::route_capabilities(args);
    return check(result.status == ggml::gemmini::MatMulStatus::success, "FP32 facade status") &&
        check(route.activation == ggml::gemmini::detail::ActivationRoute::fp32 &&
              route.weight == ggml::gemmini::detail::WeightRoute::fp32,
              "FP32 route normalization") &&
        check(capabilities.full && capabilities.sliced_dense && capabilities.sliced_compensation,
              "FP32 route exposes full and sequential facade contracts") &&
        check(same_output(facade_output, legacy_output), "FP32 facade output differs from legacy matmul") &&
        check(stripe_result.ok(), "FP32 stripe facade status") &&
        check(same_output(stripe_output, legacy_output), "FP32 stripe facade output differs from legacy matmul");
}

bool test_fp32_stripe_route_skips_quantized_compensation() {
    using namespace ggml::gemmini;
    const std::vector<float> activation = { 1.0f, -2.0f, 0.5f, 3.0f,
                                            2.0f, 1.5f, -1.0f, 4.0f };
    const std::vector<float> weights = { 0.25f, 2.0f, -1.0f, 0.5f,
                                         1.0f, -0.5f, 3.0f, 2.0f,
                                         -2.0f, 1.0f, 0.25f, -1.5f };
    std::vector<float> expected(6, 0.0f);
    std::vector<float> output(6, 0.0f);
    matmul_cpu_fp(false, true, 2, 3, 4, activation.data(), weights.data(), nullptr,
                  expected.data(), 4, 4, 0, 3);

    ggml_gemmini_args_t args{};
    args.I = 2;
    args.J = 3;
    args.K = 4;
    args.A_fp32 = activation.data();
    args.B_fp32 = weights.data();
    args.sA = 4;
    args.sB = 4;
    args.f_out = output.data();
    args.stride_f_out = 3;
    args.tiled_matmul_type = CPU;
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;
    options.stripe_rows = 1;

    const auto status = matmul(args, options);
    return check(status.ok(), "FP32 sequential stripe does not enter quantized compensation") &&
        check(same_output(output, expected), "FP32 sequential stripe output parity");
}

bool test_baseline_activation_route_facade_parity() {
    using namespace ggml::gemmini;
    using Route = baseline_activation_quant_t;

    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    const auto run = [&](auto meta, Route route, const char * label,
                         MatMulCapability expected_stripe) {
        std::vector<float> legacy_output(6, 0.0f);
        std::vector<float> facade_output(6, 0.0f);
        std::vector<float> stripe_output(6, 0.0f);
        auto legacy_args = make_args(activation, weights, legacy_output);
        legacy_args.act_quant.storage() = std::move(meta);
        if (route != Route::TENSOR) {
            legacy_args.transpose_B = true;
            legacy_args.sB = legacy_args.K;
        }
        auto facade_args = legacy_args;
        facade_args.f_out = facade_output.data();

        tiled_matmul_auto_baseline(&legacy_args, route, baseline_weight_quant_t::TENSOR);
        const auto result = MatMul(facade_args).run_full();
        auto stripe_args = facade_args;
        stripe_args.f_out = stripe_output.data();
        MatmulOptions stripe_options{};
        stripe_options.mode = MatmulInvocationMode::stripe_sequential;
        stripe_options.stripe_rows = 1;
        const auto stripe_result = matmul(stripe_args, stripe_options);
        const bool stripe_contract = expected_stripe == MatMulCapability::supported ?
            check(stripe_result.ok(), "baseline route stripe execution") :
            check(stripe_result.code == MatmulStatusCode::unsupported_route,
                  "baseline route stripe explicitly unsupported");
        return check(result.status == MatMulStatus::success, label) &&
            check(same_output(facade_output, legacy_output), "baseline route facade output differs") &&
            check(MatMul::stripe_capability(facade_args) == expected_stripe,
                  "baseline route stripe capability") &&
            stripe_contract &&
            (expected_stripe == MatMulCapability::unsupported ||
             check(same_output(stripe_output, legacy_output), "baseline route stripe output differs"));
    };

    quants::act::tensor::Meta tensor_meta;
    tensor_meta.scale = 1.0f;
    quants::act::token::Meta token_meta;
    token_meta.scales = { 1.0f, 1.0f, 1.0f };
    quants::act::block::Meta block_meta;
    block_meta.scales = { 1.0f, 1.0f, 1.0f };
    quants::act::stripe::Meta stripe_meta;
    stripe_meta.scales = { 1.0f };

    return run(std::move(tensor_meta), Route::TENSOR, "TENSOR baseline facade", MatMulCapability::supported) &&
        run(std::move(token_meta), Route::TOKEN, "TOKEN baseline facade", MatMulCapability::unsupported) &&
        run(std::move(block_meta), Route::BLOCK, "BLOCK baseline facade", MatMulCapability::unsupported) &&
        run(std::move(stripe_meta), Route::BLOCK, "STRIPE baseline facade", MatMulCapability::unsupported);
}

bool test_j131_tail_stripe_parity() {
    using namespace ggml::gemmini;
    constexpr size_t rows = 65;
    constexpr size_t columns = 131;
    constexpr size_t depth = 2;
    std::vector<float> activation(rows * depth, 1.0f);
    std::vector<float> weights(columns * depth, 1.0f);
    std::vector<float> full_output(rows * columns, 0.0f);
    std::vector<float> stripe_output(rows * columns, 0.0f);

    ggml_gemmini_args_t full_args{};
    full_args.I = rows;
    full_args.J = columns;
    full_args.K = depth;
    full_args.A_fp32 = activation.data();
    full_args.B_fp32 = weights.data();
    full_args.sA = depth;
    full_args.sB = columns;
    full_args.f_out = full_output.data();
    full_args.stride_f_out = columns;
    full_args.tiled_matmul_type = CPU;

    auto stripe_args = full_args;
    stripe_args.f_out = stripe_output.data();
    MatmulOptions stripe_options{};
    stripe_options.mode = MatmulInvocationMode::stripe_sequential;
    stripe_options.stripe_rows = 1;
    MatmulOptions full_options{};
    full_options.mode = MatmulInvocationMode::full;
    const auto full_result = matmul(full_args, full_options);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;
    options.stripe_rows = 63;
    const auto stripe_result = matmul(stripe_args, options);

    return check(full_result.ok(), "J=131 full status") &&
        check(stripe_result.ok(), "J=131 tail stripe status") &&
        check(same_output(stripe_output, full_output), "J=131 tail stripe differs from full");
}

bool test_fp32_shape_and_stride_matrix() {
    using namespace ggml::gemmini;
    constexpr size_t row_values[] = { 1, 63, 64, 65, 256 };
    constexpr size_t column_values[] = { 1, 4, 8, 131 };
    constexpr size_t depth_values[] = { 31, 32, 33, 63, 64, 65 };

    for (const size_t rows : row_values) {
        for (const size_t columns : column_values) {
            for (const size_t depth : depth_values) {
                const size_t output_stride = columns + 3;
                std::vector<float> activation(rows * depth);
                std::vector<float> weights(depth * columns);
                std::vector<float> full_output(rows * output_stride, -7.0f);
                std::vector<float> stripe_output(rows * output_stride, -7.0f);
                for (size_t i = 0; i < activation.size(); ++i) {
                    activation[i] = static_cast<float>(static_cast<int>(i % 11) - 5);
                }
                for (size_t i = 0; i < weights.size(); ++i) {
                    weights[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;
                }

                ggml_gemmini_args_t full_args{};
                full_args.I = rows;
                full_args.J = columns;
                full_args.K = depth;
                full_args.A_fp32 = activation.data();
                full_args.B_fp32 = weights.data();
                full_args.sA = depth;
                full_args.sB = depth;
                full_args.f_out = full_output.data();
                full_args.stride_f_out = output_stride;
                full_args.tiled_matmul_type = CPU;

                auto stripe_args = full_args;
                stripe_args.f_out = stripe_output.data();
                MatmulOptions full_options{};
                full_options.mode = MatmulInvocationMode::full;
                const auto full_status = matmul(full_args, full_options);
                MatmulOptions options{};
                options.mode = MatmulInvocationMode::stripe_sequential;
                options.stripe_rows = 63;
                const auto stripe_status = matmul(stripe_args, options);
                if (!check(full_status.ok() && stripe_status.ok(), "FP32 shape matrix status") ||
                    !check(same_output(full_output, stripe_output), "FP32 shape matrix parity")) {
                    std::fprintf(stderr, "shape matrix failed I=%zu J=%zu K=%zu\n",
                                 rows, columns, depth);
                    return false;
                }
            }
        }
    }
    return true;
}

bool test_live_pipeline_multistripe_matches_full() {
    using namespace ggml::gemmini;
    constexpr size_t rows = 130;
    constexpr size_t columns = 128;
    constexpr size_t depth = 2048;
    std::vector<elem_t> activation(rows * depth, 1);
    std::vector<elem_t> weights(columns * depth, 1);
    std::vector<float> full_output(rows * columns, 0.0f);
    std::vector<float> pipeline_output(rows * columns, 0.0f);
    auto full_args = make_args(activation, weights, full_output);
    full_args.I = rows;
    full_args.J = columns;
    full_args.K = depth;
    full_args.sA = depth;
    full_args.sB = columns;
    full_args.stride_f_out = columns;
    auto & full_meta = full_args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    full_meta.theta = { 0 };
    const auto full_result = MatMul(full_args).run_full();

    auto pipeline_args = make_args(activation, weights, pipeline_output);
    pipeline_args.I = rows;
    pipeline_args.J = columns;
    pipeline_args.K = depth;
    pipeline_args.sA = depth;
    pipeline_args.sB = columns;
    pipeline_args.stride_f_out = columns;
    auto & pipeline_meta = pipeline_args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    pipeline_meta.theta = { 0, 0 };
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 2;
    options.rc_shards = 2;
    options.profiling = true;
    auto execution = prepare_execution(&pipeline_args, options);
    MatmulStripeCollector collector(2);
    if (!check(full_result.status == MatMulStatus::success, "multistripe full reference") ||
        !check(execution.status().ok() && collector.start(execution), "multistripe pipeline start")) {
        return false;
    }

    const auto * sink = collector.sink();
    const bool captured =
        sink->on_ready(sink->user_data, make_ready_event(0, 0, 32)) &&
        sink->on_ready(sink->user_data, make_ready_event(1, 32, 64)) &&
        sink->on_ready(sink->user_data, make_ready_event(2, 64, 96)) &&
        sink->on_ready(sink->user_data, make_ready_event(3, 96, rows));
    const auto collector_status = collector.finish();
    const auto execution_status = finish_execution(execution);
    const auto profiles = collector.profiles();
    for (const auto & profile : profiles) {
        std::printf(
            "[matmul.stripe.synthetic] stripe_id=%zu row_begin=%zu row_end=%zu "
            "la3_cycles=%llu sf_cycles=%llu handoff_ns=%llu "
            "ws_start_ns=%llu ws_end_ns=%llu rc_start_ns=%llu rc_end_ns=%llu rc_shards=%zu\n",
            profile.stripe_id, profile.row_begin, profile.row_end,
            static_cast<unsigned long long>(profile.la3_cycles),
            static_cast<unsigned long long>(profile.sf_cycles),
            static_cast<unsigned long long>(profile.handoff.nanoseconds),
            static_cast<unsigned long long>(profile.ws_start_ns),
            static_cast<unsigned long long>(profile.ws_end_ns),
            static_cast<unsigned long long>(profile.rc_start_ns),
            static_cast<unsigned long long>(profile.rc_end_ns), profile.rc_shards);
    }
    if (!same_output(pipeline_output, full_output)) {
        for (size_t i = 0; i < pipeline_output.size(); ++i) {
            if (pipeline_output[i] != full_output[i]) {
                std::fprintf(stderr, "mismatch i=%zu row=%zu pipeline=%g full=%g\n",
                             i, i / columns, pipeline_output[i], full_output[i]);
                break;
            }
        }
    }
    return check(captured, "multistripe ready events") &&
        check(collector_status.ok(), "multistripe collector finish") &&
        check(execution_status.ok(), "multistripe execution finish") &&
        check(same_output(pipeline_output, full_output), "multistripe pipeline differs from full") &&
        check(profiles.size() == 4, "multistripe profile count") &&
        check(profiles.size() == 4 &&
                  profiles[0].stripe_id == 0 && profiles[0].row_begin == 0 && profiles[0].row_end == 32 &&
                  profiles[1].stripe_id == 1 && profiles[1].row_begin == 32 && profiles[1].row_end == 64 &&
                  profiles[2].stripe_id == 2 && profiles[2].row_begin == 64 && profiles[2].row_end == 96 &&
                  profiles[3].stripe_id == 3 && profiles[3].row_begin == 96 && profiles[3].row_end == rows,
              "multistripe profiles are contiguous and ordered");
}

bool test_block_activation_scale_compensation_parity() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> full_output(6, 0.0f);
    std::vector<float> stripe_output(6, 0.0f);
    auto full_args = make_args(activation, weights, full_output);
    auto & full_meta = full_args.act_quant.storage().emplace<quants::act::block::Meta>();
    full_meta.scales = { 0.5f, 1.25f, 2.0f };
    full_meta.outliers = { { 0, 0, 4 }, { 1, 1, 3 }, { 2, 0, -2 } };
    auto stripe_args = full_args;
    stripe_args.f_out = stripe_output.data();

    const auto full_result = MatMul(full_args).run_full();
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;
    options.stripe_rows = 1;
    const auto stripe_result = matmul(stripe_args, options);
    return check(full_result.status == MatMulStatus::success, "BLOCK compensation full status") &&
        check(stripe_result.code == MatmulStatusCode::unsupported_route,
              "BLOCK compensation stripe explicitly unsupported");
}

bool test_native_and_channel_full_facade_parity() {
    constexpr size_t rows = 2;
    constexpr size_t columns = 2;
    constexpr size_t depth = QK8_0;
    std::vector<elem_t> activation(depth, 3);

    std::array<block_q8_h1, columns> h1_blocks{};
    std::vector<float> h1_legacy(rows * columns, 0.0f);
    std::vector<float> h1_facade(rows * columns, 0.0f);
    auto h1_args = ggml_gemmini_args_t{};
    h1_args.I = rows;
    h1_args.J = columns;
    h1_args.K = depth;
    h1_args.A = activation.data();
    h1_args.sA = depth;
    h1_args.f_out = h1_legacy.data();
    h1_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    h1_args.q8_h1_blocks = h1_blocks.data();
    h1_args.q8_h1_block_count = h1_blocks.size();
    h1_args.q8_h1_rows = columns;
    h1_args.blocks_per_row = 1;
    h1_args.blocks_K = 1;
    h1_args.blocks_J = columns;
    h1_args.block_size_k = QK8_0;
    h1_args.tiled_matmul_type = CPU;
    h1_args.act_quant.storage().emplace<ggml::gemmini::quants::act::tensor::Meta>().scale = 1.0f;
    auto h1_facade_args = h1_args;
    h1_facade_args.f_out = h1_facade.data();
    ggml::gemmini::tiled_matmul_auto_im2p(&h1_args);
    const auto h1_result = ggml::gemmini::MatMul(h1_facade_args).run_full();
    std::vector<float> h1_stripe(rows * columns, 0.0f);
    auto h1_stripe_args = h1_facade_args;
    h1_stripe_args.f_out = h1_stripe.data();
    ggml::gemmini::MatmulOptions stripe_options{};
    stripe_options.mode = ggml::gemmini::MatmulInvocationMode::stripe_sequential;
    stripe_options.stripe_rows = 1;
    const auto h1_stripe_result = ggml::gemmini::matmul(h1_stripe_args, stripe_options);

    std::array<block_q8_hp1, columns> hp1_blocks{};
    std::vector<float> hp1_legacy(rows * columns, 0.0f);
    std::vector<float> hp1_facade(rows * columns, 0.0f);
    auto hp1_args = ggml_gemmini_args_t{};
    hp1_args.I = rows;
    hp1_args.J = columns;
    hp1_args.K = depth;
    hp1_args.A = activation.data();
    hp1_args.sA = depth;
    hp1_args.f_out = hp1_legacy.data();
    hp1_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
    hp1_args.q8_hp1_blocks = hp1_blocks.data();
    hp1_args.q8_hp1_block_count = hp1_blocks.size();
    hp1_args.q8_hp1_blocks_per_row = 1;
    hp1_args.tiled_matmul_type = CPU;
    hp1_args.act_quant.storage().emplace<ggml::gemmini::quants::act::tensor::Meta>().scale = 1.0f;
    auto hp1_facade_args = hp1_args;
    hp1_facade_args.f_out = hp1_facade.data();
    ggml::gemmini::tiled_matmul_auto_im2p(&hp1_args);
    const auto hp1_result = ggml::gemmini::MatMul(hp1_facade_args).run_full();
    std::vector<float> hp1_stripe(rows * columns, 0.0f);
    auto hp1_stripe_args = hp1_facade_args;
    hp1_stripe_args.f_out = hp1_stripe.data();
    const auto hp1_stripe_result = ggml::gemmini::matmul(hp1_stripe_args, stripe_options);

    std::array<block_q8_h2, columns> h2_blocks{};
    for (size_t column = 0; column < columns; ++column) {
        h2_blocks[column].m = static_cast<uint8_t>(17 + column);
        h2_blocks[column].channel_scale = 0.25f;
    }
    std::vector<float> h2_legacy(rows * columns, 0.0f);
    std::vector<float> h2_facade(rows * columns, 0.0f);
    auto h2_args = h1_args;
    h2_args.f_out = h2_legacy.data();
    h2_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h2;
    h2_args.q8_h1_blocks = nullptr;
    h2_args.q8_h1_block_count = 0;
    h2_args.q8_h1_rows = 0;
    h2_args.q8_h2_blocks = h2_blocks.data();
    h2_args.q8_h2_block_count = h2_blocks.size();
    h2_args.q8_h2_blocks_per_row = 1;
    auto h2_facade_args = h2_args;
    h2_facade_args.f_out = h2_facade.data();
    ggml::gemmini::tiled_matmul_auto_im2p(&h2_args);
    const auto h2_result = ggml::gemmini::MatMul(h2_facade_args).run_full();

    std::array<block_q8_hp2, columns> hp2_blocks{};
    for (size_t column = 0; column < columns; ++column) {
        hp2_blocks[column].m = static_cast<int16_t>(column + 1);
        hp2_blocks[column].channel_scale = 0.5f;
    }
    std::vector<float> hp2_legacy(rows * columns, 0.0f);
    std::vector<float> hp2_facade(rows * columns, 0.0f);
    auto hp2_args = h1_args;
    hp2_args.f_out = hp2_legacy.data();
    hp2_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2;
    hp2_args.q8_h1_blocks = nullptr;
    hp2_args.q8_h1_block_count = 0;
    hp2_args.q8_h1_rows = 0;
    hp2_args.q8_hp2_blocks = hp2_blocks.data();
    hp2_args.q8_hp2_block_count = hp2_blocks.size();
    hp2_args.q8_hp2_blocks_per_row = 1;
    auto hp2_facade_args = hp2_args;
    hp2_facade_args.f_out = hp2_facade.data();
    ggml::gemmini::tiled_matmul_auto_im2p(&hp2_args);
    const auto hp2_result = ggml::gemmini::MatMul(hp2_facade_args).run_full();

    constexpr size_t channel_depth = 4;
    const std::vector<elem_t> channel_codes = { 1, -2, 3, -4, 2, 1, -1, 2 };
    const std::array<float, columns> channel_scales = { 0.5f, 0.25f };
    std::vector<uint8_t> direct_rows(columns * (sizeof(float) + channel_depth));
    for (size_t column = 0; column < columns; ++column) {
        uint8_t * row = direct_rows.data() + column * (sizeof(float) + channel_depth);
        std::memcpy(row, &channel_scales[column], sizeof(float));
        std::memcpy(row + sizeof(float), channel_codes.data() + column * channel_depth, channel_depth);
    }
    std::vector<float> direct_legacy(rows * columns, 0.0f);
    std::vector<float> direct_facade(rows * columns, 0.0f);
    auto direct_args = ggml_gemmini_args_t{};
    direct_args.I = rows;
    direct_args.J = columns;
    direct_args.K = channel_depth;
    direct_args.A = activation.data();
    direct_args.sA = channel_depth;
    direct_args.B = reinterpret_cast<elem_t *>(direct_rows.data() + sizeof(float));
    direct_args.sB = sizeof(float) + channel_depth;
    direct_args.f_out = direct_legacy.data();
    direct_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel;
    direct_args.q8_channel_row_base = direct_rows.data();
    direct_args.q8_channel_row_stride = sizeof(float) + channel_depth;
    direct_args.q8_channel_row_count = columns;
    direct_args.act_quant.storage().emplace<ggml::gemmini::quants::act::tensor::Meta>();
    direct_args.tiled_matmul_type = CPU;
    auto direct_facade_args = direct_args;
    direct_facade_args.f_out = direct_facade.data();
    ggml::gemmini::tiled_matmul_auto_baseline(
        &direct_args, ggml::gemmini::baseline_activation_quant_t::TENSOR,
        ggml::gemmini::baseline_weight_quant_t::CHANNEL);
    const auto direct_result = ggml::gemmini::MatMul(direct_facade_args).run_full();
    std::vector<float> direct_stripe(rows * columns, 0.0f);
    auto direct_stripe_args = direct_facade_args;
    direct_stripe_args.f_out = direct_stripe.data();
    const auto direct_stripe_result = ggml::gemmini::matmul(direct_stripe_args, stripe_options);

    std::vector<float> sidecar_legacy(rows * columns, 0.0f);
    std::vector<float> sidecar_facade(rows * columns, 0.0f);
    auto sidecar_args = direct_args;
    sidecar_args.B = const_cast<elem_t *>(channel_codes.data());
    sidecar_args.sB = channel_depth;
    sidecar_args.f_out = sidecar_legacy.data();
    sidecar_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
    sidecar_args.q8_channel_row_base = nullptr;
    sidecar_args.q8_channel_row_stride = 0;
    sidecar_args.q8_channel_row_count = 0;
    sidecar_args.weight_channel_scales = channel_scales.data();
    sidecar_args.weight_channel_scale_count = columns;
    auto sidecar_facade_args = sidecar_args;
    sidecar_facade_args.f_out = sidecar_facade.data();
    ggml::gemmini::tiled_matmul_auto_baseline(
        &sidecar_args, ggml::gemmini::baseline_activation_quant_t::TENSOR,
        ggml::gemmini::baseline_weight_quant_t::CHANNEL);
    const auto sidecar_result = ggml::gemmini::MatMul(sidecar_facade_args).run_full();
    std::vector<float> sidecar_stripe(rows * columns, 0.0f);
    auto sidecar_stripe_args = sidecar_facade_args;
    sidecar_stripe_args.f_out = sidecar_stripe.data();
    const auto sidecar_stripe_result = ggml::gemmini::matmul(sidecar_stripe_args, stripe_options);

    return check(h1_result.status == ggml::gemmini::MatMulStatus::success &&
                     same_output(h1_facade, h1_legacy) && h1_stripe_result.ok() &&
                     same_output(h1_stripe, h1_legacy),
                 "Q8_H1 facade parity") &&
        check(hp1_result.status == ggml::gemmini::MatMulStatus::success &&
                     same_output(hp1_facade, hp1_legacy) && hp1_stripe_result.ok() &&
                     same_output(hp1_stripe, hp1_legacy),
              "Q8_HP1 facade parity") &&
        check(h2_result.status == ggml::gemmini::MatMulStatus::success &&
                     same_output(h2_facade, h2_legacy) &&
                     ggml::gemmini::MatMul::stripe_capability(h2_facade_args) ==
                         ggml::gemmini::MatMulCapability::unsupported,
              "Q8_H2 full-only facade parity") &&
        check(hp2_result.status == ggml::gemmini::MatMulStatus::success &&
                     same_output(hp2_facade, hp2_legacy) &&
                     ggml::gemmini::MatMul::stripe_capability(hp2_facade_args) ==
                         ggml::gemmini::MatMulCapability::unsupported,
              "Q8_HP2 full-only facade parity") &&
        check(direct_result.status == ggml::gemmini::MatMulStatus::success &&
                     same_output(direct_facade, direct_legacy) && direct_stripe_result.ok() &&
                     same_output(direct_stripe, direct_legacy),
              "Q8_CHANNEL direct facade parity") &&
        check(sidecar_result.status == ggml::gemmini::MatMulStatus::success &&
                     same_output(sidecar_facade, sidecar_legacy) && sidecar_stripe_result.ok() &&
                     same_output(sidecar_stripe, sidecar_legacy),
              "Q8_CHANNEL sidecar facade parity");
}

bool test_full_and_stripe_sequential_outputs_match() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> full_output(6, 0.0f);
    std::vector<float> stripe_output(6, 0.0f);
    auto full_args = make_args(activation, weights, full_output);
    auto stripe_args = make_args(activation, weights, stripe_output);
    MatmulOptions full_options{};
    full_options.mode = MatmulInvocationMode::full;
    MatmulOptions stripe_options{};
    stripe_options.mode = MatmulInvocationMode::stripe_sequential;
    stripe_options.stripe_rows = 1;
    stripe_options.rc_shards = 2;
    return check(matmul(full_args, full_options).ok(), "full public matmul") &&
        check(matmul(stripe_args, stripe_options).ok(), "stripe sequential public matmul") &&
        check(same_output(full_output, stripe_output), "full and stripe sequential output differs");
}

bool test_native_exsia_theta_row_slice_parity() {
    using namespace ggml::gemmini;
    constexpr size_t rows = 32;
    constexpr size_t columns = 1;
    constexpr size_t depth = QK8_0;
    std::vector<elem_t> activation(rows * depth, 1);
    block_q8_h1 block{};
    std::fill(std::begin(block.qs), std::end(block.qs), elem_t{1});
    block.c_b = 86;
    block.R = 4095;
    block.s_rf = 0.00032747327350080013f;

    std::vector<float> full_output(rows * columns, 0.0f);
    std::vector<float> stripe_output(rows * columns, 0.0f);
    auto full_args = ggml_gemmini_args_t{};
    full_args.I = rows;
    full_args.J = columns;
    full_args.K = depth;
    full_args.A = activation.data();
    full_args.sA = depth;
    full_args.f_out = full_output.data();
    full_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    full_args.q8_h1_blocks = &block;
    full_args.q8_h1_block_count = 1;
    full_args.q8_h1_rows = 1;
    full_args.blocks_per_row = 1;
    full_args.blocks_K = 1;
    full_args.blocks_J = columns;
    full_args.block_size_k = depth;
    full_args.tiled_matmul_type = CPU;
    full_args.act_quant.storage().emplace<ggml::gemmini::quants::act::exsia::Meta>().theta =
        { 0, 1, 2, 3 };

    auto stripe_args = full_args;
    stripe_args.f_out = stripe_output.data();
    const auto full_result = MatMul(full_args).run_full();
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;
    options.stripe_rows = 1;
    const auto stripe_result = matmul(stripe_args, options);
    return check(full_result.status == MatMulStatus::success, "native EXSIA full status") &&
        check(stripe_result.ok(), "native EXSIA stripe status") &&
        check(same_output(stripe_output, full_output), "native EXSIA theta slice differs");
}

bool test_native_exsia_multistripe_residual_parity() {
    using namespace ggml::gemmini;
    constexpr size_t rows = 32;
    constexpr size_t columns = 4;
    constexpr size_t depth = QK8_0;
    std::vector<elem_t> activation(rows * depth, 1);
    std::vector<block_q8_h1> blocks(columns);
    for (auto & block : blocks) {
        std::fill(std::begin(block.qs), std::end(block.qs), elem_t{1});
        block.c_b = 86;
        block.R = 4095;
        block.s_rf = 0.00032747327350080013f;
    }
    std::vector<float> full_output(rows * columns, 0.0f);
    std::vector<float> dense_output(rows * columns, 0.0f);
    std::vector<float> stripe_output(rows * columns, 0.0f);
    const quants::QactOutlier outlier{17, 3, 2};

    auto make_args = [&](std::vector<float> & output) {
        ggml_gemmini_args_t args{};
        args.I = rows;
        args.J = columns;
        args.K = depth;
        args.A = activation.data();
        args.sA = depth;
        args.f_out = output.data();
        args.stride_f_out = columns;
        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
        args.q8_h1_blocks = blocks.data();
        args.q8_h1_block_count = blocks.size();
        args.q8_h1_rows = columns;
        args.blocks_per_row = 1;
        args.blocks_K = 1;
        args.blocks_J = columns;
        args.block_size_k = depth;
        args.tiled_matmul_type = CPU;
        auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
        meta.theta = { 0, 1 };
        meta.outliers = { outlier };
        return args;
    };

    auto full_args = make_args(full_output);
    auto dense_args = make_args(dense_output);
    auto stripe_args = make_args(stripe_output);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;
    options.stripe_rows = 16;
    options.rc_shards = 2;
    const auto full_result = MatMul(full_args).run_full();
    const auto dense_result = MatMul(dense_args).run_dense();
    const auto stripe_result = matmul(stripe_args, options);
    return check(full_result.status == MatMulStatus::success, "native multistripe full status") &&
        check(dense_result.status == MatMulStatus::success, "native multistripe dense status") &&
        check(stripe_result.ok(), "native multistripe stripe status") &&
        check(same_output(stripe_output, full_output), "native multistripe residual differs");
}

bool test_empty_tail_and_malformed_stripe_status() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    args.act_quant.storage().emplace<ggml::gemmini::quants::act::tensor::Meta>().scale = 1.0f;
    ggml::gemmini::MatMul facade(args);

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
    auto args = make_args(activation, weights, output);
    args.act_quant.storage().emplace<ggml::gemmini::quants::act::tensor::Meta>().scale = 1.0f;
    ggml::gemmini::MatMul facade(args);

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
                 "HP2 stripe capability") &&
        check([&] {
            auto transpose_args = make_args(activation, weights, output);
            transpose_args.transpose_A = true;
            return ggml::gemmini::MatMul::stripe_capability(transpose_args) ==
                ggml::gemmini::MatMulCapability::unsupported;
        }(), "transpose-A stripe capability") &&
        check([&] {
            auto bias_args = make_args(activation, weights, output);
            std::vector<int32_t> bias(bias_args.J, 1);
            bias_args.D = bias.data();
            bias_args.repeating_bias = false;
            return ggml::gemmini::MatMul::stripe_capability(bias_args) ==
                ggml::gemmini::MatMulCapability::unsupported;
        }(), "non-repeating bias stripe capability");
}

bool test_stripe_state_lifecycle() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    args.act_quant.storage().emplace<ggml::gemmini::quants::act::tensor::Meta>().scale = 1.0f;
    ggml::gemmini::MatMul facade(args);

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
    if (!check(prepare_compensation(job).ok(), "prepare compensation") ||
        !check(execute_dense_stripe(job).ok(), "dense stripe")) {
        return false;
    }
    const size_t shards = job.snapshot().expected_shards;
    for (size_t shard = 0; shard < shards; ++shard) {
        if (!check(execute_compensation_shard(job, shard, shards).ok(), "compensation shard")) {
            return false;
        }
    }
    return check(finalize_stripe(job).ok(), "finalize stripe");
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
    options.job_capacity = 2;

    const int32_t residual[] = { 4, 5 };
    MatmulStripeInput input(1, 3, 7, residual, 2);
    const quants::QactOutlier outlier[] = {{ 1, 2, 3 }};
    MatmulStripeInput outlier_input(1, 3, 7, outlier, 1);
    MatmulJobMetrics metrics{};
    std::vector<elem_t> validation_activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> validation_weights = { 1, -1, 2, 3 };
    std::vector<float> validation_output(6, 0.0f);
    MatmulOptions validation_options{};
    validation_options.mode = MatmulInvocationMode::full;
    validation_options.validation = true;
    auto validation_args = make_args(validation_activation, validation_weights, validation_output);
    validation_args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    const auto validation_status = matmul(validation_args, validation_options);
    MatmulOptions too_many_dense_threads{};
    too_many_dense_threads.dense_threads = 2;
    auto dense_thread_args = make_args(validation_activation, validation_weights, validation_output);
    dense_thread_args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    auto dense_thread_execution = prepare_execution(dense_thread_args, too_many_dense_threads);
    MatmulOptions staged_options{};
    staged_options.mode = MatmulInvocationMode::stripe_sequential;
    MatmulExecution staged_execution;
    auto staged_args = make_args(validation_activation, validation_weights, validation_output);
    staged_args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    const auto staged_prepare = prepare_execution(staged_args, staged_options, staged_execution);
    MatmulStripeJob staged_job;
    const auto staged_capture = capture_stripe(staged_execution, MatmulStripeInput(0, 1), staged_job);
    const bool staged_contract = config::ENABLE_STRIPE_MATMUL
        ? check(staged_prepare.ok() && staged_execution.state() == MatmulExecutionState::running,
                "output-parameter prepare execution") &&
              check(staged_capture.ok() && staged_job.status().ok(),
                    "output-parameter capture stripe")
        : check(staged_prepare.code == MatmulStatusCode::unsupported_invocation,
                "disabled stripe prepare rejected as unsupported invocation");
    return check(resolve_matmul_options(defaults).options.job_capacity == 2,
                 "configured default job capacity") &&
        check(statuses[0].ok(), "success status ok") &&
        check(!statuses[1].ok(), "failure status not ok") &&
        check(std::string_view(statuses[2].message) == "invalid contract", "status message") &&
        check(options.mode == MatmulInvocationMode::stripe_pipeline && options.dense_threads == 2 &&
                  options.rc_shards == 3 && options.validation && options.profiling &&
                  options.job_capacity == 2,
              "matmul options") &&
        check(validation_status.ok(), "validation-enabled full matmul") &&
        check(dense_thread_execution.status().code == MatmulStatusCode::unsupported_invocation,
              "multi-owner dense lane rejection") &&
        staged_contract &&
        check(input.row_begin() == 1 && input.row_end() == 3 && input.stripe_id() == 7 &&
                  input.residual() == residual && input.residual_count() == 2,
              "stripe input metadata") &&
        check(outlier_input.outliers() == outlier && outlier_input.outlier_count() == 1 &&
                  outlier_input.residual() == nullptr,
              "outlier stripe input metadata") &&
        check(metrics.la.count == 0 && metrics.sf.count == 0 && metrics.handoff.count == 0 &&
                  metrics.ws.count == 0 && metrics.rc_prepare.count == 0 &&
                  metrics.rc_compute.count == 0 && metrics.rc_finalize.count == 0,
              "job metric storage");
}

bool test_matmul_option_resolution_precedence() {
    using namespace ggml::gemmini;
    ScopedEnvVar mode_env("GEMMINI_MATMUL_MODE");
    ScopedEnvVar rows_env("GEMMINI_STRIPE_ROWS");
    ScopedEnvVar shards_env("GEMMINI_RC_SHARDS");
    ScopedEnvVar capacity_env("GEMMINI_STRIPE_JOB_CAPACITY");

    const auto defaults = resolve_matmul_options({});
    if (!check(defaults.ok(), "default matmul options resolve") ||
        !check(defaults.options.mode == static_cast<MatmulInvocationMode>(config::DEFAULT_MATMUL_MODE),
               "configured default matmul mode") ||
        !check(defaults.options.stripe_rows_auto == !config::DEFAULT_STRIPE_ROWS.has_value(),
               "configured default stripe-row mode") ||
        !check(defaults.options.stripe_rows == config::DEFAULT_STRIPE_ROWS.value_or(1),
               "configured default stripe rows") ||
        !check(defaults.options.rc_shards == config::DEFAULT_RC_SHARDS,
               "configured default rc shards") ||
        !check(defaults.options.job_capacity == config::DEFAULT_STRIPE_JOB_CAPACITY,
               "configured default job capacity")) {
        return false;
    }

    constexpr auto environment_mode = config::ENABLE_STRIPE_PIPELINE
        ? MatmulInvocationMode::stripe_pipeline
        : config::ENABLE_STRIPE_MATMUL ? MatmulInvocationMode::stripe_sequential
                                       : MatmulInvocationMode::full;
    mode_env.set(config::ENABLE_STRIPE_PIPELINE
                     ? "STRIPE_PIPELINE"
                     : config::ENABLE_STRIPE_MATMUL ? "STRIPE_SEQUENTIAL" : "FULL");
    rows_env.set("5");
    shards_env.set("7");
    capacity_env.set("9");

    const auto env_resolution = resolve_matmul_options({});
    if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE) {
        if (!check(env_resolution.ok(), "environment overrides resolve") ||
            !check(env_resolution.options.mode == environment_mode,
                   "environment matmul mode precedence") ||
            !check(!env_resolution.options.stripe_rows_auto && env_resolution.options.stripe_rows == 5,
                   "environment stripe rows precedence") ||
            !check(env_resolution.options.rc_shards == 7, "environment rc shards precedence") ||
            !check(env_resolution.options.job_capacity == 9, "environment job capacity precedence")) {
            return false;
        }
    } else if (!check(env_resolution.ok(), "runtime-override-off ignores environment")) {
        return false;
    }

    MatmulOptionOverrides explicit_options{};
    explicit_options.mode = config::ENABLE_STRIPE_MATMUL
        ? MatmulInvocationMode::stripe_sequential : MatmulInvocationMode::full;
    explicit_options.stripe_rows = 3;
    explicit_options.rc_shards = 4;
    explicit_options.job_capacity = 2;
    explicit_options.validation = true;
    explicit_options.profiling = true;
    const auto explicit_resolution = resolve_matmul_options(explicit_options);
    if (!check(explicit_resolution.ok(), "explicit overrides resolve") ||
        !check(explicit_resolution.options.mode == *explicit_options.mode,
               "explicit mode precedence") ||
        !check(!explicit_resolution.options.stripe_rows_auto &&
                   explicit_resolution.options.stripe_rows == 3,
               "explicit stripe-row precedence") ||
        !check(explicit_resolution.options.rc_shards == 4, "explicit rc shard precedence") ||
        !check(explicit_resolution.options.job_capacity == 2, "explicit job capacity precedence") ||
        !check(explicit_resolution.options.validation && explicit_resolution.options.profiling,
               "explicit flags preserved")) {
        return false;
    }

    mode_env.set("NOT_A_VALID_MODE");
    const auto invalid_env = resolve_matmul_options({});
    if (config::ALLOW_RUNTIME_MATMUL_OVERRIDE) {
    return check(invalid_env.error == MatmulOptionsError::invalid_mode,
                     "invalid environment mode rejected");
    }
    return check(invalid_env.ok(), "disabled runtime override ignores invalid environment");
}

bool test_default_matmul_mode_executes_configured_backend_path() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> reference_output(6, 0.0f);
    std::vector<float> default_output(6, 0.0f);
    const auto expected_mode = static_cast<MatmulInvocationMode>(config::DEFAULT_MATMUL_MODE);

    auto reference_args = make_args(activation, weights, reference_output);
    if (expected_mode == MatmulInvocationMode::stripe_pipeline) {
        reference_args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0, 0 };
    } else {
        reference_args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    }
    const auto reference_result = MatMul(reference_args).run_full();
    if (!check(reference_result.status == MatMulStatus::success, "reference full execution")) {
        return false;
    }

    auto default_args = make_args(activation, weights, default_output);
    if (expected_mode == MatmulInvocationMode::stripe_pipeline) {
        default_args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0, 0 };
    } else {
        default_args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    }
    MatmulOptions default_options{};
    auto execution = prepare_execution(default_args, default_options);
    if (!check(execution.status().ok(), "default-mode execution prepared") ||
        !check(execution.mode() == expected_mode, "default-mode resolution")) {
        return false;
    }

    switch (execution.mode()) {
    case MatmulInvocationMode::full: {
        const auto status = execute_full(execution);
        return check(status.ok(), "default full execution") &&
            check(same_output(default_output, reference_output), "default full output parity") &&
            check(finish_execution(execution).ok(), "default full finish");
    }
    case MatmulInvocationMode::stripe_sequential: {
        auto job = capture_stripe(execution, { 0, default_args.I, 0 });
        return check(job.status().ok(), "default sequential capture") &&
            run_staged_job(job) &&
            check(finish_execution(execution).ok(), "default sequential finish") &&
            check(same_output(default_output, reference_output), "default sequential output parity");
    }
    case MatmulInvocationMode::stripe_pipeline: {
        MatmulStripeCollector collector(2);
        if (!check(collector.start(execution), "default pipeline collector start")) {
            return false;
        }
        const auto * sink = collector.sink();
        const bool captured = sink->on_ready(
            sink->user_data, make_ready_event(0, 0, default_args.I, nullptr, 0, 10, 20, 30, 50));
        const auto collector_status = collector.finish();
        const auto execution_status = finish_execution(execution);
        return check(captured, "default pipeline capture") &&
            check(collector_status.ok(), "default pipeline collector finish") &&
            check(execution_status.ok(), "default pipeline execution finish") &&
            check(same_output(default_output, reference_output), "default pipeline output parity");
    }
    default:
        return check(false, "unexpected configured matmul mode");
    }
}

bool test_pipeline_stripe_summary_contract() {
    using namespace ggml::gemmini;

    MatmulJobMetrics profile{};
    profile.run_id = 7;
    profile.stripe_id = 2;
    profile.slot = 1;
    profile.row_begin = 8;
    profile.row_end = 12;
    profile.rc_shards = 4;
    profile.la_worker_start_ns = { 100, 110, 120 };
    profile.la_worker_end_ns = { 130, 140, 150 };
    profile.sf_mask_start_ns = 150;
    profile.sf_mask_end_ns = 160;
    profile.sf_exponent_start_ns = 160;
    profile.sf_exponent_end_ns = 170;
    profile.sf_folding_start_ns = 170;
    profile.sf_folding_end_ns = 190;
    profile.sf_commit_ns = 195;
    profile.producer_wait_start_ns = 90;
    profile.producer_wait_end_ns = 95;
    profile.capture_queue_enqueue_ns = 200;
    profile.ws_start_ns = 210;
    profile.ws_end_ns = 260;
    profile.rc_enqueue_ns = 205;
    profile.rc_prepare_start_ns = 250;
    profile.rc_prepare_end_ns = 270;
    profile.rc_shard_start_ns = { 270, 275, 280, 285 };
    profile.rc_shard_end_ns = { 290, 295, 300, 305 };
    profile.merge_start_ns = 310;
    profile.merge_end_ns = 320;
    profile.rc_prepare.nanoseconds = 20;
    profile.rc_compute.nanoseconds = 80;
    profile.rc_finalize.nanoseconds = 10;
    profile.h1_histogram_available = true;
    profile.h1_histogram.residual_nnz = 9;
    profile.h1_histogram.residual_density = 0.125;
    profile.h1_histogram.active_row_groups = 4;
    profile.h1_histogram.active_k = 3;
    profile.h1_histogram.bin_1 = 1;
    profile.h1_histogram.bin_2_to_4 = 2;
    profile.h1_histogram.bin_5_to_8 = 1;
    profile.h1_histogram.bin_over_8 = 0;
    profile.h1_histogram.rows_per_active_k_mean = 1.5;
    profile.h1_histogram.rows_per_active_k_max = 2;
    profile.h1_histogram.estimated_int_mac_count = 288;
    profile.h1_histogram.ycom_write_count = 96;
    profile.h1_histogram.weight_scalar_load_count = 144;
    profile.h1_histogram.weight_vector_load_count = 48;
    profile.h1_histogram.selected_route =
        quants::dec::PreparedDecSelectedRoute::h1_small_group_2_to_4;

    const std::string cpu_summary = detail::pipeline_stripe_summary_json(
        "attn_q", 16, 32, 64, "cpu", "matmul-then-dec", profile);
    const std::string overlap_summary = detail::pipeline_stripe_summary_json(
        "attn_q", 16, 32, 64, "gemmini_ws", "matmul-dec-overlap", profile);

    std::string record_type;
    std::string schedule;
    std::string backend_route;
    uint64_t run_id = 0;
    uint64_t stripe_idx = 0;
    uint64_t stripe_rows = 0;
    uint64_t la_workers = 0;
    uint64_t sf_workers = 0;
    uint64_t dec_workers = 0;
    uint64_t ordering_violation = 0;
    uint64_t overlap_ordering_violation = 1;
    uint64_t t_la_ns = 0;
    uint64_t t_sf_ns = 0;
    uint64_t t_matmul_ns = 0;
    uint64_t t_merge_ns = 0;
    uint64_t t_dec_kernel_ns = 0;
    uint64_t t_dec_premerge_ns = 0;
    uint64_t la_service_sum_ns = 0;
    uint64_t dec_kernel_service_sum_ns = 0;
    uint64_t dec_service_sum_ns = 0;
    uint64_t residual_nnz = 0;
    uint64_t active_row_groups = 0;
    uint64_t weight_vector_load_count = 0;
    double la_efficiency = 0.0;
    double dec_kernel_efficiency = 0.0;
    double dec_service_efficiency = 0.0;
    std::string selected_route;

    const bool passed =
        check(extract_string_field(cpu_summary, "record_type", record_type) &&
                     record_type == "PIPELINE_STRIPE_SUMMARY",
                 "summary record type") &&
        check(extract_string_field(cpu_summary, "schedule", schedule) &&
                  schedule == "matmul-then-dec",
              "summary CPU schedule label") &&
        check(extract_string_field(cpu_summary, "backend_route", backend_route) &&
                  backend_route == "cpu",
              "summary backend route label") &&
        check(extract_u64_field(cpu_summary, "run_id", run_id) && run_id == 7,
              "summary run_id") &&
        check(extract_u64_field(cpu_summary, "stripe_idx", stripe_idx) && stripe_idx == 2,
              "summary stripe identity") &&
        check(extract_u64_field(cpu_summary, "stripe_rows", stripe_rows) && stripe_rows == 4,
              "summary stripe rows") &&
        check(extract_u64_field(cpu_summary, "la_workers", la_workers) && la_workers == 3,
              "summary LA worker count") &&
        check(extract_u64_field(cpu_summary, "sf_workers", sf_workers) && sf_workers == 1,
              "summary SF worker count") &&
        check(extract_u64_field(cpu_summary, "dec_workers", dec_workers) && dec_workers == 4,
              "summary DEC worker count") &&
        check(extract_u64_field(cpu_summary, "ordering_violation", ordering_violation) &&
                  ordering_violation == 1,
              "summary CPU ordering violation") &&
        check(extract_u64_field(overlap_summary, "ordering_violation", overlap_ordering_violation) &&
                  overlap_ordering_violation == 0,
              "summary overlap ordering violation") &&
        check(has_array_field(cpu_summary, "la_worker_body_start_ns") &&
                  has_array_field(cpu_summary, "la_worker_body_end_ns") &&
                  has_array_field(cpu_summary, "dec_shard_start_ns") &&
                  has_array_field(cpu_summary, "dec_shard_end_ns"),
              "summary array schema") &&
        check(has_object_field(cpu_summary, "h1_histogram") &&
                  !has_null_field(cpu_summary, "h1_histogram"),
              "summary H1 histogram object") &&
        check(extract_u64_field(cpu_summary, "T_LA_ns", t_la_ns) && t_la_ns == 50 &&
                  extract_u64_field(cpu_summary, "T_SF_ns", t_sf_ns) && t_sf_ns == 45 &&
                  extract_u64_field(cpu_summary, "T_MatMul_ns", t_matmul_ns) &&
                  t_matmul_ns == 50 &&
                  extract_u64_field(cpu_summary, "T_Merge_ns", t_merge_ns) && t_merge_ns == 10 &&
                  extract_u64_field(cpu_summary, "T_DEC_kernel_ns", t_dec_kernel_ns) &&
                  t_dec_kernel_ns == 35 &&
                  extract_u64_field(cpu_summary, "T_DEC_premerge_ns", t_dec_premerge_ns) &&
                  t_dec_premerge_ns == 55,
              "summary timing formulas") &&
        check([&] {
                  MatmulJobMetrics unavailable = profile;
                  unavailable.sf_mask_start_ns = 0;
                  unavailable.sf_commit_ns = 0;
                  unavailable.merge_start_ns = 0;
                  unavailable.merge_end_ns = 0;
                  const std::string summary = detail::pipeline_stripe_summary_json(
                      "attn_q", 16, 32, 64, "cpu", "matmul-then-dec", unavailable);
                  uint64_t sf_ns = 1;
                  uint64_t merge_ns = 1;
                  return extract_u64_field(summary, "T_SF_ns", sf_ns) && sf_ns == 0 &&
                      extract_u64_field(summary, "T_Merge_ns", merge_ns) && merge_ns == 0;
              }(),
              "summary unavailable timing defaults") &&
        check(extract_u64_field(cpu_summary, "la_service_sum_ns", la_service_sum_ns) &&
                  la_service_sum_ns == 90 &&
                  extract_u64_field(cpu_summary, "dec_kernel_service_sum_ns", dec_kernel_service_sum_ns) &&
                  dec_kernel_service_sum_ns == 80 &&
                  extract_u64_field(cpu_summary, "dec_service_sum_ns", dec_service_sum_ns) &&
                  dec_service_sum_ns == 80,
              "summary service sums") &&
        check(extract_double_field(cpu_summary, "la_efficiency", la_efficiency) &&
                  std::abs(la_efficiency - 0.6) < 1e-9 &&
                  extract_double_field(cpu_summary, "dec_kernel_efficiency", dec_kernel_efficiency) &&
                  std::abs(dec_kernel_efficiency - (80.0 / 140.0)) < 1e-9 &&
                  extract_double_field(cpu_summary, "dec_service_efficiency", dec_service_efficiency) &&
                  std::abs(dec_service_efficiency - (80.0 / 140.0)) < 1e-9,
              "summary efficiency formulas") &&
        check(cpu_summary.find("\"h1_histogram\":{\"available\":true") != std::string::npos &&
                  extract_string_field(cpu_summary, "selected_route", selected_route) &&
                  selected_route == "h1_small_group_2_to_4" &&
                  extract_u64_field(cpu_summary, "residual_nnz", residual_nnz) &&
                  residual_nnz == 9 &&
                  extract_u64_field(cpu_summary, "active_row_groups", active_row_groups) &&
                  active_row_groups == 4 &&
                  extract_u64_field(cpu_summary, "weight_vector_load_count", weight_vector_load_count) &&
                  weight_vector_load_count == 48,
              "summary histogram payload");
    if (passed) {
        std::puts("PASS contract: pipeline stripe summary schema/order");
    }
    return passed;
}

bool test_disabled_stripe_modes_are_rejected() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    MatmulOptions options{};
    bool passed = true;
    if (!ggml::gemmini::config::ENABLE_STRIPE_MATMUL) {
        options.mode = MatmulInvocationMode::stripe_sequential;
        passed = check(prepare_execution(make_args(activation, weights, output), options).status().code ==
                           MatmulStatusCode::unsupported_invocation,
                       "disabled sequential stripe rejected as unsupported invocation");
    }
    if (!config::ENABLE_STRIPE_PIPELINE) {
        options.mode = MatmulInvocationMode::stripe_pipeline;
        passed = check(prepare_execution(make_args(activation, weights, output), options).status().code ==
                           MatmulStatusCode::unsupported_invocation,
                       "disabled stripe pipeline rejected as unsupported invocation") && passed;
    }
    return passed;
}

bool test_dispatch_override_contract() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> row_output(6, 0.0f);
    std::vector<float> group_output(6, 0.0f);

    MatmulOptions row_options{};
    row_options.mode = MatmulInvocationMode::full;
    row_options.force_row_direct = true;
    auto row_args = make_args(activation, weights, row_output);
    row_args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    auto row_execution = prepare_execution(row_args, row_options);
    const MatmulStatus row_status = execute_full(row_execution);

    MatmulOptions group_options{};
    group_options.mode = MatmulInvocationMode::full;
    group_options.force_group_k_csc = true;
    auto group_args = make_args(activation, weights, group_output);
    group_args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    auto group_execution = prepare_execution(group_args, group_options);
    const MatmulStatus group_status = execute_full(group_execution);

    MatmulOptions conflicting_options{};
    conflicting_options.mode = MatmulInvocationMode::full;
    conflicting_options.force_row_direct = true;
    conflicting_options.force_group_k_csc = true;
    auto conflicting_args = make_args(activation, weights, row_output);
    conflicting_args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    auto conflicting_execution = prepare_execution(conflicting_args, conflicting_options);

    return check(row_status.ok(), "row-direct override status") &&
        check(group_status.ok(), "group-KCSC override status") &&
        check(same_output(row_output, group_output), "dispatch overrides changed output") &&
        check(conflicting_execution.status().code == MatmulStatusCode::invalid_argument,
              "conflicting dispatch overrides rejected");
}

bool test_route_capability_table() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    args.act_quant.storage().emplace<quants::act::NoneMeta>();
    if (!check(std::holds_alternative<quants::act::NoneMeta>(args.act_quant.storage()),
               "NoneMeta storage") ||
        !check(args.act_quant.kind() == quants::act::MetaKind::none, "NoneMeta activation kind")) {
        return false;
    }
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
               "Q8_CHANNEL direct normalization") ||
        !check(!detail::route_capabilities(args).full,
               "Q8_CHANNEL without activation metadata rejected")) {
        return false;
    }
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
    if (!check(detail::normalize_route(args).weight == detail::WeightRoute::q8_channel_sidecar,
               "Q8_CHANNEL sidecar normalization")) {
        return false;
    }
    args.act_quant.storage().emplace<quants::act::stripe::Meta>();
    if (!check(args.act_quant.kind() == quants::act::MetaKind::stripe, "stripe activation kind") ||
        !check(detail::normalize_route(args).activation == detail::ActivationRoute::stripe,
               "STRIPE activation normalization")) {
        return false;
    }
    args.act_quant.storage().emplace<quants::act::tensor::Meta>();
    args.tiled_matmul_type = WS;
    if (!check(detail::normalize_route(args).backend == detail::BackendRoute::gemmini_ws &&
                   detail::route_capabilities(args).full,
               "Gemmini WS backend capability")) {
        return false;
    }
    args.tiled_matmul_type = OS;
    if (!check(detail::normalize_route(args).backend == detail::BackendRoute::gemmini_os &&
                   !detail::route_capabilities(args).full,
               "Gemmini OS explicit unsupported capability")) {
        return false;
    }
    MatmulOptions full_options{};
    full_options.mode = MatmulInvocationMode::full;
    const auto os_status = matmul(args, full_options);
    return check(os_status.code == MatmulStatusCode::unsupported_backend,
                 "Gemmini OS explicit unsupported status");
}

bool test_explicit_exsia_channel_rejection() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    args.act_quant.storage().emplace<ggml::gemmini::quants::act::exsia::Meta>();
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel;
    const auto key = ggml::gemmini::detail::normalize_route(args);
    const auto caps = ggml::gemmini::detail::route_capabilities(args);
    if (!check(key.activation == ggml::gemmini::detail::ActivationRoute::exsia &&
                   key.weight == ggml::gemmini::detail::WeightRoute::q8_channel_direct,
               "EXSIA Q8_CHANNEL route normalization") ||
        !check(caps.legacy_full && !caps.full && !caps.sliced_compensation,
               "EXSIA Q8_CHANNEL explicit unsupported capability")) {
        return false;
    }
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
    const auto sidecar = ggml::gemmini::detail::route_capabilities(args);
    return check(ggml::gemmini::detail::normalize_route(args).weight ==
                     ggml::gemmini::detail::WeightRoute::q8_channel_sidecar,
                 "EXSIA Q8_CHANNEL sidecar route normalization") &&
        check(sidecar.legacy_full && !sidecar.full && !sidecar.sliced_compensation,
              "EXSIA Q8_CHANNEL sidecar explicit unsupported capability");
}

bool test_global_activation_metadata_view_boundaries() {
    using namespace ggml::gemmini::quants::act;
    std::vector<elem_t> activation(130, 1);
    std::vector<elem_t> weights = { 1 };
    std::vector<float> output(130, 0.0f);
    auto args = make_args(activation, weights, output);
    args.I = 130;
    args.J = 1;
    args.K = 1;
    args.sA = 1;
    args.sB = 1;
    args.stride_f_out = 1;
    args.tile_I = 4;

    auto & meta = args.act_quant.storage().emplace<exsia::Meta>();
    meta.theta = { -1, 0, 1 };
    const auto * theta_data = meta.theta.data();
    const ActivationMetadataView boundary(args, 63, 66);
    const ActivationMetadataView tail(args, 128, 130);
    size_t global = 0;
    int16_t theta = 0;
    float scale = 0.0f;

    return check(boundary.valid() && boundary.row_count() == 3 && boundary.stripe_count() == 2,
                 "63/64/65 metadata view bounds") &&
        check(boundary.global_row(0, global) && global == 63 &&
                  boundary.global_row(1, global) && global == 64 &&
                  boundary.global_row(2, global) && global == 65,
              "local rows map to global 63/64/65") &&
        check(boundary.global_stripe(0, global) && global == 0 &&
                  boundary.global_stripe(1, global) && global == 1,
              "local stripes preserve global boundary") &&
        check(boundary.scale(0, scale) && scale == 0.5f &&
                  boundary.scale(1, scale) && scale == 1.0f &&
                  boundary.scale(2, scale) && scale == 1.0f,
              "row scales use global stripe metadata") &&
        check(boundary.theta(0, theta) && theta == -1 &&
                  boundary.theta(1, theta) && theta == 0,
              "stripe theta uses global metadata") &&
        check(tail.valid() && tail.global_stripe(0, global) && global == 2 &&
                  tail.theta(0, theta) && theta == 1,
              "tail metadata maps to final global stripe") &&
        check(meta.theta.data() == theta_data && meta.theta.size() == 3,
              "metadata storage remains unsliced");
}

bool test_invalid_activation_scale_is_explicit_contract_error() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    args.transpose_B = true;
    args.sB = args.K;
    args.act_quant.storage().emplace<quants::act::token::Meta>().scales = { 1.0f, 0.0f, 1.0f };
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;
    options.stripe_rows = 1;

    const auto scales = quants::act::activation_scales(args, args.I);
    if (!check(scales.empty(), "invalid activation metadata has no scale fallback")) {
        return false;
    }
    return check(matmul(args, options).code == MatmulStatusCode::invalid_contract,
                 "invalid activation metadata propagates invalid_contract");
}

bool test_missing_activation_metadata_allows_dense_routes_and_fp32() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto quantized_args = make_args(activation, weights, output);
    quantized_args.act_quant.storage().emplace<quants::act::NoneMeta>();
    quants::act::ActivationMetadataView metadata(quantized_args, 0, quantized_args.I);
    float scale = 0.0f;
    MatmulOptions stripe_options{};
    stripe_options.mode = MatmulInvocationMode::stripe_sequential;
    stripe_options.stripe_rows = 1;
    MatmulOptions full_options{};
    full_options.mode = MatmulInvocationMode::full;
    const std::vector<float> activation_fp32 = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    const std::vector<float> weights_fp32 = { 1.0f, -1.0f, 2.0f, 3.0f };
    std::vector<float> fp32_expected(6, 0.0f);
    std::vector<float> fp32_output(6, 0.0f);
    ggml_gemmini_args_t fp32_args{};
    fp32_args.I = 3;
    fp32_args.J = 2;
    fp32_args.K = 2;
    fp32_args.A_fp32 = activation_fp32.data();
    fp32_args.B_fp32 = weights_fp32.data();
    fp32_args.sA = fp32_args.K;
    fp32_args.sB = fp32_args.J;
    fp32_args.f_out = fp32_output.data();
    fp32_args.col_stride_f_out = 1;
    fp32_args.stride_f_out = fp32_args.J;
    fp32_args.tiled_matmul_type = CPU;
    fp32_args.act_quant.storage().emplace<quants::act::NoneMeta>();
    matmul_cpu_fp(false, true, fp32_args.I, fp32_args.J, fp32_args.K,
                  activation_fp32.data(), weights_fp32.data(), nullptr,
                  fp32_expected.data(), fp32_args.sA, fp32_args.sB,
                  fp32_args.col_stride_f_out, fp32_args.stride_f_out);

    return check(metadata.valid(), "FP32 metadata storage remains a valid route") &&
        check(!metadata.scale(0, scale), "missing activation metadata has no scale fallback") &&
        check(quants::act::activation_scales(quantized_args, quantized_args.I).empty(),
              "missing activation metadata returns no scales") &&
        check(matmul(quantized_args, stripe_options).code == MatmulStatusCode::invalid_contract,
              "quantized stripe route rejects missing activation metadata") &&
        check(matmul(fp32_args, full_options).ok(), "FP32 route does not require activation metadata") &&
        check(same_output(fp32_output, fp32_expected),
              "FP32 NoneMeta route matches reference output");
}

bool test_copied_args_preserve_exsia_route_metadata() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1 };
    std::vector<elem_t> weights = { 1 };
    std::vector<float> output(1, 0.0f);
    auto args = make_args(activation, weights, output);
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0 };

    auto copied = args;
    if (!check(std::get_if<quants::act::exsia::Meta>(&copied.act_quant.storage()) != nullptr,
               "copied args preserve the ExSIA metadata alternative")) {
        return false;
    }
    return check(detail::normalize_route(copied).activation == detail::ActivationRoute::exsia,
                 "copied args route agrees with the ExSIA metadata alternative");
}

bool test_pointer_backed_stripes_preserve_global_metadata() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation(130, 1);
    std::vector<elem_t> weights = { 1 };
    std::vector<float> stripe_output(130, 0.0f);
    auto stripe_args = make_args(activation, weights, stripe_output);
    stripe_args.I = 130;
    stripe_args.J = 1;
    stripe_args.K = 1;
    stripe_args.sA = 1;
    stripe_args.sB = 1;
    stripe_args.stride_f_out = 1;
    stripe_args.tile_I = 4;
    auto & stripe_meta = stripe_args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    stripe_meta.theta = { -1, 0, 1 };
    stripe_meta.outliers = { { 63, 0, 1 }, { 64, 0, 2 }, { 129, 0, 4 } };
    const size_t original_theta_size = stripe_meta.theta.size();
    const size_t original_outlier_size = stripe_meta.outliers.size();

    MatMul facade(&stripe_args);
    return check(facade.begin_stripes() == MatMulStatus::success, "global metadata begin stripes") &&
        check(facade.run_stripe({ 0, 63 }) == MatMulStatus::success, "global metadata head stripe") &&
        check(stripe_meta.theta.size() == original_theta_size &&
                  stripe_meta.outliers.size() == original_outlier_size,
              "global metadata head preserves source storage") &&
        check(facade.run_stripe({ 63, 66 }) == MatMulStatus::success, "global metadata 63/64/65 stripe") &&
        check(stripe_meta.theta.size() == original_theta_size &&
                  stripe_meta.outliers.size() == original_outlier_size &&
                  stripe_meta.outliers[0].row == 63 &&
                  stripe_meta.outliers[1].row == 64 &&
                  stripe_meta.outliers[2].row == 129,
              "global metadata stripe preserves source storage and rows") &&
        check(facade.run_stripe({ 66, 128 }) == MatMulStatus::success, "global metadata middle stripe") &&
        check(stripe_meta.theta.size() == original_theta_size &&
                  stripe_meta.outliers.size() == original_outlier_size,
              "global metadata middle preserves source storage") &&
        check(facade.run_stripe({ 128, 130 }) == MatMulStatus::success, "global metadata tail stripe") &&
        check(stripe_meta.theta.size() == original_theta_size &&
                  stripe_meta.outliers.size() == original_outlier_size,
              "global metadata tail preserves source storage") &&
        check(facade.finish_stripes() == MatMulStatus::success, "global metadata finish stripes");
}

bool test_dense_residual_is_consumed_or_rejected() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> empty_output(6, 0.0f);
    std::vector<float> output(6, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;
    options.rc_shards = 1;

    auto empty_args = make_args(activation, weights, empty_output);
    empty_args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    auto empty_execution = prepare_execution(empty_args, options);
    auto empty = capture_stripe(empty_execution, MatmulStripeInput(0, 1, 0));
    if (!check(empty.status().ok() && run_staged_job(empty), "empty residual stripe execution")) {
        return false;
    }

    auto args = make_args(activation, weights, output);
    args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    auto execution = prepare_execution(args, options);
    const int32_t residual[] = { 0, 2 };
    auto job = capture_stripe(execution, MatmulStripeInput(0, 1, 0, residual, 2));
    const bool residual_rejected =
        job.status().code == MatmulStatusCode::unsupported_route ||
        job.status().code == MatmulStatusCode::invalid_contract;
    const bool residual_consumed = job.status().ok() && run_staged_job(job) &&
        output[0] == 9.0f && output[1] == 11.0f;
    if (!check(residual_rejected || residual_consumed,
               "dense residual is consumed or rejected explicitly")) {
        return false;
    }

    auto malformed_args = make_args(activation, weights, output);
    malformed_args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    auto malformed_execution = prepare_execution(malformed_args, options);
    auto malformed = capture_stripe(
        malformed_execution, MatmulStripeInput(0, 1, 0, residual, 1));
    auto overflow_args = make_args(activation, weights, output);
    overflow_args.act_quant.storage().emplace<quants::act::tensor::Meta>().scale = 1.0f;
    auto overflow_execution = prepare_execution(overflow_args, options);
    auto overflow = capture_stripe(
        overflow_execution, MatmulStripeInput(0, 1, 0, residual, 3));
    return check(malformed.status().code == MatmulStatusCode::invalid_argument,
                 "malformed dense residual cardinality rejected") &&
        check(overflow.status().code == MatmulStatusCode::invalid_argument,
              "oversized dense residual cardinality rejected");
}

bool test_non_exsia_pipeline_is_explicitly_unsupported() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    args.I = 65;
    args.K = 1;
    args.sA = 1;
    activation.assign(args.I * args.K, 1);
    args.A = activation.data();
    auto & meta = args.act_quant.storage().emplace<quants::act::token::Meta>();
    meta.scales.assign(args.I, 1.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    const auto expected = MatmulStatusCode::unsupported_invocation;
    return check(prepare_execution(args, options).status().code == expected,
                 "multi-row non-ExSIA strict pipeline unsupported");
}

bool test_explicit_unsupported_route_statuses() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation(130, 1);
    std::vector<elem_t> weights = { 1 };
    std::vector<float> output(130, 0.0f);
    auto args = make_args(activation, weights, output);
    args.I = 130;
    args.J = 1;
    args.K = 1;
    args.sA = 1;
    args.sB = 1;
    args.stride_f_out = 1;
    args.tile_I = 4;

    MatmulOptions stripe_options{};
    stripe_options.mode = MatmulInvocationMode::stripe_sequential;
    stripe_options.stripe_rows = 63;

    auto & exsia = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    exsia.theta = { -1, 0, 1 };
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h0;
    const auto q8_h0_status = matmul(args, stripe_options);

    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel;
    const auto q8_channel_direct = matmul(args, stripe_options);

    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
    const auto q8_channel_sidecar = matmul(args, stripe_options);

    return check(q8_h0_status.code == MatmulStatusCode::unsupported_route,
                 "Q8_H0 explicit unsupported_route status") &&
        check(q8_channel_direct.code == MatmulStatusCode::unsupported_route,
              "ExSIA Q8_CHANNEL direct explicit unsupported_route status") &&
        check(q8_channel_sidecar.code == MatmulStatusCode::unsupported_route,
              "ExSIA Q8_CHANNEL sidecar explicit unsupported_route status");
}

bool test_malformed_route_contract_rejected() {
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    auto args = make_args(activation, weights, output);
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    ggml::gemmini::MatMul facade(args);
    const auto result = facade.run_full();
    if (!check(result.status == ggml::gemmini::MatMulStatus::invalid_contract,
               "malformed native weight contract rejected")) {
        return false;
    }
    auto missing_dense_weight = make_args(activation, weights, output);
    missing_dense_weight.B = nullptr;
    ggml::gemmini::MatMul missing_dense_facade(missing_dense_weight);
    return check(missing_dense_facade.run_full().status == ggml::gemmini::MatMulStatus::invalid_contract,
                 "missing dense weight contract rejected");
}

bool test_bounded_pipeline_slots_and_reuse() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 2;
    options.profiling = true;
    auto args = make_args(activation, weights, output);
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0, 0 };
    auto execution = prepare_execution(args, options);
    if (!check(execution.state() == MatmulExecutionState::prepared, "execution prepared state")) {
        return false;
    }
    const quants::QactOutlier outlier[] = {{ 0, 0, 2 }};
    auto first = capture_stripe(execution, MatmulStripeInput(0, 1, 0, outlier, 1));
    auto second = capture_stripe(execution, { 1, 2 });
    auto blocked = capture_stripe(execution, { 2, 3 });

    if (!check(first.status().ok() && second.status().ok(), "pipeline captures") ||
        !check(execution.state() == MatmulExecutionState::running, "execution running state") ||
        !check(blocked.status().code == MatmulStatusCode::out_of_memory, "bounded backpressure") ||
        !check(finish_execution(execution).code == MatmulStatusCode::invalid_state, "finish with live jobs") ||
        !run_staged_job(first)) {
        return false;
    }
    auto tail = capture_stripe(execution, { 2, 3 });
    const auto first_metrics = first.metrics();
    const auto first_snapshot = first.snapshot();
    if (!(first_metrics.handoff.count == 1 && first_metrics.ws.count == 1 &&
          first_metrics.rc_prepare.count == 1 &&
          first_metrics.rc_compute.count == first_snapshot.expected_shards &&
          first_metrics.rc_finalize.count == 1)) {
        std::fprintf(stderr, "pipeline metrics were handoff=%zu ws=%zu rc_prepare=%zu rc_compute=%zu rc_finalize=%zu\n",
                     first_metrics.handoff.count, first_metrics.ws.count,
                     first_metrics.rc_prepare.count, first_metrics.rc_compute.count,
                     first_metrics.rc_finalize.count);
    }
    const bool passed = run_staged_job(second) && run_staged_job(tail) &&
        check(finish_execution(execution).ok(), "pipeline finish") &&
        check(execution.state() == MatmulExecutionState::completed, "execution completed state") &&
        check(first.metrics().handoff.count == 1 && first.metrics().ws.count == 1 &&
                  first.metrics().rc_prepare.count == 1 &&
                  first.metrics().rc_compute.count == first_snapshot.expected_shards &&
                  first.metrics().rc_finalize.count == 1,
              "pipeline metric counters");
    if (passed) {
        std::printf("PASS contract: rc_compute=%zu expected_shards=%zu\n",
                    first_metrics.rc_compute.count, first_snapshot.expected_shards);
        std::puts("PASS edge: pipeline=externally-staged capacity=2 backpressure=out_of_memory slot_reuse=yes");
    }
    return passed;
}

bool test_independent_branch_lifecycle() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;
    options.job_capacity = 1;

    const auto run_order = [&](bool dense_first) {
        std::vector<float> output(6, 0.0f);
        auto args = make_args(activation, weights, output);
        args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0, 0 };
        auto execution = prepare_execution(args, options);
        auto job = capture_stripe(execution, { 0, 1 });
        const auto run_compensation = [&]() {
            const size_t shards = job.snapshot().expected_shards;
            for (size_t shard = 0; shard < shards; ++shard) {
                if (!execute_compensation_shard(job, shard, shards).ok()) {
                    return false;
                }
            }
            return true;
        };
        if (!check(finalize_stripe(job).code == MatmulStatusCode::invalid_state,
                   "finalize before branch completion")) {
            return false;
        }
        if (dense_first) {
            if (!check(execute_dense_stripe(job).ok(), "dense-first dense branch") ||
                !check(prepare_compensation(job).ok(), "dense-first RC prepare") ||
                !check(run_compensation(), "dense-first RC branch")) {
                return false;
            }
        } else if (!check(prepare_compensation(job).ok(), "RC-first prepare") ||
                   !check(run_compensation(), "RC-first RC branch") ||
                   !check(execute_dense_stripe(job).ok(), "RC-first dense branch")) {
            return false;
        }
        if (!check(finalize_stripe(job).ok(), "branch-order finalize") ||
            !check(finalize_stripe(job).code == MatmulStatusCode::invalid_state,
                   "duplicate finalize")) {
            return false;
        }
        const auto snapshot = job.snapshot();
        if (!(snapshot.status.ok() && snapshot.captured && snapshot.finalized && snapshot.released &&
              snapshot.dense == MatmulDenseState::complete && snapshot.rc == MatmulRcState::complete &&
              snapshot.expected_shards > 0 && snapshot.completed_shards == snapshot.expected_shards)) {
            std::fprintf(stderr,
                         "branch snapshot status=%u captured=%d finalized=%d released=%d dense=%u rc=%u expected=%zu completed=%zu\n",
                         static_cast<unsigned>(snapshot.status.code), snapshot.captured, snapshot.finalized,
                         snapshot.released, static_cast<unsigned>(snapshot.dense),
                         static_cast<unsigned>(snapshot.rc), snapshot.expected_shards,
                         snapshot.completed_shards);
        }
        if (!check(snapshot.status.ok() && snapshot.captured && snapshot.finalized && snapshot.released &&
                       snapshot.dense == MatmulDenseState::complete &&
                       snapshot.rc == MatmulRcState::complete &&
                       snapshot.expected_shards > 0 &&
                       snapshot.completed_shards == snapshot.expected_shards,
                   "synchronized finalized branch snapshot")) {
            return false;
        }
        auto reused_slot = capture_stripe(execution, { 1, 3, 1 });
        return check(reused_slot.status().ok(), "finalize releases one capacity slot") &&
            run_staged_job(reused_slot) &&
            check(finish_execution(execution).ok(), "branch-order finish");
    };

    if (!run_order(true) || !run_order(false)) {
        return false;
    }

    for (size_t iteration = 0; iteration < 1000; ++iteration) {
        std::vector<float> output(6, 0.0f);
        auto args = make_args(activation, weights, output);
        args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0, 0 };
        auto execution = prepare_execution(args, options);
        auto job = capture_stripe(execution, { 0, 3 });
        MatmulStatus dense_status;
        MatmulStatus rc_status;
        std::thread dense([&] {
            std::this_thread::sleep_for(std::chrono::microseconds((iteration * 17) % 11));
            dense_status = execute_dense_stripe(job);
        });
        std::thread rc([&] {
            std::this_thread::sleep_for(std::chrono::microseconds((iteration * 31) % 13));
            rc_status = prepare_compensation(job);
            if (rc_status) {
                const size_t shards = job.snapshot().expected_shards;
                for (size_t shard = 0; shard < shards && rc_status; ++shard) {
                    rc_status = execute_compensation_shard(job, shard, shards);
                }
            }
        });
        dense.join();
        rc.join();
        if (!check(dense_status.ok() && rc_status.ok(), "random branch delay status") ||
            !check(finalize_stripe(job).ok(), "random branch delay finalize") ||
            !check(finish_execution(execution).ok(), "random branch delay finish")) {
            return false;
        }
    }
    std::puts("PASS edge: dense-first=yes rc-first=yes random-delays=1000 duplicate-finalize=invalid_state");
    return true;
}

bool test_staged_contract_errors() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(6, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;
    const auto make_valid_args = [&] {
        auto args = make_args(activation, weights, output);
        args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0, 0 };
        return args;
    };

    auto invalid_args = make_valid_args();
    invalid_args.I = 0;
    if (!check(prepare_execution(&invalid_args, options).status().code == MatmulStatusCode::invalid_argument,
               "invalid stripe shape rejected")) {
        return false;
    }

    auto null_residual_execution = prepare_execution(make_valid_args(), options);
    auto null_residual = capture_stripe(
        null_residual_execution,
        MatmulStripeInput(0, 1, 0, static_cast<const int32_t *>(nullptr), 1));
    auto null_outlier_execution = prepare_execution(make_valid_args(), options);
    auto null_outlier = capture_stripe(
        null_outlier_execution,
        MatmulStripeInput(0, 1, 0, static_cast<const quants::QactOutlier *>(nullptr), 1));
    if (!check(null_residual.status().code == MatmulStatusCode::invalid_argument,
               "null residual capture rejected before copy") ||
        !check(null_outlier.status().code == MatmulStatusCode::invalid_argument,
               "null outlier capture rejected before copy")) {
        return false;
    }

    auto order_execution = prepare_execution(make_valid_args(), options);
    auto early = capture_stripe(order_execution, { 0, 1 });
    if (!check(finalize_stripe(early).code == MatmulStatusCode::invalid_state,
               "finalize before compensation")) {
        return false;
    }

    auto contract_execution = prepare_execution(make_valid_args(), options);
    auto malformed = capture_stripe(contract_execution, { 1, 1 });
    auto first = capture_stripe(contract_execution, { 0, 1 });
    auto duplicate = capture_stripe(contract_execution, { 0, 1 });
    auto overlap = capture_stripe(contract_execution, { 0, 2 });
    auto duplicate_id = capture_stripe(contract_execution, { 2, 3, 0 });
    auto invalid_id = capture_stripe(contract_execution, { 2, 3, 3 });
    if (!check(malformed.status().code == MatmulStatusCode::invalid_argument, "malformed stripe") ||
        !check(duplicate.status().code == MatmulStatusCode::invalid_contract, "duplicate stripe") ||
        !check(overlap.status().code == MatmulStatusCode::invalid_contract, "overlapping stripe") ||
        !check(duplicate_id.status().code == MatmulStatusCode::invalid_contract, "duplicate stripe id") ||
        !check(invalid_id.status().code == MatmulStatusCode::invalid_argument, "invalid stripe id") ||
        !run_staged_job(first)) {
        return false;
    }
    auto tail = capture_stripe(contract_execution, { 2, 3 });
    const auto pipeline_rejection = MatmulStatusCode::unsupported_invocation;
    const bool passed = run_staged_job(tail) &&
        check(finish_execution(contract_execution).code == MatmulStatusCode::invalid_contract,
              "missing stripe at finish") &&
        check(prepare_execution(
                  [&] {
                      auto single_row = make_valid_args();
                      single_row.I = 1;
                      return single_row;
                  }(),
                  [] {
                      MatmulOptions options{};
                      options.mode = MatmulInvocationMode::stripe_pipeline;
                      return options;
                  }()).status().code == pipeline_rejection,
              "single-row pipeline rejected") &&
        check(matmul(make_valid_args(),
                     [] {
                         MatmulOptions options{};
                         options.mode = MatmulInvocationMode::stripe_pipeline;
                         return options;
                     }()).code == pipeline_rejection,
              "automatic pipeline is explicit unsupported invocation");
    if (passed) {
        std::puts("PASS edge: duplicate=invalid_contract overlap=invalid_contract missing=invalid_contract "
                  "early_finalize=invalid_state automatic_pipeline=unsupported_invocation");
    }
    return passed;
}

bool test_live_pipeline_worker() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(12, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 2;
    options.rc_shards = 4;
    options.profiling = true;
    auto args = make_args(activation, weights, output);
    args.I = 6;
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0, 0, 0, 0, 0 };
    const bool cpu_serial_route = detail::normalize_route(args).backend == detail::BackendRoute::cpu;
    auto execution = prepare_execution(args, options);
    MatmulStripeCollector collector(2);
    if (!check(execution.status().ok() && collector.start(execution), "live worker start")) {
        return false;
    }
    const auto * sink = collector.sink();
    const quants::QactOutlier outliers[] = {
        {0, 0, 2}, {1, 0, 2}, {2, 0, 2}, {3, 0, 2}, {4, 0, 2}, {5, 0, 2}};
    if (!check(sink->on_ready(sink->user_data, make_ready_event(0, 0, 1, &outliers[0], 1, 10, 20, 30, 50, 40, 70)), "live worker capture") ||
        !check(sink->on_ready(sink->user_data, make_ready_event(1, 1, 2, &outliers[1], 1, 11, 21, 31, 51, 41, 71)), "live worker capture") ||
        !check(sink->on_ready(sink->user_data, make_ready_event(2, 2, 3, &outliers[2], 1, 12, 22, 32, 52, 42, 72)), "live worker capture") ||
        !check(sink->on_ready(sink->user_data, make_ready_event(3, 3, 4, &outliers[3], 1, 13, 23, 33, 53, 43, 73)), "live worker capture") ||
        !check(sink->on_ready(sink->user_data, make_ready_event(4, 4, 5, &outliers[4], 1, 14, 24, 34, 54, 44, 74)), "live worker capture") ||
        !check(sink->on_ready(sink->user_data, make_ready_event(5, 5, 6, &outliers[5], 1, 15, 25, 35, 55, 45, 75)), "live worker tail capture") ||
        !check(collector.finish().ok(), "live worker finish") ||
        !check(collector.profiles().size() == 6, "live worker stripe profiles") ||
        !check(collector.profiles()[0].la_cycles == 10 && collector.profiles()[0].la3_cycles == 30 &&
                   collector.profiles()[0].sf_cycles == 20,
               "live worker producer profile") ||
        !check(collector.profiles()[0].ws_start_ns < collector.profiles()[0].ws_end_ns &&
                   collector.profiles()[0].rc_start_ns < collector.profiles()[0].rc_end_ns &&
                   (cpu_serial_route
                        ? collector.profiles()[0].ws_end_ns <= collector.profiles()[0].rc_start_ns
                        : collector.profiles()[0].rc_start_ns < collector.profiles()[0].ws_end_ns),
               cpu_serial_route ? "CPU pipeline matmul-then-dec ordering"
                                : "NPU pipeline matmul-dec-overlap ordering") ||
        !check(collector.snapshot().rc_worker_starts == 2 &&
                   collector.snapshot().rc_tasks_executed > collector.snapshot().rc_worker_starts &&
                   collector.snapshot().max_active_rc_stripes == 1 &&
                   collector.snapshot().max_rc_queue_depth > 0 &&
                   collector.snapshot().max_rc_queue_depth <= collector.snapshot().rc_worker_capacity &&
                   collector.snapshot().pending == 0 && collector.snapshot().in_flight == 0 &&
                   collector.snapshot().rc_queue_depth == 0,
               "RC workers persist across stripes") ||
        !check(collector.profiles()[0].capture_copy.count == 1 &&
                   collector.profiles()[0].producer_wait.count == 1 &&
                   collector.profiles()[0].queue_insert.count == 1 &&
                   collector.profiles()[0].sf_handoff.count == 1 &&
                   collector.profiles()[0].ws_queue.count == 1 &&
                   collector.profiles()[0].ws_service.count == 1 &&
                   collector.profiles()[0].rc_queue.count == 1 &&
                   collector.profiles()[0].rc_prepare.count == 1 &&
                   collector.profiles()[0].rc_compute.count == 2 &&
                   collector.profiles()[0].rc_finalize.count == 1 &&
                   collector.profiles()[0].rc_wait.count == 1 &&
                   collector.profiles()[0].rc_finalize.nanoseconds > 0 &&
                   collector.profiles()[0].t_RC4.nanoseconds ==
                       collector.profiles()[0].rc_prepare.nanoseconds +
                           collector.profiles()[0].rc_compute.nanoseconds +
                           collector.profiles()[0].rc_finalize.nanoseconds &&
                   collector.profiles()[0].t_RC4.count == 1,
               "pipeline stage and backpressure metrics") ||
        !check(finish_execution(execution).ok(), "live worker execution finish")) {
        return false;
    }
    const auto profile = collector.profiles()[0];
    std::printf("[matmul.stripe.stages] capture_copy_ns=%llu producer_wait_ns=%llu queue_insert_ns=%llu "
                "sf_handoff_ns=%llu ws_queue_ns=%llu ws_service_ns=%llu rc_queue_ns=%llu "
                "rc_prepare_ns=%llu rc_compute_ns=%llu rc_finalize_ns=%llu rc_wait_ns=%llu t_RC4_ns=%llu\n",
                static_cast<unsigned long long>(profile.capture_copy.nanoseconds),
                static_cast<unsigned long long>(profile.producer_wait.nanoseconds),
                static_cast<unsigned long long>(profile.queue_insert.nanoseconds),
                static_cast<unsigned long long>(profile.sf_handoff.nanoseconds),
                static_cast<unsigned long long>(profile.ws_queue.nanoseconds),
                static_cast<unsigned long long>(profile.ws_service.nanoseconds),
                static_cast<unsigned long long>(profile.rc_queue.nanoseconds),
                static_cast<unsigned long long>(profile.rc_prepare.nanoseconds),
                static_cast<unsigned long long>(profile.rc_compute.nanoseconds),
                static_cast<unsigned long long>(profile.rc_finalize.nanoseconds),
                static_cast<unsigned long long>(profile.rc_wait.nanoseconds),
                static_cast<unsigned long long>(profile.t_RC4.nanoseconds));
    std::puts("PASS edge: pipeline=live-worker capture->dense->rc->finish");
    return true;
}

bool test_live_worker_failed_capture_releases_collector_slot() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(4, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    auto args = make_args(activation, weights, output);
    args.I = 2;
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };
    auto execution = prepare_execution(args, options);
    MatmulStripeCollector collector(1);
    if (!check(execution.status().ok() && collector.start(execution),
               "failed-capture live worker start")) {
        return false;
    }
    const auto * sink = collector.sink();
    const quants::QactOutlier outlier = { 0, 0, 2 };
    const bool first_admitted = sink->on_ready(
        sink->user_data, make_ready_event(0, 0, 1, &outlier, 1));
    auto duplicate = std::async(std::launch::async, [sink] {
        const quants::QactOutlier duplicate_outlier = { 1, 0, 2 };
        return sink->on_ready(sink->user_data, make_ready_event(0, 1, 2, &duplicate_outlier, 1));
    });
    const bool duplicate_bounded = duplicate.wait_for(std::chrono::seconds(2)) ==
        std::future_status::ready;
    if (!duplicate_bounded) {
        collector.cancel();
    }
    const bool duplicate_admitted = duplicate.get();
    const auto finish = collector.finish();
    const auto execution_finish = finish_execution(execution);

    std::vector<float> replacement_output(4, 0.0f);
    auto replacement_args = make_args(activation, weights, replacement_output);
    replacement_args.I = 2;
    replacement_args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };
    auto replacement_execution = prepare_execution(replacement_args, options);
    MatmulStripeCollector replacement(1);
    const bool replacement_started = replacement.start(replacement_execution);
    const auto * replacement_sink = replacement.sink();
    const quants::QactOutlier replacement_outlier = { 0, 0, 2 };
    const bool replacement_admitted = replacement_started && replacement_sink->on_ready(
        replacement_sink->user_data, make_ready_event(0, 0, 2, &replacement_outlier, 1));
    const bool replacement_finished = replacement_admitted && replacement.finish().ok() &&
        finish_execution(replacement_execution).ok();

    const bool passed =
        check(first_admitted, "failed-capture first job admitted") &&
        check(duplicate_bounded && duplicate_admitted, "failed capture admission is bounded") &&
        check(finish.code == MatmulStatusCode::invalid_contract &&
                  execution_finish.code == MatmulStatusCode::invalid_contract,
              "failed capture status propagates") &&
        check(collector.snapshot().in_flight == 0, "failed capture releases collector slot") &&
        check(replacement_finished, "failed capture replacement capacity recovery");
    if (passed) {
        std::puts("PASS edge: failed-capture=duplicate-id bounded-finish=yes status-propagation=yes "
                  "in-flight-zero=yes replacement-capacity-recovery=yes");
    }
    return passed;
}

bool test_live_rc_workers_are_process_bounded() {
    using namespace ggml::gemmini;
    constexpr size_t rows = 2;
    constexpr size_t columns = 256;
    constexpr size_t depth = 2;
    constexpr size_t requested_shards = 99;
    std::vector<elem_t> activation(rows * depth, 1);
    std::vector<elem_t> weights(depth * columns, 1);
    std::vector<float> output(rows * columns, 0.0f);
    ggml_gemmini_args_t args{};
    args.I = rows;
    args.J = columns;
    args.K = depth;
    args.A = activation.data();
    args.B = weights.data();
    args.sA = depth;
    args.sB = columns;
    args.f_out = output.data();
    args.col_stride_f_out = 1;
    args.stride_f_out = columns;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    args.tiled_matmul_type = CPU;
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };

    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    options.rc_shards = requested_shards;
    auto execution = prepare_execution(args, options);
    MatmulStripeCollector collector(1);
    if (!check(execution.status().ok() && collector.start(execution), "bounded RC worker start")) {
        return false;
    }
    const auto running = collector.snapshot();
    const auto cancel_status = collector.cancel();
    const auto finish_status = collector.finish();
    const auto execution_status = finish_execution(execution);
    const auto stopped = collector.snapshot();
    const size_t expected_workers = expected_persistent_rc_workers(requested_shards, columns);
    const bool passed =
        check(running.rc_worker_capacity == expected_workers,
              "RC worker capacity clamps to process-safe budget") &&
        check(stopped.rc_worker_starts == expected_workers,
              "RC worker start count matches bounded capacity") &&
        check(cancel_status.code == MatmulStatusCode::cancelled &&
                  finish_status.code == MatmulStatusCode::cancelled &&
                  execution_status.code == MatmulStatusCode::cancelled,
              "bounded RC worker cleanup");
    if (passed) {
        std::printf("PASS edge: live-rc-workers requested=%zu bounded=%zu columns=%zu\n",
                    requested_shards, expected_workers, columns);
    }
    return passed;
}

bool test_malformed_event_wakes_blocked_producer() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(4, 0.0f);
    auto args = make_args(activation, weights, output);
    args.I = 2;
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    auto execution = prepare_execution(args, options);
    MatmulStripeCollector collector(1);
    collector.test_pause_dense_before_execute();
    if (!check(execution.status().ok() && collector.start(execution),
               "malformed-event blocked-producer start")) {
        return false;
    }
    const auto * sink = collector.sink();
    const quants::QactOutlier first_outlier = { 0, 0, 2 };
    const bool first_admitted = sink->on_ready(
        sink->user_data, make_ready_event(0, 0, 1, &first_outlier, 1));
    const auto in_flight_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (collector.snapshot().in_flight != 1 &&
           std::chrono::steady_clock::now() < in_flight_deadline) {
        std::this_thread::yield();
    }
    auto blocked = std::async(std::launch::async, [sink] {
        const quants::QactOutlier second_outlier = { 1, 0, 2 };
        return sink->on_ready(sink->user_data, make_ready_event(1, 1, 2, &second_outlier, 1));
    });
    const bool producer_blocked = blocked.wait_for(std::chrono::milliseconds(20)) ==
        std::future_status::timeout;
    const bool malformed_rejected = !sink->on_ready(sink->user_data, make_ready_event(2, 1, 1, nullptr, 0));
    const bool producer_woke = blocked.wait_for(std::chrono::seconds(2)) ==
        std::future_status::ready;
    if (!producer_woke) {
        collector.cancel();
    }
    const bool blocked_rejected = !blocked.get();
    const auto finish = collector.finish();
    const auto execution_finish = finish_execution(execution);
    const bool passed =
        check(first_admitted && producer_blocked, "producer blocks at collector capacity") &&
        check(malformed_rejected, "malformed producer event rejected") &&
        check(producer_woke && blocked_rejected, "malformed event wakes blocked producer") &&
        check(finish.code == MatmulStatusCode::invalid_argument &&
                  execution_finish.code == MatmulStatusCode::invalid_argument,
              "malformed event status propagates");
    if (passed) {
        std::puts("PASS edge: malformed-event=wakes-blocked-producer status-propagation=yes");
    }
    return passed;
}

#if defined(_OPENMP)
bool test_live_pipeline_worker_from_openmp_region() {
    bool passed = false;
#pragma omp parallel num_threads(2) shared(passed)
    {
#pragma omp single
        passed = test_live_pipeline_worker();
    }
    return check(passed, "live worker inside existing OpenMP region");
}
#endif

bool run_captured_compensation(size_t shard_count, std::vector<float> & output) {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;
    options.rc_shards = shard_count;
    auto args = make_args(activation, weights, output);
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0, 0 };
    auto execution = prepare_execution(args, options);
    auto job = capture_stripe(execution, { 0, 3, 0 }, std::vector<quants::QactOutlier>{{ 0, 0, 2 }});
    if (!job.status().ok() || !prepare_compensation(job).ok() ||
        !execute_dense_stripe(job).ok())
        return false;
    const size_t actual_shards = job.snapshot().expected_shards;
    for (size_t shard = 0; shard < actual_shards; ++shard)
        if (!execute_compensation_shard(job, shard, actual_shards).ok())
            return false;
    return finalize_stripe(job).ok() && finish_execution(execution).ok();
}

bool test_compensation_shard_output_is_bitwise_stable() {
    std::vector<float> one(6, 0.0f);
    const std::array<size_t, 5> shard_counts{ 1, 2, 3, 4, 99 };
    const bool one_ok = run_captured_compensation(shard_counts[0], one);
    if (!one_ok) {
        return check(false, "single compensation shard");
    }
    for (size_t i = 1; i < shard_counts.size(); ++i) {
        std::vector<float> output(6, 0.0f);
        if (!check(run_captured_compensation(shard_counts[i], output),
                   "multi compensation shards")) {
            return false;
        }
        if (!check(same_output(one, output), "compensation shard output differs")) {
            std::fprintf(stderr, "baseline shards=%zu candidate shards=%zu\n",
                         shard_counts[0], shard_counts[i]);
            return false;
        }
    }
    std::puts("PASS edge: captured-compensation shard-counts=1/2/3/4/99 stable");
    return true;
}

bool test_compensation_shards_preserve_native_scale_groups() {
    using namespace ggml::gemmini;
    constexpr size_t columns = 129;
    constexpr size_t depth = 65;
    constexpr size_t blocks = 3;
    constexpr size_t stripe_width = 17;
    std::vector<elem_t> activation(depth, 1);
    std::vector<int8_t> weights(columns * depth, 1);
    std::vector<uint8_t> c_b(columns * blocks, 86);
    std::vector<float> stripe_s_rf((columns + stripe_width - 1) / stripe_width,
                                   0.00032747327350080013f);
    std::vector<uint16_t> stripe_R(stripe_s_rf.size(), 4095);
    std::vector<float> output(columns, 0.0f);
    std::vector<float> reference_output(columns, 0.0f);

    const auto make_native_scale_args = [&](std::vector<float> & target) {
        ggml_gemmini_args_t args{};
        args.I = 1;
        args.J = columns;
        args.K = depth;
        args.A = activation.data();
        args.B = reinterpret_cast<elem_t *>(weights.data());
        args.sA = depth;
        args.sB = depth;
        args.f_out = target.data();
        args.stride_f_out = columns;
        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_0_unpacked_to_h1;
        args.c_b = c_b.data();
        args.blocks_per_row = blocks;
        args.blocks_K = blocks;
        args.blocks_J = columns;
        args.block_size_k = QK8_0;
        args.stripe_J = stripe_width;
        args.s_rf_stripe = stripe_s_rf.data();
        args.R_stripe = stripe_R.data();
        args.act_quant.storage().emplace<quants::act::tensor::Meta>();
        args.tiled_matmul_type = CPU;
        return args;
    };

    auto args = make_native_scale_args(output);
    auto reference_args = make_native_scale_args(reference_output);
    const std::vector<quants::QactOutlier> outliers = {{ 0, 64, 548339296 }};

    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_sequential;
    options.rc_shards = 4;
    auto execution = prepare_execution(args, options);
    auto job = capture_stripe(execution, { 0, 1, 0 }, outliers);
    if (!check(job.status().ok() && prepare_compensation(job).ok(),
               "native-scale compensation preparation")) {
        return false;
    }
    const size_t actual_shards = job.snapshot().expected_shards;
    if (!check(actual_shards == 4, "native-scale compensation uses four shards")) {
        return false;
    }
    if (!check(execute_dense_stripe(job).ok(), "native-scale dense stripe")) {
        return false;
    }
    for (size_t shard = 0; shard < actual_shards; ++shard) {
        if (!check(execute_compensation_shard(job, shard, actual_shards).ok(),
                   "native-scale compensation shard")) {
            return false;
        }
    }
    if (!check(finalize_stripe(job).ok(), "native-scale finalize stripe") ||
        !check(finish_execution(execution).ok(), "native-scale finish execution")) {
        return false;
    }

    auto dense_reference = MatMul(reference_args).run_dense();
    quants::dec::compensate_activation_dec(
        outliers, reference_args, "test-native-scale-reference");
    return check(dense_reference.status == MatMulStatus::success, "native-scale dense reference") &&
        check(same_output(output, reference_output),
              "native-scale staged compensation matches direct DEC reference");
}

bool run_live_worker_compensation(size_t shard_count, size_t capacity, std::vector<float> & output) {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = capacity;
    options.rc_shards = shard_count;
    auto args = make_args(activation, weights, output);
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0 };
    auto execution = prepare_execution(args, options);
    MatmulStripeCollector collector(capacity);
    if (!execution.status().ok() || !collector.start(execution)) {
        return false;
    }
    quants::QactOutlier outlier{ 0, 0, 2 };
    const auto * sink = collector.sink();
    const bool captured = sink->on_ready(
        sink->user_data, make_ready_event(0, 0, 3, &outlier, 1, 10, 20, 30, 50));
    return captured && collector.finish().ok() && finish_execution(execution).ok();
}

bool test_live_worker_serial_compensation_is_bitwise_stable() {
    using namespace ggml::gemmini;
    const std::array<size_t, 4> capacities{ 1, 2, 3, 4 };
    const std::array<size_t, 5> shard_counts{ 1, 2, 3, 4, 99 };
    std::vector<float> baseline(6, 0.0f);
    if (!check(run_live_worker_compensation(shard_counts[0], capacities[0], baseline),
               "live single-shard compensation")) {
        return false;
    }
    for (size_t capacity : capacities) {
        for (size_t shard_count : shard_counts) {
            std::vector<float> output(6, 0.0f);
            if (!check(run_live_worker_compensation(shard_count, capacity, output),
                       "live staged compensation run")) {
                return false;
            }
            if (!check(same_output(baseline, output), "live compensation output differs")) {
                std::fprintf(stderr, "baseline capacity=%zu shards=%zu candidate capacity=%zu shards=%zu\n",
                             capacities[0], shard_counts[0], capacity, shard_count);
                return false;
            }
        }
    }
    std::puts("PASS edge: live-worker compensation capacities=1/2/3/4 shard-counts=1/2/3/4/99");
    return true;
}

bool test_pipeline_cancellation() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation(8, 1);
    std::vector<elem_t> weights(4, 1);
    std::vector<float> output(8, 0.0f);
    auto args = make_args(activation, weights, output);
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0, 0 };
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 2;
    auto execution = prepare_execution(&args, options);
    MatmulStripeCollector collector(2);
    if (!check(execution.status().ok() && collector.start(execution), "pipeline cancellation start")) {
        return false;
    }
    const auto cancel_status = collector.cancel();
    const auto finish_status = collector.finish();
    const auto execution_finish_status = finish_execution(execution);
    const bool passed =
        check(cancel_status.code == MatmulStatusCode::cancelled, "pipeline cancellation status") &&
        check(finish_status.code == MatmulStatusCode::cancelled, "pipeline cancellation finish") &&
        check(execution_finish_status.code == MatmulStatusCode::cancelled,
              "pipeline cancellation execution finish") &&
        check(execution.state() == MatmulExecutionState::failed, "pipeline cancellation execution state");
    if (passed) {
        std::puts("PASS edge: pipeline-cancellation=yes collector/execution-failure-state=yes");
    }
    return passed;
}

bool test_pipeline_execution_attachment_contract() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(4, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    auto args = make_args(activation, weights, output);
    args.I = 2;
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };
    auto execution = prepare_execution(args, options);
    MatmulStripeCollector collector(1);
    if (!check(execution.status().ok() && collector.start(execution), "attachment-contract start")) {
        return false;
    }
    const bool attached_while_running = execution.test_pipeline_attached();
    MatmulStripeCollector duplicate(1);
    const bool duplicate_started = duplicate.start(execution);
    const auto duplicate_status = duplicate.status();
    const auto attached_finish_status = finish_execution(execution);
    const auto cancel_status = collector.cancel();
    const auto finish_status = collector.finish();
    const auto execution_status = finish_execution(execution);
    const bool detached_after_finish = !execution.test_pipeline_attached();
    const bool passed =
        check(attached_while_running, "execution marks live collector attachment") &&
        check(!duplicate_started &&
                  duplicate_status.code == MatmulStatusCode::invalid_state,
              "duplicate collector start rejected") &&
        check(attached_finish_status.code == MatmulStatusCode::invalid_state,
              "attached collector blocks finish without mutating execution") &&
        check(detached_after_finish, "execution detaches after collector finish") &&
        check(cancel_status.code == MatmulStatusCode::cancelled &&
                  finish_status.code == MatmulStatusCode::cancelled &&
                  execution_status.code == MatmulStatusCode::cancelled,
              "attachment-contract cleanup");
    if (passed) {
        std::puts("PASS edge: execution-collector-attachment tracked across live worker lifetime");
    }
    return passed;
}

bool test_pipeline_finish_waits_for_startup_attachment() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(4, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    auto args = make_args(activation, weights, output);
    args.I = 2;
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };
    auto execution = prepare_execution(args, options);
    MatmulStripeCollector collector(1);
    collector.test_pause_startup_after_attachment();
    auto start = std::async(std::launch::async, [&collector, &execution] {
        return collector.start(execution);
    });
    const auto attached_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!execution.test_pipeline_attached() &&
           std::chrono::steady_clock::now() < attached_deadline) {
        std::this_thread::yield();
    }
    const bool attached_while_starting = execution.test_pipeline_attached();
    auto finish = std::async(std::launch::async, [&collector] { return collector.finish(); });
    const bool finish_blocked = finish.wait_for(std::chrono::milliseconds(20)) ==
        std::future_status::timeout;
    collector.test_resume_startup();
    const bool start_bounded = start.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    const bool started = start_bounded && start.get();
    const bool finish_bounded = finish.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    const auto finish_status = finish_bounded ? finish.get() : MatmulStatus{
        MatmulStatusCode::execution_failure, "finish timed out" };
    const bool detached_after_finish = !execution.test_pipeline_attached();
    const auto execution_status = finish_execution(execution);
    const bool passed =
        check(attached_while_starting,
              "startup-attachment collector attaches execution before finish returns") &&
        check(finish_blocked, "finish waits for collector startup serialization") &&
        check(start_bounded && started, "startup serialization start completes") &&
        check(finish_bounded && finish_status.ok(), "startup serialization finish completes") &&
        check(detached_after_finish, "startup serialization detaches execution") &&
        check(execution_status.code == MatmulStatusCode::invalid_contract,
              "startup serialization preserves missing-stripe contract");
    if (passed) {
        std::puts("PASS edge: startup-serialization finish waits for attached collector startup");
    }
    return passed;
}

bool test_pipeline_thread_exceptions_fail_cleanly() {
    using namespace ggml::gemmini;
    struct Case {
        MatmulCollectorThread thread;
        MatmulCollectorThreadFailure failure;
        MatmulStatusCode expected;
        const char * label;
    };
    const std::array<Case, 4> cases{{
        { MatmulCollectorThread::worker,
          MatmulCollectorThreadFailure::exception,
          MatmulStatusCode::execution_failure,
          "worker exception" },
        { MatmulCollectorThread::compensation,
          MatmulCollectorThreadFailure::exception,
          MatmulStatusCode::execution_failure,
          "compensation exception" },
        { MatmulCollectorThread::rc_worker,
          MatmulCollectorThreadFailure::exception,
          MatmulStatusCode::execution_failure,
          "RC worker exception" },
        { MatmulCollectorThread::worker,
          MatmulCollectorThreadFailure::out_of_memory,
          MatmulStatusCode::out_of_memory,
          "worker OOM" },
    }};

    for (const auto & test_case : cases) {
        std::vector<elem_t> activation = { 1, 2, 3, 4 };
        std::vector<elem_t> weights = { 1, -1, 2, 3 };
        std::vector<float> output(4, 0.0f);
        MatmulOptions options{};
        options.mode = MatmulInvocationMode::stripe_pipeline;
        options.job_capacity = 1;
        auto args = make_args(activation, weights, output);
        args.I = 2;
        args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };
        auto execution = prepare_execution(args, options);
        MatmulStripeCollector collector(1);
        collector.test_inject_thread_exception(test_case.thread, test_case.failure);
        if (!check(collector.start(execution), test_case.label)) {
            return false;
        }
        auto finish = std::async(std::launch::async, [&collector] { return collector.finish(); });
        const bool finish_bounded = finish.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        const auto finish_status = finish_bounded ? finish.get() : MatmulStatus{
            MatmulStatusCode::execution_failure, "finish timed out" };
        const auto execution_status = finish_execution(execution);
        if (!check(finish_bounded, "thread exception finish is bounded") ||
            !check(finish_status.code == test_case.expected, "thread exception collector status") ||
            !check(execution_status.code == test_case.expected &&
                       execution.state() == MatmulExecutionState::failed,
                   "thread exception execution failure") ||
            !check(!execution.test_pipeline_attached(),
                   "thread exception detaches execution after finish")) {
            return false;
        }
    }
    std::puts("PASS edge: injected worker/compensation/RC thread exceptions fail cleanly");
    return true;
}

bool test_pipeline_cancel_sets_rc_stop_flag() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(8, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    options.rc_shards = 4;
    auto args = make_args(activation, weights, output);
    args.I = 2;
    args.J = 4;
    args.stride_f_out = 4;
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };
    auto execution = prepare_execution(args, options);
    MatmulStripeCollector collector(1);
    if (!check(execution.status().ok() && collector.start(execution), "cancel RC-stop start")) {
        return false;
    }
    const auto cancel_status = collector.cancel();
    const bool rc_stop_requested = collector.test_rc_stop_requested();
    const auto finish_status = collector.finish();
    const auto execution_status = finish_execution(execution);
    const bool passed =
        check(cancel_status.code == MatmulStatusCode::cancelled, "cancel RC-stop status") &&
        check(rc_stop_requested, "cancel sets RC stop request flag") &&
        check(finish_status.code == MatmulStatusCode::cancelled &&
                  execution_status.code == MatmulStatusCode::cancelled,
              "cancel RC-stop cleanup");
    if (passed) {
        std::puts("PASS edge: cancel raises RC stop flag and stops further RC scheduling");
    }
    return passed;
}

bool test_pipeline_start_thread_failure_returns_clean_status() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(4, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    options.rc_shards = 1;
    auto args = make_args(activation, weights, output);
    args.I = 2;
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };
    auto execution = prepare_execution(args, options);
    MatmulStripeCollector collector(1);
    collector.test_inject_thread_start_failure(3);
    auto start = std::async(std::launch::async, [&collector, &execution] {
        return collector.start(execution);
    });
    const bool bounded = start.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    const bool started = bounded && start.get();
    const auto collector_status = collector.status();
    const auto finish_status = collector.finish();
    const auto execution_status = execution.status();
    const bool passed =
        check(bounded, "thread-start failure start is bounded") &&
        check(!started, "thread-start failure returns false") &&
        check(collector_status.code == MatmulStatusCode::execution_failure,
              "thread-start failure sets collector status") &&
        check(finish_status.code == MatmulStatusCode::execution_failure,
              "thread-start failure finish preserves status") &&
        check(execution_status.code == MatmulStatusCode::execution_failure &&
                  execution.state() == MatmulExecutionState::failed,
              "thread-start failure sets execution failed state") &&
        check(!execution.test_pipeline_attached(),
              "thread-start failure clears live collector attachment");
    if (passed) {
        std::puts("PASS edge: thread-start failure returns clean status without live collector attachment");
    }
    return passed;
}

bool test_live_worker_rc_failure_releases_collector_slot_once() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(4, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    auto args = make_args(activation, weights, output);
    args.I = 2;
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };
    auto execution = prepare_execution(args, options);
    MatmulStripeCollector collector(1);
    collector.test_inject_rc_failure(
        { MatmulStatusCode::execution_failure, "injected RC worker failure" });
    if (!check(execution.status().ok() && collector.start(execution), "RC-failure live worker start")) {
        return false;
    }
    const auto * sink = collector.sink();
    const bool injected_job_admitted = sink->on_ready(
        sink->user_data, make_ready_event(0, 0, 2, nullptr, 0, 10, 20, 30, 50));
    const auto failure = collector.finish();
    const auto execution_failure = finish_execution(execution);

    std::vector<float> replacement_output(4, 0.0f);
    auto replacement_args = make_args(activation, weights, replacement_output);
    replacement_args.I = 2;
    replacement_args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };
    auto replacement_execution = prepare_execution(replacement_args, options);
    MatmulStripeCollector replacement(1);
    const bool replacement_started = replacement.start(replacement_execution);
    const auto * replacement_sink = replacement.sink();
    const bool replacement_admitted = replacement_started && replacement_sink->on_ready(
        replacement_sink->user_data, make_ready_event(0, 0, 2, nullptr, 0, 10, 20, 30, 50));
    const bool replacement_finished = replacement_admitted && replacement.finish().ok() &&
        finish_execution(replacement_execution).ok();

    const bool passed =
        check(injected_job_admitted, "RC-failure job admitted") &&
        check(failure.code == MatmulStatusCode::execution_failure &&
                  execution_failure.code == MatmulStatusCode::execution_failure,
              "RC failure propagates to collector and execution") &&
        check(collector.test_dense_state_at_release() == MatmulDenseState::complete,
              "RC failure preserves independently completed dense branch") &&
        check(collector.test_in_flight() == 0, "RC failure releases collector slot exactly once") &&
        check(replacement_finished, "replacement slot admission and capacity recovery");
    if (passed) {
        std::puts("PASS edge: RC-failure-injection=yes dense-completion-independent=yes "
                  "exactly-once-slot-release=yes slot-reuse/capacity-recovery=yes");
    }
    return passed;
}

bool test_rc_failure_external_cancel_while_dense_idle() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(4, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    auto args = make_args(activation, weights, output);
    args.I = 2;
    args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };
    const bool cpu_serial_route = detail::normalize_route(args).backend == detail::BackendRoute::cpu;
    auto execution = prepare_execution(args, options);
    MatmulStripeCollector collector(1);
    collector.test_inject_rc_failure(
        { MatmulStatusCode::execution_failure, "injected RC worker failure" });
    collector.test_pause_dense_before_execute();
    if (!check(execution.status().ok() && collector.start(execution),
               "RC-failure/cancel live worker start")) {
        return false;
    }
    const auto * sink = collector.sink();
    const bool admitted = sink->on_ready(
        sink->user_data, make_ready_event(0, 0, 2, nullptr, 0, 10, 20, 30, 50));
    if (!cpu_serial_route) {
        collector.test_wait_for_rc_failure();
    }
    const auto cancel_status = collector.cancel();
    const auto duplicate_cancel_status = collector.cancel();
    auto finish = std::async(std::launch::async, [&collector] { return collector.finish(); });
    const bool bounded = finish.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    const auto finish_status = bounded ? finish.get() : MatmulStatus{
        MatmulStatusCode::execution_failure, "finish timed out" };
    const auto execution_status = finish_execution(execution);

    std::vector<float> replacement_output(4, 0.0f);
    auto replacement_args = make_args(activation, weights, replacement_output);
    replacement_args.I = 2;
    replacement_args.act_quant.storage().emplace<quants::act::exsia::Meta>().theta = { 0, 0 };
    auto replacement_execution = prepare_execution(replacement_args, options);
    MatmulStripeCollector replacement(1);
    const bool replacement_started = replacement.start(replacement_execution);
    const auto * replacement_sink = replacement.sink();
    const bool replacement_admitted = replacement_started && replacement_sink->on_ready(
        replacement_sink->user_data, make_ready_event(0, 0, 2, nullptr, 0, 10, 20, 30, 50));
    const bool replacement_finished = replacement_admitted && replacement.finish().ok() &&
        finish_execution(replacement_execution).ok();
    const auto dense_state_at_release = collector.test_dense_state_at_release();

    const bool passed =
        check(admitted, "RC-failure/cancel job admitted") &&
        check(bounded, "RC-failure/cancel finish is bounded") &&
        check(cancel_status.code == MatmulStatusCode::cancelled &&
                  duplicate_cancel_status.code == MatmulStatusCode::cancelled &&
                  finish_status.code == MatmulStatusCode::cancelled &&
                  execution_status.code == MatmulStatusCode::cancelled,
              "external cancellation propagates after RC failure") &&
        check(cpu_serial_route
                  ? (dense_state_at_release == MatmulDenseState::idle ||
                     dense_state_at_release == MatmulDenseState::cancelled)
                  : (dense_state_at_release == MatmulDenseState::complete ||
                     dense_state_at_release == MatmulDenseState::cancelled),
              "external cancel leaves Dense complete or cancelled") &&
        check(collector.test_in_flight() == 0,
              "RC-failure/cancel releases collector slot exactly once") &&
        check(replacement_finished, "RC-failure/cancel slot recovery");
    if (passed) {
        std::puts("PASS edge: RC-failure+external-cancel=yes bounded-finish=yes in-flight-zero=yes "
                  "no-double-release=yes slot-recovery=yes");
    }
    return passed;
}

}

int main(int argc, char ** argv) {
    const bool edge_only = argc == 2 && std::string_view(argv[1]) == "--edge";
    const bool stress_only = argc == 2 && std::string_view(argv[1]) == "--stress";
    if (stress_only) {
        for (size_t iteration = 0; iteration < 1000; ++iteration) {
            if (!test_live_pipeline_worker() ||
                !test_live_worker_failed_capture_releases_collector_slot() ||
                !test_pipeline_cancellation() ||
                !test_live_worker_rc_failure_releases_collector_slot_once() ||
                !test_rc_failure_external_cancel_while_dense_idle()) {
                std::fprintf(stderr, "stress iteration %zu failed\n", iteration);
                return 1;
            }
        }
        std::puts("PASS stress: iterations=1000 worker-reuse/cancel/failure=yes");
        return 0;
    }
    const bool configuration = test_public_contract_shape() &&
        test_pipeline_stripe_summary_contract() &&
        test_matmul_option_resolution_precedence() &&
        test_default_matmul_mode_executes_configured_backend_path() &&
        test_dispatch_override_contract() &&
        test_route_capability_table() && test_disabled_stripe_modes_are_rejected();
    if (!ggml::gemmini::config::ENABLE_STRIPE_MATMUL) {
        return configuration && test_full_facade_status_and_output_match_legacy() &&
                test_malformed_route_contract_rejected()
            ? 0
            : 1;
    }
    const bool pipeline = !ggml::gemmini::config::ENABLE_STRIPE_PIPELINE ||
        (test_bounded_pipeline_slots_and_reuse() && test_live_pipeline_worker() &&
         test_live_rc_workers_are_process_bounded() &&
#if defined(_OPENMP)
         test_live_pipeline_worker_from_openmp_region() &&
#endif
         test_live_worker_failed_capture_releases_collector_slot() &&
         test_malformed_event_wakes_blocked_producer() &&
         test_live_worker_serial_compensation_is_bitwise_stable() &&
         test_pipeline_cancellation() &&
         test_pipeline_execution_attachment_contract() &&
         test_pipeline_finish_waits_for_startup_attachment() &&
         test_pipeline_thread_exceptions_fail_cleanly() &&
         test_pipeline_cancel_sets_rc_stop_flag() &&
         test_pipeline_start_thread_failure_returns_clean_status() &&
         test_live_worker_rc_failure_releases_collector_slot_once() &&
         test_rc_failure_external_cancel_while_dense_idle());
    const bool edge = configuration &&
        test_h2_and_hp2_stripe_capability_is_explicitly_unsupported() &&
        test_explicit_exsia_channel_rejection() &&
        test_global_activation_metadata_view_boundaries() &&
        test_invalid_activation_scale_is_explicit_contract_error() &&
        test_missing_activation_metadata_allows_dense_routes_and_fp32() &&
        test_fp32_stripe_route_skips_quantized_compensation() &&
        test_copied_args_preserve_exsia_route_metadata() &&
        test_pointer_backed_stripes_preserve_global_metadata() &&
        test_dense_residual_is_consumed_or_rejected() &&
        test_non_exsia_pipeline_is_explicitly_unsupported() &&
        test_explicit_unsupported_route_statuses() &&
        test_malformed_route_contract_rejected() &&
        test_independent_branch_lifecycle() &&
        test_compensation_shard_output_is_bitwise_stable() &&
        test_compensation_shards_preserve_native_scale_groups() &&
        test_staged_contract_errors() && pipeline;
    if (edge_only) {
        return edge ? 0 : 1;
    }
    return edge && test_full_facade_status_and_output_match_legacy() &&
            test_fp32_full_facade_matches_legacy() &&
            test_baseline_activation_route_facade_parity() &&
            test_j131_tail_stripe_parity() &&
            test_fp32_shape_and_stride_matrix() &&
            (!ggml::gemmini::config::ENABLE_STRIPE_PIPELINE ||
             test_live_pipeline_multistripe_matches_full()) &&
            test_native_and_channel_full_facade_parity() &&
            test_full_and_stripe_sequential_outputs_match() &&
            test_block_activation_scale_compensation_parity() &&
            test_native_exsia_theta_row_slice_parity() &&
            test_native_exsia_multistripe_residual_parity() &&
            test_empty_tail_and_malformed_stripe_status() &&
            test_duplicate_and_overlap_stripe_status() &&
            test_h2_and_hp2_stripe_capability_is_explicitly_unsupported() &&
            test_stripe_state_lifecycle()
        ? 0
        : 1;
}
