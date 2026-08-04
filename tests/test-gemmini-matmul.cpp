#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp"

#include <gemmini.h>

#include <array>
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
    return check(result.status == ggml::gemmini::MatMulStatus::success, "FP32 facade status") &&
        check(route.activation == ggml::gemmini::detail::ActivationRoute::fp32 &&
              route.weight == ggml::gemmini::detail::WeightRoute::fp32,
              "FP32 route normalization") &&
        check(same_output(facade_output, legacy_output), "FP32 facade output differs from legacy matmul") &&
        check(stripe_result.ok(), "FP32 stripe facade status") &&
        check(same_output(stripe_output, legacy_output), "FP32 stripe facade output differs from legacy matmul");
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
        return check(result.status == MatMulStatus::success, label) &&
            check(same_output(facade_output, legacy_output), "baseline route facade output differs") &&
            check(MatMul::stripe_capability(facade_args) == expected_stripe,
                  "baseline route stripe capability") &&
            check(stripe_result.ok(), "baseline route stripe execution") &&
            check(same_output(stripe_output, legacy_output), "baseline route stripe output differs");
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
        run(std::move(token_meta), Route::TOKEN, "TOKEN baseline facade", MatMulCapability::supported) &&
        run(std::move(block_meta), Route::BLOCK, "BLOCK baseline facade", MatMulCapability::supported) &&
        run(std::move(stripe_meta), Route::BLOCK, "STRIPE baseline facade", MatMulCapability::supported);
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
    const auto full_result = matmul(full_args);
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
                const auto full_status = matmul(full_args);
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
    const bool captured = sink->on_ready(sink->user_data, { 0, 0, 80 }) &&
        sink->on_ready(sink->user_data, { 1, 80, rows });
    const auto collector_status = collector.finish();
    const auto execution_status = finish_execution(execution);
    for (const auto & profile : collector.profiles()) {
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
    return check(captured, "multistripe ready events") &&
        check(collector_status.ok(), "multistripe collector finish") &&
        check(execution_status.ok(), "multistripe execution finish") &&
        check(same_output(pipeline_output, full_output), "multistripe pipeline differs from full") &&
        check(collector.profiles().size() == 2, "multistripe profile count");
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
        check(stripe_result.ok(), "BLOCK compensation stripe status") &&
        check(same_output(stripe_output, full_output), "BLOCK compensation scale differs");
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
    MatmulOptions stripe_options{};
    stripe_options.mode = MatmulInvocationMode::stripe_sequential;
    stripe_options.stripe_rows = 1;
    stripe_options.rc_shards = 2;
    return check(matmul(full_args).ok(), "full public matmul") &&
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
    validation_options.validation = true;
    const auto validation_status = matmul(
        make_args(validation_activation, validation_weights, validation_output), validation_options);
    MatmulOptions too_many_dense_threads{};
    too_many_dense_threads.dense_threads = 2;
    auto dense_thread_execution = prepare_execution(
        make_args(validation_activation, validation_weights, validation_output), too_many_dense_threads);
    MatmulOptions staged_options{};
    staged_options.mode = MatmulInvocationMode::stripe_sequential;
    MatmulExecution staged_execution;
    auto staged_args = make_args(validation_activation, validation_weights, validation_output);
    const auto staged_prepare = prepare_execution(staged_args, staged_options, staged_execution);
    MatmulStripeJob staged_job;
    const auto staged_capture = capture_stripe(staged_execution, MatmulStripeInput(0, 1), staged_job);
    return check(defaults.job_capacity == 4, "default job capacity") &&
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
        check(staged_prepare.ok() && staged_execution.state() == MatmulExecutionState::running,
              "output-parameter prepare execution") &&
        check(staged_capture.ok() && staged_job.status().ok(),
              "output-parameter capture stripe") &&
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

bool test_dispatch_override_contract() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> row_output(6, 0.0f);
    std::vector<float> group_output(6, 0.0f);

    MatmulOptions row_options{};
    row_options.force_row_direct = true;
    auto row_execution = prepare_execution(make_args(activation, weights, row_output), row_options);
    const MatmulStatus row_status = execute_full(row_execution);

    MatmulOptions group_options{};
    group_options.force_group_k_csc = true;
    auto group_execution = prepare_execution(make_args(activation, weights, group_output), group_options);
    const MatmulStatus group_status = execute_full(group_execution);

    MatmulOptions conflicting_options{};
    conflicting_options.force_row_direct = true;
    conflicting_options.force_group_k_csc = true;
    auto conflicting_execution = prepare_execution(
        make_args(activation, weights, row_output), conflicting_options);

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
    if (!check(args.act_quant.kind() == quants::act::MetaKind::none, "empty activation kind")) {
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
    const auto os_status = matmul(args);
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
    return check(key.activation == ggml::gemmini::detail::ActivationRoute::exsia &&
                     key.weight == ggml::gemmini::detail::WeightRoute::q8_channel_direct,
                 "EXSIA Q8_CHANNEL route normalization") &&
        check(!caps.full && !caps.sliced_compensation,
              "EXSIA Q8_CHANNEL explicit unsupported capability");
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
    options.rc_shards = 4;
    options.profiling = true;
    auto execution = prepare_execution(make_args(activation, weights, output), options);
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
    const bool passed = run_staged_job(second) && run_staged_job(tail) &&
        check(finish_execution(execution).ok(), "pipeline finish") &&
        check(execution.state() == MatmulExecutionState::completed, "execution completed state") &&
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

    auto invalid_args = make_args(activation, weights, output);
    invalid_args.I = 0;
    if (!check(prepare_execution(&invalid_args, options).status().code == MatmulStatusCode::invalid_argument,
               "invalid stripe shape rejected")) {
        return false;
    }

    auto null_residual_execution = prepare_execution(make_args(activation, weights, output), options);
    auto null_residual = capture_stripe(
        null_residual_execution,
        MatmulStripeInput(0, 1, 0, static_cast<const int32_t *>(nullptr), 1));
    auto null_outlier_execution = prepare_execution(make_args(activation, weights, output), options);
    auto null_outlier = capture_stripe(
        null_outlier_execution,
        MatmulStripeInput(0, 1, 0, static_cast<const quants::QactOutlier *>(nullptr), 1));
    if (!check(null_residual.status().code == MatmulStatusCode::invalid_argument,
               "null residual capture rejected before copy") ||
        !check(null_outlier.status().code == MatmulStatusCode::invalid_argument,
               "null outlier capture rejected before copy")) {
        return false;
    }

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
    const bool passed = run_staged_job(tail) &&
        check(finish_execution(contract_execution).code == MatmulStatusCode::invalid_contract,
              "missing stripe at finish") &&
        check(prepare_execution(
                  [&] {
                      auto single_row = make_args(activation, weights, output);
                      single_row.I = 1;
                      return single_row;
                  }(),
                  { MatmulInvocationMode::stripe_pipeline }).status().code ==
                  MatmulStatusCode::unsupported_invocation,
              "single-row pipeline rejected") &&
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
    std::vector<elem_t> activation = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    std::vector<elem_t> weights = { 1, -1, 2, 3 };
    std::vector<float> output(12, 0.0f);
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 2;
    auto args = make_args(activation, weights, output);
    args.I = 6;
    auto execution = prepare_execution(args, options);
    MatmulStripeCollector collector(2);
    if (!check(execution.status().ok() && collector.start(execution), "live worker start")) {
        return false;
    }
    const auto * sink = collector.sink();
    if (!check(sink->on_ready(sink->user_data, { 0, 0, 1, nullptr, 0, 10, 20, 30, 50, 40, 70 }), "live worker capture") ||
        !check(sink->on_ready(sink->user_data, { 1, 1, 2, nullptr, 0, 11, 21, 31, 51, 41, 71 }), "live worker capture") ||
        !check(sink->on_ready(sink->user_data, { 2, 2, 3, nullptr, 0, 12, 22, 32, 52, 42, 72 }), "live worker capture") ||
        !check(sink->on_ready(sink->user_data, { 3, 3, 4, nullptr, 0, 13, 23, 33, 53, 43, 73 }), "live worker capture") ||
        !check(sink->on_ready(sink->user_data, { 4, 4, 5, nullptr, 0, 14, 24, 34, 54, 44, 74 }), "live worker capture") ||
        !check(sink->on_ready(sink->user_data, { 5, 5, 6, nullptr, 0, 15, 25, 35, 55, 45, 75 }), "live worker tail capture") ||
        !check(collector.finish().ok(), "live worker finish") ||
        !check(collector.profiles().size() == 6, "live worker stripe profiles") ||
        !check(collector.profiles()[0].la_cycles == 10 && collector.profiles()[0].la3_cycles == 30 &&
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

bool test_pipeline_cancellation() {
    using namespace ggml::gemmini;
    std::vector<elem_t> activation(8, 1);
    std::vector<elem_t> weights(4, 1);
    std::vector<float> output(8, 0.0f);
    auto args = make_args(activation, weights, output);
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
    return check(cancel_status.code == MatmulStatusCode::cancelled, "pipeline cancellation status") &&
        check(finish_status.code == MatmulStatusCode::cancelled, "pipeline cancellation finish") &&
        check(execution_finish_status.code == MatmulStatusCode::cancelled,
              "pipeline cancellation execution finish") &&
        check(execution.state() == MatmulExecutionState::failed, "pipeline cancellation execution state");
}

}

int main(int argc, char ** argv) {
    const bool edge_only = argc == 2 && std::string_view(argv[1]) == "--edge";
    const bool edge = test_public_contract_shape() && test_dispatch_override_contract() &&
        test_route_capability_table() &&
        test_h2_and_hp2_stripe_capability_is_explicitly_unsupported() &&
        test_explicit_exsia_channel_rejection() &&
        test_malformed_route_contract_rejected() &&
        test_bounded_pipeline_slots_and_reuse() &&
        test_live_pipeline_worker() && test_compensation_shard_output_is_bitwise_stable() &&
        test_live_worker_parallel_compensation_is_bitwise_stable() &&
        test_pipeline_cancellation() &&
        test_staged_contract_errors();
    if (edge_only) {
        return edge ? 0 : 1;
    }
    return edge && test_full_facade_status_and_output_match_legacy() &&
            test_fp32_full_facade_matches_legacy() &&
            test_baseline_activation_route_facade_parity() &&
            test_j131_tail_stripe_parity() &&
            test_fp32_shape_and_stride_matrix() &&
            test_live_pipeline_multistripe_matches_full() &&
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
