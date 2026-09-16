#include "../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {

using namespace ggml::gemmini;
using Format = ggml_gemmini_args_t::im2p_weight_format_t;

bool initialize_activation(ggml_gemmini_args_t & args)
{
    if (!args.A.allocate(args.I, args.K, 8)) {
        return false;
    }
    for (size_t row = 0; row < args.I; ++row) {
        for (size_t column = 0; column < args.K; ++column) {
            if (!args.A.set(row, column, 1)) {
                return false;
            }
        }
    }
    return true;
}

void initialize_args(ggml_gemmini_args_t & args, float * output)
{
    args.I = 1;
    args.J = 1;
    args.K = 64;
    args.sA = args.K;
    args.f_out = output;
    args.stride_f_out = 1;
    args.col_stride_f_out = 1;
    args.tiled_matmul_type = static_cast<tiled_matmul_type_t>(2);
    args.tile_I = 1;
    args.tile_J = 1;
    args.tile_K = 1;
    args.activation_rows_per_stripe = DIM;
    args.transpose_B = true;
    args.blocks_per_row = 2;
    args.blocks_K = 2;
    args.blocks_J = 1;
    args.blocks_I = 1;
    args.block_size_k = 32;
}

void initialize_metadata(
    ggml_gemmini_args_t & args,
    size_t rows,
    std::initializer_list<float> scales)
{
    auto & meta = args.act_quant.storage().emplace<quants::act::block::Meta>();
    meta.rows = rows;
    meta.cols = args.K;
    meta.scales = scales;
}

MatmulStatus run(ggml_gemmini_args_t & args)
{
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::full;
    options.rmd_backend = RmdBackend::cpu_direct;
    return matmul(args, options);
}

bool report(
    const char * scenario,
    const MatmulStatus & status,
    float actual,
    double reference)
{
    const double error = std::fabs(static_cast<double>(actual) - reference);
    std::printf(
        "BLOCK_DENSE %s actual=%.9g reference=%.9g max_error=%.9g status=%u\n",
        scenario, static_cast<double>(actual), reference, error,
        static_cast<unsigned>(status.code));
    return status.ok() && error <= 1e-5 * std::max(1.0, std::fabs(reference));
}

bool native_h1()
{
    float output = 777.0f;
    ggml_gemmini_args_t args{};
    initialize_args(args, &output);
    if (!initialize_activation(args)) {
        return false;
    }
    initialize_metadata(args, 1, {1.0f, 10.0f});

    std::array<block_q8_h1, 2> weights{};
    std::fill(std::begin(weights[0].qs), std::end(weights[0].qs), int8_t{1});
    std::fill(std::begin(weights[1].qs), std::end(weights[1].qs), int8_t{-1});
    weights[0].c_b = 1;
    weights[0].s_rf = 0.5f;
    weights[1].c_b = 1;
    weights[1].s_rf = 2.0f;
    args.weight_format = Format::q8_h1;
    args.q8_h1_blocks = weights.data();
    args.q8_h1_block_count = weights.size();
    args.q8_h1_rows = 1;
    args.native_weight_bytes = sizeof(weights);

    const MatmulStatus status = run(args);
    return report("native_h1", status, output, 32.0 * 0.5 - 32.0 * 2.0 * 10.0);
}

bool native_hp1_slice()
{
    float output = 777.0f;
    ggml_gemmini_args_t args{};
    initialize_args(args, &output);
    args.activation_row_offset = 1;
    if (!initialize_activation(args)) {
        return false;
    }
    initialize_metadata(args, 2, {7.0f, 11.0f, 2.0f, 0.5f});

    std::array<block_q8_hp1, 2> weights{};
    std::fill(std::begin(weights[0].qs), std::end(weights[0].qs), int8_t{1});
    std::fill(std::begin(weights[1].qs), std::end(weights[1].qs), int8_t{-1});
    weights[0].channel_scale = 0.25f;
    weights[1].channel_scale = 2.0f;
    args.weight_format = Format::q8_hp1;
    args.q8_hp1_blocks = weights.data();
    args.q8_hp1_block_count = weights.size();
    args.q8_hp1_blocks_per_row = 2;
    args.native_weight_bytes = sizeof(weights);

    const MatmulStatus status = run(args);
    return report("native_hp1_slice", status, output, 32.0 * 0.25 * 2.0 - 32.0 * 2.0 * 0.5);
}

bool baseline_scalar_bias()
{
    std::array<float, 15> output;
    output.fill(777.0f);
    ggml_gemmini_args_t args{};
    initialize_args(args, output.data());
    args.I = 3;
    args.J = 2;
    args.K = 65;
    args.sA = args.K;
    args.stride_f_out = 5;
    args.col_stride_f_out = 2;
    args.tile_I = 2;
    args.tile_K = 4;
    if (!initialize_activation(args)) {
        return false;
    }
    for (size_t i = 0; i < args.I; ++i)
        for (size_t k = 0; k < args.K; ++k)
            if (!args.A.set(i, k, static_cast<int32_t>(i + 1)))
                return false;
    initialize_metadata(args, 3, {1, 10, 2, 4, 5, 6, 7, 8, 9});

    std::array<elem_t, 130> weights{};
    for (size_t j = 0; j < args.J; ++j)
        for (size_t k = 0; k < args.K; ++k)
            weights[j * args.K + k] = static_cast<elem_t>(
                (k < 32 ? 1 : -1) * static_cast<int>(j + 1));
    args.B = weights.data();
    args.sB = args.K;
    args.weight_i8_scale_active = true;
    args.weight_scale = 0.5f;
    const std::array<acc_t, 9> bias = {3, 4, 99, 5, 6, 99, 7, 8, 99};
    args.D = bias.data();
    args.sD = 3;
    args.scale_D = 2;

    const MatmulStatus status = run(args);
    const auto & scales = std::get<quants::act::block::Meta>(args.act_quant.storage()).scales;
    for (size_t i = 0; i < args.I; ++i) {
        for (size_t j = 0; j < args.J; ++j) {
            const double reference = (i + 1) * (j + 1) * 0.5 *
                (32 * scales[i * 3] - 32 * scales[i * 3 + 1] - scales[i * 3 + 2]) +
                2 * bias[i * 3 + j];
            if (!report("baseline_rows_tail_bias", status, output[i * 5 + j * 2], reference))
                return false;
        }
        if (output[i * 5 + 1] != 777 || output[i * 5 + 3] != 777 || output[i * 5 + 4] != 777)
            return false;
    }
    return true;
}

bool external_unpacked_h1()
{
    float output = 777.0f;
    ggml_gemmini_args_t args{};
    initialize_args(args, &output);
    if (!initialize_activation(args)) {
        return false;
    }
    initialize_metadata(args, 1, {1.0f, 10.0f});

    std::array<elem_t, 64> weights{};
    std::fill(weights.begin(), weights.begin() + 32, elem_t{1});
    std::fill(weights.begin() + 32, weights.end(), elem_t{-1});
    const std::array<uint8_t, 2> block_scales = {1, 3};
    const uint16_t row_offset = 1;
    const float row_scale = 0.5f;
    args.B = weights.data();
    args.sB = args.K;
    args.weight_format = Format::q8_0_unpacked_to_h1;
    args.c_b = block_scales.data();
    args.R = &row_offset;
    args.s_rf = &row_scale;

    const MatmulStatus status = run(args);
    if (!report(
        "external_unpacked_h1", status, output,
        32.0 * 2.0 * 0.5 - 32.0 * 4.0 * 0.5 * 10.0)) {
        return false;
    }

    output = 777.0f;
    auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = {0};
    const MatmulStatus non_block_status = run(args);
    std::printf(
        "BLOCK_DENSE external_unpacked_h1_non_block actual=%.9g sentinel=777 status=%u\n",
        static_cast<double>(output), static_cast<unsigned>(non_block_status.code));
    return !non_block_status.ok() && output == 777.0f;
}

bool baseline_channel()
{
    float output = 777.0f;
    ggml_gemmini_args_t args{};
    initialize_args(args, &output);
    if (!initialize_activation(args)) {
        return false;
    }
    initialize_metadata(args, 1, {1.0f, 10.0f});

    std::array<elem_t, 64> weights{};
    std::fill(weights.begin(), weights.begin() + 32, elem_t{1});
    std::fill(weights.begin() + 32, weights.end(), elem_t{-1});
    const float weight_scale = 3.0f;
    args.B = weights.data();
    args.sB = args.K;
    args.weight_format = Format::q8_channel_dense_sidecar;
    args.weight_channel_scales = &weight_scale;
    args.weight_channel_scale_count = 1;

    const MatmulStatus status = run(args);
    return report("baseline_channel", status, output, -288.0 * 3.0);
}

bool invalid_metadata_is_atomic()
{
    float output = 777.0f;
    ggml_gemmini_args_t args{};
    initialize_args(args, &output);
    if (!initialize_activation(args)) {
        return false;
    }
    initialize_metadata(
        args, 1, {1.0f, std::numeric_limits<float>::quiet_NaN()});

    std::array<elem_t, 64> weights{};
    std::fill(weights.begin(), weights.end(), elem_t{1});
    args.B = weights.data();
    args.sB = args.K;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;

    const MatmulStatus status = run(args);
    std::printf(
        "BLOCK_DENSE invalid_metadata actual=%.9g sentinel=777 status=%u\n",
        static_cast<double>(output), static_cast<unsigned>(status.code));
    return !status.ok() && output == 777.0f;
}

bool native_q4(Format format, const char * scenario)
{
    float output = 777.0f;
    ggml_gemmini_args_t args{};
    initialize_args(args, &output);
    if (!args.A.allocate(args.I, args.K, 4)) {
        return false;
    }
    for (size_t column = 0; column < args.K; ++column) {
        if (!args.A.set(0, column, 1)) {
            return false;
        }
    }
    initialize_metadata(args, 1, {1.0f, 10.0f});

    std::array<block_q4_h0, 2> h0{};
    std::array<block_q4_h1, 2> h1{};
    std::array<block_q4_hp1, 2> hp1{};
    std::memset(h0[0].qs, 0x99, sizeof(h0[0].qs));
    std::memset(h0[1].qs, 0x77, sizeof(h0[1].qs));
    std::memset(h1[0].qs, 0x99, sizeof(h1[0].qs));
    std::memset(h1[1].qs, 0x77, sizeof(h1[1].qs));
    std::memset(hp1[0].qs, 0x99, sizeof(hp1[0].qs));
    std::memset(hp1[1].qs, 0x77, sizeof(hp1[1].qs));
    h0[0].d = ggml_fp32_to_fp16(0.5f);
    h0[1].d = ggml_fp32_to_fp16(2.0f);
    h1[0].c_b = h1[1].c_b = 1;
    h1[0].s_rf = 0.5f;
    h1[1].s_rf = 2.0f;
    hp1[0].channel_scale = 0.5f;
    hp1[1].channel_scale = 2.0f;

    args.weight_format = format;
    if (format == Format::q4_h0) {
        args.q4_h0_blocks = h0.data();
        args.native_weight_bytes = sizeof(h0);
    } else if (format == Format::q4_h1) {
        args.q4_h1_blocks = h1.data();
        args.native_weight_bytes = sizeof(h1);
    } else {
        args.q4_hp1_blocks = hp1.data();
        args.native_weight_bytes = sizeof(hp1);
    }
    args.native_block_count = 2;
    args.native_blocks_per_row = 2;

    const MatmulStatus status = run(args);
    return report(scenario, status, output, 32.0 * 0.5 - 32.0 * 2.0 * 10.0);
}

}

int main()
{
#if GGML_GEMMINI_ACTIVATION_BITS == 8 && GGML_GEMMINI_WEIGHT_BITS == 8
    return native_h1() && native_hp1_slice() && external_unpacked_h1() &&
            baseline_scalar_bias() &&
            baseline_channel() && invalid_metadata_is_atomic() ?
        0 : 1;
#elif GGML_GEMMINI_ACTIVATION_BITS == 4 && GGML_GEMMINI_WEIGHT_BITS == 4
    return native_q4(Format::q4_h0, "native_q4_h0") &&
            native_q4(Format::q4_h1, "native_q4_h1") &&
            native_q4(Format::q4_hp1, "native_q4_hp1") ?
        0 : 1;
#else
    std::puts("BLOCK_DENSE skipped: requires matched A4/W4 or A8/W8 build");
    return 0;
#endif
}
