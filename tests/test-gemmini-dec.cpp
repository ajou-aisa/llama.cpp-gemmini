#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/dec/dec.hpp"
#include "../ggml/src/ggml-gemmini/quants/dec/dec_internal.hpp"
#include "../ggml/src/ggml-gemmini/quants/dec/dec_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr float kTolerance = 1e-6f;

bool close_enough(float actual, float expected) {
    return std::fabs(actual - expected) <= kTolerance;
}

bool check(bool condition, const char *message) {
    if (!condition)
        std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

ggml_gemmini_args_t dense_args(
    size_t rows,
    size_t cols,
    size_t depth,
    const std::vector<int8_t> &weights,
    std::vector<float> &output,
    float scale) {
    ggml_gemmini_args_t args{};
    args.I = rows;
    args.J = cols;
    args.K = depth;
    args.B = reinterpret_cast<elem_t *>(const_cast<int8_t *>(weights.data()));
    args.sB = cols;
    args.f_out = output.data();
    args.weight_i8_scale_active = true;
    args.weight_scale = scale;
    return args;
}

bool test_noop() {
    const std::vector<int8_t> weights = { 1, 2, 3, 4, 5, 6 };
    std::vector<float> output = { 3.0f, -2.0f, 7.0f, 11.0f };
    ggml_gemmini_args_t args = dense_args(2, 2, 3, weights, output, 0.5f);
    const auto result = ggml::gemmini::quants::dec::compensate_activation_dec({}, args, "test");
    return check(result.total_selected == 0 && result.nnz == 0 && result.unique_k_count == 0,
                 "no-op result") &&
        check(output == std::vector<float>({ 3.0f, -2.0f, 7.0f, 11.0f }), "no-op output");
}

bool test_route_plan() {
    const std::vector<int8_t> weights = { 1, 2, 3, 4, 5, 6 };
    std::vector<float> output(2, 0.0f);
    ggml_gemmini_args_t scalar_args = dense_args(1, 2, 3, weights, output, 0.5f);
    const auto scalar_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        scalar_args,
        ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    bool ok = check(scalar_plan.valid &&
                        scalar_plan.route == ggml::gemmini::quants::dec::DecWeightRoute::Dense &&
                        scalar_plan.layout == ggml::gemmini::quants::dec::WeightLayout::KxJ_RowMajor &&
                        scalar_plan.weight_stride == 2 && scalar_plan.scales.scalar_mode,
                    "scalar route plan") &&
        check(std::string(ggml::gemmini::quants::dec::dec_route_name(scalar_plan)) == "tensor-scalar",
              "scalar route name");

    const std::vector<float> block_scales = { 0.25f, 0.5f };
    ggml_gemmini_args_t block_args = dense_args(1, 2, 3, weights, output, 1.0f);
    block_args.weight_i8_scale_active = false;
    block_args.B_scales = block_scales.data();
    block_args.blocks_J = 2;
    block_args.blocks_K = 1;
    block_args.block_size_k = 3;
    const auto block_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        block_args,
        ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    ok = check(block_plan.valid && !block_plan.scales.scalar_mode && block_plan.scales.block_size == 3 &&
                   ggml::gemmini::quants::dec::dec_route_covers_k(block_plan, 3),
               "block route plan") && ok;

    const std::vector<float> channel_scales = { 0.25f, 0.5f };
    ggml_gemmini_args_t sidecar_args = dense_args(1, 2, 3, weights, output, 1.0f);
    sidecar_args.weight_i8_scale_active = false;
    sidecar_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
    sidecar_args.sB = 3;
    sidecar_args.weight_channel_scales = channel_scales.data();
    sidecar_args.weight_channel_scale_count = 2;
    const auto sidecar_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        sidecar_args,
        ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    ok = check(sidecar_plan.valid &&
                   sidecar_plan.route == ggml::gemmini::quants::dec::DecWeightRoute::Q8ChannelSidecar &&
                   sidecar_plan.scales.channel_mode,
               "channel sidecar route plan") && ok;

    ggml_gemmini_args_t h0_args = scalar_args;
    h0_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h0;
    const auto h0_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        h0_args,
        ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    ggml_gemmini_args_t malformed_channel_args = scalar_args;
    malformed_channel_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel;
    const auto malformed_channel_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        malformed_channel_args,
        ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    return check(!h0_plan.valid && !malformed_channel_plan.valid, "unsupported and malformed route plans") && ok;
}

bool test_repeated_residuals() {
    const std::vector<int8_t> weights = {
        1, 1, 1,
        2, -2, 3,
        1, 1, 1,
        -1, 4, 2,
    };
    std::vector<float> output(6, 0.0f);
    ggml_gemmini_args_t args = dense_args(2, 3, 4, weights, output, 0.25f);
    const std::vector<ggml::gemmini::quants::QactOutlier> outliers = {
        { 0, 1, 5 }, { 0, 1, -2 }, { 1, 3, 4 }, { 1, 1, -3 },
    };
    const auto result = ggml::gemmini::quants::dec::compensate_activation_dec(outliers, args, "test");
    const std::vector<float> expected = { 1.5f, -1.5f, 2.25f, -2.5f, 5.5f, -0.25f };

    bool ok = check(result.total_selected == 4 && result.nnz == 4 && result.unique_k_count == 2,
                    "repeated residual accounting");
    for (size_t index = 0; index < output.size(); ++index)
        ok = check(close_enough(output[index], expected[index]), "repeated residual output") && ok;
    return ok;
}

bool test_decode_repeated_residuals() {
    const std::vector<int8_t> weights = {
        1, 1, 1,
        2, -2, 3,
        1, 1, 1,
        -1, 4, 2,
    };
    std::vector<float> output(3, 0.0f);
    ggml_gemmini_args_t args = dense_args(1, 3, 4, weights, output, 0.25f);
    const std::vector<ggml::gemmini::quants::QactOutlier> outliers = {
        { 0, 1, 5 }, { 0, 1, -2 }, { 0, 3, 4 },
    };
    const auto result = ggml::gemmini::quants::dec::compensate_activation_dec(outliers, args, "test");
    const std::vector<float> expected = { 0.5f, 2.5f, 4.25f };

    bool ok = check(result.total_selected == 3 && result.nnz == 3 && result.unique_k_count == 2,
                    "decode repeated residual accounting");
    for (size_t index = 0; index < output.size(); ++index)
        ok = check(close_enough(output[index], expected[index]), "decode repeated residual output") && ok;
    return ok;
}

std::vector<float> channel_expected(
    size_t rows,
    size_t cols,
    size_t depth,
    const std::vector<int8_t> &codes,
    const std::vector<float> &scales,
    const std::vector<ggml::gemmini::quants::QactOutlier> &outliers) {
    std::vector<float> expected(rows * cols, 0.0f);
    for (const auto &outlier : outliers) {
        if (outlier.row < 0 || outlier.col < 0 || static_cast<size_t>(outlier.row) >= rows ||
            static_cast<size_t>(outlier.col) >= depth)
            continue;
        for (size_t j = 0; j < cols; ++j)
            expected[static_cast<size_t>(outlier.row) * cols + j] += static_cast<float>(
                static_cast<double>(outlier.residual) * codes[j * depth + static_cast<size_t>(outlier.col)] * scales[j]);
    }
    return expected;
}

bool test_integer_routes() {
    constexpr size_t rows = 2;
    constexpr size_t cols = 3;
    constexpr size_t depth = 5;
    const std::vector<int8_t> channel_codes = {
        1, -2, 3, -4, 5,
        -3, 2, 1, 4, -2,
        2, 5, -1, 3, 4,
    };
    const std::vector<float> scales = { 0.25f, 0.5f, -0.25f };
    const std::vector<ggml::gemmini::quants::QactOutlier> outliers = {
        { 0, 0, 3 }, { 0, 4, -2 }, { 1, 1, 5 }, { 1, 1, -1 },
    };
    const std::vector<float> expected = channel_expected(rows, cols, depth, channel_codes, scales, outliers);

    std::vector<int8_t> scalar_codes(depth * cols);
    for (size_t k = 0; k < depth; ++k)
        for (size_t j = 0; j < cols; ++j)
            scalar_codes[k * cols + j] = channel_codes[j * depth + k];
    std::vector<float> scalar_output(rows * cols, 0.0f);
    ggml_gemmini_args_t scalar_args = dense_args(rows, cols, depth, scalar_codes, scalar_output, 0.25f);
    const std::vector<float> scalar_expected = channel_expected(
        rows, cols, depth, channel_codes, { 0.25f, 0.25f, 0.25f }, outliers);
    auto reversed_outliers = outliers;
    std::reverse(reversed_outliers.begin(), reversed_outliers.end());
    ggml::gemmini::quants::dec::compensate_activation_dec(outliers, scalar_args, "test");
    std::vector<float> scalar_reordered(rows * cols, 0.0f);
    ggml_gemmini_args_t scalar_reordered_args = dense_args(rows, cols, depth, scalar_codes, scalar_reordered, 0.25f);
    ggml::gemmini::quants::dec::compensate_activation_dec(reversed_outliers, scalar_reordered_args, "test");
    std::vector<float> scalar_transposed(rows * cols, 0.0f);
    ggml_gemmini_args_t scalar_transposed_args = dense_args(rows, cols, depth, channel_codes, scalar_transposed, 0.25f);
    scalar_transposed_args.sB = depth;
    scalar_transposed_args.transpose_B = true;
    ggml::gemmini::quants::dec::compensate_activation_dec(outliers, scalar_transposed_args, "test");

    std::vector<float> sidecar_output(rows * cols, 0.0f);
    ggml_gemmini_args_t sidecar_args{};
    sidecar_args.I = rows;
    sidecar_args.J = cols;
    sidecar_args.K = depth;
    sidecar_args.B = reinterpret_cast<elem_t *>(const_cast<int8_t *>(channel_codes.data()));
    sidecar_args.sB = depth;
    sidecar_args.f_out = sidecar_output.data();
    sidecar_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
    sidecar_args.weight_channel_scales = scales.data();
    sidecar_args.weight_channel_scale_count = cols;
    ggml::gemmini::quants::dec::compensate_activation_dec(outliers, sidecar_args, "test");
    const std::vector<ggml::gemmini::quants::QactOutlier> decode_outliers = {
        { 0, 2, 4 }, { 0, 2, -1 }, { 0, 4, 3 },
    };
    std::vector<float> sidecar_decode_output(cols, 0.0f);
    sidecar_args.I = 1;
    sidecar_args.f_out = sidecar_decode_output.data();
    ggml::gemmini::quants::dec::compensate_activation_dec(decode_outliers, sidecar_args, "test");

    std::vector<uint8_t> direct_rows(cols * (sizeof(float) + depth));
    for (size_t j = 0; j < cols; ++j) {
        uint8_t *row = direct_rows.data() + j * (sizeof(float) + depth);
        std::memcpy(row, &scales[j], sizeof(float));
        std::memcpy(row + sizeof(float), channel_codes.data() + j * depth, depth);
    }
    std::vector<float> direct_output(rows * cols, 0.0f);
    ggml_gemmini_args_t direct_args{};
    direct_args.I = rows;
    direct_args.J = cols;
    direct_args.K = depth;
    direct_args.B = reinterpret_cast<elem_t *>(direct_rows.data() + sizeof(float));
    direct_args.sB = sizeof(float) + depth;
    direct_args.f_out = direct_output.data();
    direct_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_channel;
    direct_args.q8_channel_row_base = direct_rows.data();
    direct_args.q8_channel_row_stride = sizeof(float) + depth;
    direct_args.q8_channel_row_count = cols;
    ggml::gemmini::quants::dec::compensate_activation_dec(outliers, direct_args, "test");

    std::vector<float> direct_decode_output(cols, 0.0f);
    direct_args.I = 1;
    direct_args.f_out = direct_decode_output.data();
    ggml::gemmini::quants::dec::compensate_activation_dec(decode_outliers, direct_args, "test");
    const std::vector<float> direct_decode_expected = channel_expected(
        1, cols, depth, channel_codes, scales, decode_outliers);

    bool ok = true;
    for (size_t index = 0; index < expected.size(); ++index) {
        ok = check(close_enough(scalar_output[index], scalar_expected[index]), "scalar int64 tail output") && ok;
        ok = check(close_enough(scalar_reordered[index], scalar_output[index]), "scalar ordering invariance") && ok;
        ok = check(close_enough(scalar_transposed[index], scalar_expected[index]), "scalar transposed int64 output") && ok;
        ok = check(close_enough(sidecar_output[index], expected[index]), "sidecar int64 output") && ok;
        ok = check(close_enough(direct_output[index], expected[index]), "direct int64 output") && ok;
    }
    for (size_t j = 0; j < cols; ++j)
        ok = check(close_enough(direct_decode_output[j], direct_decode_expected[j]), "direct int64 decode output") && ok;
    for (size_t j = 0; j < cols; ++j)
        ok = check(close_enough(sidecar_decode_output[j], direct_decode_expected[j]), "sidecar int64 decode output") && ok;
    return ok;
}

bool test_output_strides() {
    const std::vector<int8_t> weights = {
        3, -2, 5,
        -1, 4, 2,
    };
    std::vector<float> output(18, -99.0f);
    for (size_t offset : { size_t {0}, size_t {2}, size_t {4}, size_t {9}, size_t {11}, size_t {13} })
        output[offset] = 1.0f;
    ggml_gemmini_args_t args = dense_args(2, 3, 2, weights, output, 0.5f);
    args.stride_f_out = 9;
    args.col_stride_f_out = 2;
    const std::vector<ggml::gemmini::quants::QactOutlier> outliers = {
        { 0, 0, 2 }, { 1, 1, -4 },
    };
    ggml::gemmini::quants::dec::compensate_activation_dec(outliers, args, "test");

    const std::vector<float> expected = { 4.0f, -1.0f, 6.0f, 3.0f, -7.0f, -3.0f };
    const std::vector<size_t> offsets = { 0, 2, 4, 9, 11, 13 };
    bool ok = true;
    for (size_t index = 0; index < offsets.size(); ++index)
        ok = check(close_enough(output[offsets[index]], expected[index]), "strided output value") && ok;
    for (size_t index = 0; index < output.size(); ++index) {
        bool used = false;
        for (size_t offset : offsets)
            used = used || index == offset;
        if (!used)
            ok = check(output[index] == -99.0f, "strided output padding") && ok;
    }
    return ok;
}

bool test_malformed_reject() {
    const std::vector<int8_t> weights = { 1, 2, 3, 4, 5, 6 };
    const std::vector<ggml::gemmini::quants::QactOutlier> outliers = { { 0, 0, 3 } };
    std::vector<float> output = { 2.0f, 2.0f };
    ggml_gemmini_args_t args = dense_args(1, 2, 3, weights, output, 1.0f);
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h0;
    const auto h0_result = ggml::gemmini::quants::dec::compensate_activation_dec(outliers, args, "test");
    const bool h0_ok = check(h0_result.total_selected == 0 && output == std::vector<float>({ 2.0f, 2.0f }),
                             "q8_h0 rejects without output mutation");

    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_0_unpacked_to_h1;
    args.sB = 1;
    const auto stride_result = ggml::gemmini::quants::dec::compensate_activation_dec(outliers, args, "test");
    return h0_ok && check(stride_result.total_selected == 0 && output == std::vector<float>({ 2.0f, 2.0f }),
                          "short weight stride rejects without output mutation");
}

void set_dec_threads(const char *value) {
#if defined(_WIN32)
    _putenv_s("DEC_THREADS", value ? value : "");
#else
    if (value)
        setenv("DEC_THREADS", value, 1);
    else
        unsetenv("DEC_THREADS");
#endif
}

bool test_thread_clamp() {
    const char *previous = std::getenv("DEC_THREADS");
    const std::string saved = previous ? previous : "";
    const bool had_previous = previous != nullptr;

    set_dec_threads("999999999999999999999999");
    bool ok = check(ggml::gemmini::quants::dec::resolve_dec_threads(2, 3) == 2,
                    "thread clamp rejects overflowing request");
    set_dec_threads("9");
    ok = check(ggml::gemmini::quants::dec::resolve_dec_threads(2, 3) == 2,
               "thread clamp honors task count") && ok;
    set_dec_threads("1");
    ok = check(ggml::gemmini::quants::dec::resolve_dec_threads(2, 3) == 1,
               "thread clamp honors valid request") && ok;
    set_dec_threads(had_previous ? saved.c_str() : nullptr);
    return ok;
}

}

int main() {
    const bool ok = test_noop() && test_route_plan() && test_repeated_residuals() && test_decode_repeated_residuals() && test_integer_routes() && test_output_strides() &&
        test_malformed_reject() && test_thread_clamp();
    std::printf("gemmini DEC baseline: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
