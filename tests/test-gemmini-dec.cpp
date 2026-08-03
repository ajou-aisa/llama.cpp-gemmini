#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/common/dequant.hpp"
#include "../ggml/src/ggml-gemmini/quants/dec/dec.hpp"
#include "../ggml/src/ggml-gemmini/quants/dec/dec_internal.hpp"
#include "../ggml/src/ggml-gemmini/quants/dec/dec_kernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#if defined(GGML_GEMMINI_HAS_OPENMP)
#include <omp.h>
#endif

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

    const std::vector<int8_t> h0_weights(10, 1);
    const std::vector<float> h0_scales = { 0.5f, 0.25f, 1.0f, 2.0f };
    std::vector<float> h0_output(2, 0.0f);
    ggml_gemmini_args_t h0_args = dense_args(1, 2, 5, h0_weights, h0_output, 1.0f);
    h0_args.weight_i8_scale_active = false;
    h0_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h0;
    h0_args.B_scales = h0_scales.data();
    h0_args.blocks_J = 2;
    h0_args.blocks_K = 2;
    h0_args.block_size_k = 3;
    const auto h0_common_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        h0_args,
        ggml::gemmini::quants::dec::WeightScaleInfoMode::CommonOutput);
    const std::array<int32_t, 2> h0_accumulator = { 4, -2 };
    ggml::gemmini::dequantize(h0_args, 3, 2, h0_accumulator.data(), h0_accumulator.size());
    ok = check(h0_common_plan.valid &&
                   h0_common_plan.route == ggml::gemmini::quants::dec::DecWeightRoute::Dense &&
                   close_enough(h0_output[0], 1.0f) && close_enough(h0_output[1], -4.0f),
               "q8_h0 common-output partial block") && ok;

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

bool test_active_row_groups() {
    using ggml::gemmini::quants::dec::ActiveRowGroup;
    using ggml::gemmini::quants::dec::ResidualGroupEntry;

    std::vector<ResidualGroupEntry> entries = {
        { 1, 35, 4 }, { 0, 31, 2 }, { 1, 32, -3 },
        { 0, 2, 7 }, { 1, 35, -1 }, { 0, 33, 5 },
    };
    std::vector<ActiveRowGroup> groups;
    ggml::gemmini::quants::dec::build_active_row_groups(entries, groups);

    bool ok = check(groups.size() == 3, "one descriptor per active row-group");
    const std::array<std::pair<uint32_t, uint32_t>, 3> expected_groups = {
        std::pair<uint32_t, uint32_t>{ 0, 0 }, { 0, 1 }, { 1, 1 },
    };
    ok = check(groups[0].row == expected_groups[0].first && groups[0].k_group == expected_groups[0].second,
               "first active row-group") && ok;
    ok = check(groups[1].row == expected_groups[1].first && groups[1].k_group == expected_groups[1].second,
               "second active row-group") && ok;
    ok = check(groups[2].row == expected_groups[2].first && groups[2].k_group == expected_groups[2].second,
               "third active row-group") && ok;
    for (const ActiveRowGroup &group : groups)
        ok = check(group.entry_begin < group.entry_end && group.entry_end <= entries.size(),
                   "active row-group entry range") && ok;

    const auto ordered_entries = entries;
    std::reverse(entries.begin(), entries.end());
    ggml::gemmini::quants::dec::build_active_row_groups(entries, groups);
    bool same_entries = entries.size() == ordered_entries.size();
    for (size_t i = 0; same_entries && i < entries.size(); ++i)
        same_entries = entries[i].row == ordered_entries[i].row &&
            entries[i].k == ordered_entries[i].k && entries[i].residual == ordered_entries[i].residual;
    ok = check(same_entries, "active row-group ordering is deterministic") && ok;

    entries.clear();
    ggml::gemmini::quants::dec::build_active_row_groups(entries, groups);
    return check(groups.empty(), "empty residual plan has no active row-groups") && ok;
}

bool test_route_metadata_rejects() {
    std::array<block_q8_h1, 2> h1_blocks{};
    std::vector<int8_t> mixed_h1_weights(64, 1);
    std::vector<float> h1_output(1, 3.0f);
    ggml_gemmini_args_t h1_args{};
    h1_args.I = 1;
    h1_args.J = 1;
    h1_args.K = 64;
    h1_args.B = reinterpret_cast<elem_t *>(mixed_h1_weights.data());
    h1_args.sB = 64;
    h1_args.f_out = h1_output.data();
    h1_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    h1_args.q8_h1_blocks = h1_blocks.data();
    h1_args.q8_h1_block_count = h1_blocks.size();
    h1_args.q8_h1_rows = 1;
    h1_args.blocks_per_row = 2;
    h1_args.blocks_K = 2;
    h1_args.block_size_k = QK8_0;
    h1_args.weight_i8_scale_active = true;
    h1_args.weight_scale = 0.5f;
    const auto mixed_h1_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        h1_args,
        ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    const auto mixed_h1_result = ggml::gemmini::quants::dec::compensate_activation_dec(
        { { 0, 33, 7 } }, h1_args, "test");
    bool ok = check(!mixed_h1_plan.valid && mixed_h1_result.total_selected == 0 && h1_output[0] == 3.0f,
                    "q8_h1 rejects scalar scale metadata beyond first block");

    const std::vector<int8_t> weights(10, 1);
    const std::vector<float> surplus_scales(6, 0.25f);
    std::vector<float> output(2, 0.0f);
    ggml_gemmini_args_t surplus_args = dense_args(1, 2, 5, weights, output, 1.0f);
    surplus_args.weight_i8_scale_active = false;
    surplus_args.B_scales = surplus_scales.data();
    surplus_args.blocks_J = 2;
    surplus_args.blocks_K = 3;
    surplus_args.block_size_k = 3;
    const auto surplus_dec_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        surplus_args,
        ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    const auto surplus_common_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        surplus_args,
        ggml::gemmini::quants::dec::WeightScaleInfoMode::CommonOutput);
    size_t partial_block = 0;
    ok = check(!surplus_dec_plan.valid && surplus_common_plan.valid &&
                   ggml::gemmini::quants::dec::dec_route_block_for_range(
                       surplus_common_plan, 3, 2, partial_block) && partial_block == 1,
               "DEC rejects surplus scales while common output keeps partial blocks") && ok;
    return ok;
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

bool test_block_integer_route() {
    const std::vector<int8_t> weights = {
        1, -2, 3, 4, -1, 2, 5, -3, 4, 2,
    };
    const std::vector<float> scales = { 0.5f, 0.25f, -0.5f, 0.75f };
    const std::vector<ggml::gemmini::quants::QactOutlier> outliers = {
        { 0, 0, 3 }, { 0, 4, -2 }, { 1, 1, 5 }, { 1, 1, -1 },
    };
    std::vector<float> output(4, 0.0f);
    ggml_gemmini_args_t args = dense_args(2, 2, 5, weights, output, 1.0f);
    args.weight_i8_scale_active = false;
    args.B_scales = scales.data();
    args.blocks_J = 2;
    args.blocks_K = 2;
    args.block_size_k = 3;
    ggml::gemmini::quants::dec::compensate_activation_dec(outliers, args, "test");
    const std::vector<float> expected = { -0.5f, 0.0f, 6.0f, -8.0f };

    std::vector<float> decode_output(2, 0.0f);
    args.I = 1;
    args.f_out = decode_output.data();
    const std::vector<ggml::gemmini::quants::QactOutlier> decode_outliers = {
        { 0, 0, 3 }, { 0, 4, -2 },
    };
    ggml::gemmini::quants::dec::compensate_activation_dec(decode_outliers, args, "test");
    bool ok = true;
    for (size_t index = 0; index < output.size(); ++index)
        ok = check(close_enough(output[index], expected[index]), "block int64 output") && ok;
    ok = check(close_enough(decode_output[0], -0.5f) && close_enough(decode_output[1], 0.0f),
               "block int64 decode output") && ok;
    return ok;
}

constexpr size_t kHierarchicalRows = 2;
constexpr size_t kHierarchicalColumns = 257;
constexpr size_t kHierarchicalBlocksPerRow = 2;
constexpr size_t kHierarchicalDepth = QK8_0 * kHierarchicalBlocksPerRow;

static_assert(QK8_0 == QK8_H2 && QK8_0 == QK8_HP, "hierarchical block widths must match");

const std::vector<ggml::gemmini::quants::QactOutlier> kHierarchicalPrefillOutliers = {
    { 0, 2, 3 }, { 0, 2, -1 }, { 0, 31, 2 }, { 1, 32, -2 }, { 1, 63, 4 },
};

const std::vector<ggml::gemmini::quants::QactOutlier> kHierarchicalDecodeOutliers = {
    { 0, 1, 4 }, { 0, 1, -3 }, { 0, 33, 2 }, { 0, 62, -5 },
};

ggml_gemmini_args_t hierarchical_args(size_t rows, std::vector<float> &output) {
    ggml_gemmini_args_t args{};
    args.I = rows;
    args.J = kHierarchicalColumns;
    args.K = kHierarchicalDepth;
    args.B = nullptr;
    args.B_blocks = nullptr;
    args.sB = kHierarchicalDepth;
    args.B_scales = nullptr;
    args.weight_channel_scales = nullptr;
    args.weight_channel_scale_count = 0;
    args.weight_i8_scale_active = false;
    args.weight_scale = 1.0f;
    args.blocks_per_row = kHierarchicalBlocksPerRow;
    args.blocks_K = kHierarchicalBlocksPerRow;
    args.blocks_J = kHierarchicalColumns;
    args.blocks_I = kHierarchicalColumns;
    args.block_size_k = QK8_0;
    args.f_out = output.data();
    return args;
}

template <typename Block, size_t BlockCount>
void initialize_hierarchical_qs(std::array<Block, BlockCount> &blocks) {
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        for (size_t offset = 0; offset < sizeof(blocks[block_index].qs); ++offset) {
            blocks[block_index].qs[offset] = static_cast<int8_t>(
                static_cast<int>((offset * 5 + block_index * 3) % 15) - 7);
        }
    }
}

template <typename Block, size_t BlockCount, typename ScaleForBlock>
std::vector<float> blockwise_expected(
    const std::array<Block, BlockCount> &blocks,
    size_t rows,
    size_t block_size,
    const std::vector<ggml::gemmini::quants::QactOutlier> &outliers,
    ScaleForBlock scale_for_block) {
    std::vector<float> expected(rows * kHierarchicalColumns, 0.0f);
    for (size_t block_index = 0; block_index < kHierarchicalBlocksPerRow; ++block_index) {
        std::array<int64_t, kHierarchicalRows * kHierarchicalColumns> accum{};
        for (const auto &outlier : outliers) {
            if (outlier.row < 0 || outlier.col < 0 ||
                static_cast<size_t>(outlier.row) >= rows ||
                static_cast<size_t>(outlier.col) >= kHierarchicalDepth ||
                static_cast<size_t>(outlier.col) / block_size != block_index) {
                continue;
            }
            for (size_t column = 0; column < kHierarchicalColumns; ++column) {
                const Block &block = blocks[column * kHierarchicalBlocksPerRow + block_index];
                accum[static_cast<size_t>(outlier.row) * kHierarchicalColumns + column] +=
                    static_cast<int64_t>(outlier.residual) *
                    block.qs[static_cast<size_t>(outlier.col) % block_size];
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < kHierarchicalColumns; ++column) {
                const Block &block = blocks[column * kHierarchicalBlocksPerRow + block_index];
                expected[row * kHierarchicalColumns + column] += static_cast<float>(
                    static_cast<double>(accum[row * kHierarchicalColumns + column]) *
                    scale_for_block(block));
            }
        }
    }
    return expected;
}

std::vector<float> h1_expected(
    const std::array<block_q8_h1, kHierarchicalColumns * kHierarchicalBlocksPerRow> &blocks,
    size_t rows,
    const std::vector<ggml::gemmini::quants::QactOutlier> &outliers) {
    std::vector<float> expected(rows * kHierarchicalColumns, 0.0f);
    for (size_t block_index = 0; block_index < kHierarchicalBlocksPerRow; ++block_index) {
        std::array<int64_t, kHierarchicalRows * kHierarchicalColumns> accum{};
        for (const auto &outlier : outliers) {
            if (outlier.row < 0 || outlier.col < 0 ||
                static_cast<size_t>(outlier.row) >= rows ||
                static_cast<size_t>(outlier.col) >= kHierarchicalDepth ||
                static_cast<size_t>(outlier.col) / QK8_0 != block_index) {
                continue;
            }
            for (size_t column = 0; column < kHierarchicalColumns; ++column) {
                const block_q8_h1 &block = blocks[column * kHierarchicalBlocksPerRow + block_index];
                accum[static_cast<size_t>(outlier.row) * kHierarchicalColumns + column] +=
                    static_cast<int64_t>(outlier.residual) *
                    block.qs[static_cast<size_t>(outlier.col) % QK8_0];
            }
        }
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < kHierarchicalColumns; ++column) {
                const block_q8_h1 &block = blocks[column * kHierarchicalBlocksPerRow + block_index];
                const uint64_t c_eff = static_cast<uint64_t>(block.c_b) + block.R;
                expected[row * kHierarchicalColumns + column] += static_cast<float>(
                    static_cast<double>(accum[row * kHierarchicalColumns + column]) * c_eff * block.s_rf);
            }
        }
    }
    return expected;
}

float h2_expected_scale(const block_q8_h2 &block) {
    return block.channel_scale * static_cast<float>(block.m) / 255.0f;
}

template <typename Block>
float hp_expected_scale(const Block &block) {
    return block.m == std::numeric_limits<int16_t>::min() ? 0.0f :
        std::ldexp(block.channel_scale, static_cast<int>(block.m));
}

bool run_hierarchical_case(
    const char *expected_route,
    ggml_gemmini_args_t &args,
    const std::vector<ggml::gemmini::quants::QactOutlier> &outliers,
    const std::vector<float> &expected) {
    const auto plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        args,
        ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    std::printf("gemmini DEC synthetic: weight_route=%s I=%zu\n",
                ggml::gemmini::quants::dec::dec_route_name(plan), args.I);
    const auto result = ggml::gemmini::quants::dec::compensate_activation_dec(outliers, args, "test");

    bool ok = check(plan.valid && plan.native_weight_blocks &&
                        std::string(ggml::gemmini::quants::dec::dec_route_name(plan)) == expected_route,
                    "hierarchical route plan") &&
        check(result.total_selected == outliers.size() && result.nnz == outliers.size(),
              "hierarchical route result");
    for (size_t index = 0; index < expected.size(); ++index)
        ok = check(close_enough(args.f_out[index], expected[index]), "hierarchical route output") && ok;
    return ok;
}

bool test_q8_h1_hierarchical_route() {
    std::array<block_q8_h1, kHierarchicalColumns * kHierarchicalBlocksPerRow> blocks{};
    initialize_hierarchical_qs(blocks);
    const std::array<uint8_t, 4> c_b = { 3, 5, 2, 9 };
    const std::array<float, 4> s_rf = { 0.125f, 0.25f, 0.0625f, 0.5f };
    const std::array<uint16_t, 4> R = { 7, 11, 13, 1 };
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        blocks[block_index].c_b = c_b[block_index % c_b.size()];
        blocks[block_index].s_rf = s_rf[block_index % s_rf.size()];
        blocks[block_index].R = R[block_index % R.size()];
    }

    std::vector<float> prefill_output(kHierarchicalRows * kHierarchicalColumns, 0.0f);
    ggml_gemmini_args_t args = hierarchical_args(kHierarchicalRows, prefill_output);
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = blocks.data();
    args.q8_h1_block_count = blocks.size();
    args.q8_h1_rows = kHierarchicalColumns;
    bool ok = run_hierarchical_case(
        "q8-h1", args, kHierarchicalPrefillOutliers,
        h1_expected(blocks, kHierarchicalRows, kHierarchicalPrefillOutliers));

    std::vector<float> decode_output(kHierarchicalColumns, 0.0f);
    args.I = 1;
    args.f_out = decode_output.data();
    ok = run_hierarchical_case(
             "q8-h1", args, kHierarchicalDecodeOutliers,
             h1_expected(blocks, 1, kHierarchicalDecodeOutliers)) && ok;
    return ok;
}

bool test_q8_h1_large_effective_scale() {
    constexpr size_t repeats = 520;
    std::array<block_q8_h1, kHierarchicalColumns * kHierarchicalBlocksPerRow> blocks{};
    for (block_q8_h1 &block : blocks) {
        for (int8_t &code : block.qs)
            code = 127;
        block.c_b = 1;
        block.R = std::numeric_limits<uint16_t>::max();
        block.s_rf = 1.0f / 65536.0f;
    }

    std::vector<ggml::gemmini::quants::QactOutlier> prefill_outliers;
    prefill_outliers.reserve(repeats * 2);
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        prefill_outliers.push_back({ 0, 1, std::numeric_limits<int32_t>::max() });
        prefill_outliers.push_back({ 1, 33, -std::numeric_limits<int32_t>::max() });
    }

    std::vector<float> prefill_output(kHierarchicalRows * kHierarchicalColumns, 0.0f);
    ggml_gemmini_args_t args = hierarchical_args(kHierarchicalRows, prefill_output);
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = blocks.data();
    args.q8_h1_block_count = blocks.size();
    args.q8_h1_rows = kHierarchicalColumns;
    bool ok = run_hierarchical_case(
        "q8-h1", args, prefill_outliers,
        h1_expected(blocks, kHierarchicalRows, prefill_outliers));

    std::vector<ggml::gemmini::quants::QactOutlier> decode_outliers(
        repeats, { 0, 33, std::numeric_limits<int32_t>::max() });
    std::vector<float> decode_output(kHierarchicalColumns, 0.0f);
    args.I = 1;
    args.f_out = decode_output.data();
    return run_hierarchical_case(
               "q8-h1", args, decode_outliers,
               h1_expected(blocks, 1, decode_outliers)) && ok;
}

bool test_q8_h2_hierarchical_route() {
    std::array<block_q8_h2, kHierarchicalColumns * kHierarchicalBlocksPerRow> blocks{};
    initialize_hierarchical_qs(blocks);
    const std::array<uint8_t, 4> m = { 17, 31, 43, 57 };
    const std::array<float, 4> channel_scales = { 0.125f, 0.25f, 0.0625f, 0.375f };
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        blocks[block_index].m = m[block_index % m.size()];
        blocks[block_index].channel_scale = channel_scales[block_index % channel_scales.size()];
    }

    std::vector<float> prefill_output(kHierarchicalRows * kHierarchicalColumns, 0.0f);
    ggml_gemmini_args_t args = hierarchical_args(kHierarchicalRows, prefill_output);
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h2;
    args.q8_h2_blocks = blocks.data();
    args.q8_h2_block_count = blocks.size();
    args.q8_h2_blocks_per_row = kHierarchicalBlocksPerRow;
    bool ok = run_hierarchical_case(
        "q8-h2", args, kHierarchicalPrefillOutliers,
        blockwise_expected(blocks, kHierarchicalRows, QK8_H2, kHierarchicalPrefillOutliers,
                           h2_expected_scale));

    std::vector<float> decode_output(kHierarchicalColumns, 0.0f);
    args.I = 1;
    args.f_out = decode_output.data();
    ok = run_hierarchical_case(
             "q8-h2", args, kHierarchicalDecodeOutliers,
             blockwise_expected(blocks, 1, QK8_H2, kHierarchicalDecodeOutliers,
                                h2_expected_scale)) && ok;
    return ok;
}

bool test_q8_hp1_hierarchical_route() {
    std::array<block_q8_hp1, kHierarchicalColumns * kHierarchicalBlocksPerRow> blocks{};
    initialize_hierarchical_qs(blocks);
    const std::array<int16_t, 4> m = { 1, -2, 3, std::numeric_limits<int16_t>::min() };
    const std::array<float, 4> channel_scales = { 0.125f, 0.5f, 0.25f, 0.75f };
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        blocks[block_index].m = m[block_index % m.size()];
        blocks[block_index].padding[0] = 0;
        blocks[block_index].padding[1] = 0;
        blocks[block_index].channel_scale = channel_scales[block_index % channel_scales.size()];
    }

    std::vector<float> prefill_output(kHierarchicalRows * kHierarchicalColumns, 0.0f);
    ggml_gemmini_args_t args = hierarchical_args(kHierarchicalRows, prefill_output);
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
    args.q8_hp1_blocks = blocks.data();
    args.q8_hp1_block_count = blocks.size();
    args.q8_hp1_blocks_per_row = kHierarchicalBlocksPerRow;
    bool ok = run_hierarchical_case(
        "q8-hp1", args, kHierarchicalPrefillOutliers,
        blockwise_expected(blocks, kHierarchicalRows, QK8_HP, kHierarchicalPrefillOutliers,
                           hp_expected_scale<block_q8_hp1>));

    std::vector<float> decode_output(kHierarchicalColumns, 0.0f);
    args.I = 1;
    args.f_out = decode_output.data();
    ok = run_hierarchical_case(
             "q8-hp1", args, kHierarchicalDecodeOutliers,
             blockwise_expected(blocks, 1, QK8_HP, kHierarchicalDecodeOutliers,
                                hp_expected_scale<block_q8_hp1>)) && ok;
    return ok;
}

bool test_q8_hp2_hierarchical_route() {
    std::array<block_q8_hp2, kHierarchicalColumns * kHierarchicalBlocksPerRow> blocks{};
    initialize_hierarchical_qs(blocks);
    const std::array<int16_t, 4> m = { 2, -1, -3, std::numeric_limits<int16_t>::min() };
    const std::array<float, 4> channel_scales = { 0.0625f, 0.5f, 0.25f, 0.625f };
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        blocks[block_index].m = m[block_index % m.size()];
        blocks[block_index].padding[0] = 0;
        blocks[block_index].padding[1] = 0;
        blocks[block_index].channel_scale = channel_scales[block_index % channel_scales.size()];
    }

    std::vector<float> prefill_output(kHierarchicalRows * kHierarchicalColumns, 0.0f);
    ggml_gemmini_args_t args = hierarchical_args(kHierarchicalRows, prefill_output);
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2;
    args.q8_hp2_blocks = blocks.data();
    args.q8_hp2_block_count = blocks.size();
    args.q8_hp2_blocks_per_row = kHierarchicalBlocksPerRow;
    bool ok = run_hierarchical_case(
        "q8-hp2", args, kHierarchicalPrefillOutliers,
        blockwise_expected(blocks, kHierarchicalRows, QK8_HP, kHierarchicalPrefillOutliers,
                           hp_expected_scale<block_q8_hp2>));

    std::vector<float> decode_output(kHierarchicalColumns, 0.0f);
    args.I = 1;
    args.f_out = decode_output.data();
    ok = run_hierarchical_case(
             "q8-hp2", args, kHierarchicalDecodeOutliers,
             blockwise_expected(blocks, 1, QK8_HP, kHierarchicalDecodeOutliers,
                                hp_expected_scale<block_q8_hp2>)) && ok;
    return ok;
}

bool rejects_hierarchical_contract(
    ggml_gemmini_args_t &args,
    std::vector<float> &output,
    const char *message) {
    const std::vector<float> before = output;
    const auto plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        args,
        ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    const auto result = ggml::gemmini::quants::dec::compensate_activation_dec(
        kHierarchicalDecodeOutliers, args, "test");
    return check(!plan.valid && result.total_selected == 0 && output == before, message);
}

bool test_malformed_hierarchical_reject() {
    std::vector<float> output(kHierarchicalColumns, 3.0f);
    std::array<block_q8_h1, kHierarchicalColumns * kHierarchicalBlocksPerRow> h1_blocks{};
    std::array<block_q8_h2, kHierarchicalColumns * kHierarchicalBlocksPerRow> h2_blocks{};
    std::array<block_q8_hp1, kHierarchicalColumns * kHierarchicalBlocksPerRow> hp1_blocks{};
    std::array<block_q8_hp2, kHierarchicalColumns * kHierarchicalBlocksPerRow> hp2_blocks{};

    ggml_gemmini_args_t h1_args = hierarchical_args(1, output);
    h1_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    h1_args.q8_h1_blocks = h1_blocks.data();
    h1_args.q8_h1_block_count = h1_blocks.size() - 1;
    h1_args.q8_h1_rows = kHierarchicalColumns;
    bool ok = rejects_hierarchical_contract(h1_args, output, "malformed q8_h1 rejects");

    ggml_gemmini_args_t h2_args = hierarchical_args(1, output);
    h2_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h2;
    h2_args.q8_h2_blocks = h2_blocks.data();
    h2_args.q8_h2_block_count = h2_blocks.size();
    h2_args.q8_h2_blocks_per_row = 1;
    ok = rejects_hierarchical_contract(h2_args, output, "malformed q8_h2 rejects") && ok;

    hp1_blocks[0].padding[0] = 1;
    ggml_gemmini_args_t hp1_args = hierarchical_args(1, output);
    hp1_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
    hp1_args.q8_hp1_blocks = hp1_blocks.data();
    hp1_args.q8_hp1_block_count = hp1_blocks.size();
    hp1_args.q8_hp1_blocks_per_row = kHierarchicalBlocksPerRow;
    ok = rejects_hierarchical_contract(hp1_args, output, "malformed q8_hp1 rejects") && ok;

    hp2_blocks[0].padding[1] = 1;
    ggml_gemmini_args_t hp2_args = hierarchical_args(1, output);
    hp2_args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2;
    hp2_args.q8_hp2_blocks = hp2_blocks.data();
    hp2_args.q8_hp2_block_count = hp2_blocks.size();
    hp2_args.q8_hp2_blocks_per_row = kHierarchicalBlocksPerRow;
    return rejects_hierarchical_contract(hp2_args, output, "malformed q8_hp2 rejects") && ok;
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

std::vector<float> h1_thread_case(
    size_t rows,
    const std::vector<ggml::gemmini::quants::QactOutlier> &outliers) {
    std::array<block_q8_h1, kHierarchicalColumns * kHierarchicalBlocksPerRow> blocks{};
    initialize_hierarchical_qs(blocks);
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        blocks[block_index].c_b = static_cast<uint8_t>(3 + block_index % 11);
        blocks[block_index].R = static_cast<uint16_t>(7 + block_index % 17);
        blocks[block_index].s_rf = 0.0625f * static_cast<float>(1 + block_index % 5);
    }

    std::vector<float> output(rows * kHierarchicalColumns, 0.0f);
    ggml_gemmini_args_t args = hierarchical_args(rows, output);
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = blocks.data();
    args.q8_h1_block_count = blocks.size();
    args.q8_h1_rows = kHierarchicalColumns;
    ggml::gemmini::quants::dec::compensate_activation_dec(outliers, args, "test");
    return output;
}

bool byte_identical(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    return lhs.size() == rhs.size() &&
        std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0;
}

bool test_thread_determinism() {
    const char *previous = std::getenv("DEC_THREADS");
    const std::string saved = previous ? previous : "";
    const bool had_previous = previous != nullptr;

    set_dec_threads("1");
    const std::vector<float> prefill_reference = h1_thread_case(
        kHierarchicalRows, kHierarchicalPrefillOutliers);
    const std::vector<float> decode_reference = h1_thread_case(
        1, kHierarchicalDecodeOutliers);

    bool ok = true;
    for (const char *thread_count : { "2", "3", "4", "99" }) {
        set_dec_threads(thread_count);
        ok = check(byte_identical(
                       h1_thread_case(kHierarchicalRows, kHierarchicalPrefillOutliers),
                       prefill_reference),
                   "INT64 H1 prefill is byte-identical across DEC_THREADS") && ok;
        ok = check(byte_identical(
                       h1_thread_case(1, kHierarchicalDecodeOutliers),
                       decode_reference),
                   "INT64 H1 decode is byte-identical across DEC_THREADS") && ok;
    }

    set_dec_threads(had_previous ? saved.c_str() : nullptr);
    return ok;
}

bool test_inside_existing_openmp_region() {
#if defined(GGML_GEMMINI_HAS_OPENMP)
    const char *previous = std::getenv("DEC_THREADS");
    const std::string saved = previous ? previous : "";
    const bool had_previous = previous != nullptr;

    set_dec_threads("1");
    const std::vector<float> expected = h1_thread_case(
        kHierarchicalRows, kHierarchicalPrefillOutliers);
    set_dec_threads("2");
    std::vector<float> nested_output;
    bool entered_parallel_region = false;
#pragma omp parallel num_threads(2) shared(nested_output, entered_parallel_region)
    {
#pragma omp single
        {
            entered_parallel_region = omp_in_parallel() != 0;
            nested_output = h1_thread_case(kHierarchicalRows, kHierarchicalPrefillOutliers);
        }
    }

    set_dec_threads(had_previous ? saved.c_str() : nullptr);
    return check(entered_parallel_region && byte_identical(nested_output, expected),
                 "DEC invocation inside existing OpenMP region remains serial and correct");
#else
    return true;
#endif
}

}

int main() {
    const bool ok = test_noop() && test_route_plan() && test_active_row_groups() && test_route_metadata_rejects() && test_repeated_residuals() && test_decode_repeated_residuals() && test_integer_routes() && test_block_integer_route() &&
        test_q8_h1_hierarchical_route() && test_q8_h1_large_effective_scale() && test_q8_h2_hierarchical_route() && test_q8_hp1_hierarchical_route() && test_q8_hp2_hierarchical_route() &&
        test_malformed_hierarchical_reject() && test_output_strides() && test_malformed_reject() && test_thread_clamp() &&
        test_thread_determinism() && test_inside_existing_openmp_region();
    std::printf("gemmini DEC baseline: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
