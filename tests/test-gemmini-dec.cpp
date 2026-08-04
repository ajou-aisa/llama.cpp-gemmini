#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/common/dequant.hpp"
#include "../ggml/src/ggml-gemmini/quants/dec/dec.hpp"
#include "../ggml/src/ggml-gemmini/quants/dec/dec_internal.hpp"
#include "../ggml/src/ggml-gemmini/quants/dec/dec_kernel.hpp"

#include <algorithm>
#include <array>
#include <chrono>
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

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
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

    std::vector<size_t> group_offsets;
    std::vector<size_t> group_row_group_indices;
    ggml::gemmini::quants::dec::build_group_major_index(
        groups, 2, group_offsets, group_row_group_indices);

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
    ok = check(group_offsets == std::vector<size_t>({ 0, 1, 3 }) &&
                   group_row_group_indices == std::vector<size_t>({ 0, 1, 2 }),
               "group-major index") && ok;
    for (const ActiveRowGroup &group : groups)
        ok = check(group.entry_begin < group.entry_end && group.entry_end <= entries.size(),
                   "active row-group entry range") && ok;

    const auto ordered_entries = entries;
    std::reverse(entries.begin(), entries.end());
    ggml::gemmini::quants::dec::build_active_row_groups(entries, groups);
    std::vector<size_t> reordered_group_offsets;
    std::vector<size_t> reordered_group_row_group_indices;
    ggml::gemmini::quants::dec::build_group_major_index(
        groups, 2, reordered_group_offsets, reordered_group_row_group_indices);
    bool same_entries = entries.size() == ordered_entries.size();
    for (size_t i = 0; same_entries && i < entries.size(); ++i)
        same_entries = entries[i].row == ordered_entries[i].row &&
            entries[i].k == ordered_entries[i].k && entries[i].residual == ordered_entries[i].residual;
    ok = check(same_entries, "active row-group ordering is deterministic") && ok;
    ok = check(reordered_group_offsets == group_offsets &&
                   reordered_group_row_group_indices == group_row_group_indices,
               "group-major index is deterministic") && ok;

    entries.clear();
    ggml::gemmini::quants::dec::build_active_row_groups(entries, groups);
    ggml::gemmini::quants::dec::build_group_major_index(
        groups, 0, group_offsets, group_row_group_indices);
    return check(groups.empty() && group_offsets == std::vector<size_t>({ 0 }) &&
                     group_row_group_indices.empty(),
                 "empty residual plan has no active row-groups") && ok;
}

bool test_group_k_csc_plan() {
    using ggml::gemmini::quants::dec::ActiveRowGroup;
    using ggml::gemmini::quants::dec::GroupKCSCPlan;
    using ggml::gemmini::quants::dec::ResidualGroupEntry;

    std::vector<ResidualGroupEntry> entries = {
        { 1, 35, 4 }, { 0, 33, 5 }, { 1, 2, 9 }, { 0, 31, 2 },
        { 1, 32, -3 }, { 0, 2, 7 }, { 1, 35, -1 }, { 2, 64, 8 },
        { 1, 64, 6 }, { 1, 33, -2 },
    };
    const auto shuffled_entries = entries;
    std::vector<ActiveRowGroup> groups;
    std::vector<size_t> group_offsets;
    std::vector<size_t> group_row_group_indices;
    ggml::gemmini::quants::dec::build_active_row_groups(entries, groups);
    ggml::gemmini::quants::dec::build_group_major_index(
        groups, 3, group_offsets, group_row_group_indices);

    GroupKCSCPlan plan;
    bool ok = check(ggml::gemmini::quants::dec::build_group_k_csc_plan(
                        entries, groups, group_offsets, group_row_group_indices, 3, plan),
                    "build group-K CSC plan");
    const std::vector<uint32_t> expected_entry_order = { 0, 3, 1, 4, 2, 5, 6, 7, 8, 9 };
    const std::vector<uint32_t> expected_active_row_offsets = { 0, 2, 4, 6 };
    const std::vector<uint32_t> expected_active_rows = { 0, 1, 0, 1, 1, 2 };
    ok = check(plan.group_size_k == 32 && plan.num_groups == 3 &&
                   plan.column_offsets.size() == 3 * 33,
               "group-K CSC dimensions and K tail") && ok;
    ok = check(plan.entry_order == expected_entry_order,
               "group-K CSC stable per-column entry order") && ok;
    ok = check(plan.active_row_offsets == expected_active_row_offsets &&
                   plan.active_rows == expected_active_rows,
               "group-K CSC active rows") && ok;
    ok = check(plan.column_offsets[2] == 0 && plan.column_offsets[3] == 2 &&
                   plan.column_offsets[31] == 2 && plan.column_offsets[32] == 3 &&
                   plan.column_offsets[33] == 3 && plan.column_offsets[34] == 4 &&
                   plan.column_offsets[35] == 6 && plan.column_offsets[37] == 8 &&
                   plan.column_offsets[65] == 8 && plan.column_offsets[66] == 8 &&
                   plan.column_offsets[67] == 10 && plan.column_offsets[98] == 10,
               "group-K CSC column offsets") && ok;
    const size_t expected_plan_bytes =
        (3 * 33 + entries.size() + 4 + 6) * sizeof(uint32_t);
    ok = check(ggml::gemmini::quants::dec::group_k_csc_plan_logical_bytes(plan) ==
                   expected_plan_bytes,
               "group-K CSC logical bytes exclude fill cursors") && ok;

    const auto canonical_entries = entries;
    auto reordered_entries = shuffled_entries;
    std::reverse(reordered_entries.begin(), reordered_entries.end());
    std::vector<ActiveRowGroup> reordered_groups;
    std::vector<size_t> reordered_group_offsets;
    std::vector<size_t> reordered_group_row_group_indices;
    ggml::gemmini::quants::dec::build_active_row_groups(reordered_entries, reordered_groups);
    ggml::gemmini::quants::dec::build_group_major_index(
        reordered_groups, 3, reordered_group_offsets, reordered_group_row_group_indices);
    GroupKCSCPlan reordered_plan;
    ok = check(ggml::gemmini::quants::dec::build_group_k_csc_plan(
                        reordered_entries, reordered_groups, reordered_group_offsets,
                        reordered_group_row_group_indices, 3, reordered_plan),
                    "build shuffled group-K CSC plan") && ok;
    bool same_entries = canonical_entries.size() == reordered_entries.size();
    for (size_t index = 0; same_entries && index < canonical_entries.size(); ++index)
        same_entries = canonical_entries[index].row == reordered_entries[index].row &&
            canonical_entries[index].k == reordered_entries[index].k &&
            canonical_entries[index].residual == reordered_entries[index].residual;
    ok = check(same_entries,
               "group-K CSC canonicalizes shuffled entries") && ok;
    ok = check(reordered_plan.column_offsets == plan.column_offsets &&
                   reordered_plan.entry_order == plan.entry_order &&
                   reordered_plan.active_row_offsets == plan.active_row_offsets &&
                   reordered_plan.active_rows == plan.active_rows &&
                   ggml::gemmini::quants::dec::group_k_csc_plan_logical_bytes(reordered_plan) ==
                       expected_plan_bytes,
               "group-K CSC plan and bytes are deterministic") && ok;

    const size_t column_capacity = plan.column_offsets.capacity();
    const size_t entry_capacity = plan.entry_order.capacity();
    const size_t active_row_capacity = plan.active_rows.capacity();
    const size_t cursor_capacity = plan.fill_cursors.capacity();
    std::vector<ResidualGroupEntry> empty_entries;
    std::vector<ActiveRowGroup> empty_groups;
    std::vector<size_t> empty_group_offsets;
    std::vector<size_t> empty_group_row_group_indices;
    ggml::gemmini::quants::dec::build_group_major_index(
        empty_groups, 0, empty_group_offsets, empty_group_row_group_indices);
    ok = check(ggml::gemmini::quants::dec::build_group_k_csc_plan(
                        empty_entries, empty_groups, empty_group_offsets,
                        empty_group_row_group_indices, 0, plan),
                    "build empty group-K CSC plan") && ok;
    return check(plan.num_groups == 0 && plan.column_offsets.empty() && plan.entry_order.empty() &&
                     plan.active_row_offsets == std::vector<uint32_t>({ 0 }) &&
                     plan.active_rows.empty() &&
                     ggml::gemmini::quants::dec::group_k_csc_plan_logical_bytes(plan) ==
                         sizeof(uint32_t) &&
                     plan.column_offsets.capacity() >= column_capacity &&
                     plan.entry_order.capacity() >= entry_capacity &&
                     plan.active_rows.capacity() >= active_row_capacity &&
                     plan.fill_cursors.capacity() >= cursor_capacity,
                 "empty group-K CSC plan retains scratch capacity") && ok;
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

    std::printf(
        "dispatch decode route=%s plan_bytes=%zu vector_loads=%zu\n",
        result.group_k_csc_plan_bytes == 0 ? "row-direct" : "group-k-csc",
        result.group_k_csc_plan_bytes,
        result.weight_vector_load_count);
    bool ok = check(result.total_selected == 3 && result.nnz == 3 && result.unique_k_count == 2 &&
                        result.group_k_csc_plan_bytes == 0 && result.weight_vector_load_count == 0,
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
    { 0, 2, 3 }, { 0, 2, -1 }, { 0, 31, 2 }, { 1, 30, 1 }, { 1, 32, -2 }, { 1, 63, 4 },
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
    if (plan.route == ggml::gemmini::quants::dec::DecWeightRoute::Q8H1 ||
        plan.route == ggml::gemmini::quants::dec::DecWeightRoute::Q8HP1) {
        ok = check(plan.scales.on_demand_mode && plan.scales.data == nullptr,
                   "H1/HP1 DEC scales stay tile-local") && ok;
    }
    for (size_t index = 0; index < expected.size(); ++index)
        ok = check(close_enough(args.f_out[index], expected[index]), "hierarchical route output") && ok;

    std::vector<float> reordered_output(args.I * args.J, 0.0f);
    ggml_gemmini_args_t reordered_args = args;
    reordered_args.f_out = reordered_output.data();
    auto reordered_outliers = outliers;
    std::reverse(reordered_outliers.begin(), reordered_outliers.end());
    ggml::gemmini::quants::dec::compensate_activation_dec(
        reordered_outliers, reordered_args, "test");
    ok = check(std::memcmp(args.f_out, reordered_output.data(), reordered_output.size() * sizeof(float)) == 0,
               "hierarchical route residual ordering is byte-identical") && ok;
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

bool test_q8_h1_preserves_ordered_scaling_bits() {
    constexpr int32_t residual = 548339296;
    constexpr uint8_t c_b = 86;
    constexpr uint16_t R = 4095;
    constexpr uint64_t c_eff = static_cast<uint64_t>(c_b) + R;
    constexpr float s_rf = 0.00032747327350080013f;

    block_q8_h1 block{};
    block.qs[0] = 1;
    block.c_b = c_b;
    block.R = R;
    block.s_rf = s_rf;

    std::vector<float> output(1, 0.0f);
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = QK8_0;
    args.f_out = output.data();
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = &block;
    args.q8_h1_block_count = 1;
    args.q8_h1_rows = 1;
    args.blocks_per_row = 1;
    args.blocks_K = 1;
    args.blocks_J = 1;
    args.block_size_k = QK8_0;

    ggml::gemmini::quants::dec::compensate_activation_dec(
        { { 0, 0, residual } }, args, "test");

    const float baseline = static_cast<float>(
        static_cast<double>(residual) * c_eff * s_rf * 1.0f);
    const float combined = static_cast<float>(
        static_cast<double>(residual) * 1.0f *
        static_cast<float>(static_cast<double>(c_eff) * s_rf));
    const float ordered = ggml::gemmini::quants::dec::apply_h1_scale_ordered(
        residual, c_eff, s_rf, 1.0f);
    return check(float_bits(combined) != float_bits(baseline),
                 "H1 ordered-scale fixture distinguishes early FP32 combination") &&
        check(float_bits(ordered) == float_bits(baseline),
              "H1 ordered helper preserves baseline bits") &&
        check(float_bits(output[0]) == float_bits(baseline),
              "H1 grouped DEC preserves ordered scaling bits");
}

bool run_q8_h1_activation_scale_case(act::Meta meta, float activation_scale, const char *message) {
    constexpr int32_t residual = 548339296;
    constexpr uint8_t c_b = 86;
    constexpr uint16_t R = 4095;
    constexpr float s_rf = 0.00032747327350080013f;

    block_q8_h1 block{};
    block.qs[0] = 1;
    block.c_b = c_b;
    block.R = R;
    block.s_rf = s_rf;

    std::vector<float> output(1, 0.0f);
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = QK8_0;
    args.f_out = output.data();
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = &block;
    args.q8_h1_block_count = 1;
    args.q8_h1_rows = 1;
    args.blocks_per_row = 1;
    args.blocks_K = 1;
    args.blocks_J = 1;
    args.block_size_k = QK8_0;
    args.act_quant = std::move(meta);

    ggml::gemmini::quants::dec::compensate_activation_dec(
        { { 0, 0, residual } }, args, "test");
    const uint64_t c_eff = static_cast<uint64_t>(c_b) + R;
    const float expected = static_cast<float>(
        static_cast<double>(residual) * c_eff * s_rf * activation_scale);
    return check(float_bits(output[0]) == float_bits(expected), message);
}

bool test_q8_h1_activation_scale_routes_preserve_bits() {
    bool ok = run_q8_h1_activation_scale_case({}, 1.0f, "H1 default activation scale bits");

    act::Meta tensor_meta;
    tensor_meta.storage().emplace<act::tensor::Meta>().scale = 0.5f;
    ok = run_q8_h1_activation_scale_case(
             std::move(tensor_meta), 0.5f, "H1 tensor activation scale bits") && ok;

    act::Meta token_meta;
    token_meta.storage().emplace<act::token::Meta>().scales = {0.5f};
    ok = run_q8_h1_activation_scale_case(
             std::move(token_meta), 0.5f, "H1 token activation scale bits") && ok;

    act::Meta stripe_meta;
    stripe_meta.storage().emplace<act::stripe::Meta>().scales = {0.5f};
    ok = run_q8_h1_activation_scale_case(
             std::move(stripe_meta), 0.5f, "H1 stripe activation scale bits") && ok;

    act::Meta exsia_meta;
    exsia_meta.storage().emplace<act::exsia::Meta>().theta = {-1};
    return run_q8_h1_activation_scale_case(
               std::move(exsia_meta), 0.5f, "H1 EXSIA activation scale bits") && ok;
}

bool run_unpacked_h1_tail_case(bool stripe_mode) {
    constexpr size_t columns = 129;
    constexpr size_t depth = 65;
    constexpr size_t blocks = 3;
    constexpr float s_rf_value = 0.00032747327350080013f;
    const std::array<ggml::gemmini::quants::QactOutlier, 3> outliers = {
        ggml::gemmini::quants::QactOutlier {0, 64, 548339296},
        ggml::gemmini::quants::QactOutlier {0, 0, 548339296},
        ggml::gemmini::quants::QactOutlier {0, 32, 548339296},
    };

    std::vector<int8_t> weights(columns * depth, 1);
    std::vector<uint8_t> c_b(columns * blocks, 86);
    std::vector<float> row_s_rf(columns, s_rf_value);
    std::vector<uint16_t> row_R(columns, 4095);
    constexpr size_t stripe_width = 17;
    const size_t stripe_count = (columns + stripe_width - 1) / stripe_width;
    std::vector<float> stripe_s_rf(stripe_count, s_rf_value);
    std::vector<uint16_t> stripe_R(stripe_count, 4095);
    std::vector<float> output(columns, 0.0f);

    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = columns;
    args.K = depth;
    args.B = reinterpret_cast<elem_t *>(weights.data());
    args.sB = depth;
    args.f_out = output.data();
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_0_unpacked_to_h1;
    args.c_b = c_b.data();
    args.blocks_per_row = blocks;
    args.blocks_K = blocks;
    args.blocks_J = columns;
    args.block_size_k = QK8_0;
    if (stripe_mode) {
        args.stripe_J = stripe_width;
        args.s_rf_stripe = stripe_s_rf.data();
        args.R_stripe = stripe_R.data();
    } else {
        args.s_rf = row_s_rf.data();
        args.R = row_R.data();
    }

    ggml::gemmini::quants::dec::compensate_activation_dec(
        {outliers.begin(), outliers.end()}, args, "test");
    float expected = 0.0f;
    for (size_t block = 0; block < blocks; ++block) {
        const uint64_t c_eff = static_cast<uint64_t>(c_b[block]) + 4095;
        expected += static_cast<float>(
            static_cast<double>(outliers[block].residual) * c_eff * s_rf_value * 1.0f);
    }
    bool ok = true;
    for (float value : output)
        ok = check(float_bits(value) == float_bits(expected),
                   stripe_mode ? "unpacked stripe H1 K/J tail bits" : "unpacked row H1 K/J tail bits") && ok;
    return ok;
}

bool test_unpacked_h1_tail_routes_preserve_bits() {
    return run_unpacked_h1_tail_case(false) && run_unpacked_h1_tail_case(true);
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

bool test_sparse_grouped_tails() {
    constexpr size_t rows = 8;
    constexpr size_t cols = 50257;
    constexpr size_t depth = 65;
    std::vector<int8_t> weights(depth * cols, 1);
    std::vector<float> output(rows * cols, 0.0f);
    ggml_gemmini_args_t args = dense_args(rows, cols, depth, weights, output, 0.5f);
    const auto result = ggml::gemmini::quants::dec::compensate_activation_dec(
        { { 6, 64, 3 } }, args, "test");

    double norm_squared = 0.0;
    bool ok = check(result.nnz == 1 && result.unique_k_count == 1, "single sparse tail residual accounting");
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < cols; ++column) {
            const float value = output[row * cols + column];
            ok = check(value == (row == 6 ? 1.5f : 0.0f), "sparse grouped tail output") && ok;
            norm_squared += static_cast<double>(value) * value;
        }
    }
    ok = check(std::sqrt(norm_squared) == std::sqrt(static_cast<double>(cols) * 2.25),
               "J=50257 result norm") && ok;

    constexpr size_t all_group_cols = 5;
    std::vector<int8_t> all_group_weights(depth * all_group_cols, 1);
    std::vector<float> all_group_output(all_group_cols, 0.0f);
    ggml_gemmini_args_t all_group_args = dense_args(
        1, all_group_cols, depth, all_group_weights, all_group_output, 0.25f);
    const auto all_group_result = ggml::gemmini::quants::dec::compensate_activation_dec(
        { { 0, 0, 1 }, { 0, 32, 2 }, { 0, 64, 3 } }, all_group_args, "test");
    ok = check(all_group_result.nnz == 3 && all_group_result.unique_k_count == 3,
               "all compute groups active accounting") && ok;
    for (float value : all_group_output)
        ok = check(value == 1.5f, "all compute groups active output") && ok;
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

void set_dec_group_k_csc_force(const char *value) {
#if defined(_WIN32)
    _putenv_s("DEC_GROUP_K_CSC_FORCE", value ? value : "");
#else
    if (value)
        setenv("DEC_GROUP_K_CSC_FORCE", value, 1);
    else
        unsetenv("DEC_GROUP_K_CSC_FORCE");
#endif
}

void set_dec_group_k_csc_enable(const char *value) {
#if defined(_WIN32)
    _putenv_s("DEC_GROUP_K_CSC_ENABLE", value ? value : "");
#else
    if (value)
        setenv("DEC_GROUP_K_CSC_ENABLE", value, 1);
    else
        unsetenv("DEC_GROUP_K_CSC_ENABLE");
#endif
}

void set_dec_group_k_csc_disable(const char *value) {
#if defined(_WIN32)
    _putenv_s("DEC_GROUP_K_CSC_DISABLE", value ? value : "");
#else
    if (value)
        setenv("DEC_GROUP_K_CSC_DISABLE", value, 1);
    else
        unsetenv("DEC_GROUP_K_CSC_DISABLE");
#endif
}

struct ScopedDecGroupKCscEnv {
    std::string saved_force;
    std::string saved_enable;
    std::string saved_disable;
    bool had_force = false;
    bool had_enable = false;
    bool had_disable = false;

    ScopedDecGroupKCscEnv() {
        if (const char *force = std::getenv("DEC_GROUP_K_CSC_FORCE")) {
            saved_force = force;
            had_force = true;
        }
        if (const char *enable = std::getenv("DEC_GROUP_K_CSC_ENABLE")) {
            saved_enable = enable;
            had_enable = true;
        }
        if (const char *disable = std::getenv("DEC_GROUP_K_CSC_DISABLE")) {
            saved_disable = disable;
            had_disable = true;
        }
    }

    ~ScopedDecGroupKCscEnv() {
        set_dec_group_k_csc_force(had_force ? saved_force.c_str() : nullptr);
        set_dec_group_k_csc_enable(had_enable ? saved_enable.c_str() : nullptr);
        set_dec_group_k_csc_disable(had_disable ? saved_disable.c_str() : nullptr);
    }
};

struct ReplayTimingSummary {
    double median_us = 0.0;
    double p10_us = 0.0;
    double p90_us = 0.0;
};

struct ReplayTimingStats {
    ReplayTimingSummary compute;
    ReplayTimingSummary total;
};

const char *group_k_csc_width_path_name(
    ggml::gemmini::quants::dec::GroupKCSCWidthPath width_path) {
    switch (width_path) {
        case ggml::gemmini::quants::dec::GroupKCSCWidthPath::Mixed:
            return "mixed";
        case ggml::gemmini::quants::dec::GroupKCSCWidthPath::AllInt32:
            return "all-int32";
        case ggml::gemmini::quants::dec::GroupKCSCWidthPath::AllInt64:
            return "all-int64";
    }
    return "unknown";
}

void print_group_k_csc_replay_cost(
    const char *label,
    const ggml::gemmini::quants::dec::GroupKCSCScalarStats &stats) {
    std::printf(
        "replay group_k_csc_cost case=%s width_path=%s classify=%zu scratch_init=%zu sparse_update=%zu merge=%zu safe_updates=%zu fallback_updates=%zu branch_entry_classify=%zu classify_cycles=%llu scratch_init_cycles=%llu sparse_update_cycles=%llu merge_cycles=%llu\n",
        label,
        group_k_csc_width_path_name(stats.width_path),
        stats.classification_work_count,
        stats.scratch_init_count,
        stats.sparse_update_count,
        stats.merge_count,
        stats.safe_update_count,
        stats.fallback_update_count,
        stats.branch_entry_classification_count,
        static_cast<unsigned long long>(stats.classification_cycles),
        static_cast<unsigned long long>(stats.scratch_init_cycles),
        static_cast<unsigned long long>(stats.sparse_update_cycles),
        static_cast<unsigned long long>(stats.merge_cycles));
}

bool check_group_k_csc_stage_cycles(
    const ggml::gemmini::quants::dec::GroupKCSCScalarStats &stats,
    bool expect_classification_cycles,
    const char *message) {
#if LOG_CYCLE
    return check((!expect_classification_cycles || stats.classification_cycles > 0) &&
                     (expect_classification_cycles || stats.classification_cycles == 0) &&
                     stats.scratch_init_cycles > 0 &&
                     stats.sparse_update_cycles > 0 &&
                     stats.merge_cycles > 0,
                 message);
#else
    (void)expect_classification_cycles;
    return check(stats.classification_cycles == 0 &&
                     stats.scratch_init_cycles == 0 &&
                     stats.sparse_update_cycles == 0 &&
                     stats.merge_cycles == 0,
                 message);
#endif
}

bool byte_identical(const std::vector<float> &lhs, const std::vector<float> &rhs);

template <typename Runner>
ReplayTimingStats benchmark_replay_kernel(
    std::vector<float> &output,
    const std::vector<float> &expected,
    Runner &&runner,
    size_t warmup_count,
    size_t measured_count,
    bool &ok) {
    using clock = std::chrono::steady_clock;
    std::vector<double> compute_samples_us;
    std::vector<double> total_samples_us;
    compute_samples_us.reserve(measured_count);
    total_samples_us.reserve(measured_count);

    auto sample_once = [&](bool record) {
        const auto total_start = clock::now();
        std::fill(output.begin(), output.end(), 0.0f);
        const auto compute_start = clock::now();
        const bool accumulated = runner();
        const auto compute_end = clock::now();
        ok = check(accumulated, "replay kernel accumulates") && ok;
        ok = check(byte_identical(output, expected), "replay kernel preserves expected output") && ok;
        const auto total_end = clock::now();
        if (!record)
            return;
        const double compute_us = std::chrono::duration<double, std::micro>(
            compute_end - compute_start).count();
        const double total_us = std::chrono::duration<double, std::micro>(
            total_end - total_start).count();
        compute_samples_us.push_back(compute_us);
        total_samples_us.push_back(total_us);
    };

    for (size_t iteration = 0; iteration < warmup_count; ++iteration)
        sample_once(false);
    for (size_t iteration = 0; iteration < measured_count; ++iteration)
        sample_once(true);

    auto summarize = [](std::vector<double> samples) {
        std::sort(samples.begin(), samples.end());
        const size_t last = samples.size() - 1;
        const auto pick = [&](double quantile) {
            const size_t index = static_cast<size_t>(quantile * static_cast<double>(last));
            return samples[index];
        };
        return ReplayTimingSummary {
            samples[last / 2],
            pick(0.10),
            pick(0.90),
        };
    };

    return ReplayTimingStats {
        summarize(compute_samples_us),
        summarize(total_samples_us),
    };
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

bool test_q8_h1_group_k_csc_matches_row_direct() {
    ScopedDecGroupKCscEnv env_guard;
    bool ok = true;
    for (const size_t columns : { size_t {4}, size_t {257} }) {
        constexpr size_t rows = 2;
        constexpr size_t blocks_per_row = 2;
        constexpr size_t depth = QK8_0 * blocks_per_row;
        std::vector<block_q8_h1> blocks(columns * blocks_per_row);
        for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
            for (size_t offset = 0; offset < sizeof(blocks[block_index].qs); ++offset)
                blocks[block_index].qs[offset] = static_cast<int8_t>(
                    static_cast<int>((offset * 5 + block_index * 3) % 15) - 7);
            blocks[block_index].c_b = static_cast<uint8_t>(3 + block_index % 7);
            blocks[block_index].R = static_cast<uint16_t>(5 + block_index % 11);
            blocks[block_index].s_rf = 0.0625f * static_cast<float>(1 + block_index % 3);
        }

        const std::vector<ggml::gemmini::quants::QactOutlier> outliers = {
            { 0, 0, 3 }, { 0, 33, -1 }, { 1, 2, 2 }, { 1, 63, 4 },
        };
        std::vector<float> row_direct_output(rows * columns, 0.0f);
        ggml_gemmini_args_t args = hierarchical_args(rows, row_direct_output);
        args.J = columns;
        args.K = depth;
        args.sB = depth;
        args.blocks_per_row = blocks_per_row;
        args.blocks_K = blocks_per_row;
        args.blocks_J = columns;
        args.blocks_I = columns;
        args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
        args.q8_h1_blocks = blocks.data();
        args.q8_h1_block_count = blocks.size();
        args.q8_h1_rows = columns;

        set_dec_group_k_csc_force(nullptr);
        set_dec_group_k_csc_enable(nullptr);
        set_dec_group_k_csc_disable("1");
        const auto row_direct_result = ggml::gemmini::quants::dec::compensate_activation_dec(
            outliers, args, "test");

        std::vector<float> group_k_csc_output(rows * columns, 0.0f);
        args.f_out = group_k_csc_output.data();
        set_dec_group_k_csc_disable(nullptr);
        set_dec_group_k_csc_enable("1");
        const auto group_k_csc_result = ggml::gemmini::quants::dec::compensate_activation_dec(
            outliers, args, "test");

        const bool output_equal = byte_identical(row_direct_output, group_k_csc_output);
        ok = check(row_direct_result.group_k_csc_plan_bytes == 0 &&
                       group_k_csc_result.group_k_csc_plan_bytes > 0 &&
                       group_k_csc_result.weight_vector_load_count > 0 &&
                       output_equal,
                   columns < 8 ?
                       "H1 GroupKCSC NR4 matches RowDirect" :
                       "H1 GroupKCSC NR8 matches RowDirect") && ok;
    }
    return ok;
}

bool test_q8_h1_group_k_csc_auto_dispatch() {
    constexpr size_t rows = 32;
    constexpr size_t columns = kHierarchicalColumns;
    constexpr size_t blocks_per_row = 2;
    std::vector<block_q8_h1> blocks(columns * blocks_per_row);
    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        for (size_t offset = 0; offset < sizeof(blocks[block_index].qs); ++offset)
            blocks[block_index].qs[offset] = static_cast<int8_t>(
                static_cast<int>((offset * 5 + block_index * 3) % 15) - 7);
        blocks[block_index].c_b = static_cast<uint8_t>(3 + block_index % 7);
        blocks[block_index].R = static_cast<uint16_t>(5 + block_index % 11);
        blocks[block_index].s_rf = 0.0625f * static_cast<float>(1 + block_index % 3);
    }
    std::vector<ggml::gemmini::quants::QactOutlier> outliers;
    outliers.reserve(rows * 2);
    for (size_t row = 0; row < rows; ++row) {
        outliers.push_back({ static_cast<int32_t>(row), 0, 3 });
        outliers.push_back({ static_cast<int32_t>(row), 33, -1 });
    }

    ScopedDecGroupKCscEnv env_guard;
    set_dec_group_k_csc_force(nullptr);
    set_dec_group_k_csc_enable(nullptr);
    set_dec_group_k_csc_disable("1");
    std::vector<float> row_direct_output(rows * columns, 0.0f);
    ggml_gemmini_args_t args = hierarchical_args(rows, row_direct_output);
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = blocks.data();
    args.q8_h1_block_count = blocks.size();
    args.q8_h1_rows = columns;
    const auto row_direct_result = ggml::gemmini::quants::dec::compensate_activation_dec(
        outliers, args, "test");

    set_dec_group_k_csc_disable(nullptr);
    std::vector<float> default_output(rows * columns, 0.0f);
    args.f_out = default_output.data();
    const auto default_result = ggml::gemmini::quants::dec::compensate_activation_dec(
        outliers, args, "test");
    set_dec_group_k_csc_enable("1");
    std::vector<float> explicit_output(rows * columns, 0.0f);
    args.f_out = explicit_output.data();
    const auto explicit_result = ggml::gemmini::quants::dec::compensate_activation_dec(
        outliers, args, "test");
    const double rows_per_active_k_mean = explicit_result.unique_k_count == 0 ? 0.0 :
        static_cast<double>(explicit_result.active_row_k_pairs) / explicit_result.unique_k_count;
    return check(row_direct_result.group_k_csc_plan_bytes == 0 &&
                     default_result.group_k_csc_plan_bytes == 0 &&
                     explicit_result.group_k_csc_plan_bytes > 0 &&
                     explicit_result.weight_vector_load_count > 0 &&
                     rows_per_active_k_mean == static_cast<double>(rows) &&
                     byte_identical(row_direct_output, default_output) &&
                     byte_identical(default_output, explicit_output),
                 "H1 high-reuse default stays RowDirect and explicit GroupKCSC preserves bits");
}

bool test_group_k_csc_production_dispatch() {
    ScopedDecGroupKCscEnv env_guard;
    set_dec_group_k_csc_force(nullptr);
    set_dec_group_k_csc_enable(nullptr);
    set_dec_group_k_csc_disable(nullptr);

    bool ok = true;
    for (const size_t cols : { size_t {4}, size_t {8} }) {
        constexpr size_t rows = 4;
        constexpr size_t depth = 2;
        std::vector<int8_t> weights(depth * cols, 0);
        for (size_t j = 0; j < cols; ++j)
            weights[cols + j] = -static_cast<int8_t>(j + 1);
        std::vector<float> output(rows * cols, 0.0f);
        ggml_gemmini_args_t args = dense_args(rows, cols, depth, weights, output, 0.25f);
        const std::vector<ggml::gemmini::quants::QactOutlier> outliers = {
            { 0, 1, 1 }, { 1, 1, 2 }, { 2, 1, 3 }, { 3, 1, 4 },
        };
        set_dec_group_k_csc_enable(nullptr);
        const auto default_result = ggml::gemmini::quants::dec::compensate_activation_dec(
            outliers, args, "test");
        std::vector<float> default_output = output;
        std::fill(output.begin(), output.end(), 0.0f);
        set_dec_group_k_csc_enable("1");
        const auto result = ggml::gemmini::quants::dec::compensate_activation_dec(
            outliers, args, "test");
        set_dec_group_k_csc_enable(nullptr);
        const double rows_per_active_k_mean = default_result.unique_k_count == 0 ? 0.0 :
            static_cast<double>(default_result.active_row_k_pairs) / default_result.unique_k_count;
        std::vector<float> expected(rows * cols);
        for (size_t row = 0; row < rows; ++row)
            for (size_t j = 0; j < cols; ++j)
                expected[row * cols + j] =
                    -0.25f * static_cast<float>((row + 1) * (j + 1));
        const bool default_output_equal = byte_identical(default_output, expected);
        const bool output_equal = byte_identical(output, expected);
        const double reuse = result.weight_scalar_load_count == 0 ? 0.0 :
            static_cast<double>(result.logical_weight_reference_count) /
                result.weight_scalar_load_count;
        std::printf(
            "dispatch dense default=%s default_plan_bytes=%zu default_vector_loads=%zu default_output_equal=%s enabled=%s plan_bytes=%zu nr=%zu vector_loads=%zu reuse=%.1f rows_per_active_k_mean=%.1f output_equal=%s\n",
            default_result.group_k_csc_plan_bytes == 0 ? "row-direct" : "group-k-csc",
            default_result.group_k_csc_plan_bytes,
            default_result.weight_vector_load_count,
            default_output_equal ? "yes" : "no",
            result.group_k_csc_plan_bytes == 0 ? "row-direct" : "group-k-csc",
            result.group_k_csc_plan_bytes,
            cols < 8 ? size_t {4} : size_t {8},
            result.weight_vector_load_count,
            reuse,
            rows_per_active_k_mean,
            output_equal ? "yes" : "no");
        ok = check(default_result.group_k_csc_plan_bytes == 0 &&
                       default_result.weight_vector_load_count == 0 &&
                       rows_per_active_k_mean == 4.0 &&
                       default_output_equal &&
                       result.group_k_csc_plan_bytes > 0 &&
                       result.logical_weight_reference_count == rows * cols &&
                       result.weight_scalar_load_count == cols &&
                       result.weight_vector_load_count == 1 && reuse == 4.0 && output_equal,
                   cols < 8 ? "dense small-reuse default stays row-direct and ENABLE selects GroupKCSC NR4" :
                              "dense small-reuse default stays row-direct and ENABLE selects GroupKCSC NR8") && ok;
    }

    return ok;
}

bool test_group_k_csc_k_and_j_tails() {
    ScopedDecGroupKCscEnv env_guard;
    bool ok = true;
    for (const size_t depth : { size_t {31}, size_t {32}, size_t {33},
                                size_t {63}, size_t {64}, size_t {65},
                                size_t {767}, size_t {768}, size_t {769} }) {
        for (const size_t columns : { size_t {1}, size_t {4}, size_t {8}, size_t {131} }) {
            constexpr size_t rows = 2;
            std::vector<int8_t> weights(depth * columns, 0);
            for (size_t k = 0; k < depth; ++k)
                for (size_t j = 0; j < columns; ++j)
                    weights[k * columns + j] = static_cast<int8_t>((k + 3 * j) % 13 - 6);

            std::vector<ggml::gemmini::quants::QactOutlier> outliers = {
                { 0, 0, 3 },
                { 0, static_cast<int32_t>(depth - 1), -2 },
                { 1, static_cast<int32_t>(depth / 2), 1 },
            };
            std::vector<float> row_direct_output(rows * columns, 0.0f);
            ggml_gemmini_args_t args = dense_args(
                rows, columns, depth, weights, row_direct_output, 0.25f);
            set_dec_group_k_csc_force(nullptr);
            set_dec_group_k_csc_enable(nullptr);
            set_dec_group_k_csc_disable("1");
            const auto row_direct_result = ggml::gemmini::quants::dec::compensate_activation_dec(
                outliers, args, "test");

            std::vector<float> group_k_csc_output(rows * columns, 0.0f);
            args.f_out = group_k_csc_output.data();
            set_dec_group_k_csc_disable(nullptr);
            set_dec_group_k_csc_enable("1");
            const auto group_k_csc_result = ggml::gemmini::quants::dec::compensate_activation_dec(
                outliers, args, "test");

            ok = check(row_direct_result.group_k_csc_plan_bytes == 0 &&
                           group_k_csc_result.group_k_csc_plan_bytes > 0 &&
                           group_k_csc_result.weight_vector_load_count > 0 &&
                           group_k_csc_result.int_mac_count == outliers.size() * columns &&
                           byte_identical(row_direct_output, group_k_csc_output),
                       "GroupKCSC preserves K boundaries and NR tails") && ok;
        }
    }
    return ok;
}

bool test_fixed_residual_replay_baseline() {
    using ggml::gemmini::quants::dec::ActiveRowGroup;
    using ggml::gemmini::quants::dec::GroupKCSCPlan;
    using ggml::gemmini::quants::dec::GroupKCSCScalarStats;
    using ggml::gemmini::quants::dec::GroupKCSCWidthPath;
    using ggml::gemmini::quants::dec::ResidualGroupEntry;

    constexpr size_t rows = 3;
    constexpr size_t cols = 5;
    constexpr size_t depth = 35;
    const std::array<int8_t, cols> k0_weights = { 1, -2, 3, -4, 5 };
    const std::array<int8_t, cols> k33_weights = { 2, 1, -3, 4, -2 };
    const std::array<int8_t, cols> k34_weights = { -1, 3, 2, 0, -4 };
    std::vector<int8_t> weights(depth * cols, 0);
    for (size_t column = 0; column < cols; ++column) {
        weights[column] = k0_weights[column];
        weights[33 * cols + column] = k33_weights[column];
        weights[34 * cols + column] = k34_weights[column];
    }

    const std::vector<ggml::gemmini::quants::QactOutlier> shuffled_outliers = {
        { 2, 34, 2 }, { 0, 0, 4 }, { 1, 33, -3 }, { 0, 0, -1 },
        { 2, 0, -2 }, { 1, 33, 5 }, { 0, 34, 1 },
    };
    std::vector<float> output(rows * cols, 0.0f);
    ggml_gemmini_args_t args = dense_args(rows, cols, depth, weights, output, 0.25f);
    ScopedDecGroupKCscEnv env_guard;
    set_dec_group_k_csc_force(nullptr);
    set_dec_group_k_csc_enable(nullptr);
    set_dec_group_k_csc_disable(nullptr);
    const auto result = ggml::gemmini::quants::dec::compensate_activation_dec(
        shuffled_outliers, args, "test");
    const std::array<uint32_t, rows * cols> expected_bits = {
        0x3f000000, 0xbf400000, 0x40300000, 0xc0400000, 0x40300000,
        0x3f800000, 0x3f000000, 0xbfc00000, 0x40000000, 0xbf800000,
        0xbf800000, 0x40200000, 0xbf000000, 0x40000000, 0xc0900000,
    };
    constexpr size_t active_row_groups = 5;
    constexpr size_t active_k_groups = 2;
    const size_t expected_current_sparse_plan_bytes =
        shuffled_outliers.size() * sizeof(ResidualGroupEntry) +
        active_row_groups * sizeof(ActiveRowGroup) +
        (active_k_groups + 1) * sizeof(size_t) +
        active_row_groups * sizeof(size_t) +
        3 * sizeof(uint32_t) + active_k_groups * sizeof(uint32_t);
    const size_t expected_group_k_csc_plan_bytes =
        (active_k_groups * (ggml::gemmini::quants::dec::kDecGroupSizeK + 1) +
         shuffled_outliers.size() + (active_k_groups + 1) + active_row_groups) * sizeof(uint32_t);

    std::fill(output.begin(), output.end(), 0.0f);
    set_dec_group_k_csc_force("1");
    const auto forced_result = ggml::gemmini::quants::dec::compensate_activation_dec(
        shuffled_outliers, args, "test");
    std::vector<float> forced_output = output;
    std::fill(output.begin(), output.end(), 0.0f);
    set_dec_group_k_csc_force(nullptr);
    set_dec_group_k_csc_disable("1");
    const auto disabled_result = ggml::gemmini::quants::dec::compensate_activation_dec(
        shuffled_outliers, args, "test");
    std::vector<float> disabled_output = output;
    set_dec_group_k_csc_disable(nullptr);
    std::printf(
        "dispatch low-reuse route=%s plan_bytes=%zu vector_loads=%zu forced_route=%s forced_plan_bytes=%zu forced_vector_loads=%zu disabled_route=%s disabled_plan_bytes=%zu disabled_vector_loads=%zu\n",
        result.group_k_csc_plan_bytes == 0 ? "row-direct" : "group-k-csc",
        result.group_k_csc_plan_bytes,
        result.weight_vector_load_count,
        forced_result.group_k_csc_plan_bytes == 0 ? "row-direct" : "group-k-csc",
        forced_result.group_k_csc_plan_bytes,
        forced_result.weight_vector_load_count,
        disabled_result.group_k_csc_plan_bytes == 0 ? "row-direct" : "group-k-csc",
        disabled_result.group_k_csc_plan_bytes,
        disabled_result.weight_vector_load_count);

    std::vector<ResidualGroupEntry> entries = {
        { 2, 34, 2 }, { 0, 0, 4 }, { 1, 33, -3 }, { 0, 0, -1 },
        { 2, 0, -2 }, { 1, 33, 5 }, { 0, 34, 1 },
    };
    std::vector<ActiveRowGroup> groups;
    ggml::gemmini::quants::dec::build_active_row_groups(entries, groups);
    std::vector<size_t> group_offsets;
    std::vector<size_t> group_row_group_indices;
    ggml::gemmini::quants::dec::build_group_major_index(
        groups, active_k_groups, group_offsets, group_row_group_indices);
    GroupKCSCPlan group_k_csc_plan;
    const bool group_k_csc_ready = ggml::gemmini::quants::dec::build_group_k_csc_plan(
        entries, groups, group_offsets, group_row_group_indices, active_k_groups, group_k_csc_plan);
    const auto scalar_route_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        args, ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    std::vector<float> current_row_grouped(rows * cols, 0.0f);
    ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar(
        args, scalar_route_plan, rows, cols, nullptr, entries, groups, group_offsets,
        group_row_group_indices, current_row_grouped.data());
    std::vector<float> group_k_csc_scalar(rows * cols, 0.0f);
    GroupKCSCScalarStats group_k_csc_stats;
    const bool group_k_csc_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            group_k_csc_scalar.data(), group_k_csc_stats);
    std::vector<float> group_k_csc_nr4(rows * cols, 0.0f);
    GroupKCSCScalarStats group_k_csc_nr4_stats;
    const bool group_k_csc_nr4_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc_nr4(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            group_k_csc_nr4.data(), group_k_csc_nr4_stats);
    std::vector<float> group_k_csc_nr8(rows * cols, 0.0f);
    GroupKCSCScalarStats group_k_csc_nr8_stats;
    const bool group_k_csc_nr8_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            group_k_csc_nr8.data(), group_k_csc_nr8_stats);
    std::vector<float> group_k_csc_mixed(rows * cols, 0.0f);
    GroupKCSCScalarStats group_k_csc_mixed_stats;
    const bool group_k_csc_mixed_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            group_k_csc_mixed.data(), group_k_csc_mixed_stats);
    std::vector<float> group_k_csc_mixed_nr4(rows * cols, 0.0f);
    GroupKCSCScalarStats group_k_csc_mixed_nr4_stats;
    const bool group_k_csc_mixed_nr4_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr4(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            group_k_csc_mixed_nr4.data(), group_k_csc_mixed_nr4_stats);

    bool ok = check(result.total_selected == shuffled_outliers.size() && result.nnz == shuffled_outliers.size() &&
                        result.unique_k_count == 3 && result.int_mac_count == 35 &&
                        result.logical_weight_reference_count == 35 &&
                        result.weight_scalar_load_count == 35 &&
                        result.weight_vector_load_count == 0 && result.estimated_weight_bytes_read == 35 &&
                        result.active_row_k_pairs == 5 && result.rows_per_active_k_max == 2 &&
                        result.ycom_global_write_count == rows * cols &&
                        result.current_sparse_plan_bytes == expected_current_sparse_plan_bytes &&
                        result.group_k_csc_plan_bytes == 0 &&
                        forced_result.group_k_csc_plan_bytes == expected_group_k_csc_plan_bytes &&
                        forced_result.weight_vector_load_count == 6 &&
                        disabled_result.group_k_csc_plan_bytes == 0 &&
                        disabled_result.weight_vector_load_count == 0 &&
                        result.thread_scratch_bytes == rows * cols * sizeof(int64_t),
                    "fixed replay defaults to row-direct and force builds GroupKCSC");
    for (size_t index = 0; index < output.size(); ++index)
        ok = check(float_bits(output[index]) == expected_bits[index], "fixed replay baseline output bits") && ok;
    ok = check(group_k_csc_ready && group_k_csc_accumulated && group_k_csc_nr4_accumulated &&
                   group_k_csc_nr8_accumulated && group_k_csc_mixed_accumulated &&
                   group_k_csc_mixed_nr4_accumulated &&
                   byte_identical(output, disabled_output) &&
                   byte_identical(output, forced_output) &&
                   byte_identical(output, current_row_grouped) &&
                   byte_identical(current_row_grouped, group_k_csc_scalar) &&
                   byte_identical(group_k_csc_scalar, group_k_csc_nr4) &&
                   byte_identical(current_row_grouped, group_k_csc_nr8) &&
                   byte_identical(current_row_grouped, group_k_csc_mixed) &&
                   byte_identical(group_k_csc_mixed, group_k_csc_mixed_nr4),
               "fixed replay CurrentRowGrouped equals GroupKCSC NR1/4/8 mixed") && ok;
    ok = check(group_k_csc_stats.logical_weight_reference_count == 35 &&
                   group_k_csc_stats.weight_scalar_load_count == 15 &&
                   group_k_csc_stats.weight_vector_load_count == 0 &&
                   group_k_csc_stats.classification_work_count == 0 &&
                   group_k_csc_stats.scratch_init_count == rows * cols &&
                   group_k_csc_stats.sparse_update_count == 35 &&
                   group_k_csc_stats.merge_count == rows * cols &&
                   group_k_csc_stats.safe_update_count == 0 &&
                   group_k_csc_stats.fallback_update_count == 35 &&
                   group_k_csc_stats.branch_entry_classification_count == 0 &&
                   group_k_csc_stats.thread_scratch_bytes == rows * cols * sizeof(int64_t) &&
                   group_k_csc_nr4_stats.logical_weight_reference_count == 35 &&
                   group_k_csc_nr4_stats.weight_scalar_load_count == 15 &&
                   group_k_csc_nr4_stats.weight_vector_load_count == 6 &&
                   group_k_csc_nr4_stats.classification_work_count == 0 &&
                   group_k_csc_nr4_stats.scratch_init_count == rows * cols &&
                   group_k_csc_nr4_stats.sparse_update_count == 35 &&
                   group_k_csc_nr4_stats.merge_count == rows * cols &&
                   group_k_csc_nr4_stats.safe_update_count == 0 &&
                   group_k_csc_nr4_stats.fallback_update_count == 35 &&
                   group_k_csc_nr4_stats.branch_entry_classification_count == 0 &&
                   group_k_csc_nr4_stats.thread_scratch_bytes == rows * cols * sizeof(int64_t) &&
                   group_k_csc_nr8_stats.logical_weight_reference_count == 35 &&
                   group_k_csc_nr8_stats.weight_scalar_load_count == 15 &&
                   group_k_csc_nr8_stats.weight_vector_load_count == 3 &&
                   group_k_csc_nr8_stats.classification_work_count == 0 &&
                   group_k_csc_nr8_stats.scratch_init_count == rows * cols &&
                   group_k_csc_nr8_stats.sparse_update_count == 35 &&
                   group_k_csc_nr8_stats.merge_count == rows * cols &&
                   group_k_csc_nr8_stats.safe_update_count == 0 &&
                   group_k_csc_nr8_stats.fallback_update_count == 35 &&
                   group_k_csc_nr8_stats.branch_entry_classification_count == 0 &&
                   group_k_csc_nr8_stats.thread_scratch_bytes == rows * cols * sizeof(int64_t) &&
                   group_k_csc_mixed_stats.logical_weight_reference_count == 35 &&
                   group_k_csc_mixed_stats.weight_scalar_load_count == 15 &&
                   group_k_csc_mixed_stats.weight_vector_load_count == 3 &&
                   group_k_csc_mixed_stats.classification_work_count == entries.size() &&
                   group_k_csc_mixed_stats.scratch_init_count == rows * cols &&
                   group_k_csc_mixed_stats.sparse_update_count == 35 &&
                   group_k_csc_mixed_stats.merge_count == rows * cols &&
                   group_k_csc_mixed_stats.safe_update_count == 35 &&
                   group_k_csc_mixed_stats.fallback_update_count == 0 &&
                   group_k_csc_mixed_stats.branch_entry_classification_count == 0 &&
                   group_k_csc_mixed_stats.thread_scratch_bytes == rows * cols * sizeof(int32_t) &&
                   group_k_csc_mixed_stats.int32_row_count == rows &&
                   group_k_csc_mixed_stats.int64_fallback_row_count == 0 &&
                   group_k_csc_mixed_stats.width_path == GroupKCSCWidthPath::AllInt32 &&
                   group_k_csc_mixed_nr4_stats.logical_weight_reference_count == 35 &&
                   group_k_csc_mixed_nr4_stats.weight_scalar_load_count == 15 &&
                   group_k_csc_mixed_nr4_stats.weight_vector_load_count == 6 &&
                   group_k_csc_mixed_nr4_stats.classification_work_count == entries.size() &&
                   group_k_csc_mixed_nr4_stats.scratch_init_count == rows * cols &&
                   group_k_csc_mixed_nr4_stats.sparse_update_count == 35 &&
                   group_k_csc_mixed_nr4_stats.merge_count == rows * cols &&
                   group_k_csc_mixed_nr4_stats.safe_update_count == 35 &&
                   group_k_csc_mixed_nr4_stats.fallback_update_count == 0 &&
                   group_k_csc_mixed_nr4_stats.branch_entry_classification_count == 0 &&
                   group_k_csc_mixed_nr4_stats.thread_scratch_bytes == rows * cols * sizeof(int32_t) &&
                   group_k_csc_mixed_nr4_stats.int32_row_count == rows &&
                   group_k_csc_mixed_nr4_stats.int64_fallback_row_count == 0 &&
                   group_k_csc_mixed_nr4_stats.width_path == GroupKCSCWidthPath::AllInt32,
               "fixed replay GroupKCSC NR4/8 reuse all-safe counters") &&
        check_group_k_csc_stage_cycles(group_k_csc_stats, false, "fixed replay scalar stage cycles") &&
        check_group_k_csc_stage_cycles(group_k_csc_nr4_stats, false, "fixed replay nr4 stage cycles") &&
        check_group_k_csc_stage_cycles(group_k_csc_nr8_stats, false, "fixed replay nr8 stage cycles") &&
        check_group_k_csc_stage_cycles(group_k_csc_mixed_stats, true, "fixed replay mixed nr8 stage cycles") &&
        check_group_k_csc_stage_cycles(group_k_csc_mixed_nr4_stats, true, "fixed replay mixed nr4 stage cycles") &&
        ok;
    print_group_k_csc_replay_cost("all-safe", group_k_csc_mixed_stats);

    constexpr size_t warmup_count = 5;
    constexpr size_t measured_count = 25;
    std::vector<float> measured_row_direct(rows * cols, 0.0f);
    const ReplayTimingStats row_direct_timing = benchmark_replay_kernel(
        measured_row_direct,
        current_row_grouped,
        [&]() {
            ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar(
                args, scalar_route_plan, rows, cols, nullptr, entries, groups, group_offsets,
                group_row_group_indices, measured_row_direct.data());
            return true;
        },
        warmup_count,
        measured_count,
        ok);
    std::vector<float> measured_nr4(rows * cols, 0.0f);
    const ReplayTimingStats nr4_timing = benchmark_replay_kernel(
        measured_nr4,
        current_row_grouped,
        [&]() {
            return ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr4(
                args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
                measured_nr4.data(), group_k_csc_mixed_nr4_stats);
        },
        warmup_count,
        measured_count,
        ok);
    std::vector<float> measured_nr8(rows * cols, 0.0f);
    const ReplayTimingStats nr8_timing = benchmark_replay_kernel(
        measured_nr8,
        current_row_grouped,
        [&]() {
            return ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
                args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
                measured_nr8.data(), group_k_csc_mixed_stats);
        },
        warmup_count,
        measured_count,
        ok);
    const double best_group_k_csc_compute = std::min(
        nr4_timing.compute.median_us, nr8_timing.compute.median_us);
    const char *keep_decision = best_group_k_csc_compute <= row_direct_timing.compute.median_us ?
        "keep" : "revert";
    std::printf(
        "replay benchmark warmup=%zu measured=%zu row_direct compute_us median=%.3f p10=%.3f p90=%.3f total_us median=%.3f p10=%.3f p90=%.3f\n",
        warmup_count, measured_count,
        row_direct_timing.compute.median_us, row_direct_timing.compute.p10_us, row_direct_timing.compute.p90_us,
        row_direct_timing.total.median_us, row_direct_timing.total.p10_us, row_direct_timing.total.p90_us);
    std::printf(
        "replay benchmark warmup=%zu measured=%zu group_k_csc_nr4 compute_us median=%.3f p10=%.3f p90=%.3f total_us median=%.3f p10=%.3f p90=%.3f\n",
        warmup_count, measured_count,
        nr4_timing.compute.median_us, nr4_timing.compute.p10_us, nr4_timing.compute.p90_us,
        nr4_timing.total.median_us, nr4_timing.total.p10_us, nr4_timing.total.p90_us);
    std::printf(
        "replay benchmark warmup=%zu measured=%zu group_k_csc_nr8 compute_us median=%.3f p10=%.3f p90=%.3f total_us median=%.3f p10=%.3f p90=%.3f\n",
        warmup_count, measured_count,
        nr8_timing.compute.median_us, nr8_timing.compute.p10_us, nr8_timing.compute.p90_us,
        nr8_timing.total.median_us, nr8_timing.total.p10_us, nr8_timing.total.p90_us);
    std::printf(
        "replay benchmark decision=%s baseline=row-direct compare=group-k-csc-min-compute\n",
        keep_decision);

    auto reordered_outliers = shuffled_outliers;
    std::reverse(reordered_outliers.begin(), reordered_outliers.end());
    std::vector<float> reordered_output(rows * cols, 0.0f);
    ggml_gemmini_args_t reordered_args = dense_args(
        rows, cols, depth, weights, reordered_output, 0.25f);
    const auto reordered_result = ggml::gemmini::quants::dec::compensate_activation_dec(
        reordered_outliers, reordered_args, "test");
    return check(byte_identical(output, reordered_output),
                 "fixed replay shuffled residual output is byte-identical") &&
        check(reordered_result.int_mac_count == result.int_mac_count &&
                  reordered_result.logical_weight_reference_count == result.logical_weight_reference_count &&
                  reordered_result.weight_scalar_load_count == result.weight_scalar_load_count &&
                  reordered_result.active_row_k_pairs == result.active_row_k_pairs &&
                  reordered_result.rows_per_active_k_max == result.rows_per_active_k_max &&
                  reordered_result.ycom_global_write_count == result.ycom_global_write_count &&
                  reordered_result.current_sparse_plan_bytes == result.current_sparse_plan_bytes &&
                  reordered_result.group_k_csc_plan_bytes == result.group_k_csc_plan_bytes &&
                  reordered_result.thread_scratch_bytes == result.thread_scratch_bytes,
               "fixed replay counters are deterministic") && ok;
}

bool test_fixed_residual_replay_high_reuse() {
    using ggml::gemmini::quants::dec::ActiveRowGroup;
    using ggml::gemmini::quants::dec::GroupKCSCPlan;
    using ggml::gemmini::quants::dec::GroupKCSCScalarStats;
    using ggml::gemmini::quants::dec::GroupKCSCWidthPath;
    using ggml::gemmini::quants::dec::ResidualGroupEntry;

    constexpr size_t rows = 256;
    constexpr size_t cols = 131;
    constexpr size_t depth = 35;
    constexpr size_t active_k_count = 2;
    constexpr size_t expected_logical_refs = rows * cols * active_k_count;
    constexpr size_t expected_row_groups = rows * active_k_count;
    constexpr size_t warmup_count = 5;
    constexpr size_t measured_count = 25;
    std::vector<int8_t> weights(depth * cols, 0);
    for (size_t k = 0; k < depth; ++k)
        for (size_t column = 0; column < cols; ++column)
            weights[k * cols + column] = static_cast<int8_t>(
                static_cast<int>((k * 11 + column * 7) % 17) - 8);

    std::vector<ggml::gemmini::quants::QactOutlier> outliers;
    outliers.reserve(rows * active_k_count);
    std::vector<ResidualGroupEntry> entries;
    entries.reserve(rows * active_k_count);
    for (size_t row = 0; row < rows; ++row) {
        const int32_t k0_residual = static_cast<int32_t>(row % 7) + 1;
        const int32_t k33_residual = -static_cast<int32_t>(row % 5) - 2;
        outliers.push_back({ static_cast<int32_t>(row), 0, k0_residual });
        outliers.push_back({ static_cast<int32_t>(row), 33, k33_residual });
        entries.push_back({ static_cast<uint32_t>(row), 0, k0_residual });
        entries.push_back({ static_cast<uint32_t>(row), 33, k33_residual });
    }

    std::vector<float> output(rows * cols, 0.0f);
    ggml_gemmini_args_t args = dense_args(rows, cols, depth, weights, output, 0.125f);
    const auto scalar_route_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        args, ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    std::vector<ActiveRowGroup> groups;
    ggml::gemmini::quants::dec::build_active_row_groups(entries, groups);
    std::vector<size_t> group_offsets;
    std::vector<size_t> group_row_group_indices;
    ggml::gemmini::quants::dec::build_group_major_index(
        groups, active_k_count, group_offsets, group_row_group_indices);
    GroupKCSCPlan group_k_csc_plan;
    const bool group_k_csc_ready = ggml::gemmini::quants::dec::build_group_k_csc_plan(
        entries, groups, group_offsets, group_row_group_indices, active_k_count, group_k_csc_plan);

    ScopedDecGroupKCscEnv env_guard;
    set_dec_group_k_csc_force(nullptr);
    set_dec_group_k_csc_enable(nullptr);
    set_dec_group_k_csc_disable(nullptr);
    const auto default_result = ggml::gemmini::quants::dec::compensate_activation_dec(
        outliers, args, "test");
    const std::vector<float> default_output = output;
    std::fill(output.begin(), output.end(), 0.0f);
    set_dec_group_k_csc_disable("1");
    const auto row_direct_result = ggml::gemmini::quants::dec::compensate_activation_dec(
        outliers, args, "test");
    const std::vector<float> row_direct_output = output;
    set_dec_group_k_csc_disable(nullptr);

    std::vector<float> current_row_grouped(rows * cols, 0.0f);
    ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar(
        args, scalar_route_plan, rows, cols, nullptr, entries, groups, group_offsets,
        group_row_group_indices, current_row_grouped.data());
    std::vector<float> group_k_csc_nr4(rows * cols, 0.0f);
    GroupKCSCScalarStats group_k_csc_nr4_stats;
    const bool group_k_csc_nr4_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr4(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            group_k_csc_nr4.data(), group_k_csc_nr4_stats);
    std::vector<float> group_k_csc_nr8(rows * cols, 0.0f);
    GroupKCSCScalarStats group_k_csc_nr8_stats;
    const bool group_k_csc_nr8_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            group_k_csc_nr8.data(), group_k_csc_nr8_stats);

    const double row_direct_reuse = row_direct_result.weight_scalar_load_count == 0 ? 0.0 :
        static_cast<double>(row_direct_result.logical_weight_reference_count) /
            static_cast<double>(row_direct_result.weight_scalar_load_count);
    const double default_reuse = default_result.weight_scalar_load_count == 0 ? 0.0 :
        static_cast<double>(default_result.logical_weight_reference_count) /
            static_cast<double>(default_result.weight_scalar_load_count);
    const double nr4_reuse = group_k_csc_nr4_stats.weight_scalar_load_count == 0 ? 0.0 :
        static_cast<double>(group_k_csc_nr4_stats.logical_weight_reference_count) /
            static_cast<double>(group_k_csc_nr4_stats.weight_scalar_load_count);
    const double nr8_reuse = group_k_csc_nr8_stats.weight_scalar_load_count == 0 ? 0.0 :
        static_cast<double>(group_k_csc_nr8_stats.logical_weight_reference_count) /
            static_cast<double>(group_k_csc_nr8_stats.weight_scalar_load_count);
    const double rows_per_active_k_mean = default_result.unique_k_count == 0 ? 0.0 :
        static_cast<double>(default_result.active_row_k_pairs) / default_result.unique_k_count;

    bool ok = check(group_k_csc_ready && scalar_route_plan.valid,
                    "high-reuse replay prepares row-direct and GroupKCSC plans");
    ok = check(default_result.group_k_csc_plan_bytes > 0 &&
                   default_result.total_selected == outliers.size() &&
                   default_result.nnz == outliers.size() &&
                   default_result.unique_k_count == active_k_count &&
                   default_result.logical_weight_reference_count == expected_logical_refs &&
                   default_result.weight_scalar_load_count == active_k_count * cols &&
                   default_result.weight_vector_load_count == group_k_csc_nr8_stats.weight_vector_load_count &&
                   default_result.active_row_k_pairs == expected_row_groups &&
                   default_result.rows_per_active_k_max == rows &&
                   rows_per_active_k_mean == static_cast<double>(rows) &&
                   row_direct_result.group_k_csc_plan_bytes == 0 &&
                   row_direct_result.weight_scalar_load_count == expected_logical_refs &&
                   row_direct_result.weight_vector_load_count == 0 &&
                   row_direct_result.active_row_k_pairs == expected_row_groups &&
                   row_direct_result.rows_per_active_k_max == rows &&
                   row_direct_reuse == 1.0 && default_reuse == static_cast<double>(rows),
               "high-reuse replay default dispatch selects GroupKCSC NR8 and disable stays row-direct") && ok;
    ok = check(group_k_csc_nr4_accumulated && group_k_csc_nr8_accumulated &&
                   byte_identical(row_direct_output, default_output) &&
                   byte_identical(default_output, current_row_grouped) &&
                   byte_identical(current_row_grouped, group_k_csc_nr4) &&
                   byte_identical(current_row_grouped, group_k_csc_nr8),
               "high-reuse replay preserves bitwise output across RowDirect and GroupKCSC NR4/NR8") && ok;
    ok = check(group_k_csc_nr4_stats.logical_weight_reference_count == expected_logical_refs &&
                   group_k_csc_nr4_stats.weight_scalar_load_count == active_k_count * cols &&
                   group_k_csc_nr4_stats.classification_work_count == entries.size() &&
                   group_k_csc_nr4_stats.sparse_update_count == expected_logical_refs &&
                   group_k_csc_nr4_stats.merge_count == rows * cols &&
                   group_k_csc_nr4_stats.safe_update_count == expected_logical_refs &&
                   group_k_csc_nr4_stats.fallback_update_count == 0 &&
                   group_k_csc_nr4_stats.branch_entry_classification_count == 0 &&
                   group_k_csc_nr4_stats.width_path == GroupKCSCWidthPath::AllInt32 &&
                   group_k_csc_nr8_stats.logical_weight_reference_count == expected_logical_refs &&
                   group_k_csc_nr8_stats.weight_scalar_load_count == active_k_count * cols &&
                   group_k_csc_nr8_stats.classification_work_count == entries.size() &&
                   group_k_csc_nr8_stats.sparse_update_count == expected_logical_refs &&
                   group_k_csc_nr8_stats.merge_count == rows * cols &&
                   group_k_csc_nr8_stats.safe_update_count == expected_logical_refs &&
                   group_k_csc_nr8_stats.fallback_update_count == 0 &&
                   group_k_csc_nr8_stats.branch_entry_classification_count == 0 &&
                   group_k_csc_nr8_stats.width_path == GroupKCSCWidthPath::AllInt32,
               "high-reuse replay NR4/NR8 counters stay all-safe") &&
        check_group_k_csc_stage_cycles(group_k_csc_nr4_stats, true, "high-reuse nr4 stage cycles") &&
        check_group_k_csc_stage_cycles(group_k_csc_nr8_stats, true, "high-reuse nr8 stage cycles") &&
        ok;

    std::printf(
        "high-reuse dispatch rows=%zu cols=%zu depth=%zu disable_route=%s default_route=%s logical_refs=%zu disable_scalar_loads=%zu disable_vector_loads=%zu disable_reuse=%.1f default_scalar_loads=%zu default_vector_loads=%zu default_reuse=%.1f rows_per_active_k_mean=%.1f output_equal=%s\n",
        rows, cols, depth,
        row_direct_result.group_k_csc_plan_bytes == 0 ? "row-direct" : "group-k-csc",
        default_result.group_k_csc_plan_bytes == 0 ? "row-direct" : "group-k-csc",
        row_direct_result.logical_weight_reference_count,
        row_direct_result.weight_scalar_load_count,
        row_direct_result.weight_vector_load_count,
        row_direct_reuse,
        default_result.weight_scalar_load_count,
        default_result.weight_vector_load_count,
        default_reuse,
        rows_per_active_k_mean,
        byte_identical(row_direct_output, default_output) ? "yes" : "no");
    std::printf(
        "high-reuse group_k_csc_nr4 logical_refs=%zu scalar_loads=%zu vector_loads=%zu reuse=%.1f\n",
        group_k_csc_nr4_stats.logical_weight_reference_count,
        group_k_csc_nr4_stats.weight_scalar_load_count,
        group_k_csc_nr4_stats.weight_vector_load_count,
        nr4_reuse);
    std::printf(
        "high-reuse group_k_csc_nr8 logical_refs=%zu scalar_loads=%zu vector_loads=%zu reuse=%.1f\n",
        group_k_csc_nr8_stats.logical_weight_reference_count,
        group_k_csc_nr8_stats.weight_scalar_load_count,
        group_k_csc_nr8_stats.weight_vector_load_count,
        nr8_reuse);
    print_group_k_csc_replay_cost("high-reuse-nr4", group_k_csc_nr4_stats);
    print_group_k_csc_replay_cost("high-reuse-nr8", group_k_csc_nr8_stats);

    std::vector<float> measured_row_direct(rows * cols, 0.0f);
    const ReplayTimingStats row_direct_timing = benchmark_replay_kernel(
        measured_row_direct,
        current_row_grouped,
        [&]() {
            ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar(
                args, scalar_route_plan, rows, cols, nullptr, entries, groups, group_offsets,
                group_row_group_indices, measured_row_direct.data());
            return true;
        },
        warmup_count,
        measured_count,
        ok);
    std::vector<float> measured_nr4(rows * cols, 0.0f);
    const ReplayTimingStats nr4_timing = benchmark_replay_kernel(
        measured_nr4,
        current_row_grouped,
        [&]() {
            return ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr4(
                args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
                measured_nr4.data(), group_k_csc_nr4_stats);
        },
        warmup_count,
        measured_count,
        ok);
    std::vector<float> measured_nr8(rows * cols, 0.0f);
    const ReplayTimingStats nr8_timing = benchmark_replay_kernel(
        measured_nr8,
        current_row_grouped,
        [&]() {
            return ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
                args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
                measured_nr8.data(), group_k_csc_nr8_stats);
        },
        warmup_count,
        measured_count,
        ok);
    const double best_group_k_csc_compute = std::min(
        nr4_timing.compute.median_us, nr8_timing.compute.median_us);
    const char *keep_decision = best_group_k_csc_compute <= row_direct_timing.compute.median_us ?
        "keep" : "revert";
    std::printf(
        "high-reuse benchmark warmup=%zu measured=%zu row_direct compute_us median=%.3f p10=%.3f p90=%.3f total_us median=%.3f p10=%.3f p90=%.3f\n",
        warmup_count, measured_count,
        row_direct_timing.compute.median_us, row_direct_timing.compute.p10_us, row_direct_timing.compute.p90_us,
        row_direct_timing.total.median_us, row_direct_timing.total.p10_us, row_direct_timing.total.p90_us);
    std::printf(
        "high-reuse benchmark warmup=%zu measured=%zu group_k_csc_nr4 compute_us median=%.3f p10=%.3f p90=%.3f total_us median=%.3f p10=%.3f p90=%.3f\n",
        warmup_count, measured_count,
        nr4_timing.compute.median_us, nr4_timing.compute.p10_us, nr4_timing.compute.p90_us,
        nr4_timing.total.median_us, nr4_timing.total.p10_us, nr4_timing.total.p90_us);
    std::printf(
        "high-reuse benchmark warmup=%zu measured=%zu group_k_csc_nr8 compute_us median=%.3f p10=%.3f p90=%.3f total_us median=%.3f p10=%.3f p90=%.3f\n",
        warmup_count, measured_count,
        nr8_timing.compute.median_us, nr8_timing.compute.p10_us, nr8_timing.compute.p90_us,
        nr8_timing.total.median_us, nr8_timing.total.p10_us, nr8_timing.total.p90_us);
    std::printf(
        "high-reuse benchmark decision=%s baseline=row-direct compare=group-k-csc-min-compute\n",
        keep_decision);
    return ok;
}

bool test_group_k_csc_nr4_transposed_j_tile() {
    using ggml::gemmini::quants::dec::ActiveRowGroup;
    using ggml::gemmini::quants::dec::GroupKCSCPlan;
    using ggml::gemmini::quants::dec::GroupKCSCScalarStats;
    using ggml::gemmini::quants::dec::ResidualGroupEntry;

    constexpr size_t rows = 3;
    constexpr size_t cols = 131;
    constexpr size_t depth = 35;
    constexpr size_t weight_stride = depth + 7;
    std::vector<int8_t> transposed_weights(cols * weight_stride, 0);
    for (size_t column = 0; column < cols; ++column)
        for (size_t k = 0; k < depth; ++k)
            transposed_weights[column * weight_stride + k] = static_cast<int8_t>(
                static_cast<int>((column * 3 + k * 5) % 17) - 8);

    std::vector<float> output(rows * cols, 0.0f);
    ggml_gemmini_args_t args = dense_args(rows, cols, depth, transposed_weights, output, 0.3125f);
    args.sB = weight_stride;
    args.transpose_B = true;
    const auto scalar_route_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        args, ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);

    constexpr int32_t fallback_positive = (1 << 24) + 1;
    const std::array<float, rows> activation_scales = { 0.5f, -0.75f, 1.25f };
    std::vector<ResidualGroupEntry> entries = {
        { 2, 34, 2 }, { 0, 0, 4 }, { 1, 33, fallback_positive }, { 0, 0, -1 },
        { 2, 0, -2 }, { 1, 33, 5 }, { 0, 34, 1 },
    };
    std::vector<ActiveRowGroup> groups;
    ggml::gemmini::quants::dec::build_active_row_groups(entries, groups);
    std::vector<size_t> group_offsets;
    std::vector<size_t> group_row_group_indices;
    constexpr size_t active_k_groups = 2;
    ggml::gemmini::quants::dec::build_group_major_index(
        groups, active_k_groups, group_offsets, group_row_group_indices);
    GroupKCSCPlan group_k_csc_plan;
    const bool group_k_csc_ready = ggml::gemmini::quants::dec::build_group_k_csc_plan(
        entries, groups, group_offsets, group_row_group_indices, active_k_groups, group_k_csc_plan);

    std::vector<float> scalar_output(rows * cols, 0.0f);
    GroupKCSCScalarStats scalar_stats;
    const bool scalar_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc(
            args, scalar_route_plan, rows, cols, activation_scales.data(), entries, group_k_csc_plan,
            scalar_output.data(), scalar_stats);
    std::vector<float> nr4_output(rows * cols, 0.0f);
    GroupKCSCScalarStats nr4_stats;
    const bool nr4_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc_nr4(
            args, scalar_route_plan, rows, cols, activation_scales.data(), entries, group_k_csc_plan,
            nr4_output.data(), nr4_stats);
    std::vector<float> nr8_output(rows * cols, 0.0f);
    GroupKCSCScalarStats nr8_stats;
    const bool nr8_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, activation_scales.data(), entries, group_k_csc_plan,
            nr8_output.data(), nr8_stats);
    const char *previous_threads = std::getenv("DEC_THREADS");
    const std::string saved_threads = previous_threads ? previous_threads : "";
    const bool had_previous_threads = previous_threads != nullptr;
    set_dec_threads("1");
    std::vector<float> mixed_output(rows * cols, 0.0f);
    GroupKCSCScalarStats mixed_stats;
    const bool mixed_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, activation_scales.data(), entries, group_k_csc_plan,
            mixed_output.data(), mixed_stats);
    std::vector<float> mixed_nr4_output(rows * cols, 0.0f);
    GroupKCSCScalarStats mixed_nr4_stats;
    const bool mixed_nr4_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr4(
            args, scalar_route_plan, rows, cols, activation_scales.data(), entries, group_k_csc_plan,
            mixed_nr4_output.data(), mixed_nr4_stats);
    bool mixed_threads_identical = true;
#if defined(GGML_GEMMINI_HAS_OPENMP)
    for (const char *thread_count : { "2", "3" }) {
        set_dec_threads(thread_count);
        std::vector<float> thread_output(rows * cols, 0.0f);
        GroupKCSCScalarStats thread_stats;
        mixed_threads_identical =
            ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
                args, scalar_route_plan, rows, cols, activation_scales.data(), entries, group_k_csc_plan,
                thread_output.data(), thread_stats) &&
            byte_identical(mixed_output, thread_output) &&
            thread_stats.logical_weight_reference_count == mixed_stats.logical_weight_reference_count &&
            thread_stats.weight_scalar_load_count == mixed_stats.weight_scalar_load_count &&
            thread_stats.thread_scratch_bytes == mixed_stats.thread_scratch_bytes &&
            thread_stats.int32_row_count == mixed_stats.int32_row_count &&
            thread_stats.int64_fallback_row_count == mixed_stats.int64_fallback_row_count &&
            mixed_threads_identical;
    }
#endif
    set_dec_threads(had_previous_threads ? saved_threads.c_str() : nullptr);

    constexpr size_t expected_logical_refs = 7 * cols;
    constexpr size_t expected_weight_loads = 3 * cols;
    constexpr size_t expected_scratch_bytes =
        rows * ggml::gemmini::quants::dec::kDecInt64JTileWidth * sizeof(int64_t);
    const bool ok = check(group_k_csc_ready && scalar_accumulated && nr4_accumulated && nr8_accumulated &&
                     mixed_accumulated && mixed_nr4_accumulated &&
                     mixed_threads_identical &&
                     scalar_route_plan.valid &&
                     scalar_route_plan.layout == ggml::gemmini::quants::dec::WeightLayout::JxK_ColMajor &&
                     scalar_route_plan.weight_stride == weight_stride &&
                     byte_identical(scalar_output, nr4_output) &&
                     byte_identical(scalar_output, nr8_output) &&
                     byte_identical(scalar_output, mixed_output) &&
                     byte_identical(scalar_output, mixed_nr4_output),
                 "GroupKCSC NR1/4/8 mixed matches transposed padded repeated-K J tail") &&
        check(scalar_stats.logical_weight_reference_count == expected_logical_refs &&
                  scalar_stats.weight_scalar_load_count == expected_weight_loads &&
                  scalar_stats.weight_vector_load_count == 0 &&
                  scalar_stats.classification_work_count == 0 &&
                  scalar_stats.scratch_init_count == rows * cols &&
                  scalar_stats.sparse_update_count == expected_logical_refs &&
                  scalar_stats.merge_count == rows * cols &&
                  scalar_stats.safe_update_count == 0 &&
                  scalar_stats.fallback_update_count == expected_logical_refs &&
                  scalar_stats.branch_entry_classification_count == 0 &&
                  scalar_stats.thread_scratch_bytes == expected_scratch_bytes &&
                  nr4_stats.logical_weight_reference_count == expected_logical_refs &&
                  nr4_stats.weight_scalar_load_count == expected_weight_loads &&
                  nr4_stats.weight_vector_load_count == 99 &&
                  nr4_stats.classification_work_count == 0 &&
                  nr4_stats.scratch_init_count == rows * cols &&
                  nr4_stats.sparse_update_count == expected_logical_refs &&
                  nr4_stats.merge_count == rows * cols &&
                  nr4_stats.safe_update_count == 0 &&
                  nr4_stats.fallback_update_count == expected_logical_refs &&
                  nr4_stats.branch_entry_classification_count == 0 &&
                  nr4_stats.thread_scratch_bytes == expected_scratch_bytes &&
                  nr8_stats.logical_weight_reference_count == expected_logical_refs &&
                  nr8_stats.weight_scalar_load_count == expected_weight_loads &&
                  nr8_stats.weight_vector_load_count == 51 &&
                  nr8_stats.classification_work_count == 0 &&
                  nr8_stats.scratch_init_count == rows * cols &&
                  nr8_stats.sparse_update_count == expected_logical_refs &&
                  nr8_stats.merge_count == rows * cols &&
                  nr8_stats.safe_update_count == 0 &&
                  nr8_stats.fallback_update_count == expected_logical_refs &&
                  nr8_stats.branch_entry_classification_count == 0 &&
                  nr8_stats.thread_scratch_bytes == expected_scratch_bytes &&
                  mixed_stats.logical_weight_reference_count == expected_logical_refs &&
                  mixed_stats.weight_scalar_load_count == expected_weight_loads &&
                  mixed_stats.weight_vector_load_count == 51 &&
                  mixed_stats.classification_work_count == entries.size() &&
                  mixed_stats.scratch_init_count == 4 * cols &&
                  mixed_stats.sparse_update_count == expected_logical_refs &&
                  mixed_stats.merge_count == rows * cols &&
                  mixed_stats.safe_update_count == 5 * cols &&
                  mixed_stats.fallback_update_count == 2 * cols &&
                  mixed_stats.branch_entry_classification_count == entries.size() &&
                  mixed_stats.thread_scratch_bytes ==
                      ggml::gemmini::quants::dec::kDecInt64JTileWidth *
                      (rows * sizeof(int32_t) + sizeof(int64_t)) &&
                  mixed_stats.int32_row_count == 2 &&
                  mixed_stats.int64_fallback_row_count == 1 &&
                  mixed_nr4_stats.logical_weight_reference_count == expected_logical_refs &&
                  mixed_nr4_stats.weight_scalar_load_count == expected_weight_loads &&
                  mixed_nr4_stats.weight_vector_load_count == 99 &&
                  mixed_nr4_stats.classification_work_count == entries.size() &&
                  mixed_nr4_stats.scratch_init_count == 4 * cols &&
                  mixed_nr4_stats.sparse_update_count == expected_logical_refs &&
                  mixed_nr4_stats.merge_count == rows * cols &&
                  mixed_nr4_stats.safe_update_count == 5 * cols &&
                  mixed_nr4_stats.fallback_update_count == 2 * cols &&
                  mixed_nr4_stats.branch_entry_classification_count == entries.size() &&
                  mixed_nr4_stats.thread_scratch_bytes ==
                      ggml::gemmini::quants::dec::kDecInt64JTileWidth *
                      (rows * sizeof(int32_t) + sizeof(int64_t)) &&
                  mixed_nr4_stats.int32_row_count == 2 &&
                  mixed_nr4_stats.int64_fallback_row_count == 1,
              "GroupKCSC NR4/8 transposed J-tile fallback counters") &&
        check_group_k_csc_stage_cycles(scalar_stats, false, "transposed scalar stage cycles") &&
        check_group_k_csc_stage_cycles(nr4_stats, false, "transposed nr4 stage cycles") &&
        check_group_k_csc_stage_cycles(nr8_stats, false, "transposed nr8 stage cycles") &&
        check_group_k_csc_stage_cycles(mixed_stats, true, "transposed mixed nr8 stage cycles") &&
        check_group_k_csc_stage_cycles(mixed_nr4_stats, true, "transposed mixed nr4 stage cycles");
    print_group_k_csc_replay_cost("one-fallback", mixed_stats);
    return ok;
}

bool test_group_k_csc_mixed_int32_boundaries() {
    using ggml::gemmini::quants::dec::ActiveRowGroup;
    using ggml::gemmini::quants::dec::GroupKCSCPlan;
    using ggml::gemmini::quants::dec::GroupKCSCScalarStats;
    using ggml::gemmini::quants::dec::ResidualGroupEntry;

    constexpr size_t rows = 6;
    constexpr size_t cols = 5;
    constexpr size_t depth = 1;
    constexpr int32_t safe_positive = 1 << 24;
    constexpr int32_t fallback_positive = safe_positive + 1;
    const std::vector<int8_t> weights = { -128, 127, -128, 127, 1 };
    std::vector<float> output(rows * cols, 0.0f);
    ggml_gemmini_args_t args = dense_args(rows, cols, depth, weights, output, 0.25f);
    const auto scalar_route_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        args, ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    std::vector<ResidualGroupEntry> entries = {
        { 0, 0, safe_positive },
        { 1, 0, fallback_positive },
        { 2, 0, -(safe_positive - 1) },
        { 3, 0, -safe_positive },
        { 4, 0, std::numeric_limits<int32_t>::min() },
        { 5, 0, fallback_positive },
        { 5, 0, -fallback_positive },
    };
    std::vector<ActiveRowGroup> groups;
    ggml::gemmini::quants::dec::build_active_row_groups(entries, groups);
    std::vector<size_t> group_offsets;
    std::vector<size_t> group_row_group_indices;
    ggml::gemmini::quants::dec::build_group_major_index(
        groups, 1, group_offsets, group_row_group_indices);
    GroupKCSCPlan group_k_csc_plan;
    const bool group_k_csc_ready = ggml::gemmini::quants::dec::build_group_k_csc_plan(
        entries, groups, group_offsets, group_row_group_indices, 1, group_k_csc_plan);

    std::vector<float> int64_output(rows * cols, 0.0f);
    GroupKCSCScalarStats int64_stats;
    const bool int64_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            int64_output.data(), int64_stats);
    std::vector<float> int64_nr4_output(rows * cols, 0.0f);
    GroupKCSCScalarStats int64_nr4_stats;
    const bool int64_nr4_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc_nr4(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            int64_nr4_output.data(), int64_nr4_stats);
    std::vector<float> mixed_output(rows * cols, 0.0f);
    GroupKCSCScalarStats mixed_stats;
    const bool mixed_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            mixed_output.data(), mixed_stats);
    std::vector<float> mixed_nr4_output(rows * cols, 0.0f);
    GroupKCSCScalarStats mixed_nr4_stats;
    const bool mixed_nr4_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr4(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            mixed_nr4_output.data(), mixed_nr4_stats);

    std::vector<ResidualGroupEntry> fallback_entries = {
        { 1, 0, fallback_positive }, { 3, 0, -safe_positive },
        { 4, 0, std::numeric_limits<int32_t>::min() },
        { 5, 0, fallback_positive }, { 5, 0, -fallback_positive },
    };
    std::vector<ActiveRowGroup> fallback_groups;
    ggml::gemmini::quants::dec::build_active_row_groups(fallback_entries, fallback_groups);
    std::vector<size_t> fallback_group_offsets;
    std::vector<size_t> fallback_group_row_group_indices;
    ggml::gemmini::quants::dec::build_group_major_index(
        fallback_groups, 1, fallback_group_offsets, fallback_group_row_group_indices);
    GroupKCSCPlan fallback_plan;
    const bool fallback_ready = ggml::gemmini::quants::dec::build_group_k_csc_plan(
        fallback_entries, fallback_groups, fallback_group_offsets, fallback_group_row_group_indices, 1,
        fallback_plan);
    std::vector<float> fallback_int64_output(rows * cols, 0.0f);
    std::vector<float> fallback_mixed_output(rows * cols, 0.0f);
    GroupKCSCScalarStats fallback_int64_stats;
    GroupKCSCScalarStats fallback_mixed_stats;
    const bool fallback_int64_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, fallback_entries, fallback_plan,
            fallback_int64_output.data(), fallback_int64_stats);
    std::vector<float> fallback_int64_nr4_output(rows * cols, 0.0f);
    GroupKCSCScalarStats fallback_int64_nr4_stats;
    const bool fallback_int64_nr4_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc_nr4(
            args, scalar_route_plan, rows, cols, nullptr, fallback_entries, fallback_plan,
            fallback_int64_nr4_output.data(), fallback_int64_nr4_stats);
    const bool fallback_mixed_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, fallback_entries, fallback_plan,
            fallback_mixed_output.data(), fallback_mixed_stats);
    std::vector<float> fallback_mixed_nr4_output(rows * cols, 0.0f);
    GroupKCSCScalarStats fallback_mixed_nr4_stats;
    const bool fallback_mixed_nr4_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr4(
            args, scalar_route_plan, rows, cols, nullptr, fallback_entries, fallback_plan,
            fallback_mixed_nr4_output.data(), fallback_mixed_nr4_stats);

    const bool ok = check(group_k_csc_ready && int64_accumulated && int64_nr4_accumulated &&
                     mixed_accumulated && mixed_nr4_accumulated &&
                     byte_identical(int64_output, int64_nr4_output) &&
                     byte_identical(int64_output, mixed_output) &&
                     byte_identical(int64_output, mixed_nr4_output),
                 "GroupKCSC NR4/8 mixed preserves INT64 boundary outputs") &&
        check(mixed_stats.logical_weight_reference_count == 7 * cols &&
                  mixed_stats.weight_scalar_load_count == cols &&
                  mixed_stats.weight_vector_load_count == 1 &&
                  mixed_stats.classification_work_count == entries.size() &&
                  mixed_stats.scratch_init_count == cols * (rows + 4) &&
                  mixed_stats.sparse_update_count == 7 * cols &&
                  mixed_stats.merge_count == cols * 6 &&
                  mixed_stats.safe_update_count == 2 * cols &&
                  mixed_stats.fallback_update_count == 5 * cols &&
                  mixed_stats.branch_entry_classification_count == entries.size() &&
                  mixed_stats.thread_scratch_bytes ==
                      cols * (rows * sizeof(int32_t) + 4 * sizeof(int64_t)) &&
                  mixed_stats.int32_row_count == 2 &&
                  mixed_stats.int64_fallback_row_count == 4 &&
                  mixed_nr4_stats.logical_weight_reference_count == 7 * cols &&
                  mixed_nr4_stats.weight_scalar_load_count == cols &&
                  mixed_nr4_stats.weight_vector_load_count == 2 &&
                  mixed_nr4_stats.classification_work_count == entries.size() &&
                  mixed_nr4_stats.scratch_init_count == cols * (rows + 4) &&
                  mixed_nr4_stats.sparse_update_count == 7 * cols &&
                  mixed_nr4_stats.merge_count == cols * 6 &&
                  mixed_nr4_stats.safe_update_count == 2 * cols &&
                  mixed_nr4_stats.fallback_update_count == 5 * cols &&
                  mixed_nr4_stats.branch_entry_classification_count == entries.size() &&
                  mixed_nr4_stats.thread_scratch_bytes ==
                      cols * (rows * sizeof(int32_t) + 4 * sizeof(int64_t)) &&
                  mixed_nr4_stats.int32_row_count == 2 &&
                  mixed_nr4_stats.int64_fallback_row_count == 4,
              "GroupKCSC NR4/8 classifies INT32 boundaries and same-K cancellation") &&
        check(fallback_ready && fallback_int64_accumulated && fallback_int64_nr4_accumulated &&
                  fallback_mixed_accumulated && fallback_mixed_nr4_accumulated &&
                  byte_identical(fallback_int64_output, fallback_int64_nr4_output) &&
                  byte_identical(fallback_int64_output, fallback_mixed_output) &&
                  byte_identical(fallback_int64_output, fallback_mixed_nr4_output) &&
                  fallback_mixed_stats.classification_work_count == fallback_entries.size() &&
                  fallback_mixed_stats.scratch_init_count == 4 * cols &&
                  fallback_mixed_stats.sparse_update_count == fallback_entries.size() * cols &&
                  fallback_mixed_stats.merge_count == 4 * cols &&
                  fallback_mixed_stats.safe_update_count == 0 &&
                  fallback_mixed_stats.fallback_update_count == fallback_entries.size() * cols &&
                  fallback_mixed_stats.branch_entry_classification_count == 0 &&
                  fallback_mixed_stats.width_path ==
                      ggml::gemmini::quants::dec::GroupKCSCWidthPath::AllInt64 &&
                  fallback_mixed_nr4_stats.classification_work_count == fallback_entries.size() &&
                  fallback_mixed_nr4_stats.scratch_init_count == 4 * cols &&
                  fallback_mixed_nr4_stats.sparse_update_count == fallback_entries.size() * cols &&
                  fallback_mixed_nr4_stats.merge_count == 4 * cols &&
                  fallback_mixed_nr4_stats.safe_update_count == 0 &&
                  fallback_mixed_nr4_stats.fallback_update_count == fallback_entries.size() * cols &&
                  fallback_mixed_nr4_stats.branch_entry_classification_count == 0 &&
                  fallback_mixed_nr4_stats.width_path ==
                      ggml::gemmini::quants::dec::GroupKCSCWidthPath::AllInt64,
              "GroupKCSC NR4/8 mixed selects AllInt64") &&
        check_group_k_csc_stage_cycles(int64_stats, false, "all-int64 scalar stage cycles") &&
        check_group_k_csc_stage_cycles(int64_nr4_stats, false, "all-int64 nr4 stage cycles") &&
        check_group_k_csc_stage_cycles(mixed_stats, true, "int32-boundary mixed nr8 stage cycles") &&
        check_group_k_csc_stage_cycles(mixed_nr4_stats, true, "int32-boundary mixed nr4 stage cycles") &&
        check_group_k_csc_stage_cycles(
            fallback_int64_stats, false, "fallback all-int64 scalar stage cycles") &&
        check_group_k_csc_stage_cycles(
            fallback_int64_nr4_stats, false, "fallback all-int64 nr4 stage cycles") &&
        check_group_k_csc_stage_cycles(
            fallback_mixed_stats, true, "fallback mixed all-int64 nr8 stage cycles") &&
        check_group_k_csc_stage_cycles(
            fallback_mixed_nr4_stats, true, "fallback mixed all-int64 nr4 stage cycles");
    print_group_k_csc_replay_cost("all-int64", fallback_mixed_stats);
    return ok;
}

bool test_group_k_csc_mixed_prefix_and_plan_rejects() {
    using ggml::gemmini::quants::dec::ActiveRowGroup;
    using ggml::gemmini::quants::dec::GroupKCSCPlan;
    using ggml::gemmini::quants::dec::GroupKCSCScalarStats;
    using ggml::gemmini::quants::dec::ResidualGroupEntry;

    constexpr size_t rows = 2;
    constexpr size_t cols = 5;
    constexpr size_t depth = 2;
    const std::vector<int8_t> weights = {
        -128, 127, -128, 127, 1,
        -128, 127, -128, 127, 1,
    };
    std::vector<float> output(rows * cols, 0.0f);
    ggml_gemmini_args_t args = dense_args(rows, cols, depth, weights, output, 0.25f);
    const auto scalar_route_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        args, ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    std::vector<ResidualGroupEntry> entries = {
        { 0, 0, 10 }, { 0, 0, -10 }, { 1, 0, 8000000 }, { 1, 1, 8000000 },
    };
    std::vector<ActiveRowGroup> groups;
    ggml::gemmini::quants::dec::build_active_row_groups(entries, groups);
    std::vector<size_t> group_offsets;
    std::vector<size_t> group_row_group_indices;
    ggml::gemmini::quants::dec::build_group_major_index(
        groups, 1, group_offsets, group_row_group_indices);
    GroupKCSCPlan group_k_csc_plan;
    const bool group_k_csc_ready = ggml::gemmini::quants::dec::build_group_k_csc_plan(
        entries, groups, group_offsets, group_row_group_indices, 1, group_k_csc_plan);

    std::vector<float> int64_output(rows * cols, 0.0f);
    GroupKCSCScalarStats int64_stats;
    const bool int64_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            int64_output.data(), int64_stats);
    std::vector<float> mixed_output(rows * cols, 0.0f);
    GroupKCSCScalarStats mixed_stats;
    const bool mixed_accumulated =
        ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            mixed_output.data(), mixed_stats);

    GroupKCSCPlan malformed_plan = group_k_csc_plan;
    const size_t first_column = malformed_plan.column_offsets[0];
    std::swap(malformed_plan.entry_order[first_column], malformed_plan.entry_order[first_column + 1]);
    const std::vector<float> untouched(rows * cols, 3.0f);
    std::vector<float> scalar_reject_output = untouched;
    std::vector<float> nr8_reject_output = untouched;
    std::vector<float> mixed_reject_output = untouched;
    GroupKCSCScalarStats scalar_reject_stats;
    GroupKCSCScalarStats nr8_reject_stats;
    GroupKCSCScalarStats mixed_reject_stats;
    const bool scalar_rejected =
        !ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc(
            args, scalar_route_plan, rows, cols, nullptr, entries, malformed_plan,
            scalar_reject_output.data(), scalar_reject_stats);
    const bool nr8_rejected =
        !ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, entries, malformed_plan,
            nr8_reject_output.data(), nr8_reject_stats);
    const bool mixed_rejected =
        !ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, entries, malformed_plan,
            mixed_reject_output.data(), mixed_reject_stats);

    return check(group_k_csc_ready && int64_accumulated && mixed_accumulated &&
                     byte_identical(int64_output, mixed_output) &&
                     mixed_stats.logical_weight_reference_count == entries.size() * cols &&
                     mixed_stats.weight_scalar_load_count == 2 * cols &&
                     mixed_stats.classification_work_count == entries.size() &&
                     mixed_stats.scratch_init_count == rows * cols &&
                     mixed_stats.sparse_update_count == entries.size() * cols &&
                     mixed_stats.merge_count == rows * cols &&
                     mixed_stats.safe_update_count == entries.size() * cols &&
                     mixed_stats.fallback_update_count == 0 &&
                     mixed_stats.branch_entry_classification_count == 0 &&
                     mixed_stats.thread_scratch_bytes == rows * cols * sizeof(int32_t) &&
                     mixed_stats.int32_row_count == rows && mixed_stats.int64_fallback_row_count == 0,
                 "GroupKCSC mixed NR8 accepts safe same-K cancellation and multi-K prefixes") &&
        check(scalar_rejected && nr8_rejected && mixed_rejected &&
                  scalar_reject_output == untouched && nr8_reject_output == untouched &&
                  mixed_reject_output == untouched,
              "GroupKCSC rejects reordered same-column entry_order");
}

bool test_group_k_csc_fallback_ratio_matrix() {
    using ggml::gemmini::quants::dec::ActiveRowGroup;
    using ggml::gemmini::quants::dec::GroupKCSCPlan;
    using ggml::gemmini::quants::dec::GroupKCSCScalarStats;
    using ggml::gemmini::quants::dec::GroupKCSCWidthPath;
    using ggml::gemmini::quants::dec::ResidualGroupEntry;

    constexpr size_t rows = 100;
    constexpr size_t cols = 8;
    constexpr size_t depth = 1;
    const std::vector<int8_t> weights(cols, 1);
    std::vector<float> scalar_output(rows * cols, 0.0f);
    const auto scalar_args = dense_args(rows, cols, depth, weights, scalar_output, 0.25f);
    const auto scalar_route_plan = ggml::gemmini::quants::dec::resolve_dec_route_plan(
        scalar_args, ggml::gemmini::quants::dec::WeightScaleInfoMode::Dec);
    bool ok = scalar_route_plan.valid;

    for (const size_t fallback_rows : { size_t {0}, size_t {1}, size_t {10}, size_t {50}, size_t {100} }) {
        std::vector<ResidualGroupEntry> entries;
        entries.reserve(rows);
        std::vector<ggml::gemmini::quants::QactOutlier> outliers;
        outliers.reserve(rows);
        for (size_t row = 0; row < rows; ++row) {
            const int32_t residual = row < fallback_rows ?
                std::numeric_limits<int32_t>::max() : 1;
            entries.push_back({ static_cast<uint32_t>(row), 0, residual });
            outliers.push_back({ static_cast<int32_t>(row), 0, residual });
        }
        std::vector<ActiveRowGroup> groups;
        ggml::gemmini::quants::dec::build_active_row_groups(entries, groups);
        std::vector<size_t> group_offsets;
        std::vector<size_t> group_row_group_indices;
        ggml::gemmini::quants::dec::build_group_major_index(
            groups, 1, group_offsets, group_row_group_indices);
        GroupKCSCPlan group_k_csc_plan;
        const bool plan_ready = ggml::gemmini::quants::dec::build_group_k_csc_plan(
            entries, groups, group_offsets, group_row_group_indices, 1, group_k_csc_plan);

        std::vector<float> int64_output(rows * cols, 0.0f);
        std::vector<float> mixed_output(rows * cols, 0.0f);
        ggml_gemmini_args_t args = scalar_args;
        args.f_out = int64_output.data();
        GroupKCSCScalarStats int64_stats;
        GroupKCSCScalarStats mixed_stats;
        const bool int64_ready = ggml::gemmini::quants::dec::accumulate_to_ycom_int64_scalar_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            int64_output.data(), int64_stats);
        args.f_out = mixed_output.data();
        const bool mixed_ready = ggml::gemmini::quants::dec::accumulate_to_ycom_int32_mixed_group_k_csc_nr8(
            args, scalar_route_plan, rows, cols, nullptr, entries, group_k_csc_plan,
            mixed_output.data(), mixed_stats);
        const double ratio = static_cast<double>(fallback_rows) / rows;
        std::printf(
            "fallback-ratio=%.2f width_path=%d int32_rows=%zu int64_rows=%zu scratch=%zu safe_updates=%zu fallback_updates=%zu\n",
            ratio, static_cast<int>(mixed_stats.width_path), mixed_stats.int32_row_count,
            mixed_stats.int64_fallback_row_count, mixed_stats.thread_scratch_bytes,
            mixed_stats.safe_update_count, mixed_stats.fallback_update_count);
        ok = check(plan_ready && int64_ready && mixed_ready &&
                       byte_identical(int64_output, mixed_output) &&
                       mixed_stats.int32_row_count == rows - fallback_rows &&
                       mixed_stats.int64_fallback_row_count == fallback_rows &&
                       mixed_stats.safe_update_count == (rows - fallback_rows) * cols &&
                       mixed_stats.fallback_update_count == fallback_rows * cols &&
                       mixed_stats.branch_entry_classification_count ==
                           (fallback_rows == 0 || fallback_rows == rows ? 0 : rows),
                   "fallback ratio matrix preserves output and path counters") && ok;
    }
    return ok;
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
    const bool ok = test_noop() && test_route_plan() && test_active_row_groups() && test_group_k_csc_plan() && test_route_metadata_rejects() && test_repeated_residuals() && test_decode_repeated_residuals() && test_integer_routes() && test_block_integer_route() &&
        test_q8_h1_hierarchical_route() && test_q8_h1_preserves_ordered_scaling_bits() && test_q8_h1_activation_scale_routes_preserve_bits() && test_unpacked_h1_tail_routes_preserve_bits() && test_q8_h1_large_effective_scale() && test_q8_h2_hierarchical_route() && test_q8_hp1_hierarchical_route() && test_q8_hp2_hierarchical_route() &&
        test_malformed_hierarchical_reject() && test_output_strides() && test_sparse_grouped_tails() && test_malformed_reject() && test_thread_clamp() &&
        test_q8_h1_group_k_csc_matches_row_direct() && test_q8_h1_group_k_csc_auto_dispatch() && test_group_k_csc_production_dispatch() && test_group_k_csc_k_and_j_tails() && test_fixed_residual_replay_baseline() &&
        test_fixed_residual_replay_high_reuse() && test_group_k_csc_nr4_transposed_j_tile() &&
        test_group_k_csc_mixed_int32_boundaries() &&
        test_group_k_csc_mixed_prefix_and_plan_rejects() &&
        test_group_k_csc_fallback_ratio_matrix() &&
        test_thread_determinism() && test_inside_existing_openmp_region();
    std::printf("gemmini DEC baseline: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
