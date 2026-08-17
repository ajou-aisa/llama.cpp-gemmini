#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/dispatch.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/quantize.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/types.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/token/types.hpp"

#include <ggml.h>
#ifndef GEMMINI_EXSIA_WRITER_TEST_ONLY
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp"
#include <gemmini.h>
#endif

#include <cstdio>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ggml::gemmini;

bool check(bool value, const char * message) {
    if (!value)
        std::fprintf(stderr, "FAIL: %s\n", message);
    return value;
}

#ifndef GEMMINI_EXSIA_WRITER_TEST_ONLY
bool test_exsia_baseline() {
    elem_t activation = 3;
    elem_t weight = 4;
    float output = 0.0f;
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = 1;
    args.A = &activation;
    args.B = &weight;
    args.f_out = &output;
    args.sA = 1;
    args.sB = 1;
    args.stride_f_out = 1;
    args.col_stride_f_out = 1;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    args.tiled_matmul_type = CPU;
    auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = { 0 };
    tiled_matmul_auto_baseline(&args, baseline_activation_quant_t::EXSIA,
                               baseline_weight_quant_t::TENSOR);
    return check(output == 12.0f, "ExSIA baseline output") &&
        check(meta.rmd_packets.empty(), "empty residual has no RMD packet");
}

bool test_dispatch_modes() {
    elem_t activation = 3;
    elem_t weight = 4;
    float output = 0.0f;
    ggml_gemmini_args_t args{};
    args.I = args.J = args.K = 1;
    args.A = &activation;
    args.B = &weight;
    args.f_out = &output;
    args.sA = args.sB = args.stride_f_out = args.col_stride_f_out = 1;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    args.tiled_matmul_type = CPU;
    auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = { 0 };
    const auto & packets = quants::act::rmd_packets(args);
    return check(packets.empty(), "dispatch exposes empty RMD packet list");
}

bool test_rmd_cpu_ws_routes() {
    elem_t weight = 4;
    ggml_gemmini_args_t args{};
    args.I = args.J = args.K = 1;
    args.B = &weight;
    args.sB = 1;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;

    rmd::RmdStripeBuilder builder;
    builder.reset(0, 0, 1, 1, 1);
    if (!builder.add_residual(0, 0, 256)) {
        return check(false, "RMD route packet input accepted");
    }
    const rmd::StripePacketHandle packet = builder.finish();
    if (!check(packet != nullptr, "RMD route packet built")) {
        return false;
    }

    rmd::CompressedOutput cpu_output;
    args.tiled_matmul_type = CPU;
    const rmd::RmdStatus cpu = rmd::execute_rmd_stripe(args, *packet, cpu_output);
    if (!check(cpu == rmd::RmdStatus::success && !cpu_output.values.empty() &&
               cpu_output.values.front() == 4, "RMD CPU compensation")) {
        return false;
    }

    rmd::CompressedOutput output;
    output.j_padded = 7;
    output.values = { 11, 22, 33 };
    const rmd::CompressedOutput unchanged = output;
    args.tiled_matmul_type = OS;
    const bool os_rejected = rmd::execute_rmd_stripe(args, *packet, output) ==
        rmd::RmdStatus::unsupported_route;
    const bool os_preserved_output = output.j_padded == unchanged.j_padded &&
        output.values == unchanged.values;
    args.tiled_matmul_type = WS;
    const rmd::RmdStatus ws = rmd::execute_rmd_stripe(args, *packet, output);
#if defined(__riscv)
    const bool ws_result = ws == rmd::RmdStatus::success && output.values == cpu_output.values;
#else
    const bool ws_result = ws == rmd::RmdStatus::unsupported_route &&
        output.j_padded == unchanged.j_padded && output.values == unchanged.values;
#endif
    return check(os_rejected, "RMD OS route rejected") &&
        check(os_preserved_output, "RMD failure preserves caller output") &&
        check(ws_result, "RMD WS route matches CPU on FPGA or rejects unsupported host");
}

bool test_rmd_cpu_direct_parity() {
    constexpr size_t rows = 17;
    constexpr size_t columns = 3;
    constexpr size_t logical_k = 65;
    constexpr size_t native_block_size = QK8_0;
    constexpr size_t native_blocks_per_row =
        (logical_k + native_block_size - 1) / native_block_size;

    std::vector<int32_t> residuals(rows * logical_k, 0);
    residuals[0 * logical_k + 0] = 128;
    residuals[0 * logical_k + 31] = -129;
    residuals[0 * logical_k + 32] = 65536;
    residuals[0 * logical_k + 64] = 256;
    residuals[1 * logical_k + 15] = -256;
    residuals[1 * logical_k + 47] = 16777216;
    residuals[1 * logical_k + 63] = 129;
    residuals[16 * logical_k + 1] = -65536;
    residuals[16 * logical_k + 33] = 16777217;

    std::vector<int8_t> baseline_activation(rows * logical_k);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t k = 0; k < logical_k; ++k) {
            baseline_activation[row * logical_k + k] =
                static_cast<int8_t>(static_cast<int>((row * 13 + k * 5) % 255) - 127);
        }
    }

    std::vector<block_q8_h1> weights(columns * native_blocks_per_row);
    for (size_t j = 0; j < columns; ++j) {
        for (size_t block_id = 0; block_id < native_blocks_per_row; ++block_id) {
            block_q8_h1 & block = weights[j * native_blocks_per_row + block_id];
            for (size_t local_k = 0; local_k < native_block_size; ++local_k) {
                const int value = static_cast<int>((j + 1) * 17 + block_id * 29 + local_k * 7);
                block.qs[local_k] = static_cast<int8_t>(value % 255 - 127);
            }
            block.c_b = static_cast<uint8_t>(2 + j + 3 * block_id);
            block.s_rf = 1.0f;
            block.R = static_cast<uint16_t>(5 + j);
        }
    }

    ggml_gemmini_args_t args{};
    args.I = rows;
    args.J = columns;
    args.K = logical_k;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = weights.data();
    args.q8_h1_block_count = weights.size();
    args.q8_h1_rows = columns;
    args.blocks_per_row = native_blocks_per_row;
    args.block_size_k = native_block_size;
    args.tiled_matmul_type = CPU;
    auto & meta = args.act_quant.storage().emplace<quants::act::exsia::Meta>();
    meta.theta = { 0 };

    rmd::RmdStripeBuilder builder;
    builder.reset(0, 0, rows, logical_k, columns);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t k = 0; k < logical_k; ++k) {
            const int32_t residual = residuals[row * logical_k + k];
            if (residual != 0 && !builder.add_residual(row, k, residual)) {
                return check(false, "RMD direct-parity residual accepted");
            }
        }
    }
    const rmd::StripePacketHandle packet = builder.finish();
    if (!check(packet != nullptr, "RMD direct-parity packet built")) {
        return false;
    }

    rmd::CompressedOutput compressed;
    rmd::RmdExecutionMetrics metrics{};
    const rmd::RmdStatus execution_status =
        rmd::execute_rmd_stripe(args, *packet, compressed, &metrics);
    if (execution_status != rmd::RmdStatus::success) {
        std::fprintf(stderr, "FAIL: RMD direct-parity CPU execution: %s\n",
                     rmd::rmd_status_message(execution_status));
        return false;
    }
    size_t expected_calls = 0;
    size_t expected_i_tiles = 0;
    const size_t j_tiles = (packet->logical_j + rmd::kArrayDim - 1) / rmd::kArrayDim;
    for (const rmd::BlockDescriptor & block : packet->blocks) {
        const size_t calls = j_tiles * (block.padded_k_count / rmd::kArrayDim);
        expected_calls += calls;
        expected_i_tiles += calls * block.active_lane_count *
            (block.rows_padded / rmd::kArrayDim);
    }
    if (!check(expected_i_tiles > expected_calls,
               "RMD reuse fixture spans multiple stacked I tiles") ||
        !check(metrics.matmul_call_count == expected_calls,
               "RMD executor coalesces lane/M tiles into one weight-loading call") ||
        !check(metrics.baseline_stacked_i_tile_count == expected_i_tiles,
               "RMD executor reports baseline stacked I tiles") ||
        !check(metrics.stacked_i_tile_count <= expected_i_tiles,
               "RMD lane partition never increases stacked I tiles") ||
        !check(metrics.weight_values_gathered ==
                   metrics.weight_baseline_address_resolutions &&
                   metrics.weight_address_resolutions <
                       metrics.weight_baseline_address_resolutions,
               "RMD executor reports reduced production weight addressing")) {
        return false;
    }
    std::printf("RMD stacked schedule: matmul_calls=%zu lane_groups=%zu "
                "stacked_i_tiles=%zu/%zu avoided_B_loads=%zu\n",
                metrics.matmul_call_count, metrics.lane_group_count,
                metrics.stacked_i_tile_count, metrics.baseline_stacked_i_tile_count,
                metrics.stacked_i_tile_count - metrics.matmul_call_count);
    std::vector<rmd::OutputValue> actual;
    if (!check(rmd::compose_rmd_output(*packet, compressed, actual) == rmd::RmdStatus::success,
               "RMD direct-parity composition")) {
        return false;
    }

    std::vector<int64_t> expected(rows * columns, 0);
    std::vector<int64_t> baseline(rows * columns, 0);
    std::vector<int64_t> direct_full(rows * columns, 0);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t j = 0; j < columns; ++j) {
            for (size_t k = 0; k < logical_k; ++k) {
                const int32_t residual = residuals[row * logical_k + k];
                const block_q8_h1 & block =
                    weights[j * native_blocks_per_row + k / native_block_size];
                const int64_t scaled_weight = static_cast<int64_t>(block.qs[k % native_block_size]) *
                    static_cast<int64_t>(static_cast<uint64_t>(block.c_b) + block.R);
                const size_t index = row * columns + j;
                expected[index] += static_cast<int64_t>(residual) * scaled_weight;
                baseline[index] += static_cast<int64_t>(baseline_activation[row * logical_k + k]) *
                    scaled_weight;
                direct_full[index] +=
                    (static_cast<int64_t>(baseline_activation[row * logical_k + k]) + residual) *
                    scaled_weight;
            }
        }
    }

    bool exact = actual == expected;
    std::printf("RMD CPU exact parity: M=%zu J=%zu K=%zu rmd_block=%zu\n",
                rows, columns, logical_k, rmd::kBlockSize);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t j = 0; j < columns; ++j) {
            const size_t index = row * columns + j;
            const int64_t compensated_full = baseline[index] + actual[index];
            std::printf("  C[%zu,%zu] residual_direct=%lld compensation=%lld "
                        "baseline=%lld direct_full=%lld baseline+comp=%lld\n", row, j,
                        static_cast<long long>(expected[index]),
                        static_cast<long long>(actual[index]),
                        static_cast<long long>(baseline[index]),
                        static_cast<long long>(direct_full[index]),
                        static_cast<long long>(compensated_full));
            exact = exact && compensated_full == direct_full[index];
        }
    }
    if (!check(exact, "direct matmul equals baseline plus RMD CPU compensation")) {
        return false;
    }

    std::vector<float> merged(rows * columns, 0.0f);
    args.f_out = merged.data();
    args.stride_f_out = columns;
    args.col_stride_f_out = 1;
    if (!check(rmd::merge_rmd_correction(args, *packet, actual) == rmd::RmdStatus::success,
               "RMD direct-parity merge")) {
        return false;
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        if (!check(merged[index] == static_cast<float>(expected[index]),
                   "merged RMD correction equals direct matmul")) {
            return false;
        }
    }

    merged.assign(rows * columns, 7.0f);
    const std::vector<float> unchanged = merged;
    auto & invalid_meta = args.act_quant.storage().emplace<quants::act::token::Meta>();
    invalid_meta.scales.assign(rows, 1.0f);
    invalid_meta.scales[1] = std::numeric_limits<float>::quiet_NaN();
    if (!check(rmd::merge_rmd_correction(args, *packet, actual) ==
                   rmd::RmdStatus::invalid_arguments && merged == unchanged,
               "RMD merge failure preserves caller output")) {
        return false;
    }
    return true;
}

bool test_rmd_lane_partition() {
    constexpr size_t logical_k = 32;
    std::vector<int32_t> residuals(logical_k, 0);
    for (size_t k = 0; k < 10; ++k) {
        residuals[k] = 257;
        residuals[16 + k] = 65536;
    }

    block_q8_h1 weights{};
    for (size_t k = 0; k < logical_k; ++k) {
        weights.qs[k] = static_cast<int8_t>(k + 1);
    }
    weights.c_b = 1;
    weights.s_rf = 1.0f;
    weights.R = 0;

    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = logical_k;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = &weights;
    args.q8_h1_block_count = 1;
    args.q8_h1_rows = 1;
    args.blocks_per_row = 1;
    args.block_size_k = logical_k;
    args.tiled_matmul_type = CPU;

    rmd::RmdStripeBuilder builder;
    builder.reset(0, 0, 1, logical_k, 1);
    for (size_t k = 0; k < logical_k; ++k) {
        if (residuals[k] != 0 && !builder.add_residual(0, k, residuals[k])) {
            return check(false, "RMD lane-partition residual accepted");
        }
    }
    const rmd::StripePacketHandle packet = builder.finish();
    if (!check(packet != nullptr, "RMD lane-partition packet built")) {
        return false;
    }

    rmd::CompressedOutput compressed;
    rmd::RmdExecutionMetrics metrics{};
    if (!check(rmd::execute_rmd_stripe(args, *packet, compressed, &metrics) ==
                   rmd::RmdStatus::success,
               "RMD lane-partition execution")) {
        return false;
    }
    std::vector<rmd::OutputValue> actual;
    if (!check(rmd::compose_rmd_output(*packet, compressed, actual) ==
                   rmd::RmdStatus::success,
               "RMD lane-partition composition")) {
        return false;
    }

    int64_t expected = 0;
    for (size_t k = 0; k < logical_k; ++k) {
        expected += static_cast<int64_t>(residuals[k]) * weights.qs[k];
    }
    return check(actual == std::vector<rmd::OutputValue>{expected},
                 "RMD lane partition preserves exact output") &&
        check(metrics.matmul_call_count == 2,
              "RMD lane partition preserves B-load count") &&
        check(metrics.active_lanes == 3 && metrics.lane_group_count == 2,
              "RMD lane partition groups overlapping and separates disjoint lanes") &&
        check(metrics.baseline_stacked_i_tile_count == 6 &&
                  metrics.stacked_i_tile_count == 3,
              "RMD lane partition halves padded I-by-K tiles");
}

bool test_rmd_weight_gather() {
    constexpr size_t logical_k = 65;
    constexpr size_t columns = 3;
    constexpr size_t blocks_per_row = 3;
    std::vector<block_q8_h1> h1(columns * blocks_per_row);
    for (size_t j = 0; j < columns; ++j) {
        for (size_t block = 0; block < blocks_per_row; ++block) {
            for (size_t k = 0; k < QK8_0; ++k) {
                h1[j * blocks_per_row + block].qs[k] =
                    static_cast<int8_t>(j * 37 + block * 11 + k - 64);
            }
        }
    }

    ggml_gemmini_args_t args{};
    args.J = columns;
    args.K = logical_k;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = h1.data();
    args.q8_h1_block_count = h1.size();
    args.q8_h1_rows = columns;
    args.blocks_per_row = blocks_per_row;
    args.block_size_k = QK8_0;

    const std::array<uint16_t, 3> local_k = { 0, 15, 31 };
    std::array<int8_t, rmd::kArrayDim * rmd::kArrayDim> tile{};
    tile.fill(0x55);
    rmd::RmdExecutionMetrics metrics{};
    if (!check(rmd::gather_weight_tile_for_test(
                   args, 1, local_k.data(), local_k.size(), 0, columns,
                   tile.data(), rmd::kArrayDim, &metrics) == rmd::RmdStatus::success,
               "native H1 gather succeeds")) {
        return false;
    }
    for (size_t k = 0; k < local_k.size(); ++k) {
        for (size_t j = 0; j < columns; ++j) {
            const int8_t expected = h1[j * blocks_per_row + 1].qs[local_k[k]];
            if (!check(tile[k * rmd::kArrayDim + j] == expected,
                       "native H1 gather matches scalar layout")) {
                return false;
            }
        }
    }
    if (!check(metrics.weight_values_gathered == local_k.size() * columns &&
               metrics.weight_baseline_address_resolutions == local_k.size() * columns &&
               metrics.weight_address_resolutions == columns,
               "native H1 gather reports row-base resolution counts")) {
        return false;
    }

    const std::array<std::pair<uint32_t, uint16_t>, 3> h1_boundaries = {
        std::pair<uint32_t, uint16_t>{0, 31},
        std::pair<uint32_t, uint16_t>{1, 0},
        std::pair<uint32_t, uint16_t>{2, 0},
    };
    for (const auto & [block_id, block_local_k] : h1_boundaries) {
        tile.fill(0x55);
        if (!check(rmd::gather_weight_tile_for_test(
                       args, block_id, &block_local_k, 1, 0, 1,
                       tile.data(), rmd::kArrayDim) == rmd::RmdStatus::success &&
                       tile[0] == h1[block_id].qs[block_local_k],
                   "native H1 gather preserves K=31/32/64 boundaries")) {
            return false;
        }
    }

    const auto unchanged_tile = tile;
    const auto unchanged_metrics = metrics;
    if (!check(rmd::gather_weight_tile_for_test(
                   args, 3, local_k.data(), local_k.size(), 0, columns,
                   tile.data(), rmd::kArrayDim, &metrics) == rmd::RmdStatus::execution_failed &&
               tile == unchanged_tile &&
               metrics.weight_values_gathered == unchanged_metrics.weight_values_gathered &&
               metrics.weight_baseline_address_resolutions ==
                   unchanged_metrics.weight_baseline_address_resolutions &&
               metrics.weight_address_resolutions == unchanged_metrics.weight_address_resolutions,
               "failed H1 gather preserves destination and metrics")) {
        return false;
    }

    constexpr size_t dense_j = 5;
    constexpr size_t dense_k = 40;
    constexpr size_t jxk_stride = dense_k + 7;
    constexpr size_t kxj_stride = dense_j + 9;
    std::vector<int8_t> dense_jxk(dense_j * jxk_stride, 0);
    std::vector<int8_t> dense_kxj(dense_k * kxj_stride, 0);
    for (size_t j = 0; j < dense_j; ++j) {
        for (size_t k = 0; k < dense_k; ++k) {
            const int8_t value = static_cast<int8_t>(j * 17 + k - 63);
            dense_jxk[j * jxk_stride + k] = value;
            dense_kxj[k * kxj_stride + j] = value;
        }
    }
    const std::array<uint16_t, 3> dense_local_k = { 0, 3, 7 };

    args = {};
    args.J = dense_j;
    args.K = dense_k;
    args.B = dense_jxk.data();
    args.sB = jxk_stride;
    args.transpose_B = true;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    tile.fill(0x55);
    metrics = {};
    if (!check(rmd::gather_weight_tile_for_test(
                   args, 1, dense_local_k.data(), dense_local_k.size(), 1, 3,
                   tile.data(), rmd::kArrayDim, &metrics) == rmd::RmdStatus::success,
               "dense JxK gather succeeds")) {
        return false;
    }
    for (size_t k = 0; k < dense_local_k.size(); ++k) {
        const size_t global_k = QK8_0 + dense_local_k[k];
        for (size_t col = 0; col < 3; ++col) {
            if (!check(tile[k * rmd::kArrayDim + col] ==
                           dense_jxk[(1 + col) * jxk_stride + global_k],
                       "dense JxK gather preserves padded stride")) {
                return false;
            }
        }
    }
    if (!check(metrics.weight_address_resolutions == 3,
               "dense JxK resolves once per valid column")) {
        return false;
    }

    const auto valid_jxk_tile = tile;
    const auto valid_jxk_metrics = metrics;
    args.sB = dense_k - 1;
    if (!check(rmd::gather_weight_tile_for_test(
                   args, 1, dense_local_k.data(), dense_local_k.size(), 1, 3,
                   tile.data(), rmd::kArrayDim, &metrics) == rmd::RmdStatus::execution_failed &&
                   tile == valid_jxk_tile &&
                   metrics.weight_values_gathered == valid_jxk_metrics.weight_values_gathered &&
                   metrics.weight_address_resolutions == valid_jxk_metrics.weight_address_resolutions,
               "invalid dense stride preserves destination and metrics")) {
        return false;
    }

    args.B = dense_kxj.data();
    args.sB = kxj_stride;
    args.transpose_B = false;
    tile.fill(0x55);
    metrics = {};
    if (!check(rmd::gather_weight_tile_for_test(
                   args, 1, dense_local_k.data(), dense_local_k.size(), 1, 3,
                   tile.data(), rmd::kArrayDim, &metrics) == rmd::RmdStatus::success,
               "dense KxJ gather succeeds")) {
        return false;
    }
    for (size_t k = 0; k < dense_local_k.size(); ++k) {
        const size_t global_k = QK8_0 + dense_local_k[k];
        for (size_t col = 0; col < 3; ++col) {
            if (!check(tile[k * rmd::kArrayDim + col] ==
                           dense_kxj[global_k * kxj_stride + 1 + col],
                       "dense KxJ gather preserves padded stride")) {
                return false;
            }
        }
    }
    return check(metrics.weight_values_gathered == dense_local_k.size() * 3 &&
                 metrics.weight_baseline_address_resolutions == dense_local_k.size() * 3 &&
                 metrics.weight_address_resolutions == dense_local_k.size(),
                 "dense KxJ resolves once per valid K row");
}

struct GatherBenchResult {
    std::string layout;
    size_t iterations = 0;
    uint64_t scalar_checksum = 0;
    uint64_t candidate_checksum = 0;
    double scalar_median_ns_per_tile = 0.0;
    double candidate_median_ns_per_tile = 0.0;
    double candidate_ratio = 0.0;
    double min_batch_ms = 0.0;
    size_t baseline_resolutions_per_tile = 0;
    size_t candidate_resolutions_per_tile = 0;
    bool checksum_match = false;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

GatherBenchResult benchmark_gather_layout(const std::string & layout,
                                          const ggml_gemmini_args_t & args,
                                          uint32_t block_count,
                                          const std::array<uint16_t, 10> & local_k) {
    using Clock = std::chrono::steady_clock;
    constexpr double minimum_batch_ns = 100000000.0;
    auto run = [&](bool scalar, size_t iterations, uint64_t & checksum,
                   rmd::RmdExecutionMetrics & metrics) {
        const auto start = Clock::now();
        const rmd::RmdStatus status = scalar
            ? rmd::repeat_scalar_weight_tile_gather_for_test(
                  args, block_count, local_k.data(), local_k.size(), 0, args.J,
                  iterations, checksum)
            : rmd::repeat_weight_tile_gather_for_test(
                  args, block_count, local_k.data(), local_k.size(), 0, args.J,
                  iterations, checksum, metrics);
        const auto stop = Clock::now();
        if (status != rmd::RmdStatus::success) {
            return -1.0;
        }
        return std::chrono::duration<double, std::nano>(stop - start).count();
    };

    size_t iterations = 1024;
    for (;;) {
        uint64_t scalar_checksum = 0;
        uint64_t candidate_checksum = 0;
        rmd::RmdExecutionMetrics metrics{};
        const double scalar_ns = run(true, iterations, scalar_checksum, metrics);
        const double candidate_ns = run(false, iterations, candidate_checksum, metrics);
        if (scalar_ns >= minimum_batch_ns && candidate_ns >= minimum_batch_ns) {
            break;
        }
        iterations *= 2;
    }

    std::vector<double> scalar_batches;
    std::vector<double> candidate_batches;
    uint64_t scalar_checksum = 0;
    uint64_t candidate_checksum = 0;
    rmd::RmdExecutionMetrics candidate_metrics{};
    for (size_t batch = 0; batch < 7; ++batch) {
        uint64_t batch_scalar_checksum = 0;
        uint64_t batch_candidate_checksum = 0;
        rmd::RmdExecutionMetrics batch_metrics{};
        const double scalar_ns = run(true, iterations, batch_scalar_checksum, batch_metrics);
        const double candidate_ns = run(false, iterations, batch_candidate_checksum, batch_metrics);
        scalar_batches.push_back(scalar_ns);
        candidate_batches.push_back(candidate_ns);
        scalar_checksum = batch_scalar_checksum;
        candidate_checksum = batch_candidate_checksum;
        candidate_metrics = batch_metrics;
    }

    const double tiles_per_batch = static_cast<double>(iterations) * block_count;
    const double scalar_median = median(scalar_batches) / tiles_per_batch;
    const double candidate_median = median(candidate_batches) / tiles_per_batch;
    GatherBenchResult result;
    result.layout = layout;
    result.iterations = iterations;
    result.scalar_checksum = scalar_checksum;
    result.candidate_checksum = candidate_checksum;
    result.scalar_median_ns_per_tile = scalar_median;
    result.candidate_median_ns_per_tile = candidate_median;
    result.candidate_ratio = candidate_median / scalar_median;
    result.min_batch_ms = std::min(*std::min_element(scalar_batches.begin(), scalar_batches.end()),
                                   *std::min_element(candidate_batches.begin(), candidate_batches.end())) /
        1000000.0;
    result.baseline_resolutions_per_tile =
        candidate_metrics.weight_baseline_address_resolutions /
        (iterations * block_count);
    result.candidate_resolutions_per_tile =
        candidate_metrics.weight_address_resolutions /
        (iterations * block_count);
    result.checksum_match = scalar_checksum == candidate_checksum;
    return result;
}

bool run_rmd_gather_benchmark(const std::filesystem::path & json_path,
                              double max_h1_ratio) {
    constexpr size_t columns = 16;
    constexpr size_t logical_k = 64;
    constexpr uint32_t block_count = 2;
    constexpr std::array<uint16_t, 10> local_k = { 0, 3, 7, 9, 12, 16, 21, 24, 27, 31 };

    std::vector<block_q8_h1> h1(columns * block_count);
    for (size_t j = 0; j < columns; ++j) {
        for (size_t block = 0; block < block_count; ++block) {
            for (size_t k = 0; k < QK8_0; ++k) {
                h1[j * block_count + block].qs[k] =
                    static_cast<int8_t>(j * 19 + block * 7 + k - 96);
            }
        }
    }
    ggml_gemmini_args_t args{};
    args.J = columns;
    args.K = logical_k;
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h1;
    args.q8_h1_blocks = h1.data();
    args.q8_h1_block_count = h1.size();
    args.q8_h1_rows = columns;
    args.blocks_per_row = block_count;
    args.block_size_k = QK8_0;

    std::vector<GatherBenchResult> results;
    results.push_back(benchmark_gather_layout("q8_h1", args, block_count, local_k));

    constexpr size_t jxk_stride = logical_k + 7;
    constexpr size_t kxj_stride = columns + 9;
    std::vector<int8_t> dense_jxk(columns * jxk_stride, 0);
    std::vector<int8_t> dense_kxj(logical_k * kxj_stride, 0);
    for (size_t j = 0; j < columns; ++j) {
        for (size_t k = 0; k < logical_k; ++k) {
            const int8_t value = static_cast<int8_t>(j * 19 + k - 96);
            dense_jxk[j * jxk_stride + k] = value;
            dense_kxj[k * kxj_stride + j] = value;
        }
    }
    args = {};
    args.J = columns;
    args.K = logical_k;
    args.B = dense_jxk.data();
    args.sB = jxk_stride;
    args.transpose_B = true;
    args.weight_i8_scale_active = true;
    args.weight_scale = 1.0f;
    results.push_back(benchmark_gather_layout("dense_jxk", args, block_count, local_k));
    args.B = dense_kxj.data();
    args.sB = kxj_stride;
    args.transpose_B = false;
    results.push_back(benchmark_gather_layout("dense_kxj", args, block_count, local_k));

    size_t baseline_resolutions = 0;
    size_t candidate_resolutions = 0;
    bool checksums_match = true;
    bool batches_long_enough = true;
    for (const GatherBenchResult & result : results) {
        baseline_resolutions += result.baseline_resolutions_per_tile;
        candidate_resolutions += result.candidate_resolutions_per_tile;
        checksums_match = checksums_match && result.checksum_match;
        batches_long_enough = batches_long_enough && result.min_batch_ms >= 100.0;
    }
    const double address_reduction = 1.0 -
        static_cast<double>(candidate_resolutions) / baseline_resolutions;
    const bool passed = checksums_match && batches_long_enough &&
        results.front().candidate_ratio <= max_h1_ratio && address_reduction >= 0.70;

    std::ostringstream json;
    json << std::fixed << std::setprecision(3)
         << "{\n  \"record_type\": \"RMD_GATHER_BENCHMARK\",\n"
         << "  \"fixture\": {\"selected_k_per_block\": 10, \"j\": 16, \"block_count\": 2},\n"
         << "  \"max_h1_ratio\": " << max_h1_ratio << ",\n"
         << "  \"address_reduction\": " << address_reduction << ",\n"
         << "  \"checksums_match\": " << (checksums_match ? "true" : "false") << ",\n"
         << "  \"batches_long_enough\": " << (batches_long_enough ? "true" : "false") << ",\n"
         << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
         << "  \"failed_field\": \""
         << (!checksums_match ? "checksum" : !batches_long_enough ? "batch_duration" :
             results.front().candidate_ratio > max_h1_ratio ? "h1_ratio" :
             address_reduction < 0.70 ? "address_reduction" : "") << "\",\n"
         << "  \"layouts\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const GatherBenchResult & result = results[i];
        json << "    {\"layout\": \"" << result.layout
             << "\", \"iterations\": " << result.iterations
             << ", \"scalar_checksum\": " << result.scalar_checksum
             << ", \"candidate_checksum\": " << result.candidate_checksum
             << ", \"checksum_match\": " << (result.checksum_match ? "true" : "false")
             << ", \"scalar_median_ns_per_tile\": " << result.scalar_median_ns_per_tile
             << ", \"candidate_median_ns_per_tile\": " << result.candidate_median_ns_per_tile
             << ", \"candidate_ratio\": " << result.candidate_ratio
             << ", \"min_batch_ms\": " << result.min_batch_ms
             << ", \"baseline_address_resolutions_per_tile\": "
             << result.baseline_resolutions_per_tile
             << ", \"candidate_address_resolutions_per_tile\": "
             << result.candidate_resolutions_per_tile << "}"
             << (i + 1 == results.size() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";

    if (!json_path.empty()) {
        std::ofstream output(json_path);
        if (!output) {
            std::fprintf(stderr, "failed to open benchmark JSON: %s\n", json_path.c_str());
            return false;
        }
        output << json.str();
    }
    std::printf("%s", json.str().c_str());
    return passed;
}
#endif

bool profile_output_routing(const std::filesystem::path & expected, bool invalid_parent) {
    if (invalid_parent) {
        std::ofstream("blocked") << "not a directory";
    }

    std::vector<float> source(64);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = index % 4 == 0 ? -0.5f : 0.5f;
    }
    std::vector<elem_t> quantized(64);
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.data = source.data();
    ggml_gemmini_args_t args{};
    args.I = 1;
    args.J = 1;
    args.K = source.size();
    args.A = quantized.data();
    args.sA = source.size();

    const bool wrote = quants::quantize_activation(&tensor, args);
    if (invalid_parent) {
        return check(!wrote, "invalid ExSIA profile parent reports existing writer failure") &&
            check(!std::filesystem::exists(expected), "invalid ExSIA profile path is absent") &&
            check(!std::filesystem::exists("log/exsia-cycle-detail.jsonl"), "invalid ExSIA profile does not fall back to legacy log");
    }

    std::ifstream input(expected);
    std::string line;
    const bool parsed = std::getline(input, line) && !line.empty() && line.front() == '{' &&
        line.back() == '}' && line.find("\"record_type\":\"TIMELINE\"") != std::string::npos;
    return check(wrote, "ExSIA profile writer succeeds") &&
        check(parsed, "ExSIA profile writer emits non-empty JSONL") &&
        check(!std::filesystem::exists("log/exsia-cycle-detail.jsonl"), "ExSIA profile writer does not create legacy log");
}

}

int main(int argc, char ** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--profile-output") {
        const bool invalid_parent = argc == 4 && std::string(argv[3]) == "--invalid-parent";
        return profile_output_routing(argv[2], invalid_parent) ? 0 : 1;
    }
#ifdef GEMMINI_EXSIA_WRITER_TEST_ONLY
    return 1;
#else
    if (argc >= 2 && std::string(argv[1]) == "--bench-rmd-gather") {
        std::filesystem::path json_path;
        double max_h1_ratio = 0.80;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--json" && i + 1 < argc) {
                json_path = argv[++i];
            } else if (std::string(argv[i]) == "--max-h1-ratio" && i + 1 < argc) {
                char * end = nullptr;
                max_h1_ratio = std::strtod(argv[++i], &end);
                if (end == argv[i] || *end != '\0') {
                    return 2;
                }
            } else {
                std::fprintf(stderr,
                             "usage: %s --bench-rmd-gather [--json path] [--max-h1-ratio value]\n",
                             argv[0]);
                return 2;
            }
        }
        return run_rmd_gather_benchmark(json_path, max_h1_ratio) ? 0 : 1;
    }

    std::string case_name = "all";
    if (argc == 2 && std::string(argv[1]).rfind("--case=", 0) == 0) {
        case_name = std::string(argv[1]).substr(7);
    } else if (argc == 3 && std::string(argv[1]) == "--case") {
        case_name = argv[2];
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--case=<name>|--case <name>]\n", argv[0]);
        return 2;
    }

    const bool known = case_name == "all" || case_name == "baseline" ||
        case_name == "dispatch" || case_name == "rmd-routes" ||
        case_name == "rmd-direct-parity" || case_name == "rmd-gather";
    if (!known) {
        std::fprintf(stderr, "unknown case: %s\n", case_name.c_str());
        return 2;
    }

    std::printf("TEST_CASE_BEGIN name=%s\n", case_name.c_str());
    const bool ok =
        (case_name == "all" && test_exsia_baseline() && test_dispatch_modes() &&
         test_rmd_cpu_ws_routes() && test_rmd_cpu_direct_parity() &&
         test_rmd_lane_partition() && test_rmd_weight_gather()) ||
        (case_name == "baseline" && test_exsia_baseline()) ||
        (case_name == "dispatch" && test_dispatch_modes()) ||
        (case_name == "rmd-routes" && test_rmd_cpu_ws_routes()) ||
        (case_name == "rmd-direct-parity" && test_rmd_cpu_direct_parity() &&
         test_rmd_lane_partition()) ||
        (case_name == "rmd-gather" && test_rmd_weight_gather());
    if (ok)
        std::printf("PASS: case=%s\n", case_name.c_str());
    return ok ? 0 : 1;
#endif
}
