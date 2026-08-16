#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/dispatch.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/quantize.hpp"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/types.hpp"

#include <ggml.h>
#ifndef GEMMINI_EXSIA_WRITER_TEST_ONLY
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-compose.hpp"
#include "../ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp"
#include <gemmini.h>
#endif

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
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
        !check(metrics.stacked_i_tile_count == expected_i_tiles,
               "RMD executor reports every stacked I tile")) {
        return false;
    }
    std::printf("RMD stacked schedule: matmul_calls=%zu stacked_i_tiles=%zu avoided_B_loads=%zu\n",
                metrics.matmul_call_count, metrics.stacked_i_tile_count,
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
    return true;
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
        case_name == "rmd-direct-parity";
    if (!known) {
        std::fprintf(stderr, "unknown case: %s\n", case_name.c_str());
        return 2;
    }

    std::printf("TEST_CASE_BEGIN name=%s\n", case_name.c_str());
    const bool ok =
        (case_name == "all" && test_exsia_baseline() && test_dispatch_modes() &&
         test_rmd_cpu_ws_routes() && test_rmd_cpu_direct_parity()) ||
        (case_name == "baseline" && test_exsia_baseline()) ||
        (case_name == "dispatch" && test_dispatch_modes()) ||
        (case_name == "rmd-routes" && test_rmd_cpu_ws_routes()) ||
        (case_name == "rmd-direct-parity" && test_rmd_cpu_direct_parity());
    if (ok)
        std::printf("PASS: case=%s\n", case_name.c_str());
    return ok ? 0 : 1;
#endif
}
