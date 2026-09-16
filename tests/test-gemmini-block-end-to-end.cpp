#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-config.hpp"
#include "../ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp"
#include "../ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp"

#include "../common/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string_view>
#include <vector>

namespace {

constexpr size_t kGroup = 32;
constexpr size_t kColumns = DIM + 3;

#if GGML_GEMMINI_WEIGHT_BITS == 4
constexpr ggml_type kWeightType = GGML_TYPE_Q4_0;
#elif GGML_GEMMINI_WEIGHT_BITS == 8
constexpr ggml_type kWeightType = GGML_TYPE_Q8_0;
#elif GGML_GEMMINI_WEIGHT_BITS == 16
constexpr ggml_type kWeightType = GGML_TYPE_Q16_0;
#else
#error "Unsupported Gemmini weight width"
#endif

struct Reconstruction {
    std::vector<float> values;
    size_t outliers = 0;
};

bool check(bool condition, const char * message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

int32_t round_to_i32(float value) {
    constexpr int64_t int32_min = std::numeric_limits<int32_t>::min();
    constexpr int64_t int32_max = std::numeric_limits<int32_t>::max();
    const double rounded = std::nearbyint(static_cast<double>(value));
    return static_cast<int32_t>(std::clamp(
        rounded, static_cast<double>(int32_min), static_cast<double>(int32_max)));
}

int32_t clip_native(int32_t value) {
    constexpr int32_t qmax = (int32_t{1} << (GGML_GEMMINI_ACTIVATION_BITS - 1)) - 1;
    constexpr int32_t qmin = -(int32_t{1} << (GGML_GEMMINI_ACTIVATION_BITS - 1));
    return std::clamp(value, qmin, qmax);
}

// This intentionally mirrors the BLOCK numeric contract without calling any
// production quantize/dequantize helper.
Reconstruction scalar_reconstruct(const std::vector<float> & source, size_t rows, size_t cols) {
    constexpr float qmax = static_cast<float>((int32_t{1} << (GGML_GEMMINI_ACTIVATION_BITS - 1)) - 1);
    Reconstruction result{std::vector<float>(source.size()), 0};
    for (size_t row = 0; row < rows; ++row) {
        for (size_t begin = 0; begin < cols; begin += kGroup) {
            const size_t end = std::min(cols, begin + kGroup);
            double sum = 0.0;
            size_t finite = 0;
            for (size_t k = begin; k < end; ++k) {
                const float value = source[row * cols + k];
                if (std::isfinite(value)) {
                    sum += value;
                    ++finite;
                }
            }
            const double mean = finite == 0 ? 0.0 : sum / static_cast<double>(finite);
            double squared = 0.0;
            for (size_t k = begin; k < end; ++k) {
                const float value = source[row * cols + k];
                if (std::isfinite(value)) squared += (value - mean) * (value - mean);
            }
            const double stddev = finite == 0 ? 0.0 : std::sqrt(squared / static_cast<double>(finite));
            const double threshold = GGML_GEMMINI_EXSIA_SIGMA * stddev;

            float inlier_max = 0.0f;
            float finite_max = 0.0f;
            bool has_inlier = false;
            std::vector<bool> selected(end - begin);
            for (size_t k = begin; k < end; ++k) {
                const float value = source[row * cols + k];
                const bool outlier =
#if GGML_GEMMINI_ENABLE_RMD
                    std::isfinite(value) && std::fabs(value - mean) > threshold;
#else
                    false;
#endif
                selected[k - begin] = outlier;
                if (std::isfinite(value)) {
                    finite_max = std::max(finite_max, std::fabs(value));
                    if (!outlier) {
                        inlier_max = std::max(inlier_max, std::fabs(value));
                        has_inlier = true;
                    }
                }
            }
            const float scale_max = has_inlier ? inlier_max : finite_max;
            const float scale = scale_max > 0.0f ? scale_max / qmax : 1.0f;
            for (size_t k = begin; k < end; ++k) {
                const float value = source[row * cols + k];
                const int32_t q32 = std::isfinite(value) ? round_to_i32(value / scale) : 0;
                const int32_t q = clip_native(q32);
#if GGML_GEMMINI_ENABLE_RMD
                const int32_t residual = selected[k - begin] ? q32 - q : 0;
                result.outliers += selected[k - begin] && residual != 0;
#else
                constexpr int32_t residual = 0;
#endif
                result.values[row * cols + k] = scale * static_cast<float>(q + residual);
            }
        }
    }
    return result;
}

std::vector<float> make_activation(size_t rows, size_t cols) {
    std::vector<float> values(rows * cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < cols / kGroup; ++block) {
            const bool zero_inliers = cols == 3 * kGroup &&
                row + 1 == rows && block + 1 == cols / kGroup;
            const float base = zero_inliers ? 0.0f : static_cast<float>((row + 1) * (block + 1));
            const float outlier = zero_inliers ? 1000.25f :
                (block % 2 == 0 ? 100.0f : -100.0f) * base;
            for (size_t local = 0; local < kGroup; ++local) {
                values[row * cols + block * kGroup + local] = local == 16 ? outlier : base;
            }
        }
    }
    return values;
}

int weight_code(size_t column, size_t block) {
    return static_cast<int>(column % 3 + 1) * (block % 2 == 0 ? 1 : -1);
}

std::vector<uint8_t> make_weights(size_t cols, size_t columns) {
    std::vector<uint8_t> encoded(columns * ggml_row_size(kWeightType, cols));
    const size_t blocks = cols / kGroup;
#if GGML_GEMMINI_WEIGHT_BITS == 4
    auto * data = reinterpret_cast<block_q4_0 *>(encoded.data());
    for (size_t column = 0; column < columns; ++column) {
        for (size_t block = 0; block < blocks; ++block) {
            block_q4_0 & encoded_block = data[column * blocks + block];
            encoded_block.d = ggml_fp32_to_fp16(1.0f);
            const uint8_t code = static_cast<uint8_t>(weight_code(column, block) + 8);
            std::memset(encoded_block.qs, static_cast<int>(code | (code << 4)), sizeof(encoded_block.qs));
        }
    }
#elif GGML_GEMMINI_WEIGHT_BITS == 8
    auto * data = reinterpret_cast<block_q8_0 *>(encoded.data());
    for (size_t column = 0; column < columns; ++column) {
        for (size_t block = 0; block < blocks; ++block) {
            block_q8_0 & encoded_block = data[column * blocks + block];
            encoded_block.d = ggml_fp32_to_fp16(1.0f);
            std::fill(std::begin(encoded_block.qs), std::end(encoded_block.qs),
                      static_cast<int8_t>(weight_code(column, block)));
        }
    }
#else
    auto * data = reinterpret_cast<block_q16_0 *>(encoded.data());
    for (size_t column = 0; column < columns; ++column) {
        for (size_t block = 0; block < blocks; ++block) {
            block_q16_0 & encoded_block = data[column * blocks + block];
            encoded_block.d = ggml_fp32_to_fp16(1.0f);
            std::fill(std::begin(encoded_block.qs), std::end(encoded_block.qs),
                      static_cast<int16_t>(weight_code(column, block)));
        }
    }
#endif
    return encoded;
}

float scalar_dot(const Reconstruction & activation, size_t row, size_t column, size_t cols) {
    float total = 0.0f;
    for (size_t k = 0; k < cols; ++k) {
        total += activation.values[row * cols + k] * weight_code(column, k / kGroup);
    }
    return total;
}

bool run_happy_case(ggml_backend_t backend, size_t rows, size_t cols) {
    ggml_init_params params{ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!check(ctx != nullptr, "context initializes")) return false;

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, kWeightType, cols, kColumns);
    ggml_tensor * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, activation);
    ggml_set_name(weights, "blk.0.attn_q.weight");
    ggml_set_name(activation, "blk.0.attn_input");
    ggml_set_name(output, "blk.0.attn_q.result");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!check(buffer != nullptr, "backend allocates public graph tensors")) {
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> source = make_activation(rows, cols);
    const std::vector<uint8_t> weights_data = make_weights(cols, kColumns);
    const Reconstruction expected_activation = scalar_reconstruct(source, rows, cols);
    ggml_backend_tensor_set(weights, weights_data.data(), 0, weights_data.size());
    ggml_backend_tensor_set(activation, source.data(), 0, source.size() * sizeof(float));

    bool ok = check(ggml_backend_supports_op(backend, output), "GEMMINI accepts valid BLOCK graph");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ok = check(status == GGML_STATUS_SUCCESS, "GEMMINI public graph compute succeeds") && ok;

    std::vector<float> actual(rows * kColumns);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        float max_abs = 0.0f;
        float max_rel = 0.0f;
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < kColumns; ++column) {
                const float expected = scalar_dot(expected_activation, row, column, cols);
                const float observed = actual[row * kColumns + column];
                const float error = std::fabs(observed - expected);
                max_abs = std::max(max_abs, error);
                max_rel = std::max(max_rel, error / std::max(1e-12f, std::fabs(expected)));
                ok = check(std::isfinite(observed) && error <= 1e-5f * std::max(1.0f, std::fabs(expected)),
                           "public output matches scalar BLOCK reconstruction") && ok;
            }
        }
        std::printf("BLOCK_E2E backend=%s rows=%zu J=%zu K=%zu outliers=%zu max_abs=%g max_rel=%g %s\n",
                    ggml_backend_name(backend), rows, kColumns, cols, expected_activation.outliers,
                    max_abs, max_rel, ok ? "PASS" : "FAIL");
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

bool verify_cycle_log([[maybe_unused]] const char * requested_path,
                      [[maybe_unused]] bool expect_im2p_compact) {
#if !LOG_CYCLE
    std::fprintf(stderr, "FAIL: --cycle requires LOG_CYCLE=1\n");
    return false;
#else
    const std::filesystem::path path = ggml::gemmini::log::resolve_output_path(requested_path);
    std::ifstream input(path);
    std::set<uint64_t> quant_runs;
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    std::set<uint64_t> rmd_runs;
    uint64_t rmd_dot_calls = 0;
    uint64_t rmd_cycles = 0;
    size_t rmd_events = 0;
#endif
    size_t quant_events = 0;
    for (std::string line; std::getline(input, line);) {
        if (line.empty()) continue;
        const nlohmann::json event = nlohmann::json::parse(line);
        const std::string op = event.value("op", "");
        const bool quant = op == "gemmini.quantize_activation";
        if (op.rfind("block.", 0) == 0) {
            std::fprintf(stderr, "FAIL: baseline BLOCK emitted detailed cycle telemetry\n");
            return false;
        }
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
        const bool im2p_rmd = op == "rmd.im2p.execute";
#else
        constexpr bool im2p_rmd = false;
#endif
        if (!quant && !im2p_rmd) continue;
        if (event.value("schema", "") != ggml::gemmini::kCycleTelemetrySchema ||
            event.value("version", 0U) != ggml::gemmini::kCycleTelemetryVersion ||
            !event.contains("run_id") || event.at("run_id").is_null()) {
            std::fprintf(stderr, "FAIL: public cycle record lacks schema or run id\n");
            return false;
        }
        const uint64_t run_id = event.at("run_id").get<uint64_t>();
        if (quant) {
            ++quant_events;
            quant_runs.insert(run_id);
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
        } else {
            ++rmd_events;
            rmd_runs.insert(run_id);
            rmd_dot_calls += event.value("rmd_dot_calls", 0ULL);
            rmd_cycles += event.value("rmd_work_total_cycles", 0ULL);
#endif
        }
    }
    if (quant_events == 0) {
        std::fprintf(stderr, "FAIL: public quantization cycle records missing\n");
        return false;
    }
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    std::set<uint64_t> shared_rmd_runs;
    std::set_intersection(quant_runs.begin(), quant_runs.end(), rmd_runs.begin(), rmd_runs.end(),
                          std::inserter(shared_rmd_runs, shared_rmd_runs.end()));
    if (expect_im2p_compact &&
        (rmd_events == 0 || rmd_dot_calls == 0 || rmd_cycles == 0 || shared_rmd_runs.empty())) {
        std::fprintf(stderr, "FAIL: public IM2P compact telemetry rmd=%zu dots=%llu cycles=%llu shared=%zu\n",
                     rmd_events, static_cast<unsigned long long>(rmd_dot_calls),
                     static_cast<unsigned long long>(rmd_cycles), shared_rmd_runs.size());
        return false;
    }
    if (expect_im2p_compact) {
        std::printf("BLOCK_E2E_IM2P rmd_events=%zu rmd_dot_calls=%llu rmd_cycles=%llu shared_run=%llu PASS\n",
                    rmd_events, static_cast<unsigned long long>(rmd_dot_calls),
                    static_cast<unsigned long long>(rmd_cycles),
                    static_cast<unsigned long long>(*shared_rmd_runs.begin()));
    }
#endif
    std::printf("BLOCK_E2E_CYCLE quant_events=%zu block_detail_events=0 PASS\n", quant_events);
    return true;
#endif
}

}

int main(int argc, char ** argv) {
    using ggml::gemmini::config::ActivationQuantAlgo;
    const char * cycle_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--cycle" && i + 1 < argc) {
            cycle_path = argv[++i];
        } else {
            return 2;
        }
    }
    if (!check(ggml::gemmini::config::CURRENT_ACTIVATION_QUANT == ActivationQuantAlgo::BLOCK,
               "test was built with BLOCK selected")) return 1;
    std::printf("BLOCK_E2E_CONFIG activation_bits=%d weight_bits=%d rmd=%d group=%zu block_size=%d dim=%d\n",
                GGML_GEMMINI_ACTIVATION_BITS, GGML_GEMMINI_WEIGHT_BITS,
                GGML_GEMMINI_ENABLE_RMD, kGroup, GGML_GEMMINI_BLOCK_SIZE, DIM);

    const auto options = ggml::gemmini::resolve_matmul_options();
    if (!check(options.ok() && options.options.mode == ggml::gemmini::MatmulInvocationMode::full,
               "BLOCK public fixture resolves FULL")) return 1;
#if GGML_GEMMINI_ENABLE_RMD
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    const auto configured_rmd = static_cast<ggml::gemmini::RmdBackend>(
        ggml::gemmini::config::DEFAULT_RMD_BACKEND);
    if (!check(options.options.rmd_backend == configured_rmd,
               "BLOCK IM2P fixture honors configured residual backend")) return 1;
    std::printf("BLOCK_E2E_IM2P_ROUTE rmd_backend=%s\n",
                configured_rmd == ggml::gemmini::RmdBackend::gemmini_ws_compact ? "compact" : "cpu_direct");
#else
    if (!check(options.options.rmd_backend == ggml::gemmini::RmdBackend::cpu_direct,
               "BLOCK host fixture selects CPU-direct residual backend")) return 1;
#endif
#endif

    std::filesystem::path cycle_output;
    if (cycle_path != nullptr) {
        cycle_output = ggml::gemmini::log::resolve_output_path(cycle_path);
        std::error_code error;
        if (!cycle_output.parent_path().empty()) std::filesystem::create_directories(cycle_output.parent_path(), error);
        std::filesystem::remove(cycle_output, error);
        if (!check(ggml::gemmini::log::cycle.set_output_path(cycle_path, true),
                   "public cycle sink initializes")) return 1;
    }
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name("GEMMINI", "llama");
    if (!check(backend != nullptr && std::strcmp(ggml_backend_name(backend), "GEMMINI") == 0,
               "public GEMMINI backend is selected")) return 1;
    const bool ok = run_happy_case(backend, 1, 32) &&
        run_happy_case(backend, static_cast<size_t>(DIM) + 1, 64) &&
        run_happy_case(backend, 2, 96);
    ggml_backend_free(backend);
    if (cycle_path != nullptr) ggml::gemmini::log::cycle.set_output(stderr);
    return ok && (cycle_path == nullptr || verify_cycle_log(
        cycle_path, options.options.rmd_backend == ggml::gemmini::RmdBackend::gemmini_ws_compact)) ? 0 : 1;
}
