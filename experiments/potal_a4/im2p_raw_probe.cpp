// Real numerical Gemmini HP1 raw-dot admission and parity, independent of ggml dispatch.
#include "im2p_sim.h"
#include "raw_batch.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct Backing {
    size_t m, n, k, weight_stride, output_stride;
    const int8_t * weights;
    int32_t * output;
    std::vector<bool> seen;
};

static int read_weight(void * context, size_t row, size_t column, size_t count, int8_t * out) {
    const auto & backing = *static_cast<Backing *>(context);
    if (row >= backing.k || column > backing.n || count > backing.n-column) return -1;
    std::copy_n(backing.weights+row*backing.weight_stride+column, count, out);
    return 0;
}

static int write_output(void * context, size_t block, size_t row, size_t column, size_t count,
                        const int64_t * values, uint32_t domain) {
    auto & backing = *static_cast<Backing *>(context);
    if (block || domain != IM2P_OUTPUT_LEGACY_FINAL || row >= backing.m ||
        column > backing.n || count > backing.n-column) return -1;
    for (size_t i = 0; i < count; ++i) {
        const size_t index = row*backing.n+column+i;
        if (backing.seen[index] || values[i] < INT32_MIN || values[i] > INT32_MAX) return -1;
        backing.output[row*backing.output_stride+column+i] = static_cast<int32_t>(values[i]);
        backing.seen[index] = true;
    }
    return 0;
}

static constexpr std::pair<const char *, uint64_t im2p_work_stats_t::*> metrics[] = {
    {"work_total_cycles", &im2p_work_stats_t::work_total_cycles},
    {"activation_read_requests", &im2p_work_stats_t::activation_read_requests},
    {"weight_read_requests", &im2p_work_stats_t::weight_read_requests},
    {"scale_read_requests", &im2p_work_stats_t::scale_read_requests},
    {"output_write_requests", &im2p_work_stats_t::output_write_requests},
    {"output_write_responses", &im2p_work_stats_t::output_write_responses},
    {"activation_wait_cycles", &im2p_work_stats_t::activation_wait_cycles},
    {"weight_wait_cycles", &im2p_work_stats_t::weight_wait_cycles},
    {"output_wait_cycles", &im2p_work_stats_t::output_wait_cycles},
    {"stripe_host_wait_cycles", &im2p_work_stats_t::stripe_host_wait_cycles},
    {"drain_cycles", &im2p_work_stats_t::drain_cycles},
    {"weight_preload_cycles", &im2p_work_stats_t::weight_preload_cycles},
    {"compute_cycles", &im2p_work_stats_t::compute_cycles},
    {"overlap_cycles", &im2p_work_stats_t::overlap_cycles},
    {"completed_fragments", &im2p_work_stats_t::completed_fragments},
    {"completed_output_tiles", &im2p_work_stats_t::completed_output_tiles},
    {"completed_stripes", &im2p_work_stats_t::completed_stripes},
    {"weight_bank_activations", &im2p_work_stats_t::weight_bank_activations},
};

struct Runtime {
    std::unique_ptr<im2p_sim_t, decltype(&im2p_sim_destroy)> sim{nullptr, im2p_sim_destroy};
    im2p_work_stats_t totals{};
    uint64_t calls = 0;

    Runtime() {
        require(im2p_sim_implementation() && std::strcmp(im2p_sim_implementation(), "gemmini-hp1-integrated-v1") == 0,
                "expected integrated Gemmini HP1 numerical runtime");
        require(im2p_sim_abi_version() == 5 && im2p_sim_activation_bits() == 4 && im2p_sim_weight_bits() == 4 &&
                im2p_sim_dim() == 32 && im2p_sim_activation_storage_bytes() == 1 && im2p_sim_weight_storage_bytes() == 1 &&
                im2p_compiled_accumulator_rows() == 512, "matched A4/W4/DIM32/ACC512 identity required");
        sim.reset(im2p_sim_create());
        require(bool(sim), "im2p_sim_create failed");
    }

    void dot(const int8_t * a, const int8_t * b, int32_t * output,
             int m, int k, int n, int a_stride, int b_stride, int output_stride) {
        im2p_matmul_desc_t dot{};
        dot.abi_version = IM2P_ABI_VERSION;
        dot.activation_bits = dot.weight_bits = 4;
        dot.activation_storage_bytes = dot.weight_storage_bytes = 1;
        dot.dim = 32;
        dot.activations = a;
        dot.weights = b;
        dot.output = output;
        dot.m = m; dot.n = n; dot.k = k;
        dot.activation_row_stride_bytes = a_stride;
        dot.weight_row_stride_bytes = b_stride;
        dot.output_row_stride = output_stride;
        dot.tile_i_rows = std::min(m, 32);
        dot.tile_j_columns = std::min(n, 32);
        dot.block_size = 1;
        dot.vector_op = IM2P_VECTOR_BYPASS;
        dot.output_domain = IM2P_OUTPUT_LEGACY_FINAL;
        // Match the existing Im2pCompactDot provider path. The legacy pointer-only
        // adapter derives scale_block_size=K when scales are absent; HP1 raw requires
        // scale_block_size=1. Provider descriptors retain the explicit block_size.
        Backing backing{size_t(m), size_t(n), size_t(k), size_t(b_stride), size_t(output_stride),
                        b, output, std::vector<bool>(size_t(m)*n, false)};
        dot.provider.context = &backing;
        dot.provider.read_weight_i8 = read_weight;
        dot.provider.write_output = write_output;
        im2p_work_stats_extended_t stats{};
        const int status = im2p_execute_matmul_extended(sim.get(), &dot, &stats);
        require(status == IM2P_OK, "real raw dot failed; no CPU fallback");
        require(std::all_of(backing.seen.begin(), backing.seen.end(), [](bool value) { return value; }), "Incomplete provider output");
        require(stats.base.work_total_cycles > 0, "real raw dot did not report RTL cycles");
        for (const auto & metric : metrics) totals.*metric.second += stats.base.*metric.second;
        ++calls;
    }
};

#ifdef POTAL_IM2P_LIBRARY
static thread_local std::unique_ptr<Runtime> runtime;
static thread_local std::string error_message, stats_json;

extern "C" size_t potal_replay_work_size() { return sizeof(replay_work); }
extern "C" const char * potal_replay_error() { return error_message.c_str(); }

extern "C" int potal_im2p_reset() {
    try {
        runtime.reset();
        runtime = std::make_unique<Runtime>();
        error_message.clear();
        return 0;
    } catch (const std::exception & error) {
        error_message = error.what();
        return -1;
    }
}

extern "C" const char * potal_im2p_stats() {
    try {
        require(bool(runtime), "Initialize the Gemmini runtime before replay");
        std::ostringstream json;
        json << "{\"implementation\":\"" << im2p_sim_implementation() << "\",\"abi\":" << im2p_sim_abi_version()
             << ",\"activation_bits\":" << im2p_sim_activation_bits() << ",\"weight_bits\":" << im2p_sim_weight_bits()
             << ",\"dim\":" << im2p_sim_dim() << ",\"accumulator_rows\":" << im2p_compiled_accumulator_rows()
             << ",\"activation_storage_bytes\":" << im2p_sim_activation_storage_bytes()
             << ",\"weight_storage_bytes\":" << im2p_sim_weight_storage_bytes() << ",\"raw_dot_calls\":" << runtime->calls;
        for (const auto & metric : metrics) json << ",\"" << metric.first << "\":" << runtime->totals.*metric.second;
        json << '}';
        stats_json = json.str();
        return stats_json.c_str();
    } catch (const std::exception & error) {
        error_message = error.what();
        return nullptr;
    }
}

extern "C" int potal_replay_batch(const replay_work * work, int count, int rows, int columns,
                                  int direct, int64_t * out, uint64_t * stats) {
    try {
        require(bool(runtime), "Initialize the Gemmini runtime before replay");
        execute_replay_batch(work, count, rows, columns, direct, out, stats,
            [](const int8_t * a, const int8_t * b, int32_t * c, int m, int k, int n, int as, int bs, int cs) {
                runtime->dot(a, b, c, m, k, n, as, bs, cs);
            });
        return 0;
    } catch (const std::exception & error) {
        error_message = error.what();
        return -1;
    }
}
#else
static void run(size_t m, size_t n, size_t k, int repeat, bool units = false, size_t padding = 0) {
    const size_t stride = n + padding;
    std::vector<int8_t> a(m * k), b(k * stride);
    std::vector<int32_t> actual(m * stride, INT32_MIN), expected(m * stride, INT32_MIN);
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<int8_t>((i * 13 + repeat) % 16) - 8;
    if (units) for (auto & value : a) value = value > 0 ? 1 : value < 0 ? -1 : 0;
    for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<int8_t>((i * 7 + repeat) % 16) - 8;
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            expected[i * stride + j] = 0;
            for (size_t r = 0; r < k; ++r)
                expected[i * stride + j] += int32_t(a[i * k + r]) * b[r * stride + j];
        }
    Runtime runtime;
    const auto begin = std::chrono::steady_clock::now();
    runtime.dot(a.data(), b.data(), actual.data(), int(m), int(k), int(n), int(k), int(stride), int(stride));
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::cout << "{\"scope\":\"real Gemmini HP1 raw op0; operator-only\",\"m\":" << m
              << ",\"n\":" << n << ",\"k\":" << k << ",\"repeat\":" << repeat
              << ",\"row_padding\":" << padding
              << ",\"activation_values\":\"" << (units ? "zero_or_unit" : "signed_nibble") << "\""
              << ",\"status\":0,\"integer_equal\":" << (actual == expected ? "true" : "false")
              << ",\"rtl_cycles\":" << runtime.totals.work_total_cycles
              << ",\"host_execute_seconds\":" << seconds << "}" << std::endl;
    require(actual == expected, "real raw dot differs from independent integer oracle");
}

int main() {
    try {
        for (int repeat = 0; repeat < 3; ++repeat) {
            run(1, 1, 1, repeat);
            run(2, 3, 31, repeat);
            run(33, 35, 32, repeat);
            run(32, 32, 32, repeat);
            run(32, 32, 32, repeat, true);
            run(3, 35, 31, repeat, false, 9);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
