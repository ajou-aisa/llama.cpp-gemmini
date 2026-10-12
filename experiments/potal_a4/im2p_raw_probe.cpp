// Real numerical Gemmini HP1 raw-dot admission and parity, independent of ggml dispatch.
#include "im2p_sim.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct Backing {
    size_t m, n, k;
    const std::vector<int8_t> & weights;
    std::vector<int32_t> & output;
    std::vector<bool> seen;
};

static int read_weight(void * context, size_t row, size_t column, size_t count, int8_t * out) {
    const auto & backing = *static_cast<Backing *>(context);
    if (row >= backing.k || column > backing.n || count > backing.n-column) return -1;
    std::copy_n(backing.weights.data()+row*backing.n+column, count, out);
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
        backing.output[index] = static_cast<int32_t>(values[i]);
        backing.seen[index] = true;
    }
    return 0;
}

static void run(size_t m, size_t n, size_t k, int repeat, bool units = false) {
    std::vector<int8_t> a(m * k), b(k * n);
    std::vector<int32_t> actual(m * n, INT32_MIN), expected(m * n, 0);
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<int8_t>((i * 13 + repeat) % 16) - 8;
    if (units) for (auto & value : a) value = value > 0 ? 1 : value < 0 ? -1 : 0;
    for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<int8_t>((i * 7 + repeat) % 16) - 8;
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t r = 0; r < k; ++r)
                expected[i * n + j] += int32_t(a[i * k + r]) * b[r * n + j];
    std::unique_ptr<im2p_sim_t, decltype(&im2p_sim_destroy)> sim(im2p_sim_create(), im2p_sim_destroy);
    require(bool(sim), "im2p_sim_create failed");
    im2p_matmul_desc_t dot{};
    dot.abi_version = IM2P_ABI_VERSION;
    dot.activation_bits = dot.weight_bits = 4;
    dot.activation_storage_bytes = dot.weight_storage_bytes = 1;
    dot.dim = 32;
    dot.activations = a.data();
    dot.weights = b.data();
    dot.output = actual.data();
    dot.m = m; dot.n = n; dot.k = k;
    dot.activation_row_stride_bytes = k;
    dot.weight_row_stride_bytes = n;
    dot.output_row_stride = n;
    dot.tile_i_rows = std::min(m, size_t{32});
    dot.tile_j_columns = std::min(n, size_t{32});
    dot.block_size = 1;
    dot.vector_op = IM2P_VECTOR_BYPASS;
    dot.output_domain = IM2P_OUTPUT_LEGACY_FINAL;
    // Match the existing Im2pCompactDot provider path. The legacy pointer-only
    // adapter derives scale_block_size=K when scales are absent; HP1 raw requires
    // scale_block_size=1. Provider descriptors retain the explicit block_size.
    Backing backing{m, n, k, b, actual, std::vector<bool>(m*n, false)};
    dot.provider.context = &backing;
    dot.provider.read_weight_i8 = read_weight;
    dot.provider.write_output = write_output;
    im2p_work_stats_extended_t stats{};
    const auto begin = std::chrono::steady_clock::now();
    const int status = im2p_execute_matmul_extended(sim.get(), &dot, &stats);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::cout << "{\"scope\":\"real Gemmini HP1 raw op0; operator-only\",\"m\":" << m
              << ",\"n\":" << n << ",\"k\":" << k << ",\"repeat\":" << repeat
              << ",\"activation_values\":\"" << (units ? "zero_or_unit" : "signed_nibble") << "\""
              << ",\"status\":" << status << ",\"integer_equal\":" << (actual == expected ? "true" : "false")
              << ",\"rtl_cycles\":" << stats.base.work_total_cycles
              << ",\"host_execute_seconds\":" << seconds << "}" << std::endl;
    require(status == IM2P_OK, "real raw dot failed; no CPU fallback");
    require(actual == expected, "real raw dot differs from independent integer oracle");
    require(std::all_of(backing.seen.begin(), backing.seen.end(), [](bool value) { return value; }), "Incomplete provider output");
    require(stats.base.work_total_cycles > 0, "real raw dot did not report RTL cycles");
}

int main() {
    try {
        require(im2p_sim_implementation() && std::strcmp(im2p_sim_implementation(), "gemmini-hp1-integrated-v1") == 0,
                "expected integrated Gemmini HP1 numerical runtime");
        require(im2p_sim_abi_version() == 5 && im2p_sim_activation_bits() == 4 && im2p_sim_weight_bits() == 4 &&
                im2p_sim_dim() == 32 && im2p_sim_activation_storage_bytes() == 1 && im2p_sim_weight_storage_bytes() == 1 &&
                im2p_compiled_accumulator_rows() == 512, "matched A4/W4/DIM32/ACC512 identity required");
        for (int repeat = 0; repeat < 3; ++repeat) {
            run(1, 1, 1, repeat);
            run(2, 3, 31, repeat);
            run(33, 35, 32, repeat);
            run(32, 32, 32, repeat);
            run(32, 32, 32, repeat, true);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
