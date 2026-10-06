#include "../ggml/src/ggml-metal/ggml-metal-quantized.h"
#include "im2p_sim.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t zero_carrier = UINT32_C(0x80000000);
size_t provider_calls = 0;

void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

std::vector<int32_t> provider(size_t m, size_t n, size_t k,
        const std::vector<int8_t> & a, const std::vector<int8_t> & w,
        const std::vector<uint32_t> & carriers,
        const std::vector<im2p_compact_run_t> & runs = {}, size_t original_k = 0) {
    std::vector<int32_t> result(m * n, -777);
    im2p_matmul_desc_t d{};
    d.abi_version = im2p_sim_abi_version();
    d.activation_bits = im2p_sim_activation_bits();
    d.weight_bits = im2p_sim_weight_bits();
    d.activation_storage_bytes = d.weight_storage_bytes = 1;
    d.dim = im2p_sim_dim();
    d.m = m; d.n = n; d.k = k;
    d.activations = a.data(); d.weights = w.data(); d.scales = carriers.data();
    d.output = result.data();
    d.activation_row_stride_bytes = k;
    d.weight_row_stride_bytes = n;
    d.output_row_stride = n;
    d.tile_i_rows = m; d.tile_j_columns = n;
    d.block_size = 32; d.scale_total_k = k;
    d.scale_row_stride = d.scale_valid_columns = n;
    d.scale_values_len = carriers.size();
    d.vector_op = IM2P_VECTOR_LEFT_SHIFT;
    d.output_domain = IM2P_OUTPUT_SCU_FINAL;
    im2p_production_geometry_v1_t g{};
    g.version = IM2P_PRODUCTION_GEOMETRY_VERSION; g.struct_size = sizeof(g);
    g.activation_bits = d.activation_bits; g.weight_bits = d.weight_bits; g.dim = d.dim;
    g.scope = IM2P_GEOMETRY_FULL;
    g.m = m; g.n = n; g.k = k;
    g.tile_i_count = g.tile_j_count = g.tile_k_count = 1;
    g.stripe_rows = g.row_count = m;
    std::unique_ptr<im2p_sim_t, decltype(&im2p_sim_destroy)> sim(im2p_sim_create(), im2p_sim_destroy);
    require(bool(sim), "provider initialization");
    im2p_compact_runs_t run_view{1, sizeof(run_view), static_cast<uint32_t>(original_k), runs.size(), runs.data()};
    const int status = runs.empty() ? im2p_execute_matmul_planned(sim.get(), &d, &g, nullptr)
        : im2p_execute_matmul_planned_runs(sim.get(), &d, &g, &run_view, nullptr);
    require(status == IM2P_OK, "reference provider execution failed");
    ++provider_calls;
    return result;
}

template <typename T>
void equal(const std::vector<T> & actual, const std::vector<T> & expected, const char * field) {
    require(actual.size() == expected.size(), "comparison shape");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s[%zu] actual=%lld expected=%lld\n", field, i,
                         static_cast<long long>(actual[i]), static_cast<long long>(expected[i]));
            throw std::runtime_error(field);
        }
    }
}

void float_equal(const std::vector<float> & actual, const std::vector<float> & expected) {
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::memcmp(&actual[i], &expected[i], sizeof(float))) {
            std::fprintf(stderr, "output[%zu] actual=%a expected=%a\n", i, actual[i], expected[i]);
            throw std::runtime_error("bitwise F32 output");
        }
    }
}

struct Fixture {
    ggml_metal_quantized_view v{};
    std::vector<int8_t> a, w;
    std::vector<uint32_t> carriers;
    std::vector<float> a_scale, w_scale;

    Fixture(size_t m, size_t n, size_t k) : a(m * k), w(n * k), carriers(((k + 31) / 32) * n),
                                           a_scale(m), w_scale(n) {
        v.profile = {GGML_METAL_QUANTIZED_HP1_EXSIA, im2p_sim_activation_bits(), im2p_sim_dim(), false};
        v.m = m; v.n = n; v.k = k; v.activation_rows_per_stripe = m;
        v.activations = a.data(); v.weights = w.data(); v.carriers = carriers.data();
        v.activation_scales = a_scale.data(); v.column_scales = w_scale.data();
        for (size_t r = 0; r < m; ++r) a_scale[r] = r & 1 ? 0.03125f : 0.125f;
        for (size_t c = 0; c < n; ++c) w_scale[c] = (c & 1 ? -0.7f : 0.3f) * float(c + 1);
    }

    std::vector<int8_t> transposed_weights() const {
        std::vector<int8_t> result(v.k * v.n);
        for (size_t c = 0; c < v.n; ++c)
            for (size_t k = 0; k < v.k; ++k) result[k * v.n + c] = w[c * v.k + k];
        return result;
    }
};

void dense_case(size_t k, unsigned pattern) {
    Fixture f(3, 5, k);
    const int max_code = f.v.profile.bits == 4 ? 7 : 127;
    const int min_code = -max_code - 1;
    for (size_t r = 0; r < f.v.m; ++r)
        for (size_t x = 0; x < k; ++x)
            f.a[r * k + x] = pattern == 0 ? int8_t(((x / 16 + r) & 1) ? -max_code : max_code)
                : int8_t(int((x * 7 + r * 13) % (2 * max_code + 2)) + min_code);
    for (size_t c = 0; c < f.v.n; ++c)
        for (size_t x = 0; x < k; ++x)
            f.w[c * k + x] = pattern == 0 ? int8_t((c & 1) ? -1 : 1)
                : int8_t(int((x * 11 + c * 17) % (2 * max_code + 2)) + min_code);
    const uint32_t exponents[] = {0, 31, 32, 32767, zero_carrier};
    for (size_t b = 0; b < (k + 31) / 32; ++b)
        for (size_t c = 0; c < f.v.n; ++c)
            f.carriers[b * f.v.n + c] = exponents[(c + (pattern == 0 ? 0 : b)) % 5];
    const auto expected = provider(f.v.m, f.v.n, k, f.a, f.transposed_weights(), f.carriers);
    const size_t fragment = std::min(f.v.profile.dim, 32u);
    const size_t fragments = (k + fragment - 1) / fragment;
    const size_t outputs = f.v.m * f.v.n;
    std::vector<int32_t> raw(outputs * fragments), scu(raw.size()), dense(outputs);
    std::vector<int64_t> correction(outputs);
    std::vector<float> output(outputs), expected_output(outputs);
    ggml_metal_quantized_trace trace{raw.data(), scu.data(), raw.size(), dense.data(), correction.data(), outputs, nullptr, 0};
    require(ggml_metal_quantized_execute(&f.v, nullptr, output.data(), &trace), ggml_metal_quantized_last_error());
    equal(dense, expected, "dense provider integer");
    for (size_t begin = 0, index = 0; begin < k; begin += fragment, ++index) {
        const size_t count = std::min(fragment, k - begin);
        std::vector<int8_t> a(f.v.m * count), w(count * f.v.n);
        std::vector<uint32_t> exponent(f.v.n), raw_exponent(f.v.n, 0);
        for (size_t r = 0; r < f.v.m; ++r)
            std::copy_n(f.a.data() + r * k + begin, count, a.data() + r * count);
        for (size_t x = 0; x < count; ++x)
            for (size_t c = 0; c < f.v.n; ++c) w[x * f.v.n + c] = f.w[c * k + begin + x];
        std::copy_n(f.carriers.data() + (begin / 32) * f.v.n, f.v.n, exponent.data());
        const auto expected_raw = provider(f.v.m, f.v.n, count, a, w, raw_exponent);
        const auto expected_scu = provider(f.v.m, f.v.n, count, a, w, exponent);
        for (size_t i = 0; i < outputs; ++i) {
            require(raw[i * fragments + index] == expected_raw[i], "raw fragment provider comparison");
            require(scu[i * fragments + index] == expected_scu[i], "SCU fragment provider comparison");
        }
    }
    for (size_t r = 0; r < f.v.m; ++r)
        for (size_t c = 0; c < f.v.n; ++c)
            expected_output[r * f.v.n + c] = float(double(expected[r * f.v.n + c]) * double(f.w_scale[c]) * double(f.a_scale[r]));
    float_equal(output, expected_output);
    std::printf("PROVIDER_METAL_DENSE A%uW%u DIM%u K%zu pattern%u raw=exact scu=exact final=exact F32=bitwise PASS\n",
                f.v.profile.bits, f.v.profile.bits, f.v.profile.dim, k, pattern);
}

void run_case(bool carry_lane, bool large_exponents) {
    Fixture f(4, 5, 256);
    f.v.profile.residual_enabled = true; f.v.request_count = 1;
    std::fill(f.a.begin(), f.a.end(), 0);
    std::fill(f.carriers.begin(), f.carriers.end(), zero_carrier);
    const uint32_t block_ids[] = {0, 3, 7};
    const size_t counts[] = {31, 1, 32};
    const uint32_t masks[] = {UINT32_C(0x7fffffff), UINT32_C(0x80000000), UINT32_MAX};
    const size_t begins[] = {0, 31, 32};
    std::vector<std::vector<uint16_t>> local_k(3);
    std::vector<ggml_metal_quantized_run> runs(3);
    std::vector<im2p_compact_run_t> provider_runs(3);
    std::vector<uint32_t> carriers(3 * f.v.n);
    const uint32_t exponents[] = {0, 31, 32, 32767, zero_carrier};
    for (size_t b = 0; b < 3; ++b) {
        for (size_t x = 0; x < counts[b]; ++x) local_k[b].push_back(b == 1 ? 31 : uint16_t(x));
        runs[b] = {block_ids[b], block_ids[b] * 32, masks[b], begins[b], counts[b], local_k[b].data()};
        provider_runs[b] = {block_ids[b], masks[b], uint32_t(begins[b]), uint32_t(counts[b])};
        for (size_t c = 0; c < f.v.n; ++c) {
            const uint32_t e = large_exponents ? exponents[c] : uint32_t(b);
            f.carriers[block_ids[b] * f.v.n + c] = carriers[b * f.v.n + c] = e;
        }
    }
    const uint32_t last_lane = carry_lane ? 32 / f.v.profile.bits : 2;
    std::vector<ggml_metal_quantized_row> rows{{0, 0}, {0, 2}, {last_lane, 0}, {last_lane, 2}};
    std::vector<int8_t> a(rows.size() * 64), w(64 * f.v.n);
    std::vector<int32_t> wide_w(w.size());
    for (size_t row = 0; row < rows.size(); ++row)
        for (size_t x = 0; x < 64; ++x)
            a[row * 64 + x] = int8_t(((x / 16 + row) & 1) ? -1 : 1);
    for (size_t b = 0; b < 3; ++b)
        for (size_t x = 0; x < counts[b]; ++x)
            for (size_t c = 0; c < f.v.n; ++c) {
                const int8_t code = int8_t((c & 1) ? -1 : 1);
                f.w[c * f.v.k + block_ids[b] * 32 + local_k[b][x]] = code;
                wide_w[(begins[b] + x) * f.v.n + c] = w[(begins[b] + x) * f.v.n + c] = code;
            }
    const auto expected_lanes = provider(rows.size(), f.v.n, 64, a, w, carriers, provider_runs, f.v.k);
    ggml_metal_quantized_request request{};
    request.m = rows.size(); request.n = f.v.n; request.k = 64;
    request.source_row_begin = 1; request.source_row_count = 3;
    request.tile_i = request.tile_j = request.tile_k = 1;
    request.run_count = runs.size(); request.runs = runs.data(); request.rows = rows.data();
    request.activations = a.data(); request.weights = wide_w.data(); request.carriers = carriers.data();
    const ggml_metal_quantized_request * ptr = &request;
    const size_t outputs = f.v.m * f.v.n;
    std::vector<int32_t> lanes(expected_lanes.size()), dense(outputs);
    std::vector<int64_t> correction(outputs), expected_correction(outputs, 0);
    std::vector<__int128> wide_correction(outputs, 0);
    std::vector<float> output(outputs), expected_output(outputs, 0);
    for (size_t r = 0; r < rows.size(); ++r)
        for (size_t c = 0; c < f.v.n; ++c)
            wide_correction[(1 + rows[r].source_row) * f.v.n + c] += __int128(expected_lanes[r * f.v.n + c]) *
                (__int128(1) << (f.v.profile.bits * rows[r].original_lane_id));
    for (size_t i = 0; i < outputs; ++i) {
        require(wide_correction[i] >= INT64_MIN && wide_correction[i] <= INT64_MAX, "provider radix overflow");
        expected_correction[i] = int64_t(wide_correction[i]);
        const float dense_zero = float(double(0) * double(f.w_scale[i % f.v.n]) * double(f.a_scale[i / f.v.n]));
        const float restored = float(double(expected_correction[i]) * double(f.w_scale[i % f.v.n]) * double(f.a_scale[i / f.v.n]));
        expected_output[i] = i / f.v.n >= request.source_row_begin ? dense_zero + restored : dense_zero;
    }
    ggml_metal_quantized_trace trace{nullptr, nullptr, 0, dense.data(), correction.data(), outputs, lanes.data(), lanes.size()};
    require(ggml_metal_quantized_execute(&f.v, &ptr, output.data(), &trace), ggml_metal_quantized_last_error());
    equal(lanes, expected_lanes, "ordered run provider lane");
    equal(correction, expected_correction, "provider lane radix correction");
    float_equal(output, expected_output);
    std::printf("PROVIDER_METAL_RUNS A%uW%u DIM%u original_blocks=0,3,7 lanes=0,%u large=%u lane=exact correction=exact F32=bitwise PASS\n",
                f.v.profile.bits, f.v.profile.bits, f.v.profile.dim, last_lane, large_exponents);
}
} // namespace

int main() {
    try {
        require(std::strcmp(im2p_sim_implementation(), "CPU_FUNCTIONAL") == 0, "reference implementation identity");
        require(im2p_compiled_accumulator_bits() == 32 && im2p_compiled_partial_bits() == 32, "reference integer widths");
        ggml_metal_quantized_reset_stats();
        for (const size_t k : {size_t(31), size_t(32), size_t(33), size_t(64), size_t(65), size_t(128)})
            for (unsigned p = 0; p < 2; ++p) dense_case(k, p);
        run_case(false, true);
        run_case(true, false);
        const auto stats = ggml_metal_quantized_get_stats();
        require(stats.dense_launches == 14 && stats.residual_launches == 2 && stats.failed_calls == 0 && stats.fallback_calls == 0,
                "actual Metal dispatch coverage");
        std::printf("PROVIDER_METAL_PASS implementation=%s semantic_revision=%s A%uW%u DIM%u provider_calls=%zu Metal_dense=%llu Metal_residual=%llu fallback=%llu\n",
                    im2p_sim_implementation(), im2p_compiled_numerical_semantics_revision(), im2p_sim_activation_bits(),
                    im2p_sim_weight_bits(), im2p_sim_dim(), provider_calls,
                    (unsigned long long)stats.dense_launches, (unsigned long long)stats.residual_launches,
                    (unsigned long long)stats.fallback_calls);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "PROVIDER_METAL_FAIL %s\n", e.what());
        return 1;
    }
}
