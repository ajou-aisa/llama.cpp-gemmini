#include "../pb/pb-exec.hpp"
#include "../matmul/dense.hpp"
#include "../quants/act/block/block.hpp"
#include <gemmini.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace pb = ggml::gemmini::pb;
namespace act = ggml::gemmini::quants::act;

namespace {
uint64_t checks = 0;
void check(bool value, const char * message) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL check=%llu: %s\n", static_cast<unsigned long long>(checks), message);
        std::exit(1);
    }
}
uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

struct Fixture {
    size_t m, j, k, blocks;
    std::vector<int8_t> a, b;
    std::vector<float> sa, sb, raw_a, raw_b;
    Fixture(size_t rows, size_t cols, size_t depth)
        : m(rows), j(cols), k(depth), blocks((depth + 31) / 32),
          a(rows * depth), b(cols * depth), sa(rows * blocks), sb(blocks * cols),
          raw_a(rows * depth), raw_b(cols * depth) {}
};

// Independent quantizer: frexp exponent and floor(abs(x)+0.5), not production helpers.
void quantize_reference(Fixture & f) {
    for (size_t i = 0; i < f.m; ++i) {
        for (size_t block = 0; block < f.blocks; ++block) {
            const size_t end = std::min(f.k, block * 32 + 32);
            float maximum = 0;
            for (size_t k = block * 32; k < end; ++k)
                maximum = std::max(maximum, std::fabs(f.raw_a[i * f.k + k]));
            int exponent = 0;
            std::frexp(maximum, &exponent);
            const float scale = maximum == 0 ? 1.0f : std::scalbn(1.0f, std::max(-127, exponent - 1) - 6);
            f.sa[i * f.blocks + block] = scale;
            for (size_t k = block * 32; k < end; ++k) {
                const double x = f.raw_a[i * f.k + k] / static_cast<double>(scale);
                const int magnitude = static_cast<int>(std::min(127.0, std::floor(std::fabs(x) + 0.5)));
                f.a[i * f.k + k] = static_cast<int8_t>(x < 0 ? -magnitude : magnitude);
            }
        }
    }
}

std::vector<int32_t> integer_reference(const Fixture & f, size_t block) {
    std::vector<int32_t> result(f.m * f.j);
    for (size_t i = 0; i < f.m; ++i)
        for (size_t j = 0; j < f.j; ++j) {
            int64_t dot = 0;
            for (size_t k = block * 32; k < std::min(f.k, block * 32 + 32); ++k)
                dot += int64_t(f.a[i * f.k + k]) * int64_t(f.b[j * f.k + k]);
            check(std::abs(dot) <= 516128, "wide independent dot fits PB bound");
            result[i * f.j + j] = static_cast<int32_t>(dot);
        }
    return result;
}

std::vector<float> float_reference(const Fixture & f) {
    std::vector<float> result(f.m * f.j, +0.0f);
    for (size_t block = 0; block < f.blocks; ++block) {
        const auto dots = integer_reference(f, block);
        for (size_t i = 0; i < f.m; ++i)
            for (size_t j = 0; j < f.j; ++j) {
                float term = static_cast<float>(dots[i * f.j + j]);
                term = term * f.sa[i * f.blocks + block];
                term = term * f.sb[block * f.j + j];
                result[i * f.j + j] = result[i * f.j + j] + term;
            }
    }
    return result;
}

struct Observation {
    const Fixture * fixture;
    size_t next = 0;
};
void observe(const pb::BlockObservation & block, void * opaque) {
    auto & observation = *static_cast<Observation *>(opaque);
    const auto & f = *observation.fixture;
    check(block.block == observation.next++, "ascending K block order");
    check(block.m == f.m && block.j == f.j &&
          block.k == std::min(size_t{32}, f.k - block.block * 32), "observer dimensions and tail");
    const auto expected = integer_reference(f, block.block);
    for (size_t i = 0; i < f.m; ++i) {
        check(bits(block.s_a[i]) == bits(f.sa[i * f.blocks + block.block]), "activation scale bits");
        for (size_t k = 0; k < block.k; ++k) {
            const int code = block.a[i * block.k + k];
            check(code >= -127 && code <= 127, "activation code range");
            check(code == f.a[i * f.k + block.block * 32 + k], "packed activation code");
        }
        for (size_t j = 0; j < f.j; ++j)
            check(block.acc[i * f.j + j] == expected[i * f.j + j], "per-block INT32 bits");
    }
    for (size_t j = 0; j < f.j; ++j) {
        check(bits(block.s_b[j]) == bits(f.sb[block.block * f.j + j]), "weight scale bits");
        for (size_t k = 0; k < block.k; ++k) {
            const int code = block.b[k * f.j + j];
            check(code >= -127 && code <= 127, "weight code range");
            check(code == f.b[j * f.k + block.block * 32 + k], "software K-by-J packing");
        }
    }
}

void compare_output(const Fixture & f, const std::vector<float> & output, size_t row_stride, size_t col_stride) {
    const auto expected = float_reference(f);
    for (size_t i = 0; i < f.m; ++i)
        for (size_t j = 0; j < f.j; ++j) {
            const float actual = output[i * row_stride + j * col_stride];
            check(bits(actual) == bits(expected[i * f.j + j]), "final FP32 bits");
            float plain = +0.0f;
            double absolute_products = 0;
            for (size_t k = 0; k < f.k; ++k) {
                const float product = f.raw_a[i * f.k + k] * f.raw_b[j * f.k + k];
                plain = plain + product;
                absolute_products += std::fabs(static_cast<double>(product));
            }
            // Quantization comparison only: 3% of absolute product sum plus 1e-5.
            check(std::fabs(double(actual) - plain) <= 0.03 * absolute_products + 1e-5,
                  "plain FP32 reference tolerance");
        }
}

void core_case(Fixture & f) {
    const size_t col_stride = 2, row_stride = 2 * f.j + 5;
    std::vector<float> output(f.m * row_stride, -1234.0f);
    Observation observation{&f};
    pb::set_block_observer(observe, &observation);
    pb::reset_tiled_call_count();
    ggml_gemmini_pb_reset_counters();
    check(pb::execute(f.a.data(), f.sa.data(), f.b.data(), f.sb.data(), f.m, f.j, f.k,
                      output.data(), row_stride, col_stride) == pb::Status::success, "core status");
    pb::set_block_observer(nullptr, nullptr);
    check(observation.next == f.blocks, "one observation per K block");
    check(pb::tiled_call_count() == f.blocks, "exactly one tiled call per K block");
    ggml_gemmini_pb_counters counters{};
    check(ggml_gemmini_pb_get_counters(GGML_GEMMINI_PB_OTHER, &counters), "counter API");
    check(counters.nodes == 1 && counters.k_blocks == f.blocks &&
          counters.int32_elements == f.blocks * f.m * f.j && counters.fallback_count == 0,
          "core node, block, readback, fallback counters");
    compare_output(f, output, row_stride, col_stride);
    for (size_t n = 0; n < output.size(); ++n) {
        const size_t within = n % row_stride;
        if (within % 2 || within >= 2 * f.j) check(output[n] == -1234.0f, "output stride canary");
    }
}

Fixture patterned(size_t m, size_t j, size_t k) {
    Fixture f(m, j, k);
    for (size_t i = 0; i < m; ++i)
        for (size_t p = 0; p < k; ++p)
            f.raw_a[i * k + p] = std::scalbn(float(int((i * 17 + p * 11) % 251) - 125) + 0.25f,
                                           int((i + p / 32) % 7) - 8);
    std::fill_n(f.raw_a.data(), std::min(k, size_t{32}), 0.0f);
    quantize_reference(f);
    for (size_t col = 0; col < j; ++col) {
        const float channel = std::scalbn(1.0f, int(col % 5) - 10);
        for (size_t block = 0; block < f.blocks; ++block) {
            const bool zero = col == 0 && block == 1;
            f.sb[block * j + col] = zero ? 0.0f : std::scalbn(channel, int((col + block) % 6));
            for (size_t p = block * 32; p < std::min(k, block * 32 + 32); ++p) {
                const int code = zero ? 0 : int((col * 23 + p * 7) % 255) - 127;
                f.b[col * k + p] = static_cast<int8_t>(code);
                f.raw_b[col * k + p] = float(code) * f.sb[block * j + col];
            }
        }
    }
    check(bits(f.sa[0]) == bits(1.0f), "zero activation scale is one");
    check(bits(f.sb[j]) == 0, "zero weight scale is positive zero");
    return f;
}

void linear_case() {
    auto f = patterned(35, 67, 96);
    ggml_gemmini_args_t args{};
    args.I = f.m; args.J = f.j; args.K = f.k;
    args.tiled_matmul_type = WS;
    args.mxint_weight = true;
    check(args.A.allocate(f.m, f.k, 8), "activation buffer allocation");
    args.act_quant.storage().emplace<act::block::Meta>();
    ggml_tensor source{};
    source.type = GGML_TYPE_F32; source.data = f.raw_a.data();
    source.ne[0] = f.k; source.ne[1] = f.m; source.ne[2] = source.ne[3] = 1;
    source.nb[0] = sizeof(float); source.nb[1] = f.k * sizeof(float);
    source.nb[2] = source.nb[3] = f.k * f.m * sizeof(float);
    check(act::block::quantize_mxint(&source, args), "production MXINT activation quantizer");
    const auto & meta = std::get<act::block::Meta>(args.act_quant.storage());
    for (size_t i = 0; i < f.m; ++i) {
        for (size_t p = 0; p < f.k; ++p)
            check(args.A.get(i, p) == f.a[i * f.k + p], "independent activation codes");
        for (size_t block = 0; block < f.blocks; ++block)
            check(bits(meta.scales[i * f.blocks + block]) == bits(f.sa[i * f.blocks + block]),
                  "independent activation scale bits");
    }
    std::vector<block_q8_hp1> weights(f.j * f.blocks);
    for (size_t j = 0; j < f.j; ++j)
        for (size_t block = 0; block < f.blocks; ++block) {
            auto & weight = weights[j * f.blocks + block];
            weight.channel_scale = std::scalbn(1.0f, int(j % 5) - 10);
            weight.m = (j == 0 && block == 1) ? INT16_MIN : int16_t((j + block) % 6);
            for (size_t p = 0; p < 32; ++p) weight.qs[p] = f.b[j * f.k + block * 32 + p];
        }
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1;
    args.q8_hp1_blocks = weights.data();
    args.q8_hp1_block_count = weights.size(); args.q8_hp1_blocks_per_row = f.blocks;
    args.native_weight_bytes = weights.size() * sizeof(weights[0]);
    args.blocks_per_row = args.blocks_K = f.blocks; args.blocks_J = f.j;
    args.block_size_k = 32; args.sA = args.sB = f.k;
    const char * names[] = {"unclassified.blk.0.attn_q.weight", "unclassified.blk.0.attn_k.weight",
        "unclassified.blk.0.attn_v.weight", "unclassified.blk.0.attn_output.weight",
        "unclassified.blk.0.ffn_gate.weight", "unclassified.blk.0.ffn_up.weight",
        "unclassified.blk.0.ffn_down.weight", "unclassified.token_embd.weight"};
    ggml_gemmini_pb_reset_counters();
    for (size_t type = 0; type < sizeof(names) / sizeof(names[0]); ++type) {
        std::vector<float> output(f.m * f.j, -99.0f);
        args.f_out = output.data(); args.stride_f_out = f.j; args.col_stride_f_out = 1;
        args.matmul_layer = names[type];
        Observation observation{&f};
        pb::set_block_observer(observe, &observation);
        pb::reset_tiled_call_count();
        ggml::gemmini::MatMul matmul(&args);
        check(matmul.run_full().status == ggml::gemmini::MatMulStatus::success, "Linear facade route");
        pb::set_block_observer(nullptr, nullptr);
        check(pb::tiled_call_count() == f.blocks && observation.next == f.blocks, "Linear tiled calls");
        compare_output(f, output, f.j, 1);
        ggml_gemmini_pb_counters counters{};
        check(ggml_gemmini_pb_get_counters(static_cast<ggml_gemmini_pb_node_type>(type), &counters),
              "per-type counter API");
        check(counters.nodes == 1 && counters.k_blocks == f.blocks &&
              counters.int32_elements == f.m * f.j * f.blocks && counters.fallback_count == 0,
              "Linear per-node-type counters");
    }
    pb::reset_validated_weights();
    std::vector<float> repeated(f.m * f.j, -99.0f);
    args.f_out = repeated.data();
    const uint64_t validations = pb::weight_validation_count();
    check(pb::execute_linear(args) == pb::Status::success && pb::execute_linear(args) == pb::Status::success &&
          pb::weight_validation_count() == validations + 1, "immutable weights validated once per tensor");
    compare_output(f, repeated, f.j, 1);
    const int8_t saved_code = weights[0].qs[0];
    weights[0].qs[0] = -128;
    pb::reset_validated_weights();
    pb::reset_tiled_call_count();
    check(pb::execute_linear(args) == pb::Status::invalid_codes && pb::tiled_call_count() == 0,
          "stored code -128 rejected at first use before array work");
    weights[0].qs[0] = saved_code;
    pb::reset_validated_weights();
    args.f_out = f.raw_a.data();
    args.weight_format = ggml_gemmini_args_t::im2p_weight_format_t::q8_h2;
    pb::reset_tiled_call_count();
    check(pb::execute_linear(args) != pb::Status::success && pb::tiled_call_count() == 0,
          "unsupported Linear fails without dispatch or fallback");
}
} // namespace

int main() {
    auto multi = patterned(67, 101, 77);
    core_case(multi);
    auto whole = patterned(65, 97, 96);
    core_case(whole);
    Fixture worst(2, 3, 64);
    std::fill(worst.a.begin(), worst.a.end(), int8_t{127});
    std::fill(worst.sa.begin(), worst.sa.end(), 1.0f);
    std::fill(worst.sb.begin(), worst.sb.end(), 1.0f);
    for (size_t j = 0; j < worst.j; ++j)
        for (size_t k = 0; k < worst.k; ++k) worst.b[j * worst.k + k] = k < 32 ? 127 : -127;
    for (size_t n = 0; n < worst.a.size(); ++n) worst.raw_a[n] = worst.a[n];
    for (size_t n = 0; n < worst.b.size(); ++n) worst.raw_b[n] = worst.b[n];
    check(integer_reference(worst, 0)[0] == 516128 && integer_reference(worst, 1)[0] == -516128,
          "worst-case magnitude 516128 and independent consecutive blocks");
    core_case(worst);
    Fixture order(1, 1, 96);
    order.a[0] = order.a[32] = order.a[64] = 1;
    order.b[0] = order.b[32] = 1; order.b[64] = -1;
    std::fill(order.sa.begin(), order.sa.end(), 1.0f);
    order.sb = {16777216.0f, 1.0f, 16777216.0f};
    for (size_t k = 0; k < order.k; ++k) {
        order.raw_a[k] = order.a[k];
        order.raw_b[k] = order.b[k] * order.sb[k / 32];
    }
    check(bits(float_reference(order)[0]) == bits(0.0f), "ascending FP32 order witness");
    volatile float reversed = -16777216.0f;
    reversed = reversed + 1.0f; reversed = reversed + 16777216.0f;
    check(reversed == 1.0f, "reversing addition changes result");
    core_case(order);
    Fixture signed_scale(1, 1, 64);
    std::fill(signed_scale.a.begin(), signed_scale.a.end(), int8_t{3});
    std::fill(signed_scale.b.begin(), signed_scale.b.end(), int8_t{7});
    signed_scale.sa = {0.5f, 2.0f};
    signed_scale.sb = {-0.37f, 0.91f};
    for (size_t k = 0; k < signed_scale.k; ++k) {
        signed_scale.raw_a[k] = signed_scale.a[k] * signed_scale.sa[k / 32];
        signed_scale.raw_b[k] = signed_scale.b[k] * signed_scale.sb[k / 32];
    }
    core_case(signed_scale);
    linear_case();
    float output = 9;
    int8_t invalid = -128, valid = 1;
    float scale = 1;
    pb::reset_tiled_call_count();
    check(pb::execute(&invalid, &scale, &valid, &scale, 1, 1, 1, &output, 1, 1) ==
          pb::Status::invalid_codes && output == 9 && pb::tiled_call_count() == 0, "reject -128 before array");
    check(!ggml_gemmini_pb_get_counters(GGML_GEMMINI_PB_NODE_TYPE_COUNT, nullptr), "invalid counter query");

    std::vector<int8_t> extreme_a(32, 127), extreme_b(32, 127);
    const float huge_scale = std::ldexp(1.0f, 121), tiny_scale = std::ldexp(1.0f, -100);
    float ordered = 0.0f;
    check(pb::execute(extreme_a.data(), &huge_scale, extreme_b.data(), &tiny_scale, 1, 1, 32, &ordered, 1, 1) ==
              pb::Status::success && bits(ordered) == 0x7f800000u,
          "activation scale multiplies before weight scale");

    std::vector<int8_t> flush_a(64, 1), flush_b(64, 1);
    const float flush_s_a[2] = {1.0f, 1.0f}, flush_s_b[2] = {1.0f, 1.0f};
    float flush_out = 0.0f;
    const auto flushes_before = pb::flush_count();
    const auto calls_before = pb::tiled_call_count();
    check(pb::execute(flush_a.data(), flush_s_a, flush_b.data(), flush_s_b, 1, 1, 64, &flush_out, 1, 1) ==
              pb::Status::success, "two-block execute for the TLB flush check");
    check(pb::tiled_call_count() - calls_before == 2 && pb::flush_count() - flushes_before == 1,
          "one Gemmini TLB flush per GEMM");

    std::vector<float> non_finite(64, 1.0f);
    non_finite[3] = std::numeric_limits<float>::quiet_NaN();
    ggml_gemmini_args_t nan_args{};
    nan_args.I = 2; nan_args.J = 2; nan_args.K = 32;
    nan_args.A.allocate(2, 32, 8);
    nan_args.act_quant.storage().emplace<act::block::Meta>();
    ggml_tensor nan_source{};
    nan_source.type = GGML_TYPE_F32; nan_source.data = non_finite.data();
    nan_source.ne[0] = 32; nan_source.ne[1] = 2; nan_source.ne[2] = nan_source.ne[3] = 1;
    nan_source.nb[0] = sizeof(float); nan_source.nb[1] = 32 * sizeof(float);
    nan_source.nb[2] = nan_source.nb[3] = 64 * sizeof(float);
    check(!act::block::quantize_mxint(&nan_source, nan_args), "non-finite activation rejected by the quantizer");
    std::printf("gemmini-pb-block-test: PASS checks=%llu\n", static_cast<unsigned long long>(checks));
    return 0;
}
