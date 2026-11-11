#include "../pb/pb-exec.hpp"
#include "pb-five-core-env.hpp"
#include "../ggml-gemmini-args.h"
#include "ggml-quants.h"
#include "ggml-cpu-quants.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__AVX__) || defined(__AVX2__) || defined(__FMA__) || defined(__F16C__)
#error "The Q8_0 reference must compile to ggml's scalar branches."
#endif

namespace pb = ggml::gemmini::pb;

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

uint16_t independent_fp16(float value) {
    const uint16_t sign = std::signbit(value) ? 0x8000 : 0;
    const double magnitude = std::fabs(double(value));
    if (magnitude == 0.0) return sign;
    if (magnitude >= 65520.0) return sign | 0x7c00;
    int exponent = 0;
    const double mantissa = std::frexp(magnitude, &exponent) * 2.0;
    --exponent;
    if (exponent < -14) return sign | static_cast<uint16_t>(std::nearbyint(std::ldexp(magnitude, 24)));
    double rounded = std::nearbyint(std::ldexp(mantissa, 10));
    if (rounded == 2048.0) { rounded = 1024.0; ++exponent; }
    return sign | static_cast<uint16_t>((exponent + 15) << 10) | static_cast<uint16_t>(rounded - 1024.0);
}
float independent_widen(uint16_t half) {
    const int exponent = (half >> 10) & 31, fraction = half & 1023;
    if (exponent == 31) {
        const float special = fraction ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
        return (half >> 15) ? -special : special;
    }
    const double magnitude = exponent == 0 ? std::ldexp(double(fraction), -24)
                                           : std::ldexp(double(1024 + fraction), exponent - 25);
    return static_cast<float>((half >> 15) ? -magnitude : magnitude);
}
void independent_block(const float * x, block_q8_0 & y) {
    float amax = 0.0f;
    for (size_t lane = 0; lane < 32; ++lane) amax = std::max(amax, std::fabs(x[lane]));
    const float delta = amax / 127.0f;
    const float inverse = delta != 0.0f ? 1.0f / delta : 0.0f;
    y.d = independent_fp16(delta);
    for (size_t lane = 0; lane < 32; ++lane) {
        const float scaled = x[lane] * inverse;
        const double magnitude = std::floor(std::fabs(double(scaled)) + 0.5);
        y.qs[lane] = static_cast<int8_t>(scaled < 0.0f ? -magnitude : magnitude);
    }
}

struct Case {
    size_t m, j, k, blocks;
    std::vector<float> raw_a, raw_b;
    std::vector<block_q8_0> a, b;
    Case(size_t rows, size_t cols, size_t depth)
        : m(rows), j(cols), k(depth), blocks(depth / 32), raw_a(rows * depth), raw_b(cols * depth),
          a(rows * blocks), b(cols * blocks) {}
};

float kernel(const Case & c, size_t i, size_t col, size_t used_blocks) {
    float value = -1.0f;
    ggml_vec_dot_q8_0_q8_0(int(32 * used_blocks), &value, 0, &c.b[col * c.blocks], 0, &c.a[i * c.blocks], 0, 1);
    return value;
}

void quantize_case(Case & c) {
    for (size_t i = 0; i < c.m; ++i)
        quantize_row_q8_0(c.raw_a.data() + i * c.k, c.a.data() + i * c.blocks, int64_t(c.k));
    check(quantize_q8_0(c.raw_b.data(), c.b.data(), int64_t(c.j), int64_t(c.k), nullptr) ==
              c.j * c.blocks * sizeof(block_q8_0), "quantize_q8_0 output size");
    std::vector<block_q8_0> own_a(c.a.size()), own_b(c.b.size());
    for (size_t i = 0; i < c.m; ++i)
        for (size_t block = 0; block < c.blocks; ++block)
            independent_block(c.raw_a.data() + i * c.k + block * 32, own_a[i * c.blocks + block]);
    for (size_t col = 0; col < c.j; ++col)
        for (size_t block = 0; block < c.blocks; ++block)
            independent_block(c.raw_b.data() + col * c.k + block * 32, own_b[col * c.blocks + block]);
    check(std::memcmp(own_a.data(), c.a.data(), own_a.size() * sizeof(block_q8_0)) == 0, "activation Q8_0 bytes match the independent quantizer");
    check(std::memcmp(own_b.data(), c.b.data(), own_b.size() * sizeof(block_q8_0)) == 0, "weight Q8_0 bytes match the independent quantizer");
    std::vector<int8_t> codes;
    std::vector<float> scales;
    check(pb::quantize_q8_0_rows(c.raw_a.data(), c.m, c.k, codes, scales) == pb::Status::success, "production activation quantizer status");
    for (size_t i = 0; i < c.m; ++i)
        for (size_t block = 0; block < c.blocks; ++block) {
            const auto & expected = c.a[i * c.blocks + block];
            check(bits(scales[i * c.blocks + block]) == bits(independent_widen(expected.d)), "production activation scale bits");
            for (size_t lane = 0; lane < 32; ++lane) {
                check(expected.qs[lane] >= -127 && expected.qs[lane] <= 127, "activation code range");
                check(codes[i * c.k + block * 32 + lane] == expected.qs[lane], "production activation codes");
            }
        }
}

struct Observation {
    const Case * c;
    size_t next = 0;
    std::vector<float> independent_sum;
};
void observe(const pb::BlockObservation & o, void * opaque) {
    auto & state = *static_cast<Observation *>(opaque);
    const Case & c = *state.c;
    check(o.block == state.next++, "ascending K blocks");
    check(o.m == c.m && o.j == c.j && o.k == 32, "per-block dimensions");
    if (state.independent_sum.empty()) state.independent_sum.assign(c.m * c.j, +0.0f);
    for (size_t i = 0; i < c.m; ++i) {
        const auto & a = c.a[i * c.blocks + o.block];
        check(bits(o.s_a[i]) == bits(independent_widen(a.d)), "activation block scale bits");
        for (size_t lane = 0; lane < 32; ++lane) check(o.a[i * 32 + lane] == a.qs[lane], "packed activation codes");
    }
    for (size_t col = 0; col < c.j; ++col) {
        const auto & b = c.b[col * c.blocks + o.block];
        check(bits(o.s_b[col]) == bits(independent_widen(b.d)), "weight block scale bits");
        for (size_t lane = 0; lane < 32; ++lane) check(o.b[lane * c.j + col] == b.qs[lane], "packed K-by-J weight codes");
    }
    for (size_t i = 0; i < c.m; ++i)
        for (size_t col = 0; col < c.j; ++col) {
            const auto & a = c.a[i * c.blocks + o.block];
            const auto & b = c.b[col * c.blocks + o.block];
            int64_t dot = 0;
            for (size_t lane = 0; lane < 32; ++lane) dot += int64_t(a.qs[lane]) * int64_t(b.qs[lane]);
            check(std::llabs(dot) <= 516128, "one-block dot bound");
            check(o.acc[i * c.j + col] == dot, "per-block INT32 readback");
            float & own = state.independent_sum[i * c.j + col];
            own = own + float(int32_t(dot)) * (independent_widen(b.d) * independent_widen(a.d));
            const float produced = o.output[i * o.row_stride + col * o.column_stride];
            const float reference = kernel(c, i, col, o.block + 1);
            check(bits(produced) == bits(reference), "scaled partial sum matches the scalar ggml dot");
            check(bits(produced) == bits(own), "scaled partial sum matches the independent expression");
        }
}

void graph_case(const Case & c, const char * name, ggml_gemmini_pb_node_type type) {
    ggml_context * ctx = ggml_init({c.m * c.k * 4 + c.j * c.k * 2 + c.m * c.j * 4 + size_t(32) * 1024 * 1024, nullptr, false});
    check(ctx != nullptr, "context allocation");
    auto * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, int64_t(c.k), int64_t(c.j));
    auto * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, int64_t(c.k), int64_t(c.m));
    check(ggml_nbytes(weight) == c.b.size() * sizeof(block_q8_0), "Q8_0 tensor byte size");
    std::memcpy(weight->data, c.b.data(), ggml_nbytes(weight));
    std::memcpy(activation->data, c.raw_a.data(), ggml_nbytes(activation));
    ggml_set_name(weight, name);
    auto * node = ggml_mul_mat(ctx, weight, activation);
    auto * backend = ggml_backend_gemmini_init();
    check(backend != nullptr, "GEMMINI backend");
    check(ggml_backend_supports_op(backend, node), "GEMMINI admits the Q8_0 matmul");
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, node);
    ggml_gemmini_pb_reset_counters();
    pb::reset_tiled_call_count();
    pb::reset_validated_weights();
    Observation state{&c};
    pb::set_block_observer(observe, &state);
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "backend graph execution");
    pb::set_block_observer(nullptr, nullptr);
    check(state.next == c.blocks && pb::tiled_call_count() == c.blocks, "exactly one tiled call per K block");
    const auto * output = static_cast<const float *>(node->data);
    for (size_t i = 0; i < c.m; ++i)
        for (size_t col = 0; col < c.j; ++col)
            check(bits(output[i * c.j + col]) == bits(kernel(c, i, col, c.blocks)), "final FP32 bits match the scalar ggml dot");
    for (int t = 0; t < GGML_GEMMINI_PB_NODE_TYPE_COUNT; ++t) {
        ggml_gemmini_pb_counters counters{};
        check(ggml_gemmini_pb_get_counters(static_cast<ggml_gemmini_pb_node_type>(t), &counters), "counter API");
        check(counters.fallback_count == 0, "no fallback");
        check(counters.nodes == (t == type ? 1u : 0u), "routed to one node type");
        check(counters.k_blocks == (t == type ? c.blocks : 0u), "K-block count");
        check(counters.int32_elements == (t == type ? c.blocks * c.m * c.j : 0u), "INT32 element count");
    }
    ggml_backend_free(backend);
    ggml_free(ctx);
}

void stride_case(const Case & c) {
    std::vector<int8_t> a(c.m * c.k), b(c.j * c.k);
    std::vector<float> s_a(c.m * c.blocks), s_b(c.blocks * c.j);
    for (size_t i = 0; i < c.m; ++i)
        for (size_t block = 0; block < c.blocks; ++block) {
            s_a[i * c.blocks + block] = independent_widen(c.a[i * c.blocks + block].d);
            std::memcpy(a.data() + i * c.k + block * 32, c.a[i * c.blocks + block].qs, 32);
        }
    for (size_t col = 0; col < c.j; ++col)
        for (size_t block = 0; block < c.blocks; ++block) {
            s_b[block * c.j + col] = independent_widen(c.b[col * c.blocks + block].d);
            std::memcpy(b.data() + col * c.k + block * 32, c.b[col * c.blocks + block].qs, 32);
        }
    const size_t row_stride = 2 * c.j + 5, col_stride = 2;
    std::vector<float> output(c.m * row_stride, -1234.0f);
    Observation state{&c};
    pb::set_block_observer(observe, &state);
    check(pb::execute(a.data(), s_a.data(), b.data(), s_b.data(), c.m, c.j, c.k, output.data(), row_stride, col_stride) ==
              pb::Status::success, "explicit-stride status");
    pb::set_block_observer(nullptr, nullptr);
    for (size_t n = 0; n < output.size(); ++n) {
        const size_t within = n % row_stride;
        if (within % 2 || within >= 2 * c.j) check(output[n] == -1234.0f, "output stride canary");
        else check(bits(output[n]) == bits(kernel(c, n / row_stride, within / 2, c.blocks)), "explicit-stride FP32 bits");
    }
}

Case patterned(size_t m, size_t j, size_t k, int seed) {
    Case c(m, j, k);
    for (size_t i = 0; i < m; ++i)
        for (size_t p = 0; p < k; ++p)
            c.raw_a[i * k + p] = std::scalbn(float(int((i * 17 + p * 11 + size_t(seed)) % 251) - 125) + 0.25f, int((i + p / 32) % 7) - 8);
    for (size_t col = 0; col < j; ++col)
        for (size_t p = 0; p < k; ++p)
            c.raw_b[col * k + p] = std::scalbn(float(int((col * 23 + p * 7 + size_t(seed) * 3) % 255) - 127) + 0.125f, int((col + p / 32) % 6) - 10);
    std::fill_n(c.raw_a.data(), std::min(k, size_t{32}), 0.0f);
    if (k > 32) std::fill_n(c.raw_b.data() + 32, 32, 0.0f);
    quantize_case(c);
    return c;
}

void format_facts() {
    check(ggml_type_size(GGML_TYPE_Q8_0) == 34 && ggml_blck_size(GGML_TYPE_Q8_0) == 32, "Q8_0 size and block length");
    check(sizeof(block_q8_0) == 34 && sizeof(block_q8_1) == 36, "Q8_0 and Q8_1 are different layouts");
    check(ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->vec_dot_type == GGML_TYPE_Q8_0, "a Q8_0 weight pairs with Q8_0 activations");
    check(ggml_get_type_traits_cpu(GGML_TYPE_Q4_1)->vec_dot_type == GGML_TYPE_Q8_1, "Q8_1 belongs to Q4_1");
    check(ggml_get_type_traits_cpu(GGML_TYPE_Q5_1)->vec_dot_type == GGML_TYPE_Q8_1, "Q8_1 belongs to Q5_1");
    const float half_values[] = {1.0f / 127.0f, 1.0f, 3.25f, 65519.996f, 65520.0f, std::ldexp(1.0f, -25),
                                 std::nextafter(std::ldexp(1.0f, -25), 1.0f), 1.5f * std::ldexp(1.0f, -24), 0.0f};
    for (const float value : half_values) {
        check(independent_fp16(value) == ggml_fp32_to_fp16(value), "FP32 to FP16 agrees with ggml");
        check(bits(independent_widen(independent_fp16(value))) == bits(ggml_fp16_to_fp32(independent_fp16(value))), "FP16 widening agrees with ggml");
    }
    check(independent_fp16(65519.996f) == 0x7bff && independent_fp16(65520.0f) == 0x7c00, "FP16 overflow boundary");
    check(independent_fp16(std::ldexp(1.0f, -25)) == 0 && independent_fp16(1.5f * std::ldexp(1.0f, -24)) == 2, "FP16 subnormal rounding");

    std::vector<float> row(32, 0.0f);
    const float ties[] = {127.0f, 2.5f, -2.5f, 3.5f, 0.5f, -0.5f, 126.5f};
    const int expected[] = {127, 3, -3, 4, 1, -1, 127};
    std::copy(std::begin(ties), std::end(ties), row.begin());
    block_q8_0 block{};
    quantize_row_q8_0(row.data(), &block, 32);
    for (size_t n = 0; n < 7; ++n) check(block.qs[n] == expected[n], "scalar round half away from zero");
    check(independent_widen(block.d) == 1.0f, "tie case scale");
    std::vector<int8_t> codes;
    std::vector<float> scales;
    check(pb::quantize_q8_0_rows(row.data(), 1, 32, codes, scales) == pb::Status::success, "production tie case status");
    for (size_t n = 0; n < 7; ++n) check(codes[n] == expected[n], "production uses ggml's rounding");
}

void scale_order_case() {
    bool found = false;
    for (int code_a = 127; code_a > 90 && !found; --code_a)
        for (int code_b = 127; code_b > 90 && !found; --code_b)
            for (uint16_t ha = 0x3001; ha < 0x3400 && !found; ha += 2)
                for (uint16_t hb = 0x3501; hb < 0x3900 && !found; hb += 2) {
                    const float sa = independent_widen(ha), sb = independent_widen(hb);
                    const float product_first = float(code_a * code_b) * (sb * sa);
                    float sequential = float(code_a * code_b) * sa;
                    sequential = sequential * sb;
                    if (bits(product_first) == bits(sequential)) continue;
                    found = true;
                    std::vector<int8_t> a(32, 0), b(32, 0);
                    a[0] = int8_t(code_a);
                    b[0] = int8_t(code_b);
                    float output = 9.0f;
                    check(pb::execute(a.data(), &sa, b.data(), &sb, 1, 1, 32, &output, 1, 1) == pb::Status::success, "scale order status");
                    check(bits(output) == bits(product_first), "the two scales multiply before the integer sum");
                    check(bits(output) != bits(sequential), "sequential scale order is distinguishable");
                    block_q8_0 x{}, y{};
                    x.d = hb; y.d = ha;
                    x.qs[0] = int8_t(code_b); y.qs[0] = int8_t(code_a);
                    float reference = -1.0f;
                    ggml_vec_dot_q8_0_q8_0(32, &reference, 0, &x, 0, &y, 0, 1);
                    check(bits(output) == bits(reference), "scale order matches the scalar ggml dot");
                }
    check(found, "a scale-order witness exists");

    std::vector<int8_t> a(96, 0), b(96, 0);
    a[0] = a[32] = a[64] = 1;
    b[0] = b[32] = 1;
    b[64] = -1;
    const float s_a[3] = {1.0f, 1.0f, 1.0f}, s_b[3] = {16777216.0f, 1.0f, 16777216.0f};
    float output = 9.0f;
    check(pb::execute(a.data(), s_a, b.data(), s_b, 1, 1, 96, &output, 1, 1) == pb::Status::success, "accumulation order status");
    check(bits(output) == bits(0.0f), "blocks accumulate in ascending order");
}

void failure_cases() {
    Case c = patterned(2, 3, 64, 5);
    ggml_context * ctx = ggml_init({size_t(32) * 1024 * 1024, nullptr, false});
    auto * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 3);
    auto * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 2);
    std::memcpy(weight->data, c.b.data(), ggml_nbytes(weight));
    std::memcpy(activation->data, c.raw_a.data(), ggml_nbytes(activation));
    auto * node = ggml_mul_mat(ctx, weight, activation);
    pb::reset_validated_weights();
    const uint64_t validations = pb::weight_validation_count();
    const auto flushes = pb::flush_count();
    pb::reset_tiled_call_count();
    check(pb::execute_linear_q8_0(*node, "x") == pb::Status::success, "first run");
    check(pb::tiled_call_count() == 2 && pb::flush_count() - flushes == 1, "two K blocks and one TLB flush per GEMM");
    check(pb::execute_linear_q8_0(*node, "x") == pb::Status::success, "second run");
    check(pb::weight_validation_count() == validations + 1, "weights validated once per tensor");

    const auto rejected = [&](ggml_tensor & probe, pb::Status expected, const char * message) {
        pb::reset_tiled_call_count();
        check(pb::execute_linear_q8_0(probe, "x") == expected && pb::tiled_call_count() == 0, message);
    };
    auto * mxint = ggml_new_tensor_2d(ctx, GGML_TYPE_MXINT8, 64, 3);
    ggml_tensor mxint_node = *node;
    mxint_node.src[0] = mxint;
    rejected(mxint_node, pb::Status::invalid_arguments, "MXINT8 weight rejected without array work");
    auto * q8_1 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_1, 64, 3);
    ggml_tensor q8_1_node = *node;
    q8_1_node.src[0] = q8_1;
    rejected(q8_1_node, pb::Status::invalid_arguments, "Q8_1 weight rejected without array work");
    auto * f16_activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 2);
    ggml_tensor f16_node = *node;
    f16_node.src[1] = f16_activation;
    rejected(f16_node, pb::Status::invalid_arguments, "non-F32 activation rejected");
    ggml_tensor odd_weight = *weight;
    odd_weight.ne[0] = 33;
    ggml_tensor odd_node = *node;
    odd_node.src[0] = &odd_weight;
    rejected(odd_node, pb::Status::invalid_arguments, "K not a multiple of 32 rejected");

    auto * stored = static_cast<block_q8_0 *>(weight->data);
    const auto saved = stored[0];
    stored[0].qs[3] = -128;
    pb::reset_validated_weights();
    rejected(*node, pb::Status::invalid_codes, "stored code -128 rejected");
    stored[0] = saved;
    stored[1].d = 0x7e00;
    pb::reset_validated_weights();
    rejected(*node, pb::Status::invalid_scales, "stored NaN scale rejected");
    stored[1] = c.b[1];

    pb::reset_validated_weights();
    auto * data = static_cast<float *>(activation->data);
    const float saved_value = data[7];
    data[7] = std::numeric_limits<float>::quiet_NaN();
    rejected(*node, pb::Status::invalid_codes, "non-finite activation rejected");
    data[7] = saved_value;

    float output = 9.0f;
    int8_t code = 1, invalid = -128;
    float scale = 1.0f, negative = -1.0f, zero = 0.0f;
    check(pb::execute(&invalid, &scale, &code, &scale, 1, 1, 1, &output, 1, 1) == pb::Status::invalid_codes, "execute rejects -128");
    check(pb::execute(&code, &negative, &code, &scale, 1, 1, 1, &output, 1, 1) == pb::Status::invalid_scales, "negative activation scale rejected");
    check(pb::execute(&code, &zero, &code, &scale, 1, 1, 1, &output, 1, 1) == pb::Status::success && bits(output) == 0, "zero Q8_0 scale accepted");
    ggml_free(ctx);
}
} // namespace

int main() {
    const bool five_core = pb_test_start_five_core();
    ggml_context * guard = ggml_init({1024, nullptr, true});
    format_facts();

    const size_t shapes[][3] = {{1, 1, 32}, {2, 3, 32}, {5, 7, 64}, {35, 67, 96}, {65, 97, 96}, {3, 5, 896}, {2, 2, 4864}, {33, 65, 160}};
    int seed = 1;
    for (const auto & shape : shapes) {
        const Case c = patterned(shape[0], shape[1], shape[2], seed++);
        stride_case(c);
        graph_case(c, "blk.0.ffn_up.weight", GGML_GEMMINI_PB_FFN_UP);
    }

    const char * names[] = {"blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight", "blk.3.attn_output.weight",
                            "blk.3.ffn_gate.weight", "blk.3.ffn_up.weight", "blk.3.ffn_down.weight", "token_embd.weight"};
    const Case routed = patterned(4, 6, 96, 40);
    for (int t = 0; t < 8; ++t) graph_case(routed, names[t], static_cast<ggml_gemmini_pb_node_type>(t));

    Case worst(2, 3, 64);
    std::fill(worst.raw_a.begin(), worst.raw_a.end(), 127.0f);
    for (size_t col = 0; col < worst.j; ++col)
        for (size_t p = 0; p < worst.k; ++p) worst.raw_b[col * worst.k + p] = p < 32 ? 127.0f : -127.0f;
    quantize_case(worst);
    stride_case(worst);
    graph_case(worst, "blk.0.ffn_gate.weight", GGML_GEMMINI_PB_FFN_GATE);
    check(bits(kernel(worst, 0, 0, 1)) == bits(516128.0f), "maximum one-block magnitude reaches 516128");

    Case all_zero(2, 2, 64);
    quantize_case(all_zero);
    graph_case(all_zero, "blk.0.ffn_down.weight", GGML_GEMMINI_PB_FFN_DOWN);

    scale_order_case();
    failure_cases();
    ggml_free(guard);
    pb_test_finish_five_core(five_core);
    std::printf("gemmini-pb-q8-0-block-test: PASS checks=%llu\n", static_cast<unsigned long long>(checks));
    return 0;
}
