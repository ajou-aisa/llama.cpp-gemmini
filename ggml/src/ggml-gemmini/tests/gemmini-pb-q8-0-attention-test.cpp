#include "../pb/pb-exec.hpp"
#include "pb-five-core-env.hpp"
#include "../matmul/attention.hpp"
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
void check(bool valid, const char * message) {
    ++checks;
    if (!valid) {
        std::fprintf(stderr, "FAIL check=%llu: %s\n", static_cast<unsigned long long>(checks), message);
        std::exit(1);
    }
}
uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
char * address(const ggml_tensor * t, size_t k, size_t row, size_t head, size_t batch) {
    return static_cast<char *>(t->data) + k*t->nb[0] + row*t->nb[1] + head*t->nb[2] + batch*t->nb[3];
}
float read(const ggml_tensor * t, size_t k, size_t row, size_t head, size_t batch) {
    const char * p = address(t, k, row, head, batch);
    if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t half;
        std::memcpy(&half, p, sizeof(half));
        return ggml_fp16_to_fp32(half);
    }
    float value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}
void write(ggml_tensor * t, size_t k, size_t row, size_t head, size_t batch, float value) {
    char * p = address(t, k, row, head, batch);
    if (t->type == GGML_TYPE_F16) {
        const auto half = ggml_fp32_to_fp16(value);
        std::memcpy(p, &half, sizeof(half));
    } else {
        std::memcpy(p, &value, sizeof(value));
    }
}

struct Quantized {
    size_t rows, k, blocks;
    std::vector<block_q8_0> data;
    const block_q8_0 * row(size_t index) const { return data.data() + index*blocks; }
};
void independent_check(const float * x, const block_q8_0 & y) {
    float amax = 0.0f;
    for (size_t lane = 0; lane < 32; ++lane) amax = std::max(amax, std::fabs(x[lane]));
    const float delta = amax/127.0f;
    const float inverse = delta != 0.0f ? 1.0f/delta : 0.0f;
    check(y.d == ggml_fp32_to_fp16(delta), "block scale is the FP16 of amax/127");
    for (size_t lane = 0; lane < 32; ++lane) {
        const float scaled = x[lane]*inverse;
        const double magnitude = std::floor(std::fabs(double(scaled))+0.5);
        check(y.qs[lane] == static_cast<int8_t>(scaled < 0.0f ? -magnitude : magnitude), "code rounds half away from zero");
        check(y.qs[lane] >= -127 && y.qs[lane] <= 127, "code range");
    }
}
Quantized reference_quantize(const ggml_tensor * t, size_t head, size_t batch) {
    Quantized q{size_t(t->ne[1]), size_t(t->ne[0]), size_t(t->ne[0])/32, {}};
    q.data.resize(q.rows*q.blocks);
    std::vector<float> row(q.k);
    for (size_t r = 0; r < q.rows; ++r) {
        for (size_t lane = 0; lane < q.k; ++lane) row[lane] = read(t, lane, r, head, batch);
        quantize_row_q8_0(row.data(), q.data.data()+r*q.blocks, int64_t(q.k));
        for (size_t block = 0; block < q.blocks; ++block)
            independent_check(row.data()+block*32, q.data[r*q.blocks+block]);
    }
    return q;
}
float scalar_dot(const Quantized & weight, size_t col, const Quantized & activation, size_t row, size_t used_blocks) {
    float value = -1.0f;
    ggml_vec_dot_q8_0_q8_0(int(32*used_blocks), &value, 0, weight.row(col), 0, activation.row(row), 0, 1);
    return value;
}
struct Head {
    Quantized a, b;
    std::vector<float> output;
};
Head reference_head(const ggml_tensor * a, const ggml_tensor * b, size_t head, size_t batch) {
    Head result{reference_quantize(b, head, batch),
                reference_quantize(a, head/(b->ne[2]/a->ne[2]), batch/(b->ne[3]/a->ne[3])), {}};
    const size_t rows = result.a.rows, columns = result.b.rows, blocks = result.a.blocks;
    result.output.resize(rows*columns);
    for (size_t row = 0; row < rows; ++row)
        for (size_t col = 0; col < columns; ++col) {
            const float dot = scalar_dot(result.b, col, result.a, row, blocks);
            float own = +0.0f;
            for (size_t block = 0; block < blocks; ++block) {
                int32_t sum = 0;
                for (size_t lane = 0; lane < 32; ++lane)
                    sum += int32_t(result.a.row(row)[block].qs[lane])*int32_t(result.b.row(col)[block].qs[lane]);
                own = own + float(sum)*(ggml_fp16_to_fp32(result.b.row(col)[block].d)*ggml_fp16_to_fp32(result.a.row(row)[block].d));
            }
            check(bits(own) == bits(dot), "independent expression matches the scalar ggml dot");
            result.output[row*columns+col] = dot;
        }
    return result;
}
struct Observation {
    const std::vector<Head> * heads;
    size_t next = 0;
};
void observe(const pb::BlockObservation & block, void * opaque) {
    auto & observation = *static_cast<Observation *>(opaque);
    const size_t blocks = observation.heads->front().a.blocks;
    check(observation.next < blocks*observation.heads->size(), "no extra executor block");
    const auto & h = (*observation.heads)[observation.next/blocks];
    check(block.block == observation.next%blocks, "ascending blocks within each head and batch");
    ++observation.next;
    check(block.m == h.a.rows && block.j == h.b.rows && block.k == 32, "executor dimensions");
    for (size_t row = 0; row < block.m; ++row) {
        const auto & a = h.a.row(row)[block.block];
        check(bits(block.s_a[row]) == bits(ggml_fp16_to_fp32(a.d)), "activation block scale bits");
        for (size_t lane = 0; lane < 32; ++lane)
            check(block.a[row*32+lane] == a.qs[lane], "activation code");
    }
    for (size_t col = 0; col < block.j; ++col) {
        const auto & b = h.b.row(col)[block.block];
        check(bits(block.s_b[col]) == bits(ggml_fp16_to_fp32(b.d)), "weight block scale bits");
        for (size_t lane = 0; lane < 32; ++lane)
            check(block.b[lane*block.j+col] == b.qs[lane], "weight code");
    }
    for (size_t row = 0; row < block.m; ++row)
        for (size_t col = 0; col < block.j; ++col) {
            const auto & a = h.a.row(row)[block.block];
            const auto & b = h.b.row(col)[block.block];
            int64_t dot = 0;
            for (size_t lane = 0; lane < 32; ++lane) dot += int64_t(a.qs[lane])*int64_t(b.qs[lane]);
            check(dot >= -516128 && dot <= 516128, "one-block dot bound");
            check(block.acc[row*block.j+col] == dot, "per-block INT32 bits and head mapping");
            check(bits(block.output[row*block.row_stride+col*block.column_stride]) ==
                  bits(scalar_dot(h.b, col, h.a, row, block.block+1)), "scaled partial sum matches the scalar ggml dot");
        }
}

ggml_tensor * weight_view(ggml_context * ctx, ggml_type type, size_t kv, bool pv, size_t batches) {
    const size_t element = ggml_type_size(type);
    const size_t k = pv ? kv : 64, rows = pv ? 64 : kv;
    const size_t row_stride = k+5, head_stride = row_stride*rows+19, batch_stride = 2*head_stride+23;
    auto * backing = ggml_new_tensor_1d(ctx, type, batches*batch_stride);
    return ggml_view_4d(ctx, backing, k, rows, 2, batches,
                        row_stride*element, head_stride*element, batch_stride*element, 0);
}
ggml_tensor * activation_view(ggml_context * ctx, size_t k, size_t rows, size_t batches) {
    const size_t head_stride = k+3, row_stride = head_stride*14, batch_stride = row_stride*rows+17;
    auto * backing = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, batches*batch_stride);
    return ggml_view_4d(ctx, backing, k, rows, 14, batches,
                        row_stride*sizeof(float), head_stride*sizeof(float), batch_stride*sizeof(float), 0);
}
void fill(ggml_tensor * t, bool probability, int salt) {
    for (size_t batch = 0; batch < size_t(t->ne[3]); ++batch)
        for (size_t head = 0; head < size_t(t->ne[2]); ++head)
            for (size_t row = 0; row < size_t(t->ne[1]); ++row)
                for (size_t k = 0; k < size_t(t->ne[0]); ++k) {
                    const int pattern = int((k*17+row*13+head*29+batch*37+size_t(salt))%255)-127;
                    float value = std::scalbn(float(pattern)+0.5f, int((k/32+head+batch)%5)-10);
                    if (probability) value = std::fabs(value)*0.125f;
                    if (row == 0 && k/32 == 1) value = 0.0f;
                    write(t, k, row, head, batch, value);
                }
}
std::vector<float> run_attention(ggml_context * ctx, ggml_backend_t backend,
                                 ggml_tensor * op, ggml_gemmini_pb_node_type type) {
    const auto * a = op->src[0];
    const auto * b = op->src[1];
    std::vector<Head> heads;
    std::vector<float> expected;
    for (size_t batch = 0; batch < size_t(b->ne[3]); ++batch)
        for (size_t head = 0; head < size_t(b->ne[2]); ++head) {
            heads.push_back(reference_head(a, b, head, batch));
            expected.insert(expected.end(), heads.back().output.begin(), heads.back().output.end());
        }
    check(ggml_backend_supports_op(backend, op), "GEMMINI admits PB attention");
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, op);
    ggml_gemmini_pb_reset_counters();
    pb::reset_tiled_call_count();
    Observation observation{&heads};
    pb::set_block_observer(observe, &observation);
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "real GEMMINI graph execution");
    pb::set_block_observer(nullptr, nullptr);
    const size_t calls = heads.size()*a->ne[0]/32;
    check(observation.next == calls && pb::tiled_call_count() == calls, "observer and real tiled calls prove PB");
    for (size_t i = 0; i < expected.size(); ++i)
        check(bits(static_cast<float *>(op->data)[i]) == bits(expected[i]), "final ordered FP32 bits");
    for (int node_type = 0; node_type < GGML_GEMMINI_PB_NODE_TYPE_COUNT; ++node_type) {
        ggml_gemmini_pb_counters counters{};
        check(ggml_gemmini_pb_get_counters(static_cast<ggml_gemmini_pb_node_type>(node_type), &counters), "counter API");
        check(counters.fallback_count == 0, "no CPU matmul fallback");
        check(counters.nodes == (node_type == type ? heads.size() : 0), "separate QK and PV invocation counts");
        check(counters.k_blocks == (node_type == type ? calls : 0), "separate QK and PV K blocks");
        check(counters.int32_elements == (node_type == type ? calls*a->ne[1]*b->ne[1] : 0), "INT32 readback counts");
    }
    return expected;
}

ggml_tensor * softmax(ggml_context * ctx, const ggml_tensor * shape, const float * input, ggml_tensor * mask) {
    auto * materialized = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                            shape->ne[0], shape->ne[1], shape->ne[2], shape->ne[3]);
    std::memcpy(materialized->data, input, ggml_nbytes(materialized));
    auto * result = ggml_soft_max_ext(ctx, materialized, mask, 0.125f, 0.0f);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    check(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS, "CPU mask and softmax");
    return result;
}

void attention_case(ggml_type cache_type, size_t rows, size_t weight_batches, size_t batches) {
    ggml_context * ctx = ggml_init({64*1024*1024, nullptr, false});
    check(ctx != nullptr, "context allocation");
    auto * backend = ggml_backend_gemmini_init();
    check(backend != nullptr, "GEMMINI backend");
    constexpr size_t kv = 96;
    auto * k = weight_view(ctx, cache_type, kv, false, weight_batches);
    auto * q = activation_view(ctx, 64, rows, batches);
    fill(k, false, 3); fill(q, false, 7);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv, rows);
    for (size_t row = 0; row < rows; ++row)
        for (size_t col = 0; col < kv; ++col)
            static_cast<float *>(mask->data)[row*kv+col] =
                col >= 73 || col > 50+row ? -std::numeric_limits<float>::infinity() : float(int(col%3)-1)*0.0625f;
    auto * qk = ggml_mul_mat(ctx, k, q);
    ggml_set_name(qk, "pb_q8_golden_qk");
    ggml_gemmini_set_attention_role(qk, GGML_GEMMINI_ATTENTION_QK);
    qk->src[2] = mask;
    auto expected = run_attention(ctx, backend, qk, GGML_GEMMINI_PB_QK);
    check(std::isfinite(static_cast<float *>(qk->data)[95]), "QK leaves masked entries unmasked");
    auto * p = softmax(ctx, qk, static_cast<float *>(qk->data), mask);
    auto * reference_p = softmax(ctx, qk, expected.data(), mask);
    for (size_t i = 0; i < expected.size(); ++i)
        check(bits(static_cast<float *>(p->data)[i]) == bits(static_cast<float *>(reference_p->data)[i]),
              "CPU softmax of PB and reference QK bitwise");
    check(static_cast<float *>(p->data)[95] == 0.0f, "padded masked probability is zero");
    const auto calls_before_cpu = pb::tiled_call_count();
    auto * v = weight_view(ctx, cache_type, kv, true, weight_batches);
    fill(v, false, 11);
    check(v->nb[1] > kv*ggml_type_size(cache_type), "PV padded cache row stride");
    auto * p_input = ggml_dup_tensor(ctx, p);
    std::memcpy(p_input->data, p->data, ggml_nbytes(p));
    auto * pv = ggml_mul_mat(ctx, v, p_input);
    ggml_set_name(pv, "pb_q8_golden_pv");
    ggml_gemmini_set_attention_role(pv, GGML_GEMMINI_ATTENTION_PV);
    check(pb::tiled_call_count() == calls_before_cpu, "CPU mask and softmax issue no PB work");
    run_attention(ctx, backend, pv, GGML_GEMMINI_PB_PV);
    ggml_gemmini_attention_stats stats{};
    check(ggml_backend_gemmini_get_attention_stats(backend, &stats) && stats.qk_nodes == 1 && stats.pv_nodes == 1,
          "backend tensor completion counters");

    const auto old = expected;
    fill(k, false, 101);
    expected = run_attention(ctx, backend, qk, GGML_GEMMINI_PB_QK);
    check(std::memcmp(old.data(), expected.data(), expected.size()*sizeof(float)) != 0,
          "dynamic cache requantized each call");
    fill(v, false, 127);
    run_attention(ctx, backend, pv, GGML_GEMMINI_PB_PV);

    auto * bad_a = weight_view(ctx, cache_type, 96, false, 1);
    auto * bad_b = activation_view(ctx, 64, 1, 1);
    auto * bad = ggml_mul_mat(ctx, bad_a, bad_b);
    ggml_set_name(bad, "pb_q8_unsupported_pv");
    ggml_gemmini_set_attention_role(bad, GGML_GEMMINI_ATTENTION_PV);
    check(!ggml_backend_supports_op(backend, bad), "unsupported PV shape rejected");
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, bad);
    std::fill_n(static_cast<float *>(bad->data), ggml_nelements(bad), -99.0f);
    const auto before = pb::tiled_call_count();
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED, "unsupported node fails without fallback");
    check(pb::tiled_call_count() == before, "unsupported node issues no array work");
    for (int64_t i = 0; i < ggml_nelements(bad); ++i)
        check(static_cast<float *>(bad->data)[i] == -99.0f, "unsupported node leaves output unchanged");

    auto * nan_q = activation_view(ctx, 64, 2, 1);
    auto * nan_k = weight_view(ctx, GGML_TYPE_F32, 96, false, 1);
    fill(nan_q, false, 5); fill(nan_k, false, 9);
    write(nan_q, 3, 1, 0, 0, std::numeric_limits<float>::quiet_NaN());
    auto * nan_node = ggml_mul_mat(ctx, nan_k, nan_q);
    ggml_set_name(nan_node, "pb_q8_nan_qk");
    ggml_gemmini_set_attention_role(nan_node, GGML_GEMMINI_ATTENTION_QK);
    auto * nan_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(nan_graph, nan_node);
    const auto before_nan = pb::tiled_call_count();
    check(ggml_backend_graph_compute(backend, nan_graph) == GGML_STATUS_FAILED, "non-finite activation fails by name");
    check(pb::tiled_call_count() == before_nan, "non-finite activation issues no array work");
    ggml_backend_free(backend);
    ggml_free(ctx);
}
}

int main() {
    const bool five_core = pb_test_start_five_core();
    ggml_context * guard = ggml_init({1024, nullptr, true});
    attention_case(GGML_TYPE_F16, 3, 1, 2);
    attention_case(GGML_TYPE_F32, 35, 2, 4);
    ggml_free(guard);
    pb_test_finish_five_core(five_core);
    std::printf("gemmini-pb-q8-0-attention-test: PASS checks=%llu\n", static_cast<unsigned long long>(checks));
    return 0;
}
