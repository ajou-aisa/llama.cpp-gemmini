#include "../pb/pb-exec.hpp"
#include "../matmul/attention.hpp"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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
    size_t rows, k;
    std::vector<int8_t> codes;
    std::vector<float> scales;
};
Quantized reference_quantize(const ggml_tensor * t, size_t head, size_t batch) {
    Quantized q{size_t(t->ne[1]), size_t(t->ne[0]), {}, {}};
    q.codes.resize(q.rows*q.k);
    q.scales.resize(q.rows*q.k/32);
    for (size_t row = 0; row < q.rows; ++row)
        for (size_t block = 0; block < q.k/32; ++block) {
            float maximum = 0;
            for (size_t lane = 0; lane < 32; ++lane)
                maximum = std::max(maximum, std::fabs(read(t, block*32+lane, row, head, batch)));
            int exponent = 0;
            std::frexp(maximum, &exponent);
            const float scale = maximum == 0 ? 1.0f : std::scalbn(1.0f, std::max(-127, exponent-1)-6);
            q.scales[row*(q.k/32)+block] = scale;
            for (size_t lane = 0; lane < 32; ++lane) {
                const double x = double(read(t, block*32+lane, row, head, batch))/double(scale);
                const double magnitude = std::min(127.0, std::floor(std::fabs(x)+0.5));
                q.codes[row*q.k+block*32+lane] = static_cast<int8_t>(x < 0 ? -magnitude : magnitude);
            }
        }
    return q;
}
struct Head {
    Quantized a, b;
    std::vector<int32_t> dots;
    std::vector<float> output;
};
Head reference_head(const ggml_tensor * a, const ggml_tensor * b, size_t head, size_t batch) {
    Head result{reference_quantize(b, head, batch),
                reference_quantize(a, head/(b->ne[2]/a->ne[2]), batch/(b->ne[3]/a->ne[3])), {}, {}};
    const size_t rows = result.a.rows, columns = result.b.rows, k = result.a.k, blocks = k/32;
    result.dots.resize(blocks*rows*columns);
    result.output.assign(rows*columns, +0.0f);
    for (size_t block = 0; block < blocks; ++block)
        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < columns; ++col) {
                int64_t dot = 0;
                for (size_t lane = block*32; lane < block*32+32; ++lane)
                    dot += int64_t(result.a.codes[row*k+lane])*int64_t(result.b.codes[col*k+lane]);
                check(dot >= -516128 && dot <= 516128, "independent wide dot bound");
                result.dots[(block*rows+row)*columns+col] = static_cast<int32_t>(dot);
                float term = static_cast<float>(dot);
                term = term * result.a.scales[row*blocks+block];
                term = term * result.b.scales[col*blocks+block];
                result.output[row*columns+col] = result.output[row*columns+col] + term;
            }
    return result;
}
struct Observation {
    const std::vector<Head> * heads;
    size_t next = 0;
};
void observe(const pb::BlockObservation & block, void * opaque) {
    auto & observation = *static_cast<Observation *>(opaque);
    const size_t blocks = observation.heads->front().a.k/32;
    check(observation.next < blocks*observation.heads->size(), "no extra executor block");
    const auto & h = (*observation.heads)[observation.next/blocks];
    check(block.block == observation.next%blocks, "ascending blocks within each head and batch");
    ++observation.next;
    check(block.m == h.a.rows && block.j == h.b.rows && block.k == 32, "executor dimensions");
    for (size_t row = 0; row < block.m; ++row) {
        check(bits(block.s_a[row]) == bits(h.a.scales[row*blocks+block.block]), "activation scale bits");
        for (size_t lane = 0; lane < 32; ++lane) {
            check(block.a[row*32+lane] == h.a.codes[row*h.a.k+block.block*32+lane], "activation code");
            check(block.a[row*32+lane] >= -127 && block.a[row*32+lane] <= 127, "activation clamp");
        }
        for (size_t col = 0; col < block.j; ++col)
            check(block.acc[row*block.j+col] == h.dots[(block.block*block.m+row)*block.j+col],
                  "per-block INT32 bits and head mapping");
    }
    for (size_t col = 0; col < block.j; ++col) {
        check(bits(block.s_b[col]) == bits(h.b.scales[col*blocks+block.block]), "weight scale bits");
        for (size_t lane = 0; lane < 32; ++lane) {
            check(block.b[lane*block.j+col] == h.b.codes[col*h.b.k+block.block*32+lane], "weight code");
            check(block.b[lane*block.j+col] >= -127 && block.b[lane*block.j+col] <= 127, "weight clamp");
        }
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
                    const int pattern = int((k*17+row*13+head*29+batch*37+salt)%255)-127;
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
    ggml_set_name(qk, "pb_golden_qk");
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
    ggml_set_name(pv, "pb_golden_pv");
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
    ggml_set_name(bad, "pb_unsupported_pv");
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

    auto * flash_q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 2, 2, 1);
    auto * flash_k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 4, 2, 1);
    auto * flash_v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 4, 2, 1);
    auto * flash = ggml_flash_attn_ext(ctx, flash_q, flash_k, flash_v, nullptr, 1.0f, 0.0f, 0.0f);
    ggml_set_name(flash, "pb_flash_attention");
    check(ggml_backend_supports_op(backend, flash), "flash attention stays on GEMMINI instead of moving to the CPU");
    auto * flash_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(flash_graph, flash);
    const auto before_flash = pb::tiled_call_count();
    check(ggml_backend_graph_compute(backend, flash_graph) == GGML_STATUS_FAILED, "flash attention fails by name");
    check(pb::tiled_call_count() == before_flash, "flash attention issues no array work");
    ggml_backend_free(backend);
    ggml_free(ctx);
}
}

int main() {
    attention_case(GGML_TYPE_F16, 3, 1, 2);
    attention_case(GGML_TYPE_F32, 35, 2, 4);
    std::printf("gemmini-pb-attention-test: PASS checks=%llu\n", static_cast<unsigned long long>(checks));
    return 0;
}
