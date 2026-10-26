// Real PoTal attention graphs. Integer goldens are independent of the executor;
// the separate CPU backend supplies FP32 fidelity measurements, not HW evidence.
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-gemmini.h"
#include "ggml-potal.h"
#include "matmul/attention.hpp"
#include "matmul/dense.hpp"
#include "support/attention-golden.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
namespace attention = ggml::potal::attention;
namespace golden = potal_test::attention_golden;
using namespace ggml::potal;
int failures = 0;

void check(const char * name, bool value) {
    std::printf("POTAL_ATTENTION_CHECK %s %s\n", value ? "PASS" : "FAIL", name);
    failures += !value;
}

void set(ggml_tensor * tensor, size_t k, size_t row, size_t head, size_t batch, float value) {
    auto * p = static_cast<char *>(tensor->data) + k * tensor->nb[0] + row * tensor->nb[1] +
               head * tensor->nb[2] + batch * tensor->nb[3];
    if (tensor->type == GGML_TYPE_F16) {
        const auto half = ggml_fp32_to_fp16(value);
        std::memcpy(p, &half, sizeof(half));
    } else {
        std::memcpy(p, &value, sizeof(value));
    }
}

ggml_tensor * operand(ggml_context * context, ggml_type type, size_t k, size_t rows, size_t heads,
                       size_t batches, bool strided) {
    auto * base = ggml_new_tensor_4d(context, type, int64_t(k + (strided ? 7 : 0)),
                                    int64_t(rows), int64_t(heads), int64_t(batches));
    return strided ? ggml_view_4d(context, base, int64_t(k), int64_t(rows), int64_t(heads), int64_t(batches),
                                  base->nb[1], base->nb[2], base->nb[3], 0) : base;
}

void run_case(ggml_backend_t hardware, ggml_backend_t cpu, attention::role role, size_t k, size_t rows,
               ggml_type type, bool strided, bool repeats, bool cancellation = false) {
    const size_t n = cancellation ? 1 : 19, heads = repeats ? 2 : 1, repeat = repeats ? 2 : 1;
    const size_t batches = repeats ? 2 : 1;
    ggml_init_params params{ 32u * 1024 * 1024, nullptr, false };
    ggml_context_ptr context(ggml_init(params));
    if (!context) { check("context allocation", false); return; }
    auto * weight = operand(context.get(), type, k, n, heads, batches, strided);
    auto * input = operand(context.get(), GGML_TYPE_F32, k, rows, heads * repeat, batches, strided);
    for (size_t b = 0; b < batches; ++b)
        for (size_t h = 0; h < heads; ++h)
            for (size_t j = 0; j < n; ++j)
                for (size_t x = 0; x < k; ++x) {
                    float value = std::ldexp(j % 2 ? -1.0f : 1.0f, int((h + b + j) % 3));
                    if (cancellation) {
                        value = x == 0 ? 1.0f : x == 16 ? -1.0f :
                                x == 32 ? std::ldexp(1.0f, -24) : x == 48 ? -std::ldexp(1.0f, -24) : 0;
                    }
                    set(weight, x, j, h, b, value);
                }
    for (size_t b = 0; b < batches; ++b)
        for (size_t h = 0; h < heads * repeat; ++h)
            for (size_t i = 0; i < rows; ++i)
                for (size_t x = 0; x < k; ++x) {
                    float value = role == attention::role::pv ? (i % 2 ? 0.5f : 1.0f) :
                                  std::ldexp(i % 2 ? -1.0f : 1.0f, int(i % 3));
                    if (cancellation) value = x == 1 ? 1.0f : 1.0f / 256;
                    set(input, x, i, h, b, value);
                }
    auto * output = ggml_mul_mat(context.get(), weight, input);
    ggml_gemmini_set_attention_role(output, role == attention::role::qk ?
                                    GGML_GEMMINI_ATTENTION_QK : GGML_GEMMINI_ATTENTION_PV);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    ggml_set_name(output, role == attention::role::qk ? "h7b_qk" : "h7b_pv");
    auto * graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, output);
    const size_t count = size_t(ggml_nelements(output));
    std::vector<float> expected(count);
    for (size_t b = 0; b < batches; ++b)
        for (size_t h = 0; h < heads * repeat; ++h) {
            attention::stationary w;
            attention::stream a;
            std::vector<int64_t> integers;
            const bool prepared = attention::prepare_stationary(*weight, h / repeat, b, role, w) == POTAL_OK &&
                                  attention::prepare_stream(*input, h, b, 0, rows, role, a) == POTAL_OK;
            if (!prepared || !golden::evaluate(a, w, role, integers)) {
                check("independent integer golden", false); return;
            }
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < n; ++j)
                    expected[((b * heads * repeat + h) * rows + i) * n + j] = float(std::ldexp(
                        double(integers[i * n + j]), a.scale + int(w.base[j])));
        }
    check("backend explicitly supports attention node", ggml_backend_supports_op(hardware, output));
    ggml_backend_potal_stats before{}, after{};
    check("hardware counters before", ggml_backend_potal_get_stats(hardware, &before));
    const auto start = std::chrono::steady_clock::now();
    const auto status = ggml_backend_graph_compute(hardware, graph);
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    check("hardware counters after", ggml_backend_potal_get_stats(hardware, &after));
    const auto * got = static_cast<const float *>(output->data);
    const bool exact = status == GGML_STATUS_SUCCESS &&
                       std::memcmp(got, expected.data(), count * sizeof(float)) == 0;
    check("hardware integer-domain exactness", exact);
    check("actual PoTal completion and balanced Scale",
          after.attention_completed == before.attention_completed + 1 &&
          after.attention_failed == before.attention_failed && after.attention_loop_ws > before.attention_loop_ws &&
          after.scale_writes - before.scale_writes == after.scale_releases - before.scale_releases &&
          (role == attention::role::qk ? after.attention_qk_shared_acc_jobs > before.attention_qk_shared_acc_jobs :
                                        after.attention_bias_mvins > before.attention_bias_mvins));
    if (!exact) return;
    std::vector<float> hardware_output(got, got + count);
    check("independent CPU FP32 backend", ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    double maximum = 0, squared = 0;
    bool finite = true;
    for (size_t i = 0; i < count; ++i) {
        finite = finite && std::isfinite(hardware_output[i]) && std::isfinite(got[i]);
        const double error = double(hardware_output[i]) - got[i];
        maximum = std::max(maximum, std::abs(error));
        squared += error * error;
    }
    check("finite hardware and FP32 outputs", finite);
    std::printf("POTAL_ATTENTION_HW role=%s K=%zu rows=%zu columns=%zu heads=%zu batches=%zu "
                "head_repeat=%zu strided=%d cancellation=%d exact=%d loop_ws=%llu bias_mvins=%llu "
                "cpu_observed_ms=%.6f fp32_max_abs=%.9g fp32_rmse=%.9g issuer_cpu=%d\n",
                role == attention::role::qk ? "QK" : "PV", k, rows, n, heads, batches, repeat, strided,
                cancellation, exact, (unsigned long long)(after.attention_loop_ws - before.attention_loop_ws),
                (unsigned long long)(after.attention_bias_mvins - before.attention_bias_mvins),
                elapsed, maximum, std::sqrt(squared / count), after.issuer_cpu);
}

void qk_saturation(ggml_backend_t hardware, ggml_backend_t cpu) {
    ggml_init_params params{1u << 20, nullptr, false};
    ggml_context_ptr context(ggml_init(params));
    if (!context) { check("QK saturation context", false); return; }
    auto * weight = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 64, 1);
    auto * input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 64, 1);
    attention::stream a;
    a.rows = 1; a.k = a.kp = 64; a.limb_count = 2;
    a.codes.assign(64, 16384);
    attention::stationary b;
    b.k = b.kp = 64; b.n = 1;
    b.base = {-34}; b.carriers = {20, 0};
    b.limbs[0].resize(64); b.limbs[1].resize(64);
    for (size_t k = 0; k < 64; ++k) {
        const int32_t code = k < 32 && k != 16 ? -129 : 16384;
        const auto digits = golden::digits(code);
        b.limbs[0][k] = digits[0]; b.limbs[1][k] = digits[1];
        set(weight, k, 0, 0, 0, k < 32 ? float(code) / 16384.0f : std::ldexp(1.0f, -20));
        set(input, k, 0, 0, 0, 1);
    }
    std::vector<int64_t> expected, forbidden;
    const bool witness = golden::evaluate(a, b, attention::role::qk, expected) &&
                         golden::evaluate(a, b, attention::role::qk, forbidden, true) &&
                         expected != forbidden;
    check("independent QK saturation witness distinguishes separate K panels", witness);
    if (!witness) return;
    auto * output = ggml_mul_mat(context.get(), weight, input);
    ggml_gemmini_set_attention_role(output, GGML_GEMMINI_ATTENTION_QK);
    auto * graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, output);
    ggml_backend_potal_stats before{}, after{};
    check("QK saturation counters before", ggml_backend_potal_get_stats(hardware, &before));
    const auto status = ggml_backend_graph_compute(hardware, graph);
    check("QK saturation counters after", ggml_backend_potal_get_stats(hardware, &after));
    const float got = *static_cast<const float *>(output->data);
    const float want = float(std::ldexp(double(expected[0]), -48));
    const float wrong = float(std::ldexp(double(forbidden[0]), -48));
    check("real QK lower and upper share the saturating INT32 ACC",
          status == GGML_STATUS_SUCCESS && got == want && got != wrong);
    check("QK saturation hardware completion and balanced Scale",
          after.attention_completed == before.attention_completed + 1 &&
          after.attention_failed == before.attention_failed &&
          after.attention_qk_shared_acc_jobs > before.attention_qk_shared_acc_jobs &&
          after.attention_loop_ws > before.attention_loop_ws &&
          after.scale_writes - before.scale_writes == after.scale_releases - before.scale_releases);
    check("QK saturation independent CPU FP32 backend",
          ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    const float fp32 = *static_cast<const float *>(output->data);
    std::printf("POTAL_ATTENTION_QK_SAT expected_integer=%lld forbidden_integer=%lld "
                "hardware=%.9g expected=%.9g forbidden=%.9g fp32=%.9g fp32_abs=%.9g loop_ws=%llu\n",
                (long long)expected[0], (long long)forbidden[0], got, want, wrong, fp32,
                std::abs(double(got) - fp32),
                (unsigned long long)(after.attention_loop_ws - before.attention_loop_ws));
}

void bias_saturation() {
    hp1_operands operands;
    if (!operands.allocate({16, 16, 32})) { check("bias DMA allocation", false); return; }
    std::unique_ptr<void, decltype(&std::free)> bias(dma_alloc(16 * sizeof(int32_t)), std::free);
    if (!bias) { check("bias allocation", false); return; }
    for (size_t i = 0; i < 16; ++i)
        for (size_t k = 0; k < 32; ++k) operands.a[i * operands.lda + k] = 127;
    for (size_t k = 0; k < 32; ++k)
        for (size_t j = 0; j < 16; ++j) operands.own.b[k * operands.ldb + j] = 1;
    for (size_t j = 0; j < 16; ++j) {
        operands.own.carriers[j] = 24;
        static_cast<int32_t *>(bias.get())[j] = -128;
    }
    potal_stripe_gemm_desc desc{};
    desc.m = 16; desc.n = 16; desc.k = 32;
    desc.a = operands.a; desc.b = operands.b; desc.c = operands.c;
    desc.lda = operands.lda; desc.ldb = operands.ldb; desc.ldc = operands.ldc;
    desc.carriers = operands.carriers; desc.carrier_stride = operands.cstride; desc.scale_block_k = 32;
    desc.tile_i = 1; desc.tile_j = 1; desc.tile_k = 2;
    planned_result completed;
    issuer_owner owner;
    std::string why;
    potal_status status = POTAL_OK;
    const bool started = owner.start([&](const owner_runners &) {
        completed = attention::run_corrected(desc, static_cast<const int32_t *>(bias.get()), 16);
        return consume_outcome{completed.fatal, completed.status, completed.stage};
    }, why, status);
    owner.join();
    bool exact = started && completed.ok;
    for (size_t i = 0; i < 16; ++i)
        for (size_t j = 0; j < 16; ++j) exact = exact && operands.c[i * operands.ldc + j] == INT32_MAX - 128;
    check("real SCU saturation then one final negative INT32 bias", exact);
    if (completed.fatal) { operands.abandon(); (void)bias.release(); }
}
} // namespace

int main() {
    ggml_init_params params{1u << 20, nullptr, true};
    ggml_context_ptr initialization(ggml_init(params));
    ggml_backend_ptr hardware(ggml_backend_potal_init()), cpu(ggml_backend_cpu_init());
    if (!initialization || !hardware || !cpu) return 1;
    ggml_backend_cpu_set_n_threads(cpu.get(), 3);
    run_case(hardware.get(), cpu.get(), attention::role::qk, 64, 4, GGML_TYPE_F16, true, true);
    run_case(hardware.get(), cpu.get(), attention::role::qk, 33, 1, GGML_TYPE_F32, true, false);
    for (size_t k : {size_t(17), size_t(33), size_t(65)})
        run_case(hardware.get(), cpu.get(), attention::role::pv, k, k == 33 ? 4 : 1, GGML_TYPE_F16, true, k == 33);
    run_case(hardware.get(), cpu.get(), attention::role::pv, 64, 1, GGML_TYPE_F32, true, false, true);
    qk_saturation(hardware.get(), cpu.get());
    bias_saturation();
    std::printf("POTAL_ATTENTION_SMOKE %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
