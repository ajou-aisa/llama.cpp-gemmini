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

// Link this executable with both --wrap options. Recording never allocates or
// throws: the owner binds storage reserved before the first possible DMA.
namespace {
struct command_record {
    potal_loop_ws loop{};
    potal_status status = POTAL_OK;
    bool validated = false;
};
struct command_probe {
    command_record * records = nullptr;
    size_t capacity = 0, used = 0;
    bool overflow = false;
};
thread_local command_probe * active_probe = nullptr;

void record_command(const potal_loop_ws & loop, potal_status status, bool validated) noexcept {
    if (!active_probe) return;
    if (active_probe->used == active_probe->capacity) {
        active_probe->overflow = true;
        return;
    }
    active_probe->records[active_probe->used++] = {loop, status, validated};
}
}

extern "C" potal_status __real_potal_loop_ws_issue(const potal_loop_ws *);
extern "C" potal_status __real_potal_gemm_ws_issue(const potal_gemm_ws *);
extern "C" potal_status __wrap_potal_loop_ws_issue(const potal_loop_ws * loop) {
    const auto status = __real_potal_loop_ws_issue(loop);
    if (active_probe && loop) record_command(*loop, status, false);
    return status;
}
extern "C" potal_status __wrap_potal_gemm_ws_issue(const potal_gemm_ws * gemm) {
    const auto status = __real_potal_gemm_ws_issue(gemm);
    if (active_probe && gemm) {
        potal_loop_ws loop{};
        loop.a = gemm->a; loop.b = gemm->b; loop.c = gemm->c;
        loop.lda = gemm->lda; loop.ldb = gemm->ldb; loop.ldc = gemm->ldc;
        loop.i_tiles = (gemm->m + POTAL_HW_DIM - 1) / POTAL_HW_DIM;
        loop.j_tiles = (gemm->n + POTAL_HW_DIM - 1) / POTAL_HW_DIM;
        loop.k_tiles = (gemm->k + POTAL_HW_DIM - 1) / POTAL_HW_DIM;
        loop.pad_i = loop.i_tiles * POTAL_HW_DIM - gemm->m;
        loop.pad_j = loop.j_tiles * POTAL_HW_DIM - gemm->n;
        loop.pad_k = loop.k_tiles * POTAL_HW_DIM - gemm->k;
        loop.scale_base = gemm->scale_base;
        loop.scale_generation = gemm->scale_generation;
        loop.passes_per_block = gemm->passes_per_block; loop.flags = gemm->flags;
        record_command(loop, status, true);
    }
    return status;
}

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

struct expected_job {
    size_t source_head, source_batch, head, batch, begin, q_limb, k_limb;
    size_t rows, n, kp;
    potal_tile_plan plan;
    std::vector<int8_t> a, b;
    std::vector<uint32_t> carriers;
    std::vector<int32_t> c;
    std::vector<int32_t> actual_c;
    std::vector<command_record> commands;
};

// Precompute in execute() order, including source-head/source-batch reuse,
// output repeats, stripe anchors and zero-valued dense Q limbs.
bool raw_qk(ggml_tensor * output, bool witness = false, std::vector<float> * expected_output = nullptr) {
    const auto & weight = *output->src[0];
    const auto & input = *output->src[1];
    const size_t rows = size_t(input.ne[1]), n = size_t(weight.ne[1]);
    const size_t kp = (size_t(input.ne[0]) + 31) / 32 * 32;
    const size_t hr = size_t(input.ne[2] / weight.ne[2]);
    const size_t br = size_t(input.ne[3] / weight.ne[3]);
    potal_tile_plan stripes;
    if (potal_tile_select(rows, n, kp, 2, &stripes) != POTAL_OK) return false;
    std::vector<expected_job> jobs;
    std::vector<float> final(size_t(ggml_nelements(output)));
    bool negative_high = false;
    size_t nonzero_q_limbs = 0;
    for (size_t sb = 0; sb < size_t(weight.ne[3]); ++sb)
        for (size_t sh = 0; sh < size_t(weight.ne[2]); ++sh) {
            attention::stationary w;
            if (attention::prepare_stationary(weight, sh, sb, attention::role::qk, w) != POTAL_OK) return false;
            negative_high = negative_high || std::any_of(w.limbs[1].begin(), w.limbs[1].end(),
                                                         [](int8_t x) { return x < 0; });
            for (size_t b = sb * br; b < (sb + 1) * br; ++b)
                for (size_t h = sh * hr; h < (sh + 1) * hr; ++h)
                    for (size_t begin = 0; begin < rows; begin += stripes.tile_I * POTAL_HW_DIM) {
                        attention::stream a;
                        if (attention::prepare_stream(input, h, b, begin,
                                std::min(stripes.tile_I * POTAL_HW_DIM, rows - begin),
                                attention::role::qk, a) != POTAL_OK) return false;
                        std::vector<int64_t> integers;
                        if (!golden::evaluate(a, w, attention::role::qk, integers)) return false;
                        for (size_t i = 0; i < a.rows; ++i)
                            for (size_t j = 0; j < n; ++j)
                                final[((b * size_t(input.ne[2]) + h) * rows + begin + i) * n + j] =
                                    float(std::ldexp(double(integers[i * n + j]), a.scale + int(w.base[j])));
                        size_t active = 0;
                        for (size_t q = 0; q < a.limb_count; ++q) {
                            std::vector<int8_t> digits(a.rows * kp);
                            for (size_t x = 0; x < digits.size(); ++x) digits[x] = golden::digits(a.codes[x])[q];
                            active += std::any_of(digits.begin(), digits.end(), [](int8_t x) { return x != 0; });
                            for (size_t pass = 0; pass < 2; ++pass) {
                                expected_job job{};
                                job.source_head = sh; job.source_batch = sb;
                                job.head = h; job.batch = b; job.begin = begin;
                                job.q_limb = q; job.k_limb = pass;
                                job.rows = a.rows; job.n = n; job.kp = kp;
                                if (potal_tile_select(a.rows, n, kp * 2, 2, &job.plan) != POTAL_OK) return false;
                                job.plan.tile_K = std::min(job.plan.tile_K, kp / POTAL_HW_DIM);
                                while (kp / POTAL_HW_DIM % job.plan.tile_K || job.plan.tile_K % 2) --job.plan.tile_K;
                                job.a = digits; job.b = w.limbs[pass]; job.carriers = w.carriers;
                                job.c = golden::qk_pass(a, w, q, pass);
                                job.actual_c.resize(job.c.size());
                                const size_t ni = (a.rows + job.plan.tile_I * POTAL_HW_DIM - 1) /
                                                  (job.plan.tile_I * POTAL_HW_DIM);
                                const size_t nj = (n + job.plan.tile_J * POTAL_HW_DIM - 1) /
                                                  (job.plan.tile_J * POTAL_HW_DIM);
                                job.commands.resize(ni * nj * (kp / (job.plan.tile_K * POTAL_HW_DIM)));
                                if (witness && q == 1 &&
                                    (job.c.size() != 1 || job.c[0] != (pass ? 1073872895 : INT32_MAX))) return false;
                                jobs.push_back(std::move(job));
                            }
                        }
                        nonzero_q_limbs = std::max(nonzero_q_limbs, active);
                    }
        }
    issuer_owner owner;
    std::string why;
    potal_status start_status = POTAL_OK;
    attention::result result;
    size_t next = 0;
    uint32_t previous_generation = 0;
    uint64_t commands = 0, stores = 0, fences = 0;
    bool raw_exact = true, epochs_exact = true, identity = true;
    const bool started = owner.start([&](const owner_runners & runners) {
        result = attention::execute(output, [&](const potal_stripe_gemm_desc & d) {
            planned_result refused;
            refused.status = POTAL_ERR_BAD_RESPONSE; refused.stage = "smoke_raw_contract";
            if (next == jobs.size()) return refused;
            auto & e = jobs[next++];
            bool same = d.m == e.rows && d.n == e.n && d.k == e.kp && d.scale_block_k == 32 &&
                        d.tile_i == e.plan.tile_I && d.tile_j == e.plan.tile_J && d.tile_k == e.plan.tile_K &&
                        d.work_kind == (e.q_limb ? POTAL_STRIPE_WORK_RESIDUAL : POTAL_STRIPE_WORK_MAIN) &&
                        d.a && d.b && d.c && d.carriers && d.lda >= e.kp && d.ldb >= e.n &&
                        d.ldc >= e.n && d.carrier_stride >= e.n;
            if (same) {
                for (size_t i = 0; i < e.rows; ++i)
                    for (size_t k = 0; k < e.kp; ++k) same = same && d.a[i * d.lda + k] == e.a[i * e.kp + k];
                // lda can exceed the logical padded K for DMA alignment.
                for (size_t i = 0; i < e.rows; ++i)
                    for (size_t k = e.kp; k < d.lda; ++k) same = same && d.a[i * d.lda + k] == 0;
                for (size_t k = 0; k < e.kp; ++k)
                    for (size_t j = 0; j < e.n; ++j) same = same && d.b[k * d.ldb + j] == e.b[k * e.n + j];
                for (size_t k = 0; k < e.kp / 32; ++k)
                    for (size_t j = 0; j < e.n; ++j)
                        same = same && d.carriers[k * d.carrier_stride + j] == e.carriers[k * e.n + j];
            }
            identity = identity && same;
            if (!same) return refused; // Nothing has issued for this descriptor.
            command_probe probe{e.commands.data(), e.commands.size(), 0, false};
            active_probe = &probe; // Set only on the actual accelerator owner.
            auto completed = runners.planned(d);
            active_probe = nullptr;
            if (!completed.ok) return completed; // Do not inspect C or alter fatal ownership.
            bool epoch = !probe.overflow && probe.used == e.commands.size() && probe.used > 0;
            size_t at = 0, expected_stores = 0, validated = 0;
            uint64_t expected_scales = 0;
            for (size_t i = 0; i < e.rows; i += d.tile_i * POTAL_HW_DIM)
                for (size_t j = 0; j < e.n; j += d.tile_j * POTAL_HW_DIM)
                    for (size_t k = 0; k < e.kp; k += d.tile_k * POTAL_HW_DIM, ++at) {
                        const size_t mi = std::min(size_t(d.tile_i) * POTAL_HW_DIM, e.rows - i);
                        const size_t nj = std::min(size_t(d.tile_j) * POTAL_HW_DIM, e.n - j);
                        const size_t kk = std::min(size_t(d.tile_k) * POTAL_HW_DIM, e.kp - k);
                        const bool last = k + kk == e.kp;
                        expected_stores += last;
                        expected_scales += kk / 32 * nj;
                        if (at >= probe.used) { epoch = false; continue; }
                        const auto & r = e.commands[at];
                        const auto & l = r.loop;
                        validated += r.validated;
                        const uint32_t generation = previous_generation % POTAL_GENERATION_MAX + 1;
                        epoch = epoch && r.status == POTAL_OK && l.a == d.a + i * d.lda + k &&
                            l.b == d.b + k * d.ldb + j &&
                            l.c == (last ? d.c + i * d.ldc + j : nullptr) &&
                            l.lda == d.lda && l.ldb == d.ldb && l.ldc == (last ? d.ldc : 0) &&
                            l.flags == (k ? POTAL_GEMM_ACCUMULATE : 0) &&
                            l.i_tiles == (mi + POTAL_HW_DIM - 1) / POTAL_HW_DIM &&
                            l.j_tiles == (nj + POTAL_HW_DIM - 1) / POTAL_HW_DIM &&
                            l.k_tiles == kk / POTAL_HW_DIM &&
                            l.pad_i == (POTAL_HW_DIM - mi % POTAL_HW_DIM) % POTAL_HW_DIM &&
                            l.pad_j == (POTAL_HW_DIM - nj % POTAL_HW_DIM) % POTAL_HW_DIM && l.pad_k == 0 &&
                            l.passes_per_block == 2 && l.scale_generation > 0 &&
                            (!previous_generation || l.scale_generation == generation);
                        previous_generation = l.scale_generation;
                        stores += r.status == POTAL_OK && l.c != nullptr;
                        commands += r.status == POTAL_OK;
                    }
            const auto & r = completed.report;
            epoch = epoch && r.loop_ws == probe.used && r.validated_gemms == validated &&
                    r.fences == probe.used && r.last_generation == previous_generation &&
                    r.scale_writes == expected_scales && r.scale_releases == expected_scales &&
                    completed.tlb_flushes == 1;
            size_t actual_stores = 0;
            for (size_t x = 0; x < probe.used; ++x)
                actual_stores += e.commands[x].status == POTAL_OK && e.commands[x].loop.c != nullptr;
            epoch = epoch && actual_stores == expected_stores;
            fences += r.fences;
            epochs_exact = epochs_exact && epoch;
            // Only a successful, fenced runner permits reading DMA output.
            if (r.fences == r.loop_ws && r.loop_ws) {
                for (size_t i = 0; i < e.rows; ++i)
                    for (size_t j = 0; j < e.n; ++j) {
                        e.actual_c[i * e.n + j] = d.c[i * d.ldc + j];
                        raw_exact = raw_exact && e.actual_c[i * e.n + j] == e.c[i * e.n + j];
                    }
            } else raw_exact = false;
            if (!epoch || !raw_exact) {
                completed.ok = false; completed.status = POTAL_ERR_BAD_RESPONSE;
                completed.stage = "smoke_raw_or_epoch";
            }
            return completed;
        });
        return consume_outcome{result.fatal, result.status, result.stage};
    }, why, start_status);
    owner.join();
    const bool success = started && result.status == POTAL_OK && !result.fatal && next == jobs.size();
    check("QK original-K A/B/carriers and frozen doubled-K selection", success && identity);
    check("QK raw signed32 low/high C independent full-pass exactness", success && raw_exact);
    check("QK actual overwrite/accumulate/store generations and fences", success && epochs_exact &&
          commands > 0 && stores > 0 && fences == commands && result.loops == commands &&
          result.qk_limb_jobs == jobs.size() && result.scale_writes == result.scale_releases);
    const bool final_exact = success && std::memcmp(output->data, final.data(), final.size() * sizeof(float)) == 0;
    check("QK final F32 stores independent of raw INT32 proof", final_exact);
    if (!witness) check("QK negative K-high and multiple nonzero Q limbs", negative_high && nonzero_q_limbs >= 2);
    if (witness && success)
        for (const auto & job : jobs)
            std::printf("POTAL_ATTENTION_WITNESS_RAW source_head=%zu source_batch=%zu head=%zu batch=%zu "
                        "stripe_begin=%zu q_limb=%zu k_limb=%zu actual_signed32=%d expected_signed32=%d "
                        "commands=%zu first_generation=%u last_generation=%u\n",
                        job.source_head, job.source_batch, job.head, job.batch, job.begin,
                        job.q_limb, job.k_limb, job.actual_c[0], job.c[0], job.commands.size(),
                        job.commands.front().loop.scale_generation, job.commands.back().loop.scale_generation);
    std::printf("POTAL_ATTENTION_RAW K=%zu jobs=%zu commands=%llu stores=%llu fences=%llu "
                "head_repeat=%zu batch_repeat=%zu negative_k_high=%d nonzero_q_limbs=%zu "
                "raw_exact=%d epochs_exact=%d final_exact=%d status=%d stage=%s\n",
                size_t(input.ne[0]), jobs.size(), (unsigned long long)commands, (unsigned long long)stores,
                (unsigned long long)fences, hr, br, negative_high, nonzero_q_limbs,
                raw_exact, epochs_exact, final_exact, int(result.status), result.stage ? result.stage : "none");
    if (expected_output) *expected_output = std::move(final);
    return success && raw_exact && epochs_exact && final_exact;
}

void run_case(ggml_backend_t hardware, ggml_backend_t cpu, attention::role role, size_t k, size_t rows,
               ggml_type type, bool strided, bool repeats, bool cancellation = false) {
    const size_t n = cancellation ? 1 : 19, heads = repeats ? 2 : 1, repeat = repeats ? 2 : 1;
    const size_t batches = repeats ? 2 : 1;
    const size_t input_batches = batches * (role == attention::role::qk ? repeat : 1);
    ggml_init_params params{ 32u * 1024 * 1024, nullptr, false };
    ggml_context_ptr context(ggml_init(params));
    if (!context) { check("context allocation", false); return; }
    auto * weight = operand(context.get(), type, k, n, heads, batches, strided);
    auto * input = operand(context.get(), role == attention::role::qk ? type : GGML_TYPE_F32,
                           k, rows, heads * repeat, input_batches, strided);
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
    for (size_t b = 0; b < input_batches; ++b)
        for (size_t h = 0; h < heads * repeat; ++h)
            for (size_t i = 0; i < rows; ++i)
                for (size_t x = 0; x < k; ++x) {
                    float value = role == attention::role::pv ? (i % 2 ? 0.5f : 1.0f) :
                                  std::ldexp((i % 2 ? -1.0f : 1.0f) *
                                      (1.0f + float((x + i + h + b) % 7) / 1024), int(i % 3));
                    if (cancellation) value = x == 1 ? 1.0f : 1.0f / 256;
                    set(input, x, i, h, b, value);
                }
    auto * output = ggml_mul_mat(context.get(), weight, input);
    ggml_gemmini_set_attention_role(output, role == attention::role::qk ?
                                    GGML_GEMMINI_ATTENTION_QK : GGML_GEMMINI_ATTENTION_PV);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    ggml_set_name(output, role == attention::role::qk ? "h7b_qk" : "h7b_pv");
    const size_t count = size_t(ggml_nelements(output));
    std::vector<float> expected(count);
    if (role == attention::role::qk) {
        const bool accepted = raw_qk(output, false, &expected);
        check("QK raw acceptance preparation and execution completed", accepted);
        if (!accepted) return;
    }
    auto * graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, output);
    // QK already has an independent, actual-stripe golden from raw_qk.
    // Preserve the existing PV fixtures and their original preparation.
    if (role == attention::role::pv) for (size_t b = 0; b < input_batches; ++b)
        for (size_t h = 0; h < heads * repeat; ++h) {
            attention::stationary w;
            attention::stream a;
            std::vector<int64_t> integers;
            const bool prepared = attention::prepare_stationary(*weight, h / repeat, b / (input_batches / batches), role, w) == POTAL_OK &&
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
          (role == attention::role::qk ? after.attention_qk_limb_jobs > before.attention_qk_limb_jobs :
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
    const auto low = golden::qk_pass(a, b, 1, 0), high = golden::qk_pass(a, b, 1, 1);
    check("independent QK destructive witness constants", witness && low[0] == INT32_MAX &&
          high[0] == 1073872895 && expected[0] == 70927089860352LL && forbidden[0] == 549755813632LL);
    if (!witness) return;
    auto * output = ggml_mul_mat(context.get(), weight, input);
    ggml_gemmini_set_attention_role(output, GGML_GEMMINI_ATTENTION_QK);
    const bool accepted = raw_qk(output, true);
    check("QK destructive witness raw acceptance completed", accepted);
    if (!accepted) return;
    auto * graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, output);
    ggml_backend_potal_stats before{}, after{};
    check("QK saturation counters before", ggml_backend_potal_get_stats(hardware, &before));
    const auto status = ggml_backend_graph_compute(hardware, graph);
    check("QK saturation counters after", ggml_backend_potal_get_stats(hardware, &after));
    const float got = *static_cast<const float *>(output->data);
    const float want = float(std::ldexp(double(expected[0]), -48));
    const float wrong = float(std::ldexp(double(forbidden[0]), -48));
    check("real QK independent ACC epochs and checked Host weighting",
          status == GGML_STATUS_SUCCESS && got == want && got != wrong && got == 0.251983642578125f);
    check("QK saturation hardware completion and balanced Scale",
          after.attention_completed == before.attention_completed + 1 &&
          after.attention_failed == before.attention_failed &&
          after.attention_qk_limb_jobs > before.attention_qk_limb_jobs &&
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

bool pv_range() {
    // Historical shape and rejection class, not unrecorded H3-7 tensor values.
    // Heads 0..10 keep the original path; head 11 has wide positive/negative
    // biases. A normalized P row puts its large code in a zero-V microtile.
    constexpr size_t k = 288, n = 64, heads = 12;
    ggml_init_params params{4u << 20, nullptr, false};
    ggml_context_ptr context(ggml_init(params));
    if (!context) { check("PV range context", false); return false; }
    auto * weight = ggml_new_tensor_4d(context.get(), GGML_TYPE_F16, k, n, heads, 1);
    auto * input = ggml_new_tensor_4d(context.get(), GGML_TYPE_F32, k, 1, heads, 1);
    for (size_t h = 0; h < heads; ++h) {
        for (size_t x = 0; x < k; ++x) {
            set(input, x, 0, h, 0, x == 0 ? std::ldexp(1.0f, -16) :
                                      x == 16 ? 1.0f - std::ldexp(1.0f, -16) : 0);
            for (size_t j = 0; j < n; ++j) {
                const float sign = j % 2 ? -1.0f : 1.0f;
                set(weight, x, j, h, 0, x == 0 ? sign :
                    x == 32 ? std::ldexp(sign, h + 1 == heads ? -24 : -1) : 0);
            }
        }
    }
    bool prepared = true;
    for (size_t h = 0; h < heads; ++h) {
        attention::stationary w;
        attention::stream a;
        std::vector<int32_t> correction{12345};
        prepared = prepared &&
            attention::prepare_stationary(*weight, h, 0, attention::role::pv, w) == POTAL_OK &&
            attention::prepare_stream(*input, h, 0, 0, 1, attention::role::pv, a) == POTAL_OK;
        if (!prepared) break;
        const auto status = attention::column_bias(w, correction);
        prepared = a.scale == -16 && a.codes[0] == 1 && a.codes[16] == 65535 &&
            status == (h + 1 == heads ? POTAL_ERR_UNSUPPORTED : POTAL_OK);
        if (h + 1 == heads) {
            prepared = prepared && correction == std::vector<int32_t>{12345};
            for (size_t j = 0; prepared && j < n; ++j) {
                prepared = w.base[j] == -30 && w.carriers[j] == 24 &&
                    w.carriers[n + j] == 0 &&
                    w.limbs[0][j] == (j % 2 ? -64 : 64) &&
                    w.limbs[0][32 * n + j] == (j % 2 ? -64 : 64);
            }
        }
    }
    check("PV range unchanged codes and exact INT32 bias rejection", prepared);
    if (!prepared) return false;
    auto * output = ggml_mul_mat(context.get(), weight, input);
    ggml_gemmini_set_attention_role(output, GGML_GEMMINI_ATTENTION_PV);
    const auto result = attention::execute_hardware(output);
    const bool success = result.status == POTAL_OK && !result.fatal;
    check("PV range real hardware execution", success);
    if (!success) return false;
    bool exact = true;
    for (size_t h = 0; h < heads; ++h) {
        for (size_t j = 0; j < n; ++j) {
            float actual;
            const auto * source = static_cast<const char *>(output->data) +
                                  h * output->nb[2] + j * output->nb[0];
            std::memcpy(&actual, source, sizeof(actual));
            exact = exact && actual == std::ldexp(j % 2 ? -1.0f : 1.0f, -16);
        }
    }
    const bool commands = result.jobs == 49 && result.pv_bitplane_stripes == 1 &&
        result.pv_bitplane_jobs == 16 && result.correction_jobs == 11 && result.bias_updates == 44 &&
        result.qk_limb_jobs == 0 && result.loops >= result.jobs && result.fences >= result.loops &&
        result.scale_writes > 0 && result.scale_writes == result.scale_releases;
    check("PV range exact independently calculated F32 output", exact);
    check("PV range old bias and new plane commands balanced", commands);
    std::printf("POTAL_ATTENTION_PV_RANGE M=1 N=%zu K=%zu heads=%zu "
                "jobs=%llu bitplane_stripes=%llu bitplane_jobs=%llu corrections=%llu bias_mvins=%llu "
                "loop_ws=%llu fences=%llu scale_writes=%llu scale_releases=%llu exact=%d status=%d\n",
                n, k, heads, (unsigned long long)result.jobs,
                (unsigned long long)result.pv_bitplane_stripes, (unsigned long long)result.pv_bitplane_jobs,
                (unsigned long long)result.correction_jobs, (unsigned long long)result.bias_updates,
                (unsigned long long)result.loops, (unsigned long long)result.fences,
                (unsigned long long)result.scale_writes, (unsigned long long)result.scale_releases,
                exact, int(result.status));
    return exact && commands;
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

int main(int argc, char ** argv) {
    const bool multi_panel_only = argc == 2 && std::strcmp(argv[1], "--multi-panel-only") == 0;
    const bool pv_range_only = argc == 2 && std::strcmp(argv[1], "--pv-range-only") == 0;
    if (argc != 1 && !multi_panel_only && !pv_range_only) {
        std::fprintf(stderr, "Usage: %s [--multi-panel-only|--pv-range-only]\n", argv[0]);
        return 2;
    }
    ggml_init_params params{1u << 20, nullptr, true};
    ggml_context_ptr initialization(ggml_init(params));
    ggml_backend_ptr hardware(ggml_backend_potal_init()), cpu(ggml_backend_cpu_init());
    if (!initialization || !hardware || !cpu) return 1;
    ggml_backend_cpu_set_n_threads(cpu.get(), 3);
    if (pv_range_only) {
        const bool success = pv_range();
        std::printf("POTAL_ATTENTION_SMOKE %s\n", success && !failures ? "PASS" : "FAIL");
        return success && !failures ? 0 : 1;
    }
    if (!multi_panel_only) {
        run_case(hardware.get(), cpu.get(), attention::role::qk, 64, 4, GGML_TYPE_F16, true, true);
        run_case(hardware.get(), cpu.get(), attention::role::qk, 33, 1, GGML_TYPE_F32, true, false);
        run_case(hardware.get(), cpu.get(), attention::role::qk, 65, 17, GGML_TYPE_F32, true, true);
        for (size_t k : {size_t(17), size_t(33), size_t(65)})
            run_case(hardware.get(), cpu.get(), attention::role::pv, k, k == 33 ? 4 : 1, GGML_TYPE_F16, true, k == 33);
        run_case(hardware.get(), cpu.get(), attention::role::pv, 64, 1, GGML_TYPE_F32, true, false, true);
        qk_saturation(hardware.get(), cpu.get());
        bias_saturation();
    }
    // Three frozen K panels per stationary limb exercise actual later-panel
    // accumulation and a final-only readback, including a one-element K tail.
    run_case(hardware.get(), cpu.get(), attention::role::qk, 4097, 1, GGML_TYPE_F32, true, false);
    std::printf("POTAL_ATTENTION_SMOKE %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
