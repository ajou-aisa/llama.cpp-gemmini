// Software-only issuer microbench. Preparation is outside every timed interval.
// Usage: potal-issuer-prepared-bench [staging|descriptor|issuer|all] [iterations]
// Staging/descriptor modes isolate metadata operations. Issuer mode runs the
// real strict executor with a command-hashing recording sink, not hardware.
#include "ggml-cpp.h"
#include "ggml.h"
#include "matmul/dense.hpp"
#include "matmul/paired-auto.hpp"
#include "include/potal-int4.h"
#include "../trace/issuer-profile.h"
#include "../../ggml-npu-common/residual/rmd/rmd-builder.hpp"

#include <chrono>
#include <cerrno>
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ggml::potal;
namespace rmd = ggml::npu::rmd;

namespace {

constexpr size_t PD = POTAL_HW_DIM;
struct free_dma { void operator()(void * p) const { std::free(p); } };

// Native packed-W4 data. The older paired-support.hpp fixture is Q8_HP1 and
// cannot validate a W4 benchmark even though the C++ interfaces are shared.
struct w4_fixture {
    static constexpr size_t M = 16, N = 140, K = 2048;
    ggml_context_ptr ctx;
    hp1_weight weight;
    potal_tile_plan main{};
    std::unique_ptr<int8_t, free_dma> a;
    size_t lda = K;

    w4_fixture() {
        ctx.reset(ggml_init({4 * 1024 * 1024, nullptr, false}));
        if (!ctx) throw std::runtime_error("ggml context");
        auto * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_MXINT4, K, N);
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, M);
        auto * y = ggml_mul_mat(ctx.get(), w, x);
        std::vector<float> values(N * K);
        for (size_t n = 0; n < N; ++n)
            for (size_t k = 0; k < K; ++k)
                values[n * K + k] = std::ldexp(float(int((3 * n + 5 * k) % 15) - 7), int(k / 32 % 4) - 4);
        ggml_quantize_chunk(GGML_TYPE_MXINT4, values.data(), w->data, 0, N, K, nullptr);
        if (potal_frozen_plan(M, N, K, 1, 2, 4, 1, &main) != POTAL_OK)
            throw std::runtime_error("frozen W4 geometry");
        npu_matmul_args args;
        if (!init_args(w, y, {M, N, K}, "qwen2", "issuer-bench",
                       ggml::npu::residual::ResidualRoute::ws_packet, main, args) ||
            !weight.allocate(N, K) || prepare_weight(args, weight) != quants::weight::hp1::status::ok)
            throw std::runtime_error("MXINT4 preparation");
        std::vector<int8_t> logical(M * K);
        for (size_t i = 0; i < logical.size(); ++i) logical[i] = int(i * 5 % 16) - 8;
        a.reset(static_cast<int8_t *>(dma_alloc(32 * K / 2)));
        if (!a || potal_int4_pack_rows(reinterpret_cast<uint8_t *>(a.get()), 32 * K / 2,
                                      K / 2, logical.data(), K, M, K) != POTAL_OK)
            throw std::runtime_error("packed A4");
    }
};

// Deliberately retain the ordinary field-construction baseline in this bench,
// rather than adding a production toggle or a raw command replay interface.
potal_paired_loop_ws ordinary_loop(const potal_paired_request & req,
                                   const potal_paired_prepared & job, const potal_paired_op & op) {
    potal_paired_loop_ws l{};
    l.loop.i_tiles = uint32_t(op.i_tiles);
    l.loop.j_tiles = uint32_t(op.j_tiles);
    l.loop.k_tiles = uint32_t(op.k_tiles);
    l.loop.pad_i = uint32_t(op.pad_i);
    l.loop.pad_j = uint32_t(op.pad_j);
    l.loop.pad_k = uint32_t(op.pad_k);
    l.loop.flags = op.first ? 0 : POTAL_GEMM_ACCUMULATE;
    l.pair = uint32_t(op.pair);
    if (op.kind == potal_paired_op_kind::main_gemm) {
        const auto & panel = job.tile.panels[op.panel];
        l.loop.a = req.a + panel.k_begin / 2;
        l.loop.b = req.weight->b +
            (panel.k_begin * req.weight->ldb + op.j_group * job.tile.tile_J * PD) / 2;
        l.loop.lda = req.lda;
        l.loop.ldb = req.weight->ldb;
        l.loop.passes_per_block = job.tile.passes_per_block;
        l.path = POTAL_PAIRED_MAIN;
    } else {
        // Both descriptor arms use direct staging: lookup cost is isolated in
        // the staging benchmark rather than credited to descriptor preparation.
        const size_t wi = job.staging.op_windows[op.id];
        l.loop.a = job.staging.a(wi);
        l.loop.lda = job.staging.windows[wi].lda;
        l.loop.passes_per_block = 1;
        l.path = POTAL_PAIRED_RESIDUAL;
        l.a_offset_tiles = uint32_t(op.a_offset_tiles);
        l.resident_b = 1;
        l.indexed = 1;
    }
    return l;
}

template<class F>
void measure(const char * name, size_t iterations, F && body) {
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) body();
    const auto stop = std::chrono::steady_clock::now();
    std::printf("%s iterations=%zu ns=%.0f scope=cpu_only\n", name, iterations,
                std::chrono::duration<double, std::nano>(stop - start).count());
}

struct command_digest {
    uint64_t hash = UINT64_C(14695981039346656037), calls = 0;
    static potal_status command(void * context, const potal_paired_cmd_record * c) {
        auto & d = *static_cast<command_digest *>(context);
        // DMA addresses may relocate between prepared jobs. Keep all command
        // fields except addresses in this benchmark's cross-job digest.
        const bool addresses = c->funct == POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB ||
                               c->funct == POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_DC;
        const bool memory_address = c->funct == POTAL_FUNCT_MVOUT || c->funct == POTAL_FUNCT_MVIN;
        d.hash = (d.hash ^ (addresses || memory_address ? 0 : c->rs1)) * UINT64_C(1099511628211);
        d.hash = (d.hash ^ (addresses ? 0 : c->rs2)) * UINT64_C(1099511628211);
        d.hash = (d.hash ^ c->funct ^ (uint64_t(c->xd) << 32)) * UINT64_C(1099511628211);
        ++d.calls;
        return POTAL_OK;
    }
};

int measure_issuer(potal_paired_request req, const potal_paired_prepared & seed, size_t iterations) {
    const size_t ldc = align_up(req.n, PD);
    std::unique_ptr<int32_t, free_dma> main_out(static_cast<int32_t *>(
        dma_alloc(align_up(req.rows, PD) * ldc * sizeof(int32_t))));
    std::unique_ptr<int32_t, free_dma> residual_out(static_cast<int32_t *>(
        dma_alloc(align_up(std::max<size_t>(seed.plan.residual_rows.size(), 1), PD) * ldc * sizeof(int32_t))));
    if (!main_out || !residual_out) return 1;
    req.c_main = main_out.get(); req.ldc_main = ldc;
    req.c_residual = residual_out.get(); req.ldc_residual = ldc;
    double wall_ns = 0, cpu_ns = 0;
    uint64_t expected_commands = 0, expected_hash = 0;
    for (size_t i = 0; i < iterations + 8; ++i) {
        auto plan = seed.plan;
        potal_paired_prepared job;
        const char * stage = nullptr;
        std::string why;
        uint32_t generation = 0;
        req.generation = &generation;
        req.prepared = nullptr;
        if (potal_paired_prepare(req, std::move(plan), job, &stage, &why) != potal_paired_status::ok) return 1;
        req.prepared = &job;
        command_digest digest;
        const potal_paired_sink sink{command_digest::command, &digest, nullptr};
        potal_paired_session * session = nullptr;
        if (potal_paired_session_open(&sink, &session) != POTAL_OK) return 1;
        potal_paired_report report;
        const auto start = std::chrono::steady_clock::now();
        const clock_t cpu_start = std::clock();
        const auto status = potal_paired_matmul_auto(req, session, report);
        const clock_t cpu_end = std::clock();
        const auto end = std::chrono::steady_clock::now();
        if (cpu_start == clock_t(-1) || cpu_end == clock_t(-1) || cpu_end < cpu_start) return 1;
        if (status != potal_paired_status::ok || report.fatal ||
            potal_paired_session_close(session) != POTAL_OK) {
            std::fprintf(stderr, "issuer status=%u stage=%s %s\n", unsigned(status),
                         report.stage ? report.stage : "-", report.detail.c_str());
            return 1;
        }
        if (i == 0) { expected_commands = digest.calls; expected_hash = digest.hash; }
        if (digest.calls != expected_commands || digest.hash != expected_hash) return 1;
        if (i >= 8) {
            wall_ns += std::chrono::duration<double, std::nano>(end - start).count();
            cpu_ns += double(cpu_end - cpu_start) * 1e9 / CLOCKS_PER_SEC;
        }
    }
    std::printf("issuer_cpu iterations=%zu wall_ns_per_job=%.2f cpu_ns_per_job=%.2f "
                "commands_per_job=%" PRIu64 " command_hash=%016" PRIx64 " scope=cpu_recording_only\n",
                iterations, wall_ns / iterations, cpu_ns / iterations, expected_commands, expected_hash);
    potal_issuer_profile_flush(stderr);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    if (mode != "all" && mode != "staging" && mode != "descriptor" && mode != "issuer") return 2;
    size_t iterations = 10000;
    if (argc > 2) {
        char * end = nullptr;
        errno = 0;
        const unsigned long long value = std::strtoull(argv[2], &end, 10);
        if (errno != 0 || argv[2][0] == '-' || end == argv[2] || *end != '\0' ||
            value == 0 || value > SIZE_MAX) return 2;
        iterations = size_t(value);
    }
    w4_fixture fixture;
    const potal_tile_plan main = fixture.main;
    rmd::RmdStripeBuilder builder;
    builder.reset(0, 0, fixture.M, fixture.K, fixture.N, 4);
    // Short fragments, full fragments and tails across several K panels.
    for (size_t block = 0; block < fixture.K / 32; ++block)
        for (size_t k = 0; k < (block % 3 == 0 ? 3u : 32u); ++k)
            if (!builder.add_residual(block % fixture.M, block * 32 + k, block % 2 ? 17 : -3)) return 1;
    const auto packet = builder.finish();
    residual::decoded_packet decoded;
    if (!packet || residual::decode_packet(*packet, decoded) != rmd::RmdStatus::success) return 1;
    potal_paired_request req;
    req.rows = fixture.M; req.n = fixture.N; req.k = fixture.K;
    req.main_plan = &main; req.a = fixture.a.get(); req.lda = fixture.lda;
    req.weight = &fixture.weight; req.residual = &decoded;
    potal_paired_tile_plan tile;
    potal_paired_execution_plan plan;
    potal_paired_prepared job;
    std::string why;
    const char * stage = nullptr;
    if (potal_paired_plan_stripe(main, req.rows, req.n, req.k, &decoded, tile, plan, &why) != potal_paired_status::ok ||
        potal_paired_prepare(req, std::move(plan), job, &stage, &why) != potal_paired_status::ok) {
        std::fprintf(stderr, "prepare stage=%s detail=%s\n", stage ? stage : "plan", why.c_str());
        return 1;
    }
    std::vector<size_t> loop_ops, residual_ops;
    for (const auto & op : job.plan.ops) {
        if (op.kind != potal_paired_op_kind::main_gemm &&
            op.kind != potal_paired_op_kind::residual_gemm) continue;
        loop_ops.push_back(op.id);
        if (op.kind == potal_paired_op_kind::residual_gemm) {
            residual_ops.push_back(op.id);
            size_t wi = 0;
            while (wi < job.staging.windows.size() &&
                   (job.staging.windows[wi].panel != op.panel || job.staging.windows[wi].index != op.window)) ++wi;
            if (wi == job.staging.windows.size() || wi != job.staging.op_windows[op.id]) return 1;
        }
        auto baseline = ordinary_loop(req, job, op);
        auto prepared = job.loops[op.id];
        // Bind identical issuer state, then use the unchanged checked encoder.
        baseline.loop.scale_base = prepared.loop.scale_base = uint32_t(job.plan.scale_windows[op.scale_window].slot_base);
        baseline.loop.scale_generation = prepared.loop.scale_generation = 1;
        potal_gemm_ws_program a{}, b{};
        if (potal_paired_loop_ws_encode(&baseline, &a) != POTAL_OK ||
            potal_paired_loop_ws_encode(&prepared, &b) != POTAL_OK || std::memcmp(&a, &b, sizeof(a)) != 0) {
            std::fprintf(stderr, "descriptor words differ at op=%zu\n", op.id);
            return 1;
        }
    }
    std::printf("loops=%zu residual_ops=%zu staged_windows=%zu descriptor_bytes=%zu descriptor_storage=%zu\n",
                loop_ops.size(), residual_ops.size(), job.staging.windows.size(), sizeof(potal_paired_loop_ws),
                job.loops.size() * sizeof(potal_paired_loop_ws));
    if (mode == "issuer") return measure_issuer(req, job, iterations);
    if (mode == "all" || mode == "staging") {
        for (bool direct : {false, true}) {
            measure(direct ? "staging_direct" : "staging_linear", iterations, [&] {
                for (size_t id : residual_ops) {
                    const auto & op = job.plan.ops[id];
                    size_t wi = 0;
                    if (direct) wi = job.staging.op_windows[id];
                    else while (wi < job.staging.windows.size() &&
                                (job.staging.windows[wi].panel != op.panel || job.staging.windows[wi].index != op.window)) ++wi;
                    const int8_t * a = job.staging.a(wi);
                    asm volatile("" : : "g"(a) : "memory");
                }
            });
        }
    }
    if (mode == "all" || mode == "descriptor") {
        for (bool prepared : {false, true}) {
            measure(prepared ? "descriptor_prepared" : "descriptor_construct", iterations, [&] {
                for (size_t id : loop_ops) {
                    const auto l = prepared ? job.loops[id] : ordinary_loop(req, job, job.plan.ops[id]);
                    asm volatile("" : : "m"(l) : "memory");
                }
            });
        }
    }
    return 0;
}
