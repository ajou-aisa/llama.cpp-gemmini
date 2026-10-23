// FireSim hardware test of the paired route (GGML_POTAL_PAIRED_AUTO, H3-2B): F32 x Q8_HP1
// MUL_MAT graphs computed by the POTAL backend in STRIPE_PIPELINE with compensation ON,
// each stripe's Main and Residual GEMMs issued as one paired execution
// (potal_paired_matmul_auto on the hart-0 issuer: paired LOOP_WS, INDEX_CONFIG /
// INDEX_WRITE, Indexed Residual LOOP_WS on the resident weight, ACC0 / ACC1
// readback). Every dst is compared element-exactly with the golden of the existing
// stripe pipeline (support/stripe-golden.hpp: FULL-style ExSIA of the same input,
// reference_gemm_i32, compact-run residual golden, merge formula). Device commands
// come only from the backend's issuer; no handwritten H2 command is issued here.
//
//   potal-paired-smoke           quick + full cases, then serial-vs-paired timing
//   potal-paired-smoke --quick   one small case
//
// Each case runs three times in this process: paired (GGML_POTAL_PAIRED_AUTO=ON, RTL
// ordering), paired with GGML_POTAL_PAIRED_FENCES=CONSERVATIVE (a fence before every
// Indexed LOOP_WS, readback and release: isolates an ordering hazard if only the RTL
// mode fails) and serial (OFF); all must match the golden bitwise. An untimed warm-up
// brings the issuer up before any measurement; the route order alternates per case.
// Timing lines report the measured steady-clock wall time of ggml_backend_graph_compute
// (one run each, valid across harts) and the backend's pipeline wall time; they are
// not a claimed speedup.
//
// Markers:
//   POTAL_PAIRED_CASE PASS|FAIL   per case and route: compute OK, dst == golden bitwise,
//                                 stripes == plan's stripes; paired route: paired +
//                                 fallback stripes == stripes, serial Main GEMMs ==
//                                 fallback stripes (exactly once), INDEX_CONFIG ==
//                                 Indexed LOOP_WS, readbacks > 0, SCALE_WRITE == RELEASE,
//                                 merges == golden packets
//   POTAL_PAIRED_PERF             per case: measured wall time of the three runs
//   POTAL_PAIRED_SMOKE PASS       all cases, at least one paired stripe with Indexed
//                                 Residual GEMMs, no failed node; exit 0 only then

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-potal.h"
#include "ggml-quants.h"
#include "log.hpp"
#include "matmul/dense.hpp"
#include "matmul/options.hpp"
#include "matmul/tiling.hpp"
#include "potal.h"
#include "support/stripe-golden.hpp"

#include "ggml-gemmini/matmul/options.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace ggml::potal;
namespace hp1 = ggml::potal::quants::weight::hp1;

namespace {

int      failures         = 0;
uint64_t paired_total     = 0;
uint64_t indexed_total    = 0;

ggml_backend_potal_stats stats_of(ggml_backend_t potal) {
    ggml_backend_potal_stats st = {};
    ggml_backend_potal_get_stats(potal, &st);
    return st;
}

ggml_tensor * hp1_weight_tensor(ggml_context * ctx, int64_t K, int64_t N, std::mt19937 & rng) {
    ggml_tensor *                   w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_HP1, K, N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>              f(N * K);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            const float mag[] = { 1.0f, 8.0f, 0.125f, 64.0f };
            f[n * K + k]      = nd(rng) * 0.02f * mag[(k / 32 + n) % 4];
        }
    }
    ggml_quantize_chunk(GGML_TYPE_Q8_HP1, f.data(), w->data, 0, N, K, nullptr);
    return w;
}

// Outliers on every `step`-th row, `blocks` per row; magnitude `outlier`.
ggml_tensor * activations(ggml_context * ctx, int64_t K, int64_t M, std::mt19937 & rng, float outlier, int blocks,
                          int64_t step) {
    ggml_tensor *                   x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float *                         d = static_cast<float *>(x->data);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            d[m * K + k] = nd(rng) * (1.0f + 0.5f * (float) (m % 3));
        }
        if (outlier != 0.0f && m % step == 0) {
            for (int b = 0; b < blocks; ++b) {
                d[m * K + (m * 7 + b * 37) % K] = outlier * ((m + b) % 4 ? -1.0f : 1.0f);
            }
        }
    }
    return x;
}

struct golden_result {
    bool                               ok = false;
    size_t                             stripes = 0;
    potal_test::pipeline_golden_counts counts;
};

golden_result golden(const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * expected) {
    golden_result       g;
    const hp1_shape     s = { x->ne[1], w->ne[1], w->ne[0] };
    potal_tile_plan     plan;
    hp1_operands        o;
    ggml_gemmini_args_t args;
    hp1::stats          ws;
    std::vector<float>  want;
    g.ok = potal_tile_select((size_t) s.m, (size_t) s.n, (size_t) s.k, hp1::block_k / POTAL_HW_DIM, &plan) ==
               POTAL_OK &&
           o.allocate(s) &&
           init_args(w, expected, s, "", "golden", ggml::gemmini::residual::ResidualRoute::ws_packet, plan, args) &&
           unpack_weights(args, o, &ws) == hp1::status::ok && quantize_activations(x, args, o) &&
           potal_test::pipeline_golden(args, o, want, &g.counts);
    if (g.ok) {
        std::memcpy(expected->data, want.data(), want.size() * sizeof(float));
        g.stripes = (size_t) ((s.m + (int64_t) (plan.tile_I * POTAL_HW_DIM) - 1) / (int64_t) (plan.tile_I * POTAL_HW_DIM));
    }
    return g;
}

bool same(const ggml_tensor * got, const ggml_tensor * want, const char * name, const char * route) {
    const float * g = static_cast<const float *>(got->data);
    const float * w = static_cast<const float *>(want->data);
    for (int64_t i = 0; i < ggml_nelements(got); ++i) {
        if (std::memcmp(&g[i], &w[i], sizeof(float)) != 0) {
            std::printf("POTAL_PAIRED_MISMATCH case=%s route=%s index=%lld got=%.9g want=%.9g\n", name, route,
                        (long long) i, (double) g[i], (double) w[i]);
            return false;
        }
    }
    return true;
}

struct route_run {
    bool     ok = false;
    double   compute_ms = 0; // steady clock around ggml_backend_graph_compute
    double   wall_ms = 0;    // the backend's pipeline wall time of the node
    uint64_t paired = 0, fallback = 0, fences = 0;
};

enum route_kind { ROUTE_PAIRED, ROUTE_PAIRED_CONSERVATIVE, ROUTE_SERIAL };

const char * route_name(route_kind k) {
    return k == ROUTE_PAIRED ? "paired" : k == ROUTE_PAIRED_CONSERVATIVE ? "paired_conservative" : "serial";
}

void select_route(route_kind k) {
    setenv("GGML_POTAL_PAIRED_AUTO", k == ROUTE_SERIAL ? "OFF" : "ON", 1);
    setenv("GGML_POTAL_PAIRED_FENCES", k == ROUTE_PAIRED_CONSERVATIVE ? "CONSERVATIVE" : "RTL", 1);
}

route_run run_route(ggml_backend_t potal, const char * name, route_kind kind, ggml_cgraph * g, ggml_tensor * y,
                    const ggml_tensor * expected, const golden_result & gold) {
    route_run  r;
    const bool paired = kind != ROUTE_SERIAL;
    select_route(kind);
    std::memset(y->data, 0xff, ggml_nbytes(y)); // NaN pattern: a stale dst never matches
    const ggml_backend_potal_stats b      = stats_of(potal);
    const auto                     t0     = std::chrono::steady_clock::now();
    const ggml_status              status = ggml_backend_graph_compute(potal, g);
    const auto                     t1     = std::chrono::steady_clock::now();
    const ggml_backend_potal_stats a      = stats_of(potal);
    r.compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.wall_ms    = (a.pipeline_wall_s - b.pipeline_wall_s) * 1e3;

    const uint64_t stripes   = a.stripes_semantic_completed - b.stripes_semantic_completed;
    const uint64_t pstripes  = a.paired_stripes - b.paired_stripes;
    const uint64_t fallback  = a.paired_fallback_stripes - b.paired_fallback_stripes;
    const uint64_t main_gemm = a.main_logical_gemms - b.main_logical_gemms;
    const uint64_t res_gemm  = a.residual_logical_gemms - b.residual_logical_gemms;
    const uint64_t p_main    = a.paired_main_loop_ws - b.paired_main_loop_ws;
    const uint64_t p_res     = a.paired_residual_loop_ws - b.paired_residual_loop_ws;
    const uint64_t p_cfg     = a.paired_index_configs - b.paired_index_configs;
    const uint64_t p_wr      = a.paired_index_writes - b.paired_index_writes;
    const uint64_t p_rb      = a.paired_readbacks - b.paired_readbacks;
    const uint64_t p_tiles   = a.paired_readback_tiles - b.paired_readback_tiles;
    const uint64_t p_fence   = a.paired_fences - b.paired_fences;
    const uint64_t p_drain   = a.paired_drains - b.paired_drains;
    const uint64_t merges    = a.merges_completed - b.merges_completed;
    const uint64_t writes    = a.scale_writes - b.scale_writes;
    const uint64_t releases  = a.scale_releases - b.scale_releases;
    const bool     matched   = gold.ok && status == GGML_STATUS_SUCCESS && same(y, expected, name, route_name(kind));
    bool ok = matched && a.hp1_failed == b.hp1_failed && a.stripe_pipeline_failed == b.stripe_pipeline_failed &&
              stripes == gold.stripes && writes == releases && merges == gold.counts.packets;
    if (paired) {
        ok = ok && pstripes + fallback == stripes && main_gemm == fallback && res_gemm <= fallback &&
             p_cfg == p_res && (pstripes == 0 || (p_main > 0 && p_rb > 0 && p_fence >= pstripes));
        paired_total += pstripes;
        indexed_total += p_res;
    } else {
        ok = ok && pstripes == 0 && fallback == 0 && main_gemm == stripes;
    }
    if (kind == ROUTE_PAIRED_CONSERVATIVE) {
        ok = ok && (pstripes == 0 || p_fence > pstripes); // the extra fences really ran
    }
    r.ok       = ok;
    r.paired   = pstripes;
    r.fallback = fallback;
    r.fences   = p_fence;
    std::printf("POTAL_PAIRED_CASE %s case=%s route=%s stripes=%llu golden_stripes=%zu paired_stripes=%llu "
                "fallback_stripes=%llu serial_main_gemms=%llu serial_residual_gemms=%llu paired_main_loop_ws=%llu "
                "paired_residual_loop_ws=%llu index_configs=%llu index_writes=%llu readbacks=%llu readback_tiles=%llu "
                "paired_fences=%llu paired_drains=%llu scale_writes=%llu scale_releases=%llu merges=%llu "
                "golden_packets=%zu compute_ms=%.3f pipeline_wall_ms=%.3f\n",
                ok ? "PASS" : "FAIL", name, route_name(kind), (unsigned long long) stripes, gold.stripes,
                (unsigned long long) pstripes, (unsigned long long) fallback, (unsigned long long) main_gemm,
                (unsigned long long) res_gemm, (unsigned long long) p_main, (unsigned long long) p_res,
                (unsigned long long) p_cfg, (unsigned long long) p_wr, (unsigned long long) p_rb,
                (unsigned long long) p_tiles, (unsigned long long) p_fence, (unsigned long long) p_drain,
                (unsigned long long) writes, (unsigned long long) releases, (unsigned long long) merges,
                gold.counts.packets, r.compute_ms, r.wall_ms);
    failures += ok ? 0 : 1;
    return r;
}

int case_index = 0;

void run_case(ggml_backend_t potal, const char * name, int64_t M, int64_t N, int64_t K, float outlier, int blocks,
              int64_t step, std::mt19937 & rng) {
    ggml_init_params ip = { (size_t) (4 * (N * K + 2 * M * N + M * K) + 64 * 1024 * 1024), nullptr, false };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor *    w        = hp1_weight_tensor(ctx.get(), K, N, rng);
    ggml_tensor *    x        = activations(ctx.get(), K, M, rng, outlier, blocks, step);
    ggml_tensor *    y        = ggml_mul_mat(ctx.get(), w, x);
    ggml_tensor *    expected = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N, M);
    ggml_cgraph *    g        = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(g, y);
    const golden_result gold = golden(w, x, expected);
    // Alternate the order so neither route always runs first (cold weight preparation).
    route_run p, c, s;
    if (case_index++ % 2 == 0) {
        p = run_route(potal, name, ROUTE_PAIRED, g, y, expected, gold);
        c = run_route(potal, name, ROUTE_PAIRED_CONSERVATIVE, g, y, expected, gold);
        s = run_route(potal, name, ROUTE_SERIAL, g, y, expected, gold);
    } else {
        s = run_route(potal, name, ROUTE_SERIAL, g, y, expected, gold);
        c = run_route(potal, name, ROUTE_PAIRED_CONSERVATIVE, g, y, expected, gold);
        p = run_route(potal, name, ROUTE_PAIRED, g, y, expected, gold);
    }
    std::printf("POTAL_PAIRED_PERF case=%s M=%lld N=%lld K=%lld paired_stripes=%llu fallback_stripes=%llu "
                "paired_ms=%.3f paired_conservative_ms=%.3f serial_ms=%.3f paired_fences=%llu "
                "conservative_fences=%llu serial_over_paired=%.4f (single runs)\n",
                name, (long long) M, (long long) N, (long long) K, (unsigned long long) p.paired,
                (unsigned long long) p.fallback, p.compute_ms, c.compute_ms, s.compute_ms,
                (unsigned long long) p.fences, (unsigned long long) c.fences,
                p.compute_ms > 0 ? s.compute_ms / p.compute_ms : 0.0);
}

// Untimed: issuer bring-up (thread, hart-0 pinning, TLB flush) and the first weight
// preparation happen here, outside every measurement.
void warm_up(ggml_backend_t potal, std::mt19937 & rng) {
    ggml_init_params ip = { (size_t) 16 * 1024 * 1024, nullptr, false };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor *    w = hp1_weight_tensor(ctx.get(), 64, 32, rng);
    ggml_tensor *    x = activations(ctx.get(), 64, 16, rng, 60.0f, 1, 4);
    ggml_tensor *    y = ggml_mul_mat(ctx.get(), w, x);
    ggml_cgraph *    g = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(g, y);
    for (route_kind k : { ROUTE_SERIAL, ROUTE_PAIRED }) {
        select_route(k);
        (void) ggml_backend_graph_compute(potal, g);
    }
    std::printf("POTAL_PAIRED_WARMUP done\n");
}

bool select_pipeline_mode() {
    namespace cfg = ggml::gemmini::config;
    using ggml::gemmini::MatmulInvocationMode;
    using ggml::gemmini::RmdBackend;
    if (cfg::DEFAULT_MATMUL_MODE != (int) MatmulInvocationMode::stripe_pipeline) {
        setenv("GEMMINI_MATMUL_MODE", "STRIPE_PIPELINE", 1);
    }
    if (options::compensation && cfg::DEFAULT_RMD_BACKEND != (int) RmdBackend::gemmini_ws_compact) {
        setenv("GEMMINI_RMD_BACKEND", "WS", 1);
    }
    const auto   r   = ggml::gemmini::resolve_matmul_options();
    const char * why = r.ok() ? options::unsupported_combination(r.options.mode, r.options.rmd_backend,
                                                                 r.options.job_capacity)
                              : "matmul options do not resolve (runtime override disabled?)";
    if (r.ok() && r.options.mode != MatmulInvocationMode::stripe_pipeline) {
        why = "GEMMINI_MATMUL_MODE did not select STRIPE_PIPELINE";
    }
    if (why == nullptr && !options::compensation) {
        why = "the paired route needs GGML_GEMMINI_ENABLE_RMD=ON (compensation)";
    }
    std::printf("POTAL_PAIRED_CONFIG activation_quant=%s compensation=%s matmul_mode=%s rmd_backend=%s "
                "job_capacity=%zu paired_auto_build_default=%s params_sha256=%s%s%s\n",
                options::activation_quant_name(), options::compensation_name,
                r.ok() ? options::matmul_mode_name(r.options.mode) : "-",
                r.ok() ? options::rmd_backend_name(r.options.rmd_backend) : "-",
                r.ok() ? r.options.job_capacity : (size_t) 0, options::paired_auto_default ? "ON" : "OFF",
                potal_hw_params_sha256(), why ? " refused=" : "", why ? why : "");
    return why == nullptr;
}

} // namespace

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    log::ensure_outputs();
    if (!select_pipeline_mode()) {
        std::printf("POTAL_PAIRED_SMOKE FAIL configuration\n");
        return 2;
    }
    std::mt19937     rng(20261023);
    ggml_backend_ptr potal(ggml_backend_potal_init());
    if (!potal) {
        std::printf("POTAL_PAIRED_SMOKE FAIL backend init\n");
        return 1;
    }

    warm_up(potal.get(), rng);
    const ggml_backend_potal_stats warm = stats_of(potal.get());

    // Quick: one stripe with a few residual rows (fits ACC1).
    run_case(potal.get(), "small_48x96x256", 48, 96, 256, 60.0f, 1, 4, rng);
    if (!quick) {
        run_case(potal.get(), "decode_1x768x768", 1, 768, 768, 400.0f, 2, 1, rng);
        run_case(potal.get(), "decode_carry_lanes_1x768x768", 1, 768, 768, 30000.0f, 6, 1, rng);
        run_case(potal.get(), "two_stripes_sparse_150x320x128", 150, 320, 128, 60.0f, 1, 8, rng);
        run_case(potal.get(), "k_panels_100x160x2048", 100, 160, 2048, 60.0f, 2, 8, rng);
        run_case(potal.get(), "gpt2_ffn_up_16x3072x768", 16, 3072, 768, 400.0f, 2, 2, rng);
        // Dense large outliers: stripes beyond ACC1 run serially (reported fallback).
        run_case(potal.get(), "dense_fallback_160x320x512", 160, 320, 512, 30000.0f, 6, 1, rng);
        // Repeat: buffers reallocated, sessions reopened, generations continue.
        for (int i = 0; i < 3; ++i) {
            run_case(potal.get(), "repeat_48x96x256", 48, 96, 256, 60.0f, 1, 4, rng);
        }
    }

    const ggml_backend_potal_stats st = stats_of(potal.get());
    std::printf("POTAL_PAIRED_STATS stripe_nodes=%llu failed_nodes=%llu paired_stripes=%llu fallback_stripes=%llu "
                "paired_main_loop_ws=%llu paired_residual_loop_ws=%llu index_configs=%llu index_writes=%llu "
                "readbacks=%llu readback_tiles=%llu paired_fences=%llu paired_drains=%llu scale_writes=%llu "
                "scale_releases=%llu tlb_flushes=%llu last_generation=%u issuer_ready=%d issuer_cpu=%d\n",
                (unsigned long long) st.stripe_pipeline_nodes, (unsigned long long) st.stripe_pipeline_failed,
                (unsigned long long) st.paired_stripes, (unsigned long long) st.paired_fallback_stripes,
                (unsigned long long) st.paired_main_loop_ws, (unsigned long long) st.paired_residual_loop_ws,
                (unsigned long long) st.paired_index_configs, (unsigned long long) st.paired_index_writes,
                (unsigned long long) st.paired_readbacks, (unsigned long long) st.paired_readback_tiles,
                (unsigned long long) st.paired_fences, (unsigned long long) st.paired_drains,
                (unsigned long long) st.scale_writes, (unsigned long long) st.scale_releases,
                (unsigned long long) st.tlb_flushes, st.last_generation, st.issuer_ready ? 1 : 0, st.issuer_cpu);
    const bool exercised = paired_total > 0 && indexed_total > 0;
    if (!exercised) {
        std::printf("POTAL_PAIRED_ROUTE_FAIL no stripe ran paired with an Indexed Residual GEMM\n");
        failures++;
    }
    if (warm.hp1_failed != 0 || warm.stripe_pipeline_failed != 0) {
        std::printf("POTAL_PAIRED_WARMUP_FAIL failed_nodes=%llu\n", (unsigned long long) warm.hp1_failed);
        failures++;
    }
    if (failures != 0 || st.stripe_pipeline_failed != 0 || st.hp1_failed != 0) {
        std::printf("POTAL_PAIRED_SMOKE FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("POTAL_PAIRED_SMOKE PASS\n");
    return 0;
}
