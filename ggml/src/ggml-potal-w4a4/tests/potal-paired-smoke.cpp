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
#include "ggml-cpu.h"
#include "ggml-potal.h"
#include "ggml-quants.h"
#include "log.hpp"
#include "matmul/dense.hpp"
#include "matmul/options.hpp"
#include "matmul/paired-auto.hpp"
#include "matmul/tiling.hpp"
#include "potal.h"
#include "support/paired-support.hpp"
#include "support/stripe-golden.hpp"

#if defined(POTAL_PLANNING_DIAGNOSTICS)
#include "matmul/planning-profile.hpp"
#endif

#include "ggml-gemmini/matmul/options.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cerrno>
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
    ggml_backend_potal_stats before = {}, after = {};
};

enum route_kind { ROUTE_PAIRED, ROUTE_PAIRED_CONSERVATIVE, ROUTE_SERIAL, ROUTE_KTILE, ROUTE_PREFETCH,
                  ROUTE_COMPLETION, ROUTE_EARLY, ROUTE_STREAM_BOUNDED, ROUTE_STREAM_CONTINUOUS, ROUTE_STREAM_EARLY };

const char * route_name(route_kind k) {
    if (k == ROUTE_KTILE) return "ktile";
    if (k == ROUTE_PREFETCH) return "prefetch";
    if (k == ROUTE_COMPLETION) return "completion";
    if (k == ROUTE_EARLY) return "early_ldb";
    if (k == ROUTE_STREAM_BOUNDED) return "stream_bounded";
    if (k == ROUTE_STREAM_CONTINUOUS) return "stream_continuous";
    if (k == ROUTE_STREAM_EARLY) return "stream_early";
    return k == ROUTE_PAIRED ? "paired" : k == ROUTE_PAIRED_CONSERVATIVE ? "paired_conservative" : "serial";
}

void select_route(route_kind k) {
    setenv("GGML_POTAL_PAIRED_AUTO", k == ROUTE_SERIAL ? "OFF" : "ON", 1);
    setenv("GGML_POTAL_PAIRED_FENCES", k == ROUTE_PAIRED_CONSERVATIVE ? "CONSERVATIVE" : "RTL", 1);
    const bool prefetch = k == ROUTE_PREFETCH || k == ROUTE_COMPLETION || k == ROUTE_EARLY;
    const bool stream = k == ROUTE_STREAM_BOUNDED || k == ROUTE_STREAM_CONTINUOUS || k == ROUTE_STREAM_EARLY;
    setenv("GGML_POTAL_PAIRED_KTILE", k == ROUTE_KTILE || prefetch || stream ? "ON" : "OFF", 1);
    setenv("GGML_POTAL_PAIRED_PREFETCH", prefetch ? "ON" : "OFF", 1);
    setenv("GGML_POTAL_PAIRED_EARLY_PREFETCH", k == ROUTE_EARLY ? "ON" : "OFF", 1);
    setenv("GGML_POTAL_PAIRED_STREAM", k == ROUTE_STREAM_BOUNDED ? "BOUNDED" :
                                     k == ROUTE_STREAM_CONTINUOUS ? "CONTINUOUS" :
                                     k == ROUTE_STREAM_EARLY ? "CONTINUOUS_EARLY" : "OFF", 1);
}

route_run run_route(ggml_backend_t potal, const char * name, route_kind kind, ggml_cgraph * g, ggml_tensor * y,
                    const ggml_tensor * expected, const golden_result & gold, bool print_case = true,
                    bool compare_completion = false) {
    route_run  r;
    const bool paired = kind != ROUTE_SERIAL;
    select_route(kind);
    if (compare_completion)
        setenv("POTAL_H2_COMPLETION_REUSE", kind == ROUTE_COMPLETION || kind == ROUTE_EARLY ? "1" : "0", 1);
    std::memset(y->data, 0xff, ggml_nbytes(y)); // NaN pattern: a stale dst never matches
    const ggml_backend_potal_stats b      = stats_of(potal);
#if defined(POTAL_PLANNING_DIAGNOSTICS)
    planning_profile::reset();
#endif
    const auto                     t0     = std::chrono::steady_clock::now();
    const ggml_status              status = ggml_backend_graph_compute(potal, g);
    const auto                     t1     = std::chrono::steady_clock::now();
    const ggml_backend_potal_stats a      = stats_of(potal);
    r.before = b;
    r.after = a;
    r.compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.wall_ms    = (a.pipeline_wall_s - b.pipeline_wall_s) * 1e3;
#if defined(POTAL_PLANNING_DIAGNOSTICS)
    // Formatting is outside graph-compute and Planning timers.
    planning_profile::dump(stdout, name, route_name(kind));
#endif

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
    if (kind == ROUTE_COMPLETION || kind == ROUTE_EARLY) {
        ok = ok && fallback == 0 && a.paired_prefetch_loads > b.paired_prefetch_loads;
        if (kind == ROUTE_EARLY)
            ok = ok && a.paired_prefetch_ldb_only > b.paired_prefetch_ldb_only &&
                 a.paired_prefetch_mvins == b.paired_prefetch_mvins;
    }
    const uint64_t stream_loops = a.paired_stream_loops - b.paired_stream_loops;
    if (kind == ROUTE_STREAM_BOUNDED || kind == ROUTE_STREAM_CONTINUOUS || kind == ROUTE_STREAM_EARLY)
        ok = ok && fallback == 0 && stream_loops > 0 && stream_loops == p_main + p_res;
    else ok = ok && stream_loops == 0;
    r.ok       = ok;
    r.paired   = pstripes;
    r.fallback = fallback;
    r.fences   = p_fence;
    if (print_case) {
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
    }
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
struct bench_case {
    const char * name;
    int64_t m, n, k;
    float outlier;
    int blocks;
    int64_t step;
};

constexpr bench_case bench_cases[] = {
    { "small_48x96x256", 48, 96, 256, 60.0f, 1, 4 },
    { "decode_1x768x768", 1, 768, 768, 400.0f, 2, 1 },
    { "decode_carry_lanes_1x768x768", 1, 768, 768, 30000.0f, 6, 1 },
    { "two_stripes_sparse_150x320x128", 150, 320, 128, 60.0f, 1, 8 },
    { "k_panels_100x160x2048", 100, 160, 2048, 60.0f, 2, 8 },
    { "gpt2_ffn_up_16x3072x768", 16, 3072, 768, 400.0f, 2, 2 },
    { "dense_fallback_160x320x512", 160, 320, 512, 30000.0f, 6, 1 },
    { "repeat_48x96x256", 48, 96, 256, 60.0f, 1, 4 },
    { "tail_63x129x544", 63, 129, 544, 400.0f, 2, 2 },
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
}

void summary(const char * name, const char * phase, route_kind kind, const std::vector<route_run> & runs) {
    std::vector<double> compute, wall, deviation;
    for (const route_run & r : runs) {
        compute.push_back(r.compute_ms);
        wall.push_back(r.wall_ms);
    }
    const double med = median(compute);
    for (double value : compute) deviation.push_back(std::abs(value - med));
    const auto mm = std::minmax_element(compute.begin(), compute.end());
    const double wall_med = median(wall);
    std::vector<double> wall_deviation;
    for (double value : wall) wall_deviation.push_back(std::abs(value - wall_med));
    const auto wm = std::minmax_element(wall.begin(), wall.end());
    std::printf("POTAL_BENCH_SUMMARY case=%s phase=%s route=%s repeats=%zu compute_median_ms=%.6f "
                "compute_min_ms=%.6f compute_max_ms=%.6f compute_mad_ms=%.6f pipeline_wall_median_ms=%.6f "
                "pipeline_wall_min_ms=%.6f pipeline_wall_max_ms=%.6f pipeline_wall_mad_ms=%.6f\n",
                name, phase, route_name(kind), runs.size(), med, *mm.first, *mm.second,
                median(deviation), wall_med, *wm.first, *wm.second, median(wall_deviation));
}

void print_stats(const char * name, const char * phase, route_kind route, int iteration, const route_run & r) {
    const auto & b = r.before;
    const auto & a = r.after;
#define TIME_FIELD(field) std::printf(" %s_ms=%.6f", #field, (a.field - b.field) * 1e3)
#define COUNT_FIELD(field) std::printf(" %s=%llu", #field, (unsigned long long) (a.field - b.field))
    std::printf("POTAL_BENCH_STATS case=%s phase=%s route=%s iteration=%d", name, phase, route_name(route), iteration);
    TIME_FIELD(scale_s); TIME_FIELD(issue_s); TIME_FIELD(fence_s); TIME_FIELD(act_quant_s);
    TIME_FIELD(weight_pack_s); TIME_FIELD(weight_alloc_s); TIME_FIELD(weight_lookup_s); TIME_FIELD(pipeline_wall_s);
    COUNT_FIELD(producer_capacity_waits); COUNT_FIELD(main_loop_ws); COUNT_FIELD(residual_loop_ws);
    COUNT_FIELD(main_fences); COUNT_FIELD(residual_fences);
    COUNT_FIELD(paired_main_loop_ws); COUNT_FIELD(paired_residual_loop_ws);
    COUNT_FIELD(paired_index_configs); COUNT_FIELD(paired_index_writes);
    COUNT_FIELD(scale_writes); COUNT_FIELD(scale_releases);
    COUNT_FIELD(paired_readbacks); COUNT_FIELD(paired_readback_tiles);
    COUNT_FIELD(paired_fences); COUNT_FIELD(paired_drains); COUNT_FIELD(paired_fallback_stripes);
    COUNT_FIELD(paired_prefetch_loads); COUNT_FIELD(paired_prefetch_mvins);
    COUNT_FIELD(paired_prefetch_ldb_only);
    COUNT_FIELD(paired_prefetch_bytes_declared); COUNT_FIELD(paired_resident_main_loop_ws);
    TIME_FIELD(paired_residual_prefetch_issue_s);
    TIME_FIELD(paired_main_prefetch_issue_s);
    COUNT_FIELD(paired_stream_loops); COUNT_FIELD(paired_stream_pair_waits);
    COUNT_FIELD(paired_stream_local_releases); COUNT_FIELD(paired_stream_batches);
    COUNT_FIELD(paired_stream_capacity_boundaries);
    COUNT_FIELD(paired_residual_staging_bytes); COUNT_FIELD(paired_residual_pitch_padding_bytes);
    COUNT_FIELD(paired_residual_valid_bytes); COUNT_FIELD(paired_residual_load_bytes_declared);
    COUNT_FIELD(stripes_semantic_completed);
    if (r.ok && r.paired && !r.fallback) {
        const uint64_t main_tiles = a.paired_main_loop_ws - b.paired_main_loop_ws;
        const uint64_t residual_tiles = a.paired_residual_loop_ws - b.paired_residual_loop_ws;
        const uint64_t resident_main = a.paired_resident_main_loop_ws - b.paired_resident_main_loop_ws;
        const uint64_t prefetch_tiles = a.paired_prefetch_loads - b.paired_prefetch_loads;
        // Whole LOOP_WS/operand Tiles, not DIM operations or measured DMA bytes.
        std::printf(" gemm_tiles_issued=%llu activation_tiles_command_declared=%llu"
                    " weight_tiles_command_declared=%llu separate_b_prefetch_tiles_command_declared=%llu",
                    (unsigned long long)(main_tiles + residual_tiles),
                    (unsigned long long)(main_tiles + residual_tiles),
                    (unsigned long long)(main_tiles - resident_main + prefetch_tiles),
                    (unsigned long long)prefetch_tiles);
    }
    std::printf("\n");
#undef TIME_FIELD
#undef COUNT_FIELD
}

void bench_one(ggml_backend_t potal, const bench_case & c, int repeat, bool compare_ktile, bool compare_prefetch,
               bool compare_early, bool compare_stream, bool compare_stream_early) {
    std::mt19937 rng(20261023); // one fixed tensor pair for every route and repeat
    ggml_init_params ip = { (size_t) (4 * (c.n * c.k + 2 * c.m * c.n + c.m * c.k) + 64 * 1024 * 1024), nullptr, false };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * w = hp1_weight_tensor(ctx.get(), c.k, c.n, rng);
    ggml_tensor * x = activations(ctx.get(), c.k, c.m, rng, c.outlier, c.blocks, c.step);
    ggml_tensor * expected = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.n, c.m);
    if (compare_stream) {
        uint64_t fingerprint = UINT64_C(14695981039346656037);
        for (const ggml_tensor * tensor : {w, x}) {
            const auto * bytes = static_cast<const uint8_t *>(tensor->data);
            for (size_t i = 0; i < ggml_nbytes(tensor); ++i)
                fingerprint = (fingerprint ^ bytes[i]) * UINT64_C(1099511628211);
        }
        std::printf("POTAL_STREAM_INPUT case=%s M=%lld N=%lld K=%lld seed=20261023 "
                    "fnv1a64=%016llx internal_pair_timers=0 stage_zero_is_unmeasured=1\n",
                    c.name, (long long)c.m, (long long)c.n, (long long)c.k,
                    (unsigned long long)fingerprint);
    }
    const golden_result gold = golden(w, x, expected);
    if (!gold.ok) {
        std::fprintf(stderr, "POTAL_BENCH_FAIL case=%s stage=golden\n", c.name);
        ++failures;
        return;
    }
    // Model weights need a tensor without arena data before backend allocation.
    ggml_init_params wip = { 2 * ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr weights_ctx(ggml_init(wip));
    ggml_tensor * cached_w = ggml_new_tensor_2d(weights_ctx.get(), GGML_TYPE_Q8_HP1, c.k, c.n);
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_ptr weight_buffer(ggml_backend_buft_alloc_buffer(
        buft, ggml_backend_buft_get_alloc_size(buft, cached_w)));
    if (!weight_buffer || ggml_backend_buffer_get_base(weight_buffer.get()) == nullptr ||
        ggml_backend_tensor_alloc(weight_buffer.get(), cached_w, ggml_backend_buffer_get_base(weight_buffer.get())) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "POTAL_BENCH_FAIL case=%s stage=weight_buffer\n", c.name);
        ++failures;
        return;
    }
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(cached_w, w->data, 0, ggml_nbytes(w));
    ggml_tensor * y = ggml_mul_mat(ctx.get(), cached_w, x);
    ggml_cgraph * g = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(g, y);
    std::vector<route_kind> routes = compare_stream ?
        std::vector<route_kind>{ROUTE_PAIRED, ROUTE_KTILE, ROUTE_PREFETCH, ROUTE_EARLY,
                               ROUTE_STREAM_BOUNDED, ROUTE_STREAM_CONTINUOUS} :
        compare_early ?
        std::vector<route_kind>{ROUTE_PAIRED, ROUTE_KTILE, ROUTE_PREFETCH, ROUTE_COMPLETION, ROUTE_EARLY} :
        std::vector<route_kind>{compare_prefetch ? ROUTE_PAIRED : ROUTE_SERIAL,
                               compare_prefetch ? ROUTE_KTILE : ROUTE_PAIRED,
                               compare_prefetch ? ROUTE_PREFETCH :
                               compare_ktile ? ROUTE_KTILE : ROUTE_PAIRED_CONSERVATIVE};
    if (compare_stream_early) routes.push_back(ROUTE_STREAM_EARLY);
    std::array<std::vector<route_run>, 10> cold, steady;
    // Untimed route warmups initialize the issuer and route code; each output is checked.
    for (route_kind route : routes) run_route(potal, c.name, route, g, y, expected, gold, false, compare_early);
    for (int i = 0; i < repeat; ++i) {
        for (size_t j = 0; j < routes.size(); ++j) {
            const route_kind route = routes[(i + (i % 2 ? routes.size() - 1 - j : j)) % routes.size()];
            ggml_backend_potal_clear_weight_cache(potal);
            route_run r = run_route(potal, c.name, route, g, y, expected, gold, false, compare_early);
            cold[(int) route].push_back(r);
            const auto & b = r.before;
            const auto & a = r.after;
            if (a.weight_cache_misses - b.weight_cache_misses != 1) ++failures;
            std::printf("POTAL_BENCH_SAMPLE case=%s phase=cold route=%s iteration=%d compute_ms=%.6f pipeline_wall_ms=%.6f "
                        "cache_hits=%llu cache_misses=%llu weight_prep_ms=%.6f act_quant_ms=%.6f ok=%d\n",
                        c.name, route_name(route), i + 1, r.compute_ms, r.wall_ms,
                        (unsigned long long) (a.weight_cache_hits - b.weight_cache_hits),
                        (unsigned long long) (a.weight_cache_misses - b.weight_cache_misses),
                        (a.weight_pack_s - b.weight_pack_s) * 1e3, (a.act_quant_s - b.act_quant_s) * 1e3, r.ok);
            print_stats(c.name, "cold", route, i + 1, r);
        }
    }
    // All steady samples reuse the same prepared weight. Order rotates and reverses.
    for (route_kind route : routes) run_route(potal, c.name, route, g, y, expected, gold, false, compare_early);
    for (int i = 0; i < repeat; ++i) {
        for (size_t j = 0; j < routes.size(); ++j) {
            const route_kind route = routes[(i + (i % 2 ? routes.size() - 1 - j : j)) % routes.size()];
            route_run r = run_route(potal, c.name, route, g, y, expected, gold, false, compare_early);
            steady[(int) route].push_back(r);
            const auto & b = r.before;
            const auto & a = r.after;
            if (a.weight_cache_hits - b.weight_cache_hits != 1) ++failures;
            std::printf("POTAL_BENCH_SAMPLE case=%s phase=steady route=%s iteration=%d compute_ms=%.6f pipeline_wall_ms=%.6f "
                        "cache_hits=%llu cache_misses=%llu weight_prep_ms=%.6f act_quant_ms=%.6f ok=%d\n",
                        c.name, route_name(route), i + 1, r.compute_ms, r.wall_ms,
                        (unsigned long long) (a.weight_cache_hits - b.weight_cache_hits),
                        (unsigned long long) (a.weight_cache_misses - b.weight_cache_misses),
                        (a.weight_pack_s - b.weight_pack_s) * 1e3, (a.act_quant_s - b.act_quant_s) * 1e3, r.ok);
            print_stats(c.name, "steady", route, i + 1, r);
        }
    }
    for (route_kind route : routes) {
        summary(c.name, "cold", route, cold[(int) route]);
        summary(c.name, "steady", route, steady[(int) route]);
    }
    const route_kind reference = compare_stream ? ROUTE_KTILE :
                                 compare_prefetch || compare_early ? ROUTE_PAIRED : ROUTE_SERIAL;
    const double reference_ms = median([&] { std::vector<double> v; for (const auto & r : steady[reference]) v.push_back(r.compute_ms); return v; }());
    for (route_kind route : routes) {
        if (route == reference) continue;
        std::vector<double> v; for (const auto & r : steady[(int) route]) v.push_back(r.compute_ms);
        if (compare_stream)
            std::printf("POTAL_STREAM_RATIO case=%s route=%s ktile_over_route=%.6f\n",
                        c.name, route_name(route), reference_ms / median(v));
        else if (compare_prefetch || compare_early)
            std::printf("POTAL_PREFETCH_RATIO case=%s route=%s h2_over_route=%.6f\n", c.name, route_name(route), reference_ms / median(v));
        else
            std::printf("POTAL_BENCH_RATIO case=%s route=%s serial_over_route=%.6f\n", c.name, route_name(route), reference_ms / median(v));
    }
    ggml_backend_potal_clear_weight_cache(potal); // buffer will be freed next
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
    auto run_one = [&](route_kind k) {
        select_route(k);
        std::fprintf(stderr, "[POTAL_PROBE] BEGIN route=%s\n", route_name(k));
        std::fflush(stderr);
        const ggml_status status = ggml_backend_graph_compute(potal, g);
        std::fprintf(stderr, "[POTAL_PROBE] END route=%s status=%d\n",
                     route_name(k), (int) status);
        std::fflush(stderr);
        if (status != GGML_STATUS_SUCCESS) ++failures;
    };

    const char * probe = std::getenv("POTAL_PAIRED_PROBE");
    if (probe != nullptr) {
        if (std::strcmp(probe, "serial") == 0)
            run_one(ROUTE_SERIAL);
        else if (std::strcmp(probe, "conservative") == 0)
            run_one(ROUTE_PAIRED_CONSERVATIVE);
        else if (std::strcmp(probe, "rtl") == 0)
            run_one(ROUTE_PAIRED);
        else {
            std::fprintf(stderr, "[POTAL_PROBE] invalid mode: %s\n", probe);
            ++failures;
        }
    } else {
        run_one(ROUTE_SERIAL);
        run_one(ROUTE_PAIRED);
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

bool ktile_raw_golden(bool with_prefetch = false, bool observe = false, bool early = false, bool bench_raw = false,
                       bool with_stream = false, bool with_stream_early = false) {
    if (POTAL_HW_DIM != 32) return false;
    uint32_t observation_generation = 0;
    const auto saturate = [](int64_t x) {
        return int32_t(std::max<int64_t>(INT32_MIN, std::min<int64_t>(INT32_MAX, x)));
    };
    const auto scaled = [&](int64_t x, uint32_t carrier) {
        if (carrier == POTAL_CARRIER_ZERO || x == 0) return int32_t(0);
        if (carrier >= 32) return x < 0 ? INT32_MIN : INT32_MAX;
        return saturate(x * (INT64_C(1) << carrier));
    };
    for (int fixture = bench_raw ? 2 : 0; fixture < (with_stream ? 5 : observe || bench_raw ? 3 : 2); ++fixture) {
        const bool gapped = fixture == 1;
        const bool full_rows = fixture == 2;
        const bool wide_residual = fixture == 3;
        const bool no_residual = fixture == 4;
        potal_test::paired_fixture f(full_rows ? 63 : 17, full_rows ? 129 : 65,
            gapped ? 512 : 544, 20261023,
            [wide_residual](size_t block, size_t column) {
                return int16_t(wide_residual ? (block + column) % 3 : block % 4 == 0 ? 24 : column % 3);
            });
        potal_tile_plan plan;
        if (!f.ok || potal_frozen_plan(f.M, f.N, f.K, full_rows ? 2 : 1,
                                     full_rows ? 4 : 2, 4, 1, &plan) != POTAL_OK) return false;
        residual::decoded_packet packet;
        if (wide_residual) {
            for (uint8_t lane = 0; lane < 4; ++lane)
                for (uint32_t row = 0; row < f.M; ++row) packet.rows.push_back({lane, row});
        } else if (full_rows) {
            for (uint32_t row = 0; row < f.M; ++row) packet.rows.push_back({0, row});
        } else if (!no_residual) {
            packet.rows = {{0, 0}, {1, 16}};
        }
        for (size_t block = 0; !no_residual && block < f.K / 32; block += gapped ? 2 : 1) {
            residual::run run;
            run.block_id = uint32_t(block);
            run.local_k = {1, 7, 29};
            packet.runs.push_back(run);
            packet.k_compact += 3;
        }
        packet.digits.resize(packet.rows.size() * packet.k_compact);
        for (size_t i = 0; i < packet.digits.size(); ++i) {
            const size_t pattern = wide_residual ? i / packet.k_compact + i % packet.k_compact : i;
            packet.digits[i] = int8_t(pattern % (wide_residual ? 5 : 3) == 0 ? -128 : 127);
        }
        const size_t ldc = align_up(f.N, POTAL_HW_DIM);
        std::unique_ptr<int32_t, potal_test::dma_deleter> main_output(
            static_cast<int32_t *>(dma_alloc(align_up(f.M, POTAL_HW_DIM) * ldc * sizeof(int32_t))));
        std::unique_ptr<int32_t, potal_test::dma_deleter> residual_output(
            no_residual ? nullptr :
            static_cast<int32_t *>(dma_alloc(align_up(packet.rows.size(), POTAL_HW_DIM) * ldc * sizeof(int32_t))));
        if (!main_output || (!no_residual && !residual_output)) return false;
        std::vector<int32_t> main_expected(f.M * f.N), residual_expected(packet.rows.size() * f.N);
        // Independent wide arithmetic; never use the production SCU helper,
        // emulator, Scale lowering, or the command plan for expected values.
        for (size_t row = 0; row < f.M; ++row) {
            for (size_t column = 0; column < f.N; ++column) {
                int32_t acc = 0;
                for (size_t block = 0; block < f.K / 32; ++block) {
                    int64_t dot = 0;
                    for (size_t k = block * 32; k < (block + 1) * 32; ++k)
                        dot += int64_t(f.a.get()[row * f.lda + k]) * f.code_at(k, column);
                    acc = saturate(int64_t(acc) + scaled(dot, f.carrier_at(block, column)));
                }
                main_expected[row * f.N + column] = acc;
            }
        }
        for (size_t row = 0; row < packet.rows.size(); ++row) {
            for (size_t column = 0; column < f.N; ++column) {
                int32_t acc = 0;
                size_t compact = 0;
                for (const auto & run : packet.runs) {
                    int64_t dot = 0;
                    for (uint16_t local : run.local_k)
                        dot += int64_t(packet.digits[row * packet.k_compact + compact++]) *
                               f.code_at(size_t(run.block_id) * 32 + local, column);
                    acc = saturate(int64_t(acc) + scaled(dot, f.carrier_at(run.block_id, column)));
                }
                residual_expected[row * f.N + column] = acc;
            }
        }
        std::array<std::vector<double>, 5> raw_samples;
        for (int sample = 0; sample < (bench_raw ? 40 : with_stream ? (with_stream_early ? 7 : 6) :
                                      (with_prefetch ? 3 : 2)); ++sample) {
            const int iteration = sample / 5;
            const int mode = bench_raw ?
                (iteration + (iteration % 2 ? 4 - sample % 5 : sample % 5)) % 5 : sample;
            const bool reuse = mode != 0;
            const bool prefetch = bench_raw ? mode >= 2 : mode == 2 || (with_stream && mode == 3);
            const bool use_early = bench_raw ? mode == 4 : with_stream ? mode == 3 : early && mode == 2;
            if (bench_raw) setenv("POTAL_H2_COMPLETION_REUSE", mode >= 3 ? "1" : "0", 1);
            if (with_stream) {
                // Every destination differs from its golden before every route.
                // A missing readback cannot reuse the previous route's result.
                for (size_t row = 0; row < f.M; ++row)
                    for (size_t col = 0; col < f.N; ++col)
                        main_output.get()[row * ldc + col] = ~main_expected[row * f.N + col];
                for (size_t row = 0; row < packet.rows.size(); ++row)
                    for (size_t col = 0; col < f.N; ++col)
                        residual_output.get()[row * ldc + col] = ~residual_expected[row * f.N + col];
            }
            potal_paired_request request;
            request.rows = f.M; request.n = f.N; request.k = f.K;
            request.main_plan = &plan; request.a = f.a.get(); request.lda = f.lda;
            request.weight = &f.weight; request.residual = no_residual ? nullptr : &packet;
            request.c_main = main_output.get(); request.ldc_main = ldc;
            request.c_residual = residual_output.get(); request.ldc_residual = no_residual ? 0 : ldc;
            request.ktile_scale_reuse = reuse;
            request.next_k_prefetch = prefetch;
            request.early_k_prefetch = use_early;
            if (with_stream && mode >= 4)
                request.stream = mode == 4 ? potal_paired_stream_mode::bounded :
                                 mode == 5 ? potal_paired_stream_mode::continuous :
                                             potal_paired_stream_mode::continuous_early;
            if (observe || bench_raw) request.generation = &observation_generation;
            potal_paired_report report;
            issuer_owner owner;
            std::string why;
            potal_status status = POTAL_OK;
            uint64_t cycles[7] = {};
            potal_status observation = POTAL_OK;
            double raw_ms = 0;
            const bool started = owner.start([&](const owner_runners & runners) {
                if (!observe) {
                    const auto begin = std::chrono::steady_clock::now();
                    runners.paired(request, report);
                    raw_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - begin).count();
                } else {
                    // This callback runs on the existing single issuer's hart.
                    potal_paired_session *session = nullptr;
                    observation = potal_paired_session_open(nullptr, &session);
                    if (observation == POTAL_OK)
                        observation = potal_paired_observation_begin(session, nullptr);
                    if (observation == POTAL_OK) {
                        potal_paired_matmul_auto(request, session, report);
                        if (report.status != potal_paired_status::ok) {
                            std::fprintf(stderr, "POTAL_OBSERVATION_EXECUTION status=%s stage=%s hw=%d detail=%s\n",
                                potal_paired_status_name(report.status), report.stage ? report.stage : "-",
                                int(report.hw_status), report.detail.c_str());
                        }
                        if (!report.fatal && !potal_paired_session_query(session).failed)
                            observation = potal_paired_observation_end(session, cycles);
                    }
                    if (session) {
                        const potal_status closed = potal_paired_h2_query(session).active ?
                            potal_paired_h2_close(&session) : potal_paired_session_close(session);
                        if (closed != POTAL_OK) {
                            report.fatal = true;
                            report.hw_status = closed;
                            report.stage = "observation_close";
                        }
                    }
                }
                return consume_outcome{report.fatal, report.hw_status, report.stage};
            }, why, status);
            owner.join();
            bool exact = started && observation == POTAL_OK && report.status == potal_paired_status::ok;
            for (size_t row = 0; exact && row < f.M; ++row)
                for (size_t column = 0; column < f.N; ++column)
                    exact = exact && main_output.get()[row * ldc + column] == main_expected[row * f.N + column];
            for (size_t row = 0; exact && row < packet.rows.size(); ++row)
                for (size_t column = 0; column < f.N; ++column)
                    exact = exact && residual_output.get()[row * ldc + column] == residual_expected[row * f.N + column];
            const size_t residual_i = align_up(packet.rows.size(), POTAL_HW_DIM) / POTAL_HW_DIM;
            size_t main_gemms = 0, residual_gemms = 0, windows = 0;
            for (const auto & panel : report.tile.panels) windows += panel.windows.size();
            for (const auto & op : report.plan.ops) {
                if (op.kind == potal_paired_op_kind::main_gemm) {
                    ++main_gemms;
                    exact = exact && op.i_tiles == plan.tile_I;
                } else if (op.kind == potal_paired_op_kind::residual_gemm) {
                    ++residual_gemms;
                    exact = exact && op.i_tiles == residual_i;
                }
            }
            exact = exact && main_gemms == report.tile.j_groups * report.tile.k_panels &&
                    residual_gemms == report.tile.j_groups * windows &&
                    (residual_gemms == 0) == no_residual;
            // D2 triggers after the last Residual; Early triggers after Main.
            // With no Residual, only Early can prefetch the next panel.
            exact = exact && report.counts.scale_writes == report.counts.scale_releases &&
                    (report.counts.scale_reused_windows > 0) == (reuse && !gapped && !no_residual) &&
                    report.counts.prefetch_loads ==
                        (request.stream == potal_paired_stream_mode::continuous_early ?
                         report.tile.j_groups * size_t(report.tile.k_panels > 1) :
                         prefetch && (!no_residual || use_early) ?
                         report.tile.j_groups * (report.tile.k_panels - 1) : 0);
            if (use_early || request.stream == potal_paired_stream_mode::continuous_early)
                exact = exact && report.counts.prefetch_ldb_only == report.counts.prefetch_loads &&
                        report.counts.prefetch_mvins == 0 &&
                        report.counts.resident_main_loop_ws == report.counts.prefetch_loads;
            std::printf("POTAL_KTILE_RAW_GOLDEN %s gapped=%d reuse=%d prefetch=%d main_i32=%zu residual_i32=%zu "
                        "reused_windows=%llu scale_writes=%llu scale_releases=%llu prefetch_loads=%llu resident_main=%llu "
                        "early=%d ldb_only=%llu stream=%u fixture=%d main_I=%zu residual_I=%zu "
                        "main_gemms=%zu residual_gemms=%zu\n",
                        exact ? "PASS" : "FAIL", int(gapped), int(reuse), int(prefetch), main_expected.size(), residual_expected.size(),
                        (unsigned long long)report.counts.scale_reused_windows,
                        (unsigned long long)report.counts.scale_writes, (unsigned long long)report.counts.scale_releases,
                        (unsigned long long)report.counts.prefetch_loads,
                        (unsigned long long)report.counts.resident_main_loop_ws, int(use_early),
                        (unsigned long long)report.counts.prefetch_ldb_only, unsigned(request.stream),
                        fixture, plan.tile_I, residual_i, main_gemms, residual_gemms);
            if (observe) {
                std::printf("POTAL_PREFETCH_OBSERVATION mode=%d gapped=%d status=%d fixture=%d rows=%zu columns=%zu "
                    "b0_write_cycles=%llu b1_write_cycles=%llu mac0_cycles=%llu mac1_cycles=%llu "
                    "mac0_b1_cycles=%llu mac1_b0_cycles=%llu cross_pair_cycles=%llu\n",
                    mode, int(gapped), int(observation), fixture, f.M, f.N,
                    (unsigned long long)cycles[0], (unsigned long long)cycles[1],
                    (unsigned long long)cycles[2], (unsigned long long)cycles[3],
                    (unsigned long long)cycles[4], (unsigned long long)cycles[5],
                    (unsigned long long)cycles[6]);
            }
            if (report.fatal) {
                f.a.release(); f.weight.abandon(); main_output.release(); residual_output.release();
            }
            if (!exact) return false;
            if (bench_raw && iteration > 0) {
                raw_samples[mode].push_back(raw_ms);
                std::printf("POTAL_RAW_BENCH_SAMPLE route=%d iteration=%d ms=%.6f exact=1 M=63 N=129 K=544\n",
                    mode, iteration, raw_ms);
            }
        }
        if (bench_raw) {
            for (int mode = 0; mode < 5; ++mode) {
                const double med = median(raw_samples[mode]);
                std::vector<double> dev;
                for (double v : raw_samples[mode]) dev.push_back(std::abs(v - med));
                std::printf("POTAL_RAW_BENCH_SUMMARY route=%d repeats=%zu median_ms=%.6f mad_ms=%.6f\n",
                    mode, raw_samples[mode].size(), med, median(dev));
            }
        }
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--stream-early-golden") == 0)
        return ktile_raw_golden(true, false, false, false, true, true) ? 0 : 1;
    if (argc == 2 && std::strcmp(argv[1], "--stream-golden") == 0)
        return ktile_raw_golden(true, false, false, false, true) ? 0 : 1;
    if (argc == 2 && std::strcmp(argv[1], "--bench-raw-early") == 0) {
        if (std::getenv("POTAL_EARLY_TRACE") || std::getenv("POTAL_RAW_TRACE") ||
            std::getenv("POTAL_PAIRED_PROFILE")) return 2;
        return ktile_raw_golden(true, false, true, true) ? 0 : 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--ktile-golden") == 0)
        return ktile_raw_golden() ? 0 : 1;
    if (argc == 2 && std::strcmp(argv[1], "--prefetch-golden") == 0)
        return ktile_raw_golden(true) ? 0 : 1;
    if (argc == 2 && std::strcmp(argv[1], "--prefetch-observe") == 0)
        return ktile_raw_golden(true, true) ? 0 : 1;
    if (argc == 2 && std::strcmp(argv[1], "--early-prefetch-observe") == 0)
        return ktile_raw_golden(true, true, true) ? 0 : 1;
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    const bool compare_ktile = argc > 1 && std::strcmp(argv[1], "--bench-ktile") == 0;
    const bool compare_prefetch = argc > 1 && std::strcmp(argv[1], "--bench-prefetch") == 0;
    const bool observe_early = argc > 1 && std::strcmp(argv[1], "--observe-early") == 0;
    const bool compare_early = observe_early || (argc > 1 && std::strcmp(argv[1], "--bench-early") == 0);
    const bool observe_stream_early = argc > 1 && std::strcmp(argv[1], "--observe-stream-early") == 0;
    const bool compare_stream_early = observe_stream_early ||
        (argc > 1 && std::strcmp(argv[1], "--bench-stream-early") == 0);
    const bool observe_stream = observe_stream_early || (argc > 1 && std::strcmp(argv[1], "--observe-stream") == 0);
    const bool compare_stream = compare_stream_early || observe_stream ||
        (argc > 1 && std::strcmp(argv[1], "--bench-stream") == 0);
    const bool bench = compare_ktile || compare_prefetch || compare_early || compare_stream ||
                       (argc > 1 && std::strcmp(argv[1], "--bench") == 0);
    const char * case_name = nullptr;
    int repeat = 7;
    if (bench) {
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 >= argc) { std::fprintf(stderr, "usage: %s --bench --case NAME [--repeat N]\n", argv[0]); return 2; }
            if (std::strcmp(argv[i], "--case") == 0) case_name = argv[i + 1];
            else if (std::strcmp(argv[i], "--repeat") == 0) {
                char * end = nullptr;
                errno = 0;
                const long value = std::strtol(argv[i + 1], &end, 10);
                if (errno || end == argv[i + 1] || *end || value < 1 || value > 1000) {
                    std::fprintf(stderr, "repeat must be 1..1000\n"); return 2;
                }
                repeat = (int) value;
            } else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
        }
        if (case_name == nullptr) { std::fprintf(stderr, "--bench requires --case NAME\n"); return 2; }
    }
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

    if (bench) {
        // Diagnostic modes inject fences, readbacks, or tracing into measured work.
        if (std::getenv("POTAL_PAIRED_PROBE") || std::getenv("POTAL_PAIRED_WINDOW_DIAG") ||
            std::getenv("POTAL_PAIRED_DIAG_INDEX_ENTRIES") || std::getenv("POTAL_PAIRED_STAGE_DIAG") ||
            std::getenv("POTAL_RAW_TRACE") || (!observe_early && std::getenv("POTAL_EARLY_TRACE"))) {
            std::fprintf(stderr, "POTAL_BENCH_FAIL diagnostics enabled\n"); return 2;
        }
        if (compare_stream) {
#if LOG_DEBUG || LOG_CYCLE || CYCLE_SIM || CYCLE_DETAIL
            if (!observe_stream) {
                std::fprintf(stderr, "POTAL_BENCH_FAIL primary measurement requires a logging-disabled build\n");
                return 2;
            }
#endif
            if (std::getenv("POTAL_PAIRED_PROFILE") ||
                (!observe_stream && (std::getenv("POTAL_STREAM_MANIFEST") || std::getenv("POTAL_STREAM_COUNTERS") ||
                                     std::getenv("POTAL_STREAM_ADMISSION_PROFILE")))) {
                std::fprintf(stderr, "POTAL_BENCH_FAIL internal profile enabled\n"); return 2;
            }
            setenv("POTAL_PAIRED_TIMING", "0", 1);
            setenv("POTAL_H2_COMPLETION_REUSE", "1", 1);
            if (observe_stream) {
                setenv("POTAL_STREAM_MANIFEST", "1", 1);
                setenv("POTAL_STREAM_COUNTERS", "1", 1);
                std::printf("POTAL_STREAM_DIAGNOSTIC case=%s timing_is_not_performance=1\n", case_name);
            }
        }
        if (observe_early) {
            if (!std::getenv("POTAL_EARLY_TRACE")) setenv("POTAL_EARLY_TRACE", "1", 1);
            std::printf("POTAL_EARLY_DIAGNOSTIC case=%s timing_is_not_performance=1\n", case_name);
        }
        for (const bench_case & c : bench_cases) {
            if (std::strcmp(c.name, case_name) == 0) {
                bench_one(potal.get(), c, repeat, compare_ktile, compare_prefetch, compare_early, compare_stream,
                          compare_stream_early);
                std::printf("POTAL_BENCH %s case=%s failures=%d\n", failures ? "FAIL" : "PASS", case_name, failures);
                return failures ? 1 : 0;
            }
        }
        std::fprintf(stderr, "unknown case: %s\n", case_name);
        return 2;
    }
    warm_up(potal.get(), rng);
    const ggml_backend_potal_stats warm = stats_of(potal.get());

    if (std::getenv("POTAL_PAIRED_PROBE") != nullptr) {
        const bool ok = failures == 0 &&
                        warm.hp1_failed == 0 &&
                        warm.stripe_pipeline_failed == 0;
        std::fprintf(stderr,
                     "[POTAL_PROBE] %s hp1_failed=%llu pipeline_failed=%llu\n",
                     ok ? "PASS" : "FAIL",
                     (unsigned long long) warm.hp1_failed,
                     (unsigned long long) warm.stripe_pipeline_failed);
        return ok ? 0 : 1;
    }

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
