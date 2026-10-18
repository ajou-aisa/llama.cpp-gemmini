// FireSim hardware test of the stripe pipeline (STRIPE_PIPELINE, ExSIA):
// F32 x Q8_HP1 MUL_MAT graphs computed by the POTAL backend. ExSIA hands every
// folding-committed stripe to the hart-0 issuer, which runs its Main GEMM and,
// with residual packets, its Residual GEMM on PoTal (RoCC), then reconstructs
// and merges on its CPU. Every dst is compared element-exactly with a golden
// computed on the CPU from the same ExSIA codes, theta and packets (FULL-style
// quantization of the same input, reference_gemm_i32, the compact-run
// residual golden and the merge formula of support/stripe-golden.hpp). Device
// commands come only from the backend's issuer; only the golden runs here.
//
//   potal-stripe-smoke           quick + full cases
//   potal-stripe-smoke --quick   two stripes with residual packets
//
// Mode: a build with GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE (and
// RMD backend WS) runs as configured; any other build gets
// GEMMINI_MATMUL_MODE=STRIPE_PIPELINE (and GEMMINI_RMD_BACKEND=WS with RMD)
// from this program, which needs GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE.
// Needs GGML_GEMMINI_ACTIVATION_QUANT=EXSIA.
//
// Markers, each printed from the assertions listed:
//   POTAL_STRIPE_CASE PASS|FAIL   per case: compute OK, dst == golden bitwise,
//                                 stripes semantically done == plan's stripes,
//                                 one TLB flush per logical GEMM, every LOOP_WS
//                                 fenced, SCALE_WRITE == RELEASE, packets ==
//                                 golden packets == Residual GEMMs == merges
//   POTAL_RESIDUAL_GEMM_PASS      Residual GEMMs ran on PoTal (residual LOOP_WS
//                                 > 0, all fenced) and their cases matched
//   POTAL_RECONSTRUCT_MERGE_PASS  every packet merged, nonzero corrections ==
//                                 golden's, and their cases matched
//   POTAL_STRIPE_PIPELINE_SMOKE PASS  all cases, >= 2 stripes in each multi-stripe
//                                 case, no failed node; exit 0 only then
// The same results go to the Gemmini debug JSONL (layer potal.smoke), next to
// the backend's [POTAL_STRIPE] stage records.

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

int failures         = 0;
int residual_cases   = 0;
int residual_failed  = 0;
int merge_cases      = 0;
int merge_failed     = 0;

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

// Activations with outliers: every other row one strong value; `blocks`
// outliers per row spread over K blocks (several residual runs), magnitude
// `outlier` (large values need the carry lanes).
ggml_tensor * activations(ggml_context * ctx, int64_t K, int64_t M, std::mt19937 & rng, float outlier, int blocks) {
    ggml_tensor *                   x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float *                         d = static_cast<float *>(x->data);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            d[m * K + k] = nd(rng) * (1.0f + 0.5f * (float) (m % 3));
        }
        if (outlier != 0.0f && m % 2 == 0) {
            for (int b = 0; b < blocks; ++b) {
                const int64_t k = ((m * 7 + b * 37) % K);
                d[m * K + k]    = outlier * ((m + b) % 4 ? -1.0f : 1.0f);
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

// FULL-style ExSIA of the same input (ws_packet route) -> pipeline golden.
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

bool same(const ggml_tensor * got, const ggml_tensor * want, const char * name) {
    const float * g = static_cast<const float *>(got->data);
    const float * w = static_cast<const float *>(want->data);
    for (int64_t i = 0; i < ggml_nelements(got); ++i) {
        if (std::memcmp(&g[i], &w[i], sizeof(float)) != 0) {
            std::printf("POTAL_STRIPE_MISMATCH case=%s index=%lld got=%.9g want=%.9g\n", name, (long long) i,
                        (double) g[i], (double) w[i]);
            return false;
        }
    }
    return true;
}

void run_case(ggml_backend_t potal, const char * name, int64_t M, int64_t N, int64_t K, float outlier, int blocks,
              std::mt19937 & rng, size_t min_stripes) {
    ggml_init_params ip = { (size_t) (4 * (N * K + 2 * M * N + M * K) + 64 * 1024 * 1024), nullptr, false };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor *    w        = hp1_weight_tensor(ctx.get(), K, N, rng);
    ggml_tensor *    x        = activations(ctx.get(), K, M, rng, outlier, blocks);
    ggml_tensor *    y        = ggml_mul_mat(ctx.get(), w, x);
    ggml_tensor *    expected = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N, M);
    ggml_cgraph *    g        = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(g, y);

    const golden_result            gold   = golden(w, x, expected);
    const ggml_backend_potal_stats b      = stats_of(potal);
    const ggml_status              status = ggml_backend_graph_compute(potal, g);
    const ggml_backend_potal_stats a      = stats_of(potal);

    const uint64_t stripes   = a.stripes_semantic_completed - b.stripes_semantic_completed;
    const uint64_t packets   = a.residual_nonempty_packets - b.residual_nonempty_packets;
    const uint64_t res_gemms = a.residual_logical_gemms - b.residual_logical_gemms;
    const uint64_t res_loops = a.residual_loop_ws - b.residual_loop_ws;
    const uint64_t res_fence = a.residual_fences - b.residual_fences;
    const uint64_t main_gemm = a.main_logical_gemms - b.main_logical_gemms;
    const uint64_t main_loop = a.main_loop_ws - b.main_loop_ws;
    const uint64_t main_fnc  = a.main_fences - b.main_fences;
    const uint64_t merges    = a.merges_completed - b.merges_completed;
    const uint64_t nonzero   = a.corrections_nonzero - b.corrections_nonzero;
    const uint64_t tlb       = a.tlb_flushes - b.tlb_flushes;
    const uint64_t writes    = a.scale_writes - b.scale_writes;
    const uint64_t releases  = a.scale_releases - b.scale_releases;
    const bool     matched   = gold.ok && status == GGML_STATUS_SUCCESS && same(y, expected, name);
    const bool     ok = matched && a.hp1_completed == b.hp1_completed + 1 && a.hp1_failed == b.hp1_failed &&
                    a.stripe_pipeline_nodes == b.stripe_pipeline_nodes + 1 &&
                    a.stripe_pipeline_failed == b.stripe_pipeline_failed && stripes == gold.stripes &&
                    stripes >= min_stripes && main_gemm == stripes && main_fnc == main_loop && main_loop > 0 &&
                    res_fence == res_loops && tlb == main_gemm + res_gemms && writes == releases &&
                    packets == gold.counts.packets && res_gemms == packets && merges == packets &&
                    nonzero == gold.counts.corrections && (packets == 0 || res_loops > 0);
    if (packets > 0) {
        residual_cases++;
        residual_failed += matched && res_loops > 0 && res_fence == res_loops && res_gemms == packets ? 0 : 1;
        merge_cases++;
        merge_failed += matched && merges == packets && nonzero == gold.counts.corrections && nonzero > 0 ? 0 : 1;
    }
    char line[640];
    std::snprintf(line, sizeof(line),
                  "POTAL_STRIPE_CASE %s case=%s M=%lld N=%lld K=%lld stripes=%llu golden_stripes=%zu "
                  "max_outstanding=%llu capacity_waits=%llu main_gemms=%llu main_loop_ws=%llu packets=%llu "
                  "golden_packets=%zu residual_gemms=%llu residual_loop_ws=%llu residual_fences=%llu "
                  "residual_compact_k=%llu residual_execution_k=%llu lane_rows=%llu merges=%llu "
                  "corrections_nonzero=%llu golden_nonzero=%zu tlb_flushes=%llu generation=%u wall_ms=%.3f",
                  ok ? "PASS" : "FAIL", name, (long long) M, (long long) N, (long long) K,
                  (unsigned long long) stripes, gold.stripes, (unsigned long long) a.max_outstanding_semantic,
                  (unsigned long long) (a.producer_capacity_waits - b.producer_capacity_waits),
                  (unsigned long long) main_gemm, (unsigned long long) main_loop, (unsigned long long) packets,
                  gold.counts.packets, (unsigned long long) res_gemms, (unsigned long long) res_loops,
                  (unsigned long long) res_fence,
                  (unsigned long long) (a.residual_compact_k - b.residual_compact_k),
                  (unsigned long long) (a.residual_execution_k - b.residual_execution_k),
                  (unsigned long long) (a.residual_active_lane_rows - b.residual_active_lane_rows),
                  (unsigned long long) merges, (unsigned long long) nonzero, gold.counts.corrections,
                  (unsigned long long) tlb, a.last_generation, (a.pipeline_wall_s - b.pipeline_wall_s) * 1e3);
    std::printf("%s\n", line);
    log::debug("potal.smoke", "[POTAL_STRIPE_SMOKE] %s", line);
    failures += ok ? 0 : 1;
}

// Configure the run for STRIPE_PIPELINE unless the build already defaults to it.
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
    const auto r = ggml::gemmini::resolve_matmul_options();
    const char * why = r.ok() ? options::unsupported_combination(r.options.mode, r.options.rmd_backend,
                                                                 r.options.job_capacity)
                              : "matmul options do not resolve (runtime override disabled?)";
    if (r.ok() && r.options.mode != MatmulInvocationMode::stripe_pipeline) {
        why = "GEMMINI_MATMUL_MODE did not select STRIPE_PIPELINE";
    }
    std::printf("POTAL_STRIPE_CONFIG activation_quant=%s compensation=%s matmul_mode=%s rmd_backend=%s "
                "residual_executor=POTAL/%s job_capacity=%zu params_sha256=%s%s%s\n",
                options::activation_quant_name(), options::compensation_name,
                r.ok() ? options::matmul_mode_name(r.options.mode) : "-",
                r.ok() ? options::rmd_backend_name(r.options.rmd_backend) : "-",
                options::residual_executor_name(MatmulInvocationMode::stripe_pipeline),
                r.ok() ? r.options.job_capacity : (size_t) 0, potal_hw_params_sha256(), why ? " refused=" : "",
                why ? why : "");
    return why == nullptr;
}

} // namespace

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    log::ensure_outputs();
    if (!select_pipeline_mode()) {
        std::printf("POTAL_STRIPE_PIPELINE_SMOKE FAIL configuration\n");
        return 2;
    }
    std::mt19937     rng(20261018);
    ggml_backend_ptr potal(ggml_backend_potal_init());
    if (!potal) {
        std::printf("POTAL_STRIPE_PIPELINE_SMOKE FAIL backend init\n");
        return 1;
    }

    // Quick: two stripes (80 + 70 rows), residual packets in both.
    run_case(potal.get(), "two_stripes_150x320x128", 150, 320, 128, 400.0f, 1, rng, 2);
    if (!quick) {
        run_case(potal.get(), "tail_3_stripes_200x320x128", 200, 320, 128, 400.0f, 1, rng, 3);
        run_case(potal.get(), "k_chunks_100x160x2048", 100, 160, 2048, 400.0f, 2, rng, 2);
        run_case(potal.get(), "runs_and_carry_lanes_160x320x512", 160, 320, 512, 30000.0f, 6, rng, 2);
        run_case(potal.get(), "gpt2_ffn_up_128x3072x768", 128, 3072, 768, 400.0f, 2, rng, 2);
        run_case(potal.get(), "decode_shaped_1x768x768", 1, 768, 768, 400.0f, 1, rng, 1);
        run_case(potal.get(), "no_outliers_160x320x128", 160, 320, 128, 0.0f, 0, rng, 2);
        // The same request again: buffers reallocated, slots reused, one TLB flush per GEMM.
        for (int i = 0; i < 3; ++i) {
            run_case(potal.get(), "repeat_150x320x128", 150, 320, 128, 400.0f, 1, rng, 2);
        }
        // Generation wrap: more than POTAL_GENERATION_MAX invocations in this process.
        while (stats_of(potal.get()).tiled_invocations <= 2u * POTAL_GENERATION_MAX && failures == 0) {
            run_case(potal.get(), "wrap_200x320x128", 200, 320, 128, 400.0f, 1, rng, 3);
        }
    }

    const ggml_backend_potal_stats st = stats_of(potal.get());
    std::printf("POTAL_STRIPE_STATS stripe_nodes=%llu failed_nodes=%llu stripes=%llu max_outstanding=%llu "
                "capacity_waits=%llu main_gemms=%llu main_loop_ws=%llu residual_packets=%llu residual_gemms=%llu "
                "residual_loop_ws=%llu merges=%llu corrections_nonzero=%llu invocations=%llu fences=%llu "
                "scale_writes=%llu scale_releases=%llu tlb_flushes=%llu last_generation=%u issuer_ready=%d "
                "issuer_cpu=%d\n",
                (unsigned long long) st.stripe_pipeline_nodes, (unsigned long long) st.stripe_pipeline_failed,
                (unsigned long long) st.stripes_semantic_completed, (unsigned long long) st.max_outstanding_semantic,
                (unsigned long long) st.producer_capacity_waits, (unsigned long long) st.main_logical_gemms,
                (unsigned long long) st.main_loop_ws, (unsigned long long) st.residual_nonempty_packets,
                (unsigned long long) st.residual_logical_gemms, (unsigned long long) st.residual_loop_ws,
                (unsigned long long) st.merges_completed, (unsigned long long) st.corrections_nonzero,
                (unsigned long long) st.tiled_invocations, (unsigned long long) st.tiled_fences,
                (unsigned long long) st.scale_writes, (unsigned long long) st.scale_releases,
                (unsigned long long) st.tlb_flushes, st.last_generation, st.issuer_ready ? 1 : 0, st.issuer_cpu);
    if (!quick) {
        const uint32_t want = (uint32_t) ((st.tiled_invocations - 1) % POTAL_GENERATION_MAX) + 1;
        const bool     wrap = st.tiled_invocations > POTAL_GENERATION_MAX && st.last_generation == want;
        std::printf("POTAL_STRIPE_GENERATION_WRAP %s invocations=%llu last_generation=%u expected=%u\n",
                    wrap ? "PASS" : "FAIL", (unsigned long long) st.tiled_invocations, st.last_generation, want);
        failures += wrap ? 0 : 1;
    }

    // Markers tied to the assertions above.
    const bool residual_ok = options::compensation && residual_cases > 0 && residual_failed == 0 &&
                             st.residual_loop_ws > 0 && st.residual_fences == st.residual_loop_ws;
    const bool merge_ok    = options::compensation && merge_cases > 0 && merge_failed == 0 &&
                          st.merges_completed == st.residual_nonempty_packets && st.corrections_nonzero > 0;
    if (options::compensation) {
        std::printf("POTAL_RESIDUAL_GEMM_%s cases=%d failed=%d residual_loop_ws=%llu\n", residual_ok ? "PASS" : "FAIL",
                    residual_cases, residual_failed, (unsigned long long) st.residual_loop_ws);
        std::printf("POTAL_RECONSTRUCT_MERGE_%s cases=%d failed=%d merges=%llu corrections_nonzero=%llu\n",
                    merge_ok ? "PASS" : "FAIL", merge_cases, merge_failed, (unsigned long long) st.merges_completed,
                    (unsigned long long) st.corrections_nonzero);
        failures += residual_ok && merge_ok ? 0 : 1;
    } else {
        std::printf("POTAL_RESIDUAL_GEMM NOT_APPLICABLE compensation=OFF (Main-only control)\n");
    }
    if (failures != 0 || st.stripe_pipeline_failed != 0 || st.hp1_failed != 0) {
        std::printf("POTAL_STRIPE_PIPELINE_SMOKE FAIL (%d)\n", failures);
        log::debug("potal.smoke", "[POTAL_STRIPE_SMOKE] FAIL failures=%d", failures);
        return 1;
    }
    std::printf("POTAL_STRIPE_PIPELINE_SMOKE PASS\n");
    log::debug("potal.smoke", "[POTAL_STRIPE_SMOKE] PASS");
    return 0;
}
