// H3-8I-25P pipeline bench: one F32 x Q8_HP1 MUL_MAT (Linear) through the POTAL backend, repeated
// under several in-process modes. Every run is compared bitwise with the CPU golden built from the
// same ExSIA codes (support/stripe-golden.hpp, as in potal-stripe-smoke), and with the first mode's
// output. Wall time is CLOCK_MONOTONIC around ggml_backend_graph_compute only.
//
//   potal-pipeline-bench --m=M --n=N --k=K [--outlier=F] [--blocks=B] [--repeat=R] [--seed=S]
//                        [--modes=serial,roles,queue,roles+queue,strict,strict+queue] [--warmup=W] [--stop-on-fail]
// strictopt / strictopt+queue: strict with the H3-8I-25T H3 optimizations (fast decode, plan validation reuse).
// strict / strict+queue: ROLES with GGML_POTAL_STRICT_ISSUER=ON; a run also fails when any paired
// preparation ran on the issuer thread or CPU (BENCH_ROLE prints the per-run counters).
// A mode may carry a paired scheduling variant: <mode>@{ktile,prefetch,early,bounded,cont,contearly}.
// Modes set GGML_POTAL_HART_ROLES and GGML_POTAL_INDEX_QUEUE per run (read per call by the backend).
// Queue modes need POTAL_INDEX_QUEUE_FRESH/BUDGET (and USED for a later process in the same FPGA lifetime).
// Lines: BENCH_CONFIG, BENCH (one per run), BENCH_SUMMARY (per mode), BENCH_VERDICT PASS|FAIL.

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
#include "trace/hw-window.hpp"
#include "trace/timeline.hpp"

#include "ggml-gemmini/matmul/options.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <vector>

using namespace ggml::potal;
namespace hp1 = ggml::potal::quants::weight::hp1;

namespace {

int64_t mono_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

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

ggml_tensor * activations(ggml_context * ctx, int64_t K, int64_t M, std::mt19937 & rng, float outlier, int blocks) {
    ggml_tensor *                   x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float *                         d = static_cast<float *>(x->data);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) d[m * K + k] = nd(rng) * (1.0f + 0.5f * (float) (m % 3));
        if (outlier != 0.0f && m % 2 == 0) {
            for (int b = 0; b < blocks; ++b) {
                const int64_t k = ((m * 7 + b * 37) % K);
                d[m * K + k]    = outlier * ((m + b) % 4 ? -1.0f : 1.0f);
            }
        }
    }
    return x;
}

bool golden(const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * expected, size_t & stripes) {
    const hp1_shape     s = { x->ne[1], w->ne[1], w->ne[0] };
    potal_tile_plan     plan;
    hp1_operands        o;
    ggml_gemmini_args_t args;
    hp1::stats          ws;
    std::vector<float>  want;
    potal_test::pipeline_golden_counts counts;
    const bool ok = potal_tile_select((size_t) s.m, (size_t) s.n, (size_t) s.k, hp1::block_k / POTAL_HW_DIM, &plan) ==
                        POTAL_OK &&
                    o.allocate(s) &&
                    init_args(w, expected, s, "", "golden", ggml::gemmini::residual::ResidualRoute::ws_packet, plan,
                              args) &&
                    unpack_weights(args, o, &ws) == hp1::status::ok && quantize_activations(x, args, o) &&
                    potal_test::pipeline_golden(args, o, want, &counts);
    if (ok) {
        std::memcpy(expected->data, want.data(), want.size() * sizeof(float));
        stripes = (size_t) ((s.m + (int64_t) (plan.tile_I * POTAL_HW_DIM) - 1) / (int64_t) (plan.tile_I * POTAL_HW_DIM));
    }
    return ok;
}

uint64_t fnv64(const ggml_tensor * t) {
    const unsigned char * p = static_cast<const unsigned char *>(t->data);
    uint64_t              h = 1469598103934665603ull;
    for (size_t i = 0; i < ggml_nbytes(t); ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

struct mode {
    std::string name;
    bool        roles = false;
    bool        queue = false;
    bool        strict = false; // GGML_POTAL_STRICT_ISSUER (H3-8I-25S): roles with an issuer-only hart 0
    bool        opt = false;    // H3-8I-25T H3 optimizations: GGML_POTAL_FAST_DECODE + GGML_POTAL_PLAN_VALIDATED_REUSE
    std::string variant;  // paired scheduling: "" (default route), ktile, prefetch, early, bounded, cont, contearly
};

// Paired scheduling env for one mode (read per node by the backend). Every variant
// keeps the CPU golden; refused combinations are never built here.
void apply_variant(const std::string & v) {
    const bool ktile = !v.empty();
    setenv("GGML_POTAL_PAIRED_KTILE", ktile ? "ON" : "OFF", 1);
    setenv("GGML_POTAL_PAIRED_PREFETCH", v == "prefetch" || v == "early" ? "ON" : "OFF", 1);
    setenv("GGML_POTAL_PAIRED_EARLY_PREFETCH", v == "early" ? "ON" : "OFF", 1);
    setenv("GGML_POTAL_PAIRED_STREAM", v == "bounded" ? "BOUNDED" : v == "cont" ? "CONTINUOUS" :
                                        v == "contearly" ? "CONTINUOUS_EARLY" : "OFF", 1);
}

// CLOCK_MONOTONIC versus CLINT mtime (rdtime) on this boot: mono = 1000 * mtime + D for a
// 1 MHz clocksource. When two MONOTONIC reads around one rdtime are equal, all three come
// from the same mtime tick and D is exact. Test-only epoch marker (H3-8I-25R).
void epoch_marker(const char * tag) {
#if defined(__riscv)
    for (int tries = 1; tries <= 4000; ++tries) {
        const int64_t a = mono_ns();
        uint64_t      m;
        asm volatile("rdtime %0" : "=r"(m));
        const int64_t b = mono_ns();
        if (a == b || tries == 4000) {
            std::printf("BENCH_EPOCH tag=%s exact=%d tries=%d mono_a_ns=%" PRId64 " mono_b_ns=%" PRId64
                        " mtime=%" PRIu64 " d_ns=%" PRId64 "\n",
                        tag, a == b ? 1 : 0, tries, a, b, m, a - (int64_t) (m * 1000));
            return;
        }
    }
#else
    std::printf("BENCH_EPOCH tag=%s unsupported=1\n", tag);
#endif
}

bool parse_modes(const std::string & list, std::vector<mode> & out) {
    size_t pos = 0;
    while (pos <= list.size()) {
        const size_t end  = list.find(',', pos);
        const std::string item = list.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        mode m;
        m.name = item;
        const size_t at = item.find('@');
        const std::string base = item.substr(0, at);
        if (at != std::string::npos) {
            m.variant = item.substr(at + 1);
            if (m.variant != "ktile" && m.variant != "prefetch" && m.variant != "early" && m.variant != "bounded" &&
                m.variant != "cont" && m.variant != "contearly")
                return false;
        }
        if (base == "serial") {
        } else if (base == "roles") {
            m.roles = true;
        } else if (base == "queue") {
            m.queue = true;
        } else if (base == "roles+queue") {
            m.roles = m.queue = true;
        } else if (base == "strict") {
            m.roles = m.strict = true;
        } else if (base == "strict+queue") {
            m.roles = m.strict = m.queue = true;
        } else if (base == "strictopt") {
            m.roles = m.strict = m.opt = true;
        } else if (base == "strictopt+queue") {
            m.roles = m.strict = m.opt = m.queue = true;
        } else {
            return false;
        }
        out.push_back(m);
        if (end == std::string::npos) break;
        pos = end + 1;
    }
    return !out.empty();
}

bool select_pipeline_mode() {
    namespace cfg = ggml::gemmini::config;
    using ggml::gemmini::MatmulInvocationMode;
    using ggml::gemmini::RmdBackend;
    if (cfg::DEFAULT_MATMUL_MODE != (int) MatmulInvocationMode::stripe_pipeline)
        setenv("GEMMINI_MATMUL_MODE", "STRIPE_PIPELINE", 1);
    if (options::compensation && cfg::DEFAULT_RMD_BACKEND != (int) RmdBackend::gemmini_ws_compact)
        setenv("GEMMINI_RMD_BACKEND", "WS", 1);
    const auto   r   = ggml::gemmini::resolve_matmul_options();
    const char * why = r.ok() ? options::unsupported_combination(r.options.mode, r.options.rmd_backend,
                                                                 r.options.job_capacity)
                              : "matmul options do not resolve";
    if (r.ok() && r.options.mode != MatmulInvocationMode::stripe_pipeline) why = "not STRIPE_PIPELINE";
    std::printf("BENCH_CONFIG matmul_mode=%s rmd_backend=%s job_capacity=%zu params_sha256=%s timeline=%d%s%s\n",
                r.ok() ? options::matmul_mode_name(r.options.mode) : "-",
                r.ok() ? options::rmd_backend_name(r.options.rmd_backend) : "-",
                r.ok() ? r.options.job_capacity : (size_t) 0, potal_hw_params_sha256(),
                ::potal::timeline::enabled() ? 1 : 0, why ? " refused=" : "", why ? why : "");
    return why == nullptr;
}

} // namespace

int main(int argc, char ** argv) {
    int64_t     M = 128, N = 768, K = 768, repeat = 3, warmup = 0;
    bool        stop_on_fail = false;
    bool        hw_selftest = false;
    float       outlier = 400.0f;
    int         blocks  = 2;
    unsigned    seed    = 20261104;
    std::string modes_arg = "serial,roles";
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (!std::strncmp(a, "--m=", 4)) M = std::atoll(a + 4);
        else if (!std::strncmp(a, "--n=", 4)) N = std::atoll(a + 4);
        else if (!std::strncmp(a, "--k=", 4)) K = std::atoll(a + 4);
        else if (!std::strncmp(a, "--repeat=", 9)) repeat = std::atoll(a + 9);
        else if (!std::strncmp(a, "--warmup=", 9)) warmup = std::atoll(a + 9);
        else if (!std::strncmp(a, "--outlier=", 10)) outlier = (float) std::atof(a + 10);
        else if (!std::strncmp(a, "--blocks=", 9)) blocks = std::atoi(a + 9);
        else if (!std::strncmp(a, "--seed=", 7)) seed = (unsigned) std::strtoul(a + 7, nullptr, 10);
        else if (!std::strncmp(a, "--modes=", 8)) modes_arg = a + 8;
        else if (!std::strcmp(a, "--stop-on-fail")) stop_on_fail = true;
        else if (!std::strcmp(a, "--hw-selftest")) hw_selftest = true;
        else {
            std::printf("BENCH_VERDICT FAIL usage: unknown argument %s\n", a);
            return 2;
        }
    }
    std::vector<mode> modes;
    if (M <= 0 || N <= 0 || K <= 0 || repeat <= 0 || warmup < 0 || !parse_modes(modes_arg, modes)) {
        std::printf("BENCH_VERDICT FAIL usage: bad shape/repeat/modes\n");
        return 2;
    }
    setvbuf(stdout, nullptr, _IOLBF, 0);
    log::ensure_outputs();
    if (!select_pipeline_mode()) {
        std::printf("BENCH_VERDICT FAIL configuration\n");
        return 2;
    }
    ggml_backend_ptr potal(ggml_backend_potal_init());
    if (!potal) {
        std::printf("BENCH_VERDICT FAIL backend init\n");
        return 1;
    }

    std::mt19937     rng(seed);
    ggml_init_params ip = { (size_t) (4 * (N * K + 2 * M * N + M * K) + 64 * 1024 * 1024), nullptr, false };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor *    w        = hp1_weight_tensor(ctx.get(), K, N, rng);
    ggml_tensor *    x        = activations(ctx.get(), K, M, rng, outlier, blocks);
    ggml_tensor *    y        = ggml_mul_mat(ctx.get(), w, x);
    ggml_tensor *    expected = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N, M);
    ggml_cgraph *    g        = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(g, y);
    size_t gold_stripes = 0;
    if (!golden(w, x, expected, gold_stripes)) {
        std::printf("BENCH_VERDICT FAIL golden\n");
        return 1;
    }
    const uint64_t gold_hash = fnv64(expected);
    std::printf("BENCH_SHAPE M=%" PRId64 " N=%" PRId64 " K=%" PRId64 " outlier=%g blocks=%d seed=%u golden_stripes=%zu "
                "golden_hash=%016" PRIx64 "\n",
                M, N, K, (double) outlier, blocks, seed, gold_stripes, gold_hash);

    int failures = 0;
    std::vector<std::vector<int64_t>> walls(modes.size());
    // Warm-up reps (rep < 0) run every mode once per rep, are golden-checked like the
    // others, and are excluded from BENCH_SUMMARY (first-use allocation, trace rings).
    epoch_marker("begin");
    bool stopped = false;
    for (int64_t rep = -warmup; rep < repeat && !stopped; ++rep) {
        for (size_t mi = 0; mi < modes.size() && !stopped; ++mi) {
            const mode & md = modes[mi];
            apply_variant(md.variant);
            setenv("GGML_POTAL_HART_ROLES", md.roles ? "ON" : "OFF", 1);
            setenv("GGML_POTAL_INDEX_QUEUE", md.queue ? "ON" : "OFF", 1);
            setenv("GGML_POTAL_STRICT_ISSUER", md.strict ? "ON" : "OFF", 1);
            setenv("GGML_POTAL_FAST_DECODE", md.opt ? "ON" : "OFF", 1);
            setenv("GGML_POTAL_PLAN_VALIDATED_REUSE", md.opt ? "ON" : "OFF", 1);
            std::memset(y->data, 0xff, ggml_nbytes(y));
            const uint32_t                 q0     = potal_index_queue_remaining();
            const ggml_backend_potal_stats b      = stats_of(potal.get());
            const int64_t                  t0     = mono_ns();
            const ggml_status              status = ggml_backend_graph_compute(potal.get(), g);
            const int64_t                  t1     = mono_ns();
            const ggml_backend_potal_stats a      = stats_of(potal.get());
            const uint32_t                 q1     = potal_index_queue_remaining();
            const uint64_t                 h      = fnv64(y);
            const uint64_t stripes = a.stripes_semantic_completed - b.stripes_semantic_completed;
            const uint64_t on_issuer = (a.prepares_on_issuer_thread - b.prepares_on_issuer_thread) +
                                       (a.prepares_on_issuer_cpu - b.prepares_on_issuer_cpu);
            const bool ok = status == GGML_STATUS_SUCCESS && h == gold_hash && (!md.strict || on_issuer == 0) &&
                            std::memcmp(y->data, expected->data, ggml_nbytes(y)) == 0 && stripes == gold_stripes &&
                            a.stripe_pipeline_failed == b.stripe_pipeline_failed &&
                            (!md.queue || q1 < q0);
            failures += ok ? 0 : 1;
            if (rep >= 0) walls[mi].push_back(t1 - t0);
            std::printf("%s mode=%s rep=%" PRId64 " ok=%d status=%d wall_ns=%" PRId64 " stripes=%" PRIu64
                        " main_gemms=%" PRIu64 " main_loop_ws=%" PRIu64 " residual_gemms=%" PRIu64
                        " residual_loop_ws=%" PRIu64 " merges=%" PRIu64 " queue_used=%u queue_remaining=%u"
                        " hash=%016" PRIx64 "\n",
                        rep < 0 ? "BENCH_WARMUP" : "BENCH", md.name.c_str(), rep, ok ? 1 : 0, (int) status, t1 - t0, stripes,
                        a.main_logical_gemms - b.main_logical_gemms, a.main_loop_ws - b.main_loop_ws,
                        a.residual_logical_gemms - b.residual_logical_gemms, a.residual_loop_ws - b.residual_loop_ws,
                        a.merges_completed - b.merges_completed, q0 - q1, q1, h);
            std::printf("BENCH_SCHED mode=%s rep=%" PRId64 " variant=%s prefetch_loads=%" PRIu64 " prefetch_ldb_only=%" PRIu64
                        " stream_loops=%" PRIu64 " stream_pair_waits=%" PRIu64 " stream_batches=%" PRIu64 "\n",
                        md.name.c_str(), rep, md.variant.empty() ? "default" : md.variant.c_str(),
                        a.paired_prefetch_loads - b.paired_prefetch_loads,
                        a.paired_prefetch_ldb_only - b.paired_prefetch_ldb_only,
                        a.paired_stream_loops - b.paired_stream_loops,
                        a.paired_stream_pair_waits - b.paired_stream_pair_waits,
                        a.paired_stream_batches - b.paired_stream_batches);
            std::printf("BENCH_ROLE mode=%s rep=%" PRId64 " strict=%d prepares_on_issuer_thread=%" PRIu64
                        " prepares_on_issuer_cpu=%" PRIu64 " prepares_on_producer=%" PRIu64 " paired_stripes=%" PRIu64
                        " paired_main_loop_ws=%" PRIu64 " paired_residual_loop_ws=%" PRIu64 " index_configs=%" PRIu64
                        " index_writes=%" PRIu64 " scale_writes=%" PRIu64 " scale_releases=%" PRIu64 " readbacks=%" PRIu64
                        " fences=%" PRIu64 " drains=%" PRIu64 "\n",
                        md.name.c_str(), rep, md.strict ? 1 : 0, a.prepares_on_issuer_thread - b.prepares_on_issuer_thread,
                        a.prepares_on_issuer_cpu - b.prepares_on_issuer_cpu,
                        a.prepares_on_producer - b.prepares_on_producer, a.paired_stripes - b.paired_stripes,
                        a.paired_main_loop_ws - b.paired_main_loop_ws, a.paired_residual_loop_ws - b.paired_residual_loop_ws,
                        a.paired_index_configs - b.paired_index_configs, a.paired_index_writes - b.paired_index_writes,
                        a.scale_writes - b.scale_writes, a.scale_releases - b.scale_releases,
                        a.paired_readbacks - b.paired_readbacks, a.paired_fences - b.paired_fences,
                        a.paired_drains - b.paired_drains);
            if (!ok && stop_on_fail) {
                std::printf("BENCH_STOP mode=%s rep=%" PRId64 " reason=run_failed\n", md.name.c_str(), rep);
                stopped = true;
            }
        }
    }
    epoch_marker("end");
    for (size_t mi = 0; mi < modes.size(); ++mi) {
        std::vector<int64_t> v = walls[mi];
        if (v.empty()) continue;  // stopped before this mode was measured
        std::sort(v.begin(), v.end());
        const size_t n   = v.size();
        const double med = n % 2 ? (double) v[n / 2] : 0.5 * ((double) v[n / 2 - 1] + (double) v[n / 2]);
        double       sum = 0, sq = 0;
        for (int64_t x : v) sum += (double) x;
        const double mean = sum / (double) n;
        for (int64_t x : v) sq += ((double) x - mean) * ((double) x - mean);
        std::vector<double> dev;
        for (int64_t x : v) dev.push_back(std::fabs((double) x - med));
        std::sort(dev.begin(), dev.end());
        const double mad = n % 2 ? dev[n / 2] : 0.5 * (dev[n / 2 - 1] + dev[n / 2]);
        // wall_ns_median keeps the I25P definition (upper middle element); median_exact averages for even n.
        std::printf("BENCH_SUMMARY mode=%s runs=%zu wall_ns_min=%" PRId64 " wall_ns_median=%" PRId64
                    " wall_ns_max=%" PRId64 " wall_ns_median_exact=%.0f wall_ns_mean=%.0f wall_ns_sd=%.0f"
                    " wall_ns_mad=%.0f\n",
                    modes[mi].name.c_str(), n, v.front(), v[n / 2], v.back(), med, mean,
                    n > 1 ? std::sqrt(sq / (double) (n - 1)) : 0.0, mad);
    }
    if (::potal::timeline::enabled() && !::potal::timeline::flush()) {
        std::printf("BENCH_TIMELINE_FLUSH FAIL\n");
        failures++;
    }
    if (::potal::timeline::enabled()) {
        const ::potal::timeline::stats_t ts = ::potal::timeline::stats();
        std::printf("BENCH_TIMELINE profile=%s threads=%u events=%" PRIu64 " dropped=%" PRIu64
                    " unregistered=%" PRIu64 " overflowed_threads=%u\n",
                    std::getenv("POTAL_TIMELINE_PROFILE") ? std::getenv("POTAL_TIMELINE_PROFILE") : "window",
                    ts.threads, ts.events, ts.dropped, ts.unregistered, ts.overflowed_threads);
    }
    if (hw_selftest) {
        // H3-8I-25AC fixture gate: all nine checks must pass before a model run starts.
        const auto t = ::potal::hw_window::selftest_evaluate();
        const bool golden = failures == 0;
        const bool pass = t.reset_accepted && t.read_response && t.monotonic && t.load_and_mac && t.ownership &&
                          t.alignment && t.no_overflow && golden;
        std::printf("HW_SELFTEST trace_reset=%d trace_read=%d monotonic=%d load_mac=%d ownership=%d alignment=%d "
                    "no_overflow=%d numeric_golden=%d rocc_completion=%d windows=%" PRIu64 " mac_cycles=%" PRIu64
                    " bdma_windows=%" PRIu64 " bracket_max=%" PRIu64 " bracket_disagree=%" PRIu64 " first_failure=%s\n",
                    t.reset_accepted, t.read_response, t.monotonic, t.load_and_mac, t.ownership, t.alignment,
                    t.no_overflow, golden, golden && t.read_response, t.windows, t.mac_cycles, t.bdma_windows,
                    t.alignment_width_max, t.alignment_inconsistent, t.first_failure ? t.first_failure : "none");
        ::potal::hw_window::flush();
        std::printf("HW_SELFTEST_VERDICT %s\n", pass ? "PASS" : "FAIL");
        if (!pass) failures += 1;
    }
    std::printf("BENCH_VERDICT %s failures=%d runs=%zu\n", failures == 0 ? "PASS" : "FAIL", failures,
                (size_t) ((repeat + warmup) * (int64_t) modes.size()));
    return failures == 0 ? 0 : 1;
}
