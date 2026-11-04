// H3-8I-25P pipeline bench: one F32 x Q8_HP1 MUL_MAT (Linear) through the POTAL backend, repeated
// under several in-process modes. Every run is compared bitwise with the CPU golden built from the
// same ExSIA codes (support/stripe-golden.hpp, as in potal-stripe-smoke), and with the first mode's
// output. Wall time is CLOCK_MONOTONIC around ggml_backend_graph_compute only.
//
//   potal-pipeline-bench --m=M --n=N --k=K [--outlier=F] [--blocks=B] [--repeat=R] [--seed=S]
//                        [--modes=serial,roles,queue,roles+queue]
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
};

bool parse_modes(const std::string & list, std::vector<mode> & out) {
    size_t pos = 0;
    while (pos <= list.size()) {
        const size_t end  = list.find(',', pos);
        const std::string item = list.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        mode m;
        m.name = item;
        if (item == "serial") {
        } else if (item == "roles") {
            m.roles = true;
        } else if (item == "queue") {
            m.queue = true;
        } else if (item == "roles+queue") {
            m.roles = m.queue = true;
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
    for (int64_t rep = -warmup; rep < repeat; ++rep) {
        for (size_t mi = 0; mi < modes.size(); ++mi) {
            const mode & md = modes[mi];
            setenv("GGML_POTAL_HART_ROLES", md.roles ? "ON" : "OFF", 1);
            setenv("GGML_POTAL_INDEX_QUEUE", md.queue ? "ON" : "OFF", 1);
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
            const bool ok = status == GGML_STATUS_SUCCESS && h == gold_hash &&
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
        }
    }
    for (size_t mi = 0; mi < modes.size(); ++mi) {
        std::vector<int64_t> v = walls[mi];
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
    std::printf("BENCH_VERDICT %s failures=%d runs=%zu\n", failures == 0 ? "PASS" : "FAIL", failures,
                (size_t) ((repeat + warmup) * (int64_t) modes.size()));
    return failures == 0 ? 0 : 1;
}
