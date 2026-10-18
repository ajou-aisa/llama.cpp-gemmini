// Hardware-free tests of the stripe pipeline (matmul/stripe.hpp) with real
// ExSIA stripe-ready events and the production run, sink and driver
// (run_stripe_pipeline). The accelerator owner is a test thread whose planned
// GEMMs run on the RTL-semantics emulator through potal_tiled_matmul_stripe;
// a gate in that runner holds one GEMM of one stripe so the overlap and
// backpressure order is checked deterministically (condition variables, no
// sleeps, no timestamp comparisons). This shows what the software allows to
// overlap on the host; it is not a measurement of NPU overlap.
//
//   A  overlap: Main(0) starts before quantization ends; Quantize(1) and
//      submit(1) complete while stripe 0's last GEMM is held; submit(2) waits
//      for capacity (never fails) and is accepted only after stripe 0's
//      semantic completion; Main(s+1) starts after SemanticDone(s); at most
//      job_capacity stripes outstanding; no quantization barrier
//   B  serialized (capacity 1) == pipelined (capacity 2) == independent golden:
//      Main INT32, Residual INT32, correction INT64, final F32 (bitwise)
//   C  accepted stripes are immutable: args.A, theta and the packet list of the
//      producer scribbled after quantization, the held GEMMs still produce the
//      golden
//   D  failures: a held GEMM failing wakes the blocked producer, a failure after
//      quantization wakes finish(), a fatal one is reported fatal; producer-side
//      violations (order, duplicates, run id, rows, theta, foreign packet,
//      cpu_direct payload, missing stripes); an unavailable owner; none of
//      them commits any output
//   E  repeated runs on one long-lived session: identical results, slot reuse,
//      scale generation wrap, WRITE == RELEASE, one Residual plan per packet
//      and no Main tile search inside the run
//   F  producer placement (Linux): pin to one CPU other than the excluded one,
//      restore the caller's mask
//
// Needs GGML_GEMMINI_ACTIVATION_QUANT=EXSIA (the pipeline's only activation);
// other configurations check the refusal and skip the rest.

#include "ggml-cpp.h"
#include "ggml-potal.h"
#include "ggml-quants.h"
#include "log.hpp"
#include "matmul/dense.hpp"
#include "matmul/execution.hpp"
#include "matmul/options.hpp"
#include "matmul/stripe.hpp"
#include "matmul/tiling.hpp"
#include "potal.h"
#include "potal_hw_params.h"
#include "support/stripe-golden.hpp"
#include "support/stripe-owner.hpp"

#include "quants/act/exsia/types.hpp"
#include "quants/act/quantize.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>

#if defined(__linux__)
#    include <pthread.h>
#    include <sched.h>
#endif

using namespace ggml::potal;
namespace grmd     = ggml::gemmini::rmd;
namespace exsia_ns = ggml::gemmini::quants::act::exsia;
using kind         = stripe_event_kind;

namespace {

int failures = 0;

void check(const std::string & name, bool ok) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    failures += ok ? 0 : 1;
}

constexpr size_t D = POTAL_HW_DIM;

// ---- fixtures ------------------------------------------------------------------

// One F32 x Q8_HP1 MUL_MAT: weights quantized by ggml, activations with
// outliers on every other row (so ExSIA seals residual packets with RMD ON).
struct node {
    size_t           M, N, K;
    ggml_context_ptr ctx;
    ggml_tensor *    w = nullptr;
    ggml_tensor *    x = nullptr;
    ggml_tensor *    y = nullptr;
    hp1_shape        s;
    potal_tile_plan  plan;
    hp1_weight       weight;
    bool             ok = false;

    node(size_t M_, size_t N_, size_t K_, uint32_t seed, float outlier) : M(M_), N(N_), K(K_) {
        ggml_init_params ip = { (M * K + N * K) * 8 + 4 * 1024 * 1024, nullptr, false };
        ctx.reset(ggml_init(ip));
        w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, (int64_t) K, (int64_t) N);
        x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, (int64_t) K, (int64_t) M);
        y = ggml_mul_mat(ctx.get(), w, x);
        std::mt19937                    rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float>              wf(N * K);
        const float                     mag[] = { 0.02f, 0.08f, 0.3f, 0.01f };
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                wf[n * K + k] = nd(rng) * mag[(k / 32 + n) % 4];
            }
        }
        ggml_quantize_chunk(GGML_TYPE_Q8_HP1, wf.data(), w->data, 0, (int64_t) N, (int64_t) K, nullptr);
        set_activations(seed + 1, outlier);
        s                       = { (int64_t) M, (int64_t) N, (int64_t) K };
        ggml_gemmini_args_t args;
        ok = potal_tile_select(M, N, K, 32 / D, &plan) == POTAL_OK && args_for(args) &&
             weight.allocate((int64_t) N, (int64_t) K) &&
             prepare_weight(args, weight) == quants::weight::hp1::status::ok;
    }

    void set_activations(uint32_t seed, float outlier) {
        std::mt19937                    rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        float *                         v = static_cast<float *>(x->data);
        for (size_t m = 0; m < M; ++m) {
            for (size_t k = 0; k < K; ++k) {
                v[m * K + k] = nd(rng) * (1.0f + 0.5f * (float) (m % 3)) * (k / 32 % 2 ? 4.0f : 1.0f);
            }
            if (outlier != 0.0f && m % 2 == 0) {
                v[m * K + (m * 7) % K] = outlier * (m % 4 ? -1.0f : 1.0f);
            }
        }
    }

    bool args_for(ggml_gemmini_args_t & args) const {
        return init_args(w, y, s, "gpt2", "pipeline-test", ggml::gemmini::residual::ResidualRoute::ws_packet, plan,
                         args);
    }

    size_t stripes() const { return (M + plan.tile_I * D - 1) / (plan.tile_I * D); }
    float * dst() const { return static_cast<float *>(y->data); }
};

// FULL-style ExSIA of the same node (no sink): the golden's codes, theta and packets.
struct golden_node {
    ggml_gemmini_args_t                  args;
    hp1_operands                         o;
    std::vector<float>                   want;
    std::map<size_t, grmd::StripePacketHandle> packets; // by stripe id
    potal_test::pipeline_golden_counts   counts;
    bool                                 ok = false;

    explicit golden_node(const node & nd) {
        ok = nd.args_for(args) && o.allocate(nd.s, nd.weight) && quantize_activations(nd.x, args, o) &&
             potal_test::pipeline_golden(args, o, want, &counts);
        if (const auto * meta = std::get_if<exsia_ns::Meta>(&args.act_quant.storage())) {
            for (const auto & p : meta->rmd_packets) {
                packets[p->stripe_id] = p;
            }
        }
    }

    // Main INT32 of rows [row_begin, row_end) from the golden codes.
    std::vector<int32_t> main_c(const node & nd, size_t row_begin, size_t row_end) const {
        std::vector<int32_t> c(o.m_pad * o.ldc, 0);
        reference_gemm_i32(o.a, o.lda, o.b, o.ldb, o.carriers, o.cstride, 32, nd.M, nd.N, nd.K, c.data(), o.ldc);
        std::vector<int32_t> rows;
        for (size_t m = row_begin; m < row_end; ++m) {
            rows.insert(rows.end(), c.begin() + m * o.ldc, c.begin() + m * o.ldc + nd.N);
        }
        return rows;
    }
};

// Every event of a run, in the order they happened, plus the data views.
struct recorder {
    struct rec {
        kind        k;
        size_t      stripe;
        size_t      slot;
        uint64_t    slot_generation;
        size_t      outstanding;
        std::string marker;
    };

    std::mutex                                mu;
    std::condition_variable                   cv;
    std::vector<rec>                          log;
    std::map<size_t, std::vector<int32_t>>    main_c;
    std::map<size_t, std::vector<int32_t>>    residual_c;
    std::map<size_t, std::vector<int64_t>>    correction;
    std::function<void(const stripe_event &)> hook; // runs on the event's thread, outside the lock

    void on(const stripe_event & ev) {
        {
            std::lock_guard<std::mutex> lock(mu);
            log.push_back({ ev.kind, ev.stripe_id, ev.slot, ev.slot_generation, ev.outstanding, {} });
            if (ev.kind == kind::main_end && ev.c != nullptr) {
                copy(main_c[ev.stripe_id], ev.c, ev.ldc, ev.rows, ev.cols);
            }
            if (ev.kind == kind::residual_end && ev.c != nullptr) {
                copy(residual_c[ev.stripe_id], ev.c, ev.ldc, ev.rows, ev.cols);
            }
            if (ev.kind == kind::radix_reconstruct_end && ev.correction != nullptr) {
                correction[ev.stripe_id].assign(ev.correction, ev.correction + ev.rows * ev.cols);
            }
            cv.notify_all();
        }
        if (hook) {
            hook(ev);
        }
    }

    static void copy(std::vector<int32_t> & dst, const int32_t * c, size_t ldc, size_t rows, size_t cols) {
        dst.clear();
        for (size_t r = 0; r < rows; ++r) {
            dst.insert(dst.end(), c + r * ldc, c + r * ldc + cols);
        }
    }

    void marker(const std::string & m) {
        std::lock_guard<std::mutex> lock(mu);
        log.push_back({ kind::run_failed, no_stripe, no_stripe, 0, 0, m });
        cv.notify_all();
    }

    // Index of the first record of (k, stripe), -1 if none.
    long seq(kind k, size_t stripe = no_stripe) {
        std::lock_guard<std::mutex> lock(mu);
        for (size_t i = 0; i < log.size(); ++i) {
            if (log[i].marker.empty() && log[i].k == k && (stripe == no_stripe || log[i].stripe == stripe)) {
                return (long) i;
            }
        }
        return -1;
    }

    long seq_marker(const std::string & m) {
        std::lock_guard<std::mutex> lock(mu);
        for (size_t i = 0; i < log.size(); ++i) {
            if (log[i].marker == m) {
                return (long) i;
            }
        }
        return -1;
    }

    size_t max_outstanding() {
        std::lock_guard<std::mutex> lock(mu);
        size_t m = 0;
        for (const rec & r : log) {
            m = std::max(m, r.outstanding);
        }
        return m;
    }

    // Waits until (k, stripe) was recorded; false after `seconds` (keeps a
    // broken pipeline from hanging the test; never a PASS criterion).
    bool wait_for(kind k, size_t stripe, int seconds = 60) {
        std::unique_lock<std::mutex> lock(mu);
        return cv.wait_for(lock, std::chrono::seconds(seconds), [&] {
            for (const rec & r : log) {
                if (r.marker.empty() && r.k == k && r.stripe == stripe) {
                    return true;
                }
            }
            return false;
        });
    }
};

// The emulator runner with one gate: the GEMM (stripe, work kind) waits until
// open(); it may then be failed instead of run.
struct gated_runner {
    potal_test::emu_session es;
    std::mutex              mu;
    std::condition_variable cv;
    bool                    armed  = false;
    size_t                  stripe = 0;
    uint32_t                work   = 0;
    bool                    opened = false;
    bool                    fail   = false;
    bool                    fatal  = false;
    bool                    held   = false; // the gated GEMM reached the gate

    void arm(size_t s, uint32_t w) {
        std::lock_guard<std::mutex> lock(mu);
        armed  = true;
        stripe = s;
        work   = w;
        opened = fail = fatal = held = false;
    }

    void open(bool fail_it = false, bool fatal_it = false) {
        std::lock_guard<std::mutex> lock(mu);
        fail   = fail_it;
        fatal  = fatal_it;
        opened = true;
        cv.notify_all();
    }

    planned_result run(const potal_stripe_gemm_desc & d) {
        bool inject = false, inject_fatal = false;
        {
            std::unique_lock<std::mutex> lock(mu);
            if (armed && d.stripe_id == stripe && d.work_kind == work) {
                held = true;
                cv.notify_all();
                cv.wait(lock, [this] { return opened; });
                armed        = false;
                inject       = fail;
                inject_fatal = fatal;
            }
        }
        if (inject) {
            planned_result r;
            r.status        = inject_fatal ? POTAL_HW_BAD_COMMAND : POTAL_ERR_INVALID_ARGUMENT;
            r.stage         = "injected";
            r.fatal         = inject_fatal;
            r.report.status = r.status;
            r.report.stage  = "injected";
            r.report.fatal  = inject_fatal ? 1 : 0;
            return r;
        }
        return es.run(d);
    }
};

struct run_result {
    pipeline_report report;
    consume_outcome consumer;
    bool            committed_untouched = false; // dst kept its sentinel everywhere
};

const uint32_t sentinel_bits = 0x7fc0beefu;

// One pipelined MUL_MAT of nd through the production driver.
run_result run_pipeline(node & nd, size_t capacity, gated_runner & runner, recorder & rec,
                        const std::function<void(ggml_gemmini_args_t &, const stripe_event &)> & producer_hook = {}) {
    ggml_gemmini_args_t args;
    run_result          r;
    if (!nd.args_for(args)) {
        r.report.stage = "fixture";
        return r;
    }
    float sentinel;
    std::memcpy(&sentinel, &sentinel_bits, sizeof(sentinel));
    std::fill(nd.dst(), nd.dst() + nd.M * nd.N, sentinel);
    pipeline_config cfg;
    cfg.job_capacity = capacity;
    cfg.residual     = options::compensation;
    cfg.layer        = "pipeline-test";
    cfg.backend      = "EMULATED";
    cfg.observer     = [&](const stripe_event & ev) {
        rec.on(ev);
        if (producer_hook) {
            producer_hook(args, ev);
        }
    };
    potal_test::thread_owner owner([&runner](const potal_stripe_gemm_desc & d) { return runner.run(d); });
    r.report = run_stripe_pipeline(nd.x, args, nd.weight, nd.plan, cfg, owner);
    owner.join();
    r.consumer            = owner.outcome;
    r.committed_untouched = std::all_of(nd.dst(), nd.dst() + nd.M * nd.N, [](float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return bits == sentinel_bits;
    });
    return r;
}

bool bitwise(const float * got, const std::vector<float> & want) {
    return std::memcmp(got, want.data(), want.size() * sizeof(float)) == 0;
}

uint32_t gated_kind() {
    return options::compensation ? POTAL_STRIPE_WORK_RESIDUAL : POTAL_STRIPE_WORK_MAIN;
}

// ---- A ------------------------------------------------------------------------------

void test_overlap(node & nd, const golden_node & g) {
    gated_runner runner;
    recorder     rec;
    runner.arm(0, gated_kind()); // stripe 0's last GEMM: Residual with RMD ON, else Main
    // Open only when the producer waits for capacity and (RMD ON: the gate holds
    // the Residual) Main(0) is already raw-complete.
    std::thread opener([&] {
        const bool waited =
            rec.wait_for(kind::queue_wait, 2) && (!options::compensation || rec.wait_for(kind::main_end, 0));
        rec.marker(waited ? "gate_open" : "gate_open_timeout");
        runner.open();
    });
    const run_result r = run_pipeline(nd, 2, runner, rec);
    opener.join();

    const long gate     = rec.seq_marker("gate_open");
    const long qend     = rec.seq(kind::quant_end);
    const long main0    = rec.seq(kind::main_begin, 0);
    const long acc1     = rec.seq(kind::submit_accepted, 1);
    const long fold1    = rec.seq(kind::folding_commit, 1);
    const long wait2    = rec.seq(kind::queue_wait, 2);
    const long acc2     = rec.seq(kind::submit_accepted, 2);
    const long end0     = rec.seq(options::compensation ? kind::residual_end : kind::main_end, 0);
    const long rawmain0 = rec.seq(kind::main_end, 0);
    const long sem0     = rec.seq(kind::semantic_complete, 0);
    const long rel0     = rec.seq(kind::capacity_release, 0);
    const long merge0   = rec.seq(kind::merge_end, 0);
    std::printf("     stripes=%zu rows/stripe=%zu packets(golden)=%zu events=%zu gate=%ld\n", nd.stripes(),
                nd.plan.tile_I * D, g.packets.size(), rec.log.size(), gate);

    check("A pipeline committed, every stripe semantically done",
          r.report.ok && r.report.counts.stripes_semantic_completed == nd.stripes() && !r.committed_untouched);
    check("A first Main starts before the whole activation is quantized", main0 >= 0 && qend > main0);
    check("A Quantize(1) and submit(1) complete while stripe 0's last GEMM is held",
          gate >= 0 && fold1 >= 0 && acc1 >= 0 && acc1 < gate && end0 > gate);
    check("A full pipeline: submit(2) waits for capacity instead of failing",
          wait2 >= 0 && wait2 < gate && acc2 > wait2 && r.report.counts.producer_capacity_waits >= 1);
    if (options::compensation) {
        check("A raw Main completion of stripe 0 returns no credit: submit(2) only after SemanticDone(0)",
              rawmain0 >= 0 && rawmain0 < gate && wait2 < gate && sem0 > gate && rel0 > sem0 && acc2 > rel0);
        check("A capacity comes back only after Residual, reconstruction and merge",
              merge0 > end0 && rel0 > merge0);
    } else {
        check("A (RMD OFF) capacity comes back only after Main(0) and its reconstruction: submit(2) after "
              "SemanticDone(0)",
              rawmain0 > gate && sem0 > rawmain0 && rel0 > sem0 && acc2 > rel0 && wait2 < gate);
    }
    bool ordered = true;
    for (size_t s = 0; s + 1 < nd.stripes(); ++s) {
        ordered = ordered && rec.seq(kind::main_begin, s + 1) > rec.seq(kind::semantic_complete, s) &&
                  rec.seq(kind::capacity_release, s) > rec.seq(kind::semantic_complete, s);
    }
    check("A Main(s+1) never starts before SemanticDone(s)", ordered);
    check("A at most job_capacity = 2 stripes outstanding",
          rec.max_outstanding() <= 2 && r.report.counts.max_outstanding_semantic == 2);
    check("A pipelined dst == independent golden (bitwise)", r.report.ok && bitwise(nd.dst(), g.want));

    bool records = r.report.completions.size() == nd.stripes();
    for (size_t s = 0; records && s < nd.stripes(); ++s) {
        const stripe_completion & c = r.report.completions[s];
        const size_t rb = s * nd.plan.tile_I * D;
        records = c.stripe_id == s && c.row_begin == rb && c.row_end == std::min(nd.M, rb + nd.plan.tile_I * D) &&
                  c.accepted_ns <= c.main_begin_ns && c.main_begin_ns <= c.semantic_done_ns &&
                  c.residual == (g.packets.count(s) == 1) && (!c.residual || (c.residual_rows > 0 &&
                                                                              c.k_execution >= c.k_compact));
    }
    check("A completion records: one per stripe, in order, rows and residual work as executed", records);
}

// ---- B ------------------------------------------------------------------------------

void test_serialized_vs_pipelined(node & nd, const golden_node & g) {
    gated_runner run1, run2;
    recorder     rec1, rec2;
    const run_result serial = run_pipeline(nd, 1, run1, rec1);
    const std::vector<float> serial_dst(nd.dst(), nd.dst() + nd.M * nd.N);
    const run_result piped = run_pipeline(nd, 2, run2, rec2);

    bool serial_order = true;
    for (size_t s = 0; s + 1 < nd.stripes(); ++s) {
        serial_order = serial_order && rec1.seq(kind::submit_accepted, s + 1) > rec1.seq(kind::capacity_release, s);
    }
    check("B capacity 1: one stripe at a time (submit(s+1) after capacity_release(s))",
          serial.report.ok && serial_order && serial.report.counts.max_outstanding_semantic == 1);

    const auto at = [](const auto & m, size_t key) {
        const auto it = m.find(key);
        return it != m.end() ? it->second : typename std::decay_t<decltype(m)>::mapped_type{};
    };
    bool same_main = true, same_res = true, same_corr = true, golden_main = true, golden_res = true;
    for (size_t s = 0; s < nd.stripes(); ++s) {
        const size_t rb = s * nd.plan.tile_I * D, re = std::min(nd.M, rb + nd.plan.tile_I * D);
        same_main       = same_main && at(rec1.main_c, s) == at(rec2.main_c, s) && !at(rec1.main_c, s).empty();
        golden_main     = golden_main && at(rec1.main_c, s) == g.main_c(nd, rb, re);
        same_res        = same_res && at(rec1.residual_c, s) == at(rec2.residual_c, s);
        same_corr       = same_corr && at(rec1.correction, s) == at(rec2.correction, s);
        const auto p    = g.packets.find(s);
        if (p != g.packets.end()) {
            std::vector<potal_test::golden_lane_row> gr;
            std::vector<int64_t>                     gc;
            std::vector<int32_t>                     flat;
            const bool                               ok = potal_test::residual_golden(
                *p->second, nd.N, D, [&](size_t k, size_t n) { return (int32_t) g.o.b[k * g.o.ldb + n]; },
                [&](size_t blk, size_t n) { return g.o.carriers[blk * g.o.cstride + n]; }, gr);
            for (const auto & row : gr) {
                flat.insert(flat.end(), row.c.begin(), row.c.end());
            }
            golden_res = golden_res && ok && potal_test::compose_golden(gr, p->second->row_count, nd.N, gc) &&
                         at(rec1.residual_c, s) == flat && at(rec1.correction, s) == gc;
        } else {
            golden_res = golden_res && rec1.residual_c.count(s) == 0 && rec2.residual_c.count(s) == 0;
        }
    }
    check("B serialized == pipelined: Main INT32 of every stripe", same_main && golden_main);
    check("B serialized == pipelined == golden: Residual INT32 and INT64 correction", same_res && same_corr &&
                                                                                          golden_res);
    check("B serialized == pipelined == golden: final F32 (bitwise)",
          serial.report.ok && piped.report.ok && bitwise(serial_dst.data(), g.want) && bitwise(nd.dst(), g.want));
    if (options::compensation) {
        check("B residual work actually ran (nonzero corrections merged)",
              piped.report.counts.residual_logical_gemms > 0 && piped.report.counts.corrections_nonzero > 0 &&
                  piped.report.counts.residual_nonempty_packets == g.packets.size() &&
                  piped.report.counts.merges_completed == g.packets.size());
    } else {
        check("B RMD OFF control: Main only, no residual work",
              piped.report.counts.residual_logical_gemms == 0 && piped.report.counts.merges_completed == 0);
    }
}

// ---- C ------------------------------------------------------------------------------

void test_immutability(const golden_node & g2, node & two) {
    // Two stripes fit the capacity: quantization ends while stripe 0's Main is
    // held; the producer then scribbles everything it owns.
    gated_runner runner;
    recorder     rec;
    runner.arm(0, POTAL_STRIPE_WORK_MAIN);
    bool scribbled = false;
    const run_result r = run_pipeline(two, 2, runner, rec, [&](ggml_gemmini_args_t & args, const stripe_event & ev) {
        if (ev.kind != kind::quant_end) {
            return;
        }
        if (args.A.valid()) {
            std::fill(args.A.bytes->begin(), args.A.bytes->end(), (uint8_t) 0x5a);
        }
        if (auto * meta = std::get_if<exsia_ns::Meta>(&args.act_quant.storage())) {
            std::fill(meta->theta.begin(), meta->theta.end(), (int16_t) 3);
            meta->rmd_packets.clear();
        }
        scribbled = true;
        runner.open();
    });
    check("C fixture: 2 stripes, quantization ended while Main(0) was held",
          two.stripes() == 2 && scribbled && rec.seq(kind::quant_end) < rec.seq(kind::main_end, 0));
    check("C accepted A rows, theta and packets are the run's own: dst == golden after the producer scribbled",
          r.report.ok && bitwise(two.dst(), g2.want));
}

// ---- D ------------------------------------------------------------------------------

void test_failures(node & nd) {
    {
        // The held GEMM of stripe 0 fails while the producer waits for capacity.
        gated_runner runner;
        recorder     rec;
        runner.arm(0, gated_kind());
        std::thread opener([&] {
            (void) rec.wait_for(kind::queue_wait, 2);
            runner.open(/* fail */ true, /* fatal */ false);
        });
        const run_result r = run_pipeline(nd, 2, runner, rec);
        opener.join();
        check("D failure while the producer is blocked: it wakes, the run fails, nothing committed",
              !r.report.ok && !r.report.fatal && r.report.stage != nullptr &&
                  std::string(r.report.stage) == (options::compensation ? "residual" : "main") &&
                  r.committed_untouched && r.report.counts.stripes_accepted == 2 &&
                  rec.seq(kind::submit_accepted, 2) < 0 && rec.seq(kind::run_failed) >= 0 && !r.consumer.fatal);
    }
    {
        // The last stripe fails after quantization ended: finish() wakes.
        gated_runner runner;
        recorder     rec;
        const size_t last = nd.stripes() - 1;
        runner.arm(last, POTAL_STRIPE_WORK_MAIN);
        std::thread opener([&] {
            (void) rec.wait_for(kind::quant_end, no_stripe); // quantization over, finish() waiting
            runner.open(true, false);
        });
        const run_result r = run_pipeline(nd, 2, runner, rec);
        opener.join();
        check("D failure after quantization: finish() wakes, nothing committed",
              !r.report.ok && r.committed_untouched && r.report.stage != nullptr &&
                  std::string(r.report.stage) == "main" && rec.seq(kind::quant_end) >= 0 &&
                  r.report.counts.stripes_accepted == nd.stripes());
    }
    {
        gated_runner runner;
        recorder     rec;
        runner.arm(1, POTAL_STRIPE_WORK_MAIN);
        runner.open(true, /* fatal */ true);
        const run_result r = run_pipeline(nd, 2, runner, rec);
        check("D fatal hardware failure: reported fatal by the run and the consumer, nothing committed",
              !r.report.ok && r.report.fatal && r.consumer.fatal && r.committed_untouched);
    }
    {
        potal_test::refusing_owner owner;
        ggml_gemmini_args_t        args;
        pipeline_config            cfg;
        cfg.residual = options::compensation;
        float sentinel;
        std::memcpy(&sentinel, &sentinel_bits, sizeof(sentinel));
        std::fill(nd.dst(), nd.dst() + nd.M * nd.N, sentinel);
        const pipeline_report r = nd.args_for(args) ? run_stripe_pipeline(nd.x, args, nd.weight, nd.plan, cfg, owner)
                                                    : pipeline_report{};
        check("D unavailable owner: refused before quantization, nothing committed",
              !r.ok && r.stage != nullptr && std::string(r.stage) == "issuer" && r.status == POTAL_ERR_NO_HARDWARE &&
                  r.counts.stripes_accepted == 0 && args.act_quant.kind() == ggml::gemmini::quants::act::MetaKind::none &&
                  nd.dst()[0] != nd.dst()[0]);
    }
    {
        // A configuration the pipeline does not run is refused, never replaced.
        potal_test::refusing_owner owner;
        ggml_gemmini_args_t        args;
        pipeline_config            cfg;
        cfg.residual     = options::compensation;
        cfg.job_capacity = 3;
        const pipeline_report r = nd.args_for(args) ? run_stripe_pipeline(nd.x, args, nd.weight, nd.plan, cfg, owner)
                                                    : pipeline_report{};
        check("D job capacity 3 refused", !r.ok && r.stage != nullptr && std::string(r.stage) == "options");
    }
}

// Producer-side contract on synthetic events, through the production run.
void test_contract(node & nd) {
    ggml_gemmini_args_t args;
    if (!nd.args_for(args)) {
        check("D contract fixture", false);
        return;
    }
    const size_t rows = nd.plan.tile_I * D;
    const auto   event = [&](uint64_t run, size_t stripe, size_t rb, size_t re) {
        exsia_ns::StripeReadyEvent e;
        e.run_id              = run;
        e.stripe_id           = stripe;
        e.row_begin           = rb;
        e.row_end             = re;
        e.activation_metadata = exsia_ns::StripeMetadataSnapshot{};
        e.activation_metadata->theta = -5;
        return e;
    };
    struct violation {
        const char *                                                         name;
        std::function<bool(stripe_run &, const ggml_gemmini_args_t &)>       apply; // false: the violating call failed
    };
    const std::vector<violation> violations = {
        { "out-of-order stripe",
          [&](stripe_run & run, const ggml_gemmini_args_t & a) { return run.submit(event(9, 1, rows, 2 * rows), a); } },
        { "duplicate stripe",
          [&](stripe_run & run, const ggml_gemmini_args_t & a) {
              return run.submit(event(9, 0, 0, rows), a) && run.submit(event(9, 0, 0, rows), a);
          } },
        { "stripe of another run id",
          [&](stripe_run & run, const ggml_gemmini_args_t & a) {
              return run.submit(event(9, 0, 0, rows), a) && run.submit(event(10, 1, rows, 2 * rows), a);
          } },
        { "rows outside the frozen stripe height",
          [&](stripe_run & run, const ggml_gemmini_args_t & a) { return run.submit(event(9, 0, 0, rows - 1), a); } },
        { "missing theta",
          [&](stripe_run & run, const ggml_gemmini_args_t & a) {
              auto e = event(9, 0, 0, rows);
              e.activation_metadata.reset();
              return run.submit(e, a);
          } },
        { "foreign residual packet",
          [&](stripe_run & run, const ggml_gemmini_args_t & a) {
              grmd::RmdStripeBuilder b;
              b.reset(5, 0, rows, nd.K, nd.N, 8);
              b.add_residual(0, 3, 300);
              auto e       = event(9, 0, 0, rows);
              e.rmd_packet = b.finish();
              return run.submit(e, a);
          } },
        { "cpu_direct payload",
          [&](stripe_run & run, const ggml_gemmini_args_t & a) {
              auto e            = event(9, 0, 0, rows);
              e.direct_residual = std::make_shared<ggml::gemmini::residual::DirectStripePayload>();
              return run.submit(e, a);
          } },
        { "missing stripes at finish",
          [&](stripe_run & run, const ggml_gemmini_args_t & a) {
              return run.submit(event(9, 0, 0, rows), a) && run.finish();
          } },
    };
    for (const violation & v : violations) {
        pipeline_config cfg;
        cfg.residual = options::compensation;
        cfg.backend  = "EMULATED";
        stripe_run   run(cfg, args, nd.weight, nd.plan);
        std::string  why;
        gated_runner runner;
        const bool   begun = run.begin(why);
        std::thread  consumer([&] { (void) run.consume([&](const potal_stripe_gemm_desc & d) { return runner.run(d); }); });
        const bool   accepted = begun && v.apply(run, args);
        run.abort("test", "end of case"); // no-op detail when already failed (sticky first error)
        consumer.join();
        const pipeline_report r = run.report();
        check(std::string("D producer contract: ") + v.name + " fails the run (" + r.detail + ")",
              begun && !accepted && !r.ok && r.stage != nullptr && std::string(r.stage) != "test");
    }
}

// ---- E ------------------------------------------------------------------------------

void test_repeated(node & nd, const golden_node & g) {
    gated_runner   runner; // one session across runs, like the issuer's
    bool           same = true, all_ok = true;
    uint64_t       selects = 0, packets = 0, loops = 0;
    size_t         reused  = 0, runs = 0;
    const uint64_t before  = potal_tile_select_count();
    while (runner.es.emu.loops <= 2u * POTAL_GENERATION_MAX + 3 && runs < 64) {
        recorder         rec;
        const run_result r = run_pipeline(nd, 2, runner, rec);
        all_ok             = all_ok && r.report.ok;
        same               = same && bitwise(nd.dst(), g.want);
        packets += r.report.counts.residual_nonempty_packets;
        loops += r.report.counts.main_loop_ws + r.report.counts.residual_loop_ws;
        for (const auto & e : rec.log) {
            reused += e.marker.empty() && e.k == kind::main_begin && e.slot_generation > 0;
        }
        runs++;
    }
    selects                = potal_tile_select_count() - before;
    const auto & gens      = runner.es.emu.loop_generations;
    bool         wrapped   = false;
    for (size_t i = 1; i < gens.size(); ++i) {
        wrapped = wrapped || (gens[i - 1] == POTAL_GENERATION_MAX && gens[i] == 1);
    }
    std::printf("     runs=%zu loop_ws=%llu residual_plans=%llu tile_selects=%llu slot_reuses=%zu\n", runs,
                (unsigned long long) loops, (unsigned long long) packets, (unsigned long long) selects, reused);
    check("E repeated runs on one session: every run committed the golden", all_ok && same && runs >= 2);
    check("E slots reused across stripes (slot generation > 0)", reused > 0);
    check("E scale generation wraps 255 -> 1 inside one session; WRITE == RELEASE",
          wrapped && runner.es.emu.writes == runner.es.emu.releases && runner.es.emu.scales.empty() &&
              loops == runner.es.emu.loops);
    check("E one Residual plan per packet, no Main tile search inside the runs", selects == packets);
}

// ---- F ------------------------------------------------------------------------------

void test_affinity() {
#if defined(__linux__)
    cpu_set_t original;
    CPU_ZERO(&original);
    pthread_getaffinity_np(pthread_self(), sizeof(original), &original);
    const int exclude = sched_getcpu();
    for (const bool one_cpu : { true, false }) {
        std::string why;
        bool        narrowed = false;
        int         chosen   = -1;
        {
            producer_affinity a;
            narrowed = a.pin(exclude, one_cpu, why);
            chosen   = a.cpu();
            cpu_set_t now;
            CPU_ZERO(&now);
            pthread_getaffinity_np(pthread_self(), sizeof(now), &now);
            cpu_set_t expected;
            CPU_ZERO(&expected);
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (CPU_ISSET(c, &original) && c != exclude && (!one_cpu || CPU_COUNT(&expected) == 0)) {
                    CPU_SET(c, &expected);
                }
            }
            narrowed = narrowed && CPU_EQUAL(&now, &expected) && !CPU_ISSET(exclude, &now) &&
                       (one_cpu ? chosen >= 0 && CPU_COUNT(&now) == 1 : chosen == -1);
        } // destructor restores
        cpu_set_t after;
        CPU_ZERO(&after);
        pthread_getaffinity_np(pthread_self(), sizeof(after), &after);
        const std::string mode = one_cpu ? "one CPU" : "every CPU but the excluded one";
        if (CPU_COUNT(&original) > 1) {
            check("F producer narrowed to " + mode + " (excluded cpu " + std::to_string(exclude) + ")", narrowed);
        } else {
            check("F single-CPU mask: placement refused explicitly (" + mode + ")", !narrowed && !why.empty());
        }
        check("F caller's affinity mask restored (" + mode + ")", CPU_EQUAL(&after, &original));
    }
#else
    std::printf("[SKIP] F producer placement needs Linux\n");
#endif
}

} // namespace

int main() {
    ggml::potal::log::ensure_outputs(); // Gemmini debug/cycle JSONL, not stdout
    std::printf("potal-stripe-pipeline-test DIM=%u backend=EMULATED (host threads, no hardware) activation=%s "
                "compensation=%s residual_executor=%s\n",
                (unsigned) POTAL_HW_DIM, options::activation_quant_name(), options::compensation_name,
                options::residual_executor_name(options::MatmulInvocationMode::stripe_pipeline));
    test_affinity();
    if (options::activation_quant != options::ActivationQuantAlgo::EXSIA) {
        node                       nd(48, 32, 64, 1, 0.0f);
        ggml_gemmini_args_t        args;
        potal_test::refusing_owner owner;
        pipeline_config            cfg;
        cfg.residual            = options::compensation;
        const pipeline_report r = nd.ok && nd.args_for(args)
                                      ? run_stripe_pipeline(nd.x, args, nd.weight, nd.plan, cfg, owner)
                                      : pipeline_report{};
        check("STRIPE_PIPELINE refused for a non-ExSIA activation (no replacement path)",
              !r.ok && r.stage != nullptr && std::string(r.stage) == "options");
        std::printf("[SKIP] pipeline cases need GGML_GEMMINI_ACTIVATION_QUANT=EXSIA\n");
    } else {
        node        nd(200, 320, 128, 7, 400.0f); // 3 stripes with outliers
        golden_node g(nd);
        check("fixture: >= 3 stripes from the frozen plan, golden built" +
                  std::string(options::compensation ? ", stripe 0 has a residual packet" : ""),
              nd.ok && g.ok && nd.stripes() >= 3 && (!options::compensation || g.packets.count(0) == 1));
        if (nd.ok && g.ok && nd.stripes() >= 3) {
            test_overlap(nd, g);
            test_serialized_vs_pipelined(nd, g);
            test_failures(nd);
            test_contract(nd);
            test_repeated(nd, g);
        }
        node        two(140, 320, 128, 9, 400.0f); // 2 stripes
        golden_node g2(two);
        if (two.ok && g2.ok) {
            test_immutability(g2, two);
        } else {
            check("C fixture", false);
        }
        node calm(200, 320, 128, 11, 0.0f);
        // Constant activations: one exponent per stripe (ExSIA's second exponent
        // is -inf), so no top block is promoted and no packet is sealed.
        float * v = static_cast<float *>(calm.x->data);
        for (size_t i = 0; i < calm.M * calm.K; ++i) {
            v[i] = 1.0f;
        }
        golden_node gc(calm);
        gated_runner runner;
        recorder     rec;
        const run_result r = calm.ok && gc.ok ? run_pipeline(calm, 2, runner, rec) : run_result{};
        check("empty residual (" + std::to_string(gc.packets.size()) + " packets of " +
                  std::to_string(calm.stripes()) + " stripes): Main only where no packet, == golden",
              r.report.ok && bitwise(calm.dst(), gc.want) && gc.packets.empty() &&
                  r.report.counts.residual_logical_gemms == gc.packets.size() &&
                  r.report.counts.stripes_semantic_completed == calm.stripes());
    }
    if (failures != 0) {
        std::printf("POTAL_STRIPE_PIPELINE_TEST FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("POTAL_STRIPE_PIPELINE_TEST PASS\n");
    return 0;
}
