// Hardware-free tests of the stripe pipeline (matmul/stripe.hpp) with real
// ExSIA stripe-ready events and the production run, sink and driver
// (run_stripe_pipeline). The accelerator owner is a test thread whose planned
// GEMMs run on the RTL-semantics emulator through the issuer's own step
// (run_planned_stripe_gemm: preflight, TLB FLUSH + FENCE, the GEMM); a gate in
// that runner holds one GEMM of one stripe so the overlap and backpressure
// order is checked deterministically (condition variables, no sleeps, no
// timestamp comparisons). This shows what the software allows to overlap on
// the host; it is not a measurement of NPU overlap.
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
//   O  event order: every run's records against the order the pipeline
//      guarantees (sequence 1..n without gaps or duplicates; per stripe
//      folding_commit < submit_accepted < main_begin < main_end < residual
//      prepare/GEMM/reconstruction/merge < semantic_complete <
//      capacity_release; capacity_release(s) < submit_accepted(s + capacity);
//      main_begin(s + 1) after semantic_complete(s)); observers that query,
//      abort or throw on either thread; a merge failure; 24 gated runs
//      alternating capacity 1/2, with and without packets, with forced
//      failures; and (LOG_DEBUG) the debug JSONL written in sequence order
//
// Event positions are logical: an event's sequence (stamped with the state
// change under the run's lock), a test marker right after the highest
// sequence observed when it was set. Observers on the producer and the
// consumer run concurrently, so their call order alone is not the run's order.
// The stamping placement itself holds by construction (stripe.hpp); a stamp
// moved out of its critical section would show up here only when the threads
// interleave inside that window, i.e. under stress, not deterministically.
// Without GEMMINI_LOG_DIR the test logs to a fresh private directory, so the
// JSONL it checks is its own (ctest may run the potal tests side by side).
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
#include "quants/common/weight_reader.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if defined(__linux__)
#    include <pthread.h>
#    include <sched.h>
#endif

using namespace ggml::potal;
namespace grmd     = ggml::gemmini::rmd;
namespace exsia_ns = ggml::gemmini::quants::act::exsia;
namespace wreader  = ggml::gemmini::quants::wreader;
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

    std::set<size_t> packet_stripes() const {
        std::set<size_t> ids;
        for (const auto & p : packets) {
            ids.insert(p.first);
        }
        return ids;
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

// Every event of a run with its sequence (the run's logical order), test
// markers, and the data views.
struct recorder {
    struct rec {
        kind        k;
        size_t      stripe;
        size_t      slot;
        uint64_t    slot_generation;
        size_t      outstanding;
        uint64_t    sequence; // event: its sequence; marker: the highest sequence observed before it
        uint64_t    t_ns;
        std::string marker;

        long position() const { return (long) (2 * sequence + (marker.empty() ? 0 : 1)); }
    };

    std::mutex                                mu;
    std::condition_variable                   cv;
    std::vector<rec>                          log;
    uint64_t                                  max_sequence = 0;
    std::map<size_t, std::vector<int32_t>>    main_c;
    std::map<size_t, std::vector<int32_t>>    residual_c;
    std::map<size_t, std::vector<int64_t>>    correction;
    std::function<void(const stripe_event &)> hook; // runs on the event's thread, outside the lock

    void on(const stripe_event & ev) {
        {
            std::lock_guard<std::mutex> lock(mu);
            log.push_back(
                { ev.kind, ev.stripe_id, ev.slot, ev.slot_generation, ev.outstanding, ev.sequence, ev.t_ns, {} });
            max_sequence = std::max(max_sequence, ev.sequence);
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

    // A marker sits right after every step observed so far: a step stamped
    // later (e.g. after a gate this marker records opening) comes after it.
    void marker(const std::string & m) {
        std::lock_guard<std::mutex> lock(mu);
        log.push_back({ kind::run_failed, no_stripe, no_stripe, 0, 0, max_sequence, 0, m });
        cv.notify_all();
    }

    // Logical position of the first (k, stripe) step, -1 if none.
    long seq(kind k, size_t stripe = no_stripe) {
        std::lock_guard<std::mutex> lock(mu);
        long first = -1;
        for (const rec & r : log) {
            if (r.marker.empty() && r.k == k && (stripe == no_stripe || r.stripe == stripe) &&
                (first < 0 || r.position() < first)) {
                first = r.position();
            }
        }
        return first;
    }

    long seq_marker(const std::string & m) {
        std::lock_guard<std::mutex> lock(mu);
        for (const rec & r : log) {
            if (r.marker == m) {
                return r.position();
            }
        }
        return -1;
    }

    std::vector<rec> events() {
        std::lock_guard<std::mutex> lock(mu);
        std::vector<rec> e;
        for (const rec & r : log) {
            if (r.marker.empty()) {
                e.push_back(r);
            }
        }
        return e;
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

// ---- order -----------------------------------------------------------------------

// The run's steps against the order the pipeline guarantees; empty when it
// holds, else the first violation. complete: the run committed, so stripes
// 0..stripes-1 have every step; otherwise each stripe's steps are a prefix of
// its chain. packets: the stripes that must carry the residual steps
// (nullptr: the stripes that show a residual_prepare_begin).
std::string order_violation(recorder & rec, size_t capacity, size_t stripes, bool complete,
                            const std::set<size_t> * packets = nullptr) {
    const std::vector<recorder::rec> ev = rec.events();
    std::vector<uint64_t>            seqs;
    for (const recorder::rec & r : ev) {
        seqs.push_back(r.sequence);
    }
    std::sort(seqs.begin(), seqs.end());
    for (size_t i = 0; i < seqs.size(); ++i) {
        if (seqs[i] != i + 1) {
            return "sequence " + std::to_string(seqs[i]) + " at rank " + std::to_string(i + 1) +
                   " (gap or duplicate)";
        }
    }
    // t_ns is stamped with the sequence: set, and never decreasing along it.
    std::vector<recorder::rec> by_sequence = ev;
    std::sort(by_sequence.begin(), by_sequence.end(),
              [](const recorder::rec & x, const recorder::rec & y) { return x.sequence < y.sequence; });
    for (size_t i = 0; i < by_sequence.size(); ++i) {
        if (by_sequence[i].t_ns == 0 || (i > 0 && by_sequence[i].t_ns < by_sequence[i - 1].t_ns)) {
            return "t_ns of sequence " + std::to_string(by_sequence[i].sequence) + " unset or before its predecessor";
        }
    }
    std::map<std::pair<kind, size_t>, std::vector<const recorder::rec *>> at;
    uint64_t                                                              last = 0;
    for (const recorder::rec & r : ev) {
        at[{ r.k, r.stripe }].push_back(&r);
        last = std::max(last, r.sequence);
        if (r.outstanding > capacity) {
            return std::string(stripe_event_name(r.k)) + " records " + std::to_string(r.outstanding) +
                   " outstanding stripes";
        }
    }
    for (const auto & e : at) {
        if (e.second.size() > 1) {
            return std::string("duplicated ") + stripe_event_name(e.first.first) + "(" +
                   (e.first.second == no_stripe ? std::string("run") : std::to_string(e.first.second)) + ")";
        }
    }
    const auto one = [&](kind k, size_t s) -> const recorder::rec * {
        const auto it = at.find({ k, s });
        return it == at.end() ? nullptr : it->second.front();
    };
    const auto name = [](kind k, size_t s) { return std::string(stripe_event_name(k)) + "(" + std::to_string(s) + ")"; };
    for (size_t s = 0; s < stripes; ++s) {
        const bool        residual = packets != nullptr ? packets->count(s) != 0
                                                        : one(kind::residual_prepare_begin, s) != nullptr;
        std::vector<kind> chain    = { kind::folding_commit, kind::submit_accepted, kind::main_begin, kind::main_end };
        if (residual) {
            chain.insert(chain.end(), { kind::residual_prepare_begin, kind::residual_prepare_end, kind::residual_begin,
                                        kind::residual_end, kind::radix_reconstruct_begin,
                                        kind::radix_reconstruct_end, kind::merge_begin, kind::merge_end });
        }
        chain.insert(chain.end(), { kind::semantic_complete, kind::capacity_release });
        const recorder::rec * previous = nullptr;
        kind                  prev_k   = chain.front();
        bool                  gap      = false;
        for (kind k : chain) {
            const recorder::rec * r = one(k, s);
            if (r == nullptr) {
                if (complete) {
                    return "missing " + name(k, s);
                }
                gap = true;
                continue;
            }
            if (gap) {
                return name(k, s) + " after a missing earlier step";
            }
            if (previous != nullptr && r->sequence <= previous->sequence) {
                return name(k, s) + " not after " + name(prev_k, s);
            }
            previous = r;
            prev_k   = k;
        }
        if (!residual && one(kind::residual_begin, s) != nullptr) {
            return "residual steps for " + name(kind::main_begin, s) + " without a packet";
        }
        const recorder::rec * wait = one(kind::queue_wait, s);
        const recorder::rec * fold = one(kind::folding_commit, s);
        const recorder::rec * acc  = one(kind::submit_accepted, s);
        if (wait != nullptr && (fold == nullptr || wait->sequence < fold->sequence ||
                                (acc != nullptr && acc->sequence < wait->sequence))) {
            return name(kind::queue_wait, s) + " outside folding_commit..submit_accepted";
        }
        const recorder::rec * begin = one(kind::main_begin, s);
        const recorder::rec * rel   = one(kind::capacity_release, s);
        if (begin != nullptr && acc != nullptr &&
            (begin->slot != acc->slot || begin->slot_generation != acc->slot_generation)) {
            return name(kind::main_begin, s) + " in another slot or generation than its acceptance";
        }
        if (begin != nullptr && rel != nullptr &&
            (rel->slot != begin->slot || rel->slot_generation != begin->slot_generation + 1)) {
            return name(kind::capacity_release, s) + " does not advance its slot's generation by one";
        }
        if (s + 1 < stripes) {
            const recorder::rec * next_main = one(kind::main_begin, s + 1);
            const recorder::rec * sem       = one(kind::semantic_complete, s);
            if (next_main != nullptr && (sem == nullptr || rel == nullptr || next_main->sequence < sem->sequence ||
                                         next_main->sequence < rel->sequence)) {
                return name(kind::main_begin, s + 1) + " before the semantic completion of stripe " + std::to_string(s);
            }
            const recorder::rec * next_fold = one(kind::folding_commit, s + 1);
            const recorder::rec * next_acc  = one(kind::submit_accepted, s + 1);
            if ((next_fold != nullptr && (fold == nullptr || next_fold->sequence < fold->sequence)) ||
                (next_acc != nullptr && (acc == nullptr || next_acc->sequence < acc->sequence))) {
                return "stripe " + std::to_string(s + 1) + " handed over before stripe " + std::to_string(s);
            }
        }
        const recorder::rec * reuse = s + capacity < stripes ? one(kind::submit_accepted, s + capacity) : nullptr;
        if (reuse != nullptr && (rel == nullptr || reuse->sequence < rel->sequence)) {
            return name(kind::submit_accepted, s + capacity) + " before " + name(kind::capacity_release, s);
        }
    }
    const recorder::rec * qbegin = one(kind::quant_begin, no_stripe);
    const recorder::rec * qend   = one(kind::quant_end, no_stripe);
    const recorder::rec * commit = one(kind::run_commit, no_stripe);
    const recorder::rec * failed = one(kind::run_failed, no_stripe);
    if (qbegin != nullptr && qbegin->sequence != 1) {
        return "quant_begin is not the run's first step";
    }
    if ((commit != nullptr && commit->sequence != last) || (failed != nullptr && failed->sequence != last) ||
        (complete && failed != nullptr) || (commit != nullptr && failed != nullptr)) {
        return "run_commit/run_failed is not the run's one last step";
    }
    for (const recorder::rec & r : ev) {
        if (qend != nullptr && (r.k == kind::folding_commit || r.k == kind::submit_accepted || r.k == kind::queue_wait) &&
            r.sequence > qend->sequence) {
            return name(r.k, r.stripe) + " after quant_end";
        }
    }
    return {};
}

void check_order(const std::string & name, recorder & rec, size_t capacity, size_t stripes, bool complete,
                 const std::set<size_t> * packets = nullptr) {
    const std::string why = order_violation(rec, capacity, stripes, complete, packets);
    check(name + (why.empty() ? std::string() : " -- " + why), why.empty());
}

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
    check("A pipeline counts activation quantization call time", r.report.counts.act_quant_s > 0);
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
                  c.accepted_ns > 0 && c.accepted_ns <= c.main_begin_ns && c.main_begin_ns <= c.semantic_done_ns &&
                  c.residual == (g.packets.count(s) == 1) && (!c.residual || (c.residual_rows > 0 &&
                                                                              c.k_execution >= c.k_compact));
    }
    check("A completion records: one per stripe, in order, rows and residual work as executed", records);
    const std::set<size_t> packets = g.packet_stripes();
    check_order("A event order: sequence 1..n, per-stripe chain, capacity_release(s) < submit_accepted(s+2), "
                "Main(s+1) after SemanticDone(s)",
                rec, 2, nd.stripes(), true, &packets);
}

// ---- B ------------------------------------------------------------------------------

void test_serialized_vs_pipelined(node & nd, const golden_node & g) {
    gated_runner run1, run2;
    recorder     rec1, rec2;
    // The weight was prepared (validated) with the fixture: from here on no
    // Gemmini reader call may touch it (GGML_GEMMINI_TESTING counters).
    wreader::test_reset_weight_reader_counters();
    const run_result serial = run_pipeline(nd, 1, run1, rec1);
    const std::vector<float> serial_dst(nd.dst(), nd.dst() + nd.M * nd.N);
    const run_result piped = run_pipeline(nd, 2, run2, rec2);
    const size_t     validations = wreader::test_weight_reader_storage_validations();
    const size_t     reads       = wreader::test_weight_reader_code_address_resolutions();
    check("B no weight revalidation in the pipeline: " + std::to_string(validations) + " wreader::validate and " +
              std::to_string(reads) + " weight reads over " +
              std::to_string(serial.report.counts.merges_completed + piped.report.counts.merges_completed) +
              " residual merges",
          validations == 0 && reads == 0 && serial.report.ok && piped.report.ok &&
              (!options::compensation ||
               serial.report.counts.merges_completed + piped.report.counts.merges_completed > 0));

    bool serial_order = true;
    for (size_t s = 0; s + 1 < nd.stripes(); ++s) {
        serial_order = serial_order && rec1.seq(kind::submit_accepted, s + 1) > rec1.seq(kind::capacity_release, s);
    }
    check("B capacity 1: one stripe at a time (submit(s+1) after capacity_release(s))",
          serial.report.ok && serial_order && serial.report.counts.max_outstanding_semantic == 1);
    const std::set<size_t> packets = g.packet_stripes();
    check_order("B event order, capacity 1", rec1, 1, nd.stripes(), true, &packets);
    check_order("B event order, capacity 2", rec2, 2, nd.stripes(), true, &packets);

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
    const std::set<size_t> packets = g2.packet_stripes();
    check_order("C event order", rec, 2, two.stripes(), true, &packets);
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
        check_order("D event order of the run failed while the producer waited", rec, 2, nd.stripes(), false);
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
        check_order("D event order of the run failed after quantization", rec, 2, nd.stripes(), false);
    }
    {
        gated_runner runner;
        recorder     rec;
        runner.arm(1, POTAL_STRIPE_WORK_MAIN);
        runner.open(true, /* fatal */ true);
        const run_result r = run_pipeline(nd, 2, runner, rec);
        check("D fatal hardware failure: reported fatal by the run and the consumer, nothing committed",
              !r.report.ok && r.report.fatal && r.consumer.fatal && r.committed_untouched);
        check_order("D event order of the fatal run", rec, 2, nd.stripes(), false);
    }
    {
        // The emulator itself fails inside the owner step (not the gate):
        // FENCE 0 is Main(0)'s TLB FENCE, FENCE 1 its first GEMM FENCE.
        gated_runner runner;
        recorder     rec;
        runner.es.emu.fail_fence_at = 1;
        const run_result r          = run_pipeline(nd, 2, runner, rec);
        check("D emulated GEMM fence failure inside the owner step: Main(0) fatal at fence, run and consumer fatal, "
              "nothing committed, the session disabled",
              !r.report.ok && r.report.fatal && r.consumer.fatal && r.committed_untouched && r.report.stage != nullptr &&
                  std::string(r.report.stage) == "main" && r.report.detail.find("step=fence fatal=1") != std::string::npos &&
                  runner.es.session.ws.report.fatal && rec.seq(kind::main_begin, 1) < 0);
        check_order("D event order of the run with an emulated hardware failure", rec, 2, nd.stripes(), false);
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
    const std::set<size_t> packet_ids = g.packet_stripes();
    std::string            order;
    uint64_t               first_sequences = 0;
    while (runner.es.emu.loops <= 2u * POTAL_GENERATION_MAX + 3 && runs < 64) {
        recorder         rec;
        const run_result r = run_pipeline(nd, 2, runner, rec);
        all_ok             = all_ok && r.report.ok;
        same               = same && bitwise(nd.dst(), g.want);
        if (order.empty()) {
            order = order_violation(rec, 2, nd.stripes(), true, &packet_ids);
        }
        for (const auto & e : rec.events()) {
            first_sequences += e.sequence == 1;
        }
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
    check("E every run's event order holds and numbers its steps from 1, while slot generations and the SCU "
          "generation (wrapping) keep their own counts" + (order.empty() ? std::string() : " -- " + order),
          order.empty() && first_sequences == runs);
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

// ---- O ------------------------------------------------------------------------------

// A run driven directly: the production stripe_run and consumer thread, fed
// the stripe-ready events the golden's FULL-style ExSIA output describes
// (codes, theta, packets), so that its observer can reach the run.
struct direct_result {
    pipeline_report report;
    bool            finished = false; // finish() returned true
    bool            exact    = false; // private output == golden, bitwise
    size_t          accepted = 0;     // submit() returned true
};

direct_result run_direct(const node & nd, const golden_node & g, size_t capacity, recorder & rec,
                         const std::function<void(stripe_run &, const stripe_event &)> & hook) {
    direct_result   out;
    const auto *    meta = std::get_if<exsia_ns::Meta>(&g.args.act_quant.storage());
    stripe_run *    self = nullptr;
    pipeline_config cfg;
    cfg.job_capacity = capacity;
    cfg.residual     = options::compensation;
    cfg.layer        = "pipeline-test";
    cfg.backend      = "EMULATED";
    cfg.observer     = [&](const stripe_event & ev) {
        rec.on(ev);
        if (hook && self != nullptr) {
            hook(*self, ev);
        }
    };
    gated_runner runner;
    stripe_run   run(cfg, g.args, nd.weight, nd.plan);
    self = &run;
    std::string why;
    if (meta == nullptr || meta->theta.size() < nd.stripes() || !run.begin(why)) {
        out.report.stage = "fixture";
        return out;
    }
    std::thread  consumer([&] { (void) run.consume([&](const potal_stripe_gemm_desc & d) { return runner.run(d); }); });
    const size_t rows = nd.plan.tile_I * D;
    for (size_t s = 0; s < nd.stripes(); ++s) {
        exsia_ns::StripeReadyEvent e;
        e.run_id                     = 5;
        e.stripe_id                  = s;
        e.row_begin                  = s * rows;
        e.row_end                    = std::min(nd.M, (s + 1) * rows);
        e.activation_metadata        = exsia_ns::StripeMetadataSnapshot{};
        e.activation_metadata->theta = meta->theta[s];
        const auto p                 = g.packets.find(s);
        if (p != g.packets.end()) {
            e.rmd_packet = p->second;
        }
        if (!run.submit(e, g.args)) {
            break;
        }
        out.accepted++;
    }
    out.finished = run.finish();
    consumer.join();
    out.report = run.report();
    out.exact  = out.report.ok && std::memcmp(run.output(), g.want.data(), g.want.size() * sizeof(float)) == 0;
    return out;
}

void test_observers(const node & nd, const golden_node & g) {
    const std::set<size_t> packets = g.packet_stripes();
    {
        // Every callback queries the run (report, poll_completed): no lock of
        // the run is held, nothing waits on it. Both threads call it.
        recorder            rec;
        std::atomic<size_t> polled{ 0 }, reports{ 0 };
        const direct_result r = run_direct(nd, g, 2, rec, [&](stripe_run & run, const stripe_event &) {
            reports += run.report().expected_stripes == nd.stripes() ? 1 : 0;
            polled += run.poll_completed().size();
        });
        check("O observer querying the run on every step (both threads): committed == golden, every completion "
              "polled once",
              r.finished && r.exact && polled == nd.stripes() && reports == rec.events().size());
        check_order("O event order with a querying observer", rec, 2, nd.stripes(), true, &packets);
    }
    {
        // The consumer's observer aborts the run at main_end(1).
        recorder            rec;
        const direct_result r = run_direct(nd, g, 2, rec, [&](stripe_run & run, const stripe_event & ev) {
            if (ev.kind == kind::main_end && ev.stripe_id == 1) {
                run.abort("observer", "test: abort from the observer");
            }
        });
        check("O observer aborting the run (consumer thread, main_end(1)): run fails, producer and finish() "
              "released, no later Main",
              !r.finished && !r.report.ok && r.report.stage != nullptr && std::string(r.report.stage) == "observer" &&
                  rec.seq(kind::main_begin, 2) < 0);
        check_order("O event order of the aborted run", rec, 2, nd.stripes(), false, &packets);
    }
    {
        // An observer throwing on the consumer thread fails the run (not fatal).
        recorder            rec;
        const direct_result r = run_direct(nd, g, 2, rec, [&](stripe_run &, const stripe_event & ev) {
            if (ev.kind == kind::semantic_complete && ev.stripe_id == 0) {
                throw std::runtime_error("test: observer exception");
            }
        });
        check("O observer throwing on the consumer thread: run fails (exception, not fatal), everything released",
              !r.finished && !r.report.ok && !r.report.fatal && r.report.stage != nullptr &&
                  std::string(r.report.stage) == "exception" && rec.seq(kind::capacity_release, 0) < 0);
        check_order("O event order of the run whose consumer observer threw", rec, 2, nd.stripes(), false,
                    &packets);
    }
    {
        // An observer throwing on capacity_release(0) while the producer waits
        // (capacity 1): the slot is already the producer's again, so the
        // failure names stripe 0 and counts its work once.
        recorder            rec;
        const direct_result r = run_direct(nd, g, 1, rec, [&](stripe_run &, const stripe_event & ev) {
            if (ev.kind == kind::capacity_release && ev.stripe_id == 0) {
                throw std::runtime_error("test: observer exception");
            }
        });
        check("O observer throwing on capacity_release(0): run fails (exception, not fatal) for stripe 0, its work "
              "counted once",
              !r.finished && !r.report.ok && !r.report.fatal && r.report.stage != nullptr &&
                  std::string(r.report.stage) == "exception" && r.report.detail.rfind("stripe 0: ", 0) == 0 &&
                  r.report.counts.main_logical_gemms == 1 && r.report.counts.stripes_semantic_completed == 1 &&
                  rec.seq(kind::main_begin, 1) < 0);
        check_order("O event order of the run whose capacity_release observer threw", rec, 1, nd.stripes(), false,
                    &packets);
    }
    {
        // An observer throwing on the producer thread fails its submit.
        recorder            rec;
        const direct_result r = run_direct(nd, g, 2, rec, [&](stripe_run &, const stripe_event & ev) {
            if (ev.kind == kind::submit_accepted && ev.stripe_id == 1) {
                throw std::runtime_error("test: observer exception");
            }
        });
        check("O observer throwing on the producer thread: submit fails, run fails, consumer released",
              !r.finished && !r.report.ok && r.report.stage != nullptr && std::string(r.report.stage) == "submit" &&
                  r.accepted == 1);
        check_order("O event order of the run whose producer observer threw", rec, 2, nd.stripes(), false,
                    &packets);
    }
}

// A merge that fails (a non-finite prepared channel scale makes the prepared
// merge refuse) fails the run at merge, drained, nothing committed.
void test_merge_failure(node & nd, const golden_node & g) {
    if (!options::compensation || g.packets.empty()) {
        std::printf("[SKIP] O merge failure needs RMD ON and a residual packet\n");
        return;
    }
    const size_t col   = 3;
    const float  saved = nd.weight.scale[col];
    nd.weight.scale[col] = std::numeric_limits<float>::quiet_NaN();
    gated_runner     runner;
    recorder         rec;
    const run_result r   = run_pipeline(nd, 2, runner, rec);
    nd.weight.scale[col] = saved;
    const size_t first   = *g.packet_stripes().begin();
    check("O merge failure (stripe " + std::to_string(first) + "): run fails at merge, not fatal, nothing committed",
          !r.report.ok && !r.report.fatal && r.report.stage != nullptr && std::string(r.report.stage) == "merge" &&
              r.committed_untouched && !r.consumer.fatal && rec.seq(kind::merge_begin, first) >= 0 &&
              rec.seq(kind::merge_end, first) < 0 && rec.seq(kind::semantic_complete, first) < 0 &&
              rec.seq(kind::run_failed) >= 0);
    check_order("O event order of the run failed at merge", rec, 2, nd.stripes(), false);
}

// 24 gated runs: capacity 2 and 1 alternating, nodes with and without
// packets, a forced failure of the held GEMM every sixth run. Stripe 0's last
// GEMM is held until the producer waits for capacity (stripe `capacity`).
void test_repetition(node & nd, const golden_node & g, node & calm, const golden_node & gc) {
    size_t      runs = 0, passed = 0, failed_runs = 0, waits = 0;
    std::string first_problem;
    for (int i = 0; i < 24; ++i) {
        const size_t        capacity = i % 2 == 0 ? 2 : 1;
        const bool          with     = i % 3 != 2;
        node &              n        = with ? nd : calm;
        const golden_node & gold     = with ? g : gc;
        const bool          fail     = i % 6 == 5;
        const bool          held_res = options::compensation && gold.packets.count(0) == 1;
        gated_runner        runner;
        recorder            rec;
        runner.arm(0, held_res ? POTAL_STRIPE_WORK_RESIDUAL : POTAL_STRIPE_WORK_MAIN);
        std::thread opener([&] {
            const bool waited = rec.wait_for(kind::queue_wait, capacity) && (!held_res || rec.wait_for(kind::main_end, 0));
            rec.marker(waited ? "gate_open" : "gate_open_timeout");
            runner.open(fail, false);
        });
        const run_result r = run_pipeline(n, capacity, runner, rec);
        opener.join();
        const std::set<size_t> ids     = gold.packet_stripes();
        const long             gate    = rec.seq_marker("gate_open");
        std::string            problem = order_violation(rec, capacity, n.stripes(), !fail, &ids);
        if (problem.empty() && gate < 0) {
            problem = "the producer never waited for capacity";
        }
        for (size_t s = 1; problem.empty() && s < capacity; ++s) {
            const long acc = rec.seq(kind::submit_accepted, s);
            if (acc < 0 || acc > gate) {
                problem = "stripe " + std::to_string(s) + " not accepted while stripe 0 was held";
            }
        }
        const long wait = rec.seq(kind::queue_wait, capacity);
        if (problem.empty() && (wait < 0 || wait > gate)) {
            problem = "no queue_wait of stripe " + std::to_string(capacity) + " before the gate opened";
        }
        if (!fail) {
            const long rel0 = rec.seq(kind::capacity_release, 0);
            if (problem.empty() && !(r.report.ok && bitwise(n.dst(), gold.want))) {
                problem = "result differs from the golden";
            } else if (problem.empty() && !(rel0 > gate && rec.seq(kind::submit_accepted, capacity) > rel0)) {
                problem = "stripe " + std::to_string(capacity) + " accepted before stripe 0 released its capacity";
            }
        } else {
            failed_runs++;
            if (problem.empty() && !(!r.report.ok && !r.report.fatal && r.committed_untouched &&
                                     rec.seq(kind::submit_accepted, capacity) < 0 && rec.seq(kind::run_failed) >= 0)) {
                problem = "forced failure not contained (waiter released, nothing committed)";
            }
        }
        waits += r.report.counts.producer_capacity_waits;
        runs++;
        passed += problem.empty() ? 1 : 0;
        if (!problem.empty() && first_problem.empty()) {
            first_problem = "run " + std::to_string(i) + " (capacity " + std::to_string(capacity) +
                            (with ? ", packets" : ", no packets") + (fail ? ", failure" : "") + "): " + problem;
        }
    }
    check("R " + std::to_string(runs) + " gated runs (capacity 2/1, with and without packets, " +
              std::to_string(failed_runs) + " forced failures, " + std::to_string(waits) +
              " capacity waits): event order, overlap, backpressure, golden or contained failure" +
              (first_problem.empty() ? std::string() : " -- " + first_problem),
          runs == 24 && passed == runs && failed_runs == 4);
}

// The debug JSONL of this process (truncated when the logger opened it): the
// [POTAL_STRIPE] records of every run in sequence order 1, 2, ... without gaps.
void test_jsonl_order() {
#if LOG_DEBUG
    std::fflush(nullptr);
    const char *      dir  = std::getenv("GEMMINI_LOG_DIR");
    const std::string path = dir != nullptr && *dir != '\0' ? std::string(dir) + "/debug-log.jsonl"
                                                            : std::string("output/log/debug-log.jsonl");
    std::ifstream     in(path);
    std::string       line, bad;
    size_t            records = 0, runs = 0;
    uint64_t          previous = 0;
    while (in && std::getline(in, line)) {
        if (line.find("[POTAL_STRIPE] event=") == std::string::npos) {
            continue; // e.g. a producer-affinity failure record
        }
        const size_t at = line.find("event_sequence=");
        if (at == std::string::npos) {
            bad = " -- a record without event_sequence";
            break;
        }
        const uint64_t sequence = std::strtoull(line.c_str() + at + std::strlen("event_sequence="), nullptr, 10);
        if (sequence == 1) {
            runs++;
        } else if (sequence != previous + 1) {
            bad = " -- event_sequence " + std::to_string(sequence) + " written after " + std::to_string(previous);
            break;
        }
        previous = sequence;
        records++;
    }
    check("O debug JSONL (" + path + "): " + std::to_string(records) + " [POTAL_STRIPE] records of " +
              std::to_string(runs) + " runs, each run written in sequence order without gaps" + bad,
          bad.empty() && records > 0 && runs > 0);
#else
    std::printf("[SKIP] O debug JSONL order: built with LOG_DEBUG=0 (no records)\n");
#endif
}

} // namespace

int main() {
    const char * log_dir = std::getenv("GEMMINI_LOG_DIR");
    if (log_dir == nullptr || *log_dir == '\0') {
        const char *      tmp  = std::getenv("TMPDIR");
        const std::string base = std::string(tmp && *tmp ? tmp : "/tmp") + "/potal-stripe-pipeline-test-XXXXXX";
        std::vector<char> dir(base.begin(), base.end());
        dir.push_back('\0');
        if (mkdtemp(dir.data()) != nullptr) {
            setenv("GEMMINI_LOG_DIR", dir.data(), 1);
        }
    }
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
        node        nd(200, 320, 128, 7, 400.0f); // DIM16: 3 stripes; DIM32: 4 stripes.
        golden_node g(nd);
        const size_t stripe_rows = D == 32 ? 64 : 80;
        const size_t stripe_count = D == 32 ? 4 : 3;
        const bool geometry = nd.plan.tile_I == (D == 32 ? 2u : 5u) &&
                              nd.plan.tile_J == (D == 32 ? 4u : 6u) &&
                              nd.plan.tile_K == (D == 32 ? 4u : 8u) &&
                              nd.plan.tile_I * D == stripe_rows && nd.stripes() == stripe_count;
        check("fixture: >= 3 stripes from the frozen plan, golden built" +
                  std::string(options::compensation ? ", stripe 0 has a residual packet" : ""),
              nd.ok && g.ok && geometry && (!options::compensation || g.packets.count(0) == 1));
        if (nd.ok && g.ok && geometry && (!options::compensation || g.packets.count(0) == 1)) {
            test_overlap(nd, g);
            test_serialized_vs_pipelined(nd, g);
            test_failures(nd);
            test_contract(nd);
            test_repeated(nd, g);
        }
        // Main(0) stays held until quant_end. Exactly two stripes must fit
        // capacity 2; a third stripe would block the producer before quant_end.
        node        two(D == 32 ? 112 : 140, 320, 128, 9, 400.0f);
        golden_node g2(two);
        if (two.ok && g2.ok && two.stripes() == 2 && two.plan.tile_I * D == stripe_rows) {
            test_immutability(g2, two);
        } else {
            check("C fixture", false);
        }
        if (nd.ok && g.ok && geometry) {
            test_observers(nd, g);
            test_merge_failure(nd, g);
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
        const std::set<size_t> none;
        check_order("empty residual: event order without residual steps", rec, 2, calm.stripes(), true, &none);
        if (nd.ok && g.ok && geometry && calm.ok && gc.ok && calm.stripes() == stripe_count &&
            (!options::compensation || g.packets.count(0) == 1)) {
            test_repetition(nd, g, calm, gc);
        }
        test_jsonl_order();
    }
    if (failures != 0) {
        std::printf("POTAL_STRIPE_PIPELINE_TEST FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("POTAL_STRIPE_PIPELINE_TEST PASS\n");
    return 0;
}
