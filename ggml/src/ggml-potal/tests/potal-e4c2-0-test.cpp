#include "potal.h"
#include "support/e4b2-emulator.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
uint64_t assertions = 0, scenarios = 0, expected_refusals = 0, ownership_violations = 0;
void require(bool value, const char *message) {
    ++assertions;
    if (!value) throw std::runtime_error(message);
}

enum class phase { free, prefetch_issued, prefetch_complete, resident, readers_in_flight, retire_safe, quarantined };
enum class status { ok, refused, stalled, missing_observable, issue_error, uncertain, cleanup_error };
struct reference {
    uint64_t owner = 0, generation = 0;
    unsigned pair = 0;
};
bool same(reference a, reference b) {
    return a.owner && a.generation && a.owner == b.owner &&
        a.generation == b.generation && a.pair == b.pair;
}
struct lease { uint64_t next = 1; bool held = false; };
using bytes = std::array<unsigned char, 64>;

// Deterministic synthetic hardware events, not a timing simulator or FPGA
// signal reader. A generator finishing is intentionally earlier than EX/ACC
// completion. DMA finishes only when the test explicitly delivers that event.
struct event_world {
    std::array<bool, 2> dma_active{};
    std::array<reference, 2> dma_identity{};
    std::array<const bytes *, 2> sources{};
    std::array<bytes, 2> b_contents{};
    bool generator = false, ex_active = false, store_active = false;
    reference ex_identity{};
    unsigned ex_path = 0;
    uint64_t ex_sequence = 0;
    bool drained() const {
        return !generator && !ex_active && !store_active && !dma_active[0] && !dma_active[1];
    }
    void dma_finish(unsigned p) {
        require(p < 2 && dma_active[p] && sources[p], "only an admitted DMA can finish");
        b_contents[p] = *sources[p];
        dma_active[p] = false;
    }
    void compute_finish() {
        require(ex_active && !generator, "ACC completion follows generator emission");
        ex_active = false;
    }
};

// A sample represents a PROPOSED checked, ordered, live prefix query. The
// existing resident C11 API exposes no such call. No per-pair hardware tag is
// invented: zero covers the entire sealed load/EX prefix, not one bank.
struct prefix_sample {
    uint64_t owner = 0, cut = 0;
    unsigned loads = 0, executes = 0;
    bool live = true, external = true, ordered = true;
};
struct pair_state {
    phase state = phase::free;
    reference identity{};
    uint64_t weight = 0;
    unsigned admitted = 0, issued_readers = 0, completed_readers = 0;
    std::array<bool, 2> scale_live{};
    std::array<bool, 2> reader_started{};
    bool source_held = false, issue_complete = false;
};
struct epoch {
    uint64_t weight = 0, next = 0, total = 0;
    bool writer = false, readback_pending = false, receipt_ready = false;
};

// Test-local ownership reducer. Its positive concurrent cases describe a
// future checked interface, not the current B2 API. Only global drain is an
// existing public completion boundary. One representative Scale column per
// reader is modeled; there is no batch-Release opcode or numerical GEMM here.
struct protocol {
    lease &owner;
    event_world &hw;
    const bool prefix_queries;
    bool owns = false, barrier = false, cleanup_attempted = false, cleaned = false;
    bool quarantined = false;
    status first_error = status::ok, cleanup_status = status::ok;
    uint64_t identity = 0, cut = 0;
    int release_pair = -1;
    unsigned release_path = 0;
    std::array<uint64_t, 2> generations{};
    std::array<pair_state, 2> pairs{};
    std::array<epoch, 2> acc{};
    std::array<bytes, 2> source{};
    reference compute_identity{};
    unsigned compute_path = 0;

    protocol(lease &l, event_world &w, bool queries) : owner(l), hw(w), prefix_queries(queries) {
        if (!l.held && l.next != std::numeric_limits<uint64_t>::max()) {
            l.held = owns = true;
            identity = l.next++;
        }
    }
    bool ready() const {
        return owns && first_error == status::ok && !barrier && !cleaned && !quarantined;
    }
    pair_state *find(reference r) {
        if (r.pair >= 2 || !same(r, pairs[r.pair].identity)) return nullptr;
        return &pairs[r.pair];
    }
    void poison(status error) {
        if (first_error == status::ok) first_error = error;
        quarantined = true;
        for (auto &p : pairs) if (p.state != phase::free) p.state = phase::quarantined;
    }
    status fill_source(unsigned p, unsigned char value) {
        if (!ready() || p >= 2 || pairs[p].source_held) return status::refused;
        source[p].fill(value);
        return status::ok;
    }
    status begin_prefetch(unsigned p, uint64_t weight, reference &r) {
        if (!ready() || p >= 2 || !weight || pairs[p].state != phase::free ||
            generations[p] == std::numeric_limits<uint64_t>::max()) return status::refused;
        r = {identity, ++generations[p], p};
        pairs[p] = {};
        pairs[p].identity = r;
        pairs[p].weight = weight;
        pairs[p].state = phase::prefetch_issued;
        pairs[p].source_held = true; // Before any possible command effect.
        hw.dma_identity[p] = r;
        hw.sources[p] = &source[p];
        ++cut;
        return status::ok;
    }
    status admit_tile(reference r) {
        auto *p = find(r);
        if (!ready() || !p || p->state != phase::prefetch_issued || p->admitted == 2)
            return status::refused;
        if (hw.generator) return status::stalled;
        ++p->admitted;
        p->issue_complete = p->admitted == 2;
        hw.dma_active[r.pair] = true;
        ++cut;
        return status::ok;
    }
    status admission_error(reference r) {
        auto *p = find(r);
        if (!ready() || !p || p->state != phase::prefetch_issued) return status::refused;
        first_error = status::issue_error;
        // Even a first-command error can have effects. Retain source/lease.
        return first_error;
    }
    status uncertain_dma(reference r) {
        if (!find(r) || !owns) return status::refused;
        poison(status::uncertain);
        return first_error;
    }
    prefix_sample sample() const {
        return {identity, cut, unsigned(hw.dma_active[0]) + unsigned(hw.dma_active[1]),
                unsigned(hw.ex_active), true, true, !hw.generator};
    }
    bool valid_sample(const prefix_sample &s) const {
        return s.owner == identity && s.cut == cut && s.live && s.external && s.ordered &&
               !hw.generator;
    }
    void complete_loads() {
        // Called only after a qualified prefix/global certificate. Hardware
        // identities/busy bits belong to the independent oracle, not this transition.
        for (unsigned p = 0; p < 2; ++p) {
            auto &s = pairs[p];
            if (s.state == phase::prefetch_issued && s.issue_complete) {
                s.state = phase::prefetch_complete;
                s.source_held = false;
            }
        }
    }
    status observe_loads(const prefix_sample &s) {
        if (!ready()) return status::refused;
        if (!prefix_queries) return status::missing_observable;
        if (!valid_sample(s) || s.loads) return status::refused;
        for (const auto &p : pairs)
            if (p.state == phase::prefetch_issued && !p.issue_complete) return status::refused;
        complete_loads();
        return status::ok;
    }
    status publish(reference r) {
        auto *p = find(r);
        if (!ready() || !p || p->state != phase::prefetch_complete) return status::refused;
        p->state = phase::resident;
        return status::ok;
    }
    status epoch_begin(unsigned path, uint64_t weight, uint64_t total) {
        if (!ready() || path >= 2 || !weight || !total || acc[path].writer ||
            acc[path].readback_pending || (acc[path].total && acc[path].next != acc[path].total))
            return status::refused;
        acc[path] = {weight, 0, total, false, false, false};
        return status::ok;
    }
    status reader_start(reference r, unsigned path, uint64_t sequence) {
        auto *p = find(r);
        if (!ready() || !p || path >= 2 || p->state != phase::resident ||
            p->issued_readers >= 2 || p->reader_started[path] || p->scale_live[path] || hw.ex_active ||
            acc[0].writer || acc[1].writer || acc[path].readback_pending ||
            acc[path].weight != p->weight || acc[path].next != sequence ||
            sequence >= acc[path].total) return status::refused;
        ++p->issued_readers;
        p->reader_started[path] = true;
        p->scale_live[path] = true;
        p->state = phase::readers_in_flight;
        acc[path].writer = true;
        acc[path].receipt_ready = false;
        compute_identity = r;
        compute_path = path;
        hw.ex_identity = r;
        hw.ex_path = path;
        hw.ex_sequence = sequence;
        hw.ex_active = hw.generator = true;
        ++cut;
        return status::ok;
    }
    void complete_compute() {
        auto *p = find(compute_identity);
        auto &e = acc[compute_path];
        if (!p || !e.writer) return;
        ++p->completed_readers;
        p->state = phase::resident; // Scale and future readers still own B.
        ++e.next;
        e.writer = false;
        compute_identity = {};
    }
    status observe_compute(const prefix_sample &s) {
        if (!ready()) return status::refused;
        if (!prefix_queries) return status::missing_observable;
        if (!valid_sample(s) || s.executes) return status::refused;
        complete_compute();
        return status::ok;
    }
    status begin_barrier(int p = -1, unsigned path = 0) {
        if (!ready() || p < -1 || p > 1 || path >= 2 ||
            (p >= 0 && !pairs[unsigned(p)].scale_live[path])) return status::refused;
        for (const auto &s : pairs)
            if (s.state == phase::prefetch_issued && !s.issue_complete) return status::refused;
        barrier = true;
        release_pair = p;
        release_path = path;
        return hw.drained() ? status::ok : status::stalled;
    }
    status accept_barrier() {
        if (!owns || !barrier || first_error != status::ok) return status::refused;
        if (!hw.drained()) return status::stalled;
        complete_loads(); // A global certificate is stronger, but not independent.
        complete_compute();
        for (auto &e : acc) if (e.readback_pending) {
            e.readback_pending = false;
            e.receipt_ready = true;
        }
        if (release_pair >= 0) {
            auto &p = pairs[unsigned(release_pair)];
            p.scale_live[release_path] = false;
            if (p.issued_readers == 2 && p.completed_readers == 2 &&
                !p.scale_live[0] && !p.scale_live[1]) p.state = phase::retire_safe;
        }
        barrier = false;
        return status::ok;
    }
    status retire(reference r) {
        auto *p = find(r);
        if (!ready() || !p || p->state != phase::retire_safe ||
            p->source_held || p->scale_live[0] || p->scale_live[1]) return status::refused;
        *p = {}; // ACC epochs and generation counters deliberately survive.
        return status::ok;
    }
    status readback_start(unsigned path) {
        if (!ready() || !hw.drained() || path >= 2 || !acc[path].total ||
            acc[0].writer || acc[1].writer ||
            acc[path].next != acc[path].total || acc[path].readback_pending || hw.store_active)
            return status::refused;
        acc[path].readback_pending = true;
        acc[path].receipt_ready = false;
        hw.store_active = true;
        ++cut;
        return status::ok;
    }
    status cleanup_begin() {
        if (!owns || first_error == status::ok || quarantined || cleanup_attempted)
            return status::refused;
        cleanup_attempted = true;
        return status::ok;
    }
    status cleanup_finish(bool success) {
        if (!cleanup_attempted || quarantined || cleaned) return status::refused;
        if (!success) {
            cleanup_status = status::cleanup_error;
            poison(status::cleanup_error);
            return first_error;
        }
        if (!hw.drained()) return status::stalled;
        pairs = {};
        acc = {};
        cleaned = true; // Failed work never publishes, despite known completion.
        return first_error;
    }
    status close() {
        if (!owns || quarantined || barrier || (!cleaned && first_error != status::ok))
            return status::refused;
        for (const auto &p : pairs) if (p.state != phase::free) return status::refused;
        for (const auto &e : acc)
            if (e.writer || e.readback_pending || (e.total && e.next != e.total)) return status::refused;
        owner.held = owns = false;
        return first_error;
    }
};

void accepted(status s, const char *m) { require(s == status::ok, m); }
void refused(status s, const char *m) {
    require(s == status::refused, m);
    ++expected_refusals;
}
void owns(bool value, const char *m) {
    if (!value) ++ownership_violations;
    require(value, m);
}
reference loaded(protocol &p, unsigned pair, uint64_t weight) {
    reference r;
    accepted(p.begin_prefetch(pair, weight, r), "reserve immutable B identity");
    accepted(p.admit_tile(r), "first B tile admission");
    accepted(p.admit_tile(r), "last B tile admission");
    p.hw.dma_finish(pair);
    accepted(p.begin_barrier(), "initial load global completion is available");
    accepted(p.accept_barrier(), "initial B completion certificate");
    accepted(p.publish(r), "initial resident publication");
    return r;
}
void consume(protocol &p, reference r, unsigned path, uint64_t sequence) {
    accepted(p.reader_start(r, path, sequence), "ordered Main/Indexed Residual reader");
    p.hw.generator = false;
    p.hw.compute_finish();
    accepted(p.begin_barrier(int(r.pair), path), "Release after physical completion");
    accepted(p.accept_barrier(), "completion-qualified Scale column release");
}
void check_ownership(const protocol &p) {
    owns(p.owner.held == p.owns, "single issuer lease is retained");
    owns(!p.hw.ex_active ||
             (same(p.compute_identity, p.hw.ex_identity) && p.compute_path == p.hw.ex_path &&
              p.acc[p.hw.ex_path].writer && p.acc[p.hw.ex_path].next == p.hw.ex_sequence),
         "active hardware reader retains its exact software B and ACC ticket");
    for (unsigned n = 0; n < 2; ++n) {
        const auto &s = p.pairs[n];
        owns(!p.hw.dma_active[n] || s.source_held, "pending DMA source remains owned");
        owns(s.state == phase::free || s.state == phase::prefetch_issued ||
                 s.state == phase::quarantined ||
                 (!p.hw.dma_active[n] && same(s.identity, p.hw.dma_identity[n])),
             "published software B agrees with the independent completion oracle");
        owns(s.state == phase::free || s.identity.pair == n, "one physical B pair per identity");
        owns(s.state != phase::retire_safe ||
                (s.completed_readers == 2 && !s.source_held &&
                 !s.scale_live[0] && !s.scale_live[1]), "retirement requires all resource lifetimes");
        owns(s.completed_readers <= s.issued_readers, "completion never precedes reader issue");
    }
}

void ordering(bool dma_first, bool independent) {
    lease owner;
    event_world hw;
    protocol p(owner, hw, independent);
    accepted(p.fill_source(0, 17), "owned source0 fill");
    const auto r0 = loaded(p, 0, 71);
    for (unsigned path = 0; path < 2; ++path)
        accepted(p.epoch_begin(path, 71, 2), "independent ACC0/ACC1 epochs");
    const auto old_zero = p.sample();
    accepted(p.reader_start(r0, 0, 0), "active Main reader");
    reference overwrite;
    refused(p.begin_prefetch(0, 72, overwrite), "active physical B cannot be overwritten");
    accepted(p.fill_source(1, 29), "inactive source fill while peer computes");
    reference r1;
    accepted(p.begin_prefetch(1, 71, r1), "reserve next pair during active compute");
    require(p.admit_tile(r1) == status::stalled && !hw.dma_active[1],
            "configured LOOP generator blocks raw MVIN dispatch");
    hw.generator = false;
    if (independent)
        refused(p.observe_loads(p.sample()), "zero before descriptor admission is not completion");
    accepted(p.admit_tile(r1), "raw load admits after generator emission");
    if (independent)
        refused(p.observe_loads(p.sample()), "partial descriptor cannot publish from zero/partial count");
    else
        require(p.observe_loads(p.sample()) == status::missing_observable,
                "no independent query exists even during partial admission");
    accepted(p.admit_tile(r1), "complete next B descriptor admission");
    if (independent) {
        refused(p.observe_loads(old_zero), "old zero cannot complete a newly issued load prefix");
        refused(p.observe_compute(old_zero), "old zero cannot complete a newly issued EX prefix");
    }
    const auto stale = p.sample();
    refused(p.publish(r1), "command admission is not resident publication");
    refused(p.reader_start(r1, 0, 0), "premature reader start");
    refused(p.fill_source(1, 93), "pending source reuse is forbidden");
    refused(p.retire(r0), "active B cannot retire");
    auto wrong = r0; ++wrong.generation;
    refused(p.reader_start(wrong, 1, 0), "wrong resident generation");

    // Merely delaying events cannot change software ownership.
    for (unsigned i = 0; i < 9; ++i) {
        check_ownership(p);
        require(p.pairs[1].state == phase::prefetch_issued && p.acc[0].next == 0,
                "delayed completion never advances software progress");
    }
    if (dma_first) hw.dma_finish(1); else hw.compute_finish();
    if (independent) {
        auto snap = p.sample(); snap.live = false;
        refused(p.observe_loads(snap), "snapshot zero is not a fresh completion certificate");
        auto config = p.sample(); config.external = false;
        refused(p.observe_loads(config), "disabled or local counter is not a load certificate");
        auto unsealed = p.sample(); ++unsealed.cut;
        refused(p.observe_loads(unsealed), "old or foreign issued prefix is refused");
        auto foreign = p.sample(); ++foreign.owner;
        refused(dma_first ? p.observe_loads(foreign) : p.observe_compute(foreign),
                "foreign owner zero is rejected when the local certificate would be ready");
        if (dma_first) {
            accepted(p.observe_loads(p.sample()), "load-prefix zero is independent of EX completion");
            accepted(p.publish(r1), "next B publishes while old EX remains pending");
            require(hw.ex_active && p.acc[0].writer, "publication does not complete ACC");
            accepted(p.fill_source(1, 93), "source may be reused only after observed DMA completion");
            require(hw.b_contents[1][0] == 29, "source poison does not change resident B");
        } else {
            accepted(p.observe_compute(p.sample()), "EX-prefix zero does not imply DMA completion");
            require(p.acc[0].next == 1 && p.pairs[1].source_held,
                    "ACC and B DMA lifetimes are independent");
        }
    } else {
        require(p.observe_loads(p.sample()) == status::missing_observable,
                "existing public interface has no independent completion query");
        refused(p.fill_source(1, 93), "physical completion alone does not release an unobserved source");
    }

    require(p.begin_barrier(0, 0) == status::stalled,
            "global Release joins remaining next DMA or current computation");
    refused(p.publish(r1), "blocked Release cannot be bypassed by later CPU operations");
    require(p.accept_barrier() == status::stalled, "early Release acceptance is impossible");
    if (dma_first) hw.compute_finish(); else hw.dma_finish(1);
    accepted(p.accept_barrier(), "Release waits for both classes to complete");
    require(p.acc[0].next == 1 && !p.pairs[1].source_held,
            "global drain reconciles both pending ownership classes");
    if (p.pairs[1].state == phase::prefetch_complete) accepted(p.publish(r1), "post-join B publication");
    if (independent)
        refused(p.observe_loads(stale), "stale nonzero sample cannot certify the later state");
    else
        require(p.observe_loads(stale) == status::missing_observable,
                "global completion does not add an independent query");
    refused(p.retire(r0), "Main completion cannot retire a B with a future Residual reader");
    refused(p.reader_start(wrong, 1, 0), "stale generation rejected when a correct reader would be ready");
    consume(p, r0, 1, 0);
    accepted(p.retire(r0), "completed old B retirement");
    require(p.acc[0].next == 1 && p.acc[1].next == 1, "B retirement preserves ACC epochs");
    consume(p, r1, 0, 1);
    consume(p, r1, 1, 1);
    accepted(p.retire(r1), "second B retires independently of final ACC readback");
    for (unsigned path = 0; path < 2; ++path) {
        accepted(p.readback_start(path), "final ACC readback admission");
        require(!p.acc[path].receipt_ready, "MVOUT admission is not host result completion");
        require(p.begin_barrier() == status::stalled, "readback stores block Fence");
        hw.store_active = false; // Synthetic acknowledged host-store completion.
        accepted(p.accept_barrier(), "Fence qualifies readback receipt");
        require(p.acc[path].receipt_ready, "result publishes only after acknowledged stores");
    }
    check_ownership(p);
    accepted(p.close(), "healthy completed close");
    ++scenarios;
}

void reuse_and_failures() {
    lease owner;
    event_world hw;
    protocol p(owner, hw, true);
    std::array<reference, 2> stale{};
    for (unsigned cycle = 0; cycle < 12; ++cycle) {
        const unsigned pair = cycle % 2;
        const auto r = loaded(p, pair, 71);
        require(!same(r, stale[pair]), "resident incarnations never alias");
        for (unsigned path = 0; path < 2; ++path) {
            accepted(p.epoch_begin(path, 71, 1), "new complete output epoch");
            if (path == 0 && stale[pair].owner)
                refused(p.reader_start(stale[pair], path, 0), "old generation cannot read a ready reused pair");
            consume(p, r, path, 0);
        }
        refused(p.publish(stale[pair]), "retired generation cannot republish");
        accepted(p.retire(r), "repeated pair retirement");
        stale[pair] = r;
        check_ownership(p);
    }
    accepted(p.close(), "repeated reuse safe close");
    ++scenarios;
    for (unsigned admitted = 0; admitted < 2; ++admitted) {
        lease l;
        event_world w;
        protocol f(l, w, true);
        reference r;
        accepted(f.begin_prefetch(1, 71, r), "premark failed admission ownership");
        if (admitted) accepted(f.admit_tile(r), "partial MVIN effects");
        require(f.admission_error(r) == status::issue_error && f.pairs[1].source_held,
                "first admission error retains possible DMA source");
        refused(f.publish(r), "failed residency is never published");
        accepted(f.cleanup_begin(), "one recoverable cleanup attempt");
        if (admitted) {
            require(f.cleanup_finish(true) == status::stalled, "cleanup cannot reclaim active DMA");
            w.dma_finish(1);
        }
        require(f.cleanup_finish(true) == status::issue_error && f.cleaned &&
                f.pairs[1].state == phase::free, "successful drain cleans without hiding first error");
        require(f.close() == status::issue_error && !l.held, "cleaned close returns sticky first error");
        ++scenarios;
    }
    for (bool uncertain : {false, true}) {
        lease l;
        event_world w;
        protocol f(l, w, true);
        reference r;
        accepted(f.begin_prefetch(1, 71, r), "quarantine reservation");
        accepted(f.admit_tile(r), "possibly active DMA");
        if (uncertain) {
            require(f.uncertain_dma(r) == status::uncertain, "uncertain completion quarantines");
            refused(f.cleanup_begin(), "unknown completion does not retry a Fence");
        } else {
            require(f.admission_error(r) == status::issue_error, "recoverable issue failure");
            accepted(f.cleanup_begin(), "single cleanup");
            require(f.cleanup_finish(false) == status::issue_error &&
                    f.cleanup_status == status::cleanup_error, "cleanup failure preserves first error");
            refused(f.cleanup_begin(), "failed cleanup is never retried");
        }
        require(f.quarantined && f.pairs[1].state == phase::quarantined &&
                f.pairs[1].source_held && l.held, "quarantine retains pair, buffer and issuer");
        refused(f.close(), "quarantined close cannot free ownership");
        refused(f.fill_source(1, 0), "quarantined source cannot be reused");
        protocol other(l, w, true);
        require(!other.owns, "quarantine excludes a second issuer");
        w.dma_finish(1);
        refused(f.observe_loads(f.sample()), "late apparent completion cannot heal quarantine");
        check_ownership(f);
        ++scenarios;
    }
}

void aggregate_prefix_scope() {
    lease l;
    event_world hw;
    protocol p(l, hw, true);
    for (unsigned path = 0; path < 2; ++path)
        accepted(p.epoch_begin(path, 71, 2), "sealed prefix ACC declaration");
    std::array<reference, 2> r;
    for (unsigned n = 0; n < 2; ++n) {
        accepted(p.fill_source(n, static_cast<unsigned char>(41 + n)), "independent source fill");
        accepted(p.begin_prefetch(n, 71, r[n]), "reserve both load prefixes");
        accepted(p.admit_tile(r[n]), "first prefix tile");
        accepted(p.admit_tile(r[n]), "second prefix tile");
    }
    refused(p.reader_start(r[1], 0, 0), "premature reader with otherwise valid ACC has no resident B");
    hw.dma_finish(1);
    refused(p.observe_loads(p.sample()), "peer load keeps aggregate count nonzero");
    require(p.pairs[1].source_held, "aggregate observation is not an independent pair ticket");
    hw.dma_finish(0);
    accepted(p.observe_loads(p.sample()), "zero certifies the entire sealed load prefix");
    for (unsigned n = 0; n < 2; ++n) {
        accepted(p.publish(r[n]), "prefix-certified resident");
        for (unsigned path = 0; path < 2; ++path) consume(p, r[n], path, n);
        accepted(p.retire(r[n]), "prefix scope retirement");
    }
    check_ownership(p);
    accepted(p.close(), "aggregate prefix close");
    ++scenarios;
}

void active_peer_quarantine() {
    lease l;
    event_world hw;
    protocol p(l, hw, true);
    const auto r0 = loaded(p, 0, 71);
    accepted(p.epoch_begin(0, 71, 1), "peer ACC epoch");
    accepted(p.reader_start(r0, 0, 0), "peer compute remains in flight");
    hw.generator = false;
    reference r1;
    accepted(p.begin_prefetch(1, 71, r1), "peer prefetch ownership");
    accepted(p.admit_tile(r1), "uncertain prefix can have partial effects");
    require(p.uncertain_dma(r1) == status::uncertain, "uncertainty poisons shared issuer");
    require(p.pairs[0].state == phase::quarantined && p.pairs[1].state == phase::quarantined &&
            p.acc[0].writer && p.pairs[1].source_held, "active B, ACC and next DMA stay owned");
    hw.dma_finish(1);
    hw.compute_finish();
    refused(p.observe_compute(p.sample()), "late peer compute completion cannot heal poison");
    refused(p.observe_loads(p.sample()), "late peer DMA completion cannot heal poison");
    refused(p.retire(r0), "unknown shared completion cannot retire the active peer");
    refused(p.close(), "unknown shared completion cannot free pools");
    check_ownership(p);
    ++scenarios;
}

struct current_transport {
    e4b2::machine device;
    static potal_status issue(void *ctx, const potal_paired_cmd_record *r) {
        return static_cast<current_transport *>(ctx)->device.command(*r);
    }
    static potal_status counter(void *ctx, const potal_paired_cmd_record *r, uint64_t *v) {
        return static_cast<current_transport *>(ctx)->device.counter(r->rs1, r->rs2, v);
    }
    static void destroy(void *ctx) { delete static_cast<current_transport *>(ctx); }
};

void current_c11_gate() {
    auto *transport = new current_transport;
    const potal_resident_ws_options options{
        {current_transport::issue, transport}, current_transport::destroy, current_transport::counter};
    potal_paired_session *s = nullptr;
    require(potal_paired_resident_open(&options, &s) == POTAL_OK &&
            potal_paired_resident_query(s).software_only, "current C11 uses explicit software transport");
    require(potal_paired_resident_prepare(s) == POTAL_OK, "current C11 prepare");
    void *view = nullptr;
    size_t bytes_count = 0;
    for (unsigned pool = 0; pool < 3; ++pool) {
        require(potal_paired_resident_buffer_view(s, static_cast<potal_resident_ws_buffer>(pool),
                    &view, &bytes_count) == POTAL_OK, "current owned buffer");
        std::fill_n(static_cast<int8_t *>(view), bytes_count, int8_t(1));
    }
    const potal_resident_ws_load b0{0,16,16,1,0,0,71,9,16}, b1{1,16,16,1,16,0,71,9,16};
    require(potal_paired_resident_load_issue(s, &b0) == POTAL_OK, "current B0 admission");
    const auto commands = potal_paired_resident_query(s).core.commands;
    require(potal_paired_resident_load_issue(s, &b1) == POTAL_ERR_BUSY &&
            potal_paired_resident_query(s).core.commands == commands,
            "current single pending-kind rejects peer DMA before commands");
    potal_resident_ws_ref resident{};
    require(potal_paired_resident_get(s, 0, &resident) == POTAL_ERR_BUSY,
            "current B admission never implies completion");
    transport->device.drain(); // Internal SOFTWARE effects; not a public completion event.
    require(potal_paired_resident_query(s).pending &&
            potal_paired_resident_get(s, 0, &resident) == POTAL_ERR_BUSY,
            "internal completion is not visible through current bookkeeping query");
    require(potal_paired_fence(s) == POTAL_OK &&
            potal_paired_resident_get(s, 0, &resident) == POTAL_OK, "current global publication");
    potal_resident_ws_epoch e{};
    e.acc_tile = 1; e.rows = e.columns = 16;
    e.output_id = 99; e.weight_id = 71; e.weight_generation = 9; e.logical_k_end = 16;
    potal_resident_ws_ref epoch_ref{}, scale_ref{};
    require(potal_paired_resident_epoch_begin(s, &e, &epoch_ref) == POTAL_OK, "current epoch");
    const uint64_t blocks[] = {0};
    const uint32_t carriers[16] = {};
    const potal_resident_ws_scales scales{resident,0,1,1,16,0,blocks,carriers,16};
    require(potal_paired_resident_scale_program(s, &scales, &scale_ref) == POTAL_OK, "current Scale");
    potal_resident_ws_run run{};
    run.resident = resident; run.scale = scale_ref; run.epochs[0] = epoch_ref;
    run.k_rows = run.j_columns = run.i_rows = 16; run.a_stride = 16;
    run.scale_generation = 1; run.acc_tile = 1; run.output_id = 99; run.first = run.last = 1;
    require(potal_paired_resident_issue(s, &run) == POTAL_OK, "current Main issue");
    const auto compute_commands = potal_paired_resident_query(s).core.commands;
    require(potal_paired_resident_load_issue(s, &b1) == POTAL_ERR_BUSY &&
            potal_paired_resident_retire(s, resident) == POTAL_ERR_BUSY &&
            potal_paired_resident_query(s).core.commands == compute_commands,
            "current compute pending blocks peer prefetch and retirement");
    require(potal_paired_fence(s) == POTAL_OK, "current compute global completion");
    for (unsigned col = 0; col < 16; ++col)
        require(potal_paired_scale_release(s,0,1,col) == POTAL_OK, "preserved global Release");
    require(potal_paired_resident_retire(s, resident) == POTAL_OK, "B retirement keeps ACC epoch");
    potal_resident_ws_readback readback{};
    readback.acc_tile = 1; readback.rows = readback.columns = readback.stride = 16;
    readback.epochs[0] = epoch_ref; readback.sequences[0] = 1;
    potal_resident_ws_ref receipt{};
    require(potal_paired_resident_readback_issue(s, &readback, &receipt) == POTAL_OK,
            "ACC readback does not require retained B residency");
    const int32_t *data = nullptr;
    size_t stride = 0;
    require(potal_paired_resident_result(s, receipt, &data, &stride) == POTAL_ERR_BUSY &&
            potal_paired_resident_load_issue(s, &b1) == POTAL_ERR_BUSY,
            "current readback pending is separate and also globally excludes DMA");
    require(potal_paired_fence(s) == POTAL_OK &&
            potal_paired_resident_result(s, receipt, &data, &stride) == POTAL_OK,
            "readback receipt needs acknowledged global drain");
    for (unsigned row = 0; row < 16; ++row)
        for (unsigned col = 0; col < 16; ++col)
            require(data[row * stride + col] == 16, "independent all-ones Main INT32 witness");
    require(potal_paired_resident_load_issue(s, &b1) == POTAL_OK &&
            potal_paired_fence(s) == POTAL_OK &&
            potal_paired_resident_get(s,1,&resident) == POTAL_OK &&
            potal_paired_resident_retire(s,resident) == POTAL_OK &&
            potal_paired_resident_close(&s) == POTAL_OK && !s,
            "current sequential path remains safe and consumes the software context");
    ++scenarios;
}
} // namespace

int main(int argc, char **) {
    if (argc != 1) {
        std::fprintf(stderr, "E4C2_0_REFUSED hardware execution is unavailable; software-only contract test.\n");
        return 2;
    }
    try {
        for (bool dma_first : {false, true})
            for (bool independent : {false, true}) ordering(dma_first, independent);
        reuse_and_failures();
        aggregate_prefix_scope();
        active_peer_quarantine();
        current_c11_gate();
        std::printf("E4C2_0_SOFTWARE_PASS scenarios=%llu assertions=%llu expected_refusals=%llu "
                    "ownership_violations=%llu hardware=0 overlap_measured=0\n",
                    static_cast<unsigned long long>(scenarios),
                    static_cast<unsigned long long>(assertions),
                    static_cast<unsigned long long>(expected_refusals),
                    static_cast<unsigned long long>(ownership_violations));
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "E4C2_0_SOFTWARE_FAIL hardware=0 %s\n", e.what());
        return 1;
    }
}
