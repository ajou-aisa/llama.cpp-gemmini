#include "support/paired-support.hpp"
#include "support/paired-h2-emulator.hpp"
#include "matmul/paired-auto.hpp"
#include "trace/timeline.hpp"

#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>

using namespace ggml::potal;
using namespace potal_test;

namespace {

void admission(const char *budget = "255") {
    setenv("POTAL_INDEX_QUEUE_FRESH", "1", 1);
    setenv("POTAL_INDEX_QUEUE_BUDGET", budget, 1);
}

// Separate process lifetimes, never a production reset seam.
void isolated(const std::string &name, const std::function<bool()> &body) {
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        const bool ok = body();
        std::fflush(nullptr);
        _exit(ok ? 0 : 1);
    }
    int status = 0;
    pid_t done;
    do { done = pid > 0 ? waitpid(pid, &status, 0) : -1; } while (done < 0 && errno == EINTR);
    check(name, done == pid && pid > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

potal_index_config descriptor(uint32_t count = 5, uint32_t pair = 0) {
    return {pair, uint32_t((2 * pair + 1) * POTAL_HW_BANK_ROWS), 1, 64, count};
}

uint64_t expected_word(unsigned generation, unsigned ticket, unsigned pair, unsigned op,
                       unsigned status, unsigned count) {
    return (UINT64_C(1) << 36) | (uint64_t(generation) << 12) | (uint64_t(ticket) << 11) |
        (uint64_t(pair) << 27) | (uint64_t(op) << 8) | status | (uint64_t(count) << 20);
}

bool open(h2_emulator &e, potal_paired_session *&s) {
    const auto sink = e.sink();
    return potal_paired_session_open(&sink, &s) == POTAL_OK;
}

void reply_tests() {
    const uint64_t raw = expected_word(255, 1, 1, 4, 6, 42);
    const potal_index_queue_reply expected{6, 0, 4, 1, 255, 42, 1, 0, 1, 0};
    const auto decoded = potal_index_queue_decode(raw);
    bool ok = decoded.status == 6 && decoded.reason == 0 && decoded.op == 4 && decoded.ticket == 1 &&
        decoded.generation == 255 && decoded.count == 42 && decoded.pair == 1 && decoded.residency == 0 &&
        decoded.owner == 1 && decoded.reserved == 0 && potal_index_queue_reply_check(raw, &expected) == POTAL_OK;
    for (unsigned bit = 0; bit < 64; ++bit)
        ok = ok && potal_index_queue_reply_check(raw ^ (UINT64_C(1) << bit), &expected) == POTAL_ERR_BAD_RESPONSE;
    const auto all = potal_index_queue_decode(UINT64_MAX);
    ok = ok && all.status == 15 && all.reason == 15 && all.op == 7 && all.ticket == 1 &&
        all.generation == 255 && all.count == 127 && all.pair == 1 && all.residency == 255 &&
        all.owner == 255 && all.reserved == 0xfffff &&
        potal_index_queue_reply_check(raw, nullptr) == POTAL_ERR_INVALID_ARGUMENT;
    check("full64 decode and independent rejection of every reply bit", ok);
}

bool protocol_and_exhaustion() {
    admission();
    h2_emulator e;
    const uint32_t entries[5] = {0, 7, 31, 32, 63};
    for (unsigned i = 0; i < 255; ++i) {
        potal_paired_session *s = nullptr;
        if (!open(e, s)) return false;
        const auto cfg = descriptor(5, i & 1);
        const size_t start = e.records.size();
        potal_index_queue_result r{};
        if (potal_index_queue_install(s, &cfg, entries, 5, &r) != POTAL_OK ||
            r.allocations != 1 || r.raw != expected_word(i + 1, i & 1, i & 1, 4, 6, 5) ||
            !potal_paired_session_query(s).pair[i & 1].index_valid || e.records.size() != start + 6 ||
            e.records[start].funct != 32 || e.records[start + 1].funct != 33 ||
            e.records[start + 1].rs1 != (1u | ((i & 1) << 16)) ||
            e.records[start + 2].rs1 != (UINT64_C(7) << 32) ||
            e.records[start + 2].rs2 != (UINT64_C(32) << 32 | 31) ||
            e.records[start + 3].rs1 != 63 || e.records[start + 3].rs2 != 0 ||
            e.records[start + 5].rs2 != 0 ||
            potal_index_queue_remaining() != 254 - i ||
            potal_paired_session_close(s) != POTAL_OK) return false;
    }
    if (!e.queue_exhausted || e.queue_allocations != 255 || e.queue_applied != 255) return false;
    potal_paired_session *s = nullptr;
    if (!open(e, s)) return false;
    const auto cfg = descriptor();
    const size_t before = e.records.size();
    potal_index_queue_result r{};
    setenv("POTAL_INDEX_QUEUE_BUDGET", "255", 1);
    if (potal_index_queue_install(s, &cfg, entries, 5, &r) != POTAL_ERR_CAPACITY ||
        r.allocations || before != e.records.size() || potal_paired_session_close(s) != POTAL_OK) return false;
    // Hardware model independently rejects its 256th BEGIN, even after retirement.
    potal_rocc_cmd header{};
    if (potal_index_config_encode(&cfg, &header) != POTAL_OK) return false;
    return e.queue_command({32, 1, header.rs1, header.rs2}) == 9 &&
        e.queue_command({33, 1, 1, 0}) == (UINT64_C(1) << 8 | 3u << 4 | 8u) &&
        e.queue_exhausted && e.queue_allocations == 255;
}

bool queue_model() {
    h2_emulator e;
    const auto cfg = descriptor(1);
    potal_rocc_cmd h{};
    if (potal_index_config_encode(&cfg, &h) != POTAL_OK) return false;
    for (unsigned i = 0; i < 2; ++i) {
        if (e.queue_command({32, 1, h.rs1, h.rs2}) != 9 ||
            e.queue_command({33, 1, 1, 0}) != expected_word(i + 1, i, 0, 1, 0, 0) ||
            e.queue_command({34, 1, 7, 0}) != expected_word(i + 1, i, 0, 2, 9, 1) ||
            e.queue_command({35, 1, 0, 0}) != expected_word(i + 1, i, 0, 3, 9, 1)) return false;
    }
    if (e.tables[0].configured) return false; // SEAL is not publication.
    const uint64_t second_key = 1 | (2u << 1) | (1u << 9);
    const auto out_of_order = potal_index_queue_decode(e.queue_command({36, 1, second_key, 0}));
    if (out_of_order.status != 4 || out_of_order.reason != 3 || e.tables[0].configured) return false;
    if (e.queue_command({36, 1, 2 | (1u << 9), 0}) != expected_word(1, 0, 0, 4, 6, 1) ||
        e.tables[0].entry[0] != 7 ||
        e.queue_command({36, 1, second_key, 0}) != expected_word(2, 1, 0, 4, 6, 1)) return false;
    const auto stale = potal_index_queue_decode(e.queue_command({36, 1, second_key, 0}));
    if (stale.status != 2 || stale.reason != 2) return false;
    if (e.queue_command({32, 1, h.rs1, h.rs2}) != 9) return false;
    (void)e.queue_command({33, 1, 1, 0});
    const auto bad_tail = potal_index_queue_decode(e.queue_command({34, 1, UINT64_C(1) << 32, 0}));
    const auto sticky = potal_index_queue_decode(e.queue_command({32, 1, h.rs1, h.rs2}));
    return bad_tail.status == 7 && bad_tail.reason == 5 && sticky.status == 7 && sticky.reason == 5;
}

bool fatal_reply(uint32_t funct, uint64_t mask) {
    admission();
    h2_emulator e;
    e.queue_corrupt_funct = funct;
    e.queue_corrupt_mask = mask;
    potal_paired_session *s = nullptr;
    if (!open(e, s)) return false;
    const auto cfg = descriptor();
    const uint32_t entries[5] = {0, 1, 2, 3, 4};
    potal_index_queue_result r{};
    if (potal_index_queue_install(s, &cfg, entries, 5, &r) != POTAL_ERR_BAD_RESPONSE) return false;
    const auto q = potal_paired_session_query(s);
    const size_t before = e.records.size();
    const potal_index_write w{0, 1};
    if (!q.failed || q.pair[0].index_valid || r.allocations != (funct == 32 ? 0u : 1u) ||
        potal_paired_index_config(s, &cfg) != POTAL_ERR_BAD_RESPONSE ||
        potal_paired_index_write(s, &w) != POTAL_ERR_BAD_RESPONSE ||
        potal_index_queue_install(s, &cfg, entries, 5, &r) != POTAL_ERR_BAD_RESPONSE ||
        potal_paired_session_cleanup(s) != POTAL_ERR_BAD_RESPONSE || e.records.size() != before ||
        potal_paired_session_close(s) != POTAL_ERR_BAD_RESPONSE) return false;
    return potal_paired_session_open(nullptr, &s) == POTAL_ERR_BUSY;
}

struct output {
    potal_paired_report report;
    std::vector<int32_t> main, residual;
};

output execute(const paired_fixture &f, const potal_tile_plan &main, const residual::decoded_packet &d,
               h2_emulator &e, bool queue, bool prefetch = false,
               potal_paired_stream_mode stream = potal_paired_stream_mode::off, bool prepared = false) {
    output out;
    const size_t ldc = align_up(f.N, PD);
    std::unique_ptr<int32_t, dma_deleter> cm(static_cast<int32_t *>(dma_alloc(PD * ldc * sizeof(int32_t))));
    std::unique_ptr<int32_t, dma_deleter> cr(
        static_cast<int32_t *>(dma_alloc(align_up(std::max<size_t>(1, d.rows.size()), PD) * ldc * sizeof(int32_t))));
    setenv("GGML_POTAL_INDEX_QUEUE", queue ? "ON" : "OFF", 1);
    uint32_t generation = 0;
    potal_paired_request req;
    req.rows = f.M; req.n = f.N; req.k = f.K;
    req.main_plan = &main; req.weight = &f.weight;
    req.a = f.a.get(); req.lda = f.lda;
    req.residual = &d;
    req.c_main = cm.get(); req.ldc_main = ldc;
    req.c_residual = cr.get(); req.ldc_residual = ldc;
    req.generation = &generation;
    req.ktile_scale_reuse = true;
    req.next_k_prefetch = prefetch;
    req.stream = stream;
    req.stream_counter = stream != potal_paired_stream_mode::off ? h2_emulator::on_counter : nullptr;
    // H3-8I-25S strict issuer: plan + potal_paired_prepare first (producer side), then the
    // executor only consumes the prepared job.
    potal_paired_tile_plan tile;
    potal_paired_execution_plan plan;
    potal_paired_prepared job;
    if (prepared) {
        std::string why;
        const char * stage = nullptr;
        if (potal_paired_plan_stripe(main, f.M, f.N, f.K, d.rows.empty() ? nullptr : &d, tile, plan, &why) !=
                potal_paired_status::ok ||
            potal_paired_prepare(req, std::move(plan), job, &stage, &why) != potal_paired_status::ok) {
            out.report.status = potal_paired_status::invalid_argument;
            return out;
        }
        req.prepared = &job;
    }
    const auto sink = e.sink();
    const auto status = potal_paired_run_session(req, &sink, out.report);
    if (status == potal_paired_status::ok) {
        for (size_t i = 0; i < f.M; ++i)
            out.main.insert(out.main.end(), cm.get() + i * ldc, cm.get() + i * ldc + f.N);
        for (size_t i = 0; i < d.rows.size(); ++i)
            out.residual.insert(out.residual.end(), cr.get() + i * ldc, cr.get() + i * ldc + f.N);
    } else if (out.report.fatal) {
        cm.release(); cr.release(); // fatal device obligations preserve DMA buffers
    }
    return out;
}

bool paired_cases(bool preflight) {
    admission(preflight ? "1" : "255");
    setenv("POTAL_PAIRED_TIMING", "0", 1);
    if (::potal::timeline::enabled())
        ::potal::timeline::register_thread(::potal::timeline::role::issuer, -1, -1);
    paired_fixture f(3, PD + 3, 160, 761, [](size_t b, size_t n) { return int16_t((b + n) % 3); });
    potal_tile_plan main;
    if (!f.ok || potal_frozen_plan(f.M, f.N, f.K, 1, 1, 2, 1, &main) != POTAL_OK) return false;
    h2_emulator scalar, queue;
    for (unsigned kind = 0; kind < 4; ++kind) {
        std::vector<res_entry> entries;
        if (kind != 3)
            for (size_t k = 0; k < f.K; ++k)
                if (kind == 2 || k % (kind == 0 ? 31 : 7) == 0)
                    entries.push_back({k % f.M, k, kind == 1 ? 65537 : int32_t(k % 127 + 1)});
        residual::decoded_packet d;
        if (!entries.empty()) {
            const auto packet = make_packet(0, 0, f.M, f.K, f.N, entries);
            if (!packet || residual::decode_packet(*packet, f.weight, d) != grmd::RmdStatus::success) return false;
        }
        if (preflight) {
            const auto refused = execute(f, main, d, queue, true);
            return refused.report.status == potal_paired_status::execution_failed &&
                refused.report.hw_status == POTAL_ERR_CAPACITY && !refused.report.issued && !refused.report.fatal &&
                queue.records.empty() && potal_index_queue_remaining() == 1;
        }
        // Same emulator/device lifetime across all Queue requests, not reset per request.
        const auto a = execute(f, main, d, scalar, false);
        const auto b = execute(f, main, d, queue, true);
        if (a.report.status != potal_paired_status::ok || b.report.status != potal_paired_status::ok ||
            a.main != b.main || a.residual != b.residual || a.report.fatal || b.report.fatal ||
            b.report.counts.queue_allocations != b.report.plan.index_programs ||
            a.report.counts.queue_allocations != 0 || b.report.counts.index_configs != 0 ||
            b.report.counts.index_writes != 0 || !scalar.errors.empty() || !queue.errors.empty()) return false;
        // Strict issuer: the prepared job issues the same commands on both Index paths.
        const auto pq = execute(f, main, d, queue, true, false, potal_paired_stream_mode::off, true);
        const auto ps = execute(f, main, d, scalar, false, false, potal_paired_stream_mode::off, true);
        if (pq.report.status != potal_paired_status::ok || ps.report.status != potal_paired_status::ok ||
            pq.main != a.main || pq.residual != a.residual || ps.main != a.main || ps.residual != a.residual ||
            pq.report.counts.queue_allocations != b.report.counts.queue_allocations ||
            pq.report.counts.index_writes != 0 || ps.report.counts.index_writes != a.report.counts.index_writes ||
            ps.report.counts.scale_writes != a.report.counts.scale_writes || !queue.errors.empty() ||
            !scalar.errors.empty()) return false;
        if (kind == 0) {
            const auto c = execute(f, main, d, queue, true, true);
            const auto s = execute(f, main, d, queue, true, false, potal_paired_stream_mode::bounded);
            if (c.report.status != potal_paired_status::ok || s.report.status != potal_paired_status::ok ||
                c.main != a.main || c.residual != a.residual || s.main != a.main || s.residual != a.residual ||
                !queue.errors.empty()) return false;
        }
    }
    // Each Residual follows a successfully installed table; no scalar Index command
    // is emitted by the Queue path (the emulator's installer is internal).
    for (const auto &r : queue.records)
        if (r.funct == POTAL_FUNCT_INDEX_CONFIG || r.funct == POTAL_FUNCT_INDEX_WRITE) return false;
    if (::potal::timeline::enabled()) {
        // Every issuer stage must execute in the native enabled-path check.
        // Inspect recorded machine fields, not CSV prose or timing estimates.
        const auto *ring = ::potal::timeline::detail::t_ring;
        if (!ring || ring->overflow.load()) return false;
        unsigned begins[9] = {}, ends[9] = {};
        for (uint32_t i = 0; i < ring->head.load(); ++i) {
            const auto &r = ring->events[i];
            if (r.stage < 30 || r.stage > 38) continue;
            const size_t stage = r.stage - 30;
            if (r.reserved || r.pad0 || r.seq != i || (i && r.t_local < ring->events[i-1].t_local)) return false;
            if (r.event_type == 1) {
                if (begins[stage] != ends[stage]) return false;
                ++begins[stage];
            } else if (r.event_type == 2) {
                if (begins[stage] != ends[stage] + 1 || r.status != (r.stage == 32 ? 6 : 0)) return false;
                ++ends[stage];
            } else return false;
        }
        for (size_t i = 0; i < 9; ++i)
            if (!begins[i] || begins[i] != ends[i]) return false;
    }
    return queue.queue_allocations == queue.queue_applied;
}

} // namespace

int main() {
    reply_tests();
    for (const char *budget : {"", "0", "256", "-1", "+1", "1x", "999999999999999999999999"}) {
        isolated(std::string("budget refusal: '") + budget + "'", [=] {
            admission(budget);
            h2_emulator e;
            potal_paired_session *s = nullptr;
            if (!open(e, s)) return false;
            const auto cfg = descriptor();
            const uint32_t entries[5] = {};
            potal_index_queue_result r{};
            return potal_index_queue_remaining() == 0 &&
                potal_index_queue_install(s, &cfg, entries, 5, &r) == POTAL_ERR_CAPACITY &&
                e.records.empty() && potal_paired_session_close(s) == POTAL_OK;
        });
    }
    for (const char *fresh : {"", "0", "ON", "11"}) {
        isolated(std::string("fresh assertion refusal: '") + fresh + "'", [=] {
            admission(); setenv("POTAL_INDEX_QUEUE_FRESH", fresh, 1);
            return potal_index_queue_remaining() == 0;
        });
    }
    isolated("missing admission and immutable process budget", [] {
        unsetenv("POTAL_INDEX_QUEUE_FRESH"); unsetenv("POTAL_INDEX_QUEUE_BUDGET");
        if (potal_index_queue_remaining() != 0) return false;
        admission();
        return potal_index_queue_remaining() == 0;
    });
    isolated("no accelerator and invalid entries issue nothing", [] {
        admission();
        potal_paired_session *s = nullptr;
        if (potal_paired_session_open(nullptr, &s) != POTAL_OK) return false;
        const auto cfg = descriptor();
        uint32_t entries[5] = {};
        potal_index_queue_result r{};
        if (potal_index_queue_install(s, &cfg, entries, 5, &r) != POTAL_ERR_NO_HARDWARE ||
            potal_paired_session_query(s).commands || potal_paired_session_close(s) != POTAL_OK) return false;
        h2_emulator e;
        if (!open(e, s)) return false;
        entries[4] = 64;
        return potal_index_queue_install(s, &cfg, entries, 5, &r) == POTAL_ERR_INVALID_ARGUMENT &&
            e.records.empty() && potal_index_queue_remaining() == 255 && potal_paired_session_close(s) == POTAL_OK;
    });
    isolated("255 allocations survive session closes and never wrap", protocol_and_exhaustion);
    isolated("two tickets, FIFO APPLY, SEAL publication, stale keys and sticky tail fault", queue_model);
    for (uint32_t funct = 32; funct <= 36; ++funct)
        isolated("unexpected full64 reply quarantines funct " + std::to_string(funct),
                 [=] { return fatal_reply(funct, UINT64_C(1) << 44); });
    isolated("APPLY_STARTED is not APPLIED", [] { return fatal_reply(36, 3); });
    isolated("wrong APPLY count is fatal", [] { return fatal_reply(36, UINT64_C(1) << 20); });
    isolated("Queue versus scalar bitwise across sparse dense multilane empty and managed requests",
             [] { return paired_cases(false); });
    isolated("whole-request preflight refusal leaves zero commands", [] { return paired_cases(true); });
    std::printf("potal-index-queue-test: %s (%d failures)\n", g_failures ? "FAIL" : "PASS", g_failures);
    return g_failures ? 1 : 0;
}
