// Hardware-free checks of the strict hardware timeline (H3-8I-25AC, trace/hw-window.hpp):
//
//   model     support/hw-trace-model.hpp against the CounterFile.scala rules, cycle by cycle
//   API       potal_paired_hw_trace_enable, _arm, _read and potal_paired_hw_counter: refusals, and a refused or
//             failed call leaves the session usable
//   pipeline  the full strict route on the H2 emulator with small real shapes: window identity and composition,
//             the legal position of every trace command, selftest_evaluate on real and mutated records
//   identity  OFF against ON: the same masked commands and the same output bytes
//   failure   injected trace failures change neither the commands nor the output; a failing Fence takes the
//             existing error path with an aborted record
//
// Every stamp here is MEASURED-NATIVE at most: a model counter or the host clock, never a hardware time.
//
//   potal-hw-timeline-test                  all of the above
//   potal-hw-timeline-test --sample <path>  one host-clock strict run, flushed to <path> and its companions
//   potal-hw-timeline-test --capacity-child internal: one run under POTAL_HW_TIMELINE_WINDOWS set by the parent

#include "ggml-cpp.h"
#include "ggml-quants.h"
#include "matmul/dense.hpp"
#include "matmul/execution.hpp"
#include "matmul/options.hpp"
#include "matmul/stripe.hpp"
#include "matmul/tiling.hpp"
#include "potal.h"
#include "potal_hw_params.h"
#include "quants/act/exsia/types.hpp"
#include "support/paired-h2-emulator.hpp"
#include "support/stripe-golden.hpp"
#include "support/stripe-owner.hpp"
#include "trace/hw-window.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace ggml::potal;
namespace hwn = ::potal::hw_window;
using model   = potal_test::hw_trace_model;
using h2_emu  = potal_test::h2_emulator;
using cmd_class = model::cmd_class;

namespace {

int checks = 0, failures = 0;

void check(const std::string & name, bool ok) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    ++checks;
    failures += ok ? 0 : 1;
}

constexpr size_t D = POTAL_HW_DIM;

potal_paired_cmd_record counter_cmd(uint64_t rs1, uint32_t xd = 1, uint64_t rs2 = 0) {
    return {POTAL_FUNCT_COUNTER, xd, rs1, rs2};
}

uint64_t send(model & m, uint64_t rs1) {
    uint64_t v = ~uint64_t(0);
    const potal_paired_cmd_record c = counter_cmd(rs1);
    m.counter_command(c, &v);
    return v;
}

uint64_t arm(model & m) { return send(m, POTAL_LDB_ONLY_TRACE_RESET); }
uint64_t read_field(model & m, uint32_t f) { return send(m, POTAL_LDB_ONLY_TRACE_READ + f); }

constexpr uint64_t flags_fence = hwn::flag_seen_fence_accept | (uint64_t(1) << 13);

// ---- model ----------------------------------------------------------------------------------------

void test_model_window() {
    {
        model m;
        m.cycles_per_step = 10;
        const bool empty = m.peek(0) == 0 && m.peek(1) == 0 && m.peek(22) == 0 && m.peek(23) == 0;
        for (int i = 0; i < 5; ++i) m.step();
        const uint64_t at_arm = m.clock_cycle;
        const uint64_t reply = arm(m);
        check("model reset and arm: reply 1, armed, start = end = clockCycle, stamps clear, armed edge records nothing",
              empty && at_arm == 50 && reply == 1 && m.regs.armed && !m.regs.overflow && m.peek(22) == 50 &&
                  m.peek(23) == 50 && m.peek(0) == hwn::flag_armed && m.peek(1) == 10);
    }
    {
        model m;
        m.cycles_per_step = 10;
        arm(m);                        // edge at 0
        m.dma_request(0);              // 10
        m.b_write(0);                  // 20
        m.mac(0);                      // 30
        m.step();                      // 40
        m.mac(0);                      // 50
        m.b_write(0);                  // 60
        m.dma_request(0);              // 70
        check("model first and last stamps, counts, seen bits",
              m.peek(24) == 10 && m.peek(25) == 70 && m.peek(3) == 20 && m.peek(4) == 60 && m.peek(7) == 30 &&
                  m.peek(8) == 50 && m.peek(13) == 2 && m.peek(11) == 2 && m.peek(12) == 0 && m.peek(14) == 0 &&
                  m.peek(5) == 0 && m.peek(9) == 0 && m.peek(26) == 0 &&
                  m.peek(0) == (hwn::flag_armed | hwn::flag_seen_b0 | hwn::flag_seen_mac0 | hwn::flag_seen_dma0));
    }
    {
        model m;
        m.cycles_per_step = 10;
        arm(m);
        m.work_begin();  // 10
        m.mac(1);        // 20
        m.fence(1, 0, 0);  // 30 request, not drained; 40 accepted
        const model::registers frozen = m.regs;
        m.mac(1);
        m.b_write(0);
        m.work_begin();
        m.fence(1, 0, 0);
        check("model Fence freeze: accepted Fence clears armed, sets end, later events move nothing",
              frozen.end == 40 && !frozen.armed && frozen.first[model::e_fence_request] == 30 &&
                  frozen.first[model::e_fence_accepted] == 40 && frozen.first[model::e_not_drained] == 10 &&
                  frozen.last[model::e_not_drained] == 30 && m.regs == frozen && m.peek(1) == 40);
    }
    {
        const auto survives = [](uint32_t xd, uint64_t rs1, uint64_t rs2) {
            model m;
            arm(m);
            m.fence(xd, rs1, rs2);
            return m.regs.armed && m.regs.seen[model::e_fence_request] && !m.regs.seen[model::e_fence_accepted];
        };
        check("model Fence with nonzero rs1, nonzero rs2 or xd=0 does not freeze",
              survives(1, 1, 0) && survives(1, 0, 1) && survives(0, 0, 0));
        model busy;
        busy.work_begin();
        busy.fence(0, 0, 0);
        const bool in_flight = busy.work_in_flight();
        const uint64_t reply = arm(busy);
        check("model non-canonical Fence waits for nothing: work stays in flight and the next reset replies 0",
              in_flight && reply == 0 && !busy.regs.armed);
    }
    for (uint32_t pair = 0; pair < 2; ++pair) {
        model m;
        arm(m);
        m.work_begin();
        m.dma_request(pair);
        m.b_write(pair);
        m.mac(pair);
        m.step();
        m.dma_request(pair);
        m.b_write(pair);
        m.mac(pair);
        m.fence(1, 0, 0);
        const hwn::field_map & f = hwn::pair_fields[pair];
        const hwn::field_map & o = hwn::pair_fields[1 - pair];
        const auto spans = [&](uint32_t first, uint32_t last) { return m.peek(first) != 0 && m.peek(first) < m.peek(last); };
        check("field map names each stamp of its own pair, pair " + std::to_string(pair),
              spans(f.mac_first, f.mac_last) && spans(f.bwrite_first, f.bwrite_last) && spans(f.bdma_first, f.bdma_last) &&
                  spans(hwn::trace_busy_first, hwn::trace_busy_last) && m.peek(f.mac_cycles) == 2 &&
                  m.peek(hwn::trace_start) < m.peek(hwn::trace_end) && m.peek(o.mac_first) == 0 && m.peek(o.mac_last) == 0 &&
                  m.peek(o.mac_cycles) == 0 && m.peek(o.bwrite_first) == 0 && m.peek(o.bwrite_last) == 0 &&
                  m.peek(o.bdma_first) == 0 && m.peek(o.bdma_last) == 0);
    }
    {
        const auto is = [](uint32_t main, uint32_t residual, bool agrees, hwn::kind k, hwn::confidence c) {
            const hwn::window_label label = hwn::label_window(main, residual, agrees);
            return label.k == k && label.c == c;
        };
        check("window label from GEMM counts, each kind and each refusal",
              is(0, 0, true, hwn::kind::no_gemm, hwn::confidence::none) && is(1, 0, true, hwn::kind::main, hwn::confidence::single_gemm) &&
                  is(0, 1, true, hwn::kind::residual, hwn::confidence::single_gemm) &&
                  is(1, 1, true, hwn::kind::main_first_residual, hwn::confidence::merged) &&
                  is(0, 2, true, hwn::kind::residual, hwn::confidence::merged) && is(0, 5, true, hwn::kind::residual, hwn::confidence::merged) &&
                  is(2, 0, true, hwn::kind::unknown, hwn::confidence::none) && is(1, 2, true, hwn::kind::unknown, hwn::confidence::none) &&
                  is(1, 0, false, hwn::kind::unknown, hwn::confidence::none) && is(0, 0, false, hwn::kind::unknown, hwn::confidence::none));
    }
    {
        model m;
        arm(m);
        m.mac(0);
        m.fence(1, 0, 0);
        const uint64_t want = m.peek(0);
        check("model read: reply is values(field), flags carry the frozen seen bits",
              read_field(m, 0) == want && (want & hwn::flag_armed) == 0 && (want & hwn::flag_seen_mac0) != 0 &&
                  (want & flags_fence) == flags_fence && read_field(m, 13) == 1 && read_field(m, 22) == 0);
    }
    {
        model m;
        arm(m);
        m.mac(1);
        m.fence(1, 0, 0);
        const model::registers frozen = m.regs;
        const model::counter_file file = m.file;
        bool same = true;
        for (uint32_t f = 0; f < 32; ++f) {
            const uint64_t a = read_field(m, f), b = read_field(m, f);
            same = same && a == b;
        }
        bool file_same = std::equal(file.counters, file.counters + 8, m.file.counters) && !m.file.snapshot_enable;
        check("model repeated read returns the same value and changes no state", same && m.regs == frozen && file_same);
    }
    {
        model m;
        m.cycles_per_step = 10;
        for (int i = 0; i < 3; ++i) m.step();
        arm(m);
        m.fence(1, 0, 0);
        bool zero = true;
        for (uint32_t f : {2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 20u, 21u, 24u, 25u, 26u, 27u}) zero = zero && m.peek(f) == 0;
        for (uint32_t f = 11; f <= 17; ++f) zero = zero && m.peek(f) == 0;
        check("model empty window: event seen bits clear, stamps and counts zero, only the Fence seen",
              zero && m.peek(0) == flags_fence && m.peek(22) == 30 && m.peek(23) == 40 && m.peek(1) == 10);
    }
    {
        model m;
        arm(m);
        m.mac(0);
        m.fence(1, 0, 0);
        const model::registers one = m.regs;
        m.work_begin();
        const uint64_t refused = arm(m);
        const bool kept = refused == 0 && m.regs == one;
        m.fence(1, 0, 0);
        const uint64_t accepted = arm(m);
        const bool cleared = accepted == 1 && m.regs.armed && !m.regs.seen[model::e_mac0] && m.regs.counts[2] == 0 &&
                             m.peek(7) == 0;
        check("model multiple windows: stamps overwrite only at the next accepted arm", kept && cleared);
    }
    {
        model m;
        arm(m);
        m.mac(0);
        m.work_begin();
        for (int i = 0; i < 100; ++i) m.step();
        check("model missing Fence leaves armed set, no timeout", m.regs.armed && (m.peek(0) & hwn::flag_armed) != 0);
    }
    {
        model m;
        arm(m);
        m.mac(0);
        m.fence(1, 0, 0);
        bool zero = true;
        for (uint32_t f = 28; f < 32; ++f) zero = zero && read_field(m, f) == 0;
        check("model invalid read: fields 28-31 read 0", zero);
    }
    {
        model m;
        m.cycles_per_step = 10;
        arm(m);          // 0
        m.step();        // 10
        const uint64_t first_read = read_field(m, 20);  // fires at 20, samples before its own stamp
        const uint64_t second = read_field(m, 20);      // fires at 30
        const uint64_t last   = read_field(m, 21);      // fires at 40, sees the stamp of 30
        const bool stamped = m.regs.first[model::e_not_drained] == 20 && m.regs.last[model::e_not_drained] == 40;
        const uint64_t live = read_field(m, 1);  // fires at 50
        check("model read while armed: not-drained stamps move as the RTL does, reply sampled before its own edge",
              first_read == 0 && second == 20 && last == 30 && stamped && live == 50 && m.regs.armed);
    }
    {
        model m;
        arm(m);
        m.saturate_clock();
        const bool before = !m.regs.overflow;
        m.step();
        const bool set = m.regs.overflow;
        m.step();
        const bool held = m.clock_cycle == model::all_ones;
        m.fence(1, 0, 0);
        model n;
        n.saturate_clock();
        arm(n);
        check("model clock saturation sets overflow, clock stays, arm at saturation sets it too",
              before && set && held && (m.peek(0) & hwn::flag_overflow) != 0 && n.regs.overflow);
    }
    {
        model m;
        arm(m);
        m.saturate_count(2);
        m.mac(0);
        check("model count saturation sets overflow and holds the count",
              m.regs.overflow && m.regs.counts[2] == model::all_ones && m.regs.seen[model::e_mac0]);
    }
    {
        model m;
        m.cycles_per_step = 10;
        arm(m);
        m.mac(0);
        m.fence(1, 0, 0);
        const model::registers frozen = m.regs;
        m.work_begin();
        const uint64_t busy = arm(m);
        const bool frozen_kept = busy == 0 && m.regs == frozen;
        m.fence(1, 0, 0);
        arm(m);
        m.mac(1);
        m.work_begin();
        const uint64_t mac_first = m.peek(9);
        const uint64_t armed_busy = arm(m);
        const bool armed_kept = armed_busy == 0 && m.regs.armed && m.peek(9) == mac_first && m.regs.seen[model::e_mac1] &&
                                m.regs.seen[model::e_not_drained];
        m.fence(1, 0, 0);
        m.force_not_drained();
        const uint64_t forced = arm(m);
        m.force_not_drained(false);
        const uint64_t idle = arm(m);
        check("model in-flight reset replies 0 and keeps stamps, drained reset replies 1",
              frozen_kept && armed_kept && forced == 0 && idle == 1);
    }
}

void test_model_counters() {
    const auto configure = [](uint32_t event, uint32_t i) { return (uint64_t(event) << 12) | (uint64_t(i) << 4) | 8; };
    model m;
    const uint64_t c0 = send(m, configure(model::ev_mac0, 0));
    const uint64_t c1 = send(m, configure(model::ev_mac1, 1));
    m.mac(0);
    m.mac(0);
    m.mac(1);
    const uint64_t r0 = send(m, 0 << 4), r1 = send(m, 1 << 4);
    const uint64_t prior_config = send(m, configure(model::ev_mac0, 0));
    const uint64_t after_config = send(m, 0 << 4), other = send(m, 1 << 4);
    check("model counter config: returns the prior value, zeroes that counter only",
          c0 == 0 && c1 == 0 && r0 == 2 && r1 == 1 && prior_config == 2 && after_config == 0 && other == 1);

    m.mac(0);
    m.mac(1);
    const uint64_t prior_snap = send(m, 4);
    m.mac(0);
    m.mac(1);
    const uint64_t snap0 = send(m, 0 << 4), snap1 = send(m, 1 << 4);
    const uint64_t prior_clear = send(m, 2);
    const uint64_t live0 = send(m, 0 << 4), live1 = send(m, 1 << 4);
    check("model counter snapshot: atomic, reads hold while counters run, clear returns live values, prior replies",
          prior_snap == 1 && snap0 == 1 && snap1 == 2 && prior_clear == 1 && live0 == 2 && live1 == 3);

    const uint64_t prior_snap2 = send(m, 4);
    const uint64_t prior_reset = send(m, 1);
    const bool reset_zero = send(m, 0 << 4) == 0 && send(m, 1 << 4) == 0 && !m.file.snapshot_enable &&
                            m.file.snapshot[1] == 0;
    m.mac(0);
    const uint64_t kept_config = send(m, 0 << 4);
    check("model counter reset: zeroes counters and snapshot, keeps config, returns the prior value",
          prior_snap2 == 2 && prior_reset == 2 && reset_zero && kept_config == 1 && m.file.config[0] == model::ev_mac0);

    const uint64_t a = send(m, 0xA);
    check("model rs1 0xA: prior value of counter 0, config 0 written, snapshot cleared",
          a == 1 && m.file.config[0] == 0 && m.file.counters[0] == 0 && send(m, POTAL_LDB_ONLY_TAG) == POTAL_LDB_ONLY_CAPABILITY);

    model t;
    send(t, configure(model::ev_mac0, 0));
    t.mac(0);
    send(t, POTAL_LDB_ONLY_TRACE_RESET);
    for (uint32_t f = 0; f < 28; ++f) read_field(t, f);
    uint64_t v = 0;
    const potal_paired_cmd_record not_canonical = counter_cmd(POTAL_LDB_ONLY_TRACE_READ + 1, 1, 7);
    const uint64_t trace_kept = send(t, 0 << 4);
    t.counter_command(not_canonical, &v);
    check("model trace commands never reach the counter file; a TRACE_READ with rs2 set is a counter command",
          trace_kept == 1 && v == 1 && t.file.counters[0] == 0);

    model x;
    send(x, configure(model::ev_cross, 2));
    arm(x);
    model::signals s;
    s.mac[0] = true;
    s.b_write[1] = true;
    x.step(s);
    s = {};
    s.mac[1] = true;
    s.b_write[0] = true;
    x.step(s);
    check("model cross-pair overlap: fields 15, 16, 17 and event 51",
          x.peek(15) == 1 && x.peek(16) == 1 && x.peek(17) == 2 && x.file.counters[2] == 2);
}

// ---- API ------------------------------------------------------------------------------------------

struct api_session {
    h2_emu                 h2;
    potal_paired_sink      sink;
    potal_paired_session * s = nullptr;
    alignas(256) int32_t   c[D * D] = {};

    api_session() {
        h2.hw_trace_mode = true;
        sink = h2.sink();
        potal_paired_session_open(&sink, &s);
    }
    ~api_session() { potal_paired_session_close(s); }

    potal_status enable(int probe = 0) { return potal_paired_hw_trace_enable(s, h2_emu::on_counter, probe); }
    potal_status arm() {
        uint64_t a = 0, b = 0;
        return potal_paired_hw_trace_arm(s, &a, &b);
    }
    potal_status readback() {
        potal_acc_readback r = {};
        r.acc_bank = 0;
        r.i_tiles = 1;
        r.j_tiles = 1;
        r.c = c;
        r.ldc = D;
        return potal_paired_acc_readback_issue(s, &r);
    }
    // The contract of every diagnostic refusal: no failure, and numeric commands still flow.
    bool still_usable() {
        const auto before = potal_paired_session_query(s);
        const bool fence = potal_paired_fence(s) == POTAL_OK;
        const bool write = potal_paired_scale_write(s, 3, 1, 0, 0) == POTAL_OK;
        const bool release = potal_paired_scale_release(s, 3, 1, 0) == POTAL_OK;
        const auto after = potal_paired_session_query(s);
        return !before.failed && fence && write && release && !after.failed && h2.errors.empty();
    }
};

void test_api() {
    {
        h2_emu h2;
        const potal_paired_sink sink = h2.sink();
        potal_paired_session * s = nullptr;
        potal_paired_session_open(&sink, &s);
        const potal_status st = potal_paired_hw_trace_enable(s, nullptr, 0);
        potal_paired_session_close(s);
        potal_paired_session * none = nullptr;
        potal_paired_session_open(nullptr, &none);
        const potal_status disagree = potal_paired_hw_trace_enable(none, h2_emu::on_counter, 0);
        const potal_status no_hw = potal_paired_hw_trace_enable(none, nullptr, 0);
        potal_paired_session_close(none);
        check("API enable refused: sink without counter, counter without sink, no hardware",
              st == POTAL_ERR_INVALID_ARGUMENT && disagree == POTAL_ERR_INVALID_ARGUMENT && no_hw == POTAL_ERR_NO_HARDWARE);
    }
    {
        api_session a;
        const potal_status ok_rb = a.readback();
        const potal_status busy = a.enable();
        check("API enable refused BUSY with work outstanding", ok_rb == POTAL_OK && busy == POTAL_ERR_BUSY && a.still_usable());
    }
    {
        api_session a;
        const potal_status begin = potal_paired_observation_begin(a.s, h2_emu::on_counter);
        const potal_status busy = a.enable();
        uint64_t cycles[7];
        const potal_status end = potal_paired_observation_end(a.s, cycles);
        check("API enable refused BUSY during an observation", begin == POTAL_OK && busy == POTAL_ERR_BUSY && end == POTAL_OK);
    }
    {
        api_session a;
        a.h2.hw_trace.ldb_only_signature = 0;
        const potal_status mismatch = a.enable(1);
        const size_t probes = a.h2.trace_records.size();
        const potal_status arm_off = a.arm();
        check("API enable with probe: capability mismatch UNSUPPORTED, session stays off",
              mismatch == POTAL_ERR_UNSUPPORTED && probes == 2 && arm_off == POTAL_ERR_INVALID_ARGUMENT && a.still_usable());
    }
    {
        api_session a;
        const uint64_t before = potal_paired_session_query(a.s).commands;
        const potal_status enabled = a.enable(1);
        const potal_status arm1 = a.arm();
        const potal_status arm2 = a.arm();
        const size_t sent = a.h2.trace_records.size();
        check("API arm refused when armed, with no command sent; enable and arm count no paired command",
              enabled == POTAL_OK && arm1 == POTAL_OK && arm2 == POTAL_ERR_BUSY && sent == 3 &&
                  potal_paired_session_query(a.s).commands == before && a.h2.records.empty());
        const potal_status fence = potal_paired_fence(a.s);
        const potal_status frozen = a.arm();
        const bool not_sent = a.h2.trace_records.size() == sent;
        uint64_t v = 0;
        const potal_status rd = potal_paired_hw_trace_read(a.s, 0, &v);
        const potal_status rearm = a.arm();
        check("API arm refused when frozen and unread; a read then allows the next arm",
              fence == POTAL_OK && frozen == POTAL_ERR_BUSY && not_sent && rd == POTAL_OK &&
                  (v & hwn::flag_seen_fence_accept) != 0 && rearm == POTAL_OK);
        check("API a window lifecycle leaves the session usable", potal_paired_fence(a.s) == POTAL_OK && a.still_usable());
    }
    {
        api_session a;
        a.enable();
        const potal_status rb = a.readback();
        const potal_status busy = a.arm();
        const size_t sent = a.h2.trace_records.size();
        uint64_t v = 0;
        const potal_status counter = potal_paired_hw_counter(a.s, 1, &v);
        check("API arm and counter refused BUSY with software work outstanding, no command sent",
              rb == POTAL_OK && busy == POTAL_ERR_BUSY && counter == POTAL_ERR_BUSY && sent == 0 &&
                  a.h2.trace_records.empty() && a.still_usable());
    }
    {
        api_session a;
        a.enable();
        a.arm();
        const size_t sent = a.h2.trace_records.size();
        const auto before = a.h2.hw_trace.regs;
        uint64_t v = 0;
        const potal_status counter = potal_paired_hw_counter(a.s, 1, &v);
        const potal_status enable = a.enable();
        check("API counter and enable refused BUSY while armed, with no command and no stamp moved",
              counter == POTAL_ERR_BUSY && enable == POTAL_ERR_BUSY && a.h2.trace_records.size() == sent && a.h2.hw_trace.regs == before);
        const potal_status fence = potal_paired_fence(a.s);
        const potal_status after = potal_paired_hw_counter(a.s, 1, &v);
        check("API counter accepted again once the Fence froze the window", fence == POTAL_OK && after == POTAL_OK);
    }
    {
        api_session a;
        a.enable();
        const potal_index_config cfg = {0, uint32_t(POTAL_HW_BANK_ROWS), 1, 64, 1};
        const potal_status opened = potal_paired_index_config(a.s, &cfg);
        const potal_status busy = a.arm();
        check("API arm refused BUSY with an Index window open, no command sent",
              opened == POTAL_OK && busy == POTAL_ERR_BUSY && a.h2.trace_records.empty());
    }
    {
        api_session a;
        a.enable();
        a.arm();
        a.h2.fail_funct = POTAL_FUNCT_FENCE;
        a.h2.fail_status = POTAL_HW_BAD_COMMAND;
        const potal_status fence = potal_paired_fence(a.s);
        const size_t sent = a.h2.trace_records.size();
        uint64_t v = 0;
        const potal_status read = potal_paired_hw_trace_read(a.s, 0, &v);
        check("API read refused after a failed Fence: the window did not freeze",
              fence != POTAL_OK && read == POTAL_ERR_BUSY && a.h2.trace_records.size() == sent);
    }
    {
        api_session a;
        a.enable();
        a.h2.hw_trace.force_not_drained();
        const potal_status busy = a.arm();
        const bool armed = a.h2.hw_trace.regs.armed;
        a.h2.hw_trace.force_not_drained(false);
        const potal_status later = a.arm();
        check("API arm refused BUSY when hardware replies 0: nothing armed, next arm succeeds",
              busy == POTAL_ERR_BUSY && !armed && later == POTAL_OK && a.still_usable());
    }
    {
        api_session a;
        a.enable();
        a.h2.hw_trace.override_reply_nth(cmd_class::trace_reset, 1, 2);
        const potal_status bad = a.arm();
        const potal_status later = a.arm();
        check("API arm refused BAD_RESPONSE when reply is 2", bad == POTAL_ERR_BAD_RESPONSE && later == POTAL_OK &&
              a.still_usable());
    }
    {
        api_session a;
        uint64_t v = 0;
        const potal_status off = potal_paired_hw_trace_read(a.s, 0, &v);
        a.enable();
        const potal_status idle = potal_paired_hw_trace_read(a.s, 0, &v);
        a.arm();
        const size_t sent = a.h2.trace_records.size();
        const auto before = a.h2.hw_trace.regs;
        const potal_status armed = potal_paired_hw_trace_read(a.s, 20, &v);
        check("API read refused before Fence: off, idle and armed, with no command and no stamp moved",
              off == POTAL_ERR_INVALID_ARGUMENT && idle == POTAL_ERR_BUSY && armed == POTAL_ERR_BUSY &&
                  a.h2.trace_records.size() == sent && a.h2.hw_trace.regs == before);
        potal_paired_fence(a.s);
        const potal_status bad28 = potal_paired_hw_trace_read(a.s, 28, &v);
        const potal_status bad31 = potal_paired_hw_trace_read(a.s, 31, &v);
        const potal_status ok27 = potal_paired_hw_trace_read(a.s, 27, &v);
        check("API read refuses field >= 28", bad28 == POTAL_ERR_INVALID_ARGUMENT && bad31 == POTAL_ERR_INVALID_ARGUMENT &&
              ok27 == POTAL_OK && a.still_usable());
    }
    {
        api_session a;
        uint64_t v = 0;
        const potal_status off = potal_paired_hw_counter(a.s, 1, &v);
        a.enable();
        bool bad = true;
        for (uint64_t rs1 : {uint64_t(3), uint64_t(5), uint64_t(0x80), uint64_t(0x18) | (uint64_t(52) << 12),
                             POTAL_LDB_ONLY_TRACE_RESET, POTAL_LDB_ONLY_TAG, uint64_t(0xA)})
            bad = bad && potal_paired_hw_counter(a.s, rs1, &v) == POTAL_ERR_INVALID_ARGUMENT;
        const size_t sent = a.h2.trace_records.size();
        bool good = true;
        for (uint64_t rs1 : {uint64_t(1), uint64_t(2), uint64_t(4), uint64_t(7) << 4,
                             (uint64_t(51) << 12) | (uint64_t(7) << 4) | 8})
            good = good && potal_paired_hw_counter(a.s, rs1, &v) == POTAL_OK;
        check("API counter refused with a bad rs1 or before enable; each listed form passes",
              off == POTAL_ERR_INVALID_ARGUMENT && bad && sent == 0 && good && a.h2.trace_records.size() == 5 &&
                  a.still_usable());
    }
    {
        api_session a;
        a.enable();
        a.h2.hw_trace.fail_nth(cmd_class::trace_reset, 1, POTAL_HW_BAD_COMMAND);
        const potal_status arm_failed = a.arm();
        a.h2.hw_trace.fail_nth(cmd_class::counter_read, 1, POTAL_HW_BAD_COMMAND);
        uint64_t v = 0;
        const potal_status counter_failed = potal_paired_hw_counter(a.s, 0, &v);
        const potal_status arm_again = a.arm();
        potal_paired_fence(a.s);
        a.h2.hw_trace.fail_nth(cmd_class::trace_read, 1, POTAL_ERR_BAD_RESPONSE);
        const potal_status read_failed = potal_paired_hw_trace_read(a.s, 0, &v);
        const auto q = potal_paired_session_query(a.s);
        check("API a failed call leaves the session usable: later Fence and Scale succeed, query shows no failure",
              arm_failed == POTAL_HW_BAD_COMMAND && counter_failed == POTAL_HW_BAD_COMMAND && arm_again == POTAL_OK &&
                  read_failed == POTAL_ERR_BAD_RESPONSE && !q.failed && q.error_stage == nullptr && a.still_usable());
    }
}

// ---- pipeline fixture -----------------------------------------------------------------------------

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

    node(size_t M_, size_t N_, size_t K_, uint32_t seed, float outlier, bool frozen) : M(M_), N(N_), K(K_) {
        ggml_init_params ip = {(M * K + N * K) * 8 + 4 * 1024 * 1024, nullptr, false};
        ctx.reset(ggml_init(ip));
        w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, (int64_t) K, (int64_t) N);
        x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, (int64_t) K, (int64_t) M);
        y = ggml_mul_mat(ctx.get(), w, x);
        std::mt19937                    rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float>              wf(N * K);
        const float                     mag[] = {0.02f, 0.08f, 0.3f, 0.01f};
        for (size_t n = 0; n < N; ++n)
            for (size_t k = 0; k < K; ++k) wf[n * K + k] = nd(rng) * mag[(k / 32 + n) % 4];
        ggml_quantize_chunk(GGML_TYPE_Q8_HP1, wf.data(), w->data, 0, (int64_t) N, (int64_t) K, nullptr);
        std::mt19937 rng2(seed + 1);
        float *      v = static_cast<float *>(x->data);
        for (size_t m = 0; m < M; ++m) {
            for (size_t k = 0; k < K; ++k) v[m * K + k] = nd(rng2) * (1.0f + 0.5f * (float) (m % 3)) * (k / 32 % 2 ? 4.0f : 1.0f);
            if (outlier != 0.0f && m % 2 == 0) v[m * K + (m * 7) % K] = outlier * (m % 4 ? -1.0f : 1.0f);
        }
        s = {(int64_t) M, (int64_t) N, (int64_t) K};
        ggml_gemmini_args_t args;
        const bool planned = frozen ? potal_frozen_plan(M, N, K, 1, 1, 64 / D, 32 / D, &plan) == POTAL_OK
                                    : potal_tile_select(M, N, K, 32 / D, &plan) == POTAL_OK;
        ok = planned && args_for(args) && weight.allocate((int64_t) N, (int64_t) K) &&
             prepare_weight(args, weight) == quants::weight::hp1::status::ok;
    }

    bool args_for(ggml_gemmini_args_t & args) const {
        return init_args(w, y, s, "gpt2", "hw-timeline-test", ggml::gemmini::residual::ResidualRoute::ws_packet, plan, args);
    }
    size_t stripes() const { return (M + plan.tile_I * D - 1) / (plan.tile_I * D); }
    float * dst() const { return static_cast<float *>(y->data); }
};

bool golden_of(const node & nd, std::vector<float> & want) {
    ggml_gemmini_args_t args;
    hp1_operands        o;
    return nd.args_for(args) && o.allocate(nd.s, nd.weight) && quantize_activations(nd.x, args, o) &&
           potal_test::pipeline_golden(args, o, want);
}

struct strict_run {
    pipeline_report                       report;
    std::vector<potal_paired_cmd_record>  records;
    std::vector<h2_emu::trace_record>     trace;
    std::vector<float>                    dst;
    potal_paired_counts                   sum{};
    hwn::stats_t                          stats{};
    std::vector<hwn::record>              windows;
    std::vector<hwn::stripe_record>       stripes;
    bool                                  dst_untouched = false;
};

const uint32_t sentinel_bits = 0x7fc0beefu;
bool ktile_route = true;  // the deployed strict route; the control without it merges several GEMMs per window
std::function<void(potal_paired_request &)> request_edit;  // one test's change to the request of the strict route

// One whole strict Linear node on the emulator. on: GGML_POTAL_STRICT_HW_TIMELINE=ON and its counter callback.
strict_run run_strict(node & nd, size_t capacity, bool on, bool host_clock = false,
                      const std::function<void(h2_emu &)> & inject = {}) {
    strict_run out;
    if (on) setenv("GGML_POTAL_STRICT_HW_TIMELINE", "ON", 1);
    else unsetenv("GGML_POTAL_STRICT_HW_TIMELINE");
    hwn::reset_for_test();
    ggml_gemmini_args_t args;
    if (!nd.args_for(args)) {
        out.report.stage = "fixture";
        return out;
    }
    float sentinel;
    std::memcpy(&sentinel, &sentinel_bits, sizeof(sentinel));
    std::fill(nd.dst(), nd.dst() + nd.M * nd.N, sentinel);
    pipeline_config cfg;
    cfg.job_capacity = capacity;
    cfg.residual = options::compensation;
    cfg.layer = "hw-timeline-test";
    cfg.backend = "EMULATED";
    cfg.measure_timing = true;
    cfg.hart_roles = true;
    cfg.paired = true;
    cfg.strict_issuer = true;
    cfg.paired_ktile_scale_reuse = ktile_route;
    auto h2 = std::make_unique<h2_emu>();
    h2->hw_trace_mode = on;
    h2->hw_trace.use_host_clock = host_clock;
    if (inject) inject(*h2);
    uint32_t generation = 0;
    potal_test::emu_session planned;
    paired_runner paired_exec = [&](const potal_paired_request & request, potal_paired_report & report) {
        auto q = request;
        q.generation = &generation;
        if (on) q.hw_counter = h2_emu::on_counter;
        if (request_edit) request_edit(q);
        const auto sink = h2->sink();
        const auto st = potal_paired_run_session(q, &sink, report);
        const potal_paired_counts & c = report.counts;
        out.sum.main_loop_ws += c.main_loop_ws;
        out.sum.residual_loop_ws += c.residual_loop_ws;
        out.sum.readbacks += c.readbacks;
        out.sum.scale_writes += c.scale_writes;
        out.sum.scale_releases += c.scale_releases;
        out.sum.index_configs += c.index_configs;
        out.sum.index_writes += c.index_writes;
        out.sum.fences += c.fences;
        return st;
    };
    potal_test::thread_owner owner([&planned](const potal_stripe_gemm_desc & d) { return planned.run(d); }, paired_exec);
    out.report = run_stripe_pipeline(nd.x, args, nd.weight, nd.plan, cfg, owner);
    owner.join();
    out.records = h2->records;
    out.trace = h2->trace_records;
    out.dst.assign(nd.dst(), nd.dst() + nd.M * nd.N);
    out.dst_untouched = std::all_of(nd.dst(), nd.dst() + nd.M * nd.N, [](float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return bits == sentinel_bits;
    });
    out.stats = hwn::stats();
    size_t n = 0, ns = 0;
    const hwn::record * w = hwn::windows(&n);
    const hwn::stripe_record * st = hwn::stripes(&ns);
    if (w != nullptr) out.windows.assign(w, w + n);
    if (st != nullptr) out.stripes.assign(st, st + ns);
    return out;
}

// As /tmp/h3-8i-25aa/tools/stream_hash_build.py masks addresses.
bool address_command(uint32_t funct) {
    return funct == POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB || funct == POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_DC ||
           funct == POTAL_FUNCT_MVOUT || funct == POTAL_FUNCT_MVIN;
}

bool same_records(const std::vector<potal_paired_cmd_record> & a, const std::vector<potal_paired_cmd_record> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const bool mask = address_command(a[i].funct);
        if (a[i].funct != b[i].funct || a[i].xd != b[i].xd || (!mask && (a[i].rs1 != b[i].rs1 || a[i].rs2 != b[i].rs2)))
            return false;
    }
    return true;
}

uint64_t fnv(uint64_t h, const void * p, size_t n) {
    const auto * b = static_cast<const uint8_t *>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

uint64_t records_hash(const std::vector<potal_paired_cmd_record> & r) {
    uint64_t h = 1469598103934665603ull;
    for (const auto & c : r) {
        const bool     mask = address_command(c.funct);
        const uint64_t words[3] = {(uint64_t(c.funct) << 1) | c.xd, mask ? 0 : c.rs1, mask ? 0 : c.rs2};
        h = fnv(h, words, sizeof(words));
    }
    return h;
}

uint64_t dst_hash(const std::vector<float> & d) { return fnv(1469598103934665603ull, d.data(), d.size() * sizeof(float)); }

bool main_with_dram_b(const potal_paired_cmd_record & r, uint64_t b_dram) {
    return r.funct == POTAL_FUNCT_LOOP_WS && ((r.rs1 >> 62) & 1) == 0 && ((r.rs1 >> 63) & 1) == 0 && b_dram != 0;
}

char trace_token(const potal_paired_cmd_record & c) {
    if (c.rs1 == POTAL_LDB_ONLY_TRACE_RESET) return 'A';
    if ((c.rs1 >> 5) == (POTAL_LDB_ONLY_TRACE_READ >> 5)) return 'R';
    if (c.rs1 == UINT64_C(0xA) || c.rs1 == POTAL_LDB_ONLY_TAG) return 'P';
    return 'C';
}

// Every trace command against the rules of the contract; the first violation, empty when all hold.
std::string trace_position_violation(const strict_run & r) {
    static const std::regex legal("^(R{1,12}A|R{1,12}C{10}((PP)?(C{8})?CA)?|(PP)?(C{8})?CA)$");
    bool pending_scale = false;
    size_t next_record = 0;
    for (size_t i = 0; i < r.trace.size();) {
        const size_t p = r.trace[i].position;
        std::string group;
        size_t j = i;
        for (; j < r.trace.size() && r.trace[j].position == p; ++j) {
            if (r.trace[j].in_flight) return "command at position " + std::to_string(p) + " with work in flight";
            group += trace_token(r.trace[j].cmd);
        }
        if (!std::regex_match(group, legal)) return "group " + group + " at position " + std::to_string(p);
        // Only Scale and Index ingress, which the drained term excludes, may sit between the Fence and a group.
        size_t back = p;
        while (back > 0 && (r.records[back - 1].funct == POTAL_FUNCT_SCALE_RELEASE ||
                            r.records[back - 1].funct == POTAL_FUNCT_INDEX_WRITE ||
                            r.records[back - 1].funct == POTAL_FUNCT_INDEX_CONFIG))
            --back;
        if (back > 0 && r.records[back - 1].funct != POTAL_FUNCT_FENCE)
            return "group at position " + std::to_string(p) + " follows funct " + std::to_string(r.records[back - 1].funct);
        for (; next_record < p; ++next_record) {
            if (r.records[next_record].funct == POTAL_FUNCT_SCALE_WRITE) pending_scale = true;
            if (r.records[next_record].funct == POTAL_FUNCT_LOOP_WS) pending_scale = false;
        }
        if (pending_scale && group.find('A') == std::string::npos)
            return "group " + group + " between a Scale write and its LOOP_WS without an arm";
        i = j;
    }
    return {};
}

struct window_tally { uint32_t main = 0, residual = 0, scale_w = 0, scale_r = 0, mvout = 0, cfg = 0, idx = 0, b_loads = 0; };

// One window per Fence, composition recounted from the emulator's commands: the executor's fields must match.
std::string window_violation(const strict_run & r, size_t * merged, size_t * single_residual, size_t * single_main) {
    std::vector<size_t> arms;
    for (const auto & t : r.trace)
        if (trace_token(t.cmd) == 'A') arms.push_back(t.position);
    if (arms.size() != r.windows.size()) return "arms " + std::to_string(arms.size()) + " windows " + std::to_string(r.windows.size());
    uint64_t b_dram = 0;
    size_t   pos = 0;
    for (size_t w = 0; w < arms.size(); ++w) {
        const hwn::record & rec = r.windows[w];
        for (; pos < arms[w]; ++pos)
            if (r.records[pos].funct == POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB) b_dram = r.records[pos].rs2;
        window_tally t;
        uint32_t loops_on_pair = 0, loads_on_pair = 0;
        for (; pos < r.records.size() && r.records[pos].funct != POTAL_FUNCT_FENCE; ++pos) {
            const potal_paired_cmd_record & c = r.records[pos];
            switch (c.funct) {
                case POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB: b_dram = c.rs2; break;
                case POTAL_FUNCT_LOOP_WS: {
                    const bool residual = ((c.rs1 >> 62) & 1) != 0;
                    (residual ? t.residual : t.main) += 1;
                    const uint32_t pair = uint32_t((c.rs1 >> 18) & 3) - 1;
                    if (pair == rec.pair) {
                        ++loops_on_pair;
                        if (main_with_dram_b(c, b_dram)) ++loads_on_pair;
                    }
                    break;
                }
                case POTAL_FUNCT_SCALE_WRITE: ++t.scale_w; break;
                case POTAL_FUNCT_SCALE_RELEASE: ++t.scale_r; break;
                case POTAL_FUNCT_INDEX_CONFIG: ++t.cfg; break;
                case POTAL_FUNCT_INDEX_WRITE: ++t.idx; break;
                case POTAL_FUNCT_MVOUT: ++t.mvout; break;
                default: break;
            }
        }
        if (pos == r.records.size()) return "window " + std::to_string(w) + " has no Fence";
        ++pos;
        const std::string at = " in window " + std::to_string(w);
        if (rec.main_loops != t.main || rec.residual_loops != t.residual) return "loop composition" + at;
        if (rec.scale_writes != t.scale_w || rec.scale_releases != t.scale_r) return "scale composition" + at;
        if (rec.index_configs != t.cfg || rec.index_writes != t.idx) return "index composition" + at;
        if ((rec.readbacks != 0) != (t.mvout != 0)) return "readback composition" + at;
        hwn::kind want = hwn::kind::unknown;
        hwn::confidence conf = hwn::confidence::none;
        if (t.main == 0 && t.residual == 0) want = hwn::kind::no_gemm;
        if (t.main == 1 && t.residual == 0) { want = hwn::kind::main; conf = hwn::confidence::single_gemm; ++*single_main; }
        if (t.main == 0 && t.residual == 1) { want = hwn::kind::residual; conf = hwn::confidence::single_gemm; ++*single_residual; }
        if (t.main == 1 && t.residual == 1) { want = hwn::kind::main_first_residual; conf = hwn::confidence::merged; ++*merged; }
        if (t.main == 0 && t.residual > 1) { want = hwn::kind::residual; conf = hwn::confidence::merged; }
        if (rec.kind != uint8_t(want) || rec.confidence != uint8_t(conf))
            return "kind " + std::to_string(rec.kind) + " confidence " + std::to_string(rec.confidence) + at + " main " +
                   std::to_string(t.main) + " residual " + std::to_string(t.residual);
        if ((rec.status & (hwn::st_not_frozen | hwn::st_overflow | hwn::st_read_failed | hwn::st_aborted)) != 0)
            return "status " + std::to_string(rec.status) + at;
        if (rec.cycle_read_end < rec.cycle_fence_after) return "read end before the Fence reply" + at;
        if ((rec.status & hwn::st_other_pair) == 0) {
            const bool mac_read = ((rec.read_mask >> hwn::f_mac_first) & 1u) != 0;
            if (mac_read != (loops_on_pair != 0) || (mac_read && rec.field[hwn::f_mac_cycles] != loops_on_pair))
                return "MAC stamps against the pair's loops" + at;
            const bool dma_read = ((rec.read_mask >> hwn::f_bdma_first) & 1u) != 0;
            if (dma_read != (loads_on_pair != 0)) return "B DMA stamps against the pair's loads" + at;
        }
    }
    return {};
}

std::string window_seq_violation(const strict_run & r) {
    size_t cursor = 0;
    for (const hwn::stripe_record & s : r.stripes) {
        for (uint32_t k = 0; k < s.windows; ++k, ++cursor) {
            if (cursor >= r.windows.size()) return "stripe " + std::to_string(s.stripe_id) + " beyond the windows";
            const hwn::record & w = r.windows[cursor];
            if (w.stripe_id != s.stripe_id || w.node_id != s.node_id || w.window_seq != k)
                return "window " + std::to_string(cursor) + " seq " + std::to_string(w.window_seq) + " wanted " + std::to_string(k);
        }
    }
    return cursor == r.windows.size() ? std::string() : "windows outside any stripe";
}

std::string budget_violation(const strict_run & r) {
    size_t reads = 0;
    for (const auto & t : r.trace) {
        const char k = trace_token(t.cmd);
        if (k == 'R') ++reads;
        if (k == 'A') {
            if (reads > 12) return "reads " + std::to_string(reads);
            if (reads + 1 > 13) return "commands";
            reads = 0;
        }
        if (k == 'C' && reads != 0) {
            if (reads > 12) return "reads " + std::to_string(reads);
            reads = 0;
        }
    }
    return reads > 12 ? "reads " + std::to_string(reads) : std::string();
}

bool counts_equal(const strict_run & r) {
    uint64_t main = 0, residual = 0, reads = 0, sw = 0, sr = 0, ic = 0, iw = 0;
    for (const auto & w : r.windows) {
        main += w.main_loops;
        residual += w.residual_loops;
        reads += w.readbacks;
        sw += w.scale_writes;
        sr += w.scale_releases;
        ic += w.index_configs;
        iw += w.index_writes;
    }
    return main == r.sum.main_loop_ws && residual == r.sum.residual_loop_ws && reads == r.sum.readbacks &&
           sw == r.sum.scale_writes && sr == r.sum.scale_releases && ic == r.sum.index_configs && iw == r.sum.index_writes;
}

void test_pipeline_shape(const char * label, node & nd, size_t capacity, size_t * merged, size_t * single_residual, size_t * single_main) {
    const std::string L = std::string("pipeline ") + label + " ";
    std::vector<float> want;
    const bool golden = golden_of(nd, want);
    const strict_run off = run_strict(nd, capacity, false);
    const strict_run on = run_strict(nd, capacity, true);
    const std::string kind_why = window_violation(on, merged, single_residual, single_main);
    const std::string pos_why = trace_position_violation(on);
    const std::string seq_why = window_seq_violation(on);
    const std::string budget_why = budget_violation(on);
    std::printf("     %s stripes=%zu records=%zu trace=%zu windows=%zu fences=%" PRIu64 " main=%" PRIu64 " residual=%" PRIu64 "\n",
                label, nd.stripes(), on.records.size(), on.trace.size(), on.windows.size(), uint64_t(on.sum.fences),
                uint64_t(on.sum.main_loop_ws), uint64_t(on.sum.residual_loop_ws));
    check(L + "ON committed the golden output", golden && on.report.ok && on.dst == want);
    check(L + "one window per fence_now Fence, windows == report fences",
          on.report.ok && on.windows.size() == on.sum.fences && on.stats.windows == on.sum.fences && !on.stats.disabled &&
              on.stats.reason == nullptr && on.windows.size() >= nd.stripes() && on.stripes.size() == nd.stripes());
    check(L + "window_seq contiguous per stripe" + (seq_why.empty() ? "" : " -- " + seq_why), seq_why.empty());
    check(L + "kinds and confidence match the issued commands" + (kind_why.empty() ? "" : " -- " + kind_why), kind_why.empty());
    check(L + "composition counts equal the sum of the executor report counters", counts_equal(on));
    check(L + "at most 12 reads and 13 commands per window from trace_records" + (budget_why.empty() ? "" : " -- " + budget_why),
          budget_why.empty());
    check(L + "every trace command at a legal position" + (pos_why.empty() ? "" : " -- " + pos_why), pos_why.empty());
    check(L + "stats commands equal trace_records", on.stats.commands == on.trace.size());
    check(L + "OFF sends no trace command and records no window",
          off.report.ok && off.trace.empty() && off.windows.empty() && off.stats.commands == 0);
    check(L + "OFF against ON: identical masked records", same_records(off.records, on.records));
    check(L + "OFF against ON: identical output bytes",
          off.dst.size() == on.dst.size() && std::memcmp(off.dst.data(), on.dst.data(), off.dst.size() * sizeof(float)) == 0);
    bool first_group = !on.trace.empty() && on.trace.front().position == 0;
    check(L + "first stripe: probe, counter configuration and reset come before any numeric command", first_group);
    uint64_t counter_loads = 0, counter_mac = 0;
    for (const auto & s : on.stripes) {
        counter_loads += s.counter[0];
        counter_mac += s.counter[1];
    }
    check(L + "stripe counters of events 9 and 24 equal the model's load and MAC pulses",
          counter_mac == on.sum.main_loop_ws + on.sum.residual_loop_ws && counter_loads <= on.sum.main_loop_ws &&
              (on.sum.main_loop_ws == 0 || counter_loads > 0));
}

void refill(const std::vector<hwn::record> & w, const std::vector<hwn::stripe_record> & s) {
    hwn::reset_for_test();
    for (const auto & r : w) *hwn::acquire_window() = r;
    for (const auto & r : s) *hwn::acquire_stripe() = r;
}

void test_selftest(node & nd) {
    const strict_run host = run_strict(nd, 2, true, true);
    const hwn::selftest_result live = hwn::selftest_evaluate();
    check(std::string("selftest passes on the host-clock model (MEASURED-NATIVE) -- ") +
              (live.first_failure ? live.first_failure : "no failure"),
          host.report.ok && live.first_failure == nullptr && live.reset_accepted && live.read_response && live.monotonic &&
              live.load_and_mac && live.ownership && live.alignment && live.no_overflow && live.mac_cycles > 0 &&
              live.alignment_inconsistent == 0);
    size_t idx = host.windows.size();
    for (size_t i = 0; i < host.windows.size(); ++i)
        if (((host.windows[i].read_mask >> hwn::f_mac_first) & 1u) != 0 && i > 0 &&
            host.windows[i - 1].stripe_id == host.windows[i].stripe_id) {
            idx = i;
            break;
        }
    check("selftest fixture has a MAC window that is not the first of its stripe", idx < host.windows.size());
    if (idx == host.windows.size()) return;
    refill(host.windows, host.stripes);
    check("selftest passes on the same records refilled (mutation control)", hwn::selftest_evaluate().first_failure == nullptr);
    const auto mutated = [&](const char * name, const std::function<void(std::vector<hwn::record> &)> & mutate,
                             const std::function<bool(const hwn::selftest_result &)> & failed) {
        std::vector<hwn::record> w = host.windows;
        mutate(w);
        refill(w, host.stripes);
        const hwn::selftest_result r = hwn::selftest_evaluate();
        check(std::string("selftest fails on mutated records: ") + name + " -- " + (r.first_failure ? r.first_failure : "passed"),
              r.first_failure != nullptr && failed(r));
    };
    mutated("not frozen", [&](auto & w) { w[idx].status |= hwn::st_not_frozen; }, [](const auto & r) { return !r.ownership; });
    mutated("overflow", [&](auto & w) { w[idx].status |= hwn::st_overflow; }, [](const auto & r) { return !r.no_overflow; });
    mutated("sequence gap", [&](auto & w) { w[idx].window_seq += 2; }, [](const auto & r) { return !r.ownership; });
    mutated("stamp order", [&](auto & w) { std::swap(w[idx].field[hwn::f_mac_first], w[idx].field[hwn::f_mac_last]);
                                            w[idx].field[hwn::f_mac_first] = w[idx].field[hwn::f_mac_last] + 1; },
            [](const auto & r) { return !r.monotonic; });
    mutated("reversed bracket", [&](auto & w) { w[idx].cycle_arm_after = w[idx].cycle_arm_before - 1; },
            [](const auto & r) { return !r.alignment; });
    mutated("no MAC", [&](auto & w) { for (auto & x : w) x.read_mask = uint16_t(x.read_mask & ~(uint16_t(1) << hwn::f_mac_first)); },
            [](const auto & r) { return !r.load_and_mac; });
    mutated("starts before the previous end", [&](auto & w) { w[idx].field[hwn::f_start] = w[idx - 1].field[hwn::f_end] - 1; },
            [](const auto & r) { return !r.monotonic; });
    mutated("activity in a window with no GEMM",
            [&](auto & w) { for (auto & x : w) if (x.kind == uint8_t(hwn::kind::no_gemm)) x.field[hwn::f_flags] |= hwn::flag_seen_mac0; },
            [](const auto & r) { return !r.ownership; });
    const strict_run counted = run_strict(nd, 2, true, false);
    const hwn::selftest_result det = hwn::selftest_evaluate();
    check("selftest rejects the deterministic model clock: its stamps match no host bracket",
          counted.report.ok && !det.alignment && det.alignment_inconsistent > 0);
}

// ---- failure injection ----------------------------------------------------------------------------

void test_failures(node & nd) {
    const strict_run off = run_strict(nd, 2, false);
    const auto scenario = [&](const char * name, const char * reason, const std::function<void(h2_emu &)> & inject) {
        const strict_run r = run_strict(nd, 2, true, false, inject);
        const std::string got = r.stats.reason != nullptr ? r.stats.reason : "none";
        std::printf("     injected %s: reason \"%s\" windows=%zu stats_commands=%" PRIu64 " trace_records=%zu\n", name, got.c_str(),
                    r.windows.size(), r.stats.commands, r.trace.size());
        check(std::string("failure ") + name + ": output and records equal the OFF run, disabled, reason " + reason + " (got " + got + ")",
              r.report.ok && r.stats.disabled && got == reason && same_records(off.records, r.records) &&
                  off.dst.size() == r.dst.size() &&
                  std::memcmp(off.dst.data(), r.dst.data(), off.dst.size() * sizeof(float)) == 0);
    };
    scenario("one TRACE_READ fails", "read status=-6",
             [](h2_emu & h) { h.hw_trace.fail_nth(cmd_class::trace_read, 5, POTAL_ERR_BAD_RESPONSE); });
    scenario("one arm fails", "arm status=7",
             [](h2_emu & h) { h.hw_trace.fail_nth(cmd_class::trace_reset, 2, POTAL_HW_BAD_COMMAND); });
    scenario("the first arm fails", "arm status=7",
             [](h2_emu & h) { h.hw_trace.fail_nth(cmd_class::trace_reset, 1, POTAL_HW_BAD_COMMAND); });
    scenario("reply 0 to one arm", "arm status=-5", [](h2_emu & h) { h.hw_trace.not_drained_nth_reset(3); });
    scenario("reply 2 to one arm", "arm status=-6",
             [](h2_emu & h) { h.hw_trace.override_reply_nth(cmd_class::trace_reset, 2, 2); });
    scenario("one counter read fails", "counter_read status=7",
             [](h2_emu & h) { h.hw_trace.fail_nth(cmd_class::counter_read, 2, POTAL_HW_BAD_COMMAND); });
    scenario("counter configuration fails", "counter_setup status=7",
             [](h2_emu & h) { h.hw_trace.fail_nth(cmd_class::counter_other, 4, POTAL_HW_BAD_COMMAND); });
    scenario("capability mismatch", "enable status=-2", [](h2_emu & h) { h.hw_trace.ldb_only_signature = 1; });

    const strict_run failed_read = run_strict(nd, 2, true, false,
        [](h2_emu & h) { h.hw_trace.fail_nth(cmd_class::trace_read, 5, POTAL_ERR_BAD_RESPONSE); });
    check("failure one TRACE_READ fails: its record says st_read_failed, with an unread field",
          !failed_read.windows.empty() && (failed_read.windows.back().status & hwn::st_read_failed) != 0 &&
              failed_read.windows.back().read_mask != 0xFFF);

    const auto route_case = [&](const char * name, const std::function<void(potal_paired_request &)> & edit) {
        request_edit = edit;
        const strict_run r = run_strict(nd, 2, true);
        request_edit = nullptr;
        const std::string reason = r.stats.reason != nullptr ? r.stats.reason : "none";
        std::printf("     %s route: ok=%d stage=%s reason \"%s\" trace_records=%zu\n", name, int(r.report.ok), r.report.stage ? r.report.stage : "-", reason.c_str(), r.trace.size());
        check(std::string("route: ") + name + " takes no trace command and disables with reason route",
              r.stats.disabled && reason.rfind("route", 0) == 0 && r.trace.empty() && r.windows.empty());
    };
    route_case("next K prefetch", [](potal_paired_request & q) { q.next_k_prefetch = true; });

    const strict_run plain = run_strict(nd, 2, true);
    size_t gemm = plain.windows.size();
    for (size_t i = 0; i < plain.windows.size(); ++i) {
        if (plain.windows[i].panel != hwn::no_id) {
            gemm = i;
            break;
        }
    }
    check("status cases have a window with a planned GEMM", gemm < plain.windows.size());
    if (gemm == plain.windows.size()) return;
    const uint64_t other_mac = plain.windows[gemm].pair != 0 ? hwn::flag_seen_mac0 : hwn::flag_seen_mac1;
    const auto flagged = [&](const char * name, uint64_t flags, uint16_t bit, const std::function<bool(const hwn::selftest_result &)> & failed) {
        const strict_run r = run_strict(nd, 2, true, false, [&](h2_emu & h) { h.hw_trace.override_flags_nth(gemm + 1, flags); });
        const hwn::selftest_result s = hwn::selftest_evaluate();
        const bool marked = gemm < r.windows.size() && (r.windows[gemm].status & bit) != 0;
        check(std::string("status ") + name + ": the record says so, output and records equal the OFF run",
              r.report.ok && marked && same_records(off.records, r.records) && off.dst.size() == r.dst.size() &&
                  std::memcmp(off.dst.data(), r.dst.data(), off.dst.size() * sizeof(float)) == 0);
        check(std::string("status ") + name + ": the selftest verdict", failed(s));
    };
    flagged("still armed", hwn::flag_armed | hwn::flag_seen_fence_accept, hwn::st_not_frozen, [](const auto & s) { return !s.ownership; });
    flagged("no accepted Fence", 0, hwn::st_not_frozen, [](const auto & s) { return !s.ownership; });
    flagged("overflow", hwn::flag_overflow | hwn::flag_seen_fence_accept, hwn::st_overflow, [](const auto & s) { return !s.no_overflow; });
    flagged("other pair", other_mac | hwn::flag_seen_fence_accept, hwn::st_other_pair, [](const auto &) { return true; });

    char self[PATH_MAX];
    const ssize_t len = readlink("/proc/self/exe", self, sizeof(self) - 1);
    self[len > 0 ? len : 0] = '\0';
    const std::string cmd = std::string("POTAL_HW_TIMELINE_WINDOWS=2 GGML_POTAL_STRICT_HW_TIMELINE=ON '") + self + "' --capacity-child";
    std::string lines;
    if (FILE * p = popen(cmd.c_str(), "r")) {
        char buf[512];
        while (std::fgets(buf, sizeof(buf), p)) lines += buf;
        const int rc = pclose(p);
        char want[256];
        std::snprintf(want, sizeof(want), "CAPACITY_CHILD ok=1 disabled=1 windows=2 records=%016" PRIx64 " dst=%016" PRIx64 " reason=window_capacity status=-1",
                      records_hash(off.records), dst_hash(off.dst));
        const bool match = rc == 0 && lines.find(want) != std::string::npos;
        if (!match) std::printf("     child rc=%d wanted \"%s\" got:\n%s", rc, want, lines.c_str());
        else std::printf("     injected capacity 2: reason \"window_capacity status=-1\"\n");
        check("failure capacity 2 via POTAL_HW_TIMELINE_WINDOWS: output and records equal the OFF run, disabled, reason window_capacity", match);
    } else {
        check("failure capacity 2: child started", false);
    }
}

// Last in the process, by contract: a Fence that did not answer leaves work outstanding, so close keeps the
// accelerator owned (potal_paired_session_close) and every later session open is refused.
void test_failing_fence(node & nd) {
    const strict_run off = run_strict(nd, 2, false);
    const strict_run fenced = run_strict(nd, 2, true, false, [](h2_emu & h) {
        h.fail_funct = POTAL_FUNCT_FENCE;
        h.fail_status = POTAL_HW_BAD_COMMAND;
    });
    const bool aborted = fenced.windows.size() == 1 && (fenced.windows[0].status & hwn::st_aborted) != 0 &&
                         fenced.windows[0].read_mask == 0;
    std::printf("     injected failing Fence: stage=%s windows=%zu aborted=%d disabled=%d\n",
                fenced.report.stage ? fenced.report.stage : "-", fenced.windows.size(), int(aborted), int(fenced.stats.disabled));
    check("failure failing Fence: aborted record, nothing read, existing error path, no partial commit",
          !fenced.report.ok && fenced.report.stage != nullptr && aborted && fenced.dst_untouched);
    check("failure failing Fence: the records stay a prefix of the OFF run with no later numeric work",
          fenced.records.size() <= off.records.size() && !fenced.records.empty() &&
              fenced.records.back().funct != POTAL_FUNCT_LOOP_WS);
    const strict_run after = run_strict(nd, 2, false);
    check("failure failing Fence: the accelerator stays owned, the next node is refused", !after.report.ok && after.records.empty());
}

// ---- flush ----------------------------------------------------------------------------------------

bool file_nonempty(const std::string & path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && st.st_size > 0;
}

bool flush_to(const std::string & path, const strict_run & r) {
    setenv("POTAL_HW_TIMELINE", path.c_str(), 1);
    const bool flushed = hwn::flush();
    unsetenv("POTAL_HW_TIMELINE");
    if (!flushed) return false;
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    uint32_t head[4] = {};
    uint64_t count = 0;
    bool ok = std::fread(head, sizeof(head), 1, f) == 1 && std::fread(&count, sizeof(count), 1, f) == 1;
    std::vector<hwn::record> back(static_cast<size_t>(count));
    ok = ok && (count == 0 || std::fread(back.data(), sizeof(hwn::record), back.size(), f) == back.size());
    std::fclose(f);
    return ok && head[0] == 0x54574850u && head[1] == hwn::format_version && head[2] == sizeof(hwn::record) &&
           head[3] == hwn::window_fields && count == r.windows.size() && !back.empty() &&
           std::memcmp(back.data(), r.windows.data(), back.size() * sizeof(hwn::record)) == 0 &&
           file_nonempty(path + ".stripes.csv") && file_nonempty(path + ".nodes.csv") && file_nonempty(path + ".meta.txt");
}

bool host_roles() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask) != 0) return false;
    std::string map;
    size_t count = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE && count < 5; ++cpu) {
        if (!CPU_ISSET(cpu, &mask)) continue;
        if (count++) map += ",";
        map += std::to_string(cpu);
    }
    if (count != 5) return false;
    setenv("GGML_POTAL_HART_MAP", map.c_str(), 1);
    // The deployed strict route, as /tmp/h3-8i-25y/tools/h0/counts-native.cpp sets it.
    for (const char * name : {"GGML_POTAL_HART_ROLES", "GGML_POTAL_PAIRED_KTILE", "GGML_POTAL_PLAN_LIVE_SCAN",
                              "GGML_POTAL_FAST_DECODE", "GGML_POTAL_PLAN_VALIDATED_REUSE"})
        setenv(name, "ON", 1);
    return true;
}

int finish() {
    std::printf("checks=%d failures=%d (values are MEASURED-NATIVE at most: no FPGA time here)\n", checks, failures);
    std::printf("POTAL_HW_TIMELINE_TEST %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (D != 32 || !options::compensation || options::activation_quant != options::ActivationQuantAlgo::EXSIA) {
        std::printf("[SKIP] the strict hardware timeline needs DIM32, compensation and ExSIA\nPOTAL_HW_TIMELINE_TEST PASS\n");
        return 0;
    }
    if (!host_roles()) {
        check("the native map has five available CPUs", false);
        return finish();
    }
    if (argc == 2 && std::strcmp(argv[1], "--capacity-child") == 0) {
        node nd(4 * D + D / 2, 64, 64, 191, 400.0f, true);
        const strict_run r = run_strict(nd, 2, true);
        std::printf("CAPACITY_CHILD ok=%d disabled=%d windows=%zu records=%016" PRIx64 " dst=%016" PRIx64 " reason=%s\n",
                    int(r.report.ok), int(r.stats.disabled), r.windows.size(), records_hash(r.records), dst_hash(r.dst),
                    r.stats.reason ? r.stats.reason : "none");
        return 0;
    }
    if (argc == 3 && std::strcmp(argv[1], "--sample") == 0) {
        node nd(200, 320, 128, 7, 400.0f, false);
        const strict_run r = run_strict(nd, 2, true, true);
        const hwn::selftest_result s = hwn::selftest_evaluate();
        const bool flushed = flush_to(argv[2], r);
        std::printf("SAMPLE ok=%d flushed=%d windows=%zu stripes=%zu selftest=%s\n", int(r.report.ok), int(flushed),
                    r.windows.size(), r.stripes.size(), s.first_failure ? s.first_failure : "pass");
        unsetenv("GGML_POTAL_STRICT_HW_TIMELINE");
        return r.report.ok && flushed && s.first_failure == nullptr ? 0 : 1;
    }

    test_model_window();
    test_model_counters();
    test_api();

    size_t merged = 0, single_residual = 0, single_main = 0;
    node five(4 * D + D / 2, 64, 64, 191, 400.0f, true);
    node big(200, 320, 128, 7, 400.0f, false);
    node two(D + D / 2, 64, 64, 177, 400.0f, true);
    check("pipeline fixtures", five.ok && big.ok && two.ok);
    if (five.ok && big.ok && two.ok) {
        test_pipeline_shape("frozen 5 stripes cap 2", five, 2, &merged, &single_residual, &single_main);
        test_pipeline_shape("frozen 2 stripes cap 1", two, 1, &merged, &single_residual, &single_main);
        test_pipeline_shape("selected 4 stripes cap 2", big, 2, &merged, &single_residual, &single_main);
        std::printf("     windows: main_first_residual=%zu residual=%zu main=%zu\n", merged, single_residual, single_main);
        check("pipeline shapes hold Main plus first Residual windows and single Residual windows", merged > 0 && single_residual > 0);

        ktile_route = false;
        size_t control_merged = 0, control_residual = 0, control_main = 0;
        test_pipeline_shape("no K-tile control", five, 2, &control_merged, &control_residual, &control_main);
        const strict_run control = run_strict(five, 2, true);
        const bool all_unknown = !control.windows.empty() &&
            std::all_of(control.windows.begin(), control.windows.end(), [](const hwn::record & w) {
                return w.main_loops == 0 || w.main_loops + w.residual_loops <= 2 ||
                       (w.kind == uint8_t(hwn::kind::unknown) && w.confidence == uint8_t(hwn::confidence::none));
            });
        check("no K-tile control: a window of several GEMMs with a Main has kind unknown, confidence none", all_unknown);
        ktile_route = true;

        // The shape of the FPGA self-test (potal-pipeline-bench in the staged run.sh), on the deployed route.
        node bench_shape(128, 768, 768, 20261104, 400.0f, false);
        check("bench shape fixture", bench_shape.ok);
        if (bench_shape.ok) {
            const strict_run dry = run_strict(bench_shape, 2, true, true);
            const hwn::selftest_result s = hwn::selftest_evaluate();
            size_t counts[4] = {0, 0, 0, 0}, unknown = 0;
            for (const auto & w : dry.windows) {
                if (w.kind == uint8_t(hwn::kind::unknown)) ++unknown;
                else ++counts[w.kind];
            }
            std::printf("     bench shape dry run: no_gemm=%zu main=%zu residual=%zu main_first_residual=%zu unknown=%zu windows=%zu stripes=%zu selftest=%s\n",
                        counts[0], counts[1], counts[2], counts[3], unknown, dry.windows.size(), dry.stripes.size(),
                        s.first_failure ? s.first_failure : "pass");
            check("selftest passes on the bench shape of the FPGA self-test (MEASURED-NATIVE)",
                  dry.report.ok && s.first_failure == nullptr && unknown == 0 && s.bdma_windows > 0);
        }

        test_selftest(big);
        test_failures(five);

        const strict_run r = run_strict(big, 2, true, true);
        char dir[] = "/tmp/potal-hw-timeline-XXXXXX";
        const bool made = mkdtemp(dir) != nullptr;
        std::printf("     flush run: ok=%d stage=%s windows=%zu stripes=%zu disabled=%d reason=%s\n", int(r.report.ok),
                    r.report.stage ? r.report.stage : "-", r.windows.size(), r.stripes.size(), int(r.stats.disabled),
                    r.stats.reason ? r.stats.reason : "none");
        check("flush with POTAL_HW_TIMELINE writes the binary and its three companions, and the records read back",
              made && flush_to(std::string(dir) + "/hw.bin", r));
        for (const char * suffix : {"", ".stripes.csv", ".nodes.csv", ".meta.txt"}) std::remove((std::string(dir) + "/hw.bin" + suffix).c_str());
        rmdir(dir);
        test_failing_fence(five);
    }
    unsetenv("GGML_POTAL_STRICT_HW_TIMELINE");
    return finish();
}
