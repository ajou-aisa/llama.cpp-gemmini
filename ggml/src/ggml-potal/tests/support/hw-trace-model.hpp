#pragma once

// Native model of the PoTal early-event window and counter file, for host tests of the strict hardware
// timeline (H3-8I-25AC). It states the registers and edge rules of CounterController and CounterFile
// (generators/potal/src/main/scala/potal/CounterFile.scala:173-245, 268-420) and the drained and Fence
// terms of Controller.scala:958-967, 986-999.
//
// One clock_edge() is one RTL clock edge. A command fires in one edge, with its reply sampled before the
// registers update. Nothing here models the latency, concurrency or duration of real work: the emulator
// stamps one fixed sequence per command, so every value is MEASURED-NATIVE at most, never a hardware time.

#include "potal.h"
#include "potal_hw_params.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace potal_test {

class hw_trace_model {
  public:
    static constexpr uint64_t all_ones = ~uint64_t(0);
    static constexpr uint32_t n_counters = 8, n_stamps = 10, n_counts = 7;

    // Indices of seen, first and last: the timestamps list of CounterFile.scala.
    enum stamp : uint32_t { e_admission, e_b0, e_b1, e_mac0, e_mac1, e_fence_request, e_fence_accepted,
                            e_not_drained, e_dma0, e_dma1 };
    // CounterEvent ids driven by this model. The POTAL_* ones follow PoTalPrefetchObservation.connect.
    enum counter_event : uint32_t { ev_load_active = 9, ev_store_active = 12, ev_exe_active = 24,
                                    ev_b0 = 45, ev_b1 = 46, ev_mac0 = 47, ev_mac1 = 48,
                                    ev_mac0_b1 = 49, ev_mac1_b0 = 50, ev_cross = 51, n_events = 52 };
    enum class cmd_class : uint32_t { trace_reset, trace_read, counter_read, counter_other, count };

    struct signals {
        bool     admission = false;
        uint32_t admission_pair = 0;
        bool     b_write[2] = {false, false};
        bool     dma_request[2] = {false, false};
        bool     mac[2] = {false, false};
        bool     fence_request = false, fence_accepted = false;
        uint64_t events = 0;  // further counter file event pulses, bit = CounterEvent id
    };

    // The window registers. clockCycle is apart: it moves on every edge, so comparing these shows state change.
    struct registers {
        bool     armed = false, overflow = false;
        uint64_t start = 0, end = 0;
        uint32_t pair = 0;
        bool     seen[n_stamps] = {};
        uint64_t first[n_stamps] = {}, last[n_stamps] = {};
        uint64_t counts[n_counts] = {};

        bool operator==(const registers & o) const {
            return armed == o.armed && overflow == o.overflow && start == o.start && end == o.end && pair == o.pair &&
                   std::equal(seen, seen + n_stamps, o.seen) && std::equal(first, first + n_stamps, o.first) &&
                   std::equal(last, last + n_stamps, o.last) && std::equal(counts, counts + n_counts, o.counts);
        }
    };

    struct counter_file {
        uint64_t counters[n_counters] = {}, snapshot[n_counters] = {};
        uint32_t config[n_counters] = {};
        bool     is_external[n_counters] = {};
        bool     snapshot_enable = false;
        uint64_t external_values[8] = {};  // CounterExternal: inputs, zero unless a test sets them
    };

    registers    regs;
    counter_file file;
    uint64_t     clock_cycle = 0;
    uint64_t     cycles_per_step = 1;  // RTL: 1. A test may use more for readable stamps
    bool         use_host_clock = false;  // clock_cycle follows potal_paired_hw_cycle(), so H0 brackets hold
    bool         ldb_only_supported = POTAL_HW_DIM == 32;
    uint64_t     ldb_only_signature = POTAL_LDB_ONLY_CAPABILITY;
    uint64_t     stream_signature = POTAL_STREAM_CAPABILITY;
    uint64_t     stream_early_signature = POTAL_STREAM_EARLY_CAPABILITY;
    uint64_t     observation_signature = UINT64_C(0x504f425301200007);
    uint64_t     resident_signature = UINT64_C(0x504f544102100007);

    // ---- Drained state --------------------------------------------------------------------------------

    bool work_in_flight() const { return in_flight_; }
    bool drained() const { return !in_flight_ && !forced_not_drained_; }

    // ---- Event entry points ---------------------------------------------------------------------------

    void step(const signals & s) { clock_edge(s, nullptr, nullptr, false); }
    void step() { step(signals{}); }

    // A command the drained term sees as busy, and optional counter file pulses.
    void work_begin(uint64_t events = 0) {
        in_flight_ = true;
        signals s;
        s.events = events;
        step(s);
    }

    void dma_request(uint32_t pair) {
        signals s;
        s.dma_request[pair] = true;
        s.events = event_bit(ev_load_active);
        step(s);
    }

    void b_write(uint32_t pair) {
        signals s;
        s.b_write[pair] = true;
        step(s);
    }

    void mac(uint32_t pair) {
        signals s;
        s.mac[pair] = true;
        s.events = event_bit(ev_exe_active);
        step(s);
    }

    void fence_request() {
        signals s;
        s.fence_request = true;
        step(s);
    }

    // ingress.cmd.fire needs allDrained: the Fence waited out all work, so this edge is drained.
    void fence_accepted() {
        in_flight_ = false;
        forced_not_drained_ = false;
        signals s;
        s.fence_request = true;
        s.fence_accepted = true;
        step(s);
    }

    // One whole Fence. Only xd=1 with rs1=rs2=0 is accepted as earlyTraceFenceAccepted.
    void fence(uint32_t xd, uint64_t rs1, uint64_t rs2) {
        if (xd == 1 && rs1 == 0 && rs2 == 0) {
            if (!drained()) fence_request();
            fence_accepted();
        } else {
            // PoTalScaleCommandIngress answers at once with INVALID_ENCODING: it waits for nothing, so work stays in flight.
            fence_request();
        }
    }

    // One funct 126 command: reply sampled at fire, registers updated on the same edge.
    potal_status counter_command(const potal_paired_cmd_record & cmd, uint64_t * value) {
        const uint32_t k = uint32_t(classify(cmd));
        ++seen_commands_[k];
        if (fail_nth_[k] != 0 && seen_commands_[k] == fail_nth_[k]) {
            fail_nth_[k] = 0;
            return fail_status_[k];
        }
        bool force_this = false;
        if (cmd_class(k) == cmd_class::trace_reset && not_drained_nth_reset_ != 0 &&
            seen_commands_[k] == not_drained_nth_reset_) {
            not_drained_nth_reset_ = 0;
            force_this = true;
        }
        clock_edge(signals{}, &cmd, value, force_this);
        if (override_nth_[k] != 0 && seen_commands_[k] == override_nth_[k]) {
            override_nth_[k] = 0;
            *value = override_value_[k];
        }
        if (flags_override_nth_ != 0 && cmd.rs1 == POTAL_LDB_ONLY_TRACE_READ && --flags_override_nth_ == 0) *value = flags_override_value_;
        return POTAL_OK;
    }

    // ---- Injection hooks ------------------------------------------------------------------------------

    void saturate_clock() { clock_cycle = all_ones; }
    void saturate_count(uint32_t index) { regs.counts[index] = all_ones; }
    // Hidden busy work: holds drained low until a Fence waits it out.
    void force_not_drained(bool on = true) { forced_not_drained_ = on; }
    // The n-th (from 1, counted over this model's life) command of a class fails without reaching the RTL state.
    void fail_nth(cmd_class c, uint64_t n, potal_status status) {
        fail_nth_[uint32_t(c)] = seen_commands_[uint32_t(c)] + n;
        fail_status_[uint32_t(c)] = status;
    }
    // The n-th TRACE_RESET from now fires while not drained: it replies 0 and keeps the stamps.
    void not_drained_nth_reset(uint64_t n) { not_drained_nth_reset_ = seen_commands_[0] + n; }
    // A reply the RTL cannot give, for the API's response check.
    void override_reply_nth(cmd_class c, uint64_t n, uint64_t value) {
        override_nth_[uint32_t(c)] = seen_commands_[uint32_t(c)] + n;
        override_value_[uint32_t(c)] = value;
    }

    // The n-th read of trace field 0 from now returns value: a window state the executor must report.
    void override_flags_nth(uint64_t n, uint64_t value) {
        flags_override_nth_ = n;
        flags_override_value_ = value;
    }

    // ---- Read-only views ------------------------------------------------------------------------------

    // values(i) as the RTL drives it this cycle; a test's view, no edge, no state change.
    uint64_t peek(uint32_t i) const {
        const registers & r = regs;
        uint64_t seen_bits = 0;
        for (uint32_t b = 0; b < n_stamps; ++b) seen_bits |= uint64_t(r.seen[b]) << b;
        switch (i) {
            case 0: return uint64_t(r.armed) | (uint64_t(r.overflow) << 1) | (seen_bits << 8) | (uint64_t(r.pair) << 20);
            case 1: return (r.armed ? clock_cycle : r.end) - r.start;
            case 2: return r.first[e_admission];
            case 3: return r.first[e_b0];
            case 4: return r.last[e_b0];
            case 5: return r.first[e_b1];
            case 6: return r.last[e_b1];
            case 7: return r.first[e_mac0];
            case 8: return r.last[e_mac0];
            case 9: return r.first[e_mac1];
            case 10: return r.last[e_mac1];
            case 11: case 12: case 13: case 14: case 15: case 16: case 17: return r.counts[i - 11];
            case 18: return r.first[e_fence_request];
            case 19: return r.first[e_fence_accepted];
            case 20: return r.first[e_not_drained];
            case 21: return r.last[e_not_drained];
            case 22: return r.start;
            case 23: return r.end;
            case 24: return r.first[e_dma0];
            case 25: return r.last[e_dma0];
            case 26: return r.first[e_dma1];
            case 27: return r.last[e_dma1];
            default: return 0;  // 28-31: values.foreach(_ := 0.U)
        }
    }

    static bool is_trace_reset(const potal_paired_cmd_record & c) {
        return c.funct == POTAL_FUNCT_COUNTER && c.xd == 1 && c.rs2 == 0 && c.rs1 == POTAL_LDB_ONLY_TRACE_RESET;
    }
    static bool is_trace_read(const potal_paired_cmd_record & c) {
        return c.funct == POTAL_FUNCT_COUNTER && c.xd == 1 && c.rs2 == 0 && (c.rs1 >> 5) == (POTAL_LDB_ONLY_TRACE_READ >> 5);
    }

  private:
    bool     in_flight_ = false, forced_not_drained_ = false;
    uint64_t seen_commands_[uint32_t(cmd_class::count)] = {};
    uint64_t fail_nth_[uint32_t(cmd_class::count)] = {};
    potal_status fail_status_[uint32_t(cmd_class::count)] = {};
    uint64_t override_nth_[uint32_t(cmd_class::count)] = {};
    uint64_t flags_override_nth_ = 0, flags_override_value_ = 0;
    uint64_t override_value_[uint32_t(cmd_class::count)] = {};
    uint64_t not_drained_nth_reset_ = 0;

    static constexpr uint64_t event_bit(uint32_t id) { return uint64_t(1) << id; }

    static cmd_class classify(const potal_paired_cmd_record & c) {
        if (is_trace_reset(c)) return cmd_class::trace_reset;
        if (is_trace_read(c)) return cmd_class::trace_read;
        if ((c.rs1 >> 7) == 0 && (c.rs1 & 0xF) == 0) return cmd_class::counter_read;
        return cmd_class::counter_other;
    }

    bool capability_query(const potal_paired_cmd_record & c, uint64_t * signature) const {
        if (c.funct != POTAL_FUNCT_COUNTER || c.xd != 1 || c.rs2 != 0 || POTAL_HW_DIM != 32) return false;
        if (c.rs1 == POTAL_STREAM_EARLY_TAG) *signature = stream_early_signature;
        else if (c.rs1 == POTAL_STREAM_TAG) *signature = stream_signature;
        else if (c.rs1 == POTAL_LDB_ONLY_TAG && ldb_only_supported) *signature = ldb_only_signature;
        else if (c.rs1 == UINT64_C(0x504f425300000000)) *signature = observation_signature;
        else if (c.rs1 == UINT64_C(0x504f544100000000)) *signature = resident_signature;
        else return false;
        return true;
    }

    // take_value of CounterFile: is_external is read at the addressed counter for every counter, as in the RTL.
    uint64_t take_value(uint32_t config, uint64_t counter, uint32_t addr) const {
        return file.is_external[addr] ? file.external_values[config & 7] : counter;
    }

    void clock_edge(const signals & in, const potal_paired_cmd_record * cmd, uint64_t * reply, bool force_not_drained_now) {
        if (use_host_clock && clock_cycle != all_ones) clock_cycle = std::max(clock_cycle, potal_paired_hw_cycle());
        const uint64_t cycle = clock_cycle;
        const bool     trace_reset = cmd != nullptr && ldb_only_supported && is_trace_reset(*cmd);
        const bool     trace_read  = cmd != nullptr && ldb_only_supported && is_trace_read(*cmd);
        // PoTalTraceReset.drained: only the canonical reset's own head is exempt.
        const bool other_drained = drained() && !force_not_drained_now;
        const bool events_drained = other_drained && (cmd == nullptr || trace_reset);
        uint64_t   signature = 0;
        const bool capability = cmd != nullptr && capability_query(*cmd, &signature);
        const bool capability_read = capability || trace_reset || trace_read;
        const uint32_t addr = cmd != nullptr ? uint32_t((cmd->rs1 >> 4) & 7) : 0;

        // Reply sampled at fire: everything below updates after this.
        if (cmd != nullptr && reply != nullptr) {
            if (trace_reset) {
                *reply = events_drained ? 1 : 0;
            } else if (trace_read) {
                *reply = peek(uint32_t(cmd->rs1 & 31));
            } else if (capability) {
                *reply = signature;
            } else {
                *reply = file.snapshot_enable ? file.snapshot[addr]
                                              : take_value(file.config[addr], file.counters[addr], addr);
            }
        }

        const bool cross0 = in.mac[0] && in.b_write[1], cross1 = in.mac[1] && in.b_write[0];
        if (trace_reset && events_drained) {
            registers next;
            next.armed = true;
            next.overflow = cycle == all_ones;
            next.start = cycle;
            next.end = cycle;
            regs = next;
        } else if (regs.armed) {
            if (cycle == all_ones) regs.overflow = true;
            const bool timestamps[n_stamps] = {in.admission, in.b_write[0], in.b_write[1], in.mac[0], in.mac[1],
                                               in.fence_request, in.fence_accepted, !events_drained,
                                               in.dma_request[0], in.dma_request[1]};
            for (uint32_t i = 0; i < n_stamps; ++i) {
                if (!timestamps[i]) continue;
                if (!regs.seen[i]) regs.first[i] = cycle;
                regs.seen[i] = true;
                regs.last[i] = cycle;
            }
            if (in.admission) regs.pair = in.admission_pair;
            const bool increments[n_counts] = {in.b_write[0], in.b_write[1], in.mac[0], in.mac[1], cross0, cross1,
                                               cross0 || cross1};
            for (uint32_t i = 0; i < n_counts; ++i) {
                if (!increments[i]) continue;
                if (regs.counts[i] == all_ones) regs.overflow = true;
                else ++regs.counts[i];
            }
            if (in.fence_accepted) {
                regs.armed = false;
                regs.end = cycle;
            }
        }

        uint64_t events = in.events;
        if (in.b_write[0]) events |= event_bit(ev_b0);
        if (in.b_write[1]) events |= event_bit(ev_b1);
        if (in.mac[0]) events |= event_bit(ev_mac0);
        if (in.mac[1]) events |= event_bit(ev_mac1);
        if (cross0) events |= event_bit(ev_mac0_b1);
        if (cross1) events |= event_bit(ev_mac1_b0);
        if (cross0 || cross1) events |= event_bit(ev_cross);
        const uint64_t rs1 = cmd != nullptr && !capability_read ? cmd->rs1 : 0;
        counter_file_edge((rs1 & 1) != 0, (rs1 & 2) != 0, (rs1 & 4) != 0, (rs1 & 8) != 0, uint32_t((rs1 >> 12) & 0x3F),
                          ((rs1 >> 31) & 1) != 0, addr, events);

        if (!use_host_clock && clock_cycle != all_ones)
            clock_cycle = cycles_per_step > all_ones - clock_cycle ? all_ones : clock_cycle + cycles_per_step;
    }

    void counter_file_edge(bool counter_reset, bool snapshot_reset, bool snapshot, bool config_valid, uint32_t config_bits,
                           bool external, uint32_t addr, uint64_t events) {
        counter_file before = file;
        if (snapshot_reset) {
            file.snapshot_enable = false;
        } else if (snapshot) {
            file.snapshot_enable = true;
            for (uint32_t i = 0; i < n_counters; ++i)
                file.snapshot[i] = take_value(before.config[i], before.counters[i], addr);
        }
        if (config_valid) {
            file.config[addr] = config_bits;
            file.is_external[addr] = external;
            file.counters[addr] = 0;
        }
        for (uint32_t i = 0; i < n_counters; ++i) {
            const uint32_t config = before.config[i];
            const bool     fired = config < n_events && ((events >> config) & 1) != 0;
            if (fired && !(config_valid && addr == i)) ++file.counters[i];
        }
        if (counter_reset) {  // withReset(reset || counter_reset): config and is_external stay
            for (uint32_t i = 0; i < n_counters; ++i) file.counters[i] = file.snapshot[i] = 0;
            file.snapshot_enable = false;
            for (uint64_t & external_value : file.external_values) external_value = 0;
        }
    }
};

}  // namespace potal_test
