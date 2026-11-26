// PoTal strict hardware timeline (H3-8I-25AC, /tmp/h3-8i-25ac/hardware-trace-api.md).
//
// Opt-in, diagnostic, OFF by default: GGML_POTAL_STRICT_HW_TIMELINE=ON|1. It applies to the strict
// Linear route only. The issuer re-arms the existing early-event window at each drained Fence and
// stores what the frozen window recorded. Only COUNTER_OP commands are added.
//
// POTAL_HW_TIMELINE=<path> writes <path> (binary records), <path>.stripes.csv, <path>.nodes.csv and
// <path>.meta.txt at flush() or exit. POTAL_HW_TIMELINE_WINDOWS sets the record capacity. The buffer
// is touched before use, so the issuer takes no first-touch page fault.
//
// A stamp is a first or last event cycle of the hardware clock. A count is a number of cycles. MAC
// stamps come from PE(0,0) only: they bound MAC activity, and the time between them is an envelope,
// not continuous array compute. There is no stamp for Activation load or Store.
#pragma once

#include <cstddef>
#include <cstdint>

namespace potal::hw_window {

constexpr uint32_t format_version = 1;
constexpr uint32_t window_fields  = 12;
constexpr uint32_t counter_count  = 8;
constexpr uint32_t no_id          = UINT32_MAX;

// Positions in record::field. Each is one TRACE_READ, at most 12 per window.
enum field_position : uint32_t {
    f_flags = 0,
    f_start,
    f_end,
    f_busy_first,
    f_busy_last,
    f_mac_first,
    f_mac_last,
    f_mac_cycles,
    f_bwrite_first,
    f_bwrite_last,
    f_bdma_first,
    f_bdma_last,
};

// TRACE_READ field numbers (CounterFile.scala values) for a position, by buffer pair.
struct field_map {
    uint32_t mac_first, mac_last, mac_cycles, bwrite_first, bwrite_last, bdma_first, bdma_last;
};
constexpr field_map pair_fields[2] = {{7, 8, 13, 3, 4, 24, 25}, {9, 10, 14, 5, 6, 26, 27}};
constexpr uint32_t trace_flags = 0, trace_busy_first = 20, trace_busy_last = 21, trace_start = 22, trace_end = 23;

// Bits of TRACE_READ field 0.
constexpr uint64_t flag_armed = 1u << 0, flag_overflow = 1u << 1, flag_seen_b0 = 1u << 9, flag_seen_b1 = 1u << 10,
                   flag_seen_mac0 = 1u << 11, flag_seen_mac1 = 1u << 12, flag_seen_fence_accept = 1u << 14,
                   flag_seen_busy = 1u << 15, flag_seen_dma0 = 1u << 16, flag_seen_dma1 = 1u << 17;

// Counter file events configured on the 8 physical counters. MAC per pair comes from the window count.
// 9 LOAD_ACTIVE, 24 EXE_ACTIVE, 12 STORE_ACTIVE, 5 MAIN_LD_EX, 4 MAIN_LD_ST, 6 MAIN_ST_EX,
// 7 MAIN_LD_ST_EX, 51 POTAL_CROSS_PAIR_OVERLAP.
constexpr uint32_t counter_events[counter_count] = {9, 24, 12, 5, 4, 6, 7, 51};

// From the issued command sequence between two drained points. INFERRED: no hardware field names it.
enum class kind : uint8_t { no_gemm = 0, main = 1, residual = 2, main_first_residual = 3, unknown = 255 };
enum class confidence : uint8_t { none = 0, merged = 1, single_gemm = 2 };

struct window_label {
    kind       k;
    confidence c;
};

// From the GEMM counts issued in a window. agrees: the plan's ops in the same range give the same counts.
constexpr window_label label_window(uint32_t main, uint32_t residual, bool agrees) {
    if (!agrees) return {kind::unknown, confidence::none};
    if (main == 0 && residual == 0) return {kind::no_gemm, confidence::none};
    if (main == 1 && residual == 0) return {kind::main, confidence::single_gemm};
    if (main == 0 && residual == 1) return {kind::residual, confidence::single_gemm};
    if (main == 1 && residual == 1) return {kind::main_first_residual, confidence::merged};
    // A Residual Scale window ends with a Release drain, which is no Fence, so its neighbor shares the window.
    if (main == 0) return {kind::residual, confidence::merged};
    return {kind::unknown, confidence::none};
}

enum status_bit : uint16_t {
    st_not_frozen  = 1,   // field 0: still armed, or no accepted Fence seen
    st_overflow    = 2,   // field 0: clock or count saturated
    st_other_pair  = 4,   // field 0: events on the pair this window did not plan; not read
    st_read_failed = 8,   // a TRACE_READ failed: fields outside read_mask are Missing
    st_aborted     = 16,  // the executor left with this window open: nothing was read
};

struct record {
    uint64_t cycle_arm_before, cycle_arm_after;      // H0 rdcycle around TRACE_RESET
    uint64_t cycle_fence_before, cycle_fence_after;  // H0 rdcycle around the closing Fence
    uint64_t cycle_read_end;                         // H0 rdcycle after the last TRACE_READ
    uint64_t field[window_fields];
    uint32_t node_id, timeline_run, layer_id, stripe_id, window_seq, first_op, last_op, j_group, panel;
    uint32_t main_loops, residual_loops, readbacks, scale_writes, scale_releases, index_configs, index_writes;
    uint16_t status, read_mask;  // read_mask bit p: field[p] was read. Clear with seen clear: no event
    uint8_t  phase, kind, confidence, pair;
    int32_t  cpu;
    uint32_t pad;
};
static_assert(sizeof(record) == 216, "the parser reads this layout");

struct stripe_record {
    uint64_t cycle_begin, cycle_end;  // H0 rdcycle after counter reset, after the counter reads
    uint64_t counter[counter_count];
    uint32_t node_id, timeline_run, stripe_id, windows, status, pad;
};

struct stats_t {
    uint64_t windows, stripes, nodes, dropped_windows, dropped_stripes, commands;
    bool     disabled;
    const char * reason;  // first failure, nullptr when none
};

// GGML_POTAL_STRICT_HW_TIMELINE, read per call. invalid: not ON, OFF, 1, 0 or unset.
bool requested(bool & invalid) noexcept;

// One Linear node on the strict route; returns its node_id (order of issue, from 1). 0 when disabled.
uint32_t begin_node(uint32_t timeline_run, uint32_t layer_id, uint8_t phase, uint32_t rows, const char * name) noexcept;

bool disabled() noexcept;
void note_failure(const char * stage, int status) noexcept;  // disables the trace; keeps the first reason
void count_commands(uint64_t n) noexcept;

// Issuer thread only. nullptr when full, which disables the trace (software trace overflow).
record *        acquire_window() noexcept;
stripe_record * acquire_stripe() noexcept;

// Once per process: capability probe and counter configuration.
bool take_probe() noexcept;
bool take_counter_config() noexcept;

stats_t stats() noexcept;
const record * windows(size_t * count) noexcept;
const stripe_record * stripes(size_t * count) noexcept;
bool flush() noexcept;

// Tests only: forget all records and process-once state.
void reset_for_test() noexcept;

// The nine fixture checks, on the records of one process.
struct selftest_result {
    bool     reset_accepted, read_response, monotonic, load_and_mac, ownership, alignment, no_overflow;
    uint64_t windows, mac_cycles, bdma_windows, alignment_width_max, alignment_inconsistent;
    const char * first_failure;
};
selftest_result selftest_evaluate() noexcept;

}  // namespace potal::hw_window
