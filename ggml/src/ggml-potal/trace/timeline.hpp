// PoTal timeline trace (H3-8I-25P contract: /tmp/h3-8i-25p/timeline-schema.md).
//
// Opt-in and OFF by default: POTAL_TIMELINE=<path> (read once at load time)
// enables recording; unset, every call below is one predictable branch on a
// constant-initialized flag and records nothing.
//
// Each registered thread owns a preallocated ring of 64-byte events
// (POTAL_TIMELINE_EVENTS per thread, default 65536). record() never blocks,
// allocates, formats or does I/O; a full ring sets the thread's overflow flag
// and drops the event. t_local is rdcycle on RISC-V, steady_clock ns
// elsewhere. Anchors pair CLOCK_MONOTONIC ns with t_local; they are taken at
// register_thread, begin_run, end_run, anchor(), thread exit and (for the
// flushing thread) flush. Never take one inside a hart0 RoCC issue window.
//
// flush() writes the CSV (also registered with atexit). It requires the
// recording threads to be quiescent: call it after the measured graph, never
// inside a node.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>

namespace potal::timeline {

enum class stage : uint16_t {
    node_total               = 1,
    exsia_prepare            = 10,
    exsia_local              = 11,
    exsia_local_seal         = 12,
    exsia_mask_assembly      = 13,
    exsia_exponent_reduction = 14,
    exsia_folding            = 15,
    exsia_packet_seal        = 16,
    stripe_submit            = 17,
    issuer_ready_wait        = 20,
    stripe_residual_prepare  = 21,
    acc_scale_write          = 30,
    acc_index_scalar         = 31,
    acc_queue_install        = 32,
    acc_main_issue           = 33,
    acc_residual_issue       = 34,
    acc_fence                = 35,
    acc_readback             = 36,
    acc_scale_release        = 37,
    acc_prefetch_issue       = 38,
    final_handoff            = 40,
    final_wait               = 41,
    final_reconstruct_int64  = 42,
    final_dequant_merge      = 43,
    final_slot_release       = 44,
    node_commit              = 50,
};

enum class event : uint8_t {
    begin        = 1,
    end          = 2,
    wait_begin   = 3,
    wait_end     = 4,
    handoff_send = 5,
    handoff_recv = 6,
    instant      = 7,
};

enum class op_type : uint8_t { unknown = 0, linear = 1, qk = 2, pv = 3 };
enum class phase : uint8_t { unknown = 0, prefill = 1, decode = 2 };

enum class role : uint8_t {
    exsia_local0 = 0,
    exsia_local1 = 1,
    exsia_fold   = 2,
    issuer       = 3,
    finalizer    = 4,
    caller       = 5,
};

constexpr uint32_t no_id       = UINT32_MAX;  // stripe_id / tile_id / ... not applicable
constexpr uint16_t no_slot     = UINT16_MAX;  // register_thread when disabled or full
constexpr uint32_t max_threads = 1024;

// One event, exactly as stored in the ring and written by flush().
struct alignas(64) record_t {
    uint64_t t_local;
    uint32_t seq;
    uint16_t thread_slot;
    uint16_t stage;
    uint8_t  event_type;
    uint8_t  status;
    uint8_t  op_type;
    uint8_t  phase;
    uint32_t run_id;
    uint32_t layer_id;
    uint32_t stripe_id;
    uint32_t tile_id;
    uint32_t buffer_id;
    uint32_t edge_id;
    uint32_t pad0;  // zero (alignment of work)
    uint64_t work;
    uint64_t reserved;  // CLOCK_MONOTONIC ns for direct-stamped events (see record()), else zero
};
static_assert(sizeof(record_t) == 64, "timeline record must be 64 bytes");

namespace detail {

struct thread_ring {
    record_t *            events;
    uint32_t              capacity;
    std::atomic<uint32_t> head;  // owner thread stores (relaxed); flush reads after quiescence
    std::atomic<uint32_t> dropped;
    std::atomic<bool>     overflow;
    uint16_t              slot;
    uint8_t               cur_op;
    uint8_t               cur_phase;
    bool                  direct_all;  // unpinned thread: every event carries CLOCK_MONOTONIC
};

// Constant-initialized: false until the POTAL_TIMELINE load-time check enables it.
extern bool g_on;
extern thread_local thread_ring * t_ring;

void note_unregistered() noexcept;

inline uint64_t now_local() noexcept {
#if defined(__riscv)
    uint64_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
#else
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
#endif
}

}  // namespace detail

// True when POTAL_TIMELINE was set (read once).
bool enabled() noexcept;

// Registers the calling thread (idempotent per thread). hart_id / cpu_id are the
// expected placement (-1 = not pinned / unknown); the module records the actual
// sched_getcpu() CPU, its /proc/cpuinfo hart and whether the affinity equals
// {cpu_id}, and re-checks the CPU at thread exit and flush.
uint16_t register_thread(role r, int hart_id, int cpu_id) noexcept;

// One graph-node invocation; takes an anchor on the calling (registered) thread
// and sets that thread's op_type/phase for its subsequent records. 0 if disabled.
uint32_t begin_run(op_type op, phase ph, uint32_t layer_id) noexcept;
void     end_run(uint32_t run_id) noexcept;

// Process-unique handoff edge id (>0); 0 when disabled.
uint32_t next_edge_id() noexcept;

// Extra anchor on the calling thread. Never inside a RoCC issue window.
void anchor() noexcept;

// Writes the CSV to the POTAL_TIMELINE path (overwrites). Returns false when
// disabled (nothing written) or when the file cannot be written (one stderr line).
bool flush() noexcept;

struct stats_t {
    uint32_t threads;
    uint64_t events;
    uint64_t dropped;
    uint64_t unregistered;  // records from threads that never registered
    uint32_t overflowed_threads;
};
stats_t stats() noexcept;

inline void record(stage st, event ev, uint8_t status, uint32_t run_id, uint32_t layer_id,
                   uint32_t stripe_id, uint32_t tile_id, uint32_t buffer_id, uint32_t edge_id,
                   uint64_t work) noexcept {
    if (__builtin_expect(!detail::g_on, 1)) {
        return;
    }
    detail::thread_ring * r = detail::t_ring;
    if (r == nullptr) {
        detail::note_unregistered();
        return;
    }
    const uint32_t h = r->head.load(std::memory_order_relaxed);
    if (h >= r->capacity) {
        r->overflow.store(true, std::memory_order_relaxed);
        r->dropped.store(r->dropped.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        return;
    }
    record_t & e   = r->events[h];
    e.t_local      = detail::now_local();
    e.seq          = h;
    e.thread_slot  = r->slot;
    e.stage        = static_cast<uint16_t>(st);
    e.event_type   = static_cast<uint8_t>(ev);
    e.status       = status;
    e.op_type      = r->cur_op;
    e.phase        = r->cur_phase;
    e.run_id       = run_id;
    e.layer_id     = layer_id;
    e.stripe_id    = stripe_id;
    e.tile_id      = tile_id;
    e.buffer_id    = buffer_id;
    e.edge_id      = edge_id;
    e.pad0         = 0;
    e.work         = work;
    // Rocket stops rdcycle while a hart sits in WFI, so t_local cannot span a blocking
    // wait. Waits and handoffs (never inside a RoCC issue window) carry CLOCK_MONOTONIC.
    if (r->direct_all || ev == event::wait_begin || ev == event::wait_end || ev == event::handoff_send ||
        ev == event::handoff_recv) {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        e.reserved = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
    } else {
        e.reserved = 0;
    }
    r->head.store(h + 1, std::memory_order_relaxed);
}

}  // namespace potal::timeline
