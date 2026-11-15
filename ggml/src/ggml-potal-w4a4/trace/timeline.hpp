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
    issuer_stripe            = 22,  // inclusive: one stripe on the issuer, RECV to SEND
    prep_decode              = 23,  // children of 21
    prep_plan                = 24,
    prep_rc_alloc            = 25,
    exec_setup               = 26,  // inclusive: executor setup before the first command
    exec_plan_copy           = 27,  // children of 26
    exec_pack_residual       = 28,
    exec_tlb_flush           = 29,
    acc_scale_write          = 30,
    acc_index_scalar         = 31,
    acc_queue_install        = 32,
    acc_main_issue           = 33,
    acc_residual_issue       = 34,
    acc_fence                = 35,
    acc_readback             = 36,
    acc_scale_release        = 37,
    acc_prefetch_issue       = 38,
    exec_run                 = 39,  // inclusive: the op/action loop and final drain
    acc_scale_window         = 45,  // one Scale window; END.work = column writes
    acc_release_window       = 46,  // one release window; END.work = column releases
    acc_index_window         = 47,  // one Index op; END.work = entries
    acc_gemm_op              = 48,  // one Main/Residual LOOP_WS op incl. staging
    acc_readback_op          = 49,  // one ACC completion/readback op
    final_handoff            = 40,
    final_wait               = 41,
    final_reconstruct_int64  = 42,
    final_dequant_merge      = 43,
    final_slot_release       = 44,
    node_commit              = 50,
    node_setup               = 51,  // caller: stripe_run construction and issuer start
    exsia_role_setup         = 52,  // caller: ExSIA role block setup before thread spawn
    exsia_thread_spawn       = 53,  // caller: role thread creation
    role_start               = 54,  // each role thread: pin + register; both ends direct-stamped
    node_report              = 55,  // caller: report and completion poll after join
    gc_total                 = 56,  // caller: potal_graph_compute (inclusive; contains node.total)
    gc_setup                 = 57,  // caller: node validation and trace-id capture
    gc_node_setup            = 58,  // caller: per-node semantic/trace context
    mm_prep                  = 59,  // caller: mul_mat_hp1 prologue through weight-cache lookup
    pipe_options             = 60,  // caller: paired option parsing and pipeline config
    pipe_prenode             = 61,  // caller: run_stripe_pipeline checks before node.total
    role_cpu_map             = 62,  // caller: find_role_cpus (/proc/cpuinfo), inside 61
    mm_post                  = 63,  // caller: stats merge and records after node.total
    issuer_wait_job          = 64,  // strict issuer: waiting for a prepared job (wait events)
    issuer_wait_hw           = 65,  // strict issuer: blocked in a drain (fence, first release of a window)
    prep_job                 = 66,  // strict issuer, hart 3: prepare one stripe's job (inclusive)
    prep_validate            = 67,  // hart 3: potal_validate_execution_plan, inside 66
    // H3-8I-25T sub-stages (inclusive parents: 23 prep.decode, 24 prep.plan, 59 mm.prep)
    decode_validate          = 68,  // rmd::validate_packet inside residual::decode_packet
    decode_rows              = 69,  // run list + digit rows inside residual::decode_packet
    plan_tile                = 70,  // potal_make_tile_plan inside potal_paired_plan_stripe
    plan_exec                = 71,  // potal_make_execution_plan inside potal_paired_plan_stripe
    mm_options               = 72,  // caller: matmul options, layer name, checks, tile select (in 59)
    mm_init_args             = 73,  // caller: init_args (activation buffers) (in 59)
    mm_weights               = 74,  // caller: weight cache lookup (in 59)
    // H3-8I-25V strict attention pipeline (matmul/attention.cpp execute_strict) and model markers
    attn_local               = 75,  // hart 1/2: Q/P stripe rows: finite reads, ExSIA local stage (QK)
    attn_stationary          = 76,  // hart 3: K/V stationary quantization and limbs (prepare_weight)
    attn_fold                = 77,  // hart 3: anchor/shift or P scale+quantization, limbs, staging, plans
    attn_issue               = 78,  // hart 0: the planned GEMM jobs of one attention stripe
    attn_recon               = 79,  // hart 4: checked INT64 reconstruction and F32 output of one stripe
    attn_node                = 80,  // caller: one strict attention node, setup to publication
    model_inference          = 81,  // application: prompt evaluation plus generation
    model_prefill            = 82,  // application: prompt evaluation up to the first token
    model_decode             = 83,  // application: one decode step (stripe field = step index)
    model_token              = 84,  // application: instant, one sampled token (work = token id)
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
// POTAL_TIMELINE_PROFILE=detail: per Scale/Index entry spans (high overhead). Default: window.
extern bool g_detail;
// POTAL_TIMELINE_PROFILE=coarse: no per-command accelerator (acc.*) spans; see op_profile().
extern bool g_coarse;
// set_recording() gate (default on): off drops every record before any ring write.
extern std::atomic<bool> g_rec;
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

inline uint64_t mono_ns() noexcept {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

}  // namespace detail

// True when POTAL_TIMELINE was set (read once).
bool enabled() noexcept;

// Inline hot-path gates: the calling thread is registered (and the timeline is on).
inline bool active() noexcept {
    return detail::g_on && detail::t_ring != nullptr;
}

// Per-entry (Scale column, Index entry) spans are recorded only in the detail profile.
inline bool detail_profile() noexcept {
    return detail::g_detail && active();
}

// Per-op / per-window / per-command accelerator spans (acc.*): off in the coarse profile.
inline bool op_profile() noexcept {
    return active() && !detail::g_coarse;
}

// Recording gate (default on). Off: record()/record_mono() write nothing and count no
// unregistered record; register_thread, begin_run, end_run and anchors are unaffected.
void set_recording(bool on) noexcept;

// POTAL_TIMELINE_REUSE=1: register_thread reuses the slot (ring, anchors, runs) of an exited
// thread with the same role, hart and cpu; seq continues. Number of such reuses (0 without it).
uint32_t reuse_count() noexcept;

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

// mono != 0 forces a direct CLOCK_MONOTONIC stamp with that value (role_start begin).
inline void record_mono(stage st, event ev, uint8_t status, uint32_t run_id, uint32_t layer_id,
                        uint32_t stripe_id, uint32_t tile_id, uint32_t buffer_id, uint32_t edge_id,
                        uint64_t work, uint64_t mono) noexcept {
    if (__builtin_expect(!detail::g_rec.load(std::memory_order_relaxed), 0)) {
        return;
    }
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
    // The ring is demand-paged: store first, so a page fault precedes both clock reads below and
    // t_local stays adjacent to the CLOCK_MONOTONIC stamp (I25Y).
    e.seq          = h;
    e.t_local      = detail::now_local();
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
    if (mono != 0) {
        e.reserved = mono;
    } else if (r->direct_all || ev == event::wait_begin || ev == event::wait_end || ev == event::handoff_send ||
               ev == event::handoff_recv) {
        e.reserved = detail::mono_ns();
    } else {
        e.reserved = 0;
    }
    r->head.store(h + 1, std::memory_order_relaxed);
}

inline void record(stage st, event ev, uint8_t status, uint32_t run_id, uint32_t layer_id,
                   uint32_t stripe_id, uint32_t tile_id, uint32_t buffer_id, uint32_t edge_id,
                   uint64_t work) noexcept {
    record_mono(st, ev, status, run_id, layer_id, stripe_id, tile_id, buffer_id, edge_id, work, 0);
}

// CLOCK_MONOTONIC now if the timeline is on, else 0 (start mark before a thread registers).
inline uint64_t mono_if_on() noexcept {
    return detail::g_on ? detail::mono_ns() : 0;
}

// role_start span of a just-registered thread: BEGIN carries mono_begin (taken before pin and
// registration, when the thread had no ring), END carries now. Both ends are direct stamps.
inline void record_role_start(uint32_t run_id, uint32_t layer_id, uint64_t mono_begin) noexcept {
    if (!active() || mono_begin == 0) return;
    record_mono(stage::role_start, event::begin, 0, run_id, layer_id, no_id, no_id, no_id, 0, 0, mono_begin);
    record_mono(stage::role_start, event::end, 0, run_id, layer_id, no_id, no_id, no_id, 0, 0, detail::mono_ns());
}

// BEGIN/END pair on the calling registered thread; inert when `on` is false.
// close() ends the span early; END carries `work` and `status` as set by the owner.
struct scoped_span {
    bool     on;
    stage    st;
    uint32_t run_id, layer_id, stripe_id, tile_id;
    uint64_t work   = 0;
    uint8_t  status = 0;

    scoped_span(bool enable, stage s, uint32_t run, uint32_t layer, uint32_t stripe, uint32_t tile) noexcept :
        on(enable), st(s), run_id(run), layer_id(layer), stripe_id(stripe), tile_id(tile) {
        if (on) record(st, event::begin, 0, run_id, layer_id, stripe_id, tile_id, no_id, 0, 0);
    }
    void close() noexcept {
        if (on) record(st, event::end, status, run_id, layer_id, stripe_id, tile_id, no_id, 0, work);
        on = false;
    }
    ~scoped_span() { close(); }
    scoped_span(const scoped_span &)             = delete;
    scoped_span & operator=(const scoped_span &) = delete;
};

}  // namespace potal::timeline
