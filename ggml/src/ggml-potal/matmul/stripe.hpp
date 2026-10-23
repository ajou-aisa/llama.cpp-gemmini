#pragma once

// Stripe pipeline of one F32 x Q8_HP1 MUL_MAT (GGML_GEMMINI_DEFAULT_MATMUL_MODE
// or GEMMINI_MATMUL_MODE = STRIPE_PIPELINE, ExSIA activations), after the
// IM2P.sim frontend PIPELINE contract (docs/HOST_STRIPE_CONTRACT.md):
//
//   producer: calling thread                consumer: accelerator owner (hart-0 issuer)
//   ExSIA stripe s: local adaptation,
//   folding + residual decomposition
//   -> StripeReadySink::on_ready(s)
//   -> submit(s): wait for capacity,         Main GEMM(s)      frozen Main plan, PoTal
//      copy A rows + snapshot, READY ------> Residual GEMM(s)  packet -> K32 lowering, PoTal
//   ExSIA stripe s+1 ...                     Main F32 reconstruction + radix INT64
//                                            reconstruction + checked merge (CPU)
//                                            SEMANTIC_DONE(s): capacity back
//   finish(): every stripe semantically done
//   commit: private output -> dst (success only)
//
// The Main tile plan is selected once, before ExSIA starts, and frozen; a
// tail stripe only limits tile_I. Each stripe's Residual plan is selected once
// when its packet is turned into a request. Order on the one accelerator:
// Main(s) -> Residual(s) -> reconstruction/merge(s) -> Main(s+1); ExSIA keeps
// quantizing stripe s+1 (and the one after, if a slot is free) meanwhile.
//
// Capacity: at most job_capacity (1 or 2) semantic stripes. A slot holds
// capacity in every state but FREE:
//   FREE -> RESERVED -> READY -> MAIN_RUNNING -> MAIN_DONE -> RESIDUAL_PREPARE
//   -> RESIDUAL_RUNNING -> RESIDUAL_DONE -> RECONSTRUCT_MERGE -> SEMANTIC_DONE -> FREE
// (no packet: MAIN_DONE -> RECONSTRUCT_MERGE). A raw Main completion never
// returns capacity. A full pipeline makes submit() wait on a condition
// variable (the mutex released), never fail: on_ready returns false only for
// a failed or aborted run. Pool slot index, stripe id, the slot's reuse
// generation and the 8-bit SCU generation are four different numbers.
//
// Accepted data is the run's own: the stripe's A codes are copied into the
// slot's DMA buffer (ExSIA's workspace and args.A may be reused or zeroed
// later), the post-fold metadata is a snapshot, the residual packet a shared
// immutable handle. The consumer never reads the producer's args, metadata
// or ExSIA state.
//
// Failure: the first error of a run is sticky. It wakes a producer waiting for
// capacity and a finish() waiter; no stripe starts after it and the private
// output is discarded (the caller poisons dst, ops.cpp). A failure with the
// accelerator drained (software/metadata errors, a refused descriptor) leaves
// the buffers freeable; a fatal one (hardware state unknown) makes the caller
// abandon() them, and the issuer disables itself. An abort cannot interrupt a
// GEMM that is already running: the hardware FENCE blocks until it drains.
//
// Records: each step is one stripe_event, stamped under the run's lock in the
// same critical section as the state change it reports (READY + queue push ->
// submit_accepted, queue pop -> main_begin, slot FREE -> capacity_release),
// with a per-run sequence 1, 2, ... and the CPU time of the step: sequence is
// the logical order of the run (submit_accepted(s) < main_begin(s),
// capacity_release(s) < submit_accepted(s + job_capacity)). The debug JSONL
// records are written in sequence order: a writer waits only for the record
// of the step stamped before its own, which its thread writes next without
// blocking, so the wait never depends on future steps and neither side waits
// for the other's work. Observers run after the step's record, outside every
// lock of the run, on the step's thread: per thread in sequence order, across
// the producer and the consumer ordered by sequence only. A record or observer
// that fails fails the run like any other step; after capacity_release the
// failure no longer touches the slot (it may be the producer's again) or
// counts the stripe's work a second time.

#include "dense.hpp"
#include "execution.hpp"
#include "paired-auto.hpp"
#include "tiling.hpp"

#include "quants/act/exsia/exsia-event.hpp"
#include "residual/rmd/rmd-types.hpp"

#include <gemmini/trace-context.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ggml::potal {

namespace exsia = ggml::gemmini::quants::act::exsia;

enum class slot_state : uint8_t {
    free,
    reserved,
    ready,
    main_running,
    main_done,
    residual_prepare,
    residual_running,
    residual_done,
    reconstruct_merge,
    semantic_done,
    // Paired route (pipeline_config::paired): MAIN_RUNNING -> PAIRED_PREPARE ->
    // PAIRED_RUNNING -> PAIRED_DONE -> RECONSTRUCT_MERGE, or back to MAIN_RUNNING
    // for a stripe whose paired plan was refused (the serial route runs it).
    paired_prepare,
    paired_running,
    paired_done,
};

const char * slot_state_name(slot_state s);

// Observable steps of a run: one debug JSONL record each ([POTAL_STRIPE]) and
// the observer of pipeline_config.
enum class stripe_event_kind : uint8_t {
    quant_begin,
    folding_commit,
    queue_wait, // submit found the pipeline full and waits for capacity
    submit_accepted,
    main_begin,
    main_end,
    residual_prepare_begin,
    residual_prepare_end,
    residual_begin,
    residual_end,
    radix_reconstruct_begin,
    radix_reconstruct_end,
    merge_begin,
    merge_end,
    semantic_complete,
    capacity_release,
    quant_end,
    run_commit,
    run_failed,
    // Paired route only. paired_prepare_*: decode of the packet and the paired
    // plans (CPU, before any command); its end says the route (paired, or
    // serial_fallback with the refusal). paired_begin / paired_end: the one
    // hardware stream in which Main and Residual GEMMs interleave; no
    // main_end / residual_* events are emitted for a paired stripe (their
    // order on the accelerator is not sequential).
    paired_prepare_begin,
    paired_prepare_end,
    paired_begin,
    paired_end,
};

const char * stripe_event_name(stripe_event_kind k);

constexpr size_t no_stripe = SIZE_MAX;

struct stripe_event {
    stripe_event_kind kind;
    uint64_t          sequence        = 0;     // logical order of the step in its run: 1, 2, ... (no gaps)
    bool              run_bound       = false; // run_id known (the first stripe was accepted)
    uint64_t          run_id          = 0;     // ExSIA run id (0 is valid)
    size_t            stripe_id       = no_stripe;
    size_t            slot            = no_stripe; // pool slot index
    uint64_t          slot_generation = 0;         // reuse count of that slot
    size_t            row_begin       = 0;
    size_t            row_end         = 0;
    size_t            outstanding     = 0; // semantic stripes held right after this step
    uint64_t          t_ns            = 0; // steady clock at the step: a CPU observation, not a device cycle

    // Views valid during the observer call only.
    const potal_stripe_gemm_report * gemm = nullptr;     // main_end, residual_end
    const int32_t *                  c    = nullptr;     // main_end (rows = stripe rows), residual_end (lane rows)
    size_t                           ldc  = 0;
    size_t                           rows = 0;
    size_t                           cols = 0;
    size_t                           k_compact   = 0;    // residual_prepare_end, residual_end
    size_t                           k_execution = 0;
    size_t                           runs        = 0;
    const int64_t *                  correction  = nullptr; // radix_reconstruct_end: stripe rows x cols
    const float *                    out         = nullptr; // merge_end, semantic_complete: first stripe row
    size_t                           out_stride  = 0;
    size_t                           nonzero     = 0;       // merge_end: nonzero correction entries
    const char *                     detail      = nullptr; // run_failed: the first error; paired_prepare_end: the refusal
    const char *                     route       = nullptr; // paired_prepare_end: "paired" | "serial_fallback"
    const potal_paired_report *      paired      = nullptr; // paired_end: the executor's report
    const int32_t *                  residual_c  = nullptr; // paired_end: Residual C (residual_rows lane rows)
    size_t                           residual_ldc  = 0;
    size_t                           residual_rows = 0;
    // submit_accepted: ExSIA's quantization interval and folding commit of the stripe (ns, CPU).
    uint64_t                         quantization_start_ns = 0;
    uint64_t                         quantization_end_ns   = 0;
    uint64_t                         folding_commit_ns     = 0;
    size_t                           producer_slot         = no_stripe; // ExSIA pipeline slot
};

using stripe_observer = std::function<void(const stripe_event &)>;

struct pipeline_config {
    size_t          job_capacity = 2;          // 1 or 2 semantic stripes in flight
    bool            residual     = false;      // compensation ON: ws_packet packets may arrive
    std::string     layer;                     // log layer of the node
    const char *    backend      = "HARDWARE"; // records' owner label: HARDWARE (issuer), EMULATED (host tests)
    // GGML_POTAL_PAIRED_AUTO (options.hpp): each stripe's Main and Residual GEMMs as
    // one paired execution (matmul/paired-auto.hpp) through owner_runners::paired;
    // a stripe whose paired plan is refused for capacity runs the serial route,
    // reported as serial_fallback. Needs residual (compensation ON).
    bool            paired       = false;
    // GGML_POTAL_PAIRED_FENCES=CONSERVATIVE: the paired executor fences before every
    // Indexed LOOP_WS, ACC readback and scale release (potal_paired_request).
    bool            paired_conservative_fences = false;
    // Optional; called on the thread of each step after its record, holding no
    // lock of the run (it may query or abort the run). Order across threads:
    // stripe_event::sequence.
    stripe_observer observer;
};

// One semantically completed stripe, in completion order (poll_completed).
struct stripe_completion {
    size_t   stripe_id           = 0;
    size_t   row_begin           = 0;
    size_t   row_end             = 0;
    size_t   slot                = 0; // pool slot index
    uint64_t slot_generation     = 0; // reuse count of that slot before this stripe
    bool     residual            = false; // a packet was executed and merged
    bool     paired              = false; // Main and Residual ran as one paired execution
    size_t   residual_rows       = 0;     // active (lane, row) pairs of its Residual GEMM
    size_t   k_compact           = 0;
    size_t   k_execution         = 0;
    size_t   corrections_nonzero = 0;
    uint64_t accepted_ns         = 0; // CPU steady clock: READY, Main start, SEMANTIC_DONE
    uint64_t main_begin_ns       = 0;
    uint64_t semantic_done_ns    = 0;
};

// Counters of one run; the caller merges them into the backend stats once,
// after the run (never concurrently with the run's threads).
struct pipeline_counts {
    uint64_t stripes_accepted           = 0;
    uint64_t stripes_semantic_completed = 0;
    uint64_t max_outstanding_semantic   = 0;
    uint64_t producer_capacity_waits    = 0;
    uint64_t main_logical_gemms         = 0;
    uint64_t main_loop_ws               = 0;
    uint64_t main_validated             = 0;
    uint64_t main_fences                = 0;
    uint64_t residual_nonempty_packets  = 0;
    uint64_t residual_logical_gemms     = 0;
    uint64_t residual_loop_ws           = 0;
    uint64_t residual_validated         = 0;
    uint64_t residual_fences            = 0;
    uint64_t residual_compact_k         = 0;
    uint64_t residual_execution_k       = 0;
    uint64_t residual_active_lane_rows  = 0;
    uint64_t radix_reconstructions      = 0;
    uint64_t merges_completed           = 0;
    uint64_t corrections_nonzero        = 0;
    uint64_t scale_writes               = 0;
    uint64_t scale_releases             = 0;
    uint64_t tlb_flushes                = 0;
    uint32_t last_generation            = 0;
    // Paired route (pipeline_config::paired); scale_writes / releases / tlb_flushes above include it.
    uint64_t paired_stripes             = 0; // stripes run as one paired execution
    uint64_t paired_fallback_stripes    = 0; // paired plan refused (capacity): the serial route ran them
    uint64_t paired_main_loop_ws        = 0;
    uint64_t paired_residual_loop_ws    = 0;
    uint64_t paired_index_configs       = 0;
    uint64_t paired_index_writes        = 0;
    uint64_t paired_readbacks           = 0;
    uint64_t paired_readback_tiles      = 0; // MVOUT commands
    uint64_t paired_fences              = 0;
    uint64_t paired_drains              = 0; // fences + released scale windows (each is a drain point)
    double   scale_s                    = 0; // guest seconds inside the GEMMs (owner thread)
    double   issue_s                    = 0;
    double   fence_s                    = 0;
};

struct pipeline_report {
    bool            ok     = false;
    bool            fatal  = false;    // buffers handed to the accelerator must be abandoned
    potal_status    status = POTAL_OK; // first failing potal_* status, POTAL_OK if not one
    const char *    stage  = nullptr;  // first failing step
    std::string     detail;
    bool            run_bound        = false;
    uint64_t        run_id           = 0;
    size_t          expected_stripes = 0;
    pipeline_counts counts;
    std::vector<stripe_completion> completions; // every semantically completed stripe, in order
};

// One run: the semantic slots, the private output and the queue between the
// producer and the consumer. Shared by the producer and the owner thread.
class stripe_run {
  public:
    // args: the node's arguments (init_args) after nothing but allocation: the
    // run reads I/J/K, the stripe geometry and the weight view from it, now.
    // w and plan are borrowed and must outlive the consumer.
    stripe_run(const pipeline_config & cfg, const ggml_gemmini_args_t & args, const hp1_weight & w,
               const potal_tile_plan & plan);
    ~stripe_run();
    stripe_run(const stripe_run &)             = delete;
    stripe_run & operator=(const stripe_run &) = delete;

    // Slots (DMA buffers), private output. false: why says it.
    bool begin(std::string & why);

    // ---- producer -------------------------------------------------------------
    // One folding-committed stripe; producer_args: the arguments ExSIA writes
    // (their A rows of the stripe are copied). Blocks while the pipeline is
    // full. false: the run failed (now or earlier), the stripe was not taken.
    bool submit(const exsia::StripeReadyEvent & e, const ggml_gemmini_args_t & producer_args);
    // Closes the input and waits until every expected stripe completed or the
    // run failed. true: all stripes semantically done.
    bool finish();
    // Fails the run (sticky; nothing new starts) and closes the input.
    void abort(const char * stage, const std::string & detail);

    // ---- consumer: on the accelerator owner thread -----------------------------
    consume_outcome consume(const owner_runners & runners);
    consume_outcome consume(const planned_runner & exec); // an owner without a paired runner

    // Completion records added since the last call (any thread).
    std::vector<stripe_completion> poll_completed();

    // ---- after the consumer returned --------------------------------------------
    pipeline_report      report() const;
    const float *        output() const { return out_.data(); } // M x N, row stride N; complete only when ok
    size_t               expected_stripes() const { return expected_; }
    void                 abandon(); // fatal: leak every DMA buffer of the run
    void                 emit_run(stripe_event_kind kind, const char * detail = nullptr);

  private:
    struct slot;
    struct stripe_result;

    stripe_result execute(slot & s, const stripe_event & main_begin, const owner_runners & runners);
    std::string   check_event(const exsia::StripeReadyEvent & e, size_t rows) const;
    std::string   check_order_locked(const exsia::StripeReadyEvent & e, size_t rows) const;
    void          fail_locked(const char * stage, potal_status status, bool fatal, std::string detail);
    void          fail(const char * stage, potal_status status, bool fatal, std::string detail);
    void          set_state(slot & s, slot_state state);

    // Events: stamped under mutex_ (sequence, t_ns, outstanding), emitted
    // after it is released. Every stamped event must be emitted exactly once,
    // with nothing that blocks or throws in between (the JSONL order waits for it).
    void         stamp_locked(stripe_event & ev);
    stripe_event slot_event_locked(stripe_event_kind kind, const slot & s);
    stripe_event event_of(stripe_event_kind kind, const slot & s);
    stripe_event advance(slot & s, slot_state state, stripe_event_kind kind); // state change + its event
    void         emit(const stripe_event & ev);

    const pipeline_config cfg_;
    const hp1_weight &    w_;
    const potal_tile_plan plan_;
    const size_t          m_, n_, k_;
    const size_t          stripe_rows_;
    size_t                expected_ = 0;
    size_t                lda_ = 0, ldc_ = 0;

    std::vector<float> out_; // private output, M x N

    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<slot>> slots_;
    std::deque<slot *>                 ready_;
    bool                               closing_     = false;
    bool                               failed_      = false;
    bool                               fatal_       = false;
    bool                               run_bound_   = false;
    uint64_t                           run_id_      = 0;
    size_t                             next_stripe_ = 0;
    size_t                             next_row_    = 0;
    size_t                             outstanding_ = 0;
    size_t                             completed_   = 0;
    potal_status                       first_status_ = POTAL_OK;
    const char *                       first_stage_  = nullptr;
    std::string                        first_detail_;
    pipeline_counts                    counts_;
    std::vector<stripe_completion>     completions_; // not yet polled
    uint64_t                           next_event_sequence_ = 1;

    // JSONL records in sequence order (emit): the last sequence written.
    std::mutex              log_mutex_;
    std::condition_variable log_cv_;
    uint64_t                logged_sequence_ = 0;
};

// The pipeline of one MUL_MAT, from quantization to commit:
//   check the configuration -> run.begin -> owner.start(consumer) ->
//   owner.pin_producer -> attach the stripe sink -> quantize_activation(src1,
//   args) (ExSIA submits every committed stripe at once) -> finish (or abort) ->
//   owner.join -> on success copy the private output into args.f_out.
// args: init_args of this node with the frozen Main plan (its
// activation_rows_per_stripe is plan.tile_I * DIM). The output is written
// only when the report is ok; a fatal report means the run abandoned its
// buffers and w must be abandoned by its owner.
pipeline_report run_stripe_pipeline(const ggml_tensor * src1, ggml_gemmini_args_t & args, const hp1_weight & w,
                                    const potal_tile_plan & plan, const pipeline_config & cfg, stripe_owner & owner);

} // namespace ggml::potal
