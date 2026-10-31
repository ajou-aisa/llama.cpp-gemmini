#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// PoTal RoCC backend (A8W8 WS Sat32, DIM16, accelerator attached to hart 0).
//
// Executes GGML_OP_MUL_MAT with src0 (weights, ne = {K, N}) and src1
// (activations, ne = {K, M}), 2-D and in host memory, in two forms:
//   - src0 and src1 GGML_TYPE_I8, contiguous, M = N = 16, K = 16 or 32: one
//     validated GEMM, every scale carrier x1; dst gets the exact INT32 sums.
//   - src0 GGML_TYPE_Q8_HP1 and src1 GGML_TYPE_F32, both contiguous (no
//     row gaps), any M, N and K % 32 == 0: INT8 activation quantization by the
//     configured Gemmini algorithm (GGML_GEMMINI_ACTIVATION_QUANT = EXSIA,
//     TENSOR, TOKEN, BLOCK or STRIPE), HP1 exponents as SCU carriers,
//     Gemmini-compatible tiling (potal_tiled_matmul_auto) on the route of the
//     activation metadata ("<activation>+block"), F32 reconstruction and, with
//     GGML_GEMMINI_ENABLE_RMD, the cpu_direct residual correction (FULL mode).
//     STRIPE_PIPELINE mode (EXSIA only): each ExSIA stripe goes to the
//     hardware as soon as its folding commits; with GGML_GEMMINI_ENABLE_RMD
//     its residual runs as a PoTal Residual GEMM (rmd_backend WS), then the
//     CPU reconstructs and merges it.
// Metadata-only view ops are accepted; every other op is rejected by
// supports_op, so ggml_backend_sched places it on another backend.
// graph_compute never computes on the CPU: a request it cannot run in hardware
// returns GGML_STATUS_FAILED and its dst is set to NaN.
//
// Static registration only (GGML_BACKEND_DL=OFF).

GGML_BACKEND_API ggml_backend_t ggml_backend_potal_init(void);

GGML_BACKEND_API bool ggml_backend_is_potal(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_potal_reg(void);

// Why the POTAL device rejects an op (supports_op == false).
enum ggml_potal_reject {
    GGML_POTAL_REJECT_NONE = 0,  // supported
    GGML_POTAL_REJECT_OP,        // neither MUL_MAT nor a metadata-only view op
    GGML_POTAL_REJECT_OP_PARAMS, // unrecognized MUL_MAT precision/role parameters
    GGML_POTAL_REJECT_TYPE,      // unsupported linear/attention operand types, or dst not F32
    GGML_POTAL_REJECT_SHAPE,     // not 2-D; I8: not M = N = 16 with K = 16 or 32
    GGML_POTAL_REJECT_LAYOUT,    // not contiguous (transposed, permuted, strided)
    GGML_POTAL_REJECT_MEMORY,    // not in host memory, or unallocated at compute time
    GGML_POTAL_REJECT_COUNT,
};

GGML_BACKEND_API const char * ggml_potal_reject_name(enum ggml_potal_reject reason);

// Route of an F32 x Q8_HP1 MUL_MAT: activation metadata x HP1 block-scaled weight.
enum ggml_potal_route {
    GGML_POTAL_ROUTE_TENSOR_BLOCK = 0, // one activation scale
    GGML_POTAL_ROUTE_TOKEN_BLOCK,      // one scale per row
    GGML_POTAL_ROUTE_STRIPE_BLOCK,     // one scale per row stripe
    GGML_POTAL_ROUTE_EXSIA_BLOCK,      // one power-of-two scale per row stripe
    GGML_POTAL_ROUTE_BLOCK_BLOCK,      // one scale per (row, 32-wide K group)
    GGML_POTAL_ROUTE_COUNT,
};

// Counters of one POTAL backend instance, plus device- and process-wide state.
// Times are cumulative guest CLOCK_MONOTONIC seconds, not FPGA cycles.
struct ggml_backend_potal_stats {
    uint64_t mul_mat_nodes;   // MUL_MAT nodes accepted by graph_compute
    uint64_t rejected_nodes;  // nodes graph_compute refused before any hardware command
    uint64_t hw_requests;     // GEMMs handed to the issuer thread
    uint64_t gemm_issued;     // potal_gemm_ws_issue() == POTAL_OK
    uint64_t gemm_fenced;     // potal_fence() == POTAL_OK after an issue
    uint64_t gemm_completed;  // scales released, C written, dst converted to F32
    uint64_t gemm_failed;     // hardware requests that failed (dst set to NaN)
    int      last_status;     // potal_status of the last hardware request
    uint32_t last_scale_slot;
    uint32_t last_generation;
    double   pack_s;          // GGML -> PoTal A/B (graph_compute thread)
    double   unpack_s;        // PoTal C -> GGML F32 (graph_compute thread)
    double   stage_s;         // DMA staging buffer copies (issuer thread)
    double   scale_s;         // SCALE_WRITE + SCALE_RELEASE (issuer thread)
    double   issue_s;         // potal_gemm_ws_issue() (issuer thread)
    double   fence_s;         // potal_fence() (issuer thread)

    // Device-wide supports_op answers; the scheduler asks repeatedly per node.
    uint64_t supports_op_accepted;
    uint64_t supports_op_rejected[GGML_POTAL_REJECT_COUNT];

    // Process-wide issuer thread, started by the first hardware request.
    bool     issuer_ready;    // pinned to the hart-0 CPU, TLB flushed and fenced
    int      issuer_cpu;      // Linux CPU whose /proc/cpuinfo hart is 0, -1 if not ready
    long     issuer_tid;      // -1 if not ready

    // Q8_HP1 x F32 path (hw_requests, scale_s, issue_s, fence_s and
    // last_status above include it).
    uint64_t hp1_nodes;          // HP1 MUL_MAT nodes accepted by graph_compute
    uint64_t hp1_completed;      // dst reconstructed from fenced hardware C (and compensated, if ON)
    uint64_t hp1_failed;         // dst set to NaN
    uint64_t prefill_shaped_nodes; // HP1 nodes with M > 1
    uint64_t decode_shaped_nodes;  // HP1 nodes with M == 1
    uint64_t route_completed[GGML_POTAL_ROUTE_COUNT]; // completed HP1 nodes per route
    uint64_t tiled_jobs;         // potal_tiled_matmul_auto calls (Block+block: one per K group)
    uint64_t tiled_invocations;  // LOOP_WS invocations issued
    uint64_t tiled_validated;    // of which through the validated potal_gemm_ws_issue()
    uint64_t tiled_fences;       // POTAL_FENCE == POTAL_OK, one per invocation
    uint64_t scale_writes;       // SCALE_WRITE == POTAL_OK
    uint64_t scale_releases;     // SCALE_RELEASE == POTAL_OK
    uint64_t tlb_flushes;        // TLB FLUSH + FENCE before each tiled job's first DMA
    uint64_t zero_carriers;      // HP1 blocks with m = INT16_MIN
    uint64_t shift_carriers;     // HP1 blocks with m > 0 (the SCU shifts, not x1)
    uint64_t residual_payloads;      // residual payloads the quantizers captured (0 with compensation OFF)
    uint64_t compensation_executed;  // residual corrections computed and merged into dst
    uint64_t residual_events;        // residual entries those corrections consumed
    uint64_t compensation_failed;    // nodes failed in the compensation step
    double   act_quant_s;        // activation quantization (configured algorithm)
    double   weight_pack_s;      // HP1 validation and B/carrier/scale unpacking, only for nodes
                                 // that prepared their weight (cache miss/bypass/uncached);
                                 // a cache hit adds nothing (see weight_cache_* below)
    double   route_wall_s;       // caller wall time of the route: submission, issuer wait,
                                 // reconstruction; not SCU or array time
    double   compensation_s;     // residual correction and merge

    // Device-wide MUL_MAT supports_op answers that were false. This counts
    // answers, not nodes or CPU executions: the scheduler asks repeatedly per
    // node, and where a rejected node runs is the scheduler's decision.
    uint64_t supports_mul_mat_rejected;

    // Prepared Q8_HP1 weights of this backend (matmul/weight_cache.hpp). A
    // weight in a GGML_BACKEND_BUFFER_USAGE_WEIGHTS buffer is validated and
    // unpacked on its first MUL_MAT and reused afterwards; any other src0 is
    // prepared per call. Budget: GGML_POTAL_WEIGHT_CACHE_MB (default 1024, 0 = off).
    uint64_t weight_cache_hits;      // nodes that reused a prepared weight (no validation, no unpacking)
    uint64_t weight_cache_misses;    // nodes that prepared a weight and cached it
    uint64_t weight_cache_bypasses;  // nodes that prepared a cacheable weight over the budget (not kept)
    uint64_t weight_cache_uncached;  // nodes whose src0 is not in a weights buffer (prepared, not kept)
    uint64_t weight_cache_entries;   // cached weights now
    uint64_t weight_cache_bytes;     // their B + carrier + channel_scale buffers
    uint64_t weight_cache_budget;    // bytes
    uint64_t weight_validations;     // full HP1 validations run (= weight preparations started)
    uint64_t weight_packs;           // complete B/carrier/scale unpackings
    uint64_t weight_pack_bytes;      // B + carrier + channel_scale bytes those unpackings wrote
    double   weight_alloc_s;         // zeroed weight buffer allocation of those preparations
    double   weight_lookup_s;        // cache hits: lookup only

    // STRIPE_PIPELINE nodes (GGML_GEMMINI_DEFAULT_MATMUL_MODE / GEMMINI_MATMUL_MODE):
    // ExSIA stripes handed to the issuer as soon as their folding commits, Main
    // and Residual GEMMs on PoTal, reconstruction and merge on the CPU. Their
    // GEMMs are also in hw_requests, tiled_*, scale_*, tlb_flushes above;
    // hp1_completed still counts nodes. Merged per node after its run ended.
    uint64_t stripe_pipeline_nodes;       // HP1 nodes run as a stripe pipeline
    uint64_t stripe_pipeline_failed;      // of which failed (dst set to NaN)
    uint64_t stripes_accepted;            // stripes submitted and accepted
    uint64_t stripes_semantic_completed;  // Main + Residual + reconstruction/merge done
    uint64_t max_outstanding_semantic;    // most stripes accepted and not yet semantically done
    uint64_t producer_capacity_waits;     // submits that found the pipeline full and waited
    uint64_t main_logical_gemms;          // Main stripe GEMMs completed
    uint64_t main_loop_ws;                // their LOOP_WS invocations
    uint64_t main_fences;                 // their POTAL_FENCE == POTAL_OK
    uint64_t residual_nonempty_packets;   // stripes whose residual was merged
    uint64_t residual_logical_gemms;      // Residual GEMMs completed
    uint64_t residual_loop_ws;
    uint64_t residual_fences;
    uint64_t residual_compact_k;          // sum of the residual packets' compact K (serial and paired stripes)
    uint64_t residual_execution_k;        // sum of the serial Residual GEMMs' K32-padded execution K
    uint64_t residual_active_lane_rows;   // sum of their rows (active (lane, row) pairs; serial and paired stripes)
    uint64_t radix_reconstructions;
    uint64_t merges_completed;
    uint64_t corrections_nonzero;         // nonzero INT64 correction entries merged
    uint64_t last_stripe_count;           // stripes of the last pipelined node
    double   pipeline_wall_s;             // caller wall time of pipelined nodes: quantization,
                                          // the overlapped issuer work and the drain

    // Paired stripes (GGML_POTAL_PAIRED_AUTO, STRIPE_PIPELINE with compensation ON):
    // Main and Residual GEMMs of a stripe as one paired execution on the H2
    // indexed-weight hardware (potal_paired_matmul_auto). Their scale commands, TLB
    // flushes and scale_s / issue_s / fence_s are also in the fields above; the
    // planned-GEMM fields (hw_requests, tiled_*, main_* / residual_* GEMMs, loop_ws
    // and fences) count the serial route only.
    uint64_t paired_stripes;              // stripes run as one paired execution
    uint64_t paired_fallback_stripes;     // paired plan refused (capacity): run by the serial route
    uint64_t paired_main_loop_ws;         // Main paired LOOP_WS submissions
    uint64_t paired_residual_loop_ws;     // Indexed Residual LOOP_WS submissions
    uint64_t paired_index_configs;        // INDEX_CONFIG commands
    uint64_t paired_index_writes;         // INDEX_WRITE commands
    uint64_t paired_readbacks;            // ACC readbacks (CONFIG_ST + MVOUTs of one ACC epoch)
    uint64_t paired_readback_tiles;       // MVOUT commands
    uint64_t paired_fences;               // POTAL_FENCE of paired sessions
    uint64_t paired_drains;               // drain points: fences + released scale windows
    uint64_t paired_prefetch_loads;        // fully submitted standalone B panels
    uint64_t paired_prefetch_mvins;        // checked raw MVIN submissions
    uint64_t paired_prefetch_ldb_only;     // checked LdB-only admissions, not internal LOAD2 count
    uint64_t paired_prefetch_bytes_declared; // command payload extent, not measured bus traffic
    uint64_t paired_resident_main_loop_ws; // Main submissions whose B DMA was suppressed
    double paired_residual_prefetch_issue_s; // Combined CPU API interval, not hardware DMA time
    double paired_main_prefetch_issue_s;   // Main + early admission + extra Fence CPU interval
    uint64_t paired_stream_loops;
    uint64_t paired_stream_pair_waits, paired_stream_local_releases;
    uint64_t paired_stream_batches, paired_stream_capacity_boundaries;
    uint64_t paired_residual_staging_bytes, paired_residual_pitch_padding_bytes;
    uint64_t paired_residual_valid_bytes, paired_residual_load_bytes_declared;

    // Tagged QK/PV attention, separate from Q8_HP1 weighted MatMul and CPU ops.
    // All MatMul partials use PoTal WS; Host preparation/reconstruction is not
    // counted as a hardware completion. Softmax and KV-cache ops remain CPU.
    uint64_t attention_qk_nodes;
    uint64_t attention_pv_nodes;
    uint64_t attention_completed;
    uint64_t attention_failed;
    uint64_t attention_limb_jobs;         // logical jobs that issued at least one LOOP_WS
    uint64_t attention_residual_jobs;     // separately reconstructed upper stream limbs
    uint64_t attention_loop_ws;
    uint64_t attention_qk_limb_jobs;       // completed independent K-low/high jobs for every Q limb
    uint64_t attention_pv_corrections;     // low-limb jobs with one final ACC bias phase
    uint64_t attention_bias_mvins;        // one full-width INT32 bias MVIN per output tile
    double   attention_wall_s;           // preparation, issuer, reconstruction; not FPGA-only time
};

GGML_BACKEND_API bool ggml_backend_potal_get_stats(ggml_backend_t backend, struct ggml_backend_potal_stats * stats);

// Frees every prepared weight of this backend. Required before a weight buffer
// is mutated or freed (and another one possibly allocated at its address)
// while the backend lives; must not overlap ggml_backend_graph_compute on it.
GGML_BACKEND_API void ggml_backend_potal_clear_weight_cache(ggml_backend_t backend);

#ifdef __cplusplus
}
#endif
