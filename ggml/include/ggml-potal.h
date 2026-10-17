#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// PoTal RoCC backend (A8W8 WS Sat32, DIM16, accelerator attached to hart 0).
//
// Executes only GGML_OP_MUL_MAT with src0 (weights, ne = {K, N}) and src1
// (activations, ne = {K, M}) of type GGML_TYPE_I8, 2-D, contiguous, in host
// memory, M = N = 16 and K = 16 or 32. dst is F32 and receives the exact INT32
// dot products: every PoTal scale carrier is x1, no scale is invented.
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
    GGML_POTAL_REJECT_OP_PARAMS, // MUL_MAT tagged beyond op_params[0] (e.g. an attention role)
    GGML_POTAL_REJECT_TYPE,      // src0/src1 not I8, or dst not F32
    GGML_POTAL_REJECT_SHAPE,     // not 2-D M = N = 16 with K = 16 or 32
    GGML_POTAL_REJECT_LAYOUT,    // not contiguous (transposed, permuted, strided)
    GGML_POTAL_REJECT_MEMORY,    // not in host memory, or unallocated at compute time
    GGML_POTAL_REJECT_COUNT,
};

GGML_BACKEND_API const char * ggml_potal_reject_name(enum ggml_potal_reject reason);

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
};

GGML_BACKEND_API bool ggml_backend_potal_get_stats(ggml_backend_t backend, struct ggml_backend_potal_stats * stats);

#ifdef __cplusplus
}
#endif
