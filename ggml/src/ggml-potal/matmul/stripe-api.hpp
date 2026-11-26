#pragma once

// potal_tiled_matmul_stripe() and its session (potal.h) on the tiled WS path
// of tiling.hpp. The opaque potal_stripe_session is a potal_ws_session: the
// issuer owns exactly one (execution.cpp) and runs the I8 GEMM, the FULL tiled
// jobs and the planned stripe GEMMs on it, so all of them share one scale
// generation sequence and one set of work counters. Tests bind an emulator by
// constructing the session with their own ops.

#include "potal.h"
#include "tiling.hpp"

#include <cstdint>

struct potal_stripe_session {
    ggml::potal::potal_ws_session ws;
};

namespace ggml::potal {

// A descriptor that passed prepare_stripe_gemm(), with its frozen plan.
// Executing it repeats no request check above the per-invocation command
// checks of the leaf (potal_loop_ws_check, the invocation's carriers), so
// neither the descriptor nor the memory it names may change between the
// preflight and the execution.
struct prepared_stripe_gemm {
    potal_stripe_gemm_desc desc = {};
    potal_tile_plan        plan;
};

// The preflight of a planned GEMM: everything potal_tiled_matmul() and the
// invocations below it would check, on the whole request (work_kind, m/n/k,
// strides, scale_block_k, the frozen tiles against the shape and the
// SPAD/ACC/WorkIdsPerLoop/scale-window limits, A/B DIM-byte and C DIM*4-byte
// alignment, operand extents within the 32-bit DRAM offset, C disjoint from A
// and B, every carrier valid). Issues no command and touches no session. On
// failure *stage names the check: request, tiles, operands or carriers.
potal_status prepare_stripe_gemm(const potal_stripe_gemm_desc & desc, prepared_stripe_gemm & out, const char ** stage);

// One planned GEMM as the accelerator owner runs it (the hart-0 issuer after
// its affinity check; an emulator session in host tests):
//   session check -> preflight -> TLB FLUSH + FENCE -> the prepared GEMM
// (SCALE_WRITE -> LOOP_WS -> FENCE -> SCALE_RELEASE per invocation). The GEMM
// may read buffers allocated since the previous one, so every accelerator
// translation is dropped before its first DMA; the previous GEMM ended with a
// FENCE. A refused descriptor issues nothing, not even the TLB FLUSH: stage
// "preflight" (report.stage names the failed check), not fatal, the session
// stays usable. A disabled session: "session_disabled", fatal, nothing issued.
// A failed TLB FLUSH or its FENCE: "tlb_flush", fatal, the session disabled.
struct planned_result {
    bool                     ok          = false;
    bool                     fatal       = false;   // hardware state unknown
    potal_status             status      = POTAL_OK;
    const char *             stage       = nullptr; // failing step, nullptr when ok
    uint64_t                 tlb_flushes = 0;       // TLB FLUSH + FENCE pairs before the GEMM
    potal_stripe_gemm_report report      = {};
};

planned_result run_planned_stripe_gemm(potal_stripe_session & session, const potal_stripe_gemm_desc & desc);

} // namespace ggml::potal
