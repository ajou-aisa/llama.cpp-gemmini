#pragma once

// PoTal hardware execution for the GGML backend. Every PoTal RoCC command of
// the process (TLB FLUSH, SCALE_WRITE/RELEASE, GEMM, FENCE) is issued by one
// issuer thread pinned to the Linux CPU of hart 0: potal.h is not thread-safe
// and custom-3 traps on every other hart.

#include "potal.h"

#include <cstdint>

namespace ggml::potal {

// One DIM16 GEMM with every scale carrier x1:
// C[16][16] = A[16][k] * B[k][16], k = 16 or 32.
struct gemm_job {
    const int8_t * a; // [16][k], row stride k
    const int8_t * b; // [k][16], row stride 16
    uint32_t       k;
    int32_t *      c; // [16][16], row stride 16; written only when the report is ok
};

struct gemm_report {
    bool         ok     = false;
    bool         fatal  = false;    // hardware state unknown: the issuer refuses all later work
    bool         issued = false;    // potal_gemm_ws_issue() == POTAL_OK
    bool         fenced = false;    // potal_fence() == POTAL_OK after the issue
    potal_status status = POTAL_OK; // failing potal_* status, POTAL_OK if the failure was not one
    const char * stage  = nullptr;  // failing step, nullptr when ok
    const char * detail = nullptr;  // issuer failure reason, if any
    uint32_t     scale_slot = 0;
    uint32_t     generation = 0;
    double       stage_s = 0;       // guest CLOCK_MONOTONIC seconds
    double       scale_s = 0;
    double       issue_s = 0;
    double       fence_s = 0;
};

struct issuer_info {
    bool ready = false;
    int  cpu   = -1;
    long tid   = -1;
};

// Runs the job on the process-wide issuer thread (started and brought up on
// the first call) and waits for it. Off RISC-V, validates the request the
// issuer would send and reports POTAL_ERR_NO_HARDWARE; nothing succeeds.
gemm_report run_gemm(const gemm_job & job);

issuer_info get_issuer_info();

} // namespace ggml::potal
