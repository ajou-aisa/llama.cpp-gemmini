#pragma once

// PoTal hardware execution for the GGML backend. Every PoTal RoCC command of
// the process (TLB FLUSH, SCALE_WRITE/RELEASE, GEMM, FENCE) is issued by one
// issuer thread pinned to the Linux CPU of hart 0: potal.h is not thread-safe
// and custom-3 traps on every other hart. The issuer runs one request at a
// time: a synchronous job (run_gemm, run_tiled) or, for a stripe pipeline, a
// consumer that keeps the issuer until its run ends (issuer_owner).

#include "potal.h"
#include "stripe-api.hpp"
#include "tiling.hpp"

#include <gemmini/trace-context.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

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

// One general GEMM through potal_tiled_matmul_auto() on DMA-ready operands:
// C[m][n] = PoTal(A[m][k], B[k][n], carriers[k/block_k][n]) (see dense.hpp).
// The buffers must stay untouched until the report returns; after a fatal
// report they must never be freed (the hardware may still access them).
struct tiled_job {
    size_t           m, n, k;
    const int8_t *   a;
    size_t           lda;
    const int8_t *   b;
    size_t           ldb;
    const uint32_t * carriers;
    size_t           cstride;
    uint32_t         block_k;
    int32_t *        c;
    size_t           ldc;
    // Caller's trace context (layer/node/request), bound on the issuer thread
    // while the job runs so its logs keep the operation's identity.
    gemmini_trace_context trace = gemmini_trace_capture();
};

struct tiled_report {
    bool            ok     = false;
    bool            fatal  = false;
    potal_status    status = POTAL_OK;
    const char *    stage  = nullptr;
    const char *    detail = nullptr; // issuer failure reason, if any
    potal_tile_plan plan;
    potal_ws_report work;             // this job only: counts and guest seconds
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

// Same issuer; off RISC-V the tiled path stops at its first hardware command
// with POTAL_ERR_NO_HARDWARE.
tiled_report run_tiled(const tiled_job & job);

// The job on a caller-owned session, on the calling thread: what the issuer
// runs after its affinity check and TLB flush. Tests bind an emulator session.
tiled_report run_tiled_in_session(const tiled_job & job, potal_ws_session & session);

issuer_info get_issuer_info();

// ---- stripe pipeline consumer (matmul/stripe.hpp) ----------------------------

// One planned logical GEMM on the accelerator owner thread, inside the owner's
// safety policy (issuer: hart-0 affinity check, then run_planned_stripe_gemm:
// preflight, TLB FLUSH + FENCE, the GEMM; see stripe-api.hpp).
using planned_runner = std::function<planned_result(const potal_stripe_gemm_desc &)>;

// How a consumer ended. fatal: the accelerator state is unknown; the issuer
// disables itself and the run's buffers must never be freed.
struct consume_outcome {
    bool         fatal  = false;
    potal_status status = POTAL_OK;
    const char * stage  = nullptr;
};

using stripe_consumer = std::function<consume_outcome(const planned_runner &)>;

// The thread that owns the accelerator while one stripe run is consumed: the
// issuer in the backend, a test thread with an emulator session in host tests.
class stripe_owner {
  public:
    virtual ~stripe_owner() = default;
    // Starts `consume` on the owner thread and returns without waiting for it.
    // false: the owner is unavailable and nothing ran; why/status say why.
    virtual bool start(stripe_consumer consume, std::string & why, potal_status & status) = 0;
    // Waits until the consumer started above returned; no-op without one.
    virtual void join() = 0;
    // After start(): place the calling (producer) thread beside the owner, on
    // one CPU (one_cpu) or on every allowed CPU but the owner's;
    // unpin_producer() restores it. The default places nothing.
    virtual bool pin_producer(bool one_cpu, std::string & why);
    virtual void unpin_producer();
};

// Narrows the calling thread's affinity mask, saved first: to one CPU of it
// other than exclude_cpu (the lowest one; one_cpu), or to the whole mask
// without exclude_cpu. restore() or destruction resets the caller; no other
// thread and no process-wide setting is touched. Threads the caller creates
// while narrowed inherit the narrowed mask (an OpenMP team started inside the
// window keeps it), so one_cpu is for a producer that runs alone. Refuses when
// no CPU would remain. Linux only; elsewhere pin() fails.
class producer_affinity {
  public:
    producer_affinity();
    ~producer_affinity();
    producer_affinity(const producer_affinity &)             = delete;
    producer_affinity & operator=(const producer_affinity &) = delete;

    bool pin(int exclude_cpu, bool one_cpu, std::string & why);
    bool restore(std::string & why); // true if nothing was pinned
    int  cpu() const;                // the one pinned CPU (one_cpu), else -1

  private:
    struct saved;
    std::unique_ptr<saved> saved_;
};

// The process-wide hart-0 issuer as the owner of one stripe run. start()
// waits for the issuer's bring-up (affinity, TLB flush) under the caller's
// unchanged affinity, then hands it the consumer and returns; the consumer
// runs planned GEMMs on the issuer's session (the I8 and FULL session) and
// the issuer keeps a strong reference to it until it returns. From start()
// until join() every other PoTal request of the process waits. pin_producer()
// moves the caller to a CPU other than the issuer's (producer_affinity).
// Off RISC-V start() fails with POTAL_ERR_NO_HARDWARE.
class issuer_owner final : public stripe_owner {
  public:
    issuer_owner();
    ~issuer_owner() override; // unpin_producer(), then join()

    bool start(stripe_consumer consume, std::string & why, potal_status & status) override;
    void join() override;
    bool pin_producer(bool one_cpu, std::string & why) override;
    void unpin_producer() override;

  private:
    struct state;
    std::unique_ptr<state> st_;
};

} // namespace ggml::potal
