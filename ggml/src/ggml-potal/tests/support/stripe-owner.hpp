#pragma once

// Accelerator owners for host tests of the stripe pipeline: the production
// consumer (stripe_run::consume) runs on a std::thread of the test instead of
// the hart-0 issuer, and its planned GEMMs run on an emulator session through
// the production potal_tiled_matmul_stripe(). Nothing here pretends to be a
// RISC-V hart: no affinity, no TLB (tlb_flushes stays 0, backend=EMULATED).

#include "matmul/execution.hpp"
#include "matmul/stripe-api.hpp"
#include "support/potal-emulator.hpp"

#include <string>
#include <thread>
#include <utility>

namespace potal_test {

// The emulator behind a potal_stripe_session; not copyable (the ops point at it).
struct emu_session {
    emulator                     emu;
    ggml::potal::potal_ws_ops    ops     = emu.ops();
    potal_stripe_session         session = { { &ops, 0, {} } };

    emu_session() = default;
    emu_session(const emu_session &) = delete;
    emu_session & operator=(const emu_session &) = delete;

    ggml::potal::planned_result run(const potal_stripe_gemm_desc & d) {
        ggml::potal::planned_result r;
        r.status = potal_tiled_matmul_stripe(&session, &d, &r.report);
        r.stage  = r.report.stage;
        r.fatal  = r.report.fatal != 0;
        r.ok     = r.status == POTAL_OK;
        return r;
    }
};

class thread_owner final : public ggml::potal::stripe_owner {
  public:
    explicit thread_owner(ggml::potal::planned_runner exec) : exec_(std::move(exec)) {}
    ~thread_owner() override { join(); }

    bool start(ggml::potal::stripe_consumer consume, std::string & why, potal_status & status) override {
        (void) why;
        (void) status;
        thread_ = std::thread([this, consume = std::move(consume)] { outcome = consume(exec_); });
        started = true;
        return true;
    }

    void join() override {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    bool                         started = false;
    ggml::potal::consume_outcome outcome;

  private:
    ggml::potal::planned_runner exec_;
    std::thread                 thread_;
};

// An owner that cannot start (an unavailable issuer).
class refusing_owner final : public ggml::potal::stripe_owner {
  public:
    bool start(ggml::potal::stripe_consumer, std::string & why, potal_status & status) override {
        why    = "test: owner unavailable";
        status = POTAL_ERR_NO_HARDWARE;
        return false;
    }
    void join() override {}
};

} // namespace potal_test
