#pragma once

// Accelerator owners for host tests of the stripe pipeline: the production
// consumer (stripe_run::consume) runs on a std::thread of the test instead of
// the hart-0 issuer, and its planned GEMMs run on an emulator session through
// the issuer's own step after its affinity check, run_planned_stripe_gemm()
// (preflight, TLB FLUSH + FENCE, the GEMM; the emulator records every command).
// Nothing here pretends to be a RISC-V hart: no affinity check, backend=EMULATED.

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
        return ggml::potal::run_planned_stripe_gemm(session, d);
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
