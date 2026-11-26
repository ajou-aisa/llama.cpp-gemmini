#pragma once
#include "gemmini.h"
#include <memory>

namespace ggml::gemmini::pb {
// Command-level INT8 WS functional model. No hardware timing or cycle claims.
// Each instance owns its operand cache and accumulator; it is used by one issuer.
class FunctionalTransport {
  public:
    struct State;
    FunctionalTransport();
    ~FunctionalTransport();
    FunctionalTransport(const FunctionalTransport &) = delete;
    FunctionalTransport &operator=(const FunctionalTransport &) = delete;
    const gemmini_transport_t &transport() const { return transport_; }
    uint64_t command_count(uint8_t funct) const;
    uint64_t drain_count() const;
  private:
    static gemmini_status_t issue(void *, const gemmini_command_t *, uint64_t *);
    static gemmini_status_t drain(void *);
    std::unique_ptr<State> state_;
    const gemmini_transport_t transport_;
};
} // namespace ggml::gemmini::pb
