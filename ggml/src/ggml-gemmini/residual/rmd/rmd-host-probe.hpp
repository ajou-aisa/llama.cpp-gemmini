#pragma once

// Diagnostic split of one residual completion's host time that no canonical PoTal host stage
// measures. Raw timestamps on the timeline clock (the clock of host_start_ns/host_end_ns) are taken
// around three kinds of code inside the completion and summed per thread:
//   instrumentation: cycle-sim and cycle-log calls (host-stage declaration and completion, call
//                    events, health checks, trace stages);
//   dispatch_host:   host preparation and result hand-off of a compact residual NPU dispatch (run
//                    view, staged output buffer, descriptor, provider statistics), which sit in the
//                    NPU dispatch envelope and are excluded from canonical host cost on every route;
//   emulation:       the functional NPU emulation call itself.
// The totals are written once per completion as a non-additive telemetry record after the
// completion's outer interval has ended. The probe declares no host stage and writes no per-call
// record, so it cannot nest a canonical interval or measure its own record writes.

#include <cstddef>
#include <cstdint>

#if CYCLE_SIM && LOG_CYCLE
#    include <gemmini/host-timing.hpp>
#endif

namespace ggml::gemmini::rmd::probe {

enum class Part : uint8_t {
    instrumentation,
    dispatch_host,
    emulation,
};
inline constexpr size_t kParts = 3;

struct Totals {
    uint64_t ns[kParts]    = {};
    uint64_t calls[kParts] = {};
    // Window of the last emulation call: canonical stages emitted inside it (frontend output
    // reconstruction) are separated from emulation time by the analysis, not by subtraction here.
    uint64_t emulation_begin = 0;
    uint64_t emulation_end   = 0;
};

#if CYCLE_SIM && LOG_CYCLE
inline thread_local Totals totals;

inline uint64_t now() noexcept {
    return cycle::timeline_now_ns();
}

inline void add(Part part, uint64_t start) noexcept {
    const uint64_t end = now();
    totals.ns[static_cast<size_t>(part)] += end > start ? end - start : 0;
    ++totals.calls[static_cast<size_t>(part)];
    if (part == Part::emulation) {
        totals.emulation_begin = start;
        totals.emulation_end   = end;
    }
}

// Returns the totals since the last take and starts a new completion.
inline Totals take() noexcept {
    const Totals result = totals;
    totals              = {};
    return result;
}

class Timer {
  public:
    explicit Timer(Part part) noexcept : part_(part), start_(now()) {}
    ~Timer() { add(part_, start_); }
    Timer(const Timer &)             = delete;
    Timer & operator=(const Timer &) = delete;

  private:
    Part     part_;
    uint64_t start_;
};
#else
inline uint64_t now() noexcept {
    return 0;
}
inline void add(Part, uint64_t) noexcept {}
inline Totals take() noexcept {
    return {};
}
class Timer {
  public:
    explicit Timer(Part) noexcept {}
};
#endif

} // namespace ggml::gemmini::rmd::probe
