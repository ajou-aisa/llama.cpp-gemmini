#include "../include/gemmini/cycle_reader.hpp"
#include "../include/gemmini/cycle_reader.h"
#if defined(__linux__) && defined(__aarch64__)
#include "cycle_reader_internal.h"

#include <atomic>

static_assert(static_cast<uint8_t>(ggml::gemmini::cycle::NativeCycleReason::counter_regression) ==
              GEMMINI_NATIVE_CYCLE_REASON_COUNTER_REGRESSION);
#endif

extern "C" uint64_t gemmini_read_cycles(void) noexcept
{
    try { return ggml::gemmini::cycle::read(); }
    catch (...) { return 0; }
}

#if defined(__linux__) && defined(__aarch64__)
extern "C" gemmini_native_cycle_sample_internal
        gemmini_read_native_cycle_sample_internal(void) noexcept
{
#if !LOG_CYCLE
    return {};
#else
    using namespace ggml::gemmini::cycle;
    read_count.fetch_add(1, std::memory_order_relaxed);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const NativeCycleSample sample = read_sample();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {sample.value, static_cast<uint8_t>(sample.valid),
            static_cast<uint8_t>(sample.reason),
            GEMMINI_NATIVE_CYCLE_SOURCE_LINUX_PERF_CPU_CYCLES,
            sample.owner_event_token, sample.generation};
#endif
}
#endif
