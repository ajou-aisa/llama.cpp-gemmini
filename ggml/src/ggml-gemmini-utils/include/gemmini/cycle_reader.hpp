// header-only
#pragma once
#include <stdint.h>
#include <atomic>

#if defined(__APPLE__) && defined(__aarch64__)
#include <mach/mach_time.h>
#elif !defined(__riscv)
#include <chrono>
#endif

namespace ggml::gemmini::cycle
{
    // This counter is a test seam for proving that disabled instrumentation does
    // not touch a clock. It is thread-local and is incremented only in enabled builds.
    inline thread_local uint64_t read_count = 0;

    static inline uint64_t read()
    {
#if !LOG_CYCLE
        return 0;
#else
        ++read_count;
        std::atomic_signal_fence(std::memory_order_seq_cst);
#ifdef __riscv
        uint64_t value;
        asm volatile("rdcycle %0" : "=r"(value) :: "memory");
#elif defined(__APPLE__) && defined(__aarch64__)
        const uint64_t value = mach_absolute_time();
#else
        const uint64_t value = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
#endif
        std::atomic_signal_fence(std::memory_order_seq_cst);
        return value;
#endif
    }

    static inline void reset_read_count_for_test() { read_count = 0; }
    static inline uint64_t read_count_for_test() { return read_count; }

    static inline const char * clock_mode()
    {
#ifdef __riscv
        return "CYCLE";
#else
        return "TIMER";
#endif
    }

    static inline const char * units()
    {
#ifdef __riscv
        return "cycles";
#else
        return "ticks";
#endif
    }

    static inline uint64_t resolution() { return 1; }
}
