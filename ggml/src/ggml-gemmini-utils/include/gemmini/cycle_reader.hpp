// header-only
#pragma once
#include <stdint.h>

#if defined(__APPLE__) && defined(__aarch64__)
#include <mach/mach_time.h>
#elif !defined(__riscv)
#include <chrono>
#endif

namespace ggml::gemmini::cycle
{
#ifdef __riscv
    static inline uint64_t read()
    {
        uint64_t cycles;
        asm volatile("rdcycle %0" : "=r"(cycles));
        return cycles;
    }
#elif defined(__APPLE__) && defined(__aarch64__)
    static inline uint64_t read()
    {
        // macOS arm64 exposes elapsed timer ticks here, not CPU cycles.
        return mach_absolute_time();
    }
#else

    static inline uint64_t read()
    {
        return static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
#endif

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

    static inline uint64_t resolution()
    {
        return 1;
    }
}
