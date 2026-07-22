// header-only
#pragma once
#include <stdint.h>

#if defined(__APPLE__) && defined(__aarch64__)
#include <mach/mach_time.h>
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
        return 0;
    }
#endif
}
