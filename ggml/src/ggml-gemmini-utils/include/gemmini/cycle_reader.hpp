// header-only
#pragma once
#include <stdint.h>

namespace ggml::gemmini::cycle
{
#ifdef __riscv
    static inline uint64_t read()
    {
        uint64_t cycles;
        asm volatile("rdcycle %0" : "=r"(cycles));
        return cycles;
    }
#else

    static inline uint64_t read()
    {
        return 0;
    }
#endif
}
