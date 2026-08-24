#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
    uint64_t gemmini_read_cycles(void) noexcept;
}
#else
    uint64_t gemmini_read_cycles(void);
#endif
