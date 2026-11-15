#pragma once
// Compatibility only. The shared stock SDK and generated parameters remain verbatim.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#if !defined(__riscv)
void gemmini_fp32_model_issue(uint64_t, uint64_t, uint32_t);
void gemmini_fp32_model_fence();
// Skip RISC-V inline assembly in xcustom.h; preserve every issued command.
#define ROCC_SOFTWARE_SRC_XCUSTOM_H_
#define ROCC_INSTRUCTION_0_R_R(x, a, b, f) \
    do { gemmini_fp32_model_issue(uint64_t(a), uint64_t(b), f); } while (0);
#define ROCC_INSTRUCTION(x, rd, a, b, f) \
    do { std::abort(); } while (0);
// The stock header's only assembly statement is asm volatile("fence").
// Its expansion inside the SDK must fence the synchronous native command model.
#define asm
#define volatile(...) gemmini_fp32_model_fence()
#endif
#include <gemmini.h>
#if !defined(__riscv)
#undef asm
#undef volatile
#endif
