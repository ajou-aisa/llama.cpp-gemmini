/*
 * PoTal raw RoCC issue layer (RISC-V only).
 *
 * Unchecked: these functions send rs1/rs2 exactly as given. Use the checked
 * API in potal.h unless a test needs a deliberately malformed command.
 *
 * Encoding: ".insn r CUSTOM_<POTAL_HW_XCUSTOM>, funct3, funct7, rd, rs1, rs2"
 * with funct3 = {xd, xs1, xs2}. Scale-control commands set xd and return the
 * hardware status; CONFIG/LOOP_WS commands use xd = 0 (no response).
 * Every command carries a compiler "memory" clobber. That orders compiler
 * accesses only; it is neither a CPU memory fence nor an accelerator
 * completion signal (use potal_fence()).
 */
#ifndef POTAL_ROCC_H
#define POTAL_ROCC_H

#include <stdint.h>

#include "potal_hw_params.h"

#if !defined(__riscv)
#error "potal_rocc.h issues RoCC instructions and requires a RISC-V target"
#endif

#if POTAL_HW_XCUSTOM != 3
#error "PoTal hardware decodes custom-3 only"
#endif

/* funct7 values (GemminiISA.scala). */
#define POTAL_FUNCT_CONFIG                    0
#define POTAL_FUNCT_LOOP_WS                   8
#define POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS     9
#define POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB   10
#define POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_DC   11
#define POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_AB 12
#define POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_DC 13
#define POTAL_FUNCT_SCALE_WRITE               23
#define POTAL_FUNCT_SCALE_RELEASE             24
#define POTAL_FUNCT_FENCE                     25

#define POTAL_ROCC_STR_(x) #x
#define POTAL_ROCC_STR(x)  POTAL_ROCC_STR_(x)
#define POTAL_ROCC_OPCODE  "CUSTOM_" POTAL_ROCC_STR(POTAL_HW_XCUSTOM)

/* xd=1, xs1=1, xs2=1: returns rd. */
#define POTAL_ROCC_R_R_R(funct, rd, rs1, rs2)                                   \
    __asm__ volatile(".insn r " POTAL_ROCC_OPCODE ", 0x7, " POTAL_ROCC_STR(funct) \
                     ", %0, %1, %2"                                             \
                     : "=r"(rd)                                                 \
                     : "r"(rs1), "r"(rs2)                                       \
                     : "memory")

/* xd=0, xs1=1, xs2=1: fire-and-forget. */
#define POTAL_ROCC_0_R_R(funct, rs1, rs2)                                       \
    __asm__ volatile(".insn r " POTAL_ROCC_OPCODE ", 0x3, " POTAL_ROCC_STR(funct) \
                     ", x0, %0, %1"                                             \
                     :                                                          \
                     : "r"(rs1), "r"(rs2)                                       \
                     : "memory")

static inline uint64_t potal_rocc_scale_write(uint64_t rs1, uint64_t rs2) {
    uint64_t rd;
    POTAL_ROCC_R_R_R(POTAL_FUNCT_SCALE_WRITE, rd, rs1, rs2);
    return rd;
}

static inline uint64_t potal_rocc_scale_release(uint64_t rs1, uint64_t rs2) {
    uint64_t rd;
    POTAL_ROCC_R_R_R(POTAL_FUNCT_SCALE_RELEASE, rd, rs1, rs2);
    return rd;
}

/* Blocks in hardware until older work drains; returns the status. */
static inline uint64_t potal_rocc_fence(uint64_t rs1, uint64_t rs2) {
    uint64_t rd;
    POTAL_ROCC_R_R_R(POTAL_FUNCT_FENCE, rd, rs1, rs2);
    return rd;
}

static inline void potal_rocc_config(uint64_t rs1, uint64_t rs2) {
    POTAL_ROCC_0_R_R(POTAL_FUNCT_CONFIG, rs1, rs2);
}

static inline void potal_rocc_loop_ws_bounds(uint64_t rs1, uint64_t rs2) {
    POTAL_ROCC_0_R_R(POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS, rs1, rs2);
}

static inline void potal_rocc_loop_ws_addrs_ab(uint64_t rs1, uint64_t rs2) {
    POTAL_ROCC_0_R_R(POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB, rs1, rs2);
}

static inline void potal_rocc_loop_ws_addrs_dc(uint64_t rs1, uint64_t rs2) {
    POTAL_ROCC_0_R_R(POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_DC, rs1, rs2);
}

static inline void potal_rocc_loop_ws_strides_ab(uint64_t rs1, uint64_t rs2) {
    POTAL_ROCC_0_R_R(POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_AB, rs1, rs2);
}

static inline void potal_rocc_loop_ws_strides_dc(uint64_t rs1, uint64_t rs2) {
    POTAL_ROCC_0_R_R(POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_DC, rs1, rs2);
}

static inline void potal_rocc_loop_ws(uint64_t rs1, uint64_t rs2) {
    POTAL_ROCC_0_R_R(POTAL_FUNCT_LOOP_WS, rs1, rs2);
}

/* CPU ordering of normal loads/stores around DMA hand-off; not completion. */
static inline void potal_cpu_fence_rw(void) {
    __asm__ volatile("fence rw, rw" ::: "memory");
}

#endif /* POTAL_ROCC_H */
