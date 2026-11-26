/*
 * PoTal raw RoCC issue layer (RISC-V only).
 *
 * Unchecked: these functions send rs1/rs2 exactly as given. Use the checked
 * API in potal.h unless a test needs a deliberately malformed command.
 *
 * Encoding: ".insn r CUSTOM_<POTAL_HW_XCUSTOM>, funct3, funct7, rd, rs1, rs2"
 * with funct3 = {xd, xs1, xs2}. Scale-control commands set xd and return the
 * hardware status; CONFIG/legacy LOOP_WS commands use xd = 0 (no response).
 * Resident-view and extended LOOP_WS return status; COUNTER returns uint64_t.
 * Every command carries a compiler "memory" clobber. That orders compiler
 * accesses only; it is neither a CPU memory fence nor an accelerator
 * completion signal (use potal_fence()).
 */
#ifndef POTAL_ROCC_H
#define POTAL_ROCC_H

#include <stdint.h>

#include "potal.h" /* POTAL_FUNCT_* */
#include "potal_hw_params.h"

#if !defined(__riscv)
#error "potal_rocc.h issues RoCC instructions and requires a RISC-V target"
#endif

#if POTAL_HW_XCUSTOM != 3
#error "PoTal hardware decodes custom-3 only"
#endif

/* funct7 values: POTAL_FUNCT_* in potal.h. */

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

/*
 * FLUSH (Controller.scala): rs1[0] = 0 flush+retry, 1 flush+skip; xd = 0.
 * Travels in order with GEMM commands and fires the accelerator TLB sfence
 * when it reaches the unrolled command queue; no response.
 */
static inline void potal_rocc_flush(uint64_t rs1, uint64_t rs2) {
    POTAL_ROCC_0_R_R(POTAL_FUNCT_FLUSH, rs1, rs2);
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

/* Opt-in resident run (rs1 bit3), unlike legacy LOOP_WS, returns status. */
static inline uint64_t potal_rocc_resident_loop_ws(uint64_t rs1, uint64_t rs2) {
    uint64_t rd;
    POTAL_ROCC_R_R_R(POTAL_FUNCT_LOOP_WS, rd, rs1, rs2);
    return rd;
}

static inline uint64_t potal_rocc_resident_view(uint64_t rs1, uint64_t rs2) {
    uint64_t rd;
    POTAL_ROCC_R_R_R(POTAL_FUNCT_RESIDENT_VIEW, rd, rs1, rs2);
    return rd;
}

static inline uint64_t potal_rocc_ldb_only(uint64_t rs1, uint64_t rs2) {
    uint64_t rd;
    POTAL_ROCC_R_R_R(POTAL_FUNCT_LDB_ONLY, rd, rs1, rs2);
    return rd;
}

static inline uint64_t potal_rocc_stream(uint64_t rs1, uint64_t rs2) {
    uint64_t rd;
    POTAL_ROCC_R_R_R(POTAL_FUNCT_STREAM, rd, rs1, rs2);
    return rd;
}

/* Counter responses are arbitrary uint64_t values, NOT status codes. */
static inline uint64_t potal_rocc_counter(uint64_t rs1, uint64_t rs2) {
    uint64_t rd;
    POTAL_ROCC_R_R_R(POTAL_FUNCT_COUNTER, rd, rs1, rs2);
    return rd;
}

/* INDEX_CONFIG / INDEX_WRITE (PoTalScaleCommandIngress): xd, return the status. */
static inline uint64_t potal_rocc_index_config(uint64_t rs1, uint64_t rs2) {
    uint64_t rd;
    POTAL_ROCC_R_R_R(POTAL_FUNCT_INDEX_CONFIG, rd, rs1, rs2);
    return rd;
}

static inline uint64_t potal_rocc_index_write(uint64_t rs1, uint64_t rs2) {
    uint64_t rd;
    POTAL_ROCC_R_R_R(POTAL_FUNCT_INDEX_WRITE, rd, rs1, rs2);
    return rd;
}

/* MVOUT (STORE_CMD): rs1 DRAM address, rs2 {rows[63:48], cols[47:32], local addr[31:0]}; no response. */
static inline void potal_rocc_mvout(uint64_t rs1, uint64_t rs2) {
    POTAL_ROCC_0_R_R(POTAL_FUNCT_MVOUT, rs1, rs2);
}

/* LOAD_CMD: full-width INT32 when the destination is ACC and CONFIG_LD shrink=0. */
static inline void potal_rocc_mvin(uint64_t rs1, uint64_t rs2) {
    POTAL_ROCC_0_R_R(POTAL_FUNCT_MVIN, rs1, rs2);
}

/* CPU memory ordering around DMA hand-off. Rocket also stalls this instruction
 * on RoCC busy. Never substitute it for a checked PoTal completion response. */
static inline void potal_cpu_fence_rw(void) {
    __asm__ volatile("fence rw, rw" ::: "memory");
}

#endif /* POTAL_ROCC_H */
