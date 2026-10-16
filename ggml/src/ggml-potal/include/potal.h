/*
 * PoTal minimal RoCC API.
 *
 * Standalone C11 interface to the PoTal SCU hardware (Gemmini-derived,
 * Sat32 accumulator). It does not depend on gemmini.h, xcustom.h or the
 * generated gemmini_params.h names; the hardware geometry comes from the
 * build-validated POTAL_HW_PARAMS header (see cmake/potal-hw-params.cmake).
 *
 * Contract sources (generators/potal):
 *   PoTalScaleCommandIngress.scala  SCALE_WRITE/RELEASE/FENCE fields, status
 *   PoTalSCU.scala                  carrier encoding
 *   PoTalCommandMetadata.scala      ScaleEntries, WorkIdsPerLoop
 *   LoopMatmul.scala                LOOP_WS bounds/addr/stride/PoTal fields
 *   GemminiISA.scala                funct7 values, CONFIG layouts
 *
 * Synchronization: issue functions return when the command was handed to the
 * accelerator, never when the work is done. Only potal_fence() returning
 * POTAL_OK means previously issued GEMMs finished and their C stores drained.
 *
 * Not thread-safe: one thread per process may use this API.
 */
#ifndef POTAL_H
#define POTAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define POTAL_SCALE_ENTRIES      256u        /* PoTalProtocol.ScaleEntries */
#define POTAL_GENERATION_MAX     255u        /* 8-bit generation */
#define POTAL_WORK_IDS_PER_LOOP  64u         /* PoTalProtocol.WorkIdsPerLoop */
#define POTAL_CARRIER_MAX_SHIFT  32767u      /* shift carriers 0..32767 (0 = x1) */
#define POTAL_CARRIER_ZERO       0x80000000u /* x0; distinct from shift 0 */

typedef enum potal_status {
    /* Hardware response codes, identical to PoTalScaleCommandStatus. */
    POTAL_OK                  = 0,
    POTAL_HW_INVALID_COLUMN   = 1,
    POTAL_HW_INVALID_SCALE    = 2,
    POTAL_HW_SCALE_CONFLICT   = 3,
    POTAL_HW_RELEASE_MISS     = 4,
    POTAL_HW_INVALID_ADDRESS  = 5,
    POTAL_HW_INVALID_ENCODING = 6,
    POTAL_HW_BAD_COMMAND      = 7,

    /* Software rejections: nothing was issued to the accelerator. */
    POTAL_ERR_INVALID_ARGUMENT = -1, /* out-of-range field, bad pointer/stride */
    POTAL_ERR_UNSUPPORTED      = -2, /* valid for RTL, not validated for this API */
    POTAL_ERR_CAPACITY         = -3, /* scale/work-id/SPAD/ACC capacity exceeded */
    POTAL_ERR_NO_HARDWARE      = -4, /* built for a non-RISC-V host */
    POTAL_ERR_BUSY             = -5, /* GEMM issued and not yet fenced */

    /* The accelerator answered with a value outside 0..7. */
    POTAL_ERR_BAD_RESPONSE     = -6,
} potal_status;

const char *potal_status_name(potal_status status);

/* Hardware identity taken from the validated parameter header. */
uint32_t    potal_hw_dim(void);
const char *potal_hw_params_sha256(void);

/* ---- Pure encoders (no hardware access; host-testable) ------------------ */

/* 1 for shift carriers 0..32767 and POTAL_CARRIER_ZERO, else 0. */
int potal_carrier_is_valid(uint32_t carrier);

/*
 * Scale control rs1: [7:0] slot, [15:8] generation, [21:16] column,
 * [63:22] zero. Rejects slot >= 256, generation > 255, column >= DIM.
 */
potal_status potal_scale_key_encode(uint32_t slot, uint32_t generation,
                                    uint32_t column, uint64_t *rs1);

/* Flags requesting LOOP_WS features; all are currently unsupported. */
#define POTAL_GEMM_A_TRANSPOSE 0x1u
#define POTAL_GEMM_B_TRANSPOSE 0x2u
#define POTAL_GEMM_BIAS        0x4u /* D operand */
#define POTAL_GEMM_RESADD      0x8u
#define POTAL_GEMM_ACCUMULATE  0x10u /* add into existing ACC contents */

/*
 * One weight-stationary PoTal GEMM: C[m][n] = sum_k A[m][k] * B[k][n], where
 * each DIM-wide K pass p is scaled per output column by the carrier in slot
 * potal_gemm_ws_scale_slot(p / passes_per_block, n_tile) and accumulated with
 * Sat32 semantics. All matrices are row-major; strides are in elements.
 *
 *   A: int8,  m rows, row stride lda >= k
 *   B: int8,  k rows, row stride ldb >= n
 *   C: int32, m rows, row stride ldc >= n (overwritten, first contribution)
 *
 * Base pointers must be POTAL_HW_MAX_BYTES aligned (the validated DMA layout).
 * Buffers must stay valid, mapped and untouched by the CPU from
 * potal_gemm_ws_issue() until potal_fence() returns POTAL_OK.
 */
typedef struct potal_gemm_ws {
    const int8_t *a;
    const int8_t *b;
    int32_t      *c;
    size_t        lda;
    size_t        ldb;
    size_t        ldc;
    uint32_t      m;
    uint32_t      n;
    uint32_t      k;
    uint32_t      scale_base;       /* first scale slot of this GEMM */
    uint32_t      scale_generation; /* 0..255, must match SCALE_WRITE */
    uint32_t      passes_per_block; /* DIM-wide K passes sharing one scale */
    uint32_t      flags;            /* POTAL_GEMM_*; must be 0 */
} potal_gemm_ws;

typedef struct potal_rocc_cmd {
    uint64_t rs1;
    uint64_t rs2;
} potal_rocc_cmd;

/* Commands in issue order; funct7 values are in potal_rocc.h. */
typedef struct potal_gemm_ws_program {
    potal_rocc_cmd config_ld_a; /* CONFIG(0) CONFIG_LD, state 0 */
    potal_rocc_cmd config_ld_b; /* CONFIG(0) CONFIG_LD, state 1 */
    potal_rocc_cmd config_ex;   /* CONFIG(0) CONFIG_EX, WS */
    potal_rocc_cmd config_st;   /* CONFIG(0) CONFIG_ST */
    potal_rocc_cmd bounds;      /* LOOP_WS_CONFIG_BOUNDS(9) */
    potal_rocc_cmd addrs_ab;    /* LOOP_WS_CONFIG_ADDRS_AB(10) */
    potal_rocc_cmd addrs_dc;    /* LOOP_WS_CONFIG_ADDRS_DC(11) */
    potal_rocc_cmd strides_ab;  /* LOOP_WS_CONFIG_STRIDES_AB(12) */
    potal_rocc_cmd strides_dc;  /* LOOP_WS_CONFIG_STRIDES_DC(13) */
    potal_rocc_cmd loop_ws;     /* LOOP_WS(8) with PoTal fields */
} potal_gemm_ws_program;

/*
 * Validate a GEMM request. Order: argument errors, then hardware capacity
 * (scale window, work IDs, SPAD/ACC), then the validated scope. The validated
 * scope is DIM16, m == n == 16, k in {16, 32}; anything else that the RTL could
 * accept returns POTAL_ERR_UNSUPPORTED.
 */
potal_status potal_gemm_ws_check(const potal_gemm_ws *gemm);

/* Scale slot read for (K scale block, N tile): base + block * n_tiles + n_tile. */
potal_status potal_gemm_ws_scale_slot(const potal_gemm_ws *gemm, uint32_t k_block,
                                      uint32_t n_tile, uint32_t *slot);

/* potal_gemm_ws_check() followed by command encoding. */
potal_status potal_gemm_ws_encode(const potal_gemm_ws *gemm, potal_gemm_ws_program *program);

/* ---- Hardware operations (RISC-V only; host builds return NO_HARDWARE) -- */

/* SCALE_WRITE. Returns the hardware status (POTAL_OK or POTAL_HW_*). */
potal_status potal_scale_write(uint32_t slot, uint32_t generation, uint32_t column,
                               uint32_t carrier);

/*
 * SCALE_RELEASE. The hardware holds the command until all older work has
 * drained, so it may follow a GEMM directly; it does not clear the
 * issued-GEMM state that potal_fence() tracks.
 */
potal_status potal_scale_release(uint32_t slot, uint32_t generation, uint32_t column);

/* POTAL_FENCE: blocks until all older PoTal/Gemmini work and stores drained. */
potal_status potal_fence(void);

/*
 * Issue one GEMM (CONFIG + LOOP_WS sequence). POTAL_OK means issued, not
 * finished. Returns POTAL_ERR_BUSY if a previous GEMM was not fenced.
 */
potal_status potal_gemm_ws_issue(const potal_gemm_ws *gemm);

#ifdef __cplusplus
}
#endif

#endif /* POTAL_H */
