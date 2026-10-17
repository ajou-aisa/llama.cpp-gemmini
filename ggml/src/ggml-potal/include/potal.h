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
 * Not thread-safe: one thread per process may use this API, and on a
 * multi-hart Linux system it must be pinned to the CPU of the hart the PoTal
 * RoCC is attached to (see tests/potal-multicore-smoke.c).
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

/*
 * Flags requesting LOOP_WS features. potal_gemm_ws accepts none of them;
 * potal_loop_ws accepts POTAL_GEMM_ACCUMULATE only.
 */
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

/*
 * One general LOOP_WS invocation: the unit a tiler (sp_tiled_matmul_ws)
 * issues. Same command sequence and encoder as potal_gemm_ws, plus the
 * LoopMatmul fields potal_gemm_ws pins: tile counts, last-tile padding,
 * accumulation into the ACC and an optional C store.
 *
 *   A: int8,  (i_tiles*DIM - pad_i) rows x (k_tiles*DIM - pad_k) cols, lda
 *   B: int8,  (k_tiles*DIM - pad_k) rows x (j_tiles*DIM - pad_j) cols, ldb
 *   C: int32, (i_tiles*DIM - pad_i) rows x (j_tiles*DIM - pad_j) cols, ldc,
 *      or NULL: no C store and the ACC half is kept for the next K chunk
 *
 * flags == POTAL_GEMM_ACCUMULATE: K pass 0 adds into the ACC (ex_accumulate,
 * not a first contribution); flags == 0: K pass 0 overwrites it. Each K pass
 * p is scaled per output column by slot
 * scale_base + (p / passes_per_block) * j_tiles + j_tile (LoopMatmul order);
 * padded columns of the last J tile read no scale.
 *
 * NOT VALIDATED ON HARDWARE beyond the potal_gemm_ws scope: multiple I/J
 * tiles, padding, accumulation across invocations and C == NULL are encoded
 * from the RTL (LoopMatmul/PoTalWriteback) but have not passed a FireSim test.
 * A/B tile origins must be DIM-byte aligned and C POTAL_HW_DIM*4-byte aligned.
 */
typedef struct potal_loop_ws {
    const int8_t *a;
    const int8_t *b;
    int32_t      *c;
    size_t        lda;
    size_t        ldb;
    size_t        ldc;
    uint32_t      i_tiles;          /* DIM-row tiles of A and C */
    uint32_t      j_tiles;          /* DIM-column tiles of B and C */
    uint32_t      k_tiles;          /* DIM-wide K passes; multiple of passes_per_block */
    uint32_t      pad_i;            /* rows missing from the last I tile, < DIM */
    uint32_t      pad_j;            /* columns missing from the last J tile, < DIM */
    uint32_t      pad_k;            /* K elements missing from the last pass, < DIM */
    uint32_t      scale_base;
    uint32_t      scale_generation;
    uint32_t      passes_per_block;
    uint32_t      flags;            /* 0 or POTAL_GEMM_ACCUMULATE */
} potal_loop_ws;

/*
 * Validate one invocation: argument errors, then capacity (scale window,
 * work IDs, per-loop SPAD/ACC half), then unsupported flags.
 */
potal_status potal_loop_ws_check(const potal_loop_ws *loop);

/* potal_loop_ws_check() followed by command encoding. */
potal_status potal_loop_ws_encode(const potal_loop_ws *loop, potal_gemm_ws_program *program);

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
 * FLUSH(rs1 = 0, flush and retry): invalidate the accelerator-side DMA TLB.
 * Call once per process before the first GEMM, from the thread that owns the
 * accelerator, so no translation cached by an earlier process is reused.
 *
 * POTAL_OK means issued, not completed. FLUSH has no response; follow it with
 * potal_fence(): FLUSH and FENCE share the in-order raw command queue and
 * FENCE only answers after the unrolled command queue is empty, so a
 * POTAL_OK fence implies the TLB flush was applied.
 *
 *   potal_tlb_flush() == POTAL_OK  ->  potal_fence() == POTAL_OK  ->  GEMMs
 *
 * Returns POTAL_ERR_BUSY if a GEMM was issued and not yet fenced (flushing
 * under in-flight DMA is not supported). Does not make concurrent use of one
 * accelerator by several processes safe.
 */
potal_status potal_tlb_flush(void);

/*
 * Issue one GEMM (CONFIG + LOOP_WS sequence). POTAL_OK means issued, not
 * finished. Returns POTAL_ERR_BUSY if a previous GEMM was not fenced.
 */
potal_status potal_gemm_ws_issue(const potal_gemm_ws *gemm);

/*
 * Issue one LOOP_WS invocation through the same issue path as
 * potal_gemm_ws_issue(). POTAL_OK means issued; ERR_BUSY if not fenced.
 */
potal_status potal_loop_ws_issue(const potal_loop_ws *loop);

#ifdef __cplusplus
}
#endif

#endif /* POTAL_H */
