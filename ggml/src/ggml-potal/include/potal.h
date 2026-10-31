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
 * scope is DIM16 or DIM32, m == n == DIM, k in {DIM, 2*DIM}; anything else
 * that the RTL could accept returns POTAL_ERR_UNSUPPORTED.
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

/* ---- Planned stripe GEMM (library potal-tiled-api, C++ inside) ----------- */

/*
 * One logical GEMM of a stripe pipeline (a Main stripe GEMM or the Residual
 * GEMM of a stripe), executed with a tile plan its caller froze earlier:
 *
 *   C[m][n] = Sat32 sum over DIM-wide K passes p of
 *             SCU(sum_{k in p} A[m][k] * B[k][n], carriers[p*DIM/scale_block_k][n])
 *
 * on the Gemmini-compatible WS tiler (tiled_matmul -> sp_tiled_matmul_ws ->
 * potal_loop_ws_issue / potal_gemm_ws_issue). The K accumulation of an output
 * tile stays in the PoTal ACC across K chunks; C is written once, after the
 * last chunk. No tile search runs: tile_i/tile_j/tile_k are used as given,
 * except that tile_i is limited to the I tiles m needs (a tail stripe).
 *
 * Synchronous: POTAL_OK means every LOOP_WS invocation of this GEMM was
 * issued, fenced (POTAL_FENCE == POTAL_OK) and its scale entries released, so
 * C is complete. This differs from the issue functions above, whose POTAL_OK
 * only means "issued". A rejected descriptor issues no command at all.
 *
 * Not thread-safe, like the rest of this API: one thread, pinned to the CPU of
 * the hart the PoTal RoCC is attached to, owns a session and calls this.
 * TLB maintenance (potal_tlb_flush + potal_fence before fresh DMA buffers) is
 * the caller's job, as for every other GEMM.
 */
typedef struct potal_stripe_session potal_stripe_session; /* opaque */

typedef enum potal_stripe_work_kind {
    POTAL_STRIPE_WORK_MAIN     = 0,
    POTAL_STRIPE_WORK_RESIDUAL = 1,
} potal_stripe_work_kind;

typedef struct potal_stripe_gemm_desc {
    /* Identity, recorded only: never changes the arithmetic. */
    uint64_t        run_id;
    uint64_t        stripe_id;
    uint64_t        row_begin;      /* first original activation row of the stripe */
    uint32_t        work_kind;      /* potal_stripe_work_kind */

    uint32_t        m;              /* rows of A and C */
    uint32_t        n;              /* columns of B and C */
    uint32_t        k;              /* reduction extent of the A/B handed over */

    const int8_t   *a;              /* [m][lda], already at the stripe's first row */
    const int8_t   *b;              /* [k][ldb] */
    int32_t        *c;              /* [m][ldc], overwritten */
    size_t          lda;            /* element strides */
    size_t          ldb;
    size_t          ldc;

    const uint32_t *carriers;       /* [k / scale_block_k][carrier_stride] SCU carriers */
    size_t          carrier_stride; /* uint32 elements, >= n */
    uint32_t        scale_block_k;  /* K extent of one carrier: a multiple of DIM dividing k */

    uint32_t        tile_i;         /* frozen plan: DIM tiles per LOOP_WS, not elements */
    uint32_t        tile_j;
    uint32_t        tile_k;         /* a multiple of scale_block_k / DIM */
} potal_stripe_gemm_desc;

typedef struct potal_stripe_gemm_report {
    potal_status status;            /* POTAL_OK or the first failure */
    const char  *stage;             /* failing step (static string), NULL when ok */
    int          fatal;             /* 1: hardware state unknown; the session refuses later work */

    /* Commands this call completed with POTAL_OK; nothing refused is counted. */
    uint64_t     loop_ws;           /* LOOP_WS invocations issued */
    uint64_t     validated_gemms;   /* of which through potal_gemm_ws_issue */
    uint64_t     fences;
    uint64_t     scale_writes;
    uint64_t     scale_releases;
    uint32_t     last_generation;   /* 8-bit scale generation of the last invocation this call
                                       started, 0 if it started none */

    /* Geometry executed (tile_i after the tail limit). */
    uint32_t     tile_i, tile_j, tile_k;
    uint32_t     i0, j0, k0;        /* LOOP_WS invocations per dimension */
    uint32_t     pad_i, pad_j, pad_k;

    double       scale_s;           /* SCALE_WRITE/RELEASE, issue, fence: guest seconds */
    double       issue_s;
    double       fence_s;
} potal_stripe_gemm_report;

/*
 * Allocates a session (scale generation counter and work counters) bound to
 * the hardware. Starts no thread and issues no command. On failure *out is
 * NULL (when out is not NULL). One session serves one accelerator owner for
 * its whole life; destroy it only when no call on it is running. Destroying
 * frees the session alone: buffers of a GEMM that reported fatal must still
 * never be freed. potal_stripe_session_destroy(NULL) does nothing.
 */
potal_status potal_stripe_session_create(potal_stripe_session **out);
void         potal_stripe_session_destroy(potal_stripe_session *session);

/*
 * Run one planned logical GEMM to completion (see above). Checks before the
 * first command: session/desc/report non-NULL, the session not disabled,
 * work_kind, m/n/k > 0, strides, scale_block_k, tiles against the shape and
 * the per-invocation capacity, A/B DIM-byte and C DIM*4-byte alignment, every
 * operand extent within the 32-bit DRAM offset, C disjoint from A and B, and
 * every carrier a valid PoTal carrier. Not supported (rejected as everywhere
 * in this API): bias/D, transposes, activation functions, scale factors.
 */
potal_status potal_tiled_matmul_stripe(potal_stripe_session *session, const potal_stripe_gemm_desc *desc,
                                       potal_stripe_gemm_report *report);

/* ---- Command funct7 values (GemminiISA.scala); the raw issue layer and the
 * paired session's command records use them. ---------------------------------- */
#define POTAL_FUNCT_CONFIG                    0
#define POTAL_FUNCT_MVIN                      2
#define POTAL_FUNCT_MVOUT                     3
#define POTAL_FUNCT_FLUSH                     7
#define POTAL_FUNCT_LOOP_WS                   8
#define POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS     9
#define POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB   10
#define POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_DC   11
#define POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_AB 12
#define POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_DC 13
#define POTAL_FUNCT_SCALE_WRITE               23
#define POTAL_FUNCT_SCALE_RELEASE             24
#define POTAL_FUNCT_FENCE                     25
#define POTAL_FUNCT_INDEX_WRITE               26
#define POTAL_FUNCT_INDEX_CONFIG              27
#define POTAL_FUNCT_RESIDENT_VIEW             28
#define POTAL_FUNCT_LDB_ONLY                  29
#define POTAL_FUNCT_STREAM                    30
#define POTAL_FUNCT_COUNTER                  126

/* ---- H2 paired / indexed execution (potal_paired_*) ---------------------- */

/*
 * The paired Main/Residual GEMM of the H2 hardware (generators/potal, branch
 * potal; validated by potal_shared_weight_*.c, potal_h2b1_a_cohabit.c,
 * potal_h2c_*.c, potal_h2d_cycle_bench.c): two physical bank pairs, each one
 * activation bank + one weight bank + one Index Table, ACC0 (Main) and ACC1
 * (Residual) halves of the accumulator, INDEX_CONFIG / INDEX_WRITE and the
 * paired LOOP_WS bits. Planning (which pair, which sp rows, which scale window,
 * when a pair may be reused) is the job of matmul/paired-auto.hpp; this layer
 * checks fields and geometry, encodes the commands and issues them inside an
 * explicitly owned paired session.
 *
 * Address units: sp row = one row of POTAL_HW_DIM int8 (0 .. BANK_NUM*BANK_ROWS-1);
 * tile = POTAL_HW_DIM rows; resident_k = original K - the resident panel's first K.
 * Pair p: activation bank 2p, weight bank 2p+1, Index buffer p, LOOP_WS A/B
 * scratchpad selector p + 1 (LoopMatmul a/b_ex_spad_id: 1 = half 0, 2 = half 1).
 */
#define POTAL_PAIRS 2u

/*
 * INDEX_CONFIG (funct7 27, xd): rs1 = {jTiles[63:32], bBase[31:0]},
 * rs2 = {activeRows[63:32], sourceKRows[31:0]}. The hardware derives the
 * Index buffer from bBase (bank 1 -> buffer 0, bank 3 -> buffer 1) and
 * requires bBase + ceil(sourceKRows/DIM) * jTiles * DIM inside that weight bank.
 */
typedef struct potal_index_config {
    uint32_t pair;          /* physical bank pair 0..POTAL_PAIRS-1 = Index buffer */
    uint32_t b_sp_base;     /* first sp row of the resident B panel, inside the pair's weight bank */
    uint32_t j_tiles;       /* N tiles of the resident panel (its row-tile stride) */
    uint32_t source_k_rows; /* K rows of the resident panel: original_k - k_begin < source_k_rows */
    uint32_t active_rows;   /* Index Table slots of this window, 1..POTAL_HW_INDEX_ENTRIES */
} potal_index_config;

/* INDEX_WRITE (funct7 26, xd): rs1 = slot, rs2 = resident_k; targets the last configured buffer. */
typedef struct potal_index_write {
    uint32_t slot;       /* window-local, < active_rows */
    uint32_t resident_k; /* < source_k_rows: never a model-global K of a partitioned B */
} potal_index_write;

potal_status potal_index_config_check(const potal_index_config *cfg);
potal_status potal_index_config_encode(const potal_index_config *cfg, potal_rocc_cmd *cmd);
potal_status potal_index_write_check(const potal_index_config *cfg, const potal_index_write *w);
potal_status potal_index_write_encode(const potal_index_config *cfg, const potal_index_write *w,
                                      potal_rocc_cmd *cmd);

typedef enum potal_paired_path {
    POTAL_PAIRED_MAIN     = 0, /* A x B -> ACC0; loads A and the original B panel (LdA + LdB) */
    POTAL_PAIRED_RESIDUAL = 1, /* packed digits x resident B through the Index Table -> ACC1 */
} potal_paired_path;

/*
 * One paired LOOP_WS. `loop` carries the geometry, pointers, strides and scale
 * fields of the regular invocation (potal_loop_ws); the paired fields select
 * the path, the pair and the scratchpad placement. Rules (potal_paired_loop_ws_check):
 *   Main      resident_b 0, indexed 0, a_offset_tiles 0, loop.b the original B
 *             panel (k rows x n cols), (i_tiles*k_tiles) * DIM <= BANK_ROWS,
 *             k_tiles * j_tiles * DIM <= BANK_ROWS (one activation / weight bank)
 *   Residual  resident_b 1, indexed 1, loop.b NULL (B DRAM address 0: no LdB,
 *             LoopMatmul asserts it), loop.a the packed window digits
 *             [i_tiles*DIM - pad_i][k_tiles*DIM - pad_k] = the Index slots in order,
 *             a_offset_tiles the bank-local A offset (tile_I * tile_K of the plan),
 *             (a_offset_tiles + i_tiles*k_tiles) * DIM <= BANK_ROWS
 *   both      loop.c NULL (results leave the ACC through potal_acc_readback),
 *             i_tiles * j_tiles * DIM <= ACC_ROWS / 2 (one ACC bank), work IDs,
 *             scale window, pads < DIM, k_tiles a multiple of passes_per_block,
 *             loop.flags 0 or POTAL_GEMM_ACCUMULATE
 * The check validates fields and geometry only; whether B is resident, the
 * Index window programmed and Main A not overwritten is the paired session's
 * runtime check (potal_paired_loop_ws_issue).
 *
 * Encoding: the regular 10-command program with LOOP_WS rs1 bit 61 (paired),
 * bit 62 (residual), bit 63 (indexed weight), a_ex_spad_id = b_ex_spad_id =
 * pair + 1 at [19:18] / [17:16], and rs2[31:3] = a_offset_tiles.
 */
typedef struct potal_paired_loop_ws {
    potal_loop_ws loop;
    uint32_t      path;           /* potal_paired_path */
    uint32_t      pair;           /* 0..POTAL_PAIRS-1 */
    uint32_t      a_offset_tiles; /* bank-local A offset, DIM-row tiles */
    uint32_t      resident_b;     /* 1: reuse the pair's resident B, no B DMA */
    uint32_t      indexed;        /* 1: Indexed Weight PRELOAD through the pair's Index Table */
} potal_paired_loop_ws;

potal_status potal_paired_loop_ws_check(const potal_paired_loop_ws *l);
potal_status potal_paired_loop_ws_encode(const potal_paired_loop_ws *l, potal_gemm_ws_program *program);

/*
 * Full-width INT32 readback of one ACC bank's result tiles: CONFIG_ST(ldc
 * bytes) then one MVOUT (funct7 3) per DIM x DIM tile, rows (i, j) at ACC row
 * bank * ACC_ROWS/2 + (i * j_tiles + j) * DIM (the LoopMatmul tile layout),
 * last tiles shortened by pad_i / pad_j. c must be DIM*4-byte aligned with
 * ldc >= the valid columns. The readback is a DMA store: it is complete only
 * after a POTAL_OK fence, and only then may the CPU read c.
 */
typedef struct potal_acc_readback {
    uint32_t acc_bank; /* 0 Main (ACC0), 1 Residual (ACC1) */
    uint32_t i_tiles;
    uint32_t j_tiles;
    uint32_t pad_i;    /* rows missing from the last I tile, < DIM */
    uint32_t pad_j;    /* columns missing from the last J tile, < DIM */
    int32_t *c;        /* [i_tiles*DIM - pad_i][ldc] */
    size_t   ldc;      /* elements */
} potal_acc_readback;

potal_status potal_acc_readback_check(const potal_acc_readback *r);
/* config_st and up to `capacity` MVOUT commands (i_tiles * j_tiles needed; *count says how many). */
potal_status potal_acc_readback_encode(const potal_acc_readback *r, potal_rocc_cmd *config_st,
                                       potal_rocc_cmd *mvouts, uint32_t capacity, uint32_t *count);

/*
 * One final signed INT32 column-bias addition to an initialized ACC bank.
 * The same values[j] row is repeated over every valid output row. Hardware
 * performs Sat32(ACC + values[j]) once per lane, without another SCU shift.
 * Values must already be exactly representable INT32; this interface neither
 * clips a wider correction nor splits it into multiple saturating additions.
 *
 * Full-width LOAD_CMD with is_acc_addr=1, accumulate=1, shrink=0 and stride=0.
 * The descriptor covers the same bank-local tile layout as ACC readback.
 * values is DIM*4-byte aligned and spans j_tiles*DIM-pad_j signed32 elements.
 * It must remain mapped and untouched until the following successful fence.
 */
typedef struct potal_acc_bias {
    uint32_t acc_bank;
    uint32_t i_tiles;
    uint32_t j_tiles;
    uint32_t pad_i;
    uint32_t pad_j;
    const int32_t *values;
} potal_acc_bias;

potal_status potal_acc_bias_check(const potal_acc_bias *bias);
potal_status potal_acc_bias_encode(const potal_acc_bias *bias, potal_rocc_cmd *config_ld,
                                   potal_rocc_cmd *mvins, uint32_t capacity, uint32_t *count);

/*
 * Paired session: the single owner of the accelerator while open. Opening
 * refuses (POTAL_ERR_BUSY) while a regular GEMM is unfenced or another session
 * is open; while open, potal_gemm_ws_issue / potal_loop_ws_issue / potal_fence /
 * potal_tlb_flush refuse with POTAL_ERR_BUSY. Inside it several paired LOOP_WS
 * may be outstanding; only potal_paired_fence() == POTAL_OK means they (and
 * every readback DMA) completed. Not thread-safe; same hart-0 rule as the rest.
 *
 * Commands go to the hardware (RISC-V) or, when `sink` is given, to the sink in
 * issue order (host tests: a record of the command stream, never a model of the
 * RTL's concurrency). Without hardware and without a sink every issue returns
 * POTAL_ERR_NO_HARDWARE after its checks and changes nothing.
 *
 * On RISC-V, POTAL_RAW_TRACE selects ENTER/EXIT command records when its value
 * starts with '1'. Selection is a session-open snapshot: environment changes
 * take effect at the next session, not during an active session.
 *
 * Runtime rules of the session (beyond the field checks):
 *   Main LOOP_WS       its pair must have no outstanding paired work (a fence
 *                      retired every reader of the pair's previous B / A);
 *                      records the pair's resident B (base, j tiles, K rows) and
 *                      Main A footprint
 *   INDEX_CONFIG       starts one transaction on its pair; refused while another
 *                      transaction is unfinished (writes < active_rows). Any pair
 *                      may be configured while the other pair executes
 *                      (cross-buffer, H2-2C); a same-buffer update is stalled by
 *                      the hardware until that buffer is idle
 *   INDEX_WRITE        inside the open transaction, slot / resident_k checked
 *                      against it; the transaction completes with its last slot
 *   Residual LOOP_WS   its pair must hold a resident B whose base, j tiles and K
 *                      rows equal the pair's completed Index window, the window's
 *                      active rows must equal k_tiles*DIM - pad_k, and the A offset
 *                      must lie behind the pair's Main A footprint
 *   scale write        refused (POTAL_ERR_CAPACITY) while the slot holds live
 *                      columns of another generation, or the column itself is
 *                      live (written, not released): generations never alias;
 *                      liveness is per (slot, column), so a padded last J tile
 *                      that writes fewer than DIM columns frees its slot when
 *                      its last written column is released; release needs the
 *                      live generation and column (POTAL_ERR_INVALID_ARGUMENT).
 *                      The hardware accepts a RELEASE only after every older
 *                      command drained (Controller.scala releaseSafe :=
 *                      allDrained, the FENCE condition): an accepted release
 *                      is a drain point and clears the outstanding work like
 *                      potal_paired_fence() does
 *   readback           outstanding until the next POTAL_OK fence
 *   failure            a hardware status other than POTAL_OK, or a bad response,
 *                      makes the session failed: the error and stage are kept,
 *                      every further issue returns that error, outstanding work
 *                      is never forgotten; potal_paired_fence() may still drain it
 *   close              POTAL_ERR_BUSY while work is outstanding or scale entries
 *                      are live and the session is not failed; a failed session
 *                      with outstanding work or live scale entries closes, but the
 *                      accelerator stays owned (every later open and regular issue
 *                      returns POTAL_ERR_BUSY): its state is unknown
 */
typedef struct potal_paired_session potal_paired_session; /* opaque */

typedef struct potal_paired_cmd_record {
    uint32_t funct;  /* potal_rocc.h POTAL_FUNCT_* */
    uint32_t xd;     /* 1: the hardware answers with a status */
    uint64_t rs1;
    uint64_t rs2;
} potal_paired_cmd_record;

/* Returns the (simulated) hardware status of an xd command; POTAL_OK for the others. */
typedef potal_status (*potal_paired_sink_fn)(void *ctx, const potal_paired_cmd_record *record);

typedef struct potal_paired_sink {
    potal_paired_sink_fn cmd;
    void                *ctx;
} potal_paired_sink;

typedef struct potal_paired_pair_status {
    uint32_t resident_b;    /* a Main LOOP_WS loaded the panel below */
    uint32_t b_sp_base;
    uint32_t b_j_tiles;
    uint32_t b_k_rows;
    uint32_t main_a_tiles;  /* Main A footprint, tiles from the bank start */
    uint32_t index_valid;   /* a complete Index window is programmed on this buffer */
    uint32_t index_active_rows;
    uint32_t outstanding;   /* paired LOOP_WS issued on this pair since the last POTAL_OK fence */
} potal_paired_pair_status;

typedef struct potal_paired_session_status {
    uint32_t     failed;
    potal_status error;
    const char  *error_stage;         /* NULL when not failed */
    uint32_t     outstanding_loops;
    uint32_t     outstanding_readbacks;
    uint32_t     outstanding_acc_updates; /* INT32 MVIN bias DMA, cleared only by a successful drain */
    uint32_t     index_transaction_open; /* a CONFIG whose writes are not complete */
    uint32_t     index_transaction_pair;
    uint32_t     index_transaction_written;
    uint32_t     live_scale_slots;
    uint64_t     commands;            /* raw commands handed to the hardware / sink */
    uint64_t     drains;              /* accepted POTAL_FENCE and SCALE_RELEASE commands: drain points */
    potal_paired_pair_status pair[POTAL_PAIRS];
} potal_paired_session_status;

potal_status potal_paired_session_open(const potal_paired_sink *sink, potal_paired_session **out);
potal_status potal_paired_session_close(potal_paired_session *session);
potal_paired_session_status potal_paired_session_query(const potal_paired_session *session);

potal_status potal_paired_tlb_flush(potal_paired_session *session); /* POTAL_ERR_BUSY while work is outstanding */
potal_status potal_paired_scale_write(potal_paired_session *session, uint32_t slot, uint32_t generation,
                                      uint32_t column, uint32_t carrier);
potal_status potal_paired_scale_release(potal_paired_session *session, uint32_t slot, uint32_t generation,
                                        uint32_t column);
potal_status potal_paired_index_config(potal_paired_session *session, const potal_index_config *cfg);
potal_status potal_paired_index_write(potal_paired_session *session, const potal_index_write *w);
potal_status potal_paired_loop_ws_issue(potal_paired_session *session, const potal_paired_loop_ws *loop);
potal_status potal_paired_acc_readback_issue(potal_paired_session *session, const potal_acc_readback *r);
/* Requires drained prior work. Later LOOP_WS, readback, flush and close require its fence. */
potal_status potal_paired_acc_bias_issue(potal_paired_session *session, const potal_acc_bias *bias);
/* POTAL_FENCE: POTAL_OK means every outstanding paired LOOP_WS and readback drained. */
potal_status potal_paired_fence(potal_paired_session *session);

/*
 * Restores a known accelerator state after a failure the hardware answered
 * (an error status of an xd command): POTAL_FENCE if work is outstanding,
 * then SCALE_RELEASE of every live (slot, column) this session wrote. The
 * session stays failed (later issues return its error; close returns it).
 * POTAL_OK: nothing outstanding, no live scale entry, close releases the
 * accelerator. Any other status: the state is unknown (the accelerator stays
 * owned after close). Never call it after a failed POTAL_FENCE: a fence that
 * did not answer OK is not retried by this API.
 */
potal_status potal_paired_session_cleanup(potal_paired_session *session);

/* ---- Opt-in managed H2 panels (physical DIM32, ordinary H2 ISA) ----------
 *
 * Begin upgrades a clean paired session, mutually exclusive with resident_ws.
 * The descriptor is copied; its entire borrowed allocation must stay mapped
 * and immutable until h2_close consumes the session. Quarantine retains that
 * obligation indefinitely, together with all A/readback/sink dependencies.
 * No capability probe, funct28, extended run or raw escape is introduced.
 *
 * A panel declares one Main reader (ordinal 0), then readers-1 Residual
 * readers in ordinal order. The immutable, prevalidated caller plan owns the
 * exact ACC and fragment manifest. This interface checks physical geometry,
 * B identity, reader counts, Index bindings and live Scale generations.
 * Begin publishes the entire borrowed B allocation with a CPU memory fence
 * while the session is idle. Its immutable lifetime permits later prefetch
 * without another CPU fence (Rocket CPU fences stall on RoCC busy).
 * Normal Main loads real B and may chain Residual without a new barrier.
 * A prefetch issues CONFIG_LD state 0 and one DIM tile per MVIN; while it is
 * pending, only completion, current ACC readback and Scale Release may issue.
 * panel_begin(prefetch=0) may reserve the next panel before current issue.
 * panel_prefetch loads that still-unused prepared reference without repeating
 * descriptor validation. Pure refusals preserve it for corrected continuation;
 * preparation is not cancellable and does not relax retirement or close.
 * Main always supplies its REAL ordinary descriptor, even after prefetch:
 * after full validation only its encoded B address is suppressed.
 *
 * Issue success is not completion. Only successful checked global Fence or
 * live Scale Release publishes fully issued B and completes issued readers.
 * get is observation only: BUSY until published, INVALID_ARGUMENT for stale
 * refs. Retirement requires all declared readers issued/completed, no global
 * pending/control/source work, closed Index transaction and zero live Scale
 * columns. It clears that generation's B/Index only, never ACC contents.
 * Save the retirement receipt in the caller; never retire a newer generation
 * when the old plan later reaches its logical retirement operation.
 *
 * Every attempted command is owned before effects, including partial loads,
 * programs and readbacks. First error is sticky. One explicit cleanup Fence
 * can recover an answered failure; successful cleanup discards identities
 * and releases actual live Scale columns. A failed Fence/Release or cleanup
 * quarantines ownership: no retry, publication, close, free or fallback.
 * Failed cleaned close consumes *session but returns the original error.
 * Sink callbacks may query status; reentrant mutation returns BUSY.
 */
typedef struct potal_h2_weight {
    const int8_t *base;
    uint64_t k, n;                 /* original prepared matrix extents */
    size_t ldb, bytes;             /* real row stride and accessible allocation */
} potal_h2_weight;

typedef struct potal_h2_panel {
    uint64_t residency_index, k_begin, j_begin; /* original element offsets */
    uint32_t pair, k_tiles, j_tiles, pad_j;
    uint32_t main_i_tiles, main_pad_i;
    uint32_t readers;              /* 1 Main + all declared Residual windows */
} potal_h2_panel;

typedef struct potal_h2_ref {
    uint64_t owner, generation;    /* nonzero, nonwrapping software incarnations */
    uint32_t pair;
} potal_h2_ref;

typedef struct potal_h2_panel_status {
    potal_h2_ref ref;
    uint64_t residency_index;
    uint32_t declared, prefetched, loading, readable;
    uint32_t readers_declared, readers_issued, readers_completed;
} potal_h2_panel_status;

typedef struct potal_h2_status {
    potal_paired_session_status core;
    uint32_t active, source_bound, pending_b, source_pending, control_pending;
    uint32_t unknown_completion, quarantined, cleanup_attempted, cleaned;
    potal_status last_drain, cleanup_status;
    uint64_t attempted_b_mvins;
    potal_h2_panel_status panel[POTAL_PAIRS];
    /* POTAL_H2_COMPLETION_REUSE=1, sampled at managed begin; default off.
     * Reuse skips only unchanged host reconciliation, never a command or CPU
     * fence. Counters describe successful checked drains, not hardware time. */
    uint32_t completion_reuse;
    uint64_t completion_calls, completion_reuses, completion_bookkeeping, completion_cpu_fences;
    uint32_t early_prefetch;
    uint64_t attempted_ldb_only;
} potal_h2_status;

potal_status potal_paired_h2_begin(potal_paired_session *session, const potal_h2_weight *weight);
potal_status potal_paired_h2_panel_begin(potal_paired_session *session,
    const potal_h2_panel *panel, uint32_t prefetch, potal_h2_ref *reference);
potal_status potal_paired_h2_panel_prefetch(potal_paired_session *session, potal_h2_ref reference);
potal_status potal_paired_h2_loop_issue(potal_paired_session *session,
    potal_h2_ref reference, uint32_t reader_ordinal, const potal_paired_loop_ws *loop);
potal_status potal_paired_h2_panel_get(potal_paired_session *session, potal_h2_ref reference);
potal_status potal_paired_h2_panel_retire(potal_paired_session *session, potal_h2_ref reference);
potal_h2_status potal_paired_h2_query(const potal_paired_session *session);
potal_status potal_paired_h2_close(potal_paired_session **session);

/* Diagnostic-only observation on the same issuer. Begin requires a new idle
 * session; end requires all commands drained and Scale entries released.
 * Seven target-clock cycle counts: B0/B1 physical writes, pair0/pair1 PE00 MAC,
 * MAC0&B1, MAC1&B0, either overlap. These are not completion certificates.
 * A software sink must supply counter with the same borrowed sink.ctx.
 * No timing or hardware claim follows from software-sink values. */
#define POTAL_PREFETCH_OBSERVATION_TAG UINT64_C(0x504f425300000000)
#define POTAL_PREFETCH_OBSERVATION_CAPABILITY UINT64_C(0x504f425301200007)
typedef potal_status (*potal_observation_counter_fn)(void *ctx,
    const potal_paired_cmd_record *record, uint64_t *value);

/* Explicit streaming mode; borrowed inputs are immutable from begin through
 * successful end. A quarantined session retains every borrowed allocation.
 * Main/residual stride is fixed for the entire stream, in bytes.
 *
 * Begin requires a clean exclusive session, probes COUNTER safely on old
 * images, then installs EX and LOAD states 0/1/2 once. Every loop sends only
 * five LOOP_WS staging words and the complete ordinary paired LOOP_WS.
 * Scale columns must already be live; same-panel consumers may share one
 * allocation/generation, but distinct live allocations cannot alias that
 * generation. Index may follow scheduled Main B
 * before physical completion. Replacing a consumed Index/A window requires
 * wait_pair (or a global Fence) first.
 *
 * wait_pair completes the submitted prefix, not other pairs or readbacks.
 * Scoped release requires a sealed panel and never certifies global completion. Retirement additionally
 * requires every declared consumer submitted/completed and its scales freed.
 * Readback requires a global drain; end requires retired panels, no live
 * scales and a final checked Fence after readback. End keeps the ordinary
 * session open for session_close.
 *
 * Invalid arguments issue nothing. Unsupported begin emits only the safe
 * COUNTER probe. A command/response failure is sticky and quarantines the
 * stream: cleanup, end and close cannot retry, free or permit fallback.
 * An absent residual source is {NULL, 0, 0}; later Residual submissions are
 * rejected. The final LOOP_WS uses xd=1; its five staging words use xd=0. */
typedef struct potal_stream_source {
    const int8_t *base;
    size_t bytes, stride;
} potal_stream_source;

typedef struct potal_stream_config {
    potal_h2_weight weight;
    potal_stream_source main, residual;
    potal_observation_counter_fn counter; /* software sink only; sink.ctx */
    uint32_t continuous;           /* 0: bounded, 1: pair-local retirement */
    uint32_t early_b_only;         /* default 0; requires continuous and its exact capability */
} potal_stream_config;

#define POTAL_STREAM_TAG UINT64_C(0x4853384800000000)
#define POTAL_STREAM_CAPABILITY UINT64_C(0x4853384801200803)
#define POTAL_STREAM_EARLY_TAG UINT64_C(0x4853384801000000)
#define POTAL_STREAM_EARLY_CAPABILITY UINT64_C(0x4853384801200807)

typedef struct potal_stream_panel_status {
    potal_h2_ref ref;
    uint64_t residency_index;
    uint32_t declared, scheduled, dma_done, sealed;
    uint32_t submitted, completed, readers;
} potal_stream_panel_status;

typedef struct potal_stream_status {
    potal_paired_session_status core;
    uint32_t active, quarantined, control_pending;
    potal_stream_panel_status panel[POTAL_PAIRS];
} potal_stream_status;

potal_status potal_paired_stream_begin(potal_paired_session *, const potal_stream_config *);
potal_status potal_paired_stream_panel_begin(potal_paired_session *, const potal_h2_panel *, potal_h2_ref *);
/* Optional, Scale-free B producer for the adjacent same-J next-K panel.
 * The other pair must have exactly its Main submitted. Target admission owns
 * the source but completes no readers. A successful target wait_pair is required
 * before Main may consume the prefetched B; it does not drain the other pair.
 * Index remains unavailable until Main is submitted. Dispatched failures
 * quarantine the stream even when no compute-loop count is outstanding. */
potal_status potal_paired_stream_panel_prefetch(potal_paired_session *, potal_h2_ref target);
potal_status potal_paired_stream_loop_issue(potal_paired_session *, potal_h2_ref,
    uint32_t ordinal, const potal_paired_loop_ws *);
potal_status potal_paired_stream_wait_pair(potal_paired_session *, potal_h2_ref);
potal_status potal_paired_stream_scale_release(potal_paired_session *, potal_h2_ref,
    uint32_t slot, uint32_t generation, uint32_t column);
potal_status potal_paired_stream_panel_retire(potal_paired_session *, potal_h2_ref);
potal_status potal_paired_stream_end(potal_paired_session *);
potal_stream_status potal_paired_stream_query(const potal_paired_session *);
/* Diagnostic only: existing target-clock events 10,19,20,41,27,29,31.
 * Configure before declaring panels; snapshot only after final global Fence.
 * Events overlap: never sum them as graph latency or treat them as bus bytes. */
#define POTAL_STREAM_COUNTER_FIELDS 7
potal_status potal_paired_stream_counters_begin(potal_paired_session *);
potal_status potal_paired_stream_counters_end(potal_paired_session *,
    uint64_t values[POTAL_STREAM_COUNTER_FIELDS]);
/* Opt-in scheduler-owned LOAD2, never raw MVIN or future compute. Enable only
 * on a new idle H2 session after an exact capability probe. The existing
 * panel_prefetch then requires the adjacent same-J next-K panel, an issued
 * Main on the other pair, and a later checked global drain before consumers.
 * Its status response acknowledges admission only. No CPU fence is omitted. */
#define POTAL_LDB_ONLY_TAG UINT64_C(0x504f4c4400000000)
#define POTAL_LDB_ONLY_CAPABILITY UINT64_C(0x504f4c4401200001)
#define POTAL_LDB_ONLY_TRACE_RESET (POTAL_LDB_ONLY_TAG | UINT64_C(0x100))
#define POTAL_LDB_ONLY_TRACE_READ (POTAL_LDB_ONLY_TAG | UINT64_C(0x200))
#define POTAL_LDB_ONLY_TRACE_FIELDS 28
potal_status potal_paired_h2_early_enable(potal_paired_session *session,
    potal_observation_counter_fn counter);
/* Diagnostic only: reset while globally drained, before Main; snapshot only
 * after the added checked Fence. Frozen target-clock events are not DMA
 * request timestamps, completion certificates, or performance samples.
 * Fields: flags, elapsed, scheduler admission; first/last B0, B1, MAC0,
 * MAC1; seven physical cycle counts; Fence request/accept; first/last busy;
 * absolute start/end; first/last accepted DMA requests for pair0 and pair1
 * (LoadController -> Scratchpad, not TileLink issue or DRAM completion).
 * Flags: active bit0, overflow bit1, seen bits8..17
 * (admission,B0,B1,MAC0,MAC1,Fence request,Fence accept,busy,DMA0,DMA1), pair bit20. */
potal_status potal_paired_h2_early_trace_begin(potal_paired_session *session);
potal_status potal_paired_h2_early_trace_end(potal_paired_session *session,
    uint64_t values[POTAL_LDB_ONLY_TRACE_FIELDS]);
potal_status potal_paired_observation_begin(potal_paired_session *session,
    potal_observation_counter_fn counter);
potal_status potal_paired_observation_end(potal_paired_session *session, uint64_t cycles[7]);

/* ---- Opt-in resident I1/P1 interface (fixed DIM16/SP4x4096/ACC1024) ------
 *
 * Uses the SAME opaque session and paired owner, not the test-only E4A
 * potal_resident_handle. Ordinary paired issuers/close/flush are blocked in
 * this mode. Checked paired Fence, Scale Release and cleanup remain shared.
 * One DMA transaction at a time; no prefetch/overlap or scheduler adoption.
 *
 * Hardware open probes the known base COUNTER126 ISA, BEFORE any extension:
 * setup rs1=0xA clears counter0 configuration/value and snapshot selection;
 * exact tagged read must return POTAL_RESIDENT_WS_CAPABILITY. This probe is
 * not safe discovery of an arbitrary unknown accelerator. It runs under the
 * paired owner before workload counters are configured. No environment
 * override exists. Native null-options open returns NO_HARDWARE.
 *
 * Non-NULL options select SOFTWARE-ONLY command emulation, even on RISC-V.
 * The old status sink ABI is unchanged. counter supplies a 64-bit response
 * for the two exact counter commands, using sink.ctx. On successful open the
 * session adopts that non-NULL context and its destructor; on failed open
 * the caller retains it. Quarantine never destroys context, buffers or owner.
 * Callbacks may query status, but reentrant mutation returns BUSY.
 *
 * All returned buffer aliases must remain untouched during pending work.
 * Retained buffers are six aligned 64 KiB pools: B0/B1, Main/Residual A,
 * and two private readback pools. Only the four input pools are mutable.
 * Result aliases expire on a later writer/readback of that bank or close.
 * Open/prepare/use/cleanup/close belong inside one issuer_owner callback.
 */
#define POTAL_RESIDENT_WS_CAPABILITY UINT64_C(0x504f544102100007)
#define POTAL_RESIDENT_WS_COUNTER_TAG UINT64_C(0x504f544100000000)
#define POTAL_RESIDENT_WS_POOL_BYTES 65536u
#define POTAL_RESIDENT_WS_MAX_J 9u
#define POTAL_RESIDENT_WS_MAX_K 28u

typedef struct potal_resident_ws_ref {
    uint64_t owner, generation; /* nonzero, nonwrapping software incarnation */
} potal_resident_ws_ref;

typedef potal_status (*potal_resident_ws_counter_fn)(
    void *ctx, const potal_paired_cmd_record *record, uint64_t *response);
typedef struct potal_resident_ws_options {
    potal_paired_sink sink;
    void (*destroy_sink)(void *);
    potal_resident_ws_counter_fn counter;
} potal_resident_ws_options;

typedef enum potal_resident_ws_buffer {
    POTAL_RESIDENT_WS_B0 = 0, POTAL_RESIDENT_WS_B1 = 1,
    POTAL_RESIDENT_WS_MAIN_A = 2, POTAL_RESIDENT_WS_RESIDUAL_A = 3
} potal_resident_ws_buffer;

typedef struct potal_resident_ws_load {
    uint32_t pair, k_rows, j_columns, main_a_tiles;
    uint64_t k_begin, j_begin, weight_id, weight_generation;
    size_t source_stride; /* bytes in the owned B pool; base offset is zero */
} potal_resident_ws_load;

/* Exact funct28 payload. Disabled means ALL fields zero. mode must be 1
 * when enabled. Bases and pitch are DIM16 rows, ACC base is bank-local. */
typedef struct potal_resident_ws_view {
    uint32_t enable, pair, indexed, acc_bank, mode;
    uint32_t b_base, b_pitch, indexed_j_origin, acc_base;
} potal_resident_ws_view;

typedef struct potal_resident_ws_scales {
    potal_resident_ws_ref resident;
    uint32_t base, generation, partials, j_columns;
    uint64_t j_begin;
    const uint64_t *original_blocks; /* one block32 ID per ordered P1 partial */
    const uint32_t *carriers;        /* [partials][carrier_stride] */
    size_t carrier_stride;          /* uint32_t elements, >= j_columns */
} potal_resident_ws_scales;

typedef struct potal_resident_ws_indices {
    potal_resident_ws_ref resident;
    uint32_t count;                 /* 1..32, including interior filler */
    const uint64_t *original_k;     /* copied; filler may repeat valid K */
} potal_resident_ws_indices;

typedef struct potal_resident_ws_fragment {
    uint32_t count;                 /* 1..16 meaningful entries, one block32 */
    uint64_t original_k[16];        /* strictly increasing meaningful entries */
} potal_resident_ws_fragment;

/* Immutable per-physical-tile declaration, copied by epoch_begin.
 * Main: fragments=NULL/count=0, total partials derived from logical K.
 * Residual: exact ordered fragment manifest; legitimate gaps are explicit.
 * No run can change its logical extent, contribution count or identities.
 */
typedef struct potal_resident_ws_epoch {
    uint32_t path, acc_tile, rows, columns;
    uint64_t output_id, original_i, original_j, weight_id, weight_generation;
    uint64_t logical_k_begin, logical_k_end;
    size_t fragment_count;
    const potal_resident_ws_fragment *fragments;
} potal_resident_ws_epoch;

typedef struct potal_resident_ws_run {
    potal_resident_ws_ref resident, scale, index; /* index all-zero for native */
    potal_resident_ws_ref epochs[POTAL_RESIDENT_WS_MAX_J];
    uint32_t path, indexed, k_offset, j_offset, k_rows, j_columns, i_rows;
    uint32_t a_offset_tiles, scale_base, scale_generation, acc_tile;
    size_t a_offset, a_stride;      /* bytes in owned path A pool */
    uint64_t output_id, original_i, sequence; /* sequence counts DIM partials */
    uint32_t first, last;
    uint32_t fragment_counts[2];    /* indexed only; final equals shape tail */
} potal_resident_ws_run;

typedef struct potal_resident_ws_program {
    potal_gemm_ws_program loop;     /* baseline nine configs + extended run */
    potal_rocc_cmd view;            /* inserted immediately before run */
} potal_resident_ws_program;

typedef struct potal_resident_ws_readback {
    uint32_t bank, acc_tile, rows, columns; /* I1, <=9 contiguous output tiles */
    size_t stride;                        /* INT32 elements in private pool */
    potal_resident_ws_ref epochs[POTAL_RESIDENT_WS_MAX_J];
    uint64_t sequences[POTAL_RESIDENT_WS_MAX_J]; /* completed contribution counts */
} potal_resident_ws_readback;

typedef enum potal_resident_ws_state {
    POTAL_RESIDENT_WS_NEW, POTAL_RESIDENT_WS_READY, POTAL_RESIDENT_WS_FAILED,
    POTAL_RESIDENT_WS_CLEANED, POTAL_RESIDENT_WS_QUARANTINED
} potal_resident_ws_state;

typedef struct potal_resident_ws_status {
    potal_paired_session_status core;
    uint32_t state, software_only, capability_matched, pending, control_pending;
    uint32_t unknown_fence, cleanup_attempted;
    potal_status last_drain, cleanup_status;
    uint64_t owned_bytes, attempted_b_mvins;
    potal_resident_ws_ref residents[POTAL_PAIRS];
    uint32_t readable[POTAL_PAIRS], loading[POTAL_PAIRS];
} potal_resident_ws_status;

/* Pure checks/encoders: no session or issue, output unchanged on rejection.
 * Run encoding checks physical geometry and A extent. The session's separate
 * run_check also checks live bindings, immutable epochs and filler bytes.
 */
/* Mathematical shifts must use this check before narrowing to carrier bits.
 * Shift zero is identity; POTAL_CARRIER_ZERO remains a separate encoded value. */
potal_status potal_resident_ws_shift_carrier(int64_t shift, uint32_t *carrier);
potal_status potal_resident_ws_load_check(const potal_resident_ws_load *load);
potal_status potal_resident_ws_view_encode(const potal_resident_ws_view *view, potal_rocc_cmd *out);
potal_status potal_resident_ws_run_check(const potal_resident_ws_load *load,
    const potal_resident_ws_run *run, const int8_t *activation, size_t bytes);
potal_status potal_resident_ws_run_encode(const potal_resident_ws_load *load,
    const potal_resident_ws_run *run, const int8_t *activation, size_t bytes,
    potal_resident_ws_program *out);
potal_status potal_resident_ws_readback_encode(const potal_resident_ws_readback *readback,
    int32_t *destination, size_t bytes, potal_rocc_cmd *config_st,
    potal_rocc_cmd *mvouts, uint32_t capacity, uint32_t *count);

potal_status potal_paired_resident_open(const potal_resident_ws_options *options,
    potal_paired_session **out);
potal_status potal_paired_resident_prepare(potal_paired_session *session);
potal_status potal_paired_resident_buffer_view(potal_paired_session *session,
    potal_resident_ws_buffer which, void **data, size_t *bytes);
potal_status potal_paired_resident_load_issue(potal_paired_session *session,
    const potal_resident_ws_load *load);
potal_status potal_paired_resident_get(potal_paired_session *session, uint32_t pair,
    potal_resident_ws_ref *resident);
potal_status potal_paired_resident_scale_program(potal_paired_session *session,
    const potal_resident_ws_scales *scales, potal_resident_ws_ref *scale);
potal_status potal_paired_resident_index_program(potal_paired_session *session,
    const potal_resident_ws_indices *indices, potal_resident_ws_ref *index);
potal_status potal_paired_resident_epoch_begin(potal_paired_session *session,
    const potal_resident_ws_epoch *epoch, potal_resident_ws_ref *reference);
potal_status potal_paired_resident_run_check(const potal_paired_session *session,
    const potal_resident_ws_run *run);
potal_status potal_paired_resident_issue(potal_paired_session *session,
    const potal_resident_ws_run *run);
/* Receipt is reserved at issue, readable only after successful global drain.
 * It binds every tile's epoch incarnation AND completed contribution count. */
potal_status potal_paired_resident_readback_issue(potal_paired_session *session,
    const potal_resident_ws_readback *readback, potal_resident_ws_ref *receipt);
potal_status potal_paired_resident_result(potal_paired_session *session,
    potal_resident_ws_ref receipt, const int32_t **data, size_t *stride);
potal_status potal_paired_resident_retire(potal_paired_session *session,
    potal_resident_ws_ref resident);
/* Refused close retains *session. Safe cleaned close consumes it but returns
 * the original sticky error. Quarantine has no retry/free/close operation. */
potal_status potal_paired_resident_close(potal_paired_session **session);
potal_resident_ws_status potal_paired_resident_query(const potal_paired_session *session);

#ifdef __cplusplus
}
#endif

#endif /* POTAL_H */
