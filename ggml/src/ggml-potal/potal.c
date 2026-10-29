/* PoTal minimal RoCC API. See include/potal.h for the contract. */

#include "potal.h"

#include "potal_hw_params.h"

#include <string.h>
#if defined(__riscv) || defined(POTAL_TEST_RAW_TRACE)
#include <stdio.h>
#endif

#if defined(__riscv)
#include "potal_rocc.h"
#endif

/*
 * Cross-check the CMake-parsed adapter against the original RTL header with
 * the C type system. The generic Gemmini names stay private to this file.
 */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#endif
#include "original-potal-params.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

_Static_assert(XCUSTOM_ACC == POTAL_HW_XCUSTOM, "XCUSTOM_ACC differs from adapter");
_Static_assert(DIM == POTAL_HW_DIM, "DIM differs from adapter");
_Static_assert(ADDR_LEN == POTAL_HW_ADDR_LEN, "ADDR_LEN differs from adapter");
_Static_assert(BANK_NUM == POTAL_HW_BANK_NUM, "BANK_NUM differs from adapter");
_Static_assert(BANK_ROWS == POTAL_HW_BANK_ROWS, "BANK_ROWS differs from adapter");
_Static_assert(ACC_ROWS == POTAL_HW_ACC_ROWS, "ACC_ROWS differs from adapter");
_Static_assert(MAX_BYTES == POTAL_HW_MAX_BYTES, "MAX_BYTES differs from adapter");
_Static_assert(sizeof(elem_t) == 1 && (elem_t)-1 < 0, "PoTal API requires signed 8-bit elem_t (A8W8)");
_Static_assert(sizeof(acc_t) == 4 && (acc_t)-1 < 0, "PoTal API requires signed 32-bit acc_t");

/* ---- RTL field layout ---------------------------------------------------- */

/* PoTalScaleCommandIngress: rs1 = {reserved[63:22], column[21:16], generation[15:8], slot[7:0]}. */
#define SCALE_GENERATION_SHIFT 8
#define SCALE_COLUMN_SHIFT     16

/* LoopMatmul LOOP_WS rs1 (GemminiISA + PoTal extension). */
#define LOOP_WS_FULL_C_SHIFT         1
#define LOOP_WS_POTAL_ENABLE_SHIFT   20
#define LOOP_WS_SCALE_BASE_SHIFT     21 /* [36:21] */
#define LOOP_WS_GENERATION_SHIFT     37 /* [44:37] */
#define LOOP_WS_PASSES_SHIFT         45 /* [60:45] */
#define LOOP_WS_FIELD16_MAX          0xFFFFu

/* Paired LOOP_WS (LoopMatmul H2): rs1[17:16] b_ex_spad_id, rs1[19:18] a_ex_spad_id (pair + 1),
 * rs1[61] paired ACC, rs1[62] residual, rs1[63] indexed weight; rs2[31:3] A bank-local offset in tiles. */
#define LOOP_WS_B_SPAD_SHIFT         16
#define LOOP_WS_A_SPAD_SHIFT         18
#define LOOP_WS_PAIRED_SHIFT         61
#define LOOP_WS_RESIDUAL_SHIFT       62
#define LOOP_WS_INDEXED_SHIFT        63
#define LOOP_WS_A_OFFSET_SHIFT       3
#define LOOP_WS_A_OFFSET_MAX         0x1FFFFFFFu /* 29 bits */

/* INDEX_CONFIG rs1 = {jTiles[63:32], bBase[31:0]}, rs2 = {activeRows[63:32], sourceKRows[31:0]}. */
#define INDEX_HIGH_SHIFT 32

/* MVOUT rs2 = {rows[63:48], cols[47:32], LocalAddr[31:0]}; LocalAddr: [31] acc, [30] accumulate,
 * [29] read_full_acc_row (the ACC readback of the H2 tests: 5 << (ADDR_LEN - 3)). */
#define MVOUT_ROWS_SHIFT        48
#define MVOUT_COLS_SHIFT        32
#define LOCAL_ADDR_ACC          ((uint64_t)1 << 31)
#define LOCAL_ADDR_ACCUMULATE   ((uint64_t)1 << 30)
#define LOCAL_ADDR_FULL_ACC_ROW ((uint64_t)1 << 29)

/* LOOP_WS_CONFIG_BOUNDS: 16-bit I/J/K iterators (rs2) and pads (rs1) at [15:0]/[31:16]/[47:32]. */
#define BOUNDS_J_SHIFT 16
#define BOUNDS_K_SHIFT 32

/* CONFIG sub-commands (rs1[1:0]). */
#define CONFIG_EX 0u
#define CONFIG_LD 1u
#define CONFIG_ST 2u

/* CONFIG_LD rs1: [63:32] scale, [31:16] block_mvin_stride, [15:8] pixel_repeats, [4:3] state id. */
#define CONFIG_LD_BLOCK_STRIDE_SHIFT  16
#define CONFIG_LD_PIXEL_REPEATS_SHIFT 8
#define CONFIG_LD_STATE_SHIFT         3

/* CONFIG_EX rs1: [31:16] a_stride, [2] dataflow (1 = WS); rs2: [63:48] c_stride. */
#define CONFIG_EX_A_STRIDE_SHIFT 16
#define CONFIG_EX_WS             (1u << 2)
#define CONFIG_EX_C_STRIDE_SHIFT 48

/* LoopMatmul.castDramOffset keeps 32 bits of every tile-origin DRAM offset. */
#define DRAM_OFFSET_LIMIT ((uint64_t)1 << 32)

/* LoopMatmul runs two concurrent loops; each owns half of SPAD and ACC. */
#define CONCURRENT_LOOPS 2u

/* Validated scope of the hardware execution path. */
#define VALIDATED_K_TILES  2u

/* Set while a paired session owns the accelerator (potal_paired_session_open);
 * stays set after a failed session closed with outstanding work. */
static int paired_owner;

const char *potal_status_name(potal_status status) {
    switch (status) {
        case POTAL_OK:                   return "OK";
        case POTAL_HW_INVALID_COLUMN:    return "HW_INVALID_COLUMN";
        case POTAL_HW_INVALID_SCALE:     return "HW_INVALID_SCALE";
        case POTAL_HW_SCALE_CONFLICT:    return "HW_SCALE_CONFLICT";
        case POTAL_HW_RELEASE_MISS:      return "HW_RELEASE_MISS";
        case POTAL_HW_INVALID_ADDRESS:   return "HW_INVALID_ADDRESS";
        case POTAL_HW_INVALID_ENCODING:  return "HW_INVALID_ENCODING";
        case POTAL_HW_BAD_COMMAND:       return "HW_BAD_COMMAND";
        case POTAL_ERR_INVALID_ARGUMENT: return "ERR_INVALID_ARGUMENT";
        case POTAL_ERR_UNSUPPORTED:      return "ERR_UNSUPPORTED";
        case POTAL_ERR_CAPACITY:         return "ERR_CAPACITY";
        case POTAL_ERR_NO_HARDWARE:      return "ERR_NO_HARDWARE";
        case POTAL_ERR_BUSY:             return "ERR_BUSY";
        case POTAL_ERR_BAD_RESPONSE:     return "ERR_BAD_RESPONSE";
    }
    return "UNKNOWN";
}

uint32_t potal_hw_dim(void) {
    return POTAL_HW_DIM;
}

const char *potal_hw_params_sha256(void) {
    return POTAL_HW_PARAMS_SHA256;
}

int potal_carrier_is_valid(uint32_t carrier) {
    return carrier <= POTAL_CARRIER_MAX_SHIFT || carrier == POTAL_CARRIER_ZERO;
}

potal_status potal_scale_key_encode(uint32_t slot, uint32_t generation,
                                    uint32_t column, uint64_t *rs1) {
    if (rs1 == NULL || slot >= POTAL_SCALE_ENTRIES ||
        generation > POTAL_GENERATION_MAX || column >= POTAL_HW_DIM) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    *rs1 = (uint64_t)slot |
           ((uint64_t)generation << SCALE_GENERATION_SHIFT) |
           ((uint64_t)column << SCALE_COLUMN_SHIFT);
    return POTAL_OK;
}

/* Bytes spanned by `rows` rows of `cols` elements with row stride `ld`; 0 on overflow. */
static uint64_t matrix_extent(uint64_t rows, uint64_t cols, uint64_t ld, uint64_t elem_size) {
    const uint64_t max = UINT64_MAX / elem_size;
    if (ld != 0 && rows - 1 > (max - cols) / ld) {
        return 0;
    }
    return ((rows - 1) * ld + cols) * elem_size;
}

static int ranges_overlap(uintptr_t a, uint64_t a_len, uintptr_t b, uint64_t b_len) {
    return (uint64_t)a < (uint64_t)b + b_len && (uint64_t)b < (uint64_t)a + a_len;
}

static int pointer_ok(const void *p, uint64_t extent) {
    const uintptr_t addr = (uintptr_t)p;
    /* DRAM address 0 tells LoopMatmul to skip the load; reject it explicitly. */
    return p != NULL && addr % POTAL_HW_MAX_BYTES == 0 &&
           extent != 0 && extent <= DRAM_OFFSET_LIMIT &&
           (uint64_t)addr <= UINT64_MAX - extent;
}

potal_status potal_gemm_ws_check(const potal_gemm_ws *g) {
    if (g == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if (g->m == 0 || g->n == 0 || g->k == 0 ||
        g->lda < g->k || g->ldb < g->n || g->ldc < g->n ||
        g->scale_generation > POTAL_GENERATION_MAX ||
        g->passes_per_block == 0 || g->passes_per_block > LOOP_WS_FIELD16_MAX) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }

    const uint64_t dim = POTAL_HW_DIM;
    const uint64_t i_tiles = (g->m + dim - 1) / dim;
    const uint64_t j_tiles = (g->n + dim - 1) / dim;
    const uint64_t k_tiles = (g->k + dim - 1) / dim;
    if (i_tiles > LOOP_WS_FIELD16_MAX || j_tiles > LOOP_WS_FIELD16_MAX ||
        k_tiles > LOOP_WS_FIELD16_MAX || k_tiles % g->passes_per_block != 0) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }

    const uint64_t a_bytes = matrix_extent(g->m, g->k, g->lda, sizeof(int8_t));
    const uint64_t b_bytes = matrix_extent(g->k, g->n, g->ldb, sizeof(int8_t));
    const uint64_t c_bytes = matrix_extent(g->m, g->n, g->ldc, sizeof(int32_t));
    if (!pointer_ok(g->a, a_bytes) || !pointer_ok(g->b, b_bytes) || !pointer_ok(g->c, c_bytes) ||
        /* CONFIG_ST carries the C row stride in bytes in rs2[31:0]. */
        g->ldc > UINT32_MAX / sizeof(int32_t) ||
        ranges_overlap((uintptr_t)g->c, c_bytes, (uintptr_t)g->a, a_bytes) ||
        ranges_overlap((uintptr_t)g->c, c_bytes, (uintptr_t)g->b, b_bytes)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }

    /* Scale window: slot = base + block * J + j must stay below ScaleEntries. */
    const uint64_t scale_blocks = k_tiles / g->passes_per_block;
    if (g->scale_base >= POTAL_SCALE_ENTRIES ||
        scale_blocks * j_tiles > POTAL_SCALE_ENTRIES - g->scale_base) {
        return POTAL_ERR_CAPACITY;
    }
    /* Work IDs, then the per-loop SPAD (A + B tiles) and ACC (C tiles) halves. */
    const uint64_t spad_rows = (uint64_t)POTAL_HW_BANK_NUM * POTAL_HW_BANK_ROWS / CONCURRENT_LOOPS;
    const uint64_t acc_rows = (uint64_t)POTAL_HW_ACC_ROWS / CONCURRENT_LOOPS;
    if (i_tiles * j_tiles > POTAL_WORK_IDS_PER_LOOP ||
        (i_tiles * k_tiles + k_tiles * j_tiles) * dim > spad_rows ||
        i_tiles * j_tiles * dim > acc_rows) {
        return POTAL_ERR_CAPACITY;
    }

    if (g->flags != 0 || (POTAL_HW_DIM != 16 && POTAL_HW_DIM != 32) ||
        g->m != dim || g->n != dim || g->k % dim != 0 || k_tiles > VALIDATED_K_TILES) {
        return POTAL_ERR_UNSUPPORTED;
    }
    return POTAL_OK;
}

potal_status potal_gemm_ws_scale_slot(const potal_gemm_ws *g, uint32_t k_block,
                                      uint32_t n_tile, uint32_t *slot) {
    const potal_status status = potal_gemm_ws_check(g);
    if (status != POTAL_OK) {
        return status;
    }
    const uint32_t j_tiles = (g->n + POTAL_HW_DIM - 1) / POTAL_HW_DIM;
    const uint32_t k_tiles = (g->k + POTAL_HW_DIM - 1) / POTAL_HW_DIM;
    if (slot == NULL || k_block >= k_tiles / g->passes_per_block || n_tile >= j_tiles) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    *slot = g->scale_base + k_block * j_tiles + n_tile;
    return POTAL_OK;
}

static potal_rocc_cmd cmd(uint64_t rs1, uint64_t rs2) {
    potal_rocc_cmd c = { rs1, rs2 };
    return c;
}

static uint64_t config_ld_rs1(uint64_t state_id) {
    /* Scale 0 (MVIN_SCALE_IDENTITY), block stride DIM, one pixel repeat. */
    return ((uint64_t)POTAL_HW_DIM << CONFIG_LD_BLOCK_STRIDE_SHIFT) |
           ((uint64_t)1 << CONFIG_LD_PIXEL_REPEATS_SHIFT) |
           (state_id << CONFIG_LD_STATE_SHIFT) | CONFIG_LD;
}

/*
 * The single CONFIG + LOOP_WS encoder of both request types. The caller has
 * validated every field; potal_gemm_ws passes zero padding, no accumulation
 * and a non-NULL C.
 */
static void encode_program(const potal_loop_ws *l, potal_gemm_ws_program *p) {
    /* 1. Gemmini execution config: mvin strides in bytes, WS, INT32 mvout. */
    p->config_ld_a = cmd(config_ld_rs1(0), l->lda * sizeof(int8_t));
    p->config_ld_b = cmd(config_ld_rs1(1), l->ldb * sizeof(int8_t));
    p->config_ex   = cmd(((uint64_t)1 << CONFIG_EX_A_STRIDE_SHIFT) | CONFIG_EX_WS | CONFIG_EX,
                         (uint64_t)1 << CONFIG_EX_C_STRIDE_SHIFT);
    p->config_st   = cmd(CONFIG_ST, l->ldc * sizeof(int32_t));

    /* 2. LOOP_WS bounds and pads, addresses (no D; C 0 = no store), strides in elements. */
    p->bounds     = cmd(((uint64_t)l->pad_k << BOUNDS_K_SHIFT) | ((uint64_t)l->pad_j << BOUNDS_J_SHIFT) | l->pad_i,
                        ((uint64_t)l->k_tiles << BOUNDS_K_SHIFT) | ((uint64_t)l->j_tiles << BOUNDS_J_SHIFT) |
                            l->i_tiles);
    p->addrs_ab   = cmd((uint64_t)(uintptr_t)l->a, (uint64_t)(uintptr_t)l->b);
    p->addrs_dc   = cmd(0, (uint64_t)(uintptr_t)l->c);
    p->strides_ab = cmd(l->lda, l->ldb);
    p->strides_dc = cmd(0, l->ldc);

    /* 3. PoTal scale metadata on LOOP_WS; full_C, optional ex_accumulate, no act/transpose. */
    p->loop_ws = cmd(((l->flags & POTAL_GEMM_ACCUMULATE) ? 1u : 0u) |
                     ((uint64_t)1 << LOOP_WS_FULL_C_SHIFT) |
                     ((uint64_t)1 << LOOP_WS_POTAL_ENABLE_SHIFT) |
                     ((uint64_t)l->scale_base << LOOP_WS_SCALE_BASE_SHIFT) |
                     ((uint64_t)l->scale_generation << LOOP_WS_GENERATION_SHIFT) |
                     ((uint64_t)l->passes_per_block << LOOP_WS_PASSES_SHIFT),
                     0);
}

/* A validated potal_gemm_ws as the equivalent single LOOP_WS invocation. */
static potal_loop_ws gemm_as_loop(const potal_gemm_ws *g) {
    potal_loop_ws l = {
        g->a, g->b, g->c, g->lda, g->ldb, g->ldc,
        g->m / POTAL_HW_DIM, g->n / POTAL_HW_DIM, g->k / POTAL_HW_DIM,
        0, 0, 0,
        g->scale_base, g->scale_generation, g->passes_per_block, 0,
    };
    return l;
}

potal_status potal_gemm_ws_encode(const potal_gemm_ws *g, potal_gemm_ws_program *p) {
    if (p == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const potal_status status = potal_gemm_ws_check(g);
    if (status != POTAL_OK) {
        return status;
    }
    const potal_loop_ws l = gemm_as_loop(g);
    encode_program(&l, p);
    return POTAL_OK;
}

/* Tile-origin pointer: non-NULL, `align`-byte aligned, extent within the 32-bit DRAM offset. */
static int tile_pointer_ok(const void *p, uint64_t extent, uint64_t align) {
    const uintptr_t addr = (uintptr_t)p;
    return p != NULL && addr % align == 0 &&
           extent != 0 && extent <= DRAM_OFFSET_LIMIT &&
           (uint64_t)addr <= UINT64_MAX - extent;
}

potal_status potal_loop_ws_check(const potal_loop_ws *l) {
    if (l == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const uint64_t dim = POTAL_HW_DIM;
    if (l->i_tiles == 0 || l->j_tiles == 0 || l->k_tiles == 0 ||
        l->i_tiles > LOOP_WS_FIELD16_MAX || l->j_tiles > LOOP_WS_FIELD16_MAX ||
        l->k_tiles > LOOP_WS_FIELD16_MAX ||
        /* LoopMatmul asserts every pad < block_size with PoTal enabled. */
        l->pad_i >= dim || l->pad_j >= dim || l->pad_k >= dim ||
        l->scale_generation > POTAL_GENERATION_MAX ||
        l->passes_per_block == 0 || l->passes_per_block > LOOP_WS_FIELD16_MAX ||
        /* The block counter restarts with every LOOP_WS: keep scale blocks whole. */
        l->k_tiles % l->passes_per_block != 0) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }

    const uint64_t m = (uint64_t)l->i_tiles * dim - l->pad_i;
    const uint64_t n = (uint64_t)l->j_tiles * dim - l->pad_j;
    const uint64_t k = (uint64_t)l->k_tiles * dim - l->pad_k;
    if (l->lda < k || l->ldb < n || (l->c != NULL && l->ldc < n) ||
        /* CONFIG_ST carries the C row stride in bytes in rs2[31:0]. */
        l->ldc > UINT32_MAX / sizeof(int32_t)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const uint64_t a_bytes = matrix_extent(m, k, l->lda, sizeof(int8_t));
    const uint64_t b_bytes = matrix_extent(k, n, l->ldb, sizeof(int8_t));
    if (!tile_pointer_ok(l->a, a_bytes, dim * sizeof(int8_t)) ||
        !tile_pointer_ok(l->b, b_bytes, dim * sizeof(int8_t))) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if (l->c != NULL) {
        const uint64_t c_bytes = matrix_extent(m, n, l->ldc, sizeof(int32_t));
        if (!tile_pointer_ok(l->c, c_bytes, dim * sizeof(int32_t)) ||
            ranges_overlap((uintptr_t)l->c, c_bytes, (uintptr_t)l->a, a_bytes) ||
            ranges_overlap((uintptr_t)l->c, c_bytes, (uintptr_t)l->b, b_bytes)) {
            return POTAL_ERR_INVALID_ARGUMENT;
        }
    }

    /* Same capacity rules as potal_gemm_ws_check(), on this invocation's tiles. */
    const uint64_t scale_blocks = l->k_tiles / l->passes_per_block;
    if (l->scale_base >= POTAL_SCALE_ENTRIES ||
        scale_blocks * l->j_tiles > POTAL_SCALE_ENTRIES - l->scale_base) {
        return POTAL_ERR_CAPACITY;
    }
    const uint64_t spad_rows = (uint64_t)POTAL_HW_BANK_NUM * POTAL_HW_BANK_ROWS / CONCURRENT_LOOPS;
    const uint64_t acc_rows  = (uint64_t)POTAL_HW_ACC_ROWS / CONCURRENT_LOOPS;
    const uint64_t i_tiles = l->i_tiles, j_tiles = l->j_tiles, k_tiles = l->k_tiles;
    if (i_tiles * j_tiles > POTAL_WORK_IDS_PER_LOOP ||
        (i_tiles * k_tiles + k_tiles * j_tiles) * dim > spad_rows ||
        i_tiles * j_tiles * dim > acc_rows) {
        return POTAL_ERR_CAPACITY;
    }

    if ((l->flags & ~POTAL_GEMM_ACCUMULATE) != 0) {
        return POTAL_ERR_UNSUPPORTED;
    }
    return POTAL_OK;
}

potal_status potal_loop_ws_encode(const potal_loop_ws *l, potal_gemm_ws_program *p) {
    if (p == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const potal_status status = potal_loop_ws_check(l);
    if (status != POTAL_OK) {
        return status;
    }
    encode_program(l, p);
    return POTAL_OK;
}

/* ---- H2 paired / indexed execution: pure checks and encoders -------------- */

#define SP_ROWS_TOTAL ((uint64_t)POTAL_HW_BANK_NUM * POTAL_HW_BANK_ROWS)

/* Weight bank of a pair: sp rows [(2p+1) * BANK_ROWS, (2p+2) * BANK_ROWS). */
static uint64_t pair_weight_begin(uint32_t pair) {
    return ((uint64_t)2 * pair + 1) * POTAL_HW_BANK_ROWS;
}

potal_status potal_index_config_check(const potal_index_config *c) {
    if (c == NULL || c->pair >= POTAL_PAIRS || c->j_tiles == 0 || c->source_k_rows == 0 ||
        c->active_rows == 0 || c->j_tiles > LOOP_WS_FIELD16_MAX) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if (c->active_rows > POTAL_HW_INDEX_ENTRIES) {
        return POTAL_ERR_CAPACITY;
    }
    /* PoTalScaleCommandIngress cfgAddressValid: the region lies inside the pair's weight bank. */
    const uint64_t dim = POTAL_HW_DIM;
    const uint64_t k_tiles = ((uint64_t)c->source_k_rows + dim - 1) / dim;
    const uint64_t region = k_tiles * c->j_tiles * dim;
    const uint64_t begin = pair_weight_begin(c->pair);
    const uint64_t end = begin + POTAL_HW_BANK_ROWS;
    if (c->b_sp_base < begin || c->b_sp_base >= end || region > end - c->b_sp_base) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    return POTAL_OK;
}

potal_status potal_index_config_encode(const potal_index_config *c, potal_rocc_cmd *out) {
    const potal_status status = potal_index_config_check(c);
    if (status != POTAL_OK) {
        return status;
    }
    if (out == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    *out = cmd(((uint64_t)c->j_tiles << INDEX_HIGH_SHIFT) | c->b_sp_base,
               ((uint64_t)c->active_rows << INDEX_HIGH_SHIFT) | c->source_k_rows);
    return POTAL_OK;
}

potal_status potal_index_write_check(const potal_index_config *c, const potal_index_write *w) {
    const potal_status status = potal_index_config_check(c);
    if (status != POTAL_OK) {
        return status;
    }
    if (w == NULL || w->slot >= c->active_rows || w->resident_k >= c->source_k_rows) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    return POTAL_OK;
}

potal_status potal_index_write_encode(const potal_index_config *c, const potal_index_write *w,
                                      potal_rocc_cmd *out) {
    const potal_status status = potal_index_write_check(c, w);
    if (status != POTAL_OK) {
        return status;
    }
    if (out == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    *out = cmd(w->slot, w->resident_k);
    return POTAL_OK;
}

potal_status potal_paired_loop_ws_check(const potal_paired_loop_ws *pl) {
    if (pl == NULL || pl->pair >= POTAL_PAIRS ||
        (pl->path != POTAL_PAIRED_MAIN && pl->path != POTAL_PAIRED_RESIDUAL) ||
        pl->resident_b > 1 || pl->indexed > 1 || pl->a_offset_tiles > LOOP_WS_A_OFFSET_MAX) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const potal_loop_ws *l = &pl->loop;
    const uint64_t dim = POTAL_HW_DIM;
    if (l->i_tiles == 0 || l->j_tiles == 0 || l->k_tiles == 0 ||
        l->i_tiles > LOOP_WS_FIELD16_MAX || l->j_tiles > LOOP_WS_FIELD16_MAX ||
        l->k_tiles > LOOP_WS_FIELD16_MAX ||
        l->pad_i >= dim || l->pad_j >= dim || l->pad_k >= dim ||
        l->scale_generation > POTAL_GENERATION_MAX ||
        l->passes_per_block == 0 || l->passes_per_block > LOOP_WS_FIELD16_MAX ||
        l->k_tiles % l->passes_per_block != 0) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const uint64_t m = (uint64_t)l->i_tiles * dim - l->pad_i;
    const uint64_t n = (uint64_t)l->j_tiles * dim - l->pad_j;
    const uint64_t k = (uint64_t)l->k_tiles * dim - l->pad_k;
    if (l->lda < k) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const uint64_t a_bytes = matrix_extent(m, k, l->lda, sizeof(int8_t));
    if (!tile_pointer_ok(l->a, a_bytes, dim * sizeof(int8_t))) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    /* The path decides B: the original panel from DRAM (Main) or the resident one (Residual). */
    if (pl->path == POTAL_PAIRED_MAIN) {
        if (pl->resident_b != 0 || pl->indexed != 0 || pl->a_offset_tiles != 0 || l->ldb < n) {
            return POTAL_ERR_INVALID_ARGUMENT;
        }
        const uint64_t b_bytes = matrix_extent(k, n, l->ldb, sizeof(int8_t));
        if (!tile_pointer_ok(l->b, b_bytes, dim * sizeof(int8_t)) ||
            ranges_overlap((uintptr_t)l->a, a_bytes, (uintptr_t)l->b, b_bytes)) {
            return POTAL_ERR_INVALID_ARGUMENT;
        }
    } else {
        /* LoopMatmul: indexed weight requires the paired residual mode and B DRAM address 0. */
        if (pl->resident_b != 1 || pl->indexed != 1 || l->b != NULL || l->ldb != 0) {
            return POTAL_ERR_INVALID_ARGUMENT;
        }
    }
    /* Results stay in the ACC bank until potal_acc_readback; the LOOP_WS C store is not used. */
    if (l->c != NULL || l->ldc != 0) {
        return POTAL_ERR_UNSUPPORTED;
    }
    if ((l->flags & ~POTAL_GEMM_ACCUMULATE) != 0) {
        return POTAL_ERR_UNSUPPORTED;
    }

    /* Scale window, work IDs, one activation bank (A behind its offset), one weight bank, one ACC bank. */
    const uint64_t scale_blocks = l->k_tiles / l->passes_per_block;
    if (l->scale_base >= POTAL_SCALE_ENTRIES ||
        scale_blocks * l->j_tiles > POTAL_SCALE_ENTRIES - l->scale_base) {
        return POTAL_ERR_CAPACITY;
    }
    const uint64_t a_tiles = (uint64_t)l->i_tiles * l->k_tiles;
    const uint64_t bank_tiles = (uint64_t)POTAL_HW_BANK_ROWS / dim;
    if ((uint64_t)l->i_tiles * l->j_tiles > POTAL_WORK_IDS_PER_LOOP ||
        a_tiles > bank_tiles - (pl->a_offset_tiles < bank_tiles ? pl->a_offset_tiles : bank_tiles) ||
        pl->a_offset_tiles >= bank_tiles ||
        (pl->path == POTAL_PAIRED_MAIN && (uint64_t)l->k_tiles * l->j_tiles > bank_tiles) ||
        (uint64_t)l->i_tiles * l->j_tiles * dim > (uint64_t)POTAL_HW_ACC_ROWS / 2) {
        return POTAL_ERR_CAPACITY;
    }
    return POTAL_OK;
}

potal_status potal_paired_loop_ws_encode(const potal_paired_loop_ws *pl, potal_gemm_ws_program *p) {
    if (p == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const potal_status status = potal_paired_loop_ws_check(pl);
    if (status != POTAL_OK) {
        return status;
    }
    encode_program(&pl->loop, p);
    const uint64_t spad = (uint64_t)pl->pair + 1;
    p->loop_ws.rs1 |= (spad << LOOP_WS_A_SPAD_SHIFT) | (spad << LOOP_WS_B_SPAD_SHIFT) |
                      ((uint64_t)1 << LOOP_WS_PAIRED_SHIFT) |
                      ((uint64_t)(pl->path == POTAL_PAIRED_RESIDUAL) << LOOP_WS_RESIDUAL_SHIFT) |
                      ((uint64_t)pl->indexed << LOOP_WS_INDEXED_SHIFT);
    p->loop_ws.rs2 = (uint64_t)pl->a_offset_tiles << LOOP_WS_A_OFFSET_SHIFT;
    /* Resident B: DRAM address 0 skips LdB (encode_program wrote the NULL pointer as 0 already). */
    return POTAL_OK;
}

potal_status potal_acc_readback_check(const potal_acc_readback *r) {
    if (r == NULL || r->acc_bank > 1 || r->i_tiles == 0 || r->j_tiles == 0 ||
        r->i_tiles > LOOP_WS_FIELD16_MAX || r->j_tiles > LOOP_WS_FIELD16_MAX ||
        r->pad_i >= POTAL_HW_DIM || r->pad_j >= POTAL_HW_DIM) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const uint64_t dim = POTAL_HW_DIM;
    const uint64_t rows = (uint64_t)r->i_tiles * dim - r->pad_i;
    const uint64_t cols = (uint64_t)r->j_tiles * dim - r->pad_j;
    if (r->ldc < cols || r->ldc > UINT32_MAX / sizeof(int32_t)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const uint64_t c_bytes = matrix_extent(rows, cols, r->ldc, sizeof(int32_t));
    if (!tile_pointer_ok(r->c, c_bytes, dim * sizeof(int32_t))) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if ((uint64_t)r->i_tiles * r->j_tiles * dim > (uint64_t)POTAL_HW_ACC_ROWS / 2) {
        return POTAL_ERR_CAPACITY;
    }
    return POTAL_OK;
}

static uint64_t acc_local_addr(uint32_t bank, uint64_t row) {
    return LOCAL_ADDR_ACC | LOCAL_ADDR_FULL_ACC_ROW | ((uint64_t)bank * (POTAL_HW_ACC_ROWS / 2) + row);
}

potal_status potal_acc_readback_encode(const potal_acc_readback *r, potal_rocc_cmd *config_st,
                                       potal_rocc_cmd *mvouts, uint32_t capacity, uint32_t *count) {
    const potal_status status = potal_acc_readback_check(r);
    if (status != POTAL_OK) {
        return status;
    }
    if (config_st == NULL || mvouts == NULL || count == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const uint64_t tiles = (uint64_t)r->i_tiles * r->j_tiles;
    *count = (uint32_t)tiles;
    if (tiles > capacity) {
        return POTAL_ERR_CAPACITY;
    }
    const uint64_t dim = POTAL_HW_DIM;
    *config_st = cmd(CONFIG_ST, r->ldc * sizeof(int32_t));
    uint32_t t = 0;
    for (uint32_t i = 0; i < r->i_tiles; ++i) {
        const uint64_t rows = dim - (i + 1 == r->i_tiles ? r->pad_i : 0);
        for (uint32_t j = 0; j < r->j_tiles; ++j, ++t) {
            const uint64_t cols = dim - (j + 1 == r->j_tiles ? r->pad_j : 0);
            const uint64_t dram = (uint64_t)(uintptr_t)r->c + ((uint64_t)i * dim * r->ldc + (uint64_t)j * dim) * sizeof(int32_t);
            const uint64_t acc_row = ((uint64_t)i * r->j_tiles + j) * dim;
            mvouts[t] = cmd(dram, (rows << MVOUT_ROWS_SHIFT) | (cols << MVOUT_COLS_SHIFT) |
                                      acc_local_addr(r->acc_bank, acc_row));
        }
    }
    return POTAL_OK;
}

potal_status potal_acc_bias_check(const potal_acc_bias *b) {
    if (b == NULL || b->acc_bank > 1 || b->i_tiles == 0 || b->j_tiles == 0 ||
        b->i_tiles > LOOP_WS_FIELD16_MAX || b->j_tiles > LOOP_WS_FIELD16_MAX ||
        b->pad_i >= POTAL_HW_DIM || b->pad_j >= POTAL_HW_DIM) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const uint64_t columns = (uint64_t)b->j_tiles * POTAL_HW_DIM - b->pad_j;
    if (!tile_pointer_ok(b->values, columns * sizeof(int32_t), POTAL_HW_DIM * sizeof(int32_t))) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const uint64_t tiles = (uint64_t)b->i_tiles * b->j_tiles;
    if (tiles > POTAL_WORK_IDS_PER_LOOP || tiles * POTAL_HW_DIM > POTAL_HW_ACC_ROWS / 2) {
        return POTAL_ERR_CAPACITY;
    }
    return POTAL_OK;
}

potal_status potal_acc_bias_encode(const potal_acc_bias *b, potal_rocc_cmd *config_ld,
                                   potal_rocc_cmd *mvins, uint32_t capacity, uint32_t *count) {
    const potal_status status = potal_acc_bias_check(b);
    if (status != POTAL_OK) {
        return status;
    }
    if (config_ld == NULL || mvins == NULL || count == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    *count = b->i_tiles * b->j_tiles;
    if (*count > capacity) {
        return POTAL_ERR_CAPACITY;
    }
    /* State 0, shrink=0, identity scale; stride 0 repeats the bias row. */
    *config_ld = cmd(config_ld_rs1(0), 0);
    uint32_t t = 0;
    for (uint32_t i = 0; i < b->i_tiles; ++i) {
        const uint64_t rows = POTAL_HW_DIM - (i + 1 == b->i_tiles ? b->pad_i : 0);
        for (uint32_t j = 0; j < b->j_tiles; ++j, ++t) {
            const uint64_t columns = POTAL_HW_DIM - (j + 1 == b->j_tiles ? b->pad_j : 0);
            const uint64_t row = (uint64_t)b->acc_bank * (POTAL_HW_ACC_ROWS / 2) +
                                 ((uint64_t)i * b->j_tiles + j) * POTAL_HW_DIM;
            const uint64_t dram = (uint64_t)(uintptr_t)b->values + (uint64_t)j * POTAL_HW_DIM * sizeof(int32_t);
            mvins[t] = cmd(dram, (rows << MVOUT_ROWS_SHIFT) | (columns << MVOUT_COLS_SHIFT) |
                                 LOCAL_ADDR_ACC | LOCAL_ADDR_ACCUMULATE | row);
        }
    }
    return POTAL_OK;
}

/* ---- Resident I1/P1: pure transport and physical admission ---------------- */

static uint32_t resident_ws_tiles(uint32_t n) {
    return n / 16u + (n % 16u != 0);
}

static uint32_t resident_ws_b_base(const potal_resident_ws_load *d) {
    return (d->pair + 1) * 8192u -
        resident_ws_tiles(d->k_rows) * resident_ws_tiles(d->j_columns) * 16u;
}

static int resident_ws_geometry(void) {
    return POTAL_HW_DIM == 16 && POTAL_HW_BANK_NUM == 4 &&
        POTAL_HW_BANK_ROWS == 4096 && POTAL_HW_ACC_ROWS == 1024 &&
        POTAL_HW_INDEX_ENTRIES == 32 && POTAL_SCALE_ENTRIES == 256;
}

potal_status potal_resident_ws_shift_carrier(int64_t shift, uint32_t *carrier) {
    if (!carrier || shift < 0 || shift > POTAL_CARRIER_MAX_SHIFT)
        return POTAL_ERR_INVALID_ARGUMENT;
    *carrier = (uint32_t)shift;
    return POTAL_OK;
}

potal_status potal_resident_ws_load_check(const potal_resident_ws_load *d) {
    if (!d || d->pair >= 2 || !d->k_rows || !d->j_columns ||
        !d->weight_id || !d->weight_generation ||
        d->k_begin % 16 || d->j_begin % 16 ||
        d->k_begin > UINT64_MAX - d->k_rows || d->j_begin > UINT64_MAX - d->j_columns ||
        d->source_stride < d->j_columns)
        return POTAL_ERR_INVALID_ARGUMENT;
    if (!resident_ws_geometry()) return POTAL_ERR_UNSUPPORTED;
    if (d->k_rows > 28u * 16u || d->j_columns > 9u * 16u ||
        d->main_a_tiles >= 256 ||
        (uint64_t)resident_ws_tiles(d->k_rows) * resident_ws_tiles(d->j_columns) * 16 > 4096)
        return POTAL_ERR_CAPACITY;
    const uint64_t span = matrix_extent(d->k_rows, d->j_columns, d->source_stride, 1);
    if (!span || span > POTAL_RESIDENT_WS_POOL_BYTES) return POTAL_ERR_INVALID_ARGUMENT;
    return POTAL_OK;
}

potal_status potal_resident_ws_view_encode(const potal_resident_ws_view *v, potal_rocc_cmd *out) {
    if (!v || !out || v->enable > 1 || v->pair > 1 || v->indexed > 1 || v->acc_bank > 1)
        return POTAL_ERR_INVALID_ARGUMENT;
    if (!resident_ws_geometry()) return POTAL_ERR_UNSUPPORTED;
    if (!v->enable) {
        if (v->pair || v->indexed || v->acc_bank || v->mode || v->b_base || v->b_pitch ||
            v->indexed_j_origin || v->acc_base) return POTAL_ERR_INVALID_ARGUMENT;
        *out = cmd(0, 0);
        return POTAL_OK;
    }
    const uint32_t begin = (2 * v->pair + 1) * 4096u;
    if (v->mode != 1 || v->b_base < begin || v->b_base >= begin + 4096 ||
        v->b_base % 16 || !v->b_pitch || v->b_pitch % 16 || v->b_pitch > 144 ||
        v->acc_base % 16 || v->acc_base > 496 ||
        (!v->indexed && v->indexed_j_origin) ||
        v->indexed_j_origin >= v->b_pitch / 16)
        return POTAL_ERR_INVALID_ARGUMENT;
    *out = cmd((uint64_t)v->b_base | ((uint64_t)v->b_pitch << 16) |
                   ((uint64_t)v->indexed_j_origin << 32) | ((uint64_t)v->acc_base << 48),
               1u | ((uint64_t)v->pair << 1) | ((uint64_t)v->indexed << 2) |
                   ((uint64_t)v->acc_bank << 3) | UINT64_C(0x10));
    return POTAL_OK;
}

potal_status potal_resident_ws_run_check(const potal_resident_ws_load *l,
    const potal_resident_ws_run *d, const int8_t *a, size_t bytes) {
    potal_status status = potal_resident_ws_load_check(l);
    if (status != POTAL_OK) return status;
    if (!d || !a || d->path > 1 || d->indexed > 1 || d->first > 1 || d->last > 1 ||
        !d->output_id || !d->i_rows || d->i_rows > 16 || !d->j_columns || !d->k_rows ||
        d->j_offset % 16 || d->scale_generation > 255 ||
        d->a_stride < d->k_rows || d->a_offset > bytes ||
        (uintptr_t)a > UINTPTR_MAX - bytes ||
        ((uintptr_t)a + d->a_offset) % 16 ||
        (d->path && !d->indexed) || (d->first && d->sequence))
        return POTAL_ERR_INVALID_ARGUMENT;
    if (d->k_rows > 448 || d->j_columns > 144) return POTAL_ERR_CAPACITY;
    if (d->j_columns > l->j_columns || d->j_offset > l->j_columns - d->j_columns ||
        d->original_i > UINT64_MAX - d->i_rows) return POTAL_ERR_INVALID_ARGUMENT;
    const uint32_t nk = resident_ws_tiles(d->k_rows), nj = resident_ws_tiles(d->j_columns);
    if (d->acc_tile >= 32 || nj > 32 - d->acc_tile ||
        d->a_offset_tiles >= 256 || nk > 256 - d->a_offset_tiles ||
        d->scale_base >= 256 || nk * nj > 256 - d->scale_base)
        return POTAL_ERR_CAPACITY;
    if ((!d->path && (d->a_offset_tiles || nk > l->main_a_tiles)) ||
        (d->path && d->a_offset_tiles < l->main_a_tiles) ||
        d->sequence > UINT64_MAX - nk)
        return POTAL_ERR_INVALID_ARGUMENT;
    const uint64_t extent = matrix_extent(d->i_rows, d->k_rows, d->a_stride, 1);
    if (!extent || extent > bytes - d->a_offset || extent > DRAM_OFFSET_LIMIT)
        return POTAL_ERR_INVALID_ARGUMENT;
    if (!d->indexed) {
        if (d->index.owner || d->index.generation || d->k_offset % 16 ||
            d->k_rows > l->k_rows || d->k_offset > l->k_rows - d->k_rows ||
            (d->k_rows % 16 && d->k_offset + d->k_rows != l->k_rows))
            return POTAL_ERR_INVALID_ARGUMENT;
    } else {
        if (d->k_rows > 32) return POTAL_ERR_CAPACITY;
        if (d->k_offset >= l->k_rows) return POTAL_ERR_INVALID_ARGUMENT;
        for (uint32_t p = 0; p < nk; ++p) {
            if (!d->fragment_counts[p] || d->fragment_counts[p] > 16 ||
                (p + 1 == nk && d->fragment_counts[p] != d->k_rows - p * 16) ||
                (!d->path && d->fragment_counts[p] != (p + 1 == nk ? d->k_rows - p * 16 : 16)))
                return POTAL_ERR_INVALID_ARGUMENT;
        }
    }
    if (!d->path && (d->k_offset % 16 || d->k_rows > l->k_rows ||
        d->k_offset > l->k_rows - d->k_rows ||
        (d->k_rows % 16 && d->k_offset + d->k_rows != l->k_rows)))
        return POTAL_ERR_INVALID_ARGUMENT;
    return POTAL_OK;
}

potal_status potal_resident_ws_run_encode(const potal_resident_ws_load *l,
    const potal_resident_ws_run *d, const int8_t *a, size_t bytes,
    potal_resident_ws_program *out) {
    if (!out) return POTAL_ERR_INVALID_ARGUMENT;
    potal_status status = potal_resident_ws_run_check(l, d, a, bytes);
    if (status != POTAL_OK) return status;
    const uint32_t nk = resident_ws_tiles(d->k_rows), nj = resident_ws_tiles(d->j_columns);
    const potal_loop_ws loop = {
        a + d->a_offset, NULL, NULL, d->a_stride, 0, 0, 1, nj, nk,
        16 - d->i_rows, nj * 16 - d->j_columns, nk * 16 - d->k_rows,
        d->scale_base, d->scale_generation, 1, d->first ? 0 : POTAL_GEMM_ACCUMULATE
    };
    const uint32_t pitch = resident_ws_tiles(l->j_columns) * 16;
    const potal_resident_ws_view view = {
        1, l->pair, d->indexed, d->path, 1,
        resident_ws_b_base(l) + (d->indexed ? 0 : (d->k_offset / 16) * pitch + d->j_offset),
        pitch, d->indexed ? d->j_offset / 16 : 0, d->acc_tile * 16
    };
    potal_resident_ws_program p;
    status = potal_resident_ws_view_encode(&view, &p.view);
    if (status != POTAL_OK) return status;
    encode_program(&loop, &p.loop);
    const uint64_t spad = l->pair + 1;
    p.loop.loop_ws.rs1 |= UINT64_C(8) | (spad << LOOP_WS_A_SPAD_SHIFT) |
        (spad << LOOP_WS_B_SPAD_SHIFT) | (UINT64_C(1) << LOOP_WS_PAIRED_SHIFT) |
        ((uint64_t)d->path << LOOP_WS_RESIDUAL_SHIFT) | ((uint64_t)d->indexed << LOOP_WS_INDEXED_SHIFT);
    p.loop.loop_ws.rs2 = (uint64_t)d->a_offset_tiles << LOOP_WS_A_OFFSET_SHIFT;
    *out = p;
    return POTAL_OK;
}

potal_status potal_resident_ws_readback_encode(const potal_resident_ws_readback *d,
    int32_t *destination, size_t bytes, potal_rocc_cmd *config_st,
    potal_rocc_cmd *mvouts, uint32_t capacity, uint32_t *count) {
    if (!d || !config_st || !mvouts || !count || d->bank > 1 ||
        !d->rows || d->rows > 16 || !d->columns || d->stride < d->columns ||
        d->stride > UINT32_MAX / sizeof(int32_t)) return POTAL_ERR_INVALID_ARGUMENT;
    if (!resident_ws_geometry()) return POTAL_ERR_UNSUPPORTED;
    if (d->columns > 144 || d->acc_tile >= 32 ||
        resident_ws_tiles(d->columns) > 32 - d->acc_tile ||
        resident_ws_tiles(d->columns) > capacity) return POTAL_ERR_CAPACITY;
    const uint64_t span = matrix_extent(d->rows, d->columns, d->stride, sizeof(int32_t));
    if (!tile_pointer_ok(destination, span, 64) || span > bytes)
        return POTAL_ERR_INVALID_ARGUMENT;
    const uint32_t n = resident_ws_tiles(d->columns);
    potal_rocc_cmd records[POTAL_RESIDENT_WS_MAX_J];
    for (uint32_t j = 0; j < n; ++j) {
        const uint32_t cols = d->columns - j * 16 < 16 ? d->columns - j * 16 : 16;
        records[j] = cmd((uint64_t)(uintptr_t)(destination + j * 16),
            ((uint64_t)d->rows << 48) | ((uint64_t)cols << 32) |
                acc_local_addr(d->bank, (d->acc_tile + j) * 16));
    }
    memcpy(mvouts, records, n * sizeof(*records));
    *config_st = cmd(CONFIG_ST, d->stride * sizeof(int32_t));
    *count = n;
    return POTAL_OK;
}

#if defined(__riscv)

/* Set by a GEMM issue, cleared by a successful POTAL_FENCE. */
static int gemm_in_flight;

static potal_status hw_status(uint64_t rd) {
    return rd <= POTAL_HW_BAD_COMMAND ? (potal_status)rd : POTAL_ERR_BAD_RESPONSE;
}

potal_status potal_scale_write(uint32_t slot, uint32_t generation, uint32_t column,
                               uint32_t carrier) {
    if (paired_owner == 2) return POTAL_ERR_BUSY;
    uint64_t rs1;
    if (potal_scale_key_encode(slot, generation, column, &rs1) != POTAL_OK ||
        !potal_carrier_is_valid(carrier)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    return hw_status(potal_rocc_scale_write(rs1, carrier));
}

potal_status potal_scale_release(uint32_t slot, uint32_t generation, uint32_t column) {
    if (paired_owner == 2) return POTAL_ERR_BUSY;
    uint64_t rs1;
    if (potal_scale_key_encode(slot, generation, column, &rs1) != POTAL_OK) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    return hw_status(potal_rocc_scale_release(rs1, 0));
}

potal_status potal_fence(void) {
    if (paired_owner) {
        return POTAL_ERR_BUSY; /* the paired session fences its own work */
    }
    const potal_status status = hw_status(potal_rocc_fence(0, 0));
    if (status == POTAL_OK) {
        gemm_in_flight = 0;
    }
    /* Order later CPU reads of C after the drained DMA stores. */
    potal_cpu_fence_rw();
    return status;
}

potal_status potal_tlb_flush(void) {
    if (gemm_in_flight || paired_owner) {
        return POTAL_ERR_BUSY;
    }
    potal_rocc_flush(0, 0);
    return POTAL_OK;
}

/* The single RoCC issue sequence of potal_gemm_ws_issue() and potal_loop_ws_issue(). */
static potal_status issue_program(const potal_gemm_ws_program *program) {
    const potal_gemm_ws_program p = *program;
    if (gemm_in_flight || paired_owner) {
        return POTAL_ERR_BUSY;
    }
    /* Make prior CPU stores to A/B visible before the DMA reads them. */
    potal_cpu_fence_rw();
    potal_rocc_config(p.config_ld_a.rs1, p.config_ld_a.rs2);
    potal_rocc_config(p.config_ld_b.rs1, p.config_ld_b.rs2);
    potal_rocc_config(p.config_ex.rs1, p.config_ex.rs2);
    potal_rocc_config(p.config_st.rs1, p.config_st.rs2);
    potal_rocc_loop_ws_bounds(p.bounds.rs1, p.bounds.rs2);
    potal_rocc_loop_ws_addrs_ab(p.addrs_ab.rs1, p.addrs_ab.rs2);
    potal_rocc_loop_ws_addrs_dc(p.addrs_dc.rs1, p.addrs_dc.rs2);
    potal_rocc_loop_ws_strides_ab(p.strides_ab.rs1, p.strides_ab.rs2);
    potal_rocc_loop_ws_strides_dc(p.strides_dc.rs1, p.strides_dc.rs2);
    potal_rocc_loop_ws(p.loop_ws.rs1, p.loop_ws.rs2);
    gemm_in_flight = 1;
    return POTAL_OK;
}

potal_status potal_gemm_ws_issue(const potal_gemm_ws *g) {
    potal_gemm_ws_program p;
    const potal_status status = potal_gemm_ws_encode(g, &p);
    return status != POTAL_OK ? status : issue_program(&p);
}

potal_status potal_loop_ws_issue(const potal_loop_ws *l) {
    potal_gemm_ws_program p;
    const potal_status status = potal_loop_ws_encode(l, &p);
    return status != POTAL_OK ? status : issue_program(&p);
}

#else /* !__riscv: validate, then report that no accelerator exists. */

potal_status potal_scale_write(uint32_t slot, uint32_t generation, uint32_t column,
                               uint32_t carrier) {
    if (paired_owner == 2) return POTAL_ERR_BUSY;
    uint64_t rs1;
    if (potal_scale_key_encode(slot, generation, column, &rs1) != POTAL_OK ||
        !potal_carrier_is_valid(carrier)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    return POTAL_ERR_NO_HARDWARE;
}

potal_status potal_scale_release(uint32_t slot, uint32_t generation, uint32_t column) {
    if (paired_owner == 2) return POTAL_ERR_BUSY;
    uint64_t rs1;
    if (potal_scale_key_encode(slot, generation, column, &rs1) != POTAL_OK) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    return POTAL_ERR_NO_HARDWARE;
}

potal_status potal_fence(void) {
    return paired_owner ? POTAL_ERR_BUSY : POTAL_ERR_NO_HARDWARE;
}

potal_status potal_tlb_flush(void) {
    return paired_owner ? POTAL_ERR_BUSY : POTAL_ERR_NO_HARDWARE;
}

potal_status potal_gemm_ws_issue(const potal_gemm_ws *g) {
    potal_gemm_ws_program p;
    const potal_status status = potal_gemm_ws_encode(g, &p);
    return status != POTAL_OK ? status : paired_owner ? POTAL_ERR_BUSY : POTAL_ERR_NO_HARDWARE;
}

potal_status potal_loop_ws_issue(const potal_loop_ws *l) {
    potal_gemm_ws_program p;
    const potal_status status = potal_loop_ws_encode(l, &p);
    return status != POTAL_OK ? status : paired_owner ? POTAL_ERR_BUSY : POTAL_ERR_NO_HARDWARE;
}

#endif

/* ---- H2 paired session ---------------------------------------------------- */

#include <stdlib.h>
#if defined(__riscv)
#include <sys/mman.h>
#endif

typedef struct resident_ws_pair {
    potal_resident_ws_load load;
    potal_resident_ws_ref token;
    int loading, readable;
} resident_ws_pair;

typedef struct resident_ws_scale {
    potal_resident_ws_ref token, resident;
    uint64_t block, column;
} resident_ws_scale;

typedef struct resident_ws_index {
    potal_resident_ws_ref token, resident;
    uint32_t count;
    uint64_t original[32];
} resident_ws_index;

typedef struct resident_ws_epoch {
    potal_resident_ws_epoch declaration;
    potal_resident_ws_fragment *fragments; /* owned copy; declaration borrows it */
    potal_resident_ws_ref token;
    uint64_t total, next;
    int initialized, final;
} resident_ws_epoch;

typedef struct resident_ws_receipt {
    potal_resident_ws_ref token;
    potal_resident_ws_readback readback;
    int ready;
} resident_ws_receipt;

enum resident_ws_pending { RESIDENT_WS_NONE, RESIDENT_WS_LOAD, RESIDENT_WS_COMPUTE, RESIDENT_WS_READBACK };

typedef struct resident_ws_state {
    potal_resident_ws_status status;
    void *pool[6];
    void (*destroy_sink)(void *);
    potal_resident_ws_counter_fn counter;
    uint64_t owner, sequence;
    int in_call, issue_complete;
    enum resident_ws_pending kind;
    uint32_t pending_pair, pending_bank, pending_tile, pending_count, pending_last;
    uint64_t pending_next;
    potal_resident_ws_ref pending_refs[9];
    resident_ws_pair pair[2];
    resident_ws_scale scale[256];
    resident_ws_index index[2];
    resident_ws_epoch epoch[2][32];
    resident_ws_receipt receipt[2];
} resident_ws_state;

typedef struct paired_pair_state {
    int      resident_b;
    uint32_t b_sp_base, b_j_tiles, b_k_rows, main_a_tiles;
    int      index_valid;        /* a complete Index window is programmed on this buffer */
    potal_index_config index;    /* that window */
    uint32_t outstanding;        /* paired LOOP_WS on this pair since the last POTAL_OK fence */
} paired_pair_state;

typedef struct h2_panel_state {
    potal_h2_panel declaration;
    potal_h2_panel_status status;
    const int8_t *source;             /* derived from the bound original matrix */
    uint32_t b_sp_base;
    int load_issued;                 /* whole B load/program handed off, not completed */
} h2_panel_state;

typedef struct h2_state {
    potal_h2_weight weight;
    potal_h2_status status;
    uint64_t owner, sequence;
    int in_call, started, program_pending;
    uint32_t acc_written;            /* ACC contents survive panel retirement */
    h2_panel_state panel[POTAL_PAIRS];
} h2_state;

struct potal_paired_session {
    potal_paired_sink sink;
    resident_ws_state *resident;            /* NULL for every legacy session */
    h2_state         *h2;                  /* mutually exclusive managed DIM32 mode */
    potal_observation_counter_fn observation_counter;
    int observation_active, observation_in_call;
    int               failed;
    int               trace_enabled;        /* POTAL_RAW_TRACE snapshot at successful open */
    potal_status      error;
    const char       *error_stage;
    uint32_t          outstanding_loops, outstanding_readbacks, outstanding_acc_updates;
    uint64_t          commands;
    uint64_t          drains;               /* accepted FENCE / RELEASE drain points */
    int               index_open;          /* a CONFIG whose writes are not complete */
    uint32_t          index_pair;
    uint32_t          index_written;
    potal_index_config index_cfg;          /* the open transaction */
    paired_pair_state pair[POTAL_PAIRS];
    uint64_t          slot_columns[POTAL_SCALE_ENTRIES];    /* live (written, not released) columns, bit c = column c */
    uint8_t           slot_generation[POTAL_SCALE_ENTRIES]; /* generation of the live columns */
};

_Static_assert(POTAL_HW_DIM <= 64, "slot_columns holds one bit per column");

static void resident_ws_completed(potal_paired_session *s);
static potal_status resident_ws_fence(potal_paired_session *s);
static potal_status resident_ws_cleanup(potal_paired_session *s);
static potal_status resident_ws_release(potal_paired_session *s, uint32_t slot,
    uint32_t generation, uint32_t column);
static void h2_completed(potal_paired_session *s);
static potal_status h2_fence(potal_paired_session *s);
static potal_status h2_cleanup(potal_paired_session *s);
static int h2_pending(const potal_paired_session *s);

static potal_status paired_fail(potal_paired_session *s, potal_status status, const char *stage) {
    if (!s->failed) {
        s->failed = 1;
        s->error = status;
        s->error_stage = stage;
    }
    if (s->h2 && status == POTAL_ERR_BAD_RESPONSE) {
        s->h2->status.unknown_completion = 1;
        s->h2->status.quarantined = 1;
    }
    return status;
}

/* A session without hardware and without a sink issues nothing. */
static int paired_can_issue(const potal_paired_session *s) {
#if defined(__riscv)
    (void)s;
    return 1;
#else
    return s->sink.cmd != NULL;
#endif
}

/*
 * One raw command in issue order: to the sink when bound, else to the
 * hardware. xd commands return the hardware status; the others POTAL_OK.
 */
static potal_status paired_raw_untraced(potal_paired_session *s, uint32_t funct, uint64_t rs1, uint64_t rs2, int xd) {
    s->commands++;
    if (s->sink.cmd != NULL) {
        const potal_paired_cmd_record rec = { funct, (uint32_t)xd, rs1, rs2 };
        if (s->h2) s->h2->in_call = 1;
        const potal_status status = s->sink.cmd(s->sink.ctx, &rec);
        if (s->h2) s->h2->in_call = 0;
        return status;
    }
#if defined(__riscv)
    switch (funct) {
        case POTAL_FUNCT_CONFIG:                    potal_rocc_config(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS:     potal_rocc_loop_ws_bounds(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB:   potal_rocc_loop_ws_addrs_ab(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_DC:   potal_rocc_loop_ws_addrs_dc(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_AB: potal_rocc_loop_ws_strides_ab(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_DC: potal_rocc_loop_ws_strides_dc(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_LOOP_WS:
            if (xd) return hw_status(potal_rocc_resident_loop_ws(rs1, rs2));
            potal_rocc_loop_ws(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_RESIDENT_VIEW:             return hw_status(potal_rocc_resident_view(rs1, rs2));
        case POTAL_FUNCT_MVOUT:                     potal_rocc_mvout(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_MVIN:                      potal_rocc_mvin(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_FLUSH:                     potal_rocc_flush(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_SCALE_WRITE:               return hw_status(potal_rocc_scale_write(rs1, rs2));
        case POTAL_FUNCT_SCALE_RELEASE:             return hw_status(potal_rocc_scale_release(rs1, rs2));
        case POTAL_FUNCT_INDEX_CONFIG:              return hw_status(potal_rocc_index_config(rs1, rs2));
        case POTAL_FUNCT_INDEX_WRITE:               return hw_status(potal_rocc_index_write(rs1, rs2));
        case POTAL_FUNCT_FENCE:                     return hw_status(potal_rocc_fence(rs1, rs2));
        default:                                    return POTAL_ERR_INVALID_ARGUMENT;
    }
#else
    (void)funct; (void)rs1; (void)rs2; (void)xd;
    return POTAL_ERR_NO_HARDWARE;
#endif
}


#if defined(__riscv) || defined(POTAL_TEST_RAW_TRACE)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline, cold))
#endif
static potal_status paired_raw_traced(
    potal_paired_session *s,
    uint32_t funct,
    uint64_t rs1,
    uint64_t rs2,
    int xd
) {
    const unsigned long long seq =
        (unsigned long long)s->commands + 1;

    fprintf(stderr,
        "[POTAL_RAW] ENTER seq=%llu funct=%u xd=%d "
        "rs1=0x%016llx rs2=0x%016llx\n",
        seq, funct, xd,
        (unsigned long long)rs1,
        (unsigned long long)rs2);
    fflush(stderr);

    const potal_status status =
        paired_raw_untraced(s, funct, rs1, rs2, xd);

    fprintf(stderr,
        "[POTAL_RAW] EXIT seq=%llu funct=%u status=%d\n",
        seq, funct, (int)status);
    fflush(stderr);

    return status;
}
#endif

static potal_status paired_raw(
    potal_paired_session *s,
    uint32_t funct,
    uint64_t rs1,
    uint64_t rs2,
    int xd
) {
#if defined(__riscv) || defined(POTAL_TEST_RAW_TRACE)
    if (s->trace_enabled) {
        return paired_raw_traced(s, funct, rs1, rs2, xd);
    }
#endif
    return paired_raw_untraced(s, funct, rs1, rs2, xd);
}

#define PAIRED_RAW(session, funct, rs1, rs2, xd, stage)                                  \
    do {                                                                               \
        const potal_status raw_status_ = paired_raw((session), (funct), (rs1), (rs2), (xd)); \
        if (raw_status_ != POTAL_OK) {                                                 \
            return paired_fail((session), raw_status_, (stage));                       \
        }                                                                              \
    } while (0)

/* Every outstanding paired LOOP_WS and readback drained (a FENCE or an accepted RELEASE). */
static void paired_drained(potal_paired_session *s) {
    s->outstanding_loops = 0;
    s->outstanding_readbacks = 0;
    s->outstanding_acc_updates = 0;
    for (uint32_t p = 0; p < POTAL_PAIRS; ++p) {
        s->pair[p].outstanding = 0;
    }
    s->drains++;
    if (s->resident) resident_ws_completed(s);
    if (s->h2) h2_completed(s);
}

static potal_status paired_ready(const potal_paired_session *s) {
    if (s == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if (s->h2 && s->h2->in_call) return POTAL_ERR_BUSY;
    if (s->observation_in_call) return POTAL_ERR_BUSY;
    if (s->failed) {
        return s->error;
    }
    if (s->h2 && s->h2->status.cleaned) return POTAL_ERR_BUSY;
    return POTAL_OK;
}

potal_status potal_paired_session_open(const potal_paired_sink *sink, potal_paired_session **out) {
    if (out == NULL || (sink != NULL && sink->cmd == NULL)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (paired_owner) {
        return POTAL_ERR_BUSY; /* another session owns the accelerator, or a failed one left it unknown */
    }
#if defined(__riscv)
    if (gemm_in_flight) {
        return POTAL_ERR_BUSY; /* regular work outstanding: potal_fence() first */
    }
#endif
    potal_paired_session *s = calloc(1, sizeof(*s));
    if (s == NULL) {
        return POTAL_ERR_CAPACITY;
    }
    if (sink != NULL) {
        s->sink = *sink;
    }
    const char *env = getenv("POTAL_RAW_TRACE");
    s->trace_enabled = env != NULL && env[0] == '1';
    paired_owner = 1;
    *out = s;
    return POTAL_OK;
}

potal_status potal_paired_session_close(potal_paired_session *s) {
    if (s == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if (s->resident || s->h2) return POTAL_ERR_BUSY; /* managed pointer-to-pointer close owns storage */
    if (s->observation_in_call || (s->observation_active && !s->failed)) return POTAL_ERR_BUSY;
    const int outstanding = s->outstanding_loops != 0 || s->outstanding_readbacks != 0 ||
                            s->outstanding_acc_updates != 0;
    int       live        = 0;
    for (uint32_t slot = 0; slot < POTAL_SCALE_ENTRIES; ++slot) {
        live |= s->slot_columns[slot] != 0;
    }
    if ((outstanding || live) && !s->failed) {
        return POTAL_ERR_BUSY; /* potal_paired_fence() / release the live scale entries first */
    }
    const potal_status status = s->failed ? s->error : POTAL_OK;
    /*
     * A failed session with undrained work, or with scale entries it could not
     * release, leaves the accelerator state unknown: keep it owned.
     */
    if (!outstanding && !live) {
        paired_owner = 0;
    }
    free(s);
    return status;
}

potal_paired_session_status potal_paired_session_query(const potal_paired_session *s) {
    potal_paired_session_status q;
    memset(&q, 0, sizeof(q));
    if (s == NULL) {
        return q;
    }
    q.failed = (uint32_t)s->failed;
    q.error = s->error;
    q.error_stage = s->failed ? s->error_stage : NULL;
    q.outstanding_loops = s->outstanding_loops;
    q.outstanding_readbacks = s->outstanding_readbacks;
    q.outstanding_acc_updates = s->outstanding_acc_updates;
    q.index_transaction_open = (uint32_t)s->index_open;
    q.index_transaction_pair = s->index_pair;
    q.index_transaction_written = s->index_written;
    q.commands = s->commands;
    q.drains = s->drains;
    for (uint32_t i = 0; i < POTAL_SCALE_ENTRIES; ++i) {
        q.live_scale_slots += s->slot_columns[i] != 0;
    }
    for (uint32_t p = 0; p < POTAL_PAIRS; ++p) {
        q.pair[p].resident_b = (uint32_t)s->pair[p].resident_b;
        q.pair[p].b_sp_base = s->pair[p].b_sp_base;
        q.pair[p].b_j_tiles = s->pair[p].b_j_tiles;
        q.pair[p].b_k_rows = s->pair[p].b_k_rows;
        q.pair[p].main_a_tiles = s->pair[p].main_a_tiles;
        q.pair[p].index_valid = (uint32_t)s->pair[p].index_valid;
        q.pair[p].index_active_rows = s->pair[p].index_valid ? s->pair[p].index.active_rows : 0;
        q.pair[p].outstanding = s->pair[p].outstanding;
    }
    return q;
}

potal_status potal_paired_tlb_flush(potal_paired_session *s) {
    if (s && s->resident) return POTAL_ERR_BUSY;
    const potal_status ready = paired_ready(s);
    if (ready != POTAL_OK) {
        return ready;
    }
    if (s->h2) {
        if (s->h2->started || h2_pending(s) || s->index_open) return POTAL_ERR_BUSY;
        for (uint32_t slot = 0; slot < POTAL_SCALE_ENTRIES; ++slot)
            if (s->slot_columns[slot]) return POTAL_ERR_BUSY;
    }
    if (s->outstanding_loops != 0 || s->outstanding_readbacks != 0 || s->outstanding_acc_updates != 0) {
        return POTAL_ERR_BUSY; /* no flush under in-flight DMA */
    }
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
    if (s->h2) s->h2->status.control_pending = 1;
    PAIRED_RAW(s, POTAL_FUNCT_FLUSH, 0, 0, 0, "tlb_flush");
    return POTAL_OK;
}

static potal_status paired_scale_write_core(potal_paired_session *s, uint32_t slot, uint32_t generation,
                                      uint32_t column, uint32_t carrier) {
    const potal_status ready = paired_ready(s);
    if (ready != POTAL_OK) {
        return ready;
    }
    uint64_t rs1;
    if (potal_scale_key_encode(slot, generation, column, &rs1) != POTAL_OK || !potal_carrier_is_valid(carrier)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    /*
     * A live slot belongs to an unreleased window: another generation would
     * alias it, and a live column would be overwritten (the hardware answers
     * SCALE_CONFLICT). A window may write fewer than DIM columns (the padded
     * last J tile), so liveness is tracked per column.
     */
    const uint64_t bit = (uint64_t)1 << column;
    if (s->slot_columns[slot] != 0 && (s->slot_generation[slot] != generation || (s->slot_columns[slot] & bit) != 0)) {
        return POTAL_ERR_CAPACITY;
    }
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
    if (s->h2) s->h2->status.control_pending = 1;
    PAIRED_RAW(s, POTAL_FUNCT_SCALE_WRITE, rs1, carrier, 1, "scale_write");
    s->slot_columns[slot] |= bit;
    s->slot_generation[slot] = (uint8_t)generation;
    if (s->h2) s->h2->status.control_pending = 0;
    return POTAL_OK;
}

static potal_status paired_scale_release_core(potal_paired_session *s, uint32_t slot, uint32_t generation,
                                        uint32_t column) {
    const potal_status ready = paired_ready(s);
    if (ready != POTAL_OK) {
        return ready;
    }
    uint64_t rs1;
    if (potal_scale_key_encode(slot, generation, column, &rs1) != POTAL_OK) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if ((s->slot_columns[slot] & ((uint64_t)1 << column)) == 0 || s->slot_generation[slot] != generation) {
        return POTAL_ERR_INVALID_ARGUMENT; /* the hardware would answer RELEASE_MISS */
    }
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
    /*
     * The ingress accepts a RELEASE only when every older command drained
     * (Controller.scala: releaseSafe := allDrained, the same signal as
     * fenceDrained): an accepted release is a drain point, like POTAL_FENCE.
     */
    const potal_status released = paired_raw(s, POTAL_FUNCT_SCALE_RELEASE, rs1, 0, 1);
    if (released != POTAL_OK) {
        if (s->h2) {
            s->h2->status.last_drain = released;
            s->h2->status.unknown_completion = 1;
            s->h2->status.quarantined = 1;
        }
        return paired_fail(s, released, "scale_release");
    }
    s->slot_columns[slot] &= ~((uint64_t)1 << column); /* the slot is free when its last live column is released */
    paired_drained(s);
    return POTAL_OK;
}

static potal_status paired_index_config_core(potal_paired_session *s, const potal_index_config *cfg) {
    const potal_status ready = paired_ready(s);
    if (ready != POTAL_OK) {
        return ready;
    }
    potal_rocc_cmd c;
    const potal_status status = potal_index_config_encode(cfg, &c);
    if (status != POTAL_OK) {
        return status;
    }
    if (s->index_open) {
        return POTAL_ERR_BUSY; /* the previous window's writes are not complete: no interleaving */
    }
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
    if (s->h2) s->h2->status.control_pending = 1;
    PAIRED_RAW(s, POTAL_FUNCT_INDEX_CONFIG, c.rs1, c.rs2, 1, "index_config");
    /* The hardware invalidated the buffer's entries: the pair's window is valid again only when complete. */
    s->pair[cfg->pair].index_valid = 0;
    s->index_open = 1;
    s->index_pair = cfg->pair;
    s->index_written = 0;
    s->index_cfg = *cfg;
    if (s->h2) s->h2->status.control_pending = 0;
    return POTAL_OK;
}

static potal_status paired_index_write_core(potal_paired_session *s, const potal_index_write *w) {
    const potal_status ready = paired_ready(s);
    if (ready != POTAL_OK) {
        return ready;
    }
    if (!s->index_open) {
        return POTAL_ERR_BUSY; /* no transaction: INDEX_CONFIG first */
    }
    potal_rocc_cmd c;
    const potal_status status = potal_index_write_encode(&s->index_cfg, w, &c);
    if (status != POTAL_OK) {
        return status;
    }
    if (w->slot != s->index_written) {
        return POTAL_ERR_INVALID_ARGUMENT; /* slots are written in order, each once */
    }
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
    if (s->h2) s->h2->status.control_pending = 1;
    PAIRED_RAW(s, POTAL_FUNCT_INDEX_WRITE, c.rs1, c.rs2, 1, "index_write");
    s->index_written++;
    if (s->index_written == s->index_cfg.active_rows) {
        paired_pair_state *pr = &s->pair[s->index_pair];
        pr->index_valid = 1;
        pr->index = s->index_cfg;
        s->index_open = 0;
    }
    if (s->h2) s->h2->status.control_pending = 0;
    return POTAL_OK;
}

potal_status potal_paired_scale_write(potal_paired_session *s, uint32_t slot, uint32_t generation,
                                      uint32_t column, uint32_t carrier) {
    if (s && s->resident) return POTAL_ERR_BUSY;
    if (s && s->h2) {
        const potal_status ready = paired_ready(s);
        if (ready != POTAL_OK) return ready;
        if (s->h2->status.pending_b || s->h2->status.control_pending) return POTAL_ERR_BUSY;
    }
    return paired_scale_write_core(s, slot, generation, column, carrier);
}

potal_status potal_paired_scale_release(potal_paired_session *s, uint32_t slot, uint32_t generation,
                                        uint32_t column) {
    if (s && s->resident) return resident_ws_release(s, slot, generation, column);
    return paired_scale_release_core(s, slot, generation, column);
}

potal_status potal_paired_index_config(potal_paired_session *s, const potal_index_config *cfg) {
    if (s && s->resident) return POTAL_ERR_BUSY;
    if (s && s->h2) {
        const potal_status ready = paired_ready(s);
        if (ready != POTAL_OK) return ready;
        if (s->h2->status.pending_b || s->h2->status.control_pending) return POTAL_ERR_BUSY;
        if (!cfg || cfg->pair >= POTAL_PAIRS) return POTAL_ERR_INVALID_ARGUMENT;
        const h2_panel_state *p = &s->h2->panel[cfg->pair];
        if (!p->status.declared || !p->status.readers_issued ||
            cfg->b_sp_base != p->b_sp_base || cfg->j_tiles != p->declaration.j_tiles ||
            cfg->source_k_rows != p->declaration.k_tiles * POTAL_HW_DIM)
            return POTAL_ERR_INVALID_ARGUMENT;
    }
    return paired_index_config_core(s, cfg);
}

potal_status potal_paired_index_write(potal_paired_session *s, const potal_index_write *w) {
    if (s && s->resident) return POTAL_ERR_BUSY;
    if (s && s->h2) {
        const potal_status ready = paired_ready(s);
        if (ready != POTAL_OK) return ready;
        if (s->h2->status.pending_b || s->h2->status.control_pending) return POTAL_ERR_BUSY;
    }
    return paired_index_write_core(s, w);
}

static potal_status paired_issue_program_view(potal_paired_session *s, const potal_gemm_ws_program *p,
    const potal_rocc_cmd *view, const char *stage) {
#if defined(__riscv)
    potal_cpu_fence_rw(); /* prior CPU stores to A/B visible before the DMA reads them */
#endif
    PAIRED_RAW(s, POTAL_FUNCT_CONFIG, p->config_ld_a.rs1, p->config_ld_a.rs2, 0, stage);
    PAIRED_RAW(s, POTAL_FUNCT_CONFIG, p->config_ld_b.rs1, p->config_ld_b.rs2, 0, stage);
    PAIRED_RAW(s, POTAL_FUNCT_CONFIG, p->config_ex.rs1, p->config_ex.rs2, 0, stage);
    PAIRED_RAW(s, POTAL_FUNCT_CONFIG, p->config_st.rs1, p->config_st.rs2, 0, stage);
    PAIRED_RAW(s, POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS, p->bounds.rs1, p->bounds.rs2, 0, stage);
    PAIRED_RAW(s, POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB, p->addrs_ab.rs1, p->addrs_ab.rs2, 0, stage);
    PAIRED_RAW(s, POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_DC, p->addrs_dc.rs1, p->addrs_dc.rs2, 0, stage);
    PAIRED_RAW(s, POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_AB, p->strides_ab.rs1, p->strides_ab.rs2, 0, stage);
    PAIRED_RAW(s, POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_DC, p->strides_dc.rs1, p->strides_dc.rs2, 0, stage);
    if (view) PAIRED_RAW(s, POTAL_FUNCT_RESIDENT_VIEW, view->rs1, view->rs2, 1, stage);
    PAIRED_RAW(s, POTAL_FUNCT_LOOP_WS, p->loop_ws.rs1, p->loop_ws.rs2, view != NULL, stage);
    return POTAL_OK;
}

/* Keep the private baseline signature used by the included-core E4A harness. */
static potal_status paired_issue_program(potal_paired_session *s, const potal_gemm_ws_program *p, const char *stage) {
    return paired_issue_program_view(s, p, NULL, stage);
}

potal_status potal_paired_loop_ws_issue(potal_paired_session *s, const potal_paired_loop_ws *pl) {
    if (s && (s->resident || s->h2)) return POTAL_ERR_BUSY;
    const potal_status ready = paired_ready(s);
    if (ready != POTAL_OK) {
        return ready;
    }
    if (s->outstanding_acc_updates != 0) {
        return POTAL_ERR_BUSY;
    }
    potal_gemm_ws_program p;
    const potal_status status = potal_paired_loop_ws_encode(pl, &p);
    if (status != POTAL_OK) {
        return status;
    }
    paired_pair_state *pr = &s->pair[pl->pair];
    const potal_loop_ws *l = &pl->loop;
    const uint32_t dim = POTAL_HW_DIM;
    if (pl->path == POTAL_PAIRED_MAIN) {
        /* The pair's banks are rewritten: every earlier reader of the pair must have drained. */
        if (pr->outstanding != 0) {
            return POTAL_ERR_BUSY;
        }
    } else {
        /* Resident B and a complete Index window that describe the same panel; A behind Main A. */
        const uint32_t active = l->k_tiles * dim - l->pad_k;
        if (!pr->resident_b || !pr->index_valid || pr->index.b_sp_base != pr->b_sp_base ||
            pr->index.j_tiles != pr->b_j_tiles || pr->index.source_k_rows != pr->b_k_rows ||
            pr->index.j_tiles != l->j_tiles || pr->index.active_rows != active ||
            (s->index_open && s->index_pair == pl->pair) || pl->a_offset_tiles < pr->main_a_tiles) {
            return POTAL_ERR_INVALID_ARGUMENT;
        }
    }
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
    const potal_status issued = paired_issue_program(s, &p, pl->path == POTAL_PAIRED_MAIN ? "main_loop_ws" : "residual_loop_ws");
    if (issued != POTAL_OK) {
        return issued;
    }
    if (pl->path == POTAL_PAIRED_MAIN) {
        pr->resident_b = 1;
        pr->b_sp_base = (uint32_t)(pair_weight_begin(pl->pair) + POTAL_HW_BANK_ROWS - (uint64_t)l->k_tiles * l->j_tiles * dim);
        pr->b_j_tiles = l->j_tiles;
        pr->b_k_rows = l->k_tiles * dim - l->pad_k;
        pr->main_a_tiles = l->i_tiles * l->k_tiles;
        pr->index_valid = 0; /* a new panel: the old window no longer describes it */
    }
    pr->outstanding++;
    s->outstanding_loops++;
    return POTAL_OK;
}

potal_status potal_paired_acc_readback_issue(potal_paired_session *s, const potal_acc_readback *r) {
    if (s && s->resident) return POTAL_ERR_BUSY;
    const potal_status ready = paired_ready(s);
    if (ready != POTAL_OK) {
        return ready;
    }
    if (s->outstanding_acc_updates != 0) {
        return POTAL_ERR_BUSY;
    }
    potal_rocc_cmd config_st;
    potal_rocc_cmd mvouts[POTAL_WORK_IDS_PER_LOOP];
    uint32_t count = 0;
    const potal_status status = potal_acc_readback_encode(r, &config_st, mvouts, POTAL_WORK_IDS_PER_LOOP, &count);
    if (status != POTAL_OK) {
        return status;
    }
    if (s->h2) {
        if (!(s->h2->acc_written & (1u << r->acc_bank)) || s->h2->status.control_pending)
            return POTAL_ERR_BUSY;
        const uint64_t extent = matrix_extent((uint64_t)r->i_tiles * POTAL_HW_DIM - r->pad_i,
            (uint64_t)r->j_tiles * POTAL_HW_DIM - r->pad_j, r->ldc, sizeof(int32_t));
        if (ranges_overlap((uintptr_t)r->c, extent, (uintptr_t)s->h2->weight.base, s->h2->weight.bytes))
            return POTAL_ERR_INVALID_ARGUMENT;
        if (s->outstanding_readbacks == UINT32_MAX) return POTAL_ERR_CAPACITY;
    }
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
    if (s->h2) s->outstanding_readbacks++; /* before even the first CONFIG can have effects */
    PAIRED_RAW(s, POTAL_FUNCT_CONFIG, config_st.rs1, config_st.rs2, 0, "acc_readback");
    for (uint32_t t = 0; t < count; ++t) {
        PAIRED_RAW(s, POTAL_FUNCT_MVOUT, mvouts[t].rs1, mvouts[t].rs2, 0, "acc_readback");
    }
    if (!s->h2) s->outstanding_readbacks++;
    return POTAL_OK;
}

potal_status potal_paired_acc_bias_issue(potal_paired_session *s, const potal_acc_bias *bias) {
    if (s && (s->resident || s->h2)) return POTAL_ERR_BUSY;
    const potal_status ready = paired_ready(s);
    if (ready != POTAL_OK) {
        return ready;
    }
    if (s->outstanding_loops || s->outstanding_readbacks || s->outstanding_acc_updates) {
        return POTAL_ERR_BUSY;
    }
    potal_rocc_cmd config_ld;
    potal_rocc_cmd mvins[POTAL_WORK_IDS_PER_LOOP];
    uint32_t count = 0;
    const potal_status status = potal_acc_bias_encode(bias, &config_ld, mvins, POTAL_WORK_IDS_PER_LOOP, &count);
    if (status != POTAL_OK) {
        return status;
    }
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
#if defined(__riscv)
    potal_cpu_fence_rw();
#endif
    PAIRED_RAW(s, POTAL_FUNCT_CONFIG, config_ld.rs1, config_ld.rs2, 0, "acc_bias");
    for (uint32_t i = 0; i < count; ++i) {
        /* Keep ownership even if a test sink reports an uncertain partial issue. */
        s->outstanding_acc_updates++;
        PAIRED_RAW(s, POTAL_FUNCT_MVIN, mvins[i].rs1, mvins[i].rs2, 0, "acc_bias");
    }
    return POTAL_OK;
}

potal_status potal_paired_session_cleanup(potal_paired_session *s) {
    if (s == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if (s->observation_in_call) return POTAL_ERR_BUSY;
    if (s->resident) return resident_ws_cleanup(s);
    if (s->h2) return h2_cleanup(s);
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
    if (s->failed && strcmp(s->error_stage != NULL ? s->error_stage : "", "fence") == 0) {
        return s->error; /* a fence that did not answer OK is not retried */
    }
    if (s->outstanding_loops != 0 || s->outstanding_readbacks != 0 || s->outstanding_acc_updates != 0) {
        const potal_status status = paired_raw(s, POTAL_FUNCT_FENCE, 0, 0, 1);
#if defined(__riscv)
        potal_cpu_fence_rw();
#endif
        if (status != POTAL_OK) {
            return paired_fail(s, status, "fence");
        }
        paired_drained(s);
    }
    for (uint32_t slot = 0; slot < POTAL_SCALE_ENTRIES; ++slot) {
        for (uint32_t col = 0; col < POTAL_HW_DIM && s->slot_columns[slot] != 0; ++col) {
            if ((s->slot_columns[slot] & ((uint64_t)1 << col)) == 0) {
                continue;
            }
            uint64_t rs1;
            if (potal_scale_key_encode(slot, s->slot_generation[slot], col, &rs1) != POTAL_OK) {
                return POTAL_ERR_INVALID_ARGUMENT;
            }
            const potal_status status = paired_raw(s, POTAL_FUNCT_SCALE_RELEASE, rs1, 0, 1);
            if (status != POTAL_OK) {
                return paired_fail(s, status, "cleanup_release");
            }
            s->slot_columns[slot] &= ~((uint64_t)1 << col);
        }
    }
    s->index_open = 0; /* a half-programmed window is never used: the next INDEX_CONFIG replaces it */
    return POTAL_OK;
}

potal_status potal_paired_fence(potal_paired_session *s) {
    if (s == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if (s->resident) return resident_ws_fence(s);
    if (s->h2) return h2_fence(s);
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
    const potal_status status = paired_raw(s, POTAL_FUNCT_FENCE, 0, 0, 1);
#if defined(__riscv)
    potal_cpu_fence_rw(); /* later CPU reads of readback buffers after the drained DMA stores */
#endif
    if (status != POTAL_OK) {
        return paired_fail(s, status, "fence"); /* outstanding work stays outstanding */
    }
    paired_drained(s);
    return POTAL_OK;
}

/* ---- Managed H2: ordinary rear-packed panels and one next-B prefetch ----- */

static uint64_t h2_owner_sequence;

static potal_status h2_ready(const potal_paired_session *s) {
    if (!s || !s->h2) return POTAL_ERR_INVALID_ARGUMENT;
    return paired_ready(s);
}

static h2_panel_state *h2_find(potal_paired_session *s, potal_h2_ref ref) {
    if (!ref.owner || !ref.generation || ref.pair >= POTAL_PAIRS) return NULL;
    h2_panel_state *p = &s->h2->panel[ref.pair];
    return p->status.declared && p->status.ref.owner == ref.owner &&
        p->status.ref.generation == ref.generation ? p : NULL;
}

static int h2_live_scales(const potal_paired_session *s) {
    for (uint32_t slot = 0; slot < POTAL_SCALE_ENTRIES; ++slot)
        if (s->slot_columns[slot]) return 1;
    return 0;
}

static int h2_pending(const potal_paired_session *s) {
    const h2_state *h = s->h2;
    return h->status.pending_b || h->status.control_pending || h->program_pending ||
        h->panel[0].status.loading || h->panel[1].status.loading ||
        s->outstanding_loops || s->outstanding_readbacks || s->outstanding_acc_updates;
}

/* Only paired_drained reaches here. Hardware progress and status queries do not. */
static void h2_completed(potal_paired_session *s) {
    h2_state *h = s->h2;
#if defined(__riscv)
    potal_cpu_fence_rw();
#endif
    for (uint32_t pair = 0; pair < POTAL_PAIRS; ++pair) {
        h2_panel_state *p = &h->panel[pair];
        if (p->status.loading) {
            p->status.readable = !s->failed && p->load_issued;
            p->status.loading = 0;
            if (p->status.readable) {
                paired_pair_state *pr = &s->pair[pair];
                pr->resident_b = 1;
                pr->b_sp_base = p->b_sp_base;
                pr->b_j_tiles = p->declaration.j_tiles;
                pr->b_k_rows = p->declaration.k_tiles * POTAL_HW_DIM;
                pr->main_a_tiles = p->declaration.main_i_tiles * p->declaration.k_tiles;
            }
        }
        if (!s->failed) p->status.readers_completed = p->status.readers_issued;
        else p->status.readable = 0;
    }
    h->status.pending_b = 0;
    h->status.control_pending = 0;
    h->status.last_drain = POTAL_OK;
    h->program_pending = 0;
}

potal_status potal_paired_h2_begin(potal_paired_session *s, const potal_h2_weight *w) {
    if (s && s->resident) return POTAL_ERR_BUSY;
    potal_status status = paired_ready(s);
    if (status != POTAL_OK) return status;
    if (s->h2) return POTAL_ERR_BUSY;
    if (POTAL_HW_DIM != 32 || POTAL_HW_BANK_NUM != 4) return POTAL_ERR_UNSUPPORTED;
    if (!w || !w->base || !w->k || !w->n || w->ldb < w->n || !w->bytes)
        return POTAL_ERR_INVALID_ARGUMENT;
    const uint64_t extent = matrix_extent(w->k, w->n, w->ldb, sizeof(int8_t));
    if (!tile_pointer_ok(w->base, extent, POTAL_HW_DIM) || extent > w->bytes ||
        w->bytes > UINTPTR_MAX - (uintptr_t)w->base)
        return POTAL_ERR_INVALID_ARGUMENT;
    if (s->outstanding_loops || s->outstanding_readbacks || s->outstanding_acc_updates ||
        s->index_open || h2_live_scales(s))
        return POTAL_ERR_BUSY;
    for (uint32_t pair = 0; pair < POTAL_PAIRS; ++pair)
        if (s->pair[pair].resident_b || s->pair[pair].index_valid || s->pair[pair].outstanding)
            return POTAL_ERR_BUSY;
    if (h2_owner_sequence == UINT64_MAX) return POTAL_ERR_CAPACITY;
    if (!paired_can_issue(s)) return POTAL_ERR_NO_HARDWARE;
    h2_state *h = calloc(1, sizeof(*h));
    if (!h) return POTAL_ERR_CAPACITY;
    h->weight = *w;
#if defined(__riscv)
    /* Publish the entire immutable borrowed B allocation while idle.
     * Rocket stalls a CPU fence on RoCC busy; repeating it at prefetch
     * would serialize the load behind the current Execute work. */
    potal_cpu_fence_rw();
#endif
    h->owner = ++h2_owner_sequence;
    h->status.active = 1;
    h->status.source_bound = 1;
    s->h2 = h;
    paired_owner = 2; /* Same managed owner exclusion as resident_ws, never another issuer. */
    return POTAL_OK;
}

/* Check every tile's source and rear SP layout before issuing any command.
 * No caller-supplied panel pointer participates in address derivation.
 */
static potal_status h2_panel_check(const h2_state *h, const potal_h2_panel *d,
    const int8_t **source, uint32_t *sp_base) {
    if (!d || d->pair >= POTAL_PAIRS || !d->readers || !d->k_tiles || !d->j_tiles ||
        !d->main_i_tiles || d->k_tiles > LOOP_WS_FIELD16_MAX ||
        d->j_tiles > LOOP_WS_FIELD16_MAX || d->main_i_tiles > LOOP_WS_FIELD16_MAX ||
        d->pad_j >= POTAL_HW_DIM || d->main_pad_i >= POTAL_HW_DIM ||
        d->k_begin % POTAL_HW_DIM || d->j_begin % POTAL_HW_DIM)
        return POTAL_ERR_INVALID_ARGUMENT;
    const uint64_t rows = (uint64_t)d->k_tiles * POTAL_HW_DIM;
    const uint64_t cols = (uint64_t)d->j_tiles * POTAL_HW_DIM - d->pad_j;
    if (d->k_begin >= h->weight.k || rows > h->weight.k - d->k_begin ||
        d->j_begin >= h->weight.n || cols > h->weight.n - d->j_begin)
        return POTAL_ERR_INVALID_ARGUMENT;
    const uint64_t bank_tiles = POTAL_HW_BANK_ROWS / POTAL_HW_DIM;
    if ((uint64_t)d->k_tiles * d->j_tiles > bank_tiles ||
        (uint64_t)d->main_i_tiles * d->k_tiles > bank_tiles ||
        (uint64_t)d->main_i_tiles * d->j_tiles > POTAL_WORK_IDS_PER_LOOP ||
        (uint64_t)d->main_i_tiles * d->j_tiles * POTAL_HW_DIM > POTAL_HW_ACC_ROWS / 2)
        return POTAL_ERR_CAPACITY;
    /* The full original extent was checked at begin: these products cannot wrap. */
    const uint64_t offset = d->k_begin * h->weight.ldb + d->j_begin;
    const uint64_t extent = matrix_extent(rows, cols, h->weight.ldb, sizeof(int8_t));
    if (offset >= h->weight.bytes || !extent || extent > h->weight.bytes - offset)
        return POTAL_ERR_INVALID_ARGUMENT;
    const uintptr_t address = (uintptr_t)h->weight.base + (uintptr_t)offset;
    if (!tile_pointer_ok((const void *)address, extent, POTAL_HW_DIM))
        return POTAL_ERR_INVALID_ARGUMENT;
    const uint64_t base = pair_weight_begin(d->pair) + POTAL_HW_BANK_ROWS -
        rows * d->j_tiles;
    if (base > UINT32_MAX || base + rows * d->j_tiles > LOCAL_ADDR_FULL_ACC_ROW)
        return POTAL_ERR_CAPACITY;
    *source = (const int8_t *)address;
    *sp_base = (uint32_t)base;
    return POTAL_OK;
}

static potal_status h2_issue_prefetch(potal_paired_session *s, h2_panel_state *p) {
    h2_state *h = s->h2;
    const potal_h2_panel *d = &p->declaration;
    /* Ownership precedes CONFIG too, not merely the first successful MVIN. */
    p->status.prefetched = 1;
    h->status.pending_b = 1;
    p->status.loading = 1;
    /* B was published at begin and cannot be modified until safe close. */
    PAIRED_RAW(s, POTAL_FUNCT_CONFIG, config_ld_rs1(0), h->weight.ldb, 0, "h2_prefetch");
    for (uint32_t k = 0; k < d->k_tiles; ++k) {
        for (uint32_t j = 0; j < d->j_tiles; ++j) {
            const uint64_t cols = POTAL_HW_DIM - (j + 1 == d->j_tiles ? d->pad_j : 0);
            const uint64_t address = (uint64_t)(uintptr_t)p->source +
                (uint64_t)k * POTAL_HW_DIM * h->weight.ldb + (uint64_t)j * POTAL_HW_DIM;
            const uint64_t row = p->b_sp_base + ((uint64_t)k * d->j_tiles + j) * POTAL_HW_DIM;
            h->status.attempted_b_mvins++;
            PAIRED_RAW(s, POTAL_FUNCT_MVIN, address,
                ((uint64_t)POTAL_HW_DIM << MVOUT_ROWS_SHIFT) | (cols << MVOUT_COLS_SHIFT) | row,
                0, "h2_prefetch");
        }
    }
    p->load_issued = 1;
    return POTAL_OK;
}

potal_status potal_paired_h2_panel_begin(potal_paired_session *s,
    const potal_h2_panel *d, uint32_t prefetch, potal_h2_ref *reference) {
    potal_status status = h2_ready(s);
    if (status != POTAL_OK) return status;
    if (!reference || prefetch > 1) return POTAL_ERR_INVALID_ARGUMENT;
    h2_state *h = s->h2;
    const int8_t *source;
    uint32_t base;
    status = h2_panel_check(h, d, &source, &base);
    if (status != POTAL_OK) return status;
    if (h->status.pending_b || h->status.control_pending || s->index_open ||
        h->panel[d->pair].status.declared || s->pair[d->pair].outstanding)
        return POTAL_ERR_BUSY;
    for (uint32_t pair = 0; pair < POTAL_PAIRS; ++pair)
        if (h->panel[pair].status.declared &&
            h->panel[pair].declaration.residency_index == d->residency_index)
            return POTAL_ERR_INVALID_ARGUMENT;
    if (h->sequence == UINT64_MAX ||
        (prefetch && h->status.attempted_b_mvins >
            UINT64_MAX - (uint64_t)d->k_tiles * d->j_tiles))
        return POTAL_ERR_CAPACITY;
    if (prefetch && !paired_can_issue(s)) return POTAL_ERR_NO_HARDWARE;
    h2_panel_state *p = &h->panel[d->pair];
    memset(p, 0, sizeof(*p));
    p->declaration = *d;
    p->source = source;
    p->b_sp_base = base;
    p->status.ref = (potal_h2_ref){h->owner, ++h->sequence, d->pair};
    p->status.residency_index = d->residency_index;
    p->status.declared = 1;
    p->status.readers_declared = d->readers;
    h->started = 1;
    *reference = p->status.ref; /* Even an interrupted load has an owned identity. */
    if (!prefetch) return POTAL_OK;
    return h2_issue_prefetch(s, p);
}

potal_status potal_paired_h2_panel_prefetch(potal_paired_session *s, potal_h2_ref reference) {
    const potal_status status = h2_ready(s);
    if (status != POTAL_OK) return status;
    h2_panel_state *p = h2_find(s, reference);
    if (!p) return POTAL_ERR_INVALID_ARGUMENT;
    h2_state *h = s->h2;
    if (h->status.pending_b || h->status.control_pending || s->index_open ||
        p->status.prefetched || p->status.loading || p->status.readable ||
        p->load_issued || p->status.readers_issued || s->pair[reference.pair].outstanding)
        return POTAL_ERR_BUSY;
    if (h->status.attempted_b_mvins >
        UINT64_MAX - (uint64_t)p->declaration.k_tiles * p->declaration.j_tiles)
        return POTAL_ERR_CAPACITY;
    return h2_issue_prefetch(s, p);
}

potal_status potal_paired_h2_loop_issue(potal_paired_session *s, potal_h2_ref reference,
    uint32_t ordinal, const potal_paired_loop_ws *pl) {
    potal_status status = h2_ready(s);
    if (status != POTAL_OK) return status;
    h2_state *h = s->h2;
    h2_panel_state *p = h2_find(s, reference);
    if (!p) return POTAL_ERR_INVALID_ARGUMENT;
    if (h->status.pending_b || h->status.control_pending || s->outstanding_acc_updates)
        return POTAL_ERR_BUSY;
    potal_gemm_ws_program program;
    status = potal_paired_loop_ws_encode(pl, &program); /* REAL ordinary descriptor */
    if (status != POTAL_OK) return status;
    const potal_loop_ws *l = &pl->loop;
    const potal_h2_panel *d = &p->declaration;
    if (ordinal != p->status.readers_issued || ordinal >= d->readers ||
        pl->pair != reference.pair || l->j_tiles != d->j_tiles || l->pad_j != d->pad_j ||
        pl->path != (ordinal ? POTAL_PAIRED_RESIDUAL : POTAL_PAIRED_MAIN))
        return POTAL_ERR_INVALID_ARGUMENT;
    paired_pair_state *pr = &s->pair[reference.pair];
    if (!ordinal) {
        if (l->i_tiles != d->main_i_tiles || l->pad_i != d->main_pad_i ||
            l->k_tiles != d->k_tiles || l->pad_k || l->b != p->source || l->ldb != h->weight.ldb)
            return POTAL_ERR_INVALID_ARGUMENT;
        if (pr->outstanding || (s->index_open && s->index_pair == reference.pair))
            return POTAL_ERR_BUSY;
        if (p->status.prefetched) {
            if (!p->status.readable) return POTAL_ERR_BUSY;
            /* The checked ordinary Main uses the exact same rear placement:
             * (2*pair+2)*BANK_ROWS - K*J*DIM, identical K/J/pad and real stride.
             * Only suppress the B address; no other field or ISA bit changes. */
            program.addrs_ab.rs2 = 0;
        }
    } else if (!pr->resident_b || !pr->index_valid ||
        pr->index.b_sp_base != p->b_sp_base || pr->index.j_tiles != d->j_tiles ||
        pr->index.source_k_rows != d->k_tiles * POTAL_HW_DIM ||
        pr->index.active_rows != l->k_tiles * POTAL_HW_DIM - l->pad_k ||
        (s->index_open && s->index_pair == reference.pair) ||
        pl->a_offset_tiles < d->main_i_tiles * d->k_tiles) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t block = 0; block < l->k_tiles / l->passes_per_block; ++block) {
        for (uint32_t j = 0; j < l->j_tiles; ++j) {
            const uint32_t slot = l->scale_base + block * l->j_tiles + j;
            const uint32_t cols = POTAL_HW_DIM - (j + 1 == l->j_tiles ? l->pad_j : 0);
            const uint64_t mask = UINT64_MAX >> (64 - cols);
            if (s->slot_generation[slot] != l->scale_generation ||
                (s->slot_columns[slot] & mask) != mask)
                return POTAL_ERR_INVALID_ARGUMENT;
        }
    }
    if (s->outstanding_loops == UINT32_MAX || pr->outstanding == UINT32_MAX)
        return POTAL_ERR_CAPACITY;
    if (!paired_can_issue(s)) return POTAL_ERR_NO_HARDWARE;
    h->program_pending = 1;
    s->outstanding_loops++;
    pr->outstanding++;
    if (!ordinal && !p->status.prefetched) p->status.loading = 1;
    status = paired_issue_program(s, &program, ordinal ? "h2_residual" : "h2_main");
    if (status != POTAL_OK) return status;
    p->status.readers_issued++;
    h->acc_written |= 1u << (ordinal ? POTAL_PAIRED_RESIDUAL : POTAL_PAIRED_MAIN);
    if (!ordinal) {
        p->load_issued = 1;
        pr->resident_b = 1; /* ordered Residual may follow; this is not publication */
        pr->b_sp_base = p->b_sp_base;
        pr->b_j_tiles = d->j_tiles;
        pr->b_k_rows = d->k_tiles * POTAL_HW_DIM;
        pr->main_a_tiles = d->main_i_tiles * d->k_tiles;
        pr->index_valid = 0;
    }
    return POTAL_OK;
}

potal_status potal_paired_h2_panel_get(potal_paired_session *s, potal_h2_ref reference) {
    const potal_status status = h2_ready(s);
    if (status != POTAL_OK) return status;
    const h2_panel_state *p = h2_find(s, reference);
    if (!p) return POTAL_ERR_INVALID_ARGUMENT;
    return p->status.readable ? POTAL_OK : POTAL_ERR_BUSY;
}

potal_status potal_paired_h2_panel_retire(potal_paired_session *s, potal_h2_ref reference) {
    const potal_status status = h2_ready(s);
    if (status != POTAL_OK) return status;
    h2_panel_state *p = h2_find(s, reference);
    if (!p) return POTAL_ERR_INVALID_ARGUMENT;
    if (!p->status.readable || p->status.readers_issued != p->declaration.readers ||
        p->status.readers_completed != p->declaration.readers || h2_pending(s) ||
        s->index_open || h2_live_scales(s))
        return POTAL_ERR_BUSY;
    memset(&s->pair[reference.pair], 0, sizeof(s->pair[reference.pair]));
    memset(p, 0, sizeof(*p)); /* No ACC clear: the caller owns the accumulator manifest. */
    return POTAL_OK;
}

potal_h2_status potal_paired_h2_query(const potal_paired_session *s) {
    potal_h2_status q;
    memset(&q, 0, sizeof(q));
    q.core = potal_paired_session_query(s);
    if (s && s->h2) {
        q = s->h2->status;
        q.core = potal_paired_session_query(s);
        for (uint32_t pair = 0; pair < POTAL_PAIRS; ++pair) {
            q.panel[pair] = s->h2->panel[pair].status;
            q.source_pending |= q.panel[pair].loading;
        }
    }
    return q;
}

static potal_status h2_fence(potal_paired_session *s) {
    const potal_status ready = h2_ready(s);
    if (ready != POTAL_OK) return ready; /* Failed recovery is cleanup, never a direct retry. */
    if (!paired_can_issue(s)) return POTAL_ERR_NO_HARDWARE;
    const potal_status status = paired_raw(s, POTAL_FUNCT_FENCE, 0, 0, 1);
    s->h2->status.last_drain = status;
    if (status != POTAL_OK) {
        s->h2->status.unknown_completion = 1;
        s->h2->status.quarantined = 1;
        return paired_fail(s, status, "fence");
    }
    paired_drained(s);
    return POTAL_OK;
}

static potal_status h2_cleanup(potal_paired_session *s) {
    h2_state *h = s->h2;
    if (h->in_call) return POTAL_ERR_BUSY;
    if (s->observation_in_call) return POTAL_ERR_BUSY;
    if (h->status.quarantined || h->status.unknown_completion) return s->error;
    if (h->status.cleaned) return h->status.cleanup_status;
    if (!s->failed || h->status.cleanup_attempted) return POTAL_ERR_BUSY;
    h->status.cleanup_attempted = 1;
    /* Always drain once, even a failed first CONFIG with zero old core counts. */
    potal_status status = paired_raw(s, POTAL_FUNCT_FENCE, 0, 0, 1);
    h->status.last_drain = status;
    if (status != POTAL_OK) goto quarantine;
    paired_drained(s); /* failed sessions never publish or complete declared readers */
    for (uint32_t slot = 0; slot < POTAL_SCALE_ENTRIES; ++slot) {
        for (uint32_t col = 0; col < POTAL_HW_DIM; ++col) {
            if (!(s->slot_columns[slot] & (UINT64_C(1) << col))) continue;
            uint64_t key;
            (void)potal_scale_key_encode(slot, s->slot_generation[slot], col, &key);
            status = paired_raw(s, POTAL_FUNCT_SCALE_RELEASE, key, 0, 1);
            if (status != POTAL_OK) goto quarantine;
            s->slot_columns[slot] &= ~(UINT64_C(1) << col);
            paired_drained(s);
        }
    }
    s->index_open = 0;
    memset(s->pair, 0, sizeof(s->pair));
    memset(h->panel, 0, sizeof(h->panel));
    h->acc_written = 0;
    h->status.cleanup_status = POTAL_OK;
    h->status.cleaned = 1;
    return POTAL_OK;

quarantine:
    h->status.cleanup_status = status;
    h->status.unknown_completion = 1; /* independent of the first error/stage */
    h->status.quarantined = 1;
    return status; /* s->error remains the original failure */
}

potal_status potal_paired_h2_close(potal_paired_session **session) {
    if (!session || !*session || !(*session)->h2) return POTAL_ERR_INVALID_ARGUMENT;
    potal_paired_session *s = *session;
    h2_state *h = s->h2;
    if (h->in_call) return POTAL_ERR_BUSY;
    if (s->observation_in_call || (s->observation_active && !s->failed)) return POTAL_ERR_BUSY;
    if (h->status.quarantined || h->status.unknown_completion || (s->failed && !h->status.cleaned))
        return s->error;
    if (h2_pending(s) || s->index_open || h2_live_scales(s)) return POTAL_ERR_BUSY;
    for (uint32_t pair = 0; pair < POTAL_PAIRS; ++pair)
        if (h->panel[pair].status.declared) return POTAL_ERR_BUSY;
    const potal_status status = s->failed ? s->error : POTAL_OK;
    *session = NULL;
    free(h);
    free(s);
    paired_owner = 0;
    return status;
}

static potal_status observation_counter(potal_paired_session *s, uint64_t rs1, uint64_t *value) {
    const potal_paired_cmd_record record = {POTAL_FUNCT_COUNTER, 1, rs1, 0};
    potal_status status;
    ++s->commands;
    if (s->sink.cmd) {
        s->observation_in_call = 1;
        status = s->observation_counter(s->sink.ctx, &record, value);
        s->observation_in_call = 0;
    } else {
#if defined(__riscv)
        *value = potal_rocc_counter(rs1, 0);
        status = POTAL_OK;
#else
        (void)value;
        status = POTAL_ERR_NO_HARDWARE;
#endif
    }
    return status == POTAL_OK ? status : paired_fail(s, status, "observation_counter");
}

potal_status potal_paired_observation_begin(potal_paired_session *s,
    potal_observation_counter_fn counter) {
    potal_status status = paired_ready(s);
    if (status != POTAL_OK) return status;
    if (s->resident || s->commands || s->observation_active) return POTAL_ERR_BUSY;
    if (POTAL_HW_DIM != 32) return POTAL_ERR_UNSUPPORTED;
    if (!!s->sink.cmd != !!counter) return POTAL_ERR_INVALID_ARGUMENT;
    if (!paired_can_issue(s)) return POTAL_ERR_NO_HARDWARE;
    s->observation_counter = counter;
    uint64_t value = 0;
    status = observation_counter(s, UINT64_C(0xA), &value);
    if (status == POTAL_OK) status = observation_counter(s, POTAL_PREFETCH_OBSERVATION_TAG, &value);
    if (status != POTAL_OK) return status;
    if (value != POTAL_PREFETCH_OBSERVATION_CAPABILITY) return POTAL_ERR_UNSUPPORTED;
    for (uint32_t i = 0; i < 7; ++i) {
        status = observation_counter(s, ((uint64_t)(45 + i) << 12) | (i << 4) | 8, &value);
        if (status != POTAL_OK) return status;
    }
    status = observation_counter(s, 1, &value);
    if (status == POTAL_OK) s->observation_active = 1;
    return status;
}

potal_status potal_paired_observation_end(potal_paired_session *s, uint64_t cycles[7]) {
    potal_status status = paired_ready(s);
    if (status != POTAL_OK) return status;
    if (!cycles || !s->observation_active) return POTAL_ERR_INVALID_ARGUMENT;
    if (s->outstanding_loops || s->outstanding_readbacks || s->outstanding_acc_updates ||
        s->index_open || h2_live_scales(s) || (s->h2 && h2_pending(s))) return POTAL_ERR_BUSY;
    uint64_t values[7], ignored = 0;
    status = observation_counter(s, 4, &ignored);
    for (uint32_t i = 0; status == POTAL_OK && i < 7; ++i)
        status = observation_counter(s, i << 4, &values[i]);
    if (status == POTAL_OK) status = observation_counter(s, 2, &ignored);
    if (status != POTAL_OK) return status;
    memcpy(cycles, values, sizeof(values));
    s->observation_active = 0;
    return POTAL_OK;
}

/* ---- Owning resident mode ------------------------------------------------ */

/* Single-threaded issuer identity, independent of the eight-bit wire generation. */
static uint64_t resident_ws_owner_sequence;

static int resident_ws_same(potal_resident_ws_ref a, potal_resident_ws_ref b) {
    return a.owner && a.generation && a.owner == b.owner && a.generation == b.generation;
}

static potal_status resident_ws_ready(const potal_paired_session *s, int allow_pending) {
    if (!s || !s->resident) return POTAL_ERR_INVALID_ARGUMENT;
    const resident_ws_state *r = s->resident;
    if (r->in_call) return POTAL_ERR_BUSY;
    if (s->failed) return s->error;
    if (r->status.state != POTAL_RESIDENT_WS_READY) return POTAL_ERR_BUSY;
    if (!allow_pending && (r->status.pending || r->status.control_pending)) return POTAL_ERR_BUSY;
    return POTAL_OK;
}

static potal_status resident_ws_token(resident_ws_state *r, potal_resident_ws_ref *out) {
    if (r->sequence == UINT64_MAX) return POTAL_ERR_CAPACITY;
    *out = (potal_resident_ws_ref){r->owner, ++r->sequence};
    return POTAL_OK;
}

static int resident_ws_find(const resident_ws_state *r, potal_resident_ws_ref token) {
    for (uint32_t p = 0; p < 2; ++p)
        if (r->pair[p].readable && resident_ws_same(r->pair[p].token, token)) return (int)p;
    return -1;
}

static potal_status resident_ws_failed(potal_paired_session *s, potal_status status,
    const char *stage, int quarantine) {
    (void)paired_fail(s, status, stage);
    s->resident->status.state = quarantine ? POTAL_RESIDENT_WS_QUARANTINED : POTAL_RESIDENT_WS_FAILED;
    return status;
}

/* Only paired_drained calls this: a matching, successful global completion.
 * Release is allDrained too; CPU ordering must precede publication there.
 * Global single-pending admission keeps every staged epoch/token immutable.
 */
static void resident_ws_completed(potal_paired_session *s) {
    resident_ws_state *r = s->resident;
#if defined(__riscv)
    potal_cpu_fence_rw();
#endif
    int publish = !s->failed && r->issue_complete;
    if (r->kind == RESIDENT_WS_LOAD) {
        publish = publish && resident_ws_same(r->pair[r->pending_pair].token, r->pending_refs[0]);
    } else if (r->kind == RESIDENT_WS_COMPUTE || r->kind == RESIDENT_WS_READBACK) {
        for (uint32_t j = 0; j < r->pending_count; ++j)
            publish = publish && resident_ws_same(
                r->epoch[r->pending_bank][r->pending_tile + j].token, r->pending_refs[j]);
    }
    r->status.last_drain = POTAL_OK;
    if (r->kind == RESIDENT_WS_LOAD) {
        resident_ws_pair *p = &r->pair[r->pending_pair];
        p->loading = 0;
        p->readable = publish;
        if (publish) {
            paired_pair_state *c = &s->pair[r->pending_pair];
            c->resident_b = 1;
            c->b_sp_base = resident_ws_b_base(&p->load);
            c->b_j_tiles = resident_ws_tiles(p->load.j_columns);
            c->b_k_rows = p->load.k_rows;
            c->main_a_tiles = p->load.main_a_tiles;
            c->index_valid = 0;
        }
    } else if (r->kind == RESIDENT_WS_COMPUTE) {
        for (uint32_t j = 0; j < r->pending_count; ++j) {
            resident_ws_epoch *e = &r->epoch[r->pending_bank][r->pending_tile + j];
            e->initialized = publish;
            e->final = publish && r->pending_last;
            if (publish) e->next = r->pending_next;
            else e->token = (potal_resident_ws_ref){0, 0};
        }
    } else if (r->kind == RESIDENT_WS_READBACK) {
        r->receipt[r->pending_bank].ready = publish;
    }
    r->kind = RESIDENT_WS_NONE;
    r->issue_complete = 0;
    r->status.pending = 0;
    r->status.control_pending = 0;
}

/* The counter path deliberately does not truncate a response to potal_status.
 * There is no public raw issuing escape, and only the two frozen commands
 * reach this private function. Trace selection remains the open-time cache.
 */
static potal_status resident_ws_counter(potal_paired_session *s, uint64_t rs1, uint64_t *value) {
    const potal_paired_cmd_record record = {POTAL_FUNCT_COUNTER, 1, rs1, 0};
    s->commands++;
#if defined(__riscv) || defined(POTAL_TEST_RAW_TRACE)
    if (s->trace_enabled) {
        fprintf(stderr, "[POTAL_RAW] ENTER seq=%llu funct=126 xd=1 rs1=0x%016llx rs2=0x0000000000000000\n",
            (unsigned long long)s->commands, (unsigned long long)rs1);
        fflush(stderr);
    }
#endif
    potal_status status;
    if (s->sink.cmd) {
        status = s->resident->counter(s->sink.ctx, &record, value);
    } else {
#if defined(__riscv)
        *value = potal_rocc_counter(rs1, 0);
        status = POTAL_OK;
#else
        (void)value;
        status = POTAL_ERR_NO_HARDWARE;
#endif
    }
#if defined(__riscv) || defined(POTAL_TEST_RAW_TRACE)
    if (s->trace_enabled) {
        fprintf(stderr, "[POTAL_RAW] EXIT seq=%llu funct=126 status=%d response=0x%016llx\n",
            (unsigned long long)s->commands, (int)status,
            (unsigned long long)(status == POTAL_OK ? *value : 0));
        fflush(stderr);
    }
#endif
    return status;
}

static void resident_ws_free(potal_paired_session *s) {
    resident_ws_state *r = s->resident;
    for (uint32_t b = 0; b < 2; ++b)
        for (uint32_t t = 0; t < 32; ++t)
            free(r->epoch[b][t].fragments);
    for (uint32_t i = 0; i < 6; ++i) {
#if defined(__riscv)
        if (!r->status.software_only && r->pool[i]) (void)munlock(r->pool[i], POTAL_RESIDENT_WS_POOL_BYTES);
#endif
        free(r->pool[i]);
    }
    if (r->destroy_sink) r->destroy_sink(s->sink.ctx);
    free(r);
    free(s);
    paired_owner = 0;
}

potal_status potal_paired_resident_open(const potal_resident_ws_options *options,
    potal_paired_session **out) {
    if (!out) return POTAL_ERR_INVALID_ARGUMENT;
    *out = NULL;
    if (options && (!options->sink.cmd || !options->sink.ctx || !options->destroy_sink || !options->counter))
        return POTAL_ERR_INVALID_ARGUMENT;
    if (!resident_ws_geometry()) return POTAL_ERR_UNSUPPORTED;
#if !defined(__riscv)
    if (!options) return POTAL_ERR_NO_HARDWARE;
#endif
    if (resident_ws_owner_sequence == UINT64_MAX) return POTAL_ERR_CAPACITY;
    potal_paired_session *s = NULL;
    potal_status status = potal_paired_session_open(options ? &options->sink : NULL, &s);
    if (status != POTAL_OK) return status;
    resident_ws_state *r = calloc(1, sizeof(*r));
    if (!r) {
        (void)potal_paired_session_close(s); /* No resident state or command yet. */
        return POTAL_ERR_CAPACITY;
    }
    s->resident = r;
    paired_owner = 2; /* Existing owner, resident mode; not another lock. */
    r->in_call = 1;
    r->status.software_only = options != NULL;
    r->counter = options ? options->counter : NULL;
    for (uint32_t i = 0; i < 6; ++i) {
        r->pool[i] = aligned_alloc(64, POTAL_RESIDENT_WS_POOL_BYTES);
        if (!r->pool[i]) { status = POTAL_ERR_CAPACITY; break; }
        memset(r->pool[i], 0, POTAL_RESIDENT_WS_POOL_BYTES);
#if defined(__riscv)
        if (!options && mlock(r->pool[i], POTAL_RESIDENT_WS_POOL_BYTES) != 0) {
            status = POTAL_ERR_CAPACITY;
            break;
        }
#endif
    }
    uint64_t value = 0;
    if (status == POTAL_OK) status = resident_ws_counter(s, UINT64_C(0xA), &value);
    if (status == POTAL_OK) status = resident_ws_counter(s, POTAL_RESIDENT_WS_COUNTER_TAG, &value);
    if (status == POTAL_OK && value != POTAL_RESIDENT_WS_CAPABILITY) status = POTAL_ERR_UNSUPPORTED;
    if (status != POTAL_OK) {
        /* Only counter configuration could have changed; no extension or DMA.
         * Failed open never adopts the caller's software context. */
        resident_ws_free(s);
        return status;
    }
    r->owner = ++resident_ws_owner_sequence;
    r->status.capability_matched = 1;
    r->status.owned_bytes = UINT64_C(6) * POTAL_RESIDENT_WS_POOL_BYTES;
    r->status.state = POTAL_RESIDENT_WS_NEW;
    r->destroy_sink = options ? options->destroy_sink : NULL;
    r->in_call = 0;
    *out = s;
    return POTAL_OK;
}

potal_status potal_paired_resident_prepare(potal_paired_session *s) {
    if (!s || !s->resident) return POTAL_ERR_INVALID_ARGUMENT;
    resident_ws_state *r = s->resident;
    if (r->in_call) return POTAL_ERR_BUSY;
    if (s->failed) return s->error;
    if (r->status.state != POTAL_RESIDENT_WS_NEW || r->status.control_pending)
        return POTAL_ERR_BUSY;
    r->status.control_pending = 1;
    r->in_call = 1;
    const potal_status status = paired_raw(s, POTAL_FUNCT_FLUSH, 0, 0, 0);
    r->in_call = 0;
    if (status != POTAL_OK) return resident_ws_failed(s, status, "resident_prepare", 0);
    return resident_ws_fence(s);
}

static potal_status resident_ws_fence(potal_paired_session *s) {
    resident_ws_state *r = s->resident;
    if (r->in_call) return POTAL_ERR_BUSY;
    if (s->failed) return s->error; /* Recovery is explicit cleanup, never a second direct Fence. */
    if (r->status.state != POTAL_RESIDENT_WS_READY &&
        !(r->status.state == POTAL_RESIDENT_WS_NEW && r->status.control_pending))
        return POTAL_ERR_BUSY;
    r->in_call = 1;
    const potal_status status = paired_raw(s, POTAL_FUNCT_FENCE, 0, 0, 1);
    r->in_call = 0;
    r->status.last_drain = status;
    if (status != POTAL_OK) {
        r->status.unknown_fence = 1;
        return resident_ws_failed(s, status, "fence", 1);
    }
    paired_drained(s);
    r->status.state = POTAL_RESIDENT_WS_READY;
    return POTAL_OK;
}

static potal_status resident_ws_release(potal_paired_session *s, uint32_t slot,
    uint32_t generation, uint32_t column) {
    potal_status status = resident_ws_ready(s, 1);
    if (status != POTAL_OK) return status;
    uint64_t key;
    if (potal_scale_key_encode(slot, generation, column, &key) != POTAL_OK ||
        !(s->slot_columns[slot] & (UINT64_C(1) << column)) || s->slot_generation[slot] != generation)
        return POTAL_ERR_INVALID_ARGUMENT;
    resident_ws_state *r = s->resident;
    r->in_call = 1;
    status = paired_scale_release_core(s, slot, generation, column);
    r->in_call = 0;
    if (status != POTAL_OK) return resident_ws_failed(s, status, "scale_release", 1);
    if (!s->slot_columns[slot]) memset(&r->scale[slot], 0, sizeof(r->scale[slot]));
    return POTAL_OK;
}

static potal_status resident_ws_cleanup(potal_paired_session *s) {
    resident_ws_state *r = s->resident;
    if (r->in_call) return POTAL_ERR_BUSY;
    if (r->status.state == POTAL_RESIDENT_WS_CLEANED) return r->status.cleanup_status;
    if (r->status.state == POTAL_RESIDENT_WS_QUARANTINED || r->status.unknown_fence)
        return s->error;
    if (r->status.state != POTAL_RESIDENT_WS_FAILED || r->status.cleanup_attempted)
        return POTAL_ERR_BUSY;
    r->status.cleanup_attempted = 1;
    r->in_call = 1;
    /* Even a failed first CONFIG/LOOP/MVOUT can have effects without core counters. */
    potal_status status = paired_raw(s, POTAL_FUNCT_FENCE, 0, 0, 1);
    r->status.last_drain = status;
    r->status.cleanup_status = status;
    if (status != POTAL_OK) {
        r->in_call = 0;
        r->status.unknown_fence = 1; /* independent of the preserved first error stage */
        return resident_ws_failed(s, status, "fence", 1);
    }
    paired_drained(s); /* s->failed prevents every publication. */
    for (uint32_t slot = 0; slot < 256; ++slot) {
        for (uint32_t c = 0; c < 16; ++c) {
            if (!(s->slot_columns[slot] & (UINT64_C(1) << c))) continue;
            uint64_t key;
            (void)potal_scale_key_encode(slot, s->slot_generation[slot], c, &key);
            status = paired_raw(s, POTAL_FUNCT_SCALE_RELEASE, key, 0, 1);
            if (status != POTAL_OK) {
                r->in_call = 0;
                r->status.cleanup_status = status;
                return resident_ws_failed(s, status, "cleanup_release", 1);
            }
            s->slot_columns[slot] &= ~(UINT64_C(1) << c);
            paired_drained(s);
        }
    }
    s->index_open = 0;
    memset(s->pair, 0, sizeof(s->pair));
    memset(r->pair, 0, sizeof(r->pair));
    memset(r->scale, 0, sizeof(r->scale));
    memset(r->index, 0, sizeof(r->index));
    memset(r->receipt, 0, sizeof(r->receipt));
    for (uint32_t b = 0; b < 2; ++b) for (uint32_t t = 0; t < 32; ++t) {
        r->epoch[b][t].token = (potal_resident_ws_ref){0, 0};
        r->epoch[b][t].initialized = 0;
    }
    r->status.cleanup_status = POTAL_OK;
    r->status.state = POTAL_RESIDENT_WS_CLEANED;
    r->in_call = 0;
    return POTAL_OK;
}

potal_status potal_paired_resident_buffer_view(potal_paired_session *s,
    potal_resident_ws_buffer which, void **data, size_t *bytes) {
    const potal_status status = resident_ws_ready(s, 0);
    if (status != POTAL_OK) return status;
    if ((unsigned)which > POTAL_RESIDENT_WS_RESIDUAL_A || !data || !bytes)
        return POTAL_ERR_INVALID_ARGUMENT;
    *data = s->resident->pool[which];
    *bytes = POTAL_RESIDENT_WS_POOL_BYTES;
    return POTAL_OK;
}

potal_resident_ws_status potal_paired_resident_query(const potal_paired_session *s) {
    potal_resident_ws_status q;
    memset(&q, 0, sizeof(q));
    if (s && s->resident) {
        const resident_ws_state *r = s->resident;
        q = r->status;
        q.core = potal_paired_session_query(s);
        for (uint32_t p = 0; p < 2; ++p) {
            q.residents[p] = r->pair[p].readable ? r->pair[p].token : (potal_resident_ws_ref){0, 0};
            q.readable[p] = (uint32_t)r->pair[p].readable;
            q.loading[p] = (uint32_t)r->pair[p].loading;
        }
    }
    return q;
}

potal_status potal_paired_resident_close(potal_paired_session **session) {
    if (!session || !*session || !(*session)->resident) return POTAL_ERR_INVALID_ARGUMENT;
    potal_paired_session *s = *session;
    resident_ws_state *r = s->resident;
    if (r->in_call) return POTAL_ERR_BUSY;
    if (r->status.state == POTAL_RESIDENT_WS_FAILED || r->status.state == POTAL_RESIDENT_WS_QUARANTINED)
        return s->error;
    if (r->status.pending || r->status.control_pending || s->index_open ||
        s->outstanding_loops || s->outstanding_readbacks || s->outstanding_acc_updates)
        return POTAL_ERR_BUSY;
    for (uint32_t p = 0; p < 2; ++p)
        if (r->pair[p].readable || r->pair[p].loading) return POTAL_ERR_BUSY;
    for (uint32_t slot = 0; slot < 256; ++slot)
        if (s->slot_columns[slot]) return POTAL_ERR_BUSY;
    for (uint32_t bank = 0; bank < 2; ++bank)
        for (uint32_t tile = 0; tile < 32; ++tile)
            if (r->epoch[bank][tile].initialized && !r->epoch[bank][tile].final)
                return POTAL_ERR_BUSY;
    const potal_status status = s->failed ? s->error : POTAL_OK;
    r->in_call = 1;
    *session = NULL;
    resident_ws_free(s);
    return status;
}

potal_status potal_paired_resident_load_issue(potal_paired_session *s,
    const potal_resident_ws_load *load) {
    potal_status status = resident_ws_ready(s, 0);
    if (status != POTAL_OK) return status;
    status = potal_resident_ws_load_check(load);
    if (status != POTAL_OK) return status;
    resident_ws_state *r = s->resident;
    const potal_resident_ws_load d = *load;
    if (r->pair[d.pair].readable || r->pair[d.pair].loading) return POTAL_ERR_BUSY;
    potal_resident_ws_ref token;
    status = resident_ws_token(r, &token);
    if (status != POTAL_OK) return status;
    r->pair[d.pair] = (resident_ws_pair){d, token, 1, 0};
    memset(&r->index[d.pair], 0, sizeof(r->index[d.pair]));
    s->pair[d.pair].resident_b = 0;
    s->pair[d.pair].index_valid = 0;
    r->kind = RESIDENT_WS_LOAD;
    r->pending_pair = d.pair;
    r->pending_refs[0] = token;
    r->status.pending = 1; /* Premarked before CONFIG and any possible B DMA. */
    r->issue_complete = 0;
    r->in_call = 1;
#if defined(__riscv)
    potal_cpu_fence_rw();
#endif
    status = paired_raw(s, POTAL_FUNCT_CONFIG, config_ld_rs1(0), d.source_stride, 0);
    const uint32_t nk = resident_ws_tiles(d.k_rows), nj = resident_ws_tiles(d.j_columns);
    const uint32_t base = resident_ws_b_base(&d);
    for (uint32_t k = 0; status == POTAL_OK && k < nk; ++k) {
        for (uint32_t j = 0; status == POTAL_OK && j < nj; ++j) {
            const uint32_t nr = d.k_rows - k * 16 < 16 ? d.k_rows - k * 16 : 16;
            const uint32_t nc = d.j_columns - j * 16 < 16 ? d.j_columns - j * 16 : 16;
            const uint64_t source = (uint64_t)(uintptr_t)r->pool[d.pair] +
                (uint64_t)k * 16 * d.source_stride + j * 16;
            const uint32_t row = base + (k * nj + j) * 16;
            r->status.attempted_b_mvins++;
            status = paired_raw(s, POTAL_FUNCT_MVIN, source,
                ((uint64_t)nr << 48) | ((uint64_t)nc << 32) | row, 0);
        }
    }
    r->in_call = 0;
    if (status != POTAL_OK) return resident_ws_failed(s, status, "resident_b_load", 0);
    r->issue_complete = 1;
    return POTAL_OK;
}

potal_status potal_paired_resident_get(potal_paired_session *s, uint32_t pair,
    potal_resident_ws_ref *resident) {
    const potal_status status = resident_ws_ready(s, 0);
    if (status != POTAL_OK) return status;
    if (pair >= 2 || !resident) return POTAL_ERR_INVALID_ARGUMENT;
    if (!s->resident->pair[pair].readable) return POTAL_ERR_BUSY;
    *resident = s->resident->pair[pair].token;
    return POTAL_OK;
}

static int resident_ws_cpu_span(const void *p, uint64_t bytes) {
    return p && bytes && bytes <= SIZE_MAX && (uintptr_t)p <= UINTPTR_MAX - bytes;
}

potal_status potal_paired_resident_scale_program(potal_paired_session *s,
    const potal_resident_ws_scales *d, potal_resident_ws_ref *scale) {
    potal_status status = resident_ws_ready(s, 0);
    if (status != POTAL_OK) return status;
    if (!d || !scale || !d->partials || !d->j_columns || d->generation > 255 ||
        d->carrier_stride < d->j_columns) return POTAL_ERR_INVALID_ARGUMENT;
    resident_ws_state *r = s->resident;
    const int pair = resident_ws_find(r, d->resident);
    if (pair < 0) return POTAL_ERR_INVALID_ARGUMENT;
    const potal_resident_ws_load *l = &r->pair[pair].load;
    const uint32_t nj = resident_ws_tiles(d->j_columns);
    if (d->base >= 256 || (uint64_t)d->partials * nj > 256 - d->base)
        return POTAL_ERR_CAPACITY;
    if (d->j_begin < l->j_begin || d->j_columns > l->j_columns ||
        d->j_begin - l->j_begin > l->j_columns - d->j_columns ||
        (d->j_begin - l->j_begin) % 16 ||
        !resident_ws_cpu_span(d->original_blocks, (uint64_t)d->partials * sizeof(uint64_t)) ||
        !resident_ws_cpu_span(d->carriers, matrix_extent(d->partials, d->j_columns,
                                                         d->carrier_stride, sizeof(uint32_t))))
        return POTAL_ERR_INVALID_ARGUMENT;
    uint32_t values[256][16];
    resident_ws_scale bindings[256];
    const uint32_t slots = d->partials * nj, base = d->base, generation = d->generation;
    const uint32_t columns = d->j_columns;
    for (uint32_t p = 0; p < d->partials; ++p) {
        const uint64_t block = d->original_blocks[p];
        if (block < l->k_begin / 32 || block > (l->k_begin + l->k_rows - 1) / 32)
            return POTAL_ERR_INVALID_ARGUMENT;
        for (uint32_t j = 0; j < nj; ++j) {
            const uint32_t local = p * nj + j, slot = base + local;
            if (s->slot_columns[slot]) return POTAL_ERR_CAPACITY;
            bindings[local].resident = d->resident;
            bindings[local].block = block;
            bindings[local].column = d->j_begin + j * 16;
            const uint32_t nc = columns - j * 16 < 16 ? columns - j * 16 : 16;
            for (uint32_t c = 0; c < nc; ++c) {
                const uint32_t v = d->carriers[(size_t)p * d->carrier_stride + j * 16 + c];
                if (!potal_carrier_is_valid(v)) return POTAL_ERR_INVALID_ARGUMENT;
                values[local][c] = v;
            }
        }
    }
    potal_resident_ws_ref token;
    status = resident_ws_token(r, &token);
    if (status != POTAL_OK) return status;
    r->status.control_pending = 1;
    r->in_call = 1;
    for (uint32_t i = 0; status == POTAL_OK && i < slots; ++i) {
        const uint32_t j = i % nj;
        const uint32_t nc = columns - j * 16 < 16 ? columns - j * 16 : 16;
        bindings[i].token = token;
        r->scale[base + i] = bindings[i];
        for (uint32_t c = 0; status == POTAL_OK && c < nc; ++c)
            status = paired_scale_write_core(s, base + i, generation, c, values[i][c]);
    }
    r->in_call = 0;
    if (status != POTAL_OK)
        return resident_ws_failed(s, status, "resident_scale", 1); /* Acceptance may be untracked. */
    r->status.control_pending = 0;
    *scale = token;
    return POTAL_OK;
}

potal_status potal_paired_resident_index_program(potal_paired_session *s,
    const potal_resident_ws_indices *d, potal_resident_ws_ref *index) {
    potal_status status = resident_ws_ready(s, 0);
    if (status != POTAL_OK) return status;
    if (!d || !index || !d->count) return POTAL_ERR_INVALID_ARGUMENT;
    if (d->count > 32) return POTAL_ERR_CAPACITY;
    if (!resident_ws_cpu_span(d->original_k, (uint64_t)d->count * sizeof(uint64_t)))
        return POTAL_ERR_INVALID_ARGUMENT;
    resident_ws_state *r = s->resident;
    const int pair = resident_ws_find(r, d->resident);
    if (pair < 0) return POTAL_ERR_INVALID_ARGUMENT;
    if (s->index_open) return POTAL_ERR_BUSY;
    const potal_resident_ws_load *l = &r->pair[pair].load;
    resident_ws_index next;
    memset(&next, 0, sizeof(next));
    next.resident = d->resident;
    next.count = d->count;
    const potal_index_config config = {
        (uint32_t)pair, resident_ws_b_base(l), resident_ws_tiles(l->j_columns), l->k_rows, d->count
    };
    status = potal_index_config_check(&config);
    if (status != POTAL_OK) return status;
    for (uint32_t i = 0; i < d->count; ++i) {
        next.original[i] = d->original_k[i];
        if (next.original[i] < l->k_begin || next.original[i] - l->k_begin >= l->k_rows)
            return POTAL_ERR_INVALID_ARGUMENT;
        const potal_index_write w = {i, (uint32_t)(next.original[i] - l->k_begin)};
        status = potal_index_write_check(&config, &w);
        if (status != POTAL_OK) return status;
    }
    status = resident_ws_token(r, &next.token);
    if (status != POTAL_OK) return status;
    memset(&r->index[pair], 0, sizeof(r->index[pair]));
    s->pair[pair].index_valid = 0;
    r->status.control_pending = 1; /* Partial Index acceptance also requires cleanup drain. */
    r->in_call = 1;
    status = paired_index_config_core(s, &config);
    for (uint32_t i = 0; status == POTAL_OK && i < next.count; ++i) {
        const potal_index_write w = {i, (uint32_t)(next.original[i] - l->k_begin)};
        status = paired_index_write_core(s, &w);
    }
    r->in_call = 0;
    if (status != POTAL_OK) return resident_ws_failed(s, status, "resident_index", 0);
    r->index[pair] = next;
    r->status.control_pending = 0;
    *index = next.token;
    return POTAL_OK;
}

potal_status potal_paired_resident_retire(potal_paired_session *s, potal_resident_ws_ref resident) {
    const potal_status status = resident_ws_ready(s, 0);
    if (status != POTAL_OK) return status;
    resident_ws_state *r = s->resident;
    const int pair = resident_ws_find(r, resident);
    if (pair < 0) return POTAL_ERR_INVALID_ARGUMENT;
    for (uint32_t slot = 0; slot < 256; ++slot)
        if (s->slot_columns[slot] && resident_ws_same(r->scale[slot].resident, resident))
            return POTAL_ERR_BUSY;
    memset(&r->pair[pair], 0, sizeof(r->pair[pair]));
    memset(&r->index[pair], 0, sizeof(r->index[pair]));
    memset(&s->pair[pair], 0, sizeof(s->pair[pair]));
    /* Physical ACC epochs retain values and identity independently of B. */
    return POTAL_OK;
}

potal_status potal_paired_resident_epoch_begin(potal_paired_session *s,
    const potal_resident_ws_epoch *d, potal_resident_ws_ref *reference) {
    potal_status status = resident_ws_ready(s, 0);
    if (status != POTAL_OK) return status;
    if (!d || !reference || d->path > 1 || d->acc_tile >= 32 ||
        !d->rows || d->rows > 16 || !d->columns || d->columns > 16 ||
        !d->output_id || !d->weight_id || !d->weight_generation ||
        d->original_i > UINT64_MAX - d->rows || d->original_j % 16 ||
        d->original_j > UINT64_MAX - d->columns ||
        d->logical_k_begin >= d->logical_k_end)
        return POTAL_ERR_INVALID_ARGUMENT;
    resident_ws_state *r = s->resident;
    resident_ws_epoch *old = &r->epoch[d->path][d->acc_tile];
    if (old->initialized && !old->final) return POTAL_ERR_BUSY;
    resident_ws_epoch next;
    memset(&next, 0, sizeof(next));
    next.declaration = *d;
    next.declaration.fragments = NULL;
    if (!d->path) {
        if (d->logical_k_begin % 16 || d->fragments || d->fragment_count)
            return POTAL_ERR_INVALID_ARGUMENT;
        const uint64_t length = d->logical_k_end - d->logical_k_begin;
        next.total = length / 16 + (length % 16 != 0);
    } else {
        if (!d->fragment_count || d->fragment_count > SIZE_MAX / sizeof(*d->fragments) ||
            !resident_ws_cpu_span(d->fragments, d->fragment_count * sizeof(*d->fragments)))
            return POTAL_ERR_INVALID_ARGUMENT;
        uint64_t previous = 0;
        int have_previous = 0;
        for (size_t p = 0; p < d->fragment_count; ++p) {
            const potal_resident_ws_fragment *f = &d->fragments[p];
            if (!f->count || f->count > 16) return POTAL_ERR_INVALID_ARGUMENT;
            for (uint32_t q = 0; q < f->count; ++q) {
                const uint64_t k = f->original_k[q];
                if (k < d->logical_k_begin || k >= d->logical_k_end ||
                    k / 32 != f->original_k[0] / 32 ||
                    (have_previous && k <= previous)) return POTAL_ERR_INVALID_ARGUMENT;
                previous = k;
                have_previous = 1;
            }
        }
        next.total = d->fragment_count;
    }
    if (r->sequence == UINT64_MAX) return POTAL_ERR_CAPACITY;
    if (d->path) {
        potal_resident_ws_fragment *copy = malloc(d->fragment_count * sizeof(*copy));
        if (!copy) return POTAL_ERR_CAPACITY;
        memcpy(copy, d->fragments, d->fragment_count * sizeof(*copy));
        next.fragments = copy;
        next.declaration.fragments = copy;
    }
    (void)resident_ws_token(r, &next.token); /* Capacity was checked before allocation/mutation. */
    free(old->fragments);
    *old = next;
    r->receipt[d->path].ready = 0;
    *reference = next.token;
    return POTAL_OK;
}

potal_status potal_paired_resident_run_check(const potal_paired_session *s,
    const potal_resident_ws_run *d) {
    potal_status status = resident_ws_ready(s, 0);
    if (status != POTAL_OK) return status;
    if (!d) return POTAL_ERR_INVALID_ARGUMENT;
    const resident_ws_state *r = s->resident;
    const int pair = resident_ws_find(r, d->resident);
    if (pair < 0 || d->path > 1) return POTAL_ERR_INVALID_ARGUMENT;
    const potal_resident_ws_load *l = &r->pair[pair].load;
    const int8_t *activation = r->pool[2 + d->path];
    status = potal_resident_ws_run_check(l, d, activation, POTAL_RESIDENT_WS_POOL_BYTES);
    if (status != POTAL_OK) return status;
    const uint32_t nk = resident_ws_tiles(d->k_rows), nj = resident_ws_tiles(d->j_columns);
    const resident_ws_index *ix = &r->index[pair];
    if (d->indexed) {
        const paired_pair_state *p = &s->pair[pair];
        if (!resident_ws_same(ix->token, d->index) || !resident_ws_same(ix->resident, d->resident) ||
            ix->count != d->k_rows || s->index_open || !p->index_valid ||
            p->index.b_sp_base != resident_ws_b_base(l) ||
            p->index.j_tiles != resident_ws_tiles(l->j_columns) ||
            p->index.source_k_rows != l->k_rows || p->index.active_rows != d->k_rows ||
            ix->original[0] < l->k_begin || ix->original[0] - l->k_begin != d->k_offset)
            return POTAL_ERR_INVALID_ARGUMENT;
    }
    /* Every destination is checked before ANY token, pending flag or command
     * changes. One stale lane rejects the entire multi-J operation. */
    for (uint32_t j = 0; j < nj; ++j) {
        const resident_ws_epoch *e = &r->epoch[d->path][d->acc_tile + j];
        const potal_resident_ws_epoch *v = &e->declaration;
        const uint32_t cols = d->j_columns - j * 16 < 16 ? d->j_columns - j * 16 : 16;
        if (!resident_ws_same(e->token, d->epochs[j]) || e->final ||
            v->path != d->path || v->acc_tile != d->acc_tile + j ||
            v->rows != d->i_rows || v->columns != cols ||
            v->output_id != d->output_id || v->original_i != d->original_i ||
            v->original_j != l->j_begin + d->j_offset + j * 16 ||
            v->weight_id != l->weight_id || v->weight_generation != l->weight_generation ||
            e->next != d->sequence || d->sequence > e->total ||
            nk > e->total - d->sequence || d->last != (d->sequence + nk == e->total) ||
            (d->first ? e->initialized : !e->initialized))
            return POTAL_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t p = 0; p < nk; ++p) {
        const uint32_t valid = d->indexed ? d->fragment_counts[p] :
            (d->k_rows - p * 16 < 16 ? d->k_rows - p * 16 : 16);
        const uint64_t first = d->indexed ? ix->original[p * 16] :
            l->k_begin + d->k_offset + p * 16;
        const uint64_t block = first / 32;
        for (uint32_t q = 0; q < valid; ++q) {
            const uint64_t k = d->indexed ? ix->original[p * 16 + q] : first + q;
            if (k < l->k_begin || k - l->k_begin >= l->k_rows || k / 32 != block)
                return POTAL_ERR_INVALID_ARGUMENT;
        }
        if (d->indexed && p + 1 < nk) {
            /* All Index entries, including filler, were range-checked at
             * programming; every interior filler A byte must be zero. */
            for (uint32_t q = valid; q < 16; ++q)
                for (uint32_t row = 0; row < d->i_rows; ++row)
                    if (activation[d->a_offset + (size_t)row * d->a_stride + p * 16 + q])
                        return POTAL_ERR_INVALID_ARGUMENT;
        }
        for (uint32_t j = 0; j < nj; ++j) {
            const resident_ws_epoch *e = &r->epoch[d->path][d->acc_tile + j];
            const potal_resident_ws_epoch *v = &e->declaration;
            if (!d->path) {
                const uint64_t expected = v->logical_k_begin + (d->sequence + p) * 16;
                const uint64_t left = v->logical_k_end - expected;
                const uint32_t expected_count = left < 16 ? (uint32_t)left : 16;
                if (first != expected || valid != expected_count) return POTAL_ERR_INVALID_ARGUMENT;
                for (uint32_t q = 0; d->indexed && q < valid; ++q)
                    if (ix->original[p * 16 + q] != expected + q) return POTAL_ERR_INVALID_ARGUMENT;
            } else {
                const potal_resident_ws_fragment *f = &v->fragments[d->sequence + p];
                if (valid != f->count) return POTAL_ERR_INVALID_ARGUMENT;
                for (uint32_t q = 0; q < valid; ++q)
                    if (ix->original[p * 16 + q] != f->original_k[q]) return POTAL_ERR_INVALID_ARGUMENT;
            }
            const uint32_t slot = d->scale_base + p * nj + j;
            const resident_ws_scale *b = &r->scale[slot];
            const uint32_t cols = d->j_columns - j * 16 < 16 ? d->j_columns - j * 16 : 16;
            const uint64_t mask = (UINT64_C(1) << cols) - 1;
            if (!resident_ws_same(b->token, d->scale) || !resident_ws_same(b->resident, d->resident) ||
                b->block != block || b->column != l->j_begin + d->j_offset + j * 16 ||
                s->slot_generation[slot] != d->scale_generation ||
                (s->slot_columns[slot] & mask) != mask)
                return POTAL_ERR_INVALID_ARGUMENT;
        }
    }
    return POTAL_OK;
}

potal_status potal_paired_resident_issue(potal_paired_session *s, const potal_resident_ws_run *run) {
    potal_status status = potal_paired_resident_run_check(s, run);
    if (status != POTAL_OK) return status;
    const potal_resident_ws_run d = *run;
    resident_ws_state *r = s->resident;
    const uint32_t pair = (uint32_t)resident_ws_find(r, d.resident);
    potal_resident_ws_program p;
    status = potal_resident_ws_run_encode(&r->pair[pair].load, &d,
        r->pool[2 + d.path], POTAL_RESIDENT_WS_POOL_BYTES, &p);
    if (status != POTAL_OK) return status;
    r->kind = RESIDENT_WS_COMPUTE;
    r->pending_pair = pair;
    r->pending_bank = d.path;
    r->pending_tile = d.acc_tile;
    r->pending_count = resident_ws_tiles(d.j_columns);
    r->pending_next = d.sequence + resident_ws_tiles(d.k_rows);
    r->pending_last = d.last;
    memcpy(r->pending_refs, d.epochs, r->pending_count * sizeof(*d.epochs));
    r->receipt[d.path].ready = 0;
    r->status.pending = 1; /* Includes a partial baseline/view/run configuration. */
    r->issue_complete = 0;
    s->pair[pair].outstanding++;
    s->outstanding_loops++;
    r->in_call = 1;
    status = paired_issue_program_view(s, &p.loop, &p.view, "resident_loop_ws");
    r->in_call = 0;
    if (status != POTAL_OK) return resident_ws_failed(s, status, "resident_loop_ws", 0);
    r->issue_complete = 1;
    return POTAL_OK;
}

potal_status potal_paired_resident_readback_issue(potal_paired_session *s,
    const potal_resident_ws_readback *readback, potal_resident_ws_ref *receipt) {
    potal_status status = resident_ws_ready(s, 0);
    if (status != POTAL_OK) return status;
    if (!readback || !receipt || readback->bank > 1) return POTAL_ERR_INVALID_ARGUMENT;
    const potal_resident_ws_readback d = *readback;
    resident_ws_state *r = s->resident;
    potal_rocc_cmd config, commands[9];
    uint32_t n;
    status = potal_resident_ws_readback_encode(&d, r->pool[4 + d.bank],
        POTAL_RESIDENT_WS_POOL_BYTES, &config, commands, 9, &n);
    if (status != POTAL_OK) return status;
    for (uint32_t j = 0; j < n; ++j) {
        const resident_ws_epoch *e = &r->epoch[d.bank][d.acc_tile + j];
        const uint32_t cols = d.columns - j * 16 < 16 ? d.columns - j * 16 : 16;
        if (!resident_ws_same(e->token, d.epochs[j]) || !e->initialized ||
            e->next != d.sequences[j] || e->declaration.rows != d.rows ||
            e->declaration.columns != cols) return POTAL_ERR_INVALID_ARGUMENT;
    }
    potal_resident_ws_ref token;
    status = resident_ws_token(r, &token);
    if (status != POTAL_OK) return status;
    r->receipt[d.bank] = (resident_ws_receipt){token, d, 0};
    r->kind = RESIDENT_WS_READBACK;
    r->pending_bank = d.bank;
    r->pending_tile = d.acc_tile;
    r->pending_count = n;
    memcpy(r->pending_refs, d.epochs, n * sizeof(*d.epochs));
    r->status.pending = 1;
    r->issue_complete = 0;
    s->outstanding_readbacks++;
    r->in_call = 1;
    status = paired_raw(s, POTAL_FUNCT_CONFIG, config.rs1, config.rs2, 0);
    for (uint32_t j = 0; status == POTAL_OK && j < n; ++j)
        status = paired_raw(s, POTAL_FUNCT_MVOUT, commands[j].rs1, commands[j].rs2, 0);
    r->in_call = 0;
    if (status != POTAL_OK) return resident_ws_failed(s, status, "resident_readback", 0);
    r->issue_complete = 1;
    *receipt = token;
    return POTAL_OK;
}

potal_status potal_paired_resident_result(potal_paired_session *s, potal_resident_ws_ref receipt,
    const int32_t **data, size_t *stride) {
    const potal_status status = resident_ws_ready(s, 0);
    if (status != POTAL_OK) return status;
    if (!data || !stride) return POTAL_ERR_INVALID_ARGUMENT;
    const resident_ws_state *r = s->resident;
    for (uint32_t b = 0; b < 2; ++b) {
        const resident_ws_receipt *snapshot = &r->receipt[b];
        if (!snapshot->ready || !resident_ws_same(snapshot->token, receipt)) continue;
        const potal_resident_ws_readback *d = &snapshot->readback;
        for (uint32_t j = 0; j < resident_ws_tiles(d->columns); ++j) {
            const resident_ws_epoch *e = &r->epoch[b][d->acc_tile + j];
            if (!e->initialized || !resident_ws_same(e->token, d->epochs[j]) ||
                e->next != d->sequences[j]) return POTAL_ERR_INVALID_ARGUMENT;
        }
        *data = r->pool[4 + b];
        *stride = d->stride;
        return POTAL_OK;
    }
    return POTAL_ERR_INVALID_ARGUMENT;
}
