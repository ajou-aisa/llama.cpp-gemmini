/* PoTal minimal RoCC API. See include/potal.h for the contract. */

#include "potal.h"

#include "potal_hw_params.h"

#include <string.h>

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
#define VALIDATED_DIM      16u
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

    if (g->flags != 0 || POTAL_HW_DIM != VALIDATED_DIM ||
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

#if defined(__riscv)

/* Set by a GEMM issue, cleared by a successful POTAL_FENCE. */
static int gemm_in_flight;

static potal_status hw_status(uint64_t rd) {
    return rd <= POTAL_HW_BAD_COMMAND ? (potal_status)rd : POTAL_ERR_BAD_RESPONSE;
}

potal_status potal_scale_write(uint32_t slot, uint32_t generation, uint32_t column,
                               uint32_t carrier) {
    uint64_t rs1;
    if (potal_scale_key_encode(slot, generation, column, &rs1) != POTAL_OK ||
        !potal_carrier_is_valid(carrier)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    return hw_status(potal_rocc_scale_write(rs1, carrier));
}

potal_status potal_scale_release(uint32_t slot, uint32_t generation, uint32_t column) {
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
    uint64_t rs1;
    if (potal_scale_key_encode(slot, generation, column, &rs1) != POTAL_OK ||
        !potal_carrier_is_valid(carrier)) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    return POTAL_ERR_NO_HARDWARE;
}

potal_status potal_scale_release(uint32_t slot, uint32_t generation, uint32_t column) {
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

typedef struct paired_pair_state {
    int      resident_b;
    uint32_t b_sp_base, b_j_tiles, b_k_rows, main_a_tiles;
    int      index_valid;        /* a complete Index window is programmed on this buffer */
    potal_index_config index;    /* that window */
    uint32_t outstanding;        /* paired LOOP_WS on this pair since the last POTAL_OK fence */
} paired_pair_state;

struct potal_paired_session {
    potal_paired_sink sink;
    int               failed;
    potal_status      error;
    const char       *error_stage;
    uint32_t          outstanding_loops, outstanding_readbacks;
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

static potal_status paired_fail(potal_paired_session *s, potal_status status, const char *stage) {
    if (!s->failed) {
        s->failed = 1;
        s->error = status;
        s->error_stage = stage;
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
static potal_status paired_raw(potal_paired_session *s, uint32_t funct, uint64_t rs1, uint64_t rs2, int xd) {
    s->commands++;
    if (s->sink.cmd != NULL) {
        const potal_paired_cmd_record rec = { funct, (uint32_t)xd, rs1, rs2 };
        return s->sink.cmd(s->sink.ctx, &rec);
    }
#if defined(__riscv)
    switch (funct) {
        case POTAL_FUNCT_CONFIG:                    potal_rocc_config(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS:     potal_rocc_loop_ws_bounds(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB:   potal_rocc_loop_ws_addrs_ab(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_DC:   potal_rocc_loop_ws_addrs_dc(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_AB: potal_rocc_loop_ws_strides_ab(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_DC: potal_rocc_loop_ws_strides_dc(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_LOOP_WS:                   potal_rocc_loop_ws(rs1, rs2); return POTAL_OK;
        case POTAL_FUNCT_MVOUT:                     potal_rocc_mvout(rs1, rs2); return POTAL_OK;
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
    for (uint32_t p = 0; p < POTAL_PAIRS; ++p) {
        s->pair[p].outstanding = 0;
    }
    s->drains++;
}

static potal_status paired_ready(const potal_paired_session *s) {
    if (s == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if (s->failed) {
        return s->error;
    }
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
    paired_owner = 1;
    *out = s;
    return POTAL_OK;
}

potal_status potal_paired_session_close(potal_paired_session *s) {
    if (s == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const int outstanding = s->outstanding_loops != 0 || s->outstanding_readbacks != 0;
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
    const potal_status ready = paired_ready(s);
    if (ready != POTAL_OK) {
        return ready;
    }
    if (s->outstanding_loops != 0 || s->outstanding_readbacks != 0) {
        return POTAL_ERR_BUSY; /* no flush under in-flight DMA */
    }
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
    PAIRED_RAW(s, POTAL_FUNCT_FLUSH, 0, 0, 0, "tlb_flush");
    return POTAL_OK;
}

potal_status potal_paired_scale_write(potal_paired_session *s, uint32_t slot, uint32_t generation,
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
    PAIRED_RAW(s, POTAL_FUNCT_SCALE_WRITE, rs1, carrier, 1, "scale_write");
    s->slot_columns[slot] |= bit;
    s->slot_generation[slot] = (uint8_t)generation;
    return POTAL_OK;
}

potal_status potal_paired_scale_release(potal_paired_session *s, uint32_t slot, uint32_t generation,
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
    PAIRED_RAW(s, POTAL_FUNCT_SCALE_RELEASE, rs1, 0, 1, "scale_release");
    s->slot_columns[slot] &= ~((uint64_t)1 << column); /* the slot is free when its last live column is released */
    paired_drained(s);
    return POTAL_OK;
}

potal_status potal_paired_index_config(potal_paired_session *s, const potal_index_config *cfg) {
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
    PAIRED_RAW(s, POTAL_FUNCT_INDEX_CONFIG, c.rs1, c.rs2, 1, "index_config");
    /* The hardware invalidated the buffer's entries: the pair's window is valid again only when complete. */
    s->pair[cfg->pair].index_valid = 0;
    s->index_open = 1;
    s->index_pair = cfg->pair;
    s->index_written = 0;
    s->index_cfg = *cfg;
    return POTAL_OK;
}

potal_status potal_paired_index_write(potal_paired_session *s, const potal_index_write *w) {
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
    PAIRED_RAW(s, POTAL_FUNCT_INDEX_WRITE, c.rs1, c.rs2, 1, "index_write");
    s->index_written++;
    if (s->index_written == s->index_cfg.active_rows) {
        paired_pair_state *pr = &s->pair[s->index_pair];
        pr->index_valid = 1;
        pr->index = s->index_cfg;
        s->index_open = 0;
    }
    return POTAL_OK;
}

static potal_status paired_issue_program(potal_paired_session *s, const potal_gemm_ws_program *p, const char *stage) {
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
    PAIRED_RAW(s, POTAL_FUNCT_LOOP_WS, p->loop_ws.rs1, p->loop_ws.rs2, 0, stage);
    return POTAL_OK;
}

potal_status potal_paired_loop_ws_issue(potal_paired_session *s, const potal_paired_loop_ws *pl) {
    const potal_status ready = paired_ready(s);
    if (ready != POTAL_OK) {
        return ready;
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
    const potal_status ready = paired_ready(s);
    if (ready != POTAL_OK) {
        return ready;
    }
    potal_rocc_cmd config_st;
    potal_rocc_cmd mvouts[POTAL_WORK_IDS_PER_LOOP];
    uint32_t count = 0;
    const potal_status status = potal_acc_readback_encode(r, &config_st, mvouts, POTAL_WORK_IDS_PER_LOOP, &count);
    if (status != POTAL_OK) {
        return status;
    }
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
    PAIRED_RAW(s, POTAL_FUNCT_CONFIG, config_st.rs1, config_st.rs2, 0, "acc_readback");
    for (uint32_t t = 0; t < count; ++t) {
        PAIRED_RAW(s, POTAL_FUNCT_MVOUT, mvouts[t].rs1, mvouts[t].rs2, 0, "acc_readback");
    }
    s->outstanding_readbacks++;
    return POTAL_OK;
}

potal_status potal_paired_session_cleanup(potal_paired_session *s) {
    if (s == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
    if (s->failed && strcmp(s->error_stage != NULL ? s->error_stage : "", "fence") == 0) {
        return s->error; /* a fence that did not answer OK is not retried */
    }
    if (s->outstanding_loops != 0 || s->outstanding_readbacks != 0) {
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
