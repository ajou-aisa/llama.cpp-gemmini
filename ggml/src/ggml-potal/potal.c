/* PoTal minimal RoCC API. See include/potal.h for the contract. */

#include "potal.h"

#include "potal_hw_params.h"

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

/* LOOP_WS_CONFIG_BOUNDS: 16-bit I/J/K iterators at [15:0]/[31:16]/[47:32]. */
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

potal_status potal_gemm_ws_encode(const potal_gemm_ws *g, potal_gemm_ws_program *p) {
    if (p == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    const potal_status status = potal_gemm_ws_check(g);
    if (status != POTAL_OK) {
        return status;
    }
    const uint64_t i_tiles = g->m / POTAL_HW_DIM;
    const uint64_t j_tiles = g->n / POTAL_HW_DIM;
    const uint64_t k_tiles = g->k / POTAL_HW_DIM;

    /* 1. Gemmini execution config: mvin strides in bytes, WS, INT32 mvout. */
    p->config_ld_a = cmd(config_ld_rs1(0), g->lda * sizeof(int8_t));
    p->config_ld_b = cmd(config_ld_rs1(1), g->ldb * sizeof(int8_t));
    p->config_ex   = cmd(((uint64_t)1 << CONFIG_EX_A_STRIDE_SHIFT) | CONFIG_EX_WS | CONFIG_EX,
                         (uint64_t)1 << CONFIG_EX_C_STRIDE_SHIFT);
    p->config_st   = cmd(CONFIG_ST, g->ldc * sizeof(int32_t));

    /* 2. LOOP_WS bounds (no padding), addresses (no D), strides in elements. */
    p->bounds     = cmd(0, (k_tiles << BOUNDS_K_SHIFT) | (j_tiles << BOUNDS_J_SHIFT) | i_tiles);
    p->addrs_ab   = cmd((uint64_t)(uintptr_t)g->a, (uint64_t)(uintptr_t)g->b);
    p->addrs_dc   = cmd(0, (uint64_t)(uintptr_t)g->c);
    p->strides_ab = cmd(g->lda, g->ldb);
    p->strides_dc = cmd(0, g->ldc);

    /* 3. PoTal scale metadata on LOOP_WS; full_C, no accumulate/act/transpose. */
    p->loop_ws = cmd(((uint64_t)1 << LOOP_WS_FULL_C_SHIFT) |
                     ((uint64_t)1 << LOOP_WS_POTAL_ENABLE_SHIFT) |
                     ((uint64_t)g->scale_base << LOOP_WS_SCALE_BASE_SHIFT) |
                     ((uint64_t)g->scale_generation << LOOP_WS_GENERATION_SHIFT) |
                     ((uint64_t)g->passes_per_block << LOOP_WS_PASSES_SHIFT),
                     0);
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
    const potal_status status = hw_status(potal_rocc_fence(0, 0));
    if (status == POTAL_OK) {
        gemm_in_flight = 0;
    }
    /* Order later CPU reads of C after the drained DMA stores. */
    potal_cpu_fence_rw();
    return status;
}

potal_status potal_tlb_flush(void) {
    if (gemm_in_flight) {
        return POTAL_ERR_BUSY;
    }
    potal_rocc_flush(0, 0);
    return POTAL_OK;
}

potal_status potal_gemm_ws_issue(const potal_gemm_ws *g) {
    potal_gemm_ws_program p;
    const potal_status status = potal_gemm_ws_encode(g, &p);
    if (status != POTAL_OK) {
        return status;
    }
    if (gemm_in_flight) {
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
    return POTAL_ERR_NO_HARDWARE;
}

potal_status potal_tlb_flush(void) {
    return POTAL_ERR_NO_HARDWARE;
}

potal_status potal_gemm_ws_issue(const potal_gemm_ws *g) {
    potal_gemm_ws_program p;
    const potal_status status = potal_gemm_ws_encode(g, &p);
    return status != POTAL_OK ? status : POTAL_ERR_NO_HARDWARE;
}

#endif
