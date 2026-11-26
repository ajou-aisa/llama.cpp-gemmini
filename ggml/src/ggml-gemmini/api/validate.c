#include "gemmini_command.h"
#include "gemmini_isa.h"

#define LOCAL_FLAG_BITS (GEMMINI_LOCAL_ACC | GEMMINI_LOCAL_ACCUMULATE | GEMMINI_LOCAL_READ_FULL)
#define ACC_ROW_MASK    0x03ffffffu

static gemmini_status_t check_local_span(const gemmini_hw_config_t *cfg, uint32_t addr, uint32_t cols,
                                         uint32_t rows, int allow_accumulate) {
    if (rows == 0 || rows > cfg->dim || cols == 0) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    if (addr == GEMMINI_LOCAL_GARBAGE) {
        return GEMMINI_OK;
    }
    const int is_acc = (addr & GEMMINI_LOCAL_ACC) != 0;
    if (!is_acc && (addr & LOCAL_FLAG_BITS) != 0) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    if (is_acc && !allow_accumulate && (addr & GEMMINI_LOCAL_ACCUMULATE) != 0) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    if (is_acc && (addr & ~(LOCAL_FLAG_BITS | ACC_ROW_MASK)) != 0) {
        return GEMMINI_ERR_UNSUPPORTED; /* normalization command bits */
    }
    const uint32_t max_cols = cfg->dim * (is_acc ? gemmini_max_block_len_acc(cfg) : gemmini_max_block_len(cfg));
    if (cols > max_cols) {
        return GEMMINI_ERR_CAPACITY;
    }
    const uint64_t row    = is_acc ? (addr & ACC_ROW_MASK) : (addr & ~LOCAL_FLAG_BITS);
    const uint64_t blocks = (cols + cfg->dim - 1) / cfg->dim;
    const uint64_t last   = row + (blocks - 1) * cfg->dim + rows;
    const uint64_t limit  = is_acc ? cfg->acc_rows : gemmini_spad_rows(cfg);
    return last <= limit ? GEMMINI_OK : GEMMINI_ERR_CAPACITY;
}

gemmini_status_t gemmini_validate_mvin(const gemmini_hw_config_t *cfg, const gemmini_mvin_desc_t *desc) {
    if (!cfg || !desc || desc->id > 2 || desc->local_addr == GEMMINI_LOCAL_GARBAGE) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    return check_local_span(cfg, desc->local_addr, desc->cols, desc->rows, 1);
}

gemmini_status_t gemmini_validate_mvout(const gemmini_hw_config_t *cfg, const gemmini_mvout_desc_t *desc) {
    if (!cfg || !desc || desc->local_addr == GEMMINI_LOCAL_GARBAGE) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    return check_local_span(cfg, desc->local_addr, desc->cols, desc->rows, 0);
}

/*
 * LoopMatmul double-buffers: one LOOP_WS may use half of the scratchpad for
 * its A and B tiles and half of the accumulator for C (the capacities
 * tiled_matmul_auto plans against).
 */
gemmini_status_t gemmini_validate_loop_ws(const gemmini_hw_config_t *cfg, const gemmini_loop_ws_desc_t *desc) {
    if (!cfg || !desc || desc->i == 0 || desc->j == 0 || desc->k == 0 || desc->pad_i >= cfg->dim ||
        desc->pad_j >= cfg->dim || desc->pad_k >= cfg->dim || (desc->a_transpose && desc->b_transpose)) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    const uint64_t spad_rows = ((uint64_t) desc->i * desc->k + (uint64_t) desc->k * desc->j) * cfg->dim;
    const uint64_t acc_rows  = (uint64_t) desc->i * desc->j * cfg->dim;
    if (spad_rows > gemmini_spad_rows(cfg) / 2 || acc_rows > cfg->acc_rows / 2) {
        return GEMMINI_ERR_CAPACITY;
    }
    return GEMMINI_OK;
}
