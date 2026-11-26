#include "gemmini_command.h"
#include "gemmini_isa.h"

#define FITS(value, bits) ((uint64_t) (value) < ((uint64_t) 1 << (bits)))

static void set_command(gemmini_command_t *command, uint32_t funct, uint64_t rs1, uint64_t rs2) {
    command->funct = funct;
    command->xd    = (uint8_t) gemmini_funct_has_response(funct);
    command->xs1   = 1;
    command->xs2   = 1;
    command->rs1   = rs1;
    command->rs2   = rs2;
}

static uint64_t pack_rows_cols_addr(uint32_t rows, uint32_t cols, uint32_t addr) {
    return ((uint64_t) rows << GEMMINI_ROWS_SHIFT) | ((uint64_t) cols << GEMMINI_COLS_SHIFT) | (uint64_t) addr;
}

static int rows_cols_fit(uint32_t rows, uint32_t cols) {
    return rows <= GEMMINI_ROWS_COLS_MAX && cols <= GEMMINI_ROWS_COLS_MAX;
}

int gemmini_funct_is_supported(uint32_t funct) {
    switch (funct) {
        case GEMMINI_FUNCT_CONFIG:
        case GEMMINI_FUNCT_MVIN2:
        case GEMMINI_FUNCT_MVIN:
        case GEMMINI_FUNCT_MVOUT:
        case GEMMINI_FUNCT_COMPUTE_PRELOADED:
        case GEMMINI_FUNCT_COMPUTE_ACCUMULATE:
        case GEMMINI_FUNCT_PRELOAD:
        case GEMMINI_FUNCT_FLUSH:
        case GEMMINI_FUNCT_LOOP_WS:
        case GEMMINI_FUNCT_LOOP_WS_CONFIG_BOUNDS:
        case GEMMINI_FUNCT_LOOP_WS_CONFIG_ADDRS_AB:
        case GEMMINI_FUNCT_LOOP_WS_CONFIG_ADDRS_DC:
        case GEMMINI_FUNCT_LOOP_WS_CONFIG_STRIDES_AB:
        case GEMMINI_FUNCT_LOOP_WS_CONFIG_STRIDES_DC:
        case GEMMINI_FUNCT_MVIN3:
        case GEMMINI_FUNCT_COUNTER:
            return 1;
        default:
            return 0;
    }
}

int gemmini_funct_has_response(uint32_t funct) {
    return funct == GEMMINI_FUNCT_COUNTER;
}

gemmini_status_t gemmini_encode_mvin(const gemmini_mvin_desc_t *desc, gemmini_command_t *command) {
    static const uint32_t funct_by_id[3] = { GEMMINI_FUNCT_MVIN, GEMMINI_FUNCT_MVIN2, GEMMINI_FUNCT_MVIN3 };
    if (!desc || !command || desc->id > 2 || !rows_cols_fit(desc->rows, desc->cols)) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    set_command(command, funct_by_id[desc->id], desc->dram_addr,
                pack_rows_cols_addr(desc->rows, desc->cols, desc->local_addr));
    return GEMMINI_OK;
}

gemmini_status_t gemmini_encode_mvout(const gemmini_mvout_desc_t *desc, gemmini_command_t *command) {
    if (!desc || !command || !rows_cols_fit(desc->rows, desc->cols)) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    set_command(command, GEMMINI_FUNCT_MVOUT, desc->dram_addr,
                pack_rows_cols_addr(desc->rows, desc->cols, desc->local_addr));
    return GEMMINI_OK;
}

gemmini_status_t gemmini_encode_preload(const gemmini_preload_desc_t *desc, gemmini_command_t *command) {
    if (!desc || !command || !rows_cols_fit(desc->bd_rows, desc->bd_cols) || !rows_cols_fit(desc->c_rows, desc->c_cols)) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    set_command(command, GEMMINI_FUNCT_PRELOAD, pack_rows_cols_addr(desc->bd_rows, desc->bd_cols, desc->bd_addr),
                pack_rows_cols_addr(desc->c_rows, desc->c_cols, desc->c_addr));
    return GEMMINI_OK;
}

gemmini_status_t gemmini_encode_compute(const gemmini_compute_desc_t *desc, gemmini_command_t *command) {
    if (!desc || !command || !rows_cols_fit(desc->a_rows, desc->a_cols) || !rows_cols_fit(desc->bd_rows, desc->bd_cols)) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    set_command(command, desc->accumulate ? GEMMINI_FUNCT_COMPUTE_ACCUMULATE : GEMMINI_FUNCT_COMPUTE_PRELOADED,
                pack_rows_cols_addr(desc->a_rows, desc->a_cols, desc->a_addr),
                pack_rows_cols_addr(desc->bd_rows, desc->bd_cols, desc->bd_addr));
    return GEMMINI_OK;
}

gemmini_status_t gemmini_encode_config_ex(const gemmini_config_ex_desc_t *desc, gemmini_command_t *command) {
    if (!desc || !command || desc->dataflow > GEMMINI_DATAFLOW_WS || !FITS(desc->activation, 4) ||
        !FITS(desc->a_stride, 16) || !FITS(desc->c_stride, 16)) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    const uint64_t rs1 = ((uint64_t) desc->acc_scale_bits << 32) | ((uint64_t) desc->a_stride << 16) |
                         ((uint64_t) desc->b_transpose << 9) | ((uint64_t) desc->a_transpose << 8) |
                         ((uint64_t) desc->set_only_strides << 7) | ((uint64_t) desc->activation << 3) |
                         ((uint64_t) desc->dataflow << 2) | GEMMINI_CONFIG_EX;
    const uint64_t rs2 = ((uint64_t) desc->c_stride << 48) | (uint64_t) desc->sys_shift;
    set_command(command, GEMMINI_FUNCT_CONFIG, rs1, rs2);
    return GEMMINI_OK;
}

gemmini_status_t gemmini_encode_config_ld(const gemmini_config_ld_desc_t *desc, gemmini_command_t *command) {
    if (!desc || !command || desc->id > 2 || !FITS(desc->block_mvin_stride, 16) || !FITS(desc->pixel_repeats, 8)) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    const uint64_t rs1 = ((uint64_t) desc->scale_bits << 32) | ((uint64_t) desc->block_mvin_stride << 16) |
                         ((uint64_t) desc->pixel_repeats << 8) | ((uint64_t) desc->id << 3) |
                         ((uint64_t) desc->shrunk << 2) | GEMMINI_CONFIG_LD;
    set_command(command, GEMMINI_FUNCT_CONFIG, rs1, desc->stride);
    return GEMMINI_OK;
}

gemmini_status_t gemmini_encode_config_st(const gemmini_config_st_desc_t *desc, gemmini_command_t *command) {
    if (!desc || !command || !FITS(desc->activation, 2)) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    const uint64_t rs1 = ((uint64_t) desc->activation << 2) | GEMMINI_CONFIG_ST;
    const uint64_t rs2 = ((uint64_t) desc->acc_scale_bits << 32) | (uint64_t) desc->stride;
    set_command(command, GEMMINI_FUNCT_CONFIG, rs1, rs2);
    return GEMMINI_OK;
}

gemmini_status_t gemmini_encode_flush(const gemmini_flush_desc_t *desc, gemmini_command_t *command) {
    if (!desc || !command) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    set_command(command, GEMMINI_FUNCT_FLUSH, (uint64_t) desc->skip, 0);
    return GEMMINI_OK;
}

gemmini_status_t gemmini_encode_loop_ws(const gemmini_loop_ws_desc_t *desc,
                                        gemmini_command_t commands[GEMMINI_LOOP_WS_COMMANDS]) {
    if (!desc || !commands || !FITS(desc->i, 16) || !FITS(desc->j, 16) || !FITS(desc->k, 16) ||
        !FITS(desc->pad_i, 16) || !FITS(desc->pad_j, 16) || !FITS(desc->pad_k, 16) || !FITS(desc->activation, 8) ||
        !FITS(desc->a_spad_id, 2) || !FITS(desc->b_spad_id, 2)) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    set_command(&commands[0], GEMMINI_FUNCT_LOOP_WS_CONFIG_BOUNDS,
                ((uint64_t) desc->pad_k << 32) | ((uint64_t) desc->pad_j << 16) | (uint64_t) desc->pad_i,
                ((uint64_t) desc->k << 32) | ((uint64_t) desc->j << 16) | (uint64_t) desc->i);
    set_command(&commands[1], GEMMINI_FUNCT_LOOP_WS_CONFIG_ADDRS_AB, desc->a, desc->b);
    set_command(&commands[2], GEMMINI_FUNCT_LOOP_WS_CONFIG_ADDRS_DC, desc->d, desc->c);
    set_command(&commands[3], GEMMINI_FUNCT_LOOP_WS_CONFIG_STRIDES_AB, desc->a_stride, desc->b_stride);
    set_command(&commands[4], GEMMINI_FUNCT_LOOP_WS_CONFIG_STRIDES_DC, desc->d_stride, desc->c_stride);
    set_command(&commands[5], GEMMINI_FUNCT_LOOP_WS,
                ((uint64_t) desc->a_spad_id << 18) | ((uint64_t) desc->b_spad_id << 16) |
                    ((uint64_t) desc->activation << 8) | ((uint64_t) desc->low_d << 2) |
                    ((uint64_t) desc->full_c << 1) | (uint64_t) desc->ex_accumulate,
                ((uint64_t) desc->is_resadd << 2) | ((uint64_t) desc->b_transpose << 1) | (uint64_t) desc->a_transpose);
    return GEMMINI_OK;
}

gemmini_status_t gemmini_encode_counter(const gemmini_counter_desc_t *desc, gemmini_command_t *command) {
    if (!desc || !command) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    uint64_t rs1 = 0;
    switch (desc->op) {
        case GEMMINI_COUNTER_OP_READ:
            if (desc->index >= GEMMINI_COUNTER_SLOTS) {
                return GEMMINI_ERR_INVALID_ARGUMENT;
            }
            rs1 = (uint64_t) desc->index << GEMMINI_COUNTER_INDEX_SHIFT;
            break;
        case GEMMINI_COUNTER_OP_CONFIGURE: {
            const uint32_t max_event = GEMMINI_EVENT_WDMA_TOTAL_LATENCY;
            if (desc->index >= GEMMINI_COUNTER_SLOTS || desc->event > max_event) {
                return GEMMINI_ERR_INVALID_ARGUMENT;
            }
            const int      external = desc->event > GEMMINI_COUNTER_INCREMENTAL_EVENTS;
            const uint32_t code     = external ? desc->event - GEMMINI_COUNTER_INCREMENTAL_EVENTS : desc->event;
            rs1 = ((uint64_t) desc->index << GEMMINI_COUNTER_INDEX_SHIFT) | GEMMINI_COUNTER_CONFIGURE |
                  ((uint64_t) (code & GEMMINI_COUNTER_EVENT_MASK) << GEMMINI_COUNTER_EVENT_SHIFT) |
                  ((uint64_t) external << GEMMINI_COUNTER_EXTERNAL_BIT);
            break;
        }
        case GEMMINI_COUNTER_OP_SNAPSHOT_TAKE:
            rs1 = GEMMINI_COUNTER_SNAPSHOT_TAKE;
            break;
        case GEMMINI_COUNTER_OP_SNAPSHOT_RESET:
            rs1 = GEMMINI_COUNTER_SNAPSHOT_RESET;
            break;
        case GEMMINI_COUNTER_OP_RESET:
            rs1 = GEMMINI_COUNTER_RESET;
            break;
        default:
            return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    set_command(command, GEMMINI_FUNCT_COUNTER, rs1, 0);
    return GEMMINI_OK;
}
