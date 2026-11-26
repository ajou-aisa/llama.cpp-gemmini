/*
 * Gemmini command representation and pure encoders.
 *
 * Encoders never issue, never touch GGML tensors and are deterministic: they
 * check every field against its bit width, then pack rs1/rs2 exactly as the
 * SDK macros did (bitwise-compared in tests/gemmini-encoding-test.c).
 * Geometry checks (SPAD/ACC bounds, DMA size) are separate:
 * gemmini_validate_* in validate.c.
 *
 * gemmini_command_t is an in-process type; it is not a wire format.
 */
#ifndef GEMMINI_COMMAND_H
#define GEMMINI_COMMAND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "gemmini_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gemmini_command {
    uint32_t funct; /* funct7 */
    uint8_t  xd;    /* 1: the command writes rd (a response is returned) */
    uint8_t  xs1;
    uint8_t  xs2;
    uint64_t rs1;
    uint64_t rs2;
} gemmini_command_t;

/* MVIN (id 0), MVIN2 (id 1), MVIN3 (id 2): the CONFIG_LD state the load uses. */
typedef struct gemmini_mvin_desc {
    uint32_t id;
    uint64_t dram_addr;
    uint32_t local_addr;
    uint32_t cols;
    uint32_t rows;
} gemmini_mvin_desc_t;

typedef struct gemmini_mvout_desc {
    uint64_t dram_addr;
    uint32_t local_addr;
    uint32_t cols;
    uint32_t rows;
} gemmini_mvout_desc_t;

typedef struct gemmini_preload_desc {
    uint32_t bd_addr;
    uint32_t c_addr;
    uint32_t bd_cols;
    uint32_t bd_rows;
    uint32_t c_cols;
    uint32_t c_rows;
} gemmini_preload_desc_t;

typedef struct gemmini_compute_desc {
    bool     accumulate; /* COMPUTE_ACCUMULATE instead of COMPUTE_PRELOADED */
    uint32_t a_addr;
    uint32_t bd_addr;
    uint32_t a_cols;
    uint32_t a_rows;
    uint32_t bd_cols;
    uint32_t bd_rows;
} gemmini_compute_desc_t;

/* Float scales travel as their IEEE-754 bit pattern. */
typedef struct gemmini_config_ex_desc {
    uint32_t dataflow;       /* GEMMINI_DATAFLOW_* */
    uint32_t activation;     /* rs1[6:3] */
    uint32_t sys_shift;      /* rs2[31:0] */
    uint32_t acc_scale_bits; /* rs1[63:32] */
    uint32_t c_stride;       /* rs2[63:48] */
    uint32_t a_stride;       /* rs1[31:16] */
    bool     a_transpose;
    bool     b_transpose;
    bool     set_only_strides;
} gemmini_config_ex_desc_t;

typedef struct gemmini_config_ld_desc {
    uint64_t stride;            /* rs2: DRAM row stride in bytes */
    uint32_t scale_bits;        /* rs1[63:32] */
    bool     shrunk;            /* rs1[2]: ACC load of elem_t data */
    uint32_t block_mvin_stride; /* rs1[31:16] */
    uint32_t pixel_repeats;     /* rs1[15:8] */
    uint32_t id;                /* rs1[4:3]: 0 MVIN, 1 MVIN2, 2 MVIN3 */
} gemmini_config_ld_desc_t;

/* CONFIG_ST without pooling: the pool/pad fields are encoded as zero. */
typedef struct gemmini_config_st_desc {
    uint32_t stride;         /* rs2[31:0]: DRAM row stride in bytes */
    uint32_t activation;     /* rs1[3:2] */
    uint32_t acc_scale_bits; /* rs2[63:32] */
} gemmini_config_st_desc_t;

typedef struct gemmini_flush_desc {
    bool skip; /* rs1[0] */
} gemmini_flush_desc_t;

/* One LOOP_WS: five LOOP_WS_CONFIG_* commands followed by LOOP_WS. */
typedef struct gemmini_loop_ws_desc {
    uint32_t i, j, k;             /* tiles of DIM, 16 bits each */
    uint32_t pad_i, pad_j, pad_k; /* elements missing from the last tile */
    uint64_t a, b, d, c;          /* DRAM addresses (0: operand absent) */
    uint64_t a_stride, b_stride, d_stride, c_stride; /* elements */
    bool     a_transpose;
    bool     b_transpose;
    bool     full_c;
    bool     low_d;
    bool     ex_accumulate;
    uint32_t activation; /* rs1[15:8] of LOOP_WS */
    uint32_t a_spad_id;  /* rs1[19:18] */
    uint32_t b_spad_id;  /* rs1[17:16] */
    bool     is_resadd;
} gemmini_loop_ws_desc_t;

#define GEMMINI_LOOP_WS_COMMANDS 6

typedef enum gemmini_counter_op {
    GEMMINI_COUNTER_OP_READ,           /* returns the slot value */
    GEMMINI_COUNTER_OP_CONFIGURE,      /* slot <- event */
    GEMMINI_COUNTER_OP_SNAPSHOT_TAKE,
    GEMMINI_COUNTER_OP_SNAPSHOT_RESET,
    GEMMINI_COUNTER_OP_RESET,
} gemmini_counter_op_t;

typedef struct gemmini_counter_desc {
    gemmini_counter_op_t op;
    uint32_t             index; /* slot for READ/CONFIGURE, < GEMMINI_COUNTER_SLOTS */
    uint32_t             event; /* GEMMINI_EVENT_* for CONFIGURE */
} gemmini_counter_desc_t;

gemmini_status_t gemmini_encode_mvin(const gemmini_mvin_desc_t *desc, gemmini_command_t *command);
gemmini_status_t gemmini_encode_mvout(const gemmini_mvout_desc_t *desc, gemmini_command_t *command);
gemmini_status_t gemmini_encode_preload(const gemmini_preload_desc_t *desc, gemmini_command_t *command);
gemmini_status_t gemmini_encode_compute(const gemmini_compute_desc_t *desc, gemmini_command_t *command);
gemmini_status_t gemmini_encode_config_ex(const gemmini_config_ex_desc_t *desc, gemmini_command_t *command);
gemmini_status_t gemmini_encode_config_ld(const gemmini_config_ld_desc_t *desc, gemmini_command_t *command);
gemmini_status_t gemmini_encode_config_st(const gemmini_config_st_desc_t *desc, gemmini_command_t *command);
gemmini_status_t gemmini_encode_flush(const gemmini_flush_desc_t *desc, gemmini_command_t *command);
/* Writes GEMMINI_LOOP_WS_COMMANDS commands in issue order. */
gemmini_status_t gemmini_encode_loop_ws(const gemmini_loop_ws_desc_t *desc,
                                        gemmini_command_t commands[GEMMINI_LOOP_WS_COMMANDS]);
gemmini_status_t gemmini_encode_counter(const gemmini_counter_desc_t *desc, gemmini_command_t *command);

/* Geometry checks against a hardware configuration (validate.c). */
gemmini_status_t gemmini_validate_mvin(const gemmini_hw_config_t *cfg, const gemmini_mvin_desc_t *desc);
gemmini_status_t gemmini_validate_mvout(const gemmini_hw_config_t *cfg, const gemmini_mvout_desc_t *desc);
gemmini_status_t gemmini_validate_loop_ws(const gemmini_hw_config_t *cfg, const gemmini_loop_ws_desc_t *desc);

/* 1 if the funct7 value is one this API encodes, else 0. */
int gemmini_funct_is_supported(uint32_t funct);
/* 1 if the command returns a response (xd), per the ISA contract. */
int gemmini_funct_has_response(uint32_t funct);

#ifdef __cplusplus
}
#endif

#endif /* GEMMINI_COMMAND_H */
