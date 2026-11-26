#ifndef GEMMINI_TYPES_H
#define GEMMINI_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum gemmini_status {
    GEMMINI_OK = 0,
    /* Software rejections: nothing was issued. */
    GEMMINI_ERR_INVALID_ARGUMENT = -1, /* field out of range, NULL pointer */
    GEMMINI_ERR_UNSUPPORTED      = -2, /* valid for the RTL, not supported by this API */
    GEMMINI_ERR_CAPACITY         = -3, /* SPAD/ACC/DMA capacity exceeded */
    GEMMINI_ERR_NO_HARDWARE      = -4, /* RoCC issue on a non-RISC-V build */
    /* Transport failures. */
    GEMMINI_ERR_TRANSPORT        = -5, /* the transport rejected or lost the command */
    GEMMINI_ERR_RESPONSE         = -6, /* response pointer does not match the command's xd */
} gemmini_status_t;

const char *gemmini_status_name(gemmini_status_t status);

/*
 * Hardware geometry the API validates against. NPU_ACC_ROWS is the total
 * accumulator row count over all banks; per-bank rows are derived.
 */
typedef struct gemmini_hw_config {
    uint32_t dim;            /* systolic array dimension D */
    uint32_t input_bits;     /* operand precision (8 for INT8) */
    uint32_t input_bytes;    /* storage bytes of one operand element (elem_t) */
    uint32_t acc_bytes;      /* storage bytes of one accumulator element (acc_t) */
    uint32_t spad_banks;
    uint32_t spad_bank_rows;
    uint32_t acc_banks;
    uint32_t acc_rows;       /* total over all ACC banks */
    uint32_t dma_max_bytes;  /* MAX_BYTES: largest single DMA request */
} gemmini_hw_config_t;

/* Geometry of this build (generated npu_config.h). */
const gemmini_hw_config_t *gemmini_hw_config_build(void);

gemmini_status_t gemmini_hw_config_check(const gemmini_hw_config_t *cfg);

uint32_t gemmini_spad_rows(const gemmini_hw_config_t *cfg);
/* MAX_BLOCK_LEN / MAX_BLOCK_LEN_ACC of the SDK. */
uint32_t gemmini_max_block_len(const gemmini_hw_config_t *cfg);
uint32_t gemmini_max_block_len_acc(const gemmini_hw_config_t *cfg);

#ifdef __cplusplus
}
#endif

#endif /* GEMMINI_TYPES_H */
