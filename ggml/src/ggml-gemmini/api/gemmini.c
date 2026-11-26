#include "gemmini.h"

#include "npu_config.h"

#if NPU_ARCH_GEMMINI != 1
#error "gemmini-api requires a GGML_NPU=GEMMINI npu_config.h"
#endif

const char *gemmini_status_name(gemmini_status_t status) {
    switch (status) {
        case GEMMINI_OK:                   return "GEMMINI_OK";
        case GEMMINI_ERR_INVALID_ARGUMENT: return "GEMMINI_ERR_INVALID_ARGUMENT";
        case GEMMINI_ERR_UNSUPPORTED:      return "GEMMINI_ERR_UNSUPPORTED";
        case GEMMINI_ERR_CAPACITY:         return "GEMMINI_ERR_CAPACITY";
        case GEMMINI_ERR_NO_HARDWARE:      return "GEMMINI_ERR_NO_HARDWARE";
        case GEMMINI_ERR_TRANSPORT:        return "GEMMINI_ERR_TRANSPORT";
        case GEMMINI_ERR_RESPONSE:         return "GEMMINI_ERR_RESPONSE";
    }
    return "GEMMINI_STATUS_UNKNOWN";
}

static const gemmini_hw_config_t build_config = {
    .dim            = NPU_D,
    .input_bits     = NPU_PRECISION_BITS,
    .input_bytes    = NPU_INPUT_STORAGE_BYTES,
    .acc_bytes      = NPU_ACC_BITS / 8,
    .spad_banks     = NPU_SPAD_BANKS,
    .spad_bank_rows = NPU_SPAD_BANK_ROWS,
    .acc_banks      = NPU_ACC_BANKS,
    .acc_rows       = NPU_ACC_ROWS,
    .dma_max_bytes  = NPU_DMA_MAX_BYTES,
};

const gemmini_hw_config_t *gemmini_hw_config_build(void) {
    return &build_config;
}

static int is_pow2(uint32_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

gemmini_status_t gemmini_hw_config_check(const gemmini_hw_config_t *cfg) {
    if (!cfg || !is_pow2(cfg->dim) || cfg->input_bits != 8 || cfg->input_bytes != 1 || cfg->acc_bytes != 4 ||
        cfg->spad_banks == 0 || cfg->spad_bank_rows == 0 || cfg->acc_banks == 0 || cfg->acc_rows == 0 ||
        cfg->acc_rows % cfg->acc_banks != 0 || cfg->dma_max_bytes == 0) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    if ((uint64_t) cfg->spad_banks * cfg->spad_bank_rows > (1u << 25) || cfg->acc_rows > (1u << 25)) {
        return GEMMINI_ERR_UNSUPPORTED;
    }
    return GEMMINI_OK;
}

uint32_t gemmini_spad_rows(const gemmini_hw_config_t *cfg) {
    return cfg->spad_banks * cfg->spad_bank_rows;
}

/* GemminiConfigs.scala generateHeader: MAX_BYTES / row bytes, or 1 when one row is wider. */
static uint32_t block_len(uint32_t dma_max_bytes, uint32_t row_bytes) {
    return row_bytes > dma_max_bytes ? 1 : dma_max_bytes / row_bytes;
}

uint32_t gemmini_max_block_len(const gemmini_hw_config_t *cfg) {
    return block_len(cfg->dma_max_bytes, cfg->dim * cfg->input_bytes);
}

uint32_t gemmini_max_block_len_acc(const gemmini_hw_config_t *cfg) {
    return block_len(cfg->dma_max_bytes, cfg->dim * cfg->acc_bytes);
}

gemmini_status_t gemmini_issue(const gemmini_transport_t *transport, const gemmini_command_t *command,
                               uint64_t *response) {
    if (!transport || !transport->issue || !command || !gemmini_funct_is_supported(command->funct)) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    if ((command->xd != 0) != (response != NULL)) {
        return GEMMINI_ERR_RESPONSE;
    }
    return transport->issue(transport->context, command, response);
}

gemmini_status_t gemmini_issue_all(const gemmini_transport_t *transport, const gemmini_command_t *commands,
                                   size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const gemmini_status_t status = gemmini_issue(transport, &commands[i], NULL);
        if (status != GEMMINI_OK) {
            return status;
        }
    }
    return GEMMINI_OK;
}

gemmini_status_t gemmini_drain(const gemmini_transport_t *transport) {
    if (!transport || !transport->drain) {
        return GEMMINI_ERR_INVALID_ARGUMENT;
    }
    return transport->drain(transport->context);
}
