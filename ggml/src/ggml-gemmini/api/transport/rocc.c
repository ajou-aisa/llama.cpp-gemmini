#include "gemmini.h"

/*
 * funct7 is an immediate of ".insn r", so every supported command has its own
 * instruction. Like the SDK's xcustom.h these asm statements carry no
 * "memory" clobber and no fence: ordering against DMA buffers is the
 * caller's gemmini_drain(), exactly as gemmini_fence() was.
 */
#if defined(__riscv)

#define GEMMINI_STR_(x) #x
#define GEMMINI_STR(x)  GEMMINI_STR_(x)

#define GEMMINI_INSN_0_R_R(funct, rs1, rs2)                                                        \
    __asm__ volatile(".insn r CUSTOM_" GEMMINI_STR(GEMMINI_XCUSTOM) ", 0x3, " GEMMINI_STR(funct) \
                     ", x0, %0, %1" ::"r"(rs1), "r"(rs2))

#define GEMMINI_INSN_R_R_R(funct, rd, rs1, rs2)                                                    \
    __asm__ volatile(".insn r CUSTOM_" GEMMINI_STR(GEMMINI_XCUSTOM) ", 0x7, " GEMMINI_STR(funct) \
                     ", %0, %1, %2"                                                              \
                     : "=r"(rd)                                                                  \
                     : "r"(rs1), "r"(rs2))

#define GEMMINI_CASE_0_R_R(funct) \
    case funct:                   \
        GEMMINI_INSN_0_R_R(funct, rs1, rs2); \
        return GEMMINI_OK

static gemmini_status_t rocc_issue(void *context, const gemmini_command_t *command, uint64_t *response) {
    (void) context;
    const uint64_t rs1 = command->rs1;
    const uint64_t rs2 = command->rs2;
    switch (command->funct) {
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_CONFIG);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_MVIN2);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_MVIN);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_MVOUT);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_COMPUTE_PRELOADED);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_COMPUTE_ACCUMULATE);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_PRELOAD);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_FLUSH);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_LOOP_WS);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_LOOP_WS_CONFIG_BOUNDS);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_LOOP_WS_CONFIG_ADDRS_AB);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_LOOP_WS_CONFIG_ADDRS_DC);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_LOOP_WS_CONFIG_STRIDES_AB);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_LOOP_WS_CONFIG_STRIDES_DC);
        GEMMINI_CASE_0_R_R(GEMMINI_FUNCT_MVIN3);
        case GEMMINI_FUNCT_COUNTER: {
            uint64_t rd;
            GEMMINI_INSN_R_R_R(GEMMINI_FUNCT_COUNTER, rd, rs1, rs2);
            *response = rd;
            return GEMMINI_OK;
        }
        default:
            return GEMMINI_ERR_INVALID_ARGUMENT;
    }
}

static gemmini_status_t rocc_drain(void *context) {
    (void) context;
    __asm__ volatile("fence");
    return GEMMINI_OK;
}

int gemmini_rocc_available(void) {
    return 1;
}

#else

static gemmini_status_t rocc_issue(void *context, const gemmini_command_t *command, uint64_t *response) {
    (void) context;
    (void) command;
    (void) response;
    return GEMMINI_ERR_NO_HARDWARE;
}

static gemmini_status_t rocc_drain(void *context) {
    (void) context;
    return GEMMINI_ERR_NO_HARDWARE;
}

int gemmini_rocc_available(void) {
    return 0;
}

#endif

static const gemmini_transport_t rocc_transport = { NULL, rocc_issue, rocc_drain };

const gemmini_transport_t *gemmini_rocc_transport(void) {
    return &rocc_transport;
}
