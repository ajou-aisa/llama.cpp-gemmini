/* Checked INT32 ACC-bias encoders and ownership. A sink is software, not hardware. */
#include "potal.h"
#include "potal_hw_params.h"

#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(name, condition) do { \
    const int ok_ = !!(condition); \
    printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); \
    failures += !ok_; \
} while (0)

typedef struct capture {
    potal_paired_cmd_record records[64];
    unsigned count;
    unsigned mvins;
    int fail_mvin;
    int fail_fence;
} capture;

static potal_status command(void *opaque, const potal_paired_cmd_record *record) {
    capture *c = opaque;
    if (c->count < 64) c->records[c->count] = *record;
    ++c->count;
    if (record->funct == POTAL_FUNCT_MVIN && ++c->mvins == (unsigned)c->fail_mvin)
        return POTAL_HW_BAD_COMMAND;
    if (record->funct == POTAL_FUNCT_FENCE && c->fail_fence)
        return POTAL_HW_BAD_COMMAND;
    return POTAL_OK;
}

int main(void) {
    _Alignas(POTAL_HW_DIM * 4) int32_t values[POTAL_HW_DIM * 2] = {0};
    potal_acc_bias bias = {1, 2, 2, 3, 7, values};
    potal_rocc_cmd config, loads[4];
    uint32_t count = 0;
    CHECK("bias encoder", potal_acc_bias_encode(&bias, &config, loads, 4, &count) == POTAL_OK && count == 4);
    CHECK("identity full-width repeated-row config",
          config.rs1 == ((uint64_t)POTAL_HW_DIM << 16 | (uint64_t)1 << 8 | 1) && config.rs2 == 0);
    CHECK("first ACC1 saturating add tile",
          loads[0].rs1 == (uint64_t)(uintptr_t)values &&
          loads[0].rs2 == ((uint64_t)POTAL_HW_DIM << 48 | (uint64_t)POTAL_HW_DIM << 32 |
                           UINT64_C(0xc0000000) | POTAL_HW_ACC_ROWS / 2));
    CHECK("tail columns and repeated bias pointer",
          loads[1].rs1 == (uint64_t)(uintptr_t)(values + POTAL_HW_DIM) &&
          (loads[1].rs2 >> 32 & 0xffff) == POTAL_HW_DIM - 7 &&
          loads[2].rs1 == loads[0].rs1 && (loads[2].rs2 >> 48) == POTAL_HW_DIM - 3);
    CHECK("no full-ACC read flag in MVIN", (loads[0].rs2 & (UINT64_C(1) << 29)) == 0);
    CHECK("null descriptor", potal_acc_bias_check(NULL) == POTAL_ERR_INVALID_ARGUMENT);
    CHECK("short output capacity", potal_acc_bias_encode(&bias, &config, loads, 3, &count) == POTAL_ERR_CAPACITY);
    potal_acc_bias bad = bias;
    bad.acc_bank = 2;
    CHECK("bad bank", potal_acc_bias_check(&bad) == POTAL_ERR_INVALID_ARGUMENT);
    bad = bias; bad.values = (const int32_t *)((const char *)values + 1);
    CHECK("unaligned bias", potal_acc_bias_check(&bad) == POTAL_ERR_INVALID_ARGUMENT);
    bad = bias; bad.i_tiles = 0;
    CHECK("empty output", potal_acc_bias_check(&bad) == POTAL_ERR_INVALID_ARGUMENT);
    bad = bias; bad.i_tiles = 65535;
    CHECK("ACC capacity", potal_acc_bias_check(&bad) == POTAL_ERR_CAPACITY);

    capture c = {0};
    potal_paired_sink sink = {command, &c};
    potal_paired_session *session = NULL;
    CHECK("open checked sink", potal_paired_session_open(&sink, &session) == POTAL_OK);
    CHECK("issue bias DMA", potal_paired_acc_bias_issue(session, &bias) == POTAL_OK);
    CHECK("pending DMA counted", potal_paired_session_query(session).outstanding_acc_updates == 4);
    CHECK("close waits for bias", potal_paired_session_close(session) == POTAL_ERR_BUSY);
    CHECK("flush waits for bias", potal_paired_tlb_flush(session) == POTAL_ERR_BUSY);
    CHECK("next LOOP waits for bias", potal_paired_loop_ws_issue(session, NULL) == POTAL_ERR_BUSY);
    CHECK("readback waits for bias", potal_paired_acc_readback_issue(session, NULL) == POTAL_ERR_BUSY);
    CHECK("only CONFIG and MVIN emitted", c.count == 5 && c.records[0].funct == POTAL_FUNCT_CONFIG &&
          c.records[1].funct == POTAL_FUNCT_MVIN && c.records[1].xd == 0);
    CHECK("successful fence drains bias", potal_paired_fence(session) == POTAL_OK &&
          potal_paired_session_query(session).outstanding_acc_updates == 0);
    CHECK("close after drain", potal_paired_session_close(session) == POTAL_OK);

    memset(&c, 0, sizeof(c)); c.fail_mvin = 2;
    CHECK("open failure case", potal_paired_session_open(&sink, &session) == POTAL_OK);
    CHECK("partial bias issue fails", potal_paired_acc_bias_issue(session, &bias) == POTAL_HW_BAD_COMMAND);
    CHECK("uncertain prefix retained", potal_paired_session_query(session).outstanding_acc_updates == 2);
    CHECK("sticky failure", potal_paired_scale_write(session, 0, 1, 0, 0) == POTAL_HW_BAD_COMMAND);
    CHECK("cleanup drains partial bias", potal_paired_session_cleanup(session) == POTAL_OK &&
          potal_paired_session_query(session).outstanding_acc_updates == 0);
    CHECK("failed close returns original error", potal_paired_session_close(session) == POTAL_HW_BAD_COMMAND);

    memset(&c, 0, sizeof(c)); c.fail_fence = 1;
    CHECK("open fence-failure case", potal_paired_session_open(&sink, &session) == POTAL_OK);
    CHECK("issue before failed fence", potal_paired_acc_bias_issue(session, &bias) == POTAL_OK);
    CHECK("failed fence", potal_paired_fence(session) == POTAL_HW_BAD_COMMAND);
    const unsigned before = c.count;
    CHECK("failed fence never retried", potal_paired_session_cleanup(session) == POTAL_HW_BAD_COMMAND &&
          c.count == before && potal_paired_session_query(session).outstanding_acc_updates == 4);
    CHECK("close retains unknown ownership", potal_paired_session_close(session) == POTAL_HW_BAD_COMMAND);
    CHECK("unknown accelerator cannot reopen", potal_paired_session_open(&sink, &session) == POTAL_ERR_BUSY);
    printf("POTAL_ATTENTION_API_TEST %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
