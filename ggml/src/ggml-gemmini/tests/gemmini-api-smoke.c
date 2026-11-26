/*
 * RISC-V hardware smoke for the standalone API (cross-compiled, run by hand
 * on a Gemmini target; never part of the native CTest run). Issues one
 * 16x16x16 INT8 LOOP_WS through RoCC, drains, and checks C against a CPU
 * integer GEMM. Exit 0 only when every element matches.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "gemmini.h"

#define N 16

static int8_t  a[N * N] __attribute__((aligned(64)));
static int8_t  b[N * N] __attribute__((aligned(64)));
static int32_t c[N * N] __attribute__((aligned(64)));

static gemmini_status_t issue(const gemmini_transport_t *t, gemmini_status_t s, const gemmini_command_t *cmd) {
    return s != GEMMINI_OK ? s : gemmini_issue(t, cmd, NULL);
}

int main(void) {
    const gemmini_hw_config_t *cfg = gemmini_hw_config_build();
    if (cfg->dim != N) {
        printf("GEMMINI_API_SMOKE SKIP: built for DIM=%u\n", cfg->dim);
        return 0;
    }
    for (int i = 0; i < N * N; ++i) {
        a[i] = (int8_t) (i * 7 - 100);
        b[i] = (int8_t) (i * 13 + 5);
    }
    memset(c, 0, sizeof(c));

    const gemmini_transport_t *t   = gemmini_rocc_transport();
    const uint32_t             one = 0x3f800000u;
    gemmini_command_t          cmd;
    gemmini_status_t           s   = GEMMINI_OK;
    const gemmini_config_ex_desc_t ex = { GEMMINI_DATAFLOW_WS, 0, 0, one, 1, 1, false, false, false };
    gemmini_encode_config_ex(&ex, &cmd);
    s = issue(t, s, &cmd);
    const gemmini_config_st_desc_t st = { N * 4, 0, one };
    gemmini_encode_config_st(&st, &cmd);
    s = issue(t, s, &cmd);
    for (uint32_t id = 0; id < 3; ++id) {
        const gemmini_config_ld_desc_t ld = { id == 2 ? 0 : N, one, false, N, 1, id };
        gemmini_encode_config_ld(&ld, &cmd);
        s = issue(t, s, &cmd);
    }
    gemmini_loop_ws_desc_t loop;
    memset(&loop, 0, sizeof(loop));
    loop.i = loop.j = loop.k = 1;
    loop.a = (uint64_t) (uintptr_t) a;
    loop.b = (uint64_t) (uintptr_t) b;
    loop.c = (uint64_t) (uintptr_t) c;
    loop.a_stride = loop.b_stride = loop.c_stride = N;
    loop.full_c = true;
    gemmini_command_t cmds[GEMMINI_LOOP_WS_COMMANDS];
    gemmini_encode_loop_ws(&loop, cmds);
    if (s == GEMMINI_OK) {
        s = gemmini_issue_all(t, cmds, GEMMINI_LOOP_WS_COMMANDS);
    }
    if (s == GEMMINI_OK) {
        s = gemmini_drain(t);
    }
    if (s != GEMMINI_OK) {
        printf("GEMMINI_API_SMOKE FAIL: %s\n", gemmini_status_name(s));
        return 1;
    }
    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t ref = 0;
            for (int k = 0; k < N; ++k) {
                ref += (int32_t) a[i * N + k] * (int32_t) b[k * N + j];
            }
            mismatches += ref != c[i * N + j];
        }
    }
    printf("GEMMINI_API_SMOKE %s (%d mismatches)\n", mismatches ? "FAIL" : "PASS", mismatches);
    return mismatches ? 1 : 0;
}
