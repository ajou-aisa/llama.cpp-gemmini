/*
 * Bitwise comparison of the standalone encoders with the rs1/rs2 expressions
 * of the legacy SDK macros (gemmini.h develop fcfcf5ce, lines 247-338 and
 * 350-421), copied below verbatim apart from the issue wrapper.
 */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "gemmini.h"

#define ADDR_LEN 32
#define DIM      16

#define SDK_MVIN_RS2(spad_addr, cols, rows) \
    (((uint64_t) (rows) << (ADDR_LEN + 16)) | ((uint64_t) (cols) << ADDR_LEN) | (spad_addr))
#define SDK_MVOUT_RS2(spad_addr, cols, rows) \
    (((uint64_t) (rows) << (ADDR_LEN + 16)) | ((uint64_t) (cols) << ADDR_LEN) | (uint64_t) (spad_addr))
#define SDK_PACK(addr, cols, rows) \
    (((uint64_t) (rows) << (ADDR_LEN + 16)) | ((uint64_t) (cols) << ADDR_LEN) | (uint64_t) (addr))
#define SDK_CONFIG_EX_RS1(dataflow, sys_act, acc_scale_bits, A_stride, A_transpose, B_transpose, set_only_strides) \
    (((uint64_t) (acc_scale_bits) << 32) | ((uint64_t) (A_stride) << 16) | (B_transpose << 9) | (A_transpose << 8) | \
     ((set_only_strides) << 7) | ((sys_act) << 3) | ((dataflow) << 2) | 0)
#define SDK_CONFIG_EX_RS2(C_stride, sys_shift) (((uint64_t) (C_stride) << 48) | (sys_shift))
#define SDK_CONFIG_LD_RS1(scale_bits, shrunk, block_mvin_stride, pixel_repeats, id) \
    (((uint64_t) (scale_bits) << 32) | ((uint64_t) (block_mvin_stride) << 16) | ((uint64_t) (pixel_repeats) << 8) | \
     ((id) << 3) | ((shrunk) << 2) | 1)
#define SDK_CONFIG_ST_RS1(acc_act) (((uint64_t) (acc_act) << 2) | 2)
#define SDK_CONFIG_ST_RS2(stride, acc_scale_bits) (((uint64_t) (acc_scale_bits) << 32) | ((uint32_t) stride))
#define SDK_LOOP_BOUNDS_RS1(pad_I, pad_J, pad_K) (((uint64_t) (pad_K) << 32) | ((uint64_t) (pad_J) << 16) | (uint64_t) (pad_I))
#define SDK_LOOP_BOUNDS_RS2(I, J, K) (((uint64_t) (K) << 32) | ((uint64_t) (J) << 16) | (uint64_t) (I))
#define SDK_LOOP_RS1(a_spad_id, b_spad_id, act, low_D, full_C, ex_accumulate) \
    (((uint64_t) (a_spad_id) << 18) | ((uint64_t) (b_spad_id) << 16) | ((uint64_t) (act) << 8) | ((low_D) << 2) | \
     ((full_C) << 1) | (ex_accumulate))
#define SDK_LOOP_RS2(is_resadd, B_transpose, A_transpose) (((is_resadd) << 2) | ((B_transpose) << 1) | (A_transpose))
#define SDK_COUNTER_CONFIGURE(index, counter_code, non_incremental) \
    ((index & 0x7) << 4 | 0x8 | (counter_code & 0x3f) << 12 | (uint32_t) (non_incremental) << 31)

static int failures = 0;

static void expect(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL %s\n", what);
        failures++;
    }
}

static void expect_cmd(const gemmini_command_t *c, uint32_t funct, uint8_t xd, uint64_t rs1, uint64_t rs2,
                       const char *what) {
    const int ok = c->funct == funct && c->xd == xd && c->xs1 == 1 && c->xs2 == 1 && c->rs1 == rs1 && c->rs2 == rs2;
    if (!ok) {
        fprintf(stderr, "FAIL %s: funct=%u xd=%u rs1=0x%016" PRIx64 " rs2=0x%016" PRIx64
                        " expected funct=%u xd=%u rs1=0x%016" PRIx64 " rs2=0x%016" PRIx64 "\n",
                what, c->funct, c->xd, c->rs1, c->rs2, funct, xd, rs1, rs2);
        failures++;
    }
}

static void test_moves(void) {
    static const uint32_t funct[3] = { GEMMINI_FUNCT_MVIN, GEMMINI_FUNCT_MVIN2, GEMMINI_FUNCT_MVIN3 };
    const uint32_t        addrs[]  = { 0, 17, 0x3fff, GEMMINI_LOCAL_ACC | 5, GEMMINI_LOCAL_ACC | GEMMINI_LOCAL_ACCUMULATE | 9 };
    const uint32_t        shapes[][2] = { { 16, 16 }, { 1, 1 }, { 64, 16 }, { 13, 7 }, { 0xffff, 0xffff } };
    gemmini_command_t     c;
    for (uint32_t id = 0; id < 3; ++id) {
        for (size_t a = 0; a < sizeof(addrs) / sizeof(addrs[0]); ++a) {
            for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
                const gemmini_mvin_desc_t d = { id, 0x80001000u + a, addrs[a], shapes[s][0], shapes[s][1] };
                expect(gemmini_encode_mvin(&d, &c) == GEMMINI_OK, "mvin encodes");
                expect_cmd(&c, funct[id], 0, 0x80001000u + a, SDK_MVIN_RS2(addrs[a], shapes[s][0], shapes[s][1]), "mvin");
            }
        }
    }
    const gemmini_mvout_desc_t o = { 0xdeadbeef00ull, GEMMINI_LOCAL_ACC | GEMMINI_LOCAL_READ_FULL | 12, 48, 9 };
    expect(gemmini_encode_mvout(&o, &c) == GEMMINI_OK, "mvout encodes");
    expect_cmd(&c, GEMMINI_FUNCT_MVOUT, 0, 0xdeadbeef00ull, SDK_MVOUT_RS2(o.local_addr, 48, 9), "mvout");

    const gemmini_mvin_desc_t bad_id = { 3, 0, 0, 1, 1 };
    expect(gemmini_encode_mvin(&bad_id, &c) == GEMMINI_ERR_INVALID_ARGUMENT, "mvin rejects id 3");
    const gemmini_mvin_desc_t wide = { 0, 0, 0, 0x10000, 1 };
    expect(gemmini_encode_mvin(&wide, &c) == GEMMINI_ERR_INVALID_ARGUMENT, "mvin rejects 17-bit cols");
    expect(gemmini_encode_mvin(NULL, &c) == GEMMINI_ERR_INVALID_ARGUMENT, "mvin rejects NULL");
}

static void test_compute(void) {
    gemmini_command_t c;
    const gemmini_preload_desc_t p = { GEMMINI_LOCAL_GARBAGE, GEMMINI_LOCAL_ACC | 32, 16, 16, 7, 3 };
    expect(gemmini_encode_preload(&p, &c) == GEMMINI_OK, "preload encodes");
    expect_cmd(&c, GEMMINI_FUNCT_PRELOAD, 0, SDK_PACK(GEMMINI_LOCAL_GARBAGE, 16, 16), SDK_PACK(GEMMINI_LOCAL_ACC | 32, 7, 3),
               "preload");
    for (int acc = 0; acc < 2; ++acc) {
        const gemmini_compute_desc_t d = { acc != 0, 128, GEMMINI_LOCAL_GARBAGE, 16, 11, DIM, DIM };
        expect(gemmini_encode_compute(&d, &c) == GEMMINI_OK, "compute encodes");
        expect_cmd(&c, acc ? GEMMINI_FUNCT_COMPUTE_ACCUMULATE : GEMMINI_FUNCT_COMPUTE_PRELOADED, 0, SDK_PACK(128, 16, 11),
                   SDK_PACK(GEMMINI_LOCAL_GARBAGE, DIM, DIM), "compute");
    }
}

static void test_config(void) {
    gemmini_command_t c;
    const uint32_t    one = 0x3f800000u;
    for (int at = 0; at < 2; ++at) {
        for (int bt = 0; bt < 2; ++bt) {
            const gemmini_config_ex_desc_t ex = { GEMMINI_DATAFLOW_WS, 1, 0, one, 1, 1, at != 0, bt != 0, false };
            expect(gemmini_encode_config_ex(&ex, &c) == GEMMINI_OK, "config_ex encodes");
            expect_cmd(&c, GEMMINI_FUNCT_CONFIG, 0, SDK_CONFIG_EX_RS1(1, 1, one, 1, at, bt, 0), SDK_CONFIG_EX_RS2(1, 0),
                       "config_ex");
        }
    }
    const gemmini_config_ex_desc_t ex_os = { GEMMINI_DATAFLOW_OS, 3, 7, 0x40000000u, 0xffff, 0xabcd, false, false, true };
    expect(gemmini_encode_config_ex(&ex_os, &c) == GEMMINI_OK, "config_ex(OS) encodes");
    expect_cmd(&c, GEMMINI_FUNCT_CONFIG, 0, SDK_CONFIG_EX_RS1(0, 3, 0x40000000u, 0xabcd, 0, 0, 1), SDK_CONFIG_EX_RS2(0xffff, 7),
               "config_ex OS");
    for (uint32_t id = 0; id < 3; ++id) {
        const gemmini_config_ld_desc_t ld = { 4096 + id, one, id == 2, DIM, 1, id };
        expect(gemmini_encode_config_ld(&ld, &c) == GEMMINI_OK, "config_ld encodes");
        expect_cmd(&c, GEMMINI_FUNCT_CONFIG, 0, SDK_CONFIG_LD_RS1(one, (id == 2), DIM, 1, id), 4096 + id, "config_ld");
    }
    const gemmini_config_st_desc_t st = { 0x12345678u, 1, one };
    expect(gemmini_encode_config_st(&st, &c) == GEMMINI_OK, "config_st encodes");
    expect_cmd(&c, GEMMINI_FUNCT_CONFIG, 0, SDK_CONFIG_ST_RS1(1), SDK_CONFIG_ST_RS2(0x12345678u, one), "config_st");

    const gemmini_config_ex_desc_t bad_act = { GEMMINI_DATAFLOW_WS, 16, 0, one, 1, 1, false, false, false };
    expect(gemmini_encode_config_ex(&bad_act, &c) == GEMMINI_ERR_INVALID_ARGUMENT, "config_ex rejects activation 16");
    const gemmini_config_ld_desc_t bad_ld = { 0, one, false, 0x10000, 1, 0 };
    expect(gemmini_encode_config_ld(&bad_ld, &c) == GEMMINI_ERR_INVALID_ARGUMENT, "config_ld rejects 17-bit stride");
    const gemmini_config_st_desc_t bad_st = { 0, 4, one };
    expect(gemmini_encode_config_st(&bad_st, &c) == GEMMINI_ERR_INVALID_ARGUMENT, "config_st rejects activation 4");

    const gemmini_flush_desc_t f = { true };
    expect(gemmini_encode_flush(&f, &c) == GEMMINI_OK, "flush encodes");
    expect_cmd(&c, GEMMINI_FUNCT_FLUSH, 0, 1, 0, "flush");
}

static void test_loop_ws(void) {
    gemmini_command_t      cmds[GEMMINI_LOOP_WS_COMMANDS];
    gemmini_loop_ws_desc_t d;
    memset(&d, 0, sizeof(d));
    d.i = 5; d.j = 0xffff; d.k = 51;
    d.pad_i = 3; d.pad_j = 15; d.pad_k = 0;
    d.a = 0x7f0000001000ull; d.b = 0; d.d = 0x7f0000002000ull; d.c = 0x7f0000003000ull;
    d.a_stride = 4096; d.b_stride = 11008; d.d_stride = 0; d.c_stride = 4096;
    d.a_transpose = false; d.b_transpose = true; d.full_c = true; d.low_d = true; d.ex_accumulate = true;
    d.activation = 1; d.a_spad_id = 2; d.b_spad_id = 1; d.is_resadd = false;
    expect(gemmini_encode_loop_ws(&d, cmds) == GEMMINI_OK, "loop_ws encodes");
    expect_cmd(&cmds[0], GEMMINI_FUNCT_LOOP_WS_CONFIG_BOUNDS, 0, SDK_LOOP_BOUNDS_RS1(3, 15, 0), SDK_LOOP_BOUNDS_RS2(5, 0xffff, 51), "bounds");
    expect_cmd(&cmds[1], GEMMINI_FUNCT_LOOP_WS_CONFIG_ADDRS_AB, 0, d.a, d.b, "addrs_ab");
    expect_cmd(&cmds[2], GEMMINI_FUNCT_LOOP_WS_CONFIG_ADDRS_DC, 0, d.d, d.c, "addrs_dc");
    expect_cmd(&cmds[3], GEMMINI_FUNCT_LOOP_WS_CONFIG_STRIDES_AB, 0, 4096, 11008, "strides_ab");
    expect_cmd(&cmds[4], GEMMINI_FUNCT_LOOP_WS_CONFIG_STRIDES_DC, 0, 0, 4096, "strides_dc");
    expect_cmd(&cmds[5], GEMMINI_FUNCT_LOOP_WS, 0, SDK_LOOP_RS1(2, 1, 1, 1, 1, 1), SDK_LOOP_RS2(0, 1, 0), "loop_ws");

    d.i = 0x10000;
    expect(gemmini_encode_loop_ws(&d, cmds) == GEMMINI_ERR_INVALID_ARGUMENT, "loop_ws rejects 17-bit I");
    d.i = 1; d.a_spad_id = 4;
    expect(gemmini_encode_loop_ws(&d, cmds) == GEMMINI_ERR_INVALID_ARGUMENT, "loop_ws rejects spad id 4");
}

static void test_counter(void) {
    gemmini_command_t c;
    const gemmini_counter_desc_t rd = { GEMMINI_COUNTER_OP_READ, 3, 0 };
    expect(gemmini_encode_counter(&rd, &c) == GEMMINI_OK, "counter read encodes");
    expect_cmd(&c, GEMMINI_FUNCT_COUNTER, 1, (3 & 0x7) << 4, 0, "counter read");
    const uint32_t events[] = { GEMMINI_EVENT_LOAD_ACTIVE_CYCLE, GEMMINI_EVENT_LOOP_MATMUL_ACTIVE_CYCLES,
                                GEMMINI_EVENT_TRANSPOSE_PRELOAD_UNROLLER_ACTIVE_CYCLES, GEMMINI_EVENT_RDMA_BYTES_REC };
    for (size_t i = 0; i < sizeof(events) / sizeof(events[0]); ++i) {
        const gemmini_counter_desc_t cfg = { GEMMINI_COUNTER_OP_CONFIGURE, (uint32_t) i, events[i] };
        size_t code = events[i];
        const int non_incremental = code > GEMMINI_COUNTER_INCREMENTAL_EVENTS;
        if (non_incremental) {
            code -= GEMMINI_COUNTER_INCREMENTAL_EVENTS;
        }
        expect(gemmini_encode_counter(&cfg, &c) == GEMMINI_OK, "counter configure encodes");
        expect_cmd(&c, GEMMINI_FUNCT_COUNTER, 1, (uint32_t) SDK_COUNTER_CONFIGURE(i, code, non_incremental), 0,
                   "counter configure");
    }
    const gemmini_counter_desc_t take = { GEMMINI_COUNTER_OP_SNAPSHOT_TAKE, 0, 0 };
    expect(gemmini_encode_counter(&take, &c) == GEMMINI_OK && c.rs1 == 0x4, "snapshot take");
    const gemmini_counter_desc_t reset = { GEMMINI_COUNTER_OP_SNAPSHOT_RESET, 0, 0 };
    expect(gemmini_encode_counter(&reset, &c) == GEMMINI_OK && c.rs1 == 0x2, "snapshot reset");
    const gemmini_counter_desc_t module = { GEMMINI_COUNTER_OP_RESET, 0, 0 };
    expect(gemmini_encode_counter(&module, &c) == GEMMINI_OK && c.rs1 == 0x1, "module reset");
    const gemmini_counter_desc_t bad_slot = { GEMMINI_COUNTER_OP_READ, 8, 0 };
    expect(gemmini_encode_counter(&bad_slot, &c) == GEMMINI_ERR_INVALID_ARGUMENT, "counter rejects slot 8");
    const gemmini_counter_desc_t bad_event = { GEMMINI_COUNTER_OP_CONFIGURE, 0, GEMMINI_EVENT_WDMA_TOTAL_LATENCY + 1 };
    expect(gemmini_encode_counter(&bad_event, &c) == GEMMINI_ERR_INVALID_ARGUMENT, "counter rejects unknown event");
}

static void test_responses(void) {
    expect(gemmini_funct_has_response(GEMMINI_FUNCT_COUNTER) == 1, "COUNTER responds");
    for (uint32_t f = 0; f <= GEMMINI_FUNCT_MVIN3; ++f) {
        expect(gemmini_funct_is_supported(f) == 1, "funct 0..14 supported");
        expect(gemmini_funct_has_response(f) == 0, "funct 0..14 has no response");
    }
    expect(gemmini_funct_is_supported(15) == 0, "LOOP_CONV_WS is not exposed");
    expect(gemmini_funct_is_supported(23) == 0, "PoTAL SCALE_WRITE is not a Gemmini command");
}

int main(void) {
    test_moves();
    test_compute();
    test_config();
    test_loop_ws();
    test_counter();
    test_responses();
    if (failures) {
        fprintf(stderr, "gemmini-encoding-test: %d failure(s)\n", failures);
        return 1;
    }
    printf("GEMMINI_ENCODING_TEST PASS\n");
    return 0;
}
