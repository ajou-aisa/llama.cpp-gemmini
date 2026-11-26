#include <stdio.h>
#include <string.h>

#include "gemmini.h"
#include "npu_config.h"

static int failures = 0;

static void expect(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "FAIL %s\n", what);
        failures++;
    }
}

static gemmini_hw_config_t d16(void) {
    const gemmini_hw_config_t c = { 16, 8, 1, 4, 4, 4096, 2, 1024, 64 };
    return c;
}

static void test_build_config(void) {
    const gemmini_hw_config_t *b = gemmini_hw_config_build();
    expect(b->dim == NPU_D && b->spad_banks == NPU_SPAD_BANKS && b->spad_bank_rows == NPU_SPAD_BANK_ROWS &&
               b->acc_banks == NPU_ACC_BANKS && b->acc_rows == NPU_ACC_ROWS && b->dma_max_bytes == NPU_DMA_MAX_BYTES,
           "build config mirrors npu_config.h");
    expect(gemmini_hw_config_check(b) == GEMMINI_OK, "build config is valid");
    expect(NPU_ACC_BANK_ROWS * NPU_ACC_BANKS == NPU_ACC_ROWS, "NPU_ACC_ROWS is the total over banks");
}

static void test_config_checks(void) {
    gemmini_hw_config_t c = d16();
    expect(gemmini_hw_config_check(&c) == GEMMINI_OK, "D16 reference config");
    expect(gemmini_spad_rows(&c) == 16384, "spad rows");
    expect(gemmini_max_block_len(&c) == 4 && gemmini_max_block_len_acc(&c) == 1, "MAX_BLOCK_LEN(_ACC)");
    c.acc_rows = 1023;
    expect(gemmini_hw_config_check(&c) == GEMMINI_ERR_INVALID_ARGUMENT, "ACC rows not divisible by banks");
    c = d16();
    c.dim = 24;
    expect(gemmini_hw_config_check(&c) == GEMMINI_ERR_INVALID_ARGUMENT, "non power-of-two D");
    c = d16();
    c.input_bits = 4;
    expect(gemmini_hw_config_check(&c) == GEMMINI_ERR_INVALID_ARGUMENT, "Gemmini INT4 has no RTL ABI");
    c = d16();
    c.dim = 32;
    expect(gemmini_max_block_len(&c) == 2 && gemmini_max_block_len_acc(&c) == 1 && gemmini_hw_config_check(&c) == GEMMINI_OK,
           "D32: ACC row wider than MAX_BYTES clamps to one block (RTL header rule)");
    c.spad_bank_rows = 1u << 24;
    expect(gemmini_hw_config_check(&c) == GEMMINI_ERR_UNSUPPORTED, "SPAD rows beyond the 25-bit local address");
    expect(gemmini_hw_config_check(NULL) == GEMMINI_ERR_INVALID_ARGUMENT, "NULL config");
}

static void test_validation(void) {
    const gemmini_hw_config_t c = d16();
    gemmini_mvin_desc_t m = { 0, 0x1000, 0, 64, 16 };
    expect(gemmini_validate_mvin(&c, &m) == GEMMINI_OK, "4 blocks of 16 in SPAD");
    m.cols = 65;
    expect(gemmini_validate_mvin(&c, &m) == GEMMINI_ERR_CAPACITY, "SPAD move wider than MAX_BLOCK_LEN");
    m.cols = 16;
    m.rows = 17;
    expect(gemmini_validate_mvin(&c, &m) == GEMMINI_ERR_INVALID_ARGUMENT, "more rows than DIM");
    m.rows = 16;
    m.local_addr = 16384 - 16;
    expect(gemmini_validate_mvin(&c, &m) == GEMMINI_OK, "last SPAD rows");
    m.local_addr = 16384 - 15;
    expect(gemmini_validate_mvin(&c, &m) == GEMMINI_ERR_CAPACITY, "past the SPAD end");
    m.local_addr = GEMMINI_LOCAL_ACC | (1024 - 16);
    expect(gemmini_validate_mvin(&c, &m) == GEMMINI_OK, "last ACC rows");
    m.cols = 32;
    expect(gemmini_validate_mvin(&c, &m) == GEMMINI_ERR_CAPACITY, "ACC move wider than MAX_BLOCK_LEN_ACC");
    m.cols = 16;
    m.local_addr = GEMMINI_LOCAL_ACC | 1024;
    expect(gemmini_validate_mvin(&c, &m) == GEMMINI_ERR_CAPACITY, "past the ACC end");
    m.local_addr = GEMMINI_LOCAL_ACCUMULATE | 4;
    expect(gemmini_validate_mvin(&c, &m) == GEMMINI_ERR_INVALID_ARGUMENT, "accumulate flag on a SPAD address");
    gemmini_mvout_desc_t o = { 0x2000, GEMMINI_LOCAL_ACC | GEMMINI_LOCAL_ACCUMULATE, 16, 16 };
    expect(gemmini_validate_mvout(&c, &o) == GEMMINI_ERR_INVALID_ARGUMENT, "mvout cannot accumulate");
    o.local_addr = GEMMINI_LOCAL_ACC | GEMMINI_LOCAL_READ_FULL;
    expect(gemmini_validate_mvout(&c, &o) == GEMMINI_OK, "full-width ACC read");

    gemmini_loop_ws_desc_t l;
    memset(&l, 0, sizeof(l));
    l.i = 5; l.j = 6; l.k = 33;
    expect(gemmini_validate_loop_ws(&c, &l) == GEMMINI_OK, "tile fits half SPAD and ACC");
    l.k = 128;
    expect(gemmini_validate_loop_ws(&c, &l) == GEMMINI_ERR_CAPACITY, "tile exceeds half SPAD");
    l.k = 1; l.i = 6; l.j = 6;
    expect(gemmini_validate_loop_ws(&c, &l) == GEMMINI_ERR_CAPACITY, "tile exceeds half ACC");
    l.i = 1; l.j = 1; l.pad_i = 16;
    expect(gemmini_validate_loop_ws(&c, &l) == GEMMINI_ERR_INVALID_ARGUMENT, "padding must be below DIM");
    l.pad_i = 0; l.a_transpose = true; l.b_transpose = true;
    expect(gemmini_validate_loop_ws(&c, &l) == GEMMINI_ERR_INVALID_ARGUMENT, "WS cannot transpose both operands");
}

static void test_status_names(void) {
    expect(strcmp(gemmini_status_name(GEMMINI_ERR_NO_HARDWARE), "GEMMINI_ERR_NO_HARDWARE") == 0, "status name");
    expect(strcmp(gemmini_status_name((gemmini_status_t) 42), "GEMMINI_STATUS_UNKNOWN") == 0, "unknown status");
}

int main(void) {
    test_build_config();
    test_config_checks();
    test_validation();
    test_status_names();
    if (failures) {
        fprintf(stderr, "gemmini-api-test: %d failure(s)\n", failures);
        return 1;
    }
    printf("GEMMINI_API_TEST PASS\n");
    return 0;
}
