/*
 * Host-runnable PoTal API checks: field packing, validation and capacity.
 *
 * Expected values are literals written from the RTL bit layouts
 * (PoTalScaleCommandIngress.scala, LoopMatmul.scala, GemminiISA.scala) and the
 * passing bare-metal tests; they are never computed with the API itself.
 * Hardware entry points are exercised only on non-RISC-V hosts, where they
 * must validate first and then return POTAL_ERR_NO_HARDWARE.
 *
 * Output: "PASS <name>" / "FAIL <name>", final "POTAL_API_TEST PASS|FAIL".
 */
#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__unix__)
#include <errno.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "potal.h"
#include "potal_hw_params.h"

static int failures;

static void check_u64(const char *name, uint64_t actual, uint64_t expected) {
    if (actual != expected) {
        printf("FAIL %s: got=0x%016" PRIx64 " expected=0x%016" PRIx64 "\n", name, actual, expected);
        failures++;
    } else {
        printf("PASS %s\n", name);
    }
}

static void check_status(const char *name, potal_status actual, potal_status expected) {
    if (actual != expected) {
        printf("FAIL %s: got=%s expected=%s\n", name, potal_status_name(actual),
               potal_status_name(expected));
        failures++;
    } else {
        printf("PASS %s\n", name);
    }
}

static void test_status_codes(void) {
    /* PoTalScaleCommandStatus values. */
    check_u64("status OK", POTAL_OK, 0);
    check_u64("status INVALID_COLUMN", POTAL_HW_INVALID_COLUMN, 1);
    check_u64("status INVALID_SCALE", POTAL_HW_INVALID_SCALE, 2);
    check_u64("status SCALE_CONFLICT", POTAL_HW_SCALE_CONFLICT, 3);
    check_u64("status RELEASE_MISS", POTAL_HW_RELEASE_MISS, 4);
    check_u64("status INVALID_ADDRESS", POTAL_HW_INVALID_ADDRESS, 5);
    check_u64("status INVALID_ENCODING", POTAL_HW_INVALID_ENCODING, 6);
    check_u64("status BAD_COMMAND", POTAL_HW_BAD_COMMAND, 7);
}

static void test_scale_key(void) {
    uint64_t rs1 = 0xdeadbeef;

    check_status("key zero", potal_scale_key_encode(0, 0, 0, &rs1), POTAL_OK);
    check_u64("key zero value", rs1, 0x0000000000000000ull);
    check_status("key slot3 gen7 col0", potal_scale_key_encode(3, 7, 0, &rs1), POTAL_OK);
    check_u64("key slot3 gen7 col0 value", rs1, 0x0000000000000703ull);

    /* Last legal column: rs1[21:16] = DIM-1. */
#if POTAL_HW_DIM == 16
    const uint64_t max_key = 0x00000000000FFFFFull;
#elif POTAL_HW_DIM == 32
    const uint64_t max_key = 0x00000000001FFFFFull;
#else
    const uint64_t max_key = 0x00000000003FFFFFull;
#endif
    check_status("key max fields", potal_scale_key_encode(255, 255, POTAL_HW_DIM - 1, &rs1), POTAL_OK);
    check_u64("key max fields value", rs1, max_key);

    rs1 = 0x1234;
    check_status("key slot 256", potal_scale_key_encode(256, 0, 0, &rs1), POTAL_ERR_INVALID_ARGUMENT);
    check_status("key generation 256", potal_scale_key_encode(0, 256, 0, &rs1), POTAL_ERR_INVALID_ARGUMENT);
    check_status("key column DIM", potal_scale_key_encode(0, 0, POTAL_HW_DIM, &rs1), POTAL_ERR_INVALID_ARGUMENT);
    check_status("key column 64 (field wrap)", potal_scale_key_encode(0, 0, 64, &rs1), POTAL_ERR_INVALID_ARGUMENT);
    check_status("key slot 0x10000003 (no truncation)", potal_scale_key_encode(0x10000003u, 0, 0, &rs1),
                 POTAL_ERR_INVALID_ARGUMENT);
    check_u64("key rejected leaves output", rs1, 0x1234);
    check_status("key null output", potal_scale_key_encode(0, 0, 0, NULL), POTAL_ERR_INVALID_ARGUMENT);
}

static void test_carrier(void) {
    check_u64("carrier 0 (x1)", potal_carrier_is_valid(0), 1);
    check_u64("carrier 1", potal_carrier_is_valid(1), 1);
    check_u64("carrier 2", potal_carrier_is_valid(2), 1);
    check_u64("carrier 32767", potal_carrier_is_valid(32767), 1);
    check_u64("carrier zero 0x80000000", potal_carrier_is_valid(0x80000000u), 1);
    check_u64("carrier 32768", potal_carrier_is_valid(32768), 0);
    check_u64("carrier 0x7fffffff", potal_carrier_is_valid(0x7fffffffu), 0);
    check_u64("carrier 0x80000001", potal_carrier_is_valid(0x80000001u), 0);
    check_u64("carrier 0xffffffff", potal_carrier_is_valid(0xffffffffu), 0);
}

#define FAKE_A ((const int8_t *)(uintptr_t)0x10000)
#define FAKE_B ((const int8_t *)(uintptr_t)0x20000)
#define FAKE_C ((int32_t *)(uintptr_t)0x30000)

static potal_gemm_ws k32_gemm(void) {
    potal_gemm_ws g = {
        .a = FAKE_A, .b = FAKE_B, .c = FAKE_C,
        .lda = 2 * POTAL_HW_DIM, .ldb = POTAL_HW_DIM, .ldc = POTAL_HW_DIM,
        .m = POTAL_HW_DIM, .n = POTAL_HW_DIM, .k = 2 * POTAL_HW_DIM,
        .scale_base = 200, .scale_generation = 0xAB, .passes_per_block = 2,
        .flags = 0,
    };
    return g;
}

#if POTAL_HW_DIM == 16
static void test_gemm_encode_d16(void) {
    potal_gemm_ws_program p;
    potal_gemm_ws g = k32_gemm();

    check_status("encode K32 ppb2", potal_gemm_ws_encode(&g, &p), POTAL_OK);
    /* CONFIG_LD: block stride 16 << 16, pixel_repeats 1 << 8, state << 3, LD = 1. */
    check_u64("config_ld_a rs1", p.config_ld_a.rs1, 0x0000000000100101ull);
    check_u64("config_ld_a rs2 (A row bytes)", p.config_ld_a.rs2, 32);
    check_u64("config_ld_b rs1", p.config_ld_b.rs1, 0x0000000000100109ull);
    check_u64("config_ld_b rs2 (B row bytes)", p.config_ld_b.rs2, 16);
    /* CONFIG_EX: a_stride 1 << 16, WS << 2; rs2 c_stride 1 << 48. */
    check_u64("config_ex rs1", p.config_ex.rs1, 0x0000000000010004ull);
    check_u64("config_ex rs2", p.config_ex.rs2, 0x0001000000000000ull);
    check_u64("config_st rs1", p.config_st.rs1, 2);
    check_u64("config_st rs2 (C row bytes)", p.config_st.rs2, 64);
    /* BOUNDS: pads 0; K=2 << 32, J=1 << 16, I=1. */
    check_u64("bounds rs1", p.bounds.rs1, 0);
    check_u64("bounds rs2", p.bounds.rs2, 0x0000000200010001ull);
    check_u64("addrs_ab rs1", p.addrs_ab.rs1, 0x10000);
    check_u64("addrs_ab rs2", p.addrs_ab.rs2, 0x20000);
    check_u64("addrs_dc rs1 (no D)", p.addrs_dc.rs1, 0);
    check_u64("addrs_dc rs2", p.addrs_dc.rs2, 0x30000);
    check_u64("strides_ab rs1", p.strides_ab.rs1, 32);
    check_u64("strides_ab rs2", p.strides_ab.rs2, 16);
    check_u64("strides_dc rs1", p.strides_dc.rs1, 0);
    check_u64("strides_dc rs2", p.strides_dc.rs2, 16);
    /* LOOP_WS: full_C bit1, enable bit20, base 200 @21, gen 0xAB @37, ppb 2 @45. */
    check_u64("loop_ws rs1 K32 ppb2", p.loop_ws.rs1, 0x0000556019100002ull);
    check_u64("loop_ws rs2 (no transpose/resadd)", p.loop_ws.rs2, 0);

    /* Same LOOP_WS rs1 as potal_gemm_d16.c: base 0, gen 1, ppb 1. */
    g.k = 16; g.lda = 16; g.scale_base = 0; g.scale_generation = 1; g.passes_per_block = 1;
    check_status("encode K16", potal_gemm_ws_encode(&g, &p), POTAL_OK);
    check_u64("loop_ws rs1 K16 (d16 reference)", p.loop_ws.rs1, 0x0000202000100002ull);
    check_u64("bounds rs2 K16", p.bounds.rs2, 0x0000000100010001ull);

    /* Scale slots: base + block * J + j. */
    uint32_t slot = 0;
    g = k32_gemm();
    g.scale_base = 8; g.passes_per_block = 1;
    check_status("slot block1", potal_gemm_ws_scale_slot(&g, 1, 0, &slot), POTAL_OK);
    check_u64("slot block1 value", slot, 9);
    check_status("slot block2 out of range", potal_gemm_ws_scale_slot(&g, 2, 0, &slot), POTAL_ERR_INVALID_ARGUMENT);
    check_status("slot n_tile1 out of range", potal_gemm_ws_scale_slot(&g, 0, 1, &slot), POTAL_ERR_INVALID_ARGUMENT);
    g.passes_per_block = 2;
    check_status("slot shared block1 out of range", potal_gemm_ws_scale_slot(&g, 1, 0, &slot), POTAL_ERR_INVALID_ARGUMENT);

    /* Scale window edge: 255 + one slot fits, 255 + two slots does not. */
    g = k32_gemm();
    g.scale_base = 255; g.passes_per_block = 2;
    check_status("scale base 255 one block", potal_gemm_ws_check(&g), POTAL_OK);
    g.passes_per_block = 1;
    check_status("scale base 255 two blocks", potal_gemm_ws_check(&g), POTAL_ERR_CAPACITY);
    g.scale_base = 256; g.passes_per_block = 2;
    check_status("scale base 256", potal_gemm_ws_check(&g), POTAL_ERR_CAPACITY);
    g = k32_gemm();
    g.n = 32; g.ldb = 32; g.ldc = 32; g.scale_base = 255;
    check_status("scale base 255 J=2", potal_gemm_ws_check(&g), POTAL_ERR_CAPACITY);

    /* Hardware capacity is reported before the validated-scope restriction. */
    g = k32_gemm();
    g.m = 144; g.n = 144; g.ldb = 144; g.ldc = 144; g.scale_base = 0;
    check_status("work ids 81 tiles", potal_gemm_ws_check(&g), POTAL_ERR_CAPACITY);
    /* Per-loop SPAD half = 4 banks * 4096 rows / 2 = 8192 rows; (258 + 258) * 16 = 8256. */
    g = k32_gemm();
    g.k = 16 * 258; g.lda = g.k; g.passes_per_block = 2; g.scale_base = 0;
    g.b = (const int8_t *)(uintptr_t)0x100000; g.c = (int32_t *)(uintptr_t)0x200000;
    check_status("SPAD capacity K=258 tiles", potal_gemm_ws_check(&g), POTAL_ERR_CAPACITY);
    g.k = 16 * 256; g.lda = g.k;
    check_status("SPAD K=256 tiles fits, then unvalidated", potal_gemm_ws_check(&g), POTAL_ERR_UNSUPPORTED);

    /* RTL-legal but outside the validated scope. */
    g = k32_gemm();
    g.m = 32; g.passes_per_block = 1;
    check_status("M=32 unsupported", potal_gemm_ws_check(&g), POTAL_ERR_UNSUPPORTED);
    g = k32_gemm();
    g.m = 8;
    check_status("M=8 padding unsupported", potal_gemm_ws_check(&g), POTAL_ERR_UNSUPPORTED);
    g = k32_gemm();
    g.k = 24; g.passes_per_block = 1;
    check_status("K=24 padding unsupported", potal_gemm_ws_check(&g), POTAL_ERR_UNSUPPORTED);
    g = k32_gemm();
    g.k = 48; g.lda = 48; g.passes_per_block = 1;
    check_status("K=48 unvalidated", potal_gemm_ws_check(&g), POTAL_ERR_UNSUPPORTED);
}
#elif POTAL_HW_DIM == 32
static void test_gemm_encode_d16(void) {
    potal_gemm_ws g = k32_gemm();
    potal_gemm_ws_program p;
    check_status("encode DIM32 K64 ppb2", potal_gemm_ws_encode(&g, &p), POTAL_OK);
    /* CONFIG_LD block stride 32 @16, repeats 1 @8, state @3. */
    check_u64("d32 config_ld_a rs1", p.config_ld_a.rs1, 0x0000000000200101ull);
    check_u64("d32 config_ld_b rs1", p.config_ld_b.rs1, 0x0000000000200109ull);
    check_u64("d32 config_ld_a rs2", p.config_ld_a.rs2, 64);
    check_u64("d32 config_ld_b rs2", p.config_ld_b.rs2, 32);
    check_u64("d32 config_ex rs1", p.config_ex.rs1, 0x0000000000010004ull);
    check_u64("d32 config_ex rs2", p.config_ex.rs2, 0x0001000000000000ull);
    check_u64("d32 config_st rs1", p.config_st.rs1, 2);
    check_u64("d32 config_st rs2", p.config_st.rs2, 128);
    check_u64("d32 bounds rs1", p.bounds.rs1, 0);
    check_u64("d32 bounds rs2", p.bounds.rs2, 0x0000000200010001ull);
    check_u64("d32 addrs_ab rs1", p.addrs_ab.rs1, 0x10000);
    check_u64("d32 addrs_ab rs2", p.addrs_ab.rs2, 0x20000);
    check_u64("d32 addrs_dc rs1", p.addrs_dc.rs1, 0);
    check_u64("d32 addrs_dc rs2", p.addrs_dc.rs2, 0x30000);
    check_u64("d32 strides_ab rs1", p.strides_ab.rs1, 64);
    check_u64("d32 strides_ab rs2", p.strides_ab.rs2, 32);
    check_u64("d32 strides_dc rs1", p.strides_dc.rs1, 0);
    check_u64("d32 strides_dc rs2", p.strides_dc.rs2, 32);
    check_u64("d32 loop_ws rs1 ppb2", p.loop_ws.rs1, 0x0000556019100002ull);
    check_u64("d32 loop_ws rs2", p.loop_ws.rs2, 0);
    g.k = g.lda = 32;
    g.scale_base = 0; g.scale_generation = 1; g.passes_per_block = 1;
    check_status("encode DIM32 K32", potal_gemm_ws_encode(&g, &p), POTAL_OK);
    check_u64("d32 K32 bounds", p.bounds.rs2, 0x0000000100010001ull);
    check_u64("d32 K32 loop_ws", p.loop_ws.rs1, 0x0000202000100002ull);
    uint32_t slot = 0;
    g = k32_gemm(); g.scale_base = 8; g.passes_per_block = 1;
    check_status("d32 block1 slot", potal_gemm_ws_scale_slot(&g, 1, 0, &slot), POTAL_OK);
    check_u64("d32 block1 slot value", slot, 9);
    check_status("d32 block2 refused", potal_gemm_ws_scale_slot(&g, 2, 0, &slot), POTAL_ERR_INVALID_ARGUMENT);
    check_status("d32 J1 refused", potal_gemm_ws_scale_slot(&g, 0, 1, &slot), POTAL_ERR_INVALID_ARGUMENT);
    g.passes_per_block = 2;
    check_status("d32 shared block1 refused", potal_gemm_ws_scale_slot(&g, 1, 0, &slot), POTAL_ERR_INVALID_ARGUMENT);
    g.scale_base = 255;
    check_status("d32 final Scale slot fits", potal_gemm_ws_check(&g), POTAL_OK);
    g.passes_per_block = 1;
    check_status("d32 two Scale slots overflow", potal_gemm_ws_check(&g), POTAL_ERR_CAPACITY);
    g.scale_base = 256;
    check_status("d32 Scale base overflow", potal_gemm_ws_check(&g), POTAL_ERR_CAPACITY);
    g = k32_gemm(); g.n = g.ldb = g.ldc = 64; g.scale_base = 255;
    check_status("d32 J2 Scale overflow", potal_gemm_ws_check(&g), POTAL_ERR_CAPACITY);
    /* ACC half holds 8 DIM32 tiles: a 3x3 output needs 9. */
    g = k32_gemm(); g.m = g.n = g.ldb = g.ldc = 96; g.scale_base = 0;
    check_status("d32 ACC 9 tiles overflow", potal_gemm_ws_check(&g), POTAL_ERR_CAPACITY);
    /* SPAD half has 4096 rows: (64+64)*32 fits; (66+66)*32 does not. */
    g = k32_gemm(); g.k = g.lda = 32 * 66; g.scale_base = 0;
    g.b = (const int8_t *)(uintptr_t)0x100000; g.c = (int32_t *)(uintptr_t)0x200000;
    check_status("d32 SPAD 66 K tiles overflow", potal_gemm_ws_check(&g), POTAL_ERR_CAPACITY);
    g.k = g.lda = 32 * 64;
    check_status("d32 SPAD edge then unvalidated", potal_gemm_ws_check(&g), POTAL_ERR_UNSUPPORTED);
    g = k32_gemm(); g.m = 64;
    check_status("d32 M64 unvalidated", potal_gemm_ws_check(&g), POTAL_ERR_UNSUPPORTED);
    g = k32_gemm(); g.m = 16;
    check_status("d32 M16 padded unvalidated", potal_gemm_ws_check(&g), POTAL_ERR_UNSUPPORTED);
    g = k32_gemm(); g.k = 48; g.passes_per_block = 1;
    check_status("d32 K48 padded unvalidated", potal_gemm_ws_check(&g), POTAL_ERR_UNSUPPORTED);
    g = k32_gemm(); g.k = g.lda = 96; g.passes_per_block = 1;
    check_status("d32 K96 unvalidated", potal_gemm_ws_check(&g), POTAL_ERR_UNSUPPORTED);
}
#else
static void test_gemm_invalid(void) {
}

static void test_gemm_encode_d16(void) {
    potal_gemm_ws g = k32_gemm();
    g.m = g.n = POTAL_HW_DIM; g.k = POTAL_HW_DIM; g.lda = g.ldb = g.ldc = POTAL_HW_DIM;
    g.passes_per_block = 1;
    check_status("GEMM unvalidated for this DIM", potal_gemm_ws_check(&g), POTAL_ERR_UNSUPPORTED);
}
#endif

#if POTAL_HW_DIM == 16 || POTAL_HW_DIM == 32
static void test_gemm_invalid(void) {
    potal_gemm_ws g;

    check_status("gemm null", potal_gemm_ws_check(NULL), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm();
    check_status("encode null program", potal_gemm_ws_encode(&g, NULL), POTAL_ERR_INVALID_ARGUMENT);

    g = k32_gemm(); g.a = NULL;
    check_status("A null (DRAM addr 0 skips load)", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.b = (const int8_t *)(uintptr_t)0x20001;
    check_status("B misaligned", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.c = (int32_t *)(uintptr_t)(0x30000 + POTAL_HW_DIM);
    check_status("C requires DIM int32 alignment", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.k = 0;
    check_status("K=0", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.m = 0;
    check_status("M=0", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.lda = 31;
    check_status("lda < K", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.ldc = 15;
    check_status("ldc < N", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.passes_per_block = 0;
    check_status("passesPerBlock 0", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.passes_per_block = 0x10000;
    check_status("passesPerBlock 16-bit overflow", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.passes_per_block = 3;
    check_status("K tiles not multiple of passesPerBlock", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.scale_generation = 256;
    check_status("generation 256", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.c = (int32_t *)(uintptr_t)0x10000;
    check_status("C overlaps A", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.ldc = 0x40000000u;
    check_status("C stride bytes exceed 32 bits", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.lda = (size_t)1 << 29;
    check_status("A extent exceeds 32-bit DRAM offset", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.a = (const int8_t *)(uintptr_t)(UINT64_MAX - 63);
    check_status("A address wrap", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);

    static const uint32_t flags[] = {
        POTAL_GEMM_A_TRANSPOSE, POTAL_GEMM_B_TRANSPOSE, POTAL_GEMM_BIAS,
        POTAL_GEMM_RESADD, POTAL_GEMM_ACCUMULATE,
    };
    for (size_t i = 0; i < sizeof(flags) / sizeof(flags[0]); ++i) {
        char name[48];
        g = k32_gemm(); g.flags = flags[i];
        snprintf(name, sizeof(name), "flag 0x%" PRIx32 " unsupported", flags[i]);
        check_status(name, potal_gemm_ws_check(&g), POTAL_ERR_UNSUPPORTED);
    }
}
#endif

/* potal_loop_ws: the general LOOP_WS invocation of the tiling layer. */
static potal_loop_ws two_by_three_loop(void) {
    const uint32_t d = POTAL_HW_DIM;
    potal_loop_ws l = {
        .a = FAKE_A, .b = FAKE_B, .c = FAKE_C,
        .lda = 4 * d, .ldb = 3 * d, .ldc = 3 * d,
        .i_tiles = 2, .j_tiles = 3, .k_tiles = 4,
        .pad_i = 0, .pad_j = 0, .pad_k = 0,
        .scale_base = 10, .scale_generation = 7, .passes_per_block = 2,
        .flags = 0,
    };
    return l;
}

static void test_loop_ws(void) {
    const uint32_t d = POTAL_HW_DIM;
    potal_gemm_ws_program p;
    potal_loop_ws l = two_by_three_loop();

    check_status("loop 2x3x4 ok", potal_loop_ws_encode(&l, &p), POTAL_OK);
    check_u64("loop bounds rs1 (no pad)", p.bounds.rs1, 0);
    check_u64("loop bounds rs2", p.bounds.rs2, (4ull << 32) | (3ull << 16) | 2ull);
    check_u64("loop addrs_dc rs2", p.addrs_dc.rs2, 0x30000);
    check_u64("loop loop_ws ex_accumulate 0", p.loop_ws.rs1 & 1, 0);
    check_u64("loop loop_ws full_C", (p.loop_ws.rs1 >> 1) & 1, 1);
    check_u64("loop loop_ws enable", (p.loop_ws.rs1 >> 20) & 1, 1);
    check_u64("loop loop_ws base", (p.loop_ws.rs1 >> 21) & 0xFFFF, 10);
    check_u64("loop loop_ws generation", (p.loop_ws.rs1 >> 37) & 0xFF, 7);
    check_u64("loop loop_ws ppb", (p.loop_ws.rs1 >> 45) & 0xFFFF, 2);

    /* Last-tile padding goes to BOUNDS rs1 [15:0]/[31:16]/[47:32]. */
    l.pad_i = 3; l.pad_j = d - 1; l.pad_k = 5; l.ldb = 3 * d - (d - 1); l.ldc = l.ldb;
    check_status("loop padded ok", potal_loop_ws_encode(&l, &p), POTAL_OK);
    check_u64("loop bounds rs1 pads", p.bounds.rs1, (5ull << 32) | ((uint64_t)(d - 1) << 16) | 3ull);
    l = two_by_three_loop(); l.pad_j = d;
    check_status("loop pad_j == DIM", potal_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    l = two_by_three_loop(); l.pad_k = 1; l.lda = 4 * d - 2;
    check_status("loop lda < padded K", potal_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);

    /* Intermediate K chunk: accumulate, no C store (C = 0 keeps the ACC half). */
    l = two_by_three_loop(); l.c = NULL; l.ldc = 0; l.flags = POTAL_GEMM_ACCUMULATE;
    check_status("loop C NULL accumulate ok", potal_loop_ws_encode(&l, &p), POTAL_OK);
    check_u64("loop addrs_dc rs2 C NULL", p.addrs_dc.rs2, 0);
    check_u64("loop loop_ws ex_accumulate 1", p.loop_ws.rs1 & 1, 1);

    /* Tile origins: DIM-byte A/B, DIM*4-byte C; NULL A/B never allowed. */
    l = two_by_three_loop(); l.a = (const int8_t *)(uintptr_t)(0x10000 + d);
    check_status("loop A at a DIM offset ok", potal_loop_ws_check(&l), POTAL_OK);
    l = two_by_three_loop(); l.a = (const int8_t *)(uintptr_t)(0x10000 + 1);
    check_status("loop A misaligned", potal_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    l = two_by_three_loop(); l.b = NULL;
    check_status("loop B NULL", potal_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    l = two_by_three_loop(); l.c = (int32_t *)(uintptr_t)(0x30000 + d);
    check_status("loop C misaligned", potal_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    l = two_by_three_loop(); l.passes_per_block = 3;
    check_status("loop K tiles not multiple of ppb", potal_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);

    /* Capacity: scale window (blocks * J), work IDs, ACC half. */
    l = two_by_three_loop(); l.scale_base = 251; /* 2 blocks * 3 = 6 > 5 */
    check_status("loop scale window", potal_loop_ws_check(&l), POTAL_ERR_CAPACITY);
    l.scale_base = 250;
    check_status("loop scale window edge", potal_loop_ws_check(&l), POTAL_OK);
    l = two_by_three_loop();
    l.i_tiles = (POTAL_HW_ACC_ROWS / 2) / d + 1; l.j_tiles = 1; l.ldb = d; l.ldc = d;
    check_status("loop ACC half", potal_loop_ws_check(&l), POTAL_ERR_CAPACITY);

    l = two_by_three_loop(); l.flags = POTAL_GEMM_BIAS;
    check_status("loop bias unsupported", potal_loop_ws_check(&l), POTAL_ERR_UNSUPPORTED);
    l = two_by_three_loop(); l.flags = POTAL_GEMM_A_TRANSPOSE;
    check_status("loop transpose unsupported", potal_loop_ws_check(&l), POTAL_ERR_UNSUPPORTED);
    check_status("loop null", potal_loop_ws_check(NULL), POTAL_ERR_INVALID_ARGUMENT);

#if POTAL_HW_DIM == 16 || POTAL_HW_DIM == 32
    /* The validated GEMM and its single-invocation loop encode identically. */
    potal_gemm_ws_program pg;
    potal_gemm_ws g = k32_gemm();
    potal_loop_ws lg = {
        .a = g.a, .b = g.b, .c = g.c, .lda = g.lda, .ldb = g.ldb, .ldc = g.ldc,
        .i_tiles = 1, .j_tiles = 1, .k_tiles = 2,
        .scale_base = g.scale_base, .scale_generation = g.scale_generation,
        .passes_per_block = g.passes_per_block, .flags = 0,
    };
    check_status("gemm encode for comparison", potal_gemm_ws_encode(&g, &pg), POTAL_OK);
    check_status("loop encode for comparison", potal_loop_ws_encode(&lg, &p), POTAL_OK);
    check_u64("gemm == loop program bytes", memcmp(&pg, &p, sizeof(p)) == 0, 1);
#endif
#if !defined(__riscv)
    l = two_by_three_loop();
    check_status("host loop issue valid", potal_loop_ws_issue(&l), POTAL_ERR_NO_HARDWARE);
    l.a = NULL;
    check_status("host loop issue invalid", potal_loop_ws_issue(&l), POTAL_ERR_INVALID_ARGUMENT);
#endif
}

#if !defined(__riscv)
static void test_host_has_no_hardware(void) {
    potal_gemm_ws g = k32_gemm();
    check_status("host write invalid column", potal_scale_write(0, 0, POTAL_HW_DIM, 0), POTAL_ERR_INVALID_ARGUMENT);
    check_status("host write invalid carrier", potal_scale_write(0, 0, 0, 32768), POTAL_ERR_INVALID_ARGUMENT);
    check_status("host write valid", potal_scale_write(0, 0, 0, 0), POTAL_ERR_NO_HARDWARE);
    check_status("host release invalid", potal_scale_release(256, 0, 0), POTAL_ERR_INVALID_ARGUMENT);
    check_status("host release valid", potal_scale_release(0, 0, 0), POTAL_ERR_NO_HARDWARE);
    check_status("host fence", potal_fence(), POTAL_ERR_NO_HARDWARE);
    check_status("host tlb flush", potal_tlb_flush(), POTAL_ERR_NO_HARDWARE);
    g.a = NULL;
    check_status("host issue invalid", potal_gemm_ws_issue(&g), POTAL_ERR_INVALID_ARGUMENT);
#if POTAL_HW_DIM == 16 || POTAL_HW_DIM == 32
    g = k32_gemm();
    check_status("host issue valid", potal_gemm_ws_issue(&g), POTAL_ERR_NO_HARDWARE);
#endif
}
#endif

/* ---- H2 paired / indexed API ----------------------------------------------- */

/* Geometry names of the H2 firmware (potal_shared_weight_indexed.c): K = 3 tiles, N = 2 tiles. */
#define H2_KTILES 3u
#define H2_JTILES 2u
#define SP_ROWS   ((uint64_t)POTAL_HW_BANK_NUM * POTAL_HW_BANK_ROWS)

static potal_index_config h2_index_config(uint32_t pair) {
    const uint64_t rows = (uint64_t)H2_KTILES * H2_JTILES * POTAL_HW_DIM; /* B_SP_ROWS */
    potal_index_config c = {
        .pair = pair,
        .b_sp_base = (uint32_t)((pair + 1) * (SP_ROWS / 2) - rows), /* B_BASE of the H2 tests */
        .j_tiles = H2_JTILES,
        .source_k_rows = H2_KTILES * POTAL_HW_DIM,
        .active_rows = 3 * POTAL_HW_DIM / 2, /* ACTIVE_R */
    };
    return c;
}

static void test_index_encoding(void) {
    potal_rocc_cmd c = { 1, 1 };
    potal_index_config cfg = h2_index_config(1);
    check_status("index config pair 1 (H2 geometry)", potal_index_config_encode(&cfg, &c), POTAL_OK);
    /* rs1 = {JTILES[63:32], B_BASE[31:0]}, rs2 = {ACTIVE_R[63:32], KROWS[31:0]} of configure_indices(). */
    check_u64("index config rs1", c.rs1, ((uint64_t)H2_JTILES << 32) | (SP_ROWS - H2_KTILES * H2_JTILES * POTAL_HW_DIM));
    check_u64("index config rs2", c.rs2, ((uint64_t)(3 * POTAL_HW_DIM / 2) << 32) | (H2_KTILES * POTAL_HW_DIM));
    cfg = h2_index_config(0);
    check_status("index config pair 0 (bank 1)", potal_index_config_encode(&cfg, &c), POTAL_OK);
    check_u64("index config pair 0 base below SP/2", c.rs1 & 0xFFFFFFFFull, SP_ROWS / 2 - H2_KTILES * H2_JTILES * POTAL_HW_DIM);

    /* Capacity: the whole table fits, one more slot does not (DIM16: 32, DIM32: 64, DIM64: 128). */
    cfg = h2_index_config(0); cfg.active_rows = POTAL_HW_INDEX_ENTRIES;
    check_status("index config full table", potal_index_config_check(&cfg), POTAL_OK);
    cfg.active_rows = POTAL_HW_INDEX_ENTRIES + 1;
    check_status("index config table + 1", potal_index_config_check(&cfg), POTAL_ERR_CAPACITY);
    check_u64("index entries are a whole number of DIM passes", POTAL_HW_INDEX_ENTRIES % POTAL_HW_DIM, 0);

    /* Region inside the pair's weight bank (ingress cfgAddressValid). */
    cfg = h2_index_config(0); cfg.b_sp_base = (uint32_t)(SP_ROWS / 4 - 1);
    check_status("index config base below bank 1", potal_index_config_check(&cfg), POTAL_ERR_INVALID_ARGUMENT);
    cfg = h2_index_config(0); cfg.b_sp_base = (uint32_t)(SP_ROWS / 2 - 1);
    check_status("index config region past bank 1 end", potal_index_config_check(&cfg), POTAL_ERR_INVALID_ARGUMENT);
    cfg = h2_index_config(1); cfg.b_sp_base = (uint32_t)(SP_ROWS / 2); /* bank 2: an activation bank */
    check_status("index config base in the activation bank of pair 1", potal_index_config_check(&cfg), POTAL_ERR_INVALID_ARGUMENT);
    cfg = h2_index_config(0); cfg.pair = 2;
    check_status("index config pair 2", potal_index_config_check(&cfg), POTAL_ERR_INVALID_ARGUMENT);
    cfg = h2_index_config(0); cfg.source_k_rows = 0;
    check_status("index config sourceKRows 0", potal_index_config_check(&cfg), POTAL_ERR_INVALID_ARGUMENT);
    cfg = h2_index_config(0); cfg.active_rows = 0;
    check_status("index config activeRows 0", potal_index_config_check(&cfg), POTAL_ERR_INVALID_ARGUMENT);
    check_status("index config null", potal_index_config_check(NULL), POTAL_ERR_INVALID_ARGUMENT);
    /* A panel whose K is not a whole number of tiles still occupies whole tiles. */
    cfg = h2_index_config(0); cfg.source_k_rows = H2_KTILES * POTAL_HW_DIM - 1;
    check_status("index config partial K tile rounds up", potal_index_config_check(&cfg), POTAL_OK);
    cfg.source_k_rows = (H2_KTILES - 1) * POTAL_HW_DIM; cfg.b_sp_base += POTAL_HW_DIM * H2_JTILES; /* one K tile less resident */
    check_status("index config smaller panel at a higher base", potal_index_config_check(&cfg), POTAL_OK);

    /* INDEX_WRITE: rs1 = slot, rs2 = resident_k (both window / panel local). */
    cfg = h2_index_config(1);
    potal_index_write w = { .slot = 5, .resident_k = 2 * ((5 * 5) % (3 * POTAL_HW_DIM / 2)) + 1 }; /* active_k[5] of the H2 test */
    check_status("index write encode", potal_index_write_encode(&cfg, &w, &c), POTAL_OK);
    check_u64("index write rs1 slot", c.rs1, 5);
    check_u64("index write rs2 resident k", c.rs2, w.resident_k);
    w.slot = cfg.active_rows;
    check_status("index write slot == activeRows", potal_index_write_check(&cfg, &w), POTAL_ERR_INVALID_ARGUMENT);
    w.slot = 0; w.resident_k = cfg.source_k_rows;
    check_status("index write resident k == sourceKRows", potal_index_write_check(&cfg, &w), POTAL_ERR_INVALID_ARGUMENT);
    /* A K panel starting at original K 480 (DIM16): original K 500 is resident K 20, not 500. */
    cfg.source_k_rows = 2 * POTAL_HW_DIM;
    w.resident_k = 500 - 480;
    check_status("index write resident K of a later panel", potal_index_write_check(&cfg, &w), POTAL_HW_DIM >= 11 ? POTAL_OK : POTAL_ERR_INVALID_ARGUMENT);
    w.resident_k = 500;
    check_status("index write global K rejected for a partitioned B", potal_index_write_check(&cfg, &w), POTAL_ERR_INVALID_ARGUMENT);
    check_status("index write null", potal_index_write_check(&cfg, NULL), POTAL_ERR_INVALID_ARGUMENT);
}

/* The H2 Main LOOP_WS of pair `pair`: A [DIM][3*DIM], B [3*DIM][2*DIM], scale phase 0, gen 7, ppb = K tiles. */
static potal_paired_loop_ws h2_main(uint32_t pair) {
    const uint32_t d = POTAL_HW_DIM;
    potal_paired_loop_ws l = {
        .loop = {
            .a = FAKE_A, .b = FAKE_B, .c = NULL,
            .lda = H2_KTILES * d, .ldb = H2_JTILES * d, .ldc = 0,
            .i_tiles = 1, .j_tiles = H2_JTILES, .k_tiles = H2_KTILES,
            .pad_i = 0, .pad_j = 0, .pad_k = 0,
            .scale_base = 0, .scale_generation = 7, .passes_per_block = H2_KTILES,
            .flags = 0,
        },
        .path = POTAL_PAIRED_MAIN, .pair = pair, .a_offset_tiles = 0, .resident_b = 0, .indexed = 0,
    };
    return l;
}

/* The H2 Residual window: packed A [DIM][2 tiles], ACTIVE_R rows, resident B, A behind Main A. */
static potal_paired_loop_ws h2_residual(uint32_t pair, uint32_t phase) {
    const uint32_t d = POTAL_HW_DIM;
    const uint32_t active = 3 * d / 2;
    potal_paired_loop_ws l = {
        .loop = {
            .a = FAKE_A, .b = NULL, .c = NULL,
            .lda = H2_KTILES * d, .ldb = 0, .ldc = 0,
            .i_tiles = 1, .j_tiles = H2_JTILES, .k_tiles = 2,
            .pad_i = 0, .pad_j = 0, .pad_k = 2 * d - active,
            .scale_base = phase * H2_JTILES, .scale_generation = 7 + phase, .passes_per_block = 2,
            .flags = phase > 1 ? POTAL_GEMM_ACCUMULATE : 0,
        },
        .path = POTAL_PAIRED_RESIDUAL, .pair = pair, .a_offset_tiles = H2_KTILES, .resident_b = 1, .indexed = 1,
    };
    return l;
}

static void test_paired_loop_ws(void) {
    potal_gemm_ws_program p;
    const uint32_t d = POTAL_HW_DIM;
    potal_paired_loop_ws l = h2_main(0);

    /* Main, pair 0: submit_main() flags of potal_h2d_cycle_bench.c (accumulate 0, phase 0). */
    check_status("paired main pair 0 encode", potal_paired_loop_ws_encode(&l, &p), POTAL_OK);
    const uint64_t h2_main_rs1 = (1ull << 61) | (1ull << 1) | (1ull << 20) | (1ull << 18) | (1ull << 16) |
                                 (0ull << 21) | (7ull << 37) | ((uint64_t)H2_KTILES << 45);
    check_u64("paired main rs1 == H2 submit_main flags", p.loop_ws.rs1, h2_main_rs1);
    check_u64("paired main rs2 (A offset 0)", p.loop_ws.rs2, 0);
    check_u64("paired main bounds", p.bounds.rs2, ((uint64_t)H2_KTILES << 32) | ((uint64_t)H2_JTILES << 16) | 1ull);
    check_u64("paired main addrs_ab A", p.addrs_ab.rs1, 0x10000);
    check_u64("paired main addrs_ab B (original B loaded)", p.addrs_ab.rs2, 0x20000);
    check_u64("paired main addrs_dc (no D, no C store)", p.addrs_dc.rs1 | p.addrs_dc.rs2, 0);
    check_u64("paired main strides_ab", p.strides_ab.rs1 | (p.strides_ab.rs2 << 32),
              (uint64_t)(H2_KTILES * d) | ((uint64_t)(H2_JTILES * d) << 32));
    check_u64("paired bits: paired 1, residual 0, indexed 0", p.loop_ws.rs1 >> 61, 1);
    l = h2_main(1);
    check_status("paired main pair 1 encode", potal_paired_loop_ws_encode(&l, &p), POTAL_OK);
    check_u64("paired main pair 1 selectors 2/2", (p.loop_ws.rs1 >> 16) & 0xF, 0xA);
    l = h2_main(0); l.loop.flags = POTAL_GEMM_ACCUMULATE;
    check_status("paired main accumulate encode", potal_paired_loop_ws_encode(&l, &p), POTAL_OK);
    check_u64("paired main accumulate bit 0", p.loop_ws.rs1 & 1, 1);

    /* Residual, pair 0, phase 1: issue_loop(1) flags of potal_shared_weight_indexed.c plus the pair selectors. */
    l = h2_residual(0, 1);
    check_status("paired residual pair 0 encode", potal_paired_loop_ws_encode(&l, &p), POTAL_OK);
    const uint64_t h2_res_rs1 = (1ull << 61) | (1ull << 62) | (1ull << 63) | (1ull << 1) | (1ull << 20) |
                                (1ull << 18) | (1ull << 16) | ((uint64_t)(1 * H2_JTILES) << 21) | (8ull << 37) | (2ull << 45);
    check_u64("paired residual rs1 == H2 residual flags", p.loop_ws.rs1, h2_res_rs1);
    check_u64("paired bits: paired 1, residual 1, indexed 1", p.loop_ws.rs1 >> 61, 7);
    check_u64("paired residual rs2[31:3] = A offset tiles", p.loop_ws.rs2, (uint64_t)H2_KTILES << 3);
    check_u64("paired residual rs2[2:0] clear", p.loop_ws.rs2 & 7, 0);
    check_u64("paired residual B DRAM address 0 (resident B)", p.addrs_ab.rs2, 0);
    check_u64("paired residual bounds pad_k", p.bounds.rs1, ((uint64_t)(2 * d - 3 * d / 2) << 32));
    check_u64("paired residual bounds tiles", p.bounds.rs2, (2ull << 32) | ((uint64_t)H2_JTILES << 16) | 1ull);
    l = h2_residual(1, 2);
    check_status("paired residual pair 1 accumulate encode", potal_paired_loop_ws_encode(&l, &p), POTAL_OK);
    check_u64("paired residual pair 1 selectors + accumulate", (p.loop_ws.rs1 >> 16) & 0xF, 0xA);
    check_u64("paired residual phase 2 accumulate", p.loop_ws.rs1 & 1, 1);
    /* A large A offset: rs2[31:3] carries 29 bits. */
    l = h2_residual(0, 1); l.a_offset_tiles = POTAL_HW_BANK_ROWS / d - 2;
    check_status("paired residual A offset at the bank end", potal_paired_loop_ws_encode(&l, &p), POTAL_OK);
    check_u64("paired residual A offset rs2", p.loop_ws.rs2 >> 3, POTAL_HW_BANK_ROWS / d - 2);

    /* Field rules. */
    l = h2_main(0); l.loop.b = NULL;
    check_status("paired main needs the original B", potal_paired_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    l = h2_main(0); l.resident_b = 1;
    check_status("paired main cannot reuse B", potal_paired_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    l = h2_main(0); l.a_offset_tiles = 1;
    check_status("paired main A offset must be 0", potal_paired_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    l = h2_residual(0, 1); l.loop.b = FAKE_B; l.loop.ldb = H2_JTILES * d;
    check_status("paired residual must not reload B", potal_paired_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    l = h2_residual(0, 1); l.indexed = 0;
    check_status("paired residual must be indexed", potal_paired_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    l = h2_residual(0, 1); l.resident_b = 0;
    check_status("paired residual must reuse the resident B", potal_paired_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    l = h2_main(0); l.loop.c = FAKE_C; l.loop.ldc = H2_JTILES * d;
    check_status("paired LOOP_WS C store unsupported (readback instead)", potal_paired_loop_ws_check(&l), POTAL_ERR_UNSUPPORTED);
    l = h2_main(0); l.pair = 2;
    check_status("paired pair 2", potal_paired_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    l = h2_main(0); l.path = 2;
    check_status("paired path 2", potal_paired_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    l = h2_main(0); l.loop.flags = POTAL_GEMM_BIAS;
    check_status("paired bias unsupported", potal_paired_loop_ws_check(&l), POTAL_ERR_UNSUPPORTED);
    l = h2_main(0); l.loop.pad_k = d;
    check_status("paired pad_k == DIM", potal_paired_loop_ws_check(&l), POTAL_ERR_INVALID_ARGUMENT);
    /* Last-tile padding of a Main panel: pad_i / pad_j go to BOUNDS rs1. */
    l = h2_main(0); l.loop.pad_i = 3; l.loop.pad_j = d - 1; l.loop.ldb = H2_JTILES * d - (d - 1);
    check_status("paired main padded encode", potal_paired_loop_ws_encode(&l, &p), POTAL_OK);
    check_u64("paired main pads", p.bounds.rs1, ((uint64_t)(d - 1) << 16) | 3ull);

    /* Physical capacity: one activation bank for A (behind its offset), one weight bank for B, one ACC bank. */
    const uint32_t bank_tiles = POTAL_HW_BANK_ROWS / d;
    l = h2_main(0); l.loop.i_tiles = 1; l.loop.k_tiles = bank_tiles; l.loop.passes_per_block = 1; l.loop.lda = bank_tiles * d;
    l.loop.j_tiles = 1; l.loop.ldb = d; l.loop.b = (const int8_t *)(uintptr_t)0x80000; /* a long K: B away from A's extent */
    check_status("paired main A fills the activation bank", potal_paired_loop_ws_check(&l), POTAL_OK);
    l.loop.k_tiles = bank_tiles + 1; l.loop.lda = (bank_tiles + 1) * d;
    check_status("paired main A beyond the activation bank", potal_paired_loop_ws_check(&l), POTAL_ERR_CAPACITY);
    l = h2_main(0); l.loop.j_tiles = bank_tiles / H2_KTILES + 1; l.loop.ldb = l.loop.j_tiles * d; l.loop.i_tiles = 1;
    check_status("paired main B beyond the weight bank", potal_paired_loop_ws_check(&l), POTAL_ERR_CAPACITY);
    l = h2_residual(0, 1); l.a_offset_tiles = bank_tiles - 1; /* 2 tiles do not fit behind it */
    check_status("paired residual A offset + footprint beyond the bank", potal_paired_loop_ws_check(&l), POTAL_ERR_CAPACITY);
    l = h2_residual(0, 1); l.a_offset_tiles = bank_tiles;
    check_status("paired residual A offset == bank tiles", potal_paired_loop_ws_check(&l), POTAL_ERR_CAPACITY);
    l = h2_main(0); l.loop.i_tiles = (POTAL_HW_ACC_ROWS / 2) / d / H2_JTILES + 1;
    check_status("paired ACC bank exceeded", potal_paired_loop_ws_check(&l), POTAL_ERR_CAPACITY);
    l = h2_main(0); l.loop.scale_base = 255;
    check_status("paired scale window", potal_paired_loop_ws_check(&l), POTAL_ERR_CAPACITY);
    check_status("paired null", potal_paired_loop_ws_check(NULL), POTAL_ERR_INVALID_ARGUMENT);
}

static void test_acc_readback(void) {
    const uint32_t d = POTAL_HW_DIM;
    potal_rocc_cmd st, mv[8];
    uint32_t count = 0;
    potal_acc_readback r = { .acc_bank = 1, .i_tiles = 1, .j_tiles = H2_JTILES, .pad_i = 0, .pad_j = 0,
                             .c = FAKE_C, .ldc = H2_JTILES * d };
    check_status("readback ACC1 1x2 encode", potal_acc_readback_encode(&r, &st, mv, 8, &count), POTAL_OK);
    check_u64("readback count", count, H2_JTILES);
    check_u64("readback config_st rs1", st.rs1, 2);
    check_u64("readback config_st rs2 (row bytes)", st.rs2, (uint64_t)H2_JTILES * d * 4);
    /* acc_desc(bank 1, j): {DIM[63:48], DIM[47:32], ACC1_FULL_ADDR + j*DIM} of the H2 tests, ACC1 = 5 << 29 + ACC_ROWS/2. */
    const uint64_t acc1 = (5ull << 29) + POTAL_HW_ACC_ROWS / 2;
    check_u64("readback mvout 0 rs2 == H2 acc_desc(1, 0)", mv[0].rs2, ((uint64_t)d << 48) | ((uint64_t)d << 32) | acc1);
    check_u64("readback mvout 1 rs2 == H2 acc_desc(1, 1)", mv[1].rs2, ((uint64_t)d << 48) | ((uint64_t)d << 32) | (acc1 + d));
    check_u64("readback mvout 0 rs1 (C)", mv[0].rs1, 0x30000);
    check_u64("readback mvout 1 rs1 (C + DIM columns)", mv[1].rs1, 0x30000 + (uint64_t)d * 4);
    r.acc_bank = 0;
    check_status("readback ACC0 encode", potal_acc_readback_encode(&r, &st, mv, 8, &count), POTAL_OK);
    check_u64("readback ACC0 base = 5 << 29", mv[0].rs2 & 0xFFFFFFFFull, 5ull << 29);
    check_u64("readback ACC0 / ACC1 never alias", (mv[0].rs2 & 0xFFFFFFFFull) != acc1, 1);
    /* Two row tiles with padding: tile (i, j) at ACC row (i * j_tiles + j) * DIM, shortened last tiles. */
    r.acc_bank = 1; r.i_tiles = 2; r.pad_i = 3; r.pad_j = 5; r.ldc = 2 * d;
    check_status("readback 2x2 padded encode", potal_acc_readback_encode(&r, &st, mv, 8, &count), POTAL_OK);
    check_u64("readback padded count", count, 4);
    check_u64("readback tile (1,1) rows/cols", mv[3].rs2 >> 32, ((uint64_t)(d - 3) << 16) | (d - 5));
    check_u64("readback tile (1,1) acc row", mv[3].rs2 & 0xFFFFFFFFull, acc1 + 3 * d);
    check_u64("readback tile (1,1) dram", mv[3].rs1, 0x30000 + ((uint64_t)d * 2 * d + d) * 4);
    check_u64("readback tile (0,1) full rows", mv[1].rs2 >> 48, d);
    check_status("readback capacity 3 of 4", potal_acc_readback_encode(&r, &st, mv, 3, &count), POTAL_ERR_CAPACITY);
    r.ldc = 2 * d - 6;
    check_status("readback ldc < columns", potal_acc_readback_check(&r), POTAL_ERR_INVALID_ARGUMENT);
    r.ldc = 2 * d; r.c = (int32_t *)(uintptr_t)(0x30000 + 4);
    check_status("readback C misaligned", potal_acc_readback_check(&r), POTAL_ERR_INVALID_ARGUMENT);
    r.c = FAKE_C; r.acc_bank = 2;
    check_status("readback bank 2", potal_acc_readback_check(&r), POTAL_ERR_INVALID_ARGUMENT);
    r.acc_bank = 0; r.i_tiles = (POTAL_HW_ACC_ROWS / 2) / d / 2 + 1; r.pad_i = 0;
    check_status("readback beyond one ACC bank", potal_acc_readback_check(&r), POTAL_ERR_CAPACITY);
    check_status("readback null", potal_acc_readback_check(NULL), POTAL_ERR_INVALID_ARGUMENT);
}

/* Command sink: records the stream, answers xd commands with a programmable status. */
#define SINK_CAPACITY 2048
typedef struct sink_state {
    potal_paired_cmd_record records[SINK_CAPACITY];
    uint32_t count;
    uint32_t fail_funct;    /* an xd command of this funct answers fail_status (once) */
    potal_status fail_status;
} sink_state;

static potal_status sink_cmd(void *ctx, const potal_paired_cmd_record *rec) {
    sink_state *st = ctx;
    if (st->count < SINK_CAPACITY) {
        st->records[st->count] = *rec;
    }
    st->count++;
    if (st->fail_funct != 0 && rec->funct == st->fail_funct) {
        st->fail_funct = 0;
        return st->fail_status;
    }
    return POTAL_OK;
}

static uint32_t sink_funct_at(const sink_state *st, uint32_t i) {
    return i < st->count && i < SINK_CAPACITY ? st->records[i].funct : 0xFFFFFFFFu;
}

static int sink_has(const sink_state *st, uint32_t funct) {
    for (uint32_t i = 0; i < st->count && i < SINK_CAPACITY; ++i) {
        if (st->records[i].funct == funct) {
            return 1;
        }
    }
    return 0;
}

static int sink_order_is(const sink_state *st, const uint32_t *functs, uint32_t n) {
    if (st->count != n) {
        return 0;
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (sink_funct_at(st, i) != functs[i]) {
            return 0;
        }
    }
    return 1;
}

/* Index window programming of the H2 residual: config + ACTIVE_R writes (resident K = active_k[t]). */
static potal_status program_h2_window(potal_paired_session *s, uint32_t pair) {
    const potal_index_config cfg = h2_index_config(pair);
    potal_status status = potal_paired_index_config(s, &cfg);
    for (uint32_t t = 0; status == POTAL_OK && t < cfg.active_rows; ++t) {
        const potal_index_write w = { .slot = t, .resident_k = 2 * ((t * 5) % cfg.active_rows) + 1 };
        status = potal_paired_index_write(s, &w);
    }
    return status;
}

static void test_paired_session(void) {
    const uint32_t d = POTAL_HW_DIM;
    sink_state sink_st;
    memset(&sink_st, 0, sizeof(sink_st));
    const potal_paired_sink sink = { sink_cmd, &sink_st };
    potal_paired_session *s = NULL;
    potal_paired_session_status q;

    /* Ownership: one session; the regular API refuses while it is open. */
    check_status("session open", potal_paired_session_open(&sink, &s), POTAL_OK);
    check_u64("session handle", s != NULL, 1);
    potal_paired_session *other = NULL;
    check_status("second session refused", potal_paired_session_open(&sink, &other), POTAL_ERR_BUSY);
    {
        potal_loop_ws l = two_by_three_loop();
        check_status("regular loop issue refused during a paired session", potal_loop_ws_issue(&l), POTAL_ERR_BUSY);
        check_status("regular fence refused during a paired session", potal_fence(), POTAL_ERR_BUSY);
        check_status("regular tlb flush refused during a paired session", potal_tlb_flush(), POTAL_ERR_BUSY);
    }

    /* The H2 sequence: TLB flush, scales, Main(pair 0), Index window, Residual, readback, fence. */
    check_status("session tlb flush (nothing outstanding)", potal_paired_tlb_flush(s), POTAL_OK);
    check_u64("flush recorded", sink_funct_at(&sink_st, 0), POTAL_FUNCT_FLUSH);
    potal_status st = POTAL_OK;
    for (uint32_t j = 0; j < H2_JTILES && st == POTAL_OK; ++j) {
        for (uint32_t col = 0; col < d && st == POTAL_OK; ++col) {
            st = potal_paired_scale_write(s, j, 7, col, 0);
        }
    }
    check_status("scale phase 0 written (2 slots x DIM columns)", st, POTAL_OK);
    check_status("scale write of a live slot with another generation", potal_paired_scale_write(s, 0, 8, 0, 0), POTAL_ERR_CAPACITY);
    check_status("scale write invalid carrier", potal_paired_scale_write(s, 3, 7, 0, 32768), POTAL_ERR_INVALID_ARGUMENT);
    potal_paired_loop_ws m0 = h2_main(0);
    const uint32_t before_main = sink_st.count;
    check_status("residual before any Main refused (no resident B)", potal_paired_loop_ws_issue(s, &(potal_paired_loop_ws){ 0 }), POTAL_ERR_INVALID_ARGUMENT);
    {
        potal_paired_loop_ws r = h2_residual(0, 1);
        check_status("residual on a pair without resident B refused", potal_paired_loop_ws_issue(s, &r), POTAL_ERR_INVALID_ARGUMENT);
    }
    check_u64("refused issues send nothing", sink_st.count, before_main);
    check_status("main pair 0 issued", potal_paired_loop_ws_issue(s, &m0), POTAL_OK);
    check_u64("main = 4 CONFIG + 5 LOOP_WS_CONFIG + LOOP_WS", sink_st.count - before_main, 10);
    check_u64("main last command is LOOP_WS", sink_funct_at(&sink_st, sink_st.count - 1), POTAL_FUNCT_LOOP_WS);
    q = potal_paired_session_query(s);
    check_u64("pair 0 resident B recorded", q.pair[0].resident_b, 1);
    check_u64("pair 0 resident B base = H2 B_BASE", q.pair[0].b_sp_base, SP_ROWS / 2 - H2_KTILES * H2_JTILES * d);
    check_u64("pair 0 resident K rows", q.pair[0].b_k_rows, H2_KTILES * d);
    check_u64("pair 0 Main A tiles", q.pair[0].main_a_tiles, H2_KTILES);
    check_u64("one loop outstanding", q.outstanding_loops, 1);

    /* Index transaction on pair 0 (no fence before it: a same-buffer update is the hardware's stall). */
    {
        const potal_index_write w = { 0, 1 };
        check_status("index write without a transaction", potal_paired_index_write(s, &w), POTAL_ERR_BUSY);
    }
    potal_index_config cfg0 = h2_index_config(0);
    check_status("index config pair 0", potal_paired_index_config(s, &cfg0), POTAL_OK);
    {
        potal_index_config cfg1 = h2_index_config(1);
        check_status("another config inside an unfinished transaction refused", potal_paired_index_config(s, &cfg1), POTAL_ERR_BUSY);
        potal_paired_loop_ws r = h2_residual(0, 1);
        check_status("residual on a half-programmed window refused", potal_paired_loop_ws_issue(s, &r), POTAL_ERR_INVALID_ARGUMENT);
        const potal_index_write w = { .slot = 1, .resident_k = 1 };
        check_status("index write out of slot order", potal_paired_index_write(s, &w), POTAL_ERR_INVALID_ARGUMENT);
        const potal_index_write bad = { .slot = 0, .resident_k = cfg0.source_k_rows };
        check_status("index write resident K beyond the panel", potal_paired_index_write(s, &bad), POTAL_ERR_INVALID_ARGUMENT);
    }
    const uint32_t before_writes = sink_st.count;
    for (uint32_t t = 0; t < cfg0.active_rows; ++t) {
        const potal_index_write w = { .slot = t, .resident_k = 2 * ((t * 5) % cfg0.active_rows) + 1 };
        st = potal_paired_index_write(s, &w);
        if (st != POTAL_OK) {
            break;
        }
    }
    check_status("index writes of the window", st, POTAL_OK);
    check_u64("one INDEX_WRITE per active slot", sink_st.count - before_writes, cfg0.active_rows);
    check_u64("index write rs1/rs2 = slot / resident K", sink_st.records[before_writes + 5].rs1 | (sink_st.records[before_writes + 5].rs2 << 32),
              5ull | ((uint64_t)(2 * ((5 * 5) % cfg0.active_rows) + 1) << 32));
    q = potal_paired_session_query(s);
    check_u64("transaction complete", q.index_transaction_open, 0);
    check_u64("pair 0 window valid", q.pair[0].index_valid, 1);

    /* Residual scales (phase 1), Residual on the resident B of pair 0, then a second Main on pair 1 outstanding together. */
    for (uint32_t j = 0; j < H2_JTILES && st == POTAL_OK; ++j) {
        for (uint32_t col = 0; col < d && st == POTAL_OK; ++col) {
            st = potal_paired_scale_write(s, H2_JTILES + j, 8, col, 0);
        }
    }
    check_status("scale phase 1 written", st, POTAL_OK);
    potal_paired_loop_ws r0 = h2_residual(0, 1);
    {
        potal_paired_loop_ws overlap = r0; overlap.a_offset_tiles = H2_KTILES - 1;
        check_status("residual A overlapping Main A refused", potal_paired_loop_ws_issue(s, &overlap), POTAL_ERR_INVALID_ARGUMENT);
        potal_paired_loop_ws wrong = r0; wrong.loop.pad_k = 0; /* active rows 32 != the window's */
        check_status("residual whose K rows differ from the window refused", potal_paired_loop_ws_issue(s, &wrong), POTAL_ERR_INVALID_ARGUMENT);
        potal_paired_loop_ws p1 = r0; p1.pair = 1;
        check_status("residual on the other pair (no B there) refused", potal_paired_loop_ws_issue(s, &p1), POTAL_ERR_INVALID_ARGUMENT);
    }
    check_status("residual pair 0 issued without a fence after Main", potal_paired_loop_ws_issue(s, &r0), POTAL_OK);
    potal_paired_loop_ws m1 = h2_main(1);
    check_status("main pair 1 issued with pair 0 work outstanding", potal_paired_loop_ws_issue(s, &m1), POTAL_OK);
    check_status("main pair 0 again while pair 0 outstanding refused", potal_paired_loop_ws_issue(s, &m0), POTAL_ERR_BUSY);
    q = potal_paired_session_query(s);
    check_u64("three loops outstanding, no fence between them", q.outstanding_loops, 3);
    check_u64("pair outstanding 2 / 1", q.pair[0].outstanding * 10 + q.pair[1].outstanding, 21);
    /* Cross-buffer: pair 1's window may be programmed while pair 0 executes (no refusal, no fence). */
    check_status("index window of pair 1 programmed while pair 0 is busy", program_h2_window(s, 1), POTAL_OK);
    q = potal_paired_session_query(s);
    check_u64("pair 1 window valid, pair 0 window kept", q.pair[0].index_valid * 10 + q.pair[1].index_valid, 11);
    check_u64("no FENCE in the stream so far", sink_has(&sink_st, POTAL_FUNCT_FENCE), 0);
    {
        /* The whole stream up to here, in order: flush, 2*DIM writes, Main(10), config, ACTIVE_R writes,
         * 2*DIM writes, Residual(10), Main(10), config, ACTIVE_R writes. */
        uint32_t expect[SINK_CAPACITY];
        uint32_t n = 0;
        expect[n++] = POTAL_FUNCT_FLUSH;
        for (uint32_t i = 0; i < 2 * d; ++i) expect[n++] = POTAL_FUNCT_SCALE_WRITE;
        const uint32_t loop[10] = { POTAL_FUNCT_CONFIG, POTAL_FUNCT_CONFIG, POTAL_FUNCT_CONFIG, POTAL_FUNCT_CONFIG,
                                    POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS, POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB,
                                    POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_DC, POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_AB,
                                    POTAL_FUNCT_LOOP_WS_CONFIG_STRIDES_DC, POTAL_FUNCT_LOOP_WS };
        for (uint32_t i = 0; i < 10; ++i) expect[n++] = loop[i];
        expect[n++] = POTAL_FUNCT_INDEX_CONFIG;
        for (uint32_t i = 0; i < cfg0.active_rows; ++i) expect[n++] = POTAL_FUNCT_INDEX_WRITE;
        for (uint32_t i = 0; i < 2 * d; ++i) expect[n++] = POTAL_FUNCT_SCALE_WRITE;
        for (uint32_t i = 0; i < 10; ++i) expect[n++] = loop[i];
        for (uint32_t i = 0; i < 10; ++i) expect[n++] = loop[i];
        expect[n++] = POTAL_FUNCT_INDEX_CONFIG;
        for (uint32_t i = 0; i < cfg0.active_rows; ++i) expect[n++] = POTAL_FUNCT_INDEX_WRITE;
        check_u64("command stream order (flush, scales, Main 0, window 0, scales, Residual 0, Main 1, window 1)",
                  sink_order_is(&sink_st, expect, n), 1);
    }

    /* Readback of ACC1, then the fence that completes everything; readback before the fence stays outstanding. */
    potal_acc_readback rb = { .acc_bank = 1, .i_tiles = 1, .j_tiles = H2_JTILES, .pad_i = 0, .pad_j = 0, .c = FAKE_C, .ldc = H2_JTILES * d };
    check_status("ACC1 readback issued", potal_paired_acc_readback_issue(s, &rb), POTAL_OK);
    check_u64("readback = CONFIG_ST + 2 MVOUT", sink_funct_at(&sink_st, sink_st.count - 3) == POTAL_FUNCT_CONFIG &&
              sink_funct_at(&sink_st, sink_st.count - 2) == POTAL_FUNCT_MVOUT && sink_funct_at(&sink_st, sink_st.count - 1) == POTAL_FUNCT_MVOUT, 1);
    check_status("close with outstanding work refused", potal_paired_session_close(s), POTAL_ERR_BUSY);
    check_status("tlb flush with outstanding work refused", potal_paired_tlb_flush(s), POTAL_ERR_BUSY);
    check_status("fence drains", potal_paired_fence(s), POTAL_OK);
    q = potal_paired_session_query(s);
    check_u64("nothing outstanding after the fence", q.outstanding_loops + q.outstanding_readbacks + q.pair[0].outstanding + q.pair[1].outstanding, 0);
    check_u64("resident B and windows survive the fence", q.pair[0].resident_b + q.pair[1].resident_b + q.pair[0].index_valid + q.pair[1].index_valid, 4);
    /* Releases: the live generation, in write order; a wrong generation is refused before issue. */
    check_status("release with the wrong generation", potal_paired_scale_release(s, 0, 9, 0), POTAL_ERR_INVALID_ARGUMENT);
    for (uint32_t j = 0; j < 2 * H2_JTILES && st == POTAL_OK; ++j) {
        for (uint32_t col = 0; col < d && st == POTAL_OK; ++col) {
            st = potal_paired_scale_release(s, j, j < H2_JTILES ? 7 : 8, col);
        }
    }
    check_status("releases of both windows", st, POTAL_OK);
    check_u64("no live scale slot", potal_paired_session_query(s).live_scale_slots, 0);
    check_status("release of a free slot", potal_paired_scale_release(s, 0, 7, 0), POTAL_ERR_INVALID_ARGUMENT);
    /* Main on pair 0 again after the fence: the pair's readers drained. */
    check_status("main pair 0 after the fence", potal_paired_loop_ws_issue(s, &m0), POTAL_OK);
    check_u64("new panel invalidates pair 0's window", potal_paired_session_query(s).pair[0].index_valid, 0);
    /* An accepted SCALE_RELEASE is a drain point (releaseSafe == fenceDrained): outstanding work is gone. */
    for (uint32_t col = 0; col < d && st == POTAL_OK; ++col) {
        st = potal_paired_scale_write(s, 9, 3, col, 0);
    }
    check_status("scale slot 9 written", st, POTAL_OK);
    check_u64("one loop outstanding before the release", potal_paired_session_query(s).outstanding_loops, 1);
    check_status("release slot 9 column 0", potal_paired_scale_release(s, 9, 3, 0), POTAL_OK);
    q = potal_paired_session_query(s);
    check_u64("accepted release drained the outstanding loop", q.outstanding_loops + q.pair[0].outstanding, 0);
    check_u64("slot 9 still live until its last column", q.live_scale_slots, 1);
    check_status("main pair 0 again right after the release (drained)", potal_paired_loop_ws_issue(s, &m0), POTAL_OK);
    for (uint32_t col = 1; col < d && st == POTAL_OK; ++col) {
        st = potal_paired_scale_release(s, 9, 3, col);
    }
    check_status("release slot 9 remaining columns", st, POTAL_OK);
    check_u64("slot 9 free", potal_paired_session_query(s).live_scale_slots, 0);
    /* A padded last J tile writes fewer than DIM columns: its slot is free once those columns are released. */
    for (uint32_t col = 0; col + 4 < d && st == POTAL_OK; ++col) {
        st = potal_paired_scale_write(s, 10, 4, col, 0);
    }
    check_status("slot 10 written for DIM-4 columns", st, POTAL_OK);
    check_status("slot 10 column rewrite (same generation) refused", potal_paired_scale_write(s, 10, 4, 0, 0), POTAL_ERR_CAPACITY);
    check_status("slot 10 unwritten column of the live generation accepted", potal_paired_scale_write(s, 10, 4, d - 1, 0), POTAL_OK);
    check_status("release slot 10 never-written column refused", potal_paired_scale_release(s, 10, 4, d - 2), POTAL_ERR_INVALID_ARGUMENT);
    for (uint32_t col = 0; col + 4 < d && st == POTAL_OK; ++col) {
        st = potal_paired_scale_release(s, 10, 4, col);
    }
    check_status("release slot 10 written columns", st, POTAL_OK);
    check_status("release slot 10 last live column", potal_paired_scale_release(s, 10, 4, d - 1), POTAL_OK);
    check_u64("slot 10 free after its written columns", potal_paired_session_query(s).live_scale_slots, 0);
    check_status("slot 10 rewritten with a new generation", potal_paired_scale_write(s, 10, 5, 0, 0), POTAL_OK);
    check_status("slot 10 released again", potal_paired_scale_release(s, 10, 5, 0), POTAL_OK);
    check_status("fence again", potal_paired_fence(s), POTAL_OK);
    /* Live scale entries keep a healthy session from closing: the hardware cache would hold them. */
    check_status("scale write slot 3 before close", potal_paired_scale_write(s, 3, 9, 0, 0), POTAL_OK);
    check_status("close with a live scale entry refused", potal_paired_session_close(s), POTAL_ERR_BUSY);
    check_status("release slot 3", potal_paired_scale_release(s, 3, 9, 0), POTAL_OK);
    check_status("session close", potal_paired_session_close(s), POTAL_OK);
#if !defined(__riscv)
    {
        /* Host only: on the target this would issue a real GEMM on fake addresses. */
        potal_loop_ws l = two_by_three_loop();
        check_status("regular loop issue after close (host: no hardware)", potal_loop_ws_issue(&l), POTAL_ERR_NO_HARDWARE);
    }
#endif

    /* Failure: a hardware status on INDEX_CONFIG makes the session failed; outstanding work is kept until a fence. */
    memset(&sink_st, 0, sizeof(sink_st));
    check_status("session open (failure case)", potal_paired_session_open(&sink, &s), POTAL_OK);
    check_status("main pair 0 (failure case)", potal_paired_loop_ws_issue(s, &m0), POTAL_OK);
    sink_st.fail_funct = POTAL_FUNCT_INDEX_CONFIG;
    sink_st.fail_status = POTAL_HW_INVALID_ADDRESS;
    check_status("index config answered INVALID_ADDRESS", potal_paired_index_config(s, &cfg0), POTAL_HW_INVALID_ADDRESS);
    q = potal_paired_session_query(s);
    check_u64("session failed with the error and stage", q.failed && q.error == POTAL_HW_INVALID_ADDRESS && q.error_stage != NULL &&
              strcmp(q.error_stage, "index_config") == 0, 1);
    check_u64("outstanding loop kept after the failure", q.outstanding_loops, 1);
    check_u64("no transaction opened by the refused config", q.index_transaction_open, 0);
    check_status("further issue returns the sticky error", potal_paired_loop_ws_issue(s, &m1), POTAL_HW_INVALID_ADDRESS);
    check_status("scale write of a failed session refused", potal_paired_scale_write(s, 0, 1, 0, 0), POTAL_HW_INVALID_ADDRESS);
    sink_st.count = 0;
    check_status("cleanup drains a failed session", potal_paired_session_cleanup(s), POTAL_OK);
    check_u64("cleanup issued one FENCE", sink_st.count == 1 && sink_st.records[0].funct == POTAL_FUNCT_FENCE, 1);
    check_u64("drained", potal_paired_session_query(s).outstanding_loops, 0);
    check_status("close returns the sticky error, accelerator released", potal_paired_session_close(s), POTAL_HW_INVALID_ADDRESS);
    /* Cleanup releases the scale entries a failed session left live. */
    memset(&sink_st, 0, sizeof(sink_st));
    check_status("session open (scale cleanup case)", potal_paired_session_open(&sink, &s), POTAL_OK);
    for (uint32_t col = 0; col < 3; ++col) {
        check_status("scale write before the failure", potal_paired_scale_write(s, 4, 6, col, 0), POTAL_OK);
    }
    sink_st.fail_funct = POTAL_FUNCT_SCALE_WRITE;
    sink_st.fail_status = POTAL_HW_SCALE_CONFLICT;
    check_status("scale write answered SCALE_CONFLICT", potal_paired_scale_write(s, 4, 6, 3, 0), POTAL_HW_SCALE_CONFLICT);
    sink_st.count = 0;
    check_status("cleanup releases the three live columns", potal_paired_session_cleanup(s), POTAL_OK);
    check_u64("three releases, no fence (nothing outstanding)", sink_st.count == 3 && sink_st.records[0].funct == POTAL_FUNCT_SCALE_RELEASE &&
              sink_st.records[2].rs1 == ((2ull << 16) | (6ull << 8) | 4ull), 1);
    check_u64("no live slot after cleanup", potal_paired_session_query(s).live_scale_slots, 0);
    check_status("close after cleanup returns the error", potal_paired_session_close(s), POTAL_HW_SCALE_CONFLICT);
    check_status("session open after a drained failure", potal_paired_session_open(&sink, &s), POTAL_OK);
    check_status("session close", potal_paired_session_close(s), POTAL_OK);
    /* A failed fence (bad response) with outstanding work: close keeps the accelerator owned. */
    memset(&sink_st, 0, sizeof(sink_st));
    check_status("session open (bad fence case)", potal_paired_session_open(&sink, &s), POTAL_OK);
    check_status("main pair 0 (bad fence case)", potal_paired_loop_ws_issue(s, &m0), POTAL_OK);
    sink_st.fail_funct = POTAL_FUNCT_FENCE;
    sink_st.fail_status = POTAL_ERR_BAD_RESPONSE;
    check_status("fence answered badly", potal_paired_fence(s), POTAL_ERR_BAD_RESPONSE);
    check_u64("work still outstanding", potal_paired_session_query(s).outstanding_loops, 1);
    check_status("cleanup refuses to retry a failed fence", potal_paired_session_cleanup(s), POTAL_ERR_BAD_RESPONSE);
    check_status("close of a failed session with outstanding work", potal_paired_session_close(s), POTAL_ERR_BAD_RESPONSE);
    check_status("accelerator stays owned: open refused", potal_paired_session_open(&sink, &s), POTAL_ERR_BUSY);
    (void)other;
}

/* Without a sink the host issues nothing: every issue validates, returns POTAL_ERR_NO_HARDWARE and changes no state. */
static void test_paired_host_no_hardware_before_sessions(void) {
#if !defined(__riscv)
    potal_paired_session *s = NULL;
    check_status("host session open without a sink", potal_paired_session_open(NULL, &s), POTAL_OK);
    potal_paired_loop_ws m0 = h2_main(0);
    check_status("host main issue: no hardware", potal_paired_loop_ws_issue(s, &m0), POTAL_ERR_NO_HARDWARE);
    m0.pair = 2;
    check_status("host main issue: invalid first", potal_paired_loop_ws_issue(s, &m0), POTAL_ERR_INVALID_ARGUMENT);
    const potal_index_config cfg = h2_index_config(0);
    check_status("host index config: no hardware", potal_paired_index_config(s, &cfg), POTAL_ERR_NO_HARDWARE);
    check_status("host scale write: no hardware", potal_paired_scale_write(s, 0, 1, 0, 0), POTAL_ERR_NO_HARDWARE);
    check_status("host tlb flush: no hardware", potal_paired_tlb_flush(s), POTAL_ERR_NO_HARDWARE);
    check_status("host fence: no hardware", potal_paired_fence(s), POTAL_ERR_NO_HARDWARE);
    const potal_paired_session_status q = potal_paired_session_query(s);
    check_u64("host: nothing issued, nothing outstanding, not failed",
              q.commands == 0 && q.outstanding_loops == 0 && q.index_transaction_open == 0 && q.failed == 0 && q.pair[0].resident_b == 0, 1);
    check_status("host session close", potal_paired_session_close(s), POTAL_OK);
#endif
}

/* Managed H2 is command-level software validation, not accelerator timing. */
static potal_h2_weight managed_weight(void) {
    const size_t d = POTAL_HW_DIM;
    const potal_h2_weight w = {FAKE_B, 5 * d, 3 * d - 5, 4 * d,
        (5 * d - 1) * 4 * d + 3 * d - 5};
    return w;
}

#if POTAL_HW_DIM == 32
typedef struct managed_sink_state {
    sink_state trace;
    potal_paired_session *session;
    uint32_t fail_at;               /* absolute command number; also injects CONFIG failures */
    uint32_t reenter, pending_seen, readback_seen, control_seen, mvins, completed_mvins;
    uint64_t early_capability;
    uint32_t early_probes, early_words, early_target, early_trace_reads;
} managed_sink_state;

static potal_status managed_sink(void *ctx, const potal_paired_cmd_record *record) {
    managed_sink_state *st = ctx;
    const potal_status result = sink_cmd(&st->trace, record);
    const potal_h2_status q = potal_paired_h2_query(st->session);
    st->pending_seen |= q.pending_b && q.source_pending;
    st->readback_seen |= q.core.outstanding_readbacks != 0;
    st->control_seen |= q.control_pending;
    if (record->funct == POTAL_FUNCT_MVIN) st->mvins++;
    if (st->early_words) {
        const potal_h2_panel_status *p = &q.panel[st->early_target];
        check_u64("early ownership before every staging and run word",
            q.pending_b && q.source_bound && q.source_pending &&
            p->declared && p->prefetched && p->loading && !p->readable, 1);
        check_u64("early admission has no reader or ACC work",
            p->readers_issued | p->readers_completed | q.core.outstanding_readbacks |
            q.core.outstanding_acc_updates, 0);
        check_u64("early admission retains exactly current Main",
            q.core.outstanding_loops, 1);
        st->early_words--;
    }
    if (st->reenter) {
        st->reenter = 0;
        potal_h2_ref ref = {0};
        potal_paired_session *alias = st->session;
        check_u64("managed callback can observe premarked ownership", q.pending_b, 1);
        check_status("managed reentrant begin", potal_paired_h2_begin(alias, NULL), POTAL_ERR_BUSY);
        check_status("managed reentrant panel", potal_paired_h2_panel_begin(alias, NULL, 0, &ref), POTAL_ERR_BUSY);
        check_status("managed reentrant prepared prefetch", potal_paired_h2_panel_prefetch(alias, ref), POTAL_ERR_BUSY);
        check_status("managed reentrant loop", potal_paired_h2_loop_issue(alias, ref, 0, NULL), POTAL_ERR_BUSY);
        check_status("managed reentrant retire", potal_paired_h2_panel_retire(alias, ref), POTAL_ERR_BUSY);
        check_status("managed reentrant scale", potal_paired_scale_write(alias, 0, 7, 0, 0), POTAL_ERR_BUSY);
        check_status("managed reentrant release", potal_paired_scale_release(alias, 0, 7, 0), POTAL_ERR_BUSY);
        check_status("managed reentrant index config", potal_paired_index_config(alias, NULL), POTAL_ERR_BUSY);
        check_status("managed reentrant index write", potal_paired_index_write(alias, NULL), POTAL_ERR_BUSY);
        check_status("managed reentrant readback", potal_paired_acc_readback_issue(alias, NULL), POTAL_ERR_BUSY);
        check_status("managed reentrant flush", potal_paired_tlb_flush(alias), POTAL_ERR_BUSY);
        check_status("managed reentrant fence", potal_paired_fence(alias), POTAL_ERR_BUSY);
        check_status("managed reentrant cleanup", potal_paired_session_cleanup(alias), POTAL_ERR_BUSY);
        check_status("managed reentrant default close", potal_paired_session_close(alias), POTAL_ERR_BUSY);
        check_status("managed reentrant close", potal_paired_h2_close(&alias), POTAL_ERR_BUSY);
        check_u64("managed reentrant close retains pointer", alias == st->session, 1);
    }
    if (st->fail_at && st->trace.count == st->fail_at) {
        st->fail_at = 0;
        return POTAL_HW_INVALID_ADDRESS;
    }
    return result;
}

static potal_status managed_early_counter(void *ctx, const potal_paired_cmd_record *record,
    uint64_t *value) {
    managed_sink_state *st = ctx; /* The same borrowed context as managed_sink. */
    check_u64("early probe uses bound session context",
        potal_paired_h2_query(st->session).active, 1);
    check_u64("early probe COUNTER126 xd1 rs2zero",
        record->funct == POTAL_FUNCT_COUNTER && record->xd == 1 && !record->rs2, 1);
    if (record->rs1 == POTAL_LDB_ONLY_TRACE_RESET ||
        (record->rs1 >= POTAL_LDB_ONLY_TRACE_READ &&
         record->rs1 < POTAL_LDB_ONLY_TRACE_READ + POTAL_LDB_ONLY_TRACE_FIELDS)) {
        const potal_status status = sink_cmd(&st->trace, record);
        if (record->rs1 == POTAL_LDB_ONLY_TRACE_RESET) *value = 1;
        else {
            const uint64_t field = record->rs1 - POTAL_LDB_ONLY_TRACE_READ;
            /* Literal full-width callback data: protocol validation only,
             * never native hardware timestamps or simulated physical events. */
            *value = field ? UINT64_C(0x100000000) + field : UINT64_C(1) << 14;
            st->early_trace_reads++;
        }
        return status;
    }
    check_u64("early safe setup then exact tagged query", record->rs1,
        st->early_probes++ ? UINT64_C(0x504f4c4400000000) : UINT64_C(0xA));
    check_status("early counter mutation excluded",
        potal_paired_h2_early_enable(st->session, managed_early_counter), POTAL_ERR_BUSY);
    const potal_status status = sink_cmd(&st->trace, record);
    *value = record->rs1 == UINT64_C(0x504f4c4400000000) ? st->early_capability : 0;
    return status;
}

static potal_h2_panel managed_panel(uint32_t pair, uint32_t readers) {
    const potal_h2_panel p = {
        .residency_index = 100 + pair, .k_begin = pair ? 3 * POTAL_HW_DIM : 0,
        .j_begin = POTAL_HW_DIM, .pair = pair, .k_tiles = pair ? 2 : 3,
        .j_tiles = 2, .pad_j = 5, .main_i_tiles = 1, .main_pad_i = 3, .readers = readers,
    };
    return p;
}

static potal_paired_loop_ws managed_main(const potal_h2_panel *p) {
    potal_paired_loop_ws l = h2_main(p->pair);
    l.loop.b = (const int8_t *)((uintptr_t)FAKE_B + p->k_begin * 4 * POTAL_HW_DIM + p->j_begin);
    l.loop.ldb = 4 * POTAL_HW_DIM;
    l.loop.k_tiles = p->k_tiles;
    l.loop.passes_per_block = p->k_tiles;
    l.loop.pad_i = p->main_pad_i;
    l.loop.pad_j = p->pad_j;
    return l;
}

static potal_paired_loop_ws managed_residual(const potal_h2_panel *p) {
    potal_paired_loop_ws l = h2_residual(p->pair, 1);
    l.loop.pad_i = p->main_pad_i;
    l.loop.pad_j = p->pad_j;
    l.a_offset_tiles = p->k_tiles;
    return l;
}

static potal_status managed_scales(potal_paired_session *s, const potal_paired_loop_ws *pl, int release) {
    const potal_loop_ws *l = &pl->loop;
    for (uint32_t b = 0; b < l->k_tiles / l->passes_per_block; ++b) {
        for (uint32_t j = 0; j < l->j_tiles; ++j) {
            const uint32_t slot = l->scale_base + b * l->j_tiles + j;
            const uint32_t columns = POTAL_HW_DIM - (j + 1 == l->j_tiles ? l->pad_j : 0);
            for (uint32_t col = 0; col < columns; ++col) {
                const potal_status status = release ?
                    potal_paired_scale_release(s, slot, l->scale_generation, col) :
                    potal_paired_scale_write(s, slot, l->scale_generation, col, 0);
                if (status != POTAL_OK) return status;
            }
        }
    }
    return POTAL_OK;
}

static potal_index_config managed_index(const potal_h2_panel *p) {
    const potal_index_config cfg = {
        .pair = p->pair,
        .b_sp_base = (2 * p->pair + 2) * POTAL_HW_BANK_ROWS - p->k_tiles * p->j_tiles * POTAL_HW_DIM,
        .j_tiles = p->j_tiles, .source_k_rows = p->k_tiles * POTAL_HW_DIM,
        .active_rows = 3 * POTAL_HW_DIM / 2,
    };
    return cfg;
}

static void managed_open(managed_sink_state *st, potal_paired_session **s) {
    memset(st, 0, sizeof(*st));
    const potal_paired_sink sink = {managed_sink, st};
    const potal_h2_weight w = managed_weight();
    check_status("managed session open", potal_paired_session_open(&sink, s), POTAL_OK);
    st->session = *s;
    check_status("managed begin", potal_paired_h2_begin(*s, &w), POTAL_OK);
}

static void managed_program_equal(const sink_state *trace, uint32_t start,
    const potal_gemm_ws_program *p, int suppress) {
    const potal_rocc_cmd expected[] = {p->config_ld_a, p->config_ld_b, p->config_ex, p->config_st,
        p->bounds, p->addrs_ab, p->addrs_dc, p->strides_ab, p->strides_dc, p->loop_ws};
    const uint32_t functs[] = {0, 0, 0, 0, 9, 10, 11, 12, 13, 8};
    check_u64("managed ordinary program has ten commands", trace->count - start, 10);
    for (uint32_t i = 0; i < 10; ++i) {
        const potal_paired_cmd_record *got = &trace->records[start + i];
        check_u64("managed only B address differs", got->funct == functs[i] && got->xd == 0 &&
            got->rs1 == expected[i].rs1 && got->rs2 == (suppress && i == 5 ? 0 : expected[i].rs2), 1);
    }
}

static void test_managed_completion(void) {
    managed_sink_state st;
    potal_paired_session *s = NULL;
    memset(&st, 0, sizeof(st));
    const potal_paired_sink sink = {managed_sink, &st};
    const potal_h2_weight weight = managed_weight();
    check_status("completion session open", potal_paired_session_open(&sink, &s), POTAL_OK);
    st.session = s;
    check_status("completion older legacy drain", potal_paired_fence(s), POTAL_OK);
    check_status("completion managed begin", potal_paired_h2_begin(s, &weight), POTAL_OK);
    check_u64("completion excludes legacy drain", potal_paired_h2_query(s).completion_calls, 0);
    st.trace.count = 0;
    const char *requested = getenv("POTAL_H2_COMPLETION_REUSE");
    const uint32_t reuse = requested && strcmp(requested, "1") == 0;
    check_u64("completion mode matches independent requested value",
        potal_paired_h2_query(s).completion_reuse, reuse);
    check_status("completion initial certificate", potal_paired_fence(s), POTAL_OK);
    check_status("completion repeated checked Fence", potal_paired_fence(s), POTAL_OK);
    check_u64("completion unchanged reconciliation reused",
        potal_paired_h2_query(s).completion_reuses, reuse);
    check_status("completion Scale effect", potal_paired_scale_write(s, 20, 7, 0, 0), POTAL_OK);
    check_status("completion second live column", potal_paired_scale_write(s, 20, 7, 1, 0), POTAL_OK);
    check_status("completion Scale effect needs reconciliation", potal_paired_fence(s), POTAL_OK);
    check_u64("completion Scale Write invalidates certificate",
        potal_paired_h2_query(s).completion_reuses, reuse);
    check_status("completion first checked Release", potal_paired_scale_release(s, 20, 7, 0), POTAL_OK);
    check_status("completion second checked Release", potal_paired_scale_release(s, 20, 7, 1), POTAL_OK);
    check_u64("completion both real Releases may reuse host state",
        potal_paired_h2_query(s).completion_reuses, 3 * reuse);
    check_status("completion new Scale generation", potal_paired_scale_write(s, 20, 8, 0, 0), POTAL_OK);
    check_status("completion generation effect needs reconciliation", potal_paired_fence(s), POTAL_OK);
    check_u64("completion new generation invalidates certificate",
        potal_paired_h2_query(s).completion_reuses, 3 * reuse);
    const uint32_t before = st.trace.count;
    check_status("completion stale Release still rejected",
        potal_paired_scale_release(s, 20, 7, 0), POTAL_ERR_INVALID_ARGUMENT);
    check_u64("completion stale Release has no effects", st.trace.count, before);
    check_status("completion matching Release", potal_paired_scale_release(s, 20, 8, 0), POTAL_OK);
    check_status("completion final checked Fence", potal_paired_fence(s), POTAL_OK);
    const potal_h2_status q = potal_paired_h2_query(s);
    check_u64("completion all eight managed drain responses retained", q.completion_calls, 8);
    check_u64("completion core also retains legacy drain", q.core.drains, 9);
    check_u64("completion five redundant reconciliations", q.completion_reuses, 5 * reuse);
    check_u64("completion bookkeeping count", q.completion_bookkeeping, 8 - 5 * reuse);
#if defined(__riscv)
    check_u64("completion CPU fences all retained", q.completion_cpu_fences, 8);
#else
    check_u64("completion native reports no CPU fences", q.completion_cpu_fences, 0);
#endif
    const uint32_t expected[] = {POTAL_FUNCT_FENCE, POTAL_FUNCT_FENCE,
        POTAL_FUNCT_SCALE_WRITE, POTAL_FUNCT_SCALE_WRITE, POTAL_FUNCT_FENCE,
        POTAL_FUNCT_SCALE_RELEASE, POTAL_FUNCT_SCALE_RELEASE, POTAL_FUNCT_SCALE_WRITE,
        POTAL_FUNCT_FENCE, POTAL_FUNCT_SCALE_RELEASE, POTAL_FUNCT_FENCE};
    check_u64("completion all eleven raw commands retained", st.trace.count, 11);
    for (uint32_t i = 0; i < st.trace.count && i < 11; ++i) {
        const uint32_t generation = i >= 7 ? 8 : 7;
        const uint32_t column = i == 3 || i == 6;
        const uint64_t key = 20 | ((uint64_t)generation << 8) | ((uint64_t)column << 16);
        check_u64("completion raw opcode", st.trace.records[i].funct, expected[i]);
        check_u64("completion raw response requested", st.trace.records[i].xd, 1);
        check_u64("completion raw key",
            st.trace.records[i].rs1, expected[i] == POTAL_FUNCT_FENCE ? 0 : key);
        check_u64("completion raw payload", st.trace.records[i].rs2, 0);
    }
    check_status("completion safe close", potal_paired_h2_close(&s), POTAL_OK);
}

static void test_managed_h2(void) {
    managed_sink_state st;
    potal_paired_session *s = NULL;
    managed_open(&st, &s);
    const potal_h2_weight w = managed_weight();
    potal_h2_panel p0 = managed_panel(0, 2), p1 = managed_panel(1, 1);
    potal_h2_ref r0 = {0}, r1 = {0};
    potal_paired_loop_ws m0 = managed_main(&p0), m1 = managed_main(&p1), residual = managed_residual(&p0);
    potal_index_config index = managed_index(&p0);
    check_status("managed duplicate begin", potal_paired_h2_begin(s, &w), POTAL_ERR_BUSY);
    check_status("managed initial flush", potal_paired_tlb_flush(s), POTAL_OK);
    check_u64("managed flush control premarked", st.control_seen, 1);
    check_status("managed flush requires drain", potal_paired_h2_panel_begin(s, &p0, 0, &r0), POTAL_ERR_BUSY);
    check_status("managed initial fence", potal_paired_fence(s), POTAL_OK);

    const uint32_t before = st.trace.count;
    potal_h2_panel bad = p0;
    bad.k_begin = UINT64_MAX - 31;
    check_status("managed K offset overflow refused", potal_paired_h2_panel_begin(s, &bad, 1, &r0), POTAL_ERR_INVALID_ARGUMENT);
    bad = p0; bad.j_begin++;
    check_status("managed unaligned J refused", potal_paired_h2_panel_begin(s, &bad, 1, &r0), POTAL_ERR_INVALID_ARGUMENT);
    bad = p0; bad.k_tiles = 6;
    check_status("managed source extent refused", potal_paired_h2_panel_begin(s, &bad, 1, &r0), POTAL_ERR_INVALID_ARGUMENT);
    bad = p0; bad.readers = 0;
    check_status("managed zero readers refused", potal_paired_h2_panel_begin(s, &bad, 0, &r0), POTAL_ERR_INVALID_ARGUMENT);
    bad = p0; bad.main_i_tiles = POTAL_HW_BANK_ROWS;
    check_status("managed physical capacity refused", potal_paired_h2_panel_begin(s, &bad, 0, &r0), POTAL_ERR_CAPACITY);
    check_u64("managed rejected panels have no effects", st.trace.count, before);

    check_status("managed normal declaration", potal_paired_h2_panel_begin(s, &p0, 0, &r0), POTAL_OK);
    check_status("managed declaration is not publication", potal_paired_h2_panel_get(s, r0), POTAL_ERR_BUSY);
    check_status("managed declared reader not complete after fence", potal_paired_fence(s), POTAL_OK);
    check_u64("managed unissued reader stays uncompleted", potal_paired_h2_query(s).panel[0].readers_completed, 0);
    check_status("managed retire unissued refused", potal_paired_h2_panel_retire(s, r0), POTAL_ERR_BUSY);
    check_status("managed no default issue bypass", potal_paired_loop_ws_issue(s, &m0), POTAL_ERR_BUSY);
    check_status("managed no default close bypass", potal_paired_session_close(s), POTAL_ERR_BUSY);
    check_status("managed no standalone scale bypass", potal_scale_write(0, 7, 0, 0), POTAL_ERR_BUSY);
    check_status("managed no standalone release bypass", potal_scale_release(0, 7, 0), POTAL_ERR_BUSY);
    check_status("managed no bias bypass", potal_paired_acc_bias_issue(s, NULL), POTAL_ERR_BUSY);
    check_status("managed no flush after declaration", potal_paired_tlb_flush(s), POTAL_ERR_BUSY);
    check_status("managed missing scales refused", potal_paired_h2_loop_issue(s, r0, 0, &m0), POTAL_ERR_INVALID_ARGUMENT);
    potal_h2_ref wrong_ref = r0; wrong_ref.pair = 1;
    check_status("managed reference cannot select another pair", potal_paired_h2_loop_issue(s, wrong_ref, 0, &m0), POTAL_ERR_INVALID_ARGUMENT);
    check_status("managed Main scales", managed_scales(s, &m0, 0), POTAL_OK);
    check_status("managed Residual scales", managed_scales(s, &residual, 0), POTAL_OK);
    check_status("managed Residual before Main refused", potal_paired_h2_loop_issue(s, r0, 1, &residual), POTAL_ERR_INVALID_ARGUMENT);
    potal_gemm_ws_program expected;
    check_status("managed normal reference encode", potal_paired_loop_ws_encode(&m0, &expected), POTAL_OK);
    uint32_t start = st.trace.count;
    check_status("managed normal Main", potal_paired_h2_loop_issue(s, r0, 0, &m0), POTAL_OK);
    managed_program_equal(&st.trace, start, &expected, 0);
    check_status("managed Main issue is not publication", potal_paired_h2_panel_get(s, r0), POTAL_ERR_BUSY);
    check_status("managed duplicate Main refused", potal_paired_h2_loop_issue(s, r0, 0, &m0), POTAL_ERR_INVALID_ARGUMENT);
    potal_index_config wrong_index = index; wrong_index.b_sp_base += 32;
    check_status("managed Index identity checked", potal_paired_index_config(s, &wrong_index), POTAL_ERR_INVALID_ARGUMENT);
    check_status("managed Index config without Main fence", potal_paired_index_config(s, &index), POTAL_OK);
    check_status("managed no prefetch inside Index transaction", potal_paired_h2_panel_begin(s, &p1, 1, &r1), POTAL_ERR_BUSY);
    for (uint32_t i = 0; i < index.active_rows; ++i) {
        const potal_index_write entry = {i, i};
        check_status("managed Index entry", potal_paired_index_write(s, &entry), POTAL_OK);
    }
    start = st.trace.count;
    check_status("managed prepare next before current Residual", potal_paired_h2_panel_begin(s, &p1, 0, &r1), POTAL_OK);
    potal_h2_ref bad_prepared = r1; bad_prepared.generation++;
    check_status("managed stale prepared reference", potal_paired_h2_panel_prefetch(s, bad_prepared), POTAL_ERR_INVALID_ARGUMENT);
    potal_paired_loop_ws invalid_current = residual; invalid_current.loop.j_tiles++;
    check_status("managed prepared peer survives current refusal",
        potal_paired_h2_loop_issue(s, r0, 1, &invalid_current), POTAL_ERR_INVALID_ARGUMENT);
    check_u64("managed preparation and refusal issue no command", st.trace.count, start);
    potal_h2_status prepared = potal_paired_h2_query(s);
    check_u64("managed current identity retained", prepared.panel[0].ref.generation, r0.generation);
    check_u64("managed next identity retained unused", prepared.panel[1].ref.generation, r1.generation);
    check_u64("managed prepared is neither loading nor published",
        prepared.pending_b | prepared.panel[1].loading | prepared.panel[1].readable |
        prepared.panel[1].prefetched | prepared.panel[1].readers_issued | prepared.quarantined, 0);
    check_status("managed healthy preparation cannot be discarded by cleanup", potal_paired_session_cleanup(s), POTAL_ERR_BUSY);
    check_status("managed prepared close retains owner", potal_paired_h2_close(&s), POTAL_ERR_BUSY);
    check_status("managed Residual chained without fence", potal_paired_h2_loop_issue(s, r0, 1, &residual), POTAL_OK);
    check_status("managed duplicate Residual refused", potal_paired_h2_loop_issue(s, r0, 1, &residual), POTAL_ERR_INVALID_ARGUMENT);
    check_status("managed reader count bounded", potal_paired_h2_loop_issue(s, r0, 2, &residual), POTAL_ERR_INVALID_ARGUMENT);
    check_u64("managed both readers in flight", potal_paired_h2_query(s).core.outstanding_loops, 2);

    st.reenter = 1;
    start = st.trace.count;
    check_status("managed prepared prefetch with peer readers in flight", potal_paired_h2_panel_prefetch(s, r1), POTAL_OK);
    check_u64("managed prefetch CONFIG plus four tiles", st.trace.count - start, 5);
    check_u64("managed CONFIG state0 identity block32", st.trace.records[start].rs1, UINT64_C(0x00200101));
    check_u64("managed CONFIG original stride", st.trace.records[start].rs2, 128);
    for (uint32_t k = 0; k < 2; ++k) for (uint32_t j = 0; j < 2; ++j) {
        const potal_paired_cmd_record *rec = &st.trace.records[start + 1 + k * 2 + j];
        check_u64("managed MVIN exact source from original matrix", rec->rs1,
            UINT64_C(0x20000) + (96 + k * 32) * 128 + 32 + j * 32);
        check_u64("managed MVIN rear pair1 row and J tail", rec->rs2,
            (UINT64_C(32) << 48) | ((uint64_t)(j ? 27 : 32) << 32) |
            (4 * POTAL_HW_BANK_ROWS - 128 + (k * 2 + j) * 32));
        check_u64("managed raw MVIN opcode", rec->funct, POTAL_FUNCT_MVIN);
    }
    check_u64("managed pending was visible before first CONFIG", st.pending_seen, 1);
    /* A deterministic sink-model DMA completion event, not a hardware claim.
     * Software sees no barrier response at this event and cannot publish B. */
    st.completed_mvins = st.mvins;
    check_u64("managed sink load event contains four tiles", st.completed_mvins, 4);
    check_status("managed physical progress cannot publish", potal_paired_h2_panel_get(s, r1), POTAL_ERR_BUSY);
    check_u64("managed observation cannot complete peer readers", potal_paired_h2_query(s).panel[0].readers_completed, 0);
    check_status("managed pending B blocks Main", potal_paired_h2_loop_issue(s, r1, 0, &m1), POTAL_ERR_BUSY);
    check_status("managed pending B blocks metadata", potal_paired_scale_write(s, 9, 1, 0, 0), POTAL_ERR_BUSY);
    check_status("managed pending B blocks Index config", potal_paired_index_config(s, &index), POTAL_ERR_BUSY);
    const potal_index_write extra_index = {0, 0};
    check_status("managed pending B blocks Index write", potal_paired_index_write(s, &extra_index), POTAL_ERR_BUSY);
    check_status("managed only one prefetch pending", potal_paired_h2_panel_begin(s, &p1, 1, &r1), POTAL_ERR_BUSY);
    check_status("managed prepared prefetch cannot repeat", potal_paired_h2_panel_prefetch(s, r1), POTAL_ERR_BUSY);
    potal_acc_readback rb = {.acc_bank = 0, .i_tiles = 1, .j_tiles = 2, .pad_i = 3,
        .pad_j = 5, .c = FAKE_C, .ldc = 64};
    potal_acc_readback alias = rb; alias.c = (int32_t *)(uintptr_t)FAKE_B;
    check_status("managed readback cannot overwrite borrowed B", potal_paired_acc_readback_issue(s, &alias), POTAL_ERR_INVALID_ARGUMENT);
    check_status("managed current readback allowed during prefetch", potal_paired_acc_readback_issue(s, &rb), POTAL_OK);
    check_u64("managed readback premarked before CONFIG", st.readback_seen, 1);
    const uint64_t drains = potal_paired_h2_query(s).core.drains;
    check_status("managed invalid Release not a certificate", potal_paired_scale_release(s, 0, 8, 0), POTAL_ERR_INVALID_ARGUMENT);
    check_u64("managed invalid Release leaves pending B", potal_paired_h2_query(s).pending_b, 1);
    check_status("managed valid Release publishes globally", potal_paired_scale_release(s, 0, 7, 0), POTAL_OK);
    potal_h2_status q = potal_paired_h2_query(s);
    check_u64("managed one global certificate", q.core.drains, drains + 1);
    check_u64("managed peer readers completed once", q.panel[0].readers_completed, 2);
    check_u64("managed prefetch did not complete unissued Main", q.panel[1].readers_completed, 0);
    check_u64("managed pending source cleared by certificate", q.pending_b | q.source_pending, 0);
    check_status("managed prefetched get published", potal_paired_h2_panel_get(s, r1), POTAL_OK);
    check_status("managed retire blocked by global live scales", potal_paired_h2_panel_retire(s, r0), POTAL_ERR_BUSY);
    for (uint32_t j = 0; j < 2; ++j) for (uint32_t col = 0; col < (j ? 27u : 32u); ++col) {
        if (j == 0 && col == 0) continue;
        check_status("managed remaining Main scale release", potal_paired_scale_release(s, j, 7, col), POTAL_OK);
    }
    check_status("managed Residual release", managed_scales(s, &residual, 1), POTAL_OK);
    potal_index_config final_index = index; final_index.active_rows = 1;
    check_status("managed final Index transaction", potal_paired_index_config(s, &final_index), POTAL_OK);
    check_status("managed retirement requires closed Index transaction", potal_paired_h2_panel_retire(s, r0), POTAL_ERR_BUSY);
    check_status("managed final Index slot", potal_paired_index_write(s, &extra_index), POTAL_OK);
    check_status("managed retire completed peer", potal_paired_h2_panel_retire(s, r0), POTAL_OK);
    check_status("managed retire prefetched but unissued reader refused", potal_paired_h2_panel_retire(s, r1), POTAL_ERR_BUSY);
    check_u64("managed retirement preserves other pair", potal_paired_h2_query(s).panel[1].readable, 1);

    check_status("managed next Main scales", managed_scales(s, &m1, 0), POTAL_OK);
    potal_paired_loop_ws wrong = m1; wrong.loop.b = NULL;
    check_status("managed prefetched Main still requires real B", potal_paired_h2_loop_issue(s, r1, 0, &wrong), POTAL_ERR_INVALID_ARGUMENT);
    wrong = m1; wrong.loop.b = FAKE_B;
    check_status("managed wrong original B rejected", potal_paired_h2_loop_issue(s, r1, 0, &wrong), POTAL_ERR_INVALID_ARGUMENT);
    wrong = m1; wrong.loop.ldb = 96;
    check_status("managed wrong real stride rejected", potal_paired_h2_loop_issue(s, r1, 0, &wrong), POTAL_ERR_INVALID_ARGUMENT);
    wrong = m1; wrong.loop.pad_k = 1;
    check_status("managed Main K padding forbidden", potal_paired_h2_loop_issue(s, r1, 0, &wrong), POTAL_ERR_INVALID_ARGUMENT);
    wrong = m1; wrong.resident_b = 1;
    check_status("managed old resident signature forbidden", potal_paired_h2_loop_issue(s, r1, 0, &wrong), POTAL_ERR_INVALID_ARGUMENT);
    check_status("managed prefetch reference encode", potal_paired_loop_ws_encode(&m1, &expected), POTAL_OK);
    st.trace.count = 0; /* Bound storage for the exact ten-command comparison. */
    check_status("managed prefetched Main issues ordinary program", potal_paired_h2_loop_issue(s, r1, 0, &m1), POTAL_OK);
    managed_program_equal(&st.trace, 0, &expected, 1);
    check_u64("managed no resident extension command", sink_has(&st.trace, POTAL_FUNCT_RESIDENT_VIEW), 0);
    check_status("managed release completes Main", managed_scales(s, &m1, 1), POTAL_OK);
    check_status("managed retire pair1", potal_paired_h2_panel_retire(s, r1), POTAL_OK);

    const potal_h2_ref stale = r0;
    p0.readers = 1; p0.residency_index = 102;
    st.trace.count = 0;
    check_status("managed next generation pair0 prefetch", potal_paired_h2_panel_begin(s, &p0, 1, &r0), POTAL_OK);
    check_u64("managed generation grows", r0.generation > stale.generation, 1);
    for (uint32_t k = 0; k < 3; ++k) for (uint32_t j = 0; j < 2; ++j) {
        const potal_paired_cmd_record *rec = &st.trace.records[1 + k * 2 + j];
        check_u64("managed pair0 rear layout", rec->rs2 & UINT64_C(0xffffffff),
            2 * POTAL_HW_BANK_ROWS - 192 + (k * 2 + j) * 32);
        check_u64("managed pair0 source stride", rec->rs1, UINT64_C(0x20000) + k * 32 * 128 + 32 + j * 32);
    }
    check_status("managed stale get refused", potal_paired_h2_panel_get(s, stale), POTAL_ERR_INVALID_ARGUMENT);
    check_status("managed stale retirement refused", potal_paired_h2_panel_retire(s, stale), POTAL_ERR_INVALID_ARGUMENT);
    check_u64("managed stale retire cannot clear new B", potal_paired_h2_query(s).panel[0].loading, 1);
    check_status("managed explicit Fence publishes", potal_paired_fence(s), POTAL_OK);
    check_status("managed new Main scales", managed_scales(s, &m0, 0), POTAL_OK);
    check_status("managed new Main", potal_paired_h2_loop_issue(s, r0, 0, &m0), POTAL_OK);
    check_status("managed new scale release", managed_scales(s, &m0, 1), POTAL_OK);
    check_status("managed new retirement", potal_paired_h2_panel_retire(s, r0), POTAL_OK);
    check_status("managed retirement preserves ACC readback", potal_paired_acc_readback_issue(s, &rb), POTAL_OK);
    check_status("managed close waits for readback", potal_paired_h2_close(&s), POTAL_ERR_BUSY);
    check_status("managed final readback Fence", potal_paired_fence(s), POTAL_OK);
    check_status("managed close consumes session", potal_paired_h2_close(&s), POTAL_OK);
    check_u64("managed pointer consumed", s == NULL, 1);

    managed_open(&st, &s);
    check_status("managed another owner panel", potal_paired_h2_panel_begin(s, &p0, 1, &r0), POTAL_OK);
    check_u64("managed owner incarnation changes", r0.owner != stale.owner, 1);
    check_status("managed old owner rejected", potal_paired_h2_panel_get(s, stale), POTAL_ERR_INVALID_ARGUMENT);
    check_status("managed owner publication", potal_paired_fence(s), POTAL_OK);
    check_status("managed owner scales", managed_scales(s, &m0, 0), POTAL_OK);
    check_status("managed owner Main", potal_paired_h2_loop_issue(s, r0, 0, &m0), POTAL_OK);
    check_status("managed owner release", managed_scales(s, &m0, 1), POTAL_OK);
    check_status("managed owner retirement", potal_paired_h2_panel_retire(s, r0), POTAL_OK);
    check_status("managed owner close", potal_paired_h2_close(&s), POTAL_OK);
}

/* Failure positions cover first CONFIG, first/partial/final MVIN, partial
 * program and readback. A rejected last command is not a fully issued reader.
 */
static void test_managed_recovery(void) {
    for (uint32_t kind = 0; kind < 6; ++kind) {
        const uint32_t last = kind == 0 || kind == 4 ? 7 : kind == 1 || kind == 5 ? 10 : kind == 2 ? 3 : 1;
        for (uint32_t position = 1; position <= last; ++position) {
            managed_sink_state st;
            potal_paired_session *s = NULL;
            managed_open(&st, &s);
            potal_h2_panel panel = managed_panel(0, kind == 5 ? 2 : 1);
            potal_h2_ref ref = {0};
            potal_paired_loop_ws main_loop = managed_main(&panel);
            potal_paired_loop_ws residual_loop = managed_residual(&panel);
            check_status("managed recovery live Scale", potal_paired_scale_write(s, 20, 4, 0, 0), POTAL_OK);
            if (kind) {
                check_status("managed recovery declare", potal_paired_h2_panel_begin(s, &panel, 0, &ref), POTAL_OK);
                check_status("managed recovery scales", managed_scales(s, &main_loop, 0), POTAL_OK);
                if (kind == 5)
                    check_status("managed recovery Residual scales", managed_scales(s, &residual_loop, 0), POTAL_OK);
                if (kind == 2 || kind == 3 || kind == 5)
                    check_status("managed recovery Main", potal_paired_h2_loop_issue(s, ref, 0, &main_loop), POTAL_OK);
            }
            if (kind == 5) {
                const potal_index_config cfg = managed_index(&panel);
                check_status("managed recovery Index", potal_paired_index_config(s, &cfg), POTAL_OK);
                for (uint32_t i = 0; i < cfg.active_rows; ++i) {
                    const potal_index_write entry = {i, i};
                    check_status("managed recovery Index slot", potal_paired_index_write(s, &entry), POTAL_OK);
                }
                const potal_h2_panel next = managed_panel(1, 1);
                potal_h2_ref next_ref;
                check_status("managed recovery prepared successor", potal_paired_h2_panel_begin(s, &next, 0, &next_ref), POTAL_OK);
            }
            st.fail_at = st.trace.count + position;
            potal_status failed;
            if (kind == 0) failed = potal_paired_h2_panel_begin(s, &panel, 1, &ref);
            else if (kind == 1) failed = potal_paired_h2_loop_issue(s, ref, 0, &main_loop);
            else if (kind == 4) failed = potal_paired_h2_panel_prefetch(s, ref);
            else if (kind == 5) failed = potal_paired_h2_loop_issue(s, ref, 1, &residual_loop);
            else if (kind == 2) {
                const potal_acc_readback rb = {.acc_bank = 0, .i_tiles = 1, .j_tiles = 2,
                    .pad_i = 3, .pad_j = 5, .c = FAKE_C, .ldc = 64};
                failed = potal_paired_acc_readback_issue(s, &rb);
            } else {
                const potal_index_config cfg = managed_index(&panel);
                failed = potal_paired_index_config(s, &cfg);
            }
            check_status("managed injected answered failure", failed, POTAL_HW_INVALID_ADDRESS);
            potal_h2_status q = potal_paired_h2_query(s);
            check_u64("managed first failure is sticky", q.core.failed && q.core.error == POTAL_HW_INVALID_ADDRESS, 1);
            check_u64("managed failed operation never publishes", q.panel[0].readable, 0);
            if (kind == 0 || kind == 4) {
                check_u64("managed partial B owns pending and source", q.pending_b && q.source_pending, 1);
                check_u64("managed attempted MVIN count", q.attempted_b_mvins, position - 1);
                check_u64("managed partial B has zero old loop counters", q.core.outstanding_loops, 0);
            } else if (kind == 1 || kind == 5) {
                check_u64("managed partial program premarked", q.core.outstanding_loops, kind == 5 ? 2 : 1);
                check_u64("managed failed program not issued reader", q.panel[0].readers_issued, kind == 5 ? 1 : 0);
                if (kind == 5)
                    check_u64("managed failed current retains prepared successor", q.panel[1].declared, 1);
            } else if (kind == 2) {
                check_u64("managed partial readback premarked", q.core.outstanding_readbacks, 1);
            } else {
                check_u64("managed failed metadata has control ownership", q.control_pending, 1);
            }
            check_status("managed failed direct Fence refused", potal_paired_fence(s), POTAL_HW_INVALID_ADDRESS);
            check_status("managed failed close retains", potal_paired_h2_close(&s), POTAL_HW_INVALID_ADDRESS);
            uint32_t start = st.trace.count;
            check_status("managed one recoverable cleanup", potal_paired_session_cleanup(s), POTAL_OK);
            check_u64("managed cleanup starts with Fence despite zero old counters",
                st.trace.records[start].funct, POTAL_FUNCT_FENCE);
            uint32_t fences = 0;
            for (uint32_t i = start; i < st.trace.count; ++i)
                fences += st.trace.records[i].funct == POTAL_FUNCT_FENCE;
            check_u64("managed cleanup has exactly one Fence", fences, 1);
            q = potal_paired_h2_query(s);
            check_u64("managed cleanup clears actual pending and Scale work",
                q.pending_b | q.source_pending | q.control_pending | q.core.live_scale_slots |
                q.core.outstanding_loops | q.core.outstanding_readbacks, 0);
            check_u64("managed cleanup discards failed identity", q.panel[0].declared, 0);
            check_u64("managed cleanup discards prepared successor", q.panel[1].declared, 0);
            check_u64("managed cleaned retains source until close", q.cleaned && q.source_bound, 1);
            start = st.trace.count;
            check_status("managed cleanup idempotent without retry", potal_paired_session_cleanup(s), POTAL_OK);
            check_u64("managed second cleanup emits nothing", st.trace.count, start);
            check_status("managed cleaned close returns original error", potal_paired_h2_close(&s), POTAL_HW_INVALID_ADDRESS);
            check_u64("managed cleaned close consumes", s == NULL, 1);
        }
    }
}

static void managed_fatal_case(uint32_t kind) {
    managed_sink_state st;
    potal_paired_session *s = NULL;
    managed_open(&st, &s);
    potal_h2_panel panel = managed_panel(0, 1);
    potal_h2_ref ref = {0};
    check_status("managed fatal live Scale", potal_paired_scale_write(s, 20, 4, 0, 0), POTAL_OK);
    const potal_status first = kind == 1 || kind == 3 ? POTAL_HW_INVALID_ADDRESS : POTAL_ERR_BAD_RESPONSE;
    if (kind == 1 || kind == 3) st.fail_at = st.trace.count + 3;
    check_status("managed fatal prefetch", potal_paired_h2_panel_begin(s, &panel, 1, &ref),
        kind == 1 || kind == 3 ? POTAL_HW_INVALID_ADDRESS : POTAL_OK);
    st.trace.fail_funct = kind == 2 || kind == 3 ? POTAL_FUNCT_SCALE_RELEASE : POTAL_FUNCT_FENCE;
    st.trace.fail_status = POTAL_ERR_BAD_RESPONSE;
    if (kind == 0) check_status("managed unknown Fence", potal_paired_fence(s), POTAL_ERR_BAD_RESPONSE);
    else if (kind == 2)
        check_status("managed unknown Release", potal_paired_scale_release(s, 20, 4, 0), POTAL_ERR_BAD_RESPONSE);
    else check_status("managed cleanup failure", potal_paired_session_cleanup(s), POTAL_ERR_BAD_RESPONSE);
    const potal_h2_status q = potal_paired_h2_query(s);
    check_u64("managed independent unknown latch and quarantine", q.unknown_completion && q.quarantined, 1);
    check_status("managed quarantine preserves first error", q.core.error, first);
    check_u64("managed quarantine never publishes", q.panel[0].readable, 0);
    check_u64("managed quarantine retains source binding", q.source_bound, 1);
    const uint32_t start = st.trace.count;
    check_status("managed quarantine Fence no retry", potal_paired_fence(s), first);
    check_status("managed quarantine cleanup no retry", potal_paired_session_cleanup(s), first);
    check_status("managed quarantine issue refuses", potal_paired_h2_panel_begin(s, &panel, 1, &ref), first);
    check_status("managed quarantine get refuses", potal_paired_h2_panel_get(s, ref), first);
    check_status("managed quarantine retire refuses", potal_paired_h2_panel_retire(s, ref), first);
    check_status("managed quarantine close refuses", potal_paired_h2_close(&s), first);
    check_u64("managed quarantine retains opaque handle", s != NULL, 1);
    check_u64("managed quarantine has no new commands", st.trace.count, start);
    potal_paired_session *other = NULL;
    const potal_paired_sink sink = {managed_sink, &st};
    check_status("managed quarantine retains global owner", potal_paired_session_open(&sink, &other), POTAL_ERR_BUSY);
}

static void test_managed_quarantine(void) {
    for (uint32_t kind = 0; kind < 4; ++kind) {
#if defined(__unix__)
        fflush(NULL);
        const int before = failures;
        const pid_t child = fork();
        if (child == 0) {
            managed_fatal_case(kind);
            fflush(NULL);
            _exit(failures == before ? 0 : 1);
        }
        check_u64("managed fatal case forked", child >= 0, 1);
        if (child < 0) continue;
        int status = 0;
        pid_t result;
        do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
        check_u64("managed isolated quarantine case", result == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, 1);
#else
        check_u64("managed quarantine requires fork support, not skipped", 0, 1);
        (void)kind;
        (void)managed_fatal_case;
#endif
    }
}

static void early_prepare(managed_sink_state *st, potal_paired_session **s, uint32_t current_pair,
    uint32_t readers, potal_h2_panel *current, potal_h2_panel *next,
    potal_h2_ref *current_ref, potal_h2_ref *next_ref) {
    managed_open(st, s);
    st->early_capability = UINT64_C(0x504f4c4401200001);
    check_status("early exact capability enables",
        potal_paired_h2_early_enable(*s, managed_early_counter), POTAL_OK);
    *current = managed_panel(0, readers);
    *next = managed_panel(1, 1);
    current->pair = current_pair;
    next->pair = 1 - current_pair;
    check_status("early current declaration", potal_paired_h2_panel_begin(*s, current, 0, current_ref), POTAL_OK);
    check_status("early successor prepared before Main", potal_paired_h2_panel_begin(*s, next, 0, next_ref), POTAL_OK);
    const uint32_t before = st->trace.count;
    check_status("early cannot load before current Main", potal_paired_h2_panel_prefetch(*s, *next_ref), POTAL_ERR_BUSY);
    check_u64("early pre-Main refusal has no effects", st->trace.count, before);
    const potal_paired_loop_ws main = managed_main(current);
    check_status("early current scales", managed_scales(*s, &main, 0), POTAL_OK);
    check_status("early current Main", potal_paired_h2_loop_issue(*s, *current_ref, 0, &main), POTAL_OK);
}

static void test_early_trace(void) {
    managed_sink_state st;
    potal_paired_session *s = NULL;
    managed_open(&st, &s);
    check_status("early trace requires enabled capability",
        potal_paired_h2_early_trace_begin(s), POTAL_ERR_UNSUPPORTED);
    st.early_capability = POTAL_LDB_ONLY_CAPABILITY;
    check_status("early trace capability", potal_paired_h2_early_enable(s, managed_early_counter), POTAL_OK);
    check_status("early trace reset at idle", potal_paired_h2_early_trace_begin(s), POTAL_OK);
    check_status("early trace duplicate reset refused", potal_paired_h2_early_trace_begin(s), POTAL_ERR_BUSY);
    const potal_h2_panel panel = managed_panel(0, 1);
    const potal_paired_loop_ws main = managed_main(&panel);
    potal_h2_ref ref = {0};
    check_status("early trace declaration", potal_paired_h2_panel_begin(s, &panel, 0, &ref), POTAL_OK);
    check_status("early trace scales", managed_scales(s, &main, 0), POTAL_OK);
    check_status("early trace Main", potal_paired_h2_loop_issue(s, ref, 0, &main), POTAL_OK);
    uint64_t values[POTAL_LDB_ONLY_TRACE_FIELDS] = {0};
    check_status("early trace cannot snapshot pending Main", potal_paired_h2_early_trace_end(s, values), POTAL_ERR_BUSY);
    check_u64("early trace pending snapshot issues no reads", st.early_trace_reads, 0);
    check_status("early trace real checked Fence retained", potal_paired_fence(s), POTAL_OK);
    check_status("early trace frozen snapshot", potal_paired_h2_early_trace_end(s, values), POTAL_OK);
    for (uint32_t i = 0; i < POTAL_LDB_ONLY_TRACE_FIELDS; ++i)
        check_u64("early trace preserves each full-width field", values[i],
            i ? UINT64_C(0x100000000) + i : UINT64_C(1) << 14);
    check_u64("early trace reads exactly declared fields", st.early_trace_reads, 28);
    check_status("early trace consumed snapshot cannot repeat", potal_paired_h2_early_trace_end(s, values), POTAL_ERR_INVALID_ARGUMENT);
    check_status("early trace actual scales released", managed_scales(s, &main, 1), POTAL_OK);
    check_status("early trace actual panel retired", potal_paired_h2_panel_retire(s, ref), POTAL_OK);
    check_status("early trace safe close", potal_paired_h2_close(&s), POTAL_OK);
}

static void test_early_capability(void) {
    check_u64("early exact tag", POTAL_LDB_ONLY_TAG, UINT64_C(0x504f4c4400000000));
    check_u64("early exact capability", POTAL_LDB_ONLY_CAPABILITY, UINT64_C(0x504f4c4401200001));
    check_u64("early exact opcode", POTAL_FUNCT_LDB_ONLY, 29);
    const uint64_t refused[] = {0, POTAL_RESIDENT_WS_CAPABILITY, POTAL_PREFETCH_OBSERVATION_CAPABILITY,
        UINT64_C(0x504f4c4401200000), UINT64_C(0x504f4c4401100001), UINT64_C(0x504f4c4401200003)};
    for (uint32_t i = 0; i < sizeof(refused) / sizeof(refused[0]); ++i) {
        managed_sink_state st;
        potal_paired_session *s = NULL;
        managed_open(&st, &s);
        check_u64("early managed default off", potal_paired_h2_query(s).early_prefetch, 0);
        check_status("early sink requires wide callback",
            potal_paired_h2_early_enable(s, NULL), POTAL_ERR_INVALID_ARGUMENT);
        check_u64("early missing callback issues nothing", st.trace.count, 0);
        st.early_capability = refused[i];
        check_status("early old or mismatched image refused",
            potal_paired_h2_early_enable(s, managed_early_counter), POTAL_ERR_UNSUPPORTED);
        check_u64("early refusal only two safe COUNTER commands", st.trace.count, 2);
        check_u64("early refusal no new opcode", sink_has(&st.trace, POTAL_FUNCT_LDB_ONLY), 0);
        const potal_h2_status q = potal_paired_h2_query(s);
        check_u64("early refusal has no pending work or sticky failure",
            q.early_prefetch | q.pending_b | q.source_pending | q.core.failed | q.attempted_ldb_only, 0);
        check_status("early refusal safe close", potal_paired_h2_close(&s), POTAL_OK);
    }
}

static void test_early_h2(void) {
    potal_h2_ref old = {0};
    for (uint32_t target = 0; target < 2; ++target) {
        managed_sink_state st;
        potal_paired_session *s = NULL;
        potal_h2_panel current, next;
        potal_h2_ref cr = {0}, nr = {0};
        early_prepare(&st, &s, 1 - target, 1, &current, &next, &cr, &nr);
        potal_paired_loop_ws nm = managed_main(&next);
        uint32_t before = st.trace.count;
        check_status("early enabling twice or after declaration refused",
            potal_paired_h2_early_enable(s, managed_early_counter), POTAL_ERR_BUSY);
        potal_h2_ref stale = nr; stale.generation++;
        check_status("early stale prefetch refused", potal_paired_h2_panel_prefetch(s, stale), POTAL_ERR_INVALID_ARGUMENT);
        stale = nr; stale.owner++;
        check_status("early foreign owner prefetch refused", potal_paired_h2_panel_prefetch(s, stale), POTAL_ERR_INVALID_ARGUMENT);
        stale = nr; stale.pair = 1 - target;
        check_status("early wrong pair prefetch refused", potal_paired_h2_panel_prefetch(s, stale), POTAL_ERR_INVALID_ARGUMENT);
        check_status("early current panel cannot be target", potal_paired_h2_panel_prefetch(s, cr), POTAL_ERR_BUSY);
        check_status("early prepared close retains", potal_paired_h2_close(&s), POTAL_ERR_BUSY);
        check_u64("early refusals emit no command", st.trace.count, before);
        st.early_target = target;
        st.early_words = 6;
        st.reenter = 1;
        check_status("early prepared admission", potal_paired_h2_panel_prefetch(s, nr), POTAL_OK);
        check_u64("early all six callbacks owned", st.early_words, 0);
        const potal_paired_cmd_record expected[] = {
            {9, 0, UINT64_C(5) << 16, UINT64_C(0x0000000200020001)},
            {10, 0, 0, UINT64_C(0x20000) + UINT64_C(96) * 128 + 32},
            {11, 0, 0, 0}, {12, 0, 0, 128}, {13, 0, 0, 0},
            {29, 1, UINT64_C(0x10) | target, 0},
        };
        check_u64("early exactly five words and one status run", st.trace.count - before, 6);
        for (uint32_t i = 0; i < 6; ++i) {
            const potal_paired_cmd_record *got = &st.trace.records[before + i];
            check_u64("early exact command and status fields",
                got->funct == expected[i].funct && got->xd == expected[i].xd &&
                got->rs1 == expected[i].rs1 && got->rs2 == expected[i].rs2, 1);
        }
        potal_h2_status q = potal_paired_h2_query(s);
        check_u64("early descriptor is admission, not raw tile MVINs",
            q.attempted_ldb_only == 1 && !q.attempted_b_mvins && !st.mvins, 1);
        check_status("early acknowledgement cannot publish", potal_paired_h2_panel_get(s, nr), POTAL_ERR_BUSY);
        check_u64("early status query cannot complete readers", q.panel[cr.pair].readers_completed, 0);
        check_status("early pending Main refused", potal_paired_h2_loop_issue(s, nr, 0, &nm), POTAL_ERR_BUSY);
        check_status("early pending Scale refused", potal_paired_scale_write(s, 20, 1, 0, 0), POTAL_ERR_BUSY);
        const potal_index_config index = managed_index(&current);
        check_status("early pending Index refused", potal_paired_index_config(s, &index), POTAL_ERR_BUSY);
        check_status("early pending retire refused", potal_paired_h2_panel_retire(s, nr), POTAL_ERR_BUSY);
        check_status("early pending close refused", potal_paired_h2_close(&s), POTAL_ERR_BUSY);
        check_status("early stale Release cannot publish", potal_paired_scale_release(s, 0, 8, 0), POTAL_ERR_INVALID_ARGUMENT);
        check_u64("early rejected Release leaves pending", potal_paired_h2_query(s).pending_b, 1);
        /* Exercise both global publication paths without weakening Release. */
        if (target) check_status("early successful global Fence", potal_paired_fence(s), POTAL_OK);
        else check_status("early successful global Release", potal_paired_scale_release(s, 0, 7, 0), POTAL_OK);
        q = potal_paired_h2_query(s);
        check_u64("early successful certificate publishes only load and issued reader",
            q.panel[target].readable && !q.pending_b && !q.source_pending &&
            q.panel[cr.pair].readers_completed == 1 && !q.panel[target].readers_completed, 1);
        check_u64("early rear resident identity",
            q.core.pair[target].b_sp_base, (2 * target + 2) * POTAL_HW_BANK_ROWS - 128);
        check_u64("early exact resident J and K",
            q.core.pair[target].b_j_tiles == 2 && q.core.pair[target].b_k_rows == 64, 1);
        check_status("early published get", potal_paired_h2_panel_get(s, nr), POTAL_OK);
        check_status("early duplicate prefetch refused", potal_paired_h2_panel_prefetch(s, nr), POTAL_ERR_BUSY);
        for (uint32_t j = 0; j < 2; ++j) for (uint32_t col = 0; col < (j ? 27u : 32u); ++col) {
            if (!target && !j && !col) continue;
            check_status("early current scales released", potal_paired_scale_release(s, j, 7, col), POTAL_OK);
        }
        check_status("early retire current", potal_paired_h2_panel_retire(s, cr), POTAL_OK);
        check_status("early stale retired get", potal_paired_h2_panel_get(s, cr), POTAL_ERR_INVALID_ARGUMENT);
        check_status("early next scales", managed_scales(s, &nm, 0), POTAL_OK);
        potal_paired_loop_ws bad = nm; bad.loop.b = NULL;
        check_status("early resident still needs real B", potal_paired_h2_loop_issue(s, nr, 0, &bad), POTAL_ERR_INVALID_ARGUMENT);
        bad = nm; bad.loop.b = FAKE_B;
        check_status("early resident rejects wrong B", potal_paired_h2_loop_issue(s, nr, 0, &bad), POTAL_ERR_INVALID_ARGUMENT);
        bad = nm; bad.loop.ldb = 96;
        check_status("early resident rejects wrong stride", potal_paired_h2_loop_issue(s, nr, 0, &bad), POTAL_ERR_INVALID_ARGUMENT);
        potal_gemm_ws_program program;
        check_status("early ordinary Main encoding", potal_paired_loop_ws_encode(&nm, &program), POTAL_OK);
        before = st.trace.count;
        check_status("early resident Main", potal_paired_h2_loop_issue(s, nr, 0, &nm), POTAL_OK);
        managed_program_equal(&st.trace, before, &program, 1);
        check_status("early next Release", managed_scales(s, &nm, 1), POTAL_OK);
        check_status("early next retirement", potal_paired_h2_panel_retire(s, nr), POTAL_OK);
        /* Reuse a physical pair, but never a software generation. */
        next.residency_index++;
        potal_h2_ref newer = {0};
        check_status("early next generation declared", potal_paired_h2_panel_begin(s, &next, 0, &newer), POTAL_OK);
        check_u64("early generation increases", newer.generation > nr.generation, 1);
        check_status("early prior generation cannot prefetch", potal_paired_h2_panel_prefetch(s, nr), POTAL_ERR_INVALID_ARGUMENT);
        check_status("early prior generation cannot retire", potal_paired_h2_panel_retire(s, nr), POTAL_ERR_INVALID_ARGUMENT);
        if (old.owner) check_status("early prior session cannot get new B", potal_paired_h2_panel_get(s, old), POTAL_ERR_INVALID_ARGUMENT);
        check_status("early new generation scales", managed_scales(s, &nm, 0), POTAL_OK);
        check_status("early new generation normal Main", potal_paired_h2_loop_issue(s, newer, 0, &nm), POTAL_OK);
        check_status("early new generation Release", managed_scales(s, &nm, 1), POTAL_OK);
        check_status("early new generation retire", potal_paired_h2_panel_retire(s, newer), POTAL_OK);
        old = newer;
        check_status("early safe close consumes", potal_paired_h2_close(&s), POTAL_OK);
        check_u64("early close consumed pointer", s == NULL, 1);
    }
}

static void test_early_guards(void) {
    for (uint32_t kind = 0; kind < 6; ++kind) {
        managed_sink_state st;
        potal_paired_session *s = NULL;
        managed_open(&st, &s);
        st.early_capability = POTAL_LDB_ONLY_CAPABILITY;
        check_status("early guard enable", potal_paired_h2_early_enable(s, managed_early_counter), POTAL_OK);
        potal_h2_panel current = managed_panel(0, kind == 4 ? 2 : 1), next = managed_panel(1, 1);
        potal_h2_ref cr = {0}, nr = {0};
        const potal_paired_loop_ws cm = managed_main(&current), residual = managed_residual(&current);
        check_status("early guard declare current", potal_paired_h2_panel_begin(s, &current, 0, &cr), POTAL_OK);
        check_status("early guard scales", managed_scales(s, &cm, 0), POTAL_OK);
        check_status("early guard Main", potal_paired_h2_loop_issue(s, cr, 0, &cm), POTAL_OK);
        if (kind == 4) {
            check_status("early guard Residual scales", managed_scales(s, &residual, 0), POTAL_OK);
            const potal_index_config index = managed_index(&current);
            check_status("early guard Index", potal_paired_index_config(s, &index), POTAL_OK);
            for (uint32_t i = 0; i < index.active_rows; ++i) {
                const potal_index_write w = {i, i};
                check_status("early guard Index slot", potal_paired_index_write(s, &w), POTAL_OK);
            }
            check_status("early guard Residual issued", potal_paired_h2_loop_issue(s, cr, 1, &residual), POTAL_OK);
        }
        potal_h2_panel bad = next;
        if (kind == 0) bad.k_begin -= 32;
        if (kind == 1) bad.j_begin = 0;
        if (kind == 2) bad.pad_j++;
        if (kind == 5) bad.j_tiles = 1;
        if (kind == 3) check_status("early guard Main already completed", potal_paired_fence(s), POTAL_OK);
        const uint32_t before = st.trace.count;
        check_status("early nonadjacent, different J, completed or Residual refusal",
            potal_paired_h2_panel_begin(s, &bad, 1, &nr),
            kind < 3 || kind == 5 ? POTAL_ERR_INVALID_ARGUMENT : POTAL_ERR_BUSY);
        check_u64("early guard rejection has no effects", st.trace.count, before);
        check_u64("early guard rejection reserves no target", potal_paired_h2_query(s).panel[1].declared, 0);
        check_status("early guard release Main", managed_scales(s, &cm, 1), POTAL_OK);
        if (kind == 4) check_status("early guard release Residual", managed_scales(s, &residual, 1), POTAL_OK);
        check_status("early guard retire", potal_paired_h2_panel_retire(s, cr), POTAL_OK);
        check_status("early guard close", potal_paired_h2_close(&s), POTAL_OK);
    }
}

static void test_early_recovery(void) {
    for (uint32_t position = 1; position <= 6; ++position) {
        managed_sink_state st;
        potal_paired_session *s = NULL;
        potal_h2_panel current, next;
        potal_h2_ref cr = {0}, nr = {0};
        early_prepare(&st, &s, position % 2, 1, &current, &next, &cr, &nr);
        st.early_target = nr.pair;
        st.early_words = position;
        st.fail_at = st.trace.count + position;
        check_status("early injected staging or run failure",
            potal_paired_h2_panel_prefetch(s, nr), POTAL_HW_INVALID_ADDRESS);
        potal_h2_status q = potal_paired_h2_query(s);
        check_u64("early partial load retains pending and source",
            q.pending_b && q.source_pending && q.panel[nr.pair].loading && !q.panel[nr.pair].readable, 1);
        check_u64("early run attempted only at sixth word", q.attempted_ldb_only, position == 6);
        check_u64("early partial staging has no raw MVIN", q.attempted_b_mvins, 0);
        check_status("early failed get sticky", potal_paired_h2_panel_get(s, nr), POTAL_HW_INVALID_ADDRESS);
        check_status("early failed Fence is not cleanup", potal_paired_fence(s), POTAL_HW_INVALID_ADDRESS);
        check_status("early failed close retains ownership", potal_paired_h2_close(&s), POTAL_HW_INVALID_ADDRESS);
        const uint32_t before = st.trace.count;
        check_status("early one recoverable cleanup", potal_paired_session_cleanup(s), POTAL_OK);
        uint32_t fences = 0;
        for (uint32_t i = before; i < st.trace.count; ++i) fences += st.trace.records[i].funct == POTAL_FUNCT_FENCE;
        check_u64("early cleanup begins with one Fence",
            st.trace.records[before].funct == POTAL_FUNCT_FENCE && fences == 1, 1);
        q = potal_paired_h2_query(s);
        check_u64("early cleanup never publishes failed load",
            q.pending_b | q.source_pending | q.panel[0].declared | q.panel[1].declared |
            q.core.outstanding_loops | q.core.live_scale_slots, 0);
        check_status("early cleanup preserves first status", q.core.error, POTAL_HW_INVALID_ADDRESS);
        check_u64("early cleanup retains immutable allocation until close", q.cleaned && q.source_bound, 1);
        const uint32_t after = st.trace.count;
        check_status("early cleanup is not retried", potal_paired_session_cleanup(s), POTAL_OK);
        check_u64("early cleanup repeat has no commands", st.trace.count, after);
        check_status("early cleaned close returns sticky error", potal_paired_h2_close(&s), POTAL_HW_INVALID_ADDRESS);
        check_u64("early cleaned close consumes handle", s == NULL, 1);
    }
}

static void early_fatal_case(uint32_t kind) {
    managed_sink_state st;
    potal_paired_session *s = NULL;
    potal_h2_panel current, next;
    potal_h2_ref cr = {0}, nr = {0};
    early_prepare(&st, &s, 0, 1, &current, &next, &cr, &nr);
    if (kind == 1) st.fail_at = st.trace.count + 3;
    if (kind == 2) {
        st.trace.fail_funct = POTAL_FUNCT_LDB_ONLY;
        st.trace.fail_status = POTAL_ERR_BAD_RESPONSE;
    }
    const potal_status first = kind == 1 ? POTAL_HW_INVALID_ADDRESS : POTAL_ERR_BAD_RESPONSE;
    check_status("early fatal admission", potal_paired_h2_panel_prefetch(s, nr),
        kind == 1 ? POTAL_HW_INVALID_ADDRESS : kind == 2 ? POTAL_ERR_BAD_RESPONSE : POTAL_OK);
    if (kind != 2) {
        st.trace.fail_funct = kind == 3 ? POTAL_FUNCT_SCALE_RELEASE : POTAL_FUNCT_FENCE;
        st.trace.fail_status = POTAL_ERR_BAD_RESPONSE;
        if (kind == 1) check_status("early failed cleanup drain", potal_paired_session_cleanup(s), POTAL_ERR_BAD_RESPONSE);
        else if (kind == 3) check_status("early unknown Release", potal_paired_scale_release(s, 0, 7, 0), POTAL_ERR_BAD_RESPONSE);
        else check_status("early unknown Fence", potal_paired_fence(s), POTAL_ERR_BAD_RESPONSE);
    }
    const potal_h2_status q = potal_paired_h2_query(s);
    check_u64("early unknown completion quarantines borrowed source",
        q.unknown_completion && q.quarantined && q.source_bound && q.source_pending &&
        !q.panel[nr.pair].readable, 1);
    check_status("early quarantine preserves first error", q.core.error, first);
    const uint32_t before = st.trace.count;
    check_status("early quarantine cleanup cannot retry", potal_paired_session_cleanup(s), first);
    check_status("early quarantine Fence cannot retry", potal_paired_fence(s), first);
    check_status("early quarantine cannot replace B", potal_paired_h2_panel_prefetch(s, nr), first);
    check_status("early quarantine cannot retire B", potal_paired_h2_panel_retire(s, nr), first);
    check_status("early quarantine cannot close", potal_paired_h2_close(&s), first);
    check_u64("early quarantine retains session without effects", s != NULL && st.trace.count == before, 1);
    const potal_paired_sink sink = {managed_sink, &st};
    potal_paired_session *other = NULL;
    check_status("early quarantine retains global owner", potal_paired_session_open(&sink, &other), POTAL_ERR_BUSY);
}

static void test_early_quarantine(void) {
    for (uint32_t kind = 0; kind < 4; ++kind) {
#if defined(__unix__)
        fflush(NULL);
        const int before = failures;
        const pid_t child = fork();
        if (child == 0) {
            early_fatal_case(kind);
            fflush(NULL);
            _exit(failures == before ? 0 : 1);
        }
        check_u64("early terminal poison isolated by fork", child >= 0, 1);
        if (child < 0) continue;
        int status = 0;
        pid_t result;
        do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
        check_u64("early isolated quarantine case", result == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, 1);
#else
        check_u64("early quarantine requires fork support, not skipped", 0, 1);
        (void)kind;
        (void)early_fatal_case;
#endif
    }
}
#endif

static void test_managed_admission(void) {
    sink_state st;
    memset(&st, 0, sizeof(st));
    const potal_paired_sink sink = {sink_cmd, &st};
    potal_paired_session *s = NULL;
    check_status("managed admission session", potal_paired_session_open(&sink, &s), POTAL_OK);
    potal_h2_weight w = managed_weight();
#if POTAL_HW_DIM == 32 && !defined(__riscv)
    check_status("managed admission close initial sink", potal_paired_session_close(s), POTAL_OK);
    check_status("managed no-sink session", potal_paired_session_open(NULL, &s), POTAL_OK);
    check_status("managed unavailable hardware refused before binding", potal_paired_h2_begin(s, &w), POTAL_ERR_NO_HARDWARE);
    const potal_h2_status unavailable = potal_paired_h2_query(s);
    check_u64("managed unavailable admission has no effects",
        unavailable.active | unavailable.source_bound | unavailable.core.failed | unavailable.core.commands, 0);
    check_status("managed unavailable admission ordinary close", potal_paired_session_close(s), POTAL_OK);
    check_status("managed admission reopens owner", potal_paired_session_open(&sink, &s), POTAL_OK);
#endif
#if POTAL_HW_DIM == 32
    potal_h2_weight bad = w; bad.bytes--;
    check_status("managed insufficient accessible bytes", potal_paired_h2_begin(s, &bad), POTAL_ERR_INVALID_ARGUMENT);
    bad = w; bad.base = (const int8_t *)((uintptr_t)w.base + 1);
    check_status("managed unaligned original base", potal_paired_h2_begin(s, &bad), POTAL_ERR_INVALID_ARGUMENT);
    bad = w; bad.ldb = (size_t)w.n - 1;
    check_status("managed original stride too short", potal_paired_h2_begin(s, &bad), POTAL_ERR_INVALID_ARGUMENT);
    bad = w; bad.k = UINT64_MAX; bad.ldb = SIZE_MAX;
    check_status("managed original extent overflow", potal_paired_h2_begin(s, &bad), POTAL_ERR_INVALID_ARGUMENT);
    bad = w; bad.base = (const int8_t *)(UINTPTR_MAX - 31);
    check_status("managed original pointer overflow", potal_paired_h2_begin(s, &bad), POTAL_ERR_INVALID_ARGUMENT);
    check_status("managed scale before begin", potal_paired_scale_write(s, 0, 1, 0, 0), POTAL_OK);
    check_status("managed begin cannot adopt live Scale", potal_paired_h2_begin(s, &w), POTAL_ERR_BUSY);
    check_status("managed admission release", potal_paired_scale_release(s, 0, 1, 0), POTAL_OK);
    check_status("managed admission begin", potal_paired_h2_begin(s, &w), POTAL_OK);
    w.base = NULL; w.ldb = 0; /* descriptor copied; caller storage need not survive */
    check_u64("managed no capability counter query", sink_has(&st, POTAL_FUNCT_COUNTER), 0);
    check_status("managed empty close", potal_paired_h2_close(&s), POTAL_OK);
#else
    check_status("managed physical DIM16 explicitly refused", potal_paired_h2_begin(s, &w), POTAL_ERR_UNSUPPORTED);
    check_u64("managed refusal emits no command", st.count, 0);
    check_u64("managed refusal preserves ordinary mode", potal_paired_h2_query(s).active, 0);
    check_status("managed refusal leaves ordinary close", potal_paired_session_close(s), POTAL_OK);
#endif
}

typedef struct observation_test_state {
    sink_state trace;
    potal_paired_session *session;
    uint64_t capability;
    uint32_t calls;
    int preserve_b0, early_supported;
    uint32_t b0_event;
    uint64_t b0_value;
} observation_test_state;

static potal_status observation_test_counter(void *ctx, const potal_paired_cmd_record *record,
    uint64_t *value) {
    observation_test_state *st = ctx;
    check_status("observation reentrant cleanup refused", potal_paired_session_cleanup(st->session), POTAL_ERR_BUSY);
    check_status("observation reentrant close refused", potal_paired_session_close(st->session), POTAL_ERR_BUSY);
    st->calls++;
    check_u64("observation only COUNTER126 xd1 rs2zero",
        record->funct == POTAL_FUNCT_COUNTER && record->xd == 1 && record->rs2 == 0, 1);
    if (st->preserve_b0) {
        if (record->rs1 == POTAL_PREFETCH_OBSERVATION_TAG) *value = st->capability;
        else if (record->rs1 == POTAL_LDB_ONLY_TAG)
            *value = st->early_supported ? POTAL_LDB_ONLY_CAPABILITY : 0;
        else if (record->rs1 & 8) {
            if (((record->rs1 >> 4) & 7) == 0) {
                st->b0_event = (uint32_t)(record->rs1 >> 12);
                st->b0_value = 0; /* CounterFile resets a counter on configuration writes. */
            }
            *value = 0;
        } else if (record->rs1 == 1) {
            st->b0_value = 0;
            *value = 0;
        } else *value = record->rs1 == 0 ? st->b0_value :
            UINT64_C(0x100000000) + (record->rs1 >> 4);
        return POTAL_OK;
    }
    if (st->calls == 1) check_u64("observation clears unsupported counter0 before probe", record->rs1, 0xA);
    if (st->calls == 2) check_u64("observation exact tagged query", record->rs1, POTAL_PREFETCH_OBSERVATION_TAG);
    if (st->calls >= 3 && st->calls <= 9)
        check_u64("observation exact seven event selectors", record->rs1,
            ((uint64_t)(45 + st->calls - 3) << 12) | ((st->calls - 3) << 4) | 8);
    *value = record->rs1 == POTAL_PREFETCH_OBSERVATION_TAG ? st->capability :
        UINT64_C(0x100000000) + (record->rs1 >> 4);
    return POTAL_OK;
}

static void test_observation(void) {
    observation_test_state st = {0};
    const potal_paired_sink sink = {sink_cmd, &st};
    potal_paired_session *s = NULL;
    check_status("observation session", potal_paired_session_open(&sink, &s), POTAL_OK);
    st.session = s;
#if POTAL_HW_DIM == 32
    check_status("observation requires wide-response sink",
        potal_paired_observation_begin(s, NULL), POTAL_ERR_INVALID_ARGUMENT);
    check_status("observation rejects old image",
        potal_paired_observation_begin(s, observation_test_counter), POTAL_ERR_UNSUPPORTED);
    check_u64("observation rejected image only safe probe", st.calls, 2);
    check_status("observation rejected image closes", potal_paired_session_close(s), POTAL_OK);
    memset(&st, 0, sizeof(st));
    st.capability = POTAL_PREFETCH_OBSERVATION_CAPABILITY;
    check_status("observation supported session", potal_paired_session_open(&sink, &s), POTAL_OK);
    st.session = s;
    check_status("observation supported begin",
        potal_paired_observation_begin(s, observation_test_counter), POTAL_OK);
    check_u64("observation probe configure reset", st.calls, 10);
    check_status("observation active close refused", potal_paired_session_close(s), POTAL_ERR_BUSY);
    check_status("observation scale", potal_paired_scale_write(s, 0, 7, 0, 0), POTAL_OK);
    uint64_t cycles[7] = {0};
    check_status("observation cannot snapshot live Scale", potal_paired_observation_end(s, cycles), POTAL_ERR_BUSY);
    check_u64("observation refusal issues nothing", st.calls, 10);
    check_status("observation release", potal_paired_scale_release(s, 0, 7, 0), POTAL_OK);
    check_status("observation drained snapshot", potal_paired_observation_end(s, cycles), POTAL_OK);
    for (uint32_t i = 0; i < 7; ++i)
        check_u64("observation preserves full response width", cycles[i], UINT64_C(0x100000000) + i);
    check_u64("observation snapshot seven reads clear", st.calls, 19);
#else
    check_status("observation DIM16 refusal",
        potal_paired_observation_begin(s, observation_test_counter), POTAL_ERR_UNSUPPORTED);
    check_u64("observation DIM16 no command", st.calls, 0);
#endif
    check_status("observation close", potal_paired_session_close(s), POTAL_OK);
#if POTAL_HW_DIM == 32
    for (int supported = 0; supported < 2; ++supported) {
        memset(&st, 0, sizeof(st));
        st.capability = POTAL_PREFETCH_OBSERVATION_CAPABILITY;
        st.preserve_b0 = 1;
        st.early_supported = supported;
        check_status("observation early session", potal_paired_session_open(&sink, &s), POTAL_OK);
        st.session = s;
        check_status("observation before early probe",
            potal_paired_observation_begin(s, observation_test_counter), POTAL_OK);
        const potal_h2_weight weight = managed_weight();
        check_status("observation managed begin", potal_paired_h2_begin(s, &weight), POTAL_OK);
        st.b0_value = UINT64_C(0x100000011);
        check_status("observation early probe",
            potal_paired_h2_early_enable(s, observation_test_counter),
            supported ? POTAL_OK : POTAL_ERR_UNSUPPORTED);
        check_u64("early probe restores B0 event", st.b0_event, 45);
        if (st.b0_event == 45) st.b0_value += 9;
        uint64_t measured[7] = {0};
        check_status("observation after early probe",
            potal_paired_observation_end(s, measured), POTAL_OK);
        check_u64("early probe retains full-width B0 prefix and later events",
            measured[0], UINT64_C(0x10000001a));
        for (uint32_t i = 1; i < 7; ++i)
            check_u64("early probe leaves other counters unchanged", measured[i], UINT64_C(0x100000000) + i);
        check_status("observation early close", potal_paired_h2_close(&s), POTAL_OK);
    }
#endif
}

#if POTAL_HW_DIM == 32
typedef struct stream_sink_state {
    sink_state trace;
    uint64_t capability, early_capability;
    uint32_t fail_at;
    potal_paired_session *session;
} stream_sink_state;

static potal_status stream_sink(void *ctx, const potal_paired_cmd_record *record) {
    stream_sink_state *st = ctx;
    const potal_status status = sink_cmd(&st->trace, record);
    if (st->fail_at && st->trace.count == st->fail_at) return POTAL_ERR_BAD_RESPONSE;
    return status;
}

static potal_status stream_counter(void *ctx, const potal_paired_cmd_record *record, uint64_t *value) {
    stream_sink_state *st = ctx;
    check_u64("stream probe is COUNTER xd1", record->funct == 126 && record->xd == 1 && !record->rs2, 1);
    check_u64("stream first probe safely resets old counter", st->trace.count || record->rs1 == 10, 1);
    *value = record->rs1 == 10 ? 0 :
        record->rs1 == POTAL_STREAM_EARLY_TAG ? st->early_capability : st->capability;
    return stream_sink(ctx, record);
}

static potal_stream_config stream_config(void) {
    const potal_stream_config c = {
        .weight = {FAKE_B, 160, 91, 128, 20443},
        .main = {FAKE_A, 16384, H2_KTILES * POTAL_HW_DIM},
        .residual = {(const int8_t *)(uintptr_t)0x40000, 16384, 2 * POTAL_HW_DIM},
        .counter = stream_counter,
    };
    return c;
}

static potal_status stream_release_scales(potal_paired_session *s, potal_h2_ref ref,
    const potal_paired_loop_ws *pl) {
    const potal_loop_ws *l = &pl->loop;
    for (uint32_t j = 0; j < l->j_tiles; ++j)
        for (uint32_t col = 0; col < POTAL_HW_DIM - (j + 1 == l->j_tiles ? l->pad_j : 0); ++col) {
            const potal_status status = potal_paired_stream_scale_release(s, ref,
                l->scale_base + j, l->scale_generation, col);
            if (status != POTAL_OK) return status;
        }
    return POTAL_OK;
}

static void test_stream(uint32_t continuous) {
    stream_sink_state st = {0};
    const potal_paired_sink sink = {stream_sink, &st};
    potal_paired_session *s = NULL;
    potal_stream_config config = stream_config();
    config.continuous = continuous;
    if (!continuous) {
        const char * previous = getenv("POTAL_RAW_TRACE");
        char * saved = previous ? strdup(previous) : NULL;
        check_u64("stream trace fixture allocation", previous == NULL || saved != NULL, 1);
        if (previous && !saved) return;
        check_u64("stream trace fixture enables cached raw trace", setenv("POTAL_RAW_TRACE", "1", 1) == 0, 1);
        check_status("stream traced session open", potal_paired_session_open(&sink, &s), POTAL_OK);
        unsetenv("POTAL_RAW_TRACE"); // The session snapshot, not current getenv, controls tracing.
        check_status("stream cached hot trace refused", potal_paired_stream_begin(s, &config), POTAL_ERR_UNSUPPORTED);
        check_u64("stream hot trace refusal has no effects", st.trace.count, 0);
        check_status("stream traced session closes idle", potal_paired_session_close(s), POTAL_OK);
        if (saved) { setenv("POTAL_RAW_TRACE", saved, 1); free(saved); }
        s = NULL;
    }
    check_status("stream open", potal_paired_session_open(&sink, &s), POTAL_OK);
    st.session = s;
    potal_stream_config invalid = config;
    invalid.main.bytes = 0;
    check_status("stream invalid source", potal_paired_stream_begin(s, &invalid), POTAL_ERR_INVALID_ARGUMENT);
    check_u64("stream invalid source no command", st.trace.count, 0);
    check_status("stream old image", potal_paired_stream_begin(s, &config), POTAL_ERR_UNSUPPORTED);
    check_u64("stream old image only two safe counters", st.trace.count == 2 &&
        st.trace.records[0].funct == 126 && st.trace.records[1].funct == 126, 1);
    check_status("stream unsupported close", potal_paired_session_close(s), POTAL_OK);

    memset(&st, 0, sizeof(st));
    st.capability = POTAL_STREAM_CAPABILITY;
    check_status("stream supported open", potal_paired_session_open(&sink, &s), POTAL_OK);
    st.session = s;
    check_status("stream begin", potal_paired_stream_begin(s, &config), POTAL_OK);
    check_u64("stream probe exact tag", st.trace.records[1].rs1, UINT64_C(0x4853384800000000));
    check_u64("stream setup seven commands", st.trace.count, 7);
    check_u64("stream checked mode begin", st.trace.records[2].funct == 30 &&
        st.trace.records[2].xd == 1 && st.trace.records[2].rs1 == (continuous ? 2u : 1u) &&
        !st.trace.records[2].rs2, 1);
    check_u64("stream static EX", st.trace.records[3].rs1, UINT64_C(0x10004));
    check_u64("stream Main load state0", st.trace.records[4].rs1, UINT64_C(0x200101));
    check_u64("stream B load state1", st.trace.records[5].rs1, UINT64_C(0x200109));
    check_u64("stream residual load state2", st.trace.records[6].rs1, UINT64_C(0x200111));
    check_u64("stream real B pitch", st.trace.records[5].rs2, 128);
    check_u64("stream fixed residual pitch", st.trace.records[6].rs2, 64);
    uint64_t observed[POTAL_STREAM_COUNTER_FIELDS] = {0};
    check_status("stream diagnostic counter setup", potal_paired_stream_counters_begin(s), POTAL_OK);
    const uint64_t counter_words[] = {0xa008, 0x13018, 0x14028, 0x29038, 0x1b048, 0x1d058, 0x1f068, 1};
    check_u64("stream diagnostic setup count", st.trace.count, 15);
    for (uint32_t i = 0; i < 8; ++i)
        check_u64("stream diagnostic raw counter fields", st.trace.records[7+i].funct == 126 &&
            st.trace.records[7+i].xd == 1 && st.trace.records[7+i].rs1 == counter_words[i] &&
            !st.trace.records[7+i].rs2, 1);
    check_status("stream diagnostic cannot snapshot before final completion",
        potal_paired_stream_counters_end(s, observed), POTAL_ERR_BUSY);
    check_status("stream no legacy loop escape", potal_paired_loop_ws_issue(s, NULL), POTAL_ERR_BUSY);
    check_status("stream no managed H2 escape", potal_paired_h2_begin(s, &config.weight), POTAL_ERR_BUSY);
    potal_h2_panel p0 = managed_panel(0, 2), p1 = managed_panel(1, 1);
    potal_h2_ref r0, r1;
    check_status("stream declare zero", potal_paired_stream_panel_begin(s, &p0, &r0), POTAL_OK);
    check_status("stream declare one", potal_paired_stream_panel_begin(s, &p1, &r1), POTAL_OK);
    potal_paired_loop_ws m0 = managed_main(&p0), m1 = managed_main(&p1);
    m1.loop.scale_base = 20;
    m1.loop.scale_generation = 9;
    check_status("stream prewrite zero", managed_scales(s, &m0, 0), POTAL_OK);
    check_status("stream prewrite one", managed_scales(s, &m1, 0), POTAL_OK);
    potal_paired_loop_ws conflict = m1;
    conflict.loop.scale_base = 40;
    conflict.loop.scale_generation = m0.loop.scale_generation;
    check_status("stream prewrite distinct generation collision", managed_scales(s, &conflict, 0), POTAL_OK);
    uint32_t before = st.trace.count;
    potal_paired_loop_ws bad = m0;
    bad.loop.a = (const int8_t *)(uintptr_t)0x50000;
    check_status("stream immutable A extent", potal_paired_stream_loop_issue(s, r0, 0, &bad),
        POTAL_ERR_INVALID_ARGUMENT);
    bad = m0; bad.loop.scale_generation = 256;
    check_status("stream no truncated generation", potal_paired_stream_loop_issue(s, r0, 0, &bad),
        POTAL_ERR_INVALID_ARGUMENT);
    potal_h2_ref stale = r0; stale.owner++;
    check_status("stream foreign owner", potal_paired_stream_loop_issue(s, stale, 0, &m0),
        POTAL_ERR_INVALID_ARGUMENT);
    check_u64("stream all refusals side effect free", st.trace.count, before);
    check_status("stream Main zero", potal_paired_stream_loop_issue(s, r0, 0, &m0), POTAL_OK);
    check_u64("stream six words only", st.trace.count - before, 6);
    const uint32_t functs[] = {9, 10, 11, 12, 13, 8};
    for (uint32_t i = 0; i < 6; ++i)
        check_u64("stream exact command sequence", st.trace.records[before + i].funct == functs[i] &&
            st.trace.records[before + i].xd == (i == 5), 1);
    check_u64("stream bounds independent", st.trace.records[before].rs2, UINT64_C(0x300020001));
    check_u64("stream real B retained", st.trace.records[before + 1].rs2, UINT64_C(0x20020));
    check_u64("stream full Main LOOP independent", st.trace.records[before + 5].rs1,
        (UINT64_C(1) << 61) | (UINT64_C(3) << 45) | (UINT64_C(7) << 37) | UINT64_C(0x150002));
    check_u64("stream Main LOOP offset zero", st.trace.records[before + 5].rs2, 0);
    check_u64("stream scheduled not completed", potal_paired_stream_query(s).panel[0].scheduled &&
        !potal_paired_stream_query(s).panel[0].dma_done, 1);
    before = st.trace.count;
    check_status("stream distinct live allocation generation collision",
        potal_paired_stream_loop_issue(s, r1, 0, &conflict), POTAL_ERR_BUSY);
    check_u64("stream generation collision issues nothing", st.trace.count, before);
    check_status("stream Main one", potal_paired_stream_loop_issue(s, r1, 0, &m1), POTAL_OK);
    potal_index_config index = managed_index(&p0);
    check_status("stream Index on scheduled B", potal_paired_index_config(s, &index), POTAL_OK);
    for (uint32_t i = 0; i < index.active_rows; ++i) {
        const potal_index_write w = {i, i};
        check_status("stream Index row", potal_paired_index_write(s, &w), POTAL_OK);
    }
    check_status("stream prefix wait", potal_paired_stream_wait_pair(s, r0), POTAL_OK);
    potal_stream_status q = potal_paired_stream_query(s);
    check_u64("stream local wait leaves other pair outstanding", q.core.outstanding_loops == 1 &&
        q.panel[0].completed == 1 && !q.panel[1].completed && !q.core.drains, 1);
    check_status("stream cannot retire unsealed", potal_paired_stream_panel_retire(s, r0), POTAL_ERR_BUSY);
    check_status("stream unsealed Main release refused", stream_release_scales(s, r0, &m0), POTAL_ERR_BUSY);
    check_u64("stream local Release not global", potal_paired_stream_query(s).core.outstanding_loops, 1);
    potal_paired_loop_ws residual = managed_residual(&p0);
    residual.loop.a = config.residual.base;
    residual.loop.lda = config.residual.stride;
    residual.loop.scale_base = m0.loop.scale_base;
    residual.loop.scale_generation = m0.loop.scale_generation;
    check_status("stream Residual", potal_paired_stream_loop_issue(s, r0, 1, &residual), POTAL_OK);
    check_status("stream protects own Index", potal_paired_index_config(s, &index), POTAL_ERR_BUSY);
    check_status("stream pair zero completed", potal_paired_stream_wait_pair(s, r0), POTAL_OK);
    check_status("stream residual release", stream_release_scales(s, r0, &residual), POTAL_OK);
    check_status("stream zero retire", potal_paired_stream_panel_retire(s, r0), POTAL_OK);
    potal_h2_ref reused;
    check_status("stream pair zero reuse while one busy", potal_paired_stream_panel_begin(s, &p0, &reused), POTAL_OK);
    check_u64("stream software generation advances", reused.generation > r0.generation, 1);
    check_status("stream old reference rejected", potal_paired_stream_wait_pair(s, r0), POTAL_ERR_INVALID_ARGUMENT);
    check_status("stream reused scales", managed_scales(s, &m0, 0), POTAL_OK);
    check_status("stream reused Main", potal_paired_stream_loop_issue(s, reused, 0, &m0), POTAL_OK);
    check_status("stream reused Index", potal_paired_index_config(s, &index), POTAL_OK);
    for (uint32_t i = 0; i < index.active_rows; ++i) {
        const potal_index_write w = {i, i};
        check_status("stream reused Index row", potal_paired_index_write(s, &w), POTAL_OK);
    }
    check_status("stream reuse Main wait", potal_paired_stream_wait_pair(s, reused), POTAL_OK);
    check_status("stream reuse unsealed Main release refused", stream_release_scales(s, reused, &m0), POTAL_ERR_BUSY);
    check_status("stream reused Residual", potal_paired_stream_loop_issue(s, reused, 1, &residual), POTAL_OK);
    potal_acc_readback readback = {.acc_bank = 0, .i_tiles = 1, .j_tiles = 2,
        .pad_i = 3, .pad_j = 5, .c = FAKE_C, .ldc = 64};
    check_status("stream readback requires global drain", potal_paired_acc_readback_issue(s, &readback),
        POTAL_ERR_BUSY);
    check_status("stream conservative global drain", potal_paired_fence(s), POTAL_OK);
    check_status("stream reused residual release", stream_release_scales(s, reused, &residual), POTAL_OK);
    check_status("stream reused retire", potal_paired_stream_panel_retire(s, reused), POTAL_OK);
    check_status("stream one release", stream_release_scales(s, r1, &m1), POTAL_OK);
    check_status("stream one retire", potal_paired_stream_panel_retire(s, r1), POTAL_OK);
    check_status("stream unused collision allocation released globally", managed_scales(s, &conflict, 1), POTAL_OK);
    readback.c = (int32_t *)(uintptr_t)config.main.base;
    check_status("stream output cannot overwrite immutable source",
        potal_paired_acc_readback_issue(s, &readback), POTAL_ERR_INVALID_ARGUMENT);
    readback.c = FAKE_C;
    check_status("stream drained readback", potal_paired_acc_readback_issue(s, &readback), POTAL_OK);
    readback.acc_bank = 1;
    check_status("stream second ACC readback follows first store",
        potal_paired_acc_readback_issue(s, &readback), POTAL_OK);
    check_u64("stream both readbacks remain outstanding",
        potal_paired_stream_query(s).core.outstanding_readbacks, 2);
    check_status("stream end cannot skip final fence", potal_paired_stream_end(s), POTAL_ERR_BUSY);
    check_status("stream legacy Release fixture", potal_paired_scale_write(s, 255, 22, 0, 0), POTAL_OK);
    check_status("stream legacy Release still drains", potal_paired_scale_release(s, 255, 22, 0), POTAL_OK);
    check_u64("stream legacy Release clears readback count",
        potal_paired_stream_query(s).core.outstanding_readbacks, 0);
    check_status("stream Release cannot replace CPU consumption fence", potal_paired_stream_end(s), POTAL_ERR_BUSY);
    check_status("stream final fence", potal_paired_fence(s), POTAL_OK);
    check_status("stream end waits for diagnostic snapshot", potal_paired_stream_end(s), POTAL_ERR_BUSY);
    const uint32_t snapshot_begin = st.trace.count;
    check_status("stream completed diagnostic snapshot", potal_paired_stream_counters_end(s, observed), POTAL_OK);
    check_u64("stream diagnostic records fit trace storage",
        snapshot_begin + POTAL_STREAM_COUNTER_FIELDS <= SINK_CAPACITY, 1);
    for (uint32_t i = 0; i < POTAL_STREAM_COUNTER_FIELDS; ++i) {
        check_u64("stream diagnostic snapshot is full width", observed[i], POTAL_STREAM_CAPABILITY);
        if (snapshot_begin + i < SINK_CAPACITY)
            check_u64("stream diagnostic snapshot selects exact counter",
                st.trace.records[snapshot_begin+i].rs1, (uint64_t)i << 4);
    }
    check_status("stream end", potal_paired_stream_end(s), POTAL_OK);
    check_status("stream close", potal_paired_session_close(s), POTAL_OK);

    memset(&st, 0, sizeof(st));
    st.capability = POTAL_STREAM_CAPABILITY;
    config.residual = (potal_stream_source){0};
    check_status("stream zero residual open", potal_paired_session_open(&sink, &s), POTAL_OK);
    check_status("stream absent residual begin", potal_paired_stream_begin(s, &config), POTAL_OK);
    check_u64("stream absent residual harmless pitch", st.trace.records[6].rs2, 32);
    p0.readers = 1;
    check_status("stream absent residual declare", potal_paired_stream_panel_begin(s, &p0, &r0), POTAL_OK);
    check_status("stream absent residual scales", managed_scales(s, &m0, 0), POTAL_OK);
    check_status("stream absent residual Main", potal_paired_stream_loop_issue(s, r0, 0, &m0), POTAL_OK);
    before = st.trace.count;
    check_status("stream absent residual submit refused", potal_paired_stream_loop_issue(s, r0, 1, &residual),
        POTAL_ERR_INVALID_ARGUMENT);
    check_u64("stream absent residual refusal no effect", st.trace.count, before);
    check_status("stream absent residual wait", potal_paired_stream_wait_pair(s, r0), POTAL_OK);
    check_status("stream absent residual release", stream_release_scales(s, r0, &m0), POTAL_OK);
    check_status("stream absent residual retire", potal_paired_stream_panel_retire(s, r0), POTAL_OK);
    check_status("stream absent residual final fence", potal_paired_fence(s), POTAL_OK);
    check_status("stream absent residual end", potal_paired_stream_end(s), POTAL_OK);
    check_status("stream absent residual close", potal_paired_session_close(s), POTAL_OK);
}

static void test_stream_early(uint32_t with_residual, uint32_t global_completion) {
    stream_sink_state st = {0};
    st.capability = POTAL_STREAM_CAPABILITY;
    potal_paired_session *s = NULL;
    const potal_paired_sink sink = {stream_sink, &st};
    potal_stream_config config = stream_config();
    config.continuous = 1;
    config.early_b_only = 1;
    if (!with_residual) config.residual = (potal_stream_source){0};
    check_status("stream Early probe open", potal_paired_session_open(&sink, &s), POTAL_OK);
    potal_stream_config invalid = config;
    invalid.continuous = 0;
    check_status("stream Early bounded refused", potal_paired_stream_begin(s, &invalid), POTAL_ERR_INVALID_ARGUMENT);
    invalid = config; invalid.early_b_only = 2;
    check_status("stream Early invalid option refused", potal_paired_stream_begin(s, &invalid), POTAL_ERR_INVALID_ARGUMENT);
    check_u64("stream Early invalid options have no effects", st.trace.count, 0);
    check_status("stream Early old composition refused", potal_paired_stream_begin(s, &config), POTAL_ERR_UNSUPPORTED);
    check_u64("stream Early old image only probes", st.trace.count, 3);
    check_u64("stream Early exact probe", st.trace.records[2].rs1, UINT64_C(0x4853384801000000));
    check_u64("stream Early probe cannot change counter options", st.trace.records[2].rs1 & UINT64_C(0xff), 0);
    check_status("stream Early unsupported close", potal_paired_session_close(s), POTAL_OK);

    memset(&st, 0, sizeof(st));
    st.capability = POTAL_STREAM_CAPABILITY;
    st.early_capability = POTAL_STREAM_EARLY_CAPABILITY;
    check_status("stream Early open", potal_paired_session_open(&sink, &s), POTAL_OK);
    check_status("stream Early begin", potal_paired_stream_begin(s, &config), POTAL_OK);
    check_u64("stream Early negotiated mode", st.trace.count == 8 &&
        st.trace.records[3].funct == 30 && st.trace.records[3].rs1 == 6 &&
        st.trace.records[3].xd == 1 && !st.trace.records[3].rs2, 1);
    const potal_h2_panel p0 = managed_panel(0, with_residual ? 2 : 1), p1 = managed_panel(1, 1);
    potal_h2_ref r0, r1;
    check_status("stream Early current declare", potal_paired_stream_panel_begin(s, &p0, &r0), POTAL_OK);
    check_status("stream Early target declare", potal_paired_stream_panel_begin(s, &p1, &r1), POTAL_OK);
    uint32_t before = st.trace.count;
    check_status("stream Early needs current Main", potal_paired_stream_panel_prefetch(s, r1), POTAL_ERR_BUSY);
    check_u64("stream Early no-Main refusal has no effects", st.trace.count, before);
    potal_paired_loop_ws m0 = managed_main(&p0), m1 = managed_main(&p1);
    m1.loop.scale_base = 20; m1.loop.scale_generation = 9;
    check_status("stream Early current Scale", managed_scales(s, &m0, 0), POTAL_OK);
    check_status("stream Early current Main", potal_paired_stream_loop_issue(s, r0, 0, &m0), POTAL_OK);
    before = st.trace.count;
    check_status("stream Early B-only", potal_paired_stream_panel_prefetch(s, r1), POTAL_OK);
    check_u64("stream Early only six staging/admission words", st.trace.count - before, 6);
    const uint32_t functs[] = {9, 10, 11, 12, 13, 29};
    for (uint32_t i = 0; i < 6; ++i)
        check_u64("stream Early raw sequence", st.trace.records[before+i].funct == functs[i] &&
            st.trace.records[before+i].xd == (i == 5), 1);
    check_u64("stream Early B dimensions", st.trace.records[before].rs2, UINT64_C(0x200020001));
    check_u64("stream Early B padding", st.trace.records[before].rs1, UINT64_C(5) << 16);
    check_u64("stream Early has no A source", st.trace.records[before+1].rs1, 0);
    check_u64("stream Early immutable B source", st.trace.records[before+1].rs2, (uintptr_t)m1.loop.b);
    check_u64("stream Early B stride", st.trace.records[before+3].rs2, config.weight.ldb);
    check_u64("stream Early selector", st.trace.records[before+5].rs1, UINT64_C(0x11));
    potal_stream_status q = potal_paired_stream_query(s);
    check_u64("stream Early admission is no reader or completion",
        q.core.outstanding_loops == 1 && !q.panel[1].scheduled && !q.panel[1].submitted &&
        !q.panel[1].completed && !q.panel[1].dma_done, 1);
    before = st.trace.count;
    check_status("stream Early duplicate refused", potal_paired_stream_panel_prefetch(s, r1), POTAL_ERR_BUSY);
    potal_h2_ref stale = r1; stale.generation++;
    check_status("stream Early stale reference", potal_paired_stream_panel_prefetch(s, stale), POTAL_ERR_INVALID_ARGUMENT);
    check_status("stream Early Main waits for producer", potal_paired_stream_loop_issue(s, r1, 0, &m1), POTAL_ERR_BUSY);
    const potal_index_config target_index = managed_index(&p1);
    check_status("stream Early cannot expose Index before Main", potal_paired_index_config(s, &target_index),
        POTAL_ERR_INVALID_ARGUMENT);
    check_u64("stream Early refusals have no effects", st.trace.count, before);
    potal_paired_loop_ws residual = managed_residual(&p0);
    residual.loop.a = config.residual.base;
    residual.loop.lda = config.residual.stride;
    residual.loop.scale_base = m0.loop.scale_base;
    residual.loop.scale_generation = m0.loop.scale_generation;
    if (with_residual) {
        const potal_index_config index = managed_index(&p0);
        check_status("stream Early current Index", potal_paired_index_config(s, &index), POTAL_OK);
        for (uint32_t i = 0; i < index.active_rows; ++i) {
            const potal_index_write w = {i, i};
            check_status("stream Early current Index row", potal_paired_index_write(s, &w), POTAL_OK);
        }
        check_status("stream Early current Residual before B wait",
            potal_paired_stream_loop_issue(s, r0, 1, &residual), POTAL_OK);
    }
    check_status("stream Early producer completion",
        global_completion ? potal_paired_fence(s) : potal_paired_stream_wait_pair(s, r1), POTAL_OK);
    q = potal_paired_stream_query(s);
    check_u64("stream Early completed producer remains no reader",
        q.panel[1].dma_done && !q.panel[1].scheduled && !q.panel[1].submitted && !q.panel[1].completed, 1);
    check_u64("stream Early local/global completion distinction",
        q.core.outstanding_loops == (global_completion ? 0 : p0.readers) &&
        q.panel[0].completed == (global_completion ? p0.readers : 0) &&
        q.core.drains == global_completion, 1);
    check_status("stream Early target Scale", managed_scales(s, &m1, 0), POTAL_OK);
    potal_gemm_ws_program ordinary;
    check_status("stream Early canonical Main encoder", potal_paired_loop_ws_encode(&m1, &ordinary), POTAL_OK);
    before = st.trace.count;
    check_status("stream Early resident Main", potal_paired_stream_loop_issue(s, r1, 0, &m1), POTAL_OK);
    const potal_rocc_cmd words[] = {ordinary.bounds, ordinary.addrs_ab, ordinary.addrs_dc,
        ordinary.strides_ab, ordinary.strides_dc, ordinary.loop_ws};
    check_u64("stream Early Main is still six words", st.trace.count - before, 6);
    for (uint32_t i = 0; i < 6; ++i)
        check_u64("stream Early only suppresses Main B address",
            st.trace.records[before+i].rs1 == words[i].rs1 &&
            st.trace.records[before+i].rs2 == (i == 1 ? 0 : words[i].rs2), 1);
    check_status("stream Early target wait", potal_paired_stream_wait_pair(s, r1), POTAL_OK);
    check_status("stream Early target release", stream_release_scales(s, r1, &m1), POTAL_OK);
    check_status("stream Early target retire", potal_paired_stream_panel_retire(s, r1), POTAL_OK);
    if (with_residual && !global_completion) {
        potal_h2_ref reused;
        check_status("stream Early target reuse", potal_paired_stream_panel_begin(s, &p1, &reused), POTAL_OK);
        before = st.trace.count;
        check_status("stream Early old incarnation refused", potal_paired_stream_panel_prefetch(s, r1),
            POTAL_ERR_INVALID_ARGUMENT);
        check_status("stream Early too-late opportunity refused", potal_paired_stream_panel_prefetch(s, reused),
            POTAL_ERR_BUSY);
        check_u64("stream Early late refusal no effect", st.trace.count, before);
        check_status("stream Early ordinary reuse Scale", managed_scales(s, &m1, 0), POTAL_OK);
        before = st.trace.count;
        check_status("stream Early ordinary reuse Main", potal_paired_stream_loop_issue(s, reused, 0, &m1), POTAL_OK);
        check_u64("stream Early refusal did not suppress B", st.trace.records[before+1].rs2, (uintptr_t)m1.loop.b);
        check_status("stream Early reuse wait", potal_paired_stream_wait_pair(s, reused), POTAL_OK);
        check_status("stream Early reuse release", stream_release_scales(s, reused, &m1), POTAL_OK);
        check_status("stream Early reuse retire", potal_paired_stream_panel_retire(s, reused), POTAL_OK);
    }
    check_status("stream Early current wait", potal_paired_stream_wait_pair(s, r0), POTAL_OK);
    check_status("stream Early current release", stream_release_scales(s, r0, with_residual ? &residual : &m0), POTAL_OK);
    check_status("stream Early current retire", potal_paired_stream_panel_retire(s, r0), POTAL_OK);
    check_status("stream Early final fence", potal_paired_fence(s), POTAL_OK);
    check_status("stream Early end", potal_paired_stream_end(s), POTAL_OK);
    check_status("stream Early close", potal_paired_session_close(s), POTAL_OK);
}

static void test_stream_early_geometry(void) {
    for (uint32_t mismatch = 0; mismatch < 4; ++mismatch) {
        stream_sink_state st = {0};
        st.capability = POTAL_STREAM_CAPABILITY;
        st.early_capability = POTAL_STREAM_EARLY_CAPABILITY;
        const potal_paired_sink sink = {stream_sink, &st};
        potal_paired_session *s = NULL;
        potal_stream_config config = stream_config();
        config.continuous = 1; config.early_b_only = 1;
        check_status("stream Early geometry open", potal_paired_session_open(&sink, &s), POTAL_OK);
        check_status("stream Early geometry begin", potal_paired_stream_begin(s, &config), POTAL_OK);
        const potal_h2_panel p0 = managed_panel(0, 1);
        potal_h2_panel p1 = managed_panel(1, 1);
        switch (mismatch) {
            case 0: p1.k_begin = 0; break;
            case 1: p1.j_begin = 0; break;
            case 2: p1.j_tiles = 1; break;
            case 3: p1.pad_j = 6; break;
        }
        potal_h2_ref r0, r1;
        check_status("stream Early geometry current declare", potal_paired_stream_panel_begin(s, &p0, &r0), POTAL_OK);
        check_status("stream Early geometry target remains valid", potal_paired_stream_panel_begin(s, &p1, &r1), POTAL_OK);
        potal_paired_loop_ws m0 = managed_main(&p0), m1 = managed_main(&p1);
        m1.loop.j_tiles = p1.j_tiles;
        m1.loop.scale_base = 20; m1.loop.scale_generation = 9;
        check_status("stream Early geometry current Scale", managed_scales(s, &m0, 0), POTAL_OK);
        check_status("stream Early geometry current Main", potal_paired_stream_loop_issue(s, r0, 0, &m0), POTAL_OK);
        uint32_t before = st.trace.count;
        check_status("stream Early geometry mismatch refused", potal_paired_stream_panel_prefetch(s, r1),
            POTAL_ERR_INVALID_ARGUMENT);
        check_u64("stream Early geometry refusal has no effects", st.trace.count, before);
        check_status("stream Early geometry ordinary Scale", managed_scales(s, &m1, 0), POTAL_OK);
        before = st.trace.count;
        check_status("stream Early geometry ordinary Main", potal_paired_stream_loop_issue(s, r1, 0, &m1), POTAL_OK);
        check_u64("stream Early geometry refusal retains ordinary B", st.trace.records[before+1].rs2,
            (uintptr_t)m1.loop.b);
        check_status("stream Early geometry target wait", potal_paired_stream_wait_pair(s, r1), POTAL_OK);
        check_status("stream Early geometry target release", stream_release_scales(s, r1, &m1), POTAL_OK);
        check_status("stream Early geometry target retire", potal_paired_stream_panel_retire(s, r1), POTAL_OK);
        check_status("stream Early geometry current wait", potal_paired_stream_wait_pair(s, r0), POTAL_OK);
        check_status("stream Early geometry current release", stream_release_scales(s, r0, &m0), POTAL_OK);
        check_status("stream Early geometry current retire", potal_paired_stream_panel_retire(s, r0), POTAL_OK);
        check_status("stream Early geometry final fence", potal_paired_fence(s), POTAL_OK);
        check_status("stream Early geometry end", potal_paired_stream_end(s), POTAL_OK);
        check_status("stream Early geometry close", potal_paired_session_close(s), POTAL_OK);
    }
}

static void stream_early_failure_case(uint32_t stage) {
    stream_sink_state st = {0};
    st.capability = POTAL_STREAM_CAPABILITY;
    st.early_capability = POTAL_STREAM_EARLY_CAPABILITY;
    const potal_paired_sink sink = {stream_sink, &st};
    potal_paired_session *s = NULL, *other = NULL;
    potal_stream_config config = stream_config();
    config.continuous = 1; config.early_b_only = 1;
    check_status("stream Early failure open", potal_paired_session_open(&sink, &s), POTAL_OK);
    if (stage < 8) {
        st.fail_at = stage + 1;
        check_status("stream Early partial begin", potal_paired_stream_begin(s, &config), POTAL_ERR_BAD_RESPONSE);
    } else {
        check_status("stream Early failure begin", potal_paired_stream_begin(s, &config), POTAL_OK);
        const potal_h2_panel p0 = managed_panel(0, 1), p1 = managed_panel(1, 1);
        potal_paired_loop_ws m0 = managed_main(&p0), m1 = managed_main(&p1);
        m1.loop.scale_base = 20; m1.loop.scale_generation = 9;
        potal_h2_ref r0, r1;
        check_status("stream Early failure current declare", potal_paired_stream_panel_begin(s, &p0, &r0), POTAL_OK);
        check_status("stream Early failure target declare", potal_paired_stream_panel_begin(s, &p1, &r1), POTAL_OK);
        check_status("stream Early failure current Scale", managed_scales(s, &m0, 0), POTAL_OK);
        check_status("stream Early failure current Main", potal_paired_stream_loop_issue(s, r0, 0, &m0), POTAL_OK);
        if (stage < 14) st.fail_at = st.trace.count + stage - 7;
        check_status("stream Early partial producer", potal_paired_stream_panel_prefetch(s, r1),
            stage < 14 ? POTAL_ERR_BAD_RESPONSE : POTAL_OK);
        if (stage == 14) {
            check_status("stream Early producer-only current wait", potal_paired_stream_wait_pair(s, r0), POTAL_OK);
            check_status("stream Early producer-only current release", stream_release_scales(s, r0, &m0), POTAL_OK);
            check_status("stream Early producer-only current retire", potal_paired_stream_panel_retire(s, r0), POTAL_OK);
            const potal_stream_status before = potal_paired_stream_query(s);
            check_u64("stream Early producer-only has zero compute and Scale",
                !before.core.outstanding_loops && !before.core.live_scale_slots && before.panel[1].declared &&
                !before.panel[1].submitted && !before.panel[1].dma_done, 1);
            st.fail_at = st.trace.count + 1;
            check_status("stream Early failed producer wait", potal_paired_stream_wait_pair(s, r1), POTAL_ERR_BAD_RESPONSE);
        } else if (stage >= 15) {
            check_status("stream Early before failed Main wait", potal_paired_stream_wait_pair(s, r1), POTAL_OK);
            check_status("stream Early before failed Main Scale", managed_scales(s, &m1, 0), POTAL_OK);
            st.fail_at = st.trace.count + stage - 14;
            check_status("stream Early partial resident Main",
                potal_paired_stream_loop_issue(s, r1, 0, &m1), POTAL_ERR_BAD_RESPONSE);
        }
    }
    const potal_stream_status q = potal_paired_stream_query(s);
    check_u64("stream Early failure quarantines", q.quarantined && q.core.failed, 1);
    const uint32_t count = st.trace.count;
    check_status("stream Early no failed fence retry", potal_paired_fence(s), POTAL_ERR_BAD_RESPONSE);
    check_status("stream Early no failed cleanup", potal_paired_session_cleanup(s), POTAL_ERR_BAD_RESPONSE);
    check_status("stream Early no failed end", potal_paired_stream_end(s), POTAL_ERR_BAD_RESPONSE);
    check_status("stream Early cannot free quarantine", potal_paired_session_close(s), POTAL_ERR_BUSY);
    check_u64("stream Early failure has no follow-up commands", st.trace.count, count);
    check_status("stream Early failure prevents fallback", potal_paired_session_open(&sink, &other), POTAL_ERR_BUSY);
}

static void stream_failure_case(uint32_t stage) {
    stream_sink_state st = {0};
    st.capability = POTAL_STREAM_CAPABILITY;
    potal_paired_session *s = NULL, *other = NULL;
    const potal_paired_sink sink = {stream_sink, &st};
    const potal_stream_config config = stream_config();
    check_status("stream failure open", potal_paired_session_open(&sink, &s), POTAL_OK);
    st.session = s;
    if (stage < 7) {
        st.fail_at = stage + 1;
        check_status("stream partial begin fails", potal_paired_stream_begin(s, &config), POTAL_ERR_BAD_RESPONSE);
    } else {
        check_status("stream failure begin", potal_paired_stream_begin(s, &config), POTAL_OK);
        const potal_h2_panel panel = managed_panel(0, 1);
        const potal_paired_loop_ws main = managed_main(&panel);
        potal_h2_ref ref;
        check_status("stream failure declare", potal_paired_stream_panel_begin(s, &panel, &ref), POTAL_OK);
        check_status("stream failure scales", managed_scales(s, &main, 0), POTAL_OK);
        if (stage < 13) st.fail_at = st.trace.count + stage - 6;
        check_status("stream partial program",
            potal_paired_stream_loop_issue(s, ref, 0, &main), stage < 13 ? POTAL_ERR_BAD_RESPONSE : POTAL_OK);
        if (stage == 13) {
            st.fail_at = st.trace.count + 1;
            check_status("stream failed wait", potal_paired_stream_wait_pair(s, ref), POTAL_ERR_BAD_RESPONSE);
        } else if (stage == 14) {
            check_status("stream before failed release", potal_paired_stream_wait_pair(s, ref), POTAL_OK);
            st.fail_at = st.trace.count + 1;
            check_status("stream failed local release", potal_paired_stream_scale_release(s, ref, 0, 7, 0),
                POTAL_ERR_BAD_RESPONSE);
        }
    }
    potal_stream_status q = potal_paired_stream_query(s);
    check_u64("stream failure quarantines owner", q.quarantined && q.core.failed, 1);
    const uint32_t count = st.trace.count;
    check_status("stream no failed fence retry", potal_paired_fence(s), POTAL_ERR_BAD_RESPONSE);
    check_status("stream no automatic cleanup", potal_paired_session_cleanup(s), POTAL_ERR_BAD_RESPONSE);
    check_status("stream no failed end", potal_paired_stream_end(s), POTAL_ERR_BAD_RESPONSE);
    check_status("stream cannot free quarantined session", potal_paired_session_close(s), POTAL_ERR_BUSY);
    check_status("stream failure first status", potal_paired_stream_query(s).core.error, POTAL_ERR_BAD_RESPONSE);
    check_u64("stream quarantine no commands", st.trace.count, count);
    check_status("stream quarantine excludes fallback", potal_paired_session_open(&sink, &other), POTAL_ERR_BUSY);
}

static void run_stream_failure_cases(uint32_t count, void (*run_case)(uint32_t)) {
    for (uint32_t stage = 0; stage < count; ++stage) {
#if defined(__unix__)
        fflush(NULL);
        const int before = failures;
        const pid_t child = fork();
        if (child == 0) {
            run_case(stage);
            fflush(NULL);
            _exit(failures == before ? 0 : 1);
        }
        check_u64("stream failure fork", child >= 0, 1);
        if (child < 0) continue;
        int status = 0;
        pid_t result;
        do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
        check_u64("stream isolated failure", result == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, 1);
#else
        check_u64("stream failure isolation requires fork", 0, 1);
        (void)run_case;
#endif
    }
}
#endif

int main(void) {
    printf("POTAL_API_TEST DIM=%u params_sha256=%s\n", (unsigned)potal_hw_dim(), potal_hw_params_sha256());
    check_u64("hw dim matches adapter", potal_hw_dim(), POTAL_HW_DIM);

    test_status_codes();
    test_scale_key();
    test_carrier();
    test_gemm_encode_d16();
    test_gemm_invalid();
    test_loop_ws();
#if !defined(__riscv)
    test_host_has_no_hardware();
#endif
    test_index_encoding();
    test_paired_loop_ws();
    test_acc_readback();
    test_paired_host_no_hardware_before_sessions();
    test_managed_admission();
    test_observation();
#if POTAL_HW_DIM == 32
    test_stream(0);
    test_stream(1);
    test_stream_early(1, 0);
    test_stream_early(1, 1);
    test_stream_early(0, 1);
    test_stream_early_geometry();
    run_stream_failure_cases(15, stream_failure_case);
    run_stream_failure_cases(21, stream_early_failure_case);
    test_managed_completion();
    test_managed_h2();
    test_managed_recovery();
    test_managed_quarantine();
    test_early_trace();
    test_early_capability();
    test_early_h2();
    test_early_guards();
    test_early_recovery();
    test_early_quarantine();
#endif
    /* Legacy terminal poison remains last, after every managed case. */
    test_paired_session();

    if (failures != 0) {
        printf("POTAL_API_TEST FAIL (%d)\n", failures);
        return 1;
    }
    printf("POTAL_API_TEST PASS\n");
    return 0;
}
