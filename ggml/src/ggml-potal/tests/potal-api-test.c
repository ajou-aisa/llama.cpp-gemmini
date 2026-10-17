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
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

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
        .lda = 32, .ldb = 16, .ldc = 16,
        .m = 16, .n = 16, .k = 32,
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

#if POTAL_HW_DIM == 16
static void test_gemm_invalid(void) {
    potal_gemm_ws g;

    check_status("gemm null", potal_gemm_ws_check(NULL), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm();
    check_status("encode null program", potal_gemm_ws_encode(&g, NULL), POTAL_ERR_INVALID_ARGUMENT);

    g = k32_gemm(); g.a = NULL;
    check_status("A null (DRAM addr 0 skips load)", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
    g = k32_gemm(); g.b = (const int8_t *)(uintptr_t)0x20001;
    check_status("B misaligned", potal_gemm_ws_check(&g), POTAL_ERR_INVALID_ARGUMENT);
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

#if POTAL_HW_DIM == 16
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
#if POTAL_HW_DIM == 16
    g = k32_gemm();
    check_status("host issue valid", potal_gemm_ws_issue(&g), POTAL_ERR_NO_HARDWARE);
#endif
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

    if (failures != 0) {
        printf("POTAL_API_TEST FAIL (%d)\n", failures);
        return 1;
    }
    printf("POTAL_API_TEST PASS\n");
    return 0;
}
