#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "potal.h"
#include "potal_hw_params.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if POTAL_HW_DIM != 16
#error "potal-paired-index-hw-probe requires DIM16"
#endif

enum {
    D = POTAL_HW_DIM, ROWS = D, J_TILES = 2, COLS = J_TILES * D,
    MAIN_K_TILES = 4, ORIGINAL_K = MAIN_K_TILES * D,
    MAIN_LDA = ORIGINAL_K,
    RES_LDA = (D + POTAL_HW_MAX_BYTES - 1) / POTAL_HW_MAX_BYTES * POTAL_HW_MAX_BYTES,
    MAX_LDB = ORIGINAL_K, PAIR = 0, A_OFFSET = MAIN_K_TILES,
    MAIN_SCALE_BASE = 0, MAIN_SCALE_SLOTS = MAIN_K_TILES / 2 * J_TILES,
    RES_SCALE_BASE = MAIN_SCALE_SLOTS, RES_SCALE_SLOTS = J_TILES,
    MAIN_GENERATION = 1, RES_GENERATION = 2,
    B_BASE = (2 * PAIR + 2) * POTAL_HW_BANK_ROWS - MAIN_K_TILES * J_TILES * D,
    DMA_ALIGN = POTAL_HW_MAX_BYTES > D * (int) sizeof(int32_t) ? POTAL_HW_MAX_BYTES : D * (int) sizeof(int32_t),
    RECORD_CAPACITY = 256
};

_Static_assert(POTAL_HW_BANK_NUM >= 4, "paired bank geometry");
_Static_assert(POTAL_HW_BANK_ROWS >= MAIN_K_TILES * J_TILES * D, "resident B capacity");
_Static_assert(POTAL_HW_INDEX_ENTRIES >= D, "one complete Index fragment");
_Static_assert(POTAL_SCALE_ENTRIES >= MAIN_SCALE_SLOTS + RES_SCALE_SLOTS, "disjoint scale windows");

/* Static DMA storage also stays mapped if a fatal failure requires parking the process. */
_Alignas(DMA_ALIGN) static int8_t main_a[ROWS * MAIN_LDA];
_Alignas(DMA_ALIGN) static int8_t residual_a[ROWS * RES_LDA];
_Alignas(DMA_ALIGN) static int8_t weight_b[ORIGINAL_K * MAX_LDB];
_Alignas(DMA_ALIGN) static int32_t main_c[ROWS * COLS];
_Alignas(DMA_ALIGN) static int32_t residual_c[ROWS * COLS];
static int32_t expected_main[ROWS * COLS], expected_residual[ROWS * COLS];

enum test_case {
    ZERO_A, ONEHOT_CONSTANT_B, ONEHOT_UNIQUE_B, SIGNED_DENSE_A,
    K_TAG_B, N_TAG_B, FIXED_K0_UNIQUE_B,
    FIXED_SLOT1, SLOT0_REMAP11, ALL_SLOTS_K0, CASE_COUNT
};
static const char * const case_names[] = {
    "ZERO_A", "ONEHOT_CONSTANT_B", "ONEHOT_UNIQUE_B", "SIGNED_DENSE_A",
    "K_TAG_B", "N_TAG_B", "FIXED_K0_UNIQUE_B",
    "FIXED_SLOT1", "SLOT0_REMAP11", "ALL_SLOTS_K0"
};
static const char * const case_args[] = {
    "zero-a", "onehot-constant", "onehot-unique", "signed-dense",
    "k-tag", "n-tag", "fixed-k0-unique",
    "fixed-slot1", "slot0-remap11", "all-slots-k0"
};

static int k_tag_case(enum test_case which) {
    return which == K_TAG_B || which == FIXED_SLOT1 || which == SLOT0_REMAP11 || which == ALL_SLOTS_K0;
}

static unsigned activation_slot(enum test_case which, unsigned row, unsigned bank) {
    if (which == FIXED_K0_UNIQUE_B || (bank == 1 && which == SLOT0_REMAP11)) return 0;
    if (bank == 1 && which == FIXED_SLOT1) return 1;
    return row;
}

static unsigned source_k(enum test_case which, unsigned slot) {
    if (which == SLOT0_REMAP11) return slot == 0 ? 11 : 0;
    return which == ALL_SLOTS_K0 ? 0 : slot;
}

static int32_t sat32(int64_t value) {
    return value > INT32_MAX ? INT32_MAX : value < INT32_MIN ? INT32_MIN : (int32_t) value;
}

static int8_t unique_b(unsigned k, unsigned col) {
    return (int8_t) ((int) ((37u * k + 13u * col + 23u) % 256u) - 128);
}

static void prepare(enum test_case which, size_t ldb) {
    memset(main_a, 0, sizeof(main_a));
    memset(residual_a, 0, sizeof(residual_a));
    memset(weight_b, 0x55, sizeof(weight_b));
    for (unsigned k = 0; k < ORIGINAL_K; ++k)
        for (unsigned col = 0; col < COLS; ++col)
            weight_b[k * ldb + col] = which == ONEHOT_CONSTANT_B ? 1 :
                k_tag_case(which) ? (int8_t) (k + 1) :
                which == N_TAG_B ? (int8_t) (col + 1) : unique_b(k, col);
    for (unsigned row = 0; row < ROWS; ++row) {
        for (unsigned k = 0; k < D; ++k) {
            const int8_t value = which == SIGNED_DENSE_A
                ? (int8_t) ((int) ((29u * row + 17u * k + 9u) % 255u) - 127)
                : (int8_t) (k == activation_slot(which, row, 0));
            main_a[row * MAIN_LDA + k] = value;
            residual_a[row * RES_LDA + k] = which == ZERO_A ? 0 : which == SIGNED_DENSE_A ? value :
                (int8_t) (k == activation_slot(which, row, 1));
        }
        for (unsigned col = 0; col < COLS; ++col) {
            int32_t acc = 0;
            for (unsigned pass = 0; pass < MAIN_K_TILES; ++pass) {
                int64_t partial = 0;
                for (unsigned x = 0; x < D; ++x) {
                    const unsigned k = pass * D + x;
                    partial += (int32_t) main_a[row * MAIN_LDA + k] * weight_b[k * ldb + col];
                }
                const int32_t value = sat32(partial); /* carrier 0 is multiplication by one */
                acc = pass == 0 ? value : sat32((int64_t) acc + value);
            }
            expected_main[row * COLS + col] = acc;
            int64_t partial = 0;
            for (unsigned k = 0; k < D; ++k)
                partial += (int32_t) residual_a[row * RES_LDA + k] * weight_b[source_k(which, k) * ldb + col];
            expected_residual[row * COLS + col] = sat32(partial);
            main_c[row * COLS + col] = residual_c[row * COLS + col] = INT32_MIN;
        }
    }
}

static potal_paired_loop_ws main_loop(size_t ldb) {
    const potal_paired_loop_ws loop = {
        .loop = { .a = main_a, .b = weight_b, .c = NULL,
            .lda = MAIN_LDA, .ldb = ldb, .ldc = 0,
            .i_tiles = 1, .j_tiles = J_TILES, .k_tiles = MAIN_K_TILES,
            .scale_base = MAIN_SCALE_BASE, .scale_generation = MAIN_GENERATION, .passes_per_block = 2 },
        .path = POTAL_PAIRED_MAIN, .pair = PAIR
    };
    return loop;
}

static potal_paired_loop_ws residual_loop(void) {
    const potal_paired_loop_ws loop = {
        .loop = { .a = residual_a, .b = NULL, .c = NULL,
            .lda = RES_LDA, .ldb = 0, .ldc = 0,
            .i_tiles = 1, .j_tiles = J_TILES, .k_tiles = 1,
            .scale_base = RES_SCALE_BASE, .scale_generation = RES_GENERATION, .passes_per_block = 1 },
        .path = POTAL_PAIRED_RESIDUAL, .pair = PAIR, .a_offset_tiles = A_OFFSET,
        .resident_b = 1, .indexed = 1
    };
    return loop;
}

static potal_acc_readback readback(unsigned bank) {
    const potal_acc_readback rb = { .acc_bank = bank, .i_tiles = 1, .j_tiles = J_TILES,
        .c = bank == 0 ? main_c : residual_c, .ldc = COLS };
    return rb;
}

static potal_index_config index_config(void) {
    const potal_index_config cfg = { .pair = PAIR, .b_sp_base = B_BASE,
        .j_tiles = J_TILES, .source_k_rows = ORIGINAL_K, .active_rows = D };
    return cfg;
}

#if defined(__riscv)
static int pin_hart0(void) {
    FILE * f = fopen("/proc/cpuinfo", "r");
    int processor = -1, cpu = -1, valid = f != NULL;
    char line[256];
    if (!f) return 0;
    while (fgets(line, sizeof(line), f)) {
        int v;
        if (sscanf(line, "processor : %d", &v) == 1) processor = v;
        else if (sscanf(line, "hart : %d", &v) == 1 && v == 0) {
            if (processor < 0 || processor >= CPU_SETSIZE || cpu != -1) valid = 0;
            cpu = processor;
        }
    }
    valid = valid && !ferror(f) && cpu >= 0;
    fclose(f);
    if (!valid) return 0;
    cpu_set_t set, got;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0 || sched_getaffinity(0, sizeof(got), &got) != 0 ||
        !CPU_EQUAL(&set, &got) || sched_getcpu() != cpu) return 0;
    printf("[POTAL_INDEX_HW] AFFINITY_PASS linux_cpu=%d physical_hart=0\n", cpu);
    fflush(stdout);
    return 1;
}
#endif

static void describe(enum test_case which, size_t ldb) {
    printf("[POTAL_INDEX_HW] BEGIN case=%s dim=%u pair=%u bank_num=%u bank_rows=%u acc_rows=%u "
           "main_i=1 main_j=%u main_k=%u residual_i=1 residual_j=%u residual_k=1 "
           "source_k_rows=%u active_rows=%u pad_i=0 pad_j=0 pad_k=0 "
           "main_lda=%u residual_lda=%u b_ldb=%zu ldc=%u alignment=%u "
           "b_sp_base=%u b_sp_rows=%u a_offset_tiles=%u main_ppb=2 residual_ppb=1 "
           "main_scale_base=%u main_scale_slots=%u main_generation=%u "
           "residual_scale_base=%u residual_scale_slots=%u residual_generation=%u carrier=0 "
           "main_a=0x%" PRIxPTR " residual_a=0x%" PRIxPTR " b=0x%" PRIxPTR "\n",
           case_names[which], D, PAIR, POTAL_HW_BANK_NUM, POTAL_HW_BANK_ROWS, POTAL_HW_ACC_ROWS,
           J_TILES, MAIN_K_TILES, J_TILES, ORIGINAL_K, D, MAIN_LDA, RES_LDA, ldb, COLS, DMA_ALIGN,
           B_BASE, MAIN_K_TILES * J_TILES * D, A_OFFSET, MAIN_SCALE_BASE, MAIN_SCALE_SLOTS, MAIN_GENERATION,
           RES_SCALE_BASE, RES_SCALE_SLOTS, RES_GENERATION,
           (uintptr_t) main_a, (uintptr_t) residual_a, (uintptr_t) weight_b);
    fflush(stdout);
}

static size_t count_bad(const int32_t * got, const int32_t * expected) {
    size_t bad = 0;
    for (unsigned i = 0; i < ROWS * COLS; ++i) bad += got[i] != expected[i];
    return bad;
}

static void mismatch(enum test_case which, size_t ldb, unsigned row, unsigned col, unsigned bank) {
    const unsigned begin = which == SIGNED_DENSE_A ? 0 : activation_slot(which, row, bank);
    const unsigned end = which == SIGNED_DENSE_A ? D : begin + 1;
    printf("[POTAL_INDEX_HW] MISMATCH case=%s acc_bank=%u row=%u col=%u logical_activation_k=",
        case_names[which], bank, row, col);
    for (unsigned k = begin; k < end; ++k) printf("%s%u", k == begin ? "" : ",", k);
    printf(" index_slot=");
    if (bank == 0) printf("none");
    else for (unsigned k = begin; k < end; ++k) printf("%s%u", k == begin ? "" : ",", k);
    printf(" original_k=");
    for (unsigned k = begin; k < end; ++k) printf("%s%u", k == begin ? "" : ",", bank == 1 ? source_k(which, k) : k);
    printf(" programmed_resident_k=");
    for (unsigned k = begin; k < end; ++k) printf("%s%u", k == begin ? "" : ",", bank == 1 ? source_k(which, k) : k);
    printf(" expected_b=");
    for (unsigned k = begin; k < end; ++k)
        printf("%s%d", k == begin ? "" : ",", weight_b[(bank == 1 ? source_k(which, k) : k) * ldb + col]);
    const unsigned i = row * COLS + col;
    printf(" expected_main=%" PRId32 " expected_residual=%" PRId32 " actual_main=%" PRId32
           " actual_residual=%" PRId32 " main_lda=%u residual_lda=%u b_ldb=%zu b_sp_base=%u "
           "a_offset_tiles=%u main_scale_base=%u main_generation=%u residual_scale_base=%u residual_generation=%u\n",
           expected_main[i], expected_residual[i], main_c[i], residual_c[i], MAIN_LDA, RES_LDA, ldb, B_BASE,
           A_OFFSET, MAIN_SCALE_BASE, MAIN_GENERATION, RES_SCALE_BASE, RES_GENERATION);
    fflush(stdout);
}

static void coordinate_matches(enum test_case which, size_t ldb, const char * path,
                               unsigned row, unsigned col, unsigned bank, int32_t actual) {
    const unsigned slot = activation_slot(which, row, bank);
    const unsigned selected_k = bank == 1 ? source_k(which, slot) : slot;
    printf("[POTAL_INDEX_HW] B_COORDINATES case=%s path=%s row=%u col=%u actual=%" PRId32,
        case_names[which], path, row, col, actual);
    if (which == SIGNED_DENSE_A || (which == ZERO_A && strcmp(path, "RESIDUAL") == 0)) {
        printf(" applicable=0 reason=not_single_B_element\n");
        return;
    }
    printf(" selected_k=%u expected_j=%u expected_lane=%u same_col_k_candidates=", selected_k, col / D, col % D);
    unsigned same_col = 0, same_k = 0, other_j = 0, total = 0;
    for (unsigned k = 0; k < ORIGINAL_K; ++k)
        if (weight_b[k * ldb + col] == actual) printf("%s%u", same_col++ ? "," : "", k);
    if (!same_col) printf("none");
    printf(" same_k_col_candidates=");
    for (unsigned n = 0; n < COLS; ++n)
        if (weight_b[selected_k * ldb + n] == actual) {
            printf("%s%u(j%u,lane%u)", same_k++ ? "," : "", n, n / D, n % D);
            other_j += n / D != col / D;
        }
    if (!same_k) printf("none");
    printf(" all_B_candidates_first8=");
    for (unsigned k = 0; k < ORIGINAL_K; ++k)
        for (unsigned n = 0; n < COLS; ++n)
            if (weight_b[k * ldb + n] == actual) {
                if (total < 8) printf("%s(k%u,n%u,j%u,lane%u)", total ? "," : "", k, n, n / D, n % D);
                ++total;
            }
    if (!total) printf("none");
    /* Repeated int8 values are candidates, not proof of a unique address mapping. */
    printf(" same_col_count=%u same_k_count=%u other_j_same_k_count=%u all_B_count=%u ambiguous=%u\n",
        same_col, same_k, other_j, total, total > 1);
}

static void report_outputs(enum test_case which, size_t ldb, const char * path, unsigned bank,
                           const int32_t * got, const int32_t * expected) {
    unsigned tile_bad[J_TILES] = {0}, printed = 0;
    for (unsigned row = 0; row < ROWS; ++row)
        for (unsigned col = 0; col < COLS; ++col) {
            const unsigned i = row * COLS + col;
            if (got[i] == expected[i]) continue;
            ++tile_bad[col / D];
            if (printed++ >= 4) continue;
            printf("[POTAL_INDEX_HW] FIRST_MISMATCH case=%s path=%s acc_bank=%u row=%u col=%u j_tile=%u "
                "expected=%" PRId32 " actual=%" PRId32 "\n",
                case_names[which], path, bank, row, col, col / D, expected[i], got[i]);
            mismatch(which, ldb, row, col, bank);
            coordinate_matches(which, ldb, path, row, col, bank, got[i]);
        }
    printf("[POTAL_INDEX_HW] MISMATCH_COUNTS case=%s path=%s acc_bank=%u total=%zu",
        case_names[which], path, bank, count_bad(got, expected));
    for (unsigned j = 0; j < J_TILES; ++j) printf(" j%u_bad=%u", j, tile_bad[j]);
    printf("\n");
    for (unsigned row = 0; row < 2; ++row) {
        printf("[POTAL_INDEX_HW] OUTPUT_ROW case=%s path=%s acc_bank=%u row=%u cols=0..%u values=",
            case_names[which], path, bank, row, COLS - 1);
        for (unsigned col = 0; col < COLS; ++col) printf("%s%" PRId32, col ? "," : "", got[row * COLS + col]);
        printf("\n");
    }
    if (k_tag_case(which))
        for (unsigned row = 0; row < 4; ++row)
            for (unsigned col = 0; col < COLS; col += D) {
                const unsigned slot = activation_slot(which, row, bank);
                const int32_t actual = got[row * COLS + col];
                printf("[POTAL_INDEX_HW] K_SLOT_SAMPLE case=%s path=%s acc_bank=%u row=%u col=%u j_tile=%u "
                    "expected=%" PRId32 " actual=%" PRId32 " logical_activation_k=%u index_slot=",
                    case_names[which], path, bank, row, col, col / D, expected[row * COLS + col], actual, slot);
                if (bank == 0) printf("none");
                else printf("%u", slot);
                printf(" programmed_source_k=%u effective_k=", bank == 1 ? source_k(which, slot) : slot);
                if (actual >= 1 && actual <= ORIGINAL_K) printf("%" PRId32 " effective_k_valid=1\n", actual - 1);
                else printf("invalid effective_k_valid=0\n");
            }
    fflush(stdout);
}

static int execute(enum test_case which, size_t ldb, const potal_paired_sink * sink) {
    potal_paired_session * session = NULL;
    const potal_paired_loop_ws main = main_loop(ldb), residual = residual_loop();
    const potal_acc_readback rb_main = readback(0), rb_residual = readback(1);
    const potal_index_config cfg = index_config();
    const int hardware = sink == NULL;
    potal_status status = POTAL_OK;
    const char * stage = "session_open";
    int failed_fence = 0;
    size_t main_bad = 0, residual_bad = 0, pair_bad = 0;
#define CHECK(label, call) do { stage = label; status = (call); if (status != POTAL_OK) goto failed; } while (0)
#define FENCE(label) do { failed_fence = 1; CHECK(label, potal_paired_fence(session)); failed_fence = 0; } while (0)
#define LOG(...) do { if (hardware) { printf(__VA_ARGS__); fflush(stdout); } } while (0)
    if (hardware) describe(which, ldb);
    CHECK("session_open", potal_paired_session_open(sink, &session));
    CHECK("tlb_flush", potal_paired_tlb_flush(session));
    FENCE("tlb_fence");
    LOG("[POTAL_INDEX_HW] TLB_FLUSH_FENCE_PASS case=%s\n", case_names[which]);
    for (unsigned slot = MAIN_SCALE_BASE; slot < MAIN_SCALE_BASE + MAIN_SCALE_SLOTS; ++slot)
        for (unsigned col = 0; col < D; ++col)
            CHECK("main_scale_write", potal_paired_scale_write(session, slot, MAIN_GENERATION, col, 0));
    LOG("[POTAL_INDEX_HW] MAIN_SCALE_PASS base=%u slots=%u generation=%u columns=%u carrier=0\n",
        MAIN_SCALE_BASE, MAIN_SCALE_SLOTS, MAIN_GENERATION, D);
    potal_gemm_ws_program encoded;
    CHECK("main_encode", potal_paired_loop_ws_encode(&main, &encoded));
    LOG("[POTAL_INDEX_HW] LOOP_ENCODING path=MAIN config_ld_a_rs2=%" PRIu64 " config_ld_b_rs2=%" PRIu64
        " bounds_rs1=0x%016" PRIx64 " bounds_rs2=0x%016" PRIx64 " loop_rs1=0x%016" PRIx64
        " loop_rs2=0x%016" PRIx64 "\n", encoded.config_ld_a.rs2, encoded.config_ld_b.rs2,
        encoded.bounds.rs1, encoded.bounds.rs2, encoded.loop_ws.rs1, encoded.loop_ws.rs2);
    CHECK("main_loop_ws", potal_paired_loop_ws_issue(session, &main));
    LOG("[POTAL_INDEX_HW] MAIN_ISSUED case=%s api=potal_paired_loop_ws_issue path=MAIN flags=0 acc_bank=0 "
        "i=1 j=%u k=%u ppb=2 b_ldb=%zu\n", case_names[which], J_TILES, MAIN_K_TILES, ldb);
    FENCE("main_fence");
    LOG("[POTAL_INDEX_HW] MAIN_FENCE_PASS case=%s\n", case_names[which]);
    CHECK("acc0_readback", potal_paired_acc_readback_issue(session, &rb_main));
    FENCE("acc0_readback_fence");
    LOG("[POTAL_INDEX_HW] ACC0_READBACK_PASS case=%s acc_bank=0 full_int32=1 ldc=%u tiles=%u\n",
        case_names[which], COLS, J_TILES);
    if (hardware) {
        main_bad = count_bad(main_c, expected_main);
        LOG("[POTAL_INDEX_HW] MAIN_%s case=%s main_bad=%zu total=%u numerical_checked=1\n",
            main_bad == 0 ? "PASS" : "FAIL", case_names[which], main_bad, ROWS * COLS);
    }
    const potal_paired_session_status resident = potal_paired_session_query(session);
    if (!resident.pair[PAIR].resident_b || resident.pair[PAIR].b_sp_base != B_BASE ||
        resident.pair[PAIR].b_j_tiles != J_TILES || resident.pair[PAIR].b_k_rows != ORIGINAL_K ||
        resident.pair[PAIR].main_a_tiles != A_OFFSET) {
        stage = "resident_b_geometry";
        status = POTAL_ERR_INVALID_ARGUMENT;
        goto failed;
    }
    CHECK("index_config", potal_paired_index_config(session, &cfg));
    LOG("[POTAL_INDEX_HW] INDEX_CONFIG_PASS case=%s pair=%u b_sp_base=%u j_tiles=%u source_k_rows=%u active_rows=%u\n",
        case_names[which], PAIR, cfg.b_sp_base, cfg.j_tiles, cfg.source_k_rows, cfg.active_rows);
    for (unsigned k = 0; k < D; ++k) {
        const potal_index_write entry = { .slot = k, .resident_k = source_k(which, k) };
        CHECK("index_write", potal_paired_index_write(session, &entry));
    }
    if (hardware) {
        printf("[POTAL_INDEX_HW] INDEX_WRITES_PASS case=%s slots=0..%u count=%u order=ascending programmed_source_k=",
            case_names[which], D - 1, D);
        for (unsigned k = 0; k < D; ++k) printf("%s%u", k ? "," : "", source_k(which, k));
        printf("\n");
        fflush(stdout);
    }
    for (unsigned slot = RES_SCALE_BASE; slot < RES_SCALE_BASE + RES_SCALE_SLOTS; ++slot)
        for (unsigned col = 0; col < D; ++col)
            CHECK("residual_scale_write", potal_paired_scale_write(session, slot, RES_GENERATION, col, 0));
    LOG("[POTAL_INDEX_HW] RESIDUAL_SCALE_PASS base=%u slots=%u generation=%u columns=%u carrier=0\n",
        RES_SCALE_BASE, RES_SCALE_SLOTS, RES_GENERATION, D);
    CHECK("residual_encode", potal_paired_loop_ws_encode(&residual, &encoded));
    LOG("[POTAL_INDEX_HW] LOOP_ENCODING path=RESIDUAL config_ld_a_rs2=%" PRIu64 " config_ld_b_rs2=%" PRIu64
        " bounds_rs1=0x%016" PRIx64 " bounds_rs2=0x%016" PRIx64 " loop_rs1=0x%016" PRIx64
        " loop_rs2=0x%016" PRIx64 "\n", encoded.config_ld_a.rs2, encoded.config_ld_b.rs2,
        encoded.bounds.rs1, encoded.bounds.rs2, encoded.loop_ws.rs1, encoded.loop_ws.rs2);
    CHECK("residual_loop_ws", potal_paired_loop_ws_issue(session, &residual));
    LOG("[POTAL_INDEX_HW] RESIDUAL_ISSUED case=%s api=potal_paired_loop_ws_issue path=RESIDUAL indexed=1 "
        "resident_b=1 b_dram=0 flags=0 acc_bank=1 i=1 j=%u k=1 ppb=1 a_offset_tiles=%u\n",
        case_names[which], J_TILES, A_OFFSET);
    FENCE("residual_fence");
    LOG("[POTAL_INDEX_HW] RESIDUAL_FENCE_PASS case=%s\n", case_names[which]);
    CHECK("acc1_readback", potal_paired_acc_readback_issue(session, &rb_residual));
    FENCE("acc1_readback_fence");
    if (hardware) {
        residual_bad = count_bad(residual_c, expected_residual);
        for (unsigned i = 0; i < ROWS * COLS && which != ZERO_A; ++i)
            pair_bad += residual_c[i] != main_c[i];
        LOG("[POTAL_INDEX_HW] ACC1_READBACK_%s case=%s completion_status=POTAL_OK full_int32=1 "
            "numerical_checked=1 residual_bad=%zu main_residual_bad=%zu total=%u pair_comparison_informational=1\n",
            residual_bad == 0 ? "PASS" : "FAIL", case_names[which], residual_bad, pair_bad, ROWS * COLS);
        report_outputs(which, ldb, "MAIN", 0, main_c, expected_main);
        report_outputs(which, ldb, "RESIDUAL", 1, residual_c, expected_residual);
    }
    for (unsigned slot = MAIN_SCALE_BASE; slot < MAIN_SCALE_BASE + MAIN_SCALE_SLOTS; ++slot)
        for (unsigned col = 0; col < D; ++col)
            CHECK("main_scale_release", potal_paired_scale_release(session, slot, MAIN_GENERATION, col));
    for (unsigned slot = RES_SCALE_BASE; slot < RES_SCALE_BASE + RES_SCALE_SLOTS; ++slot)
        for (unsigned col = 0; col < D; ++col)
            CHECK("residual_scale_release", potal_paired_scale_release(session, slot, RES_GENERATION, col));
    LOG("[POTAL_INDEX_HW] SCALE_RELEASE_PASS main_base=%u main_slots=%u residual_base=%u residual_slots=%u\n",
        MAIN_SCALE_BASE, MAIN_SCALE_SLOTS, RES_SCALE_BASE, RES_SCALE_SLOTS);
    const potal_paired_session_status done = potal_paired_session_query(session);
    if (done.failed || done.outstanding_loops || done.outstanding_readbacks || done.live_scale_slots) {
        stage = "session_not_idle";
        status = POTAL_ERR_BUSY;
        goto failed;
    }
    status = potal_paired_session_close(session);
    session = NULL;
    if (status != POTAL_OK) { stage = "session_close"; goto failed; }
    LOG("[POTAL_INDEX_HW] CASE_%s case=%s main_bad=%zu residual_bad=%zu main_residual_bad=%zu total=%u "
        "earliest_difference=%s pair_comparison_informational=1\n", main_bad || residual_bad ? "FAIL" : "PASS", case_names[which],
        main_bad, residual_bad, pair_bad, ROWS * COLS,
        main_bad ? "MAIN_ACC0" : residual_bad ? "INDEXED_RESIDUAL_ACC1" : "none");
    return main_bad || residual_bad ? 1 : 0;

failed:
    printf("[POTAL_INDEX_HW] STATUS_FAIL case=%s stage=%s status=%s failed_fence=%d\n",
        case_names[which], stage, potal_status_name(status), failed_fence);
    fflush(stdout);
    int unresolved = 0;
    if (session) {
        const potal_status cleanup = failed_fence ? status : potal_paired_session_cleanup(session);
        const potal_paired_session_status q = potal_paired_session_query(session);
        unresolved = q.outstanding_loops != 0 || q.outstanding_readbacks != 0 || q.live_scale_slots != 0 || cleanup != POTAL_OK;
        const potal_status close_status = potal_paired_session_close(session);
        printf("[POTAL_INDEX_HW] CASE_FAIL case=%s cleanup=%s close=%s outstanding_loops=%u "
            "outstanding_readbacks=%u live_scale_slots=%u\n", case_names[which],
            potal_status_name(cleanup), potal_status_name(close_status),
            q.outstanding_loops, q.outstanding_readbacks, q.live_scale_slots);
    }
    fflush(stdout);
    if (hardware && unresolved) {
        printf("[POTAL_INDEX_HW] FATAL case=%s dma_buffers_retained=1 process_parked=1 hardware_reset_required=1\n", case_names[which]);
        fflush(stdout);
        for (;;) pause(); /* Never unmap buffers while DMA completion is unknown. */
    }
    return 1;
#undef LOG
#undef FENCE
#undef CHECK
}

struct recording_sink {
    potal_paired_cmd_record records[RECORD_CAPACITY];
    unsigned count;
};

static potal_status record_command(void * context, const potal_paired_cmd_record * record) {
    struct recording_sink * recording = context;
    if (recording->count >= RECORD_CAPACITY) return POTAL_ERR_CAPACITY;
    recording->records[recording->count++] = *record;
    return POTAL_OK;
}

static int encoder_self_test(void) {
    unsigned vectors = 0;
    int good = sat32(INT64_MAX) == INT32_MAX && sat32(INT64_MIN) == INT32_MIN && sat32(-7) == -7 &&
        sat32((int64_t) INT32_MAX + 1) == INT32_MAX && sat32((int64_t) INT32_MIN - 1) == INT32_MIN;
    for (unsigned which = 0; which < CASE_COUNT; ++which) {
        for (size_t ldb = COLS; ldb <= MAX_LDB; ldb += COLS) {
            prepare((enum test_case) which, ldb);
            ++vectors;
            good = good && (uintptr_t) main_a % DMA_ALIGN == 0 && (uintptr_t) residual_a % DMA_ALIGN == 0 &&
                (uintptr_t) weight_b % DMA_ALIGN == 0 && (uintptr_t) main_c % DMA_ALIGN == 0 &&
                (uintptr_t) residual_c % DMA_ALIGN == 0 && ldb % D == 0;
            for (unsigned k = 0; k < ORIGINAL_K; ++k)
                for (unsigned col = 0; col < ldb; ++col) {
                    const int8_t expected = col >= COLS ? 0x55 : which == ONEHOT_CONSTANT_B ? 1 :
                        k_tag_case((enum test_case) which) ? (int8_t) (k + 1) :
                        which == N_TAG_B ? (int8_t) (col + 1) : unique_b(k, col);
                    good = good && weight_b[k * ldb + col] == expected;
                }
            if (k_tag_case((enum test_case) which))
                for (unsigned row = 0; row < ROWS; ++row) {
                    const unsigned slot = which == FIXED_SLOT1 ? 1 : which == SLOT0_REMAP11 ? 0 : row;
                    for (unsigned k = 0; k < MAIN_LDA; ++k)
                        good = good && main_a[row * MAIN_LDA + k] == (int8_t) (k == row);
                    for (unsigned k = 0; k < RES_LDA; ++k)
                        good = good && residual_a[row * RES_LDA + k] == (int8_t) (k == slot);
                }
            for (unsigned row = 0; row < ROWS; ++row)
                for (unsigned col = 0; col < COLS; ++col) {
                    const unsigned i = row * COLS + col;
                    if (which != SIGNED_DENSE_A) {
                        const unsigned k = which == FIXED_K0_UNIQUE_B ? 0 : row;
                        const int32_t expected = k_tag_case((enum test_case) which) ? (int32_t) row + 1 :
                            which == N_TAG_B ? (int32_t) col + 1 : weight_b[k * ldb + col];
                        const int32_t residual_expected = which == FIXED_SLOT1 ? 2 : which == SLOT0_REMAP11 ? 12 :
                            which == ALL_SLOTS_K0 ? 1 : which == ZERO_A ? 0 : expected;
                        good = good && expected_main[i] == expected &&
                            expected_residual[i] == residual_expected;
                    } else good = good && expected_main[i] == expected_residual[i];
                }
            if (which == SIGNED_DENSE_A)
                good = good && expected_main[0] == 31122 && expected_main[31] == -20883 &&
                    expected_main[15 * COLS] == -30978 && expected_main[16 * COLS - 1] == 16362;
            const potal_paired_loop_ws main = main_loop(ldb), residual = residual_loop();
            const potal_index_config cfg = index_config();
            potal_gemm_ws_program m, r;
            potal_rocc_cmd command, store, mvouts[J_TILES];
            uint32_t count = 0;
            good = good && potal_paired_loop_ws_encode(&main, &m) == POTAL_OK &&
                potal_paired_loop_ws_encode(&residual, &r) == POTAL_OK &&
                m.bounds.rs2 == (1ull | ((uint64_t) J_TILES << 16) | ((uint64_t) MAIN_K_TILES << 32)) &&
                r.bounds.rs2 == (1ull | ((uint64_t) J_TILES << 16) | (1ull << 32)) &&
                m.strides_ab.rs1 == MAIN_LDA && m.strides_ab.rs2 == ldb &&
                r.strides_ab.rs1 == RES_LDA && r.addrs_ab.rs2 == 0 &&
                (m.loop_ws.rs1 >> 61) == 1 && (r.loop_ws.rs1 >> 61) == 7 &&
                ((m.loop_ws.rs1 >> 45) & 0xffffu) == 2 && ((r.loop_ws.rs1 >> 45) & 0xffffu) == 1 &&
                (m.loop_ws.rs1 & 1) == 0 && (r.loop_ws.rs1 & 1) == 0 &&
                m.loop_ws.rs2 == 0 && r.loop_ws.rs2 == (uint64_t) A_OFFSET << 3 &&
                potal_index_config_encode(&cfg, &command) == POTAL_OK &&
                command.rs1 == ((uint64_t) J_TILES << 32 | B_BASE) &&
                command.rs2 == ((uint64_t) D << 32 | ORIGINAL_K);
            for (unsigned k = 0; k < D; ++k) {
                const unsigned expected_k = which == SLOT0_REMAP11 ? (k == 0 ? 11 : 0) : which == ALL_SLOTS_K0 ? 0 : k;
                const potal_index_write entry = { k, source_k((enum test_case) which, k) };
                good = good && potal_index_write_encode(&cfg, &entry, &command) == POTAL_OK &&
                    command.rs1 == k && command.rs2 == expected_k;
            }
            const potal_index_write outside = { D, 0 };
            good = good && potal_index_write_check(&cfg, &outside) == POTAL_ERR_INVALID_ARGUMENT;
            for (unsigned bank = 0; bank < 2; ++bank) {
                const potal_acc_readback rb = readback(bank);
                good = good && potal_acc_readback_encode(&rb, &store, mvouts, J_TILES, &count) == POTAL_OK &&
                    count == J_TILES && store.rs2 == COLS * sizeof(int32_t);
                for (unsigned j = 0; j < J_TILES; ++j)
                    good = good && mvouts[j].rs1 == (uintptr_t) (rb.c + j * D) &&
                        (mvouts[j].rs2 >> 48) == D && ((mvouts[j].rs2 >> 32) & 0xffffu) == D &&
                        (uint32_t) mvouts[j].rs2 == ((1u << 31) | (1u << 29) | (bank * (POTAL_HW_ACC_ROWS / 2) + j * D));
            }
            struct recording_sink recording = {0};
            const potal_paired_sink sink = { record_command, &recording };
            good = good && execute((enum test_case) which, ldb, &sink) == 0 && recording.count == 241;
            unsigned counts[28] = {0}, loops = 0, index_writes = 0;
            for (unsigned i = 0; i < recording.count; ++i) {
                const potal_paired_cmd_record * rec = &recording.records[i];
                if (rec->funct < 28) ++counts[rec->funct];
                if (rec->funct == POTAL_FUNCT_LOOP_WS) {
                    const int is_main = loops++ == 0;
                    good = good && i > 0 && i + 1 < recording.count && recording.records[i + 1].funct == POTAL_FUNCT_FENCE &&
                        rec->rs1 == (is_main ? m.loop_ws.rs1 : r.loop_ws.rs1) &&
                        rec->rs2 == (is_main ? m.loop_ws.rs2 : r.loop_ws.rs2);
                }
                if (rec->funct == POTAL_FUNCT_INDEX_WRITE) {
                    const unsigned expected_k = which == SLOT0_REMAP11 ? (index_writes == 0 ? 11 : 0) :
                        which == ALL_SLOTS_K0 ? 0 : index_writes;
                    good = good && rec->rs1 == index_writes && rec->rs2 == expected_k;
                    ++index_writes;
                }
                if (rec->funct == POTAL_FUNCT_SCALE_WRITE) good = good && rec->rs2 == 0;
            }
            good = good && loops == 2 && index_writes == D && counts[POTAL_FUNCT_FENCE] == 5 &&
                counts[POTAL_FUNCT_MVOUT] == 4 && counts[POTAL_FUNCT_SCALE_WRITE] == 96 &&
                counts[POTAL_FUNCT_SCALE_RELEASE] == 96 && counts[POTAL_FUNCT_FLUSH] == 1 &&
                counts[POTAL_FUNCT_INDEX_CONFIG] == 1;
        }
    }
    printf("[POTAL_INDEX_HW] ENCODER_SELF_TEST_%s vectors=%u numerical_hardware_validation=NOT_RUN\n",
        good ? "PASS" : "FAIL", vectors);
    fflush(stdout);
    return good ? 0 : 1;
}

static int usage(const char * name) {
    fprintf(stderr, "Usage: %s --case zero-a|onehot-constant|onehot-unique|signed-dense|k-tag|n-tag|fixed-k0-unique|"
        "fixed-slot1|slot0-remap11|all-slots-k0 "
        "[--b-stride 32|64]\n       %s --self-test\n"
        "Exit codes: 0=PASS, 1=FAIL, 2=usage/platform/affinity error.\n", name, name);
    return 2;
}

int main(int argc, char ** argv) {
    int which = -1, self_test = 0, stride_given = 0;
    size_t ldb = MAX_LDB;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--self-test") == 0 && !self_test) self_test = 1;
        else if (strcmp(argv[i], "--case") == 0 && which == -1 && i + 1 < argc) {
            ++i;
            for (unsigned c = 0; c < CASE_COUNT; ++c) if (strcmp(argv[i], case_args[c]) == 0) which = (int) c;
            if (which == -1) return usage(argv[0]);
        } else if (strcmp(argv[i], "--b-stride") == 0 && !stride_given && i + 1 < argc) {
            ++i;
            if (strcmp(argv[i], "32") == 0) ldb = COLS;
            else if (strcmp(argv[i], "64") == 0) ldb = MAX_LDB;
            else return usage(argv[0]);
            stride_given = 1;
        } else return usage(argv[0]);
    }
    if (self_test) return which != -1 || stride_given ? usage(argv[0]) : encoder_self_test();
    if (which == -1) return usage(argv[0]);
#if !defined(__riscv)
    fprintf(stderr, "[POTAL_INDEX_HW] CASE_FAIL case=%s stage=platform status=NO_HARDWARE b_ldb=%zu; use --self-test on host\n", case_names[which], ldb);
    return 2;
#else
    if (!pin_hart0()) {
        fprintf(stderr, "[POTAL_INDEX_HW] CASE_FAIL case=%s stage=affinity physical_hart=0 "
            "cannot_identify_or_pin_cpu=1 errno=%d error=%s\n", case_names[which], errno, strerror(errno));
        fflush(stderr);
        return 2;
    }
    prepare((enum test_case) which, ldb);
    return execute((enum test_case) which, ldb, NULL);
#endif
}
