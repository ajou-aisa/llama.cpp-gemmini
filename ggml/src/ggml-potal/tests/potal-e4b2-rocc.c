/*
 * Full generated-LOOP RTL execution fixture. Build with the existing bare-metal
 * runtime for Chipyard/FireSim simulation, or as a Linux executable on hart0.
 * The only arithmetic result under test comes from addressed ACC MVOUT.
 * The original-tensor oracle is never used to manufacture an accelerator result.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "support/e4b2-fixtures.h"

#if !defined(__riscv)
#error "This fixture requires a RISC-V PoTal execution target"
#endif

#define E4B2_STR_(x) #x
#define E4B2_STR(x) E4B2_STR_(x)
#define ISSUE0(f, a, b) \
    __asm__ volatile(".insn r CUSTOM_3, 0x3, " E4B2_STR(f) ", x0, %0, %1" \
                     : : "r"((uint64_t)(a)), "r"((uint64_t)(b)) : "memory")
#define ISSUE1(f, result, a, b) \
    __asm__ volatile(".insn r CUSTOM_3, 0x7, " E4B2_STR(f) ", %0, %1, %2" \
                     : "=r"(result) : "r"((uint64_t)(a)), "r"((uint64_t)(b)) : "memory")

enum { D = 16, KT = 28, JT = 9, K = KT * D, N = JT * D };
static int8_t weight[K][N] __attribute__((aligned(64)));
static int8_t activation[D][K] __attribute__((aligned(64)));
static volatile int32_t actual[32][D][D] __attribute__((aligned(64)));
static int32_t expected[2][512][D];
static uint16_t known[2][512];
static uint32_t sources[KT][D], lengths[KT];
static unsigned selected_pair, resident_k, resident_j, weight_mode, generation = 1;
static uint64_t counts[128], checks, runs, view_configurations, b_mvins;

static void ordered(void) { __asm__ volatile("fence rw,rw" : : : "memory"); }
static void fail(const char *stage, uint64_t status) {
    printf("E4B2_ROCC_FAIL stage=%s status=%llu\n", stage, (unsigned long long)status);
    exit(1);
}
static void ok(const char *stage, uint64_t status) {
    if (status) fail(stage, status);
}
static void fence_potal(void) {
    uint64_t response;
    ++counts[25];
    ISSUE1(25, response, 0, 0);
    ok("fence", response);
    ordered();
}
static void config(uint64_t a, uint64_t b) {
    ++counts[0];
    ISSUE0(0, a, b);
}
static unsigned ceiling(unsigned n) { return (n + D - 1) / D; }
static uint32_t b_base(void) {
    return (selected_pair + 1) * 8192 - ceiling(resident_k) * ceiling(resident_j) * D;
}
static uint64_t shape(unsigned rows, unsigned columns, uint32_t address) {
    return (uint64_t)rows << 48 | (uint64_t)columns << 32 | address;
}
static void load_weight(unsigned pair, unsigned mode, unsigned rows, unsigned columns) {
    selected_pair = pair;
    resident_k = rows;
    resident_j = columns;
    weight_mode = mode;
    for (unsigned k = 0; k < K; ++k)
        for (unsigned j = 0; j < N; ++j)
            weight[k][j] = e4b2_weight(pair, mode, k, j);
    ordered();
    config((uint64_t)D << 16 | UINT64_C(0x101), N);
    for (unsigned k = 0; k < ceiling(rows); ++k)
        for (unsigned j = 0; j < ceiling(columns); ++j) {
            const unsigned nr = rows - k * D < D ? rows - k * D : D;
            const unsigned nc = columns - j * D < D ? columns - j * D : D;
            const uint32_t address = b_base() + (k * ceiling(columns) + j) * D;
            ++counts[2];
            ++b_mvins;
            ISSUE0(2, &weight[k * D][j * D], shape(nr, nc, address));
        }
    fence_potal();
    memset(weight, 93, sizeof(weight));
    ordered();
    printf("E4B2_B_RESIDENT pair=%u base=%u pitch=%u K=%u J=%u source_poisoned=1\n",
           pair, b_base(), ceiling(columns) * D, rows, columns);
}

static void read_known(void) {
    for (unsigned bank = 0; bank < 2; ++bank) {
        /* A completed DMA must replace poison, not reuse another bank's data. */
        memset((void *)actual, 0xa5, sizeof(actual));
        ordered();
        config(2, D * sizeof(int32_t));
        for (unsigned tile = 0; tile < 32; ++tile) {
            unsigned live = 0;
            for (unsigned i = 0; i < D; ++i) live |= known[bank][tile * D + i];
            if (!live) continue;
            ++counts[3];
            ISSUE0(3, &actual[tile][0][0],
                   shape(D, D, UINT32_C(0xa0000000) | (bank * 512 + tile * D)));
        }
        fence_potal();
        for (unsigned row = 0; row < 512; ++row)
            for (unsigned c = 0; c < D; ++c) if (known[bank][row] & (1u << c)) {
                ++checks;
                const int32_t got = actual[row / D][row % D][c];
                if (got != expected[bank][row][c]) {
                    printf("E4B2_INTEGER_FAIL bank=%u row=%u col=%u expected=%ld actual=%ld\n",
                           bank, row, c, (long)expected[bank][row][c], (long)got);
                    exit(1);
                }
            }
    }
}

static void loop_configuration(unsigned k_tiles, unsigned j_tiles, unsigned k_pad,
                               unsigned j_pad, unsigned i_pad) {
    const uint64_t pads = (uint64_t)k_pad << 32 | (uint64_t)j_pad << 16 | i_pad;
    const uint64_t bounds = (uint64_t)k_tiles << 32 | (uint64_t)j_tiles << 16 | 1;
    config((uint64_t)D << 16 | UINT64_C(0x101), K);
    config((uint64_t)D << 16 | UINT64_C(0x109), N);
    config(UINT64_C(1) << 16 | 4, UINT64_C(1) << 48);
    config(2, N * sizeof(int32_t));
    ++counts[9];  ISSUE0(9, pads, bounds);
    ++counts[10]; ISSUE0(10, activation, 0);
    ++counts[11]; ISSUE0(11, 0, 0);
    ++counts[12]; ISSUE0(12, K, 0);
    ++counts[13]; ISSUE0(13, 0, 0);
}

static uint64_t run_flags(unsigned path, unsigned indexed, unsigned accumulate) {
    return (uint64_t)accumulate | UINT64_C(1) << 1 | UINT64_C(1) << 3 |
        (uint64_t)(selected_pair + 1) << 16 | (uint64_t)(selected_pair + 1) << 18 |
        UINT64_C(1) << 20 | (uint64_t)generation << 37 | UINT64_C(1) << 45 |
        UINT64_C(1) << 61 | (uint64_t)path << 62 | (uint64_t)indexed << 63;
}

static void scales(unsigned partials, unsigned columns, unsigned j_origin,
                   unsigned arithmetic_mode, int release) {
    for (unsigned p = 0; p < partials; ++p)
        for (unsigned jt = 0; jt < ceiling(columns); ++jt)
            for (unsigned c = 0; c < D && jt * D + c < columns; ++c) {
                const unsigned slot = p * ceiling(columns) + jt;
                const uint64_t key = (uint64_t)c << 16 | (uint64_t)generation << 8 | slot;
                uint64_t response;
                if (release) {
                    ++counts[24];
                    ISSUE1(24, response, key, 0);
                } else {
                    ++counts[23];
                    const uint32_t carrier = e4b2_carrier(arithmetic_mode,
                        sources[p][0] / 32, j_origin + jt * D + c);
                    ISSUE1(23, response, key, carrier);
                }
                ok(release ? "ScaleRelease" : "ScaleWrite", response);
            }
}

static void index_program(unsigned partials) {
    const unsigned active = (partials - 1) * D + lengths[partials - 1];
    uint64_t response;
    ++counts[27];
    ISSUE1(27, response, (uint64_t)ceiling(resident_j) << 32 | b_base(),
           (uint64_t)active << 32 | resident_k);
    ok("IndexConfig", response);
    for (unsigned p = 0; p < partials; ++p)
        for (unsigned x = 0; x < D && p * D + x < active; ++x) {
            const uint32_t source = x < lengths[p] ? sources[p][x] : sources[p][0];
            ++counts[26];
            ISSUE1(26, response, p * D + x, source);
            ok("IndexWrite", response);
        }
}

static void execute(const char *name, unsigned path, unsigned indexed, unsigned partials,
                    unsigned j_origin, unsigned columns, unsigned acc_base,
                    unsigned rows, unsigned arithmetic_mode, unsigned accumulate) {
    const unsigned active = (partials - 1) * D + lengths[partials - 1];
    memset(activation, 0, sizeof(activation));
    for (unsigned i = 0; i < rows; ++i)
        for (unsigned p = 0; p < partials; ++p)
            for (unsigned x = 0; x < lengths[p]; ++x)
                activation[i][p * D + x] = e4b2_activation(arithmetic_mode, i, sources[p][x]);
    scales(partials, columns, j_origin, arithmetic_mode, 0);
    if (indexed) index_program(partials);
    loop_configuration(partials, ceiling(columns), partials * D - active,
                       ceiling(columns) * D - columns, D - rows);
    const uint32_t base = b_base() + (indexed ? 0 :
        sources[0][0] / D * ceiling(resident_j) * D + j_origin);
    const uint64_t view1 = (uint64_t)acc_base << 48 | (uint64_t)(indexed ? j_origin / D : 0) << 32 |
        (uint64_t)(ceiling(resident_j) * D) << 16 | base;
    const uint64_t view2 = UINT64_C(0x11) | (uint64_t)selected_pair << 1 |
        (uint64_t)indexed << 2 | (uint64_t)path << 3;
    uint64_t response;
    ++counts[28];
    ++view_configurations;
    ISSUE1(28, response, view1, view2);
    ok("ResidentView", response);
    ordered();
    ++counts[8];
    ISSUE1(8, response, run_flags(path, indexed, accumulate), path ? (uint64_t)KT << 3 : 0);
    ok("ExtendedLoop", response);
    ++runs;
    fence_potal();

    for (unsigned i = 0; i < rows; ++i)
        for (unsigned j = 0; j < columns; ++j) {
            const unsigned row = acc_base + (j / D) * D + i;
            int32_t value = accumulate ? expected[path][row][j % D] : 0;
            if (accumulate && !(known[path][row] & (1u << (j % D))))
                fail("oracle_missing_prior_contribution", 1);
            for (unsigned p = 0; p < partials; ++p) {
                const int32_t partial = e4b2_oracle_partial(selected_pair, weight_mode,
                    arithmetic_mode, i, j_origin + j, sources[p], lengths[p]);
                value = e4b2_clamp((int64_t)value + partial);
            }
            expected[path][row][j % D] = value;
            known[path][row] |= (uint16_t)(1u << (j % D));
        }
    read_known();
    scales(partials, columns, j_origin, arithmetic_mode, 1);
    ++generation;
    printf("E4B2_ROCC_CASE PASS name=%s pair=%u path=%u indexed=%u Kfirst=%u "
           "Jfirst=%u Bbase=%u pitch=%u ACC=%u activeK=%u columns=%u rows=%u "
           "B_dram=0 partials=%u\n", name, selected_pair, path, indexed, sources[0][0],
           j_origin, base, ceiling(resident_j) * D, path * 512 + acc_base,
           active, columns, rows, partials);
}

static unsigned canonical(unsigned first, unsigned count) {
    const unsigned partials = ceiling(count);
    for (unsigned p = 0; p < partials; ++p) {
        lengths[p] = count - p * D < D ? count - p * D : D;
        for (unsigned x = 0; x < lengths[p]; ++x) sources[p][x] = first + p * D + x;
    }
    return partials;
}

#if !defined(E4B2_RTL_SATURATION)
static void rejected_runs(void) {
    uint64_t response;
    loop_configuration(1, 1, 0, 0, 0);
    ++counts[8];
    ISSUE1(8, response, run_flags(0, 0, 0), 0);
    if (response == 0 || response > 7) fail("missing_view_must_refuse", response);
    ++counts[28];
    ISSUE1(28, response, UINT64_C(144) << 16 | b_base(), UINT64_C(0x111));
    if (response == 0 || response > 7) fail("reserved_view_must_refuse", response);
    ++counts[28];
    ISSUE1(28, response, 0, 0);
    ok("disable_view", response);
    loop_configuration(1, 1, 0, 0, 0);
    ++counts[8];
    ISSUE1(8, response, run_flags(0, 0, 0), 0);
    if (response == 0 || response > 7) fail("disabled_view_must_refuse", response);
    fence_potal();
    printf("E4B2_ROCC_REFUSAL PASS missing=1 reserved=1 disabled=1\n");
}

static void addressing(unsigned pair) {
    load_weight(pair, 0, K, N);
    rejected_runs();
    execute("first", 0, 0, canonical(0, 16), 0, 16, 16, 16, 0, 0);
    execute("interior_K3_J5_row4672", 0, 0, canonical(48, 16), 80, 16, 80, 16, 0, 0);
    execute("last", 0, 0, canonical(432, 16), 128, 16, 128, 16, 0, 0);
    execute("odd_phase_first_J", 0, 0, canonical(112, 32), 0, 16, 160, 16, 0, 0);
    execute("odd_phase_interior_J", 0, 0, canonical(112, 48), 48, 32, 192, 16, 0, 0);
    execute("last_J", 0, 0, canonical(160, 16), 128, 16, 240, 16, 0, 0);
    execute("full_K28_J9", 0, 0, canonical(0, K), 0, N, 16, 16, 0, 0);
    execute("indexed_Main_ACC0", 0, 1, canonical(112, 32), 48, 32, 320, 16, 0, 0);
    lengths[0] = lengths[1] = 3;
    sources[0][0] = 96; sources[0][1] = 98; sources[0][2] = 126;
    sources[1][0] = 128; sources[1][1] = 135; sources[1][2] = 143;
    execute("indexed_Residual_ACC1", 1, 1, 2, 48, 32, 80, 16, 0, 0);
    execute("ordered_first", 0, 0, canonical(0, 16), 80, 16, 80, 16, 0, 0);
    execute("ordered_second", 0, 0, canonical(16, 16), 80, 16, 80, 16, 0, 1);
    execute("ordered_third", 0, 0, canonical(32, 16), 80, 16, 80, 16, 0, 1);
    execute("seed_native_tail_output", 0, 0, canonical(432, 16), 128, 16, 400, 16, 0, 0);
    execute("seed_indexed_tail_output", 0, 1, canonical(432, 16), 128, 16, 432, 16, 0, 0);
    execute("seed_combined_tail_output", 0, 0, canonical(416, 32), 112, 32, 464, 16, 0, 0);
    load_weight(pair, 0, 443, 139);
    execute("K443_J139_native_tail", 0, 0, canonical(432, 11), 128, 11, 400, 7, 0, 0);
    execute("K443_J139_indexed_tail", 0, 1, canonical(432, 11), 128, 11, 432, 7, 0, 0);
    execute("combined_tail", 0, 0, canonical(416, 27), 112, 27, 464, 7, 0, 0);
}
#endif

static void witnesses(void) {
    load_weight(0, 1, K, N);
    execute("MAX_MAX_MIN_first", 0, 0, canonical(0, 16), 0, 16, 0, 16, 1, 0);
    execute("MAX_MAX_MIN_second", 0, 0, canonical(16, 16), 0, 16, 0, 16, 1, 1);
    execute("MAX_MAX_MIN_third", 0, 0, canonical(32, 16), 0, 16, 0, 16, 1, 1);
    if (expected[0][0][0] != -1) fail("ordered_witness_constant", 1);
    load_weight(1, 2, K, N);
    lengths[0] = 2; sources[0][0] = 96; sources[0][1] = 112;
    execute("compact_same_block_cancellation", 1, 1, 1, 80, 16, 240, 16, 2, 0);
    if (expected[1][240][0] != 0) fail("compact_witness_constant", 1);
    execute("dense_same_block_cancellation", 0, 0, canonical(96, 32), 80, 16, 240, 16, 2, 0);
    if (expected[0][240][0] != -1) fail("dense_witness_constant", 1);
    load_weight(0, 3, K, N);
    lengths[0] = 16; lengths[1] = 1;
    for (unsigned x = 0; x < 16; ++x) sources[0][x] = x;
    sources[1][0] = 16;
    execute("carrier24_partial_saturation", 1, 1, 2, 0, 16, 352, 16, 3, 0);
    if (expected[1][352][0] != 16777215) fail("carrier24_witness_constant", 1);
    printf("E4B2_ROCC_SATURATION PASS ordered=-1 compact=0 dense=-1 carrier24=16777215\n");
}

#if defined(__linux__)
int main(int argc, char **argv) {
    const int probe_only = argc == 2 && strcmp(argv[1], "--probe-only") == 0;
    if (argc != 1 && !probe_only) return 2;
#else
int main(void) {
    const int probe_only = 0;
#endif
    uint64_t response;
    ++counts[126]; ISSUE1(126, response, UINT64_C(0xa), 0);
    ++counts[126]; ISSUE1(126, response, UINT64_C(0x504f544100000000), 0);
    printf("E4B2_CAPABILITY actual=0x%016llx\n", (unsigned long long)response);
    if (response != UINT64_C(0x504f544102100007)) {
        printf("E4B2_UNSUPPORTED no_extended_command_issued=1\n");
        return 2;
    }
    if (probe_only) {
        printf("E4B2_CAPABILITY_SUPPORTED no_extended_command_issued=1\n");
        return 0;
    }
#if defined(__linux__)
    /* Unlike bare-metal BSS, Linux DMA needs writable, resident backing. */
    memset((void *)actual, 0xa5, sizeof(actual));
    if (mlock(weight, sizeof(weight)) != 0 ||
        mlock(activation, sizeof(activation)) != 0 ||
        mlock((void *)actual, sizeof(actual)) != 0) {
        perror("E4B2 raw DMA buffer lock");
        return 1;
    }
    printf("E4B2_DMA_BUFFERS_READY prefaulted=1 locked=1\n");
#endif
    ++counts[7]; ISSUE0(7, 0, 0);
    fence_potal();
#if !defined(E4B2_RTL_SATURATION)
    addressing(0);
    addressing(1);
#endif
    witnesses();
    fence_potal();
    for (unsigned f = 0; f < 128; ++f) if (counts[f])
        printf("E4B2_ROCC_COMMAND funct=%u count=%llu\n", f, (unsigned long long)counts[f]);
    printf("E4B2_ROCC_PASS runs=%llu checks=%llu B_only_MVIN=%llu view_configurations=%llu "
           "scope=generated_LOOP_numeric actual_source=ACC_MVOUT\n",
           (unsigned long long)runs, (unsigned long long)checks,
           (unsigned long long)b_mvins, (unsigned long long)view_configurations);
    return 0;
}
