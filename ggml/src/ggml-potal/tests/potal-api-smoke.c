/*
 * PoTal API hardware smoke test (RISC-V Linux or bare-metal, DIM16 Sat32 SoC).
 *
 * Sections follow the passing bare-metal references in
 * generators/potal/software/gemmini-rocc-tests/bareMetalC:
 *   A. scale control  (potal_scale_control.c)
 *   B. DIM16 K16 GEMM (potal_gemm_d16.c, non-identity A, SCU saturation)
 *   C. DIM16 K32 GEMM (potal_gemm_k32.c / potal_sat32_d16.c), passesPerBlock 1 and 2
 *
 * Golden model: per K pass, SCU = sat32(raw << shift) (0 for the zero carrier,
 * +/-INT32 saturation for shift >= 32 when raw != 0), then ACC = sat32(ACC + SCU),
 * with the first pass overwriting ACC. Computed in int64 independently of the API.
 *
 * Output markers: "PASS <step>", "FAIL <step> ...", final "POTAL_API_SMOKE PASS".
 * Exit status: 0 pass, 1 fail.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "potal.h"
#include "potal_hw_params.h"
#include "potal_rocc.h"

#if POTAL_HW_DIM != 16
#error "potal-api-smoke validates the DIM16 GEMM path only"
#endif

#define DIM      16
#define K_MAX    (2 * DIM)
#define SENTINEL (-9999)

static int8_t  A[DIM][K_MAX] __attribute__((aligned(POTAL_HW_MAX_BYTES)));
static int8_t  B[K_MAX][DIM] __attribute__((aligned(POTAL_HW_MAX_BYTES)));
static int32_t C[DIM][DIM]   __attribute__((aligned(POTAL_HW_MAX_BYTES)));

static void fail(const char *step, int detail) {
    printf("FAIL %s (%d)\n", step, detail);
    printf("POTAL_API_SMOKE FAIL\n");
    exit(1);
}

static void expect(const char *step, potal_status actual, potal_status expected) {
    if (actual != expected) {
        printf("FAIL %s: got=%d(%s) expected=%d(%s)\n", step, (int)actual,
               potal_status_name(actual), (int)expected, potal_status_name(expected));
        printf("POTAL_API_SMOKE FAIL\n");
        exit(1);
    }
    printf("PASS %s\n", step);
}

/* Raw command for deliberately malformed encodings the checked API refuses. */
static potal_status raw_write(uint64_t rs1, uint64_t rs2) {
    const uint64_t rd = potal_rocc_scale_write(rs1, rs2);
    return rd <= POTAL_HW_BAD_COMMAND ? (potal_status)rd : POTAL_ERR_BAD_RESPONSE;
}

static uint64_t key(uint64_t slot, uint64_t generation, uint64_t column) {
    return slot | (generation << 8) | (column << 16);
}

/* ---- A. Scale control --------------------------------------------------- */

static void test_scale_control(void) {
    expect("A write shift carrier", potal_scale_write(3, 7, 0, 3), POTAL_OK);
    expect("A write zero carrier", potal_scale_write(3, 7, 15, POTAL_CARRIER_ZERO), POTAL_OK);
    expect("A duplicate write", potal_scale_write(3, 8, 0, 4), POTAL_HW_SCALE_CONFLICT);
    expect("A wrong generation release", potal_scale_release(3, 8, 0), POTAL_HW_RELEASE_MISS);
    expect("A release never written", potal_scale_release(5, 1, 0), POTAL_HW_RELEASE_MISS);
    expect("A correct generation release", potal_scale_release(3, 7, 0), POTAL_OK);
    expect("A reuse slot", potal_scale_write(3, 9, 0, 7), POTAL_OK);

    /* The checked API refuses these before issuing anything. */
    expect("A api rejects carrier", potal_scale_write(4, 1, 0, 0xffffffffu), POTAL_ERR_INVALID_ARGUMENT);
    expect("A api rejects column", potal_scale_write(6, 1, DIM, 1), POTAL_ERR_INVALID_ARGUMENT);
    expect("A api rejects slot", potal_scale_write(256, 1, 0, 1), POTAL_ERR_INVALID_ARGUMENT);

    /* Hardware checks on raw encodings, priority per PoTalScaleCommandIngress. */
    expect("A hw invalid carrier", raw_write(key(4, 1, 0), 0xffffffffu), POTAL_HW_INVALID_SCALE);
    expect("A hw invalid column", raw_write(key(6, 1, DIM), 1), POTAL_HW_INVALID_COLUMN);
    expect("A hw reserved rs1 bit", raw_write(key(5, 1, 0) | (1ull << 22), 1), POTAL_HW_INVALID_ENCODING);
    expect("A hw nonzero rs2 high", raw_write(key(5, 1, 0), 1ull << 32), POTAL_HW_INVALID_ENCODING);
    expect("A hw encoding before column", raw_write(key(6, 1, DIM) | (1ull << 22), 1), POTAL_HW_INVALID_ENCODING);
    expect("A hw column before carrier", raw_write(key(6, 1, DIM), 0xffffffffu), POTAL_HW_INVALID_COLUMN);
    expect("A hw fence nonzero rs1", (potal_status)potal_rocc_fence(1, 0), POTAL_HW_INVALID_ENCODING);
    /* Rejected writes must not have allocated their slots. */
    expect("A rejected slot 4 not allocated", potal_scale_release(4, 1, 0), POTAL_HW_RELEASE_MISS);

    expect("A fence", potal_fence(), POTAL_OK);
    expect("A release reused slot", potal_scale_release(3, 9, 0), POTAL_OK);
    expect("A release zero carrier", potal_scale_release(3, 7, 15), POTAL_OK);
    expect("A final fence", potal_fence(), POTAL_OK);
}

/* ---- Golden model ------------------------------------------------------- */

static int32_t sat32(int64_t v) {
    return v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : (int32_t)v;
}

static int32_t scu(int32_t raw, uint32_t carrier) {
    if (carrier == POTAL_CARRIER_ZERO || raw == 0) {
        return 0;
    }
    if (carrier >= 32) {
        return raw > 0 ? INT32_MAX : INT32_MIN;
    }
    return sat32((int64_t)raw * ((int64_t)1 << carrier));
}

static int32_t raw_partial(int pass, int row, int col) {
    int32_t sum = 0;
    for (int k = pass * DIM; k < (pass + 1) * DIM; ++k) {
        sum += (int32_t)A[row][k] * (int32_t)B[k][col];
    }
    return sum;
}

/* Per-column carriers covering x1, small shifts, zero, SCU saturation, >= 32. */
static uint32_t carrier_for(int block, int col) {
    static const uint32_t carriers[DIM] = {
        0, 1, POTAL_CARRIER_ZERO, 2, 5, 13, 20, 31,
        32, 32767, 3, POTAL_CARRIER_ZERO, 7, 0, 1, 4,
    };
    return carriers[(col + 5 * block) % DIM];
}

static void fill_inputs(int k, int seed) {
    for (int i = 0; i < DIM; ++i) {
        for (int kk = 0; kk < k; ++kk) {
            /* Full signed int8 range, including -128 and 127. */
            A[i][kk] = (int8_t)(((i * 37 + kk * 11 + seed) & 0xff) - 128);
        }
    }
    for (int kk = 0; kk < k; ++kk) {
        for (int j = 0; j < DIM; ++j) {
            B[kk][j] = (int8_t)(((kk * 23 + j * 59 + 3 * seed) & 0xff) - 128);
        }
    }
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            C[i][j] = SENTINEL;
        }
    }
}

/* ---- B/C. GEMM ---------------------------------------------------------- */

static void run_gemm(const char *name, int k, uint32_t base, uint32_t generation, uint32_t passes) {
    const int k_tiles = k / DIM;
    const int blocks = k_tiles / (int)passes;
    potal_gemm_ws g = {
        .a = &A[0][0], .b = &B[0][0], .c = &C[0][0],
        .lda = K_MAX, .ldb = DIM, .ldc = DIM,
        .m = DIM, .n = DIM, .k = (uint32_t)k,
        .scale_base = base, .scale_generation = generation, .passes_per_block = passes,
        .flags = 0,
    };

    printf("%s: K=%d passesPerBlock=%u base=%u gen=%u\n", name, k, (unsigned)passes,
           (unsigned)base, (unsigned)generation);
    fill_inputs(k, (int)base + k);

    for (int b = 0; b < blocks; ++b) {
        uint32_t slot;
        if (potal_gemm_ws_scale_slot(&g, (uint32_t)b, 0, &slot) != POTAL_OK) {
            fail("scale slot", b);
        }
        for (int col = 0; col < DIM; ++col) {
            const potal_status s = potal_scale_write(slot, generation, (uint32_t)col, carrier_for(b, col));
            if (s != POTAL_OK) {
                printf("FAIL %s scale write block=%d col=%d status=%s\n", name, b, col, potal_status_name(s));
                fail(name, (int)s);
            }
        }
    }
    printf("PASS %s scale initialization\n", name);

    expect("GEMM issue", potal_gemm_ws_issue(&g), POTAL_OK);
    expect("GEMM issue while in flight", potal_gemm_ws_issue(&g), POTAL_ERR_BUSY);
    expect("GEMM fence", potal_fence(), POTAL_OK);

    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            int32_t acc = 0;
            for (int pass = 0; pass < k_tiles; ++pass) {
                const int32_t part = scu(raw_partial(pass, i, j), carrier_for(pass / (int)passes, j));
                acc = pass == 0 ? part : sat32((int64_t)acc + part);
            }
            if (C[i][j] != acc) {
                printf("FAIL %s C[%d][%d] expected=%ld actual=%ld raw0=%ld raw1=%ld\n", name, i, j,
                       (long)acc, (long)C[i][j], (long)raw_partial(0, i, j),
                       (long)(k_tiles > 1 ? raw_partial(1, i, j) : 0));
                fail(name, i * DIM + j);
            }
        }
    }
    printf("PASS %s numerical comparison\n", name);

    for (int b = 0; b < blocks; ++b) {
        uint32_t slot;
        (void)potal_gemm_ws_scale_slot(&g, (uint32_t)b, 0, &slot);
        for (int col = 0; col < DIM; ++col) {
            if (potal_scale_release(slot, generation, (uint32_t)col) != POTAL_OK) {
                printf("FAIL %s release block=%d col=%d\n", name, b, col);
                fail(name, b);
            }
        }
    }
    printf("PASS %s release\n", name);
}

int main(void) {
    printf("POTAL_API_SMOKE starting DIM=%u params_sha256=%s\n", (unsigned)potal_hw_dim(),
           potal_hw_params_sha256());

    test_scale_control();
    printf("PASS A scale control\n");

    run_gemm("B DIM16 K16", DIM, 0, 1, 1);
    run_gemm("C DIM16 K32 ppb1", 2 * DIM, 8, 7, 1);
    run_gemm("C DIM16 K32 ppb2", 2 * DIM, 40, 9, 2);

    printf("POTAL_API_SMOKE PASS\n");
    return 0;
}
