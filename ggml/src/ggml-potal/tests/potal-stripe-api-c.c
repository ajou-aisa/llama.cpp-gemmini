/*
 * C11 caller of the planned stripe GEMM API (potal.h): the header compiles
 * as C, and the C++ library potal-tiled-api links into a C translation unit.
 * Only refused descriptors are sent, so nothing reaches an accelerator on
 * any platform. Called by potal-stripe-api-test.
 */
#include "potal.h"

#include <stddef.h>
#include <string.h>

size_t potal_stripe_api_c_layout(size_t *out, size_t count);
int    potal_stripe_api_c_caller(void);

/* The ABI layout seen by the C compiler; the C++ side compares its own. */
size_t potal_stripe_api_c_layout(size_t *out, size_t count) {
    const size_t v[] = {
        sizeof(potal_stripe_gemm_desc),
        offsetof(potal_stripe_gemm_desc, work_kind),
        offsetof(potal_stripe_gemm_desc, m),
        offsetof(potal_stripe_gemm_desc, a),
        offsetof(potal_stripe_gemm_desc, lda),
        offsetof(potal_stripe_gemm_desc, carriers),
        offsetof(potal_stripe_gemm_desc, scale_block_k),
        offsetof(potal_stripe_gemm_desc, tile_k),
        sizeof(potal_stripe_gemm_report),
        offsetof(potal_stripe_gemm_report, fatal),
        offsetof(potal_stripe_gemm_report, loop_ws),
        offsetof(potal_stripe_gemm_report, last_generation),
        offsetof(potal_stripe_gemm_report, tile_i),
        offsetof(potal_stripe_gemm_report, pad_k),
        offsetof(potal_stripe_gemm_report, fence_s),
    };
    const size_t n = sizeof(v) / sizeof(v[0]);
    size_t       i;
    for (i = 0; i < count && i < n; ++i) {
        out[i] = v[i];
    }
    return n;
}

/* 0 when every check passed, else the number of the first failing one. */
int potal_stripe_api_c_caller(void) {
    potal_stripe_session    *session = NULL;
    potal_stripe_gemm_desc   desc;
    potal_stripe_gemm_report report;

    if (potal_stripe_session_create(NULL) != POTAL_ERR_INVALID_ARGUMENT) {
        return 1;
    }
    if (potal_stripe_session_create(&session) != POTAL_OK || session == NULL) {
        return 2;
    }
    memset(&desc, 0, sizeof(desc));
    desc.work_kind     = POTAL_STRIPE_WORK_MAIN;
    desc.m             = 16;
    desc.n             = 16;
    desc.k             = 32;
    desc.lda           = 32;
    desc.ldb           = 16;
    desc.ldc           = 16;
    desc.scale_block_k = 32;
    desc.tile_i        = 1;
    desc.tile_j        = 1;
    desc.tile_k        = 2;
    /* a, b, c and carriers stay NULL: refused before any command. */
    if (potal_tiled_matmul_stripe(session, &desc, &report) != POTAL_ERR_INVALID_ARGUMENT ||
        report.status != POTAL_ERR_INVALID_ARGUMENT || report.stage == NULL || report.fatal != 0 ||
        report.loop_ws != 0 || report.scale_writes != 0 || report.fences != 0) {
        return 3;
    }
    if (potal_tiled_matmul_stripe(session, &desc, NULL) != POTAL_ERR_INVALID_ARGUMENT) {
        return 4;
    }
    if (potal_tiled_matmul_stripe(NULL, &desc, &report) != POTAL_ERR_INVALID_ARGUMENT || report.fatal != 0) {
        return 5;
    }
    if (potal_tiled_matmul_stripe(session, NULL, &report) != POTAL_ERR_INVALID_ARGUMENT) {
        return 6;
    }
    potal_stripe_session_destroy(session);
    potal_stripe_session_destroy(NULL);
    return 0;
}
