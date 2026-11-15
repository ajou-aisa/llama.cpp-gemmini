/* Test-local replacement for potal-api, NOT an additional potal-api object.
 * Compile as C11 with exactly potal-api's generated/original header paths.
 * This keeps the current private issuer, trace, counters and sticky failures.
 */
#include "../potal.c"
#include "potal-e3-context.h"

potal_status potal_e3_acc_restore_issue(potal_paired_session *s,
                                       const potal_e3_acc_restore *r) {
    const potal_status ready = paired_ready(s);
    if (ready != POTAL_OK) {
        return ready;
    }
    if (s->outstanding_loops || s->outstanding_readbacks || s->outstanding_acc_updates) {
        return POTAL_ERR_BUSY;
    }
    if (r == NULL) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    /* Reuse the checked full-width matrix geometry, not the broadcast bias
     * encoder. This cast is validation-only; the source is never written.
     */
    const potal_acc_readback shape = {
        r->acc_bank, r->i_tiles, r->j_tiles, r->pad_i, r->pad_j,
        (int32_t *)r->matrix, r->stride
    };
    const potal_status status = potal_acc_readback_check(&shape);
    if (status != POTAL_OK) {
        return status;
    }
    const uint64_t tiles = (uint64_t)r->i_tiles * r->j_tiles;
    if (tiles > POTAL_WORK_IDS_PER_LOOP) {
        return POTAL_ERR_CAPACITY; /* bounded to the existing readback seam */
    }
    const uint64_t rows = (uint64_t)r->i_tiles * POTAL_HW_DIM - r->pad_i;
    const uint64_t cols = (uint64_t)r->j_tiles * POTAL_HW_DIM - r->pad_j;
    const uint64_t bytes = matrix_extent(rows, cols, r->stride, sizeof(int32_t));
    if (bytes > UINTPTR_MAX || (uintptr_t)r->matrix > UINTPTR_MAX - bytes) {
        return POTAL_ERR_INVALID_ARGUMENT;
    }
    if (!paired_can_issue(s)) {
        return POTAL_ERR_NO_HARDWARE;
    }
#if defined(__riscv)
    potal_cpu_fence_rw();
#endif
    /* State zero, shrink=0, identity, one pixel repeat, full-width ACC input.
     * Nonzero row stride is essential: stride zero broadcasts one bias row.
     */
    PAIRED_RAW(s, POTAL_FUNCT_CONFIG, config_ld_rs1(0),
               r->stride * sizeof(int32_t), 0, "e3_acc_restore");
    for (uint32_t i = 0; i < r->i_tiles; ++i) {
        const uint64_t nr = POTAL_HW_DIM - (i + 1 == r->i_tiles ? r->pad_i : 0);
        for (uint32_t j = 0; j < r->j_tiles; ++j) {
            const uint64_t nc = POTAL_HW_DIM - (j + 1 == r->j_tiles ? r->pad_j : 0);
            const uint64_t dram = (uint64_t)(uintptr_t)r->matrix +
                ((uint64_t)i * POTAL_HW_DIM * r->stride + (uint64_t)j * POTAL_HW_DIM) * sizeof(int32_t);
            const uint64_t row = (uint64_t)r->acc_bank * (POTAL_HW_ACC_ROWS / 2) +
                ((uint64_t)i * r->j_tiles + j) * POTAL_HW_DIM;
            /* acc=1, accumulate=0, full-readback flag=0. Mark BEFORE issue:
             * a sink error can represent a command that partially reached DMA.
             */
            ++s->outstanding_acc_updates;
            PAIRED_RAW(s, POTAL_FUNCT_MVIN, dram,
                       (nr << MVOUT_ROWS_SHIFT) | (nc << MVOUT_COLS_SHIFT) | LOCAL_ADDR_ACC | row,
                       0, "e3_acc_restore");
        }
    }
    return POTAL_OK;
}
