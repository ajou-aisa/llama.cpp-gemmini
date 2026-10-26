#ifndef POTAL_E3_CONTEXT_H
#define POTAL_E3_CONTEXT_H

#include "potal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Test-only raw signed32 value epoch, not a restored hardware work ID.
 * Source spans (I*DIM-pad_i) rows of stride elements, with J*DIM-pad_j
 * valid columns. It must be DIM*4-byte aligned, mapped and unchanged until
 * a successful checked fence/cleanup drain. On unknown completion retain
 * the allocation and accelerator ownership for the remainder of the process.
 * Caller supplies a real allocation of this extent; this API cannot discover
 * allocation lengths. No SCU transform is applied.
 */
typedef struct potal_e3_acc_restore {
    uint32_t acc_bank;
    uint32_t i_tiles, j_tiles;
    uint32_t pad_i, pad_j;
    const int32_t *matrix;
    size_t stride;
} potal_e3_acc_restore;

/* Requires all prior readers/writers drained. Pending restores exclude
 * LOOP_WS, readback, another restore, bias, flush and healthy close.
 * Fence before consuming/restoring again. Use cleanup after an issue failure;
 * never retry a failed fence. Original session sticky-error rules apply.
 */
potal_status potal_e3_acc_restore_issue(potal_paired_session *session,
                                       const potal_e3_acc_restore *restore);

#ifdef __cplusplus
}
#endif
#endif
