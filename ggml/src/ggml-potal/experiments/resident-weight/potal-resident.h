/* E4A test-only resident interface. No production API or ISA changes.
 * One issuer, exclusive process/hart, trusted clean accelerator startup.
 * All commands in this harness MUST use this interface: legacy Scale and raw
 * RoCC functions bypass paired_owner and are explicitly outside its contract.
 */
#ifndef POTAL_RESIDENT_H
#define POTAL_RESIDENT_H

#include "potal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct potal_resident_handle potal_resident_handle;
typedef struct potal_resident_token {
    uint64_t owner, generation;
} potal_resident_token;

typedef enum potal_resident_state {
    POTAL_RESIDENT_NEW,
    POTAL_RESIDENT_READY,
    POTAL_RESIDENT_FAILED,
    POTAL_RESIDENT_CLEANED,
    POTAL_RESIDENT_QUARANTINED
} potal_resident_state;

typedef enum potal_resident_buffer {
    POTAL_RESIDENT_B0,
    POTAL_RESIDENT_B1,
    POTAL_RESIDENT_MAIN_A,
    POTAL_RESIDENT_RESIDUAL_A
} potal_resident_buffer;

typedef struct potal_resident_load {
    uint32_t pair, bank, b_sp_base, b_row_stride;
    uint32_t k_rows, j_columns, main_a_tiles;
    uint64_t k_begin, j_begin, weight_id, weight_generation;
    size_t source_stride;
} potal_resident_load;

/* One Scale window, bound to original K32 block identities. Carriers are
 * block-major with carrier_stride entries per block; j_columns is the exact
 * valid column extent, not padded width. Input arrays are copied before issue.
 */
typedef struct potal_resident_scales {
    potal_resident_token resident;
    uint32_t base, generation, blocks, j_columns;
    uint64_t j_begin;
    const uint64_t *original_blocks;
    const uint32_t *carriers;
    size_t carrier_stride;
} potal_resident_scales;

/* The embedded descriptor retains all existing geometry/Scale/path fields.
 * Main supplies its REAL retained B source view for pure checked encoding.
 * Only after resident layout equivalence is proved does E4 suppress B DMA.
 * Residual keeps the existing B=NULL/indexed contract.
 *
 * Supported execution is deliberately bounded to full I1/J2 output:
 * native Main K2/P2 suffix, or Indexed Residual K1/P1 with 16 entries.
 * Other valid ISA shapes are UNSUPPORTED here, not claimed impossible in RTL.
 * first/last are logical output-epoch flags; RTL work completion remains per
 * invocation. A continued epoch requires the next original Main K interval
 * or next Residual fragment sequence, identical output identity and shape.
 */
typedef struct potal_resident_microtile {
    potal_paired_loop_ws command;
    potal_resident_token resident, epoch;
    uint64_t output_id, original_k_begin, sequence, index_generation;
    uint32_t b_sp_base, b_row_stride, acc_bank, acc_base;
    uint32_t first, last;
} potal_resident_microtile;

typedef struct potal_resident_pair_info {
    potal_resident_token token;
    potal_resident_load description;
    uint32_t loading, readable, consumers;
} potal_resident_pair_info;

typedef struct potal_resident_acc_info {
    potal_resident_token token;
    uint64_t output_id, next_sequence, next_original_k;
    uint32_t initialized, complete, pending, readback_ready;
} potal_resident_acc_info;

typedef struct potal_resident_status {
    potal_resident_state state;
    potal_status error, last_drain, cleanup_status;
    uint32_t pending, unknown_fence, cleanup_attempted, control_pending;
    uint64_t owned_bytes, attempted_b_mvins, index_generation;
    potal_paired_session_status core;
    potal_resident_pair_info pair[POTAL_PAIRS];
    potal_resident_acc_info acc[2];
} potal_resident_status;

/* On success E4 owns its aligned DMA allocations and the supplied sink context.
 * A non-null sink requires a destructor; open failure leaves context ownership
 * with the caller. Destruction happens only after safe close. Quarantine keeps
 * handle, core, sink context and all DMA allocations for process lifetime.
 * Sink callbacks must not issue commands or reenter any mutating API.
 */
potal_status potal_resident_open(const potal_paired_sink *sink,
                               void (*destroy_sink)(void *),
                               potal_resident_handle **out);
potal_status potal_resident_prepare(potal_resident_handle *h);
/* Mutable CPU views only in initialized, healthy, idle state. Each allocation
 * is 65536 bytes, 64-byte aligned. No caller may free it or retain a right to
 * access it during pending DMA. B bytes may be poisoned after load completion
 * to test residency; the resident SP bytes and identity do not then change.
 */
potal_status potal_resident_buffer_view(potal_resident_handle *h,
                                        potal_resident_buffer which,
                                        void **data, size_t *bytes);
potal_status potal_resident_load_issue(potal_resident_handle *h,
                                      const potal_resident_load *load);
/* No resident token is published until a successful qualifying drain. */
potal_status potal_resident_get(potal_resident_handle *h, uint32_t pair,
                               potal_resident_token *token);
potal_status potal_resident_scale_program(potal_resident_handle *h,
                                         const potal_resident_scales *scales);
potal_status potal_resident_scale_release(potal_resident_handle *h,
                                         uint32_t slot, uint32_t generation,
                                         uint32_t column);
potal_status potal_resident_index_program(potal_resident_handle *h,
                                         potal_resident_token resident,
                                         const uint64_t *original_k,
                                         uint32_t count, uint64_t *generation);
potal_status potal_resident_microtile_issue(potal_resident_handle *h,
                                           const potal_resident_microtile *tile);
potal_status potal_resident_readback_issue(potal_resident_handle *h,
                                          uint32_t bank, potal_resident_token epoch);
/* Returns a const completed readback view only. Failed/partial DMA never
 * publishes this view. A subsequent readback or close invalidates old views.
 */
potal_status potal_resident_result(potal_resident_handle *h, uint32_t bank,
                                   potal_resident_token epoch,
                                   const int32_t **data, size_t *stride);
potal_status potal_resident_fence(potal_resident_handle *h);
potal_status potal_resident_retire(potal_resident_handle *h,
                                  potal_resident_token resident);
potal_status potal_resident_cleanup(potal_resident_handle *h);
/* No implicit drain. Busy/quarantined close retains *h. Known-safe cleaned
 * failure consumes *h while returning the original sticky error.
 */
potal_status potal_resident_close(potal_resident_handle **h);
potal_resident_status potal_resident_query(const potal_resident_handle *h);

/* Checked nonwrapping software token arithmetic, also used by the handle.
 * It is distinct from legal eight-bit Scale reuse after release.
 */
potal_status potal_resident_generation_next(uint64_t current, uint64_t *next);

#ifdef __cplusplus
}
#endif
#endif
