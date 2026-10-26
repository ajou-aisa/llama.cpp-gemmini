/* Test-local replacement for potal-api, never linked alongside another copy.
 * The core remains unchanged and owns the accelerator. This closed harness
 * mediates every command and owns DMA storage that the core cannot track.
 */
#include "../potal.c"
#include "potal-resident.h"
#if defined(__riscv)
#include <sys/mman.h>
#endif

#define E4_BYTES 65536u
#define E4_POOLS 6u
enum e4_pending { E4_NONE, E4_B_LOAD, E4_COMPUTE, E4_READBACK };
typedef struct e4_scale {
    potal_resident_token resident;
    uint64_t block, column;
} e4_scale;
typedef struct e4_index {
    potal_resident_token resident;
    uint64_t generation, original[POTAL_HW_INDEX_ENTRIES];
    uint32_t count;
} e4_index;

struct potal_resident_handle {
    potal_paired_session *core;
    potal_resident_status q;
    void *pool[E4_POOLS];
    void (*destroy_sink)(void *);
    void *sink_context;
    uint64_t owner, sequence;
    int in_call;
    enum e4_pending kind;
    uint32_t pending_pair, pending_bank, issue_complete, pending_last;
    uint64_t pending_next_sequence, pending_next_k;
    uint64_t acc_j_begin[2], acc_weight_id[2], acc_weight_generation[2];
    e4_scale scale[POTAL_SCALE_ENTRIES];
    e4_index index[POTAL_PAIRS];
};

/* Single issuer, like the core. Never wrap a handle identity. */
static uint64_t e4_owner_sequence;

potal_status potal_resident_generation_next(uint64_t current, uint64_t *next) {
    if (next == NULL) return POTAL_ERR_INVALID_ARGUMENT;
    if (current == UINT64_MAX) return POTAL_ERR_CAPACITY;
    *next = current + 1;
    return POTAL_OK;
}

static int e4_same(potal_resident_token a, potal_resident_token b) {
    return a.owner == b.owner && a.generation == b.generation && a.owner && a.generation;
}

static potal_status e4_ready(const potal_resident_handle *h, int allow_pending) {
    if (h == NULL) return POTAL_ERR_INVALID_ARGUMENT;
    if (h->in_call) return POTAL_ERR_BUSY;
    if (h->q.state == POTAL_RESIDENT_FAILED || h->q.state == POTAL_RESIDENT_CLEANED ||
        h->q.state == POTAL_RESIDENT_QUARANTINED) return h->q.error;
    if (h->q.state != POTAL_RESIDENT_READY) return POTAL_ERR_BUSY;
    if (!allow_pending && (h->q.pending || h->q.control_pending)) return POTAL_ERR_BUSY;
    return paired_ready(h->core);
}

static potal_status e4_token(potal_resident_handle *h, potal_resident_token *out) {
    uint64_t next;
    const potal_status s = potal_resident_generation_next(h->sequence, &next);
    if (s != POTAL_OK) return s;
    h->sequence = next;
    *out = (potal_resident_token){h->owner, next};
    return POTAL_OK;
}

static int e4_pair(const potal_resident_handle *h, potal_resident_token token) {
    for (uint32_t p = 0; p < POTAL_PAIRS; ++p)
        if (h->q.pair[p].readable && e4_same(h->q.pair[p].token, token)) return (int)p;
    return -1;
}

static int e4_span(const potal_resident_handle *h, uint32_t pool,
                   const void *pointer, uint64_t bytes) {
    if (pool >= E4_POOLS || pointer == NULL || bytes == 0 || bytes > E4_BYTES) return 0;
    const uintptr_t base = (uintptr_t)h->pool[pool], address = (uintptr_t)pointer;
    return address >= base && address - base <= E4_BYTES - bytes;
}

static potal_status e4_failed(potal_resident_handle *h, potal_status status,
                              const char *stage, int quarantine) {
    (void)paired_fail(h->core, status, stage);
    h->q.error = h->core->error;
    h->q.state = quarantine ? POTAL_RESIDENT_QUARANTINED : POTAL_RESIDENT_FAILED;
    return status;
}

static void e4_complete(potal_resident_handle *h, int publish) {
    if (h->kind == E4_B_LOAD) {
        potal_resident_pair_info *r = &h->q.pair[h->pending_pair];
        r->loading = 0;
        r->readable = publish && h->issue_complete;
        if (r->readable) {
            paired_pair_state *p = &h->core->pair[h->pending_pair];
            p->resident_b = 1;
            p->b_sp_base = r->description.b_sp_base;
            p->b_j_tiles = (r->description.j_columns + POTAL_HW_DIM - 1) / POTAL_HW_DIM;
            p->b_k_rows = r->description.k_rows;
            p->main_a_tiles = r->description.main_a_tiles;
            p->index_valid = 0;
        }
    } else if (h->kind == E4_COMPUTE) {
        potal_resident_acc_info *a = &h->q.acc[h->pending_bank];
        a->pending = 0;
        a->initialized = publish && h->issue_complete;
        a->complete = a->initialized && h->pending_last;
        if (a->initialized) {
            a->next_sequence = h->pending_next_sequence;
            a->next_original_k = h->pending_next_k;
        }
        h->q.pair[h->pending_pair].consumers = 0;
    } else if (h->kind == E4_READBACK) {
        h->q.acc[h->pending_bank].pending = 0;
        h->q.acc[h->pending_bank].readback_ready = publish && h->issue_complete;
    }
    h->q.pending = 0;
    h->q.control_pending = 0;
    h->kind = E4_NONE;
    h->issue_complete = 0;
}

static void e4_free(potal_resident_handle *h) {
    for (uint32_t i = 0; i < E4_POOLS; ++i) {
#if defined(__riscv)
        if (h->pool[i]) (void)munlock(h->pool[i], E4_BYTES);
#endif
        free(h->pool[i]);
    }
    if (h->destroy_sink) h->destroy_sink(h->sink_context);
    free(h);
}

potal_status potal_resident_open(const potal_paired_sink *sink,
                               void (*destroy_sink)(void *),
                               potal_resident_handle **out) {
    if (out == NULL || (sink && (!sink->cmd || !sink->ctx || !destroy_sink)) ||
        (!sink && destroy_sink)) return POTAL_ERR_INVALID_ARGUMENT;
    *out = NULL;
    uint64_t owner;
    potal_status status = potal_resident_generation_next(e4_owner_sequence, &owner);
    if (status != POTAL_OK) return status;
    potal_resident_handle *h = calloc(1, sizeof(*h));
    if (!h) return POTAL_ERR_CAPACITY;
    status = potal_paired_session_open(sink, &h->core);
    if (status != POTAL_OK) { free(h); return status; }
    for (uint32_t i = 0; i < E4_POOLS; ++i) {
        h->pool[i] = aligned_alloc(64, E4_BYTES);
        if (!h->pool[i]) { status = POTAL_ERR_CAPACITY; break; }
        memset(h->pool[i], 0, E4_BYTES);
#if defined(__riscv)
        if (mlock(h->pool[i], E4_BYTES) != 0) { status = POTAL_ERR_CAPACITY; break; }
#endif
    }
    if (status != POTAL_OK) {
        (void)potal_paired_session_close(h->core); /* No command or DMA occurred. */
        e4_free(h); /* The caller still owns the sink on open failure. */
        return status;
    }
    e4_owner_sequence = owner;
    h->owner = owner;
    h->q.state = POTAL_RESIDENT_NEW;
    h->q.owned_bytes = E4_POOLS * E4_BYTES;
    h->sink_context = sink ? sink->ctx : NULL;
    h->destroy_sink = destroy_sink;
    *out = h;
    return POTAL_OK;
}

potal_status potal_resident_prepare(potal_resident_handle *h) {
    if (!h) return POTAL_ERR_INVALID_ARGUMENT;
    if (h->in_call || h->q.state != POTAL_RESIDENT_NEW || h->q.control_pending)
        return h->q.error != POTAL_OK ? h->q.error : POTAL_ERR_BUSY;
    if (!paired_can_issue(h->core)) return POTAL_ERR_NO_HARDWARE;
    h->q.control_pending = 1;
    h->in_call = 1;
    const potal_status status = potal_paired_tlb_flush(h->core);
    h->in_call = 0;
    if (status != POTAL_OK) return e4_failed(h, status, "e4_prepare", 0);
    return potal_resident_fence(h);
}

potal_status potal_resident_buffer_view(potal_resident_handle *h,
                                        potal_resident_buffer which,
                                        void **data, size_t *bytes) {
    const potal_status ready = e4_ready(h, 0);
    if (ready != POTAL_OK) return ready;
    if ((unsigned)which > POTAL_RESIDENT_RESIDUAL_A || !data || !bytes)
        return POTAL_ERR_INVALID_ARGUMENT;
    *data = h->pool[which];
    *bytes = E4_BYTES;
    return POTAL_OK;
}

static potal_status e4_load_raw(potal_resident_handle *h) {
    potal_paired_session *s = h->core;
    const potal_resident_load *d = &h->q.pair[h->pending_pair].description;
    const uint32_t kt = (d->k_rows + POTAL_HW_DIM - 1) / POTAL_HW_DIM;
    const uint32_t jt = (d->j_columns + POTAL_HW_DIM - 1) / POTAL_HW_DIM;
#if defined(__riscv)
    potal_cpu_fence_rw();
#endif
    PAIRED_RAW(s, POTAL_FUNCT_CONFIG, config_ld_rs1(0), d->source_stride, 0, "e4_b_load");
    for (uint32_t k = 0; k < kt; ++k) for (uint32_t j = 0; j < jt; ++j) {
        const uint64_t nr = d->k_rows - k * POTAL_HW_DIM < POTAL_HW_DIM ?
            d->k_rows - k * POTAL_HW_DIM : POTAL_HW_DIM;
        const uint64_t nc = d->j_columns - j * POTAL_HW_DIM < POTAL_HW_DIM ?
            d->j_columns - j * POTAL_HW_DIM : POTAL_HW_DIM;
        const uint64_t source = (uint64_t)(uintptr_t)h->pool[d->pair] +
            (uint64_t)k * POTAL_HW_DIM * d->source_stride + (uint64_t)j * POTAL_HW_DIM;
        const uint64_t row = d->b_sp_base + (uint64_t)k * d->b_row_stride + (uint64_t)j * POTAL_HW_DIM;
        h->q.attempted_b_mvins++;
        PAIRED_RAW(s, POTAL_FUNCT_MVIN, source,
                   (nr << MVOUT_ROWS_SHIFT) | (nc << MVOUT_COLS_SHIFT) | row, 0, "e4_b_load");
    }
    return POTAL_OK;
}

potal_status potal_resident_load_issue(potal_resident_handle *h,
                                      const potal_resident_load *d) {
    const potal_status ready = e4_ready(h, 0);
    if (ready != POTAL_OK) return ready;
    if (!d || d->pair >= POTAL_PAIRS || d->bank != 2 * d->pair + 1 ||
        !d->k_rows || d->k_rows > 28 * POTAL_HW_DIM || !d->j_columns ||
        d->j_columns > POTAL_HW_BANK_ROWS || !d->weight_id || !d->weight_generation ||
        d->source_stride < d->j_columns || d->main_a_tiles >= POTAL_HW_BANK_ROWS / POTAL_HW_DIM ||
        d->k_begin > UINT64_MAX - d->k_rows || d->j_begin > UINT64_MAX - d->j_columns)
        return POTAL_ERR_INVALID_ARGUMENT;
    if (h->q.pair[d->pair].readable || h->q.pair[d->pair].loading) return POTAL_ERR_BUSY;
    const uint64_t kt = (d->k_rows + POTAL_HW_DIM - 1) / POTAL_HW_DIM;
    const uint64_t jt = (d->j_columns + POTAL_HW_DIM - 1) / POTAL_HW_DIM;
    const uint64_t rows = kt * jt * POTAL_HW_DIM;
    const uint64_t end = pair_weight_begin(d->pair) + POTAL_HW_BANK_ROWS;
    if (rows > POTAL_HW_BANK_ROWS) return POTAL_ERR_CAPACITY;
    if (d->b_sp_base != end - rows || d->b_row_stride != jt * POTAL_HW_DIM ||
        !e4_span(h, d->pair, h->pool[d->pair],
                 matrix_extent(d->k_rows, d->j_columns, d->source_stride, 1)))
        return POTAL_ERR_INVALID_ARGUMENT;
    if (!paired_can_issue(h->core)) return POTAL_ERR_NO_HARDWARE;
    potal_resident_token token;
    const potal_status allocated = e4_token(h, &token);
    if (allocated != POTAL_OK) return allocated;
    h->q.pair[d->pair] = (potal_resident_pair_info){token, *d, 1, 0, 0};
    memset(&h->index[d->pair], 0, sizeof(h->index[d->pair]));
    h->core->pair[d->pair].resident_b = 0;
    h->core->pair[d->pair].index_valid = 0;
    h->kind = E4_B_LOAD;
    h->pending_pair = d->pair;
    h->q.pending = 1; /* Before CONFIG or the first possible DMA. */
    h->in_call = 1;
    const potal_status status = e4_load_raw(h);
    h->in_call = 0;
    if (status != POTAL_OK) return e4_failed(h, status, "e4_b_load", 0);
    h->issue_complete = 1;
    return POTAL_OK;
}

potal_status potal_resident_get(potal_resident_handle *h, uint32_t pair,
                               potal_resident_token *token) {
    const potal_status ready = e4_ready(h, 0);
    if (ready != POTAL_OK) return ready;
    if (pair >= POTAL_PAIRS || !token) return POTAL_ERR_INVALID_ARGUMENT;
    if (!h->q.pair[pair].readable) return POTAL_ERR_BUSY;
    *token = h->q.pair[pair].token;
    return POTAL_OK;
}

potal_status potal_resident_scale_program(potal_resident_handle *h,
                                         const potal_resident_scales *d) {
    const potal_status ready = e4_ready(h, 0);
    if (ready != POTAL_OK) return ready;
    if (!d || !d->blocks || !d->j_columns || !d->original_blocks || !d->carriers ||
        d->generation > POTAL_GENERATION_MAX || d->carrier_stride < d->j_columns)
        return POTAL_ERR_INVALID_ARGUMENT;
    const int pair = e4_pair(h, d->resident);
    if (pair < 0) return POTAL_ERR_INVALID_ARGUMENT;
    const potal_resident_load *r = &h->q.pair[pair].description;
    const uint64_t jt = ((uint64_t)d->j_columns + POTAL_HW_DIM - 1) / POTAL_HW_DIM;
    if (d->base >= POTAL_SCALE_ENTRIES || (uint64_t)d->blocks * jt > POTAL_SCALE_ENTRIES - d->base)
        return POTAL_ERR_CAPACITY;
    if (d->j_begin < r->j_begin || d->j_columns > r->j_columns ||
        d->j_begin - r->j_begin > r->j_columns - d->j_columns ||
        (d->j_begin - r->j_begin) % POTAL_HW_DIM ||
        matrix_extent(d->blocks, d->j_columns, d->carrier_stride, sizeof(uint32_t)) == 0)
        return POTAL_ERR_INVALID_ARGUMENT;
    uint32_t values[POTAL_SCALE_ENTRIES][POTAL_HW_DIM];
    uint64_t blocks[POTAL_SCALE_ENTRIES];
    for (uint32_t b = 0; b < d->blocks; ++b) {
        const uint64_t original = d->original_blocks[b];
        if (original < r->k_begin / 32 || original > (r->k_begin + r->k_rows - 1) / 32)
            return POTAL_ERR_INVALID_ARGUMENT;
        for (uint32_t j = 0; j < jt; ++j) {
            const uint32_t local = b * (uint32_t)jt + j, slot = d->base + local;
            blocks[local] = original;
            const uint32_t columns = d->j_columns - j * POTAL_HW_DIM < POTAL_HW_DIM ?
                d->j_columns - j * POTAL_HW_DIM : POTAL_HW_DIM;
            for (uint32_t c = 0; c < columns; ++c) {
                const uint32_t value = d->carriers[(size_t)b * d->carrier_stride + j * POTAL_HW_DIM + c];
                if (!potal_carrier_is_valid(value)) return POTAL_ERR_INVALID_ARGUMENT;
                if (h->core->slot_columns[slot]) return POTAL_ERR_CAPACITY;
                values[local][c] = value;
            }
        }
    }
    if (!paired_can_issue(h->core)) return POTAL_ERR_NO_HARDWARE;
    h->in_call = 1;
    for (uint32_t b = 0; b < d->blocks; ++b) for (uint32_t j = 0; j < jt; ++j) {
        const uint32_t local = b * (uint32_t)jt + j, slot = d->base + local;
        const uint32_t columns = d->j_columns - j * POTAL_HW_DIM < POTAL_HW_DIM ?
            d->j_columns - j * POTAL_HW_DIM : POTAL_HW_DIM;
        h->scale[slot] = (e4_scale){d->resident, blocks[local], d->j_begin + j * POTAL_HW_DIM};
        for (uint32_t c = 0; c < columns; ++c) {
            const potal_status status = potal_paired_scale_write(h->core, slot, d->generation, c, values[local][c]);
            if (status != POTAL_OK) {
                h->in_call = 0;
                return e4_failed(h, status, "e4_scale", 1); /* Possible untracked live entry. */
            }
        }
    }
    h->in_call = 0;
    return POTAL_OK;
}

potal_status potal_resident_scale_release(potal_resident_handle *h,
                                         uint32_t slot, uint32_t generation,
                                         uint32_t column) {
    const potal_status ready = e4_ready(h, 1);
    if (ready != POTAL_OK) return ready;
    uint64_t key;
    if (potal_scale_key_encode(slot, generation, column, &key) != POTAL_OK ||
        !(h->core->slot_columns[slot] & (UINT64_C(1) << column)) ||
        h->core->slot_generation[slot] != generation) return POTAL_ERR_INVALID_ARGUMENT;
    if (!paired_can_issue(h->core)) return POTAL_ERR_NO_HARDWARE;
    h->in_call = 1;
    const potal_status status = potal_paired_scale_release(h->core, slot, generation, column);
    h->in_call = 0;
    if (status != POTAL_OK) return e4_failed(h, status, "e4_release", 1);
#if defined(__riscv)
    potal_cpu_fence_rw(); /* Core Release supplies drain, but not this CPU ordering. */
#endif
    h->q.last_drain = POTAL_OK;
    e4_complete(h, 1);
    if (!h->core->slot_columns[slot]) memset(&h->scale[slot], 0, sizeof(h->scale[slot]));
    return POTAL_OK;
}

potal_status potal_resident_index_program(potal_resident_handle *h,
                                         potal_resident_token resident,
                                         const uint64_t *original_k,
                                         uint32_t count, uint64_t *generation) {
    const potal_status ready = e4_ready(h, 0);
    if (ready != POTAL_OK) return ready;
    if (!count || !original_k || !generation) return POTAL_ERR_INVALID_ARGUMENT;
    if (count > POTAL_HW_INDEX_ENTRIES) return POTAL_ERR_CAPACITY;
    const int pair = e4_pair(h, resident);
    if (pair < 0) return POTAL_ERR_INVALID_ARGUMENT;
    const potal_resident_load *r = &h->q.pair[pair].description;
    uint64_t originals[POTAL_HW_INDEX_ENTRIES];
    for (uint32_t i = 0; i < count; ++i) {
        originals[i] = original_k[i];
        if (originals[i] < r->k_begin || originals[i] - r->k_begin >= r->k_rows ||
            (i && originals[i] <= originals[i - 1])) return POTAL_ERR_INVALID_ARGUMENT;
    }
    const potal_index_config cfg = {
        (uint32_t)pair, r->b_sp_base, (r->j_columns + POTAL_HW_DIM - 1) / POTAL_HW_DIM,
        r->k_rows, count
    };
    const potal_status checked = potal_index_config_check(&cfg);
    if (checked != POTAL_OK) return checked;
    if (!paired_can_issue(h->core)) return POTAL_ERR_NO_HARDWARE;
    potal_resident_token token;
    const potal_status allocated = e4_token(h, &token);
    if (allocated != POTAL_OK) return allocated;
    memset(&h->index[pair], 0, sizeof(h->index[pair]));
    h->in_call = 1;
    potal_status status = potal_paired_index_config(h->core, &cfg);
    for (uint32_t i = 0; status == POTAL_OK && i < count; ++i) {
        const potal_index_write w = {i, (uint32_t)(originals[i] - r->k_begin)};
        status = potal_paired_index_write(h->core, &w);
    }
    h->in_call = 0;
    if (status != POTAL_OK) return e4_failed(h, status, "e4_index", 0);
    h->index[pair].resident = resident;
    h->index[pair].generation = token.generation;
    h->index[pair].count = count;
    memcpy(h->index[pair].original, originals, count * sizeof(uint64_t));
    h->q.index_generation = token.generation;
    *generation = token.generation;
    return POTAL_OK;
}

potal_status potal_resident_microtile_issue(potal_resident_handle *h,
                                           const potal_resident_microtile *d) {
    const potal_status ready = e4_ready(h, 0);
    if (ready != POTAL_OK) return ready;
    if (!d || d->first > 1 || d->last > 1 || !d->output_id) return POTAL_ERR_INVALID_ARGUMENT;
    const int pair = e4_pair(h, d->resident);
    if (pair < 0 || d->command.pair != (uint32_t)pair) return POTAL_ERR_INVALID_ARGUMENT;
    const potal_paired_loop_ws *p = &d->command;
    const potal_loop_ws *l = &p->loop;
    if (p->path != POTAL_PAIRED_MAIN && p->path != POTAL_PAIRED_RESIDUAL) return POTAL_ERR_INVALID_ARGUMENT;
    potal_gemm_ws_program program;
    const potal_status checked = potal_paired_loop_ws_encode(p, &program);
    if (checked != POTAL_OK) return checked;
    const uint32_t bank = p->path == POTAL_PAIRED_RESIDUAL;
    const potal_resident_load *r = &h->q.pair[pair].description;
    if (d->acc_bank != bank || d->acc_base != 0 || d->b_sp_base != r->b_sp_base ||
        d->b_row_stride != r->b_row_stride) return POTAL_ERR_INVALID_ARGUMENT;
    if (l->i_tiles != 1 || l->j_tiles != 2 || l->pad_i || l->pad_j || l->pad_k ||
        r->j_columns != 32 || r->k_rows % 16 || r->main_a_tiles < 2 ||
        (bank == 0 && (l->k_tiles != 2 || l->passes_per_block != 2)) ||
        (bank == 1 && (l->k_tiles != 1 || l->passes_per_block != 1)))
        return POTAL_ERR_UNSUPPORTED;
    const uint64_t a_span = matrix_extent(16, bank ? 16 : 32, l->lda, 1);
    if (!e4_span(h, bank ? POTAL_RESIDENT_RESIDUAL_A : POTAL_RESIDENT_MAIN_A, l->a, a_span))
        return POTAL_ERR_INVALID_ARGUMENT;
    uint64_t original_block, next_k;
    if (!bank) {
        if (r->k_rows < 32 || d->original_k_begin != r->k_begin + r->k_rows - 32 ||
            d->original_k_begin % 32 || d->index_generation)
            return POTAL_ERR_INVALID_ARGUMENT;
        const int8_t *source = (const int8_t *)h->pool[pair] + (r->k_rows - 32) * r->source_stride;
        if (l->b != source || l->ldb != r->source_stride ||
            !e4_span(h, (uint32_t)pair, l->b, matrix_extent(32, 32, l->ldb, 1)))
            return POTAL_ERR_INVALID_ARGUMENT;
        const uint64_t native = pair_weight_begin((uint32_t)pair) + POTAL_HW_BANK_ROWS - 64;
        if (native != (uint64_t)r->b_sp_base + ((r->k_rows / 16) - 2) * r->b_row_stride)
            return POTAL_ERR_UNSUPPORTED;
        original_block = d->original_k_begin / 32;
        next_k = d->original_k_begin + 32;
    } else {
        const e4_index *ix = &h->index[pair];
        if (!e4_same(ix->resident, d->resident) || ix->generation != d->index_generation ||
            !d->index_generation || ix->count != 16 || d->original_k_begin != ix->original[0] ||
            p->a_offset_tiles < r->main_a_tiles) return POTAL_ERR_INVALID_ARGUMENT;
        original_block = ix->original[0] / 32;
        for (uint32_t i = 0; i < ix->count; ++i)
            if (ix->original[i] / 32 != original_block) return POTAL_ERR_UNSUPPORTED;
        next_k = ix->original[15] + 1;
    }
    for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t slot = l->scale_base + j;
        if (!e4_same(h->scale[slot].resident, d->resident) ||
            h->scale[slot].block != original_block || h->scale[slot].column != r->j_begin + j * 16 ||
            h->core->slot_generation[slot] != l->scale_generation ||
            h->core->slot_columns[slot] != UINT64_C(0xffff)) return POTAL_ERR_INVALID_ARGUMENT;
    }
    potal_resident_acc_info *a = &h->q.acc[bank];
    if (d->first) {
        if (a->initialized && !a->complete) return POTAL_ERR_BUSY;
        if (d->sequence || d->epoch.owner || d->epoch.generation || l->flags)
            return POTAL_ERR_INVALID_ARGUMENT;
    } else {
        if (!a->initialized || a->complete || !e4_same(a->token, d->epoch) ||
            a->output_id != d->output_id || d->sequence != a->next_sequence ||
            h->acc_j_begin[bank] != r->j_begin || h->acc_weight_id[bank] != r->weight_id ||
            h->acc_weight_generation[bank] != r->weight_generation ||
            l->flags != POTAL_GEMM_ACCUMULATE ||
            (!bank && d->original_k_begin != a->next_original_k) ||
            (bank && d->original_k_begin < a->next_original_k)) return POTAL_ERR_INVALID_ARGUMENT;
    }
    uint64_t next_sequence;
    const potal_status next = potal_resident_generation_next(d->sequence, &next_sequence);
    if (next != POTAL_OK) return next;
    if (!paired_can_issue(h->core)) return POTAL_ERR_NO_HARDWARE;
    if (d->first) {
        potal_resident_token epoch;
        const potal_status allocated = e4_token(h, &epoch);
        if (allocated != POTAL_OK) return allocated;
        memset(a, 0, sizeof(*a));
        a->token = epoch;
        a->output_id = d->output_id;
        h->acc_j_begin[bank] = r->j_begin;
        h->acc_weight_id[bank] = r->weight_id;
        h->acc_weight_generation[bank] = r->weight_generation;
    }
    a->pending = 1;
    a->readback_ready = 0;
    h->q.pair[pair].consumers = 1;
    h->q.pending = 1;
    h->kind = E4_COMPUTE;
    h->pending_pair = (uint32_t)pair;
    h->pending_bank = bank;
    h->pending_last = d->last;
    h->pending_next_sequence = next_sequence;
    h->pending_next_k = next_k;
    h->in_call = 1;
    potal_status status;
    if (!bank) {
        program.addrs_ab.rs2 = 0; /* Proven native suffix, not a fabricated B pointer. */
        status = paired_issue_program(h->core, &program, "e4_main_resident");
        if (status == POTAL_OK) {
            h->core->pair[pair].outstanding++;
            h->core->outstanding_loops++;
        }
    } else {
        status = potal_paired_loop_ws_issue(h->core, p);
    }
    h->in_call = 0;
    if (status != POTAL_OK) return e4_failed(h, status, "e4_compute", 0);
    h->issue_complete = 1;
    return POTAL_OK;
}

potal_status potal_resident_readback_issue(potal_resident_handle *h,
                                          uint32_t bank, potal_resident_token epoch) {
    const potal_status ready = e4_ready(h, 0);
    if (ready != POTAL_OK) return ready;
    if (bank > 1 || !e4_same(h->q.acc[bank].token, epoch) || !h->q.acc[bank].initialized)
        return POTAL_ERR_INVALID_ARGUMENT;
    if (!paired_can_issue(h->core)) return POTAL_ERR_NO_HARDWARE;
    const potal_acc_readback r = {bank, 1, 2, 0, 0, h->pool[4 + bank], 32};
    h->kind = E4_READBACK;
    h->pending_bank = bank;
    h->q.pending = 1;
    h->q.acc[bank].pending = 1;
    h->q.acc[bank].readback_ready = 0;
    h->in_call = 1;
    const potal_status status = potal_paired_acc_readback_issue(h->core, &r);
    h->in_call = 0;
    if (status != POTAL_OK) return e4_failed(h, status, "e4_readback", 0);
    h->issue_complete = 1;
    return POTAL_OK;
}

potal_status potal_resident_result(potal_resident_handle *h, uint32_t bank,
                                   potal_resident_token epoch,
                                   const int32_t **data, size_t *stride) {
    const potal_status ready = e4_ready(h, 0);
    if (ready != POTAL_OK) return ready;
    if (bank > 1 || !data || !stride || !e4_same(h->q.acc[bank].token, epoch) ||
        !h->q.acc[bank].readback_ready) return POTAL_ERR_INVALID_ARGUMENT;
    *data = h->pool[4 + bank];
    *stride = 32;
    return POTAL_OK;
}

potal_status potal_resident_fence(potal_resident_handle *h) {
    if (!h) return POTAL_ERR_INVALID_ARGUMENT;
    if (h->in_call) return POTAL_ERR_BUSY;
    if (h->q.state == POTAL_RESIDENT_FAILED) return potal_resident_cleanup(h);
    if (h->q.state == POTAL_RESIDENT_QUARANTINED || h->q.state == POTAL_RESIDENT_CLEANED)
        return h->q.error;
    if (h->q.state == POTAL_RESIDENT_NEW && !h->q.control_pending) return POTAL_ERR_BUSY;
    if (!paired_can_issue(h->core)) return POTAL_ERR_NO_HARDWARE;
    h->in_call = 1;
    const potal_status status = potal_paired_fence(h->core);
    h->in_call = 0;
    h->q.last_drain = status;
    if (status != POTAL_OK) {
        h->q.unknown_fence = 1;
        return e4_failed(h, status, "fence", 1);
    }
    e4_complete(h, 1);
    h->q.state = POTAL_RESIDENT_READY;
    return POTAL_OK;
}

potal_status potal_resident_retire(potal_resident_handle *h,
                                  potal_resident_token resident) {
    const potal_status ready = e4_ready(h, 0);
    if (ready != POTAL_OK) return ready;
    const int pair = e4_pair(h, resident);
    if (pair < 0) return POTAL_ERR_INVALID_ARGUMENT;
    for (uint32_t slot = 0; slot < POTAL_SCALE_ENTRIES; ++slot)
        if (e4_same(h->scale[slot].resident, resident) && h->core->slot_columns[slot])
            return POTAL_ERR_BUSY;
    h->q.pair[pair].readable = 0;
    h->core->pair[pair].resident_b = 0;
    h->core->pair[pair].index_valid = 0;
    memset(&h->index[pair], 0, sizeof(h->index[pair]));
    /* Completed ACC values have their own output epoch, not a B lifetime. */
    return POTAL_OK;
}

potal_status potal_resident_cleanup(potal_resident_handle *h) {
    if (!h) return POTAL_ERR_INVALID_ARGUMENT;
    if (h->in_call) return POTAL_ERR_BUSY;
    if (h->q.state == POTAL_RESIDENT_CLEANED) return h->q.cleanup_status;
    if (h->q.state == POTAL_RESIDENT_QUARANTINED || h->q.unknown_fence) return h->q.error;
    if (h->q.state != POTAL_RESIDENT_FAILED || h->q.cleanup_attempted) return POTAL_ERR_BUSY;
    if (!paired_can_issue(h->core)) return POTAL_ERR_NO_HARDWARE;
    h->q.cleanup_attempted = 1;
    h->in_call = 1;
    potal_status status = potal_paired_fence(h->core); /* Even when core counters are zero. */
    h->q.last_drain = status;
    if (status != POTAL_OK) {
        h->in_call = 0;
        h->q.unknown_fence = 1;
        h->q.cleanup_status = status;
        return e4_failed(h, status, "fence", 1);
    }
    e4_complete(h, 0);
    memset(h->q.pair, 0, sizeof(h->q.pair));
    memset(h->q.acc, 0, sizeof(h->q.acc));
    memset(h->index, 0, sizeof(h->index));
    memset(h->scale, 0, sizeof(h->scale));
    for (uint32_t p = 0; p < POTAL_PAIRS; ++p) {
        h->core->pair[p].resident_b = 0;
        h->core->pair[p].index_valid = 0;
    }
    status = potal_paired_session_cleanup(h->core); /* Already drained: no second Fence. */
    h->in_call = 0;
    h->q.cleanup_status = status;
    if (status != POTAL_OK) return e4_failed(h, status, "e4_cleanup", 1);
    h->q.state = POTAL_RESIDENT_CLEANED;
    return POTAL_OK;
}

potal_status potal_resident_close(potal_resident_handle **handle) {
    if (!handle || !*handle) return POTAL_ERR_INVALID_ARGUMENT;
    potal_resident_handle *h = *handle;
    if (h->in_call) return POTAL_ERR_BUSY;
    if (h->q.state == POTAL_RESIDENT_QUARANTINED || h->q.state == POTAL_RESIDENT_FAILED)
        return h->q.error;
    if (h->q.pending || h->q.control_pending || h->core->outstanding_loops ||
        h->core->outstanding_readbacks || h->core->outstanding_acc_updates || h->core->index_open)
        return POTAL_ERR_BUSY;
    for (uint32_t s = 0; s < POTAL_SCALE_ENTRIES; ++s)
        if (h->core->slot_columns[s]) return POTAL_ERR_BUSY;
    for (uint32_t p = 0; p < POTAL_PAIRS; ++p)
        if (h->q.pair[p].loading || h->q.pair[p].readable || h->q.pair[p].consumers)
            return POTAL_ERR_BUSY;
    const potal_status status = potal_paired_session_close(h->core);
    *handle = NULL;
    e4_free(h);
    return status;
}

potal_resident_status potal_resident_query(const potal_resident_handle *h) {
    potal_resident_status q;
    memset(&q, 0, sizeof(q));
    if (h) {
        q = h->q;
        q.core = potal_paired_session_query(h->core);
    }
    return q;
}
