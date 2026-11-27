#ifndef POTAL_ISSUER_PROFILE_H
#define POTAL_ISSUER_PROFILE_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Diagnostic counters, not an admission-ready/valid or accelerator-busy trace.
 * Each sampled API/request is partitioned into local software, raw round trips,
 * and explicit CPU fences. Nested scope totals are inclusive: do not add them.
 * Raw round trips include dispatch/response transport. Counter-read overhead
 * is retained and reported separately.
 */
typedef enum {
    POTAL_ISSUER_SCALE_WRITE,
    POTAL_ISSUER_SCALE_RELEASE,
    POTAL_ISSUER_SCALE_WINDOW,
    POTAL_ISSUER_RELEASE_WINDOW,
    POTAL_ISSUER_INDEX_CONFIG,
    POTAL_ISSUER_INDEX_WRITE,
    POTAL_ISSUER_INDEX_WINDOW,
    POTAL_ISSUER_LOOP,
    POTAL_ISSUER_READBACK,
    POTAL_ISSUER_DRAIN,
    POTAL_ISSUER_REQUEST,
    POTAL_ISSUER_MAIN_OP,
    POTAL_ISSUER_RESIDUAL_OP,
    POTAL_ISSUER_SCOPE_COUNT
} potal_issuer_scope_kind;

typedef struct {
    uint64_t ticks;
    uint64_t instructions;
} potal_issuer_stamp;

typedef struct {
    uint64_t calls, samples, ticks, instructions;
    uint64_t pre_ticks, between_ticks, post_ticks, raw_ticks, fence_ticks;
    uint64_t pre_instructions, between_instructions, post_instructions, raw_instructions, fence_instructions;
} potal_issuer_scope_stats;

typedef struct {
    uint64_t samples, ticks, instructions, errors;
} potal_issuer_segment_stats;

typedef struct {
    potal_issuer_scope_stats scope[POTAL_ISSUER_SCOPE_COUNT];
    potal_issuer_segment_stats raw[128];
    potal_issuer_segment_stats cpu_fence[2]; /* 0: input publication, 1: completion ordering */
    uint64_t empty_ticks, empty_instructions;
    uint32_t sample_log2, instructions_valid;
    const char * tick_source;
} potal_issuer_profile_stats;

typedef struct potal_issuer_probe {
    potal_issuer_stamp start, last, segment;
    uint64_t pre_ticks, between_ticks, raw_ticks, fence_ticks;
    uint64_t pre_instructions, between_instructions, raw_instructions, fence_instructions;
    uint32_t entered, sampled, kind, segments, raw_depth;
    struct potal_issuer_probe * parent;
} potal_issuer_probe;

#if defined(POTAL_ISSUER_DIAGNOSTICS)
void potal_issuer_profile_enter(potal_issuer_probe * probe, potal_issuer_scope_kind kind);
void potal_issuer_profile_leave(potal_issuer_probe * probe);
void potal_issuer_profile_raw_begin(uint32_t funct);
void potal_issuer_profile_raw_end(uint32_t funct, uint32_t status);
void potal_issuer_profile_fence_begin(uint32_t completion);
void potal_issuer_profile_fence_end(uint32_t completion);
/* Read/dump on the issuer thread after its task's command sequence has ended.
 * Flush resets report totals, but preserves sampling sequence and calibration.
 */
void potal_issuer_profile_snapshot(potal_issuer_profile_stats * out);
void potal_issuer_profile_flush(FILE * out);
#else
static inline void potal_issuer_profile_flush(FILE * out) { (void)out; }
#endif

#ifdef __cplusplus
}
#endif

#if defined(POTAL_ISSUER_DIAGNOSTICS)
#if defined(__cplusplus)
struct potal_issuer_scope {
    potal_issuer_probe probe{};
    explicit potal_issuer_scope(potal_issuer_scope_kind kind) { potal_issuer_profile_enter(&probe, kind); }
    ~potal_issuer_scope() { potal_issuer_profile_leave(&probe); }
    potal_issuer_scope(const potal_issuer_scope &) = delete;
    potal_issuer_scope & operator=(const potal_issuer_scope &) = delete;
};
#define POTAL_ISSUER_SCOPE(kind) potal_issuer_scope potal_profile_scope_(kind)
#else
#define POTAL_ISSUER_SCOPE(kind) \
    potal_issuer_probe potal_profile_scope_ __attribute__((cleanup(potal_issuer_profile_leave))) = {0}; \
    potal_issuer_profile_enter(&potal_profile_scope_, (kind))
#endif
#define POTAL_ISSUER_RAW_BEGIN(funct) potal_issuer_profile_raw_begin(funct)
#define POTAL_ISSUER_RAW_END(funct, status) potal_issuer_profile_raw_end((funct), (uint32_t)(status))
#define POTAL_ISSUER_FENCE_BEGIN(completion) potal_issuer_profile_fence_begin(completion)
#define POTAL_ISSUER_FENCE_END(completion) potal_issuer_profile_fence_end(completion)
#else
#define POTAL_ISSUER_SCOPE(kind) ((void)0)
#define POTAL_ISSUER_RAW_BEGIN(funct) ((void)0)
#define POTAL_ISSUER_RAW_END(funct, status) ((void)0)
#define POTAL_ISSUER_FENCE_BEGIN(completion) ((void)0)
#define POTAL_ISSUER_FENCE_END(completion) ((void)0)
#endif
#endif
