#include "issuer-profile.h"

#if defined(POTAL_ISSUER_DIAGNOSTICS)
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local int initialized, enabled;
static _Thread_local uint32_t depth;
static _Thread_local uint64_t sequence[POTAL_ISSUER_SCOPE_COUNT];
static _Thread_local potal_issuer_probe * active;
static _Thread_local potal_issuer_profile_stats totals;

static potal_issuer_stamp stamp(void) {
    potal_issuer_stamp s = {0, 0};
#if defined(__riscv) && __riscv_xlen == 64
    __asm__ volatile("rdcycle %0\n\trdinstret %1" : "=r"(s.ticks), "=r"(s.instructions) :: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    uint32_t lo, hi;
    __asm__ volatile("lfence\n\trdtsc" : "=a"(lo), "=d"(hi) :: "memory");
    s.ticks = ((uint64_t)hi << 32) | lo;
#elif defined(__aarch64__)
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(s.ticks) :: "memory");
#else
#error "Issuer diagnostics require RV64 counters, x86 TSC, or AArch64 CNTVCT"
#endif
    return s;
}

static void init(void) {
    if (initialized) return;
    initialized = 1;
    const char * option = getenv("POTAL_ISSUER_CPU_PROFILE");
    enabled = option && strcmp(option, "1") == 0;
    totals.sample_log2 = 10;
    const char * text = getenv("POTAL_ISSUER_CPU_SAMPLE_LOG2");
    if (text) {
        char * end = NULL;
        const unsigned long value = strtoul(text, &end, 10);
        if (*text == '\0' || *end != '\0' || value > 20) {
            enabled = 0;
            return;
        }
        totals.sample_log2 = (uint32_t)value;
    }
#if defined(__riscv) && __riscv_xlen == 64
    totals.tick_source = "riscv_rdcycle";
    totals.instructions_valid = 1;
#elif defined(__aarch64__)
    totals.tick_source = "aarch64_cntvct";
#else
    totals.tick_source = "x86_tsc";
#endif
    if (!enabled) return;
    totals.empty_ticks = UINT64_MAX;
    for (unsigned i = 0; i < 32; ++i) {
        const potal_issuer_stamp a = stamp(), b = stamp();
        if (b.ticks - a.ticks < totals.empty_ticks) {
            totals.empty_ticks = b.ticks - a.ticks;
            totals.empty_instructions = b.instructions - a.instructions;
        }
    }
}

void potal_issuer_profile_enter(potal_issuer_probe * p, potal_issuer_scope_kind kind) {
    init();
    if (!enabled) return;
    p->entered = 1;
    ++depth;
    p->kind = (uint32_t)kind;
    ++totals.scope[kind].calls;
    const uint64_t mask = ((uint64_t)1 << totals.sample_log2) - 1;
    if ((sequence[kind]++ & mask) != 0) return;
    p->sampled = 1;
    p->start = p->last = stamp();
    p->parent = active;
    active = p;
}

static void gap(potal_issuer_probe * p, potal_issuer_stamp now) {
    if (p->segments) {
        p->between_ticks += now.ticks - p->last.ticks;
        p->between_instructions += now.instructions - p->last.instructions;
    } else {
        p->pre_ticks += now.ticks - p->last.ticks;
        p->pre_instructions += now.instructions - p->last.instructions;
    }
    p->segment = now;
}

void potal_issuer_profile_leave(potal_issuer_probe * p) {
    if (!p->entered) return;
    --depth;
    if (!p->sampled) return;
    const potal_issuer_stamp end = stamp();
    potal_issuer_scope_stats * s = &totals.scope[p->kind];
    ++s->samples;
    s->ticks += end.ticks - p->start.ticks;
    s->instructions += end.instructions - p->start.instructions;
    s->pre_ticks += p->pre_ticks;
    s->between_ticks += p->between_ticks;
    s->pre_instructions += p->pre_instructions;
    s->between_instructions += p->between_instructions;
    if (p->segments) {
        s->post_ticks += end.ticks - p->last.ticks;
        s->post_instructions += end.instructions - p->last.instructions;
    } else {
        s->pre_ticks += end.ticks - p->last.ticks;
        s->pre_instructions += end.instructions - p->last.instructions;
    }
    s->raw_ticks += p->raw_ticks;
    s->fence_ticks += p->fence_ticks;
    s->raw_instructions += p->raw_instructions;
    s->fence_instructions += p->fence_instructions;
    active = p->parent;
}

void potal_issuer_profile_raw_begin(uint32_t funct) {
    (void)funct;
    if (!active) return;
    const potal_issuer_stamp now = stamp();
    for (potal_issuer_probe * p = active; p; p = p->parent)
        if (p->raw_depth++ == 0) gap(p, now);
}

void potal_issuer_profile_raw_end(uint32_t funct, uint32_t status) {
    if (!active) return;
    const potal_issuer_stamp end = stamp();
    int recorded = 0;
    for (potal_issuer_probe * p = active; p; p = p->parent) {
        if (--p->raw_depth != 0) continue;
        if (!recorded) {
            potal_issuer_segment_stats * s = &totals.raw[funct];
            ++s->samples;
            s->ticks += end.ticks - p->segment.ticks;
            s->instructions += end.instructions - p->segment.instructions;
            s->errors += status != 0;
            recorded = 1;
        }
        p->raw_ticks += end.ticks - p->segment.ticks;
        p->raw_instructions += end.instructions - p->segment.instructions;
        p->last = end;
        ++p->segments;
    }
}

void potal_issuer_profile_fence_begin(uint32_t completion) {
    (void)completion;
    if (!active) return;
    const potal_issuer_stamp now = stamp();
    for (potal_issuer_probe * p = active; p; p = p->parent) gap(p, now);
}

void potal_issuer_profile_fence_end(uint32_t completion) {
    if (!active) return;
    const potal_issuer_stamp end = stamp();
    potal_issuer_segment_stats * s = &totals.cpu_fence[completion];
    ++s->samples;
    s->ticks += end.ticks - active->segment.ticks;
    s->instructions += end.instructions - active->segment.instructions;
    for (potal_issuer_probe * p = active; p; p = p->parent) {
        p->fence_ticks += end.ticks - p->segment.ticks;
        p->fence_instructions += end.instructions - p->segment.instructions;
        p->last = end;
        ++p->segments;
    }
}

void potal_issuer_profile_snapshot(potal_issuer_profile_stats * out) {
    init();
    *out = totals;
}

void potal_issuer_profile_flush(FILE * out) {
    if (!initialized || !enabled || depth) return;
    uint64_t samples = 0;
    for (unsigned i = 0; i < POTAL_ISSUER_SCOPE_COUNT; ++i) samples += totals.scope[i].samples;
    if (!samples) return;
    static const char * names[POTAL_ISSUER_SCOPE_COUNT] = {
        "scale_write", "scale_release", "scale_window", "release_window",
        "index_config", "index_write", "index_window", "loop", "readback", "drain",
        "request", "main_op", "residual_op"
    };
    fprintf(out, "[POTAL_ISSUER_CPU] source=%s instructions_valid=%u sample_log2=%u "
        "sampling=api_hierarchy scope_totals_inclusive=1 empty_ticks=%" PRIu64 " empty_instructions=%" PRIu64
        " overhead_subtracted=0 raw_is_admission_only=0\n",
        totals.tick_source, totals.instructions_valid, totals.sample_log2,
        totals.empty_ticks, totals.empty_instructions);
    const char * timing = getenv("POTAL_PAIRED_TIMING");
    const char * timeline = getenv("POTAL_TIMELINE");
    fprintf(out, "[POTAL_ISSUER_CPU] legacy_timing_enabled=%u legacy_timeline_enabled=%u\n",
            !(timing && strcmp(timing, "0") == 0), timeline && *timeline ? 1u : 0u);
    for (unsigned i = 0; i < POTAL_ISSUER_SCOPE_COUNT; ++i) {
        const potal_issuer_scope_stats * s = &totals.scope[i];
        if (!s->calls) continue;
        fprintf(out, "[POTAL_ISSUER_CPU] api=%s calls=%" PRIu64 " samples=%" PRIu64
            " ticks=%" PRIu64 " instructions=%" PRIu64 " pre_ticks=%" PRIu64
            " between_ticks=%" PRIu64 " post_ticks=%" PRIu64 " raw_ticks=%" PRIu64
            " cpu_fence_ticks=%" PRIu64 " pre_instructions=%" PRIu64
            " between_instructions=%" PRIu64 " post_instructions=%" PRIu64
            " raw_instructions=%" PRIu64 " cpu_fence_instructions=%" PRIu64 "\n",
            names[i], s->calls, s->samples, s->ticks,
            s->instructions, s->pre_ticks, s->between_ticks, s->post_ticks, s->raw_ticks, s->fence_ticks,
            s->pre_instructions, s->between_instructions, s->post_instructions,
            s->raw_instructions, s->fence_instructions);
    }
    for (unsigned i = 0; i < 128; ++i) {
        const potal_issuer_segment_stats * s = &totals.raw[i];
        if (!s->samples) continue;
        fprintf(out, "[POTAL_ISSUER_CPU] funct=%u samples=%" PRIu64 " ticks=%" PRIu64
            " instructions=%" PRIu64 " errors=%" PRIu64 "\n",
            i, s->samples, s->ticks, s->instructions, s->errors);
    }
    for (unsigned i = 0; i < 2; ++i) {
        const potal_issuer_segment_stats * s = &totals.cpu_fence[i];
        if (!s->samples) continue;
        fprintf(out, "[POTAL_ISSUER_CPU] cpu_fence=%s samples=%" PRIu64
            " ticks=%" PRIu64 " instructions=%" PRIu64 "\n",
            i ? "completion" : "publication", s->samples, s->ticks, s->instructions);
    }
    memset(totals.scope, 0, sizeof(totals.scope));
    memset(totals.raw, 0, sizeof(totals.raw));
    memset(totals.cpu_fence, 0, sizeof(totals.cpu_fence));
}
#endif
