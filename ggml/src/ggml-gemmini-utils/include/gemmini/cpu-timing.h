#pragma once

#include <stdint.h>
#include "log.h"
#include "trace-context.h"

#ifdef __cplusplus
extern "C" {
#endif

enum gemmini_cpu_counter_source {
    GEMMINI_CPU_COUNTER_UNAVAILABLE = 0,
    GEMMINI_CPU_COUNTER_THREAD_PERF = 1,
};

typedef struct gemmini_cpu_sample {
    uint64_t ns, tid, thread_cpu_ns;
    uint64_t counter, owner_token, generation;
    uint8_t thread_cpu_valid, native_valid, native_reason, native_source;
    gemmini_trace_context trace;
} gemmini_cpu_sample;

typedef struct gemmini_cpu_totals {
    uint64_t interval_count;
    uint64_t cycles, cycles_valid_count;
    uint64_t thread_cpu_ns, thread_cpu_valid_count;
    const char *cycles_reason;
    const char *thread_cpu_reason;
} gemmini_cpu_totals;

gemmini_cpu_sample gemmini_cpu_timing_read(void);
void gemmini_cpu_timing_add(gemmini_cpu_totals *totals,
                           const gemmini_cpu_sample *start, const gemmini_cpu_sample *end);
void gemmini_cpu_timing_merge(gemmini_cpu_totals *totals, const gemmini_cpu_totals *other);
// Preserve sampled endpoints for offline selection; this does not add a second summary interval.
void gemmini_cpu_timing_record(const gemmini_cycle_record_v2 *identity,
                              const gemmini_cpu_sample *start, const gemmini_cpu_sample *end)
#ifdef __cplusplus
    noexcept
#endif
    ;
// Raw operator/task segments do not participate in the legacy aggregate
// cardinality contract; their captured origin is in operator_context.
void gemmini_cpu_timing_record_segment(const gemmini_cycle_record_v2 *identity,
                                      const gemmini_cpu_sample *start,
                                      const gemmini_cpu_sample *end)
#ifdef __cplusplus
    noexcept
#endif
    ;
// Structural task/operator envelopes are explicit; the logger never infers
// them from op-name strings.
void gemmini_cpu_timing_record_envelope(const gemmini_cycle_record_v2 *identity,
                                       const gemmini_cpu_sample *start,
                                       const gemmini_cpu_sample *end)
#ifdef __cplusplus
    noexcept
#endif
    ;
void gemmini_cpu_timing_emit(const char *layer, const char *scope, const uint64_t *run_id,
                            int operation_success, const gemmini_cpu_sample *start,
                            const gemmini_cpu_sample *end, const gemmini_cpu_totals *totals);

#ifdef __cplusplus
}
#endif
