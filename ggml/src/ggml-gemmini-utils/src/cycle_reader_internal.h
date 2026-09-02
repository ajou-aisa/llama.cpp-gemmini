#pragma once

#include <stdint.h>

#ifdef __cplusplus
#define GEMMINI_INTERNAL_NOEXCEPT noexcept
extern "C" {
#else
#define GEMMINI_INTERNAL_NOEXCEPT
#endif

enum gemmini_native_cycle_source_internal {
    GEMMINI_NATIVE_CYCLE_SOURCE_NONE = 0,
    GEMMINI_NATIVE_CYCLE_SOURCE_RISCV_CYCLE = 1,
    GEMMINI_NATIVE_CYCLE_SOURCE_APPLE_HOST_TICK = 2,
    GEMMINI_NATIVE_CYCLE_SOURCE_LINUX_PERF_CPU_CYCLES = 3,
    GEMMINI_NATIVE_CYCLE_SOURCE_STEADY_CLOCK = 4,
};

enum gemmini_native_cycle_reason_internal {
    GEMMINI_NATIVE_CYCLE_REASON_NONE = 0,
    GEMMINI_NATIVE_CYCLE_REASON_UNAVAILABLE_EVENT = 1,
    GEMMINI_NATIVE_CYCLE_REASON_UNAVAILABLE_DIRECT_MAPPING = 2,
    GEMMINI_NATIVE_CYCLE_REASON_MULTIPLEXED = 3,
    GEMMINI_NATIVE_CYCLE_REASON_SEQLOCK_EXHAUSTED = 4,
    GEMMINI_NATIVE_CYCLE_REASON_INVALID_START = 5,
    GEMMINI_NATIVE_CYCLE_REASON_INVALID_END = 6,
    GEMMINI_NATIVE_CYCLE_REASON_SOURCE_MISMATCH = 7,
    GEMMINI_NATIVE_CYCLE_REASON_EVENT_OWNER_MISMATCH = 8,
    GEMMINI_NATIVE_CYCLE_REASON_EVENT_GENERATION_MISMATCH = 9,
    GEMMINI_NATIVE_CYCLE_REASON_STRUCTURALLY_CROSS_TASK = 10,
    GEMMINI_NATIVE_CYCLE_REASON_COUNTER_REGRESSION = 11,
    GEMMINI_NATIVE_CYCLE_REASON_SCALAR_PROVENANCE_UNAVAILABLE = 12,
};

typedef struct gemmini_native_cycle_sample_internal {
    uint64_t value;
    uint8_t valid;
    uint8_t reason;
    uint8_t source;
    uint64_t owner_event_token;
    uint64_t generation;
} gemmini_native_cycle_sample_internal;

struct gemmini_cycle_record_v2;

uint8_t gemmini_log_cycle_record_v2_checked_internal(
    const struct gemmini_cycle_record_v2 * record,
    const gemmini_native_cycle_sample_internal * start,
    const gemmini_native_cycle_sample_internal * end,
    int structurally_same_owner_eligible) GEMMINI_INTERNAL_NOEXCEPT;

#ifdef __cplusplus
}

#include <string>
namespace ggml::gemmini::log {
struct CycleRecord;
std::string serialize_checked_cycle_record(const CycleRecord & record, bool valid,
                                           const char * reason);
}
#endif
#undef GEMMINI_INTERNAL_NOEXCEPT
