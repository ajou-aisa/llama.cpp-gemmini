#pragma once

#include "direct-types.hpp"
#include "../rmd/rmd-types.hpp"

#include <cstdint>
#include <optional>
#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::residual {

enum class DirectCpuTileReason : uint8_t {
    none,
    unavailable_event,
    unavailable_direct_mapping,
    multiplexed,
    seqlock_exhausted,
    invalid_start,
    invalid_end,
    source_mismatch,
    event_owner_mismatch,
    event_generation_mismatch,
    structurally_cross_task,
    counter_regression,
};

enum class DirectCpuTileSource : uint8_t {
    perf_cpu_cycles,
};

struct DirectCpuTileRecord {
    uint64_t run_id = 0;
    size_t stripe_id = 0;
    size_t worker_id = 0;
    size_t tile_index = 0;
    size_t j_begin = 0;
    size_t j_end = 0;
    uint64_t start_cycle = 0;
    uint64_t end_cycle = 0;
    std::optional<uint64_t> delta_cycles;
    bool valid = false;
    DirectCpuTileReason reason = DirectCpuTileReason::invalid_start;
    DirectCpuTileReason sample_reason = DirectCpuTileReason::none;
    DirectCpuTileSource source = DirectCpuTileSource::perf_cpu_cycles;
    uint64_t owner_event_token = 0;
    uint64_t generation = 0;
};

#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING)
namespace testing {

enum class DirectCpuSamplePoint : uint8_t {
    tile_start,
    tile_end,
};

struct DirectCpuSample {
    uint64_t value = 0;
    bool valid = false;
    uint64_t owner = 0;
    uint64_t generation = 0;
    DirectCpuTileReason reason = DirectCpuTileReason::unavailable_event;
    DirectCpuTileSource source = DirectCpuTileSource::perf_cpu_cycles;
};

using DirectCpuSampleReader = DirectCpuSample (*)(
    DirectCpuSamplePoint point, size_t tile_index, void * context);

struct DirectExecutionTestHooks {
    DirectCpuSampleReader sample_reader = nullptr;
    void * context = nullptr;
};

}
#endif

struct DirectExecutionMetrics {
    uint64_t run_id = 0;
    size_t event_count = 0;
    size_t call_count = 0;
    size_t native_q8_values = 0;
    size_t j_tile_count = 0;
    std::vector<DirectCpuTileRecord> cpu_tiles;
};

// Computes one immutable direct payload into a staged row-major [row_count, J]
// correction. H1/HP1 return integer-block-scaled values; H0 returns values with
// floating block scales already applied in double. The caller's output is
// replaced only after complete success.
rmd::RmdStatus execute_direct_stripe(const ggml_gemmini_args_t & args,
                                     const DirectStripePayload & payload,
                                     rmd::DirectOutput & correction,
                                     DirectExecutionMetrics * metrics = nullptr);

#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING)
rmd::RmdStatus execute_direct_stripe(
    const ggml_gemmini_args_t & args,
    const DirectStripePayload & payload,
    rmd::DirectOutput & correction,
    DirectExecutionMetrics * metrics,
    const testing::DirectExecutionTestHooks & hooks);
#endif

}
