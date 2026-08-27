#pragma once

#include "direct-types.hpp"
#include "../rmd/rmd-types.hpp"

#include <cstdint>
#include <optional>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::residual {

struct DirectCpuDetailMetrics {
    std::optional<uint64_t> serial_pre_cycles;
    std::optional<uint64_t> tile_cycles;
    std::optional<uint64_t> serial_post_cycles;
    std::optional<uint64_t> total_cycles;
    const char * coverage = "algorithm_cpu_leaves";
    bool valid = false;
};

#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING)
namespace testing {

enum class DirectCpuSamplePoint : uint8_t {
    serial_pre_start,
    serial_pre_end,
    tile_start,
    tile_end,
    serial_post_start,
    serial_post_end,
};

struct DirectCpuSample {
    uint64_t value = 0;
    bool valid = false;
    uint64_t owner = 0;
    uint64_t generation = 0;
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
    size_t event_count = 0;
    size_t call_count = 0;
    size_t native_q8_values = 0;
    size_t j_tile_count = 0;
    std::optional<DirectCpuDetailMetrics> cpu_detail;
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
