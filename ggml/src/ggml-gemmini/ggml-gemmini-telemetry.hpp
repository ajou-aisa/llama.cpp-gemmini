#pragma once

#include <cstdint>
#include <string>

namespace ggml::gemmini {

inline constexpr const char * kCycleTelemetrySchema = "gemmini.cycle";
inline constexpr std::uint32_t kCycleTelemetryVersion = 2;
#ifdef __riscv
inline constexpr const char * kNativeCycleSource = "riscv_cycle";
inline constexpr const char * kNativeCycleUnit = "cycle";
#else
inline constexpr const char * kNativeCycleSource = "host_tick";
inline constexpr const char * kNativeCycleUnit = "tick";
#endif

struct CycleIntervalTelemetry {
    std::string source = kNativeCycleSource;
    std::string unit = kNativeCycleUnit;
    std::string layer;
    std::string op;
    std::uint64_t start = 0;
    std::uint64_t end = 0;
};

struct WsLoopTelemetry {
    std::uint64_t problem_i = 0;
    std::uint64_t problem_j = 0;
    std::uint64_t problem_k = 0;
    std::uint64_t tile_i = 0;
    std::uint64_t tile_j = 0;
    std::uint64_t tile_k = 0;
    std::uint64_t gemmini_outer_i = 0;
    std::uint64_t gemmini_outer_j = 0;
    std::uint64_t gemmini_outer_k = 0;
    std::uint64_t ws_inner_calls = 0;
    std::uint64_t containing_interval_cycles = 0;
    std::uint32_t load_occupancy_cycles = 0;
    std::uint32_t execute_occupancy_cycles = 0;
    std::uint32_t store_occupancy_cycles = 0;
    std::uint32_t loop_occupancy_cycles = 0;
};

struct Im2pExecutionTelemetry {
    std::string layer;
    std::uint64_t run_id = 0;
    std::string mode;
    std::uint8_t activation_bits = 0;
    std::uint8_t weight_bits = 0;
    std::uint32_t dim = 0;
    std::uint64_t problem_i = 0;
    std::uint64_t problem_j = 0;
    std::uint64_t problem_k = 0;
    std::uint64_t tile_i = 0;
    std::uint64_t tile_j = 0;
    std::uint64_t tile_k = 0;

    // These are direct 64-bit RTL projections. Detail counters can overlap and
    // must not be summed to manufacture a total.
    std::uint64_t rtl_work_total_cycles = 0;
    std::uint64_t rtl_compute_cycles = 0;
    std::uint64_t rtl_drain_cycles = 0;
    std::uint64_t rtl_activation_wait_cycles = 0;
    std::uint64_t rtl_weight_wait_cycles = 0;
    std::uint64_t rtl_scale_wait_cycles = 0;
    std::uint64_t rtl_output_wait_cycles = 0;
    std::uint64_t rtl_overlap_cycles = 0;
    std::uint64_t rtl_activation_overlap_cycles = 0;
    std::uint64_t rtl_weight_overlap_cycles = 0;
    std::uint64_t rtl_scale_overlap_cycles = 0;
    std::uint64_t rtl_completed_output_works = 0;
    std::uint64_t rtl_completed_fragments = 0;
    std::uint64_t rtl_scheduler_groups_completed = 0;
    std::uint64_t rtl_stripes_published = 0;
    std::uint64_t rtl_stripe_rows_published = 0;

    // Additive RMD projection. False preserves the dense record byte-for-byte.
    bool residual_domain = false;
    bool residual_aggregate = false;
    std::uint64_t stripe_id = 0;
    std::uint64_t slot = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end = 0;
    std::uint64_t rmd_dot_calls = 0;
};

struct RmdTelemetryRecord;

struct Im2pStripeTelemetry {
    std::string layer;
    std::uint64_t run_id = 0;
    std::uint64_t stripe_id = 0;
    std::uint64_t slot = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end = 0;
    std::uint64_t publish_cycle = 0;
    std::uint64_t completion_cycle = 0;
};

struct QuantizationStripeTelemetry {
    std::string layer;
    std::uint64_t run_id = 0;
    std::uint64_t stripe_id = 0;
    std::uint64_t slot = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end = 0;
    std::uint64_t start = 0;
    std::uint64_t end = 0;
};

struct PipelineStripeTelemetry {
    std::string layer;
    std::uint64_t run_id = 0;
    std::uint64_t stripe_id = 0;
    std::uint64_t slot = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end = 0;
    std::uint64_t queue_start_ns = 0;
    std::uint64_t queue_end_ns = 0;
    std::uint64_t dense_start_ns = 0;
    std::uint64_t dense_end_ns = 0;
    std::uint64_t rmd_start_ns = 0;
    std::uint64_t rmd_end_ns = 0;
    std::uint64_t compose_start_ns = 0;
    std::uint64_t compose_end_ns = 0;
    std::uint64_t finalize_start_ns = 0;
    std::uint64_t finalize_end_ns = 0;
};

std::string serialize_cycle_telemetry(const CycleIntervalTelemetry & record);
std::string serialize_cycle_telemetry(const WsLoopTelemetry & record);
std::string serialize_cycle_telemetry(const Im2pExecutionTelemetry & record);
std::string serialize_cycle_telemetry(const Im2pStripeTelemetry & record);
std::string serialize_cycle_telemetry(const QuantizationStripeTelemetry & record);
std::string serialize_cycle_telemetry(const PipelineStripeTelemetry & record);
std::string serialize_cycle_telemetry(const RmdTelemetryRecord & record);

void emit_cycle_telemetry(const CycleIntervalTelemetry & record);
void emit_cycle_telemetry(const WsLoopTelemetry & record);
void emit_cycle_telemetry(const Im2pExecutionTelemetry & record);
void emit_cycle_telemetry(const Im2pStripeTelemetry & record);
void emit_cycle_telemetry(const QuantizationStripeTelemetry & record);
void emit_cycle_telemetry(const PipelineStripeTelemetry & record);
void emit_cycle_telemetry(const RmdTelemetryRecord & record);

} // namespace ggml::gemmini
