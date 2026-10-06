#pragma once

#include "ggml-gemmini-im2p.hpp"
#include "matmul/options.hpp"
#include "residual/rmd/rmd-types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ggml::gemmini {

inline constexpr const char *  kCycleTelemetrySchema  = "gemmini.cycle";
inline constexpr std::uint32_t kCycleTelemetryVersion = 2;
#ifdef __riscv
inline constexpr const char * kNativeCycleSource = "riscv_cycle";
inline constexpr const char * kNativeCycleUnit   = "cycle";
#elif defined(__linux__) && defined(__aarch64__)
inline constexpr const char * kNativeCycleSource = "linux_perf_cpu_cycles";
inline constexpr const char * kNativeCycleUnit   = "cycle";
#else
inline constexpr const char * kNativeCycleSource = "host_tick";
inline constexpr const char * kNativeCycleUnit   = "tick";
#endif

struct CycleIntervalTelemetry {
    std::string   source = kNativeCycleSource;
    std::string   unit   = kNativeCycleUnit;
    std::string   layer;
    std::string   op;
    std::uint64_t start = 0;
    std::uint64_t end   = 0;
};

struct WsLoopTelemetry {
    std::uint64_t problem_i                  = 0;
    std::uint64_t problem_j                  = 0;
    std::uint64_t problem_k                  = 0;
    std::uint64_t tile_i                     = 0;
    std::uint64_t tile_j                     = 0;
    std::uint64_t tile_k                     = 0;
    std::uint64_t gemmini_outer_i            = 0;
    std::uint64_t gemmini_outer_j            = 0;
    std::uint64_t gemmini_outer_k            = 0;
    std::uint64_t ws_inner_calls             = 0;
    std::uint64_t containing_interval_cycles = 0;
    std::uint32_t load_occupancy_cycles      = 0;
    std::uint32_t execute_occupancy_cycles   = 0;
    std::uint32_t store_occupancy_cycles     = 0;
    std::uint32_t loop_occupancy_cycles      = 0;
};

struct Im2pExecutionTelemetry {
    std::string   layer;
    std::uint64_t run_id = 0;
    std::string   mode;
    std::uint8_t  activation_bits = 0;
    std::uint8_t  weight_bits     = 0;
    std::uint32_t dim             = 0;
    std::uint64_t problem_i       = 0;
    std::uint64_t problem_j       = 0;
    std::uint64_t problem_k       = 0;
    std::uint64_t tile_i          = 0;
    std::uint64_t tile_j          = 0;
    std::uint64_t tile_k          = 0;

    // These are direct 64-bit RTL projections. Detail counters can overlap and
    // must not be summed to manufacture a total.
    std::uint64_t rtl_work_total_cycles          = 0;
    std::uint64_t rtl_compute_cycles             = 0;
    std::uint64_t rtl_drain_cycles               = 0;
    std::uint64_t rtl_activation_wait_cycles     = 0;
    std::uint64_t rtl_weight_wait_cycles         = 0;
    std::uint64_t rtl_scale_wait_cycles          = 0;
    std::uint64_t rtl_output_wait_cycles         = 0;
    std::uint64_t rtl_overlap_cycles             = 0;
    std::uint64_t rtl_activation_overlap_cycles  = 0;
    std::uint64_t rtl_weight_overlap_cycles      = 0;
    std::uint64_t rtl_scale_overlap_cycles       = 0;
    std::uint64_t rtl_completed_output_works     = 0;
    std::uint64_t rtl_completed_fragments        = 0;
    std::uint64_t rtl_scheduler_groups_completed = 0;
    std::uint64_t rtl_stripes_published          = 0;
    std::uint64_t rtl_stripe_rows_published      = 0;

    // Additive RMD projection. False preserves the dense record byte-for-byte.
    bool          residual_domain    = false;
    bool          residual_aggregate = false;
    std::uint64_t stripe_id          = 0;
    std::uint64_t slot               = 0;
    std::uint64_t row_begin          = 0;
    std::uint64_t row_end            = 0;
    std::uint64_t rmd_dot_calls      = 0;

    std::optional<im2p_adapter::Stats> provider_stats;
    std::string                        backend;
    std::string                        clock_domain;
    std::string                        numerical_contract;
    std::string                        scale_mode;
    std::optional<std::uint8_t>        vector_op;
    std::optional<std::uint8_t>        output_domain;
};

struct RmdTelemetryRecord;
struct RmdTelemetryCheckResult;
struct MatmulCpuSample;
struct MatmulCpuInterval;
namespace residual {
struct DirectStripePayload;
}
namespace rmd {
struct RmdExecutionMetrics;
}

struct RmdStripeTelemetry {
    std::string                  layer;
    std::optional<std::uint64_t> run_id;
    std::uint64_t                stripe_id = 0;
    std::optional<std::uint64_t> slot;
    std::uint64_t                row_begin = 0, row_end = 0;
    std::string                  backend;
    bool                         success = false;
    std::string                  reason;
    // Borrowed only during synchronous serialization; CycleLog retains owned JSON.
    const rmd::RmdExecutionMetrics * metrics = nullptr;
};

struct Im2pStripeTelemetry {
    std::string   layer;
    std::uint64_t run_id           = 0;
    std::uint64_t stripe_id        = 0;
    std::uint64_t slot             = 0;
    std::uint64_t row_begin        = 0;
    std::uint64_t row_end          = 0;
    std::uint64_t publish_cycle    = 0;
    std::uint64_t completion_cycle = 0;
};

struct QuantizationStripeTelemetry {
    std::string   layer;
    std::uint64_t run_id    = 0;
    std::uint64_t stripe_id = 0;
    std::uint64_t slot      = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end   = 0;
    std::uint64_t start_ns  = 0;
    std::uint64_t end_ns    = 0;
};

struct PipelineStripeTelemetry {
    std::string   layer;
    std::uint64_t run_id                     = 0;
    std::uint64_t stripe_id                  = 0;
    std::uint64_t slot                       = 0;
    std::uint64_t row_begin                  = 0;
    std::uint64_t row_end                    = 0;
    std::uint64_t queue_start_ns             = 0;
    std::uint64_t queue_end_ns               = 0;
    std::uint64_t queue_start_tid            = 0;
    std::uint64_t queue_end_tid              = 0;
    std::uint64_t dense_start_ns             = 0;
    std::uint64_t dense_end_ns               = 0;
    std::uint64_t dense_start_tid            = 0;
    std::uint64_t dense_end_tid              = 0;
    std::uint64_t rmd_start_ns               = 0;
    std::uint64_t rmd_end_ns                 = 0;
    std::uint64_t residual_backend_start_ns  = 0;
    std::uint64_t residual_backend_end_ns    = 0;
    std::uint64_t residual_backend_start_tid = 0;
    std::uint64_t residual_backend_end_tid   = 0;
    std::uint64_t compose_start_ns           = 0;
    std::uint64_t compose_end_ns             = 0;
    std::uint64_t compose_start_tid          = 0;
    std::uint64_t compose_end_tid            = 0;
    std::uint64_t finalize_start_ns          = 0;
    std::uint64_t finalize_end_ns            = 0;
    std::uint64_t finalize_start_tid         = 0;
    std::uint64_t finalize_end_tid           = 0;
};

std::string serialize_cycle_telemetry(const CycleIntervalTelemetry & record);
std::string serialize_cycle_telemetry(const WsLoopTelemetry & record);
std::string serialize_cycle_telemetry(const Im2pExecutionTelemetry & record);
std::string serialize_cycle_telemetry(const Im2pStripeTelemetry & record);
std::string serialize_cycle_telemetry(const QuantizationStripeTelemetry & record);
std::string serialize_cycle_telemetry(const PipelineStripeTelemetry & record);
std::string serialize_cycle_telemetry(const RmdTelemetryRecord & record);
std::string serialize_cycle_telemetry(const RmdStripeTelemetry & record);

void emit_cycle_telemetry(const CycleIntervalTelemetry & record);
void emit_cycle_telemetry(const WsLoopTelemetry & record);
void emit_cycle_telemetry(const Im2pExecutionTelemetry & record);
void emit_cycle_telemetry(const Im2pStripeTelemetry & record);
void emit_cycle_telemetry(const QuantizationStripeTelemetry & record);
void emit_cycle_telemetry(const PipelineStripeTelemetry & record);
void emit_cycle_telemetry(const RmdTelemetryRecord & record);
void emit_cycle_telemetry(const RmdStripeTelemetry & record);

struct MatmulJobMetrics;
namespace log {
struct CycleRecord;
}
void project_matmul_cpu_identity(log::CycleRecord &       record,
                                 const MatmulJobMetrics * profile,
                                 std::optional<uint64_t>  invocation_run_id = {});

std::string serialize_matmul_cpu_interval(log::CycleRecord          record,
                                          const MatmulCpuSample &   start,
                                          const MatmulCpuSample &   end,
                                          bool                      operation_success,
                                          const MatmulCpuInterval * explicit_interval = nullptr);

// Stable pure seams: Todo 10 can consume records without parsing log prose.
std::string             serialize_rmd_telemetry(const RmdTelemetryRecord & record);
RmdTelemetryCheckResult check_rmd_telemetry(const RmdTelemetryRecord & record,
                                            std::string_view           expected_units,
                                            bool                       comparison_mode);
RmdTelemetryCheckResult compare_rmd_telemetry_proofs(const RmdTelemetryRecord & lhs,
                                                     const RmdTelemetryRecord & rhs);
std::string             rmd_input_hash(const residual::DirectStripePayload & payload);
std::string             rmd_input_hash(const rmd::StripePacket & packet);
std::string             rmd_correction_hash(const rmd::Correction & correction);
std::string rmd_output_hash(const ggml_gemmini_args_t & args, size_t row_begin, size_t row_end);
std::string resolve_rmd_model_id(const char * environment_model_id, std::string_view model_arch);
RmdTelemetryRecord make_rmd_telemetry_record(RmdBackend                backend,
                                             MatmulOptionSource        source,
                                             std::string               runtime_bundle_id,
                                             std::string               model_id,
                                             std::string               layer,
                                             uint64_t                  run_id,
                                             const MatmulCpuInterval & invocation_total,
                                             const std::vector<MatmulJobMetrics> & profiles);

namespace detail {
PipelineStripeTelemetry pipeline_stripe_telemetry(const char *             layer,
                                                  const MatmulJobMetrics & profile);
}

} // namespace ggml::gemmini
