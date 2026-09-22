#pragma once

#include "rmd-executor.hpp"

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) ||                        \
    defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
#include <im2p_sim.h>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gemmini/optrace.hpp>
#if CYCLE_SIM
#include <gemmini/cycle_sim_log.hpp>
#endif

struct im2p_sim;
typedef struct im2p_sim im2p_sim_t;

namespace ggml::gemmini::rmd {

struct RunAwareRequest;
struct Im2pFullExecutor;
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) ||                        \
    defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
struct Im2pFullExecutor {
  void *context = nullptr;
    // Synchronous borrowed descriptor. Success requires every checked output
    // callback and device completion. A failure is never retried in software.
    int (*execute)(void *, const im2p_matmul_desc_t *,
                   im2p_work_stats_extended_t *) = nullptr;
  // Production HP1 requires this callback. No fallback to raw execute.
  int (*execute_planned)(void *, const im2p_matmul_desc_t *,
                         const im2p_production_geometry_v1_t *,
                         im2p_work_stats_extended_t *) = nullptr;
  // Packet-level HP1 capability. Run metadata and every descriptor buffer are
  // borrowed synchronously; there is no fallback to block-local execution.
  int (*execute_planned_runs)(void *, const im2p_matmul_desc_t *,
                              const im2p_production_geometry_v1_t *,
                              const im2p_compact_runs_t *,
                              im2p_work_stats_extended_t *) = nullptr;
};
#endif

enum class Im2pProviderTestFault : uint8_t {
    none,
    read_failure,
    write_failure,
    watchdog,
    duplicate_output,
    missing_output,
    output_index,
    stats_overflow,
    k_accumulation_overflow,
    block_scale_overflow,
    cancel_after_first_dot,
};

// Executes a validated compact packet through the typed IM2P provider ABI. The
// caller owns `sim`; output and metrics are committed only after every dot and
// output callback has completed successfully.
RmdStatus execute_rmd_stripe_im2p(im2p_sim_t *sim,
                                  const ggml_gemmini_args_t &args,
                                  const StripePacket &packet,
                                  CompressedOutput &output,
                                  RmdExecutionMetrics *metrics = nullptr,
                                  const Im2pFullExecutor *executor = nullptr);

RmdStatus execute_rmd_stripe_im2p(im2p_sim_t *sim,
                                  const ggml_gemmini_args_t &args,
                                  const StripePacket &packet,
                                  Correction &correction,
                                  RmdExecutionMetrics *metrics = nullptr,
                                  const Im2pFullExecutor *executor = nullptr);

#if defined(GGML_GEMMINI_TESTING)
RmdStatus execute_rmd_stripe_im2p_for_test(im2p_sim_t *sim,
                                           const ggml_gemmini_args_t &args,
                                           const StripePacket &packet,
                                           CompressedOutput &output,
                                           RmdExecutionMetrics *metrics,
    Im2pProviderTestFault fault);
RmdStatus execute_rmd_stripe_im2p_for_test(im2p_sim_t *sim,
                                           const ggml_gemmini_args_t &args,
                                           const StripePacket &packet,
                                           Correction &correction,
                                           RmdExecutionMetrics *metrics,
    Im2pProviderTestFault fault);
RmdStatus execute_rmd_stripe_im2p_missing_runs_for_test(
    const ggml_gemmini_args_t &args, const StripePacket &packet,
    Correction &correction, RmdExecutionMetrics *metrics);
void reset_im2p_provider_dot_attempts_for_test();
[[nodiscard]] size_t im2p_provider_dot_attempts_for_test();
#endif

namespace detail {
struct Im2pRunAwareWork;

RmdStatus execute_rmd_stripe_im2p_with_weights(
    im2p_sim_t *sim, const ggml_gemmini_args_t &args,
    const StripePacket &packet, Correction &correction,
    RmdWeightPreparation &weights, RmdExecutionMetrics *metrics = nullptr);

struct Im2pProviderStatsAggregate {
    RmdProviderStats stats{};
};

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) ||                        \
    defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
void expand_im2p_provider_stats(
    const RmdProviderStats &source,
                                im2p_work_stats_extended_t &destination) noexcept;
#endif

struct Im2pCompactDot {
    uint8_t operand_bits = 0;
  const void *activations = nullptr;
    size_t rows = 0;
    size_t activation_row_stride_bytes = 0;
  const int32_t *weights = nullptr;
    size_t columns = 0;
    size_t weight_row_stride = 0;
    size_t k = 0;
    // One carrier per output column; nullptr selects the non-HP1 path.
    const uint32_t *hp1_carriers = nullptr;
    gemmini_cycle_record_v2 timing_identity{};
    // Original weight-block identity shared by the provider and trace.
    uint32_t block_id = 0;
    size_t lane_group = 0, k_offset = 0, column_offset = 0;
    std::shared_ptr<const ggml::gemmini::optrace::Context> trace_context{};
    std::string trace_layer{};
    uint64_t source_row_begin = 0, source_row_count = 0;
    uint64_t stripe_id = 0;
#if CYCLE_SIM
    ggml::gemmini::cycle_sim::Context cycle_sim_context{};
    std::vector<uint64_t> required_host_stage_ids{};
#endif
};

RmdStatus execute_im2p_compact_dot(
    im2p_sim_t *sim, const Im2pCompactDot &dot, OutputValue *output,
    size_t output_row_stride, Im2pProviderStatsAggregate &aggregate,
    Im2pProviderTestFault fault = Im2pProviderTestFault::none,
    const Im2pFullExecutor *executor = nullptr);

// Executes one packet-owned cross-block request. `output` and `aggregate` are
// unchanged unless the one-shot M*N callback and all collection obligations
// complete successfully.
RmdStatus execute_im2p_run_aware(
    im2p_sim_t *sim, const Im2pRunAwareWork &work,
    std::vector<OutputValue> &output, Im2pProviderStatsAggregate &aggregate,
    const Im2pFullExecutor *executor = nullptr);

struct Im2pRunAwareWork {
  const RunAwareRequest *request = nullptr;
  gemmini_cycle_record_v2 timing_identity{};
  std::shared_ptr<const ggml::gemmini::optrace::Context> trace_context{};
  std::string trace_layer{};
#if CYCLE_SIM
  ggml::gemmini::cycle_sim::Context cycle_sim_context{};
  std::vector<uint64_t> required_host_stage_ids{};
#endif
};

} // namespace detail
} // namespace ggml::gemmini::rmd
