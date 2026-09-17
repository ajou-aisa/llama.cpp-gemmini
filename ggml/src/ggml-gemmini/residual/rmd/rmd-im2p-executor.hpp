#pragma once

#include "rmd-executor.hpp"

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) || defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
#include <im2p_sim.h>
#endif

#include <cstddef>
#include <cstdint>

struct im2p_sim;
typedef struct im2p_sim im2p_sim_t;

namespace ggml::gemmini::rmd {

struct Im2pFullExecutor;
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) || defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
struct Im2pFullExecutor {
    void * context = nullptr;
    // Synchronous borrowed descriptor. Success requires every checked output
    // callback and device completion. A failure is never retried in software.
    int (*execute)(void *, const im2p_matmul_desc_t *,
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
RmdStatus execute_rmd_stripe_im2p(im2p_sim_t * sim,
                                  const ggml_gemmini_args_t & args,
                                  const StripePacket & packet,
                                  CompressedOutput & output,
                                  RmdExecutionMetrics * metrics = nullptr,
                                  const Im2pFullExecutor * executor = nullptr);

RmdStatus execute_rmd_stripe_im2p(im2p_sim_t * sim,
                                  const ggml_gemmini_args_t & args,
                                  const StripePacket & packet,
                                  Correction & correction,
                                  RmdExecutionMetrics * metrics = nullptr,
                                  const Im2pFullExecutor * executor = nullptr);

#if defined(GGML_GEMMINI_TESTING)
RmdStatus execute_rmd_stripe_im2p_for_test(
    im2p_sim_t * sim,
    const ggml_gemmini_args_t & args,
    const StripePacket & packet,
    CompressedOutput & output,
    RmdExecutionMetrics * metrics,
    Im2pProviderTestFault fault);
RmdStatus execute_rmd_stripe_im2p_for_test(
    im2p_sim_t * sim,
    const ggml_gemmini_args_t & args,
    const StripePacket & packet,
    Correction & correction,
    RmdExecutionMetrics * metrics,
    Im2pProviderTestFault fault);
void reset_im2p_provider_dot_attempts_for_test();
[[nodiscard]] size_t im2p_provider_dot_attempts_for_test();
#endif

namespace detail {
RmdStatus execute_rmd_stripe_im2p_with_weights(im2p_sim_t * sim,
    const ggml_gemmini_args_t & args, const StripePacket & packet,
    Correction & correction, RmdWeightPreparation & weights,
    RmdExecutionMetrics * metrics = nullptr);

struct Im2pProviderStatsAggregate {
    RmdProviderStats stats{};
};

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) || defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
void expand_im2p_provider_stats(const RmdProviderStats &source,
                                im2p_work_stats_extended_t &destination) noexcept;
#endif

struct Im2pCompactDot {
    uint8_t operand_bits = 0;
    const void * activations = nullptr;
    size_t rows = 0;
    size_t activation_row_stride_bytes = 0;
    const int32_t * weights = nullptr;
    size_t columns = 0;
    size_t weight_row_stride = 0;
    size_t k = 0;
    gemmini_cycle_record_v2 timing_identity{};
    size_t block_id = 0, lane_group = 0, k_offset = 0, column_offset = 0;
};

RmdStatus execute_im2p_compact_dot(
    im2p_sim_t * sim,
    const Im2pCompactDot & dot,
    OutputValue * output,
    size_t output_row_stride,
    Im2pProviderStatsAggregate & aggregate,
    Im2pProviderTestFault fault = Im2pProviderTestFault::none,
    const Im2pFullExecutor * executor = nullptr);

} // namespace detail
} // namespace ggml::gemmini::rmd
