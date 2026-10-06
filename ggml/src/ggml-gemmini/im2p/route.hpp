#pragma once

#include <cstdint>

namespace im2p::gemmini {
enum class Mode : std::uint8_t;
struct Status;
struct FenceResult;
} // namespace im2p::gemmini

namespace ggml::gemmini::im2p_adapter {

enum class Error : std::uint8_t {
    success,
    invalid_argument,
    invalid_contract,
    unsupported_route,
    invalid_state,
    backpressure,
    out_of_memory,
    execution_failure,
};

struct Result {
    Error        error           = Error::success;
    const char * message         = "success";
    bool         native_contract = false;

    [[nodiscard]] bool ok() const noexcept {
        return error == Error::success;
    }
};

struct Stats {
    // Complete raw 64-bit RTL statistics in provider ABI order. Wait and
    // overlap counters are independent observations, not additive totals.
    std::uint64_t rtl_work_total_cycles                        = 0;
    std::uint64_t rtl_activation_read_requests                 = 0;
    std::uint64_t rtl_weight_read_requests                     = 0;
    std::uint64_t rtl_scale_read_requests                      = 0;
    std::uint64_t rtl_output_write_requests                    = 0;
    std::uint64_t rtl_output_write_responses                   = 0;
    std::uint64_t rtl_activation_wait_cycles                   = 0;
    std::uint64_t rtl_weight_wait_cycles                       = 0;
    std::uint64_t rtl_scale_wait_cycles                        = 0;
    std::uint64_t rtl_output_wait_cycles                       = 0;
    std::uint64_t rtl_stripe_host_wait_cycles                  = 0;
    std::uint64_t rtl_drain_cycles                             = 0;
    std::uint64_t rtl_weight_preload_cycles                    = 0;
    std::uint64_t rtl_same_block_scale_hits                    = 0;
    std::uint64_t rtl_next_scale_hits                          = 0;
    std::uint64_t rtl_scale_demand_misses                      = 0;
    std::uint64_t rtl_compute_cycles                           = 0;
    std::uint64_t rtl_overlap_cycles                           = 0;
    std::uint64_t rtl_activation_overlap_cycles                = 0;
    std::uint64_t rtl_weight_overlap_cycles                    = 0;
    std::uint64_t rtl_scale_overlap_cycles                     = 0;
    std::uint64_t rtl_completed_fragments                      = 0;
    std::uint64_t rtl_completed_output_works                   = 0;
    std::uint64_t rtl_scheduler_groups_completed               = 0;
    std::uint64_t rtl_stripes_published                        = 0;
    std::uint64_t rtl_stripe_rows_published                    = 0;
    std::uint64_t rtl_weight_bank_activations                  = 0;
    std::uint64_t rtl_cross_stripe_overlap_cycles              = 0;
    std::uint64_t rtl_lookahead_prepared                       = 0;
    std::uint64_t rtl_first_publish_cycle                      = 0;
    std::uint64_t rtl_first_activation_read_cycle              = 0;
    std::uint64_t rtl_first_weight_read_cycle                  = 0;
    std::uint64_t rtl_weight_preload_cycle                     = 0;
    std::uint64_t rtl_lookahead_weight_requests                = 0;
    std::uint64_t rtl_lookahead_weight_reuse_hits              = 0;
    std::uint64_t rtl_first_scale_read_cycle                   = 0;
    std::uint64_t rtl_lookahead_scale_requests                 = 0;
    std::uint64_t rtl_lookahead_scale_reuses                   = 0;
    std::uint64_t rtl_current_scheduler_group_completion_cycle = 0;
    std::uint64_t rtl_lookahead_ready_cycle                    = 0;
    std::uint64_t rtl_lookahead_start_cycle                    = 0;
};

struct Completion {
    Result        result{};
    Stats         stats{};
    std::uint64_t run_id                    = 0;
    std::uint64_t semantic_completion_count = 0;
    std::uint64_t rmd_dot_calls             = 0;
    // Independent residual-simulator counters. They are never added to stats.
    Stats rmd_stats{};
};

enum class PublicMode : std::uint8_t {
    full,
    stripe_pipeline,
};

enum class WeightFamily : std::uint8_t {
    h0,
    h1,
    hp1,
    h2,
    hp2,
    unsupported,
    channel,
};

enum class ResidualBackend : std::uint8_t {
    cpu_direct,
    compact_ws,
};

enum class BuildIdentity : std::uint8_t {
    im2p_sim_ws,
    hardware_ws,
    hardware_cpu,
    hardware_os,
    unsupported,
};

struct ExsiaRouteRequest {
    bool            exsia                    = true;
    std::uint8_t    activation_bits          = 0;
    std::uint8_t    weight_bits              = 0;
    std::uint8_t    artifact_activation_bits = 0;
    std::uint8_t    artifact_weight_bits     = 0;
    bool            rmd_enabled              = false;
    PublicMode      mode                     = PublicMode::full;
    WeightFamily    family                   = WeightFamily::unsupported;
    ResidualBackend residual_backend         = ResidualBackend::cpu_direct;
    BuildIdentity   build_identity           = BuildIdentity::unsupported;
    bool            block_activation         = false;
};

[[nodiscard]] Result     translate(const ::im2p::gemmini::Status & status) noexcept;
[[nodiscard]] Completion translate(const ::im2p::gemmini::FenceResult & result,
                                   ::im2p::gemmini::Mode                mode,
                                   std::uint64_t                        expected_publications,
                                   std::uint64_t expected_published_rows) noexcept;
[[nodiscard]] Result     gate_route(const ExsiaRouteRequest & request) noexcept;
[[nodiscard]] Result     gate_route(bool         exsia,
                                    std::uint8_t activation_bits,
                                    bool         rmd_enabled,
                                    bool         cpu_direct_rmd,
                                    std::uint8_t weight_bits = 8) noexcept;

} // namespace ggml::gemmini::im2p_adapter
