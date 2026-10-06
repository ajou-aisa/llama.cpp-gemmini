#include "route.hpp"

#include "ggml-gemmini-args.h"
#include "ggml-gemmini-matmul.hpp"
#include "ggml-gemmini-telemetry.hpp"
#include "im2p_gemmini_frontend.hpp"
#include "residual/rmd/rmd-types.hpp"

#include <algorithm>
#include <limits>

namespace ggml::gemmini::im2p_adapter {
namespace route_detail {

WeightFamily concrete_weight_family(ggml_gemmini_args_t::im2p_weight_format_t format) noexcept {
    using Format = ggml_gemmini_args_t::im2p_weight_format_t;
    switch (format) {
    case Format::q4_h0:
    case Format::q8_h0:
    case Format::q16_h0:
        return WeightFamily::h0;
    case Format::q8_0_unpacked_to_h1:
    case Format::q16_h1:
        return WeightFamily::h1;
    case Format::q4_hp1:
    case Format::q8_hp1:
    case Format::q16_hp1:
        return WeightFamily::hp1;
    default:
        return WeightFamily::unsupported;
    }
}

Result integrated_eligibility(uint8_t      activation_bits,
                              uint8_t      weight_bits,
                              WeightFamily family,
                              bool         exsia) noexcept {
#if defined(IM2P_SIM_IMPLEMENTATION_GEMMINI_HP1) && !CYCLE_SIM
    if (family == WeightFamily::h1)
        return {Error::unsupported_route,
                "GEMMINI_HP1 does not implement H1 unsigned-multiply execution",
                false};
#endif
    if (exsia && family == WeightFamily::h0 && activation_bits == 8 && weight_bits == 8)
        return {Error::unsupported_route,
                "integrated A8/W8 ExSIA H0 has no block-backed float provider",
                false};
    return {};
}

::im2p::gemmini::Status to_frontend_status(const Result & result) noexcept {
    using Code = ::im2p::gemmini::StatusCode;
    Code code  = Code::execution_failure;
    switch (result.error) {
    case Error::success:
        code = Code::success;
        break;
    case Error::invalid_argument:
        code = Code::invalid_argument;
        break;
    case Error::invalid_contract:
        code = Code::invalid_contract;
        break;
    case Error::unsupported_route:
        code = Code::unsupported_route;
        break;
    case Error::invalid_state:
        code = Code::invalid_state;
        break;
    case Error::backpressure:
        code = Code::backpressure;
        break;
    case Error::out_of_memory:
        code = Code::out_of_memory;
        break;
    case Error::execution_failure:
        break;
    }
    return {code, ::im2p::gemmini::Route::unknown, result.native_contract, result.message};
}

Result from_rmd_status(rmd::RmdStatus status) noexcept {
    using Source = rmd::RmdStatus;
    Error error  = Error::execution_failure;
    switch (status) {
    case Source::success:
        return {};
    case Source::invalid_arguments:
    case Source::invalid_packet:
        error = Error::invalid_argument;
        break;
    case Source::unsupported_route:
    case Source::residual_too_wide:
        error = Error::unsupported_route;
        break;
    case Source::allocation_failure:
        error = Error::out_of_memory;
        break;
    case Source::overflow:
    case Source::execution_failed:
        break;
    }
    return {error, rmd::rmd_status_message(status), false};
}

Result from_matmul_status(const MatmulStatus & status) noexcept {
    if (status.ok()) {
        return {};
    }
    Error error = Error::execution_failure;
    switch (status.code) {
    case MatmulStatusCode::invalid_argument:
        error = Error::invalid_argument;
        break;
    case MatmulStatusCode::unsupported_invocation:
        error = Error::unsupported_route;
        break;
    case MatmulStatusCode::out_of_memory:
        error = Error::out_of_memory;
        break;
    case MatmulStatusCode::invalid_state:
        error = Error::invalid_state;
        break;
    default:
        break;
    }
    return {error, status.message, false};
}

Stats translate_stats(const im2p_work_stats_extended_t & source) noexcept {
    const auto & base = source.base;
    return Stats{
        base.work_total_cycles,
        base.activation_read_requests,
        base.weight_read_requests,
        base.scale_read_requests,
        base.output_write_requests,
        base.output_write_responses,
        base.activation_wait_cycles,
        base.weight_wait_cycles,
        base.scale_wait_cycles,
        base.output_wait_cycles,
        base.stripe_host_wait_cycles,
        base.drain_cycles,
        base.weight_preload_cycles,
        base.same_block_scale_hits,
        base.next_scale_hits,
        base.scale_demand_misses,
        base.compute_cycles,
        base.overlap_cycles,
        base.activation_overlap_cycles,
        base.weight_overlap_cycles,
        base.scale_overlap_cycles,
        base.completed_fragments,
        base.completed_output_tiles,
        base.completed_stripes,
        base.stripes_published,
        base.stripe_rows_published,
        base.weight_bank_activations,
        source.cross_stripe_overlap_cycles,
        source.lookahead_prepared,
        source.lookahead_publish_cycle,
        source.lookahead_first_activation_cycle,
        source.lookahead_first_weight_cycle,
        source.lookahead_weight_preload_cycle,
        source.lookahead_weight_requests,
        source.lookahead_weight_reuse_hits,
        source.lookahead_scale_cycle,
        source.lookahead_scale_requests,
        source.lookahead_scale_reuses,
        source.current_stripe_completion_cycle,
        source.lookahead_ready_cycle,
        source.lookahead_start_cycle,
    };
}

void device_diagnostics(Im2pExecutionTelemetry &    record,
                        const ggml_gemmini_args_t & args,
                        const Stats &               stats,
                        bool                        residual = false) {
    record.provider_stats  = stats;
    record.backend         = "im2p_sim";
    record.clock_domain    = residual ? "independent_rmd_simulator" : "dense_simulator";
    record.activation_bits = GGML_GEMMINI_ACTIVATION_BITS;
    record.weight_bits     = GGML_GEMMINI_WEIGHT_BITS;
    record.dim             = DIM;
    record.problem_i       = args.I;
    record.problem_j       = args.J;
    record.problem_k       = args.K;
}

Result validate_stripe_timings(const ::im2p::gemmini::StripeRtlTimingView & timings,
                               const ggml_gemmini_args_t &                  args,
                               const Stats &                                stats,
                               std::uint64_t expected_run_id) noexcept {
    if (args.activation_rows_per_stripe == 0) {
        return {Error::invalid_contract,
                "PIPELINE stripe timing geometry has zero rows per stripe",
                false};
    }
    const std::uint64_t expected_count =
        1 + (static_cast<std::uint64_t>(args.I) - 1) /
                static_cast<std::uint64_t>(args.activation_rows_per_stripe);
    if (timings.data == nullptr || timings.size != expected_count ||
        stats.rtl_stripes_published != expected_count ||
        stats.rtl_stripe_rows_published != args.I) {
        return {Error::invalid_contract,
                "PIPELINE stripe timing count does not match publication statistics",
                false};
    }
    std::size_t expected_row_begin = 0;
    for (std::size_t index = 0; index < timings.size; ++index) {
        const auto &      timing = timings[index];
        const std::size_t expected_row_end =
            std::min(args.I, expected_row_begin + args.activation_rows_per_stripe);
        if (timing.run_id != expected_run_id || timing.stripe_id != index ||
            timing.slot != index % 2 || timing.row_begin != expected_row_begin ||
            timing.row_end != expected_row_end ||
            timing.publish_to_completion_cycles != timing.completion_cycle - timing.publish_cycle) {
            return {Error::invalid_contract,
                    "PIPELINE stripe timing metadata is malformed or out of order",
                    false};
        }
        expected_row_begin = expected_row_end;
    }
    if (expected_row_begin != args.I) {
        return {Error::invalid_contract,
                "PIPELINE stripe timings do not partition the output rows",
                false};
    }
    return {};
}

Result validate_residual_stripe_timings(const ::im2p::gemmini::FenceResult & result,
                                        std::uint64_t expected_run_id) noexcept {
    const Result status = translate(result.status);
    if (!status.ok())
        return status;

    const auto count = result.semantic_completion_count;
    if (result.semantic_stripes.size != count || result.residual_stripe_timings.size != count ||
        (count != 0 && (result.semantic_stripes.data == nullptr ||
                        result.residual_stripe_timings.data == nullptr))) {
        return {Error::invalid_contract, "semantic and RMD telemetry counts do not match", false};
    }

    std::uint64_t summed_calls = 0;
    std::uint64_t summed_work  = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const auto & semantic = result.semantic_stripes[index];
        const auto & timing   = result.residual_stripe_timings[index];
        if (semantic.run_id != expected_run_id || timing.run_id != expected_run_id ||
            semantic.stripe_id != index || timing.stripe_id != index ||
            semantic.slot != timing.slot || semantic.row_begin != timing.row_begin ||
            semantic.row_end != timing.row_end || semantic.row_end < semantic.row_begin ||
            timing.rmd_dot_calls > std::numeric_limits<std::uint64_t>::max() - summed_calls ||
            timing.rmd_stats.base.work_total_cycles >
                std::numeric_limits<std::uint64_t>::max() - summed_work) {
            return {
                Error::invalid_contract, "semantic or RMD stripe telemetry is malformed", false};
        }
        summed_calls += timing.rmd_dot_calls;
        summed_work += timing.rmd_stats.base.work_total_cycles;
    }
    if (summed_calls != result.rmd_dot_calls ||
        summed_work != result.rmd_stats.base.work_total_cycles) {
        return {Error::invalid_contract,
                "RMD aggregate telemetry does not match stripe durations",
                false};
    }

    return {};
}

} // namespace route_detail

using route_detail::integrated_eligibility;
using route_detail::translate_stats;

Result translate(const ::im2p::gemmini::Status & status) noexcept {
    using Source = ::im2p::gemmini::StatusCode;

    Error error = Error::execution_failure;
    switch (status.code) {
    case Source::success:
        error = Error::success;
        break;
    case Source::invalid_argument:
        error = Error::invalid_argument;
        break;
    case Source::invalid_contract:
        error = Error::invalid_contract;
        break;
    case Source::unsupported_route:
        error = Error::unsupported_route;
        break;
    case Source::invalid_state:
        error = Error::invalid_state;
        break;
    case Source::backpressure:
        error = Error::backpressure;
        break;
    case Source::out_of_memory:
        error = Error::out_of_memory;
        break;
    case Source::execution_failure:
        error = Error::execution_failure;
        break;
    }
    return {error, status.message, status.native_contract};
}

Completion translate(const ::im2p::gemmini::FenceResult & result,
                     ::im2p::gemmini::Mode                mode,
                     std::uint64_t                        expected_publications,
                     std::uint64_t                        expected_published_rows) noexcept {
    const auto & base       = result.stats.base;
    Stats        stats      = translate_stats(result.stats);
    Result       translated = translate(result.status);
    if (!translated.ok()) {
        return {translated, stats};
    }

    if (mode == ::im2p::gemmini::Mode::full) {
        if (expected_publications != 0 || expected_published_rows != 0 ||
            base.stripes_published != 0 || base.stripe_rows_published != 0) {
            return {{Error::invalid_contract,
                     "FULL IM2P statistics must publish zero stripes and rows",
                     false},
                    stats};
        }
    } else if (expected_publications == 0 || expected_published_rows == 0 ||
               base.stripes_published != expected_publications ||
               base.stripe_rows_published != expected_published_rows) {
        return {{Error::invalid_contract,
                 "PIPELINE IM2P publication statistics do not match canonical geometry",
                 false},
                stats};
    }
    Completion completion{translated, stats};
    completion.semantic_completion_count = result.semantic_completion_count;
    completion.rmd_dot_calls             = result.rmd_dot_calls;
    completion.rmd_stats                 = translate_stats(result.rmd_stats);
    return completion;
}

Result gate_route(const ExsiaRouteRequest & request) noexcept {
    const auto supported_width = [](std::uint8_t bits) {
        return bits == 4 || bits == 8 || bits == 16;
    };
    if (!supported_width(request.activation_bits)) {
        return {Error::unsupported_route, "unsupported IM2P activation width", false};
    }
    if (!supported_width(request.weight_bits)) {
        return {Error::unsupported_route, "unsupported IM2P weight width", false};
    }
    if (request.artifact_activation_bits != request.activation_bits ||
        request.artifact_weight_bits != request.weight_bits) {
        return {Error::invalid_contract,
                "IM2P artifact identity does not match the requested route",
                false};
    }
    if (request.mode != PublicMode::full && request.mode != PublicMode::stripe_pipeline) {
        return {Error::unsupported_route, "unsupported public matmul mode", false};
    }
    if (request.residual_backend != ResidualBackend::cpu_direct &&
        request.residual_backend != ResidualBackend::compact_ws) {
        return {Error::unsupported_route, "unsupported residual backend", false};
    }
    const auto eligibility = integrated_eligibility(
        request.activation_bits, request.weight_bits, request.family, request.exsia);
    if (!eligibility.ok())
        return eligibility;
    if (!request.exsia) {
        if (request.block_activation && request.mode != PublicMode::full) {
            return {Error::unsupported_route, "BLOCK IM2P requires FULL mode", false};
        }
        if (request.rmd_enabled) {
            if (request.mode != PublicMode::full) {
                return {Error::unsupported_route, "baseline IM2P RMD requires FULL mode", false};
            }
            if (request.build_identity != BuildIdentity::im2p_sim_ws) {
                return {Error::unsupported_route,
                        "baseline IM2P RMD requires the WS+IM2P_SIM build identity",
                        false};
            }
            if (request.activation_bits != request.weight_bits) {
                return {Error::unsupported_route,
                        "baseline IM2P RMD requires matched activation and weight widths",
                        false};
            }
            switch (request.family) {
            case WeightFamily::h0:
                return request.residual_backend == ResidualBackend::cpu_direct
                           ? Result{}
                           : Result{Error::unsupported_route,
                                    "H0 baseline requires CPU-direct residual execution",
                                    false};
            case WeightFamily::h1:
            case WeightFamily::hp1:
                return {};
            case WeightFamily::channel:
                return request.weight_bits == 8 && !request.block_activation
                           ? Result{}
                           : Result{Error::unsupported_route,
                                    "channel RMD requires an 8-bit non-BLOCK activation",
                                    false};
            case WeightFamily::h2:
            case WeightFamily::hp2:
                return {Error::unsupported_route,
                        "H2/HP2 baseline residual formats are unsupported",
                        false};
            case WeightFamily::unsupported:
                return {
                    Error::unsupported_route, "unsupported baseline residual weight family", false};
            }
            return {Error::unsupported_route, "unsupported baseline residual weight family", false};
        }
        if (request.activation_bits == request.weight_bits) {
            return {};
        }
        return {Error::unsupported_route,
                "IM2P routes require matched activation and weight widths",
                false};
    }
    if (request.build_identity != BuildIdentity::im2p_sim_ws) {
        return {
            Error::unsupported_route, "ExSIA IM2P requires the WS+IM2P_SIM build identity", false};
    }
    if (!request.rmd_enabled) {
        return {Error::unsupported_route, "ExSIA IM2P requires RMD", false};
    }
    if (request.activation_bits != request.weight_bits) {
        return {Error::unsupported_route,
                "ExSIA IM2P requires matched activation and weight widths",
                false};
    }
    switch (request.family) {
    case WeightFamily::h0:
        if (request.residual_backend != ResidualBackend::cpu_direct) {
            return {
                Error::unsupported_route, "H0 ExSIA requires CPU-direct residual execution", false};
        }
        return {};
    case WeightFamily::h1:
    case WeightFamily::hp1:
        return {};
    case WeightFamily::h2:
    case WeightFamily::hp2:
        return {Error::unsupported_route, "H2/HP2 ExSIA residual formats are unsupported", false};
    case WeightFamily::unsupported:
    case WeightFamily::channel:
        return {Error::unsupported_route, "unsupported ExSIA residual weight family", false};
    }
    return {Error::unsupported_route, "unsupported ExSIA route", false};
}

Result gate_route(bool         exsia,
                  std::uint8_t activation_bits,
                  bool         rmd_enabled,
                  bool         cpu_direct_rmd,
                  std::uint8_t weight_bits) noexcept {
    return gate_route({exsia,
                       activation_bits,
                       weight_bits,
                       activation_bits,
                       weight_bits,
                       rmd_enabled,
                       PublicMode::full,
                       WeightFamily::h1,
                       cpu_direct_rmd ? ResidualBackend::cpu_direct : ResidualBackend::compact_ws,
                       BuildIdentity::im2p_sim_ws});
}

} // namespace ggml::gemmini::im2p_adapter
