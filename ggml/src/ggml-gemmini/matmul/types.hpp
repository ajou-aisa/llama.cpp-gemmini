#pragma once

#include "../ggml-gemmini-args.h"
#include "../../ggml-npu-common/matmul/options.hpp"
#include "../../ggml-npu-common/profiling/matmul-types.hpp"
#include "../../ggml-npu-common/profiling/telemetry.hpp"
#include "../../ggml-npu-common/quantization/activation/exsia/exsia-event.hpp"
#include "../../ggml-npu-common/residual/rmd/rmd-compose.hpp"
#include "../../ggml-npu-common/residual/metrics.hpp"

#include <array>
#if defined(__linux__)
#include <sched.h>
#endif
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../../ggml-npu-common/profiling/include/npu/cycle_reader.hpp"
#include "../../ggml-npu-common/profiling/include/npu/host-timing.hpp"
#include "../../ggml-npu-common/profiling/include/npu/cpu_log_context.hpp"

#if !defined(GGML_GEMMINI_CONFIG_HAS_ACTIVATION_QUANT)
namespace ggml::npu::config {
inline constexpr int ACTIVATION_QUANT = static_cast<int>(CURRENT_ACTIVATION_QUANT);
}
#endif

namespace ggml::gemmini {

namespace detail {

enum class ActivationRoute : uint8_t { unknown, fp32, exsia, tensor, token, block, stripe };
enum class WeightRoute : uint8_t {
    unknown,
    fp32,
    tensor_i8,
    channel_i8,
    block_i8,
    affine,
    q8_hp1,
    q8_h2,
    q8_hp2,
    q8_channel_direct,
    q8_channel_sidecar,
    q8_h0
};
enum class BackendRoute : uint8_t { cpu, gemmini_ws, gemmini_os, ws_sim };

struct RouteKey {
    ActivationRoute activation = ActivationRoute::unknown;
    WeightRoute     weight     = WeightRoute::unknown;
    BackendRoute    backend    = BackendRoute::cpu;
};

struct RouteCapabilities {
    bool legacy_full             = false;
    bool full                    = false;
    bool sliced_dense            = false;
    bool sliced_compensation     = false;
    bool live_stripe_producer    = false;
    bool internal_parallel_dense = false;
    bool deprecated              = false;
};

RouteKey          normalize_route(const ggml_gemmini_args_t & args);
RouteCapabilities route_capabilities(const ggml_gemmini_args_t & args);
const char *      activation_route_name(ActivationRoute route);
const char *      weight_route_name(WeightRoute route);
const char *      backend_route_name(BackendRoute route);

} // namespace detail

enum class MatMulStatus {
    success,
    empty_stripes,
    malformed_stripe,
    duplicate_stripe,
    overlapping_stripe,
    missing_stripes,
    unsupported,
    invalid_contract,
    invalid_state,
    invalid_arguments,
};

enum class MatMulCapability {
    supported,
    unsupported,
};

enum class MatMulState {
    idle,
    accepting_stripes,
    completed,
};

struct MatMulStripe {
    size_t row_begin;
    size_t row_end;
};

struct MatMulResult {
    MatMulStatus     status;
    MatMulCapability capability;
};

enum class MatmulStatusCode : uint8_t {
    success,
    invalid_argument,
    invalid_contract,
    unsupported_route,
    unsupported_backend,
    unsupported_invocation,
    invalid_state,
    out_of_memory,
    execution_failure,
    cancelled,
};

struct MatmulStatus {
    MatmulStatusCode code       = MatmulStatusCode::success;
    const char *     message    = "success";
    MatMulCapability capability = MatMulCapability::supported;

    bool ok() const {
        return code == MatmulStatusCode::success;
    }

    explicit operator bool() const {
        return ok();
    }
};

#if defined(GGML_GEMMINI_TESTING)
struct MatmulTestCounters {
    uint64_t execution_constructions   = 0;
    uint64_t allocation_attempts       = 0;
    uint64_t dense_dispatches          = 0;
    uint64_t residual_dispatches       = 0;
    uint64_t hardware_dispatches       = 0;
    uint64_t fallback_dispatches       = 0;
    uint64_t native_integer_block_dots = 0;
    uint64_t native_post_dot_scales    = 0;
};

void               test_reset_matmul_counters();
MatmulTestCounters test_matmul_counters();
void               test_inject_output_stage_allocation_failure();
#endif

namespace detail {

struct MatmulCaptureTiming {
    MatmulStageMetrics capture_copy;
    MatmulStageMetrics producer_wait;
    MatmulStageMetrics queue_insert;
    MatmulStageMetrics rmd_pack;
    uint64_t           producer_wait_start_ns = 0;
    uint64_t           producer_wait_end_ns   = 0;
    uint64_t           queued_ns              = 0;
    uint64_t           dequeued_ns            = 0;
    uint64_t           enqueue_tid            = 0;
    uint64_t           dequeue_tid            = 0;
    uint64_t           telemetry_queued_tick  = 0;
};

struct MatmulCapturedStripe {
    gemmini_trace_context                                     trace_origin{};
    uint32_t                                                  cpu_identity_mask = 0;
    uint64_t                                                  run_id            = 0;
    size_t                                                    stripe_id         = 0;
    size_t                                                    slot              = 0;
    size_t                                                    row_begin         = 0;
    size_t                                                    row_end           = 0;
    std::optional<quants::act::exsia::StripeMetadataSnapshot> activation_metadata;
    rmd::StripePacketHandle                                   rmd_packet;
    residual::DirectStripePayloadHandle                       direct_residual;
    uint64_t                                                  la3_ns               = 0;
    uint64_t                                                  sf1_ns               = 0;
    uint64_t                                                  sf_mask_start_ns     = 0;
    uint64_t                                                  sf_mask_end_ns       = 0;
    uint64_t                                                  sf_exponent_start_ns = 0;
    uint64_t                                                  sf_exponent_end_ns   = 0;
    uint64_t                                                  sf_folding_start_ns  = 0;
    uint64_t                                                  sf_folding_end_ns    = 0;
    uint64_t                                                  sf_commit_ns         = 0;
    MatmulCaptureTiming                                       timing;
};

} // namespace detail

struct MatmulCollectorSnapshot {
    MatmulStatus status;
    size_t       capacity  = 0;
    size_t       pending   = 0;
    size_t       in_flight = 0;
    bool         running   = false;
};

enum class MatmulDenseState : uint8_t;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
enum class MatmulCollectorThread : uint8_t {
    worker,
};

enum class MatmulCollectorThreadFailure : uint8_t {
    exception,
    out_of_memory,
};
#endif

enum class MatmulExecutionState {
    empty,
    prepared,
    running,
    finishing,
    completed,
    failed,
};

enum class MatmulDenseState : uint8_t {
    idle,
    running,
    complete,
    failed,
    cancelled,
};

enum class MatmulResidualState : uint8_t {
    idle,
    ready,
    running,
    complete,
    failed,
    cancelled,
};

struct MatmulStripeJobSnapshot {
    MatmulStatus        status;
    MatmulJobMetrics    metrics;
    MatmulDenseState    dense     = MatmulDenseState::idle;
    MatmulResidualState residual  = MatmulResidualState::idle;
    bool                captured  = false;
    bool                finalized = false;
    bool                released  = false;
};

} // namespace ggml::gemmini
