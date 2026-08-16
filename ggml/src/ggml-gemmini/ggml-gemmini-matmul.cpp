#ifndef GGML_GEMMINI_PIPELINE_WRITER_TEST_ONLY
#define GGML_GEMMINI_MATMUL_IMPLEMENTATION 1
#endif
#include "ggml-gemmini-matmul.hpp"

#include <gemmini/log.hpp>

#include <cstdio>
#include <filesystem>
#include <string>

namespace ggml::gemmini::detail {

bool append_pipeline_stripe_summary_jsonl(const std::string & json_record) {
    const std::filesystem::path path =
        log::resolve_output_path(GEMMINI_LOG_DEFAULT_DEBUG_PATH);
    if (!log::prepare_output_parent(path))
        return false;
    FILE * out = std::fopen(path.string().c_str(), "a");
    if (out == nullptr) {
        return false;
    }
    const bool wrote = std::fputs(json_record.c_str(), out) >= 0 &&
        std::fputc('\n', out) != EOF;
    const bool closed = std::fclose(out) == 0;
    return wrote && closed;
}

}

#ifndef GGML_GEMMINI_PIPELINE_WRITER_TEST_ONLY

#include "quants/act/quantize.hpp"
#include "quants/act/dispatch.hpp"
#include "residual/rmd/rmd-builder.hpp"

#include <gemmini.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ggml::gemmini {
namespace {

bool checked_offset(size_t row, size_t stride, size_t & offset) {
    if (stride != 0 && row > std::numeric_limits<size_t>::max() / stride) {
        return false;
    }
    offset = row * stride;
    return true;
}

bool row_invariant_activation(const ggml_gemmini_args_t & args) {
    const auto & storage = args.act_quant.storage();
    return std::holds_alternative<quants::act::NoneMeta>(storage) ||
        std::holds_alternative<quants::act::tensor::Meta>(storage);
}

bool supports_row_slice_activation(const ggml_gemmini_args_t & args) {
    const auto & storage = args.act_quant.storage();
    return row_invariant_activation(args) ||
        std::holds_alternative<quants::act::exsia::Meta>(storage) ||
        std::holds_alternative<quants::act::token::Meta>(storage) ||
        std::holds_alternative<quants::act::block::Meta>(storage) ||
        std::holds_alternative<quants::act::stripe::Meta>(storage);
}

bool uses_baseline_channel_route(const ggml_gemmini_args_t & args) {
    return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel ||
        args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
}

bool valid_matmul_shape(const ggml_gemmini_args_t & args) {
    return args.I != 0 && args.J != 0 && args.K != 0 && args.f_out != nullptr &&
        (args.A != nullptr || args.A_fp32 != nullptr) &&
        ((args.A_fp32 == nullptr) == (args.B_fp32 == nullptr));
}

bool valid_activation_metadata(const ggml_gemmini_args_t & args) {
    if (std::holds_alternative<quants::act::NoneMeta>(args.act_quant.storage())) {
        return args.A == nullptr && args.A_fp32 != nullptr && args.B_fp32 != nullptr;
    }
    if (args.I > std::numeric_limits<size_t>::max() - args.activation_row_offset) {
        return false;
    }
    const quants::act::ActivationMetadataView metadata(
        args, args.activation_row_offset, args.activation_row_offset + args.I);
    float scale = 0.0f;
    for (size_t row = 0; metadata.valid() && row < args.I; ++row) {
        if (!metadata.scale(row, scale)) {
            return false;
        }
    }
    return metadata.valid();
}

bool finite_output(const ggml_gemmini_args_t & args) {
    const size_t row_stride = args.stride_f_out != 0 ? args.stride_f_out : args.J;
    const size_t col_stride = args.col_stride_f_out != 0 ? args.col_stride_f_out : 1;
    for (size_t row = 0; row < args.I; ++row) {
        for (size_t col = 0; col < args.J; ++col) {
            if (!std::isfinite(args.f_out[row * row_stride + col * col_stride])) {
                return false;
            }
        }
    }
    return true;
}

baseline_activation_quant_t baseline_activation_for(const ggml_gemmini_args_t & args) {
    const auto & storage = args.act_quant.storage();
    if (std::holds_alternative<quants::act::tensor::Meta>(storage)) {
        return baseline_activation_quant_t::TENSOR;
    }
    if (std::holds_alternative<quants::act::token::Meta>(storage)) {
        return baseline_activation_quant_t::TOKEN;
    }
    if (std::holds_alternative<quants::act::block::Meta>(storage) ||
        std::holds_alternative<quants::act::stripe::Meta>(storage)) {
        return baseline_activation_quant_t::BLOCK;
    }
    return baseline_activation_quant_t::EXSIA;
}

void execute_dense(ggml_gemmini_args_t &args) {
    if (args.A_fp32 != nullptr || args.B_fp32 != nullptr) {
        if (args.A_fp32 == nullptr || args.B_fp32 == nullptr || args.f_out == nullptr) {
            return;
        }
        matmul_cpu_fp(false, true, args.I, args.J, args.K,
                      args.A_fp32, args.B_fp32, nullptr, args.f_out,
                      args.sA, args.sB, args.col_stride_f_out, args.stride_f_out);
        return;
    }
    if (uses_baseline_channel_route(args)) {
        tiled_matmul_auto_baseline(&args, baseline_activation_for(args), baseline_weight_quant_t::CHANNEL);
    } else if (args.weight_i8_scale_active) {
        tiled_matmul_auto_baseline(&args, baseline_activation_for(args), baseline_weight_quant_t::TENSOR);
    } else {
        tiled_matmul_auto_im2p(&args);
    }
}

MatMulStatus execute_stripe(ggml_gemmini_args_t args, MatMulStripe stripe, size_t stripe_id,
                            bool metadata_is_local = false) {
    const size_t input_stride = args.sA ? args.sA : args.K;
    const size_t output_stride = args.stride_f_out ? args.stride_f_out : args.J;
    size_t input_offset = 0;
    size_t output_offset = 0;
    if (!checked_offset(stripe.row_begin, input_stride, input_offset) ||
        !checked_offset(stripe.row_begin, output_stride, output_offset)) {
        return MatMulStatus::invalid_arguments;
    }

    if (args.tile_I == 0 || args.tile_J == 0 || args.tile_K == 0) {
        gemmini_set_tile_ws(&args);
    }
    const size_t metadata_tile_I = args.tile_I;
    if (stripe.row_begin > std::numeric_limits<size_t>::max() - args.activation_row_offset) {
        return MatMulStatus::invalid_arguments;
    }
    args.activation_row_offset += stripe.row_begin;

    args.I = stripe.row_end - stripe.row_begin;
    gemmini_set_tile_ws(&args);
    args.tile_I = metadata_tile_I;
    if (args.A != nullptr) {
        args.A += input_offset;
    }
    if (args.A_fp32 != nullptr) {
        args.A_fp32 += input_offset;
    }
    args.f_out += output_offset;

    (void) stripe_id;
    (void) metadata_is_local;

    execute_dense(args);
    return MatMulStatus::success;
}

MatmulStatus to_public_status(MatMulStatus status, MatMulCapability capability,
                              const ggml_gemmini_args_t * args = nullptr) {
    MatmulStatusCode code = MatmulStatusCode::invalid_state;
    const char * message = "invalid state";
    switch (status) {
        case MatMulStatus::success:
            code = MatmulStatusCode::success;
            message = "success";
            break;
        case MatMulStatus::empty_stripes:
            code = MatmulStatusCode::invalid_contract;
            message = "missing stripes";
            break;
        case MatMulStatus::malformed_stripe:
            code = MatmulStatusCode::invalid_argument;
            message = "invalid stripe bounds";
            break;
        case MatMulStatus::duplicate_stripe:
            code = MatmulStatusCode::invalid_contract;
            message = "duplicate stripe";
            break;
        case MatMulStatus::overlapping_stripe:
            code = MatmulStatusCode::invalid_contract;
            message = "overlapping stripe";
            break;
        case MatMulStatus::missing_stripes:
            code = MatmulStatusCode::invalid_contract;
            message = "missing stripes";
            break;
        case MatMulStatus::unsupported:
            if (args != nullptr) {
                const auto backend = detail::normalize_route(*args).backend;
                if (backend == detail::BackendRoute::gemmini_os ||
                    backend == detail::BackendRoute::ws_sim) {
                    code = MatmulStatusCode::unsupported_backend;
                    message = "unsupported Gemmini backend";
                    break;
                }
            }
            code = MatmulStatusCode::unsupported_route;
            message = "unsupported route";
            break;
        case MatMulStatus::invalid_contract:
            code = MatmulStatusCode::invalid_contract;
            message = "invalid route contract";
            break;
        case MatMulStatus::invalid_state:
            break;
        case MatMulStatus::invalid_arguments:
            code = MatmulStatusCode::invalid_argument;
            message = "invalid argument";
            break;
    }
    return { code, message, capability };
}

MatmulStatus make_status(MatmulStatusCode code, const char * message,
                         MatMulCapability capability = MatMulCapability::supported) {
    return { code, message, capability };
}

MatmulStatus invalid_state(const char * message = "invalid state") {
    return make_status(MatmulStatusCode::invalid_state, message);
}

MatmulStatus invalid_contract(const char * message) {
    return make_status(MatmulStatusCode::invalid_contract, message);
}

MatmulStatus unsupported_backend(const char * message) {
    return make_status(MatmulStatusCode::unsupported_backend, message, MatMulCapability::unsupported);
}

MatmulStatus from_rmd_status(rmd::RmdStatus status) {
    switch (status) {
        case rmd::RmdStatus::success:
            return {};
        case rmd::RmdStatus::unsupported_route:
            return unsupported_backend(rmd::rmd_status_message(status));
        case rmd::RmdStatus::invalid_arguments:
        case rmd::RmdStatus::invalid_packet:
            return make_status(MatmulStatusCode::invalid_argument,
                               rmd::rmd_status_message(status), MatMulCapability::unsupported);
        case rmd::RmdStatus::allocation_failure:
            return make_status(MatmulStatusCode::out_of_memory, rmd::rmd_status_message(status));
        case rmd::RmdStatus::residual_too_wide:
        case rmd::RmdStatus::overflow:
        case rmd::RmdStatus::execution_failed:
            return make_status(MatmulStatusCode::execution_failure,
                               rmd::rmd_status_message(status));
    }
    return make_status(MatmulStatusCode::execution_failure, "rmd: unknown status");
}

using Clock = std::chrono::steady_clock;

void record_metric(MatmulStageMetrics & metric, bool enabled, Clock::time_point start) {
    if (!enabled) {
        return;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - start).count();
    metric.nanoseconds += static_cast<uint64_t>(std::max<int64_t>(1, elapsed));
    ++metric.count;
}

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count());
}

}

namespace detail {

namespace {

struct RouteDescriptor {
    bool legacy_full;
    bool facade_full;
    bool facade_sequential;
    bool facade_pipeline;
    bool deprecated;
};

struct WeightDescriptor {
    bool legacy_full;
    bool facade_full;
    bool facade_sliced;
    bool deprecated;
};

constexpr size_t activation_route_count = static_cast<size_t>(ActivationRoute::stripe) + 1;
constexpr size_t weight_route_count = static_cast<size_t>(WeightRoute::q8_h0) + 1;

constexpr std::array<WeightDescriptor, weight_route_count> weight_descriptors = {{
    { false, false, false, false },
    { true,  true,  true,  false },
    { true,  true,  true,  false },
    { true,  false, false, false },
    { true,  false, false, false },
    { true,  true,  true,  false },
    { true,  true,  true,  false },
    { true,  true,  false, true  },
    { true,  true,  false, true  },
    { true,  true,  true,  false },
    { true,  true,  true,  false },
    { false, false, false, false },
}};

constexpr auto make_route_descriptors() {
    std::array<std::array<RouteDescriptor, weight_route_count>, activation_route_count> matrix{};
    for (size_t activation = 0; activation < activation_route_count; ++activation) {
        for (size_t weight = 0; weight < weight_route_count; ++weight) {
            const auto & base = weight_descriptors[weight];
            const bool known_activation = activation != static_cast<size_t>(ActivationRoute::unknown);
            const bool channel = weight == static_cast<size_t>(WeightRoute::q8_channel_direct) ||
                weight == static_cast<size_t>(WeightRoute::q8_channel_sidecar);
            const bool exsia_or_fp32_channel = channel &&
                (activation == static_cast<size_t>(ActivationRoute::exsia) ||
                 activation == static_cast<size_t>(ActivationRoute::fp32));
            const bool full = known_activation && base.facade_full && !exsia_or_fp32_channel;
            const bool activation_is_sliceable =
                activation == static_cast<size_t>(ActivationRoute::fp32) ||
                activation == static_cast<size_t>(ActivationRoute::exsia) ||
                activation == static_cast<size_t>(ActivationRoute::tensor);
            const bool sequential = full && base.facade_sliced && activation_is_sliceable &&
                (!channel || activation == static_cast<size_t>(ActivationRoute::tensor));
            matrix[activation][weight] = {
                known_activation && base.legacy_full,
                full,
                sequential,
                sequential && activation == static_cast<size_t>(ActivationRoute::exsia),
                base.deprecated,
            };
        }
    }
    return matrix;
}

constexpr auto route_descriptors = make_route_descriptors();

const RouteDescriptor & route_descriptor(const RouteKey & key) {
    return route_descriptors[static_cast<size_t>(key.activation)][static_cast<size_t>(key.weight)];
}

}

RouteKey normalize_route(const ggml_gemmini_args_t & args) {
    RouteKey key{};
    switch (args.tiled_matmul_type) {
        case CPU: key.backend = BackendRoute::cpu; break;
        case WS: key.backend = BackendRoute::gemmini_ws; break;
        case OS: key.backend = BackendRoute::gemmini_os; break;
        default: key.backend = BackendRoute::ws_sim; break;
    }
    switch (args.act_quant.kind()) {
        case quants::act::MetaKind::exsia: key.activation = ActivationRoute::exsia; break;
        case quants::act::MetaKind::tensor: key.activation = ActivationRoute::tensor; break;
        case quants::act::MetaKind::token: key.activation = ActivationRoute::token; break;
        case quants::act::MetaKind::block: key.activation = ActivationRoute::block; break;
        case quants::act::MetaKind::stripe: key.activation = ActivationRoute::stripe; break;
        case quants::act::MetaKind::none: key.activation = ActivationRoute::fp32; break;
    }

    if (args.A_fp32 != nullptr || args.B_fp32 != nullptr) {
        key.activation = ActivationRoute::fp32;
        key.weight = WeightRoute::fp32;
        return key;
    }

    using Format = ggml_gemmini_args_t::im2p_weight_format_t;
    switch (args.weight_format) {
        case Format::q8_0_unpacked_to_h1:
            key.weight = args.weight_i8_scale_active ? WeightRoute::tensor_i8 : WeightRoute::q8_h1;
            break;
        case Format::q8_h0: key.weight = WeightRoute::q8_h0; break;
        case Format::q8_h1: key.weight = WeightRoute::q8_h1; break;
        case Format::q8_hp1: key.weight = WeightRoute::q8_hp1; break;
        case Format::q8_h2: key.weight = WeightRoute::q8_h2; break;
        case Format::q8_hp2: key.weight = WeightRoute::q8_hp2; break;
        case Format::q8_channel: key.weight = WeightRoute::q8_channel_direct; break;
        case Format::q8_channel_dense_sidecar: key.weight = WeightRoute::q8_channel_sidecar; break;
    }
    return key;
}

RouteCapabilities route_capabilities(const ggml_gemmini_args_t & args) {
    const RouteKey key = normalize_route(args);
    const RouteDescriptor & descriptor = route_descriptor(key);
    RouteCapabilities caps{};
    caps.legacy_full = descriptor.legacy_full;
    caps.deprecated = descriptor.deprecated;
    caps.full = descriptor.facade_full;
    caps.sliced_dense = descriptor.facade_sequential;
    caps.sliced_compensation = descriptor.facade_sequential;
    caps.live_stripe_producer = descriptor.facade_pipeline;
    caps.internal_parallel_dense = caps.full;
    if (key.backend == BackendRoute::gemmini_os || key.backend == BackendRoute::ws_sim) {
        caps = {};
        caps.legacy_full = descriptor.legacy_full;
        caps.deprecated = descriptor.deprecated;
    }
    return caps;
}

const char * activation_route_name(ActivationRoute route) {
    switch (route) {
        case ActivationRoute::fp32: return "fp32";
        case ActivationRoute::exsia: return "exsia";
        case ActivationRoute::tensor: return "tensor";
        case ActivationRoute::token: return "token";
        case ActivationRoute::block: return "block";
        case ActivationRoute::stripe: return "stripe";
        case ActivationRoute::unknown: return "unknown";
    }
    return "unknown";
}

const char * weight_route_name(WeightRoute route) {
    switch (route) {
        case WeightRoute::fp32: return "fp32";
        case WeightRoute::tensor_i8: return "tensor_i8";
        case WeightRoute::channel_i8: return "channel_i8";
        case WeightRoute::block_i8: return "block_i8";
        case WeightRoute::q8_h1: return "q8_h1";
        case WeightRoute::q8_hp1: return "q8_hp1";
        case WeightRoute::q8_h2: return "q8_h2";
        case WeightRoute::q8_hp2: return "q8_hp2";
        case WeightRoute::q8_channel_direct: return "q8_channel_direct";
        case WeightRoute::q8_channel_sidecar: return "q8_channel_sidecar";
        case WeightRoute::q8_h0: return "q8_h0";
        case WeightRoute::unknown: return "unknown";
    }
    return "unknown";
}

const char * backend_route_name(BackendRoute route) {
    switch (route) {
        case BackendRoute::cpu: return "cpu";
        case BackendRoute::gemmini_ws: return "gemmini_ws";
        case BackendRoute::gemmini_os: return "gemmini_os";
        case BackendRoute::ws_sim: return "ws_sim";
    }
    return "unknown";
}

std::string pipeline_stripe_summary_json(const char * layer,
                                         size_t I,
                                         size_t J,
                                         size_t K,
                                         const char * backend_route,
                                         const char * schedule,
                                         const MatmulJobMetrics & profile) {
    auto json_string = [](std::ostringstream & out, const char * value) {
        out << '"';
        for (const char * p = value != nullptr ? value : ""; *p != '\0'; ++p) {
            switch (*p) {
                case '\\': out << "\\\\"; break;
                case '"': out << "\\\""; break;
                case '\n': out << "\\n"; break;
                case '\r': out << "\\r"; break;
                case '\t': out << "\\t"; break;
                default: out << *p; break;
            }
        }
        out << '"';
    };
    auto append_u64_array = [](std::ostringstream & out, const auto & values) {
        out << '[';
        for (size_t i = 0; i < values.size(); ++i) {
            if (i != 0) out << ',';
            out << values[i];
        }
        out << ']';
    };
    auto duration_ns = [](uint64_t start_ns, uint64_t end_ns) {
        return end_ns >= start_ns ? end_ns - start_ns : uint64_t{0};
    };
    auto ratio = [](uint64_t numerator, uint64_t denominator) {
        return denominator == 0 ? 0.0 : static_cast<double>(numerator) / static_cast<double>(denominator);
    };

    uint64_t la_service_sum_ns = 0;
    uint64_t la_window_start_ns = 0;
    uint64_t la_window_end_ns = 0;
    for (size_t worker = 0; worker < profile.la_worker_start_ns.size(); ++worker) {
        la_service_sum_ns += duration_ns(profile.la_worker_start_ns[worker], profile.la_worker_end_ns[worker]);
        if (profile.la_worker_start_ns[worker] != 0 &&
            (la_window_start_ns == 0 || profile.la_worker_start_ns[worker] < la_window_start_ns)) {
            la_window_start_ns = profile.la_worker_start_ns[worker];
        }
        la_window_end_ns = std::max(la_window_end_ns, profile.la_worker_end_ns[worker]);
    }
    const uint64_t t_la_ns = duration_ns(la_window_start_ns, la_window_end_ns);
    const uint64_t t_sf_ns = profile.sf_mask_start_ns == 0 || profile.sf_commit_ns == 0 ?
        0 : duration_ns(profile.sf_mask_start_ns, profile.sf_commit_ns);
    const uint64_t t_merge_ns = profile.merge_start_ns == 0 || profile.merge_end_ns == 0 ?
        0 : duration_ns(profile.merge_start_ns, profile.merge_end_ns);
    const double la_efficiency = ratio(la_service_sum_ns, 3 * t_la_ns);
    // A single NPU stream runs dense WS then RMD in the same worker, so the dense stage
    // must always close before the residual stage opens.
    const uint64_t ordering_violation =
        profile.rmd_start_ns != 0 && profile.ws_end_ns > profile.rmd_start_ns ? 1 : 0;

    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\"record_type\":\"PIPELINE_STRIPE_SUMMARY\",\"layer\":";
    json_string(out, layer);
    out << ",\"run_id\":" << profile.run_id
        << ",\"stripe_idx\":" << profile.stripe_id
        << ",\"I\":" << I
        << ",\"J\":" << J
        << ",\"K\":" << K
        << ",\"stripe_rows\":" << (profile.row_end - profile.row_begin)
        << ",\"slot\":" << profile.slot
        << ",\"backend_route\":";
    json_string(out, backend_route);
    out << ",\"schedule\":";
    json_string(out, schedule);
    out << ",\"la_workers\":3,\"sf_workers\":1,\"rmd_workers\":1"
        << ",\"la_worker_body_start_ns\":";
    append_u64_array(out, profile.la_worker_start_ns);
    out << ",\"la_worker_body_end_ns\":";
    append_u64_array(out, profile.la_worker_end_ns);
    out << ",\"sf_mask_start_ns\":" << profile.sf_mask_start_ns
        << ",\"sf_mask_end_ns\":" << profile.sf_mask_end_ns
        << ",\"sf_exponent_start_ns\":" << profile.sf_exponent_start_ns
        << ",\"sf_exponent_end_ns\":" << profile.sf_exponent_end_ns
        << ",\"sf_folding_start_ns\":" << profile.sf_folding_start_ns
        << ",\"sf_folding_end_ns\":" << profile.sf_folding_end_ns
        << ",\"sf_commit_ns\":" << profile.sf_commit_ns
        << ",\"producer_wait_start_ns\":" << profile.producer_wait_start_ns
        << ",\"producer_wait_end_ns\":" << profile.producer_wait_end_ns
        << ",\"matmul_enqueue_ns\":" << profile.capture_queue_enqueue_ns
        << ",\"matmul_start_ns\":" << profile.ws_start_ns
        << ",\"matmul_end_ns\":" << profile.ws_end_ns
        << ",\"rmd_enqueue_ns\":" << profile.rmd_enqueue_ns
        << ",\"rmd_start_ns\":" << profile.rmd_start_ns
        << ",\"rmd_end_ns\":" << profile.rmd_end_ns
        << ",\"merge_start_ns\":" << profile.merge_start_ns
        << ",\"merge_end_ns\":" << profile.merge_end_ns
        // Stage timings, separated per the RMD stripe contract.
        << ",\"T_LA_ns\":" << t_la_ns
        << ",\"T_SF_ns\":" << t_sf_ns
        << ",\"T_RMD_PREP_ns\":"
        << (profile.rmd_decompose.nanoseconds + profile.rmd_index.nanoseconds +
            profile.rmd_pack.nanoseconds)
        << ",\"T_WS_ns\":" << duration_ns(profile.ws_start_ns, profile.ws_end_ns)
        << ",\"T_RMD_NPU_ns\":" << profile.rmd_execute.nanoseconds
        << ",\"T_RMD_COMPOSE_ns\":" << profile.rmd_compose.nanoseconds
        << ",\"T_FINALIZE_ns\":" << profile.rmd_finalize.nanoseconds
        << ",\"T_Merge_ns\":" << t_merge_ns
        << ",\"rmd_decompose_ns\":" << profile.rmd_decompose.nanoseconds
        << ",\"rmd_index_ns\":" << profile.rmd_index.nanoseconds
        << ",\"rmd_pack_ns\":" << profile.rmd_pack.nanoseconds
        << ",\"rmd_queue_ns\":" << profile.rmd_queue.nanoseconds
        << ",\"rmd_execute_ns\":" << profile.rmd_execute.nanoseconds
        << ",\"rmd_output_read_ns\":" << profile.rmd_output_read.nanoseconds
        << ",\"rmd_compose_ns\":" << profile.rmd_compose.nanoseconds
        << ",\"rmd_finalize_ns\":" << profile.rmd_finalize.nanoseconds
        << ",\"la_service_sum_ns\":" << la_service_sum_ns
        << ",\"la_efficiency\":" << la_efficiency
        << ",\"la_service_efficiency\":" << la_efficiency
        << ",\"ordering_violation\":" << ordering_violation
        << ",\"rmd\":{\"active_blocks\":" << profile.rmd.active_blocks
        << ",\"active_lanes\":" << profile.rmd.active_lanes
        << ",\"compact_k_count\":" << profile.rmd.compact_k_count
        << ",\"padded_k_count\":" << profile.rmd.padded_k_count
        << ",\"physical_tile_count\":" << profile.rmd.physical_tile_count
        << ",\"matmul_call_count\":" << profile.rmd.matmul_call_count
        << ",\"stacked_i_tile_count\":" << profile.rmd.stacked_i_tile_count
        << ",\"packet_bytes\":" << profile.rmd.packet_bytes
        << ",\"compressed_output_values\":" << profile.rmd.compressed_output_values
        << ",\"block_padding_zeros\":" << profile.rmd.block_padding_zeros
        << ",\"row_padding_zeros\":" << profile.rmd.row_padding_zeros
        << ",\"j_padding_zeros\":" << profile.rmd.j_padding_zeros
        << "}}";
    return out.str();

}

}

MatMul::MatMul(ggml_gemmini_args_t args) : owned_args_(std::move(args)), args_ptr_(&owned_args_) {}

MatMul::MatMul(ggml_gemmini_args_t * args) : args_ptr_(args) {}

MatMul::MatMul(MatMul && other) noexcept
    : owned_args_(std::move(other.owned_args_)),
      args_ptr_(other.args_ptr_ == &other.owned_args_ ? &owned_args_ : other.args_ptr_),
      first_row_(other.first_row_), last_row_begin_(other.last_row_begin_),
      last_row_end_(other.last_row_end_), covered_rows_(other.covered_rows_),
      has_stripes_(other.has_stripes_), state_(other.state_) {}

MatMul & MatMul::operator=(MatMul && other) noexcept {
    if (this != &other) {
        owned_args_ = std::move(other.owned_args_);
        args_ptr_ = other.args_ptr_ == &other.owned_args_ ? &owned_args_ : other.args_ptr_;
        first_row_ = other.first_row_;
        last_row_begin_ = other.last_row_begin_;
        last_row_end_ = other.last_row_end_;
        covered_rows_ = other.covered_rows_;
        has_stripes_ = other.has_stripes_;
        state_ = other.state_;
    }
    return *this;
}

ggml_gemmini_args_t & MatMul::args() { return *args_ptr_; }
const ggml_gemmini_args_t & MatMul::args() const { return *args_ptr_; }

MatMulResult MatMul::run_dense() {
    if (state_ != MatMulState::idle) {
        return { MatMulStatus::invalid_state, MatMulCapability::supported };
    }
    if (!valid_matmul_shape(args())) {
        return { MatMulStatus::invalid_arguments, MatMulCapability::unsupported };
    }
    if (!valid_activation_metadata(args())) {
        return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
    }
    if (!detail::route_capabilities(args()).full) {
        return { MatMulStatus::unsupported, MatMulCapability::unsupported };
    }
    const auto format = args().weight_format;
    const bool metadata_weight =
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h1 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2;
    if (args().B == nullptr && args().B_fp32 == nullptr && !metadata_weight) {
        return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
    }
    switch (args().weight_format) {
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_channel:
            if (!args().has_q8_channel_direct_read_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar:
            if (!args().has_q8_channel_dense_sidecar_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_h1:
            if (!args().has_q8_h1_im2p_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1:
            if (!args().has_q8_hp1_im2p_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_h2:
            if (!args().has_q8_h2_im2p_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2:
            if (!args().has_q8_hp2_im2p_contract()) {
                return { MatMulStatus::invalid_contract, MatMulCapability::unsupported };
            }
            break;
        default:
            break;
    }

    execute_dense(args());
    return { MatMulStatus::success, MatMulCapability::supported };
}

MatMulResult MatMul::run_full() {
    const MatMulResult dense = run_dense();
    if (dense.status != MatMulStatus::success) {
        return dense;
    }
#if GGML_GEMMINI_ENABLE_RMD
    // FULL mode replays every stripe packet the quantizer produced through the same
    // executor/composer the stripe pipeline uses.
    for (const auto & packet : quants::activation_rmd_packets(args())) {
        if (packet == nullptr) {
            continue;
        }
        if (rmd::apply_rmd_packet(args(), *packet) != rmd::RmdStatus::success) {
            return { MatMulStatus::unsupported, MatMulCapability::unsupported };
        }
    }
#endif
    state_ = MatMulState::completed;
    return { MatMulStatus::success, MatMulCapability::supported };
}

MatMulStatus MatMul::begin_stripes() {
    if (state_ != MatMulState::idle) {
        return MatMulStatus::invalid_state;
    }
    if (!valid_matmul_shape(args())) {
        return MatMulStatus::invalid_arguments;
    }
    const bool live_exsia_metadata = std::holds_alternative<quants::act::exsia::Meta>(
        args().act_quant.storage());
    if (!live_exsia_metadata && !valid_activation_metadata(args())) {
        return MatMulStatus::invalid_contract;
    }
    if (stripe_capability(args()) == MatMulCapability::unsupported) {
        return MatMulStatus::unsupported;
    }
    first_row_ = 0;
    last_row_begin_ = 0;
    last_row_end_ = 0;
    covered_rows_ = 0;
    has_stripes_ = false;
    state_ = MatMulState::accepting_stripes;
    return MatMulStatus::success;
}

MatMulStatus MatMul::run_stripe(MatMulStripe stripe) {
    return run_stripe(stripe, 0);
}

MatMulStatus MatMul::run_stripe(MatMulStripe stripe, size_t stripe_id) {
    if (state_ != MatMulState::accepting_stripes) {
        return MatMulStatus::invalid_state;
    }
    if (stripe.row_begin >= stripe.row_end || stripe.row_end > args().I) {
        return MatMulStatus::malformed_stripe;
    }

    if (has_stripes_) {
        if (last_row_begin_ == stripe.row_begin && last_row_end_ == stripe.row_end) {
            return MatMulStatus::duplicate_stripe;
        }
        if (stripe.row_begin < last_row_end_) {
            return MatMulStatus::overlapping_stripe;
        }
    }

    const MatMulStatus status = execute_stripe(args(), stripe, stripe_id);
    if (status == MatMulStatus::success) {
        if (!has_stripes_) {
            first_row_ = stripe.row_begin;
        }
        last_row_begin_ = stripe.row_begin;
        last_row_end_ = stripe.row_end;
        covered_rows_ += stripe.row_end - stripe.row_begin;
        has_stripes_ = true;
    }
    return status;
}

MatMulStatus MatMul::run_staged_stripe(MatMulStripe stripe, size_t stripe_id) {
    if (state_ != MatMulState::accepting_stripes) {
        return MatMulStatus::invalid_state;
    }
    if (stripe.row_begin >= stripe.row_end || stripe.row_end > args().I) {
        return MatMulStatus::malformed_stripe;
    }
    const MatMulStatus status = execute_stripe(args(), stripe, stripe_id, true);
    if (status == MatMulStatus::success) {
        if (!has_stripes_) {
            first_row_ = stripe.row_begin;
        }
        last_row_begin_ = stripe.row_begin;
        last_row_end_ = stripe.row_end;
        covered_rows_ += stripe.row_end - stripe.row_begin;
        has_stripes_ = true;
    }
    return status;
}

MatMulStatus MatMul::finish_stripes() {
    if (state_ != MatMulState::accepting_stripes) {
        return MatMulStatus::invalid_state;
    }
    if (!has_stripes_) {
        state_ = MatMulState::idle;
        return MatMulStatus::empty_stripes;
    }
    if (first_row_ != 0 || last_row_end_ != args().I || covered_rows_ != args().I) {
        state_ = MatMulState::idle;
        return MatMulStatus::missing_stripes;
    }
    state_ = MatMulState::completed;
    return MatMulStatus::success;
}

MatMulCapability MatMul::stripe_capability(const ggml_gemmini_args_t & args) {
    const auto format = args.weight_format;
    const auto capabilities = detail::route_capabilities(args);
    if (format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2 ||
        !capabilities.sliced_compensation ||
        args.transpose_A || (args.D != nullptr && !args.repeating_bias) ||
        !supports_row_slice_activation(args)) {
        return MatMulCapability::unsupported;
    }
    if (uses_baseline_channel_route(args) &&
        !std::holds_alternative<quants::act::tensor::Meta>(args.act_quant.storage())) {
        return MatMulCapability::unsupported;
    }
    return MatMulCapability::supported;
}

MatMulState MatMul::state() const {
    return state_;
}

MatmulStripeInput::MatmulStripeInput(size_t row_begin, size_t row_end)
    : row_begin_(row_begin), row_end_(row_end), stripe_id_(row_begin),
      residual_(nullptr), residual_count_(0) {}

MatmulStripeInput::MatmulStripeInput(size_t row_begin, size_t row_end, size_t stripe_id,
                                    const int32_t * residual, size_t residual_count)
    : row_begin_(row_begin), row_end_(row_end), stripe_id_(stripe_id),
      residual_(residual), residual_count_(residual_count) {}

size_t MatmulStripeInput::row_begin() const {
    return row_begin_;
}

size_t MatmulStripeInput::row_end() const {
    return row_end_;
}

size_t MatmulStripeInput::stripe_id() const {
    return stripe_id_;
}

const int32_t * MatmulStripeInput::residual() const {
    return residual_;
}

size_t MatmulStripeInput::residual_count() const {
    return residual_count_;
}

MatmulExecution::MatmulExecution(ggml_gemmini_args_t args, MatmulOptions options)
    : total_rows_(args.I), facade_(std::move(args)), options_(options) {
    state_ = MatmulExecutionState::prepared;
    if (options_.mode == MatmulInvocationMode::stripe_pipeline) {
        ggml_gemmini_args_t staged_args = facade_.args();
        staged_args.act_quant.reset();
        staged_facade_ = std::make_unique<MatMul>(std::move(staged_args));
    }
    if (options_.dense_threads > 1) {
        status_ = make_status(MatmulStatusCode::unsupported_invocation,
                              "dense stripe execution has one owner lane");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode != MatmulInvocationMode::full && options_.job_capacity == 0) {
        status_ = make_status(MatmulStatusCode::invalid_argument, "job capacity must be nonzero");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode == MatmulInvocationMode::stripe_pipeline && total_rows_ <= 1) {
        status_ = make_status(MatmulStatusCode::unsupported_invocation,
                              "stripe pipeline requires more than one row");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode == MatmulInvocationMode::stripe_pipeline &&
        !std::holds_alternative<quants::act::NoneMeta>(facade_.args().act_quant.storage()) &&
        !detail::route_capabilities(facade_.args()).live_stripe_producer) {
        status_ = make_status(MatmulStatusCode::unsupported_invocation,
                              "stripe pipeline requires an ExSIA live producer route");
        state_ = MatmulExecutionState::failed;
        return;
    }
    const bool defer_pipeline_route_validation =
        options_.mode == MatmulInvocationMode::stripe_pipeline &&
        std::holds_alternative<quants::act::NoneMeta>(facade_.args().act_quant.storage());
    if ((options_.mode == MatmulInvocationMode::stripe_sequential ||
         options_.mode == MatmulInvocationMode::stripe_pipeline) &&
        !defer_pipeline_route_validation) {
        const MatMulStatus status = facade_.begin_stripes();
        status_ = to_public_status(
            status, status == MatMulStatus::unsupported ? MatMulCapability::unsupported : MatMulCapability::supported,
            &facade_.args());
        if (!status_.ok()) {
            state_ = MatmulExecutionState::failed;
        }
    }
}

MatmulExecution::MatmulExecution()
    : total_rows_(0), facade_(static_cast<ggml_gemmini_args_t *>(nullptr)) {
    status_ = invalid_state("execution is not prepared");
}

MatmulExecution::MatmulExecution(MatmulStatus status)
    : total_rows_(0), facade_(static_cast<ggml_gemmini_args_t *>(nullptr)), status_(status),
      state_(status.ok() ? MatmulExecutionState::prepared : MatmulExecutionState::failed) {}

MatmulExecution::MatmulExecution(MatmulExecution && other) noexcept
    : MatmulExecution() {
    *this = std::move(other);
}

MatmulExecution::MatmulExecution(ggml_gemmini_args_t * args, MatmulOptions options)
    : total_rows_(args != nullptr ? args->I : 0), facade_(args), options_(options) {
    state_ = MatmulExecutionState::prepared;
    if (args == nullptr) {
        status_ = make_status(MatmulStatusCode::invalid_argument, "null execution args");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode == MatmulInvocationMode::stripe_pipeline) {
        ggml_gemmini_args_t staged_args = *args;
        staged_args.act_quant.reset();
        staged_facade_ = std::make_unique<MatMul>(std::move(staged_args));
    }
    if (options_.dense_threads > 1) {
        status_ = make_status(MatmulStatusCode::unsupported_invocation,
                              "dense stripe execution has one owner lane");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode != MatmulInvocationMode::full && options_.job_capacity == 0) {
        status_ = make_status(MatmulStatusCode::invalid_argument, "job capacity must be nonzero");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode == MatmulInvocationMode::stripe_pipeline && total_rows_ <= 1) {
        status_ = make_status(MatmulStatusCode::unsupported_invocation,
                              "stripe pipeline requires more than one row");
        state_ = MatmulExecutionState::failed;
        return;
    }
    if (options_.mode == MatmulInvocationMode::stripe_pipeline &&
        !std::holds_alternative<quants::act::NoneMeta>(facade_.args().act_quant.storage()) &&
        !detail::route_capabilities(facade_.args()).live_stripe_producer) {
        status_ = make_status(MatmulStatusCode::unsupported_invocation,
                              "stripe pipeline requires an ExSIA live producer route");
        state_ = MatmulExecutionState::failed;
        return;
    }
    const bool defer_pipeline_route_validation =
        options_.mode == MatmulInvocationMode::stripe_pipeline &&
        std::holds_alternative<quants::act::NoneMeta>(facade_.args().act_quant.storage());
    if ((options_.mode == MatmulInvocationMode::stripe_sequential ||
         options_.mode == MatmulInvocationMode::stripe_pipeline) &&
        !defer_pipeline_route_validation) {
        const MatMulStatus status = facade_.begin_stripes();
        status_ = to_public_status(
            status, status == MatMulStatus::unsupported ? MatMulCapability::unsupported : MatMulCapability::supported,
            &facade_.args());
        if (!status_.ok()) {
            state_ = MatmulExecutionState::failed;
        }
    }
}

MatmulExecution & MatmulExecution::operator=(MatmulExecution && other) noexcept {
    if (this == &other) {
        return *this;
    }
    assert_pipeline_detached();
    other.assert_pipeline_detached();
    total_rows_ = other.total_rows_;
    facade_ = std::move(other.facade_);
    options_ = other.options_;
    status_ = other.status_;
    state_ = other.state_;
    state_mutex_ = std::move(other.state_mutex_);
    active_jobs_ = other.active_jobs_;
    captured_rows_ = other.captured_rows_;
    finalized_rows_ = other.finalized_rows_;
    first_row_ = other.first_row_;
    last_row_begin_ = other.last_row_begin_;
    last_row_end_ = other.last_row_end_;
    has_captures_ = other.has_captures_;
    captured_stripe_ids_ = std::move(other.captured_stripe_ids_);
    staged_facade_ = std::move(other.staged_facade_);
    staged_metadata_active_ = other.staged_metadata_active_;
    pipeline_attached_ = other.pipeline_attached_;
    other.total_rows_ = 0;
    other.active_jobs_ = 0;
    other.captured_rows_ = 0;
    other.finalized_rows_ = 0;
    other.first_row_ = 0;
    other.last_row_begin_ = 0;
    other.last_row_end_ = 0;
    other.has_captures_ = false;
    other.staged_metadata_active_ = false;
    other.pipeline_attached_ = false;
    return *this;
}

MatmulExecution::~MatmulExecution() {
    assert_pipeline_detached();
}

MatmulInvocationMode MatmulExecution::mode() const {
    return options_.mode;
}

MatmulExecutionState MatmulExecution::state() const {
    std::lock_guard<std::mutex> lock(*state_mutex_);
    return state_;
}

MatmulStatus MatmulExecution::status() const {
    std::lock_guard<std::mutex> lock(*state_mutex_);
    return status_;
}

void MatmulExecution::assert_pipeline_detached() const {
    if (!state_mutex_) {
        return;
    }
    std::lock_guard<std::mutex> lock(*state_mutex_);
    GGML_ASSERT(!pipeline_attached_);
    GGML_ASSERT(active_jobs_ == 0);
}

#if defined(GGML_GEMMINI_TEST_OBSERVER)
bool MatmulExecution::test_pipeline_attached() const {
    std::lock_guard<std::mutex> lock(*state_mutex_);
    return pipeline_attached_;
}
#endif

namespace {
#if defined(GGML_GEMMINI_TEST_OBSERVER)
MatmulStripeCollector * test_residual_failure_collector = nullptr;
MatmulStatus test_residual_failure;
MatmulStripeCollector * test_dense_observer_collector = nullptr;
MatmulDenseState observed_dense_state_at_release = MatmulDenseState::idle;
#endif

bool dense_state_is_terminal(MatmulDenseState state) {
    return state == MatmulDenseState::complete ||
        state == MatmulDenseState::failed ||
        state == MatmulDenseState::cancelled;
}
}

MatmulStripeCollector::MatmulStripeCollector(size_t capacity)
    : capacity_(capacity), sink_{this, &MatmulStripeCollector::on_ready} {
    if (capacity == 0) {
        status_ = make_status(MatmulStatusCode::invalid_argument, "collector capacity must be nonzero");
    }
}

MatmulStripeCollector::~MatmulStripeCollector() {
    finish();
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    std::lock_guard<std::mutex> lock(mutex_);
    if (test_residual_failure_collector == this) {
        test_residual_failure_collector = nullptr;
        test_residual_failure = {};
    }
    if (test_dense_observer_collector == this) {
        test_dense_observer_collector = nullptr;
        observed_dense_state_at_release = MatmulDenseState::idle;
    }
#endif
}

bool MatmulStripeCollector::start(MatmulExecution & execution) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!status_ || worker_started_ || startup_in_progress_ ||
            execution.mode() != MatmulInvocationMode::stripe_pipeline) {
            return false;
        }
        worker_started_ = true;
        startup_in_progress_ = true;
    }
    {
        std::lock_guard<std::mutex> execution_lock(*execution.state_mutex_);
        if (execution.pipeline_attached_) {
            std::lock_guard<std::mutex> lock(mutex_);
            status_ = invalid_state("execution already has a live stripe collector");
            worker_started_ = false;
            startup_in_progress_ = false;
            condition_.notify_all();
            return false;
        }
        execution.pipeline_attached_ = true;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        execution_ = &execution;
        dense_done_ = false;
        stop_requested_ = false;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        test_thread_start_attempts_ = 0;
#endif
    }
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (test_pause_startup_) {
            condition_.notify_all();
            condition_.wait(lock, [this] { return !test_pause_startup_; });
        }
    }
#endif
    const auto fail_start = [&](MatmulStatus failure) {
        std::vector<std::shared_ptr<MatmulStripeJob>> jobs;
        MatmulExecution * attached_execution = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            status_ = failure;
            stop_requested_ = true;
            dense_done_ = true;
            pending_.clear();
            attached_execution = execution_;
            for (const auto & weak_job : jobs_) {
                if (auto job = weak_job.lock()) {
                    jobs.push_back(std::move(job));
                }
            }
            execution_ = nullptr;
            worker_started_ = false;
            startup_in_progress_ = false;
        }
        for (const auto & job : jobs) {
            job->cancel(failure);
            release_in_flight_once(job);
        }
        condition_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        if (attached_execution != nullptr) {
            std::lock_guard<std::mutex> execution_lock(*attached_execution->state_mutex_);
            attached_execution->status_ = failure;
            attached_execution->state_ = MatmulExecutionState::failed;
            attached_execution->pipeline_attached_ = false;
        }
        condition_.notify_all();
        return false;
    };
    try {
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (test_fail_thread_start_attempt_ != 0 &&
                ++test_thread_start_attempts_ == test_fail_thread_start_attempt_) {
                throw std::system_error(
                    std::make_error_code(std::errc::resource_unavailable_try_again),
                    "injected thread start failure");
            }
        }
#endif
        worker_ = std::thread(&MatmulStripeCollector::worker_loop, this);
    } catch (const std::bad_alloc &) {
        return fail_start(make_status(
            MatmulStatusCode::out_of_memory, "collector startup allocation failed"));
    } catch (const std::system_error &) {
        return fail_start(make_status(
            MatmulStatusCode::execution_failure, "collector worker thread creation failed"));
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        startup_in_progress_ = false;
    }
    condition_.notify_all();
    return true;
}

MatmulStatus MatmulStripeCollector::cancel() {
    std::vector<std::shared_ptr<MatmulStripeJob>> jobs;
    MatmulStatus cancelled = make_status(MatmulStatusCode::cancelled, "stripe pipeline cancelled");
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !startup_in_progress_; });
        if (!worker_started_) {
            return make_status(MatmulStatusCode::invalid_state, "stripe pipeline is not running");
        }
        if (status_.ok()) {
            status_ = cancelled;
        } else {
            cancelled = status_;
        }
        stop_requested_ = true;
        pending_.clear();
        for (const auto & weak_job : jobs_) {
            if (auto job = weak_job.lock()) {
                jobs.push_back(std::move(job));
            }
        }
    }
    for (const auto & job : jobs) {
        job->cancel(cancelled);
        release_in_flight_once(job);
    }
    condition_.notify_all();
    return cancelled;
}

void MatmulStripeCollector::fail(MatmulStatus failure) {
    std::vector<std::shared_ptr<MatmulStripeJob>> jobs;
    MatmulExecution * execution = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (status_.ok()) {
            status_ = failure;
        } else {
            failure = status_;
        }
        stop_requested_ = true;
        pending_.clear();
        for (const auto & weak_job : jobs_) {
            if (auto job = weak_job.lock()) {
                jobs.push_back(std::move(job));
            }
        }
        execution = execution_;
    }
    for (const auto & job : jobs) {
        job->cancel(failure);
        release_in_flight_once(job);
    }
    if (execution != nullptr) {
        std::lock_guard<std::mutex> execution_lock(*execution->state_mutex_);
        execution->status_ = failure;
        execution->state_ = MatmulExecutionState::failed;
    }
    condition_.notify_all();
}

void MatmulStripeCollector::release_in_flight_once(
        const std::shared_ptr<MatmulStripeJob> & job) {
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    const MatmulDenseState dense_state = job->snapshot().dense;
#endif
    std::lock_guard<std::mutex> lock(mutex_);
    if (job->collector_slot_released_) {
        return;
    }
    job->collector_slot_released_ = true;
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    if (test_dense_observer_collector == this) {
        observed_dense_state_at_release = dense_state;
    }
#endif
    --in_flight_;
}

MatmulStatus MatmulStripeCollector::finish() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !startup_in_progress_; });
        if (!worker_started_) {
            return status_;
        }
        stop_requested_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dense_done_ = true;
    }
    condition_.notify_all();
    MatmulExecution * execution = nullptr;
    MatmulStatus status = {};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        execution = execution_;
        status = status_;
        execution_ = nullptr;
        worker_started_ = false;
    }
    if (execution != nullptr) {
        std::lock_guard<std::mutex> execution_lock(*execution->state_mutex_);
        if (!status) {
            execution->status_ = status;
            execution->state_ = MatmulExecutionState::failed;
        }
        execution->pipeline_attached_ = false;
    }
    return status;
}

// One NPU stream: dense WS and RMD run back to back in this worker, in that order.
void MatmulStripeCollector::worker_loop() {
    try {
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (test_thread_exception_ == MatmulCollectorThread::worker) {
                const auto failure = test_thread_exception_failure_;
                test_thread_exception_.reset();
                if (failure == MatmulCollectorThreadFailure::out_of_memory) {
                    throw std::bad_alloc();
                }
                throw std::runtime_error("injected worker thread failure");
            }
        }
#endif
        for (;;) {
            CapturedStripe captured{};
            MatmulExecution * execution = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] {
                    return stop_requested_ || (!pending_.empty() && in_flight_ < capacity_);
                });
                if (pending_.empty()) {
                    break;
                }
                execution = execution_;
                captured = std::move(pending_.front());
                pending_.pop_front();
                ++in_flight_;
                condition_.notify_all();
            }

            std::shared_ptr<MatmulStripeJob> job;
            try {
                job = std::make_shared<MatmulStripeJob>(capture_stripe(
                    *execution,
                    MatmulStripeInput(captured.row_begin, captured.row_end, captured.stripe_id),
                    std::move(captured.rmd_packet)));
            } catch (const std::exception &) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    --in_flight_;
                }
                condition_.notify_all();
                throw;
            }
            {
                std::lock_guard<std::mutex> job_lock(*job->job_mutex_);
                job->metrics_.run_id = captured.run_id;
                job->metrics_.slot = captured.slot;
                job->metrics_.la_cycles = captured.la_cycles;
                job->metrics_.la3_cycles = captured.la3_cycles;
                job->metrics_.sf_cycles = captured.sf_cycles;
                job->metrics_.la3_ns = captured.la3_ns;
                job->metrics_.sf1_ns = captured.sf1_ns;
                job->metrics_.la_worker_start_ns = captured.la_worker_start_ns;
                job->metrics_.la_worker_end_ns = captured.la_worker_end_ns;
                job->metrics_.sf_mask_start_ns = captured.sf_mask_start_ns;
                job->metrics_.sf_mask_end_ns = captured.sf_mask_end_ns;
                job->metrics_.sf_exponent_start_ns = captured.sf_exponent_start_ns;
                job->metrics_.sf_exponent_end_ns = captured.sf_exponent_end_ns;
                job->metrics_.sf_folding_start_ns = captured.sf_folding_start_ns;
                job->metrics_.sf_folding_end_ns = captured.sf_folding_end_ns;
                job->metrics_.sf_commit_ns = captured.sf_commit_ns;
                job->metrics_.la.nanoseconds = captured.la3_ns;
                job->metrics_.la.count = captured.la3_ns != 0 ? 1 : 0;
                job->metrics_.sf.nanoseconds = captured.sf1_ns;
                job->metrics_.sf.count = captured.sf1_ns != 0 ? 1 : 0;
                job->metrics_.capture_copy = captured.capture_copy;
                job->metrics_.producer_wait = captured.producer_wait;
                job->metrics_.queue_insert = captured.queue_insert;
                job->metrics_.producer_wait_start_ns = captured.producer_wait_start_ns;
                job->metrics_.producer_wait_end_ns = captured.producer_wait_end_ns;
                job->metrics_.capture_queue_enqueue_ns = captured.queued_ns;
                job->metrics_.rmd_pack = captured.rmd_pack;
                job->metrics_.sf_handoff.nanoseconds =
                    captured.sf1_ns + job->metrics_.handoff.nanoseconds;
                job->metrics_.sf_handoff.count = 1;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                jobs_.push_back(job);
            }
            MatmulStatus status = job->status();
            if (!status) {
                release_in_flight_once(job);
                fail(status);
                break;
            }

            {
                std::lock_guard<std::mutex> job_lock(*job->job_mutex_);
                job->metrics_.ws_start_ns = now_ns();
                job->rmd_queued_ns_ = captured.queued_ns;
                job->metrics_.rmd_enqueue_ns = captured.queued_ns;
                if (job->execution_->options_.profiling) {
                    job->metrics_.ws_queue.nanoseconds =
                        job->metrics_.ws_start_ns - captured.queued_ns;
                    job->metrics_.ws_queue.count = 1;
                }
            }
#if defined(GGML_GEMMINI_TEST_OBSERVER)
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return !test_pause_dense_ || stop_requested_; });
            }
#endif
            MatmulStatus collector_status = this->status();
            if (!collector_status) {
                job->cancel(collector_status);
                release_in_flight_once(job);
                break;
            }

            status = execute_dense_stripe(*job);
#if defined(GGML_GEMMINI_TEST_OBSERVER)
            if (status && test_residual_failure_collector == this && !test_residual_failure) {
                status = test_residual_failure;
                job->record_failure(status, false);
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    test_residual_failure_observed_ = true;
                }
                condition_.notify_all();
            }
#endif
            if (status) status = execute_rmd_stripe(*job);
            if (status) status = compose_rmd_stripe(*job);
            if (status) status = finalize_stripe(*job);
            if (status) {
                const MatmulJobMetrics profile = job->metrics();
                std::lock_guard<std::mutex> lock(mutex_);
                profiles_.push_back(profile);
            }
            release_in_flight_once(job);
            condition_.notify_all();
            if (!status) {
                fail(status);
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            dense_done_ = true;
        }
        condition_.notify_all();
    } catch (const std::bad_alloc &) {
        fail(make_status(MatmulStatusCode::out_of_memory, "collector worker thread allocation failed"));
    } catch (const std::exception &) {
        fail(make_status(MatmulStatusCode::execution_failure, "collector worker thread failed"));
    }
}

const quants::act::exsia::StripeReadySink * MatmulStripeCollector::sink() const {
    return &sink_;
}

MatmulStatus MatmulStripeCollector::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

MatmulCollectorSnapshot MatmulStripeCollector::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {status_, capacity_, pending_.size(), in_flight_, worker_started_};
}

std::vector<MatmulJobMetrics> MatmulStripeCollector::profiles() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return profiles_;
}

rmd::StripePacketHandle MatmulStripeCollector::captured_packet(size_t stripe) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stripes_.at(stripe).rmd_packet;
}

#if defined(GGML_GEMMINI_TEST_OBSERVER)
void MatmulStripeCollector::test_inject_residual_failure(MatmulStatus failure) {
    std::lock_guard<std::mutex> lock(mutex_);
    test_residual_failure_collector = this;
    test_residual_failure = failure;
    test_dense_observer_collector = this;
    observed_dense_state_at_release = MatmulDenseState::idle;
}

void MatmulStripeCollector::test_inject_thread_start_failure(size_t attempt) {
    std::lock_guard<std::mutex> lock(mutex_);
    test_fail_thread_start_attempt_ = attempt;
}

void MatmulStripeCollector::test_inject_thread_exception(
        MatmulCollectorThread thread, MatmulCollectorThreadFailure failure) {
    std::lock_guard<std::mutex> lock(mutex_);
    test_thread_exception_ = thread;
    test_thread_exception_failure_ = failure;
}

void MatmulStripeCollector::test_pause_dense_before_execute() {
    std::lock_guard<std::mutex> lock(mutex_);
    test_pause_dense_ = true;
}

void MatmulStripeCollector::test_pause_startup_after_attachment() {
    std::lock_guard<std::mutex> lock(mutex_);
    test_pause_startup_ = true;
}

void MatmulStripeCollector::test_resume_startup() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        test_pause_startup_ = false;
    }
    condition_.notify_all();
}

void MatmulStripeCollector::test_wait_for_residual_failure() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return test_residual_failure_observed_; });
}

size_t MatmulStripeCollector::test_in_flight() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return in_flight_;
}

MatmulDenseState MatmulStripeCollector::test_dense_state_at_release() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return test_dense_observer_collector == this ?
        observed_dense_state_at_release : MatmulDenseState::idle;
}

#endif

bool MatmulStripeCollector::on_ready(
        void * user_data, const quants::act::exsia::StripeReadyEvent & event) {
    auto & collector = *static_cast<MatmulStripeCollector *>(user_data);
    if (event.row_begin >= event.row_end) {
        {
            std::lock_guard<std::mutex> lock(collector.mutex_);
            collector.status_ = make_status(MatmulStatusCode::invalid_argument, "invalid stripe event");
            collector.stop_requested_ = true;
        }
        collector.condition_.notify_all();
        return false;
    }
    const auto make_captured = [&event](MatmulStageMetrics capture_copy,
                                        MatmulStageMetrics producer_wait,
                                        uint64_t producer_wait_start_ns,
                                        uint64_t producer_wait_end_ns) {
        CapturedStripe captured{};
        captured.run_id = event.run_id;
        captured.stripe_id = event.stripe_id;
        captured.slot = event.slot;
        captured.row_begin = event.row_begin;
        captured.row_end = event.row_end;
        // Shared handle only: the packet owns its buffers, so it outlives the ExSIA slot.
        captured.rmd_packet = event.rmd_packet;
        captured.la_cycles = event.local_end_cycle >= event.local_start_cycle ?
            event.local_end_cycle - event.local_start_cycle : 0;
        captured.la3_cycles = event.local_group3_end_cycle >= event.local_group3_start_cycle ?
            event.local_group3_end_cycle - event.local_group3_start_cycle : 0;
        captured.sf_cycles = event.folding_end_cycle >= event.folding_start_cycle ?
            event.folding_end_cycle - event.folding_start_cycle : 0;
        captured.la3_ns = event.local_end_ns >= event.local_start_ns ?
            event.local_end_ns - event.local_start_ns : 0;
        captured.sf1_ns = event.folding_end_ns >= event.folding_start_ns ?
            event.folding_end_ns - event.folding_start_ns : 0;
        captured.la_worker_start_ns = event.local_worker_start_ns;
        captured.la_worker_end_ns = event.local_worker_end_ns;
        captured.sf_mask_start_ns = event.mask_assembly_start_ns;
        captured.sf_mask_end_ns = event.mask_assembly_end_ns;
        captured.sf_exponent_start_ns = event.exponent_reduction_start_ns;
        captured.sf_exponent_end_ns = event.exponent_reduction_end_ns;
        captured.sf_folding_start_ns = event.folding_start_ns;
        captured.sf_folding_end_ns = event.folding_end_ns;
        captured.sf_commit_ns = event.folding_commit_ns;
        captured.capture_copy = capture_copy;
        captured.producer_wait = producer_wait;
        captured.producer_wait_start_ns = producer_wait_start_ns;
        captured.producer_wait_end_ns = producer_wait_end_ns;
        if (event.rmd_packet != nullptr) {
            captured.rmd_pack.nanoseconds = event.rmd_pack_ns;
            captured.rmd_pack.count = 1;
        }
        return captured;
    };
    try {
        const auto copy_start = Clock::now();
        MatmulStageMetrics capture_copy;
        record_metric(capture_copy, true, copy_start);
        std::unique_lock<std::mutex> lock(collector.mutex_);
        if (!collector.status_ || collector.stop_requested_) {
            return false;
        }
        if (collector.worker_started_) {
            const auto wait_start = Clock::now();
            const uint64_t producer_wait_start_ns = now_ns();
            collector.condition_.wait(lock, [&collector] {
                return collector.stop_requested_ || !collector.status_ ||
                    collector.pending_.size() + collector.in_flight_ < collector.capacity_;
            });
            if (collector.stop_requested_ || !collector.status_) {
                return false;
            }
            MatmulStageMetrics producer_wait;
            record_metric(producer_wait, true, wait_start);
            CapturedStripe captured = make_captured(
                capture_copy, producer_wait, producer_wait_start_ns, now_ns());
            const auto insert_start = Clock::now();
            collector.pending_.push_back(std::move(captured));
            record_metric(collector.pending_.back().queue_insert, true, insert_start);
            collector.pending_.back().queued_ns = now_ns();
            lock.unlock();
            collector.condition_.notify_all();
            return true;
        }
        if (collector.stripes_.size() >= collector.capacity_) {
            collector.status_ = make_status(MatmulStatusCode::out_of_memory, "collector capacity exhausted");
            collector.stop_requested_ = true;
            lock.unlock();
            collector.condition_.notify_all();
            return false;
        }
        CapturedStripe captured = make_captured(capture_copy, {}, 0, 0);
        const auto insert_start = Clock::now();
        collector.stripes_.push_back(std::move(captured));
        record_metric(collector.stripes_.back().queue_insert, true, insert_start);
        collector.stripes_.back().queued_ns = now_ns();
    } catch (const std::bad_alloc &) {
        {
            std::lock_guard<std::mutex> lock(collector.mutex_);
            collector.status_ = make_status(MatmulStatusCode::out_of_memory, "stripe capture allocation failed");
            collector.stop_requested_ = true;
        }
        collector.condition_.notify_all();
        return false;
    }
    return true;
}

MatmulStripeJob::MatmulStripeJob(
        MatmulExecution * execution, MatmulStripeInput input, MatmulStatus status,
        rmd::StripePacketHandle rmd_packet)
    : execution_(execution), input_(std::move(input)), status_(status),
      rmd_packet_(std::move(rmd_packet)),
      captured_(status.ok()) {}

MatmulStripeJob::MatmulStripeJob()
    : execution_(nullptr), input_(0, 0), status_(invalid_state("job is not captured")), captured_(false) {}

MatmulStripeJob::MatmulStripeJob(MatmulStripeJob && other) noexcept
    : execution_(other.execution_), input_(std::move(other.input_)), status_(other.status_),
      metrics_(other.metrics_),
      staged_activation_meta_(std::move(other.staged_activation_meta_)),
      owns_slot_(other.owns_slot_),
      released_(other.released_), collector_slot_released_(other.collector_slot_released_),
      rmd_queued_ns_(other.rmd_queued_ns_), job_mutex_(std::move(other.job_mutex_)),
      rmd_packet_(std::move(other.rmd_packet_)),
      rmd_output_(std::move(other.rmd_output_)),
      rmd_correction_(std::move(other.rmd_correction_)),
      dense_state_(other.dense_state_), residual_state_(other.residual_state_),
      captured_(other.captured_), finalized_(other.finalized_) {
    other.execution_ = nullptr;
    other.owns_slot_ = false;
    other.released_ = true;
    other.collector_slot_released_ = true;
}

MatmulStripeJob & MatmulStripeJob::operator=(MatmulStripeJob && other) noexcept {
    if (this != &other) {
        release_slot();
        execution_ = other.execution_;
        input_ = std::move(other.input_);
        status_ = other.status_;
        metrics_ = other.metrics_;
        staged_activation_meta_ = std::move(other.staged_activation_meta_);
        rmd_packet_ = std::move(other.rmd_packet_);
        rmd_output_ = std::move(other.rmd_output_);
        rmd_correction_ = std::move(other.rmd_correction_);
        owns_slot_ = other.owns_slot_;
        released_ = other.released_;
        collector_slot_released_ = other.collector_slot_released_;
        rmd_queued_ns_ = other.rmd_queued_ns_;
        dense_state_ = other.dense_state_;
        residual_state_ = other.residual_state_;
        captured_ = other.captured_;
        finalized_ = other.finalized_;
        job_mutex_ = std::move(other.job_mutex_);
        other.execution_ = nullptr;
        other.owns_slot_ = false;
        other.released_ = true;
        other.collector_slot_released_ = true;
    }
    return *this;
}

MatmulStripeJob::~MatmulStripeJob() {
    release_slot();
}

void MatmulStripeJob::release_slot() {
    if (!owns_slot_ || released_ || execution_ == nullptr || job_mutex_ == nullptr) {
        return;
    }
    MatmulExecution * execution = nullptr;
    {
        std::lock_guard<std::mutex> lock(*job_mutex_);
        if (!owns_slot_ || released_ || execution_ == nullptr) {
            return;
        }
        owns_slot_ = false;
        released_ = true;
        execution = execution_;
    }
    std::lock_guard<std::mutex> state_lock(*execution->state_mutex_);
    if (execution->active_jobs_ != 0) {
        --execution->active_jobs_;
    }
}

void MatmulStripeJob::cancel(MatmulStatus status) {
    {
        std::lock_guard<std::mutex> lock(*job_mutex_);
        if (finalized_) {
            return;
        }
        if (status_.ok()) {
            status_ = status;
        }
        if (dense_state_ != MatmulDenseState::complete && dense_state_ != MatmulDenseState::failed) {
            dense_state_ = MatmulDenseState::cancelled;
        }
        if (residual_state_ != MatmulResidualState::complete &&
            residual_state_ != MatmulResidualState::failed) {
            residual_state_ = MatmulResidualState::cancelled;
        }
    }
    lifecycle_condition_.notify_all();
}

void MatmulStripeJob::record_failure(MatmulStatus status, bool dense_branch) {
    {
        std::lock_guard<std::mutex> lock(*job_mutex_);
        if (status_.ok()) {
            status_ = status;
            if (dense_branch) {
                dense_state_ = MatmulDenseState::failed;
            } else {
                residual_state_ = MatmulResidualState::failed;
            }
        }
    }
    lifecycle_condition_.notify_all();
}

MatmulStatus MatmulStripeJob::status() const {
    std::lock_guard<std::mutex> lock(*job_mutex_);
    return status_;
}

MatmulJobMetrics MatmulStripeJob::metrics() const {
    std::lock_guard<std::mutex> lock(*job_mutex_);
    return metrics_;
}

MatmulStripeJobSnapshot MatmulStripeJob::snapshot() const {
    std::lock_guard<std::mutex> lock(*job_mutex_);
    return {status_, metrics_, dense_state_, residual_state_, captured_, finalized_, released_};
}

MatmulExecution prepare_execution(const ggml_gemmini_args_t & args, MatmulOptions options) {
    return MatmulExecution(args, options);
}

MatmulExecution prepare_execution(ggml_gemmini_args_t * args, MatmulOptions options) {
    return MatmulExecution(args, options);
}

MatmulStatus prepare_execution(ggml_gemmini_args_t & args, const MatmulOptions & options,
                               MatmulExecution & execution) {
    execution = MatmulExecution(&args, options);
    return execution.status();
}

MatmulStatus execute_full(MatmulExecution & execution) {
    if (execution.options_.mode != MatmulInvocationMode::full) {
        execution.status_ = make_status(
            MatmulStatusCode::unsupported_invocation, "full execution requires full mode");
        execution.state_ = MatmulExecutionState::failed;
        return execution.status_;
    }
    const MatMulResult result = execution.facade_.run_full();
    execution.status_ = to_public_status(result.status, result.capability, &execution.facade_.args());
    if (execution.status_ && execution.options_.validation &&
        !finite_output(execution.facade_.args())) {
        execution.status_ = make_status(MatmulStatusCode::execution_failure,
                                        "output validation failed");
    }
    execution.state_ = execution.status_.ok() ? MatmulExecutionState::completed : MatmulExecutionState::failed;
    return execution.status_;
}

MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input) {
    if (input.residual_count() != 0 && input.residual() == nullptr) {
        return MatmulStripeJob(
            &execution,
            std::move(input),
            make_status(MatmulStatusCode::invalid_argument, "null stripe capture payload"));
    }
    if (input.residual_count() != 0) {
        const size_t rows = input.row_end() > input.row_begin() ?
            input.row_end() - input.row_begin() : 0;
        if (rows == 0 || execution.facade_.args().K == 0 ||
            rows > std::numeric_limits<size_t>::max() / execution.facade_.args().K ||
            input.residual_count() != rows * execution.facade_.args().K) {
            return MatmulStripeJob(
                &execution,
                std::move(input),
                make_status(MatmulStatusCode::invalid_argument,
                            "invalid dense residual cardinality"));
        }
        return MatmulStripeJob(
            &execution,
            std::move(input),
            make_status(MatmulStatusCode::unsupported_route,
                        "raw residual stripe payload is unsupported",
                        MatMulCapability::unsupported));
    }
#if GGML_GEMMINI_ENABLE_RMD
    // No live packet was handed over (sequential stripe mode): rebuild one for exactly
    // this row range out of the packets the quantizer produced.
    const size_t row_begin = input.row_begin();
    const size_t row_end = input.row_end();
    const size_t stripe_id = input.stripe_id();
    rmd::RmdStatus slice_status = rmd::RmdStatus::success;
    rmd::StripePacketHandle packet;
    try {
        packet = rmd::slice_packets(
            quants::activation_rmd_packets(execution.facade_.args()),
            row_begin, row_end, stripe_id, slice_status);
    } catch (const std::bad_alloc &) {
        return MatmulStripeJob(
            &execution, std::move(input),
            make_status(MatmulStatusCode::out_of_memory, "stripe capture allocation failed"));
    }
    if (slice_status != rmd::RmdStatus::success) {
        return MatmulStripeJob(&execution, std::move(input), from_rmd_status(slice_status));
    }
    return capture_stripe(execution, std::move(input), std::move(packet));
#else
    return capture_stripe(execution, std::move(input), nullptr);
#endif
}

MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input,
                               rmd::StripePacketHandle rmd_packet) {
    const auto start = Clock::now();
    MatmulStatus status{};
    std::lock_guard<std::mutex> state_lock(*execution.state_mutex_);
    if (!execution.status_.ok()) {
        status = execution.status_;
    } else if (execution.options_.mode == MatmulInvocationMode::full) {
        status = make_status(MatmulStatusCode::unsupported_invocation, "stripe capture requires stripe mode");
    } else if (execution.options_.mode == MatmulInvocationMode::stripe_pipeline &&
               execution.facade_.state() == MatMulState::idle) {
        const MatMulStatus begin_status = execution.facade_.begin_stripes();
        if (begin_status != MatMulStatus::success) {
            status = to_public_status(
                begin_status,
                begin_status == MatMulStatus::unsupported ? MatMulCapability::unsupported :
                    MatMulCapability::supported);
        }
    } else if (execution.facade_.state() != MatMulState::accepting_stripes) {
        status = invalid_state("execution is not accepting stripes");
    } else if (input.row_begin() >= input.row_end() || input.row_end() > execution.total_rows_ ||
               input.stripe_id() >= execution.total_rows_ ||
               ((input.residual() == nullptr) != (input.residual_count() == 0))) {
        status = make_status(MatmulStatusCode::invalid_argument, "invalid stripe input or id");
    } else if (execution.captured_stripe_ids_.find(input.stripe_id()) !=
               execution.captured_stripe_ids_.end()) {
        status = invalid_contract("duplicate stripe id");
    } else if (execution.active_jobs_ >= execution.options_.job_capacity) {
        status = make_status(MatmulStatusCode::out_of_memory, "job capacity exhausted");
    } else if (execution.has_captures_ && input.row_begin() == execution.last_row_begin_ &&
               input.row_end() == execution.last_row_end_) {
        status = invalid_contract("duplicate stripe");
    } else if (execution.has_captures_ && input.row_begin() < execution.last_row_end_) {
        status = invalid_contract("overlapping stripe");
    } else if (rmd_packet != nullptr &&
               (rmd_packet->row_begin != input.row_begin() ||
                rmd_packet->row_count != input.row_end() - input.row_begin())) {
        status = invalid_contract("rmd packet does not cover the stripe rows");
    }

    MatmulStripeJob job(&execution, std::move(input), status, std::move(rmd_packet));
    if (job.status_.ok()) {
        if (!execution.has_captures_) {
            execution.first_row_ = job.input_.row_begin();
        }
        execution.last_row_begin_ = job.input_.row_begin();
        execution.last_row_end_ = job.input_.row_end();
        execution.captured_rows_ += job.input_.row_end() - job.input_.row_begin();
        execution.captured_stripe_ids_.insert(job.input_.stripe_id());
        execution.has_captures_ = true;
        ++execution.active_jobs_;
        job.owns_slot_ = true;
        if (execution.state_ == MatmulExecutionState::prepared) {
            execution.state_ = MatmulExecutionState::running;
        }
        record_metric(job.metrics_.handoff, execution.options_.profiling, start);
    }
    return job;
}

MatmulStatus capture_stripe(MatmulExecution & execution, const MatmulStripeInput & input,
                            MatmulStripeJob & job) {
    MatmulStripeJob captured = capture_stripe(execution, MatmulStripeInput(
        input.row_begin(), input.row_end(), input.stripe_id(),
        input.residual(), input.residual_count()));
    job = std::move(captured);
    return job.status();
}

MatmulStatus execute_dense_stripe(MatmulStripeJob & job) {
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ ||
            job.dense_state_ != MatmulDenseState::idle) {
            return invalid_state("dense execution requires captured Dense idle state");
        }
        job.dense_state_ = MatmulDenseState::running;
    }
    const auto start = Clock::now();
    const MatMulStatus status = job.execution_->facade_.run_stripe(
        { job.input_.row_begin(), job.input_.row_end() }, job.input_.stripe_id());
    const MatmulStatus dense_status = to_public_status(
        status, status == MatMulStatus::unsupported ? MatMulCapability::unsupported : MatMulCapability::supported,
        &job.execution_->facade_.args());
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        job.metrics_.ws_end_ns = now_ns();
    }
    if (!dense_status) {
        job.record_failure(dense_status, true);
        return dense_status;
    }
    MatmulStatus result;
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        job.dense_state_ = MatmulDenseState::complete;
        record_metric(job.metrics_.ws, job.execution_->options_.profiling, start);
        job.metrics_.ws_service = job.metrics_.ws;
        result = dense_status;
    }
    job.lifecycle_condition_.notify_all();
    return result;
}

// Executes the stripe's RMD packet on the NPU stream and produces the canonical
// block-scaled INT64 compressed output.
MatmulStatus execute_rmd_stripe(MatmulStripeJob & job) {
    if (job.job_mutex_ == nullptr) {
        return invalid_state("residual state unavailable");
    }
    rmd::StripePacketHandle packet;
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ || !job.status_ ||
            job.residual_state_ != MatmulResidualState::idle) {
            return invalid_state("residual execution requires captured idle state");
        }
        packet = job.rmd_packet_;
        job.residual_state_ = MatmulResidualState::running;
        job.metrics_.rmd_start_ns = now_ns();
        if (job.execution_->options_.profiling && job.rmd_queued_ns_ != 0) {
            job.metrics_.rmd_queue.nanoseconds = job.metrics_.rmd_start_ns - job.rmd_queued_ns_;
            job.metrics_.rmd_queue.count = 1;
        }
    }
    if (packet == nullptr) {
        // Empty residual: nothing to execute, output stays unchanged.
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        job.rmd_output_ = {};
        job.rmd_correction_.clear();
        job.residual_state_ = MatmulResidualState::complete;
        job.lifecycle_condition_.notify_all();
        return {};
    }

    const auto start = Clock::now();
    rmd::CompressedOutput output;
    rmd::RmdExecutionMetrics metrics{};
    const rmd::RmdStatus status = rmd::execute_rmd_stripe(
        job.execution_->facade_.args(), *packet, output, &metrics);
    if (status != rmd::RmdStatus::success) {
        const MatmulStatus failure = from_rmd_status(status);
        job.record_failure(failure, false);
        return failure;
    }
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (!job.status_) {
            return job.status_;
        }
        job.rmd_output_ = std::move(output);
        job.metrics_.rmd = metrics;
        record_metric(job.metrics_.rmd_execute, job.execution_->options_.profiling, start);
    }
    job.lifecycle_condition_.notify_all();
    return {};
}

// Reads the canonical compressed output and performs the radix composition.
MatmulStatus compose_rmd_stripe(MatmulStripeJob & job) {
    if (job.job_mutex_ == nullptr) {
        return invalid_state("residual state unavailable");
    }
    rmd::StripePacketHandle packet;
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ || !job.status_ ||
            (job.residual_state_ != MatmulResidualState::running &&
             job.residual_state_ != MatmulResidualState::complete)) {
            return invalid_state("compose requires an executed residual stripe");
        }
        packet = job.rmd_packet_;
        if (packet == nullptr) {
            job.residual_state_ = MatmulResidualState::complete;
            return {};
        }
    }

    const auto read_start = Clock::now();
    std::vector<rmd::OutputValue> correction;
    const rmd::RmdStatus status = rmd::compose_rmd_output(*packet, job.rmd_output_, correction);
    if (status != rmd::RmdStatus::success) {
        const MatmulStatus failure = from_rmd_status(status);
        job.record_failure(failure, false);
        return failure;
    }
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (!job.status_) {
            return job.status_;
        }
        job.rmd_correction_ = std::move(correction);
        record_metric(job.metrics_.rmd_compose, job.execution_->options_.profiling, read_start);
        job.metrics_.rmd_output_read = job.metrics_.rmd_compose;
        job.residual_state_ = MatmulResidualState::complete;
        job.metrics_.rmd_end_ns = now_ns();
    }
    job.lifecycle_condition_.notify_all();
    return {};
}

MatmulStatus finalize_stripe(MatmulStripeJob & job) {
    const auto start = Clock::now();
    MatmulStatus merge_failure{};
    {
        std::lock_guard<std::mutex> lock(*job.job_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_) {
            return invalid_state("stripe is not finalizable");
        }
        if (!job.status_) {
            return job.status_;
        }
        if (job.dense_state_ != MatmulDenseState::complete ||
            job.residual_state_ != MatmulResidualState::complete) {
            return invalid_state("finalize requires dense and residual completion");
        }
        job.finalized_ = true;
        job.metrics_.stripe_id = job.input_.stripe_id();
        job.metrics_.row_begin = job.input_.row_begin();
        job.metrics_.row_end = job.input_.row_end();
        if (job.rmd_packet_ != nullptr && !job.rmd_correction_.empty()) {
            job.metrics_.merge_start_ns = now_ns();
            const rmd::RmdStatus status = rmd::merge_rmd_correction(
                job.execution_->facade_.args(), *job.rmd_packet_, job.rmd_correction_);
            job.metrics_.merge_end_ns = now_ns();
            if (status != rmd::RmdStatus::success) {
                merge_failure = from_rmd_status(status);
            }
        }
        record_metric(job.metrics_.rmd_finalize, job.execution_->options_.profiling, start);
    }
    if (!merge_failure.ok()) {
        job.record_failure(merge_failure, false);
        return merge_failure;
    }
    {
        std::lock_guard<std::mutex> state_lock(*job.execution_->state_mutex_);
        job.execution_->finalized_rows_ += job.input_.row_end() - job.input_.row_begin();
    }
    job.release_slot();
    job.lifecycle_condition_.notify_all();
    return {};
}

MatmulStatus finish_execution(MatmulExecution & execution) {
    std::lock_guard<std::mutex> state_lock(*execution.state_mutex_);
    if (!execution.status_.ok()) {
        execution.state_ = MatmulExecutionState::failed;
        return execution.status_;
    }
    if (execution.options_.mode == MatmulInvocationMode::full) {
        return execution.facade_.state() == MatMulState::completed ? execution.status_ : invalid_state();
    }
    if (execution.pipeline_attached_) {
        return invalid_state("cannot finish while stripe collector is attached");
    }
    execution.state_ = MatmulExecutionState::finishing;
    if (execution.active_jobs_ != 0) {
        execution.state_ = MatmulExecutionState::running;
        return invalid_state("cannot finish with live jobs");
    }
    if (!execution.has_captures_ || execution.first_row_ != 0 ||
        execution.last_row_end_ != execution.total_rows_ ||
        execution.captured_rows_ != execution.total_rows_ ||
        execution.finalized_rows_ != execution.total_rows_) {
        execution.state_ = MatmulExecutionState::running;
        return invalid_contract("missing stripes");
    }
    MatMul * dense_facade = execution.staged_metadata_active_ && execution.staged_facade_ != nullptr ?
        execution.staged_facade_.get() : &execution.facade_;
    const MatMulStatus status = dense_facade->finish_stripes();
    execution.status_ = to_public_status(status, MatMulCapability::supported);
    if (execution.status_ && execution.options_.validation &&
        !finite_output(execution.facade_.args())) {
        execution.status_ = make_status(MatmulStatusCode::execution_failure,
                                        "output validation failed");
    }
    execution.state_ = execution.status_.ok() ? MatmulExecutionState::completed : MatmulExecutionState::failed;
    return execution.status_;
}

static MatmulStatus matmul_impl(MatmulExecution execution, const ggml_gemmini_args_t & args,
                                MatmulOptions options) {
    if (!execution.status()) {
        return execution.status();
    }
    if (options.mode == MatmulInvocationMode::full) {
        return execute_full(execution);
    }
    if (options.mode == MatmulInvocationMode::stripe_pipeline) {
        return make_status(MatmulStatusCode::unsupported_invocation,
                           "pipeline mode requires externally staged stripes");
    }
    if (options.stripe_rows == 0) {
        return make_status(MatmulStatusCode::invalid_argument, "stripe rows must be nonzero");
    }

    const detail::RouteKey route = detail::normalize_route(args);
    if (route.activation == detail::ActivationRoute::fp32 &&
        route.weight == detail::WeightRoute::fp32) {
        MatMul facade(args);
        const MatMulStatus begin_status = facade.begin_stripes();
        if (begin_status != MatMulStatus::success) {
            return to_public_status(begin_status, MatMulCapability::supported, &args);
        }
        for (size_t row_begin = 0; row_begin < args.I;) {
            const size_t row_end = row_begin + std::min(options.stripe_rows, args.I - row_begin);
            const MatMulStatus status = facade.run_stripe({ row_begin, row_end });
            if (status != MatMulStatus::success) {
                return to_public_status(status, MatMulCapability::supported, &args);
            }
            row_begin = row_end;
        }
        return to_public_status(facade.finish_stripes(), MatMulCapability::supported, &args);
    }

    for (size_t row_begin = 0; row_begin < args.I;) {
        const size_t remaining = args.I - row_begin;
        const size_t row_end = row_begin + std::min(options.stripe_rows, remaining);
        MatmulStripeJob job = capture_stripe(execution, MatmulStripeInput(row_begin, row_end));
        MatmulStatus status = job.status();
        if (status) status = execute_dense_stripe(job);
        if (status) status = execute_rmd_stripe(job);
        if (status) status = compose_rmd_stripe(job);
        if (status) status = finalize_stripe(job);
        if (!status) {
            return status;
        }
        row_begin = row_end;
    }
    return finish_execution(execution);
}

MatmulStatus matmul(ggml_gemmini_args_t & args, MatmulOptions options) {
    return matmul_impl(prepare_execution(&args, options), args, options);
}

MatmulStatus matmul(const ggml_gemmini_args_t & args, MatmulOptions options) {
    return matmul_impl(prepare_execution(args, options), args, options);
}

MatmulStatus execute_post_fold_pipeline(
        const ggml_gemmini_args_t & args, MatmulStripeCollector & collector) {
    if (!collector.status_) {
        return collector.status_;
    }
    MatmulOptions options{};
    options.mode = MatmulInvocationMode::stripe_pipeline;
    options.job_capacity = 1;
    MatmulExecution execution = prepare_execution(args, options);
    if (!execution.status()) {
        return execution.status();
    }
    for (auto & captured : collector.stripes_) {
        MatmulStripeJob job = capture_stripe(
            execution,
            MatmulStripeInput(captured.row_begin, captured.row_end, captured.stripe_id),
            std::move(captured.rmd_packet));
        MatmulStatus status = job.status();
        if (status) status = execute_dense_stripe(job);
        if (status) status = execute_rmd_stripe(job);
        if (status) status = compose_rmd_stripe(job);
        if (status) status = finalize_stripe(job);
        if (!status) {
            return status;
        }
    }
    return finish_execution(execution);
}

}

#endif
