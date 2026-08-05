#define GGML_GEMMINI_MATMUL_IMPLEMENTATION 1
#include "ggml-gemmini-matmul.hpp"

#include "quants/act/quantize.hpp"
#include "quants/dec/dec.hpp"

#include <gemmini.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
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

MatmulStatus from_dec_status(quants::dec::DecStatus status, const char * message) {
    switch (status) {
        case quants::dec::DecStatus::success:
            return {};
        case quants::dec::DecStatus::unsupported:
            return unsupported_backend(message);
        case quants::dec::DecStatus::invalid_arguments:
        case quants::dec::DecStatus::invalid_shard:
            return make_status(MatmulStatusCode::invalid_argument, message,
                               MatMulCapability::unsupported);
        case quants::dec::DecStatus::allocation_failure:
            return make_status(MatmulStatusCode::out_of_memory, message);
        case quants::dec::DecStatus::execution_failed:
            return make_status(MatmulStatusCode::execution_failure, message);
    }
    return make_status(MatmulStatusCode::execution_failure, message);
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

size_t persistent_rc_worker_budget() {
    const size_t hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        return 1;
    }
    return hw > 2 ? hw - 2 : size_t {1};
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
    caps.external_rc_shards = caps.sliced_compensation;
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
    return run_full(quants::dec::DispatchOverride::automatic);
}

MatMulResult MatMul::run_full(quants::dec::DispatchOverride dispatch_override) {
    const MatMulResult dense = run_dense();
    if (dense.status != MatMulStatus::success) {
        return dense;
    }
#if ERROR_COMPENSATION
    quants::dec::compensate_activation_dec(
        quants::activation_outliers(args()), args(), "ggml-gemmini-matmul", dispatch_override);
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
    if (!valid_activation_metadata(args())) {
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

MatmulStripeInput::MatmulStripeInput(size_t row_begin, size_t row_end, size_t stripe_id,
                                     const quants::QactOutlier * outliers, size_t outlier_count)
    : row_begin_(row_begin), row_end_(row_end), stripe_id_(stripe_id),
      residual_(nullptr), residual_count_(0), outliers_(outliers), outlier_count_(outlier_count) {}

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

const quants::QactOutlier * MatmulStripeInput::outliers() const {
    return outliers_;
}

size_t MatmulStripeInput::outlier_count() const {
    return outlier_count_;
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
    if (options_.force_row_direct && options_.force_group_k_csc) {
        status_ = make_status(MatmulStatusCode::invalid_argument,
                              "conflicting DEC dispatch overrides");
        state_ = MatmulExecutionState::failed;
        return;
    }
    dispatch_override_ = options_.force_row_direct ?
        quants::dec::DispatchOverride::row_direct :
        options_.force_group_k_csc ? quants::dec::DispatchOverride::group_k_csc :
        quants::dec::DispatchOverride::automatic;
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
    if (options_.force_row_direct && options_.force_group_k_csc) {
        status_ = make_status(MatmulStatusCode::invalid_argument,
                              "conflicting DEC dispatch overrides");
        state_ = MatmulExecutionState::failed;
        return;
    }
    dispatch_override_ = options_.force_row_direct ?
        quants::dec::DispatchOverride::row_direct :
        options_.force_group_k_csc ? quants::dec::DispatchOverride::group_k_csc :
        quants::dec::DispatchOverride::automatic;
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
    dispatch_override_ = other.dispatch_override_;
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
MatmulStripeCollector * test_rc_failure_collector = nullptr;
MatmulStatus test_rc_failure;
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
    if (test_rc_failure_collector == this) {
        test_rc_failure_collector = nullptr;
        test_rc_failure = {};
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
        if (!status_ || worker_started_ || execution.mode() != MatmulInvocationMode::stripe_pipeline) {
            return false;
        }
    }
    {
        std::lock_guard<std::mutex> execution_lock(*execution.state_mutex_);
        if (execution.pipeline_attached_) {
            std::lock_guard<std::mutex> lock(mutex_);
            status_ = invalid_state("execution already has a live stripe collector");
            return false;
        }
        execution.pipeline_attached_ = true;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        execution_ = &execution;
        worker_started_ = true;
        dense_done_ = false;
        stop_requested_ = false;
        rc_stop_requested_ = false;
        rc_worker_capacity_ = std::min({
            std::max(size_t {1}, execution.options_.rc_shards),
            std::max(size_t {1}, execution.facade_.args().J),
            persistent_rc_worker_budget(),
        });
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        test_thread_start_attempts_ = 0;
#endif
    }
    const auto fail_start = [&](MatmulStatus failure) {
        std::vector<std::shared_ptr<MatmulStripeJob>> jobs;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            status_ = failure;
            stop_requested_ = true;
            dense_done_ = true;
            rc_stop_requested_ = true;
            pending_.clear();
            compensation_pending_.clear();
            rc_pending_.clear();
            rc_tasks_remaining_ = 0;
            rc_batch_status_ = failure;
            for (const auto & weak_job : jobs_) {
                if (auto job = weak_job.lock()) {
                    jobs.push_back(std::move(job));
                }
            }
        }
        for (const auto & job : jobs) {
            job->cancel(failure);
            release_in_flight_once(job);
        }
        condition_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        if (compensation_worker_.joinable()) {
            compensation_worker_.join();
        }
        for (auto & worker : rc_workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        rc_workers_.clear();
        {
            std::lock_guard<std::mutex> execution_lock(*execution.state_mutex_);
            execution.status_ = status_;
            execution.state_ = MatmulExecutionState::failed;
            execution.pipeline_attached_ = false;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            execution_ = nullptr;
            worker_started_ = false;
            rc_worker_capacity_ = 0;
        }
        return false;
    };
    try {
        rc_workers_.reserve(rc_worker_capacity_);
        const auto maybe_fail_thread_start = [&] {
#if defined(GGML_GEMMINI_TEST_OBSERVER)
            std::lock_guard<std::mutex> lock(mutex_);
            if (test_fail_thread_start_attempt_ != 0 &&
                ++test_thread_start_attempts_ == test_fail_thread_start_attempt_) {
                throw std::system_error(
                    std::make_error_code(std::errc::resource_unavailable_try_again),
                    "injected thread start failure");
            }
#endif
        };
        for (size_t worker = 0; worker < rc_worker_capacity_; ++worker) {
            maybe_fail_thread_start();
            rc_workers_.emplace_back(&MatmulStripeCollector::rc_worker_loop, this);
        }
        maybe_fail_thread_start();
        worker_ = std::thread(&MatmulStripeCollector::worker_loop, this);
        maybe_fail_thread_start();
        compensation_worker_ = std::thread(&MatmulStripeCollector::compensation_loop, this);
    } catch (const std::bad_alloc &) {
        return fail_start(make_status(
            MatmulStatusCode::out_of_memory, "collector startup allocation failed"));
    } catch (const std::system_error &) {
        return fail_start(make_status(
            MatmulStatusCode::execution_failure, "collector worker thread creation failed"));
    }
    return true;
}

MatmulStatus MatmulStripeCollector::cancel() {
    std::vector<std::shared_ptr<MatmulStripeJob>> jobs;
    MatmulStatus cancelled = make_status(MatmulStatusCode::cancelled, "stripe pipeline cancelled");
    {
        std::lock_guard<std::mutex> lock(mutex_);
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
        compensation_pending_.clear();
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
        compensation_pending_.clear();
    }
    for (const auto & job : jobs) {
        job->cancel(failure);
        release_in_flight_once(job);
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
    if (!worker_started_) {
        return status_;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
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
    if (compensation_worker_.joinable()) {
        compensation_worker_.join();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        rc_stop_requested_ = true;
    }
    condition_.notify_all();
    for (auto & worker : rc_workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    rc_workers_.clear();
    if (!status_ && execution_ != nullptr) {
        std::lock_guard<std::mutex> execution_lock(*execution_->state_mutex_);
        execution_->status_ = status_;
        execution_->state_ = MatmulExecutionState::failed;
    }
    if (execution_ != nullptr) {
        std::lock_guard<std::mutex> execution_lock(*execution_->state_mutex_);
        execution_->pipeline_attached_ = false;
    }
    execution_ = nullptr;
    worker_started_ = false;
    return status_;
}

void MatmulStripeCollector::worker_loop() {
    for (;;) {
        CapturedStripe captured{};
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] {
                return stop_requested_ || (!pending_.empty() && in_flight_ < capacity_);
            });
            if (pending_.empty()) {
                break;
            }
            captured = std::move(pending_.front());
            pending_.pop_front();
            ++in_flight_;
            condition_.notify_all();
        }

        auto job = std::make_shared<MatmulStripeJob>(capture_stripe(
            *execution_,
            MatmulStripeInput(captured.row_begin, captured.row_end, captured.stripe_id),
            std::move(captured.outliers)));
        {
            std::lock_guard<std::mutex> job_lock(*job->shard_mutex_);
            job->metrics_.la_cycles = captured.la_cycles;
            job->metrics_.la3_cycles = captured.la3_cycles;
            job->metrics_.sf_cycles = captured.sf_cycles;
            job->metrics_.la3_ns = captured.la3_ns;
            job->metrics_.sf1_ns = captured.sf1_ns;
            job->metrics_.la.nanoseconds = captured.la3_ns;
            job->metrics_.la.count = captured.la3_ns != 0 ? 1 : 0;
            job->metrics_.sf.nanoseconds = captured.sf1_ns;
            job->metrics_.sf.count = captured.sf1_ns != 0 ? 1 : 0;
            job->metrics_.capture_copy = captured.capture_copy;
            job->metrics_.producer_wait = captured.producer_wait;
            job->metrics_.queue_insert = captured.queue_insert;
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
        const uint64_t rc_queued_ns = now_ns();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            compensation_pending_.push_back(job);
            condition_.notify_all();
        }
        std::unique_lock<std::mutex> job_lock(*job->shard_mutex_);
        job->lifecycle_condition_.wait(job_lock, [&job] {
            return job->metrics_.rc_start_ns != 0 || !job->status_;
        });
        status = job->status_;
        job_lock.unlock();
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return !test_pause_dense_ || stop_requested_; });
        }
#endif
        {
            std::lock_guard<std::mutex> job_lock(*job->shard_mutex_);
            job->metrics_.ws_start_ns = now_ns();
            if (job->execution_->options_.profiling) {
                job->metrics_.ws_queue.nanoseconds = job->metrics_.ws_start_ns - captured.queued_ns;
                job->metrics_.ws_queue.count = 1;
                job->metrics_.rc_queue.nanoseconds = job->metrics_.rc_start_ns - rc_queued_ns;
                job->metrics_.rc_queue.count = 1;
            }
        }
        const MatmulStatus dense_status = execute_dense_stripe(*job);
        if (!dense_status) {
            release_in_flight_once(job);
            fail(dense_status);
            break;
        }
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dense_done_ = true;
    }
    condition_.notify_all();
}

void MatmulStripeCollector::compensation_loop() {
    for (;;) {
        std::shared_ptr<MatmulStripeJob> job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] {
                return dense_done_ || !compensation_pending_.empty();
            });
            if (compensation_pending_.empty()) {
                if (dense_done_) {
                    break;
                }
                continue;
            }
            job = std::move(compensation_pending_.front());
            compensation_pending_.pop_front();
            ++active_rc_stripes_;
            max_active_rc_stripes_ = std::max(max_active_rc_stripes_, active_rc_stripes_);
        }

        MatmulStatus status = job->status();
#if defined(GGML_GEMMINI_TEST_OBSERVER)
        if (status && test_rc_failure_collector == this && !test_rc_failure) {
            status = test_rc_failure;
            job->record_failure(status, false);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                test_rc_failure_observed_ = true;
            }
            condition_.notify_all();
        }
#endif
        {
            std::lock_guard<std::mutex> job_lock(*job->shard_mutex_);
            job->metrics_.rc_start_ns = now_ns();
        }
        job->lifecycle_condition_.notify_all();
        if (status) {
            status = prepare_compensation(*job);
        }
        const size_t shard_count = job->prepared_dec_ != nullptr ?
            job->prepared_dec_->shard_count() : 0;
        if (status && shard_count != 0) {
            {
                std::lock_guard<std::mutex> lock(*job->shard_mutex_);
                job->expected_shards_ = shard_count;
                job->completed_shards_ = 0;
                job->parallel_shards_ = true;
                job->rc_state_ = MatmulRcState::running;
            }
            {
                std::unique_lock<std::mutex> lock(mutex_);
                rc_tasks_remaining_ = shard_count;
                rc_batch_status_ = {};
                for (size_t shard_id = 0; shard_id < shard_count; ++shard_id) {
                    rc_pending_.push_back({job, shard_id, shard_count});
                }
                max_rc_queue_depth_ = std::max(max_rc_queue_depth_, rc_pending_.size());
                condition_.notify_all();
                condition_.wait(lock, [this] { return rc_tasks_remaining_ == 0; });
                status = rc_batch_status_ ? job->status() : rc_batch_status_;
            }
            {
                std::lock_guard<std::mutex> lock(*job->shard_mutex_);
                job->parallel_shards_ = false;
            }
        }
        if (status) {
            const auto wait_start = Clock::now();
            std::unique_lock<std::mutex> lock(*job->shard_mutex_);
            job->lifecycle_condition_.wait(lock, [&job] {
                return dense_state_is_terminal(job->dense_state_);
            });
            status = job->status_;
            record_metric(job->metrics_.rc_wait, job->execution_->options_.profiling, wait_start);
        } else {
            std::unique_lock<std::mutex> lock(*job->shard_mutex_);
            job->lifecycle_condition_.wait(lock, [&job] {
                return dense_state_is_terminal(job->dense_state_);
            });
        }
        if (status) status = finalize_stripe(*job);
        {
            std::lock_guard<std::mutex> job_lock(*job->shard_mutex_);
            job->metrics_.rc_end_ns = now_ns();
        }
        if (status) {
            const MatmulJobMetrics profile = job->metrics();
            std::lock_guard<std::mutex> lock(mutex_);
            profiles_.push_back(profile);
        }
        release_in_flight_once(job);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --active_rc_stripes_;
        }
        condition_.notify_all();
        if (!status) {
            fail(status);
            break;
        }
    }
}

void MatmulStripeCollector::rc_worker_loop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++rc_worker_starts_;
    }
    for (;;) {
        RcTask task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return rc_stop_requested_ || !rc_pending_.empty(); });
            if (rc_pending_.empty()) {
                if (rc_stop_requested_) {
                    break;
                }
                continue;
            }
            task = std::move(rc_pending_.front());
            rc_pending_.pop_front();
        }
        MatmulStatus status = task.job->status();
        if (status) {
            status = execute_compensation_shard(*task.job, task.shard_id, task.shard_count);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!status && rc_batch_status_) {
                rc_batch_status_ = status;
            }
            ++rc_tasks_executed_;
            --rc_tasks_remaining_;
        }
        condition_.notify_all();
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
    return {status_, capacity_, pending_.size(), in_flight_, rc_pending_.size(),
            rc_worker_capacity_, rc_worker_starts_, rc_tasks_executed_,
            max_active_rc_stripes_, max_rc_queue_depth_, worker_started_};
}

std::vector<MatmulJobMetrics> MatmulStripeCollector::profiles() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return profiles_;
}

quants::QactOutlier MatmulStripeCollector::captured_outlier(size_t stripe, size_t outlier) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stripes_.at(stripe).outliers.at(outlier);
}

#if defined(GGML_GEMMINI_TEST_OBSERVER)
void MatmulStripeCollector::test_inject_rc_failure(MatmulStatus failure) {
    std::lock_guard<std::mutex> lock(mutex_);
    test_rc_failure_collector = this;
    test_rc_failure = failure;
    test_dense_observer_collector = this;
    observed_dense_state_at_release = MatmulDenseState::idle;
}

void MatmulStripeCollector::test_inject_thread_start_failure(size_t attempt) {
    std::lock_guard<std::mutex> lock(mutex_);
    test_fail_thread_start_attempt_ = attempt;
}

void MatmulStripeCollector::test_pause_dense_before_execute() {
    std::lock_guard<std::mutex> lock(mutex_);
    test_pause_dense_ = true;
}

void MatmulStripeCollector::test_wait_for_rc_failure() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return test_rc_failure_observed_; });
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
    if (event.row_begin >= event.row_end ||
        ((event.outliers == nullptr) != (event.outlier_count == 0))) {
        {
            std::lock_guard<std::mutex> lock(collector.mutex_);
            collector.status_ = make_status(MatmulStatusCode::invalid_argument, "invalid stripe event");
            collector.stop_requested_ = true;
        }
        collector.condition_.notify_all();
        return false;
    }
    try {
        const auto copy_start = Clock::now();
        std::vector<quants::QactOutlier> outliers;
        if (event.outlier_count != 0) {
            outliers.assign(event.outliers, event.outliers + event.outlier_count);
        }
        MatmulStageMetrics capture_copy;
        record_metric(capture_copy, true, copy_start);
        std::unique_lock<std::mutex> lock(collector.mutex_);
        if (!collector.status_ || collector.stop_requested_) {
            return false;
        }
        if (collector.worker_started_) {
            const auto wait_start = Clock::now();
            collector.condition_.wait(lock, [&collector] {
                return collector.stop_requested_ || !collector.status_ ||
                    collector.pending_.size() + collector.in_flight_ < collector.capacity_;
            });
            if (collector.stop_requested_ || !collector.status_) {
                return false;
            }
            MatmulStageMetrics producer_wait;
            record_metric(producer_wait, true, wait_start);
            CapturedStripe captured{
                 event.stripe_id, event.row_begin, event.row_end, std::move(outliers),
                 event.local_end_cycle >= event.local_start_cycle ?
                     event.local_end_cycle - event.local_start_cycle : 0,
                 event.local_group3_end_cycle >= event.local_group3_start_cycle ?
                     event.local_group3_end_cycle - event.local_group3_start_cycle : 0,
                 event.folding_end_cycle >= event.folding_start_cycle ?
                     event.folding_end_cycle - event.folding_start_cycle : 0,
                 event.local_end_ns >= event.local_start_ns ?
                     event.local_end_ns - event.local_start_ns : 0,
                 event.folding_end_ns >= event.folding_start_ns ?
                     event.folding_end_ns - event.folding_start_ns : 0, {}, {}, {}, 0};
            captured.capture_copy = capture_copy;
            captured.producer_wait = producer_wait;
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
        CapturedStripe captured{
             event.stripe_id, event.row_begin, event.row_end, std::move(outliers),
             event.local_end_cycle >= event.local_start_cycle ?
                 event.local_end_cycle - event.local_start_cycle : 0,
             event.local_group3_end_cycle >= event.local_group3_start_cycle ?
                 event.local_group3_end_cycle - event.local_group3_start_cycle : 0,
             event.folding_end_cycle >= event.folding_start_cycle ?
                 event.folding_end_cycle - event.folding_start_cycle : 0,
             event.local_end_ns >= event.local_start_ns ?
                 event.local_end_ns - event.local_start_ns : 0,
             event.folding_end_ns >= event.folding_start_ns ?
                 event.folding_end_ns - event.folding_start_ns : 0, {}, {}, {}, 0};
        captured.capture_copy = capture_copy;
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
        std::vector<quants::QactOutlier> outliers)
    : execution_(execution), input_(std::move(input)), status_(status),
      compensation_outliers_(std::move(outliers)), has_captured_outliers_(true),
      captured_(status.ok()) {}

MatmulStripeJob::MatmulStripeJob()
    : execution_(nullptr), input_(0, 0), status_(invalid_state("job is not captured")), captured_(false) {}

MatmulStripeJob::MatmulStripeJob(MatmulStripeJob && other) noexcept
    : execution_(other.execution_), input_(std::move(other.input_)), status_(other.status_),
      metrics_(other.metrics_), compensation_outliers_(std::move(other.compensation_outliers_)),
      staged_activation_meta_(std::move(other.staged_activation_meta_)),
      has_captured_outliers_(other.has_captured_outliers_), owns_slot_(other.owns_slot_),
      released_(other.released_), collector_slot_released_(other.collector_slot_released_),
      expected_shards_(other.expected_shards_), completed_shards_(other.completed_shards_),
      parallel_shards_(other.parallel_shards_), shard_mutex_(std::move(other.shard_mutex_)),
      prepared_dec_(std::move(other.prepared_dec_)),
      compensation_ycom_(std::move(other.compensation_ycom_)),
      dense_state_(other.dense_state_), rc_state_(other.rc_state_), captured_(other.captured_),
      finalized_(other.finalized_) {
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
        compensation_outliers_ = std::move(other.compensation_outliers_);
        staged_activation_meta_ = std::move(other.staged_activation_meta_);
        compensation_ycom_ = std::move(other.compensation_ycom_);
        has_captured_outliers_ = other.has_captured_outliers_;
        owns_slot_ = other.owns_slot_;
        released_ = other.released_;
        collector_slot_released_ = other.collector_slot_released_;
        expected_shards_ = other.expected_shards_;
        completed_shards_ = other.completed_shards_;
        parallel_shards_ = other.parallel_shards_;
        prepared_dec_ = std::move(other.prepared_dec_);
        dense_state_ = other.dense_state_;
        rc_state_ = other.rc_state_;
        captured_ = other.captured_;
        finalized_ = other.finalized_;
        shard_mutex_ = std::move(other.shard_mutex_);
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
    if (!owns_slot_ || released_ || execution_ == nullptr || shard_mutex_ == nullptr) {
        return;
    }
    MatmulExecution * execution = nullptr;
    {
        std::lock_guard<std::mutex> lock(*shard_mutex_);
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
        std::lock_guard<std::mutex> lock(*shard_mutex_);
        if (finalized_) {
            return;
        }
        if (status_.ok()) {
            status_ = status;
        }
        if (dense_state_ != MatmulDenseState::complete && dense_state_ != MatmulDenseState::failed) {
            dense_state_ = MatmulDenseState::cancelled;
        }
        if (rc_state_ != MatmulRcState::complete && rc_state_ != MatmulRcState::failed) {
            rc_state_ = MatmulRcState::cancelled;
        }
    }
    lifecycle_condition_.notify_all();
}

void MatmulStripeJob::record_failure(MatmulStatus status, bool dense_branch) {
    {
        std::lock_guard<std::mutex> lock(*shard_mutex_);
        if (status_.ok()) {
            status_ = status;
            if (dense_branch) {
                dense_state_ = MatmulDenseState::failed;
            } else {
                rc_state_ = MatmulRcState::failed;
            }
        }
    }
    lifecycle_condition_.notify_all();
}

MatmulStatus MatmulStripeJob::status() const {
    std::lock_guard<std::mutex> lock(*shard_mutex_);
    return status_;
}

MatmulJobMetrics MatmulStripeJob::metrics() const {
    std::lock_guard<std::mutex> lock(*shard_mutex_);
    return metrics_;
}

MatmulStripeJobSnapshot MatmulStripeJob::snapshot() const {
    std::lock_guard<std::mutex> lock(*shard_mutex_);
    return {status_, metrics_, dense_state_, rc_state_, expected_shards_, completed_shards_,
            captured_, finalized_, released_};
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
    const MatMulResult result = execution.facade_.run_full(execution.dispatch_override_);
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
    if ((input.residual_count() != 0 && input.residual() == nullptr) ||
        (input.outlier_count() != 0 && input.outliers() == nullptr)) {
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
    const bool has_outliers = input.residual_count() != 0 ||
        input.outliers() != nullptr || input.outlier_count() != 0;
    std::vector<quants::QactOutlier> outliers;
    try {
        if (input.outlier_count() != 0) {
            outliers.assign(input.outliers(), input.outliers() + input.outlier_count());
        }
    } catch (const std::bad_alloc &) {
        return MatmulStripeJob(
            &execution, std::move(input),
            make_status(MatmulStatusCode::out_of_memory, "stripe capture allocation failed"));
    }
    MatmulStripeJob job = capture_stripe(execution, std::move(input), std::move(outliers));
    job.has_captured_outliers_ = has_outliers;
    return job;
}

MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input,
                               std::vector<quants::QactOutlier> outliers) {
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
               ((input.residual() == nullptr) != (input.residual_count() == 0)) ||
               ((input.outliers() == nullptr) != (input.outlier_count() == 0)) ||
               (input.residual() != nullptr && input.outliers() != nullptr)) {
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
    }

    MatmulStripeJob job(&execution, std::move(input), status, std::move(outliers));
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
    MatmulStripeJob captured = input.outliers() != nullptr || input.outlier_count() != 0
        ? capture_stripe(execution, MatmulStripeInput(
            input.row_begin(), input.row_end(), input.stripe_id(), input.outliers(), input.outlier_count()))
        : capture_stripe(execution, MatmulStripeInput(
            input.row_begin(), input.row_end(), input.stripe_id(), input.residual(), input.residual_count()));
    job = std::move(captured);
    return job.status();
}

MatmulStatus prepare_compensation(MatmulStripeJob & job) {
    if (job.shard_mutex_ == nullptr) {
        return invalid_state("compensation state unavailable");
    }
    std::vector<quants::QactOutlier> outliers;
    bool has_captured_outliers = false;
    {
        std::lock_guard<std::mutex> lock(*job.shard_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ ||
            job.rc_state_ != MatmulRcState::idle || !job.status_) {
            return invalid_state("compensation preparation requires captured RC idle state");
        }
        job.rc_state_ = MatmulRcState::preparing;
        outliers = job.compensation_outliers_;
        has_captured_outliers = job.has_captured_outliers_;
    }
    const auto start = Clock::now();
    const auto & args = job.execution_->facade_.args();
    std::vector<float> compensation_ycom;
    std::shared_ptr<const quants::dec::PreparedDecSlice> prepared_dec;
    try {
        const quants::act::ActivationMetadataView metadata(
            args, job.input_.row_begin(), job.input_.row_end());
        if (!metadata.valid()) {
            const MatmulStatus failure = invalid_contract(
                "stripe activation metadata does not cover global rows");
            job.record_failure(failure, false);
            return failure;
        }
        if (!has_captured_outliers) {
            const auto & all_outliers = quants::activation_outliers_view(args);
            outliers.reserve(all_outliers.size());
            for (const auto & outlier : all_outliers) {
                if (metadata.contains(outlier)) {
                    outliers.push_back(outlier);
                }
            }
        }
        if (outliers.empty()) {
            std::lock_guard<std::mutex> lock(*job.shard_mutex_);
            if (!job.status_) {
                return job.status_;
            }
            job.compensation_outliers_.clear();
            job.has_captured_outliers_ = true;
            job.compensation_ycom_.clear();
            job.prepared_dec_.reset();
            job.expected_shards_ = 1;
            job.completed_shards_ = 1;
            job.rc_state_ = MatmulRcState::complete;
            record_metric(job.metrics_.rc_prepare, job.execution_->options_.profiling, start);
            job.lifecycle_condition_.notify_all();
            return {};
        }
        const size_t rows = job.input_.row_end() - job.input_.row_begin();
        const size_t elements = rows * args.J;
        if (rows != 0 && elements / rows != args.J) {
            const MatmulStatus failure = make_status(
                MatmulStatusCode::out_of_memory, "compensation scratch size overflow");
            job.record_failure(failure, false);
            return failure;
        }
        compensation_ycom.assign(elements, 0.0f);
        const size_t requested_shards = job.execution_->options_.rc_shards == 0 ?
            size_t {1} : job.execution_->options_.rc_shards;
        const MatmulStatus plan_status = from_dec_status(
            quants::dec::prepare_activation_dec_slice(
                outliers, args, job.input_.row_begin(), job.input_.row_end(),
                requested_shards, job.execution_->dispatch_override_, prepared_dec),
            "compensation preparation failed");
        if (!plan_status) {
            job.record_failure(plan_status, false);
            return plan_status;
        }
    } catch (const std::bad_alloc &) {
        const MatmulStatus failure = make_status(
            MatmulStatusCode::out_of_memory, "compensation preparation allocation failed");
        job.record_failure(failure, false);
        return failure;
    }
    {
        std::lock_guard<std::mutex> lock(*job.shard_mutex_);
        if (!job.status_) {
            return job.status_;
        }
        job.compensation_outliers_ = std::move(outliers);
        job.has_captured_outliers_ = true;
        job.compensation_ycom_ = std::move(compensation_ycom);
        job.prepared_dec_ = std::move(prepared_dec);
        job.expected_shards_ = job.prepared_dec_->shard_count();
        job.rc_state_ = MatmulRcState::prepared;
        record_metric(job.metrics_.rc_prepare, job.execution_->options_.profiling, start);
    }
    job.lifecycle_condition_.notify_all();
    return {};
}

MatmulStatus execute_dense_stripe(MatmulStripeJob & job) {
    {
        std::lock_guard<std::mutex> lock(*job.shard_mutex_);
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
        std::lock_guard<std::mutex> lock(*job.shard_mutex_);
        job.metrics_.ws_end_ns = now_ns();
    }
    if (!dense_status) {
        job.record_failure(dense_status, true);
        return dense_status;
    }
    MatmulStatus result;
    {
        std::lock_guard<std::mutex> lock(*job.shard_mutex_);
        job.dense_state_ = MatmulDenseState::complete;
        record_metric(job.metrics_.ws, job.execution_->options_.profiling, start);
        job.metrics_.ws_service = job.metrics_.ws;
        result = dense_status;
    }
    job.lifecycle_condition_.notify_all();
    return result;
}

MatmulStatus execute_compensation_shard(MatmulStripeJob & job) {
    return execute_compensation_shard(job, 0, 1);
}

MatmulStatus execute_compensation_shard(MatmulStripeJob & job, size_t shard_id, size_t shard_count) {
    if (job.shard_mutex_ == nullptr) {
        return invalid_state("compensation shard state unavailable");
    }
    {
        std::lock_guard<std::mutex> lock(*job.shard_mutex_);
        if (job.execution_ != nullptr && job.captured_ && !job.finalized_ && job.status_ &&
            job.rc_state_ == MatmulRcState::complete && job.prepared_dec_ == nullptr &&
            job.expected_shards_ == 1 && job.completed_shards_ == 1 &&
            shard_id == 0 && shard_count == 1) {
            return {};
        }
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ ||
            (job.rc_state_ != MatmulRcState::prepared && job.rc_state_ != MatmulRcState::running) ||
            !job.status_ || job.prepared_dec_ == nullptr) {
            return invalid_state("compensation execution requires RC prepared state");
        }
        const size_t prepared_shard_count = job.prepared_dec_->shard_count();
        if (shard_count == 0 || shard_count != prepared_shard_count || shard_id >= shard_count) {
            return invalid_state("compensation shards must complete in order");
        }
        if (job.parallel_shards_) {
            if (job.expected_shards_ != prepared_shard_count) {
                return invalid_state("parallel compensation shard count changed");
            }
            job.rc_state_ = MatmulRcState::running;
        } else if (job.completed_shards_ == 0) {
            job.expected_shards_ = prepared_shard_count;
            job.rc_state_ = MatmulRcState::running;
        } else if (job.expected_shards_ != prepared_shard_count || shard_id != job.completed_shards_) {
            return invalid_state("compensation shards must complete in order");
        }
    }
    const auto start = Clock::now();
    MatmulStatus result{};
    static thread_local quants::dec::DecShardScratch scratch;
    result = from_dec_status(
        quants::dec::execute_prepared_dec_shard(
            *job.prepared_dec_, shard_id, scratch,
            job.compensation_ycom_.empty() ? nullptr : job.compensation_ycom_.data(),
            job.compensation_ycom_.empty() ? 0 : job.execution_->facade_.args().J),
        "compensation execution failed");
    if (!result.ok()) {
        job.record_failure(result, false);
        return result;
    }
    {
        std::lock_guard<std::mutex> lock(*job.shard_mutex_);
        if (job.status_) {
            ++job.completed_shards_;
            if (job.completed_shards_ == job.expected_shards_) {
                job.rc_state_ = MatmulRcState::complete;
            }
            record_metric(job.metrics_.rc_compute, job.execution_->options_.profiling, start);
        } else {
            result = job.status_;
        }
    }
    job.lifecycle_condition_.notify_all();
    return result;
}

MatmulStatus finalize_stripe(MatmulStripeJob & job) {
    const auto start = Clock::now();
    {
        std::lock_guard<std::mutex> lock(*job.shard_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_) {
            return invalid_state("stripe is not finalizable");
        }
        if (!job.status_) {
            return job.status_;
        }
        if (job.dense_state_ != MatmulDenseState::complete || job.rc_state_ != MatmulRcState::complete) {
            return invalid_state("finalize requires Dense and RC completion");
        }
        job.finalized_ = true;
        job.metrics_.stripe_id = job.input_.stripe_id();
        job.metrics_.row_begin = job.input_.row_begin();
        job.metrics_.row_end = job.input_.row_end();
        job.metrics_.rc_shards = job.expected_shards_;
        if (!job.compensation_ycom_.empty()) {
            const auto & args = job.execution_->facade_.args();
            const size_t row_stride = args.stride_f_out ? args.stride_f_out : args.J;
            const size_t col_stride = args.col_stride_f_out ? args.col_stride_f_out : 1;
            for (size_t row = 0; row < job.input_.row_end() - job.input_.row_begin(); ++row) {
                float * dst = args.f_out + (job.input_.row_begin() + row) * row_stride;
                const float * src = job.compensation_ycom_.data() + row * args.J;
                for (size_t col = 0; col < args.J; ++col) {
                    dst[col * col_stride] += src[col];
                }
            }
        }
        record_metric(job.metrics_.rc_finalize, job.execution_->options_.profiling, start);
        if (job.execution_->options_.profiling) {
            job.metrics_.t_RC4.nanoseconds = job.metrics_.rc_prepare.nanoseconds +
                job.metrics_.rc_compute.nanoseconds + job.metrics_.rc_finalize.nanoseconds;
            job.metrics_.t_RC4.count = 1;
        }
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
        MatmulStatus status = prepare_compensation(job);
        if (!status) {
            return status;
        }
        status = execute_dense_stripe(job);
        if (!status) {
            return status;
        }
        const size_t shard_count = job.snapshot().expected_shards;
        for (size_t shard_id = 0; status && shard_id < shard_count; ++shard_id)
            status = execute_compensation_shard(job, shard_id, shard_count);
        if (!status) {
            return status;
        }
        status = finalize_stripe(job);
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
            std::move(captured.outliers));
        MatmulStatus status = prepare_compensation(job);
        if (status) status = execute_dense_stripe(job);
        const size_t shard_count = job.snapshot().expected_shards;
        for (size_t shard_id = 0; status && shard_id < shard_count; ++shard_id)
            status = execute_compensation_shard(job, shard_id, shard_count);
        if (status) status = finalize_stripe(job);
        if (!status) {
            return status;
        }
    }
    return finish_execution(execution);
}

}
