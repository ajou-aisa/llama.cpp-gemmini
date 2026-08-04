#include "ggml-gemmini-matmul.hpp"

#include "quants/act/quantize.hpp"
#include "quants/dec/dec.hpp"

#include <gemmini.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
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

template <typename Vector>
bool slice_row_vector(Vector & values, size_t row_begin, size_t row_end) {
    if (row_begin > row_end || row_end > values.size()) {
        return false;
    }
    values = Vector(values.begin() + row_begin, values.begin() + row_end);
    return true;
}

template <typename Meta>
void slice_outliers(Meta & meta, size_t row_begin, size_t row_end) {
    std::vector<quants::QactOutlier> local;
    local.reserve(meta.outliers.size());
    for (const auto & outlier : meta.outliers) {
        if (outlier.row < 0 || static_cast<size_t>(outlier.row) < row_begin ||
            static_cast<size_t>(outlier.row) >= row_end) {
            continue;
        }
        local.push_back({ outlier.row - static_cast<int>(row_begin), outlier.col, outlier.residual });
    }
    meta.outliers = std::move(local);
}

bool slice_activation_metadata(ggml_gemmini_args_t & args, size_t row_begin, size_t row_end) {
    if (args.tile_I != 0 && args.tile_I > std::numeric_limits<size_t>::max() / DIM) {
        return false;
    }
    const size_t rows_per_tile = args.tile_I == 0 ? args.I : args.tile_I * DIM;
    if (rows_per_tile == 0) {
        return false;
    }
    if (row_end > std::numeric_limits<size_t>::max() - (rows_per_tile - 1)) {
        return false;
    }

    auto & storage = args.act_quant.storage();
    if (auto * meta = std::get_if<quants::act::exsia::Meta>(&storage)) {
        const size_t first_tile = row_begin / rows_per_tile;
        const size_t last_tile = (row_end + rows_per_tile - 1) / rows_per_tile;
        if (!slice_row_vector(meta->theta, first_tile, last_tile)) {
            return false;
        }
        slice_outliers(*meta, row_begin, row_end);
        args.activation_row_offset = 0;
    } else if (auto * meta = std::get_if<quants::act::token::Meta>(&storage)) {
        if (!slice_row_vector(meta->scales, row_begin, row_end)) {
            return false;
        }
        slice_outliers(*meta, row_begin, row_end);
        args.activation_row_offset = 0;
    } else if (auto * meta = std::get_if<quants::act::block::Meta>(&storage)) {
        if (!slice_row_vector(meta->scales, row_begin, row_end)) {
            return false;
        }
        slice_outliers(*meta, row_begin, row_end);
        args.activation_row_offset = 0;
    } else if (auto * meta = std::get_if<quants::act::stripe::Meta>(&storage)) {
        const size_t first_tile = row_begin / rows_per_tile;
        const size_t last_tile = (row_end + rows_per_tile - 1) / rows_per_tile;
        if (!slice_row_vector(meta->scales, first_tile, last_tile)) {
            return false;
        }
        slice_outliers(*meta, row_begin, row_end);
        args.activation_row_offset = 0;
    }
    return true;
}

bool snapshot_exsia_metadata(const ggml_gemmini_args_t & args, size_t row_begin, size_t row_end,
                             const std::vector<quants::QactOutlier> & outliers,
                             quants::act::Meta & snapshot) {
    const auto * source = std::get_if<quants::act::exsia::Meta>(&args.act_quant.storage());
    if (source == nullptr || row_begin >= row_end) {
        return false;
    }
    const size_t rows_per_tile = args.tile_I == 0 ? args.I : args.tile_I * DIM;
    if (rows_per_tile == 0 || row_end > std::numeric_limits<size_t>::max() - (rows_per_tile - 1)) {
        return false;
    }
    const size_t first_tile = row_begin / rows_per_tile;
    const size_t last_tile = (row_end + rows_per_tile - 1) / rows_per_tile;
    if (last_tile > source->theta.size()) {
        return false;
    }
    quants::act::exsia::Meta local;
    local.e_s = source->e_s;
    local.rho = source->rho;
    local.sigma = source->sigma;
    local.theta.assign(source->theta.begin() + first_tile, source->theta.begin() + last_tile);
    local.outliers.reserve(outliers.size());
    for (const auto & outlier : outliers) {
        if (outlier.row >= 0 && static_cast<size_t>(outlier.row) >= row_begin &&
            static_cast<size_t>(outlier.row) < row_end) {
            local.outliers.push_back({outlier.row - static_cast<int>(row_begin), outlier.col, outlier.residual});
        }
    }
    snapshot.storage_ = std::move(local);
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
    if (!metadata_is_local && !slice_activation_metadata(args, stripe.row_begin, stripe.row_end)) {
        return MatMulStatus::invalid_arguments;
    }

    args.I = stripe.row_end - stripe.row_begin;
    gemmini_set_tile_ws(&args);
    if (args.A != nullptr) {
        args.A += input_offset;
    }
    if (args.A_fp32 != nullptr) {
        args.A_fp32 += input_offset;
    }
    args.f_out += output_offset;

    (void) stripe_id;

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

using Clock = std::chrono::steady_clock;

void record_metric(MatmulStageMetrics & metric, bool enabled, Clock::time_point start) {
    if (!enabled) {
        return;
    }
    metric.nanoseconds += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    ++metric.count;
}

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count());
}

}

namespace detail {

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
    RouteCapabilities caps{};
    caps.deprecated = key.weight == WeightRoute::q8_h2 || key.weight == WeightRoute::q8_hp2;
    caps.full = key.weight != WeightRoute::unknown && key.weight != WeightRoute::q8_h0;
    caps.sliced_dense = caps.full && !caps.deprecated;
    caps.sliced_compensation = caps.sliced_dense;
    caps.live_stripe_producer = key.activation == ActivationRoute::exsia && caps.sliced_compensation;
    caps.external_rc_shards = caps.sliced_compensation;
    caps.internal_parallel_dense = caps.full;
    if (key.activation == ActivationRoute::exsia &&
        (key.weight == WeightRoute::q8_channel_direct ||
         key.weight == WeightRoute::q8_channel_sidecar)) {
        caps = {};
    }
    if ((key.weight == WeightRoute::q8_channel_direct ||
         key.weight == WeightRoute::q8_channel_sidecar) &&
        key.activation == ActivationRoute::fp32) {
        caps = {};
    }
    if (key.backend == BackendRoute::gemmini_os || key.backend == BackendRoute::ws_sim) {
        caps = {};
        caps.deprecated = key.weight == WeightRoute::q8_h2 || key.weight == WeightRoute::q8_hp2;
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

MatmulStripeCollector::MatmulStripeCollector(size_t capacity)
    : capacity_(capacity), sink_{this, &MatmulStripeCollector::on_ready} {
    if (capacity == 0) {
        status_ = make_status(MatmulStatusCode::invalid_argument, "collector capacity must be nonzero");
    }
}

MatmulStripeCollector::~MatmulStripeCollector() {
    finish();
}

bool MatmulStripeCollector::start(MatmulExecution & execution) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!status_ || worker_started_ || execution.mode() != MatmulInvocationMode::stripe_pipeline) {
        return false;
    }
    execution_ = &execution;
    worker_started_ = true;
    dense_done_ = false;
    worker_ = std::thread(&MatmulStripeCollector::worker_loop, this);
    compensation_worker_ = std::thread(&MatmulStripeCollector::compensation_loop, this);
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
    }
    condition_.notify_all();
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
    if (!status_ && execution_ != nullptr) {
        std::lock_guard<std::mutex> execution_lock(*execution_->state_mutex_);
        execution_->status_ = status_;
        execution_->state_ = MatmulExecutionState::failed;
    }
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
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push_back(job);
        }
        MatmulStatus status = prepare_compensation(*job);
        if (status) {
            std::lock_guard<std::mutex> lock(mutex_);
            compensation_pending_.push_back(job);
            condition_.notify_all();
        }
        if (status) {
            std::unique_lock<std::mutex> job_lock(*job->shard_mutex_);
            job->lifecycle_condition_.wait(job_lock, [&job] {
                return job->metrics_.rc_start_ns != 0 || !job->status_;
            });
            status = job->status_;
        }
        {
            std::lock_guard<std::mutex> job_lock(*job->shard_mutex_);
            job->metrics_.ws_start_ns = now_ns();
        }
        if (status) status = execute_dense_stripe(*job);
        if (!status) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --in_flight_;
            }
            fail(status);
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
        }

        MatmulStatus status = job->status();
        {
            std::lock_guard<std::mutex> job_lock(*job->shard_mutex_);
            job->metrics_.rc_start_ns = now_ns();
        }
        job->lifecycle_condition_.notify_all();
        const size_t shard_count = std::max<size_t>(1, std::min(
            execution_->options_.rc_shards == 0 ? size_t {1} : execution_->options_.rc_shards,
            execution_->facade_.args().J));
        const auto rc_start = Clock::now();
        if (status && shard_count == 1) {
            status = execute_compensation_shard(*job, 0, 1);
        } else if (status) {
            {
                std::lock_guard<std::mutex> lock(*job->shard_mutex_);
                job->parallel_shards_ = true;
                job->expected_shards_ = shard_count;
                job->completed_shards_ = 0;
                job->rc_state_ = MatmulRcState::running;
            }
            std::vector<MatmulStatus> shard_status(shard_count);
            std::vector<std::thread> shard_workers;
            shard_workers.reserve(shard_count);
            for (size_t shard_id = 0; shard_id < shard_count; ++shard_id) {
                shard_workers.emplace_back([&job, &shard_status, shard_id, shard_count] {
                    shard_status[shard_id] = execute_compensation_shard(*job, shard_id, shard_count);
                });
            }
            for (auto & shard_worker : shard_workers) {
                shard_worker.join();
            }
            {
                std::lock_guard<std::mutex> lock(*job->shard_mutex_);
                job->metrics_.rc_compute = {};
                record_metric(job->metrics_.rc_compute, execution_->options_.profiling, rc_start);
            }
            for (const auto & shard_result : shard_status) {
                if (!shard_result.ok()) {
                    status = shard_result;
                    break;
                }
            }
        }
        if (status) {
            std::unique_lock<std::mutex> lock(*job->shard_mutex_);
            job->lifecycle_condition_.wait(lock, [&job] {
                return job->dense_state_ == MatmulDenseState::complete || !job->status_;
            });
            status = job->status_;
        }
        if (status) status = finalize_stripe(*job);
        {
            std::lock_guard<std::mutex> job_lock(*job->shard_mutex_);
            job->metrics_.rc_end_ns = now_ns();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (status) {
                profiles_.push_back(job->metrics());
                --in_flight_;
            } else {
                --in_flight_;
            }
        }
        condition_.notify_all();
        if (!status) {
            fail(status);
            break;
        }
    }
}

const quants::act::exsia::StripeReadySink * MatmulStripeCollector::sink() const {
    return &sink_;
}

MatmulStatus MatmulStripeCollector::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

std::vector<MatmulJobMetrics> MatmulStripeCollector::profiles() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return profiles_;
}

quants::QactOutlier MatmulStripeCollector::captured_outlier(size_t stripe, size_t outlier) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stripes_.at(stripe).outliers.at(outlier);
}

bool MatmulStripeCollector::on_ready(
        void * user_data, const quants::act::exsia::StripeReadyEvent & event) {
    auto & collector = *static_cast<MatmulStripeCollector *>(user_data);
    if (event.row_begin >= event.row_end ||
        ((event.outliers == nullptr) != (event.outlier_count == 0))) {
        std::lock_guard<std::mutex> lock(collector.mutex_);
        collector.status_ = make_status(MatmulStatusCode::invalid_argument, "invalid stripe event");
        collector.stop_requested_ = true;
        return false;
    }
    try {
        std::vector<quants::QactOutlier> outliers;
        if (event.outlier_count != 0) {
            outliers.assign(event.outliers, event.outliers + event.outlier_count);
        }
        std::unique_lock<std::mutex> lock(collector.mutex_);
        if (!collector.status_ || collector.stop_requested_) {
            return false;
        }
        if (collector.worker_started_) {
            collector.condition_.wait(lock, [&collector] {
                return collector.stop_requested_ || !collector.status_ ||
                    collector.pending_.size() + collector.in_flight_ < collector.capacity_;
            });
            if (collector.stop_requested_ || !collector.status_) {
                return false;
            }
            collector.pending_.push_back(
                {event.stripe_id, event.row_begin, event.row_end, std::move(outliers),
                 event.local_end_cycle >= event.local_start_cycle ?
                     event.local_end_cycle - event.local_start_cycle : 0,
                 event.local_group3_end_cycle >= event.local_group3_start_cycle ?
                     event.local_group3_end_cycle - event.local_group3_start_cycle : 0,
                 event.folding_end_cycle >= event.folding_start_cycle ?
                     event.folding_end_cycle - event.folding_start_cycle : 0,
                 event.local_end_ns >= event.local_start_ns ?
                     event.local_end_ns - event.local_start_ns : 0,
                 event.folding_end_ns >= event.folding_start_ns ?
                     event.folding_end_ns - event.folding_start_ns : 0});
            lock.unlock();
            collector.condition_.notify_all();
            return true;
        }
        if (collector.stripes_.size() >= collector.capacity_) {
            collector.status_ = make_status(MatmulStatusCode::out_of_memory, "collector capacity exhausted");
            collector.stop_requested_ = true;
            return false;
        }
        collector.stripes_.push_back(
            {event.stripe_id, event.row_begin, event.row_end, std::move(outliers),
             event.local_end_cycle >= event.local_start_cycle ?
                 event.local_end_cycle - event.local_start_cycle : 0,
             event.local_group3_end_cycle >= event.local_group3_start_cycle ?
                 event.local_group3_end_cycle - event.local_group3_start_cycle : 0,
             event.folding_end_cycle >= event.folding_start_cycle ?
                 event.folding_end_cycle - event.folding_start_cycle : 0,
             event.local_end_ns >= event.local_start_ns ?
                 event.local_end_ns - event.local_start_ns : 0,
             event.folding_end_ns >= event.folding_start_ns ?
                 event.folding_end_ns - event.folding_start_ns : 0});
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(collector.mutex_);
        collector.status_ = make_status(MatmulStatusCode::out_of_memory, "stripe capture allocation failed");
        collector.stop_requested_ = true;
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
      metrics_(other.metrics_), staged_residual_(std::move(other.staged_residual_)),
      compensation_outliers_(std::move(other.compensation_outliers_)),
      compensation_ycom_(std::move(other.compensation_ycom_)),
      staged_activation_meta_(std::move(other.staged_activation_meta_)),
      has_captured_outliers_(other.has_captured_outliers_), owns_slot_(other.owns_slot_),
      released_(other.released_),
      expected_shards_(other.expected_shards_), completed_shards_(other.completed_shards_),
      parallel_shards_(other.parallel_shards_), shard_mutex_(std::move(other.shard_mutex_)),
      dense_state_(other.dense_state_), rc_state_(other.rc_state_), captured_(other.captured_),
      finalized_(other.finalized_) {
    other.execution_ = nullptr;
    other.owns_slot_ = false;
    other.released_ = true;
}

MatmulStripeJob & MatmulStripeJob::operator=(MatmulStripeJob && other) noexcept {
    if (this != &other) {
        release_slot();
        execution_ = other.execution_;
        input_ = std::move(other.input_);
        status_ = other.status_;
        metrics_ = other.metrics_;
        staged_residual_ = std::move(other.staged_residual_);
        compensation_outliers_ = std::move(other.compensation_outliers_);
        compensation_ycom_ = std::move(other.compensation_ycom_);
        staged_activation_meta_ = std::move(other.staged_activation_meta_);
        has_captured_outliers_ = other.has_captured_outliers_;
        owns_slot_ = other.owns_slot_;
        released_ = other.released_;
        expected_shards_ = other.expected_shards_;
        completed_shards_ = other.completed_shards_;
        parallel_shards_ = other.parallel_shards_;
        dense_state_ = other.dense_state_;
        rc_state_ = other.rc_state_;
        captured_ = other.captured_;
        finalized_ = other.finalized_;
        shard_mutex_ = std::move(other.shard_mutex_);
        other.execution_ = nullptr;
        other.owns_slot_ = false;
        other.released_ = true;
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
                if (rc_state_ != MatmulRcState::complete) {
                    rc_state_ = MatmulRcState::cancelled;
                }
            } else {
                rc_state_ = MatmulRcState::failed;
                if (dense_state_ != MatmulDenseState::complete) {
                    dense_state_ = MatmulDenseState::cancelled;
                }
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
    const bool has_outliers = input.outliers() != nullptr || input.outlier_count() != 0;
    std::vector<int32_t> staged_residual;
    if (input.residual_count() != 0) {
        staged_residual.assign(input.residual(), input.residual() + input.residual_count());
    }
    std::vector<quants::QactOutlier> outliers;
    if (input.outlier_count() != 0) {
        outliers.assign(input.outliers(), input.outliers() + input.outlier_count());
    }
    MatmulStripeJob job = capture_stripe(execution, std::move(input), std::move(outliers));
    job.has_captured_outliers_ = has_outliers;
    job.staged_residual_ = std::move(staged_residual);
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
    std::unique_ptr<quants::act::Meta> staged_activation_meta;
    std::vector<float> compensation_ycom;
    try {
        if (!has_captured_outliers) {
            const auto & all_outliers = quants::activation_outliers_view(args);
            outliers.reserve(all_outliers.size());
            for (const auto & outlier : all_outliers) {
                if (outlier.row >= 0 &&
                    static_cast<size_t>(outlier.row) >= job.input_.row_begin() &&
                    static_cast<size_t>(outlier.row) < job.input_.row_end()) {
                    outliers.push_back(outlier);
                }
            }
        }
        if (job.execution_->options_.mode == MatmulInvocationMode::stripe_pipeline &&
            std::holds_alternative<quants::act::exsia::Meta>(args.act_quant.storage())) {
            staged_activation_meta = std::make_unique<quants::act::Meta>();
            if (!snapshot_exsia_metadata(
                    args, job.input_.row_begin(), job.input_.row_end(),
                    outliers, *staged_activation_meta)) {
                const MatmulStatus failure = make_status(
                    MatmulStatusCode::invalid_contract, "pipeline stripe metadata is not ready");
                job.record_failure(failure, false);
                return failure;
            }
        }
        if (!outliers.empty()) {
            const size_t rows = job.input_.row_end() - job.input_.row_begin();
            const size_t elements = rows * args.J;
            if (rows != 0 && elements / rows != args.J) {
                const MatmulStatus failure = make_status(
                    MatmulStatusCode::out_of_memory, "compensation scratch size overflow");
                job.record_failure(failure, false);
                return failure;
            }
            compensation_ycom.assign(elements, 0.0f);
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
        job.staged_activation_meta_ = std::move(staged_activation_meta);
        job.compensation_ycom_ = std::move(compensation_ycom);
        job.rc_state_ = MatmulRcState::prepared;
        record_metric(job.metrics_.rc_prepare, job.execution_->options_.profiling, start);
    }
    if (job.staged_activation_meta_ != nullptr) {
        std::lock_guard<std::mutex> execution_lock(*job.execution_->state_mutex_);
        job.execution_->staged_metadata_active_ = true;
    }
    job.lifecycle_condition_.notify_all();
    return {};
}

MatmulStatus execute_dense_stripe(MatmulStripeJob & job) {
    quants::act::Meta * staged_activation_meta = nullptr;
    {
        std::lock_guard<std::mutex> lock(*job.shard_mutex_);
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ ||
            job.dense_state_ != MatmulDenseState::idle || !job.status_) {
            return invalid_state("dense execution requires captured Dense idle state");
        }
        job.dense_state_ = MatmulDenseState::running;
        staged_activation_meta = job.staged_activation_meta_.get();
    }
    const auto start = Clock::now();
    MatMul * dense_facade = &job.execution_->facade_;
    if (staged_activation_meta != nullptr && job.execution_->staged_facade_ != nullptr) {
        dense_facade = job.execution_->staged_facade_.get();
        if (dense_facade->state() == MatMulState::idle) {
            const MatMulStatus begin_status = dense_facade->begin_stripes();
            if (begin_status != MatMulStatus::success) {
                const MatmulStatus failure = to_public_status(
                    begin_status,
                    begin_status == MatMulStatus::unsupported ? MatMulCapability::unsupported :
                        MatMulCapability::supported);
                job.record_failure(failure, true);
                return failure;
            }
        }
        dense_facade->args().act_quant = *staged_activation_meta;
    }
    const MatMulStatus status = staged_activation_meta != nullptr ?
        dense_facade->run_staged_stripe(
            { job.input_.row_begin(), job.input_.row_end() }, job.input_.stripe_id()) :
        dense_facade->run_stripe(
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
        if (job.status_) {
            job.dense_state_ = MatmulDenseState::complete;
            record_metric(job.metrics_.ws, job.execution_->options_.profiling, start);
        }
        result = job.status_;
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
        if (job.execution_ == nullptr || !job.captured_ || job.finalized_ ||
            (job.rc_state_ != MatmulRcState::prepared && job.rc_state_ != MatmulRcState::running) ||
            !job.status_) {
            return invalid_state("compensation execution requires RC prepared state");
        }
        if (shard_count == 0 || shard_id >= shard_count) {
            const MatmulStatus failure = make_status(
                MatmulStatusCode::invalid_argument, "invalid compensation shard");
            job.status_ = failure;
            job.rc_state_ = MatmulRcState::failed;
            if (job.dense_state_ != MatmulDenseState::complete) {
                job.dense_state_ = MatmulDenseState::cancelled;
            }
            job.lifecycle_condition_.notify_all();
            return failure;
        }
        if (job.parallel_shards_) {
            if (job.expected_shards_ != shard_count) {
                return invalid_state("parallel compensation shard count changed");
            }
            job.rc_state_ = MatmulRcState::running;
        } else if (job.completed_shards_ == 0) {
            job.expected_shards_ = shard_count;
            job.rc_state_ = MatmulRcState::running;
        } else if (job.expected_shards_ != shard_count || shard_id != job.completed_shards_) {
            return invalid_state("compensation shards must complete in order");
        }
    }
    const auto start = Clock::now();
    auto dec_status = quants::dec::ActivationDECRowSliceStatus::success;
    MatmulStatus result{};
    const size_t col_begin = job.execution_->facade_.args().J * shard_id / shard_count;
    const size_t col_end = job.execution_->facade_.args().J * (shard_id + 1) / shard_count;
    char dec_layer[64];
    std::snprintf(dec_layer, sizeof(dec_layer), "ggml-gemmini-matmul.stripe-%zu",
                  job.input_.stripe_id());
#if ERROR_COMPENSATION
    if (!job.compensation_outliers_.empty()) {
        dec_status = quants::dec::compensate_activation_dec_rows_columns(
            job.compensation_outliers_, job.execution_->facade_.args(),
            job.input_.row_begin(), job.input_.row_end(), col_begin, col_end,
            dec_layer, job.execution_->dispatch_override_,
            job.compensation_ycom_.data() + col_begin, job.execution_->facade_.args().J);
    }
#else
    (void) col_begin;
    (void) col_end;
    (void) dec_layer;
#endif
    if (result.ok()) {
        if (dec_status == quants::dec::ActivationDECRowSliceStatus::unsupported) {
            result = unsupported_backend("compensation execution is unsupported by backend");
        } else if (dec_status == quants::dec::ActivationDECRowSliceStatus::invalid_arguments) {
            result = make_status(
            MatmulStatusCode::invalid_argument, "invalid compensation shard", MatMulCapability::unsupported);
        }
    }
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

MatmulStatus matmul_impl(MatmulExecution execution, const ggml_gemmini_args_t & args,
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
        const size_t shard_count = std::max<size_t>(1, std::min(
            options.rc_shards == 0 ? size_t {1} : options.rc_shards, args.J));
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
        const size_t shard_count = std::max<size_t>(1, std::min(
            options.rc_shards == 0 ? size_t {1} : options.rc_shards, args.J));
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
