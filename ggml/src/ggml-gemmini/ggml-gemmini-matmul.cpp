#include "ggml-gemmini-matmul.hpp"

#include "quants/act/quantize.hpp"
#include "quants/dec/dec.hpp"

#include <gemmini.h>

#include <algorithm>
#include <chrono>
#include <limits>
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

bool uses_baseline_channel_route(const ggml_gemmini_args_t & args) {
    return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel ||
        args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
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
    if (uses_baseline_channel_route(args)) {
        tiled_matmul_auto_baseline(&args, baseline_activation_for(args), baseline_weight_quant_t::CHANNEL);
    } else if (args.weight_i8_scale_active) {
        tiled_matmul_auto_baseline(&args, baseline_activation_for(args), baseline_weight_quant_t::TENSOR);
    } else {
        tiled_matmul_auto_im2p(&args);
    }
}

MatMulStatus execute_stripe(ggml_gemmini_args_t args, MatMulStripe stripe) {
    const size_t input_stride = args.sA ? args.sA : args.K;
    const size_t output_stride = args.stride_f_out ? args.stride_f_out : args.J;
    size_t input_offset = 0;
    size_t output_offset = 0;
    if (!checked_offset(stripe.row_begin, input_stride, input_offset) ||
        !checked_offset(stripe.row_begin, output_stride, output_offset)) {
        return MatMulStatus::invalid_arguments;
    }

    args.I = stripe.row_end - stripe.row_begin;
    args.A += input_offset;
    args.f_out += output_offset;

    execute_dense(args);
    return MatMulStatus::success;
}

MatmulStatus to_public_status(MatMulStatus status, MatMulCapability capability) {
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
            code = MatmulStatusCode::unsupported_route;
            message = "unsupported route";
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

}

MatMul::MatMul(ggml_gemmini_args_t args) : args_(std::move(args)) {}

MatMulResult MatMul::run_dense() {
    if (state_ != MatMulState::idle) {
        return { MatMulStatus::invalid_state, MatMulCapability::supported };
    }
    if (args_.I == 0 || args_.J == 0 || args_.K == 0 || args_.A == nullptr || args_.f_out == nullptr) {
        return { MatMulStatus::invalid_arguments, MatMulCapability::unsupported };
    }

    execute_dense(args_);
    return { MatMulStatus::success, MatMulCapability::supported };
}

MatMulResult MatMul::run_full() {
    const MatMulResult dense = run_dense();
    if (dense.status != MatMulStatus::success) {
        return dense;
    }
    quants::dec::compensate_activation_dec(
        quants::activation_outliers(args_), args_, "ggml-gemmini-matmul");
    state_ = MatMulState::completed;
    return { MatMulStatus::success, MatMulCapability::supported };
}

MatMulStatus MatMul::begin_stripes() {
    if (state_ != MatMulState::idle) {
        return MatMulStatus::invalid_state;
    }
    if (stripe_capability(args_) == MatMulCapability::unsupported) {
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
    if (state_ != MatMulState::accepting_stripes) {
        return MatMulStatus::invalid_state;
    }
    if (stripe.row_begin >= stripe.row_end || stripe.row_end > args_.I) {
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

    const MatMulStatus status = execute_stripe(args_, stripe);
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
    if (first_row_ != 0 || last_row_end_ != args_.I || covered_rows_ != args_.I) {
        state_ = MatMulState::idle;
        return MatMulStatus::missing_stripes;
    }
    state_ = MatMulState::completed;
    return MatMulStatus::success;
}

MatMulCapability MatMul::stripe_capability(const ggml_gemmini_args_t & args) {
    const auto format = args.weight_format;
    if (format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2 ||
        args.transpose_A || (args.D != nullptr && !args.repeating_bias) ||
        !row_invariant_activation(args)) {
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
    if (options_.mode != MatmulInvocationMode::full && options_.job_capacity == 0) {
        status_ = make_status(MatmulStatusCode::invalid_argument, "job capacity must be nonzero");
        return;
    }
    if (options_.mode == MatmulInvocationMode::stripe_sequential ||
        options_.mode == MatmulInvocationMode::stripe_pipeline) {
        const MatMulStatus status = facade_.begin_stripes();
        status_ = to_public_status(
            status, status == MatMulStatus::unsupported ? MatMulCapability::unsupported : MatMulCapability::supported);
    }
}

MatmulInvocationMode MatmulExecution::mode() const {
    return options_.mode;
}

const MatmulStatus & MatmulExecution::status() const {
    return status_;
}

MatmulStripeJob::MatmulStripeJob(
        MatmulExecution * execution, MatmulStripeInput input, MatmulStatus status)
    : execution_(execution), input_(std::move(input)), status_(status) {}

MatmulStripeJob::MatmulStripeJob(MatmulStripeJob && other) noexcept
    : execution_(other.execution_), input_(std::move(other.input_)), status_(other.status_),
      metrics_(other.metrics_), compensation_outliers_(std::move(other.compensation_outliers_)),
      owns_slot_(other.owns_slot_), state_(other.state_) {
    other.execution_ = nullptr;
    other.owns_slot_ = false;
}

MatmulStripeJob & MatmulStripeJob::operator=(MatmulStripeJob && other) noexcept {
    if (this != &other) {
        release_slot();
        execution_ = other.execution_;
        input_ = std::move(other.input_);
        status_ = other.status_;
        metrics_ = other.metrics_;
        compensation_outliers_ = std::move(other.compensation_outliers_);
        owns_slot_ = other.owns_slot_;
        state_ = other.state_;
        other.execution_ = nullptr;
        other.owns_slot_ = false;
    }
    return *this;
}

MatmulStripeJob::~MatmulStripeJob() {
    release_slot();
}

void MatmulStripeJob::release_slot() {
    if (owns_slot_ && execution_ != nullptr) {
        --execution_->active_jobs_;
        owns_slot_ = false;
    }
}

const MatmulStatus & MatmulStripeJob::status() const {
    return status_;
}

const MatmulJobMetrics & MatmulStripeJob::metrics() const {
    return metrics_;
}

MatmulExecution prepare_execution(const ggml_gemmini_args_t & args, MatmulOptions options) {
    return MatmulExecution(args, options);
}

MatmulStatus execute_full(MatmulExecution & execution) {
    if (execution.options_.mode != MatmulInvocationMode::full) {
        execution.status_ = make_status(
            MatmulStatusCode::unsupported_invocation, "full execution requires full mode");
        return execution.status_;
    }
    const MatMulResult result = execution.facade_.run_full();
    execution.status_ = to_public_status(result.status, result.capability);
    return execution.status_;
}

MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input) {
    const auto start = Clock::now();
    MatmulStatus status{};
    if (!execution.status_.ok()) {
        status = execution.status_;
    } else if (execution.options_.mode == MatmulInvocationMode::full) {
        status = make_status(MatmulStatusCode::unsupported_invocation, "stripe capture requires stripe mode");
    } else if (execution.facade_.state() != MatMulState::accepting_stripes) {
        status = invalid_state("execution is not accepting stripes");
    } else if (input.row_begin() >= input.row_end() || input.row_end() > execution.total_rows_ ||
               ((input.residual() == nullptr) != (input.residual_count() == 0))) {
        status = make_status(MatmulStatusCode::invalid_argument, "invalid stripe input");
    } else if (execution.active_jobs_ >= execution.options_.job_capacity) {
        status = make_status(MatmulStatusCode::out_of_memory, "job capacity exhausted");
    } else if (execution.has_captures_ && input.row_begin() == execution.last_row_begin_ &&
               input.row_end() == execution.last_row_end_) {
        status = invalid_contract("duplicate stripe");
    } else if (execution.has_captures_ && input.row_begin() < execution.last_row_end_) {
        status = invalid_contract("overlapping stripe");
    }

    MatmulStripeJob job(&execution, std::move(input), status);
    if (status.ok()) {
        if (!execution.has_captures_) {
            execution.first_row_ = job.input_.row_begin();
        }
        execution.last_row_begin_ = job.input_.row_begin();
        execution.last_row_end_ = job.input_.row_end();
        execution.captured_rows_ += job.input_.row_end() - job.input_.row_begin();
        execution.has_captures_ = true;
        ++execution.active_jobs_;
        job.owns_slot_ = true;
        record_metric(job.metrics_.handoff, execution.options_.profiling, start);
    }
    return job;
}

MatmulStatus prepare_compensation(MatmulStripeJob & job) {
    if (job.execution_ == nullptr || job.state_ != MatmulStripeJob::State::captured || !job.status_) {
        job.status_ = invalid_state("compensation preparation requires captured state");
        return job.status_;
    }
    const auto start = Clock::now();
    const auto & args = job.execution_->facade_.args_;
    const auto & storage = args.act_quant.storage();
    if (!std::holds_alternative<quants::act::NoneMeta>(storage) &&
        !std::holds_alternative<quants::act::tensor::Meta>(storage)) {
        job.status_ = unsupported_backend("compensation preparation is unsupported by backend");
        job.execution_->status_ = job.status_;
        return job.status_;
    }
    job.compensation_outliers_ = quants::activation_outliers(args);
    job.state_ = MatmulStripeJob::State::compensation_prepared;
    job.status_ = {};
    record_metric(job.metrics_.rc_prepare, job.execution_->options_.profiling, start);
    return job.status_;
}

MatmulStatus execute_dense_stripe(MatmulStripeJob & job) {
    if (job.execution_ == nullptr || job.state_ != MatmulStripeJob::State::compensation_prepared || !job.status_) {
        job.status_ = invalid_state("dense execution requires compensation preparation");
        return job.status_;
    }
    const auto start = Clock::now();
    const MatMulStatus status = job.execution_->facade_.run_stripe(
        { job.input_.row_begin(), job.input_.row_end() });
    job.status_ = to_public_status(
        status, status == MatMulStatus::unsupported ? MatMulCapability::unsupported : MatMulCapability::supported);
    job.execution_->status_ = job.status_;
    if (job.status_) {
        job.state_ = MatmulStripeJob::State::dense_complete;
        record_metric(job.metrics_.ws, job.execution_->options_.profiling, start);
    }
    return job.status_;
}

MatmulStatus execute_compensation_shard(MatmulStripeJob & job) {
    if (job.execution_ == nullptr || job.state_ != MatmulStripeJob::State::dense_complete || !job.status_) {
        job.status_ = invalid_state("compensation execution requires dense completion");
        return job.status_;
    }
    const auto start = Clock::now();
    const auto status = quants::dec::compensate_activation_dec_rows(
        job.compensation_outliers_, job.execution_->facade_.args_,
        job.input_.row_begin(), job.input_.row_end(), "ggml-gemmini-matmul");
    if (status == quants::dec::ActivationDECRowSliceStatus::unsupported) {
        job.status_ = unsupported_backend("compensation execution is unsupported by backend");
    } else if (status == quants::dec::ActivationDECRowSliceStatus::invalid_arguments) {
        job.status_ = make_status(
            MatmulStatusCode::invalid_argument, "invalid compensation shard", MatMulCapability::unsupported);
    } else {
        job.status_ = {};
        job.state_ = MatmulStripeJob::State::compensation_complete;
        record_metric(job.metrics_.rc_compute, job.execution_->options_.profiling, start);
    }
    job.execution_->status_ = job.status_;
    return job.status_;
}

MatmulStatus finalize_stripe(MatmulStripeJob & job) {
    if (job.execution_ == nullptr || job.state_ != MatmulStripeJob::State::compensation_complete || !job.status_) {
        job.status_ = invalid_state("finalize requires compensation completion");
        return job.status_;
    }
    const auto start = Clock::now();
    job.state_ = MatmulStripeJob::State::finalized;
    job.execution_->finalized_rows_ += job.input_.row_end() - job.input_.row_begin();
    record_metric(job.metrics_.rc_finalize, job.execution_->options_.profiling, start);
    job.release_slot();
    job.status_ = {};
    return job.status_;
}

MatmulStatus finish_execution(MatmulExecution & execution) {
    if (execution.options_.mode == MatmulInvocationMode::full) {
        return execution.facade_.state() == MatMulState::completed ? execution.status_ : invalid_state();
    }
    if (execution.active_jobs_ != 0) {
        return invalid_state("cannot finish with live jobs");
    }
    if (!execution.has_captures_ || execution.first_row_ != 0 ||
        execution.last_row_end_ != execution.total_rows_ ||
        execution.captured_rows_ != execution.total_rows_ ||
        execution.finalized_rows_ != execution.total_rows_) {
        return invalid_contract("missing stripes");
    }
    const MatMulStatus status = execution.facade_.finish_stripes();
    execution.status_ = to_public_status(status, MatMulCapability::supported);
    return execution.status_;
}

MatmulStatus matmul(const ggml_gemmini_args_t & args, MatmulOptions options) {
    MatmulExecution execution = prepare_execution(args, options);
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
        status = execute_compensation_shard(job);
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

}
