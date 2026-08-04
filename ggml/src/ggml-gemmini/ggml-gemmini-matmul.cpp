#include "ggml-gemmini-matmul.hpp"

#include "quants/act/quantize.hpp"
#include "quants/dec/dec.hpp"

#include <gemmini.h>

#include <algorithm>
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

void execute_dense(ggml_gemmini_args_t &args) {
    if (uses_baseline_channel_route(args)) {
        tiled_matmul_auto_baseline(
            &args, baseline_activation_quant_t::TENSOR, baseline_weight_quant_t::CHANNEL);
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
    switch (status) {
        case MatMulStatus::success:            code = MatmulStatusCode::success; break;
        case MatMulStatus::empty_stripes:      code = MatmulStatusCode::empty_stripes; break;
        case MatMulStatus::malformed_stripe:   code = MatmulStatusCode::malformed_stripe; break;
        case MatMulStatus::duplicate_stripe:   code = MatmulStatusCode::duplicate_stripe; break;
        case MatMulStatus::overlapping_stripe: code = MatmulStatusCode::overlapping_stripe; break;
        case MatMulStatus::unsupported:        code = MatmulStatusCode::unsupported; break;
        case MatMulStatus::invalid_state:      code = MatmulStatusCode::invalid_state; break;
        case MatMulStatus::invalid_arguments:  code = MatmulStatusCode::invalid_arguments; break;
    }
    return { code, capability };
}

MatmulStatus invalid_state() {
    return { MatmulStatusCode::invalid_state, MatMulCapability::supported };
}

MatmulStatus unsupported() {
    return { MatmulStatusCode::unsupported, MatMulCapability::unsupported };
}

}

MatMul::MatMul(ggml_gemmini_args_t args) : args_(std::move(args)) {}

MatMulResult MatMul::run_full() {
    if (state_ != MatMulState::idle) {
        return { MatMulStatus::invalid_state, MatMulCapability::supported };
    }
    if (args_.I == 0 || args_.J == 0 || args_.K == 0 || args_.A == nullptr || args_.f_out == nullptr) {
        return { MatMulStatus::invalid_arguments, MatMulCapability::unsupported };
    }

    execute_dense(args_);
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
    stripes_.clear();
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

    for (const MatMulStripe & previous : stripes_) {
        if (previous.row_begin == stripe.row_begin && previous.row_end == stripe.row_end) {
            return MatMulStatus::duplicate_stripe;
        }
        if (stripe.row_begin < previous.row_end && previous.row_begin < stripe.row_end) {
            return MatMulStatus::overlapping_stripe;
        }
    }

    const MatMulStatus status = execute_stripe(args_, stripe);
    if (status == MatMulStatus::success) {
        stripes_.push_back(stripe);
    }
    return status;
}

MatMulStatus MatMul::finish_stripes() {
    if (state_ != MatMulState::accepting_stripes) {
        return MatMulStatus::invalid_state;
    }
    if (stripes_.empty()) {
        state_ = MatMulState::idle;
        return MatMulStatus::empty_stripes;
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

MatmulStripeInput::MatmulStripeInput(size_t row_begin, size_t row_end)
    : row_begin_(row_begin), row_end_(row_end) {}

size_t MatmulStripeInput::row_begin() const {
    return row_begin_;
}

size_t MatmulStripeInput::row_end() const {
    return row_end_;
}

MatmulExecution::MatmulExecution(ggml_gemmini_args_t args, MatmulOptions options)
    : facade_(std::move(args)), options_(options) {
    if (options_.mode == MatmulInvocationMode::stripe_sequential) {
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

const MatmulStatus & MatmulStripeJob::status() const {
    return status_;
}

MatmulExecution prepare_execution(const ggml_gemmini_args_t & args, MatmulOptions options) {
    return MatmulExecution(args, options);
}

MatmulStatus execute_full(MatmulExecution & execution) {
    if (execution.options_.mode != MatmulInvocationMode::full) {
        execution.status_ = invalid_state();
        return execution.status_;
    }
    const MatMulResult result = execution.facade_.run_full();
    execution.status_ = to_public_status(result.status, result.capability);
    return execution.status_;
}

MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input) {
    MatmulStatus status = execution.options_.mode == MatmulInvocationMode::stripe_sequential &&
            execution.facade_.state() == MatMulState::accepting_stripes
        ? MatmulStatus{}
        : invalid_state();
    return MatmulStripeJob(&execution, std::move(input), status);
}

MatmulStatus prepare_compensation(MatmulStripeJob & job) {
    if (job.execution_ == nullptr || job.state_ != MatmulStripeJob::State::captured || !job.status_) {
        job.status_ = invalid_state();
        return job.status_;
    }
    const auto & args = job.execution_->facade_.args_;
    const auto & storage = args.act_quant.storage();
    if (!std::holds_alternative<quants::act::NoneMeta>(storage) &&
        !std::holds_alternative<quants::act::tensor::Meta>(storage)) {
        job.status_ = unsupported();
        job.execution_->status_ = job.status_;
        return job.status_;
    }
    job.compensation_outliers_ = quants::activation_outliers(args);
    job.compensation_prepared_ = true;
    job.status_ = {};
    return job.status_;
}

MatmulStatus execute_dense_stripe(MatmulStripeJob & job) {
    if (job.execution_ == nullptr || job.state_ != MatmulStripeJob::State::captured || !job.status_) {
        job.status_ = invalid_state();
        return job.status_;
    }
    const MatMulStatus status = job.execution_->facade_.run_stripe(
        { job.input_.row_begin(), job.input_.row_end() });
    job.status_ = to_public_status(
        status, status == MatMulStatus::unsupported ? MatMulCapability::unsupported : MatMulCapability::supported);
    job.execution_->status_ = job.status_;
    if (job.status_) {
        job.state_ = MatmulStripeJob::State::dense_complete;
    }
    return job.status_;
}

MatmulStatus execute_compensation_shard(MatmulStripeJob & job) {
    if (job.execution_ == nullptr || job.state_ != MatmulStripeJob::State::dense_complete ||
        !job.compensation_prepared_ || !job.status_) {
        job.status_ = invalid_state();
        return job.status_;
    }
    const auto status = quants::dec::compensate_activation_dec_rows(
        job.compensation_outliers_, job.execution_->facade_.args_,
        job.input_.row_begin(), job.input_.row_end(), "ggml-gemmini-matmul");
    if (status == quants::dec::ActivationDECRowSliceStatus::unsupported) {
        job.status_ = unsupported();
    } else if (status == quants::dec::ActivationDECRowSliceStatus::invalid_arguments) {
        job.status_ = { MatmulStatusCode::invalid_arguments, MatMulCapability::unsupported };
    } else {
        job.status_ = {};
    }
    job.execution_->status_ = job.status_;
    return job.status_;
}

MatmulStatus finalize_stripe(MatmulStripeJob & job) {
    if (job.state_ != MatmulStripeJob::State::dense_complete) {
        job.status_ = invalid_state();
        return job.status_;
    }
    job.state_ = MatmulStripeJob::State::finalized;
    job.status_ = {};
    return job.status_;
}

MatmulStatus finish_execution(MatmulExecution & execution) {
    if (execution.options_.mode == MatmulInvocationMode::full) {
        return execution.facade_.state() == MatMulState::completed ? execution.status_ : invalid_state();
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
    if (options.stripe_rows == 0) {
        return { MatmulStatusCode::invalid_arguments, MatMulCapability::supported };
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
