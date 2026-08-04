#pragma once

#include "ggml-gemmini-args.h"

#include <cstddef>
#include <vector>

namespace ggml::gemmini {

enum class MatMulStatus {
    success,
    empty_stripes,
    malformed_stripe,
    duplicate_stripe,
    overlapping_stripe,
    unsupported,
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
    MatMulStatus status;
    MatMulCapability capability;
};

class MatMul {
public:
    explicit MatMul(ggml_gemmini_args_t args);

    MatMulResult run_full();
    MatMulStatus begin_stripes();
    MatMulStatus run_stripe(MatMulStripe stripe);
    MatMulStatus finish_stripes();

    static MatMulCapability stripe_capability(const ggml_gemmini_args_t & args);
    MatMulState state() const;

private:
    ggml_gemmini_args_t args_;
    std::vector<MatMulStripe> stripes_;
    MatMulState state_ = MatMulState::idle;
};

enum class MatmulStatusCode {
    success,
    empty_stripes,
    malformed_stripe,
    duplicate_stripe,
    overlapping_stripe,
    unsupported,
    invalid_state,
    invalid_arguments,
};

struct MatmulStatus {
    MatmulStatusCode code = MatmulStatusCode::success;
    MatMulCapability capability = MatMulCapability::supported;

    explicit operator bool() const {
        return code == MatmulStatusCode::success;
    }
};

enum class MatmulInvocationMode {
    full,
    stripe_sequential,
};

struct MatmulOptions {
    MatmulInvocationMode mode = MatmulInvocationMode::full;
    size_t stripe_rows = 0;
};

class MatmulStripeInput {
public:
    MatmulStripeInput(size_t row_begin, size_t row_end);
    MatmulStripeInput(const MatmulStripeInput &) = delete;
    MatmulStripeInput & operator=(const MatmulStripeInput &) = delete;
    MatmulStripeInput(MatmulStripeInput &&) noexcept = default;
    MatmulStripeInput & operator=(MatmulStripeInput &&) noexcept = default;

    size_t row_begin() const;
    size_t row_end() const;

private:
    size_t row_begin_;
    size_t row_end_;
};

class MatmulStripeJob;

class MatmulExecution {
public:
    MatmulExecution(const MatmulExecution &) = delete;
    MatmulExecution & operator=(const MatmulExecution &) = delete;
    MatmulExecution(MatmulExecution &&) noexcept = default;
    MatmulExecution & operator=(MatmulExecution &&) noexcept = default;

    MatmulInvocationMode mode() const;
    const MatmulStatus & status() const;

private:
    friend MatmulExecution prepare_execution(const ggml_gemmini_args_t &, MatmulOptions);
    friend MatmulStatus execute_full(MatmulExecution &);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
    friend class MatmulStripeJob;
    friend MatmulStatus finish_execution(MatmulExecution &);

    MatmulExecution(ggml_gemmini_args_t args, MatmulOptions options);

    MatMul facade_;
    MatmulOptions options_;
    MatmulStatus status_;
};

class MatmulStripeJob {
public:
    MatmulStripeJob(const MatmulStripeJob &) = delete;
    MatmulStripeJob & operator=(const MatmulStripeJob &) = delete;
    MatmulStripeJob(MatmulStripeJob &&) noexcept = default;
    MatmulStripeJob & operator=(MatmulStripeJob &&) noexcept = default;

    const MatmulStatus & status() const;

private:
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStatus prepare_compensation(MatmulStripeJob &);
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus execute_compensation_shard(MatmulStripeJob &);
    friend MatmulStatus finalize_stripe(MatmulStripeJob &);

    MatmulStripeJob(MatmulExecution * execution, MatmulStripeInput input, MatmulStatus status);

    enum class State {
        captured,
        dense_complete,
        finalized,
    };

    MatmulExecution * execution_;
    MatmulStripeInput input_;
    MatmulStatus status_;
    State state_ = State::captured;
};

MatmulExecution prepare_execution(const ggml_gemmini_args_t & args, MatmulOptions options = {});
MatmulStatus execute_full(MatmulExecution & execution);
MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input);
MatmulStatus prepare_compensation(MatmulStripeJob & job);
MatmulStatus execute_dense_stripe(MatmulStripeJob & job);
MatmulStatus execute_compensation_shard(MatmulStripeJob & job);
MatmulStatus finalize_stripe(MatmulStripeJob & job);
MatmulStatus finish_execution(MatmulExecution & execution);
MatmulStatus matmul(const ggml_gemmini_args_t & args, MatmulOptions options = {});

}
