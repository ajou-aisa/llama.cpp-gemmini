#pragma once

#include "ggml-gemmini-args.h"
#include "quants/act/exsia/exsia.hpp"

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace ggml::gemmini {

enum class MatMulStatus {
    success,
    empty_stripes,
    malformed_stripe,
    duplicate_stripe,
    overlapping_stripe,
    missing_stripes,
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

class MatmulExecution;
class MatmulStripeJob;
struct MatmulStatus;
MatmulStatus prepare_compensation(MatmulStripeJob &);
MatmulStatus execute_compensation_shard(MatmulStripeJob &);

class MatMul {
public:
    explicit MatMul(ggml_gemmini_args_t args);
    explicit MatMul(ggml_gemmini_args_t * args);

    MatMulResult run_dense();
    MatMulResult run_full();
    MatMulStatus begin_stripes();
    MatMulStatus run_stripe(MatMulStripe stripe);
    MatMulStatus finish_stripes();

    static MatMulCapability stripe_capability(const ggml_gemmini_args_t & args);
    MatMulState state() const;
    ggml_gemmini_args_t & args();
    const ggml_gemmini_args_t & args() const;

private:
    friend MatmulStatus prepare_compensation(MatmulStripeJob &);
    friend MatmulStatus execute_compensation_shard(MatmulStripeJob &);
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);

    MatMulStatus run_stripe(MatMulStripe stripe, size_t stripe_id);

    ggml_gemmini_args_t args_;
    ggml_gemmini_args_t * live_args_ = nullptr;
    size_t first_row_ = 0;
    size_t last_row_begin_ = 0;
    size_t last_row_end_ = 0;
    size_t covered_rows_ = 0;
    bool has_stripes_ = false;
    MatMulState state_ = MatMulState::idle;
};

enum class MatmulStatusCode {
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
    MatmulStatusCode code = MatmulStatusCode::success;
    const char * message = "success";
    MatMulCapability capability = MatMulCapability::supported;

    bool ok() const {
        return code == MatmulStatusCode::success;
    }

    explicit operator bool() const { return ok(); }
};

class MatmulStripeCollector {
public:
    explicit MatmulStripeCollector(size_t capacity);
    ~MatmulStripeCollector();
    bool start(MatmulExecution & execution);
    MatmulStatus finish();
    const quants::act::exsia::StripeReadySink * sink() const;
    const MatmulStatus & status() const;
    const quants::QactOutlier & captured_outlier(size_t stripe, size_t outlier) const;

private:
    struct CapturedStripe {
        size_t stripe_id;
        size_t row_begin;
        size_t row_end;
        std::vector<quants::QactOutlier> outliers;
    };
    static bool on_ready(void *, const quants::act::exsia::StripeReadyEvent &);
    friend MatmulStatus execute_post_fold_pipeline(const ggml_gemmini_args_t &, MatmulStripeCollector &);
    void worker_loop();
    bool worker_started_ = false;
    bool stop_requested_ = false;
    std::thread worker_;
    MatmulExecution * execution_ = nullptr;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<CapturedStripe> pending_;
    size_t capacity_;
    std::vector<CapturedStripe> stripes_;
    MatmulStatus status_;
    quants::act::exsia::StripeReadySink sink_;
};

enum class MatmulInvocationMode {
    full,
    stripe_sequential,
    stripe_pipeline,
};

struct MatmulOptions {
    MatmulInvocationMode mode = MatmulInvocationMode::full;
    size_t stripe_rows = 0;
    size_t dense_threads = 0;
    size_t rc_shards = 0;
    bool validation = false;
    bool profiling = false;
    bool force = false;
    size_t job_capacity = 4;
};

struct MatmulStageMetrics {
    uint64_t nanoseconds = 0;
    size_t count = 0;
};

struct MatmulJobMetrics {
    MatmulStageMetrics la;
    MatmulStageMetrics sf;
    MatmulStageMetrics handoff;
    MatmulStageMetrics ws;
    MatmulStageMetrics rc_prepare;
    MatmulStageMetrics rc_compute;
    MatmulStageMetrics rc_finalize;
};

class MatmulStripeInput {
public:
    MatmulStripeInput(size_t row_begin, size_t row_end, size_t stripe_id = 0,
                      const int32_t * residual = nullptr, size_t residual_count = 0);
    MatmulStripeInput(const MatmulStripeInput &) = delete;
    MatmulStripeInput & operator=(const MatmulStripeInput &) = delete;
    MatmulStripeInput(MatmulStripeInput &&) noexcept = default;
    MatmulStripeInput & operator=(MatmulStripeInput &&) noexcept = default;

    size_t row_begin() const;
    size_t row_end() const;
    size_t stripe_id() const;
    const int32_t * residual() const;
    size_t residual_count() const;

private:
    size_t row_begin_;
    size_t row_end_;
    size_t stripe_id_;
    const int32_t * residual_;
    size_t residual_count_;
};

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
    friend MatmulExecution prepare_execution(ggml_gemmini_args_t &, MatmulOptions);
    friend MatmulStatus execute_full(MatmulExecution &);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput, std::vector<quants::QactOutlier>);
    friend MatmulStatus prepare_compensation(MatmulStripeJob &);
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus execute_compensation_shard(MatmulStripeJob &);
    friend MatmulStatus finalize_stripe(MatmulStripeJob &);
    friend class MatmulStripeJob;
    friend MatmulStatus finish_execution(MatmulExecution &);

    MatmulExecution(ggml_gemmini_args_t args, MatmulOptions options);
    MatmulExecution(ggml_gemmini_args_t * args, MatmulOptions options);

    size_t total_rows_;
    MatMul facade_;
    MatmulOptions options_;
    MatmulStatus status_;
    size_t active_jobs_ = 0;
    size_t captured_rows_ = 0;
    size_t finalized_rows_ = 0;
    size_t first_row_ = 0;
    size_t last_row_begin_ = 0;
    size_t last_row_end_ = 0;
    bool has_captures_ = false;
};

class MatmulStripeJob {
public:
    MatmulStripeJob(const MatmulStripeJob &) = delete;
    MatmulStripeJob & operator=(const MatmulStripeJob &) = delete;
    MatmulStripeJob(MatmulStripeJob && other) noexcept;
    MatmulStripeJob & operator=(MatmulStripeJob && other) noexcept;
    ~MatmulStripeJob();

    const MatmulStatus & status() const;
    const MatmulJobMetrics & metrics() const;

private:
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput, std::vector<quants::QactOutlier>);
    friend MatmulStatus prepare_compensation(MatmulStripeJob &);
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus execute_compensation_shard(MatmulStripeJob &);
    friend MatmulStatus finalize_stripe(MatmulStripeJob &);

    MatmulStripeJob(MatmulExecution * execution, MatmulStripeInput input, MatmulStatus status,
                    std::vector<quants::QactOutlier> outliers = {});
    void release_slot();

    enum class State {
        captured,
        compensation_prepared,
        dense_complete,
        compensation_complete,
        finalized,
    };

    MatmulExecution * execution_;
    MatmulStripeInput input_;
    MatmulStatus status_;
    MatmulJobMetrics metrics_;
    std::vector<quants::QactOutlier> compensation_outliers_;
    bool has_captured_outliers_ = false;
    bool owns_slot_ = false;
    State state_ = State::captured;
};

MatmulExecution prepare_execution(const ggml_gemmini_args_t & args, MatmulOptions options = {});
MatmulExecution prepare_execution(ggml_gemmini_args_t & args, MatmulOptions options = {});
MatmulStatus execute_full(MatmulExecution & execution);
MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input);
MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input,
                               std::vector<quants::QactOutlier> outliers);
MatmulStatus prepare_compensation(MatmulStripeJob & job);
MatmulStatus execute_dense_stripe(MatmulStripeJob & job);
MatmulStatus execute_compensation_shard(MatmulStripeJob & job);
MatmulStatus finalize_stripe(MatmulStripeJob & job);
MatmulStatus finish_execution(MatmulExecution & execution);
MatmulStatus matmul(const ggml_gemmini_args_t & args, MatmulOptions options = {});
MatmulStatus execute_post_fold_pipeline(const ggml_gemmini_args_t & args, MatmulStripeCollector & collector);

}
