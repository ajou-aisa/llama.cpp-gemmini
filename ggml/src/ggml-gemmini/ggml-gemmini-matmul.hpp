#pragma once

#include "ggml-gemmini-args.h"
#include "quants/act/exsia/exsia.hpp"
#include "quants/dec/dec.hpp"

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

namespace ggml::gemmini {

namespace detail {

enum class ActivationRoute : uint8_t { unknown, fp32, exsia, tensor, token, block, stripe };
enum class WeightRoute : uint8_t {
    unknown, fp32, tensor_i8, channel_i8, block_i8, q8_h1, q8_hp1, q8_h2, q8_hp2,
    q8_channel_direct, q8_channel_sidecar, q8_h0
};
enum class BackendRoute : uint8_t { cpu, gemmini_ws, gemmini_os, ws_sim };

struct RouteKey {
    ActivationRoute activation = ActivationRoute::unknown;
    WeightRoute weight = WeightRoute::unknown;
    BackendRoute backend = BackendRoute::cpu;
};

struct RouteCapabilities {
    bool full = false;
    bool sliced_dense = false;
    bool sliced_compensation = false;
    bool live_stripe_producer = false;
    bool external_rc_shards = false;
    bool internal_parallel_dense = false;
    bool deprecated = false;
};

RouteKey normalize_route(const ggml_gemmini_args_t & args);
RouteCapabilities route_capabilities(const ggml_gemmini_args_t & args);
const char * activation_route_name(ActivationRoute route);
const char * weight_route_name(WeightRoute route);
const char * backend_route_name(BackendRoute route);

}

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
    MatMulStatus status;
    MatMulCapability capability;
};

class MatmulStripeJob;
class MatmulExecution;
class MatmulStripeInput;
struct MatmulStatus;
MatmulStatus prepare_compensation(MatmulStripeJob &);
MatmulStatus execute_compensation_shard(MatmulStripeJob &);
MatmulStatus execute_compensation_shard(MatmulStripeJob &, size_t shard_id, size_t shard_count);

class MatMul {
public:
    explicit MatMul(ggml_gemmini_args_t args);
    explicit MatMul(ggml_gemmini_args_t * args);
    MatMul(MatMul && other) noexcept;
    MatMul & operator=(MatMul && other) noexcept;
    MatMul(const MatMul &) = delete;
    MatMul & operator=(const MatMul &) = delete;

    MatMulResult run_dense();
    MatMulResult run_full();
    MatMulStatus begin_stripes();
    MatMulStatus run_stripe(MatMulStripe stripe);
    MatMulStatus finish_stripes();

    static MatMulCapability stripe_capability(const ggml_gemmini_args_t & args);
    MatMulState state() const;

private:
    friend class MatmulExecution;
    friend class MatmulStripeCollector;
    friend MatmulStatus execute_full(MatmulExecution &);
    friend MatmulStatus finish_execution(MatmulExecution &);
    friend MatmulStatus prepare_compensation(MatmulStripeJob &);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput, std::vector<quants::QactOutlier>);
    friend MatmulStatus execute_compensation_shard(MatmulStripeJob &);
    friend MatmulStatus execute_compensation_shard(MatmulStripeJob &, size_t, size_t);
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus finalize_stripe(MatmulStripeJob &);

    MatMulStatus run_stripe(MatMulStripe stripe, size_t stripe_id);
    MatMulStatus run_staged_stripe(MatMulStripe stripe, size_t stripe_id);
    MatMulResult run_full(quants::dec::DispatchOverride dispatch_override);

    ggml_gemmini_args_t & args();
    const ggml_gemmini_args_t & args() const;

    ggml_gemmini_args_t owned_args_{};
    ggml_gemmini_args_t * args_ptr_ = nullptr;
    size_t first_row_ = 0;
    size_t last_row_begin_ = 0;
    size_t last_row_end_ = 0;
    size_t covered_rows_ = 0;
    bool has_stripes_ = false;
    MatMulState state_ = MatMulState::idle;
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
    MatmulStatusCode code = MatmulStatusCode::success;
    const char * message = "success";
    MatMulCapability capability = MatMulCapability::supported;

    bool ok() const {
        return code == MatmulStatusCode::success;
    }

    explicit operator bool() const { return ok(); }
};

struct MatmulStageMetrics {
    uint64_t nanoseconds = 0;
    size_t count = 0;
};

struct MatmulJobMetrics {
    size_t stripe_id = 0;
    size_t row_begin = 0;
    size_t row_end = 0;
    size_t rc_shards = 0;
    uint64_t la_cycles = 0;
    uint64_t la3_cycles = 0;
    uint64_t sf_cycles = 0;
    uint64_t la3_ns = 0;
    uint64_t sf1_ns = 0;
    MatmulStageMetrics la;
    MatmulStageMetrics sf;
    MatmulStageMetrics handoff;
    MatmulStageMetrics ws;
    MatmulStageMetrics rc_prepare;
    MatmulStageMetrics rc_compute;
    MatmulStageMetrics rc_finalize;
    uint64_t ws_start_ns = 0;
    uint64_t ws_end_ns = 0;
    uint64_t rc_start_ns = 0;
    uint64_t rc_end_ns = 0;
};

class MatmulStripeCollector {
public:
    explicit MatmulStripeCollector(size_t capacity);
    ~MatmulStripeCollector();
    bool start(MatmulExecution & execution);
    MatmulStatus cancel();
    MatmulStatus finish();
    const quants::act::exsia::StripeReadySink * sink() const;
    MatmulStatus status() const;
    std::vector<MatmulJobMetrics> profiles() const;
    quants::QactOutlier captured_outlier(size_t stripe, size_t outlier) const;

private:
    struct CapturedStripe {
        size_t stripe_id;
        size_t row_begin;
        size_t row_end;
        std::vector<quants::QactOutlier> outliers;
        uint64_t la_cycles = 0;
        uint64_t la3_cycles = 0;
        uint64_t sf_cycles = 0;
        uint64_t la3_ns = 0;
        uint64_t sf1_ns = 0;
    };
    static bool on_ready(void *, const quants::act::exsia::StripeReadyEvent &);
    friend MatmulStatus execute_post_fold_pipeline(const ggml_gemmini_args_t &, MatmulStripeCollector &);
    void fail(MatmulStatus status);
    void worker_loop();
    void compensation_loop();
    bool worker_started_ = false;
    bool stop_requested_ = false;
    bool dense_done_ = false;
    std::thread worker_;
    std::thread compensation_worker_;
    MatmulExecution * execution_ = nullptr;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<CapturedStripe> pending_;
    std::deque<std::shared_ptr<MatmulStripeJob>> compensation_pending_;
    std::vector<std::weak_ptr<MatmulStripeJob>> jobs_;
    size_t capacity_;
    size_t in_flight_ = 0;
    std::vector<CapturedStripe> stripes_;
    std::vector<MatmulJobMetrics> profiles_;
    MatmulStatus status_;
    quants::act::exsia::StripeReadySink sink_;
};

enum class MatmulInvocationMode {
    full,
    stripe_sequential,
    stripe_pipeline,
};

enum class MatmulExecutionState {
    empty,
    prepared,
    running,
    finishing,
    completed,
    failed,
};

struct MatmulOptions {
    MatmulInvocationMode mode = MatmulInvocationMode::full;
    size_t stripe_rows = 0;
    size_t dense_threads = 0;
    size_t rc_shards = 0;
    bool validation = false;
    bool profiling = false;
    bool force_row_direct = false;
    bool force_group_k_csc = false;
    size_t job_capacity = 4;
};

class MatmulStripeInput {
public:
    MatmulStripeInput(size_t row_begin, size_t row_end);
    MatmulStripeInput(size_t row_begin, size_t row_end, size_t stripe_id,
                      const int32_t * residual = nullptr, size_t residual_count = 0);
    MatmulStripeInput(size_t row_begin, size_t row_end, size_t stripe_id,
                      const quants::QactOutlier * outliers, size_t outlier_count);
    MatmulStripeInput(const MatmulStripeInput &) = default;
    MatmulStripeInput & operator=(const MatmulStripeInput &) = default;
    MatmulStripeInput(MatmulStripeInput &&) noexcept = default;
    MatmulStripeInput & operator=(MatmulStripeInput &&) noexcept = default;

    size_t row_begin() const;
    size_t row_end() const;
    size_t stripe_id() const;
    const int32_t * residual() const;
    size_t residual_count() const;
    const quants::QactOutlier * outliers() const;
    size_t outlier_count() const;

private:
    size_t row_begin_;
    size_t row_end_;
    size_t stripe_id_;
    const int32_t * residual_;
    size_t residual_count_;
    const quants::QactOutlier * outliers_ = nullptr;
    size_t outlier_count_ = 0;
};

class MatmulExecution {
public:
    MatmulExecution();
    MatmulExecution(const MatmulExecution &) = delete;
    MatmulExecution & operator=(const MatmulExecution &) = delete;
    MatmulExecution(MatmulExecution &&) noexcept = default;
    MatmulExecution & operator=(MatmulExecution &&) noexcept = default;

    MatmulInvocationMode mode() const;
    MatmulExecutionState state() const;
    MatmulStatus status() const;

private:
    friend MatmulExecution prepare_execution(const ggml_gemmini_args_t &, MatmulOptions);
    friend MatmulExecution prepare_execution(ggml_gemmini_args_t *, MatmulOptions);
    friend MatmulStatus prepare_execution(ggml_gemmini_args_t &, const MatmulOptions &, MatmulExecution &);
    friend MatmulStatus execute_full(MatmulExecution &);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput, std::vector<quants::QactOutlier>);
    friend MatmulStatus capture_stripe(MatmulExecution &, const MatmulStripeInput &, MatmulStripeJob &);
    friend MatmulStatus prepare_compensation(MatmulStripeJob &);
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus execute_compensation_shard(MatmulStripeJob &);
    friend MatmulStatus execute_compensation_shard(MatmulStripeJob &, size_t, size_t);
    friend MatmulStatus finalize_stripe(MatmulStripeJob &);
    friend class MatmulStripeCollector;
    friend class MatmulStripeCollector;
    friend class MatmulStripeJob;
    friend MatmulStatus finish_execution(MatmulExecution &);

    MatmulExecution(ggml_gemmini_args_t args, MatmulOptions options);
    MatmulExecution(ggml_gemmini_args_t * args, MatmulOptions options);

    size_t total_rows_;
    MatMul facade_;
    MatmulOptions options_;
    MatmulStatus status_;
    quants::dec::DispatchOverride dispatch_override_ =
        quants::dec::DispatchOverride::automatic;
    MatmulExecutionState state_ = MatmulExecutionState::empty;
    std::shared_ptr<std::mutex> state_mutex_ = std::make_shared<std::mutex>();
    size_t active_jobs_ = 0;
    size_t captured_rows_ = 0;
    size_t finalized_rows_ = 0;
    size_t first_row_ = 0;
    size_t last_row_begin_ = 0;
    size_t last_row_end_ = 0;
    bool has_captures_ = false;
    std::unordered_set<size_t> captured_stripe_ids_;
    std::unique_ptr<MatMul> staged_facade_;
    bool staged_metadata_active_ = false;
};

enum class MatmulDenseState : uint8_t {
    idle,
    running,
    complete,
    failed,
    cancelled,
};

enum class MatmulRcState : uint8_t {
    idle,
    preparing,
    prepared,
    running,
    complete,
    failed,
    cancelled,
};

struct MatmulStripeJobSnapshot {
    MatmulStatus status;
    MatmulJobMetrics metrics;
    MatmulDenseState dense = MatmulDenseState::idle;
    MatmulRcState rc = MatmulRcState::idle;
    size_t expected_shards = 1;
    size_t completed_shards = 0;
    bool captured = false;
    bool finalized = false;
    bool released = false;
};

class MatmulStripeJob {
public:
    MatmulStripeJob();
    MatmulStripeJob(const MatmulStripeJob &) = delete;
    MatmulStripeJob & operator=(const MatmulStripeJob &) = delete;
    MatmulStripeJob(MatmulStripeJob && other) noexcept;
    MatmulStripeJob & operator=(MatmulStripeJob && other) noexcept;
    ~MatmulStripeJob();

    MatmulStatus status() const;
    MatmulJobMetrics metrics() const;
    MatmulStripeJobSnapshot snapshot() const;

private:
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput);
    friend MatmulStripeJob capture_stripe(MatmulExecution &, MatmulStripeInput, std::vector<quants::QactOutlier>);
    friend MatmulStatus prepare_compensation(MatmulStripeJob &);
    friend MatmulStatus execute_dense_stripe(MatmulStripeJob &);
    friend MatmulStatus execute_compensation_shard(MatmulStripeJob &);
    friend MatmulStatus execute_compensation_shard(MatmulStripeJob &, size_t, size_t);
    friend MatmulStatus finalize_stripe(MatmulStripeJob &);
    friend class MatmulStripeCollector;

    MatmulStripeJob(MatmulExecution * execution, MatmulStripeInput input, MatmulStatus status,
                    std::vector<quants::QactOutlier> outliers = {});
    void cancel(MatmulStatus status);
    void release_slot();
    void record_failure(MatmulStatus status, bool dense_branch);

    MatmulExecution * execution_;
    MatmulStripeInput input_;
    MatmulStatus status_;
    MatmulJobMetrics metrics_;
    std::vector<int32_t> staged_residual_;
    std::vector<quants::QactOutlier> compensation_outliers_;
    std::unique_ptr<quants::act::Meta> staged_activation_meta_;
    bool has_captured_outliers_ = false;
    bool owns_slot_ = false;
    bool released_ = false;
    size_t expected_shards_ = 1;
    size_t completed_shards_ = 0;
    bool parallel_shards_ = false;
    std::shared_ptr<std::mutex> shard_mutex_ = std::make_shared<std::mutex>();
    std::condition_variable lifecycle_condition_;
    std::vector<float> compensation_ycom_;
    MatmulDenseState dense_state_ = MatmulDenseState::idle;
    MatmulRcState rc_state_ = MatmulRcState::idle;
    bool captured_ = true;
    bool finalized_ = false;
};

MatmulExecution prepare_execution(const ggml_gemmini_args_t & args, MatmulOptions options = {});
MatmulExecution prepare_execution(ggml_gemmini_args_t * args, MatmulOptions options = {});
MatmulStatus prepare_execution(ggml_gemmini_args_t & args, const MatmulOptions & options,
                               MatmulExecution & execution);
MatmulStatus execute_full(MatmulExecution & execution);
MatmulStatus capture_stripe(MatmulExecution & execution, const MatmulStripeInput & input,
                            MatmulStripeJob & job);
MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input);
MatmulStripeJob capture_stripe(MatmulExecution & execution, MatmulStripeInput input,
                               std::vector<quants::QactOutlier> outliers);
MatmulStatus capture_stripe(MatmulExecution & execution, const MatmulStripeInput & input,
                            MatmulStripeJob & job);
MatmulStatus prepare_compensation(MatmulStripeJob & job);
MatmulStatus execute_dense_stripe(MatmulStripeJob & job);
MatmulStatus execute_compensation_shard(MatmulStripeJob & job);
MatmulStatus execute_compensation_shard(MatmulStripeJob & job, size_t shard_id, size_t shard_count);
MatmulStatus finalize_stripe(MatmulStripeJob & job);
MatmulStatus finish_execution(MatmulExecution & execution);
MatmulStatus matmul(ggml_gemmini_args_t & args, MatmulOptions options = {});
MatmulStatus matmul(const ggml_gemmini_args_t & args, MatmulOptions options = {});
MatmulStatus execute_post_fold_pipeline(const ggml_gemmini_args_t & args, MatmulStripeCollector & collector);

}
