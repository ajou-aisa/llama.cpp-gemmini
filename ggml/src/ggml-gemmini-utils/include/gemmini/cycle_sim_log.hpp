#pragma once

#ifndef CYCLE_SIM
#define CYCLE_SIM 0
#endif
#define GEMMINI_LOG_DEFAULT_NPU_TRACE_PATH "log/npu-cycle-trace.jsonl"

#if CYCLE_SIM
#include <im2p_geometry.h>
#include <gemmini/cpu_log_context.hpp>
#include <gemmini/semantic.hpp>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ggml::gemmini::cycle_sim {

struct RunInfo {
    std::string model, profile, hardware_contract_json, hardware_contract_sha256;
    uint32_t activation_bits = 0, weight_bits = 0, dim = 0;
    uint64_t prompt_tokens = 0, requested_generated_tokens = 0;
    std::string run_config_id = "workload-0";
};
RunInfo compiled_run_info(const std::string &model, uint64_t prompt_tokens,
                          uint64_t requested_generated_tokens);

class Session;
struct Context {
    std::shared_ptr<Session> session;
    uint64_t phase_id = 0;
    std::optional<uint64_t> operation_id, dispatch_id;
    std::optional<uint64_t> node_id{}, call_id{};
    std::optional<uint64_t> host_stage_id{};
    std::shared_ptr<const semantic::Context> semantic_context{};
    explicit operator bool() const noexcept { return bool(session); }
};
Context current_context();
std::shared_ptr<Session> active_session();
Context context_for(const void *node_key);
class ScopedContext {
public:
    explicit ScopedContext(Context context);
    ~ScopedContext() noexcept;
    ScopedContext(const ScopedContext &) = delete;
    ScopedContext &operator=(const ScopedContext &) = delete;
private:
    Context previous_;
    log::CpuCorrelation previous_correlation_;
};

struct Operation {
    std::string layer, operation = "MUL_MAT", actual_backend, activation_type, weight_type;
    uint64_t m = 0, n = 0, k = 0;
    bool target_eligible = false;
    std::string reason{};
    std::shared_ptr<const semantic::Context> semantic_context{};
};
struct Work {
    im2p_production_geometry_v1_t geometry{};
    std::string provenance = "dense_main", scope = "full";
    uint64_t m = 0, row_begin = 0, row_count = 0;
    std::optional<uint64_t> stripe_id, host_slot, original_block_id;
    uint64_t activation_stride_bytes = 0, weight_stride_bytes = 0;
    uint64_t output_stride_bytes = 0, scale_stride_elements = 0;
    uint32_t block_size = 32, vector_op = 5, output_domain = 2;
    uint64_t work_context = 0, source_row_begin = 0, source_row_count = 0;
    uint64_t column_begin = 0, group_index = 0;
    bool rmd_raw = false, host_integer_block_multiply = false;
    std::vector<uint64_t> required_host_stage_ids;
};
struct HostStage {
    std::string stage_name, execution_class, source_owner, source_location;
    std::vector<uint64_t> required_work_ids, required_host_stage_ids;
};

enum class CallKind { Full, Stripe, Fence, ResidualPrepare, ResidualCompact, ResidualRecompose, ResidualMerge };
enum class CallStage { Prepare, Invoke, CompleteRequired, Continuation, Publish, Fence };
struct PolicyQuery {
    void *context = nullptr;
    bool (*eligible)(void *, const void *) = nullptr;
};

class Session : public std::enable_shared_from_this<Session> {
public:
    static std::shared_ptr<Session> start(const RunInfo &info);
    ~Session() noexcept;
    Context phase(const std::string &kind, std::optional<uint64_t> decode_index,
                  uint64_t input_tokens);
    Context register_operation(const void *node_key, const Operation &operation,
                               const Context &phase_context);
    Context new_dispatch(const Context &operation_context);
    Context call_begin(const Context &context, CallKind kind);
    void call_event(const Context &context, CallStage stage,
                    const std::vector<uint64_t> &required_work_ids = {});
    Context host_stage_begin(const Context &context, const HostStage &stage);
    void host_stage_end(const Context &context, bool success = true);
    void set_policy_query(PolicyQuery policy);
    bool target_eligible(const void *node_key);
    Context find_operation(const void *node_key);
    Context current_phase_context();
    Context context_for(const void *node_key) { return find_operation(node_key); }
    bool finish_operation(const Context &context, bool success = true);
    uint64_t work(const Context &context, const Work &work);
    void ensure_healthy();
    void finish(bool success = true, const std::string &reason = "");
    void record_failure(std::string_view reason) noexcept;
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
private:
    struct Impl;
    explicit Session(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
}
#endif
