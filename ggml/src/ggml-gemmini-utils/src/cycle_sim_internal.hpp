#pragma once
#include <gemmini/cycle_sim_log.hpp>
#if CYCLE_SIM
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <set>
#include <tuple>

namespace ggml::gemmini::cycle_sim {
std::string json_string(const std::string &value);
std::string number(std::optional<uint64_t> value);
void require(bool condition, const char *message);
std::string work_fields(const Work &work);
std::string semantic_fields(const Context &context);
struct OperationState {
    Operation descriptor;
    uint64_t phase_id = 0, works = 0;
    std::optional<uint64_t> parent_id;
    std::optional<im2p_production_geometry_v1_t> parent_geometry;
    std::optional<uint64_t> fence_call_id;
    std::set<uint64_t> fence_required_work_ids;
    bool completed = false;
    uint64_t active_calls = 0;
    uint64_t active_host_stages = 0;
};
struct CallState {
    uint64_t operation_id;
    std::optional<uint64_t> parent_id;
    CallKind kind;
    std::set<CallStage> stages;
    std::set<uint64_t> own_work_ids, required_work_ids;
};
struct HostStageState {
    HostStage descriptor;
    uint64_t operation_id, phase_id;
    std::optional<uint64_t> parent_id, call_id;
    bool complete = false;
};
struct ProducerDenseStripe {
    uint64_t work_id, parent_id, row_begin, row_end;
};
struct Session::Impl {
    explicit Impl(RunInfo value) : info(std::move(value)) {}
    ~Impl() { if (file) std::fclose(file); }
    RunInfo info;
    FILE *file = nullptr;
    std::mutex mutex;
    bool finished = false;
    std::string failure;
    uint64_t sequence = 0, phases = 0, works = 0, dispatches = 0, call_count = 0;
    uint64_t registered = 0, completed = 0;
    uint64_t host_stage_count = 0, completed_host_stages = 0;
    std::map<std::string, uint64_t> classifications;
    std::unordered_map<const void *, uint64_t> nodes;
    std::map<uint64_t, OperationState> operations;
    std::map<uint64_t, uint64_t> dispatch_operations;
    std::map<uint64_t, CallState> calls;
    std::map<uint64_t, uint64_t> call_operations;
    std::set<uint64_t> completed_residual_merge_calls;
    std::map<uint64_t, uint64_t> work_operations;
    std::map<std::tuple<uint64_t, uint64_t, uint64_t>, uint64_t> stripe_work_ids;
    std::map<std::pair<uint64_t, uint64_t>, ProducerDenseStripe> dense_stripes;
    std::map<uint64_t, std::vector<ProducerResidualBinding>> residual_bindings;
    std::map<std::pair<uint64_t, uint64_t>, std::vector<uint64_t>> stripe_all_work_ids;
    std::map<std::tuple<uint64_t, uint64_t, uint64_t>, std::vector<uint64_t>> stripe_merge_call_ids;
    std::set<uint64_t> striped_operations;
    std::map<uint64_t, std::vector<uint64_t>> pipeline_work_ids;
    std::vector<ProducerRecord> producer_records;
    std::map<uint64_t, HostStageState> host_stages;
    PolicyQuery policy;
    std::string current_phase_kind;
    std::optional<uint64_t> current_decode_index;
    void check() const;
    void emit(const char *kind, const std::string &fields);
    OperationState &operation(const Context &context);
    void emit_call(const Context &context, CallStage stage, const std::vector<uint64_t> &work_ids);
    void emit_host_stage(const Context &context, const char *event, const char *status);
};
}
#endif
