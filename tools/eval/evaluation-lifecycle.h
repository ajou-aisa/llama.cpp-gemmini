#pragma once

#include "../../common/json.hpp"
#if CYCLE_SIM
#include <gemmini/cycle_sim_log.hpp>
#endif
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class evaluation_lifecycle {
public:
    evaluation_lifecycle(const std::string & role, const std::string & source, uint64_t expected,
                         bool forced_cost_only = false, bool pipeline = false)
        : expected_(expected), forced_(forced_cost_only), pipeline_(pipeline) {
        require(!forced_ || (role == "full_cpu" && expected == 128), "forced trajectory requires FullCPU and exactly 128 tokens");
        require(!pipeline_ || (role == "potal_collection" && !forced_), "pipeline declaration requires PoTal free generation");
        nlohmann::json run = {{"source_role", role}, {"source_commit", source}, {"expected_samples", forced_ ? 0 : expected},
            {"execution_policy", "blocking-llama-decode-synchronize-v1"},
            {"graph_policy", "semantic-session-completes-before-next-graph-v1"},
            {"operation_exit_policy", "ALL_MEMBER_COMPLETIONS"},
            {"target_mode_scope", pipeline_ ? "STRIPE_PIPELINE_ONLY" : "FULL_ONLY"},
            {"requires_binary_manifest_binding", true}};
        if (pipeline_) run.update({{"producer_ownership_source", "CPU_FUNCTIONAL"},
            {"actual_rtl_acceptance_in_collection", "NOT_APPLICABLE"},
            {"workspace_slot_domain", "EXSIA_SCRATCH"}, {"target_npu_slot_domain", "UNDECLARED"}});
        if (forced_) run.update({{"execution_kind", "FORCED_CPU_COST_ONLY"}, {"trajectory_source", "POTAL"},
            {"expected_tokens", expected}, {"performance_scope", "ORDINARY_CPU_COST_ONLY"}, {"target_mode_scope", "CPU_ONLY"}});
        emit("RUN", std::move(run));
    }
    void phase(const std::string & kind, std::optional<uint64_t> decode, uint64_t graphs) {
        require(!active_ && graphs == graphs_ && !finished_, "phase while dispatch incomplete");
        require((phase_count_ == 0 && kind == "prefill" && !decode) ||
            (phase_count_ != 0 && kind == "decode" && decode && *decode + 1 == tokens_),
            "phase lacks preceding completed token");
        require(phase_count_ == tokens_, "duplicate phase without completed token");
        phase_ = kind; decode_ = decode; phase_graph_begin_ = graphs;
        emit("PHASE", {{"phase_kind", kind}, {"decode_index", optional(decode)},
            {"graph_begin", graphs}, {"phase_ordinal", phase_count_++}});
    }
    void request_start(uint64_t graphs) {
        require(pipeline_ && !request_started_ && !active_ && !finished_ &&
                phase_ == "prefill" && graphs == graphs_ && dispatches_ == 0,
                "invalid pipeline request start");
        request_started_ = true;
        emit("REQUEST_START", {{"request_id", 0}, {"phase_kind", "prefill"},
            {"decode_index", nullptr}, {"graph_begin", graphs}});
    }
    void prefill_batch_ready(uint64_t batch_index, uint64_t graphs) {
        require(pipeline_ && request_started_ && !active_ && !finished_ &&
                phase_ == "prefill" && graphs == graphs_ &&
                batch_index == prefill_batches_ && dispatches_ == prefill_batches_,
                "invalid prefill batch readiness");
        emit("PREFILL_BATCH_READY", {{"batch_index", batch_index},
            {"dispatch_id", dispatches_}, {"graph_begin", graphs},
            {"phase_kind", "prefill"}, {"decode_index", nullptr}});
        ++prefill_batches_;
    }
    void begin_dispatch(uint64_t graphs) {
        require(!active_ && !phase_.empty() && !finished_ && graphs == graphs_ &&
                (!pipeline_ || (request_started_ &&
                    (phase_ != "prefill" || prefill_batches_ == dispatches_ + 1))),
                "invalid dispatch begin");
        active_ = true;
        emit("DISPATCH_BEGIN", {{"dispatch_id", dispatches_}, {"graph_begin", graphs},
            {"phase_kind", phase_}, {"decode_index", optional(decode_)}});
    }
    void end_dispatch(uint64_t graphs, int status) {
        require(active_ && graphs >= graphs_ && !finished_, "invalid dispatch end");
        emit("DISPATCH_END", {{"dispatch_id", dispatches_++}, {"graph_end", graphs},
            {"status", status}, {"synchronized", true}, {"phase_kind", phase_},
            {"decode_index", optional(decode_)}});
        graphs_ = graphs; active_ = false;
    }
    void sample(uint64_t index, int32_t token) {
        require(!forced_, "forced cost-only execution cannot record sampling");
        require_token(index);
        emit("SAMPLE", {{"sample_index", index}, {"token_id", token}, {"graph_end", graphs_},
            {"dispatch_id", dispatches_ - 1}, {"token_ready", true},
            {"phase_kind", phase_}, {"decode_index", optional(decode_)}});
        ++samples_; ++tokens_;
    }
    void forced_token(uint64_t index, int32_t token) {
        require(forced_, "free generation cannot record a forced token");
        require_token(index);
        emit("FORCED_TOKEN", {{"token_index", index}, {"token_id", token}, {"graph_end", graphs_},
            {"dispatch_id", dispatches_ - 1}, {"token_ready", true}, {"actual_sampling", false},
            {"phase_kind", phase_}, {"decode_index", optional(decode_)}});
        ++tokens_;
    }
    void pipeline_parent(nlohmann::json fields) {
        require(pipeline_ && !active_ && !finished_, "invalid pipeline parent declaration");
        const uint64_t operation_id = fields.at("operation_id").get<uint64_t>();
        const uint64_t parent_id = fields.at("parent_id").get<uint64_t>();
        const auto work_ids = fields.at("required_work_ids").get<std::vector<uint64_t>>();
        const auto fence_work_ids = fields.at("fence_required_work_ids").get<std::vector<uint64_t>>();
        (void) fields.at("fence_call_id").get<uint64_t>();
        require(!work_ids.empty() && pipeline_parents_.count(operation_id) == 0,
                "duplicate or empty pipeline parent");
        const std::set<uint64_t> unique(work_ids.begin(), work_ids.end());
        require(unique.size() == work_ids.size() &&
                std::set<uint64_t>(fence_work_ids.begin(), fence_work_ids.end()) == unique &&
                fence_work_ids.size() == unique.size(), "pipeline fence work mismatch");
        pipeline_parents_.emplace(operation_id, std::make_pair(parent_id, unique));
        emit("PIPELINE_PARENT", std::move(fields));
    }
    void pipeline_owner(nlohmann::json fields) {
        require(pipeline_ && !active_ && !finished_, "invalid pipeline owner declaration");
        const uint64_t operation_id = fields.at("operation_id").get<uint64_t>();
        const auto parent = pipeline_parents_.find(operation_id);
        require(parent != pipeline_parents_.end() &&
                fields.at("parent_id").get<uint64_t>() == parent->second.first &&
                parent->second.second.count(fields.at("work_id").get<uint64_t>()) &&
                fields.at("producer_sequence").get<uint64_t>() == owner_count_ &&
                fields.at("workspace_slot").get<uint64_t>() < 2 &&
                fields.at("target_npu_slot").is_null() &&
                fields.at("row_begin").get<uint64_t>() < fields.at("row_end").get<uint64_t>(),
                "pipeline owner identity mismatch");
        for (const auto &id : fields.at("required_work_ids"))
            require(parent->second.second.count(id.get<uint64_t>()), "pipeline owner requires foreign work");
        ++owner_count_;
        emit("PIPELINE_OWNER", std::move(fields));
    }
#if CYCLE_SIM
    void pipeline_session(ggml::gemmini::cycle_sim::Session &target) {
        require(pipeline_, "pipeline session in non-pipeline lifecycle");
        for (const auto &parent : target.producer_parents()) {
            const auto &g = parent.geometry;
            nlohmann::json residual_bindings = nlohmann::json::array();
            for (const auto &binding : parent.residual_bindings)
                residual_bindings.push_back({{"work_id", binding.work_id},
                    {"call_id", binding.call_id}, {"child_parent_id", binding.child_parent_id},
                    {"dense_work_id", binding.dense_work_id},
                    {"dense_parent_id", binding.dense_parent_id},
                    {"stripe_id", binding.stripe_id}, {"row_begin", binding.row_begin},
                    {"row_end", binding.row_end}, {"source_row_begin", binding.source_row_begin},
                    {"source_row_count", binding.source_row_count}});
            pipeline_parent({{"phase_id", parent.phase_id},
                {"operation_id", parent.operation_id}, {"parent_id", parent.parent_id},
                {"required_work_ids", parent.work_ids}, {"fence_call_id", parent.fence_call_id},
                {"fence_required_work_ids", parent.fence_required_work_ids},
                {"residual_bindings", std::move(residual_bindings)},
                {"production_geometry_version", g.version},
                {"scope", "STREAM"}, {"activation_bits", g.activation_bits},
                {"weight_bits", g.weight_bits}, {"dim", g.dim}, {"parent_m", g.m},
                {"n", g.n}, {"k", g.k}, {"tile_i_count", g.tile_i_count},
                {"tile_j_count", g.tile_j_count}, {"tile_k_count", g.tile_k_count}});
        }
        for (const auto &record : target.producer_events()) {
            require(record.parent_id && record.work_id, "pipeline producer identity incomplete");
            const auto [resource, transition] = producer_label(record.event.kind);
            pipeline_owner({{"producer_sequence", record.sequence},
                {"phase_id", record.phase_id}, {"operation_id", record.operation_id},
                {"parent_id", *record.parent_id}, {"work_id", *record.work_id},
                {"producer_run_id", record.event.run_id}, {"stripe_id", record.event.stripe_id},
                {"workspace_slot", *record.event.workspace_slot}, {"target_npu_slot", nullptr},
                {"row_begin", record.event.row_begin}, {"row_end", record.event.row_end},
                {"resource", resource}, {"transition", transition},
                {"rmd_packet", record.event.rmd_packet}, {"direct_residual", record.event.direct_residual},
                {"required_work_ids", record.required_work_ids},
                {"required_call_ids", record.required_call_ids},
                {"observed_call_id", record.event.call_id
                    ? nlohmann::json(*record.event.call_id) : nlohmann::json()},
                {"source_owner", record.event.source_location.rfind("frontend/", 0) == 0
                    ? "IM2P.sim" : "llama.cpp-gemmini"},
                {"source_location", record.event.source_location}});
        }
    }
#endif
    const std::vector<nlohmann::json> & finish(bool success, uint64_t graphs) {
        require(!active_ && !finished_ && graphs == graphs_, "finish while dispatch incomplete");
        require(!success || (tokens_ == expected_ && phase_count_ == tokens_ &&
                            (!pipeline_ || (request_started_ && prefill_batches_ > 0 &&
                                            !pipeline_parents_.empty()))),
                "incomplete token lifecycle");
        nlohmann::json end = {{"success", success}, {"samples", samples_}, {"phases", phase_count_},
            {"dispatches", dispatches_}, {"graphs", graphs}};
        if (forced_) end.update({{"forced_tokens", tokens_}, {"completed_tokens", tokens_}});
        emit("RUN_END", std::move(end));
        finished_ = true;
        return events_;
    }
private:
#if CYCLE_SIM
    static std::pair<const char *, const char *> producer_label(
            ggml::gemmini::cycle_sim::ProducerEventKind kind) {
        using Kind = ggml::gemmini::cycle_sim::ProducerEventKind;
        switch (kind) {
        case Kind::ExsiaWorkspaceAcquire: return {"EXSIA_WORKSPACE", "ACQUIRE"};
        case Kind::ActivationRowsCommit: return {"ACTIVATION_ROWS", "COMMIT"};
        case Kind::ResidualPacketSeal: return {"RESIDUAL_PAYLOAD", "SEAL"};
        case Kind::FrontendCapacityAcquire: return {"FRONTEND_OUTSTANDING", "ACQUIRE"};
        case Kind::FrontendQueueEnqueue: return {"FRONTEND_QUEUE", "ENQUEUE"};
        case Kind::FrontendQueueDequeue: return {"FRONTEND_QUEUE", "DEQUEUE"};
        case Kind::StreamWorkAccepted: return {"CPU_FUNCTIONAL_STREAM", "ACCEPTED"};
        case Kind::StreamWorkCompleted: return {"CPU_FUNCTIONAL_STREAM", "COMPLETED"};
        case Kind::ResidualHostMergeCompleted: return {"RESIDUAL_MERGE_CALL", "COMPLETE"};
        case Kind::ResidualCallbackCompleted: return {"RESIDUAL_CALLBACK", "COMPLETE"};
        case Kind::FrontendCapacityRelease: return {"FRONTEND_OUTSTANDING", "RELEASE"};
        case Kind::ExsiaWorkspaceRelease: return {"EXSIA_WORKSPACE", "RELEASE"};
        }
        throw std::runtime_error("unknown producer event");
    }
#endif
    void require_token(uint64_t index) const {
        require(!active_ && !finished_ && index == tokens_ && phase_count_ == tokens_ + 1 &&
            graphs_ > phase_graph_begin_, "token completion outside completed phase");
    }
    static void require(bool condition, const char * message) {
        if (!condition) throw std::runtime_error(std::string("evaluation lifecycle: ") + message);
    }
    static nlohmann::json optional(std::optional<uint64_t> value) {
        return value ? nlohmann::json(*value) : nlohmann::json();
    }
    void emit(const char * kind, nlohmann::json fields) {
        fields.update({{"schema", "potal-execution-lifecycle"}, {"version", pipeline_ ? 3 : forced_ ? 2 : 1},
            {"kind", kind}, {"sequence", events_.size()}});
        events_.push_back(std::move(fields));
    }
    std::vector<nlohmann::json> events_;
    std::string phase_;
    std::optional<uint64_t> decode_;
    uint64_t expected_ = 0, graphs_ = 0, phase_count_ = 0, samples_ = 0, tokens_ = 0, dispatches_ = 0, phase_graph_begin_ = 0;
    uint64_t owner_count_ = 0;
    uint64_t prefill_batches_ = 0;
    std::map<uint64_t, std::pair<uint64_t, std::set<uint64_t>>> pipeline_parents_;
    bool active_ = false, finished_ = false, forced_ = false, pipeline_ = false, request_started_ = false;
};
