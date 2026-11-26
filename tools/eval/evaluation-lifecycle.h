#pragma once

#include "../../common/json.hpp"
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
        require(!forced_ || (role == "full_cpu" && (expected == 128 || expected == 1)),
                "forced trajectory requires FullCPU and exactly 128 tokens (or the 1-token diagnostic smoke)");
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
