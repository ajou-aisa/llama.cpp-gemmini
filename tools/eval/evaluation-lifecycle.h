#pragma once

#include "json.hpp"
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

class evaluation_lifecycle {
public:
    evaluation_lifecycle(const std::string & role, const std::string & source, uint64_t expected,
                         bool forced_cost_only = false)
        : expected_(expected), forced_(forced_cost_only) {
        require(!forced_ || (role == "full_cpu" && expected == 128), "forced trajectory requires FullCPU and exactly 128 tokens");
        nlohmann::json run = {{"source_role", role}, {"source_commit", source}, {"expected_samples", forced_ ? 0 : expected},
            {"execution_policy", "blocking-llama-decode-synchronize-v1"},
            {"graph_policy", "semantic-session-completes-before-next-graph-v1"},
            {"operation_exit_policy", "ALL_MEMBER_COMPLETIONS"}, {"target_mode_scope", "FULL_ONLY"},
            {"requires_binary_manifest_binding", true}};
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
    void begin_dispatch(uint64_t graphs) {
        require(!active_ && !phase_.empty() && !finished_ && graphs == graphs_, "invalid dispatch begin");
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
    const std::vector<nlohmann::json> & finish(bool success, uint64_t graphs) {
        require(!active_ && !finished_ && graphs == graphs_, "finish while dispatch incomplete");
        require(!success || (tokens_ == expected_ && phase_count_ == tokens_), "incomplete token lifecycle");
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
        fields.update({{"schema", "potal-execution-lifecycle"}, {"version", forced_ ? 2 : 1},
            {"kind", kind}, {"sequence", events_.size()}});
        events_.push_back(std::move(fields));
    }
    std::vector<nlohmann::json> events_;
    std::string phase_;
    std::optional<uint64_t> decode_;
    uint64_t expected_ = 0, graphs_ = 0, phase_count_ = 0, samples_ = 0, tokens_ = 0, dispatches_ = 0, phase_graph_begin_ = 0;
    bool active_ = false, finished_ = false, forced_ = false;
};
