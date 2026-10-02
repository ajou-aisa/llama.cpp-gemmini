#include "cycle_sim_internal.hpp"
#if CYCLE_SIM
#include <filesystem>
#include <sstream>

namespace ggml::gemmini::cycle_sim {
std::string semantic_fields(const Context &context) {
    require(bool(context.semantic_context), "pre-partition semantic context missing");
    return semantic::identity_fields(context.semantic_context->identity);
}
namespace {
std::string ids(const std::vector<uint64_t> &values) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
    out << ']';
    return out.str();
}
void validate_source(const HostStage &stage) {
    require(!stage.stage_name.empty(), "host stage name missing");
    require(stage.execution_class == "POTAL_HOST" || stage.execution_class == "FUNCTIONAL_EMULATION",
            "invalid host execution class");
    require(stage.source_owner == "llama.cpp-gemmini" || stage.source_owner == "IM2P.sim", "invalid host source owner");
    const auto separator = stage.source_location.find(':');
    require(stage.source_location.find('\\') == std::string::npos, "portable relative source location required");
    require(stage.source_location.find(":/") == std::string::npos, "absolute source location forbidden");
    require(separator != std::string::npos && separator > 0 && separator + 1 < stage.source_location.size(),
            "relative source file and function required");
    const std::filesystem::path path(stage.source_location.substr(0, separator));
    require(!path.is_absolute(), "absolute source location forbidden");
    for (const auto &part : path) require(part != "..", "source traversal forbidden");
}
}
void Session::Impl::emit_host_stage(const Context &context, const char *event, const char *status) {
    const auto &stage = host_stages.at(*context.host_stage_id).descriptor;
    emit("HOST_STAGE", semantic_fields(context) + ",\"phase_id\":" + std::to_string(context.phase_id) +
        ",\"operation_id\":" + number(context.operation_id) + ",\"node_id\":" + number(context.node_id) +
        ",\"parent_id\":" + number(context.dispatch_id) + ",\"call_id\":" + number(context.call_id) +
        ",\"host_stage_id\":" + number(context.host_stage_id) + ",\"execution_class\":" + json_string(stage.execution_class) +
        ",\"stage_name\":" + json_string(stage.stage_name) + ",\"source_owner\":" + json_string(stage.source_owner) +
        ",\"source_location\":" + json_string(stage.source_location) +
        ",\"required_work_ids\":" + ids(stage.required_work_ids) +
        ",\"required_host_stage_ids\":" + ids(stage.required_host_stage_ids) +
        ",\"event\":" + json_string(event) + ",\"status\":" + json_string(status));
}
Context Session::host_stage_begin(const Context &context, const HostStage &stage) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(context.session.get() == this, "foreign host stage context");
    auto &operation = impl_->operation(context);
    validate_source(stage);
    if (context.dispatch_id) require(impl_->dispatch_operations.count(*context.dispatch_id) &&
        impl_->dispatch_operations.at(*context.dispatch_id) == *context.operation_id, "host stage parent mismatch");
    if (context.call_id) require(impl_->call_operations.count(*context.call_id) &&
        impl_->call_operations.at(*context.call_id) == *context.operation_id, "host stage call mismatch");
    std::set<uint64_t> unique;
    for (const auto id : stage.required_work_ids) {
        require(unique.insert(id).second && impl_->work_operations.count(id) &&
            impl_->work_operations.at(id) == *context.operation_id, "unknown/duplicate/cross-operation host work dependency");
    }
    unique.clear();
    for (const auto id : stage.required_host_stage_ids) {
        require(unique.insert(id).second && impl_->host_stages.count(id), "unknown/duplicate host stage dependency");
        const auto &dependency = impl_->host_stages.at(id);
        require(dependency.complete && dependency.operation_id == *context.operation_id,
                "incomplete/cross-operation host stage dependency");
    }
    Context result = context;
    result.host_stage_id = impl_->host_stage_count++;
    impl_->host_stages.emplace(*result.host_stage_id, HostStageState{stage, *context.operation_id,
        context.phase_id, context.dispatch_id, context.call_id, false});
    impl_->emit_host_stage(result, "BEGIN", "declared");
    ++operation.active_host_stages;
    ++impl_->classifications[stage.execution_class];
    return result;
}
void Session::host_stage_end(const Context &context, bool success) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(context.session.get() == this, "foreign host stage completion");
    auto &operation = impl_->operation(context);
    require(context.host_stage_id && impl_->host_stages.count(*context.host_stage_id), "undeclared host stage");
    auto &stage = impl_->host_stages.at(*context.host_stage_id);
    require(!stage.complete && stage.operation_id == *context.operation_id && stage.phase_id == context.phase_id &&
            stage.parent_id == context.dispatch_id && stage.call_id == context.call_id, "host stage ownership/completion mismatch");
    impl_->emit_host_stage(context, "END", success ? "success" : "failed");
    stage.complete = true;
    --operation.active_host_stages;
    ++impl_->completed_host_stages;
    if (!success) impl_->failure = "host stage failed";
}
}
#endif
