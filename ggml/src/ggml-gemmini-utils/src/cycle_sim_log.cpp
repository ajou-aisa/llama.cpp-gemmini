#include "cycle_sim_internal.hpp"
#if CYCLE_SIM
#include <gemmini/log.hpp>
#include <cycle-sim-build-config.hpp>
#include <gemmini/cycle_sim_context.h>
#include <algorithm>
#include <utility>

namespace ggml::gemmini::cycle_sim {
namespace {
std::mutex active_mutex;
std::weak_ptr<Session> active;
thread_local Context bound_context;
}

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(std::string("cycle-sim: ") + message);
}
RunInfo compiled_run_info(const std::string &model, uint64_t prompt, uint64_t generated) {
    RunInfo info;
    fill_compiled_cycle_sim_info(info);
    info.model = model;
    info.prompt_tokens = prompt;
    info.requested_generated_tokens = generated;
    return info;
}
Context current_context() { return bound_context; }
std::shared_ptr<Session> active_session() {
    std::lock_guard<std::mutex> lock(active_mutex);
    return active.lock();
}
Context context_for(const void *node_key) {
    const auto session = active_session();
    return session ? session->find_operation(node_key) : Context{};
}
ScopedContext::ScopedContext(Context context) : previous_(std::move(bound_context)) {
    bound_context = std::move(context);
    auto correlation = log::current_cpu_correlation();
    correlation.present = bool(bound_context);
    if (bound_context) {
        correlation.collection_run_id = 1;
        correlation.phase_id = bound_context.phase_id;
        correlation.operation_id = bound_context.operation_id.value_or(UINT64_MAX);
        correlation.target_node_id = bound_context.node_id.value_or(UINT64_MAX);
        correlation.parent_id = bound_context.dispatch_id.value_or(UINT64_MAX);
        correlation.call_id = bound_context.call_id.value_or(UINT64_MAX);
        correlation.host_stage_id = bound_context.host_stage_id.value_or(UINT64_MAX);
        if (bound_context.semantic_context) correlation.semantic_context = bound_context.semantic_context;
    }
    previous_correlation_ = log::exchange_cpu_correlation(correlation);
}
ScopedContext::~ScopedContext() noexcept {
    log::exchange_cpu_correlation(previous_correlation_);
    bound_context = std::move(previous_);
}
Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Session::~Session() noexcept = default;

void Session::Impl::check() const {
    require(failure.empty(), failure.c_str());
    require(!finished, "session already finished");
}
void Session::Impl::emit(const char *kind, const std::string &fields) {
    check();
    const std::string line = "{\"schema\":\"im2p-npu-cycle-trace\",\"version\":2,\"kind\":" +
        json_string(kind) + ",\"sequence\":" + std::to_string(sequence++) +
        ",\"run_id\":\"run-0\",\"collection_run_id\":1,\"run_config_id\":" + json_string(info.run_config_id) +
        ",\"source_role\":" + json_string(semantic::source_name(semantic::Source::PotalCollection)) + fields + "}\n";
    if (std::fwrite(line.data(), 1, line.size(), file) != line.size() || std::fflush(file)) {
        failure = "dedicated log write failed";
        throw std::runtime_error("cycle-sim: " + failure);
    }
}
OperationState &Session::Impl::operation(const Context &context) {
    check();
    require(context.operation_id.has_value(), "operation context required");
    require(context.node_id == context.operation_id, "operation/node identity mismatch");
    auto it = operations.find(*context.operation_id);
    require(it != operations.end() && !it->second.completed, "unknown/completed operation");
    require(context.semantic_context == it->second.descriptor.semantic_context, "semantic context ownership mismatch");
    require(it->second.phase_id == context.phase_id && context.phase_id + 1 == phases,
            "stale operation phase");
    return it->second;
}
std::shared_ptr<Session> Session::start(const RunInfo &info) {
    std::lock_guard<std::mutex> active_lock(active_mutex);
    require(active.expired(), "only one active session is supported");
    require(!info.run_config_id.empty(), "run configuration sidecar reference missing");
    const auto compiled = compiled_run_info(info.model, info.prompt_tokens, info.requested_generated_tokens);
    require(info.profile == compiled.profile && info.hardware_contract_json == compiled.hardware_contract_json &&
            info.hardware_contract_sha256 == compiled.hardware_contract_sha256 &&
            info.activation_bits == compiled.activation_bits && info.weight_bits == compiled.weight_bits &&
            info.dim == compiled.dim, "run metadata differs from compiled target contract");
    auto impl = std::make_unique<Impl>(info);
    const auto path = log::resolve_output_path(GEMMINI_LOG_DEFAULT_NPU_TRACE_PATH);
    require(!path.empty() && log::prepare_output_parent(path), "invalid dedicated log path");
    impl->file = std::fopen(path.string().c_str(), "wx");
    require(impl->file != nullptr, "cannot create dedicated log (existing files are not overwritten)");
    impl->emit("RUN", ",\"model\":" + json_string(info.model) + ",\"profile\":" + json_string(info.profile) +
        ",\"activation_bits\":" + std::to_string(info.activation_bits) + ",\"weight_bits\":" +
        std::to_string(info.weight_bits) + ",\"dim\":" + std::to_string(info.dim) +
        ",\"prompt_tokens\":" + std::to_string(info.prompt_tokens) + ",\"requested_generated_tokens\":" +
        std::to_string(info.requested_generated_tokens) + ",\"producer_execution_kind\":\"CPU_FUNCTIONAL\"" +
        ",\"actual_rtl_acceptance_in_collection\":\"NOT_APPLICABLE\",\"collection_scope\":\"prompt_and_generation\",\"hardware_contract\":" +
        info.hardware_contract_json + ",\"hardware_contract_sha256\":" + json_string(info.hardware_contract_sha256) +
        ",\"residual_work_revision\":\"cross-block-run-aware-v1\"");
    auto result = std::shared_ptr<Session>(new Session(std::move(impl)));
    active = result;
    return result;
}
Context Session::phase(const std::string &kind, std::optional<uint64_t> decode_index, uint64_t tokens) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    require(impl_->registered == impl_->completed, "phase transition with incomplete operations");
    require((kind == "prefill" && !decode_index) || (kind == "decode" && decode_index), "invalid phase");
    Context result{shared_from_this(), impl_->phases++, {}, {}};
    impl_->current_phase_kind = kind;
    impl_->current_decode_index = decode_index;
    impl_->emit("PHASE", ",\"phase_id\":" + std::to_string(result.phase_id) + ",\"phase_kind\":" +
                json_string(kind) + ",\"decode_index\":" + number(decode_index) + ",\"input_tokens\":" + std::to_string(tokens));
    impl_->nodes.clear();
    return result;
}
Context Session::register_operation(const void *key, const Operation &operation, const Context &phase_context) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    require(phase_context.session.get() == this && phase_context.phase_id + 1 == impl_->phases,
            "operation registered in stale/foreign phase");
    require(key != nullptr, "operation node key is null");
    const auto existing = impl_->nodes.find(key);
    require(existing == impl_->nodes.end() || impl_->operations.at(existing->second).completed,
            "operation node already active");
    Operation descriptor = operation;
    if (!descriptor.semantic_context) descriptor.semantic_context = semantic::context_for(key);
    require(bool(descriptor.semantic_context), "pre-partition semantic operation identity missing");
    const auto &semantic_context = *descriptor.semantic_context;
    require(semantic_context.run_config_id == impl_->info.run_config_id &&
            semantic_context.duration_source == semantic::Source::PotalCollection &&
            semantic_context.identity.phase_kind == impl_->current_phase_kind &&
            semantic_context.identity.decode_index == impl_->current_decode_index, "semantic run/phase/source mismatch");
    const uint64_t id = impl_->registered++;
    impl_->nodes[key] = id;
    impl_->operations.emplace(id, OperationState{descriptor, phase_context.phase_id, 0, {}, {}, {}, {}, false});
    return {shared_from_this(), phase_context.phase_id, id, {}, id, {}, {}, descriptor.semantic_context};
}
Context Session::find_operation(const void *key) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    const auto it = impl_->nodes.find(key);
    if (it == impl_->nodes.end() || impl_->operations.at(it->second).completed) return {};
    const auto &operation = impl_->operations.at(it->second);
    return {shared_from_this(), operation.phase_id, it->second, {}, it->second, {}, {}, operation.descriptor.semantic_context};
}
Context Session::current_phase_context() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    if (!impl_->phases) return {};
    return {shared_from_this(), impl_->phases - 1, {}, {}};
}
Context Session::new_dispatch(const Context &context) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(context.session.get() == this, "foreign operation context");
    auto &operation = impl_->operation(context);
    Context result = context;
    result.dispatch_id = impl_->dispatches++;
    if (!operation.parent_id) operation.parent_id = result.dispatch_id;
    impl_->dispatch_operations.emplace(*result.dispatch_id, *result.operation_id);
    return result;
}
Context Session::dispatch_context(const Context &context) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(context.session.get() == this, "foreign dispatch lookup context");
    const auto &operation = impl_->operation(context);
    require(operation.parent_id.has_value(), "operation dispatch is not registered");
    Context result = context;
    result.dispatch_id = operation.parent_id;
    return result;
}
bool Session::finish_operation(const Context &context, bool success) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(context.session.get() == this, "foreign operation context");
    auto &state = impl_->operation(context);
    require(!success || (state.active_calls == 0 && state.active_host_stages == 0), "operation has incomplete calls/host stages");
    const auto &op = state.descriptor;
    const char *classification = !success ? "UNSUPPORTED" : (!op.m || !op.n || !op.k) ? "EXCLUDED" :
        state.works ? "TARGET_NPU" : "ORDINARY_CPU";
    const std::string reason = !op.reason.empty() ? op.reason : !success ? "operation_failed" :
        (!op.m || !op.n || !op.k) ? "empty_operation" : state.works ? "production_gemmini_dispatch" :
        op.target_eligible ? "eligible_operation_routed_to_cpu" : "production_cpu_dispatch";
    impl_->emit("TARGET_OPERATION", semantic_fields(context) + ",\"phase_id\":" + std::to_string(context.phase_id) +
        ",\"operation_id\":" + number(context.operation_id) + ",\"node_id\":" + number(context.node_id) + ",\"layer\":" + json_string(op.layer) +
        ",\"operation\":" + json_string(op.operation) + ",\"actual_backend\":" + json_string(op.actual_backend) +
        ",\"activation_type\":" + json_string(op.activation_type) + ",\"weight_type\":" + json_string(op.weight_type) +
        ",\"m\":" + std::to_string(op.m) + ",\"n\":" + std::to_string(op.n) + ",\"k\":" + std::to_string(op.k) +
        ",\"target_eligible\":" + (op.target_eligible ? "true" : "false") + ",\"selected_target\":" + json_string(classification) +
        ",\"reason\":" + json_string(reason) +
        ",\"status\":" + json_string(success ? "success" : "failed") + ",\"npu_work_count\":" + std::to_string(state.works));
    state.completed = true;
    ++impl_->completed;
    ++impl_->classifications[classification];
    return success && state.works > 0;
}
void Session::ensure_healthy() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
}
void Session::record_failure(std::string_view reason) noexcept {
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->failure.empty()) impl_->failure = reason;
    } catch (...) { std::terminate(); }
}
bool Session::producer_parent_geometry(const Context &context,
                                       const im2p_production_geometry_v1_t &geometry) noexcept {
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        require(context.session.get() == this, "foreign producer parent context");
        auto &operation = impl_->operation(context);
        require(!operation.parent_geometry && geometry.version == IM2P_PRODUCTION_GEOMETRY_VERSION &&
                geometry.struct_size == sizeof(geometry) && geometry.scope == IM2P_GEOMETRY_STREAM &&
                geometry.row_begin == 0 && geometry.row_count == geometry.m &&
                geometry.m == operation.descriptor.m && geometry.n == operation.descriptor.n &&
                geometry.k == operation.descriptor.k && geometry.activation_bits == impl_->info.activation_bits &&
                geometry.weight_bits == impl_->info.weight_bits && geometry.dim == impl_->info.dim &&
                geometry.tile_i_count && geometry.tile_j_count && geometry.tile_k_count,
                "invalid source parent geometry");
        operation.parent_geometry = geometry;
        return true;
    } catch (const std::exception &error) {
        record_failure(error.what());
        return false;
    } catch (...) {
        record_failure("producer parent geometry failed");
        return false;
    }
}
bool Session::producer_event(const Context &context, ProducerEvent event) noexcept {
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        require(context.session.get() == this, "foreign producer context");
        const auto &operation = impl_->operation(context);
        require(event.row_begin < event.row_end &&
                event.workspace_slot && *event.workspace_slot < 2,
                "invalid producer stripe identity");
        require(event.source_location.find(':') != std::string::npos &&
                event.source_location.find("..") == std::string::npos &&
                event.source_location.front() != '/',
                "invalid producer source location");
        const auto key = std::make_tuple(*context.operation_id, event.run_id, event.stripe_id);
        if (event.kind == ProducerEventKind::ResidualHostMergeCompleted) {
            require(event.call_id &&
                    impl_->completed_residual_merge_calls.count(*event.call_id) &&
                    impl_->call_operations.at(*event.call_id) == *context.operation_id,
                    "residual merge call has not completed");
            impl_->stripe_merge_call_ids[key].push_back(*event.call_id);
        }
        std::vector<uint64_t> required_work_ids, required_call_ids;
        if (event.kind == ProducerEventKind::ResidualHostMergeCompleted)
            required_call_ids.push_back(*event.call_id);
        if (event.kind == ProducerEventKind::StreamWorkCompleted ||
            event.kind == ProducerEventKind::ResidualCallbackCompleted ||
            event.kind == ProducerEventKind::FrontendCapacityRelease) {
            const auto work = impl_->stripe_all_work_ids.find({*context.operation_id, event.stripe_id});
            require(work != impl_->stripe_all_work_ids.end() && !work->second.empty(),
                    "producer release without selected stripe work");
            required_work_ids = work->second;
        }
        if (event.kind == ProducerEventKind::ResidualCallbackCompleted ||
            event.kind == ProducerEventKind::FrontendCapacityRelease) {
            const auto calls = impl_->stripe_merge_call_ids.find(key);
            if (calls != impl_->stripe_merge_call_ids.end()) required_call_ids = calls->second;
            require((!event.rmd_packet && !event.direct_residual) || !required_call_ids.empty(),
                    "residual release lacks completed merge call");
        }
        impl_->producer_records.push_back({std::move(event),
            static_cast<uint64_t>(impl_->producer_records.size()), context.phase_id,
            *context.operation_id, operation.parent_id, {},
            std::move(required_work_ids), std::move(required_call_ids)});
        return true;
    } catch (const std::exception &error) {
        record_failure(error.what());
        return false;
    } catch (...) {
        record_failure("producer event failed");
        return false;
    }
}
std::vector<ProducerRecord> Session::producer_events() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(impl_->finished, "producer snapshot before session finish");
    auto records = impl_->producer_records;
    for (auto &record : records) {
        record.parent_id = impl_->operations.at(record.operation_id).parent_id;
        const auto key = std::make_tuple(record.operation_id, record.event.run_id,
                                         record.event.stripe_id);
        const auto work = impl_->stripe_work_ids.find(key);
        require(work != impl_->stripe_work_ids.end(), "producer stripe without selected dense work");
        record.work_id = work->second;
    }
    return records;
}
std::vector<ProducerParent> Session::producer_parents() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(impl_->finished, "producer parent snapshot before session finish");
    require(impl_->pipeline_work_ids.size() == impl_->striped_operations.size(),
            "mixed FULL and STRIPE_PIPELINE target work");
    std::vector<ProducerParent> parents;
    for (const auto operation_id : impl_->striped_operations) {
        const auto &operation = impl_->operations.at(operation_id);
        require(operation.parent_id && operation.parent_geometry && operation.fence_call_id,
                "pipeline work without final parent geometry or fence");
        const auto &work_ids = impl_->pipeline_work_ids.at(operation_id);
        require(std::set<uint64_t>(work_ids.begin(), work_ids.end()) ==
                    operation.fence_required_work_ids,
                "pipeline fence omits parent work");
        parents.push_back({operation_id, *operation.parent_id, *operation.fence_call_id,
                           operation.phase_id, *operation.parent_geometry, work_ids,
                           {operation.fence_required_work_ids.begin(),
                            operation.fence_required_work_ids.end()},
                           impl_->residual_bindings[operation_id]});
    }
    return parents;
}
void Session::finish(bool success, const std::string &reason) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    require(!success || impl_->registered == impl_->completed, "run has incomplete operations");
    impl_->emit("RUN_END", ",\"status\":" + json_string(success ? "success" : "failed") + ",\"reason\":" + json_string(reason) +
        ",\"registered_operation_count\":" + std::to_string(impl_->registered) +
        ",\"completed_operation_count\":" + std::to_string(impl_->completed) +
        ",\"target_npu_count\":" + std::to_string(impl_->classifications["TARGET_NPU"]) +
        ",\"ordinary_cpu_count\":" + std::to_string(impl_->classifications["ORDINARY_CPU"]) +
        ",\"unsupported_count\":" + std::to_string(impl_->classifications["UNSUPPORTED"]) +
        ",\"excluded_count\":" + std::to_string(impl_->classifications["EXCLUDED"]) +
        ",\"npu_work_count\":" + std::to_string(impl_->works) + ",\"call_count\":" + std::to_string(impl_->call_count) +
        ",\"host_stage_count\":" + std::to_string(impl_->host_stage_count) +
        ",\"completed_host_stage_count\":" + std::to_string(impl_->completed_host_stages) +
        ",\"potal_host_count\":" + std::to_string(impl_->classifications["POTAL_HOST"]) +
        ",\"functional_emulation_count\":" + std::to_string(impl_->classifications["FUNCTIONAL_EMULATION"]) +
        ",\"phase_count\":" + std::to_string(impl_->phases));
    impl_->finished = true;
    if (std::fclose(impl_->file)) {
        impl_->file = nullptr;
        throw std::runtime_error("cycle-sim: close failed");
    }
    impl_->file = nullptr;
}
void Session::set_policy_query(PolicyQuery policy) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->check();
    require(policy.eligible != nullptr && impl_->registered == 0, "policy must be installed before target operations");
    impl_->policy = policy;
}
bool Session::target_eligible(const void *node_key) {
    PolicyQuery policy;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->check();
        policy = impl_->policy;
    }
    require(policy.eligible != nullptr, "production target policy was not installed");
    return policy.eligible(policy.context, node_key);
}
}
extern "C" void *gemmini_cycle_sim_context_enter(const void *node_key) {
    using namespace ggml::gemmini::cycle_sim;
    try {
        Context context = node_key ? context_for(node_key) : current_context();
        if (!context) {
            if (const auto session = active_session()) context = session->current_phase_context();
        }
        if (context && !context.semantic_context && node_key)
            context.semantic_context = ggml::gemmini::semantic::context_for(node_key);
        return context ? new ScopedContext(std::move(context)) : nullptr;
    } catch (const std::exception &error) {
        if (const auto session = active_session()) session->record_failure(error.what());
        return nullptr;
    }
}
extern "C" void gemmini_cycle_sim_context_exit(void *scope) {
    delete static_cast<ggml::gemmini::cycle_sim::ScopedContext *>(scope);
}
#endif
