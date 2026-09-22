#include "cycle_sim_internal.hpp"
#if CYCLE_SIM
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace ggml::gemmini::cycle_sim {
std::string json_string(const std::string &value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\') out << '\\' << byte;
        else if (byte < 0x20) out << "\\u" << std::hex << std::setfill('0') << std::setw(4) << unsigned(byte);
        else out << byte;
    }
    out << '"';
    return out.str();
}
std::string number(std::optional<uint64_t> value) {
    return value ? std::to_string(*value) : "null";
}
std::string work_fields(const Work &w) {
    const auto &g = w.geometry;
    require(g.version == IM2P_PRODUCTION_GEOMETRY_VERSION && g.struct_size == sizeof(g), "invalid geometry ABI");
    require(g.m && g.n && g.k && g.tile_i_count && g.tile_j_count && g.tile_k_count, "missing exact final geometry");
    require(w.m == g.row_count && w.row_count == g.row_count && w.row_begin == g.row_begin &&
            w.row_begin <= g.m && w.row_count <= g.m - w.row_begin, "work/geometry row mismatch");
    require(w.provenance == "dense_main" || w.provenance == "residual", "invalid work provenance");
    require(!w.rmd_raw && !w.host_integer_block_multiply, "non-production residual semantics");
    require(!w.host_slot || *w.host_slot <= 1, "invalid target host slot");
    if (w.provenance == "residual") {
        require(w.scope == "residual_compact" && !w.original_block_id && w.original_k &&
                !w.runs.empty() && w.row_map.size() == w.m && w.source_row_count &&
                w.source_row_count <= UINT32_MAX && g.m <= UINT32_MAX &&
                g.n <= UINT32_MAX && g.k <= UINT32_MAX,
                "run-aware residual metadata required");
        uint64_t cursor = 0;
        uint64_t previous_block = 0;
        for (size_t index = 0; index < w.runs.size(); ++index) {
            const auto &run = w.runs[index];
            uint32_t bits = run.original_k_mask;
            uint32_t selected = 0;
            while (bits) { ++selected; bits &= bits - 1; }
            const uint64_t start = uint64_t(run.original_block_id) * 32;
            const uint32_t available = start < *w.original_k
                ? static_cast<uint32_t>(std::min<uint64_t>(32, *w.original_k - start)) : 0;
            const uint32_t valid_mask = available == 32 ? UINT32_MAX :
                                        available == 0 ? 0 : (uint32_t{1} << available) - 1;
            require((index == 0 || run.original_block_id > previous_block) &&
                    run.original_k_mask && !(run.original_k_mask & ~valid_mask) &&
                    selected == run.compact_k_count && run.compact_k_begin == cursor,
                    "invalid original block or compact run coverage");
            previous_block = run.original_block_id;
            cursor += run.compact_k_count;
        }
        require(cursor == w.geometry.k, "compact runs do not cover logical K");
        uint64_t previous_row_key = 0;
        for (size_t index = 0; index < w.row_map.size(); ++index) {
            const auto &row = w.row_map[index];
            const uint64_t key = uint64_t(row.lane_id) * w.source_row_count + row.source_row;
            require(row.source_row < w.source_row_count &&
                    row.lane_id < 32 / w.geometry.activation_bits + 1 &&
                    (index == 0 || key > previous_row_key), "invalid global residual row map");
            previous_row_key = key;
        }
    } else {
        require(!w.original_k && w.runs.empty() && w.row_map.empty() &&
                !w.original_block_id, "dense work cannot carry residual run metadata");
    }
    if (w.scope == "stripe") {
        require(w.stripe_id && w.host_slot && *w.stripe_id == g.stripe_id && g.scope == IM2P_GEOMETRY_STRIPE,
                "stripe identity/scope mismatch");
    } else {
        require(w.scope == "full" || w.scope == "residual_compact", "invalid work scope");
        require(w.row_begin == 0 && w.row_count == g.m && g.scope == IM2P_GEOMETRY_FULL,
                "non-stripe work lacks complete range");
    }
    std::ostringstream out;
    out << ",\"provenance\":" << json_string(w.provenance) << ",\"scope\":" << json_string(w.scope)
        << ",\"activation_bits\":" << g.activation_bits << ",\"weight_bits\":" << g.weight_bits
        << ",\"dim\":" << g.dim << ",\"m\":" << w.m << ",\"n\":" << g.n << ",\"k\":" << g.k
        << ",\"tile_i_count\":" << g.tile_i_count << ",\"tile_j_count\":" << g.tile_j_count
        << ",\"tile_k_count\":" << g.tile_k_count << ",\"parent_m\":" << g.m
        << ",\"production_geometry_version\":" << g.version
        << ",\"row_begin\":" << w.row_begin << ",\"row_count\":" << w.row_count
        << ",\"stripe_id\":" << number(w.stripe_id) << ",\"host_slot\":" << number(w.host_slot)
        << ",\"original_block_id\":" << number(w.original_block_id)
        << ",\"activation_stride_bytes\":" << w.activation_stride_bytes
        << ",\"weight_stride_bytes\":" << w.weight_stride_bytes
        << ",\"output_stride_bytes\":" << w.output_stride_bytes
        << ",\"scale_stride_elements\":" << w.scale_stride_elements
        << ",\"block_size\":" << w.block_size << ",\"vector_op\":" << w.vector_op
        << ",\"output_domain\":" << w.output_domain << ",\"work_context\":" << w.work_context
        << ",\"source_row_begin\":" << w.source_row_begin << ",\"source_row_count\":" << w.source_row_count
        << ",\"column_begin\":" << w.column_begin << ",\"group_index\":" << w.group_index
        << ",\"rmd_raw\":false,\"host_integer_block_multiply\":false,\"required_host_stage_ids\":[";
    for (size_t i = 0; i < w.required_host_stage_ids.size(); ++i)
        out << (i ? "," : "") << w.required_host_stage_ids[i];
    out << "],\"original_k\":" << (w.original_k ? std::to_string(*w.original_k) : "null")
        << ",\"residual_work_revision\":"
        << (w.provenance == "residual" ? json_string("cross-block-run-aware-v1") : "null")
        << ",\"runs\":[";
    for (size_t i = 0; i < w.runs.size(); ++i) {
        const auto &run = w.runs[i];
        out << (i ? "," : "") << "{\"original_block_id\":" << run.original_block_id
            << ",\"original_k_mask\":" << run.original_k_mask
            << ",\"compact_k_begin\":" << run.compact_k_begin
            << ",\"compact_k_count\":" << run.compact_k_count << '}';
    }
    out << "],\"row_map\":[";
    for (size_t i = 0; i < w.row_map.size(); ++i) {
        const auto &row = w.row_map[i];
        out << (i ? "," : "") << "{\"source_row\":" << row.source_row
            << ",\"lane_id\":" << row.lane_id << '}';
    }
    out << ']';
    return out.str();
}
uint64_t Session::work(const Context &context, const Work &work) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(context.session.get() == this, "foreign work context");
    auto &operation = impl_->operation(context);
    require(context.call_id && impl_->calls.count(*context.call_id), "declared target call required");
    auto &call = impl_->calls.at(*context.call_id);
    require(call.operation_id == *context.operation_id && call.parent_id == context.dispatch_id &&
            call.stages.count(CallStage::Invoke) && !call.stages.count(CallStage::CompleteRequired) &&
            call.own_work_ids.empty(), "call/work ownership or ordering mismatch");
    require((call.kind == CallKind::Full && work.scope == "full") ||
            (call.kind == CallKind::Stripe && work.scope == "stripe") ||
            (call.kind == CallKind::ResidualCompact && work.scope == "residual_compact"), "call/work kind mismatch");
    require(context.dispatch_id && impl_->dispatch_operations.count(*context.dispatch_id) &&
            impl_->dispatch_operations.at(*context.dispatch_id) == *context.operation_id, "undeclared/foreign dispatch");
    require(work.geometry.activation_bits == impl_->info.activation_bits &&
            work.geometry.weight_bits == impl_->info.weight_bits && work.geometry.dim == impl_->info.dim, "work profile mismatch");
    std::set<uint64_t> stage_ids;
    for (const auto id : work.required_host_stage_ids) {
        require(stage_ids.insert(id).second && impl_->host_stages.count(id), "unknown/duplicate work host dependency");
        const auto &stage = impl_->host_stages.at(id);
        require(stage.complete && stage.operation_id == *context.operation_id, "incomplete/cross-operation work host dependency");
    }
    const auto fields = work_fields(work);
    const uint64_t id = impl_->works++;
    impl_->emit("NPU_WORK", semantic_fields(context) + ",\"phase_id\":" + std::to_string(context.phase_id) +
        ",\"operation_id\":" + number(context.operation_id) + ",\"parent_id\":" + number(context.dispatch_id) +
        ",\"node_id\":" + number(context.node_id) + ",\"call_id\":" + number(context.call_id) +
        ",\"work_id\":" + std::to_string(id) + ",\"layer\":" + json_string(operation.descriptor.layer) +
        ",\"operation\":" + json_string(operation.descriptor.operation) + fields +
        ",\"hardware_contract_sha256\":" + json_string(impl_->info.hardware_contract_sha256));
    ++operation.works;
    impl_->work_operations.emplace(id, *context.operation_id);
    call.own_work_ids.insert(id);
    return id;
}
namespace {
const char *kind_name(CallKind kind) {
    switch (kind) {
    case CallKind::Full: return "FULL";
    case CallKind::Stripe: return "STRIPE";
    case CallKind::Fence: return "FENCE";
    case CallKind::ResidualPrepare: return "RESIDUAL_PREPARE";
    case CallKind::ResidualCompact: return "RESIDUAL_COMPACT";
    case CallKind::ResidualRecompose: return "RESIDUAL_RECOMPOSE";
    case CallKind::ResidualMerge: return "RESIDUAL_MERGE";
    }
    throw std::runtime_error("cycle-sim: unknown call kind");
}
const char *stage_name(CallStage stage) {
    switch (stage) {
    case CallStage::Prepare: return "PREPARE";
    case CallStage::Invoke: return "INVOKE";
    case CallStage::CompleteRequired: return "COMPLETE_REQUIRED";
    case CallStage::Continuation: return "CONTINUATION";
    case CallStage::Publish: return "PUBLISH";
    case CallStage::Fence: return "FENCE";
    }
    throw std::runtime_error("cycle-sim: unknown call stage");
}
}
void Session::Impl::emit_call(const Context &context, CallStage stage, const std::vector<uint64_t> &work_ids) {
    std::ostringstream ids;
    ids << '[';
    for (size_t i = 0; i < work_ids.size(); ++i) ids << (i ? "," : "") << work_ids[i];
    ids << ']';
    emit("NPU_CALL", semantic_fields(context) + ",\"phase_id\":" + std::to_string(context.phase_id) +
        ",\"operation_id\":" + number(context.operation_id) + ",\"node_id\":" + number(context.node_id) +
        ",\"parent_id\":" + number(context.dispatch_id) + ",\"call_id\":" + number(context.call_id) +
        ",\"call_kind\":" + json_string(kind_name(calls.at(*context.call_id).kind)) +
        ",\"stage\":" + json_string(stage_name(stage)) + ",\"required_work_ids\":" + ids.str());
}
Context Session::call_begin(const Context &context, CallKind kind) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(context.session.get() == this, "foreign call context");
    auto &operation = impl_->operation(context);
    kind_name(kind);
    if (context.dispatch_id) require(impl_->dispatch_operations.at(*context.dispatch_id) == *context.operation_id,
                                     "foreign call parent");
    Context result = context;
    result.call_id = impl_->call_count++;
    impl_->calls.emplace(*result.call_id, CallState{*context.operation_id, context.dispatch_id, kind,
                         {CallStage::Prepare}, {}, {}});
    impl_->call_operations.emplace(*result.call_id, *context.operation_id);
    impl_->emit_call(result, CallStage::Prepare, {});
    ++operation.active_calls;
    return result;
}
void Session::call_event(const Context &context, CallStage stage, const std::vector<uint64_t> &work_ids) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(context.session.get() == this, "foreign call context");
    auto &operation = impl_->operation(context);
    require(context.call_id && impl_->calls.count(*context.call_id), "unknown/completed call");
    auto &call = impl_->calls.at(*context.call_id);
    require(call.operation_id == *context.operation_id && call.parent_id == context.dispatch_id, "call ownership mismatch");
    require(stage != CallStage::Prepare && !call.stages.count(stage), "duplicate/invalid call stage");
    require(stage == CallStage::Invoke || call.stages.count(CallStage::Invoke), "call invocation missing");
    require(stage == CallStage::CompleteRequired || work_ids.empty(), "work requirements belong to COMPLETE_REQUIRED");
    std::set<uint64_t> required;
    for (const uint64_t id : work_ids) {
        require(impl_->work_operations.count(id) && impl_->work_operations.at(id) == *context.operation_id,
                "unknown/cross-operation required work");
        require(required.insert(id).second, "duplicate required work");
    }
    if (stage == CallStage::Publish) require(call.kind == CallKind::Stripe && call.own_work_ids.size() == 1 &&
        !call.stages.count(CallStage::CompleteRequired), "invalid stripe publication boundary");
    if (stage == CallStage::Fence) require(call.kind == CallKind::Fence && call.stages.count(CallStage::CompleteRequired),
                                         "fence lacks completion requirements");
    if (stage == CallStage::CompleteRequired) {
        if (call.kind == CallKind::Full || call.kind == CallKind::Stripe || call.kind == CallKind::ResidualCompact)
            require(call.own_work_ids.size() == 1, "completion requirement precedes selected work");
        require(std::includes(required.begin(), required.end(), call.own_work_ids.begin(), call.own_work_ids.end()),
                "completion requirements omit own selected work");
        if (call.kind == CallKind::Stripe) require(call.stages.count(CallStage::Publish), "stripe publication missing");
        call.required_work_ids = required;
    }
    if (stage == CallStage::Continuation) {
        if (call.kind == CallKind::Full || call.kind == CallKind::Stripe || call.kind == CallKind::ResidualCompact)
            require(call.own_work_ids.size() == 1 && call.stages.count(CallStage::CompleteRequired),
                    "continuation lacks selected work completion requirement");
        if (call.kind == CallKind::Fence) require(call.stages.count(CallStage::Fence), "continuation precedes fence");
    }
    impl_->emit_call(context, stage, work_ids);
    call.stages.insert(stage);
    if (stage == CallStage::Continuation) {
        --operation.active_calls;
        impl_->calls.erase(*context.call_id);
    }
}
}
#endif
