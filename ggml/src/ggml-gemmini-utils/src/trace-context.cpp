#include <gemmini/trace-context.hpp>
#include <gemmini/log.hpp>
#include "json.hpp"
#include <atomic>
#include <cstring>
#include <limits>

namespace {
thread_local gemmini_trace_context bound_context{};
std::atomic<uint64_t> next_identity{1};
}
extern "C" uint64_t gemmini_trace_reserve_ids(uint64_t count) noexcept {
#if LOG_CYCLE
    if (!count) return 0;
    auto current = next_identity.load(std::memory_order_relaxed);
    do {
        if (count > std::numeric_limits<uint64_t>::max() - current) return 0;
    } while (!next_identity.compare_exchange_weak(current, current + count,
                                                  std::memory_order_relaxed));
    return current;
#else
    (void)count; return 0;
#endif
}
extern "C" gemmini_trace_context gemmini_trace_capture(void) noexcept {
#if LOG_CYCLE
    if (bound_context.flags & GEMMINI_TRACE_BOUND) return bound_context;
    const auto current = ggml::gemmini::performance::capture_context();
    gemmini_trace_context result{};
    result.flags = GEMMINI_TRACE_CAPTURED;
    result.request_id = current.request_id;
    result.inference_operation_id = current.operation_id;
    result.phase = current.phase == ggml::gemmini::performance::Phase::decode ? 1 : 0;
    return result;
#else
    return {};
#endif
}
extern "C" gemmini_trace_context gemmini_trace_bind(gemmini_trace_context context) noexcept {
    const auto previous = bound_context;
#if LOG_CYCLE
    context.flags |= GEMMINI_TRACE_CAPTURED | GEMMINI_TRACE_BOUND;
    bound_context = context;
#else
    (void)context;
#endif
    return previous;
}
extern "C" void gemmini_trace_restore(gemmini_trace_context previous) noexcept {
    bound_context = previous;
}
extern "C" gemmini_trace_context gemmini_trace_fork(gemmini_trace_context parent) noexcept {
#if LOG_CYCLE
    parent.parent_task_id = parent.task_id;
    parent.parent_span_id = parent.span_id;
    parent.task_id = gemmini_trace_reserve_ids(1);
    parent.span_id = parent.task_id;
    parent.flags |= GEMMINI_TRACE_CAPTURED;
#endif
    return parent;
}
extern "C" gemmini_trace_context gemmini_trace_operator(gemmini_trace_context parent,
        uint64_t graph_id, uint64_t operator_id, uint64_t node_id,
        const char *name, uint64_t worker_id, int worker_present, int multitask) noexcept {
#if LOG_CYCLE
    auto result = gemmini_trace_fork(parent);
    result.graph_id = graph_id; result.operator_id = operator_id; result.node_id = node_id;
    result.worker_id = worker_id;
    result.flags |= GEMMINI_TRACE_OPERATOR;
    if (worker_present) result.flags |= GEMMINI_TRACE_WORKER;
    else result.flags &= ~GEMMINI_TRACE_WORKER;
    if (multitask) result.flags |= GEMMINI_TRACE_MULTITASK;
    else result.flags &= ~GEMMINI_TRACE_MULTITASK;
    const size_t length = name ? std::min(std::strlen(name), sizeof(result.operator_name)-1) : 0;
    std::memset(result.operator_name, 0, sizeof(result.operator_name));
    if (length) std::memcpy(result.operator_name, name, length);
    result.role = name && std::strcmp(name, "MUL_MAT") == 0 ? GEMMINI_TRACE_ROLE_DENSE : GEMMINI_TRACE_ROLE_OPERATOR;
    return result;
#else
    (void)graph_id; (void)operator_id; (void)node_id; (void)name;
    (void)worker_id; (void)worker_present; (void)multitask; return parent;
#endif
}
namespace ggml::gemmini::trace {
performance::Context inference_context(const gemmini_trace_context &context) noexcept {
    if (!(context.flags & GEMMINI_TRACE_CAPTURED)) return performance::capture_context();
    return {context.request_id, context.inference_operation_id,
            context.phase ? performance::Phase::decode : performance::Phase::prefill};
}
std::string append_metadata(std::string text, const gemmini_trace_context &context,
                            uint64_t segment_id, bool structural_envelope) {
#if LOG_CYCLE
    if (!(context.flags & GEMMINI_TRACE_OPERATOR) && !context.task_id) return text;
#if !CYCLE_DETAIL
    // High-frequency compact intervals already have a fixed op/kind prefix.
    // Avoid reparsing them with nlohmann::json just to append structural IDs.
    const bool fast_compact_interval = text.rfind("{\"op\":", 0) == 0 &&
        (text.find("\"kind\":\"cpu\"") != std::string::npos ||
         text.find("\"kind\":\"segment\"") != std::string::npos ||
         text.find("\"kind\":\"cycle\"") != std::string::npos);
    if (fast_compact_interval) {
        if (text.find("\"operator_context\":") != std::string::npos) return text;
        const bool residual = context.role == GEMMINI_TRACE_ROLE_RESIDUAL;
        std::string info = "{";
        bool first = true;
        auto key = [&](const char *name) {
            if (!first) info += ',';
            first = false;
            info += '\"'; info += name; info += "\":";
        };
        auto number = [&](const char *name, uint64_t value) {
            if (!value) return;
            key(name); info += std::to_string(value);
        };
        auto string = [&](const char *name, const char *value) {
            if (!value || !*value) return;
            key(name); info += nlohmann::json(value).dump();
        };
        number("operator_id", context.operator_id);
        number("graph_id", context.graph_id);
        string("operator_kind", context.operator_name);
        number("task_id", context.task_id);
        number("parent_task_id", context.parent_task_id);
        number("segment_id", structural_envelope ? context.span_id : segment_id);
        number("parent_segment_id", structural_envelope ? context.parent_span_id : context.span_id);
        string("role", residual ? "residual" :
            context.role == GEMMINI_TRACE_ROLE_DENSE ? "dense" : "operator");
        if (structural_envelope && (context.flags & GEMMINI_TRACE_MULTITASK))
            string("structural_reason", "structurally_cross_task");
        info += '}';
        const auto close = text.rfind('}');
        if (close != std::string::npos) text.insert(close, ",\"operator_context\":" + info);
        return text;
    }
#endif
    using Json = nlohmann::json;
    // Formatting occurs when bounded records drain, after endpoint capture.
    // Preserve the legacy row and schema; common carries device-neutral keys.
    auto row = Json::parse(text);
    if (!row.is_object() || row.contains("operator_context")) return text;
    const auto type = row.value("record_type", std::string());
    const auto compact_kind = row.value("kind", std::string());
    const bool compact_interval = type.empty() &&
        (compact_kind == "cpu" || compact_kind == "segment" || compact_kind == "cycle");
    if ((!compact_interval && type.empty()) || type == "INFERENCE_EVENT" ||
        type.find("CONFIGURATION") != std::string::npos ||
        type.find("SUMMARY") != std::string::npos) return text;
    const bool residual = context.role == GEMMINI_TRACE_ROLE_RESIDUAL;
    auto nullable = [](uint64_t n) -> Json { return n ? Json(n) : Json(); };
#if CYCLE_DETAIL
    const auto string_value = [&](const char *key, std::string fallback = {}) {
        const auto found = row.find(key);
        return found != row.end() && found->is_string() ? found->get<std::string>() : fallback;
    };
    const bool npu = type == "NPU_OPERATOR_SEGMENT" || type == "WS_LOOP_TELEMETRY" ||
        type == "IM2P_EXECUTION_TELEMETRY" || type == "IM2P_STRIPE_TELEMETRY" ||
        type == "IM2P_RMD_STRIPE_TELEMETRY" || type == "IM2P_RMD_EXECUTION_TELEMETRY" ||
        (type == "RESOURCE_SAMPLE" && row.value("kind", std::string()) == "npu");
    Json backend = row.contains("backend") ? row["backend"] : Json();
    Json domain = row.contains("clock_domain") ? row["clock_domain"] :
        row.contains("domain") ? row["domain"] : row.contains("source") ? row["source"] : Json();
    Json info = {{"version",1},{"request_id",nullable(context.request_id)},
        {"inference_operation_id",nullable(context.inference_operation_id)},
        {"phase",context.inference_operation_id ? Json(performance::phase_name(
            context.phase ? performance::Phase::decode : performance::Phase::prefill)) : Json()},
        {"operator_id",nullable(context.operator_id)},
        {"graph_id",nullable(context.graph_id)},{"node_id",context.flags & GEMMINI_TRACE_OPERATOR ? Json(context.node_id) : Json()},
        {"operator_kind",context.operator_name[0] ? Json(context.operator_name) : Json()},
        {"task_id",nullable(context.task_id)},{"parent_task_id",nullable(context.parent_task_id)},
        {"segment_id",nullable(structural_envelope ? context.span_id : segment_id)},
        {"parent_segment_id",nullable(structural_envelope ? context.parent_span_id : context.span_id)},
        {"worker_id",context.flags & GEMMINI_TRACE_WORKER ? Json(context.worker_id) : Json()},
        {"role",residual ? "residual" : context.role == GEMMINI_TRACE_ROLE_DENSE ? "dense" : "operator"},
        {"device",npu ? "npu" : "cpu"},{"backend",backend},{"clock_domain",domain},{"additive",false},
        {"scope",npu ? "device_counter" : structural_envelope ? "caller_thread_envelope" : "owner_segment"},
        {"structural_reason",structural_envelope && (context.flags & GEMMINI_TRACE_MULTITASK) ? Json("structurally_cross_task") : Json()}};
    Json counter = {{"metric",npu ? "work_total_cycles" : "cpu_cycles"},
        {"unit","cycle"},{"source",row.contains("source") ? row["source"] : Json()},
        {"domain",domain},{"value",nullptr},{"valid",false},
        {"reason","no_single_counter"},{"start",nullptr},{"end",nullptr}};
    if (npu && type != "WS_LOOP_TELEMETRY") {
        for (const char *key : {"cycles", "rtl_work_total_cycles", "rmd_work_total_cycles", "latency_cycles"}) {
            if (!row.contains(key)) continue;
            counter["value"] = row[key];
            counter["metric"] = std::string(key) == "latency_cycles" ? "latency_cycles" : "work_total_cycles";
            counter["valid"] = row.contains("valid") ? row["valid"] : Json(row[key].is_number());
            counter["reason"] = row.contains("reason") ? row["reason"] :
                row[key].is_number() ? Json() : Json("provider_counter_unavailable");
            break;
        }
        if (row.contains("publish_cycle")) counter["start"] = row["publish_cycle"];
        if (row.contains("completion_cycle")) counter["end"] = row["completion_cycle"];
    } else if (npu) {
        counter["reason"] = "unavailable_device_elapsed_counter";
    } else if (row.contains("native_cycles") && row["native_cycles"].is_object()) {
        const auto &native = row["native_cycles"];
        counter["value"] = native.value("delta",Json());
        counter["valid"] = native.value("valid",false);
        counter["reason"] = native.value("reason",Json("unavailable_sample"));
        if (native.contains("start") && native["start"].is_object())
            counter["start"] = native["start"].value("value",Json());
        if (native.contains("end") && native["end"].is_object())
            counter["end"] = native["end"].value("value",Json());
    } else if (type == "RESOURCE_SAMPLE" && string_value("kind") == "cpu") {
        counter["value"] = row.value("cycles",Json());
        counter["valid"] = row.contains("cycles") && row["cycles"].is_number();
        counter["reason"] = row.value("cycles_reason",Json());
    } else if (row.contains("delta") && string_value("source") == "linux_perf_cpu_cycles") {
        counter["value"] = row["delta"];
        counter["valid"] = row.value("valid",false);
        counter["reason"] = row.value("reason",Json());
        counter["start"] = row.value("start",Json());
        counter["end"] = row.value("end",Json());
    }
    info["counter"] = std::move(counter);
    info["wall_time"] = row.contains("host_timing") ? row["host_timing"] : Json();
#else
    // Normal cycle mode keeps only structural attribution that cannot be
    // recovered from the interval itself or inference_context.
    Json info = {{"operator_id",nullable(context.operator_id)},
        {"graph_id",nullable(context.graph_id)},
        {"operator_kind",context.operator_name[0] ? Json(context.operator_name) : Json()},
        {"task_id",nullable(context.task_id)},
        {"parent_task_id",nullable(context.parent_task_id)},
        {"segment_id",nullable(structural_envelope ? context.span_id : segment_id)},
        {"parent_segment_id",nullable(structural_envelope ? context.parent_span_id : context.span_id)},
        {"role",residual ? "residual" : context.role == GEMMINI_TRACE_ROLE_DENSE ? "dense" : "operator"},
        {"scope",structural_envelope ? "caller_thread_envelope" : "owner_segment"}};
    if (structural_envelope && (context.flags & GEMMINI_TRACE_MULTITASK))
        info["structural_reason"] = "structurally_cross_task";
#endif
    // Append instead of reserializing the legacy object so existing exact
    // schema checks and field ordering remain compatible.
    const auto close = text.rfind('}');
    if (close != std::string::npos) text.insert(close, ",\"operator_context\":" + info.dump());
#endif
    return text;
}
std::string annotate_origin(std::string json, const gemmini_trace_context &context) {
    if (!(context.flags & GEMMINI_TRACE_CAPTURED)) return json;
    json = append_metadata(std::move(json), context, gemmini_trace_reserve_ids(1));
    if (json.find("\"inference_context\":") == std::string::npos) {
        const auto origin = performance::serialize_context(inference_context(context));
        const auto close = json.rfind('}');
        if (close != std::string::npos)
            json.insert(close, ",\"inference_context\":" + (origin.empty() ? std::string("null") : origin));
    }
    return json;
}
void CpuStage::finish(bool success) noexcept {
    if (finished_) return;
    finished_ = true;
#if LOG_CYCLE
    const auto end = gemmini_cpu_timing_read();
    gemmini_cycle_record_v2 identity{};
    identity.interval.layer = layer_; identity.interval.op = stage_;
    // Native ownership and timing validity are independent of operation outcome.
    try {
        log::cycle.write_cpu(identity, start_, end, success, true,
            scope_ == Scope::envelope);
    } catch (...) { log::cycle.report_failure("operator stage"); }
#else
    (void)success;
#endif
}
}
