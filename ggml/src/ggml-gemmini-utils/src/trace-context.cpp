#include <gemmini/trace-context.hpp>
#include <gemmini/log.hpp>
#include "json.hpp"
#include <atomic>
#include <cstring>
#include <limits>
#include <algorithm>
#include <cctype>

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
                            uint64_t segment_id) {
#if LOG_CYCLE
    if (!(context.flags & GEMMINI_TRACE_OPERATOR) && !context.task_id) return text;
    using Json = nlohmann::json;
    // Formatting occurs when bounded records drain, after endpoint capture.
    // Preserve the legacy row and schema; common carries device-neutral keys.
    auto row = Json::parse(text);
    if (!row.is_object() || row.contains("operator_context")) return text;
    const auto type = row.value("record_type", std::string());
    if (type.empty() || type == "INFERENCE_EVENT" || type.find("CONFIGURATION") != std::string::npos ||
        type.find("SUMMARY") != std::string::npos) return text;
    const auto string_value = [&](const char *key, std::string fallback = {}) {
        const auto found = row.find(key);
        return found != row.end() && found->is_string() ? found->get<std::string>() : fallback;
    };
    const auto op = string_value("op", string_value("metric"));
    std::string lower = op;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    const bool npu = type == "NPU_OPERATOR_SEGMENT" || type == "WS_LOOP_TELEMETRY" || type == "IM2P_EXECUTION_TELEMETRY" ||
        type == "IM2P_STRIPE_TELEMETRY" || type == "IM2P_RMD_STRIPE_TELEMETRY" ||
        type == "IM2P_RMD_EXECUTION_TELEMETRY" ||
        (type == "RESOURCE_SAMPLE" && row.value("kind", std::string()) == "npu");
    const bool envelope = op == "operator.host_dispatch" || op == "task.host_work" ||
        op == "gemmini.matmul_worker" || op == "im2p.simulation_worker";
    const bool residual = context.role == GEMMINI_TRACE_ROLE_RESIDUAL ||
        lower.find("rmd") != std::string::npos || lower.find("residual") != std::string::npos ||
        lower.find("correction") != std::string::npos;
    std::string kind = "compute", stage = lower;
    if (envelope) { kind = "envelope"; stage = "dispatch"; }
    else if (lower.find("wait") != std::string::npos || lower.find("fence") != std::string::npos ||
             lower.find("join") != std::string::npos || lower.find("barrier") != std::string::npos) {
        kind = "wait";
    } else if (lower.find("copy") != std::string::npos || lower.find("transfer") != std::string::npos ||
               lower.find("send") != std::string::npos || lower.find("receive") != std::string::npos) {
        kind = "transfer";
    } else if (lower.find("submit") != std::string::npos || lower.find("handoff") != std::string::npos ||
               lower.find("queue") != std::string::npos) kind = "synchronization";
    if (npu) stage = "execute";
    else if (!envelope && lower.find("host_call") != std::string::npos)
        stage = "device_call";
    else if (!envelope && (lower.find("matmul") != std::string::npos ||
             lower.find("mul_mat") != std::string::npos || lower.find("ggml_compute_forward") != std::string::npos))
        stage = "execute";
    else {
        for (const char *prefix : {"cpu.", "gemmini.", "im2p.", "rmd.", "frontend."}) {
            const auto n = std::strlen(prefix);
            if (stage.compare(0, n, prefix) == 0) stage.erase(0,n);
        }
    }
    auto nullable = [](uint64_t n) -> Json { return n ? Json(n) : Json(); };
    const std::string backend = npu ? string_value("backend", type == "WS_LOOP_TELEMETRY" ? "gemmini" : "im2p_sim") : "cpu";
    Json domain = row.contains("clock_domain") ? row["clock_domain"] :
        row.contains("domain") ? row["domain"] : row.contains("source") ? row["source"] : Json();
    Json info = {{"version",1},{"request_id",nullable(context.request_id)},
        {"inference_operation_id",nullable(context.inference_operation_id)},
        {"phase",context.inference_operation_id ? Json(context.phase ? "decode" : "prefill") : Json()},
        {"operator_id",nullable(context.operator_id)},
        {"graph_id",nullable(context.graph_id)},{"node_id",context.flags & GEMMINI_TRACE_OPERATOR ? Json(context.node_id) : Json()},
        {"operator_kind",context.operator_name[0] ? Json(context.operator_name) : Json()},
        {"task_id",nullable(context.task_id)},{"parent_task_id",nullable(context.parent_task_id)},
        {"segment_id",nullable(envelope ? context.span_id : segment_id)},
        {"parent_segment_id",nullable(envelope ? context.parent_span_id : context.span_id)},
        {"worker_id",context.flags & GEMMINI_TRACE_WORKER ? Json(context.worker_id) : Json()},
        {"role",residual ? "residual" : context.role == GEMMINI_TRACE_ROLE_DENSE ? "dense" : "operator"},
        {"stage",stage},{"stage_kind",kind},{"device",npu ? "npu" : "cpu"},
        {"backend",backend},{"clock_domain",domain},{"additive",false},
        {"scope",npu ? "device_counter" : envelope ? "caller_thread_envelope" :
            op == "openmp.task_wait" ? "runtime_wait_envelope" : "owner_segment"},
        {"structural_reason",envelope && (context.flags & GEMMINI_TRACE_MULTITASK) ? Json("structurally_cross_task") : Json()}};
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
    // This is a projection of one observed counter, not an aggregate. In
    // particular CPU wrapper cost is never projected as NPU elapsed cycles.
    info["counter"] = std::move(counter);
    info["wall_time"] = row.contains("host_timing") ? row["host_timing"] : Json();
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
        log::cycle.write_cpu(identity, start_, end, success, true);
    } catch (...) { log::cycle.report_failure("operator stage"); }
#else
    (void)success;
#endif
}
}
