#include <gemmini/trace-context.hpp>
#include <gemmini/host-timing.hpp>
#include <gemmini/log.hpp>
#include "../common/json.hpp"
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace ggml::gemmini;
using Json = nlohmann::json;
namespace {
bool check(bool condition, const char *message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}
uint64_t now() { return cycle::read_host_sample().ns; }
std::vector<Json> read(const std::filesystem::path &path) {
    std::ifstream input(path); std::vector<Json> rows; std::string line;
    while (std::getline(input,line)) if (!line.empty()) rows.push_back(Json::parse(line));
    return rows;
}
gemmini_cycle_record_v2 identity(const char *name) {
    gemmini_cycle_record_v2 result{};
    result.interval.layer = "trace.test"; result.interval.op = name;
    result.identity_mask = GEMMINI_CYCLE_HAS_NODE_ID | GEMMINI_CYCLE_HAS_WORKER_ID;
    result.node_id = 0; result.worker_id = 0;
    return result;
}
bool test_context_and_delayed_records(const std::filesystem::path &path) {
    log::CycleLog logger;
    if (!logger.set_output_path(path.c_str(), true)) return false;
    logger.set_buffered(true);
    performance::reset();
    performance::start_request(now());
    performance::begin_operation(performance::Phase::prefill, now());
    const auto base = gemmini_trace_reserve_ids(2);
    const auto origin = gemmini_trace_operator(gemmini_trace_capture(), base, base+1,
                                               0, "MUL_MAT", 0, 1, 1);
    const auto empty_sample = gemmini_cpu_sample{};
    (void)empty_sample;
    gemmini_cpu_sample worker_start{}, worker_end{}, empty_start{}, empty_end{};
    cycle::WorkerCpuTiming callback;
    uint64_t child_id = 0, grandchild_id = 0;
    bool restored = false;
    {
        trace::ScopedContext root(origin);
        callback.origin = gemmini_trace_capture();
        const auto parent = gemmini_trace_capture();
        {
            trace::ScopedContext child(parent, true);
            child_id = gemmini_trace_capture().task_id;
            {
                trace::ScopedContext grandchild(gemmini_trace_capture(), true);
                grandchild_id = gemmini_trace_capture().task_id;
                if (!check(gemmini_trace_capture().parent_task_id == child_id,
                           "nested task stores its actual parent")) return false;
            }
            if (!check(gemmini_trace_capture().task_id == child_id,
                       "nested task restores interrupted same-thread task")) return false;
        }
        if (!check(gemmini_trace_capture().task_id == parent.task_id && (child_id != grandchild_id || !EXPECT_LOG_CYCLE),
                   "task reuse does not overwrite parent context")) return false;
        std::thread worker([&] {
            const auto before = gemmini_trace_capture();
            cycle::WorkerCpuTiming::observe(&callback, true);
            worker_start = gemmini_cpu_timing_read();
            worker_end = gemmini_cpu_timing_read();
            cycle::WorkerCpuTiming::observe(&callback, false);
            restored = gemmini_trace_capture().task_id == before.task_id;
        });
        worker.join();
    }
    // Explicitly unassociated samples must stay unassociated even while another
    // request becomes active. This is distinct from legacy synthetic fixtures.
    {
        gemmini_trace_context empty{}; empty.flags = GEMMINI_TRACE_CAPTURED;
        trace::ScopedContext no_request(empty);
        empty_start = gemmini_cpu_timing_read(); empty_end = gemmini_cpu_timing_read();
    }
    performance::end_operation(now(), true);
    performance::finish_request(now());
    performance::start_request(now());
    performance::begin_operation(performance::Phase::decode, now());
    const auto current = performance::capture_context();
    auto id = identity("delayed.worker");
    logger.write_cpu(id, worker_start, worker_end);
    id.interval.op = "delayed.raw.segment";
    logger.write_cpu(id, callback.start, callback.end, false, true);
    id.interval.op = "explicit.no.request";
    logger.write_cpu(id, empty_start, empty_end);
    if (!logger.flush()) return false;
    const auto rows = read(path);
#if EXPECT_LOG_CYCLE
    if (!check(rows.size() == 3 && restored, "delayed records are retained and callback restores TLS")) return false;
    const auto &legacy = rows[0]; const auto &raw = rows[1]; const auto &empty = rows[2];
    if (!check(legacy["inference_context"]["request_id"] == origin.request_id &&
               legacy["inference_context"]["operation_id"] == origin.inference_operation_id &&
               origin.request_id != current.request_id,
               "late worker retains origin instead of latest global request")) return false;
#if EXPECT_CYCLE_DETAIL
    if (!check(raw["record_type"] == "OPERATOR_SEGMENT" && !raw.contains("cpu_interval_sequence") &&
               raw["operator_context"]["request_id"] == origin.request_id &&
               raw["operator_context"]["inference_operation_id"] == origin.inference_operation_id &&
               raw["operation_success"] == false &&
               raw["thread_cpu_timing"]["valid"] == true,
               "detail raw canceled segment retains owner timing independently of outcome")) return false;
#else
    if (!check(raw["kind"] == "segment" && !raw.contains("cpu_interval_sequence") &&
               raw["inference_context"]["request_id"] == origin.request_id &&
               raw["inference_context"]["operation_id"] == origin.inference_operation_id &&
               raw["operation_success"] == false && !raw.contains("thread_cpu_timing") &&
               !raw.contains("host_timing") && !raw.contains("record_type") &&
               raw["ns_start"].get<uint64_t>() <= raw["ns_end"].get<uint64_t>(),
               "compact raw segment retains origin, shared timeline, and outcome without nested timing metadata")) return false;
#endif
    if (!check(!empty.contains("inference_context") && !empty.contains("operator_context"),
               "captured-empty origin never inherits unrelated request")) return false;
    const auto &meta = legacy["operator_context"];
#if EXPECT_CYCLE_DETAIL
    if (!check(meta["operator_id"] == origin.operator_id && meta["operator_kind"] == "MUL_MAT" &&
               meta["node_id"] == 0 && meta["worker_id"] == 0 && meta["role"] == "dense" &&
               meta["device"] == "cpu" && meta["parent_task_id"] == origin.task_id &&
               meta["task_id"] != origin.task_id &&
               legacy["host_timing"]["start_tid"] == legacy["host_timing"]["end_tid"] &&
               worker_start.tid != cycle::host_thread_id(),
               "detail metadata preserves duplicated device/timing attribution")) return false;
#else
    if (!check(meta["operator_id"] == origin.operator_id && meta["operator_kind"] == "MUL_MAT" &&
               meta["role"] == "dense" && meta["parent_task_id"] == origin.task_id &&
               meta["task_id"] != origin.task_id && legacy["node_id"] == 0 && legacy["worker_id"] == 0 &&
               !meta.contains("node_id") && !meta.contains("worker_id") && !meta.contains("device") &&
               worker_start.tid != cycle::host_thread_id(),
               "compact metadata keeps task lineage while top-level identity is not duplicated")) return false;
#endif
#else
    if (!check(rows.empty() && base == 0 && origin.flags == 0,
               "disabled logging allocates no identity and emits no records")) return false;
#endif
    performance::end_operation(now(), true); performance::finish_request(now());
    return true;
}
bool test_counter_task_separation() {
#if EXPECT_LOG_CYCLE
    gemmini_cpu_sample a{}, b{};
    a.ns = b.ns = 10; a.tid = b.tid = 7;
    a.thread_cpu_valid = b.thread_cpu_valid = 1;
    a.thread_cpu_ns = b.thread_cpu_ns = 20;
    a.trace.task_id = 11; b.trace.task_id = 12;
    auto json = Json::parse(cycle::serialize_cpu_native(a,b));
    if (!check(json["delta"].is_null() && json["reason"] == "structurally_cross_task",
               "same TID cannot authorize cross-task subtraction")) return false;
    a.trace.task_id = b.trace.task_id;
    gemmini_cpu_totals zero{};
    gemmini_cpu_timing_add(&zero, &a, &b);
    if (!check(zero.thread_cpu_valid_count == 1 && zero.thread_cpu_ns == 0,
               "valid zero thread time is preserved independently of cycle detail output")) return false;
    b.tid = 8;
    json = Json::parse(cycle::serialize_cpu_native(a,b));
#if defined(__linux__) && defined(__aarch64__)
    if (!check(json["reason"] == "thread_mismatch", "cross-thread native delta is invalid")) return false;
#endif
#endif
    return true;
}
bool test_uniform_device_metadata() {
#if EXPECT_LOG_CYCLE
    auto context = gemmini_trace_operator(gemmini_trace_capture(), 51, 52, 0, "MUL_MAT", 0, 1, 1);
    context.role = GEMMINI_TRACE_ROLE_RESIDUAL;
    const auto cpu = Json::parse(trace::append_metadata(
        R"({"schema":"gemmini.cycle","version":2,"record_type":"CPU_INTERVAL","op":"cpu.matmul.int","source":"linux_perf_cpu_cycles"})",context,61));
    const auto npu = Json::parse(trace::append_metadata(
        R"({"schema":"gemmini.cycle","version":2,"record_type":"NPU_OPERATOR_SEGMENT","op":"rmd.matmul.execute","backend":"im2p_sim","clock_domain":"independent_rmd_simulator","cycles":0,"valid":true})",context,62));
    const auto &a = cpu["operator_context"]; const auto &b = npu["operator_context"];
#if EXPECT_CYCLE_DETAIL
    if (!check(a["operator_kind"] == b["operator_kind"] &&
               a["role"] == b["role"] && a["device"] == "cpu" && b["device"] == "npu" &&
               b["backend"] == "im2p_sim" && npu["cycles"] == 0 && npu["valid"] == true,
               "detail metadata carries uniform device attribution")) return false;
#else
    if (!check(a["operator_kind"] == b["operator_kind"] &&
               a["role"] == b["role"] && !a.contains("device") && !b.contains("device") &&
               npu["backend"] == "im2p_sim" && npu["cycles"] == 0 && npu["valid"] == true,
               "compact metadata avoids duplicating device/backend fields")) return false;
#endif
    auto envelope = Json::parse(trace::append_metadata(
        R"({"record_type":"OPERATOR_SEGMENT","op":"operator.host_dispatch"})",context,63,true));
    if (!check(envelope["operator_context"]["structural_reason"] == "structurally_cross_task" &&
               envelope["operator_context"]["scope"] == "caller_thread_envelope" &&
               (!a.contains("structural_reason") || a["structural_reason"].is_null()),
               "parent cross-task structure does not invalidate measurable children")) return false;
#endif
    return true;
}
bool test_thread_reuse_buffered_logging(const std::filesystem::path &path) {
    log::CycleLog logger;
    if (!logger.set_output_path(path.c_str(),true)) return false;
    logger.set_buffered(true);
    auto root = gemmini_trace_operator(gemmini_trace_capture(), 71, 72, 0, "ADD", 0, 1, 1);
    constexpr size_t threads = 4, per_thread = 160;
    std::vector<std::thread> workers;
    for (size_t thread = 0; thread < threads; ++thread) workers.emplace_back([&,thread] {
        for (size_t i=0; i<per_thread; ++i) {
            auto origin = root; origin.worker_id = thread;
            trace::ScopedContext task(origin,true);
            const auto start = gemmini_cpu_timing_read(); const auto end = gemmini_cpu_timing_read();
            auto id = identity("task.host_work");
            logger.write_cpu(id,start,end,true,true);
        }
    });
    for (auto &worker : workers) worker.join();
    if (!logger.flush() || !logger.healthy()) return false;
    const auto rows = read(path);
#if EXPECT_LOG_CYCLE
    std::set<uint64_t> task_ids, segment_ids;
    for (const auto &row : rows) {
        const auto &c = row.at("operator_context");
        task_ids.insert(c.at("task_id").get<uint64_t>());
        segment_ids.insert(c.at("segment_id").get<uint64_t>());
#if EXPECT_CYCLE_DETAIL
        if (!check(c["operator_id"] == 72 && c["parent_task_id"] == root.task_id &&
                   row["host_timing"]["start_tid"] == row["host_timing"]["end_tid"],
                   "detail reused OS worker preserves distinct task identity")) return false;
#else
        if (!check(c["operator_id"] == 72 && c["parent_task_id"] == root.task_id &&
                   row["kind"] == "segment" && !row.contains("host_timing") &&
                   row["ns_start"].get<uint64_t>() <= row["ns_end"].get<uint64_t>(),
                   "compact reused worker preserves task identity on the shared timeline")) return false;
#endif
    }
    if (!check(rows.size() == threads*per_thread && task_ids.size() == rows.size() &&
               segment_ids.size() == rows.size(),
               "buffered concurrent logging retains every unique raw segment")) return false;
#else
    if (!check(rows.empty(), "OFF concurrent recording remains empty")) return false;
#endif
    return true;
}
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    try {
        const std::filesystem::path root = argv[1];
        std::filesystem::create_directories(root);
        log::cycle.set_output(nullptr);
        const bool ok = test_context_and_delayed_records(root/"delayed.jsonl") &&
            test_counter_task_separation() && test_uniform_device_metadata() &&
            test_thread_reuse_buffered_logging(root/"workers.jsonl");
        if (ok) std::puts("TRACE_CONTEXT_PASS delayed ownership, task reuse, cross-task rejection, device naming, buffer integrity");
        return ok ? 0 : 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr,"FAIL: trace context exception: %s\n",error.what()); return 1;
    }
}
