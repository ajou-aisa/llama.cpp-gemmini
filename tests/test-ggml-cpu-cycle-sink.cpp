#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "../common/json.hpp"
#include <gemmini/layer.h>
#include <gemmini/log.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

static std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) return 1;
    std::array<char, 64> layer{};
    gemmini_get_layer("Qcur-7", layer.data(), layer.size());
    if (std::string(layer.data()) != "blk.7.qcur") return 11;
    gemmini_get_layer("Qcur-7 (reshaped)", layer.data(), layer.size());
    if (std::string(layer.data()) != "blk.7.qcur") return 14;
    gemmini_get_layer("cache_k_l7 (view) (copy of Kcur-7 (reshaped))",
                      layer.data(), layer.size());
    if (std::string(layer.data()) != "blk.7.cache_k_l7") return 15;
    gemmini_get_layer("blk.3.attn.q_proj", layer.data(), layer.size());
    if (std::string(layer.data()) != "blk.3.attn.q_proj") return 12;
    gemmini_get_layer("inp_embd", layer.data(), layer.size());
    if (std::string(layer.data()) != "inp_embd") return 13;

    const bool preserve = argc == 3 && std::string(argv[2]) == "--preserve";
    const std::filesystem::path root = std::filesystem::absolute(argv[1]);
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directories(root / "work", error);
    if (error) return 2;
    std::filesystem::current_path(root / "work", error);
    if (error) return 3;
    const auto selected = root / "selected-cycle.jsonl";
    if (!gemmini_log_cycle_set_output_path(selected.c_str())) return 4;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return 5;
    ggml_backend_cpu_set_n_threads(backend, 1);
    ggml_init_params params{ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false), nullptr, true};
    ggml_context * context = ggml_init(params);
    if (!context) return 6;
    ggml_tensor * lhs = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
    ggml_tensor * rhs = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
    ggml_tensor * sum = ggml_add(context, lhs, rhs);
    ggml_tensor * result = ggml_add(context, sum, rhs);
    ggml_set_name(sum, "attn_norm-7");
    ggml_set_name(result, "ffn_norm-7");
    ggml_cgraph * graph = ggml_new_graph_custom(context, 8, false);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!buffer) return 7;
    const std::array<float, 4> left{1, 2, 3, 4};
    const std::array<float, 4> right{5, 6, 7, 8};
    ggml_backend_tensor_set(lhs, left.data(), 0, sizeof(left));
    ggml_backend_tensor_set(rhs, right.data(), 0, sizeof(right));
    const std::array<int, 6> requested_workers{2, 2, 2, 1, 2, 2};
    constexpr size_t aborted_run = 4;
    ggml_threadpool_t threadpool = nullptr;
    for (size_t run = 0; run < requested_workers.size(); ++run) {
        if (run == 1) {
            auto pool_params = ggml_threadpool_params_default(2);
            threadpool = ggml_threadpool_new(&pool_params);
            if (!threadpool) return 16;
            ggml_backend_cpu_set_threadpool(backend, threadpool);
        }
        const bool abort = run == aborted_run;
        const auto abort_callback = +[](void *) { return true; };
        ggml_backend_cpu_set_abort_callback(backend, abort ? abort_callback : nullptr, nullptr);
        ggml_backend_cpu_set_n_threads(backend, requested_workers[run]);
        const std::array<float, 4> sentinel{-1, -1, -1, -1};
        ggml_backend_tensor_set(result, sentinel.data(), 0, sizeof(sentinel));
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != (abort ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS)) return 8;
        std::array<float, 4> actual{};
        ggml_backend_tensor_get(result, actual.data(), 0, sizeof(actual));
        const std::array<float, 4> expected{11, 14, 17, 20};
        if (actual != (abort ? sentinel : expected)) return 17;
    }
    gemmini_log_cycle_set_output(stderr);
    ggml_backend_cpu_set_threadpool(backend, nullptr);
    ggml_threadpool_free(threadpool);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);

    const std::string output = read_file(selected);
    const auto default_path = root / "work/output/log/cycle-log.jsonl";
#if defined(EXPECT_CPU_CYCLE_LOG) && (!EXPECT_CPU_CYCLE_LOG || !EXPECT_LOG_CYCLE)
    if (!output.empty() || std::filesystem::exists(selected) != bool(EXPECT_LOG_CYCLE) ||
        std::filesystem::exists(default_path) ||
        std::filesystem::exists(root / "work/output/log/npu-cycle-trace.jsonl")) return 9;
#else
#if CYCLE_DETAIL
    if (output.find("\"version\":2") == std::string::npos ||
        output.find("\"op\":\"cpu.add\"") == std::string::npos ||
        output.find("\"layer\":\"blk.7.attn_norm\"") == std::string::npos ||
        output.find("\"run_id\":null") != std::string::npos ||
        output.find("\"stripe_id\":null") == std::string::npos ||
        output.find("\"node_id\":0") == std::string::npos ||
        output.find("\"worker_id\":0") == std::string::npos ||
        output.find("\"stripe_id\":0") != std::string::npos ||
        output.find("\"valid\":true") == std::string::npos ||
        output.find("scalar_provenance_unavailable") != std::string::npos ||
        std::filesystem::exists(default_path)) return 9;

    std::vector<nlohmann::json> summaries;
    std::vector<nlohmann::json> operations;
    std::map<uint64_t, std::set<uint64_t>> workers;
    std::map<uint64_t, std::map<uint64_t, nlohmann::json>> worker_intervals;
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> operation_ids;
    std::istringstream records(output);
    for (std::string line; std::getline(records, line);) {
        const auto record = nlohmann::json::parse(line);
        if (record.at("record_type") == "CPU_WORK_SUMMARY") summaries.push_back(record);
        if (record.at("op") == "cpu.add") {
            const uint64_t run_id = record.at("run_id");
            const uint64_t worker_id = record.at("worker_id");
            const uint64_t node_id = record.at("node_id");
            if (!operation_ids.emplace(run_id, node_id, worker_id).second || node_id > 1 ||
                record.at("layer") != (node_id == 0 ? "blk.7.attn_norm" : "blk.7.ffn_norm")) return 22;
            workers[run_id].insert(worker_id);
            operations.push_back(record);
        }
        if (record.at("op") == "cpu.graph_worker") {
            if (!record.at("node_id").is_null() || record.at("layer") != "cpu.graph" ||
                !worker_intervals[record.at("run_id").get<uint64_t>()].emplace(
                    record.at("worker_id").get<uint64_t>(), record).second) return 23;
        }
        if (record.at("op") == "cpu.add" || record.at("op") == "cpu.graph_worker") {
            const auto & wall = record.at("host_timing");
            const auto & cpu = record.at("thread_cpu_timing");
            const auto & native = record.at("native_cycles");
            if (record.at("record_type") != "CPU_INTERVAL" || record.at("additive") != false ||
                wall.at("valid") != true || wall.at("start_tid") != wall.at("end_tid") ||
                wall.at("start_ns").get<uint64_t>() > wall.at("end_ns").get<uint64_t>() ||
                wall.at("duration_ns").get<uint64_t>() !=
                    wall.at("end_ns").get<uint64_t>() - wall.at("start_ns").get<uint64_t>() ||
                native.at("delta").is_null() == native.at("valid").get<bool>()) return 24;
#if defined(__APPLE__) || defined(__linux__)
            if (cpu.at("valid") != true || cpu.at("duration_ns").get<uint64_t>() !=
                    cpu.at("end_ns").get<uint64_t>() - cpu.at("start_ns").get<uint64_t>()) return 25;
#endif
            if (native.at("valid").get<bool>()) {
                if (record.at("source") != "linux_perf_cpu_cycles" ||
                    native.at("start").at("valid") != true || native.at("end").at("valid") != true ||
                    native.at("start").at("owner_token") != native.at("end").at("owner_token") ||
                    native.at("start").at("generation") != native.at("end").at("generation") ||
                    native.at("delta").get<uint64_t>() != native.at("end").at("value").get<uint64_t>() -
                        native.at("start").at("value").get<uint64_t>()) return 26;
            } else if (native.at("reason").is_null()) return 27;
#if defined(__APPLE__)
            if (!native.at("start").at("value").is_null() ||
                !native.at("end").at("value").is_null() ||
                native.at("reason") != "not_thread_cpu_counter") return 28;
#endif
        }
    }
    if (summaries.size() != requested_workers.size() || workers.size() != summaries.size() ||
        worker_intervals.size() != summaries.size()) return 18;
    for (const auto & operation : operations) {
        const auto & parent = worker_intervals.at(operation.at("run_id").get<uint64_t>()).at(
            operation.at("worker_id").get<uint64_t>()).at("host_timing");
        const auto & child = operation.at("host_timing");
        if (parent.at("execution_id") != child.at("execution_id") ||
            parent.at("start_tid") != child.at("start_tid") ||
            parent.at("start_ns").get<uint64_t>() > child.at("start_ns").get<uint64_t>() ||
            parent.at("end_ns").get<uint64_t>() < child.at("end_ns").get<uint64_t>()) return 29;
    }
    std::set<uint64_t> run_ids;
    for (size_t run = 0; run < summaries.size(); ++run) {
        const auto & summary = summaries[run];
        const uint64_t run_id = summary.at("run_id");
        const auto & actual_workers = workers.at(run_id);
        const auto & totals = summary.at("cpu_workers");
        const auto & intervals = worker_intervals.at(run_id);
        size_t operation_count = 0;
        for (const auto & operation : operations) operation_count += operation.at("run_id") == run_id;
        if (intervals.size() != actual_workers.size() ||
            operation_count != actual_workers.size() * (run == aborted_run ? 1 : 2)) return 30;
        uint64_t cycles = 0, cycle_count = 0, cpu_ns = 0, cpu_count = 0;
        for (const auto & interval : intervals) {
            const auto & native = interval.second.at("native_cycles");
            const auto & cpu = interval.second.at("thread_cpu_timing");
            if (native.at("valid").get<bool>()) { cycles += native.at("delta").get<uint64_t>(); ++cycle_count; }
            if (cpu.at("valid").get<bool>()) { cpu_ns += cpu.at("duration_ns").get<uint64_t>(); ++cpu_count; }
        }
        if (totals.at("cycles_valid_count") != cycle_count || totals.at("thread_cpu_valid_count") != cpu_count ||
            (!totals.at("cycles").is_null() && totals.at("cycles") != cycles) ||
            (!totals.at("thread_cpu_ns").is_null() && totals.at("thread_cpu_ns") != cpu_ns)) return 31;
        if (!run_ids.insert(run_id).second || actual_workers.empty() ||
            actual_workers.count(0) != 1 || actual_workers.size() > size_t(requested_workers[run]) ||
            summary.at("op") != "cpu.graph_workers" || summary.at("layer") != "cpu.graph" ||
            summary.at("operation_success") != (run != aborted_run) || summary.at("additive") != false ||
            summary.at("host_timing").at("valid") != true ||
            totals.at("interval_count") != actual_workers.size() ||
            totals.at("cycles_valid_count").get<uint64_t>() > actual_workers.size() ||
            totals.at("thread_cpu_valid_count").get<uint64_t>() > actual_workers.size() ||
            totals.at("cycles").is_null() != (totals.at("cycles_valid_count") != actual_workers.size()) ||
            totals.at("thread_cpu_ns").is_null() != (totals.at("thread_cpu_valid_count") != actual_workers.size()))
            return 19;
#if defined(__APPLE__) || defined(__linux__)
        if (totals.at("thread_cpu_valid_count") != actual_workers.size()) return 20;
#endif
#if defined(__APPLE__)
        if (!totals.at("cycles").is_null() || totals.at("cycles_reason") != "not_thread_cpu_counter") return 21;
#endif
    }
#else
    if (output.find("\"kind\":\"cpu\"") == std::string::npos ||
        output.find("\"op\":\"cpu.add\"") == std::string::npos ||
        output.find("\"layer\":\"blk.7.attn_norm\"") == std::string::npos ||
        output.find("\"node_id\":0") == std::string::npos ||
        output.find("\"worker_id\":0") == std::string::npos ||
        output.find("\"duration_role\":\"OBSERVATION_ONLY\"") == std::string::npos ||
        output.find("\"exclusion_reason\":\"outside_collection\"") == std::string::npos ||
        std::filesystem::exists(default_path)) return 9;
#endif
#endif
    if (!preserve) std::filesystem::remove_all(root, error);
    return error ? 10 : 0;
}
