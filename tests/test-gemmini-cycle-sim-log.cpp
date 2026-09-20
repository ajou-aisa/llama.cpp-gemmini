#include <gemmini/cycle_sim_log.hpp>
#include <gemmini/cycle_sim_context.h>
#include <gemmini/cycle_reader.hpp>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <thread>
#include <unistd.h>

using namespace ggml::gemmini::cycle_sim;

static void rejects(const std::function<void()> &action) {
    bool rejected = false;
    try { action(); } catch (const std::exception &) { rejected = true; }
    assert(rejected);
}

int main(int argc, char **argv) {
    char directory[] = "/tmp/im2p-npu-trace-writer-XXXXXX";
    if (!mkdtemp(directory) || setenv("GEMMINI_LOG_DIR", directory, 1)) return 1;
    const auto info = compiled_run_info("writer-fixture", 32, 1);
    const auto path = std::filesystem::path(directory) / "npu-cycle-trace.jsonl";
    { std::ofstream existing(path); existing << "preserve"; }
    rejects([&] { Session::start(info); });
    assert(std::filesystem::file_size(path) == 8);
    std::filesystem::rename(path, std::filesystem::path(directory) / "preserved-existing.jsonl");
    auto session = Session::start(info);
    rejects([&] { Session::start(info); });
    int node = 0;
    session->set_policy_query({&node, [](void *expected, const void *candidate) { return expected == candidate; }});
    assert(session->target_eligible(&node));
    const auto phase = session->phase("prefill", {}, 32);
    Operation operation{"layer.0", "MUL_MAT", "GEMMINI", "F32", "Q8_HP1", 1, 3, info.dim, true};
    operation.semantic_context = std::make_shared<ggml::gemmini::semantic::Context>(
        ggml::gemmini::semantic::Context{{"prefill", {}, 0, 0}, ggml::gemmini::semantic::Source::PotalCollection, info.run_config_id});
    const auto context = session->register_operation(&node, operation, phase);
    const auto dispatch = session->new_dispatch(context);
    const auto call = session->call_begin(dispatch, CallKind::Full);
    if (argc == 2 && std::string(argv[1]) == "--failure") {
        try { session->call_event(call, CallStage::CompleteRequired, {999}); }
        catch (const std::exception &error) { session->record_failure(error.what()); }
        rejects([&] { session->ensure_healthy(); });
        rejects([&] { session->finish(); });
        std::cout << "PASS invalid call rejected and latched; successful completion rejected\n";
        return 0;
    }
    ggml::gemmini::cycle::reset_read_count_for_test();
    session->call_event(call, CallStage::Invoke);
    rejects([&] { session->call_event(call, CallStage::Invoke); });
    const auto prepare = session->host_stage_begin(call, {"fixture.prepare", "POTAL_HOST", "llama.cpp-gemmini",
        "tests/test-gemmini-cycle-sim-log.cpp:main", {}, {}});
    {
        ScopedContext scope(prepare);
        const auto correlation = ggml::gemmini::log::current_cpu_correlation();
        assert(correlation.host_stage_id == *prepare.host_stage_id && correlation.semantic_context == operation.semantic_context);
    }
    rejects([&] { session->finish_operation(context); });
    rejects([&] { session->host_stage_begin(call, {"fixture.too-early", "POTAL_HOST", "llama.cpp-gemmini",
        "tests/test-gemmini-cycle-sim-log.cpp:main", {}, {*prepare.host_stage_id}}); });
    session->host_stage_end(prepare);
    rejects([&] { session->host_stage_end(prepare); });
    Work work;
    work.geometry = {1, sizeof(im2p_production_geometry_v1_t), info.activation_bits, info.weight_bits, info.dim,
                     IM2P_GEOMETRY_FULL, 1, 3, info.dim, 1, 1, 1, 1, 0, 1, 0};
    work.m = work.row_count = 1;
    work.activation_stride_bytes = info.dim;
    work.weight_stride_bytes = 3;
    work.output_stride_bytes = 12;
    work.scale_stride_elements = 3;
    work.required_host_stage_ids = {*prepare.host_stage_id};
    Work invalid_dependency = work;
    invalid_dependency.required_host_stage_ids.push_back(999);
    rejects([&] { session->work(call, invalid_dependency); });
    const auto work_id = session->work(call, work);
    assert(work_id == 0);
    const auto emulation = session->host_stage_begin(call, {"fixture.reference", "FUNCTIONAL_EMULATION", "IM2P.sim",
        "frontend/src/im2p_cpu_functional_compute.cpp:execute", {work_id}, {*prepare.host_stage_id}});
    session->host_stage_end(emulation);
    rejects([&] { session->host_stage_begin(call, {"fixture.bad", "POTAL_HOST", "llama.cpp-gemmini",
        "../escape.cpp:bad", {}, {}}); });
    rejects([&] { session->finish_operation(context); });
    rejects([&] { session->call_event(call, CallStage::Continuation); });
    {
        ScopedContext scope(call);
        const auto correlation = ggml::gemmini::log::current_cpu_correlation();
        assert(correlation.collection_run_id == 1 && correlation.call_id == *call.call_id);
        std::thread worker([call] {
            ScopedContext worker_scope(call);
            assert(ggml::gemmini::log::current_cpu_correlation().target_node_id == *call.node_id);
        });
        worker.join();
    }
    assert(!ggml::gemmini::log::current_cpu_correlation().present);
    session->call_event(call, CallStage::CompleteRequired, {work_id});
    session->call_event(call, CallStage::Continuation);
    rejects([&] { session->call_event(call, CallStage::Continuation); });
    const auto fence = session->call_begin(dispatch, CallKind::Fence);
    session->call_event(fence, CallStage::Invoke);
    session->call_event(fence, CallStage::CompleteRequired, {work_id});
    session->call_event(fence, CallStage::Fence);
    session->call_event(fence, CallStage::Continuation);
    assert(session->finish_operation(context));
    const auto decode = session->phase("decode", 0, 1);
    operation.actual_backend = "CPU";
    operation.target_eligible = false;
    operation.semantic_context = std::make_shared<ggml::gemmini::semantic::Context>(
        ggml::gemmini::semantic::Context{{"decode", 0, 0, 0}, ggml::gemmini::semantic::Source::PotalCollection, info.run_config_id});
    const auto cpu_context = session->register_operation(&node, operation, decode);
    auto *cpu_scope = gemmini_cycle_sim_context_enter(&node);
    assert(cpu_scope && ggml::gemmini::log::current_cpu_correlation().operation_id == *cpu_context.operation_id);
    gemmini_cycle_sim_context_exit(cpu_scope);
    assert(!session->finish_operation(cpu_context));
    session->finish();
    assert(ggml::gemmini::cycle::read_count_for_test() == 0);
    std::ifstream input(path);
    const std::string data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    assert(data.find("\"schema\":\"im2p-npu-cycle-trace\"") != std::string::npos);
    assert(data.find("\"registered_operation_count\":2") != std::string::npos);
    assert(data.find("\"call_count\":2") != std::string::npos);
    assert(data.find("\"host_stage_count\":2") != std::string::npos);
    assert(data.find("\"completed_host_stage_count\":2") != std::string::npos);
    assert(data.find("\"semantic_node_ordinal\":0") != std::string::npos);
    assert(data.find("\"selected_target\":\"ORDINARY_CPU\"") != std::string::npos);
    assert(data.find("\"selected_target\":\"TARGET_NPU\"") != std::string::npos);
    for (const char *forbidden : {"CPU_INTERVAL", "TRACE_OVERHEAD", "\"start\":", "\"end\":", "\"delta\":"})
        assert(data.find(forbidden) == std::string::npos);
    for (const char *other : {"cycle-log.jsonl", "cycle-sim-log.jsonl", "optrace.jsonl"})
        assert(!std::filesystem::exists(std::filesystem::path(directory) / other));
    std::cout << "PASS NPU-only writer, call ownership and independent CPU correlation: " << path << '\n';
}
