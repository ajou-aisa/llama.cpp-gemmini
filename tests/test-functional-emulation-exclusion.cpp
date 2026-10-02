#include <gemmini/log.hpp>
#if LOG_CYCLE || CYCLE_SIM
#include <gemmini/semantic.hpp>
#endif

#include <cassert>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

namespace log = ggml::gemmini::log;

[[maybe_unused]] static std::string interval(const char *op,
        ggml::gemmini::cycle::TimingIntervalClass interval_class =
            ggml::gemmini::cycle::TimingIntervalClass::diagnostic) {
    auto record = log::CycleRecord{"layer", op, 100, 200};
    record.timing_interval_class = interval_class;
    return log::serialize_cycle_record(record);
}

#if LOG_CYCLE || CYCLE_SIM
static void source_authority() {
    namespace semantic = ggml::gemmini::semantic;
    auto unbound = log::CycleRecord{"layer", "cpu.mul_mat", 100, 200};
    unbound.correlation = log::current_cpu_correlation();
    auto full_cpu = std::make_shared<semantic::Context>();
    full_cpu->identity = {"decode", 2, 3, 4};
    full_cpu->duration_source = semantic::Source::FullCpu;
    full_cpu->run_config_id = "fixture-workload";
    log::CpuCorrelation correlation;
    correlation.semantic_context = full_cpu;
    correlation.worker_count = 2;
    const log::ScopedCpuCorrelation full_scope(correlation);
    const std::string ordinary = interval("cpu.mul_mat",
        ggml::gemmini::cycle::TimingIntervalClass::per_worker_cpu_work);
    assert(ordinary.find("\"duration_source\":\"FULL_CPU\"") != std::string::npos);
    assert(ordinary.find("\"duration_role\":\"ORDINARY_CPU_REFERENCE\"") != std::string::npos);
    assert(ordinary.find("\"run_config_id\":\"fixture-workload\"") != std::string::npos);
    assert(ordinary.find("\"worker_count\":2") != std::string::npos);
    assert(ordinary.find("\"delta\":100,\"valid\":true") != std::string::npos);
    assert(log::serialize_cycle_record(unbound).find("\"exclusion_reason\":\"outside_collection\"") != std::string::npos);
    auto stored = log::CycleRecord{"layer", "cpu.mul_mat", 100, 200};
    stored.correlation = correlation;
    auto collection = std::make_shared<semantic::Context>(*full_cpu);
    collection->duration_source = semantic::Source::PotalCollection;
    correlation.semantic_context = collection;
    const log::ScopedCpuCorrelation collection_scope(correlation);
    const std::string observed = interval("cpu.mul_mat");
    assert(observed.find("\"duration_source\":\"POTAL_COLLECTION\"") != std::string::npos);
    assert(observed.find("\"duration_role\":\"OBSERVATION_ONLY\"") != std::string::npos);
    assert(observed.find("\"delta\":100,\"valid\":true") != std::string::npos);
    assert(log::serialize_cycle_record(stored).find("\"duration_source\":\"FULL_CPU\"") != std::string::npos);
    assert(interval("production.radix").find("\"duration_role\":\"OBSERVATION_ONLY\"") != std::string::npos);
    correlation.host_stage_id = 41;
    const log::ScopedCpuCorrelation stage_scope(correlation);
    const auto host = interval("production.radix",
        ggml::gemmini::cycle::TimingIntervalClass::canonical_additive);
    assert(host.find("\"duration_role\":\"POTAL_HOST\"") != std::string::npos);
    assert(host.find("\"host_stage_id\":41") != std::string::npos);
    assert(host.find("\"delta\":100,\"valid\":true") != std::string::npos);
    assert(interval("cpu.mul_mat").find("\"duration_role\":\"OBSERVATION_ONLY\"") != std::string::npos);
#if CYCLE_SIM
    const log::ScopedFunctionalEmulationSuppression guard;
    const auto excluded = interval("production.radix");
    assert(excluded.find("\"duration_role\":\"OBSERVATION_ONLY\"") != std::string::npos);
    assert(excluded.find("\"delta\":null") != std::string::npos);
    assert(excluded.find("\"valid\":false") != std::string::npos);
#endif
}
#endif

int main() {
#if LOG_CYCLE || CYCLE_SIM
    source_authority();
#endif
    const ggml::gemmini::log::CycleRecord record{
        "layer", "im2p.fence_host_call", 100, 1000000};
    const std::string json = ggml::gemmini::log::serialize_cycle_record(record);
#if LOG_CYCLE || CYCLE_SIM
    assert(json.find("\"duration_role\":\"OBSERVATION_ONLY\"") != std::string::npos);
    assert(json.find("\"exclusion_reason\":\"outside_collection\"") != std::string::npos);
#endif
#if CYCLE_SIM
    assert(json.find("\"cpu_service\":false") != std::string::npos);
    assert(json.find("\"delta\":null") != std::string::npos);
    assert(json.find("\"start\":100") != std::string::npos);
    assert(json.find("\"end\":1000000") != std::string::npos);
    log::CpuCorrelation correlation;
    correlation.present = true;
    correlation.collection_run_id = 11;
    correlation.phase_id = 12;
    correlation.operation_id = 13;
    correlation.target_node_id = 14;
    correlation.call_id = 15;
    log::ScopedCpuCorrelation scope(correlation);
    const auto start = log::capture_cpu_exclusion();
    const std::string before = interval("production.quantization");
    std::string emulation;
    std::string other_worker;
    try {
        log::ScopedFunctionalEmulationSuppression guard;
        {
            log::ScopedFunctionalEmulationSuppression nested;
            assert(log::capture_cpu_exclusion().active);
        }
        assert(log::capture_cpu_exclusion().active);
        std::thread worker([&] { other_worker = interval("cpu.mul_mat"); });
        worker.join();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        emulation = interval("cpu.mul_mat");
        throw std::runtime_error("test unwind");
    } catch (const std::runtime_error &) {}
    assert(!log::capture_cpu_exclusion().active);
    assert(std::string(log::cpu_exclusion_since(start)) == "functional_emulation");
    auto enclosing = log::CycleRecord{"layer", "coarse", 100, 1000000};
    enclosing.cpu_service_exclusion = log::cpu_exclusion_since(start);
    const std::string outer = log::serialize_cycle_record(enclosing);
    assert(outer.find("\"delta\":null") != std::string::npos);
    assert(outer.find("\"reason\":\"functional_emulation\"") != std::string::npos);
    assert(emulation.find("\"cpu_service\":false") != std::string::npos);
    assert(emulation.find("\"delta\":null") != std::string::npos);
    assert(other_worker.find("\"cpu_service\":true") != std::string::npos);
    assert(other_worker.find("\"delta\":100") != std::string::npos);
    assert(other_worker.find("collection_run_id") == std::string::npos);
    for (const std::string &legitimate : {before, interval("production.radix"),
            interval("production.output_reconstruction"), interval("im2p.residual_backend_host_call")}) {
        assert(legitimate.find("\"cpu_service\":true") != std::string::npos);
        assert(legitimate.find("\"delta\":100") != std::string::npos);
        assert(legitimate.find("\"collection_run_id\":11") != std::string::npos);
        assert(legitimate.find("\"operation_id\":13") != std::string::npos);
        assert(legitimate.find("\"target_node_id\":14") != std::string::npos);
        assert(legitimate.find("\"call_id\":15") != std::string::npos);
        assert(legitimate.find("\"work_id\":null") != std::string::npos);
    }
    {
        auto inner = correlation;
        inner.operation_id = 99;
        log::ScopedCpuCorrelation nested(inner);
        assert(log::current_cpu_correlation().operation_id == 99);
    }
    assert(log::current_cpu_correlation().operation_id == 13);
#else
    assert(json.find("\"delta\":999900") != std::string::npos);
    assert(json.find("cpu_service") == std::string::npos);
#endif
    FILE *sink = std::tmpfile();
    assert(sink != nullptr);
    gemmini_log_cycle_set_output(sink);
    gemmini_cycle_record_v2 ordinary{{"layer", "cpu.mul_mat", 100, 200, nullptr, 0, nullptr},
        GEMMINI_CYCLE_HAS_RUN_ID | GEMMINI_CYCLE_HAS_NODE_ID, 7, 0, 0, 8, 0};
    {
        log::ScopedFunctionalEmulationSuppression guard;
        gemmini_log_cycle_record_v2(&ordinary);
    }
    ordinary.interval.op = "production.radix";
    gemmini_log_cycle_record_v2(&ordinary);
    assert(gemmini_log_cycle_flush());
    std::rewind(sink);
    std::string emitted;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), sink)) emitted += buffer;
    gemmini_log_cycle_set_output(stderr);
    std::fclose(sink);
#if LOG_CYCLE
#if CYCLE_SIM || CYCLE_DETAIL
    assert(emitted.find("\"schema\":\"gemmini.cycle\"") != std::string::npos);
#else
    assert(emitted.find("\"kind\":\"cycle\"") != std::string::npos);
#endif
    assert(emitted.find("\"run_id\":7") != std::string::npos);
    assert(emitted.find("\"node_id\":8") != std::string::npos);
#if CYCLE_SIM
    assert(emitted.find("\"cpu_service_exclusion\":\"functional_emulation\"") != std::string::npos);
    assert(emitted.find("\"collection_run_id\":11") != std::string::npos);
#endif
#else
    assert(emitted.empty());
#endif
#if LOG_CYCLE
    FILE *metric_sink = std::tmpfile();
    assert(metric_sink != nullptr);
    gemmini_log_cycle_set_output(metric_sink);
    auto semantic_context = std::make_shared<ggml::gemmini::semantic::Context>();
    semantic_context->identity = {"prefill", {}, 0, 0};
    semantic_context->duration_source = ggml::gemmini::semantic::Source::FullCpu;
    semantic_context->run_config_id = "metric-contract";
    log::CpuCorrelation metric_correlation;
    metric_correlation.semantic_context = semantic_context;
    metric_correlation.worker_count = 1;
    const log::ScopedCpuCorrelation metric_scope(metric_correlation);
    gemmini_cycle_record_v2 metric_identity{{"layer", "cpu.mul_mat", 0, 0, nullptr, 0, nullptr},
        GEMMINI_CYCLE_HAS_NODE_ID | GEMMINI_CYCLE_HAS_WORKER_ID, 0, 0, 0, 0, 0};
    gemmini_cpu_sample metric_start{};
    metric_start.ns = 1000;
    metric_start.tid = 77;
    metric_start.thread_cpu_ns = 2000;
    metric_start.thread_cpu_valid = 1;
    metric_start.counter = 3000;
    metric_start.native_valid = 1;
    metric_start.native_source = GEMMINI_CPU_COUNTER_THREAD_PERF;
    metric_start.owner_token = 9;
    metric_start.generation = 4;
    gemmini_cpu_sample metric_end = metric_start;
    metric_end.ns = 1150;
    metric_end.thread_cpu_ns = 2120;
    metric_end.counter = 3100;
    gemmini_cpu_timing_record(&metric_identity, &metric_start, &metric_end);
    assert(gemmini_log_cycle_flush());
    gemmini_log_cycle_set_output(stderr);
    std::rewind(metric_sink);
    std::string metric_json;
    while (std::fgets(buffer, sizeof(buffer), metric_sink)) metric_json += buffer;
    std::fclose(metric_sink);
    assert(metric_json.find("\"cpu_work_cycles\":100") != std::string::npos);
    assert(metric_json.find("\"thread_cpu_ns\":120") != std::string::npos);
    assert(metric_json.find("\"host_elapsed_ns\":150") != std::string::npos);
    assert(metric_json.find("\"thread_id\":77") != std::string::npos);
    assert(metric_json.find("\"interval_class\":\"PER_WORKER_CPU_WORK\"") != std::string::npos);
#endif
    std::puts("functional emulation exclusion PASS");
}
