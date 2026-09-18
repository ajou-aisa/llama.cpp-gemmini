#include <gemmini/cycle_reader.h>
#include <gemmini/log.hpp>
#include <gemmini/performance.hpp>
#if defined(__linux__) && defined(__aarch64__)
#include "cycle_reader_internal.h"
#endif

static_assert(noexcept(gemmini_read_cycles()));
static_assert(noexcept(gemmini_log_cycle_set_buffered(1)));
static_assert(noexcept(gemmini_log_cycle_flush()));

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

extern "C" int gemmini_log_c_boundary_call(int operation, const char * path);

static std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

static int current_process_id() {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

#if defined(__linux__) && defined(__aarch64__)
static bool checked_bridge_matrix(const std::filesystem::path & path) {
#if !EXPECT_LOG_CYCLE
    (void) path;
    return true;
#else
    if (!gemmini_log_cycle_set_output_path(path.c_str())) return false;
    const gemmini_cycle_record_v2 record{{"private", "matrix", 0, 0, nullptr, 0, nullptr}, 0, 0, 0, 0, 0, 0};
    const gemmini_native_cycle_sample_internal valid_start{10, 1, GEMMINI_NATIVE_CYCLE_REASON_NONE,
        GEMMINI_NATIVE_CYCLE_SOURCE_LINUX_PERF_CPU_CYCLES, 7, 9};
    const gemmini_native_cycle_sample_internal valid_end{12, 1, GEMMINI_NATIVE_CYCLE_REASON_NONE,
        GEMMINI_NATIVE_CYCLE_SOURCE_LINUX_PERF_CPU_CYCLES, 7, 9};
    auto emit = [&](gemmini_native_cycle_sample_internal start,
                    gemmini_native_cycle_sample_internal end, int eligible) {
        gemmini_log_cycle_record_v2_checked_internal(&record, &start, &end, eligible);
    };
    emit(valid_start, valid_end, 1);
    auto invalid_start = valid_start; invalid_start.valid = 0;
    invalid_start.reason = GEMMINI_NATIVE_CYCLE_REASON_UNAVAILABLE_EVENT;
    emit(invalid_start, valid_end, 1);
    auto invalid_end = valid_end; invalid_end.valid = 0;
    emit(valid_start, invalid_end, 1);
    auto source_end = valid_end; source_end.source = GEMMINI_NATIVE_CYCLE_SOURCE_APPLE_HOST_TICK;
    emit(valid_start, source_end, 1);
    auto owner_end = valid_end; owner_end.owner_event_token = 8;
    emit(valid_start, owner_end, 1);
    auto generation_end = valid_end; generation_end.generation = 10;
    emit(valid_start, generation_end, 1);
    emit(valid_start, valid_end, 0);
    auto regression_end = valid_end; regression_end.value = 9;
    emit(valid_start, regression_end, 1);
    auto zero_end = valid_end; zero_end.value = valid_start.value;
    emit(valid_start, zero_end, 1);
    const ggml::gemmini::log::CycleRecord multiplexed{
        "private", "matrix.multiplexed", 10, 12, nullptr, 0, nullptr,
        "linux_perf_cpu_cycles", "cycle"};
    ggml::gemmini::log::cycle.write_json(
        ggml::gemmini::log::serialize_checked_cycle_record(multiplexed, false, "multiplexed"));
    gemmini_log_cycle_set_output(stderr);

    const std::string output = read_file(path);
    const char * reasons[] = {"invalid_start", "invalid_end", "source_mismatch",
        "event_owner_mismatch", "event_generation_mismatch", "structurally_cross_task",
        "counter_regression", "multiplexed"};
    if (output.find("\"source\":\"linux_perf_cpu_cycles\",\"unit\":\"cycle\"") == std::string::npos ||
        output.find("\"delta\":2,\"valid\":true") == std::string::npos ||
        output.find("\"delta\":0,\"valid\":true") == std::string::npos) return false;
    for (const char * reason : reasons) {
        if (output.find(std::string("\"delta\":null,\"valid\":false,\"reason\":\"") + reason + "\"") ==
            std::string::npos) return false;
    }
    if (output.find("\"reason\":\"invalid_start\",\"sample_reason\":\"unavailable_event\"") ==
            std::string::npos) return false;
    return true;
#endif
}
#endif

static int open_descriptor_count() {
#if defined(_WIN32)
    return 0;
#else
    struct rlimit limit {};
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) return -1;
    int count = 0;
    for (int fd = 0; fd < static_cast<int>(limit.rlim_cur); ++fd) {
        if (fcntl(fd, F_GETFD) != -1) ++count;
    }
    return count;
#endif
}

static bool test_buffered_cycle_output(const std::filesystem::path & root) {
    using namespace ggml::gemmini::log;
    const auto path = root / "buffered.jsonl";
    const auto targeted = root / "buffered-target.jsonl";
    const auto borrowed_path = root / "borrowed.jsonl";
    CycleLog output;
    output.set_buffered(true);
    if (!output.set_output_path(path.c_str())) return false;
    output.write_json("{\"buffered\":1}");
    if (!read_file(path).empty() || !output.flush()) return false;
#if EXPECT_LOG_CYCLE
    if (read_file(path) != "{\"buffered\":1}\n") return false;
    output.write_json("{\"pre_truncate\":1}");
    if (!output.set_output_path(path.c_str(), true)) return false;
    output.write_json("{\"after_truncate\":1}");
    if (!output.flush() || read_file(path) != "{\"after_truncate\":1}\n") return false;
    output(file(targeted.c_str()), "buffered", "targeted", 1, 2);
    if (read_file(targeted).find("targeted") == std::string::npos) return false;
    const std::string large_record = "{\"large\":\"" + std::string(3 * BUFSIZ, 'x') + "\"}\n";
    output.write_json(large_record);
    output(file(path.c_str()), "buffered", "same_file_target", 1, 2);
    if (read_file(path).find(large_record) == std::string::npos) return false;
    output.write_json("{\"disable_buffering\":1}");
    output.set_buffered(false);
    if (read_file(path).find("disable_buffering") == std::string::npos) return false;
    output.write_json("{\"immediate\":1}");
    if (read_file(path).find("immediate") == std::string::npos) return false;

    FILE * borrowed = std::fopen(borrowed_path.c_str(), "w");
    if (!borrowed) return false;
    output.set_buffered(true);
    output.set_output(borrowed);
    output.write_json("{\"borrowed\":1}");
    const bool borrowed_flushed = read_file(borrowed_path) == "{\"borrowed\":1}\n";
    output.set_output(nullptr);
    const bool borrowed_open = std::fputs("still-open\n", borrowed) >= 0;
    const bool borrowed_closed = std::fclose(borrowed) == 0;
    if (!borrowed_flushed || !borrowed_open || !borrowed_closed) return false;

#if !defined(_WIN32)
    const auto stdout_path = root / "stdout.jsonl";
    std::fflush(stdout);
    const int saved_stdout = dup(fileno(stdout));
    FILE * stdout_file = std::fopen(stdout_path.c_str(), "w");
    if (saved_stdout < 0 || !stdout_file) return false;
    if (dup2(fileno(stdout_file), fileno(stdout)) < 0) return false;
    output.set_output(stdout);
    output.write_json("{\"stdout\":1}");
    const bool stdout_flushed = read_file(stdout_path) == "{\"stdout\":1}\n";
    output.set_output(nullptr);
    const bool stdout_restored = dup2(saved_stdout, fileno(stdout)) >= 0;
    close(saved_stdout);
    std::fclose(stdout_file);
    if (!stdout_flushed || !stdout_restored) return false;
    if (!output.set_output_path("/dev/null")) return false;
    testing::set_log_fault(testing::LogFault::flush);
    output.write_json("{\"nonregular\":1}");
    if (output.flush()) return false;
#endif

    if (!gemmini_log_cycle_set_output_path(path.c_str()) ||
        !gemmini_log_c_boundary_call(7, nullptr)) return false;
    testing::set_log_fault(testing::LogFault::flush);
    gemmini_log_cycle("buffered", "flush_failure", 1, 2);
    if (gemmini_log_c_boundary_call(8, nullptr) != 0) return false;
    gemmini_log_cycle("buffered", "disabled_record", 2, 3);
    if (gemmini_log_cycle_flush() != 0 ||
        read_file(path).find("disabled_record") != std::string::npos) return false;
    if (!gemmini_log_cycle_set_output_path(path.c_str())) return false;
    gemmini_log_cycle("buffered", "recovered", 3, 4);
    if (!gemmini_log_cycle_flush() || read_file(path).find("recovered") == std::string::npos) return false;
    testing::set_log_fault(testing::LogFault::mutex);
    if (gemmini_log_c_boundary_call(8, nullptr) != 0) return false;
#else
    output(file(targeted.c_str()), "off", "targeted", 1, 2);
    if (!gemmini_log_c_boundary_call(7, nullptr) || !gemmini_log_c_boundary_call(8, nullptr) ||
        std::filesystem::exists(path) || std::filesystem::exists(targeted)) return false;
#endif
    gemmini_log_cycle_set_buffered(0);
    gemmini_log_cycle_set_output(stderr);
    return true;
}

static bool test_inference_log_context(const std::filesystem::path & root) {
    namespace perf = ggml::gemmini::performance;
    ggml::gemmini::log::CycleLog output;
    const auto path = root / "inference-context.jsonl";
    output.set_buffered(true);
    if (!output.set_output_path(path.c_str())) return false;
    perf::reset();
    output.write_json("{\"warmup\":true}");
    perf::start_request(100);
    output.write_json("{}");
    perf::begin_operation(perf::Phase::prefill, 110);
    const std::string batch = "{\"text\":\"escaped\\nline\",\"host_timing\":{\"start_tid\":123}}\n"
                              "{\"batch\":2}\n";
    output.write_json(batch);
    perf::end_operation(120, true);
    perf::begin_operation(perf::Phase::decode, 130);
    output.write_json("{\"decode\":true}");
    perf::end_operation(140, true);
    perf::finish_request(150);
    output.write_json("{\"after\":true}");
    const bool flushed = output.flush();
    perf::reset();
    if (!flushed) return false;
#if EXPECT_LOG_CYCLE
    const std::string prefill = "\"inference_context\":{\"request_id\":1,\"operation_id\":1,"
                                "\"phase\":\"prefill\",\"included\":true}";
    const std::string expected = "{\"warmup\":true}\n"
        "{\"inference_context\":{\"request_id\":1,\"operation_id\":null,\"phase\":null,\"included\":true}}\n"
        "{\"text\":\"escaped\\nline\",\"host_timing\":{\"start_tid\":123}," + prefill + "}\n"
        "{\"batch\":2," + prefill + "}\n"
        "{\"decode\":true,\"inference_context\":{\"request_id\":1,\"operation_id\":2,"
        "\"phase\":\"decode\",\"included\":true}}\n{\"after\":true}\n";
    return read_file(path) == expected && batch.find("inference_context") == std::string::npos;
#else
    return !std::filesystem::exists(path);
#endif
}

static bool test_hardware_cycle_summary(const std::filesystem::path & root) {
    using namespace ggml::gemmini::performance;
    const auto path = root / "hardware-summary.jsonl";
    if (!gemmini_log_cycle_set_output_path(path.c_str())) return false;
    gemmini_log_cycle_set_buffered(1);
    reset();
    start_request(0);
    begin_operation(Phase::prefill, 1);
    gemmini_cycle_record_v2 identity{};
    identity.interval.layer = "hw.scope";
    identity.identity_mask = GEMMINI_CYCLE_HAS_RUN_ID | GEMMINI_CYCLE_HAS_STRIPE_ID |
        GEMMINI_CYCLE_HAS_WORKER_ID;
    {
        const ggml::gemmini::log::ScopedWsCycleIdentity dense(identity, "gemmini_hw_dense");
        gemmini_log_ws_cycle(1000, 11, 22, 33, 44, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0);
        {
            identity.stripe_id = 1;
            const ggml::gemmini::log::ScopedWsCycleIdentity residual(identity, "gemmini_hw_residual");
            gemmini_log_ws_cycle(0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0);
        }
        gemmini_log_ws_cycle(10, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0);
    }
    end_operation(2, true);
    begin_operation(Phase::decode, 3);
    gemmini_log_ws_cycle(10, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0);
    end_operation(4, true);
    finish_request(5);
    finish_recording();
    const bool flushed = gemmini_log_cycle_flush() != 0;
    const auto replay = read_summary(path);
    const std::string summary = replay.serialize();
    gemmini_log_cycle_set_buffered(0);
    gemmini_log_cycle_set_output(stderr);
    if (!flushed) return false;
#if EXPECT_LOG_CYCLE
    if (!replay.available) return false;
    const std::string raw = read_file(path);
    const auto count = [&raw](const std::string & needle) {
        size_t found = 0;
        for (size_t at = 0; (at = raw.find(needle, at)) != std::string::npos; at += needle.size()) ++found;
        return found;
    };
    if (count("\"layer\":\"hw.scope\",\"domain\":\"gemmini_hw_dense\",\"run_id\":0,\"stripe_id\":0") != 2 ||
        count("\"layer\":\"hw.scope\",\"domain\":\"gemmini_hw_residual\",\"run_id\":0,\"stripe_id\":1") != 1 ||
        count("\"layer\":null,\"domain\":\"gemmini_hw_unknown\",\"run_id\":null") != 1 ||
        count("\"worker_id\":0,") != 3 || count("\"load_occupancy_cycles\":0,") != 1) return false;
    for (const char * metric : {"load_occupancy_cycles\",\"cycles\":null,",
                               "execute_occupancy_cycles\",\"cycles\":null,",
                               "store_occupancy_cycles\",\"cycles\":null,",
                               "loop_occupancy_cycles\",\"cycles\":null,"}) {
        if (summary.find(std::string("\"metric\":\"") + metric) == std::string::npos) {
            std::fprintf(stderr, "missing hardware counter metric %s in %s\n", metric, summary.c_str());
            return false;
        }
    }
    return summary.find("\"backend\":\"HARDWARE\",\"domain\":\"gemmini_hw_dense\"") != std::string::npos &&
        summary.find("\"domain\":\"gemmini_hw_residual\"") != std::string::npos &&
        summary.find("\"domain\":\"gemmini_hw_unknown\"") != std::string::npos &&
        summary.find("\"metric\":\"work_total_cycles\",\"cycles\":null") != std::string::npos &&
        summary.find("unavailable_device_elapsed_counter") != std::string::npos &&
        summary.find("device_counter_window_and_wrap_unverified") != std::string::npos &&
        summary.find("\"cycles\":1000") == std::string::npos;
#else
    return summary.find("gemmini_hw_") == std::string::npos;
#endif
}

static bool test_worker_cycle_buffers(const std::filesystem::path & root) {
#if EXPECT_LOG_CYCLE
    using namespace ggml::gemmini::log;
    const auto path = root / "worker-buffers.jsonl";
    CycleLog output;
    output.set_buffered(true);
    if (!output.set_output_path(path.c_str()) || output.output_path() != path) return false;
    constexpr std::size_t thread_count = 4;
    constexpr std::size_t records = 3 * CycleLog::BufferStats::max_entries + 7;
    std::vector<std::thread> workers;
    for (std::size_t worker = 0; worker < thread_count; ++worker) {
        workers.emplace_back([&, worker] {
            for (std::size_t record = 0; record < records; ++record) {
                char layer[80], op[80];
                std::snprintf(layer, sizeof(layer), "worker_%zu_record_%zu", worker, record);
                std::snprintf(op, sizeof(op), "stack_op_%zu_%zu", worker, record);
                gemmini_cycle_record_v2 identity{};
                identity.interval.layer = layer;
                identity.interval.op = op;
                identity.identity_mask = GEMMINI_CYCLE_HAS_WORKER_ID;
                identity.worker_id = worker;
                gemmini_cpu_sample start{}, end{};
                start.ns = 10; end.ns = 20;
                start.tid = end.tid = worker + 1;
                output.write_cpu(identity, start, end);
                std::fill(std::begin(layer), std::end(layer), '?');
                std::fill(std::begin(op), std::end(op), '?');
                start.ns = end.ns = 99;
            }
        });
    }
    for (auto & worker : workers) worker.join();
    const auto stats = output.buffer_stats_for_test();
    if (stats.workers != 0 || stats.peak_entries != CycleLog::BufferStats::max_entries ||
        stats.peak_bytes == 0 || stats.peak_bytes > CycleLog::BufferStats::max_bytes ||
        !output.healthy() || !output.flush()) return false;
    const std::string json = read_file(path);
    if (static_cast<std::size_t>(std::count(json.begin(), json.end(), '\n')) !=
        thread_count * records) return false;
    for (std::size_t worker = 0; worker < thread_count; ++worker) {
        for (std::size_t record = 0; record < records; ++record) {
            const std::string suffix = std::to_string(worker) + "_" + std::to_string(record);
#if EXPECT_CYCLE_DETAIL
            const std::string names = "\"layer\":\"worker_" + std::to_string(worker) +
                "_record_" + std::to_string(record) + "\",\"op\":\"stack_op_" + suffix + "\"";
#else
            const std::string names = "{\"op\":\"stack_op_" + suffix +
                "\",\"kind\":\"cpu\",\"layer\":\"worker_" + std::to_string(worker) +
                "_record_" + std::to_string(record) + "\"";
#endif
            if (json.find(names) == std::string::npos) return false;
        }
    }
#if EXPECT_CYCLE_DETAIL
    if (json.find("\"start_ns\":99") != std::string::npos ||
        json.find("\"start_ns\":10") == std::string::npos) return false;
#else
    if (json.find("\"start_ns\"") != std::string::npos ||
        json.find("\"thread_cpu_timing\"") != std::string::npos) return false;
#endif
#else
    (void) root;
#endif
    return true;
}

static bool test_cycle_buffer_bytes(const std::filesystem::path & root) {
#if EXPECT_LOG_CYCLE
    using namespace ggml::gemmini::log;
    namespace perf = ggml::gemmini::performance;
    const auto path = root / "buffer-bytes.jsonl";
    CycleLog output;
    output.set_buffered(true);
    if (!output.set_output_path(path.c_str())) return false;
    std::string expected;
    const std::size_t sizes[] = {3, 8193, CycleLog::BufferStats::max_bytes / 4,
        CycleLog::BufferStats::max_bytes / 2 + 13, CycleLog::BufferStats::max_bytes + 31};
    for (int pass = 0; pass < 3; ++pass) {
        for (const auto size : sizes) {
            std::string record = "{\"payload\":\"" + std::string(size, 'a' + pass) + "\"}\n";
            expected += record;
            output.write_json(record);
            record.assign(record.size(), '?');
        }
        perf::Measurement measurement;
        measurement.kind = perf::Measurement::Kind::wall_gap;
        measurement.sequence = pass + 1;
        measurement.reason = "owned_measurement_" + std::string(8193, 'x' + pass);
        expected += perf::serialize_measurement(measurement) + '\n';
        output.write_measurement(measurement);
        measurement.reason.assign(measurement.reason.size(), '?');
    }
    const auto stats = output.buffer_stats_for_test();
    if (stats.workers != 1 || stats.peak_entries == 0 ||
        stats.peak_entries > CycleLog::BufferStats::max_entries || stats.peak_bytes == 0 ||
        stats.peak_bytes > CycleLog::BufferStats::max_bytes || !output.healthy() ||
        !output.flush() || read_file(path) != expected) return false;
#else
    (void) root;
#endif
    return true;
}

static bool test_cycle_buffer_ownership(const std::filesystem::path & root) {
#if EXPECT_LOG_CYCLE
    using namespace ggml::gemmini::log;
    const auto first_path = root / "first-owner.jsonl";
    const auto second_path = root / "second-owner.jsonl";
    const auto routed_path = root / "rerouted-owner.jsonl";
    {
        CycleLog first, second;
        first.set_buffered(true);
        second.set_buffered(true);
        if (!first.set_output_path(first_path.c_str()) ||
            !second.set_output_path(second_path.c_str())) return false;
        first.write_json("{\"first\":1}");
        second.write_json("{\"second\":1}");
        if (first.buffer_stats_for_test().workers != 0 ||
            second.buffer_stats_for_test().workers != 1 || !first.flush() ||
            read_file(first_path) != "{\"first\":1}\n") return false;
        first.write_json("{\"before_route\":1}");
        if (second.buffer_stats_for_test().workers != 0 || !second.flush() ||
            read_file(second_path) != "{\"second\":1}\n") return false;
        if (!first.set_output_path(routed_path.c_str()) ||
            first.output_path() != routed_path ||
            read_file(first_path) != "{\"first\":1}\n{\"before_route\":1}\n") return false;
        first.write_json("{\"after_route\":1}");
        first.set_buffered(false);
        if (read_file(routed_path) != "{\"after_route\":1}\n") return false;
        first.set_buffered(true);
        first.write_json("{\"destructor\":1}");
    }
    if (read_file(routed_path) != "{\"after_route\":1}\n{\"destructor\":1}\n") return false;
    CycleLog replacement;
    replacement.set_buffered(true);
    if (!replacement.set_output_path(second_path.c_str())) return false;
    replacement.write_json("{\"replacement\":1}");
    if (!replacement.flush() || replacement.buffer_stats_for_test().workers != 1 ||
        read_file(second_path) != "{\"second\":1}\n{\"replacement\":1}\n") return false;
    std::promise<void> queued, release;
    auto released = release.get_future();
    std::thread worker;
    const auto detached_path = root / "detached-worker.jsonl";
    {
        CycleLog transient;
        transient.set_buffered(true);
        if (!transient.set_output_path(detached_path.c_str())) return false;
        worker = std::thread([&] {
            transient.write_json("{\"detached_worker\":1}");
            queued.set_value();
            released.wait();
        });
        queued.get_future().wait();
    }
    release.set_value();
    worker.join();
    if (read_file(detached_path) != "{\"detached_worker\":1}\n") return false;
#else
    (void) root;
#endif
    return true;
}

static bool test_queued_cycle_faults(const std::filesystem::path & root) {
#if EXPECT_LOG_CYCLE
    using namespace ggml::gemmini::log;
    {
        CycleLog output;
        output.set_buffered(true);
        const auto path = root / "queued-hot-path.jsonl";
        if (!output.set_output_path(path.c_str())) return false;
        gemmini_cycle_record_v2 identity{};
        identity.interval.op = "queued_without_global_lock";
        const gemmini_cpu_sample start{}, end{};
        output.write_cpu(identity, start, end);
        testing::set_log_fault(testing::LogFault::mutex);
        output.write_cpu(identity, start, end);
        testing::clear_log_fault();
        if (!output.healthy() || !output.flush()) return false;
        const std::string json = read_file(path);
        if (std::count(json.begin(), json.end(), '\n') != 2) return false;
    }
    for (const auto fault : {testing::LogFault::write, testing::LogFault::format}) {
        CycleLog output;
        output.set_buffered(true);
        const auto path = root / (fault == testing::LogFault::write ? "queued-write.jsonl" : "queued-format.jsonl");
        if (!output.set_output_path(path.c_str())) return false;
        output.write_json("{\"queued\":1}");
        if (!output.healthy() || !read_file(path).empty()) return false;
        testing::set_log_fault(fault);
        const bool flushed = output.flush();
        testing::clear_log_fault();
        if (flushed || output.healthy() || output.flush()) return false;
    }
    {
        CycleLog output;
        output.set_buffered(true);
        const auto old_path = root / "failed-route-old.jsonl";
        const auto new_path = root / "failed-route-new.jsonl";
        if (!output.set_output_path(old_path.c_str())) return false;
        output.write_json("{\"lost_on_route\":1}");
        testing::set_log_fault(testing::LogFault::format);
        const bool replaced = output.set_output_path(new_path.c_str());
        testing::clear_log_fault();
        if (replaced || output.healthy() || output.flush() || output.output_path() != old_path ||
            std::filesystem::exists(new_path)) return false;
        if (!output.set_output_path(new_path.c_str()) || !output.healthy()) return false;
        output.write_json("{\"recovered_route\":1}");
        if (!output.flush() || read_file(new_path) != "{\"recovered_route\":1}\n") return false;
    }
    const auto path = root / "queued-allocation.jsonl";
    if (!gemmini_log_cycle_set_output_path(path.c_str())) return false;
    gemmini_log_cycle_set_buffered(1);
    gemmini_cycle_record_v2 identity{};
    identity.interval.layer = "allocation";
    identity.interval.op = "dropped";
    const gemmini_cpu_sample start{}, end{};
    testing::set_log_fault(testing::LogFault::allocation);
    gemmini_cpu_timing_record(&identity, &start, &end);
    testing::clear_log_fault();
    const bool failed = !cycle.healthy() && !gemmini_log_cycle_flush();
    identity.interval.op = "after_allocation_failure";
    gemmini_cpu_timing_record(&identity, &start, &end);
    const bool still_failed = !gemmini_log_cycle_flush();
    gemmini_log_cycle_set_buffered(0);
    gemmini_log_cycle_set_output(stderr);
    const std::string json = read_file(path);
    if (!failed || !still_failed || json.find("dropped") != std::string::npos ||
        json.find("after_allocation_failure") == std::string::npos) return false;
#else
    (void) root;
#endif
    return true;
}

int main() {
    using ggml::gemmini::log::testing::LogFault;
    // Synthetic checked output, not a physical PMU reading on this host.
    const ggml::gemmini::log::CycleRecord checked_fixture{
        "fixture", "sample_failure", 0, 5000, nullptr, 0, nullptr,
        "linux_perf_cpu_cycles", "cycle"};
    const std::string checked_failure = ggml::gemmini::log::serialize_checked_cycle_record(
        checked_fixture, false, "invalid_start", "unavailable_event");
    const std::string checked_zero = ggml::gemmini::log::serialize_checked_cycle_record(
        {"fixture", "valid_zero", 0, 0, nullptr, 0, nullptr,
         "linux_perf_cpu_cycles", "cycle"}, true, nullptr);
    if (checked_failure.find(
            "\"delta\":null,\"valid\":false,\"reason\":\"invalid_start\","
            "\"sample_reason\":\"unavailable_event\"") == std::string::npos ||
        checked_zero.find("\"delta\":0,\"valid\":true") == std::string::npos ||
        checked_zero.find("\"sample_reason\"") != std::string::npos) {
        std::fprintf(stderr, "checked output must preserve sample failure and valid zero\n");
        return 19;
    }
    const std::string scalar = ggml::gemmini::log::serialize_cycle_record(
        {"scalar", "public", 10, 12, nullptr, 0, nullptr});
    const std::string legacy_equal = ggml::gemmini::log::serialize_cycle_record(
        {"scalar", "public.equal", 10, 10, nullptr, 0, nullptr});
    const std::string legacy_regression = ggml::gemmini::log::serialize_cycle_record(
        {"scalar", "public", 12, 10, nullptr, 0, nullptr});
#if EXPECT_CYCLE_DETAIL && (!defined(__linux__) || !defined(__aarch64__))
    const bool regression_matches = legacy_regression.find(
        "\"start\":12,\"end\":10,\"delta\":0,\"valid\":false") != std::string::npos &&
        legacy_regression.find("\"reason\"") == std::string::npos;
#else
    const bool regression_matches = legacy_regression.find(
        "\"start\":12,\"end\":10,\"delta\":null,\"valid\":false,\"reason\":\"counter_regression\"") !=
            std::string::npos;
#endif
    const std::string linux_monotonic =
        ggml::gemmini::log::testing::serialize_linux_aarch64_scalar_cycle_record_for_test(
            {"scalar", "linux.monotonic", 10, 12, nullptr, 0, nullptr});
    const std::string linux_equal =
        ggml::gemmini::log::testing::serialize_linux_aarch64_scalar_cycle_record_for_test(
            {"scalar", "linux.equal", 10, 10, nullptr, 0, nullptr});
    if (scalar.find("\"start\":10,\"end\":12,\"delta\":2,\"valid\":true") == std::string::npos ||
        legacy_equal.find("\"start\":10,\"end\":10,\"delta\":0,\"valid\":true") == std::string::npos ||
        scalar.find("scalar_provenance_unavailable") != std::string::npos ||
        !regression_matches ||
#if EXPECT_CYCLE_DETAIL
        linux_monotonic.find("\"source\":\"linux_perf_cpu_cycles\",\"unit\":\"cycle\"") == std::string::npos ||
#else
        linux_monotonic.find("{\"op\":\"linux.monotonic\",\"kind\":\"cycle\"") == std::string::npos ||
        linux_monotonic.find("\"source\"") != std::string::npos ||
#endif
        linux_monotonic.find("\"start\":10,\"end\":12,\"delta\":2,\"valid\":true") == std::string::npos ||
        linux_equal.find("\"start\":10,\"end\":10,\"delta\":0,\"valid\":true") == std::string::npos) {
        std::fprintf(stderr, "scalar records must retain platform arithmetic: %s",
                     legacy_regression.c_str());
        return 16;
    }

    struct ScalarCase {
        uint64_t start;
        uint64_t end;
        const char * reason;
        bool valid;
        uint64_t delta;
    };
    const ScalarCase scalar_cases[] = {
        {0, 0, "invalid_start", false, 0},
        {0, 7, "invalid_start", false, 0},
        {7, 0, "invalid_end", false, 0},
        {7, 6, "counter_regression", false, 0},
        {7, 7, nullptr, true, 0},
    };
    for (const ScalarCase & scalar_case : scalar_cases) {
        const std::string json =
            ggml::gemmini::log::testing::serialize_linux_aarch64_scalar_cycle_record_for_test(
                {"scalar", "linux.sentinel", scalar_case.start, scalar_case.end,
                 nullptr, 0, nullptr});
        const std::string interval = "\"start\":" + std::to_string(scalar_case.start) +
            ",\"end\":" + std::to_string(scalar_case.end) + ",\"delta\":" +
            (scalar_case.valid ? std::to_string(scalar_case.delta) : "null") +
            ",\"valid\":" + (scalar_case.valid ? "true" : "false");
        const bool reason_matches = scalar_case.reason
            ? json.find(std::string("\"reason\":\"") + scalar_case.reason + "\"") != std::string::npos
            : json.find("\"reason\"") == std::string::npos;
        if (json.find(interval) == std::string::npos || !reason_matches) {
            std::fprintf(stderr, "Linux-AArch64 scalar policy mismatch: %s", json.c_str());
            return 18;
        }
    }
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("gemmini-log-c-boundary-" + std::to_string(current_process_id()));
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directory(root, error);
    if (error) return 1;
    if (!test_buffered_cycle_output(root)) return 20;
    if (!test_hardware_cycle_summary(root)) return 21;
    if (!test_inference_log_context(root)) return 22;
    if (!test_worker_cycle_buffers(root)) return 23;
    if (!test_cycle_buffer_bytes(root)) return 24;
    if (!test_cycle_buffer_ownership(root)) return 25;
    if (!test_queued_cycle_faults(root)) return 26;
    const auto cycle_path = root / "cycle.jsonl";
    const auto debug_path = root / "debug.jsonl";
    const auto fault_path = root / "fault.jsonl";
    const auto targeted_path = root / "targeted.jsonl";
#if defined(__linux__) && defined(__aarch64__)
    const auto checked_path = root / "checked.jsonl";
    if (!checked_bridge_matrix(checked_path)) return 17;
#endif

    if (!gemmini_log_cycle_set_output_path(cycle_path.c_str()) ||
        !gemmini_log_debug_set_output_path(debug_path.c_str())) return 2;

#if EXPECT_LOG_CYCLE
    ggml::gemmini::log::testing::set_log_fault(LogFault::filesystem);
    if (gemmini_log_c_boundary_call(0, fault_path.c_str()) != 0) return 3;
#else
    if (!gemmini_log_c_boundary_call(0, fault_path.c_str())) return 3;
#endif

#if EXPECT_LOG_DEBUG
    ggml::gemmini::log::testing::set_log_fault(LogFault::format);
    if (!gemmini_log_c_boundary_call(1, nullptr)) return 4;
    ggml::gemmini::log::testing::set_log_fault(LogFault::mutex);
    if (!gemmini_log_c_boundary_call(2, nullptr)) return 5;
#else
    if (!gemmini_log_c_boundary_call(1, nullptr) || !gemmini_log_c_boundary_call(2, nullptr)) return 4;
#endif

#if EXPECT_LOG_CYCLE
    ggml::gemmini::log::testing::set_log_fault(LogFault::allocation);
    if (!gemmini_log_c_boundary_call(3, nullptr)) return 6;
#else
    if (!gemmini_log_c_boundary_call(3, nullptr)) return 6;
#endif

#if EXPECT_LOG_DEBUG
#if !defined(_WIN32)
    struct rlimit descriptor_limit {};
    if (getrlimit(RLIMIT_NOFILE, &descriptor_limit) != 0) return 7;
    descriptor_limit.rlim_cur = descriptor_limit.rlim_cur < 32 ? descriptor_limit.rlim_cur : 32;
    if (setrlimit(RLIMIT_NOFILE, &descriptor_limit) != 0) return 8;
#endif
    const int descriptors_before = open_descriptor_count();
    for (const LogFault fault : {LogFault::format, LogFault::allocation}) {
        for (int operation = 4; operation <= 6; ++operation) {
            for (int attempt = 0; attempt != 16; ++attempt) {
                ggml::gemmini::log::testing::set_log_fault(fault);
                if (!gemmini_log_c_boundary_call(operation, targeted_path.c_str())) return 9;
            }
            if (open_descriptor_count() != descriptors_before) return 10;
        }
    }
    for (int operation = 4; operation <= 6; ++operation) {
        if (!gemmini_log_c_boundary_call(operation, targeted_path.c_str())) return 11;
    }
#else
    for (int operation = 4; operation <= 6; ++operation) {
        if (!gemmini_log_c_boundary_call(operation, targeted_path.c_str())) return 9;
    }
#endif

    gemmini_log_debug("healthy-debug-boundary");
    gemmini_log_cycle("healthy-cycle-boundary", "after-fault", 10, 12);
    const gemmini_cycle_record_v2 scalar_v2{
        {"healthy-v2-boundary", "after-fault-v2", 20, 23, nullptr, 0, nullptr},
        GEMMINI_CYCLE_HAS_RUN_ID, 41, 0, 0, 0, 0};
    gemmini_log_cycle_record_v2(&scalar_v2);
    gemmini_log_debug_set_output(stderr);
    gemmini_log_cycle_set_output(stderr);

    const std::string debug = read_file(debug_path);
    const std::string cycle = read_file(cycle_path);
    bool ok = true;
#if EXPECT_LOG_DEBUG
    ok = ok && debug.find("c-format-boundary") == std::string::npos &&
        debug.find("c-mutex-boundary") == std::string::npos &&
        debug.find("healthy-debug-boundary") != std::string::npos &&
        debug.find("c-target-format") == std::string::npos &&
        read_file(targeted_path).find("c-target-loc-6") != std::string::npos;
#else
    ok = ok && !std::filesystem::exists(debug_path) && !std::filesystem::exists(targeted_path);
#endif
#if EXPECT_LOG_CYCLE
    ok = ok && cycle.find("WS_LOOP_TELEMETRY") == std::string::npos &&
        cycle.find("\"op\":\"after-fault\"") != std::string::npos &&
        cycle.find("\"start\":10,\"end\":12,\"delta\":2,\"valid\":true") != std::string::npos &&
        cycle.find("\"op\":\"after-fault-v2\"") != std::string::npos &&
        cycle.find("\"run_id\":41") != std::string::npos &&
        cycle.find("\"start\":20,\"end\":23,\"delta\":3,\"valid\":true") != std::string::npos &&
        cycle.find("scalar_provenance_unavailable") == std::string::npos &&
        !std::filesystem::exists(fault_path);
#else
    ok = ok && !std::filesystem::exists(cycle_path) && !std::filesystem::exists(fault_path);
#endif
    std::filesystem::remove_all(root, error);
    return ok && !error ? 0 : 7;
}
