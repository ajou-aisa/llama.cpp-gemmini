#include <gemmini/cycle_reader.h>
#include <gemmini/log.hpp>
#if defined(__linux__) && defined(__aarch64__)
#include "cycle_reader_internal.h"
#endif

static_assert(noexcept(gemmini_read_cycles()));

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

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
    gemmini_log_cycle_set_output(stderr);

    const std::string output = read_file(path);
    const char * reasons[] = {"invalid_start", "invalid_end", "source_mismatch",
        "event_owner_mismatch", "event_generation_mismatch", "structurally_cross_task",
        "counter_regression"};
    if (output.find("\"source\":\"linux_perf_cpu_cycles\",\"unit\":\"cycle\"") == std::string::npos ||
        output.find("\"delta\":2,\"valid\":true") == std::string::npos ||
        output.find("\"delta\":0,\"valid\":true") == std::string::npos) return false;
    for (const char * reason : reasons) {
        if (output.find(std::string("\"delta\":null,\"valid\":false,\"reason\":\"") + reason + "\"") ==
            std::string::npos) return false;
    }
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

int main() {
    using ggml::gemmini::log::testing::LogFault;
#if defined(__linux__) && defined(__aarch64__)
    const std::string jetson_scalar = ggml::gemmini::log::serialize_cycle_record(
        {"scalar", "public", 10, 12, nullptr, 0, nullptr,
         "linux_perf_cpu_cycles", "cycle"});
    if (jetson_scalar.find("\"delta\":null,\"valid\":false,\"reason\":\"scalar_provenance_unavailable\"") ==
        std::string::npos) {
        std::fprintf(stderr, "RED: Jetson scalar provenance must fail closed\n");
        return 16;
    }
#else
    const std::string legacy_regression = ggml::gemmini::log::serialize_cycle_record(
        {"scalar", "public", 12, 10, nullptr, 0, nullptr});
    if (legacy_regression.find("\"start\":12,\"end\":10,\"delta\":0,\"valid\":false") ==
            std::string::npos || legacy_regression.find("\"reason\"") != std::string::npos) {
        return 16;
    }
#endif
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("gemmini-log-c-boundary-" + std::to_string(current_process_id()));
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directory(root, error);
    if (error) return 1;
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
        cycle.find("healthy-cycle-boundary") != std::string::npos &&
#if defined(__linux__) && defined(__aarch64__)
        cycle.find("\"delta\":null,\"valid\":false,\"reason\":\"scalar_provenance_unavailable\"") != std::string::npos &&
#else
        cycle.find("\"start\":10,\"end\":12,\"delta\":2,\"valid\":true") != std::string::npos &&
#endif
        !std::filesystem::exists(fault_path);
#else
    ok = ok && !std::filesystem::exists(cycle_path) && !std::filesystem::exists(fault_path);
#endif
    std::filesystem::remove_all(root, error);
    return ok && !error ? 0 : 7;
}
