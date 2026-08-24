#include <gemmini/cycle_reader.h>
#include <gemmini/log.hpp>

static_assert(noexcept(gemmini_read_cycles()));

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/resource.h>
#endif

extern "C" int gemmini_log_c_boundary_call(int operation, const char * path);

static std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

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
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "gemmini-log-c-boundary";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::create_directory(root, error);
    if (error) return 1;
    const auto cycle_path = root / "cycle.jsonl";
    const auto debug_path = root / "debug.jsonl";
    const auto fault_path = root / "fault.jsonl";
    const auto targeted_path = root / "targeted.jsonl";

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
        !std::filesystem::exists(fault_path);
#else
    ok = ok && !std::filesystem::exists(cycle_path) && !std::filesystem::exists(fault_path);
#endif
    std::filesystem::remove_all(root, error);
    return ok && !error ? 0 : 7;
}
