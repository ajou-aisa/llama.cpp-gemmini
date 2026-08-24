#include <gemmini/log.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace {
using lease_fn = void (*)(void);
using set_output_path_fn = int (*)(const char *);
using ws_cycle_fn = void (*)(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t,
                             uint64_t, uint64_t, uint64_t,
                             uint64_t, uint64_t, uint64_t,
                             uint64_t, uint64_t, uint64_t,
                             uint64_t, uint64_t);

std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

template<typename Function>
Function load_symbol(void * image, const char * name) {
    dlerror();
    void * symbol = dlsym(image, name);
    if (const char * error = dlerror()) {
        std::cerr << "dlsym(" << name << ") failed: " << error << '\n';
        return nullptr;
    }
    return reinterpret_cast<Function>(symbol);
}
} // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: test-gemmini-log-cross-image BACKEND CYCLE_JSONL STDERR_CAPTURE\n";
        return 1;
    }

    const std::filesystem::path cycle_path = std::filesystem::absolute(argv[2]);
    const std::filesystem::path stderr_path = std::filesystem::absolute(argv[3]);
    std::error_code error;
    std::filesystem::remove(cycle_path, error);
    std::filesystem::remove(stderr_path, error);

    void * image = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!image) {
        std::cerr << "dlopen failed: " << dlerror() << '\n';
        return 2;
    }

    const auto image_acquire = load_symbol<lease_fn>(image, "gemmini_hardware_counter_lease_acquire");
    const auto image_release = load_symbol<lease_fn>(image, "gemmini_hardware_counter_lease_release");
    const auto image_emit = load_symbol<ws_cycle_fn>(image, "gemmini_log_ws_cycle");
    const auto image_set_output = load_symbol<set_output_path_fn>(image, "gemmini_log_cycle_set_output_path");
    if (!image_acquire || !image_release || !image_emit || !image_set_output) {
        dlclose(image);
        return 2;
    }

    if (reinterpret_cast<void *>(image_acquire) !=
            reinterpret_cast<void *>(&gemmini_hardware_counter_lease_acquire) ||
        reinterpret_cast<void *>(image_emit) != reinterpret_cast<void *>(&gemmini_log_ws_cycle) ||
        reinterpret_cast<void *>(image_set_output) !=
            reinterpret_cast<void *>(&gemmini_log_cycle_set_output_path)) {
        std::cerr << "backend and executable resolved distinct Gemmini log/counter state images\n";
        dlclose(image);
        return 3;
    }

    gemmini_hardware_counter_lease_acquire();
    gemmini_hardware_counter_lease_release();
    image_acquire();
    image_release();

    if (!gemmini_log_cycle_set_output_path(cycle_path.c_str())) {
        std::cerr << "could not select cross-image cycle path\n";
        dlclose(image);
        return 4;
    }

    std::fflush(stderr);
    const int saved_stderr = dup(STDERR_FILENO);
    const int captured_stderr = open(stderr_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (saved_stderr < 0 || captured_stderr < 0 || dup2(captured_stderr, STDERR_FILENO) < 0) {
        std::cerr << "could not redirect stderr\n";
        if (captured_stderr >= 0) close(captured_stderr);
        if (saved_stderr >= 0) close(saved_stderr);
        dlclose(image);
        return 4;
    }
    close(captured_stderr);

    image_emit(100, 10, 20, 30, 40, 256, 768, 768, 5, 3, 6, 4, 29, 1, 0, 1);
    std::fflush(stderr);
    const bool restore_ok = dup2(saved_stderr, STDERR_FILENO) >= 0;
    close(saved_stderr);
    gemmini_log_cycle_set_output(stderr);

    const std::string cycle_output = read_file(cycle_path);
    const std::string stderr_output = read_file(stderr_path);
    dlclose(image);
    if (!restore_ok || !stderr_output.empty() ||
        cycle_output.find("\"record_type\":\"WS_LOOP_TELEMETRY\"") == std::string::npos ||
        cycle_output.find("\"valid\":true") == std::string::npos ||
        cycle_output.find('\n') == std::string::npos) {
        std::cerr << "cross-image CycleLog routing failed: cycle_bytes=" << cycle_output.size()
                  << " stderr_bytes=" << stderr_output.size() << '\n';
        return 5;
    }
    return 0;
}
