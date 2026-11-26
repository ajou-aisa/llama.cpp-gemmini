#pragma once
// Runs a PB test in five-core mode when GGML_GEMMINI_PB_TEST_FIVE_CORE="ISSUER_CPU:HELPER,HELPER,..." is set.
// Unset, the test runs the single-thread path as before. The calling thread is the coordinator lane.
#include "../pb/pb-exec.hpp"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

inline bool pb_test_start_five_core() {
    const char * spec = std::getenv("GGML_GEMMINI_PB_TEST_FIVE_CORE");
    if (!spec || !*spec) return false;
    const std::string s(spec);
    const size_t colon = s.find(':');
    ggml::gemmini::pb::RuntimeConfig config;
    try {
        config.issuer_cpu = std::stoi(s.substr(0, colon));
        std::stringstream in(colon == std::string::npos ? "" : s.substr(colon + 1));
        for (std::string item; std::getline(in, item, ',');) config.helper_cpus.push_back(std::stoi(item));
    } catch (const std::exception &) {
        std::fprintf(stderr, "bad GGML_GEMMINI_PB_TEST_FIVE_CORE: %s\n", spec);
        std::exit(3);
    }
    const auto status = ggml::gemmini::pb::start_runtime(config);
    if (status != ggml::gemmini::pb::Status::success) {
        std::fprintf(stderr, "PB five-core runtime failed to start (status=%d)\n", static_cast<int>(status));
        std::exit(3);
    }
    std::printf("PB_FIVE_CORE_TEST start issuer_cpu=%d helpers=%zu\n", config.issuer_cpu, config.helper_cpus.size());
    return true;
}

// Prints one line per thread and stops the runtime. Exits non-zero when a lane never worked or ran off its CPU.
inline void pb_test_finish_five_core(bool active) {
    if (!active) return;
    bool ok = true;
    for (const auto & t : ggml::gemmini::pb::runtime_report()) {
        const bool issuer = t.lane < 0;
        const bool worked = issuer ? t.gemm_calls > 0 : t.phases > 0 && t.zeroed + t.scaled_elements + t.packed_b_bytes > 0;
        ok = ok && worked && t.cpu_mismatch == 0;
        std::printf("PB_FIVE_CORE_TEST thread role=%s lane=%d cpu=%d/%d/%d mismatch=%llu sessions=%llu phases=%llu gemm=%llu "
                    "flushes=%llu zeroed=%llu scaled=%llu packed_a=%llu packed_b=%llu quant_rows=%llu\n",
                    t.role, t.lane, t.cpu_assigned, t.cpu_first, t.cpu_last, (unsigned long long) t.cpu_mismatch,
                    (unsigned long long) t.sessions, (unsigned long long) t.phases, (unsigned long long) t.gemm_calls,
                    (unsigned long long) t.flushes, (unsigned long long) t.zeroed, (unsigned long long) t.scaled_elements,
                    (unsigned long long) t.packed_a_bytes, (unsigned long long) t.packed_b_bytes,
                    (unsigned long long) t.quantized_rows);
    }
    ggml::gemmini::pb::stop_runtime();
    if (!ok) {
        std::printf("PB_FIVE_CORE_TEST FAIL: a lane did no work or ran off its CPU\n");
        std::exit(4);
    }
}
