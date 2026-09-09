#pragma once

#include <cstdint>
#include <string>

namespace ggml::gemmini::cycle {

struct HostSample {
    uint64_t ns = 0;
    uint64_t tid = 0;
    uint64_t thread_cpu_ns = 0;
    bool thread_cpu_valid = false;
};

uint64_t host_thread_id() noexcept;
HostSample read_host_sample() noexcept;
std::string serialize_host_timing(uint64_t start_ns, uint64_t end_ns,
                                  uint64_t start_tid, uint64_t end_tid);
std::string serialize_thread_cpu_timing(const HostSample &start, const HostSample &end);

}
