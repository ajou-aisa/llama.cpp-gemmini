#pragma once

#include <cstdint>
#include <string>
#include <optional>
#include "cpu-timing.h"

namespace ggml::gemmini::cycle {

struct HostSample {
    uint64_t ns = 0;
    uint64_t tid = 0;
    uint64_t thread_cpu_ns = 0;
    bool thread_cpu_valid = false;
};

uint64_t host_thread_id() noexcept;
const std::string &host_execution_id();
HostSample read_host_sample() noexcept;
std::string serialize_host_timing(uint64_t start_ns, uint64_t end_ns,
                                  uint64_t start_tid, uint64_t end_tid);
std::string serialize_thread_cpu_timing(const HostSample &start, const HostSample &end);
std::string serialize_cpu_totals(const gemmini_cpu_totals &totals);
std::string serialize_cpu_native(const gemmini_cpu_sample &start, const gemmini_cpu_sample &end);

struct WorkerCpuTiming {
    gemmini_cpu_sample start{}, end{};
    // Capture on the submitting caller; begin/end callbacks run on the owner.
    gemmini_trace_context origin = gemmini_trace_capture();
    gemmini_trace_context previous_context{};
    bool started = false, finished = false;

    static void observe(void *context, bool begin) noexcept;
    void emit(const char *layer, const char *scope, std::optional<uint64_t> run_id, bool success) const noexcept;
};

}
