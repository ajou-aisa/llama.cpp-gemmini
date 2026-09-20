#pragma once

#include "cpu-timing.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>

namespace ggml::gemmini::performance {

enum class Phase { prefill, decode };

constexpr const char * phase_name(Phase phase) noexcept {
    switch (phase) {
        case Phase::prefill: return "prefill";
        case Phase::decode:  return "decode";
    }
    return "unknown";
}

struct Context {
    uint64_t request_id = 0;
    uint64_t operation_id = 0;
    Phase phase = Phase::prefill;
};

Context capture_context() noexcept;
std::string serialize_context(Context context);
uint64_t next_cpu_interval_sequence(Context context) noexcept;

struct Measurement {
    enum class Kind { cpu, wall, wall_gap, npu };
    Kind kind = Kind::cpu;
    uint64_t sequence = 0;
    gemmini_cpu_sample start{}, end{};
    uint64_t cycles = 0, thread_ns = 0;
    std::string cycles_reason, thread_reason;
    uint64_t begin_ns = 0, end_ns = 0;
    std::string backend, domain, metric, reason;
    bool collected = false;
    gemmini_trace_context trace{};
};

std::string serialize_measurement(const Measurement & measurement);

class Summary {
public:
    bool available = false;
    std::string reason;
    size_t peak_operation_samples = 0;
    std::string serialize() const;
    void print(FILE * output) const;

private:
    std::string json_, text_;
    friend Summary read_summary(std::istream & input);
};

Summary read_summary(const std::filesystem::path & path);
Summary read_summary(std::istream & input);

void reset();
void finish_recording();
void start_request(uint64_t start_ns);
void begin_operation(Phase phase, uint64_t start_ns);
void end_operation(uint64_t end_ns, bool success);
void token_ready(uint64_t ready_ns, std::optional<int32_t> token_id = {});
void finish_request(uint64_t end_ns);

void record_cpu(const gemmini_cpu_sample & start, const gemmini_cpu_sample & end,
                const gemmini_cpu_totals & interval) noexcept;
void record_cpu_wall(uint64_t begin_ns, uint64_t end_ns) noexcept;
void incomplete_cpu_wall(const char * reason) noexcept;
void record_npu(const char * backend, const char * domain, const char * metric,
                uint64_t cycles, bool valid, const char * reason) noexcept;
void set_npu_frequency(uint64_t hz);

std::string log_context();
std::string serialize();

}
