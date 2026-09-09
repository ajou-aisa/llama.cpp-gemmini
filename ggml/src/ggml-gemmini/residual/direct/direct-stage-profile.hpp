#pragma once

#include <gemmini/host-timing.hpp>
#if defined(__linux__) && defined(__aarch64__)
#include <gemmini/cycle_reader.hpp>
#endif

#include <array>
#include <limits>
#include <ostream>

namespace ggml::gemmini::residual::detail {

struct DirectStageTotals {
    uint64_t calls = 0;
    uint64_t wall_ns = 0;
    uint64_t thread_cpu_ns = 0;
    uint64_t cycles = 0;
    bool wall_valid = true;
    bool cpu_valid = true;
    bool cycles_valid = true;
    const char * cycles_reason = "none";

    static bool accumulate(uint64_t value, uint64_t & sum) noexcept {
        if (value > std::numeric_limits<uint64_t>::max() - sum) return false;
        sum += value;
        return true;
    }

    void write_json(std::ostream & out) const {
        out << "{\"calls\":" << calls << ",\"wall_ns\":";
        if (calls != 0 && wall_valid) out << wall_ns; else out << "null";
        out << ",\"thread_cpu_ns\":";
        if (calls != 0 && cpu_valid) out << thread_cpu_ns; else out << "null";
        out << ",\"cycles\":";
        if (calls != 0 && cycles_valid) out << cycles; else out << "null";
        out << ",\"cycles_valid\":" << (calls != 0 && cycles_valid ? "true" : "false")
            << ",\"cycles_reason\":\"" << (calls == 0 ? "no_samples" : cycles_reason) << "\"}";
    }
};

class DirectStageProbe {
public:
    explicit DirectStageProbe(std::array<DirectStageTotals, 3> * stages) noexcept : stages_(stages) {
        begin();
    }
    ~DirectStageProbe() noexcept { finish(); }
    DirectStageProbe(const DirectStageProbe &) = delete;
    DirectStageProbe & operator=(const DirectStageProbe &) = delete;

    void next(size_t index) noexcept {
        finish();
        index_ = index;
        begin();
    }

private:
    void begin() noexcept {
        if (stages_ == nullptr) return;
        host_start_ = cycle::read_host_sample();
#if defined(__linux__) && defined(__aarch64__)
        cycle_start_ = cycle::read_sample();
#endif
    }

    void finish() noexcept {
        if (stages_ == nullptr) return;
#if defined(__linux__) && defined(__aarch64__)
        const auto cycle_end = cycle::read_sample();
#endif
        const auto host_end = cycle::read_host_sample();
        auto & stage = (*stages_)[index_];
        ++stage.calls;
        const bool same_thread = host_start_.tid != 0 && host_start_.tid == host_end.tid;
        stage.wall_valid = stage.wall_valid && same_thread && host_end.ns >= host_start_.ns &&
            DirectStageTotals::accumulate(host_end.ns - host_start_.ns, stage.wall_ns);
        stage.cpu_valid = stage.cpu_valid && same_thread && host_start_.thread_cpu_valid &&
            host_end.thread_cpu_valid && host_end.thread_cpu_ns >= host_start_.thread_cpu_ns &&
            DirectStageTotals::accumulate(host_end.thread_cpu_ns - host_start_.thread_cpu_ns,
                                          stage.thread_cpu_ns);
#if defined(__linux__) && defined(__aarch64__)
        const auto interval = cycle::evaluate_interval(cycle_start_, cycle_end, same_thread);
        if (stage.cycles_valid) {
            if (!interval.valid) {
                stage.cycles_valid = false;
                stage.cycles_reason = cycle::reason_name(
                    interval.sample_reason != cycle::NativeCycleReason::none ?
                        interval.sample_reason : interval.reason);
            } else if (!DirectStageTotals::accumulate(interval.value, stage.cycles)) {
                stage.cycles_valid = false;
                stage.cycles_reason = "accumulator_overflow";
            }
        }
#else
        stage.cycles_valid = false;
        stage.cycles_reason = "unsupported_platform";
#endif
    }

    std::array<DirectStageTotals, 3> * stages_;
    size_t index_ = 0;
    cycle::HostSample host_start_{};
#if defined(__linux__) && defined(__aarch64__)
    cycle::NativeCycleSample cycle_start_{};
#endif
};

}
