#pragma once

#include <cstdint>
#include <sched.h>
#include <pthread.h>
#include <sys/resource.h>
#include <time.h>

namespace potal::timeline {

inline bool valid_cycle_interval(uint64_t begin, uint64_t end, int cpu_begin, int cpu_end,
                                 bool pinned_begin, bool pinned_end) noexcept {
    return pinned_begin && pinned_end && cpu_begin >= 0 && cpu_begin == cpu_end &&
           begin != 0 && end >= begin;
}

inline bool pinned_to_cpu(int cpu) noexcept {
    if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    return pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask) == 0 &&
           CPU_COUNT(&mask) == 1 && CPU_ISSET(cpu, &mask);
}
// Raw hart counters are not thread PMU counters. Zero is never a valid fallback.
struct cpu_sample {
    uint64_t mono_ns = 0;
    uint64_t thread_ns = 0;
    uint64_t process_ns = 0;
    uint64_t cycles = 0;
    uint64_t instructions = 0;
    int cpu = -1;
    bool thread_valid = false;
    bool process_valid = false;
    bool counters_valid = false;
};

inline uint64_t clock_ns(clockid_t clock, bool & valid) noexcept {
    timespec ts{};
    valid = clock_gettime(clock, &ts) == 0;
    return valid ? uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec) : 0;
}

inline cpu_sample sample_cpu(int pinned_cpu, bool model_boundary) noexcept {
    cpu_sample out;
    const int before = sched_getcpu();
    out.thread_ns = clock_ns(CLOCK_THREAD_CPUTIME_ID, out.thread_valid);
    if (model_boundary) {
        rusage usage{};
        out.process_valid = getrusage(RUSAGE_SELF, &usage) == 0;
        if (out.process_valid) {
            out.process_ns = (uint64_t(usage.ru_utime.tv_sec) + uint64_t(usage.ru_stime.tv_sec)) * 1000000000ull
                           + (uint64_t(usage.ru_utime.tv_usec) + uint64_t(usage.ru_stime.tv_usec)) * 1000ull;
        }
    }
    bool mono_valid;
    out.mono_ns = clock_ns(CLOCK_MONOTONIC, mono_valid);
#if defined(__riscv) && __riscv_xlen == 64
    // Only opt-in, verified single-CPU threads read CSRs. The OS must grant U-mode access.
    if (pinned_cpu >= 0 && before == pinned_cpu) {
        __asm__ volatile("rdcycle %0\n\trdinstret %1" : "=r"(out.cycles), "=r"(out.instructions) : : "memory");
        out.counters_valid = true;
    }
#endif
    out.cpu = sched_getcpu();
    out.counters_valid = out.counters_valid && mono_valid && out.cpu == before && out.cpu == pinned_cpu;
    return out;
}

}  // namespace potal::timeline
