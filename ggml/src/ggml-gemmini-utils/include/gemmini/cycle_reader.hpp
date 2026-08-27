// header-only except for the Linux AArch64 native reader
#pragma once
#include <stdint.h>
#include <atomic>
#include <chrono>

#if defined(__APPLE__) && defined(__aarch64__)
#include <mach/mach_time.h>
#endif

namespace ggml::gemmini::cycle
{
    // Process-wide test seam covering every native cycle and profiling timestamp read.
    inline std::atomic<uint64_t> read_count{0};

    enum class NativeCycleSource : uint8_t
    {
        none,
        riscv_cycle,
        apple_host_tick,
        perf_cpu_cycles,
        steady_clock,
    };

    enum class NativeCycleReason : uint8_t
    {
        none,
        unavailable_event,
        unavailable_direct_mapping,
        multiplexed,
        seqlock_exhausted,
        invalid_start,
        invalid_end,
        source_mismatch,
        event_owner_mismatch,
        event_generation_mismatch,
        structurally_cross_task,
        counter_regression,
        scalar_provenance_unavailable,
    };

    struct NativeCycleSample
    {
        uint64_t value = 0;
        bool valid = false;
        NativeCycleReason reason = NativeCycleReason::unavailable_event;
        NativeCycleSource source = NativeCycleSource::none;
        uint64_t owner_event_token = 0;
        uint64_t generation = 0;
    };

    struct NativeCycleDelta
    {
        uint64_t value = 0;
        bool valid = false;
        NativeCycleReason reason = NativeCycleReason::none;
        NativeCycleReason sample_reason = NativeCycleReason::none;
    };

#if defined(__linux__) && defined(__aarch64__)
    NativeCycleSample read_sample() noexcept;

#if defined(GGML_GEMMINI_TESTING)
    namespace testing
    {
        struct DirectReadInput
        {
            bool cap_user_rdpmc = true;
            uint32_t index = 32;
            int64_t offset = 0;
            uint16_t pmc_width = 32;
            uint64_t time_enabled = 1;
            uint64_t time_running = 1;
            uint64_t raw_value = 0;
            uint64_t owner_event_token = 1;
            uint64_t generation = 1;
            bool seqlock_exhausted = false;
        };

        NativeCycleSample sample_from_input(const DirectReadInput & input) noexcept;
        int event_fd() noexcept;
    }
#endif
#else
    static inline NativeCycleSample read_sample() noexcept
    {
#if !LOG_CYCLE
        return {};
#elif defined(__riscv)
        uint64_t value;
        asm volatile("rdcycle %0" : "=r"(value) :: "memory");
        return {value, true, NativeCycleReason::none, NativeCycleSource::riscv_cycle, 1, 1};
#elif defined(__APPLE__) && defined(__aarch64__)
        return {mach_absolute_time(), true, NativeCycleReason::none,
                NativeCycleSource::apple_host_tick, 1, 1};
#else
        return {static_cast<uint64_t>(
                    std::chrono::steady_clock::now().time_since_epoch().count()),
                true, NativeCycleReason::none, NativeCycleSource::steady_clock, 1, 1};
#endif
    }
#endif

    static inline NativeCycleDelta evaluate_interval(
            const NativeCycleSample & start, const NativeCycleSample & end,
            bool structurally_same_owner_eligible = true) noexcept
    {
        if (!start.valid) return {0, false, NativeCycleReason::invalid_start, start.reason};
        if (!end.valid) return {0, false, NativeCycleReason::invalid_end, end.reason};
        if (start.source != end.source) return {0, false, NativeCycleReason::source_mismatch, NativeCycleReason::none};
        if (start.owner_event_token != end.owner_event_token)
            return {0, false, NativeCycleReason::event_owner_mismatch, NativeCycleReason::none};
        if (start.generation != end.generation)
            return {0, false, NativeCycleReason::event_generation_mismatch, NativeCycleReason::none};
        if (!structurally_same_owner_eligible)
            return {0, false, NativeCycleReason::structurally_cross_task, NativeCycleReason::none};
        if (end.value < start.value)
            return {0, false, NativeCycleReason::counter_regression, NativeCycleReason::none};
        return {end.value - start.value, true, NativeCycleReason::none, NativeCycleReason::none};
    }

    static inline const char * reason_name(NativeCycleReason reason) noexcept
    {
        switch (reason)
        {
            case NativeCycleReason::none: return "none";
            case NativeCycleReason::unavailable_event: return "unavailable_event";
            case NativeCycleReason::unavailable_direct_mapping: return "unavailable_direct_mapping";
            case NativeCycleReason::multiplexed: return "multiplexed";
            case NativeCycleReason::seqlock_exhausted: return "seqlock_exhausted";
            case NativeCycleReason::invalid_start: return "invalid_start";
            case NativeCycleReason::invalid_end: return "invalid_end";
            case NativeCycleReason::source_mismatch: return "source_mismatch";
            case NativeCycleReason::event_owner_mismatch: return "event_owner_mismatch";
            case NativeCycleReason::event_generation_mismatch: return "event_generation_mismatch";
            case NativeCycleReason::structurally_cross_task: return "structurally_cross_task";
            case NativeCycleReason::counter_regression: return "counter_regression";
            case NativeCycleReason::scalar_provenance_unavailable: return "scalar_provenance_unavailable";
        }
        return "unknown";
    }

    static inline uint64_t read()
    {
#if !LOG_CYCLE
        return 0;
#else
        read_count.fetch_add(1, std::memory_order_relaxed);
        std::atomic_signal_fence(std::memory_order_seq_cst);
        const NativeCycleSample sample = read_sample();
        const uint64_t value = sample.valid ? sample.value : 0;
        std::atomic_signal_fence(std::memory_order_seq_cst);
        return value;
#endif
    }

    static inline uint64_t timestamp_ns()
    {
#if !LOG_CYCLE
        return 0;
#else
        read_count.fetch_add(1, std::memory_order_relaxed);
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
    }

    static inline void reset_read_count_for_test() { read_count.store(0, std::memory_order_relaxed); }
    static inline uint64_t read_count_for_test() { return read_count.load(std::memory_order_relaxed); }

    static inline const char * clock_mode()
    {
#if defined(__riscv) || (defined(__linux__) && defined(__aarch64__))
        return "CYCLE";
#else
        return "TIMER";
#endif
    }

    static inline const char * units()
    {
#if defined(__riscv) || (defined(__linux__) && defined(__aarch64__))
        return "cycles";
#else
        return "ticks";
#endif
    }

    static inline uint64_t resolution() { return 1; }
}
