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

#if defined(__linux__) && defined(__aarch64__)
    enum class NativeCycleSource : uint8_t
    {
        perf_cpu_cycles,
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
    };

    struct NativeCycleSample
    {
        uint64_t value = 0;
        bool valid = false;
        NativeCycleReason reason = NativeCycleReason::unavailable_event;
        NativeCycleSource source = NativeCycleSource::perf_cpu_cycles;
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

    NativeCycleSample read_sample() noexcept;
    NativeCycleDelta evaluate_interval(const NativeCycleSample & start,
                                       const NativeCycleSample & end,
                                       bool structurally_same_owner_eligible = true) noexcept;
    const char * reason_name(NativeCycleReason reason) noexcept;

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
#endif

    static inline uint64_t read()
    {
#if !LOG_CYCLE
        return 0;
#else
        read_count.fetch_add(1, std::memory_order_relaxed);
        std::atomic_signal_fence(std::memory_order_seq_cst);
#ifdef __riscv
        uint64_t value;
        asm volatile("rdcycle %0" : "=r"(value) :: "memory");
#elif defined(__APPLE__) && defined(__aarch64__)
        const uint64_t value = mach_absolute_time();
#elif defined(__linux__) && defined(__aarch64__)
        const NativeCycleSample sample = read_sample();
        const uint64_t value = sample.valid ? sample.value : 0;
#else
        const uint64_t value = static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
#endif
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
