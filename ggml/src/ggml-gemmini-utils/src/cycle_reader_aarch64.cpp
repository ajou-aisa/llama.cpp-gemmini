#include "../include/gemmini/cycle_reader.hpp"

#if defined(__linux__) && defined(__aarch64__)

#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace ggml::gemmini::cycle
{
namespace
{
    constexpr uint64_t kDirectReadConfig = uint64_t{1} << 1;
    constexpr uint32_t kPmccntrMetadataIndex = 32;
    constexpr unsigned int kSeqlockAttempts = 16;
    std::atomic<uint64_t> next_generation{1};

    struct EventOwner
    {
        int fd = -1;
        perf_event_mmap_page * page = nullptr;
        size_t page_size = 0;
        NativeCycleReason failure = NativeCycleReason::unavailable_event;
        uint64_t owner_event_token = static_cast<uint64_t>(syscall(SYS_gettid));
        uint64_t generation = next_generation.fetch_add(1, std::memory_order_relaxed);

        EventOwner() noexcept
        {
            perf_event_attr attributes{};
            attributes.type = PERF_TYPE_HARDWARE;
            attributes.size = sizeof(attributes);
            attributes.config = PERF_COUNT_HW_CPU_CYCLES;
            attributes.config1 = kDirectReadConfig;
            attributes.disabled = 0;
            attributes.inherit = 0;
            attributes.pinned = 0;
            attributes.exclude_user = 0;
            attributes.exclude_kernel = 0;
            attributes.exclude_hv = 0;
            attributes.exclude_guest = 0;

            fd = static_cast<int>(syscall(SYS_perf_event_open, &attributes, 0, -1, -1, 0));
            if (fd < 0)
            {
                return;
            }
            const long system_page_size = sysconf(_SC_PAGESIZE);
            if (system_page_size <= 0)
            {
                close(fd);
                fd = -1;
                failure = NativeCycleReason::unavailable_direct_mapping;
                return;
            }
            page_size = static_cast<size_t>(system_page_size);
            void * const mapping = mmap(nullptr, page_size, PROT_READ, MAP_SHARED, fd, 0);
            if (mapping == MAP_FAILED)
            {
                close(fd);
                fd = -1;
                failure = NativeCycleReason::unavailable_direct_mapping;
                return;
            }
            page = static_cast<perf_event_mmap_page *>(mapping);
            failure = NativeCycleReason::none;
        }

        ~EventOwner()
        {
            if (page != nullptr)
            {
                munmap(page, page_size);
            }
            if (fd >= 0)
            {
                close(fd);
            }
        }

        EventOwner(const EventOwner &) = delete;
        EventOwner & operator=(const EventOwner &) = delete;
    };

    EventOwner & current_event() noexcept
    {
        thread_local EventOwner owner;
        return owner;
    }

    NativeCycleSample invalid_sample(const EventOwner & owner, NativeCycleReason reason) noexcept
    {
        return NativeCycleSample{0, false, reason, NativeCycleSource::perf_cpu_cycles,
                                 owner.owner_event_token, owner.generation};
    }

    struct CounterSnapshot
    {
        bool cap_user_rdpmc = false;
        uint32_t index = 0;
        int64_t offset = 0;
        uint16_t width = 0;
        uint64_t enabled = 0;
        uint64_t running = 0;
        uint64_t raw = 0;
        bool exhausted = false;
    };

    NativeCycleSample sample_from_snapshot(CounterSnapshot snapshot, uint64_t owner_token,
                                           uint64_t generation) noexcept
    {
        NativeCycleSample sample{0, false, NativeCycleReason::unavailable_direct_mapping,
                                 NativeCycleSource::perf_cpu_cycles, owner_token, generation};
        if (snapshot.exhausted)
        {
            sample.reason = NativeCycleReason::seqlock_exhausted;
            return sample;
        }
        if (snapshot.enabled != snapshot.running)
        {
            sample.reason = NativeCycleReason::multiplexed;
            return sample;
        }
        if (!snapshot.cap_user_rdpmc || snapshot.index != kPmccntrMetadataIndex ||
            snapshot.width == 0 || snapshot.width > 64 || snapshot.enabled == 0)
        {
            return sample;
        }
        if (snapshot.width < 64)
        {
            const uint64_t mask = (uint64_t{1} << snapshot.width) - 1;
            const uint64_t sign_bit = uint64_t{1} << (snapshot.width - 1);
            snapshot.raw &= mask;
            if ((snapshot.raw & sign_bit) != 0)
            {
                snapshot.raw |= ~mask;
            }
        }
        sample.value = static_cast<uint64_t>(snapshot.offset) + snapshot.raw;
        sample.valid = true;
        sample.reason = NativeCycleReason::none;
        return sample;
    }
}

NativeCycleSample read_sample() noexcept
{
    EventOwner & owner = current_event();
    if (owner.page == nullptr)
    {
        return invalid_sample(owner, owner.failure);
    }

    for (unsigned int attempt = 0; attempt < kSeqlockAttempts; ++attempt)
    {
        const uint32_t sequence = __atomic_load_n(&owner.page->lock, __ATOMIC_ACQUIRE);
        if ((sequence & 1U) != 0)
        {
            continue;
        }
        CounterSnapshot snapshot{owner.page->cap_user_rdpmc != 0, owner.page->index,
                                 owner.page->offset, owner.page->pmc_width,
                                 owner.page->time_enabled, owner.page->time_running, 0, false};
        const bool direct_mapping = snapshot.cap_user_rdpmc &&
            snapshot.index == kPmccntrMetadataIndex && snapshot.width > 0 &&
            snapshot.width <= 64 && snapshot.enabled != 0 && snapshot.enabled == snapshot.running;
        if (direct_mapping)
        {
            asm volatile("mrs %0, pmccntr_el0" : "=r"(snapshot.raw));
        }
        const uint32_t final_sequence = __atomic_load_n(&owner.page->lock, __ATOMIC_ACQUIRE);
        if (sequence == final_sequence)
        {
            return sample_from_snapshot(snapshot, owner.owner_event_token, owner.generation);
        }
    }
    return invalid_sample(owner, NativeCycleReason::seqlock_exhausted);
}

#if defined(GGML_GEMMINI_TESTING)
namespace testing
{
NativeCycleSample sample_from_input(const DirectReadInput & input) noexcept
{
    const CounterSnapshot snapshot{input.cap_user_rdpmc, input.index, input.offset,
                                   input.pmc_width, input.time_enabled, input.time_running,
                                   input.raw_value, input.seqlock_exhausted};
    return sample_from_snapshot(snapshot, input.owner_event_token, input.generation);
}

int event_fd() noexcept
{
    return current_event().fd;
}
}
#endif
}

#endif
