#include <gemmini/cycle_reader.h>
#include <gemmini/cycle_reader.hpp>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sched.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

namespace cycle = ggml::gemmini::cycle;
using cycle::NativeCycleReason;
using cycle::NativeCycleSample;

static_assert(noexcept(gemmini_read_cycles()));

namespace
{
bool expect(bool condition, const char * message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

NativeCycleSample valid(uint64_t value, uint64_t owner = 11, uint64_t generation = 7)
{
    return {value, true, NativeCycleReason::none,
            cycle::NativeCycleSource::perf_cpu_cycles, owner, generation};
}

NativeCycleSample invalid(NativeCycleReason reason, uint64_t owner = 11, uint64_t generation = 7)
{
    return {0, false, reason, cycle::NativeCycleSource::perf_cpu_cycles, owner, generation};
}

bool deterministic_matrix()
{
    bool ok = true;
    const auto case_a = cycle::evaluate_interval(valid(58392), invalid(NativeCycleReason::multiplexed));
    ok = expect(!case_a.valid && case_a.reason == NativeCycleReason::invalid_end &&
                case_a.sample_reason == NativeCycleReason::multiplexed, "Case A invalid end priority") && ok;
    const auto case_b = cycle::evaluate_interval(invalid(NativeCycleReason::unavailable_event), valid(5000));
    ok = expect(!case_b.valid && case_b.reason == NativeCycleReason::invalid_start &&
                case_b.sample_reason == NativeCycleReason::unavailable_event, "Case B invalid start priority") && ok;
    const auto case_c = cycle::evaluate_interval(invalid(NativeCycleReason::unavailable_event),
                                                  invalid(NativeCycleReason::multiplexed));
    ok = expect(!case_c.valid && case_c.reason == NativeCycleReason::invalid_start,
                "Case C zero sentinels are invalid_start, not valid zero") && ok;
    unsigned int independent_reads = 0;
    const NativeCycleSample equal_start = (++independent_reads, valid(5000));
    const NativeCycleSample equal_end = (++independent_reads, valid(5000));
    const auto case_d = cycle::evaluate_interval(equal_start, equal_end);
    ok = expect(case_d.valid && case_d.value == 0 && independent_reads == 2,
                "Case D valid zero requires two independent samples") && ok;
    const auto owner_mismatch = cycle::evaluate_interval(valid(10, 11, 7), valid(9, 12, 8), false);
    ok = expect(!owner_mismatch.valid && owner_mismatch.reason == NativeCycleReason::event_owner_mismatch,
                "owner mismatch wins over generation, structure, and regression") && ok;
    const auto case_f = cycle::evaluate_interval(valid(10), valid(9, 11, 8), false);
    ok = expect(!case_f.valid && case_f.reason == NativeCycleReason::event_generation_mismatch,
                "Case F generation mismatch") && ok;
    const auto regression = cycle::evaluate_interval(valid(12), valid(10));
    ok = expect(!regression.valid && regression.reason == NativeCycleReason::counter_regression,
                "counter regression is invalid") && ok;

    cycle::testing::DirectReadInput input{};
    input.cap_user_rdpmc = false;
    ok = expect(cycle::testing::sample_from_input(input).reason == NativeCycleReason::unavailable_direct_mapping,
                "missing direct capability is unavailable_direct_mapping") && ok;
    input.cap_user_rdpmc = true;
    input.index = 0;
    ok = expect(cycle::testing::sample_from_input(input).reason == NativeCycleReason::unavailable_direct_mapping,
                "index zero is unavailable_direct_mapping") && ok;
    input.index = 31;
    ok = expect(cycle::testing::sample_from_input(input).reason == NativeCycleReason::unavailable_direct_mapping,
                "unsupported mapping is unavailable_direct_mapping") && ok;
    input.index = 32;
    input.seqlock_exhausted = true;
    ok = expect(cycle::testing::sample_from_input(input).reason == NativeCycleReason::seqlock_exhausted,
                "seqlock exhaustion is deterministic") && ok;
    input.seqlock_exhausted = false;
    input.time_enabled = 2;
    input.time_running = 1;
    ok = expect(cycle::testing::sample_from_input(input).reason == NativeCycleReason::multiplexed,
                "multiplexing is rejected without scaling") && ok;
    input.time_enabled = input.time_running = 1;
    input.offset = 2147483647;
    input.pmc_width = 32;
    input.raw_value = 0xffffffffU;
    const auto width_sample = cycle::testing::sample_from_input(input);
    ok = expect(width_sample.valid && width_sample.value == 2147483646ULL,
                "32-bit raw value is sign-extended before modulo offset addition") && ok;
    return ok;
}

void cpu_work()
{
    volatile uint64_t value = 1;
    for (uint64_t i = 1; i < 500000; ++i)
    {
        value = value * 1664525U + i;
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

struct WorkerResult
{
    NativeCycleSample start{};
    NativeCycleSample end{};
};

int physical_qa()
{
    cpu_set_t affinity_before{};
    cpu_set_t affinity_after{};
    if (sched_getaffinity(0, sizeof(affinity_before), &affinity_before) != 0)
    {
        std::perror("sched_getaffinity before");
        return 1;
    }
    std::mutex mutex;
    std::condition_variable condition;
    unsigned int ready = 0;
    bool released = false;
    std::array<WorkerResult, 2> results{};
    std::array<std::thread, 2> workers;
    for (size_t i = 0; i < workers.size(); ++i)
    {
        workers[i] = std::thread([&, i] {
            results[i].start = cycle::read_sample();
            {
                std::unique_lock<std::mutex> lock(mutex);
                ++ready;
                condition.notify_all();
                condition.wait(lock, [&] { return released; });
            }
            cpu_work();
            results[i].end = cycle::read_sample();
        });
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] { return ready == workers.size(); });
        released = true;
    }
    condition.notify_all();
    for (auto & worker : workers)
    {
        worker.join();
    }
    bool ok = true;
    for (size_t i = 0; i < results.size(); ++i)
    {
        const auto delta = cycle::evaluate_interval(results[i].start, results[i].end);
        if (!results[i].start.valid)
        {
            std::fprintf(stderr, "BLOCKED physical sample: %s\n", cycle::reason_name(results[i].start.reason));
            return 77;
        }
        ok = expect(delta.valid && delta.value > 0, "same-owner worker CPU delta is positive") && ok;
    }
    ok = expect(results[0].start.owner_event_token != results[1].start.owner_event_token,
                "worker TLS event owners are distinct") && ok;

    const NativeCycleSample direct_before = cycle::read_sample();
    const int fd = cycle::testing::event_fd();
    uint64_t fd_value = 0;
    const ssize_t bytes = fd >= 0 ? ::read(fd, &fd_value, sizeof(fd_value)) : -1;
    const NativeCycleSample direct_after = cycle::read_sample();
    ok = expect(bytes == static_cast<ssize_t>(sizeof(fd_value)) && direct_before.valid && direct_after.valid &&
                direct_before.value <= fd_value && fd_value <= direct_after.value,
                "same event direct reads bracket read(fd)") && ok;

    std::mutex wait_mutex;
    std::condition_variable wait_condition;
    bool wait_worker_ready = false, wait_worker_released = false, wait_worker_done = false;
    std::thread wait_worker([&] {
        {
            std::unique_lock<std::mutex> lock(wait_mutex);
            wait_worker_ready = true;
            wait_condition.notify_all();
            wait_condition.wait(lock, [&] { return wait_worker_released; });
        }
        for (unsigned int i = 0; i < 8; ++i)
        {
            cpu_work();
        }
        {
            std::lock_guard<std::mutex> lock(wait_mutex);
            wait_worker_done = true;
        }
        wait_condition.notify_all();
    });
    std::unique_lock<std::mutex> wait_lock(wait_mutex);
    wait_condition.wait(wait_lock, [&] { return wait_worker_ready; });
    const NativeCycleSample wait_before = cycle::read_sample(); wait_worker_released = true;
    wait_condition.notify_all(); wait_condition.wait(wait_lock, [&] { return wait_worker_done; });
    const NativeCycleSample wait_after = cycle::read_sample(); wait_lock.unlock();
    wait_worker.join(); const NativeCycleSample active_before = cycle::read_sample(); cpu_work();
    const NativeCycleSample active_after = cycle::read_sample();
    const auto wait_delta = cycle::evaluate_interval(wait_before, wait_after),
               active_delta = cycle::evaluate_interval(active_before, active_after);
    ok = expect(wait_delta.valid && active_delta.valid && wait_delta.value < active_delta.value,
                "descheduled barrier wait is not reported as active CPU work") && ok;

    const NativeCycleSample kernel_before = cycle::read_sample();
    for (unsigned int i = 0; i < 20000; ++i)
    {
        (void) syscall(SYS_getpid);
    }
    const NativeCycleSample kernel_after = cycle::read_sample();
    const auto kernel_delta = cycle::evaluate_interval(kernel_before, kernel_after);
    ok = expect(kernel_delta.valid && kernel_delta.value > 0, "kernel-inclusive syscall work advances") && ok;
    ok = expect(sched_getaffinity(0, sizeof(affinity_after), &affinity_after) == 0 &&
                std::memcmp(&affinity_before, &affinity_after, sizeof(affinity_before)) == 0,
                "QA does not change affinity") && ok;
    if (ok) {
        std::printf("PASS physical worker0=%llu worker1=%llu owners=%llu,%llu wait=%llu active=%llu kernel=%llu direct_fd=%llu\n",
                    static_cast<unsigned long long>(cycle::evaluate_interval(results[0].start, results[0].end).value),
                    static_cast<unsigned long long>(cycle::evaluate_interval(results[1].start, results[1].end).value),
                    static_cast<unsigned long long>(results[0].start.owner_event_token),
                    static_cast<unsigned long long>(results[1].start.owner_event_token),
                    static_cast<unsigned long long>(wait_delta.value),
                    static_cast<unsigned long long>(active_delta.value),
                    static_cast<unsigned long long>(kernel_delta.value),
                    static_cast<unsigned long long>(fd_value));
    }
    return ok ? 0 : 1;
}
}

int main(int argc, char ** argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0)
    {
        std::puts("usage: test-gemmini-cycle-reader-aarch64 [--physical|--inject-multiplexed|--help]");
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--physical") == 0)
    {
        return physical_qa();
    }
    if (argc == 2 && std::strcmp(argv[1], "--inject-multiplexed") == 0)
    {
        cycle::testing::DirectReadInput input{};
        input.time_enabled = 2;
        input.time_running = 1;
        const auto sample = cycle::testing::sample_from_input(input);
        std::printf("valid=%s reason=%s scaled=false\n", sample.valid ? "true" : "false",
                    cycle::reason_name(sample.reason));
        return sample.valid || sample.reason != NativeCycleReason::multiplexed ? 1 : 0;
    }
    if (argc != 1)
    {
        std::fprintf(stderr, "error: unknown argument: %s\n", argv[1]);
        return 2;
    }

    cycle::reset_read_count_for_test();
    (void) gemmini_read_cycles();
    const bool public_api_ok = expect(cycle::read_count_for_test() == 1,
                                      "public API performs one physical read/projection");
    const bool native_mode = expect(std::strcmp(cycle::clock_mode(), "CYCLE") == 0 &&
                                    std::strcmp(cycle::units(), "cycles") == 0,
                                    "Linux AArch64 public API is CYCLE/cycles");
    const bool ok = public_api_ok && native_mode && deterministic_matrix();
    if (ok)
    {
        std::puts("PASS native reader matrix");
    }
    return ok ? 0 : 1;
}
