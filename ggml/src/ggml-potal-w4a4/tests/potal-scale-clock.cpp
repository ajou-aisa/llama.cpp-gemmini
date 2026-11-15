#include <chrono>
#include <cstdint>

extern "C" uint64_t potal_scale_bench_steady_ns(void);
extern "C" uint64_t potal_scale_bench_steady_ns(void) {
    static_assert(std::chrono::steady_clock::is_steady, "monotonic clock required");
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
