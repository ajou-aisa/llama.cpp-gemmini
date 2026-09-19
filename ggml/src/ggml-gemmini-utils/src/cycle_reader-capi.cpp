#include "../include/gemmini/cycle_reader.hpp"
#include "../include/gemmini/cycle_reader.h"
#include "cycle_reader_internal.h"

#if LOG_CYCLE && !CYCLE_DETAIL
#include "../include/gemmini/host-timing.hpp"
#include <array>
#include <cstddef>

namespace {
struct ScalarCycleEndpoint {
    uint64_t value = 0;
    uint64_t ns = 0;
    uint64_t tid = 0;
    uint64_t sequence = 0;
    bool active = false;
};

constexpr std::size_t kScalarCycleEndpointCapacity = 4096;
thread_local std::array<ScalarCycleEndpoint, kScalarCycleEndpointCapacity> scalar_cycle_endpoints{};
thread_local uint64_t scalar_cycle_sequence = 0;

void remember_scalar_cycle_endpoint(uint64_t value, uint64_t ns, uint64_t tid) noexcept {
    const uint64_t sequence = ++scalar_cycle_sequence;
    scalar_cycle_endpoints[(sequence - 1) % kScalarCycleEndpointCapacity] =
        {value, ns, tid, sequence, true};
}

const ScalarCycleEndpoint * endpoint_for_sequence(uint64_t sequence) noexcept {
    if (sequence == 0 || scalar_cycle_sequence - sequence >= kScalarCycleEndpointCapacity) return nullptr;
    const auto & endpoint = scalar_cycle_endpoints[(sequence - 1) % kScalarCycleEndpointCapacity];
    return endpoint.sequence == sequence && endpoint.active ? &endpoint : nullptr;
}
}
#endif

extern "C" uint64_t gemmini_read_cycles(void) noexcept
{
    try {
#if LOG_CYCLE && !CYCLE_DETAIL
        const uint64_t ns = ggml::gemmini::cycle::timeline_now_ns();
        const uint64_t tid = ggml::gemmini::cycle::host_thread_id();
        const uint64_t value = ggml::gemmini::cycle::read();
        remember_scalar_cycle_endpoint(value, ns, tid);
        return value;
#else
        return ggml::gemmini::cycle::read();
#endif
    }
    catch (...) { return 0; }
}

#if !CYCLE_DETAIL
extern "C" uint8_t gemmini_take_scalar_cycle_interval_internal(
        uint64_t start, uint64_t end,
        gemmini_scalar_cycle_interval_internal * interval) noexcept
{
#if !LOG_CYCLE
    (void) start; (void) end; (void) interval;
    return 0;
#else
    if (interval == nullptr || scalar_cycle_sequence == 0) return 0;

    const uint64_t oldest = scalar_cycle_sequence >= kScalarCycleEndpointCapacity
        ? scalar_cycle_sequence - kScalarCycleEndpointCapacity + 1 : 1;
    uint64_t end_sequence = 0;
    uint64_t start_sequence = 0;
    ScalarCycleEndpoint end_endpoint{};
    ScalarCycleEndpoint start_endpoint{};

    for (uint64_t sequence = scalar_cycle_sequence; sequence >= oldest; --sequence) {
        const ScalarCycleEndpoint * endpoint = endpoint_for_sequence(sequence);
        if (endpoint != nullptr) {
            if (end_sequence == 0) {
                if (endpoint->value == end) {
                    end_sequence = sequence;
                    end_endpoint = *endpoint;
                }
            } else if (endpoint->value == start) {
                start_sequence = sequence;
                start_endpoint = *endpoint;
                break;
            }
        }
        if (sequence == oldest) break;
    }

    if (start_sequence == 0 || end_sequence == 0 || start_sequence >= end_sequence) return 0;
    *interval = {start_endpoint.ns, end_endpoint.ns, start_endpoint.tid, end_endpoint.tid};

    for (uint64_t sequence = start_sequence; sequence <= end_sequence; ++sequence) {
        auto & endpoint = scalar_cycle_endpoints[(sequence - 1) % kScalarCycleEndpointCapacity];
        if (endpoint.sequence == sequence) endpoint.active = false;
        if (sequence == end_sequence) break;
    }
    return 1;
#endif
}
#endif
