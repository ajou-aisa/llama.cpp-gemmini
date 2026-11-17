#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <new>
#include <type_traits>

// Included after the selected backend's timeline.hpp; W4 and W8 share the wire format.
namespace potal::timeline::binary_trace {

constexpr uint32_t block_events = 4096;
static_assert(std::is_standard_layout<record_t>::value && std::is_trivially_copyable<record_t>::value,
              "binary events require a fixed trivial layout");
static_assert(sizeof(record_t) == 64 && offsetof(record_t, t_local) == 0 &&
              offsetof(record_t, seq) == 8 && offsetof(record_t, thread_slot) == 12 &&
              offsetof(record_t, stage) == 14 && offsetof(record_t, event_type) == 16 &&
              offsetof(record_t, status) == 17 && offsetof(record_t, op_type) == 18 &&
              offsetof(record_t, phase) == 19 && offsetof(record_t, run_id) == 20 &&
              offsetof(record_t, layer_id) == 24 && offsetof(record_t, stripe_id) == 28 &&
              offsetof(record_t, tile_id) == 32 && offsetof(record_t, buffer_id) == 36 &&
              offsetof(record_t, edge_id) == 40 && offsetof(record_t, pad0) == 44 &&
              offsetof(record_t, work) == 48 && offsetof(record_t, reserved) == 56,
              "timeline binary v1 event layout changed");

struct cpu_record {
    uint64_t mono_ns, thread_ns, process_ns, cycles, instructions;
    uint32_t seq;
    int32_t cpu;
    uint32_t flags, reserved;
};
static_assert(sizeof(cpu_record) == 56 && offsetof(cpu_record, seq) == 40 &&
              offsetof(cpu_record, cpu) == 44 && offsetof(cpu_record, flags) == 48 &&
              offsetof(cpu_record, reserved) == 52, "timeline binary v1 CPU layout changed");

inline bool supported() noexcept {
    const uint32_t one = 1;
    if (*reinterpret_cast<const unsigned char *>(&one) == 1) return true;
    std::fprintf(stderr, "potal timeline: binary v1 requires little-endian; use csv\n");
    return false;
}

inline void header(FILE * f, bool cpu) noexcept {
    std::fprintf(f, "# potal_timeline_binary v1\n#BINARY,endian=little,event_bytes=64,cpu_bytes=56,cpu=%u,counter_source=%s\n",
                 cpu ? 1 : 0,
#if defined(__riscv) && __riscv_xlen == 64
                 "riscv_hart_csr"
#else
                 "unavailable"
#endif
    );
}

inline bool write_ring(FILE * f, const detail::thread_ring & ring) noexcept {
    // Bounded heap scratch only at shutdown, never on a Rocket worker's recording path.
    std::unique_ptr<cpu_record[]> samples;
    if (ring.cpu_samples != nullptr) {
        samples.reset(new (std::nothrow) cpu_record[block_events]);
        if (!samples) return false;
    }
    const uint32_t total = ring.head.load(std::memory_order_relaxed);
    for (uint32_t start = 0; start < total;) {
        const uint32_t count = std::min(block_events, total - start);
        uint32_t sampled = 0;
        if (samples) {
            for (uint32_t j = start; j < start + count; ++j) {
                const auto & s = ring.cpu_samples[j];
                if (s.mono_ns == 0) continue;
                const uint32_t flags = uint32_t(s.thread_valid) | (uint32_t(s.process_valid) << 1) |
                                       (uint32_t(s.counters_valid) << 2);
                samples[sampled++] = {s.mono_ns, s.thread_ns, s.process_ns, s.cycles, s.instructions,
                                      j, int32_t(s.cpu), flags, 0};
            }
        }
        if (std::fprintf(f, "#BLOCK,slot=%u,events=%u,samples=%u\n", ring.slot, count, sampled) < 0 ||
            std::fwrite(ring.events + start, sizeof(record_t), count, f) != count ||
            (sampled != 0 && std::fwrite(samples.get(), sizeof(cpu_record), sampled, f) != sampled) ||
            std::fputc('\n', f) == EOF) return false;
        start += count;
    }
    return true;
}

}  // namespace potal::timeline::binary_trace
