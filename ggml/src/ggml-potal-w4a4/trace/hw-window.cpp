// H3-8I-25AC: records of the strict hardware timeline. See hw-window.hpp.
#include "trace/hw-window.hpp"

#include <sys/mman.h>

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace potal::hw_window {
namespace {

constexpr size_t kDefaultWindows = 524288;
constexpr size_t kStripes        = 65536;
constexpr size_t kNodes          = 16384;
constexpr size_t kNameBytes      = 64;

struct node_row {
    uint32_t timeline_run, layer_id, rows;
    uint8_t  phase;
    char     name[kNameBytes];
};

struct state_t {
    record *              windows = nullptr;
    stripe_record *       stripes = nullptr;
    node_row *            nodes   = nullptr;
    size_t                window_cap = 0;
    std::atomic<uint64_t> window_count{0}, stripe_count{0}, node_count{0};
    std::atomic<uint64_t> dropped_windows{0}, dropped_stripes{0}, commands{0};
    std::atomic<bool>     disabled{false};
    std::atomic<bool>     reason_set{false};
    char                  reason[128] = {};
    bool                  probe_done = false, counters_done = false, atexit_done = false;
};

state_t & state() {
    static state_t * s = new state_t();  // never destroyed: flush runs from atexit
    return *s;
}

template<class T>
T * touched_array(size_t n) {
    void * p = mmap(nullptr, n * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    std::memset(p, 0, n * sizeof(T));
    return static_cast<T *>(p);
}

void flush_at_exit() {
    flush();
}

// Allocation happens at load time when the option is set there, or at the first node otherwise.
bool ensure_buffers() noexcept {
    state_t & s = state();
    if (s.windows != nullptr) return true;
    size_t cap = kDefaultWindows;
    if (const char * v = std::getenv("POTAL_HW_TIMELINE_WINDOWS")) {
        const unsigned long long parsed = std::strtoull(v, nullptr, 10);
        if (parsed > 0) cap = size_t(parsed);
    }
    s.stripes = touched_array<stripe_record>(kStripes);
    s.nodes   = touched_array<node_row>(kNodes);
    s.windows = touched_array<record>(cap);
    if (s.windows == nullptr || s.stripes == nullptr || s.nodes == nullptr) {
        note_failure("allocation", -1);
        return false;
    }
    s.window_cap = cap;
    if (!s.atexit_done) {
        s.atexit_done = true;
        if (std::atexit(flush_at_exit) != 0) note_failure("atexit", -1);
    }
    return true;
}

struct load_time_init {
    load_time_init() {
        bool invalid = false;
        if (requested(invalid)) ensure_buffers();
    }
} load_time_init_instance;

}  // namespace

bool requested(bool & invalid) noexcept {
    invalid = false;
    const char * v = std::getenv("GGML_POTAL_STRICT_HW_TIMELINE");
    if (v == nullptr || *v == '\0') return false;
    if (std::strcmp(v, "ON") == 0 || std::strcmp(v, "1") == 0) return true;
    if (std::strcmp(v, "OFF") == 0 || std::strcmp(v, "0") == 0) return false;
    invalid = true;
    return false;
}

uint32_t begin_node(uint32_t timeline_run, uint32_t layer_id, uint8_t phase, uint32_t rows, const char * name) noexcept {
    state_t & s = state();
    if (s.disabled.load(std::memory_order_relaxed) || !ensure_buffers()) return 0;
    const uint64_t n = s.node_count.load(std::memory_order_relaxed);
    if (n >= kNodes) {
        note_failure("node_table_full", -1);
        return 0;
    }
    node_row & row   = s.nodes[n];
    row.timeline_run = timeline_run;
    row.layer_id     = layer_id;
    row.rows         = rows;
    row.phase        = phase;
    std::snprintf(row.name, sizeof(row.name), "%s", name != nullptr ? name : "");
    s.node_count.store(n + 1, std::memory_order_relaxed);
    return uint32_t(n + 1);
}

bool disabled() noexcept {
    return state().disabled.load(std::memory_order_relaxed);
}

void note_failure(const char * stage, int status) noexcept {
    state_t & s = state();
    s.disabled.store(true, std::memory_order_relaxed);
    if (!s.reason_set.exchange(true)) {
        std::snprintf(s.reason, sizeof(s.reason), "%s status=%d", stage, status);
        std::fprintf(stderr, "[POTAL_HW_TIMELINE] disabled: %s\n", s.reason);
    }
}

void count_commands(uint64_t n) noexcept {
    state().commands.fetch_add(n, std::memory_order_relaxed);
}

record * acquire_window() noexcept {
    state_t & s = state();
    const uint64_t n = s.window_count.load(std::memory_order_relaxed);
    if (s.windows == nullptr || n >= s.window_cap) {
        s.dropped_windows.fetch_add(1, std::memory_order_relaxed);
        note_failure("window_capacity", -1);
        return nullptr;
    }
    s.window_count.store(n + 1, std::memory_order_relaxed);
    return &s.windows[n];
}

stripe_record * acquire_stripe() noexcept {
    state_t & s = state();
    const uint64_t n = s.stripe_count.load(std::memory_order_relaxed);
    if (s.stripes == nullptr || n >= kStripes) {
        s.dropped_stripes.fetch_add(1, std::memory_order_relaxed);
        note_failure("stripe_capacity", -1);
        return nullptr;
    }
    s.stripe_count.store(n + 1, std::memory_order_relaxed);
    return &s.stripes[n];
}

bool take_probe() noexcept {
    state_t & s = state();
    const bool first = !s.probe_done;
    s.probe_done = true;
    return first;
}

bool take_counter_config() noexcept {
    state_t & s = state();
    const bool first = !s.counters_done;
    s.counters_done = true;
    return first;
}

stats_t stats() noexcept {
    state_t & s = state();
    stats_t out{};
    out.windows         = s.window_count.load(std::memory_order_relaxed);
    out.stripes         = s.stripe_count.load(std::memory_order_relaxed);
    out.nodes           = s.node_count.load(std::memory_order_relaxed);
    out.dropped_windows = s.dropped_windows.load(std::memory_order_relaxed);
    out.dropped_stripes = s.dropped_stripes.load(std::memory_order_relaxed);
    out.commands        = s.commands.load(std::memory_order_relaxed);
    out.disabled        = s.disabled.load(std::memory_order_relaxed);
    out.reason          = s.reason_set.load() ? s.reason : nullptr;
    return out;
}

const record * windows(size_t * count) noexcept {
    state_t & s = state();
    if (count != nullptr) *count = size_t(s.window_count.load(std::memory_order_relaxed));
    return s.windows;
}

const stripe_record * stripes(size_t * count) noexcept {
    state_t & s = state();
    if (count != nullptr) *count = size_t(s.stripe_count.load(std::memory_order_relaxed));
    return s.stripes;
}

bool flush() noexcept {
    state_t & s = state();
    const char * path = std::getenv("POTAL_HW_TIMELINE");
    if (path == nullptr || *path == '\0' || s.windows == nullptr) return false;
    const stats_t st = stats();
    bool ok = true;
    if (FILE * f = std::fopen(path, "wb")) {
        const uint32_t head[4] = {0x54574850u, format_version, uint32_t(sizeof(record)), window_fields};
        ok = std::fwrite(head, sizeof(head), 1, f) == 1;
        const uint64_t count = st.windows;
        ok = ok && std::fwrite(&count, sizeof(count), 1, f) == 1;
        ok = ok && (count == 0 || std::fwrite(s.windows, sizeof(record), size_t(count), f) == size_t(count));
        ok = (std::fclose(f) == 0) && ok;
    } else {
        ok = false;
    }
    const std::string base(path);
    if (FILE * f = std::fopen((base + ".stripes.csv").c_str(), "w")) {
        std::fprintf(f, "node_id,timeline_run,stripe_id,windows,status,cycle_begin,cycle_end");
        for (uint32_t i = 0; i < counter_count; ++i) std::fprintf(f, ",event_%u", counter_events[i]);
        std::fputc('\n', f);
        for (uint64_t i = 0; i < st.stripes; ++i) {
            const stripe_record & r = s.stripes[i];
            std::fprintf(f, "%u,%u,%u,%u,%u,%" PRIu64 ",%" PRIu64, r.node_id, r.timeline_run, r.stripe_id, r.windows,
                         r.status, r.cycle_begin, r.cycle_end);
            for (uint32_t c = 0; c < counter_count; ++c) std::fprintf(f, ",%" PRIu64, r.counter[c]);
            std::fputc('\n', f);
        }
        ok = (std::fclose(f) == 0) && ok;
    } else {
        ok = false;
    }
    if (FILE * f = std::fopen((base + ".nodes.csv").c_str(), "w")) {
        std::fprintf(f, "node_id,timeline_run,layer_id,phase,rows,name\n");
        for (uint64_t i = 0; i < st.nodes; ++i) {
            const node_row & r = s.nodes[i];
            std::fprintf(f, "%" PRIu64 ",%u,%u,%u,%u,%s\n", i + 1, r.timeline_run, r.layer_id, unsigned(r.phase), r.rows, r.name);
        }
        ok = (std::fclose(f) == 0) && ok;
    } else {
        ok = false;
    }
    if (FILE * f = std::fopen((base + ".meta.txt").c_str(), "w")) {
        std::fprintf(f, "format_version=%u\nrecord_bytes=%zu\nwindow_fields=%u\n", format_version, sizeof(record), window_fields);
        std::fprintf(f, "windows=%" PRIu64 "\nstripes=%" PRIu64 "\nnodes=%" PRIu64 "\n", st.windows, st.stripes, st.nodes);
        std::fprintf(f, "dropped_windows=%" PRIu64 "\ndropped_stripes=%" PRIu64 "\ncommands=%" PRIu64 "\n",
                     st.dropped_windows, st.dropped_stripes, st.commands);
        std::fprintf(f, "disabled=%d\nreason=%s\n", st.disabled ? 1 : 0, st.reason != nullptr ? st.reason : "none");
#if defined(__riscv)
        std::fprintf(f, "clock=rdcycle\n");
#else
        std::fprintf(f, "clock=host\n");
#endif
        ok = (std::fclose(f) == 0) && ok;
    } else {
        ok = false;
    }
    if (!ok) std::fprintf(stderr, "[POTAL_HW_TIMELINE] flush failed: %s\n", path);
    return ok;
}

void reset_for_test() noexcept {
    state_t & s = state();
    s.window_count.store(0);
    s.stripe_count.store(0);
    s.node_count.store(0);
    s.dropped_windows.store(0);
    s.dropped_stripes.store(0);
    s.commands.store(0);
    s.disabled.store(false);
    s.reason_set.store(false);
    s.reason[0]     = '\0';
    s.probe_done    = false;
    s.counters_done = false;
}

selftest_result selftest_evaluate() noexcept {
    selftest_result out{};
    const stats_t st = stats();
    size_t n = 0, nstripes = 0;
    const record * w = windows(&n);
    const stripe_record * stripe = stripes(&nstripes);
    const auto fail = [&out](bool & flag, const char * why) {
        flag = false;
        if (out.first_failure == nullptr) out.first_failure = why;
    };
    out.windows = n;
    out.reset_accepted = out.read_response = out.monotonic = out.load_and_mac = true;
    out.ownership = out.alignment = out.no_overflow = true;
    const bool arm_failure = st.reason != nullptr && std::strncmp(st.reason, "arm", 3) == 0;
    if (n == 0 || arm_failure) fail(out.reset_accepted, "no armed window");
    if (st.dropped_windows != 0 || st.dropped_stripes != 0) fail(out.no_overflow, "software trace overflow");
    if (st.disabled) fail(out.read_response, st.reason != nullptr ? st.reason : "disabled");
    if (nstripes == 0) fail(out.ownership, "no stripe record");
    bool saw_mac = false;
    uint64_t counted = 0;
    for (size_t i = 0; i < n; ++i) {
        const record & r = w[i];
        if ((r.status & (st_read_failed | st_aborted)) != 0 || (r.read_mask & 7u) != 7u)
            fail(out.read_response, "window not fully read");
        if ((r.status & st_overflow) != 0) fail(out.no_overflow, "hardware window overflow");
        if ((r.status & st_not_frozen) != 0) fail(out.ownership, "window not frozen by its Fence");
        if (r.kind == uint8_t(kind::unknown)) fail(out.ownership, "window composition unknown");
        constexpr uint64_t activity = flag_seen_mac0 | flag_seen_mac1 | flag_seen_b0 | flag_seen_b1 | flag_seen_dma0 | flag_seen_dma1;
        if (r.kind == uint8_t(kind::no_gemm) && (r.field[f_flags] & activity) != 0)
            fail(out.ownership, "MAC or weight activity in a window with no GEMM");
        const bool same_stripe = i > 0 && w[i - 1].node_id == r.node_id && w[i - 1].stripe_id == r.stripe_id;
        if (r.window_seq != (same_stripe ? w[i - 1].window_seq + 1 : 0)) fail(out.ownership, "window sequence gap");
        if (r.field[f_start] > r.field[f_end]) fail(out.monotonic, "start after end");
        if (i > 0 && (w[i - 1].read_mask & 4u) != 0 && r.field[f_start] < w[i - 1].field[f_end])
            fail(out.monotonic, "window starts before the previous one ended");
        const auto order = [&](uint32_t first, uint32_t last) {
            if (((r.read_mask >> first) & 1u) == 0) return;
            if (r.field[first] < r.field[f_start] || r.field[last] < r.field[first] || r.field[f_end] < r.field[last])
                fail(out.monotonic, "stamps out of order");
        };
        order(f_busy_first, f_busy_last);
        order(f_mac_first, f_mac_last);
        order(f_bwrite_first, f_bwrite_last);
        order(f_bdma_first, f_bdma_last);
        if (((r.read_mask >> f_mac_first) & 1u) != 0) {
            saw_mac = true;
            out.mac_cycles += r.field[f_mac_cycles];
        }
        if (((r.read_mask >> f_bdma_first) & 1u) != 0 && ((r.read_mask >> f_bwrite_first) & 1u) != 0) ++out.bdma_windows;
        if (r.cycle_arm_after < r.cycle_arm_before || r.cycle_fence_after < r.cycle_fence_before)
            fail(out.alignment, "H0 bracket reversed");
        const int64_t arm_low    = int64_t(r.field[f_start] - r.cycle_arm_after);
        const int64_t arm_high   = int64_t(r.field[f_start] - r.cycle_arm_before);
        const int64_t fence_low  = int64_t(r.field[f_end] - r.cycle_fence_after);
        const int64_t fence_high = int64_t(r.field[f_end] - r.cycle_fence_before);
        const uint64_t width = r.cycle_arm_after - r.cycle_arm_before;
        if (width > out.alignment_width_max) out.alignment_width_max = width;
        if (arm_high < fence_low || fence_high < arm_low) ++out.alignment_inconsistent;
        ++counted;
    }
    if (!saw_mac || out.bdma_windows == 0) fail(out.load_and_mac, "no MAC or weight load event");
    if (counted != 0 && out.alignment_inconsistent * 100 > counted) fail(out.alignment, "H0 brackets disagree with the hardware clock");
    size_t cursor = 0;
    for (size_t i = 0; i < nstripes; ++i) {
        const uint32_t expected = stripe[i].windows;
        for (uint32_t k = 0; k < expected; ++k, ++cursor) {
            if (cursor >= n || w[cursor].node_id != stripe[i].node_id || w[cursor].stripe_id != stripe[i].stripe_id) {
                fail(out.ownership, "stripe record does not own its windows");
                return out;
            }
        }
    }
    if (cursor != n) fail(out.ownership, "windows without a stripe record");
    return out;
}

}  // namespace potal::hw_window

extern "C" __attribute__((visibility("default"))) bool ggml_backend_potal_hw_timeline_stats(
    uint64_t * windows, uint64_t * stripes, uint64_t * dropped, uint64_t * commands, bool * disabled, const char ** reason) {
    const auto st = ::potal::hw_window::stats();
    if (windows != nullptr) *windows = st.windows;
    if (stripes != nullptr) *stripes = st.stripes;
    if (dropped != nullptr) *dropped = st.dropped_windows + st.dropped_stripes;
    if (commands != nullptr) *commands = st.commands;
    if (disabled != nullptr) *disabled = st.disabled;
    if (reason != nullptr) *reason = st.reason;
    bool invalid = false;
    return ::potal::hw_window::requested(invalid);
}

extern "C" __attribute__((visibility("default"))) bool ggml_backend_potal_hw_timeline_flush(void) {
    return ::potal::hw_window::flush();
}
