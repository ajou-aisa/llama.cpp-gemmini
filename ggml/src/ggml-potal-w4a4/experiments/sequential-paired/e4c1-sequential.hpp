#pragma once

// Opt-in acceptance executor, not a production scheduler or a hardware model.
// NULL transport selects the real checked resident API. A non-NULL transport
// is explicitly software-only. Every call belongs on the existing owner thread.
#include "matmul/paired-auto.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace potal_test::e4c1 {

struct options {
    uint32_t main_k_tiles = 4;
    uint32_t acc_tile = 1;
    bool poison_b = true;
    bool trace = false;
    bool profile = false;
};

enum class scope : std::size_t {
    planning, prepare, b_issue, b_completion, scale_program, index_program,
    main_issue, residual_issue, completion, scale_release, readback,
    retirement, close, count
};

struct interval {
    uint64_t ns = 0, calls = 0;
};

enum class event_kind { load, main, residual, retire };

struct event {
    event_kind kind = event_kind::load;
    uint32_t pair = 0, bank = 0, base = 0, pitch = 0, acc_tile = 0;
    uint32_t k_rows = 0, columns = 0, scale_base = 0, scale_generation = 0;
    uint64_t original_k = 0, original_j = 0, sequence = 0;
    uint32_t group = 0, indexed = 0, first = 0, last = 0;
};

struct statistics {
    uint64_t main_microtiles = 0, residual_microtiles = 0;
    uint64_t main_partials = 0, residual_partials = 0;
    uint64_t main_to_residual = 0, residual_to_main = 0;
    uint64_t main_only_tail = 0, residual_only_tail = 0;
    uint64_t b_loads = 0, b_mvins = 0, b_retirements = 0;
    uint64_t scale_programs = 0, scale_slot_instances = 0;
    uint64_t main_scale_writes = 0, residual_scale_writes = 0;
    uint64_t main_scale_releases = 0, residual_scale_releases = 0;
    uint64_t scale_generations = 0, generation_wraps = 0;
    uint64_t release_boundaries = 0, index_configs = 0, index_writes = 0;
    uint64_t fences = 0, readbacks = 0, mvouts = 0;
    uint64_t views = 0, loops = 0, commands = 0, drains = 0;
    uint64_t peak_scale_slots = 0, peak_index_entries = 0, peak_b_residencies = 0;
    uint64_t skipped_zero_group_fragments = 0;
    bool software_only = true;
    std::array<interval, static_cast<std::size_t>(scope::count)> intervals{};
    std::vector<event> events;
};

// Reuses the supplied validated H2 plan (or builds it with the existing
// planner), quantized operands and decoded sealed packet. It splits only the
// execution into bounded I1/P1 calls. Output is copied to caller buffers only
// after every contribution, readback, retirement and close succeeds.
// No fallback, ACC spill/restore, prefetch or new raw issuing path exists.
ggml::potal::potal_paired_status run(
    const ggml::potal::potal_paired_request & request,
    const potal_resident_ws_options * transport, const options & config,
    ggml::potal::potal_paired_report & report, statistics & stats);

void add(statistics & total, const statistics & value);

} // namespace potal_test::e4c1
