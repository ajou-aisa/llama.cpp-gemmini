// Test-only E3 diagnostic. Build like potal-paired-inventory, with the GNU
// --wrap flags listed below. Do not enable LTO across the checked C API seam:
// wrapping requires unresolved references from paired-auto.cpp to potal.c.
//
// Required linker symbols (each needs -Wl,--wrap=SYMBOL):
// potal_paired_scale_write, potal_paired_scale_release,
// potal_paired_index_config, potal_paired_index_write,
// potal_paired_loop_ws_issue, potal_paired_acc_readback_issue,
// potal_paired_fence, potal_paired_session_close.
//
// Usage: potal-e3-window-diag --case NAME [--hardware] [--prefence]
// NAME: decode_1x768x768 or gpt2_ffn_up_16x3072x768.
// Default: software-only H2 command emulator, including on RISC-V.
// --hardware: existing issuer_owner only (Linux CPU1 / hart0), no fallback.
// --prefence: intrusive extra checked Fence before a Scale window's first
// Release when the C11 software state reports outstanding work. This changes
// the command stream. Neither interval is pure FPGA waiting or production
// latency. A successful Fence guarantees allDrained, not a breakdown of waits.
//
// Timers enclose only __real_* calls. Mapping, queries and aggregation are
// outside those intervals. All records print after execution. Counts and
// extents are logical / command_declared, not actual DMA or overlap. Internal
// C11 cleanup raw commands do not cross these public wrappers; a failed run
// reports failure rather than claiming complete normal-path attribution.
// POTAL_PAIRED_PROFILE is left to the caller; it is never enabled here.

#include "ggml-cpp.h"
#include "ggml-quants.h"
#include "matmul/dense.hpp"
#include "matmul/paired-auto.hpp"
#include "quants/act/exsia/types.hpp"
#include "support/paired-h2-emulator.hpp"
#include "support/stripe-golden.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <variant>
#include <vector>

using namespace ggml::potal;

extern "C" {
potal_status __real_potal_paired_scale_write(potal_paired_session *, uint32_t, uint32_t, uint32_t, uint32_t);
potal_status __real_potal_paired_scale_release(potal_paired_session *, uint32_t, uint32_t, uint32_t);
potal_status __real_potal_paired_index_config(potal_paired_session *, const potal_index_config *);
potal_status __real_potal_paired_index_write(potal_paired_session *, const potal_index_write *);
potal_status __real_potal_paired_loop_ws_issue(potal_paired_session *, const potal_paired_loop_ws *);
potal_status __real_potal_paired_acc_readback_issue(potal_paired_session *, const potal_acc_readback *);
potal_status __real_potal_paired_fence(potal_paired_session *);
potal_status __real_potal_paired_session_close(potal_paired_session *);
}

namespace {

constexpr size_t D = POTAL_HW_DIM;
constexpr size_t none = potal_paired_no_index;
using clock_type = std::chrono::steady_clock;
enum api { scale_write, scale_release, index_config, index_write, loop_issue, readback, fence, api_count };
const char * const api_names[] = {
    "Scale_Write", "Scale_Release", "Index_Config", "Index_Write", "LOOP_WS", "readback", "Fence"
};

struct measurement {
    uint64_t calls = 0, succeeded = 0, ns = 0;
    potal_status first_error = POTAL_OK;
    void add(potal_status status, uint64_t elapsed) {
        ++calls;
        succeeded += status == POTAL_OK;
        ns += elapsed;
        if (first_error == POTAL_OK && status != POTAL_OK) first_error = status;
    }
};

struct window_record {
    std::array<measurement, api_count> measurements{};
    measurement extra_fence;
    uint32_t generation = 0;
    bool release_seen = false;
    potal_paired_session_status before_release{};
    potal_paired_session_status after_prefence{};
    uint64_t first_release_ns = 0, first_release_sequence = 0;
};

struct completion_record {
    api kind = fence;
    size_t op = none;
    uint64_t sequence = 0, ns = 0;
    potal_status status = POTAL_OK;
    potal_paired_session_status before{}, after{};
};

struct binding {
    size_t window = none;
    uint32_t generation = 0;
};

struct diagnostic {
    const potal_paired_execution_plan & plan;
    // Retain the addresses for reporting even if a fatal run abandons ownership.
    const uint32_t * carriers;
    size_t carrier_stride;
    bool prefence = false, valid = true, close_seen = false;
    size_t program_cursor = 0, index_cursor = 0, index_written = 0, loop_cursor = 0, read_cursor = 0;
    size_t current_index = none, last_op = none;
    uint64_t sequence = 0;
    std::vector<size_t> programs, indices, loops, reads, index_windows;
    std::vector<window_record> windows;
    std::vector<completion_record> completions;
    std::array<measurement, api_count> totals{};
    measurement extra_fences;
    binding live[POTAL_SCALE_ENTRIES][D]{};
    potal_paired_session_status final_state{};
    potal_status close_status = POTAL_ERR_BUSY;

    diagnostic(const potal_paired_execution_plan & p, const hp1_weight & w, bool pre) :
        plan(p), carriers(w.carriers), carrier_stride(w.cstride), prefence(pre),
        index_windows(p.ops.size(), none), windows(p.scale_windows.size()) {
        for (const auto & op : p.ops) {
            switch (op.kind) {
                case potal_paired_op_kind::scale_program: programs.push_back(op.id); break;
                case potal_paired_op_kind::index_program: indices.push_back(op.id); break;
                case potal_paired_op_kind::main_gemm:
                case potal_paired_op_kind::residual_gemm: loops.push_back(op.id); break;
                case potal_paired_op_kind::acc_complete: reads.push_back(op.id); break;
                default: break;
            }
        }
        for (size_t id : indices) {
            const auto & op = p.ops[id];
            for (const auto & window : p.scale_windows) {
                if (window.residual && window.residency == op.residency && window.window == op.window) {
                    index_windows[id] = window.index;
                }
            }
            valid &= index_windows[id] != none;
        }
        // No allocations in a wrapper, even on an unexpected command stream.
        completions.reserve(p.ops.size() * 2 + 8);
    }

    size_t scale_key(uint32_t slot, uint32_t generation, uint32_t column) {
        if (slot >= POTAL_SCALE_ENTRIES || column >= D) {
            valid = false;
            return none;
        }
        const auto & key = live[slot][column];
        if (key.window == none || key.generation != generation) valid = false;
        return key.generation == generation ? key.window : none;
    }

    size_t begin_write(uint32_t slot, uint32_t generation, uint32_t column, uint32_t carrier) {
        if (program_cursor >= programs.size()) {
            valid = false;
            return none;
        }
        const auto & op = plan.ops[programs[program_cursor]];
        const auto & w = plan.scale_windows[op.scale_window];
        auto & record = windows[w.index];
        const auto & residency = plan.residencies[w.residency];
        const size_t ordinal = record.measurements[scale_write].calls;
        const size_t block = ordinal / residency.cols;
        const size_t col = ordinal % residency.cols;
        if (block >= w.blocks) {
            valid = false;
            return none;
        }
        const size_t expected_slot = w.slot_base + block * w.j_tiles + col / D;
        const auto & panel = plan.tile.panels[op.panel];
        const size_t original_block = w.residual
            ? plan.tile.fragments[panel.windows[w.window].fragment_begin + block].block_id
            : panel.block_begin + block;
        if (ordinal == 0) record.generation = generation;
        valid &= block < w.blocks && slot == expected_slot && column == col % D &&
                 generation == record.generation &&
                 carrier == carriers[original_block * carrier_stride +
                     op.j_group * plan.tile.tile_J * D + col];
        last_op = op.id;
        return w.index;
    }

    void add(api kind, size_t window, potal_status status, uint64_t elapsed) {
        totals[kind].add(status, elapsed);
        if (window != none) windows[window].measurements[kind].add(status, elapsed);
        ++sequence;
    }

    void completion(api kind, size_t op, potal_status status, uint64_t elapsed,
                    const potal_paired_session_status & before, const potal_paired_session_status & after) {
        if (completions.size() == completions.capacity()) {
            valid = false;
            return;
        }
        completions.push_back({ kind, op, sequence, elapsed, status, before, after });
    }
};

// Installed only on the thread executing the paired request. Issuer bring-up
// commands, before this scope, pass through unchanged and are not timed here.
thread_local diagnostic * active = nullptr;

uint64_t elapsed_ns(clock_type::time_point start) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() - start).count();
}

bool pending(const potal_paired_session_status & state) {
    return state.outstanding_loops || state.outstanding_readbacks || state.outstanding_acc_updates;
}

bool safe(const potal_paired_session_status & state) {
    return !state.failed && !pending(state) && !state.live_scale_slots && !state.index_transaction_open;
}

} // namespace

extern "C" potal_status __wrap_potal_paired_scale_write(
    potal_paired_session * session, uint32_t slot, uint32_t generation, uint32_t column, uint32_t carrier) {
    auto * d = active;
    const size_t window = d ? d->begin_write(slot, generation, column, carrier) : none;
    const auto start = clock_type::now();
    const auto status = __real_potal_paired_scale_write(session, slot, generation, column, carrier);
    const uint64_t elapsed = elapsed_ns(start);
    if (d) {
        d->add(scale_write, window, status, elapsed);
        if (window != none && status == POTAL_OK) {
            if (slot < POTAL_SCALE_ENTRIES && column < D) {
                d->valid &= d->live[slot][column].window == none;
                d->live[slot][column] = { window, generation };
            }
            const auto & w = d->plan.scale_windows[window];
            if (d->windows[window].measurements[scale_write].calls ==
                w.blocks * d->plan.residencies[w.residency].cols) ++d->program_cursor;
        }
    }
    return status;
}

extern "C" potal_status __wrap_potal_paired_scale_release(
    potal_paired_session * session, uint32_t slot, uint32_t generation, uint32_t column) {
    auto * d = active;
    const size_t window = d ? d->scale_key(slot, generation, column) : none;
    bool first = false;
    if (d && window != none) {
        auto & record = d->windows[window];
        d->last_op = d->plan.scale_windows[window].release_op;
        first = !record.release_seen;
        if (first) {
            record.release_seen = true;
            // Exactly one pre-Release outstanding query per Scale window,
            // never one per slot or column.
            record.before_release = potal_paired_session_query(session);
            if (d->prefence && pending(record.before_release)) {
                const auto start = clock_type::now();
                const auto status = __real_potal_paired_fence(session);
                const uint64_t elapsed = elapsed_ns(start);
                record.extra_fence.add(status, elapsed);
                d->extra_fences.add(status, elapsed);
                ++d->sequence;
                record.after_prefence = potal_paired_session_query(session);
                // A failed Fence makes the session sticky-failed. The unchanged
                // Release below sees that error. C11 cleanup refuses to retry a
                // failed Fence; the normal executor retains fatal DMA buffers.
            }
        }
    }
    const auto start = clock_type::now();
    const auto status = __real_potal_paired_scale_release(session, slot, generation, column);
    const uint64_t elapsed = elapsed_ns(start);
    if (d) {
        d->add(scale_release, window, status, elapsed);
        if (window != none) {
            if (first) {
                d->windows[window].first_release_ns = elapsed;
                d->windows[window].first_release_sequence = d->sequence;
            }
            if (status == POTAL_OK) d->live[slot][column] = {};
        }
    }
    return status;
}

extern "C" potal_status __wrap_potal_paired_index_config(
    potal_paired_session * session, const potal_index_config * config) {
    auto * d = active;
    size_t window = none;
    if (d) {
        d->valid &= d->current_index == none && d->index_cursor < d->indices.size();
        if (d->index_cursor < d->indices.size()) {
            const auto & op = d->plan.ops[d->indices[d->index_cursor++]];
            d->current_index = op.id;
            d->last_op = op.id;
            d->index_written = 0;
            window = d->index_windows[op.id];
            d->valid &= window != none && config && config->pair == op.pair &&
                config->b_sp_base == op.index_b_base && config->j_tiles == op.index_j_tiles &&
                config->source_k_rows == op.index_source_k_rows && config->active_rows == op.index_active_rows;
        }
    }
    const auto start = clock_type::now();
    const auto status = __real_potal_paired_index_config(session, config);
    const uint64_t elapsed = elapsed_ns(start);
    if (d) d->add(index_config, window, status, elapsed);
    return status;
}

extern "C" potal_status __wrap_potal_paired_index_write(
    potal_paired_session * session, const potal_index_write * write) {
    auto * d = active;
    size_t window = none;
    if (d) {
        if (d->current_index == none) {
            d->valid = false;
        } else {
            const auto & op = d->plan.ops[d->current_index];
            window = d->index_windows[op.id];
            if (d->index_written >= op.entry_end - op.entry_begin) {
                d->valid = false;
            } else {
                const auto & entry = d->plan.index_entries[op.entry_begin + d->index_written++];
                d->valid &= write && write->slot == entry.slot && write->resident_k == entry.resident_k;
            }
        }
    }
    const auto start = clock_type::now();
    const auto status = __real_potal_paired_index_write(session, write);
    const uint64_t elapsed = elapsed_ns(start);
    if (d) d->add(index_write, window, status, elapsed);
    return status;
}

extern "C" potal_status __wrap_potal_paired_loop_ws_issue(
    potal_paired_session * session, const potal_paired_loop_ws * loop) {
    auto * d = active;
    size_t window = none;
    if (d) {
        if (d->loop_cursor >= d->loops.size()) {
            d->valid = false;
        } else {
            const auto & op = d->plan.ops[d->loops[d->loop_cursor++]];
            window = op.scale_window;
            d->last_op = op.id;
            const bool residual = op.kind == potal_paired_op_kind::residual_gemm;
            d->valid &= loop && loop->pair == op.pair && loop->indexed == residual &&
                loop->resident_b == residual && loop->path == (residual ? POTAL_PAIRED_RESIDUAL : POTAL_PAIRED_MAIN) &&
                loop->a_offset_tiles == op.a_offset_tiles && loop->loop.i_tiles == op.i_tiles &&
                loop->loop.j_tiles == op.j_tiles && loop->loop.k_tiles == op.k_tiles &&
                loop->loop.pad_i == op.pad_i && loop->loop.pad_j == op.pad_j && loop->loop.pad_k == op.pad_k &&
                loop->loop.scale_base == d->plan.scale_windows[window].slot_base &&
                loop->loop.scale_generation == d->windows[window].generation &&
                loop->loop.flags == (op.first ? 0u : POTAL_GEMM_ACCUMULATE) &&
                (residual ? loop->loop.b == nullptr : loop->loop.b != nullptr);
            if (residual) {
                if (d->current_index == none) {
                    d->valid = false;
                } else {
                    const auto & index = d->plan.ops[d->current_index];
                    d->valid &= index.residency == op.residency && index.window == op.window &&
                               d->index_written == index.entry_end - index.entry_begin;
                }
                d->current_index = none;
            } else {
                d->valid &= d->current_index == none;
            }
        }
    }
    const auto start = clock_type::now();
    const auto status = __real_potal_paired_loop_ws_issue(session, loop);
    const uint64_t elapsed = elapsed_ns(start);
    if (d) d->add(loop_issue, window, status, elapsed);
    return status;
}

extern "C" potal_status __wrap_potal_paired_acc_readback_issue(
    potal_paired_session * session, const potal_acc_readback * result) {
    auto * d = active;
    size_t op_id = none, window = none;
    potal_paired_session_status before{};
    if (d) {
        before = potal_paired_session_query(session);
        if (d->read_cursor >= d->reads.size()) {
            d->valid = false;
        } else {
            op_id = d->reads[d->read_cursor++];
            const auto & op = d->plan.ops[op_id];
            const auto & tile = d->plan.tile;
            const size_t j_tiles = d->plan.residencies[op.residency].j_tiles;
            const size_t pad_j = op.j_group + 1 == tile.j_groups ? tile.pad_J : 0;
            d->last_op = op_id;
            d->valid &= result && result->acc_bank == op.acc_bank &&
                result->i_tiles == (op.acc_bank == 0 ? tile.tile_I : tile.residual_row_tiles) &&
                result->j_tiles == j_tiles &&
                result->pad_i == (op.acc_bank == 0 ? tile.pad_I : tile.residual_pad_I) && result->pad_j == pad_j;
            // Attribute epoch readback to its last contributing Scale window,
            // not to every window that contributed to the same ACC epoch.
            for (size_t id : d->loops) {
                const auto & contribution = d->plan.ops[id];
                if (id < op_id && contribution.acc_bank == op.acc_bank &&
                    contribution.acc_epoch == op.acc_epoch) window = contribution.scale_window;
            }
        }
    }
    const auto start = clock_type::now();
    const auto status = __real_potal_paired_acc_readback_issue(session, result);
    const uint64_t elapsed = elapsed_ns(start);
    if (d) {
        d->add(readback, window, status, elapsed);
        d->completion(readback, op_id, status, elapsed, before, potal_paired_session_query(session));
    }
    return status;
}

extern "C" potal_status __wrap_potal_paired_fence(potal_paired_session * session) {
    auto * d = active;
    const auto before = d ? potal_paired_session_query(session) : potal_paired_session_status{};
    const auto start = clock_type::now();
    const auto status = __real_potal_paired_fence(session);
    const uint64_t elapsed = elapsed_ns(start);
    if (d) {
        d->add(fence, none, status, elapsed);
        d->completion(fence, d->last_op, status, elapsed, before, potal_paired_session_query(session));
    }
    return status;
}

extern "C" potal_status __wrap_potal_paired_session_close(potal_paired_session * session) {
    auto * d = active;
    if (d) {
        d->close_seen = true;
        d->final_state = potal_paired_session_query(session);
    }
    const auto status = __real_potal_paired_session_close(session);
    if (d) d->close_status = status;
    return status;
}

namespace {

struct bench_case { const char * name; size_t m, n, k, step; };
struct free_dma { void operator()(void * p) const { std::free(p); } };

// Exact bench_one generation order and separate normal-distribution caches.
void fill_tensors(const bench_case & c, ggml_tensor * w, ggml_tensor * x) {
    std::mt19937 rng(20261023);
    std::normal_distribution<float> weight_normal(0.0f, 1.0f);
    std::vector<float> values(c.n * c.k);
    for (size_t n = 0; n < c.n; ++n) {
        for (size_t k = 0; k < c.k; ++k) {
            const float magnitude[] = { 1.0f, 8.0f, 0.125f, 64.0f };
            values[n * c.k + k] = weight_normal(rng) * 0.02f * magnitude[(k / 32 + n) % 4];
        }
    }
    ggml_quantize_chunk(GGML_TYPE_Q8_HP1, values.data(), w->data, 0, c.n, c.k, nullptr);
    std::normal_distribution<float> activation_normal(0.0f, 1.0f);
    auto * a = static_cast<float *>(x->data);
    for (size_t m = 0; m < c.m; ++m) {
        for (size_t k = 0; k < c.k; ++k) {
            a[m * c.k + k] = activation_normal(rng) * (1.0f + 0.5f * (float) (m % 3));
        }
        if (m % c.step == 0) {
            for (size_t b = 0; b < 2; ++b) {
                a[m * c.k + (m * 7 + b * 37) % c.k] = 400.0f * ((m + b) % 4 ? -1.0f : 1.0f);
            }
        }
    }
}

std::ostream & record(const bench_case & c, size_t stripe, const char * kind) {
    return std::cout << "POTAL_E3 case=" << c.name << " stripe=" << stripe << " record=" << kind;
}

void print_measurement(std::ostream & out, const measurement & m) {
    out << " calls=" << m.calls << " succeeded=" << m.succeeded << " cpu_api_ns=" << m.ns
        << " first_error=" << potal_status_name(m.first_error);
}

void print_diagnostic(const bench_case & c, size_t stripe, const diagnostic & d) {
    for (const auto & w : d.plan.scale_windows) {
        const auto & op = d.plan.ops[w.consumer_op];
        const auto & r = d.windows[w.index];
        record(c, stripe, "window") << " basis=logical path=" << (w.residual ? "residual" : "main")
            << " scale_window=" << w.index << " plan_generation=" << w.generation
            << " hardware_key_generation=" << r.generation << " slot_begin=" << w.slot_base
            << " slots=" << w.slots << " residency=" << w.residency << " pair=" << op.pair
            << " panel=" << op.panel << " window=" << w.window << " ACC_bank=" << op.acc_bank
            << " ACC_epoch=" << op.acc_epoch << " I_tiles=" << op.i_tiles << " J_tiles=" << op.j_tiles
            << " K_tiles=" << op.k_tiles << " lane_rows=" << (w.residual ? d.plan.residual_rows.size() : 0);
        if (w.residual) {
            const auto & index = d.plan.tile.panels[op.panel].windows[w.window];
            std::cout << " active_K=" << index.active_rows - index.filler_slots
                << " indexed_K=" << index.active_rows << " padded_K=" << index.k_tiles * D
                << " fillers=" << index.filler_slots << " tail_pad_K=" << index.pad_k;
        }
        std::cout << '\n';
        // Fences are global drain points, not exclusive work of one window.
        // Their sequence and preceding op are reported separately below.
        for (size_t kind = 0; kind < fence; ++kind) {
            auto & out = record(c, stripe, "window_api");
            out << " basis=command_declared scale_window=" << w.index << " api=" << api_names[kind];
            print_measurement(out, r.measurements[kind]);
            out << '\n';
        }
        record(c, stripe, "first_release") << " scale_window=" << w.index << " observed=" << r.release_seen
            << " sequence=" << r.first_release_sequence
            << " software_state=" << (!r.release_seen ? "Not_observed" :
                pending(r.before_release) ? "software_pending" : "software_idle")
            << " software_loops=" << r.before_release.outstanding_loops
            << " software_readbacks=" << r.before_release.outstanding_readbacks
            << " software_acc_updates=" << r.before_release.outstanding_acc_updates
            << " unchanged_Release_cpu_api_ns=" << r.first_release_ns
            << " extra_Fence_calls=" << r.extra_fence.calls << " extra_Fence_cpu_api_ns=" << r.extra_fence.ns
            << " extra_Fence_status=" << potal_status_name(r.extra_fence.first_error)
            << " extra_Fence_observed=" << (r.extra_fence.calls != 0)
            << " software_pending_after_extra_Fence=" << pending(r.after_prefence)
            << " allDrained_guaranteed_by_extra_Fence=" << (r.extra_fence.succeeded != 0)
            << " hardware_wait_category=Not_measured\n";
        const auto & panel = d.plan.tile.panels[op.panel];
        for (size_t block = 0; block < w.blocks; ++block) {
            const size_t original = w.residual
                ? d.plan.tile.fragments[panel.windows[w.window].fragment_begin + block].block_id
                : panel.block_begin + block;
            for (size_t col = 0; col < d.plan.residencies[w.residency].cols; ++col) {
                const size_t n = op.j_group * d.plan.tile.tile_J * D + col;
                record(c, stripe, "carrier") << " basis=logical scale_window=" << w.index
                    << " slot=" << w.slot_base + block * w.j_tiles + col / D << " column=" << col % D
                    << " original_block=" << original << " output_column=" << n
                    << " carrier=" << d.carriers[original * d.carrier_stride + n] << '\n';
            }
        }
    }
    for (const auto & op : d.plan.ops) {
        for (const auto & dep : op.deps) {
            record(c, stripe, "dependency") << " basis=logical op=" << op.id
                << " kind=" << potal_paired_op_name(op.kind) << " predecessor=" << dep.op
                << " completion_required=" << dep.complete << '\n';
        }
        if (op.kind != potal_paired_op_kind::index_program) continue;
        record(c, stripe, "index_program") << " basis=command_declared op=" << op.id
            << " residency=" << op.residency << " window=" << op.window << " pair=" << op.pair
            << " b_sp_base=" << op.index_b_base << " source_K_rows=" << op.index_source_k_rows
            << " active_rows=" << op.index_active_rows
            << " same_buffer_predecessor=" << op.same_buffer_predecessor << '\n';
        for (size_t i = op.entry_begin; i < op.entry_end; ++i) {
            const auto & entry = d.plan.index_entries[i];
            record(c, stripe, "index_entry") << " basis=logical op=" << op.id << " slot=" << entry.slot
                << " fragment=" << entry.fragment << " original_K=" << entry.original_k
                << " resident_K=" << entry.resident_k << " compact_K=" << entry.compact_k
                << " filler=" << entry.filler << '\n';
        }
    }
    for (size_t row = 0; row < d.plan.residual_rows.size(); ++row) {
        const auto & lane = d.plan.residual_rows[row];
        record(c, stripe, "lane") << " basis=logical row=" << row << " lane=" << (unsigned) lane.lane
            << " source_row=" << lane.source_row << '\n';
    }
    for (const auto & e : d.completions) {
        record(c, stripe, "completion") << " api=" << api_names[e.kind] << " sequence=" << e.sequence
            << " preceding_or_readback_op=" << e.op << " cpu_api_ns=" << e.ns
            << " status=" << potal_status_name(e.status)
            << " software_loops_before=" << e.before.outstanding_loops
            << " software_readbacks_before=" << e.before.outstanding_readbacks
            << " software_pending_after=" << pending(e.after)
            << " allDrained_guaranteed_by_successful_Fence=" << (e.kind == fence && e.status == POTAL_OK) << '\n';
    }
    for (size_t kind = 0; kind < api_count; ++kind) {
        auto & out = record(c, stripe, "api_total");
        out << " basis=command_declared api=" << api_names[kind];
        print_measurement(out, d.totals[kind]);
        out << '\n';
    }
    auto & out = record(c, stripe, "extra_Fence_total");
    out << " basis=command_declared intrusive=" << d.prefence;
    print_measurement(out, d.extra_fences);
    out << " extra_commands=" << d.extra_fences.calls
        << " extra_drains=" << d.extra_fences.succeeded << '\n';
}

bool run_case(const bench_case & c, bool hardware, bool prefence) {
    ggml_init_params ip = { 4 * (c.n * c.k + 2 * c.m * c.n + c.m * c.k) + 64 * 1024 * 1024, nullptr, false };
    ggml_context_ptr ctx(ggml_init(ip));
    if (!ctx) return false;
    auto * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, c.k, c.n);
    auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.k, c.m);
    fill_tensors(c, w, x);
    auto * y = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.n, c.m);
    const hp1_shape shape = { (int64_t) c.m, (int64_t) c.n, (int64_t) c.k };
    potal_tile_plan serial;
    ggml_gemmini_args_t args;
    hp1_operands operands;
    quants::weight::hp1::stats weight_stats;
    if (potal_tile_select(c.m, c.n, c.k, 32 / D, &serial) != POTAL_OK || !operands.allocate(shape) ||
        !init_args(w, y, shape, "", "golden", ggml::gemmini::residual::ResidualRoute::ws_packet, serial, args) ||
        unpack_weights(args, operands, &weight_stats) != quants::weight::hp1::status::ok ||
        !quantize_activations(x, args, operands)) {
        record(c, 0, "failure") << " stage=preparation\n";
        return false;
    }
    namespace exsia = ggml::gemmini::quants::act::exsia;
    const auto * meta = std::get_if<exsia::Meta>(&args.act_quant.storage());
    const size_t stripe_rows = serial.tile_I * D;
    const size_t stripes = (c.m + stripe_rows - 1) / stripe_rows;
    if (!meta || meta->theta.size() != stripes) return false;
    std::map<size_t, const ggml::gemmini::rmd::StripePacket *> packets;
    for (const auto & packet : meta->rmd_packets) {
        if (!packet || packet->stripe_id >= stripes || !packets.emplace(packet->stripe_id, packet.get()).second) return false;
    }
    if (packets.empty()) return false; // No diagnostic PASS without residual windows.

    // Independent existing Main golden, not the paired schedule or H2 emulator.
    std::vector<int32_t> main_golden(operands.m_pad * operands.ldc, 0);
    reference_gemm_i32(operands.a, operands.lda, operands.b, operands.ldb, operands.carriers,
                      operands.cstride, 32, c.m, c.n, c.k, main_golden.data(), operands.ldc);
    std::fill(operands.c, operands.c + operands.m_pad * operands.ldc, INT32_MIN);
    for (size_t stripe = 0; stripe < stripes; ++stripe) {
        const size_t row_begin = stripe * stripe_rows, rows = std::min(stripe_rows, c.m - row_begin);
        residual::decoded_packet decoded;
        const residual::decoded_packet * decoded_ptr = nullptr;
        std::vector<potal_test::golden_lane_row> residual_golden;
        const auto packet = packets.find(stripe);
        if (packet != packets.end()) {
            if (residual::decode_packet(*packet->second, operands.own, decoded) !=
                    ggml::gemmini::rmd::RmdStatus::success ||
                !potal_test::residual_golden(*packet->second, c.n, D,
                    [&](size_t k, size_t n) { return (int32_t) operands.b[k * operands.ldb + n]; },
                    [&](size_t block, size_t n) { return operands.carriers[block * operands.cstride + n]; },
                    residual_golden)) return false;
            decoded_ptr = &decoded;
        }
        potal_paired_tile_plan tile;
        potal_paired_execution_plan plan;
        std::string why;
        if (potal_paired_plan_stripe(serial, rows, c.n, c.k, decoded_ptr, tile, plan, &why) != potal_paired_status::ok) {
            record(c, stripe, "failure") << " stage=plan detail=" << why << '\n';
            return false;
        }
        std::unique_ptr<int32_t, free_dma> residual_c(static_cast<int32_t *>(
            dma_alloc(align_up(std::max<size_t>(decoded.rows.size(), 1), D) * operands.ldc * sizeof(int32_t))));
        if (!residual_c) return false;
        std::fill(residual_c.get(), residual_c.get() + align_up(std::max<size_t>(decoded.rows.size(), 1), D) *
                  operands.ldc, INT32_MIN);
        uint32_t generation = 0;
        potal_paired_request request;
        request.stripe_id = stripe;
        request.row_begin = row_begin;
        request.rows = rows;
        request.n = c.n;
        request.k = c.k;
        request.main_plan = &serial;
        request.a = operands.a + row_begin * operands.lda;
        request.lda = operands.lda;
        request.weight = &operands.own;
        request.residual = decoded_ptr;
        request.c_main = operands.c + row_begin * operands.ldc;
        request.ldc_main = operands.ldc;
        request.c_residual = residual_c.get();
        request.ldc_residual = operands.ldc;
        request.generation = &generation;
        request.plan = &plan;
        diagnostic d(plan, operands.own, prefence);
        potal_paired_report report;
        auto status = potal_paired_status::execution_failed;
        potal_test::h2_emulator emulator;
        bool started = true;
        potal_status owner_status = POTAL_OK;
        uint64_t executor_wall_ns = 0;
        if (hardware) {
#if defined(__riscv)
            issuer_owner owner;
            started = owner.start([&](const owner_runners & runners) {
                active = &d;
                const auto start = clock_type::now();
                status = runners.paired(request, report);
                executor_wall_ns = elapsed_ns(start);
                active = nullptr;
                return consume_outcome{ report.fatal, report.hw_status, report.stage };
            }, why, owner_status);
            owner.join();
#else
            started = false;
            owner_status = POTAL_ERR_NO_HARDWARE;
#endif
        } else {
            auto sink = emulator.sink();
            active = &d;
            const auto start = clock_type::now();
            status = potal_paired_run_session(request, &sink, report);
            executor_wall_ns = elapsed_ns(start);
            active = nullptr;
        }
        // Preserve every DMA operand before any reporting or comparison can
        // allocate or throw. The executor already preserves its own staging.
        const bool unsafe = report.fatal || (d.close_seen &&
            (pending(d.final_state) || d.final_state.live_scale_slots || d.close_status == POTAL_ERR_BUSY));
        if (unsafe) {
            residual_c.release();
            operands.abandon();
        }
        bool golden_ok = !unsafe && status == potal_paired_status::ok;
        if (golden_ok) {
            for (size_t row = 0; row < rows; ++row) {
                for (size_t n = 0; n < c.n; ++n) {
                    golden_ok &= request.c_main[row * operands.ldc + n] ==
                        main_golden[(row_begin + row) * operands.ldc + n];
                }
            }
            golden_ok &= residual_golden.size() == plan.residual_rows.size();
            for (size_t row = 0; row < plan.residual_rows.size(); ++row) {
                const auto & provenance = plan.residual_rows[row];
                const auto found = std::find_if(residual_golden.begin(), residual_golden.end(), [&](const auto & g) {
                    return g.lane == provenance.lane && g.row == provenance.source_row;
                });
                if (found == residual_golden.end()) { golden_ok = false; continue; }
                for (size_t n = 0; n < c.n; ++n) {
                    golden_ok &= request.c_residual[row * operands.ldc + n] == found->c[n];
                }
            }
        }
        bool counts_ok = report.counts.main_loop_ws == plan.main_gemms &&
            report.counts.residual_loop_ws == plan.residual_gemms &&
            d.totals[loop_issue].succeeded == plan.main_gemms + plan.residual_gemms &&
            d.totals[index_config].succeeded == plan.index_programs &&
            d.totals[index_write].succeeded == plan.index_writes &&
            d.totals[scale_write].succeeded == report.counts.scale_writes &&
            d.totals[scale_release].succeeded == report.counts.scale_releases &&
            d.totals[scale_write].succeeded == d.totals[scale_release].succeeded &&
            d.totals[readback].succeeded == report.counts.readbacks &&
            d.totals[fence].succeeded == report.counts.fences &&
            d.final_state.commands == 10 * d.totals[loop_issue].succeeded +
                d.totals[index_config].succeeded + d.totals[index_write].succeeded +
                d.totals[scale_write].succeeded + d.totals[scale_release].succeeded +
                d.totals[readback].succeeded + report.counts.readback_tiles +
                d.totals[fence].succeeded + d.extra_fences.succeeded + report.counts.tlb_flushes;
        for (const auto & w : plan.scale_windows) {
            const auto & r = d.windows[w.index];
            const size_t entries = w.blocks * plan.residencies[w.residency].cols;
            counts_ok &= r.measurements[scale_write].succeeded == entries &&
                r.measurements[scale_release].succeeded == entries && r.release_seen &&
                r.measurements[loop_issue].succeeded == 1 &&
                r.extra_fence.calls == (prefence && pending(r.before_release) ? 1u : 0u) &&
                r.extra_fence.calls == r.extra_fence.succeeded &&
                (!r.extra_fence.succeeded || !pending(r.after_prefence));
            if (w.residual) {
                const auto & op = plan.ops[w.consumer_op];
                const auto & index = plan.tile.panels[op.panel].windows[w.window];
                counts_ok &= r.measurements[index_config].succeeded == 1 &&
                    r.measurements[index_write].succeeded == index.active_rows;
            }
        }
        if (!hardware) {
            counts_ok &= emulator.errors.empty() && !emulator.scale_miss &&
                emulator.main_loops == plan.main_gemms && emulator.residual_loops == plan.residual_gemms &&
                emulator.b_loads == plan.main_gemms &&
                emulator.fences == d.totals[fence].succeeded + d.extra_fences.succeeded;
            for (const auto & slot : emulator.scales) {
                for (const auto & column : slot) counts_ok &= !column.valid;
            }
        }
        const bool ok = started && status == potal_paired_status::ok && !report.fatal &&
            d.valid && d.close_seen && safe(d.final_state) && d.close_status == POTAL_OK &&
            counts_ok && golden_ok && d.program_cursor == d.programs.size() &&
            d.loop_cursor == d.loops.size() && d.index_cursor == d.indices.size() &&
            d.read_cursor == d.reads.size() && d.current_index == none;
        print_diagnostic(c, stripe, d);
        record(c, stripe, "result") << " mode=" << (hardware ? "hardware" : "software-only_H2")
            << " result=" << (ok ? (hardware ? "Hardware_PASS" : "Software_only_H2_PASS") : "FAIL")
            << " status=" << potal_paired_status_name(status) << " owner_status=" << potal_status_name(owner_status)
            << " stage=" << (report.stage ? report.stage : "-") << " detail=" << report.detail
            << " mapping_ok=" << d.valid << " counts_ok=" << counts_ok << " INT32_golden_ok=" << golden_ok
            << " safe_final_state=" << (d.close_seen && safe(d.final_state) && d.close_status == POTAL_OK)
            << " raw_commands=" << d.final_state.commands << " raw_drains=" << d.final_state.drains
            << " instrumented_executor_wall_ns=" << executor_wall_ns
            << " intrusive=" << prefence << " production_latency=Not_measured"
            << " DMA_wait=Not_measured Execute_wait=Not_measured readback_wait=Not_measured"
            << " hardware_overlap=Not_measured accelerator_cycles=Not_measured"
            << " architectural_CPU_counters=Not_measured\n";
        if (!ok) return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    const bench_case cases[] = {
        { "decode_1x768x768", 1, 768, 768, 1 },
        { "gpt2_ffn_up_16x3072x768", 16, 3072, 768, 2 },
    };
    const char * name = nullptr;
    bool hardware = false, prefence = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--case") == 0 && i + 1 < argc && !name) name = argv[++i];
        else if (std::strcmp(argv[i], "--hardware") == 0 && !hardware) hardware = true;
        else if (std::strcmp(argv[i], "--prefence") == 0 && !prefence) prefence = true;
        else {
            std::cerr << "Usage: potal-e3-window-diag --case NAME [--hardware] [--prefence]\n";
            return 2;
        }
    }
    const bench_case * selected = nullptr;
    for (const auto & c : cases) if (name && std::strcmp(name, c.name) == 0) selected = &c;
    if (!selected) {
        std::cerr << "Choose --case decode_1x768x768 or --case gpt2_ffn_up_16x3072x768.\n";
        return 2;
    }
#if !defined(__riscv)
    if (hardware) {
        std::cerr << "--hardware requires RISC-V Linux and issuer_owner; no CPU fallback.\n";
        return 2;
    }
#endif
    const char * window_diag = std::getenv("POTAL_PAIRED_WINDOW_DIAG");
    const char * raw_trace = std::getenv("POTAL_RAW_TRACE");
    if ((window_diag && std::strcmp(window_diag, "1") == 0) || (raw_trace && raw_trace[0] == '1')) {
        std::cerr << "Unset POTAL_PAIRED_WINDOW_DIAG and POTAL_RAW_TRACE for deferred diagnostic output.\n";
        return 2;
    }
    ggml_time_init();
    try {
        return run_case(*selected, hardware, prefence) ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "POTAL_E3 FAIL exception=" << error.what() << '\n';
        return 1;
    }
}
