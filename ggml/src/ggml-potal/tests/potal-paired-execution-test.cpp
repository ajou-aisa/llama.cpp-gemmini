// Hardware-free tests of potal_paired_matmul_auto (matmul/paired-auto.hpp, H3-2B.1): the
// validated execution plan through the H3-2A paired session on the command-level H2
// emulator (support/paired-h2-emulator.hpp).
//
//   T  command translation: one Main LOOP_WS per (J group, panel), one Indexed LOOP_WS per
//      window, INDEX_CONFIG + ordered writes before each Indexed GEMM, B DRAM 0 and the
//      plan's A offset on Residual, bits 61-63, first / accumulate flags, readback tiles,
//      scale carriers and generations, no fence between Main and its Residual window,
//      no duplicate contribution
//   P  staging: window columns = Index slots, filler columns zero, lane rows in order
//   N  numerics: Main INT32, Residual INT32, radix INT64 and the merged F32 equal the
//      serialized STRIPE_PIPELINE reference (support/paired-support.hpp) bitwise, on
//      several geometries (panels, windows, J groups, tails, empty / zero residual)
//   F  failures: a hardware status makes the session fail with the stage; outstanding
//      work marks the report fatal; the host without a sink issues nothing
//   S  stripe pipeline (H3-2B.2): real ExSIA quantization through run_stripe_pipeline,
//      once on the serial route (planned GEMMs on the RTL-semantics emulator) and once
//      paired (paired runner on the H2 emulator): committed F32 bitwise equal; each
//      paired stripe ran once through the paired runner and never through the planned
//      one; paired event order; capacity refusals run serially as serial_fallback;
//      a missing paired runner and an injected fatal report fail the run uncommitted

#include "support/paired-support.hpp"
#include "support/paired-h2-emulator.hpp"
#include "support/stripe-owner.hpp"

#include "matmul/options.hpp"
#include "matmul/paired-auto.hpp"
#include "matmul/stripe.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <map>
#include <mutex>
#include <sys/wait.h>
#include <unistd.h>
#if defined(POTAL_TEST_CLOCK_OBSERVER)
#include <chrono>

static thread_local const bool * stream_clock_active = nullptr;
static thread_local uint64_t stream_clock_reads = 0;

// Wrap the actual libstdc++ entry, including builds that call syscall directly
// instead of libc clock_gettime. The real clock implementation remains intact.
extern "C" {
std::chrono::steady_clock::time_point __real__ZNSt6chrono3_V212steady_clock3nowEv() noexcept;
std::chrono::steady_clock::time_point __wrap__ZNSt6chrono3_V212steady_clock3nowEv() noexcept;
}
extern "C" std::chrono::steady_clock::time_point __wrap__ZNSt6chrono3_V212steady_clock3nowEv() noexcept {
    if (stream_clock_active && *stream_clock_active) ++stream_clock_reads;
    return __real__ZNSt6chrono3_V212steady_clock3nowEv();
}
#endif

using namespace ggml::potal;
using namespace potal_test;

namespace potal_test {
bool main_select_ok(const paired_fixture & f, potal_tile_plan & main);
}

namespace {

constexpr uint32_t PPB = 32 / PD;

struct run_result {
    potal_paired_report              report;
    potal_paired_status              status = potal_paired_status::invalid_argument;
    std::vector<int32_t>             main_c;     // rows x N
    std::vector<int32_t>             residual_c; // lane rows x N
    std::vector<residual::lane_row>  rows;
    std::unique_ptr<int32_t, dma_deleter> c_main, c_res;
    potal_paired_session_status      session;
    uint32_t                         generation = 0;
    uint64_t                         hot_clock_reads = 0;
};

// One stripe (stripe 0, rows = min(stripe_rows, M)) of a fixture through the new executor.
run_result run_paired(const paired_fixture & f, const potal_tile_plan & main, const paired_inputs & in, h2_emulator & emu,
                      bool with_sink = true, uint32_t generation = 0, bool conservative = false,
                      const potal_paired_sink * override_sink = nullptr, int scale_reuse = -1,
                      const potal_paired_execution_plan * plan = nullptr, bool prefetch = false,
                      bool early = false, potal_observation_counter_fn early_counter = nullptr,
                      potal_paired_stream_mode stream = potal_paired_stream_mode::off,
                      potal_observation_counter_fn stream_counter = nullptr,
                      const potal_paired_geometry * hw = nullptr) {
    run_result   r;
    const size_t rows = std::min(main.tile_I * PD, f.M);
    residual::decoded_packet d;
    const bool   has_res = in.packets[0] != nullptr &&
                         residual::decode_packet(*in.packets[0], f.weight, d) == grmd::RmdStatus::success;
    const size_t ldc = align_up(f.N, PD);
    r.c_main.reset(static_cast<int32_t *>(dma_alloc(align_up(rows, PD) * ldc * sizeof(int32_t))));
    r.c_res.reset(static_cast<int32_t *>(dma_alloc(align_up(std::max<size_t>(d.rows.size(), 1), PD) * ldc * sizeof(int32_t))));
    potal_paired_session * s    = nullptr;
    potal_paired_sink      sink = override_sink != nullptr ? *override_sink : emu.sink();
    if (potal_paired_session_open(with_sink ? &sink : nullptr, &s) != POTAL_OK) {
        return r;
    }
    potal_paired_request q;
    q.rows         = rows;
    q.n            = f.N;
    q.k            = f.K;
    q.main_plan    = &main;
    q.a            = f.a.get();
    q.lda          = f.lda;
    q.weight       = &f.weight;
    q.residual     = has_res ? &d : nullptr;
    q.c_main       = r.c_main.get();
    q.ldc_main     = ldc;
    q.c_residual   = r.c_res.get();
    q.ldc_residual = ldc;
    r.generation   = generation;
    q.generation   = &r.generation;
    q.conservative_fences = conservative;
    if (scale_reuse >= 0) q.ktile_scale_reuse = scale_reuse != 0;
    q.plan = plan;
    q.hw = hw;
    q.next_k_prefetch = prefetch;
    q.early_k_prefetch = early;
    q.early_counter = early ? (early_counter ? early_counter : h2_emulator::on_counter) : nullptr;
    q.stream = stream;
    q.stream_counter = stream != potal_paired_stream_mode::off ?
        (stream_counter ? stream_counter : h2_emulator::on_counter) : nullptr;
#if defined(POTAL_TEST_CLOCK_OBSERVER)
    stream_clock_reads = 0;
    stream_clock_active = &emu.stream_active;
#endif
    r.status       = potal_paired_matmul_auto(q, s, r.report);
#if defined(POTAL_TEST_CLOCK_OBSERVER)
    r.hot_clock_reads = stream_clock_reads;
    stream_clock_active = nullptr;
#endif
    r.session      = potal_paired_session_query(s);
    if (r.status == potal_paired_status::ok) {
        r.main_c.resize(rows * f.N);
        for (size_t m = 0; m < rows; ++m) {
            std::memcpy(r.main_c.data() + m * f.N, r.c_main.get() + m * ldc, f.N * sizeof(int32_t));
        }
        r.rows = r.report.plan.residual_rows;
        r.residual_c.resize(r.rows.size() * f.N);
        for (size_t i = 0; i < r.rows.size(); ++i) {
            std::memcpy(r.residual_c.data() + i * f.N, r.c_res.get() + i * ldc, f.N * sizeof(int32_t));
        }
    }
    if (r.report.fatal) {
        r.c_main.release(); // the accelerator may still write them: leaked like the production path
        r.c_res.release();
        // Test housekeeping only: the emulator answers the next FENCE, so the leaked state can be drained
        // and the accelerator handed to the next case (production disables the issuer instead).
        if (!prefetch && stream == potal_paired_stream_mode::off) {
            emu.fail_funct = 0xFFFFFFFFu;
            (void) potal_paired_fence(s);
        }
    }
    if (potal_paired_h2_query(s).active)
        (void) potal_paired_h2_close(&s);
    else
        (void) potal_paired_session_close(s);
    return r;
}

size_t count_funct(const h2_emulator & emu, uint32_t funct) {
    size_t n = 0;
    for (const potal_paired_cmd_record & r : emu.records) {
        n += r.funct == funct;
    }
    return n;
}

// Numerical comparison of one stripe against the serialized reference; returns the first differing stage.
std::string compare_stripe(const paired_fixture & f, const paired_inputs & in, const serialized_result & ref,
                           const run_result & r, size_t rows) {
    if (!ref.main_ok || !ref.legacy_ok) {
        return "reference";
    }
    for (size_t m = 0; m < rows; ++m) {
        for (size_t n = 0; n < f.N; ++n) {
            if (r.main_c[m * f.N + n] != ref.main_c[m * f.N + n]) {
                return "main_int32 at row " + std::to_string(m) + " col " + std::to_string(n);
            }
        }
    }
    const bool has_res = in.packets[0] != nullptr;
    if (has_res) {
        if (r.rows.size() != ref.rows[0].size()) {
            return "residual rows";
        }
        for (size_t i = 0; i < r.rows.size(); ++i) {
            if (r.rows[i].lane != ref.rows[0][i].lane || r.rows[i].source_row != ref.rows[0][i].source_row) {
                return "residual row order";
            }
            for (size_t n = 0; n < f.N; ++n) {
                if (r.residual_c[i * f.N + n] != ref.residual_c[0][i * f.N + n]) {
                    return "residual_int32 at lane row " + std::to_string(i) + " col " + std::to_string(n);
                }
            }
        }
        grmd::BlockScaledInt64Correction corr;
        if (residual::reconstruct_rows(*in.packets[0], r.rows, r.residual_c.data(), f.N, f.N, corr) != grmd::RmdStatus::success) {
            return "radix_reconstruct";
        }
        if (corr.values != ref.correction[0]) {
            return "radix_int64";
        }
        // Merge as stripe.cpp does: Main F32, then the checked HP1 merge.
        std::vector<float> out(f.M * f.N, 0.0f);
        const double       act = (double) std::ldexp(1.0f, in.theta[0]);
        for (size_t m = 0; m < rows; ++m) {
            reconstruct_row(r.main_c.data() + m * f.N, f.N, act, f.weight.scale, out.data() + m * f.N, 1);
        }
        ggml::gemmini::quants::act::exsia::StripeMetadataSnapshot snap;
        snap.theta = in.theta[0];
        if (residual::merge_prepared_hp1(f.weight, 0, rows, snap, corr, out.data(), f.M, f.N, 1, nullptr) != grmd::RmdStatus::success) {
            return "merge";
        }
        for (size_t m = 0; m < rows; ++m) {
            for (size_t n = 0; n < f.N; ++n) {
                if (std::memcmp(&out[m * f.N + n], &ref.legacy[m * f.N + n], sizeof(float)) != 0) {
                    return "final_f32 at row " + std::to_string(m) + " col " + std::to_string(n);
                }
            }
        }
    } else {
        std::vector<float> out(f.M * f.N, 0.0f);
        const double       act = (double) std::ldexp(1.0f, in.theta[0]);
        for (size_t m = 0; m < rows; ++m) {
            reconstruct_row(r.main_c.data() + m * f.N, f.N, act, f.weight.scale, out.data() + m * f.N, 1);
            for (size_t n = 0; n < f.N; ++n) {
                if (std::memcmp(&out[m * f.N + n], &ref.legacy[m * f.N + n], sizeof(float)) != 0) {
                    return "final_f32 (no residual) at row " + std::to_string(m);
                }
            }
        }
    }
    return "";
}

void print_report(const char * name, const run_result & r) {
    const potal_paired_counts & c = r.report.counts;
    std::printf("  %s: %s stage=%s hw=%s fatal=%d | main_loops=%llu residual_loops=%llu index=%llu/%llu scale=%llu/%llu "
                "readbacks=%llu(%llu tiles) fences=%llu drains=%llu ops=%llu gen=%u | tile_K=%zu panels=%zu windows=%zu j_groups=%zu %s\n",
                name, potal_paired_status_name(r.status), r.report.stage ? r.report.stage : "-",
                potal_status_name(r.report.hw_status), (int) r.report.fatal, (unsigned long long) c.main_loop_ws,
                (unsigned long long) c.residual_loop_ws, (unsigned long long) c.index_configs, (unsigned long long) c.index_writes,
                (unsigned long long) c.scale_writes, (unsigned long long) c.scale_releases, (unsigned long long) c.readbacks,
                (unsigned long long) c.readback_tiles, (unsigned long long) c.fences, (unsigned long long) c.drains,
                (unsigned long long) c.ops, r.report.last_generation, r.report.tile.tile_K, r.report.tile.k_panels,
                r.report.tile.residual_windows, r.report.tile.j_groups, r.report.detail.c_str());
}

struct case_spec {
    const char * name;
    size_t       M, N, K;
    uint32_t     seed;
    entries_fn   entries; // nullptr: no residual
};

void test_early_case(const paired_fixture & f, const potal_tile_plan & main, const paired_inputs & in,
                     const potal_paired_execution_plan & plan);
void test_stream_case(const paired_fixture & f, const potal_tile_plan & main, const paired_inputs & in,
                      const potal_paired_execution_plan & plan, const potal_paired_geometry * hw = nullptr);

// Entries of `fn` whose K lies inside the fixture's K.
entries_fn within_k(const entries_fn & fn, size_t K) {
    return [=](size_t s, size_t rows) {
        std::vector<res_entry> kept;
        for (const res_entry & x : fn(s, rows)) {
            if (x.k < K) {
                kept.push_back(x);
            }
        }
        return kept;
    };
}

// Residual entries: a few rows, K selected per the given stride / block pattern.
entries_fn dense_entries(size_t row_step, size_t k_step, int32_t amplitude) {
    return [=](size_t, size_t rows) {
        std::vector<res_entry> e;
        for (size_t row = 0; row < rows; row += row_step) {
            for (size_t k = row % 3; k < 100000; k += k_step) {
                if (k >= 4096) {
                    break;
                }
                e.push_back({ row, k, (int32_t) ((int) (row * 31 + k * 17) % (2 * amplitude + 1) - amplitude) });
            }
        }
        return e;
    };
}

bool run_case(const case_spec & c, bool verbose, std::string & why, run_result * keep = nullptr, h2_emulator * keep_emu = nullptr,
              bool conservative = false, bool early_compare = false) {
    paired_fixture f(c.M, c.N, c.K, c.seed, [](size_t blk, size_t n) { return (int16_t) ((blk * 7 + n) % 4); });
    potal_tile_plan main;
    if (!f.ok || !main_select_ok(f, main)) {
        why = "fixture";
        return false;
    }
    paired_inputs in;
    if (!make_inputs(f, main.tile_I * PD, c.entries ? within_k(c.entries, f.K) : entries_fn{}, in)) {
        why = "inputs";
        return false;
    }
    const serialized_result ref = serialized_of(f, main.tile_I * PD, in);
    h2_emulator             local_emu;
    h2_emulator &           emu = keep_emu != nullptr ? *keep_emu : local_emu;
    run_result              r   = run_paired(f, main, in, emu, true, 0, conservative);
    if (verbose) {
        print_report(c.name, r);
    }
    if (r.status != potal_paired_status::ok) {
        why = std::string("executor: ") + (r.report.stage ? r.report.stage : "?") + " " + r.report.detail;
        return false;
    }
    if (!emu.errors.empty() || emu.scale_miss) {
        why = "emulator: " + emu.errors + (emu.scale_miss ? " scale miss" : "");
        return false;
    }
    const size_t rows = std::min(main.tile_I * PD, f.M);
    why               = compare_stripe(f, in, ref, r, rows);
    if (early_compare && PD == 32 && why.empty()) {
        test_early_case(f, main, in, r.report.plan);
        test_stream_case(f, main, in, r.report.plan);
    }
    if (keep != nullptr) {
        *keep = std::move(r);
    }
    return why.empty();
}

} // namespace

// potal_tile_select on the fixture's shape (the Main auto-tiler).
bool potal_test::main_select_ok(const paired_fixture & f, potal_tile_plan & main) {
    return potal_tile_select(f.M, f.N, f.K, PPB, &main) == POTAL_OK;
}

namespace {

void test_translation() {
    // 64 x 96 x 256: one J group? tile_J from the Main tiler; residual in blocks 0..7 (stripe rows 64).
    const case_spec c = { "T", 64, 96, 256, 5, dense_entries(5, 7, 200) };
    std::string     why;
    run_result      r;
    h2_emulator     emu;
    const bool      ok = run_case(c, true, why, &r, &emu);
    check("T the translated stream computes the serialized reference (" + why + ")", ok);
    if (!ok) {
        return;
    }
    const potal_paired_tile_plan & t = r.report.tile;
    const potal_paired_counts &    n = r.report.counts;
    check("T one Main LOOP_WS per (J group, K panel), one Indexed LOOP_WS per (J group, window)",
          n.main_loop_ws == t.j_groups * t.k_panels && n.residual_loop_ws == t.j_groups * t.residual_windows &&
              emu.main_loops == n.main_loop_ws && emu.residual_loops == n.residual_loop_ws);
    check("T one INDEX_CONFIG per Indexed GEMM, writes = the plan's entries, no duplicate B load",
          n.index_configs == n.residual_loop_ws && n.index_writes == r.report.plan.index_writes &&
              emu.index_writes == n.index_writes && emu.b_loads == n.main_loop_ws);
    // Order: every Indexed LOOP_WS is preceded (since the previous LOOP_WS) by one INDEX_CONFIG and its writes.
    bool   order = true, residual_flags = true, main_flags = true;
    size_t configs_since = 0, writes_since = 0;
    size_t first_main = 0, first_res = 0;
    for (const potal_paired_cmd_record & rec : emu.records) {
        if (rec.funct == POTAL_FUNCT_INDEX_CONFIG) {
            configs_since++;
        } else if (rec.funct == POTAL_FUNCT_INDEX_WRITE) {
            writes_since++;
        } else if (rec.funct == POTAL_FUNCT_LOOP_WS) {
            const bool residual = ((rec.rs1 >> 62) & 1) != 0;
            if (residual) {
                order = order && configs_since == 1 && writes_since > 0 && writes_since <= POTAL_HW_INDEX_ENTRIES;
                residual_flags = residual_flags && ((rec.rs1 >> 61) & 7) == 7 && (rec.rs2 >> 3) == t.residual_a_offset_tiles &&
                                 emu.b_dram == 0;
                first_res += (rec.rs1 & 1) == 0;
            } else {
                order      = order && configs_since == 0;
                main_flags = main_flags && ((rec.rs1 >> 61) & 7) == 1 && rec.rs2 == 0;
                first_main += (rec.rs1 & 1) == 0;
            }
            configs_since = writes_since = 0;
        }
    }
    check("T INDEX_CONFIG + writes directly before each Indexed LOOP_WS, never before a Main one", order);
    check("T Residual LOOP_WS: bits 61-63 set, rs2[31:3] = the plan's A offset, B DRAM 0 (resident weight)", residual_flags);
    check("T Main LOOP_WS: paired bit only, A offset 0", main_flags);
    check("T first contribution (accumulate 0) once per ACC epoch for Main and Residual",
          first_main == t.j_groups && first_res == t.j_groups);
    check("T readback: one per ACC epoch with the epoch's tiles, after a final fence",
          n.readbacks == 2 * t.j_groups && emu.mvouts == n.readback_tiles && emu.records.back().funct == POTAL_FUNCT_FENCE);
    check("T scale: every window written and released (counts equal), generations 1.." + std::to_string(r.report.last_generation),
          n.scale_writes == n.scale_releases && emu.scale_writes == emu.scale_releases && r.report.last_generation == r.report.plan.scale_windows.size());
    // No fence between a Main LOOP_WS and the following Indexed LOOP_WS of the same residency (shared stream).
    bool shared = true;
    for (size_t i = 0; i + 1 < emu.records.size(); ++i) {
        if (emu.records[i].funct == POTAL_FUNCT_LOOP_WS && ((emu.records[i].rs1 >> 62) & 1) == 0) {
            for (size_t j = i + 1; j < emu.records.size(); ++j) {
                if (emu.records[j].funct == POTAL_FUNCT_LOOP_WS) {
                    break;
                }
                shared = shared && emu.records[j].funct != POTAL_FUNCT_FENCE && emu.records[j].funct != POTAL_FUNCT_SCALE_RELEASE;
            }
        }
    }
    check("T no FENCE / RELEASE between a Main LOOP_WS and its panel's first Indexed LOOP_WS", shared && t.residual_windows > 0);
    check("T explicit fences: only the final drain; every released scale window is one drain point",
          n.fences == 1 && n.drains == r.report.plan.scale_windows.size() + n.fences);
    check("T session idle at the end: nothing outstanding, no open transaction, not failed",
          r.session.outstanding_loops == 0 && r.session.outstanding_readbacks == 0 && r.session.index_transaction_open == 0 &&
              r.session.failed == 0);
}

void test_staging() {
    // Runs of PD + PD/4 (tail), 3 (filler before the next fragment), 32: fillers must be zero digits.
    paired_fixture f(16, 32, 128, 9, [](size_t blk, size_t n) { return (int16_t) ((blk + n) % 3); });
    potal_tile_plan main;
    bool            ok = f.ok && main_select_ok(f, main);
    std::vector<res_entry> entries;
    for (size_t k = 0; k < (PD == 32 ? 32u : 20u); ++k) {
        entries.push_back({ 0, k, (int32_t) (k + 1) });
    }
    if (PD == 32) {
        // A K32 block holds exactly one DIM32 fragment. A separate block
        // supplies the 8-K tail; the next short block must produce fillers.
        for (size_t k = 32; k < 40; ++k) entries.push_back({ 0, k, 7 });
    }
    const size_t short_begin = PD == 32 ? 64 : 32;
    for (size_t k = short_begin; k < short_begin + 3; ++k) {
        entries.push_back({ 3, k, (int32_t) -5 });
    }
    const size_t full_begin = PD == 32 ? 96 : 64;
    for (size_t k = full_begin; k < full_begin + 32; ++k) {
        entries.push_back({ 5, k, (int32_t) (300 + k) }); // two lanes
    }
    const grmd::StripePacketHandle packet = ok ? make_packet(0, 0, 16, f.K, f.N, entries) : nullptr;
    residual::decoded_packet       d;
    potal_paired_tile_plan         tile;
    potal_paired_execution_plan    plan;
    potal_paired_staging           st;
    std::string                    why;
    ok = ok && packet != nullptr && residual::decode_packet(*packet, f.weight, d) == grmd::RmdStatus::success &&
         potal_make_tile_plan(main, 16, f.N, f.K, &d, tile, &why) == potal_paired_status::ok &&
         potal_make_execution_plan(tile, &d, plan, &why) == potal_paired_status::ok &&
         potal_paired_pack_residual(plan, d, st, &why) == potal_paired_status::ok;
    check("P staging of a packet with a tail and a short fragment (" + why + ")", ok);
    if (!ok) {
        return;
    }
    bool columns = st.windows.size() == tile.residual_windows && st.rows == d.rows.size();
    size_t fillers = 0, packed = 0;
    for (const potal_paired_op & op : plan.ops) {
        if (op.kind != potal_paired_op_kind::index_program || op.j_group != 0) {
            continue;
        }
        size_t wi = 0;
        while (wi < st.windows.size() && !(st.windows[wi].panel == op.panel && st.windows[wi].index == op.window)) {
            ++wi;
        }
        const potal_paired_staging::window & sw = st.windows[wi];
        columns = columns && sw.lda % POTAL_HW_MAX_BYTES == 0 && sw.cols == tile.panels[op.panel].windows[op.window].k_tiles * PD;
        std::vector<uint8_t> used(sw.cols, 0);
        for (size_t e = op.entry_begin; e < op.entry_end; ++e) {
            const potal_paired_index_entry & en = plan.index_entries[e];
            used[en.slot] = 1;
            for (size_t r = 0; r < st.rows; ++r) {
                const int8_t v = st.a(wi)[r * sw.lda + en.slot];
                columns = columns && (en.filler ? v == 0 : v == d.digits[r * d.k_compact + en.compact_k]);
            }
            fillers += en.filler;
            packed += !en.filler;
        }
        for (size_t s = 0; s < sw.cols; ++s) {
            for (size_t r = 0; r < st.rows && !used[s]; ++r) {
                columns = columns && st.a(wi)[r * sw.lda + s] == 0; // pad_k tail
            }
        }
    }
    check("P every window column = the digits of its Index slot, filler and tail columns zero, DMA-aligned rows",
          columns && fillers == tile.residual_filler_slots && packed + fillers == plan.index_writes / tile.j_groups && fillers > 0);
    potal_paired_staging fixed;
    const auto fixed_status = potal_paired_pack_residual(plan, d, fixed, &why, potal_paired_hw_geometry(), true);
    size_t pitch = 0;
    for (const auto & window : st.windows) pitch = std::max(pitch, window.lda);
    bool fixed_ok = fixed_status == potal_paired_status::ok && fixed.rows == st.rows &&
                    fixed.windows.size() == st.windows.size();
    for (size_t w = 0; fixed_ok && w < st.windows.size(); ++w) {
        const auto & old = st.windows[w];
        const auto & now = fixed.windows[w];
        fixed_ok = now.panel == old.panel && now.index == old.index && now.cols == old.cols &&
                   now.lda == pitch && now.lda % POTAL_HW_MAX_BYTES == 0;
        for (size_t row = 0; row < fixed.rows; ++row) {
            for (size_t col = 0; col < pitch; ++col) {
                const int8_t want = col < old.cols ? st.a(w)[row * old.lda + col] : 0;
                fixed_ok = fixed_ok && fixed.a(w)[row * pitch + col] == want;
            }
        }
    }
    check("H SOFTWARE-ONLY fixed pitch preserves compact digits/order and zeros every padding byte", fixed_ok);
    if (PD == 32) {
        check("P DIM32 staging: four legal K32 runs, windows 40/64 active rows, 24/0 pad, 29 filler columns",
              tile.fragments.size() == 4 && tile.k_panels == 1 && tile.residual_windows == 2 &&
                  tile.panels[0].windows[0].active_rows == 40 && tile.panels[0].windows[0].pad_k == 24 &&
                  tile.panels[0].windows[1].active_rows == 64 && tile.panels[0].windows[1].pad_k == 0 &&
                  fillers == 29 && packed == 75);
    }
    // The executor on this packet computes the reference as well.
    paired_inputs in;
    in.packets = { packet };
    in.theta   = { -2 };
    in.bind();
    const serialized_result ref = serialized_of(f, main.tile_I * PD, in);
    h2_emulator             emu;
    const run_result        r = run_paired(f, main, in, emu);
    print_report("P", r);
    check("P tail / filler packet: executor == reference (" + (r.status == potal_paired_status::ok ? compare_stripe(f, in, ref, r, 16) : std::string("status")) + ")",
          r.status == potal_paired_status::ok && emu.errors.empty() && compare_stripe(f, in, ref, r, 16).empty());
}

void test_numerics() {
    const case_spec cases[] = {
        { "N single panel, no residual", 48, 48, 160, 1, nullptr },
        { "N single panel, one window", 48, 48, 160, 2, dense_entries(7, 4096, 100) },
        { "N several panels (K 2048), several J groups, residual spanning panels", 80, 160, 2048, 3, dense_entries(9, 61, 150) },
        { "N many windows per panel (dense residual)", 16, 96, 768, 4, dense_entries(1, 1, 60) },
        { "N tails: I/J/K padding", PD == 32 ? 54u : 70u, 140, 2048, 6, dense_entries(11, 97, 120) },
        { "N large residual values (two lanes, saturation range)", 32, 64, 256, 7, dense_entries(2, 3, 30000) },
        { "N decode shape M = 1", 1, 768, 768, 8, dense_entries(1, 5, 90) },
        { "N multi-panel without Residual", 65, 97, 2080, 129, nullptr },
        { "N sparse first panel, later panels without Residual", 80, 160, 2048, 3, dense_entries(9, 4096, 150) },
        { "N full Main ACC half without Residual", 256, 32, 256, 229, nullptr },
        { "N full resident B bank without Residual", 1, 32, 2048, 233, nullptr },
        { "N full Residual ACC half", 128, 32, 64, 239, dense_entries(1, 1, 30000) },
    };
    for (const case_spec & c : cases) {
        std::string why;
        run_result  r;
        const bool  ok = run_case(c, true, why, &r, nullptr, false, true);
        check(std::string(c.name) + ": Main INT32, Residual INT32, radix INT64, final F32 == serialized reference (" + why + ")", ok);
        if (PD == 32 && c.seed == 229)
            check("H Main ACC half capacity reached exactly", r.report.tile.main_acc_rows == POTAL_HW_ACC_ROWS / 2);
        if (PD == 32 && c.seed == 233)
            check("H full Main A and B banks reached exactly",
                  r.report.tile.main_a_rows == POTAL_HW_BANK_ROWS && r.report.tile.weight_rows == POTAL_HW_BANK_ROWS);
        if (PD == 32 && c.seed == 239)
            check("H Residual ACC half capacity reached exactly", r.report.tile.residual_acc_rows == POTAL_HW_ACC_ROWS / 2);
        // Geometry the case name claims: explicit DIM-specific constants,
        // never expectations copied from the plan under test.
        const std::string name = c.name;
        if (name == "N single panel, one window") {
            check("N one panel and one window actually execute",
                  ok && r.report.tile.k_panels == 1 && r.report.tile.j_groups == 1 &&
                      r.report.tile.residual_windows == 1 && r.report.counts.main_loop_ws == 1 &&
                      r.report.counts.residual_loop_ws == 1);
        }
        if (PD == 16 && name.rfind("N several panels", 0) == 0) {
            check("N several panels: 3 K panels x 2 J groups = exactly 6 Main LOOP_WS, more than one window",
                  ok && r.report.tile.k_panels == 3 && r.report.tile.j_groups == 2 && r.report.counts.main_loop_ws == 6 &&
                      r.report.tile.residual_windows > 1);
        }
        if (PD == 32 && name.rfind("N several panels", 0) == 0) {
            check("N DIM32 several panels: I2/J2/K31, panels 31/31/2, three J groups, nine Main LOOP_WS",
                  ok && r.report.tile.tile_I == 2 && r.report.tile.tile_J == 2 && r.report.tile.tile_K == 31 &&
                      r.report.tile.k_panels == 3 && r.report.tile.j_groups == 3 &&
                      r.report.tile.panels[2].k_tiles == 2 && r.report.counts.main_loop_ws == 9 &&
                      r.report.tile.residual_windows > 1);
        }
        if (name.rfind("N many windows", 0) == 0) {
            size_t most = 0;
            for (const potal_paired_k_panel & p : r.report.tile.panels) {
                most = std::max(most, p.windows.size());
            }
            check("N many windows: one panel holds several Index Windows", ok && most > 2);
            if (PD == 32) {
                check("N DIM32 dense windows: I1/J3/K21, panels 21/3, windows 11/2",
                      ok && r.report.tile.tile_I == 1 && r.report.tile.tile_J == 3 && r.report.tile.tile_K == 21 &&
                          r.report.tile.k_panels == 2 && r.report.tile.panels[0].windows.size() == 11 &&
                          r.report.tile.panels[1].windows.size() == 2 && r.report.tile.residual_windows == 13);
            }
        }
        if (PD == 16 && name.rfind("N tails", 0) == 0) {
            check("N tails: J groups of 5 and 4 tiles (pad_j 4), I tail pad 10",
                  ok && r.report.tile.j_groups == 2 && r.report.tile.pad_J == 4 && r.report.tile.pad_I == 10);
        }
        if (PD == 32 && name.rfind("N tails", 0) == 0) {
            check("N DIM32 tails: I pad10, J groups 2/2/1 pad20, K panels 31/31/2",
                  ok && r.report.tile.tile_I == 2 && r.report.tile.tile_J == 2 && r.report.tile.tile_K == 31 &&
                      r.report.tile.j_groups == 3 && r.report.tile.pad_J == 20 && r.report.tile.pad_I == 10 &&
                      r.report.tile.k_panels == 3 && r.report.tile.panels[2].k_tiles == 2);
        }
    }
    // Zero residual fragments: a packet whose digits are nonzero only in one K of a run of 20.
    {
        paired_fixture f(16, 32, 128, 12, [](size_t blk, size_t n) { return (int16_t) ((blk + n) % 2); });
        potal_tile_plan main;
        bool            ok = f.ok && main_select_ok(f, main);
        std::vector<res_entry> entries = { { 2, 17, 9 } }; // block 0: one selected K
        for (size_t k = 32; k < 40; ++k) {
            entries.push_back({ 4, k, (int32_t) (k * 3) });
        }
        paired_inputs in;
        in.packets = { ok ? make_packet(0, 0, 16, f.K, f.N, entries) : nullptr };
        in.theta   = { 1 };
        in.bind();
        const serialized_result ref = serialized_of(f, main.tile_I * PD, in);
        h2_emulator             emu;
        const run_result        r = run_paired(f, main, in, emu);
        print_report("N sparse", r);
        check("N sparse packet (single K in a block): executor == reference",
              ok && in.packets[0] != nullptr && r.status == potal_paired_status::ok && compare_stripe(f, in, ref, r, 16).empty());
    }
}

void test_early_case(const paired_fixture & f, const potal_tile_plan & main, const paired_inputs & in,
                     const potal_paired_execution_plan & plan) {
    // Same immutable manifest, generations, tensors and independently computed
    // goldens. No expected result comes from the paired planner or lowering.
    const golden_result golden = golden_of(f, main.tile_I * PD, in);
    serialized_result expected;
    expected.main_ok = expected.legacy_ok = golden.ok;
    expected.main_c = golden.main_c;
    expected.legacy = golden.legacy;
    expected.correction = golden.correction;
    expected.rows.resize(in.theta.size());
    expected.residual_c.resize(in.theta.size());
    for (const auto & row : golden.residual[0]) {
        expected.rows[0].push_back({row.lane, row.row});
        expected.residual_c[0].insert(expected.residual_c[0].end(), row.c.begin(), row.c.end());
    }
    h2_emulator baseline_emu, early_emu;
    const run_result baseline = run_paired(f, main, in, baseline_emu, true, 250, false, nullptr, 1, &plan);
    const run_result early = run_paired(f, main, in, early_emu, true, 250, false, nullptr, 1, &plan, true, true);
    const bool ok = golden.ok && baseline.status == potal_paired_status::ok && early.status == potal_paired_status::ok;
    check("G baseline and early execute same manifest (" + std::to_string(f.M) + "x" +
          std::to_string(f.N) + "x" + std::to_string(f.K) + ")", ok);
    if (!ok) {
        print_report("G baseline", baseline);
        print_report("G early", early);
        return;
    }
    const size_t rows = std::min(main.tile_I * PD, f.M);
    check("G exact raw Main/Residual INT32, radix INT64 and merged F32 equal independent goldens",
          compare_stripe(f, in, expected, baseline, rows).empty() &&
          compare_stripe(f, in, expected, early, rows).empty() &&
          same_bits(baseline.main_c, early.main_c) && same_bits(baseline.residual_c, early.residual_c));
    const size_t admissions = early.report.tile.j_groups * (early.report.tile.k_panels - 1);
    const auto & b = baseline.report.counts;
    const auto & e = early.report.counts;
    check("G counts are CPU admissions, never raw MVINs or duplicate B transfers",
          e.prefetch_loads == admissions && e.prefetch_ldb_only == admissions &&
          early_emu.ldb_only_loads == admissions && e.resident_main_loop_ws == admissions &&
          early_emu.resident_main_loops == admissions && early_emu.raw_b_mvins == 0 && e.prefetch_mvins == 0 &&
          early_emu.b_loads + admissions == early_emu.main_loops &&
          early_emu.b_loads == early.report.tile.j_groups &&
          early_emu.ldb_only_bytes == e.prefetch_bytes_declared);
    check("G retains old Scale releases, windows, generations, drains and final readbacks",
          b.scale_writes == e.scale_writes && b.scale_releases == e.scale_releases &&
          b.scale_reused_windows == e.scale_reused_windows && b.scale_reused_columns == e.scale_reused_columns &&
          b.main_loop_ws == e.main_loop_ws && b.residual_loop_ws == e.residual_loop_ws &&
          b.index_configs == e.index_configs && b.index_writes == e.index_writes &&
          b.readbacks == e.readbacks && b.readback_tiles == e.readback_tiles &&
          e.fences == b.fences + admissions && e.drains == b.drains + admissions &&
          early.session.drains == baseline.session.drains + admissions &&
          early.generation == baseline.generation && e.ops == b.ops);
    // No wall-clock thresholds: the combined interval is included in issue_s,
    // not added again to fence_s or the old Residual combined interval.
    check("G combined Main interval is included once, not in Residual interval",
          e.main_prefetch_issue_s >= 0 && e.main_prefetch_issue_s <= e.issue_s &&
          e.residual_prefetch_issue_s == 0 && b.main_prefetch_issue_s == 0);

    bool order = true, pair_seen[2] = {false, false};
    size_t main_ordinal = 0, seen_admissions = 0;
    for (size_t i = 0; i < early_emu.records.size(); ++i) {
        const auto & command = early_emu.records[i];
        if (command.funct != POTAL_FUNCT_LOOP_WS || (command.rs1 >> 62 & 1)) continue;
        const size_t panel_index = main_ordinal % early.report.tile.k_panels;
        const size_t group = main_ordinal++ / early.report.tile.k_panels;
        if (panel_index + 1 == early.report.tile.k_panels) continue;
        if (i + 7 >= early_emu.records.size()) { order = false; break; }
        const auto & next = early.report.tile.panels[panel_index + 1];
        const uint64_t pair = ((command.rs1 >> 16) & 3) == 1 ? 1 : 0;
        pair_seen[pair] = true;
        const uint64_t j_begin = group * early.report.tile.tile_J * PD;
        const uint64_t columns = std::min(early.report.tile.tile_J * PD, f.N - j_begin);
        const uint64_t J = (columns + PD - 1) / PD, pad_j = J * PD - columns;
        const potal_paired_cmd_record words[] = {
            {9, 0, pad_j << 16, (uint64_t(next.k_tiles) << 32) | (J << 16) | 1},
            {10, 0, 0, reinterpret_cast<uintptr_t>(f.weight.b) + next.k_begin * f.weight.ldb + j_begin},
            {11, 0, 0, 0}, {12, 0, 0, f.weight.ldb}, {13, 0, 0, 0},
            {29, 1, UINT64_C(0x10) | pair, 0}, {POTAL_FUNCT_FENCE, 1, 0, 0},
        };
        for (size_t w = 0; w < 7; ++w) {
            const auto & got = early_emu.records[i + 1 + w];
            order = order && got.funct == words[w].funct && got.xd == words[w].xd &&
                    got.rs1 == words[w].rs1 && got.rs2 == words[w].rs2;
        }
        ++seen_admissions;
    }
    check("G Main immediately followed by exact next-B words and checked Fence before consumers",
          order && seen_admissions == admissions && main_ordinal == e.main_loop_ws);
    if (early.report.tile.k_panels > 2)
        check("G multi-K admission uses both opposite pairs", pair_seen[0] && pair_seen[1]);

    // Remove only the proven extra descriptor/Fence; relocate owned DMA
    // buffers and restore the already-checked suppressed Main B address.
    // Every old Fence/Release and Main/Residual word must remain in order.
    const auto canonical = [&](const h2_emulator & emulator, const run_result & result, bool enabled) {
        std::vector<potal_paired_cmd_record> records;
        uint64_t staging = UINT64_MAX;
        for (size_t i = 0; i + 4 < emulator.records.size(); ++i)
            if (emulator.records[i].funct == POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB &&
                emulator.records[i + 4].funct == POTAL_FUNCT_LOOP_WS &&
                (emulator.records[i + 4].rs1 >> 62 & 1))
                staging = std::min(staging, emulator.records[i].rs1);
        const uint64_t cm = reinterpret_cast<uintptr_t>(result.c_main.get());
        const uint64_t cr = reinterpret_cast<uintptr_t>(result.c_res.get());
        const uint64_t main_bytes = align_up(rows, PD) * align_up(f.N, PD) * sizeof(int32_t);
        size_t main_index = 0;
        for (size_t i = 0; i < emulator.records.size(); ++i) {
            auto record = emulator.records[i];
            if (record.funct == POTAL_FUNCT_COUNTER) continue;
            if (enabled && i + 6 < emulator.records.size() &&
                record.funct == POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS &&
                emulator.records[i + 5].funct == POTAL_FUNCT_LDB_ONLY) {
                i += 6;
                continue;
            }
            if (record.funct == POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB && i + 4 < emulator.records.size() &&
                emulator.records[i + 4].funct == POTAL_FUNCT_LOOP_WS) {
                if (emulator.records[i + 4].rs1 >> 62 & 1) record.rs1 -= staging;
                else {
                    const size_t panel = main_index % result.report.tile.k_panels;
                    const size_t group = main_index++ / result.report.tile.k_panels;
                    const uint64_t real_b = reinterpret_cast<uintptr_t>(f.weight.b) +
                        result.report.tile.panels[panel].k_begin * f.weight.ldb +
                        group * result.report.tile.tile_J * PD;
                    order = order && record.rs2 == (enabled && panel ? 0 : real_b);
                    record.rs2 = real_b;
                }
            }
            if (record.funct == POTAL_FUNCT_MVOUT)
                record.rs1 = record.rs1 >= cm && record.rs1 < cm + main_bytes ?
                             record.rs1 - cm : (UINT64_C(1) << 63) | (record.rs1 - cr);
            records.push_back(record);
        }
        return records;
    };
    const auto old_words = canonical(baseline_emu, baseline, false);
    const auto new_words = canonical(early_emu, early, true);
    bool identical = old_words.size() == new_words.size();
    for (size_t i = 0; identical && i < old_words.size(); ++i)
        identical = old_words[i].funct == new_words[i].funct && old_words[i].xd == new_words[i].xd &&
                    old_words[i].rs1 == new_words[i].rs1 && old_words[i].rs2 == new_words[i].rs2;
    check("G every old command retained, original Main order and exact real/resident B identity", order && identical);
    check("G software-only emulator has no Scale misses or unknown commands; session ends idle",
          early_emu.errors.empty() && baseline_emu.errors.empty() && !early_emu.scale_miss &&
          !baseline_emu.scale_miss && !early.session.failed && !early.session.live_scale_slots &&
          !early.session.outstanding_loops && !early.session.outstanding_readbacks &&
          !early.session.index_transaction_open && !early.report.fatal);
}

void test_stream_case(const paired_fixture & f, const potal_tile_plan & main, const paired_inputs & in,
                      const potal_paired_execution_plan & plan, const potal_paired_geometry * hw) {
    if (PD != 32) return;
    const auto golden = golden_of(f, main.tile_I * PD, in);
    serialized_result expected;
    expected.main_ok = expected.legacy_ok = golden.ok;
    expected.main_c = golden.main_c;
    expected.legacy = golden.legacy;
    expected.correction = golden.correction;
    expected.rows.resize(in.theta.size());
    expected.residual_c.resize(in.theta.size());
    for (const auto & row : golden.residual[0]) {
        expected.rows[0].push_back({row.lane, row.row});
        expected.residual_c[0].insert(expected.residual_c[0].end(), row.c.begin(), row.c.end());
    }
    h2_emulator old_emu;
    const auto old = run_paired(f, main, in, old_emu, true, 250, false, nullptr, 1, &plan,
                                false, false, nullptr, potal_paired_stream_mode::off, nullptr, hw);
    check("H SOFTWARE-ONLY old Ktile matches independent golden",
          golden.ok && old.status == potal_paired_status::ok &&
          compare_stripe(f, in, expected, old, std::min(main.tile_I * PD, f.M)).empty());
    std::vector<potal_paired_cmd_record> continuous_order;
    uint64_t continuous_waits = 0;
    for (auto mode : {potal_paired_stream_mode::bounded, potal_paired_stream_mode::continuous,
                      potal_paired_stream_mode::continuous_early}) {
        h2_emulator emu;
        const auto result = run_paired(f, main, in, emu, true, 250, false, nullptr, 1, &plan,
                                       false, false, nullptr, mode, nullptr, hw);
        const bool ok = result.status == potal_paired_status::ok;
        check("H SOFTWARE-ONLY stream executes unchanged manifest", ok);
        if (!ok) { print_report("H stream", result); continue; }
#if defined(POTAL_TEST_CLOCK_OBSERVER)
        check("H actual issuer clock calls absent between stream BEGIN and END", result.hot_clock_reads == 0);
#endif
        check("H exact raw Main/Residual INT32, host radix INT64 and final F32",
              golden.ok && compare_stripe(f, in, expected, result, std::min(main.tile_I * PD, f.M)).empty() &&
              result.main_c == old.main_c && result.residual_c == old.residual_c &&
              emu.errors.empty() && !emu.scale_miss);
        const auto & tile = result.report.tile;
        const auto & counts = result.report.counts;
        check("H capacity handling never changes Main I/J/K or planned windows",
              tile.tile_I == plan.tile.tile_I && tile.tile_J == plan.tile.tile_J &&
              tile.tile_K == plan.tile.tile_K && tile.k_panels == plan.tile.k_panels &&
              tile.residual_windows == plan.tile.residual_windows && tile.j_groups == plan.tile.j_groups &&
              counts.main_loop_ws == old.report.counts.main_loop_ws &&
              counts.residual_loop_ws == old.report.counts.residual_loop_ws);
        bool descriptors = true, config_outside = true;
        size_t loops = 0;
        bool hot = false;
        for (size_t i = 0; i < emu.records.size(); ++i) {
            const auto & record = emu.records[i];
            if (record.funct == POTAL_FUNCT_CONFIG && hot)
                config_outside = config_outside && (record.rs1 & 3) == 2 &&
                                 i + 1 < emu.records.size() && emu.records[i + 1].funct == POTAL_FUNCT_MVOUT;
            if (record.funct != POTAL_FUNCT_LOOP_WS) continue;
            hot = true;
            ++loops;
            if (i < 5) { descriptors = false; continue; }
            for (size_t word = 0; word < 5; ++word) {
                const auto & config = emu.records[i - 5 + word];
                descriptors = descriptors && config.funct == POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS + word &&
                              config.xd == 0;
            }
        }
        check("H hot LOOP_WS has exactly five descriptor words and no ordinary CONFIG",
              descriptors && config_outside && loops == counts.main_loop_ws + counts.residual_loop_ws);
        // Compare arithmetic, Scale and Index words, not relocated A staging
        // pointers. B-only adds no compute, metadata or retirement operation.
        std::vector<potal_paired_cmd_record> order;
        for (const auto & record : emu.records)
            if (record.funct == POTAL_FUNCT_LOOP_WS || record.funct == POTAL_FUNCT_SCALE_WRITE ||
                record.funct == POTAL_FUNCT_INDEX_CONFIG || record.funct == POTAL_FUNCT_INDEX_WRITE ||
                record.funct == POTAL_FUNCT_SCALE_RELEASE ||
                (record.funct == POTAL_FUNCT_STREAM && (record.rs1 & 15) == 5))
                order.push_back(record);
        if (mode == potal_paired_stream_mode::continuous) {
            continuous_order = order;
            continuous_waits = emu.stream_waits;
        }
        if (mode == potal_paired_stream_mode::continuous_early) {
            bool unchanged_order = order.size() == continuous_order.size();
            for (size_t i = 0; unchanged_order && i < order.size(); ++i)
                unchanged_order = order[i].funct == continuous_order[i].funct &&
                                  order[i].xd == continuous_order[i].xd &&
                                  order[i].rs1 == continuous_order[i].rs1 &&
                                  order[i].rs2 == continuous_order[i].rs2;
            bool placement = true, identity = true, checked_begin = false;
            size_t main_seen = 0, producers = 0;
            for (size_t i = 0; i < emu.records.size(); ++i) {
                const auto & record = emu.records[i];
                if (record.funct == POTAL_FUNCT_STREAM && record.rs1 == 6)
                    checked_begin = record.xd == 1 && record.rs2 == 0;
                if (record.funct == POTAL_FUNCT_LOOP_WS && !(record.rs1 >> 62 & 1)) ++main_seen;
                if (record.funct != POTAL_FUNCT_LDB_ONLY) continue;
                ++producers;
                // Only five B staging words may intervene: no forced wait,
                // global drain, Scale/Index programming or future compute.
                if (i < 6 || !main_seen || main_seen >= plan.residencies.size()) {
                    placement = identity = false;
                    continue;
                }
                const auto & current = plan.residencies[main_seen - 1];
                const auto & next = plan.residencies[main_seen];
                const auto & panel = tile.panels[next.panel];
                placement = placement && emu.records[i - 6].funct == POTAL_FUNCT_LOOP_WS &&
                            !(emu.records[i - 6].rs1 >> 62 & 1) &&
                            current.j_group == next.j_group && next.panel == current.panel + 1 &&
                            next.pair != current.pair;
                for (size_t word = 0; word < 5; ++word)
                    placement = placement &&
                        emu.records[i - 5 + word].funct == POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS + word &&
                        emu.records[i - 5 + word].xd == 0;
                const size_t pad_j = next.j_tiles * PD - next.cols;
                const auto source = reinterpret_cast<uintptr_t>(f.weight.b +
                    panel.k_begin * f.weight.ldb + next.j_group * tile.tile_J * PD);
                identity = identity && record.xd == 1 && record.rs1 == (UINT64_C(0x10) | next.pair) &&
                    record.rs2 == 0 &&
                    emu.records[i - 5].rs1 == uint64_t(pad_j) << 16 &&
                    emu.records[i - 5].rs2 ==
                        ((uint64_t(panel.k_tiles) << 32) | (uint64_t(next.j_tiles) << 16) | 1) &&
                    emu.records[i - 4].rs1 == 0 && emu.records[i - 4].rs2 == source &&
                    emu.records[i - 3].rs1 == 0 && emu.records[i - 3].rs2 == 0 &&
                    emu.records[i - 2].rs1 == 0 && emu.records[i - 2].rs2 == f.weight.ldb &&
                    emu.records[i - 1].rs1 == 0 && emu.records[i - 1].rs2 == 0;
            }
            check("H Early command_declared B-only immediately follows adjacent same-J Main before Residual",
                  placement && identity && checked_begin &&
                  producers == emu.ldb_only_loads &&
                  (tile.k_panels == 1 ? producers == 0 : producers >= tile.j_groups) &&
                  (counts.stream_capacity_boundaries != 0 ||
                   producers == (tile.k_panels > 1 ? tile.j_groups : 0)));
            check("H Early logical B loaded exactly once without fake compute or metadata counts",
                  unchanged_order && emu.b_loads + emu.ldb_only_loads == counts.main_loop_ws &&
                  emu.resident_main_loops == emu.ldb_only_loads && !emu.raw_b_mvins &&
                  emu.main_loops == counts.main_loop_ws && emu.residual_loops == counts.residual_loop_ws &&
                  counts.prefetch_loads == emu.ldb_only_loads &&
                  counts.prefetch_ldb_only == emu.ldb_only_loads && counts.prefetch_mvins == 0 &&
                  counts.prefetch_bytes_declared == emu.ldb_only_bytes &&
                  emu.stream_waits == continuous_waits + emu.ldb_only_loads &&
                  count_funct(emu, POTAL_FUNCT_COUNTER) == 3);
        } else {
            check("H legacy streaming emits no B-only producer or resident Main",
                  !emu.ldb_only_loads && !emu.resident_main_loops &&
                  emu.b_loads == counts.main_loop_ws && count_funct(emu, POTAL_FUNCT_COUNTER) == 2);
        }
        check("H bounded retains global RELEASE; continuous uses only scoped release",
              (count_funct(emu, POTAL_FUNCT_SCALE_RELEASE) > 0) ==
                  (mode == potal_paired_stream_mode::bounded));
        size_t window_boundaries = 0;
        size_t max_pitch = 0, packed_columns = 0, logical_columns = 0, load_columns = 0;
        for (const auto & panel : tile.panels) {
            if (!panel.windows.empty()) window_boundaries += panel.windows.size() - 1;
            for (const auto & window : panel.windows) {
                max_pitch = std::max(max_pitch, align_up(window.k_tiles * PD, POTAL_HW_MAX_BYTES));
                logical_columns += window.k_tiles * PD;
                load_columns += window.active_rows;
            }
        }
        for (const auto & fragment : tile.fragments) packed_columns += fragment.count;
        const size_t expected_fences = mode != potal_paired_stream_mode::bounded ? 2 * tile.j_groups :
            2 * tile.j_groups + ((tile.k_panels - 1) / 2 + window_boundaries) * tile.j_groups +
            counts.stream_capacity_boundaries;
        check("H only exact declared epoch/bounded/window Fence boundaries",
              counts.stream_batches > 0 && counts.fences == emu.fences && counts.fences == expected_fences);
        check("H logical staging/padding/valid/load bytes independent of DMA timing",
              counts.residual_staging_bytes == tile.residual_lane_rows * max_pitch * tile.residual_windows &&
              counts.residual_pitch_padding_bytes ==
                  tile.residual_lane_rows * (max_pitch * tile.residual_windows - logical_columns) &&
              counts.residual_valid_bytes == tile.residual_lane_rows * packed_columns &&
              counts.residual_load_bytes_declared == tile.residual_lane_rows * load_columns * tile.j_groups);
        check("H real LOAD3 arithmetic and local release/wait observations match report",
              emu.load3_loops == counts.residual_loop_ws && emu.stream_waits == counts.stream_pair_waits &&
              emu.stream_releases == counts.stream_local_releases &&
              emu.stream_releases + emu.scale_releases == emu.scale_writes && !emu.stream_active);
        if (tile.k_panels > 2)
            check("H both physical pairs reused with completed incarnations",
                  emu.stream_pairs[0].incarnation > 1 && emu.stream_pairs[1].incarnation > 0 &&
                  emu.stream_pairs[0].submitted == emu.stream_pairs[0].completed &&
                  emu.stream_pairs[1].submitted == emu.stream_pairs[1].completed);
        if (tile.k_panels == 260 && f.N == 1) {
            std::vector<uint32_t> generations;
            for (const auto & record : emu.records)
                if (record.funct == POTAL_FUNCT_SCALE_WRITE) generations.push_back((record.rs1 >> 8) & 255);
            check("H more than 255 actual allocations wrap only after old generation release",
                  generations.size() == 260 && generations[0] == generations[255] &&
                  generations[0] != 0 && emu.stream_releases + emu.scale_releases == 260 &&
                  emu.stream_pairs[0].incarnation == 130 && emu.stream_pairs[1].incarnation == 130);
        }
        check("H stream closes idle with no outstanding work or live Scale entries",
              !result.report.fatal && !result.session.failed && !result.session.live_scale_slots &&
              !result.session.outstanding_loops && !result.session.outstanding_readbacks &&
              !result.session.index_transaction_open);
    }
}

void test_stream_generation_wrap() {
    if (PD != 32) return;
    // 260 separate allocations, not merely starting the counter near 255.
    paired_fixture f(1, 1, 260 * PD, 193, [](size_t, size_t) { return int16_t(0); });
    potal_tile_plan main;
    paired_inputs in;
    const bool ok = f.ok && potal_frozen_plan(f.M, f.N, f.K, 1, 1, 1, PPB, &main) == POTAL_OK &&
                    make_inputs(f, PD, {}, in);
    check("H wrap fixture has 260 allocations and repeated pair reuse", ok);
    if (!ok) return;
    h2_emulator emu;
    const auto old = run_paired(f, main, in, emu, true, 0, false, nullptr, 1);
    check("H wrap manifest retains 260 one-tile panels", old.status == potal_paired_status::ok &&
          old.report.tile.k_panels == 260 && old.report.tile.tile_K == 1);
    if (old.status == potal_paired_status::ok) test_stream_case(f, main, in, old.report.plan);
#if defined(POTAL_TEST_CLOCK_OBSERVER)
    if (old.status == potal_paired_status::ok) {
        const char * previous = std::getenv("POTAL_STREAM_ADMISSION_PROFILE");
        const bool had_previous = previous != nullptr;
        const std::string saved = previous ? previous : "";
        setenv("POTAL_STREAM_ADMISSION_PROFILE", "1", 1);
        h2_emulator observed;
        const auto profiled = run_paired(f, main, in, observed, true, 0, false, nullptr, 1,
            &old.report.plan, false, false, nullptr, potal_paired_stream_mode::continuous_early);
        if (had_previous) setenv("POTAL_STREAM_ADMISSION_PROFILE", saved.c_str(), 1);
        else unsetenv("POTAL_STREAM_ADMISSION_PROFILE");
        check("H explicit CPU API diagnostic preserves software-only numerical results",
              profiled.status == potal_paired_status::ok && profiled.main_c == old.main_c &&
              profiled.residual_c == old.residual_c && observed.errors.empty());
        check("H explicit CPU API diagnostic reads clocks only around B-only calls",
              profiled.report.counts.prefetch_loads > 0 &&
              profiled.hot_clock_reads == 2 * profiled.report.counts.prefetch_loads);
    }
#endif
}

void test_stream_request_refusals() {
    check("H Early appends stream value three without changing legacy values",
          static_cast<unsigned>(potal_paired_stream_mode::off) == 0 &&
          static_cast<unsigned>(potal_paired_stream_mode::bounded) == 1 &&
          static_cast<unsigned>(potal_paired_stream_mode::continuous) == 2 &&
          static_cast<unsigned>(potal_paired_stream_mode::continuous_early) == 3);
    check("H stream request defaults off without callback",
          potal_paired_request{}.stream == potal_paired_stream_mode::off &&
          potal_paired_request{}.stream_counter == nullptr);
    paired_fixture f(1, 32, 64, 197, [](size_t, size_t) { return int16_t(0); });
    potal_tile_plan main;
    paired_inputs in;
    const bool ok = f.ok && main_select_ok(f, main) && make_inputs(f, main.tile_I * PD, {}, in);
    check("H refusal fixture", ok);
    if (!ok) return;
    if (PD == 32) {
        h2_emulator unavailable;
        unavailable.stream_capability = 0;
        const auto r = run_paired(f, main, in, unavailable, true, 0, false, nullptr, 1,
                                  nullptr, false, false, nullptr, potal_paired_stream_mode::continuous);
        check("H unsupported capability refuses before stream opcode or compute without fallback",
              r.status == potal_paired_status::execution_failed && r.report.hw_status == POTAL_ERR_UNSUPPORTED &&
              !r.report.fatal && count_funct(unavailable, POTAL_FUNCT_COUNTER) == 2 &&
              !count_funct(unavailable, POTAL_FUNCT_STREAM) && !unavailable.main_loops &&
              !unavailable.residual_loops && r.main_c.empty() && r.residual_c.empty());
        for (uint64_t capability : {UINT64_C(0), POTAL_STREAM_CAPABILITY,
                                    POTAL_STREAM_EARLY_CAPABILITY ^ UINT64_C(1)}) {
            h2_emulator early;
            early.stream_early_capability = capability;
            const auto refused = run_paired(f, main, in, early, true, 0, false, nullptr, 1,
                nullptr, false, false, nullptr, potal_paired_stream_mode::continuous_early);
            check("H Early requires exact capability before stream CONFIG or compute without fallback",
                  refused.status == potal_paired_status::execution_failed &&
                  refused.report.hw_status == POTAL_ERR_UNSUPPORTED && !refused.report.fatal &&
                  count_funct(early, POTAL_FUNCT_COUNTER) == 3 &&
                  !count_funct(early, POTAL_FUNCT_STREAM) && !count_funct(early, POTAL_FUNCT_CONFIG) &&
                  !early.main_loops && !early.residual_loops && !early.ldb_only_loads &&
                  refused.main_c.empty() && refused.residual_c.empty());
        }
    }
    for (auto requested : {potal_paired_stream_mode::continuous, potal_paired_stream_mode::continuous_early})
    for (unsigned kind = 0; kind < 5; ++kind) {
        h2_emulator emu;
        const auto mode = kind == 4 ? static_cast<potal_paired_stream_mode>(255) :
                                     requested;
        const auto r = run_paired(f, main, in, emu, true, 0, kind == 1, nullptr, kind == 0 ? 0 : 1,
                                  nullptr, kind == 2, kind == 3, nullptr, mode);
        check("H malformed stream combination refuses before any compute without fallback",
              r.status == potal_paired_status::invalid_argument && !r.report.issued && !r.report.fatal &&
              emu.records.empty() && r.main_c.empty() && r.residual_c.empty());
    }
}

void test_stream_capacity() {
    if (PD != 32) return;
    {
        paired_fixture f(1, 1, 2048, 227, [](size_t, size_t) { return int16_t(0); });
        potal_tile_plan main;
        paired_inputs in;
        const entries_fn entries = [](size_t, size_t) {
            std::vector<res_entry> out;
            for (size_t block = 0; block < 64; block += 2) out.push_back({0, block * 32 + 3, 7});
            return out;
        };
        auto hw = potal_paired_hw_geometry();
        hw.scale_entries = 60; // One 48-slot panel fits, two do not.
        // K32 leaves room for Residual A beside Main A in the same 64-tile bank.
        const bool ok = f.ok && potal_frozen_plan(f.M, f.N, f.K, 1, 1, 32, PPB, &main) == POTAL_OK &&
                        make_inputs(f, PD, entries, in);
        check("H bounded live-set splitting fixture", ok);
        if (ok) {
            h2_emulator baseline;
            const auto old = run_paired(f, main, in, baseline, true, 0, false, nullptr, 1, nullptr,
                false, false, nullptr, potal_paired_stream_mode::off, nullptr, &hw);
            check("H capacity split retains two K32 panels", old.status == potal_paired_status::ok &&
                  old.report.tile.tile_K == 32 && old.report.tile.k_panels == 2);
            if (old.status == potal_paired_status::ok) {
                test_stream_case(f, main, in, old.report.plan, &hw);
                for (auto mode : {potal_paired_stream_mode::bounded, potal_paired_stream_mode::continuous,
                                  potal_paired_stream_mode::continuous_early}) {
                    h2_emulator emu;
                    const auto split = run_paired(f, main, in, emu, true, 0, false, nullptr, 1, &old.report.plan,
                        false, false, nullptr, mode, nullptr, &hw);
                    check("H safe metadata boundary instead of hidden retiling",
                          split.status == potal_paired_status::ok && split.report.counts.stream_capacity_boundaries == 1 &&
                          split.main_c == old.main_c && split.residual_c == old.residual_c);
                }
            }
        }
    }
    paired_fixture f(1, 1, 512, 223, [](size_t, size_t) { return int16_t(0); });
    potal_tile_plan main;
    paired_inputs in;
    const entries_fn entries = [](size_t, size_t) {
        std::vector<res_entry> out;
        for (size_t block = 0; block < 16; block += 2) out.push_back({0, block * 32 + 3, 7});
        return out;
    };
    const bool ok = f.ok && potal_frozen_plan(f.M, f.N, f.K, 1, 1, 16, PPB, &main) == POTAL_OK &&
                    make_inputs(f, PD, entries, in);
    check("H capacity fixture has discontinuous nonaliasing windows", ok);
    if (!ok) return;
    auto hw = potal_paired_hw_geometry();
    hw.scale_entries = 20; // Main 16 + one Residual 2 fits; Main + all four windows 24 does not.
    h2_emulator old_emu;
    const auto old = run_paired(f, main, in, old_emu, true, 0, false, nullptr, 1, nullptr,
                                false, false, nullptr, potal_paired_stream_mode::off, nullptr, &hw);
    check("H old schedule fits twenty slots without retiling",
          old.status == potal_paired_status::ok && old.report.tile.tile_K == 16 &&
          old.report.tile.k_panels == 1 && old.report.tile.residual_windows == 4);
    if (old.status != potal_paired_status::ok) return;
    for (auto mode : {potal_paired_stream_mode::bounded, potal_paired_stream_mode::continuous,
                      potal_paired_stream_mode::continuous_early}) {
        h2_emulator emu;
        const auto result = run_paired(f, main, in, emu, true, 0, false, nullptr, 1, &old.report.plan,
                                       false, false, nullptr, mode, nullptr, &hw);
        check("H oversized panel live set refuses before issue, never silently retiles",
              result.status == potal_paired_status::scale_window_capacity && !result.report.issued &&
              !result.report.fatal && result.report.tile.tile_K == 16 && emu.records.empty());
    }
}

void test_stream_emulator_lifetimes() {
    if (PD != 32) return;
    h2_emulator e;
    check("H emulator continuous begin", e.command({POTAL_FUNCT_STREAM, 1, 2, 0}) == POTAL_OK);
    int8_t sources[2][4] = {{1, 2, 3, 4}, {-1, -2, -3, -4}};
    for (uint32_t pair = 0; pair < 2; ++pair) {
        e.stream_pairs[pair].scheduled = true;
        e.stream_pairs[pair].submitted = 1;
        e.borrow(pair, reinterpret_cast<uintptr_t>(sources[pair]), 1, 4, 4);
        e.scales[pair][0] = {true, pair + 1, 0};
        e.scale_pairs[pair] = 1u << pair;
    }
    check("H scheduled B is not an observed completion",
          !e.stream_pairs[0].dma_done && !e.stream_pairs[1].dma_done);
    check("H pair-local wait leaves opposite pair pending",
          e.command({POTAL_FUNCT_STREAM, 1, 4, 0}) == POTAL_OK &&
          e.stream_pairs[0].dma_done && e.stream_pairs[0].completed == 1 &&
          !e.stream_pairs[1].dma_done && e.stream_pairs[1].completed == 0 && e.fences == 0);
    check("H local generation release preserves other pair allocation",
          e.command({POTAL_FUNCT_STREAM, 1, 5, UINT64_C(1) << 8}) == POTAL_OK &&
          !e.scales[0][0].valid && e.scales[1][0].valid && !e.scale_pairs[0] && e.scale_pairs[1] == 2);
    check("H global Fence observes all scheduled work",
          e.command({POTAL_FUNCT_FENCE, 1, 0, 0}) == POTAL_OK &&
          e.stream_pairs[1].dma_done && e.stream_pairs[1].completed == 1);
    check("H remaining generation released and end closes stream",
          e.command({POTAL_FUNCT_STREAM, 1, 5, (UINT64_C(2) << 8) | 1}) == POTAL_OK &&
          e.command({POTAL_FUNCT_STREAM, 1, 3, 0}) == POTAL_OK && !e.stream_active && e.errors.empty());
    h2_emulator changed;
    check("H source immutability setup", changed.command({POTAL_FUNCT_STREAM, 1, 2, 0}) == POTAL_OK);
    changed.stream_pairs[0].scheduled = true;
    changed.stream_pairs[0].submitted = 1;
    changed.borrow(0, reinterpret_cast<uintptr_t>(sources[0]), 1, 4, 4);
    (void) changed.command({POTAL_FUNCT_STREAM, 1, 4, 0});
    sources[0][0] = 99; // Successful local completion does not end borrowed-source lifetime.
    (void) changed.command({POTAL_FUNCT_STREAM, 1, 3, 0});
    check("H emulator detects mutation even after local wait before stream end", !changed.errors.empty());
}

struct stream_failure_sink {
    h2_emulator emulator;
    size_t position = 0, seen = 0, failed_at = SIZE_MAX;
    bool lose_response = false, armed = false;

    static potal_status counter(void * ctx, const potal_paired_cmd_record * command, uint64_t * value) {
        return h2_emulator::on_counter(&static_cast<stream_failure_sink *>(ctx)->emulator, command, value);
    }
    static potal_status command(void * ctx, const potal_paired_cmd_record * command) {
        auto & self = *static_cast<stream_failure_sink *>(ctx);
        if (command->funct == POTAL_FUNCT_LOOP_WS_CONFIG_BOUNDS && !self.seen) self.armed = true;
        if (self.armed && ++self.seen == self.position) {
            self.armed = false;
            self.failed_at = self.emulator.records.size();
            if (self.lose_response) {
                // Execute the command, then lose its response: cannot infer
                // safety from a sink error or synchronous test arithmetic.
                (void) self.emulator.command(*command);
                return POTAL_ERR_NO_HARDWARE;
            }
            self.emulator.records.push_back(*command);
            return POTAL_HW_INVALID_ADDRESS;
        }
        return self.emulator.command(*command);
    }
};

void test_stream_execution_failures() {
    if (PD != 32) return;
    for (size_t position = 1; position <= 6; ++position) {
        // Quarantine intentionally retains issuer and DMA buffers. Isolate
        // every fault, including a lost accepted LOOP_WS response.
        std::fflush(nullptr);
        const pid_t child = fork();
        if (child == 0) {
            const int before = g_failures;
            paired_fixture f(1, 32, 160, 211, [](size_t, size_t) { return int16_t(0); });
            potal_tile_plan main;
            paired_inputs in;
            const bool ok = f.ok && potal_frozen_plan(f.M, f.N, f.K, 1, 1, 2, PPB, &main) == POTAL_OK &&
                            make_inputs(f, PD, {}, in);
            check("H fault fixture", ok);
            if (ok) {
                stream_failure_sink failure;
                failure.position = position;
                failure.lose_response = position == 6;
                const potal_paired_sink sink{stream_failure_sink::command, &failure};
                const auto r = run_paired(f, main, in, failure.emulator, true, 0, false, &sink, 1,
                    nullptr, false, false, nullptr, potal_paired_stream_mode::continuous,
                    stream_failure_sink::counter);
                check("H partial descriptor or response loss fails without fallback",
                      failure.failed_at != SIZE_MAX && r.status == potal_paired_status::execution_failed &&
                      r.report.hw_status == (position == 6 ? POTAL_ERR_NO_HARDWARE : POTAL_HW_INVALID_ADDRESS) &&
                      r.main_c.empty() && r.residual_c.empty() &&
                      failure.emulator.main_loops == (position == 6 ? 1u : 0u) &&
                      !failure.emulator.residual_loops && failure.emulator.errors.empty());
                check("H unknown stream effects quarantine with no later commands",
                      r.report.fatal && !r.report.cleaned && r.session.failed &&
                      failure.failed_at + 1 == failure.emulator.records.size());
                potal_paired_session * other = nullptr;
                check("H quarantined stream retains exclusive owner",
                      potal_paired_session_open(&sink, &other) == POTAL_ERR_BUSY);
            }
            std::fflush(nullptr);
            _exit(g_failures == before ? 0 : 1);
        }
        check("H fault process created", child >= 0);
        if (child >= 0) {
            int status = 0;
            pid_t result;
            do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
            check("H partial issue and lost response ownership isolated",
                  result == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
        }
    }
}

void test_early_emulator() {
    if (PD != 32) return;
    for (uint32_t mode : {0u, 1u, 2u, 6u})
    for (uint32_t pair : {0u, 1u}) {
        h2_emulator e;
        if (mode) {
            check("H SOFTWARE-ONLY emulator accepts legacy or Early stream begin",
                  e.command({POTAL_FUNCT_STREAM, 1, mode, 0}) == POTAL_OK);
            e.stream_pairs[1 - pair].scheduled = true;
            e.stream_pairs[1 - pair].submitted = 1;
        }
        const uint32_t K = pair ? 1 : 2, J = 2, pad_j = 5;
        const size_t stride = 3 * PD;
        std::unique_ptr<int8_t, dma_deleter> source(static_cast<int8_t *>(dma_alloc(2 * PD * stride)));
        check("G emulator source allocation", source != nullptr);
        if (!source) return;
        for (size_t i = 0; i < 2 * PD * stride; ++i) source.get()[i] = int8_t(int(i % 251) - 125);
        std::fill(e.sp.begin(), e.sp.end(), int8_t(91));
        std::fill(e.acc.begin(), e.acc.end(), -1234567);
        e.scales[3][2] = {true, 9, 17};
        e.tables[0].configured = true;
        e.tables[0].b_base = 123;
        e.tables[0].entry = {2, 7};
        e.tables[0].valid = {1, 1};
        e.selected_table = 1;
        e.ld_config[1] = (uint64_t(PD) << 16) | UINT64_C(0x109);
        e.ld_stride[1] = stride;
        const auto old_sp = e.sp;
        const auto old_acc = e.acc;
        const potal_paired_cmd_record words[] = {
            {9, 0, uint64_t(pad_j) << 16, (uint64_t(K) << 32) | (uint64_t(J) << 16) | 1},
            {10, 0, 0, reinterpret_cast<uintptr_t>(source.get())},
            {11, 0, 0, 0}, {12, 0, 0, stride}, {13, 0, 0, 0},
            {29, 1, UINT64_C(0x10) | pair, 0},
        };
        bool accepted = true;
        for (const auto & word : words) accepted = accepted && e.command(word) == POTAL_OK;
        if (mode == 1 || mode == 2) {
            check("H legacy stream modes reject B-only without data or compute effects",
                  !accepted && !e.errors.empty() && e.sp == old_sp && e.acc == old_acc &&
                  !e.ldb_only_loads && !e.main_loops && !e.residual_loops &&
                  !e.stream_pairs[pair].producer && !e.stream_pairs[pair].submitted);
            continue;
        }
        const size_t first = (2 * pair + 2) * POTAL_HW_BANK_ROWS - K * J * PD;
        bool bytes = accepted;
        for (size_t row = 0; row < h2_emulator::sp_rows; ++row) for (size_t col = 0; col < PD; ++col) {
            int8_t want = old_sp[row * PD + col];
            const bool loaded = row >= first && row < first + K * J * PD;
            if (loaded) {
                const size_t tile = (row - first) / PD;
                const size_t k = tile / J * PD + (row - first) % PD;
                const size_t j = tile % J * PD + col;
                want = j < J * PD - pad_j ? source.get()[k * stride + j] : 0;
            }
            bytes = bytes && e.sp[row * PD + col] == want &&
                    e.sp_written[row] == (loaded ? UINT64_C(0xffffffff) : 0);
        }
        bool metadata = e.tables[0].configured && e.tables[0].b_base == 123 &&
                        e.tables[0].entry == std::vector<uint32_t>({2, 7}) &&
                        e.tables[0].valid == std::vector<uint8_t>({1, 1}) &&
                        !e.tables[1].configured && e.selected_table == 1;
        for (size_t slot = 0; slot < POTAL_SCALE_ENTRIES; ++slot) for (size_t col = 0; col < PD; ++col) {
            const auto & scale = e.scales[slot][col];
            metadata = metadata && scale.valid == (slot == 3 && col == 2);
            if (scale.valid) metadata = metadata && scale.generation == 9 && scale.carrier == 17;
        }
        check("G emulator both pairs and K/J tails copy only independently addressed B", bytes);
        check("G load-only has no A/ACC/Execute/Scale/Index effects",
              e.acc == old_acc && metadata && !e.main_loops && !e.residual_loops &&
              !e.index_configs && !e.index_writes && !e.scale_writes && !e.scale_releases &&
              !e.mvouts && !e.raw_b_mvins && !e.b_loads && e.ldb_only_loads == 1 &&
              e.ld_config[0] == 0 && e.ld_stride[0] == 0 && e.st_stride == 0 && e.errors.empty());
        if (mode == 6) {
            const auto & producer = e.stream_pairs[pair];
            check("H Early admission owns B but creates zero logical compute readers",
                  producer.producer && !producer.scheduled && !producer.dma_done &&
                  producer.submitted == 0 && producer.completed == 0 &&
                  producer.borrowed.size() == 1 && e.stream_waits == 0 && e.fences == 0);
            {
                h2_emulator pending = e;
                pending.scales[3][2].valid = false;
                check("H producer-only pending epoch prevents emulator stream end with zero pending readers",
                      pending.command({POTAL_FUNCT_STREAM, 1, 4 | (uint64_t(1 - pair) << 4), 0}) == POTAL_OK &&
                      pending.command({POTAL_FUNCT_STREAM, 1, 3, 0}) == POTAL_HW_BAD_COMMAND &&
                      pending.stream_active && pending.stream_pairs[pair].submitted == 0 &&
                      !pending.stream_pairs[pair].dma_done);
                check("H global drain reconciles accepted producer without inventing compute completion",
                      pending.command({POTAL_FUNCT_FENCE, 1, 0, 0}) == POTAL_OK &&
                      pending.stream_pairs[pair].dma_done &&
                      pending.stream_pairs[pair].submitted == 0 && pending.stream_pairs[pair].completed == 0 &&
                      !pending.main_loops && !pending.residual_loops && pending.fences == 1 &&
                      pending.command({POTAL_FUNCT_STREAM, 1, 3, 0}) == POTAL_OK && pending.errors.empty());
            }
            check("H pair-local producer wait completes B without completing opposite Main",
                  e.command({POTAL_FUNCT_STREAM, 1, 4 | (uint64_t(pair) << 4), 0}) == POTAL_OK &&
                  producer.dma_done && producer.submitted == 0 && producer.completed == 0 &&
                  producer.borrowed.empty() && !e.stream_pairs[1 - pair].dma_done &&
                  e.stream_pairs[1 - pair].completed == 0 && !e.main_loops && !e.residual_loops &&
                  e.stream_waits == 1 && e.fences == 0);
            e.scales[3][2].valid = false; // Remove this test's sentinel, not a command-owned allocation.
            check("H stream end clears Early permission after all producers and readers complete",
                  e.command({POTAL_FUNCT_STREAM, 1, 4 | (uint64_t(1 - pair) << 4), 0}) == POTAL_OK &&
                  e.command({POTAL_FUNCT_STREAM, 1, 3, 0}) == POTAL_OK &&
                  !e.stream_active && !e.stream_early && !e.stream_continuous && e.errors.empty());
        }
    }
}

struct early_failure_sink {
    h2_emulator emulator;
    size_t position = 0, seen = 0, failed_at = 0;
    bool armed = false;

    static potal_status counter(void *ctx, const potal_paired_cmd_record *record, uint64_t *value) {
        return h2_emulator::on_counter(&static_cast<early_failure_sink *>(ctx)->emulator, record, value);
    }
    static potal_status command(void *ctx, const potal_paired_cmd_record *record) {
        auto & s = *static_cast<early_failure_sink *>(ctx);
        if (s.armed && ++s.seen == s.position) {
            s.armed = false;
            s.failed_at = s.emulator.records.size();
            s.emulator.records.push_back(*record);
            return POTAL_HW_INVALID_ADDRESS;
        }
        const auto status = s.emulator.command(*record);
        if (record->funct == POTAL_FUNCT_LOOP_WS && !(record->rs1 >> 62 & 1)) s.armed = true;
        return status;
    }
};

void early_execution_failure(size_t position) {
    const int before = g_failures;
    paired_fixture f(65, 97, 2080, 129, [](size_t, size_t n) { return int16_t(n % 3); });
    potal_tile_plan main;
    paired_inputs in;
    const bool fixture = f.ok && potal_frozen_plan(f.M, f.N, f.K, 2, 2, 4, PPB, &main) == POTAL_OK &&
                         make_inputs(f, main.tile_I * PD, within_k(dense_entries(11, 97, 120), f.K), in);
    check("G execution failure fixture", fixture);
    if (!fixture) return;
    early_failure_sink failure;
    failure.position = position;
    const potal_paired_sink sink = {early_failure_sink::command, &failure};
    const run_result r = run_paired(f, main, in, failure.emulator, true, 250, false, &sink, 1,
                                   nullptr, true, true, early_failure_sink::counter);
    const auto & records = failure.emulator.records;
    check("G answered early failure stops executor before future compute without fallback",
          r.status == potal_paired_status::execution_failed && r.report.hw_status == POTAL_HW_INVALID_ADDRESS &&
          failure.emulator.main_loops == 1 && failure.emulator.residual_loops == 0 &&
          r.main_c.empty() && r.residual_c.empty());
    check("G reports descriptor admissions separately from zero raw MVIN attempts",
          r.report.counts.prefetch_mvins == 0 && r.report.counts.prefetch_ldb_only == (position >= 6 ? 1u : 0u));
    if (position < 7) {
        check("G partial staging/run gets exactly one cleanup Fence then actual live Releases",
              r.report.cleaned && !r.report.fatal && r.session.failed &&
              !r.session.outstanding_loops && !r.session.live_scale_slots &&
              failure.failed_at + 1 < records.size() &&
              records[failure.failed_at + 1].funct == POTAL_FUNCT_FENCE &&
              std::count_if(records.begin() + failure.failed_at + 1, records.end(),
                  [](const potal_paired_cmd_record & record) { return record.funct == POTAL_FUNCT_FENCE; }) == 1);
    } else {
        check("G failed extra Fence quarantines with no cleanup or retried Fence",
              r.report.fatal && !r.report.cleaned && r.session.failed &&
              failure.failed_at + 1 == records.size() && records.back().funct == POTAL_FUNCT_FENCE);
        potal_paired_session *other = nullptr;
        check("G failed extra Fence retains global owner", potal_paired_session_open(&sink, &other) == POTAL_ERR_BUSY);
        std::fflush(nullptr);
        _exit(g_failures == before ? 0 : 1); // Borrowed A/B and sink remain alive through process exit.
    }
}

void test_early_execution_failures() {
    if (PD != 32) return;
    for (size_t position = 1; position <= 6; ++position) early_execution_failure(position);
    // Unknown completion is terminal. Keep its immutable tensors and session
    // alive until process exit, never poison the suite's next issuer.
    std::fflush(nullptr);
    const int before = g_failures;
    const pid_t child = fork();
    if (child == 0) {
        early_execution_failure(7);
        std::fflush(nullptr);
        _exit(g_failures == before ? 0 : 1);
    }
    check("G extra Fence failure forked", child >= 0);
    if (child >= 0) {
        int status = 0;
        pid_t result;
        do { result = waitpid(child, &status, 0); } while (result < 0 && errno == EINTR);
        check("G extra Fence failure isolated", result == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
}

void test_early_saturation() {
    if (PD != 32) return;
    // Four shifted positive/negative DIM partials saturate repeatedly.
    // The last unshifted block reverses sign after a K-panel boundary.
    paired_fixture f(33, 35, 160, 137,
        [](size_t block, size_t) { return int16_t(block < 4 ? 16 : 0); },
        [](size_t k, size_t, std::mt19937 &) { return int8_t(k < 128 ? 127 : -1); },
        [](size_t row, size_t, std::mt19937 &) { return int8_t(row == 0 ? 127 : -128); });
    potal_tile_plan main;
    paired_inputs in;
    const entries_fn entries = [](size_t, size_t rows) {
        std::vector<res_entry> result;
        for (size_t k = 0; k < 160; ++k) {
            result.push_back({0, k, 127});
            if (rows > 1) result.push_back({rows - 1, k, -128});
        }
        return result;
    };
    const bool fixture = f.ok && potal_frozen_plan(f.M, f.N, f.K, 2, 2, 2, PPB, &main) == POTAL_OK &&
                         make_inputs(f, main.tile_I * PD, entries, in);
    check("G saturation/reversal fixture with I/J/K tails", fixture);
    if (!fixture) return;
    const auto golden = golden_of(f, main.tile_I * PD, in);
    check("G independent signed saturation reverses at final panel",
          golden.ok && golden.main_c[0] == INT32_MAX - 4064 &&
          golden.main_c[32 * f.N] == INT32_MIN + 4096 &&
          golden.residual[0].size() == 2 &&
          golden.residual[0][0].c[0] == INT32_MAX - 4064 &&
          golden.residual[0][1].c[0] == INT32_MIN + 4096);
    h2_emulator emulator;
    const auto baseline = run_paired(f, main, in, emulator);
    check("G saturation manifest keeps three K panels", baseline.status == potal_paired_status::ok &&
          baseline.report.tile.k_panels == 3 && baseline.report.tile.panels.back().k_tiles == 1);
    if (baseline.status == potal_paired_status::ok) {
        test_early_case(f, main, in, baseline.report.plan);
        test_stream_case(f, main, in, baseline.report.plan);
    }
}

void test_early_request_refusals() {
    paired_fixture f(1, 32, 64, 141, [](size_t, size_t) { return int16_t(0); });
    potal_tile_plan main;
    paired_inputs in;
    const bool fixture = f.ok && main_select_ok(f, main) && make_inputs(f, main.tile_I * PD, {}, in);
    check("G early request refusal fixture", fixture);
    if (!fixture) return;
    for (uint32_t kind = 0; kind < 3; ++kind) {
        h2_emulator emulator;
        emulator.ldb_only_capability = 0; // Ordinary old image, not a new instruction response.
        const auto result = run_paired(f, main, in, emulator, true, 0, false, nullptr,
                                       kind == 1 ? 0 : 1, nullptr, kind != 0, true);
        if (kind == 2 && PD == 32) {
            check("G executor old-image refusal before new op or compute",
                  result.status == potal_paired_status::execution_failed &&
                  result.report.hw_status == POTAL_ERR_UNSUPPORTED && !result.report.fatal &&
                  count_funct(emulator, POTAL_FUNCT_COUNTER) == 2 &&
                  count_funct(emulator, POTAL_FUNCT_LDB_ONLY) == 0 &&
                  count_funct(emulator, POTAL_FUNCT_LOOP_WS) == 0 &&
                  emulator.raw_b_mvins == 0);
        } else {
            check("G early requires DIM32, next-K and Ktile before commands",
                  result.status == potal_paired_status::invalid_argument &&
                  result.report.hw_status == POTAL_ERR_UNSUPPORTED && !result.report.issued &&
                  !result.report.fatal && emulator.records.empty());
        }
    }
}

void test_ktile_scale_reuse() {
    check("R Scale reuse defaults off", !potal_paired_request{}.ktile_scale_reuse);
    for (bool gapped : { false, true }) {
        // Partial compact fragments, signed multi-lane digits, J/I tails and a
        // final short Main panel. Constant carriers deliberately cannot prove aliasing.
        paired_fixture f(65, 97, gapped ? 256 : 2080, 129,
                         [](size_t, size_t n) { return (int16_t) (n % 3); });
        potal_tile_plan main;
        paired_inputs in;
        const entries_fn entries = [&](size_t, size_t rows) {
            std::vector<res_entry> out;
            for (size_t block = 0; block < f.K / 32; block += gapped ? 2 : 1) {
                for (size_t row = 0; row < rows; ++row) {
                    // The final one-row stripe has only one endpoint. Duplicate
                    // coordinates would invalidate the packet's digit_nnz.
                    if (row != 0 && row + 1 != rows) continue;
                    for (size_t k : { size_t(1), size_t(7), size_t(29) })
                        out.push_back({ row, block * 32 + k, row == 0 ? 513 : -65537 });
                }
            }
            return out;
        };
        const bool fixture = f.ok &&
                             (gapped ? potal_frozen_plan(f.M, f.N, f.K, 1, 1, 128 / PD, PPB, &main) == POTAL_OK :
                              PD == 32 ? potal_frozen_plan(f.M, f.N, f.K, 2, 2, 4, PPB, &main) == POTAL_OK :
                                         main_select_ok(f, main)) &&
                             make_inputs(f, main.tile_I * PD, entries, in);
        check("R fixture", fixture);
        if (!fixture) return;
        h2_emulator baseline_emu;
        const run_result baseline = run_paired(f, main, in, baseline_emu, true, 250);
        check("R default stream succeeds without reuse", baseline.status == potal_paired_status::ok &&
              baseline.report.counts.scale_reused_windows == 0 && baseline.report.counts.scale_reused_columns == 0);
        if (baseline.status != potal_paired_status::ok) return;
        const auto & plan = baseline.report.plan;
        // Compare every command field, relocating only DMA allocations owned by
        // each invocation. Staging's first residual address is its allocation base.
        const auto commands = [&](const h2_emulator & source, const run_result & result, bool omit_scale) {
            std::vector<potal_paired_cmd_record> records;
            uint64_t staging = UINT64_MAX;
            for (const auto & record : source.records)
                if (record.funct == POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB && record.rs2 == 0)
                    staging = std::min(staging, record.rs1);
            const uint64_t cm = reinterpret_cast<uintptr_t>(result.c_main.get());
            const uint64_t cr = reinterpret_cast<uintptr_t>(result.c_res.get());
            for (auto record : source.records) {
                if (omit_scale && (record.funct == POTAL_FUNCT_SCALE_WRITE ||
                                   record.funct == POTAL_FUNCT_SCALE_RELEASE || record.funct == POTAL_FUNCT_FENCE))
                    continue;
                if (record.funct == POTAL_FUNCT_LOOP_WS_CONFIG_ADDRS_AB && record.rs2 == 0)
                    record.rs1 -= staging;
                if (record.funct == POTAL_FUNCT_MVOUT) {
                    const uint64_t main_bytes = align_up(result.main_c.size() / f.N, PD) * align_up(f.N, PD) * sizeof(int32_t);
                    record.rs1 = record.rs1 >= cm && record.rs1 < cm + main_bytes ?
                                 record.rs1 - cm : (UINT64_C(1) << 63) | (record.rs1 - cr);
                }
                if (omit_scale && record.funct == POTAL_FUNCT_LOOP_WS && ((record.rs1 >> 62) & 1))
                    record.rs1 &= ~(((UINT64_C(1) << 24) - 1) << 21); // Only Scale base and generation may change.
                records.push_back(record);
            }
            return records;
        };
        const auto same_commands = [](const std::vector<potal_paired_cmd_record> & a,
                                      const std::vector<potal_paired_cmd_record> & b) {
            if (a.size() != b.size()) return false;
            for (size_t i = 0; i < a.size(); ++i)
                if (a[i].funct != b[i].funct || a[i].xd != b[i].xd ||
                    a[i].rs1 != b[i].rs1 || a[i].rs2 != b[i].rs2) return false;
            return true;
        };
        h2_emulator disabled_emu;
        const run_result disabled = run_paired(f, main, in, disabled_emu, true, 250, false, nullptr, 0, &plan);
        check("R default and explicit false retain every command field apart from DMA relocation",
              disabled.status == potal_paired_status::ok &&
              same_commands(commands(baseline_emu, baseline, false), commands(disabled_emu, disabled, false)));
        h2_emulator emu;
        run_result reused = run_paired(f, main, in, emu, true, 250, false, nullptr, 1, &plan);
        if (PD == 16) {
            check("R DIM16 explicit reuse refuses before commands", reused.status == potal_paired_status::invalid_argument &&
                  reused.report.hw_status == POTAL_ERR_UNSUPPORTED && !reused.report.issued &&
                  !reused.report.fatal && emu.records.empty() && reused.generation == 250);
            continue;
        }
        residual::decoded_packet decoded;
        const bool decoded_ok = residual::decode_packet(*in.packets[0], f.weight, decoded) == grmd::RmdStatus::success;
        const size_t rows = std::min(main.tile_I * PD, f.M);
        const golden_result golden = golden_of(f, main.tile_I * PD, in);
        std::vector<int32_t> golden_residual;
        for (const auto & row : golden.residual[0])
            golden_residual.insert(golden_residual.end(), row.c.begin(), row.c.end());
        grmd::BlockScaledInt64Correction correction;
        const bool numbers = decoded_ok && golden.ok && reused.status == potal_paired_status::ok &&
            baseline.main_c == reused.main_c && baseline.residual_c == reused.residual_c &&
            reused.main_c.size() == rows * f.N &&
            std::equal(reused.main_c.begin(), reused.main_c.end(), golden.main_c.begin()) &&
            reused.residual_c == golden_residual &&
            residual::reconstruct_rows(*in.packets[0], reused.rows, reused.residual_c.data(), f.N, f.N, correction) ==
                grmd::RmdStatus::success && correction.values == golden.correction[0];
        check("R default/opt raw Main, Residual and reconstructed INT64 equal independent golden", numbers);
        if (!numbers) continue;

        h2_emulator prefetched_emu;
        const run_result prefetched = run_paired(f, main, in, prefetched_emu, true, 250,
                                                 false, nullptr, 1, &plan, true);
        check("D2 managed next K completes", prefetched.status == potal_paired_status::ok);
        if (prefetched.status != potal_paired_status::ok) {
            std::fprintf(stderr, "D2 stage=%s detail=%s hw=%d\n", prefetched.report.stage,
                         prefetched.report.detail.c_str(), int(prefetched.report.hw_status));
        }
        check("D2 raw Main and Residual equal independently verified Ktile",
              prefetched.main_c == reused.main_c && prefetched.residual_c == reused.residual_c);
        check("D2 real MVIN count matches managed loads and suppressed Main B",
              prefetched.report.counts.prefetch_loads > 0 &&
              prefetched.report.counts.prefetch_loads == prefetched_emu.resident_main_loops &&
              prefetched.report.counts.prefetch_mvins == prefetched_emu.raw_b_mvins &&
              prefetched.report.counts.prefetch_bytes_declared == prefetched_emu.raw_b_bytes);
        check("D2 no extra Release drain or Scale work",
              prefetched.report.counts.scale_writes == reused.report.counts.scale_writes &&
              prefetched.report.counts.scale_releases == reused.report.counts.scale_releases &&
              prefetched.session.drains == reused.session.drains);
        test_early_case(f, main, in, plan);
        test_stream_case(f, main, in, plan);

#if !defined(__riscv)
        potal_paired_request unavailable;
        uint32_t unavailable_generation = 250;
        unavailable.rows = rows; unavailable.n = f.N; unavailable.k = f.K;
        unavailable.main_plan = &main; unavailable.a = f.a.get(); unavailable.lda = f.lda;
        unavailable.weight = &f.weight; unavailable.c_main = baseline.c_main.get();
        unavailable.ldc_main = align_up(f.N, PD); unavailable.generation = &unavailable_generation;
        unavailable.tlb_flush = false;
        unavailable.ktile_scale_reuse = true; unavailable.next_k_prefetch = true;
        potal_paired_report unavailable_report;
        const auto unavailable_status = potal_paired_run_session(unavailable, nullptr, unavailable_report);
        check("D2 no-sink admission returns nonfatal without issue",
              unavailable_status == potal_paired_status::execution_failed &&
              unavailable_report.hw_status == POTAL_ERR_NO_HARDWARE &&
              !unavailable_report.fatal && !unavailable_report.issued && unavailable_generation == 250);
        potal_paired_session * reopened = nullptr;
        check("D2 no-sink wrapper releases owner", potal_paired_session_open(nullptr, &reopened) == POTAL_OK);
        if (reopened) check("D2 no-sink reopened close", potal_paired_session_close(reopened) == POTAL_OK);
#endif

        h2_emulator failed_prefetch;
        failed_prefetch.fail_funct = POTAL_FUNCT_MVIN;
        failed_prefetch.fail_status = POTAL_ERR_NO_HARDWARE;
        const run_result rejected = run_paired(f, main, in, failed_prefetch, true, 250,
                                               false, nullptr, 1, &plan, true);
        check("D2 sink NO_HARDWARE after issue is cleaned, not treated as no effects",
              rejected.status == potal_paired_status::execution_failed &&
              rejected.report.hw_status == POTAL_ERR_NO_HARDWARE &&
              rejected.report.cleaned && !rejected.report.fatal &&
              rejected.report.counts.prefetch_mvins == 1 &&
              rejected.session.failed && rejected.session.live_scale_slots == 0 &&
              rejected.session.outstanding_loops == 0 && rejected.session.outstanding_readbacks == 0);
        const auto attempted = std::find_if(failed_prefetch.records.begin(), failed_prefetch.records.end(),
            [](const potal_paired_cmd_record & record) { return record.funct == POTAL_FUNCT_MVIN; });
        check("D2 attempted prefetch failure starts cleanup with real Fence",
              attempted != failed_prefetch.records.end() && attempted + 1 != failed_prefetch.records.end() &&
              (attempted + 1)->funct == POTAL_FUNCT_FENCE);

        uint64_t windows = 0, columns = 0;
        std::vector<bool> aliases;
        std::vector<size_t> alias_offsets;
        for (const auto & op : plan.ops) {
            if (op.kind != potal_paired_op_kind::residual_gemm) continue;
            const auto & panel = plan.tile.panels[op.panel];
            const auto & window = panel.windows[op.window];
            bool consecutive = true;
            for (size_t i = window.fragment_begin + 1; i < window.fragment_end; ++i)
                consecutive = consecutive && plan.tile.fragments[i].block_id == plan.tile.fragments[i - 1].block_id + 1;
            aliases.push_back(consecutive);
            alias_offsets.push_back((plan.tile.fragments[window.fragment_begin].block_id - panel.block_begin) * op.j_tiles);
            if (consecutive) {
                ++windows;
                const size_t first_col = op.j_group * plan.tile.tile_J * PD;
                columns += (window.fragment_end - window.fragment_begin) *
                           std::min(op.j_tiles * PD, f.N - first_col);
            }
        }
        const auto & a = baseline.report.counts;
        const auto & b = reused.report.counts;
        check("R exact skipped P2 columns/windows, balanced Main releases and real alias fences",
              (windows > 0) == !gapped &&
              b.scale_reused_windows == windows && b.scale_reused_columns == columns &&
              a.scale_writes == b.scale_writes + columns && a.scale_releases == b.scale_releases + columns &&
              b.scale_writes == b.scale_releases && emu.scale_writes == b.scale_writes &&
              emu.scale_releases == b.scale_releases && b.fences == a.fences + windows);
        bool boundaries = true, pairs[2] = { false, false }, wrapped = false;
        size_t residual_loop = 0;
        uint64_t main_key = 0;
        for (size_t i = 0; i < emu.records.size(); ++i) {
            const auto & record = emu.records[i];
            if (record.funct == POTAL_FUNCT_SCALE_WRITE)
                wrapped = wrapped || ((record.rs1 >> 8) & 255) < 250;
            if (record.funct != POTAL_FUNCT_LOOP_WS) continue;
            const size_t selector = (record.rs1 >> 16) & 3;
            if (selector >= 1 && selector <= 2) pairs[selector - 1] = true;
            if (((record.rs1 >> 62) & 1) == 0) {
                main_key = record.rs1;
                continue;
            }
            const size_t window = residual_loop++;
            const bool alias = window < aliases.size() && aliases[window];
            if (alias) {
                // A final ACC readback precedes the logical release. Its
                // CONFIG_ST/MVOUT commands must complete at the same Fence.
                size_t boundary = i + 1;
                while (boundary < emu.records.size() &&
                       (emu.records[boundary].funct == POTAL_FUNCT_CONFIG ||
                        emu.records[boundary].funct == POTAL_FUNCT_MVOUT)) ++boundary;
                boundaries = boundaries && boundary < emu.records.size() &&
                             emu.records[boundary].funct == POTAL_FUNCT_FENCE && emu.records[boundary].xd == 1 &&
                             ((record.rs1 >> 37) & 255) == ((main_key >> 37) & 255) &&
                             ((record.rs1 >> 21) & 65535) == ((main_key >> 21) & 65535) + alias_offsets[window];
            }
        }
        check("R every alias release boundary issues checked Fence, never releases Main early",
              boundaries && residual_loop == aliases.size() && emu.errors.empty() && !emu.scale_miss);
        check("R same LOOP/Index/B loading counts and idle session",
              a.main_loop_ws == b.main_loop_ws && a.residual_loop_ws == b.residual_loop_ws &&
              a.index_configs == b.index_configs && a.index_writes == b.index_writes &&
              baseline_emu.b_loads == emu.b_loads && reused.session.live_scale_slots == 0 &&
              reused.session.outstanding_loops == 0 && reused.session.outstanding_readbacks == 0 &&
              reused.session.index_transaction_open == 0 && !reused.session.failed &&
              reused.generation == baseline.generation);
        check("R no reordered or changed Index/LOOP/loading/readback commands except alias Scale keys",
              same_commands(commands(baseline_emu, baseline, true), commands(emu, reused, true)));
        // Logical generation advancement includes skipped programs. Every actual
        // Scale write must therefore retain its baseline key, generation and carrier.
        size_t baseline_write = 0;
        bool generations = true;
        for (const auto & record : emu.records) {
            if (record.funct != POTAL_FUNCT_SCALE_WRITE) continue;
            while (baseline_write < baseline_emu.records.size()) {
                const auto & original = baseline_emu.records[baseline_write++];
                if (original.funct == record.funct && original.rs1 == record.rs1 &&
                    original.rs2 == record.rs2 && original.xd == record.xd) break;
            }
            generations = generations && baseline_write <= baseline_emu.records.size() &&
                baseline_write > 0 && baseline_emu.records[baseline_write - 1].funct == record.funct &&
                baseline_emu.records[baseline_write - 1].rs1 == record.rs1 &&
                baseline_emu.records[baseline_write - 1].rs2 == record.rs2;
        }
        check("R skipped programs still advance logical generations; actual writes keep baseline generations", generations);
        check("R original validated plan retained", reused.report.plan.ops.size() == plan.ops.size() &&
              reused.report.tile.tile_I == plan.tile.tile_I && reused.report.tile.tile_J == plan.tile.tile_J &&
              reused.report.tile.tile_K == plan.tile.tile_K);
        if (gapped) continue;
        check("R consecutive fixture spans pairs, generation wrap, compact/lane/J/K tails",
              pairs[0] && pairs[1] && wrapped && plan.tile.k_panels > 1 &&
              plan.tile.panels.back().k_tiles < plan.tile.tile_K &&
              plan.tile.residual_pad_I > 0 && plan.tile.pad_J > 0 &&
              plan.tile.residual_filler_slots > 0 && decoded.rows.back().lane > 0);
        const uint32_t generation = reused.generation;
        run_result again = run_paired(f, main, in, emu, true, generation, false, nullptr, 1, &plan);
        check("R repeated reuse call safely recycles both pairs and generations",
              again.status == potal_paired_status::ok && again.main_c == baseline.main_c &&
              again.residual_c == baseline.residual_c && again.generation != generation &&
              again.session.live_scale_slots == 0 && again.session.outstanding_loops == 0 &&
              again.session.outstanding_readbacks == 0 && emu.errors.empty() && !emu.scale_miss);
        h2_emulator recoverable;
        recoverable.fail_funct = POTAL_FUNCT_INDEX_WRITE;
        recoverable.fail_status = POTAL_HW_INVALID_ADDRESS;
        const run_result failed = run_paired(f, main, in, recoverable, true, 250, false, nullptr, 1, &plan);
        check("R recoverable Index failure drains and releases aliased Main allocations",
              failed.status == potal_paired_status::execution_failed &&
              failed.report.hw_status == POTAL_HW_INVALID_ADDRESS && failed.report.cleaned && !failed.report.fatal &&
              failed.session.failed && failed.session.live_scale_slots == 0 &&
              failed.session.outstanding_loops == 0 && failed.session.outstanding_readbacks == 0 &&
              recoverable.scale_writes == recoverable.scale_releases && failed.main_c.empty());
        const pid_t child = fork();
        if (child == 0) {
            h2_emulator fatal;
            fatal.fail_funct = POTAL_FUNCT_FENCE; // First Fence is the first alias release boundary.
            fatal.fail_status = POTAL_ERR_BAD_RESPONSE;
            const run_result failed_alias = run_paired(f, main, in, fatal, true, 250, false, nullptr, 1, &plan);
            const bool correct = failed_alias.status == potal_paired_status::execution_failed &&
                failed_alias.report.hw_status == POTAL_ERR_BAD_RESPONSE && failed_alias.report.fatal &&
                !failed_alias.report.cleaned && failed_alias.main_c.empty() &&
                failed_alias.session.failed && failed_alias.session.live_scale_slots > 0 &&
                count_funct(fatal, POTAL_FUNCT_SCALE_RELEASE) == 0 &&
                count_funct(fatal, POTAL_FUNCT_FENCE) == 2; // Failed boundary plus run_paired housekeeping only.
            _exit(correct ? 0 : 1);
        }
        int status = 0;
        check("R failed alias Fence stays fatal without poisoning parent issuer",
              child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
}

void test_pairs_and_generations() {
    // Six residencies (3 panels x 2 J groups): pairs alternate; the generation counter wraps safely across calls.
    const case_spec c = { "G", 80, 160, 2048, 21, dense_entries(13, 50, 80) };
    paired_fixture  f(c.M, c.N, c.K, c.seed, [](size_t blk, size_t n) { return (int16_t) ((blk * 5 + n) % 3); });
    potal_tile_plan main;
    bool            ok = f.ok && main_select_ok(f, main);
    paired_inputs   in;
    ok = ok && make_inputs(f, main.tile_I * PD, within_k(c.entries, f.K), in);
    const serialized_result ref = serialized_of(f, main.tile_I * PD, in);
    h2_emulator             emu;
    run_result              r = run_paired(f, main, in, emu, true, 250); // generation near the 8-bit wrap
    print_report("G", r);
    const size_t rows = PD == 32 ? 64 : 80;
    const bool first_ok = ok && r.status == potal_paired_status::ok && emu.errors.empty() && compare_stripe(f, in, ref, r, rows).empty();
    check("G three panels across both pairs, generations wrapping at 255: executor == reference", first_ok);
    bool pairs = first_ok;
    size_t loops = 0;
    for (const potal_paired_cmd_record & rec : emu.records) {
        if (rec.funct == POTAL_FUNCT_LOOP_WS && ((rec.rs1 >> 62) & 1) == 0) {
            const size_t residency = loops++;
            pairs = pairs && ((rec.rs1 >> 16) & 3) == (residency % 2) + 1 && ((rec.rs1 >> 18) & 3) == (residency % 2) + 1;
        }
    }
    check("G Main LOOP_WS r selects pair r % 2 (selectors pair + 1); " + std::to_string(loops) + " residencies",
          pairs && loops == (PD == 32 ? 9u : 6u));
    // A second stripe on the same emulator state (B of both pairs still resident): the generations continue.
    const uint32_t gen_after = r.generation;
    run_result     r2        = run_paired(f, main, in, emu, true, gen_after);
    check("G a second call after the first (counter continues, pairs reused after drains): executor == reference",
          r2.status == potal_paired_status::ok && emu.errors.empty() && compare_stripe(f, in, ref, r2, rows).empty() &&
              r2.generation != gen_after);
}

void test_failures() {
    const case_spec c = { "F", 48, 48, 160, 31, dense_entries(7, 40, 100) };
    paired_fixture  f(c.M, c.N, c.K, c.seed, [](size_t blk, size_t n) { return (int16_t) ((blk + n) % 3); });
    potal_tile_plan main;
    bool            ok = f.ok && main_select_ok(f, main);
    paired_inputs   in;
    ok = ok && make_inputs(f, main.tile_I * PD, within_k(c.entries, f.K), in);
    check("F fixture", ok);
    if (!ok) {
        return;
    }
    // An INDEX_WRITE answered INVALID_ADDRESS: the session fails at that stage with work outstanding (fatal).
    {
        h2_emulator emu;
        emu.fail_funct  = POTAL_FUNCT_INDEX_WRITE;
        emu.fail_status = POTAL_HW_INVALID_ADDRESS;
        const run_result r = run_paired(f, main, in, emu);
        print_report("F index_write", r);
        bool fenced_after = false, released_after = false;
        for (size_t i = 0; i < emu.records.size(); ++i) {
            if (emu.records[i].funct == POTAL_FUNCT_INDEX_WRITE) {
                for (size_t j = i + 1; j < emu.records.size(); ++j) {
                    fenced_after |= emu.records[j].funct == POTAL_FUNCT_FENCE;
                    released_after |= emu.records[j].funct == POTAL_FUNCT_SCALE_RELEASE;
                }
            }
        }
        check("F hardware status on INDEX_WRITE: EXECUTION_FAILED at index_write, status propagated, then drained "
              "(fence) and the live scale entries released: cleaned, not fatal",
              r.status == potal_paired_status::execution_failed && r.report.stage != nullptr &&
                  std::string(r.report.stage) == "index_write" && r.report.hw_status == POTAL_HW_INVALID_ADDRESS &&
                  !r.report.fatal && r.report.cleaned && r.session.failed && r.session.outstanding_loops == 0 &&
                  r.session.live_scale_slots == 0 && fenced_after && released_after && emu.scale_writes == emu.scale_releases);
    }
    // A bad FENCE at the end: outstanding readbacks, fatal, nothing claimed.
    {
        h2_emulator emu;
        emu.fail_funct  = POTAL_FUNCT_FENCE;
        emu.fail_status = POTAL_ERR_BAD_RESPONSE;
        const run_result r = run_paired(f, main, in, emu);
        print_report("F fence", r);
        check("F bad final FENCE: EXECUTION_FAILED at fence, not cleaned (a failed fence is never retried), fatal, results not claimed",
              r.status == potal_paired_status::execution_failed && std::string(r.report.stage ? r.report.stage : "") == "fence" &&
                  r.report.fatal && !r.report.cleaned && r.session.failed && r.main_c.empty());
    }
    // A SCALE_WRITE conflict before any LOOP_WS: failed but nothing outstanding (not fatal).
    {
        h2_emulator emu;
        emu.fail_funct  = POTAL_FUNCT_SCALE_WRITE;
        emu.fail_status = POTAL_HW_SCALE_CONFLICT;
        const run_result r = run_paired(f, main, in, emu);
        print_report("F scale_write", r);
        check("F SCALE_WRITE conflict before the first GEMM: failed at scale_write, cleaned, not fatal",
              r.status == potal_paired_status::execution_failed && std::string(r.report.stage ? r.report.stage : "") == "scale_write" &&
                  !r.report.fatal && r.report.cleaned && r.session.outstanding_loops == 0 && r.session.live_scale_slots == 0);
    }
#if !defined(__riscv)
    // Host without a sink: planning done, nothing issued, NO_HARDWARE at the TLB flush, not fatal.
    // (On the target a session without a sink would issue real commands.)
    {
        h2_emulator      emu;
        const run_result r = run_paired(f, main, in, emu, false);
        print_report("F no sink", r);
        check("F host without hardware: plans built, NO_HARDWARE at the first command, nothing outstanding",
              r.status == potal_paired_status::execution_failed && r.report.hw_status == POTAL_ERR_NO_HARDWARE &&
                  !r.report.fatal && r.report.tile.tile_K > 0 && emu.records.empty());
    }
#endif
    // Request rejected before planning: misaligned C.
    {
        potal_paired_session * s = nullptr;
        h2_emulator            emu;
        potal_paired_sink      sink = emu.sink();
        potal_paired_report    rep;
        uint32_t               gen = 0;
        potal_paired_request   q;
        q.rows = 48; q.n = f.N; q.k = f.K; q.main_plan = &main; q.a = f.a.get(); q.lda = f.lda; q.weight = &f.weight;
        q.c_main = reinterpret_cast<int32_t *>(1); q.ldc_main = f.N; q.generation = &gen;
        const bool opened = potal_paired_session_open(&sink, &s) == POTAL_OK;
        const potal_paired_status st = opened ? potal_paired_matmul_auto(q, s, rep) : potal_paired_status::ok;
        check("F misaligned Main C refused before any command",
              opened && st == potal_paired_status::invalid_argument && !rep.issued && emu.records.empty());
        if (opened) {
            (void) potal_paired_session_close(s);
        }
    }
}

} // namespace

namespace {

// ---- X: review follow-ups -----------------------------------------------------------

// Independent golden of one stripe from the fixture's codes and carriers (SCU / Sat32 of the
// RTL test double): Main by DIM K passes in K order; Residual per lane row by run, by DIM
// fragment of the run's compact K, every fragment (zero ones included).
struct direct_golden {
    std::vector<int32_t> main_c, residual_c;
};

direct_golden golden_direct(const paired_fixture & f, size_t rows, const residual::decoded_packet & d) {
    direct_golden g;
    g.main_c.assign(rows * f.N, 0);
    for (size_t m = 0; m < rows; ++m) {
        for (size_t n = 0; n < f.N; ++n) {
            int32_t acc = 0;
            for (size_t p = 0; p < f.K / PD; ++p) {
                int32_t partial = 0;
                for (size_t x = 0; x < PD; ++x) {
                    partial += (int32_t) f.a.get()[m * f.lda + p * PD + x] * f.code_at(p * PD + x, n);
                }
                const int32_t v = emulator::scu(partial, f.carrier_at(p * PD / 32, n));
                acc             = p == 0 ? v : emulator::sat_add(acc, v);
            }
            g.main_c[m * f.N + n] = acc;
        }
    }
    g.residual_c.assign(d.rows.size() * f.N, 0);
    for (size_t i = 0; i < d.rows.size(); ++i) {
        for (size_t n = 0; n < f.N; ++n) {
            int32_t acc = 0;
            bool    first = true;
            size_t  compact = 0;
            for (const residual::run & r : d.runs) {
                for (size_t f0 = 0; f0 < r.local_k.size(); f0 += PD) {
                    int32_t partial = 0;
                    for (size_t x = f0; x < std::min(r.local_k.size(), f0 + PD); ++x) {
                        partial += (int32_t) d.digits[i * d.k_compact + compact + x] *
                                   f.code_at((size_t) r.block_id * 32 + r.local_k[x], n);
                    }
                    const int32_t v = emulator::scu(partial, f.carrier_at(r.block_id, n));
                    acc             = first ? v : emulator::sat_add(acc, v);
                    first           = false;
                }
                compact += r.local_k.size();
            }
            g.residual_c[i * f.N + n] = acc;
        }
    }
    return g;
}

struct direct_run {
    potal_paired_status  status = potal_paired_status::invalid_argument;
    potal_paired_report  report;
    std::vector<int32_t> main_c, residual_c;
};

direct_run run_direct(const paired_fixture & f, const potal_tile_plan & main, size_t rows,
                      const residual::decoded_packet * d, h2_emulator & emu, const potal_paired_execution_plan * plan = nullptr,
                      const potal_paired_geometry * hw = nullptr) {
    direct_run   r;
    const size_t ldc = align_up(f.N, PD);
    std::unique_ptr<int32_t, dma_deleter> cm(static_cast<int32_t *>(dma_alloc(align_up(rows, PD) * ldc * sizeof(int32_t))));
    const size_t lane_rows = d != nullptr ? std::max<size_t>(d->rows.size(), 1) : 1;
    std::unique_ptr<int32_t, dma_deleter> cr(static_cast<int32_t *>(dma_alloc(align_up(lane_rows, PD) * ldc * sizeof(int32_t))));
    potal_paired_session * s    = nullptr;
    potal_paired_sink      sink = emu.sink();
    if (potal_paired_session_open(&sink, &s) != POTAL_OK) {
        return r;
    }
    uint32_t             gen = 0;
    potal_paired_request q;
    q.rows = rows; q.n = f.N; q.k = f.K; q.main_plan = &main; q.a = f.a.get(); q.lda = f.lda; q.weight = &f.weight;
    q.residual = d; q.c_main = cm.get(); q.ldc_main = ldc; q.c_residual = cr.get(); q.ldc_residual = ldc;
    q.generation = &gen; q.plan = plan; q.hw = hw;
    r.status = potal_paired_matmul_auto(q, s, r.report);
    (void) potal_paired_session_close(s);
    if (r.status == potal_paired_status::ok) {
        r.main_c.resize(rows * f.N);
        for (size_t m = 0; m < rows; ++m) {
            std::memcpy(r.main_c.data() + m * f.N, cm.get() + m * ldc, f.N * sizeof(int32_t));
        }
        const size_t lr = r.report.plan.residual_rows.size();
        r.residual_c.resize(lr * f.N);
        for (size_t i = 0; i < lr; ++i) {
            std::memcpy(r.residual_c.data() + i * f.N, cr.get() + i * ldc, f.N * sizeof(int32_t));
        }
    }
    return r;
}

void test_review_followups() {
    paired_fixture  f(16, 64, 128, 77, [](size_t blk, size_t n) { return (int16_t) ((blk * 3 + n) % 4); });
    potal_tile_plan main;
    const bool      ok = f.ok && main_select_ok(f, main);
    check("X fixture", ok);
    if (!ok) {
        return;
    }
    // Zero fragment: run block 0 has 20 compact K (fragments of 16 and 4); the 4-K fragment's digits are zero
    // in every lane row, so it is left out of the windows and the entries, and the results still equal the
    // golden computed over every fragment.
    residual::decoded_packet d;
    d.rows = { { 0, 0 }, { 0, 3 }, { 1, 3 } };
    residual::run r0, zero, r2;
    r0.block_id = 0;
    for (uint16_t k = 0; k < (PD == 32 ? 32u : 20u); ++k) r0.local_k.push_back(k);
    zero.block_id = 1;
    zero.local_k = { 0, 1, 2, 3 };
    r2.block_id = 2;
    r2.local_k  = { 1, 5, 9 };
    d.runs      = PD == 32 ? std::vector<residual::run>{ r0, zero, r2 } : std::vector<residual::run>{ r0, r2 };
    d.k_compact = PD + 7;
    d.digits.assign(d.rows.size() * d.k_compact, 0);
    for (size_t i = 0; i < d.rows.size(); ++i) {
        for (size_t c = 0; c < d.k_compact; ++c) {
            d.digits[i * d.k_compact + c] = (c >= PD && c < PD + 4) ? 0 : (int8_t) ((int) (i * 37 + c * 11) % 255 - 127);
        }
    }
    h2_emulator      emu;
    const direct_run run = run_direct(f, main, 16, &d, emu);
    const direct_golden g = golden_direct(f, 16, d);
    check("X zero fragment: skipped (1 of 3), only its 4 K missing from the Index writes, Main and Residual == golden",
          run.status == potal_paired_status::ok && emu.errors.empty() && run.report.tile.residual_zero_fragments == 1 &&
              run.report.tile.fragments.size() == 2 &&
              run.report.counts.index_writes == run.report.tile.j_groups * (PD + 3) && run.main_c == g.main_c &&
              run.residual_c == g.residual_c);

    // The caller's plan is executed as is (no re-planning); a plan of another request is refused before any command.
    potal_paired_tile_plan      tile;
    potal_paired_execution_plan plan;
    std::string                 why;
    const bool planned = potal_paired_plan_stripe(main, 16, f.N, f.K, &d, tile, plan, &why) == potal_paired_status::ok;
    h2_emulator      emu2;
    const direct_run given = run_direct(f, main, 16, &d, emu2, planned ? &plan : nullptr);
    check("X the routing plan handed to the executor runs as is: same command stream, same results",
          planned && given.status == potal_paired_status::ok && given.main_c == run.main_c &&
              given.residual_c == run.residual_c && emu2.records.size() == emu.records.size());
    potal_paired_execution_plan other = plan;
    other.tile.rows = 15;
    h2_emulator      emu3;
    const direct_run wrong = run_direct(f, main, 16, &d, emu3, &other);
    check("X a plan describing another request is refused before any command",
          wrong.status == potal_paired_status::invalid_argument && std::string(wrong.report.stage ? wrong.report.stage : "") == "plan" &&
              emu3.records.empty());

    // More rows than one Main I group: refused before any command (never truncated).
    potal_tile_plan small = main;
    small.tile_I = 1; // DIM rows per group
    paired_fixture  f32(2 * PD, 64, 128, 78, [](size_t blk, size_t n) { return (int16_t) ((blk + n) % 2); });
    h2_emulator     emu4;
    const direct_run multi = f32.ok ? run_direct(f32, small, 2 * PD, nullptr, emu4) : direct_run{};
    check("X rows beyond the Main plan's I group refused (planner), no command",
          multi.status == potal_paired_status::invalid_argument && emu4.records.empty());

    // A packet with runs but no rows: the planner's refusal, not silently "no residual".
    residual::decoded_packet runs_only = d;
    runs_only.rows.clear();
    runs_only.digits.clear();
    h2_emulator      emu5;
    const direct_run bad = run_direct(f, main, 16, &runs_only, emu5);
    check("X residual with runs but no rows refused, no command",
          bad.status == potal_paired_status::invalid_argument && emu5.records.empty());

    // The owner step: off the owner's CPU a fatal refusal before any session.
    potal_paired_report  rep;
    uint32_t             gen = 0;
    potal_paired_request q;
    const potal_paired_status off = run_paired_on_owner(q, &gen, false, rep);
    check("X owner step off the owner's CPU: fatal 'affinity' refusal, no session opened",
          off == potal_paired_status::execution_failed && rep.fatal && std::string(rep.stage ? rep.stage : "") == "affinity" &&
              !rep.issued);
}

void test_conservative() {
    const case_spec c = { "C", 80, 160, 2048, 91, dense_entries(9, 61, 150) };
    std::string     why_rtl, why_cons;
    run_result      rtl, cons;
    h2_emulator     emu_rtl, emu_cons;
    const bool      ok_rtl  = run_case(c, true, why_rtl, &rtl, &emu_rtl, false);
    const bool      ok_cons = run_case(c, true, why_cons, &cons, &emu_cons, true);
    check("C conservative fences: same numerics as the reference (" + why_cons + ")", ok_rtl && ok_cons);
    // Every Indexed LOOP_WS and MVOUT has no unfenced LOOP_WS before it; every SCALE_RELEASE no unfenced LOOP_WS / MVOUT.
    bool   placed = true, loop_since = false, any_since = false;
    size_t checked = 0;
    for (const potal_paired_cmd_record & rec : emu_cons.records) {
        if (rec.funct == POTAL_FUNCT_FENCE) {
            loop_since = any_since = false;
        } else if (rec.funct == POTAL_FUNCT_LOOP_WS) {
            if (((rec.rs1 >> 62) & 1) != 0) {
                placed = placed && !loop_since;
                ++checked;
            }
            loop_since = any_since = true;
        } else if (rec.funct == POTAL_FUNCT_MVOUT) {
            placed = placed && !loop_since;
            any_since = true;
            ++checked;
        } else if (rec.funct == POTAL_FUNCT_SCALE_RELEASE) {
            placed = placed && !any_since;
            ++checked;
        }
    }
    check("C conservative: a FENCE before every Indexed LOOP_WS, ACC readback and scale release; RTL mode only the final one",
          ok_cons && placed && checked > 0 && cons.report.counts.fences > rtl.report.counts.fences &&
              rtl.report.counts.fences == 1 && cons.report.counts.main_loop_ws == rtl.report.counts.main_loop_ws &&
              cons.report.counts.residual_loop_ws == rtl.report.counts.residual_loop_ws);
}

std::string capture_window_diag(const char * value, const std::function<void()> & run) {
    const char * previous = std::getenv("POTAL_PAIRED_WINDOW_DIAG");
    const bool had_previous = previous != nullptr;
    const std::string saved_value = had_previous ? previous : "";
    std::fflush(stderr);
    FILE * capture = std::tmpfile();
    const int saved_fd = dup(STDERR_FILENO);
    const bool ready = capture != nullptr && saved_fd >= 0 && dup2(fileno(capture), STDERR_FILENO) >= 0;
    check("D stderr capture", ready);
    if (!ready) {
        if (capture != nullptr) std::fclose(capture);
        if (saved_fd >= 0) close(saved_fd);
        return {};
    }
    if (value != nullptr) setenv("POTAL_PAIRED_WINDOW_DIAG", value, 1);
    else unsetenv("POTAL_PAIRED_WINDOW_DIAG");
    run();
    std::fflush(stderr);
    const bool restored = dup2(saved_fd, STDERR_FILENO) >= 0;
    close(saved_fd);
    if (had_previous) setenv("POTAL_PAIRED_WINDOW_DIAG", saved_value.c_str(), 1);
    else unsetenv("POTAL_PAIRED_WINDOW_DIAG");
    check("D stderr restored", restored);
    std::rewind(capture);
    std::string text;
    char buffer[4096];
    size_t bytes;
    while ((bytes = std::fread(buffer, 1, sizeof(buffer), capture)) != 0) text.append(buffer, bytes);
    std::fclose(capture);
    return text;
}

size_t window_diag_field(const std::string & line, const char * name) {
    const std::string key = std::string(" ") + name + "=";
    const size_t p = line.find(key);
    return p == std::string::npos ? SIZE_MAX : (size_t) std::strtoull(line.c_str() + p + key.size(), nullptr, 10);
}

std::string capture_index_diag(const char * entries, const char * window, const std::function<void()> & run) {
    const char * previous = std::getenv("POTAL_PAIRED_DIAG_INDEX_ENTRIES");
    const bool had_previous = previous != nullptr;
    const std::string saved = had_previous ? previous : "";
    if (entries != nullptr) setenv("POTAL_PAIRED_DIAG_INDEX_ENTRIES", entries, 1);
    else unsetenv("POTAL_PAIRED_DIAG_INDEX_ENTRIES");
    const std::string output = capture_window_diag(window, run);
    if (had_previous) setenv("POTAL_PAIRED_DIAG_INDEX_ENTRIES", saved.c_str(), 1);
    else unsetenv("POTAL_PAIRED_DIAG_INDEX_ENTRIES");
    return output;
}

void test_index16_plan() {
    paired_fixture f(16, 32, 128, 111, [](size_t block, size_t n) { return (int16_t) ((block + n) % 4); });
    potal_tile_plan main;
    if (!f.ok || !main_select_ok(f, main)) {
        check("I16 fixture", false);
        return;
    }
    residual::decoded_packet d;
    d.rows = { { 0, 0 }, { 0, 3 }, { 1, 3 } };
    residual::run r0, r2;
    r0.block_id = 0;
    for (uint16_t k = 0; k < 32; ++k) if (k != 16) r0.local_k.push_back(k);
    r2.block_id = 2;
    r2.local_k = { 1, 5, 9 };
    d.runs = { r0, r2 };
    d.k_compact = 34;
    d.digits.resize(d.rows.size() * d.k_compact);
    for (size_t i = 0; i < d.digits.size(); ++i) d.digits[i] = (int8_t) ((int) (i * 11 % 255) - 127);

    const potal_paired_geometry physical = potal_paired_hw_geometry();
    potal_paired_geometry single = physical;
    single.index_entries = PD;
    potal_paired_tile_plan default_tile, physical_tile, single_tile;
    potal_paired_execution_plan default_plan, physical_plan, single_plan;
    std::string why;
    const bool planned =
        potal_paired_plan_stripe(main, 16, f.N, f.K, &d, default_tile, default_plan, &why) == potal_paired_status::ok &&
        potal_paired_plan_stripe(main, 16, f.N, f.K, &d, physical_tile, physical_plan, &why, physical) == potal_paired_status::ok &&
        potal_paired_plan_stripe(main, 16, f.N, f.K, &d, single_tile, single_plan, &why, single) == potal_paired_status::ok;
    check("Index plans from physical two-fragment geometry and a local one-fragment copy (" + why + ")", planned &&
          physical.index_entries == 2 * PD && POTAL_HW_INDEX_ENTRIES == 2 * PD && single.index_entries == PD);
    if (!planned) return;

    bool unchanged = default_plan.ops.size() == physical_plan.ops.size() &&
                     default_plan.index_entries.size() == physical_plan.index_entries.size();
    for (size_t i = 0; i < default_plan.ops.size() && unchanged; ++i)
        unchanged = potal_paired_op_text(default_plan.ops[i]) == potal_paired_op_text(physical_plan.ops[i]);
    for (size_t i = 0; i < default_plan.index_entries.size() && unchanged; ++i) {
        const auto & a = default_plan.index_entries[i];
        const auto & b = physical_plan.index_entries[i];
        unchanged = a.slot == b.slot && a.fragment == b.fragment && a.original_k == b.original_k &&
                    a.resident_k == b.resident_k && a.compact_k == b.compact_k && a.filler == b.filler;
    }
    check("Index default physical plan stays unchanged; one-fragment plan preserves Main I/J/K", unchanged &&
          default_tile.fragments_per_window == 2 && single_tile.fragments_per_window == 1 &&
          default_tile.tile_I == single_tile.tile_I && default_tile.tile_J == single_tile.tile_J &&
          default_tile.tile_K == single_tile.tile_K && single_tile.residual_windows == single_tile.fragments.size());

    potal_paired_staging staging;
    bool packed = potal_validate_execution_plan(single_plan, &d, &why, single) == potal_paired_status::ok &&
                  potal_paired_pack_residual(single_plan, d, staging, &why, single) == potal_paired_status::ok;
    size_t windows = 0;
    for (const auto & op : single_plan.ops) {
        if (op.kind != potal_paired_op_kind::index_program) continue;
        const auto & panel = single_tile.panels[op.panel];
        const auto & window = panel.windows[op.window];
        const auto & fragment = single_tile.fragments[window.fragment_begin];
        const auto & run = d.runs[fragment.run];
        packed = packed && window.fragment_end == window.fragment_begin + 1 && window.k_tiles == 1 &&
                 window.active_rows == fragment.count && window.active_rows <= PD &&
                 window.pad_k == PD - window.active_rows && op.entry_end - op.entry_begin == fragment.count;
        size_t sw = 0;
        while (sw < staging.windows.size() &&
               (staging.windows[sw].panel != op.panel || staging.windows[sw].index != op.window)) ++sw;
        packed = packed && sw < staging.windows.size();
        if (!packed) break;
        for (size_t x = 0; x < fragment.count; ++x) {
            const auto & entry = single_plan.index_entries[op.entry_begin + x];
            const size_t original_k = (size_t) run.block_id * 32 + run.local_k[fragment.local_begin + x];
            packed = packed && !entry.filler && entry.slot == x && entry.slot < physical.index_entries &&
                     entry.original_k == original_k && entry.resident_k == original_k - panel.k_begin &&
                     entry.compact_k == fragment.compact_begin + x;
            for (size_t row = 0; row < d.rows.size(); ++row)
                packed = packed && staging.a(sw)[row * staging.windows[sw].lda + x] ==
                                   d.digits[row * d.k_compact + fragment.compact_begin + x];
        }
        for (size_t x = fragment.count; x < PD; ++x)
            for (size_t row = 0; row < d.rows.size(); ++row)
                packed = packed && staging.a(sw)[row * staging.windows[sw].lda + x] == 0;
        ++windows;
    }
    check("I16 one-fragment windows, packed original digits, Index K mapping and zero tail", packed &&
          windows == single_tile.j_groups * single_tile.fragments.size());

    h2_emulator emu;
    direct_run result;
    const std::string log = capture_window_diag("1", [&] {
        result = run_direct(f, main, 16, &d, emu, &single_plan, &single);
    });
    const direct_golden golden = golden_direct(f, 16, d);
    check("I16 H2 Main and Residual INT32 equal independent original-K golden; every prefix matches",
          result.status == potal_paired_status::ok && emu.errors.empty() && result.main_c == golden.main_c &&
          result.residual_c == golden.residual_c && log.find("[POTAL_WINDOW_DIAG]") != std::string::npos &&
          log.find("[POTAL_WINDOW_MISMATCH]") == std::string::npos);

    bool checked = true, scales = true;
    size_t indexed_loops = 0, residual_scales = 0;
    for (const auto & record : emu.records) {
        if (record.funct == POTAL_FUNCT_INDEX_CONFIG)
            checked = checked && (record.rs2 >> 32) <= PD;
        if (record.funct == POTAL_FUNCT_INDEX_WRITE)
            checked = checked && record.rs1 < PD && record.rs1 < physical.index_entries;
        if (record.funct == POTAL_FUNCT_LOOP_WS && ((record.rs1 >> 62) & 1) != 0) {
            checked = checked && ((record.rs1 >> 45) & 0xFFFF) == 1;
            ++indexed_loops;
        }
        if (record.funct == POTAL_FUNCT_SCALE_WRITE) {
            const size_t generation = (record.rs1 >> 8) & 0xFF;
            if (generation == 0 || generation > single_plan.scale_windows.size()) { scales = false; continue; }
            const auto & scale = single_plan.scale_windows[generation - 1];
            if (!scale.residual) continue;
            const auto & residency = single_plan.residencies[scale.residency];
            const auto & window = single_tile.panels[residency.panel].windows[scale.window];
            const auto & fragment = single_tile.fragments[window.fragment_begin];
            const size_t slot = record.rs1 & 0xFF, col = (record.rs1 >> 16) & 0x3F;
            const size_t original_col = residency.j_group * single_tile.tile_J * PD +
                                        (slot - scale.slot_base) * PD + col;
            scales = scales && scale.blocks == 1 && slot >= scale.slot_base &&
                     slot < scale.slot_base + scale.j_tiles && record.rs2 == f.carrier_at(fragment.block_id, original_col);
            ++residual_scales;
        }
    }
    check("Index checked C11 commands stay within one-fragment/physical table bounds, ppb1, original block carrier", checked &&
          scales && residual_scales > 0 && indexed_loops == single_tile.residual_windows * single_tile.j_groups);
    if (PD == 32) {
        check("Index DIM32 physical64 versus local32: two fragments, one versus two windows",
              default_tile.fragments.size() == 2 && default_tile.residual_windows == 1 &&
                  single_tile.residual_windows == 2 && single_tile.residual_filler_slots == 0 &&
                  default_tile.residual_filler_slots == 1 && single_tile.panels[0].windows[0].active_rows == 31 &&
                  single_tile.panels[0].windows[1].active_rows == 3);
    }

    potal_paired_geometry wrong = single;
    wrong.index_entries = 8;
    h2_emulator rejected;
    const direct_run mismatch = run_direct(f, main, 16, &d, rejected, &single_plan, &wrong);
    check("I16 executor validates the given plan against request.hw before any hardware command",
          mismatch.status == potal_paired_status::invalid_execution_plan && rejected.records.empty());
}

void test_window_diag() {
    paired_fixture f(17, 140, 2048, 101, [](size_t block, size_t) {
        const int16_t exponents[] = { 20, 0, 1, INT16_MIN };
        return exponents[block % 4];
    });
    potal_tile_plan main;
    paired_inputs in;
    const bool ready = f.ok && main_select_ok(f, main) && make_inputs(f, main.tile_I * PD, [](size_t, size_t rows) {
        std::vector<res_entry> entries;
        const size_t counts[] = { 3, 19, 32 };
        for (size_t row = 0; row < rows; row += 8) {
            for (size_t block = 0; block < 64; ++block) {
                for (size_t k = 0; k < counts[block % 3]; ++k) {
                    entries.push_back({ row, block * 32 + k, (int32_t) (300 + (row + k * 17) % 127) });
                }
            }
        }
        return entries;
    }, in);
    check("D fixture: original K gaps, fragment tails, mixed carriers and radix lanes", ready);
    if (!ready) return;

    h2_emulator normal_emu, diag_emu;
    run_result normal, diag;
    const std::string normal_log = capture_window_diag(nullptr, [&] { normal = run_paired(f, main, in, normal_emu); });
    const std::string diag_log = capture_window_diag("1", [&] { diag = run_paired(f, main, in, diag_emu); });
    const size_t windows = diag.report.counts.residual_loop_ws;
    const bool ran = normal.status == potal_paired_status::ok && diag.status == potal_paired_status::ok &&
                     normal_emu.errors.empty() && diag_emu.errors.empty() && windows > 0;
    check("D snapshots preserve Main and Residual hardware output", ran && normal.main_c == diag.main_c &&
          normal.residual_c == diag.residual_c && normal_log.find("[POTAL_WINDOW_") == std::string::npos);
    check("D diagnostic fences and readbacks stay outside production counters", ran &&
          normal.report.counts.fences == diag.report.counts.fences &&
          normal.report.counts.readbacks == diag.report.counts.readbacks &&
          normal.report.counts.readback_tiles == diag.report.counts.readback_tiles &&
          diag_emu.fences == normal_emu.fences + 2 * windows);

    bool summaries = ran, many_windows = false;
    size_t records = 0, pos = 0;
    while ((pos = diag_log.find("[POTAL_WINDOW_DIAG]", pos)) != std::string::npos) {
        const std::string line = diag_log.substr(pos, diag_log.find('\n', pos) - pos);
        const size_t group = window_diag_field(line, "j_group");
        const size_t cols = group < diag.report.tile.j_groups
                                ? std::min(diag.report.tile.tile_J * PD, f.N - group * diag.report.tile.tile_J * PD) : 0;
        summaries = summaries && cols != 0 && window_diag_field(line, "prefix_bad") == 0 &&
                    window_diag_field(line, "prefix_total") == diag.rows.size() * cols &&
                    window_diag_field(line, "pair") < 2 &&
                    window_diag_field(line, "diag_fences") == 2 * (records + 1) &&
                    window_diag_field(line, "diag_readbacks") == records + 1;
        ++records;
        ++pos;
    }
    for (const potal_paired_k_panel & panel : diag.report.tile.panels) many_windows |= panel.windows.size() > 1;
    check("D every cumulative snapshot matches across J groups, panels, filler slots and tails", summaries &&
          records == windows && many_windows && diag.report.tile.j_groups > 1 && diag.report.tile.k_panels > 1 &&
          diag.report.tile.residual_filler_slots > 0 && diag.report.tile.pad_J != 0 &&
          diag_log.find("[POTAL_WINDOW_MISMATCH]") == std::string::npos &&
          diag_log.find("[POTAL_WINDOW_INPUT]") != std::string::npos &&
          diag_log.find("original_k=") != std::string::npos && diag_log.find("resident_k=") != std::string::npos &&
          diag_log.find("carriers=") != std::string::npos);

    bool ordered = ran, separate = ran;
    size_t snapshots = 0, extra_tiles = 0;
    const uintptr_t main_begin = reinterpret_cast<uintptr_t>(diag.c_main.get());
    const uintptr_t main_end = main_begin + align_up(f.M, PD) * align_up(f.N, PD) * sizeof(int32_t);
    const uintptr_t residual_begin = reinterpret_cast<uintptr_t>(diag.c_res.get());
    const uintptr_t residual_end = residual_begin + align_up(diag.rows.size(), PD) * align_up(f.N, PD) * sizeof(int32_t);
    for (size_t i = 0; i < diag_emu.records.size(); ++i) {
        const potal_paired_cmd_record & rec = diag_emu.records[i];
        if (rec.funct != POTAL_FUNCT_LOOP_WS || ((rec.rs1 >> 62) & 1) == 0) continue;
        size_t p = i + 1;
        ordered = ordered && p < diag_emu.records.size() && diag_emu.records[p++].funct == POTAL_FUNCT_FENCE;
        ordered = ordered && p < diag_emu.records.size() && diag_emu.records[p++].funct == POTAL_FUNCT_CONFIG;
        size_t tiles = 0;
        while (p < diag_emu.records.size() && diag_emu.records[p].funct == POTAL_FUNCT_MVOUT) {
            const uintptr_t address = (uintptr_t) diag_emu.records[p].rs1;
            separate = separate && !(address >= main_begin && address < main_end) &&
                       !(address >= residual_begin && address < residual_end) && address % (PD * sizeof(int32_t)) == 0;
            ++tiles;
            ++p;
        }
        ordered = ordered && tiles != 0 && p < diag_emu.records.size() &&
                  diag_emu.records[p].funct == POTAL_FUNCT_FENCE;
        extra_tiles += tiles;
        ++snapshots;
    }
    check("D every Indexed LOOP_WS immediately gets fence, separate aligned ACC1 readback, fence",
          ordered && separate && snapshots == windows &&
          diag_emu.mvouts == normal_emu.mvouts + extra_tiles);

    struct corrupt_index {
        h2_emulator emu;
        static potal_status command(void * context, const potal_paired_cmd_record * record) {
            auto & self = *static_cast<corrupt_index *>(context);
            potal_paired_cmd_record changed = *record;
            if (changed.funct == POTAL_FUNCT_INDEX_WRITE) {
                const auto & table = self.emu.tables[self.emu.selected_table];
                changed.rs2 = (changed.rs2 + 1) % table.source_k_rows;
            }
            return self.emu.command(changed);
        }
    } corrupted;
    const potal_paired_sink sink = { &corrupt_index::command, &corrupted };
    run_result bad;
    const std::string bad_log = capture_window_diag("1", [&] {
        bad = run_paired(f, main, in, corrupted.emu, true, 0, false, &sink);
    });
    const size_t first_record = bad_log.find("[POTAL_WINDOW_DIAG]");
    const std::string first_line = first_record == std::string::npos ? "" :
        bad_log.substr(first_record, bad_log.find('\n', first_record) - first_record);
    check("D corrupted hardware Index lookup differs from independent original-K reference in first window",
          bad.status == potal_paired_status::ok && corrupted.emu.errors.empty() && !first_line.empty() &&
          window_diag_field(first_line, "prefix_bad") > 0 &&
          window_diag_field(first_line, "prefix_bad") != SIZE_MAX &&
          bad_log.find("[POTAL_WINDOW_MISMATCH]") != std::string::npos &&
          bad_log.find("lane_row=") != std::string::npos && bad_log.find("source_row=") != std::string::npos &&
          bad_log.find(" got=") != std::string::npos && bad_log.find(" expected=") != std::string::npos &&
          bad.main_c == normal.main_c && bad.residual_c != normal.residual_c &&
          bad.report.counts.ops == normal.report.counts.ops &&
          bad.report.counts.residual_loop_ws == windows);

    h2_emulator disabled_emu;
    run_result disabled;
    const std::string disabled_log = capture_window_diag("true", [&] {
        disabled = run_paired(f, main, in, disabled_emu);
    });
    check("D only exact POTAL_PAIRED_WINDOW_DIAG=1 enables snapshots", disabled.status == potal_paired_status::ok &&
          disabled_emu.records.size() == normal_emu.records.size() && disabled_log.find("[POTAL_WINDOW_") == std::string::npos);

    struct fail_diagnostic_command {
        h2_emulator emu;
        uint32_t funct;
        size_t ordinal, seen = 0;
        static potal_status command(void * context, const potal_paired_cmd_record * record) {
            auto & self = *static_cast<fail_diagnostic_command *>(context);
            if (record->funct == self.funct && ++self.seen == self.ordinal) {
                self.emu.fail_funct = self.funct;
                self.emu.fail_status = POTAL_ERR_BAD_RESPONSE;
            }
            return self.emu.command(*record);
        }
    };
    for (size_t fence = 1; fence <= 2; ++fence) {
        // A fatal early fence intentionally keeps the accelerator owned; contain that state in a child.
        const pid_t child = fork();
        if (child == 0) {
            fail_diagnostic_command failure = { {}, POTAL_FUNCT_FENCE, fence };
            const potal_paired_sink failing_sink = { &fail_diagnostic_command::command, &failure };
            run_result failed;
            capture_window_diag("1", [&] {
                failed = run_paired(f, main, in, failure.emu, true, 0, false, &failing_sink);
            });
            const bool correct = failed.status == potal_paired_status::execution_failed &&
                failed.report.hw_status == POTAL_ERR_BAD_RESPONSE &&
                std::string(failed.report.stage ? failed.report.stage : "") == "fence" && failed.report.fatal &&
                !failed.report.cleaned && failed.main_c.empty() &&
                count_funct(failure.emu, POTAL_FUNCT_FENCE) == fence + 1; // One test-housekeeping fence in run_paired.
            _exit(correct ? 0 : 1);
        }
        int status = 0;
        const bool correct = child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        check("D failed diagnostic fence " + std::to_string(fence) + " stays fatal and is never retried by executor", correct);
    }
    fail_diagnostic_command failure = { {}, POTAL_FUNCT_MVOUT, 1 };
    const potal_paired_sink failing_sink = { &fail_diagnostic_command::command, &failure };
    run_result failed;
    capture_window_diag("1", [&] {
        failed = run_paired(f, main, in, failure.emu, true, 0, false, &failing_sink);
    });
    check("D failed diagnostic readback propagates hardware status and releases live scales",
          failed.status == potal_paired_status::execution_failed && failed.report.hw_status == POTAL_ERR_BAD_RESPONSE &&
          std::string(failed.report.stage ? failed.report.stage : "") == "window_diag_readback" &&
          failed.report.cleaned && !failed.report.fatal && failed.session.live_scale_slots == 0);
}

// Last in main(): leaves the accelerator owned on purpose (a failed cleanup keeps it).
void test_close_keeps_owner_with_live_entries() {
    h2_emulator            emu;
    potal_paired_sink      sink = emu.sink();
    potal_paired_session * s    = nullptr;
    bool ok = potal_paired_session_open(&sink, &s) == POTAL_OK && potal_paired_scale_write(s, 7, 3, 0, 0) == POTAL_OK &&
              potal_paired_scale_write(s, 7, 3, 1, 0) == POTAL_OK;
    potal_index_config cfg = { 0, (uint32_t) (POTAL_HW_BANK_ROWS * 2 - 2 * PD), 1, 2 * PD, 4 };
    emu.fail_funct  = POTAL_FUNCT_INDEX_CONFIG;
    emu.fail_status = POTAL_HW_INVALID_ADDRESS;
    ok = ok && potal_paired_index_config(s, &cfg) == POTAL_HW_INVALID_ADDRESS;
    emu.fail_funct  = POTAL_FUNCT_SCALE_RELEASE;
    emu.fail_status = POTAL_HW_RELEASE_MISS;
    ok = ok && potal_paired_session_cleanup(s) == POTAL_HW_RELEASE_MISS && potal_paired_session_query(s).live_scale_slots == 1;
    ok = ok && potal_paired_session_close(s) == POTAL_HW_INVALID_ADDRESS;
    potal_paired_session * again = nullptr;
    check("X a failed session whose cleanup could not release its scale entries keeps the accelerator owned after close",
          ok && potal_paired_session_open(&sink, &again) == POTAL_ERR_BUSY);
}

// ---- S: stripe pipeline integration -------------------------------------------------

// One F32 x Q8_HP1 MUL_MAT for the pipeline: ggml-quantized weight, activations with
// outliers on every `outlier_step`-th row (ExSIA seals residual packets with RMD ON).
struct pipe_node {
    size_t           M, N, K;
    ggml_context_ptr ctx;
    ggml_tensor *    w = nullptr;
    ggml_tensor *    x = nullptr;
    ggml_tensor *    y = nullptr;
    hp1_shape        s;
    potal_tile_plan  plan;
    hp1_weight       weight;
    bool             ok = false;

    pipe_node(size_t M_, size_t N_, size_t K_, uint32_t seed, float outlier, size_t outlier_step, size_t outliers_per_row) :
        M(M_), N(N_), K(K_) {
        ggml_init_params ip = { (M * K + N * K) * 8 + 4 * 1024 * 1024, nullptr, false };
        ctx.reset(ggml_init(ip));
        w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, (int64_t) K, (int64_t) N);
        x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, (int64_t) K, (int64_t) M);
        y = ggml_mul_mat(ctx.get(), w, x);
        std::mt19937                    rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float>              wf(N * K);
        const float                     mag[] = { 0.02f, 0.08f, 0.3f, 0.01f };
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                wf[n * K + k] = nd(rng) * mag[(k / 32 + n) % 4];
            }
        }
        ggml_quantize_chunk(GGML_TYPE_Q8_HP1, wf.data(), w->data, 0, (int64_t) N, (int64_t) K, nullptr);
        float * v = static_cast<float *>(x->data);
        for (size_t m = 0; m < M; ++m) {
            for (size_t k = 0; k < K; ++k) {
                v[m * K + k] = nd(rng) * (1.0f + 0.5f * (float) (m % 3)) * (k / 32 % 2 ? 4.0f : 1.0f);
            }
            if (outlier != 0.0f && m % outlier_step == 0) {
                for (size_t o = 0; o < outliers_per_row; ++o) {
                    v[m * K + (m * 7 + o * 61) % K] = outlier * ((m + o) % 4 ? -1.0f : 1.0f) * (1.0f + 0.25f * (float) o);
                }
            }
        }
        s  = { (int64_t) M, (int64_t) N, (int64_t) K };
        ggml_gemmini_args_t args;
        ok = potal_tile_select(M, N, K, 32 / PD, &plan) == POTAL_OK && args_for(args) &&
             weight.allocate((int64_t) N, (int64_t) K) &&
             prepare_weight(args, weight) == quants::weight::hp1::status::ok;
    }

    bool args_for(ggml_gemmini_args_t & args) const {
        return init_args(w, y, s, "gpt2", "paired-pipeline-test", ggml::gemmini::residual::ResidualRoute::ws_packet,
                         plan, args);
    }
    size_t  stripes() const { return (M + plan.tile_I * PD - 1) / (plan.tile_I * PD); }
    float * dst() const { return static_cast<float *>(y->data); }
};

struct event_record {
    stripe_event_kind kind;
    uint64_t          sequence;
    size_t            stripe;
    std::string       route;
};

struct pipe_result {
    pipeline_report           report;
    consume_outcome           consumer;
    std::vector<float>        out;
    bool                      untouched = false;
    std::vector<event_record> events;
    size_t                    paired_calls = 0;
    uint64_t                  decode_calls = 0; // only the actual pipeline execution, not fixture/golden work
    std::string               planned_commands; // emulator record of the serial (planned) GEMMs
    uint64_t                  h2_main = 0, h2_residual = 0, h2_index = 0;
    std::string               h2_errors;
    bool                      conservative_req = false; // a request carried conservative_fences
};

const uint32_t sentinel_bits = 0x7fc0beefu;

// mode: 0 serial, 1 paired, 2 paired configured without a paired runner, 3 paired runner reporting fatal,
// 4 paired with conservative fences.
pipe_result run_pipe(pipe_node & nd, size_t capacity, int mode,
                     const std::function<void(const potal_paired_request &)> & inspect = {}, bool ktile = false,
                     bool prefetch = false, bool early = false,
                     potal_paired_stream_mode stream = potal_paired_stream_mode::off) {
    pipe_result         r;
    ggml_gemmini_args_t args;
    if (!nd.args_for(args)) {
        r.report.stage = "fixture";
        return r;
    }
    float sentinel;
    std::memcpy(&sentinel, &sentinel_bits, sizeof(sentinel));
    std::fill(nd.dst(), nd.dst() + nd.M * nd.N, sentinel);
    std::mutex      ev_mutex;
    pipeline_config cfg;
    cfg.job_capacity = capacity;
    cfg.residual     = options::compensation;
    cfg.paired       = mode != 0;
    cfg.paired_conservative_fences = mode == 4;
    cfg.paired_ktile_scale_reuse = ktile;
    cfg.paired_next_k_prefetch = prefetch;
    cfg.paired_early_k_prefetch = early;
    cfg.paired_stream = stream;
    cfg.layer        = "paired-pipeline-test";
    cfg.backend      = "EMULATED";
    cfg.observer     = [&](const stripe_event & ev) {
        std::lock_guard<std::mutex> lock(ev_mutex);
        r.events.push_back({ ev.kind, ev.sequence, ev.stripe_id, ev.route ? ev.route : "" });
    };
    potal_test::emu_session es;
    h2_emulator             h2;
    uint32_t                generation  = 0;
    bool                    fences_seen = false;
    paired_runner           runner;
    if (mode == 1 || mode == 3 || mode == 4) {
        runner = [&](const potal_paired_request & req, potal_paired_report & rep) {
            r.paired_calls++;
            if (inspect) inspect(req);
            if (mode == 3) {
                rep        = potal_paired_report{};
                rep.status = potal_paired_status::execution_failed;
                rep.stage  = "injected";
                rep.issued = true;
                rep.fatal  = true;
                return rep.status;
            }
            potal_paired_request q = req;
            q.generation           = &generation;
            if (q.early_k_prefetch) q.early_counter = h2_emulator::on_counter;
            if (q.stream != potal_paired_stream_mode::off) q.stream_counter = h2_emulator::on_counter;
            fences_seen            = fences_seen || q.conservative_fences;
            const potal_paired_sink sink = h2.sink();
            return potal_paired_run_session(q, &sink, rep);
        };
    }
    potal_test::thread_owner owner([&es](const potal_stripe_gemm_desc & d) { return es.run(d); }, runner);
    residual::reset_decode_packet_calls();
    const uint64_t decode_before = residual::decode_packet_calls();
    r.report = run_stripe_pipeline(nd.x, args, nd.weight, nd.plan, cfg, owner);
    owner.join();
    r.decode_calls = residual::decode_packet_calls() - decode_before;
    r.consumer         = owner.outcome;
    r.planned_commands = es.emu.commands;
    r.h2_main          = h2.main_loops;
    r.h2_residual      = h2.residual_loops;
    r.h2_index         = h2.index_configs;
    r.h2_errors        = h2.errors + (h2.scale_miss ? " scale miss" : "");
    r.conservative_req = fences_seen;
    r.out.assign(nd.dst(), nd.dst() + nd.M * nd.N);
    r.untouched = std::all_of(r.out.begin(), r.out.end(), [](float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return bits == sentinel_bits;
    });
    std::sort(r.events.begin(), r.events.end(), [](const event_record & a, const event_record & b) {
        return a.sequence < b.sequence;
    });
    return r;
}

long seq_of(const pipe_result & r, stripe_event_kind kind, size_t stripe) {
    for (const event_record & e : r.events) {
        if (e.kind == kind && e.stripe == stripe) {
            return (long) e.sequence;
        }
    }
    return -1;
}

std::string route_of(const pipe_result & r, size_t stripe) {
    for (const event_record & e : r.events) {
        if (e.kind == stripe_event_kind::paired_prepare_end && e.stripe == stripe) {
            return e.route;
        }
    }
    return "";
}

void print_pipe(const char * name, const pipe_result & r) {
    const pipeline_counts & c = r.report.counts;
    std::printf("  %s: ok=%d stage=%s fatal=%d stripes=%llu paired=%llu fallback=%llu main_loop_ws(planned)=%llu "
                "residual_loop_ws(planned)=%llu paired_main=%llu paired_residual=%llu index=%llu/%llu readbacks=%llu "
                "fences=%llu drains=%llu runner_calls=%zu planned_cmds=%zu h2(main=%llu res=%llu idx=%llu) decodes=%llu %s\n",
                name, (int) r.report.ok, r.report.stage ? r.report.stage : "-", (int) r.report.fatal,
                (unsigned long long) c.stripes_semantic_completed, (unsigned long long) c.paired_stripes,
                (unsigned long long) c.paired_fallback_stripes, (unsigned long long) c.main_loop_ws,
                (unsigned long long) c.residual_loop_ws, (unsigned long long) c.paired_main_loop_ws,
                (unsigned long long) c.paired_residual_loop_ws, (unsigned long long) c.paired_index_configs,
                (unsigned long long) c.paired_index_writes, (unsigned long long) c.paired_readbacks,
                (unsigned long long) c.paired_fences, (unsigned long long) c.paired_drains, r.paired_calls,
                r.planned_commands.size(), (unsigned long long) r.h2_main, (unsigned long long) r.h2_residual,
                (unsigned long long) r.h2_index, (unsigned long long) r.decode_calls, r.report.detail.c_str());
}

void test_stream_pipeline() {
#if defined(POTAL_TEST_CLOCK_OBSERVER)
    const bool observing = true;
    stream_clock_reads = 0;
    stream_clock_active = &observing;
    (void) std::chrono::steady_clock::now();
    stream_clock_active = nullptr;
    check("H clock observer detects actual steady_clock calls", stream_clock_reads == 1);
#endif
    const char * previous = std::getenv("GGML_POTAL_PAIRED_STREAM");
    const bool had = previous != nullptr;
    const std::string saved = had ? previous : "";
    unsetenv("GGML_POTAL_PAIRED_STREAM");
    bool invalid = true;
    check("H unset stream option defaults off",
          options::paired_stream_requested(&invalid) == potal_paired_stream_mode::off && !invalid);
    const std::pair<const char *, potal_paired_stream_mode> valid[] = {
        {"OFF", potal_paired_stream_mode::off},
        {"BOUNDED", potal_paired_stream_mode::bounded},
        {"CONTINUOUS", potal_paired_stream_mode::continuous},
        {"CONTINUOUS_EARLY", potal_paired_stream_mode::continuous_early},
    };
    for (const auto & option : valid) {
        setenv("GGML_POTAL_PAIRED_STREAM", option.first, 1);
        check("H valid stream option", options::paired_stream_requested(&invalid) == option.second && !invalid);
    }
    for (const char * value : {"ON", "1", "0", "CONTINUOUSx", " BOUNDED", "BOUNDED ",
                              "CONTINUOUS_EARLYx", " CONTINUOUS_EARLY", "CONTINUOUS_EARLY "}) {
        setenv("GGML_POTAL_PAIRED_STREAM", value, 1);
        (void) options::paired_stream_requested(&invalid);
        check("H malformed stream flag rejected", invalid);
    }
    if (had) setenv("GGML_POTAL_PAIRED_STREAM", saved.c_str(), 1);
    else unsetenv("GGML_POTAL_PAIRED_STREAM");
    if (!options::compensation || options::activation_quant != options::ActivationQuantAlgo::EXSIA) return;
    pipe_node nd(33, 35, 160, 199, 60.0f, 2, 1);
    check("H pipeline fixture", nd.ok);
    if (!nd.ok) return;
    const auto no_commands = [](const pipe_result & r) {
        return !r.report.ok && !r.report.fatal && r.untouched && r.paired_calls == 0 &&
               !r.h2_main && !r.h2_residual && r.planned_commands.empty() &&
               !r.report.counts.paired_fallback_stripes;
    };
    for (auto mode : {potal_paired_stream_mode::bounded, potal_paired_stream_mode::continuous,
                      potal_paired_stream_mode::continuous_early}) {
        check("H stream without explicit paired refuses", no_commands(run_pipe(nd, 1, 0, {}, true, false, false, mode)));
        check("H stream without explicit Ktile refuses", no_commands(run_pipe(nd, 1, 1, {}, false, false, false, mode)));
        check("H stream with legacy prefetch refuses", no_commands(run_pipe(nd, 1, 1, {}, true, true, false, mode)));
        check("H stream with early prefetch refuses", no_commands(run_pipe(nd, 1, 1, {}, true, true, true, mode)));
        check("H stream with conservative fences refuses", no_commands(run_pipe(nd, 1, 4, {}, true, false, false, mode)));
        if (PD != 32) {
            check("H stream DIM16 refuses", no_commands(run_pipe(nd, 1, 1, {}, true, false, false, mode)));
            continue;
        }
        bool propagated = true;
        const auto baseline = run_pipe(nd, 1, 1, {}, true);
        const auto streamed = run_pipe(nd, 1, 1, [&](const potal_paired_request & q) {
            propagated = propagated && q.stream == mode && q.stream_counter == nullptr &&
                         q.ktile_scale_reuse && !q.next_k_prefetch && !q.early_k_prefetch &&
                         !q.conservative_fences;
        }, true, false, false, mode);
        check("H pipeline propagates stream and computes exact old-Ktile output without fallback",
              baseline.report.ok && streamed.report.ok && propagated && streamed.paired_calls > 0 &&
              streamed.out == baseline.out && streamed.h2_errors.empty() &&
              streamed.planned_commands.empty() && !streamed.report.counts.paired_fallback_stripes);
    }
}

void test_ktile_pipeline() {
    const char * previous = std::getenv("GGML_POTAL_PAIRED_KTILE");
    const bool had_previous = previous != nullptr;
    const std::string saved = had_previous ? previous : "";
    unsetenv("GGML_POTAL_PAIRED_KTILE");
    bool invalid = true;
    check("R unset Ktile defaults off and clears invalid", !options::paired_ktile_requested(&invalid) && !invalid);
    for (const char * value : { "ON", "on", "On", "1", "OFF", "off", "OfF", "0", "" }) {
        setenv("GGML_POTAL_PAIRED_KTILE", value, 1);
        invalid = true;
        const bool enabled = options::paired_ktile_requested(&invalid);
        const bool expected = std::strcmp(value, "1") == 0 ||
                              std::strcmp(value, "ON") == 0 || std::strcmp(value, "on") == 0 || std::strcmp(value, "On") == 0;
        check(std::string("R Ktile parse ") + value, enabled == expected && !invalid);
    }
    for (const char * value : { "true", "ONx", "2" }) {
        setenv("GGML_POTAL_PAIRED_KTILE", value, 1);
        invalid = false;
        check(std::string("R invalid Ktile parse '") + value + "'", !options::paired_ktile_requested(&invalid) && invalid);
    }
    if (had_previous) setenv("GGML_POTAL_PAIRED_KTILE", saved.c_str(), 1);
    else unsetenv("GGML_POTAL_PAIRED_KTILE");

    check("R pipeline Ktile defaults off", !pipeline_config{}.paired_ktile_scale_reuse);
    check("D2 request and pipeline prefetch default off",
          !potal_paired_request{}.next_k_prefetch && !pipeline_config{}.paired_next_k_prefetch);
    const char * previous_prefetch = std::getenv("GGML_POTAL_PAIRED_PREFETCH");
    const bool had_prefetch = previous_prefetch != nullptr;
    const std::string saved_prefetch = had_prefetch ? previous_prefetch : "";
    unsetenv("GGML_POTAL_PAIRED_PREFETCH");
    invalid = true;
    check("D2 unset prefetch clears invalid and stays off", !options::paired_prefetch_requested(&invalid) && !invalid);
    for (const char * value : {"ON", "on", "1", "OFF", "0", "", "true", "ONx"}) {
        setenv("GGML_POTAL_PAIRED_PREFETCH", value, 1);
        const bool enabled = options::paired_prefetch_requested(&invalid);
        const bool expected = std::strcmp(value, "ON") == 0 || std::strcmp(value, "on") == 0 ||
                              std::strcmp(value, "1") == 0;
        const bool bad = std::strcmp(value, "true") == 0 || std::strcmp(value, "ONx") == 0;
        check(std::string("D2 prefetch parser ") + value, enabled == expected && invalid == bad);
    }
    if (had_prefetch) setenv("GGML_POTAL_PAIRED_PREFETCH", saved_prefetch.c_str(), 1);
    else unsetenv("GGML_POTAL_PAIRED_PREFETCH");
    check("G early request, callback and pipeline default off",
          !potal_paired_request{}.early_k_prefetch && potal_paired_request{}.early_counter == nullptr &&
          !pipeline_config{}.paired_early_k_prefetch);
    const char * previous_early = std::getenv("GGML_POTAL_PAIRED_EARLY_PREFETCH");
    const bool had_early = previous_early != nullptr;
    const std::string saved_early = had_early ? previous_early : "";
    unsetenv("GGML_POTAL_PAIRED_EARLY_PREFETCH");
    invalid = true;
    check("G unset early prefetch stays off and clears invalid",
          !options::paired_early_prefetch_requested(&invalid) && !invalid);
    for (const char * value : {"ON", "on", "On", "1", "OFF", "off", "OfF", "0", "", "true", "ONx", "2", " ON"}) {
        setenv("GGML_POTAL_PAIRED_EARLY_PREFETCH", value, 1);
        invalid = true;
        const bool enabled = options::paired_early_prefetch_requested(&invalid);
        const bool expected = std::strcmp(value, "ON") == 0 || std::strcmp(value, "on") == 0 ||
                              std::strcmp(value, "On") == 0 || std::strcmp(value, "1") == 0;
        const bool bad = std::strcmp(value, "true") == 0 || std::strcmp(value, "ONx") == 0 ||
                         std::strcmp(value, "2") == 0 || std::strcmp(value, " ON") == 0;
        check(std::string("G early prefetch parser '") + value + "'", enabled == expected && invalid == bad);
    }
    if (had_early) setenv("GGML_POTAL_PAIRED_EARLY_PREFETCH", saved_early.c_str(), 1);
    else unsetenv("GGML_POTAL_PAIRED_EARLY_PREFETCH");
    if (!options::compensation || options::activation_quant != options::ActivationQuantAlgo::EXSIA) {
        std::printf("  R pipeline skipped: needs EXSIA and RMD\n");
        return;
    }
    pipe_node nd(96, 96, 256, 41, 60.0f, 4, 1);
    check("R pipeline fixture", nd.ok);
    if (!nd.ok) return;
    const auto no_commands = [](const pipe_result & r) {
        return !r.report.ok && !r.report.fatal && r.untouched && r.paired_calls == 0 &&
               r.h2_main == 0 && r.h2_residual == 0 && r.planned_commands.empty() &&
               r.report.counts.paired_fallback_stripes == 0;
    };
    const pipe_result serial_opt = run_pipe(nd, 1, 0, {}, true);
    check("R Ktile without paired refuses before commands", no_commands(serial_opt));
    check("D2 prefetch without Ktile refuses before commands",
          no_commands(run_pipe(nd, 1, 1, {}, false, true)));
    check("D2 prefetch without paired refuses before commands",
          no_commands(run_pipe(nd, 1, 0, {}, true, true)));
    check("G early without next-K prefetch refuses before commands",
          no_commands(run_pipe(nd, 1, 1, {}, true, false, true)));
    check("G early without Ktile refuses before commands",
          no_commands(run_pipe(nd, 1, 1, {}, false, true, true)));
    check("G early without paired refuses before commands",
          no_commands(run_pipe(nd, 1, 0, {}, true, true, true)));
    bool propagated = true;
    const pipe_result opt = run_pipe(nd, 1, 1, [&](const potal_paired_request & q) {
        propagated = propagated && q.ktile_scale_reuse;
    }, true);
    if (PD == 16) {
        check("R DIM16 Ktile pipeline refuses before commands", no_commands(opt));
        check("D2 DIM16 prefetch refuses without fallback", no_commands(run_pipe(nd, 1, 1, {}, true, true)));
        check("G DIM16 early refuses without fallback", no_commands(run_pipe(nd, 1, 1, {}, true, true, true)));
        return;
    }
    bool default_off = true;
    const pipe_result normal = run_pipe(nd, 1, 1, [&](const potal_paired_request & q) {
        default_off = default_off && !q.ktile_scale_reuse && !q.early_k_prefetch && q.early_counter == nullptr;
    });
    check("R pipeline propagates opt only, preserves output and never falls back",
          propagated && default_off && normal.report.ok && opt.report.ok &&
          opt.paired_calls == nd.stripes() && opt.h2_errors.empty() && opt.out == normal.out &&
          opt.report.counts.paired_fallback_stripes == 0 && opt.planned_commands.empty());
    bool prefetch_propagated = true;
    const pipe_result prefetched = run_pipe(nd, 1, 1, [&](const potal_paired_request & q) {
        prefetch_propagated = prefetch_propagated && q.ktile_scale_reuse && q.next_k_prefetch;
    }, true, true);
    check("D2 pipeline propagates explicit prefetch and preserves exact output without fallback",
          prefetch_propagated && prefetched.report.ok && prefetched.h2_errors.empty() &&
          prefetched.paired_calls == nd.stripes() && prefetched.out == opt.out &&
          prefetched.report.counts.paired_fallback_stripes == 0 && prefetched.planned_commands.empty());
    pipe_node early_node(65, 97, 2080, 129, 60.0f, 4, 1);
    check("G multi-K pipeline fixture", early_node.ok);
    if (!early_node.ok) return;
    const pipe_result early_baseline = run_pipe(early_node, 1, 1, {}, true);
    bool early_propagated = true;
    const pipe_result early = run_pipe(early_node, 1, 1, [&](const potal_paired_request & q) {
        early_propagated = early_propagated && q.ktile_scale_reuse && q.next_k_prefetch &&
                           q.early_k_prefetch && q.early_counter == nullptr;
    }, true, true, true);
    check("G pipeline propagates early only, preserves exact F32, never falls back",
          early_propagated && early_baseline.report.ok && early.report.ok && early.h2_errors.empty() &&
          early.paired_calls == early_node.stripes() && same_bits(early.out, early_baseline.out) &&
          early.report.counts.paired_fallback_stripes == 0 && early.planned_commands.empty());
    check("G pipeline aggregates CPU admissions, zero raw MVINs and only extra checked drains",
          early.report.counts.paired_prefetch_loads > 0 &&
          early.report.counts.paired_prefetch_ldb_only == early.report.counts.paired_prefetch_loads &&
          early.report.counts.paired_prefetch_mvins == 0 &&
          early.report.counts.paired_fences == early_baseline.report.counts.paired_fences +
              early.report.counts.paired_prefetch_ldb_only &&
          early.report.counts.paired_drains == early_baseline.report.counts.paired_drains +
              early.report.counts.paired_prefetch_ldb_only &&
          early.report.counts.paired_main_prefetch_issue_s >= 0 &&
          early.report.counts.paired_main_prefetch_issue_s <= early.report.counts.issue_s &&
          early.report.counts.paired_residual_prefetch_issue_s == 0);
    pipe_node dense(96, 96, 256, 43, 3000.0f, 1, 6);
    check("R Ktile capacity fixture", dense.ok);
    if (dense.ok) {
        const pipe_result rejected = run_pipe(dense, 1, 1, {}, true);
        check("R explicit Ktile capacity refusal cannot silently run Serial", no_commands(rejected));
    }
}

void test_pipeline() {
    if (!options::compensation || options::activation_quant != options::ActivationQuantAlgo::EXSIA) {
        std::printf("  S skipped: needs GGML_GEMMINI_ACTIVATION_QUANT=EXSIA and GGML_GEMMINI_ENABLE_RMD=ON\n");
        return;
    }
    // Sparse outliers: every stripe fits the paired resources.
    pipe_node nd(96, 96, 256, 41, 60.0f, 4, 1);
    check("S fixture", nd.ok);
    if (!nd.ok) {
        return;
    }
    for (size_t capacity : { (size_t) 1, (size_t) 2 }) {
        const pipe_result serial = run_pipe(nd, capacity, 0);
        const pipe_result paired = run_pipe(nd, capacity, 1);
        const std::string cap    = " (capacity " + std::to_string(capacity) + ")";
        print_pipe(("S serial" + cap).c_str(), serial);
        print_pipe(("S paired" + cap).c_str(), paired);
        const size_t stripes = nd.stripes();
        check("S serial and paired runs commit every stripe" + cap,
              serial.report.ok && paired.report.ok && !serial.untouched && !paired.untouched &&
                  paired.report.counts.stripes_semantic_completed == stripes);
        check("S paired F32 output == serial F32 output, bitwise" + cap,
              serial.out.size() == paired.out.size() &&
                  std::memcmp(serial.out.data(), paired.out.data(), serial.out.size() * sizeof(float)) == 0);
        const pipeline_counts & c = paired.report.counts;
        check("S every stripe paired, none serial: one paired runner call per stripe, no planned GEMM, no serial LOOP_WS" + cap,
              c.paired_stripes == stripes && c.paired_fallback_stripes == 0 && paired.paired_calls == stripes &&
                  paired.planned_commands.empty() && c.main_loop_ws == 0 && c.residual_loop_ws == 0 &&
                  c.main_logical_gemms == 0 && c.residual_logical_gemms == 0);
        check("S paired commands reached the H2 emulator without contract errors, counts agree" + cap,
              paired.h2_errors.empty() && paired.h2_main == c.paired_main_loop_ws &&
                  paired.h2_residual == c.paired_residual_loop_ws && paired.h2_index == c.paired_index_configs &&
                  c.paired_readbacks > 0 && c.paired_fences >= stripes);
        check("S residual packets were merged once per packet stripe" + cap,
              c.merges_completed == c.residual_nonempty_packets && c.residual_nonempty_packets > 0 &&
                  c.radix_reconstructions == c.residual_nonempty_packets);
        check("S Serial and successful Paired each decode exactly once per nonempty packet" + cap,
              serial.report.ok && paired.report.ok && serial.report.counts.residual_nonempty_packets > 0 &&
                  serial.report.counts.residual_nonempty_packets == c.residual_nonempty_packets &&
                  serial.decode_calls == serial.report.counts.residual_nonempty_packets &&
                  paired.decode_calls == c.residual_nonempty_packets);
        // Paired event order per stripe; no serial GEMM events for a paired stripe.
        bool order = true, no_serial_events = true;
        for (size_t st = 0; st < stripes; ++st) {
            const long mb = seq_of(paired, stripe_event_kind::main_begin, st);
            const long pb = seq_of(paired, stripe_event_kind::paired_prepare_begin, st);
            const long pe = seq_of(paired, stripe_event_kind::paired_prepare_end, st);
            const long xb = seq_of(paired, stripe_event_kind::paired_begin, st);
            const long xe = seq_of(paired, stripe_event_kind::paired_end, st);
            const long sc = seq_of(paired, stripe_event_kind::semantic_complete, st);
            const long cr = seq_of(paired, stripe_event_kind::capacity_release, st);
            const long rb = seq_of(paired, stripe_event_kind::radix_reconstruct_begin, st);
            const long me = seq_of(paired, stripe_event_kind::merge_end, st);
            order = order && mb >= 0 && mb < pb && pb < pe && pe < xb && xb < xe && xe < sc && sc < cr &&
                    route_of(paired, st) == "paired" && (rb < 0 || (xe < rb && rb < me && me < sc));
            no_serial_events = no_serial_events && seq_of(paired, stripe_event_kind::main_end, st) < 0 &&
                               seq_of(paired, stripe_event_kind::residual_begin, st) < 0 &&
                               seq_of(paired, stripe_event_kind::residual_end, st) < 0 &&
                               seq_of(paired, stripe_event_kind::residual_prepare_begin, st) < 0;
            if (st + capacity < stripes) {
                order = order && cr < seq_of(paired, stripe_event_kind::submit_accepted, st + capacity);
            }
        }
        check("S paired stripes: main_begin < paired_prepare < paired_begin < paired_end < radix / merge < "
              "semantic_complete < capacity_release; capacity_release(s) < submit_accepted(s + capacity)" + cap,
              order);
        check("S paired stripes emit no main_end / residual_* events (the stream is not sequential)" + cap, no_serial_events);
    }

    {
        const pipe_result serial = run_pipe(nd, 2, 0);
        const pipe_result cons   = run_pipe(nd, 2, 4);
        print_pipe("S paired conservative", cons);
        check("S conservative fences reach the executor and keep the output bitwise equal to serial",
              cons.report.ok && cons.conservative_req && cons.report.counts.paired_fences > cons.report.counts.paired_stripes &&
                  std::memcmp(serial.out.data(), cons.out.data(), serial.out.size() * sizeof(float)) == 0);
    }

    // Dense large outliers on every row: two radix lanes per row exceed ACC1 -> serial_fallback.
    pipe_node dense(96, 96, 256, 43, 3000.0f, 1, 6);
    check("S dense fixture", dense.ok);
    if (dense.ok) {
        const pipe_result serial = run_pipe(dense, 2, 0);
        const pipe_result paired = run_pipe(dense, 2, 1);
        print_pipe("S dense serial", serial);
        print_pipe("S dense paired", paired);
        const pipeline_counts & c = paired.report.counts;
        bool fallback_routes = c.paired_fallback_stripes > 0;
        for (size_t st = 0; st < dense.stripes(); ++st) {
            const std::string route = route_of(paired, st);
            if (route == "serial_fallback") {
                fallback_routes = fallback_routes && seq_of(paired, stripe_event_kind::paired_begin, st) < 0 &&
                                  seq_of(paired, stripe_event_kind::main_end, st) > seq_of(paired, stripe_event_kind::paired_prepare_end, st);
            }
        }
        check("S capacity refusal (lane rows beyond ACC1): those stripes run serially, reported serial_fallback, "
              "never through the paired runner",
              paired.report.ok && fallback_routes && c.paired_stripes + c.paired_fallback_stripes == dense.stripes() &&
                  paired.paired_calls == c.paired_stripes && c.main_logical_gemms == c.paired_fallback_stripes);
        check("S dense run with fallback stripes: F32 output == serial, bitwise",
              serial.report.ok && std::memcmp(serial.out.data(), paired.out.data(), serial.out.size() * sizeof(float)) == 0);
        check("S dense Serial decodes once per packet; capacity-rejected Paired stripes reuse that decode",
              serial.report.ok && paired.report.ok && c.paired_fallback_stripes > 0 &&
                  c.residual_nonempty_packets == dense.stripes() &&
                  serial.report.counts.residual_nonempty_packets == c.residual_nonempty_packets &&
                  serial.decode_calls == serial.report.counts.residual_nonempty_packets &&
                  paired.decode_calls == c.residual_nonempty_packets);
    }

    // Failures: no paired runner; a paired runner reporting fatal.
    {
        const pipe_result r = run_pipe(nd, 2, 2);
        print_pipe("S no paired runner", r);
        check("S paired configured without a paired runner: the run fails at stage paired, not fatal, dst untouched",
              !r.report.ok && r.report.stage != nullptr && std::string(r.report.stage) == "paired" && !r.report.fatal &&
                  r.untouched && r.planned_commands.empty());
    }
    {
        const pipe_result r = run_pipe(nd, 2, 3);
        print_pipe("S fatal paired report", r);
        check("S a fatal paired report fails the run fatal, dst untouched, the consumer reports fatal",
              !r.report.ok && r.report.fatal && r.consumer.fatal && r.untouched && r.paired_calls == 1);
    }
}

void test_index16_pipeline() {
    if (!options::compensation || options::activation_quant != options::ActivationQuantAlgo::EXSIA) return;
    pipe_node nd(16, 32, 64, 121, 60.0f, 1, 4);
    check("I16 pipeline fixture", nd.ok);
    if (!nd.ok) return;
    if (PD == 32) {
        // This environment option deliberately remains DIM16-only in stripe.cpp.
        // DIM32 still runs the default physical64 pipeline; requesting Index16
        // must fail before a command rather than silently select another table.
        pipe_result normal, serial;
        capture_index_diag(nullptr, nullptr, [&] {
            normal = run_pipe(nd, 1, 4);
            serial = run_pipe(nd, 1, 0);
        });
        check("I16 DIM32 control: physical64 paired pipeline runs and equals Serial",
              normal.report.ok && serial.report.ok && normal.paired_calls == 1 &&
                  normal.h2_errors.empty() && normal.out == serial.out && normal.h2_residual > 0);
        for (const char * entries : { "16", "32", "16x", "" }) {
            pipe_result rejected;
            capture_index_diag(entries, "1", [&] { rejected = run_pipe(nd, 1, 4); });
            check(std::string("I16 DIM32 rejects unsupported diagnostic entries before commands: ") + entries,
                  !rejected.report.ok && rejected.report.stage != nullptr &&
                      std::string(rejected.report.stage) == "paired_plan" && !rejected.report.fatal &&
                      rejected.untouched && rejected.paired_calls == 0 && rejected.h2_main == 0 &&
                      rejected.h2_residual == 0 && rejected.planned_commands.empty());
        }
        return;
    }
    pipe_result normal, diagnostic;
    potal_paired_tile_plan normal_tile;
    bool normal_geometry = true, same_geometry = true;
    capture_index_diag(nullptr, nullptr, [&] {
        normal = run_pipe(nd, 1, 4, [&](const potal_paired_request & req) {
            normal_geometry = normal_geometry && req.hw == nullptr && req.main_plan != nullptr && req.plan != nullptr &&
                req.main_plan->tile_I == nd.plan.tile_I && req.main_plan->tile_J == nd.plan.tile_J &&
                req.main_plan->tile_K == nd.plan.tile_K;
            if (req.plan != nullptr) normal_tile = req.plan->tile;
        });
    });
    const std::string log = capture_index_diag("16", "1", [&] {
        diagnostic = run_pipe(nd, 1, 4, [&](const potal_paired_request & req) {
            same_geometry = same_geometry && req.hw != nullptr && req.hw->index_entries == 16 &&
                            req.plan != nullptr && req.residual != nullptr;
            if (!same_geometry) return;
            potal_paired_geometry expected = potal_paired_hw_geometry();
            expected.index_entries = 16;
            const auto & hw = *req.hw;
            same_geometry = same_geometry && hw.dim == expected.dim && hw.bank_num == expected.bank_num &&
                hw.bank_rows == expected.bank_rows && hw.acc_rows == expected.acc_rows &&
                hw.scale_entries == expected.scale_entries && hw.work_ids == expected.work_ids &&
                hw.block_k == expected.block_k && req.plan->tile.fragments_per_window == 1 &&
                req.plan->tile.tile_I == normal_tile.tile_I && req.plan->tile.tile_J == normal_tile.tile_J &&
                req.plan->tile.tile_K == normal_tile.tile_K && req.main_plan->tile_K == normal_tile.tile_K;
            std::string why;
            potal_paired_staging staging;
            same_geometry = same_geometry &&
                potal_validate_execution_plan(*req.plan, req.residual, &why, hw) == potal_paired_status::ok &&
                potal_paired_pack_residual(*req.plan, *req.residual, staging, &why, hw) == potal_paired_status::ok;
        });
    });
    check("I16 pipeline uses the same local geometry for planning, request, validation and packing", normal_geometry &&
          same_geometry && normal.report.ok && diagnostic.report.ok && diagnostic.paired_calls == 1 &&
          diagnostic.h2_errors.empty() && diagnostic.out == normal.out &&
          normal_tile.fragments_per_window == 2 && diagnostic.h2_residual > normal.h2_residual &&
          log.find("[POTAL_WINDOW_DIAG]") != std::string::npos &&
          log.find("[POTAL_WINDOW_MISMATCH]") == std::string::npos);

    pipe_node bound(16, 16, 4096, 122, 3000.0f, 1, 6);
    bool bound_geometry = true, binding_fixture = false;
    potal_paired_tile_plan bound_normal;
    pipe_result bound_default, bound_single;
    capture_index_diag(nullptr, nullptr, [&] {
        bound_default = run_pipe(bound, 1, 4, [&](const potal_paired_request & req) {
            if (req.plan != nullptr) bound_normal = req.plan->tile;
        });
    });
    capture_index_diag("16", "1", [&] {
        bound_single = run_pipe(bound, 1, 4, [&](const potal_paired_request & req) {
            bound_geometry = bound_geometry && req.hw != nullptr && req.plan != nullptr && req.residual != nullptr;
            if (!bound_geometry) return;
            potal_paired_tile_plan unfrozen;
            potal_paired_execution_plan unfrozen_plan;
            std::string why;
            binding_fixture = potal_paired_plan_stripe(bound.plan, req.rows, req.n, req.k, req.residual,
                unfrozen, unfrozen_plan, &why, *req.hw) == potal_paired_status::ok &&
                unfrozen.tile_K > bound_normal.tile_K;
            bound_geometry = bound_geometry && req.plan->tile.tile_I == bound_normal.tile_I &&
                req.plan->tile.tile_J == bound_normal.tile_J && req.plan->tile.tile_K == bound_normal.tile_K &&
                req.main_plan->tile_K == bound_normal.tile_K;
        });
    });
    check("I16 preserves common Main I/J/K when smaller Residual reservation would otherwise increase K", bound.ok &&
          bound_default.report.ok && bound_single.report.ok && binding_fixture && bound_geometry &&
          bound_default.out == bound_single.out && bound_single.h2_errors.empty());

    const std::pair<const char *, const char *> invalid[] = {
        { "32", "1" }, { "16x", "1" }, { "", "1" }, { "16", nullptr }, { "16", "true" },
    };
    for (const auto & setting : invalid) {
        pipe_result rejected;
        capture_index_diag(setting.first, setting.second, [&] { rejected = run_pipe(nd, 1, 4); });
        check(std::string("I16 unsupported setting rejected before hardware: entries='") + setting.first +
              "' window='" + (setting.second ? setting.second : "unset") + "'",
              !rejected.report.ok && rejected.report.stage != nullptr &&
              std::string(rejected.report.stage) == "paired_plan" && !rejected.report.fatal &&
              rejected.untouched && rejected.paired_calls == 0 && rejected.h2_main == 0 &&
              rejected.h2_residual == 0 && rejected.planned_commands.empty());
    }
}

} // namespace

int main() {
    test_translation();
    test_staging();
    test_numerics();
    test_early_emulator();
    test_early_execution_failures();
    test_early_saturation();
    test_early_request_refusals();
    test_stream_generation_wrap();
    test_stream_request_refusals();
    test_stream_capacity();
    test_stream_emulator_lifetimes();
    test_stream_execution_failures();
    test_pairs_and_generations();
    test_ktile_scale_reuse();
    test_failures();
    test_review_followups();
    test_conservative();
    test_window_diag();
    test_index16_plan();
    test_pipeline();
    test_ktile_pipeline();
    test_stream_pipeline();
    test_index16_pipeline();
    test_close_keeps_owner_with_live_entries(); // last: leaves the accelerator owned
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
