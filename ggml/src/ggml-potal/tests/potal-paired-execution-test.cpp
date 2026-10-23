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
#include <cmath>
#include <map>
#include <mutex>

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
};

// One stripe (stripe 0, rows = min(stripe_rows, M)) of a fixture through the new executor.
run_result run_paired(const paired_fixture & f, const potal_tile_plan & main, const paired_inputs & in, h2_emulator & emu,
                      bool with_sink = true, uint32_t generation = 0, bool conservative = false) {
    run_result   r;
    const size_t rows = std::min(main.tile_I * PD, f.M);
    residual::decoded_packet d;
    const bool   has_res = in.packets[0] != nullptr &&
                         residual::decode_packet(*in.packets[0], f.weight, d) == grmd::RmdStatus::success;
    const size_t ldc = align_up(f.N, PD);
    r.c_main.reset(static_cast<int32_t *>(dma_alloc(align_up(rows, PD) * ldc * sizeof(int32_t))));
    r.c_res.reset(static_cast<int32_t *>(dma_alloc(align_up(std::max<size_t>(d.rows.size(), 1), PD) * ldc * sizeof(int32_t))));
    potal_paired_session * s    = nullptr;
    potal_paired_sink      sink = emu.sink();
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
    r.status       = potal_paired_matmul_auto(q, s, r.report);
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
        emu.fail_funct = 0xFFFFFFFFu;
        (void) potal_paired_fence(s);
    }
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
              bool conservative = false) {
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
    for (size_t k = 0; k < PD + PD / 4 && k < 32; ++k) {
        entries.push_back({ 0, k, (int32_t) (k + 1) });
    }
    for (size_t k = 32; k < 35; ++k) {
        entries.push_back({ 3, k, (int32_t) -5 });
    }
    for (size_t k = 64; k < 96; ++k) {
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
        { "N single panel, one window", 48, 48, 160, 2, dense_entries(7, 40, 100) },
        { "N several panels (K 2048), several J groups, residual spanning panels", 80, 160, 2048, 3, dense_entries(9, 61, 150) },
        { "N many windows per panel (dense residual)", 16, 96, 768, 4, dense_entries(1, 1, 60) },
        { "N tails: rows 70, N 140, K 2048", 70, 140, 2048, 6, dense_entries(11, 97, 120) },
        { "N large residual values (two lanes, saturation range)", 32, 64, 256, 7, dense_entries(2, 3, 30000) },
        { "N decode shape M = 1", 1, 768, 768, 8, dense_entries(1, 5, 90) },
    };
    for (const case_spec & c : cases) {
        std::string why;
        run_result  r;
        const bool  ok = run_case(c, true, why, &r);
        check(std::string(c.name) + ": Main INT32, Residual INT32, radix INT64, final F32 == serialized reference (" + why + ")", ok);
        // Geometry the case name claims, hard-coded from the Main tiler at DIM16 (not taken from the plan under test).
        const std::string name = c.name;
        if (PD == 16 && name.rfind("N several panels", 0) == 0) {
            check("N several panels: 3 K panels x 2 J groups = exactly 6 Main LOOP_WS, more than one window",
                  ok && r.report.tile.k_panels == 3 && r.report.tile.j_groups == 2 && r.report.counts.main_loop_ws == 6 &&
                      r.report.tile.residual_windows > 1);
        }
        if (PD == 16 && name.rfind("N many windows", 0) == 0) {
            size_t most = 0;
            for (const potal_paired_k_panel & p : r.report.tile.panels) {
                most = std::max(most, p.windows.size());
            }
            check("N many windows: one panel holds several Index Windows", ok && most > 2);
        }
        if (PD == 16 && name.rfind("N tails", 0) == 0) {
            check("N tails: J groups of 5 and 4 tiles (pad_j 4), I tail pad 10",
                  ok && r.report.tile.j_groups == 2 && r.report.tile.pad_J == 4 && r.report.tile.pad_I == 10);
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
    const bool first_ok = ok && r.status == potal_paired_status::ok && emu.errors.empty() && compare_stripe(f, in, ref, r, 80).empty();
    check("G three panels x two J groups across both pairs, generations wrapping at 255: executor == reference", first_ok);
    bool pairs = first_ok;
    size_t loops = 0;
    for (const potal_paired_cmd_record & rec : emu.records) {
        if (rec.funct == POTAL_FUNCT_LOOP_WS && ((rec.rs1 >> 62) & 1) == 0) {
            const size_t residency = loops++;
            pairs = pairs && ((rec.rs1 >> 16) & 3) == (residency % 2) + 1 && ((rec.rs1 >> 18) & 3) == (residency % 2) + 1;
        }
    }
    check("G Main LOOP_WS r selects pair r % 2 (selectors pair + 1); " + std::to_string(loops) + " residencies", pairs && loops == 6);
    // A second stripe on the same emulator state (B of both pairs still resident): the generations continue.
    const uint32_t gen_after = r.generation;
    run_result     r2        = run_paired(f, main, in, emu, true, gen_after);
    check("G a second call after the first (counter continues, pairs reused after drains): executor == reference",
          r2.status == potal_paired_status::ok && emu.errors.empty() && compare_stripe(f, in, ref, r2, 80).empty() &&
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
                      const residual::decoded_packet * d, h2_emulator & emu, const potal_paired_execution_plan * plan = nullptr) {
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
    q.generation = &gen; q.plan = plan;
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
    residual::run r0, r2;
    r0.block_id = 0;
    for (uint16_t k = 0; k < 20; ++k) r0.local_k.push_back(k);
    r2.block_id = 2;
    r2.local_k  = { 1, 5, 9 };
    d.runs      = { r0, r2 };
    d.k_compact = 23;
    d.digits.assign(d.rows.size() * d.k_compact, 0);
    for (size_t i = 0; i < d.rows.size(); ++i) {
        for (size_t c = 0; c < d.k_compact; ++c) {
            d.digits[i * d.k_compact + c] = (c >= 16 && c < 20) ? 0 : (int8_t) ((int) (i * 37 + c * 11) % 255 - 127);
        }
    }
    h2_emulator      emu;
    const direct_run run = run_direct(f, main, 16, &d, emu);
    const direct_golden g = golden_direct(f, 16, d);
    check("X zero fragment: skipped (1 of 3), only its 4 K missing from the Index writes, Main and Residual == golden",
          run.status == potal_paired_status::ok && emu.errors.empty() && run.report.tile.residual_zero_fragments == 1 &&
              run.report.counts.index_writes == run.report.tile.j_groups * (16 + 3) && run.main_c == g.main_c &&
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
    small.tile_I = 1; // 16 rows per group
    paired_fixture  f32(32, 64, 128, 78, [](size_t blk, size_t n) { return (int16_t) ((blk + n) % 2); });
    h2_emulator     emu4;
    const direct_run multi = f32.ok ? run_direct(f32, small, 32, nullptr, emu4) : direct_run{};
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
    std::string               planned_commands; // emulator record of the serial (planned) GEMMs
    uint64_t                  h2_main = 0, h2_residual = 0, h2_index = 0;
    std::string               h2_errors;
    bool                      conservative_req = false; // a request carried conservative_fences
};

const uint32_t sentinel_bits = 0x7fc0beefu;

// mode: 0 serial, 1 paired, 2 paired configured without a paired runner, 3 paired runner reporting fatal,
// 4 paired with conservative fences.
pipe_result run_pipe(pipe_node & nd, size_t capacity, int mode) {
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
            fences_seen            = fences_seen || q.conservative_fences;
            const potal_paired_sink sink = h2.sink();
            return potal_paired_run_session(q, &sink, rep);
        };
    }
    potal_test::thread_owner owner([&es](const potal_stripe_gemm_desc & d) { return es.run(d); }, runner);
    r.report = run_stripe_pipeline(nd.x, args, nd.weight, nd.plan, cfg, owner);
    owner.join();
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
                "fences=%llu drains=%llu runner_calls=%zu planned_cmds=%zu h2(main=%llu res=%llu idx=%llu) %s\n",
                name, (int) r.report.ok, r.report.stage ? r.report.stage : "-", (int) r.report.fatal,
                (unsigned long long) c.stripes_semantic_completed, (unsigned long long) c.paired_stripes,
                (unsigned long long) c.paired_fallback_stripes, (unsigned long long) c.main_loop_ws,
                (unsigned long long) c.residual_loop_ws, (unsigned long long) c.paired_main_loop_ws,
                (unsigned long long) c.paired_residual_loop_ws, (unsigned long long) c.paired_index_configs,
                (unsigned long long) c.paired_index_writes, (unsigned long long) c.paired_readbacks,
                (unsigned long long) c.paired_fences, (unsigned long long) c.paired_drains, r.paired_calls,
                r.planned_commands.size(), (unsigned long long) r.h2_main, (unsigned long long) r.h2_residual,
                (unsigned long long) r.h2_index, r.report.detail.c_str());
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

} // namespace

int main() {
    test_translation();
    test_staging();
    test_numerics();
    test_pairs_and_generations();
    test_failures();
    test_review_followups();
    test_conservative();
    test_pipeline();
    test_close_keeps_owner_with_live_entries(); // last: leaves the accelerator owned
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
