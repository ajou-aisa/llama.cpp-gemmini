// Hardware-free tests of the potal_paired_matmul_auto planner (matmul/paired-auto.hpp, H3-0):
//
//   G  geometry: the generated header, the two physical pairs, address units
//   M  Main-derived I/J: potal_tile_select's tile_I (tail-limited) and tile_J
//      are kept, the final K is propagated into the Main execution plan
//   K  final K = min(Main K, residual limit, hardware limit), every limit
//      recomputed here from the geometry, the binding limit named per case
//   O  occupancy: activation bank (Main + Residual A), weight bank, ACC0/ACC1,
//      Index Window rows, scale slots
//   A  Residual A begins at tile_I * tile_K tiles behind Main A, never overlaps it
//   P  K panels: cover K once, a block never straddles two panels, original K
//      -> resident K, panels without residual carry no window
//   W  Index Windows: capacity, non-DIM-aligned tails (pad_k), filler slots,
//      no compact K lost when the table is smaller than the selection
//   R  real RMD packets through residual::decode_packet: the fragments equal
//      the paired descriptor's (paired.hpp) and read the same original K
//   E  refusals with diagnostics: inconsistent residual, Main plan, Main ACC,
//      Residual ACC (I/J kept), weight / activation / scale capacity, Index
//      Table below DIM; the same request always gives the same plan
//
// H3-1, potal_make_execution_plan / potal_validate_execution_plan:
//   X1  single K panel without residual: the exact op sequence
//   X2  single panel, one Index Window: Main then Residual on the same resident B
//   X3  several K panels with residual: per-panel order, ACC0 / ACC1 first and
//       last only at the true first / last contribution
//   X4  residual in some panels only: no Index transaction for an empty panel
//   X5  several windows in one panel, same-buffer updates after the previous
//       Indexed gemm (hardware-stalled), Residual A reload after its completion
//   X6  non-DIM-aligned tails: entries, pad_k, filler slots
//   X7  pairs alternate; a pair is loaded again only after its retire
//   X8  retire waits for the completion of Main and every Residual consumer
//   X9  cross-buffer Index programming does not wait for the other buffer
//   X12 all-zero fragments (digits known) are left out of windows and entries
//   X13 scale windows: generations in program order, Main at slot 0, Residual
//       behind it, release only after the consumer completed, no live overlap
//   X14 I / J / K tails: pad_i, pad_j, last J group, last panel
//   X15 determinism, X16 mutated plans refused, X17 every logical contribution once
//   XR  real RMD packets: entries follow the paired descriptor's fragments and
//       original K, rows keep their (lane, source row) provenance

#include "support/paired-support.hpp"

#include "matmul/paired-auto.hpp"

#include <algorithm>
#include <functional>
#include <utility>

using namespace ggml::potal;
using namespace potal_test;

namespace {

constexpr uint32_t PPB = 32 / PD;

using run_spec = std::pair<uint32_t, std::vector<uint16_t>>; // block id, block-local K

// A decoded packet without digits: `lane_rows` rows over `stripe_rows` source rows (lane-major).
residual::decoded_packet synth(size_t lane_rows, size_t stripe_rows, const std::vector<run_spec> & runs) {
    residual::decoded_packet d;
    for (size_t i = 0; i < lane_rows; ++i) {
        d.rows.push_back({ (uint8_t) (i / stripe_rows), (uint32_t) (i % stripe_rows) });
    }
    for (const run_spec & r : runs) {
        residual::run run;
        run.block_id = r.first;
        run.local_k  = r.second;
        d.k_compact += run.local_k.size();
        d.runs.push_back(std::move(run));
    }
    return d;
}

std::vector<uint16_t> first_k(size_t count) {
    std::vector<uint16_t> k;
    for (size_t i = 0; i < count; ++i) {
        k.push_back((uint16_t) i);
    }
    return k;
}

// Every block [0, blocks) with `per_block` selected K.
std::vector<run_spec> dense_runs(size_t blocks, size_t per_block) {
    std::vector<run_spec> r;
    for (size_t b = 0; b < blocks; ++b) {
        r.emplace_back((uint32_t) b, first_k(per_block));
    }
    return r;
}

struct expected_limits {
    size_t weight, act_main, act_res, scale_main, scale_res;
};

// The limits of paired-auto.hpp, computed here from the geometry alone.
expected_limits limits_of(const potal_paired_geometry & hw, size_t tile_I, size_t tile_J, size_t res_row_tiles,
                          size_t fragments_per_window) {
    const size_t D     = hw.dim;
    const auto   floor = [](size_t v) { return v / PPB * PPB; };
    expected_limits e;
    e.weight            = floor(hw.bank_rows / (tile_J * D));
    e.act_main          = floor(hw.bank_rows / (tile_I * D));
    const size_t res_a  = res_row_tiles * fragments_per_window * D;
    e.act_res           = floor((hw.bank_rows > res_a ? hw.bank_rows - res_a : 0) / (tile_I * D));
    e.scale_main        = (hw.scale_entries / tile_J) * PPB;
    const size_t res_sl = fragments_per_window * tile_J;
    e.scale_res         = res_sl > hw.scale_entries ? 0 : ((hw.scale_entries - res_sl) / tile_J) * PPB;
    return e;
}

bool same_plan(const potal_paired_tile_plan & a, const potal_paired_tile_plan & b) {
    if (a.tile_I != b.tile_I || a.tile_J != b.tile_J || a.tile_K != b.tile_K || a.k_panels != b.k_panels ||
        a.residual_windows != b.residual_windows || a.fragments.size() != b.fragments.size() ||
        a.panels.size() != b.panels.size() || a.residual_a_rows != b.residual_a_rows ||
        a.residual_filler_slots != b.residual_filler_slots) {
        return false;
    }
    for (size_t p = 0; p < a.panels.size(); ++p) {
        const potal_paired_k_panel & x = a.panels[p], & y = b.panels[p];
        if (x.k_begin != y.k_begin || x.k_rows != y.k_rows || x.fragment_begin != y.fragment_begin ||
            x.fragment_end != y.fragment_end || x.windows.size() != y.windows.size()) {
            return false;
        }
        for (size_t w = 0; w < x.windows.size(); ++w) {
            if (x.windows[w].active_rows != y.windows[w].active_rows || x.windows[w].pad_k != y.windows[w].pad_k ||
                x.windows[w].filler_slots != y.windows[w].filler_slots) {
                return false;
            }
        }
    }
    return true;
}

void print_plan(const char * name, const potal_paired_tile_plan & p) {
    std::printf("  %s: rows=%zu N=%zu K=%zu tile_I=%zu tile_J=%zu tile_K=%zu (Main %zu, limits weight %zu "
                "activation %zu scale %zu, hardware %zu residual %zu) panels=%zu j_groups=%zu | Main A %zu + Residual A %zu "
                "rows (offset %zu tiles), W %zu rows, ACC0 %zu ACC1 %zu, slots %u+%u | residual rows=%zu tiles=%zu "
                "fragments=%zu windows=%zu filler=%zu empty_panels=%zu\n",
                name, p.rows, p.n, p.k, p.tile_I, p.tile_J, p.tile_K, p.tile_K_main, p.tile_K_weight_limit,
                p.tile_K_activation_limit, p.tile_K_scale_limit, p.tile_K_hardware_limit, p.tile_K_residual_limit,
                p.k_panels, p.j_groups, p.main_a_rows, p.residual_a_rows, p.residual_a_offset_tiles, p.weight_rows,
                p.main_acc_rows, p.residual_acc_rows, p.scale_slots_main, p.scale_slots_residual,
                p.residual_lane_rows, p.residual_row_tiles, p.fragments.size(), p.residual_windows,
                p.residual_filler_slots, p.residual_empty_panels);
}

bool main_select(size_t m, size_t n, size_t k, potal_tile_plan & main) {
    return potal_tile_select(m, n, k, PPB, &main) == POTAL_OK;
}

// Occupancy invariants of an accepted plan against the geometry it was made for.
bool occupancy_holds(const potal_paired_tile_plan & p, const potal_paired_geometry & hw) {
    const size_t D  = hw.dim;
    bool         ok = p.tile_K > 0 && p.tile_K % PPB == 0 && p.tile_K <= p.tile_K_main &&
              p.main_a_rows == p.tile_I * p.tile_K * D && p.weight_rows == p.tile_K * p.tile_J * D &&
              p.main_a_rows + p.residual_a_rows <= hw.bank_rows && p.weight_rows <= hw.bank_rows &&
              p.main_acc_rows == p.tile_I * p.tile_J * D && p.main_acc_rows <= hw.acc_rows / 2 &&
              p.residual_acc_rows <= hw.acc_rows / 2 && p.tile_I * p.tile_J <= hw.work_ids &&
              p.residual_row_tiles * p.tile_J <= hw.work_ids &&
              (size_t) p.scale_slots_main + p.scale_slots_residual <= hw.scale_entries &&
              p.scale_slots_main == (p.tile_K / PPB) * p.tile_J && p.main.tile_K == p.tile_K &&
              p.main.tile_I == p.tile_I && p.main.tile_J == p.tile_J && p.k_panels == p.main.K0 &&
              p.panels.size() == p.k_panels;
    for (const potal_paired_k_panel & panel : p.panels) {
        ok = ok && panel.main_a_rows <= p.main_a_rows && panel.weight_rows <= p.weight_rows &&
             panel.residual_a_rows <= p.residual_a_rows;
        for (const potal_paired_index_window & w : panel.windows) {
            ok = ok && w.k_tiles >= 1 && w.k_tiles <= p.fragments_per_window && w.active_rows <= hw.index_entries &&
                 w.active_rows <= w.k_tiles * D && w.pad_k < D && w.pad_k == w.k_tiles * D - w.active_rows &&
                 w.a_rows == p.residual_row_tiles * w.k_tiles * D && w.a_rows <= p.residual_a_rows;
        }
    }
    return ok;
}

void test_geometry() {
    const potal_paired_geometry hw = potal_paired_hw_geometry();
    check("G geometry is the generated header: DIM, banks, ACC, Index and scale entries, work IDs, K32 blocks",
          hw.dim == POTAL_HW_DIM && hw.bank_num == POTAL_HW_BANK_NUM && hw.bank_rows == POTAL_HW_BANK_ROWS &&
              hw.acc_rows == POTAL_HW_ACC_ROWS && hw.index_entries == POTAL_HW_INDEX_ENTRIES &&
              hw.scale_entries == POTAL_HW_SCALE_ENTRIES && hw.work_ids == POTAL_WORK_IDS_PER_LOOP &&
              hw.block_k == 32 && hw.block_k / hw.dim == PPB && hw.index_entries >= hw.dim);
    const potal_paired_pair_layout p0 = potal_paired_pair(0), p1 = potal_paired_pair(1);
    check("G pair 0 = banks 0 (A) / 1 (B) = Index buffer 0, spad id 1; pair 1 = banks 2 / 3, buffer 1, spad id 2",
          p0.activation_bank == 0 && p0.weight_bank == 1 && p0.index_buffer == 0 && p0.spad_id == 1 &&
              p1.activation_bank == 2 && p1.weight_bank == 3 && p1.index_buffer == 1 && p1.spad_id == 2);
    check("G pair sp rows: activation [2p*R, 2p*R+R), weight [(2p+1)*R, (2p+2)*R) = LoopMatmul half p",
          p0.activation_begin == 0 && p0.activation_end == hw.bank_rows && p0.weight_begin == hw.bank_rows &&
              p0.weight_end == 2 * hw.bank_rows && p1.activation_begin == 2 * hw.bank_rows &&
              p1.weight_end == 4 * hw.bank_rows && p1.weight_begin == 3 * hw.bank_rows);
    check("G status names are distinct",
          std::string(potal_paired_status_name(potal_paired_status::ok)) == "OK" &&
              std::string(potal_paired_status_name(potal_paired_status::residual_acc_capacity)) !=
                  potal_paired_status_name(potal_paired_status::main_acc_capacity));
}

void test_main_tiles_kept() {
    struct shape {
        size_t m, n, k;
    };
    const shape shapes[] = { { 150, 320, 128 }, { 200, 320, 128 }, { 100, 160, 2048 }, { 160, 320, 512 },
                             { 128, 3072, 768 }, { 1, 768, 768 },  { 48, 48, 160 },     { 80, 768, 768 },
                             { 16, 768, 768 } };
    bool kept = true, propagated = true, residual_kept = true, panels_ok = true;
    for (const shape & s : shapes) {
        potal_tile_plan main;
        if (!main_select(s.m, s.n, s.k, main)) {
            kept = false;
            continue;
        }
        const size_t stripe = main.tile_I * PD;
        const size_t tail   = s.m % stripe;
        for (size_t rows : { std::min(stripe, s.m), tail == 0 ? (size_t) 1 : tail }) {
            potal_paired_tile_plan p;
            std::string            why;
            const potal_paired_status st = potal_make_tile_plan(main, rows, s.n, s.k, nullptr, p, &why);
            print_plan("M", p);
            const size_t i_tiles = (rows + PD - 1) / PD;
            kept = kept && st == potal_paired_status::ok && why.empty() && p.tile_I == std::min(main.tile_I, i_tiles) &&
                   p.tile_J == main.tile_J && p.tile_K_main == main.tile_K && p.tile_K <= main.tile_K &&
                   p.tile_K % PPB == 0 && !p.residual;
            propagated = propagated && p.main.tile_K == p.tile_K && p.main.tile_I == p.tile_I &&
                         p.main.tile_J == p.tile_J && p.main.passes_per_block == PPB &&
                         p.k_panels == (p.k_tiles + p.tile_K - 1) / p.tile_K && p.j_groups == main.J0 &&
                         p.pad_J == main.pad_J && p.pad_I == i_tiles * PD - rows;
            panels_ok = panels_ok && p.panels.size() == p.k_panels && p.residual_windows == 0 &&
                        p.residual_empty_panels == 0;
            // A small residual (3 lane rows, one K per block in the first blocks) changes neither I nor J.
            const residual::decoded_packet r = synth(std::min<size_t>(3, rows), rows, dense_runs(std::min<size_t>(3, s.k / 32), 1));
            potal_paired_tile_plan         q;
            const potal_paired_status      sr = potal_make_tile_plan(main, rows, s.n, s.k, &r, q, &why);
            residual_kept = residual_kept && sr == potal_paired_status::ok && q.tile_I == p.tile_I && q.tile_J == p.tile_J &&
                            q.tile_K <= p.tile_K && q.residual && q.residual_lane_rows == r.rows.size();
        }
    }
    check("M tile_I (tail-limited) and tile_J are the Main auto-tiler's; the final K never exceeds its K", kept);
    check("M the final K is propagated into the Main execution plan (chunks, padding, J groups)", propagated);
    check("M a residual changes neither tile_I nor tile_J", residual_kept);
    check("M without a residual: one panel per Main K chunk, no Index Window, no empty-panel count", panels_ok);
}

void test_k_limits() {
    const potal_paired_geometry real = potal_paired_hw_geometry();
    struct k_case {
        const char *          name;
        size_t                m, n, k;
        size_t                lane_rows; // 0: no residual
        size_t                blocks, per_block;
        potal_paired_geometry hw;
        const char *          binding; // which limit must equal the final K
    };
    potal_paired_geometry small_bank = real, small_scale = real, tiny_bank = real;
    small_bank.bank_rows  = 1024;
    small_scale.scale_entries = 64;
    tiny_bank.bank_rows   = 512;
    const k_case cases[] = {
        { "real, no residual", 128, 3072, 768, 0, 0, 0, real, "main" },
        { "real, residual", 80, 768, 768, 96, 24, 5, real, "main" },
        { "activation bank binds through the residual", 80, 768, 768, 96, 24, 5, small_bank, "activation" },
        { "scale cache binds through the residual window", 80, 768, 768, 16, 24, 32, small_scale, "scale" },
        { "weight bank binds", 16, 768, 768, 20, 24, 3, tiny_bank, "weight" },
    };
    bool formula = true, binding = true, order = true;
    for (const k_case & c : cases) {
        potal_tile_plan main;
        if (!main_select(c.m, c.n, c.k, main)) {
            formula = false;
            continue;
        }
        const size_t                   rows = std::min(main.tile_I * PD, c.m); // one stripe of the Main plan
        const residual::decoded_packet r    = synth(c.lane_rows, rows, dense_runs(c.blocks, c.per_block));
        potal_paired_tile_plan         p;
        std::string                    why;
        const potal_paired_status      st = potal_make_tile_plan(main, rows, c.n, c.k, c.lane_rows ? &r : nullptr, p, &why,
                                                                 c.hw);
        std::printf("  K case %s: status %s %s\n", c.name, potal_paired_status_name(st), why.c_str());
        print_plan("K", p);
        if (st != potal_paired_status::ok) {
            formula = false;
            continue;
        }
        const size_t fpw = c.lane_rows ? std::min(c.hw.index_entries / c.hw.dim, p.fragments.size()) : 0;
        const expected_limits e = limits_of(c.hw, p.tile_I, p.tile_J, p.residual_row_tiles, fpw);
        const size_t hardware   = std::min({ e.weight, e.act_main, e.scale_main });
        const size_t res_limit  = std::min({ e.weight, e.act_res, e.scale_res });
        const size_t expected   = std::min({ main.tile_K, res_limit, hardware });
        formula = formula && p.fragments_per_window == fpw && p.tile_K_weight_limit == e.weight &&
                  p.tile_K_activation_limit == e.act_res && p.tile_K_scale_limit == e.scale_res &&
                  p.tile_K_hardware_limit == hardware && p.tile_K_residual_limit == res_limit && p.tile_K == expected;
        const std::string b = c.binding;
        binding = binding && ((b == "main" && p.tile_K == main.tile_K) || (b == "weight" && p.tile_K == e.weight) ||
                              (b == "activation" && p.tile_K == e.act_res && e.act_res < e.act_main &&
                               e.act_res < main.tile_K) ||
                              (b == "scale" && p.tile_K == e.scale_res && e.scale_res < main.tile_K));
        order = order && p.tile_K_residual_limit <= p.tile_K_hardware_limit && occupancy_holds(p, c.hw);
    }
    check("K final K = min(Main K, residual limit, hardware limit); each limit as recomputed from the geometry", formula);
    check("K the named limit binds in every case (Main, activation+residual, scale+residual window, weight)", binding);
    check("K the residual limit never exceeds the hardware limit; occupancy holds for the shrunken geometries", order);
}

void test_occupancy_and_offset() {
    const potal_paired_geometry hw = potal_paired_hw_geometry();
    struct shape {
        size_t m, n, k, lane_rows, blocks, per_block;
    };
    const shape shapes[] = { { 80, 768, 768, 96, 24, 32 }, { 16, 768, 768, 48, 24, 7 },   { 1, 768, 768, 5, 24, 32 },
                             { 48, 48, 160, 48, 5, 32 },   { 100, 160, 2048, 96, 64, 2 }, { 200, 320, 128, 6, 4, 32 },
                             { 128, 3072, 768, 96, 24, 32 } };
    bool occupancy = true, offset = true;
    for (const shape & s : shapes) {
        potal_tile_plan main;
        if (!main_select(s.m, s.n, s.k, main)) {
            occupancy = false;
            continue;
        }
        const size_t                   rows = std::min(main.tile_I * PD, s.m);
        const residual::decoded_packet r    = synth(std::min(s.lane_rows, 5 * rows), rows, dense_runs(s.blocks, s.per_block));
        potal_paired_tile_plan         p;
        std::string                    why;
        const potal_paired_status      st = potal_make_tile_plan(main, rows, s.n, s.k, &r, p, &why);
        std::printf("  O status %s %s\n", potal_paired_status_name(st), why.c_str());
        print_plan("O", p);
        if (st != potal_paired_status::ok) {
            occupancy = false;
            continue;
        }
        occupancy = occupancy && occupancy_holds(p, hw) && p.residual_acc_rows == p.residual_row_tiles * p.tile_J * PD &&
                    p.residual_row_tiles == (r.rows.size() + PD - 1) / PD && p.residual_a_rows ==
                    p.residual_row_tiles * p.fragments_per_window * PD;
        // Residual A: tile offset = Main A tiles; [offset, offset + rows) inside the activation bank of both pairs.
        const size_t res_begin = p.residual_a_offset_tiles * PD;
        for (size_t pair = 0; pair < potal_paired_pairs; ++pair) {
            const potal_paired_pair_layout l = potal_paired_pair(pair, hw);
            offset = offset && p.residual_a_offset_tiles == p.tile_I * p.tile_K && res_begin == p.main_a_rows &&
                     l.activation_begin + res_begin + p.residual_a_rows <= l.activation_end &&
                     l.weight_end - p.weight_rows >= l.weight_begin;
        }
    }
    check("O Main A + Residual A <= activation bank, B <= weight bank, ACC0 / ACC1 <= half ACC, windows <= Index entries, "
          "slots <= scale cache",
          occupancy);
    check("A Residual A starts at tile_I * tile_K tiles = right behind Main A and ends inside the activation bank of each pair",
          offset);
}

void test_panels_and_windows() {
    const potal_paired_geometry hw = potal_paired_hw_geometry();
    potal_tile_plan             main;
    bool                        ok = main_select(80, 160, 2048, main);
    // Residual in blocks 0, 1, 5, 6, 23 only: runs of PD + PD/4 (two fragments, a tail), 3 and 32 K.
    const std::vector<run_spec> runs = { { 0, first_k(PD + PD / 4) }, { 1, first_k(3) }, { 5, first_k(32) },
                                         { 6, first_k(1) }, { 23, first_k(PD) } };
    const residual::decoded_packet r = synth(40, 80, runs);
    potal_paired_tile_plan         p;
    std::string                    why;
    ok = ok && potal_make_tile_plan(main, 80, 160, 2048, &r, p, &why) == potal_paired_status::ok;
    print_plan("P", p);
    check("P plan with a sparse residual is accepted", ok);
    if (!ok) {
        return;
    }
    // Panels cover K once, in order, in whole scale blocks of at most tile_K tiles.
    bool cover = true, blocks = true, resident = true, empty = true, windows = true, no_loss = true;
    size_t next_k = 0, next_f = 0, fragments_seen = 0, compact_seen = 0, empty_panels = 0;
    for (const potal_paired_k_panel & panel : p.panels) {
        cover = cover && panel.k_begin == next_k && panel.k_rows == panel.k_tiles * PD && panel.k_tiles >= PPB &&
                panel.k_tiles <= p.tile_K && panel.k_tiles % PPB == 0;
        next_k = panel.k_begin + panel.k_rows;
        blocks = blocks && panel.block_begin == panel.k_begin / 32 && panel.block_end - panel.block_begin == panel.k_tiles / PPB;
        cover  = cover && panel.fragment_begin == next_f && panel.fragment_end >= panel.fragment_begin;
        next_f = panel.fragment_end;
        for (size_t f = panel.fragment_begin; f < panel.fragment_end; ++f) {
            const potal_paired_fragment & fr = p.fragments[f];
            blocks = blocks && fr.block_id >= panel.block_begin && fr.block_id < panel.block_end;
            for (size_t x = 0; x < fr.count; ++x) {
                const size_t orig = potal_paired_fragment_original_k(r, fr, x, hw);
                size_t       res  = SIZE_MAX;
                resident = resident && potal_paired_resident_k(panel, orig, &res) && res == orig - panel.k_begin &&
                           res < panel.k_rows && orig == fr.block_id * 32 + r.runs[fr.run].local_k[fr.local_begin + x];
                // The same original K is resident in no other panel.
                for (const potal_paired_k_panel & other : p.panels) {
                    resident = resident && (other.index == panel.index || !potal_paired_resident_k(other, orig, nullptr));
                }
            }
            compact_seen += fr.count;
            fragments_seen++;
        }
        if (panel.fragment_end == panel.fragment_begin) {
            empty_panels++;
            empty = empty && panel.windows.empty() && panel.residual_a_rows == 0 && panel.residual_k_compact == 0;
        }
        // Windows: contiguous over the panel's fragments, within the table, tails and fillers accounted.
        size_t wf = panel.fragment_begin, active_sum = 0, filler_sum = 0;
        for (const potal_paired_index_window & w : panel.windows) {
            windows = windows && w.fragment_begin == wf && w.fragment_end > w.fragment_begin &&
                      w.fragment_end - w.fragment_begin == w.k_tiles && w.k_tiles <= p.fragments_per_window &&
                      w.active_rows <= hw.index_entries && w.pad_k == w.k_tiles * PD - w.active_rows && w.pad_k < PD &&
                      w.active_rows == (w.k_tiles - 1) * PD + p.fragments[w.fragment_end - 1].count;
            size_t filler = 0;
            for (size_t f = w.fragment_begin; f + 1 < w.fragment_end; ++f) {
                filler += PD - p.fragments[f].count;
            }
            windows = windows && w.filler_slots == filler;
            wf         = w.fragment_end;
            active_sum += w.active_rows;
            filler_sum += w.filler_slots;
        }
        windows = windows && wf == panel.fragment_end;
        no_loss = no_loss && active_sum - filler_sum == panel.residual_k_compact;
    }
    const size_t expected_fragments = (PD + PD / 4 + PD - 1) / PD + 1 + (32 + PD - 1) / PD + 1 + 1;
    check("P panels cover the original K once, in order, in whole scale blocks of at most tile_K tiles",
          cover && next_k == 2048 && p.k_panels >= 2);
    check("P a scale block never straddles two panels; every fragment's block lies in its panel", blocks);
    check("P original K -> resident K = original - k_begin, resident in exactly one panel", resident);
    check("P panels without residual carry no Index Window (no Indexed LOOP_WS)",
          empty && empty_panels == p.residual_empty_panels && empty_panels > 0);
    check("W windows are contiguous, hold <= index_entries / DIM fragments, pad the last fragment's tail, count fillers",
          windows && p.fragments_per_window == std::min(hw.index_entries / PD, p.fragments.size()));
    check("W no compact K is lost: every fragment is in exactly one window, active - filler = compact K per panel",
          no_loss && fragments_seen == p.fragments.size() && next_f == p.fragments.size() &&
              p.fragments.size() == expected_fragments && compact_seen == r.k_compact);
    // A run longer than DIM is split into DIM fragments; the tail fragment keeps the remainder.
    const potal_paired_fragment & f0 = p.fragments[0], & f1 = p.fragments[1];
    check("W run of DIM + DIM/4 K: fragments of DIM and DIM/4, the second a non-DIM-aligned tail",
          f0.count == PD && f0.local_begin == 0 && f1.count == PD / 4 && f1.local_begin == PD && f1.run == 0 &&
              f1.compact_begin == PD);
    // More windows than one table holds: a residual of 6 full blocks in panel 0 needs 6 * (32 / DIM) fragments.
    const residual::decoded_packet big = synth(16, 16, dense_runs(6, 32));
    potal_tile_plan                m16;
    potal_paired_tile_plan         q;
    const bool                     big_ok = main_select(16, 768, 768, m16) &&
                        potal_make_tile_plan(m16, 16, 768, 768, &big, q, &why) == potal_paired_status::ok;
    print_plan("W", q);
    const size_t need = (q.fragments.size() + q.fragments_per_window - 1) / q.fragments_per_window;
    check("W a compact K selection larger than the Index Table is split into as many windows as needed",
          big_ok && q.fragments.size() == 6 * PPB && q.residual_windows == need && q.residual_windows >= 2 &&
              q.residual_filler_slots == 0);
}

void test_real_packets() {
    paired_fixture f(48, 64, 128, 11, [](size_t blk, size_t n) { return (int16_t) ((blk + n) % 3); });
    potal_tile_plan main;
    bool            ok = f.ok && main_select(f.M, f.N, f.K, main);
    const size_t    stripe = ok ? main.tile_I * PD : 16;
    std::vector<res_entry> entries;
    for (size_t row = 0; row < std::min<size_t>(stripe, f.M); row += 3) {
        for (size_t k = row % 7; k < f.K; k += 5 + row % 4) {
            entries.push_back({ row, k, (int32_t) ((int) (row * 31 + k * 17) % 601 - 300) });
        }
    }
    const grmd::StripePacketHandle packet = ok ? make_packet(0, 0, std::min<size_t>(stripe, f.M), f.K, f.N, entries) : nullptr;
    residual::decoded_packet       d;
    ok = ok && packet != nullptr && residual::decode_packet(*packet, f.weight, d) == grmd::RmdStatus::success;
    potal_paired_tile_plan  p;
    std::string             why;
    paired_residual_request req;
    ok = ok && potal_make_tile_plan(main, std::min<size_t>(stripe, f.M), f.N, f.K, &d, p, &why) == potal_paired_status::ok &&
         build_paired_residual_request(*packet, f.weight, stripe, req) == grmd::RmdStatus::success;
    check("R a real RMD packet decodes into an accepted paired plan (" + why + ")", ok);
    if (!ok) {
        return;
    }
    print_plan("R", p);
    bool same = p.fragments.size() == req.fragments.size() && p.residual_lane_rows == req.rows.size() &&
                p.residual_k_compact == req.k_compact && d.runs.size() == req.runs.size();
    for (size_t i = 0; same && i < p.fragments.size(); ++i) {
        const potal_paired_fragment & a = p.fragments[i];
        const paired_fragment &       b = req.fragments[i];
        same = a.run == b.run && a.block_id == b.block_id && a.compact_begin == b.compact_begin &&
               a.local_begin == b.local_begin && a.count == b.count;
        for (size_t x = 0; same && x < a.count; ++x) {
            same = potal_paired_fragment_original_k(d, a, x, potal_paired_hw_geometry()) == req.original_k(b, x);
        }
    }
    check("R fragments (run, block, compact / local begin, count) and every original K equal the paired descriptor's", same);
    // Every selected K of the packet is resident in exactly one panel, and the panels' compact K add up.
    size_t compact = 0;
    bool   resident = true;
    for (const potal_paired_k_panel & panel : p.panels) {
        compact += panel.residual_k_compact;
        for (size_t fi = panel.fragment_begin; fi < panel.fragment_end; ++fi) {
            for (size_t x = 0; x < p.fragments[fi].count; ++x) {
                size_t res = 0;
                resident = resident && potal_paired_resident_k(panel, potal_paired_fragment_original_k(d, p.fragments[fi], x, potal_paired_hw_geometry()), &res) && res < panel.k_rows;
            }
        }
    }
    check("R the packet's compact K is partitioned over the panels without loss", resident && compact == d.k_compact);
}

void test_failures() {
    const potal_paired_geometry real = potal_paired_hw_geometry();
    potal_tile_plan             m80, m16;
    bool                        ok = main_select(80, 768, 768, m80) && main_select(16, 768, 768, m16);
    check("E Main plans for the refusal cases", ok);
    if (!ok) {
        return;
    }
    const auto plan_of = [&](const potal_tile_plan & main, size_t rows, size_t n, size_t k,
                             const residual::decoded_packet * r, const potal_paired_geometry & hw,
                             potal_paired_tile_plan * out = nullptr) {
        potal_paired_tile_plan    p;
        std::string               why;
        const potal_paired_status st = potal_make_tile_plan(main, rows, n, k, r, p, &why, hw);
        std::printf("  E %s: %s\n", potal_paired_status_name(st), why.c_str());
        if (out != nullptr) {
            *out = p;
        }
        return std::make_pair(st, why);
    };
    const auto refused = [](const std::pair<potal_paired_status, std::string> & r, potal_paired_status want) {
        return r.first == want && !r.second.empty();
    };
    const residual::decoded_packet good = synth(32, 80, dense_runs(4, 8));

    // Inconsistent residual geometry.
    residual::decoded_packet unsorted_rows = good;
    std::swap(unsorted_rows.rows[0], unsorted_rows.rows[1]);
    residual::decoded_packet descending_blocks = good;
    std::swap(descending_blocks.runs[0].block_id, descending_blocks.runs[1].block_id);
    residual::decoded_packet local_beyond = good;
    local_beyond.runs[0].local_k.back() = 32;
    residual::decoded_packet bad_compact = good;
    bad_compact.k_compact += 1;
    residual::decoded_packet rows_only = good;
    rows_only.runs.clear();
    residual::decoded_packet block_beyond = good;
    block_beyond.runs.back().block_id = 24;
    residual::decoded_packet row_beyond = good;
    row_beyond.rows.back().source_row = 80;
    check("E residual geometry refused: rows not ascending, blocks not ascending, local K >= 32, k_compact mismatch, "
          "runs without rows, block beyond K, row beyond the stripe",
          refused(plan_of(m80, 80, 768, 768, &unsorted_rows, real), potal_paired_status::invalid_argument) &&
              refused(plan_of(m80, 80, 768, 768, &descending_blocks, real), potal_paired_status::invalid_argument) &&
              refused(plan_of(m80, 80, 768, 768, &local_beyond, real), potal_paired_status::invalid_argument) &&
              refused(plan_of(m80, 80, 768, 768, &bad_compact, real), potal_paired_status::invalid_argument) &&
              refused(plan_of(m80, 80, 768, 768, &rows_only, real), potal_paired_status::invalid_argument) &&
              refused(plan_of(m80, 80, 768, 768, &block_beyond, real), potal_paired_status::invalid_argument) &&
              refused(plan_of(m80, 80, 768, 768, &row_beyond, real), potal_paired_status::invalid_argument));

    // Main plan / shape mismatches.
    potal_tile_plan odd_k = m80;
    odd_k.tile_K += 1; // not a whole scale block
    potal_tile_plan wide_j = m80;
    wide_j.tile_J = 49; // more J tiles than N has
    potal_tile_plan wrong_ppb = m80;
    wrong_ppb.passes_per_block = PPB + 1;
    check("E Main plan refused: K not a whole scale block, tile_J beyond N, other passes_per_block, K not a block multiple",
          refused(plan_of(odd_k, 80, 768, 768, nullptr, real), potal_paired_status::invalid_argument) &&
              refused(plan_of(wide_j, 80, 768, 768, nullptr, real), potal_paired_status::invalid_argument) &&
              refused(plan_of(wrong_ppb, 80, 768, 768, nullptr, real), potal_paired_status::invalid_argument) &&
              refused(plan_of(m80, 80, 768, 760, nullptr, real), potal_paired_status::invalid_argument));

    // Main output tiles beyond ACC0 of a smaller accumulator.
    potal_paired_geometry small_acc = real;
    small_acc.acc_rows        = m80.tile_I * m80.tile_J * PD; // ACC0 = half of it
    check("E Main ACC0 capacity refused (the Main tiles are not shrunk here either)",
          refused(plan_of(m80, 80, 768, 768, nullptr, small_acc), potal_paired_status::main_acc_capacity));

    // Residual lane rows beyond ACC1 with the Main I/J kept.
    const residual::decoded_packet five_lanes = synth(5 * 80, 80, dense_runs(4, 8));
    potal_paired_tile_plan         kept;
    const auto                     acc1 = plan_of(m80, 80, 768, 768, &five_lanes, real, &kept);
    check("E Residual ACC1 capacity refused with diagnostics; tile_I / tile_J stay the Main tiler's",
          refused(acc1, potal_paired_status::residual_acc_capacity) && kept.tile_I == m80.tile_I &&
              kept.tile_J == m80.tile_J && kept.residual_lane_rows == 400 && kept.tile_K == 0);

    // Index Table below one DIM fragment.
    potal_paired_geometry small_index = real;
    small_index.index_entries   = PD - 1;
    check("E Index Table smaller than DIM refused",
          refused(plan_of(m80, 80, 768, 768, &good, small_index), potal_paired_status::index_window_capacity));

    // Weight bank: not even one scale block of B with tile_J fits.
    potal_paired_geometry tiny_weight = real;
    tiny_weight.bank_rows         = m16.tile_J * PD * PPB - 1;
    potal_paired_tile_plan        wb;
    const auto                    weight = plan_of(m16, 16, 768, 768, nullptr, tiny_weight, &wb);
    check("E weight bank capacity refused: tile_K_weight_limit 0, I/J kept",
          refused(weight, potal_paired_status::weight_bank_capacity) && wb.tile_K_weight_limit == 0 &&
              wb.tile_J == m16.tile_J);

    // Activation bank: Main A of one block fits alone, not beside the Residual A.
    potal_tile_plan m80n16;
    ok = main_select(80, 16, 768, m80n16);
    potal_paired_geometry tiny_act = real;
    tiny_act.bank_rows          = m80n16.tile_I * PD * PPB + PD * PPB; // one Main A block + half a Residual window
    const residual::decoded_packet two_tiles = synth(2 * PD, 80, dense_runs(4, 32));
    potal_paired_tile_plan         ab;
    const auto                     act = plan_of(m80n16, 80, 16, 768, &two_tiles, tiny_act, &ab);
    check("E activation bank capacity refused only because of the residual: hardware limit > 0, residual limit 0",
          ok && refused(act, potal_paired_status::activation_bank_capacity) && ab.tile_K_hardware_limit >= PPB &&
              ab.tile_K_residual_limit == 0 && ab.tile_K_activation_limit == 0);

    // Scale cache: one Main scale block with tile_J slots does not fit.
    potal_paired_geometry tiny_scale = real;
    tiny_scale.scale_entries     = m16.tile_J - 1;
    check("E scale cache capacity refused",
          refused(plan_of(m16, 16, 768, 768, nullptr, tiny_scale), potal_paired_status::scale_window_capacity));

    // One paired call executes one Main I group: more rows than tile_I * DIM are refused.
    potal_tile_plan m128;
    const bool      m128_ok = main_select(128, 768, 768, m128) && m128.tile_I * PD < 128;
    potal_paired_tile_plan wide;
    const auto     too_many = plan_of(m128, 128, 768, 768, nullptr, real, &wide);
    check("E rows beyond one Main I group (128 rows, tile_I " + std::to_string(m128.tile_I) + ") refused, never truncated",
          m128_ok && refused(too_many, potal_paired_status::invalid_argument) &&
              too_many.second.find("one stripe") != std::string::npos);
    // Residual rows whose digits are all zero: no fragment survives -> refused (decode_packet never yields it).
    residual::decoded_packet zero = synth(4, 80, dense_runs(2, 8));
    zero.digits.assign(zero.rows.size() * zero.k_compact, 0);
    check("E residual rows with every digit zero refused",
          refused(plan_of(m80, 80, 768, 768, &zero, real), potal_paired_status::invalid_argument));

    // Determinism: the same request twice gives the same plan.
    potal_paired_tile_plan a, b;
    std::string            why;
    const bool             det = potal_make_tile_plan(m80, 80, 768, 768, &good, a, &why) == potal_paired_status::ok &&
                     potal_make_tile_plan(m80, 80, 768, 768, &good, b, &why) == potal_paired_status::ok && same_plan(a, b);
    check("E the same request always gives the same plan", det);
}

// ---- H3-1 ---------------------------------------------------------------------------------

using op_kind = potal_paired_op_kind;

struct exec_case {
    potal_tile_plan             main;
    residual::decoded_packet    residual;
    bool                        has_residual = false;
    potal_paired_tile_plan      tile;
    potal_paired_execution_plan exec;
    potal_paired_status         status = potal_paired_status::invalid_argument;
    std::string                 why;
};

bool build_case(exec_case & c, size_t m, size_t n, size_t k, size_t rows,
                const potal_paired_geometry & hw = potal_paired_hw_geometry()) {
    if (!main_select(m, n, k, c.main)) {
        return false;
    }
    c.status = potal_make_tile_plan(c.main, rows, n, k, c.has_residual ? &c.residual : nullptr, c.tile, &c.why, hw);
    if (c.status != potal_paired_status::ok) {
        std::printf("  tile plan refused: %s %s\n", potal_paired_status_name(c.status), c.why.c_str());
        return false;
    }
    c.status = potal_make_execution_plan(c.tile, c.has_residual ? &c.residual : nullptr, c.exec, &c.why, hw);
    if (c.status != potal_paired_status::ok) {
        std::printf("  execution plan refused: %s %s\n", potal_paired_status_name(c.status), c.why.c_str());
        return false;
    }
    return true;
}

void print_exec(const char * name, const potal_paired_execution_plan & e, size_t max_ops = 40) {
    std::printf("  %s: ops=%zu main=%zu residual=%zu weight_loads=%zu index=%zu writes=%zu scale=%zu/%zu acc=%zu "
                "retire=%zu complete_deps=%zu residencies=%zu\n",
                name, e.ops.size(), e.main_gemms, e.residual_gemms, e.weight_loads, e.index_programs, e.index_writes,
                e.scale_programs, e.scale_releases, e.acc_completes, e.pair_retires, e.complete_deps,
                e.residencies.size());
    for (size_t i = 0; i < e.ops.size() && i < max_ops; ++i) {
        std::printf("    %s\n", potal_paired_op_text(e.ops[i]).c_str());
    }
}

std::vector<op_kind> kinds_of(const potal_paired_execution_plan & e) {
    std::vector<op_kind> k;
    for (const potal_paired_op & op : e.ops) {
        k.push_back(op.kind);
    }
    return k;
}

const potal_paired_op * find_op(const potal_paired_execution_plan & e, op_kind kind, size_t residency,
                                size_t window = potal_paired_no_index) {
    for (const potal_paired_op & op : e.ops) {
        if (op.kind == kind && op.residency == residency && (window == potal_paired_no_index || op.window == window)) {
            return &op;
        }
    }
    return nullptr;
}

bool depends(const potal_paired_op & op, size_t on, bool complete) {
    for (const potal_paired_dep & d : op.deps) {
        if (d.op == on && (!complete || d.complete)) {
            return true;
        }
    }
    return false;
}

// Properties every accepted plan must have, checked here without the validator.
bool exec_invariants(const exec_case & c) {
    const potal_paired_execution_plan & e    = c.exec;
    const potal_paired_tile_plan &      t    = c.tile;
    const size_t                        res  = t.j_groups * t.k_panels;
    bool                                ok   = e.residencies.size() == res && e.main_gemms == res &&
              e.residual_gemms == t.j_groups * t.residual_windows && e.pair_retires == res && e.weight_loads == res &&
              e.residual_rows.size() == t.residual_lane_rows &&
              potal_validate_execution_plan(e, c.has_residual ? &c.residual : nullptr, nullptr) == potal_paired_status::ok;
    // Residencies in (J group, K panel) order, pairs alternating, every panel's Main once, every window once.
    std::vector<size_t> main_count(res, 0), window_count(res, 0);
    size_t              prev_residency = 0;
    for (const potal_paired_op & op : e.ops) {
        ok = ok && op.pair == op.residency % 2 && op.residency == op.j_group * t.k_panels + op.panel;
        if (op.kind == op_kind::main_gemm) {
            main_count[op.residency]++;
            ok = ok && op.residency >= prev_residency;
            prev_residency = op.residency;
        }
        if (op.kind == op_kind::residual_gemm) {
            window_count[op.residency]++;
        }
    }
    for (size_t r = 0; r < res; ++r) {
        ok = ok && main_count[r] == 1 && window_count[r] == t.panels[r % t.k_panels].windows.size() &&
             e.residencies[r].index == r && e.residencies[r].pair == r % 2 && e.residencies[r].consumers == 1 + window_count[r];
    }
    // The residual contributions of a J group appear in panel / window order with one first and one last.
    for (size_t g = 0; g < t.j_groups; ++g) {
        size_t firsts = 0, lasts = 0, seen = 0, main_firsts = 0, main_lasts = 0;
        for (const potal_paired_op & op : e.ops) {
            if (op.j_group != g) {
                continue;
            }
            if (op.kind == op_kind::residual_gemm) {
                ok = ok && op.first == (seen == 0);
                seen++;
                firsts += op.first;
                lasts += op.last;
                ok = ok && (!op.last || seen == t.residual_windows);
            }
            if (op.kind == op_kind::main_gemm) {
                main_firsts += op.first;
                main_lasts += op.last;
                ok = ok && op.first == (op.panel == 0) && op.last == (op.panel + 1 == t.k_panels);
            }
        }
        ok = ok && main_firsts == 1 && main_lasts == 1 && (t.residual_windows == 0 || (firsts == 1 && lasts == 1));
    }
    return ok;
}

void test_exec_single_panel() {
    exec_case c;
    bool      ok = build_case(c, 48, 48, 160, 48) && exec_invariants(c);
    print_exec("X1", c.exec);
    const std::vector<op_kind> want = { op_kind::scale_program, op_kind::load_main_a, op_kind::load_weight,
                                        op_kind::main_gemm,     op_kind::acc_complete, op_kind::scale_release,
                                        op_kind::pair_retire };
    check("X1 single K panel without residual: scale, loads, Main GEMM (first and last), ACC0 complete, release, retire",
          ok && c.tile.k_panels == 1 && c.tile.j_groups == 1 && kinds_of(c.exec) == want && c.exec.ops[3].first &&
              c.exec.ops[3].last && c.exec.ops[3].acc_bank == 0 && c.exec.ops[4].acc_bank == 0 &&
              !c.exec.ops[4].reconstruction_required && depends(c.exec.ops[5], 3, true) && depends(c.exec.ops[6], 3, true) &&
              c.exec.complete_deps == 2 && c.exec.pairs[0].phase == potal_paired_pair_phase::retired);

    exec_case r;
    r.has_residual = true;
    r.residual     = synth(8, 48, { { 0, first_k(5) } });
    ok             = build_case(r, 48, 48, 160, 48) && exec_invariants(r);
    print_exec("X2", r.exec);
    const std::vector<op_kind> want_r = { op_kind::scale_program,   op_kind::load_main_a,   op_kind::load_weight,
                                          op_kind::main_gemm,       op_kind::acc_complete,  op_kind::index_program,
                                          op_kind::scale_program,   op_kind::load_residual_a, op_kind::residual_gemm,
                                          op_kind::acc_complete,    op_kind::scale_release, op_kind::scale_release,
                                          op_kind::pair_retire };
    const potal_paired_op * ip = ok ? find_op(r.exec, op_kind::index_program, 0) : nullptr;
    const potal_paired_op * rg = ok ? find_op(r.exec, op_kind::residual_gemm, 0) : nullptr;
    check("X2 one panel, one Index Window: Residual follows Main on the same resident B, entries 0..4, ACC1 complete needs reconstruction",
          ok && kinds_of(r.exec) == want_r && ip != nullptr && rg != nullptr && ip->index_b_base == r.exec.ops[2].b_sp_begin &&
              ip->index_source_k_rows == 160 && ip->index_active_rows == 5 && ip->entry_end - ip->entry_begin == 5 &&
              ip->same_buffer_predecessor == potal_paired_no_index && rg->first && rg->last && rg->pad_k == PD - 5 &&
              rg->k_tiles == 1 && rg->b_sp_begin == ip->index_b_base && r.exec.ops[9].reconstruction_required &&
              depends(*rg, ip->id, false) && depends(*rg, 2, false) && depends(r.exec.ops[12], rg->id, true) &&
              r.exec.ops[11].scale_window == 0 && depends(r.exec.ops[11], 3, true) && r.exec.index_writes == 5);
    bool entries = ok;
    for (size_t e = 0; ok && e < 5; ++e) {
        const potal_paired_index_entry & en = r.exec.index_entries[ip->entry_begin + e];
        entries = entries && en.slot == e && en.resident_k == e && en.original_k == e && en.compact_k == e && !en.filler;
    }
    check("X2 Index entries: slot = resident_k = original_k - k_begin = compact k, no filler", entries);
}

void test_exec_multi_panel() {
    // 80 x 160 x 2048: tile_K 50 -> 3 panels (50, 50, 28 tiles), 2 J groups.
    exec_case c;
    c.has_residual = true;
    c.residual     = synth(40, 80, dense_runs(64, 7));
    bool ok        = build_case(c, 80, 160, 2048, 80) && exec_invariants(c);
    print_exec("X3", c.exec, 60);
    check("X3 several K panels with residual: every (J group, panel) has Main then its windows; first / last once per ACC epoch",
          ok && c.tile.k_panels == 3 && c.tile.j_groups == 2 && c.exec.residual_gemms == 2 * c.tile.residual_windows);
    // Per residency: Main GEMM precedes every Residual GEMM; windows ascend.
    bool order = ok;
    for (const potal_paired_residency & r : c.exec.residencies) {
        size_t last_w = 0;
        for (const potal_paired_op & op : c.exec.ops) {
            if (op.residency != r.index || op.kind != op_kind::residual_gemm) {
                continue;
            }
            order = order && op.id > r.main_gemm_op && op.window >= last_w;
            last_w = op.window;
        }
        order = order && r.windows == c.tile.panels[r.panel].windows.size();
    }
    check("X3 Main GEMM before the panel's Residual GEMMs, windows in order", order);
    // X11: ACC flags across panels: first at panel 0 / window 0, last at panel 2 / last window, never elsewhere.
    bool flags = ok;
    for (const potal_paired_op & op : c.exec.ops) {
        if (op.kind == op_kind::residual_gemm) {
            flags = flags && op.first == (op.panel == 0 && op.window == 0) &&
                    op.last == (op.panel == 2 && op.window + 1 == c.tile.panels[2].windows.size());
        }
        if (op.kind == op_kind::main_gemm) {
            flags = flags && op.first == (op.panel == 0) && op.last == (op.panel == 2) && op.k_tiles == c.tile.panels[op.panel].k_tiles;
        }
    }
    check("X11 ACC0 / ACC1 first and last only at the true first / last contribution across panels and windows", flags);

    // X4: residual in blocks 0 and 63 only: panel 1 has no Index transaction.
    exec_case s;
    s.has_residual = true;
    s.residual     = synth(10, 80, { { 0, first_k(PD) }, { 63, first_k(3) } });
    ok             = build_case(s, 80, 160, 2048, 80) && exec_invariants(s);
    print_exec("X4", s.exec, 0);
    size_t index_ops[3] = { 0, 0, 0 }, residual_ops[3] = { 0, 0, 0 };
    for (const potal_paired_op & op : s.exec.ops) {
        if (op.kind == op_kind::index_program) {
            index_ops[op.panel]++;
        }
        if (op.kind == op_kind::residual_gemm) {
            residual_ops[op.panel]++;
        }
    }
    const potal_paired_op * first = ok ? find_op(s.exec, op_kind::residual_gemm, 0) : nullptr;
    const potal_paired_op * last  = ok ? find_op(s.exec, op_kind::residual_gemm, 2) : nullptr;
    check("X4 residual spanning panels 0 and 2: panel 1 has no Index transaction and no Indexed GEMM; first at panel 0, last at panel 2",
          ok && index_ops[0] == 2 && index_ops[1] == 0 && index_ops[2] == 2 && residual_ops[1] == 0 && first != nullptr &&
              last != nullptr && first->first && !first->last && last->last && !last->first &&
              s.tile.residual_empty_panels == 1);
}

void test_exec_windows() {
    // 16 x 768 x 768: tile_K 28 (weight bank) -> panels of 14 and 10 blocks; blocks 0..5 = 12 fragments = 6 windows
    // in panel 0, blocks 14..16 = 3 windows in panel 1 (pair 1).
    std::vector<run_spec> runs = dense_runs(6, 32);
    for (uint32_t blk = 14; blk < 17; ++blk) {
        runs.emplace_back(blk, first_k(32));
    }
    exec_case c;
    c.has_residual = true;
    c.residual     = synth(16, 16, runs);
    bool ok        = build_case(c, 16, 768, 768, 16) && exec_invariants(c);
    print_exec("X5", c.exec, 0);
    bool chain = ok && c.tile.panels[0].windows.size() == 6;
    for (size_t w = 1; ok && w < 6; ++w) {
        const potal_paired_op * ip   = find_op(c.exec, op_kind::index_program, 0, w);
        const potal_paired_op * prev = find_op(c.exec, op_kind::residual_gemm, 0, w - 1);
        const potal_paired_op * lr   = find_op(c.exec, op_kind::load_residual_a, 0, w);
        chain = chain && ip != nullptr && prev != nullptr && lr != nullptr && ip->same_buffer_predecessor == prev->id &&
                depends(*ip, prev->id, false) && depends(*lr, prev->id, true) && ip->id > prev->id &&
                ip->index_active_rows == 2 * PD;
    }
    check("X5 / X10 six windows in one panel: each same-buffer Index update follows the previous Indexed GEMM "
          "(hardware-stalled), each Residual A reload waits for its completion",
          chain && c.tile.k_panels == 2 && c.tile.panels[1].windows.size() == 3);
    // X9: the first window of residency 1 (buffer 1) does not depend on any buffer-0 gemm.
    const potal_paired_op * ip1 = ok ? find_op(c.exec, op_kind::index_program, 1, 0) : nullptr;
    bool                    cross = ok && ip1 != nullptr && ip1->index_buffer == 1 && ip1->same_buffer_predecessor == potal_paired_no_index;
    for (const potal_paired_op & op : c.exec.ops) {
        if (ok && ip1 != nullptr && op.residency == 0 && op.kind == op_kind::residual_gemm) {
            cross = cross && !depends(*ip1, op.id, false);
        }
    }
    // ... and residency 2 (buffer 0 again) waits for residency 0's last Indexed GEMM only through the retire.
    const potal_paired_op * ip2 = ok ? find_op(c.exec, op_kind::index_program, 2, 0) : nullptr;
    check("X9 cross-buffer Index programming: buffer 1's first window depends on no buffer-0 gemm; buffer 0's next "
          "residency names its predecessor gemm",
          cross && ip2 != nullptr && ip2->index_buffer == 0 &&
              ip2->same_buffer_predecessor == c.exec.residencies[0].last_residual_op);

    // X6: tails. Runs of PD + PD/4, 3, 32, 1 K in blocks 0, 1, 2, 3 -> windows: [PD, PD/4] [3, PD] [PD, 1].
    exec_case t;
    t.has_residual = true;
    t.residual     = synth(5, 16, { { 0, first_k(PD + PD / 4) }, { 1, first_k(3) }, { 2, first_k(32) }, { 3, first_k(1) } });
    ok             = build_case(t, 16, 768, 768, 16) && exec_invariants(t);
    print_exec("X6", t.exec, 0);
    const potal_paired_op * w0 = ok ? find_op(t.exec, op_kind::index_program, 0, 0) : nullptr;
    const potal_paired_op * w1 = ok ? find_op(t.exec, op_kind::index_program, 0, 1) : nullptr;
    const potal_paired_op * g0 = ok ? find_op(t.exec, op_kind::residual_gemm, 0, 0) : nullptr;
    const potal_paired_op * g1 = ok ? find_op(t.exec, op_kind::residual_gemm, 0, 1) : nullptr;
    bool tails = ok && w0 != nullptr && w1 != nullptr && g0 != nullptr && g1 != nullptr && w0->index_active_rows == PD + PD / 4 &&
                 g0->pad_k == PD - PD / 4 && w1->index_active_rows == 2 * PD && g1->pad_k == 0;
    size_t fillers = 0, actives = 0;
    for (size_t e = ok ? w1->entry_begin : 0; ok && e < w1->entry_end; ++e) {
        const potal_paired_index_entry & en = t.exec.index_entries[e];
        if (en.filler) {
            fillers++;
            tails = tails && en.compact_k == potal_paired_no_index && en.fragment == 2 && en.slot >= 3 && en.slot < PD &&
                    en.resident_k == 32 && en.original_k == 32; // block 1's first K, a resident row
        } else {
            actives++;
            tails = tails && en.original_k - en.resident_k == 0 && en.resident_k < t.tile.panels[0].k_rows;
        }
    }
    check("X6 tails: window 0 active PD+PD/4 with pad_k PD-PD/4, window 1 = [3 K + PD-3 fillers] [PD], fillers point at a resident row with no compact K",
          tails && fillers == PD - 3 && actives == 3 + PD);
}

void test_exec_pairs() {
    exec_case c;
    c.has_residual = true;
    c.residual     = synth(32, 80, dense_runs(64, 2));
    bool ok        = build_case(c, 80, 160, 2048, 80) && exec_invariants(c);
    print_exec("X7", c.exec, 0);
    // X7 / X8: pair r % 2; residency r (r >= 2) loads only after pair_retire(r - 2), which completes Main and every Residual GEMM of r - 2.
    bool alternate = ok, retire = ok;
    for (const potal_paired_residency & r : c.exec.residencies) {
        alternate = alternate && r.pair == r.index % 2;
        const potal_paired_op & ret = c.exec.ops[r.retire_op];
        retire = retire && ret.kind == op_kind::pair_retire && depends(ret, r.main_gemm_op, true);
        for (const potal_paired_op & op : c.exec.ops) {
            if (op.residency == r.index && op.kind == op_kind::residual_gemm) {
                retire = retire && depends(ret, op.id, true) && op.id < ret.id;
            }
        }
        if (r.index >= 2) {
            const potal_paired_residency & prev = c.exec.residencies[r.index - 2];
            const potal_paired_op * la = find_op(c.exec, op_kind::load_main_a, r.index);
            const potal_paired_op * lb = find_op(c.exec, op_kind::load_weight, r.index);
            alternate = alternate && la != nullptr && lb != nullptr && prev.retire_op < la->id && prev.retire_op < lb->id &&
                        depends(*la, prev.retire_op, false) && depends(*lb, prev.retire_op, false) && prev.pair == r.pair;
        }
    }
    check("X7 pairs alternate per residency; a pair is loaded again only after the retire of its previous residency", alternate);
    check("X8 retire completes Main and every Residual consumer of the residency before the weight bank is rewritten", retire);
    // Transitions: free -> loading -> resident -> retired -> loading ... per pair, in op order.
    bool phases = ok;
    potal_paired_pair_phase phase[2] = { potal_paired_pair_phase::free, potal_paired_pair_phase::free };
    for (const potal_paired_pair_transition & tr : c.exec.transitions) {
        phases = phases && tr.from == phase[tr.pair] &&
                 ((tr.to == potal_paired_pair_phase::loading && (tr.from == potal_paired_pair_phase::free || tr.from == potal_paired_pair_phase::retired)) ||
                  (tr.to == potal_paired_pair_phase::resident && tr.from == potal_paired_pair_phase::loading) ||
                  (tr.to == potal_paired_pair_phase::retired && tr.from == potal_paired_pair_phase::resident));
        phase[tr.pair] = tr.to;
    }
    check("X7 pair phases: FREE -> LOADING -> RESIDENT -> RETIRED -> LOADING ..., both pairs end RETIRED",
          phases && c.exec.pairs[0].phase == potal_paired_pair_phase::retired &&
              c.exec.pairs[1].phase == potal_paired_pair_phase::retired && c.exec.pairs[0].residencies + c.exec.pairs[1].residencies == 6);
}

void test_exec_zero_fragments() {
    // Digits known: run 0 (2 fragments) has zeros in every row for its second fragment; run 1 is nonzero.
    residual::decoded_packet d = synth(4, 16, { { 0, first_k(PD + 2) }, { 1, first_k(2) } });
    d.digits.assign(d.rows.size() * d.k_compact, 0);
    for (size_t row = 0; row < d.rows.size(); ++row) {
        d.digits[row * d.k_compact + 0]      = 1;            // fragment 0 of run 0
        d.digits[row * d.k_compact + PD + 2] = (int8_t) -3;  // run 1
    }
    exec_case c;
    c.has_residual = true;
    c.residual     = d;
    const bool ok  = build_case(c, 16, 768, 768, 16) && exec_invariants(c);
    print_exec("X12", c.exec, 0);
    bool entries = ok;
    for (const potal_paired_index_entry & en : c.exec.index_entries) {
        entries = entries && (en.filler || en.compact_k < PD || en.compact_k >= PD + 2); // never the zero fragment's K
    }
    check("X12 an all-zero fragment is left out: 2 fragments executed, one window, no entry of the zero K, counted as skipped",
          ok && c.tile.residual_zero_fragments == 1 && c.tile.fragments.size() == 2 && c.tile.residual_windows == 1 &&
              c.exec.residual_gemms == c.tile.j_groups && entries && c.exec.index_writes == c.tile.j_groups * (PD + 2) &&
              c.tile.fragments[1].run == 1);
}

void test_exec_scale() {
    exec_case c;
    c.has_residual = true;
    c.residual     = synth(32, 80, dense_runs(64, 2));
    bool ok        = build_case(c, 80, 160, 2048, 80) && exec_invariants(c);
    bool gens = ok, lifetime = ok, placement = ok;
    size_t expect_gen = 1;
    std::vector<int> live(c.exec.scale_windows.size(), 0);
    for (const potal_paired_op & op : c.exec.ops) {
        if (op.kind == op_kind::scale_program) {
            const potal_paired_scale_window & w = c.exec.scale_windows[op.scale_window];
            gens = gens && w.generation == expect_gen++ && w.program_op == op.id;
            for (size_t o = 0; o < live.size(); ++o) {
                const potal_paired_scale_window & x = c.exec.scale_windows[o];
                lifetime = lifetime && !(live[o] && x.slot_base < w.slot_base + w.slots && w.slot_base < x.slot_base + x.slots);
            }
            live[op.scale_window] = 1;
            placement = placement && (w.residual ? w.slot_base == c.tile.scale_slots_main : w.slot_base == 0) &&
                        w.slot_base + w.slots <= POTAL_HW_SCALE_ENTRIES &&
                        w.blocks == (w.residual ? c.tile.panels[op.panel].windows[w.window].k_tiles
                                                : c.tile.panels[op.panel].k_tiles / c.tile.passes_per_block);
        }
        if (op.kind == op_kind::scale_release) {
            const potal_paired_scale_window & w = c.exec.scale_windows[op.scale_window];
            lifetime = lifetime && live[op.scale_window] == 1 && w.release_op == op.id && w.consumer_op < op.id &&
                       depends(op, w.consumer_op, true) &&
                       c.exec.ops[w.consumer_op].scale_window == op.scale_window;
            live[op.scale_window] = 0;
        }
        if (op.kind == op_kind::main_gemm || op.kind == op_kind::residual_gemm) {
            lifetime = lifetime && live[op.scale_window] == 1;
        }
    }
    for (int l : live) {
        lifetime = lifetime && l == 0;
    }
    check("X13 scale windows: generations 1..N in program order, Main at slot 0 and Residual behind it, one block per "
          "Main scale block / Residual fragment",
          gens && placement && expect_gen == c.exec.scale_windows.size() + 1);
    check("X13 a scale window is live from its program to a release that waits for its consumer's completion; live "
          "windows never overlap; every window is released",
          lifetime);
}

void test_exec_tails() {
    // rows 70 (pad_i 10), N 140 (9 tiles: J groups of 5 and 4 with pad_j 4), K 2048 (panels 50, 50, 28).
    exec_case c;
    c.has_residual = true;
    c.residual     = synth(21, 70, { { 0, first_k(9) }, { 40, first_k(32) }, { 63, first_k(PD + 1) } });
    bool ok        = build_case(c, 70, 140, 2048, 70) && exec_invariants(c);
    print_exec("X14", c.exec, 0);
    bool tails = ok && c.tile.j_groups == 2 && c.tile.k_panels == 3 && c.tile.pad_I == 80 - 70 && c.tile.pad_J == 144 - 140 &&
                 c.tile.residual_pad_I == 2 * PD - 21;
    for (const potal_paired_op & op : c.exec.ops) {
        if (op.kind == op_kind::main_gemm) {
            tails = tails && op.pad_i == 10 && op.pad_j == (op.j_group == 1 ? 4 : 0) && op.j_tiles == (op.j_group == 1 ? 4u : 5u) &&
                    op.k_tiles == (op.panel == 2 ? 28u : 50u) && op.b_sp_rows == op.k_tiles * op.j_tiles * PD;
        }
        if (op.kind == op_kind::residual_gemm) {
            tails = tails && op.pad_i == 2 * PD - 21 && op.pad_j == (op.j_group == 1 ? 4 : 0) && op.i_tiles == 2 &&
                    op.work_ids == 2 * op.j_tiles;
        }
        if (op.kind == op_kind::index_program) {
            tails = tails && op.index_j_tiles == (op.j_group == 1 ? 4u : 5u) &&
                    op.index_source_k_rows == (op.panel == 2 ? 28u * PD : 50u * PD);
        }
    }
    check("X14 I / J / K tails: pad_i 10, last J group 4 tiles with pad_j 4, last panel 28 tiles; Residual rows padded to 2 tiles; "
          "INDEX_CONFIG per (J group, panel)",
          tails);
}

void test_exec_determinism_and_mutations() {
    exec_case a, b;
    a.has_residual = b.has_residual = true;
    a.residual = b.residual = synth(32, 80, dense_runs(64, 3));
    bool ok = build_case(a, 80, 160, 2048, 80) && build_case(b, 80, 160, 2048, 80) && exec_invariants(a);
    bool same = ok && a.exec.ops.size() == b.exec.ops.size() && a.exec.index_entries.size() == b.exec.index_entries.size();
    for (size_t i = 0; same && i < a.exec.ops.size(); ++i) {
        same = potal_paired_op_text(a.exec.ops[i]) == potal_paired_op_text(b.exec.ops[i]);
    }
    for (size_t i = 0; same && i < a.exec.index_entries.size(); ++i) {
        same = a.exec.index_entries[i].slot == b.exec.index_entries[i].slot &&
               a.exec.index_entries[i].resident_k == b.exec.index_entries[i].resident_k &&
               a.exec.index_entries[i].filler == b.exec.index_entries[i].filler;
    }
    check("X15 the same tile plan and packet always give the same operations, entries and dependencies", same);

    // X16: mutations of an accepted plan must be refused, each with a diagnostic.
    const auto refused = [&](const char * name, const std::function<void(potal_paired_execution_plan &)> & mutate) {
        potal_paired_execution_plan m = a.exec;
        mutate(m);
        std::string               why;
        const potal_paired_status st = potal_validate_execution_plan(m, &a.residual, &why);
        std::printf("  X16 %s: %s %s\n", name, potal_paired_status_name(st), why.c_str());
        return st == potal_paired_status::invalid_execution_plan && !why.empty();
    };
    const auto first_of = [&](potal_paired_execution_plan & m, op_kind kind, size_t residency, size_t window = potal_paired_no_index) -> potal_paired_op & {
        for (potal_paired_op & op : m.ops) {
            if (op.kind == kind && op.residency == residency && (window == potal_paired_no_index || op.window == window)) {
                return op;
            }
        }
        return m.ops[0];
    };
    bool all = ok;
    all = all && refused("pair of residency 2 changed to pair 1 (still held by residency 1)", [&](potal_paired_execution_plan & m) {
        for (potal_paired_op & op : m.ops) {
            if (op.residency == 2) {
                op.pair = 1;
            }
        }
    });
    all = all && refused("weight load before the previous residency's retire (dependency dropped)", [&](potal_paired_execution_plan & m) {
        first_of(m, op_kind::load_weight, 2).deps.clear();
    });
    all = all && refused("retire without the completion of a Residual consumer", [&](potal_paired_execution_plan & m) {
        potal_paired_op & ret = m.ops[m.residencies[0].retire_op];
        ret.deps.resize(1); // keeps the Main dependency only
    });
    all = all && refused("Residual A placed over Main A", [&](potal_paired_execution_plan & m) {
        first_of(m, op_kind::load_residual_a, 0, 0).a_sp_begin = 0;
    });
    all = all && refused("weight region not rear-placed", [&](potal_paired_execution_plan & m) {
        first_of(m, op_kind::load_weight, 0).b_sp_begin -= PD;
    });
    all = all && refused("Index buffer of the other pair", [&](potal_paired_execution_plan & m) {
        first_of(m, op_kind::index_program, 0, 0).index_buffer = 1;
    });
    all = all && refused("Index entry with a resident K outside the panel", [&](potal_paired_execution_plan & m) {
        const potal_paired_op & ip = first_of(m, op_kind::index_program, 0, 0);
        m.index_entries[ip.entry_begin].resident_k += m.tile.panels[0].k_rows;
        m.index_entries[ip.entry_begin].original_k += m.tile.panels[0].k_rows;
    });
    all = all && refused("Index entry whose original K is not the packet's", [&](potal_paired_execution_plan & m) {
        const potal_paired_op & ip = first_of(m, op_kind::index_program, 0, 0);
        m.index_entries[ip.entry_begin].original_k += 1;
        m.index_entries[ip.entry_begin].resident_k += 1;
    });
    all = all && refused("Index entries out of slot order", [&](potal_paired_execution_plan & m) {
        const potal_paired_op & ip = first_of(m, op_kind::index_program, 0, 0);
        std::swap(m.index_entries[ip.entry_begin], m.index_entries[ip.entry_begin + 1]);
    });
    all = all && refused("active rows beyond the window", [&](potal_paired_execution_plan & m) {
        first_of(m, op_kind::index_program, 0, 0).index_active_rows += 1;
    });
    all = all && refused("same-buffer Index update without its predecessor dependency", [&](potal_paired_execution_plan & m) {
        potal_paired_op & ip = first_of(m, op_kind::index_program, 0, 1);
        ip.same_buffer_predecessor = potal_paired_no_index;
    });
    all = all && refused("Index Table reprogrammed before the window's gemm (ops swapped)", [&](potal_paired_execution_plan & m) {
        potal_paired_op & g  = first_of(m, op_kind::residual_gemm, 0, 0);
        potal_paired_op & ip = first_of(m, op_kind::index_program, 0, 1);
        std::swap(g.id, ip.id);
        std::swap(m.ops[g.id], m.ops[ip.id]);
    });
    all = all && refused("Main scale window overlapping a live Residual window", [&](potal_paired_execution_plan & m) {
        for (potal_paired_scale_window & w : m.scale_windows) {
            if (w.residual) {
                w.slot_base = 0;
            }
        }
    });
    all = all && refused("scale release before its consumer completed", [&](potal_paired_execution_plan & m) {
        potal_paired_op & rel = m.ops[m.scale_windows[0].release_op];
        rel.deps[0].complete = false;
    });
    all = all && refused("ACC1 first flag dropped", [&](potal_paired_execution_plan & m) {
        first_of(m, op_kind::residual_gemm, 0, 0).first = false;
    });
    all = all && refused("ACC0 last flag on an inner panel", [&](potal_paired_execution_plan & m) {
        first_of(m, op_kind::main_gemm, 0).last = true;
    });
    all = all && refused("ACC1 completion without reconstruction", [&](potal_paired_execution_plan & m) {
        for (potal_paired_op & op : m.ops) {
            if (op.kind == op_kind::acc_complete && op.acc_bank == 1) {
                op.reconstruction_required = false;
            }
        }
    });
    all = all && refused("duplicate Main contribution (second Main GEMM of residency 0)", [&](potal_paired_execution_plan & m) {
        first_of(m, op_kind::main_gemm, 1) = first_of(m, op_kind::main_gemm, 0);
        for (size_t i = 0; i < m.ops.size(); ++i) {
            m.ops[i].id = i;
        }
    });
    all = all && refused("missing Residual contribution (window turned into a release)", [&](potal_paired_execution_plan & m) {
        first_of(m, op_kind::residual_gemm, 0, 0).kind = op_kind::scale_release;
    });
    all = all && refused("work IDs beyond WorkIdsPerLoop", [&](potal_paired_execution_plan & m) {
        first_of(m, op_kind::main_gemm, 0).work_ids = POTAL_WORK_IDS_PER_LOOP + 1;
    });
    all = all && refused("K panels out of order (residency 1 before residency 0)", [&](potal_paired_execution_plan & m) {
        for (potal_paired_op & op : m.ops) {
            if (op.residency < 2 && (op.kind != op_kind::pair_retire)) {
                op.residency ^= 1;
                op.panel ^= 1;
                op.pair ^= 1;
            }
        }
    });
    check("X16 every mutated plan is refused by the validator with a diagnostic", all);

    // X17: every logical contribution exactly once (independent count over ops).
    std::vector<size_t> main_seen(a.tile.j_groups * a.tile.k_panels, 0);
    std::vector<std::vector<size_t>> win_seen(a.tile.j_groups * a.tile.k_panels);
    for (size_t r = 0; r < win_seen.size(); ++r) {
        win_seen[r].assign(a.tile.panels[r % a.tile.k_panels].windows.size(), 0);
    }
    for (const potal_paired_op & op : a.exec.ops) {
        if (op.kind == op_kind::main_gemm) {
            main_seen[op.residency]++;
        }
        if (op.kind == op_kind::residual_gemm) {
            win_seen[op.residency][op.window]++;
        }
    }
    bool once = ok;
    for (size_t r = 0; r < main_seen.size(); ++r) {
        once = once && main_seen[r] == 1;
        for (size_t w : win_seen[r]) {
            once = once && w == 1;
        }
    }
    check("X17 every Main panel and every Residual window of every J group contributes exactly once", once);
}

void test_exec_real_packets() {
    paired_fixture f(48, 64, 128, 23, [](size_t blk, size_t n) { return (int16_t) ((blk * 3 + n) % 5); });
    potal_tile_plan main;
    bool            ok = f.ok && main_select(f.M, f.N, f.K, main);
    std::vector<res_entry> entries;
    for (size_t row = 1; row < f.M; row += 4) {
        for (size_t k = row % 5; k < f.K; k += 3 + row % 3) {
            entries.push_back({ row, k, (int32_t) ((int) (row * 53 + k * 29) % 1201 - 600) });
        }
    }
    const grmd::StripePacketHandle packet = ok ? make_packet(0, 0, f.M, f.K, f.N, entries) : nullptr;
    residual::decoded_packet       d;
    paired_residual_request        req;
    ok = ok && packet != nullptr && residual::decode_packet(*packet, f.weight, d) == grmd::RmdStatus::success &&
         build_paired_residual_request(*packet, f.weight, main.tile_I * PD, req) == grmd::RmdStatus::success;
    exec_case c;
    c.has_residual = true;
    c.residual     = d;
    c.main         = main;
    std::string why;
    ok = ok && potal_make_tile_plan(main, f.M, f.N, f.K, &d, c.tile, &why) == potal_paired_status::ok &&
         potal_make_execution_plan(c.tile, &d, c.exec, &why) == potal_paired_status::ok && exec_invariants(c);
    print_exec("XR", c.exec, 0);
    check("XR a real RMD packet gives an accepted execution plan (" + why + ")", ok);
    if (!ok) {
        return;
    }
    // Entries of a J group, concatenated over windows, visit the paired descriptor's fragments in order and read
    // the same original K; rows keep the descriptor's (lane, source row) order.
    bool same_rows = c.exec.residual_rows.size() == req.rows.size();
    for (size_t i = 0; same_rows && i < req.rows.size(); ++i) {
        same_rows = c.exec.residual_rows[i].lane == req.rows[i].lane && c.exec.residual_rows[i].source_row == req.rows[i].source_row;
    }
    bool   same_k = c.tile.fragments.size() == req.fragments.size() && c.tile.residual_zero_fragments == 0;
    size_t next_f = 0, x = 0;
    for (const potal_paired_op & op : c.exec.ops) {
        if (op.kind != op_kind::index_program || op.j_group != 0) {
            continue;
        }
        for (size_t e = op.entry_begin; e < op.entry_end; ++e) {
            const potal_paired_index_entry & en = c.exec.index_entries[e];
            if (en.filler) {
                continue;
            }
            same_k = same_k && en.fragment == next_f && en.original_k == req.original_k(req.fragments[next_f], x) &&
                     en.original_k == c.tile.panels[op.panel].k_begin + en.resident_k;
            if (++x == req.fragments[next_f].count) {
                next_f++;
                x = 0;
            }
        }
    }
    check("XR entries visit the paired descriptor's fragments in order with the same original K; residual rows keep their provenance",
          same_rows && same_k && next_f == req.fragments.size());
}

} // namespace

int main() {
    test_geometry();
    test_main_tiles_kept();
    test_k_limits();
    test_occupancy_and_offset();
    test_panels_and_windows();
    test_real_packets();
    test_failures();
    test_exec_single_panel();
    test_exec_multi_panel();
    test_exec_windows();
    test_exec_pairs();
    test_exec_zero_fragments();
    test_exec_scale();
    test_exec_tails();
    test_exec_determinism_and_mutations();
    test_exec_real_packets();
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
