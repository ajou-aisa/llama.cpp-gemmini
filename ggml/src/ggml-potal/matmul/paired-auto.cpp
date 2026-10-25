// H3-0 planner of the paired Main/Residual GEMM: Main-derived I/J, the final
// common K, every scratchpad / ACC / Index / scale capacity, the K panels and
// their Index Windows (paired-auto.hpp). No RoCC command, no hardware access.

#include "paired-auto.hpp"

#include "quants/common/hp1_scu.hpp"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

namespace ggml::potal {

namespace {

// Overflow-safe size_t arithmetic: false when the result does not fit.
bool mul(size_t a, size_t b, size_t & out) {
    return !__builtin_mul_overflow(a, b, &out);
}

bool add(size_t a, size_t b, size_t & out) {
    return !__builtin_add_overflow(a, b, &out);
}

size_t tiles_of(size_t elements, size_t dim) {
    return elements / dim + (elements % dim != 0 ? 1 : 0);
}

// Largest multiple of `step` not above `value`.
size_t floor_to(size_t value, size_t step) {
    return value / step * step;
}

std::string num(size_t v) {
    return std::to_string((unsigned long long) v);
}

void explain(std::string * why, std::string text) {
    if (why != nullptr) {
        *why = std::move(text);
    }
}

constexpr size_t loop_ws_field_max = 65535; // 16-bit LOOP_WS tile counts

} // namespace

potal_paired_geometry potal_paired_hw_geometry() {
    potal_paired_geometry g;
    g.dim           = POTAL_HW_DIM;
    g.bank_num      = POTAL_HW_BANK_NUM;
    g.bank_rows     = POTAL_HW_BANK_ROWS;
    g.acc_rows      = POTAL_HW_ACC_ROWS;
    g.index_entries = POTAL_HW_INDEX_ENTRIES;
    g.scale_entries = POTAL_HW_SCALE_ENTRIES;
    g.work_ids      = POTAL_WORK_IDS_PER_LOOP;
    g.block_k       = residual::rmd::kBlockSize;
    return g;
}

potal_paired_pair_layout potal_paired_pair(size_t pair, const potal_paired_geometry & hw) {
    potal_paired_pair_layout l;
    l.pair             = pair;
    l.activation_bank  = 2 * pair;
    l.weight_bank      = 2 * pair + 1;
    l.activation_begin = l.activation_bank * hw.bank_rows;
    l.activation_end   = l.activation_begin + hw.bank_rows;
    l.weight_begin     = l.weight_bank * hw.bank_rows;
    l.weight_end       = l.weight_begin + hw.bank_rows;
    l.index_buffer     = pair;
    l.spad_id          = pair + 1;
    return l;
}

const char * potal_paired_status_name(potal_paired_status s) {
    switch (s) {
        case potal_paired_status::ok: return "OK";
        case potal_paired_status::invalid_argument: return "INVALID_ARGUMENT";
        case potal_paired_status::allocation_failure: return "ALLOCATION_FAILURE";
        case potal_paired_status::main_acc_capacity: return "MAIN_ACC_CAPACITY";
        case potal_paired_status::index_window_capacity: return "INDEX_WINDOW_CAPACITY";
        case potal_paired_status::residual_acc_capacity: return "RESIDUAL_ACC_CAPACITY";
        case potal_paired_status::weight_bank_capacity: return "WEIGHT_BANK_CAPACITY";
        case potal_paired_status::activation_bank_capacity: return "ACTIVATION_BANK_CAPACITY";
        case potal_paired_status::scale_window_capacity: return "SCALE_WINDOW_CAPACITY";
        case potal_paired_status::invalid_execution_plan: return "INVALID_EXECUTION_PLAN";
        case potal_paired_status::execution_failed: return "EXECUTION_FAILED";
    }
    return "?";
}

potal_paired_status potal_paired_fragments(const residual::decoded_packet & residual, size_t rows, size_t k,
                                           const potal_paired_geometry & hw, std::vector<potal_paired_fragment> & out,
                                           std::string * why, size_t * skipped) try {
    out.clear();
    if (skipped != nullptr) {
        *skipped = 0;
    }
    const size_t D = hw.dim, BK = hw.block_k;
    if (D == 0 || BK == 0 || BK % D != 0 || rows == 0 || k == 0 || k % BK != 0) {
        explain(why, "residual geometry: DIM, scale block, rows or K invalid");
        return potal_paired_status::invalid_argument;
    }
    if (residual.runs.empty() != residual.rows.empty()) {
        explain(why, "residual geometry: runs without rows or rows without runs");
        return potal_paired_status::invalid_argument;
    }
    const size_t blocks = k / BK;
    if (residual.runs.size() > blocks || residual.runs.size() > UINT32_MAX) {
        explain(why, "residual geometry: more runs than weight blocks");
        return potal_paired_status::invalid_argument;
    }

    // Runs: block ids ascending below K / block_k, block-local K ascending below block_k.
    std::vector<potal_paired_fragment> fragments;
    size_t                             compact = 0;
    for (size_t ri = 0; ri < residual.runs.size(); ++ri) {
        const residual::run & r = residual.runs[ri];
        if (r.local_k.empty() || (ri != 0 && r.block_id <= residual.runs[ri - 1].block_id) || r.block_id >= blocks) {
            explain(why, "residual geometry: run " + num(ri) + " empty, out of order or beyond K");
            return potal_paired_status::invalid_argument;
        }
        for (size_t i = 0; i < r.local_k.size(); ++i) {
            if (r.local_k[i] >= BK || (i != 0 && r.local_k[i] <= r.local_k[i - 1])) {
                explain(why, "residual geometry: run " + num(ri) + " local K out of order or beyond the block");
                return potal_paired_status::invalid_argument;
            }
        }
        const size_t c = r.local_k.size();
        if (compact > UINT32_MAX - c) {
            explain(why, "residual geometry: compact K overflow");
            return potal_paired_status::invalid_argument;
        }
        for (size_t f = 0; f < c; f += D) {
            potal_paired_fragment fr;
            fr.run           = (uint32_t) ri;
            fr.block_id      = r.block_id;
            fr.compact_begin = (uint32_t) (compact + f);
            fr.local_begin   = (uint32_t) f;
            fr.count         = (uint32_t) std::min(D, c - f);
            fragments.push_back(fr);
        }
        compact += c;
    }
    if (compact != residual.k_compact) {
        explain(why, "residual geometry: k_compact " + num(residual.k_compact) + " != sum of run K " + num(compact));
        return potal_paired_status::invalid_argument;
    }

    // Rows: strictly ascending (lane, source row), source rows inside the stripe.
    for (size_t i = 0; i < residual.rows.size(); ++i) {
        const residual::lane_row & row = residual.rows[i];
        if (row.source_row >= rows) {
            explain(why, "residual geometry: lane row " + num(i) + " beyond the stripe");
            return potal_paired_status::invalid_argument;
        }
        if (i != 0) {
            const residual::lane_row & prev = residual.rows[i - 1];
            if (row.lane < prev.lane || (row.lane == prev.lane && row.source_row <= prev.source_row)) {
                explain(why, "residual geometry: lane rows not ascending at row " + num(i));
                return potal_paired_status::invalid_argument;
            }
        }
    }
    size_t digits = 0;
    if (!residual.digits.empty() &&
        (!mul(residual.rows.size(), residual.k_compact, digits) || digits != residual.digits.size())) {
        explain(why, "residual geometry: digits do not cover rows x k_compact");
        return potal_paired_status::invalid_argument;
    }
    // With the digits known, a fragment that is zero in every lane row contributes nothing: leave it out.
    if (!residual.digits.empty()) {
        std::vector<potal_paired_fragment> kept;
        for (const potal_paired_fragment & f : fragments) {
            bool nonzero = false;
            for (size_t row = 0; row < residual.rows.size() && !nonzero; ++row) {
                const int8_t * d = residual.digits.data() + row * residual.k_compact + f.compact_begin;
                for (uint32_t x = 0; x < f.count; ++x) {
                    if (d[x] != 0) {
                        nonzero = true;
                        break;
                    }
                }
            }
            if (nonzero) {
                kept.push_back(f);
            } else if (skipped != nullptr) {
                ++*skipped;
            }
        }
        fragments = std::move(kept);
    }
    out = std::move(fragments);
    return potal_paired_status::ok;
} catch (const std::bad_alloc &) {
    explain(why, "out of memory");
    return potal_paired_status::allocation_failure;
}

uint32_t potal_paired_fragment_original_k(const residual::decoded_packet & residual, const potal_paired_fragment & f,
                                          size_t x, const potal_paired_geometry & hw) {
    return f.block_id * (uint32_t) hw.block_k + residual.runs[f.run].local_k[f.local_begin + x];
}

bool potal_paired_resident_k(const potal_paired_k_panel & panel, size_t original_k, size_t * resident_k) {
    if (original_k < panel.k_begin || original_k - panel.k_begin >= panel.k_rows) {
        return false;
    }
    if (resident_k != nullptr) {
        *resident_k = original_k - panel.k_begin;
    }
    return true;
}

potal_paired_status potal_make_tile_plan(const potal_tile_plan & main, size_t rows, size_t n, size_t k,
                                         const residual::decoded_packet * residual, potal_paired_tile_plan & plan,
                                         std::string * why, const potal_paired_geometry & hw) try {
    plan = potal_paired_tile_plan{};
    if (why != nullptr) {
        why->clear();
    }
    const auto refuse = [&](potal_paired_status s, std::string text) {
        explain(why, std::move(text));
        return s;
    };
    const auto overflow = [&]() { return refuse(potal_paired_status::invalid_argument, "size arithmetic overflow"); };

    // 0. Geometry and request.
    const size_t D = hw.dim;
    if (D == 0 || hw.bank_num < 2 * potal_paired_pairs || hw.bank_rows == 0 || hw.acc_rows < 2 ||
        hw.scale_entries == 0 || hw.work_ids == 0 || hw.block_k == 0 || hw.block_k % D != 0) {
        return refuse(potal_paired_status::invalid_argument, "hardware geometry: DIM, banks, ACC, scale cache or scale block");
    }
    const uint32_t ppb = (uint32_t) (hw.block_k / D);
    if (rows == 0 || n == 0 || k == 0 || k % hw.block_k != 0) {
        return refuse(potal_paired_status::invalid_argument,
                      "shape: rows, n and k must be positive and k a multiple of the scale block " + num(hw.block_k));
    }
    const size_t i_tiles = tiles_of(rows, D), j_tiles = tiles_of(n, D), k_tiles = k / D;
    if (main.tile_I == 0 || main.tile_J == 0 || main.tile_K == 0 || main.passes_per_block != ppb ||
        main.tile_K % ppb != 0 || main.tile_J > j_tiles || main.tile_K > k_tiles || main.tile_I > loop_ws_field_max ||
        main.tile_J > loop_ws_field_max || main.tile_K > loop_ws_field_max) {
        return refuse(potal_paired_status::invalid_argument,
                      "Main plan: tile_I " + num(main.tile_I) + " tile_J " + num(main.tile_J) + " tile_K " +
                          num(main.tile_K) + " passes_per_block " + num(main.passes_per_block) +
                          " does not describe this shape");
    }
    plan.rows             = rows;
    plan.n                = n;
    plan.k                = k;
    plan.passes_per_block = ppb;
    plan.i_tiles          = i_tiles;
    plan.j_tiles          = j_tiles;
    plan.k_tiles          = k_tiles;
    if (main.tile_I < i_tiles) {
        return refuse(potal_paired_status::invalid_argument,
                      "rows " + num(rows) + " need " + num(i_tiles) + " I tiles; the Main plan's tile_I " +
                          num(main.tile_I) + " covers one I group of " + num(main.tile_I * D) +
                          " rows (one paired call executes one stripe: no I loop)");
    }
    plan.tile_I           = std::min(main.tile_I, i_tiles); // a tail stripe needs fewer I tiles
    plan.tile_J           = main.tile_J;
    plan.tile_K_main      = main.tile_K;
    const size_t tile_I = plan.tile_I, tile_J = plan.tile_J;

    // 1. Main output tiles: ACC0 and WorkIdsPerLoop (the Main tiler's limits, against hw).
    size_t main_out_tiles = 0;
    if (!mul(tile_I, tile_J, main_out_tiles) || !mul(main_out_tiles, D, plan.main_acc_rows)) {
        return overflow();
    }
    if (plan.main_acc_rows > hw.acc_rows / 2 || main_out_tiles > hw.work_ids) {
        return refuse(potal_paired_status::main_acc_capacity,
                      "Main ACC0: " + num(tile_I) + " x " + num(tile_J) + " tiles = " + num(plan.main_acc_rows) +
                          " rows, ACC0 " + num(hw.acc_rows / 2) + " rows, work IDs " + num(hw.work_ids));
    }

    // 2. Residual geometry: lane rows, fragments, the largest Index Window.
    const bool has_residual = residual != nullptr && !(residual->rows.empty() && residual->runs.empty());
    if (has_residual) {
        const potal_paired_status fs =
            potal_paired_fragments(*residual, rows, k, hw, plan.fragments, why, &plan.residual_zero_fragments);
        if (fs != potal_paired_status::ok) {
            return fs;
        }
        if (hw.index_entries < D) {
            return refuse(potal_paired_status::index_window_capacity,
                          "Index Table: " + num(hw.index_entries) + " entries hold no DIM " + num(D) + " fragment");
        }
        if (plan.fragments.empty() && !residual->rows.empty()) {
            return refuse(potal_paired_status::invalid_argument,
                          "residual geometry: " + num(residual->rows.size()) + " lane rows but every fragment is zero");
        }
        plan.residual             = true;
        plan.residual_lane_rows   = residual->rows.size();
        plan.residual_row_tiles   = tiles_of(plan.residual_lane_rows, D);
        plan.residual_pad_I       = plan.residual_row_tiles * D - plan.residual_lane_rows;
        plan.residual_k_compact   = residual->k_compact;
        plan.fragments_per_window = std::min(hw.index_entries / D, plan.fragments.size());
        size_t res_out_tiles      = 0;
        if (!mul(plan.residual_row_tiles, tile_J, res_out_tiles) || !mul(res_out_tiles, D, plan.residual_acc_rows)) {
            return overflow();
        }
        if (plan.residual_acc_rows > hw.acc_rows / 2 || res_out_tiles > hw.work_ids ||
            plan.residual_row_tiles > loop_ws_field_max) {
            return refuse(potal_paired_status::residual_acc_capacity,
                          "Residual ACC1: " + num(plan.residual_lane_rows) + " lane rows = " +
                              num(plan.residual_row_tiles) + " row tiles x " + num(tile_J) + " J tiles = " +
                              num(plan.residual_acc_rows) + " rows, ACC1 " + num(hw.acc_rows / 2) +
                              " rows, work IDs " + num(hw.work_ids) + " (Main tile_I " + num(tile_I) + " is kept)");
        }
        size_t window_tiles = 0;
        if (!mul(plan.residual_row_tiles, plan.fragments_per_window, window_tiles) ||
            !mul(window_tiles, D, plan.residual_a_rows)) {
            return overflow();
        }
        size_t res_slots = 0;
        if (!mul(plan.fragments_per_window, tile_J, res_slots) || res_slots > UINT32_MAX) {
            return overflow();
        }
        plan.scale_slots_residual = (uint32_t) res_slots;
    }

    // 3. K limits, in DIM tiles of whole scale blocks.
    size_t weight_tile_rows = 0, main_tile_rows = 0;
    if (!mul(tile_J, D, weight_tile_rows) || !mul(tile_I, D, main_tile_rows)) {
        return overflow();
    }
    const auto k_limit = [&](size_t capacity_rows, size_t rows_per_k_tile) {
        return floor_to(capacity_rows / rows_per_k_tile, ppb);
    };
    const auto scale_limit = [&](size_t reserved_slots) -> size_t {
        if (reserved_slots > hw.scale_entries) {
            return 0;
        }
        size_t tiles = 0;
        return mul((hw.scale_entries - reserved_slots) / tile_J, ppb, tiles) ? tiles : loop_ws_field_max;
    };
    const size_t limit_weight       = k_limit(hw.bank_rows, weight_tile_rows);
    const size_t limit_act_main     = k_limit(hw.bank_rows, main_tile_rows);
    const size_t limit_act_residual = k_limit(hw.bank_rows > plan.residual_a_rows ? hw.bank_rows - plan.residual_a_rows : 0,
                                              main_tile_rows);
    const size_t limit_scale_main     = scale_limit(0);
    const size_t limit_scale_residual = scale_limit(plan.scale_slots_residual);
    plan.tile_K_weight_limit          = limit_weight;
    plan.tile_K_activation_limit      = limit_act_residual;
    plan.tile_K_scale_limit           = limit_scale_residual;
    plan.tile_K_hardware_limit        = std::min({ limit_weight, limit_act_main, limit_scale_main });
    plan.tile_K_residual_limit        = std::min({ limit_weight, limit_act_residual, limit_scale_residual });
    plan.tile_K = std::min({ plan.tile_K_main, plan.tile_K_residual_limit, plan.tile_K_hardware_limit });
    if (plan.tile_K == 0) {
        const std::string limits = "tile_I " + num(tile_I) + " tile_J " + num(tile_J) + ": K limits weight " +
                                   num(limit_weight) + " activation " + num(limit_act_residual) + " (Residual A " +
                                   num(plan.residual_a_rows) + " of " + num(hw.bank_rows) + " rows) scale " +
                                   num(limit_scale_residual) + " (Residual slots " + num(plan.scale_slots_residual) +
                                   " of " + num(hw.scale_entries) + ") tiles; one scale block = " + num(ppb) + " tiles";
        if (limit_weight == 0) {
            return refuse(potal_paired_status::weight_bank_capacity, "weight bank: " + limits);
        }
        if (limit_act_residual == 0) {
            return refuse(potal_paired_status::activation_bank_capacity, "activation bank: " + limits);
        }
        return refuse(potal_paired_status::scale_window_capacity, "scale cache: " + limits);
    }

    // 4. The Main execution plan with the final K (chunk counts, padding) through the
    //    existing frozen-plan path; its own limits are the built hardware's.
    const potal_status frozen = potal_frozen_plan(rows, n, k, tile_I, tile_J, plan.tile_K, ppb, &plan.main);
    if (frozen != POTAL_OK) {
        return refuse(potal_paired_status::invalid_argument,
                      std::string("potal_frozen_plan refused tile_K ") + num(plan.tile_K) + ": POTAL_" +
                          potal_status_name(frozen));
    }
    plan.j_groups = plan.main.J0;
    plan.k_panels = plan.main.K0;
    plan.pad_I    = plan.main.pad_I;
    plan.pad_J    = plan.main.pad_J;

    // 5. Footprints of the largest panel.
    size_t main_a_tiles = 0, slots_main = 0, a_total = 0;
    if (!mul(tile_I, plan.tile_K, main_a_tiles) || !mul(main_a_tiles, D, plan.main_a_rows) ||
        !mul(plan.tile_K, weight_tile_rows, plan.weight_rows) || !mul(plan.tile_K / ppb, tile_J, slots_main) ||
        slots_main > UINT32_MAX || !add(plan.main_a_rows, plan.residual_a_rows, a_total)) {
        return overflow();
    }
    plan.residual_a_offset_tiles = main_a_tiles;
    plan.scale_slots_main        = (uint32_t) slots_main;
    if (a_total > hw.bank_rows || plan.weight_rows > hw.bank_rows || plan.scale_slots_main > hw.scale_entries ||
        (size_t) plan.scale_slots_main + plan.scale_slots_residual > hw.scale_entries) {
        return refuse(potal_paired_status::invalid_argument, "internal: footprint exceeds a limit it was derived from");
    }

    // 6. K panels and their Index Windows.
    plan.panels.reserve(plan.k_panels);
    size_t fi = 0; // next unassigned fragment
    for (size_t p = 0; p < plan.k_panels; ++p) {
        potal_paired_k_panel panel;
        panel.index       = p;
        panel.k_tiles     = std::min(plan.tile_K, k_tiles - p * plan.tile_K);
        panel.k_begin     = p * plan.tile_K * D;
        panel.k_rows      = panel.k_tiles * D;
        panel.block_begin = panel.k_begin / hw.block_k;
        panel.block_end   = (panel.k_begin + panel.k_rows) / hw.block_k;
        panel.weight_rows = panel.k_tiles * weight_tile_rows;
        panel.main_a_rows = panel.k_tiles * main_tile_rows;

        panel.fragment_begin = fi;
        while (fi < plan.fragments.size() && plan.fragments[fi].block_id < panel.block_end) {
            panel.residual_k_compact += plan.fragments[fi].count;
            ++fi;
        }
        panel.fragment_end = fi;

        for (size_t f0 = panel.fragment_begin; f0 < panel.fragment_end; f0 += plan.fragments_per_window) {
            potal_paired_index_window w;
            w.index          = panel.windows.size();
            w.fragment_begin = f0;
            w.fragment_end   = std::min(f0 + plan.fragments_per_window, panel.fragment_end);
            w.k_tiles        = w.fragment_end - w.fragment_begin;
            w.active_rows    = (w.k_tiles - 1) * D + plan.fragments[w.fragment_end - 1].count;
            w.pad_k          = w.k_tiles * D - w.active_rows;
            for (size_t f = w.fragment_begin; f + 1 < w.fragment_end; ++f) {
                w.filler_slots += D - plan.fragments[f].count;
            }
            w.a_rows = plan.residual_row_tiles * w.k_tiles * D;
            if (w.active_rows > hw.index_entries || w.a_rows > plan.residual_a_rows) {
                return refuse(potal_paired_status::invalid_argument, "internal: Index Window exceeds its reservation");
            }
            panel.residual_a_rows = std::max(panel.residual_a_rows, w.a_rows);
            plan.residual_filler_slots += w.filler_slots;
            panel.windows.push_back(w);
        }
        plan.residual_windows += panel.windows.size();
        if (plan.residual && panel.windows.empty()) {
            plan.residual_empty_panels++;
        }
        plan.panels.push_back(std::move(panel));
    }
    if (fi != plan.fragments.size()) {
        return refuse(potal_paired_status::invalid_argument, "internal: residual fragments beyond the last K panel");
    }
    return potal_paired_status::ok;
} catch (const std::bad_alloc &) {
    explain(why, "out of memory");
    return potal_paired_status::allocation_failure;
}

// ---- execution plan (H3-1) ---------------------------------------------------------------

const char * potal_paired_op_name(potal_paired_op_kind k) {
    switch (k) {
        case potal_paired_op_kind::load_main_a: return "LOAD_MAIN_A";
        case potal_paired_op_kind::load_weight: return "LOAD_WEIGHT";
        case potal_paired_op_kind::scale_program: return "SCALE_PROGRAM";
        case potal_paired_op_kind::main_gemm: return "MAIN_GEMM";
        case potal_paired_op_kind::load_residual_a: return "LOAD_RESIDUAL_A";
        case potal_paired_op_kind::index_program: return "INDEX_PROGRAM";
        case potal_paired_op_kind::residual_gemm: return "RESIDUAL_GEMM";
        case potal_paired_op_kind::scale_release: return "SCALE_RELEASE";
        case potal_paired_op_kind::acc_complete: return "ACC_COMPLETE";
        case potal_paired_op_kind::pair_retire: return "PAIR_RETIRE";
    }
    return "?";
}

const char * potal_paired_pair_phase_name(potal_paired_pair_phase p) {
    switch (p) {
        case potal_paired_pair_phase::free: return "FREE";
        case potal_paired_pair_phase::loading: return "LOADING";
        case potal_paired_pair_phase::resident: return "RESIDENT";
        case potal_paired_pair_phase::retired: return "RETIRED";
    }
    return "?";
}

std::string potal_paired_op_text(const potal_paired_op & op) {
    std::string t = "#" + num(op.id) + " " + potal_paired_op_name(op.kind) + " r" + num(op.residency) + " g" +
                    num(op.j_group) + " p" + num(op.panel) + " pair" + num(op.pair);
    if (op.window != potal_paired_no_index) {
        t += " w" + num(op.window);
    }
    switch (op.kind) {
        case potal_paired_op_kind::load_main_a:
        case potal_paired_op_kind::load_residual_a:
            t += " A sp[" + num(op.a_sp_begin) + "," + num(op.a_sp_begin + op.a_sp_rows) + ") offset " +
                 num(op.a_offset_tiles) + " tiles";
            break;
        case potal_paired_op_kind::load_weight:
            t += " B sp[" + num(op.b_sp_begin) + "," + num(op.b_sp_begin + op.b_sp_rows) + ") gen " +
                 num(op.weight_generation);
            break;
        case potal_paired_op_kind::main_gemm:
        case potal_paired_op_kind::residual_gemm:
            t += " " + num(op.i_tiles) + "x" + num(op.j_tiles) + "x" + num(op.k_tiles) + " pad " + num(op.pad_i) +
                 "/" + num(op.pad_j) + "/" + num(op.pad_k) + " ACC" + num(op.acc_bank) + " epoch " +
                 num(op.acc_epoch) + (op.first ? " first" : "") + (op.last ? " last" : "") + " scale w" +
                 num(op.scale_window) + " work " + num(op.work_ids);
            break;
        case potal_paired_op_kind::index_program:
            t += " buffer " + num(op.index_buffer) + " bBase " + num(op.index_b_base) + " jTiles " +
                 num(op.index_j_tiles) + " sourceKRows " + num(op.index_source_k_rows) + " active " +
                 num(op.index_active_rows) + " entries [" + num(op.entry_begin) + "," + num(op.entry_end) + ")";
            break;
        case potal_paired_op_kind::scale_program:
        case potal_paired_op_kind::scale_release: t += " scale w" + num(op.scale_window); break;
        case potal_paired_op_kind::acc_complete:
            t += " ACC" + num(op.acc_bank) + " epoch " + num(op.acc_epoch) +
                 (op.reconstruction_required ? " reconstruct" : "");
            break;
        case potal_paired_op_kind::pair_retire: break;
    }
    for (const potal_paired_dep & d : op.deps) {
        t += (d.complete ? " <complete #" : " <#") + num(d.op);
    }
    return t;
}

namespace {

constexpr size_t no_index = potal_paired_no_index;

struct plan_builder {
    const potal_paired_tile_plan &   tile;
    const residual::decoded_packet * residual;
    const potal_paired_geometry &    hw;
    potal_paired_execution_plan      plan;
    size_t                           next_generation = 1;
    size_t                           last_main_release = no_index, last_residual_release = no_index;
    size_t                           last_indexed_gemm[potal_paired_pairs] = { no_index, no_index };

    plan_builder(const potal_paired_tile_plan & t, const residual::decoded_packet * r, const potal_paired_geometry & g) :
        tile(t),
        residual(r),
        hw(g) {}

    size_t add(potal_paired_op op) {
        op.id = plan.ops.size();
        plan.ops.push_back(std::move(op));
        return plan.ops.back().id;
    }

    void dep(size_t op, size_t on, bool complete) {
        if (on == no_index) {
            return;
        }
        plan.ops[op].deps.push_back({ on, complete });
        plan.complete_deps += complete ? 1 : 0;
    }

    void transition(size_t op, size_t pair, potal_paired_pair_phase to, size_t residency) {
        potal_paired_pair_state & st = plan.pairs[pair];
        plan.transitions.push_back({ op, pair, st.phase, to, residency });
        st.phase     = to;
        st.residency = residency;
    }

    potal_paired_op base(potal_paired_op_kind kind, const potal_paired_residency & r) {
        potal_paired_op op;
        op.kind              = kind;
        op.j_group           = r.j_group;
        op.panel             = r.panel;
        op.residency         = r.index;
        op.pair              = r.pair;
        op.weight_generation = r.index;
        return op;
    }

    size_t scale_window(bool residual_window, size_t residency, size_t window, size_t base, size_t blocks, size_t j_tiles) {
        potal_paired_scale_window w;
        w.index      = plan.scale_windows.size();
        w.residual   = residual_window;
        w.residency  = residency;
        w.window     = window;
        w.slot_base  = base;
        w.blocks     = blocks;
        w.j_tiles    = j_tiles;
        w.slots      = blocks * j_tiles;
        w.generation = next_generation++;
        plan.scale_windows.push_back(w);
        return w.index;
    }

    // Retire residency r: its pair may take another panel once every consumer completed.
    void retire(potal_paired_residency & r) {
        potal_paired_op op = base(potal_paired_op_kind::pair_retire, r);
        const size_t    id = add(op);
        dep(id, r.main_gemm_op, true);
        for (size_t o = r.first_residual_op; r.first_residual_op != no_index && o <= r.last_residual_op; ++o) {
            if (plan.ops[o].kind == potal_paired_op_kind::residual_gemm && plan.ops[o].residency == r.index) {
                dep(id, o, true);
            }
        }
        r.retire_op = id;
        plan.pair_retires++;
        transition(id, r.pair, potal_paired_pair_phase::retired, r.index);
        potal_paired_pair_state & st = plan.pairs[r.pair];
        st.weight_valid = st.main_a_valid = st.residual_a_valid = false;
        st.pending_main = st.pending_residual = 0;
    }
};

} // namespace

potal_paired_status potal_make_execution_plan(const potal_paired_tile_plan & tile,
                                              const residual::decoded_packet * residual,
                                              potal_paired_execution_plan & out, std::string * why,
                                              const potal_paired_geometry & hw) try {
    if (why != nullptr) {
        why->clear();
    }
    const auto refuse = [&](potal_paired_status s, std::string text) {
        explain(why, std::move(text));
        return s;
    };
    const size_t D = hw.dim;
    if (D == 0 || tile.tile_K == 0 || tile.tile_I == 0 || tile.tile_J == 0 || tile.passes_per_block == 0 ||
        tile.k_panels == 0 || tile.j_groups == 0 || tile.panels.size() != tile.k_panels ||
        tile.main.K0 != tile.k_panels || tile.main.J0 != tile.j_groups || tile.main.tile_K != tile.tile_K ||
        tile.main.tile_I != tile.tile_I || tile.main.tile_J != tile.tile_J || hw.bank_num < 2 * potal_paired_pairs ||
        hw.block_k != tile.passes_per_block * D) {
        return refuse(potal_paired_status::invalid_argument, "tile plan incomplete or not of this geometry");
    }
    if (tile.residual) {
        if (residual == nullptr || residual->rows.size() != tile.residual_lane_rows ||
            residual->k_compact != tile.residual_k_compact || tile.residual_row_tiles == 0) {
            return refuse(potal_paired_status::invalid_argument, "tile plan has a residual the decoded packet does not match");
        }
        for (const potal_paired_fragment & f : tile.fragments) {
            if (f.run >= residual->runs.size() || f.local_begin + f.count > residual->runs[f.run].local_k.size() ||
                f.count == 0 || f.count > D) {
                return refuse(potal_paired_status::invalid_argument, "tile plan fragment outside the decoded packet");
            }
        }
    } else if (tile.residual_windows != 0 || !tile.fragments.empty()) {
        return refuse(potal_paired_status::invalid_argument, "tile plan without residual carries residual work");
    }

    plan_builder b(tile, residual, hw);
    b.plan.tile = tile;
    if (tile.residual) {
        b.plan.residual_rows = residual->rows;
    }
    for (size_t q = 0; q < potal_paired_pairs; ++q) {
        b.plan.pairs[q].pair = q;
    }

    // The ACC1 chain of every J group: first / last (panel, window) with residual work.
    size_t first_panel = no_index, first_window = no_index, last_panel = no_index, last_window = no_index;
    for (const potal_paired_k_panel & panel : tile.panels) {
        if (panel.windows.empty()) {
            continue;
        }
        if (first_panel == no_index) {
            first_panel  = panel.index;
            first_window = 0;
        }
        last_panel  = panel.index;
        last_window = panel.windows.size() - 1;
    }
    const size_t last_J = tile.j_tiles - (tile.j_groups - 1) * tile.tile_J;

    for (size_t g = 0; g < tile.j_groups; ++g) {
        const size_t j_tiles_g = g + 1 == tile.j_groups ? last_J : tile.tile_J;
        const size_t pad_j     = g + 1 == tile.j_groups ? tile.pad_J : 0;
        for (size_t p = 0; p < tile.k_panels; ++p) {
            const potal_paired_k_panel &   panel  = tile.panels[p];
            const size_t                   r      = g * tile.k_panels + p;
            const size_t                   q      = r % potal_paired_pairs;
            const potal_paired_pair_layout layout = potal_paired_pair(q, hw);
            potal_paired_residency         res;
            res.index   = r;
            res.j_group = g;
            res.panel   = p;
            res.pair    = q;
            res.j_tiles = j_tiles_g;
            res.cols    = j_tiles_g * D - pad_j;

            // 0. The pair's previous residency must be retired before its banks are rewritten.
            size_t retire_dep = no_index;
            if (r >= potal_paired_pairs) {
                potal_paired_residency & prev = b.plan.residencies[r - potal_paired_pairs];
                b.retire(prev);
                retire_dep = prev.retire_op;
            }

            // 1. Main scale window: slots [0, blocks * j_tiles), a new generation after the previous Main release.
            const size_t main_blocks = panel.k_tiles / tile.passes_per_block;
            const size_t sw_main     = b.scale_window(false, r, no_index, 0, main_blocks, j_tiles_g);
            {
                potal_paired_op op = b.base(potal_paired_op_kind::scale_program, res);
                op.scale_window    = sw_main;
                const size_t id    = b.add(op);
                b.dep(id, b.last_main_release, false);
                b.plan.scale_windows[sw_main].program_op = id;
                b.plan.scale_programs++;
            }

            // 2. Loads of the Main LOOP_WS: A from the bank start, B rear-placed in the weight bank.
            const size_t main_a_rows = tile.tile_I * panel.k_tiles * D;
            const size_t b_rows      = panel.k_tiles * j_tiles_g * D;
            const size_t b_begin     = layout.weight_end - b_rows;
            size_t       load_a = 0, load_b = 0;
            {
                potal_paired_op op = b.base(potal_paired_op_kind::load_main_a, res);
                op.i_tiles         = tile.tile_I;
                op.k_tiles         = panel.k_tiles;
                op.a_sp_begin      = layout.activation_begin;
                op.a_sp_rows       = main_a_rows;
                load_a             = b.add(op);
                b.dep(load_a, retire_dep, false);
                b.transition(load_a, q, potal_paired_pair_phase::loading, r);
                b.plan.pairs[q].main_a_valid = true;
            }
            {
                potal_paired_op op = b.base(potal_paired_op_kind::load_weight, res);
                op.j_tiles         = j_tiles_g;
                op.k_tiles         = panel.k_tiles;
                op.b_sp_begin      = b_begin;
                op.b_sp_rows       = b_rows;
                load_b             = b.add(op);
                b.dep(load_b, retire_dep, false);
                b.plan.pairs[q].weight_valid = true;
                b.plan.weight_loads++;
            }
            res.load_weight_op = load_b;

            // 3. Main GEMM into ACC0 (epoch = J group): first on panel 0, last on the last panel.
            {
                potal_paired_op op = b.base(potal_paired_op_kind::main_gemm, res);
                op.i_tiles         = tile.tile_I;
                op.j_tiles         = j_tiles_g;
                op.k_tiles         = panel.k_tiles;
                op.pad_i           = tile.pad_I;
                op.pad_j           = pad_j;
                op.pad_k           = 0;
                op.a_sp_begin      = layout.activation_begin;
                op.a_sp_rows       = main_a_rows;
                op.b_sp_begin      = b_begin;
                op.b_sp_rows       = b_rows;
                op.scale_window    = sw_main;
                op.acc_bank        = 0;
                op.acc_epoch       = g;
                op.work_ids        = tile.tile_I * j_tiles_g;
                op.first           = p == 0;
                op.last            = p + 1 == tile.k_panels;
                const size_t id    = b.add(op);
                b.dep(id, load_a, false);
                b.dep(id, load_b, false);
                b.dep(id, b.plan.scale_windows[sw_main].program_op, false);
                b.plan.scale_windows[sw_main].consumer_op = id;
                res.main_gemm_op                          = id;
                res.consumers++;
                b.plan.main_gemms++;
                b.transition(id, q, potal_paired_pair_phase::resident, r);
                b.plan.pairs[q].pending_main++;
                if (op.last) {
                    potal_paired_op done = b.base(potal_paired_op_kind::acc_complete, res);
                    done.acc_bank        = 0;
                    done.acc_epoch       = g;
                    const size_t did     = b.add(done);
                    b.dep(did, id, false);
                    b.plan.acc_completes++;
                }
            }

            // 4. Residual windows of the panel on the same resident B through the pair's Index Table.
            for (size_t w = 0; w < panel.windows.size(); ++w) {
                const potal_paired_index_window & win = panel.windows[w];
                // INDEX_CONFIG + INDEX_WRITE of the window: one transaction on the pair's buffer.
                size_t index_op = 0;
                {
                    potal_paired_op op          = b.base(potal_paired_op_kind::index_program, res);
                    op.window                   = w;
                    op.index_buffer             = layout.index_buffer;
                    op.index_b_base             = b_begin;
                    op.index_j_tiles            = j_tiles_g;
                    op.index_source_k_rows      = panel.k_rows;
                    op.index_active_rows        = win.active_rows;
                    op.b_sp_begin               = b_begin;
                    op.b_sp_rows                = b_rows;
                    op.entry_begin              = b.plan.index_entries.size();
                    op.same_buffer_predecessor  = b.last_indexed_gemm[q];
                    for (size_t f = win.fragment_begin; f < win.fragment_end; ++f) {
                        const potal_paired_fragment & fr   = tile.fragments[f];
                        const size_t                  slot0 = (f - win.fragment_begin) * D;
                        const size_t first_k = potal_paired_fragment_original_k(*residual, fr, 0, hw);
                        for (size_t x = 0; x < fr.count; ++x) {
                            potal_paired_index_entry e;
                            e.slot       = slot0 + x;
                            e.fragment   = f;
                            e.original_k = potal_paired_fragment_original_k(*residual, fr, x, hw);
                            e.resident_k = e.original_k - panel.k_begin;
                            e.compact_k  = fr.compact_begin + x;
                            b.plan.index_entries.push_back(e);
                        }
                        if (f + 1 < win.fragment_end) {
                            for (size_t x = fr.count; x < D; ++x) {
                                potal_paired_index_entry e; // filler: a resident row, zero digits
                                e.slot       = slot0 + x;
                                e.fragment   = f;
                                e.original_k = first_k;
                                e.resident_k = first_k - panel.k_begin;
                                e.filler     = true;
                                b.plan.index_entries.push_back(e);
                            }
                        }
                    }
                    op.entry_end = b.plan.index_entries.size();
                    index_op     = b.add(op);
                    b.dep(index_op, load_b, false);
                    b.dep(index_op, b.last_indexed_gemm[q], false); // same buffer: hardware-stalled while busy
                    b.plan.pairs[q].index_window = index_op;
                    b.plan.index_programs++;
                    b.plan.index_writes += op.entry_end - op.entry_begin;
                }
                // Residual scale window: slots behind the Main window, one block per fragment.
                const size_t sw_res = b.scale_window(true, r, w, tile.scale_slots_main, win.k_tiles, j_tiles_g);
                {
                    potal_paired_op op = b.base(potal_paired_op_kind::scale_program, res);
                    op.window          = w;
                    op.scale_window    = sw_res;
                    const size_t id    = b.add(op);
                    b.dep(id, b.last_residual_release, false);
                    b.plan.scale_windows[sw_res].program_op = id;
                    b.plan.scale_programs++;
                }
                // Packed digits of the window behind Main A; a later window reuses the region
                // only after the previous Indexed gemm of this pair completed (conservative).
                const size_t res_a_begin = layout.activation_begin + tile.residual_a_offset_tiles * D;
                size_t       load_r      = 0;
                {
                    potal_paired_op op = b.base(potal_paired_op_kind::load_residual_a, res);
                    op.window          = w;
                    op.i_tiles         = tile.residual_row_tiles;
                    op.k_tiles         = win.k_tiles;
                    op.a_offset_tiles  = tile.residual_a_offset_tiles;
                    op.a_sp_begin      = res_a_begin;
                    op.a_sp_rows       = win.a_rows;
                    load_r             = b.add(op);
                    b.dep(load_r, retire_dep, false);
                    b.dep(load_r, b.last_indexed_gemm[q], true);
                    b.plan.pairs[q].residual_a_valid = true;
                }
                {
                    potal_paired_op op = b.base(potal_paired_op_kind::residual_gemm, res);
                    op.window          = w;
                    op.i_tiles         = tile.residual_row_tiles;
                    op.j_tiles         = j_tiles_g;
                    op.k_tiles         = win.k_tiles;
                    op.pad_i           = tile.residual_pad_I;
                    op.pad_j           = pad_j;
                    op.pad_k           = win.pad_k;
                    op.a_offset_tiles  = tile.residual_a_offset_tiles;
                    op.a_sp_begin      = res_a_begin;
                    op.a_sp_rows       = win.a_rows;
                    op.b_sp_begin      = b_begin;
                    op.b_sp_rows       = b_rows;
                    op.index_buffer    = layout.index_buffer;
                    op.scale_window    = sw_res;
                    op.acc_bank        = 1;
                    op.acc_epoch       = g;
                    op.work_ids        = tile.residual_row_tiles * j_tiles_g;
                    op.first           = p == first_panel && w == first_window;
                    op.last            = p == last_panel && w == last_window;
                    const size_t id    = b.add(op);
                    b.dep(id, index_op, false);
                    b.dep(id, load_r, false);
                    b.dep(id, load_b, false);
                    b.dep(id, b.plan.scale_windows[sw_res].program_op, false);
                    b.plan.scale_windows[sw_res].consumer_op = id;
                    b.last_indexed_gemm[q]                   = id;
                    if (res.first_residual_op == no_index) {
                        res.first_residual_op = id;
                    }
                    res.last_residual_op = id;
                    res.windows++;
                    res.consumers++;
                    b.plan.residual_gemms++;
                    b.plan.pairs[q].pending_residual++;
                    if (op.last) {
                        potal_paired_op done         = b.base(potal_paired_op_kind::acc_complete, res);
                        done.window                  = w;
                        done.acc_bank                = 1;
                        done.acc_epoch               = g;
                        done.reconstruction_required = true;
                        const size_t did             = b.add(done);
                        b.dep(did, id, false);
                        b.plan.acc_completes++;
                    }
                    potal_paired_op rel = b.base(potal_paired_op_kind::scale_release, res);
                    rel.window          = w;
                    rel.scale_window    = sw_res;
                    const size_t rid    = b.add(rel);
                    b.dep(rid, id, true);
                    b.plan.scale_windows[sw_res].release_op = rid;
                    b.last_residual_release                 = rid;
                    b.plan.scale_releases++;
                }
            }
            // 5. The Main scale window is released after the panel's residual windows were
            //    issued: an accepted SCALE_RELEASE is a hardware drain point (releaseSafe ==
            //    fenceDrained), so releasing it earlier would serialize the Residual GEMMs
            //    behind Main instead of letting them share the stream (distinct slot ranges).
            {
                potal_paired_op rel = b.base(potal_paired_op_kind::scale_release, res);
                rel.scale_window    = sw_main;
                const size_t rid    = b.add(rel);
                b.dep(rid, res.main_gemm_op, true);
                b.plan.scale_windows[sw_main].release_op = rid;
                b.last_main_release                      = rid;
                b.plan.scale_releases++;
            }
            b.plan.pairs[q].residencies++;
            b.plan.residencies.push_back(res);
        }
    }
    // The last residencies of both pairs retire at the end.
    const size_t total = b.plan.residencies.size();
    for (size_t r = total >= potal_paired_pairs ? total - potal_paired_pairs : 0; r < total; ++r) {
        b.retire(b.plan.residencies[r]);
    }

    const potal_paired_status checked = potal_validate_execution_plan(b.plan, residual, why, hw);
    if (checked != potal_paired_status::ok) {
        return checked;
    }
    out = std::move(b.plan);
    return potal_paired_status::ok;
} catch (const std::bad_alloc &) {
    explain(why, "out of memory");
    return potal_paired_status::allocation_failure;
}

// ---- validator ---------------------------------------------------------------------------

namespace {

struct acc_identity_state {
    size_t contributions = 0;
    size_t last_op       = no_index; // the op flagged last
    size_t complete_op   = no_index;
    size_t final_op      = no_index; // the last contribution seen
};

struct validator {
    const potal_paired_execution_plan & plan;
    const residual::decoded_packet *    residual;
    const potal_paired_geometry &       hw;
    std::string *                       why;

    validator(const potal_paired_execution_plan & p, const residual::decoded_packet * r, const potal_paired_geometry & g,
              std::string * w) :
        plan(p),
        residual(r),
        hw(g),
        why(w) {}

    bool fail(const potal_paired_op & op, const std::string & rule) {
        explain(why, "op " + num(op.id) + " " + potal_paired_op_name(op.kind) + ": " + rule);
        return false;
    }

    bool fail(const std::string & rule) {
        explain(why, rule);
        return false;
    }

    static bool has_dep(const potal_paired_op & op, size_t on, bool complete) {
        for (const potal_paired_dep & d : op.deps) {
            if (d.op == on && (!complete || d.complete)) {
                return true;
            }
        }
        return false;
    }

    bool run() {
        const potal_paired_tile_plan & tile = plan.tile;
        const size_t                   D    = hw.dim;
        if (D == 0 || tile.tile_K == 0 || tile.k_panels == 0 || tile.j_groups == 0 ||
            tile.panels.size() != tile.k_panels) {
            return fail("tile plan incomplete");
        }
        if (tile.residual && (plan.residual_rows.size() != tile.residual_lane_rows ||
                              tile.residual_row_tiles * tile.tile_J * D > hw.acc_rows / 2 ||
                              tile.residual_row_tiles * tile.tile_J > hw.work_ids)) {
            return fail("residual rows do not fit ACC1 / WorkIdsPerLoop or lack provenance");
        }
        if (tile.main_acc_rows > hw.acc_rows / 2 || tile.tile_I * tile.tile_J > hw.work_ids) {
            return fail("Main tiles do not fit ACC0 / WorkIdsPerLoop");
        }
        if (tile.main.I0 != 1 || tile.tile_I * D < tile.rows || tile.rows == 0) {
            return fail("rows " + num(tile.rows) + " are not one Main I group of " + num(tile.tile_I) + " tiles");
        }
        if ((size_t) tile.scale_slots_main + tile.scale_slots_residual > hw.scale_entries) {
            return fail("scale windows exceed the cache");
        }
        const size_t total_residencies = tile.j_groups * tile.k_panels;
        const size_t last_J            = tile.j_tiles - (tile.j_groups - 1) * tile.tile_J;

        // Per-residency bookkeeping.
        struct residency_state {
            bool   seen = false, retired = false;
            size_t pair = 0, load_a = no_index, load_b = no_index, main = no_index, main_scale = no_index;
            size_t main_a_end = 0, b_begin = 0, b_rows = 0;
            std::vector<size_t> residual_gemms;
            size_t windows_seen = 0;
            size_t last_index_program = no_index;   // most recent index_program (consumed or not)
            bool   index_consumed = true;           // its residual_gemm was issued
            size_t last_residual_gemm = no_index;
        };
        std::vector<residency_state> rs(total_residencies);
        // Per pair: the residency currently holding it.
        size_t pair_owner[potal_paired_pairs];
        bool   pair_free[potal_paired_pairs];
        size_t buffer_last_gemm[potal_paired_pairs];
        for (size_t q = 0; q < potal_paired_pairs; ++q) {
            pair_owner[q]       = no_index;
            pair_free[q]        = true;
            buffer_last_gemm[q] = no_index;
        }
        // Scale windows.
        std::vector<int> window_state(plan.scale_windows.size(), 0); // 0 unprogrammed, 1 live, 2 released
        std::vector<size_t> window_consumer(plan.scale_windows.size(), no_index);
        size_t last_generation = 0;
        // ACC identities: bank 0 / 1 x epoch.
        std::vector<acc_identity_state> acc0(tile.j_groups), acc1(tile.j_groups);
        std::vector<uint8_t> main_seen(total_residencies, 0);
        size_t next_residency = 0;
        // Drain points as H3-2B realizes them: an accepted SCALE_RELEASE completes every
        // older command; a `complete` dependency not yet covered is a POTAL_FENCE there.
        // `complete_upto`: every op with a smaller id has completed.
        size_t              complete_upto = 0;
        std::vector<size_t> gemm_order; // ids of main / residual gemms in issue order

        for (size_t i = 0; i < plan.ops.size(); ++i) {
            const potal_paired_op & op = plan.ops[i];
            if (op.id != i) {
                return fail(op, "id out of sequence");
            }
            for (const potal_paired_dep & d : op.deps) {
                if (d.op >= i) {
                    return fail(op, "dependency on a later or own op");
                }
                if (d.complete && d.op >= complete_upto && op.kind != potal_paired_op_kind::scale_release) {
                    complete_upto = i; // fenced before this op
                }
            }
            if (op.kind == potal_paired_op_kind::main_gemm || op.kind == potal_paired_op_kind::residual_gemm) {
                // LoopMatmul has two loop slots: a LOOP_WS two positions later reuses the slot (and
                // its work IDs) of this one, so that one must have completed (RTL audit, Q8).
                if (gemm_order.size() >= 2 && gemm_order[gemm_order.size() - 2] >= complete_upto) {
                    return fail(op, "LOOP_WS reuses the LoopMatmul slot of LOOP_WS op " +
                                        num(gemm_order[gemm_order.size() - 2]) + " with no drain point in between");
                }
                gemm_order.push_back(i);
            }
            if (op.residency >= total_residencies || op.residency != op.j_group * tile.k_panels + op.panel ||
                op.pair != op.residency % potal_paired_pairs || op.panel >= tile.k_panels || op.j_group >= tile.j_groups) {
                return fail(op, "residency / pair / panel / J group identity");
            }
            residency_state &              st     = rs[op.residency];
            const potal_paired_k_panel &   panel  = tile.panels[op.panel];
            const potal_paired_pair_layout layout = potal_paired_pair(op.pair, hw);
            const size_t j_tiles_g = op.j_group + 1 == tile.j_groups ? last_J : tile.tile_J;
            const size_t pad_j     = op.j_group + 1 == tile.j_groups ? tile.pad_J : 0;
            const bool   residual_op = op.kind == potal_paired_op_kind::load_residual_a ||
                                     op.kind == potal_paired_op_kind::index_program ||
                                     op.kind == potal_paired_op_kind::residual_gemm;
            if (residual_op && (!tile.residual || op.window >= panel.windows.size())) {
                return fail(op, "residual op without a window of its panel");
            }
            if (op.kind != potal_paired_op_kind::pair_retire && !st.seen) {
                // A residency starts in order and takes its pair only when the pair is free.
                if (op.residency != next_residency) {
                    return fail(op, "residencies must start in ascending (J group, K panel) order");
                }
                if (!pair_free[op.pair]) {
                    return fail(op, "pair " + num(op.pair) + " still holds residency " + num(pair_owner[op.pair]));
                }
                st.seen            = true;
                st.pair            = op.pair;
                pair_free[op.pair] = false;
                pair_owner[op.pair] = op.residency;
                next_residency++;
            }
            if (st.retired && op.kind != potal_paired_op_kind::pair_retire) {
                return fail(op, "op of a retired residency");
            }
            switch (op.kind) {
                case potal_paired_op_kind::scale_program: {
                    if (op.scale_window >= plan.scale_windows.size() || window_state[op.scale_window] != 0) {
                        return fail(op, "scale window unknown or programmed twice");
                    }
                    const potal_paired_scale_window & w = plan.scale_windows[op.scale_window];
                    if (w.program_op != op.id || w.residency != op.residency || w.slots != w.blocks * w.j_tiles ||
                        w.j_tiles != j_tiles_g || w.slots == 0 || w.slot_base + w.slots > hw.scale_entries ||
                        w.generation != last_generation + 1) {
                        return fail(op, "scale window fields / generation sequence");
                    }
                    if (w.residual) {
                        if (op.window != w.window || op.window >= panel.windows.size() ||
                            w.slot_base != tile.scale_slots_main || w.blocks != panel.windows[op.window].k_tiles) {
                            return fail(op, "residual scale window does not describe its Index Window");
                        }
                    } else if (w.slot_base != 0 || w.blocks != panel.k_tiles / tile.passes_per_block) {
                        return fail(op, "Main scale window does not describe its panel");
                    }
                    for (size_t o = 0; o < plan.scale_windows.size(); ++o) {
                        const potal_paired_scale_window & x = plan.scale_windows[o];
                        if (window_state[o] == 1 && x.slot_base < w.slot_base + w.slots && w.slot_base < x.slot_base + x.slots) {
                            return fail(op, "scale slots overwritten while window " + num(o) + " is live");
                        }
                    }
                    window_state[op.scale_window] = 1;
                    last_generation               = w.generation;
                    break;
                }
                case potal_paired_op_kind::scale_release: {
                    if (op.scale_window >= plan.scale_windows.size() || window_state[op.scale_window] != 1) {
                        return fail(op, "release of a window that is not live");
                    }
                    const potal_paired_scale_window & w = plan.scale_windows[op.scale_window];
                    const size_t consumer               = window_consumer[op.scale_window];
                    if (w.release_op != op.id || consumer == no_index || w.consumer_op != consumer ||
                        !has_dep(op, consumer, true)) {
                        return fail(op, "release before its consumer completed");
                    }
                    window_state[op.scale_window] = 2;
                    complete_upto                 = i + 1; // accepted only after every older command drained
                    break;
                }
                case potal_paired_op_kind::load_main_a: {
                    if (st.load_a != no_index) {
                        return fail(op, "Main A loaded twice for one residency");
                    }
                    if (op.a_sp_begin != layout.activation_begin || op.a_sp_rows != tile.tile_I * panel.k_tiles * D ||
                        op.a_sp_begin + op.a_sp_rows > layout.activation_end || op.a_offset_tiles != 0) {
                        return fail(op, "Main A region outside the activation bank or not at its start");
                    }
                    if (op.residency >= potal_paired_pairs) {
                        const size_t prev = op.residency - potal_paired_pairs;
                        if (!rs[prev].retired || !has_dep(op, plan.residencies[prev].retire_op, false)) {
                            return fail(op, "load into a pair before its previous residency retired");
                        }
                    }
                    st.load_a     = op.id;
                    st.main_a_end = op.a_sp_begin + op.a_sp_rows;
                    break;
                }
                case potal_paired_op_kind::load_weight: {
                    if (st.load_b != no_index) {
                        return fail(op, "weight panel loaded twice in one residency");
                    }
                    const size_t rows = panel.k_tiles * j_tiles_g * D;
                    if (op.b_sp_rows != rows || op.b_sp_begin + rows != layout.weight_end ||
                        op.b_sp_begin < layout.weight_begin || op.weight_generation != op.residency) {
                        return fail(op, "weight panel not rear-placed inside the weight bank / generation");
                    }
                    if (op.residency >= potal_paired_pairs) {
                        const size_t prev = op.residency - potal_paired_pairs;
                        if (!rs[prev].retired || !has_dep(op, plan.residencies[prev].retire_op, false)) {
                            return fail(op, "weight overwritten before the previous residency retired");
                        }
                    }
                    st.load_b  = op.id;
                    st.b_begin = op.b_sp_begin;
                    st.b_rows  = rows;
                    break;
                }
                case potal_paired_op_kind::main_gemm: {
                    if (st.main != no_index || main_seen[op.residency]) {
                        return fail(op, "duplicate Main contribution of a residency");
                    }
                    if (st.load_a == no_index || st.load_b == no_index || !has_dep(op, st.load_a, false) ||
                        !has_dep(op, st.load_b, false)) {
                        return fail(op, "Main GEMM before its loads");
                    }
                    if (op.scale_window >= plan.scale_windows.size() || window_state[op.scale_window] != 1 ||
                        plan.scale_windows[op.scale_window].residual ||
                        plan.scale_windows[op.scale_window].residency != op.residency ||
                        !has_dep(op, plan.scale_windows[op.scale_window].program_op, false)) {
                        return fail(op, "Main GEMM without its live Main scale window");
                    }
                    if (op.i_tiles != tile.tile_I || op.j_tiles != j_tiles_g || op.k_tiles != panel.k_tiles ||
                        op.pad_i != tile.pad_I || op.pad_j != pad_j || op.pad_k != 0 ||
                        op.a_sp_begin != layout.activation_begin || op.a_sp_rows != tile.tile_I * panel.k_tiles * D ||
                        op.b_sp_begin != st.b_begin || op.b_sp_rows != st.b_rows || op.acc_bank != 0 ||
                        op.acc_epoch != op.j_group || op.work_ids != op.i_tiles * op.j_tiles || op.work_ids > hw.work_ids ||
                        op.i_tiles * op.j_tiles * D > hw.acc_rows / 2) {
                        return fail(op, "Main GEMM geometry / ACC0 / work IDs");
                    }
                    acc_identity_state & a = acc0[op.j_group];
                    if (op.first != (a.contributions == 0) || (op.last != (op.panel + 1 == tile.k_panels)) ||
                        a.last_op != no_index) {
                        return fail(op, "ACC0 first / last flags");
                    }
                    a.contributions++;
                    a.final_op = op.id;
                    if (op.last) {
                        a.last_op = op.id;
                    }
                    window_consumer[op.scale_window] = op.id;
                    st.main                          = op.id;
                    main_seen[op.residency]          = 1;
                    break;
                }
                case potal_paired_op_kind::index_program: {
                    const potal_paired_index_window & win = panel.windows[op.window];
                    if (op.window != st.windows_seen) {
                        return fail(op, "Index Windows of a panel must be programmed in order, once each");
                    }
                    if (!st.index_consumed) {
                        return fail(op, "Index Table reprogrammed before the previous window's gemm was issued");
                    }
                    if (op.index_buffer != layout.index_buffer || op.index_buffer != op.pair ||
                        st.load_b == no_index || !has_dep(op, st.load_b, false) || op.index_b_base != st.b_begin ||
                        op.index_j_tiles != j_tiles_g || op.index_source_k_rows != panel.k_rows ||
                        op.index_active_rows != win.active_rows || op.index_active_rows > hw.index_entries ||
                        op.entry_end - op.entry_begin != win.active_rows || op.entry_end > plan.index_entries.size() ||
                        op.index_active_rows != (win.k_tiles - 1) * D + tile.fragments[win.fragment_end - 1].count) {
                        return fail(op, "INDEX_CONFIG fields do not describe the resident panel / window");
                    }
                    if (op.same_buffer_predecessor != buffer_last_gemm[op.pair] ||
                        (buffer_last_gemm[op.pair] != no_index && !has_dep(op, buffer_last_gemm[op.pair], false))) {
                        return fail(op, "same-buffer Index update must follow the previous Indexed gemm of that buffer");
                    }
                    // Entries: slots in order from 0, one fragment after another, fillers only inside.
                    size_t slot = 0, f = win.fragment_begin, x = 0;
                    for (size_t e = op.entry_begin; e < op.entry_end; ++e, ++slot) {
                        const potal_paired_index_entry & en = plan.index_entries[e];
                        if (en.slot != slot || f >= win.fragment_end || en.fragment != f) {
                            return fail(op, "index entry " + num(slot) + ": slot / fragment order");
                        }
                        const potal_paired_fragment & fr = tile.fragments[f];
                        if (en.resident_k >= panel.k_rows || en.original_k != panel.k_begin + en.resident_k) {
                            return fail(op, "index entry " + num(slot) + ": resident K outside the panel");
                        }
                        if (!en.filler) {
                            if (x >= fr.count || en.compact_k != fr.compact_begin + x) {
                                return fail(op, "index entry " + num(slot) + ": compact K order");
                            }
                            if (residual != nullptr &&
                                en.original_k != potal_paired_fragment_original_k(*residual, fr, x, hw)) {
                                return fail(op, "index entry " + num(slot) + ": original K is not the packet's");
                            }
                        } else if (x < fr.count || f + 1 >= win.fragment_end || en.compact_k != no_index) {
                            return fail(op, "index entry " + num(slot) + ": filler outside a short non-last fragment");
                        }
                        ++x;
                        if (x == D || (x == fr.count && f + 1 == win.fragment_end)) {
                            ++f;
                            x = 0;
                        }
                    }
                    if (f != win.fragment_end || x != 0) {
                        return fail(op, "index entries do not cover the window's fragments");
                    }
                    st.last_index_program = op.id;
                    st.index_consumed     = false;
                    st.windows_seen++;
                    break;
                }
                case potal_paired_op_kind::load_residual_a: {
                    const potal_paired_index_window & win = panel.windows[op.window];
                    if (op.a_offset_tiles != tile.residual_a_offset_tiles ||
                        op.a_sp_begin != layout.activation_begin + tile.residual_a_offset_tiles * D ||
                        op.a_sp_rows != win.a_rows || op.a_sp_begin + op.a_sp_rows > layout.activation_end ||
                        st.load_a == no_index || op.a_sp_begin < st.main_a_end || op.i_tiles != tile.residual_row_tiles ||
                        op.k_tiles != win.k_tiles) {
                        return fail(op, "Residual A region overlaps Main A or leaves the activation bank");
                    }
                    if (st.last_residual_gemm != no_index && !has_dep(op, st.last_residual_gemm, true)) {
                        return fail(op, "Residual A reloaded before the previous window's gemm completed");
                    }
                    if (op.residency >= potal_paired_pairs) {
                        const size_t prev = op.residency - potal_paired_pairs;
                        if (!rs[prev].retired || !has_dep(op, plan.residencies[prev].retire_op, false)) {
                            return fail(op, "Residual A loaded before the previous residency retired");
                        }
                    }
                    break;
                }
                case potal_paired_op_kind::residual_gemm: {
                    const potal_paired_index_window & win = panel.windows[op.window];
                    if (st.windows_seen != op.window + 1 || st.index_consumed || st.last_index_program == no_index ||
                        plan.ops[st.last_index_program].window != op.window || !has_dep(op, st.last_index_program, false)) {
                        return fail(op, "Indexed GEMM without its own programmed window");
                    }
                    size_t load_r = no_index;
                    for (size_t o = st.last_index_program; o < op.id; ++o) {
                        const potal_paired_op & x = plan.ops[o];
                        if (x.kind == potal_paired_op_kind::load_residual_a && x.residency == op.residency && x.window == op.window) {
                            load_r = o;
                        }
                    }
                    if (load_r == no_index || !has_dep(op, load_r, false) || st.load_b == no_index ||
                        !has_dep(op, st.load_b, false)) {
                        return fail(op, "Indexed GEMM before its loads");
                    }
                    if (op.scale_window >= plan.scale_windows.size() || window_state[op.scale_window] != 1 ||
                        !plan.scale_windows[op.scale_window].residual ||
                        plan.scale_windows[op.scale_window].window != op.window ||
                        plan.scale_windows[op.scale_window].residency != op.residency ||
                        !has_dep(op, plan.scale_windows[op.scale_window].program_op, false)) {
                        return fail(op, "Indexed GEMM without its live Residual scale window");
                    }
                    if (op.i_tiles != tile.residual_row_tiles || op.j_tiles != j_tiles_g || op.k_tiles != win.k_tiles ||
                        op.pad_i != tile.residual_pad_I || op.pad_j != pad_j || op.pad_k != win.pad_k ||
                        op.b_sp_begin != st.b_begin || op.b_sp_rows != st.b_rows || op.index_buffer != op.pair ||
                        op.acc_bank != 1 || op.acc_epoch != op.j_group || op.work_ids != op.i_tiles * op.j_tiles ||
                        op.work_ids > hw.work_ids || op.i_tiles * op.j_tiles * D > hw.acc_rows / 2) {
                        return fail(op, "Indexed GEMM geometry / ACC1 / work IDs");
                    }
                    acc_identity_state & a = acc1[op.j_group];
                    if (op.first != (a.contributions == 0) || a.last_op != no_index) {
                        return fail(op, "ACC1 first / last flags");
                    }
                    a.contributions++;
                    a.final_op = op.id;
                    if (op.last) {
                        a.last_op = op.id;
                    }
                    window_consumer[op.scale_window] = op.id;
                    st.index_consumed                = true;
                    st.last_residual_gemm            = op.id;
                    st.residual_gemms.push_back(op.id);
                    buffer_last_gemm[op.pair] = op.id;
                    break;
                }
                case potal_paired_op_kind::acc_complete: {
                    acc_identity_state & a = op.acc_bank == 0 ? acc0[op.j_group] : acc1[op.j_group];
                    if (op.acc_bank > 1 || op.acc_epoch != op.j_group || a.last_op == no_index ||
                        a.complete_op != no_index || !has_dep(op, a.last_op, false) ||
                        op.reconstruction_required != (op.acc_bank == 1)) {
                        return fail(op, "ACC completion without a last contribution / twice / reconstruction flag");
                    }
                    a.complete_op = op.id;
                    break;
                }
                case potal_paired_op_kind::pair_retire: {
                    if (!st.seen || st.retired || st.main == no_index || !has_dep(op, st.main, true) ||
                        pair_owner[op.pair] != op.residency || plan.residencies.size() <= op.residency ||
                        plan.residencies[op.residency].retire_op != op.id) {
                        return fail(op, "retire of a residency that is not resident or already retired");
                    }
                    if (st.windows_seen != panel.windows.size() || st.residual_gemms.size() != panel.windows.size()) {
                        return fail(op, "retire before every residual window of the panel was executed");
                    }
                    for (size_t gemm : st.residual_gemms) {
                        if (!has_dep(op, gemm, true)) {
                            return fail(op, "retire without completion of residual gemm " + num(gemm));
                        }
                    }
                    st.retired          = true;
                    pair_free[op.pair]  = true;
                    pair_owner[op.pair] = no_index;
                    break;
                }
            }
        }

        // Whole-plan accounting.
        if (next_residency != total_residencies) {
            return fail("missing residencies: " + num(next_residency) + " of " + num(total_residencies));
        }
        for (size_t r = 0; r < total_residencies; ++r) {
            if (!rs[r].retired || rs[r].main == no_index) {
                return fail("residency " + num(r) + " never retired or without Main contribution");
            }
        }
        for (size_t o = 0; o < plan.scale_windows.size(); ++o) {
            if (window_state[o] != 2) {
                return fail("scale window " + num(o) + " never programmed or never released");
            }
        }
        for (size_t g = 0; g < tile.j_groups; ++g) {
            const acc_identity_state & a = acc0[g];
            if (a.contributions != tile.k_panels || a.last_op == no_index || a.last_op != a.final_op || a.complete_op == no_index) {
                return fail("ACC0 epoch " + num(g) + ": contributions " + num(a.contributions) + " of " +
                            num(tile.k_panels) + ", last / completion missing or not final");
            }
            const acc_identity_state & b = acc1[g];
            if (b.contributions != tile.residual_windows ||
                (tile.residual_windows != 0 && (b.last_op == no_index || b.last_op != b.final_op || b.complete_op == no_index)) ||
                (tile.residual_windows == 0 && b.complete_op != no_index)) {
                return fail("ACC1 epoch " + num(g) + ": contributions " + num(b.contributions) + " of " +
                            num(tile.residual_windows) + ", last / completion missing or not final");
            }
        }
        if (plan.main_gemms != total_residencies || plan.residual_gemms != tile.j_groups * tile.residual_windows ||
            plan.pair_retires != total_residencies || plan.weight_loads != total_residencies) {
            return fail("op counts do not match the tile plan");
        }
        return true;
    }
};

} // namespace

potal_paired_status potal_validate_execution_plan(const potal_paired_execution_plan & plan,
                                                  const residual::decoded_packet * residual, std::string * why,
                                                  const potal_paired_geometry & hw) try {
    if (why != nullptr) {
        why->clear();
    }
    validator v(plan, residual, hw, why);
    return v.run() ? potal_paired_status::ok : potal_paired_status::invalid_execution_plan;
} catch (const std::bad_alloc &) {
    explain(why, "out of memory");
    return potal_paired_status::allocation_failure;
}

// ---- execution (H3-2B) -------------------------------------------------------------------

potal_paired_staging::~potal_paired_staging() {
    std::free(buffer_);
}

void potal_paired_staging::abandon() {
    buffer_ = nullptr; // leaked on purpose: the accelerator may still read it
    bytes_  = 0;
}

potal_paired_status potal_paired_pack_residual(const potal_paired_execution_plan & plan,
                                               const residual::decoded_packet & residual, potal_paired_staging & out,
                                               std::string * why, const potal_paired_geometry & hw) try {
    const potal_paired_tile_plan & tile = plan.tile;
    const size_t                   D    = hw.dim;
    out.windows.clear();
    out.rows = 0;
    std::free(out.buffer_);
    out.buffer_ = nullptr;
    out.bytes_  = 0;
    if (!tile.residual || residual.rows.size() != tile.residual_lane_rows || residual.digits.empty() ||
        residual.digits.size() != residual.rows.size() * residual.k_compact) {
        explain(why, "staging: no residual in the plan or no digits in the packet");
        return potal_paired_status::invalid_argument;
    }
    // Layout: one [rows][lda] matrix per window, lda a multiple of the DMA beat.
    size_t bytes = 0;
    for (const potal_paired_k_panel & panel : tile.panels) {
        for (const potal_paired_index_window & w : panel.windows) {
            potal_paired_staging::window sw;
            sw.panel  = panel.index;
            sw.index  = w.index;
            sw.cols   = w.k_tiles * D;
            sw.lda    = align_up(sw.cols, POTAL_HW_MAX_BYTES);
            sw.offset = bytes;
            size_t rows_bytes = 0;
            if (!mul(residual.rows.size(), sw.lda, rows_bytes) || !add(bytes, rows_bytes, bytes)) {
                explain(why, "staging: size overflow");
                return potal_paired_status::invalid_argument;
            }
            out.windows.push_back(sw);
        }
    }
    if (out.windows.size() != tile.residual_windows) {
        explain(why, "staging: window count differs from the plan");
        return potal_paired_status::invalid_argument;
    }
    out.rows    = residual.rows.size();
    out.buffer_ = static_cast<int8_t *>(dma_alloc(bytes == 0 ? POTAL_HW_MAX_BYTES : bytes)); // zeroed
    if (out.buffer_ == nullptr) {
        explain(why, "staging: out of memory");
        return potal_paired_status::allocation_failure;
    }
    out.bytes_ = bytes;
    // Columns from the plan's entries (J group 0: every group shares the packed digits).
    size_t packed = 0;
    for (const potal_paired_op & op : plan.ops) {
        if (op.kind != potal_paired_op_kind::index_program || op.j_group != 0) {
            continue;
        }
        size_t wi = 0;
        while (wi < out.windows.size() && !(out.windows[wi].panel == op.panel && out.windows[wi].index == op.window)) {
            ++wi;
        }
        if (wi == out.windows.size()) {
            explain(why, "staging: index window not in the plan's panels");
            return potal_paired_status::invalid_argument;
        }
        const potal_paired_staging::window & sw = out.windows[wi];
        for (size_t e = op.entry_begin; e < op.entry_end; ++e) {
            const potal_paired_index_entry & en = plan.index_entries[e];
            if (en.slot >= sw.cols) {
                explain(why, "staging: entry slot beyond the window");
                return potal_paired_status::invalid_argument;
            }
            if (en.filler) {
                continue; // zero digits: no contribution through the filler's resident row
            }
            if (en.compact_k >= residual.k_compact) {
                explain(why, "staging: entry beyond the packet's compact K");
                return potal_paired_status::invalid_argument;
            }
            for (size_t r = 0; r < out.rows; ++r) {
                out.buffer_[sw.offset + r * sw.lda + en.slot] = residual.digits[r * residual.k_compact + en.compact_k];
            }
            ++packed;
        }
    }
    if (packed + tile.residual_filler_slots != plan.index_writes / tile.j_groups) {
        explain(why, "staging: packed columns differ from the plan's entries");
        return potal_paired_status::invalid_argument;
    }
    return potal_paired_status::ok;
} catch (const std::bad_alloc &) {
    explain(why, "out of memory");
    return potal_paired_status::allocation_failure;
}

namespace {

using exec_clock = std::chrono::steady_clock;

double seconds_since(exec_clock::time_point t0) {
    return std::chrono::duration<double>(exec_clock::now() - t0).count();
}

struct executor {
    const potal_paired_request &  req;
    potal_paired_session *        session;
    potal_paired_report &         report;
    const potal_paired_geometry & hw;
    potal_paired_staging          staging;
    std::vector<uint32_t>         hw_generation; // per scale window
    size_t                        drained_upto = 0; // every op with id < drained_upto completed
    size_t                        last_op      = 0; // ops [0, last_op) were handled
    size_t                        pending_readback[2] = { potal_paired_no_index, potal_paired_no_index };
    bool                          window_diag = false;
    std::vector<int32_t>           window_expected; // original lane rows x N, separate J-group prefixes
    std::unique_ptr<int32_t, decltype(&std::free)> window_snapshot{ nullptr, &std::free };
    size_t                        window_ldc = 0;
    size_t                        window_fences = 0, window_readbacks = 0, window_readback_tiles = 0;

    bool profile = false;
    exec_clock::time_point plan_start;
    double plan_s = 0, pack_s = 0, scale_write_s = 0, scale_release_s = 0;
    double index_s = 0, main_issue_s = 0, residual_issue_s = 0, readback_s = 0;

    ~executor() {
        if (!profile) return;
        if (plan_start != exec_clock::time_point{}) plan_s += seconds_since(plan_start);
        std::fprintf(stderr,
                     "[POTAL_PROFILE] run_id=%" PRIu64 " stripe=%zu status=%s plan_ms=%.6f pack_ms=%.6f "
                     "scale_write_ms=%.6f scale_release_ms=%.6f index_ms=%.6f main_issue_ms=%.6f "
                     "residual_issue_ms=%.6f readback_ms=%.6f fence_ms=%.6f\n",
                     req.run_id, req.stripe_id, potal_paired_status_name(report.status), plan_s * 1e3, pack_s * 1e3,
                     scale_write_s * 1e3, scale_release_s * 1e3, index_s * 1e3, main_issue_s * 1e3,
                     residual_issue_s * 1e3, readback_s * 1e3, report.counts.fence_s * 1e3);
    }

    executor(const potal_paired_request & r, potal_paired_session * s, potal_paired_report & rep,
             const potal_paired_geometry & g) :
        req(r),
        session(s),
        report(rep),
        hw(g) {}

    // After a failure with commands issued: an answered failure is cleaned (drain and
    // release this call's scale entries) so the accelerator is known idle again; a
    // failed fence is never retried. Allocation-free: also the exception path.
    void settle(potal_status hw_st, bool fence_failed) noexcept {
        if (!report.issued) {
            return;
        }
        if (!fence_failed && hw_st != POTAL_ERR_NO_HARDWARE) {
            report.cleaned = potal_paired_session_cleanup(session) == POTAL_OK;
        }
        const potal_paired_session_status q = potal_paired_session_query(session);
        report.fatal = q.outstanding_loops != 0 || q.outstanding_readbacks != 0 || q.live_scale_slots != 0 ||
                       (q.failed != 0 && !report.cleaned);
        if (hw_st == POTAL_ERR_NO_HARDWARE && q.outstanding_loops == 0 && q.outstanding_readbacks == 0 &&
            q.live_scale_slots == 0) {
            report.fatal = false; // nothing reached any hardware
        }
        if (report.fatal) {
            staging.abandon();
            (void) window_snapshot.release(); // outstanding diagnostic DMA must keep its destination alive
        }
    }

    potal_paired_status fail(potal_paired_status st, const char * stage, std::string detail, potal_status hw_st) {
        report.status    = st;
        report.stage     = stage;
        report.hw_status = hw_st;
        settle(hw_st, std::strcmp(stage, "fence") == 0); // before anything that may allocate
        report.detail = std::move(detail);
        if (report.issued && !report.cleaned && !report.fatal && hw_st != POTAL_ERR_NO_HARDWARE &&
            std::strcmp(stage, "fence") != 0) {
            report.detail += "; cleanup failed";
        }
        return st;
    }

    // An exception (out of memory while formatting a failure, ...) after commands may
    // have been issued: the same settlement as fail(), nothing allocated.
    potal_paired_status on_exception(potal_paired_status st, const char * stage) noexcept {
        report.status = st;
        report.stage  = stage;
        const potal_paired_session_status q = potal_paired_session_query(session);
        settle(POTAL_OK, q.failed != 0 && q.error_stage != nullptr && std::strcmp(q.error_stage, "fence") == 0);
        return st;
    }

    potal_paired_status hw_fail(const char * stage, potal_status hw_st, const potal_paired_op * op) {
        return fail(potal_paired_status::execution_failed, stage,
                    std::string("POTAL_") + potal_status_name(hw_st) + (op != nullptr ? " at " + potal_paired_op_text(*op) : ""),
                    hw_st);
    }

    uint32_t next_generation() {
        *req.generation = *req.generation % POTAL_GENERATION_MAX + 1; // 1..255
        report.last_generation = *req.generation;
        return *req.generation;
    }

    // Columns of J group g: tiles and the valid columns of its last tile.
    size_t group_j_tiles(size_t g) const {
        const potal_paired_tile_plan & t = report.tile;
        return g + 1 == t.j_groups ? t.j_tiles - (t.j_groups - 1) * t.tile_J : t.tile_J;
    }
    size_t group_pad_j(size_t g) const { return g + 1 == report.tile.j_groups ? report.tile.pad_J : 0; }

    potal_status fence_now() {
        const auto         t  = exec_clock::now();
        const potal_status st = potal_paired_fence(session);
        report.counts.fence_s += seconds_since(t);
        if (st != POTAL_OK) {
            return st;
        }
        report.counts.fences++;
        report.counts.drains++;
        drained_upto = last_op;
        return POTAL_OK;
    }

    // Explicit fence when a completion is required that no drain point covered yet.
    potal_status ensure_complete(size_t dep_op) { return dep_op < drained_upto ? POTAL_OK : fence_now(); }

    // Conservative fences: everything issued so far completes first.
    potal_status drain_all() { return req.conservative_fences && drained_upto < last_op ? fence_now() : POTAL_OK; }

    potal_paired_status prepare_window_diag() {
        const char * enabled = std::getenv("POTAL_PAIRED_WINDOW_DIAG");
        window_diag = enabled != nullptr && std::strcmp(enabled, "1") == 0 && report.plan.residual_gemms != 0;
        if (!window_diag) {
            return potal_paired_status::ok;
        }
        const potal_paired_tile_plan & t = report.tile;
        size_t elements = 0, bytes = 0, rounded = 0;
        window_ldc = t.tile_J * hw.dim;
        if (!mul(t.residual_lane_rows, req.n, elements) || !mul(elements, sizeof(int32_t), bytes) ||
            !mul(t.residual_row_tiles * hw.dim, window_ldc, bytes) || !mul(bytes, sizeof(int32_t), bytes) ||
            !add(bytes, POTAL_HW_MAX_BYTES - 1, rounded)) {
            return fail(potal_paired_status::invalid_argument, "window_diag", "diagnostic buffer size overflow", POTAL_OK);
        }
        window_expected.assign(elements, 0);
        window_snapshot.reset(static_cast<int32_t *>(dma_alloc(bytes)));
        if (!window_snapshot) {
            return fail(potal_paired_status::allocation_failure, "window_diag", "diagnostic buffer allocation failed", POTAL_OK);
        }
        return potal_paired_status::ok;
    }

    potal_paired_status diagnose_window(const potal_paired_op & op) {
        // Keep these commands out of performance counts and leave the plan's drain bookkeeping unchanged.
        potal_status st = potal_paired_fence(session);
        if (st != POTAL_OK) {
            return hw_fail("fence", st, &op);
        }
        ++window_fences;
        potal_acc_readback r = {};
        r.acc_bank = 1;
        r.i_tiles  = (uint32_t) op.i_tiles;
        r.j_tiles  = (uint32_t) op.j_tiles;
        r.pad_i    = (uint32_t) op.pad_i;
        r.pad_j    = (uint32_t) op.pad_j;
        r.c        = window_snapshot.get();
        r.ldc      = window_ldc;
        st = potal_paired_acc_readback_issue(session, &r);
        if (st != POTAL_OK) {
            return hw_fail("window_diag_readback", st, &op);
        }
        ++window_readbacks;
        window_readback_tiles += (size_t) r.i_tiles * r.j_tiles;
        st = potal_paired_fence(session);
        if (st != POTAL_OK) {
            return hw_fail("fence", st, &op);
        }
        ++window_fences;

        namespace hp1ref = ggml::gemmini::quants::hp1;
        const residual::decoded_packet & d = *req.residual;
        const potal_paired_k_panel & pan = report.tile.panels[op.panel];
        const potal_paired_index_window & win = pan.windows[op.window];
        const size_t col_begin = op.j_group * report.tile.tile_J * hw.dim;
        const size_t cols = op.j_tiles * hw.dim - op.pad_j;
        // Never use staged A or Index entries here: either mapping may be the source of the mismatch.
        for (size_t f = win.fragment_begin; f < win.fragment_end; ++f) {
            const potal_paired_fragment & fr = report.tile.fragments[f];
            const residual::run & run = d.runs[fr.run];
            for (size_t row = 0; row < d.rows.size(); ++row) {
                const int8_t * digits = d.digits.data() + row * d.k_compact + fr.compact_begin;
                for (size_t c = 0; c < cols; ++c) {
                    const size_t col = col_begin + c;
                    int32_t partial = 0;
                    for (size_t x = 0; x < fr.count; ++x) {
                        const size_t k = (size_t) run.block_id * hw.block_k + run.local_k[fr.local_begin + x];
                        partial += (int32_t) digits[x] * req.weight->b[k * req.weight->ldb + col];
                    }
                    const uint32_t carrier = req.weight->carriers[(size_t) run.block_id * req.weight->cstride + col];
                    const int32_t value = hp1ref::apply_validated(partial, carrier);
                    int32_t & acc = window_expected[row * req.n + col];
                    acc = op.first && f == win.fragment_begin ? value : hp1ref::accumulate(acc, value);
                }
            }
        }
        size_t bad = 0;
        size_t mismatch_row[4], mismatch_col[4];
        for (size_t row = 0; row < d.rows.size(); ++row) {
            for (size_t c = 0; c < cols; ++c) {
                if (r.c[row * r.ldc + c] != window_expected[row * req.n + col_begin + c]) {
                    if (bad < 4) {
                        mismatch_row[bad] = row;
                        mismatch_col[bad] = c;
                    }
                    ++bad;
                }
            }
        }
        std::fprintf(stderr,
                     "[POTAL_WINDOW_DIAG] j_group=%zu panel=%zu window=%zu pair=%zu active_rows=%zu pad_k=%zu "
                     "fragment_begin=%zu fragment_end=%zu prefix_bad=%zu prefix_total=%zu "
                     "i_tiles=%zu j_tiles=%zu k_tiles=%zu pad_i=%zu pad_j=%zu col_begin=%zu "
                     "fragments_per_window=%zu passes_per_block=1 index_entries=%zu "
                     "run_id=%" PRIu64 " stripe_id=%zu row_begin=%zu "
                     "diag_fences=%zu diag_readbacks=%zu diag_readback_tiles=%zu\n",
                     op.j_group, op.panel, op.window, op.pair, win.active_rows, op.pad_k,
                     win.fragment_begin, win.fragment_end, bad, d.rows.size() * cols,
                     op.i_tiles, op.j_tiles, op.k_tiles, op.pad_i, op.pad_j, col_begin,
                     report.tile.fragments_per_window, hw.index_entries,
                     req.run_id, req.stripe_id, req.row_begin,
                     window_fences, window_readbacks, window_readback_tiles);
        std::fflush(stderr);
        for (size_t i = 0; i < std::min<size_t>(bad, 4); ++i) {
            const size_t row = mismatch_row[i], c = mismatch_col[i];
            std::fprintf(stderr,
                         "[POTAL_WINDOW_MISMATCH] j_group=%zu panel=%zu window=%zu lane_row=%zu "
                         "source_row=%zu col=%zu got=%" PRId32 " expected=%" PRId32 " lane=%u\n",
                         op.j_group, op.panel, op.window, row, (size_t) d.rows[row].source_row, col_begin + c,
                         r.c[row * r.ldc + c], window_expected[row * req.n + col_begin + c],
                         (unsigned) d.rows[row].lane);
            std::fflush(stderr);
        }
        const potal_paired_op * index = nullptr;
        for (const potal_paired_dep & dep : op.deps) {
            if (report.plan.ops[dep.op].kind == potal_paired_op_kind::index_program) {
                index = &report.plan.ops[dep.op];
                break;
            }
        }
        for (size_t f = win.fragment_begin; f < std::min(win.fragment_end, win.fragment_begin + 4); ++f) {
            const potal_paired_fragment & fr = report.tile.fragments[f];
            const residual::run & run = d.runs[fr.run];
            const size_t slot = (f - win.fragment_begin) * hw.dim;
            const size_t sample_k = std::min<size_t>(fr.count, hw.index_entries == 16 ? 16 : 8);
            std::fprintf(stderr,
                         "[POTAL_WINDOW_INPUT] j_group=%zu panel=%zu window=%zu pair=%zu fragment=%zu "
                         "block=%u slot_begin=%zu count=%u original_k=",
                         op.j_group, op.panel, op.window, op.pair, f, (unsigned) run.block_id, slot, (unsigned) fr.count);
            for (size_t x = 0; x < sample_k; ++x) {
                std::fprintf(stderr, "%s%zu", x == 0 ? "" : ",",
                             (size_t) run.block_id * hw.block_k + run.local_k[fr.local_begin + x]);
            }
            std::fprintf(stderr, " resident_k=");
            if (index != nullptr) {
                for (size_t x = 0; x < sample_k; ++x) {
                    std::fprintf(stderr, "%s%zu", x == 0 ? "" : ",",
                                 report.plan.index_entries[index->entry_begin + slot + x].resident_k);
                }
            }
            std::fprintf(stderr, " carriers_col_begin=%zu carriers=", col_begin);
            for (size_t c = 0; c < std::min<size_t>(cols, 4); ++c) {
                std::fprintf(stderr, "%s0x%08" PRIx32, c == 0 ? "" : ",",
                             req.weight->carriers[(size_t) run.block_id * req.weight->cstride + col_begin + c]);
            }
            std::fprintf(stderr, "\n");
            std::fflush(stderr);
        }
        return potal_paired_status::ok;
    }

    potal_paired_status scale_program(const potal_paired_op & op) {
        const potal_paired_scale_window & w   = report.plan.scale_windows[op.scale_window];
        const potal_paired_k_panel &      pan = report.tile.panels[op.panel];
        const uint32_t                    gen = next_generation();
        hw_generation[w.index]                = gen;
        const size_t col_base = op.j_group * report.tile.tile_J * hw.dim;
        const size_t pad_j    = group_pad_j(op.j_group);
        const auto   t        = exec_clock::now();
        for (size_t b = 0; b < w.blocks; ++b) {
            // Main: scale block of the panel; Residual: the fragment's original block.
            const size_t block = w.residual ? report.tile.fragments[pan.windows[op.window].fragment_begin + b].block_id
                                            : pan.block_begin + b;
            for (size_t j = 0; j < w.j_tiles; ++j) {
                const size_t valid = hw.dim - (j + 1 == w.j_tiles ? pad_j : 0);
                for (size_t c = 0; c < valid; ++c) {
                    const uint32_t carrier = req.weight->carriers[block * req.weight->cstride + col_base + j * hw.dim + c];
                    const potal_status st  = potal_paired_scale_write(session, (uint32_t) (w.slot_base + b * w.j_tiles + j),
                                                                      gen, (uint32_t) c, carrier);
                    report.issued = true;
                    if (st != POTAL_OK) {
                        const double elapsed = seconds_since(t);
                        report.counts.scale_s += elapsed;
                        if (profile) scale_write_s += elapsed;
                        return hw_fail("scale_write", st, &op);
                    }
                    report.counts.scale_writes++;
                }
            }
        }
        const double elapsed = seconds_since(t);
        report.counts.scale_s += elapsed;
        if (profile) scale_write_s += elapsed;
        return potal_paired_status::ok;
    }

    potal_paired_status scale_release(const potal_paired_op & op) {
        const potal_status ds = drain_all();
        if (ds != POTAL_OK) {
            return hw_fail("fence", ds, &op);
        }
        const potal_paired_scale_window & w     = report.plan.scale_windows[op.scale_window];
        const size_t                      pad_j = group_pad_j(op.j_group);
        const auto                        t     = exec_clock::now();
        for (size_t b = 0; b < w.blocks; ++b) {
            for (size_t j = 0; j < w.j_tiles; ++j) {
                const size_t valid = hw.dim - (j + 1 == w.j_tiles ? pad_j : 0);
                for (size_t c = 0; c < valid; ++c) {
                    const potal_status st = potal_paired_scale_release(session, (uint32_t) (w.slot_base + b * w.j_tiles + j),
                                                                       hw_generation[w.index], (uint32_t) c);
                    if (st != POTAL_OK) {
                        const double elapsed = seconds_since(t);
                        report.counts.scale_s += elapsed;
                        if (profile) scale_release_s += elapsed;
                        return hw_fail("scale_release", st, &op);
                    }
                    report.counts.scale_releases++;
                }
            }
        }
        const double elapsed = seconds_since(t);
        report.counts.scale_s += elapsed;
        if (profile) scale_release_s += elapsed;
        // The hardware accepted the releases only after every older command drained.
        report.counts.drains++;
        drained_upto = last_op;
        return potal_paired_status::ok;
    }

    potal_paired_status issue_loop(const potal_paired_loop_ws & l, const char * stage, const potal_paired_op & op) {
        const auto         t  = exec_clock::now();
        const potal_status st = potal_paired_loop_ws_issue(session, &l);
        const double elapsed = seconds_since(t);
        report.counts.issue_s += elapsed;
        if (profile) (op.kind == potal_paired_op_kind::main_gemm ? main_issue_s : residual_issue_s) += elapsed;
        report.issued = true;
        return st == POTAL_OK ? potal_paired_status::ok : hw_fail(stage, st, &op);
    }

    potal_paired_status main_gemm(const potal_paired_op & op) {
        const potal_paired_tile_plan & t   = report.tile;
        const potal_paired_k_panel &   pan = t.panels[op.panel];
        if (op.first && pending_readback[0] != potal_paired_no_index) {
            const potal_status st = ensure_complete(pending_readback[0]); // ACC0 of the previous epoch read back
            if (st != POTAL_OK) {
                return hw_fail("fence", st, &op);
            }
        }
        potal_paired_loop_ws l   = {};
        l.loop.a                 = req.a + pan.k_begin;
        l.loop.b                 = req.weight->b + pan.k_begin * req.weight->ldb + op.j_group * t.tile_J * hw.dim;
        l.loop.c                 = nullptr;
        l.loop.lda               = req.lda;
        l.loop.ldb               = req.weight->ldb;
        l.loop.ldc               = 0;
        l.loop.i_tiles           = (uint32_t) op.i_tiles;
        l.loop.j_tiles           = (uint32_t) op.j_tiles;
        l.loop.k_tiles           = (uint32_t) op.k_tiles;
        l.loop.pad_i             = (uint32_t) op.pad_i;
        l.loop.pad_j             = (uint32_t) op.pad_j;
        l.loop.pad_k             = (uint32_t) op.pad_k;
        l.loop.scale_base        = (uint32_t) report.plan.scale_windows[op.scale_window].slot_base;
        l.loop.scale_generation  = hw_generation[op.scale_window];
        l.loop.passes_per_block  = t.passes_per_block;
        l.loop.flags             = op.first ? 0 : POTAL_GEMM_ACCUMULATE;
        l.path                   = POTAL_PAIRED_MAIN;
        l.pair                   = (uint32_t) op.pair;
        l.a_offset_tiles         = 0;
        l.resident_b             = 0;
        l.indexed                = 0;
        const potal_paired_status st = issue_loop(l, "main_loop_ws", op);
        if (st == potal_paired_status::ok) {
            report.counts.main_loop_ws++;
        }
        return st;
    }

    potal_paired_status index_program(const potal_paired_op & op) {
        potal_index_config cfg;
        cfg.pair          = (uint32_t) op.index_buffer;
        cfg.b_sp_base     = (uint32_t) op.index_b_base;
        cfg.j_tiles       = (uint32_t) op.index_j_tiles;
        cfg.source_k_rows = (uint32_t) op.index_source_k_rows;
        cfg.active_rows   = (uint32_t) op.index_active_rows;
        const auto   t    = exec_clock::now();
        potal_status st   = potal_paired_index_config(session, &cfg);
        report.issued     = true;
        if (st != POTAL_OK) {
            const double elapsed = seconds_since(t);
            report.counts.issue_s += elapsed;
            if (profile) index_s += elapsed;
            return hw_fail("index_config", st, &op);
        }
        report.counts.index_configs++;
        for (size_t e = op.entry_begin; e < op.entry_end; ++e) {
            const potal_paired_index_entry & en = report.plan.index_entries[e];
            potal_index_write                w  = { (uint32_t) en.slot, (uint32_t) en.resident_k };
            st                                  = potal_paired_index_write(session, &w);
            if (st != POTAL_OK) {
                const double elapsed = seconds_since(t);
                report.counts.issue_s += elapsed;
                if (profile) index_s += elapsed;
                return hw_fail("index_write", st, &op);
            }
            report.counts.index_writes++;
        }
        const double elapsed = seconds_since(t);
        report.counts.issue_s += elapsed;
        if (profile) index_s += elapsed;
        return potal_paired_status::ok;
    }

    potal_paired_status residual_gemm(const potal_paired_op & op) {
        if (op.first && pending_readback[1] != potal_paired_no_index) {
            const potal_status st = ensure_complete(pending_readback[1]);
            if (st != POTAL_OK) {
                return hw_fail("fence", st, &op);
            }
        }
        const potal_status ds = drain_all(); // conservative: the resident B and every older command complete
        if (ds != POTAL_OK) {
            return hw_fail("fence", ds, &op);
        }
        size_t wi = 0;
        while (wi < staging.windows.size() && !(staging.windows[wi].panel == op.panel && staging.windows[wi].index == op.window)) {
            ++wi;
        }
        if (wi == staging.windows.size()) {
            return fail(potal_paired_status::invalid_execution_plan, "staging", "no staged window for the op", POTAL_OK);
        }
        potal_paired_loop_ws l   = {};
        l.loop.a                 = staging.a(wi);
        l.loop.b                 = nullptr;
        l.loop.c                 = nullptr;
        l.loop.lda               = staging.windows[wi].lda;
        l.loop.ldb               = 0;
        l.loop.ldc               = 0;
        l.loop.i_tiles           = (uint32_t) op.i_tiles;
        l.loop.j_tiles           = (uint32_t) op.j_tiles;
        l.loop.k_tiles           = (uint32_t) op.k_tiles;
        l.loop.pad_i             = (uint32_t) op.pad_i;
        l.loop.pad_j             = (uint32_t) op.pad_j;
        l.loop.pad_k             = (uint32_t) op.pad_k;
        l.loop.scale_base        = (uint32_t) report.plan.scale_windows[op.scale_window].slot_base;
        l.loop.scale_generation  = hw_generation[op.scale_window];
        l.loop.passes_per_block  = 1; // one carrier per fragment
        l.loop.flags             = op.first ? 0 : POTAL_GEMM_ACCUMULATE;
        l.path                   = POTAL_PAIRED_RESIDUAL;
        l.pair                   = (uint32_t) op.pair;
        l.a_offset_tiles         = (uint32_t) op.a_offset_tiles;
        l.resident_b             = 1;
        l.indexed                = 1;
        const potal_paired_status st = issue_loop(l, "residual_loop_ws", op);
        if (st == potal_paired_status::ok) {
            report.counts.residual_loop_ws++;
            if (window_diag) {
                return diagnose_window(op);
            }
        }
        return st;
    }

    potal_paired_status acc_complete(const potal_paired_op & op) {
        const potal_status ds = drain_all(); // conservative: the epoch's last contribution complete
        if (ds != POTAL_OK) {
            return hw_fail("fence", ds, &op);
        }
        const potal_paired_tile_plan & t = report.tile;
        potal_acc_readback r;
        r.acc_bank = (uint32_t) op.acc_bank;
        r.i_tiles  = (uint32_t) (op.acc_bank == 0 ? t.tile_I : t.residual_row_tiles);
        r.j_tiles  = (uint32_t) group_j_tiles(op.j_group);
        r.pad_i    = (uint32_t) (op.acc_bank == 0 ? t.pad_I : t.residual_pad_I);
        r.pad_j    = (uint32_t) group_pad_j(op.j_group);
        const size_t col = op.j_group * t.tile_J * hw.dim;
        r.c   = (op.acc_bank == 0 ? req.c_main : req.c_residual) + col;
        r.ldc = op.acc_bank == 0 ? req.ldc_main : req.ldc_residual;
        const auto         tm = exec_clock::now();
        const potal_status st = potal_paired_acc_readback_issue(session, &r);
        const double elapsed = seconds_since(tm);
        report.counts.issue_s += elapsed;
        if (profile) readback_s += elapsed;
        report.issued = true;
        if (st != POTAL_OK) {
            return hw_fail("acc_readback", st, &op);
        }
        report.counts.readbacks++;
        report.counts.readback_tiles += (uint64_t) r.i_tiles * r.j_tiles;
        pending_readback[op.acc_bank] = op.id;
        return potal_paired_status::ok;
    }

    potal_paired_status run() {
        const potal_paired_execution_plan & plan = report.plan;
        hw_generation.assign(plan.scale_windows.size(), 0);
        for (const potal_paired_op & op : plan.ops) {
            // Completion the plan requires before this op (retire, residual A reuse): a
            // drain point already passed, or an explicit fence.
            if (op.kind != potal_paired_op_kind::scale_release) { // the release is held by the hardware itself
                for (const potal_paired_dep & d : op.deps) {
                    if (d.complete) {
                        const potal_status st = ensure_complete(d.op);
                        if (st != POTAL_OK) {
                            return hw_fail("fence", st, &op);
                        }
                    }
                }
            }
            potal_paired_status st = potal_paired_status::ok;
            switch (op.kind) {
                case potal_paired_op_kind::scale_program:   st = scale_program(op); break;
                case potal_paired_op_kind::scale_release:   st = scale_release(op); break;
                case potal_paired_op_kind::main_gemm:       st = main_gemm(op); break;
                case potal_paired_op_kind::index_program:   st = index_program(op); break;
                case potal_paired_op_kind::residual_gemm:   st = residual_gemm(op); break;
                case potal_paired_op_kind::acc_complete:    st = acc_complete(op); break;
                case potal_paired_op_kind::load_main_a:     // the Main LOOP_WS loads it
                case potal_paired_op_kind::load_weight:     // the Main LOOP_WS loads it
                case potal_paired_op_kind::load_residual_a: // the Indexed LOOP_WS loads it
                case potal_paired_op_kind::pair_retire:     // its completion deps were ensured above
                    break;
            }
            if (st != potal_paired_status::ok) {
                return st;
            }
            report.counts.ops++;
            last_op = op.id + 1;
        }
        // Every readback DMA must have landed before the CPU reads c_main / c_residual.
        const potal_status st = fence_now();
        if (st != POTAL_OK) {
            return hw_fail("fence", st, nullptr);
        }
        return potal_paired_status::ok;
    }
};

bool aligned_to(const void * p, size_t to) {
    return p != nullptr && reinterpret_cast<uintptr_t>(p) % to == 0;
}

} // namespace

potal_paired_status potal_paired_plan_stripe(const potal_tile_plan & main, size_t rows, size_t n, size_t k,
                                             const residual::decoded_packet * residual, potal_paired_tile_plan & tile,
                                             potal_paired_execution_plan & plan, std::string * why,
                                             const potal_paired_geometry & hw) {
    const potal_paired_status st = potal_make_tile_plan(main, rows, n, k, residual, tile, why, hw);
    if (st != potal_paired_status::ok) {
        return st;
    }
    return potal_make_execution_plan(tile, tile.residual ? residual : nullptr, plan, why, hw);
}

bool potal_paired_capacity_refusal(potal_paired_status s) {
    switch (s) {
        case potal_paired_status::main_acc_capacity:
        case potal_paired_status::index_window_capacity:
        case potal_paired_status::residual_acc_capacity:
        case potal_paired_status::weight_bank_capacity:
        case potal_paired_status::activation_bank_capacity:
        case potal_paired_status::scale_window_capacity: return true;
        default: return false;
    }
}

potal_paired_status run_paired_on_owner(const potal_paired_request & req, uint32_t * generation, bool on_owner_cpu,
                                        potal_paired_report & report) {
    if (!on_owner_cpu) {
        report        = potal_paired_report{};
        report.status = potal_paired_status::execution_failed;
        report.stage  = "affinity";
        report.detail = "paired stripe off the accelerator owner's CPU";
        report.fatal  = true;
        return report.status;
    }
    potal_paired_request r = req;
    r.generation           = generation;
    return potal_paired_run_session(r, nullptr, report);
}

potal_paired_status potal_paired_run_session(const potal_paired_request & req, const potal_paired_sink * sink,
                                             potal_paired_report & report) {
    report                       = potal_paired_report{};
    potal_paired_session * s     = nullptr;
    const potal_status     open  = potal_paired_session_open(sink, &s);
    if (open != POTAL_OK) {
        report.status    = potal_paired_status::execution_failed;
        report.stage     = "session_open";
        report.hw_status = open;
        report.detail    = std::string("potal_paired_session_open POTAL_") + potal_status_name(open);
        return report.status;
    }
    const potal_paired_status st = potal_paired_matmul_auto(req, s, report);
    // Not fatal: drained (or cleaned), the accelerator is free. Fatal: close keeps it
    // owned when work may be outstanding; the owner treats the accelerator as unknown.
    (void) potal_paired_session_close(s);
    return st;
}

potal_paired_status potal_paired_matmul_auto(const potal_paired_request & req, potal_paired_session * session,
                                             potal_paired_report & report) {
    report = potal_paired_report{};
    const potal_paired_geometry hw = req.hw != nullptr ? *req.hw : potal_paired_hw_geometry();
    executor                    ex(req, session, report, hw); // outlives the try: the staging is settled on any exit
    const char * profile = std::getenv("POTAL_PAIRED_PROFILE");
    ex.profile = profile != nullptr && std::strcmp(profile, "1") == 0;
    if (ex.profile) ex.plan_start = exec_clock::now();
    try {
        const hp1_weight * w = req.weight;
        // A packet with neither rows nor runs is no residual; an inconsistent one is the planner's refusal.
        const bool has_residual =
            req.residual != nullptr && !(req.residual->rows.empty() && req.residual->runs.empty());

        // 0. Request: pointers, alignment, the weight's geometry, the generation counter.
        if (session == nullptr || req.main_plan == nullptr || w == nullptr || req.a == nullptr || req.c_main == nullptr ||
            req.generation == nullptr || req.rows == 0 || req.n == 0 || req.k == 0 || hw.dim == 0) {
            return ex.fail(potal_paired_status::invalid_argument, "request", "missing operand, plan, session or generation", POTAL_OK);
        }
        if (w->b == nullptr || w->carriers == nullptr || w->scale == nullptr || w->n != req.n || w->k != req.k ||
            w->ldb < req.n || w->cstride < req.n) {
            return ex.fail(potal_paired_status::invalid_argument, "request", "prepared weight does not match n / k", POTAL_OK);
        }
        if (req.lda < req.k || !aligned_to(req.a, hw.dim) || !aligned_to(w->b, hw.dim) || req.ldc_main < req.n ||
            !aligned_to(req.c_main, hw.dim * sizeof(int32_t))) {
            return ex.fail(potal_paired_status::invalid_argument, "request", "A / B / C alignment or stride", POTAL_OK);
        }
        if (has_residual) {
            if (req.residual->digits.empty() || req.c_residual == nullptr || req.ldc_residual < req.n ||
                !aligned_to(req.c_residual, hw.dim * sizeof(int32_t))) {
                return ex.fail(potal_paired_status::invalid_argument, "request",
                               "residual digits or residual C missing / misaligned", POTAL_OK);
            }
        }
        if (*req.generation > POTAL_GENERATION_MAX) {
            return ex.fail(potal_paired_status::invalid_argument, "request", "generation counter beyond 8 bits", POTAL_OK);
        }

        // 1. Plans: Main-derived I/J, the common K, the ordered validated ops; or the caller's
        //    plan, validated again and checked against this request.
        std::string         why;
        potal_paired_status st = potal_paired_status::ok;
        if (req.plan != nullptr) {
            const potal_paired_tile_plan & t       = req.plan->tile;
            const size_t                   i_tiles = (req.rows + hw.dim - 1) / hw.dim;
            if (t.rows != req.rows || t.n != req.n || t.k != req.k || t.tile_J != req.main_plan->tile_J ||
                t.tile_K_main != req.main_plan->tile_K || t.tile_I != std::min(req.main_plan->tile_I, i_tiles) ||
                t.residual != has_residual) {
                return ex.fail(potal_paired_status::invalid_argument, "plan", "the given plan describes another request", POTAL_OK);
            }
            st = potal_validate_execution_plan(*req.plan, has_residual ? req.residual : nullptr, &why, hw);
            if (st != potal_paired_status::ok) {
                return ex.fail(st, "plan", why, POTAL_OK);
            }
            report.plan = *req.plan;
            report.tile = req.plan->tile;
        } else {
            st = potal_paired_plan_stripe(*req.main_plan, req.rows, req.n, req.k, has_residual ? req.residual : nullptr,
                                          report.tile, report.plan, &why, hw);
            if (st != potal_paired_status::ok) {
                return ex.fail(st, "plan", why, POTAL_OK);
            }
        }

        if (ex.profile) {
            ex.plan_s = seconds_since(ex.plan_start);
            ex.plan_start = exec_clock::time_point{};
        }
        const auto pack_start = ex.profile ? exec_clock::now() : exec_clock::time_point{};
        // 2. Residual staging (DMA-ready, slot order), before any command.
        if (report.tile.residual) {
            st = potal_paired_pack_residual(report.plan, *req.residual, ex.staging, &why, hw);
            if (st != potal_paired_status::ok) {
                return ex.fail(st, "pack", why, POTAL_OK);
            }
        }

        if (ex.profile) ex.pack_s = seconds_since(pack_start);

        // 3. The session must be idle: nothing outstanding, no open Index transaction, not failed.
        const potal_paired_session_status q = potal_paired_session_query(session);
        if (q.failed || q.outstanding_loops != 0 || q.outstanding_readbacks != 0 || q.index_transaction_open != 0) {
            return ex.fail(potal_paired_status::invalid_argument, "session", "paired session busy or failed", q.error);
        }

        st = ex.prepare_window_diag();
        if (st != potal_paired_status::ok) {
            return st;
        }

        // 4. TLB maintenance before the first DMA of this stripe (the session is drained).
        if (req.tlb_flush) {
            const auto         t  = exec_clock::now();
            const potal_status fs = potal_paired_tlb_flush(session);
            report.counts.issue_s += seconds_since(t);
            report.issued = true;
            if (fs != POTAL_OK) {
                return ex.hw_fail("tlb_flush", fs, nullptr);
            }
            report.counts.tlb_flushes++;
        }

        // 5. The plan's ops, then the final drain.
        st = ex.run();
        if (st != potal_paired_status::ok) {
            return st;
        }
        report.status = potal_paired_status::ok;
        report.stage  = nullptr;
        return potal_paired_status::ok;
    } catch (const std::bad_alloc &) {
        return ex.on_exception(potal_paired_status::allocation_failure, "allocation");
    } catch (...) {
        return ex.on_exception(potal_paired_status::execution_failed, "exception");
    }
}

} // namespace ggml::potal
