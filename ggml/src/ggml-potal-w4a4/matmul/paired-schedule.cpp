// Paired micro-tile schedule of one stripe and its static checks (paired.hpp).
//
// Per stripe: for every N tile group (ACC epoch), for every common K window,
// one window instance u = j_group * k_windows + window on SP pair u % 2:
//
//   [loads(0)]                                         (instance 0, before the loop)
//   window_begin(u)                                    pair READY -> ACTIVE
//   loads(u + 1) into the other pair                   prefetch while u computes
//   Main stream     (pass, N tile), K order            ACC bank 0
//   Residual stream groups in the window's order,      ACC bank 1
//                   each (fragment, N tile)
//   interleaved 1:1 (Main first) while both have work, then the rest
//   [acc_complete Main, Residual exit]                 after the J group's last window
//   window_end(u)                                      pair -> EMPTY
//
// loads(u): Main A, the shared weight (once), the scale context and the limbs
// of the instance's first Residual group. A Residual group switch inside a
// window saves the leaving group (raw INT32) if it has work left in a later
// window, else completes it (MVOUT); the entering group is restored if saved,
// else zero-initialized, and its limbs are loaded unless prefetched. The
// group order of consecutive non-empty windows alternates direction, so the
// last group of a window is the first of the next one and stays in ACC-R.

#include "paired.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <tuple>

namespace ggml::potal {

namespace {

constexpr size_t   D   = paired_dim;
constexpr uint32_t PPB = (uint32_t) (paired_block_k / D);

uint32_t tiles_of(size_t elements) {
    return (uint32_t) ((elements + D - 1) / D);
}

struct res_item {
    bool      is_switch = false;
    paired_op tile;
};

} // namespace

const char * paired_op_name(paired_op_kind k) {
    switch (k) {
        case paired_op_kind::main_a_load: return "main_a_load";
        case paired_op_kind::residual_a_load: return "residual_a_load";
        case paired_op_kind::weight_load: return "weight_load";
        case paired_op_kind::scale_load: return "scale_load";
        case paired_op_kind::window_begin: return "window_begin";
        case paired_op_kind::weight_preload: return "weight_preload";
        case paired_op_kind::main_tile: return "main_tile";
        case paired_op_kind::residual_tile: return "residual_tile";
        case paired_op_kind::mode_switch: return "mode_switch";
        case paired_op_kind::group_init: return "group_init";
        case paired_op_kind::group_save: return "group_save";
        case paired_op_kind::group_restore: return "group_restore";
        case paired_op_kind::acc_complete: return "acc_complete";
        case paired_op_kind::window_end: return "window_end";
    }
    return "?";
}

const char * paired_path_name(paired_path p) {
    return p == paired_path::main ? "MAIN" : p == paired_path::residual ? "RESIDUAL" : "-";
}

std::string paired_op_text(const paired_op & op) {
    char buf[256];
    int  n = std::snprintf(buf, sizeof(buf), "%-15s %-8s s%u jg%u w%u pair%u", paired_op_name(op.kind),
                           paired_path_name(op.path), op.stripe, op.j_group, op.window, op.pair);
    const auto add = [&](const char * fmt, auto... v) {
        if (n >= 0 && (size_t) n < sizeof(buf)) {
            n += std::snprintf(buf + n, sizeof(buf) - (size_t) n, fmt, v...);
        }
    };
    if (op.group != paired_none) {
        add(" g%u", op.group);
    }
    switch (op.kind) {
        case paired_op_kind::main_tile:
        case paired_op_kind::residual_tile:
        case paired_op_kind::weight_preload:
            if (op.path == paired_path::main) {
                add(" blk%u k%u+%u", op.block_id, op.k_begin, op.k_count);
            } else {
                add(" frag%u blk%u k#%u", op.fragment, op.block_id, op.k_count);
            }
            add(" j%u sp_w=%u", op.j, op.sp_w);
            if (op.kind != paired_op_kind::weight_preload) {
                add(" sp_a=%u acc%u@%u slot%u%s%s%s", op.sp_a, op.acc_bank, op.acc_addr, op.scale_slot,
                    op.first ? " first" : "", op.last ? " last" : "", op.tail ? " tail" : "");
            }
            break;
        case paired_op_kind::main_a_load:
        case paired_op_kind::residual_a_load:
            add(" rows=%u sp_a=%u stride=%u", op.rows, op.sp_a, op.sp_a_stride);
            break;
        case paired_op_kind::weight_load:
            add(" k%u+%u cols=%u sp_w=%u", op.k_begin, op.k_count, op.cols, op.sp_w);
            break;
        case paired_op_kind::scale_load:
            add(" blk%u..+%u slots=%u", op.block_id, op.k_count, op.scale_slot);
            break;
        case paired_op_kind::group_init:
        case paired_op_kind::group_save:
        case paired_op_kind::group_restore:
        case paired_op_kind::acc_complete:
            add(" rows=%u cols=%u acc%u", op.rows, op.cols, op.acc_bank);
            break;
        default:
            break;
    }
    return buf;
}

paired_schedule build_paired_schedule(const paired_tile_plan & plan, size_t stripe,
                                      const paired_residual_request & residual) {
    paired_schedule sch;
    sch.stripe                = stripe;
    const size_t   rows       = paired_stripe_rows(plan, stripe);
    const uint32_t i_tiles    = tiles_of(rows);
    const size_t   W          = plan.k_windows;
    const size_t   U          = plan.j_groups * W;
    const size_t   G          = residual.groups.size();
    const size_t   frag_count = residual.fragments.size();
    sch.group_order.resize(U);

    // Window fragment ranges (fragments are in block order) and per-group work.
    std::vector<size_t>                           wf0(W), wf1(W), wb0(W), wb1(W);
    std::vector<std::vector<std::vector<uint32_t>>> work(G, std::vector<std::vector<uint32_t>>(W));
    std::vector<uint32_t>                         total(G, 0);
    for (size_t w = 0, f = 0; w < W; ++w) {
        paired_window_blocks(plan, w, wb0[w], wb1[w]);
        while (f < frag_count && residual.fragments[f].block_id < wb0[w]) {
            ++f;
        }
        wf0[w] = f;
        while (f < frag_count && residual.fragments[f].block_id < wb1[w]) {
            ++f;
        }
        wf1[w] = f;
        for (size_t g = 0; g < G; ++g) {
            for (size_t x = wf0[w]; x < wf1[w]; ++x) {
                if (residual.groups[g].nonzero[x]) {
                    work[g][w].push_back((uint32_t) x);
                    total[g]++;
                }
            }
        }
    }
    // Group order per window: alternate direction over the non-empty windows.
    std::vector<std::vector<uint32_t>> order(W);
    bool                               forward = true;
    for (size_t w = 0; w < W; ++w) {
        for (size_t x = 0; x < G; ++x) {
            const uint32_t g = (uint32_t) (forward ? x : G - 1 - x);
            if (!work[g][w].empty()) {
                order[w].push_back(g);
            }
        }
        forward = order[w].empty() ? forward : !forward;
    }
    for (size_t u = 0; u < U; ++u) {
        sch.group_order[u] = order[u % W];
    }

    const auto base = [&](paired_op_kind kind, paired_path path, size_t u) {
        paired_op op;
        op.kind    = kind;
        op.path    = path;
        op.stripe  = (uint32_t) stripe;
        op.j_group = (uint32_t) (u / W);
        op.window  = (uint32_t) (u % W);
        op.pair    = (uint32_t) (u % paired_sp_pairs);
        return op;
    };
    const auto cols_of = [&](size_t u) { return (uint32_t) paired_group_cols(plan, u / W); };
    const auto passes_of = [&](size_t u) { return (uint32_t) ((wb1[u % W] - wb0[u % W]) * PPB); };
    const auto weight_base = [&](size_t u) {
        return (uint32_t) (paired_layout(u % paired_sp_pairs).weight_end - passes_of(u) * tiles_of(cols_of(u)) * D);
    };
    const auto group_op = [&](paired_op_kind kind, size_t u, uint32_t g) {
        paired_op op = base(kind, paired_path::residual, u);
        op.group     = g;
        op.rows      = (uint32_t) residual.groups[g].rows;
        op.i_tiles   = tiles_of(op.rows);
        op.cols      = cols_of(u);
        op.acc_bank  = 1;
        op.acc_addr  = (uint32_t) paired_acc_bank_rows;
        return op;
    };
    const auto residual_load = [&](size_t u, uint32_t g) {
        paired_op op   = group_op(paired_op_kind::residual_a_load, u, g);
        const size_t w = u % W;
        op.fragment    = (uint32_t) wf0[w];
        op.k_count     = (uint32_t) (wf1[w] - wf0[w]); // fragments of the window
        op.sp_a        = (uint32_t) paired_layout(op.pair).residual_a_base;
        op.sp_a_stride = op.k_count * (uint32_t) D;
        return op;
    };
    const auto emit_loads = [&](size_t u) {
        const size_t           w   = u % W;
        const paired_sp_layout l   = paired_layout(u % paired_sp_pairs);
        paired_op              a   = base(paired_op_kind::main_a_load, paired_path::main, u);
        a.rows                     = (uint32_t) rows;
        a.i_tiles                  = i_tiles;
        a.k_begin                  = (uint32_t) (wb0[w] * paired_block_k);
        a.k_count                  = passes_of(u) * (uint32_t) D;
        a.sp_a                     = (uint32_t) l.main_a_base;
        a.sp_a_stride              = passes_of(u) * (uint32_t) D;
        sch.ops.push_back(a);
        paired_op b                = base(paired_op_kind::weight_load, paired_path::none, u);
        b.k_begin                  = a.k_begin;
        b.k_count                  = a.k_count;
        b.cols                     = cols_of(u);
        b.sp_w                     = weight_base(u);
        sch.ops.push_back(b);
        paired_op s                = base(paired_op_kind::scale_load, paired_path::none, u);
        s.block_id                 = (uint32_t) wb0[w];
        s.k_count                  = (uint32_t) (wb1[w] - wb0[w]);
        s.cols                     = cols_of(u);
        s.scale_slot               = s.k_count * tiles_of(s.cols);
        sch.ops.push_back(s);
        if (!order[w].empty()) {
            sch.ops.push_back(residual_load(u, order[w][0]));
        }
    };

    if (U != 0) {
        emit_loads(0);
    }
    uint32_t              resident = paired_none;
    std::vector<uint8_t>  saved(G, 0);
    std::vector<uint32_t> remaining(G, 0), seq(G, 0);
    for (size_t u = 0; u < U; ++u) {
        const size_t   w      = u % W;
        const uint32_t cols   = cols_of(u);
        const uint32_t j_run  = tiles_of(cols);
        const uint32_t passes = passes_of(u);
        const uint32_t wbase  = weight_base(u);
        const auto     l      = paired_layout(u % paired_sp_pairs);
        if (w == 0) {
            std::copy(total.begin(), total.end(), remaining.begin());
            std::fill(seq.begin(), seq.end(), 0);
            std::fill(saved.begin(), saved.end(), 0);
        }

        sch.ops.push_back(base(paired_op_kind::window_begin, paired_path::none, u));
        if (u + 1 < U) {
            emit_loads(u + 1);
        }

        // Main stream: (pass, N tile), K order.
        std::vector<paired_op> main;
        for (uint32_t p = 0; p < passes; ++p) {
            for (uint32_t j = 0; j < j_run; ++j) {
                paired_op op   = base(paired_op_kind::main_tile, paired_path::main, u);
                op.k_begin     = (uint32_t) (wb0[w] * paired_block_k) + p * (uint32_t) D;
                op.k_count     = (uint32_t) D;
                op.block_id    = op.k_begin / (uint32_t) paired_block_k;
                op.j           = j;
                op.rows        = (uint32_t) rows;
                op.i_tiles     = i_tiles;
                op.cols        = std::min<uint32_t>((uint32_t) D, cols - j * (uint32_t) D);
                op.sp_a        = (uint32_t) l.main_a_base + p * (uint32_t) D;
                op.sp_a_stride = passes * (uint32_t) D;
                op.sp_w        = wbase + (p * j_run + j) * (uint32_t) D;
                op.acc_bank    = 0;
                op.acc_addr    = j * (uint32_t) D;
                op.scale_slot  = (op.block_id - (uint32_t) wb0[w]) * j_run + j;
                op.first       = w == 0 && p == 0;
                op.last        = w + 1 == W && p + 1 == passes;
                main.push_back(op);
            }
        }
        // Residual stream: groups in order, each (fragment, N tile).
        std::vector<res_item> res;
        for (uint32_t g : order[w]) {
            res.push_back({ true, group_op(paired_op_kind::group_init, u, g) });
            for (size_t xi = 0; xi < work[g][w].size(); ++xi) {
                const uint32_t          x  = work[g][w][xi];
                const paired_fragment & fr = residual.fragments[x];
                for (uint32_t j = 0; j < j_run; ++j) {
                    paired_op op   = group_op(paired_op_kind::residual_tile, u, g);
                    op.fragment    = x;
                    op.block_id    = fr.block_id;
                    op.k_count     = fr.count;
                    op.j           = j;
                    op.cols        = std::min<uint32_t>((uint32_t) D, cols - j * (uint32_t) D);
                    op.sp_a        = (uint32_t) l.residual_a_base + (uint32_t) (x - wf0[w]) * (uint32_t) D;
                    op.sp_a_stride = (uint32_t) (wf1[w] - wf0[w]) * (uint32_t) D;
                    op.sp_w        = wbase;
                    op.acc_addr    = (uint32_t) paired_acc_bank_rows + j * (uint32_t) D;
                    op.scale_slot  = (fr.block_id - (uint32_t) wb0[w]) * j_run + j;
                    op.first       = seq[g] + xi == 0;
                    op.last        = seq[g] + xi + 1 == total[g];
                    res.push_back({ false, op });
                }
            }
            seq[g] += (uint32_t) work[g][w].size();
        }

        // Residual group exit: save if it has work later, else complete.
        const auto exit_group = [&](uint32_t g) {
            if (remaining[g] > 0) {
                sch.ops.push_back(group_op(paired_op_kind::group_save, u, g));
                saved[g] = 1;
            } else {
                sch.ops.push_back(group_op(paired_op_kind::acc_complete, u, g));
            }
        };
        const auto enter_group = [&](uint32_t g, bool prefetched) {
            if (resident == g) {
                return; // the previous window's last group: kept in ACC-R
            }
            if (resident != paired_none) {
                exit_group(resident);
            }
            if (!prefetched) {
                sch.ops.push_back(residual_load(u, g));
            }
            if (saved[g]) {
                sch.ops.push_back(group_op(paired_op_kind::group_restore, u, g));
                saved[g] = 0;
            } else {
                sch.ops.push_back(group_op(paired_op_kind::group_init, u, g));
            }
            resident = g;
        };

        // Interleave 1:1, Main first; then whatever is left.
        // Tiles beyond the 2 * min(Main, Residual) alternating ones are a path's tail.
        size_t res_count = 0;
        for (const res_item & it : res) {
            res_count += it.is_switch ? 0 : 1;
        }
        const size_t alternating = 2 * std::min(main.size(), res_count);
        size_t       emitted     = 0;
        size_t      mi = 0, ri = 0;
        bool        take_main     = true;
        bool        first_switch  = true;
        paired_path last          = paired_path::none;
        const auto  res_tiles_left = [&]() { return ri < res.size(); };
        while (mi < main.size() || res_tiles_left()) {
            const bool        main_left = mi < main.size();
            const bool        res_left  = res_tiles_left();
            const paired_path next = main_left && (take_main || !res_left) ? paired_path::main : paired_path::residual;
            if (last != paired_path::none && last != next) {
                sch.ops.push_back(base(paired_op_kind::mode_switch, next, u));
            }
            paired_op tile;
            if (next == paired_path::main) {
                tile = main[mi++];
            } else {
                while (res[ri].is_switch) {
                    enter_group(res[ri].tile.group, first_switch);
                    first_switch = false;
                    ++ri;
                }
                tile = res[ri++].tile;
                if (tile.j + 1 == j_run) {
                    remaining[tile.group]--; // remaining counts fragments: this was the last N tile
                }
            }
            tile.tail     = emitted++ >= alternating;
            paired_op pre = tile;
            pre.kind      = paired_op_kind::weight_preload;
            sch.ops.push_back(pre);
            sch.ops.push_back(tile);
            take_main = next != paired_path::main;
            last      = next;
        }

        if (w + 1 == W) {
            paired_op done = base(paired_op_kind::acc_complete, paired_path::main, u);
            done.rows      = (uint32_t) rows;
            done.i_tiles   = i_tiles;
            done.cols      = cols;
            done.acc_bank  = 0;
            sch.ops.push_back(done);
            if (resident != paired_none) {
                exit_group(resident);
                resident = paired_none;
            }
        }
        sch.ops.push_back(base(paired_op_kind::window_end, paired_path::none, u));
    }
    return sch;
}

potal_status validate_paired_schedule(const paired_tile_plan & plan, const paired_residual_request & residual,
                                      const paired_schedule & schedule, size_t * peak_spill_bytes,
                                      std::string * why) {
    const auto bad = [&](const std::string & what, size_t index) {
        if (why != nullptr) {
            *why = what + " at op " + std::to_string(index) +
                   (index < schedule.ops.size() ? ": " + paired_op_text(schedule.ops[index]) : std::string());
        }
        return POTAL_ERR_INVALID_ARGUMENT;
    };
    const size_t W = plan.k_windows;
    const size_t U = plan.j_groups * W;
    const size_t G = residual.groups.size();

    // Remaining tiles per (instance, path), for the alternation rule.
    std::vector<size_t> left_main(U, 0), left_res(U, 0), weight_loads(U, 0), alternating(U, 0), seen(U, 0);
    for (const paired_op & op : schedule.ops) {
        const size_t u = (size_t) op.j_group * W + op.window;
        if (u >= U) {
            return bad("window instance out of range", 0);
        }
        left_main[u] += op.kind == paired_op_kind::main_tile;
        left_res[u] += op.kind == paired_op_kind::residual_tile;
    }
    for (size_t u = 0; u < U; ++u) {
        alternating[u] = 2 * std::min(left_main[u], left_res[u]);
    }
    std::vector<uint32_t> group_total(G, 0);
    for (size_t g = 0; g < G; ++g) {
        for (uint8_t nz : residual.groups[g].nonzero) {
            group_total[g] += nz;
        }
    }

    std::map<std::tuple<uint32_t, uint32_t>, uint32_t>           main_next;   // (j_group, j) -> next K
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, int64_t>  res_last;    // (j_group, g, j) -> last fragment
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> res_seen;
    std::map<std::pair<uint32_t, uint32_t>, size_t>              saved;       // (j_group, g) -> bytes
    size_t                                                       spill = 0, peak = 0;
    paired_path                                                  last_path = paired_path::none;
    size_t                                                       last_u    = SIZE_MAX;
    const paired_op *                                            preload   = nullptr;

    for (size_t i = 0; i < schedule.ops.size(); ++i) {
        const paired_op &      op = schedule.ops[i];
        const size_t           u  = (size_t) op.j_group * W + op.window;
        const paired_sp_layout l  = paired_layout(op.pair);
        if (op.pair != u % paired_sp_pairs || op.stripe != schedule.stripe) {
            return bad("pair / stripe mismatch", i);
        }
        if (u != last_u && (op.kind == paired_op_kind::main_tile || op.kind == paired_op_kind::residual_tile)) {
            last_path = paired_path::none;
            last_u    = u;
        }
        switch (op.kind) {
            case paired_op_kind::main_a_load:
                if (op.sp_a != l.main_a_base || (size_t) op.i_tiles * op.sp_a_stride > paired_partition_rows) {
                    return bad("Main activation outside its partition", i);
                }
                break;
            case paired_op_kind::residual_a_load:
                if (op.group >= G || op.sp_a != l.residual_a_base ||
                    (size_t) op.i_tiles * op.sp_a_stride > paired_partition_rows) {
                    return bad("Residual limbs outside their partition", i);
                }
                break;
            case paired_op_kind::weight_load: {
                const size_t rows = (size_t) (op.k_count / paired_dim) * ((op.cols + paired_dim - 1) / paired_dim) *
                                    paired_dim;
                if (op.sp_w < l.weight_bank || op.sp_w + rows != l.weight_end) {
                    return bad("weight not rear-placed in its bank", i);
                }
                if (++weight_loads[u] != 1) {
                    return bad("weight loaded twice in one window", i);
                }
                break;
            }
            case paired_op_kind::scale_load:
                if (op.scale_slot > scale_window_slots) {
                    return bad("scale window exceeded", i);
                }
                break;
            case paired_op_kind::weight_preload:
                preload = &op;
                break;
            case paired_op_kind::main_tile:
            case paired_op_kind::residual_tile: {
                const bool is_main = op.kind == paired_op_kind::main_tile;
                if (preload == nullptr || preload->path != op.path || preload->k_begin != op.k_begin ||
                    preload->fragment != op.fragment || preload->j != op.j || preload->j_group != op.j_group) {
                    return bad("micro-tile without its weight preload", i);
                }
                preload = nullptr;
                if (op.k_count == 0 || op.k_count > paired_dim || op.cols == 0 || op.rows == 0) {
                    return bad("dummy micro-tile", i);
                }
                // 1:1 while both have work: a repeated path needs the other one finished.
                size_t & mine  = is_main ? left_main[u] : left_res[u];
                size_t & other = is_main ? left_res[u] : left_main[u];
                if (last_path == op.path && other != 0) {
                    return bad("alternation broken", i);
                }
                if (op.tail != (seen[u]++ >= alternating[u])) {
                    return bad("tail flag", i);
                }
                mine--;
                last_path = op.path;
                if ((size_t) op.i_tiles * ((paired_group_cols(plan, op.j_group) + paired_dim - 1) / paired_dim) *
                        paired_dim > paired_acc_bank_rows) {
                    return bad("ACC tile exceeds its bank", i);
                }
                if (is_main) {
                    uint32_t & next = main_next[{ op.j_group, op.j }];
                    if (op.k_begin != next || op.first != (op.k_begin == 0) ||
                        op.last != (op.k_begin + paired_dim == plan.k) || op.acc_bank != 0) {
                        return bad("Main K order / contribution flags", i);
                    }
                    next += (uint32_t) paired_dim;
                } else {
                    if (op.group >= G || op.fragment >= residual.fragments.size() ||
                        !residual.groups[op.group].nonzero[op.fragment] || op.acc_bank != 1) {
                        return bad("dummy Residual micro-tile", i);
                    }
                    const auto key  = std::make_tuple(op.j_group, op.group, op.j);
                    auto       it   = res_last.find(key);
                    const bool none = it == res_last.end();
                    if ((!none && (int64_t) op.fragment <= it->second) || op.first != none) {
                        return bad("Residual K order / first flag", i);
                    }
                    res_last[key] = op.fragment;
                    if (op.last != (++res_seen[key] == group_total[op.group])) {
                        return bad("Residual last flag", i);
                    }
                }
                break;
            }
            case paired_op_kind::group_save: {
                const size_t bytes = (size_t) op.rows * op.cols * sizeof(int32_t);
                if (!saved.emplace(std::make_pair(op.j_group, op.group), bytes).second) {
                    return bad("group saved twice", i);
                }
                spill += bytes;
                peak = std::max(peak, spill);
                break;
            }
            case paired_op_kind::group_restore: {
                auto it = saved.find({ op.j_group, op.group });
                if (it == saved.end()) {
                    return bad("restore without a save", i);
                }
                spill -= it->second;
                saved.erase(it);
                break;
            }
            default:
                break;
        }
    }
    for (size_t u = 0; u < U; ++u) {
        if (weight_loads[u] != 1) {
            return bad("window instance without its weight load", u);
        }
    }
    if (!saved.empty()) {
        return bad("saved group never restored", schedule.ops.size());
    }
    if (peak_spill_bytes != nullptr) {
        *peak_spill_bytes = peak;
    }
    return POTAL_OK;
}

} // namespace ggml::potal
