// Hardware-free tests of the paired micro-tile schedule (matmul/paired.hpp,
// P3 / P5), on the real schedule trace, not on flags:
//
//   T  representative 2-stripe / 3-group case (printed): Main/Residual 1:1
//      alternation, the Residual-only tail, one shared weight load per window
//      read by both paths, K-window switches with prefetch into the other
//      pair, group order reversed between windows, ACC save/restore
//   A  alternation cases: equal micro-tile counts (no tail), Main finishes
//      first, Residual finishes first, Main only (no Residual op at all)
//   G  one group: no save/restore; several groups over several windows: the
//      reversal saves transfers against a same-direction traversal; a group
//      with an all-zero window is skipped (no load, no transfer), no dummy
//   K  per-output K order: Main passes ascending per (N group, N tile),
//      Residual fragments ascending per (group, N tile), first/last flags
//   D  deterministic: the same descriptor gives the same trace
//   V  the static validator rejects a broken K order, a missing preload, a
//      second weight load and a broken alternation

#include "support/paired-support.hpp"

#include <map>
#include <set>
#include <tuple>

using namespace ggml::potal;
using namespace potal_test;

namespace {

using kind = paired_op_kind;

struct case_data {
    std::unique_ptr<paired_fixture> f;
    paired_tile_plan                plan;
    paired_inputs                   in;
    std::vector<paired_residual_request> req;
    std::vector<paired_schedule>    sch;
    bool                            ok = false;
};

// Frozen tiles so the trace stays small: stripe rows 32, two N groups, two K windows.
bool make_case(case_data & c, size_t M, size_t N, size_t K, size_t ti, size_t tj, size_t tk, const entries_fn & e) {
    c.f.reset(new paired_fixture(M, N, K, 11, [](size_t blk, size_t n) { return (int16_t) ((blk + n) % 3); }));
    if (!c.f->ok || paired_frozen_plan(M, N, K, ti, tj, tk, c.plan) != POTAL_OK ||
        !make_inputs(*c.f, c.plan.stripe_rows, e, c.in)) {
        return false;
    }
    c.req.resize(c.plan.stripes);
    for (size_t s = 0; s < c.plan.stripes; ++s) {
        if (c.in.packets[s] != nullptr &&
            build_paired_residual_request(*c.in.packets[s], c.f->weight, c.plan.stripe_rows, c.req[s]) !=
                grmd::RmdStatus::success) {
            return false;
        }
        c.sch.push_back(build_paired_schedule(c.plan, s, c.req[s]));
        std::string why;
        if (validate_paired_schedule(c.plan, c.req[s], c.sch[s], nullptr, &why) != POTAL_OK) {
            std::printf("  validator: %s\n", why.c_str());
            return false;
        }
    }
    c.ok = true;
    return true;
}

size_t count(const paired_schedule & s, kind k, uint32_t jg = paired_none) {
    size_t n = 0;
    for (const paired_op & op : s.ops) {
        n += op.kind == k && (jg == paired_none || op.j_group == jg);
    }
    return n;
}

// Tile path sequence of one window instance: "MRMR..".
std::string paths(const paired_schedule & s, uint32_t jg, uint32_t w) {
    std::string p;
    for (const paired_op & op : s.ops) {
        if (op.j_group == jg && op.window == w && (op.kind == kind::main_tile || op.kind == kind::residual_tile)) {
            p += op.kind == kind::main_tile ? 'M' : 'R';
        }
    }
    return p;
}

// Strict alternation while both have work, then one path only.
bool alternates(const std::string & p) {
    size_t m = std::count(p.begin(), p.end(), 'M'), r = std::count(p.begin(), p.end(), 'R');
    const size_t both = 2 * std::min(m, r);
    for (size_t i = 0; i < both; ++i) {
        if (p[i] != (i % 2 == 0 ? 'M' : 'R')) {
            return false;
        }
    }
    return p.find_first_not_of(m > r ? 'M' : 'R', both) == std::string::npos;
}

// Rows with residuals in three lanes (0x10101 + r) on blocks 0 and 3: 3 lane rows each.
std::vector<res_entry> three_lane_entries(size_t rows) {
    std::vector<res_entry> e;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t i = 0; i < 17; ++i) {
            e.push_back({ r, i, 0x10101 + (int32_t) (r * 3 + i) });
        }
        for (size_t i = 0; i < 3; ++i) {
            e.push_back({ r, 96 + 4 * i, -(0x20202 + (int32_t) r) });
        }
    }
    return e;
}

void test_trace() {
    case_data c;
    const bool made = make_case(c, 48, 48, 128, 2, 2, 4, [](size_t s, size_t rows) {
        if (s == 0) {
            return three_lane_entries(rows);
        }
        std::vector<res_entry> e; // stripe 1: lane 0 only, one group
        for (size_t r = 0; r < rows; ++r) {
            e.push_back({ r, 40 + r % 8, (int32_t) (r % 2 ? -7 : 9) });
        }
        return e;
    });
    check("T 2-stripe / 3-group case built and validated", made);
    if (!made) {
        return;
    }
    const paired_schedule & s0 = c.sch[0];
    std::printf("  plan: stripe_rows=%zu stripes=%zu j_groups=%zu k_windows=%zu (window %zu blocks); stripe 0: %zu "
                "lane rows in %zu groups, %zu fragments; stripe 1: %zu lane rows in %zu group\n",
                c.plan.stripe_rows, c.plan.stripes, c.plan.j_groups, c.plan.k_windows, c.plan.window_blocks,
                c.req[0].rows.size(), c.req[0].groups.size(), c.req[0].fragments.size(), c.req[1].rows.size(),
                c.req[1].groups.size());
    std::printf("  ---- TRACE stripe 0, N group 0 (both K windows) ----\n");
    for (const paired_op & op : s0.ops) {
        if (op.j_group == 0 || (op.j_group == 1 && op.window == 0 && op.kind != kind::window_begin &&
                                (op.kind == kind::main_a_load || op.kind == kind::weight_load ||
                                 op.kind == kind::scale_load || op.kind == kind::residual_a_load))) {
            std::printf("  %s\n", paired_op_text(op).c_str());
        }
    }
    std::printf("  ---- end of trace (N group 1 and stripe 1 omitted) ----\n");

    check("T stripe 0: 3 groups of 32 lane rows; stripe 1: 1 group",
          c.req[0].groups.size() == 3 && c.req[1].groups.size() == 1);
    const std::string w0 = paths(s0, 0, 0), w1 = paths(s0, 0, 1);
    std::printf("  tile paths: window 0 %s | window 1 %s\n", w0.c_str(), w1.c_str());
    check("T 1:1 alternation then the Residual-only tail (Residual has more micro-tiles)",
          alternates(w0) && alternates(w1) && w0.back() == 'R' && std::count(w0.begin(), w0.end(), 'R') >
                                                                       std::count(w0.begin(), w0.end(), 'M'));
    check("T group order reversed: window 0 G0 G1 G2, window 1 G2 G1 G0",
          s0.group_order[0] == std::vector<uint32_t>({ 0, 1, 2 }) &&
              s0.group_order[1] == std::vector<uint32_t>({ 2, 1, 0 }));
    // Window 0 saves G0, G1 (work left in window 1); G2 stays in ACC-R across the switch.
    std::vector<uint32_t> saves, restores;
    bool                  g2_kept = true;
    bool                  in_w1   = false;
    for (const paired_op & op : s0.ops) {
        if (op.j_group != 0) {
            continue;
        }
        if (op.kind == kind::group_save) {
            saves.push_back(op.group);
        }
        if (op.kind == kind::group_restore) {
            restores.push_back(op.group);
        }
        if (op.kind == kind::window_begin && op.window == 1) {
            in_w1 = true;
        }
        if (in_w1 && op.group == 2 && (op.kind == kind::group_restore || op.kind == kind::group_init)) {
            g2_kept = false;
        }
    }
    check("T ACC save/restore: saves G0 G1 in window 0, restores G1 G0 in window 1, G2 kept resident",
          saves == std::vector<uint32_t>({ 0, 1 }) && restores == std::vector<uint32_t>({ 1, 0 }) && g2_kept);
    check("T one shared weight load per window instance, Main and Residual preloads from that bank",
          count(s0, kind::weight_load) == c.plan.j_groups * c.plan.k_windows && [&] {
              for (const paired_op & op : s0.ops) {
                  if (op.kind == kind::weight_preload && op.sp_w < paired_layout(op.pair).weight_bank) {
                      return false;
                  }
              }
              return true;
          }());
    // K-window switch: the next instance is loaded into the other pair while this one is active.
    bool prefetch = true;
    int  active   = -1;
    for (const paired_op & op : s0.ops) {
        if (op.kind == kind::window_begin) {
            active = (int) op.pair;
        } else if (op.kind == kind::window_end) {
            active = -1;
        } else if ((op.kind == kind::main_a_load || op.kind == kind::weight_load) && active == (int) op.pair) {
            prefetch = false;
        }
    }
    check("T K-window switches alternate SP pairs; loads never target the ACTIVE pair",
          prefetch && count(s0, kind::window_begin) == c.plan.j_groups * c.plan.k_windows);

    // The emulator runs it: report counters agree with the trace.
    paired_gemm_desc        d = desc_of(*c.f, c.plan, c.in);
    paired_output           out;
    paired_execution_report rep;
    const potal_status      st = paired_tiled_matmul_auto(d, out, rep);
    const paired_counts &   n  = rep.stripes.empty() ? rep.totals : rep.stripes[0].counts;
    std::printf("  stripe 0 (emulated): main %llu, residual %llu micro-tiles, %llu mode switches, residual-only "
                "tail %llu, main-only tail %llu, dummy %llu, saves %llu (%llu B), restores %llu (%llu B), weight "
                "loads %llu, preloads %llu, indexed reads %llu\n",
                (unsigned long long) n.main_micro_tiles, (unsigned long long) n.residual_micro_tiles,
                (unsigned long long) n.mode_switches, (unsigned long long) n.residual_only_tail_micro_tiles,
                (unsigned long long) n.main_only_tail_micro_tiles, (unsigned long long) n.dummy_micro_tiles,
                (unsigned long long) n.residual_acc_saves, (unsigned long long) n.residual_acc_save_bytes,
                (unsigned long long) n.residual_acc_restores, (unsigned long long) n.residual_acc_restore_bytes,
                (unsigned long long) n.logical_weight_loads, (unsigned long long) n.logical_weight_preloads,
                (unsigned long long) n.indexed_weight_reads);
    check("T emulated counts == trace counts (micro-tiles, switches, saves, no dummy)",
          st == POTAL_OK && n.main_micro_tiles == count(s0, kind::main_tile) &&
              n.residual_micro_tiles == count(s0, kind::residual_tile) &&
              n.mode_switches == count(s0, kind::mode_switch) && n.residual_acc_saves == count(s0, kind::group_save) &&
              n.dummy_micro_tiles == 0 && n.shared_weight_windows == c.plan.j_groups * c.plan.k_windows);
}

void test_alternation() {
    // Equal counts: one K32 block per window (2 passes) x 1 N tile = 2 Main; one run of 17 K = 2 fragments.
    {
        case_data c;
        const bool ok = make_case(c, 16, 16, 64, 1, 1, 2, [](size_t, size_t rows) {
            std::vector<res_entry> e;
            for (size_t i = 0; i < 17; ++i) {
                e.push_back({ i % rows, i, 5 });
                e.push_back({ i % rows, 32 + i, -3 });
            }
            return e;
        });
        const std::string p = ok ? paths(c.sch[0], 0, 0) + "|" + paths(c.sch[0], 0, 1) : "";
        check("A equal micro-tile counts: MRMR in each window, no tail, no dummy (" + p + ")",
              ok && p == "MRMR|MRMR" && [&] {
                  for (const paired_op & op : c.sch[0].ops) {
                      if (op.tail) {
                          return false;
                      }
                  }
                  return true;
              }());
    }
    // Residual first: one fragment against 8 Main micro-tiles.
    {
        case_data c;
        const bool ok = make_case(c, 32, 64, 64, 2, 4, 2, [](size_t, size_t) {
            return std::vector<res_entry>{ { 0, 1, 7 } };
        });
        const std::string p = ok ? paths(c.sch[0], 0, 0) : "";
        check("A Residual finishes first: MRMRMRMR then Main only (" + p + ")",
              ok && p == "MRMRMRMRMMMM" && alternates(p));
    }
    // Main only: no packet.
    {
        case_data c;
        const bool ok = make_case(c, 32, 32, 64, 2, 2, 2, {});
        check("A Main only: no Residual op, no mode switch, every Main tile is a tail",
              ok && count(c.sch[0], kind::residual_tile) == 0 && count(c.sch[0], kind::mode_switch) == 0 &&
                  count(c.sch[0], kind::residual_a_load) == 0 && count(c.sch[0], kind::group_init) == 0 &&
                  alternates(paths(c.sch[0], 0, 0)));
    }
}

void test_groups() {
    // One group over 4 windows: never saved.
    {
        case_data c;
        const bool ok = make_case(c, 16, 16, 128, 1, 1, 2, [](size_t, size_t rows) {
            std::vector<res_entry> e;
            for (size_t b = 0; b < 4; ++b) {
                e.push_back({ b % rows, b * 32 + 5, 11 });
            }
            return e;
        });
        check("G one group across 4 K windows: no save / restore, one init, one completion",
              ok && c.req[0].groups.size() == 1 && count(c.sch[0], kind::group_save) == 0 &&
                  count(c.sch[0], kind::group_restore) == 0 && count(c.sch[0], kind::group_init) == 1);
    }
    // Four groups (16 rows x 4 lanes... 64 lane rows / 16) over 4 windows.
    {
        case_data c;
        const bool ok = make_case(c, 16, 16, 128, 1, 1, 2, [](size_t, size_t rows) {
            std::vector<res_entry> e;
            for (size_t r = 0; r < rows; ++r) {
                for (size_t b = 0; b < 4; ++b) {
                    e.push_back({ r, b * 32 + r % 32, 0x1010101 + (int32_t) r });
                }
            }
            return e;
        });
        const size_t G = ok ? c.req[0].groups.size() : 0, W = c.plan.k_windows;
        const size_t saves = ok ? count(c.sch[0], kind::group_save) : 0;
        // Same-direction traversal: every window boundary evicts the last group too.
        const size_t naive = G * (W - 1);
        std::printf("  %zu groups x %zu windows: %zu saves / %zu restores with reversal, %zu with a same-direction "
                    "traversal\n",
                    G, W, saves, ok ? count(c.sch[0], kind::group_restore) : 0, naive);
        check("G several groups over several windows: reversal keeps one group per boundary (saves = (G-1)(W-1))",
              ok && G >= 3 && saves == (G - 1) * (W - 1) && saves < naive &&
                  count(c.sch[0], kind::group_restore) == saves);
    }
    // A group whose rows have no digit in window 1 (lane-1 rows only see block 0).
    {
        case_data c;
        const bool ok = make_case(c, 16, 16, 128, 1, 1, 2, [](size_t, size_t rows) {
            std::vector<res_entry> e;
            for (size_t r = 0; r < rows; ++r) {
                e.push_back({ r, 3, 300 + (int32_t) r }); // block 0: lanes 0 and 1
                e.push_back({ r, 70, 5 });                 // block 2: lane 0 only
            }
            return e;
        });
        bool skipped = ok && c.req[0].groups.size() == 2;
        for (const paired_op & op : ok ? c.sch[0].ops : std::vector<paired_op>{}) {
            if (op.group == 1 && op.window == 1 && op.kind != kind::window_begin) {
                skipped = false; // group 1 (lane 1) has nothing in window 1: no load, tile or transfer there
            }
        }
        check("G empty (group, window): no load, no tile, no transfer; the group completes from its last window",
              skipped && count(c.sch[0], kind::acc_complete) == 3);
    }
}

void test_k_order_and_determinism() {
    case_data c;
    const bool ok = make_case(c, 48, 48, 128, 2, 2, 4, [](size_t, size_t rows) { return three_lane_entries(rows); });
    bool       order = ok;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, int64_t> last_main, last_res;
    for (const paired_schedule & s : c.sch) {
        last_main.clear();
        last_res.clear();
        for (const paired_op & op : s.ops) {
            if (op.kind == kind::main_tile) {
                auto key = std::make_tuple(op.j_group, op.j, 0u);
                auto it  = last_main.find(key);
                order    = order && (it == last_main.end() ? op.k_begin == 0 && op.first
                                                           : (int64_t) op.k_begin == it->second + (int64_t) PD);
                last_main[key] = op.k_begin;
            } else if (op.kind == kind::residual_tile) {
                auto key = std::make_tuple(op.j_group, op.group, op.j);
                auto it  = last_res.find(key);
                order    = order && (it == last_res.end() ? op.first : (int64_t) op.fragment > it->second);
                last_res[key] = op.fragment;
            }
        }
    }
    check("K per-output K order kept: Main passes contiguous ascending, Residual fragments ascending", order);

    case_data c2;
    const bool ok2  = make_case(c2, 48, 48, 128, 2, 2, 4, [](size_t, size_t rows) { return three_lane_entries(rows); });
    bool       same = ok && ok2 && c.sch.size() == c2.sch.size();
    for (size_t s = 0; s < c.sch.size() && same; ++s) {
        same = c.sch[s].ops.size() == c2.sch[s].ops.size();
        for (size_t i = 0; i < c.sch[s].ops.size() && same; ++i) {
            same = paired_op_text(c.sch[s].ops[i]) == paired_op_text(c2.sch[s].ops[i]);
        }
        const paired_schedule again = build_paired_schedule(c.plan, s, c.req[s]);
        for (size_t i = 0; i < again.ops.size() && same; ++i) {
            same = paired_op_text(again.ops[i]) == paired_op_text(c.sch[s].ops[i]);
        }
    }
    check("D same descriptor -> identical trace (rebuilt and regenerated)", same);

    if (!ok) {
        return;
    }
    // V: tampered schedules.
    const paired_schedule & good = c.sch[0];
    std::string             why;
    const auto              find = [&](kind k, size_t nth) {
        for (size_t i = 0; i < good.ops.size(); ++i) {
            if (good.ops[i].kind == k && nth-- == 0) {
                return i;
            }
        }
        return good.ops.size();
    };
    paired_schedule k_swap = good; // two Main tiles of one N tile, K order swapped
    {
        const size_t a = find(kind::main_tile, 0), b = find(kind::main_tile, 2); // j0 pass 0 / pass 1
        std::swap(k_swap.ops[a], k_swap.ops[b]);
        std::swap(k_swap.ops[a - 1], k_swap.ops[b - 1]);
    }
    paired_schedule no_pre = good;
    no_pre.ops.erase(no_pre.ops.begin() + (long) find(kind::weight_preload, 3));
    paired_schedule twice = good;
    twice.ops.insert(twice.ops.begin() + (long) find(kind::weight_load, 0), good.ops[find(kind::weight_load, 0)]);
    paired_schedule alt = good; // two Main tiles back to back while Residual has work
    {
        const size_t r = find(kind::residual_tile, 0);
        alt.ops.erase(alt.ops.begin() + (long) r - 1, alt.ops.begin() + (long) r + 1);
    }
    check("V validator rejects: K order swap, missing preload, second weight load, broken alternation",
          validate_paired_schedule(c.plan, c.req[0], k_swap, nullptr, &why) != POTAL_OK &&
              validate_paired_schedule(c.plan, c.req[0], no_pre, nullptr, &why) != POTAL_OK &&
              validate_paired_schedule(c.plan, c.req[0], twice, nullptr, &why) != POTAL_OK &&
              validate_paired_schedule(c.plan, c.req[0], alt, nullptr, &why) != POTAL_OK);
}

} // namespace

int main() {
    test_trace();
    test_alternation();
    test_groups();
    test_k_order_and_determinism();
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
