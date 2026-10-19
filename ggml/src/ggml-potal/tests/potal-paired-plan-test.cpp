// Hardware-free tests of the paired plan and the shared-weight Residual
// descriptor (matmul/paired.hpp, P1 / P2):
//
//   G  geometry from potal_hw_params.h: 4 SP banks, Current/Next pairs,
//      Main/Residual activation halves 1:1, weight bank rear placement,
//      ACC bank 0 Main / bank 1 Residual = the half potal_tile_select uses
//   S  paired_tile_select on the test shapes: every paired limit holds,
//      stripe height = tile_I * DIM, K windows of whole K32 blocks, tails
//   F  frozen Main tiles: t_I / t_J / stripe height kept, only the K window
//      shrinks (in K32 steps); capacity failure instead of a smaller t_I/t_J
//   R  Residual descriptor == the serialized request's runs / rows / digits;
//      fragments of <= DIM K inside one run; original K indexing reads the
//      same weight rows the serialized path copies; no B copy; groups
//   E  invalid shapes / tiles / K index / carrier / stripe mapping / spill /
//      tampered SP addresses refused, with zero emulated ops

#include "support/paired-support.hpp"

#include <algorithm>
#include <set>

using namespace ggml::potal;
using namespace potal_test;

namespace {

constexpr uint32_t PPB = 32 / PD;

bool plan_limits_hold(const paired_tile_plan & p) {
    return p.tile_I * p.tile_K * PD <= p.partition_rows && p.tile_K * p.tile_J * PD <= p.weight_bank_rows &&
           p.tile_I * p.tile_J * PD <= p.acc_bank_rows && p.tile_I * p.tile_J <= POTAL_WORK_IDS_PER_LOOP &&
           p.scale_slots <= POTAL_SCALE_ENTRIES && p.tile_K % PPB == 0 && p.stripe_rows == p.tile_I * PD &&
           p.window_blocks * PPB == p.tile_K;
}

void print_plan(const char * name, const paired_tile_plan & p) {
    std::printf("  %s: M=%zu N=%zu K=%zu tile_I=%zu tile_J=%zu tile_K=%zu (requested %zu) stripe_rows=%zu "
                "stripes=%zu j_groups=%zu window_blocks=%zu k_windows=%zu | Main A %zu/%zu rows, Residual A <= %zu/%zu, "
                "W %zu/%zu, ACC %zu/%zu, scale slots %u\n",
                name, p.m, p.n, p.k, p.tile_I, p.tile_J, p.tile_K, p.requested_tile_K, p.stripe_rows, p.stripes,
                p.j_groups, p.window_blocks, p.k_windows, p.main_a_rows, p.partition_rows, p.residual_a_rows,
                p.partition_rows, p.weight_rows, p.weight_bank_rows, p.acc_rows, p.acc_bank_rows, p.scale_slots);
}

void test_geometry() {
    const paired_sp_layout a = paired_layout(0), b = paired_layout(1);
    check("G SP geometry from the generated header: 4 banks x 4096 rows x DIM16 = 256 KiB",
          paired_sp_banks == POTAL_HW_BANK_NUM && paired_sp_bank_rows == POTAL_HW_BANK_ROWS &&
              paired_sp_banks * paired_sp_bank_rows * PD == 256 * 1024);
    check("G pair 0 = banks 0/1, pair 1 = banks 2/3 (LoopMatmul loop halves)",
          a.main_a_base == 0 && a.weight_bank == paired_sp_bank_rows && a.weight_end == 2 * paired_sp_bank_rows &&
              b.main_a_base == 2 * paired_sp_bank_rows && b.weight_end == 4 * paired_sp_bank_rows);
    check("G Main / Residual activation halves 1:1, disjoint, inside the activation bank",
          a.residual_a_base - a.main_a_base == paired_partition_rows &&
              a.residual_a_base + paired_partition_rows == a.weight_bank &&
              2 * paired_partition_rows == paired_sp_bank_rows &&
              b.residual_a_base - b.main_a_base == paired_partition_rows);
    potal_tile_plan serial;
    const bool      ok = potal_tile_select(256, 256, 256, PPB, &serial) == POTAL_OK;
    check("G ACC bank 0 Main / bank 1 Residual: each = the ACC half potal_tile_select already uses (not halved again)",
          ok && paired_acc_banks * paired_acc_bank_rows == POTAL_HW_ACC_ROWS &&
              paired_acc_bank_rows == serial.max_acc_rows);
}

void test_select() {
    struct shape {
        size_t m, n, k;
    };
    const shape shapes[] = { { 150, 320, 128 }, { 200, 320, 128 }, { 100, 160, 2048 }, { 160, 320, 512 },
                             { 128, 3072, 768 }, { 1, 768, 768 }, { 48, 48, 160 } };
    bool all = true;
    for (const shape & s : shapes) {
        paired_tile_plan p;
        const bool       ok = paired_tile_select(s.m, s.n, s.k, p) == POTAL_OK && plan_limits_hold(p);
        print_plan("select", p);
        // Windows cover every K32 block once, in order; the last one may be partial.
        size_t next = 0;
        for (size_t w = 0; w < p.k_windows; ++w) {
            size_t b0 = 0, b1 = 0;
            paired_window_blocks(p, w, b0, b1);
            all = all && b0 == next && b1 > b0 && b1 - b0 <= p.window_blocks;
            next = b1;
        }
        size_t rows = 0, cols = 0;
        for (size_t st = 0; st < p.stripes; ++st) {
            rows += paired_stripe_rows(p, st);
        }
        for (size_t g = 0; g < p.j_groups; ++g) {
            cols += paired_group_cols(p, g);
        }
        all = all && ok && next == s.k / 32 && rows == s.m && cols == s.n;
    }
    check("S paired_tile_select: every paired limit, K32 windows covering K, tail M/N/K covered exactly", all);

    paired_tile_plan p;
    check("S tail shape M=150 N=320 K=128: tail stripe and tail N group sized by the remainder",
          paired_tile_select(150, 320, 128, p) == POTAL_OK &&
              paired_stripe_rows(p, p.stripes - 1) == 150 - (p.stripes - 1) * p.stripe_rows &&
              paired_group_cols(p, p.j_groups - 1) == 320 - (p.j_groups - 1) * p.tile_J * PD);
    check("S K not a multiple of 32 / empty shape refused",
          paired_tile_select(16, 16, 48, p) == POTAL_ERR_INVALID_ARGUMENT &&
              paired_tile_select(0, 16, 32, p) == POTAL_ERR_INVALID_ARGUMENT);
}

void test_frozen() {
    // The STRIPE_PIPELINE plan of these shapes (A+B share a loop half) fixes
    // the ExSIA stripe height; the paired plan keeps it and shrinks K only.
    bool all = true, shrunk = false;
    const size_t shapes[][3] = { { 128, 3072, 768 }, { 160, 320, 512 }, { 100, 160, 2048 }, { 200, 320, 128 } };
    for (const auto & s : shapes) {
        potal_tile_plan  serial;
        paired_tile_plan p;
        const bool ok = potal_tile_select(s[0], s[1], s[2], PPB, &serial) == POTAL_OK &&
                        paired_frozen_plan(s[0], s[1], s[2], serial.tile_I, serial.tile_J, serial.tile_K, p) == POTAL_OK;
        print_plan("frozen", p);
        std::printf("    serialized plan: tile_I=%zu tile_J=%zu tile_K=%zu\n", serial.tile_I, serial.tile_J,
                    serial.tile_K);
        all    = all && ok && p.frozen && p.tile_I == serial.tile_I && p.tile_J == serial.tile_J &&
              p.stripe_rows == serial.tile_I * PD && p.tile_K <= serial.tile_K && plan_limits_hold(p) &&
              (serial.tile_K - p.tile_K) % PPB == 0;
        shrunk = shrunk || p.tile_K < serial.tile_K;
    }
    check("F frozen t_I / t_J / stripe height kept, K window shrunk in K32 steps only", all);
    check("F at least one serialized plan needed the common K-window shrink", shrunk);

    paired_tile_plan p;
    // tile_I = 80: 80 * 2 * 16 = 2560 rows > the 2048-row Main partition even for one K32 block.
    check("F no K32 window fits the frozen t_I: POTAL_ERR_CAPACITY (no smaller t_I, no fallback)",
          paired_frozen_plan(80 * 16, 64, 64, 80, 1, 2, p) == POTAL_ERR_CAPACITY);
    check("F frozen t_I*t_J beyond one ACC bank: POTAL_ERR_CAPACITY",
          paired_frozen_plan(256, 256, 64, 8, 8, 2, p) == POTAL_ERR_CAPACITY);
    check("F frozen tile_K not whole K32 blocks / tile_J wider than N refused",
          paired_frozen_plan(64, 64, 64, 1, 1, 3, p) == POTAL_ERR_INVALID_ARGUMENT &&
              paired_frozen_plan(64, 64, 64, 1, 5, 2, p) == POTAL_ERR_INVALID_ARGUMENT);
}

// Compact run lengths 1, 15, 16, 17, 31, 32 over non-consecutive blocks, three lanes, signs.
std::vector<res_entry> edge_entries(size_t rows) {
    const size_t           blocks[] = { 0, 2, 3, 5, 6, 7 };
    const size_t           counts[] = { 1, 15, 16, 17, 31, 32 };
    std::vector<res_entry> e;
    for (size_t b = 0; b < 6; ++b) {
        for (size_t i = 0; i < counts[b]; ++i) {
            const size_t row = (b * 5 + i) % rows;
            const int32_t v  = (int32_t) ((i % 3 == 0 ? 3 : i % 3 == 1 ? 300 : 70000) + i) * (i % 2 ? -1 : 1);
            e.push_back({ row, blocks[b] * 32 + (counts[b] <= 16 ? i * 2 : i), v });
        }
    }
    return e;
}

void test_descriptor() {
    paired_fixture f(40, 48, 256, 7, [](size_t blk, size_t n) { return (int16_t) ((blk + n) % 5); });
    if (!f.ok) {
        check("R fixture", false);
        return;
    }
    grmd::StripePacketHandle packet = make_packet(0, 0, 32, f.K, f.N, edge_entries(32));
    residual::request        serial;
    paired_residual_request  pr;
    const bool built = packet != nullptr && residual::build_request(*packet, f.weight, serial) == grmd::RmdStatus::success &&
                       build_paired_residual_request(*packet, f.weight, 32, pr) == grmd::RmdStatus::success;
    check("R both descriptors built from one packet", built);
    if (!built) {
        return;
    }
    bool same_runs = pr.runs.size() == serial.runs.size();
    for (size_t r = 0; r < pr.runs.size() && same_runs; ++r) {
        same_runs = pr.runs[r].block_id == serial.runs[r].block_id && pr.runs[r].local_k == serial.runs[r].local_k;
    }
    bool same_rows = pr.rows.size() == serial.rows.size();
    for (size_t i = 0; i < pr.rows.size() && same_rows; ++i) {
        same_rows = pr.rows[i].lane == serial.rows[i].lane && pr.rows[i].source_row == serial.rows[i].source_row;
    }
    check("R runs (block id, selected local K) and lane-major (lane, source row) rows == serialized request",
          same_runs && same_rows && pr.k_compact == serial.k_compact);
    std::set<size_t> counts;
    for (const auto & r : pr.runs) {
        counts.insert(r.local_k.size());
    }
    check("R compact run lengths 1, 15, 16, 17, 31, 32 present",
          counts == std::set<size_t>{ 1, 15, 16, 17, 31, 32 });

    // Digits: the paired compact limbs are the serialized K32-padded A without its zero padding.
    bool same_digits = true;
    for (size_t i = 0; i < pr.rows.size(); ++i) {
        size_t c = 0;
        for (size_t r = 0; r < pr.runs.size(); ++r) {
            for (size_t x = 0; x < pr.runs[r].local_k.size(); ++x, ++c) {
                same_digits = same_digits && pr.limbs[i * pr.k_compact + c] == serial.a[i * serial.lda + r * 32 + x];
            }
        }
    }
    check("R compact limbs == serialized digits (compact order, original block identity kept)", same_digits);

    // Fragments: ceil(c/DIM) per run, never across runs, compact order.
    bool   frag_ok = true;
    size_t f_index = 0, compact = 0;
    for (size_t r = 0; r < pr.runs.size(); ++r) {
        const size_t c = pr.runs[r].local_k.size();
        for (size_t x = 0; x < c; x += PD, ++f_index) {
            const paired_fragment & fr = pr.fragments.at(f_index);
            frag_ok = frag_ok && fr.run == r && fr.block_id == pr.runs[r].block_id && fr.compact_begin == compact + x &&
                      fr.count == std::min(PD, c - x) && fr.count >= 1 && fr.count <= PD;
        }
        compact += c;
    }
    check("R fragments: ceil(c/DIM) per run, <= DIM K, one run each, compact order",
          frag_ok && f_index == pr.fragments.size());

    // Indexed weight: compact K -> original K -> the same row the serialized path copies.
    bool same_weight = true;
    for (const paired_fragment & fr : pr.fragments) {
        for (size_t x = 0; x < fr.count; ++x) {
            const size_t k  = pr.original_k(fr, x);
            const size_t kk = fr.run * 32 + fr.local_begin + x; // serialized K32-padded row
            same_weight     = same_weight && k / 32 == fr.block_id &&
                          std::memcmp(f.weight.b + k * f.weight.ldb, serial.b + kk * serial.ldb, f.N) == 0;
        }
    }
    check("R original-K indexing reads exactly the weight rows build_request gathers (shared, not copied)",
          same_weight && pr.weight == &f.weight);
    check("R paired descriptor holds only limbs (rows x compact K): no Residual B, no gathered carriers",
          pr.limbs.size() == pr.rows.size() * pr.k_compact && pr.limbs.size() < serial.lda * pr.rows.size());

    // Groups: ceil(R / stripe_rows), consecutive, the last one shorter.
    paired_residual_request g16;
    const bool              gb = build_paired_residual_request(*packet, f.weight, 16, g16) == grmd::RmdStatus::success;
    bool                    groups_ok = gb && g16.groups.size() == (g16.rows.size() + 15) / 16;
    size_t                  next = 0;
    for (const auto & g : g16.groups) {
        groups_ok = groups_ok && g.first_row == next && g.rows >= 1 && g.rows <= 16 &&
                    g.nonzero.size() == g16.fragments.size();
        next += g.rows;
    }
    check("R residual rows > stripe rows: ceil(R / I_s) groups of <= I_s consecutive lane rows",
          groups_ok && next == g16.rows.size() && g16.groups.size() >= 2);
    std::printf("  descriptor: %zu runs, %zu fragments, %zu lane rows, compact K %zu, groups(I_s=16) %zu\n",
                pr.runs.size(), pr.fragments.size(), pr.rows.size(), pr.k_compact, g16.groups.size());
}

void test_failures() {
    paired_fixture f(40, 48, 256, 9, [](size_t blk, size_t n) { return (int16_t) ((blk + 2 * n) % 4); });
    paired_tile_plan p;
    if (!f.ok || paired_frozen_plan(f.M, f.N, f.K, 2, 2, 4, p) != POTAL_OK) {
        check("E fixture", false);
        return;
    }
    paired_inputs in;
    const bool    made = make_inputs(f, p.stripe_rows, [](size_t, size_t rows) {
        std::vector<res_entry> e;
        for (size_t r = 0; r < rows; ++r) {
            e.push_back({ r, 33 + r % 7, 0x10101 + (int32_t) r });
            e.push_back({ r, 130, -(int32_t) (5 + r) });
        }
        return e;
    }, in);
    check("E inputs", made);
    const auto run = [&](const paired_gemm_desc & d, paired_execution_report & rep) {
        paired_output out;
        out.main_c = { 12345 };
        const potal_status st = paired_tiled_matmul_auto(d, out, rep);
        return std::make_pair(st, out.main_c.size() == 1 && out.main_c[0] == 12345 && out.out.empty());
    };
    paired_execution_report rep;
    {
        paired_gemm_desc d = desc_of(f, p, in);
        auto             r = run(d, rep);
        check("E baseline request succeeds", r.first == POTAL_OK && rep.executed_ops > 0);
    }
    {
        paired_gemm_desc d = desc_of(f, p, in);
        d.a                = nullptr;
        auto r             = run(d, rep);
        check("E invalid descriptor (A null): refused, 0 ops, output untouched",
              r.first == POTAL_ERR_INVALID_ARGUMENT && rep.executed_ops == 0 && r.second &&
                  std::string(rep.stage) == "request");
    }
    {
        paired_gemm_desc d = desc_of(f, p, in);
        d.stripe_rows      = p.stripe_rows + 16;
        auto r             = run(d, rep);
        check("E ExSIA stripe height != paired stripe rows: refused, 0 ops",
              r.first == POTAL_ERR_INVALID_ARGUMENT && rep.executed_ops == 0 && r.second &&
                  std::string(rep.stage) == "stripes");
    }
    {
        grmd::StripePacket bad = *in.packets[0];
        bad.k_indices[0]       = 40; // not a block-local K
        paired_inputs b        = in;
        b.stripes[0].packet    = &bad;
        paired_gemm_desc d     = desc_of(f, p, b);
        auto             r     = run(d, rep);
        check("E invalid K index in the packet: refused, 0 ops",
              r.first == POTAL_ERR_INVALID_ARGUMENT && rep.executed_ops == 0 && r.second &&
                  std::string(rep.stage) == "residual_prepare");
    }
    {
        paired_inputs b     = in;
        b.stripes[1].packet = in.packets[0].get(); // stripe 0's packet on stripe 1
        paired_gemm_desc d  = desc_of(f, p, b);
        auto             r  = run(d, rep);
        check("E packet of another stripe: refused, 0 ops",
              r.first == POTAL_ERR_INVALID_ARGUMENT && rep.executed_ops == 0 && r.second);
    }
    {
        const uint32_t saved = f.weight.carriers[3];
        f.weight.carriers[3] = 40000; // not a PoTal carrier
        paired_gemm_desc d   = desc_of(f, p, in);
        auto             r   = run(d, rep);
        f.weight.carriers[3] = saved;
        check("E invalid carrier: refused (scales), 0 ops",
              r.first == POTAL_ERR_INVALID_ARGUMENT && rep.executed_ops == 0 && r.second &&
                  std::string(rep.stage) == "scales");
    }
    {
        paired_inputs b    = in;
        b.stripes[0].theta = INT16_MIN;
        paired_gemm_desc d = desc_of(f, p, b);
        auto             r = run(d, rep);
        check("E missing theta: scale validation fails, 0 ops",
              r.first == POTAL_ERR_INVALID_ARGUMENT && rep.executed_ops == 0 && r.second);
    }
    {
        paired_gemm_desc d     = desc_of(f, p, in);
        d.spill_capacity_bytes = 0;
        auto r                 = run(d, rep);
        check("E spill backing too small for the group saves: POTAL_ERR_CAPACITY, 0 ops",
              r.first == POTAL_ERR_CAPACITY && rep.executed_ops == 0 && r.second && std::string(rep.stage) == "spill");
    }
    {
        // 65 I tiles x one K32 block = 2080 rows > the 2048-row Main partition.
        paired_fixture big(65 * PD, 16, 32, 3, [](size_t, size_t) { return (int16_t) 1; });
        paired_gemm_desc d;
        d.m = big.M, d.n = big.N, d.k = big.K, d.a = big.a.get(), d.lda = big.lda, d.weight = &big.weight;
        d.tile_I = 65, d.tile_J = 1, d.tile_K = 2;
        d.stripe_rows = 65 * PD;
        auto r        = run(d, rep);
        check("E SP capacity exceeded by the frozen t_I: POTAL_ERR_CAPACITY at plan, 0 ops, output untouched",
              big.ok && r.first == POTAL_ERR_CAPACITY && std::string(rep.stage) == "plan" && rep.executed_ops == 0 &&
                  r.second);
    }

    // Tampered SP addresses: the static validator and the emulator both refuse.
    paired_residual_request req;
    if (build_paired_residual_request(*in.packets[0], f.weight, p.stripe_rows, req) != grmd::RmdStatus::success) {
        check("E request", false);
        return;
    }
    const paired_schedule good = build_paired_schedule(p, 0, req);
    std::string           why;
    check("E untampered schedule validates", validate_paired_schedule(p, req, good, nullptr, &why) == POTAL_OK);
    const auto tamper = [&](paired_op_kind kind, const std::function<void(paired_op &)> & change) {
        paired_schedule s = good;
        for (paired_op & op : s.ops) {
            if (op.kind == kind) {
                change(op);
                break;
            }
        }
        return s;
    };
    paired_gemm_desc d    = desc_of(f, p, in);
    const auto       emu  = [&](const paired_schedule & s) {
        std::vector<int32_t> mc = { 7 }, rc = { 7 };
        paired_counts        n;
        uint64_t             ops = 0;
        std::string          w;
        const potal_status   st = execute_paired_emulated(d, p, req, s, mc, rc, n, ops, w);
        return st != POTAL_OK && mc.size() == 1 && rc.size() == 1;
    };
    const paired_schedule overlap = tamper(paired_op_kind::main_a_load, [](paired_op & op) { op.sp_a += 2048; });
    const paired_schedule front_w = tamper(paired_op_kind::weight_load, [](paired_op & op) { op.sp_w -= 16; });
    const paired_schedule wrong_pair = tamper(paired_op_kind::main_a_load, [](paired_op & op) { op.pair ^= 1; });
    check("E Main activation moved into the Residual partition: validator and emulator refuse",
          validate_paired_schedule(p, req, overlap, nullptr, &why) != POTAL_OK && emu(overlap));
    check("E weight not rear-placed: validator and emulator refuse",
          validate_paired_schedule(p, req, front_w, nullptr, &why) != POTAL_OK && emu(front_w));
    check("E load into the wrong pair: validator and emulator refuse, no partial output",
          validate_paired_schedule(p, req, wrong_pair, nullptr, &why) != POTAL_OK && emu(wrong_pair));
    check("E untampered schedule runs on the emulator", !emu(good));
}

} // namespace

int main() {
    test_geometry();
    test_select();
    test_frozen();
    test_descriptor();
    test_failures();
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
