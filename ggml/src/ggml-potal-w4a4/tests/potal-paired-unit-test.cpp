// Hardware-free numeric tests of the paired GEMM (matmul/paired.hpp, P4-P7).
// Every case compares, bit for bit,
//   A  the independent golden (support/paired-support.hpp, stripe-golden.hpp)
//   B  the serialized Main / Residual GEMMs of the existing path on the
//      RTL-semantics emulator (potal_tiled_matmul_auto, build_request, reconstruct)
//   C  paired_tiled_matmul_auto (PAIRED_EMULATED)
// for Main INT32, Residual INT32 (rows and values) and the radix INT64
// correction; then LEGACY_F32 (== the STRIPE_PIPELINE output, A and B) and
// INT64 merge (== its own golden) separately. Their F32 difference is printed,
// not required to be zero.
//
//   N  the required shapes (paired plan), residual compact run lengths
//      1/15/16/17/31/32, several blocks, lanes 0..4, signs, blocks without residual
//   X  edge cases with frozen small tiles (several windows and groups): Sat32
//      saturation, cancellation, all-zero group fragments, partial last window,
//      saturation right after a group restore, tail M/N, INT64 overflow
//   F  failures: hardware backend (unsupported, nothing run), spill corruption,
//      injected emulator failures; no partial commit; repeated runs identical;
//      the existing tile selection untouched
//   I  host integration: real ExSIA stripes and packets of a GGML node (RMD ON)
//      and the prepared Q8_HP1 weight, with the paired plan and with the
//      STRIPE_PIPELINE plan frozen; LEGACY_F32 == pipeline_golden bitwise

#include "support/paired-support.hpp"

#include "quants/act/exsia/types.hpp"

#include <map>
#include <variant>

using namespace ggml::potal;
using namespace potal_test;

namespace {

namespace exsia_ns = ggml::gemmini::quants::act::exsia;

struct case_result {
    bool                    ok = false;
    three_way               t;
    paired_execution_report rep;
    paired_output           out, out64;
};

void print_counts(const char * name, const paired_execution_report & r, const three_way & t) {
    const paired_counts & n = r.totals;
    std::printf("  %s: tiles I/J/K %zu/%zu/%zu stripe_rows %zu windows %zu groups %zu | main %llu residual %llu "
                "micro-tiles, switches %llu, tails M %llu R %llu, dummy %llu, skipped zero (group,frag,N) %llu | "
                "weight loads %llu (%llu B) preloads %llu indexed reads %llu | ACC saves %llu restores %llu (%llu + "
                "%llu B) | peaks SP %llu B ACC-M %llu B ACC-R %llu B spill %llu B [%s] | legacy vs INT64 merge: %zu "
                "cells differ\n",
                name, r.plan.tile_I, r.plan.tile_J, r.plan.tile_K, r.stripe_rows, r.common_k_windows,
                r.residual_groups, (unsigned long long) n.main_micro_tiles, (unsigned long long) n.residual_micro_tiles,
                (unsigned long long) n.mode_switches, (unsigned long long) n.main_only_tail_micro_tiles,
                (unsigned long long) n.residual_only_tail_micro_tiles, (unsigned long long) n.dummy_micro_tiles,
                (unsigned long long) n.skipped_zero_fragments, (unsigned long long) n.logical_weight_loads,
                (unsigned long long) n.logical_weight_dma_bytes, (unsigned long long) n.logical_weight_preloads,
                (unsigned long long) n.indexed_weight_reads, (unsigned long long) n.residual_acc_saves,
                (unsigned long long) n.residual_acc_restores, (unsigned long long) n.residual_acc_save_bytes,
                (unsigned long long) n.residual_acc_restore_bytes, (unsigned long long) n.scratchpad_peak_bytes,
                (unsigned long long) n.main_acc_peak_bytes, (unsigned long long) n.residual_acc_peak_bytes,
                (unsigned long long) n.spill_peak_bytes, r.traffic_basis, t.merge_differences);
}

case_result run_case(const std::string & name, const paired_fixture & f, const paired_tile_plan & plan,
                     const paired_inputs & in) {
    case_result c;
    const golden_result     g = golden_of(f, plan.stripe_rows, in);
    const serialized_result b = serialized_of(f, plan.stripe_rows, in);
    paired_gemm_desc        d = desc_of(f, plan, in);
    paired_execution_report rep64;
    const potal_status      s1 = paired_tiled_matmul_auto(d, c.out, c.rep);
    d.merge                    = paired_merge_mode::int64;
    const potal_status s2      = paired_tiled_matmul_auto(d, c.out64, rep64);
    c.ok                       = g.ok && s1 == POTAL_OK && s2 == POTAL_OK;
    if (!c.ok) {
        std::printf("  %s: golden %d paired %s / %s stage %s %s\n", name.c_str(), (int) g.ok, potal_status_name(s1),
                    potal_status_name(s2), c.rep.stage ? c.rep.stage : "-", c.rep.detail.c_str());
        check(name + ": ran", false);
        return c;
    }
    c.t                              = compare(f, g, b, c.out, c.out64);
    c.rep.main_int32_match           = c.t.main ? paired_check::match : paired_check::mismatch;
    c.rep.residual_int32_match       = c.t.residual ? paired_check::match : paired_check::mismatch;
    c.rep.reconstruction_int64_match = c.t.int64 ? paired_check::match : paired_check::mismatch;
    print_counts(name.c_str(), c.rep, c.t);
    check(name + ": Main INT32 A == B == C", c.t.main);
    check(name + ": Residual INT32 A == B == C (lane rows and values)", c.t.residual);
    check(name + ": radix INT64 A == B == C", c.t.int64);
    check(name + ": LEGACY_F32 merge == golden == serialized (bitwise)", c.t.legacy);
    check(name + ": INT64 merge == its golden (bitwise)", c.t.int64_merge);
    check(name + ": no dummy micro-tile, PAIRED_EMULATED",
          c.rep.totals.dummy_micro_tiles == 0 &&
              c.rep.execution_backend == paired_execution_backend::paired_emulated);
    return c;
}

int16_t mixed_exponent(size_t blk, size_t n) {
    if ((blk * 7 + n) % 11 == 0) {
        return std::numeric_limits<int16_t>::min(); // zero carrier
    }
    return (int16_t) ((blk * 3 + n * 5) % 9);
}

// Residuals on 2 of every 3 blocks, compact run lengths cycling 1/15/16/17/31/32,
// values over lanes 0..4 (0x7fffffff reaches the carry lane), both signs.
// value_kinds < 8 keeps the carry lanes out (a saturated carry lane leaves INT64).
std::vector<res_entry> mixed_entries(size_t stripe, size_t rows, size_t K, size_t value_kinds = 8) {
    static const size_t  counts[] = { 1, 15, 16, 17, 31, 32 };
    static const int32_t values[] = { 3, 77, 300, 4000, 70000, 0x123456, 0x1234567, 0x7fffffff };
    std::vector<res_entry> e;
    size_t                 v = stripe;
    for (size_t b = 0, idx = 0; b < K / 32; ++b) {
        if (b % 3 == 1) {
            continue; // an original block without any residual
        }
        const size_t c = counts[idx++ % 6];
        for (size_t i = 0; i < c; ++i, ++v) {
            const size_t local = c <= 16 ? (i * 2 + b) % 32 : i;
            const size_t row   = (b * 7 + i * 3 + stripe) % rows;
            int32_t      value = values[v % value_kinds];
            value              = (v % 3 == 0) ? -value : value;
            if (value == -0x7fffffff && v % 2) {
                value = std::numeric_limits<int32_t>::min(); // the INT32_MIN residual
            }
            e.push_back({ row, b * 32 + local, value });
        }
    }
    return e;
}

void test_shapes() {
    const size_t shapes[][3] = { { 150, 320, 128 }, { 200, 320, 128 }, { 100, 160, 2048 },
                                 { 160, 320, 512 }, { 128, 3072, 768 }, { 1, 768, 768 } };
    uint32_t     seed        = 100;
    for (const auto & s : shapes) {
        const std::string name = "N M=" + std::to_string(s[0]) + " N=" + std::to_string(s[1]) +
                                 " K=" + std::to_string(s[2]);
        paired_fixture   f(s[0], s[1], s[2], seed++, mixed_exponent);
        paired_tile_plan plan;
        paired_inputs    in;
        const size_t     K = s[2];
        if (!f.ok || paired_tile_select(s[0], s[1], s[2], plan) != POTAL_OK ||
            !make_inputs(f, plan.stripe_rows, [K](size_t st, size_t rows) { return mixed_entries(st, rows, K); },
                         in)) {
            check(name + ": fixture", false);
            continue;
        }
        run_case(name, f, plan, in);
    }
}

void test_edges() {
    // Sat32 saturation: shifts 20..28, both signs, multiple windows and groups.
    {
        paired_fixture   f(40, 40, 160, 21, [](size_t blk, size_t n) { return (int16_t) (20 + (blk + n) % 9); });
        paired_tile_plan plan;
        paired_inputs    in;
        const bool       ok = f.ok && paired_frozen_plan(40, 40, 160, 2, 2, 4, plan) == POTAL_OK &&
                        make_inputs(f, plan.stripe_rows,
                                    [](size_t st, size_t rows) { return mixed_entries(st, rows, 160, 6); }, in);
        if (ok) {
            case_result c = run_case("X Sat32 saturation, tail M/N (40x40), partial last K window (5 blocks / 2)", f,
                                     plan, in);
            size_t saturated = 0;
            for (int32_t v : c.out.main_c) {
                saturated += v == INT32_MAX || v == INT32_MIN;
            }
            for (const auto & st : c.out.stripes) {
                for (int32_t v : st.residual_c) {
                    saturated += v == INT32_MAX || v == INT32_MIN;
                }
            }
            check("X   saturation present in the outputs, last window partial (k_windows 3, last 1 block)",
                  saturated > 0 && plan.k_windows == 3 && plan.k / 32 - 2 * plan.window_blocks == 1);
        } else {
            check("X saturation fixture", false);
        }
    }
    // Cancellation: run order matters under Sat32 (block 0 saturates up, block 1 down).
    {
        paired_fixture f(16, 16, 64, 22, [](size_t blk, size_t) { return (int16_t) (blk == 0 ? 24 : 23); },
                         [](size_t k, size_t, std::mt19937 &) { return (int8_t) (k < 32 ? 127 : -127); });
        paired_tile_plan plan;
        paired_inputs    in;
        const bool ok = f.ok && paired_frozen_plan(16, 16, 64, 1, 1, 2, plan) == POTAL_OK &&
                        make_inputs(f, plan.stripe_rows, [](size_t, size_t) {
                            std::vector<res_entry> e;
                            for (size_t i = 0; i < 17; ++i) {
                                e.push_back({ 0, i, 100 });
                                e.push_back({ 0, 32 + i, 100 });
                            }
                            return e;
                        }, in);
        case_result c = ok ? run_case("X cancellation across runs (Sat32 order)", f, plan, in) : case_result{};
        // Exact per-block sums would cancel to ~0; the Sat32 run order does not.
        check("X   cancellation keeps the Sat32 order (lane-0 row != 0)",
              c.ok && !c.out.stripes[0].residual_c.empty() && c.out.stripes[0].residual_c[0] != 0);
    }
    // All-zero group fragments and an empty (group, window); saturation right after a restore.
    {
        paired_fixture   f(32, 48, 128, 23, [](size_t blk, size_t n) { return (int16_t) (blk >= 2 ? 26 : (blk + n) % 4); });
        paired_tile_plan plan;
        paired_inputs    in;
        const bool ok = f.ok && paired_frozen_plan(32, 48, 128, 1, 2, 2, plan) == POTAL_OK &&
                        make_inputs(f, plan.stripe_rows, [](size_t, size_t rows) {
                            std::vector<res_entry> e;
                            for (size_t r = 0; r < rows; ++r) {
                                e.push_back({ r, 3 + r, 0x10101 + (int32_t) r });   // block 0: lanes 0..2
                                e.push_back({ r, 40 + r % 8, 9 });                  // block 1: lane 0 only
                                e.push_back({ r, 64 + r, -(0x10101 + (int32_t) r) }); // block 2: saturating
                                e.push_back({ r, 100, 0x20000 });                   // block 3: lane 2 only
                            }
                            return e;
                        }, in);
        case_result c = ok ? run_case("X all-zero group fragments, empty (group, window), saturation after restore",
                                      f, plan, in)
                           : case_result{};
        check("X   zero fragments skipped, groups saved and restored, saturated Residual values",
              c.ok && c.rep.totals.skipped_zero_fragments > 0 && c.rep.totals.residual_acc_restores > 0 && [&] {
                  for (const auto & st : c.out.stripes) {
                      for (int32_t v : st.residual_c) {
                          if (v == INT32_MAX || v == INT32_MIN) {
                              return true;
                          }
                      }
                  }
                  return false;
              }());
    }
    // INT64 overflow of the radix reconstruction: A, B and C all refuse it.
    {
        paired_fixture f(16, 16, 64, 24, [](size_t, size_t) { return (int16_t) 30; },
                         [](size_t, size_t, std::mt19937 &) { return (int8_t) 127; });
        paired_tile_plan plan;
        paired_inputs    in;
        const bool ok = f.ok && paired_frozen_plan(16, 16, 64, 1, 1, 2, plan) == POTAL_OK &&
                        make_inputs(f, plan.stripe_rows, [](size_t, size_t) {
                            std::vector<res_entry> e = { { 0, 0, 0x7fffffff } }; // lanes 0, 3 (-128) and 4 (+1)
                            for (size_t i = 0; i < 17; ++i) {
                                e.push_back({ 0, 32 + i, 0x7f000000 }); // lane 3 (+127): two fragments
                            }
                            return e;
                        }, in);
        const golden_result     g = ok ? golden_of(f, plan.stripe_rows, in) : golden_result{};
        const serialized_result b = ok ? serialized_of(f, plan.stripe_rows, in) : serialized_result{};
        paired_gemm_desc        d = desc_of(f, plan, in);
        paired_output           out;
        out.main_c = { 4242 };
        paired_execution_report rep;
        const potal_status      st = ok ? paired_tiled_matmul_auto(d, out, rep) : POTAL_ERR_INVALID_ARGUMENT;
        check("X INT64 reconstruction overflow: golden, serialized and paired all refuse; nothing committed",
              ok && !g.ok && g.overflow_stripes == std::vector<size_t>{ 0 } && b.composed[0] == grmd::RmdStatus::overflow &&
                  st != POTAL_OK && rep.stage != nullptr && std::string(rep.stage) == "radix_reconstruct" &&
                  out.main_c.size() == 1 && out.main_c[0] == 4242);
    }
}

void test_failures() {
    paired_fixture   f(40, 48, 128, 31, mixed_exponent);
    paired_tile_plan plan;
    paired_inputs    in;
    if (!f.ok || paired_frozen_plan(40, 48, 128, 2, 2, 4, plan) != POTAL_OK ||
        !make_inputs(f, plan.stripe_rows, [](size_t, size_t rows) {
            std::vector<res_entry> e;
            for (size_t r = 0; r < rows; ++r) {
                e.push_back({ r, 5 + r, 0x10101 });
                e.push_back({ r, 70 + r % 9, -3 });
            }
            return e;
        }, in)) {
        check("F fixture", false);
        return;
    }
    const auto untouched = [](const paired_output & o) {
        return o.main_c.size() == 1 && o.main_c[0] == 99 && o.out.empty() && o.stripes.empty();
    };
    const auto sentinel = []() {
        paired_output o;
        o.main_c = { 99 };
        return o;
    };
    paired_gemm_desc        d = desc_of(f, plan, in);
    paired_execution_report rep;
    {
        paired_gemm_desc h = d;
        h.backend          = paired_backend::hardware;
        paired_output   o  = sentinel();
        const uint64_t  before = potal_tile_select_count();
        const potal_status st = paired_tiled_matmul_auto(h, o, rep);
        check("F hardware backend: PAIRED_HARDWARE_UNSUPPORTED before any work (no serialized fallback)",
              st == POTAL_ERR_UNSUPPORTED &&
                  rep.execution_backend == paired_execution_backend::paired_hardware_unsupported &&
                  rep.executed_ops == 0 && untouched(o) && potal_tile_select_count() == before);
    }
    paired_output           ref;
    paired_execution_report ref_rep;
    std::vector<paired_schedule> ref_sched;
    const uint64_t          selects = potal_tile_select_count();
    const bool              ran     = paired_tiled_matmul_auto(d, ref, ref_rep, &ref_sched) == POTAL_OK;
    check("F reference run (saves and restores present)",
          ran && ref_rep.totals.residual_acc_saves > 0 && ref_rep.executed_ops > 0);
    check("F the paired path never calls the existing tile selection (potal_tile_select_count unchanged)",
          potal_tile_select_count() == selects);
    {
        paired_faults    fl;
        fl.corrupt_spill = true;
        paired_gemm_desc c = d;
        c.faults           = &fl;
        paired_output    o = sentinel();
        const potal_status st = paired_tiled_matmul_auto(c, o, rep);
        check("F corrupted spill backing detected at restore: failure, nothing committed",
              st == POTAL_ERR_BAD_RESPONSE && std::string(rep.stage) == "emulator" && untouched(o));
    }
    const int64_t points[] = { 0, (int64_t) ref_rep.executed_ops / 2, (int64_t) ref_rep.executed_ops - 1 };
    bool          inj      = ran;
    for (int64_t at : points) {
        paired_faults fl;
        fl.fail_at_op      = at;
        paired_gemm_desc c = d;
        c.faults           = &fl;
        paired_output      o  = sentinel();
        const potal_status st = paired_tiled_matmul_auto(c, o, rep);
        inj = inj && st == POTAL_HW_BAD_COMMAND && rep.executed_ops == (uint64_t) at && untouched(o);
    }
    check("F injected emulator failure (first / middle / last op): stops there, no partial commit", inj);

    bool same = ran;
    for (int i = 0; i < 3 && same; ++i) {
        paired_output                o;
        paired_execution_report      r;
        std::vector<paired_schedule> sch;
        same = paired_tiled_matmul_auto(d, o, r, &sch) == POTAL_OK && same_bits(o.main_c, ref.main_c) &&
               same_bits(o.out, ref.out) && r.executed_ops == ref_rep.executed_ops &&
               r.totals.residual_acc_save_bytes == ref_rep.totals.residual_acc_save_bytes &&
               sch.size() == ref_sched.size();
        for (size_t s = 0; s < sch.size() && same; ++s) {
            same = sch[s].ops.size() == ref_sched[s].ops.size();
            for (size_t k = 0; k < sch[s].ops.size() && same; ++k) {
                same = paired_op_text(sch[s].ops[k]) == paired_op_text(ref_sched[s].ops[k]);
            }
            for (size_t k = 0; k < o.stripes[s].residual_c.size() && same; ++k) {
                same = o.stripes[s].residual_c[k] == ref.stripes[s].residual_c[k];
            }
        }
    }
    check("F repeated runs after failures: identical outputs, counters and traces", same);
}

// ---- I: real ExSIA packets ----------------------------------------------------------

struct node {
    size_t           M, N, K;
    ggml_context_ptr ctx;
    ggml_tensor *    w = nullptr;
    ggml_tensor *    x = nullptr;
    ggml_tensor *    y = nullptr;
    hp1_weight       weight;
    bool             ok = false;

    node(size_t M_, size_t N_, size_t K_, uint32_t seed) : M(M_), N(N_), K(K_) {
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
            if (m % 2 == 0) {
                v[m * K + (m * 7) % K] = 900.0f * (m % 4 ? -1.0f : 1.0f); // outliers: ExSIA seals packets
            }
        }
        potal_tile_plan    plan;
        ggml_gemmini_args_t args;
        hp1_shape          s = { (int64_t) M, (int64_t) N, (int64_t) K };
        ok = potal_tile_select(M, N, K, 32 / PD, &plan) == POTAL_OK &&
             init_args(w, y, s, "gpt2", "paired-integration", ggml::gemmini::residual::ResidualRoute::ws_packet, plan,
                       args) &&
             weight.allocate((int64_t) N, (int64_t) K) && prepare_weight(args, weight) == quants::weight::hp1::status::ok;
    }
};

void integration(const std::string & name, node & nd, const paired_tile_plan & plan) {
    // The ExSIA stripe height is the paired plan's: quantize with its tiles.
    potal_tile_plan     exsia_plan;
    ggml_gemmini_args_t args;
    hp1_operands        o;
    hp1_shape           s = { (int64_t) nd.M, (int64_t) nd.N, (int64_t) nd.K };
    bool ok = potal_frozen_plan(nd.M, nd.N, nd.K, plan.tile_I, plan.tile_J, plan.tile_K, 32 / PD, &exsia_plan) ==
                  POTAL_OK &&
              init_args(nd.w, nd.y, s, "gpt2", "paired-integration", ggml::gemmini::residual::ResidualRoute::ws_packet,
                        exsia_plan, args) &&
              args.activation_rows_per_stripe == plan.stripe_rows && o.allocate(s, nd.weight) &&
              quantize_activations(nd.x, args, o);
    const auto * meta = ok ? std::get_if<exsia_ns::Meta>(&args.act_quant.storage()) : nullptr;
    std::vector<float> want;
    ok = ok && meta != nullptr && meta->theta.size() == plan.stripes && potal_test::pipeline_golden(args, o, want);
    if (!ok) {
        check(name + ": ExSIA fixture", false);
        return;
    }
    std::vector<paired_stripe_input> stripes(plan.stripes);
    size_t                           packets = 0;
    for (size_t st = 0; st < plan.stripes; ++st) {
        stripes[st].theta = meta->theta[st];
    }
    for (const auto & p : meta->rmd_packets) {
        stripes.at(p->stripe_id).packet = p.get();
        packets++;
    }
    paired_gemm_desc d;
    d.m = nd.M, d.n = nd.N, d.k = nd.K, d.a = o.a, d.lda = o.lda, d.weight = &nd.weight;
    d.stripe_rows  = plan.stripe_rows;
    d.stripes      = stripes.data();
    d.stripe_count = stripes.size();
    if (plan.frozen) {
        d.tile_I = plan.tile_I, d.tile_J = plan.tile_J, d.tile_K = plan.requested_tile_K;
    }
    paired_output           out;
    paired_execution_report rep;
    const potal_status      st = paired_tiled_matmul_auto(d, out, rep);
    std::vector<int32_t>    c(o.m_pad * o.ldc, 0), main(nd.M * nd.N);
    reference_gemm_i32(o.a, o.lda, o.b, o.ldb, o.carriers, o.cstride, 32, nd.M, nd.N, nd.K, c.data(), o.ldc);
    for (size_t m = 0; m < nd.M; ++m) {
        std::memcpy(main.data() + m * nd.N, c.data() + m * o.ldc, nd.N * sizeof(int32_t));
    }
    const three_way t{};
    print_counts(name.c_str(), rep, t);
    std::printf("    %zu ExSIA stripes, %zu sealed packets\n", plan.stripes, packets);
    check(name + ": ran on real ExSIA packets with Residual work", st == POTAL_OK && packets > 0 &&
                                                                       rep.totals.residual_micro_tiles > 0);
    check(name + ": Main INT32 == reference_gemm_i32 on the staged codes", st == POTAL_OK && same_bits(main, out.main_c));
    check(name + ": LEGACY_F32 == pipeline_golden (STRIPE_PIPELINE contract) bitwise",
          st == POTAL_OK && same_bits(want, out.out));
}

void test_integration() {
    node nd(96, 192, 768, 77); // its STRIPE_PIPELINE plan needs the common K-window shrink
    if (!nd.ok) {
        check("I node", false);
        return;
    }
    paired_tile_plan p;
    if (paired_tile_select(nd.M, nd.N, nd.K, p) == POTAL_OK) {
        integration("I ExSIA, paired plan", nd, p);
    }
    potal_tile_plan serial;
    if (potal_tile_select(nd.M, nd.N, nd.K, 32 / PD, &serial) == POTAL_OK &&
        paired_frozen_plan(nd.M, nd.N, nd.K, serial.tile_I, serial.tile_J, serial.tile_K, p) == POTAL_OK) {
        std::printf("  STRIPE_PIPELINE plan %zu/%zu/%zu -> paired window tile_K %zu\n", serial.tile_I, serial.tile_J,
                    serial.tile_K, p.tile_K);
        check("I frozen STRIPE_PIPELINE plan: same t_I / t_J / ExSIA stripe height, K window shrunk",
              p.tile_I == serial.tile_I && p.tile_J == serial.tile_J && p.tile_K < serial.tile_K);
        integration("I ExSIA, STRIPE_PIPELINE plan frozen", nd, p);
    }
}

} // namespace

int main() {
    test_shapes();
    test_edges();
    test_failures();
    test_integration();
    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
