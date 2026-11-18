// Parametric W4 micro-tile schedule regression. Logical host emulation only.
// Covers shared loads, alternation/tails, group reversal/spills, K order,
// deterministic traces, malformed schedules, and independent wide numerics.
#include "support/w4a4-golden.hpp"
#include "matmul/paired.hpp"
#include "potal-int4.h"
#include "residual/rmd/rmd-builder.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
using namespace ggml::potal;
namespace gold = w4a4_golden;
namespace rmd = ggml::gemmini::rmd;
namespace {
constexpr size_t D = POTAL_HW_DIM, BK = POTAL_SCALE_BLOCK_K, PPB = BK / D;
void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
bool alternates(const std::string & paths) {
    const size_t m = std::count(paths.begin(), paths.end(), 'M');
    const size_t r = paths.size() - m, both = 2 * std::min(m, r);
    for (size_t i = 0; i < both; ++i) if (paths[i] != (i % 2 ? 'R' : 'M')) return false;
    return paths.find_first_not_of(m > r ? 'M' : 'R', both) == std::string::npos;
}
void run_case(int mode) {
    const size_t M = 2 * D + 1, N = 2 * D + 1, K = 4 * BK;
    hp1_weight weight;
    require(weight.allocate(N, K), "W4 weight allocation");
    gold::mxint4_weight w;
    w.n = N; w.k = K; w.q.resize(N * K); w.carrier.resize(K / BK * N); w.scale.assign(N, 0.125f);
    for (size_t k = 0; k < K; ++k) for (size_t n = 0; n < N; ++n) {
        const int8_t code = int((k * 3 + n * 5) % 16) - 8;
        w.q[n * K + k] = code;
        potal_int4_set(reinterpret_cast<uint8_t *>(weight.b) + k * (weight.ldb / 2), n, code);
    }
    for (size_t b = 0; b < K / BK; ++b) for (size_t n = 0; n < N; ++n)
        weight.carriers[b * weight.cstride + n] = w.carrier[b * N + n] = (b + n) % 4;
    std::copy(w.scale.begin(), w.scale.end(), weight.scale);
    const size_t lda = K;
    std::unique_ptr<int8_t, decltype(&std::free)> a(static_cast<int8_t *>(dma_alloc(align_up(M, D) * lda / 2)), std::free);
    require(bool(a), "W4 activation allocation");
    std::vector<int8_t> logical(M * K);
    for (size_t row = 0; row < M; ++row) for (size_t k = 0; k < K; ++k) {
        logical[row * K + k] = int((row * 7 + k) % 16) - 8;
        potal_int4_set(reinterpret_cast<uint8_t *>(a.get()) + row * lda / 2, k, logical[row * K + k]);
    }
    paired_tile_plan plan;
    require(paired_frozen_plan(M, N, K, 2, 2, 2 * PPB, plan) == POTAL_OK, "W4 frozen schedule geometry");
    std::vector<rmd::StripePacketHandle> packets(plan.stripes);
    std::vector<paired_stripe_input> inputs(plan.stripes);
    std::vector<std::vector<int32_t>> source(plan.stripes);
    size_t expected_main = 0, expected_residual = 0, expected_saves = 0;
    for (size_t s = 0; s < plan.stripes; ++s) {
        const size_t rows = paired_stripe_rows(plan, s);
        source[s].assign(rows * K, 0);
        if (mode) {
            rmd::RmdStripeBuilder builder;
            builder.reset(s, s * plan.stripe_rows, rows, K, N, 4);
            for (size_t row = 0; row < rows; ++row) for (size_t k = 0; k < K; ++k) {
                if (mode == 2 && k >= BK) continue;
                const int32_t value = row % 2 ? -0x222 : 0x111;
                source[s][row * K + k] = value;
                require(builder.add_residual(row, k, value), "W4 radix16 source");
            }
            packets[s] = builder.finish();
            require(bool(packets[s]), "W4 sealed packet");
        }
        inputs[s] = {0, packets[s].get()};
        paired_residual_request residual;
        if (packets[s]) require(build_paired_residual_request(*packets[s], weight, plan.stripe_rows, residual) == rmd::RmdStatus::success,
                                "W4 residual schedule descriptor");
        const auto schedule = build_paired_schedule(plan, s, residual);
        const auto again = build_paired_schedule(plan, s, residual);
        std::string why;
        require(validate_paired_schedule(plan, residual, schedule, nullptr, &why) == POTAL_OK, why.c_str());
        require(schedule.ops.size() == again.ops.size(), "deterministic trace size");
        std::map<std::pair<size_t,size_t>, std::string> paths;
        std::map<std::tuple<size_t,size_t,size_t>, size_t> last;
        size_t loads = 0, saves = 0, restores = 0;
        int active = -1;
        for (size_t i = 0; i < schedule.ops.size(); ++i) {
            const auto & op = schedule.ops[i];
            require(paired_op_text(op) == paired_op_text(again.ops[i]), "deterministic trace fields");
            if (op.kind == paired_op_kind::window_begin) active = int(op.pair);
            if (op.kind == paired_op_kind::window_end) active = -1;
            if (op.kind == paired_op_kind::weight_load || op.kind == paired_op_kind::main_a_load)
                require(active != int(op.pair), "prefetch overwrote an ACTIVE bank");
            loads += op.kind == paired_op_kind::weight_load;
            saves += op.kind == paired_op_kind::group_save;
            restores += op.kind == paired_op_kind::group_restore;
            if (op.kind != paired_op_kind::main_tile && op.kind != paired_op_kind::residual_tile) continue;
            const bool main = op.kind == paired_op_kind::main_tile;
            (main ? expected_main : expected_residual)++;
            paths[{op.j_group, op.window}] += main ? 'M' : 'R';
            const auto key = std::make_tuple(op.j_group, main ? size_t(-1) : op.group, op.j);
            const size_t k = main ? op.k_begin : op.fragment;
            require(!last.count(key) || k > last[key], "per-output K contributions reordered");
            last[key] = k;
        }
        for (const auto & p : paths) require(alternates(p.second), "alternation before single-path tail");
        require(loads == plan.j_groups * plan.k_windows && saves == restores, "one shared weight load and matched spills");
        if (!mode) require(!expected_residual && !saves && residual.groups.empty(), "Main-only emits no residual work");
        if (mode == 1 && residual.groups.size() > 1) {
            require(saves == (residual.groups.size()-1) * (plan.k_windows-1) * plan.j_groups,
                    "group reversal must retain one group at each window boundary");
        }
        expected_saves += saves;
        const auto rejected = [&](paired_schedule bad) {
            require(validate_paired_schedule(plan, residual, bad, nullptr, &why) != POTAL_OK, "malformed schedule accepted");
        };
        auto bad = schedule;
        const auto pre = std::find_if(bad.ops.begin(), bad.ops.end(), [](const auto & op) { return op.kind == paired_op_kind::weight_preload; });
        require(pre != bad.ops.end(), "fixture needs stationary preload"); bad.ops.erase(pre); rejected(bad);
        bad = schedule;
        const auto load = std::find_if(bad.ops.begin(), bad.ops.end(), [](const auto & op) { return op.kind == paired_op_kind::weight_load; });
        bad.ops.insert(load, *load); rejected(bad);
        bad = schedule;
        std::vector<size_t> main_ops;
        for (size_t i = 0; i < bad.ops.size(); ++i) if (bad.ops[i].kind == paired_op_kind::main_tile) main_ops.push_back(i);
        require(main_ops.size() > 1, "fixture needs K-order mutation");
        std::swap(bad.ops[main_ops.front()], bad.ops[main_ops.back()]); rejected(bad);
        if (mode) {
            // Drop one Residual tile with its mode switch: two Main tiles meet while Residual has work.
            bad = schedule;
            const auto res = std::find_if(bad.ops.begin(), bad.ops.end(), [](const auto & op) { return op.kind == paired_op_kind::residual_tile; });
            require(res != bad.ops.end() && res != bad.ops.begin(), "fixture needs a Residual tile");
            bad.ops.erase(res - 1, res + 1); rejected(bad);
        }
    }
    paired_gemm_desc desc;
    desc.m = M; desc.n = N; desc.k = K; desc.a = a.get(); desc.lda = lda; desc.weight = &weight;
    desc.stripe_rows = plan.stripe_rows; desc.stripes = inputs.data(); desc.stripe_count = inputs.size();
    desc.tile_I = plan.tile_I; desc.tile_J = plan.tile_J; desc.tile_K = plan.requested_tile_K;
    paired_output out; paired_execution_report report;
    require(paired_tiled_matmul_auto(desc, out, report) == POTAL_OK, report.detail.c_str());
    require(report.totals.main_micro_tiles == expected_main && report.totals.residual_micro_tiles == expected_residual &&
            report.totals.residual_acc_saves == expected_saves && !report.totals.dummy_micro_tiles, "emulator versus trace counts");
    std::vector<int32_t> main(N);
    for (size_t row = 0; row < M; ++row) {
        gold::scu_row(logical.data() + row * K, w, main.data());
        for (size_t n = 0; n < N; ++n) require(out.main_c[row * N+n] == main[n], "independent Main DIM-pass INT32");
    }
    for (size_t s = 0; s < out.stripes.size(); ++s) {
        const size_t rows = paired_stripe_rows(plan, s);
        std::vector<int8_t> digits(K); std::vector<int8_t> decomposed;
        std::vector<gold::i128> correction(rows * N, 0);
        std::vector<std::pair<uint8_t,uint32_t>> expected_rows;
        if (mode) for (uint8_t lane = 0; lane < 3; ++lane) for (size_t row = 0; row < rows; ++row) {
            for (size_t k = 0; k < K; ++k) {
                require(gold::radix_digits(source[s][row*K+k], decomposed), "independent radix digits");
                digits[k] = lane < decomposed.size() ? decomposed[lane] : 0;
            }
            gold::scu_row(digits.data(), w, main.data());
            const size_t rr = expected_rows.size(); expected_rows.push_back({lane, uint32_t(row)});
            for (size_t n = 0; n < N; ++n) {
                require(out.stripes[s].residual_c[rr*N+n] == main[n], "independent Residual DIM-pass INT32");
                correction[row*N+n] += gold::i128(main[n]) * (gold::i128(1) << (4*lane));
            }
        }
        require(out.stripes[s].rows.size() == expected_rows.size(), "active residual lane/source rows");
        for (size_t i = 0; i < expected_rows.size(); ++i)
            require(out.stripes[s].rows[i].lane == expected_rows[i].first && out.stripes[s].rows[i].source_row == expected_rows[i].second,
                    "lane/source row order");
        if (mode) for (size_t i = 0; i < correction.size(); ++i)
            require(out.stripes[s].correction[i] == correction[i], "wide INT64 radix reconstruction");
    }
    std::printf("W4_SCHEDULE mode=%d DIM=%zu Bk=%zu PASS main=%zu residual=%zu saves=%zu\n", mode,D,BK,expected_main,expected_residual,expected_saves);
}
}
int main() try {
    for (int mode : {0,1,2}) run_case(mode);
    std::puts("PASS: 0 failure(s)"); return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1;
}
