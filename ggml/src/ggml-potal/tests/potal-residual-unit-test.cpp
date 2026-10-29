// Hardware-free tests of the PoTal WS residual of one stripe packet
// (residual/stripe-residual.hpp), executed through the production
// potal_tiled_matmul_stripe() on the RTL-semantics emulator (host and target):
//
//   A  run-aware request: runs in block order, compact K ascending per run,
//      lane-major active (lane, source row) rows, K32 padded operands gathered
//      from the prepared weight; compact run lengths 1, 15, 16, 17, 31, 32 and
//      [1, 31], [17, 3, 32] over non-consecutive blocks, non-consecutive K and
//      source rows, lane 0 and the carry lane
//   B  the padded Residual GEMM == the independent compact-run golden
//      (support/stripe-golden.hpp, no padding), INT32 bit-exact, for zero,
//      shift-0, shifting and saturating carriers, positive and negative
//      saturation
//   C  cancellation across runs: the Sat32 run order differs from summing
//      exact per-block results first, and the transport keeps the Sat32 order
//   D  radix reconstruction INT64 == golden (row/lane mapping); checked merge
//      == golden F32, only the stripe's rows touched
//   E  failures: invalid packet, A4 packet, foreign shape, invalid carrier
//      (refused without a command), duplicate/unordered/out-of-range rows,
//      INT64 overflow (and the INT64_MIN edge that fits), non-finite merge and
//      missing theta (output unchanged); a stripe without residual seals no packet
//   M  the prepared merge (merge_prepared_hp1) == the Gemmini checked merge it
//      replaced (support/gemmini-merge.hpp), status, nonzero count and every
//      output bit: corrections of every A/B/D case (saturating SCU included),
//      signs, zeros, INT64 extremes, theta < 0 / 0 / > 0 and its limits, tiny
//      and huge channel scales, wrong sizes and bounds, strides, offsets and a
//      tail stripe, several stripes of one weight; a non-finite prepared scale
//      refused; the output and the count untouched by every failure; repeated
//      merges never run the Gemmini reader's weight validation (its test
//      counters, GGML_GEMMINI_TESTING), while the Gemmini merge runs it per call
//
// The golden is independent of the lowering: it never pads, never tiles and
// accumulates per (lane, row, column) in run order with Sat32. CPU-direct RMD
// is not used as a golden (it differs under saturation by design).

#include "ggml-cpp.h"
#include "ggml-potal.h"
#include "ggml-quants.h"
#include "matmul/dense.hpp"
#include "matmul/stripe-api.hpp"
#include "matmul/tiling.hpp"
#include "potal.h"
#include "potal_hw_params.h"
#include "residual/stripe-residual.hpp"
#include "support/gemmini-merge.hpp"
#include "support/stripe-golden.hpp"
#include "support/stripe-owner.hpp"

#include "quants/common/weight_reader.hpp"
#include "residual/rmd/rmd-builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace ggml::potal;
namespace grmd = ggml::gemmini::rmd;
namespace exsia_ns = ggml::gemmini::quants::act::exsia;
namespace wreader  = ggml::gemmini::quants::wreader;

namespace {

int failures = 0;

void check(const std::string & name, bool ok) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    failures += ok ? 0 : 1;
}

constexpr size_t D = POTAL_HW_DIM;

struct entry {
    size_t  row; // stripe-local
    size_t  k;   // original K
    int32_t value;
};

// A Q8_HP1 weight N x K built block by block (codes, exponent m, channel
// scale), the node's arguments over it and its prepared operands.
struct fixture {
    size_t               M, N, K;
    ggml_context_ptr     ctx;
    ggml_tensor *        w = nullptr;
    ggml_tensor *        x = nullptr;
    ggml_tensor *        y = nullptr;
    std::vector<int8_t>  codes;  // [K][N], the golden's own copy
    std::vector<int16_t> exps;   // [K / 32][N]; INT16_MIN = zero carrier
    std::vector<float>   scales; // [N]
    ggml_gemmini_args_t  args;
    hp1_weight           weight;
    bool                 ok = false;

    fixture(size_t M_, size_t N_, size_t K_, uint32_t seed, const std::function<int16_t(size_t, size_t)> & exponent,
            const std::function<int8_t(size_t, size_t, std::mt19937 &)> & code,
            const std::function<float(size_t)> & channel_scale = {}) :
        M(M_), N(N_), K(K_) {
        ggml_init_params ip = { (M * K + N * K) * 8 + 4 * 1024 * 1024, nullptr, false };
        ctx.reset(ggml_init(ip));
        w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_HP1, (int64_t) K, (int64_t) N);
        x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, (int64_t) K, (int64_t) M);
        y = ggml_mul_mat(ctx.get(), w, x);
        std::mt19937 rng(seed);
        codes.resize(K * N);
        exps.resize(K / 32 * N);
        scales.resize(N);
        for (size_t n = 0; n < N; ++n) {
            scales[n] = channel_scale ? channel_scale(n) : 0.003f * (float) (1 + n % 7);
        }
        auto * blocks = static_cast<block_q8_hp1 *>(w->data);
        for (size_t n = 0; n < N; ++n) {
            for (size_t blk = 0; blk < K / 32; ++blk) {
                block_q8_hp1 & b = blocks[n * (K / 32) + blk];
                std::memset(&b, 0, sizeof(b));
                for (size_t i = 0; i < 32; ++i) {
                    const size_t k = blk * 32 + i;
                    codes[k * N + n] = code(k, n, rng);
                    b.qs[i]          = codes[k * N + n];
                }
                exps[blk * N + n] = exponent(blk, n);
                b.m               = exps[blk * N + n];
                b.channel_scale   = scales[n];
            }
        }
        hp1_shape       s = { (int64_t) M, (int64_t) N, (int64_t) K };
        potal_tile_plan plan;
        ok = potal_tile_select(M, N, K, 32 / D, &plan) == POTAL_OK &&
             init_args(w, y, s, "gpt2", "residual-unit", ggml::gemmini::residual::ResidualRoute::ws_packet, plan,
                       args) &&
             weight.allocate((int64_t) N, (int64_t) K) && prepare_weight(args, weight) == quants::weight::hp1::status::ok;
    }

    int32_t  code_at(size_t k, size_t n) const { return codes[k * N + n]; }
    uint32_t carrier_at(size_t blk, size_t n) const {
        const int16_t e = exps[blk * N + n];
        return e == std::numeric_limits<int16_t>::min() ? POTAL_CARRIER_ZERO : (uint32_t) e;
    }
};

int8_t random_code(size_t, size_t, std::mt19937 & rng) {
    return (int8_t) std::uniform_int_distribution<int>(-128, 127)(rng);
}

grmd::StripePacketHandle make_packet(size_t stripe_id, size_t row_begin, size_t rows, size_t K, size_t N,
                                     const std::vector<entry> & entries, uint8_t bits = 8) {
    grmd::RmdStripeBuilder b;
    b.reset(stripe_id, row_begin, rows, K, N, bits);
    for (const entry & e : entries) {
        if (!b.add_residual(e.row, e.k, e.value)) {
            return nullptr;
        }
    }
    return b.finish();
}

// The production path of one packet: request, Residual GEMM on the emulator.
struct residual_run {
    residual::request        req;
    grmd::RmdStatus          built = grmd::RmdStatus::invalid_arguments;
    potal_status             gemm  = POTAL_ERR_UNSUPPORTED;
    potal_stripe_gemm_report report{};
};

void execute_request(const grmd::StripePacket & p, potal_test::emu_session & es, residual_run & r) {
    if (r.built != grmd::RmdStatus::success) {
        return;
    }
    potal_stripe_gemm_desc d = {};
    d.stripe_id              = p.stripe_id;
    d.row_begin              = p.row_begin;
    d.work_kind              = POTAL_STRIPE_WORK_RESIDUAL;
    d.m                      = (uint32_t) r.req.m;
    d.n                      = (uint32_t) r.req.n;
    d.k                      = (uint32_t) r.req.k_execution;
    d.a                      = r.req.a;
    d.b                      = r.req.b;
    d.c                      = r.req.c;
    d.lda                    = r.req.lda;
    d.ldb                    = r.req.ldb;
    d.ldc                    = r.req.ldc;
    d.carriers               = r.req.carriers;
    d.carrier_stride         = r.req.cstride;
    d.scale_block_k          = 32;
    d.tile_i                 = (uint32_t) r.req.plan.tile_I;
    d.tile_j                 = (uint32_t) r.req.plan.tile_J;
    d.tile_k                 = (uint32_t) r.req.plan.tile_K;
    r.gemm                   = potal_tiled_matmul_stripe(&es.session, &d, &r.report);
}

void execute(const fixture & f, const grmd::StripePacket & p, potal_test::emu_session & es, residual_run & r) {
    r.built = residual::build_request(p, f.weight, r.req);
    execute_request(p, es, r);
}

bool golden_of(const fixture & f, const grmd::StripePacket & p, std::vector<potal_test::golden_lane_row> & g) {
    return potal_test::residual_golden(
        p, f.N, D, [&f](size_t k, size_t n) { return f.code_at(k, n); },
        [&f](size_t blk, size_t n) { return f.carrier_at(blk, n); }, g);
}

// One merge through the prepared HP1 path and through the Gemmini checked
// merge it replaced, each on its own copy of `base` (rows of f.M with
// row_stride, columns with col_stride): status, nonzero count, every bit.
struct merge_outcome {
    grmd::RmdStatus    prepared         = grmd::RmdStatus::invalid_arguments;
    grmd::RmdStatus    gemmini          = grmd::RmdStatus::invalid_arguments;
    size_t             prepared_nonzero = SIZE_MAX; // SIZE_MAX: not written
    size_t             gemmini_nonzero  = SIZE_MAX;
    std::vector<float> prepared_out;
    std::vector<float> gemmini_out;

    bool ok() const { return prepared == grmd::RmdStatus::success; }
    bool same() const {
        return prepared == gemmini && prepared_nonzero == gemmini_nonzero &&
               prepared_out.size() == gemmini_out.size() &&
               std::memcmp(prepared_out.data(), gemmini_out.data(), prepared_out.size() * sizeof(float)) == 0;
    }
    // A failure leaves both outputs and both counts as they were.
    bool untouched(const std::vector<float> & base) const {
        return !ok() && prepared_nonzero == SIZE_MAX && gemmini_nonzero == SIZE_MAX &&
               std::memcmp(prepared_out.data(), base.data(), base.size() * sizeof(float)) == 0 &&
               std::memcmp(gemmini_out.data(), base.data(), base.size() * sizeof(float)) == 0;
    }
};

merge_outcome merge_both(const fixture & f, size_t row_begin, size_t row_end, int16_t theta,
                         const grmd::BlockScaledInt64Correction & correction, const std::vector<float> & base,
                         size_t row_stride = 0, size_t col_stride = 1) {
    merge_outcome m;
    const size_t  rs = row_stride != 0 ? row_stride : f.N;
    m.prepared_out   = base;
    m.gemmini_out    = base;
    exsia_ns::StripeMetadataSnapshot snap;
    snap.theta = theta;
    m.prepared = residual::merge_prepared_hp1(f.weight, row_begin, row_end, snap, correction, m.prepared_out.data(),
                                              f.M, rs, col_stride, &m.prepared_nonzero);
    ggml_gemmini_args_t args;
    potal_test::init_merge_args(f.args, m.gemmini_out.data(), rs, args);
    args.col_stride_f_out                = col_stride;
    grmd::BlockScaledInt64Correction copy = correction;
    m.gemmini = potal_test::gemmini_merge(args, row_begin, row_end, 11, snap, std::move(copy), &m.gemmini_nonzero);
    return m;
}

bool same_as_golden(const residual::request & req, const std::vector<potal_test::golden_lane_row> & g, size_t N) {
    if (g.size() != req.m || req.rows.size() != req.m) {
        return false;
    }
    for (size_t i = 0; i < req.m; ++i) {
        if (g[i].lane != req.rows[i].lane || g[i].row != req.rows[i].source_row) {
            return false;
        }
        for (size_t n = 0; n < N; ++n) {
            if (req.c[i * req.ldc + n] != g[i].c[n]) {
                return false;
            }
        }
    }
    return true;
}

// What the request must look like, derived from the entries alone.
struct expected_shape {
    std::vector<std::pair<uint32_t, std::vector<uint16_t>>> runs;
    std::vector<std::pair<uint8_t, uint32_t>>               rows;
};

expected_shape expect(const std::vector<entry> & entries, size_t rows) {
    std::map<uint32_t, std::set<uint16_t>> blocks;
    std::set<std::pair<uint8_t, uint32_t>> active;
    for (const entry & e : entries) {
        if (e.value == 0) {
            continue;
        }
        blocks[(uint32_t) (e.k / 32)].insert((uint16_t) (e.k % 32));
        grmd::NativeBalancedDigits digits;
        (void) grmd::decompose_balanced_radix(e.value, 8, digits);
        for (uint8_t lane = 0; lane < digits.lane_capacity; ++lane) {
            if (digits.digits[lane] != 0) {
                active.insert({ lane, (uint32_t) e.row });
            }
        }
    }
    expected_shape s;
    for (const auto & b : blocks) {
        s.runs.push_back({ b.first, std::vector<uint16_t>(b.second.begin(), b.second.end()) });
    }
    for (const auto & r : active) { // std::set orders (lane, row): lane-major
        if (r.second < rows) {
            s.rows.push_back(r);
        }
    }
    return s;
}

bool shape_matches(const residual::request & req, const expected_shape & s) {
    if (req.runs.size() != s.runs.size() || req.rows.size() != s.rows.size() || req.k_execution != 32 * s.runs.size()) {
        return false;
    }
    size_t k_compact = 0;
    for (size_t r = 0; r < s.runs.size(); ++r) {
        if (req.runs[r].block_id != s.runs[r].first || req.runs[r].local_k != s.runs[r].second) {
            return false;
        }
        k_compact += s.runs[r].second.size();
    }
    for (size_t i = 0; i < s.rows.size(); ++i) {
        if (req.rows[i].lane != s.rows[i].first || req.rows[i].source_row != s.rows[i].second) {
            return false;
        }
    }
    return req.k_compact == k_compact;
}

// The K32 lowering in the operands themselves: zero A/B tails behind every run.
bool padded_operands_ok(const fixture & f, const residual::request & req) {
    for (size_t r = 0; r < req.runs.size(); ++r) {
        const size_t c_r = req.runs[r].local_k.size();
        for (size_t kk = 0; kk < 32; ++kk) {
            const size_t k = r * 32 + kk;
            for (size_t n = 0; n < f.N; ++n) {
                const int8_t want = kk < c_r ? (int8_t) f.code_at(req.runs[r].block_id * 32 + req.runs[r].local_k[kk], n)
                                             : (int8_t) 0;
                if (req.b[k * req.ldb + n] != want) {
                    return false;
                }
            }
            for (size_t i = 0; kk >= c_r && i < req.m; ++i) {
                if (req.a[i * req.lda + k] != 0) {
                    return false;
                }
            }
        }
        for (size_t n = 0; n < f.N; ++n) {
            if (req.carriers[r * req.cstride + n] != f.carrier_at(req.runs[r].block_id, n)) {
                return false;
            }
        }
    }
    return true;
}

bool same_request(const residual::request & a, const residual::request & b) {
    if (a.m != b.m || a.n != b.n || a.k_compact != b.k_compact || a.k_execution != b.k_execution ||
        a.lda != b.lda || a.ldb != b.ldb || a.ldc != b.ldc || a.cstride != b.cstride ||
        a.runs.size() != b.runs.size() || a.rows.size() != b.rows.size()) {
        return false;
    }
    for (size_t i = 0; i < a.runs.size(); ++i) {
        if (a.runs[i].block_id != b.runs[i].block_id || a.runs[i].local_k != b.runs[i].local_k) {
            return false;
        }
    }
    for (size_t i = 0; i < a.rows.size(); ++i) {
        if (a.rows[i].lane != b.rows[i].lane || a.rows[i].source_row != b.rows[i].source_row) {
            return false;
        }
    }
    const auto & x = a.plan;
    const auto & y = b.plan;
    const bool plan = std::tie(x.tile_I, x.tile_J, x.tile_K, x.I0, x.J0, x.K0, x.pad_I, x.pad_J, x.pad_K,
                              x.spad_rows, x.max_spad_rows, x.acc_rows, x.max_acc_rows, x.passes_per_block, x.scale_slots) ==
                      std::tie(y.tile_I, y.tile_J, y.tile_K, y.I0, y.J0, y.K0, y.pad_I, y.pad_J, y.pad_K,
                              y.spad_rows, y.max_spad_rows, y.acc_rows, y.max_acc_rows, y.passes_per_block, y.scale_slots);
    const size_t m_pad = align_up(a.m, D);
    return plan && std::memcmp(a.a, b.a, m_pad * a.lda) == 0 &&
           std::memcmp(a.b, b.b, a.k_execution * a.ldb) == 0 &&
           std::memcmp(a.carriers, b.carriers, a.runs.size() * a.cstride * sizeof(uint32_t)) == 0 &&
           std::memcmp(a.c, b.c, m_pad * a.ldc * sizeof(int32_t)) == 0;
}

bool buffer_padding_ok(const residual::request & req) {
    if (!req.a || !req.b || !req.carriers || !req.c ||
        reinterpret_cast<uintptr_t>(req.a) % POTAL_HW_MAX_BYTES != 0 ||
        reinterpret_cast<uintptr_t>(req.b) % POTAL_HW_MAX_BYTES != 0 ||
        reinterpret_cast<uintptr_t>(req.carriers) % POTAL_HW_MAX_BYTES != 0 ||
        reinterpret_cast<uintptr_t>(req.c) % POTAL_HW_MAX_BYTES != 0 ||
        req.lda != align_up(req.k_execution, POTAL_HW_MAX_BYTES) || req.ldb < req.n ||
        req.cstride < req.n || req.ldc != align_up(req.n, D)) {
        return false;
    }
    for (size_t i = 0; i < align_up(req.m, D); ++i) {
        for (size_t k = i < req.m ? req.k_execution : 0; k < req.lda; ++k) {
            if (req.a[i * req.lda + k] != 0) {
                return false;
            }
        }
        for (size_t n = 0; n < req.ldc; ++n) {
            if (req.c[i * req.ldc + n] != 0) {
                return false;
            }
        }
    }
    for (size_t k = 0; k < req.k_execution; ++k) {
        for (size_t n = req.n; n < req.ldb; ++n) {
            if (req.b[k * req.ldb + n] != 0) {
                return false;
            }
        }
    }
    for (size_t r = 0; r < req.runs.size(); ++r) {
        for (size_t n = req.n; n < req.cstride; ++n) {
            if (req.carriers[r * req.cstride + n] != 0) {
                return false;
            }
        }
    }
    return true;
}

// One complete case: A shape, B transport == golden, D reconstruction + merge.
void run_case(const std::string & name, const fixture & f, size_t row_begin, size_t rows,
              const std::vector<entry> & entries, int16_t theta) {
    const std::string        tag    = name;
    grmd::StripePacketHandle packet = make_packet(3, row_begin, rows, f.K, f.N, entries);
    if (!f.ok || packet == nullptr) {
        check(tag + ": fixture and packet", false);
        return;
    }
    potal_test::emu_session es;
    residual_run            run;
    run.built = residual::build_request(*packet, f.weight, run.req);
    residual::decoded_packet decoded;
    const grmd::RmdStatus decoded_status = residual::decode_packet(*packet, f.weight, decoded);
    potal_test::emu_session decoded_es;
    residual_run            decoded_run;
    // Reuse a populated destination: lowering must reset buffers and metadata,
    // not retain old C values or append runs/rows.
    const grmd::RmdStatus seeded = residual::build_request(*packet, f.weight, decoded_run.req);
    if (seeded == grmd::RmdStatus::success) {
        std::fill(decoded_run.req.c, decoded_run.req.c + align_up(decoded_run.req.m, D) * decoded_run.req.ldc, 123);
        decoded_run.req.runs.push_back({});
        decoded_run.req.rows.push_back({});
    }
    if (decoded_status == grmd::RmdStatus::success) {
        decoded_run.built = residual::build_request_from_decoded(std::move(decoded), f.weight, decoded_run.req);
    }
    const bool both_built = run.built == grmd::RmdStatus::success &&
                            decoded_run.built == grmd::RmdStatus::success;
    check(tag + ": public and decoded-move lowering match all metadata, plan fields and full padded buffers",
          both_built && seeded == grmd::RmdStatus::success && same_request(run.req, decoded_run.req));
    check(tag + ": both requests have aligned buffers and zero DMA padding and initial C",
          both_built && buffer_padding_ok(run.req) && buffer_padding_ok(decoded_run.req));
    execute_request(*packet, es, run);
    execute_request(*packet, decoded_es, decoded_run);
    check(tag + ": request = runs in block order, compact K, lane-major rows",
          both_built && shape_matches(run.req, expect(entries, rows)) &&
              shape_matches(decoded_run.req, expect(entries, rows)));
    check(tag + ": K32 padded operands gathered from the prepared weight",
          both_built && padded_operands_ok(f, run.req) && padded_operands_ok(f, decoded_run.req));

    std::vector<potal_test::golden_lane_row> g;
    const bool golden = golden_of(f, *packet, g);
    check(tag + ": padded Residual GEMM == compact-run golden (INT32 bit-exact), " +
              std::to_string(run.req.m) + " lane rows, K " + std::to_string(run.req.k_compact) + " -> " +
              std::to_string(run.req.k_execution),
          run.gemm == POTAL_OK && golden && same_as_golden(run.req, g, f.N) &&
              run.report.fences == run.report.loop_ws && es.emu.scales.empty() && !es.emu.scale_miss);
    check(tag + ": decoded-move Residual GEMM == independent compact-run golden (INT32 bit-exact)",
          decoded_run.gemm == POTAL_OK && golden && same_as_golden(decoded_run.req, g, f.N) &&
              decoded_run.report.fences == decoded_run.report.loop_ws &&
              decoded_es.emu.scales.empty() && !decoded_es.emu.scale_miss);

    grmd::BlockScaledInt64Correction corr;
    std::vector<int64_t>             want_r;
    const bool                       composed = run.gemm == POTAL_OK &&
                          residual::reconstruct(*packet, run.req, corr) == grmd::RmdStatus::success;
    const bool golden_r = golden && potal_test::compose_golden(g, rows, f.N, want_r);
    check(tag + ": radix INT64 reconstruction == golden (row/lane mapping)",
          composed && golden_r && corr.values == want_r);
    grmd::BlockScaledInt64Correction decoded_corr;
    const bool decoded_composed = decoded_run.gemm == POTAL_OK &&
        residual::reconstruct(*packet, decoded_run.req, decoded_corr) == grmd::RmdStatus::success;
    check(tag + ": decoded-move radix INT64 reconstruction == independent golden",
          decoded_composed && golden_r && decoded_corr.values == want_r);

    std::vector<float> before(f.M * f.N);
    for (size_t i = 0; i < before.size(); ++i) {
        before[i] = 0.125f * (float) ((int) (i % 17) - 8);
    }
    const merge_outcome      m       = merge_both(f, row_begin, row_begin + rows, theta, corr, before);
    const std::vector<float> out     = m.prepared_out;
    const size_t             nonzero = m.prepared_nonzero;
    const bool               merged  = composed && m.ok();
    check(tag + ": prepared merge == Gemmini checked merge (status, nonzero, every output bit)", composed && m.same());
    const merge_outcome decoded_m = merge_both(f, row_begin, row_begin + rows, theta, decoded_corr, before);
    check(tag + ": decoded-move prepared merge == Gemmini checked merge (status, nonzero, every output bit)",
          decoded_composed && decoded_m.ok() && decoded_m.same());
    bool   exact    = merged && golden_r;
    bool decoded_exact = decoded_composed && decoded_m.ok() && golden_r;
    size_t want_nz  = 0;
    for (size_t m = 0; exact && m < f.M; ++m) {
        for (size_t n = 0; exact && n < f.N; ++n) {
            const size_t i     = m * f.N + n;
            const bool   in    = m >= row_begin && m < row_begin + rows;
            const float  wantv = in ? potal_test::merge_golden(before[i], want_r[(m - row_begin) * f.N + n], theta,
                                                                f.scales[n])
                                    : before[i];
            want_nz += in && want_r[(m - row_begin) * f.N + n] != 0;
            exact = std::memcmp(&out[i], &wantv, sizeof(float)) == 0;
            decoded_exact = decoded_exact &&
                std::memcmp(&decoded_m.prepared_out[i], &wantv, sizeof(float)) == 0;
        }
    }
    check(tag + ": checked merge == golden F32, only the stripe's rows, nonzero correction merged",
          exact && nonzero == want_nz && want_nz > 0);
    check(tag + ": decoded-move checked merge == independent golden F32, only the stripe's rows",
          exact && decoded_exact && decoded_m.prepared_nonzero == want_nz && want_nz > 0);
}

// One nonzero residual per local K of `block` in one row: |base| .. |base| + 4
// with the sign of base (lane 0 only while |base| + 4 < 128).
std::vector<entry> run_of(size_t row, size_t block, const std::vector<size_t> & local, int32_t base) {
    std::vector<entry> e;
    for (size_t i = 0; i < local.size(); ++i) {
        const int32_t step = (int32_t) (i % 5);
        e.push_back({ row, block * 32 + local[i], base > 0 ? base + step : base - step });
    }
    return e;
}

std::vector<size_t> first_k(size_t count, size_t stride = 1) {
    std::vector<size_t> v;
    for (size_t i = 0; i < count; ++i) {
        v.push_back((i * stride) % 32);
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// ---- A, B, D -------------------------------------------------------------------

void test_runs() {
    const auto shift3 = [](size_t, size_t) { return (int16_t) 3; };
    fixture    f(40, 40, 256, 1, shift3, random_code);
    for (size_t c_r : { 1u, 15u, 16u, 17u, 31u, 32u }) {
        run_case("A/B/D single run c_r=" + std::to_string(c_r), f, 16, 12, run_of(0, 2, first_k(c_r), 9), -6);
    }
    std::vector<entry> two = run_of(1, 1, first_k(1), 7);
    for (const entry & e : run_of(1, 4, first_k(31), -5)) {
        two.push_back(e);
    }
    run_case("A/B/D runs [1, 31] in blocks 1, 4", f, 16, 12, two, -6);
    std::vector<entry> three = run_of(2, 0, first_k(17), 3);
    for (const entry & e : run_of(2, 2, { 3, 9, 30 }, -40)) {
        three.push_back(e);
    }
    for (const entry & e : run_of(5, 7, first_k(32), 11)) {
        three.push_back(e);
    }
    run_case("A/B/D runs [17, 3, 32] in blocks 0, 2, 7, rows 2 and 5", f, 0, 16, three, -4);

    // Non-consecutive K and source rows; lane 0, the carry lanes and mixed signs.
    std::vector<entry> lanes = {
        { 1, 3 * 32 + 2, 255 },             // [-1, 1]: lanes 0 and 1
        { 4, 3 * 32 + 5, -300 },            // lanes 0 and 1
        { 9, 3 * 32 + 30, INT32_MAX },      // [-1, 0, 0, -128, 1]: lanes 0, 3, 4
        { 9, 6 * 32 + 17, INT32_MIN + 1 },  // the other extreme
        { 4, 6 * 32 + 1, 100 },
    };
    run_case("A/B/D non-consecutive K and rows, lanes 0..4", f, 24, 12, lanes, -20);

    std::vector<entry> dense;
    // Both radix lanes cross an output-tile boundary, including DIM32.
    // D+1 source rows produce 2*(D+1) lane rows: three padded I tiles.
    for (size_t row = 0; row < D + 1; ++row) {
        for (size_t k = 0; k < 64; ++k) {
            dense.push_back({ row, k, (row + k) % 2 ? -300 : 255 });
        }
    }
    run_case("A/B/D dense two-run multi-lane rows across DIM boundary", f, 0, D + 1, dense, -12);
}

void test_carriers() {
    // Per (block, column): zero, shift 0, shift 5, large valid shifts (31, 40).
    const auto mixed = [](size_t blk, size_t n) {
        switch ((blk + n) % 5) {
            case 0:  return std::numeric_limits<int16_t>::min();
            case 1:  return (int16_t) 0;
            case 2:  return (int16_t) 5;
            case 3:  return (int16_t) 31;
            default: return (int16_t) 40;
        }
    };
    const auto strong = [](size_t, size_t n, std::mt19937 &) { return (int8_t) (n % 2 ? -127 : 127); };
    fixture    f(32, 48, 192, 2, mixed, strong);
    std::vector<entry> e = run_of(0, 1, first_k(20), 120);
    for (const entry & x : run_of(3, 1, first_k(9, 3), -120)) {
        e.push_back(x);
    }
    for (const entry & x : run_of(3, 5, first_k(32), 64)) {
        e.push_back(x);
    }
    run_case("B carriers zero/0/5/31/40, positive and negative SCU saturation", f, 0, 16, e, -12);
}

// ---- C ---------------------------------------------------------------------------

void test_cancellation() {
    // Run 0 (block 0, carrier 20) saturates high; run 1 (block 1, carrier 10) is
    // negative. Sat32 order: INT32_MAX + q1. Exact per-block sum first: still
    // above INT32_MAX, so it saturates to INT32_MAX.
    const auto carriers = [](size_t blk, size_t) { return (int16_t) (blk == 0 ? 20 : blk == 1 ? 10 : 2); };
    const auto codes    = [](size_t k, size_t, std::mt19937 &) {
        return (int8_t) (k < 16 ? 127 : (k >= 32 && k < 48) ? -127 : 1);
    };
    fixture            f(16, 16, 64, 3, carriers, codes);
    std::vector<entry> e = run_of(0, 0, first_k(16), 100);
    for (entry & x : e) {
        x.value = 100;
    }
    for (size_t k = 32; k < 48; ++k) {
        e.push_back({ 0, k, 100 });
    }
    grmd::StripePacketHandle packet = make_packet(0, 0, 16, 64, 16, e);
    potal_test::emu_session  es;
    residual_run             run;
    if (packet != nullptr) {
        execute(f, *packet, es, run);
    }
    const int64_t p0       = 16 * 100 * 127;
    const int64_t p1       = -16 * 100 * 127;
    const int32_t q1       = (int32_t) (p1 * 1024); // carrier 10: no SCU saturation
    const int32_t sat32    = (int32_t) ((int64_t) INT32_MAX + q1);
    const int64_t exact    = p0 * ((int64_t) 1 << 20) + p1 * ((int64_t) 1 << 10);
    const int32_t per_blk  = exact > INT32_MAX ? INT32_MAX : (int32_t) exact;
    const bool    ran      = packet != nullptr && run.gemm == POTAL_OK && run.req.m == 1;
    check("C the case is meaningful: Sat32 run order != exact per-block sum", sat32 != per_blk);
    check("C padded transport keeps the Sat32 run order (INT32_MAX + q1)", ran && run.req.c[0] == sat32);
    std::vector<potal_test::golden_lane_row> g;
    check("C golden keeps it too", golden_of(f, *packet, g) && g.size() == 1 && g[0].c[0] == sat32 &&
                                       ran && same_as_golden(run.req, g, f.N));
}

// ---- E ---------------------------------------------------------------------------

void test_failures() {
    const auto shift2 = [](size_t, size_t) { return (int16_t) 2; };
    fixture    f(32, 32, 128, 4, shift2, random_code);
    std::vector<entry> e = run_of(0, 1, first_k(5), 50);
    e.push_back({ 3, 2 * 32 + 4, INT32_MAX }); // lanes 0, 3, 4 of row 3
    grmd::StripePacketHandle packet = make_packet(0, 0, 16, f.K, f.N, e);
    if (!f.ok || packet == nullptr) {
        check("E fixture", false);
        return;
    }

    {
        grmd::StripePacket bad = *packet;
        bad.version            = 99;
        residual::request req;
        check("E invalid packet (version) refused", residual::build_request(bad, f.weight, req) ==
                                                        grmd::RmdStatus::invalid_packet);
    }
    {
        grmd::StripePacket bad = *packet;
        bad.k_indices[0]       = 40;
        residual::request req;
        check("E invalid packet (block-local K beyond 32) refused",
              residual::build_request(bad, f.weight, req) != grmd::RmdStatus::success);
    }
    {
        grmd::StripePacketHandle a4 = make_packet(0, 0, 16, f.K, f.N, run_of(0, 1, first_k(3), 3), 4);
        residual::request        req;
        check("E A4 packet refused (PoTal is A8W8)",
              a4 != nullptr && residual::build_request(*a4, f.weight, req) == grmd::RmdStatus::unsupported_route);
    }
    {
        grmd::StripePacketHandle other = make_packet(0, 0, 16, f.K, f.N + 16, run_of(0, 1, first_k(3), 3));
        residual::request        req;
        check("E packet of another shape (wrong stripe/node) refused",
              other != nullptr && residual::build_request(*other, f.weight, req) == grmd::RmdStatus::invalid_arguments);
    }
    {
        potal_test::emu_session es;
        residual_run            run;
        run.built = residual::build_request(*packet, f.weight, run.req);
        run.req.carriers[0] = 40000; // not a PoTal carrier
        potal_stripe_gemm_desc d = {};
        d.work_kind              = POTAL_STRIPE_WORK_RESIDUAL;
        d.m = (uint32_t) run.req.m, d.n = (uint32_t) run.req.n, d.k = (uint32_t) run.req.k_execution;
        d.a = run.req.a, d.b = run.req.b, d.c = run.req.c;
        d.lda = run.req.lda, d.ldb = run.req.ldb, d.ldc = run.req.ldc;
        d.carriers = run.req.carriers, d.carrier_stride = run.req.cstride, d.scale_block_k = 32;
        d.tile_i = (uint32_t) run.req.plan.tile_I, d.tile_j = (uint32_t) run.req.plan.tile_J;
        d.tile_k = (uint32_t) run.req.plan.tile_K;
        const potal_status st = potal_tiled_matmul_stripe(&es.session, &d, &run.report);
        check("E invalid carrier refused before any command",
              run.built == grmd::RmdStatus::success && st == POTAL_ERR_INVALID_ARGUMENT && es.emu.writes == 0 &&
                  es.emu.loops == 0);
    }

    potal_test::emu_session es;
    residual_run            run;
    execute(f, *packet, es, run);
    const bool ran = run.gemm == POTAL_OK && run.req.m >= 3;
    check("E fixture GEMM ran", ran);
    if (!ran) {
        return;
    }
    grmd::BlockScaledInt64Correction corr;
    {
        std::vector<residual::lane_row> saved = run.req.rows;
        run.req.rows[1] = run.req.rows[0];
        const grmd::RmdStatus dup = residual::reconstruct(*packet, run.req, corr);
        std::swap(run.req.rows[0], run.req.rows[2]);
        const grmd::RmdStatus order = residual::reconstruct(*packet, run.req, corr);
        run.req.rows                = saved;
        run.req.rows.back().source_row = (uint32_t) packet->row_count;
        const grmd::RmdStatus range = residual::reconstruct(*packet, run.req, corr);
        run.req.rows                = saved;
        check("E duplicate / unordered / out-of-range lane rows refused, correction untouched",
              dup == grmd::RmdStatus::invalid_packet && order == grmd::RmdStatus::invalid_packet &&
                  range == grmd::RmdStatus::invalid_packet && corr.values.empty());
    }
    {
        // Row 3 carries lanes 0, 3, 4 (INT32_MAX). Overwrite its GEMM results.
        size_t l3 = SIZE_MAX, l4 = SIZE_MAX, l0 = SIZE_MAX;
        for (size_t i = 0; i < run.req.m; ++i) {
            if (run.req.rows[i].source_row == 3) {
                (run.req.rows[i].lane == 0 ? l0 : run.req.rows[i].lane == 3 ? l3 : l4) = i;
            }
        }
        const bool found = l0 != SIZE_MAX && l3 != SIZE_MAX && l4 != SIZE_MAX;
        if (found) {
            for (size_t i : { l0, l3, l4 }) {
                std::fill(run.req.c + i * run.req.ldc, run.req.c + i * run.req.ldc + f.N, 0);
            }
            run.req.c[l4 * run.req.ldc] = INT32_MIN; // -2^63 exactly: fits
        }
        const bool edge = found && residual::reconstruct(*packet, run.req, corr) == grmd::RmdStatus::success &&
                          corr.values[3 * f.N] == std::numeric_limits<int64_t>::min();
        grmd::BlockScaledInt64Correction untouched = corr;
        if (found) {
            run.req.c[l0 * run.req.ldc] = -1; // -2^63 - 1
        }
        const bool low = found && residual::reconstruct(*packet, run.req, corr) == grmd::RmdStatus::overflow;
        if (found) {
            run.req.c[l0 * run.req.ldc] = 0;
            run.req.c[l4 * run.req.ldc] = INT32_MAX;
            run.req.c[l3 * run.req.ldc] = INT32_MAX; // 2^63 - 2^32 + 2^55 - 2^24
        }
        const bool high = found && residual::reconstruct(*packet, run.req, corr) == grmd::RmdStatus::overflow;
        check("E INT64 composition: -2^63 fits, below and above INT64 refused (transactional)",
              edge && low && high && corr.values == untouched.values);
    }
    {
        execute(f, *packet, es, run); // fresh GEMM results
        grmd::BlockScaledInt64Correction good;
        const bool composed = residual::reconstruct(*packet, run.req, good) == grmd::RmdStatus::success;
        const std::vector<float> out(f.M * f.N, 1.0f);
        const merge_outcome      inf  = merge_both(f, 0, 16, 127, good, out); // 2^127 x R: not finite as F32
        const merge_outcome none = merge_both(f, 0, 16, std::numeric_limits<int16_t>::min(), good, out); // no theta
        check("E non-finite merge and missing theta refused, output unchanged",
              composed && inf.prepared == grmd::RmdStatus::overflow && none.prepared != grmd::RmdStatus::success &&
                  inf.same() && none.same() && inf.untouched(out) && none.untouched(out));
    }
    {
        grmd::RmdStripeBuilder b;
        b.reset(0, 0, 16, f.K, f.N, 8);
        b.add_residual(0, 3, 0);
        check("E a stripe without nonzero residual seals no packet (residual GEMM skipped)", b.finish() == nullptr);
    }
}

// ---- M ---------------------------------------------------------------------------

// Correction of `rows` x N stripe-local values: value(row, col).
grmd::BlockScaledInt64Correction correction_of(size_t rows, size_t N,
                                               const std::function<int64_t(size_t, size_t)> & value) {
    grmd::BlockScaledInt64Correction c;
    c.values.resize(rows * N);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t n = 0; n < N; ++n) {
            c.values[r * N + n] = value(r, n);
        }
    }
    return c;
}

std::vector<float> base_output(size_t count) {
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = 0.375f * (float) ((int) (i % 23) - 11);
    }
    return out;
}

void test_merge() {
    const auto shift1 = [](size_t, size_t) { return (int16_t) 1; };
    fixture    f(40, 48, 64, 5, shift1, random_code);
    if (!f.ok) {
        check("M fixture", false);
        return;
    }
    const std::vector<float> base = base_output(f.M * f.N);
    const int64_t            big  = std::numeric_limits<int64_t>::max();
    const int64_t            low  = std::numeric_limits<int64_t>::min();
    // Signs, zeros, small and INT64-extreme values in one stripe.
    const auto mixed = [&](size_t r, size_t n) -> int64_t {
        switch ((r * 7 + n) % 9) {
            case 0:  return 0;
            case 1:  return 1;
            case 2:  return -1;
            case 3:  return (int64_t) (r * 1000 + n);
            case 4:  return -(int64_t) (r * 977 + n * 31);
            case 5:  return big;
            case 6:  return low;
            case 7:  return (int64_t) 1 << 62;
            default: return -((int64_t) 1 << 61) + (int64_t) n;
        }
    };
    const grmd::BlockScaledInt64Correction corr = correction_of(16, f.N, mixed);

    bool   same = true, any_ok = false, any_overflow = false;
    size_t cases = 0;
    for (int16_t theta : { -60, -40, -20, -1, 0, 1, 10, 40, 100 }) {
        const merge_outcome m = merge_both(f, 8, 24, theta, corr, base);
        same                  = same && m.same() && (m.ok() || m.untouched(base));
        any_ok                = any_ok || m.ok();
        any_overflow          = any_overflow || m.prepared == grmd::RmdStatus::overflow;
        cases++;
    }
    check("M1-4,6 signs, zeros, INT64 extremes, theta -60..100 (" + std::to_string(cases) +
              " merges): prepared == Gemmini bitwise, failures untouched",
          same && any_ok && any_overflow);

    const grmd::BlockScaledInt64Correction zeros = correction_of(16, f.N, [](size_t, size_t) { return (int64_t) 0; });
    const merge_outcome                    z     = merge_both(f, 0, 16, -3, zeros, base);
    check("M3 all-zero correction: success, nonzero 0, output bits unchanged (x + 0.0f)",
          z.same() && z.ok() && z.prepared_nonzero == 0 &&
              std::memcmp(z.prepared_out.data(), base.data(), base.size() * sizeof(float)) == 0);

    // Theta at and beyond the F32 scale limits: INT16_MIN (missing), 2^-150
    // (0), 2^128 (inf) refused; 2^-149 (smallest subnormal) and 2^127 run.
    const grmd::BlockScaledInt64Correction small = correction_of(16, f.N, [](size_t r, size_t n) {
        return (int64_t) ((r + 1) * (n % 5)) - 2;
    });
    bool limits = true;
    for (int16_t theta : { std::numeric_limits<int16_t>::min(), (int16_t) -150, (int16_t) 128, (int16_t) 200 }) {
        const merge_outcome m = merge_both(f, 0, 16, theta, small, base);
        limits = limits && m.same() && m.untouched(base) && m.prepared == grmd::RmdStatus::invalid_arguments;
    }
    const merge_outcome tiny = merge_both(f, 0, 16, -149, small, base);
    const merge_outcome edge = merge_both(f, 0, 16, 127, small, base);
    check("M6,8 theta INT16_MIN/-150/128/200 refused (invalid_arguments), -149 and 127 == Gemmini",
          limits && tiny.same() && tiny.ok() && edge.same());

    // Wrong sizes and bounds.
    bool sizes = true;
    for (size_t count : { (size_t) 0, 16 * f.N - 1, 16 * f.N + 1 }) {
        grmd::BlockScaledInt64Correction c = corr;
        c.values.resize(count);
        const merge_outcome m = merge_both(f, 0, 16, -5, c, base);
        sizes = sizes && m.same() && m.untouched(base) && m.prepared == grmd::RmdStatus::invalid_arguments;
    }
    const merge_outcome past  = merge_both(f, 32, 48, -5, corr, base);           // row_end > M
    const merge_outcome order = merge_both(f, 20, 4, -5, corr, base);            // row_begin > row_end
    const merge_outcome empty = merge_both(f, 8, 8, -5, grmd::BlockScaledInt64Correction{}, base); // no rows
    check("M9 wrong correction size (0, -1, +1), rows past M, reversed and empty ranges refused == Gemmini",
          sizes && past.same() && past.untouched(base) && order.same() && order.untouched(base) && empty.same() &&
              empty.untouched(base) && empty.prepared == grmd::RmdStatus::invalid_arguments);

    // A non-finite main output inside the stripe makes its merged cell
    // non-finite (overflow); strides whose last offset overflows size_t are
    // refused before any access (invalid_arguments). Both paths, untouched.
    std::vector<float> inf_out = base, nan_out = base;
    inf_out[(8 + 3) * f.N + 7]  = std::numeric_limits<float>::infinity();
    nan_out[(8 + 3) * f.N + 7]  = std::numeric_limits<float>::quiet_NaN();
    const merge_outcome inf_dst  = merge_both(f, 8, 24, -6, corr, inf_out);
    const merge_outcome nan_dst  = merge_both(f, 8, 24, -6, corr, nan_out);
    const merge_outcome row_over = merge_both(f, 8, 24, -6, corr, base, SIZE_MAX / 2, 1);
    const merge_outcome col_over = merge_both(f, 8, 24, -6, corr, base, f.N, SIZE_MAX / 4);
    check("M8,9,11 Inf/NaN main output in the stripe (overflow) and size_t-overflowing row/column strides "
          "(invalid_arguments) refused == Gemmini, output and count untouched",
          inf_dst.same() && inf_dst.untouched(inf_out) && inf_dst.prepared == grmd::RmdStatus::overflow &&
              nan_dst.same() && nan_dst.untouched(nan_out) && nan_dst.prepared == grmd::RmdStatus::overflow &&
              row_over.same() && row_over.untouched(base) && row_over.prepared == grmd::RmdStatus::invalid_arguments &&
              col_over.same() && col_over.untouched(base) && col_over.prepared == grmd::RmdStatus::invalid_arguments);

    // Strided destination: row stride 2N + 3, column stride 2, stripes at
    // rows 0, 16 and the 8-row tail 32..40, merged one after the other into
    // the same buffer (one weight, several stripes); the gaps never change.
    const size_t       rs = 2 * f.N + 3, cs = 2;
    std::vector<float> strided_p = base_output((f.M - 1) * rs + (f.N - 1) * cs + 1);
    std::vector<float> strided_g = strided_p;
    const std::vector<float> strided_base = strided_p;
    bool strided = true;
    size_t stripes = 0;
    for (const auto & range : { std::make_pair((size_t) 0, (size_t) 16), std::make_pair((size_t) 16, (size_t) 32),
                                std::make_pair((size_t) 32, f.M) }) {
        const grmd::BlockScaledInt64Correction c = correction_of(range.second - range.first, f.N, mixed);
        const merge_outcome m = merge_both(f, range.first, range.second, -24, c, strided_p, rs, cs);
        // Chain: each path continues on its own previous output.
        const merge_outcome g = merge_both(f, range.first, range.second, -24, c, strided_g, rs, cs);
        strided = strided && m.ok() && g.ok() && m.same() && g.same() &&
                  std::memcmp(m.prepared_out.data(), g.gemmini_out.data(), m.prepared_out.size() * sizeof(float)) == 0;
        strided_p = m.prepared_out;
        strided_g = g.gemmini_out;
        stripes++;
    }
    bool gaps = true;
    for (size_t i = 0; i < strided_p.size(); ++i) {
        const bool cell = (i % rs) < f.N * cs && (i % rs) % cs == 0;
        gaps = gaps && (cell || std::memcmp(&strided_p[i], &strided_base[i], sizeof(float)) == 0);
    }
    check("M10,12 row stride 2N+3, column stride 2, " + std::to_string(stripes) +
              " stripes of one weight incl. the 8-row tail: == Gemmini after each, gaps untouched",
          strided && gaps && stripes == 3);

    // Channel scales at the ends of the normal F32 range (1e-30 / 1e30): the
    // huge ones overflow F32 for large corrections, in both paths alike.
    fixture extreme(40, 48, 64, 6, shift1, random_code, [](size_t n) { return n % 2 ? 1e30f : 1e-30f; });
    bool    scales_same = extreme.ok, scales_ok = false, scales_overflow = false;
    for (int16_t theta : { -40, 0, 10 }) {
        for (int64_t magnitude : { (int64_t) 3, (int64_t) 1 << 20, (int64_t) 1 << 40 }) {
            const grmd::BlockScaledInt64Correction c = correction_of(16, extreme.N, [&](size_t r, size_t n) {
                return (r + n) % 3 == 0 ? -magnitude : magnitude + (int64_t) r;
            });
            const merge_outcome m = extreme.ok ? merge_both(extreme, 0, 16, theta, c, base) : merge_outcome{};
            scales_same           = scales_same && m.same() && (m.ok() || m.untouched(base));
            scales_ok             = scales_ok || m.ok();
            scales_overflow       = scales_overflow || m.prepared == grmd::RmdStatus::overflow;
        }
    }
    check("M7 channel scales 1e-30 / 1e30: prepared == Gemmini bitwise (successes and overflows)",
          scales_same && scales_ok && scales_overflow);

    // A non-finite prepared scale cannot come out of prepare_weight(); the
    // prepared merge still refuses it (unsupported_route), output untouched.
    bool nonfinite = true;
    for (float bad : { std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                       -std::numeric_limits<float>::infinity() }) {
        const float saved = f.weight.scale[5];
        f.weight.scale[5] = bad;
        std::vector<float>               out = base;
        size_t                           nz  = SIZE_MAX;
        exsia_ns::StripeMetadataSnapshot snap;
        snap.theta              = -4;
        const grmd::RmdStatus st = residual::merge_prepared_hp1(f.weight, 0, 16, snap, corr, out.data(), f.M, f.N, 1,
                                                                &nz);
        f.weight.scale[5] = saved;
        nonfinite = nonfinite && st == grmd::RmdStatus::unsupported_route && nz == SIZE_MAX &&
                    std::memcmp(out.data(), base.data(), base.size() * sizeof(float)) == 0;
    }
    check("M8,11 NaN/+Inf/-Inf channel scale refused (unsupported_route), output and count untouched", nonfinite);

    // A failure in the last cell leaves every earlier staged cell unwritten.
    grmd::BlockScaledInt64Correction late = correction_of(16, f.N, [](size_t, size_t) { return (int64_t) 7; });
    late.values.back()                    = big; // x 2^100: the last cell overflows
    const merge_outcome last_cell         = merge_both(f, 0, 16, 100, late, base);
    check("M11 overflow in the last cell: nothing committed, count untouched (both paths)",
          last_cell.same() && last_cell.untouched(base) && last_cell.prepared == grmd::RmdStatus::overflow);

    // Weight revalidation: the Gemmini reader's storage validations
    // (wreader::validate) per merge, prepared vs Gemmini path.
    wreader::test_reset_weight_reader_counters();
    std::vector<float>               out = base;
    exsia_ns::StripeMetadataSnapshot snap;
    snap.theta    = -6;
    size_t merges = 0;
    bool   all_ok = true;
    for (size_t round = 0; round < 8; ++round) {
        for (size_t row = 0; row + 16 <= f.M; row += 16) {
            size_t nz = 0;
            all_ok    = all_ok && residual::merge_prepared_hp1(f.weight, row, row + 16, snap, corr, out.data(), f.M,
                                                               f.N, 1, &nz) == grmd::RmdStatus::success;
            merges++;
        }
    }
    const size_t prepared_validations = wreader::test_weight_reader_storage_validations();
    const size_t prepared_reads       = wreader::test_weight_reader_code_address_resolutions();
    wreader::test_reset_weight_reader_counters();
    const merge_outcome reference          = merge_both(f, 0, 16, -6, corr, base);
    const size_t        gemmini_validations = wreader::test_weight_reader_storage_validations();
    std::printf("     weight validations: prepared %zu over %zu merges, Gemmini merge %zu per merge\n",
                prepared_validations, merges, gemmini_validations);
    check("M13 " + std::to_string(merges) + " prepared merges of one weight: no wreader::validate(), no weight read "
          "(the Gemmini merge validates the whole weight on every call)",
          all_ok && prepared_validations == 0 && prepared_reads == 0 && reference.same() && gemmini_validations >= 1);
}

} // namespace

int main() {
    std::printf("potal-residual-unit-test DIM=%u backend=EMULATED (no hardware) radix=256 lanes=%u\n",
                (unsigned) POTAL_HW_DIM, (unsigned) grmd::balanced_radix_contract(8).lane_capacity);
    test_runs();
    test_carriers();
    test_cancellation();
    test_failures();
    test_merge();
    if (failures != 0) {
        std::printf("POTAL_RESIDUAL_UNIT_TEST FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("POTAL_RESIDUAL_UNIT_TEST PASS\n");
    return 0;
}
