#pragma once

// Shared pieces of the paired GEMM host tests (potal-paired-*-test):
//
//   paired_fixture   a Q8_HP1 weight N x K built block by block (codes,
//                    exponent, channel scale), prepared once (weight_cache
//                    layout), and Main A8 codes M x K
//   packets          sealed RMD packets per stripe (RmdStripeBuilder)
//   A  golden        independent of every production lowering: Main by
//                    DIM K passes with the emulator's SCU/Sat32 (RTL test
//                    double), Residual / INT64 / legacy merge by
//                    support/stripe-golden.hpp, INT64 merge by its formula
//   B  serialized    the existing path: potal_tiled_matmul_auto (Main) and
//                    residual::build_request + potal_tiled_matmul_auto
//                    (Residual, K32 padded) on the RTL-semantics emulator,
//                    residual::reconstruct, reconstruct_row + merge_prepared_hp1
//   C  paired        paired_tiled_matmul_auto (PAIRED_EMULATED)

#include "ggml-cpp.h"
#include "ggml-quants.h"
#include "matmul/dense.hpp"
#include "matmul/paired.hpp"
#include "matmul/tiling.hpp"
#include "potal.h"
#include "potal_hw_params.h"
#include "residual/stripe-residual.hpp"
#include "support/potal-emulator.hpp"
#include "support/stripe-golden.hpp"

#include "quants/act/exsia/exsia-event.hpp"
#include "residual/rmd/rmd-builder.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace potal_test {

namespace gp = ggml::potal;

inline int g_failures = 0;

inline void check(const std::string & name, bool ok) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    g_failures += ok ? 0 : 1;
}

constexpr size_t PD = POTAL_HW_DIM;

struct res_entry {
    size_t  row; // stripe-local
    size_t  k;   // original K
    int32_t value;
};

using exponent_fn = std::function<int16_t(size_t blk, size_t n)>;
using code_fn     = std::function<int8_t(size_t k, size_t n, std::mt19937 &)>;

inline int8_t random_i8(size_t, size_t, std::mt19937 & rng) {
    return (int8_t) std::uniform_int_distribution<int>(-128, 127)(rng);
}

struct dma_deleter {
    void operator()(void * p) const { std::free(p); }
};

struct paired_fixture {
    size_t               M, N, K;
    ggml_context_ptr     ctx;
    ggml_tensor *        w = nullptr;
    ggml_tensor *        x = nullptr;
    ggml_tensor *        y = nullptr;
    std::vector<int8_t>  codes; // [K][N]
    std::vector<int16_t> exps;  // [K / 32][N]; INT16_MIN: zero carrier
    std::vector<float>   scales;
    std::unique_ptr<int8_t, dma_deleter> a; // [M][lda]
    size_t               lda = 0;
    gp::hp1_weight       weight;
    bool                 ok = false;

    paired_fixture(size_t M_, size_t N_, size_t K_, uint32_t seed, const exponent_fn & exponent,
                   const code_fn & code = random_i8, const code_fn & act = random_i8) :
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
        auto * blocks = static_cast<block_q8_hp1 *>(w->data);
        for (size_t n = 0; n < N; ++n) {
            scales[n] = 0.003f * (float) (1 + n % 7);
            for (size_t blk = 0; blk < K / 32; ++blk) {
                block_q8_hp1 & b = blocks[n * (K / 32) + blk];
                std::memset(&b, 0, sizeof(b));
                for (size_t i = 0; i < 32; ++i) {
                    const size_t k   = blk * 32 + i;
                    codes[k * N + n] = code(k, n, rng);
                    b.qs[i]          = codes[k * N + n];
                }
                exps[blk * N + n] = exponent(blk, n);
                b.m               = exps[blk * N + n];
                b.channel_scale   = scales[n];
            }
        }
        lda = gp::align_up(K, POTAL_HW_MAX_BYTES);
        a.reset(static_cast<int8_t *>(gp::dma_alloc(gp::align_up(M, PD) * lda)));
        for (size_t m = 0; m < M && a; ++m) {
            for (size_t k = 0; k < K; ++k) {
                a.get()[m * lda + k] = act(m, k, rng);
            }
        }
        gp::hp1_shape           s = { (int64_t) M, (int64_t) N, (int64_t) K };
        gp::potal_tile_plan     plan;
        ggml_gemmini_args_t     args;
        ok = a != nullptr && gp::potal_tile_select(M, N, K, 32 / PD, &plan) == POTAL_OK &&
             gp::init_args(w, y, s, "gpt2", "paired-test", ggml::gemmini::residual::ResidualRoute::ws_packet, plan,
                           args) &&
             weight.allocate((int64_t) N, (int64_t) K) &&
             gp::prepare_weight(args, weight) == gp::quants::weight::hp1::status::ok;
    }

    int32_t  code_at(size_t k, size_t n) const { return codes[k * N + n]; }
    uint32_t carrier_at(size_t blk, size_t n) const {
        const int16_t e = exps[blk * N + n];
        return e == std::numeric_limits<int16_t>::min() ? POTAL_CARRIER_ZERO : (uint32_t) e;
    }
};

inline grmd::StripePacketHandle make_packet(size_t stripe_id, size_t row_begin, size_t rows, size_t K, size_t N,
                                            const std::vector<res_entry> & entries) {
    grmd::RmdStripeBuilder b;
    b.reset(stripe_id, row_begin, rows, K, N, 8);
    for (const res_entry & e : entries) {
        if (!b.add_residual(e.row, e.k, e.value)) {
            return nullptr;
        }
    }
    return b.finish();
}

// One test case's stripes: theta and packet (nullptr: none).
struct paired_inputs {
    std::vector<grmd::StripePacketHandle> packets;
    std::vector<int16_t>                  theta;
    std::vector<gp::paired_stripe_input>  stripes;

    void bind() {
        stripes.resize(theta.size());
        for (size_t s = 0; s < theta.size(); ++s) {
            stripes[s].theta  = theta[s];
            stripes[s].packet = packets[s].get();
        }
    }
};

using entries_fn = std::function<std::vector<res_entry>(size_t stripe, size_t rows)>;

inline bool make_inputs(const paired_fixture & f, size_t stripe_rows, const entries_fn & entries, paired_inputs & in) {
    const size_t stripes = (f.M + stripe_rows - 1) / stripe_rows;
    in.packets.assign(stripes, nullptr);
    in.theta.assign(stripes, 0);
    for (size_t s = 0; s < stripes; ++s) {
        const size_t rows = std::min(stripe_rows, f.M - s * stripe_rows);
        in.theta[s]       = (int16_t) (-6 + (int) (s % 5)); // negative, zero and positive
        if (entries) {
            const std::vector<res_entry> e = entries(s, rows);
            if (!e.empty()) {
                in.packets[s] = make_packet(s, s * stripe_rows, rows, f.K, f.N, e);
                if (in.packets[s] == nullptr) {
                    return false;
                }
            }
        }
    }
    in.bind();
    return true;
}

inline gp::paired_gemm_desc desc_of(const paired_fixture & f, const gp::paired_tile_plan & plan,
                                    const paired_inputs & in) {
    gp::paired_gemm_desc d;
    d.m            = f.M;
    d.n            = f.N;
    d.k            = f.K;
    d.a            = f.a.get();
    d.lda          = f.lda;
    d.weight       = &f.weight;
    d.stripe_rows  = plan.stripe_rows;
    d.stripes      = in.stripes.data();
    d.stripe_count = in.stripes.size();
    if (plan.frozen) {
        d.tile_I = plan.tile_I;
        d.tile_J = plan.tile_J;
        d.tile_K = plan.requested_tile_K;
    }
    return d;
}

// ---- A: independent golden ---------------------------------------------------------

struct golden_result {
    bool                                       ok = false;       // every stripe composed in INT64
    std::vector<int32_t>                       main_c;            // M x N
    std::vector<std::vector<golden_lane_row>>  residual;          // per stripe
    std::vector<std::vector<int64_t>>          correction;        // per stripe (empty: none)
    std::vector<size_t>                        overflow_stripes;  // stripes whose R left INT64
    std::vector<float>                         legacy, int64_merge;
    bool                                       int64_merge_ok = true;
};

inline golden_result golden_of(const paired_fixture & f, size_t stripe_rows, const paired_inputs & in) {
    golden_result g;
    g.main_c.assign(f.M * f.N, 0);
    for (size_t m = 0; m < f.M; ++m) {
        for (size_t n = 0; n < f.N; ++n) {
            int32_t acc = 0;
            for (size_t p = 0; p < f.K / PD; ++p) {
                int32_t partial = 0;
                for (size_t kk = 0; kk < PD; ++kk) {
                    const size_t k = p * PD + kk;
                    partial += (int32_t) f.a.get()[m * f.lda + k] * f.code_at(k, n);
                }
                acc = emulator::sat_add(acc, emulator::scu(partial, f.carrier_at(p * PD / 32, n)));
            }
            g.main_c[m * f.N + n] = acc;
        }
    }
    const size_t stripes = in.theta.size();
    g.residual.resize(stripes);
    g.correction.resize(stripes);
    g.ok = true;
    for (size_t s = 0; s < stripes; ++s) {
        if (in.packets[s] == nullptr) {
            continue;
        }
        if (!residual_golden(
                *in.packets[s], f.N, PD, [&f](size_t k, size_t n) { return f.code_at(k, n); },
                [&f](size_t blk, size_t n) { return f.carrier_at(blk, n); }, g.residual[s])) {
            g.ok = false;
        } else if (!compose_golden(g.residual[s], in.packets[s]->row_count, f.N, g.correction[s])) {
            g.overflow_stripes.push_back(s);
            g.correction[s].clear();
            g.ok = false;
        }
    }
    if (!g.ok) {
        return g;
    }
    g.legacy.assign(f.M * f.N, 0.0f);
    g.int64_merge.assign(f.M * f.N, 0.0f);
    for (size_t m = 0; m < f.M; ++m) {
        const size_t  s     = m / stripe_rows;
        const size_t  row   = m - s * stripe_rows;
        const int16_t theta = in.theta[s];
        for (size_t n = 0; n < f.N; ++n) {
            const int32_t c = g.main_c[m * f.N + n];
            const int64_t r = g.correction[s].empty() ? 0 : g.correction[s][row * f.N + n];
            float         v = main_golden(c, theta, f.scales[n]);
            if (in.packets[s] != nullptr) {
                v = merge_golden(v, r, theta, f.scales[n]);
            }
            g.legacy[m * f.N + n] = v;
            golden_wide y         = (golden_wide) c + r;
            if (y > std::numeric_limits<int64_t>::max() || y < std::numeric_limits<int64_t>::min()) {
                g.int64_merge_ok = false;
                continue;
            }
            g.int64_merge[m * f.N + n] =
                (float) ((double) (int64_t) y * ((double) f.scales[n] * (double) std::ldexp(1.0f, theta)));
        }
    }
    return g;
}

// ---- B: serialized Main / Residual (existing path, emulator) -----------------------

struct serialized_result {
    bool                                         main_ok = false;
    std::vector<int32_t>                         main_c; // M x N
    std::vector<grmd::RmdStatus>                 built, composed;
    std::vector<potal_status>                    gemm;
    std::vector<std::vector<gp::residual::lane_row>> rows;
    std::vector<std::vector<int32_t>>            residual_c; // per stripe, rows x N
    std::vector<std::vector<int64_t>>            correction;
    std::vector<float>                           legacy;
    bool                                         legacy_ok = false;
};

inline serialized_result serialized_of(const paired_fixture & f, size_t stripe_rows, const paired_inputs & in) {
    serialized_result r;
    const size_t      ldc = gp::align_up(f.N, PD);
    std::unique_ptr<int32_t, dma_deleter> c(
        static_cast<int32_t *>(gp::dma_alloc(gp::align_up(f.M, PD) * ldc * sizeof(int32_t))));
    emulator                     emu;
    gp::potal_ws_ops             ops     = emu.ops();
    gp::potal_ws_session         session = { &ops, 0, {} };
    const gp::potal_scale_operand scales = { f.weight.carriers, f.weight.cstride, 32 };
    r.main_ok = c && gp::potal_tiled_matmul_auto(f.M, f.N, f.K, f.a.get(), f.weight.b, nullptr, c.get(), f.lda,
                                                 f.weight.ldb, 0, ldc, 0, 0, 0, gp::POTAL_NO_ACTIVATION, 0, 0, false,
                                                 false, false, true, false, 0, gp::POTAL_WS, scales,
                                                 session) == POTAL_OK;
    r.main_c.assign(f.M * f.N, 0);
    for (size_t m = 0; m < f.M && r.main_ok; ++m) {
        std::memcpy(r.main_c.data() + m * f.N, c.get() + m * ldc, f.N * sizeof(int32_t));
    }
    const size_t stripes = in.theta.size();
    r.built.assign(stripes, grmd::RmdStatus::success);
    r.composed.assign(stripes, grmd::RmdStatus::success);
    r.gemm.assign(stripes, POTAL_OK);
    r.rows.resize(stripes);
    r.residual_c.resize(stripes);
    r.correction.resize(stripes);
    bool all = r.main_ok;
    for (size_t s = 0; s < stripes; ++s) {
        if (in.packets[s] == nullptr) {
            continue;
        }
        gp::residual::request req;
        r.built[s] = gp::residual::build_request(*in.packets[s], f.weight, req);
        if (r.built[s] != grmd::RmdStatus::success) {
            all = false;
            continue;
        }
        const gp::potal_scale_operand rs = { req.carriers, req.cstride, 32 };
        r.gemm[s] = gp::potal_tiled_matmul(req.m, req.n, req.k_execution, req.a, req.b, nullptr, req.c, req.lda,
                                           req.ldb, 0, req.ldc, 0, 0, 0, gp::POTAL_NO_ACTIVATION, 0, 0, false,
                                           req.plan.tile_I, req.plan.tile_J, req.plan.tile_K, false, false, true,
                                           false, 0, gp::POTAL_WS, rs, session);
        if (r.gemm[s] != POTAL_OK) {
            all = false;
            continue;
        }
        r.rows[s] = req.rows;
        r.residual_c[s].resize(req.m * f.N);
        for (size_t i = 0; i < req.m; ++i) {
            std::memcpy(r.residual_c[s].data() + i * f.N, req.c + i * req.ldc, f.N * sizeof(int32_t));
        }
        grmd::BlockScaledInt64Correction corr;
        r.composed[s] = gp::residual::reconstruct(*in.packets[s], req, corr);
        if (r.composed[s] != grmd::RmdStatus::success) {
            all = false;
            continue;
        }
        r.correction[s] = std::move(corr.values);
    }
    if (!all) {
        return r;
    }
    // STRIPE_PIPELINE output: Main F32, then the checked merge, per stripe.
    r.legacy.assign(f.M * f.N, 0.0f);
    r.legacy_ok = true;
    for (size_t s = 0; s < stripes; ++s) {
        const size_t row_begin = s * stripe_rows;
        const size_t rows      = std::min(stripe_rows, f.M - row_begin);
        const double act       = (double) std::ldexp(1.0f, in.theta[s]);
        for (size_t m = row_begin; m < row_begin + rows; ++m) {
            gp::reconstruct_row(r.main_c.data() + m * f.N, f.N, act, f.weight.scale, r.legacy.data() + m * f.N, 1);
        }
        if (in.packets[s] != nullptr) {
            ggml::gemmini::quants::act::exsia::StripeMetadataSnapshot snap;
            snap.theta = in.theta[s];
            grmd::BlockScaledInt64Correction corr;
            corr.values = r.correction[s];
            r.legacy_ok = r.legacy_ok && gp::residual::merge_prepared_hp1(f.weight, row_begin, row_begin + rows, snap,
                                                                         corr, r.legacy.data(), f.M, f.N, 1,
                                                                         nullptr) == grmd::RmdStatus::success;
        }
    }
    return r;
}

// ---- comparisons ----------------------------------------------------------------

template <class T> bool same_bits(const std::vector<T> & a, const std::vector<T> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

inline bool same_residual(const std::vector<golden_lane_row> & g, const std::vector<gp::residual::lane_row> & rows,
                          const std::vector<int32_t> & c, size_t N) {
    if (g.size() != rows.size() || c.size() != rows.size() * N) {
        return false;
    }
    for (size_t i = 0; i < g.size(); ++i) {
        if (g[i].lane != rows[i].lane || g[i].row != rows[i].source_row ||
            std::memcmp(g[i].c.data(), c.data() + i * N, N * sizeof(int32_t)) != 0) {
            return false;
        }
    }
    return true;
}

struct three_way {
    bool main = false, residual = false, int64 = false, legacy = false, int64_merge = false;
    size_t merge_differences = 0; // legacy F32 vs INT64 merge cells (informational)
};

// A == B == C for one case (the paired run in `out` used legacy_f32; `out64` int64).
inline three_way compare(const paired_fixture & f, const golden_result & g, const serialized_result & b,
                         const gp::paired_output & out, const gp::paired_output & out64) {
    three_way t;
    t.main      = b.main_ok && same_bits(g.main_c, b.main_c) && same_bits(g.main_c, out.main_c);
    t.residual  = out.stripes.size() == g.residual.size();
    t.int64     = t.residual;
    for (size_t s = 0; s < g.residual.size() && t.residual; ++s) {
        t.residual = t.residual && same_residual(g.residual[s], b.rows[s], b.residual_c[s], f.N) &&
                     same_residual(g.residual[s], out.stripes[s].rows, out.stripes[s].residual_c, f.N);
        t.int64 = t.int64 && same_bits(g.correction[s], b.correction[s]) &&
                  same_bits(g.correction[s], out.stripes[s].correction);
    }
    t.int64       = t.int64 && t.residual;
    t.legacy      = b.legacy_ok && same_bits(g.legacy, b.legacy) && same_bits(g.legacy, out.out);
    t.int64_merge = g.int64_merge_ok && same_bits(g.int64_merge, out64.out);
    for (size_t i = 0; i < out.out.size() && i < out64.out.size(); ++i) {
        t.merge_differences += std::memcmp(&out.out[i], &out64.out[i], sizeof(float)) != 0;
    }
    return t;
}

} // namespace potal_test
