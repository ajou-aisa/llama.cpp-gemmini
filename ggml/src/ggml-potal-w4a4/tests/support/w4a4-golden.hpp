#pragma once

// Independent golden of the EXISTING ggml-gemmini four-bit (A4W4) algorithm as frozen by
// /tmp/h3-8i-25aq-w4a4/work/impl/contract.md sections (1), (3), (4) and (5). Header-only and
// self-contained: no PoTAL, Gemmini, ggml or emulator code is called, so a wrong production
// lowering cannot bless itself. Shared by tests/potal-w4a4-golden-test.cpp and the offline
// FPGA dump replay /tmp/h3-8i-25aq-w4a4/tools/golden_replay_w4.cpp.
//
// Dense (MXINT4 weight x F32 activation, ExSIA rho 2, sigma 2, Top-N anchor, RMD radix 16):
//   weight     block_q4_hp1 (24 B: qs[16], int16 m, padding[2], float channel_scale);
//              q[j] = (qs[j] & 15) - 8, q[j + 16] = (qs[j] >> 4) - 8; carrier m, INT16_MIN -> zero
//   local      per (row, K32 block): distinct top-exponent mask, theta_pre = e_pre - rho, RNE,
//              magnitude sigma test on the unmasked population, requantize from the floats
//   stripe     Top-3 distinct block exponents (anchor 3: e3, else e2; one exponent: e1, none: 0)
//   fold       u = shift(q, e_b - e_s) (RNE right shift, Sat32 left shift); selected values split
//              into a balanced low digit a in [-8,7] and r = u - a, INT32_MAX exception -> clamp
//   Main       C = Sat32 ordered sum over K32 blocks of Sat32(dot << carrier)
//   Residual   r in balanced radix-16 digits (lanes 0..8); per (lane, row) the same SCU GEMM;
//              correction = sum C_l * 16^l in __int128, checked INT64
//   F32        main = float(double(C) * double(2^theta) * double(scale)); stripes with residual:
//              out = main + float(double(correction) * double(scale) * double(2^theta))
// Attention (G/matmul/attention.cpp, pe_bits 4, code_bits 8, radix 16, zero point 8):
//   K: theta = e - 6, code clamp [-128,119], K0 = lowdigit(code), K1 = (code - K0) / 16
//   V: theta = e - 2, code clamp [-8,7]; colsum = sum_b 2^carrier * sum_k V (checked INT64)
//   Q: ExSIA local rho 6, anchor fold; P: theta = (max > 0 ? e(max) : 0) - 7, q clamp [0,255]
//   main = lowdigit(code - zp), upper = (code - main - zp) / 16 in balanced radix-16 lanes
//   acc  = pass p (K0, K1 | V): main << 4p, upper lane l << 4(l + 1 + p), checked per add;
//          PV + 8 * colsum; F32 = float(ldexp(double(acc), stream_scale + base[j]))

#include "potal_hw_params.h" // generated geometry constants only (physical DIM)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace w4a4_golden {

__extension__ typedef __int128 i128;

constexpr int16_t  NEG                = std::numeric_limits<int16_t>::min();
constexpr uint32_t ZERO_CARRIER       = 0x80000000u;
#ifndef POTAL_SCALE_BLOCK_K
#define POTAL_SCALE_BLOCK_K 32
#endif
constexpr size_t   BK                 = POTAL_SCALE_BLOCK_K;
constexpr size_t   PHYSICAL_DIM       = POTAL_HW_DIM;
constexpr int      PE_BITS            = 4;
constexpr int      RADIX              = 16;
constexpr int      DENSE_RHO          = PE_BITS - 2;
constexpr int      QK_RHO             = 2 * PE_BITS - 2;
constexpr int      ZERO_POINT         = RADIX / 2;
constexpr size_t   MAX_LANES          = 9;
constexpr size_t   MXINT4_BLOCK_BYTES = 24;

inline int32_t sat32(i128 v) {
    return v > INT32_MAX ? INT32_MAX : v < INT32_MIN ? INT32_MIN : (int32_t) v;
}

// SCU: zero carrier or zero raw -> 0, carrier >= 32 -> signed endpoint, else Sat32(raw * 2^carrier).
inline int32_t scu(int64_t raw, uint32_t carrier) {
    if (carrier == ZERO_CARRIER || raw == 0) return 0;
    if (carrier >= 32) return raw < 0 ? INT32_MIN : INT32_MAX;
    return sat32((i128) raw * ((i128) 1 << carrier));
}

inline int32_t acc_add(int32_t acc, int32_t v) { return sat32((i128) acc + v); }

// Balanced signed low four-bit digit, by explicit remainder (not the production bit trick).
inline int32_t low_digit(int64_t v) {
    int64_t r = v % RADIX;
    if (r < 0) r += RADIX;      // 0..15
    return (int32_t) (r >= 8 ? r - RADIX : r); // -8..7
}

// Balanced radix-16 digits, lane 0 first; at most MAX_LANES (false otherwise).
inline bool radix_digits(int64_t v, std::vector<int8_t> & d) {
    d.clear();
    i128 q = v;
    while (q != 0) {
        const int32_t x = low_digit((int64_t) q);
        d.push_back((int8_t) x);
        q = (q - x) / RADIX;
        if (d.size() > MAX_LANES) return false;
    }
    return true;
}

inline int16_t uexp(float x) {
    return x == 0.0f || !std::isfinite(x) ? NEG : (int16_t) std::ilogb(std::fabs(x));
}

// RNE(x * 2^-theta), Sat32; invalid theta or nonfinite x -> 0.
inline int32_t qround(float x, int theta) {
    if (theta == NEG || !std::isfinite(x)) return 0;
    const double s  = std::ldexp((double) x, -theta);
    const double f  = std::floor(s);
    const double fr = s - f;
    const double r  = fr < 0.5 ? f : fr > 0.5 ? f + 1 : (std::fmod(f, 2.0) == 0.0 ? f : f + 1);
    if (r <= -2147483648.0) return INT32_MIN;
    if (r >= 2147483647.0) return INT32_MAX;
    return (int32_t) r;
}

// Fold: delta > 0 Sat32 multiply by 2^min(delta,31); delta < 0 RNE magnitude shift, >= 32 -> 0.
inline int32_t fold(int32_t q, int delta) {
    if (delta > 0) return sat32((i128) q * ((i128) 1 << std::min(delta, 31)));
    if (delta < 0) {
        const int sh = -delta;
        if (sh >= 32) return 0;
        const int64_t a = q < 0 ? -(int64_t) q : q, half = (int64_t) 1 << (sh - 1);
        const int64_t quo = a >> sh, rem = a & (((int64_t) 1 << sh) - 1);
        const int64_t m = quo + int64_t(rem > half || (rem == half && (quo & 1) != 0));
        return (int32_t) (q < 0 ? -m : m);
    }
    return q;
}

// ExSIA local stage of one K32 block (x: BK values, padding zero). Returns codes, the selection
// mask and the block exponent e_b (NEG if the block has no valid nonzero value).
inline void local_block(const float * x, int rho, int32_t * q, bool * mask, int16_t & e_b) {
    int16_t e[BK], e1 = NEG, e2 = NEG;
    for (size_t i = 0; i < BK; ++i) {
        e[i] = uexp(x[i]);
        if (e[i] > e1) { e2 = e1; e1 = e[i]; }
        else if (e[i] < e1 && e[i] > e2) e2 = e[i];
    }
    for (size_t i = 0; i < BK; ++i) mask[i] = e2 != NEG && e[i] != NEG && e[i] == e1;
    const int16_t epre = e2 != NEG ? e2 : e1;
    const int     tp   = epre == NEG ? NEG : epre - rho;
    i128 S = 0, SS = 0, n = 0;
    for (size_t i = 0; i < BK; ++i) {
        q[i] = e[i] == NEG ? 0 : qround(x[i], tp);
        if (e[i] == NEG || mask[i]) continue;
        const i128 m = q[i] < 0 ? -(i128) q[i] : q[i];
        S += m; SS += m * m; ++n;
    }
    const i128 var   = n * SS - S * S;
    const bool valid = n > 0 && var > 0;
    bool    hit = false;
    int16_t fin = NEG;
    for (size_t i = 0; i < BK; ++i) {
        if (e[i] == NEG || mask[i]) continue;
        const i128 m = q[i] < 0 ? -(i128) q[i] : q[i];
        const i128 c = n * m - S;
        if (valid && c > 0 && c * c > (i128) 4 * var) { mask[i] = true; hit = true; } // sigma = 2
        else fin = std::max(fin, e[i]);
    }
    e_b = epre;
    if (hit) {
        e_b = fin;
        const int tb = fin == NEG ? NEG : fin - rho;
        if (tb != tp)
            for (size_t i = 0; i < BK; ++i) q[i] = tb == NEG ? 0 : qround(x[i], tb);
    }
}

struct top3 {
    int16_t e1 = NEG, e2 = NEG, e3 = NEG;
    void add(int16_t v) {
        if (v > e1) { e3 = e2; e2 = e1; e1 = v; }
        else if (v < e1 && v > e2) { e3 = e2; e2 = v; }
        else if (v < e2 && v > e3) e3 = v;
    }
};

// ---- MXINT4 weight -------------------------------------------------------------------------------

struct mxint4_weight {
    size_t                n = 0, k = 0;
    std::vector<int8_t>   q;       // [n][k]
    std::vector<uint32_t> carrier; // [k / BK][n]
    std::vector<float>    scale;   // [n]
};

// data: n rows of k / 32 native blocks (BK / 32 equal halves per scale group), row stride
// row_bytes. false (why) on a malformed block or an unequal native pair.
inline bool decode_mxint4(const uint8_t * data, size_t row_bytes, size_t n, size_t k, mxint4_weight & w,
                          std::string & why) {
    if (k == 0 || k % BK != 0 || n == 0 || row_bytes < k / 32 * MXINT4_BLOCK_BYTES) { why = "shape"; return false; }
    const size_t blocks = k / BK;
    w.n = n; w.k = k;
    w.q.assign(n * k, 0);
    w.carrier.assign(blocks * n, ZERO_CARRIER);
    w.scale.assign(n, 0.0f);
    for (size_t j = 0; j < n; ++j)
        for (size_t b = 0; b < k / 32; ++b) {
            const uint8_t * blk = data + j * row_bytes + b * MXINT4_BLOCK_BYTES;
            int16_t m; float s;
            std::memcpy(&m, blk + 16, 2);
            std::memcpy(&s, blk + 20, 4);
            if (blk[18] != 0 || blk[19] != 0) { why = "padding"; return false; }
            if (!std::isfinite(s) || s < 0.0f) { why = "channel_scale"; return false; }
            if (b == 0) w.scale[j] = s;
            else if (std::memcmp(&s, &w.scale[j], 4) != 0) { why = "channel_scale_differs_within_row"; return false; }
            if (m != NEG && m < 0) { why = "negative_m"; return false; }
            const uint32_t carrier = m == NEG ? ZERO_CARRIER : (uint32_t) m;
            const size_t logical = b / (BK / 32);
            if (b % (BK / 32) && w.carrier[logical * n + j] != carrier) { why = "native_pair_mismatch"; return false; }
            w.carrier[logical * n + j] = carrier;
            for (size_t i = 0; i < 16; ++i) {
                w.q[j * k + b * 32 + i]      = (int8_t) ((blk[i] & 15) - 8);
                w.q[j * k + b * 32 + i + 16] = (int8_t) ((blk[i] >> 4) - 8);
            }
        }
    return true;
}

// ---- dense ---------------------------------------------------------------------------------------

struct dense_result {
    size_t m = 0, n = 0, k = 0, rows_per_stripe = 0;
    std::vector<int16_t> theta;     // per stripe
    std::vector<int8_t>  codes;     // [m][k] Main digits
    std::vector<int64_t> residual;  // [m][k] r = u - code
    std::vector<int32_t> main_c;    // [m][n]
    std::map<std::pair<uint8_t, uint32_t>, std::vector<int32_t>> lane_c; // (lane, global row) -> [n]
    std::vector<int64_t> correction; // [m][n]
    std::vector<uint8_t> stripe_residual;
    std::vector<float>   out;        // [m][n]
    size_t outliers = 0, residual_nnz = 0, max_lanes = 0, int32_max_exceptions = 0;
};

// One SCU GEMM row: a[k] digits against w (all K blocks, original order).
inline void scu_row(const int8_t * a, const mxint4_weight & w, int32_t * c) {
    for (size_t j = 0; j < w.n; ++j) {
        int32_t acc = 0;
        for (size_t begin = 0; begin < w.k; begin += PHYSICAL_DIM) {
            i128 raw = 0;
            for (size_t i = begin; i < begin + PHYSICAL_DIM; ++i) raw += (i128) a[i] * w.q[j * w.k + i];
            acc = acc_add(acc, scu((int64_t) raw, w.carrier[(begin / BK) * w.n + j]));
        }
        c[j] = acc;
    }
}

// x: [m][k] row-major F32. anchor: 2 or 3.
inline bool dense(const float * x, size_t m, size_t k, const mxint4_weight & w, size_t rows_per_stripe, int anchor,
                  dense_result & r, std::string & why) {
    if (k != w.k || m == 0 || rows_per_stripe == 0) { why = "shape"; return false; }
    for (size_t i = 0; i < m * k; ++i)
        if (!std::isfinite(x[i])) { why = "nonfinite_activation"; return false; }
    const size_t BPR = k / BK, S = (m + rows_per_stripe - 1) / rows_per_stripe, n = w.n;
    r = dense_result{};
    r.m = m; r.n = n; r.k = k; r.rows_per_stripe = rows_per_stripe;
    std::vector<int32_t> qw(m * k);
    std::vector<uint8_t> sel(m * k, 0);
    std::vector<int16_t> eb(m * BPR);
    for (size_t row = 0; row < m; ++row)
        for (size_t b = 0; b < BPR; ++b) {
            bool mask[BK];
            local_block(x + row * k + b * BK, DENSE_RHO, qw.data() + row * k + b * BK, mask, eb[row * BPR + b]);
            for (size_t i = 0; i < BK; ++i) sel[row * k + b * BK + i] = mask[i];
        }
    r.codes.assign(m * k, 0);
    r.residual.assign(m * k, 0);
    r.theta.assign(S, NEG);
    r.stripe_residual.assign(S, 0);
    for (size_t s = 0; s < S; ++s) {
        const size_t r0 = s * rows_per_stripe, r1 = std::min(m, r0 + rows_per_stripe);
        top3 t;
        for (size_t row = r0; row < r1; ++row)
            for (size_t b = 0; b < BPR; ++b) t.add(eb[row * BPR + b]);
        int16_t es; bool promote;
        if (t.e1 == NEG) { es = 0; promote = false; }
        else if (t.e2 == NEG) { es = t.e1; promote = false; }
        else { es = anchor == 3 && t.e3 != NEG ? t.e3 : t.e2; promote = true; }
        r.theta[s] = (int16_t) (es - DENSE_RHO);
        for (size_t row = r0; row < r1; ++row)
            for (size_t b = 0; b < BPR; ++b) {
                const int16_t v = eb[row * BPR + b];
                for (size_t i = 0; i < BK; ++i) {
                    const size_t idx = row * k + b * BK + i;
                    if (promote && v != NEG && v > es) sel[idx] = 1;
                    const int32_t u   = fold(qw[idx], v == NEG ? 0 : v - es);
                    const int32_t sat = std::clamp(u, -8, 7);
                    if (!sel[idx]) { r.codes[idx] = (int8_t) sat; continue; }
                    ++r.outliers;
                    const int32_t a  = low_digit(u);
                    const int64_t rr = (int64_t) u - a;
                    if (rr > INT32_MAX) { r.codes[idx] = (int8_t) sat; r.residual[idx] = (int64_t) u - sat; ++r.int32_max_exceptions; }
                    else { r.codes[idx] = (int8_t) a; r.residual[idx] = rr; }
                    if (r.residual[idx] != 0) { ++r.residual_nnz; r.stripe_residual[s] = 1; }
                }
            }
    }
    r.main_c.assign(m * n, 0);
    for (size_t row = 0; row < m; ++row) scu_row(r.codes.data() + row * k, w, r.main_c.data() + row * n);
    std::vector<i128> corr(m * n, 0);
    std::vector<int8_t> digits;
    for (size_t row = 0; row < m; ++row) {
        std::vector<std::vector<int8_t>> lanes;
        for (size_t c = 0; c < k; ++c) {
            if (r.residual[row * k + c] == 0) continue;
            if (!radix_digits(r.residual[row * k + c], digits)) { why = "residual_lanes"; return false; }
            if (lanes.size() < digits.size()) lanes.resize(digits.size(), std::vector<int8_t>(k, 0));
            for (size_t l = 0; l < digits.size(); ++l) lanes[l][c] = digits[l];
        }
        r.max_lanes = std::max(r.max_lanes, lanes.size());
        for (size_t l = 0; l < lanes.size(); ++l) {
            if (std::all_of(lanes[l].begin(), lanes[l].end(), [](int8_t d) { return d == 0; })) continue;
            std::vector<int32_t> c(n);
            // Compact support is the union of all residual lanes/source rows in this stripe.
            for (size_t j = 0; j < n; ++j) {
                int32_t acc = 0;
                const size_t r0 = row / rows_per_stripe * rows_per_stripe;
                for (size_t b = 0; b < BPR; ++b) {
                    std::vector<size_t> support;
                    for (size_t local = 0; local < BK; ++local) {
                        bool live = false;
                        for (size_t rr = r0; rr < std::min(m, r0 + rows_per_stripe); ++rr)
                            live = live || r.residual[rr * k + b * BK + local] != 0;
                        if (live) support.push_back(b * BK + local);
                    }
                    for (size_t f = 0; f < support.size(); f += PHYSICAL_DIM) {
                        i128 raw = 0;
                        for (size_t i = f; i < std::min(support.size(), f + PHYSICAL_DIM); ++i)
                            raw += (i128) lanes[l][support[i]] * w.q[j * k + support[i]];
                        acc = acc_add(acc, scu((int64_t) raw, w.carrier[b * n + j]));
                    }
                }
                c[j] = acc;
            }
            i128 place = 1;
            for (size_t p = 0; p < l; ++p) place *= RADIX;
            for (size_t j = 0; j < n; ++j) corr[row * n + j] += (i128) c[j] * place;
            r.lane_c[{ (uint8_t) l, (uint32_t) row }] = std::move(c);
        }
    }
    r.correction.assign(m * n, 0);
    for (size_t i = 0; i < m * n; ++i) {
        if (corr[i] > INT64_MAX || corr[i] < INT64_MIN) { why = "correction_out_of_int64"; return false; }
        r.correction[i] = (int64_t) corr[i];
    }
    r.out.assign(m * n, 0.0f);
    for (size_t row = 0; row < m; ++row) {
        const size_t s     = row / rows_per_stripe;
        const double act   = (double) std::ldexp(1.0f, r.theta[s]);
        for (size_t j = 0; j < n; ++j) {
            float v = (float) ((double) r.main_c[row * n + j] * act * (double) w.scale[j]);
            if (r.stripe_residual[s]) {
                const double domain = (double) r.correction[row * n + j] * (double) w.scale[j];
                const float  delta  = (float) (domain * act);
                if (!std::isfinite(domain) || !std::isfinite(delta)) { why = "nonfinite_merge"; return false; }
                v = v + delta;
            }
            if (!std::isfinite(v)) { why = "nonfinite_output"; return false; }
            r.out[row * n + j] = v;
        }
    }
    return true;
}

// ---- attention -----------------------------------------------------------------------------------

struct att_stationary {
    bool                  qk = true;
    size_t                k = 0, n = 0, kp = 0;
    std::vector<int16_t>  base;     // [n]
    std::vector<uint32_t> carriers; // [kp / BK][n]
    std::vector<int8_t>   c0, c1;   // [kp][n]: K0/K1, or V
    std::vector<int64_t>  colsum;   // PV: [n]
};

// get(kk, j): stationary value at K index kk < k, column j < n.
template <class Get>
bool att_prepare_stationary(Get get, size_t k, size_t n, bool qk, att_stationary & w, std::string & why) {
    w = att_stationary{};
    w.qk = qk; w.k = k; w.n = n; w.kp = (k + BK - 1) / BK * BK;
    const size_t blocks = w.kp / BK;
    w.base.assign(n, 0);
    w.carriers.assign(blocks * n, ZERO_CARRIER);
    w.c0.assign(w.kp * n, 0);
    if (qk) w.c1.assign(w.kp * n, 0);
    std::vector<int16_t> theta(blocks);
    for (size_t j = 0; j < n; ++j) {
        int16_t base = std::numeric_limits<int16_t>::max();
        for (size_t b = 0; b < blocks; ++b) {
            const size_t valid = std::min(BK, k - b * BK);
            float        v[BK];
            int16_t      e = NEG;
            for (size_t i = 0; i < valid; ++i) {
                v[i] = get(b * BK + i, j);
                if (!std::isfinite(v[i])) { why = "nonfinite_stationary"; return false; }
                e = std::max(e, uexp(v[i]));
            }
            theta[b] = e == NEG ? NEG : (int16_t) (e - (qk ? QK_RHO : DENSE_RHO));
            if (e == NEG) continue;
            base = std::min(base, theta[b]);
            for (size_t i = 0; i < valid; ++i) {
                const int32_t code = std::clamp(qround(v[i], theta[b]), qk ? -128 : -8, qk ? 119 : 7);
                const size_t  idx  = (b * BK + i) * n + j;
                if (qk) {
                    const int32_t lo = low_digit(code);
                    w.c0[idx] = (int8_t) lo;
                    w.c1[idx] = (int8_t) ((code - lo) / RADIX);
                } else {
                    w.c0[idx] = (int8_t) code;
                }
            }
        }
        w.base[j] = base == std::numeric_limits<int16_t>::max() ? 0 : base;
        for (size_t b = 0; b < blocks; ++b)
            w.carriers[b * n + j] = theta[b] == NEG ? ZERO_CARRIER : (uint32_t) (theta[b] - w.base[j]);
    }
    if (!qk) {
        w.colsum.assign(n, 0);
        for (size_t b = 0; b < blocks; ++b)
            for (size_t j = 0; j < n; ++j) {
                const uint32_t car = w.carriers[b * n + j];
                if (car == ZERO_CARRIER) continue;
                int64_t sum = 0;
                for (size_t i = 0; i < BK; ++i) sum += w.c0[(b * BK + i) * n + j];
                if (sum == 0) continue;
                if (car > 63) { why = "colsum_shift"; return false; }
                const i128 t = (i128) w.colsum[j] + (i128) sum * ((i128) 1 << car);
                if (t > INT64_MAX || t < INT64_MIN) { why = "colsum_overflow"; return false; }
                w.colsum[j] = (int64_t) t;
            }
    }
    return true;
}

struct att_stream {
    bool                             qk = true;
    size_t                           rows = 0, k = 0, kp = 0;
    int                              scale = 0;
    std::vector<int32_t>             codes; // [rows][kp]: folded Q or clamped unsigned P
    std::vector<int8_t>              main;  // [rows][kp]
    std::vector<std::vector<int8_t>> upper; // [lane][rows][kp]; lane l = original digit l + 1
};

// get(i, kk): stream value at row i < rows, K index kk < k. anchor: 2 or 3 (Q only).
template <class Get>
bool att_prepare_stream(Get get, size_t rows, size_t k, bool qk, int anchor, att_stream & s, std::string & why,
                        bool q4_residual = false) {
    s = att_stream{};
    s.qk = qk; s.rows = rows; s.k = k; s.kp = (k + BK - 1) / BK * BK;
    const size_t kp = s.kp, blocks = kp / BK;
    std::vector<float> v(rows * kp, 0.0f);
    for (size_t i = 0; i < rows; ++i)
        for (size_t c = 0; c < k; ++c) {
            v[i * kp + c] = get(i, c);
            if (!std::isfinite(v[i * kp + c])) { why = "nonfinite_stream"; return false; }
            if (!qk && (v[i * kp + c] < 0.0f || v[i * kp + c] > 1.0f)) { why = "probability_outside_0_1"; return false; }
        }
    s.codes.assign(rows * kp, 0);
    if (qk) {
        std::vector<int16_t> eb(rows * blocks);
        std::vector<uint8_t> selected(rows * kp, 0);
        top3 t;
        for (size_t i = 0; i < rows; ++i)
            for (size_t b = 0; b < blocks; ++b) {
                bool mask[BK];
                local_block(v.data() + i * kp + b * BK, q4_residual ? DENSE_RHO : QK_RHO,
                            s.codes.data() + i * kp + b * BK, mask, eb[i * blocks + b]);
                for (size_t c = 0; c < BK; ++c) selected[i * kp + b * BK + c] = mask[c];
                t.add(eb[i * blocks + b]);
            }
        const int16_t a = anchor == 3 && t.e3 != NEG ? t.e3 : t.e2 != NEG ? t.e2 : t.e1 != NEG ? t.e1 : 0;
        s.scale = a - (q4_residual ? DENSE_RHO : QK_RHO);
        for (size_t i = 0; i < rows; ++i)
            for (size_t c = 0; c < k; ++c) {
                const int16_t e = eb[i * blocks + c / BK];
                s.codes[i * kp + c] = fold(s.codes[i * kp + c], e == NEG ? 0 : e - a);
                if (q4_residual && e <= a && !selected[i * kp + c])
                    s.codes[i * kp + c] = std::clamp(s.codes[i * kp + c], -8, 7);
            }
    } else {
        const float mx = *std::max_element(v.begin(), v.end());
        s.scale = (mx > 0 ? uexp(mx) : 0) - (2 * PE_BITS - 1);
        for (size_t i = 0; i < rows * kp; ++i) s.codes[i] = std::clamp(qround(v[i], s.scale), 0, 255);
    }
    s.main.assign(rows * kp, 0);
    std::vector<int8_t> d;
    for (size_t i = 0; i < rows; ++i)
        for (size_t c = 0; c < k; ++c) {
            const size_t  idx  = i * kp + c;
            const int64_t code = s.codes[idx] - (qk ? 0 : ZERO_POINT);
            const int32_t lo   = low_digit(code);
            s.main[idx]        = (int8_t) lo;
            if (!radix_digits((code - lo) / RADIX, d)) { why = "upper_lanes"; return false; }
            if (s.upper.size() < d.size()) s.upper.resize(d.size(), std::vector<int8_t>(rows * kp, 0));
            for (size_t l = 0; l < d.size(); ++l) s.upper[l][idx] = d[l];
        }
    return true;
}

// Main digit-pair SCU GEMM: a [rows][kp] x stationary digits b [kp][n]. Every LOOP_WS pass is
// PHYSICAL_DIM K wide, SCU-shifted by its BK block carrier and Sat32-added in ascending K.
inline std::vector<int32_t> att_gemm_main(const std::vector<int8_t> & a, const std::vector<int8_t> & b,
                                          const att_stationary & w, size_t rows) {
    std::vector<int32_t> c(rows * w.n, 0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < w.n; ++j) {
            int32_t acc = 0;
            for (size_t begin = 0; begin < w.kp; begin += PHYSICAL_DIM) {
                i128 raw = 0;
                for (size_t kk = begin; kk < begin + PHYSICAL_DIM; ++kk)
                    raw += (i128) a[i * w.kp + kk] * b[kk * w.n + j];
                acc = acc_add(acc, scu((int64_t) raw, w.carriers[(begin / BK) * w.n + j]));
            }
            c[i * w.n + j] = acc;
        }
    return c;
}

// Upper-lane SCU GEMMs, one result [rows][n] per lane. Per PHYSICAL_DIM-row execution slice and BK
// block, the compact K support is the ascending union over all lanes and rows of the slice; it is
// cut into fragments of <= PHYSICAL_DIM compact K, each one SCU (block carrier) + Sat32 add.
inline std::vector<std::vector<int32_t>> att_gemm_upper(const std::vector<std::vector<int8_t>> & lanes,
                                                        const std::vector<int8_t> & b, const att_stationary & w,
                                                        size_t rows) {
    std::vector<std::vector<int32_t>> c(lanes.size(), std::vector<int32_t>(rows * w.n, 0));
    for (size_t r0 = 0; r0 < rows; r0 += PHYSICAL_DIM) {
        const size_t r1 = std::min(rows, r0 + PHYSICAL_DIM);
        std::vector<std::vector<size_t>> fragments;
        for (size_t blk = 0; blk < w.kp / BK; ++blk) {
            std::vector<size_t> support;
            for (size_t t = 0; t < BK; ++t) {
                bool live = false;
                for (const auto & lane : lanes)
                    for (size_t i = r0; i < r1; ++i) live = live || lane[i * w.kp + blk * BK + t] != 0;
                if (live) support.push_back(blk * BK + t);
            }
            for (size_t f = 0; f < support.size(); f += PHYSICAL_DIM)
                fragments.emplace_back(support.begin() + f, support.begin() + std::min(support.size(), f + PHYSICAL_DIM));
        }
        for (size_t l = 0; l < lanes.size(); ++l)
            for (size_t i = r0; i < r1; ++i)
                for (size_t j = 0; j < w.n; ++j) {
                    int32_t acc = 0;
                    for (const auto & fragment : fragments) {
                        i128 raw = 0;
                        for (size_t kk : fragment) raw += (i128) lanes[l][i * w.kp + kk] * b[kk * w.n + j];
                        acc = acc_add(acc, scu((int64_t) raw, w.carriers[(fragment.front() / BK) * w.n + j]));
                    }
                    c[l][i * w.n + j] = acc;
                }
    }
    return c;
}

inline bool scaled_add(int64_t & acc, int64_t part, int shift) {
    if (!part) return true;
    if (shift < 0 || shift > 63) return false;
    const i128 sum = (i128) acc + (i128) part * ((i128) 1 << shift);
    if (sum > INT64_MAX || sum < INT64_MIN) return false;
    acc = (int64_t) sum;
    return true;
}

// ints: [rows][n] reconstructed INT64 (after the PV zero-point bias); out: F32.
inline bool att_multiply(const att_stream & s, const att_stationary & w, std::vector<int64_t> & ints,
                         std::vector<float> & out, std::string & why) {
    if (s.kp != w.kp || s.qk != w.qk) { why = "shape"; return false; }
    ints.assign(s.rows * w.n, 0);
    const size_t passes = w.qk ? 2 : 1;
    for (size_t p = 0; p < passes; ++p) {
        const std::vector<int8_t> & st = p == 0 ? w.c0 : w.c1;
        const auto main = att_gemm_main(s.main, st, w, s.rows);
        for (size_t i = 0; i < main.size(); ++i)
            if (!scaled_add(ints[i], main[i], PE_BITS * (int) p)) { why = "int64_overflow"; return false; }
        const auto upper = att_gemm_upper(s.upper, st, w, s.rows);
        for (size_t l = 0; l < upper.size(); ++l) {
            const auto & up = upper[l];
            for (size_t i = 0; i < up.size(); ++i)
                if (!scaled_add(ints[i], up[i], PE_BITS * ((int) l + 1 + (int) p))) { why = "int64_overflow"; return false; }
        }
    }
    out.assign(s.rows * w.n, 0.0f);
    for (size_t i = 0; i < s.rows; ++i)
        for (size_t j = 0; j < w.n; ++j) {
            if (!w.qk && !scaled_add(ints[i * w.n + j], w.colsum[j], PE_BITS - 1)) { why = "bias_overflow"; return false; }
            const float f = (float) std::ldexp((double) ints[i * w.n + j], s.scale + w.base[j]);
            if (!std::isfinite(f)) { why = "nonfinite_output"; return false; }
            out[i * w.n + j] = f;
        }
    return true;
}

// Exact wide dot (no SCU or ACC saturation), same digits, shifts and PV bias as att_multiply:
// exact[i][j] = sum_p sum_digits sum_k a * b * 2^carrier(k / BK) * 16^shift. Diagnostic only.
inline void att_exact(const att_stream & s, const att_stationary & w, std::vector<i128> & exact) {
    exact.assign(s.rows * w.n, 0);
    const size_t passes = w.qk ? 2 : 1;
    for (size_t p = 0; p < passes; ++p) {
        const std::vector<int8_t> & st = p == 0 ? w.c0 : w.c1;
        for (size_t d = 0; d <= s.upper.size(); ++d) {
            const std::vector<int8_t> & a = d == 0 ? s.main : s.upper[d - 1];
            for (size_t i = 0; i < s.rows; ++i)
                for (size_t j = 0; j < w.n; ++j) {
                    i128 sum = 0;
                    for (size_t kk = 0; kk < w.kp; ++kk) {
                        const uint32_t car = w.carriers[(kk / BK) * w.n + j];
                        if (car == ZERO_CARRIER) continue;
                        sum += (i128) a[i * w.kp + kk] * st[kk * w.n + j] * ((i128) 1 << car);
                    }
                    exact[i * w.n + j] += sum * ((i128) 1 << (PE_BITS * (d + p)));
                }
        }
    }
    if (!w.qk)
        for (size_t i = 0; i < s.rows; ++i)
            for (size_t j = 0; j < w.n; ++j) exact[i * w.n + j] += (i128) w.colsum[j] * ZERO_POINT;
}

} // namespace w4a4_golden
