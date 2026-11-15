#pragma once

// Independent goldens of the stripe pipeline numerics (host tests and the
// FireSim smoke). None of them calls the production lowering
// (residual/stripe-residual.cpp, matmul/tiling.cpp, potal_tiled_matmul_stripe):
//
//   residual_golden  the compact run-aware Residual GEMM, without padding: for
//                    every (lane, source row) with a nonzero digit (lane-major,
//                    then row), every column n: runs in packet (block) order,
//                    each run's compact K in fragments of at most DIM,
//                      p   = exact INT8 dot of the fragment
//                      q   = Sat32(HP1 shift of p by the run's carrier)
//                      acc = q for the first fragment, then Sat32(acc + q)
//   compose_golden   R[row][n] = sum C[lane, row][n] * 256^lane in __int128, INT64 range
//   main_golden      float(double(C) * double(2^theta) * double(channel_scale))
//   merge_golden     main + float(double(R) * double(channel_scale) * double(2^theta))
//   pipeline_golden  a whole node from FULL-style ExSIA output (codes, theta,
//                    packets): Main by reference_gemm_i32, then every packet
//
// Digits and K lists come from the packet (rmd::read_packet_digit, k_indices):
// the packet format is the authority, and a run's K list fixes its fragments.

#include "matmul/dense.hpp"
#include "potal_hw_params.h"
#include "quants/act/exsia/types.hpp"
#include "quants/common/hp1_scu.hpp"
#include "residual/rmd/rmd-builder.hpp"
#include "residual/rmd/rmd-types.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

namespace potal_test {

namespace grmd = ggml::gemmini::rmd;

__extension__ typedef __int128 golden_wide;

struct golden_lane_row {
    uint8_t              lane = 0;
    uint32_t             row  = 0; // stripe-local
    std::vector<int32_t> c;        // [n]
};

// code(k, n): INT8 weight code at original K k, column n; carrier(block, n).
template <class Code, class Carrier>
bool residual_golden(const grmd::StripePacket & p, size_t n, size_t dim, Code code, Carrier carrier,
                     std::vector<golden_lane_row> & out) {
    namespace hp1 = ggml::gemmini::quants::hp1;
    out.clear();
    for (uint8_t lane = 0; lane < p.lane_capacity; ++lane) {
        for (size_t row = 0; row < p.row_count; ++row) {
            std::vector<std::vector<int32_t>> digits(p.blocks.size()); // [run][compact k]
            bool                              active = false;
            for (size_t r = 0; r < p.blocks.size(); ++r) {
                const grmd::BlockDescriptor & b = p.blocks[r];
                digits[r].assign(b.compact_k_count, 0);
                for (uint8_t pos = 0; pos < b.active_lane_count; ++pos) {
                    if (b.lane_ids[pos] != lane) {
                        continue;
                    }
                    for (size_t k = 0; k < b.compact_k_count; ++k) {
                        int32_t d = 0;
                        if (grmd::read_packet_digit(p, b, pos, row, k, d) != grmd::RmdStatus::success) {
                            return false;
                        }
                        digits[r][k] = d;
                        active       = active || d != 0;
                    }
                }
            }
            if (!active) {
                continue;
            }
            golden_lane_row g;
            g.lane = lane;
            g.row  = (uint32_t) row;
            g.c.assign(n, 0);
            for (size_t col = 0; col < n; ++col) {
                int32_t acc   = 0;
                bool    first = true;
                for (size_t r = 0; r < p.blocks.size(); ++r) {
                    const grmd::BlockDescriptor & b   = p.blocks[r];
                    const uint32_t                car = carrier(b.block_id, col);
                    for (size_t f = 0; f < b.compact_k_count; f += dim) {
                        int64_t partial = 0;
                        for (size_t k = f; k < b.compact_k_count && k < f + dim; ++k) {
                            const size_t original_k =
                                (size_t) b.block_id * grmd::kBlockSize + p.k_indices[b.k_index_offset + k];
                            partial += (int64_t) digits[r][k] * code(original_k, col);
                        }
                        const int32_t q = hp1::apply_validated((int32_t) partial, car);
                        acc             = first ? q : hp1::accumulate(acc, q);
                        first           = false;
                    }
                }
                g.c[col] = acc;
            }
            out.push_back(std::move(g));
        }
    }
    return true;
}

// false: some value outside INT64.
inline bool compose_golden(const std::vector<golden_lane_row> & rows, size_t row_count, size_t n,
                           std::vector<int64_t> & out) {
    std::vector<golden_wide> acc(row_count * n, 0);
    for (const golden_lane_row & g : rows) {
        golden_wide place = 1;
        for (uint8_t l = 0; l < g.lane; ++l) {
            place *= 256;
        }
        for (size_t col = 0; col < n; ++col) {
            acc[(size_t) g.row * n + col] += (golden_wide) g.c[col] * place;
        }
    }
    out.assign(acc.size(), 0);
    for (size_t i = 0; i < acc.size(); ++i) {
        if (acc[i] > std::numeric_limits<int64_t>::max() || acc[i] < std::numeric_limits<int64_t>::min()) {
            return false;
        }
        out[i] = (int64_t) acc[i];
    }
    return true;
}

inline float main_golden(int32_t c, int16_t theta, float channel_scale) {
    return (float) ((double) c * (double) std::ldexp(1.0f, theta) * (double) channel_scale);
}

inline float merge_golden(float main, int64_t r, int16_t theta, float channel_scale) {
    const double scaled = (double) r * (double) channel_scale * (double) std::ldexp(1.0f, theta);
    return main + (float) scaled;
}

struct pipeline_golden_counts {
    size_t packets      = 0;
    size_t corrections  = 0; // nonzero INT64 correction entries
    size_t lane_rows    = 0;
};

// One node's pipeline result from FULL-style ExSIA output: args after
// quantize_activations (ExSIA metadata and packets), o its staged codes and
// prepared weight. false if anything is missing or a value leaves INT64.
inline bool pipeline_golden(const ggml_gemmini_args_t & args, const ggml::potal::hp1_operands & o,
                            std::vector<float> & want, pipeline_golden_counts * counts = nullptr) {
    namespace exsia = ggml::gemmini::quants::act::exsia;
    const auto * meta = std::get_if<exsia::Meta>(&args.act_quant.storage());
    const size_t I = args.I, J = args.J, K = args.K, rows_per = args.activation_rows_per_stripe;
    if (meta == nullptr || rows_per == 0 || meta->theta.size() != (I + rows_per - 1) / rows_per) {
        return false;
    }
    std::vector<int32_t> c(o.m_pad * o.ldc, 0);
    ggml::potal::reference_gemm_i32(o.a, o.lda, o.b, o.ldb, o.carriers, o.cstride, 32, I, J, K, c.data(), o.ldc);
    want.assign(I * J, 0.0f);
    for (size_t m = 0; m < I; ++m) {
        const int16_t theta = meta->theta[m / rows_per];
        for (size_t n = 0; n < J; ++n) {
            want[m * J + n] = main_golden(c[m * o.ldc + n], theta, o.scale[n]);
        }
    }
    pipeline_golden_counts local;
    for (const auto & packet : meta->rmd_packets) {
        if (packet == nullptr || packet->stripe_id >= meta->theta.size()) {
            return false;
        }
        std::vector<golden_lane_row> rows;
        std::vector<int64_t>         r;
        if (!residual_golden(
                *packet, J, POTAL_HW_DIM, [&](size_t k, size_t n) { return (int32_t) o.b[k * o.ldb + n]; },
                [&](size_t blk, size_t n) { return o.carriers[blk * o.cstride + n]; }, rows) ||
            !compose_golden(rows, packet->row_count, J, r)) {
            return false;
        }
        const int16_t theta = meta->theta[packet->stripe_id];
        for (size_t row = 0; row < packet->row_count; ++row) {
            for (size_t n = 0; n < J; ++n) {
                float & v = want[(packet->row_begin + row) * J + n];
                v         = merge_golden(v, r[row * J + n], theta, o.scale[n]);
                local.corrections += r[row * J + n] != 0;
            }
        }
        local.packets++;
        local.lane_rows += rows.size();
    }
    if (counts != nullptr) {
        *counts = local;
    }
    return true;
}

} // namespace potal_test
