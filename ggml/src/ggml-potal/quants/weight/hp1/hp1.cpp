#include "hp1.hpp"

#include "ggml-gemmini-args.h"
#include "quants/common/hp1_scu.hpp"
#include "quants/common/weight_route.hpp"

#include <algorithm>
#include <cstring>

static_assert(QK8_HP == ggml::potal::quants::weight::hp1::block_k, "HP1 scale block differs from QK8_HP");

namespace ggml::potal::quants::weight::hp1 {

namespace {

namespace wroute = ggml::gemmini::quants::wroute;

uint32_t float_bits(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

} // namespace

const char * status_name(status s) {
    switch (s) {
        case status::ok:             return "ok";
        case status::invalid:        return "invalid";
        case status::route:          return "route";
        case status::metadata:       return "metadata";
        case status::channel_scale:  return "channel_scale";
        case status::scale_mismatch: return "scale_mismatch";
    }
    return "unknown";
}

status unpack(const ggml_gemmini_args_t & args, int8_t * B, size_t ldb, uint32_t * carriers, size_t cstride,
              float * channel_scale, stats * st, wreader::WeightReaderStatus * reader) {
    if (B == nullptr || carriers == nullptr || channel_scale == nullptr || st == nullptr || ldb < args.J ||
        cstride < args.J) {
        return status::invalid;
    }
    // The carrier domain the NPU consumes (as rmd-compose resolves it for HP1).
    const wroute::WeightRoutePlan plan =
        wroute::resolve_weight_route_plan(args, wroute::WeightScaleInfoMode::ResidualHp1Scu);
    if (!plan.valid) {
        // The resolver runs wreader::validate() on HP1 plans and keeps only the
        // route status. Repeat it on the rejected plan (reads are bounds-checked)
        // so a block metadata failure reports as metadata with its reader status.
        if (plan.hp1_carriers && plan.status == wroute::WeightRouteStatus::InvalidMetadata) {
            const wreader::WeightReaderStatus rs = wreader::validate(args, plan);
            if (rs != wreader::WeightReaderStatus::Success) {
                if (reader != nullptr) {
                    *reader = rs;
                }
                return status::metadata;
            }
        }
        return status::route;
    }
    if (!plan.hp1_carriers || plan.scales.block_size != block_k) {
        return status::route;
    }
    // A valid plan already passed wreader::validate() inside the resolver.
    if (reader != nullptr) {
        *reader = wreader::WeightReaderStatus::Success;
    }

    const size_t nb = args.q8_hp1_blocks_per_row;
    stats        s;
    bool         have_m     = false;
    bool         have_scale = false;
    for (size_t n = 0; n < args.J; ++n) {
        const wreader::ColumnScaleResult cs = wreader::read_column_scale_validated(args, plan, n, 0);
        if (!cs.ok()) {
            if (reader != nullptr) {
                *reader = cs.status;
            }
            return status::metadata;
        }
        const float    row_scale  = cs.column_scale;
        const uint32_t scale_bits = float_bits(row_scale);
        if ((scale_bits & 0x80000000u) != 0) {
            return status::channel_scale;
        }
        const bool zero_scale = scale_bits == 0;
        for (size_t b = 0; b < nb; ++b) {
            const wreader::Hp1CarrierResult c = wreader::read_hp1_carrier_validated(args, plan, n, b);
            if (!c.ok()) {
                if (reader != nullptr) {
                    *reader = c.status;
                }
                return status::metadata;
            }
            if (float_bits(c.column_scale) != scale_bits) {
                return status::scale_mismatch;
            }
            if (c.carrier == ggml::gemmini::quants::hp1::zero_carrier) {
                s.zero_blocks++;
            } else {
                if (zero_scale) {
                    return status::channel_scale;
                }
                const int32_t m = static_cast<int32_t>(c.carrier);
                s.min_m         = have_m ? std::min(s.min_m, m) : m;
                s.max_m         = have_m ? std::max(s.max_m, m) : m;
                s.sum_m += m;
                s.shift_blocks += m > 0 ? 1 : 0;
                have_m = true;
            }
            carriers[b * cstride + n] = c.carrier;
        }
        s.blocks += nb;
        channel_scale[n] = row_scale;
        if (zero_scale) {
            s.zero_rows++;
        } else {
            s.min_channel_scale = have_scale ? std::min(s.min_channel_scale, row_scale) : row_scale;
            s.max_channel_scale = have_scale ? std::max(s.max_channel_scale, row_scale) : row_scale;
            have_scale          = true;
        }
    }

    // qs -> B[k][n]. Every block passed the reader's checks above, so the qs
    // are read per block (as wreader::read_code_tile_validated does), not per
    // element through read_code_validated. Column tiles keep the transpose in
    // cache: each B row segment is written contiguously.
    constexpr size_t tile_n = 64;
    for (size_t n0 = 0; n0 < args.J; n0 += tile_n) {
        const size_t n1 = std::min(args.J, n0 + tile_n);
        for (size_t b = 0; b < nb; ++b) {
            const block_q8_hp1 * row_blocks[tile_n];
            for (size_t n = n0; n < n1; ++n) {
                row_blocks[n - n0] = args.q8_hp1_block(n, b);
                if (row_blocks[n - n0] == nullptr) {
                    return status::metadata; // unreachable after validation
                }
            }
            for (size_t i = 0; i < block_k; ++i) {
                int8_t * dst = B + (b * block_k + i) * ldb;
                for (size_t n = n0; n < n1; ++n) {
                    dst[n] = row_blocks[n - n0]->qs[i];
                }
            }
        }
    }
    *st = s;
    return status::ok;
}

} // namespace ggml::potal::quants::weight::hp1
