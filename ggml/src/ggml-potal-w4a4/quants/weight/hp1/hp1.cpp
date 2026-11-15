#include "hp1.hpp"

#include "ggml-gemmini-args.h"
#include "quants/common/hp1_scu.hpp"
#include "quants/common/weight_route.hpp"
#include "potal-int4.h"

#include <algorithm>
#include <cstring>

static_assert(QK4_HP == ggml::potal::quants::weight::hp1::block_k, "HP1 scale block differs from QK4_HP");

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
        cstride < args.J || (ldb & 1) ||
        args.weight_format != ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1) {
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

    const size_t nb = args.native_blocks_per_row;
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

    // The validated reader decodes GGUF offset-binary half-block order.
    // Transpose logical J-by-K codes, then pack adjacent output columns.
    for (size_t k = 0; k < args.K; ++k) {
        uint8_t *dst = reinterpret_cast<uint8_t *>(B) + k * (ldb / 2);
        std::memset(dst, 0, ldb / 2);
        for (size_t n = 0; n < args.J; ++n) {
            const auto code = wreader::read_code_validated(args, plan, n, k);
            if (!code.ok()) {
                if (reader != nullptr) *reader = code.status;
                return status::metadata;
            }
            potal_int4_set(dst, n, static_cast<int8_t>(code.value));
        }
    }
    *st = s;
    return status::ok;
}

} // namespace ggml::potal::quants::weight::hp1
