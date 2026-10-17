#include "hp1.hpp"

#include "../../../ggml-potal-args.h"
#include "../../common/hp1_scu.hpp"

#include <algorithm>
#include <cstring>

static_assert(QK8_HP == ggml::potal::quants::weight::hp1::block_k, "HP1 scale block differs from QK8_HP");

namespace ggml::potal::quants::weight::hp1 {

namespace {

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
        case status::metadata:       return "metadata";
        case status::channel_scale:  return "channel_scale";
        case status::scale_mismatch: return "scale_mismatch";
    }
    return "unknown";
}

status unpack(const ggml_potal_args_t & args, int8_t * B, size_t ldb, uint32_t * carriers, size_t cstride,
              float * channel_scale, stats * st, wreader::WeightReaderStatus * reader) {
    if (B == nullptr || carriers == nullptr || channel_scale == nullptr || st == nullptr || ldb < args.J ||
        cstride < args.J) {
        return status::invalid;
    }
    const wreader::WeightReaderStatus rs = wreader::validate(args);
    if (reader != nullptr) {
        *reader = rs;
    }
    if (rs != wreader::WeightReaderStatus::Success) {
        return status::metadata;
    }

    const size_t nb = args.q8_hp1_blocks_per_row;
    stats        s;
    bool         have_m     = false;
    bool         have_scale = false;
    for (size_t n = 0; n < args.J; ++n) {
        const float    row_scale  = wreader::read_column_scale_validated(args, n, 0).column_scale;
        const uint32_t scale_bits = float_bits(row_scale);
        if ((scale_bits & 0x80000000u) != 0) {
            return status::channel_scale;
        }
        const bool zero_scale = scale_bits == 0;
        for (size_t b = 0; b < nb; ++b) {
            const wreader::Hp1CarrierResult c = wreader::read_hp1_carrier_validated(args, n, b);
            if (float_bits(c.column_scale) != scale_bits) {
                return status::scale_mismatch;
            }
            if (c.carrier == quants::hp1::zero_carrier) {
                s.zero_blocks++;
            } else {
                if (zero_scale) {
                    return status::channel_scale;
                }
                const int32_t m = static_cast<int32_t>(c.carrier);
                s.min_m         = have_m ? std::min(s.min_m, m) : m;
                s.max_m         = have_m ? std::max(s.max_m, m) : m;
                s.sum_m += m;
                have_m = true;
            }
            carriers[b * cstride + n] = c.carrier;
            for (size_t i = 0; i < block_k; ++i) {
                const size_t k = b * block_k + i;
                B[k * ldb + n] = static_cast<int8_t>(wreader::read_code_validated(args, n, k).value);
            }
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
    *st = s;
    return status::ok;
}

} // namespace ggml::potal::quants::weight::hp1
