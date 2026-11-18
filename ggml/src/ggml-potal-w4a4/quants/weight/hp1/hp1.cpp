#include "hp1.hpp"

#include "ggml-gemmini-args.h"
#include "quants/common/hp1_scu.hpp"
#include "quants/common/weight_route.hpp"
#include "potal-int4.h"

#include <gemmini_params.h>

#include <algorithm>
#include <cstring>

static_assert(QK4_HP == 32 && (POTAL_SCALE_BLOCK_K == 32 || POTAL_SCALE_BLOCK_K == 64),
              "HP1 native32 storage supports logical scale groups 32/64");

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
        case status::pair_mismatch:
            return "pair_mismatch (Bk64 build: native32 halves of a 64-K group differ in m/channel_scale; "
                   "requantize with MXINT4_B64, a Bk32 MXINT4 model is not accepted)";
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
    if (!plan.hp1_carriers || plan.scales.block_size != QK4_HP || args.K % block_k != 0) {
        return status::route;
    }
    // A valid plan already passed wreader::validate() inside the resolver.
    if (reader != nullptr) {
        *reader = wreader::WeightReaderStatus::Success;
    }

    const size_t nb = args.native_blocks_per_row;
    constexpr size_t halves = block_k / QK4_HP;
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
        for (size_t b = 0; b < nb; b += halves) {
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
            if constexpr (halves == 2) {
                const auto other = wreader::read_hp1_carrier_validated(args, plan, n, b + 1);
                if (!other.ok()) {
                    if (reader != nullptr) *reader = other.status;
                    return status::metadata;
                }
                if (other.carrier != c.carrier || float_bits(other.column_scale) != scale_bits) {
                    return status::pair_mismatch;
                }
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
            carriers[(b / halves) * cstride + n] = c.carrier;
        }
        s.blocks += nb / halves;
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
    // Keep all metadata/row checks above: rejected inputs retain their partial
    // carrier/scale writes and never touch B. Validated storage is immutable.
    // Resolve a native block once per column and K tile, not once per code.
    static_assert(DIM >= 2 && DIM % 2 == 0, "packed column tiles must start on byte boundaries");
    constexpr size_t tile_k = DIM < QK4_HP ? DIM : QK4_HP;
    uint16_t local_k[DIM];
    int32_t values[DIM * DIM];
    for (size_t k0 = 0; k0 < args.K; k0 += tile_k) {
        const size_t nk = std::min(tile_k, args.K - k0);
        for (size_t i = 0; i < nk; ++i) {
            local_k[i] = static_cast<uint16_t>(k0 % QK4_HP + i);
            std::memset(reinterpret_cast<uint8_t *>(B) + (k0 + i) * (ldb / 2), 0, ldb / 2);
        }
        for (size_t n0 = 0; n0 < args.J; n0 += DIM) {
            const size_t nn = std::min(static_cast<size_t>(DIM), args.J - n0);
            size_t resolutions = 0;
            const auto rs = wreader::read_code_tile_validated(
                args, plan, k0 / QK4_HP, local_k, nk, n0, nn, values, resolutions);
            if (rs != wreader::WeightReaderStatus::Success) {
                if (reader != nullptr) *reader = rs;
                return status::metadata;
            }
            for (size_t i = 0; i < nk; ++i) {
                uint8_t * dst = reinterpret_cast<uint8_t *>(B) + (k0 + i) * (ldb / 2) + n0 / 2;
                size_t n = 0;
                for (; n + 1 < nn; n += 2) {
                    dst[n / 2] = potal_int4_pack_pair(static_cast<int8_t>(values[i * DIM + n]),
                                                    static_cast<int8_t>(values[i * DIM + n + 1]));
                }
                if (n < nn) {
                    dst[n / 2] = potal_int4_pack_pair(static_cast<int8_t>(values[i * DIM + n]), 0);
                }
            }
        }
    }
    *st = s;
    return status::ok;
}

} // namespace ggml::potal::quants::weight::hp1
