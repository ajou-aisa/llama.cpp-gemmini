#include "stripe-residual.hpp"

#include "potal_hw_params.h"

#include "residual/rmd/rmd-builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace ggml::potal::residual {

namespace {

namespace exsia = ggml::gemmini::quants::act::exsia;

constexpr size_t HW_DIM = POTAL_HW_DIM; // not DIM: <gemmini_params.h> defines that macro
constexpr size_t RUN_K  = quants::weight::hp1::block_k; // execution K of one run: one HP1 scale block

static_assert(rmd::kBlockSize == RUN_K, "a residual run is one original weight block = one HP1 scale block");
static_assert(RUN_K % HW_DIM == 0, "a run is a whole number of DIM-wide passes");

__extension__ typedef __int128 wide_t; // checked radix composition (no INT64 narrowing in between)

// Position of `lane` among the block's active lanes, or -1 when the block has no digit of it.
int lane_position(const rmd::BlockDescriptor & block, uint8_t lane) {
    for (uint8_t p = 0; p < block.active_lane_count; ++p) {
        if (block.lane_ids[p] == lane) {
            return p;
        }
    }
    return -1;
}

// Finite tests on the bits, as rmd-compose.cpp's finite_*_representation: an
// -ffast-math build may assume std::isfinite true.
bool finite_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const volatile uint32_t observed = bits;
    return (observed & 0x7f800000u) != 0x7f800000u;
}

bool finite_bits(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const volatile uint64_t observed = bits;
    return (observed & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

} // namespace

request::~request() {
    for (void * p : blocks) {
        std::free(p);
    }
}

void request::abandon() {
    for (void *& p : blocks) {
        p = nullptr;
    }
}

void request::reset() {
    for (void *& p : blocks) {
        std::free(p);
        p = nullptr;
    }
    runs.clear();
    rows.clear();
    m = n = k_compact = k_execution = 0;
    a        = nullptr;
    b        = nullptr;
    carriers = nullptr;
    c        = nullptr;
    lda = ldb = cstride = ldc = 0;
    plan                      = {};
}

bool request::allocate(size_t m_rows, size_t n_cols, size_t k_exec, size_t runs_count, size_t weight_ldb,
                       size_t weight_cstride) {
    const size_t m_pad = align_up(m_rows, HW_DIM);
    lda                = align_up(k_exec, POTAL_HW_MAX_BYTES);
    ldb                = weight_ldb;
    cstride            = weight_cstride;
    ldc                = align_up(n_cols, HW_DIM);
    a        = static_cast<int8_t *>(blocks[0] = dma_alloc(m_pad * lda));
    b        = static_cast<int8_t *>(blocks[1] = dma_alloc(k_exec * ldb));
    carriers = static_cast<uint32_t *>(blocks[2] = dma_alloc(runs_count * cstride * sizeof(uint32_t)));
    c        = static_cast<int32_t *>(blocks[3] = dma_alloc(m_pad * ldc * sizeof(int32_t)));
    return a && b && carriers && c;
}

rmd::RmdStatus decode_packet(const rmd::StripePacket & packet, const hp1_weight & w, decoded_packet & out) try {
    out = decoded_packet{};
    if (rmd::validate_packet(packet) != rmd::RmdStatus::success) {
        return rmd::RmdStatus::invalid_packet;
    }
    if (packet.digit_bits != digit_bits) {
        return rmd::RmdStatus::unsupported_route; // PoTal is A8W8
    }
    if (packet.logical_j != w.n || packet.logical_k != w.k || w.b == nullptr || w.carriers == nullptr) {
        return rmd::RmdStatus::invalid_arguments;
    }
    const rmd::BalancedRadixContract radix = rmd::balanced_radix_contract(packet.digit_bits);

    // Runs: the packet's blocks (validated: block id ascending), their compact K.
    const size_t        runs = packet.blocks.size();
    std::vector<size_t> offset; // compact offset of each run
    out.runs.reserve(runs);
    for (const rmd::BlockDescriptor & block : packet.blocks) {
        if ((size_t) block.block_id > (w.k - RUN_K) / RUN_K ||
            block.k_index_offset + (size_t) block.compact_k_count > packet.k_indices.size()) {
            return rmd::RmdStatus::invalid_packet;
        }
        run r;
        r.block_id = block.block_id;
        for (size_t i = 0; i < block.compact_k_count; ++i) {
            const uint16_t local_k = packet.k_indices[block.k_index_offset + i];
            if (local_k >= RUN_K || (!r.local_k.empty() && local_k <= r.local_k.back())) {
                return rmd::RmdStatus::invalid_packet;
            }
            r.local_k.push_back(local_k);
        }
        offset.push_back(out.k_compact);
        out.k_compact += r.local_k.size();
        out.runs.push_back(std::move(r));
    }

    // Rows: lane-major, then source row; one compact digit row each.
    std::vector<int8_t> row(out.k_compact);
    for (uint8_t lane = 0; lane < packet.lane_capacity; ++lane) {
        for (size_t src = 0; src < packet.row_count; ++src) {
            std::fill(row.begin(), row.end(), int8_t{ 0 });
            bool active = false;
            for (size_t r = 0; r < runs; ++r) {
                const rmd::BlockDescriptor & block = packet.blocks[r];
                const int                    pos   = lane_position(block, lane);
                if (pos < 0) {
                    continue; // no digit of this lane in this run: zeros
                }
                for (size_t k = 0; k < block.compact_k_count; ++k) {
                    int32_t              digit  = 0;
                    const rmd::RmdStatus status = rmd::read_packet_digit(packet, block, (uint8_t) pos, src, k, digit);
                    if (status != rmd::RmdStatus::success) {
                        return status;
                    }
                    if (digit < radix.digit_min || digit > radix.digit_max) {
                        return rmd::RmdStatus::overflow;
                    }
                    row[offset[r] + k] = (int8_t) digit;
                    active             = active || digit != 0;
                }
            }
            if (active) {
                out.rows.push_back({ lane, (uint32_t) src });
                out.digits.insert(out.digits.end(), row.begin(), row.end());
            }
        }
    }
    if (out.rows.empty()) {
        return rmd::RmdStatus::invalid_packet; // a sealed packet carries at least one nonzero digit
    }
    return rmd::RmdStatus::success;
} catch (const std::bad_alloc &) {
    return rmd::RmdStatus::allocation_failure;
} catch (const std::length_error &) {
    return rmd::RmdStatus::overflow;
}

rmd::RmdStatus build_request(const rmd::StripePacket & packet, const hp1_weight & w, request & out) try {
    out.reset();
    decoded_packet       decoded;
    const rmd::RmdStatus status = decode_packet(packet, w, decoded);
    if (status != rmd::RmdStatus::success) {
        return status;
    }
    const size_t runs = decoded.runs.size();
    out.runs          = std::move(decoded.runs);
    out.rows          = std::move(decoded.rows);
    out.k_compact     = decoded.k_compact;
    out.k_execution   = RUN_K * runs;
    out.n             = w.n;
    out.m             = out.rows.size();

    if (!out.allocate(out.m, out.n, out.k_execution, runs, w.ldb, w.cstride)) {
        return rmd::RmdStatus::allocation_failure;
    }
    // K32 padded lowering: run r's compact digits first, then zero K.
    for (size_t i = 0; i < out.m; ++i) {
        const int8_t * src = decoded.digits.data() + i * out.k_compact;
        for (size_t r = 0; r < runs; ++r) {
            std::memcpy(out.a + i * out.lda + r * RUN_K, src, out.runs[r].local_k.size());
            src += out.runs[r].local_k.size();
        }
    }
    // Gather from the prepared weight: run r's compact K first, then zero K.
    for (size_t r = 0; r < runs; ++r) {
        const run & cur = out.runs[r];
        for (size_t kk = 0; kk < cur.local_k.size(); ++kk) {
            const size_t k_src = (size_t) cur.block_id * RUN_K + cur.local_k[kk];
            std::memcpy(out.b + (r * RUN_K + kk) * out.ldb, w.b + k_src * w.ldb, w.n);
        }
        std::memcpy(out.carriers + r * out.cstride, w.carriers + (size_t) cur.block_id * w.cstride,
                    w.n * sizeof(uint32_t));
    }

    // The residual plan: once, now that the packet fixed M_r and K_execution.
    if (potal_tile_select(out.m, out.n, out.k_execution, (uint32_t) (RUN_K / HW_DIM), &out.plan) != POTAL_OK) {
        return rmd::RmdStatus::unsupported_route;
    }
    return rmd::RmdStatus::success;
} catch (const std::bad_alloc &) {
    return rmd::RmdStatus::allocation_failure;
} catch (const std::length_error &) {
    return rmd::RmdStatus::overflow;
}

rmd::RmdStatus reconstruct(const rmd::StripePacket & packet, const request & req,
                           rmd::BlockScaledInt64Correction & out) {
    if (req.m != req.rows.size()) {
        return rmd::RmdStatus::invalid_arguments;
    }
    return reconstruct_rows(packet, req.rows, req.c, req.ldc, req.n, out);
}

rmd::RmdStatus reconstruct_rows(const rmd::StripePacket & packet, const std::vector<lane_row> & rows,
                                const int32_t * c_rows, size_t ldc, size_t n,
                                rmd::BlockScaledInt64Correction & out) try {
    const rmd::BalancedRadixContract radix = rmd::balanced_radix_contract(packet.digit_bits);
    size_t                           value_count = 0;
    if (radix.radix == 0 || c_rows == nullptr || n != packet.logical_j ||
        __builtin_mul_overflow(packet.row_count, n, &value_count)) {
        return rmd::RmdStatus::invalid_arguments;
    }
    std::vector<wide_t> wide(value_count, 0);
    for (size_t i = 0; i < rows.size(); ++i) {
        const lane_row & row = rows[i];
        if (row.lane >= radix.lane_capacity || row.source_row >= packet.row_count ||
            (i > 0 && (row.lane < rows[i - 1].lane ||
                       (row.lane == rows[i - 1].lane && row.source_row <= rows[i - 1].source_row)))) {
            return rmd::RmdStatus::invalid_packet;
        }
        wide_t place = 1;
        for (uint8_t lane = 0; lane < row.lane; ++lane) {
            if (__builtin_mul_overflow(place, (wide_t) radix.radix, &place)) {
                return rmd::RmdStatus::overflow;
            }
        }
        const int32_t * c   = c_rows + i * ldc;
        wide_t *        dst = wide.data() + (size_t) row.source_row * n;
        for (size_t col = 0; col < n; ++col) {
            wide_t term = c[col];
            if (__builtin_mul_overflow(term, place, &term) || __builtin_add_overflow(dst[col], term, &dst[col])) {
                return rmd::RmdStatus::overflow;
            }
        }
    }
    rmd::BlockScaledInt64Correction staged;
    staged.values.resize(value_count);
    for (size_t i = 0; i < value_count; ++i) {
        if (wide[i] > std::numeric_limits<int64_t>::max() || wide[i] < std::numeric_limits<int64_t>::min()) {
            return rmd::RmdStatus::overflow;
        }
        staged.values[i] = (int64_t) wide[i];
    }
    out = std::move(staged);
    return rmd::RmdStatus::success;
} catch (const std::bad_alloc &) {
    return rmd::RmdStatus::allocation_failure;
} catch (const std::length_error &) {
    return rmd::RmdStatus::overflow;
}

rmd::RmdStatus merge_prepared_hp1(const hp1_weight & weight, size_t row_begin, size_t row_end,
                                  const exsia::StripeMetadataSnapshot & metadata,
                                  const rmd::BlockScaledInt64Correction & correction, float * out, size_t out_rows,
                                  size_t row_stride, size_t col_stride, size_t * nonzero) try {
    // Layout: rmd-compose.cpp prepare_merge_layout.
    const size_t n = weight.n;
    if (out == nullptr || weight.scale == nullptr || n == 0 || row_begin > row_end || row_end > out_rows ||
        row_stride == 0 || col_stride == 0) {
        return rmd::RmdStatus::invalid_arguments;
    }
    const size_t rows   = row_end - row_begin;
    size_t       values = 0, row_offset = 0, col_offset = 0, last = 0;
    if (__builtin_mul_overflow(rows, n, &values) || correction.values.size() != values) {
        return rmd::RmdStatus::invalid_arguments;
    }
    if (rows != 0 && (__builtin_mul_overflow(row_end - 1, row_stride, &row_offset) ||
                      __builtin_mul_overflow(n - 1, col_stride, &col_offset) ||
                      __builtin_add_overflow(row_offset, col_offset, &last))) {
        return rmd::RmdStatus::invalid_arguments;
    }

    // Column scales: validated with the whole weight by prepare_weight(), only
    // re-tested here as the Gemmini merge tests its prepared ones.
    for (size_t col = 0; col < n; ++col) {
        if (!finite_bits(weight.scale[col])) {
            return rmd::RmdStatus::unsupported_route;
        }
    }

    // ActivationMetadataView over this stripe's one theta: rows present, a
    // theta, and float(2^theta) finite and positive.
    if (rows == 0 || metadata.theta == std::numeric_limits<int16_t>::min()) {
        return rmd::RmdStatus::invalid_arguments;
    }
    const float activation_scale = std::ldexp(1.0f, metadata.theta);
    if (!finite_bits(activation_scale) || !(activation_scale > 0.0f)) {
        return rmd::RmdStatus::invalid_arguments;
    }

    // merge_rmd_correction_checked, integer route: count the raw corrections
    // (lane terms may cancel; a zero scale must not hide them), stage every
    // cell, then commit.
    std::vector<float> staged(values);
    size_t             staged_nonzero = 0;
    for (size_t row = 0; row < rows; ++row) {
        const float *   dst = out + (row_begin + row) * row_stride;
        const int64_t * src = correction.values.data() + row * n;
        float *         stg = staged.data() + row * n;
        for (size_t col = 0; col < n; ++col) {
            const int64_t value = src[col];
            staged_nonzero += value != 0;
            const double domain_value = static_cast<double>(value) * static_cast<double>(weight.scale[col]);
            const double scaled       = domain_value * static_cast<double>(activation_scale);
            const float  delta        = static_cast<float>(scaled);
            const float  merged       = dst[col * col_stride] + delta;
            if (!finite_bits(domain_value) || !finite_bits(scaled) || !finite_bits(delta) || !finite_bits(merged)) {
                return rmd::RmdStatus::overflow;
            }
            stg[col] = merged;
        }
    }
    for (size_t row = 0; row < rows; ++row) {
        float *       dst = out + (row_begin + row) * row_stride;
        const float * stg = staged.data() + row * n;
        for (size_t col = 0; col < n; ++col) {
            dst[col * col_stride] = stg[col];
        }
    }
    if (nonzero != nullptr) {
        *nonzero = staged_nonzero;
    }
    return rmd::RmdStatus::success;
} catch (const std::bad_alloc &) {
    return rmd::RmdStatus::allocation_failure;
}

} // namespace ggml::potal::residual
