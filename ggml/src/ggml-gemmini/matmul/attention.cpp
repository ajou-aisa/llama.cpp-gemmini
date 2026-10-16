#include "attention.hpp"

#include "ggml-gemmini-matmul-config.hpp"
#include "../ggml-gemmini-args.h"
#include <gemmini.h>
#include "../quants/act/exsia/local.hpp"
#include "../quants/act/exsia/exsia_shift.hpp"
#include "../quants/common/hp1_scu.hpp"
#include "../quants/common/math.hpp"
#include "../residual/rmd/rmd-bitmap-builder.hpp"
#include "../residual/rmd/rmd-im2p-executor.hpp"
#include "../residual/rmd/rmd-run-aware.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace ggml::gemmini {
#if GGML_GEMMINI_ENABLE_A4NKS_ATTENTION
namespace {
using rmd::RmdStatus;
namespace exsia = quants::act::exsia;
constexpr size_t bk = 32;
constexpr int pe_bits = GGML_GEMMINI_ACTIVATION_BITS;
constexpr int code_bits = 2 * pe_bits;
constexpr int radix = 1 << pe_bits;
constexpr int zero_point = radix / 2;
constexpr int16_t neg = std::numeric_limits<int16_t>::min();

void require(bool valid, RmdStatus status = RmdStatus::invalid_arguments) {
    if (!valid) throw status;
}

struct NearestRounding {
    int previous = std::fegetround();
    NearestRounding() { require(previous != -1 && std::fesetround(FE_TONEAREST) == 0); }
    ~NearestRounding() { std::fesetround(previous); }
};

float read_value(const ggml_tensor & t, size_t k, size_t row, size_t head, size_t batch) {
    const auto * p = static_cast<const char *>(t.data) + k*t.nb[0] + row*t.nb[1] + head*t.nb[2] + batch*t.nb[3];
    float value;
    if (t.type == GGML_TYPE_F16) {
        ggml_fp16_t half;
        std::memcpy(&half, p, sizeof(half));
        value = ggml_fp16_to_fp32(half);
    } else {
        std::memcpy(&value, p, sizeof(value));
    }
    require(std::isfinite(value));
    return value;
}

void scaled_add(int64_t & accumulator, int64_t part, int shift) {
    if (!part) return;
    require(shift >= 0 && shift <= 63, RmdStatus::overflow);
    const __int128 sum = static_cast<__int128>(accumulator) +
                         static_cast<__int128>(part) * (static_cast<__int128>(1) << shift);
    require(sum >= std::numeric_limits<int64_t>::min() && sum <= std::numeric_limits<int64_t>::max(),
            RmdStatus::overflow);
    accumulator = static_cast<int64_t>(sum);
}

size_t stripe_rows(size_t rows, size_t columns, size_t k) {
    const auto geometry = make_potal_geometry({rows, columns, k}, pe_bits);
    require(geometry.ok());
    return geometry.geometry.stripe_rows;
}

struct Stationary {
    size_t k, n, kp;
    std::vector<int16_t> base;
    std::vector<uint32_t> carriers;
    std::array<std::vector<int32_t>, 2> codes;
    std::vector<int64_t> colsum;
};

Stationary stationary(const ggml_tensor & t, size_t head, size_t batch, bool qk) {
    Stationary w{size_t(t.ne[0]), size_t(t.ne[1]), rmd::align_up(size_t(t.ne[0]), bk), {}, {}, {}, {}};
    w.base.resize(w.n);
    w.carriers.resize(w.kp/bk*w.n);
    w.codes[0].resize(w.kp*w.n);
    if (qk) w.codes[1].resize(w.kp*w.n);
    std::array<float, bk> values{};
    std::vector<int32_t> codes(bk);
    std::vector<int16_t> theta(w.kp/bk);
    exsia::ExpScanner scanner;
    exsia::WideQuantizer quantizer;
    for (size_t j = 0; j < w.n; ++j) {
        w.base[j] = std::numeric_limits<int16_t>::max();
        for (size_t b = 0; b < w.kp/bk; ++b) {
            const size_t valid = std::min(bk, w.k-b*bk);
            int16_t e = neg;
            for (size_t k = 0; k < valid; ++k) {
                values[k] = read_value(t, b*bk+k, j, head, batch);
                e = std::max(e, scanner.unbiased_exp(values[k]));
            }
            theta[b] = e == neg ? neg : e - (qk ? code_bits-2 : pe_bits-2);
            if (e == neg) continue;
            w.base[j] = std::min(w.base[j], theta[b]);
            quantizer.quantize_block(values.data(), valid, theta[b], codes);
            for (size_t k = 0; k < valid; ++k) {
                const int32_t code = std::clamp(codes[k], qk ? -(1 << (code_bits-1)) : -zero_point,
                                                qk ? (1 << (code_bits-1))-zero_point-1 : zero_point-1);
                const size_t index = (b*bk+k)*w.n+j;
                w.codes[0][index] = qk ? lowdigit(code, pe_bits) : code;
                if (!w.codes[1].empty()) w.codes[1][index] = (code-w.codes[0][index])/radix;
            }
        }
        if (w.base[j] == std::numeric_limits<int16_t>::max()) w.base[j] = 0;
        for (size_t b = 0; b < theta.size(); ++b)
            require(quants::hp1::encode_carrier(theta[b] == neg ? neg : theta[b]-w.base[j],
                                                w.carriers[b*w.n+j]));
    }
    if (!qk) {
        w.colsum.resize(w.n);
        for (size_t b = 0; b < w.kp/bk; ++b) {
            for (size_t j = 0; j < w.n; ++j) {
                if (w.carriers[b*w.n+j] == quants::hp1::zero_carrier) continue;
                int64_t sum = 0;
                for (size_t k = 0; k < bk; ++k) sum += w.codes[0][(b*bk+k)*w.n+j];
                scaled_add(w.colsum[j], sum, w.carriers[b*w.n+j]);
            }
        }
    }
    return w;
}

struct StreamStripe {
    size_t rows, kp;
    int scale = 0;
    std::vector<int8_t> main;
    rmd::StripePacketHandle upper;
};

StreamStripe stream(const ggml_tensor & t, size_t head, size_t batch, size_t begin,
                    size_t rows, size_t n, bool qk) {
    const size_t k = t.ne[0], kp = rmd::align_up(k, bk), blocks = kp/bk;
    StreamStripe st{rows, kp, 0, std::vector<int8_t>(rows*kp), {}};
    std::vector<float> values(rows*kp);
    for (size_t i = 0; i < rows; ++i)
        for (size_t c = 0; c < k; ++c) {
            values[i*kp+c] = read_value(t, c, begin+i, head, batch);
            require(qk || (values[i*kp+c] >= 0 && values[i*kp+c] <= 1));
        }
    std::vector<int32_t> codes(rows*kp);
    exsia::ExpScanner scanner;
    if (qk) {
        exsia::Meta meta;
        meta.rho = code_bits-2;
        exsia::ExSIAState state;
        state.B_size = bk;
        state.K_logical = k;
        state.K_padded = kp;
        state.blocks_per_row = blocks;
        exsia::StripeScratch scratch;
        require(scratch.prepare(bk), RmdStatus::allocation_failure);
        exsia::LocalStage local;
        exsia::StripeState stripe;
        std::vector<int16_t> exponents(rows*blocks);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t b = 0; b < blocks; ++b) {
                uint64_t mask_word = 0;
                exsia::BlockMask mask(&mask_word, bk);
#if EXSIA_BRANCH_COUNTS_ENABLED
                exsia::LocalBlockCycleSample sample;
#endif
                require(local.run_optimized(meta, state, values.data()+i*kp+b*bk, std::min(bk, k-b*bk),
                                           bk, i, b, scratch, mask, codes.data()+i*kp+b*bk, exponents[i*blocks+b]
#if EXSIA_BRANCH_COUNTS_ENABLED
                                           , sample
#endif
                                           ), RmdStatus::execution_failed);
                scanner.update_stripe_top2_exp(stripe, exponents[i*blocks+b]);
            }
        }
        const int16_t anchor = stripe.e2 != neg ? stripe.e2 : stripe.e1 != neg ? stripe.e1 : 0;
        st.scale = anchor-meta.rho;
        for (size_t i = 0; i < rows; ++i)
            for (size_t c = 0; c < k; ++c) {
                const int16_t e = exponents[i*blocks+c/bk];
                const int16_t delta = e == neg ? 0 : e-anchor;
                codes[i*kp+c] = exsia::detail::shift_q_i32(codes[i*kp+c], delta);
            }
    } else {
        const float maximum = *std::max_element(values.begin(), values.end());
        st.scale = (maximum > 0 ? scanner.unbiased_exp(maximum) : 0)-(code_bits-1);
        exsia::WideQuantizer quantizer;
        quantizer.quantize_block(values.data(), values.size(), static_cast<int16_t>(st.scale), codes);
    }
    std::vector<uint64_t> selected((rows*kp+63)/64, UINT64_MAX);
    rmd::RmdBitmapBuilder builder;
    builder.reset(begin, begin, rows, k, n, pe_bits, selected, kp);
    require(builder.status() == RmdStatus::success, builder.status());
    for (size_t i = 0; i < rows; ++i) {
        for (size_t c = 0; c < k; ++c) {
            const size_t index = i*kp+c;
            const int32_t code = qk ? codes[index] : std::clamp(codes[index], 0, (1 << code_bits)-1);
            const int32_t main = lowdigit(code - (qk ? 0 : zero_point), pe_bits);
            st.main[index] = static_cast<int8_t>(main);
            // Packet lane 0 represents original lane 1. Dividing by the radix avoids
            // overflowing INT32 when a positive saturated code has low digit -1.
            const int32_t upper = static_cast<int32_t>((int64_t(code)-main-(qk ? 0 : zero_point))/radix);
            if (upper && !builder.emit(i, c, upper)) throw builder.status();
        }
    }
    if (!builder.empty()) {
        st.upper = builder.finish();
        require(bool(st.upper), builder.status());
    }
    return st;
}

enum class QkSchedule { vanilla, tile, reuse };

QkSchedule qk_schedule() {
    const char * value = std::getenv("GGML_GEMMINI_QK_SCHEDULE");
    if (!value || !*value || std::strcmp(value, "vanilla") == 0) return QkSchedule::vanilla;
    if (std::strcmp(value, "tile") == 0) return QkSchedule::tile;
    require(std::strcmp(value, "reuse") == 0);
    return QkSchedule::reuse;
}

struct QkMask {
    size_t tiles = 0;
    std::vector<uint8_t> live;

    QkMask(const ggml_tensor * mask, size_t rows, size_t columns, QkSchedule schedule) {
        if (!mask || schedule == QkSchedule::vanilla) return;
        require(mask->data && (mask->type == GGML_TYPE_F32 || mask->type == GGML_TYPE_F16) &&
                ggml_is_contiguous(mask) && ggml_is_matrix(mask) &&
                size_t(mask->ne[0]) == columns && size_t(mask->ne[1]) >= rows);
        tiles = (columns+rmd::kArrayDim-1)/rmd::kArrayDim;
        live.resize(rows*tiles);
        for (size_t i = 0; i < rows; ++i)
            for (size_t tile = 0; tile < tiles; ++tile)
                for (size_t j = tile*rmd::kArrayDim; j < std::min(columns, (tile+1)*rmd::kArrayDim); ++j) {
                    const char * p = static_cast<const char *>(mask->data) + i*mask->nb[1] + j*mask->nb[0];
                    float value;
                    if (mask->type == GGML_TYPE_F16) {
                        ggml_fp16_t half;
                        std::memcpy(&half, p, sizeof(half));
                        value = ggml_fp16_to_fp32(half);
                    } else std::memcpy(&value, p, sizeof(value));
                    if (value != -std::numeric_limits<float>::infinity()) {
                        live[i*tiles+tile] = 1;
                        break;
                    }
                }
        if (std::all_of(live.begin(), live.end(), [](uint8_t value) { return value != 0; })) live.clear();
    }
};

struct QkRegion {
    size_t row, rows, column, columns;
};

std::vector<QkRegion> qk_regions(const QkMask & mask, size_t begin, size_t rows, size_t columns,
                                  size_t group_rows, const std::vector<rmd::RunAwareRow> * row_map = nullptr) {
    std::vector<QkRegion> regions;
    for (size_t row = 0; row < rows; row += group_rows) {
        const size_t count = std::min(group_rows, rows-row);
        auto live = [&](size_t tile) {
            for (size_t i = row; i < row+count; ++i) {
                const size_t original_row = begin + (row_map ? (*row_map)[i].source_row : i);
                if (mask.live[original_row*mask.tiles+tile]) return true;
            }
            return false;
        };
        size_t first = 0;
        while (first < mask.tiles) {
            if (!live(first)) { ++first; continue; }
            size_t last = first+1;
            while (last < mask.tiles && live(last)) ++last;
            regions.push_back({row, count, first*rmd::kArrayDim,
                               std::min(columns, last*rmd::kArrayDim)-first*rmd::kArrayDim});
            first = last;
        }
    }
    return regions;
}

struct ScuGemm {
    rmd::detail::Im2pProviderStatsAggregate aggregate;
    const char * layer;
#if GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION
    std::unique_ptr<im2p_sim_t, decltype(&im2p_sim_destroy)> sim{nullptr, im2p_sim_destroy};
#endif
    explicit ScuGemm(const char * name) : layer(name) {
#if GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION
        require(im2p_sim_implementation() && std::strcmp(im2p_sim_implementation(), "gemmini-hp1-integrated-v1") == 0 &&
                im2p_sim_abi_version() == IM2P_ABI_VERSION && im2p_sim_dim() == 32 &&
                im2p_sim_activation_bits() == pe_bits && im2p_sim_weight_bits() == pe_bits &&
                im2p_sim_activation_storage_bytes() == 1 && im2p_sim_weight_storage_bytes() == 1 &&
                im2p_compiled_accumulator_rows() == 512,
                RmdStatus::unsupported_route);
        sim.reset(im2p_sim_create());
        require(bool(sim), RmdStatus::execution_failed);
#endif
    }
#if !GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION
    static void software(const int8_t * a, size_t rows, size_t stride, const int32_t * w,
                         size_t n, size_t k, size_t stationary_stride, const uint32_t * carriers,
                         std::vector<int64_t> & out, const std::vector<rmd::RunAwareRun> * runs = nullptr) {
        out.resize(rows*n);
        const size_t blocks = runs ? runs->size() : (k+bk-1)/bk;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < n; ++j) {
                int32_t acc = 0;
                for (size_t b = 0; b < blocks; ++b) {
                    const size_t begin = runs ? (*runs)[b].compact_k_begin : b*bk;
                    const size_t count = runs ? (*runs)[b].compact_k_count : std::min(bk, k-begin);
                    for (size_t c0 = 0; c0 < count; c0 += rmd::kArrayDim) {
                        int32_t part = 0;
                        for (size_t c = c0; c < std::min(count, c0+rmd::kArrayDim); ++c) {
                            const size_t original_k = runs
                                ? (*runs)[b].original_global_k_begin + (*runs)[b].original_local_k[c]
                                : begin+c;
                            part += int32_t(a[i*stride+begin+c])*w[original_k*stationary_stride+j];
                        }
                        const size_t block = runs ? (*runs)[b].original_block_id : b;
                        acc = quants::hp1::accumulate(acc, quants::hp1::apply_validated(part, carriers[block*stationary_stride+j]));
                    }
                }
                out[i*n+j] = acc;
            }
    }
#endif
    void run(const StreamStripe & st, const Stationary & w, size_t pass,
             const QkRegion & region, size_t begin, std::vector<int64_t> & out) {
#if GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION
        rmd::detail::Im2pCompactDot dot;
        dot.operand_bits = pe_bits;
        dot.activations = st.main.data()+region.row*st.kp;
        dot.rows = region.rows;
        dot.activation_row_stride_bytes = st.kp;
        dot.weights = w.codes[pass].data()+region.column;
        dot.columns = region.columns;
        dot.weight_row_stride = w.n;
        dot.k = w.k;
        dot.scu_carriers = w.carriers.data()+region.column;
        dot.scu_row_stride = w.n;
        dot.vector_op = IM2P_VECTOR_LEFT_SHIFT;
        dot.trace_role = GEMMINI_TRACE_ROLE_DENSE;
        dot.timing_identity.interval.layer = layer;
        dot.timing_identity.identity_mask = GEMMINI_CYCLE_HAS_STRIPE_ID;
        dot.timing_identity.stripe_id = begin;
        dot.stripe_id = begin;
        dot.source_row_begin = begin+region.row;
        dot.source_row_count = region.rows;
        dot.column_offset = region.column;
        const auto status = rmd::detail::execute_im2p_compact_dot(sim.get(), dot, out, aggregate);
        require(status == RmdStatus::success, status);
#else
        (void)begin;
        software(st.main.data()+region.row*st.kp, region.rows, st.kp, w.codes[pass].data()+region.column,
                 region.columns, w.k, w.n, w.carriers.data()+region.column, out);
#endif
    }
    void run(const rmd::RunAwareRequest & request, const Stationary & w, size_t pass,
             const QkRegion & region, std::vector<int64_t> & out) {
#if GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION
        rmd::detail::Im2pRunAwareWork work;
        work.request = &request;
        work.original_weights = &w.codes[pass];
        work.original_carriers = &w.carriers;
        work.row_begin = region.row;
        work.row_count = region.rows;
        work.column_begin = region.column;
        work.column_count = region.columns;
        work.timing_identity.interval.layer = layer;
        work.timing_identity.identity_mask = GEMMINI_CYCLE_HAS_STRIPE_ID;
        work.timing_identity.stripe_id = request.stripe_id;
        const auto status = rmd::detail::execute_im2p_run_aware(sim.get(), work, out, aggregate);
        require(status == RmdStatus::success, status);
#else
        software(request.activations.data()+region.row*request.k, region.rows, request.k,
                 w.codes[pass].data()+region.column, region.columns, request.k, request.n,
                 w.carriers.data()+region.column, out, &request.runs);
#endif
    }
};

void multiply(ScuGemm & scu, const StreamStripe & st, const Stationary & w, bool qk, size_t begin,
              const QkMask & mask, QkSchedule schedule, float * output, AttentionMetrics & metrics) {
    std::vector<int64_t> acc(st.rows*w.n), part;
    auto merge = [&](size_t i, size_t source, int shift, const QkRegion & region) {
        for (size_t j = 0; j < region.columns; ++j)
            scaled_add(acc[i*w.n+region.column+j], part[source*region.columns+j], shift);
    };
    rmd::RunAwareRequest residual;
    uint64_t residual_k_fragments = 0;
    std::vector<QkRegion> main_regions, residual_regions;
    if (!mask.live.empty()) {
        size_t main_rows = rmd::kArrayDim;
        if (schedule == QkSchedule::reuse) {
            ggml_gemmini_args_t selected{};
            selected.I = st.rows;
            selected.J = w.n;
            selected.K = w.k;
            gemmini_set_tile_ws(&selected);
            main_rows = selected.tile_I*rmd::kArrayDim;
        }
        require(main_rows);
        main_regions = qk_regions(mask, begin, st.rows, w.n, main_rows);
    }
    if (st.upper && (mask.live.empty() || !main_regions.empty())) {
        const auto status = rmd::build_run_aware_activations(*st.upper, residual);
        require(status == RmdStatus::success, status);
        for (const auto & run : residual.runs)
            residual_k_fragments += (run.compact_k_count+rmd::kArrayDim-1)/rmd::kArrayDim;
        if (!mask.live.empty()) {
            const size_t residual_rows = schedule == QkSchedule::reuse
                ? residual.tile_i*rmd::kArrayDim : rmd::kArrayDim;
            require(residual_rows);
            residual_regions = qk_regions(mask, begin, residual.m, w.n, residual_rows, &residual.rows);
        }
    }
    for (size_t pass = 0; pass < (w.codes[1].empty() ? 1u : 2u); ++pass) {
        auto main = [&](const QkRegion & region) {
            scu.run(st, w, pass, region, begin, part);
            ++metrics.main_dots;
            (pass ? metrics.residual_fragments : metrics.main_fragments) +=
                ((region.rows+rmd::kArrayDim-1)/rmd::kArrayDim)*
                ((region.columns+rmd::kArrayDim-1)/rmd::kArrayDim)*(w.kp/bk);
            for (size_t i = 0; i < region.rows; ++i) merge(region.row+i, i, int(pass)*pe_bits, region);
        };
        if (mask.live.empty()) main({0, st.rows, 0, w.n});
        else for (const auto & region : main_regions) main(region);
        if (residual.empty()) continue;
        auto upper = [&](const QkRegion & region) {
            scu.run(residual, w, pass, region, part);
            ++metrics.residual_dots;
            metrics.residual_fragments += ((region.rows+rmd::kArrayDim-1)/rmd::kArrayDim)*
                ((region.columns+rmd::kArrayDim-1)/rmd::kArrayDim)*residual_k_fragments;
            for (size_t i = 0; i < region.rows; ++i) {
                const auto & row = residual.rows[region.row+i];
                merge(row.source_row, i, pe_bits*(int(row.original_lane_id)+1+int(pass)), region);
            }
        };
        if (mask.live.empty()) upper({0, residual.m, 0, w.n});
        else for (const auto & region : residual_regions) upper(region);
    }
    for (size_t i = 0; i < st.rows; ++i)
        for (size_t j = 0; j < w.n; ++j) {
            if (!qk) scaled_add(acc[i*w.n+j], w.colsum[j], pe_bits-1);
            const float value = static_cast<float>(std::ldexp(static_cast<double>(acc[i*w.n+j]), st.scale+w.base[j]));
            require(std::isfinite(value), RmdStatus::overflow);
            output[i*w.n+j] = value;
        }
    if (!qk) metrics.add_ops += st.rows*w.n;
}
} // namespace

bool attention_supported(const ggml_tensor * op) {
    if (!op ||
        ggml_gemmini_get_attention_role(op) == GGML_GEMMINI_ATTENTION_NONE ||
        op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op) || !op->src[0] || !op->src[1]) return false;
    const auto & a = *op->src[0];
    const auto & b = *op->src[1];
    if ((a.type != GGML_TYPE_F32 && a.type != GGML_TYPE_F16) || b.type != GGML_TYPE_F32) return false;
    for (int d = 0; d < 4; ++d)
        if (a.ne[d] <= 0 || b.ne[d] <= 0 || a.nb[d] < ggml_type_size(a.type) || b.nb[d] < sizeof(float)) return false;
    return a.ne[0] == b.ne[0] && b.ne[2]%a.ne[2] == 0 && b.ne[3]%a.ne[3] == 0 &&
           op->ne[0] == a.ne[1] && op->ne[1] == b.ne[1] && op->ne[2] == b.ne[2] && op->ne[3] == b.ne[3];
}

rmd::RmdStatus attention_execute(ggml_tensor * op, AttentionMetrics & metrics) {
    if (!attention_supported(op)) return RmdStatus::unsupported_route;
    if (!op->data || !op->src[0]->data || !op->src[1]->data) return RmdStatus::invalid_arguments;
    try {
        NearestRounding rounding;
        ScuGemm scu(op->name);
        const auto & a = *op->src[0];
        const auto & b = *op->src[1];
        const bool qk = ggml_gemmini_get_attention_role(op) == GGML_GEMMINI_ATTENTION_QK;
        const size_t rows = b.ne[1], n = a.ne[1], rs = stripe_rows(rows, n, b.ne[0]);
        const auto schedule = qk ? qk_schedule() : QkSchedule::vanilla;
        const QkMask mask(qk ? op->src[2] : nullptr, rows, n, schedule);
        const size_t head_repeat = b.ne[2]/a.ne[2], batch_repeat = b.ne[3]/a.ne[3];
        std::vector<float> staged(size_t(ggml_nelements(op)));
        AttentionMetrics pending = metrics;
        for (size_t ab = 0; ab < size_t(a.ne[3]); ++ab)
            for (size_t ah = 0; ah < size_t(a.ne[2]); ++ah) {
                const auto w = stationary(a, ah, ab, qk);
                if (!qk) pending.add_ops += w.k*w.n;
                for (size_t batch = ab*batch_repeat; batch < (ab+1)*batch_repeat; ++batch)
                    for (size_t head = ah*head_repeat; head < (ah+1)*head_repeat; ++head)
                        for (size_t begin = 0; begin < rows; begin += rs) {
                            const auto st = stream(b, head, batch, begin, std::min(rs, rows-begin), n, qk);
                            multiply(scu, st, w, qk, begin, mask, schedule,
                                     staged.data()+((batch*b.ne[2]+head)*rows+begin)*n, pending);
                        }
            }
        const auto stats_status = rmd::checked_accumulate_provider_stats(pending.provider, scu.aggregate.stats);
        require(stats_status == RmdStatus::success, stats_status);
        ++(qk ? pending.qk_nodes : pending.pv_nodes);
        std::memcpy(op->data, staged.data(), staged.size()*sizeof(float));
        metrics = pending;
        return RmdStatus::success;
    } catch (RmdStatus status) {
        return status;
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    } catch (const std::length_error &) {
        return RmdStatus::overflow;
    }
}
#else
bool attention_supported(const ggml_tensor *) {
    return false;
}

rmd::RmdStatus attention_execute(ggml_tensor *, AttentionMetrics &) {
    return rmd::RmdStatus::unsupported_route;
}
#endif
} // namespace ggml::gemmini
