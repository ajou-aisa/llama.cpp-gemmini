#include "attention.hpp"

#include "ggml-gemmini-matmul-config.hpp"
#include "../quants/act/exsia/local.hpp"
#include "../quants/act/exsia/exsia_shift.hpp"
#include "../residual/rmd/rmd-bitmap-builder.hpp"
#include "../residual/rmd/rmd-im2p-executor.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ggml::gemmini {
namespace {
using rmd::RmdStatus;
namespace exsia = quants::act::exsia;
constexpr size_t bk = 32;
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

int32_t low_digit(int32_t value) {
    return static_cast<int32_t>((static_cast<int64_t>(value) + 8) & 15) - 8;
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
    // The quantization contract uses 4*4096 scratchpad rows and 512 accumulator rows,
    // independently of the physical provider's tiling of each raw dot.
    const size_t ip = rmd::align_up(rows, bk)/bk, jp = rmd::align_up(columns, bk)/bk;
    const size_t kp = rmd::align_up(k, bk)/bk;
    size_t i = std::min(ip, size_t{2}), j = std::min(jp, size_t{2}), b = std::min(kp, size_t{64});
    for (;;) {
        bool grew = false;
        if (j < jp && (i*b + b*(j+1))*bk <= 8192 && i*(j+1)*bk <= 256) { ++j; grew = true; }
        if (i < ip && ((i+1)*b + b*j)*bk <= 8192 && (i+1)*j*bk <= 256) { ++i; grew = true; }
        if (b < kp && (i*(b+1) + (b+1)*j)*bk <= 8192) { ++b; grew = true; }
        if (!grew) return i*bk;
    }
}

struct Stationary {
    size_t k, n, kp;
    int base = 0;
    std::vector<int16_t> theta;
    std::array<std::vector<int32_t>, 2> codes;
    std::vector<int64_t> colsum;
};

Stationary stationary(const ggml_tensor & t, size_t head, size_t batch, bool qk) {
    Stationary w{size_t(t.ne[0]), size_t(t.ne[1]), rmd::align_up(size_t(t.ne[0]), bk), 0, {}, {}, {}};
    w.theta.resize(w.kp/bk*w.n);
    w.codes[0].resize(w.kp*w.n);
    if (qk) w.codes[1].resize(w.kp*w.n);
    std::array<float, bk> values{};
    std::vector<int32_t> codes(bk);
    exsia::ExpScanner scanner;
    exsia::WideQuantizer quantizer;
    for (size_t b = 0; b < w.kp/bk; ++b) {
        const size_t valid = std::min(bk, w.k-b*bk);
        for (size_t j = 0; j < w.n; ++j) {
            int16_t e = neg;
            for (size_t k = 0; k < valid; ++k) {
                values[k] = read_value(t, b*bk+k, j, head, batch);
                e = std::max(e, scanner.unbiased_exp(values[k]));
            }
            const int16_t theta = e == neg ? 0 : e - (qk ? 6 : 2);
            w.theta[b*w.n+j] = theta;
            quantizer.quantize_block(values.data(), valid, theta, codes);
            for (size_t k = 0; k < valid; ++k) {
                const int32_t code = std::clamp(codes[k], qk ? -128 : -8, qk ? 119 : 7);
                const size_t index = (b*bk+k)*w.n+j;
                w.codes[0][index] = qk ? low_digit(code) : code;
                if (qk) w.codes[1][index] = (code-w.codes[0][index])/16;
            }
        }
    }
    w.base = *std::min_element(w.theta.begin(), w.theta.end());
    if (!qk) {
        w.colsum.resize(w.n);
        for (size_t b = 0; b < w.kp/bk; ++b) {
            for (size_t j = 0; j < w.n; ++j) {
                int64_t sum = 0;
                for (size_t k = 0; k < bk; ++k) sum += w.codes[0][(b*bk+k)*w.n+j];
                scaled_add(w.colsum[j], sum, w.theta[b*w.n+j]-w.base);
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
        meta.rho = 6;
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
        st.scale = anchor-6;
        for (size_t i = 0; i < rows; ++i)
            for (size_t c = 0; c < k; ++c) {
                const int16_t e = exponents[i*blocks+c/bk];
                codes[i*kp+c] = exsia::detail::shift_q_i32(codes[i*kp+c], e == neg ? 0 : e-anchor);
            }
    } else {
        const float maximum = *std::max_element(values.begin(), values.end());
        st.scale = (maximum > 0 ? scanner.unbiased_exp(maximum) : 0)-7;
        exsia::WideQuantizer quantizer;
        quantizer.quantize_block(values.data(), values.size(), static_cast<int16_t>(st.scale), codes);
    }
    std::vector<uint64_t> selected((rows*kp+63)/64, UINT64_MAX);
    rmd::RmdBitmapBuilder builder;
    builder.reset(begin, begin, rows, k, n, 4, selected, kp);
    require(builder.status() == RmdStatus::success, builder.status());
    for (size_t i = 0; i < rows; ++i) {
        for (size_t c = 0; c < k; ++c) {
            const size_t index = i*kp+c;
            const int32_t code = qk ? codes[index] : std::clamp(codes[index], 0, 255);
            const int32_t main = qk ? low_digit(code) : (code & 15)-8;
            st.main[index] = static_cast<int8_t>(main);
            // Packet lane 0 represents original lane 1. Dividing by 16 also avoids
            // overflowing INT32 when a positive saturated code has low digit -1.
            const int32_t upper = static_cast<int32_t>((int64_t(code)-main-(qk ? 0 : 8))/16);
            if (upper && !builder.emit(i, c, upper)) throw builder.status();
        }
    }
    if (!builder.empty()) {
        st.upper = builder.finish();
        require(bool(st.upper), builder.status());
    }
    return st;
}

struct RawDots {
    rmd::detail::Im2pProviderStatsAggregate aggregate;
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    std::unique_ptr<im2p_sim_t, decltype(&im2p_sim_destroy)> sim{nullptr, im2p_sim_destroy};
    RawDots() {
        require(im2p_sim_implementation() && std::strcmp(im2p_sim_implementation(), "gemmini-hp1-integrated-v1") == 0 &&
                im2p_sim_abi_version() == IM2P_ABI_VERSION && im2p_sim_dim() == 32 &&
                im2p_sim_activation_bits() == 4 && im2p_sim_weight_bits() == 4 &&
                im2p_sim_activation_storage_bytes() == 1 && im2p_sim_weight_storage_bytes() == 1 &&
                im2p_compiled_accumulator_rows() == 512,
                RmdStatus::unsupported_route);
        sim.reset(im2p_sim_create());
        require(bool(sim), RmdStatus::execution_failed);
    }
#endif
    void run(const int8_t * a, size_t rows, size_t stride, const int32_t * w,
             size_t n, size_t k, std::vector<int64_t> & out) {
        out.resize(rows*n);
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
        rmd::detail::Im2pCompactDot dot;
        dot.operand_bits = 4;
        dot.activations = a;
        dot.rows = rows;
        dot.activation_row_stride_bytes = stride;
        dot.weights = w;
        dot.columns = n;
        dot.weight_row_stride = n;
        dot.k = k;
        dot.vector_op = IM2P_VECTOR_BYPASS;
        const auto status = rmd::detail::execute_im2p_compact_dot(sim.get(), dot, out.data(), n, aggregate);
        require(status == RmdStatus::success, status);
#else
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < n; ++j) {
                int64_t sum = 0;
                for (size_t c = 0; c < k; ++c) sum += int32_t(a[i*stride+c])*w[c*n+j];
                out[i*n+j] = sum;
            }
#endif
    }
};

void multiply(RawDots & raw, const StreamStripe & st, const Stationary & w, bool qk,
              float * output, AttentionMetrics & metrics) {
    std::vector<int64_t> acc(st.rows*w.n), part;
    std::vector<int32_t> compact(bk*w.n);
    auto merge = [&](size_t i, size_t source, size_t b, int extra) {
        for (size_t j = 0; j < w.n; ++j)
            scaled_add(acc[i*w.n+j], part[source*w.n+j], w.theta[b*w.n+j]-w.base+extra);
    };
    const uint64_t ntiles = (w.n+bk-1)/bk;
    for (size_t pass = 0; pass < (qk ? 2u : 1u); ++pass) {
        for (size_t b = 0; b < w.kp/bk; ++b) {
            raw.run(st.main.data()+b*bk, st.rows, st.kp, w.codes[pass].data()+b*bk*w.n, w.n, bk, part);
            ++metrics.main_dots;
            (pass ? metrics.residual_fragments : metrics.main_fragments) += ((st.rows+bk-1)/bk)*ntiles;
            for (size_t i = 0; i < st.rows; ++i) merge(i, i, b, int(pass)*4);
        }
        if (!st.upper) continue;
        for (const auto & block : st.upper->blocks) {
            for (const auto & group : block.groups) {
                std::fill(compact.begin(), compact.end(), 0);
                size_t ck = 0;
                for (size_t c = 0; c < bk; ++c) {
                    if (!(group.k_mask & (uint32_t{1} << c))) continue;
                    std::copy_n(w.codes[pass].data()+(block.global_k_begin+c)*w.n, w.n, compact.data()+ck*w.n);
                    ++ck;
                }
                raw.run(st.upper->stacked_activation.signed_int8.data()+group.activation_offset,
                        group.row_ids.size(), group.padded_k_count, compact.data(), w.n, group.padded_k_count, part);
                ++metrics.residual_dots;
                metrics.residual_fragments += ((group.row_ids.size()+bk-1)/bk)*ntiles;
                for (size_t l = 0; l < group.lane_positions.size(); ++l) {
                    const int shift = 4*(int(block.lane_ids[group.lane_positions[l]])+1+int(pass));
                    for (size_t row = group.row_offsets[l]; row < group.row_offsets[l+1]; ++row)
                        merge(group.row_ids[row], row, block.block_id, shift);
                }
            }
        }
    }
    for (size_t i = 0; i < st.rows; ++i)
        for (size_t j = 0; j < w.n; ++j) {
            if (!qk) scaled_add(acc[i*w.n+j], w.colsum[j], 3);
            const float value = static_cast<float>(std::ldexp(static_cast<double>(acc[i*w.n+j]), st.scale+w.base));
            require(std::isfinite(value), RmdStatus::overflow);
            output[i*w.n+j] = value;
        }
    if (!qk) metrics.add_ops += st.rows*w.n;
}
} // namespace

bool attention_supported(const ggml_tensor * op) {
    if (!GGML_GEMMINI_ENABLE_A4NKS_ATTENTION || !op ||
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
        RawDots raw;
        const auto & a = *op->src[0];
        const auto & b = *op->src[1];
        const bool qk = ggml_gemmini_get_attention_role(op) == GGML_GEMMINI_ATTENTION_QK;
        const size_t rows = b.ne[1], n = a.ne[1], rs = stripe_rows(rows, n, b.ne[0]);
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
                            multiply(raw, st, w, qk, staged.data()+((batch*b.ne[2]+head)*rows+begin)*n, pending);
                        }
            }
        const auto stats_status = rmd::checked_accumulate_provider_stats(pending.provider, raw.aggregate.stats);
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
} // namespace ggml::gemmini
