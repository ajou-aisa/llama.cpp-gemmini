#include "folding.hpp"
#include "exsia_shift.hpp"
#include "ggml-gemmini-args.h"
#include "../../../ggml-gemmini-evaluation-observer.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace ggml::gemmini::quants::act::exsia {
using namespace detail;

namespace {

inline int32_t saturate_activation(int32_t q) {
    constexpr int32_t qmax = config::GGML_GEMMINI_ACTIVATION_QMAX;
    constexpr int32_t qmin = config::GGML_GEMMINI_ACTIVATION_QMIN;
    return q > qmax ? qmax : (q < qmin ? qmin : q);
}

// PreparedShift for 1 <= shift <= 31 in 32-bit lanes so the block loop vectorizes:
// |q| <= 2^31 fits uint32 and the round-to-nearest-even result is at most 2^30 + 1.
inline int32_t round_shift_right_even(int32_t q, int shift) {
    const bool     negative  = q < 0;
    const uint32_t magnitude = negative ? 0u - static_cast<uint32_t>(q) : static_cast<uint32_t>(q);
    const uint32_t half      = uint32_t{1} << (shift - 1);
    const uint32_t quotient  = magnitude >> shift;
    const uint32_t remainder = magnitude & ((half << 1) - 1);
    const uint32_t rounded =
        quotient + (remainder > half || (remainder == half && (quotient & 1) != 0));
    return negative ? -static_cast<int32_t>(rounded) : static_cast<int32_t>(rounded);
}

// Bits [start, start + count) of a row-major bit matrix, count <= 64.
inline uint64_t load_mask_bits(const std::vector<uint64_t> & words, size_t start, size_t count) {
    const size_t word   = start / 64;
    const size_t offset = start % 64;
    uint64_t     bits   = words[word] >> offset;
    if (offset != 0 && offset + count > 64)
        bits |= words[word + 1] << (64 - offset);
    return count == 64 ? bits : bits & ((uint64_t{1} << count) - 1);
}

// Shifted values and their saturated dense codes for one block, as if no column were selected.
template <typename T>
void fold_block_dense(T * codes, int32_t * shifted, const int32_t * q, size_t n, int16_t delta) {
    if (delta == 0) {
        for (size_t i = 0; i < n; ++i) {
            shifted[i] = q[i];
            codes[i]   = static_cast<T>(saturate_activation(q[i]));
        }
    } else if (delta < 0 && delta > -32) {
        const int shift = -static_cast<int>(delta);
        for (size_t i = 0; i < n; ++i) {
            shifted[i] = round_shift_right_even(q[i], shift);
            codes[i]   = static_cast<T>(saturate_activation(shifted[i]));
        }
    } else {
        const detail::PreparedShift prepared(delta);
        for (size_t i = 0; i < n; ++i) {
            shifted[i] = prepared(q[i]);
            codes[i]   = static_cast<T>(saturate_activation(shifted[i]));
        }
    }
}

// Direct block stores need every saturated activation code to be representable in A, which is
// what A.set() would otherwise check per column.
bool dense_rows_writable(const QuantizedActivationBuffer & a, size_t row_end, size_t cols) {
    if (!a.valid() || row_end > a.rows || cols > a.cols || (a.bits != 4 && a.bits != 8 && a.bits != 16) ||
        a.bits < GGML_GEMMINI_ACTIVATION_BITS)
        return false;
    const size_t element_bytes = a.bits == 16 ? sizeof(int16_t) : 1;
    return a.row_stride_bytes >= cols * element_bytes &&
           a.row_offset + row_end <= a.bytes->size() / a.row_stride_bytes;
}

} // namespace

void OutlierMarker::mark_valid_range(StripeState & stripe,
                                     size_t        row,
                                     size_t        col_begin,
                                     size_t        col_end) const {
    BitMask &    mask = stripe.outlier_mask;
    const size_t base = stripe.local_row(row) * mask.cols;
    assert(col_begin <= col_end && col_end <= mask.cols);
    // Whole-word ORs over the row's bit range: the same final words as one BitMask::set per column.
    for (size_t idx = base + col_begin, end = base + col_end; idx < end;) {
        const size_t   lo   = idx % 64;
        const size_t   n    = std::min<size_t>(64 - lo, end - idx);
        const uint64_t bits = (n == 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1) << lo;
#if defined(GGML_GEMMINI_HAS_OPENMP)
#pragma omp atomic update
        mask.words[idx / 64] |= bits;
#else
        mask.words[idx / 64] |= bits;
#endif
        idx += n;
    }
}

std::pair<int32_t, int32_t> ResidualClipper::clip_with_residual(int32_t q, bool decompose) {
    constexpr int32_t qmax = config::GGML_GEMMINI_ACTIVATION_QMAX;
    constexpr int32_t qmin = config::GGML_GEMMINI_ACTIVATION_QMIN;
    const int32_t     sat  = q > qmax ? qmax : (q < qmin ? qmin : q);
    if (!decompose || q == sat)
        return {sat, 0};
    // u = sum_l L_l 2^(n l) with signed n-bit limbs. L0 is the balanced low digit, so it fits the
    // dense operand, and u - L0 is a multiple of 2^n: the RMD builders see lane 0 == 0 and stack
    // only L_{l>=1}.
    constexpr uint32_t radix = uint32_t{1} << GGML_GEMMINI_ACTIVATION_BITS;
    constexpr uint32_t half  = radix / 2;
    const int32_t      l0  = static_cast<int32_t>((static_cast<uint32_t>(q) + half) & (radix - 1)) -
                             static_cast<int32_t>(half);
    const int64_t      res = static_cast<int64_t>(q) - l0;
    // q in [INT32_MAX - half + 1, INT32_MAX] has a negative L0 that pushes u - L0 past INT32
    // (shift_q_i32 saturates to INT32_MAX, so this is reachable); saturation is exact there too.
    if (res > std::numeric_limits<int32_t>::max())
        return {sat, q - sat};
    return {l0, static_cast<int32_t>(res)};
}

bool StripeFolding::run(Meta &                           meta,
                        ExSIAState &                     state,
                        StripeState &                    stripe,
                        ggml_gemmini_args_t &            args,
                        size_t                           stripe_idx,
                        const std::vector<int32_t> &     stripe_q_wide,
                        const std::vector<int16_t> &     stripe_block_exp,
                        std::vector<int32_t> &           residual,
                        residual::TimedResidualCapture & rmd_builder) {
    const int16_t neg_inf = std::numeric_limits<int16_t>::min();
#if !EXSIA_VALIDATION
    (void)residual;
#endif
#if GGML_GEMMINI_RESIDUAL_METRICS
    if (args.evaluation_context)
        args.evaluation_context->main_stripe(
            stripe_idx, stripe.row_start, stripe.row_count(), args.J, args.K);
#endif
#if GGML_GEMMINI_SCALE_METRICS
    evaluation::observe_dense_scu(args, stripe_idx, stripe.row_count());
#endif
#if EXSIA_STAGE_PROFILE_ENABLED
    stripe.selected_positions = 0;
    stripe.residual_nnz       = 0;
#endif
#if GGML_GEMMINI_ENABLE_RMD
    rmd_builder.reset(stripe_idx,
                      stripe.row_start,
                      stripe.row_count(),
                      args.K,
                      args.J,
                      &stripe.outlier_mask.words,
                      state.K_padded);
#else
    (void)rmd_builder;
#endif

    if (stripe.e1 == neg_inf) {
        stripe.e_s               = 0;
        stripe.promote_top_block = false;
    } else if (stripe.e2 == neg_inf) {
        stripe.e_s               = stripe.e1;
        stripe.promote_top_block = false;
    } else {
        stripe.e_s               = args.folding_anchor == 3 && stripe.e3 != neg_inf
                                       ? stripe.e3 : stripe.e2;
        stripe.promote_top_block = true;
    }

    const int16_t theta_s = exp_to_theta(stripe.e_s, meta.rho);
    GGML_ASSERT(stripe_idx < meta.theta.size());
    meta.theta[stripe_idx] = theta_s;

    GGML_ASSERT(args.A.valid());
    GGML_ASSERT(state.B_size > 0);
    GGML_ASSERT(state.K_padded >= args.K);
    GGML_ASSERT(state.blocks_per_row == state.K_padded / state.B_size);
    GGML_ASSERT(stripe.row_start <= stripe.row_end && stripe.row_end <= args.I);
    GGML_ASSERT(stripe_q_wide.size() >= stripe.row_count() * state.K_padded);
    GGML_ASSERT(stripe_block_exp.size() >= stripe.row_count() * state.blocks_per_row);
    GGML_ASSERT(stripe.outlier_mask.rows == stripe.row_count());
    GGML_ASSERT(stripe.outlier_mask.cols >= state.K_padded);
    if (stripe.outlier_mask.rows != stripe.row_count() || stripe.outlier_mask.cols < state.K_padded)
        return false;

#if !(GGML_GEMMINI_ACT_QUANT_METRICS || EXSIA_STAGE_PROFILE_ENABLED)
    if (state.B_size <= 64 && dense_rows_writable(args.A, stripe.row_end, args.K))
        return run_blocks(state, stripe, args, stripe_q_wide, stripe_block_exp, residual, rmd_builder);
#endif

    // The slot owns this disjoint row range for both dense int8 and residual writes.
    for (size_t r = stripe.row_start; r < stripe.row_end; ++r) {
        GGML_ASSERT(r < args.I);
        const size_t local_row = stripe.local_row(r);

        for (size_t b = 0; b < state.blocks_per_row; ++b) {
            const size_t block_offset  = b * state.B_size;
            const size_t block_exp_idx = local_row * state.blocks_per_row + b;
            GGML_ASSERT(block_exp_idx < stripe_block_exp.size());

            const int16_t block_exp     = stripe_block_exp[block_exp_idx];
            const int16_t delta_theta_b = block_exp == neg_inf || stripe.e_s == neg_inf
                                              ? 0
                                              : static_cast<int16_t>(block_exp - stripe.e_s);
            // Select masked correction or whole-block correction once per block.
            const detail::PreparedShift shift(delta_theta_b);
            const auto fold_block = [&](auto selection) -> bool {
                for (size_t i = 0; i < state.B_size; ++i) {
                    const size_t  col        = block_offset + i;
                    const size_t  padded_idx = local_row * state.K_padded + col;
                    const int32_t q_shifted =
                        shift(stripe_q_wide[padded_idx]);
                    bool outlier;
                    if constexpr (decltype(selection)::value == 2)
                        outlier = col < args.K;
                    else
                        outlier = col < args.K && stripe.outlier_mask.is_set(local_row, col);
                    // Outliers and promoted top blocks keep their low limb in the dense operand and
                    // carry the higher limbs as residual; every other position saturates.
                    const auto [q8, residual_i32] = unit_clip_.clip_with_residual(
                        q_shifted, outlier && GGML_GEMMINI_ENABLE_RMD);

                    if (col < args.K)
                        if (!args.A.set(r, col, q8))
                            return false;

#if GGML_GEMMINI_ACT_QUANT_METRICS
                    if (col < args.K && args.evaluation_context)
                        args.evaluation_context->position(r, col, outlier, residual_i32 != 0);
#endif
#if EXSIA_STAGE_PROFILE_ENABLED
                    stripe.selected_positions += outlier;
                    stripe.residual_nnz += residual_i32 != 0;
#endif
#if EXSIA_VALIDATION
                    const size_t global_idx = r * state.K_padded + col;
                    GGML_ASSERT(global_idx < residual.size());
                    residual[global_idx] = residual_i32;
#endif

#if GGML_GEMMINI_ENABLE_RMD
                    if (outlier && residual_i32 != 0 &&
                        !rmd_builder.add_residual(local_row, col, residual_i32)) {
                        return false;
                    }
#endif
                }
                return true;
            };

            if (stripe.promote_top_block && block_exp > stripe.e_s) {
                // The valid columns are still marked, in one range write: the RMD bitmap builder
                // reads outlier_mask as its selection.
                if (block_offset < args.K)
                    unit_outlier_.mark_valid_range(
                        stripe, r, block_offset, std::min(block_offset + state.B_size, args.K));
                if (!fold_block(std::integral_constant<int, 2>{}))
                    return false;
            } else if (!fold_block(std::integral_constant<int, 1>{})) {
                return false;
            }
        }
    }

    return true;
}

// Same results as the per-column loop in run(). Each block is first folded as if no column
// were selected (shift + saturate, branch-free per element); only selected columns whose folded
// value leaves the activation range, visited through their mask bits, are split into limbs and
// emitted as residuals: a selected in-range column folds to the saturated code with no residual.
template <typename T>
bool StripeFolding::run_blocks_typed(const ExSIAState &               state,
                                     StripeState &                    stripe,
                                     ggml_gemmini_args_t &            args,
                                     const std::vector<int32_t> &     stripe_q_wide,
                                     const std::vector<int16_t> &     stripe_block_exp,
                                     std::vector<int32_t> &           residual,
                                     residual::TimedResidualCapture & rmd_builder) {
    const int16_t                     neg_inf   = std::numeric_limits<int16_t>::min();
    const size_t                      B         = state.B_size;
    const size_t                      K         = args.K;
    const size_t                      mask_cols = stripe.outlier_mask.cols;
    const QuantizedActivationBuffer & dense     = args.A;
#if !EXSIA_VALIDATION
    (void)residual;
#else
    GGML_ASSERT(residual.size() >= stripe.row_end * state.K_padded);
#endif
#if !GGML_GEMMINI_ENABLE_RMD
    (void)rmd_builder;
#endif
    T       codes[64];
    int32_t shifted[64];

    for (size_t r = stripe.row_start; r < stripe.row_end; ++r) {
        const size_t    local_row = stripe.local_row(r);
        const int32_t * q_row     = stripe_q_wide.data() + local_row * state.K_padded;
        const int16_t * exp_row   = stripe_block_exp.data() + local_row * state.blocks_per_row;
        uint8_t * dense_row = dense.bytes->data() + (dense.row_offset + r) * dense.row_stride_bytes;

        for (size_t b = 0; b < state.blocks_per_row; ++b) {
            const size_t block_offset = b * B;
            if (block_offset >= K)
                break;
            const size_t  valid     = std::min(B, K - block_offset);
            const int16_t block_exp = exp_row[b];
            const int16_t delta     = block_exp == neg_inf || stripe.e_s == neg_inf
                                          ? 0
                                          : static_cast<int16_t>(block_exp - stripe.e_s);
            fold_block_dense(codes, shifted, q_row + block_offset, valid, delta);

            uint64_t selected = 0;
            if (stripe.promote_top_block && block_exp > stripe.e_s) {
                // The RMD bitmap builder reads outlier_mask as its selection.
                unit_outlier_.mark_valid_range(stripe, r, block_offset, block_offset + valid);
                selected = valid == 64 ? ~uint64_t{0} : (uint64_t{1} << valid) - 1;
            } else {
                selected = load_mask_bits(
                    stripe.outlier_mask.words, local_row * mask_cols + block_offset, valid);
            }
#if GGML_GEMMINI_ENABLE_RMD
            while (selected != 0) {
                const size_t i = static_cast<size_t>(__builtin_ctzll(selected));
                selected &= selected - 1;
                if (shifted[i] == codes[i])
                    continue;
                const auto [low_limb, residual_i32] = unit_clip_.clip_with_residual(shifted[i], true);
                codes[i] = static_cast<T>(low_limb);
                if (residual_i32 == 0)
                    continue;
#if EXSIA_VALIDATION
                residual[r * state.K_padded + block_offset + i] = residual_i32;
#endif
                if (!rmd_builder.add_residual(local_row, block_offset + i, residual_i32))
                    return false;
            }
#else
            (void)selected;
#endif
            std::memcpy(dense_row + block_offset * sizeof(T), codes, valid * sizeof(T));
        }
    }
    return true;
}

bool StripeFolding::run_blocks(const ExSIAState &               state,
                               StripeState &                    stripe,
                               ggml_gemmini_args_t &            args,
                               const std::vector<int32_t> &     stripe_q_wide,
                               const std::vector<int16_t> &     stripe_block_exp,
                               std::vector<int32_t> &           residual,
                               residual::TimedResidualCapture & rmd_builder) {
    return args.A.bits == 16 ? run_blocks_typed<int16_t>(state, stripe, args, stripe_q_wide,
                                                         stripe_block_exp, residual, rmd_builder)
                             : run_blocks_typed<int8_t>(state, stripe, args, stripe_q_wide,
                                                        stripe_block_exp, residual, rmd_builder);
}

} // namespace ggml::gemmini::quants::act::exsia
