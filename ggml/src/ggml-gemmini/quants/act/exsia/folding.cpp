#include "folding.hpp"
#include "exsia_shift.hpp"
#include "exsia_float.hpp"
#include "ggml-gemmini-args.h"
#include "../../../ggml-gemmini-evaluation-observer.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace ggml::gemmini::quants::act::exsia {
using namespace detail;

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
                        residual::TimedResidualCapture & rmd_builder,
                        const float * original) {
    const bool direct = original != nullptr;
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
    GGML_ASSERT(direct || stripe_q_wide.size() >= stripe.row_count() * state.K_padded);
    GGML_ASSERT(stripe_block_exp.size() >= stripe.row_count() * state.blocks_per_row);
    GGML_ASSERT(stripe.outlier_mask.rows == stripe.row_count());
    GGML_ASSERT(stripe.outlier_mask.cols >= state.K_padded);
    if (stripe.outlier_mask.rows != stripe.row_count() || stripe.outlier_mask.cols < state.K_padded)
        return false;

    // The slot owns this disjoint row range for both dense int8 and residual writes.
    for (size_t r = stripe.row_start; r < stripe.row_end; ++r) {
        GGML_ASSERT(r < args.I);
        const size_t local_row = stripe.local_row(r);

        for (size_t b = 0; b < state.blocks_per_row; ++b) {
            const size_t block_offset  = b * state.B_size;
            const size_t block_exp_idx = local_row * state.blocks_per_row + b;
            GGML_ASSERT(block_exp_idx < stripe_block_exp.size());

            const int16_t block_exp     = stripe_block_exp[block_exp_idx];
            const int16_t delta_theta_b = direct || block_exp == neg_inf || stripe.e_s == neg_inf
                                              ? 0
                                              : static_cast<int16_t>(block_exp - stripe.e_s);
            // Select masked correction or whole-block correction once per block.
            const detail::PreparedShift shift(delta_theta_b);
            const auto fold_block = [&](auto selection) -> bool {
                for (size_t i = 0; i < state.B_size; ++i) {
                    const size_t  col        = block_offset + i;
                    const size_t  padded_idx = local_row * state.K_padded + col;
                    const int32_t q_shifted =
                        direct ? (col < args.K
                                      ? quantize_f32_stripe_rn(original[r * args.K + col], theta_s) : 0)
                               : shift(stripe_q_wide[padded_idx]);
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

} // namespace ggml::gemmini::quants::act::exsia
