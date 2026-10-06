#include "folding.hpp"
#include "exsia_shift.hpp"
#include "ggml-gemmini-args.h"
#include "../../../ggml-gemmini-evaluation-observer.hpp"

#include <limits>
#include <utility>
#include <vector>

namespace ggml::gemmini::quants::act::exsia {
using namespace detail;

void OutlierMarker::mark_outlier(StripeState &   stripe,
                                 size_t          row,
                                 size_t          blk_idx,
                                 size_t          blk_size,
                                 const BitMask & d_mask) const {
    size_t       n         = blk_size;
    const size_t local_row = stripe.local_row(row);
    for (size_t i = 0; i < n; ++i) {
        size_t col = blk_idx * n + i;
        if (d_mask.is_set(0, i))
            stripe.outlier_mask.set(local_row, col);
    }
}

std::pair<int32_t, int32_t> ResidualClipper::clip_with_residual(int32_t q) {
    const int32_t qmax    = config::GGML_GEMMINI_ACTIVATION_QMAX;
    const int32_t qmin    = config::GGML_GEMMINI_ACTIVATION_QMIN;
    int32_t       clipped = q > qmax ? qmax : (q < qmin ? qmin : q);
    int32_t       res     = q - clipped;
    return {clipped, res};
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
        stripe.e_s               = stripe.e2;
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
            if (stripe.promote_top_block && block_exp == stripe.e1) {
                BitMask & block_inlier_mask = stripe.scratch.folding_inlier_mask;
                block_inlier_mask.clear_active_bits();

                for (size_t i = 0; i < state.B_size; ++i) {
                    const size_t col = block_offset + i;
                    if (col < args.K && !stripe.outlier_mask.is_set(local_row, col))
                        block_inlier_mask.set(0, i);
                }
                unit_outlier_.mark_outlier(stripe, r, b, state.B_size, block_inlier_mask);
            }

            for (size_t i = 0; i < state.B_size; ++i) {
                const size_t  col        = block_offset + i;
                const size_t  padded_idx = local_row * state.K_padded + col;
                const int32_t q_shifted =
                    detail::shift_q_i32(stripe_q_wide[padded_idx], delta_theta_b);
                const auto [q8, res] = unit_clip_.clip_with_residual(q_shifted);

                if (col < args.K)
                    if (!args.A.set(r, col, q8))
                        return false;

                const bool    outlier = col < args.K && stripe.outlier_mask.is_set(local_row, col);
                const int32_t residual_i32 = outlier ? res : 0;
#if GGML_GEMMINI_ACT_QUANT_METRICS
                if (col < args.K && args.evaluation_context)
                    args.evaluation_context->position(r, col, outlier, residual_i32 != 0);
#endif
#if EXSIA_STAGE_PROFILE_ENABLED
                stripe.selected_positions += outlier;
                stripe.residual_nnz += residual_i32 != 0;
#endif
                const size_t global_idx = r * state.K_padded + col;
                GGML_ASSERT(global_idx < residual.size());
                residual[global_idx] = residual_i32;

#if GGML_GEMMINI_ENABLE_RMD
                if (outlier && residual_i32 != 0 &&
                    !rmd_builder.add_residual(local_row, col, residual_i32)) {
                    return false;
                }
#endif
            }
        }
    }

    return true;
}

} // namespace ggml::gemmini::quants::act::exsia
