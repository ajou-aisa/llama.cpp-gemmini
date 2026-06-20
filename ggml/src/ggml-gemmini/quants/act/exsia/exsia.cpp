#include "ggml-impl.h"

#include "exsia.hpp"
#include "types.hpp"

#include "ggml-gemmini-args.h"
#include "../../common/tensor_util.hpp"

#include <algorithm>
#include <cmath>

namespace ggml::gemmini::quants::act::exsia
{
    int16_t ExpScanner::unbiased_exp(const float &x)
    {
        if (x == 0.f || !std::isfinite(x))
            return std::numeric_limits<int16_t>::min();

        return static_cast<int16_t>(std::ilogb(std::abs(x)));
    }

    void ExpScanner::scan_top2_exp(const std::vector<float> &x,
                                   BlockState &blk)
    {
        const size_t n = x.size();
        blk.blk_size = n;
        for (size_t i = 0; i < n; ++i)
        {
            int16_t exp = unbiased_exp(x[i]);
            blk.x.push_back(x[i]);
            blk.e.push_back(exp);
            if (exp > blk.e1)
            {
                blk.e2 = blk.e1;
                blk.e1 = exp;
            }
            else if (exp < blk.e1 && exp > blk.e2)
                blk.e2 = exp;
        }
    }

    void ExpScanner::update_block_top2_exp(const BitMask &mask,
                                           size_t row,
                                           size_t blk_idx,
                                           BlockState &blk)
    {
        blk.e1 = std::numeric_limits<int16_t>::min();
        blk.e2 = std::numeric_limits<int16_t>::min();

        const size_t n = blk.blk_size;
        for (size_t i = 0; i < n; ++i)
        {
            size_t col = blk_idx * n + i;
            if (mask.is_set(row, col))
                continue;

            const int16_t exp = blk.e[i];
            if (exp > blk.e1)
            {
                blk.e2 = blk.e1;
                blk.e1 = exp;
            }
            else if (exp < blk.e1 && exp > blk.e2)
                blk.e2 = exp;
        }
    }

    void ExpScanner::update_stripe_top2_exp(StripeState &stripe, int16_t exp)
    {
        if (exp > stripe.e1)
        {
            stripe.e2 = stripe.e1;
            stripe.e1 = exp;
        }
        else if (exp < stripe.e1 && exp > stripe.e2)
            stripe.e2 = exp;
    }

    void OutlierMarker::mark_outlier(StripeState &stripe,
                                     size_t row,
                                     size_t blk_idx,
                                     size_t blk_size,
                                     const BitMask &d_mask) const
    {
        size_t n = blk_size;
        for (size_t i = 0; i < n; ++i)
        {
            size_t col = blk_idx * n + i;
            if (d_mask.is_set(0, i))
                stripe.outlier_mask.set(row, col);
        }
    }

    std::vector<int32_t> WideQuantizer::quantize_block(const std::vector<float> &x,
                                                       int16_t theta_b)
    {
        std::vector<int32_t> q;
        q.reserve(x.size());

        if (theta_b == std::numeric_limits<int16_t>::min())
        {
            q.resize(x.size(), 0);
            return q;
        }

        for (auto &e : x)
            q.push_back(static_cast<int32_t>(std::lrint(std::ldexp(e, -theta_b))));

        return q;
    }

    std::tuple<std::vector<int32_t>, int64_t, int64_t>
    WideQuantizer::quantize_block(const std::vector<float> &x,
                                  size_t row,
                                  size_t col_offset,
                                  const BitMask &mask,
                                  int16_t theta_b)
    {
        std::vector<int32_t> q;
        size_t n = x.size();
        q.reserve(n);

        int64_t S = 0;
        int64_t SS = 0;
        const bool use_mask = mask.rows != 0 && mask.cols != 0;
        const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
        for (size_t i = 0; i < n; ++i)
        {
            size_t col = col_offset + i;
            int32_t tmp = null_theta ? 0 : std::lrint(std::ldexp(x[i], -theta_b));
            q.push_back(static_cast<int32_t>(tmp));
            if (!use_mask || !mask.is_set(row, col))
            {
                S += tmp;
                SS += static_cast<int64_t>(tmp) * tmp;
            }
        }
        return {q, S, SS};
    }

    bool SigmaDetector::detect_3sigma(int32_t q, int64_t S, int64_t SS, size_t N)
    {
        if (N == 0)
            return false;

        const int64_t n = static_cast<int64_t>(N);
        const int64_t centered = n * q - S;
        return centered * centered > 9 * (n * SS - S * S);
    }

    std::pair<int8_t, int32_t> ResidualClipper::clip_with_residual(int32_t q)
    {
        int32_t q8 = q > 127 ? 127 : (q < -128 ? -128 : q);
        int32_t res = q - q8;
        return {static_cast<int8_t>(q8), res};
    }

    namespace
    {
        static int32_t round_shift_right_i32(
            int32_t q,
            int shift)
        {
            if (shift <= 0)
                return q;

            const int64_t x = q;
            const int64_t offset =
                int64_t{1} << (shift - 1);

            if (x >= 0)
                return static_cast<int32_t>(
                    (x + offset) >> shift);

            return static_cast<int32_t>(
                -(((-x) + offset) >> shift));
        }

        int32_t shift_q_i32(int32_t q, int16_t delta_theta)
        {
            if (delta_theta > 0)
            {
                const int shift = std::min<int>(delta_theta, 31);
                const int64_t shifted = static_cast<int64_t>(q) << shift;
                return static_cast<int32_t>(std::clamp<int64_t>(
                    shifted,
                    std::numeric_limits<int32_t>::min(),
                    std::numeric_limits<int32_t>::max()));
            }

            if (delta_theta < 0)
            {
                const int shift = std::min<int>(-delta_theta, 31);
                return q >> round_shift_right_i32(q, shift);
            }

            return q;
        }
    }

    bool LocalStage::run(
        Meta &meta,
        ExSIAState &state,
        StripeState &stripe,
        const std::vector<float> &x,
        size_t row,
        size_t blk_idx)
    {
        const size_t blk_size = state.B_size;
        const size_t base = row * state.K_padded + blk_idx * blk_size;

        GGML_ASSERT(x.size() == blk_size);
        BlockState blk;

        // Step 1: scan block exponents and identify top-2 exponent buckets.
        unit_exp_.scan_top2_exp(x, blk);

        // Step 2: promote elements in the top-1 exponent bucket into the mask.
        const int16_t neg_inf = std::numeric_limits<int16_t>::min();
        BitMask top1_exp_mask;

        top1_exp_mask.resize(1, blk_size);
        bool has_second_bucket = (blk.e2 != neg_inf);

        if (has_second_bucket)
        {
            for (size_t i = 0; i < blk_size; ++i)
            {
                const size_t col = blk_idx * blk_size + i;
                if (col < state.K_logical && blk.e[i] != neg_inf && blk.e[i] == blk.e1)
                    top1_exp_mask.set(0, i);
            }
        }

        GGML_ASSERT(stripe.outlier_mask.rows > row);
        GGML_ASSERT(stripe.outlier_mask.cols >= state.K_padded);
        unit_outlier_.mark_outlier(stripe, row, blk_idx, blk_size, top1_exp_mask);

        // Step 3.1: set a provisional block scale from the second exponent bucket.
        const int16_t theta_pre = has_second_bucket ? static_cast<int16_t>(blk.e2 - meta.rho)
                                                    : static_cast<int16_t>(blk.e1 - meta.rho);

        // Step 3.2: wide-quantize remaining inliers and collect local statistics.
        auto [q_tmp, S, SS] = unit_quant_.quantize_block(blk.x,
                                                         row,
                                                         blk_idx * blk_size,
                                                         stripe.outlier_mask,
                                                         theta_pre);

        // Step 3.3: detect integer-domain outliers once for this block.
        BitMask int_outlier_mask;
        int_outlier_mask.resize(1, blk_size);
        size_t unmasked_count = 0;
        for (size_t i = 0; i < blk_size; ++i)
        {
            const size_t col = blk_idx * blk_size + i;
            if (col < state.K_logical && !stripe.outlier_mask.is_set(row, col))
                ++unmasked_count;
        }

        bool has_int_outlier = false;
        for (size_t i = 0; i < blk_size; ++i)
        {
            const size_t col = blk_idx * blk_size + i;
            if (col >= state.K_logical || stripe.outlier_mask.is_set(row, col))
                continue;

            if (unit_sigma_.detect_3sigma(q_tmp[i], S, SS, unmasked_count))
            {
                int_outlier_mask.set(0, i);
                has_int_outlier = true;
            }
        }

        std::vector<int32_t> q_final;
        if (!has_int_outlier)
        {
            // Step 4.1: reuse the pre-quantized block if no extra outlier is found.
            blk.e_b = has_second_bucket ? blk.e2 : blk.e1;
            blk.theta_b = theta_pre;
            q_final = std::move(q_tmp);
        }
        else
        {
            // Step 4.2: promote integer outliers and update the inlier range.
            unit_outlier_.mark_outlier(stripe, row, blk_idx, blk_size, int_outlier_mask);
            unit_exp_.update_block_top2_exp(stripe.outlier_mask, row, blk_idx, blk);

            // Step 4.3: re-quantize the block using the updated exponent.
            blk.e_b = blk.e1;
            blk.theta_b = blk.e_b == neg_inf
                              ? neg_inf
                              : static_cast<int16_t>(blk.e_b - meta.rho);
            q_final = unit_quant_.quantize_block(blk.x, blk.theta_b);
        }

        // Step 5: store the local result for stripe folding.
        GGML_ASSERT(state.q_wide.size() >= base + blk_size);
        for (size_t i = 0; i < blk_size; ++i)
            state.q_wide[base + i] = q_final[i];

        const size_t block_exp_idx = row * state.blocks_per_row + blk_idx;
        GGML_ASSERT(state.block_exp.size() > block_exp_idx);
        state.block_exp[block_exp_idx] = blk.e_b;

        unit_exp_.update_stripe_top2_exp(stripe, blk.e_b);

        return true;
    }

    bool StripeFolding::run(Meta &meta,
                            ExSIAState &state,
                            StripeState &stripe,
                            ggml_gemmini_args_t &args,
                            size_t stripe_idx,
                            int8_t *dst,
                            int32_t *residual)
    {
        const int16_t neg_inf = std::numeric_limits<int16_t>::min();

        // Step 1: determine stripe exponent from top-2 block exponents.
        if (stripe.e2 == neg_inf)
        {
            stripe.e_s = stripe.e1;
            stripe.promote_top_block = false;
        }
        else
        {
            stripe.e_s = stripe.e2;
            stripe.promote_top_block = true;
        }

        // Step 2: store the per-stripe dequantization exponent.
        const int16_t theta_s = stripe.e_s == neg_inf
                                    ? neg_inf
                                    : static_cast<int16_t>(stripe.e_s - meta.rho);
        if (meta.theta.size() <= stripe_idx)
            meta.theta.resize(stripe_idx + 1, std::numeric_limits<int16_t>::min());
        meta.theta[stripe_idx] = theta_s;

        GGML_ASSERT(dst != nullptr);
        GGML_ASSERT(residual != nullptr);
        GGML_ASSERT(state.B_size > 0);
        GGML_ASSERT(state.K_padded >= args.K);
        GGML_ASSERT(state.blocks_per_row == state.K_padded / state.B_size);
        GGML_ASSERT(state.q_wide.size() >= args.I * state.K_padded);
        GGML_ASSERT(state.x_f32.size() >= args.I * state.K_padded);
        GGML_ASSERT(state.residual.size() >= args.I * state.K_padded);

        if (stripe.outlier_mask.rows == 0 || stripe.outlier_mask.cols == 0)
            stripe.outlier_mask.resize(args.I, state.K_padded);

        for (size_t r = stripe.row_start; r < stripe.row_end; ++r)
        {
            GGML_ASSERT(r < args.I);

            for (size_t b = 0; b < state.blocks_per_row; ++b)
            {
                const size_t block_offset = b * state.B_size;
                const size_t block_exp_idx = r * state.blocks_per_row + b;
                GGML_ASSERT(block_exp_idx < state.block_exp.size());

                // Step 3.1: compute the block-to-stripe folding shift.
                const int16_t block_exp = state.block_exp[block_exp_idx];
                const int16_t delta_theta_b = block_exp == neg_inf || stripe.e_s == neg_inf
                                                  ? 0
                                                  : static_cast<int16_t>(block_exp - stripe.e_s);

                // Step 3.2: promote inliers of the top-exponent block.
                if (stripe.promote_top_block && block_exp == stripe.e1)
                {
                    BitMask block_inlier_mask;
                    block_inlier_mask.resize(1, state.B_size);
                    for (size_t i = 0; i < state.B_size; ++i)
                    {
                        const size_t col = block_offset + i;
                        if (col < args.K && !stripe.outlier_mask.is_set(r, col))
                            block_inlier_mask.set(0, i);
                    }
                    unit_outlier_.mark_outlier(stripe, r, b, state.B_size, block_inlier_mask);
                }

                for (size_t i = 0; i < state.B_size; ++i)
                {
                    const size_t col = block_offset + i;
                    const size_t padded_idx = r * state.K_padded + col;

                    // Step 4.1: shift the wide integer to the stripe scale.
                    const int32_t q_shifted = shift_q_i32(state.q_wide[padded_idx], delta_theta_b);

                    // Step 4.2: clip once and compute the residual in the stripe integer domain.
                    const auto [q8, res] = unit_clip_.clip_with_residual(q_shifted);

                    // Step 4.3: store dense int8 activation.
                    if (col < args.K)
                        dst[r * args.K + col] = q8;

                    // Step 4.4: store residual only for marked outliers.
                    // The outlier index is the sparse index of residual correction.
                    const bool outlier = col < args.K && stripe.outlier_mask.is_set(r, col);

                    const int32_t residual_i32 = outlier ? res : 0;

                    state.residual[padded_idx] = residual_i32;
                    residual[padded_idx] = residual_i32;

                    if (outlier)
                    {
                        meta.outliers.push_back({
                            static_cast<int>(r),
                            static_cast<int>(col),
                            residual_i32,
                        });
                    }
                }
            }
        }

        return true;
    }

    bool ExSIA::run(
        Meta &meta,
        const ggml_tensor *A,
        ggml_gemmini_args_t &args)
    {
        state_.B_size = BLOCK_SIZE;
        state_.K_logical = args.K;
        state_.K_padded = ((args.K + state_.B_size - 1) / state_.B_size) * state_.B_size;
        state_.blocks_per_row = state_.K_padded / state_.B_size;

        meta.theta.clear();
        meta.outliers.clear();

        const size_t rows_per_stripe = args.tile_I > 0 ? args.tile_I * DIM : args.I;
        const size_t num_stripes = (args.I + rows_per_stripe - 1) / rows_per_stripe;

        const float *src_data = ggml::gemmini::activation_data(A);
        GGML_ASSERT(src_data != nullptr);

        state_.q_wide.assign(args.I * state_.K_padded, 0);
        state_.x_f32.assign(args.I * state_.K_padded, 0.f);
        state_.block_exp.assign(args.I * state_.blocks_per_row, std::numeric_limits<int16_t>::min());
        state_.residual.assign(args.I * state_.K_padded, 0);

        state_.stripe.assign(num_stripes, StripeState{});
        for (size_t s = 0; s < num_stripes; ++s)
        {
            StripeState &stripe = state_.stripe[s];
            stripe.row_start = s * rows_per_stripe;
            stripe.row_end = std::min((s + 1) * rows_per_stripe, args.I);
            stripe.outlier_mask.resize(args.I, state_.K_padded);
        }

        for (size_t r = 0; r < args.I; ++r)
        {
            for (size_t b = 0; b < state_.blocks_per_row; ++b)
            {
                std::vector<float> block_x;
                block_x.reserve(state_.B_size);

                const size_t col_offset = b * state_.B_size;
                for (size_t i = 0; i < state_.B_size; ++i)
                {
                    const size_t col = col_offset + i;
                    const float value = col < args.K ? src_data[r * args.K + col] : 0.f;
                    block_x.push_back(value);
                    state_.x_f32[r * state_.K_padded + col] = value;
                }

                const size_t stripe_idx = r / rows_per_stripe;
                local_.run(meta, state_, state_.stripe[stripe_idx], block_x, r, b);
            }
        }

        int8_t *dst = reinterpret_cast<int8_t *>(args.A);
        int32_t *residual_ptr = state_.residual.data();
        for (size_t s = 0; s < num_stripes; ++s)
            folding_.run(meta, state_, state_.stripe[s], args, s, dst, residual_ptr);

        return true;
    }

    void dequantize(
        const ggml_gemmini_args_t &args,
        size_t k_offset,
        size_t block_k,
        const int32_t *acc32,
        size_t acc_stride)
    {
        (void)args;
        (void)k_offset;
        (void)block_k;
        (void)acc32;
        (void)acc_stride;

        // Step 1: Read per-stripe theta from meta.resolve_stripe_theta(stripe_idx)
        //         and rho from meta.rho.
        // Step 2: For each accumulator element, apply inverse activation scaling.
        // Step 3: Apply sparse residual correction using meta.outliers.
        //         Each outlier entry stores (row, col, residual_i32), where
        //         residual_i32 = q_shifted - q_i8 in the folded stripe integer domain.
        // TODO: implement dequantize / residual-correction algorithm.
    }
}
