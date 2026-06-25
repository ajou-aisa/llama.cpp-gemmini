#include "ggml-impl.h"

#include "exsia.hpp"
#include "exsia_shift.hpp"
#include "types.hpp"

#include "ggml-gemmini-args.h"
#include "../../common/tensor_util.hpp"

#include <algorithm>
#include <cmath>
#include <variant>

namespace ggml::gemmini::quants::act::exsia
{
    namespace
    {
        bool checked_mul_size(size_t lhs, size_t rhs, size_t &out)
        {
            if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
                return false;

            out = lhs * rhs;
            return true;
        }

        bool checked_add_size(size_t lhs, size_t rhs, size_t &out)
        {
            if (lhs > std::numeric_limits<size_t>::max() - rhs)
                return false;

            out = lhs + rhs;
            return true;
        }

        bool checked_round_up_multiple(size_t value, size_t multiple, size_t &out)
        {
            if (multiple == 0)
                return false;

            size_t adjusted = 0;
            if (!checked_add_size(value, multiple - 1, adjusted))
                return false;

            out = (adjusted / multiple) * multiple;
            return true;
        }
    }

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

        __int128_t S = 0;
        __int128_t SS = 0;
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
                SS += static_cast<__int128_t>(tmp) * tmp;
            }
        }
        return {q, S, SS};
    }

    bool SigmaDetector::detect_3sigma(int32_t q, __int128_t S, __int128_t SS, size_t N)
    {
        if (N == 0)
            return false;

        const __int128_t n = static_cast<__int128_t>(N);
        const __int128_t centered = n * q - S;
        return centered * centered > 9 * (n * SS - S * S);
    }

    std::pair<int8_t, int32_t> ResidualClipper::clip_with_residual(int32_t q)
    {
        int32_t q8 = q > 127 ? 127 : (q < -128 ? -128 : q);
        int32_t res = q - q8;
        return {static_cast<int8_t>(q8), res};
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

        if (!top1_exp_mask.resize(1, blk_size))
            return false;
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
        if (!int_outlier_mask.resize(1, blk_size))
            return false;
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
        {
            if (!stripe.outlier_mask.resize(args.I, state.K_padded))
                return false;
        }

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
                    if (!block_inlier_mask.resize(1, state.B_size))
                        return false;
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
                    const int32_t q_shifted = detail::shift_q_i32(state.q_wide[padded_idx], delta_theta_b);

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
        if (args.I == 0 || args.K == 0 ||
            !checked_round_up_multiple(args.K, state_.B_size, state_.K_padded))
        {
            return false;
        }

        state_.blocks_per_row = state_.K_padded / state_.B_size;

        meta.theta.clear();
        meta.outliers.clear();

        size_t rows_per_stripe = args.I;
        if (args.tile_I > 0 && !checked_mul_size(args.tile_I, DIM, rows_per_stripe))
            return false;

        if (rows_per_stripe == 0)
            return false;

        size_t stripe_round_input = 0;
        if (!checked_add_size(args.I, rows_per_stripe - 1, stripe_round_input))
            return false;

        const size_t num_stripes = stripe_round_input / rows_per_stripe;

        const float *src_data = ggml::gemmini::activation_data(A);
        if (!src_data)
            return false;

        size_t padded_elem_count = 0;
        size_t block_exp_count = 0;
        if (!checked_mul_size(args.I, state_.K_padded, padded_elem_count) ||
            !checked_mul_size(args.I, state_.blocks_per_row, block_exp_count))
        {
            return false;
        }

        state_.q_wide.assign(padded_elem_count, 0);
        state_.x_f32.assign(padded_elem_count, 0.f);
        state_.block_exp.assign(block_exp_count, std::numeric_limits<int16_t>::min());
        state_.residual.assign(padded_elem_count, 0);

        state_.stripe.assign(num_stripes, StripeState{});
        for (size_t s = 0; s < num_stripes; ++s)
        {
            StripeState &stripe = state_.stripe[s];
            stripe.row_start = s * rows_per_stripe;
            stripe.row_end = std::min((s + 1) * rows_per_stripe, args.I);
            if (!stripe.outlier_mask.resize(args.I, state_.K_padded))
                return false;
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
                if (!local_.run(meta, state_, state_.stripe[stripe_idx], block_x, r, b))
                    return false;
            }
        }

        int8_t *dst = reinterpret_cast<int8_t *>(args.A);
        int32_t *residual_ptr = state_.residual.data();
        for (size_t s = 0; s < num_stripes; ++s)
        {
            if (!folding_.run(meta, state_, state_.stripe[s], args, s, dst, residual_ptr))
                return false;
        }

        return true;
    }

    bool dequantize_activation(
        float *dst,
        size_t dst_row_stride,
        size_t dst_col_stride,
        size_t rows,
        size_t cols,
        const ggml_gemmini_args_t &args)
    {
        const int8_t *src = reinterpret_cast<const int8_t *>(args.A);
        if (!src || !dst || args.I == 0 || args.K == 0 ||
            dst_row_stride == 0 || dst_col_stride == 0 ||
            rows == 0 || cols == 0)
        {
            return false;
        }

        const auto *meta_ptr = std::get_if<Meta>(&args.act_quant.storage());
        if (!meta_ptr)
        {
            return false;
        }
        const Meta &meta = *meta_ptr;

        size_t rows_per_stripe = args.I;
        if (args.tile_I > 0 && !checked_mul_size(args.tile_I, DIM, rows_per_stripe))
        {
            return false;
        }
        if (rows_per_stripe == 0)
        {
            return false;
        }

        if (args.sA != 0 && args.sA != args.K)
        {
            return false;
        }

        const size_t src_row_stride = args.K;
        const size_t row_count = std::min(rows, args.I);
        const size_t col_count = std::min(cols, args.K);
        const size_t max_size = std::numeric_limits<size_t>::max();
        if (row_count != 0 && col_count > max_size / row_count)
        {
            return false;
        }

        std::vector<int32_t> residuals(row_count * col_count, 0);
        for (const auto &outlier : meta.outliers)
        {
            if (outlier.row < 0 || outlier.col < 0)
            {
                continue;
            }

            const size_t row = static_cast<size_t>(outlier.row);
            const size_t col = static_cast<size_t>(outlier.col);
            if (row < row_count && col < col_count)
            {
                residuals[row * col_count + col] += outlier.residual;
            }
        }

        const int16_t invalid_theta = std::numeric_limits<int16_t>::min();
        for (size_t row = 0; row < row_count; ++row)
        {
            const size_t stripe_idx = row / rows_per_stripe;
            const int16_t theta = meta.resolve_stripe_theta(static_cast<int>(stripe_idx));
            if (theta == invalid_theta)
            {
                return false;
            }

            for (size_t col = 0; col < col_count; ++col)
            {
                if ((row != 0 && src_row_stride > max_size / row) ||
                    (row != 0 && dst_row_stride > max_size / row) ||
                    (col != 0 && dst_col_stride > max_size / col))
                {
                    return false;
                }

                const size_t src_row_offset = row * src_row_stride;
                if (src_row_offset > max_size - col)
                {
                    return false;
                }

                const size_t src_idx = src_row_offset + col;
                const size_t dst_row_offset = row * dst_row_stride;
                const size_t dst_col_offset = col * dst_col_stride;
                if (dst_row_offset > max_size - dst_col_offset)
                {
                    return false;
                }

                const int32_t q_int =
                    static_cast<int32_t>(src[src_idx]) +
                    residuals[row * col_count + col];
                dst[dst_row_offset + dst_col_offset] =
                    std::ldexp(static_cast<float>(q_int), theta);
            }
        }

        return true;
    }

}
