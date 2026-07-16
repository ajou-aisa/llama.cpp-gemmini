#include "exsia.hpp"
#include "exsia_shift.hpp"
#include "types.hpp"

#include "ggml-gemmini-args.h"
#include "../../common/tensor_util.hpp"

#include <gemmini/cycle_reader.hpp>
#include <gemmini/log.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <variant>

namespace ggml::gemmini::quants::act::exsia
{
    static_assert(GGML_GEMMINI_EXSIA_SIGMA > 0, "GGML_GEMMINI_EXSIA_SIGMA must be positive");

    namespace
    {
        template <typename T>
        void release_vector(std::vector<T> &values)
        {
            std::vector<T>().swap(values);
        }

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

        static inline int16_t exp_to_theta(int16_t e, int16_t rho)
        {
            const int16_t neg_inf = std::numeric_limits<int16_t>::min();

            if (e == neg_inf)
                return neg_inf;

            return static_cast<int16_t>(static_cast<int>(e) - static_cast<int>(rho));
        }

        static inline int32_t quantize_to_i32(float x, int16_t theta)
        {
            const int16_t neg_inf = std::numeric_limits<int16_t>::min();

            if (theta == neg_inf || !std::isfinite(x))
                return 0;

            const double scaled = std::ldexp(static_cast<double>(x), -static_cast<int>(theta));

            if (!std::isfinite(scaled))
                return scaled < 0.0
                           ? std::numeric_limits<int32_t>::min()
                           : std::numeric_limits<int32_t>::max();

            const double min_i32 = static_cast<double>(std::numeric_limits<int32_t>::min());
            const double max_i32 = static_cast<double>(std::numeric_limits<int32_t>::max());

            if (scaled <= min_i32)
                return std::numeric_limits<int32_t>::min();
            if (scaled >= max_i32)
                return std::numeric_limits<int32_t>::max();

            return static_cast<int32_t>(std::lrint(scaled));
        }

        static inline int64_t magnitude_i32(int32_t value) noexcept
        {
            const int64_t widened = static_cast<int64_t>(value);
            return widened < 0 ? -widened : widened;
        }
    }

#if CYCLE_DETAIL
#define EXSIA_CYCLE_READ() ggml::gemmini::cycle::read()
#define EXSIA_LOG_CYCLE(op, start, end)                                          \
    do                                                                          \
    {                                                                           \
        const char *path = std::getenv("GGML_GEMMINI_CYCLE_DETAIL_LOG");         \
        ggml::gemmini::log::cycle(                                              \
            ggml::gemmini::log::file(path && path[0] ? path : "log/exsia-cycle-detail.jsonl"), \
            "exsia",                                                           \
            op,                                                               \
            start,                                                             \
            end);                                                              \
} while (false)
#else
#define EXSIA_CYCLE_READ() static_cast<uint64_t>(0)
#define EXSIA_LOG_CYCLE(op, start, end)                                          \
    do                                                                          \
    {                                                                           \
        (void)sizeof(op);                                                        \
        (void)sizeof(start);                                                     \
        (void)sizeof(end);                                                       \
    } while (false)
#endif

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
        GGML_ASSERT(blk.e.size() >= n);
        blk.reset();
        blk.blk_size = n;
        for (size_t i = 0; i < n; ++i)
        {
            int16_t exp = unbiased_exp(x[i]);
            blk.e[i] = exp;
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
        const size_t local_row = stripe.local_row(row);
        for (size_t i = 0; i < n; ++i)
        {
            size_t col = blk_idx * n + i;
            if (d_mask.is_set(0, i))
                stripe.outlier_mask.set(local_row, col);
        }
    }

    std::vector<int32_t> WideQuantizer::quantize_block(const std::vector<float> &x,
                                                       int16_t theta_b)
    {
        std::vector<int32_t> q(x.size());
        quantize_block(x, theta_b, q);
        return q;
    }

    void WideQuantizer::quantize_block(const std::vector<float> &x,
                                       int16_t theta_b,
                                       std::vector<int32_t> &q) const
    {
        GGML_ASSERT(q.size() >= x.size());
        const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
        for (size_t i = 0; i < x.size(); ++i)
            q[i] = null_theta ? 0 : quantize_to_i32(x[i], theta_b);
    }

    std::tuple<std::vector<int32_t>, __int128_t, __int128_t>
    WideQuantizer::quantize_block(const std::vector<float> &x,
                                  size_t row,
                                  size_t col_offset,
                                  const BitMask &mask,
                                  int16_t theta_b)
    {
        std::vector<int32_t> q(x.size());
        __int128_t S = 0;
        __int128_t SS = 0;
        quantize_block(x, row, col_offset, mask, theta_b, q, S, SS);
        return {q, S, SS};
    }

    void WideQuantizer::quantize_block(const std::vector<float> &x,
                                       size_t row,
                                       size_t col_offset,
                                       const BitMask &mask,
                                       int16_t theta_b,
                                       std::vector<int32_t> &q,
                                       __int128_t &S,
                                       __int128_t &SS) const
    {
        const size_t n = x.size();
        GGML_ASSERT(q.size() >= n);

        S = 0;
        SS = 0;
        const bool use_mask = mask.rows != 0 && mask.cols != 0;
        const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
        for (size_t i = 0; i < n; ++i)
        {
            size_t col = col_offset + i;
            const int32_t tmp = null_theta ? 0 : quantize_to_i32(x[i], theta_b);
            q[i] = tmp;
            if (!use_mask || !mask.is_set(row, col))
            {
                const __int128_t magnitude = static_cast<__int128_t>(magnitude_i32(tmp));
                S += magnitude;
                SS += magnitude * magnitude;
            }
        }
    }

    bool SigmaDetector::detect_sigma(int32_t q, __int128_t S, __int128_t SS, size_t N)
    {
        if (N == 0)
            return false;

        const __int128_t n = static_cast<__int128_t>(N);
        const __int128_t sigma = GGML_GEMMINI_EXSIA_SIGMA;
        const __int128_t centered = n * static_cast<__int128_t>(magnitude_i32(q)) - S;
        const __int128_t variance_numer = n * SS - S * S;
        if (centered <= 0 || variance_numer <= 0)
            return false;

        return centered * centered > sigma * sigma * variance_numer;
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
        size_t blk_idx,
        std::vector<int32_t> &stripe_q_wide,
        std::vector<int16_t> &stripe_block_exp,
        uint64_t &cycle_delta)
    {
        const size_t blk_size = state.B_size;
        const size_t local_row = stripe.local_row(row);
        const size_t base = local_row * state.K_padded + blk_idx * blk_size;
        cycle_delta = 0;

        GGML_ASSERT(x.size() == blk_size);
        StripeScratch &scratch = stripe.scratch;
        BlockState &blk = scratch.block;
        const int16_t neg_inf = std::numeric_limits<int16_t>::min();
        bool has_second_bucket = false;
        std::vector<int32_t> &q_tmp = scratch.q_tmp;
        std::vector<int32_t> &q_final = scratch.q_final;
        __int128_t S = 0;
        __int128_t SS = 0;
        size_t unmasked_count = 0;
        bool has_int_outlier = false;
        BitMask &top1_exp_mask = scratch.top1_exp_mask;
        BitMask &int_outlier_mask = scratch.int_outlier_mask;
        int16_t e_pre = neg_inf;
        int16_t theta_pre = neg_inf;

        {
            // Step 1: scan block exponents and identify top-2 distinct exponent buckets.
            const uint64_t cycle_start = EXSIA_CYCLE_READ();
            unit_exp_.scan_top2_exp(x, blk);
            has_second_bucket = (blk.e2 != neg_inf);
            const uint64_t cycle_end = EXSIA_CYCLE_READ();
            EXSIA_LOG_CYCLE("exsia.Local.Stage.1.scan_top2_exp", cycle_start, cycle_end);
            cycle_delta += cycle_end >= cycle_start ? cycle_end - cycle_start : 0;
        }

        {
            // Step 2: mark the top-1 exponent bucket.
            // Step 2.1: single-bucket block; disable exponent preselection.
            // Step 2.2: multi-bucket block; remove the range-dominating bucket.
            const uint64_t cycle_start = EXSIA_CYCLE_READ();
            if (has_second_bucket)
            {
                top1_exp_mask.clear_active_bits();

                for (size_t i = 0; i < blk_size; ++i)
                {
                    const size_t col = blk_idx * blk_size + i;
                    if (col < state.K_logical && blk.e[i] != neg_inf && blk.e[i] == blk.e1)
                        top1_exp_mask.set(0, i);
                }
                unit_outlier_.mark_outlier(stripe, row, blk_idx, blk_size, top1_exp_mask);
            }
            const uint64_t cycle_end = EXSIA_CYCLE_READ();
            EXSIA_LOG_CYCLE("exsia.Local.Stage.2.mark_top1_exp_bucket", cycle_start, cycle_end);
            cycle_delta += cycle_end >= cycle_start ? cycle_end - cycle_start : 0;
        }

        int_outlier_mask.clear_active_bits();
        GGML_ASSERT(stripe.outlier_mask.rows > local_row);
        GGML_ASSERT(stripe.outlier_mask.cols >= state.K_padded);

        {
            // Step 3: quantize the remaining elements for integer-domain analysis.
            const uint64_t cycle_start = EXSIA_CYCLE_READ();

            // Step 3.1: set a provisional block scale from the selected inlier exponent.
            e_pre = has_second_bucket ? blk.e2 : blk.e1;
            theta_pre = exp_to_theta(e_pre, meta.rho);

            // Step 3.2: wide-quantize without clipping and collect local inlier statistics.
            unit_quant_.quantize_block(
                blk.x,
                local_row,
                blk_idx * blk_size,
                stripe.outlier_mask,
                theta_pre,
                q_tmp,
                S,
                SS);

            // Step 3.3: refine the residual mask using one-sided upper-tail magnitude τ.
            for (size_t i = 0; i < blk_size; ++i)
            {
                const size_t col = blk_idx * blk_size + i;
                if (col < state.K_logical && !stripe.outlier_mask.is_set(local_row, col))
                    ++unmasked_count;
            }

            for (size_t i = 0; i < blk_size; ++i)
            {
                const size_t col = blk_idx * blk_size + i;
                if (col >= state.K_logical || stripe.outlier_mask.is_set(local_row, col))
                    continue;

                if (unit_sigma_.detect_sigma(q_tmp[i], S, SS, unmasked_count))
                {
                    int_outlier_mask.set(0, i);
                    has_int_outlier = true;
                }
            }
            const uint64_t cycle_end = EXSIA_CYCLE_READ();
            EXSIA_LOG_CYCLE("exsia.Local.Stage.3.one_sided_magnitude_sigma_selection", cycle_start, cycle_end);
            cycle_delta += cycle_end >= cycle_start ? cycle_end - cycle_start : 0;
        }

        {
            // Step 4: finalize block exponent and wide-quantized values.
            const uint64_t cycle_start = EXSIA_CYCLE_READ();
            if (!has_int_outlier)
            {
                // Step 4.1: reuse the pre-quantized block if no extra outlier is found.
                blk.e_b = e_pre;
                blk.theta_b = theta_pre;
                std::copy_n(q_tmp.begin(), blk_size, q_final.begin());
            }
            else
            {
                // Step 4.2: promote integer outliers and update the inlier exponent.
                unit_outlier_.mark_outlier(stripe, row, blk_idx, blk_size, int_outlier_mask);
                unit_exp_.update_block_top2_exp(stripe.outlier_mask, local_row, blk_idx, blk);
                blk.e_b = blk.e1;
                blk.theta_b = exp_to_theta(blk.e_b, meta.rho);

                if (blk.theta_b == theta_pre)
                {
                    // Step 4.3: the existing wide integers were already computed at the final scale.
                    std::copy_n(q_tmp.begin(), blk_size, q_final.begin());
                }
                else
                {
                    // Step 4.4: inlier exponent changed; recompute wide integers at finer scale.
                    unit_quant_.quantize_block(blk.x, blk.theta_b, q_final);
                }
            }
            const uint64_t cycle_end = EXSIA_CYCLE_READ();
            EXSIA_LOG_CYCLE("exsia.Local.Stage.4.finalize_block_quantization", cycle_start, cycle_end);
            cycle_delta += cycle_end >= cycle_start ? cycle_end - cycle_start : 0;
        }

        GGML_ASSERT(stripe_q_wide.size() >= base + blk_size);
        {
            // Step 5: store the local result for stripe folding.
            const uint64_t cycle_start = EXSIA_CYCLE_READ();
            for (size_t i = 0; i < blk_size; ++i)
                stripe_q_wide[base + i] = q_final[i];

            const size_t block_exp_idx = local_row * state.blocks_per_row + blk_idx;
            GGML_ASSERT(stripe_block_exp.size() > block_exp_idx);
            stripe_block_exp[block_exp_idx] = blk.e_b;

            unit_exp_.update_stripe_top2_exp(stripe, blk.e_b);
            const uint64_t cycle_end = EXSIA_CYCLE_READ();
            EXSIA_LOG_CYCLE("exsia.Local.Stage.5.store_result_for_stripe_folding", cycle_start, cycle_end);
            cycle_delta += cycle_end >= cycle_start ? cycle_end - cycle_start : 0;
        }

        return true;
    }

    bool StripeFolding::run(Meta &meta,
                            ExSIAState &state,
                            StripeState &stripe,
                            ggml_gemmini_args_t &args,
                            size_t stripe_idx,
                            int8_t *dst,
                            const std::vector<int32_t> &stripe_q_wide,
                            const std::vector<int16_t> &stripe_block_exp,
                            uint64_t &cycle_delta)
    {
        const int16_t neg_inf = std::numeric_limits<int16_t>::min();
        cycle_delta = 0;

        {
            // Step 1: determine stripe exponent from top-2 distinct block exponents.
            const uint64_t cycle_start = EXSIA_CYCLE_READ();
            if (stripe.e1 == neg_inf)
            {
                stripe.e_s = 0;
                // Step 1.1: no valid exponent exists (all zeros or sanitized non-finite inputs); keep promotion off.
                stripe.promote_top_block = false;
            }
            else if (stripe.e2 == neg_inf)
            {
                stripe.e_s = stripe.e1;
                // Step 1.2: only one distinct block exponent; keep promotion off.
                stripe.promote_top_block = false;
            }
            else
            {
                stripe.e_s = stripe.e2;
                // Step 1.3: multiple exponents exist; promote top-exponent blocks.
                stripe.promote_top_block = true;
            }
            const uint64_t cycle_end = EXSIA_CYCLE_READ();
            EXSIA_LOG_CYCLE("exsia.Stripe.Folding.1.determine_stripe_exp", cycle_start, cycle_end);
            cycle_delta += cycle_end >= cycle_start ? cycle_end - cycle_start : 0;
        }

        {
            // Step 2: store the per-stripe dequantization exponent.
            const uint64_t cycle_start = EXSIA_CYCLE_READ();
            const int16_t theta_s = exp_to_theta(stripe.e_s, meta.rho);
            GGML_ASSERT(stripe_idx < meta.theta.size());
            meta.theta[stripe_idx] = theta_s;
            const uint64_t cycle_end = EXSIA_CYCLE_READ();
            EXSIA_LOG_CYCLE("exsia.Stripe.Folding.2.store_stripe_exp", cycle_start, cycle_end);
            cycle_delta += cycle_end >= cycle_start ? cycle_end - cycle_start : 0;
        }

        GGML_ASSERT(dst != nullptr);
        GGML_ASSERT(state.B_size > 0);
        GGML_ASSERT(state.K_padded >= args.K);
        GGML_ASSERT(state.blocks_per_row == state.K_padded / state.B_size);
        GGML_ASSERT(stripe_q_wide.size() >= stripe.row_count() * state.K_padded);
        GGML_ASSERT(stripe_block_exp.size() >= stripe.row_count() * state.blocks_per_row);

        if (stripe.outlier_mask.rows == 0 || stripe.outlier_mask.cols == 0)
        {
            if (!stripe.outlier_mask.prepare(stripe.row_count(), state.K_padded))
                return false;
        }

        for (size_t r = stripe.row_start; r < stripe.row_end; ++r)
        {
            GGML_ASSERT(r < args.I);
            const size_t local_row = stripe.local_row(r);

            for (size_t b = 0; b < state.blocks_per_row; ++b)
            {
                const size_t block_offset = b * state.B_size;
                const size_t block_exp_idx = local_row * state.blocks_per_row + b;
                GGML_ASSERT(block_exp_idx < stripe_block_exp.size());

                const int16_t block_exp = stripe_block_exp[block_exp_idx];
                int16_t delta_theta_b = 0;
                {
                    // Step 3: fold block-local scales into the stripe scale.
                    const uint64_t cycle_start = EXSIA_CYCLE_READ();

                    // Step 3.1: compute the block-to-stripe folding shift.
                    delta_theta_b = block_exp == neg_inf || stripe.e_s == neg_inf
                                       ? 0
                                       : static_cast<int16_t>(block_exp - stripe.e_s);

                    // Step 3.2: mark currently unmasked inliers of top-exponent blocks.
                    if (stripe.promote_top_block && block_exp == stripe.e1)
                    {
                        BitMask &block_inlier_mask = stripe.scratch.folding_inlier_mask;
                        block_inlier_mask.clear_active_bits();

                        for (size_t i = 0; i < state.B_size; ++i)
                        {
                            const size_t col = block_offset + i;
                            if (col < args.K && !stripe.outlier_mask.is_set(local_row, col))
                                block_inlier_mask.set(0, i);
                        }
                        unit_outlier_.mark_outlier(stripe, r, b, state.B_size, block_inlier_mask);
                    }
                    const uint64_t cycle_end = EXSIA_CYCLE_READ();
                    EXSIA_LOG_CYCLE("exsia.Stripe.Folding.3.fold_block_scales_into_stripe_scale", cycle_start, cycle_end);
                    cycle_delta += cycle_end >= cycle_start ? cycle_end - cycle_start : 0;
                }

                {
                    // Step 4: generate dense int8 values and residual error.
                    const uint64_t cycle_start = EXSIA_CYCLE_READ();
                    for (size_t i = 0; i < state.B_size; ++i)
                    {
                        const size_t col = block_offset + i;
                        const size_t padded_idx = local_row * state.K_padded + col;

                        // Step 4.1: shift the wide integer to the stripe scale.
                        // For delta_theta_b < 0, this is a rounded-right-shift.
                        const int32_t q_shifted = detail::shift_q_i32(stripe_q_wide[padded_idx], delta_theta_b);

                        // Step 4.2: clip once and compute the residual.
                        const auto [q8, res] = unit_clip_.clip_with_residual(q_shifted);

                        // Step 4.3: store the dense int8 activation.
                        if (col < args.K)
                            dst[r * args.K + col] = q8;

                        // Step 4.4: store residual only for marked positions.
                        const bool outlier = col < args.K && stripe.outlier_mask.is_set(local_row, col);
                        const int32_t residual_i32 = outlier ? res : 0;

                        if (outlier && residual_i32 != 0)
                        {
                            meta.outliers.push_back({
                                static_cast<int>(r),
                                static_cast<int>(col),
                                residual_i32,
                            });
                        }
                    }
                    const uint64_t cycle_end = EXSIA_CYCLE_READ();
                    EXSIA_LOG_CYCLE("exsia.Stripe.Folding.4.generate_dense_int8_and_residual", cycle_start, cycle_end);
                    cycle_delta += cycle_end >= cycle_start ? cycle_end - cycle_start : 0;
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
        const uint64_t end_to_end_cycle_start = EXSIA_CYCLE_READ();
        const int16_t invalid_theta = std::numeric_limits<int16_t>::min();
        int8_t *dst = reinterpret_cast<int8_t *>(args.A);
        size_t logical_elem_count = 0;
        const bool logical_elem_count_ok = checked_mul_size(args.I, args.K, logical_elem_count);
        const auto fail = [&]() {
            if (dst != nullptr && logical_elem_count_ok)
                std::fill_n(dst, logical_elem_count, int8_t{0});

            meta.reset();
            state_ = ExSIAState{};
            return false;
        };

        meta.sigma = GGML_GEMMINI_EXSIA_SIGMA;
        state_ = ExSIAState{};
        state_.B_size = BLOCK_SIZE;
        state_.K_logical = args.K;
        if (dst == nullptr ||
            args.I == 0 || args.K == 0 ||
            !logical_elem_count_ok ||
            !checked_round_up_multiple(args.K, state_.B_size, state_.K_padded))
        {
            return fail();
        }

        state_.blocks_per_row = state_.K_padded / state_.B_size;

        size_t rows_per_stripe = args.I;
        if (args.tile_I > 0 && !checked_mul_size(args.tile_I, DIM, rows_per_stripe))
            return fail();

        if (rows_per_stripe == 0)
            return fail();

        size_t stripe_round_input = 0;
        if (!checked_add_size(args.I, rows_per_stripe - 1, stripe_round_input))
            return fail();

        const size_t num_stripes = stripe_round_input / rows_per_stripe;
        const float *src_data = ggml::gemmini::activation_data(A);
        if (!src_data)
            return fail();

        size_t max_stripe_rows = std::min(args.I, rows_per_stripe);
        size_t max_stripe_elem_count = 0;
        size_t max_stripe_block_count = 0;
        if (!checked_mul_size(max_stripe_rows, state_.K_padded, max_stripe_elem_count) ||
            !checked_mul_size(max_stripe_rows, state_.blocks_per_row, max_stripe_block_count))
        {
            return fail();
        }

        meta.theta.assign(num_stripes, invalid_theta);
        meta.outliers.clear();
        meta.outliers.reserve(logical_elem_count);

        release_vector(state_.x_f32);
        release_vector(state_.q_wide);
        release_vector(state_.block_exp);
        release_vector(state_.residual);

        stream_.prepare(max_stripe_elem_count, max_stripe_block_count);
        state_.stripe.assign(num_stripes, StripeState{});
        for (size_t s = 0; s < num_stripes; ++s)
        {
            StripeState &stripe = state_.stripe[s];
            stripe.row_start = s * rows_per_stripe;
            stripe.row_end = std::min((s + 1) * rows_per_stripe, args.I);
            if (!stripe.outlier_mask.prepare(stripe.row_count(), state_.K_padded) ||
                !stripe.scratch.prepare(state_.B_size))
                return fail();
        }

        uint64_t local_stage_cycle_sum = 0;
        uint64_t stripe_folding_cycle_sum = 0;
        for (size_t s = 0; s < num_stripes; ++s)
        {
            StripeState &stripe = state_.stripe[s];
            size_t stripe_elem_count = 0;
            size_t stripe_block_count = 0;
            if (!checked_mul_size(stripe.row_count(), state_.K_padded, stripe_elem_count) ||
                !checked_mul_size(stripe.row_count(), state_.blocks_per_row, stripe_block_count))
            {
                return fail();
            }

            stream_.prepare(stripe_elem_count, stripe_block_count);
            for (size_t r = stripe.row_start; r < stripe.row_end; ++r)
            {
                for (size_t b = 0; b < state_.blocks_per_row; ++b)
                {
                    uint64_t local_cycle_delta = 0;
                    std::vector<float> &block_x = stripe.scratch.block.x;

                    const size_t col_offset = b * state_.B_size;
                    for (size_t i = 0; i < state_.B_size; ++i)
                    {
                        const size_t col = col_offset + i;
                        block_x[i] = col < args.K ? src_data[r * args.K + col] : 0.f;
                    }

                    if (!local_.run(meta, state_, stripe, block_x, r, b,
                                    stream_.q_wide, stream_.block_exp, local_cycle_delta))
                        return fail();
                    local_stage_cycle_sum += local_cycle_delta;
                }
            }

            uint64_t stripe_cycle_delta = 0;
            if (!folding_.run(meta, state_, stripe, args, s, dst,
                              stream_.q_wide, stream_.block_exp, stripe_cycle_delta))
                return fail();
            stripe_folding_cycle_sum += stripe_cycle_delta;
        }

        ggml::gemmini::log::debug(
            "exsia",
            "[exsia] I=%zu K=%zu stripes=%zu tau=%d outliers=%zu",
            args.I,
            args.K,
            num_stripes,
            meta.sigma,
            meta.outliers.size());

        const uint64_t end_to_end_cycle_end = EXSIA_CYCLE_READ();
        EXSIA_LOG_CYCLE("exsia.local_stage_total", 0, local_stage_cycle_sum);
        EXSIA_LOG_CYCLE("exsia.stripe_folding_total", 0, stripe_folding_cycle_sum);
        EXSIA_LOG_CYCLE("exsia.end_to_end", end_to_end_cycle_start, end_to_end_cycle_end);

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
        const auto run = [&]() -> bool
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
        };

#if CYCLE_DETAIL
        const uint64_t cycle_start = EXSIA_CYCLE_READ();
        const bool ok = run();
        EXSIA_LOG_CYCLE("exsia.dequantize_activation", cycle_start, EXSIA_CYCLE_READ());
        return ok;
#else
        return run();
#endif
    }

}
