#include "ggml-impl.h"

#include "exsia.hpp"
#include "types.hpp"

#include "ggml-gemmini-args.h"

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

        for (auto &e : x)
            q.push_back(static_cast<int32_t>(std::lrint(std::ldexp(e, -theta_b))));

        return q;
    }

    std::tuple<std::vector<int32_t>, int64_t, int64_t>
    WideQuantizer::quantize_block(const std::vector<float> &x,
                                  size_t row,
                                  size_t blk_idx,
                                  const BitMask &mask,
                                  int16_t theta_b)
    {
        std::vector<int32_t> q;
        size_t n = x.size();
        q.reserve(n);

        int64_t S = 0;
        int64_t SS = 0;
        for (size_t i = 0; i < n; ++i)
        {
            size_t col = blk_idx * n + i;
            int32_t tmp = std::lrint(std::ldexp(x[i], -theta_b));
            q.push_back(static_cast<int32_t>(tmp));
            if (!mask.is_set(row, col))
            {
                S += tmp;
                SS += tmp * tmp;
            }
        }
        return {q, S, SS};
    }

    bool SigmaDetector::detect_3sigma(int32_t q, int64_t S, int64_t SS, size_t N)
    {
        return (N * q - S) * (N * q - S) > 9 * (N * SS - S * S);
    }

    std::pair<int8_t, int32_t> ResidualClipper::clip_with_residual(int32_t q)
    {
        int32_t q8 = q > 127 ? 127 : (q < -127 ? -127 : q);
        int32_t res = q - q8;
        return {static_cast<int8_t>(q8), res};
    }

    bool LocalStage::run(
        Meta &meta,
        ExSIAState &state,
        StripeState &stripe,
        BlockState &blk,
        size_t row,
        size_t blk_idx)
    {
        (void)meta;
        (void)state;
        (void)stripe;
        (void)blk;
        (void)row;
        (void)blk_idx;

        return false;
    }

    bool StripeFolding::run(
        Meta &,
        ExSIAState &,
        StripeState &,
        ggml_gemmini_args_t &,
        size_t,
        int8_t *,
        int32_t *)
    {
        GGML_ASSERT(false && "not yet implemented");
        return false;
    }

    bool ExSIA::run(
        Meta &meta,
        const ggml_tensor *A,
        ggml_gemmini_args_t &args)
    {
        (void)meta;
        (void)A;

        state_.B_size = BLOCK_SIZE;
        state_.K_padded = ((args.K + state_.B_size - 1) / state_.B_size) * state_.B_size;
        state_.blocks_per_row = state_.K_padded / state_.B_size;

        GGML_ASSERT(false && "ExSIA::run not yet implemented");
        return false;
    }

    void dequantize(
        const ggml_gemmini_args_t &,
        size_t,
        size_t,
        const int32_t *,
        size_t)
    {
        GGML_ASSERT(false && "dequantize not yet implemented");
    }
}
