#include "ggml-impl.h"

#include "exsia.hpp"
#include "types.hpp"

#include <cmath>

namespace ggml::gemmini::quants::act::exsia
{
    int16_t ExpScanner::unbiased_exp(const float &x)
    {
        if (x == 0.f || !std::isfinite(x))
            return std::numeric_limits<int16_t>::min();

        return static_cast<int16_t>(std::ilogb(std::abs(x)));
    }

    void ExpScanner::scan_top2_exp(const Meta &meta, const std::vector<float> &x, BlockState &blk)
    {
        const size_t n = meta.B_size < x.size() ? meta.B_size : x.size();
        for (size_t i = 0; i < n; ++i)
        {
            int16_t exp = unbiased_exp(x[i]);
            blk.e.push_back(exp);
            if (exp > blk.e1) {
                blk.e2 = blk.e1;
                blk.e1 = exp;
            } else if (exp > blk.e2) {
                blk.e2 = exp;
            }
        }
    }

    void ExpScanner::masked_top2_exp(const BitMask &, size_t, size_t, BlockState &)
    {
        GGML_ASSERT(false && "not yet implemented");
    }

    void ExpScanner::update_stripe_top2_exp(StripeState &, int16_t)
    {
        GGML_ASSERT(false && "not yet implemented");
    }

    void OutlierMarker::mark_outlier(StripeState &, const BitMask &) const
    {
        GGML_ASSERT(false && "not yet implemented");
    }

    int32_t WideQuantizer::quantize(float, int16_t)
    {
        GGML_ASSERT(false && "not yet implemented");
        return 0;
    }

    std::tuple<int32_t, int64_t, int64_t> WideQuantizer::quantize(float, size_t, size_t, const BitMask &, int16_t)
    {
        GGML_ASSERT(false && "not yet implemented");
        return {0, 0, 0};
    }

    bool SigmaDetector::detect_3sigma(int32_t, int64_t, int64_t)
    {
        GGML_ASSERT(false && "not yet implemented");
        return false;
    }

    std::pair<int8_t, int32_t> ResidualClipper::clip_with_residual(int32_t q)
    {
        (void)q;

        GGML_ASSERT(false && "not yet implemented");
        return {0, 0};
    }

    bool LocalStage::run(
        Meta &,
        ExSIAState &,
        ggml_gemmini_args_t &,
        size_t)
    {
        GGML_ASSERT(false && "not yet implemented");
        return false;
    }

    bool StripeFolding::run(
        Meta &,
        ExSIAState &,
        ggml_gemmini_args_t &,
        size_t,
        size_t,
        size_t,
        size_t,
        int8_t *,
        int32_t *)
    {
        GGML_ASSERT(false && "not yet implemented");
        return false;
    }

    bool ExSIA::run(
        Meta &,
        const ggml_tensor *,
        ggml_gemmini_args_t &)
    {
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
