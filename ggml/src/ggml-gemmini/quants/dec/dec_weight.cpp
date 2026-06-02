#include "dec_weight.hpp"
#include "dec_types.hpp"
#include "../../ggml-gemmini-args.h"

namespace ggml::gemmini::quants::dec { namespace
{
    bool is_q80_r_weight_args(const ggml_gemmini_args_t &args)
    {
        return args.B &&
               !args.B_scales &&
               args.c_b &&
               ((args.stripe_J > 1) || (args.s_rf && args.R)) &&
               args.blocks_per_row > 0;
    }

    WeightLayout resolve_weight_layout(const ggml_gemmini_args_t &args)
    {
        if (is_q80_r_weight_args(args) || args.transpose_B)
        {
            return WeightLayout::JxK_ColMajor;
        }

        return WeightLayout::KxJ_RowMajor;
    }

    size_t resolve_weight_stride_elems(const ggml_gemmini_args_t &args)
    {
        if (is_q80_r_weight_args(args))
        {
            return args.K;
        }

        const size_t fallback_stride = args.transpose_B ? args.K : args.J;
        return args.sB ? args.sB : fallback_stride;
    }
}

void load_weight_row_scaled(
    size_t k,
    const ggml_gemmini_args_t &args,
    const float *weight_scales,
    size_t scale_rows,
    size_t blocks_k,
    size_t block_size_k,
    float *Wk_f)
{
    const int8_t *weights = reinterpret_cast<const int8_t *>(args.B);
    const size_t J = args.J;
    if (!weights || !Wk_f || J == 0)
        return;

    const size_t weight_stride = resolve_weight_stride_elems(args);
    if (weight_stride == 0)
        return;

    if (resolve_weight_layout(args) == WeightLayout::KxJ_RowMajor)
    {
        const int8_t *row = weights + k * weight_stride;
        for (size_t j = 0; j < J; ++j)
            Wk_f[j] = static_cast<float>(row[j]);
    }
    else
    {
        for (size_t j = 0; j < J; ++j)
            Wk_f[j] = static_cast<float>(weights[j * weight_stride + k]);
    }

    if (weight_scales && block_size_k > 0 && blocks_k > 0)
    {
        const size_t blk = k / block_size_k;

        for (size_t j = 0; j < J; ++j)
        {
            if (j < scale_rows && blk < blocks_k)
                Wk_f[j] *= weight_scales[j * blocks_k + blk];
        }
    }
}
} // namespace ggml::gemmini::quants::dec
