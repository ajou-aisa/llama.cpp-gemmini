#include "dec_weight.hpp"
#include "dec_internal.hpp"
#include "dec_types.hpp"
#include "../../ggml-gemmini-args.h"

namespace ggml::gemmini::quants::dec {
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
