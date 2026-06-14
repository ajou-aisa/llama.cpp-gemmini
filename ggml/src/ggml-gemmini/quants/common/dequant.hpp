#pragma once

#include <cstddef>
#include <cstdint>

#include "../stripe.hpp"

struct ggml_gemmini_args_t;

namespace ggml::gemmini
{

void dequantize(
    const ggml_gemmini_args_t &args,
    size_t k_offset,
    size_t block_k,
    const int32_t *acc32,
    size_t acc_stride);

template <typename ScaleFn>
void update_q80_r_output_impl(
    const quants::Stripe &stripe_c,
    const int64_t *acc64,
    size_t acc_stride,
    ScaleFn scale_for_column,
    float activation_scale,
    float *dst,
    size_t dst_row_stride,
    size_t dst_col_stride)
{
    if (stripe_c.empty() || !acc64 || !dst || acc_stride == 0 || dst_row_stride == 0 || dst_col_stride == 0)
        return;

    for (size_t i = 0; i < stripe_c.I; ++i) {
        const int64_t *row_acc64 = acc64 + i * acc_stride;
        float *row_out = dst + i * dst_row_stride;

        for (size_t j = 0; j < stripe_c.J; ++j) {
            const size_t global_j = stripe_c.col_offset + j;
            float contrib = static_cast<float>(
                static_cast<double>(row_acc64[j]) * static_cast<double>(scale_for_column(global_j)));
            contrib *= activation_scale;
            row_out[j * dst_col_stride] += contrib;
        }
    }
}

} // namespace ggml::gemmini
