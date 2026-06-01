#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../../ggml-gemmini-args.h"
#include "math.hpp"

namespace ggml::gemmini {

void dequant_acc_block_with_activation_exponent(
    const ggml_gemmini_args_t &args,
    size_t k_offset,
    size_t block_k,
    const int32_t *acc32,
    size_t acc_stride,
    int16_t activation_e_t) {
    if (!args.f_out || !acc32 || block_k == 0 || args.I == 0 || args.J == 0) {
        return;
    }

    const size_t weight_block_size = args.block_size_k > 0 ? args.block_size_k : static_cast<size_t>(QK8_0);
    const size_t k_begin = k_offset;
    const size_t k_end = k_offset + block_k;
    const size_t total_weight_scale_elems = args.blocks_K * args.blocks_J;
    const size_t out_col_stride = args.col_stride_f_out ? args.col_stride_f_out : 1;
    const size_t out_row_stride = args.stride_f_out ? args.stride_f_out : args.J;
    const bool tile_scope_weight_avg = (args.group_scope == GGML_GEMMINI_GROUP_TILE);

    for (size_t i = 0; i < args.I; ++i) {
        const int32_t *row_acc32 = acc32 + i * acc_stride;
        float *row_out = args.f_out + i * out_row_stride;

        for (size_t j = 0; j < args.J; ++j) {
            float scale_w = 1.0f;

            if (args.B_scales && args.blocks_K > 0) {
                if (tile_scope_weight_avg) {
                    const size_t weight_blk_begin = k_begin / weight_block_size;
                    const size_t weight_blk_end = (k_end - 1) / weight_block_size;
                    double weighted_sum = 0.0;
                    size_t weighted_count = 0;

                    for (size_t wb = weight_blk_begin; wb <= weight_blk_end; ++wb) {
                        if (wb >= args.blocks_K) {
                            break;
                        }

                        const size_t blk_begin = wb * weight_block_size;
                        const size_t blk_end = std::min(blk_begin + weight_block_size, k_end);
                        const size_t ov_begin = std::max(blk_begin, k_begin);
                        if (blk_end <= ov_begin) {
                            continue;
                        }

                        const size_t overlap = blk_end - ov_begin;
                        const size_t weight_scale_idx = j * args.blocks_K + wb;
                        if (weight_scale_idx >= total_weight_scale_elems) {
                            continue;
                        }

                        weighted_sum += static_cast<double>(args.B_scales[weight_scale_idx]) * static_cast<double>(overlap);
                        weighted_count += overlap;
                    }

                    if (weighted_count > 0) {
                        scale_w = static_cast<float>(weighted_sum / static_cast<double>(weighted_count));
                    }
                } else {
                    const size_t weight_blk = k_begin / weight_block_size;
                    if (weight_blk < args.blocks_K) {
                        const size_t weight_scale_idx = j * args.blocks_K + weight_blk;
                        if (weight_scale_idx < total_weight_scale_elems) {
                            scale_w = args.B_scales[weight_scale_idx];
                        }
                    }
                }
            }

            float contrib = static_cast<float>(row_acc32[j]) * scale_w;
            contrib = apply_activation_exponent(contrib, activation_e_t, args.activation_m);
            row_out[j * out_col_stride] += contrib;
        }
    }
}

}
