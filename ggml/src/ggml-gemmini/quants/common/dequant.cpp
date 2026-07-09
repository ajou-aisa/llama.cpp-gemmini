#include "dequant.hpp"

#include "../../ggml-gemmini-args.h"
#include "../act/dispatch.hpp"
#include "../dec/dec_internal.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace ggml::gemmini {

namespace {

struct WeightScaleInfo
{
    const float *data = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    size_t block_size = 0;
    float scalar = 1.0f;
    bool scalar_mode = false;
    bool supported = true;
};

struct Q8H1OutputDequantScratch
{
    std::vector<float> weight_scales;
};

Q8H1OutputDequantScratch &get_q8_h1_output_dequant_scratch()
{
    static thread_local Q8H1OutputDequantScratch scratch;
    return scratch;
}

bool checked_mul_size(size_t lhs, size_t rhs, size_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        return false;

    out = lhs * rhs;
    return true;
}

WeightScaleInfo build_weight_scale_info(const ggml_gemmini_args_t &args)
{
    WeightScaleInfo result{};
    if (args.weight_i8_scale_active) {
        if (!args.B)
            return result;

        result.rows = args.J;
        result.cols = 1;
        result.block_size = 1;
        result.scalar = args.weight_scale;
        result.scalar_mode = true;
        return result;
    }

    if (quants::dec::is_q8_h1_weight_args(args)) {
        auto &scratch = get_q8_h1_output_dequant_scratch();
        const size_t rows = args.blocks_J ? args.blocks_J : args.J;
        const size_t cols = args.blocks_per_row;
        const size_t block_size = QK8_0;
        const bool stripe_mode = args.stripe_J > 1;
        size_t scale_count = 0;

        if (rows == 0 || cols == 0 ||
            (args.block_size_k != 0 && args.block_size_k != block_size) ||
            !checked_mul_size(rows, cols, scale_count)) {
            return result;
        }

        if (stripe_mode && (!args.s_rf_stripe || !args.R_stripe)) {
            result.supported = false;
            return result;
        }

        if (!stripe_mode && (!args.s_rf || !args.R))
            return result;

        scratch.weight_scales.resize(scale_count);
        for (size_t j = 0; j < rows; ++j) {
            const size_t stripe_idx = stripe_mode ? (j / args.stripe_J) : 0;
            const float s_rf = stripe_mode ? args.s_rf_stripe[stripe_idx] : args.s_rf[j];
            const uint16_t R = stripe_mode ? args.R_stripe[stripe_idx] : args.R[j];

            for (size_t blk = 0; blk < cols; ++blk) {
                const size_t idx = j * cols + blk;
                const uint64_t c_eff =
                    static_cast<uint64_t>(static_cast<uint16_t>(args.c_b[idx])) +
                    static_cast<uint64_t>(R);
                scratch.weight_scales[idx] = static_cast<float>(
                    static_cast<double>(s_rf) * static_cast<double>(c_eff));
            }
        }

        result.data = scratch.weight_scales.data();
        result.rows = rows;
        result.cols = cols;
        result.block_size = block_size;
        return result;
    }

    if (!args.B_scales)
        return result;

    result.data = args.B_scales;
    result.rows = args.blocks_J ? args.blocks_J : args.J;
    result.cols = args.blocks_K;
    result.block_size = args.block_size_k ? args.block_size_k : QK8_0;
    return result;
}

bool output_offset(size_t row, size_t col, size_t row_stride, size_t col_stride, size_t &offset)
{
    size_t row_offset = 0;
    size_t col_offset = 0;
    if (!checked_mul_size(row, row_stride, row_offset) ||
        !checked_mul_size(col, col_stride, col_offset) ||
        row_offset > std::numeric_limits<size_t>::max() - col_offset) {
        return false;
    }

    offset = row_offset + col_offset;
    return true;
}

bool checked_k_end(size_t k_offset, size_t block_k, size_t &k_end)
{
    if (block_k == 0 || k_offset > std::numeric_limits<size_t>::max() - (block_k - 1))
        return false;

    k_end = k_offset + block_k - 1;
    return true;
}

}

void dequantize(
    const ggml_gemmini_args_t &args,
    size_t k_offset,
    size_t block_k,
    const int32_t *acc32,
    size_t acc_stride)
{
    if (!args.f_out || !acc32 || args.I == 0 || args.J == 0 || block_k == 0 || acc_stride == 0)
        return;

    if (acc_stride < args.J)
        return;

    const WeightScaleInfo scales = build_weight_scale_info(args);
    if (!scales.supported || scales.rows < args.J) {
        return;
    }

    size_t first_block = 0;
    if (!scales.scalar_mode) {
        if (!scales.data || scales.cols == 0 || scales.block_size == 0)
            return;

        size_t k_end = 0;
        if (!checked_k_end(k_offset, block_k, k_end))
            return;

        first_block = k_offset / scales.block_size;
        const size_t last_block = k_end / scales.block_size;
        if (first_block != last_block || first_block >= scales.cols)
            return;
    }

    const size_t row_stride = args.stride_f_out ? args.stride_f_out : args.J;
    const size_t col_stride = args.col_stride_f_out ? args.col_stride_f_out : 1;
    const std::vector<float> activation_scales = quants::act::activation_scales(args, args.I);

    for (size_t i = 0; i < args.I; ++i) {
        const float activation_scale = i < activation_scales.size() ? activation_scales[i] : 1.0f;
        const int32_t *row_acc32 = acc32 + i * acc_stride;

        for (size_t j = 0; j < args.J; ++j) {
            size_t dst_offset = 0;
            if (!output_offset(i, j, row_stride, col_stride, dst_offset))
                return;

            const float weight_scale = scales.scalar_mode ?
                scales.scalar :
                scales.data[j * scales.cols + first_block];
            const double scaled = static_cast<double>(row_acc32[j]) *
                                  static_cast<double>(weight_scale) *
                                  static_cast<double>(activation_scale);
            args.f_out[dst_offset] += static_cast<float>(scaled);
        }
    }
}

}
