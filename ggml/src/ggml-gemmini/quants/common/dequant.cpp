#include "dequant.hpp"

#include "../../ggml-gemmini-args.h"
#include "../act/dispatch.hpp"
#include "../dec/dec_internal.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace ggml::gemmini {

namespace {

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

quants::dec::WeightScaleInfo build_weight_scale_info_impl(
    const ggml_gemmini_args_t &args,
    quants::dec::WeightScaleInfoMode mode)
{
    quants::dec::WeightScaleInfo result{};
    if (quants::dec::is_q8_channel_direct_read_args(args)) {
        if (!args.has_q8_channel_direct_read_contract()) {
            result.supported = false;
            return result;
        }

        result.rows = args.J;
        result.cols = 1;
        result.block_size = 1;
        result.row_header_mode = true;
        return result;
    }

    if (args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar) {
        if (!quants::dec::is_q8_channel_dense_sidecar_args(args)) {
            result.supported = false;
            return result;
        }

        result.data = args.weight_channel_scales;
        result.rows = args.J;
        result.cols = 1;
        result.block_size = 1;
        result.channel_mode = true;
        return result;
    }

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

    if (quants::dec::is_q8_hp1_args(args)) {
        static thread_local std::vector<float> weight_scales;
        const size_t rows = args.J;
        const size_t cols = args.q8_hp1_blocks_per_row;
        size_t scale_count = 0;
        if (rows == 0 || cols == 0 || !checked_mul_size(rows, cols, scale_count))
            return result;

        weight_scales.resize(scale_count);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols; ++block) {
                const block_q8_hp1 *qblock = args.q8_hp1_block(row, block);
                if (qblock == nullptr) {
                    result.supported = false;
                    return result;
                }
                const int16_t m = qblock->m;
                const float channel_scale = qblock->channel_scale;
                weight_scales[row * cols + block] =
                    m == INT16_MIN ? 0.0f : gemmini_ldexp_fast_pos(channel_scale, static_cast<int>(m));
            }
        }

        result.data = weight_scales.data();
        result.rows = rows;
        result.cols = cols;
        result.block_size = QK8_HP;
        return result;
    }

    if (quants::dec::is_q8_hp2_args(args)) {
        static thread_local std::vector<float> weight_scales;
        const size_t rows = args.J;
        const size_t cols = args.q8_hp2_blocks_per_row;
        size_t scale_count = 0;
        if (rows == 0 || cols == 0 || !checked_mul_size(rows, cols, scale_count))
            return result;

        weight_scales.resize(scale_count);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols; ++block) {
                const block_q8_hp2 *qblock = args.q8_hp2_block(row, block);
                if (qblock == nullptr) {
                    result.supported = false;
                    return result;
                }
                const int16_t m = qblock->m;
                const float channel_scale = qblock->channel_scale;
                weight_scales[row * cols + block] =
                    m == INT16_MIN ? 0.0f : gemmini_ldexp_fast_pos(channel_scale, static_cast<int>(m));
            }
        }

        result.data = weight_scales.data();
        result.rows = rows;
        result.cols = cols;
        result.block_size = QK8_HP;
        return result;
    }

    if (quants::dec::is_q8_h2_args(args)) {
        static thread_local std::vector<float> weight_scales;
        const size_t rows = args.J;
        const size_t cols = args.q8_h2_blocks_per_row;
        size_t scale_count = 0;
        if (rows == 0 || cols == 0 || !checked_mul_size(rows, cols, scale_count))
            return result;

        weight_scales.resize(scale_count);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols; ++block) {
                const block_q8_h2 *qblock = args.q8_h2_block(row, block);
                if (qblock == nullptr) {
                    result.supported = false;
                    return result;
                }
                weight_scales[row * cols + block] = qblock->channel_scale * qblock->m / 255.0f;
            }
        }

        result.data = weight_scales.data();
        result.rows = rows;
        result.cols = cols;
        result.block_size = QK8_H2;
        return result;
    }

    if (quants::dec::is_q8_h1_weight_args(args)) {
        auto &scratch = get_q8_h1_output_dequant_scratch();
        const size_t rows = args.blocks_J ? args.blocks_J : args.J;
        const size_t cols = args.blocks_per_row;
        const size_t block_size = QK8_0;
        const bool stripe_mode = args.stripe_J > 1;
        const bool native_h1 = quants::dec::is_q8_h1_args(args);
        size_t scale_count = 0;

        if (rows == 0 || cols == 0 ||
            (args.block_size_k != 0 && args.block_size_k != block_size) ||
            !checked_mul_size(rows, cols, scale_count)) {
            return result;
        }

        if (!native_h1 && stripe_mode && (!args.s_rf_stripe || !args.R_stripe)) {
            result.supported = false;
            return result;
        }

        if (!native_h1 && !stripe_mode && (!args.s_rf || !args.R))
            return result;

        scratch.weight_scales.resize(scale_count);
        for (size_t j = 0; j < rows; ++j) {
            const size_t stripe_idx = stripe_mode ? (j / args.stripe_J) : 0;

            for (size_t blk = 0; blk < cols; ++blk) {
                const size_t idx = j * cols + blk;
                const block_q8_h1 *native_block = native_h1 ? args.q8_h1_block(j, blk) : nullptr;
                if (native_h1 && native_block == nullptr) {
                    result.supported = false;
                    return result;
                }
                const float s_rf = native_h1 ? native_block->s_rf :
                    (stripe_mode ? args.s_rf_stripe[stripe_idx] : args.s_rf[j]);
                const uint16_t R = native_h1 ? native_block->R :
                    (stripe_mode ? args.R_stripe[stripe_idx] : args.R[j]);
                const uint64_t c_eff =
                    static_cast<uint64_t>(native_h1 ? native_block->c_b : static_cast<uint16_t>(args.c_b[idx])) +
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
    result.rows = mode == quants::dec::WeightScaleInfoMode::Dec ?
        args.blocks_J : (args.blocks_J ? args.blocks_J : args.J);
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

namespace quants::dec {

WeightScaleInfo build_weight_scale_info(
    const ggml_gemmini_args_t &args,
    WeightScaleInfoMode mode)
{
    return build_weight_scale_info_impl(args, mode);
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

    const quants::dec::WeightScaleInfo scales = quants::dec::build_weight_scale_info(
        args,
        quants::dec::WeightScaleInfoMode::CommonOutput);
    if (!scales.supported || scales.rows < args.J) {
        return;
    }

    size_t first_block = 0;
    if (!scales.scalar_mode && !scales.row_header_mode && !scales.channel_mode) {
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

            const float weight_scale = scales.row_header_mode ?
                args.q8_channel_scale(j) :
                (scales.scalar_mode ? scales.scalar :
                 (scales.channel_mode ? scales.data[j] : scales.data[j * scales.cols + first_block]));
            const double scaled = static_cast<double>(row_acc32[j]) *
                                  static_cast<double>(weight_scale) *
                                  static_cast<double>(activation_scale);
            args.f_out[dst_offset] += static_cast<float>(scaled);
        }
    }
}

}
