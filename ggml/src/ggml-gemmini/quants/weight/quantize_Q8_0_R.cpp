// Q8_0_R weight quantization implementation (ggml-free)

#include "quantize_Q8_0_R.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ggml::gemmini::quants { namespace {

constexpr size_t q8_0_block_size = 32;
constexpr float q8_0_r_scale_bins = 255.0f;

uint8_t round_to_u8(float value)
{
    if (!std::isfinite(value) || value <= 0.0f) {
        return 0;
    }

    const double rounded = std::round(static_cast<double>(value));
    const double clamped = std::min(255.0, std::max(0.0, rounded));
    return static_cast<uint8_t>(clamped);
}

uint8_t quantize_scale_code(float s_b, float s_rf, uint16_t R)
{
    if (!std::isfinite(s_b) || !std::isfinite(s_rf) || s_rf <= 0.0f) {
        return 0;
    }

    const double c_eff = std::round(static_cast<double>(s_b) / static_cast<double>(s_rf));
    const double shifted_code = c_eff - static_cast<double>(R);
    const double clamped = std::max(0.0, std::min(255.0, shifted_code));
    return static_cast<uint8_t>(clamped);
}

void set_constant_scale(float s_b, int blocks_per_row, BlockQ8_0_R *dst)
{
    if (s_b > 0.0f && std::isfinite(s_b)) {
        dst->s_rf = s_b;
        dst->R = 1;
    } else {
        dst->s_rf = 0.0f;
        dst->R = 0;
    }

    std::memset(dst->c_b, 0, static_cast<size_t>(blocks_per_row));
}

} // namespace

bool quantize_row_q8_0_r(
    const block_q8_0 *src_blocks,
    int blocks_per_row,
    BlockQ8_0_R *dst
) {
    if (!src_blocks || !dst || !dst->c_b || !dst->qs || blocks_per_row <= 0) {
        return false;
    }

    float min_s = std::numeric_limits<float>::infinity();
    float max_s = -std::numeric_limits<float>::infinity();

    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const block_q8_0 &src_block = src_blocks[block_idx];
        const float s_b = fp16_to_fp32(src_block.d);
        if (!std::isfinite(s_b)) {
            return false;
        }

        min_s = std::min(min_s, s_b);
        max_s = std::max(max_s, s_b);

        std::memcpy(
            dst->qs + static_cast<size_t>(block_idx) * q8_0_block_size,
            src_block.qs,
            q8_0_block_size
        );
    }

    const float scale_range = max_s - min_s;
    if (!std::isfinite(scale_range) || scale_range <= 0.0f) {
        set_constant_scale(min_s, blocks_per_row, dst);
        return true;
    }

    dst->s_rf = scale_range / q8_0_r_scale_bins;
    if (!std::isfinite(dst->s_rf) || dst->s_rf <= 0.0f) {
        set_constant_scale(min_s, blocks_per_row, dst);
        return true;
    }

    const double r_val = std::round(static_cast<double>(min_s) / static_cast<double>(dst->s_rf));
    dst->R = static_cast<uint16_t>(std::min(65535.0, std::max(0.0, r_val)));

    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const float s_b = fp16_to_fp32(src_blocks[block_idx].d);
        dst->c_b[block_idx] = quantize_scale_code(s_b, dst->s_rf, dst->R);
    }

    return true;
}

bool quantize_panel_q8_0_r(
    const block_q8_0 *src_blocks,
    int blocks_per_row,
    int num_rows,
    BlockQ8_0_R *dst,
    float *dst_s_rf_panel,
    uint16_t *dst_R_panel
) {
    if (blocks_per_row <= 0 || num_rows < 0) {
        return false;
    }
    if (num_rows == 0) {
        return true;
    }
    if (!src_blocks || !dst || !dst_s_rf_panel || !dst_R_panel) {
        return false;
    }

    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
        if (!dst[row_idx].c_b || !dst[row_idx].qs) {
            return false;
        }
    }

    float min_s = std::numeric_limits<float>::infinity();
    float max_s = -std::numeric_limits<float>::infinity();

    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
        const size_t row_offset = static_cast<size_t>(row_idx) * static_cast<size_t>(blocks_per_row);
        BlockQ8_0_R &dst_row = dst[row_idx];

        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const block_q8_0 &src_block = src_blocks[row_offset + static_cast<size_t>(block_idx)];
            const float s_b = fp16_to_fp32(src_block.d);
            if (!std::isfinite(s_b)) {
                return false;
            }

            min_s = std::min(min_s, s_b);
            max_s = std::max(max_s, s_b);

            std::memcpy(
                dst_row.qs + static_cast<size_t>(block_idx) * q8_0_block_size,
                src_block.qs,
                q8_0_block_size
            );
        }
    }

    const float scale_range = max_s - min_s;
    if (!std::isfinite(scale_range) || scale_range <= 0.0f) {
        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
            set_constant_scale(min_s, blocks_per_row, &dst[row_idx]);
        }
        *dst_s_rf_panel = dst[0].s_rf;
        *dst_R_panel = dst[0].R;
        return true;
    }

    const float s_rf = scale_range / q8_0_r_scale_bins;
    if (!std::isfinite(s_rf) || s_rf <= 0.0f) {
        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
            set_constant_scale(min_s, blocks_per_row, &dst[row_idx]);
        }
        *dst_s_rf_panel = dst[0].s_rf;
        *dst_R_panel = dst[0].R;
        return true;
    }

    const double r_val = std::round(static_cast<double>(min_s) / static_cast<double>(s_rf));
    const uint16_t R = static_cast<uint16_t>(std::min(65535.0, std::max(0.0, r_val)));

    *dst_s_rf_panel = s_rf;
    *dst_R_panel = R;

    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
        const size_t row_offset = static_cast<size_t>(row_idx) * static_cast<size_t>(blocks_per_row);
        BlockQ8_0_R &dst_row = dst[row_idx];
        dst_row.s_rf = s_rf;
        dst_row.R = R;

        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const float s_b = fp16_to_fp32(src_blocks[row_offset + static_cast<size_t>(block_idx)].d);
            dst_row.c_b[block_idx] = quantize_scale_code(s_b, s_rf, R);
        }
    }

    return true;
}

float recover_block_scale(const BlockQ8_0_R *block, int block_idx)
{
    if (!block || !block->c_b || block_idx < 0) {
        return 0.0f;
    }

    const uint64_t c_eff =
        static_cast<uint64_t>(block->c_b[block_idx]) + static_cast<uint64_t>(block->R);
    return block->s_rf * static_cast<float>(c_eff);
}

} // namespace ggml::gemmini::quants
