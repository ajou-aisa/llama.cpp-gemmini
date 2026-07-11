// Q8_H1 weight quantization implementation (ggml-free)

#include "quantize_Q8_H1.hpp"
#include "../common/fp16_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace ggml::gemmini::quants { namespace {

constexpr size_t q8_0_block_size = 32;
constexpr float q8_h1_scale_bins = 255.0f;

uint8_t quantize_scale_code(float s_b, float s_rf, uint16_t R)
{
    if (!std::isfinite(s_b) || !std::isfinite(s_rf) || s_rf <= 0.0f)
        return 0;

    const double c_eff = std::round(static_cast<double>(s_b) / static_cast<double>(s_rf));
    const double shifted_code = c_eff - static_cast<double>(R);
    const double clamped = std::max(0.0, std::min(255.0, shifted_code));
    return static_cast<uint8_t>(clamped);
}

void set_constant_scale(float s_b, int blocks_per_row, BlockQ8_H1 *dst)
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

bool quantize_row_q8_h1(
    const block_q8_0 *src_blocks,
    int blocks_per_row,
    BlockQ8_H1 *dst
) {
    if (!src_blocks || !dst || !dst->c_b || !dst->qs || blocks_per_row <= 0)
        return false;

    float min_s = std::numeric_limits<float>::max();
    float max_s = std::numeric_limits<float>::lowest();

    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const block_q8_0 &src_block = src_blocks[block_idx];
        const float s_b = fp16_to_fp32(src_block.d);
        if (!std::isfinite(s_b))
            return false;

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

    dst->s_rf = scale_range / q8_h1_scale_bins;
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

float recover_block_scale(const BlockQ8_H1 *block, int block_idx)
{
    if (!block || !block->c_b || block_idx < 0)
        return 0.0f;

    const uint64_t c_eff =
        static_cast<uint64_t>(block->c_b[block_idx]) + static_cast<uint64_t>(block->R);
    return block->s_rf * static_cast<float>(c_eff);
}

} // namespace ggml::gemmini::quants
