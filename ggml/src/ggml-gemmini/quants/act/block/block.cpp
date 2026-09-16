#include "block.hpp"

#include "../../../ggml-gemmini-args.h"
#include "../../../residual/direct/direct-builder.hpp"
#include "../../../residual/rmd/rmd-compose.hpp"
#include "../../common/tensor_util.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <variant>

#ifndef GGML_GEMMINI_EXSIA_SIGMA
#define GGML_GEMMINI_EXSIA_SIGMA 2
#endif

namespace ggml::gemmini::quants::act::block {
namespace {

uint64_t next_block_run_id() {
    static std::atomic<uint64_t> next{0};
    return next.fetch_add(1, std::memory_order_relaxed);
}

bool checked_mul_size(size_t lhs, size_t rhs, size_t & out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        return false;
    out = lhs * rhs;
    return true;
}


int32_t quantize_value(float value, float scale) {
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0f) {
        return 0;
    }

    const double scaled = static_cast<double>(value) / static_cast<double>(scale);
    if (!std::isfinite(scaled)) {
        return scaled < 0.0 ? std::numeric_limits<int32_t>::min()
                            : std::numeric_limits<int32_t>::max();
    }
    if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
        return std::numeric_limits<int32_t>::min();
    if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
        return std::numeric_limits<int32_t>::max();
    return static_cast<int32_t>(std::nearbyint(scaled));
}

bool quantize_block(const float * data,
                    ggml_gemmini_args_t & args,
                    size_t row,
                    size_t k_begin,
                    float & scale,
                    ggml::gemmini::residual::TimedResidualCapture * capture,
                    size_t stripe_row) {
    const size_t count = std::min(kGroupSize, args.K - k_begin);
    double finite_max_abs = 0.0;
#if GGML_GEMMINI_ENABLE_RMD
    double sum = 0.0;
    double sum_sq = 0.0;
    size_t finite_count = 0;
#endif
    for (size_t offset = 0; offset < count; ++offset) {
        const float value = data[row * args.K + k_begin + offset];
        if (!std::isfinite(value))
            continue;
        const double x = static_cast<double>(value);
        finite_max_abs = std::max(finite_max_abs, std::fabs(x));
#if GGML_GEMMINI_ENABLE_RMD
        sum += x;
        sum_sq += x * x;
        ++finite_count;
#endif
    }

    double scale_max_abs = finite_max_abs;
#if GGML_GEMMINI_ENABLE_RMD
    std::array<bool, kGroupSize> outliers{};
    double inlier_max_abs = 0.0;
    size_t inlier_count = 0;
    const double mean = finite_count == 0 ? 0.0 : sum / static_cast<double>(finite_count);
    const double variance = finite_count == 0 ? 0.0 :
        std::max(0.0, sum_sq / static_cast<double>(finite_count) - mean * mean);
    const double sigma = std::sqrt(variance);
    for (size_t offset = 0; offset < count; ++offset) {
        const float value = data[row * args.K + k_begin + offset];
        if (!std::isfinite(value))
            continue;
        const double x = static_cast<double>(value);
        if (sigma != 0.0 && std::fabs(x - mean) > GGML_GEMMINI_EXSIA_SIGMA * sigma) {
            outliers[offset] = true;
        } else {
            inlier_max_abs = std::max(inlier_max_abs, std::fabs(x));
            ++inlier_count;
        }
    }
    if (inlier_count != 0)
        scale_max_abs = inlier_max_abs;
#endif

    scale = 1.0f;
    if (scale_max_abs != 0.0) {
        scale = static_cast<float>(
            scale_max_abs /
            static_cast<double>(ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_QMAX));
        if (!std::isfinite(scale) || scale <= 0.0f)
            return false;
    }

    for (size_t offset = 0; offset < count; ++offset) {
        const size_t k = k_begin + offset;
        const int32_t q32 = quantize_value(data[row * args.K + k], scale);
        const int32_t q = std::clamp(
            q32,
            ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_QMIN,
            ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_QMAX);
        if (!args.A.set(row, k, q))
            return false;
#if GGML_GEMMINI_ENABLE_RMD
        const int64_t residual = static_cast<int64_t>(q32) - static_cast<int64_t>(q);
        if (residual < std::numeric_limits<int32_t>::min() ||
            residual > std::numeric_limits<int32_t>::max())
            return false;
        if (outliers[offset] && residual != 0 &&
            (capture == nullptr || !capture->add_residual(
                stripe_row, k, static_cast<int32_t>(residual))))
            return false;
#else
        (void) capture;
        (void) stripe_row;
#endif
    }
    return true;
}

}

bool quantize(const ggml_tensor * src, ggml_gemmini_args_t & args) {
    if (src == nullptr || src->type != GGML_TYPE_F32 || !args.A.valid() ||
        args.I == 0 || args.K == 0) {
        return false;
    }
    auto * meta = std::get_if<Meta>(&args.act_quant.storage());
    const float * data = ggml::gemmini::activation_data(src);
    if (meta == nullptr || data == nullptr) {
        return false;
    }

    const size_t blocks_per_row =
        args.K / kGroupSize + (args.K % kGroupSize != 0);
    size_t scale_count = 0;
    if (!checked_mul_size(args.I, blocks_per_row, scale_count))
        return false;
    meta->reset();
    meta->run_id = next_block_run_id();
    meta->rows = args.I;
    meta->cols = args.K;
    meta->scales.assign(scale_count, 1.0f);

    const auto geometry = args.activation_quant_geometry();
    if (!geometry.ok())
        return false;
    const size_t rows_per_stripe = geometry.geometry.stripe_rows;
    for (size_t row_begin = 0, stripe_id = 0;
         row_begin < args.I;
         row_begin += rows_per_stripe, ++stripe_id) {
        const size_t row_count = std::min(rows_per_stripe, args.I - row_begin);
#if GGML_GEMMINI_ENABLE_RMD
        ggml::gemmini::residual::TimedResidualCapture capture(args.residual_route);
        capture.set_context(meta->run_id, args.matmul_layer.empty() ? nullptr : args.matmul_layer.c_str());
        capture.reset(stripe_id, row_begin, row_count, args.K, args.J);
#endif
        for (size_t local_row = 0; local_row < row_count; ++local_row) {
            const size_t row = row_begin + local_row;
            for (size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
                if (!quantize_block(
                        data, args, row, block_index * kGroupSize,
                        meta->scales[row * blocks_per_row + block_index],
#if GGML_GEMMINI_ENABLE_RMD
                        &capture,
#else
                        nullptr,
#endif
                        local_row))
                    return false;
            }
        }
#if GGML_GEMMINI_ENABLE_RMD
        const auto payload = capture.finish();
        if (payload.packet)
            meta->rmd_packets.push_back(payload.packet);
        if (payload.direct)
            meta->direct_residuals.push_back(payload.direct);
        if (capture.status() != ggml::gemmini::rmd::RmdStatus::success)
            return false;
#endif
    }
    return true;
}

bool dequantize_activation(float * dst,
                           size_t dst_row_stride,
                           size_t dst_col_stride,
                           size_t rows,
                           size_t cols,
                           const ggml_gemmini_args_t & args) {
    const auto * meta = std::get_if<Meta>(&args.act_quant.storage());
    if (dst == nullptr || dst_row_stride == 0 || dst_col_stride == 0 ||
        !args.A.valid() || args.I == 0 || args.K == 0 ||
        meta == nullptr || meta->rows == 0 || meta->cols != args.K ||
        (args.sA != 0 && args.sA != args.K)) {
        return false;
    }

    const size_t row_count = std::min(rows, args.I);
    const size_t col_count = std::min(cols, args.K);
    if (row_count == 0 || col_count == 0 ||
        args.activation_row_offset > meta->rows ||
        row_count > meta->rows - args.activation_row_offset)
        return false;
    const size_t blocks_per_row =
        meta->cols / kGroupSize + (meta->cols % kGroupSize != 0);
    size_t scale_count = 0;
    if (!checked_mul_size(meta->rows, blocks_per_row, scale_count) ||
        meta->scales.size() != scale_count ||
        (!meta->rmd_packets.empty() && !meta->direct_residuals.empty()))
        return false;

    size_t residual_count = 0;
    if (!checked_mul_size(row_count, col_count, residual_count))
        return false;
    std::vector<int32_t> residuals(residual_count, 0);
#if GGML_GEMMINI_ENABLE_RMD
    const size_t global_row_begin = args.activation_row_offset;
    const size_t global_row_end = global_row_begin + row_count;
    if (!meta->rmd_packets.empty() &&
        ggml::gemmini::rmd::expand_packets_to_plane(
            meta->rmd_packets, global_row_begin, global_row_end,
            col_count, residuals) != ggml::gemmini::rmd::RmdStatus::success)
        return false;
    if (!meta->direct_residuals.empty() &&
        ggml::gemmini::residual::expand_direct_payloads_to_plane(
            meta->direct_residuals, global_row_begin, global_row_end,
            meta->cols, args.J, col_count, residuals) !=
                ggml::gemmini::rmd::RmdStatus::success)
        return false;
#endif
    for (size_t row = 0; row < row_count; ++row) {
        if (row != 0 && dst_row_stride > std::numeric_limits<size_t>::max() / row) {
            return false;
        }
        const size_t row_offset = row * dst_row_stride;
        for (size_t col = 0; col < col_count; ++col) {
            const size_t global_row = args.activation_row_offset + row;
            const float scale = meta->scales[
                global_row * blocks_per_row + col / kGroupSize];
            if (!std::isfinite(scale) || scale <= 0.0f)
                return false;
            if (col != 0 && dst_col_stride > std::numeric_limits<size_t>::max() / col) {
                return false;
            }
            const size_t col_offset = col * dst_col_stride;
            if (row_offset > std::numeric_limits<size_t>::max() - col_offset) {
                return false;
            }
            const int64_t restored = static_cast<int64_t>(args.A.get(row, col)) +
                residuals[row * col_count + col];
            if (restored < std::numeric_limits<int32_t>::min() ||
                restored > std::numeric_limits<int32_t>::max())
                return false;
            dst[row_offset + col_offset] = static_cast<float>(restored) * scale;
        }
    }
    return true;
}

} // namespace ggml::gemmini::quants::act::block
