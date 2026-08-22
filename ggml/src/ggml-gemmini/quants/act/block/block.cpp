#include "block.hpp"

#include "../../../ggml-gemmini-args.h"
#include "../../common/tensor_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <variant>

namespace ggml::gemmini::quants::act::block {
namespace {

int32_t quantize_value(float value, float scale) {
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0f) {
        return 0;
    }

    const double scaled = static_cast<double>(value) / static_cast<double>(scale);
    if (!std::isfinite(scaled)) {
        return scaled < 0.0 ? ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_QMIN
                            : ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_QMAX;
    }
    const double rounded = std::nearbyint(scaled);
    return static_cast<int32_t>(std::clamp(
        rounded,
        static_cast<double>(ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_QMIN),
        static_cast<double>(ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_QMAX)));
}

} // namespace

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

    meta->scales.assign(args.I, 1.0f);
    meta->rmd_packets.clear();
    meta->direct_residuals.clear();

    constexpr size_t rows_per_block = GGML_GEMMINI_BLOCK_SIZE;
    static_assert(rows_per_block > 0);
    for (size_t row_begin = 0; row_begin < args.I; row_begin += rows_per_block) {
        const size_t row_end = row_begin + std::min(rows_per_block, args.I - row_begin);
        double max_abs = 0.0;
        for (size_t row = row_begin; row < row_end; ++row) {
            for (size_t col = 0; col < args.K; ++col) {
                const float value = data[row * args.K + col];
                if (std::isfinite(value)) {
                    max_abs = std::max(max_abs, std::fabs(static_cast<double>(value)));
                }
            }
        }

        float scale = 1.0f;
        if (max_abs > 0.0) {
            scale = static_cast<float>(
                max_abs / static_cast<double>(ggml::gemmini::config::GGML_GEMMINI_ACTIVATION_QMAX));
            if (!std::isfinite(scale) || scale <= 0.0f) {
                return false;
            }
        }

        for (size_t row = row_begin; row < row_end; ++row) {
            meta->scales[row] = scale;
            for (size_t col = 0; col < args.K; ++col) {
                if (!args.A.set(row, col, quantize_value(data[row * args.K + col], scale))) {
                    return false;
                }
            }
        }
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
        meta == nullptr || meta->scales.size() != args.I ||
        (args.sA != 0 && args.sA != args.K)) {
        return false;
    }

    const size_t row_count = std::min(rows, args.I);
    const size_t col_count = std::min(cols, args.K);
    for (size_t row = 0; row < row_count; ++row) {
        const float scale = meta->scales[row];
        if (!std::isfinite(scale) || scale <= 0.0f ||
            (row != 0 && dst_row_stride > std::numeric_limits<size_t>::max() / row)) {
            return false;
        }
        const size_t row_offset = row * dst_row_stride;
        for (size_t col = 0; col < col_count; ++col) {
            if (col != 0 && dst_col_stride > std::numeric_limits<size_t>::max() / col) {
                return false;
            }
            const size_t col_offset = col * dst_col_stride;
            if (row_offset > std::numeric_limits<size_t>::max() - col_offset) {
                return false;
            }
            dst[row_offset + col_offset] = static_cast<float>(args.A.get(row, col)) * scale;
        }
    }
    return true;
}

} // namespace ggml::gemmini::quants::act::block
