#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <variant>
#include <vector>

#include "../../../ggml-gemmini-args.h"
#include "../../common/tensor_util.hpp"
#include "token.hpp"

#include <gemmini/layer.hpp>
#include <gemmini/log.hpp>

namespace ggml::gemmini::quants::act::token
{
    namespace
    {
        int32_t quantize_to_i32(float value, float scale)
        {
            if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0f)
            {
                return 0;
            }

            const double scaled = std::round(static_cast<double>(value) / static_cast<double>(scale));
            if (!std::isfinite(scaled))
            {
                return scaled < 0.0
                           ? std::numeric_limits<int32_t>::min()
                           : std::numeric_limits<int32_t>::max();
            }

            const double min_i32 = static_cast<double>(std::numeric_limits<int32_t>::min());
            const double max_i32 = static_cast<double>(std::numeric_limits<int32_t>::max());
            if (scaled <= min_i32)
            {
                return std::numeric_limits<int32_t>::min();
            }
            if (scaled >= max_i32)
            {
                return std::numeric_limits<int32_t>::max();
            }

            return static_cast<int32_t>(scaled);
        }

        int8_t clip_to_i8(int32_t value)
        {
            if (value > 127)
            {
                return 127;
            }
            if (value < -128)
            {
                return -128;
            }
            return static_cast<int8_t>(value);
        }

        float resolve_scale(const Meta &meta, size_t row)
        {
            if (row >= meta.scales.size())
            {
                return 1.0f;
            }

            const float scale = meta.scales[row];
            return std::isfinite(scale) && scale > 0.0f ? scale : 1.0f;
        }

        Meta &active_meta()
        {
            static thread_local Meta meta;
            return meta;
        }

        bool set_scale(const float *src_data, const ggml_gemmini_args_t &args, Meta &meta)
        {
            if (!src_data || args.I == 0 || args.K == 0)
            {
                return false;
            }

            meta.scales.assign(args.I, 1.0f);
            for (size_t row = 0; row < args.I; ++row)
            {
                double max_abs = 0.0;
                for (size_t col = 0; col < args.K; ++col)
                {
                    const float value = src_data[row * args.K + col];
                    if (std::isfinite(value))
                    {
                        max_abs = std::max(max_abs, std::fabs(static_cast<double>(value)));
                    }
                }

                if (max_abs != 0.0)
                {
                    const float scale = static_cast<float>(max_abs / 127.0);
                    meta.scales[row] = std::isfinite(scale) && scale > 0.0f ? scale : 1.0f;
                }
            }

            return true;
        }
    }

    void set_config(Meta &meta)
    {
        (void)meta;
    }

    bool quantize(const ggml_tensor *src, ggml_gemmini_args_t &args)
    {
        int8_t *dst = reinterpret_cast<int8_t *>(args.A);
        if (!src || src->type != GGML_TYPE_F32 || !dst || args.I == 0 || args.K == 0)
        {
            return false;
        }

        const float *src_data = ggml::gemmini::activation_data(src);
        Meta &meta = active_meta();
        if (!src_data || !set_scale(src_data, args, meta))
        {
            return false;
        }

        meta.outliers.clear();

        for (size_t row = 0; row < args.I; ++row)
        {
            const float scale = resolve_scale(meta, row);
            for (size_t col = 0; col < args.K; ++col)
            {
                const size_t idx = row * args.K + col;
                const int32_t q32 = quantize_to_i32(src_data[idx], scale);
                dst[idx] = clip_to_i8(q32);
            }
        }

        const char *layer = ggml::gemmini::types::to_string(args.layer_type);
        ggml::gemmini::log::debug(layer,
                                  "[quantize_token] I=%zu K=%zu row_scales=%zu",
                                  args.I, args.K, meta.scales.size());

        return true;
    }

    bool dequantize_activation(
        float *dst,
        size_t dst_row_stride,
        size_t dst_col_stride,
        size_t rows,
        size_t cols,
        const ggml_gemmini_args_t &args)
    {
        const int8_t *src = reinterpret_cast<const int8_t *>(args.A);
        if (!src || !dst || args.I == 0 || args.K == 0 ||
            dst_row_stride == 0 || dst_col_stride == 0 ||
            rows == 0 || cols == 0)
        {
            return false;
        }

        const Meta &meta = active_meta();

        if (args.sA != 0 && args.sA != args.K)
        {
            return false;
        }

        const size_t src_row_stride = args.K;
        const size_t row_count = std::min(rows, args.I);
        const size_t col_count = std::min(cols, args.K);
        const size_t max_size = std::numeric_limits<size_t>::max();

        for (size_t row = 0; row < row_count; ++row)
        {
            const float scale = resolve_scale(meta, row);
            for (size_t col = 0; col < col_count; ++col)
            {
                if ((row != 0 && src_row_stride > max_size / row) ||
                    (row != 0 && dst_row_stride > max_size / row) ||
                    (col != 0 && dst_col_stride > max_size / col))
                {
                    return false;
                }

                const size_t src_row_offset = row * src_row_stride;
                if (src_row_offset > max_size - col)
                {
                    return false;
                }

                const size_t dst_row_offset = row * dst_row_stride;
                const size_t dst_col_offset = col * dst_col_stride;
                if (dst_row_offset > max_size - dst_col_offset)
                {
                    return false;
                }

                const size_t src_idx = src_row_offset + col;
                const size_t dst_idx = dst_row_offset + dst_col_offset;
                dst[dst_idx] = static_cast<float>(src[src_idx]) * scale;
            }
        }

        return true;
    }

}
