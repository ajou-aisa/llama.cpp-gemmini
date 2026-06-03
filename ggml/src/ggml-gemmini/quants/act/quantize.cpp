#include "quantize.hpp"
#include "ethos/ethos.hpp"
#include "ethos/types.hpp"
#include "../../ggml-gemmini-args.h"
#include "../common/tensor_util.hpp"

#include <gemmini/log.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace ggml::gemmini::quants { namespace
{
    size_t clamp_tile_extent(size_t total, size_t offset, size_t tile_span)
    {
        if (offset >= total)
        {
            return 0;
        }

        return std::min(tile_span, total - offset);
    }

    size_t align_up(size_t value, size_t alignment)
    {
        if (alignment == 0)
        {
            return value;
        }

        return ((value + alignment - 1) / alignment) * alignment;
    }

    template <typename Args>
    void store_tile_stripe_activation_e_t(Args &args, int tile_row, int tile_col, int16_t e_t) {
        (void)tile_col;
        if (tile_row < 0) {
            return;
        }

        if (tile_row == 0) {
            args.activation_e_t_per_stripe_i.clear();
        }

        const size_t stripe_idx = static_cast<size_t>(tile_row);
        if (args.activation_e_t_per_stripe_i.size() <= stripe_idx) {
            args.activation_e_t_per_stripe_i.resize(stripe_idx + 1, e_t);
        }

        args.activation_e_t_per_stripe_i[stripe_idx] = e_t;
    }

    void copy_tile_k_chunk(
        const int8_t *src,
        size_t src_stride_elems,
        int8_t *dst,
        size_t dst_stride_elems,
        size_t rows,
        size_t tile_col_offset,
        size_t tile_k_actual) {
        if (!src || !dst || src_stride_elems == 0 || dst_stride_elems == 0 || rows == 0 || tile_k_actual == 0) {
            return;
        }

        for (size_t row = 0; row < rows; ++row) {
            int8_t *dst_row = dst + row * dst_stride_elems;
            std::fill_n(dst_row, dst_stride_elems, int8_t {0});

            const int8_t *src_row = src + row * src_stride_elems + tile_col_offset;
            std::memcpy(dst_row, src_row, tile_k_actual * sizeof(int8_t));
        }
    }
}

size_t qact_outlier_count(const ActivationQuantResult &result)
{
    return result.outliers.size();
}

const QactOutlier *qact_outliers(const ActivationQuantResult &result)
{
    if (result.outliers.empty())
    {
        return nullptr;
    }

    return result.outliers.data();
}

ActivationQuantResult quantize_activation_f32(
    const ggml_tensor *src,
    ggml_gemmini_args_t &args,
    int8_t *dst,
    ActivationQuantConfig &cfg)
{
    ActivationQuantConfig internal_cfg = cfg;
    internal_cfg.result.outliers.clear();
    internal_cfg.result.e_t = std::numeric_limits<int16_t>::min();

    if (!src || src->type != GGML_TYPE_F32 || !dst || args.I == 0 || args.K == 0)
    {
        return internal_cfg.result;
    }

    const float *src_data = ggml::gemmini::activation_data(src);
    if (!src_data)
    {
        return internal_cfg.result;
    }

    const size_t stride_k_bytes = src->nb[0] ? src->nb[0] : sizeof(float);
    const size_t stride_i_bytes = src->nb[1] ? src->nb[1] : args.K * stride_k_bytes;

    if (internal_cfg.block_size == 0 || (args.K % internal_cfg.block_size) != 0)
    {
        return internal_cfg.result;
    }

    act::ethos::Ethos ethos;
    if (!ethos.run(
            internal_cfg,
            reinterpret_cast<const char *>(src_data),
            stride_i_bytes,
            stride_k_bytes,
            args.I,
            args.K,
            0,
            0,
            dst))
    {
        return internal_cfg.result;
    }

    return internal_cfg.result;
}

ActivationQuantResult quantize_activation_f32_tile(
    const ggml_tensor *src,
    ggml_gemmini_args_t &args,
    int8_t *dst,
    ActivationQuantConfig &cfg,
    int tile_row)
{
    ActivationQuantConfig internal_cfg = cfg;
    internal_cfg.result.outliers.clear();
    internal_cfg.result.e_t = std::numeric_limits<int16_t>::min();
    internal_cfg.result.e_t_per_tile.clear();
    internal_cfg.result.num_padding_blocks = 0;

    if (!src || src->type != GGML_TYPE_F32 || tile_row < 0 || !dst)
    {
        return internal_cfg.result;
    }

    const float *src_data = ggml::gemmini::activation_data(src);
    if (!src_data)
    {
        return internal_cfg.result;
    }

    const size_t stride_k_bytes = src->nb[0] ? src->nb[0] : sizeof(float);
    const size_t stride_i_bytes = src->nb[1] ? src->nb[1] : args.K * stride_k_bytes;
    const size_t tile_row_offset = static_cast<size_t>(tile_row) * args.tile_I;
    const size_t real_I = clamp_tile_extent(args.I, tile_row_offset, args.tile_I);
    const size_t real_K = args.K;

    if (real_I == 0 || real_K == 0)
    {
        return internal_cfg.result;
    }

    if (internal_cfg.block_size == 0)
    {
        return internal_cfg.result;
    }

    const char *quant_src = reinterpret_cast<const char *>(src_data) + tile_row_offset * stride_i_bytes;
    size_t quant_stride_i_bytes = stride_i_bytes;
    size_t quant_stride_k_bytes = stride_k_bytes;

    const size_t padded_K = align_up(real_K, internal_cfg.block_size);

    std::vector<float> padded_buf;

    if (padded_K != real_K)
    {
        padded_buf.assign(real_I * padded_K, 0.0f);

        for (size_t i = 0; i < real_I; ++i)
        {
            const char *src_row = quant_src + i * quant_stride_i_bytes;
            float *dst_row = padded_buf.data() + i * padded_K;
            for (size_t k = 0; k < real_K; ++k)
            {
                dst_row[k] = *reinterpret_cast<const float *>(src_row + k * quant_stride_k_bytes);
            }
        }

        quant_src = reinterpret_cast<const char *>(padded_buf.data());
        quant_stride_i_bytes = padded_K * sizeof(float);
        quant_stride_k_bytes = sizeof(float);
    }

    const size_t num_real_blocks = real_I * (real_K / internal_cfg.block_size);
    internal_cfg.num_real_blocks = num_real_blocks;

    act::ethos::Ethos ethos;
    if (!ethos.run(
            internal_cfg,
            quant_src,
            quant_stride_i_bytes,
            quant_stride_k_bytes,
            real_I,
            padded_K,
            tile_row_offset,
            0,
            dst))
    {
        return internal_cfg.result;
    }

    return internal_cfg.result;
}

void reset_activation_quant_state(ggml_gemmini_args_t &args) {
    args.activation_outliers.clear();
    args.activation_e_t = std::numeric_limits<int16_t>::min();
    args.activation_m = 0;
    args.gemmini_call_k_logical = 0;
    args.gemmini_call_k_aligned = 0;
    args.gemmini_call_tile_k_elems = 0;
}

ActivationQuantConfig make_activation_quant_config(const ggml_gemmini_args_t &args) {
    ActivationQuantConfig cfg {};
    cfg.block_size = static_cast<size_t>(GGML_GEMMINI_BLOCK_SIZE);
    if (args.layer_type != ggml::gemmini::types::LayerType::unknown) {
        cfg.preset.second = args.layer_type;
    }

    switch (ggml::gemmini::config::CURRENT_ACTIVATION_QUANT) {
    case ggml::gemmini::config::ActivationQuantAlgo::ETHOS:
    default:
        act::ethos::set_config(cfg);
        break;
    }
    if (args.ethos_override_enabled) {
        const int q = args.ethos_q;
        if (q >= 2 && q <= 8) {
            cfg.m = static_cast<int16_t>(q - 2);
            cfg.qmax = static_cast<int8_t>((1 << (q - 1)) - 1);
        }
        cfg.delta = args.ethos_delta;
        cfg.l2_on = args.ethos_l2_enabled;
        cfg.l2 = {args.ethos_l2_c, args.ethos_l2_d};
    }

    return cfg;
}

void capture_activation_quant_result(
    ggml_gemmini_args_t &args,
    const ActivationQuantConfig &cfg,
    const ActivationQuantResult &res) {
    args.activation_e_t = res.e_t;
    args.activation_m = cfg.m;

    size_t resolved_group_size = cfg.block_size > 0 ? cfg.block_size : static_cast<size_t>(QK8_0);
    resolved_group_size = std::max<size_t>(1, resolved_group_size);

    const size_t logical_k = args.K > 0 ? args.K : 1;
    const size_t resolved_group_size_k = std::max<size_t>(1, std::min(resolved_group_size, logical_k));

    args.effective_group_size = resolved_group_size;
    args.effective_group_size_k = resolved_group_size_k;
    args.effective_group_size_aligned = std::max<size_t>(16, ((resolved_group_size_k + 15) / 16) * 16);

    const size_t outlier_count = qact_outlier_count(res);
    if (outlier_count == 0) {
        return;
    }

    const auto *outliers = qact_outliers(res);
    args.activation_outliers.clear();
    args.activation_outliers.reserve(outlier_count);
    for (size_t i = 0; i < outlier_count; ++i) {
        const auto &outlier = outliers[i];
        args.activation_outliers.push_back({outlier.row, outlier.col, outlier.original, outlier.saturated});
    }
}

void ggml_gemmini_quantize_activation_tile(
    const ggml_tensor *src,
    ggml_gemmini_args_t &args,
    int8_t *dst,
    int tile_row,
    int tile_col) {
    reset_activation_quant_state(args);
    if (!src || !dst || tile_row < 0 || tile_col < 0) {
        return;
    }

    const size_t tile_row_offset = static_cast<size_t>(tile_row) * args.tile_I;
    const size_t tile_i_actual = clamp_tile_extent(args.I, tile_row_offset, args.tile_I);
    if (tile_i_actual == 0 || args.K == 0) {
        return;
    }

    const size_t tile_col_offset = static_cast<size_t>(tile_col) * args.tile_K;
    const size_t tile_k_actual = clamp_tile_extent(args.K, tile_col_offset, args.tile_K);
    if (tile_k_actual == 0) {
        return;
    }

    auto cfg = make_activation_quant_config(args);
    ggml::gemmini::log::debug(
        ggml::gemmini::types::to_string(args.layer_type),
        "[ethos] final cfg model_arch=%s override=%d m=%d qmax=%d delta=%d l2_on=%d l2.c=%d l2.d=%d",
        args.model_arch ? args.model_arch : "",
        args.ethos_override_enabled ? 1 : 0,
        static_cast<int>(cfg.m),
        static_cast<int>(cfg.qmax),
        cfg.delta,
        cfg.l2_on ? 1 : 0,
        cfg.l2.c,
        cfg.l2.d);

    args.group_scope = GGML_GEMMINI_GROUP_BLOCK;

    const size_t padded_view_k = align_up(args.K, cfg.block_size);
    std::vector<int8_t> quantized_tile_row(tile_i_actual * padded_view_k, 0);
    auto res = quantize_activation_f32_tile(src, args, quantized_tile_row.data(), cfg, tile_row);

    const size_t dst_stride_elems = align_up(tile_k_actual, cfg.block_size);
    copy_tile_k_chunk(
        quantized_tile_row.data(),
        padded_view_k,
        dst,
        dst_stride_elems,
        tile_i_actual,
        tile_col_offset,
        tile_k_actual);

    store_tile_stripe_activation_e_t(args, tile_row, tile_col, res.e_t);
    capture_activation_quant_result(args, cfg, res);
}
} // namespace ggml::gemmini::quants
