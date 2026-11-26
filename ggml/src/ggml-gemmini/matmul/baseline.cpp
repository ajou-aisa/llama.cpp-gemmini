// Derived from Gemmini SDK develop fcfcf5ce; see matmul/README.md and LICENSE.gemmini*.
#include "baseline.hpp"
#include "execution.hpp"
#include "ggml-gemmini-args.h"
#include "gemmini.h"
#include "../../ggml-npu-common/profiling/include/npu/log.hpp"
#include "../../ggml-npu-common/profiling/include/npu/cycle_reader.hpp"
#include "../../ggml-npu-common/quantization/activation/dispatch.hpp"
#include "../../ggml-npu-common/quantization/common/dequant.hpp"
#include <algorithm>
#include <climits>
#include <optional>
namespace ggml::gemmini {
void tiled_matmul_auto_fp(ggml_gemmini_args_t *args) {
    if (args == nullptr) return;
    const char *layer = args->matmul_layer.c_str();
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    if (test_observe_semantic_layer(TestSemanticLayerSite::physical_auto_fp, layer)) return;
#endif
    // Retained SDK semantic/profiling hook. Production FP compute is owned by
    // dense.cpp's matmul_cpu_fp route and the separate FP32 backend.
    const uint64_t start = ggml::npu::cycle::read();
    const uint64_t end = ggml::npu::cycle::read();
    gemmini_log_cycle(layer, "[tiled_matmul_auto_fp32] npu.Gemmini HW tiled_matmul", start, end);
}
const baseline_route_t *baseline_route_for(
      baseline_activation_quant_t activation_quant,
      baseline_weight_quant_t weight_quant) {
    static const baseline_route_t routes[] = {
      {baseline_activation_quant_t::EXSIA, baseline_weight_quant_t::TENSOR,
       "cpu.matmul.exsia_tensor", "gemmini.matmul.exsia_tensor"},
      {baseline_activation_quant_t::TENSOR, baseline_weight_quant_t::TENSOR,
       "cpu.matmul.tensor_tensor", "gemmini.matmul.tensor_tensor"},
      {baseline_activation_quant_t::TOKEN, baseline_weight_quant_t::TENSOR,
       "cpu.matmul.token_tensor", "gemmini.matmul.token_tensor"},
      {baseline_activation_quant_t::TENSOR, baseline_weight_quant_t::CHANNEL,
       "cpu.matmul.tensor_channel", "gemmini.matmul.tensor_channel"},
      {baseline_activation_quant_t::TENSOR, baseline_weight_quant_t::BLOCK,
       "cpu.matmul.tensor_block", "gemmini.matmul.tensor_block"},
      {baseline_activation_quant_t::TOKEN, baseline_weight_quant_t::CHANNEL,
       "cpu.matmul.token_channel", "gemmini.matmul.token_channel"},
      {baseline_activation_quant_t::TOKEN, baseline_weight_quant_t::BLOCK,
       "cpu.matmul.token_block", "gemmini.matmul.token_block"},
      {baseline_activation_quant_t::BLOCK, baseline_weight_quant_t::TENSOR,
       "cpu.matmul.block_tensor", "gemmini.matmul.block_tensor"},
      {baseline_activation_quant_t::BLOCK, baseline_weight_quant_t::CHANNEL,
       "cpu.matmul.block_channel", "gemmini.matmul.block_channel"},
      {baseline_activation_quant_t::BLOCK, baseline_weight_quant_t::BLOCK,
       "cpu.matmul.block_block", "gemmini.matmul.block_block"},
    };

    for (const auto &route : routes)
      if (route.activation_quant == activation_quant && route.weight_quant == weight_quant)
        return &route;

    return nullptr;
  }
namespace {
bool baseline_weight_scale_provider(const ggml_gemmini_args_t &args,
                                             baseline_weight_quant_t weight_quant,
                                             size_t channel,
                                             size_t k_block,
                                             float &scale) {
    switch (weight_quant) {
      case baseline_weight_quant_t::TENSOR:
        if (!args.weight_i8_scale_active || !std::isfinite(args.weight_scale))
          return false;
        scale = args.weight_scale;
        return true;
      case baseline_weight_quant_t::CHANNEL:
      {
        const bool q8_channel_direct_read = args.has_q8_channel_direct_read_contract();
        const bool q8_channel_dense_sidecar = args.has_q8_channel_dense_sidecar_contract();
        if (q8_channel_direct_read) {
          if (channel >= args.J)
            return false;
          const float candidate = args.q8_channel_scale(channel);
          if (!std::isfinite(candidate))
            return false;

          scale = candidate;
          return true;
        }

        if (!q8_channel_dense_sidecar || channel >= args.J)
            return false;

        const float candidate = args.weight_channel_scales[channel];
        if (!std::isfinite(candidate))
          return false;

        scale = candidate;
        return true;
      }
      case baseline_weight_quant_t::BLOCK:
      {
        const size_t block_size = args.block_size_k ? args.block_size_k : GGML_GEMMINI_BLOCK_SIZE;
        if (args.B_scales == nullptr || args.J == 0 || args.K == 0 ||
            args.blocks_J != args.J || args.blocks_K == 0 || block_size == 0 ||
            args.K > std::numeric_limits<size_t>::max() - (block_size - 1) ||
            args.blocks_K != (args.K + block_size - 1) / block_size ||
            channel >= args.blocks_J || k_block >= args.blocks_K ||
            channel > (std::numeric_limits<size_t>::max() - k_block) / args.blocks_K)
          return false;

        const size_t offset = channel * args.blocks_K + k_block;
        const float candidate = args.B_scales[offset];
        if (!std::isfinite(candidate))
          return false;

        scale = candidate;
        return true;
      }
      case baseline_weight_quant_t::FLOAT:
        return false;
    }

    return false;
  }
bool baseline_activation_scale_provider(
      const ggml_gemmini_args_t &args,
      baseline_activation_quant_t activation_quant,
      std::vector<float> &scales) {
    const auto &storage = args.act_quant.storage();
    switch (activation_quant) {
      case baseline_activation_quant_t::EXSIA:
        if (std::get_if<ggml::npu::quants::act::exsia::Meta>(&storage) == nullptr)
          return false;
        break;
      case baseline_activation_quant_t::TENSOR:
        if (std::get_if<ggml::npu::quants::act::tensor::Meta>(&storage) == nullptr)
          return false;
        break;
      case baseline_activation_quant_t::TOKEN:
      {
        const auto *meta = std::get_if<ggml::npu::quants::act::token::Meta>(&storage);
        GGML_ASSERT(meta != nullptr && "TOKEN baseline route requires token::Meta");
        GGML_ASSERT(meta->scales.size() == args.I &&
                    std::all_of(meta->scales.begin(), meta->scales.end(),
                                [](float scale) { return std::isfinite(scale) && scale > 0.0f; }) &&
                    "TOKEN baseline route requires one finite positive activation scale per row");
        break;
      }
      case baseline_activation_quant_t::BLOCK:
      {
        const auto *block_meta = std::get_if<ggml::npu::quants::act::block::Meta>(&storage);
        GGML_ASSERT((block_meta != nullptr ||
                     std::get_if<ggml::npu::quants::act::stripe::Meta>(&storage) != nullptr) &&
                    "BLOCK baseline route requires block::Meta or stripe::Meta");
        if (block_meta != nullptr) {
          GGML_ASSERT(block_meta->scales.size() == args.I &&
                      std::all_of(block_meta->scales.begin(), block_meta->scales.end(),
                                  [](float scale) { return std::isfinite(scale) && scale > 0.0f; }) &&
                      "BLOCK baseline route requires one finite positive activation scale per row");
        }
        break;
      }
      case baseline_activation_quant_t::FLOAT:
        return false;
    }

    scales = ggml::npu::quants::act::activation_scales(args, args.I);
    return scales.size() == args.I;
  }
}
DenseMatmulStatus tiled_matmul_auto_baseline_dense(
      struct ggml_gemmini_args_t *args,
      const baseline_route_t &route) {
    if (args == nullptr || args->A == nullptr || args->f_out == nullptr)
      return DenseMatmulStatus::invalid_contract;

    const char *layer = args->matmul_layer.c_str();
#if defined(GGML_GEMMINI_TEST_OBSERVER)
    if (test_observe_semantic_layer(TestSemanticLayerSite::physical_baseline_dense, layer))
      return DenseMatmulStatus::success;
#endif
    const bool native_q8 = args->weight_format == ggml_gemmini_args_t::weight_format_t::q8_h2 ||
        args->weight_format == ggml_gemmini_args_t::weight_format_t::q8_hp1 ||
        args->weight_format == ggml_gemmini_args_t::weight_format_t::q8_hp2;
    if (route.activation_quant == baseline_activation_quant_t::TENSOR &&
        route.weight_quant == baseline_weight_quant_t::TENSOR && native_q8) {
      return gemmini_block_quantized_matmul(args, args->tiled_matmul_type == CPU,
          "[tiled_matmul_auto_A-tensor_B-native-q8] tiled matmul");
    }

    if (args->B == nullptr)
      return DenseMatmulStatus::invalid_contract;

    const size_t dim_I = args->I;
    const size_t dim_J = args->J;
    const size_t dim_K = args->K;
    if (dim_I == 0 || dim_J == 0 || dim_K == 0 || args->sB == 0)
      return DenseMatmulStatus::invalid_contract;

    if (route.activation_quant == baseline_activation_quant_t::EXSIA &&
        route.weight_quant == baseline_weight_quant_t::TENSOR &&
        (!args->weight_i8_scale_active || !std::isfinite(args->weight_scale)))
      return DenseMatmulStatus::invalid_contract;

    const bool q8_channel_route = route.weight_quant == baseline_weight_quant_t::CHANNEL;
    const bool q8_channel_direct_read = q8_channel_route &&
        args->has_q8_channel_direct_read_contract();
    const bool q8_channel_dense_sidecar = q8_channel_route &&
        args->has_q8_channel_dense_sidecar_contract();
    if (q8_channel_route && !q8_channel_direct_read && !q8_channel_dense_sidecar)
      return DenseMatmulStatus::invalid_contract;

    if (route.activation_quant == baseline_activation_quant_t::TOKEN) {
      if (q8_channel_route) {
        GGML_ASSERT(args->transpose_B &&
                    (q8_channel_direct_read || q8_channel_dense_sidecar) &&
                    "TOKEN Q8_CHANNEL route requires transpose_B and a valid channel contract");
      } else {
        GGML_ASSERT(args->transpose_B && args->sB == args->K &&
                    "TOKEN baseline route requires dense row-major I8 weights");
      }
      if (route.weight_quant == baseline_weight_quant_t::TENSOR) {
        GGML_ASSERT(args->weight_i8_scale_active &&
                    std::isfinite(args->weight_scale) &&
                    args->weight_scale > 0.0f &&
                    "TOKEN/TENSOR requires a finite positive scalar weight scale");
      }
    }

    if (route.weight_quant == baseline_weight_quant_t::TENSOR &&
        args->weight_i8_scale_active && !std::isfinite(args->weight_scale))
      return DenseMatmulStatus::invalid_contract;

    GGML_ASSERT(args->act == GEMMINI_ACT_NONE);

    std::vector<float> activation_scales;
    if (!baseline_activation_scale_provider(*args, route.activation_quant, activation_scales))
      return DenseMatmulStatus::invalid_contract;

    const size_t block_size = args->block_size_k ? args->block_size_k : GGML_GEMMINI_BLOCK_SIZE;
    if (block_size == 0 || dim_K > std::numeric_limits<size_t>::max() - (block_size - 1))
      return DenseMatmulStatus::invalid_contract;
    const size_t weight_block_count = route.weight_quant == baseline_weight_quant_t::BLOCK ?
        (dim_K + block_size - 1) / block_size : 1;
    for (size_t j = 0; j < dim_J; ++j)
      for (size_t k_block = 0; k_block < weight_block_count; ++k_block) {
        float ignored_scale = 0.0f;
        if (route.weight_quant != baseline_weight_quant_t::TENSOR || args->weight_i8_scale_active) {
          if (!baseline_weight_scale_provider(*args, route.weight_quant, j, k_block, ignored_scale))
            return DenseMatmulStatus::invalid_contract;
        }
      }

    if (args->tile_I == 0 || args->tile_J == 0 || args->tile_K == 0)
      gemmini_set_tile_ws(args);

    const uint64_t start = ggml::npu::cycle::read();
    const elem_t *A = args->A;
    const elem_t *B = args->B;
    const size_t stride_A = args->sA ? args->sA : dim_K;
    const size_t row_stride_B = args->sB;
    const size_t out_row_stride = args->stride_f_out ? args->stride_f_out : dim_J;
    const size_t out_col_stride = args->col_stride_f_out ? args->col_stride_f_out : 1;
    const bool block_weight_route = route.weight_quant == baseline_weight_quant_t::BLOCK;
    std::vector<acc_t> acc32(dim_I * dim_J, 0);
    std::vector<double> block_acc;
    if (block_weight_route)
      block_acc.assign(dim_I * dim_J, 0.0);

    for (size_t i = 0; i < dim_I; ++i)
      for (size_t j = 0; j < dim_J; ++j)
        args->f_out[i * out_row_stride + j * out_col_stride] = 0.0f;

    const bool use_tiled_matmul = args->tiled_matmul_type != CPU && !block_weight_route;
    if (use_tiled_matmul) {
      const auto owner = args->transport;
      const gemmini_transport_t transport = owner ? *owner : *gemmini_rocc_transport();
      tiled_matmul(transport, dim_I, dim_J, dim_K,
          A, B, args->D, acc32.data(),
          stride_A, row_stride_B, args->sD, dim_J,
          1.0f, args->scale_B, args->scale_D,
          args->act, args->scale, args->bert_scale, args->repeating_bias,
          args->tile_I, args->tile_J, args->tile_K,
          args->transpose_A, args->transpose_B,
          true, args->low_D,
          args->weightA,
          args->tiled_matmul_type);
    } else {
      for (size_t i = 0; i < dim_I; ++i)
        for (size_t j = 0; j < dim_J; ++j) {
          int32_t acc = 0;
          for (size_t k = 0; k < dim_K; ++k) {
            const elem_t a = args->transpose_A ? A[k * stride_A + i] : A[i * stride_A + k];
            const int32_t product = static_cast<int32_t>(a) *
                static_cast<int32_t>(B[j * row_stride_B + k]);
            if (block_weight_route) {
              float weight_scale = 0.0f;
              if (!baseline_weight_scale_provider(*args, route.weight_quant, j, k / block_size, weight_scale))
                return DenseMatmulStatus::invalid_contract;
              block_acc[i * dim_J + j] += static_cast<double>(product) * weight_scale;
            } else {
              acc += product;
            }
          }
          if (!block_weight_route)
            acc32[i * dim_J + j] = acc;
        }
    }

    if (route.weight_quant == baseline_weight_quant_t::TENSOR && !args->weight_i8_scale_active) {
      ggml::npu::dequantize(*args, 0, dim_K, acc32.data(), dim_J);
    } else {
      for (size_t i = 0; i < dim_I; ++i)
        for (size_t j = 0; j < dim_J; ++j) {
          const size_t output_offset = i * out_row_stride + j * out_col_stride;
          if (block_weight_route) {
            args->f_out[output_offset] += static_cast<float>(block_acc[i * dim_J + j] * activation_scales[i]);
          } else {
            float weight_scale = 0.0f;
            if (!baseline_weight_scale_provider(*args, route.weight_quant, j, 0, weight_scale))
              return DenseMatmulStatus::invalid_contract;
            args->f_out[output_offset] += static_cast<float>(
                static_cast<double>(acc32[i * dim_J + j]) * weight_scale * activation_scales[i]);
          }
        }
    }

    const uint64_t end = ggml::npu::cycle::read();
    gemmini_log_cycle(layer,
        args->tiled_matmul_type == CPU ? route.cpu_cycle_label : route.npu_cycle_label,
        start, end);
    return DenseMatmulStatus::success;
  }
DenseMatmulStatus tiled_matmul_auto_baseline(
      struct ggml_gemmini_args_t *args,
      baseline_activation_quant_t activation_quant,
      baseline_weight_quant_t weight_quant) {
    if (activation_quant == baseline_activation_quant_t::FLOAT &&
        weight_quant == baseline_weight_quant_t::FLOAT) {
      if (args == nullptr)
        return DenseMatmulStatus::invalid_contract;
      tiled_matmul_auto_fp(args);
      return DenseMatmulStatus::success;
    }

    const baseline_route_t *route = baseline_route_for(activation_quant, weight_quant);
    if (route == nullptr)
      return DenseMatmulStatus::unsupported;
    return tiled_matmul_auto_baseline_dense(args, *route);
  }
} // namespace ggml::gemmini
