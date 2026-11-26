#pragma once
#include "block-quantized.hpp"
namespace ggml::gemmini {
  enum class baseline_activation_quant_t : uint8_t {
    FLOAT = 0,
    EXSIA,
    TENSOR,
    TOKEN,
    BLOCK,
  };

  enum class baseline_weight_quant_t : uint8_t {
    FLOAT = 0,
    TENSOR,
    CHANNEL,
    BLOCK,
  };

  struct baseline_route_t {
    baseline_activation_quant_t activation_quant;
    baseline_weight_quant_t weight_quant;
    const char *cpu_cycle_label;
    const char *npu_cycle_label;
  };


const baseline_route_t *baseline_route_for(
      baseline_activation_quant_t activation_quant,
      baseline_weight_quant_t weight_quant);
DenseMatmulStatus tiled_matmul_auto_baseline_dense(
      struct ggml_gemmini_args_t *args,
      const baseline_route_t &route);
DenseMatmulStatus tiled_matmul_auto_baseline(
      struct ggml_gemmini_args_t *args,
      baseline_activation_quant_t activation_quant,
      baseline_weight_quant_t weight_quant);
void tiled_matmul_auto_fp(ggml_gemmini_args_t *args);
} // namespace ggml::gemmini
