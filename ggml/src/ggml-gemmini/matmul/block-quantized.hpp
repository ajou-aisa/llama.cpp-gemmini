#pragma once
#include "ggml-gemmini-args.h"
namespace ggml::gemmini {
  enum class DenseMatmulStatus : uint8_t {
    success,
    invalid_contract,
    unsupported,
  };


DenseMatmulStatus gemmini_block_quantized_matmul(
      struct ggml_gemmini_args_t *args,
      bool use_cpu_dot,
      const char *cycle_label);
DenseMatmulStatus gemmini_block_quantized_matmul_auto(struct ggml_gemmini_args_t *args);
} // namespace ggml::gemmini
