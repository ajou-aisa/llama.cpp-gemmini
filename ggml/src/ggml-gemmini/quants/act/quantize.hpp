#pragma once

#include "types.hpp"
#include "ggml-gemmini-config.hpp"

struct ggml_tensor;
struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants
{
    using ActivationQuantConfig = act::Config;
    using ActivationQuantResult = act::Result;

// Accessor functions for outliers (encapsulated, not direct vector access)
size_t qact_outlier_count(const ActivationQuantResult &result);
const QactOutlier *qact_outliers(const ActivationQuantResult &result);

// Main quantization function: convert f32 activations to int8
// - src/args: source tensor metadata and Gemmini dimensions
// - dst: destination buffer (row-major int8, size I*K)
// - cfg: configuration options
// Returns: quantization result with scale(s) and optional outlier info
ActivationQuantResult quantize_activation_f32(
    const ggml_tensor *src,
    ggml_gemmini_args_t &args,
    int8_t *dst,
    ActivationQuantConfig &cfg);

void reset_activation_quant_state(ggml_gemmini_args_t &args);

ActivationQuantConfig make_activation_quant_config(const ggml_gemmini_args_t &args);

void capture_activation_quant_result(
    ggml_gemmini_args_t &args,
    const ActivationQuantConfig &cfg,
    const ActivationQuantResult &res);


} // namespace ggml::gemmini::quants
