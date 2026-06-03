// quantize.hpp
#pragma once

#include "ethos/types.hpp"
#include "ggml-gemmini-config.hpp"

struct ggml_tensor;
struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants
{
    using ActivationQuantConfig = act::ethos::Config;
    using ActivationQuantResult = act::ethos::Result;

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

// Tile-aware quantization: handles tile-row tensors with padded buffer allocation.
// Pads K to block_size alignment, allocates zero-initialized
// buffer, copies real data, and runs ethos with the padded view.
// Returns per-tile result with e_t_per_tile populated.
ActivationQuantResult quantize_activation_f32_tile(
    const ggml_tensor *src,
    ggml_gemmini_args_t &args,
    int8_t *dst,
    ActivationQuantConfig &cfg,
    int tile_row);

void reset_activation_quant_state(ggml_gemmini_args_t &args);

ActivationQuantConfig make_activation_quant_config(const ggml_gemmini_args_t &args);

void capture_activation_quant_result(
    ggml_gemmini_args_t &args,
    const ActivationQuantConfig &cfg,
    const ActivationQuantResult &res);

void ggml_gemmini_quantize_activation_tile(
    const ggml_tensor *src,
    ggml_gemmini_args_t &args,
    int8_t *dst,
    int tile_row,
    int tile_col);
} // namespace ggml::gemmini::quants
