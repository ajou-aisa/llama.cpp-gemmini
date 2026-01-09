#pragma once

#include "ggml-gemmini-args.h"

// Activation tensor quantization helper shared across Gemmini helpers.
// Implementation lives in ggml-gemmini-quantize.cpp.
void ggml_gemmini_quantize_activation(const ggml_tensor *src,
                                      ggml_gemmini_args_t &args,
                                      int8_t *dst);
