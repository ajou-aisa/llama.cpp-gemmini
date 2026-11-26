#pragma once
#include <cstdint>
#include <vector>

#include "../../residual/residual-capture.hpp"

#ifndef GGML_GEMMINI_BLOCK_SIZE
#define GGML_GEMMINI_BLOCK_SIZE 32
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE GGML_GEMMINI_BLOCK_SIZE
#endif

namespace ggml::npu::quants::act {
// Residual compensation payload produced by the quantizers: one RMD stripe packet per
// activation stripe. Empty stripes contribute no packet.
using RmdPacketList      = std::vector<ggml::npu::rmd::StripePacketHandle>;
using DirectResidualList = std::vector<ggml::npu::residual::DirectStripePayloadHandle>;
using ResidualRoute      = ggml::npu::residual::ResidualRoute;
} // namespace ggml::npu::quants::act
