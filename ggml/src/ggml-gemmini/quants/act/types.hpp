#pragma once
#include <cstdint>
#include <vector>

#include "../../residual/rmd/rmd-types.hpp"

#ifndef GGML_GEMMINI_BLOCK_SIZE
#define GGML_GEMMINI_BLOCK_SIZE 32
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE GGML_GEMMINI_BLOCK_SIZE
#endif

namespace ggml::gemmini::quants::act
{
// Residual compensation payload produced by the quantizers: one RMD stripe packet per
// activation stripe. Empty stripes contribute no packet.
using RmdPacketList = std::vector<ggml::gemmini::rmd::StripePacketHandle>;
}
