#pragma once

#include "../types.hpp"

#include <vector>

namespace ggml::npu::quants::act::tensor {

struct Meta {
    float              scale = 1.0f;
    RmdPacketList      rmd_packets;
    DirectResidualList direct_residuals;

    inline void reset() {
        scale = 1.0f;
        rmd_packets.clear();
        direct_residuals.clear();
    }
};

} // namespace ggml::npu::quants::act::tensor
