#pragma once

#include "../types.hpp"

#include <vector>

namespace ggml::gemmini::quants::act::stripe
{

struct Meta
{
    std::vector<float> scales;
    RmdPacketList rmd_packets;
    DirectResidualList direct_residuals;

    inline void reset() {
        scales.clear();
        rmd_packets.clear();
        direct_residuals.clear();
    }
};

}