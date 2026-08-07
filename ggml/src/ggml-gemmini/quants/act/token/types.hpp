#pragma once

#include "../types.hpp"

#include <vector>

namespace ggml::gemmini::quants::act::token
{

struct Meta
{
    std::vector<float> scales;
    RmdPacketList rmd_packets;

    inline void reset() {
        scales.clear();
        rmd_packets.clear();
    }
};

}
