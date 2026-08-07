#pragma once

#include "../types.hpp"

#include <vector>

namespace ggml::gemmini::quants::act::tensor
{

struct Meta
{
    float scale = 1.0f;
    RmdPacketList rmd_packets;

    inline void reset() {
        scale = 1.0f;
        rmd_packets.clear();
    }
};

}
