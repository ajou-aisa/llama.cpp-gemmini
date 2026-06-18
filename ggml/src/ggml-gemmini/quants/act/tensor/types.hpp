#pragma once

namespace ggml::gemmini::quants::act::tensor
{

struct Meta
{
    float scale = 1.0f;

    inline void reset() {
        scale = 1.0f;
    }
};

}
