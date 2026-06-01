#pragma once

namespace ggml::gemmini::quants
{
    struct QactOutlier
    {
        int row = 0;
        int col = 0;
        float original = 0.f;
        float saturated = 0.f;
    };

} // namespace ggml::gemmini::quants
