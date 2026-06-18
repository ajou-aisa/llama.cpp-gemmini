#pragma once

#include "../types.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ggml::gemmini::quants::act::exsia
{

struct Meta
{
    int16_t e_s = std::numeric_limits<int16_t>::min();
    int16_t rho = 6;
    std::vector<int16_t> theta;
    std::vector<ggml_gemmini_qact_outlier> outliers;

    void reset()
    {
        e_s = std::numeric_limits<int16_t>::min();
        rho = 6;
        theta.clear();
        outliers.clear();
    }

    int16_t resolve_stripe_theta(int stripe_idx) const
    {
        if (stripe_idx < 0 || static_cast<size_t>(stripe_idx) >= theta.size()) {
            return e_s;
        }
        return theta[static_cast<size_t>(stripe_idx)];
    }
};
}
