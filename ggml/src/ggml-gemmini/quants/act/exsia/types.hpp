#pragma once

#include "../types.hpp"
#include "../../../ggml-gemmini-config.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#ifndef GGML_GEMMINI_EXSIA_SIGMA
#define GGML_GEMMINI_EXSIA_SIGMA 2
#endif

namespace ggml::gemmini::quants::act::exsia
{

struct Meta
{
    int16_t e_s = std::numeric_limits<int16_t>::min();
    int16_t rho = config::GGML_GEMMINI_ACTIVATION_RHO;
    int32_t sigma = GGML_GEMMINI_EXSIA_SIGMA;
    std::vector<int16_t> theta;
    RmdPacketList rmd_packets;
    DirectResidualList direct_residuals;

    void reset()
    {
        e_s = std::numeric_limits<int16_t>::min();
        rho = config::GGML_GEMMINI_ACTIVATION_RHO;
        sigma = GGML_GEMMINI_EXSIA_SIGMA;
        theta.clear();
        rmd_packets.clear();
        direct_residuals.clear();
    }

    int16_t resolve_stripe_theta(int stripe_idx) const
    {
        if (stripe_idx < 0 || static_cast<size_t>(stripe_idx) >= theta.size())
            return std::numeric_limits<int16_t>::min();

        return theta[static_cast<size_t>(stripe_idx)];
    }
};
}
