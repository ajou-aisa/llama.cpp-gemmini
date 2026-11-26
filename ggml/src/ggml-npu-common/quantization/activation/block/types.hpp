#pragma once

#include "../types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ggml::npu::quants::act::block {

inline constexpr size_t kGroupSize = ggml::npu::rmd::kNativeWeightScaleGroup;
static_assert(kGroupSize == 32);

struct Meta {
    size_t                  rows = 0;
    size_t                  cols = 0;
    std::vector<float>      scales;
    RmdPacketList           rmd_packets;
    DirectResidualList      direct_residuals;
    std::optional<uint64_t> run_id;

    inline void reset() {
        rows = 0;
        cols = 0;
        scales.clear();
        rmd_packets.clear();
        direct_residuals.clear();
        run_id.reset();
    }
};

} // namespace ggml::npu::quants::act::block
