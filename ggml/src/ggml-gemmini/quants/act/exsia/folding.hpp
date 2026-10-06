#pragma once

#include "types.hpp"
#include "exsia-state.hpp"
#include "exsia-profile.hpp"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::act::exsia {
class OutlierMarker {
  public:
    // mark outliers in a stripe based on the final stripe scale exponent and a bitmask, setting the
    // corresponding bits in the bitmask for outlier positions
    void mark_outlier(StripeState &   stripe,
                      size_t          row,
                      size_t          blk_idx,
                      size_t          blk_size,
                      const BitMask & d_mask) const;
};

class ResidualClipper {
  public:
    std::pair<int32_t, int32_t> clip_with_residual(int32_t q);
};

class StripeFolding {
  public:
    bool
    run(Meta &                       meta,
        ExSIAState &                 state,
        StripeState &                stripe,
        ggml_gemmini_args_t &        args,
        size_t                       stripe_idx,
        const std::vector<int32_t> & stripe_q_wide,
        const std::vector<int16_t> & stripe_block_exp,
        std::vector<int32_t> &       residual, // dense global output, I * K_padded
        ggml::gemmini::residual::TimedResidualCapture & rmd_builder); // route-specific stripe sink

  private:
    OutlierMarker   unit_outlier_;
    ResidualClipper unit_clip_;
};

} // namespace ggml::gemmini::quants::act::exsia
