#pragma once

#include "types.hpp"
#include "exsia-state.hpp"
#include "exsia-profile.hpp"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

struct npu_matmul_args;

namespace ggml::npu::quants::act::exsia {
class OutlierMarker {
  public:
    // mark columns [col_begin, col_end) of a stripe row as outliers (compensation targets) with
    // whole-word writes; used for blocks above the anchor exponent, whose valid columns all compensate
    void mark_valid_range(StripeState & stripe, size_t row, size_t col_begin, size_t col_end) const;
};

class ResidualClipper {
  public:
    // decompose == false: saturate to the activation range, residual is 0.
    // decompose == true: split the folded value u into signed n-bit limbs and return
    // {L0, u - L0}; the caller keeps L0 in the dense operand and stacks u - L0 as residual.
    std::pair<int32_t, int32_t> clip_with_residual(int32_t q, bool decompose);
};

class StripeFolding {
  public:
    bool
    run(Meta &                       meta,
        ExSIAState &                 state,
        StripeState &                stripe,
        npu_matmul_args &        args,
        size_t                       stripe_idx,
        const std::vector<int32_t> & stripe_q_wide,
        const std::vector<int16_t> & stripe_block_exp,
        std::vector<int32_t> &       residual, // dense global output, I * K_padded
        ggml::npu::residual::TimedResidualCapture & rmd_builder); // route-specific stripe sink

  private:
    bool run_blocks(const ExSIAState &                              state,
                    StripeState &                                   stripe,
                    npu_matmul_args &                           args,
                    const std::vector<int32_t> &                    stripe_q_wide,
                    const std::vector<int16_t> &                    stripe_block_exp,
                    std::vector<int32_t> &                          residual,
                    ggml::npu::residual::TimedResidualCapture & rmd_builder);
    template <typename T>
    bool run_blocks_typed(const ExSIAState &                              state,
                          StripeState &                                   stripe,
                          npu_matmul_args &                           args,
                          const std::vector<int32_t> &                    stripe_q_wide,
                          const std::vector<int16_t> &                    stripe_block_exp,
                          std::vector<int32_t> &                          residual,
                          ggml::npu::residual::TimedResidualCapture & rmd_builder);

    OutlierMarker   unit_outlier_;
    ResidualClipper unit_clip_;
};

} // namespace ggml::npu::quants::act::exsia
