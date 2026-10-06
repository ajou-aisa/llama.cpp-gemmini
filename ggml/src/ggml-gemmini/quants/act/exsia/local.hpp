#pragma once

#include "types.hpp"
#include "exsia-state.hpp"
#include "exsia-profile.hpp"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace ggml::gemmini::quants::act::exsia {
class ExpScanner {
  public:
    int16_t unbiased_exp(const float & x);
    void    scan_top2_exp(const std::vector<float> & x,
                          BlockState & blk); // scan top-2 distinct exponents for a block and store
                                             // them in the block state
    void scan_top2_exp(const float * x, size_t count, BlockState & blk);

    void update_block_top2_exp(const BlockMask & mask, BlockState & blk);

    // update the stripe-level top-2 distinct exponents based on a block's max exponent
    void update_stripe_top2_exp(StripeState & stripe, int16_t e_b);
};

#if EXSIA_VALIDATION
void   reset_validation_block_top2_exp_rescan_count();
size_t validation_block_top2_exp_rescan_count();
#endif

class WideQuantizer {
  public:
    std::vector<int32_t> quantize_block(const std::vector<float> & x, int16_t theta_b); //
    void
    quantize_block(const std::vector<float> & x, int16_t theta_b, std::vector<int32_t> & q) const;
    std::tuple<std::vector<int32_t>, __int128_t, __int128_t>
         quantize_block(const std::vector<float> & x,
                        size_t                     row,
                        size_t                     col,
                        const BitMask &            mask,
                        int16_t theta_b); // returns q, sum(|q|), and sum(|q|^2) for inliers
    void quantize_block(const std::vector<float> & x,
                        size_t                     row,
                        size_t                     col,
                        const BitMask &            mask,
                        int16_t                    theta_b,
                        std::vector<int32_t> &     q,
                        __int128_t &               S,
                        __int128_t &               SS) const;
    void quantize_block(const std::vector<float> & x,
                        const BlockMask &          mask,
                        int16_t                    theta_b,
                        std::vector<int32_t> &     q,
                        __int128_t &               S,
                        __int128_t &               SS) const;
    void quantize_block(const float *          x,
                        size_t                 count,
                        const BlockMask &      mask,
                        int16_t                theta_b,
                        std::vector<int32_t> & q,
                        __int128_t &           S,
                        __int128_t &           SS) const;
    void
    quantize_block(const float * x, size_t count, int16_t theta_b, std::vector<int32_t> & q) const;
};

class SigmaDetector {
  public:
    struct SigmaContext {
        __int128_t n         = 0;
        __int128_t S         = 0;
        __int128_t threshold = 0;
        bool       valid     = false;
    };

    SigmaContext prepare(__int128_t S, __int128_t SS, size_t N) const;
    bool         detect(int32_t q, const SigmaContext & context) const;

    // Detect a one-sided upper-tail outlier from magnitude statistics.
    bool detect_sigma(int32_t q, __int128_t S, __int128_t SS, size_t N);
};

class LocalStage {
  public:
    // Ablation regenerates final codes after the unchanged selection and scale decisions.
    void set_force_recompute(bool enabled) {
        force_recompute_ = enabled;
    }
    // q_out addresses one caller-owned, block-disjoint slot q_wide range and never aliases x.
    bool run_optimized(Meta &          meta,
                       ExSIAState &    state,
                       const float *   x,
                       size_t          valid_count,
                       size_t          block_size,
                       size_t          local_row,
                       size_t          blk_idx,
                       StripeScratch & scratch,
                       BlockMask &     block_mask,
                       int32_t *       q_out,
                       int16_t &       block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
                       ,
                       LocalBlockCycleSample & cycle_sample);
#else
    );
#endif

#if EXSIA_VALIDATION
    // Frozen oracle is validation-only and never enters production builds.
    bool run_reference(Meta &                     meta,
                       ExSIAState &               state,
                       const std::vector<float> & x,
                       size_t                     local_row,
                       size_t                     blk_idx,
                       StripeScratch &            scratch,
                       BlockMask &                block_mask,
                       std::vector<int32_t> &     stripe_q_wide,
                       std::vector<int16_t> &     stripe_block_exp,
                       LocalBlockCycleSample &    cycle_sample);
#endif

  private:
    bool run_optimized_full(Meta &          meta,
                            const float *   x,
                            size_t          block_size,
                            StripeScratch & scratch,
                            BlockMask &     block_mask,
                            int32_t *       q_out,
                            int16_t &       block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
                            ,
                            LocalBlockCycleSample & cycle_sample);
#else
    );
#endif
    bool run_optimized_partial(Meta &          meta,
                               const float *   x,
                               size_t          valid_count,
                               size_t          block_size,
                               StripeScratch & scratch,
                               BlockMask &     block_mask,
                               int32_t *       q_out,
                               int16_t &       block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
                               ,
                               LocalBlockCycleSample & cycle_sample);
#else
    );
#endif

    ExpScanner    unit_exp_;
    WideQuantizer unit_quant_;
    SigmaDetector unit_sigma_;
    bool          force_recompute_ = false;
};

} // namespace ggml::gemmini::quants::act::exsia
