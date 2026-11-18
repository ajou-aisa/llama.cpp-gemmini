#pragma once

#include "rmd-types.hpp"

namespace ggml::gemmini::rmd {

class RmdBitmapBuilder {
  public:
    // Borrow the selection mask during emission, in row/K order. Zero residuals
    // may be omitted; finish uses an owned bitmap of emitted nonzero positions.
    void               reset(size_t                        stripe_id,
                             size_t                        row_begin,
                             size_t                        rows,
                             size_t                        k,
                             size_t                        j,
                             uint8_t                       bits,
                             const std::vector<uint64_t> & selection,
                             size_t                        stride,
                             bool                          reuse_completed_packet = false);
    bool               emit(size_t row, size_t k, int32_t residual);
    StripePacketHandle finish();
    bool               empty() const noexcept {
        return digit_count_ == 0;
    }
    RmdStatus status() const noexcept {
        return status_;
    }

  private:
    const std::vector<uint64_t> * selection_ = nullptr;
    size_t                        stride_ = 0, cursor_ = 0, row_words_ = 0;
    std::vector<uint64_t>         nonzero_mask_;
    std::vector<size_t>           nonzero_words_;
    std::vector<uint16_t>         limb_masks_;
    std::vector<int8_t>           digits_; // storage; only [0, digit_count_) is live
    size_t                        digit_count_ = 0;
    std::vector<std::array<KMask, kMaxNativeRadixLanes>> block_lane_masks_;
    std::vector<uint64_t>                                   lane_rows_;
    std::vector<uint8_t>                                    row_seen_;
    std::vector<uint32_t>                                   block_indices_, row_ranks_;
    std::vector<std::array<uint8_t, kMaxNativeRadixLanes>>    lane_groups_;
    StripePacket                                            metadata_;
    bool                                                    reuse_completed_packet_ = false;
    std::shared_ptr<StripePacket>                            cached_packet_;
    std::vector<BlockDescriptor>                             recycled_blocks_;
    RmdStatus                                               status_ = RmdStatus::invalid_arguments;
};

} // namespace ggml::gemmini::rmd
