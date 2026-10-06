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
                             size_t                        stride);
    bool               emit(size_t row, size_t k, int32_t residual);
    StripePacketHandle finish();
    bool               empty() const noexcept {
        return digits_.empty();
    }
    RmdStatus status() const noexcept {
        return status_;
    }

  private:
    const std::vector<uint64_t> * selection_ = nullptr;
    size_t                        stride_ = 0, cursor_ = 0, row_words_ = 0;
    std::vector<uint64_t>         nonzero_mask_;
    std::vector<uint16_t>         limb_masks_;
    std::vector<int8_t>           digits_;
    std::vector<std::array<uint32_t, kMaxNativeRadixLanes>> block_lane_masks_;
    std::vector<uint64_t>                                   lane_rows_;
    std::vector<uint8_t>                                    row_seen_;
    StripePacket                                            metadata_;
    RmdStatus                                               status_ = RmdStatus::invalid_arguments;
};

} // namespace ggml::gemmini::rmd
