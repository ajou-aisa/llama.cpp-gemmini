#pragma once

#include "rmd-types.hpp"

#include <map>
#include <set>

namespace ggml::gemmini::rmd {

// Incremental packet builder. ExSIA Stripe Folding calls add_residual() the moment a
// final residual value exists, so no INT32 residual plane or outlier list is ever
// materialised for the compensation path.
class RmdStripeBuilder {
public:
    RmdStripeBuilder() = default;

    void reset(size_t stripe_id, size_t row_begin, size_t row_count,
               size_t logical_k, size_t logical_j);
    void reset(size_t stripe_id, size_t row_begin, size_t row_count,
               size_t logical_k, size_t logical_j, uint8_t digit_bits);

    bool add_residual(size_t local_row, size_t original_k, int32_t residual);

    bool empty() const { return entries_.empty(); }
    RmdStatus status() const { return status_; }

    // Returns nullptr when the stripe carries no residual or the builder failed.
    StripePacketHandle finish();

private:
    struct DigitEntry {
        uint32_t block_id;
        uint32_t local_row;
        uint16_t block_local_k;
        uint8_t lane;
        int32_t digit;
    };

    struct BlockAccum {
        std::set<uint16_t> k;
        uint8_t lane_mask = 0;
    };

    RmdStatus status_ = RmdStatus::success;
    size_t stripe_id_ = 0;
    size_t row_begin_ = 0;
    size_t row_count_ = 0;
    size_t logical_k_ = 0;
    size_t logical_j_ = 0;
    uint8_t digit_bits_ = 8;
    size_t residual_event_count_ = 0;
    std::vector<DigitEntry> entries_;
    std::map<uint32_t, BlockAccum> blocks_;
};

// Validates every structural invariant of a finished packet. Used by finish(), by the
// executor, and by tests that construct packets by hand.
RmdStatus validate_packet(const StripePacket & packet);

// Reads one decoded signed digit. On failure, `digit` is unchanged.
RmdStatus read_packet_digit(const StripePacket & packet,
                            const BlockDescriptor & block,
                            uint8_t lane_position,
                            size_t row,
                            size_t k,
                            int32_t & digit);

// Rebuilds a packet restricted to [row_begin, row_end) out of one or more packets that
// may use a different stripe granularity. Used by the sequential stripe mode, where the
// matmul row slicing does not have to match the ExSIA stripe slicing.
StripePacketHandle slice_packets(const std::vector<StripePacketHandle> & packets,
                                 size_t row_begin,
                                 size_t row_end,
                                 size_t stripe_id,
                                 RmdStatus & status);

}
