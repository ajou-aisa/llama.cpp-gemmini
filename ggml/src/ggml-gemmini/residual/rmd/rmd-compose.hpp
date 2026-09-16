#pragma once

#include "rmd-types.hpp"

struct ggml_gemmini_args_t;

namespace ggml::gemmini::rmd {
namespace detail {
class RmdWeightPreparation;
// The immutable packet must already have passed execution validation.
RmdStatus merge_rmd_correction_with_weights(const ggml_gemmini_args_t & args,
    float * destination, const StripePacket & packet, const Correction & correction,
    RmdWeightPreparation & weights, size_t * nonzero_count = nullptr);
}


// Radix composition of the canonical block-scaled INT64 output.
//
//     correction[row][j] = sum over blocks, lanes of
//         output[block][lane position][row][j] * radix(digit_bits) ^ lane_id
//
// Reconstruction uses checked integer Horner steps for radix 16, 256, or 65536.
//
// The block scale is NOT re-applied here; the executor already did it. Original K
// indices are not used: they only exist for input compaction and weight gather.
RmdStatus compose_rmd_output(const StripePacket & packet,
                             const CompressedOutput & output,
                             Correction & correction); // row_count * logical_j

RmdStatus compose_block_rmd_output(const ggml_gemmini_args_t & args,
                                   const StripePacket & packet,
                                   const CompressedOutput & output,
                                   Correction & correction);

// Applies any remaining scales required by the correction's tagged domain, then
// commits the fully staged result. Fully scaled BLOCK corrections are added directly.
// The output and optional raw correction nonzero count are unchanged on every failure.
RmdStatus merge_rmd_correction_to(const ggml_gemmini_args_t & args,
                                  float * destination,
                                  size_t global_row_begin,
                                  size_t global_row_end,
                                  const Correction & correction,
                                  size_t * nonzero_count = nullptr);

RmdStatus merge_rmd_correction(const ggml_gemmini_args_t & args,
                               size_t global_row_begin,
                               size_t global_row_end,
                               const Correction & correction,
                               size_t * nonzero_count = nullptr);

// The weight-stationary packet path preserves packet-scoped weight validation, then
// delegates scaling and atomic output update to the common checked implementation.
RmdStatus merge_rmd_correction_to(const ggml_gemmini_args_t & args,
                                  float * destination,
                                  const StripePacket & packet,
                                  const Correction & correction,
                                  size_t * nonzero_count = nullptr);

RmdStatus merge_rmd_correction(const ggml_gemmini_args_t & args,
                               const StripePacket & packet,
                               const Correction & correction,
                               size_t * nonzero_count = nullptr);

// Rebuilds the dense INT32 residual plane carried by valid width-native stripe packets.
// Only the activation dequantizers (validation / FLOAT parity) need this; the
// compensation path never materialises a residual plane. Publication is transactional.
RmdStatus expand_packets_to_plane(
    const std::vector<StripePacketHandle> & packets,
    size_t global_row_begin,
    size_t global_row_end,
    size_t col_count,
    std::vector<int32_t> & plane);

void expand_packets_to_plane(const std::vector<StripePacketHandle> & packets,
                             size_t row_count,
                             size_t col_count,
                             std::vector<int32_t> & plane);

}
