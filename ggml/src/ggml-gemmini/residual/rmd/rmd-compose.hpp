#pragma once

#include "rmd-types.hpp"

struct ggml_gemmini_args_t;

namespace ggml::gemmini::rmd {

// Radix composition of the canonical block-scaled INT64 output.
//
//     correction[row][j] = sum over blocks, lanes of
//         output[block][lane position][row][j] * 256 ^ lane_id
//
// The block scale is NOT re-applied here; the executor already did it. Original K
// indices are not used: they only exist for input compaction and weight gather.
RmdStatus compose_rmd_output(const StripePacket & packet,
                             const CompressedOutput & output,
                             std::vector<OutputValue> & correction); // row_count * logical_j

// Applies the common per-column weight scale and per-row activation scale and adds the
// result into args.f_out. This is the only floating point step of the RMD path.
RmdStatus merge_rmd_correction(const ggml_gemmini_args_t & args,
                               const StripePacket & packet,
                               const std::vector<OutputValue> & correction);

// execute -> compose -> common scale / final merge, for callers that do not need to
// observe the intermediate compressed output.
RmdStatus apply_rmd_packet(const ggml_gemmini_args_t & args, const StripePacket & packet);

// Rebuilds the dense INT32 residual plane carried by a set of stripe packets. Only the
// activation dequantizers (validation / FLOAT parity) need this; the compensation path
// never materialises a residual plane.
void expand_packets_to_plane(const std::vector<StripePacketHandle> & packets,
                             size_t row_count,
                             size_t col_count,
                             std::vector<int32_t> & plane);

}
