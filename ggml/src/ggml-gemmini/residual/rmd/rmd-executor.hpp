#pragma once

#include "rmd-types.hpp"

struct ggml_gemmini_args_t;

namespace ggml::gemmini::rmd {

// Raw NPU tile result. The physical tile order is an executor-internal detail: the
// assembler below normalises it into the canonical compressed output layout before
// anything outside the executor can observe it.
struct PhysicalTile {
    uint32_t packet_block_index = 0;
    uint8_t lane_position = 0;
    uint8_t lane_id = 0;
    uint32_t m_tile = 0;
    uint32_t j_tile = 0;
    uint16_t valid_rows = 0;
    uint16_t valid_cols = 0;
    const OutputValue * values = nullptr; // kArrayDim * kArrayDim, row major
};

// Accepts physical tiles in any order and rejects duplicates, missing tiles and
// out-of-range tags.
class RmdOutputAssembler {
public:
    RmdStatus begin(const StripePacket & packet, CompressedOutput & output);
    RmdStatus submit(const PhysicalTile & tile);
    RmdStatus finish();

private:
    const StripePacket * packet_ = nullptr;
    CompressedOutput * output_ = nullptr;
    std::vector<uint8_t> seen_;
    std::vector<size_t> tile_offset_;   // per block: index of its first tile slot
    size_t m_tiles_ = 0;
    size_t j_tiles_ = 0;
    size_t expected_ = 0;
    size_t submitted_ = 0;
};

struct RmdExecutionMetrics {
    size_t active_blocks = 0;
    size_t active_lanes = 0;
    size_t compact_k_count = 0;
    size_t padded_k_count = 0;
    size_t physical_tile_count = 0;
    size_t packet_bytes = 0;
    size_t compressed_output_values = 0;
    size_t block_padding_zeros = 0;
    size_t row_padding_zeros = 0;
    size_t j_padding_zeros = 0;
};

void collect_packet_metrics(const StripePacket & packet, RmdExecutionMetrics & metrics);

// Runs every block of the packet in ascending original block id, applies the block
// integer scale exactly once, and writes the canonical block-scaled INT64 output.
RmdStatus execute_rmd_stripe(const ggml_gemmini_args_t & args,
                             const StripePacket & packet,
                             CompressedOutput & output,
                             RmdExecutionMetrics * metrics = nullptr);

// True when the weight route can express its scale as
// integer_block_scale(j, block) * column_scale(j).
bool weight_route_supports_rmd(const ggml_gemmini_args_t & args);

}
