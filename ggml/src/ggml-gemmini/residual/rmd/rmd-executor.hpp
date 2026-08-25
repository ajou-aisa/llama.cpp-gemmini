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

#if defined(GGML_GEMMINI_TESTING)
struct WsCallObservation {
    size_t rows = 0;
    size_t cols = 0;
    size_t k = 0;
    uint8_t lane_id = 0;
    elem_t first_activation = 0;
    elem_t first_weight = 0;
    int64_t raw_value = 0;
    size_t raw_nonzero_count = 0;
    uint64_t block_scale = 0;
    int64_t scaled_value = 0;
    int64_t compressed_value = 0;
    int64_t composed_value = 0;
};
#endif

struct RmdExecutionMetrics {
    size_t direct_event_count = 0;
    size_t direct_call_count = 0;
    size_t packet_call_count = 0;
    size_t ws_call_count = 0;
    size_t active_blocks = 0;
    size_t active_lanes = 0;
    size_t compact_k_count = 0;
    size_t padded_k_count = 0;
    size_t physical_tile_count = 0;
    size_t matmul_call_count = 0;
    size_t lane_group_count = 0;
    size_t baseline_stacked_i_tile_count = 0;
    size_t stacked_i_tile_count = 0;
    size_t packet_bytes = 0;
    size_t compressed_output_values = 0;
    size_t block_padding_zeros = 0;
    size_t row_padding_zeros = 0;
    size_t j_padding_zeros = 0;
    size_t weight_values_gathered = 0;
    size_t weight_baseline_address_resolutions = 0;
    size_t weight_address_resolutions = 0;
#if defined(GGML_GEMMINI_TESTING)
    std::vector<WsCallObservation> ws_observations;
    std::vector<int64_t> raw_lane_values;
#endif
};

void collect_packet_metrics(const StripePacket & packet, RmdExecutionMetrics & metrics);

// Executes every block of the compact packet, applies the block integer scale exactly
// once, and writes canonical block-scaled INT64 output. H1/HP1 routes use checked
// checked Rocket C++ lane dots; hardware compact routes use the normal Gemmini WS
// entry point and are unavailable on host builds.
RmdStatus execute_rmd_stripe_ws(const ggml_gemmini_args_t & args,
                                const StripePacket & packet,
                                CompressedOutput & output,
                                RmdExecutionMetrics * metrics = nullptr);

// True when the weight route can express its scale as
// integer_block_scale(j, block) * column_scale(j).
bool weight_route_supports_rmd(const ggml_gemmini_args_t & args);

#if defined(GGML_GEMMINI_TESTING)
// Scalar packet oracle for tests only. Production callers cannot select or invoke it.
RmdStatus execute_rmd_stripe_reference(const ggml_gemmini_args_t & args,
                                       const StripePacket & packet,
                                       CompressedOutput & output,
                                       RmdExecutionMetrics * metrics = nullptr);

// Instantiates the native Gemmini path in host test builds. Widened codes are
// preflighted against elem_t and fail before tiled_matmul or metric commit.
RmdStatus execute_rmd_stripe_gemmini_for_test(
    const ggml_gemmini_args_t & args,
    const StripePacket & packet,
    CompressedOutput & output,
    RmdExecutionMetrics * metrics = nullptr);

RmdStatus gather_weight_tile_for_test(const ggml_gemmini_args_t & args,
                                      uint32_t block_id,
                                      const uint16_t * local_k,
                                      size_t valid_k,
                                      size_t col_base,
                                      size_t valid_cols,
                                      elem_t * tile,
                                      size_t tile_stride,
                                      RmdExecutionMetrics * metrics = nullptr);

RmdStatus gather_wide_weight_tile_for_test(
    const ggml_gemmini_args_t & args,
    uint32_t block_id,
    const uint16_t * local_k,
    size_t valid_k,
    size_t col_base,
    size_t valid_cols,
    int32_t * tile,
    size_t tile_stride,
    RmdExecutionMetrics * metrics = nullptr);

RmdStatus repeat_weight_tile_gather_for_test(const ggml_gemmini_args_t & args,
                                             uint32_t block_count,
                                             const uint16_t * local_k,
                                             size_t valid_k,
                                             size_t col_base,
                                             size_t valid_cols,
                                             size_t iterations,
                                             uint64_t & checksum,
                                             RmdExecutionMetrics & metrics);

RmdStatus repeat_scalar_weight_tile_gather_for_test(const ggml_gemmini_args_t & args,
                                                    uint32_t block_count,
                                                    const uint16_t * local_k,
                                                    size_t valid_k,
                                                    size_t col_base,
                                                    size_t valid_cols,
                                                    size_t iterations,
                                                    uint64_t & checksum);
#endif

}
