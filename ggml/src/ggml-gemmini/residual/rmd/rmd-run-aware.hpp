#pragma once

#include "rmd-types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::rmd {

struct RunAwareRun {
    uint32_t original_block_id = 0;
    uint32_t original_global_k_begin = 0;
    uint32_t union_k_mask = 0;
    size_t compact_k_begin = 0;
    size_t compact_k_count = 0;
    std::vector<uint16_t> original_local_k;

    bool operator==(const RunAwareRun &other) const {
        return original_block_id == other.original_block_id &&
               original_global_k_begin == other.original_global_k_begin &&
               union_k_mask == other.union_k_mask &&
               compact_k_begin == other.compact_k_begin &&
               compact_k_count == other.compact_k_count &&
               original_local_k == other.original_local_k;
    }
};

struct RunAwareRow {
    uint8_t original_lane_id = 0;
    uint32_t source_row = 0;

    bool operator==(const RunAwareRow &other) const {
        return original_lane_id == other.original_lane_id &&
               source_row == other.source_row;
    }
};

// Fully materialized packet-level work. Activations and weights are row-major
// scalar values; carriers are run-major [run, N]. Geometry factors are the
// exact result of the production WS selector for the final M/N/K shape.
struct RunAwareRequest {
    uint8_t operand_bits = 0;
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    size_t original_k = 0;

    size_t stripe_id = 0;
    size_t source_row_begin = 0;
    size_t source_row_count = 0;

    size_t tile_i = 0;
    size_t tile_j = 0;
    size_t tile_k = 0;

    std::vector<RunAwareRun> runs;
    std::vector<RunAwareRow> rows;
    std::vector<int8_t> activations; // [M,K]
    std::vector<int32_t> weights;    // [K,N]
    std::vector<uint32_t> carriers;  // [run,N]

    bool empty() const { return runs.empty(); }
    bool operator==(const RunAwareRequest &other) const {
        return operand_bits == other.operand_bits && m == other.m &&
               n == other.n && k == other.k && original_k == other.original_k &&
               stripe_id == other.stripe_id &&
               source_row_begin == other.source_row_begin &&
               source_row_count == other.source_row_count &&
               tile_i == other.tile_i && tile_j == other.tile_j &&
               tile_k == other.tile_k && runs == other.runs &&
               rows == other.rows && activations == other.activations &&
               weights == other.weights && carriers == other.carriers;
    }
};

// Transactional: on failure, `out` is unchanged. A null packet is the empty
// residual contract and returns success without invoking the geometry selector.
RmdStatus build_run_aware_request(const ggml_gemmini_args_t &args,
                                  const StripePacketHandle &packet,
                                  RunAwareRequest &out);
RmdStatus build_run_aware_request(const ggml_gemmini_args_t &args,
                                  const StripePacket &packet,
                                  RunAwareRequest &out);

} // namespace ggml::gemmini::rmd
