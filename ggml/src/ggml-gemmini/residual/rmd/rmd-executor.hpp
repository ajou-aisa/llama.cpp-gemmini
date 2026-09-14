#pragma once

#include "rmd-types.hpp"
#include "../../quants/common/weight_route.hpp"

#include <array>
#include <limits>
#include <mutex>

struct ggml_gemmini_args_t;

namespace ggml::gemmini::rmd {
namespace detail { struct RmdAssemblerAccess; }

constexpr bool compact_rmd_backend_available(bool hardware_target,
                                             bool im2p_build) {
    return hardware_target || im2p_build;
}

// Raw NPU tile result. Physical tile order is executor-internal; the assembler
// validates tags before storing compressed output or composing the correction.
struct PhysicalTile {
    uint32_t packet_block_index = 0;
    uint8_t lane_position = 0;
    uint8_t lane_id = 0;
    uint32_t m_tile = 0;
    uint32_t j_tile = 0;
    uint16_t valid_rows = 0;
    uint16_t valid_cols = 0;
    const OutputValue * values = nullptr; // valid_rows rows, row stride kArrayDim
};

// Accepts physical tiles in any order and rejects duplicates, missing tiles and
// out-of-range tags.
class RmdOutputAssembler {
public:
    RmdStatus begin(const StripePacket & packet, CompressedOutput & output);
    RmdStatus begin(const StripePacket & packet, Correction & correction);
    RmdStatus submit(const PhysicalTile & tile);
    RmdStatus finish();

private:
    friend struct detail::RmdAssemblerAccess;
    RmdStatus begin_validated(const StripePacket & packet, CompressedOutput & output);
    RmdStatus begin_validated(const StripePacket & packet, Correction & correction);
    RmdStatus begin(const StripePacket & packet);
    const StripePacket * packet_ = nullptr;
    CompressedOutput * output_ = nullptr;
    Correction * correction_ = nullptr;
    std::vector<__int128> correction_values_;
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

struct RmdProviderStats {
    static constexpr size_t field_count = 41;
    std::array<uint64_t, field_count> fields{};

    [[nodiscard]] uint64_t work_total_cycles() const noexcept {
        return fields[0];
    }
    [[nodiscard]] uint64_t output_write_requests() const noexcept {
        return fields[4];
    }
};

inline RmdStatus checked_accumulate_provider_stats(
    RmdProviderStats &aggregate, const RmdProviderStats &value) noexcept {
    auto staged = aggregate;
    for (size_t index = 0; index < RmdProviderStats::field_count; ++index) {
        if (staged.fields[index] >
            std::numeric_limits<uint64_t>::max() - value.fields[index])
            return RmdStatus::overflow;
        staged.fields[index] += value.fields[index];
    }
    aggregate = staged;
    return RmdStatus::success;
}

struct RmdExecutionMetrics {
    size_t direct_event_count = 0;
    size_t direct_call_count = 0;
    size_t packet_call_count = 0;
    size_t ws_call_count = 0;
    size_t im2p_dot_calls = 0;
    size_t active_blocks = 0;
    size_t active_lanes = 0;
    size_t compact_k_count = 0;
    size_t padded_k_count = 0;
    size_t physical_tile_count = 0; // logical per-lane tiles submitted to the assembler
    size_t matmul_call_count = 0;
    size_t lane_group_count = 0;
    size_t baseline_stacked_i_tile_count = 0;
    size_t stacked_i_tile_count = 0; // dispatched group row tiles, including K/J repeats
    size_t packet_bytes = 0;
    size_t compressed_output_values = 0;
    // Per-axis padding counts overlap at cells padded in both rows and K.
    size_t block_padding_zeros = 0;
    size_t row_padding_zeros = 0;
    size_t j_padding_zeros = 0;
    size_t weight_values_gathered = 0;
    size_t weight_baseline_address_resolutions = 0;
    size_t weight_address_resolutions = 0;
    // Transactional aggregate from the independent residual simulator.
    RmdProviderStats im2p_stats{};
#if defined(GGML_GEMMINI_TESTING)
    std::vector<WsCallObservation> ws_observations;
    std::vector<int64_t> raw_lane_values;
#endif
};

void collect_packet_metrics(const StripePacket & packet, RmdExecutionMetrics & metrics);

namespace detail {
// One immutable matmul weight lifetime. Activation metadata may vary by stripe;
// changing weights or their shape requires a new context. Preparation alone locks.
class RmdWeightPreparation {
public:
    const quants::wroute::WeightRoutePlan & route_plan(const ggml_gemmini_args_t & args);
#if defined(GGML_GEMMINI_TESTING)
    size_t column_preparations() const { return column_preparations_; }
    size_t selected_block_preparations() const { return selected_block_preparations_; }
#endif

private:
    friend RmdStatus merge_rmd_correction_with_weights(const ggml_gemmini_args_t &,
        float *, const StripePacket &, const Correction &, RmdWeightPreparation &, size_t *);
    RmdStatus prepare_columns(const ggml_gemmini_args_t & args, const StripePacket & packet);
    std::mutex mutex_;
    bool plan_ready_ = false;
    bool columns_ready_ = false;
    quants::wroute::WeightRoutePlan plan_;
    std::vector<float> column_scale_;
    std::vector<uint8_t> checked_blocks_;
#if defined(GGML_GEMMINI_TESTING)
    size_t column_preparations_ = 0;
    size_t selected_block_preparations_ = 0;
#endif
};

RmdStatus execute_rmd_stripe_ws_with_weights(const ggml_gemmini_args_t & args,
    const StripePacket & packet, Correction & correction,
    RmdWeightPreparation & weights, RmdExecutionMetrics * metrics = nullptr);

}

// Executes every block of the compact packet, applies the block integer scale exactly
// once, and writes canonical block-scaled INT64 output. Matched IM2P_SIM H1/HP1
// routes use the typed provider executor; non-IM2P builds retain their existing
// hardware/host policy. The explicit checked-software oracle remains test-only.
RmdStatus execute_rmd_stripe_ws(const ggml_gemmini_args_t & args,
                                const StripePacket & packet,
                                CompressedOutput & output,
                                RmdExecutionMetrics * metrics = nullptr);

// Composes block-scaled tiles as they arrive and publishes the final INT64
// correction only after all tiles and overflow checks succeed.
RmdStatus execute_rmd_stripe_ws(const ggml_gemmini_args_t & args,
                                const StripePacket & packet,
                                Correction & correction,
                                RmdExecutionMetrics * metrics = nullptr);

#if defined(GGML_GEMMINI_TESTING)
// Scalar packet oracle for tests only. Production callers cannot select or invoke it.
RmdStatus execute_rmd_stripe_reference(const ggml_gemmini_args_t & args,
                                       const StripePacket & packet,
                                       CompressedOutput & output,
                                       RmdExecutionMetrics * metrics = nullptr);

RmdStatus execute_rmd_stripe_reference(const ggml_gemmini_args_t & args,
                                       const StripePacket & packet,
                                       Correction & correction,
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
