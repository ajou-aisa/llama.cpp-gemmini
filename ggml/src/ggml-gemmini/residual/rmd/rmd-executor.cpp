#include "rmd-executor.hpp"

#include "rmd-builder.hpp"

#include "../../ggml-gemmini-args.h"
#include "../../quants/common/weight_reader.hpp"
#include "../../quants/common/weight_route.hpp"

#include <gemmini.h>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace ggml::gemmini::rmd {

static_assert(kNativeWeightScaleGroup == QK8_0);

namespace {

namespace wreader = quants::wreader;
namespace wroute = quants::wroute;

bool checked_mul_i64(int64_t lhs, int64_t rhs, int64_t & out) {
    const __int128 product = static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
    if (product > static_cast<__int128>(std::numeric_limits<int64_t>::max()) ||
        product < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
        return false;
    }
    out = static_cast<int64_t>(product);
    return true;
}

bool checked_add_i64(int64_t lhs, int64_t rhs, int64_t & out) {
    const __int128 sum = static_cast<__int128>(lhs) + static_cast<__int128>(rhs);
    if (sum > static_cast<__int128>(std::numeric_limits<int64_t>::max()) ||
        sum < static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
        return false;
    }
    out = static_cast<int64_t>(sum);
    return true;
}

struct WeightGatherCounts {
    size_t values = 0;
    size_t baseline_address_resolutions = 0;
    size_t address_resolutions = 0;
};

class WeightGather {
public:
    WeightGather(const ggml_gemmini_args_t & args, const wroute::WeightRoutePlan & plan)
        : args_(args), plan_(plan) {}

    bool valid() const {
        return plan_.valid &&
            wroute::weight_route_status(plan_, wroute::WeightExecutionPath::Compact) ==
                wroute::WeightRouteStatus::Success;
    }

    template <typename TileElement>
    RmdStatus fill_tile(uint32_t block_id,
                        const uint16_t * local_k,
                        size_t valid_k,
                        size_t col_base,
                        size_t valid_cols,
                        TileElement * tile,
                        size_t tile_stride,
                        WeightGatherCounts & counts) const {
        static_assert(std::is_integral_v<TileElement> && std::is_signed_v<TileElement>,
                      "compact tiles require signed integer elements");
        if (!valid() || local_k == nullptr || tile == nullptr ||
            valid_k == 0 || valid_k > kArrayDim ||
            valid_cols == 0 || valid_cols > kArrayDim || tile_stride < valid_cols ||
            col_base > args_.J || valid_cols > args_.J - col_base ||
            args_.K == 0 || static_cast<size_t>(block_id) > (args_.K - 1) / kBlockSize) {
            return RmdStatus::execution_failed;
        }

        std::array<size_t, kArrayDim> global_k{};
        for (size_t k = 0; k < valid_k; ++k) {
            if (local_k[k] >= kBlockSize) {
                return RmdStatus::execution_failed;
            }
            global_k[k] = static_cast<size_t>(block_id) * kBlockSize + local_k[k];
            if (global_k[k] >= args_.K) {
                return RmdStatus::execution_failed;
            }
        }

        std::array<TileElement, kArrayDim * kArrayDim> staged{};
        for (size_t k = 0; k < valid_k; ++k) {
            for (size_t col = 0; col < valid_cols; ++col) {
                const wreader::WeightCodeResult code = wreader::read_code(
                    args_, plan_, col_base + col, global_k[k]);
                if (!code.ok()) {
                    return RmdStatus::execution_failed;
                }
                if (code.value < static_cast<int64_t>(
                        std::numeric_limits<TileElement>::min()) ||
                    code.value > static_cast<int64_t>(
                        std::numeric_limits<TileElement>::max())) {
                    return RmdStatus::overflow;
                }
                staged[k * kArrayDim + col] = static_cast<TileElement>(code.value);
            }
        }
        for (size_t k = 0; k < valid_k; ++k) {
            std::copy_n(staged.data() + k * kArrayDim, valid_cols,
                        tile + k * tile_stride);
        }

        const bool column_addressed = plan_.native_weight_blocks ||
            plan_.layout == wroute::WeightLayout::JxK_ColMajor;
        counts.values = valid_k * valid_cols;
        counts.baseline_address_resolutions = counts.values;
        counts.address_resolutions = column_addressed ? valid_cols : valid_k;
        return RmdStatus::success;
    }

    RmdStatus validate_native_packet(const StripePacket & packet) const {
        std::array<elem_t, kArrayDim * kArrayDim> tile{};
        for (const BlockDescriptor & block : packet.blocks) {
            for (size_t k_base = 0; k_base < block.compact_k_count;
                 k_base += kArrayDim) {
                const size_t valid_k = std::min(
                    kArrayDim, static_cast<size_t>(block.compact_k_count) - k_base);
                const uint16_t * local_k = packet.k_indices.data() +
                    block.k_index_offset + k_base;
                for (size_t col_base = 0; col_base < packet.logical_j;
                     col_base += kArrayDim) {
                    const size_t valid_cols = std::min(
                        kArrayDim, packet.logical_j - col_base);
                    WeightGatherCounts ignored{};
                    const RmdStatus status = fill_tile(
                        block.block_id, local_k, valid_k, col_base, valid_cols,
                        tile.data(), kArrayDim, ignored);
                    if (status != RmdStatus::success) {
                        return status;
                    }
                }
            }
        }
        return RmdStatus::success;
    }

private:
    const ggml_gemmini_args_t & args_;
    const wroute::WeightRoutePlan & plan_;
};

struct LaneGroup {
    std::vector<uint8_t> lane_positions;
    std::vector<uint16_t> compact_positions;
    std::vector<elem_t> activation;
    size_t padded_k_count = 0;
};

enum class CompactExecutorBackend : uint8_t {
    gemmini_ws,
    software_ws,
};

size_t count_bits(uint32_t value) {
    size_t count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
}

void choose_lane_partition(const std::array<uint32_t, kMaxLanes> & lane_support,
                           uint8_t lane_count,
                           size_t m_tiles,
                           uint8_t lane_position,
                           uint8_t group_count,
                           std::array<uint8_t, kMaxLanes> & assignment,
                           std::array<uint8_t, kMaxLanes> & best_assignment,
                           size_t baseline_calls,
                           size_t & best_i_tiles) {
    if (lane_position == lane_count) {
        std::array<uint32_t, kMaxLanes> group_support{};
        std::array<size_t, kMaxLanes> group_lanes{};
        for (uint8_t lane = 0; lane < lane_count; ++lane) {
            group_support[assignment[lane]] |= lane_support[lane];
            ++group_lanes[assignment[lane]];
        }
        size_t calls = 0;
        size_t i_tiles = 0;
        for (uint8_t group = 0; group < group_count; ++group) {
            const size_t k_tiles =
                (count_bits(group_support[group]) + kArrayDim - 1) / kArrayDim;
            calls += k_tiles;
            i_tiles += k_tiles * group_lanes[group] * m_tiles;
        }
        if (calls <= baseline_calls && i_tiles < best_i_tiles) {
            best_i_tiles = i_tiles;
            best_assignment = assignment;
        }
        return;
    }

    for (uint8_t group = 0; group <= group_count; ++group) {
        assignment[lane_position] = group;
        choose_lane_partition(lane_support, lane_count, m_tiles,
                              static_cast<uint8_t>(lane_position + 1),
                              static_cast<uint8_t>(group_count + (group == group_count)),
                              assignment, best_assignment, baseline_calls, best_i_tiles);
    }
}

void build_lane_groups(const StripePacket & packet,
                       const BlockDescriptor & block,
                       size_t m_tiles,
                       std::vector<LaneGroup> & groups) {
    std::array<uint32_t, kMaxLanes> lane_support{};
    for (uint8_t lane = 0; lane < block.active_lane_count; ++lane) {
        const size_t lane_base = block.activation_offset +
            static_cast<size_t>(lane) * block.rows_padded * block.padded_k_count;
        for (size_t row = 0; row < packet.row_count; ++row) {
            const int8_t * source = packet.stacked_activation.data() +
                lane_base + row * block.padded_k_count;
            for (uint16_t k = 0; k < block.compact_k_count; ++k) {
                if (source[k] != 0) {
                    lane_support[lane] |= uint32_t{1} << k;
                }
            }
        }
    }

    std::array<uint8_t, kMaxLanes> assignment{};
    std::array<uint8_t, kMaxLanes> best_assignment{};
    const size_t baseline_calls = block.padded_k_count / kArrayDim;
    size_t best_i_tiles = baseline_calls * block.active_lane_count * m_tiles;
    choose_lane_partition(lane_support, block.active_lane_count, m_tiles, 1, 1,
                          assignment, best_assignment, baseline_calls, best_i_tiles);

    const uint8_t group_count = static_cast<uint8_t>(
        *std::max_element(best_assignment.begin(),
                          best_assignment.begin() + block.active_lane_count) + 1);
    groups.assign(group_count, LaneGroup{});
    for (uint8_t lane = 0; lane < block.active_lane_count; ++lane) {
        groups[best_assignment[lane]].lane_positions.push_back(lane);
    }
    for (LaneGroup & group : groups) {
        uint32_t support = 0;
        for (uint8_t lane : group.lane_positions) {
            support |= lane_support[lane];
        }
        for (uint16_t k = 0; k < block.compact_k_count; ++k) {
            if ((support & (uint32_t{1} << k)) != 0) {
                group.compact_positions.push_back(k);
            }
        }
        group.padded_k_count = align_up(group.compact_positions.size(), kArrayDim);
        group.activation.assign(group.lane_positions.size() * block.rows_padded *
                                group.padded_k_count, elem_t{0});
        for (size_t group_lane = 0; group_lane < group.lane_positions.size(); ++group_lane) {
            const uint8_t packet_lane = group.lane_positions[group_lane];
            const size_t source_base = block.activation_offset +
                static_cast<size_t>(packet_lane) * block.rows_padded * block.padded_k_count;
            const size_t destination_base = group_lane * block.rows_padded * group.padded_k_count;
            for (size_t row = 0; row < packet.row_count; ++row) {
                for (size_t k = 0; k < group.compact_positions.size(); ++k) {
                    group.activation[destination_base + row * group.padded_k_count + k] =
                        static_cast<elem_t>(packet.stacked_activation[
                            source_base + row * block.padded_k_count +
                            group.compact_positions[k]]);
                }
            }
        }
    }
}

}

bool weight_route_supports_rmd(const ggml_gemmini_args_t & args) {
    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    return plan.valid && wroute::route_supports_integer_block_scale(plan);
}

#if defined(GGML_GEMMINI_TESTING)
RmdStatus gather_weight_tile_for_test(const ggml_gemmini_args_t & args,
                                      uint32_t block_id,
                                      const uint16_t * local_k,
                                      size_t valid_k,
                                      size_t col_base,
                                      size_t valid_cols,
                                      elem_t * tile,
                                      size_t tile_stride,
                                      RmdExecutionMetrics * metrics) {
    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid || !wroute::route_supports_integer_block_scale(plan)) {
        return RmdStatus::unsupported_route;
    }
    const WeightGather weights(args, plan);
    if (!weights.valid()) {
        return RmdStatus::unsupported_route;
    }
    WeightGatherCounts counts{};
    const RmdStatus gather_status = weights.fill_tile(
        block_id, local_k, valid_k, col_base, valid_cols,
        tile, tile_stride, counts);
    if (gather_status != RmdStatus::success) {
        return gather_status;
    }
    if (metrics != nullptr) {
        metrics->weight_values_gathered += counts.values;
        metrics->weight_baseline_address_resolutions += counts.baseline_address_resolutions;
        metrics->weight_address_resolutions += counts.address_resolutions;
    }
    return RmdStatus::success;
}

RmdStatus gather_wide_weight_tile_for_test(
    const ggml_gemmini_args_t & args,
    uint32_t block_id,
    const uint16_t * local_k,
    size_t valid_k,
    size_t col_base,
    size_t valid_cols,
    int32_t * tile,
    size_t tile_stride,
    RmdExecutionMetrics * metrics) {
    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid || !wroute::route_supports_integer_block_scale(plan)) {
        return RmdStatus::unsupported_route;
    }
    const WeightGather weights(args, plan);
    if (!weights.valid()) {
        return RmdStatus::unsupported_route;
    }
    WeightGatherCounts counts{};
    const RmdStatus gather_status = weights.fill_tile(
        block_id, local_k, valid_k, col_base, valid_cols,
        tile, tile_stride, counts);
    if (gather_status != RmdStatus::success) {
        return gather_status;
    }
    if (metrics != nullptr) {
        metrics->weight_values_gathered += counts.values;
        metrics->weight_baseline_address_resolutions += counts.baseline_address_resolutions;
        metrics->weight_address_resolutions += counts.address_resolutions;
    }
    return RmdStatus::success;
}

RmdStatus repeat_weight_tile_gather_for_test(const ggml_gemmini_args_t & args,
                                             uint32_t block_count,
                                             const uint16_t * local_k,
                                             size_t valid_k,
                                             size_t col_base,
                                             size_t valid_cols,
                                             size_t iterations,
                                             uint64_t & checksum,
                                             RmdExecutionMetrics & metrics) {
    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid || !wroute::route_supports_integer_block_scale(plan)) {
        return RmdStatus::unsupported_route;
    }
    const WeightGather weights(args, plan);
    if (!weights.valid()) {
        return RmdStatus::unsupported_route;
    }

    std::array<elem_t, kArrayDim * kArrayDim> tile{};
    uint64_t local_checksum = 0;
    RmdExecutionMetrics local_metrics{};
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        for (uint32_t block_id = 0; block_id < block_count; ++block_id) {
            WeightGatherCounts counts{};
            const RmdStatus gather_status = weights.fill_tile(
                block_id, local_k, valid_k, col_base, valid_cols,
                tile.data(), kArrayDim, counts);
            if (gather_status != RmdStatus::success) {
                return gather_status;
            }
            for (size_t k = 0; k < valid_k; ++k) {
                for (size_t col = 0; col < valid_cols; ++col) {
                    local_checksum += static_cast<uint8_t>(tile[k * kArrayDim + col]);
                }
            }
            local_metrics.weight_values_gathered += counts.values;
            local_metrics.weight_baseline_address_resolutions += counts.baseline_address_resolutions;
            local_metrics.weight_address_resolutions += counts.address_resolutions;
        }
    }
    checksum = local_checksum;
    metrics = local_metrics;
    return RmdStatus::success;
}

RmdStatus repeat_scalar_weight_tile_gather_for_test(const ggml_gemmini_args_t & args,
                                                    uint32_t block_count,
                                                    const uint16_t * local_k,
                                                    size_t valid_k,
                                                    size_t col_base,
                                                    size_t valid_cols,
                                                    size_t iterations,
                                                    uint64_t & checksum) {
    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid || !wroute::route_supports_integer_block_scale(plan) ||
        plan.weight_bits != 8 || local_k == nullptr ||
        valid_k == 0 || valid_k > kArrayDim ||
        valid_cols == 0 || valid_cols > kArrayDim ||
        col_base > args.J || valid_cols > args.J - col_base) {
        return RmdStatus::unsupported_route;
    }

    std::array<elem_t, kArrayDim * kArrayDim> tile{};
    uint64_t local_checksum = 0;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        for (uint32_t block_id = 0; block_id < block_count; ++block_id) {
            for (size_t k = 0; k < valid_k; ++k) {
                const size_t global_k = static_cast<size_t>(block_id) * kBlockSize + local_k[k];
                if (local_k[k] >= kBlockSize || global_k >= args.K) {
                    return RmdStatus::execution_failed;
                }
                for (size_t col = 0; col < valid_cols; ++col) {
                    const wreader::WeightCodeResult code = wreader::read_code(
                        args, plan, col_base + col, global_k);
                    if (!code.ok() ||
                        code.value < std::numeric_limits<elem_t>::min() ||
                        code.value > std::numeric_limits<elem_t>::max()) {
                        return RmdStatus::execution_failed;
                    }
                    tile[k * kArrayDim + col] = static_cast<elem_t>(code.value);
                }
            }
            for (size_t k = 0; k < valid_k; ++k) {
                for (size_t col = 0; col < valid_cols; ++col) {
                    local_checksum += static_cast<uint8_t>(tile[k * kArrayDim + col]);
                }
            }
        }
    }
    checksum = local_checksum;
    return RmdStatus::success;
}
#endif

RmdStatus RmdOutputAssembler::begin(const StripePacket & packet, CompressedOutput & output) {
    const RmdStatus validation = validate_packet(packet);
    if (validation != RmdStatus::success) {
        return validation;
    }
    packet_ = &packet;
    output_ = &output;
    m_tiles_ = (packet.row_count + kArrayDim - 1) / kArrayDim;
    j_tiles_ = (packet.logical_j + kArrayDim - 1) / kArrayDim;
    if (m_tiles_ == 0 || j_tiles_ == 0) {
        return RmdStatus::invalid_packet;
    }

    try {
        output.domain = CompressedOutput::Domain::block_scaled_int64;
        output.j_padded = packet.j_padded;
        output.values.assign(packet.total_output_values, OutputValue{0});

        tile_offset_.assign(packet.blocks.size(), 0);
        size_t cursor = 0;
        for (size_t index = 0; index < packet.blocks.size(); ++index) {
            tile_offset_[index] = cursor;
            cursor += static_cast<size_t>(packet.blocks[index].active_lane_count) * m_tiles_ * j_tiles_;
        }
        expected_ = cursor;
        submitted_ = 0;
        seen_.assign(expected_, 0);
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }
    return RmdStatus::success;
}

RmdStatus RmdOutputAssembler::submit(const PhysicalTile & tile) {
    if (packet_ == nullptr || output_ == nullptr || tile.values == nullptr) {
        return RmdStatus::invalid_arguments;
    }
    if (tile.packet_block_index >= packet_->blocks.size()) {
        return RmdStatus::invalid_arguments;
    }
    const BlockDescriptor & block = packet_->blocks[tile.packet_block_index];
    if (tile.lane_position >= block.active_lane_count ||
        tile.lane_id != block.lane_ids[tile.lane_position] ||
        tile.m_tile >= m_tiles_ || tile.j_tile >= j_tiles_) {
        return RmdStatus::invalid_arguments;
    }

    const size_t row_base = static_cast<size_t>(tile.m_tile) * kArrayDim;
    const size_t col_base = static_cast<size_t>(tile.j_tile) * kArrayDim;
    const size_t expected_rows = std::min(kArrayDim, packet_->row_count - row_base);
    const size_t expected_cols = std::min(kArrayDim, packet_->logical_j - col_base);
    if (tile.valid_rows != expected_rows || tile.valid_cols != expected_cols) {
        return RmdStatus::invalid_arguments;
    }

    const size_t slot = tile_offset_[tile.packet_block_index] +
        (static_cast<size_t>(tile.lane_position) * m_tiles_ + tile.m_tile) * j_tiles_ + tile.j_tile;
    if (slot >= seen_.size()) {
        return RmdStatus::invalid_arguments;
    }
    if (seen_[slot] != 0) {
        return RmdStatus::invalid_arguments; // duplicate tile
    }
    seen_[slot] = 1;
    ++submitted_;

    const size_t lane_base = block.output_value_offset +
        static_cast<size_t>(tile.lane_position) * block.lane_stride_values;
    for (size_t row = 0; row < tile.valid_rows; ++row) {
        const size_t destination = lane_base + (row_base + row) * output_->j_padded + col_base;
        if (destination + tile.valid_cols > output_->values.size()) {
            return RmdStatus::invalid_packet;
        }
        std::copy_n(tile.values + row * kArrayDim, tile.valid_cols,
                    output_->values.begin() + static_cast<ptrdiff_t>(destination));
    }
    return RmdStatus::success;
}

RmdStatus RmdOutputAssembler::finish() {
    if (packet_ == nullptr) {
        return RmdStatus::invalid_arguments;
    }
    const RmdStatus status = submitted_ == expected_ ? RmdStatus::success : RmdStatus::invalid_packet;
    packet_ = nullptr;
    output_ = nullptr;
    seen_.clear();
    tile_offset_.clear();
    return status;
}

void collect_packet_metrics(const StripePacket & packet, RmdExecutionMetrics & metrics) {
    metrics.active_blocks = packet.blocks.size();
    metrics.active_lanes = 0;
    metrics.compact_k_count = 0;
    metrics.padded_k_count = 0;
    metrics.block_padding_zeros = 0;
    metrics.row_padding_zeros = 0;
    metrics.compressed_output_values = packet.total_output_values;
    metrics.packet_bytes = packet.blocks.size() * sizeof(BlockDescriptor) +
        packet.k_indices.size() * sizeof(uint16_t) +
        packet.stacked_activation.size() * sizeof(int8_t) +
        sizeof(StripePacket);

    const size_t m_tiles = (packet.row_count + kArrayDim - 1) / kArrayDim;
    const size_t j_tiles = (packet.logical_j + kArrayDim - 1) / kArrayDim;
    metrics.physical_tile_count = 0;
    metrics.matmul_call_count = 0;
    metrics.lane_group_count = 0;
    metrics.baseline_stacked_i_tile_count = 0;
    metrics.stacked_i_tile_count = 0;
    metrics.weight_values_gathered = 0;
    metrics.weight_baseline_address_resolutions = 0;
    metrics.weight_address_resolutions = 0;

    for (const BlockDescriptor & block : packet.blocks) {
        metrics.active_lanes += block.active_lane_count;
        metrics.compact_k_count += block.compact_k_count;
        metrics.padded_k_count += block.padded_k_count;
        metrics.physical_tile_count += static_cast<size_t>(block.active_lane_count) * m_tiles * j_tiles;
        metrics.baseline_stacked_i_tile_count +=
            (block.padded_k_count / kArrayDim) * block.active_lane_count * m_tiles * j_tiles;
        const size_t k_pad = block.padded_k_count - block.compact_k_count;
        const size_t row_pad = block.rows_padded - packet.row_count;
        metrics.block_padding_zeros += static_cast<size_t>(block.active_lane_count) *
            block.rows_padded * k_pad;
        metrics.row_padding_zeros += static_cast<size_t>(block.active_lane_count) *
            row_pad * block.padded_k_count;
    }
    metrics.j_padding_zeros = (packet.j_padded - packet.logical_j) * packet.row_count *
        metrics.active_lanes;
}

static RmdStatus validate_execution_request(const ggml_gemmini_args_t & args,
                                            const StripePacket & packet) {
    const RmdStatus validation = validate_packet(packet);
    if (validation != RmdStatus::success) {
        return validation;
    }
    if (packet.logical_j != args.J || packet.logical_k != args.K) {
        return RmdStatus::invalid_arguments;
    }
    return RmdStatus::success;
}

static RmdStatus compact_plan_status(const ggml_gemmini_args_t & args,
                                     const wroute::WeightRoutePlan & plan) {
    if (!plan.valid) {
        if (plan.route == wroute::WeightRouteKind::HP1 &&
            wreader::validate(args, plan) ==
                wreader::WeightReaderStatus::ScaleOverflow) {
            return RmdStatus::overflow;
        }
        return RmdStatus::unsupported_route;
    }
    if (wroute::weight_route_status(plan, wroute::WeightExecutionPath::Compact) !=
        wroute::WeightRouteStatus::Success) {
        return RmdStatus::unsupported_route;
    }
    // A populated activation buffer identifies the runtime artifact width. Some
    // focused packet tests intentionally omit it, but a present identity may
    // never disagree with the weight reader selected for this compact request.
    if (args.A.valid() && args.A.bits != plan.weight_bits) {
        return RmdStatus::unsupported_route;
    }
    return RmdStatus::success;
}

template<CompactExecutorBackend Backend>
RmdStatus execute_rmd_stripe_impl(const ggml_gemmini_args_t & args,
                                  const StripePacket & packet,
                                  CompressedOutput & output,
                                  RmdExecutionMetrics * metrics) {

    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    const RmdStatus plan_status = compact_plan_status(args, plan);
    if (plan_status != RmdStatus::success) {
        return plan_status;
    }
    const WeightGather weights(args, plan);
    if (!weights.valid()) {
        return RmdStatus::unsupported_route;
    }
    if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
        // Validate every selected code against the native Gemmini element type
        // before the first dispatch. A widened code is never narrowed to fit.
        const RmdStatus native_status = weights.validate_native_packet(packet);
        if (native_status != RmdStatus::success) {
            return native_status;
        }
    }

    CompressedOutput staged_output;
    RmdOutputAssembler assembler;
    const RmdStatus begin_status = assembler.begin(packet, staged_output);
    if (begin_status != RmdStatus::success) {
        return begin_status;
    }

    const size_t m_tiles = (packet.row_count + kArrayDim - 1) / kArrayDim;
    const size_t j_tiles = (packet.logical_j + kArrayDim - 1) / kArrayDim;
    size_t matmul_call_count = 0;
    size_t lane_group_count = 0;
    size_t stacked_i_tile_count = 0;
    size_t weight_values_gathered = 0;
    size_t weight_baseline_address_resolutions = 0;
    size_t weight_address_resolutions = 0;

    size_t max_stacked_rows = 0;
    for (const BlockDescriptor & block : packet.blocks) {
        if (block.active_lane_count != 0 &&
            block.rows_padded > std::numeric_limits<size_t>::max() / block.active_lane_count) {
            return RmdStatus::invalid_packet;
        }
        max_stacked_rows = std::max(max_stacked_rows,
            static_cast<size_t>(block.active_lane_count) * block.rows_padded);
    }
    if (max_stacked_rows > std::numeric_limits<size_t>::max() / kArrayDim) {
        return RmdStatus::invalid_packet;
    }

    std::vector<OutputValue> stacked_values;
    std::vector<int32_t> software_weight_tile;
    std::vector<elem_t> native_weight_tile;
    std::vector<acc_t> ws_values;
    std::vector<uint64_t> block_scales;
    try {
        stacked_values.assign(max_stacked_rows * kArrayDim, OutputValue{0});
        if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
            native_weight_tile.assign(kArrayDim * kArrayDim, elem_t{0});
            ws_values.assign(max_stacked_rows * kArrayDim, acc_t{0});
        } else {
            software_weight_tile.assign(kArrayDim * kArrayDim, int32_t{0});
        }
        block_scales.assign(kArrayDim, uint64_t{0});
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    for (size_t block_index = 0; block_index < packet.blocks.size(); ++block_index) {
        const BlockDescriptor & block = packet.blocks[block_index];
        std::vector<LaneGroup> groups;
        try {
            build_lane_groups(packet, block, m_tiles, groups);
        } catch (const std::bad_alloc &) {
            return RmdStatus::allocation_failure;
        }
        lane_group_count += groups.size();

        for (size_t j_tile = 0; j_tile < j_tiles; ++j_tile) {
            const size_t col_base = j_tile * kArrayDim;
            const size_t valid_cols = std::min(kArrayDim, packet.logical_j - col_base);

            // Integer block scale, resolved once per (block, output column).
            for (size_t col = 0; col < valid_cols; ++col) {
                block_scales[col] = wroute::route_block_scale(plan, args, col_base + col, block.block_id);
            }

            for (const LaneGroup & group : groups) {
                const size_t k_tiles = group.padded_k_count / kArrayDim;
                const size_t stacked_rows = group.lane_positions.size() * block.rows_padded;
                const size_t stacked_value_count = stacked_rows * kArrayDim;
                std::fill_n(stacked_values.begin(), stacked_value_count, OutputValue{0});

                for (size_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                    const size_t k_base = k_tile * kArrayDim;
                    const size_t valid_k = group.compact_positions.size() > k_base ?
                        std::min(kArrayDim, group.compact_positions.size() - k_base) : 0;
                    if (valid_k == 0) {
                        continue;
                    }
                    ++matmul_call_count;
                    stacked_i_tile_count += stacked_rows / kArrayDim;
                    std::array<uint16_t, kArrayDim> local_k{};
                    for (size_t k = 0; k < valid_k; ++k) {
                        local_k[k] = packet.k_indices[block.k_index_offset +
                            group.compact_positions[k_base + k]];
                    }
                    WeightGatherCounts gather_counts{};
                    RmdStatus gather_status = RmdStatus::execution_failed;
                    if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
                        gather_status = weights.fill_tile(
                            block.block_id, local_k.data(), valid_k, col_base,
                            valid_cols, native_weight_tile.data(), kArrayDim,
                            gather_counts);
                    } else {
                        gather_status = weights.fill_tile(
                            block.block_id, local_k.data(), valid_k, col_base,
                            valid_cols, software_weight_tile.data(), kArrayDim,
                            gather_counts);
                    }
                    if (gather_status != RmdStatus::success) {
                        return gather_status;
                    }
                    weight_values_gathered += gather_counts.values;
                    weight_baseline_address_resolutions +=
                        gather_counts.baseline_address_resolutions;
                    weight_address_resolutions += gather_counts.address_resolutions;
                    const elem_t * activation = group.activation.data() + k_base;
                    if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
                        std::fill_n(ws_values.begin(), stacked_value_count, acc_t{0});
                        if (metrics != nullptr) {
                            // This is the live dispatch observer. Unlike aggregate
                            // metrics committed below, it exposes a call that began
                            // before a later failure.
                            ++metrics->ws_call_count;
                        }
                        tiled_matmul(stacked_rows, valid_cols, valid_k,
                            activation, native_weight_tile.data(), nullptr, ws_values.data(),
                            group.padded_k_count, kArrayDim, 0, kArrayDim,
                            1.0f, 1.0f, 1.0f,
                            NO_ACTIVATION, ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY, false,
                            1, 1, 1, false, false, true, false, 0, WS);
                        for (size_t row = 0; row < stacked_rows; ++row) {
                            OutputValue * accumulator = stacked_values.data() + row * kArrayDim;
                            for (size_t col = 0; col < valid_cols; ++col) {
                                if (!checked_add_i64(accumulator[col],
                                                     ws_values[row * kArrayDim + col],
                                                     accumulator[col])) {
                                    return RmdStatus::overflow;
                                }
                            }
                        }
                    } else {
                        // H1/HP1 remain compact packet routes, but their lane dots run in
                        // Rocket software. INT32 tiles preserve Q16 codes exactly; the
                        // per-block integer scale below is still applied before radix and
                        // cross-block composition.
                        for (size_t row = 0; row < stacked_rows; ++row) {
                            const elem_t * activation_row =
                                activation + row * group.padded_k_count;
                            OutputValue * accumulator = stacked_values.data() + row * kArrayDim;
                            for (size_t k = 0; k < valid_k; ++k) {
                                const int64_t digit = activation_row[k];
                                if (digit == 0) {
                                    continue;
                                }
                                const int32_t * weight_row =
                                    software_weight_tile.data() + k * kArrayDim;
                                for (size_t col = 0; col < valid_cols; ++col) {
                                    int64_t product = 0;
                                    if (!checked_mul_i64(digit, weight_row[col], product) ||
                                        !checked_add_i64(accumulator[col], product, accumulator[col])) {
                                        return RmdStatus::overflow;
                                    }
                                }
                            }
                        }
                    }
                }

#if defined(GGML_GEMMINI_TESTING)
                WsCallObservation observation{};
                if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
                    observation.rows = stacked_rows;
                    observation.cols = valid_cols;
                    observation.k = group.compact_positions.size();
                    observation.lane_id = block.lane_ids[group.lane_positions.front()];
                    observation.first_activation = group.activation.front();
                    observation.first_weight = native_weight_tile.front();
                    observation.raw_value = stacked_values.front();
                if (metrics != nullptr) {
                    for (size_t group_lane = 0; group_lane < group.lane_positions.size(); ++group_lane) {
                        metrics->raw_lane_values.push_back(
                            stacked_values[group_lane * block.rows_padded * kArrayDim]);
                    }
                }
                for (size_t row = 0; row < stacked_rows; ++row) {
                    for (size_t col = 0; col < valid_cols; ++col) {
                        if (stacked_values[row * kArrayDim + col] != 0) {
                            ++observation.raw_nonzero_count;
                        }
                    }
                }
                observation.block_scale = block_scales.front();
                }
#endif
                // Block integer scale is applied exactly once after all compact K tiles.
                for (size_t row = 0; row < stacked_rows; ++row) {
                    OutputValue * accumulator = stacked_values.data() + row * kArrayDim;
                    for (size_t col = 0; col < valid_cols; ++col) {
                        if (block_scales[col] > static_cast<uint64_t>(
                                std::numeric_limits<int64_t>::max()) ||
                            !checked_mul_i64(accumulator[col],
                                             static_cast<int64_t>(block_scales[col]),
                                             accumulator[col])) {
                            return RmdStatus::overflow;
                        }
                    }
                }

#if defined(GGML_GEMMINI_TESTING)
                if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
                    observation.scaled_value = stacked_values.front();
                    observation.compressed_value = observation.scaled_value;
                    const int64_t place = int64_t{1} << (8 * observation.lane_id);
                    if (!checked_mul_i64(observation.scaled_value, place,
                                         observation.composed_value)) {
                        return RmdStatus::overflow;
                    }
                    if (metrics != nullptr) {
                        metrics->ws_observations.push_back(observation);
                    }
                }
#endif
                for (size_t group_lane = 0;
                     group_lane < group.lane_positions.size(); ++group_lane) {
                    const uint8_t lane_position = group.lane_positions[group_lane];
                    const size_t lane_row_base = group_lane * block.rows_padded;
                    for (size_t m_tile = 0; m_tile < m_tiles; ++m_tile) {
                        const size_t row_base = m_tile * kArrayDim;
                        const size_t valid_rows = std::min(kArrayDim, packet.row_count - row_base);
                        PhysicalTile tile{};
                        tile.packet_block_index = static_cast<uint32_t>(block_index);
                        tile.lane_position = lane_position;
                        tile.lane_id = block.lane_ids[lane_position];
                        tile.m_tile = static_cast<uint32_t>(m_tile);
                        tile.j_tile = static_cast<uint32_t>(j_tile);
                        tile.valid_rows = static_cast<uint16_t>(valid_rows);
                        tile.valid_cols = static_cast<uint16_t>(valid_cols);
                        tile.values = stacked_values.data() +
                            (lane_row_base + row_base) * kArrayDim;
                        const RmdStatus submit_status = assembler.submit(tile);
                        if (submit_status != RmdStatus::success) {
                            return submit_status;
                        }
                    }
                }
            }
        }
    }

    const RmdStatus finish_status = assembler.finish();
    if (finish_status != RmdStatus::success) {
        return finish_status;
    }
    output = std::move(staged_output);
    if (metrics != nullptr) {
        collect_packet_metrics(packet, *metrics);
        metrics->matmul_call_count = matmul_call_count;
        metrics->lane_group_count = lane_group_count;
        metrics->stacked_i_tile_count = stacked_i_tile_count;
        metrics->weight_values_gathered = weight_values_gathered;
        metrics->weight_baseline_address_resolutions =
            weight_baseline_address_resolutions;
        metrics->weight_address_resolutions = weight_address_resolutions;
        metrics->packet_call_count = 1;
        if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
            metrics->ws_call_count = matmul_call_count;
        }
    }
    return RmdStatus::success;
}

RmdStatus execute_rmd_stripe_ws(const ggml_gemmini_args_t & args,
                                const StripePacket & packet,
                                CompressedOutput & output,
                                RmdExecutionMetrics * metrics) {
    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    const RmdStatus plan_status = compact_plan_status(args, plan);
    if (plan_status != RmdStatus::success) {
        return plan_status;
    }
    const RmdStatus validation = validate_execution_request(args, packet);
    if (validation != RmdStatus::success) {
        return validation;
    }

    if (plan.route == wroute::WeightRouteKind::H1 ||
        plan.route == wroute::WeightRouteKind::HP1) {
        return execute_rmd_stripe_impl<CompactExecutorBackend::software_ws>(
            args, packet, output, metrics);
    }

#if !defined(__riscv)
    (void) output;
    (void) metrics;
    return RmdStatus::unsupported_route;
#else
    return execute_rmd_stripe_impl<CompactExecutorBackend::gemmini_ws>(
        args, packet, output, metrics);
#endif
}

#if defined(GGML_GEMMINI_TESTING)
RmdStatus execute_rmd_stripe_reference(const ggml_gemmini_args_t & args,
                                       const StripePacket & packet,
                                       CompressedOutput & output,
                                       RmdExecutionMetrics * metrics) {
    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    const RmdStatus plan_status = compact_plan_status(args, plan);
    if (plan_status != RmdStatus::success) {
        return plan_status;
    }
    const RmdStatus validation = validate_execution_request(args, packet);
    if (validation != RmdStatus::success) {
        return validation;
    }
    return execute_rmd_stripe_impl<CompactExecutorBackend::software_ws>(
        args, packet, output, metrics);
}

RmdStatus execute_rmd_stripe_gemmini_for_test(
    const ggml_gemmini_args_t & args,
    const StripePacket & packet,
    CompressedOutput & output,
    RmdExecutionMetrics * metrics) {
    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    const RmdStatus plan_status = compact_plan_status(args, plan);
    if (plan_status != RmdStatus::success) {
        return plan_status;
    }
    const RmdStatus validation = validate_execution_request(args, packet);
    if (validation != RmdStatus::success) {
        return validation;
    }
    return execute_rmd_stripe_impl<CompactExecutorBackend::gemmini_ws>(
        args, packet, output, metrics);
}
#endif

}
