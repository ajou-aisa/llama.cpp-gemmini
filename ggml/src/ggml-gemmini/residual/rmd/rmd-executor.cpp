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
static_assert(sizeof(elem_t) == GGML_GEMMINI_ACTIVATION_STORAGE_BYTES,
              "native activation staging must match elem_t storage");
static_assert(sizeof(elem_t) == GGML_GEMMINI_WEIGHT_STORAGE_BYTES,
              "native weight staging must match elem_t storage");
static_assert(GGML_GEMMINI_ACTIVATION_BITS == 16
                  ? std::is_same_v<elem_t, int16_t>
                  : std::is_same_v<elem_t, int8_t>,
              "native operand staging must use signed width-native elem_t");
static_assert(std::is_integral_v<acc_t> && std::is_signed_v<acc_t> &&
                  sizeof(acc_t) * 8 >= 32,
              "native SRMD requires a signed accumulator of at least 32 bits");

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
    // Keep the decoded packet width until the selected native boundary performs
    // its checked conversion. In particular, W16 must never transit int8_t.
    std::vector<int32_t> activation;
    size_t padded_k_count = 0;
};

enum class CompactExecutorBackend : uint8_t {
    gemmini_ws,
    checked_software,
};

// Owns one matrix in the selected native mvin representation. Logical strides
// remain element-based; only the transport bytes differ by configured width.
class NativeOperandBuffer {
public:
    RmdStatus assign(const int32_t * values, size_t count) {
        if (values == nullptr || count == 0) {
            return RmdStatus::invalid_arguments;
        }
        constexpr int32_t qmin =
            -(int32_t{1} << (GGML_GEMMINI_ACTIVATION_BITS - 1));
        constexpr int32_t qmax =
            (int32_t{1} << (GGML_GEMMINI_ACTIVATION_BITS - 1)) - 1;
        for (size_t index = 0; index < count; ++index) {
            if (values[index] < qmin || values[index] > qmax) {
                return RmdStatus::overflow;
            }
        }
        try {
#if GGML_GEMMINI_ACTIVATION_BITS == 4
            const size_t byte_count = count / 2 + count % 2;
            packed_int4_.assign(byte_count, uint8_t{0});
            for (size_t index = 0; index < count; ++index) {
                const uint8_t nibble =
                    static_cast<uint8_t>(values[index]) & 0x0fu;
                packed_int4_[index / 2] |= static_cast<uint8_t>(
                    nibble << ((index % 2) * 4));
            }
#elif GGML_GEMMINI_ACTIVATION_BITS == 8
            signed_int8_.resize(count);
            for (size_t index = 0; index < count; ++index) {
                signed_int8_[index] = static_cast<int8_t>(values[index]);
            }
#elif GGML_GEMMINI_ACTIVATION_BITS == 16
            signed_int16_.resize(count);
            for (size_t index = 0; index < count; ++index) {
                signed_int16_[index] = static_cast<int16_t>(values[index]);
            }
#else
#error "unsupported native Gemmini activation width"
#endif
        } catch (const std::bad_alloc &) {
            return RmdStatus::allocation_failure;
        }
        logical_count_ = count;
        return RmdStatus::success;
    }

    const elem_t * data(size_t logical_offset = 0) const {
        return reinterpret_cast<const elem_t *>(bytes(logical_offset));
    }

    const uint8_t * bytes(size_t logical_offset = 0) const {
        if (logical_offset >= logical_count_) {
            return nullptr;
        }
#if GGML_GEMMINI_ACTIVATION_BITS == 4
        if (logical_offset % 2 != 0) {
            return nullptr;
        }
        return packed_int4_.data() + logical_offset / 2;
#elif GGML_GEMMINI_ACTIVATION_BITS == 8
        return reinterpret_cast<const uint8_t *>(
            signed_int8_.data() + logical_offset);
#else
        return reinterpret_cast<const uint8_t *>(
            signed_int16_.data() + logical_offset);
#endif
    }

    size_t byte_count(size_t logical_offset = 0) const {
        if (logical_offset >= logical_count_) {
            return 0;
        }
#if GGML_GEMMINI_ACTIVATION_BITS == 4
        return (logical_count_ - logical_offset) / 2 +
            (logical_count_ - logical_offset) % 2;
#elif GGML_GEMMINI_ACTIVATION_BITS == 8
        return logical_count_ - logical_offset;
#else
        return (logical_count_ - logical_offset) * sizeof(int16_t);
#endif
    }

private:
    std::vector<uint8_t> packed_int4_;
    std::vector<int8_t> signed_int8_;
    std::vector<int16_t> signed_int16_;
    size_t logical_count_ = 0;
};

size_t count_bits(uint32_t value) {
    size_t count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
}

void choose_lane_partition(
    const std::array<uint32_t, kMaxNativeRadixLanes> & lane_support,
    uint8_t lane_count,
    size_t m_tiles,
    uint8_t lane_position,
    uint8_t group_count,
    std::array<uint8_t, kMaxNativeRadixLanes> & assignment,
    std::array<uint8_t, kMaxNativeRadixLanes> & best_assignment,
    size_t baseline_calls,
    size_t & best_i_tiles) {
    if (lane_position == lane_count) {
        std::array<uint32_t, kMaxNativeRadixLanes> group_support{};
        std::array<size_t, kMaxNativeRadixLanes> group_lanes{};
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

RmdStatus build_lane_groups(const StripePacket & packet,
                            const BlockDescriptor & block,
                            size_t m_tiles,
                            std::vector<LaneGroup> & groups) {
    std::array<uint32_t, kMaxNativeRadixLanes> lane_support{};
    for (uint8_t lane = 0; lane < block.active_lane_count; ++lane) {
        for (size_t row = 0; row < packet.row_count; ++row) {
            for (uint16_t k = 0; k < block.compact_k_count; ++k) {
                int32_t digit = 0;
                const RmdStatus status = read_packet_digit(
                    packet, block, lane, row, k, digit);
                if (status != RmdStatus::success) {
                    return status;
                }
                if (digit != 0) {
                    lane_support[lane] |= uint32_t{1} << k;
                }
            }
        }
    }

    std::array<uint8_t, kMaxNativeRadixLanes> assignment{};
    std::array<uint8_t, kMaxNativeRadixLanes> best_assignment{};
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
                                group.padded_k_count, int32_t{0});
        for (size_t group_lane = 0; group_lane < group.lane_positions.size(); ++group_lane) {
            const uint8_t packet_lane = group.lane_positions[group_lane];
            const size_t destination_base = group_lane * block.rows_padded * group.padded_k_count;
            for (size_t row = 0; row < packet.row_count; ++row) {
                for (size_t k = 0; k < group.compact_positions.size(); ++k) {
                    int32_t digit = 0;
                    const RmdStatus status = read_packet_digit(
                        packet, block, packet_lane, row, group.compact_positions[k], digit);
                    if (status != RmdStatus::success) {
                        return status;
                    }
                    group.activation[destination_base + row * group.padded_k_count + k] =
                        digit;
                }
            }
        }
    }
    return RmdStatus::success;
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
        packet.stacked_activation.packed_int4.size() +
        packet.stacked_activation.signed_int8.size() * sizeof(int8_t) +
        packet.stacked_activation.signed_int16.size() * sizeof(int16_t) +
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
    if constexpr (Backend != CompactExecutorBackend::checked_software) {
        if (packet.digit_bits != GGML_GEMMINI_ACTIVATION_BITS ||
            plan.weight_bits != GGML_GEMMINI_WEIGHT_BITS ||
            packet.digit_bits != plan.weight_bits) {
            return RmdStatus::unsupported_route;
        }
        // Validate every selected code before the first dispatch. Conversion to
        // packed W4, scalar W8, or scalar W16 happens only after this pass.
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
    RmdExecutionMetrics staged_metrics{};

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
    std::vector<int32_t> weight_tile;
    NativeOperandBuffer native_weight_tile;
    std::vector<acc_t> ws_values;
    std::vector<uint64_t> block_scales;
    try {
        stacked_values.assign(max_stacked_rows * kArrayDim, OutputValue{0});
        weight_tile.assign(kArrayDim * kArrayDim, int32_t{0});
        if constexpr (Backend == CompactExecutorBackend::gemmini_ws) {
            ws_values.assign(max_stacked_rows * kArrayDim, acc_t{0});
        }
        block_scales.assign(kArrayDim, uint64_t{0});
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    for (size_t block_index = 0; block_index < packet.blocks.size(); ++block_index) {
        const BlockDescriptor & block = packet.blocks[block_index];
        std::vector<LaneGroup> groups;
        try {
            const RmdStatus group_status = build_lane_groups(packet, block, m_tiles, groups);
            if (group_status != RmdStatus::success) {
                return group_status;
            }
        } catch (const std::bad_alloc &) {
            return RmdStatus::allocation_failure;
        }
        lane_group_count += groups.size();

        for (size_t j_tile = 0; j_tile < j_tiles; ++j_tile) {
            const size_t col_base = j_tile * kArrayDim;
            const size_t valid_cols = std::min(kArrayDim, packet.logical_j - col_base);

            // Integer block scale, resolved once per (block, output column).
            for (size_t col = 0; col < valid_cols; ++col) {
                block_scales[col] = wroute::route_block_scale(
                    plan, args, col_base + col, block.block_id);
            }

            for (const LaneGroup & group : groups) {
                const size_t k_tiles = group.padded_k_count / kArrayDim;
                const size_t stacked_rows = group.lane_positions.size() * block.rows_padded;
                const size_t stacked_value_count = stacked_rows * kArrayDim;
                std::fill_n(stacked_values.begin(), stacked_value_count, OutputValue{0});

                NativeOperandBuffer native_activation;
                if constexpr (Backend != CompactExecutorBackend::checked_software) {
                    const RmdStatus staging_status = native_activation.assign(
                        group.activation.data(), group.activation.size());
                    if (staging_status != RmdStatus::success) {
                        return staging_status;
                    }
                }

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
                    const RmdStatus gather_status = weights.fill_tile(
                        block.block_id, local_k.data(), valid_k, col_base,
                        valid_cols, weight_tile.data(), kArrayDim, gather_counts);
                    if (gather_status != RmdStatus::success) {
                        return gather_status;
                    }
                    weight_values_gathered += gather_counts.values;
                    weight_baseline_address_resolutions +=
                        gather_counts.baseline_address_resolutions;
                    weight_address_resolutions += gather_counts.address_resolutions;
                    const int32_t * activation = group.activation.data() + k_base;
                    if constexpr (Backend != CompactExecutorBackend::checked_software) {
                        const RmdStatus staging_status = native_weight_tile.assign(
                            weight_tile.data(), weight_tile.size());
                        if (staging_status != RmdStatus::success) {
                            return staging_status;
                        }
                        const elem_t * native_activation_tile =
                            native_activation.data(k_base);
                        const elem_t * native_weight = native_weight_tile.data();
                        if (native_activation_tile == nullptr || native_weight == nullptr) {
                            return RmdStatus::execution_failed;
                        }
                        ++staged_metrics.ws_call_count;
                        std::fill_n(ws_values.begin(), stacked_value_count, acc_t{0});
                        tiled_matmul(stacked_rows, valid_cols, valid_k,
                            native_activation_tile, native_weight, nullptr, ws_values.data(),
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
                            const int32_t * activation_row =
                                activation + row * group.padded_k_count;
                            OutputValue * accumulator = stacked_values.data() + row * kArrayDim;
                            for (size_t k = 0; k < valid_k; ++k) {
                                const int64_t digit = activation_row[k];
                                if (digit == 0) {
                                    continue;
                                }
                                const int32_t * weight_row =
                                    weight_tile.data() + k * kArrayDim;
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
                if constexpr (Backend != CompactExecutorBackend::checked_software) {
                    observation.rows = stacked_rows;
                    observation.cols = valid_cols;
                    observation.k = group.compact_positions.size();
                    observation.lane_id = block.lane_ids[group.lane_positions.front()];
                    observation.first_activation =
                        static_cast<elem_t>(group.activation.front());
                    observation.first_weight =
                        static_cast<elem_t>(weight_tile.front());
                    observation.raw_value = stacked_values.front();
                if (metrics != nullptr) {
                    for (size_t group_lane = 0; group_lane < group.lane_positions.size(); ++group_lane) {
                        staged_metrics.raw_lane_values.push_back(
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
                if constexpr (Backend != CompactExecutorBackend::checked_software) {
                    observation.scaled_value = stacked_values.front();
                    observation.compressed_value = observation.scaled_value;
                    const BalancedRadixContract radix =
                        balanced_radix_contract(packet.digit_bits);
                    int64_t place = 1;
                    for (uint8_t lane = 0; lane < observation.lane_id; ++lane) {
                        if (!checked_mul_i64(
                                place, static_cast<int64_t>(radix.radix), place)) {
                            return RmdStatus::overflow;
                        }
                    }
                    if (!checked_mul_i64(observation.scaled_value, place,
                                         observation.composed_value)) {
                        return RmdStatus::overflow;
                    }
                    if (metrics != nullptr) {
                        staged_metrics.ws_observations.push_back(observation);
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
        collect_packet_metrics(packet, staged_metrics);
        staged_metrics.matmul_call_count = matmul_call_count;
        staged_metrics.lane_group_count = lane_group_count;
        staged_metrics.stacked_i_tile_count = stacked_i_tile_count;
        staged_metrics.weight_values_gathered = weight_values_gathered;
        staged_metrics.weight_baseline_address_resolutions =
            weight_baseline_address_resolutions;
        staged_metrics.weight_address_resolutions = weight_address_resolutions;
        staged_metrics.packet_call_count = 1;
        if constexpr (Backend != CompactExecutorBackend::checked_software) {
            staged_metrics.ws_call_count = matmul_call_count;
        }
        *metrics = std::move(staged_metrics);
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
        return execute_rmd_stripe_impl<CompactExecutorBackend::checked_software>(
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
    return execute_rmd_stripe_impl<CompactExecutorBackend::checked_software>(
        args, packet, output, metrics);
}

RmdStatus execute_rmd_stripe_gemmini_for_test(
    const ggml_gemmini_args_t & args,
    const StripePacket & packet,
    CompressedOutput & output,
    RmdExecutionMetrics * metrics) {
    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (packet.digit_bits != GGML_GEMMINI_ACTIVATION_BITS ||
        plan.weight_bits != GGML_GEMMINI_WEIGHT_BITS ||
        packet.digit_bits != plan.weight_bits) {
        return RmdStatus::unsupported_route;
    }
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
