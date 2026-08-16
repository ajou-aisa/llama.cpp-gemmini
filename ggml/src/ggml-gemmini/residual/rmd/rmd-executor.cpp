#include "rmd-executor.hpp"

#include "rmd-builder.hpp"

#include "../../ggml-gemmini-args.h"
#include "../../quants/common/weight_route.hpp"

#include <gemmini.h>

#include <algorithm>
#include <limits>
#include <new>

namespace ggml::gemmini::rmd {

static_assert(kNativeWeightScaleGroup == QK8_0);

namespace {

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

// Weight code for one (original block, block-local K, output column).
class WeightGather {
public:
    WeightGather(const ggml_gemmini_args_t & args, const wroute::WeightRoutePlan & plan)
        : args_(args),
          native_h1_(plan.route == wroute::WeightRouteKind::Q8H1 && plan.native_weight_blocks),
          dense_(reinterpret_cast<const int8_t *>(args.B)),
          stride_(plan.weight_stride),
          column_major_(plan.layout == wroute::WeightLayout::JxK_ColMajor) {}

    bool valid() const { return native_h1_ || dense_ != nullptr; }

    bool code(uint32_t block_id, uint16_t block_local_k, size_t j, int8_t & out) const {
        if (native_h1_) {
            const block_q8_h1 * block = args_.q8_h1_block(j, block_id);
            if (block == nullptr) {
                return false;
            }
            out = static_cast<int8_t>(block->qs[block_local_k]);
            return true;
        }
        const size_t global_k = static_cast<size_t>(block_id) * kBlockSize + block_local_k;
        if (global_k >= args_.K) {
            return false;
        }
        out = column_major_ ? dense_[j * stride_ + global_k] : dense_[global_k * stride_ + j];
        return true;
    }

private:
    const ggml_gemmini_args_t & args_;
    bool native_h1_;
    const int8_t * dense_;
    size_t stride_;
    bool column_major_;
};

}

bool weight_route_supports_rmd(const ggml_gemmini_args_t & args) {
    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    return plan.valid && wroute::route_supports_integer_block_scale(plan);
}

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
    metrics.stacked_i_tile_count = 0;

    for (const BlockDescriptor & block : packet.blocks) {
        metrics.active_lanes += block.active_lane_count;
        metrics.compact_k_count += block.compact_k_count;
        metrics.padded_k_count += block.padded_k_count;
        metrics.physical_tile_count += static_cast<size_t>(block.active_lane_count) * m_tiles * j_tiles;
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

RmdStatus execute_rmd_stripe(const ggml_gemmini_args_t & args,
                             const StripePacket & packet,
                             CompressedOutput & output,
                             RmdExecutionMetrics * metrics) {
    const RmdStatus validation = validate_packet(packet);
    if (validation != RmdStatus::success) {
        return validation;
    }
    if (packet.logical_j != args.J || packet.logical_k != args.K) {
        return RmdStatus::invalid_arguments;
    }
    if (args.tiled_matmul_type == OS) {
        return RmdStatus::unsupported_route;
    }
#if !defined(__riscv)
    if (args.tiled_matmul_type == WS) {
        return RmdStatus::unsupported_route;
    }
#endif
    if (args.tiled_matmul_type != CPU && args.tiled_matmul_type != WS) {
        return RmdStatus::invalid_arguments;
    }

    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid) {
        return RmdStatus::unsupported_route;
    }
    if (!wroute::route_supports_integer_block_scale(plan)) {
        return RmdStatus::unsupported_route;
    }
    const WeightGather weights(args, plan);
    if (!weights.valid()) {
        return RmdStatus::unsupported_route;
    }

    RmdOutputAssembler assembler;
    const RmdStatus begin_status = assembler.begin(packet, output);
    if (begin_status != RmdStatus::success) {
        return begin_status;
    }

    const size_t m_tiles = (packet.row_count + kArrayDim - 1) / kArrayDim;
    const size_t j_tiles = (packet.logical_j + kArrayDim - 1) / kArrayDim;
    size_t matmul_call_count = 0;
    size_t stacked_i_tile_count = 0;

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
    std::vector<int8_t> weight_tile;
    std::vector<acc_t> ws_values;
    std::vector<uint64_t> block_scales;
    try {
        stacked_values.assign(max_stacked_rows * kArrayDim, OutputValue{0});
        weight_tile.assign(kArrayDim * kArrayDim, int8_t{0});
        if (args.tiled_matmul_type == WS) {
            ws_values.assign(max_stacked_rows * kArrayDim, acc_t{0});
        }
        block_scales.assign(kArrayDim, uint64_t{0});
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    for (size_t block_index = 0; block_index < packet.blocks.size(); ++block_index) {
        const BlockDescriptor & block = packet.blocks[block_index];
        const size_t k_tiles = block.padded_k_count / kArrayDim;
        const size_t stacked_rows =
            static_cast<size_t>(block.active_lane_count) * block.rows_padded;
        const size_t stacked_value_count = stacked_rows * kArrayDim;

        for (size_t j_tile = 0; j_tile < j_tiles; ++j_tile) {
            const size_t col_base = j_tile * kArrayDim;
            const size_t valid_cols = std::min(kArrayDim, packet.logical_j - col_base);
            std::fill_n(stacked_values.begin(), stacked_value_count, OutputValue{0});

            // Integer block scale, resolved once per (block, output column).
            for (size_t col = 0; col < valid_cols; ++col) {
                block_scales[col] = wroute::route_block_scale(plan, args, col_base + col, block.block_id);
            }

            // All lane/M rows are contiguous in the packet. One call per K tile lets
            // Gemmini keep this B tile resident while its internal i0 loop advances.
            for (size_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                const size_t k_base = k_tile * kArrayDim;
                const size_t valid_k = block.compact_k_count > k_base ?
                    std::min(kArrayDim, block.compact_k_count - k_base) : 0;
                if (valid_k == 0) {
                    continue;
                }
                ++matmul_call_count;
                stacked_i_tile_count += stacked_rows / kArrayDim;
                for (size_t k = 0; k < valid_k; ++k) {
                    const uint16_t local_k =
                        packet.k_indices[block.k_index_offset + k_base + k];
                    for (size_t col = 0; col < valid_cols; ++col) {
                        int8_t code = 0;
                        if (!weights.code(block.block_id, local_k, col_base + col, code)) {
                            return RmdStatus::execution_failed;
                        }
                        weight_tile[k * kArrayDim + col] = code;
                    }
                }
                const int8_t * activation = packet.stacked_activation.data() +
                    block.activation_offset + k_base;
                if (args.tiled_matmul_type == WS) {
                    std::fill_n(ws_values.begin(), stacked_value_count, acc_t{0});
                    tiled_matmul(stacked_rows, valid_cols, valid_k,
                        activation, weight_tile.data(), nullptr, ws_values.data(),
                        block.padded_k_count, kArrayDim, 0, kArrayDim,
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
                    for (size_t row = 0; row < stacked_rows; ++row) {
                        const int8_t * activation_row =
                            activation + row * block.padded_k_count;
                        OutputValue * accumulator = stacked_values.data() + row * kArrayDim;
                        for (size_t k = 0; k < valid_k; ++k) {
                            const int64_t digit = activation_row[k];
                            if (digit == 0) {
                                continue;
                            }
                            const int8_t * weight_row = weight_tile.data() + k * kArrayDim;
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

            for (uint8_t lane_position = 0; lane_position < block.active_lane_count; ++lane_position) {
                const size_t lane_row_base =
                    static_cast<size_t>(lane_position) * block.rows_padded;
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

    const RmdStatus finish_status = assembler.finish();
    if (finish_status != RmdStatus::success) {
        return finish_status;
    }
    if (metrics != nullptr) {
        collect_packet_metrics(packet, *metrics);
        metrics->matmul_call_count = matmul_call_count;
        metrics->stacked_i_tile_count = stacked_i_tile_count;
    }
    return RmdStatus::success;
}

}
