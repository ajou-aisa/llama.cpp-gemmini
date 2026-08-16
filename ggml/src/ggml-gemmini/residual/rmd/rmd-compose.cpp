#include "rmd-compose.hpp"

#include "rmd-builder.hpp"
#include "rmd-executor.hpp"

#include "../../ggml-gemmini-args.h"
#include "../../quants/act/dispatch.hpp"
#include "../../quants/common/weight_route.hpp"

#include <cmath>
#include <limits>
#include <new>

namespace ggml::gemmini::rmd {

namespace {

namespace wroute = quants::wroute;

constexpr __int128 kInt64Max = static_cast<__int128>(std::numeric_limits<int64_t>::max());
constexpr __int128 kInt64Min = static_cast<__int128>(std::numeric_limits<int64_t>::min());

// 256 ^ lane_id for lane_id in [0, 3]; no signed shift is used anywhere.
constexpr int64_t kRadixPlace[kMaxLanes] = {1, 256, 65536, 16777216};

RmdStatus check_offsets(const StripePacket & packet, const CompressedOutput & output) {
    if (output.domain != CompressedOutput::Domain::block_scaled_int64) {
        return RmdStatus::invalid_arguments;
    }
    if (output.j_padded != packet.j_padded ||
        output.values.size() != packet.total_output_values) {
        return RmdStatus::invalid_arguments;
    }

    size_t cursor = 0;
    for (const BlockDescriptor & block : packet.blocks) {
        if (block.output_value_offset != cursor) {
            return RmdStatus::invalid_packet; // overlapping or gapped block regions
        }
        const size_t span = static_cast<size_t>(block.active_lane_count) * block.lane_stride_values;
        if (span > output.values.size() - cursor) {
            return RmdStatus::invalid_packet;
        }
        cursor += span;
    }
    return cursor == output.values.size() ? RmdStatus::success : RmdStatus::invalid_packet;
}

}

RmdStatus compose_rmd_output(const StripePacket & packet,
                             const CompressedOutput & output,
                             std::vector<OutputValue> & correction) {
    const RmdStatus validation = validate_packet(packet);
    if (validation != RmdStatus::success) {
        return validation;
    }
    const RmdStatus offsets = check_offsets(packet, output);
    if (offsets != RmdStatus::success) {
        return offsets;
    }

    std::vector<uint8_t> needs_wide;
    try {
        const size_t value_count = packet.row_count * packet.logical_j;
        correction.assign(value_count, OutputValue{0});
        needs_wide.assign(packet.logical_j, uint8_t{0});
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    for (size_t row = 0; row < packet.row_count; ++row) {
        std::fill(needs_wide.begin(), needs_wide.end(), uint8_t{0});
        OutputValue * destination = correction.data() + row * packet.logical_j;
        for (const BlockDescriptor & block : packet.blocks) {
            for (uint8_t lane_position = 0; lane_position < block.active_lane_count; ++lane_position) {
                const uint8_t lane_id = block.lane_ids[lane_position];
                if (lane_id >= kMaxLanes) {
                    return RmdStatus::invalid_packet;
                }
                const int64_t place = kRadixPlace[lane_id];
                const size_t lane_base = block.output_value_offset +
                    static_cast<size_t>(lane_position) * block.lane_stride_values;
                const OutputValue * source = output.values.data() + lane_base + row * output.j_padded;
                for (size_t j = 0; j < packet.logical_j; ++j) {
                    if (needs_wide[j] != 0) {
                        continue;
                    }
                    int64_t scaled = 0;
                    int64_t sum = 0;
                    if (__builtin_mul_overflow(source[j], place, &scaled) ||
                        __builtin_add_overflow(destination[j], scaled, &sum)) {
                        needs_wide[j] = 1;
                    } else {
                        destination[j] = sum;
                    }
                }
            }
        }

        for (size_t j = 0; j < packet.logical_j; ++j) {
            if (needs_wide[j] == 0) {
                continue;
            }
            __int128 wide = 0;
            for (const BlockDescriptor & block : packet.blocks) {
                for (uint8_t lane_position = 0; lane_position < block.active_lane_count; ++lane_position) {
                    const uint8_t lane_id = block.lane_ids[lane_position];
                    const size_t lane_base = block.output_value_offset +
                        static_cast<size_t>(lane_position) * block.lane_stride_values;
                    const OutputValue source = output.values[
                        lane_base + row * output.j_padded + j];
                    wide += static_cast<__int128>(source) *
                        static_cast<__int128>(kRadixPlace[lane_id]);
                }
            }
            if (wide > kInt64Max || wide < kInt64Min) {
                return RmdStatus::overflow;
            }
            destination[j] = static_cast<int64_t>(wide);
        }
    }
    return RmdStatus::success;
}

RmdStatus apply_rmd_packet(const ggml_gemmini_args_t & args, const StripePacket & packet) {
    CompressedOutput output;
    RmdStatus status = execute_rmd_stripe(args, packet, output);
    if (status != RmdStatus::success) {
        return status;
    }
    std::vector<OutputValue> correction;
    status = compose_rmd_output(packet, output, correction);
    if (status != RmdStatus::success) {
        return status;
    }
    return merge_rmd_correction(args, packet, correction);
}

void expand_packets_to_plane(const std::vector<StripePacketHandle> & packets,
                             size_t row_count,
                             size_t col_count,
                             std::vector<int32_t> & plane) {
    plane.assign(row_count * col_count, 0);
    for (const StripePacketHandle & handle : packets) {
        if (!handle) {
            continue;
        }
        const StripePacket & packet = *handle;
        for (const BlockDescriptor & block : packet.blocks) {
            for (uint8_t lane_position = 0; lane_position < block.active_lane_count; ++lane_position) {
                const uint8_t lane_id = block.lane_ids[lane_position];
                if (lane_id >= kMaxLanes) {
                    continue;
                }
                const int32_t place = static_cast<int32_t>(kRadixPlace[lane_id]);
                const size_t lane_base = block.activation_offset +
                    static_cast<size_t>(lane_position) * block.rows_padded * block.padded_k_count;
                for (size_t row = 0; row < packet.row_count; ++row) {
                    const size_t global_row = packet.row_begin + row;
                    if (global_row >= row_count) {
                        continue;
                    }
                    const int8_t * source =
                        packet.stacked_activation.data() + lane_base + row * block.padded_k_count;
                    for (size_t k = 0; k < block.compact_k_count; ++k) {
                        if (source[k] == 0) {
                            continue;
                        }
                        const size_t column = static_cast<size_t>(block.global_k_begin) +
                            packet.k_indices[block.k_index_offset + k];
                        if (column >= col_count) {
                            continue;
                        }
                        plane[global_row * col_count + column] +=
                            static_cast<int32_t>(source[k]) * place;
                    }
                }
            }
        }
    }
}

RmdStatus merge_rmd_correction(const ggml_gemmini_args_t & args,
                               const StripePacket & packet,
                               const std::vector<OutputValue> & correction) {
    if (args.f_out == nullptr || packet.logical_j != args.J ||
        correction.size() != packet.row_count * packet.logical_j) {
        return RmdStatus::invalid_arguments;
    }

    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid || !wroute::route_supports_integer_block_scale(plan)) {
        return RmdStatus::unsupported_route;
    }

    // The column scale must not depend on the K block, otherwise the executor could not
    // have folded the whole block factor into an integer. Q8_H1 replicates the row scale
    // into every block; verify that for the blocks this packet actually touches.
    if (plan.route == wroute::WeightRouteKind::Q8H1 && plan.native_weight_blocks) {
        for (const BlockDescriptor & block : packet.blocks) {
            for (size_t j = 0; j < args.J; ++j) {
                const block_q8_h1 * reference = args.q8_h1_block(j, 0);
                const block_q8_h1 * current = args.q8_h1_block(j, block.block_id);
                if (reference == nullptr || current == nullptr ||
                    reference->s_rf != current->s_rf) {
                    return RmdStatus::unsupported_route;
                }
            }
        }
    }

    const quants::act::ActivationMetadataView metadata(
        args, args.activation_row_offset + packet.row_begin,
        args.activation_row_offset + packet.row_begin + packet.row_count);
    if (!metadata.valid()) {
        return RmdStatus::invalid_arguments;
    }

    const size_t row_stride = args.stride_f_out != 0 ? args.stride_f_out : args.J;
    const size_t col_stride = args.col_stride_f_out != 0 ? args.col_stride_f_out : 1;

    std::vector<float> column_scale;
    try {
        column_scale.resize(args.J);
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }
    for (size_t j = 0; j < args.J; ++j) {
        column_scale[j] = wroute::route_column_scale(plan, args, j);
        if (!std::isfinite(column_scale[j])) {
            return RmdStatus::unsupported_route;
        }
    }

    for (size_t row = 0; row < packet.row_count; ++row) {
        float activation_scale = 1.0f;
        if (!metadata.scale(row, activation_scale)) {
            return RmdStatus::invalid_arguments;
        }
        float * destination = args.f_out + (packet.row_begin + row) * row_stride;
        const OutputValue * source = correction.data() + row * packet.logical_j;
        for (size_t j = 0; j < args.J; ++j) {
            const double value = static_cast<double>(source[j]) *
                static_cast<double>(column_scale[j]) *
                static_cast<double>(activation_scale);
            destination[j * col_stride] += static_cast<float>(value);
        }
    }
    return RmdStatus::success;
}

}
