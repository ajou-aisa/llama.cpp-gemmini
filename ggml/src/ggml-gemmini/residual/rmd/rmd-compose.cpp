#include "rmd-compose.hpp"

#include "rmd-builder.hpp"
#include "rmd-executor.hpp"

#include "../../ggml-gemmini-args.h"
#include "../../quants/act/dispatch.hpp"
#include "../../quants/common/weight_route.hpp"

#include <cmath>
#include <limits>
#include <new>
#include <utility>

namespace ggml::gemmini::rmd {

namespace {

namespace wroute = quants::wroute;

constexpr __int128 kInt64Max = static_cast<__int128>(std::numeric_limits<int64_t>::max());
constexpr __int128 kInt64Min = static_cast<__int128>(std::numeric_limits<int64_t>::min());

// 256 ^ lane_id for lane_id in [0, 3]; no signed shift is used anywhere.
constexpr int64_t kRadixPlace[kMaxLanes] = {1, 256, 65536, 16777216};

bool checked_add_size(size_t left, size_t right, size_t & result) {
    return !__builtin_add_overflow(left, right, &result);
}

bool checked_mul_size(size_t left, size_t right, size_t & result) {
    return !__builtin_mul_overflow(left, right, &result);
}

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

    std::vector<OutputValue> staged_correction;
    std::vector<uint8_t> needs_wide;
    try {
        const size_t value_count = packet.row_count * packet.logical_j;
        staged_correction.assign(value_count, OutputValue{0});
        needs_wide.assign(packet.logical_j, uint8_t{0});
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    for (size_t row = 0; row < packet.row_count; ++row) {
        std::fill(needs_wide.begin(), needs_wide.end(), uint8_t{0});
        OutputValue * destination = staged_correction.data() + row * packet.logical_j;
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
    correction = std::move(staged_correction);
    return RmdStatus::success;
}

RmdStatus apply_rmd_packet_ws(const ggml_gemmini_args_t & args, const StripePacket & packet) {
    CompressedOutput output;
    RmdStatus status = execute_rmd_stripe_ws(args, packet, output);
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

namespace {

struct MergeLayout {
    float * destination = nullptr;
    size_t global_row_begin = 0;
    size_t metadata_row_begin = 0;
    size_t metadata_row_end = 0;
    size_t row_count = 0;
    size_t value_count = 0;
    size_t row_stride = 0;
    size_t col_stride = 0;
};

RmdStatus prepare_merge_layout(const ggml_gemmini_args_t & args,
                               float * destination,
                               size_t global_row_begin,
                               size_t global_row_end,
                               const std::vector<OutputValue> & correction,
                               MergeLayout & layout) {
    if (destination == nullptr || global_row_begin > global_row_end ||
        global_row_end > args.I) {
        return RmdStatus::invalid_arguments;
    }

    layout.destination = destination;
    layout.global_row_begin = global_row_begin;
    layout.row_count = global_row_end - global_row_begin;
    layout.row_stride = args.stride_f_out != 0 ? args.stride_f_out : args.J;
    layout.col_stride = args.col_stride_f_out != 0 ? args.col_stride_f_out : 1;
    if (!checked_mul_size(layout.row_count, args.J, layout.value_count) ||
        correction.size() != layout.value_count ||
        !checked_add_size(args.activation_row_offset, global_row_begin,
                          layout.metadata_row_begin) ||
        !checked_add_size(args.activation_row_offset, global_row_end,
                          layout.metadata_row_end)) {
        return RmdStatus::invalid_arguments;
    }

    if (layout.row_count != 0 && args.J != 0) {
        size_t row_offset = 0;
        size_t column_offset = 0;
        size_t final_offset = 0;
        if (!checked_mul_size(global_row_end - 1, layout.row_stride, row_offset) ||
            !checked_mul_size(args.J - 1, layout.col_stride, column_offset) ||
            !checked_add_size(row_offset, column_offset, final_offset)) {
            return RmdStatus::invalid_arguments;
        }
    }
    return RmdStatus::success;
}

bool q8_h1_scale_matches(const ggml_gemmini_args_t & args,
                         size_t j,
                         size_t block_id) {
    const block_q8_h1 * reference = args.q8_h1_block(j, 0);
    const block_q8_h1 * current = args.q8_h1_block(j, block_id);
    return reference != nullptr && current != nullptr &&
        reference->s_rf == current->s_rf;
}

RmdStatus merge_rmd_correction_checked(const ggml_gemmini_args_t & args,
                                       const MergeLayout & layout,
                                       const wroute::WeightRoutePlan & plan,
                                       const std::vector<OutputValue> & correction) {
    const quants::act::ActivationMetadataView metadata(
        args, layout.metadata_row_begin, layout.metadata_row_end);
    if (!metadata.valid()) {
        return RmdStatus::invalid_arguments;
    }

    std::vector<float> column_scale;
    std::vector<float> activation_scale;
    std::vector<float> staged_output;
    try {
        column_scale.resize(args.J);
        activation_scale.resize(layout.row_count);
        staged_output.resize(layout.value_count);
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    for (size_t j = 0; j < args.J; ++j) {
        column_scale[j] = wroute::route_column_scale(plan, args, j);
        if (!std::isfinite(column_scale[j])) {
            return RmdStatus::unsupported_route;
        }
    }
    for (size_t row = 0; row < layout.row_count; ++row) {
        if (!metadata.scale(row, activation_scale[row])) {
            return RmdStatus::invalid_arguments;
        }
    }

    for (size_t row = 0; row < layout.row_count; ++row) {
        const size_t destination_row =
            (layout.global_row_begin + row) * layout.row_stride;
        const size_t source_row = row * args.J;
        for (size_t j = 0; j < args.J; ++j) {
            const double scaled = static_cast<double>(correction[source_row + j]) *
                static_cast<double>(column_scale[j]) *
                static_cast<double>(activation_scale[row]);
            const float delta = static_cast<float>(scaled);
            const float merged = layout.destination[destination_row + j * layout.col_stride] + delta;
            if (!std::isfinite(delta) || !std::isfinite(merged)) {
                return RmdStatus::overflow;
            }
            staged_output[source_row + j] = merged;
        }
    }

    for (size_t row = 0; row < layout.row_count; ++row) {
        const size_t destination_row =
            (layout.global_row_begin + row) * layout.row_stride;
        const size_t source_row = row * args.J;
        for (size_t j = 0; j < args.J; ++j) {
            layout.destination[destination_row + j * layout.col_stride] =
                staged_output[source_row + j];
        }
    }
    return RmdStatus::success;
}

}

RmdStatus merge_rmd_correction_to(const ggml_gemmini_args_t & args,
                                  float * destination,
                                  size_t global_row_begin,
                                  size_t global_row_end,
                                  const std::vector<OutputValue> & correction) {
    MergeLayout layout;
    const RmdStatus dimensions = prepare_merge_layout(
        args, destination, global_row_begin, global_row_end, correction, layout);
    if (dimensions != RmdStatus::success) {
        return dimensions;
    }

    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid || !wroute::route_supports_integer_block_scale(plan)) {
        return RmdStatus::unsupported_route;
    }
    if (plan.route == wroute::WeightRouteKind::Q8H1 && plan.native_weight_blocks) {
        for (size_t j = 0; j < args.J; ++j) {
            for (size_t block = 0; block < args.blocks_per_row; ++block) {
                if (!q8_h1_scale_matches(args, j, block)) {
                    return RmdStatus::unsupported_route;
                }
            }
        }
    }
    return merge_rmd_correction_checked(args, layout, plan, correction);
}

RmdStatus merge_rmd_correction_to(const ggml_gemmini_args_t & args,
                                  float * destination,
                                  const StripePacket & packet,
                                  const std::vector<OutputValue> & correction) {
    size_t global_row_end = 0;
    size_t value_count = 0;
    if (packet.logical_j != args.J ||
        !checked_add_size(packet.row_begin, packet.row_count, global_row_end) ||
        !checked_mul_size(packet.row_count, packet.logical_j, value_count) ||
        correction.size() != value_count) {
        return RmdStatus::invalid_arguments;
    }

    MergeLayout layout;
    const RmdStatus dimensions = prepare_merge_layout(
        args, destination, packet.row_begin, global_row_end, correction, layout);
    if (dimensions != RmdStatus::success) {
        return dimensions;
    }

    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid || !wroute::route_supports_integer_block_scale(plan)) {
        return RmdStatus::unsupported_route;
    }
    if (plan.route == wroute::WeightRouteKind::Q8H1 && plan.native_weight_blocks) {
        for (const BlockDescriptor & block : packet.blocks) {
            for (size_t j = 0; j < args.J; ++j) {
                if (!q8_h1_scale_matches(args, j, block.block_id)) {
                    return RmdStatus::unsupported_route;
                }
            }
        }
    }
    return merge_rmd_correction_checked(args, layout, plan, correction);
}


RmdStatus merge_rmd_correction(const ggml_gemmini_args_t & args,
                               size_t global_row_begin,
                               size_t global_row_end,
                               const std::vector<OutputValue> & correction) {
    return merge_rmd_correction_to(
        args, args.f_out, global_row_begin, global_row_end, correction);
}

RmdStatus merge_rmd_correction(const ggml_gemmini_args_t & args,
                               const StripePacket & packet,
                               const std::vector<OutputValue> & correction) {
    return merge_rmd_correction_to(args, args.f_out, packet, correction);
}

}
