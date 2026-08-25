#include "rmd-compose.hpp"

#include "rmd-builder.hpp"
#include "rmd-executor.hpp"

#include "../../ggml-gemmini-args.h"
#include "../../quants/act/dispatch.hpp"
#include "../../quants/common/weight_reader.hpp"
#include "../../quants/common/weight_route.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace ggml::gemmini::rmd {

namespace {

namespace wreader = quants::wreader;
namespace wroute = quants::wroute;

constexpr __int128 kInt64Max = static_cast<__int128>(std::numeric_limits<int64_t>::max());
constexpr __int128 kInt64Min = static_cast<__int128>(std::numeric_limits<int64_t>::min());

bool checked_add_size(size_t left, size_t right, size_t & result) {
    return !__builtin_add_overflow(left, right, &result);
}

bool checked_mul_size(size_t left, size_t right, size_t & result) {
    return !__builtin_mul_overflow(left, right, &result);
}

bool finite_float_representation(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const volatile uint32_t observed = bits;
    return (observed & 0x7f800000u) != 0x7f800000u;
}

bool finite_double_representation(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const volatile uint64_t observed = bits;
    return (observed & UINT64_C(0x7ff0000000000000)) !=
        UINT64_C(0x7ff0000000000000);
}

RmdStatus check_offsets(const StripePacket & packet, const CompressedOutput & output) {
    if (output.domain != CompressedOutput::Domain::block_scaled_int64 ||
        output.j_padded != packet.j_padded ||
        output.values.size() != packet.total_output_values) {
        return RmdStatus::invalid_arguments;
    }

    size_t cursor = 0;
    for (const BlockDescriptor & block : packet.blocks) {
        if (block.output_value_offset != cursor) {
            return RmdStatus::invalid_packet; // overlapping or gapped block regions
        }
        const size_t span = static_cast<size_t>(block.active_lane_count) * block.lane_stride_values;
        if (cursor > output.values.size() || span > output.values.size() - cursor) {
            return RmdStatus::invalid_packet;
        }
        cursor += span;
    }
    return cursor == output.values.size() ? RmdStatus::success : RmdStatus::invalid_packet;
}

}

RmdStatus compose_rmd_output(const StripePacket & packet,
                             const CompressedOutput & output,
                             Correction & correction) {
    const RmdStatus validation = validate_packet(packet);
    if (validation != RmdStatus::success) {
        return validation;
    }
    const RmdStatus offsets = check_offsets(packet, output);
    if (offsets != RmdStatus::success) {
        return offsets;
    }

    const BalancedRadixContract contract = balanced_radix_contract(packet.digit_bits);
    if (contract.radix == 0 || packet.lane_capacity != contract.lane_capacity) {
        return RmdStatus::invalid_packet;
    }

    size_t value_count = 0;
    if (!checked_mul_size(packet.row_count, packet.logical_j, value_count)) {
        return RmdStatus::overflow;
    }
    std::vector<OutputValue> staged_correction;
    try {
        staged_correction.assign(value_count, OutputValue{0});
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    for (size_t row = 0; row < packet.row_count; ++row) {
        for (size_t j = 0; j < packet.logical_j; ++j) {
            __int128 total = 0;
            for (const BlockDescriptor & block : packet.blocks) {
                __int128 block_value = 0;
                size_t lane_position = block.active_lane_count;
                for (uint8_t lane = packet.lane_capacity; lane-- > 0;) {
                    if (__builtin_mul_overflow(
                            block_value, static_cast<__int128>(contract.radix),
                            &block_value)) {
                        return RmdStatus::overflow;
                    }
                    if (lane_position != 0 &&
                        block.lane_ids[lane_position - 1] == lane) {
                        --lane_position;
                        const size_t lane_base = block.output_value_offset +
                            lane_position * block.lane_stride_values;
                        const OutputValue source = output.values[
                            lane_base + row * output.j_padded + j];
                        if (__builtin_add_overflow(
                                block_value, static_cast<__int128>(source),
                                &block_value)) {
                            return RmdStatus::overflow;
                        }
                    }
                }
                if (lane_position != 0 ||
                    __builtin_add_overflow(total, block_value, &total)) {
                    return RmdStatus::overflow;
                }
            }
            if (total > kInt64Max || total < kInt64Min) {
                return RmdStatus::overflow;
            }
            staged_correction[row * packet.logical_j + j] =
                static_cast<int64_t>(total);
        }
    }
    Correction staged = BlockScaledInt64Correction{std::move(staged_correction)};
    correction.swap(staged);
    return RmdStatus::success;
}

RmdStatus apply_rmd_packet_ws(const ggml_gemmini_args_t & args, const StripePacket & packet) {
    CompressedOutput output;
    RmdStatus status = execute_rmd_stripe_ws(args, packet, output);
    if (status != RmdStatus::success) {
        return status;
    }
    Correction correction = BlockScaledInt64Correction{};
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
    size_t value_count = 0;
    if (!checked_mul_size(row_count, col_count, value_count)) {
        return;
    }
    std::vector<int32_t> staged;
    try {
        staged.assign(value_count, 0);
    } catch (const std::bad_alloc &) {
        return;
    }

    for (const StripePacketHandle & handle : packets) {
        if (!handle) {
            continue;
        }
        if (validate_packet(*handle) != RmdStatus::success) {
            return;
        }
        const StripePacket & packet = *handle;
        const BalancedRadixContract contract =
            balanced_radix_contract(packet.digit_bits);
        for (const BlockDescriptor & block : packet.blocks) {
            for (size_t row = 0; row < packet.row_count; ++row) {
                const size_t global_row = packet.row_begin + row;
                if (global_row >= row_count) {
                    continue;
                }
                for (size_t k = 0; k < block.compact_k_count; ++k) {
                    int64_t reconstructed = 0;
                    size_t lane_position = block.active_lane_count;
                    for (uint8_t lane = packet.lane_capacity; lane-- > 0;) {
                        if (__builtin_mul_overflow(
                                reconstructed, static_cast<int64_t>(contract.radix),
                                &reconstructed)) {
                            return;
                        }
                        if (lane_position != 0 &&
                            block.lane_ids[lane_position - 1] == lane) {
                            --lane_position;
                            int32_t digit = 0;
                            if (read_packet_digit(packet, block,
                                                  static_cast<uint8_t>(lane_position),
                                                  row, k, digit) != RmdStatus::success ||
                                __builtin_add_overflow(
                                    reconstructed, static_cast<int64_t>(digit),
                                    &reconstructed)) {
                                return;
                            }
                        }
                    }
                    const size_t column = static_cast<size_t>(block.global_k_begin) +
                        packet.k_indices[block.k_index_offset + k];
                    if (column >= col_count || lane_position != 0 ||
                        reconstructed < std::numeric_limits<int32_t>::min() ||
                        reconstructed > std::numeric_limits<int32_t>::max()) {
                        return;
                    }
                    int32_t sum = 0;
                    if (__builtin_add_overflow(
                            staged[global_row * col_count + column],
                            static_cast<int32_t>(reconstructed), &sum)) {
                        return;
                    }
                    staged[global_row * col_count + column] = sum;
                }
            }
        }
    }
    plane.swap(staged);
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
                               size_t correction_size,
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
        correction_size != layout.value_count ||
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

bool common_column_scale_matches(const ggml_gemmini_args_t & args,
                                 const wroute::WeightRoutePlan & plan,
                                 size_t j,
                                 size_t block_id) {
    if (plan.route != wroute::WeightRouteKind::H1 &&
        plan.route != wroute::WeightRouteKind::HP1) {
        return true;
    }
    const wreader::WeightScaleResult reference =
        wreader::read_scale(args, plan, j, 0);
    const wreader::WeightScaleResult current =
        wreader::read_scale(args, plan, j, block_id);
    return reference.ok() && current.ok() &&
        reference.domain == wroute::WeightScaleDomain::IntegerBlockTimesColumn &&
        current.domain == reference.domain &&
        current.column_scale == reference.column_scale;
}

double saturate_signed_32(double value) {
    return std::clamp(
        value,
        static_cast<double>(std::numeric_limits<int32_t>::min()),
        static_cast<double>(std::numeric_limits<int32_t>::max()));
}

RmdStatus merge_rmd_correction_checked(const ggml_gemmini_args_t & args,
                                       const MergeLayout & layout,
                                       const wroute::WeightRoutePlan & plan,
                                       const Correction & correction) {
    const auto * integer = std::get_if<BlockScaledInt64Correction>(&correction);
    const auto * floating = std::get_if<PreScaledFloat64Correction>(&correction);
    const bool integer_route =
        plan.scale_domain == wroute::WeightScaleDomain::IntegerBlockTimesColumn &&
        wroute::route_supports_integer_block_scale(plan);
    const bool floating_route =
        plan.route == wroute::WeightRouteKind::H0 &&
        plan.scale_domain == wroute::WeightScaleDomain::FloatingBlock;
    if ((integer != nullptr) != integer_route || (floating != nullptr) != floating_route) {
        return RmdStatus::unsupported_route;
    }

    const quants::act::ActivationMetadataView metadata(
        args, layout.metadata_row_begin, layout.metadata_row_end);
    if (!metadata.valid()) {
        return RmdStatus::invalid_arguments;
    }

    std::vector<float> column_scale;
    std::vector<float> activation_scale;
    std::vector<float> staged_output;
    try {
        if (integer != nullptr) column_scale.resize(args.J);
        activation_scale.resize(layout.row_count);
        staged_output.resize(layout.value_count);
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    if (integer != nullptr) {
        for (size_t j = 0; j < args.J; ++j) {
            column_scale[j] = wroute::route_column_scale(plan, args, j);
            if (!finite_float_representation(column_scale[j])) {
                return RmdStatus::unsupported_route;
            }
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
            double domain_value = 0.0;
            if (integer != nullptr) {
                domain_value = static_cast<double>(integer->values[source_row + j]) *
                    static_cast<double>(column_scale[j]);
            } else {
                const double value = floating->values[source_row + j];
                if (!finite_double_representation(value)) return RmdStatus::overflow;
                domain_value = saturate_signed_32(value);
            }
            const double scaled = domain_value *
                static_cast<double>(activation_scale[row]);
            const float delta = static_cast<float>(scaled);
            const float merged = layout.destination[destination_row + j * layout.col_stride] + delta;
            if (!finite_double_representation(domain_value) ||
                !finite_double_representation(scaled) ||
                !finite_float_representation(delta) ||
                !finite_float_representation(merged)) {
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
                                  const Correction & correction) {
    MergeLayout layout;
    const RmdStatus dimensions = prepare_merge_layout(
        args, destination, global_row_begin, global_row_end,
        correction_size(correction), layout);
    if (dimensions != RmdStatus::success) {
        return dimensions;
    }

    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid) {
        return RmdStatus::unsupported_route;
    }
    if (wroute::route_supports_integer_block_scale(plan)) {
        for (size_t j = 0; j < args.J; ++j) {
            for (size_t block = 0; block < plan.scales.cols; ++block) {
                if (!common_column_scale_matches(args, plan, j, block)) {
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
                                  const Correction & correction) {
    if (std::get_if<BlockScaledInt64Correction>(&correction) == nullptr) {
        return RmdStatus::unsupported_route;
    }
    size_t global_row_end = 0;
    size_t value_count = 0;
    if (packet.logical_j != args.J ||
        !checked_add_size(packet.row_begin, packet.row_count, global_row_end) ||
        !checked_mul_size(packet.row_count, packet.logical_j, value_count) ||
        correction_size(correction) != value_count) {
        return RmdStatus::invalid_arguments;
    }

    MergeLayout layout;
    const RmdStatus dimensions = prepare_merge_layout(
        args, destination, packet.row_begin, global_row_end,
        correction_size(correction), layout);
    if (dimensions != RmdStatus::success) {
        return dimensions;
    }

    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid || !wroute::route_supports_integer_block_scale(plan)) {
        return RmdStatus::unsupported_route;
    }
    for (const BlockDescriptor & block : packet.blocks) {
        for (size_t j = 0; j < args.J; ++j) {
            if (!common_column_scale_matches(args, plan, j, block.block_id)) {
                return RmdStatus::unsupported_route;
            }
        }
    }
    return merge_rmd_correction_checked(args, layout, plan, correction);
}


RmdStatus merge_rmd_correction(const ggml_gemmini_args_t & args,
                               size_t global_row_begin,
                               size_t global_row_end,
                               const Correction & correction) {
    return merge_rmd_correction_to(
        args, args.f_out, global_row_begin, global_row_end, correction);
}

RmdStatus merge_rmd_correction(const ggml_gemmini_args_t & args,
                               const StripePacket & packet,
                               const Correction & correction) {
    return merge_rmd_correction_to(args, args.f_out, packet, correction);
}

}
