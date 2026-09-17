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
#include <stdexcept>
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

}

RmdStatus compose_rmd_output(const StripePacket & packet,
                             const CompressedOutput & output,
                             Correction & correction,
                             RmdExecutionMetrics * metrics) {
    const RmdStatus validation = validate_packet(packet);
    if (validation != RmdStatus::success) {
        return validation;
    }
    if (output.domain != CompressedOutput::Domain::block_scaled_int64 ||
        output.j_padded != packet.j_padded ||
        output.values.size() != packet.total_output_values) {
        return RmdStatus::invalid_arguments;
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
                for (uint8_t lane_position = 0;
                     lane_position < block.active_lane_count; ++lane_position) {
                    const uint8_t lane = block.lane_ids[lane_position];
                    const size_t lane_base = block.output_value_offset +
                        lane_position * block.lane_stride_values;
                    __int128 contribution = output.values[
                        lane_base + row * output.j_padded + j];
                    if (lane != 0) {
                        const __int128 place = static_cast<__int128>(1) <<
                            (packet.digit_bits * lane);
                        if (__builtin_mul_overflow(contribution, place, &contribution)) {
                            return RmdStatus::overflow;
                        }
                    }
                    if (__builtin_add_overflow(total, contribution, &total)) {
                        return RmdStatus::overflow;
                    }
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
    if (metrics != nullptr) metrics->correction_bytes = value_count * sizeof(OutputValue);
    return RmdStatus::success;
}

RmdStatus expand_packets_to_plane(
        const std::vector<StripePacketHandle> & packets,
        size_t global_row_begin,
        size_t global_row_end,
        size_t col_count,
        std::vector<int32_t> & plane) {
    if (global_row_begin >= global_row_end || col_count == 0) {
        return RmdStatus::invalid_arguments;
    }
    const size_t row_count = global_row_end - global_row_begin;
    size_t value_count = 0;
    if (!checked_mul_size(row_count, col_count, value_count)) {
        return RmdStatus::overflow;
    }
    std::vector<int32_t> staged;
    try {
        staged.assign(value_count, 0);
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    } catch (const std::length_error &) {
        return RmdStatus::allocation_failure;
    }

    for (const StripePacketHandle & handle : packets) {
        if (!handle) {
            return RmdStatus::invalid_packet;
        }
        const RmdStatus validation = validate_packet(*handle);
        if (validation != RmdStatus::success) {
            return validation;
        }
        const StripePacket & packet = *handle;
        const BalancedRadixContract contract =
            balanced_radix_contract(packet.digit_bits);
        for (const BlockDescriptor & block : packet.blocks) {
            for (size_t row = 0; row < packet.row_count; ++row) {
                const size_t global_row = packet.row_begin + row;
                for (size_t k = 0; k < block.compact_k_count; ++k) {
                    int64_t reconstructed = 0;
                    size_t lane_position = block.active_lane_count;
                    for (uint8_t lane = packet.lane_capacity; lane-- > 0;) {
                        if (__builtin_mul_overflow(
                                reconstructed, static_cast<int64_t>(contract.radix),
                                &reconstructed)) {
                            return RmdStatus::overflow;
                        }
                        if (lane_position != 0 &&
                            block.lane_ids[lane_position - 1] == lane) {
                            --lane_position;
                            int32_t digit = 0;
                            const RmdStatus read_status =
                                read_packet_digit(
                                    packet, block,
                                    static_cast<uint8_t>(lane_position),
                                    row, k, digit);
                            if (read_status != RmdStatus::success) {
                                return read_status;
                            }
                            if (__builtin_add_overflow(
                                    reconstructed, static_cast<int64_t>(digit),
                                    &reconstructed)) {
                                return RmdStatus::overflow;
                            }
                        }
                    }
                    size_t column = 0;
                    if (!checked_add_size(
                            static_cast<size_t>(block.global_k_begin),
                            packet.k_indices[block.k_index_offset + k],
                            column) ||
                        lane_position != 0 ||
                        reconstructed < std::numeric_limits<int32_t>::min() ||
                        reconstructed > std::numeric_limits<int32_t>::max()) {
                        return RmdStatus::invalid_packet;
                    }
                    if (global_row < global_row_begin ||
                        global_row >= global_row_end ||
                        column >= col_count) {
                        continue;
                    }
                    const size_t local_row = global_row - global_row_begin;
                    int32_t sum = 0;
                    if (__builtin_add_overflow(
                            staged[local_row * col_count + column],
                            static_cast<int32_t>(reconstructed), &sum)) {
                        return RmdStatus::overflow;
                    }
                    staged[local_row * col_count + column] = sum;
                }
            }
        }
    }
    plane.swap(staged);
    return RmdStatus::success;
}

void expand_packets_to_plane(const std::vector<StripePacketHandle> & packets,
                             size_t row_count,
                             size_t col_count,
                             std::vector<int32_t> & plane) {
    (void) expand_packets_to_plane(
        packets, 0, row_count, col_count, plane);
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

RmdStatus prepare_column_scales(const ggml_gemmini_args_t & args,
                                const wroute::WeightRoutePlan & plan,
                                const StripePacket * packet,
                                std::vector<float> & column_scale) {
    if (plan.route != wroute::WeightRouteKind::H1 &&
        plan.route != wroute::WeightRouteKind::HP1) return RmdStatus::success;
    try {
        column_scale.resize(args.J);
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }
    for (size_t j = 0; j < args.J; ++j) {
        const auto reference = wreader::read_scale_validated(args, plan, j, 0);
        if (!reference.ok() || reference.domain !=
                wroute::WeightScaleDomain::IntegerBlockTimesColumn) {
            return RmdStatus::unsupported_route;
        }
        // Cache the column factor for all rows, but verify it matches each relevant block.
        // Packet merges check selected blocks; callers without a packet check every block.
        column_scale[j] = reference.column_scale;
        const size_t block_count = packet != nullptr ? packet->blocks.size() : plan.scales.cols;
        for (size_t block = 0; block < block_count; ++block) {
            const size_t block_id = packet != nullptr ? packet->blocks[block].block_id : block;
            if (block_id == 0) continue;
            const auto current = wreader::read_scale_validated(args, plan, j, block_id);
            if (!current.ok() || current.domain != reference.domain ||
                current.column_scale != reference.column_scale) {
                return RmdStatus::unsupported_route;
            }
        }
        if (!finite_float_representation(column_scale[j])) return RmdStatus::unsupported_route;
    }
    return RmdStatus::success;
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
                                       const std::vector<float> & prepared_column_scale,
                                       const Correction & correction,
                                       size_t * nonzero_count,
                                       RmdExecutionMetrics * metrics) {
    detail::RmdHostStageScope preparation(metrics, RmdHostStage::final_metadata);
    const auto * integer = std::get_if<BlockScaledInt64Correction>(&correction);
    const auto * floating = std::get_if<PreScaledFloat64Correction>(&correction);
    const auto * fully_scaled = std::get_if<FullyScaledFloat64Correction>(&correction);
    const bool integer_route =
        plan.scale_domain == wroute::WeightScaleDomain::IntegerBlockTimesColumn &&
        wroute::route_supports_integer_block_scale(plan);
    const bool floating_route =
        plan.route == wroute::WeightRouteKind::H0 &&
        plan.scale_domain == wroute::WeightScaleDomain::FloatingBlock;
    const bool block_activation =
        std::holds_alternative<quants::act::block::Meta>(args.act_quant.storage());
    if (fully_scaled != nullptr ?
            (!block_activation || (!integer_route && !floating_route)) :
            (block_activation || (integer != nullptr) != integer_route ||
             (floating != nullptr) != floating_route)) {
        return RmdStatus::unsupported_route;
    }

    const quants::act::ActivationMetadataView metadata(
        args, layout.metadata_row_begin, layout.metadata_row_end);
    if (!metadata.valid()) return RmdStatus::invalid_arguments;

    const bool columns_prepared = !prepared_column_scale.empty();
    std::vector<float> unprepared_column_scale;
    const auto & column_scale = columns_prepared ? prepared_column_scale : unprepared_column_scale;
    std::vector<float> activation_scale;
    std::vector<float> staged_output;
    try {
        if (integer != nullptr && !columns_prepared) unprepared_column_scale.resize(args.J);
        if (fully_scaled == nullptr) activation_scale.resize(layout.row_count);
        staged_output.resize(layout.value_count);
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    if (integer != nullptr && !columns_prepared) {
        for (size_t j = 0; j < args.J; ++j) {
            unprepared_column_scale[j] = wroute::route_column_scale(plan, args, j);
            if (!finite_float_representation(column_scale[j])) {
                return RmdStatus::unsupported_route;
            }
        }
    }
    if (fully_scaled == nullptr) {
        for (size_t row = 0; row < layout.row_count; ++row) {
            if (!metadata.scale(row, activation_scale[row])) {
                return RmdStatus::invalid_arguments;
            }
        }
    }
    preparation.finish();
    detail::RmdHostStageScope combine(metrics, RmdHostStage::final_scale_combine_stage);

    // Count composed raw corrections, before scales or H0 saturation: lane terms may cancel,
    // and a zero scale must not erase a nonzero correction from this statistic.
    size_t staged_nonzero_count = 0;
    for (size_t row = 0; row < layout.row_count; ++row) {
        const size_t destination_row =
            (layout.global_row_begin + row) * layout.row_stride;
        const size_t source_row = row * args.J;
        for (size_t j = 0; j < args.J; ++j) {
            double domain_value = 0.0;
            if (fully_scaled != nullptr) {
                domain_value = fully_scaled->values[source_row + j];
                staged_nonzero_count += domain_value != 0.0;
                if (!finite_double_representation(domain_value)) return RmdStatus::overflow;
            } else if (integer != nullptr) {
                const int64_t value = integer->values[source_row + j];
                staged_nonzero_count += value != 0;
                domain_value = static_cast<double>(value) *
                    static_cast<double>(column_scale[j]);
            } else {
                const double value = floating->values[source_row + j];
                staged_nonzero_count += value != 0;
                if (!finite_double_representation(value)) return RmdStatus::overflow;
                domain_value = saturate_signed_32(value);
            }
            const double scaled = fully_scaled != nullptr ? domain_value :
                domain_value * static_cast<double>(activation_scale[row]);
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

    combine.finish();
    detail::RmdHostStageScope store(metrics, RmdHostStage::output_store);
    for (size_t row = 0; row < layout.row_count; ++row) {
        const size_t destination_row =
            (layout.global_row_begin + row) * layout.row_stride;
        const size_t source_row = row * args.J;
        for (size_t j = 0; j < args.J; ++j) {
            layout.destination[destination_row + j * layout.col_stride] =
                staged_output[source_row + j];
        }
    }
    // Publish the count only after the whole merge succeeds, just like the destination.
    if (nonzero_count != nullptr) *nonzero_count = staged_nonzero_count;
    if (metrics != nullptr) {
        metrics->final_scale_values_bytes = (layout.row_count + (integer != nullptr ? args.J : 0)) * sizeof(float);
        metrics->final_output_store_bytes = layout.value_count * sizeof(float);
    }
    return RmdStatus::success;
}

}

RmdStatus compose_block_rmd_output(const ggml_gemmini_args_t & args,
                                   const StripePacket & packet,
                                   const CompressedOutput & output,
                                   Correction & correction) {
    const RmdStatus validation = validate_packet(packet);
    if (validation != RmdStatus::success) return validation;
    if (!std::holds_alternative<quants::act::block::Meta>(args.act_quant.storage()) ||
        output.domain != CompressedOutput::Domain::block_scaled_int64 ||
        output.j_padded != packet.j_padded ||
        output.values.size() != packet.total_output_values ||
        packet.logical_k != args.K || packet.logical_j != args.J) {
        return RmdStatus::invalid_arguments;
    }

    size_t metadata_row_begin = 0;
    size_t metadata_row_end = 0;
    size_t packet_row_end = 0;
    if (!checked_add_size(packet.row_begin, packet.row_count, packet_row_end) ||
        !checked_add_size(args.activation_row_offset, packet.row_begin,
                          metadata_row_begin) ||
        !checked_add_size(args.activation_row_offset, packet_row_end,
                          metadata_row_end)) {
        return RmdStatus::overflow;
    }
    const quants::act::ActivationMetadataView metadata(
        args, metadata_row_begin, metadata_row_end);
    if (!metadata.valid()) return RmdStatus::invalid_arguments;

    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid || (plan.route != wroute::WeightRouteKind::H1 &&
                        plan.route != wroute::WeightRouteKind::HP1)) {
        return RmdStatus::unsupported_route;
    }
    std::vector<float> column_scale;
    const RmdStatus scales = prepare_column_scales(args, plan, &packet, column_scale);
    if (scales != RmdStatus::success) return scales;

    size_t value_count = 0;
    if (!checked_mul_size(packet.row_count, packet.logical_j, value_count)) {
        return RmdStatus::overflow;
    }
    FullyScaledFloat64Correction staged;
    try {
        staged.values.assign(value_count, 0.0);
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    for (size_t row = 0; row < packet.row_count; ++row) {
        for (const BlockDescriptor & block : packet.blocks) {
            float activation_scale = 0.0f;
            if (!metadata.scale(row, block.global_k_begin, activation_scale)) {
                return RmdStatus::invalid_arguments;
            }
            for (size_t j = 0; j < packet.logical_j; ++j) {
                __int128 block_total = 0;
                for (uint8_t lane_position = 0;
                     lane_position < block.active_lane_count; ++lane_position) {
                    const uint8_t lane = block.lane_ids[lane_position];
                    const size_t lane_base = block.output_value_offset +
                        static_cast<size_t>(lane_position) * block.lane_stride_values;
                    __int128 contribution = output.values[
                        lane_base + row * output.j_padded + j];
                    const __int128 place = static_cast<__int128>(1) <<
                        (packet.digit_bits * lane);
                    if (__builtin_mul_overflow(contribution, place, &contribution) ||
                        __builtin_add_overflow(block_total, contribution, &block_total)) {
                        return RmdStatus::overflow;
                    }
                }
                if (block_total > kInt64Max || block_total < kInt64Min) {
                    return RmdStatus::overflow;
                }
                const double scaled = static_cast<double>(static_cast<int64_t>(block_total)) *
                    static_cast<double>(activation_scale) *
                    static_cast<double>(column_scale[j]);
                double & total = staged.values[row * packet.logical_j + j];
                const double sum = total + scaled;
                if (!finite_double_representation(scaled) ||
                    !finite_double_representation(sum)) {
                    return RmdStatus::overflow;
                }
                total = sum;
            }
        }
    }
    correction = std::move(staged);
    return RmdStatus::success;
}

RmdStatus merge_rmd_correction_to(const ggml_gemmini_args_t & args,
                                  float * destination,
                                  size_t global_row_begin,
                                  size_t global_row_end,
                                  const Correction & correction,
                                  size_t * nonzero_count,
                                  RmdExecutionMetrics * metrics) {
    detail::RmdHostStageScope preparation(metrics, RmdHostStage::final_metadata);
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
    std::vector<float> column_scale;
    const RmdStatus scales = prepare_column_scales(args, plan, nullptr, column_scale);
    if (scales != RmdStatus::success) return scales;
    preparation.finish();
    return merge_rmd_correction_checked(args, layout, plan, column_scale, correction, nonzero_count, metrics);
}

RmdStatus merge_rmd_correction_to(const ggml_gemmini_args_t & args,
                                  float * destination,
                                  const StripePacket & packet,
                                  const Correction & correction,
                                  size_t * nonzero_count,
                                  RmdExecutionMetrics * metrics) {
    detail::RmdHostStageScope preparation(metrics, RmdHostStage::final_metadata);
    if (std::get_if<BlockScaledInt64Correction>(&correction) == nullptr &&
        std::get_if<FullyScaledFloat64Correction>(&correction) == nullptr) {
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
    std::vector<float> column_scale;
    const RmdStatus scales = prepare_column_scales(args, plan, &packet, column_scale);
    if (scales != RmdStatus::success) return scales;
    preparation.finish();
    return merge_rmd_correction_checked(args, layout, plan, column_scale, correction, nonzero_count, metrics);
}

namespace detail {
const wroute::WeightRoutePlan & RmdWeightPreparation::route_plan(
    const ggml_gemmini_args_t & args) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!plan_ready_) {
        plan_ = wroute::resolve_weight_route_plan(args, wroute::WeightScaleInfoMode::Residual);
        plan_ready_ = true;
    }
    return plan_;
}

RmdStatus RmdWeightPreparation::prepare_columns(
    const ggml_gemmini_args_t & args, const StripePacket & packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool native_columns = plan_.route == wroute::WeightRouteKind::H1 ||
        plan_.route == wroute::WeightRouteKind::HP1;
    if (!columns_ready_) {
        std::vector<float> columns;
        std::vector<uint8_t> checked;
        try {
            columns.resize(args.J);
            if (native_columns) checked.resize(plan_.scales.cols, 0);
        } catch (const std::bad_alloc &) {
            return RmdStatus::allocation_failure;
        } catch (const std::length_error &) {
            return RmdStatus::allocation_failure;
        }
        for (size_t j = 0; j < args.J; ++j) {
            if (native_columns) {
                const auto reference = wreader::read_scale_validated(args, plan_, j, 0);
                if (!reference.ok() || reference.domain !=
                        wroute::WeightScaleDomain::IntegerBlockTimesColumn) {
                    return RmdStatus::unsupported_route;
                }
                columns[j] = reference.column_scale;
            } else {
                columns[j] = wroute::route_column_scale(plan_, args, j);
            }
            if (!finite_float_representation(columns[j])) return RmdStatus::unsupported_route;
        }
        column_scale_.swap(columns);
        checked_blocks_.swap(checked);
        columns_ready_ = true;
#if defined(GGML_GEMMINI_TESTING)
        ++column_preparations_;
#endif
    }
    if (!native_columns) return RmdStatus::success;
    for (const auto & block : packet.blocks) {
        if (block.block_id >= checked_blocks_.size()) return RmdStatus::unsupported_route;
        if (block.block_id == 0 || checked_blocks_[block.block_id] != 0) continue;
        for (size_t j = 0; j < args.J; ++j) {
            const auto current = wreader::read_scale_validated(args, plan_, j, block.block_id);
            if (!current.ok() || current.domain !=
                    wroute::WeightScaleDomain::IntegerBlockTimesColumn ||
                current.column_scale != column_scale_[j]) {
                return RmdStatus::unsupported_route;
            }
        }
        checked_blocks_[block.block_id] = 1;
#if defined(GGML_GEMMINI_TESTING)
        ++selected_block_preparations_;
#endif
    }
    return RmdStatus::success;
}

RmdStatus merge_rmd_correction_with_weights(const ggml_gemmini_args_t & args,
    float * destination, const StripePacket & packet, const Correction & correction,
    RmdWeightPreparation & weights, size_t * nonzero_count, RmdExecutionMetrics * metrics) {
    RmdHostStageScope preparation(metrics, RmdHostStage::final_metadata);
    if (std::get_if<BlockScaledInt64Correction>(&correction) == nullptr &&
        std::get_if<FullyScaledFloat64Correction>(&correction) == nullptr) {
        return RmdStatus::unsupported_route;
    }
    size_t global_row_end = 0;
    if (packet.logical_j != args.J || packet.logical_k != args.K ||
        !checked_add_size(packet.row_begin, packet.row_count, global_row_end)) {
        return RmdStatus::invalid_arguments;
    }
    MergeLayout layout;
    const RmdStatus dimensions = prepare_merge_layout(args, destination,
        packet.row_begin, global_row_end, correction_size(correction), layout);
    if (dimensions != RmdStatus::success) return dimensions;
    const auto & plan = weights.route_plan(args);
    if (!plan.valid || !wroute::route_supports_integer_block_scale(plan)) {
        return RmdStatus::unsupported_route;
    }
    const RmdStatus scales = weights.prepare_columns(args, packet);
    if (scales != RmdStatus::success) return scales;
    preparation.finish();
    return merge_rmd_correction_checked(
        args, layout, plan, weights.column_scale_, correction, nonzero_count, metrics);
}

}


RmdStatus merge_rmd_correction(const ggml_gemmini_args_t & args,
                               size_t global_row_begin,
                               size_t global_row_end,
                               const Correction & correction,
                               size_t * nonzero_count,
                               RmdExecutionMetrics * metrics) {
    return merge_rmd_correction_to(
        args, args.f_out, global_row_begin, global_row_end, correction, nonzero_count, metrics);
}

RmdStatus merge_rmd_correction(const ggml_gemmini_args_t & args,
                               const StripePacket & packet,
                               const Correction & correction,
                               size_t * nonzero_count,
                               RmdExecutionMetrics * metrics) {
    return merge_rmd_correction_to(args, args.f_out, packet, correction, nonzero_count, metrics);
}

}
