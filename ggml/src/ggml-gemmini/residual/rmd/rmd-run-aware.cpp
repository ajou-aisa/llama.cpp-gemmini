#include "rmd-run-aware.hpp"

#include "rmd-builder.hpp"
#include "../../ggml-gemmini-args.h"
#include "../../quants/common/weight_reader.hpp"
#include "../../quants/common/weight_route.hpp"

#include <gemmini.h>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace ggml::gemmini::rmd {
namespace {

namespace wreader = quants::wreader;
namespace wroute = quants::wroute;

bool checked_add(size_t left, size_t right, size_t &out) {
    if (right > std::numeric_limits<size_t>::max() - left) return false;
    out = left + right;
    return true;
}

bool checked_mul(size_t left, size_t right, size_t &out) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) return false;
    out = left * right;
    return true;
}

const BlockDescriptor *find_block_lane(const BlockDescriptor &block,
                                       uint8_t lane_id,
                                       uint8_t &lane_position) {
    for (uint8_t position = 0; position < block.active_lane_count; ++position) {
        if (block.lane_ids[position] == lane_id) {
            lane_position = position;
            return &block;
        }
    }
    return nullptr;
}

RmdStatus map_plan_status(const wroute::WeightRoutePlan &plan) {
    if (plan.status == wroute::WeightRouteStatus::InvalidMetadata)
        return RmdStatus::invalid_arguments;
    return RmdStatus::unsupported_route;
}

} // namespace

RmdStatus build_run_aware_request(const ggml_gemmini_args_t &args,
                                  const StripePacketHandle &packet,
                                  RunAwareRequest &out) {
    if (!packet) {
        out = {};
        return RmdStatus::success;
    }
    return build_run_aware_request(args, *packet, out);
}

RmdStatus build_run_aware_request(const ggml_gemmini_args_t &args,
                                  const StripePacket &packet,
                                  RunAwareRequest &out) {
    if (validate_packet(packet) != RmdStatus::success)
        return RmdStatus::invalid_packet;
    if (args.J != packet.logical_j || args.K != packet.logical_k)
        return RmdStatus::invalid_arguments;

    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::ResidualHp1Scu);
    if (!plan.valid) return map_plan_status(plan);
    if (!plan.hp1_carriers || plan.route != wroute::WeightRouteKind::HP1)
        return RmdStatus::unsupported_route;

    try {
        RunAwareRequest staged;
        staged.operand_bits = packet.digit_bits;
        staged.n = packet.logical_j;
        staged.original_k = packet.logical_k;
        staged.stripe_id = packet.stripe_id;
        staged.source_row_begin = packet.row_begin;
        staged.source_row_count = packet.row_count;
        staged.runs.reserve(packet.blocks.size());

        size_t compact_k = 0;
        for (const BlockDescriptor &block : packet.blocks) {
            RunAwareRun run;
            run.original_block_id = block.block_id;
            run.original_global_k_begin = block.global_k_begin;
            run.compact_k_begin = compact_k;
            run.compact_k_count = block.compact_k_count;
            run.original_local_k.reserve(block.compact_k_count);
            for (size_t index = 0; index < block.compact_k_count; ++index) {
                const uint16_t local_k = packet.k_indices[block.k_index_offset + index];
                run.original_local_k.push_back(local_k);
                run.union_k_mask |= uint32_t{1} << local_k;
            }
            if (!checked_add(compact_k, block.compact_k_count, compact_k))
                return RmdStatus::overflow;
            staged.runs.push_back(std::move(run));
        }
        staged.k = compact_k;
        if (staged.k == 0) return RmdStatus::invalid_packet;

        if (packet.row_count > std::numeric_limits<uint32_t>::max())
            return RmdStatus::overflow;
        for (uint8_t lane_id = 0; lane_id < packet.lane_capacity; ++lane_id) {
            for (size_t row = 0; row < packet.row_count; ++row) {
                bool active = false;
                for (const BlockDescriptor &block : packet.blocks) {
                    uint8_t position = 0;
                    if (!find_block_lane(block, lane_id, position)) continue;
                    for (size_t k = 0; k < block.compact_k_count; ++k) {
                        int32_t digit = 0;
                        const RmdStatus status = read_packet_digit(
                            packet, block, position, row, k, digit);
                        if (status != RmdStatus::success) return status;
                        if (digit != 0) {
                            active = true;
                            break;
                        }
                    }
                    if (active) break;
                }
                if (active)
                    staged.rows.push_back({lane_id, static_cast<uint32_t>(row)});
            }
        }
        staged.m = staged.rows.size();
        if (staged.m == 0) return RmdStatus::invalid_packet;

        size_t activation_count = 0;
        size_t weight_count = 0;
        size_t carrier_count = 0;
        if (!checked_mul(staged.m, staged.k, activation_count) ||
            !checked_mul(staged.k, staged.n, weight_count) ||
            !checked_mul(staged.runs.size(), staged.n, carrier_count) ||
            activation_count > staged.activations.max_size() ||
            weight_count > staged.weights.max_size() ||
            carrier_count > staged.carriers.max_size()) {
            return RmdStatus::overflow;
        }
        staged.activations.assign(activation_count, 0);
        staged.weights.assign(weight_count, 0);
        staged.carriers.resize(carrier_count);

        for (size_t row_index = 0; row_index < staged.rows.size(); ++row_index) {
            const RunAwareRow &row = staged.rows[row_index];
            for (size_t run_index = 0; run_index < staged.runs.size(); ++run_index) {
                const BlockDescriptor &block = packet.blocks[run_index];
                uint8_t position = 0;
                if (!find_block_lane(block, row.original_lane_id, position))
                    continue;
                for (size_t k = 0; k < block.compact_k_count; ++k) {
                    int32_t digit = 0;
                    const RmdStatus status = read_packet_digit(
                        packet, block, position, row.source_row, k, digit);
                    if (status != RmdStatus::success) return status;
                    if (digit < std::numeric_limits<int8_t>::min() ||
                        digit > std::numeric_limits<int8_t>::max())
                        return RmdStatus::overflow;
                    staged.activations[row_index * staged.k +
                                       staged.runs[run_index].compact_k_begin + k] =
                        static_cast<int8_t>(digit);
                }
            }
        }

        std::array<int32_t, DIM * DIM> tile{};
        for (size_t run_index = 0; run_index < staged.runs.size(); ++run_index) {
            const RunAwareRun &run = staged.runs[run_index];
            for (size_t k_base = 0; k_base < run.compact_k_count; k_base += DIM) {
                const size_t valid_k =
                    std::min(run.compact_k_count - k_base, static_cast<size_t>(DIM));
                for (size_t col_base = 0; col_base < staged.n; col_base += DIM) {
                    const size_t valid_cols =
                        std::min(staged.n - col_base, static_cast<size_t>(DIM));
                    size_t resolutions = 0;
                    const auto status = wreader::read_code_tile_validated(
                        args, plan, run.original_block_id,
                        run.original_local_k.data() + k_base, valid_k, col_base,
                        valid_cols, tile.data(), resolutions);
                    if (status != wreader::WeightReaderStatus::Success)
                        return RmdStatus::invalid_arguments;
                    for (size_t k = 0; k < valid_k; ++k)
                        for (size_t col = 0; col < valid_cols; ++col)
                            staged.weights[(run.compact_k_begin + k_base + k) *
                                               staged.n + col_base + col] =
                                tile[k * DIM + col];
                }
            }
            for (size_t column = 0; column < staged.n; ++column) {
                const auto carrier = wreader::read_hp1_carrier_validated(
                    args, plan, column, run.original_block_id);
                if (!carrier.ok()) return RmdStatus::invalid_arguments;
                staged.carriers[run_index * staged.n + column] = carrier.carrier;
            }
        }

        // Shape is final here. This is the sole tiling call in the builder.
        ggml_gemmini_args_t selected{};
        selected.I = staged.m;
        selected.J = staged.n;
        selected.K = staged.k;
        ggml::gemmini::gemmini_set_tile_ws(&selected);
        if (selected.tile_I == 0 || selected.tile_J == 0 || selected.tile_K == 0)
            return RmdStatus::overflow;
        staged.tile_i = selected.tile_I;
        staged.tile_j = selected.tile_J;
        staged.tile_k = selected.tile_K;

        out = std::move(staged);
        return RmdStatus::success;
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    } catch (const std::length_error &) {
        return RmdStatus::overflow;
    }
}

} // namespace ggml::gemmini::rmd
