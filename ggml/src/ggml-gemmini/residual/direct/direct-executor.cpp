#include "direct-executor.hpp"

#include "direct-builder.hpp"
#include "../../ggml-gemmini-args.h"
#include "../../quants/common/weight_route.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <new>

namespace ggml::gemmini::residual {

namespace {

namespace wroute = quants::wroute;

constexpr size_t kJTile = 16;
constexpr __int128 kInt64Min = static_cast<__int128>(std::numeric_limits<int64_t>::min());
constexpr __int128 kInt64Max = static_cast<__int128>(std::numeric_limits<int64_t>::max());

bool checked_add(int64_t lhs, __int128 rhs, int64_t & result) {
    const __int128 sum = static_cast<__int128>(lhs) + rhs;
    if (sum < kInt64Min || sum > kInt64Max) return false;
    result = static_cast<int64_t>(sum);
    return true;
}

bool checked_size_product(size_t lhs, size_t rhs, size_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) return false;
    result = lhs * rhs;
    return true;
}

bool dense_route_is_addressable(const wroute::WeightRoutePlan & plan,
                                size_t k_count, size_t j_count) {
    if (plan.native_weight_blocks) return true;
    if (plan.weight_stride == 0) return false;
    const bool column_major = plan.layout == wroute::WeightLayout::JxK_ColMajor;
    const size_t major_count = column_major ? j_count : k_count;
    const size_t minor_count = column_major ? k_count : j_count;
    if (plan.weight_stride < minor_count) return false;
    size_t major_offset = 0;
    return checked_size_product(major_count - 1, plan.weight_stride, major_offset) &&
        minor_count - 1 <= std::numeric_limits<size_t>::max() - major_offset;
}

bool read_weight_code(const ggml_gemmini_args_t & args,
                      const wroute::WeightRoutePlan & plan,
                      size_t k, size_t j, int8_t & code) {
    if (plan.route == wroute::WeightRouteKind::Q8H1 && plan.native_weight_blocks) {
        const block_q8_h1 * block = args.q8_h1_block(j, k / rmd::kBlockSize);
        if (block == nullptr) return false;
        code = static_cast<int8_t>(block->qs[k % rmd::kBlockSize]);
        return true;
    }
    const int8_t * dense = reinterpret_cast<const int8_t *>(args.B);
    if (dense == nullptr) return false;
    const size_t offset = plan.layout == wroute::WeightLayout::JxK_ColMajor
        ? j * plan.weight_stride + k
        : k * plan.weight_stride + j;
    code = dense[offset];
    return true;
}

}

rmd::RmdStatus execute_direct_stripe(const ggml_gemmini_args_t & args,
                                     const DirectStripePayload & payload,
                                     std::vector<rmd::OutputValue> & correction,
                                     DirectExecutionMetrics * metrics) {
    if (validate_direct_payload(payload) != rmd::RmdStatus::success)
        return rmd::RmdStatus::invalid_packet;
    if (args.K != payload.logical_k || args.J != payload.logical_j)
        return rmd::RmdStatus::invalid_arguments;

    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid || !wroute::route_supports_integer_block_scale(plan))
        return rmd::RmdStatus::unsupported_route;
    if (!wroute::route_covers_k(plan, payload.logical_k) ||
        !dense_route_is_addressable(plan, payload.logical_k, payload.logical_j))
        return rmd::RmdStatus::unsupported_route;

    size_t output_count = 0;
    if (!checked_size_product(payload.row_count, payload.logical_j, output_count))
        return rmd::RmdStatus::overflow;

    std::vector<rmd::OutputValue> staged;
    if (output_count > staged.max_size()) return rmd::RmdStatus::allocation_failure;
    try {
        staged.assign(output_count, rmd::OutputValue{0});
    } catch (const std::bad_alloc &) {
        return rmd::RmdStatus::allocation_failure;
    }

    // Events are canonical row/K order. For each J tile, consume contiguous
    // row/block/K spans, then scale the completed K sum once for that block.
    for (size_t j_begin = 0; j_begin < payload.logical_j; j_begin += kJTile) {
        const size_t tile_j = std::min(kJTile, payload.logical_j - j_begin);
        size_t event_index = 0;
        while (event_index < payload.events.size()) {
            const ResidualEvent & first = payload.events[event_index];
            const size_t row = first.local_row;
            const size_t block_id = first.original_k / rmd::kBlockSize;
            size_t span_end = event_index + 1;
            while (span_end < payload.events.size() &&
                   payload.events[span_end].local_row == row &&
                   payload.events[span_end].original_k / rmd::kBlockSize == block_id) {
                ++span_end;
            }

            std::array<int64_t, kJTile> block_sum{};
            for (size_t index = event_index; index < span_end; ++index) {
                const ResidualEvent & event = payload.events[index];
                for (size_t local_j = 0; local_j < tile_j; ++local_j) {
                    int8_t code = 0;
                    if (!read_weight_code(args, plan, event.original_k,
                                          j_begin + local_j, code))
                        return rmd::RmdStatus::execution_failed;
                    const __int128 product = static_cast<__int128>(event.residual) * code;
                    if (!checked_add(block_sum[local_j], product, block_sum[local_j]))
                        return rmd::RmdStatus::overflow;
                }
            }

            for (size_t local_j = 0; local_j < tile_j; ++local_j) {
                const size_t j = j_begin + local_j;
                const uint64_t scale = wroute::route_block_scale(plan, args, j, block_id);
                if (scale > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                    return rmd::RmdStatus::overflow;
                const __int128 scaled = static_cast<__int128>(block_sum[local_j]) *
                    static_cast<int64_t>(scale);
                const size_t output_index = row * payload.logical_j + j;
                if (!checked_add(staged[output_index], scaled, staged[output_index]))
                    return rmd::RmdStatus::overflow;
            }
            event_index = span_end;
        }
    }

    correction.swap(staged);
    if (metrics != nullptr) {
        metrics->event_count = payload.events.size();
        metrics->call_count = 1;
    }
    return rmd::RmdStatus::success;
}

}
