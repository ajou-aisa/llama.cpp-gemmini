#include "direct-executor.hpp"

#include "direct-builder.hpp"
#include "../../ggml-gemmini-args.h"
#include "../../quants/common/weight_reader.hpp"
#include "../../quants/common/weight_route.hpp"
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
#include <gemmini/cycle_reader.hpp>
#include <gemmini/log.h>
#include "../../ggml-gemmini-utils/src/cycle_reader_internal.h"
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#if defined(GGML_GEMMINI_HAS_OPENMP)
#include <omp.h>
#endif
#include <new>
#include <utility>

namespace ggml::gemmini::residual {

// allow: SIZE_OK — one staged execution state machine owns failure atomicity and leaf boundaries.
namespace {

namespace wreader = quants::wreader;
namespace wroute = quants::wroute;

constexpr size_t kJTile = 16;

#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING) || \
    (CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__))
struct CpuSample {
    uint64_t value = 0;
    bool valid = false;
    uint64_t owner = 0;
    uint64_t generation = 0;
    DirectCpuTileReason reason = DirectCpuTileReason::unavailable_event;
    DirectCpuTileSource source = DirectCpuTileSource::perf_cpu_cycles;
};

struct CpuInterval {
    uint64_t value = 0;
    bool valid = false;
    DirectCpuTileReason reason = DirectCpuTileReason::invalid_start;
    DirectCpuTileReason sample_reason = DirectCpuTileReason::none;
};

#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING)
CpuSample read_cpu_sample(const testing::DirectExecutionTestHooks * hooks,
                          testing::DirectCpuSamplePoint point,
                          size_t tile_index) {
    if (hooks != nullptr && hooks->sample_reader != nullptr) {
        const auto sample = hooks->sample_reader(point, tile_index, hooks->context);
        return {sample.value, sample.valid, sample.owner, sample.generation,
                sample.reason, sample.source};
    }
    return {};
}
#else
CpuSample read_cpu_sample() {
    const auto sample = cycle::read_sample();
    return {sample.value, sample.valid, sample.owner_event_token, sample.generation,
            static_cast<DirectCpuTileReason>(sample.reason),
            DirectCpuTileSource::perf_cpu_cycles};
}
#endif

CpuInterval cpu_interval(const CpuSample & start, const CpuSample & end) {
#if CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__)
    const cycle::NativeCycleSample native_start{
        start.value, start.valid, static_cast<cycle::NativeCycleReason>(start.reason),
        cycle::NativeCycleSource::perf_cpu_cycles, start.owner, start.generation};
    const cycle::NativeCycleSample native_end{
        end.value, end.valid, static_cast<cycle::NativeCycleReason>(end.reason),
        cycle::NativeCycleSource::perf_cpu_cycles, end.owner, end.generation};
    const cycle::NativeCycleDelta delta = cycle::evaluate_interval(native_start, native_end);
    return {delta.value, delta.valid,
            static_cast<DirectCpuTileReason>(delta.reason),
            static_cast<DirectCpuTileReason>(delta.sample_reason)};
#else
    if (!start.valid)
        return {0, false, DirectCpuTileReason::invalid_start, start.reason};
    if (!end.valid)
        return {0, false, DirectCpuTileReason::invalid_end, end.reason};
    if (start.source != end.source)
        return {0, false, DirectCpuTileReason::source_mismatch,
                DirectCpuTileReason::none};
    if (start.owner != end.owner)
        return {0, false, DirectCpuTileReason::event_owner_mismatch,
                DirectCpuTileReason::none};
    if (start.generation != end.generation)
        return {0, false, DirectCpuTileReason::event_generation_mismatch,
                DirectCpuTileReason::none};
    if (end.value < start.value)
        return {0, false, DirectCpuTileReason::counter_regression,
                DirectCpuTileReason::none};
    return {end.value - start.value, true, DirectCpuTileReason::none,
            DirectCpuTileReason::none};
#endif
}

size_t direct_worker_id() {
#if defined(GGML_GEMMINI_HAS_OPENMP)
    return static_cast<size_t>(omp_get_thread_num());
#else
    return 0;
#endif
}

#endif

bool checked_add(int64_t lhs, int64_t rhs, int64_t & result) {
    if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool checked_multiply(int64_t lhs, int64_t rhs, int64_t & result) {
    if (lhs > 0) {
        if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) ||
            (rhs < 0 && rhs < std::numeric_limits<int64_t>::min() / lhs)) {
            return false;
        }
    } else if (lhs < 0) {
        if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() / rhs) ||
            (rhs < 0 && lhs < std::numeric_limits<int64_t>::max() / rhs)) {
            return false;
        }
    }
    result = lhs * rhs;
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

bool uses_reader_scale(const wroute::WeightRoutePlan & plan) {
    return plan.route == wroute::WeightRouteKind::H0 ||
        plan.route == wroute::WeightRouteKind::H1 ||
        plan.route == wroute::WeightRouteKind::HP1;
}

bool finite_double(double value) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double representation");
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

rmd::RmdStatus reader_failure(wreader::WeightReaderStatus status) {
    return status == wreader::WeightReaderStatus::ScaleOverflow ?
        rmd::RmdStatus::overflow : rmd::RmdStatus::execution_failed;
}

}

#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING)
static rmd::RmdStatus execute_direct_stripe_impl(
    const ggml_gemmini_args_t & args,
    const DirectStripePayload & payload,
    rmd::DirectOutput & correction,
    DirectExecutionMetrics * metrics,
    const testing::DirectExecutionTestHooks * hooks) {
#else
rmd::RmdStatus execute_direct_stripe(const ggml_gemmini_args_t & args,
                                     const DirectStripePayload & payload,
                                     rmd::DirectOutput & correction,
                                     DirectExecutionMetrics * metrics) {
#endif
    if (validate_direct_payload(payload) != rmd::RmdStatus::success)
        return rmd::RmdStatus::invalid_packet;
    if (args.K != payload.logical_k || args.J != payload.logical_j)
        return rmd::RmdStatus::invalid_arguments;

    const wroute::WeightRoutePlan plan = wroute::resolve_weight_route_plan(
        args, wroute::WeightScaleInfoMode::Residual);
    if (!plan.valid) {
        if (wreader::validate(args, plan) == wreader::WeightReaderStatus::ScaleOverflow)
            return rmd::RmdStatus::overflow;
        return rmd::RmdStatus::unsupported_route;
    }
    if (wroute::weight_route_status(plan, wroute::WeightExecutionPath::CpuDirect) !=
        wroute::WeightRouteStatus::Success) {
        return rmd::RmdStatus::unsupported_route;
    }

    const bool floating_block =
        plan.scale_domain == wroute::WeightScaleDomain::FloatingBlock;
    const bool integer_block =
        plan.scale_domain == wroute::WeightScaleDomain::IntegerBlockTimesColumn;
    if ((floating_block && plan.route != wroute::WeightRouteKind::H0) ||
        (!floating_block && (!integer_block ||
         !wroute::route_supports_integer_block_scale(plan)))) {
        return rmd::RmdStatus::unsupported_route;
    }
    if (!wroute::route_covers_k(plan, payload.logical_k) ||
        !dense_route_is_addressable(plan, payload.logical_k, payload.logical_j)) {
        return rmd::RmdStatus::unsupported_route;
    }
    const bool native_q8_route = plan.native_weight_blocks &&
        plan.weight_bits == 8 &&
        (plan.route == wroute::WeightRouteKind::H1 ||
         plan.route == wroute::WeightRouteKind::HP1);

    size_t output_count = 0;
    if (!checked_size_product(payload.row_count, payload.logical_j, output_count))
        return rmd::RmdStatus::overflow;

    std::vector<rmd::OutputValue> staged_integer;
    std::vector<double> staged_floating;
    if ((integer_block && output_count > staged_integer.max_size()) ||
        (floating_block && output_count > staged_floating.max_size())) {
        return rmd::RmdStatus::allocation_failure;
    }
    try {
        if (floating_block) {
            staged_floating.assign(output_count, 0.0);
        } else {
            staged_integer.assign(output_count, rmd::OutputValue{0});
        }
    } catch (const std::bad_alloc &) {
        return rmd::RmdStatus::allocation_failure;
    }

    const size_t j_tile_count =
        (payload.logical_j + kJTile - 1) / kJTile;
    std::vector<rmd::RmdStatus> tile_status;
    std::vector<size_t> tile_native_q8_values;
#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING) || \
    (CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__))
    std::vector<DirectCpuTileRecord> tile_cpu_records;
    const uint64_t direct_run_id = metrics != nullptr ? metrics->run_id : 0;
#endif
    try {
        tile_status.assign(j_tile_count, rmd::RmdStatus::success);
        tile_native_q8_values.assign(j_tile_count, 0);
#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING) || \
    (CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__))
        tile_cpu_records.assign(j_tile_count, DirectCpuTileRecord{});
#endif
    } catch (const std::bad_alloc &) {
        return rmd::RmdStatus::allocation_failure;
    }

    // Events are canonical row/K order. For each J tile, consume contiguous
    // row/block/K spans, then apply that block's scale exactly once.
    auto execute_j_tile = [&](size_t tile_index) {
#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING) || \
    (CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__))
        const CpuSample tile_start =
#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING)
            read_cpu_sample(hooks, testing::DirectCpuSamplePoint::tile_start, tile_index);
#else
            read_cpu_sample();
#endif
#endif
        const rmd::RmdStatus status = [&] {
            const size_t j_begin = tile_index * kJTile;
            const size_t tile_j = std::min(kJTile, payload.logical_j - j_begin);
            size_t & native_q8_values = tile_native_q8_values[tile_index];
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
                for (size_t local_j = 0; local_j < tile_j; ++local_j) {
                    const size_t j = j_begin + local_j;
                    for (size_t index = event_index; index < span_end; ++index) {
                        const ResidualEvent & event = payload.events[index];
                        const wreader::WeightCodeResult code = wreader::read_code_validated(
                            args, plan, j, event.original_k);
                        if (!code.ok()) return reader_failure(code.status);
                        // A block has at most 32 signed INT16 codes and INT32
                        // residuals, so its complete dot product fits in INT64.
                        block_sum[local_j] +=
                            static_cast<int64_t>(event.residual) * code.value;
                        if (native_q8_route) ++native_q8_values;
                    }
                }

                for (size_t local_j = 0; local_j < tile_j; ++local_j) {
                    const size_t j = j_begin + local_j;
                    wreader::WeightScaleResult scale{};
                    if (uses_reader_scale(plan)) {
                        scale = wreader::read_scale_validated(args, plan, j, block_id);
                        if (!scale.ok()) return reader_failure(scale.status);
                    } else {
                        scale.status = wreader::WeightReaderStatus::Success;
                        scale.domain = plan.scale_domain;
                        scale.integer_block_scale =
                            wroute::route_block_scale(plan, args, j, block_id);
                    }
                    if (scale.domain != plan.scale_domain)
                        return rmd::RmdStatus::execution_failed;

                    const size_t output_index = row * payload.logical_j + j;
                    if (floating_block) {
                        const double scaled = static_cast<double>(block_sum[local_j]) *
                            static_cast<double>(scale.floating_block_scale);
                        const double sum = staged_floating[output_index] + scaled;
                        if (!finite_double(scaled) || !finite_double(sum))
                            return rmd::RmdStatus::overflow;
                        staged_floating[output_index] = sum;
                    } else {
                        if (scale.integer_block_scale >
                            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                            return rmd::RmdStatus::overflow;
                        }
                        int64_t scaled = 0;
                        if (!checked_multiply(
                                block_sum[local_j],
                                static_cast<int64_t>(scale.integer_block_scale), scaled) ||
                            !checked_add(staged_integer[output_index], scaled,
                                         staged_integer[output_index])) {
                            return rmd::RmdStatus::overflow;
                        }
                    }
                }
                event_index = span_end;
            }
            return rmd::RmdStatus::success;
        }();
#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING) || \
    (CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__))
#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING)
            const CpuSample tile_end = read_cpu_sample(hooks,
                testing::DirectCpuSamplePoint::tile_end, tile_index);
#else
            const CpuSample tile_end = read_cpu_sample();
#endif
            const CpuInterval interval = cpu_interval(tile_start, tile_end);
            DirectCpuTileRecord & record = tile_cpu_records[tile_index];
            record.run_id = direct_run_id;
            record.stripe_id = payload.stripe_id;
            record.worker_id = direct_worker_id();
            record.tile_index = tile_index;
            record.j_begin = tile_index * kJTile;
            record.j_end = std::min(payload.logical_j, record.j_begin + kJTile);
            record.start_cycle = tile_start.value;
            record.end_cycle = tile_end.value;
            if (interval.valid) record.delta_cycles = interval.value;
            record.valid = interval.valid;
            record.reason = interval.reason;
            record.sample_reason = interval.sample_reason;
            record.source = tile_start.source;
            record.owner_event_token = tile_start.owner;
            record.generation = tile_start.generation;
#if !defined(GGML_GEMMINI_DIRECT_METRICS_TESTING)
            const gemmini_native_cycle_sample_internal start_sample{
                tile_start.value, static_cast<uint8_t>(tile_start.valid),
                static_cast<uint8_t>(tile_start.reason),
                GEMMINI_NATIVE_CYCLE_SOURCE_LINUX_PERF_CPU_CYCLES,
                tile_start.owner, tile_start.generation};
            const gemmini_native_cycle_sample_internal end_sample{
                tile_end.value, static_cast<uint8_t>(tile_end.valid),
                static_cast<uint8_t>(tile_end.reason),
                GEMMINI_NATIVE_CYCLE_SOURCE_LINUX_PERF_CPU_CYCLES,
                tile_end.owner, tile_end.generation};
            uint32_t identity_mask =
                static_cast<uint32_t>(GEMMINI_CYCLE_HAS_STRIPE_ID) |
                static_cast<uint32_t>(GEMMINI_CYCLE_HAS_NODE_ID) |
                static_cast<uint32_t>(GEMMINI_CYCLE_HAS_WORKER_ID);
            if (direct_run_id != 0) {
                identity_mask |= static_cast<uint32_t>(GEMMINI_CYCLE_HAS_RUN_ID);
            }
            const gemmini_cycle_record_v2 detail{{nullptr, "rmd_direct_j_tile_interval",
                tile_start.value, tile_end.value, nullptr, 0, nullptr},
                identity_mask, direct_run_id, payload.stripe_id, 0, tile_index,
                record.worker_id};
            gemmini_log_cycle_record_v2_checked_internal(
                &detail, &start_sample, &end_sample, 1);
#endif
#endif
        return status;
    };

#if defined(GGML_GEMMINI_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(j_tile_count > 1)
#endif
    for (std::ptrdiff_t tile_index = 0;
         tile_index < static_cast<std::ptrdiff_t>(j_tile_count);
         ++tile_index) {
        tile_status[static_cast<size_t>(tile_index)] =
            execute_j_tile(static_cast<size_t>(tile_index));
    }
    size_t native_q8_values = 0;
    for (size_t tile_index = 0; tile_index < j_tile_count; ++tile_index) {
        if (tile_status[tile_index] != rmd::RmdStatus::success) {
            return tile_status[tile_index];
        }
        native_q8_values += tile_native_q8_values[tile_index];
    }

    rmd::DirectOutput staged_output = floating_block ?
        rmd::DirectOutput(rmd::PreScaledFloat64Correction{std::move(staged_floating)}) :
        rmd::DirectOutput(rmd::BlockScaledInt64Correction{std::move(staged_integer)});
    correction.swap(staged_output);

    if (metrics != nullptr) {
        metrics->event_count = payload.events.size();
        metrics->call_count = 1;
        metrics->native_q8_values = native_q8_values;
        metrics->j_tile_count = j_tile_count;
#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING) || \
    (CYCLE_DETAIL && defined(__linux__) && defined(__aarch64__))
        metrics->cpu_tiles = std::move(tile_cpu_records);
#endif
    }
    return rmd::RmdStatus::success;
}

#if defined(GGML_GEMMINI_DIRECT_METRICS_TESTING)
rmd::RmdStatus execute_direct_stripe(const ggml_gemmini_args_t & args,
                                     const DirectStripePayload & payload,
                                     rmd::DirectOutput & correction,
                                     DirectExecutionMetrics * metrics) {
    return execute_direct_stripe_impl(args, payload, correction, metrics, nullptr);
}

rmd::RmdStatus execute_direct_stripe(
    const ggml_gemmini_args_t & args,
    const DirectStripePayload & payload,
    rmd::DirectOutput & correction,
    DirectExecutionMetrics * metrics,
    const testing::DirectExecutionTestHooks & hooks) {
    return execute_direct_stripe_impl(args, payload, correction, metrics, &hooks);
}
#endif

}
