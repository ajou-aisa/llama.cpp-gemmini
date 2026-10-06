#include <gemmini/trace-context.hpp>
#include "dense.hpp"
#include "detail.hpp"
#include "../quants/common/weight_reader.hpp"
#include <gemmini/log.hpp>
#include <gemmini/performance.hpp>

#include "../quants/act/quantize.hpp"
#include "../quants/act/dispatch.hpp"
#include "../residual/rmd/rmd-builder.hpp"
#include "../residual/direct/direct-builder.hpp"
#include "../residual/direct/direct-executor.hpp"

#include <gemmini.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <new>
#include <sstream>
#include <cstring>
#include <tuple>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ggml::gemmini {

using detail::MatmulCycleDrain;
using detail::record_matmul_cpu_wall;
using detail::emit_matmul_cpu_interval;
using detail::emit_rmd_stripe_metrics;

namespace test_detail {
#if defined(GGML_GEMMINI_TESTING)
struct AtomicMatmulTestCounters {
    std::atomic<uint64_t> execution_constructions{0};
    std::atomic<uint64_t> allocation_attempts{0};
    std::atomic<uint64_t> dense_dispatches{0};
    std::atomic<uint64_t> residual_dispatches{0};
    std::atomic<uint64_t> hardware_dispatches{0};
    std::atomic<uint64_t> fallback_dispatches{0};
    std::atomic<uint64_t> native_integer_block_dots{0};
    std::atomic<uint64_t> native_post_dot_scales{0};
    std::atomic<bool>     fail_output_stage_allocation{false};
};

AtomicMatmulTestCounters counters;

static void increment(std::atomic<uint64_t> & counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
}

void observe_execution_construction() {
    increment(counters.execution_constructions);
}
void observe_allocation_attempt() {
    increment(counters.allocation_attempts);
}
static void observe_dense_dispatch() {
    increment(counters.dense_dispatches);
}
void observe_residual_dispatch() {
    increment(counters.residual_dispatches);
}
void observe_backend_dispatch(bool fallback) {
    increment(fallback ? counters.fallback_dispatches : counters.hardware_dispatches);
}
static void observe_native_integer_block_dot() {
    increment(counters.native_integer_block_dots);
}
static void observe_native_post_dot_scale() {
    increment(counters.native_post_dot_scales);
}
#else
void        observe_execution_construction() {}
void        observe_allocation_attempt() {}
static void observe_dense_dispatch() {}
void        observe_residual_dispatch() {}
void        observe_backend_dispatch(bool) {}
static void observe_native_integer_block_dot() {}
static void observe_native_post_dot_scale() {}
#endif
} // namespace test_detail

#if defined(GGML_GEMMINI_TESTING)
void test_reset_matmul_counters() {
    test_detail::counters.execution_constructions.store(0, std::memory_order_relaxed);
    test_detail::counters.allocation_attempts.store(0, std::memory_order_relaxed);
    test_detail::counters.dense_dispatches.store(0, std::memory_order_relaxed);
    test_detail::counters.residual_dispatches.store(0, std::memory_order_relaxed);
    test_detail::counters.hardware_dispatches.store(0, std::memory_order_relaxed);
    test_detail::counters.fallback_dispatches.store(0, std::memory_order_relaxed);
    test_detail::counters.native_integer_block_dots.store(0, std::memory_order_relaxed);
    test_detail::counters.native_post_dot_scales.store(0, std::memory_order_relaxed);
    test_detail::counters.fail_output_stage_allocation.store(false, std::memory_order_relaxed);
}

void test_inject_output_stage_allocation_failure() {
    test_detail::counters.fail_output_stage_allocation.store(true, std::memory_order_relaxed);
}

MatmulTestCounters test_matmul_counters() {
    return {
        test_detail::counters.execution_constructions.load(std::memory_order_relaxed),
        test_detail::counters.allocation_attempts.load(std::memory_order_relaxed),
        test_detail::counters.dense_dispatches.load(std::memory_order_relaxed),
        test_detail::counters.residual_dispatches.load(std::memory_order_relaxed),
        test_detail::counters.hardware_dispatches.load(std::memory_order_relaxed),
        test_detail::counters.fallback_dispatches.load(std::memory_order_relaxed),
        test_detail::counters.native_integer_block_dots.load(std::memory_order_relaxed),
        test_detail::counters.native_post_dot_scales.load(std::memory_order_relaxed),
    };
}
#endif

namespace {

bool checked_offset(size_t row, size_t stride, size_t & offset) {
    if (stride != 0 && row > std::numeric_limits<size_t>::max() / stride) {
        return false;
    }
    offset = row * stride;
    return true;
}

bool row_invariant_activation(const ggml_gemmini_args_t & args) {
    const auto & storage = args.act_quant.storage();
    return std::holds_alternative<quants::act::NoneMeta>(storage) ||
           std::holds_alternative<quants::act::tensor::Meta>(storage);
}

bool supports_row_slice_activation(const ggml_gemmini_args_t & args) {
    const auto & storage = args.act_quant.storage();
    return row_invariant_activation(args) ||
           std::holds_alternative<quants::act::exsia::Meta>(storage) ||
           std::holds_alternative<quants::act::token::Meta>(storage) ||
           std::holds_alternative<quants::act::block::Meta>(storage) ||
           std::holds_alternative<quants::act::stripe::Meta>(storage);
}

bool uses_baseline_channel_route(const ggml_gemmini_args_t & args) {
    return args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_channel ||
           args.weight_format ==
               ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar;
}

bool valid_matmul_shape(const ggml_gemmini_args_t & args) {
    return args.I != 0 && args.J != 0 && args.K != 0 && args.f_out != nullptr &&
           (args.A.valid() || args.A_fp32 != nullptr) &&
           ((args.A_fp32 == nullptr) == (args.B_fp32 == nullptr));
}

bool valid_activation_metadata(const ggml_gemmini_args_t & args) {
    if (std::holds_alternative<quants::act::NoneMeta>(args.act_quant.storage())) {
        return !args.A.valid() && args.A_fp32 != nullptr && args.B_fp32 != nullptr;
    }
    if (args.I > std::numeric_limits<size_t>::max() - args.activation_row_offset) {
        return false;
    }
    const quants::act::ActivationMetadataView metadata(
        args, args.activation_row_offset, args.activation_row_offset + args.I);
    if (std::holds_alternative<quants::act::block::Meta>(args.act_quant.storage())) {
        return metadata.valid();
    }
    float scale = 0.0f;
    for (size_t row = 0; metadata.valid() && row < args.I; ++row) {
        if (!metadata.scale(row, scale)) {
            return false;
        }
    }
    return metadata.valid();
}

bool finite_float_bits(const float * value) {
    static_assert(sizeof(float) == sizeof(uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);
    uint32_t bits = 0;
    std::memcpy(&bits, static_cast<const void *>(value), sizeof(bits));
    constexpr uint32_t exponent_mask = UINT32_C(0x7f800000);
    return (bits & exponent_mask) != exponent_mask;
}

bool finite_output(const ggml_gemmini_args_t & args) {
    const size_t row_stride = args.stride_f_out != 0 ? args.stride_f_out : args.J;
    const size_t col_stride = args.col_stride_f_out != 0 ? args.col_stride_f_out : 1;
    for (size_t row = 0; row < args.I; ++row) {
        for (size_t col = 0; col < args.J; ++col) {
            if (!finite_float_bits(&args.f_out[row * row_stride + col * col_stride])) {
                return false;
            }
        }
    }
    return true;
}

baseline_activation_quant_t baseline_activation_for(const ggml_gemmini_args_t & args) {
    const auto & storage = args.act_quant.storage();
    if (std::holds_alternative<quants::act::tensor::Meta>(storage)) {
        return baseline_activation_quant_t::TENSOR;
    }
    if (std::holds_alternative<quants::act::token::Meta>(storage)) {
        return baseline_activation_quant_t::TOKEN;
    }
    if (std::holds_alternative<quants::act::block::Meta>(storage) ||
        std::holds_alternative<quants::act::stripe::Meta>(storage)) {
        return baseline_activation_quant_t::BLOCK;
    }
    return baseline_activation_quant_t::EXSIA;
}

MatMulStatus execute_native_matched_int_dense(ggml_gemmini_args_t & args) {
    using namespace quants::wreader;
    using namespace quants::wroute;

    const WeightRoutePlan plan = resolve_weight_route_plan(args, WeightScaleInfoMode::CommonOutput);
    if (!plan.valid ||
        weight_route_status(plan, WeightExecutionPath::CpuDirect) != WeightRouteStatus::Success ||
        validate(args, plan) != WeightReaderStatus::Success || args.f_out == nullptr ||
        !args.A.valid() || args.A.bits != plan.weight_bits || !route_covers_k(plan, args.K)) {
        return MatMulStatus::invalid_contract;
    }

    const size_t block_size =
        plan.scales.block_size != 0 ? plan.scales.block_size : args.block_size_k;
    if (block_size == 0 || args.I == 0 || args.J == 0 || args.K == 0 ||
        args.I > std::numeric_limits<size_t>::max() / args.J ||
        args.activation_row_offset > std::numeric_limits<size_t>::max() - args.I) {
        return MatMulStatus::invalid_contract;
    }
    if (args.tile_I == 0 || args.tile_J == 0 || args.tile_K == 0) {
        gemmini_set_tile_ws(&args);
    }
    if (args.tile_I == 0 || args.tile_J == 0 ||
        args.tile_I > std::numeric_limits<size_t>::max() / DIM ||
        args.tile_J > std::numeric_limits<size_t>::max() / DIM) {
        return MatMulStatus::invalid_contract;
    }
    const size_t rows_per_tile    = args.tile_I * DIM;
    const size_t columns_per_tile = args.tile_J * DIM;
    const size_t row_tile_count =
        args.I / rows_per_tile + static_cast<size_t>(args.I % rows_per_tile != 0);
    const size_t column_tile_count =
        args.J / columns_per_tile + static_cast<size_t>(args.J % columns_per_tile != 0);
    if (row_tile_count == 0 || column_tile_count == 0 ||
        row_tile_count > std::numeric_limits<size_t>::max() / column_tile_count) {
        return MatMulStatus::invalid_contract;
    }
    const size_t tile_pair_count = row_tile_count * column_tile_count;
    if (tile_pair_count > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        return MatMulStatus::invalid_arguments;
    }

    const size_t                              global_row_end = args.activation_row_offset + args.I;
    const quants::act::ActivationMetadataView activation_meta(
        args, args.activation_row_offset, global_row_end);
    if (!activation_meta.valid()) {
        return MatMulStatus::invalid_contract;
    }
    const bool block_activation =
        std::holds_alternative<quants::act::block::Meta>(args.act_quant.storage());

    std::vector<float> activation_scales;
    try {
        if (!block_activation)
            activation_scales.resize(args.I);
    } catch (const std::bad_alloc &) {
        return MatMulStatus::invalid_arguments;
    } catch (const std::length_error &) {
        return MatMulStatus::invalid_arguments;
    }
    for (size_t row = 0; row < activation_scales.size(); ++row) {
        if (!activation_meta.scale(row, activation_scales[row])) {
            return MatMulStatus::invalid_contract;
        }
    }

    const size_t bias_stride = args.sD != 0 ? args.sD : args.J;
    if (args.D != nullptr) {
        if (!args.repeating_bias && args.I > 1 &&
            bias_stride > (std::numeric_limits<size_t>::max() - (args.J - 1)) / (args.I - 1)) {
            return MatMulStatus::invalid_arguments;
        }
    }

    const size_t output_row_stride = args.stride_f_out != 0 ? args.stride_f_out : args.J;
    const size_t output_col_stride = args.col_stride_f_out != 0 ? args.col_stride_f_out : 1;
    if (args.J - 1 > std::numeric_limits<size_t>::max() / output_col_stride) {
        return MatMulStatus::invalid_arguments;
    }
    const size_t max_column_offset = (args.J - 1) * output_col_stride;
    if (args.I - 1 > std::numeric_limits<size_t>::max() / output_row_stride) {
        return MatMulStatus::invalid_arguments;
    }
    const size_t max_row_offset = (args.I - 1) * output_row_stride;
    if (max_row_offset > std::numeric_limits<size_t>::max() - max_column_offset) {
        return MatMulStatus::invalid_arguments;
    }
#if defined(GGML_GEMMINI_HAS_OPENMP)
    const bool output_tiles_do_not_overlap = args.I <= 1 || args.J <= 1 ||
                                             output_row_stride > max_column_offset ||
                                             output_col_stride > max_row_offset;
#endif
    std::atomic<bool> execution_valid{true};

    struct TileScratch {
        std::vector<int32_t> activation_codes;
        std::vector<int32_t> weight_codes;
        std::vector<double>  weight_scales;
        std::vector<double>  accumulations;
    };
    const size_t max_rows_in_tile    = std::min(rows_per_tile, args.I);
    const size_t max_columns_in_tile = std::min(columns_per_tile, args.J);
    if (max_rows_in_tile > std::numeric_limits<size_t>::max() / block_size ||
        max_columns_in_tile > std::numeric_limits<size_t>::max() / block_size ||
        max_rows_in_tile > std::numeric_limits<size_t>::max() / max_columns_in_tile) {
        return MatMulStatus::invalid_arguments;
    }
    size_t scratch_count = 1;
#if defined(GGML_GEMMINI_HAS_OPENMP)
    const bool parallel_tile_execution = tile_pair_count > 1 && output_tiles_do_not_overlap;
    if (parallel_tile_execution) {
        scratch_count =
            std::min(tile_pair_count, static_cast<size_t>(std::max(1, omp_get_max_threads())));
    }
#endif
    std::vector<TileScratch> tile_scratch;
    try {
        tile_scratch.resize(scratch_count);
        for (TileScratch & scratch : tile_scratch) {
            scratch.activation_codes.resize(max_rows_in_tile * block_size);
            scratch.weight_codes.resize(max_columns_in_tile * block_size);
            scratch.weight_scales.resize(max_columns_in_tile);
            scratch.accumulations.resize(max_rows_in_tile * max_columns_in_tile);
        }
    } catch (const std::bad_alloc &) {
        return MatMulStatus::invalid_arguments;
    } catch (const std::length_error &) {
        return MatMulStatus::invalid_arguments;
    }

    auto execute_tile_pair = [&](size_t tile_pair) {
        if (!execution_valid.load(std::memory_order_relaxed)) {
            return;
        }
        const size_t row_tile      = tile_pair / column_tile_count;
        const size_t column_tile   = tile_pair % column_tile_count;
        const size_t row_begin     = row_tile * rows_per_tile;
        const size_t column_begin  = column_tile * columns_per_tile;
        const size_t row_count     = std::min(rows_per_tile, args.I - row_begin);
        const size_t column_count  = std::min(columns_per_tile, args.J - column_begin);
        size_t       scratch_index = 0;
#if defined(GGML_GEMMINI_HAS_OPENMP)
        if (parallel_tile_execution) {
            scratch_index = static_cast<size_t>(omp_get_thread_num());
        }
#endif
        TileScratch & scratch = tile_scratch[scratch_index];
        std::fill_n(scratch.accumulations.begin(), row_count * column_count, 0.0);

        for (size_t block_begin = 0; block_begin < args.K;) {
            const size_t block_index = block_begin / block_size;
            const size_t weight_block_end =
                block_begin + std::min(block_size - block_begin % block_size, args.K - block_begin);
            const size_t activation_block_end =
                block_activation
                    ? block_begin + std::min(quants::act::block::kGroupSize -
                                                 block_begin % quants::act::block::kGroupSize,
                                             args.K - block_begin)
                    : weight_block_end;
            const size_t block_count =
                std::min(weight_block_end, activation_block_end) - block_begin;
            bool tile_valid = true;

            for (size_t local_column = 0; local_column < column_count; ++local_column) {
                const size_t            column = column_begin + local_column;
                const WeightScaleResult scale =
                    read_scale_validated(args, plan, column, block_index);
                if (!scale.ok()) {
                    tile_valid = false;
                    break;
                }
                if (scale.domain == WeightScaleDomain::FloatingBlock) {
                    scratch.weight_scales[local_column] = scale.floating_block_scale;
                } else if (scale.domain == WeightScaleDomain::IntegerBlockTimesColumn) {
                    scratch.weight_scales[local_column] =
                        static_cast<double>(scale.integer_block_scale) *
                        static_cast<double>(scale.column_scale);
                } else {
                    tile_valid = false;
                    break;
                }

                int32_t * decoded = scratch.weight_codes.data() + local_column * block_size;
                for (size_t local_k = 0; local_k < block_count; ++local_k) {
                    const WeightCodeResult code =
                        read_code_validated(args, plan, column, block_begin + local_k);
                    if (!code.ok()) {
                        tile_valid = false;
                        break;
                    }
                    decoded[local_k] = code.value;
                }
                if (!tile_valid) {
                    break;
                }
            }
            if (!tile_valid) {
                execution_valid.store(false, std::memory_order_relaxed);
                return;
            }

            for (size_t local_row = 0; local_row < row_count; ++local_row) {
                int32_t * decoded = scratch.activation_codes.data() + local_row * block_size;
                for (size_t local_k = 0; local_k < block_count; ++local_k) {
                    decoded[local_k] = args.A.get(row_begin + local_row, block_begin + local_k);
                }
            }

            for (size_t local_row = 0; local_row < row_count; ++local_row) {
                const int32_t * activation =
                    scratch.activation_codes.data() + local_row * block_size;
                float activation_scale = 1.0f;
                if (block_activation &&
                    !activation_meta.scale(row_begin + local_row, block_begin, activation_scale)) {
                    execution_valid.store(false, std::memory_order_relaxed);
                    return;
                }
                for (size_t local_column = 0; local_column < column_count; ++local_column) {
                    const int32_t * weight =
                        scratch.weight_codes.data() + local_column * block_size;
                    int64_t block_dot = 0;
                    for (size_t local_k = 0; local_k < block_count; ++local_k) {
                        block_dot += static_cast<int64_t>(activation[local_k]) *
                                     static_cast<int64_t>(weight[local_k]);
                    }
                    test_detail::observe_native_integer_block_dot();
                    scratch.accumulations[local_row * column_count + local_column] +=
                        static_cast<double>(block_dot) * scratch.weight_scales[local_column] *
                        static_cast<double>(activation_scale);
                    test_detail::observe_native_post_dot_scale();
                }
            }
            block_begin += block_count;
        }

        for (size_t local_row = 0; local_row < row_count; ++local_row) {
            const size_t row = row_begin + local_row;
            for (size_t local_column = 0; local_column < column_count; ++local_column) {
                const size_t column = column_begin + local_column;
                double       value = scratch.accumulations[local_row * column_count + local_column];
                if (!block_activation) {
                    value *= static_cast<double>(activation_scales[row]);
                }
                if (args.D != nullptr) {
                    const size_t bias_row   = args.repeating_bias ? 0 : row;
                    const size_t bias_index = bias_row * bias_stride + column;
                    value +=
                        args.low_D
                            ? static_cast<double>(static_cast<const elem_t *>(args.D)[bias_index]) *
                                  static_cast<double>(args.scale_D)
                            : static_cast<double>(static_cast<const acc_t *>(args.D)[bias_index]) *
                                  static_cast<double>(args.scale_D);
                }
                args.f_out[row * output_row_stride + column * output_col_stride] =
                    static_cast<float>(value);
            }
        }
    };

    const auto dense_task_origin = gemmini_trace_capture();
#if defined(GGML_GEMMINI_HAS_OPENMP)
#pragma omp parallel if (parallel_tile_execution)
#endif
    {
        trace::ScopedContext dense_task(dense_task_origin, true);
        trace::CpuStage      task_lifetime(
            args.matmul_layer.c_str(), "task.host_work", trace::CpuStage::Scope::envelope);
#if LOG_CYCLE
        const auto worker_cpu_start = gemmini_cpu_timing_read();
#endif
#if defined(GGML_GEMMINI_HAS_OPENMP)
#pragma omp for schedule(static) nowait
#endif
        for (std::ptrdiff_t tile_pair = 0; tile_pair < static_cast<std::ptrdiff_t>(tile_pair_count);
             ++tile_pair) {
            execute_tile_pair(static_cast<size_t>(tile_pair));
        }
#if LOG_CYCLE
        const auto worker_cpu_end = gemmini_cpu_timing_read();
#if defined(GGML_GEMMINI_HAS_OPENMP)
#pragma omp barrier
        const auto barrier_cpu_end = gemmini_cpu_timing_read();
#endif
        gemmini_cpu_totals worker_cpu{};
        gemmini_cpu_timing_add(&worker_cpu, &worker_cpu_start, &worker_cpu_end);
#if defined(GGML_GEMMINI_HAS_OPENMP)
        gemmini_cpu_timing_add(&worker_cpu, &worker_cpu_end, &barrier_cpu_end);
#endif
        gemmini_cycle_record_v2 worker_record{};
        worker_record.interval.layer = args.matmul_layer.c_str();
        worker_record.interval.op    = "cpu.dense_worker";
        worker_record.identity_mask  = GEMMINI_CYCLE_HAS_WORKER_ID;
#if defined(GGML_GEMMINI_HAS_OPENMP)
        worker_record.worker_id = static_cast<uint64_t>(omp_get_thread_num());
#endif
        if (const auto run_id = matmul_cpu_run_id(args)) {
            worker_record.identity_mask |= GEMMINI_CYCLE_HAS_RUN_ID;
            worker_record.run_id = *run_id;
        }
        gemmini_cpu_timing_record(&worker_record, &worker_cpu_start, &worker_cpu_end);
#if defined(GGML_GEMMINI_HAS_OPENMP)
        worker_record.interval.op = "cpu.dense_worker_barrier";
        gemmini_cpu_timing_record(&worker_record, &worker_cpu_end, &barrier_cpu_end);
#endif
#endif
    }
    return execution_valid.load(std::memory_order_relaxed) ? MatMulStatus::success
                                                           : MatMulStatus::invalid_contract;
}

MatMulStatus execute_dense(ggml_gemmini_args_t & args, std::optional<uint64_t> stripe_id = {}) {
#if LOG_CYCLE
    gemmini_cycle_record_v2 ws_identity{};
    ws_identity.interval.layer = args.matmul_layer.c_str();
    if (const auto run_id = matmul_cpu_run_id(args)) {
        ws_identity.identity_mask |= GEMMINI_CYCLE_HAS_RUN_ID;
        ws_identity.run_id = *run_id;
    }
    if (stripe_id) {
        ws_identity.identity_mask |= GEMMINI_CYCLE_HAS_STRIPE_ID;
        ws_identity.stripe_id = *stripe_id;
    }
    const log::ScopedWsCycleIdentity ws_scope(ws_identity, "gemmini_hw_dense");
#else
    (void)stripe_id;
#endif
    const auto cpu_start = read_matmul_cpu_sample();
    const auto status    = [&]() -> MatMulStatus {
        if (args.A_fp32 != nullptr || args.B_fp32 != nullptr) {
            if (args.A_fp32 == nullptr || args.B_fp32 == nullptr || args.f_out == nullptr) {
                return MatMulStatus::invalid_contract;
            }
            test_detail::observe_dense_dispatch();
            test_detail::observe_backend_dispatch(true);
            matmul_cpu_fp(false,
                          true,
                          args.I,
                          args.J,
                          args.K,
                          args.A_fp32,
                          args.B_fp32,
                          nullptr,
                          args.f_out,
                          args.sA,
                          args.sB,
                          args.col_stride_f_out,
                          args.stride_f_out);
            return MatMulStatus::success;
        }
        test_detail::observe_dense_dispatch();
        // GGUF Q4_0/Q8_0 blocks as stored (H0): block x block, then dequantize, on
        // the CPU im2p path. Dense-B q8_h0 keeps the baseline route below.
        if (args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q4_h0 ||
            (args.weight_format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h0 &&
             args.B_blocks != nullptr)) {
            if (args.tiled_matmul_type != CPU) {
                return MatMulStatus::unsupported;
            }
            test_detail::observe_backend_dispatch(true);
            return tiled_matmul_auto_im2p(&args) == DenseMatmulStatus::success
                       ? MatMulStatus::success
                       : MatMulStatus::invalid_contract;
        }
        if (quants::wroute::is_native_matched_width_format(args)) {
            if (args.tiled_matmul_type != CPU) {
                return MatMulStatus::unsupported;
            }
            test_detail::observe_backend_dispatch(true);
            return execute_native_matched_int_dense(args);
        }
        test_detail::observe_backend_dispatch(args.tiled_matmul_type == CPU);
        if (uses_baseline_channel_route(args)) {
            tiled_matmul_auto_baseline(
                &args, baseline_activation_for(args), baseline_weight_quant_t::CHANNEL);
        } else if (args.weight_i8_scale_active) {
            tiled_matmul_auto_baseline(
                &args, baseline_activation_for(args), baseline_weight_quant_t::TENSOR);
        } else {
            using Format = ggml_gemmini_args_t::im2p_weight_format_t;
            switch (args.weight_format) {
            case Format::q8_h2:
            case Format::q8_hp1:
            case Format::q8_hp2:
                break;
            case Format::q8_h0:
            case Format::q8_channel:
            case Format::q8_channel_dense_sidecar:
            default:
                return MatMulStatus::unsupported;
            }
            if (args.tiled_matmul_type != CPU && args.tiled_matmul_type != WS) {
                return MatMulStatus::unsupported;
            }
            tiled_matmul_auto_im2p(&args);
        }
        return MatMulStatus::success;
    }();
    const auto cpu_end = read_matmul_cpu_sample();
    if (args.tiled_matmul_type == CPU || (args.A_fp32 != nullptr && args.B_fp32 != nullptr)) {
        record_matmul_cpu_wall(cpu_start, cpu_end);
    } else {
#if LOG_CYCLE
        // The backend call also contains device execution and waits.
        performance::incomplete_cpu_wall("accelerator_cpu_stage_coverage_incomplete");
#endif
    }
    return status;
}

MatMulStatus execute_stripe(ggml_gemmini_args_t args,
                            MatMulStripe        stripe,
                            size_t              stripe_id,
                            bool                metadata_is_local = false) {
    const size_t input_stride  = args.sA ? args.sA : args.K;
    const size_t output_stride = args.stride_f_out ? args.stride_f_out : args.J;
    size_t       input_offset  = 0;
    size_t       output_offset = 0;
    if (!checked_offset(stripe.row_begin, input_stride, input_offset) ||
        !checked_offset(stripe.row_begin, output_stride, output_offset)) {
        return MatMulStatus::invalid_arguments;
    }

    if (args.tile_I == 0 || args.tile_J == 0 || args.tile_K == 0) {
        gemmini_set_tile_ws(&args);
    }
    const size_t metadata_tile_I = args.tile_I;
    if (!metadata_is_local &&
        stripe.row_begin > std::numeric_limits<size_t>::max() - args.activation_row_offset) {
        return MatMulStatus::invalid_arguments;
    }
    auto original_A = args.A;
    if (metadata_is_local) {
        args.activation_row_offset = 0;
    } else {
        args.activation_row_offset += stripe.row_begin;
    }

    args.I = stripe.row_end - stripe.row_begin;
    gemmini_set_tile_ws(&args);
    args.tile_I = metadata_tile_I;
    if (args.A.valid()) {
        args.A = original_A.slice_rows(stripe.row_begin, args.I);
    }
    if (args.A_fp32 != nullptr) {
        args.A_fp32 += input_offset;
    }
    args.f_out += output_offset;

    return execute_dense(args, stripe_id);
}

} // namespace

namespace detail {

namespace {

struct RouteDescriptor {
    bool legacy_full;
    bool facade_full;
    bool facade_sequential;
    bool facade_pipeline;
    bool deprecated;
};

struct WeightDescriptor {
    bool legacy_full;
    bool facade_full;
    bool facade_sliced;
    bool deprecated;
};

constexpr size_t activation_route_count = static_cast<size_t>(ActivationRoute::stripe) + 1;
constexpr size_t weight_route_count     = static_cast<size_t>(WeightRoute::q8_h0) + 1;

constexpr std::array<WeightDescriptor, weight_route_count> weight_descriptors = {{
    {false, false, false, false},
    {true, true, true, false},
    {true, true, true, false},
    {true, false, false, false},
    {true, false, false, false},
    {true, true, true, false},
    {true, true, true, false},
    {true, true, false, true},
    {true, true, false, true},
    {true, true, true, false},
    {true, true, true, false},
    {true, true, true, false},
}};

constexpr auto make_route_descriptors() {
    std::array<std::array<RouteDescriptor, weight_route_count>, activation_route_count> matrix{};
    for (size_t activation = 0; activation < activation_route_count; ++activation) {
        for (size_t weight = 0; weight < weight_route_count; ++weight) {
            const auto & base = weight_descriptors[weight];
            const bool   known_activation =
                activation != static_cast<size_t>(ActivationRoute::unknown);
            const bool channel = weight == static_cast<size_t>(WeightRoute::q8_channel_direct) ||
                                 weight == static_cast<size_t>(WeightRoute::q8_channel_sidecar);
            const bool exsia_or_fp32_channel =
                channel && (activation == static_cast<size_t>(ActivationRoute::exsia) ||
                            activation == static_cast<size_t>(ActivationRoute::fp32));
            const bool full = known_activation && base.facade_full && !exsia_or_fp32_channel;
            const bool activation_is_sliceable =
                activation == static_cast<size_t>(ActivationRoute::fp32) ||
                activation == static_cast<size_t>(ActivationRoute::exsia) ||
                activation == static_cast<size_t>(ActivationRoute::tensor);
            const bool sequential =
                full && base.facade_sliced && activation_is_sliceable &&
                (!channel || activation == static_cast<size_t>(ActivationRoute::tensor));
            matrix[activation][weight] = {
                known_activation && base.legacy_full,
                full,
                sequential,
                sequential && activation == static_cast<size_t>(ActivationRoute::exsia),
                base.deprecated,
            };
        }
    }
    return matrix;
}

constexpr auto route_descriptors = make_route_descriptors();

const RouteDescriptor & route_descriptor(const RouteKey & key) {
    return route_descriptors[static_cast<size_t>(key.activation)][static_cast<size_t>(key.weight)];
}

} // namespace

RouteKey normalize_route(const ggml_gemmini_args_t & args) {
    RouteKey key{};
    switch (args.tiled_matmul_type) {
    case CPU:
        key.backend = BackendRoute::cpu;
        break;
    case WS:
        key.backend = BackendRoute::gemmini_ws;
        break;
    case OS:
        key.backend = BackendRoute::gemmini_os;
        break;
    default:
        key.backend = BackendRoute::ws_sim;
        break;
    }
    switch (args.act_quant.kind()) {
    case quants::act::MetaKind::exsia:
        key.activation = ActivationRoute::exsia;
        break;
    case quants::act::MetaKind::tensor:
        key.activation = ActivationRoute::tensor;
        break;
    case quants::act::MetaKind::token:
        key.activation = ActivationRoute::token;
        break;
    case quants::act::MetaKind::block:
        key.activation = ActivationRoute::block;
        break;
    case quants::act::MetaKind::stripe:
        key.activation = ActivationRoute::stripe;
        break;
    case quants::act::MetaKind::none:
        key.activation = ActivationRoute::fp32;
        break;
    }

    if (args.A_fp32 != nullptr || args.B_fp32 != nullptr) {
        key.activation = ActivationRoute::fp32;
        key.weight     = WeightRoute::fp32;
        return key;
    }

    using Format = ggml_gemmini_args_t::im2p_weight_format_t;
    switch (args.weight_format) {
    case Format::q8_0_unpacked_to_h1:
        key.weight = args.weight_i8_scale_active ? WeightRoute::tensor_i8 : WeightRoute::affine;
        break;
    case Format::q4_h0:
    case Format::q8_h0:
    case Format::q16_h0:
        key.weight = WeightRoute::q8_h0;
        break;
    case Format::q16_h1:
        key.weight = WeightRoute::affine;
        break;
    case Format::q4_hp1:
    case Format::q8_hp1:
    case Format::q16_hp1:
        key.weight = WeightRoute::q8_hp1;
        break;
    case Format::q8_h2:
        key.weight = WeightRoute::q8_h2;
        break;
    case Format::q8_hp2:
        key.weight = WeightRoute::q8_hp2;
        break;
    case Format::q8_channel:
        key.weight = WeightRoute::q8_channel_direct;
        break;
    case Format::q8_channel_dense_sidecar:
        key.weight = WeightRoute::q8_channel_sidecar;
        break;
    }
    return key;
}

RouteCapabilities route_capabilities(const ggml_gemmini_args_t & args) {
    const RouteKey          key        = normalize_route(args);
    const RouteDescriptor & descriptor = route_descriptor(key);
    RouteCapabilities       caps{};
    caps.legacy_full             = descriptor.legacy_full;
    caps.deprecated              = descriptor.deprecated;
    caps.full                    = descriptor.facade_full;
    caps.sliced_dense            = descriptor.facade_sequential;
    caps.sliced_compensation     = descriptor.facade_sequential;
    caps.live_stripe_producer    = descriptor.facade_pipeline;
    caps.internal_parallel_dense = caps.full;
    if (key.backend == BackendRoute::gemmini_os || key.backend == BackendRoute::ws_sim) {
        caps             = {};
        caps.legacy_full = descriptor.legacy_full;
        caps.deprecated  = descriptor.deprecated;
    }
    return caps;
}

const char * activation_route_name(ActivationRoute route) {
    switch (route) {
    case ActivationRoute::fp32:
        return "fp32";
    case ActivationRoute::exsia:
        return "exsia";
    case ActivationRoute::tensor:
        return "tensor";
    case ActivationRoute::token:
        return "token";
    case ActivationRoute::block:
        return "block";
    case ActivationRoute::stripe:
        return "stripe";
    case ActivationRoute::unknown:
        return "unknown";
    }
    return "unknown";
}

const char * weight_route_name(WeightRoute route) {
    switch (route) {
    case WeightRoute::fp32:
        return "fp32";
    case WeightRoute::tensor_i8:
        return "tensor_i8";
    case WeightRoute::channel_i8:
        return "channel_i8";
    case WeightRoute::block_i8:
        return "block_i8";
    case WeightRoute::affine:
        return "affine";
    case WeightRoute::q8_hp1:
        return "q8_hp1";
    case WeightRoute::q8_h2:
        return "q8_h2";
    case WeightRoute::q8_hp2:
        return "q8_hp2";
    case WeightRoute::q8_channel_direct:
        return "q8_channel_direct";
    case WeightRoute::q8_channel_sidecar:
        return "q8_channel_sidecar";
    case WeightRoute::q8_h0:
        return "q8_h0";
    case WeightRoute::unknown:
        return "unknown";
    }
    return "unknown";
}

const char * backend_route_name(BackendRoute route) {
    switch (route) {
    case BackendRoute::cpu:
        return "cpu";
    case BackendRoute::gemmini_ws:
        return "gemmini_ws";
    case BackendRoute::gemmini_os:
        return "gemmini_os";
    case BackendRoute::ws_sim:
        return "ws_sim";
    }
    return "unknown";
}

} // namespace detail

MatMul::MatMul(ggml_gemmini_args_t args) : owned_args_(std::move(args)), args_ptr_(&owned_args_) {}

MatMul::MatMul(ggml_gemmini_args_t * args) : args_ptr_(args) {}

MatMul::MatMul(MatMul && other) noexcept
    : owned_args_(std::move(other.owned_args_)),
      args_ptr_(other.args_ptr_ == &other.owned_args_ ? &owned_args_ : other.args_ptr_),
      first_row_(other.first_row_), last_row_begin_(other.last_row_begin_),
      last_row_end_(other.last_row_end_), covered_rows_(other.covered_rows_),
      has_stripes_(other.has_stripes_), state_(other.state_),
      output_destination_(other.output_destination_), output_row_stride_(other.output_row_stride_),
      output_col_stride_(other.output_col_stride_), output_stage_(std::move(other.output_stage_)) {
    if (output_destination_ != nullptr && args_ptr_ != nullptr) {
        args().f_out = output_stage_.data();
    }
    other.output_destination_ = nullptr;
    other.output_row_stride_  = 0;
    other.output_col_stride_  = 0;
}

MatMul & MatMul::operator=(MatMul && other) noexcept {
    if (this != &other) {
        discard_output_transaction();
        owned_args_     = std::move(other.owned_args_);
        args_ptr_       = other.args_ptr_ == &other.owned_args_ ? &owned_args_ : other.args_ptr_;
        first_row_      = other.first_row_;
        last_row_begin_ = other.last_row_begin_;
        last_row_end_   = other.last_row_end_;
        covered_rows_   = other.covered_rows_;
        has_stripes_    = other.has_stripes_;
        state_          = other.state_;
        output_destination_ = other.output_destination_;
        output_row_stride_  = other.output_row_stride_;
        output_col_stride_  = other.output_col_stride_;
        output_stage_       = std::move(other.output_stage_);
        if (output_destination_ != nullptr && args_ptr_ != nullptr) {
            args().f_out = output_stage_.data();
        }
        other.output_destination_ = nullptr;
        other.output_row_stride_  = 0;
        other.output_col_stride_  = 0;
    }
    return *this;
}

MatMul::~MatMul() {
    discard_output_transaction();
}

ggml_gemmini_args_t & MatMul::args() {
    return *args_ptr_;
}
const ggml_gemmini_args_t & MatMul::args() const {
    return *args_ptr_;
}

MatMulStatus MatMul::begin_output_transaction() {
    if (output_destination_ != nullptr || args_ptr_ == nullptr || args().f_out == nullptr ||
        args().I == 0 || args().J == 0) {
        return MatMulStatus::invalid_state;
    }
    const size_t row_stride    = args().stride_f_out != 0 ? args().stride_f_out : args().J;
    const size_t col_stride    = args().col_stride_f_out != 0 ? args().col_stride_f_out : 1;
    size_t       row_offset    = 0;
    size_t       column_offset = 0;
    size_t       final_offset  = 0;
    size_t       output_span   = 0;
    if (__builtin_mul_overflow(args().I - 1, row_stride, &row_offset) ||
        __builtin_mul_overflow(args().J - 1, col_stride, &column_offset) ||
        __builtin_add_overflow(row_offset, column_offset, &final_offset) ||
        __builtin_add_overflow(final_offset, size_t{1}, &output_span) ||
        output_span > output_stage_.max_size()) {
        return MatMulStatus::invalid_arguments;
    }

    std::vector<float> staged;
    try {
        test_detail::observe_allocation_attempt();
#if defined(GGML_GEMMINI_TESTING)
        if (test_detail::counters.fail_output_stage_allocation.exchange(
                false, std::memory_order_relaxed)) {
            throw std::bad_alloc();
        }
#endif
        staged.assign(args().f_out, args().f_out + output_span);
    } catch (const std::bad_alloc &) {
        return MatMulStatus::invalid_arguments;
    } catch (const std::length_error &) {
        return MatMulStatus::invalid_arguments;
    }
    output_destination_ = args().f_out;
    output_row_stride_  = row_stride;
    output_col_stride_  = col_stride;
    output_stage_       = std::move(staged);
    args().f_out        = output_stage_.data();
    return MatMulStatus::success;
}

void MatMul::commit_output_transaction() {
    if (output_destination_ == nullptr || args_ptr_ == nullptr)
        return;
    const auto commit_start = read_matmul_cpu_sample();
    for (size_t row = 0; row < args().I; ++row) {
        for (size_t column = 0; column < args().J; ++column) {
            const size_t offset         = row * output_row_stride_ + column * output_col_stride_;
            output_destination_[offset] = output_stage_[offset];
        }
    }
    const auto commit_end = read_matmul_cpu_sample();
    emit_matmul_cpu_interval(args().matmul_layer.c_str(),
                             "matmul_output_commit_cycles",
                             commit_start,
                             commit_end,
                             true,
                             nullptr,
                             nullptr,
                             matmul_cpu_run_id(args()));
    args().f_out        = output_destination_;
    output_destination_ = nullptr;
    output_row_stride_  = 0;
    output_col_stride_  = 0;
    output_stage_.clear();
}

void MatMul::discard_output_transaction() {
    if (output_destination_ == nullptr)
        return;
    if (args_ptr_ != nullptr)
        args().f_out = output_destination_;
    output_destination_ = nullptr;
    output_row_stride_  = 0;
    output_col_stride_  = 0;
    output_stage_.clear();
}

MatMulResult MatMul::run_dense() {
    const MatmulCycleDrain drain;
    return run_dense(false);
}

MatMulResult MatMul::run_dense(bool transactional) {
    if (state_ != MatMulState::idle) {
        return {MatMulStatus::invalid_state, MatMulCapability::supported};
    }
    if (!valid_matmul_shape(args())) {
        return {MatMulStatus::invalid_arguments, MatMulCapability::unsupported};
    }
    if (!valid_activation_metadata(args())) {
        return {MatMulStatus::invalid_contract, MatMulCapability::unsupported};
    }
    if (!transactional && (!quants::activation_rmd_packets(args()).empty() ||
                           !quants::activation_direct_residuals(args()).empty())) {
        return {MatMulStatus::invalid_contract, MatMulCapability::unsupported};
    }
    if (!detail::route_capabilities(args()).full) {
        return {MatMulStatus::unsupported, MatMulCapability::unsupported};
    }
    const auto format          = args().weight_format;
    const bool metadata_weight = format == ggml_gemmini_args_t::im2p_weight_format_t::q4_h0 ||
                                 format == ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1 ||
                                 format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h0 ||
                                 format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1 ||
                                 format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2 ||
                                 format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2 ||
                                 format == ggml_gemmini_args_t::im2p_weight_format_t::q16_h0 ||
                                 format == ggml_gemmini_args_t::im2p_weight_format_t::q16_h1 ||
                                 format == ggml_gemmini_args_t::im2p_weight_format_t::q16_hp1;
    if (args().B == nullptr && args().B_fp32 == nullptr && !metadata_weight) {
        return {MatMulStatus::invalid_contract, MatMulCapability::unsupported};
    }
    switch (args().weight_format) {
    case ggml_gemmini_args_t::im2p_weight_format_t::q8_channel:
        if (!args().has_q8_channel_direct_read_contract()) {
            return {MatMulStatus::invalid_contract, MatMulCapability::unsupported};
        }
        break;
    case ggml_gemmini_args_t::im2p_weight_format_t::q8_channel_dense_sidecar:
        if (!args().has_q8_channel_dense_sidecar_contract()) {
            return {MatMulStatus::invalid_contract, MatMulCapability::unsupported};
        }
        break;
    case ggml_gemmini_args_t::im2p_weight_format_t::q4_h0:
    case ggml_gemmini_args_t::im2p_weight_format_t::q4_hp1:
    case ggml_gemmini_args_t::im2p_weight_format_t::q16_h0:
    case ggml_gemmini_args_t::im2p_weight_format_t::q16_h1:
    case ggml_gemmini_args_t::im2p_weight_format_t::q16_hp1:
        if (!args().has_native_matched_width_contract()) {
            return {MatMulStatus::invalid_contract, MatMulCapability::unsupported};
        }
        break;
    case ggml_gemmini_args_t::im2p_weight_format_t::q8_h0:
        if (args().B_blocks != nullptr && !args().has_q8_h0_contract()) {
            return {MatMulStatus::invalid_contract, MatMulCapability::unsupported};
        }
        break;
    case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp1:
        if (!args().has_q8_hp1_im2p_contract()) {
            return {MatMulStatus::invalid_contract, MatMulCapability::unsupported};
        }
        break;
    case ggml_gemmini_args_t::im2p_weight_format_t::q8_h2:
        if (!args().has_q8_h2_im2p_contract()) {
            return {MatMulStatus::invalid_contract, MatMulCapability::unsupported};
        }
        break;
    case ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2:
        if (!args().has_q8_hp2_im2p_contract()) {
            return {MatMulStatus::invalid_contract, MatMulCapability::unsupported};
        }
        break;
    default:
        break;
    }

    if (transactional) {
        const MatMulStatus transaction = begin_output_transaction();
        if (transaction != MatMulStatus::success) {
            return {transaction, MatMulCapability::unsupported};
        }
    }
    const MatMulStatus dense_status = execute_dense(args());
    if (dense_status != MatMulStatus::success) {
        if (transactional) {
            discard_output_transaction();
        }
        return {dense_status, MatMulCapability::unsupported};
    }
    return {MatMulStatus::success, MatMulCapability::supported};
}

MatMulResult MatMul::run_full() {
    const MatmulCycleDrain drain;
    const auto             run_id      = matmul_cpu_run_id(args());
    const auto             dense_start = read_matmul_cpu_sample();
    const MatMulResult     dense       = run_dense(true);
    const auto             dense_end   = read_matmul_cpu_sample();
    emit_matmul_cpu_interval(args().matmul_layer.c_str(),
                             "dense_backend_host_call",
                             dense_start,
                             dense_end,
                             dense.status == MatMulStatus::success,
                             nullptr,
                             nullptr,
                             run_id);
    if (dense.status != MatMulStatus::success) {
        discard_output_transaction();
        return dense;
    }
#if GGML_GEMMINI_ENABLE_RMD
    rmd::RmdStatus residual_status = rmd::RmdStatus::success;
    if (args().residual_route == residual::ResidualRoute::cpu_direct) {
        for (const auto & payload : quants::activation_direct_residuals(args())) {
            if (payload == nullptr)
                continue;
            size_t                     row_end  = 0;
            rmd::RmdExecutionMetrics * measured = nullptr;
#if LOG_CYCLE
            MatmulJobMetrics profile;
            profile.cpu_identity_mask   = run_id ? GEMMINI_CYCLE_HAS_RUN_ID : 0;
            profile.run_id              = run_id.value_or(0);
            profile.stripe_id           = payload->stripe_id;
            profile.row_begin           = payload->row_begin;
            profile.rmd.timing_identity = {
                {args().matmul_layer.c_str(), nullptr, 0, 0, nullptr, 0, nullptr},
                profile.cpu_identity_mask | GEMMINI_CYCLE_HAS_STRIPE_ID,
                profile.run_id,
                payload->stripe_id,
                0,
                0,
                0};
            measured = &profile.rmd;
#endif
            rmd::Correction correction = rmd::BlockScaledInt64Correction{};
            residual_status =
                __builtin_add_overflow(payload->row_begin, payload->row_count, &row_end)
                    ? rmd::RmdStatus::invalid_arguments
                    : ([&] {
                          test_detail::observe_residual_dispatch();
                          test_detail::observe_backend_dispatch(true);
                          residual::DirectExecutionMetrics direct_metrics;
                          if (const auto * meta = std::get_if<quants::act::exsia::Meta>(
                                  &args().act_quant.storage()))
                              direct_metrics.run_id = meta->run_id;
                          const auto backend_start = read_matmul_cpu_sample();
                          const auto status        = residual::execute_direct_stripe(
                              args(), *payload, correction, &direct_metrics);
                          const auto backend_end = read_matmul_cpu_sample();
#if LOG_CYCLE
                          profile.rmd.direct_call_count  = direct_metrics.call_count;
                          profile.rmd.direct_event_count = direct_metrics.event_count;
#endif
                          record_matmul_cpu_wall(backend_start, backend_end);
                          emit_matmul_cpu_interval(args().matmul_layer.c_str(),
                                                   "residual_backend_host_call",
                                                   backend_start,
                                                   backend_end,
                                                   status == rmd::RmdStatus::success,
                                                   nullptr,
                                                   nullptr,
                                                   run_id);
                          return status;
                      })();
#if LOG_CYCLE
            profile.row_end = row_end;
#endif
            if (residual_status == rmd::RmdStatus::success) {
#if LOG_CYCLE
                const auto observation_start = read_matmul_cpu_sample();
                rmd::collect_direct_metrics(*payload, profile.rmd);
                const auto observation_end = read_matmul_cpu_sample();
                emit_matmul_cpu_interval(args().matmul_layer.c_str(),
                                         "residual_workload_observation",
                                         observation_start,
                                         observation_end,
                                         true,
                                         &profile);
#endif
                const auto merge_start = read_matmul_cpu_sample();
                residual_status        = rmd::merge_rmd_correction(
                    args(), payload->row_begin, row_end, correction, nullptr, measured);
                const auto merge_end = read_matmul_cpu_sample();
                record_matmul_cpu_wall(merge_start, merge_end);
                emit_matmul_cpu_interval(args().matmul_layer.c_str(),
                                         "rmd_merge_cycles",
                                         merge_start,
                                         merge_end,
                                         residual_status == rmd::RmdStatus::success,
                                         nullptr,
                                         nullptr,
                                         run_id);
            }
#if LOG_CYCLE
            emit_rmd_stripe_metrics(args().matmul_layer,
                                    profile,
                                    RmdBackend::cpu_direct,
                                    residual_status == rmd::RmdStatus::success,
                                    rmd::rmd_status_message(residual_status),
                                    profile.rmd.residual_observations_valid ? &profile.rmd
                                                                            : nullptr);
#endif
            if (residual_status != rmd::RmdStatus::success)
                break;
        }
    } else {
        rmd::detail::RmdWeightPreparation weights;
        for (const auto & packet : quants::activation_rmd_packets(args())) {
            if (packet == nullptr)
                continue;
            rmd::RmdExecutionMetrics * measured = nullptr;
#if LOG_CYCLE
            MatmulJobMetrics profile;
            profile.cpu_identity_mask   = run_id ? GEMMINI_CYCLE_HAS_RUN_ID : 0;
            profile.run_id              = run_id.value_or(0);
            profile.stripe_id           = packet->stripe_id;
            profile.row_begin           = packet->row_begin;
            profile.row_end             = packet->row_begin + packet->row_count;
            profile.rmd.timing_identity = {
                {args().matmul_layer.c_str(), nullptr, 0, 0, nullptr, 0, nullptr},
                profile.cpu_identity_mask | GEMMINI_CYCLE_HAS_STRIPE_ID,
                profile.run_id,
                packet->stripe_id,
                0,
                0,
                0};
            measured = &profile.rmd;
#endif
            rmd::Correction correction = rmd::BlockScaledInt64Correction{};
            test_detail::observe_residual_dispatch();
            test_detail::observe_backend_dispatch(false);
            const auto backend_start = read_matmul_cpu_sample();
            residual_status          = rmd::detail::execute_rmd_stripe_ws_with_weights(
                args(), *packet, correction, weights, measured);
            const auto backend_end = read_matmul_cpu_sample();
#if LOG_CYCLE
            performance::incomplete_cpu_wall("accelerator_cpu_stage_coverage_incomplete");
#endif
            emit_matmul_cpu_interval(args().matmul_layer.c_str(),
                                     "residual_backend_host_call",
                                     backend_start,
                                     backend_end,
                                     residual_status == rmd::RmdStatus::success,
                                     nullptr,
                                     nullptr,
                                     run_id);
            if (residual_status == rmd::RmdStatus::success) {
                const auto merge_start = read_matmul_cpu_sample();
                residual_status        = rmd::detail::merge_rmd_correction_with_weights(
                    args(), args().f_out, *packet, correction, weights, nullptr, measured);
                const auto merge_end = read_matmul_cpu_sample();
                record_matmul_cpu_wall(merge_start, merge_end);
                emit_matmul_cpu_interval(args().matmul_layer.c_str(),
                                         "output_correction_apply",
                                         merge_start,
                                         merge_end,
                                         residual_status == rmd::RmdStatus::success,
                                         nullptr,
                                         nullptr,
                                         run_id);
            }
#if LOG_CYCLE
            emit_rmd_stripe_metrics(args().matmul_layer,
                                    profile,
                                    RmdBackend::gemmini_ws_compact,
                                    residual_status == rmd::RmdStatus::success,
                                    rmd::rmd_status_message(residual_status),
                                    profile.rmd.residual_observations_valid ? &profile.rmd
                                                                            : nullptr);
#endif
            if (residual_status != rmd::RmdStatus::success)
                break;
        }
    }
    if (residual_status != rmd::RmdStatus::success) {
        discard_output_transaction();
        return {residual_status == rmd::RmdStatus::unsupported_route
                    ? MatMulStatus::unsupported
                    : MatMulStatus::invalid_arguments,
                MatMulCapability::unsupported};
    }
#endif
    const auto output_start = read_matmul_cpu_sample();
    const auto result       = [&]() -> MatMulResult {
        if (!finite_output(args())) {
            discard_output_transaction();
            return {MatMulStatus::invalid_contract, MatMulCapability::unsupported};
        }
        commit_output_transaction();
        state_ = MatMulState::completed;
        return {MatMulStatus::success, MatMulCapability::supported};
    }();
    const auto output_end = read_matmul_cpu_sample();
    record_matmul_cpu_wall(output_start, output_end);
    emit_matmul_cpu_interval(args().matmul_layer.c_str(),
                             "matmul_output_validation_and_publish",
                             output_start,
                             output_end,
                             result.status == MatMulStatus::success,
                             nullptr,
                             nullptr,
                             run_id);
    return result;
}

MatMulStatus MatMul::begin_stripes() {
    if (state_ != MatMulState::idle) {
        return MatMulStatus::invalid_state;
    }
    if (!valid_matmul_shape(args())) {
        return MatMulStatus::invalid_arguments;
    }
    const bool live_exsia_metadata =
        std::holds_alternative<quants::act::exsia::Meta>(args().act_quant.storage());
    if (!live_exsia_metadata && !valid_activation_metadata(args())) {
        return MatMulStatus::invalid_contract;
    }
    if (stripe_capability(args()) == MatMulCapability::unsupported) {
        return MatMulStatus::unsupported;
    }
    const MatMulStatus transaction = begin_output_transaction();
    if (transaction != MatMulStatus::success) {
        return transaction;
    }
    first_row_      = 0;
    last_row_begin_ = 0;
    last_row_end_   = 0;
    covered_rows_   = 0;
    has_stripes_    = false;
    state_          = MatMulState::accepting_stripes;
    return MatMulStatus::success;
}

MatMulStatus MatMul::run_stripe(MatMulStripe stripe) {
    return run_stripe(stripe, 0);
}

MatMulStatus MatMul::run_stripe(MatMulStripe stripe, size_t stripe_id) {
    if (state_ != MatMulState::accepting_stripes) {
        return MatMulStatus::invalid_state;
    }
    if (stripe.row_begin >= stripe.row_end || stripe.row_end > args().I) {
        discard_output_transaction();
        return MatMulStatus::malformed_stripe;
    }

    if (has_stripes_) {
        if (last_row_begin_ == stripe.row_begin && last_row_end_ == stripe.row_end) {
            discard_output_transaction();
            return MatMulStatus::duplicate_stripe;
        }
        if (stripe.row_begin < last_row_end_) {
            discard_output_transaction();
            return MatMulStatus::overlapping_stripe;
        }
    }

    const MatMulStatus status = execute_stripe(args(), stripe, stripe_id);
    if (status == MatMulStatus::success) {
        if (!has_stripes_) {
            first_row_ = stripe.row_begin;
        }
        last_row_begin_ = stripe.row_begin;
        last_row_end_   = stripe.row_end;
        covered_rows_ += stripe.row_end - stripe.row_begin;
        has_stripes_ = true;
    } else {
        discard_output_transaction();
    }
    return status;
}

MatMulStatus MatMul::run_staged_stripe(MatMulStripe              stripe,
                                       size_t                    stripe_id,
                                       const quants::act::Meta & activation_metadata) {
    if (state_ != MatMulState::accepting_stripes) {
        return MatMulStatus::invalid_state;
    }
    if (stripe.row_begin >= stripe.row_end || stripe.row_end > args().I) {
        discard_output_transaction();
        return MatMulStatus::malformed_stripe;
    }
    ggml_gemmini_args_t staged_args = owned_args_;
    staged_args.act_quant           = activation_metadata;
    staged_args.f_out               = output_stage_.data();
    staged_args.stride_f_out        = output_row_stride_;
    staged_args.col_stride_f_out    = output_col_stride_;
    const MatMulStatus status = execute_stripe(std::move(staged_args), stripe, stripe_id, true);
    if (status == MatMulStatus::success) {
        if (!has_stripes_) {
            first_row_ = stripe.row_begin;
        }
        last_row_begin_ = stripe.row_begin;
        last_row_end_   = stripe.row_end;
        covered_rows_ += stripe.row_end - stripe.row_begin;
        has_stripes_ = true;
    } else {
        discard_output_transaction();
    }
    return status;
}

MatMulStatus MatMul::finish_stripes() {
    const auto output_start = read_matmul_cpu_sample();
    const auto result       = [&]() -> MatMulStatus {
        if (state_ != MatMulState::accepting_stripes) {
            return MatMulStatus::invalid_state;
        }
        if (!has_stripes_) {
            discard_output_transaction();
            state_ = MatMulState::idle;
            return MatMulStatus::empty_stripes;
        }
        if (first_row_ != 0 || last_row_end_ != args().I || covered_rows_ != args().I) {
            discard_output_transaction();
            state_ = MatMulState::idle;
            return MatMulStatus::missing_stripes;
        }
        if (!finite_output(args())) {
            discard_output_transaction();
            state_ = MatMulState::idle;
            return MatMulStatus::invalid_contract;
        }
        commit_output_transaction();
        state_ = MatMulState::completed;
        return MatMulStatus::success;
    }();
    const auto output_end = read_matmul_cpu_sample();
    record_matmul_cpu_wall(output_start, output_end);
    emit_matmul_cpu_interval(args().matmul_layer.c_str(),
                             "matmul_output_validation_and_publish",
                             output_start,
                             output_end,
                             result == MatMulStatus::success,
                             nullptr,
                             nullptr,
                             matmul_cpu_run_id(args()));
    return result;
}

MatMulCapability MatMul::stripe_capability(const ggml_gemmini_args_t & args) {
    const auto format       = args.weight_format;
    const auto capabilities = detail::route_capabilities(args);
    if (format == ggml_gemmini_args_t::im2p_weight_format_t::q8_h2 ||
        format == ggml_gemmini_args_t::im2p_weight_format_t::q8_hp2 ||
        !capabilities.sliced_compensation || args.transpose_A ||
        (args.D != nullptr && !args.repeating_bias) || !supports_row_slice_activation(args)) {
        return MatMulCapability::unsupported;
    }
    if (uses_baseline_channel_route(args) &&
        !std::holds_alternative<quants::act::tensor::Meta>(args.act_quant.storage())) {
        return MatMulCapability::unsupported;
    }
    return MatMulCapability::supported;
}

MatMulState MatMul::state() const {
    return state_;
}

} // namespace ggml::gemmini
