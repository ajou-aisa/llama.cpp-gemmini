#include "rmd-im2p-executor.hpp"

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
#include <im2p_sim.h>

#include <algorithm>
#include <limits>
#include <new>
#include <vector>
#endif

#if defined(GGML_GEMMINI_TESTING)
#include <atomic>
#endif

namespace ggml::gemmini::rmd::detail {

#if defined(GGML_GEMMINI_TESTING)
std::atomic<size_t> provider_dot_attempts{0};
#endif

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
namespace {

struct ProviderContext {
    const Im2pCompactDot * dot = nullptr;
    OutputValue * output = nullptr;
    size_t output_row_stride = 0;
    std::vector<uint8_t> seen;
    size_t seen_count = 0;
    bool fail_read = false;
    bool fail_write = false;
};

int read_weight_i8(void * opaque, size_t row, size_t column, size_t count,
                   int8_t * out) {
    auto * context = static_cast<ProviderContext *>(opaque);
    if (context == nullptr || context->dot == nullptr || out == nullptr ||
        context->fail_read || context->dot->operand_bits == 16 ||
        row >= context->dot->k ||
        column > context->dot->columns || count > context->dot->columns - column) {
        return -1;
    }
    for (size_t index = 0; index < count; ++index) {
        const int32_t value = context->dot->weights[
            row * context->dot->weight_row_stride + column + index];
        const int32_t minimum = context->dot->operand_bits == 4 ? -8 : -128;
        const int32_t maximum = context->dot->operand_bits == 4 ? 7 : 127;
        if (value < minimum || value > maximum) return -1;
        out[index] = static_cast<int8_t>(value);
    }
    return 0;
}

int read_weight_i16(void * opaque, size_t row, size_t column, size_t count,
                    int16_t * out) {
    auto * context = static_cast<ProviderContext *>(opaque);
    if (context == nullptr || context->dot == nullptr || out == nullptr ||
        context->fail_read || context->dot->operand_bits != 16 ||
        row >= context->dot->k ||
        column > context->dot->columns || count > context->dot->columns - column) {
        return -1;
    }
    for (size_t index = 0; index < count; ++index) {
        const int32_t value = context->dot->weights[
            row * context->dot->weight_row_stride + column + index];
        if (value < std::numeric_limits<int16_t>::min() ||
            value > std::numeric_limits<int16_t>::max()) return -1;
        out[index] = static_cast<int16_t>(value);
    }
    return 0;
}

int write_output(void * opaque, size_t block, size_t row, size_t column,
                 size_t count, const int64_t * values) {
    auto * context = static_cast<ProviderContext *>(opaque);
    if (context == nullptr || context->dot == nullptr || values == nullptr ||
        context->fail_write || block != 0 || row >= context->dot->rows ||
        column > context->dot->columns || count > context->dot->columns - column) {
        return -1;
    }
    for (size_t index = 0; index < count; ++index) {
        const size_t logical = row * context->dot->columns + column + index;
        if (logical >= context->seen.size() || context->seen[logical] != 0) {
            return -1;
        }
    }
    for (size_t index = 0; index < count; ++index) {
        const size_t logical = row * context->dot->columns + column + index;
        context->seen[logical] = 1;
        ++context->seen_count;
        context->output[row * context->output_row_stride + column + index] = values[index];
    }
    return 0;
}

} // namespace

void expand_im2p_provider_stats(
    const RmdProviderStats &source,
    im2p_work_stats_extended_t &destination) noexcept {
    size_t index = 0;
#define IM2P_EXPAND(object, field) (object).field = source.fields[index++]
    auto &base = destination.base;
    IM2P_EXPAND(base, work_total_cycles);
    IM2P_EXPAND(base, activation_read_requests);
    IM2P_EXPAND(base, weight_read_requests);
    IM2P_EXPAND(base, scale_read_requests);
    IM2P_EXPAND(base, output_write_requests);
    IM2P_EXPAND(base, output_write_responses);
    IM2P_EXPAND(base, activation_wait_cycles);
    IM2P_EXPAND(base, weight_wait_cycles);
    IM2P_EXPAND(base, scale_wait_cycles);
    IM2P_EXPAND(base, output_wait_cycles);
    IM2P_EXPAND(base, stripe_host_wait_cycles);
    IM2P_EXPAND(base, drain_cycles);
    IM2P_EXPAND(base, weight_preload_cycles);
    IM2P_EXPAND(base, same_block_scale_hits);
    IM2P_EXPAND(base, next_scale_hits);
    IM2P_EXPAND(base, scale_demand_misses);
    IM2P_EXPAND(base, compute_cycles);
    IM2P_EXPAND(base, overlap_cycles);
    IM2P_EXPAND(base, activation_overlap_cycles);
    IM2P_EXPAND(base, weight_overlap_cycles);
    IM2P_EXPAND(base, scale_overlap_cycles);
    IM2P_EXPAND(base, completed_fragments);
    IM2P_EXPAND(base, completed_output_tiles);
    IM2P_EXPAND(base, completed_stripes);
    IM2P_EXPAND(base, stripes_published);
    IM2P_EXPAND(base, stripe_rows_published);
    IM2P_EXPAND(base, weight_bank_activations);
    IM2P_EXPAND(destination, cross_stripe_overlap_cycles);
    IM2P_EXPAND(destination, lookahead_prepared);
    IM2P_EXPAND(destination, lookahead_publish_cycle);
    IM2P_EXPAND(destination, lookahead_first_activation_cycle);
    IM2P_EXPAND(destination, lookahead_first_weight_cycle);
    IM2P_EXPAND(destination, lookahead_weight_preload_cycle);
    IM2P_EXPAND(destination, lookahead_weight_requests);
    IM2P_EXPAND(destination, lookahead_weight_reuse_hits);
    IM2P_EXPAND(destination, lookahead_scale_cycle);
    IM2P_EXPAND(destination, lookahead_scale_requests);
    IM2P_EXPAND(destination, lookahead_scale_reuses);
    IM2P_EXPAND(destination, current_stripe_completion_cycle);
    IM2P_EXPAND(destination, lookahead_ready_cycle);
    IM2P_EXPAND(destination, lookahead_start_cycle);
#undef IM2P_EXPAND
}

namespace {

RmdStatus aggregate_stats(const im2p_work_stats_extended_t &stats,
                          Im2pProviderStatsAggregate &aggregate) {
    RmdProviderStats source{};
    size_t index = 0;
#define IM2P_FLATTEN(object, field) source.fields[index++] = (object).field
    const auto &base = stats.base;
    IM2P_FLATTEN(base, work_total_cycles);
    IM2P_FLATTEN(base, activation_read_requests);
    IM2P_FLATTEN(base, weight_read_requests);
    IM2P_FLATTEN(base, scale_read_requests);
    IM2P_FLATTEN(base, output_write_requests);
    IM2P_FLATTEN(base, output_write_responses);
    IM2P_FLATTEN(base, activation_wait_cycles);
    IM2P_FLATTEN(base, weight_wait_cycles);
    IM2P_FLATTEN(base, scale_wait_cycles);
    IM2P_FLATTEN(base, output_wait_cycles);
    IM2P_FLATTEN(base, stripe_host_wait_cycles);
    IM2P_FLATTEN(base, drain_cycles);
    IM2P_FLATTEN(base, weight_preload_cycles);
    IM2P_FLATTEN(base, same_block_scale_hits);
    IM2P_FLATTEN(base, next_scale_hits);
    IM2P_FLATTEN(base, scale_demand_misses);
    IM2P_FLATTEN(base, compute_cycles);
    IM2P_FLATTEN(base, overlap_cycles);
    IM2P_FLATTEN(base, activation_overlap_cycles);
    IM2P_FLATTEN(base, weight_overlap_cycles);
    IM2P_FLATTEN(base, scale_overlap_cycles);
    IM2P_FLATTEN(base, completed_fragments);
    IM2P_FLATTEN(base, completed_output_tiles);
    IM2P_FLATTEN(base, completed_stripes);
    IM2P_FLATTEN(base, stripes_published);
    IM2P_FLATTEN(base, stripe_rows_published);
    IM2P_FLATTEN(base, weight_bank_activations);
    IM2P_FLATTEN(stats, cross_stripe_overlap_cycles);
    IM2P_FLATTEN(stats, lookahead_prepared);
    IM2P_FLATTEN(stats, lookahead_publish_cycle);
    IM2P_FLATTEN(stats, lookahead_first_activation_cycle);
    IM2P_FLATTEN(stats, lookahead_first_weight_cycle);
    IM2P_FLATTEN(stats, lookahead_weight_preload_cycle);
    IM2P_FLATTEN(stats, lookahead_weight_requests);
    IM2P_FLATTEN(stats, lookahead_weight_reuse_hits);
    IM2P_FLATTEN(stats, lookahead_scale_cycle);
    IM2P_FLATTEN(stats, lookahead_scale_requests);
    IM2P_FLATTEN(stats, lookahead_scale_reuses);
    IM2P_FLATTEN(stats, current_stripe_completion_cycle);
    IM2P_FLATTEN(stats, lookahead_ready_cycle);
    IM2P_FLATTEN(stats, lookahead_start_cycle);
#undef IM2P_FLATTEN
    return checked_accumulate_provider_stats(aggregate.stats, source);
}

int synthetic_execute(const im2p_matmul_desc_t * descriptor,
                      im2p_work_stats_extended_t * stats,
                      Im2pProviderTestFault fault) {
    if (descriptor == nullptr || descriptor->provider.write_output == nullptr ||
        fault == Im2pProviderTestFault::watchdog) return IM2P_ERROR;
    std::vector<int64_t> row(descriptor->n, 0);
    for (size_t i = 0; i < descriptor->m; ++i) {
        for (size_t j = 0; j < descriptor->n; ++j) {
            int64_t sum = 0;
            for (size_t k = 0; k < descriptor->k; ++k) {
                if (descriptor->weight_bits == 16) {
                    int16_t value = 0;
                    if (descriptor->provider.read_weight_i16(
                            descriptor->provider.context, k, j, 1, &value) != 0) return IM2P_ERROR;
                    sum += static_cast<const int16_t *>(descriptor->activations)[
                               i * descriptor->activation_row_stride_bytes / sizeof(int16_t) + k] *
                           static_cast<int64_t>(value);
                } else {
                    int8_t value = 0;
                    if (descriptor->provider.read_weight_i8(
                            descriptor->provider.context, k, j, 1, &value) != 0) return IM2P_ERROR;
                    sum += static_cast<const int8_t *>(descriptor->activations)[
                               i * descriptor->activation_row_stride_bytes + k] *
                           static_cast<int64_t>(value);
                }
            }
            row[j] = sum;
        }
        if (fault != Im2pProviderTestFault::missing_output &&
            descriptor->provider.write_output(
                descriptor->provider.context, 0,
                fault == Im2pProviderTestFault::output_index ? descriptor->m : i,
                0, descriptor->n, row.data()) != 0) return IM2P_ERROR;
        if (fault == Im2pProviderTestFault::duplicate_output &&
            descriptor->provider.write_output(descriptor->provider.context, 0, i, 0,
                                              descriptor->n, row.data()) != 0) return IM2P_ERROR;
    }
    if (stats != nullptr) {
        stats->base.work_total_cycles = fault == Im2pProviderTestFault::stats_overflow
            ? std::numeric_limits<uint64_t>::max() : 1;
        stats->base.output_write_requests = descriptor->m;
        stats->base.output_write_responses = descriptor->m;
    }
    return IM2P_OK;
}

} // namespace
#endif

RmdStatus execute_im2p_compact_dot(
    im2p_sim_t * sim,
    const Im2pCompactDot & dot,
    OutputValue * output,
    size_t output_row_stride,
    Im2pProviderStatsAggregate & aggregate,
    Im2pProviderTestFault fault) {
#if !defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    (void) sim; (void) dot; (void) output; (void) output_row_stride;
    (void) aggregate; (void) fault;
    return RmdStatus::unsupported_route;
#else
    if (sim == nullptr || dot.activations == nullptr || dot.weights == nullptr ||
        output == nullptr || dot.rows == 0 || dot.columns == 0 || dot.k == 0 ||
        dot.activation_row_stride < dot.k || dot.weight_row_stride < dot.columns ||
        output_row_stride < dot.columns ||
        (dot.operand_bits != 4 && dot.operand_bits != 8 && dot.operand_bits != 16) ||
        dot.operand_bits != GGML_GEMMINI_ACTIVATION_BITS ||
        dot.operand_bits != GGML_GEMMINI_WEIGHT_BITS) {
        return RmdStatus::invalid_arguments;
    }

    ProviderContext context{};
    context.dot = &dot;
    context.output = output;
    context.output_row_stride = output_row_stride;
    context.fail_read = fault == Im2pProviderTestFault::read_failure;
    context.fail_write = fault == Im2pProviderTestFault::write_failure;
    try {
        context.seen.assign(dot.rows * dot.columns, uint8_t{0});
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    std::vector<int8_t> activation_i8;
    std::vector<int16_t> activation_i16;
    const void * activation_data = nullptr;
    try {
        if (dot.operand_bits == 16) {
            activation_i16.resize(dot.rows * dot.k);
            for (size_t row = 0; row < dot.rows; ++row) {
                for (size_t k = 0; k < dot.k; ++k) {
                    const int32_t value = dot.activations[row * dot.activation_row_stride + k];
                    if (value < std::numeric_limits<int16_t>::min() ||
                        value > std::numeric_limits<int16_t>::max()) return RmdStatus::overflow;
                    activation_i16[row * dot.k + k] = static_cast<int16_t>(value);
                }
            }
            activation_data = activation_i16.data();
        } else {
            const int32_t minimum = dot.operand_bits == 4 ? -8 : -128;
            const int32_t maximum = dot.operand_bits == 4 ? 7 : 127;
            activation_i8.resize(dot.rows * dot.k);
            for (size_t row = 0; row < dot.rows; ++row) {
                for (size_t k = 0; k < dot.k; ++k) {
                    const int32_t value = dot.activations[row * dot.activation_row_stride + k];
                    if (value < minimum || value > maximum) return RmdStatus::overflow;
                    // A4 crosses the ABI unpacked: one signed byte per low-nibble-first digit.
                    activation_i8[row * dot.k + k] = static_cast<int8_t>(value);
                }
            }
            activation_data = activation_i8.data();
        }
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    im2p_matmul_desc_t descriptor{};
    descriptor.abi_version = IM2P_ABI_VERSION;
    descriptor.activation_bits = dot.operand_bits;
    descriptor.activation_storage_bytes = dot.operand_bits == 16 ? 2 : 1;
    descriptor.weight_bits = dot.operand_bits;
    descriptor.weight_storage_bytes = dot.operand_bits == 16 ? 2 : 1;
    descriptor.dim = DIM;
    descriptor.activations = activation_data;
    descriptor.weights = nullptr;
    descriptor.scales = nullptr;
    descriptor.output = nullptr;
    descriptor.m = dot.rows;
    descriptor.n = dot.columns;
    descriptor.k = dot.k;
    descriptor.activation_row_stride_bytes = dot.k * descriptor.activation_storage_bytes;
    descriptor.weight_row_stride_bytes = dot.columns * descriptor.weight_storage_bytes;
    descriptor.output_row_stride = output_row_stride;
    descriptor.tile_i_rows = std::min(dot.rows, static_cast<size_t>(DIM));
    descriptor.tile_j_columns = std::min(dot.columns, static_cast<size_t>(DIM));
    descriptor.block_size = 1;
    descriptor.vector_op = IM2P_VECTOR_BYPASS;
    descriptor.provider.context = &context;
    descriptor.provider.read_weight_i8 = dot.operand_bits == 16 ? nullptr : read_weight_i8;
    descriptor.provider.read_weight_i16 = dot.operand_bits == 16 ? read_weight_i16 : nullptr;
    descriptor.provider.read_scale = nullptr;
    descriptor.provider.write_output = write_output;

    im2p_work_stats_extended_t stats{};
#if defined(GGML_GEMMINI_TESTING)
    provider_dot_attempts.fetch_add(1, std::memory_order_relaxed);
#endif
    const int provider_status = fault == Im2pProviderTestFault::none
        ? im2p_execute_matmul_extended(sim, &descriptor, &stats)
        : synthetic_execute(&descriptor, &stats, fault);
    if (provider_status != IM2P_OK) return RmdStatus::execution_failed;
    if (context.seen_count != dot.rows * dot.columns) return RmdStatus::invalid_packet;
    return aggregate_stats(stats, aggregate);
#endif
}

} // namespace ggml::gemmini::rmd::detail

#if defined(GGML_GEMMINI_TESTING)
namespace ggml::gemmini::rmd {

void reset_im2p_provider_dot_attempts_for_test() {
    detail::provider_dot_attempts.store(0, std::memory_order_relaxed);
}

size_t im2p_provider_dot_attempts_for_test() {
    return detail::provider_dot_attempts.load(std::memory_order_relaxed);
}

} // namespace ggml::gemmini::rmd
#endif
