#include "rmd-im2p-executor.hpp"
#include "rmd-run-aware.hpp"
#include <gemmini/trace-context.hpp>
#include <gemmini/log.hpp>
#include "../../../../../common/json.hpp"

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) ||                        \
    defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
#include "../../ggml-gemmini-args.h"
#include "../../quants/common/hp1_scu.hpp"
#include <gemmini.h>
#include <im2p_sim.h>
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
#include <im2p_production_trace.hpp>
#if CYCLE_SIM
#include <im2p_cycle_sim.hpp>
#endif
#endif

#include <algorithm>
#include <cstring>
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

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) ||                        \
    defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
namespace {

struct ProviderContext {
  const Im2pCompactDot *dot = nullptr;
  OutputValue *output = nullptr;
    size_t output_row_stride = 0;
    std::vector<uint8_t> seen;
    size_t seen_count = 0;
    bool fail_read = false;
    bool fail_write = false;
};

struct RunAwareProviderContext {
  const RunAwareRequest *request = nullptr;
  std::vector<OutputValue> staged;
  size_t writes = 0;
};

int read_run_weight_i8(void *opaque, size_t row, size_t column, size_t count,
                       int8_t *out) {
  const auto *context = static_cast<const RunAwareProviderContext *>(opaque);
  if (!context || !context->request || !out ||
      row >= context->request->k || column > context->request->n ||
      count > context->request->n - column)
    return IM2P_ERROR;
  const int32_t minimum = context->request->operand_bits == 4 ? -8 : -128;
  const int32_t maximum = context->request->operand_bits == 4 ? 7 : 127;
  const int32_t *source = context->request->weights.data() +
                          row * context->request->n + column;
  for (size_t index = 0; index < count; ++index) {
    if (source[index] < minimum || source[index] > maximum)
      return IM2P_INVALID_LAYOUT;
    out[index] = static_cast<int8_t>(source[index]);
  }
  return IM2P_OK;
}

int read_run_scale(void *opaque, size_t run, size_t column, size_t count,
                   uint32_t *out) {
  const auto *context = static_cast<const RunAwareProviderContext *>(opaque);
  if (!context || !context->request || !out ||
      run >= context->request->runs.size() ||
      column > context->request->n || count > context->request->n - column)
    return IM2P_ERROR;
  const uint32_t *source = context->request->carriers.data() +
                           run * context->request->n + column;
  for (size_t index = 0; index < count; ++index) {
    if (!quants::hp1::valid_carrier(source[index]))
      return IM2P_INVALID_LAYOUT;
    out[index] = source[index];
  }
  return IM2P_OK;
}

int write_run_output(void *opaque, size_t block, size_t row, size_t column,
                     size_t count, const int64_t *values,
                     uint32_t output_domain) {
  auto *context = static_cast<RunAwareProviderContext *>(opaque);
  if (!context || !context->request || !values || context->writes != 0 ||
      block != 0 || row != 0 || column != 0 ||
      output_domain != IM2P_OUTPUT_SCU_FINAL ||
      count != context->staged.size())
    return IM2P_ERROR;
  for (size_t index = 0; index < count; ++index)
    if (values[index] < INT32_MIN || values[index] > INT32_MAX)
      return IM2P_INVALID_LAYOUT;
  std::copy_n(values, count, context->staged.begin());
  context->writes = 1;
  return IM2P_OK;
}

RmdStatus prepare_run_view(const RunAwareRequest &request,
                           std::vector<im2p_compact_run_t> &entries) {
  size_t activations = 0;
  size_t weights = 0;
  size_t carriers = 0;
  if ((request.operand_bits != 4 && request.operand_bits != 8) ||
      request.operand_bits != GGML_GEMMINI_ACTIVATION_BITS ||
      request.operand_bits != GGML_GEMMINI_WEIGHT_BITS || !request.m ||
      !request.n || !request.k || !request.original_k || request.runs.empty() ||
      request.rows.size() != request.m || !request.source_row_count ||
      !request.tile_i || !request.tile_j || !request.tile_k ||
      request.m > UINT32_MAX || request.n > UINT32_MAX ||
      request.k > UINT32_MAX || request.original_k > UINT32_MAX ||
      request.m > SIZE_MAX / request.k ||
      (activations = request.m * request.k,
       request.k > SIZE_MAX / request.n) ||
      (weights = request.k * request.n,
       request.runs.size() > SIZE_MAX / request.n) ||
      (carriers = request.runs.size() * request.n,
       request.activations.size() != activations) ||
      request.weights.size() != weights ||
      request.carriers.size() != carriers)
    return RmdStatus::invalid_arguments;

  try {
    entries.reserve(request.runs.size());
  } catch (const std::bad_alloc &) {
    return RmdStatus::allocation_failure;
  }
  uint64_t cursor = 0;
  uint32_t previous_block = 0;
  for (size_t index = 0; index < request.runs.size(); ++index) {
    const RunAwareRun &run = request.runs[index];
    uint32_t local_mask = 0;
    uint16_t previous_local = 0;
    for (size_t local_index = 0; local_index < run.original_local_k.size();
         ++local_index) {
      const uint16_t local_k = run.original_local_k[local_index];
      if (local_k >= 32 || (local_index && local_k <= previous_local))
        return RmdStatus::invalid_arguments;
      local_mask |= uint32_t{1} << local_k;
      previous_local = local_k;
    }
    const uint64_t original_begin = uint64_t{run.original_block_id} * 32;
    if (!run.compact_k_count || run.compact_k_count > 32 ||
        run.compact_k_begin != cursor ||
        run.original_local_k.size() != run.compact_k_count ||
        run.union_k_mask != local_mask || !run.union_k_mask ||
        (index && run.original_block_id <= previous_block) ||
        run.original_global_k_begin != original_begin ||
        original_begin >= request.original_k ||
        run.compact_k_begin > UINT32_MAX ||
        run.compact_k_count > UINT32_MAX)
      return RmdStatus::invalid_arguments;
    for (const uint16_t local_k : run.original_local_k)
      if (original_begin + local_k >= request.original_k)
        return RmdStatus::invalid_arguments;
    entries.push_back({run.original_block_id, run.union_k_mask,
                       static_cast<uint32_t>(run.compact_k_begin),
                       static_cast<uint32_t>(run.compact_k_count)});
    cursor += run.compact_k_count;
    previous_block = run.original_block_id;
  }
  if (cursor != request.k)
    return RmdStatus::invalid_arguments;

  uint64_t previous_row = 0;
  const uint32_t lane_capacity = 32 / request.operand_bits + 1;
  for (size_t index = 0; index < request.rows.size(); ++index) {
    const RunAwareRow &row = request.rows[index];
    if (row.source_row >= request.source_row_count ||
        row.original_lane_id >= lane_capacity)
      return RmdStatus::invalid_arguments;
    const uint64_t key = uint64_t{row.original_lane_id} *
                             request.source_row_count +
                         row.source_row;
    if (index && key <= previous_row)
      return RmdStatus::invalid_arguments;
    previous_row = key;
  }
  return RmdStatus::success;
}

int read_weight_i8(void *opaque, size_t row, size_t column, size_t count,
                   int8_t *out) {
  auto *context = static_cast<ProviderContext *>(opaque);
    if (context == nullptr || context->dot == nullptr || out == nullptr ||
        context->fail_read || context->dot->operand_bits == 16 ||
      row >= context->dot->k || column > context->dot->columns ||
      count > context->dot->columns - column) {
        return -1;
    }
    for (size_t index = 0; index < count; ++index) {
    const int32_t value =
        context->dot
            ->weights[row * context->dot->weight_row_stride + column + index];
        const int32_t minimum = context->dot->operand_bits == 4 ? -8 : -128;
        const int32_t maximum = context->dot->operand_bits == 4 ? 7 : 127;
    if (value < minimum || value > maximum)
      return -1;
        out[index] = static_cast<int8_t>(value);
    }
    return 0;
}

int read_weight_i16(void *opaque, size_t row, size_t column, size_t count,
                    int16_t *out) {
  auto *context = static_cast<ProviderContext *>(opaque);
    if (context == nullptr || context->dot == nullptr || out == nullptr ||
        context->fail_read || context->dot->operand_bits != 16 ||
      row >= context->dot->k || column > context->dot->columns ||
      count > context->dot->columns - column) {
        return -1;
    }
    for (size_t index = 0; index < count; ++index) {
    const int32_t value =
        context->dot
            ->weights[row * context->dot->weight_row_stride + column + index];
        if (value < std::numeric_limits<int16_t>::min() ||
        value > std::numeric_limits<int16_t>::max())
      return -1;
        out[index] = static_cast<int16_t>(value);
    }
    return 0;
}

int read_hp1_scale(void *opaque, size_t block, size_t column, size_t count,
                   uint32_t *out) {
  const auto *context = static_cast<const ProviderContext *>(opaque);
  if (!context || !context->dot || !out || context->fail_read || block != 0 ||
      !context->dot->hp1_carriers || column > context->dot->columns ||
      count > context->dot->columns - column)
    return IM2P_ERROR;
  for (size_t j = 0; j < count; ++j) {
    const auto carrier = context->dot->hp1_carriers[column + j];
    if (!quants::hp1::valid_carrier(carrier))
      return IM2P_INVALID_LAYOUT;
    out[j] = carrier;
  }
  return IM2P_OK;
}

int write_output(void *opaque, size_t block, size_t row, size_t column,
                 size_t count, const int64_t *values, uint32_t output_domain) {
  auto *context = static_cast<ProviderContext *>(opaque);
    if (context == nullptr || context->dot == nullptr || values == nullptr ||
      context->fail_write ||
      output_domain != (context->dot->hp1_carriers
                            ? IM2P_OUTPUT_SCU_FINAL
                            : IM2P_OUTPUT_LEGACY_FINAL) ||
      block != 0 || row >= context->dot->rows ||
      column > context->dot->columns ||
      count > context->dot->columns - column) {
        return -1;
    }
    for (size_t index = 0; index < count; ++index) {
        const size_t logical = row * context->dot->columns + column + index;
    if (logical >= context->seen.size() || context->seen[logical] != 0 ||
        (context->dot->hp1_carriers &&
         (values[index] < INT32_MIN || values[index] > INT32_MAX))) {
            return -1;
        }
    }
    for (size_t index = 0; index < count; ++index) {
        const size_t logical = row * context->dot->columns + column + index;
        context->seen[logical] = 1;
        ++context->seen_count;
    context->output[row * context->output_row_stride + column + index] =
        values[index];
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

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
int synthetic_execute(const im2p_matmul_desc_t *descriptor,
                      im2p_work_stats_extended_t *stats,
                      Im2pProviderTestFault fault) {
    if (descriptor == nullptr || descriptor->provider.write_output == nullptr ||
      fault == Im2pProviderTestFault::watchdog)
    return IM2P_ERROR;
    std::vector<int64_t> row(descriptor->n, 0);
    for (size_t i = 0; i < descriptor->m; ++i) {
        for (size_t j = 0; j < descriptor->n; ++j) {
            int64_t sum = 0;
      int32_t scaled_acc = 0;
      uint32_t carrier = 0;
      const bool scaled = descriptor->vector_op == IM2P_VECTOR_LEFT_SHIFT;
      if (scaled &&
          (!descriptor->provider.read_scale ||
           descriptor->provider.read_scale(descriptor->provider.context, 0, j,
                                           1, &carrier) != IM2P_OK ||
           !quants::hp1::valid_carrier(carrier)))
        return IM2P_ERROR;
            for (size_t k = 0; k < descriptor->k; ++k) {
                if (descriptor->weight_bits == 16) {
                    int16_t value = 0;
          if (descriptor->provider.read_weight_i16(descriptor->provider.context,
                                                   k, j, 1, &value) != 0)
            return IM2P_ERROR;
          sum += static_cast<const int16_t *>(descriptor->activations)
                     [i * descriptor->activation_row_stride_bytes /
                          sizeof(int16_t) +
                      k] *
                           static_cast<int64_t>(value);
                } else {
                    int8_t value = 0;
          if (descriptor->provider.read_weight_i8(descriptor->provider.context,
                                                  k, j, 1, &value) != 0)
            return IM2P_ERROR;
          sum += static_cast<const int8_t *>(descriptor->activations)
                     [i * descriptor->activation_row_stride_bytes + k] *
                           static_cast<int64_t>(value);
                }
        if (scaled && ((k + 1) % std::min<size_t>(DIM, 32) == 0 ||
                       k + 1 == descriptor->k)) {
          if (sum < INT32_MIN || sum > INT32_MAX)
            return IM2P_ERROR;
          const auto q =
              quants::hp1::apply_validated(static_cast<int32_t>(sum), carrier);
          scaled_acc = k < std::min<size_t>(DIM, 32)
                           ? q
                           : quants::hp1::accumulate(scaled_acc, q);
          sum = 0;
            }
      }
      row[j] = scaled ? scaled_acc : sum;
        }
        if (fault != Im2pProviderTestFault::missing_output &&
            descriptor->provider.write_output(
                descriptor->provider.context, 0,
            fault == Im2pProviderTestFault::output_index ? descriptor->m : i, 0,
            descriptor->n, row.data(), descriptor->output_domain) != 0)
      return IM2P_ERROR;
        if (fault == Im2pProviderTestFault::duplicate_output &&
            descriptor->provider.write_output(descriptor->provider.context, 0, i, 0,
                                          descriptor->n, row.data(),
                                          descriptor->output_domain) != 0)
      return IM2P_ERROR;
    }
    if (stats != nullptr) {
    stats->base.work_total_cycles =
        fault == Im2pProviderTestFault::stats_overflow
            ? std::numeric_limits<uint64_t>::max()
            : 1;
        stats->base.output_write_requests = descriptor->m;
        stats->base.output_write_responses = descriptor->m;
    }
    return IM2P_OK;
}
#endif

} // namespace
#endif

RmdStatus execute_im2p_run_aware(
    im2p_sim_t *sim, const Im2pRunAwareWork &work,
    std::vector<OutputValue> &output, Im2pProviderStatsAggregate &aggregate,
    const Im2pFullExecutor *executor) {
#if CYCLE_SIM
  struct CompletionGuard {
    const cycle_sim::Context &context;
    bool success = false;
    ~CompletionGuard() {
      if (context && !success)
        context.session->record_failure(
            "CPU-functional run-aware residual dispatch failed");
    }
  } completion{work.cycle_sim_context};
#endif
#if !defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) &&                       \
    !defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
  (void)sim;
  (void)work;
  (void)output;
  (void)aggregate;
  (void)executor;
  return RmdStatus::unsupported_route;
#else
  if (!work.request)
    return RmdStatus::invalid_arguments;
  if (work.trace_context)
    return RmdStatus::unsupported_route;
  if (executor ? executor->execute_planned_runs == nullptr : sim == nullptr)
    return executor ? RmdStatus::unsupported_route
                    : RmdStatus::invalid_arguments;
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
  if (!executor)
    return RmdStatus::unsupported_route;
#endif
#if CYCLE_SIM
  if ((work.cycle_sim_context && !work.cycle_sim_context.operation_id) ||
      executor || std::strcmp(im2p_sim_implementation(), "CPU_FUNCTIONAL") != 0)
    return RmdStatus::unsupported_route;
#endif

  const RunAwareRequest &request = *work.request;
  std::vector<im2p_compact_run_t> run_entries;
  const RmdStatus view_status = prepare_run_view(request, run_entries);
  if (view_status != RmdStatus::success)
    return view_status;
  size_t output_count = 0;
  if (__builtin_mul_overflow(request.m, request.n, &output_count))
    return RmdStatus::overflow;

  RunAwareProviderContext context;
  context.request = &request;
  try {
    context.staged.resize(output_count);
  } catch (const std::bad_alloc &) {
    return RmdStatus::allocation_failure;
  }

  im2p_matmul_desc_t descriptor{};
  descriptor.abi_version = IM2P_ABI_VERSION;
  descriptor.activation_bits = request.operand_bits;
  descriptor.activation_storage_bytes = 1;
  descriptor.weight_bits = request.operand_bits;
  descriptor.weight_storage_bytes = 1;
  descriptor.dim = DIM;
  descriptor.activations = request.activations.data();
  descriptor.m = request.m;
  descriptor.n = request.n;
  descriptor.k = request.k;
  descriptor.activation_row_stride_bytes = request.k;
  descriptor.weight_row_stride_bytes = request.n;
  descriptor.output_row_stride = request.n;
  descriptor.tile_i_rows = std::min(request.m, static_cast<size_t>(DIM));
  descriptor.tile_j_columns = std::min(request.n, static_cast<size_t>(DIM));
  descriptor.block_size = 32;
  descriptor.scale_total_k = request.original_k;
  descriptor.scale_row_stride = request.n;
  descriptor.scale_valid_columns = request.n;
  descriptor.scale_values_len = request.carriers.size();
  descriptor.vector_op = IM2P_VECTOR_LEFT_SHIFT;
  descriptor.output_domain = IM2P_OUTPUT_SCU_FINAL;
  descriptor.work_context = request.stripe_id;
  descriptor.provider = {&context, read_run_weight_i8, nullptr, read_run_scale,
                         write_run_output};

  const im2p_production_geometry_v1_t geometry{
      IM2P_PRODUCTION_GEOMETRY_VERSION,
      sizeof(im2p_production_geometry_v1_t),
      request.operand_bits,
      request.operand_bits,
      DIM,
      IM2P_GEOMETRY_FULL,
      request.m,
      request.n,
      request.k,
      request.tile_i,
      request.tile_j,
      request.tile_k,
      request.m,
      0,
      request.m,
      0};
  const im2p_compact_runs_t runs{IM2P_COMPACT_RUNS_VERSION,
                                 sizeof(im2p_compact_runs_t),
                                 static_cast<uint32_t>(request.original_k),
                                 run_entries.size(), run_entries.data()};

#if CYCLE_SIM
  cycle_sim::Context cycle_context;
  cycle_sim::Work selected_work;
  try {
    if (work.cycle_sim_context)
      cycle_context = work.cycle_sim_context.session->new_dispatch(
          work.cycle_sim_context);
    selected_work = im2p::gemmini::cycle_sim::full(descriptor, geometry);
    selected_work.provenance = "residual";
    selected_work.scope = "residual_compact";
    selected_work.stripe_id = request.stripe_id;
    selected_work.original_k = static_cast<uint32_t>(request.original_k);
    selected_work.runs = run_entries;
    selected_work.row_map.reserve(request.rows.size());
    for (const RunAwareRow &row : request.rows)
      selected_work.row_map.push_back(
          {row.source_row, row.original_lane_id});
    selected_work.source_row_begin = request.source_row_begin;
    selected_work.source_row_count = request.source_row_count;
    selected_work.required_host_stage_ids = work.required_host_stage_ids;
  } catch (const std::bad_alloc &) {
    return RmdStatus::allocation_failure;
  } catch (...) {
    return RmdStatus::execution_failed;
  }
  cycle_sim::ScopedContext cycle_scope(cycle_context);
  im2p::gemmini::cycle_sim::DispatchEvents dispatch_events(
      cycle_context, &selected_work, cycle_sim::CallKind::ResidualCompact);
  im2p::cpu_functional::TimingRegistration timing_registration(
      dispatch_events.observer());
#endif

  im2p_work_stats_extended_t stats{};
#if defined(GGML_GEMMINI_TESTING)
  provider_dot_attempts.fetch_add(1, std::memory_order_relaxed);
#endif
  trace::ScopedRole residual_role(GEMMINI_TRACE_ROLE_RESIDUAL);
  trace::CpuStage host_call(work.timing_identity.interval.layer,
                            "rmd.device_host_call");
  int provider_status = IM2P_ERROR;
  if (executor)
    provider_status = executor->execute_planned_runs(
        executor->context, &descriptor, &geometry, &runs, &stats);
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
  else
    provider_status = im2p_execute_matmul_planned_runs(
        sim, &descriptor, &geometry, &runs, &stats);
#endif
  host_call.finish(provider_status == IM2P_OK);
  if (provider_status != IM2P_OK)
    return RmdStatus::execution_failed;
#if CYCLE_SIM
  if (cycle_context) {
    try {
      dispatch_events.complete();
      cycle_context.session->ensure_healthy();
    } catch (...) {
      return RmdStatus::execution_failed;
    }
  }
#endif
  if (context.writes != 1)
    return RmdStatus::invalid_packet;
  const RmdStatus stats_status = aggregate_stats(stats, aggregate);
  if (stats_status != RmdStatus::success)
    return stats_status;
  output.swap(context.staged);
#if CYCLE_SIM
  completion.success = true;
#endif
  return RmdStatus::success;
#endif
}

RmdStatus execute_im2p_compact_dot(im2p_sim_t *sim, const Im2pCompactDot &dot,
                                   OutputValue *output,
    size_t output_row_stride,
                                   Im2pProviderStatsAggregate &aggregate,
    Im2pProviderTestFault fault,
                                   const Im2pFullExecutor *executor) {
#if CYCLE_SIM
  struct CompletionGuard {
    const cycle_sim::Context &context;
    bool success = false;
    ~CompletionGuard() {
      if (context && !success)
        context.session->record_failure("CPU-functional residual dispatch failed");
    }
  } completion{dot.cycle_sim_context};
#endif
#if !defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM) &&                       \
    !defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
  (void)sim;
  (void)dot;
  (void)output;
  (void)output_row_stride;
  (void)aggregate;
  (void)fault;
  (void)executor;
    return RmdStatus::unsupported_route;
#else
  if ((executor != nullptr ? (executor->execute == nullptr &&
                              executor->execute_planned == nullptr)
                           : sim == nullptr) ||
      dot.activations == nullptr || dot.weights == nullptr ||
        output == nullptr || dot.rows == 0 || dot.columns == 0 || dot.k == 0 ||
      dot.activation_row_stride_bytes <
          dot.k * (dot.operand_bits == 16 ? sizeof(int16_t) : 1) ||
      (dot.operand_bits == 16 &&
       dot.activation_row_stride_bytes % sizeof(int16_t) != 0) ||
      dot.weight_row_stride < dot.columns || output_row_stride < dot.columns ||
      (dot.operand_bits != 4 && dot.operand_bits != 8 &&
       dot.operand_bits != 16) ||
        dot.operand_bits != GGML_GEMMINI_ACTIVATION_BITS ||
        dot.operand_bits != GGML_GEMMINI_WEIGHT_BITS) {
        return RmdStatus::invalid_arguments;
    }

#if defined(GGML_GEMMINI_EXECUTION_BACKEND_FPGA_UART)
    if (executor == nullptr || fault != Im2pProviderTestFault::none)
        return RmdStatus::unsupported_route;
#endif
  const bool scaled = dot.hp1_carriers != nullptr;
#if CYCLE_SIM
  if (dot.trace_context || (dot.cycle_sim_context &&
      (!dot.cycle_sim_context.operation_id || !scaled || executor ||
       fault != Im2pProviderTestFault::none)))
    return RmdStatus::unsupported_route;
#endif
  if (dot.trace_context &&
      (!*dot.trace_context || !scaled || executor ||
       fault != Im2pProviderTestFault::none))
    return RmdStatus::unsupported_route;
  if (scaled &&
      ((dot.operand_bits != 4 && dot.operand_bits != 8) || dot.k > 32))
    return RmdStatus::unsupported_route;
  if (scaled && executor && !executor->execute_planned)
    return RmdStatus::unsupported_route;
  if (!scaled && executor && !executor->execute)
    return RmdStatus::unsupported_route;
  if (scaled)
    for (size_t j = 0; j < dot.columns; ++j)
      if (!quants::hp1::valid_carrier(dot.hp1_carriers[j]))
        return RmdStatus::invalid_arguments;
  if (dot.rows > SIZE_MAX / dot.columns ||
      dot.rows > SIZE_MAX / output_row_stride ||
      dot.rows > SIZE_MAX / dot.activation_row_stride_bytes ||
      dot.k > SIZE_MAX / dot.weight_row_stride)
    return RmdStatus::overflow;
    ProviderContext context{};
  std::vector<OutputValue> staged_values;
    context.dot = &dot;
  context.output_row_stride = dot.columns;
    context.fail_read = fault == Im2pProviderTestFault::read_failure;
    context.fail_write = fault == Im2pProviderTestFault::write_failure;
    try {
        context.seen.assign(dot.rows * dot.columns, uint8_t{0});
    staged_values.resize(dot.rows * dot.columns);
    context.output = staged_values.data();
    } catch (const std::bad_alloc &) {
        return RmdStatus::allocation_failure;
    }

    im2p_matmul_desc_t descriptor{};
    descriptor.abi_version = IM2P_ABI_VERSION;
    descriptor.activation_bits = dot.operand_bits;
  // A4 is already unpacked to signed bytes by the caller; A16 uses two bytes
  // per digit. Preserve the caller's byte stride so padded native rows need no
  // repacking here.
    descriptor.activation_storage_bytes = dot.operand_bits == 16 ? 2 : 1;
    descriptor.weight_bits = dot.operand_bits;
    descriptor.weight_storage_bytes = dot.operand_bits == 16 ? 2 : 1;
    descriptor.dim = DIM;
    descriptor.activations = dot.activations;
    descriptor.weights = nullptr;
    descriptor.scales = nullptr;
    descriptor.output = nullptr;
    descriptor.m = dot.rows;
    descriptor.n = dot.columns;
    descriptor.k = dot.k;
    descriptor.activation_row_stride_bytes = dot.activation_row_stride_bytes;
  descriptor.weight_row_stride_bytes =
      dot.columns * descriptor.weight_storage_bytes;
    descriptor.output_row_stride = output_row_stride;
    descriptor.tile_i_rows = std::min(dot.rows, static_cast<size_t>(DIM));
    descriptor.tile_j_columns = std::min(dot.columns, static_cast<size_t>(DIM));
  descriptor.block_size = scaled ? 32 : 1;
  descriptor.vector_op = scaled ? IM2P_VECTOR_LEFT_SHIFT : IM2P_VECTOR_BYPASS;
  descriptor.output_domain =
      scaled ? IM2P_OUTPUT_SCU_FINAL : IM2P_OUTPUT_LEGACY_FINAL;
  descriptor.scale_total_k = scaled ? dot.k : 0;
  descriptor.scale_row_stride = scaled ? dot.columns : 0;
  descriptor.scale_valid_columns = scaled ? dot.columns : 0;
  descriptor.work_context = dot.block_id;
    descriptor.provider.context = &context;
  descriptor.provider.read_weight_i8 =
      dot.operand_bits == 16 ? nullptr : read_weight_i8;
  descriptor.provider.read_weight_i16 =
      dot.operand_bits == 16 ? read_weight_i16 : nullptr;
  descriptor.provider.read_scale = scaled ? read_hp1_scale : nullptr;
    descriptor.provider.write_output = write_output;

  im2p_production_geometry_v1_t geometry{};
  if (scaled) {
    // A compact GEMM is a new logical shape. The SAME production software
    // selector chooses its factors once; every downstream layer copies them.
    ggml_gemmini_args_t selected{};
    selected.I = dot.rows;
    selected.J = dot.columns;
    selected.K = dot.k;
    ggml::gemmini::gemmini_set_tile_ws(&selected);
    geometry = {IM2P_PRODUCTION_GEOMETRY_VERSION,
                sizeof(geometry),
                dot.operand_bits,
                dot.operand_bits,
                DIM,
                IM2P_GEOMETRY_FULL,
                dot.rows,
                dot.columns,
                dot.k,
                selected.tile_I,
                selected.tile_J,
                selected.tile_K,
                dot.rows,
                0,
                dot.rows,
                0};
  }
#if CYCLE_SIM
  cycle_sim::Context cycle_context;
  try {
    if (dot.cycle_sim_context)
      cycle_context = dot.cycle_sim_context.session->new_dispatch(dot.cycle_sim_context);
  } catch (...) {
    return RmdStatus::execution_failed;
  }
  cycle_sim::ScopedContext cycle_scope(cycle_context);
  auto selected_work = im2p::gemmini::cycle_sim::full(descriptor, geometry);
  selected_work.provenance = "residual";
  selected_work.scope = "residual_compact";
  selected_work.original_block_id = dot.block_id;
  selected_work.source_row_begin = dot.source_row_begin;
  selected_work.source_row_count = dot.source_row_count;
  selected_work.stripe_id = dot.stripe_id;
  selected_work.column_begin = dot.column_offset;
  selected_work.group_index = dot.lane_group;
  selected_work.required_host_stage_ids = dot.required_host_stage_ids;
  im2p::gemmini::cycle_sim::DispatchEvents dispatch_events(
      cycle_context, &selected_work, cycle_sim::CallKind::ResidualCompact);
  im2p::cpu_functional::TimingRegistration timing_registration(dispatch_events.observer());
#endif
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
  optrace::Context trace_parent;
  optrace::Work trace_work;
  if (dot.trace_context) {
    trace_work = im2p::gemmini::production_trace::full(
        descriptor, geometry, dot.trace_layer);
    trace_work.provenance = "residual";
    trace_work.scope = "residual_compact";
    trace_work.original_block_id = dot.block_id;
    trace_work.source_row_begin = dot.source_row_begin;
    trace_work.source_row_count = dot.source_row_count;
    trace_work.stripe_id = dot.stripe_id;
    trace_work.column_begin = dot.column_offset;
    trace_work.group_index = dot.lane_group;
    try {
      trace_parent = dot.trace_context->session->parent_begin(*dot.trace_context, trace_work);
    } catch (...) {
      return RmdStatus::execution_failed;
    }
  }
#endif
  im2p_work_stats_extended_t stats{};
#if defined(GGML_GEMMINI_TESTING)
    provider_dot_attempts.fetch_add(1, std::memory_order_relaxed);
#endif
    trace::ScopedRole residual_role(GEMMINI_TRACE_ROLE_RESIDUAL);
    trace::CpuStage host_call(dot.timing_identity.interval.layer, "rmd.device_host_call");
    int provider_status = IM2P_ERROR;
    if (executor != nullptr) {
    if (fault != Im2pProviderTestFault::none)
      return RmdStatus::invalid_arguments;
    provider_status =
        scaled ? executor->execute_planned(executor->context, &descriptor,
                                           &geometry, &stats)
               : executor->execute(executor->context, &descriptor, &stats);
    }
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
    else {
    provider_status =
        fault == Im2pProviderTestFault::none
            ? (scaled ? im2p_execute_matmul_planned(sim, &descriptor, &geometry,
                                                    &stats)
                      : im2p_execute_matmul_extended(sim, &descriptor, &stats))
            : synthetic_execute(&descriptor, &stats, fault);
    }
#endif
    host_call.finish(provider_status == IM2P_OK);
#if LOG_CYCLE && !CYCLE_SIM
    try {
        using Json = nlohmann::json;
        const bool real_provider = fault == Im2pProviderTestFault::none;
        const bool counter_valid = real_provider && provider_status == IM2P_OK;
        const auto optional_id = [&](uint32_t flag, uint64_t value) -> Json {
            return dot.timing_identity.identity_mask & flag ? Json(value) : Json();
        };
        Json record = {{"schema","gemmini.cycle"},{"version",2},
            {"record_type","NPU_OPERATOR_SEGMENT"},{"op","rmd.matmul.execute"},
            {"layer",dot.timing_identity.interval.layer ? Json(dot.timing_identity.interval.layer) : Json()},
            {"command_id",gemmini_trace_reserve_ids(1)},
            {"backend",executor ? "external_executor" : "im2p_sim"},
            {"source",real_provider ? (executor ? "provider_device_cycles" : "im2p_rtl") : "synthetic_test_provider"},
            {"unit","cycle"},{"clock_domain",executor ? "external_rmd_device" : "independent_rmd_simulator"},
            {"counter_semantics","one_matmul_provider_call"},{"additive",false},
            {"run_id",optional_id(GEMMINI_CYCLE_HAS_RUN_ID,dot.timing_identity.run_id)},
            {"stripe_id",optional_id(GEMMINI_CYCLE_HAS_STRIPE_ID,dot.timing_identity.stripe_id)},
            {"slot",optional_id(GEMMINI_CYCLE_HAS_SLOT,dot.timing_identity.slot)},
            {"block_id",dot.block_id},{"lane_group",dot.lane_group},
            {"k_offset",dot.k_offset},{"column_offset",dot.column_offset},
            {"m",dot.rows},{"n",dot.columns},{"k",dot.k},{"operand_bits",dot.operand_bits},
            {"raw_cycles",stats.base.work_total_cycles},
            {"cycles",counter_valid ? Json(stats.base.work_total_cycles) : Json()},
            {"valid",counter_valid},
            {"reason",counter_valid ? Json() : Json(real_provider ? "provider_did_not_complete" : "synthetic_test_provider")},
            {"operation_success",provider_status == IM2P_OK && context.seen_count == dot.rows * dot.columns},
            {"provider_status",provider_status}};
        // Keep each observed hardware counter separately. They overlap by
        // definition and are never added to manufacture a total cycle count.
        record["counters"] = {{"compute_cycles",stats.base.compute_cycles},
            {"activation_wait_cycles",stats.base.activation_wait_cycles},
            {"weight_wait_cycles",stats.base.weight_wait_cycles},
            {"scale_wait_cycles",stats.base.scale_wait_cycles},
            {"output_wait_cycles",stats.base.output_wait_cycles},
            {"drain_cycles",stats.base.drain_cycles},
            {"activation_read_requests",stats.base.activation_read_requests},
            {"weight_read_requests",stats.base.weight_read_requests},
            {"output_write_requests",stats.base.output_write_requests},
            {"output_write_responses",stats.base.output_write_responses}};
        log::cycle.write_json(record.dump());
    } catch (...) { log::cycle.report_failure("RMD device segment"); }
#endif
  if (provider_status != IM2P_OK)
    return RmdStatus::execution_failed;
#if CYCLE_SIM
  if (cycle_context) {
    try {
      dispatch_events.complete();
      cycle_context.session->ensure_healthy();
    } catch (...) {
      return RmdStatus::execution_failed;
    }
  }
#endif
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
  if (dot.trace_context) {
    try {
      trace_parent.session->accepted(trace_parent, trace_work);
    } catch (...) {
      return RmdStatus::execution_failed;
    }
  }
#endif
  if (context.seen_count != dot.rows * dot.columns)
    return RmdStatus::invalid_packet;
  const auto status = aggregate_stats(stats, aggregate);
  if (status != RmdStatus::success)
    return status;
#if defined(GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM)
  if (trace_parent) {
    try {
      trace_parent.session->parent_end(trace_parent);
    } catch (...) {
      return RmdStatus::execution_failed;
    }
  }
#endif
  const auto publish = [&] {
    for (size_t row = 0; row < dot.rows; ++row)
      std::copy_n(staged_values.data() + row * dot.columns, dot.columns,
                  output + row * output_row_stride);
  };
#if CYCLE_SIM
  if (!im2p::gemmini::cycle_sim::publish_output(cycle_context, "im2p.residual_output_publish",
          "llama.cpp-gemmini", "ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:execute_im2p_compact_dot",
          dot.trace_layer.c_str(), output, dot.rows, dot.columns, output_row_stride, 1,
          dispatch_events.required_work(), {}, publish))
    return RmdStatus::execution_failed;
  completion.success = true;
#else
  publish();
#endif
  return RmdStatus::success;
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
