#include <gemmini/trace-context.hpp>
#include "exsia.hpp"

#include "../../../ggml-gemmini-telemetry.hpp"
#include "../../../residual/rmd/rmd-compose.hpp"
#include "types.hpp"

#include "ggml-gemmini-args.h"
#include "../../../ggml-gemmini-evaluation-observer.hpp"
#include "../../common/tensor_util.hpp"

#include <gemmini/cycle_reader.hpp>
#include <gemmini/host-timing.hpp>
#include <gemmini/log.h>
#include <gemmini/log.hpp>
#include <gemmini/performance.hpp>
#if CYCLE_SIM
#include <gemmini/cycle_sim_log.hpp>
#endif
#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
#include <gemmini/log.h>
#include "../../../../ggml-gemmini-utils/src/cycle_reader_internal.h"
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#if EXSIA_PROFILE_LOG_ENABLED
#include <fstream>
#include <mutex>
#endif
#if LOG_CYCLE
#include <sstream>
#include <string>
#endif
#include <tuple>
#include <utility>
#include <variant>

#if defined(GGML_GEMMINI_HAS_OPENMP)
#include <omp.h>
#endif

// Validation/test builds snapshot each stripe's outlier mask into state_.stripe so the
// mask-inspection tests keep working. Production builds leave it 0 and keep no per-stripe
// mask array (the single workspace owns the only live mask).
#ifndef EXSIA_VALIDATION
#define EXSIA_VALIDATION 0
#endif

#if EXSIA_PROFILE_COLLECTION_ENABLED
#define EXSIA_PROFILE_COLLECT(...) __VA_ARGS__
#else
#define EXSIA_PROFILE_COLLECT(...)
#endif

#if EXSIA_PROFILE_LOG_ENABLED
#define EXSIA_PROFILE_LOG(...) __VA_ARGS__
#else
#define EXSIA_PROFILE_LOG(...)
#endif

#if EXSIA_BRANCH_COUNTS_ENABLED
#define EXSIA_STATS_PARAMETER , StripeCycleStats & stats
#define EXSIA_STATS_ARGUMENT(stats) , stats
#else
#define EXSIA_STATS_PARAMETER
#define EXSIA_STATS_ARGUMENT(stats)
#endif

namespace ggml::gemmini::quants::act::exsia {
using namespace detail;

namespace {
template <typename T> void release_vector(std::vector<T> & values) {
    std::vector<T>().swap(values);
}

bool checked_mul_size(size_t lhs, size_t rhs, size_t & out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        return false;

    out = lhs * rhs;
    return true;
}

bool checked_add_size(size_t lhs, size_t rhs, size_t & out) {
    if (lhs > std::numeric_limits<size_t>::max() - rhs)
        return false;

    out = lhs + rhs;
    return true;
}

bool checked_round_up_multiple(size_t value, size_t multiple, size_t & out) {
    if (multiple == 0)
        return false;

    size_t adjusted = 0;
    if (!checked_add_size(value, multiple - 1, adjusted))
        return false;

    out = (adjusted / multiple) * multiple;
    return true;
}

} // namespace

uint64_t next_exsia_run_id() {
    static std::atomic<uint64_t> next{0};
    return next.fetch_add(1, std::memory_order_relaxed);
}

std::array<ExSIAState::ExecutionModeAvailability, 3> execution_mode_availability() {
    using Mode = ExSIAState::ExecutionMode;

#if defined(GGML_GEMMINI_HAS_OPENMP)
    constexpr const char * local_parallel_reason = "fixed four-task OpenMP Local stage";
    constexpr const char * pipeline_reason       = "two-slot OpenMP Local/Folding pipeline";
#else
    constexpr const char * local_parallel_reason = "OpenMP unavailable";
    constexpr const char * pipeline_reason       = "OpenMP unavailable";
#endif

    return {{
        {Mode::Sequential, "Sequential", true, "RUNNABLE", "default execution mode"},
#if defined(GGML_GEMMINI_HAS_OPENMP)
        {Mode::LocalParallel, "LocalParallel", true, "RUNNABLE", local_parallel_reason},
#else
        {Mode::LocalParallel, "LocalParallel", false, "BLOCKED", local_parallel_reason},
#endif
#if defined(GGML_GEMMINI_HAS_OPENMP)
        {Mode::LocalFoldingPipeline, "LocalFoldingPipeline", true, "RUNNABLE", pipeline_reason},
#else
        {Mode::LocalFoldingPipeline, "LocalFoldingPipeline", false, "BLOCKED", pipeline_reason},
#endif
    }};
}

namespace {
bool assemble_stripe_mask(StripePipelineSlot & slot, const ExSIAState & state) {
    StripeState & stripe             = slot.stripe;
    const size_t  active_block_count = stripe.row_count() * state.blocks_per_row;
    if (stripe.outlier_mask.rows != stripe.row_count() ||
        stripe.outlier_mask.cols != state.K_padded || slot.active_block_count != active_block_count)
        return false;

    stripe.outlier_mask.clear_active_bits();
    for (size_t block = 0; block < active_block_count; ++block) {
        const size_t    local_row  = block / state.blocks_per_row;
        const size_t    blk_idx    = block % state.blocks_per_row;
        const BlockMask block_mask = slot.block_mask(block, state.B_size);
        for (size_t i = 0; i < block_mask.bit_count; ++i) {
            const size_t global_col = blk_idx * state.B_size + i;
            if (global_col >= state.K_logical || !block_mask.is_set(i))
                continue;

            const size_t mask_idx = local_row * stripe.outlier_mask.cols + global_col;
            stripe.outlier_mask.words[mask_idx / 64] |= uint64_t{1} << (mask_idx % 64);
        }
    }

    return true;
}

void reduce_stripe_exponents(StripePipelineSlot & slot, size_t active_block_count) {
    StripeState & stripe = slot.stripe;
    stripe.e1            = std::numeric_limits<int16_t>::min();
    stripe.e2            = std::numeric_limits<int16_t>::min();

    GGML_ASSERT(active_block_count <= slot.block_exp.size());
    ExpScanner reducer;
    for (size_t block = 0; block < active_block_count; ++block)
        reducer.update_stripe_top2_exp(stripe, slot.block_exp[block]);
}
} // namespace

// Seals the stripe's RMD packet and publishes the shared handle. Called once per
// stripe, right after folding commits, by the thread that ran folding.
static bool
seal_stripe_packet(Meta & meta, StripePipelineSlot & slot, const ggml_gemmini_args_t & args) {
#if GGML_GEMMINI_ENABLE_RMD
    const residual::ResidualStripePayload payload = slot.rmd_builder.finish();
    slot.rmd_packet                               = payload.packet;
    slot.direct_residual                          = payload.direct;
    slot.rmd_pack_ns                              = payload.capture_ns;
    if (slot.rmd_builder.status() != rmd::RmdStatus::success)
        return false;
    if (slot.rmd_packet)
        meta.rmd_packets.push_back(slot.rmd_packet);
    if (slot.direct_residual)
        meta.direct_residuals.push_back(slot.direct_residual);
#else
    (void)meta;
    slot.rmd_packet.reset();
    slot.direct_residual.reset();
    slot.rmd_pack_ns = 0;
#endif
#if GGML_GEMMINI_RESIDUAL_METRICS
    if (args.evaluation_context && args.evaluation_context->residual_enabled()) {
        if (!GGML_GEMMINI_ENABLE_RMD || args.residual_route != residual::ResidualRoute::ws_packet)
            throw std::runtime_error("evaluation metrics: RES requires producer RMD packet route");
        args.evaluation_context->radix_stripe(
            slot.stripe_idx, slot.rmd_packet ? slot.rmd_packet->required_planes : 0);
    }
#else
    (void)args;
#endif
    return true;
}

const char * failure_code_name(ExSIAState::FailureCode code) noexcept {
    switch (code) {
    case ExSIAState::FailureCode::None:
        return "None";
    case ExSIAState::FailureCode::InvalidInput:
        return "InvalidInput";
    case ExSIAState::FailureCode::OpenMPUnavailable:
        return "OpenMPUnavailable";
    case ExSIAState::FailureCode::WrongTeamSize:
        return "WrongTeamSize";
    case ExSIAState::FailureCode::ExternalOpenMPRegionUnsupported:
        return "ExternalOpenMPRegionUnsupported";
    case ExSIAState::FailureCode::LocalBlockFailure:
        return "LocalBlockFailure";
    case ExSIAState::FailureCode::MaskAssemblyFailure:
        return "MaskAssemblyFailure";
    case ExSIAState::FailureCode::ExponentReductionFailure:
        return "ExponentReductionFailure";
    case ExSIAState::FailureCode::FoldingFailure:
        return "FoldingFailure";
    case ExSIAState::FailureCode::ValidationSnapshotFailure:
        return "ValidationSnapshotFailure";
    case ExSIAState::FailureCode::StripeReadySinkFailure:
        return "StripeReadySinkFailure";
    case ExSIAState::FailureCode::ProfileIntervalInvalid:
        return "ProfileIntervalInvalid";
    case ExSIAState::FailureCode::ProfileFlushFailure:
        return "ProfileFlushFailure";
    case ExSIAState::FailureCode::Exception:
        return "Exception";
    }
    return "Unknown";
}

const char * failure_origin_name(ExSIAState::FailureCode code) noexcept {
    switch (code) {
    case ExSIAState::FailureCode::None:
        return "none";
    case ExSIAState::FailureCode::InvalidInput:
    case ExSIAState::FailureCode::OpenMPUnavailable:
    case ExSIAState::FailureCode::WrongTeamSize:
    case ExSIAState::FailureCode::ExternalOpenMPRegionUnsupported:
        return "exsia_setup";
    case ExSIAState::FailureCode::LocalBlockFailure:
    case ExSIAState::FailureCode::MaskAssemblyFailure:
    case ExSIAState::FailureCode::ExponentReductionFailure:
    case ExSIAState::FailureCode::FoldingFailure:
        return "exsia_compute";
    case ExSIAState::FailureCode::ValidationSnapshotFailure:
    case ExSIAState::FailureCode::ProfileIntervalInvalid:
    case ExSIAState::FailureCode::ProfileFlushFailure:
        return "exsia_validation";
    case ExSIAState::FailureCode::StripeReadySinkFailure:
        return "downstream_sink";
    case ExSIAState::FailureCode::Exception:
        return "exception";
    }
    return "unknown";
}

void ExSIA::reset_failure_state() {
    first_failure_code_.store(ExSIAState::FailureCode::None, std::memory_order_relaxed);
    first_failure_stripe_.store(ExSIAState::no_failure_stripe, std::memory_order_relaxed);
}

void ExSIA::record_failure(ExSIAState::FailureCode code, size_t stripe) {
    if (code == ExSIAState::FailureCode::None)
        return;

    ExSIAState::FailureCode expected = ExSIAState::FailureCode::None;
    if (first_failure_code_.compare_exchange_strong(
            expected, code, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        first_failure_stripe_.store(stripe, std::memory_order_release);
    }
}

bool ExSIA::run(Meta & meta, const ggml_tensor * A, ggml_gemmini_args_t & args) {
    return run(meta, A, args, nullptr);
}

bool ExSIA::run(Meta &                  meta,
                const ggml_tensor *     A,
                ggml_gemmini_args_t &   args,
                const StripeReadySink * sink) {
    const char *                layer             = args.matmul_layer.c_str();
    [[maybe_unused]] const auto task_trace_origin = gemmini_trace_capture();
    const uint64_t              run_id            = next_exsia_run_id();
    meta.run_id                                   = run_id;
#if CYCLE_SIM
    const auto record_producer = [&](cycle_sim::ProducerEventKind kind,
                                     const StripePipelineSlot &   slot,
                                     const char *                 source_location) {
        if (sink == nullptr || sink->on_ready == nullptr || !args.cycle_sim_context)
            return true;
        return args.cycle_sim_context.session->producer_event(
            args.cycle_sim_context,
            {kind,
             run_id,
             slot.stripe_idx,
             slot.row_start,
             slot.row_end,
             slot.stripe_idx % EXSIA_PIPELINE_SLOT_COUNT,
             bool(slot.rmd_packet),
             bool(slot.direct_residual),
             source_location});
    };
#endif
    const char * force_recompute = std::getenv("GGML_GEMMINI_EXSIA_FORCE_RECOMPUTE");
    local_.set_force_recompute(force_recompute != nullptr &&
                               std::strcmp(force_recompute, "1") == 0);
#if LOG_CYCLE
    ExsiaRunTiming run_timing{layer, run_id};
#endif
    CpuWallInterval run_cpu_wall(layer, run_id, "exsia.run.cpu");
    EXSIA_PROFILE_LOG(run_cpu_wall.pause();
                      const ProfileConfig profile_config = compile_profile_config();
                      run_cpu_wall.resume();)
    EXSIA_PROFILE_COLLECT(ProfileInterval run_profile; start_profile_interval(run_profile);)
    EXSIA_PROFILE_LOG(const char * mode =
                          requested_mode_ == ExSIAState::ExecutionMode::Sequential ? "Sequential"
                          : requested_mode_ == ExSIAState::ExecutionMode::LocalParallel
                              ? "LocalParallel"
                              : "LocalFoldingPipeline";)
    const int16_t invalid_theta = std::numeric_limits<int16_t>::min();
    // Global outputs are args.A (dense quantized), state_.residual, meta.theta, meta.rmd_packets.
    size_t     logical_elem_count    = 0;
    const bool logical_elem_count_ok = checked_mul_size(args.I, args.K, logical_elem_count);
    reset_failure_state();
    ggml::gemmini::GemminiGeometry geometry;
    if (!args.activation_quant_geometry_matches(geometry)) {
        record_failure(ExSIAState::FailureCode::InvalidInput, ExSIAState::no_failure_stripe);
        for (StripePipelineSlot & slot : pipeline_slots_)
            slot.reset_for_run();
        local_workspace_.reset_for_run();
        meta.reset();
        meta.rho              = config::GGML_GEMMINI_ACTIVATION_RHO;
        state_                = ExSIAState{};
        state_.mode           = requested_mode_;
        state_.run_id         = run_id;
        state_.failure_code   = ExSIAState::FailureCode::InvalidInput;
        state_.failure_stripe = ExSIAState::no_failure_stripe;
        return false;
    }
    const auto fail = [&](ExSIAState::FailureCode code   = ExSIAState::FailureCode::Exception,
                          size_t                  stripe = ExSIAState::no_failure_stripe) {
        record_failure(code, stripe);
        const ExSIAState::FailureCode failure_code =
            first_failure_code_.load(std::memory_order_acquire);
        const size_t failure_stripe = first_failure_stripe_.load(std::memory_order_acquire);
        if (args.A.valid() && logical_elem_count_ok)
            args.A.zero_fill();

        for (StripePipelineSlot & slot : pipeline_slots_)
            slot.reset_for_run();
        local_workspace_.reset_for_run();
        meta.reset();
        meta.rho = config::GGML_GEMMINI_ACTIVATION_RHO;
#if EXSIA_VALIDATION && EXSIA_PROFILE_COLLECTION_ENABLED
        const ExSIAState::ProfileSnapshot profile_snapshot = state_.profile_snapshot;
#endif
        state_                = ExSIAState{};
        state_.mode           = requested_mode_;
        state_.run_id         = run_id;
        state_.failure_code   = failure_code;
        state_.failure_stripe = failure_stripe;
#if EXSIA_VALIDATION && EXSIA_PROFILE_COLLECTION_ENABLED
        state_.profile_snapshot = profile_snapshot;
#endif
        return false;
    };
    EXSIA_PROFILE_LOG(
        if (!profile_config.setup_ok) return fail(ExSIAState::FailureCode::ProfileFlushFailure);)

    meta.rho   = config::GGML_GEMMINI_ACTIVATION_RHO;
    meta.sigma = GGML_GEMMINI_EXSIA_SIGMA;
    for (StripePipelineSlot & slot : pipeline_slots_)
        slot.reset_for_run();
    local_workspace_.reset_for_run();
    state_        = ExSIAState{};
    state_.mode   = requested_mode_;
    state_.run_id = run_id;
#if !defined(GGML_GEMMINI_HAS_OPENMP)
    if (state_.mode == ExSIAState::ExecutionMode::LocalParallel ||
        state_.mode == ExSIAState::ExecutionMode::LocalFoldingPipeline)
        return fail(ExSIAState::FailureCode::OpenMPUnavailable);
#endif
    if (state_.mode != ExSIAState::ExecutionMode::Sequential &&
        state_.mode != ExSIAState::ExecutionMode::LocalParallel &&
        state_.mode != ExSIAState::ExecutionMode::LocalFoldingPipeline) {
        return fail(ExSIAState::FailureCode::InvalidInput);
    }
#if defined(GGML_GEMMINI_HAS_OPENMP)
    if ((state_.mode == ExSIAState::ExecutionMode::LocalParallel ||
         state_.mode == ExSIAState::ExecutionMode::LocalFoldingPipeline) &&
        (omp_in_parallel() || omp_get_active_level() > 0)) {
        return fail(ExSIAState::FailureCode::ExternalOpenMPRegionUnsupported);
    }
#endif
    state_.B_size    = BLOCK_SIZE;
    state_.K_logical = args.K;
    if (!args.A.valid() || args.I == 0 || args.K == 0 || !logical_elem_count_ok ||
        !checked_round_up_multiple(args.K, state_.B_size, state_.K_padded)) {
        return fail(ExSIAState::FailureCode::InvalidInput);
    }

    state_.blocks_per_row = state_.K_padded / state_.B_size;

    const size_t  rows_per_stripe = geometry.stripe_rows;
    const size_t  num_stripes     = geometry.stripe_count;
    const float * src_data        = ggml::gemmini::activation_data(A);
    if (!src_data)
        return fail(ExSIAState::FailureCode::InvalidInput);
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS || GGML_GEMMINI_SCALE_METRICS
    args.evaluation_context.reset();
    if (const auto session = evaluation::active_session())
        args.evaluation_context = session->invocation(args.matmul_layer, args.I, args.K, src_data);
#endif

    size_t max_stripe_rows        = std::min(args.I, rows_per_stripe);
    size_t max_stripe_elem_count  = 0;
    size_t max_stripe_block_count = 0;
    if (!checked_mul_size(max_stripe_rows, state_.K_padded, max_stripe_elem_count) ||
        !checked_mul_size(max_stripe_rows, state_.blocks_per_row, max_stripe_block_count)) {
        return fail(ExSIAState::FailureCode::InvalidInput);
    }

    meta.theta.assign(num_stripes, invalid_theta);
    meta.rmd_packets.clear();
    meta.direct_residuals.clear();

    // Dense residual matrix is a global output, sized to the full padded activation.
    // Each stripe writes its own disjoint global row range (K_padded stride).
    size_t padded_elem_count = 0;
    if (!checked_mul_size(args.I, state_.K_padded, padded_elem_count))
        return fail(ExSIAState::FailureCode::InvalidInput);

    release_vector(state_.x_f32);
    release_vector(state_.q_wide);
    release_vector(state_.block_exp);
    state_.residual.assign(padded_elem_count, 0);

    for (StripePipelineSlot & slot : pipeline_slots_) {
        slot.rmd_builder.select(args.residual_route);
        slot.rmd_builder.set_context(run_id, layer);
        if (!slot.prepare(max_stripe_elem_count,
                          max_stripe_block_count,
                          max_stripe_rows,
                          state_.K_padded,
                          state_.B_size))
            return fail(ExSIAState::FailureCode::InvalidInput);
    }
    if (!local_workspace_.prepare(max_stripe_block_count, state_.B_size))
        return fail(ExSIAState::FailureCode::InvalidInput);

#if GGML_GEMMINI_ENABLE_RMD
    const unsigned capture_kind = args.residual_route == residual::ResidualRoute::cpu_direct ? 0
                                  : GGML_GEMMINI_ACTIVATION_BITS == 16                       ? 1
                                                                                             : 2;
    static std::atomic<unsigned> reported_capture_kinds{0};
    if ((reported_capture_kinds.fetch_or(1u << capture_kind, std::memory_order_relaxed) &
         (1u << capture_kind)) == 0) {
        const char * names[] = {"cpu_direct (compaction unused)",
                                "ws_packet compaction=a16",
                                "ws_packet compaction=bitmap"};
        std::fprintf(stderr, "gemmini: ExSIA RMD capture=%s\n", names[capture_kind]);
    }
#endif

    // state_.stripe carries per-stripe row metadata only; the workspace owns the live
    // mask/scratch. (Validation builds additionally snapshot each mask below.)
    state_.stripe.assign(num_stripes, StripeState{});
#if EXSIA_OBSERVATION_ENABLED
    if (state_.mode == ExSIAState::ExecutionMode::LocalParallel ||
        state_.mode == ExSIAState::ExecutionMode::LocalFoldingPipeline)
        state_.local_parallel_observations.resize(num_stripes);
#endif
    for (size_t s = 0; s < num_stripes; ++s) {
        StripeState & meta_stripe = state_.stripe[s];
        meta_stripe.row_start     = s * rows_per_stripe;
        meta_stripe.row_end       = std::min((s + 1) * rows_per_stripe, args.I);
#if EXSIA_VALIDATION
        if (!meta_stripe.outlier_mask.prepare(meta_stripe.row_count(), state_.K_padded))
            return fail(ExSIAState::FailureCode::ValidationSnapshotFailure, s);
#endif
    }

    const auto snapshot_validation_mask = [&](size_t stripe_idx, const BitMask & mask) {
#if EXSIA_VALIDATION
        BitMask &    snapshot   = state_.stripe[stripe_idx].outlier_mask;
        const size_t word_count = mask.active_word_count();
        if (snapshot.words.size() < word_count)
            return false;
        snapshot.rows = mask.rows;
        snapshot.cols = mask.cols;
        std::copy_n(mask.words.begin(), word_count, snapshot.words.begin());
#else
        (void)stripe_idx;
        (void)mask;
#endif
        return true;
    };
    const auto notify_stripe_ready = [&](StripePipelineSlot & slot,
                                         uint64_t             run_id,
                                         bool                 measure_stripe_ready_handoff,
                                         CpuWallInterval &    cpu_wall
#if EXSIA_PROFILE_COLLECTION_ENABLED
                                         ,
                                         StripeProfileRecord * profile
#endif
                                     ) {
        (void)measure_stripe_ready_handoff;
#if EXSIA_STAGE_PROFILE_ENABLED
        profile->selected_positions = slot.stripe.selected_positions;
        profile->residual_nnz       = slot.stripe.residual_nnz;
#endif
        if (sink == nullptr || sink->on_ready == nullptr)
            return true;

        StripeReadyEvent event{};
        event.run_id        = run_id;
        event.stripe_id     = slot.stripe_idx;
        event.slot          = slot.stripe_idx % EXSIA_PIPELINE_SLOT_COUNT;
        event.row_begin     = slot.row_start;
        event.row_end       = slot.row_end;
        const int16_t theta = meta.resolve_stripe_theta(static_cast<int>(slot.stripe_idx));
        if (theta == std::numeric_limits<int16_t>::min()) {
            ggml::gemmini::log::debug(
                layer,
                "[exsia] stripe ready handoff failed run_id=%llu stripe=%zu reason=missing_theta",
                static_cast<unsigned long long>(run_id),
                slot.stripe_idx);
            return false;
        }
        event.activation_metadata   = StripeMetadataSnapshot{meta.e_s, meta.rho, meta.sigma, theta};
        event.quantization_start    = slot.quantization_start;
        event.quantization_end      = slot.quantization_end;
        event.quantization_start_ns = slot.quantization_start_ns;
        event.quantization_end_ns   = slot.quantization_end_ns;
        event.rmd_packet            = slot.rmd_packet;
        event.direct_residual       = slot.direct_residual;
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS || GGML_GEMMINI_SCALE_METRICS
        event.evaluation_context = args.evaluation_context;
#endif
        event.rmd_pack_ns = slot.rmd_pack_ns;
#if CYCLE_SIM
        if (profile)
            event.cycle_sim_host_dependencies = profile_host_stage_ids(*profile);
#endif
#if EXSIA_PROFILE_COLLECTION_ENABLED
        if (profile != nullptr) {
            event.local_start_ns              = profile->local.start_ns;
            event.local_end_ns                = profile->local.end_ns;
            event.folding_start_ns            = profile->folding.start_ns;
            event.folding_end_ns              = profile->folding.end_ns;
            event.mask_assembly_start_ns      = profile->mask_assembly.start_ns;
            event.mask_assembly_end_ns        = profile->mask_assembly.end_ns;
            event.exponent_reduction_start_ns = profile->exponent_reduction.start_ns;
            event.exponent_reduction_end_ns   = profile->exponent_reduction.end_ns;
        }
#endif
        event.folding_commit_ns = slot.folding_commit_ns;
        cpu_wall.pause();
#if LOG_CYCLE
        event.collect_submission_timing = true;
        const auto submission_start     = gemmini_cpu_timing_read();
#endif
#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
        ggml::gemmini::cycle::NativeCycleSample stripe_ready_handoff_start{};
        if (measure_stripe_ready_handoff)
            stripe_ready_handoff_start = ggml::gemmini::cycle::read_sample();
#endif
        const bool accepted = sink->on_ready(sink->user_data, event);
        if (!accepted) {
            ggml::gemmini::log::debug(
                layer,
                "[exsia] stripe ready handoff failed run_id=%llu stripe=%zu reason=sink_rejected",
                static_cast<unsigned long long>(run_id),
                slot.stripe_idx);
        }
#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
        if (measure_stripe_ready_handoff) {
            const auto stripe_ready_handoff_end = ggml::gemmini::cycle::read_sample();
#if LOG_CYCLE
            run_timing.submission(event, submission_start, gemmini_cpu_timing_read(), accepted);
#endif
            const gemmini_native_cycle_sample_internal stripe_ready_handoff_start_sample{
                stripe_ready_handoff_start.value,
                static_cast<uint8_t>(stripe_ready_handoff_start.valid),
                static_cast<uint8_t>(stripe_ready_handoff_start.reason),
                GEMMINI_NATIVE_CYCLE_SOURCE_LINUX_PERF_CPU_CYCLES,
                stripe_ready_handoff_start.owner_event_token,
                stripe_ready_handoff_start.generation};
            const gemmini_native_cycle_sample_internal stripe_ready_handoff_end_sample{
                stripe_ready_handoff_end.value,
                static_cast<uint8_t>(stripe_ready_handoff_end.valid),
                static_cast<uint8_t>(stripe_ready_handoff_end.reason),
                GEMMINI_NATIVE_CYCLE_SOURCE_LINUX_PERF_CPU_CYCLES,
                stripe_ready_handoff_end.owner_event_token,
                stripe_ready_handoff_end.generation};
            const gemmini_cycle_record_v2 stripe_ready_handoff_record{
                {layer,
                 "exsia.stripe_ready_handoff",
                 stripe_ready_handoff_start.value,
                 stripe_ready_handoff_end.value,
                 nullptr,
                 0,
                 nullptr},
                GEMMINI_CYCLE_HAS_RUN_ID | GEMMINI_CYCLE_HAS_STRIPE_ID | GEMMINI_CYCLE_HAS_SLOT,
                event.run_id,
                event.stripe_id,
                event.slot,
                0,
                0};
            gemmini_log_cycle_record_v2_checked_internal(&stripe_ready_handoff_record,
                                                         &stripe_ready_handoff_start_sample,
                                                         &stripe_ready_handoff_end_sample,
                                                         1);
        }
#endif
#if LOG_CYCLE
#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
        if (!measure_stripe_ready_handoff)
#endif
            run_timing.submission(event, submission_start, gemmini_cpu_timing_read(), accepted);
#endif
        cpu_wall.resume();
        return accepted;
    };
#if defined(GGML_GEMMINI_HAS_OPENMP)
#if EXSIA_BRANCH_COUNTS_ENABLED
    const auto record_sample = [](StripeCycleStats & stats, const LocalBlockCycleSample & sample) {
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
        stats.p0.add(sample.stage_intervals[0]);
        stats.p1.add(sample.stage_intervals[1]);
        stats.p2.add(sample.stage_intervals[2]);
        stats.p3.add(sample.stage_intervals[3]);
#else
        stats.p0.add(sample.p0);
        stats.p1.add(sample.p1);
        stats.p2.add(sample.p2);
        stats.p3.add(sample.p3);
#endif
        stats.forced_recompute_count += sample.forced_recompute_count;
#endif
        switch (sample.p3_path) {
        case P3Path::BypassNoIntegerOutlier:
            ++stats.p3_bypass_no_int_count;
            break;
        case P3Path::BypassSameScale:
            ++stats.p3_bypass_same_scale_count;
            break;
        case P3Path::Replay:
            ++stats.p3_replay_count;
            break;
        }
    };
#endif
    const auto run_local_block = [&](StripePipelineSlot & slot,
                                     StripeScratch &      scratch,
                                     size_t               row,
                                     size_t block         EXSIA_STATS_PARAMETER) {
#if EXSIA_BRANCH_COUNTS_ENABLED
        LocalBlockCycleSample sample;
#endif
        const size_t col_offset = block * state_.B_size;
        GGML_ASSERT(col_offset < args.K);
        const size_t valid_count   = std::min(state_.B_size, args.K - col_offset);
        const size_t local_row     = slot.stripe.local_row(row);
        const size_t block_base    = local_row * state_.K_padded + col_offset;
        const size_t block_exp_idx = local_row * state_.blocks_per_row + block;
        GGML_ASSERT(slot.q_wide.size() >= block_base + state_.B_size);
        GGML_ASSERT(block_exp_idx < slot.block_exp.size());
        BlockMask block_mask =
            slot.block_mask(local_row * state_.blocks_per_row + block, state_.B_size);
        if (!local_.run_optimized(meta,
                                  state_,
                                  src_data + row * args.K + col_offset,
                                  valid_count,
                                  state_.B_size,
                                  local_row,
                                  block,
                                  scratch,
                                  block_mask,
                                  slot.q_wide.data() + block_base,
                                  slot.block_exp[block_exp_idx]
#if EXSIA_BRANCH_COUNTS_ENABLED
                                  ,
                                  sample
#endif
                                  ))
            return false;

#if EXSIA_BRANCH_COUNTS_ENABLED
        record_sample(stats, sample);
#endif
#if GGML_GEMMINI_ACT_QUANT_METRICS
        if (scratch.actual_requantized && args.evaluation_context)
            args.evaluation_context->requantized(row, block);
#endif
        return true;
    };
#endif

    EXSIA_PROFILE_COLLECT(std::vector<StripeProfileRecord> stripe_profiles(num_stripes);)
    if (state_.mode == ExSIAState::ExecutionMode::LocalFoldingPipeline) {
#if defined(GGML_GEMMINI_HAS_OPENMP)
        std::vector<uint8_t>       prepared_storage(num_stripes);
        std::vector<uint8_t>       worker_done_storage(num_stripes * EXSIA_LOCAL_WORKER_COUNT);
        std::vector<uint8_t>       local_sealed_storage(num_stripes);
        std::vector<uint8_t>       slot_released_storage(num_stripes);
        [[maybe_unused]] uint8_t * prepared      = prepared_storage.data();
        [[maybe_unused]] uint8_t * worker_done   = worker_done_storage.data();
        [[maybe_unused]] uint8_t * local_sealed  = local_sealed_storage.data();
        [[maybe_unused]] uint8_t * slot_released = slot_released_storage.data();
        [[maybe_unused]] uint8_t   post_chain    = 0;
        std::atomic<bool>          pipeline_ok{true};
        run_cpu_wall.pause();
#if LOG_CYCLE
        // Task bodies are measured below; OpenMP scheduling outside them is not.
        performance::incomplete_cpu_wall("exsia_task_scheduler_cpu_wall_unmeasured");
#endif
#pragma omp parallel num_threads(EXSIA_OMP_THREAD_COUNT)
        {
            trace::ScopedContext team_context(task_trace_origin, true);
            trace::CpuStage      team_lifetime(
                layer, "task.host_work", trace::CpuStage::Scope::envelope);
#if LOG_CYCLE
            const bool collect_worker_cpu = cycle::host_thread_id() != run_timing.start.tid;
            const auto worker_start =
                collect_worker_cpu ? gemmini_cpu_timing_read() : gemmini_cpu_sample{};
#endif
#pragma omp single
            {
                const size_t observed_team_size = static_cast<size_t>(omp_get_num_threads());
                if (observed_team_size != EXSIA_OMP_THREAD_COUNT) {
                    record_failure(ExSIAState::FailureCode::WrongTeamSize);
                    pipeline_ok.store(false, std::memory_order_relaxed);
                }
#pragma omp taskgroup
                {
                    for (size_t s = 0; s < num_stripes; ++s) {
                        const size_t slot_idx  = s % EXSIA_PIPELINE_SLOT_COUNT;
                        const size_t row_start = s * rows_per_stripe;
                        const size_t row_end   = std::min((s + 1) * rows_per_stripe, args.I);
                        if (s == 0) {
#pragma omp task depend(out : prepared[s])                                                         \
    firstprivate(s, slot_idx, row_start, row_end, observed_team_size)
                            {
                                trace::ScopedContext task_context(task_trace_origin, true);
                                trace::CpuStage      task_lifetime(
                                    layer, "task.host_work", trace::CpuStage::Scope::envelope);
                                CpuWallInterval task_cpu_wall(layer, run_id, "exsia.prepare", s);
                                try {
                                    if (pipeline_ok.load(std::memory_order_relaxed)) {
                                        StripePipelineSlot & slot = pipeline_slots_[slot_idx];
                                        slot.acquire(s);
                                        slot.reset_for_stripe(s,
                                                              row_start,
                                                              row_end,
                                                              state_.K_padded,
                                                              state_.blocks_per_row);
#if CYCLE_SIM
                                        (void)record_producer(
                                            cycle_sim::ProducerEventKind::ExsiaWorkspaceAcquire,
                                            slot,
                                            "ggml/src/ggml-gemmini/quants/act/exsia/"
                                            "exsia.cpp:prepare_slot_first");
#endif
                                        local_workspace_.reset_for_stripe(
                                            s, row_start, row_end, state_.blocks_per_row);
                                        slot.mark_quantization_started(0, aggregate_now_ns());
#if EXSIA_OBSERVATION_ENABLED
                                        LocalParallelStripeObservation & observation =
                                            state_.local_parallel_observations[s];
                                        observation            = LocalParallelStripeObservation{};
                                        observation.stripe_idx = s;
                                        observation.observed_team_size   = observed_team_size;
                                        observation.scheduled_task_count = EXSIA_LOCAL_WORKER_COUNT;
                                        const size_t total_blocks =
                                            slot.stripe.row_count() * state_.blocks_per_row;
                                        const size_t expected_blocks_per_task =
                                            total_blocks / EXSIA_LOCAL_WORKER_COUNT +
                                            (total_blocks % EXSIA_LOCAL_WORKER_COUNT != 0 ? 1 : 0);
                                        for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT;
                                             ++task_id) {
                                            const LocalWorkerContext & worker =
                                                local_workspace_.workers[task_id];
                                            LocalParallelTaskRecord & record =
                                                observation.tasks[task_id];
                                            record.task_id     = task_id;
                                            record.row_start   = worker.row_start;
                                            record.row_end     = worker.row_end;
                                            record.block_start = worker.block_start;
                                            record.block_end   = worker.block_end;
                                            record.populated_block_count =
                                                worker.block_end - worker.block_start;
                                            record.empty = record.populated_block_count == 0;
                                            record.short_task =
                                                !record.empty && record.populated_block_count <
                                                                     expected_blocks_per_task;
                                        }
#endif
                                        EXSIA_PROFILE_COLLECT(
                                            StripeProfileRecord & profile = stripe_profiles[s];
                                            profile                       = StripeProfileRecord{};
                                            profile.stripe_idx            = s;
                                            profile.row_start             = row_start;
                                            profile.row_end               = row_end;
                                            profile.team_size             = observed_team_size;
                                            start_profile_interval(profile.stripe_total);
                                            start_profile_interval(profile.local);)
                                    }
                                } catch (...) {
                                    record_failure(ExSIAState::FailureCode::Exception, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
                            }
                        } else if (s == 1) {
#pragma omp task depend(in : local_sealed[s - 1]) depend(out : prepared[s])                        \
    firstprivate(s, slot_idx, row_start, row_end, observed_team_size)
                            {
                                trace::ScopedContext task_context(task_trace_origin, true);
                                trace::CpuStage      task_lifetime(
                                    layer, "task.host_work", trace::CpuStage::Scope::envelope);
                                CpuWallInterval task_cpu_wall(layer, run_id, "exsia.prepare", s);
                                try {
                                    if (pipeline_ok.load(std::memory_order_relaxed)) {
                                        StripePipelineSlot & slot = pipeline_slots_[slot_idx];
                                        slot.acquire(s);
                                        slot.reset_for_stripe(s,
                                                              row_start,
                                                              row_end,
                                                              state_.K_padded,
                                                              state_.blocks_per_row);
#if CYCLE_SIM
                                        (void)record_producer(
                                            cycle_sim::ProducerEventKind::ExsiaWorkspaceAcquire,
                                            slot,
                                            "ggml/src/ggml-gemmini/quants/act/exsia/"
                                            "exsia.cpp:prepare_slot_next");
#endif
                                        local_workspace_.reset_for_stripe(
                                            s, row_start, row_end, state_.blocks_per_row);
                                        slot.mark_quantization_started(0, aggregate_now_ns());
#if EXSIA_OBSERVATION_ENABLED
                                        LocalParallelStripeObservation & observation =
                                            state_.local_parallel_observations[s];
                                        observation            = LocalParallelStripeObservation{};
                                        observation.stripe_idx = s;
                                        observation.observed_team_size   = observed_team_size;
                                        observation.scheduled_task_count = EXSIA_LOCAL_WORKER_COUNT;
                                        const size_t total_blocks =
                                            slot.stripe.row_count() * state_.blocks_per_row;
                                        const size_t expected_blocks_per_task =
                                            total_blocks / EXSIA_LOCAL_WORKER_COUNT +
                                            (total_blocks % EXSIA_LOCAL_WORKER_COUNT != 0 ? 1 : 0);
                                        for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT;
                                             ++task_id) {
                                            const LocalWorkerContext & worker =
                                                local_workspace_.workers[task_id];
                                            LocalParallelTaskRecord & record =
                                                observation.tasks[task_id];
                                            record.task_id     = task_id;
                                            record.row_start   = worker.row_start;
                                            record.row_end     = worker.row_end;
                                            record.block_start = worker.block_start;
                                            record.block_end   = worker.block_end;
                                            record.populated_block_count =
                                                worker.block_end - worker.block_start;
                                            record.empty = record.populated_block_count == 0;
                                            record.short_task =
                                                !record.empty && record.populated_block_count <
                                                                     expected_blocks_per_task;
                                        }
#endif
                                        EXSIA_PROFILE_COLLECT(
                                            StripeProfileRecord & profile = stripe_profiles[s];
                                            profile                       = StripeProfileRecord{};
                                            profile.stripe_idx            = s;
                                            profile.row_start             = row_start;
                                            profile.row_end               = row_end;
                                            profile.team_size             = observed_team_size;
                                            start_profile_interval(profile.stripe_total);
                                            start_profile_interval(profile.local);)
                                    }
                                } catch (...) {
                                    record_failure(ExSIAState::FailureCode::Exception, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
                            }
                        } else {
#pragma omp task depend(in : local_sealed[s - 1], slot_released[s - 2]) depend(out : prepared[s])  \
    firstprivate(s, slot_idx, row_start, row_end, observed_team_size)
                            {
                                trace::ScopedContext task_context(task_trace_origin, true);
                                trace::CpuStage      task_lifetime(
                                    layer, "task.host_work", trace::CpuStage::Scope::envelope);
                                CpuWallInterval task_cpu_wall(layer, run_id, "exsia.prepare", s);
                                try {
                                    if (pipeline_ok.load(std::memory_order_relaxed)) {
                                        StripePipelineSlot & slot = pipeline_slots_[slot_idx];
                                        slot.acquire(s);
                                        slot.reset_for_stripe(s,
                                                              row_start,
                                                              row_end,
                                                              state_.K_padded,
                                                              state_.blocks_per_row);
#if CYCLE_SIM
                                        (void)record_producer(
                                            cycle_sim::ProducerEventKind::ExsiaWorkspaceAcquire,
                                            slot,
                                            "ggml/src/ggml-gemmini/quants/act/exsia/"
                                            "exsia.cpp:prepare_slot_reuse");
#endif
                                        local_workspace_.reset_for_stripe(
                                            s, row_start, row_end, state_.blocks_per_row);
                                        slot.mark_quantization_started(0, aggregate_now_ns());
#if EXSIA_OBSERVATION_ENABLED
                                        LocalParallelStripeObservation & observation =
                                            state_.local_parallel_observations[s];
                                        observation            = LocalParallelStripeObservation{};
                                        observation.stripe_idx = s;
                                        observation.observed_team_size   = observed_team_size;
                                        observation.scheduled_task_count = EXSIA_LOCAL_WORKER_COUNT;
                                        const size_t total_blocks =
                                            slot.stripe.row_count() * state_.blocks_per_row;
                                        const size_t expected_blocks_per_task =
                                            total_blocks / EXSIA_LOCAL_WORKER_COUNT +
                                            (total_blocks % EXSIA_LOCAL_WORKER_COUNT != 0 ? 1 : 0);
                                        for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT;
                                             ++task_id) {
                                            const LocalWorkerContext & worker =
                                                local_workspace_.workers[task_id];
                                            LocalParallelTaskRecord & record =
                                                observation.tasks[task_id];
                                            record.task_id     = task_id;
                                            record.row_start   = worker.row_start;
                                            record.row_end     = worker.row_end;
                                            record.block_start = worker.block_start;
                                            record.block_end   = worker.block_end;
                                            record.populated_block_count =
                                                worker.block_end - worker.block_start;
                                            record.empty = record.populated_block_count == 0;
                                            record.short_task =
                                                !record.empty && record.populated_block_count <
                                                                     expected_blocks_per_task;
                                        }
#endif
                                        EXSIA_PROFILE_COLLECT(
                                            StripeProfileRecord & profile = stripe_profiles[s];
                                            profile                       = StripeProfileRecord{};
                                            profile.stripe_idx            = s;
                                            profile.row_start             = row_start;
                                            profile.row_end               = row_end;
                                            profile.team_size             = observed_team_size;
                                            start_profile_interval(profile.stripe_total);
                                            start_profile_interval(profile.local);)
                                    }
                                } catch (...) {
                                    record_failure(ExSIAState::FailureCode::Exception, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
                            }
                        }

                        for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT; ++task_id) {
#pragma omp task depend(in : prepared[s])                                                          \
    depend(out : worker_done[s * EXSIA_LOCAL_WORKER_COUNT + task_id])                              \
    firstprivate(s, slot_idx, task_id)
                            {
                                trace::ScopedContext task_context(task_trace_origin, true);
                                trace::CpuStage      task_lifetime(
                                    layer, "task.host_work", trace::CpuStage::Scope::envelope);
                                CpuWallInterval task_cpu_wall(
                                    layer, run_id, "exsia.local", s, task_id);
                                try {
                                    if (pipeline_ok.load(std::memory_order_relaxed)) {
                                        StripePipelineSlot & slot = pipeline_slots_[slot_idx];
                                        LocalWorkerContext & worker =
                                            local_workspace_.workers[task_id];
                                        LocalTaskRuntime & task_runtime =
                                            local_workspace_.local_tasks[task_id];
                                        EXSIA_PROFILE_COLLECT(
                                            StripeProfileRecord & profile = stripe_profiles[s];
                                            start_profile_interval(profile.local_groups[task_id],
                                                                   &args,
                                                                   "exsia.local_group",
                                                                   s);)
                                        bool ok = true;
                                        for (size_t block = worker.block_start;
                                             block < worker.block_end;
                                             ++block) {
                                            const size_t local_row = block / state_.blocks_per_row;
                                            const size_t block_idx = block % state_.blocks_per_row;
                                            const size_t global_row =
                                                slot.stripe.row_start + local_row;
                                            if (!run_local_block(slot,
                                                                 worker.scratch,
                                                                 global_row,
                                                                 block_idx
#if EXSIA_BRANCH_COUNTS_ENABLED
                                                                 ,
                                                                 task_runtime.cycle_stats
#endif
                                                                 )) {
                                                record_failure(
                                                    ExSIAState::FailureCode::LocalBlockFailure, s);
                                                pipeline_ok.store(false, std::memory_order_relaxed);
                                                ok = false;
                                                break;
                                            }
                                        }
                                        task_runtime.completed = ok;
                                        EXSIA_PROFILE_COLLECT(
                                            if (!end_profile_interval(
                                                    profile.local_groups[task_id])) {
                                                record_failure(
                                                    ExSIAState::FailureCode::ProfileIntervalInvalid,
                                                    s);
                                                pipeline_ok.store(false, std::memory_order_relaxed);
                                            })
                                    }
                                } catch (...) {
                                    record_failure(ExSIAState::FailureCode::Exception, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
                            }
                        }

#if GGML_GEMMINI_EXSIA_LOCAL_WORKERS == 3
#pragma omp task depend(in : worker_done[s * EXSIA_LOCAL_WORKER_COUNT],                            \
                            worker_done[s * EXSIA_LOCAL_WORKER_COUNT + 1],                         \
                            worker_done[s * EXSIA_LOCAL_WORKER_COUNT + 2])                         \
    depend(out : local_sealed[s]) firstprivate(s, slot_idx)
#elif GGML_GEMMINI_EXSIA_LOCAL_WORKERS == 4
#pragma omp task depend(in : worker_done[s * EXSIA_LOCAL_WORKER_COUNT],                            \
                            worker_done[s * EXSIA_LOCAL_WORKER_COUNT + 1],                         \
                            worker_done[s * EXSIA_LOCAL_WORKER_COUNT + 2],                         \
                            worker_done[s * EXSIA_LOCAL_WORKER_COUNT + 3])                         \
    depend(out : local_sealed[s]) firstprivate(s, slot_idx)
#else
#error "Unsupported ExSIA local worker count"
#endif
                        {
                            trace::ScopedContext task_context(task_trace_origin, true);
                            trace::CpuStage      task_lifetime(
                                layer, "task.host_work", trace::CpuStage::Scope::envelope);
                            CpuWallInterval task_cpu_wall(layer, run_id, "exsia.local_seal", s);
                            try {
                                if (pipeline_ok.load(std::memory_order_relaxed)) {
                                    StripePipelineSlot & slot = pipeline_slots_[slot_idx];
                                    (void)slot;
#if EXSIA_OBSERVATION_ENABLED
                                    LocalParallelStripeObservation & observation =
                                        state_.local_parallel_observations[s];
#endif
#if EXSIA_BRANCH_COUNTS_ENABLED
                                    slot.cycle_stats.reset();
#endif
                                    bool ok = true;
                                    for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT;
                                         ++task_id) {
                                        const LocalTaskRuntime & task_runtime =
                                            local_workspace_.local_tasks[task_id];
#if EXSIA_OBSERVATION_ENABLED
                                        observation.completed_task_count +=
                                            task_runtime.completed ? 1 : 0;
                                        observation.tasks[task_id].completed =
                                            task_runtime.completed;
#endif
                                        ok = ok && task_runtime.completed;
#if EXSIA_BRANCH_COUNTS_ENABLED
                                        const StripeCycleStats & task_stats =
                                            task_runtime.cycle_stats;
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
                                        slot.cycle_stats.p0.merge(task_stats.p0);
                                        slot.cycle_stats.p1.merge(task_stats.p1);
                                        slot.cycle_stats.p2.merge(task_stats.p2);
                                        slot.cycle_stats.p3.merge(task_stats.p3);
#else
                                        slot.cycle_stats.p0.sum += task_stats.p0.sum;
                                        slot.cycle_stats.p0.max =
                                            std::max(slot.cycle_stats.p0.max, task_stats.p0.max);
                                        slot.cycle_stats.p0.count += task_stats.p0.count;
                                        slot.cycle_stats.p1.sum += task_stats.p1.sum;
                                        slot.cycle_stats.p1.max =
                                            std::max(slot.cycle_stats.p1.max, task_stats.p1.max);
                                        slot.cycle_stats.p1.count += task_stats.p1.count;
                                        slot.cycle_stats.p2.sum += task_stats.p2.sum;
                                        slot.cycle_stats.p2.max =
                                            std::max(slot.cycle_stats.p2.max, task_stats.p2.max);
                                        slot.cycle_stats.p2.count += task_stats.p2.count;
                                        slot.cycle_stats.p3.sum += task_stats.p3.sum;
                                        slot.cycle_stats.p3.max =
                                            std::max(slot.cycle_stats.p3.max, task_stats.p3.max);
                                        slot.cycle_stats.p3.count += task_stats.p3.count;
#endif
                                        slot.cycle_stats.forced_recompute_count +=
                                            task_stats.forced_recompute_count;
#endif
                                        slot.cycle_stats.p3_bypass_no_int_count +=
                                            task_stats.p3_bypass_no_int_count;
                                        slot.cycle_stats.p3_bypass_same_scale_count +=
                                            task_stats.p3_bypass_same_scale_count;
                                        slot.cycle_stats.p3_replay_count +=
                                            task_stats.p3_replay_count;
#endif
                                    }
                                    if (!ok) {
                                        record_failure(ExSIAState::FailureCode::LocalBlockFailure,
                                                       s);
                                        pipeline_ok.store(false, std::memory_order_relaxed);
                                    } else {
#if EXSIA_VALIDATION
                                        state_.validation_p3_branch_counts[0] +=
                                            slot.cycle_stats.p3_bypass_no_int_count;
                                        state_.validation_p3_branch_counts[1] +=
                                            slot.cycle_stats.p3_bypass_same_scale_count;
                                        state_.validation_p3_branch_counts[2] +=
                                            slot.cycle_stats.p3_replay_count;
#endif
#if EXSIA_PROFILE_COLLECTION_ENABLED
                                        StripeProfileRecord & profile = stripe_profiles[s];
#if EXSIA_STAGE_PROFILE_ENABLED
                                        profile.stats = slot.cycle_stats;
#endif
                                        // local_total ends at the worker join, before Mask
                                        // Assembly, Exponent Reduction, and Folding.
                                        if (!end_profile_interval(profile.local)) {
                                            record_failure(
                                                ExSIAState::FailureCode::ProfileIntervalInvalid, s);
                                            pipeline_ok.store(false, std::memory_order_relaxed);
                                        }
#endif
                                    }
                                }
                            } catch (...) {
                                record_failure(ExSIAState::FailureCode::Exception, s);
                                pipeline_ok.store(false, std::memory_order_relaxed);
                            }
                        }

#pragma omp task depend(in : local_sealed[s]) depend(inout : post_chain)                           \
    depend(out : slot_released[s]) firstprivate(s, slot_idx)
                        {
                            trace::ScopedContext task_context(task_trace_origin, true);
                            trace::CpuStage      task_lifetime(
                                layer, "task.host_work", trace::CpuStage::Scope::envelope);
                            CpuWallInterval task_cpu_wall(layer, run_id, "exsia.mask_assembly", s);
                            try {
                                if (pipeline_ok.load(std::memory_order_relaxed)) {
                                    StripePipelineSlot & slot = pipeline_slots_[slot_idx];
                                    const size_t         active_block_count =
                                        slot.stripe.row_count() * state_.blocks_per_row;
                                    EXSIA_PROFILE_COLLECT(
                                        StripeProfileRecord & profile = stripe_profiles[s];
                                        start_profile_interval(profile.mask_assembly,
                                                               &args,
                                                               "exsia.mask_assembly",
                                                               s,
                                                               profile_host_stage_ids(profile));)
                                    const bool assembled = assemble_stripe_mask(slot, state_);
                                    EXSIA_PROFILE_COLLECT(
                                        if (!end_profile_interval(profile.mask_assembly)) {
                                            record_failure(
                                                ExSIAState::FailureCode::ProfileIntervalInvalid, s);
                                            pipeline_ok.store(false, std::memory_order_relaxed);
                                        })
                                    if (!pipeline_ok.load(std::memory_order_relaxed)) {
                                    } else if (!assembled) {
                                        record_failure(ExSIAState::FailureCode::MaskAssemblyFailure,
                                                       s);
                                        pipeline_ok.store(false, std::memory_order_relaxed);
                                    } else if (active_block_count > slot.block_exp.size()) {
                                        record_failure(
                                            ExSIAState::FailureCode::ExponentReductionFailure, s);
                                        pipeline_ok.store(false, std::memory_order_relaxed);
                                    } else {
                                        task_cpu_wall.next("exsia.exponent_reduction");
                                        EXSIA_PROFILE_COLLECT(
                                            start_profile_interval(profile.exponent_reduction);)
                                        reduce_stripe_exponents(slot, active_block_count);
                                        EXSIA_PROFILE_COLLECT(if (!end_profile_interval(
                                                                      profile.exponent_reduction)) {
                                            record_failure(
                                                ExSIAState::FailureCode::ProfileIntervalInvalid, s);
                                            pipeline_ok.store(false, std::memory_order_relaxed);
                                        })
                                        slot.mark_local_filled();
                                        if (pipeline_ok.load(std::memory_order_relaxed)) {
                                            task_cpu_wall.next("exsia.folding_and_pack");
                                            EXSIA_PROFILE_COLLECT(
                                                start_profile_interval(profile.folding);)
                                            if (!folding_.run(meta,
                                                              state_,
                                                              slot.stripe,
                                                              args,
                                                              s,
                                                              slot.q_wide,
                                                              slot.block_exp,
                                                              state_.residual,
                                                              slot.rmd_builder)) {
                                                record_failure(
                                                    ExSIAState::FailureCode::FoldingFailure, s);
                                                pipeline_ok.store(false, std::memory_order_relaxed);
                                            } else if (!seal_stripe_packet(meta, slot, args)) {
                                                record_failure(
                                                    ExSIAState::FailureCode::FoldingFailure, s);
                                                pipeline_ok.store(false, std::memory_order_relaxed);
                                            } else {
                                                slot.mark_folding_committed(aggregate_now_ns());
#if CYCLE_SIM
                                                (void)record_producer(
                                                    cycle_sim::ProducerEventKind::
                                                        ActivationRowsCommit,
                                                    slot,
                                                    "ggml/src/ggml-gemmini/quants/act/exsia/"
                                                    "exsia.cpp:folding_commit");
                                                (void)record_producer(
                                                    cycle_sim::ProducerEventKind::
                                                        ResidualPacketSeal,
                                                    slot,
                                                    "ggml/src/ggml-gemmini/quants/act/exsia/"
                                                    "exsia.cpp:seal_stripe_packet");
#endif
                                                if (!snapshot_validation_mask(
                                                        s, slot.stripe.outlier_mask)) {
                                                    record_failure(ExSIAState::FailureCode::
                                                                       ValidationSnapshotFailure,
                                                                   s);
                                                    pipeline_ok.store(false,
                                                                      std::memory_order_relaxed);
                                                } else {
                                                    EXSIA_PROFILE_COLLECT(
                                                        if (!end_profile_interval(
                                                                profile.folding)) {
                                                            record_failure(
                                                                ExSIAState::FailureCode::
                                                                    ProfileIntervalInvalid,
                                                                s);
                                                            pipeline_ok.store(
                                                                false, std::memory_order_relaxed);
                                                        })
                                                    bool stripe_ready_accepted = true;
                                                    task_cpu_wall.next("exsia.publish");
                                                    if (pipeline_ok.load(
                                                            std::memory_order_relaxed)) {
                                                        stripe_ready_accepted =
                                                            notify_stripe_ready(slot,
                                                                                run_id,
                                                                                true,
                                                                                task_cpu_wall
#if EXSIA_PROFILE_COLLECTION_ENABLED
                                                                                ,
                                                                                &profile
#endif
                                                            );
                                                    }
                                                    if (!stripe_ready_accepted) {
                                                        record_failure(ExSIAState::FailureCode::
                                                                           StripeReadySinkFailure,
                                                                       s);
                                                        pipeline_ok.store(
                                                            false, std::memory_order_relaxed);
                                                    } else {
                                                        slot.release();
#if CYCLE_SIM
                                                        (void)record_producer(
                                                            cycle_sim::ProducerEventKind::
                                                                ExsiaWorkspaceRelease,
                                                            slot,
                                                            "ggml/src/ggml-gemmini/quants/act/"
                                                            "exsia/"
                                                            "exsia.cpp:release_slot_after_sink");
#endif
                                                        EXSIA_PROFILE_COLLECT(
                                                            if (!end_profile_interval(
                                                                    profile.stripe_total)) {
                                                                record_failure(
                                                                    ExSIAState::FailureCode::
                                                                        ProfileIntervalInvalid,
                                                                    s);
                                                                pipeline_ok.store(
                                                                    false,
                                                                    std::memory_order_relaxed);
                                                            })
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } catch (...) {
                                record_failure(ExSIAState::FailureCode::Exception, s);
                                pipeline_ok.store(false, std::memory_order_relaxed);
                            }
                        }
                    }
                }
            }
#if LOG_CYCLE
            // The single barrier joins all tasks; the final parallel barrier is excluded.
            if (collect_worker_cpu) {
                const auto worker_end = gemmini_cpu_timing_read();
                gemmini_cpu_timing_add(
                    &run_timing.worker_cpu[omp_get_thread_num()], &worker_start, &worker_end);
                const auto identity = cpu_identity(layer, run_id, "exsia.worker");
                gemmini_cpu_timing_record(&identity, &worker_start, &worker_end);
            }
#endif
        }
        run_cpu_wall.resume();
        if (!pipeline_ok.load(std::memory_order_relaxed))
            return fail();
#else
        return fail(ExSIAState::FailureCode::OpenMPUnavailable);
#endif
    } else {
        run_cpu_wall.pause();
        for (size_t s = 0; s < num_stripes; ++s) {
            CpuWallInterval      stripe_cpu_wall(layer, run_id, "exsia.prepare", s);
            const size_t         row_start = s * rows_per_stripe;
            const size_t         row_end   = std::min((s + 1) * rows_per_stripe, args.I);
            StripePipelineSlot & slot      = pipeline_slots_[s % EXSIA_PIPELINE_SLOT_COUNT];
            slot.acquire(s);
            slot.reset_for_stripe(s, row_start, row_end, state_.K_padded, state_.blocks_per_row);
#if CYCLE_SIM
            (void)record_producer(
                cycle_sim::ProducerEventKind::ExsiaWorkspaceAcquire,
                slot,
                "ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:prepare_slot_sequential");
#endif
            local_workspace_.reset_for_stripe(s, row_start, row_end, state_.blocks_per_row);
            slot.mark_quantization_started(aggregate_now_tick(), aggregate_now_ns());
            StripeState & stripe = slot.stripe;
            EXSIA_PROFILE_COLLECT(
                StripeProfileRecord & profile = stripe_profiles[s]; profile = StripeProfileRecord{};
                profile.stripe_idx                                          = s;
                profile.row_start                                           = row_start;
                profile.row_end                                             = row_end;
                profile.team_size                                           = 1;
                start_profile_interval(profile.stripe_total);
                start_profile_interval(
                    profile.local,
                    requested_mode_ == ExSIAState::ExecutionMode::Sequential ? &args : nullptr,
                    "exsia.local",
                    s);)
#if EXSIA_BRANCH_COUNTS_ENABLED
            const auto record_sample = [](StripeCycleStats &            stats,
                                          const LocalBlockCycleSample & sample) {
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
                stats.p0.add(sample.stage_intervals[0]);
                stats.p1.add(sample.stage_intervals[1]);
                stats.p2.add(sample.stage_intervals[2]);
                stats.p3.add(sample.stage_intervals[3]);
#else
                stats.p0.add(sample.p0);
                stats.p1.add(sample.p1);
                stats.p2.add(sample.p2);
                stats.p3.add(sample.p3);
#endif
                stats.forced_recompute_count += sample.forced_recompute_count;
#endif
                switch (sample.p3_path) {
                case P3Path::BypassNoIntegerOutlier:
                    ++stats.p3_bypass_no_int_count;
                    break;
                case P3Path::BypassSameScale:
                    ++stats.p3_bypass_same_scale_count;
                    break;
                case P3Path::Replay:
                    ++stats.p3_replay_count;
                    break;
                }
            };
#endif
            const auto run_local_block =
                [&](StripeScratch & scratch, size_t r, size_t b EXSIA_STATS_PARAMETER) {
#if EXSIA_BRANCH_COUNTS_ENABLED
                    LocalBlockCycleSample sample;
#endif
                    const size_t col_offset = b * state_.B_size;
                    GGML_ASSERT(col_offset < args.K);
                    const size_t valid_count   = std::min(state_.B_size, args.K - col_offset);
                    const size_t local_row     = stripe.local_row(r);
                    const size_t block_base    = local_row * state_.K_padded + col_offset;
                    const size_t block_exp_idx = local_row * state_.blocks_per_row + b;
                    GGML_ASSERT(slot.q_wide.size() >= block_base + state_.B_size);
                    GGML_ASSERT(block_exp_idx < slot.block_exp.size());
                    BlockMask block_mask =
                        slot.block_mask(local_row * state_.blocks_per_row + b, state_.B_size);
                    if (!local_.run_optimized(meta,
                                              state_,
                                              src_data + r * args.K + col_offset,
                                              valid_count,
                                              state_.B_size,
                                              local_row,
                                              b,
                                              scratch,
                                              block_mask,
                                              slot.q_wide.data() + block_base,
                                              slot.block_exp[block_exp_idx]
#if EXSIA_BRANCH_COUNTS_ENABLED
                                              ,
                                              sample
#endif
                                              )) {
                        return false;
                    }

#if EXSIA_BRANCH_COUNTS_ENABLED
                    record_sample(stats, sample);
#endif
#if GGML_GEMMINI_ACT_QUANT_METRICS
                    if (scratch.actual_requantized && args.evaluation_context)
                        args.evaluation_context->requantized(r, b);
#endif
                    return true;
                };

            stripe_cpu_wall.next("exsia.local");
            if (state_.mode == ExSIAState::ExecutionMode::LocalParallel) {
#if defined(GGML_GEMMINI_HAS_OPENMP)
#if EXSIA_OBSERVATION_ENABLED
                LocalParallelStripeObservation & observation =
                    state_.local_parallel_observations[s];
                observation                      = LocalParallelStripeObservation{};
                observation.stripe_idx           = s;
                observation.scheduled_task_count = EXSIA_LOCAL_WORKER_COUNT;
                const size_t total_blocks        = stripe.row_count() * state_.blocks_per_row;
                const size_t expected_blocks_per_task =
                    total_blocks / EXSIA_LOCAL_WORKER_COUNT +
                    (total_blocks % EXSIA_LOCAL_WORKER_COUNT != 0 ? 1 : 0);
                for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT; ++task_id) {
                    const LocalWorkerContext & worker = local_workspace_.workers[task_id];
                    LocalParallelTaskRecord &  record = observation.tasks[task_id];
                    record.task_id                    = task_id;
                    record.row_start                  = worker.row_start;
                    record.row_end                    = worker.row_end;
                    record.block_start                = worker.block_start;
                    record.block_end                  = worker.block_end;
                    record.populated_block_count      = worker.block_end - worker.block_start;
                    record.empty                      = record.populated_block_count == 0;
                    record.short_task =
                        !record.empty && record.populated_block_count < expected_blocks_per_task;
                }
#endif

                std::atomic<bool> local_parallel_ok{true};
                size_t            observed_team_size = 0;
#pragma omp parallel num_threads(EXSIA_OMP_THREAD_COUNT)
                {
                    trace::ScopedContext team_context(task_trace_origin, true);
                    trace::CpuStage      team_lifetime(
                        layer, "task.host_work", trace::CpuStage::Scope::envelope);
#if LOG_CYCLE
                    const bool collect_worker_cpu = cycle::host_thread_id() != run_timing.start.tid;
                    const auto worker_start =
                        collect_worker_cpu ? gemmini_cpu_timing_read() : gemmini_cpu_sample{};
#endif
#pragma omp single
                    {
#if EXSIA_OBSERVATION_ENABLED
                        observation.observed_team_size = static_cast<size_t>(omp_get_num_threads());
                        observed_team_size             = observation.observed_team_size;
#else
                        observed_team_size = static_cast<size_t>(omp_get_num_threads());
#endif
                        if (observed_team_size != EXSIA_OMP_THREAD_COUNT) {
                            record_failure(ExSIAState::FailureCode::WrongTeamSize, s);
                            local_parallel_ok.store(false, std::memory_order_relaxed);
                        }
                        for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT; ++task_id) {
#pragma omp task firstprivate(task_id)
                            {
                                trace::ScopedContext task_context(task_trace_origin, true);
                                trace::CpuStage      task_lifetime(
                                    layer, "task.host_work", trace::CpuStage::Scope::envelope);
                                CpuWallInterval task_cpu_wall(
                                    layer, run_id, "exsia.local", s, task_id);
                                try {
                                    if (local_parallel_ok.load(std::memory_order_relaxed)) {
                                        LocalWorkerContext & worker =
                                            local_workspace_.workers[task_id];
                                        LocalTaskRuntime & task_runtime =
                                            local_workspace_.local_tasks[task_id];
                                        EXSIA_PROFILE_COLLECT(
                                            start_profile_interval(profile.local_groups[task_id],
                                                                   &args,
                                                                   "exsia.local_group",
                                                                   s);)
                                        bool ok = true;
                                        for (size_t block = worker.block_start;
                                             block < worker.block_end;
                                             ++block) {
                                            const size_t local_row  = block / state_.blocks_per_row;
                                            const size_t block_idx  = block % state_.blocks_per_row;
                                            const size_t global_row = stripe.row_start + local_row;
                                            if (!run_local_block(worker.scratch,
                                                                 global_row,
                                                                 block_idx
#if EXSIA_BRANCH_COUNTS_ENABLED
                                                                 ,
                                                                 task_runtime.cycle_stats
#endif
                                                                 )) {
                                                record_failure(
                                                    ExSIAState::FailureCode::LocalBlockFailure, s);
                                                local_parallel_ok.store(false,
                                                                        std::memory_order_relaxed);
                                                ok = false;
                                                break;
                                            }
                                        }
                                        task_runtime.completed = ok;
                                        EXSIA_PROFILE_COLLECT(
                                            if (!end_profile_interval(
                                                    profile.local_groups[task_id])) {
                                                record_failure(
                                                    ExSIAState::FailureCode::ProfileIntervalInvalid,
                                                    s);
                                                local_parallel_ok.store(false,
                                                                        std::memory_order_relaxed);
                                            })
                                    }
                                } catch (...) {
                                    record_failure(ExSIAState::FailureCode::Exception, s);
                                    local_parallel_ok.store(false, std::memory_order_relaxed);
                                }
                            }
                        }
                        {
                            trace::CpuStage taskwait(layer, "openmp.task_wait");
#pragma omp taskwait
                        }
                    }
#if LOG_CYCLE
                    // Sample after the single barrier, before the final parallel barrier.
                    if (collect_worker_cpu) {
                        const auto worker_end = gemmini_cpu_timing_read();
                        gemmini_cpu_timing_add(&run_timing.worker_cpu[omp_get_thread_num()],
                                               &worker_start,
                                               &worker_end);
                        const auto identity = cpu_identity(layer, run_id, "exsia.worker", s);
                        gemmini_cpu_timing_record(&identity, &worker_start, &worker_end);
                    }
#endif
                }

                EXSIA_PROFILE_COLLECT(profile.team_size = observed_team_size;)

                if (!local_parallel_ok.load(std::memory_order_relaxed))
                    return fail();

#if EXSIA_BRANCH_COUNTS_ENABLED
                slot.cycle_stats.reset();
#endif
                for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT; ++task_id) {
                    const LocalTaskRuntime & task_runtime = local_workspace_.local_tasks[task_id];
#if EXSIA_OBSERVATION_ENABLED
                    observation.completed_task_count += task_runtime.completed ? 1 : 0;
                    observation.tasks[task_id].completed = task_runtime.completed;
#endif
                    if (!task_runtime.completed)
                        return fail();

#if EXSIA_BRANCH_COUNTS_ENABLED
                    const StripeCycleStats & task_stats = task_runtime.cycle_stats;
#if EXSIA_STAGE_PROFILE_ENABLED
#if defined(__linux__) && defined(__aarch64__)
                    slot.cycle_stats.p0.merge(task_stats.p0);
                    slot.cycle_stats.p1.merge(task_stats.p1);
                    slot.cycle_stats.p2.merge(task_stats.p2);
                    slot.cycle_stats.p3.merge(task_stats.p3);
#else
                    slot.cycle_stats.p0.sum += task_stats.p0.sum;
                    slot.cycle_stats.p0.max = std::max(slot.cycle_stats.p0.max, task_stats.p0.max);
                    slot.cycle_stats.p0.count += task_stats.p0.count;
                    slot.cycle_stats.p1.sum += task_stats.p1.sum;
                    slot.cycle_stats.p1.max = std::max(slot.cycle_stats.p1.max, task_stats.p1.max);
                    slot.cycle_stats.p1.count += task_stats.p1.count;
                    slot.cycle_stats.p2.sum += task_stats.p2.sum;
                    slot.cycle_stats.p2.max = std::max(slot.cycle_stats.p2.max, task_stats.p2.max);
                    slot.cycle_stats.p2.count += task_stats.p2.count;
                    slot.cycle_stats.p3.sum += task_stats.p3.sum;
                    slot.cycle_stats.p3.max = std::max(slot.cycle_stats.p3.max, task_stats.p3.max);
                    slot.cycle_stats.p3.count += task_stats.p3.count;
#endif
                    slot.cycle_stats.forced_recompute_count += task_stats.forced_recompute_count;
#endif
                    slot.cycle_stats.p3_bypass_no_int_count += task_stats.p3_bypass_no_int_count;
                    slot.cycle_stats.p3_bypass_same_scale_count +=
                        task_stats.p3_bypass_same_scale_count;
                    slot.cycle_stats.p3_replay_count += task_stats.p3_replay_count;
#endif
                }
#else
                return fail(ExSIAState::FailureCode::OpenMPUnavailable);
#endif
            } else {
                for (size_t r = stripe.row_start; r < stripe.row_end; ++r) {
                    for (size_t b = 0; b < state_.blocks_per_row; ++b) {
                        if (!run_local_block(
                                stripe.scratch, r, b EXSIA_STATS_ARGUMENT(slot.cycle_stats)))
                            return fail(ExSIAState::FailureCode::LocalBlockFailure, s);
                    }
                }
            }
#if EXSIA_VALIDATION
            state_.validation_p3_branch_counts[0] += slot.cycle_stats.p3_bypass_no_int_count;
            state_.validation_p3_branch_counts[1] += slot.cycle_stats.p3_bypass_same_scale_count;
            state_.validation_p3_branch_counts[2] += slot.cycle_stats.p3_replay_count;
#endif
            // local_total ends after Local and before Mask Assembly, Exponent Reduction, and
            // Folding.
            EXSIA_PROFILE_COLLECT(if (!end_profile_interval(profile.local)) return fail(
                                      ExSIAState::FailureCode::ProfileIntervalInvalid, s);)
            const size_t active_block_count = stripe.row_count() * state_.blocks_per_row;
            stripe_cpu_wall.next("exsia.mask_assembly");
            EXSIA_PROFILE_COLLECT(start_profile_interval(profile.mask_assembly);)
            const bool assembled = assemble_stripe_mask(slot, state_);
            EXSIA_PROFILE_COLLECT(if (!end_profile_interval(profile.mask_assembly)) return fail(
                                      ExSIAState::FailureCode::ProfileIntervalInvalid, s);)
            if (!assembled)
                return fail(ExSIAState::FailureCode::MaskAssemblyFailure, s);
            if (active_block_count > slot.block_exp.size())
                return fail(ExSIAState::FailureCode::ExponentReductionFailure, s);
            stripe_cpu_wall.next("exsia.exponent_reduction");
            EXSIA_PROFILE_COLLECT(start_profile_interval(profile.exponent_reduction);)
            reduce_stripe_exponents(slot, active_block_count);
            EXSIA_PROFILE_COLLECT(
                if (!end_profile_interval(profile.exponent_reduction)) return fail(
                    ExSIAState::FailureCode::ProfileIntervalInvalid, s);)
            slot.mark_local_filled();
            stripe_cpu_wall.next("exsia.folding_and_pack");
            EXSIA_PROFILE_COLLECT(start_profile_interval(profile.folding);)

            if (!folding_.run(meta,
                              state_,
                              stripe,
                              args,
                              s,
                              slot.q_wide,
                              slot.block_exp,
                              state_.residual,
                              slot.rmd_builder))
                return fail(ExSIAState::FailureCode::FoldingFailure, s);

            // Seal the stripe packet and hand the shared handle to the metadata. Stripes
            // run in row order, so meta.rmd_packets stays ordered by row_begin.
            if (!seal_stripe_packet(meta, slot, args))
                return fail(ExSIAState::FailureCode::FoldingFailure, s);
            slot.mark_folding_committed(aggregate_now_ns(), aggregate_now_tick());
#if CYCLE_SIM
            (void)record_producer(
                cycle_sim::ProducerEventKind::ActivationRowsCommit,
                slot,
                "ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:folding_commit_sequential");
            (void)record_producer(
                cycle_sim::ProducerEventKind::ResidualPacketSeal,
                slot,
                "ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:seal_stripe_packet_sequential");
#endif

            if (!snapshot_validation_mask(s, slot.stripe.outlier_mask))
                return fail(ExSIAState::FailureCode::ValidationSnapshotFailure, s);

            EXSIA_PROFILE_COLLECT(if (!end_profile_interval(profile.folding)) return fail(
                                      ExSIAState::FailureCode::ProfileIntervalInvalid, s);)
#if EXSIA_STAGE_PROFILE_ENABLED
            profile.stats = slot.cycle_stats;
#endif
            stripe_cpu_wall.next("exsia.publish");
            if (!notify_stripe_ready(slot,
                                     run_id,
                                     true,
                                     stripe_cpu_wall
#if EXSIA_PROFILE_COLLECTION_ENABLED
                                     ,
                                     &profile
#endif
                                     ))
                return fail(ExSIAState::FailureCode::StripeReadySinkFailure, s);
            slot.release();
#if CYCLE_SIM
            (void)record_producer(cycle_sim::ProducerEventKind::ExsiaWorkspaceRelease,
                                  slot,
                                  "ggml/src/ggml-gemmini/quants/act/exsia/"
                                  "exsia.cpp:release_slot_after_sink_sequential");
#endif
            EXSIA_PROFILE_COLLECT(if (!end_profile_interval(profile.stripe_total)) return fail(
                                      ExSIAState::FailureCode::ProfileIntervalInvalid, s);)
        }
        run_cpu_wall.resume();
    }

#if EXSIA_PROFILE_COLLECTION_ENABLED
    if (!end_profile_interval(run_profile))
        return fail(ExSIAState::FailureCode::ProfileIntervalInvalid);
#if CYCLE_SIM
    for (const auto & profile : stripe_profiles) {
        const auto dependencies = profile_host_stage_ids(profile);
        args.cycle_sim_host_dependencies.insert(
            args.cycle_sim_host_dependencies.end(), dependencies.begin(), dependencies.end());
    }
#endif
#if EXSIA_VALIDATION
    state_.profile_snapshot.run_id  = run_id;
    state_.profile_snapshot.mode    = state_.mode;
    state_.profile_snapshot.run     = run_profile;
    state_.profile_snapshot.stripes = stripe_profiles;
#endif
#endif
    run_cpu_wall.pause();
    EXSIA_PROFILE_LOG(
        const ExSIAState::FailureCode profile_failure = flush_profile(profile_config,
                                                                      layer,
                                                                      run_id,
                                                                      mode,
                                                                      stripe_profiles,
                                                                      run_profile,
                                                                      state_.K_logical,
                                                                      state_.K_padded);
        if (profile_failure != ExSIAState::FailureCode::None) return fail(profile_failure);)
    ggml::gemmini::log::debug(layer,
                              "[exsia] I=%zu K=%zu stripes=%zu tau=%d rmd_packets=%zu",
                              args.I,
                              args.K,
                              num_stripes,
                              meta.sigma,
                              meta.rmd_packets.size());

#if LOG_CYCLE
    run_timing.success = true;
#endif
#if GGML_GEMMINI_ACT_QUANT_METRICS || GGML_GEMMINI_RESIDUAL_METRICS || GGML_GEMMINI_SCALE_METRICS
    if (args.evaluation_context)
        args.evaluation_context->finish_activation();
#endif
    return true;
}

bool dequantize_activation(float *                     dst,
                           size_t                      dst_row_stride,
                           size_t                      dst_col_stride,
                           size_t                      rows,
                           size_t                      cols,
                           const ggml_gemmini_args_t & args) {
    const auto run = [&]() -> bool {
        if (!args.A.valid() || !dst || args.I == 0 || args.K == 0 || dst_row_stride == 0 ||
            dst_col_stride == 0 || rows == 0 || cols == 0) {
            return false;
        }

        const auto * meta_ptr = std::get_if<Meta>(&args.act_quant.storage());
        if (!meta_ptr) {
            return false;
        }
        const Meta & meta = *meta_ptr;

        if (args.sA != 0 && args.sA != args.K) {
            return false;
        }

        const size_t src_row_stride = args.K;
        const size_t row_count      = std::min(rows, args.I);
        const size_t col_count      = std::min(cols, args.K);
        const size_t max_size       = std::numeric_limits<size_t>::max();
        if (row_count != 0 && col_count > max_size / row_count) {
            return false;
        }
        if (args.activation_row_offset > max_size - row_count) {
            return false;
        }
        const size_t global_row_begin = args.activation_row_offset;
        const size_t global_row_end   = global_row_begin + row_count;

        size_t rows_per_stripe = args.activation_rows_per_stripe;
        if (rows_per_stripe == 0) {
            const auto geometry = args.activation_quant_geometry();
            if (!geometry.ok()) {
                return false;
            }
            rows_per_stripe = geometry.geometry.stripe_rows;
        }
        if (rows_per_stripe == 0 || meta.theta.empty()) {
            return false;
        }

        std::vector<int32_t> residuals;
        rmd::RmdStatus       residual_status = rmd::RmdStatus::success;
        if (args.residual_route == residual::ResidualRoute::cpu_direct) {
            if (!meta.rmd_packets.empty()) {
                return false;
            }
            residual_status = residual::expand_direct_payloads_to_plane(meta.direct_residuals,
                                                                        global_row_begin,
                                                                        global_row_end,
                                                                        args.K,
                                                                        args.J,
                                                                        col_count,
                                                                        residuals);
        } else {
            if (!meta.direct_residuals.empty()) {
                return false;
            }
            residual_status = rmd::expand_packets_to_plane(
                meta.rmd_packets, global_row_begin, global_row_end, col_count, residuals);
        }
        if (residual_status != rmd::RmdStatus::success ||
            residuals.size() != row_count * col_count) {
            return false;
        }

        std::vector<float> staged;
        try {
            staged.resize(row_count * col_count);
        } catch (const std::bad_alloc &) {
            return false;
        } catch (const std::length_error &) {
            return false;
        }
        const int16_t invalid_theta = std::numeric_limits<int16_t>::min();
        for (size_t row = 0; row < row_count; ++row) {
            const size_t  global_row = global_row_begin + row;
            const size_t  stripe_idx = global_row / rows_per_stripe;
            const int16_t theta      = meta.resolve_stripe_theta(static_cast<int>(stripe_idx));
            if (theta == invalid_theta) {
                return false;
            }

            for (size_t col = 0; col < col_count; ++col) {
                if ((row != 0 && src_row_stride > max_size / row) ||
                    (row != 0 && dst_row_stride > max_size / row) ||
                    (col != 0 && dst_col_stride > max_size / col)) {
                    return false;
                }

                const size_t src_row_offset = row * src_row_stride;
                if (src_row_offset > max_size - col) {
                    return false;
                }

                int32_t q_int = 0;
                if (__builtin_add_overflow(
                        args.A.get(row, col), residuals[row * col_count + col], &q_int)) {
                    return false;
                }
                const float value = std::ldexp(static_cast<float>(q_int), theta);
                if (!std::isfinite(value)) {
                    return false;
                }
                staged[row * col_count + col] = value;
            }
        }

        if ((row_count > 1 && dst_row_stride > max_size / (row_count - 1)) ||
            (col_count > 1 && dst_col_stride > max_size / (col_count - 1))) {
            return false;
        }
        const size_t last_row_offset = (row_count - 1) * dst_row_stride;
        const size_t last_col_offset = (col_count - 1) * dst_col_stride;
        if (last_row_offset > max_size - last_col_offset) {
            return false;
        }
        for (size_t row = 0; row < row_count; ++row) {
            for (size_t col = 0; col < col_count; ++col) {
                dst[row * dst_row_stride + col * dst_col_stride] = staged[row * col_count + col];
            }
        }
        return true;
    };

    return run();
}

} // namespace ggml::gemmini::quants::act::exsia
