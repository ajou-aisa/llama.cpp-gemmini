#pragma once

#include "types.hpp"
#include "exsia-event.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include <gemmini/cpu-timing.h>
#include <gemmini/cpu_log_context.hpp>
#include <gemmini/cycle_reader.hpp>
#include <gemmini/host-timing.hpp>
#include <gemmini/performance.hpp>
#if CYCLE_SIM
#include <gemmini/cycle_sim_log.hpp>
#endif

#ifndef GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE
#define GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE 0
#endif

#ifndef EXSIA_VALIDATION
#define EXSIA_VALIDATION 0
#endif

#define EXSIA_PROFILE_COLLECTION_ENABLED                                                           \
    ((CYCLE_DETAIL && GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE != 0) || CYCLE_SIM)
#define EXSIA_PROFILE_LOG_ENABLED                                                                  \
    (CYCLE_DETAIL && GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE != 0 && LOG_CYCLE)
#define EXSIA_STAGE_PROFILE_ENABLED (CYCLE_DETAIL && GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE == 2)
#define EXSIA_BRANCH_COUNTS_ENABLED (EXSIA_STAGE_PROFILE_ENABLED || EXSIA_VALIDATION)
#define EXSIA_OBSERVATION_ENABLED (EXSIA_VALIDATION || EXSIA_PROFILE_COLLECTION_ENABLED)

#if EXSIA_PROFILE_LOG_ENABLED && !EXSIA_PROFILE_COLLECTION_ENABLED
#error "ExSIA profile logging requires profile collection"
#endif

#if CYCLE_DETAIL && !LOG_CYCLE
#error "CYCLE_DETAIL requires LOG_CYCLE"
#endif

#if GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE != 0 && !CYCLE_DETAIL
#error "ExSIA profiling requires CYCLE_DETAIL"
#endif

#if EXSIA_VALIDATION && !EXSIA_BRANCH_COUNTS_ENABLED
#error "EXSIA_VALIDATION requires P3 branch counts"
#endif

struct ggml_gemmini_args_t;

namespace ggml::gemmini::quants::act::exsia {
enum class ProfileCycleStatus : uint8_t {
    complete,
    missing_component,
    invalid_start,
    invalid_end,
    source_mismatch,
    event_owner_mismatch,
    event_generation_mismatch,
    structurally_cross_task,
    counter_regression,
    sum_overflow,
};

struct ProfileCycleValue {
    std::optional<uint64_t> cycles;
    ProfileCycleStatus      status = ProfileCycleStatus::missing_component;
#if defined(__linux__) && defined(__aarch64__)
    ggml::gemmini::cycle::NativeCycleReason sample_reason =
        ggml::gemmini::cycle::NativeCycleReason::none;
#endif
};

#if EXSIA_STAGE_PROFILE_ENABLED
struct StageCycleStats {
    uint64_t           sum   = 0;
    uint64_t           max   = 0;
    uint64_t           count = 0; // valid samples; sum/max are partial when count < total_count
    uint64_t           total_count   = 0;
    ProfileCycleStatus first_invalid = ProfileCycleStatus::complete;
#if defined(__linux__) && defined(__aarch64__)
    ggml::gemmini::cycle::NativeCycleReason sample_reason =
        ggml::gemmini::cycle::NativeCycleReason::none;
#endif

    void add(uint64_t value) noexcept {
        sum += value;
        max = std::max(max, value);
        ++count;
    }

    void add(const ProfileCycleValue & value) noexcept {
        ++total_count;
        if (value.cycles.has_value()) {
            const uint64_t previous = sum;
            add(*value.cycles);
            if (sum < previous && first_invalid == ProfileCycleStatus::complete)
                first_invalid = ProfileCycleStatus::sum_overflow;
        } else if (first_invalid == ProfileCycleStatus::complete) {
            first_invalid = value.status;
#if defined(__linux__) && defined(__aarch64__)
            sample_reason = value.sample_reason;
#endif
        }
    }

    void merge(const StageCycleStats & other) noexcept {
        const uint64_t previous = sum;
        sum += other.sum;
        max = std::max(max, other.max);
        count += other.count;
        total_count += other.total_count;
        if (first_invalid == ProfileCycleStatus::complete) {
            first_invalid = other.first_invalid;
#if defined(__linux__) && defined(__aarch64__)
            sample_reason = other.sample_reason;
#endif
        }
        if (sum < previous && first_invalid == ProfileCycleStatus::complete)
            first_invalid = ProfileCycleStatus::sum_overflow;
    }

    ProfileCycleStatus cycle_status() const noexcept {
        return total_count == 0 ? ProfileCycleStatus::missing_component : first_invalid;
    }

    void reset() noexcept {
        sum           = 0;
        max           = 0;
        count         = 0;
        total_count   = 0;
        first_invalid = ProfileCycleStatus::complete;
#if defined(__linux__) && defined(__aarch64__)
        sample_reason = ggml::gemmini::cycle::NativeCycleReason::none;
#endif
    }
};
#endif

#if EXSIA_VALIDATION && EXSIA_STAGE_PROFILE_ENABLED && EXSIA_PROFILE_LOG_ENABLED
std::string serialize_stage_sum_for_test(const StageCycleStats & stats);
#endif

enum class P3Path {
    BypassNoIntegerOutlier,
    BypassSameScale,
    Replay,
};

struct LocalBlockCycleSample {
#if EXSIA_STAGE_PROFILE_ENABLED
    // P0 scans exponents and marks top buckets. P1 writes provisional q_out and
    // accumulates S/SS. P2 marks integer outliers and tracks the final exponent.
    // P3 commits it, retaining q_out or overwriting it only for changed-scale replay.
    uint64_t p0                     = 0;
    uint64_t p1                     = 0;
    uint64_t p2                     = 0;
    uint64_t p3                     = 0;
    uint64_t forced_recompute_count = 0;
#if defined(__linux__) && defined(__aarch64__)
    std::array<ggml::gemmini::cycle::NativeCycleSample, 5> stage_endpoints{};
    std::array<ProfileCycleValue, 4>                       stage_intervals{};
#endif
#endif
    P3Path p3_path = P3Path::BypassNoIntegerOutlier;
#if EXSIA_BRANCH_COUNTS_ENABLED
    size_t q_tmp_to_q_final_copy_count  = 0;
    size_t q_final_to_q_wide_copy_count = 0;
    size_t replay_overwrite_count       = 0;
    size_t non_replay_overwrite_count   = 0;
    size_t block_exp_commit_count       = 0;
    size_t sigma_context_prepare_count  = 0;
    bool   has_int_outlier              = false;
#endif
#if EXSIA_VALIDATION
    int16_t final_remaining_exp = std::numeric_limits<int16_t>::min();
#endif
};

#if EXSIA_BRANCH_COUNTS_ENABLED
struct StripeCycleStats {
#if EXSIA_STAGE_PROFILE_ENABLED
    // P0: exponent/top-bucket selection; P1: direct q_out/statistics;
    // P2: integer-outlier/final-exponent selection; P3: q_out replay decision.
    StageCycleStats p0;
    StageCycleStats p1;
    StageCycleStats p2;
    StageCycleStats p3;
    uint64_t        forced_recompute_count = 0;
#endif
    uint64_t p3_bypass_no_int_count     = 0;
    uint64_t p3_bypass_same_scale_count = 0;
    uint64_t p3_replay_count            = 0;

    void reset() noexcept {
#if EXSIA_STAGE_PROFILE_ENABLED
        p0.reset();
        p1.reset();
        p2.reset();
        p3.reset();
        forced_recompute_count = 0;
#endif
        p3_bypass_no_int_count     = 0;
        p3_bypass_same_scale_count = 0;
        p3_replay_count            = 0;
    }
};

#endif

#if EXSIA_PROFILE_COLLECTION_ENABLED
struct ProfileInterval {
    uint64_t start           = 0;
    uint64_t end             = 0;
    uint64_t start_ns        = 0;
    uint64_t end_ns          = 0;
    uint64_t start_tid       = 0;
    uint64_t end_tid         = 0;
    uint64_t start_thread_id = 0;
    uint64_t end_thread_id   = 0;
#if defined(__linux__) && defined(__aarch64__)
    ggml::gemmini::cycle::NativeCycleSample start_sample{};
    ggml::gemmini::cycle::NativeCycleSample end_sample{};
#endif
    bool valid = false;
#if CYCLE_SIM
    cycle_sim::Context  host_stage{};
    log::CpuCorrelation correlation{};
    const char *        host_operation = nullptr;
    std::string         host_layer;
    uint64_t            stripe_id = UINT64_MAX;
#if LOG_CYCLE
    gemmini_cpu_sample host_start_sample{};
#endif
#endif
};

struct StripeProfileRecord {
    size_t                                                stripe_idx = 0;
    size_t                                                row_start  = 0;
    size_t                                                row_end    = 0;
    ProfileInterval                                       local;
    std::array<ProfileInterval, EXSIA_LOCAL_WORKER_COUNT> local_groups;
    ProfileInterval                                       mask_assembly;
    ProfileInterval                                       exponent_reduction;
    ProfileInterval                                       folding;
    ProfileInterval                                       stripe_total;
    size_t                                                team_size = 1;
#if EXSIA_STAGE_PROFILE_ENABLED
    StripeCycleStats stats;
    uint64_t         selected_positions = 0;
    uint64_t         residual_nnz       = 0;
#endif
};

ProfileCycleValue checked_profile_interval(const ProfileInterval & interval,
                                           bool structurally_same_owner_eligible = true) noexcept;

#endif

namespace detail {
uint64_t aggregate_now_ns();
uint64_t aggregate_now_tick();
#if LOG_CYCLE
gemmini_cycle_record_v2 cpu_identity(const char * layer,
                                     uint64_t     run_id,
                                     const char * op,
                                     uint64_t     stripe = UINT64_MAX,
                                     uint64_t     node   = UINT64_MAX);
#endif
#if EXSIA_PROFILE_COLLECTION_ENABLED
void                  start_profile_interval(ProfileInterval &           interval,
                                             const ggml_gemmini_args_t * args         = nullptr,
                                             const char *                operation    = nullptr,
                                             uint64_t                    stripe_id    = UINT64_MAX,
                                             std::vector<uint64_t>       dependencies = {});
bool                  end_profile_interval(ProfileInterval & interval);
std::vector<uint64_t> profile_host_stage_ids(const StripeProfileRecord & profile);
#endif
#if EXSIA_STAGE_PROFILE_ENABLED && defined(__linux__) && defined(__aarch64__)
void record_stage_cycles(LocalBlockCycleSample &                                        sample,
                         const std::array<ggml::gemmini::cycle::NativeCycleSample, 5> & endpoints);
#endif
#if EXSIA_PROFILE_LOG_ENABLED
struct ProfileConfig {
    std::string           log_path;
    std::string           requested_path;
    gemmini_trace_context origin = [] {
        auto captured = gemmini_trace_capture();
        if (!(captured.flags & GEMMINI_TRACE_CAPTURED)) {
            const auto context              = performance::capture_context();
            captured.flags                  = GEMMINI_TRACE_CAPTURED;
            captured.request_id             = context.request_id;
            captured.inference_operation_id = context.operation_id;
            captured.phase                  = context.phase == performance::Phase::decode;
        }
        return captured;
    }();
    std::string execution_id = cycle::host_execution_id();
    bool        setup_ok     = false;
};

ProfileConfig compile_profile_config();
FailureCode   flush_profile(const ProfileConfig &                    config,
                            const char *                             layer,
                            uint64_t                                 run_id,
                            const char *                             mode,
                            const std::vector<StripeProfileRecord> & profiles,
                            const ProfileInterval &                  run_interval,
                            size_t                                   K_logical,
                            size_t                                   K_padded);
#endif
class CpuWallInterval {
#if LOG_CYCLE
    gemmini_cycle_record_v2 identity_{};
    gemmini_cpu_sample      start_{};
#endif

  public:
    CpuWallInterval(const char * layer,
                    uint64_t     run_id,
                    const char * op,
                    uint64_t     stripe = UINT64_MAX,
                    uint64_t     node   = UINT64_MAX);

    void pause();

    void resume();
    void next(const char * op);
    ~CpuWallInterval();
};

#if LOG_CYCLE
struct ExsiaRunTiming {
    const char *                                           layer;
    uint64_t                                               run_id;
    gemmini_cpu_sample                                     start = gemmini_cpu_timing_read();
    std::array<gemmini_cpu_totals, EXSIA_OMP_THREAD_COUNT> worker_cpu{};
    uint64_t                                               handoff_ns          = 0;
    uint64_t                                               wait_ns             = 0;
    uint64_t                                               handoff_calls       = 0;
    uint64_t                                               wait_measured_calls = 0;
    bool                                                   success             = false;

    void submission(const StripeReadyEvent &   event,
                    const gemmini_cpu_sample & begin,
                    const gemmini_cpu_sample & end,
                    bool                       accepted);

    ~ExsiaRunTiming();
};
#endif
} // namespace detail
} // namespace ggml::gemmini::quants::act::exsia
