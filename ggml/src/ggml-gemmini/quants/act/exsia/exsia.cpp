#include "exsia.hpp"

#include "../../../residual/rmd/rmd-compose.hpp"
#include "exsia_shift.hpp"
#include "types.hpp"

#include "ggml-gemmini-args.h"
#include "../../common/tensor_util.hpp"

#include <gemmini/cycle_reader.hpp>
#include <gemmini/log.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if EXSIA_PROFILE_LOG_ENABLED
#include <fstream>
#include <mutex>
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

#if CYCLE_DETAIL && !LOG_CYCLE
#error "CYCLE_DETAIL requires LOG_CYCLE"
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
#define EXSIA_STATS_PARAMETER , StripeCycleStats &stats
#define EXSIA_STATS_ARGUMENT(stats) , stats
#else
#define EXSIA_STATS_PARAMETER
#define EXSIA_STATS_ARGUMENT(stats)
#endif

#if GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE != 0 && !CYCLE_DETAIL
#error "ExSIA profiling requires CYCLE_DETAIL"
#endif

#if EXSIA_VALIDATION && !EXSIA_BRANCH_COUNTS_ENABLED
#error "EXSIA_VALIDATION requires P3 branch counts"
#endif

namespace ggml::gemmini::quants::act::exsia
{
    static_assert(GGML_GEMMINI_EXSIA_SIGMA > 0, "GGML_GEMMINI_EXSIA_SIGMA must be positive");

    namespace
    {
        template <typename T>
        void release_vector(std::vector<T> &values)
        {
            std::vector<T>().swap(values);
        }

        bool checked_mul_size(size_t lhs, size_t rhs, size_t &out)
        {
            if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
                return false;

            out = lhs * rhs;
            return true;
        }

        bool checked_add_size(size_t lhs, size_t rhs, size_t &out)
        {
            if (lhs > std::numeric_limits<size_t>::max() - rhs)
                return false;

            out = lhs + rhs;
            return true;
        }

        bool checked_round_up_multiple(size_t value, size_t multiple, size_t &out)
        {
            if (multiple == 0)
                return false;

            size_t adjusted = 0;
            if (!checked_add_size(value, multiple - 1, adjusted))
                return false;

            out = (adjusted / multiple) * multiple;
            return true;
        }

        static inline int16_t exp_to_theta(int16_t e, int16_t rho)
        {
            const int16_t neg_inf = std::numeric_limits<int16_t>::min();

            if (e == neg_inf)
                return neg_inf;

            return static_cast<int16_t>(static_cast<int>(e) - static_cast<int>(rho));
        }

        static inline int32_t quantize_to_i32(float x, int16_t theta)
        {
            const int16_t neg_inf = std::numeric_limits<int16_t>::min();

            if (theta == neg_inf || !std::isfinite(x))
                return 0;

            const double scaled = std::ldexp(static_cast<double>(x), -static_cast<int>(theta));

            if (!std::isfinite(scaled))
                return scaled < 0.0
                           ? std::numeric_limits<int32_t>::min()
                           : std::numeric_limits<int32_t>::max();

            const double min_i32 = static_cast<double>(std::numeric_limits<int32_t>::min());
            const double max_i32 = static_cast<double>(std::numeric_limits<int32_t>::max());

            if (scaled <= min_i32)
                return std::numeric_limits<int32_t>::min();
            if (scaled >= max_i32)
                return std::numeric_limits<int32_t>::max();

            return static_cast<int32_t>(std::lrint(scaled));
        }

        static inline int64_t magnitude_i32(int32_t value) noexcept
        {
            const int64_t widened = static_cast<int64_t>(value);
            return widened < 0 ? -widened : widened;
        }

#if EXSIA_VALIDATION
        std::atomic<size_t> validation_block_top2_exp_rescan_counter{0};
#endif

#if EXSIA_PROFILE_LOG_ENABLED
        static inline const char *cycle_detail_log_path()
        {
            const char *path = std::getenv("GGML_GEMMINI_CYCLE_DETAIL_LOG");
            return path && path[0] ? path : "log/exsia-cycle-detail.jsonl";
        }

        struct ProfileConfig
        {
            std::string log_path;
        };

        static std::once_flag profile_log_init_once;

        ProfileConfig compile_profile_config()
        {
            ProfileConfig config;
            config.log_path = cycle_detail_log_path();
            std::call_once(profile_log_init_once, [&config] {
                std::ofstream file(config.log_path, std::ios::out | std::ios::trunc);
            });
            return config;
        }
#endif

        static inline uint64_t next_exsia_run_id()
        {
            static std::atomic<uint64_t> next{0};
            return next.fetch_add(1, std::memory_order_relaxed);
        }

        uint64_t steady_now_ns()
        {
            return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        }

#if EXSIA_PROFILE_COLLECTION_ENABLED
        uint64_t profile_now()
        {
            return ggml::gemmini::cycle::read();
        }

        uint64_t profile_now_ns()
        {
            return steady_now_ns();
        }

        uint64_t profile_thread_id()
        {
#if defined(GGML_GEMMINI_HAS_OPENMP)
            return omp_in_parallel() ? static_cast<uint64_t>(omp_get_thread_num()) : 0;
#else
            return 0;
#endif
        }

        void start_profile_interval(ProfileInterval &interval)
        {
            interval.valid = true;
            interval.start_thread_id = profile_thread_id();
            interval.start = profile_now();
            interval.start_ns = profile_now_ns();
        }

        bool end_profile_interval(ProfileInterval &interval)
        {
            assert(interval.valid);
            if (!interval.valid)
                return false;
            interval.end = profile_now();
            interval.end_ns = profile_now_ns();
            interval.end_thread_id = profile_thread_id();
            return interval.end >= interval.start;
        }
#endif

#if EXSIA_PROFILE_LOG_ENABLED
        void write_json_string(std::ostream &out, const std::string &value)
        {
            out.put('"');
            for (const char character : value)
            {
                switch (character)
                {
                case '\\': out << "\\\\"; break;
                case '"': out << "\\\""; break;
                case '\n': out << "\\n"; break;
                case '\r': out << "\\r"; break;
                case '\t': out << "\\t"; break;
                default: out.put(character); break;
                }
            }
            out.put('"');
        }
#endif

    }

#if EXSIA_STAGE_PROFILE_ENABLED
#define EXSIA_STAGE_CYCLE_READ() ggml::gemmini::cycle::read()
#else
#define EXSIA_STAGE_CYCLE_READ() static_cast<uint64_t>(0)
#endif

#if EXSIA_PROFILE_LOG_ENABLED
    namespace
    {
        static inline bool profile_interval_valid(const ProfileInterval &interval)
        {
            return interval.valid && interval.end >= interval.start;
        }

        static inline size_t expected_profile_team_size(const char *mode)
        {
            return std::strcmp(mode, "Sequential") == 0 ? 1 : EXSIA_OMP_THREAD_COUNT;
        }

        static inline void write_timeline_event(std::ostream &out,
                                                const char *layer,
                                                uint64_t run_id,
                                                const char *mode,
                                                size_t stripe_idx,
                                                const char *stage,
                                                const char *suffix,
                                                const ProfileInterval &interval,
                                                size_t team_size)
        {
            out << "{\"record_type\":\"TIMELINE\",\"layer\":";
            write_json_string(out, layer);
            out << ",\"op\":\"exsia.timeline.run." << run_id << ".stripe."
                << stripe_idx << "." << suffix
                << "\",\"run_id\":" << run_id << ",\"mode\":";
            write_json_string(out, mode);
            out << ",\"stripe_id\":" << stripe_idx << ",\"stage\":";
            write_json_string(out, stage);
            out << ",\"start\":" << interval.start << ",\"end\":" << interval.end
                << ",\"start_thread_id\":" << interval.start_thread_id
                << ",\"end_thread_id\":" << interval.end_thread_id
                << ",\"clock_mode\":";
            write_json_string(out, ggml::gemmini::cycle::clock_mode());
            out << ",\"units\":";
            write_json_string(out, ggml::gemmini::cycle::units());
            out << ",\"timer_resolution\":" << ggml::gemmini::cycle::resolution()
                << ",\"team_size\":" << team_size
                << ",\"elapsed\":" << (interval.end - interval.start) << "}\n";
        }

        static inline void write_timeline_run_event(std::ostream &out,
                                                    const char *layer,
                                                    uint64_t run_id,
                                                    const char *mode,
                                                     const ProfileInterval &interval,
                                                     size_t team_size)
        {
            out << "{\"record_type\":\"TIMELINE\",\"layer\":";
            write_json_string(out, layer);
            out << ",\"op\":\"exsia.timeline.run." << run_id
                << ".end_to_end_total\",\"run_id\":" << run_id << ",\"mode\":";
            write_json_string(out, mode);
            out << ",\"stripe_id\":" << ExSIAState::no_failure_stripe << ",\"stage\":\"Run\""
                << ",\"start\":" << interval.start << ",\"end\":" << interval.end
                << ",\"start_thread_id\":" << interval.start_thread_id
                << ",\"end_thread_id\":" << interval.end_thread_id
                << ",\"clock_mode\":";
            write_json_string(out, ggml::gemmini::cycle::clock_mode());
            out << ",\"units\":";
            write_json_string(out, ggml::gemmini::cycle::units());
            out << ",\"timer_resolution\":" << ggml::gemmini::cycle::resolution()
                << ",\"team_size\":" << team_size
                << ",\"elapsed\":" << (interval.end - interval.start) << "}\n";
        }

#if EXSIA_STAGE_PROFILE_ENABLED
        static inline void write_stage_metric(std::ostream &out,
                                              const char *layer,
                                              uint64_t run_id,
                                              const char *mode,
                                              size_t stripe_idx,
                                              const char *suffix,
                                              uint64_t value,
                                              const char *value_units,
                                              size_t team_size)
        {
            out << "{\"record_type\":\"STAGE\",\"layer\":";
            write_json_string(out, layer);
            out << ",\"op\":\"exsia.stage.run." << run_id << ".stripe."
                << stripe_idx << "." << suffix
                << "\",\"run_id\":" << run_id << ",\"mode\":";
            write_json_string(out, mode);
            out << ",\"stripe_id\":" << stripe_idx << ",\"metric\":";
            write_json_string(out, suffix);
            out << ",\"value\":" << value << ",\"value_units\":";
            write_json_string(out, value_units);
            out << ",\"team_size\":" << team_size << "}\n";
        }
#endif

        static std::mutex profile_flush_mutex;

        static inline ExSIAState::FailureCode flush_profile(const ProfileConfig &config,
                                                            const char *layer,
                                                            uint64_t run_id,
                                                            const char *mode,
                                                            const std::vector<StripeProfileRecord> &profiles,
                                                            const ProfileInterval &run_interval)
        {
            std::ostringstream trace;
            const size_t expected_team_size = expected_profile_team_size(mode);
            const bool sequential = std::strcmp(mode, "Sequential") == 0;

            for (const StripeProfileRecord &profile : profiles)
            {
                if (!profile_interval_valid(profile.local) ||
                    !profile_interval_valid(profile.mask_assembly) ||
                    !profile_interval_valid(profile.exponent_reduction) ||
                    !profile_interval_valid(profile.folding) ||
                    !profile_interval_valid(profile.stripe_total) ||
                    profile.team_size != expected_team_size)
                    return ExSIAState::FailureCode::ProfileIntervalInvalid;
                write_timeline_event(trace, layer, run_id, mode, profile.stripe_idx, "Local",
                                     "local_total", profile.local, profile.team_size);
                if (!sequential)
                {
                    for (size_t group = 0; group < profile.local_groups.size(); ++group)
                    {
                        const ProfileInterval &interval = profile.local_groups[group];
                        if (!profile_interval_valid(interval))
                            return ExSIAState::FailureCode::ProfileIntervalInvalid;
                        char suffix[48];
                        std::snprintf(suffix, sizeof(suffix), "local_group_%zu_total", group);
                        write_timeline_event(trace, layer, run_id, mode, profile.stripe_idx,
                                             "LocalGroup", suffix, interval, profile.team_size);
                    }
                }
                write_timeline_event(trace, layer, run_id, mode, profile.stripe_idx,
                                     "MaskAssembly", "mask_assembly_total",
                                     profile.mask_assembly, profile.team_size);
                write_timeline_event(trace, layer, run_id, mode, profile.stripe_idx,
                                     "ExponentReduction", "exponent_reduction_total",
                                     profile.exponent_reduction, profile.team_size);
                write_timeline_event(trace, layer, run_id, mode, profile.stripe_idx,
                                     "Folding", "folding_total", profile.folding, profile.team_size);
                write_timeline_event(trace, layer, run_id, mode, profile.stripe_idx,
                                     "Stripe", "stripe_total", profile.stripe_total, profile.team_size);
#if EXSIA_STAGE_PROFILE_ENABLED
                const StageCycleStats *stages[] = {
                    &profile.stats.p0,
                    &profile.stats.p1,
                    &profile.stats.p2,
                    &profile.stats.p3,
                };
                for (size_t stage = 0; stage < 4; ++stage)
                {
                    char suffix[48];
                    std::snprintf(suffix, sizeof(suffix), "local.p%zu.sum", stage);
                    write_stage_metric(trace, layer, run_id, mode, profile.stripe_idx,
                                       suffix, stages[stage]->sum,
                                       ggml::gemmini::cycle::units(), profile.team_size);
                    std::snprintf(suffix, sizeof(suffix), "local.p%zu.count", stage);
                    write_stage_metric(trace, layer, run_id, mode, profile.stripe_idx,
                                       suffix, stages[stage]->count, "count", profile.team_size);
                    std::snprintf(suffix, sizeof(suffix), "local.p%zu.max", stage);
                    write_stage_metric(trace, layer, run_id, mode, profile.stripe_idx,
                                       suffix, stages[stage]->max,
                                       ggml::gemmini::cycle::units(), profile.team_size);
                }
                write_stage_metric(trace, layer, run_id, mode, profile.stripe_idx,
                                    "local.p3.bypass_no_int.count",
                                    profile.stats.p3_bypass_no_int_count, "count", profile.team_size);
                write_stage_metric(trace, layer, run_id, mode, profile.stripe_idx,
                                    "local.p3.bypass_same_scale.count",
                                    profile.stats.p3_bypass_same_scale_count, "count", profile.team_size);
                write_stage_metric(trace, layer, run_id, mode, profile.stripe_idx,
                                    "local.p3.replay.count",
                                    profile.stats.p3_replay_count, "count", profile.team_size);
#endif
            }
            if (!profile_interval_valid(run_interval))
                return ExSIAState::FailureCode::ProfileIntervalInvalid;
            const size_t run_team_size = profiles.empty() ? expected_team_size : profiles.front().team_size;
            write_timeline_run_event(trace, layer, run_id, mode, run_interval, run_team_size);

            std::lock_guard<std::mutex> lock(profile_flush_mutex);
            std::ofstream file(config.log_path, std::ios::app);
            if (!file)
                return ExSIAState::FailureCode::ProfileFlushFailure;
            file << trace.str();
            file.flush();
            return file ? ExSIAState::FailureCode::None : ExSIAState::FailureCode::ProfileFlushFailure;
        }
    }
#endif

    std::array<ExSIAState::ExecutionModeAvailability, 3> execution_mode_availability()
    {
        using Mode = ExSIAState::ExecutionMode;

#if defined(GGML_GEMMINI_HAS_OPENMP)
        constexpr const char * local_parallel_reason = "fixed four-task OpenMP Local stage";
        constexpr const char * pipeline_reason = "two-slot OpenMP Local/Folding pipeline";
#else
        constexpr const char * local_parallel_reason = "OpenMP unavailable";
        constexpr const char * pipeline_reason = "OpenMP unavailable";
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

    int16_t ExpScanner::unbiased_exp(const float &x)
    {
        if (x == 0.f || !std::isfinite(x))
            return std::numeric_limits<int16_t>::min();

        return static_cast<int16_t>(std::ilogb(std::abs(x)));
    }

    void ExpScanner::scan_top2_exp(const std::vector<float> &x,
                                   BlockState &blk)
    {
        const size_t n = x.size();
        GGML_ASSERT(blk.e.size() >= n);
        blk.reset();
        blk.blk_size = n;
        for (size_t i = 0; i < n; ++i)
        {
            int16_t exp = unbiased_exp(x[i]);
            blk.e[i] = exp;
            if (exp > blk.e1)
            {
                blk.e2 = blk.e1;
                blk.e1 = exp;
            }
            else if (exp < blk.e1 && exp > blk.e2)
                blk.e2 = exp;
        }
    }

    void ExpScanner::scan_top2_exp(const float *x, size_t count, BlockState &blk)
    {
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(blk.e.size() >= count);
        blk.reset();
        blk.blk_size = count;
        for (size_t i = 0; i < count; ++i)
        {
            const int16_t exp = unbiased_exp(x[i]);
            blk.e[i] = exp;
            if (exp > blk.e1)
            {
                blk.e2 = blk.e1;
                blk.e1 = exp;
            }
            else if (exp < blk.e1 && exp > blk.e2)
                blk.e2 = exp;
        }
    }

    void ExpScanner::update_block_top2_exp(const BlockMask &mask, BlockState &blk)
    {
#if EXSIA_VALIDATION
        validation_block_top2_exp_rescan_counter.fetch_add(1, std::memory_order_relaxed);
#endif
        blk.e1 = std::numeric_limits<int16_t>::min();
        blk.e2 = std::numeric_limits<int16_t>::min();

        const size_t n = blk.blk_size;
        for (size_t i = 0; i < n; ++i)
        {
            if (mask.is_set(i))
                continue;

            const int16_t exp = blk.e[i];
            if (exp > blk.e1)
            {
                blk.e2 = blk.e1;
                blk.e1 = exp;
            }
            else if (exp < blk.e1 && exp > blk.e2)
                blk.e2 = exp;
        }
    }

#if EXSIA_VALIDATION
    void reset_validation_block_top2_exp_rescan_count()
    {
        validation_block_top2_exp_rescan_counter.store(0, std::memory_order_relaxed);
    }

    size_t validation_block_top2_exp_rescan_count()
    {
        return validation_block_top2_exp_rescan_counter.load(std::memory_order_relaxed);
    }
#endif

    void ExpScanner::update_stripe_top2_exp(StripeState &stripe, int16_t exp)
    {
        if (exp > stripe.e1)
        {
            stripe.e2 = stripe.e1;
            stripe.e1 = exp;
        }
        else if (exp < stripe.e1 && exp > stripe.e2)
            stripe.e2 = exp;
    }

    void OutlierMarker::mark_outlier(StripeState &stripe,
                                     size_t row,
                                     size_t blk_idx,
                                     size_t blk_size,
                                     const BitMask &d_mask) const
    {
        size_t n = blk_size;
        const size_t local_row = stripe.local_row(row);
        for (size_t i = 0; i < n; ++i)
        {
            size_t col = blk_idx * n + i;
            if (d_mask.is_set(0, i))
                stripe.outlier_mask.set(local_row, col);
        }
    }

    std::vector<int32_t> WideQuantizer::quantize_block(const std::vector<float> &x,
                                                       int16_t theta_b)
    {
        std::vector<int32_t> q(x.size());
        quantize_block(x, theta_b, q);
        return q;
    }

    void WideQuantizer::quantize_block(const std::vector<float> &x,
                                       int16_t theta_b,
                                       std::vector<int32_t> &q) const
    {
        GGML_ASSERT(q.size() >= x.size());
        const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
        for (size_t i = 0; i < x.size(); ++i)
            q[i] = null_theta ? 0 : quantize_to_i32(x[i], theta_b);
    }

    std::tuple<std::vector<int32_t>, __int128_t, __int128_t>
    WideQuantizer::quantize_block(const std::vector<float> &x,
                                  size_t row,
                                  size_t col_offset,
                                  const BitMask &mask,
                                  int16_t theta_b)
    {
        std::vector<int32_t> q(x.size());
        __int128_t S = 0;
        __int128_t SS = 0;
        quantize_block(x, row, col_offset, mask, theta_b, q, S, SS);
        return {q, S, SS};
    }

    void WideQuantizer::quantize_block(const std::vector<float> &x,
                                        size_t row,
                                       size_t col_offset,
                                       const BitMask &mask,
                                       int16_t theta_b,
                                       std::vector<int32_t> &q,
                                       __int128_t &S,
                                       __int128_t &SS) const
    {
        const size_t n = x.size();
        GGML_ASSERT(q.size() >= n);

        S = 0;
        SS = 0;
        const bool use_mask = mask.rows != 0 && mask.cols != 0;
        const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
        for (size_t i = 0; i < n; ++i)
        {
            size_t col = col_offset + i;
            const int32_t tmp = null_theta ? 0 : quantize_to_i32(x[i], theta_b);
            q[i] = tmp;
            if (!use_mask || !mask.is_set(row, col))
            {
                const __int128_t magnitude = static_cast<__int128_t>(magnitude_i32(tmp));
                S += magnitude;
                SS += magnitude * magnitude;
            }
        }
    }

    void WideQuantizer::quantize_block(const std::vector<float> &x,
                                        const BlockMask &mask,
                                        int16_t theta_b,
                                        std::vector<int32_t> &q,
                                        __int128_t &S,
                                        __int128_t &SS) const
    {
        const size_t n = x.size();
        GGML_ASSERT(q.size() >= n);
        GGML_ASSERT(mask.bit_count >= n);

        S = 0;
        SS = 0;
        const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
        for (size_t i = 0; i < n; ++i)
        {
            const int32_t tmp = null_theta ? 0 : quantize_to_i32(x[i], theta_b);
            q[i] = tmp;
            if (!mask.is_set(i))
            {
                const __int128_t magnitude = static_cast<__int128_t>(magnitude_i32(tmp));
                S += magnitude;
                SS += magnitude * magnitude;
            }
        }
    }

    void WideQuantizer::quantize_block(const float *x,
                                        size_t count,
                                        const BlockMask &mask,
                                        int16_t theta_b,
                                        std::vector<int32_t> &q,
                                        __int128_t &S,
                                        __int128_t &SS) const
    {
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(q.size() >= count);
        GGML_ASSERT(mask.bit_count >= count);

        S = 0;
        SS = 0;
        const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
        for (size_t i = 0; i < count; ++i)
        {
            const int32_t tmp = null_theta ? 0 : quantize_to_i32(x[i], theta_b);
            q[i] = tmp;
            if (!mask.is_set(i))
            {
                const __int128_t magnitude = static_cast<__int128_t>(magnitude_i32(tmp));
                S += magnitude;
                SS += magnitude * magnitude;
            }
        }
    }

    void WideQuantizer::quantize_block(const float *x,
                                        size_t count,
                                        int16_t theta_b,
                                        std::vector<int32_t> &q) const
    {
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(q.size() >= count);
        const bool null_theta = theta_b == std::numeric_limits<int16_t>::min();
        for (size_t i = 0; i < count; ++i)
            q[i] = null_theta ? 0 : quantize_to_i32(x[i], theta_b);
    }

    SigmaDetector::SigmaContext SigmaDetector::prepare(__int128_t S, __int128_t SS, size_t N) const
    {
        SigmaContext context;
        context.n = static_cast<__int128_t>(N);
        context.S = S;
        if (N == 0)
            return context;

        const __int128_t variance_numer = context.n * SS - S * S;
        if (variance_numer <= 0)
            return context;

        const __int128_t tau = GGML_GEMMINI_EXSIA_SIGMA;
        context.threshold = tau * tau * variance_numer;
        context.valid = true;
        return context;
    }

    bool SigmaDetector::detect(int32_t q, const SigmaContext &context) const
    {
        if (!context.valid)
            return false;

        const __int128_t centered = context.n * static_cast<__int128_t>(magnitude_i32(q)) - context.S;
        return centered > 0 && centered * centered > context.threshold;
    }

    bool SigmaDetector::detect_sigma(int32_t q, __int128_t S, __int128_t SS, size_t N)
    {
        return detect(q, prepare(S, SS, N));
    }

    std::pair<int8_t, int32_t> ResidualClipper::clip_with_residual(int32_t q)
    {
        int32_t q8 = q > 127 ? 127 : (q < -128 ? -128 : q);
        int32_t res = q - q8;
        return {static_cast<int8_t>(q8), res};
    }


    bool LocalStage::run_optimized(
        Meta &meta,
        ExSIAState &state,
        const float *x,
        size_t valid_count,
        size_t block_size,
        size_t local_row,
        size_t blk_idx,
        StripeScratch &scratch,
        BlockMask &block_mask,
        int32_t *q_out,
        int16_t &block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
        ,
        LocalBlockCycleSample &cycle_sample)
#else
        )
#endif
    {
        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(q_out != nullptr);
        GGML_ASSERT(block_size == state.B_size);
        GGML_ASSERT(valid_count <= block_size);
        (void) local_row;
        (void) blk_idx;

        if (valid_count == block_size)
        {
            return run_optimized_full(meta, x, block_size, scratch, block_mask, q_out, block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
                                      , cycle_sample
#endif
            );
        }
        return run_optimized_partial(meta, x, valid_count, block_size, scratch, block_mask,
                                     q_out, block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
                                     , cycle_sample
#endif
        );
    }

    bool LocalStage::run_optimized_full(
        Meta &meta,
        const float *x,
        size_t block_size,
        StripeScratch &scratch,
        BlockMask &block_mask,
        int32_t *q_out,
        int16_t &block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
        ,
        LocalBlockCycleSample &cycle_sample)
#else
        )
#endif
    {
#if EXSIA_BRANCH_COUNTS_ENABLED
        cycle_sample = LocalBlockCycleSample{};
#endif

        GGML_ASSERT(x != nullptr);
        GGML_ASSERT(q_out != nullptr);
        BlockState &blk = scratch.block;
        const int16_t neg_inf = std::numeric_limits<int16_t>::min();
        __int128_t S = 0;
        __int128_t SS = 0;
        size_t unmasked_count = 0;
        bool has_int_outlier = false;
        int16_t final_exp = neg_inf;
        int16_t e_pre = neg_inf;
        int16_t theta_pre = neg_inf;
#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t0 = EXSIA_STAGE_CYCLE_READ();
#endif

        blk.reset();
        blk.blk_size = block_size;
        block_mask.clear();
        for (size_t i = 0; i < block_size; ++i)
        {
            const int16_t exp = unit_exp_.unbiased_exp(x[i]);
            blk.e[i] = exp;
            if (exp > blk.e1)
            {
                blk.e2 = blk.e1;
                blk.e1 = exp;
                block_mask.clear();
                block_mask.set(i);
            }
            else if (exp == blk.e1 && exp != neg_inf)
                block_mask.set(i);
            else if (exp < blk.e1 && exp > blk.e2)
                blk.e2 = exp;
        }
        const bool has_second_bucket = blk.e2 != neg_inf;
        if (!has_second_bucket)
            block_mask.clear();
        e_pre = has_second_bucket ? blk.e2 : blk.e1;
#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t1 = EXSIA_STAGE_CYCLE_READ();
#endif

        theta_pre = exp_to_theta(e_pre, meta.rho);
        const bool null_theta = theta_pre == neg_inf;
        for (size_t i = 0; i < block_size; ++i)
        {
            const int32_t tmp = null_theta ? 0 : quantize_to_i32(x[i], theta_pre);
            q_out[i] = tmp;
            if (block_mask.is_set(i))
                continue;

            const __int128_t magnitude = static_cast<__int128_t>(magnitude_i32(tmp));
            S += magnitude;
            SS += magnitude * magnitude;
            ++unmasked_count;
        }
#if EXSIA_VALIDATION
        scratch.reference.p0_e1 = blk.e1;
        scratch.reference.p0_e2 = blk.e2;
        scratch.reference.p0_e_pre = e_pre;
        std::copy_n(block_mask.words, BlockMask::word_count(block_size), scratch.reference.p0_top_mask_words.begin());
        scratch.reference.p1_S = S;
        scratch.reference.p1_SS = SS;
        scratch.reference.p1_N = unmasked_count;
#endif

#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t2 = EXSIA_STAGE_CYCLE_READ();
#endif

        const SigmaDetector::SigmaContext sigma_context = unit_sigma_.prepare(S, SS, unmasked_count);
#if EXSIA_BRANCH_COUNTS_ENABLED
        ++cycle_sample.sigma_context_prepare_count;
#endif
        for (size_t i = 0; i < block_size; ++i)
        {
            if (block_mask.is_set(i))
                continue;

            if (unit_sigma_.detect(q_out[i], sigma_context))
            {
                block_mask.set(i);
                has_int_outlier = true;
            }
            else
                final_exp = std::max(final_exp, blk.e[i]);
        }
#if EXSIA_BRANCH_COUNTS_ENABLED
        cycle_sample.has_int_outlier = has_int_outlier;
#endif
#if EXSIA_VALIDATION
        cycle_sample.final_remaining_exp = final_exp;
#endif

#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t3 = EXSIA_STAGE_CYCLE_READ();
#endif

        if (!has_int_outlier)
        {
            blk.e_b = e_pre;
            blk.theta_b = theta_pre;
#if EXSIA_BRANCH_COUNTS_ENABLED
            cycle_sample.p3_path = P3Path::BypassNoIntegerOutlier;
#endif
        }
        else
        {
            blk.e_b = final_exp;
            blk.theta_b = exp_to_theta(blk.e_b, meta.rho);

            if (blk.theta_b == theta_pre)
            {
#if EXSIA_BRANCH_COUNTS_ENABLED
                cycle_sample.p3_path = P3Path::BypassSameScale;
#endif
            }
            else
            {
                const bool final_null_theta = blk.theta_b == neg_inf;
                for (size_t i = 0; i < block_size; ++i)
                    q_out[i] = final_null_theta ? 0 : quantize_to_i32(x[i], blk.theta_b);
#if EXSIA_BRANCH_COUNTS_ENABLED
                ++cycle_sample.replay_overwrite_count;
                cycle_sample.p3_path = P3Path::Replay;
#endif
            }
        }

        block_exp_out = blk.e_b;
#if EXSIA_BRANCH_COUNTS_ENABLED
        ++cycle_sample.block_exp_commit_count;
#endif

#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t4 = EXSIA_STAGE_CYCLE_READ();
        cycle_sample.p0 = t1 >= t0 ? t1 - t0 : 0;
        cycle_sample.p1 = t2 >= t1 ? t2 - t1 : 0;
        cycle_sample.p2 = t3 >= t2 ? t3 - t2 : 0;
        cycle_sample.p3 = t4 >= t3 ? t4 - t3 : 0;
#endif

        return true;
    }

    bool LocalStage::run_optimized_partial(
        Meta &meta,
        const float *x,
        size_t valid_count,
        size_t block_size,
        StripeScratch &scratch,
        BlockMask &block_mask,
        int32_t *q_out,
        int16_t &block_exp_out
#if EXSIA_BRANCH_COUNTS_ENABLED
        ,
        LocalBlockCycleSample &cycle_sample)
#else
        )
#endif
    {
#if EXSIA_BRANCH_COUNTS_ENABLED
        cycle_sample = LocalBlockCycleSample{};
#endif
        BlockState &blk = scratch.block;
        const int16_t neg_inf = std::numeric_limits<int16_t>::min();
        std::copy_n(x, valid_count, blk.x.begin());
        std::fill(blk.x.begin() + valid_count, blk.x.begin() + block_size, 0.0f);
        __int128_t S = 0;
        __int128_t SS = 0;
        size_t unmasked_count = 0;
        bool has_int_outlier = false;
        int16_t final_exp = neg_inf;
        int16_t e_pre = neg_inf;
        int16_t theta_pre = neg_inf;
#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t0 = EXSIA_STAGE_CYCLE_READ();
#endif

        blk.reset();
        blk.blk_size = block_size;
        block_mask.clear();
        for (size_t i = 0; i < valid_count; ++i)
        {
            const int16_t exp = unit_exp_.unbiased_exp(blk.x[i]);
            blk.e[i] = exp;
            if (exp > blk.e1)
            {
                blk.e2 = blk.e1;
                blk.e1 = exp;
                block_mask.clear();
                block_mask.set(i);
            }
            else if (exp == blk.e1 && exp != neg_inf)
                block_mask.set(i);
            else if (exp < blk.e1 && exp > blk.e2)
                blk.e2 = exp;
        }
        std::fill(blk.e.begin() + valid_count, blk.e.begin() + block_size, neg_inf);
        const bool has_second_bucket = blk.e2 != neg_inf;
        if (!has_second_bucket)
            block_mask.clear();
        e_pre = has_second_bucket ? blk.e2 : blk.e1;
#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t1 = EXSIA_STAGE_CYCLE_READ();
#endif

        theta_pre = exp_to_theta(e_pre, meta.rho);
        const bool null_theta = theta_pre == neg_inf;
        for (size_t i = 0; i < block_size; ++i)
        {
            const int32_t tmp = null_theta ? 0 : quantize_to_i32(blk.x[i], theta_pre);
            q_out[i] = tmp;
            if (i >= valid_count || block_mask.is_set(i))
                continue;

            const __int128_t magnitude = static_cast<__int128_t>(magnitude_i32(tmp));
            S += magnitude;
            SS += magnitude * magnitude;
            ++unmasked_count;
        }
#if EXSIA_VALIDATION
        scratch.reference.p0_e1 = blk.e1;
        scratch.reference.p0_e2 = blk.e2;
        scratch.reference.p0_e_pre = e_pre;
        std::copy_n(block_mask.words, BlockMask::word_count(block_size), scratch.reference.p0_top_mask_words.begin());
        scratch.reference.p1_S = S;
        scratch.reference.p1_SS = SS;
        scratch.reference.p1_N = unmasked_count;
#endif
#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t2 = EXSIA_STAGE_CYCLE_READ();
#endif

        const SigmaDetector::SigmaContext sigma_context = unit_sigma_.prepare(S, SS, unmasked_count);
#if EXSIA_BRANCH_COUNTS_ENABLED
        ++cycle_sample.sigma_context_prepare_count;
#endif
        for (size_t i = 0; i < valid_count; ++i)
        {
            if (block_mask.is_set(i))
                continue;
            if (unit_sigma_.detect(q_out[i], sigma_context))
            {
                block_mask.set(i);
                has_int_outlier = true;
            }
            else
                final_exp = std::max(final_exp, blk.e[i]);
        }
#if EXSIA_BRANCH_COUNTS_ENABLED
        cycle_sample.has_int_outlier = has_int_outlier;
#endif
#if EXSIA_VALIDATION
        cycle_sample.final_remaining_exp = final_exp;
#endif
#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t3 = EXSIA_STAGE_CYCLE_READ();
#endif

        if (!has_int_outlier)
        {
            blk.e_b = e_pre;
            blk.theta_b = theta_pre;
#if EXSIA_BRANCH_COUNTS_ENABLED
            cycle_sample.p3_path = P3Path::BypassNoIntegerOutlier;
#endif
        }
        else
        {
            blk.e_b = final_exp;
            blk.theta_b = exp_to_theta(blk.e_b, meta.rho);
            if (blk.theta_b == theta_pre)
            {
#if EXSIA_BRANCH_COUNTS_ENABLED
                cycle_sample.p3_path = P3Path::BypassSameScale;
#endif
            }
            else
            {
                const bool final_null_theta = blk.theta_b == neg_inf;
                for (size_t i = 0; i < block_size; ++i)
                    q_out[i] = final_null_theta ? 0 : quantize_to_i32(blk.x[i], blk.theta_b);
#if EXSIA_BRANCH_COUNTS_ENABLED
                ++cycle_sample.replay_overwrite_count;
                cycle_sample.p3_path = P3Path::Replay;
#endif
            }
        }

        block_exp_out = blk.e_b;
#if EXSIA_BRANCH_COUNTS_ENABLED
        ++cycle_sample.block_exp_commit_count;
#endif
#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t4 = EXSIA_STAGE_CYCLE_READ();
        cycle_sample.p0 = t1 >= t0 ? t1 - t0 : 0;
        cycle_sample.p1 = t2 >= t1 ? t2 - t1 : 0;
        cycle_sample.p2 = t3 >= t2 ? t3 - t2 : 0;
        cycle_sample.p3 = t4 >= t3 ? t4 - t3 : 0;
#endif
        return true;
    }

#if EXSIA_VALIDATION
    bool LocalStage::run_reference(
        Meta &meta,
        ExSIAState &state,
        const std::vector<float> &x,
        size_t local_row,
        size_t blk_idx,
        StripeScratch &scratch,
        BlockMask &block_mask,
        std::vector<int32_t> &stripe_q_wide,
        std::vector<int16_t> &stripe_block_exp,
        LocalBlockCycleSample &cycle_sample)
    {
        const size_t blk_size = state.B_size;
        const size_t base = local_row * state.K_padded + blk_idx * blk_size;
        cycle_sample = LocalBlockCycleSample{};

        GGML_ASSERT(x.size() == blk_size);
        BlockState &blk = scratch.block;
        const int16_t neg_inf = std::numeric_limits<int16_t>::min();
        bool has_second_bucket = false;
        std::vector<int32_t> &q_tmp = scratch.reference.q_tmp;
        std::vector<int32_t> &q_final = scratch.reference.q_final;
        __int128_t S = 0;
        __int128_t SS = 0;
        size_t unmasked_count = 0;
        bool has_int_outlier = false;
        int16_t e_pre = neg_inf;
        int16_t theta_pre = neg_inf;
#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t0 = EXSIA_STAGE_CYCLE_READ();
#endif

        unit_exp_.scan_top2_exp(x, blk);
        has_second_bucket = (blk.e2 != neg_inf);
        block_mask.clear();

        if (has_second_bucket)
        {
            for (size_t i = 0; i < blk_size; ++i)
            {
                const size_t col = blk_idx * blk_size + i;
                if (col < state.K_logical && blk.e[i] != neg_inf && blk.e[i] == blk.e1)
                    block_mask.set(i);
            }
        }

        e_pre = has_second_bucket ? blk.e2 : blk.e1;
#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t1 = EXSIA_STAGE_CYCLE_READ();
#endif

        theta_pre = exp_to_theta(e_pre, meta.rho);
        unit_quant_.quantize_block(blk.x, block_mask, theta_pre, q_tmp, S, SS);

        for (size_t i = 0; i < blk_size; ++i)
        {
            const size_t col = blk_idx * blk_size + i;
            if (col < state.K_logical && !block_mask.is_set(i))
                ++unmasked_count;
        }

#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t2 = EXSIA_STAGE_CYCLE_READ();
#endif

        for (size_t i = 0; i < blk_size; ++i)
        {
            const size_t col = blk_idx * blk_size + i;
            if (col >= state.K_logical || block_mask.is_set(i))
                continue;

            if (unit_sigma_.detect_sigma(q_tmp[i], S, SS, unmasked_count))
            {
                block_mask.set(i);
                has_int_outlier = true;
            }
        }

#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t3 = EXSIA_STAGE_CYCLE_READ();
#endif

        if (!has_int_outlier)
        {
            blk.e_b = e_pre;
            blk.theta_b = theta_pre;
            std::copy_n(q_tmp.begin(), blk_size, q_final.begin());
            cycle_sample.p3_path = P3Path::BypassNoIntegerOutlier;
        }
        else
        {
            unit_exp_.update_block_top2_exp(block_mask, blk);
            blk.e_b = blk.e1;
            blk.theta_b = exp_to_theta(blk.e_b, meta.rho);

            if (blk.theta_b == theta_pre)
            {
                std::copy_n(q_tmp.begin(), blk_size, q_final.begin());
                cycle_sample.p3_path = P3Path::BypassSameScale;
            }
            else
            {
                unit_quant_.quantize_block(blk.x, blk.theta_b, q_final);
                cycle_sample.p3_path = P3Path::Replay;
            }
        }

        GGML_ASSERT(stripe_q_wide.size() >= base + blk_size);
        for (size_t i = 0; i < blk_size; ++i)
            stripe_q_wide[base + i] = q_final[i];

        const size_t block_exp_idx = local_row * state.blocks_per_row + blk_idx;
        GGML_ASSERT(stripe_block_exp.size() > block_exp_idx);
        stripe_block_exp[block_exp_idx] = blk.e_b;

#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t t4 = EXSIA_STAGE_CYCLE_READ();
        cycle_sample.p0 = t1 >= t0 ? t1 - t0 : 0;
        cycle_sample.p1 = t2 >= t1 ? t2 - t1 : 0;
        cycle_sample.p2 = t3 >= t2 ? t3 - t2 : 0;
        cycle_sample.p3 = t4 >= t3 ? t4 - t3 : 0;
#endif

        return true;
    }
#endif

    namespace
    {
        bool assemble_stripe_mask(StripePipelineSlot &slot, const ExSIAState &state)
        {
            StripeState &stripe = slot.stripe;
            const size_t active_block_count = stripe.row_count() * state.blocks_per_row;
            if (stripe.outlier_mask.rows != stripe.row_count() ||
                stripe.outlier_mask.cols != state.K_padded ||
                slot.active_block_count != active_block_count)
                return false;

            stripe.outlier_mask.clear_active_bits();
            for (size_t block = 0; block < active_block_count; ++block)
            {
                const size_t local_row = block / state.blocks_per_row;
                const size_t blk_idx = block % state.blocks_per_row;
                const BlockMask block_mask = slot.block_mask(block, state.B_size);
                for (size_t i = 0; i < block_mask.bit_count; ++i)
                {
                    const size_t global_col = blk_idx * state.B_size + i;
                    if (global_col >= state.K_logical || !block_mask.is_set(i))
                        continue;

                    const size_t mask_idx = local_row * stripe.outlier_mask.cols + global_col;
                    stripe.outlier_mask.words[mask_idx / 64] |= uint64_t{1} << (mask_idx % 64);
                }
            }

            return true;
        }

        void reduce_stripe_exponents(StripePipelineSlot &slot, size_t active_block_count)
        {
            StripeState &stripe = slot.stripe;
            stripe.e1 = std::numeric_limits<int16_t>::min();
            stripe.e2 = std::numeric_limits<int16_t>::min();

            GGML_ASSERT(active_block_count <= slot.block_exp.size());
            ExpScanner reducer;
            for (size_t block = 0; block < active_block_count; ++block)
                reducer.update_stripe_top2_exp(stripe, slot.block_exp[block]);
        }
    }

    bool StripeFolding::run(Meta &meta,
                            ExSIAState &state,
                            StripeState &stripe,
                            ggml_gemmini_args_t &args,
                            size_t stripe_idx,
                            int8_t *dst,
                            const std::vector<int32_t> &stripe_q_wide,
                            const std::vector<int16_t> &stripe_block_exp,
                            std::vector<int32_t> &residual,
                            rmd::RmdStripeBuilder &rmd_builder)
    {
        const int16_t neg_inf = std::numeric_limits<int16_t>::min();
        rmd_builder.reset(stripe_idx, stripe.row_start, stripe.row_count(), args.K, args.J);

        if (stripe.e1 == neg_inf)
        {
            stripe.e_s = 0;
            stripe.promote_top_block = false;
        }
        else if (stripe.e2 == neg_inf)
        {
            stripe.e_s = stripe.e1;
            stripe.promote_top_block = false;
        }
        else
        {
            stripe.e_s = stripe.e2;
            stripe.promote_top_block = true;
        }

        const int16_t theta_s = exp_to_theta(stripe.e_s, meta.rho);
        GGML_ASSERT(stripe_idx < meta.theta.size());
        meta.theta[stripe_idx] = theta_s;

        GGML_ASSERT(dst != nullptr);
        GGML_ASSERT(state.B_size > 0);
        GGML_ASSERT(state.K_padded >= args.K);
        GGML_ASSERT(state.blocks_per_row == state.K_padded / state.B_size);
        GGML_ASSERT(stripe.row_start <= stripe.row_end && stripe.row_end <= args.I);
        GGML_ASSERT(stripe_q_wide.size() >= stripe.row_count() * state.K_padded);
        GGML_ASSERT(stripe_block_exp.size() >= stripe.row_count() * state.blocks_per_row);
        GGML_ASSERT(stripe.outlier_mask.rows == stripe.row_count());
        GGML_ASSERT(stripe.outlier_mask.cols >= state.K_padded);
        if (stripe.outlier_mask.rows != stripe.row_count() ||
            stripe.outlier_mask.cols < state.K_padded)
            return false;

        // The slot owns this disjoint row range for both dense int8 and residual writes.
        for (size_t r = stripe.row_start; r < stripe.row_end; ++r)
        {
            GGML_ASSERT(r < args.I);
            const size_t local_row = stripe.local_row(r);

            for (size_t b = 0; b < state.blocks_per_row; ++b)
            {
                const size_t block_offset = b * state.B_size;
                const size_t block_exp_idx = local_row * state.blocks_per_row + b;
                GGML_ASSERT(block_exp_idx < stripe_block_exp.size());

                const int16_t block_exp = stripe_block_exp[block_exp_idx];
                const int16_t delta_theta_b = block_exp == neg_inf || stripe.e_s == neg_inf
                                                  ? 0
                                                  : static_cast<int16_t>(block_exp - stripe.e_s);
                if (stripe.promote_top_block && block_exp == stripe.e1)
                {
                    BitMask &block_inlier_mask = stripe.scratch.folding_inlier_mask;
                    block_inlier_mask.clear_active_bits();

                    for (size_t i = 0; i < state.B_size; ++i)
                    {
                        const size_t col = block_offset + i;
                        if (col < args.K && !stripe.outlier_mask.is_set(local_row, col))
                            block_inlier_mask.set(0, i);
                    }
                    unit_outlier_.mark_outlier(stripe, r, b, state.B_size, block_inlier_mask);
                }

                for (size_t i = 0; i < state.B_size; ++i)
                {
                    const size_t col = block_offset + i;
                    const size_t padded_idx = local_row * state.K_padded + col;
                    const int32_t q_shifted = detail::shift_q_i32(stripe_q_wide[padded_idx], delta_theta_b);
                    const auto [q8, res] = unit_clip_.clip_with_residual(q_shifted);

                    if (col < args.K)
                        dst[r * args.K + col] = q8;

                    const bool outlier = col < args.K && stripe.outlier_mask.is_set(local_row, col);
                    const int32_t residual_i32 = outlier ? res : 0;
                    const size_t global_idx = r * state.K_padded + col;
                    GGML_ASSERT(global_idx < residual.size());
                    residual[global_idx] = residual_i32;

                    // Balanced radix-256 decomposition happens the moment the final
                    // residual exists; no residual list survives this loop.
                    if (outlier && residual_i32 != 0 &&
                        !rmd_builder.add_residual(local_row, col, residual_i32))
                    {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    // Seals the stripe's RMD packet and publishes the shared handle. Called once per
    // stripe, right after folding commits, by the thread that ran folding.
    static bool seal_stripe_packet(Meta &meta, StripePipelineSlot &slot)
    {
        const uint64_t seal_start_ns = steady_now_ns();
        slot.rmd_packet = slot.rmd_builder.finish();
        const uint64_t seal_end_ns = steady_now_ns();
        slot.rmd_pack_ns = seal_end_ns >= seal_start_ns ? seal_end_ns - seal_start_ns : 0;
        if (slot.rmd_packet == nullptr)
            return slot.rmd_builder.status() == rmd::RmdStatus::success;
        meta.rmd_packets.push_back(slot.rmd_packet);
        return true;
    }

    void ExSIA::reset_failure_state()
    {
        first_failure_code_.store(ExSIAState::FailureCode::None, std::memory_order_relaxed);
        first_failure_stripe_.store(ExSIAState::no_failure_stripe, std::memory_order_relaxed);
    }

    void ExSIA::record_failure(ExSIAState::FailureCode code, size_t stripe)
    {
        if (code == ExSIAState::FailureCode::None)
            return;

        ExSIAState::FailureCode expected = ExSIAState::FailureCode::None;
        if (first_failure_code_.compare_exchange_strong(
                expected, code, std::memory_order_acq_rel, std::memory_order_relaxed))
        {
            first_failure_stripe_.store(stripe, std::memory_order_release);
        }
    }

    bool ExSIA::run(
        Meta &meta,
        const ggml_tensor *A,
        ggml_gemmini_args_t &args)
    {
        return run(meta, A, args, nullptr);
    }

    bool ExSIA::run(
        Meta &meta,
        const ggml_tensor *A,
        ggml_gemmini_args_t &args,
        const StripeReadySink *sink)
    {
        const char *layer = ggml::gemmini::types::to_string(args.layer_type);
        const uint64_t run_id = next_exsia_run_id();
        EXSIA_PROFILE_LOG(
        const ProfileConfig profile_config = compile_profile_config();
        )
        EXSIA_PROFILE_COLLECT(
        ProfileInterval run_profile;
        start_profile_interval(run_profile);
        )
        EXSIA_PROFILE_LOG(
        const char *mode = requested_mode_ == ExSIAState::ExecutionMode::Sequential
                               ? "Sequential"
                               : requested_mode_ == ExSIAState::ExecutionMode::LocalParallel
                                     ? "LocalParallel"
                                     : "LocalFoldingPipeline";
        )
        const int16_t invalid_theta = std::numeric_limits<int16_t>::min();
        // Global outputs are dst (dense int8), state_.residual, meta.theta, meta.rmd_packets.
        int8_t *dst = reinterpret_cast<int8_t *>(args.A);
        size_t logical_elem_count = 0;
        const bool logical_elem_count_ok = checked_mul_size(args.I, args.K, logical_elem_count);
        reset_failure_state();
        const auto fail = [&](ExSIAState::FailureCode code = ExSIAState::FailureCode::Exception,
                              size_t stripe = ExSIAState::no_failure_stripe) {
            record_failure(code, stripe);
            const ExSIAState::FailureCode failure_code = first_failure_code_.load(
                std::memory_order_acquire);
            const size_t failure_stripe = first_failure_stripe_.load(std::memory_order_acquire);
            if (dst != nullptr && logical_elem_count_ok)
                std::fill_n(dst, logical_elem_count, int8_t{0});

            for (StripePipelineSlot &slot : pipeline_slots_)
                slot.reset_for_run();
            local_workspace_.reset_for_run();
            meta.reset();
#if EXSIA_VALIDATION && EXSIA_PROFILE_COLLECTION_ENABLED
            const ExSIAState::ProfileSnapshot profile_snapshot = state_.profile_snapshot;
#endif
            state_ = ExSIAState{};
            state_.mode = requested_mode_;
            state_.failure_code = failure_code;
            state_.failure_stripe = failure_stripe;
#if EXSIA_VALIDATION && EXSIA_PROFILE_COLLECTION_ENABLED
            state_.profile_snapshot = profile_snapshot;
#endif
            return false;
        };

        meta.sigma = GGML_GEMMINI_EXSIA_SIGMA;
        for (StripePipelineSlot &slot : pipeline_slots_)
            slot.reset_for_run();
        local_workspace_.reset_for_run();
        state_ = ExSIAState{};
        state_.mode = requested_mode_;
#if !defined(GGML_GEMMINI_HAS_OPENMP)
        if (state_.mode == ExSIAState::ExecutionMode::LocalParallel ||
            state_.mode == ExSIAState::ExecutionMode::LocalFoldingPipeline)
            return fail(ExSIAState::FailureCode::OpenMPUnavailable);
#endif
        if (state_.mode != ExSIAState::ExecutionMode::Sequential &&
            state_.mode != ExSIAState::ExecutionMode::LocalParallel &&
            state_.mode != ExSIAState::ExecutionMode::LocalFoldingPipeline)
        {
            return fail(ExSIAState::FailureCode::InvalidInput);
        }
#if defined(GGML_GEMMINI_HAS_OPENMP)
        if ((state_.mode == ExSIAState::ExecutionMode::LocalParallel ||
             state_.mode == ExSIAState::ExecutionMode::LocalFoldingPipeline) &&
            (omp_in_parallel() || omp_get_active_level() > 0))
        {
            return fail(ExSIAState::FailureCode::ExternalOpenMPRegionUnsupported);
        }
#endif
        state_.B_size = BLOCK_SIZE;
        state_.K_logical = args.K;
        if (dst == nullptr ||
            args.I == 0 || args.K == 0 ||
            !logical_elem_count_ok ||
            !checked_round_up_multiple(args.K, state_.B_size, state_.K_padded))
        {
            return fail(ExSIAState::FailureCode::InvalidInput);
        }

        state_.blocks_per_row = state_.K_padded / state_.B_size;

        size_t rows_per_stripe = args.I;
        if (args.tile_I > 0 && !checked_mul_size(args.tile_I, DIM, rows_per_stripe))
            return fail(ExSIAState::FailureCode::InvalidInput);

        if (rows_per_stripe == 0)
            return fail(ExSIAState::FailureCode::InvalidInput);

        size_t stripe_round_input = 0;
        if (!checked_add_size(args.I, rows_per_stripe - 1, stripe_round_input))
            return fail(ExSIAState::FailureCode::InvalidInput);

        const size_t num_stripes = stripe_round_input / rows_per_stripe;
        const float *src_data = ggml::gemmini::activation_data(A);
        if (!src_data)
            return fail(ExSIAState::FailureCode::InvalidInput);

        size_t max_stripe_rows = std::min(args.I, rows_per_stripe);
        size_t max_stripe_elem_count = 0;
        size_t max_stripe_block_count = 0;
        if (!checked_mul_size(max_stripe_rows, state_.K_padded, max_stripe_elem_count) ||
            !checked_mul_size(max_stripe_rows, state_.blocks_per_row, max_stripe_block_count))
        {
            return fail(ExSIAState::FailureCode::InvalidInput);
        }

        meta.theta.assign(num_stripes, invalid_theta);
        meta.rmd_packets.clear();

        // Dense residual matrix is a global output, sized to the full padded activation.
        // Each stripe writes its own disjoint global row range (K_padded stride).
        size_t padded_elem_count = 0;
        if (!checked_mul_size(args.I, state_.K_padded, padded_elem_count))
            return fail(ExSIAState::FailureCode::InvalidInput);

        release_vector(state_.x_f32);
        release_vector(state_.q_wide);
        release_vector(state_.block_exp);
        state_.residual.assign(padded_elem_count, 0);

        for (StripePipelineSlot &slot : pipeline_slots_)
        {
            if (!slot.prepare(max_stripe_elem_count, max_stripe_block_count,
                              max_stripe_rows, state_.K_padded, state_.B_size))
                return fail(ExSIAState::FailureCode::InvalidInput);
        }
        if (!local_workspace_.prepare(max_stripe_block_count, state_.B_size))
            return fail(ExSIAState::FailureCode::InvalidInput);

        // state_.stripe carries per-stripe row metadata only; the workspace owns the live
        // mask/scratch. (Validation builds additionally snapshot each mask below.)
        state_.stripe.assign(num_stripes, StripeState{});
#if EXSIA_OBSERVATION_ENABLED
        if (state_.mode == ExSIAState::ExecutionMode::LocalParallel ||
            state_.mode == ExSIAState::ExecutionMode::LocalFoldingPipeline)
            state_.local_parallel_observations.resize(num_stripes);
#endif
        for (size_t s = 0; s < num_stripes; ++s)
        {
            StripeState &meta_stripe = state_.stripe[s];
            meta_stripe.row_start = s * rows_per_stripe;
            meta_stripe.row_end = std::min((s + 1) * rows_per_stripe, args.I);
#if EXSIA_VALIDATION
            if (!meta_stripe.outlier_mask.prepare(meta_stripe.row_count(), state_.K_padded))
                return fail(ExSIAState::FailureCode::ValidationSnapshotFailure, s);
#endif
        }

        const auto snapshot_validation_mask = [&](size_t stripe_idx, const BitMask &mask) {
#if EXSIA_VALIDATION
            BitMask &snapshot = state_.stripe[stripe_idx].outlier_mask;
            const size_t word_count = mask.active_word_count();
            if (snapshot.words.size() < word_count)
                return false;
            snapshot.rows = mask.rows;
            snapshot.cols = mask.cols;
            std::copy_n(mask.words.begin(), word_count, snapshot.words.begin());
#else
            (void) stripe_idx;
            (void) mask;
#endif
            return true;
        };
        const auto notify_stripe_ready = [&](StripePipelineSlot &slot, uint64_t run_id
#if EXSIA_PROFILE_COLLECTION_ENABLED
                                             , const StripeProfileRecord *profile
#endif
                                             ) {
            if (sink == nullptr || sink->on_ready == nullptr)
                return true;

            StripeReadyEvent event{};
            event.run_id = run_id;
            event.stripe_id = slot.stripe_idx;
            event.slot = slot.stripe_idx % EXSIA_PIPELINE_SLOT_COUNT;
            event.row_begin = slot.row_start;
            event.row_end = slot.row_end;
            event.rmd_packet = slot.rmd_packet;
            event.rmd_pack_ns = slot.rmd_pack_ns;
#if EXSIA_PROFILE_COLLECTION_ENABLED
            if (profile != nullptr) {
                event.local_start_cycle = profile->local.start;
                event.local_end_cycle = profile->local.end;
                if (profile->local_groups.size() > 3) {
                    event.local_group3_start_cycle = profile->local_groups[3].start;
                    event.local_group3_end_cycle = profile->local_groups[3].end;
                }
                event.folding_start_cycle = profile->folding.start;
                event.folding_end_cycle = profile->folding.end;
                event.local_start_ns = profile->local.start_ns;
                event.local_end_ns = profile->local.end_ns;
                event.folding_start_ns = profile->folding.start_ns;
                event.folding_end_ns = profile->folding.end_ns;
                for (size_t worker = 0; worker < event.local_worker_start_ns.size() &&
                                        worker < profile->local_groups.size(); ++worker) {
                    event.local_worker_start_ns[worker] = profile->local_groups[worker].start_ns;
                    event.local_worker_end_ns[worker] = profile->local_groups[worker].end_ns;
                }
                event.mask_assembly_start_ns = profile->mask_assembly.start_ns;
                event.mask_assembly_end_ns = profile->mask_assembly.end_ns;
                event.exponent_reduction_start_ns = profile->exponent_reduction.start_ns;
                event.exponent_reduction_end_ns = profile->exponent_reduction.end_ns;
                event.folding_commit_ns = slot.folding_commit_ns;
            }
#endif
            const bool accepted = sink->on_ready(
                sink->user_data,
                event);
            return accepted;
        };
#if defined(GGML_GEMMINI_HAS_OPENMP)
#if EXSIA_BRANCH_COUNTS_ENABLED
        const auto record_sample = [](StripeCycleStats &stats,
                                      const LocalBlockCycleSample &sample) {
#if EXSIA_STAGE_PROFILE_ENABLED
            stats.p0.add(sample.p0);
            stats.p1.add(sample.p1);
            stats.p2.add(sample.p2);
            stats.p3.add(sample.p3);
#endif
            switch (sample.p3_path)
            {
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
        const auto run_local_block = [&](StripePipelineSlot &slot,
                                         StripeScratch &scratch,
                                         size_t row,
                                         size_t block EXSIA_STATS_PARAMETER) {
#if EXSIA_BRANCH_COUNTS_ENABLED
            LocalBlockCycleSample sample;
#endif
            const size_t col_offset = block * state_.B_size;
            GGML_ASSERT(col_offset < args.K);
            const size_t valid_count = std::min(state_.B_size, args.K - col_offset);
            const size_t local_row = slot.stripe.local_row(row);
            const size_t block_base = local_row * state_.K_padded + col_offset;
            const size_t block_exp_idx = local_row * state_.blocks_per_row + block;
            GGML_ASSERT(slot.q_wide.size() >= block_base + state_.B_size);
            GGML_ASSERT(block_exp_idx < slot.block_exp.size());
            BlockMask block_mask = slot.block_mask(
                local_row * state_.blocks_per_row + block, state_.B_size);
            if (!local_.run_optimized(meta, state_, src_data + row * args.K + col_offset,
                             valid_count, state_.B_size, local_row, block, scratch, block_mask,
                             slot.q_wide.data() + block_base, slot.block_exp[block_exp_idx]
#if EXSIA_BRANCH_COUNTS_ENABLED
                             , sample
#endif
                             ))
                return false;

#if EXSIA_BRANCH_COUNTS_ENABLED
            record_sample(stats, sample);
#endif
            return true;
        };
#endif

        EXSIA_PROFILE_COLLECT(
        std::vector<StripeProfileRecord> stripe_profiles(num_stripes);
        )
        if (state_.mode == ExSIAState::ExecutionMode::LocalFoldingPipeline)
        {
#if defined(GGML_GEMMINI_HAS_OPENMP)
            std::vector<uint8_t> prepared_storage(num_stripes);
            std::vector<uint8_t> worker_done_storage(num_stripes * EXSIA_LOCAL_WORKER_COUNT);
            std::vector<uint8_t> local_sealed_storage(num_stripes);
            std::vector<uint8_t> slot_released_storage(num_stripes);
            uint8_t *prepared = prepared_storage.data();
            uint8_t *worker_done = worker_done_storage.data();
            uint8_t *local_sealed = local_sealed_storage.data();
            uint8_t *slot_released = slot_released_storage.data();
            uint8_t post_chain = 0;
            std::atomic<bool> pipeline_ok{true};
#pragma omp parallel num_threads(EXSIA_OMP_THREAD_COUNT)
            {
#pragma omp single
                {
                    const size_t observed_team_size = static_cast<size_t>(omp_get_num_threads());
                    if (observed_team_size != EXSIA_OMP_THREAD_COUNT)
                    {
                        record_failure(ExSIAState::FailureCode::WrongTeamSize);
                        pipeline_ok.store(false, std::memory_order_relaxed);
                    }
#pragma omp taskgroup
                    {
                    for (size_t s = 0; s < num_stripes; ++s)
                    {
                        const size_t slot_idx = s % EXSIA_PIPELINE_SLOT_COUNT;
                        const size_t row_start = s * rows_per_stripe;
                        const size_t row_end = std::min((s + 1) * rows_per_stripe, args.I);
                        if (s == 0)
                        {
#pragma omp task depend(out : prepared[s]) firstprivate(s, slot_idx, row_start, row_end, observed_team_size)
                            {
                                try
                                {
                                if (pipeline_ok.load(std::memory_order_relaxed))
                                {
                                StripePipelineSlot &slot = pipeline_slots_[slot_idx];
                                slot.acquire(s);
                                slot.reset_for_stripe(s, row_start, row_end,
                                                     state_.K_padded, state_.blocks_per_row);
                                local_workspace_.reset_for_stripe(
                                    s, row_start, row_end, state_.blocks_per_row);
#if EXSIA_OBSERVATION_ENABLED
                                LocalParallelStripeObservation &observation =
                                    state_.local_parallel_observations[s];
                                observation = LocalParallelStripeObservation{};
                                observation.stripe_idx = s;
                                observation.observed_team_size = observed_team_size;
                                observation.scheduled_task_count = EXSIA_LOCAL_WORKER_COUNT;
                                const size_t total_blocks =
                                    slot.stripe.row_count() * state_.blocks_per_row;
                                const size_t expected_blocks_per_task =
                                    total_blocks / EXSIA_LOCAL_WORKER_COUNT +
                                    (total_blocks % EXSIA_LOCAL_WORKER_COUNT != 0 ? 1 : 0);
                                for (size_t task_id = 0;
                                     task_id < EXSIA_LOCAL_WORKER_COUNT; ++task_id)
                                {
                                    const LocalWorkerContext &worker =
                                        local_workspace_.workers[task_id];
                                    LocalParallelTaskRecord &record = observation.tasks[task_id];
                                    record.task_id = task_id;
                                    record.row_start = worker.row_start;
                                    record.row_end = worker.row_end;
                                    record.block_start = worker.block_start;
                                    record.block_end = worker.block_end;
                                    record.populated_block_count =
                                        worker.block_end - worker.block_start;
                                    record.empty = record.populated_block_count == 0;
                                    record.short_task = !record.empty &&
                                        record.populated_block_count < expected_blocks_per_task;
                                }
#endif
                                 EXSIA_PROFILE_COLLECT(
                                 StripeProfileRecord &profile = stripe_profiles[s];
                                 profile = StripeProfileRecord{};
                                 profile.stripe_idx = s;
                                 profile.row_start = row_start;
                                 profile.row_end = row_end;
                                 profile.team_size = observed_team_size;
                                 start_profile_interval(profile.stripe_total);
                                 start_profile_interval(profile.local);
                                 )
                                 }
                                }
                                catch (...)
                                {
                                    record_failure(ExSIAState::FailureCode::Exception, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
                            }
                        }
                        else if (s == 1)
                        {
#pragma omp task depend(in : local_sealed[s - 1]) depend(out : prepared[s]) firstprivate(s, slot_idx, row_start, row_end, observed_team_size)
                            {
                                try
                                {
                                if (pipeline_ok.load(std::memory_order_relaxed))
                                {
                                StripePipelineSlot &slot = pipeline_slots_[slot_idx];
                                slot.acquire(s);
                                slot.reset_for_stripe(s, row_start, row_end,
                                                     state_.K_padded, state_.blocks_per_row);
                                local_workspace_.reset_for_stripe(
                                    s, row_start, row_end, state_.blocks_per_row);
#if EXSIA_OBSERVATION_ENABLED
                                LocalParallelStripeObservation &observation =
                                    state_.local_parallel_observations[s];
                                observation = LocalParallelStripeObservation{};
                                observation.stripe_idx = s;
                                observation.observed_team_size = observed_team_size;
                                observation.scheduled_task_count = EXSIA_LOCAL_WORKER_COUNT;
                                const size_t total_blocks =
                                    slot.stripe.row_count() * state_.blocks_per_row;
                                const size_t expected_blocks_per_task =
                                    total_blocks / EXSIA_LOCAL_WORKER_COUNT +
                                    (total_blocks % EXSIA_LOCAL_WORKER_COUNT != 0 ? 1 : 0);
                                for (size_t task_id = 0;
                                     task_id < EXSIA_LOCAL_WORKER_COUNT; ++task_id)
                                {
                                    const LocalWorkerContext &worker =
                                        local_workspace_.workers[task_id];
                                    LocalParallelTaskRecord &record = observation.tasks[task_id];
                                    record.task_id = task_id;
                                    record.row_start = worker.row_start;
                                    record.row_end = worker.row_end;
                                    record.block_start = worker.block_start;
                                    record.block_end = worker.block_end;
                                    record.populated_block_count =
                                        worker.block_end - worker.block_start;
                                    record.empty = record.populated_block_count == 0;
                                    record.short_task = !record.empty &&
                                        record.populated_block_count < expected_blocks_per_task;
                                }
#endif
                                 EXSIA_PROFILE_COLLECT(
                                 StripeProfileRecord &profile = stripe_profiles[s];
                                 profile = StripeProfileRecord{};
                                 profile.stripe_idx = s;
                                 profile.row_start = row_start;
                                 profile.row_end = row_end;
                                 profile.team_size = observed_team_size;
                                 start_profile_interval(profile.stripe_total);
                                 start_profile_interval(profile.local);
                                 )
                                 }
                                }
                                catch (...)
                                {
                                    record_failure(ExSIAState::FailureCode::Exception, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
                            }
                        }
                        else
                        {
#pragma omp task depend(in : local_sealed[s - 1], slot_released[s - 2]) depend(out : prepared[s]) firstprivate(s, slot_idx, row_start, row_end, observed_team_size)
                            {
                                try
                                {
                                if (pipeline_ok.load(std::memory_order_relaxed))
                                {
                                StripePipelineSlot &slot = pipeline_slots_[slot_idx];
                                slot.acquire(s);
                                slot.reset_for_stripe(s, row_start, row_end,
                                                     state_.K_padded, state_.blocks_per_row);
                                local_workspace_.reset_for_stripe(
                                    s, row_start, row_end, state_.blocks_per_row);
#if EXSIA_OBSERVATION_ENABLED
                                LocalParallelStripeObservation &observation =
                                    state_.local_parallel_observations[s];
                                observation = LocalParallelStripeObservation{};
                                observation.stripe_idx = s;
                                observation.observed_team_size = observed_team_size;
                                observation.scheduled_task_count = EXSIA_LOCAL_WORKER_COUNT;
                                const size_t total_blocks =
                                    slot.stripe.row_count() * state_.blocks_per_row;
                                const size_t expected_blocks_per_task =
                                    total_blocks / EXSIA_LOCAL_WORKER_COUNT +
                                    (total_blocks % EXSIA_LOCAL_WORKER_COUNT != 0 ? 1 : 0);
                                for (size_t task_id = 0;
                                     task_id < EXSIA_LOCAL_WORKER_COUNT; ++task_id)
                                {
                                    const LocalWorkerContext &worker =
                                        local_workspace_.workers[task_id];
                                    LocalParallelTaskRecord &record = observation.tasks[task_id];
                                    record.task_id = task_id;
                                    record.row_start = worker.row_start;
                                    record.row_end = worker.row_end;
                                    record.block_start = worker.block_start;
                                    record.block_end = worker.block_end;
                                    record.populated_block_count =
                                        worker.block_end - worker.block_start;
                                    record.empty = record.populated_block_count == 0;
                                    record.short_task = !record.empty &&
                                        record.populated_block_count < expected_blocks_per_task;
                                }
#endif
                                 EXSIA_PROFILE_COLLECT(
                                 StripeProfileRecord &profile = stripe_profiles[s];
                                 profile = StripeProfileRecord{};
                                 profile.stripe_idx = s;
                                 profile.row_start = row_start;
                                 profile.row_end = row_end;
                                 profile.team_size = observed_team_size;
                                 start_profile_interval(profile.stripe_total);
                                 start_profile_interval(profile.local);
                                 )
                                 }
                                }
                                catch (...)
                                {
                                    record_failure(ExSIAState::FailureCode::Exception, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
                            }
                        }

                        for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT; ++task_id)
                        {
#pragma omp task depend(in : prepared[s]) depend(out : worker_done[s * EXSIA_LOCAL_WORKER_COUNT + task_id]) firstprivate(s, slot_idx, task_id)
                            {
                                try
                                {
                                if (pipeline_ok.load(std::memory_order_relaxed))
                                {
                                StripePipelineSlot &slot = pipeline_slots_[slot_idx];
                                LocalWorkerContext &worker = local_workspace_.workers[task_id];
                                LocalTaskRuntime &task_runtime = local_workspace_.local_tasks[task_id];
                                EXSIA_PROFILE_COLLECT(
                                StripeProfileRecord &profile = stripe_profiles[s];
                                start_profile_interval(profile.local_groups[task_id]);
                                )
                                bool ok = true;
                                for (size_t block = worker.block_start;
                                     block < worker.block_end; ++block)
                                {
                                    const size_t local_row = block / state_.blocks_per_row;
                                    const size_t block_idx = block % state_.blocks_per_row;
                                    const size_t global_row = slot.stripe.row_start + local_row;
                                    if (!run_local_block(slot, worker.scratch, global_row, block_idx
#if EXSIA_BRANCH_COUNTS_ENABLED
                                                         , task_runtime.cycle_stats
#endif
                                                         ))
                                    {
                                        record_failure(ExSIAState::FailureCode::LocalBlockFailure, s);
                                        pipeline_ok.store(false, std::memory_order_relaxed);
                                        ok = false;
                                        break;
                                    }
                                }
                                task_runtime.completed = ok;
                                EXSIA_PROFILE_COLLECT(
                                if (!end_profile_interval(profile.local_groups[task_id]))
                                {
                                    record_failure(ExSIAState::FailureCode::ProfileIntervalInvalid, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
                                )
                                }
                                }
                                catch (...)
                                {
                                    record_failure(ExSIAState::FailureCode::Exception, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
                            }
                        }

#if GGML_GEMMINI_EXSIA_LOCAL_WORKERS == 3
#pragma omp task depend(in : worker_done[s * EXSIA_LOCAL_WORKER_COUNT], worker_done[s * EXSIA_LOCAL_WORKER_COUNT + 1], worker_done[s * EXSIA_LOCAL_WORKER_COUNT + 2]) depend(out : local_sealed[s]) firstprivate(s, slot_idx)
#elif GGML_GEMMINI_EXSIA_LOCAL_WORKERS == 4
#pragma omp task depend(in : worker_done[s * EXSIA_LOCAL_WORKER_COUNT], worker_done[s * EXSIA_LOCAL_WORKER_COUNT + 1], worker_done[s * EXSIA_LOCAL_WORKER_COUNT + 2], worker_done[s * EXSIA_LOCAL_WORKER_COUNT + 3]) depend(out : local_sealed[s]) firstprivate(s, slot_idx)
#else
#error "Unsupported ExSIA local worker count"
#endif
                        {
                            try
                            {
                            if (pipeline_ok.load(std::memory_order_relaxed))
                            {
                            StripePipelineSlot &slot = pipeline_slots_[slot_idx];
                            (void) slot;
#if EXSIA_OBSERVATION_ENABLED
                            LocalParallelStripeObservation &observation =
                                state_.local_parallel_observations[s];
#endif
#if EXSIA_BRANCH_COUNTS_ENABLED
                            slot.cycle_stats.reset();
#endif
                            bool ok = true;
                            for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT; ++task_id)
                            {
                                const LocalTaskRuntime &task_runtime = local_workspace_.local_tasks[task_id];
#if EXSIA_OBSERVATION_ENABLED
                                observation.completed_task_count += task_runtime.completed ? 1 : 0;
                                observation.tasks[task_id].completed = task_runtime.completed;
#endif
                                ok = ok && task_runtime.completed;
#if EXSIA_BRANCH_COUNTS_ENABLED
                                const StripeCycleStats &task_stats = task_runtime.cycle_stats;
#if EXSIA_STAGE_PROFILE_ENABLED
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
                                slot.cycle_stats.p3_bypass_no_int_count += task_stats.p3_bypass_no_int_count;
                                slot.cycle_stats.p3_bypass_same_scale_count += task_stats.p3_bypass_same_scale_count;
                                slot.cycle_stats.p3_replay_count += task_stats.p3_replay_count;
#endif
                            }
                            if (!ok)
                            {
                                record_failure(ExSIAState::FailureCode::LocalBlockFailure, s);
                                pipeline_ok.store(false, std::memory_order_relaxed);
                            }
                            else
                            {
#if EXSIA_VALIDATION
                                state_.validation_p3_branch_counts[0] += slot.cycle_stats.p3_bypass_no_int_count;
                                state_.validation_p3_branch_counts[1] += slot.cycle_stats.p3_bypass_same_scale_count;
                                state_.validation_p3_branch_counts[2] += slot.cycle_stats.p3_replay_count;
#endif
#if EXSIA_PROFILE_COLLECTION_ENABLED
                                StripeProfileRecord &profile = stripe_profiles[s];
#if EXSIA_STAGE_PROFILE_ENABLED
                                profile.stats = slot.cycle_stats;
#endif
                                // local_total ends at the worker join, before Mask Assembly,
                                // Exponent Reduction, and Folding.
                                if (!end_profile_interval(profile.local))
                                {
                                    record_failure(ExSIAState::FailureCode::ProfileIntervalInvalid, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
#endif
                            }
                            }
                            }
                            catch (...)
                            {
                                record_failure(ExSIAState::FailureCode::Exception, s);
                                pipeline_ok.store(false, std::memory_order_relaxed);
                            }
                        }

#pragma omp task depend(in : local_sealed[s]) depend(inout : post_chain) depend(out : slot_released[s]) firstprivate(s, slot_idx)
                        {
                            try
                            {
                            if (pipeline_ok.load(std::memory_order_relaxed))
                            {
                                StripePipelineSlot &slot = pipeline_slots_[slot_idx];
                                const size_t active_block_count =
                                    slot.stripe.row_count() * state_.blocks_per_row;
                                EXSIA_PROFILE_COLLECT(
                                StripeProfileRecord &profile = stripe_profiles[s];
                                start_profile_interval(profile.mask_assembly);
                                )
                                const bool assembled = assemble_stripe_mask(slot, state_);
                                EXSIA_PROFILE_COLLECT(
                                if (!end_profile_interval(profile.mask_assembly))
                                {
                                    record_failure(ExSIAState::FailureCode::ProfileIntervalInvalid, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
                                )
                                if (!pipeline_ok.load(std::memory_order_relaxed))
                                {
                                }
                                else if (!assembled)
                                {
                                    record_failure(ExSIAState::FailureCode::MaskAssemblyFailure, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
                                else if (active_block_count > slot.block_exp.size())
                                {
                                    record_failure(ExSIAState::FailureCode::ExponentReductionFailure, s);
                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                }
                                else
                                {
                                    EXSIA_PROFILE_COLLECT(start_profile_interval(profile.exponent_reduction);)
                                    reduce_stripe_exponents(slot, active_block_count);
                                    EXSIA_PROFILE_COLLECT(
                                    if (!end_profile_interval(profile.exponent_reduction))
                                    {
                                        record_failure(ExSIAState::FailureCode::ProfileIntervalInvalid, s);
                                        pipeline_ok.store(false, std::memory_order_relaxed);
                                    }
                                    )
                                    slot.mark_local_filled();
                                    if (pipeline_ok.load(std::memory_order_relaxed))
                                    {
                                        EXSIA_PROFILE_COLLECT(start_profile_interval(profile.folding);)
                                        if (!folding_.run(meta, state_, slot.stripe, args, s, dst,
                                                          slot.q_wide, slot.block_exp,
                                                          state_.residual, slot.rmd_builder))
                                        {
                                            record_failure(ExSIAState::FailureCode::FoldingFailure, s);
                                            pipeline_ok.store(false, std::memory_order_relaxed);
                                        }
                                        else if (!seal_stripe_packet(meta, slot))
                                        {
                                            record_failure(ExSIAState::FailureCode::FoldingFailure, s);
                                            pipeline_ok.store(false, std::memory_order_relaxed);
                                        }
                                        else
                                        {
                                            slot.mark_folding_committed(steady_now_ns());
                                            if (!snapshot_validation_mask(s, slot.stripe.outlier_mask))
                                            {
                                                record_failure(ExSIAState::FailureCode::ValidationSnapshotFailure, s);
                                                pipeline_ok.store(false, std::memory_order_relaxed);
                                            }
                                            else
                                            {
                                                EXSIA_PROFILE_COLLECT(
                                                if (!end_profile_interval(profile.folding))
                                                {
                                                    record_failure(ExSIAState::FailureCode::ProfileIntervalInvalid, s);
                                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                                }
                                                )
                                                if (pipeline_ok.load(std::memory_order_relaxed) &&
                                                    !notify_stripe_ready(slot, run_id
#if EXSIA_PROFILE_COLLECTION_ENABLED
                                                                          , &profile
#endif
                                                                          ))
                                                {
                                                    record_failure(ExSIAState::FailureCode::StripeReadySinkFailure, s);
                                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                                }
                                                else
                                                {
                                                slot.release();
                                                EXSIA_PROFILE_COLLECT(
                                                if (!end_profile_interval(profile.stripe_total))
                                                {
                                                    record_failure(ExSIAState::FailureCode::ProfileIntervalInvalid, s);
                                                    pipeline_ok.store(false, std::memory_order_relaxed);
                                                }
                                                )
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            }
                            catch (...)
                            {
                                record_failure(ExSIAState::FailureCode::Exception, s);
                                pipeline_ok.store(false, std::memory_order_relaxed);
                            }
                        }
                    }
                    }
                }
            }
            if (!pipeline_ok.load(std::memory_order_relaxed))
                return fail();
#else
            return fail(ExSIAState::FailureCode::OpenMPUnavailable);
#endif
        }
        else
        {
        for (size_t s = 0; s < num_stripes; ++s)
        {
            const size_t row_start = s * rows_per_stripe;
            const size_t row_end = std::min((s + 1) * rows_per_stripe, args.I);
            StripePipelineSlot &slot = pipeline_slots_[s % EXSIA_PIPELINE_SLOT_COUNT];
            slot.acquire(s);
            slot.reset_for_stripe(s, row_start, row_end,
                                  state_.K_padded, state_.blocks_per_row);
            local_workspace_.reset_for_stripe(s, row_start, row_end, state_.blocks_per_row);
            StripeState &stripe = slot.stripe;
            EXSIA_PROFILE_COLLECT(
            StripeProfileRecord &profile = stripe_profiles[s];
            profile = StripeProfileRecord{};
            profile.stripe_idx = s;
            profile.row_start = row_start;
            profile.row_end = row_end;
            profile.team_size = 1;
            start_profile_interval(profile.stripe_total);
            start_profile_interval(profile.local);
            )
#if EXSIA_BRANCH_COUNTS_ENABLED
            const auto record_sample = [](StripeCycleStats &stats,
                                          const LocalBlockCycleSample &sample) {
#if EXSIA_STAGE_PROFILE_ENABLED
                stats.p0.add(sample.p0);
                stats.p1.add(sample.p1);
                stats.p2.add(sample.p2);
                stats.p3.add(sample.p3);
#endif
                switch (sample.p3_path)
                {
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
            const auto run_local_block = [&](StripeScratch &scratch,
                                             size_t r,
                                             size_t b EXSIA_STATS_PARAMETER) {
#if EXSIA_BRANCH_COUNTS_ENABLED
                LocalBlockCycleSample sample;
#endif
                const size_t col_offset = b * state_.B_size;
                GGML_ASSERT(col_offset < args.K);
                const size_t valid_count = std::min(state_.B_size, args.K - col_offset);
                const size_t local_row = stripe.local_row(r);
                const size_t block_base = local_row * state_.K_padded + col_offset;
                const size_t block_exp_idx = local_row * state_.blocks_per_row + b;
                GGML_ASSERT(slot.q_wide.size() >= block_base + state_.B_size);
                GGML_ASSERT(block_exp_idx < slot.block_exp.size());
                BlockMask block_mask = slot.block_mask(
                    local_row * state_.blocks_per_row + b, state_.B_size);
                if (!local_.run_optimized(meta, state_, src_data + r * args.K + col_offset,
                                 valid_count, state_.B_size, local_row, b, scratch, block_mask,
                                 slot.q_wide.data() + block_base, slot.block_exp[block_exp_idx]
#if EXSIA_BRANCH_COUNTS_ENABLED
                                 , sample
#endif
                                 ))
                {
                    return false;
                }

#if EXSIA_BRANCH_COUNTS_ENABLED
                record_sample(stats, sample);
#endif
                return true;
            };

            if (state_.mode == ExSIAState::ExecutionMode::LocalParallel)
            {
#if defined(GGML_GEMMINI_HAS_OPENMP)
#if EXSIA_OBSERVATION_ENABLED
                LocalParallelStripeObservation &observation = state_.local_parallel_observations[s];
                observation = LocalParallelStripeObservation{};
                observation.stripe_idx = s;
                observation.scheduled_task_count = EXSIA_LOCAL_WORKER_COUNT;
                const size_t total_blocks = stripe.row_count() * state_.blocks_per_row;
                const size_t expected_blocks_per_task =
                    total_blocks / EXSIA_LOCAL_WORKER_COUNT +
                    (total_blocks % EXSIA_LOCAL_WORKER_COUNT != 0 ? 1 : 0);
                for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT; ++task_id)
                {
                    const LocalWorkerContext &worker = local_workspace_.workers[task_id];
                    LocalParallelTaskRecord &record = observation.tasks[task_id];
                    record.task_id = task_id;
                    record.row_start = worker.row_start;
                    record.row_end = worker.row_end;
                    record.block_start = worker.block_start;
                    record.block_end = worker.block_end;
                    record.populated_block_count = worker.block_end - worker.block_start;
                    record.empty = record.populated_block_count == 0;
                    record.short_task = !record.empty &&
                                        record.populated_block_count < expected_blocks_per_task;
                }
#endif

                std::atomic<bool> local_parallel_ok{true};
                size_t observed_team_size = 0;
#pragma omp parallel num_threads(EXSIA_OMP_THREAD_COUNT)
                {
#pragma omp single
                    {
#if EXSIA_OBSERVATION_ENABLED
                        observation.observed_team_size = static_cast<size_t>(omp_get_num_threads());
                        observed_team_size = observation.observed_team_size;
#else
                        observed_team_size = static_cast<size_t>(omp_get_num_threads());
#endif
                        if (observed_team_size != EXSIA_OMP_THREAD_COUNT)
                        {
                            record_failure(ExSIAState::FailureCode::WrongTeamSize, s);
                            local_parallel_ok.store(false, std::memory_order_relaxed);
                        }
                        for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT; ++task_id)
                        {
#pragma omp task firstprivate(task_id)
                            {
                                try
                                {
                                if (local_parallel_ok.load(std::memory_order_relaxed))
                                {
                                LocalWorkerContext &worker = local_workspace_.workers[task_id];
                                LocalTaskRuntime &task_runtime = local_workspace_.local_tasks[task_id];
                                EXSIA_PROFILE_COLLECT(start_profile_interval(profile.local_groups[task_id]);)
                                bool ok = true;
                                for (size_t block = worker.block_start;
                                     block < worker.block_end; ++block)
                                {
                                    const size_t local_row = block / state_.blocks_per_row;
                                    const size_t block_idx = block % state_.blocks_per_row;
                                    const size_t global_row = stripe.row_start + local_row;
                                    if (!run_local_block(worker.scratch, global_row, block_idx
#if EXSIA_BRANCH_COUNTS_ENABLED
                                                         , task_runtime.cycle_stats
#endif
                                                         ))
                                    {
                                        record_failure(ExSIAState::FailureCode::LocalBlockFailure, s);
                                        local_parallel_ok.store(false, std::memory_order_relaxed);
                                        ok = false;
                                        break;
                                    }
                                }
                                task_runtime.completed = ok;
                                EXSIA_PROFILE_COLLECT(
                                if (!end_profile_interval(profile.local_groups[task_id]))
                                {
                                    record_failure(ExSIAState::FailureCode::ProfileIntervalInvalid, s);
                                    local_parallel_ok.store(false, std::memory_order_relaxed);
                                }
                                )
                                }
                                }
                                catch (...)
                                {
                                    record_failure(ExSIAState::FailureCode::Exception, s);
                                    local_parallel_ok.store(false, std::memory_order_relaxed);
                                }
                            }
                        }
#pragma omp taskwait
                    }
                }

                EXSIA_PROFILE_COLLECT(profile.team_size = observed_team_size;)

                if (!local_parallel_ok.load(std::memory_order_relaxed))
                    return fail();

#if EXSIA_BRANCH_COUNTS_ENABLED
                slot.cycle_stats.reset();
#endif
                for (size_t task_id = 0; task_id < EXSIA_LOCAL_WORKER_COUNT; ++task_id)
                {
                    const LocalTaskRuntime &task_runtime = local_workspace_.local_tasks[task_id];
#if EXSIA_OBSERVATION_ENABLED
                    observation.completed_task_count += task_runtime.completed ? 1 : 0;
                    observation.tasks[task_id].completed = task_runtime.completed;
#endif
                    if (!task_runtime.completed)
                        return fail();

#if EXSIA_BRANCH_COUNTS_ENABLED
                    const StripeCycleStats &task_stats = task_runtime.cycle_stats;
#if EXSIA_STAGE_PROFILE_ENABLED
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
                    slot.cycle_stats.p3_bypass_no_int_count += task_stats.p3_bypass_no_int_count;
                    slot.cycle_stats.p3_bypass_same_scale_count += task_stats.p3_bypass_same_scale_count;
                    slot.cycle_stats.p3_replay_count += task_stats.p3_replay_count;
#endif
                }
#else
                return fail(ExSIAState::FailureCode::OpenMPUnavailable);
#endif
            }
            else
            {
                for (size_t r = stripe.row_start; r < stripe.row_end; ++r)
                {
                    for (size_t b = 0; b < state_.blocks_per_row; ++b)
                    {
                        if (!run_local_block(stripe.scratch, r, b
                                             EXSIA_STATS_ARGUMENT(slot.cycle_stats)))
                            return fail(ExSIAState::FailureCode::LocalBlockFailure, s);
                    }
                }
            }
#if EXSIA_VALIDATION
            state_.validation_p3_branch_counts[0] += slot.cycle_stats.p3_bypass_no_int_count;
            state_.validation_p3_branch_counts[1] += slot.cycle_stats.p3_bypass_same_scale_count;
            state_.validation_p3_branch_counts[2] += slot.cycle_stats.p3_replay_count;
#endif
            // local_total ends after Local and before Mask Assembly, Exponent Reduction, and Folding.
            EXSIA_PROFILE_COLLECT(
            if (!end_profile_interval(profile.local))
                return fail(ExSIAState::FailureCode::ProfileIntervalInvalid, s);
            )
            const size_t active_block_count = stripe.row_count() * state_.blocks_per_row;
            EXSIA_PROFILE_COLLECT(start_profile_interval(profile.mask_assembly);)
            const bool assembled = assemble_stripe_mask(slot, state_);
            EXSIA_PROFILE_COLLECT(
            if (!end_profile_interval(profile.mask_assembly))
                return fail(ExSIAState::FailureCode::ProfileIntervalInvalid, s);
            )
            if (!assembled)
                return fail(ExSIAState::FailureCode::MaskAssemblyFailure, s);
            if (active_block_count > slot.block_exp.size())
                return fail(ExSIAState::FailureCode::ExponentReductionFailure, s);
            EXSIA_PROFILE_COLLECT(start_profile_interval(profile.exponent_reduction);)
            reduce_stripe_exponents(slot, active_block_count);
            EXSIA_PROFILE_COLLECT(
            if (!end_profile_interval(profile.exponent_reduction))
                return fail(ExSIAState::FailureCode::ProfileIntervalInvalid, s);
            )
            slot.mark_local_filled();
            EXSIA_PROFILE_COLLECT(start_profile_interval(profile.folding);)

            if (!folding_.run(meta, state_, stripe, args, s, dst,
                               slot.q_wide, slot.block_exp,
                               state_.residual, slot.rmd_builder))
                return fail(ExSIAState::FailureCode::FoldingFailure, s);

            // Seal the stripe packet and hand the shared handle to the metadata. Stripes
            // run in row order, so meta.rmd_packets stays ordered by row_begin.
            if (!seal_stripe_packet(meta, slot))
                return fail(ExSIAState::FailureCode::FoldingFailure, s);
            slot.mark_folding_committed(steady_now_ns());

            if (!snapshot_validation_mask(s, slot.stripe.outlier_mask))
                return fail(ExSIAState::FailureCode::ValidationSnapshotFailure, s);

            EXSIA_PROFILE_COLLECT(
            if (!end_profile_interval(profile.folding))
                return fail(ExSIAState::FailureCode::ProfileIntervalInvalid, s);
            )
            if (!notify_stripe_ready(slot, run_id
#if EXSIA_PROFILE_COLLECTION_ENABLED
                                     , &profile
#endif
                                     ))
                return fail(ExSIAState::FailureCode::StripeReadySinkFailure, s);
#if EXSIA_STAGE_PROFILE_ENABLED
            profile.stats = slot.cycle_stats;
#endif
            slot.release();
            EXSIA_PROFILE_COLLECT(
            if (!end_profile_interval(profile.stripe_total))
                return fail(ExSIAState::FailureCode::ProfileIntervalInvalid, s);
            )
        }
        }

#if EXSIA_PROFILE_COLLECTION_ENABLED
        if (!end_profile_interval(run_profile))
            return fail(ExSIAState::FailureCode::ProfileIntervalInvalid);
#if EXSIA_VALIDATION
        state_.profile_snapshot.run_id = run_id;
        state_.profile_snapshot.mode = state_.mode;
        state_.profile_snapshot.run = run_profile;
        state_.profile_snapshot.stripes = stripe_profiles;
#endif
#endif
        EXSIA_PROFILE_LOG(
        const ExSIAState::FailureCode profile_failure = flush_profile(
            profile_config, layer, run_id, mode, stripe_profiles, run_profile);
        if (profile_failure != ExSIAState::FailureCode::None)
            return fail(profile_failure);
        )
        ggml::gemmini::log::debug(
            layer,
            "[exsia] I=%zu K=%zu stripes=%zu tau=%d rmd_packets=%zu",
            args.I,
            args.K,
            num_stripes,
            meta.sigma,
            meta.rmd_packets.size());

        return true;
    }

    bool dequantize_activation(
        float *dst,
        size_t dst_row_stride,
        size_t dst_col_stride,
        size_t rows,
        size_t cols,
        const ggml_gemmini_args_t &args)
    {
        const auto run = [&]() -> bool
        {
            const int8_t *src = reinterpret_cast<const int8_t *>(args.A);
            if (!src || !dst || args.I == 0 || args.K == 0 ||
                dst_row_stride == 0 || dst_col_stride == 0 ||
                rows == 0 || cols == 0)
            {
                return false;
            }

            const auto *meta_ptr = std::get_if<Meta>(&args.act_quant.storage());
            if (!meta_ptr)
            {
                return false;
            }
            const Meta &meta = *meta_ptr;

            size_t rows_per_stripe = args.I;
            if (args.tile_I > 0 && !checked_mul_size(args.tile_I, DIM, rows_per_stripe))
            {
                return false;
            }
            if (rows_per_stripe == 0)
            {
                return false;
            }

            if (args.sA != 0 && args.sA != args.K)
            {
                return false;
            }

            const size_t src_row_stride = args.K;
            const size_t row_count = std::min(rows, args.I);
            const size_t col_count = std::min(cols, args.K);
            const size_t max_size = std::numeric_limits<size_t>::max();
            if (row_count != 0 && col_count > max_size / row_count)
            {
                return false;
            }

            std::vector<int32_t> residuals;
            rmd::expand_packets_to_plane(meta.rmd_packets, row_count, col_count, residuals);

            const int16_t invalid_theta = std::numeric_limits<int16_t>::min();
            for (size_t row = 0; row < row_count; ++row)
            {
                const size_t stripe_idx = row / rows_per_stripe;
                const int16_t theta = meta.resolve_stripe_theta(static_cast<int>(stripe_idx));
                if (theta == invalid_theta)
                {
                    return false;
                }

                for (size_t col = 0; col < col_count; ++col)
                {
                    if ((row != 0 && src_row_stride > max_size / row) ||
                        (row != 0 && dst_row_stride > max_size / row) ||
                        (col != 0 && dst_col_stride > max_size / col))
                    {
                        return false;
                    }

                    const size_t src_row_offset = row * src_row_stride;
                    if (src_row_offset > max_size - col)
                    {
                        return false;
                    }

                    const size_t src_idx = src_row_offset + col;
                    const size_t dst_row_offset = row * dst_row_stride;
                    const size_t dst_col_offset = col * dst_col_stride;
                    if (dst_row_offset > max_size - dst_col_offset)
                    {
                        return false;
                    }

                    const int32_t q_int =
                        static_cast<int32_t>(src[src_idx]) +
                        residuals[row * col_count + col];
                    dst[dst_row_offset + dst_col_offset] =
                        std::ldexp(static_cast<float>(q_int), theta);
                }
            }

            return true;
        };

        return run();
    }

}
