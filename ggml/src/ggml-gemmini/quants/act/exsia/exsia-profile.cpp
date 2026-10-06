#include "exsia-profile.hpp"
#include <gemmini/trace-context.hpp>
#include "../../../../ggml-gemmini-utils/src/trace-metadata.hpp"

#include "ggml-gemmini-args.h"
#include "../../../ggml-gemmini-telemetry.hpp"
#include <gemmini/log.h>
#include <gemmini/log.hpp>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <utility>
#if defined(GGML_GEMMINI_HAS_OPENMP)
#include <omp.h>
#endif

namespace ggml::gemmini::quants::act::exsia {
using namespace detail;

namespace detail {
#if LOG_CYCLE
void write_json_string(std::ostream & out, const std::string & value);
#endif
#if EXSIA_PROFILE_LOG_ENABLED
static inline std::filesystem::path cycle_detail_log_path() {
    const char * path = std::getenv("GGML_GEMMINI_CYCLE_DETAIL_LOG");
    return ggml::gemmini::log::resolve_output_path(
        path && path[0] ? path : GEMMINI_LOG_DEFAULT_EXSIA_DETAIL_PATH);
}

static std::once_flag profile_log_init_once;
static bool           profile_log_setup_ok = false;

ProfileConfig compile_profile_config() {
    ProfileConfig config;
    if (const char * path = std::getenv("GGML_GEMMINI_CYCLE_DETAIL_LOG"); path && path[0])
        config.requested_path = path;
    config.log_path = cycle_detail_log_path().string();
    std::call_once(profile_log_init_once, [&config] {
        const std::filesystem::path requested(config.requested_path);
        std::error_code             ec;
        if (requested.is_relative() && !requested.parent_path().empty()) {
            const std::filesystem::file_status status =
                std::filesystem::status(requested.parent_path(), ec);
            if ((ec && ec != std::errc::no_such_file_or_directory) ||
                (std::filesystem::exists(status) && !std::filesystem::is_directory(status)))
                return;
        }
        const std::filesystem::path path(config.log_path);
        if (!ggml::gemmini::log::prepare_output_parent(path))
            return;
        std::ofstream file(path, std::ios::out | std::ios::trunc);
        profile_log_setup_ok = file.good();
    });
    config.setup_ok = profile_log_setup_ok;
    return config;
}
#endif

CpuWallInterval::CpuWallInterval(
    const char * layer, uint64_t run_id, const char * op, uint64_t stripe, uint64_t node) {
#if LOG_CYCLE
    identity_ = cpu_identity(layer, run_id, op, stripe, node);
#else
    (void)layer;
    (void)run_id;
    (void)op;
    (void)stripe;
    (void)node;
#endif
    resume();
}

void CpuWallInterval::pause() {
#if LOG_CYCLE
    if (start_.ns != 0) {
        const auto end = gemmini_cpu_timing_read();
        performance::record_cpu_wall(start_.ns, end.ns);
        gemmini_cpu_timing_record(&identity_, &start_, &end);
        start_ = {};
    }
#endif
}

void CpuWallInterval::resume() {
#if LOG_CYCLE
    start_ = gemmini_cpu_timing_read();
#endif
}

void CpuWallInterval::next(const char * op) {
    pause();
#if LOG_CYCLE
    identity_.interval.op = op;
#else
    (void)op;
#endif
    resume();
}

CpuWallInterval::~CpuWallInterval() {
    pause();
}

#if LOG_CYCLE
void ExsiaRunTiming::submission(const StripeReadyEvent &   event,
                                const gemmini_cpu_sample & begin,
                                const gemmini_cpu_sample & end,
                                bool                       accepted) {
    const uint64_t elapsed = end.ns - begin.ns;
    const bool     measured =
        event.submission_wait_ns.has_value() && *event.submission_wait_ns <= elapsed;
    ++handoff_calls;
    handoff_ns += elapsed;
    if (measured) {
        ++wait_measured_calls;
        wait_ns += *event.submission_wait_ns;
    }
    const auto identity = cpu_identity(layer, run_id, "exsia.submission_callback", event.stripe_id);
    gemmini_cpu_timing_record(&identity, &begin, &end);
#if CYCLE_DETAIL
    log::CycleRecord record{};
    record.layer         = layer;
    record.op            = "exsia.stripe_submission";
    record.source        = "steady_clock";
    record.unit          = "nanosecond";
    record.start         = begin.ns;
    record.end           = end.ns;
    record.identity_mask = GEMMINI_CYCLE_HAS_RUN_ID | GEMMINI_CYCLE_HAS_STRIPE_ID;
    record.identity_mask |= GEMMINI_CYCLE_HAS_SLOT;
    record.run_id    = run_id;
    record.stripe_id = event.stripe_id;
    record.slot      = event.slot;
    std::string tail =
        std::string(",\"additive\":false,\"operation_success\":") + (accepted ? "true" : "false") +
        ",\"host_timing\":" + cycle::serialize_host_timing(begin.ns, end.ns, begin.tid, end.tid) +
        ",\"submission_wait_ns\":" +
        (measured ? std::to_string(*event.submission_wait_ns) : "null") +
        ",\"handoff_nonwait_ns\":" +
        (measured ? std::to_string(elapsed - *event.submission_wait_ns) : "null");
    const auto origin =
        begin.trace.flags & GEMMINI_TRACE_CAPTURED ? begin.trace : gemmini_trace_capture();
    tail += trace::metadata_suffix(trace::origin_fields(
        {{"record_type", "CYCLE_INTERVAL"},
         {"source", "steady_clock"},
         {"host_timing", trace::host_timing_facts(begin.ns, end.ns, begin.tid, end.tid)}},
        origin,
        gemmini_trace_reserve_ids(1)));
    log::cycle.write_decorated(log::serialize_cycle_record(record, tail));
#else
    (void)accepted;
#endif
}

ExsiaRunTiming::~ExsiaRunTiming() {
    const auto end      = gemmini_cpu_timing_read();
    const auto identity = cpu_identity(layer, run_id, "exsia.run.caller");
    gemmini_cpu_timing_record(&identity, &start, &end);
#if CYCLE_DETAIL
    gemmini_cpu_totals cpu_workers{};
    gemmini_cpu_timing_add(&cpu_workers, &start, &end);
    for (const auto & worker : worker_cpu)
        gemmini_cpu_timing_merge(&cpu_workers, &worker);
    const uint64_t     elapsed  = end.ns - start.ns;
    const bool         complete = handoff_calls == wait_measured_calls;
    std::ostringstream out;
    out << "{\"schema\":\"gemmini.cycle\",\"version\":2,"
        << "\"record_type\":\"EXSIA_RUN_SUMMARY\",\"op\":\"exsia.run.summary\","
        << "\"source\":\"steady_clock\",\"unit\":\"nanosecond\",\"layer\":";
    write_json_string(out, layer);
    out << ",\"run_id\":" << run_id
        << ",\"host_timing\":" << cycle::serialize_host_timing(start.ns, end.ns, start.tid, end.tid)
        << ",\"cpu_workers\":" << cycle::serialize_cpu_totals(cpu_workers)
        << ",\"run_wall_ns\":" << elapsed << ",\"handoff_wall_ns\":" << handoff_ns
        << ",\"outside_handoff_wall_ns\":" << elapsed - handoff_ns
        << ",\"submission_wait_ns\":" << (complete ? std::to_string(wait_ns) : "null")
        << ",\"handoff_nonwait_ns\":" << (complete ? std::to_string(handoff_ns - wait_ns) : "null")
        << ",\"handoff_calls\":" << handoff_calls
        << ",\"wait_measured_calls\":" << wait_measured_calls
        << ",\"operation_success\":" << (success ? "true" : "false")
        << ",\"valid\":true,\"additive\":false";
    const auto origin =
        start.trace.flags & GEMMINI_TRACE_CAPTURED ? start.trace : gemmini_trace_capture();
    out << trace::metadata_suffix(trace::origin_fields(
               {{"record_type", "EXSIA_RUN_SUMMARY"}}, origin, gemmini_trace_reserve_ids(1)))
        << '}';
    log::cycle.write_decorated(out.str());
#endif
}
#endif

uint64_t aggregate_now_ns() {
#if LOG_CYCLE && CYCLE_DETAIL
    return ggml::gemmini::cycle::timestamp_ns();
#else
    return 0;
#endif
}

uint64_t aggregate_now_tick() {
#if LOG_CYCLE
    return ggml::gemmini::cycle::read();
#else
    return 0;
#endif
}

#if LOG_CYCLE
uint64_t cpu_worker_id() {
#if defined(GGML_GEMMINI_HAS_OPENMP)
    return static_cast<uint64_t>(omp_get_thread_num());
#else
    return 0;
#endif
}

gemmini_cycle_record_v2
cpu_identity(const char * layer, uint64_t run_id, const char * op, uint64_t stripe, uint64_t node) {
    gemmini_cycle_record_v2 identity{};
    identity.interval.layer = layer;
    identity.interval.op    = op;
    identity.identity_mask  = GEMMINI_CYCLE_HAS_RUN_ID | GEMMINI_CYCLE_HAS_WORKER_ID;
    identity.run_id         = run_id;
    identity.worker_id      = cpu_worker_id();
    if (stripe != UINT64_MAX) {
        identity.identity_mask |= GEMMINI_CYCLE_HAS_STRIPE_ID | GEMMINI_CYCLE_HAS_SLOT;
        identity.stripe_id = stripe;
        identity.slot      = stripe % EXSIA_PIPELINE_SLOT_COUNT;
    }
    if (node != UINT64_MAX) {
        identity.identity_mask |= GEMMINI_CYCLE_HAS_NODE_ID;
        identity.node_id = node;
    }
    return identity;
}
#endif

#if EXSIA_PROFILE_COLLECTION_ENABLED
#if !defined(__linux__) || !defined(__aarch64__)
uint64_t profile_now() {
    return ggml::gemmini::cycle::read();
}
#endif

uint64_t profile_now_ns() {
    return aggregate_now_ns();
}

uint64_t profile_thread_id() {
#if defined(GGML_GEMMINI_HAS_OPENMP)
    return omp_in_parallel() ? static_cast<uint64_t>(omp_get_thread_num()) : 0;
#else
    return 0;
#endif
}

void start_profile_interval(ProfileInterval &                            interval,
                            [[maybe_unused]] const ggml_gemmini_args_t * args,
                            [[maybe_unused]] const char *                operation,
                            [[maybe_unused]] uint64_t                    stripe_id,
                            [[maybe_unused]] std::vector<uint64_t>       dependencies) {
#if CYCLE_SIM
    if (args && operation && args->cycle_sim_context) {
        dependencies.insert(dependencies.end(),
                            args->cycle_sim_host_dependencies.begin(),
                            args->cycle_sim_host_dependencies.end());
        interval.host_stage = args->cycle_sim_context.session->host_stage_begin(
            args->cycle_sim_context,
            {operation,
             "POTAL_HOST",
             "llama.cpp-gemmini",
             "ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:ExSIA::run",
             {},
             std::move(dependencies)});
        const cycle_sim::ScopedContext scope(interval.host_stage);
        interval.correlation              = log::current_cpu_correlation();
        interval.correlation.worker_count = 1;
        interval.host_operation           = operation;
        interval.host_layer               = args->matmul_layer;
        interval.stripe_id                = stripe_id;
#if LOG_CYCLE
        interval.host_start_sample = gemmini_cpu_timing_read();
#endif
    }
#endif
    interval.valid           = true;
    interval.start_thread_id = profile_thread_id();
    interval.start_tid       = ggml::gemmini::cycle::host_thread_id();
#if defined(__linux__) && defined(__aarch64__)
    interval.start_sample = ggml::gemmini::cycle::read_sample();
    interval.start        = interval.start_sample.value;
#else
    interval.start = profile_now();
#endif
    interval.start_ns = profile_now_ns();
}

[[maybe_unused]] std::vector<uint64_t> profile_host_stage_ids(const StripeProfileRecord & profile) {
    std::vector<uint64_t> result;
#if CYCLE_SIM
    const auto add = [&](const ProfileInterval & interval) {
        if (interval.host_stage.host_stage_id)
            result.push_back(*interval.host_stage.host_stage_id);
    };
    add(profile.local);
    for (const auto & group : profile.local_groups)
        add(group);
    add(profile.mask_assembly);
    add(profile.exponent_reduction);
    add(profile.folding);
#else
    (void)profile;
#endif
    return result;
}

bool end_profile_interval(ProfileInterval & interval) {
    assert(interval.valid);
    if (!interval.valid)
        return false;
#if defined(__linux__) && defined(__aarch64__)
    interval.end_sample = ggml::gemmini::cycle::read_sample();
    interval.end        = interval.end_sample.value;
#else
    interval.end = profile_now();
#endif
    interval.end_ns        = profile_now_ns();
    interval.end_tid       = ggml::gemmini::cycle::host_thread_id();
    interval.end_thread_id = profile_thread_id();
#if CYCLE_SIM
    if (interval.host_stage) {
#if LOG_CYCLE
        const cycle_sim::ScopedContext scope(interval.host_stage);
        const auto                     host_end_sample = gemmini_cpu_timing_read();
        gemmini_cycle_record_v2        record{};
        record.interval.layer = interval.host_layer.c_str();
        record.interval.op    = interval.host_operation;
        record.identity_mask  = GEMMINI_CYCLE_HAS_WORKER_ID;
        record.worker_id      = 0;
        if (interval.stripe_id != UINT64_MAX) {
            record.identity_mask |= GEMMINI_CYCLE_HAS_STRIPE_ID;
            record.stripe_id = interval.stripe_id;
        }
        log::cycle.write_cpu(record,
                             interval.host_start_sample,
                             host_end_sample,
                             true,
                             false,
                             false,
                             cycle::TimingIntervalClass::canonical_additive);
#endif
        interval.host_stage.session->host_stage_end(interval.host_stage);
    }
#endif
#if defined(__linux__) && defined(__aarch64__)
    return true;
#else
    return interval.end >= interval.start;
#endif
}
#endif

#if LOG_CYCLE
void write_json_string(std::ostream & out, const std::string & value) {
    out.put('"');
    for (const char character : value) {
        switch (character) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(character) < 0x20) {
                const char hex[] = "0123456789abcdef";
                out << "\\u00" << hex[(character >> 4) & 0xf] << hex[character & 0xf];
            } else {
                out.put(character);
            }
            break;
        }
    }
    out.put('"');
}

#endif

} // namespace detail

#if EXSIA_PROFILE_COLLECTION_ENABLED
ProfileCycleValue checked_profile_interval(const ProfileInterval & interval,
                                           bool structurally_same_owner_eligible) noexcept {
    if (!interval.valid)
        return {{},
                ProfileCycleStatus::missing_component
#if defined(__linux__) && defined(__aarch64__)
                ,
                ggml::gemmini::cycle::NativeCycleReason::none
#endif
        };
#if defined(__linux__) && defined(__aarch64__)
    const ggml::gemmini::cycle::NativeCycleDelta delta = ggml::gemmini::cycle::evaluate_interval(
        interval.start_sample, interval.end_sample, structurally_same_owner_eligible);
    if (delta.valid)
        return {delta.value, ProfileCycleStatus::complete, delta.sample_reason};
    ProfileCycleStatus status = ProfileCycleStatus::counter_regression;
    switch (delta.reason) {
    case ggml::gemmini::cycle::NativeCycleReason::invalid_start:
        status = ProfileCycleStatus::invalid_start;
        break;
    case ggml::gemmini::cycle::NativeCycleReason::invalid_end:
        status = ProfileCycleStatus::invalid_end;
        break;
    case ggml::gemmini::cycle::NativeCycleReason::source_mismatch:
        status = ProfileCycleStatus::source_mismatch;
        break;
    case ggml::gemmini::cycle::NativeCycleReason::event_owner_mismatch:
        status = ProfileCycleStatus::event_owner_mismatch;
        break;
    case ggml::gemmini::cycle::NativeCycleReason::event_generation_mismatch:
        status = ProfileCycleStatus::event_generation_mismatch;
        break;
    case ggml::gemmini::cycle::NativeCycleReason::structurally_cross_task:
        status = ProfileCycleStatus::structurally_cross_task;
        break;
    case ggml::gemmini::cycle::NativeCycleReason::counter_regression:
        status = ProfileCycleStatus::counter_regression;
        break;
    case ggml::gemmini::cycle::NativeCycleReason::none:
    case ggml::gemmini::cycle::NativeCycleReason::unavailable_event:
    case ggml::gemmini::cycle::NativeCycleReason::unavailable_direct_mapping:
    case ggml::gemmini::cycle::NativeCycleReason::multiplexed:
    case ggml::gemmini::cycle::NativeCycleReason::seqlock_exhausted:
        break;
    }
    return {{}, status, delta.sample_reason};
#else
    if (!structurally_same_owner_eligible)
        return {{}, ProfileCycleStatus::structurally_cross_task};
    if (interval.end < interval.start)
        return {{}, ProfileCycleStatus::counter_regression};
    return {interval.end - interval.start, ProfileCycleStatus::complete};
#endif
}

#endif

#if EXSIA_STAGE_PROFILE_ENABLED && defined(__linux__) && defined(__aarch64__)
namespace detail {
void record_stage_cycles(LocalBlockCycleSample &                                        sample,
                         const std::array<ggml::gemmini::cycle::NativeCycleSample, 5> & endpoints) {
    sample.stage_endpoints = endpoints;
    for (size_t stage = 0; stage < sample.stage_intervals.size(); ++stage) {
        ProfileInterval interval{};
        interval.valid                = true;
        interval.start_sample         = endpoints[stage];
        interval.end_sample           = endpoints[stage + 1];
        sample.stage_intervals[stage] = checked_profile_interval(interval);
    }
    sample.p0 = sample.stage_intervals[0].cycles.value_or(0);
    sample.p1 = sample.stage_intervals[1].cycles.value_or(0);
    sample.p2 = sample.stage_intervals[2].cycles.value_or(0);
    sample.p3 = sample.stage_intervals[3].cycles.value_or(0);
}
} // namespace detail
#endif

#if EXSIA_PROFILE_LOG_ENABLED
namespace detail {
static inline bool profile_interval_valid(const ProfileInterval & interval) {
#if defined(__linux__) && defined(__aarch64__)
    return interval.valid;
#else
    return interval.valid && interval.end >= interval.start;
#endif
}

static const char * profile_cycle_status_name(ProfileCycleStatus status) {
    switch (status) {
    case ProfileCycleStatus::complete:
        return "complete";
    case ProfileCycleStatus::missing_component:
        return "missing_component";
    case ProfileCycleStatus::invalid_start:
        return "invalid_start";
    case ProfileCycleStatus::invalid_end:
        return "invalid_end";
    case ProfileCycleStatus::source_mismatch:
        return "source_mismatch";
    case ProfileCycleStatus::event_owner_mismatch:
        return "event_owner_mismatch";
    case ProfileCycleStatus::event_generation_mismatch:
        return "event_generation_mismatch";
    case ProfileCycleStatus::structurally_cross_task:
        return "structurally_cross_task";
    case ProfileCycleStatus::counter_regression:
        return "counter_regression";
    case ProfileCycleStatus::sum_overflow:
        return "sum_overflow";
    }
    return "missing_component";
}

static inline size_t expected_profile_team_size(const char * mode) {
    return std::strcmp(mode, "Sequential") == 0 ? 1 : EXSIA_OMP_THREAD_COUNT;
}

static inline void write_nullable_json_string(std::ostream & out, const char * value) {
    if (value == nullptr || *value == '\0')
        out << "null";
    else
        write_json_string(out, value);
}

static std::string profile_tail(const ProfileConfig & config) {
    const auto context = performance::serialize_context(trace::inference_context(config.origin));
    return ",\"execution_id\":" + nlohmann::json(config.execution_id).dump() +
           ",\"inference_context\":" + (context.empty() ? "null" : context);
}

static inline void write_timeline_event(log::ProfileRows &      batch,
                                        const ProfileConfig &   config,
                                        const char *            layer,
                                        uint64_t                run_id,
                                        const char *            mode,
                                        size_t                  stripe_idx,
                                        const char *            op,
                                        const ProfileInterval & interval,
                                        size_t                  team_size,
                                        const size_t *          worker_id) {
    std::ostringstream out;
    out << "{\"schema\":\"gemmini.cycle\",\"version\":2,"
        << "\"record_type\":\"TIMELINE\",\"op\":";
    write_json_string(out, op);
    out << ",\"layer\":";
    write_nullable_json_string(out, layer);
    out << ",\"run_id\":" << run_id << ",\"mode\":";
    write_json_string(out, mode);
    out << ",\"stripe_id\":" << stripe_idx << ",\"slot\":" << stripe_idx % EXSIA_PIPELINE_SLOT_COUNT
        << ",\"node_id\":null,\"worker_id\":";
    if (worker_id == nullptr)
        out << "null";
    else
        out << *worker_id;
    const bool pipeline_cross_task =
        std::strcmp(mode, "LocalFoldingPipeline") == 0 &&
        (std::strcmp(op, "exsia.local") == 0 || std::strcmp(op, "exsia.stripe_total") == 0);
    const ProfileCycleValue checked = checked_profile_interval(interval, !pipeline_cross_task);
    out << ",\"start\":" << interval.start << ",\"end\":" << interval.end
        << ",\"start_thread_id\":" << interval.start_thread_id
        << ",\"end_thread_id\":" << interval.end_thread_id << ",\"host_timing\":"
        << ggml::gemmini::cycle::serialize_host_timing(
               interval.start_ns, interval.end_ns, interval.start_tid, interval.end_tid)
        << ",\"clock_mode\":";
    write_json_string(out, ggml::gemmini::cycle::clock_mode());
    out << ",\"units\":";
    write_json_string(out, ggml::gemmini::cycle::units());
    out << ",\"source\":";
    write_json_string(out, kNativeCycleSource);
    out << ",\"unit\":";
    write_json_string(out, kNativeCycleUnit);
    out << ",\"timer_resolution\":" << ggml::gemmini::cycle::resolution()
        << ",\"team_size\":" << team_size << ",\"elapsed\":";
    const char * excluded = ggml::gemmini::log::cpu_service_exclusion(op);
    if (checked.cycles.has_value() && !excluded)
        out << *checked.cycles;
    else
        out << "null";
    out << ",\"cycle_status\":";
    write_json_string(out, profile_cycle_status_name(checked.status));
#if defined(__linux__) && defined(__aarch64__)
    out << ",\"sample_reason\":";
    write_json_string(out, ggml::gemmini::cycle::reason_name(checked.sample_reason));
#endif
    out << ggml::gemmini::log::serialize_cpu_service_metadata(op, excluded) << profile_tail(config);
    batch.rows.push_back({out.str(),
                          log::ProfileRowKind::timeline,
                          kNativeCycleSource,
                          {interval.start_ns, interval.start_tid},
                          {interval.end_ns, interval.end_tid}});
}

static inline void write_timeline_run_event(log::ProfileRows &      batch,
                                            const ProfileConfig &   config,
                                            const char *            layer,
                                            uint64_t                run_id,
                                            const char *            mode,
                                            const ProfileInterval & interval,
                                            size_t                  team_size) {
    std::ostringstream out;
    out << "{\"schema\":\"gemmini.cycle\",\"version\":2,"
        << "\"record_type\":\"TIMELINE\",\"op\":\"exsia.run_total\",\"layer\":";
    write_nullable_json_string(out, layer);
    out << ",\"run_id\":" << run_id << ",\"mode\":";
    write_json_string(out, mode);
    const ProfileCycleValue checked = checked_profile_interval(interval);
    out << ",\"stripe_id\":null,\"slot\":null,\"node_id\":null,\"worker_id\":null"
        << ",\"start\":" << interval.start << ",\"end\":" << interval.end
        << ",\"start_thread_id\":" << interval.start_thread_id
        << ",\"end_thread_id\":" << interval.end_thread_id << ",\"host_timing\":"
        << ggml::gemmini::cycle::serialize_host_timing(
               interval.start_ns, interval.end_ns, interval.start_tid, interval.end_tid)
        << ",\"clock_mode\":";
    write_json_string(out, ggml::gemmini::cycle::clock_mode());
    out << ",\"units\":";
    write_json_string(out, ggml::gemmini::cycle::units());
    out << ",\"source\":";
    write_json_string(out, kNativeCycleSource);
    out << ",\"unit\":";
    write_json_string(out, kNativeCycleUnit);
    out << ",\"timer_resolution\":" << ggml::gemmini::cycle::resolution()
        << ",\"team_size\":" << team_size << ",\"elapsed\":";
    const char * excluded = ggml::gemmini::log::cpu_service_exclusion("exsia.run_total");
    if (checked.cycles.has_value() && !excluded)
        out << *checked.cycles;
    else
        out << "null";
    out << ",\"cycle_status\":";
    write_json_string(out, profile_cycle_status_name(checked.status));
#if defined(__linux__) && defined(__aarch64__)
    out << ",\"sample_reason\":";
    write_json_string(out, ggml::gemmini::cycle::reason_name(checked.sample_reason));
#endif
    out << ggml::gemmini::log::serialize_cpu_service_metadata("exsia.run_total", excluded)
        << profile_tail(config);
    batch.rows.push_back({out.str(),
                          log::ProfileRowKind::timeline,
                          kNativeCycleSource,
                          {interval.start_ns, interval.start_tid},
                          {interval.end_ns, interval.end_tid}});
}

#if EXSIA_STAGE_PROFILE_ENABLED
static inline void write_stage_metric(log::ProfileRows &      batch,
                                      const ProfileConfig &   config,
                                      const char *            layer,
                                      uint64_t                run_id,
                                      const char *            mode,
                                      size_t                  stripe_idx,
                                      const char *            suffix,
                                      uint64_t                value,
                                      ProfileCycleStatus      status,
                                      const char *            value_units,
                                      size_t                  team_size,
                                      const StageCycleStats * stats = nullptr) {
    std::ostringstream out;
    if (stats != nullptr && std::strcmp(value_units, "count") != 0)
        status = stats->cycle_status();
    out << "{\"schema\":\"gemmini.cycle\",\"version\":2,"
        << "\"record_type\":\"STAGE\",\"op\":\"exsia.stage_metric\",\"layer\":";
    write_nullable_json_string(out, layer);
    out << ",\"run_id\":" << run_id << ",\"mode\":";
    write_json_string(out, mode);
    out << ",\"stripe_id\":" << stripe_idx << ",\"slot\":" << stripe_idx % EXSIA_PIPELINE_SLOT_COUNT
        << ",\"node_id\":null,\"worker_id\":null,\"metric\":";
    write_json_string(out, suffix);
    out << ",\"value\":";
    if (status == ProfileCycleStatus::complete)
        out << value;
    else
        out << "null";
    out << ",\"value_units\":";
    write_json_string(out, value_units);
    out << ",\"source\":";
    write_json_string(out, kNativeCycleSource);
    out << ",\"unit\":";
    write_json_string(out, std::strcmp(value_units, "count") == 0 ? "count" : kNativeCycleUnit);
    out << ",\"team_size\":" << team_size << ",\"cycle_status\":";
    write_json_string(out, profile_cycle_status_name(status));
    if (stats != nullptr) {
        out << ",\"total_count\":" << stats->total_count << ",\"valid_count\":" << stats->count
            << ",\"invalid_count\":" << stats->total_count - stats->count;
#if defined(__linux__) && defined(__aarch64__)
        out << ",\"sample_reason\":";
        write_json_string(out, ggml::gemmini::cycle::reason_name(stats->sample_reason));
#endif
    }
    out << ggml::gemmini::log::serialize_cpu_service_metadata("exsia.stage_metric")
        << profile_tail(config);
    batch.rows.push_back({out.str(), log::ProfileRowKind::stage, kNativeCycleSource});
}
#endif

static std::mutex profile_flush_mutex;

FailureCode flush_profile(const ProfileConfig &                    config,
                          const char *                             layer,
                          uint64_t                                 run_id,
                          const char *                             mode,
                          const std::vector<StripeProfileRecord> & profiles,
                          const ProfileInterval &                  run_interval,
                          size_t                                   K_logical,
                          size_t                                   K_padded) {
    log::ProfileRows batch;
    const size_t     expected_team_size = expected_profile_team_size(mode);
    const bool       sequential         = std::strcmp(mode, "Sequential") == 0;

    for (const StripeProfileRecord & profile : profiles) {
        if (!profile_interval_valid(profile.local) ||
            !profile_interval_valid(profile.mask_assembly) ||
            !profile_interval_valid(profile.exponent_reduction) ||
            !profile_interval_valid(profile.folding) ||
            !profile_interval_valid(profile.stripe_total) ||
            profile.team_size != expected_team_size)
            return FailureCode::ProfileIntervalInvalid;
        write_timeline_event(batch,
                             config,
                             layer,
                             run_id,
                             mode,
                             profile.stripe_idx,
                             "exsia.local",
                             profile.local,
                             profile.team_size,
                             nullptr);
        if (!sequential) {
            for (size_t group = 0; group < profile.local_groups.size(); ++group) {
                const ProfileInterval & interval = profile.local_groups[group];
                if (!profile_interval_valid(interval))
                    return FailureCode::ProfileIntervalInvalid;
                write_timeline_event(batch,
                                     config,
                                     layer,
                                     run_id,
                                     mode,
                                     profile.stripe_idx,
                                     "exsia.local_group",
                                     interval,
                                     profile.team_size,
                                     &group);
            }
        }
        write_timeline_event(batch,
                             config,
                             layer,
                             run_id,
                             mode,
                             profile.stripe_idx,
                             "exsia.mask_assembly",
                             profile.mask_assembly,
                             profile.team_size,
                             nullptr);
        write_timeline_event(batch,
                             config,
                             layer,
                             run_id,
                             mode,
                             profile.stripe_idx,
                             "exsia.exponent_reduction",
                             profile.exponent_reduction,
                             profile.team_size,
                             nullptr);
        write_timeline_event(batch,
                             config,
                             layer,
                             run_id,
                             mode,
                             profile.stripe_idx,
                             "exsia.folding",
                             profile.folding,
                             profile.team_size,
                             nullptr);
        write_timeline_event(batch,
                             config,
                             layer,
                             run_id,
                             mode,
                             profile.stripe_idx,
                             "exsia.stripe_total",
                             profile.stripe_total,
                             profile.team_size,
                             nullptr);
#if EXSIA_STAGE_PROFILE_ENABLED
        const uint64_t     blocks  = profile.stats.p3_bypass_no_int_count +
                                     profile.stats.p3_bypass_same_scale_count +
                                     profile.stats.p3_replay_count;
        const uint64_t     logical = (profile.row_end - profile.row_start) * K_logical;
        const uint64_t     padded  = (profile.row_end - profile.row_start) * K_padded;
        std::ostringstream trace;
        trace << "{\"schema\":\"gemmini.cycle\",\"version\":2,"
              << "\"record_type\":\"EXSIA_WORKLOAD\",\"op\":\"exsia.workload\",\"layer\":";
        write_nullable_json_string(trace, layer);
        trace << ",\"run_id\":" << run_id << ",\"stripe_id\":" << profile.stripe_idx
              << ",\"slot\":" << profile.stripe_idx % EXSIA_PIPELINE_SLOT_COUNT << ",\"mode\":";
        write_json_string(trace, mode);
        trace << ",\"row_begin\":" << profile.row_start << ",\"row_end\":" << profile.row_end
              << ",\"logical_elements\":" << logical << ",\"padded_elements\":" << padded
              << ",\"padding_elements\":" << padded - logical << ",\"processed_blocks\":" << blocks
              << ",\"reused_blocks\":"
              << blocks - profile.stats.p3_replay_count - profile.stats.forced_recompute_count
              << ",\"regenerated_blocks\":"
              << profile.stats.p3_replay_count + profile.stats.forced_recompute_count
              << ",\"forced_recomputed_blocks\":" << profile.stats.forced_recompute_count
              << ",\"selected_positions\":" << profile.selected_positions
              << ",\"residual_nnz\":" << profile.residual_nnz << ",\"host_timing\":"
              << cycle::serialize_host_timing(profile.stripe_total.start_ns,
                                              profile.stripe_total.end_ns,
                                              profile.stripe_total.start_tid,
                                              profile.stripe_total.end_tid)
              << ",\"source\":\"host_observation\",\"unit\":\"count\",\"valid\":true"
              << profile_tail(config);
        batch.rows.push_back({trace.str(),
                              log::ProfileRowKind::workload,
                              "host_observation",
                              {profile.stripe_total.start_ns, profile.stripe_total.start_tid},
                              {profile.stripe_total.end_ns, profile.stripe_total.end_tid}});

        const StageCycleStats * stages[] = {
            &profile.stats.p0,
            &profile.stats.p1,
            &profile.stats.p2,
            &profile.stats.p3,
        };
        for (size_t stage = 0; stage < 4; ++stage) {
#if defined(__linux__) && defined(__aarch64__)
            const StageCycleStats * checked_stats = stages[stage];
#else
            const StageCycleStats * checked_stats = nullptr;
#endif
            char suffix[48];
            std::snprintf(suffix, sizeof(suffix), "local.p%zu.sum", stage);
            write_stage_metric(batch,
                               config,
                               layer,
                               run_id,
                               mode,
                               profile.stripe_idx,
                               suffix,
                               stages[stage]->sum,
                               ProfileCycleStatus::complete,
                               ggml::gemmini::cycle::units(),
                               profile.team_size,
                               checked_stats);
            std::snprintf(suffix, sizeof(suffix), "local.p%zu.count", stage);
            write_stage_metric(batch,
                               config,
                               layer,
                               run_id,
                               mode,
                               profile.stripe_idx,
                               suffix,
                               stages[stage]->count,
                               ProfileCycleStatus::complete,
                               "count",
                               profile.team_size,
                               checked_stats);
            std::snprintf(suffix, sizeof(suffix), "local.p%zu.max", stage);
            write_stage_metric(batch,
                               config,
                               layer,
                               run_id,
                               mode,
                               profile.stripe_idx,
                               suffix,
                               stages[stage]->max,
                               ProfileCycleStatus::complete,
                               ggml::gemmini::cycle::units(),
                               profile.team_size,
                               checked_stats);
        }
        write_stage_metric(batch,
                           config,
                           layer,
                           run_id,
                           mode,
                           profile.stripe_idx,
                           "local.p3.bypass_no_int.count",
                           profile.stats.p3_bypass_no_int_count,
                           ProfileCycleStatus::complete,
                           "count",
                           profile.team_size);
        write_stage_metric(batch,
                           config,
                           layer,
                           run_id,
                           mode,
                           profile.stripe_idx,
                           "local.p3.bypass_same_scale.count",
                           profile.stats.p3_bypass_same_scale_count,
                           ProfileCycleStatus::complete,
                           "count",
                           profile.team_size);
        write_stage_metric(batch,
                           config,
                           layer,
                           run_id,
                           mode,
                           profile.stripe_idx,
                           "local.p3.replay.count",
                           profile.stats.p3_replay_count,
                           ProfileCycleStatus::complete,
                           "count",
                           profile.team_size);
#endif
    }
    if (!profile_interval_valid(run_interval))
        return FailureCode::ProfileIntervalInvalid;
    const size_t run_team_size = profiles.empty() ? expected_team_size : profiles.front().team_size;
    write_timeline_run_event(batch, config, layer, run_id, mode, run_interval, run_team_size);

    (void)K_logical;
    (void)K_padded;
    std::string serialized;
    for (const auto & row : batch.rows)
        serialized += row.body + "}\n";
    log::cycle.write_profile(std::move(batch), config.origin);

    std::lock_guard<std::mutex> lock(profile_flush_mutex);
    if (!ggml::gemmini::log::prepare_output_parent(config.log_path))
        return FailureCode::ProfileFlushFailure;
    std::ofstream file(config.log_path, std::ios::app);
    if (!file)
        return FailureCode::ProfileFlushFailure;
    file << serialized;
    file.flush();
    return file ? FailureCode::None : FailureCode::ProfileFlushFailure;
}
} // namespace detail
#endif

#if EXSIA_VALIDATION && EXSIA_STAGE_PROFILE_ENABLED && EXSIA_PROFILE_LOG_ENABLED
std::string serialize_stage_sum_for_test(const StageCycleStats & stats) {
    log::ProfileRows batch;
    write_stage_metric(batch,
                       ProfileConfig{},
                       "test-layer",
                       0,
                       "Sequential",
                       0,
                       "local.p0.sum",
                       stats.sum,
                       ProfileCycleStatus::complete,
                       ggml::gemmini::cycle::units(),
                       1,
                       &stats);
    return batch.rows.front().body + "}\n";
}
#endif

} // namespace ggml::gemmini::quants::act::exsia
