#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp"
#include "../common/json.hpp"

#include <ggml.h>
#include <gemmini/host-timing.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {

using namespace ggml::gemmini::quants::act::exsia;

constexpr std::array<uint64_t, 4> kStageSentinels{
    100001, 100003, 100007, 100009};

bool check(bool value, const char * message) {
    if (!value) std::fprintf(stderr, "FAIL: %s\n", message);
    return value;
}

ProfileInterval interval(uint64_t cycles, uint64_t owner, uint64_t generation) {
#if !defined(__linux__) || !defined(__aarch64__)
    (void) generation;
#endif
    ProfileInterval result{};
    result.start = 100;
    result.end = 100 + cycles;
    result.start_ns = 1000;
    result.end_ns = 1000 + cycles;
    result.start_thread_id = owner;
    result.end_thread_id = owner;
#if defined(__linux__) && defined(__aarch64__)
    result.start_sample = {result.start, true,
        ggml::gemmini::cycle::NativeCycleReason::none,
        ggml::gemmini::cycle::NativeCycleSource::perf_cpu_cycles,
        owner, generation};
    result.end_sample = {result.end, true,
        ggml::gemmini::cycle::NativeCycleReason::none,
        ggml::gemmini::cycle::NativeCycleSource::perf_cpu_cycles,
        owner, generation};
#endif
    result.valid = true;
    return result;
}

StripeProfileRecord worker_fixture() {
    StripeProfileRecord profile{};
    profile.local = interval(997, 70, 4);
    profile.mask_assembly = interval(23, 71, 4);
    profile.exponent_reduction = interval(29, 72, 4);
    profile.folding = interval(31, 73, 4);
    profile.stripe_total = interval(3000, 74, 4);
    for (size_t worker = 0; worker < profile.local_groups.size(); ++worker) {
        profile.local_groups[worker] = interval(
            11 + static_cast<uint64_t>(worker) * 2,
            100 + static_cast<uint64_t>(worker), 9);
    }
#if EXSIA_STAGE_PROFILE_ENABLED
    profile.stats.p0.sum = kStageSentinels[0];
    profile.stats.p1.sum = kStageSentinels[1];
    profile.stats.p2.sum = kStageSentinels[2];
    profile.stats.p3.sum = kStageSentinels[3];
#endif
    return profile;
}

bool test_each_configured_worker_keeps_individual_provenance() {
    const StripeProfileRecord profile = worker_fixture();
    bool ok = true;
    for (size_t worker = 0; worker < profile.local_groups.size(); ++worker) {
        const ProfileCycleValue value = checked_profile_interval(
            profile.local_groups[worker]);
        ok = check(value.cycles.has_value() &&
                       *value.cycles == 11 + static_cast<uint64_t>(worker) * 2 &&
                       value.status == ProfileCycleStatus::complete,
                   "each configured ExSIA worker retains its own checked interval") && ok;
#if defined(__linux__) && defined(__aarch64__)
        ok = check(profile.local_groups[worker].start_sample.owner_event_token ==
                       100 + static_cast<uint64_t>(worker) &&
                       profile.local_groups[worker].end_sample.owner_event_token ==
                       100 + static_cast<uint64_t>(worker) &&
                       profile.local_groups[worker].start_sample.generation == 9 &&
                       profile.local_groups[worker].end_sample.generation == 9,
                   "each ExSIA worker retains owner and generation provenance") && ok;
#endif
    }
#if EXSIA_STAGE_PROFILE_ENABLED
    ok = check(profile.stats.p0.sum == kStageSentinels[0] &&
                   profile.stats.p1.sum == kStageSentinels[1] &&
                   profile.stats.p2.sum == kStageSentinels[2] &&
                   profile.stats.p3.sum == kStageSentinels[3],
               "Local P0-P3 statistics remain independent diagnostics") && ok;
#endif
    return ok;
}

bool test_worker_failure_is_local_to_that_worker() {
    StripeProfileRecord profile = worker_fixture();
#if defined(__linux__) && defined(__aarch64__)
    profile.local_groups[1].end_sample.owner_event_token += 1000;
    if (profile.local_groups.size() > 2)
        profile.local_groups[2].end_sample.generation += 1;
#endif

    bool ok = true;
    for (size_t worker = 0; worker < profile.local_groups.size(); ++worker) {
        const ProfileCycleValue value = checked_profile_interval(
            profile.local_groups[worker]);
#if defined(__linux__) && defined(__aarch64__)
        if (worker == 1) {
            ok = check(!value.cycles.has_value() &&
                           value.status == ProfileCycleStatus::event_owner_mismatch,
                       "owner mismatch invalidates only its ExSIA worker") && ok;
        } else if (worker == 2) {
            ok = check(!value.cycles.has_value() &&
                           value.status == ProfileCycleStatus::event_generation_mismatch,
                       "generation mismatch invalidates only its ExSIA worker") && ok;
        } else
#endif
        {
            ok = check(value.cycles.has_value() &&
                           value.status == ProfileCycleStatus::complete,
                       "unrelated ExSIA workers remain individually valid") && ok;
        }
    }

    const ProfileCycleValue mask = checked_profile_interval(profile.mask_assembly);
    const ProfileCycleValue exponent = checked_profile_interval(profile.exponent_reduction);
    const ProfileCycleValue folding = checked_profile_interval(profile.folding);
    return check(mask.cycles == 23 && exponent.cycles == 29 && folding.cycles == 31,
                 "Mask, Exponent, and Folding remain independent statistics") && ok;
}

bool test_structural_eligibility_is_individual() {
    const StripeProfileRecord profile = worker_fixture();
    const ProfileCycleValue broad_local = checked_profile_interval(profile.local, false);
    const ProfileCycleValue broad_stripe = checked_profile_interval(profile.stripe_total, false);
    const ProfileCycleValue worker = checked_profile_interval(profile.local_groups.front());
    return check(!broad_local.cycles.has_value() &&
                     broad_local.status == ProfileCycleStatus::structurally_cross_task,
                 "cross-task Local envelope has no numeric PMU delta") &&
           check(!broad_stripe.cycles.has_value() &&
                     broad_stripe.status == ProfileCycleStatus::structurally_cross_task,
                 "cross-task stripe_total envelope has no numeric PMU delta") &&
           check(worker.cycles.has_value() && worker.status == ProfileCycleStatus::complete,
                 "cross-task parent does not poison an individual worker interval");
}

bool test_stage_aggregation_preserves_missing_and_invalid_samples() {
    StageCycleStats stats;
    if (!check(stats.cycle_status() == ProfileCycleStatus::missing_component,
               "unexecuted stage has no measured total")) return false;
    stats.add(ProfileCycleValue{0, ProfileCycleStatus::complete});
    stats.add(ProfileCycleValue{17, ProfileCycleStatus::complete});
    if (!check(stats.sum == 17 && stats.count == 2 && stats.total_count == 2 &&
                   stats.cycle_status() == ProfileCycleStatus::complete,
               "valid zero contributes a valid measurement")) return false;
    stats.add(ProfileCycleValue{{}, ProfileCycleStatus::invalid_end});
    StageCycleStats other;
    other.add(ProfileCycleValue{{}, ProfileCycleStatus::event_owner_mismatch});
    other.add(ProfileCycleValue{23, ProfileCycleStatus::complete});
    stats.merge(other);
    if (!check(stats.sum == 40 && stats.max == 23 && stats.count == 3 &&
                   stats.total_count == 5 &&
                   stats.cycle_status() == ProfileCycleStatus::invalid_end,
               "partial sum keeps first invalid reason and all sample counts")) return false;
#if defined(__linux__) && defined(__aarch64__)
    StageCycleStats failed;
    failed.add(ProfileCycleValue{{}, ProfileCycleStatus::invalid_start,
        ggml::gemmini::cycle::NativeCycleReason::multiplexed});
    StageCycleStats merged;
    merged.merge(failed);
    if (!check(merged.count == 0 && merged.total_count == 1 &&
                   merged.sample_reason == ggml::gemmini::cycle::NativeCycleReason::multiplexed,
               "aggregate retains the endpoint failure reason")) return false;
#endif
    const std::string invalid_json = serialize_stage_sum_for_test(stats);
    if (!check(invalid_json.find("\"value\":null") != std::string::npos &&
                   invalid_json.find("\"cycle_status\":\"invalid_end\"") != std::string::npos &&
                   invalid_json.find("\"total_count\":5") != std::string::npos &&
                   invalid_json.find("\"valid_count\":3") != std::string::npos &&
                   invalid_json.find("\"invalid_count\":2") != std::string::npos,
               "serialized partial aggregate is null with reason and complete counts")) return false;
    StageCycleStats zero;
    zero.add(ProfileCycleValue{0, ProfileCycleStatus::complete});
    const std::string zero_json = serialize_stage_sum_for_test(zero);
    if (!check(zero_json.find("\"source\":") != std::string::npos &&
                   zero_json.find("\"unit\":") != std::string::npos,
               "detail output identifies its counter source and unit")) return false;
    if (!check(zero_json.find("\"value\":0") != std::string::npos &&
                   zero_json.find("\"cycle_status\":\"complete\"") != std::string::npos &&
                   zero_json.find("\"run_id\":0") != std::string::npos,
               "serialized valid zero and real run zero survive")) return false;
    stats.reset();
    return check(stats.count == 0 && stats.total_count == 0 && stats.sum == 0 &&
                     stats.cycle_status() == ProfileCycleStatus::missing_component,
                 "reused stage statistics clear validity and counts");
}

bool test_full_origin_context_reset() {
    Meta meta;
    if (!check(!meta.run_id.has_value(), "unassigned quantization run is absent")) return false;
    meta.run_id = 0;
    if (!check(meta.run_id.has_value() && *meta.run_id == 0,
               "first quantization run is a real zero ID")) return false;
    meta.reset();
    return check(!meta.run_id.has_value(), "reset clears retained FULL run context");
}

bool test_stage_sum_overflow_is_not_complete() {
    StageCycleStats accumulated;
    accumulated.add(ProfileCycleValue{UINT64_MAX, ProfileCycleStatus::complete});
    accumulated.add(ProfileCycleValue{1, ProfileCycleStatus::complete});
    StageCycleStats merged;
    merged.add(ProfileCycleValue{UINT64_MAX, ProfileCycleStatus::complete});
    StageCycleStats child;
    child.add(ProfileCycleValue{1, ProfileCycleStatus::complete});
    merged.merge(child);
    for (const auto *stats : {&accumulated, &merged}) {
        const std::string json = serialize_stage_sum_for_test(*stats);
        if (!check(stats->cycle_status() != ProfileCycleStatus::complete &&
                       json.find("\"value\":null") != std::string::npos &&
                       json.find("\"cycle_status\":\"sum_overflow\"") != std::string::npos,
                   "overflowed stage sum is unavailable, not a wrapped valid number"))
            return false;
    }
    return true;
}

bool test_three_stripe_publication_and_worker_profiles(const std::filesystem::path &detail_path) {
    constexpr size_t rows = 2 * DIM + 1;
    constexpr size_t columns = 32;
    std::vector<float> source(rows * columns, 0.5f);
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.data = source.data();
    const ExSIAState::ExecutionMode modes[] = {
        ExSIAState::ExecutionMode::Sequential,
#if defined(GGML_GEMMINI_HAS_OPENMP)
        ExSIAState::ExecutionMode::LocalParallel,
        ExSIAState::ExecutionMode::LocalFoldingPipeline,
#endif
    };
    for (const auto mode : modes) {
        ggml_gemmini_args_t args{};
        args.I = rows; args.J = 1; args.K = columns; args.sA = columns;
        args.tile_I = 1; args.activation_rows_per_stripe = DIM;
        args.matmul_layer = "worker-profile-test";
        if (!args.A.allocate(rows, columns, GGML_GEMMINI_ACTIVATION_BITS)) return false;
        struct Trace { size_t count = 0; std::optional<uint64_t> run_id; } trace;
        const StripeReadySink sink{&trace, [](void *opaque, const StripeReadyEvent &event) {
            auto &trace = *static_cast<Trace *>(opaque);
            if (event.stripe_id != trace.count || event.slot != trace.count % 2 ||
                (trace.run_id.has_value() && trace.run_id != event.run_id)) return false;
            trace.run_id = event.run_id;
            ++trace.count;
            return true;
        }};
        Meta meta;
        ExSIA quantizer;
        quantizer.set_execution_mode(mode);
        if (!check(quantizer.run(meta, &tensor, args, &sink) && trace.count == 3 &&
                       trace.run_id.has_value() && meta.run_id == trace.run_id,
                   "all supported modes publish real context across slot reuse")) return false;
        const auto &profiles = quantizer.state().profile_snapshot.stripes;
        if (!check(profiles.size() == 3, "all three stripe profiles survive publication")) return false;
        std::ifstream input(detail_path);
        std::vector<nlohmann::json> events;
        for (std::string line; std::getline(input, line);) {
            const auto event = nlohmann::json::parse(line);
            if (event.at("record_type") == "TIMELINE" && event.at("run_id") == *meta.run_id)
                events.push_back(event);
        }
        const auto &run = quantizer.state().profile_snapshot.run;
        const auto check_timeline = [&](const ProfileInterval &interval, const char *op,
                                        std::optional<size_t> stripe, std::optional<size_t> worker,
                                        bool cross_task = false) {
            const auto event = std::find_if(events.begin(), events.end(), [&](const auto &row) {
                return row.at("op") == op &&
                    (stripe ? row.at("stripe_id") == *stripe : row.at("stripe_id").is_null()) &&
                    (worker ? row.at("worker_id") == *worker : row.at("worker_id").is_null());
            });
            if (!check(event != events.end() && event->contains("host_timing"),
                       "each actual timeline event carries host timing")) return false;
            const auto &host = event->at("host_timing");
            return check(interval.start_ns >= run.start_ns && interval.end_ns <= run.end_ns &&
                             interval.end_ns >= interval.start_ns &&
                             interval.start_tid != 0 && interval.end_tid != 0,
                         "captured host interval stays inside the run time axis") &&
                check(host.at("start_ns") == interval.start_ns &&
                          host.at("end_ns") == interval.end_ns &&
                          host.at("start_tid") == interval.start_tid &&
                          host.at("end_tid") == interval.end_tid &&
                          host.at("duration_ns") == interval.end_ns - interval.start_ns &&
                          host.at("valid") == true &&
                          host.at("clock") == "steady_clock" && host.at("unit") == "nanosecond" &&
                          host.contains("execution_id") && host.contains("thread_id_kind"),
                      "delayed output preserves actual endpoint timestamps and worker identities") &&
                check(!cross_task || (event->at("elapsed").is_null() &&
                          event->at("cycle_status") == "structurally_cross_task"),
                      "cross-task parent keeps valid host time without a numeric cycle delta");
        };
        if (!check_timeline(run, "exsia.run_total", {}, {})) return false;
        for (const auto &profile : profiles) {
            const bool pipeline = mode == ExSIAState::ExecutionMode::LocalFoldingPipeline;
            if (!check_timeline(profile.local, "exsia.local", profile.stripe_idx, {}, pipeline) ||
                !check_timeline(profile.stripe_total, "exsia.stripe_total", profile.stripe_idx, {}, pipeline) ||
                !check_timeline(profile.mask_assembly, "exsia.mask_assembly", profile.stripe_idx, {}) ||
                !check_timeline(profile.exponent_reduction, "exsia.exponent_reduction", profile.stripe_idx, {}) ||
                !check_timeline(profile.folding, "exsia.folding", profile.stripe_idx, {})) return false;
            if (mode == ExSIAState::ExecutionMode::Sequential) continue;
            for (size_t index = 0; index < profile.local_groups.size(); ++index) {
                const auto &worker = profile.local_groups[index];
                if (!check(worker.valid && worker.start_thread_id == worker.end_thread_id &&
                               worker.start_tid == worker.end_tid &&
                               (worker.start_thread_id == 0 || worker.start_tid != run.start_tid),
                           "local task records its executing thread, independent of the output thread") ||
                    !check_timeline(worker, "exsia.local_group", profile.stripe_idx, index)) return false;
            }
            if (mode == ExSIAState::ExecutionMode::LocalFoldingPipeline &&
                !check(!checked_profile_interval(profile.local, false).cycles.has_value() &&
                           !checked_profile_interval(profile.stripe_total, false).cycles.has_value(),
                       "real pipeline parent remains nonnumeric across task boundaries")) return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    const std::filesystem::path detail_path = argc > 1 ? std::filesystem::path(argv[1]) :
        std::filesystem::temp_directory_path() / ("gemmini-exsia-host-" +
            std::to_string(ggml::gemmini::cycle::host_thread_id()) + "-" +
            std::to_string(ggml::gemmini::cycle::timestamp_ns()) + ".jsonl");
#if defined(_WIN32)
    if (_putenv_s("GGML_GEMMINI_CYCLE_DETAIL_LOG", detail_path.string().c_str()) != 0) return 1;
#else
    if (setenv("GGML_GEMMINI_CYCLE_DETAIL_LOG", detail_path.string().c_str(), 1) != 0) return 1;
#endif
    const bool ok = test_each_configured_worker_keeps_individual_provenance() &&
                    test_worker_failure_is_local_to_that_worker() &&
                    test_structural_eligibility_is_individual() &&
                    test_stage_aggregation_preserves_missing_and_invalid_samples() &&
                    test_stage_sum_overflow_is_not_complete() &&
                    test_full_origin_context_reset() &&
                    test_three_stripe_publication_and_worker_profiles(detail_path);
    if (ok && argc == 1) std::filesystem::remove(detail_path);
    if (ok) std::printf("PASS: ExSIA individual worker provenance workers=%zu\n",
                        EXSIA_LOCAL_WORKER_COUNT);
    return ok ? 0 : 1;
}
