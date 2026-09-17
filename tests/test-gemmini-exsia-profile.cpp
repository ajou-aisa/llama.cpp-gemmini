#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp"
#include "../common/json.hpp"

#include <ggml.h>
#include <gemmini/host-timing.hpp>
#include <gemmini/log.hpp>
#include <gemmini/performance.hpp>

#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

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

bool check_cpu_workers(const nlohmann::json &summary, uint64_t expected_intervals) {
    const auto &cpu = summary.at("cpu_workers");
    const uint64_t cycles_valid = cpu.at("cycles_valid_count");
    const uint64_t thread_cpu_valid = cpu.at("thread_cpu_valid_count");
    if (!check(cpu.at("interval_count") == expected_intervals &&
                   cycles_valid <= expected_intervals && thread_cpu_valid <= expected_intervals,
               "CPU totals count the caller once and each noncaller once per entered team")) return false;
    if (!check((cycles_valid == expected_intervals ?
                    (cpu.at("cycles").is_number_unsigned() && cpu.at("cycles_reason").is_null()) :
                    (cpu.at("cycles").is_null() && cpu.at("cycles_reason").is_string())) &&
                   (thread_cpu_valid == expected_intervals ?
                    (cpu.at("thread_cpu_ns").is_number_unsigned() && cpu.at("thread_cpu_reason").is_null()) :
                    (cpu.at("thread_cpu_ns").is_null() && cpu.at("thread_cpu_reason").is_string())),
               "only complete CPU aggregates have numeric totals")) return false;
#if !defined(__linux__) || !defined(__aarch64__)
    if (!check(cpu.at("cycles").is_null() && cycles_valid == 0 &&
                   cpu.at("cycles_reason") == "not_thread_cpu_counter",
               "host ticks are never reported as worker CPU cycles")) return false;
#endif
#if (defined(__linux__) || defined(__APPLE__)) && defined(CLOCK_THREAD_CPUTIME_ID)
    if (!check(cpu.at("thread_cpu_ns").is_number_unsigned() && thread_cpu_valid == expected_intervals,
               "every participating thread contributes its own CPU time")) return false;
#endif
    return true;
}

bool test_invalid_run_keeps_caller_cpu(const std::filesystem::path &cycle_path) {
    ggml_gemmini_args_t args{};
    args.matmul_layer = "invalid-worker-profile-test";
    ggml_tensor tensor{};
    Meta meta;
    ExSIA quantizer;
    if (!check(!quantizer.run(meta, &tensor, args), "invalid input fails before creating a team")) return false;
    std::ifstream input(cycle_path);
    size_t summaries = 0;
    for (std::string line; std::getline(input, line);) {
        const auto event = nlohmann::json::parse(line);
        if (event.at("record_type") != "EXSIA_RUN_SUMMARY" || event.at("layer") != args.matmul_layer) continue;
        ++summaries;
        if (!check(event.at("operation_success") == false && event.at("handoff_calls") == 0,
                   "early failure still produces a failed run summary") ||
            !check_cpu_workers(event, 1)) return false;
    }
    return check(summaries == 1, "early failure records the caller exactly once");
}

bool test_workload_and_recompute(const std::filesystem::path &detail_path,
                                const std::filesystem::path &cycle_path) {
    namespace perf = ggml::gemmini::performance;
    constexpr size_t rows = 2 * DIM + 1, columns = BLOCK_SIZE + 3;
    std::vector<float> source(rows * columns, 0.5f);
    for (size_t row = 0; row + 1 < rows; ++row) {
        if (row % 3 == 1) {
            source[row * columns] = 1.5f;
            source[row * columns + 1] = 0.75f;
        } else if (row % 3 == 2) {
            std::fill_n(source.begin() + row * columns, BLOCK_SIZE, 0.25f);
            source[row * columns] = 1.0f;
            source[row * columns + 1] = 0.5f;
        }
        source[row * columns + BLOCK_SIZE - 1] = 0.0f;
    }
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
    std::vector<uint8_t> baseline_dense;
    std::vector<int32_t> baseline_residual;
    std::vector<int16_t> baseline_theta;
    std::vector<std::vector<uint64_t>> baseline_masks;
    for (const auto mode : modes) for (const bool force : {false, true}) {
#if defined(_WIN32)
        if (_putenv_s("GGML_GEMMINI_EXSIA_FORCE_RECOMPUTE", force ? "1" : "0") != 0) return false;
#else
        if (setenv("GGML_GEMMINI_EXSIA_FORCE_RECOMPUTE", force ? "1" : "0", 1) != 0) return false;
#endif
        ggml_gemmini_args_t args{};
        args.I = rows; args.J = 1; args.K = columns; args.sA = columns;
        args.tile_I = 1; args.activation_rows_per_stripe = DIM;
        args.matmul_layer = "workload-recompute-test";
        if (!args.A.allocate(rows, columns, GGML_GEMMINI_ACTIVATION_BITS)) return false;
        perf::reset();
        perf::start_request(ggml::gemmini::cycle::timestamp_ns());
        perf::begin_operation(perf::Phase::prefill, ggml::gemmini::cycle::timestamp_ns());
        const auto producer_context = nlohmann::json::parse(perf::log_context());
        const StripeReadySink sink{nullptr, [](void *, const StripeReadyEvent &event) {
            if (event.stripe_id == 0) {
                perf::end_operation(ggml::gemmini::cycle::timestamp_ns(), true);
                perf::begin_operation(perf::Phase::decode, ggml::gemmini::cycle::timestamp_ns());
            }
            event.submission_wait_ns = 0;
            return true;
        }};
        Meta meta;
        ExSIA quantizer;
        quantizer.set_execution_mode(mode);
        if (!check(quantizer.run(meta, &tensor, args, &sink), "workload fixture quantizes")) return false;
        perf::end_operation(ggml::gemmini::cycle::timestamp_ns(), true);
        perf::finish_request(ggml::gemmini::cycle::timestamp_ns());
        perf::finish_recording();
        if (!gemmini_log_cycle_flush()) return false;
        const auto &state = quantizer.state();
        std::vector<std::vector<uint64_t>> masks;
        for (const auto &stripe : state.stripe) masks.push_back(stripe.outlier_mask.words);
        if (baseline_dense.empty()) {
            baseline_dense = *args.A.bytes;
            baseline_residual = state.residual;
            baseline_theta = meta.theta;
            baseline_masks = masks;
            if (!check(meta.run_id == 0, "workload preserves first invocation zero")) return false;
        } else if (!check(*args.A.bytes == baseline_dense && state.residual == baseline_residual &&
                              meta.theta == baseline_theta && masks == baseline_masks,
                          "forced recomputation and worker modes preserve final codes scales masks residuals")) return false;
        if (!check(state.validation_p3_branch_counts[0] > 0 && state.validation_p3_branch_counts[1] > 0 &&
                       state.validation_p3_branch_counts[2] > 0,
                   "fixture executes every P3 decision with nonzero counts")) return false;
        std::array<std::vector<nlohmann::json>, 2> records;
        size_t stream = 0;
        for (const auto &path : {detail_path, cycle_path}) {
            std::ifstream input(path);
            for (std::string line; std::getline(input, line);) {
                auto event = nlohmann::json::parse(line);
                if (event.value("layer", nlohmann::json()) == args.matmul_layer &&
                    event.value("run_id", nlohmann::json()) == *meta.run_id &&
                    (event["record_type"] == "TIMELINE" || event["record_type"] == "STAGE" ||
                     event["record_type"] == "EXSIA_WORKLOAD")) records[stream].push_back(std::move(event));
            }
            ++stream;
        }
        if (!check(!records[0].empty() && records[0] == records[1],
                   "main and detail retain identical captured profile records")) return false;
        size_t workloads = 0;
        for (const auto &event : records[0]) {
            if (!check(event.at("inference_context") == producer_context,
                       "delayed profile serialization keeps producer prefill context")) return false;
            if (!check(event.at("execution_id") == ggml::gemmini::cycle::host_execution_id(),
                       "all profile records retain execution identity")) return false;
            if (event.at("record_type") != "EXSIA_WORKLOAD") continue;
            const size_t index = event.at("stripe_id");
            const auto &stripe = state.stripe[index];
            const auto &stats = state.profile_snapshot.stripes[index].stats;
            uint64_t selected = 0, nnz = 0;
            for (size_t row = stripe.row_start; row < stripe.row_end; ++row)
                for (size_t col = 0; col < columns; ++col) {
                    selected += stripe.outlier_mask.is_set(row - stripe.row_start, col);
                    nnz += state.residual[row * state.K_padded + col] != 0;
                }
            const uint64_t logical = stripe.row_count() * columns;
            const uint64_t padded = stripe.row_count() * 2 * BLOCK_SIZE;
            const uint64_t blocks = stripe.row_count() * 2;
            const uint64_t eligible = stats.p3_bypass_no_int_count + stats.p3_bypass_same_scale_count;
            if (!check(event.at("logical_elements") == logical && event.at("padded_elements") == padded &&
                           event.at("padding_elements") == padded - logical && event.at("processed_blocks") == blocks &&
                           event.at("selected_positions") == selected && event.at("residual_nnz") == nnz &&
                           event.at("reused_blocks") == (force ? 0 : eligible) &&
                           event.at("regenerated_blocks") == (force ? blocks : stats.p3_replay_count) &&
                           event.at("forced_recomputed_blocks") == (force ? eligible : 0) &&
                           event.at("host_timing").contains("execution_id"),
                       "workload counts logical padding selection sparse output and actual recomputation")) return false;
            if (!check(index == 2 ? selected == 0 && nnz == 0 : selected > nnz && nnz > 0,
                       "selected positions differ from nnz and zero residual stripe is retained")) return false;
            ++workloads;
        }
        if (!check(workloads == 3, "one workload survives for every stripe")) return false;
    }
#if defined(_WIN32)
    _putenv_s("GGML_GEMMINI_EXSIA_FORCE_RECOMPUTE", "");
#else
    unsetenv("GGML_GEMMINI_EXSIA_FORCE_RECOMPUTE");
#endif
    perf::reset();
    return true;
}

bool test_three_stripe_publication_and_worker_profiles(const std::filesystem::path &detail_path,
                                                     const std::filesystem::path &cycle_path) {
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
    enum class SinkKind { NoWait, Blocking, Uninstrumented, Rejected };
    for (const auto mode : modes) for (const auto kind : {
            SinkKind::NoWait, SinkKind::Blocking, SinkKind::Uninstrumented, SinkKind::Rejected}) {
        ggml_gemmini_args_t args{};
        args.I = rows; args.J = 1; args.K = columns; args.sA = columns;
        args.tile_I = 1; args.activation_rows_per_stripe = DIM;
        args.matmul_layer = "worker-profile-test";
        if (!args.A.allocate(rows, columns, GGML_GEMMINI_ACTIVATION_BITS)) return false;
        struct Trace {
            SinkKind kind;
            size_t count = 0;
            std::optional<uint64_t> run_id;
            bool valid = true;
            uint64_t wait_ns = 0;
            std::array<uint64_t, 3> waits{};
            std::array<uint64_t, 3> callback_start_ns{};
            std::array<uint64_t, 3> callback_end_ns{};
            std::array<uint64_t, 3> quantization_end_ns{};
            std::mutex mutex;
            std::condition_variable cv;
            size_t requested = 0;
            size_t released = 0;
            bool stop = false;
        } trace{};
        trace.kind = kind;
        std::thread releaser;
        if (kind == SinkKind::Blocking) {
            releaser = std::thread([&trace] {
                std::unique_lock<std::mutex> lock(trace.mutex);
                while (true) {
                    trace.cv.wait(lock, [&trace] { return trace.stop || trace.requested > trace.released; });
                    if (trace.stop) return;
                    trace.released = trace.requested;
                    trace.cv.notify_all();
                }
            });
        }
        const StripeReadySink sink{&trace, [](void *opaque, const StripeReadyEvent &event) {
            auto &trace = *static_cast<Trace *>(opaque);
            if (trace.count >= trace.waits.size() || event.stripe_id != trace.count ||
                event.slot != trace.count % 2 || !event.collect_submission_timing ||
                (trace.run_id.has_value() && trace.run_id != event.run_id)) {
                trace.valid = false;
                return false;
            }
            const size_t index = trace.count;
            trace.callback_start_ns[index] = ggml::gemmini::cycle::timestamp_ns();
            trace.quantization_end_ns[index] = event.quantization_end_ns;
            trace.valid = trace.valid && event.quantization_start_ns <= event.quantization_end_ns &&
                          event.quantization_end_ns <= trace.callback_start_ns[index];
            trace.run_id = event.run_id;
            ++trace.count;
            if (trace.kind == SinkKind::Blocking) {
                std::unique_lock<std::mutex> lock(trace.mutex);
                ++trace.requested;
                trace.cv.notify_all();
                const uint64_t start = ggml::gemmini::cycle::timestamp_ns();
                trace.cv.wait(lock, [&trace] { return trace.released == trace.requested; });
                event.submission_wait_ns = ggml::gemmini::cycle::timestamp_ns() - start;
            } else if (trace.kind != SinkKind::Uninstrumented) {
                event.submission_wait_ns = 0;
            }
            trace.waits[index] = event.submission_wait_ns.value_or(0);
            trace.wait_ns += trace.waits[index];
            trace.callback_end_ns[index] = ggml::gemmini::cycle::timestamp_ns();
            return trace.kind != SinkKind::Rejected || trace.count < 2;
        }};
        Meta meta;
        ExSIA quantizer;
        quantizer.set_execution_mode(mode);
        const bool success = quantizer.run(meta, &tensor, args, &sink);
        if (releaser.joinable()) {
            {
                std::lock_guard<std::mutex> lock(trace.mutex);
                trace.stop = true;
            }
            trace.cv.notify_all();
            releaser.join();
        }
        const bool expected_success = kind != SinkKind::Rejected;
        if (!check(success == expected_success && trace.valid &&
                       trace.count == (expected_success ? 3 : 2) && trace.run_id.has_value() &&
                       (!expected_success || meta.run_id == trace.run_id),
                   "all supported modes publish real context across slot reuse")) return false;
        std::ifstream cycle_input(cycle_path);
        std::vector<nlohmann::json> summaries;
        std::vector<nlohmann::json> submissions;
        std::vector<nlohmann::json> raw_cpu, canonical_timeline;
        for (std::string line; std::getline(cycle_input, line);) {
            const auto event = nlohmann::json::parse(line);
            if (!event.contains("run_id") || event.at("run_id") != *trace.run_id) continue;
            if (event.at("record_type") == "EXSIA_RUN_SUMMARY") summaries.push_back(event);
            if (event.at("record_type") == "CPU_INTERVAL") raw_cpu.push_back(event);
            if (event.at("record_type") == "TIMELINE") canonical_timeline.push_back(event);
            if (event.at("record_type") == "CYCLE_INTERVAL" && event.at("op") == "exsia.stripe_submission")
                submissions.push_back(event);
        }
        if (!check(summaries.size() == 1 && submissions.size() == trace.count,
                   "every run has one summary and every invoked callback has submission timing")) return false;
        const auto &summary = summaries.front();
        const uint64_t expected_cpu_intervals = mode == ExSIAState::ExecutionMode::Sequential ? 1 :
            mode == ExSIAState::ExecutionMode::LocalParallel ?
                1 + trace.count * (EXSIA_OMP_THREAD_COUNT - 1) : EXSIA_OMP_THREAD_COUNT;
        if (!check_cpu_workers(summary, expected_cpu_intervals)) return false;
        size_t raw_callers = 0, raw_workers = 0, raw_callbacks = 0, raw_stages = 0;
        for (const auto &record : raw_cpu) {
            if (!check(record.at("layer") == args.matmul_layer &&
                           record.at("worker_id").is_number_unsigned() &&
                           record.at("host_timing").at("valid") == true &&
                           record.at("native_cycles").at("start").contains("owner_token") &&
                           record.at("native_cycles").at("end").contains("generation") &&
                           record.contains("thread_cpu_timing") && record.at("additive") == false,
                       "ExSIA raw spans retain identity, wall time and native provenance")) return false;
            const std::string op = record.at("op");
            if (op == "exsia.run.caller") ++raw_callers;
            else if (op == "exsia.worker") ++raw_workers;
            else if (op == "exsia.submission_callback") ++raw_callbacks;
            else if (!record.at("stripe_id").is_null()) ++raw_stages;
        }
        if (!check(raw_callers == 1 && raw_workers + 1 == expected_cpu_intervals &&
                       raw_callbacks == trace.count && raw_stages >= trace.count,
                   "ExSIA preserves caller, worker, callback and stripe-stage raw records")) return false;
        const uint64_t run_ns = summary.at("run_wall_ns");
        const uint64_t handoff_ns = summary.at("handoff_wall_ns");
        const auto &run_host = summary.at("host_timing");
        const bool measured = kind != SinkKind::Uninstrumented;
        if (!check(summary.at("op") == "exsia.run.summary" &&
                       summary.at("source") == "steady_clock" && summary.at("unit") == "nanosecond" &&
                       summary.at("layer") == args.matmul_layer &&
                       summary.at("operation_success") == expected_success &&
                       summary.at("handoff_calls") == trace.count &&
                       summary.at("wait_measured_calls") == (measured ? trace.count : 0) &&
                       run_host.at("duration_ns") == run_ns && run_host.at("valid") == true &&
                       run_ns >= handoff_ns &&
                       summary.at("outside_handoff_wall_ns") == run_ns - handoff_ns,
                   "summary preserves identity, success, callback counts, and run wall-time partition")) return false;
        if (!check(measured ?
                       (handoff_ns >= trace.wait_ns && summary.at("submission_wait_ns") == trace.wait_ns &&
                        summary.at("handoff_nonwait_ns") == handoff_ns - trace.wait_ns) :
                       (summary.at("submission_wait_ns").is_null() && summary.at("handoff_nonwait_ns").is_null()),
                   "measured zero and real waits stay numeric while unsupported sinks remain null")) return false;
        if (!check(kind != SinkKind::Blocking || trace.wait_ns > 0,
                   "condition-variable rendezvous performs a measured wait")) return false;
        uint64_t submission_ns = 0;
        for (const auto &event : submissions) {
            const size_t index = event.at("stripe_id");
            if (!check(index < trace.count && event.at("slot") == index % 2 &&
                           event.at("layer") == args.matmul_layer && event.contains("host_timing") &&
                           event.at("source") == "steady_clock" && event.at("unit") == "nanosecond" &&
                           event.at("operation_success") == (expected_success || index + 1 < trace.count),
                       "submission detail preserves each callback identity")) return false;
            const auto &host = event.at("host_timing");
            const uint64_t start = host.at("start_ns");
            const uint64_t end = host.at("end_ns");
            if (!check(start >= trace.quantization_end_ns[index] &&
                           run_host.at("start_ns") <= start && run_host.at("end_ns") >= end &&
                           start <= trace.callback_start_ns[index] && end >= trace.callback_end_ns[index] &&
                           host.at("duration_ns") == end - start && host.at("valid") == true &&
                           host.at("clock") == "steady_clock" && host.at("unit") == "nanosecond" &&
                           host.at("start_tid") == host.at("end_tid") && host.at("start_tid") != 0,
                       "submission wall interval encloses only the sink after quantization")) return false;
            const uint64_t duration = host.at("duration_ns");
            submission_ns += duration;
            if (!check(measured ?
                           (duration >= trace.waits[index] && event.at("submission_wait_ns") == trace.waits[index] &&
                            event.at("handoff_nonwait_ns") == duration - trace.waits[index]) :
                           (event.at("submission_wait_ns").is_null() && event.at("handoff_nonwait_ns").is_null()),
                       "each submission preserves the sink wait measurement and wall-time partition")) return false;
        }
        if (!check(submission_ns == handoff_ns,
                   "run handoff total includes exactly the invoked callback intervals")) return false;
        if (!expected_success) continue;
        const auto &profiles = quantizer.state().profile_snapshot.stripes;
        if (!check(profiles.size() == 3, "all three stripe profiles survive publication")) return false;
        std::ifstream input(detail_path);
        std::vector<nlohmann::json> events;
        for (std::string line; std::getline(input, line);) {
            const auto event = nlohmann::json::parse(line);
            if (event.at("record_type") == "TIMELINE" && event.at("run_id") == *meta.run_id)
                events.push_back(event);
        }
        if (!check(canonical_timeline.size() == events.size(),
                   "canonical cycle log retains every existing ExSIA timeline event")) return false;
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
    const std::filesystem::path cycle_path = detail_path.string() + ".cycle.jsonl";
    if (!ggml::gemmini::log::cycle.set_output_path(cycle_path.string().c_str(), true)) return 1;
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
                    test_workload_and_recompute(detail_path, cycle_path) &&
                    test_invalid_run_keeps_caller_cpu(cycle_path) &&
                    test_three_stripe_publication_and_worker_profiles(detail_path, cycle_path);
    if (ok && argc == 1) {
        std::filesystem::remove(detail_path);
        std::filesystem::remove(cycle_path);
    }
    if (ok) std::printf("PASS: ExSIA individual worker provenance workers=%zu\n",
                        EXSIA_LOCAL_WORKER_COUNT);
    return ok ? 0 : 1;
}
