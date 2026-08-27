#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp"

#include <ggml.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

using namespace ggml::gemmini::quants::act::exsia;

bool check(bool value, const char *message) {
    if (!value) std::fprintf(stderr, "FAIL: %s\n", message);
    return value;
}

ProfileInterval interval(uint64_t cycles) {
    ProfileInterval result{};
    result.start = 100;
    result.end = 100 + cycles;
    result.start_ns = 1000;
    result.end_ns = 1000 + cycles;
    result.start_thread_id = 91;
    result.end_thread_id = 92;
#if defined(__linux__) && defined(__aarch64__)
    result.start_sample = {result.start, true,
        ggml::gemmini::cycle::NativeCycleReason::none,
        ggml::gemmini::cycle::NativeCycleSource::perf_cpu_cycles, 7, 3};
    result.end_sample = {result.end, true,
        ggml::gemmini::cycle::NativeCycleReason::none,
        ggml::gemmini::cycle::NativeCycleSource::perf_cpu_cycles, 7, 3};
#endif
    result.valid = true;
    return result;
}

StripeProfileRecord parallel_fixture() {
    StripeProfileRecord profile{};
    profile.local = interval(1000);
    profile.mask_assembly = interval(23);
    profile.exponent_reduction = interval(29);
    profile.folding = interval(31);
    profile.stripe_total = interval(3000);
    constexpr std::array<uint64_t, 4> worker_cycles{11, 13, 17, 19};
    for (size_t worker = 0; worker < profile.local_groups.size(); ++worker)
        profile.local_groups[worker] = interval(worker_cycles[worker]);
#if EXSIA_STAGE_PROFILE_ENABLED
    profile.stats.p0.sum = 10000;
    profile.stats.p1.sum = 20000;
    profile.stats.p2.sum = 30000;
    profile.stats.p3.sum = 40000;
#endif
    return profile;
}

bool test_parallel_selects_every_worker_and_canonical_stages() {
    // Given: configured workers and misleading parent/nested diagnostics.
    const StripeProfileRecord profile = parallel_fixture();

    // When: canonical Quantize CPU work is selected for a parallel mode.
    const QuantizeProfileCycles result = aggregate_quantize_profile(false, profile);

    // Then: only all worker leaves plus Mask, Exponent, and Folding contribute.
    constexpr uint64_t expected = EXSIA_LOCAL_WORKER_COUNT == 3 ? 124 : 143;
    return check(result.total.cycles.has_value() && *result.total.cycles == expected,
                 "parallel canonical total uses every configured worker") &&
           check(result.total.status == ProfileCycleStatus::complete,
                 "parallel canonical total is complete") &&
           check(result.local.cycles.has_value() &&
                     *result.local.cycles == expected - 23 - 29 - 31,
                 "parallel Local is the checked worker sum") &&
           check(result.mask.cycles == 23 && result.exponent.cycles == 29 &&
                     result.folding.cycles == 31,
                 "canonical stage components remain distinct");
}

bool test_sequential_selects_only_local_envelope() {
    // Given: a sequential profile whose worker groups are misleading sentinels.
    StripeProfileRecord profile = parallel_fixture();
    profile.local = interval(11);

    // When: canonical Quantize CPU work is selected for Sequential.
    const QuantizeProfileCycles result = aggregate_quantize_profile(true, profile);

    // Then: only Local, Mask, Exponent, and Folding contribute.
    return check(result.total.cycles.has_value() && *result.total.cycles == 94,
                 "sequential canonical total selects profile.local") &&
           check(result.local.cycles == 11,
                 "sequential Local excludes worker-group sentinels");
}

bool test_missing_worker_rejects_partial_total() {
    // Given: one configured parallel worker has no completed interval.
    StripeProfileRecord profile = parallel_fixture();
    profile.local_groups.back() = ProfileInterval{};

    // When: the parallel profile is reduced.
    const QuantizeProfileCycles result = aggregate_quantize_profile(false, profile);

    // Then: Local and total are null with the exact missing-component reason.
    return check(!result.local.cycles.has_value() && !result.total.cycles.has_value(),
                 "missing worker publishes no partial cycle sum") &&
           check(result.local.status == ProfileCycleStatus::missing_component &&
                     result.total.status == ProfileCycleStatus::missing_component,
                 "missing worker reason is deterministic");
}

bool test_checked_overflow_rejects_partial_total() {
    // Given: individually valid worker intervals whose checked sum overflows.
    StripeProfileRecord profile = parallel_fixture();
    profile.local_groups[0] = interval(std::numeric_limits<uint64_t>::max() - 100);
    profile.local_groups[0].start = 0; profile.local_groups[0].end = std::numeric_limits<uint64_t>::max();
    profile.local_groups[1] = interval(1);
    profile.local_groups[1].start = 0; profile.local_groups[1].end = 1;
#if defined(__linux__) && defined(__aarch64__)
    profile.local_groups[0].start_sample.value = 0;
    profile.local_groups[0].end_sample.value = std::numeric_limits<uint64_t>::max();
    profile.local_groups[1].start_sample.value = 0;
    profile.local_groups[1].end_sample.value = 1;
#endif

    // When: the parallel profile is reduced.
    const QuantizeProfileCycles result = aggregate_quantize_profile(false, profile);

    // Then: Local and total are null rather than a wrapped or partial value.
    return check(!result.local.cycles.has_value() && !result.total.cycles.has_value(),
                 "overflow publishes no partial cycle sum") &&
           check(result.local.status == ProfileCycleStatus::arithmetic_overflow &&
                     result.total.status == ProfileCycleStatus::arithmetic_overflow,
                 "overflow reason is deterministic");
}

bool test_empty_worker_is_present_as_valid_zero() {
    // Given: one configured worker completed an empty task with a valid zero interval.
    StripeProfileRecord profile = parallel_fixture();
    profile.local_groups.back() = interval(0);

    // When: the parallel profile is reduced.
    const QuantizeProfileCycles result = aggregate_quantize_profile(false, profile);

    // Then: the worker is present and contributes exactly zero.
    constexpr uint64_t expected = EXSIA_LOCAL_WORKER_COUNT == 3 ? 107 : 124;
    return check(result.total.cycles.has_value() && *result.total.cycles == expected,
                 "empty worker has deterministic zero component presence");
}

#if defined(__linux__) && defined(__aarch64__)
bool test_native_provenance_is_authoritative() {
    // Given: projected scalars are misleading but native samples share one owner.
    ProfileInterval profile = interval(0);
    profile.start = 9000;
    profile.end = 1;
    profile.start_sample.value = 5000; profile.end_sample.value = 5000;

    // When: the interval is checked.
    const ProfileCycleValue equal = checked_profile_interval(profile);
    profile.end_sample.owner_event_token += 1;
    const ProfileCycleValue owner_mismatch = checked_profile_interval(profile);
    profile.end_sample.owner_event_token = profile.start_sample.owner_event_token;
    profile.end_sample.generation += 1;
    const ProfileCycleValue generation_mismatch = checked_profile_interval(profile);
    const ProfileCycleValue priority = checked_profile_interval(profile, false);
    profile.end_sample.generation = profile.start_sample.generation;
    const ProfileCycleValue cross_task = checked_profile_interval(profile, false);

    // Then: valid zero and invalid reasons come only from native provenance.
    return check(equal.cycles.has_value() && *equal.cycles == 0,
                 "native valid zero ignores scalar ordering") &&
           check(owner_mismatch.status == ProfileCycleStatus::event_owner_mismatch &&
                     !owner_mismatch.cycles.has_value(),
                 "same stripe cannot override PMU owner mismatch") &&
           check(generation_mismatch.status ==
                     ProfileCycleStatus::event_generation_mismatch &&
                     !generation_mismatch.cycles.has_value(),
                 "generation mismatch rejects numeric cycles") &&
           check(priority.status == ProfileCycleStatus::event_generation_mismatch,
                 "provenance failure keeps priority over structural eligibility") &&
           check(cross_task.status == ProfileCycleStatus::structurally_cross_task &&
                     !cross_task.cycles.has_value(),
                 "cross-task diagnostic keeps no PMU numeric value");
}
#endif

#if defined(GGML_GEMMINI_HAS_OPENMP)
struct EventTrace {
    std::vector<StripeReadyEvent> events;
};

bool capture_event(void *opaque, const StripeReadyEvent &event) {
    static_cast<EventTrace *>(opaque)->events.push_back(event);
    return true;
}

bool test_profiled_parallel_mode(ExSIAState::ExecutionMode mode) {
    // Given: three stripes with enough blocks to schedule every configured worker.
    constexpr size_t rows = 65;
    constexpr size_t columns = 64;
    std::vector<float> source(rows * columns);
    for (size_t index = 0; index < source.size(); ++index)
        source[index] = index % 17 == 0 ? 32.0f : 0.5f;
    ggml_tensor tensor{}; tensor.type = GGML_TYPE_F32; tensor.data = source.data();
    ggml_gemmini_args_t args{};
    args.I = rows; args.J = 8; args.K = columns; args.sA = columns;
    args.tile_I = 2; args.tile_J = 3; args.tile_K = 4;
    args.activation_rows_per_stripe = 32;
    args.residual_route = ggml::gemmini::residual::ResidualRoute::cpu_direct;
    if (!args.A.allocate(rows, columns, GGML_GEMMINI_ACTIVATION_BITS)) return false;
    Meta meta; EventTrace trace;
    StripeReadySink sink{&trace, capture_event}; ExSIA exsia;
    exsia.set_execution_mode(mode);

    // When: the selected OpenMP mode executes through the real ExSIA surface.
    const bool ran = exsia.run(meta, &tensor, args, &sink);

    // Then: every stripe publishes complete canonical leaves and preserves its ns timeline.
    bool complete = ran && trace.events.size() == 3 &&
        exsia.state().profile_snapshot.stripes.size() == 3;
    for (size_t stripe = 0; complete && stripe < trace.events.size(); ++stripe) {
        const StripeReadyEvent &event = trace.events[stripe];
        const StripeProfileRecord &profile = exsia.state().profile_snapshot.stripes[stripe];
        complete = event.quantize_cpu_work.total.cycles.has_value() &&
            event.quantize_cpu_work.total.status == ProfileCycleStatus::complete &&
            event.local_start_ns == profile.local.start_ns &&
            event.local_end_ns == profile.local.end_ns &&
            profile.stripe_total.end_ns >= profile.stripe_total.start_ns;
    }
    if (mode == ExSIAState::ExecutionMode::LocalFoldingPipeline && complete) {
        const ProfileCycleValue broad = checked_profile_interval(
            exsia.state().profile_snapshot.stripes.front().local, false);
        complete = !broad.cycles.has_value() &&
            broad.status == ProfileCycleStatus::structurally_cross_task;
    }
    if (!complete)
        std::fprintf(stderr, "mode=%u ran=%d failure=%u stripe=%zu events=%zu profiles=%zu\n",
            static_cast<unsigned>(mode), ran ? 1 : 0,
            static_cast<unsigned>(exsia.state().failure_code), exsia.state().failure_stripe,
            trace.events.size(), exsia.state().profile_snapshot.stripes.size());
    return check(complete, mode == ExSIAState::ExecutionMode::LocalParallel
        ? "LocalParallel publishes complete canonical work"
        : "pipeline publishes worker work and keeps broad PMU null");
}
#endif

bool test_invalid_nested_diagnostic_does_not_poison_local() {
    // Given: canonical intervals are valid while P0/P2 diagnostics are invalid.
    StripeProfileRecord profile = parallel_fixture();
#if EXSIA_STAGE_PROFILE_ENABLED
    profile.stats.p0.sum = 0;
    profile.stats.p2.sum = 0;
#endif

    // When: canonical Quantize CPU work is reduced.
    const QuantizeProfileCycles result = aggregate_quantize_profile(false, profile);

    // Then: independently valid canonical Local remains complete.
    return check(result.local.cycles.has_value() &&
                     result.local.status == ProfileCycleStatus::complete,
                 "invalid nested diagnostics do not poison canonical Local");
}

} // namespace

int main() {
    const bool ok = test_parallel_selects_every_worker_and_canonical_stages() &&
                    test_sequential_selects_only_local_envelope() &&
                    test_missing_worker_rejects_partial_total() &&
                    test_checked_overflow_rejects_partial_total() &&
                    test_empty_worker_is_present_as_valid_zero() &&
#if defined(__linux__) && defined(__aarch64__)
                    test_native_provenance_is_authoritative() &&
#endif
#if defined(GGML_GEMMINI_HAS_OPENMP)
                    test_profiled_parallel_mode(ExSIAState::ExecutionMode::LocalParallel) &&
                    test_profiled_parallel_mode(ExSIAState::ExecutionMode::LocalFoldingPipeline) &&
#endif
                    test_invalid_nested_diagnostic_does_not_poison_local();
    if (ok) std::printf("PASS: ExSIA profile aggregation workers=%zu total=%u\n", EXSIA_LOCAL_WORKER_COUNT, EXSIA_LOCAL_WORKER_COUNT == 3 ? 124U : 143U);
    return ok ? 0 : 1;
}
