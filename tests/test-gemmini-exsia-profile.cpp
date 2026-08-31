#include "../ggml/src/ggml-gemmini/ggml-gemmini-args.h"
#include "../ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp"

#include <ggml.h>

#include <array>
#include <cstdint>
#include <cstdio>

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

} // namespace

int main() {
    const bool ok = test_each_configured_worker_keeps_individual_provenance() &&
                    test_worker_failure_is_local_to_that_worker() &&
                    test_structural_eligibility_is_individual();
    if (ok) std::printf("PASS: ExSIA individual worker provenance workers=%zu\n",
                        EXSIA_LOCAL_WORKER_COUNT);
    return ok ? 0 : 1;
}
