#include "../ggml/src/ggml-gemmini/residual/residual-capture.hpp"

#include <gemmini/cycle_reader.hpp>

#include <cstdio>

namespace {

namespace cycle = ggml::gemmini::cycle;
namespace residual = ggml::gemmini::residual;

bool check(bool condition, const char * message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

bool direct_finish_preserves_the_canonical_payload() {
    residual::TimedResidualCapture capture(residual::ResidualRoute::cpu_direct);
    capture.reset(3, 7, 2, 64, 17);
    if (!capture.add_residual(1, 33, -129) || !capture.add_residual(0, 2, 1)) {
        return check(false, "direct fixture accepts residuals");
    }
    const residual::ResidualStripePayload payload = capture.finish();
#if !LOG_CYCLE
    if (!check(payload.capture_ns == 0, "cycle-off direct finish has no ns interval")) return false;
#endif
    return check(payload.direct != nullptr && payload.packet == nullptr,
              "direct finish remains route-exclusive") &&
        check(payload.direct->events.size() == 2 &&
                  payload.direct->events[0] == residual::ResidualEvent{0, 2, 1} &&
                  payload.direct->events[1] == residual::ResidualEvent{1, 33, -129},
              "direct finish preserves canonical sorted events");
}

bool packet_finish_preserves_the_canonical_payload() {
    ggml::gemmini::rmd::RmdStripeBuilder legacy;
    legacy.reset(3, 7, 2, 64, 17);
    legacy.add_residual(1, 33, -129);
    legacy.add_residual(0, 2, 1);
    const auto expected = legacy.finish();

    residual::TimedResidualCapture capture(residual::ResidualRoute::ws_packet);
    capture.reset(3, 7, 2, 64, 17);
    capture.add_residual(1, 33, -129);
    capture.add_residual(0, 2, 1);
    const residual::ResidualStripePayload payload = capture.finish();
#if !LOG_CYCLE
    if (!check(payload.capture_ns == 0, "cycle-off packet finish has no ns interval")) return false;
#endif
    return check(payload.direct == nullptr && payload.packet != nullptr && expected != nullptr,
              "packet finish remains route-exclusive") &&
        check(payload.packet->k_indices == expected->k_indices &&
                  payload.packet->stacked_activation == expected->stacked_activation,
              "packet finish preserves canonical bytes");
}

bool empty_capture_performs_zero_reads() {
    residual::TimedResidualCapture direct(residual::ResidualRoute::cpu_direct);
    direct.reset(0, 0, 1, 1, 1);
    residual::TimedResidualCapture packet(residual::ResidualRoute::ws_packet);
    packet.reset(0, 0, 1, 1, 1);
    cycle::reset_read_count_for_test();
    const auto direct_payload = direct.finish();
    const uint64_t direct_reads = cycle::read_count_for_test();
    cycle::reset_read_count_for_test();
    const auto packet_payload = packet.finish();
    const uint64_t packet_reads = cycle::read_count_for_test();
    return check(direct_payload.empty() && packet_payload.empty() &&
                     direct_payload.capture_ns == 0 && packet_payload.capture_ns == 0 &&
                     direct_reads == 0 && packet_reads == 0,
                 "each empty structural return performs zero local reads");
}

#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
bool mismatches_are_invalid_with_an_exact_reason() {
    using cycle::NativeCycleReason;
    const cycle::NativeCycleSample start{
        100, true, NativeCycleReason::none, cycle::NativeCycleSource::perf_cpu_cycles, 7, 11};
    const cycle::NativeCycleSample owner_end{
        105, true, NativeCycleReason::none, cycle::NativeCycleSource::perf_cpu_cycles, 8, 11};
    const cycle::NativeCycleSample generation_end{
        105, true, NativeCycleReason::none, cycle::NativeCycleSource::perf_cpu_cycles, 7, 12};
    const cycle::NativeCycleSample unavailable{
        0, false, NativeCycleReason::multiplexed,
        cycle::NativeCycleSource::perf_cpu_cycles, 7, 11};

    const auto owner = cycle::evaluate_interval(start, owner_end);
    const auto generation = cycle::evaluate_interval(start, generation_end);
    const auto failed = cycle::evaluate_interval(start, unavailable);
    return check(!owner.valid && owner.reason == NativeCycleReason::event_owner_mismatch,
                 "owner mismatch publishes no valid interval and retains its reason") &&
        check(!generation.valid &&
                  generation.reason == NativeCycleReason::event_generation_mismatch,
              "generation mismatch publishes no valid interval and retains its reason") &&
        check(!failed.valid && failed.reason == NativeCycleReason::invalid_end &&
                  failed.sample_reason == NativeCycleReason::multiplexed,
              "failed endpoint publishes null-equivalent validity plus the endpoint reason");
}
#endif

} // namespace

int main() {
    const bool ok = direct_finish_preserves_the_canonical_payload() &&
        packet_finish_preserves_the_canonical_payload() &&
        empty_capture_performs_zero_reads()
#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
        && mismatches_are_invalid_with_an_exact_reason()
#endif
        ;
    if (ok) std::puts("PASS: standalone residual capture-finish contract");
    return ok ? 0 : 1;
}
