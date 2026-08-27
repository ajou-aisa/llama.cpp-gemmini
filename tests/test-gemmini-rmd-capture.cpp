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

bool test_capture_ns_boundaries_when_direct_finish_has_work() {
    // Given: a direct capture with unsorted residuals.
    residual::TimedResidualCapture capture(residual::ResidualRoute::cpu_direct);
    capture.reset(3, 7, 2, 64, 17);
    if (!capture.add_residual(1, 33, -129) ||
        !capture.add_residual(0, 2, 1)) {
        return check(false, "direct fixture accepts residuals");
    }
    cycle::reset_read_count_for_test();

    // When: the capture seals its direct payload.
    const residual::ResidualStripePayload payload = capture.finish();

    // Then: the existing timestamp pair brackets finish and preserves canonical output.
#if LOG_CYCLE
    return check(cycle::read_count_for_test() == 2,
                 "direct finish keeps exactly the timestamp pair") &&
        check(payload.direct != nullptr && payload.packet == nullptr,
              "direct capture keeps its route-exclusive payload") &&
        check(payload.direct->events.size() == 2 &&
                  payload.direct->events[0] == residual::ResidualEvent{0, 2, 1} &&
                  payload.direct->events[1] == residual::ResidualEvent{1, 33, -129},
              "direct capture keeps the Folding canonical value") &&
#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
        check(payload.capture_finish_route == residual::ResidualRoute::cpu_direct &&
                  payload.capture_finish_valid == payload.capture_finish_cycles.has_value(),
              "direct finish carries only a valid route-specific PMU interval");
#else
        check(!payload.capture_finish_route && !payload.capture_finish_valid &&
                  !payload.capture_finish_cycles,
              "non-Jetson direct finish omits PMU capture data");
#endif
#else
    return check(cycle::read_count_for_test() == 0 && payload.capture_ns == 0,
                 "cycle-off direct finish has no timing boundary");
#endif
}

bool test_capture_ns_boundaries_when_packet_finish_has_work() {
    // Given: matching legacy and capture packet inputs.
    ggml::gemmini::rmd::RmdStripeBuilder legacy;
    legacy.reset(3, 7, 2, 64, 17);
    legacy.add_residual(1, 33, -129);
    legacy.add_residual(0, 2, 1);
    const ggml::gemmini::rmd::StripePacketHandle expected = legacy.finish();
    residual::TimedResidualCapture capture(residual::ResidualRoute::ws_packet);
    capture.reset(3, 7, 2, 64, 17);
    capture.add_residual(1, 33, -129);
    capture.add_residual(0, 2, 1);
    cycle::reset_read_count_for_test();

    // When: the capture seals its packet payload.
    const residual::ResidualStripePayload payload = capture.finish();

    // Then: timestamps remain confined to finish and packet bytes stay canonical.
#if LOG_CYCLE
    return check(cycle::read_count_for_test() == 2,
                 "packet finish keeps exactly the timestamp pair") &&
        check(payload.direct == nullptr && payload.packet != nullptr && expected != nullptr,
              "packet capture keeps its route-exclusive payload") &&
        check(payload.packet->k_indices == expected->k_indices &&
                  payload.packet->stacked_activation == expected->stacked_activation,
              "packet capture keeps the Folding canonical value") &&
#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
        check(payload.capture_finish_route == residual::ResidualRoute::ws_packet &&
                  payload.capture_finish_valid == payload.capture_finish_cycles.has_value(),
              "packet finish carries only a valid route-specific PMU interval");
#else
        check(!payload.capture_finish_route && !payload.capture_finish_valid &&
                  !payload.capture_finish_cycles,
              "non-Jetson packet finish omits PMU capture data");
#endif
#else
    return check(cycle::read_count_for_test() == 0 && payload.capture_ns == 0,
                 "cycle-off packet finish has no timing boundary");
#endif
}

bool test_empty_capture_when_finish_has_no_work() {
    // Given: empty direct and packet captures.
    residual::TimedResidualCapture direct(residual::ResidualRoute::cpu_direct);
    direct.reset(0, 0, 1, 1, 1);
    residual::TimedResidualCapture packet(residual::ResidualRoute::ws_packet);
    packet.reset(0, 0, 1, 1, 1);
    cycle::reset_read_count_for_test();

    // When: both captures finish without residuals.
    const residual::ResidualStripePayload direct_payload = direct.finish();
    const residual::ResidualStripePayload packet_payload = packet.finish();

    // Then: empty captures have no payload or timing work.
    return check(direct_payload.empty() && packet_payload.empty() &&
                     direct_payload.capture_ns == 0 && packet_payload.capture_ns == 0 &&
                     !direct_payload.capture_finish_cycles && !packet_payload.capture_finish_cycles &&
                     !direct_payload.capture_finish_route && !packet_payload.capture_finish_route &&
                     cycle::read_count_for_test() == 0,
                 "empty capture remains absent without timing reads");
}

#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
bool test_finish_interval_is_invalid_when_owner_or_generation_changes() {
    // Given: valid sample pairs that cannot belong to one PMU event.
    using cycle::NativeCycleReason;
    const cycle::NativeCycleSample start{
        100, true, NativeCycleReason::none, cycle::NativeCycleSource::perf_cpu_cycles, 7, 11};
    const cycle::NativeCycleSample owner_end{
        105, true, NativeCycleReason::none, cycle::NativeCycleSource::perf_cpu_cycles, 8, 11};
    const cycle::NativeCycleSample generation_end{
        105, true, NativeCycleReason::none, cycle::NativeCycleSource::perf_cpu_cycles, 7, 12};

    // When: the finish interval is evaluated.
    const auto owner_delta = cycle::evaluate_interval(start, owner_end);
    const auto generation_delta = cycle::evaluate_interval(start, generation_end);

    // Then: neither malformed ownership interval is publishable.
    return check(!owner_delta.valid &&
                     owner_delta.reason == NativeCycleReason::event_owner_mismatch &&
                     !generation_delta.valid &&
                     generation_delta.reason == NativeCycleReason::event_generation_mismatch,
                 "owner and generation changes invalidate finish samples");
}
#endif

} // namespace

int main() {
    const bool ok = test_capture_ns_boundaries_when_direct_finish_has_work() &&
        test_capture_ns_boundaries_when_packet_finish_has_work() &&
        test_empty_capture_when_finish_has_no_work()
#if defined(__linux__) && defined(__aarch64__) && CYCLE_DETAIL
        && test_finish_interval_is_invalid_when_owner_or_generation_changes()
#endif
        ;
    if (ok) std::puts("PASS: residual capture baseline contract");
    return ok ? 0 : 1;
}
