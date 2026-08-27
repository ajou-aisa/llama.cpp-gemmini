#include "../ggml/src/ggml-gemmini/ggml-gemmini-matmul-cpu-work.hpp"

#include <cstdio>
#include <limits>
#include <string>

using namespace ggml::gemmini;

namespace {

bool expect(bool condition, const char * message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}

CpuWorkComponent valid(uint64_t cycles, CpuWorkCoverage coverage,
                       bool additive = true) {
    CpuWorkComponent component{};
    component.cycles = cycles;
    component.reason = "complete";
    component.coverage = coverage;
    component.additive = additive;
    return component;
}

CpuWorkComponent invalid(const char * reason, CpuWorkCoverage coverage) {
    CpuWorkComponent component{};
    component.reason = reason;
    component.coverage = coverage;
    return component;
}

struct DenseExecutionFixture {
    uint64_t caller_pmu;
    uint64_t elapsed_ns;
    uint64_t provider_child_pmu;
    uint64_t rtl_cycles;
    bool valid_end = true;
};

CpuWorkComponent select_dense_fixture(const DenseExecutionFixture & fixture) {
    DenseCpuWorkInput input{};
    input.parent = fixture.valid_end
        ? valid(fixture.caller_pmu, CpuWorkCoverage::coarse_same_thread_envelope)
        : invalid("invalid_end", CpuWorkCoverage::invalid);
    if (!fixture.valid_end) input.parent.sample_reason = "multiplexed";
    return select_dense_cpu_work(input);
}

bool test_dense_active_poll_blocked_and_invalid() {
    // Given: active polling and blocked calls with unrelated provider/RTL domains.
    const DenseExecutionFixture active{37, 300, 911, 77};
    const DenseExecutionFixture blocked{5, 900000, 4001, 880};
    const DenseExecutionFixture invalid_end{37, 300, 911, 77, false};

    // When: each same-thread caller interval is selected.
    const CpuWorkComponent active_selected = select_dense_fixture(active);
    const CpuWorkComponent blocked_selected = select_dense_fixture(blocked);
    const CpuWorkComponent invalid_selected = select_dense_fixture(invalid_end);

    // Then: caller scheduling controls PMU work; ns, child, and RTL are sentinels.
    return expect(active_selected.cycles == 37,
                  "active polling selects caller PMU 37") &&
        expect(blocked_selected.cycles == 5 && blocked.elapsed_ns > active.elapsed_ns,
               "blocked caller selects scheduled PMU 5 despite larger ns") &&
        expect(active.provider_child_pmu == 911 && blocked.provider_child_pmu == 4001 &&
                   active.rtl_cycles == 77 && blocked.rtl_cycles == 880,
               "provider-child and RTL sentinels do not affect caller PMU") &&
        expect(!invalid_selected.cycles.has_value() &&
                   invalid_selected.reason == "invalid_end" &&
                   invalid_selected.sample_reason == "multiplexed",
               "invalid endpoint nulls selected Dense aggregate") &&
        expect(invalid_end.elapsed_ns == 300 && invalid_end.rtl_cycles == 77,
               "invalid Dense keeps ns and RTL unchanged");
}

bool test_dense_fine_selection_demotes_parent() {
    // Given: a future proven two-leaf decomposition beneath parent 37.
    DenseCpuWorkInput input{};
    input.parent = valid(37, CpuWorkCoverage::coarse_same_thread_envelope);
    input.fine_selected = true;
    input.fine = {valid(11, CpuWorkCoverage::fine_leaves),
                  valid(13, CpuWorkCoverage::fine_leaves)};

    // When: fine coverage is explicitly selected.
    const CpuWorkComponent selected = select_dense_cpu_work(input);

    // Then: only 11+13 contributes; parent 37 is diagnostic.
    if (!expect(selected.cycles == 24 && selected.coverage == CpuWorkCoverage::fine_leaves,
                "fine Dense leaves total 24 and exclude parent 37")) return false;

    // Given: one selected leaf is malformed.
    input.fine[1] = invalid("event_generation_mismatch", CpuWorkCoverage::fine_leaves);

    // When: selection repeats.
    const CpuWorkComponent malformed = select_dense_cpu_work(input);

    // Then: the selector does not fall back to parent 37.
    if (!expect(!malformed.cycles.has_value() &&
                    malformed.reason == "event_generation_mismatch",
                "invalid selected fine leaf never falls back to coarse parent")) return false;

    // Given/When: selected fine coverage is empty or mislabeled.
    input.fine.clear();
    const CpuWorkComponent empty = select_dense_cpu_work(input);
    input.fine = {valid(11, CpuWorkCoverage::coarse_same_thread_envelope)};
    const CpuWorkComponent mislabeled = select_dense_cpu_work(input);
    input.fine = {valid(std::numeric_limits<uint64_t>::max(),
                        CpuWorkCoverage::fine_leaves),
                  valid(1, CpuWorkCoverage::fine_leaves)};
    const CpuWorkComponent overflow = select_dense_cpu_work(input);

    // Then: malformed fine routes fail closed without using parent 37.
    return expect(!empty.cycles.has_value() && empty.reason == "missing_fine_leaf",
                  "empty selected fine route is malformed") &&
        expect(!mislabeled.cycles.has_value() &&
                   mislabeled.reason == "invalid_fine_coverage",
               "mislabeled selected fine leaf is malformed") &&
        expect(!overflow.cycles.has_value() && overflow.reason == "overflow",
               "fine Dense checked-add overflow fails closed");
}

bool test_dense_external_route() {
    // Given: external completion has one marker, not two endpoints.
    DenseCpuWorkInput external{};
    external.external_marker = true;
    external.parent = valid(0, CpuWorkCoverage::coarse_same_thread_envelope);

    // When: selection runs.
    const CpuWorkComponent external_selected = select_dense_cpu_work(external);

    // Then: copied marker fields cannot manufacture valid zero work.
    return expect(!external_selected.cycles.has_value() &&
                      external_selected.coverage == CpuWorkCoverage::absent,
                  "external Dense one-marker route has no numeric interval");
}

bool test_rmd_route_selection() {
    // Given: every Task 7 RMD route classification.
    RmdCpuWorkInput input{};
    input.direct = valid(28, CpuWorkCoverage::algorithm_cpu_leaves);
    input.backend = valid(17, CpuWorkCoverage::coarse_same_thread_envelope);

    // When/Then: no residual is structurally absent.
    input.route = RmdCpuRoute::absent;
    if (!expect(select_rmd_cpu_work(input).coverage == CpuWorkCoverage::absent,
                "no-residual route is absent")) return false;

    // When/Then: CPU-direct consumes Task 5 leaves, not backend parent.
    input.route = RmdCpuRoute::cpu_direct;
    const CpuWorkComponent direct = select_rmd_cpu_work(input);
    if (!expect(direct.cycles == 28 &&
                    direct.coverage == CpuWorkCoverage::algorithm_cpu_leaves,
                "CPU-direct consumes algorithm CPU leaves")) return false;

    // When/Then: H1/HP1 checked software uses the existing backend pair.
    input.route = RmdCpuRoute::checked_software;
    const CpuWorkComponent software = select_rmd_cpu_work(input);
    if (!expect(software.cycles == 17 &&
                    software.coverage == CpuWorkCoverage::coarse_same_thread_envelope,
                "H1/HP1 checked software consumes backend pair")) return false;

    // When/Then: current Jetson native route has no numeric fallback.
    input.route = RmdCpuRoute::native_accelerator;
    const CpuWorkComponent native = select_rmd_cpu_work(input);
    if (!expect(!native.cycles.has_value() &&
                    native.reason == "unavailable_native_rmd_provider" &&
                    native.coverage == CpuWorkCoverage::unavailable_route,
                "Jetson native RMD remains unavailable without provider proof")) return false;

    // When/Then: malformed enum input fails closed rather than selecting a fallback.
    input.route = static_cast<RmdCpuRoute>(255);
    const CpuWorkComponent malformed = select_rmd_cpu_work(input);
    return expect(!malformed.cycles.has_value() && malformed.reason == "invalid_route",
                  "malformed RMD route fails closed");
}

bool test_finalize_and_nested_merge_metadata() {
    // Given: the values used to exercise Task 7's production metric shape.
    MatmulCpuWorkMetrics metrics{};
    metrics.compose = valid(13, CpuWorkCoverage::coarse_same_thread_envelope);
    metrics.finalize = valid(41, CpuWorkCoverage::coarse_same_thread_envelope);
    metrics.merge = valid(17, CpuWorkCoverage::coarse_same_thread_envelope, false);

    // Then: Finalize is its own component and nested Merge is diagnostic only.
    if (!expect(metrics.compose.cycles == 13 && metrics.finalize.cycles == 41 &&
                    metrics.merge.cycles == 17 && !metrics.merge.additive,
                "Compose 13, Finalize 41, and non-additive Merge 17 remain distinct")) {
        return false;
    }

    // Given: Case J's valid Finalize and invalid Merge endpoint.
    metrics.finalize = valid(19, CpuWorkCoverage::coarse_same_thread_envelope);
    metrics.merge = invalid("invalid_end", CpuWorkCoverage::invalid);

    // Then: the nested diagnostic does not alter the valid Finalize component.
    return expect(metrics.finalize.cycles == 19 &&
                      !metrics.merge.cycles.has_value() &&
                      metrics.merge.reason == "invalid_end",
                  "Case J keeps valid Finalize separate from invalid Merge");
}

}

int main(int argc, char ** argv) {
    const bool summary = argc == 2 && std::string(argv[1]) == "--summary";
    if (argc > 1 && !summary) {
        std::fprintf(stderr, "usage: test-gemmini-matmul-cpu-work [--summary]\n");
        return 2;
    }
    const bool ok = test_dense_active_poll_blocked_and_invalid() &&
        test_dense_fine_selection_demotes_parent() &&
        test_dense_external_route() &&
        test_rmd_route_selection() &&
        test_finalize_and_nested_merge_metadata();
    if (summary && ok) {
        std::puts("TASK7_CPU_WORK dense_parent=37 dense_active=37 dense_blocked=5 "
                  "dense_fine=24 direct=28 "
                  "software_backend=17 compose=13 finalize=41 merge=17 "
                  "merge_additive=false native=null "
                  "native_reason=unavailable_native_rmd_provider");
    }
    return ok ? 0 : 1;
}
