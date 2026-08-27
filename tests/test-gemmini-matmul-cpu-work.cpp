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

bool test_dense_coarse_when_fine_is_unavailable() {
    // Given: Case G's valid same-thread Dense host envelope.
    DenseCpuWorkInput input{};
    input.parent = valid(37, CpuWorkCoverage::coarse_same_thread_envelope);

    // When: the current opaque provider has no proven fine leaves.
    const CpuWorkComponent selected = select_dense_cpu_work(input);

    // Then: scheduled caller work is the canonical coarse contribution.
    return expect(selected.cycles == 37, "Case G selects Dense parent 37") &&
        expect(selected.coverage == CpuWorkCoverage::coarse_same_thread_envelope,
               "Case G reports coarse same-thread coverage") &&
        expect(selected.additive, "selected Dense parent is additive");
}

bool test_dense_invalid_endpoint_fails_closed() {
    // Given: Case H's invalid end while ns and RTL remain separate values.
    DenseCpuWorkInput input{};
    input.parent = invalid("invalid_end", CpuWorkCoverage::invalid);
    input.parent.sample_reason = "multiplexed";
    constexpr uint64_t dense_ns = 300;
    constexpr uint64_t dense_rtl_cycles = 77;

    // When: Dense selection runs without a fine decomposition.
    const CpuWorkComponent selected = select_dense_cpu_work(input);

    // Then: CPU work is null and unrelated domains are untouched.
    return expect(!selected.cycles.has_value() && selected.reason == "invalid_end" &&
                      selected.sample_reason == "multiplexed",
                  "Case H preserves invalid Dense end and sample reason") &&
        expect(dense_ns == 300 && dense_rtl_cycles == 77,
               "Dense ns and RTL remain independent");
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

    // Then: malformed fine routes fail closed without using parent 37.
    return expect(!empty.cycles.has_value() && empty.reason == "missing_fine_leaf",
                  "empty selected fine route is malformed") &&
        expect(!mislabeled.cycles.has_value() &&
                   mislabeled.reason == "invalid_fine_coverage",
               "mislabeled selected fine leaf is malformed");
}

bool test_dense_blocked_and_external_routes() {
    // Given: a blocked call with large wall time but only five scheduled cycles.
    DenseCpuWorkInput blocked{};
    blocked.parent = valid(5, CpuWorkCoverage::coarse_same_thread_envelope);
    constexpr uint64_t blocked_ns = 900000;

    // When: the coarse pair is selected.
    const CpuWorkComponent blocked_selected = select_dense_cpu_work(blocked);

    // Then: only caller PMU work contributes, not wall time.
    if (!expect(blocked_selected.cycles == 5 && blocked_ns == 900000,
                "blocked Dense counts scheduled caller cycles only")) return false;

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

bool test_compose_finalize_and_merge_nesting() {
    // Given: direct/no-packet post work.
    RmdPostCpuWorkInput direct{};
    direct.rmd = valid(17, CpuWorkCoverage::coarse_same_thread_envelope);
    direct.compose = valid(13, CpuWorkCoverage::coarse_same_thread_envelope);
    direct.finalize = valid(41, CpuWorkCoverage::coarse_same_thread_envelope);
    direct.merge = valid(17, CpuWorkCoverage::coarse_same_thread_envelope, false);

    // When: direct bookkeeping is selected.
    const RmdPostCpuWorkSelection direct_selected = select_rmd_post_cpu_work(direct);

    // Then: Compose is absent and Finalize contains non-additive Merge.
    if (!expect(!direct_selected.compose.cycles.has_value() &&
                    direct_selected.compose.coverage == CpuWorkCoverage::absent,
                "direct route emits no packet Compose")) return false;
    if (!expect(direct_selected.canonical_cycles == 58,
                "direct post total is backend 17 plus Finalize 41")) return false;

    // Given: packet Compose is required.
    RmdPostCpuWorkInput packet = direct;
    packet.packet = true;

    // When: packet post work is selected.
    const RmdPostCpuWorkSelection packet_selected = select_rmd_post_cpu_work(packet);

    // Then: Compose contributes, while Merge remains visible and non-additive.
    return expect(packet_selected.compose.cycles == 13,
                  "packet route emits Compose 13") &&
        expect(packet_selected.finalize_canonical_cycles == 41,
               "Finalize canonical contribution is 41, not Finalize plus Merge") &&
        expect(packet_selected.finalize.cycles == 41 &&
                   packet_selected.merge.cycles == 17 &&
                   !packet_selected.merge.additive,
               "Finalize 41 keeps Merge 17 visible and non-additive") &&
        expect(packet_selected.canonical_cycles == 71,
               "packet canonical total excludes nested Merge");
}

bool test_failed_merge_and_overflow_publish_no_partial_total() {
    // Given: Case J's valid Finalize and invalid Merge diagnostic.
    RmdPostCpuWorkInput case_j{};
    case_j.rmd = valid(17, CpuWorkCoverage::coarse_same_thread_envelope);
    case_j.finalize = valid(19, CpuWorkCoverage::coarse_same_thread_envelope);
    case_j.merge = invalid("invalid_end", CpuWorkCoverage::invalid);

    // When: Finalize itself succeeds.
    const RmdPostCpuWorkSelection selected = select_rmd_post_cpu_work(case_j);

    // Then: invalid nested Merge does not poison canonical Finalize.
    if (!expect(selected.canonical_cycles == 36 &&
                    !selected.merge.cycles.has_value() &&
                    selected.merge.reason == "invalid_end",
                "Case J uses valid Finalize once despite invalid Merge diagnostic")) return false;

    // Given: Merge operation fails after timing.
    case_j.merge_succeeded = false;

    // When: post selection runs.
    const RmdPostCpuWorkSelection failed = select_rmd_post_cpu_work(case_j);

    // Then: no partial canonical total is published.
    if (!expect(!failed.canonical_cycles.has_value() &&
                    !failed.finalize_canonical_cycles.has_value() &&
                    failed.reason == "failed_operation",
                "failed Merge invalidates canonical post total")) return false;

    // Given: checked addition would overflow.
    case_j.merge_succeeded = true;
    case_j.rmd = valid(std::numeric_limits<uint64_t>::max(),
                       CpuWorkCoverage::coarse_same_thread_envelope);
    case_j.finalize = valid(1, CpuWorkCoverage::coarse_same_thread_envelope);

    // When/Then: overflow is null, never wrapped or partial.
    const RmdPostCpuWorkSelection overflow = select_rmd_post_cpu_work(case_j);
    return expect(!overflow.canonical_cycles.has_value() && overflow.reason == "overflow",
                  "post CPU work checked-add overflow fails closed");
}

}

int main(int argc, char ** argv) {
    const bool summary = argc == 2 && std::string(argv[1]) == "--summary";
    if (argc > 1 && !summary) {
        std::fprintf(stderr, "usage: test-gemmini-matmul-cpu-work [--summary]\n");
        return 2;
    }
    const bool ok = test_dense_coarse_when_fine_is_unavailable() &&
        test_dense_invalid_endpoint_fails_closed() &&
        test_dense_fine_selection_demotes_parent() &&
        test_dense_blocked_and_external_routes() &&
        test_rmd_route_selection() &&
        test_compose_finalize_and_merge_nesting() &&
        test_failed_merge_and_overflow_publish_no_partial_total();
    if (summary && ok) {
        std::puts("TASK7_CPU_WORK dense_parent=37 dense_fine=24 direct=28 "
                  "software_backend=17 compose=13 finalize=41 merge=17 "
                  "packet_post=71 merge_additive=false native=null "
                  "native_reason=unavailable_native_rmd_provider");
    }
    return ok ? 0 : 1;
}
