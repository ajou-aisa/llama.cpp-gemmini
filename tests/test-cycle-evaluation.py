# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B -m pytest -q tests/test-cycle-evaluation.py
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))

import target_admission as admission
from campaign_build import (
    INCLUDE_REPO,
    PLATFORMS,
    Platform,
    cached,
    llama_plan,
    platform_profile,
    include_headers,
    semantic_options,
    source_state,
    with_platform,
)
from certified_reconstruction import publication_row, reconstructed_row
from e2e_timeline import Axis, pack_lanes, performance, timeline_rows, write_timeline
from eval_common import EvaluationError, Json, Record, sha256, write_json
from run_cycle_evaluation import instrumentation_markers, pmu_contract, timing_validity

IM2P = ROOT.parent / "IM2P.sim"
KINDS = (("potal-host", "STRIPE_PIPELINE", ()), ("fullcpu-host", "FULL", ()), ("cycle", "STRIPE_PIPELINE", ("llama-perplexity",)))


# Architecture / build configuration

def test_semantic_options_do_not_depend_on_platform() -> None:
    # Given every supported host and every build kind of the evaluation.
    for kind, matmul, extra in KINDS:
        plan = llama_plan(kind, "a8w8", 32, IM2P, matmul, extra)
        digests = {semantic_options(with_platform(plan.options, platform_profile(system, machine))).__repr__()
                   for system, machine in PLATFORMS}
        # Then the semantic option set is identical everywhere; only library suffix/format differ.
        assert len(digests) == 1
    assert {platform_profile(s, m).shared_library_suffix for s, m in PLATFORMS} == {".dylib", ".so"}


def test_unsupported_host_and_semantic_platform_options_are_rejected() -> None:
    with pytest.raises(EvaluationError, match="unsupported evaluation host"):
        platform_profile("Windows", "x86_64")
    forged = Platform("Linux", "x86_64", "ELF", ".so", "rpath", (("GGML_GEMMINI_DIM", "16"),))
    with pytest.raises(EvaluationError, match="platform options may not name semantic"):
        with_platform(llama_plan("cycle", "a8w8", 32, IM2P, "STRIPE_PIPELINE").options, forged)


def test_extra_target_build_shares_potal_semantics_but_fullcpu_does_not() -> None:
    potal = llama_plan("potal-host", "a8w8", 32, IM2P, "STRIPE_PIPELINE")
    extra = llama_plan("cycle", "a8w8", 32, IM2P, "STRIPE_PIPELINE", ("llama-perplexity",))
    fullcpu = llama_plan("fullcpu-host", "a8w8", 32, IM2P)
    # An extra CMake target and the cycle-log instrumentation are not semantic options.
    assert potal.semantic_options_sha256 == extra.semantic_options_sha256
    assert fullcpu.semantic_options_sha256 != potal.semantic_options_sha256
    assert "llama-perplexity" in extra.targets


def test_linux_dry_run_generates_the_same_semantic_plan() -> None:
    plans = {}
    for system, machine in (("Darwin", "arm64"), ("Linux", "x86_64"), ("Linux", "aarch64")):
        done = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/eval/run_cycle_evaluation.py"),
                               "--model", "m.gguf", "--prompt-file", "d.raw", "--dry-run",
                               "--system", system, "--machine", machine],
                              capture_output=True, text=True, check=True, cwd=ROOT)
        plans[machine] = json.loads(done.stdout)
    assert plans["x86_64"]["platform"]["shared_library_suffix"] == ".so"
    assert plans["arm64"]["platform"]["binary_format"] == "Mach-O"
    semantic = {machine: {name: build["semantic_options_sha256"] for name, build in plan["builds"].items()
                          if isinstance(build, dict)} for machine, plan in plans.items()}
    assert semantic["arm64"] == semantic["x86_64"] == semantic["aarch64"]


def test_build_cache_rebuilds_when_identity_changes(tmp_path: Path) -> None:
    built: list[Path] = []

    def make(path: Path) -> None:
        path.mkdir()
        built.append(path)
    first, hit = cached(tmp_path, "kind", {"source": "a"}, make)
    again, second_hit = cached(tmp_path, "kind", {"source": "a"}, make)
    changed, changed_hit = cached(tmp_path, "kind", {"source": "b"}, make)
    assert (hit, second_hit, changed_hit) == (False, True, False)
    assert first == again != changed and built == [first, changed]


# Timeline

def fixture(root: Path, npu1_accept: int = 130, tokens: int = 2,
            validation: str | None = None) -> tuple[Path, Path, Path]:
    """request, prep, NPU prefill overlapping FullCPU work, sample 0, idle gap, NPU decode, sample 1."""
    def span(start: int, end: int) -> Record:
        return {"accepted_ns": {"numerator": start, "denominator": 1},
                "result_ready_ns": {"numerator": end, "denominator": 1},
                "resource_ready_ns": {"numerator": end, "denominator": 1}}

    def cpu(resource: str, start: int, end: int, stage: str) -> Record:
        return {**span(start, end), "worker_intervals": [{"resource": resource, "worker_id": 0,
                "duration_ns": {"numerator": end - start, "denominator": 1}, "source": "steady_clock",
                "unit": "nanosecond", "quantity": end - start,
                "measurement": {"thread_id": 7, "stage": stage, "host_elapsed_ns": end - start,
                                "thread_cpu_valid": False, "cpu_work_cycles": 3 * (end - start),
                                "cpu_work_cycles_valid": True, "cpu_work_cycles_source": "linux_perf_cpu_cycles",
                                "cpu_work_cycles_scope": "user+kernel", "host_cpu_core_start": 1,
                                "host_cpu_core_end": 1 if stage != "norm" else 4,
                                "cpu_migrated": stage == "norm"}}]}
    nodes: list[tuple[str, str, str, Record]] = [
        ("application:request:begin", "BARRIER", "prefill:None", span(0, 0)),
        ("application:prefill-prep:0", "APPLICATION_CPU", "prefill:None", cpu("cpu:potal-lane-0", 0, 10, "prep")),
        ("npu:0", "NPU", '{"decode_index": null, "kind": "prefill"}',
         {**span(10, 110), "accepted_cycle": 10, "evidence_id": "e0"}),
        ("ordinary:0", "CPU", '{"decode_index": null, "kind": "prefill"}', cpu("cpu:fullcpu-worker-0", 20, 60, "norm")),
        ("application:sample:0", "APPLICATION_CPU", "prefill:None", cpu("cpu:potal-lane-0", 110, 115, "sample")),
        ("npu:1", "NPU", '{"decode_index": 0, "kind": "decode"}',
         {**span(npu1_accept, npu1_accept + 50), "accepted_cycle": npu1_accept, "evidence_id": "e1"}),
        ("application:sample:1", "APPLICATION_CPU", "decode:0", cpu("cpu:potal-lane-0", 180, 184, "sample")),
    ][:5 if tokens == 1 else None]
    bundle, schedule, results = root / "ir.sqlite", root / "schedule.sqlite", root / "npu.jsonl"
    with sqlite3.connect(bundle) as ir, sqlite3.connect(schedule) as out:
        ir.execute("CREATE TABLE nodes(identity TEXT PRIMARY KEY, kind TEXT, operation TEXT, body TEXT, pending INTEGER)")
        out.execute("CREATE TABLE results(identity TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, body TEXT)")
        for ordinal, (identity, kind, phase, body) in enumerate(nodes):
            ir.execute("INSERT INTO nodes VALUES(?,?,?,?,0)", (identity, kind, "op", json.dumps(
                {"node_id": identity, "kind": kind, "phase": phase, "operation_id": "op"})))
            out.execute("INSERT INTO results VALUES(?,?,?)", (identity, ordinal, json.dumps({"node_id": identity, **body})))
    authority = {} if validation is None else {"cycle_model_validation": validation}
    results.write_text("".join(json.dumps({"sequence": index, "work_id": index, "layer": "blk.0", "operation": "MUL_MAT",
                                           "provenance": provenance, "scope": scope,
                                           "modeled": {"total_cycles": cycles}, **authority}) + "\n"
                               for index, cycles, provenance, scope in ((0, 100, "dense_main", "stripe"),
                                                                        (1, 50, "residual", "residual_compact"))))
    return schedule, bundle, results


def timeline(root: Path, **options: int) -> tuple[list[Record], Record, Record]:
    schedule, bundle, results = fixture(root, **options)
    path = root / "timeline.jsonl"
    write_timeline(path, timeline_rows(schedule, bundle, results, Axis(1_000_000_000, False)))
    result, checks = performance(path, 2, None)
    return [json.loads(line) for line in path.read_text().splitlines()], result, checks


def test_timeline_keeps_overlap_idle_gap_order_and_token_boundaries(tmp_path: Path) -> None:
    rows, result, checks = timeline(tmp_path)
    npu = [row for row in rows if row["kind"] == "npu"]
    # Then NPU rows sit exactly where the scheduler put them: no packing, no summing.
    assert [(row["start_cycle"], row["end_cycle"]) for row in npu] == [(10, 110), (130, 180)]
    assert checks["status"] == "PASS" and checks["npu_order_or_overlap_violations"] == 0
    assert checks["cpu_npu_overlap_cycles"] == 40 and checks["npu_idle_cycles_in_request"] == 184 - 150
    assert all(row["start_ns"] is None and row["npu_ms"] is None for row in npu)
    cpu = [row for row in rows if row["kind"] == "cpu"]
    assert {row["source"] for row in cpu} == {"application", "fullcpu"}
    assert all(row["target_cpu_cycles"] is None and row["timing_source"] == "HOST_MEASURED_DEVELOPMENT" for row in cpu)
    assert result["ttft"]["npu_cycles"] == 100 and result["ttft"]["e2e_cycles"] is None
    assert result["tpot"]["per_token_npu_cycles"] == [50] and result["tpot"]["e2e_tpot_cycles"] is None
    assert result["ttft"]["interface_status"] == "UNMODELED" and result["ttft"]["interface_cycles"] is None
    assert result["diagnostic_schedule"]["ttft_axis_cycles"] == 115


def test_overlapping_npu_work_and_token_boundary_breaks_fail(tmp_path: Path) -> None:
    # Given a decode NPU work accepted before token 0 is ready and before the prefill work released the NPU.
    _, _, checks = timeline(tmp_path, npu1_accept=100)
    assert checks["status"] == "FAIL"
    assert checks["npu_order_or_overlap_violations"] == 1 and checks["token_boundary_violations"] == [1]


def test_validated_clock_fills_ns_and_ms_only_then(tmp_path: Path) -> None:
    schedule, bundle, results = fixture(tmp_path)
    path = tmp_path / "timeline.jsonl"
    write_timeline(path, timeline_rows(schedule, bundle, results, Axis(1_000_000_000, True)))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result, _ = performance(path, 2, 500_000_000)
    assert all(row["start_ns"] is not None for row in rows)
    assert result["ttft"]["npu_ms"] == pytest.approx(100 * 1000 / 500_000_000)


def test_lane_packing_never_merges_concurrent_threads() -> None:
    lanes = pack_lanes({("x", 1): (0, 100), ("x", 2): (10, 20), ("x", 3): (21, 30), ("x", 4): (20, 25)})
    assert lanes[("x", 1)] != lanes[("x", 2)] and lanes[("x", 2)] != lanes[("x", 4)]
    assert lanes[("x", 3)] == lanes[("x", 2)]


# Publication (fail closed)

WORKLOAD: Record = {"model_sha256": "m", "input_tokens_sha256": "i", "generated_tokens_sha256": "g",
                    "profile": "a8w8-d32-hp1"}


def artifacts(root: Path, **changes: Json) -> Record:
    evidence = root / "raw-evidence.bin"
    evidence.write_bytes(b"target counters")
    reference = {"path": str(evidence), "sha256": sha256(evidence)}
    host: Record = {"schema": "potal-target-host-timing", "version": 1,
                    "target": {"host_id": "target", "system": "Linux", "machine": "aarch64", "board": "fpga",
                               "cpu_model": "a53"},
                    "measurement": {"kind": "TARGET_MEASURED", "unit": "cycle", "counter": "PMCCNTR"},
                    "clock": {"cpu_frequency_hz": 1_200_000_000, "provenance": "board clock tree"},
                    "cpu_service_count": 3, "evidence": reference, "workload": dict(WORKLOAD)}
    interface: Record = {"schema": "potal-target-interface-cost", "version": 1, "status": "MEASURED",
                         "stages": {stage: {"cycles": 10, "provenance": "DMA counter"} for stage in
                                    ("interface.input_transport", "interface.output_transport")},
                         "evidence": reference, "workload": dict(WORKLOAD)}
    documents = {"target_host_timing": host, "target_interface_cost": interface,
                 "clock_selection": {"selected_frequency_hz": 800_000_000}}
    for name, value in changes.items():
        documents[name.split("__")[0]] = value if "__" not in name else {**documents[name.split("__")[0]],
                                                                          name.split("__")[1]: value}
    inputs: Record = {}
    for name, document in documents.items():
        if document is None:
            continue
        path = root / (name + ".json")
        write_json(path, document)
        inputs[name] = {"path": str(path), "sha256": sha256(path)}
    return inputs


def readiness(root: Path, hosts: set[str] | None = None, **changes: Json) -> Record:
    with patch.object(admission, "clock_frequency", return_value=800_000_000):
        return admission.publication_readiness(artifacts(root, **changes), WORKLOAD, hosts or {"target"}, True, IM2P)


def test_all_admitted_evidence_is_the_only_ready_publication(tmp_path: Path) -> None:
    ready = readiness(tmp_path)
    assert ready["TARGET_LATENCY_READY"] is True and ready["unmodeled_target_cost_count"] == 0


@pytest.mark.parametrize(("changes", "hosts", "code"), [
    ({"clock_selection": None}, None, "NOT_READY_MISSING_OPERATING_CLOCK"),
    ({"target_host_timing": None}, None, "NOT_READY_MISSING_TARGET_HOST_ADMISSION"),
    ({"target_interface_cost": None}, None, "NOT_READY_MISSING_TARGET_INTERFACE_COST"),
    ({"target_host_timing__schema": "forged"}, None, "NOT_READY_INVALID_TARGET_HOST_TIMING"),
    ({"target_host_timing__workload": {**WORKLOAD, "model_sha256": "other"}}, None, "NOT_READY_WORKLOAD_IDENTITY_MISMATCH"),
    ({"target_host_timing": None}, {"mac-development-host"}, "NOT_READY_MISSING_TARGET_HOST_ADMISSION"),
    ({"target_host_timing__target": {"host_id": "target", "system": "Darwin", "machine": "arm64", "board": "mac",
                                     "cpu_model": "m"}}, None, "NOT_READY_INVALID_TARGET_HOST_TIMING"),
    ({}, {"mac-development-host"}, "NOT_READY_INVALID_TARGET_HOST_TIMING"),
    ({"target_interface_cost__status": "UNMODELED"}, None, "NOT_READY_TARGET_INTERFACE_UNMODELED"),
])
def test_publication_rejects_missing_or_invalid_evidence(tmp_path: Path, changes: dict[str, Json],
                                                         hosts: set[str] | None, code: str) -> None:
    ready = readiness(tmp_path, hosts, **changes)
    assert ready["TARGET_LATENCY_READY"] is False
    assert any(str(item).startswith(code) for item in ready["codes"]), ready["codes"]
    with pytest.raises(EvaluationError, match="service publication NOT_READY"):
        publication_row({"target_latency": "NOT_READY"}, ready)


def test_invalid_clock_and_unmodeled_interface_are_counted(tmp_path: Path) -> None:
    with patch.object(admission, "clock_frequency", side_effect=ValueError("not a post-route clock")):
        ready = admission.publication_readiness(artifacts(tmp_path, target_interface_cost=None), WORKLOAD,
                                                {"target"}, True, IM2P)
    assert ready["clock_ready"] is False and ready["unmodeled_target_cost_count"] == 2
    assert any(str(code).startswith("NOT_READY_INVALID_OPERATING_CLOCK") for code in ready["codes"])


def test_reconstruction_keeps_the_target_latency_deficiency_marker() -> None:
    row = reconstructed_row({"target_latency": "NOT_READY_REQUIRES_CERTIFIED_REPLAY_JOIN_CLOCK_AND_SERVICE_PROOF"},
                            {"ttft_ns": 1}, {"schema": "proof"})
    assert row["measurement_kind"] == "VALIDATED_RECONSTRUCTION"
    assert row["target_latency"].startswith("NOT_READY")


def test_build_identity_binds_the_include_repo_header_bytes(tmp_path: Path) -> None:
    # Given the sibling Gemmini include repo on the CMake include path.
    headers = include_headers()
    # Then gemmini.h/gemmini_params.h bytes are part of the source identity (cache key), not only git HEAD.
    assert headers["gemmini.h"] == sha256(INCLUDE_REPO / "gemmini.h")
    assert headers["gemmini_params.h"] == sha256(INCLUDE_REPO / "gemmini_params.h")
    assert source_state()["RISC-V-DynDNN-gemmini-include"]["headers"] == headers
    # And an untracked header edit changes the identity although HEAD does not move.
    copy = tmp_path / "include-copy"
    (copy / "include").mkdir(parents=True)
    for name in ("gemmini.h", "gemmini_params.h"):
        (copy / name).write_bytes((INCLUDE_REPO / name).read_bytes())
    before = include_headers(copy)
    (copy / "gemmini.h").write_bytes((INCLUDE_REPO / "gemmini.h").read_bytes() + b"\n")
    assert include_headers(copy) != before


def test_smoke_single_token_has_ttft_and_no_tpot_under_the_local_authority(tmp_path: Path) -> None:
    # Given a 256+1 smoke timeline whose NPU results are NANO_LOCAL_VALIDATED.
    schedule, bundle, results = fixture(tmp_path, tokens=1, validation="NANO_LOCAL_VALIDATED")
    path = tmp_path / "timeline.jsonl"
    write_timeline(path, timeline_rows(schedule, bundle, results, Axis(1_000_000_000, False)))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result, checks = performance(path, 1, None, "NANO_LOCAL_VALIDATED")
    # Then TTFT exists, TPOT is NOT_APPLICABLE (never zero or TTFT), and nothing claims certification.
    assert checks["status"] == "PASS" and result["ttft"]["npu_cycles"] == 100
    assert result["tpot"]["status"] == "NOT_APPLICABLE" and result["tpot"]["npu_tpot_cycles"] is None
    assert result["tpot"]["per_token_npu_cycles"] == [] and result["tpot"]["npu_tpot_ms"] is None
    assert "not certified" in result["ttft"]["readiness"]["npu_cycles"]
    npu = [row for row in rows if row["kind"] == "npu"]
    assert [row["timing_source"] for row in npu] == ["CYCLE_SIM_ISOLATED_NANO_LOCAL_VALIDATED"]
    # And synthetic scheduler lanes are never CPU cores.
    cpu = [row for row in rows if row["kind"] == "cpu"]
    assert all(row["resource_class"] == "SYNTHETIC_SCHEDULER_LANE" and row["scheduler_lane"] == row["resource"] and
               row["host_cpu_core_start"] == 1 and row["target_cpu_core"] is None and row["host_thread_id"] == 7
               for row in cpu)
    assert all(row["resource_class"] == "NPU" and row["scheduler_lane"] is None for row in npu)


def test_certified_results_keep_the_certified_npu_label(tmp_path: Path) -> None:
    schedule, bundle, results = fixture(tmp_path, validation="CURRENT_CERTIFIED")
    path = tmp_path / "timeline.jsonl"
    write_timeline(path, timeline_rows(schedule, bundle, results, Axis(1_000_000_000, False)))
    npu = [json.loads(line) for line in path.read_text().splitlines() if '"kind":"npu"' in line]
    assert {row["timing_source"] for row in npu} == {"CYCLE_SIM_ISOLATED_CERTIFIED"}
    result, _ = performance(path, 2, None)
    assert result["ttft"]["readiness"]["npu_cycles"] == "READY: certified isolated cycle simulator"


@pytest.mark.parametrize("paranoid, user_access, missing", [("2", "1", ["perf_event_paranoid=1"]),
                                                            ("1", "0", ["perf_user_access=1"]),
                                                            ("2", "0", ["perf_event_paranoid=1", "perf_user_access=1"])])
def test_pmu_preflight_fails_fast_with_the_settings_to_apply(paranoid: str, user_access: str, missing: list[str]) -> None:
    values = {"perf_event_paranoid": paranoid, "perf_user_access": user_access}
    with patch("run_cycle_evaluation.platform_profile", return_value=platform_profile("Linux", "aarch64")), \
         patch.object(Path, "is_file", lambda self: self.name in values), \
         patch.object(Path, "read_text", lambda self, *args, **kwargs: values[self.name] + "\n"):
        with pytest.raises(EvaluationError) as failure:
            pmu_contract()
    assert all("sudo sysctl kernel." + setting in str(failure.value) for setting in missing)


PROBE_PASS = {"status": "PASS", "exclude_kernel": 0, "cap_user_rdpmc": 1, "index": 32, "direct_read_delta_cycles": 5}


def test_pmu_preflight_records_the_measurement_contract() -> None:
    values = {"perf_event_paranoid": "1", "perf_user_access": "1", "current_clocksource": "arch_sys_counter"}
    with patch("run_cycle_evaluation.platform_profile", return_value=platform_profile("Linux", "aarch64")), \
         patch("run_cycle_evaluation.pmu_probe", return_value=PROBE_PASS), \
         patch("run_cycle_evaluation.cpu_model", return_value="Cortex-A78AE"), \
         patch.object(Path, "is_file", lambda self: self.name in values), \
         patch.object(Path, "read_text", lambda self, *args, **kwargs: values[self.name] + "\n"):
        contract = pmu_contract()
    assert contract["status"] == "READY" and contract["exclude_kernel"] == 0 and contract["cycle_scope"] == "user + kernel"
    assert contract["kernel.perf_event_paranoid"] == 1 and contract["kernel.perf_user_access"] == 1
    assert contract["probe"] == PROBE_PASS and contract["cpu_model"] == "Cortex-A78AE"
    assert contract["clocksource"] == "arch_sys_counter"
    with patch("run_cycle_evaluation.platform_profile", return_value=platform_profile("Darwin", "arm64")):
        assert pmu_contract()["status"] == "NOT_APPLICABLE"


@pytest.mark.parametrize("probe", [{"status": "FAILED", "stage": "perf_event_open", "errno": 13},
                                   {"status": "FAILED", "stage": "direct_read_index", "index": 0}])
def test_pmu_preflight_fails_fast_when_the_collector_probe_fails(probe: dict[str, object]) -> None:
    values = {"perf_event_paranoid": "1", "perf_user_access": "1"}
    with patch("run_cycle_evaluation.platform_profile", return_value=platform_profile("Linux", "aarch64")), \
         patch("run_cycle_evaluation.pmu_probe", return_value=probe), \
         patch.object(Path, "is_file", lambda self: self.name in values), \
         patch.object(Path, "read_text", lambda self, *args, **kwargs: values[self.name] + "\n"):
        with pytest.raises(EvaluationError, match="collector-equivalent PMU probe"):
            pmu_contract()


def test_real_pmu_probe_on_this_host() -> None:
    from run_cycle_evaluation import pmu_probe
    if platform_profile().machine != "aarch64":
        pytest.skip("aarch64 collector probe")
    result = pmu_probe()
    assert result["status"] in ("PASS", "FAILED") and len(str(result["source_sha256"])) == 64


def test_timeline_cpu_rows_carry_observed_cores_and_pmu_cycles(tmp_path: Path) -> None:
    rows, result, _ = timeline(tmp_path)
    cpu = {row["op"]: row for row in rows if row["kind"] == "cpu"}
    # Then a migrated interval keeps both cores and no single core; the lane is never a core.
    assert (cpu["norm"]["host_cpu_core_start"], cpu["norm"]["host_cpu_core_end"], cpu["norm"]["cpu_migrated"],
            cpu["norm"]["host_cpu_core"]) == (1, 4, True, None)
    assert (cpu["prep"]["host_cpu_core"], cpu["prep"]["cpu_migrated"]) == (1, False)
    assert cpu["prep"]["cpu_cycles"] == 30 and cpu["prep"]["cpu_cycle_scope"] == "user+kernel"
    assert all(row["scheduler_lane"] == row["resource"] and row["target_cpu_core"] is None for row in cpu.values())
    # And TTFT/TPOT expose separate CPU (host-measured) and NPU (service cycles) components, never summed.
    ttft = result["ttft"]["cpu_measured"]
    assert ttft["cpu_service_cycles"] == 3 * (10 + 40 + 5) and ttft["cpu_cycle_rows_invalid"] == 0
    assert ttft["cpu_host_elapsed_ns"] == 55 and ttft["cpu_thread_ns"] is None
    assert result["ttft"]["npu_service_cycles"] == 100 and result["ttft"]["e2e_cycles"] is None
    assert result["tpot"]["cpu_measured_per_token"][0]["cpu_service_cycles"] == 12
    assert result["tpot"]["npu_service_cycles_per_token"] == [50]


def collection(root: Path, rows: list[dict[str, object]]) -> Path:
    chunk = root / "native/chunk-0"
    chunk.mkdir(parents=True)
    (chunk / "cycle-log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "native/application-cpu.jsonl").write_text("")
    return root


def timed(valid: bool = True, start: object = 0, end: object = 0, service: bool = True) -> dict[str, object]:
    return {"cpu_service": service, "cpu_work_cycles_valid": valid,
            "cpu_work_cycles_source": "linux_perf_cpu_cycles" if valid else None,
            "cpu_work_cycles_reason": None if valid else "unavailable_sample",
            "cpu_work_cycles_sample_reason": None if valid else "unavailable_direct_mapping",
            "host_cpu_core_start": start, "host_cpu_core_end": end}


def test_timing_validity_separates_applicable_invalid_and_non_applicable_rows(tmp_path: Path) -> None:
    rows = [timed(start=0, end=0), timed(start=1, end=4), timed(valid=False, start=2, end=2),
            timed(service=False), {"record_type": "MATMUL_CONFIGURATION"}]
    summary = timing_validity(collection(tmp_path, rows), True)
    assert (summary["applicable"], summary["valid"], summary["invalid"], summary["non_applicable"]) == (3, 2, 1, 2)
    assert summary["invalid_reasons"] == {"unavailable_direct_mapping": 1} and summary["cycle_log_ready"] is False
    cores = summary["cores"]
    assert (cores["migrated"], cores["not_migrated"]) == (1, 2) and cores["start_end_matrix"]["1->4"] == 1
    assert cores["migration_ratio"] == pytest.approx(1 / 3)


def test_instrumentation_markers_fail_closed_on_invalid_cpu_timing(tmp_path: Path) -> None:
    good = timing_validity(collection(tmp_path / "a", [timed(start=0, end=0)]), True)
    bad = timing_validity(collection(tmp_path / "b", [timed(valid=False, start=0, end=0)]), True)
    unknown = timing_validity(collection(tmp_path / "c", [timed(start=None, end=0)]), True)
    result = {"ttft": {"npu_cycles": 5, "cpu_measured": {"cpu_service_cycles": 3, "cpu_thread_ns": 1}}, "tpot": {}}
    pmu = {"status": "READY"}
    ready = instrumentation_markers({"pmu": pmu, "potal": good, "fullcpu": good}, result, 1)
    assert ready["CPU_STAGE_CYCLE_LOG_READY"] == ready["CPU_CORE_TIMELINE_READY"] == "READY"
    assert ready["TPOT_CPU_COMPONENT_READY"] == ready["TPOT_NPU_COMPONENT_READY"] == "NOT_APPLICABLE"
    invalid = instrumentation_markers({"pmu": pmu, "potal": good, "fullcpu": bad}, result, 1)
    assert invalid["CPU_STAGE_CYCLE_LOG_READY"] == invalid["TTFT_CPU_COMPONENT_READY"] == "NOT_READY"
    malformed = instrumentation_markers({"pmu": pmu, "potal": good, "fullcpu": unknown}, result, 1)
    assert malformed["CPU_CORE_TIMELINE_READY"] == "NOT_READY"
    missing = instrumentation_markers({"potal": good, "fullcpu": good}, result, 1)
    assert missing["NANO_HOST_PMU_READY"] == missing["CPU_STAGE_CYCLE_LOG_READY"] == "NOT_READY"


# Offline stage cache

def test_stage_cache_reuses_only_the_identical_identity_and_verifies_bytes(tmp_path: Path) -> None:
    from stage_cache import StageCache
    cache = StageCache(tmp_path / "cache")
    calls: list[str] = []

    def produce(value: str):
        def run(entry: Path) -> None:
            calls.append(value)
            (entry / "out.txt").write_text(value)
        return run
    first = cache.run("stage", {"input": "a"}, ("out.txt",), tmp_path / "run1", produce("a"))
    again = cache.run("stage", {"input": "a"}, ("out.txt",), tmp_path / "run2", produce("a"))
    other = cache.run("stage", {"input": "b"}, ("out.txt",), tmp_path / "run3", produce("b"))
    # Then only the identical identity is served from the cache, and every run receives the published bytes.
    assert (first["cache_hit"], again["cache_hit"], other["cache_hit"]) == (False, True, False) and calls == ["a", "b"]
    assert (tmp_path / "run2/out.txt").read_text() == "a" and (tmp_path / "run3/out.txt").read_text() == "b"
    assert (tmp_path / "run2/stage.stage-receipt.json").is_file()
    # And published outputs are read-only; a changed entry is refused instead of served.
    entry = Path(str(first["entry"]))
    assert not ((entry / "out.txt").stat().st_mode & 0o222)
    (entry / "out.txt").chmod(0o644)
    (entry / "out.txt").write_text("tampered")
    with pytest.raises(EvaluationError, match="changed after publication"):
        cache.run("stage", {"input": "a"}, ("out.txt",), tmp_path / "run4", produce("a"))


def test_stage_cache_discards_an_interrupted_entry_and_rejects_missing_outputs(tmp_path: Path) -> None:
    from stage_cache import StageCache

    def crash(entry: Path) -> None:
        (entry / "partial.txt").write_text("x")
        raise RuntimeError("interrupted")
    cache = StageCache(tmp_path / "cache")
    with pytest.raises(RuntimeError):
        cache.run("stage", {"input": "a"}, ("out.txt",), tmp_path / "run1", crash)
    result = cache.run("stage", {"input": "a"}, ("out.txt",), tmp_path / "run2",
                       lambda entry: (entry / "out.txt").write_text("ok"))
    assert result["cache_hit"] is False and (tmp_path / "run2/out.txt").read_text() == "ok"
    with pytest.raises(EvaluationError, match="did not produce"):
        cache.run("stage", {"input": "c"}, ("out.txt",), tmp_path / "run3", lambda entry: None)


def test_every_stage_identity_carries_its_own_and_all_upstream_code_digests() -> None:
    import run_cycle_evaluation as runner
    # Then each stage is keyed on the code of every stage whose output it reads, transitively.
    assert set(runner.STAGE_CODE) == set(runner.STAGE_UPSTREAM) == set(runner.OFFLINE_STAGES)
    assert set(runner.stage_code("replay")) == {"im2p_code"}
    assert set(runner.stage_code("worker-scenario")) == {"timeline_code"}
    for stage in ("producer-lifecycle", "execution-ir", "schedule", "timeline"):
        assert set(runner.stage_code(stage)) == {"im2p_code", "timeline_code"}
    for stage, upstream in runner.STAGE_UPSTREAM.items():
        assert all(set(runner.stage_code(parent)) <= set(runner.stage_code(stage)) for parent in upstream)


def test_upstream_source_change_misses_downstream_stages_with_identical_outputs(tmp_path: Path) -> None:
    import argparse

    import run_cycle_evaluation as runner
    from stage_cache import StageCache

    class Run:
        def __init__(self) -> None:
            self.cache: list[Json] = []

        def step(self, name: str, action):  # type: ignore[no-untyped-def]
            return action()

        def write(self, status: str) -> None:
            pass
    cache, calls = StageCache(tmp_path / "cache"), []

    def timeline(im2p: str, target: str) -> Record:
        def produce(entry: Path) -> None:
            calls.append(target)
            (entry / "timeline.jsonl").write_text("same bytes")
        with patch.object(runner, "source_digest", lambda root, patterns: im2p if root == runner.IM2P else "timeline"):
            return runner.cached_stage(argparse.Namespace(stop_after=None), Run(), cache, "timeline",  # type: ignore[arg-type]
                                       {"schedule": "identical-schedule"}, ("timeline.jsonl",), tmp_path / target, produce)
    # Given: the same schedule bytes before and after an IM2P (scheduler) source change.
    first, again, changed = timeline("a", "run1"), timeline("a", "run2"), timeline("b", "run3")
    # Then: the timeline is reused only under the unchanged upstream source and rebuilt after the change.
    assert (first["cache_hit"], again["cache_hit"], changed["cache_hit"]) == (False, True, False)
    assert calls == ["run1", "run3"]
    with (pytest.raises(EvaluationError, match="stale code digest"),
          patch.object(runner, "source_digest", lambda root, patterns: "current")):
        runner.cached_stage(argparse.Namespace(stop_after=None), Run(), cache, "schedule",  # type: ignore[arg-type]
                            {"im2p_code": "old"}, ("x",), tmp_path / "run4", lambda entry: None)


def test_hash_memo_rehashes_when_the_file_changes(tmp_path: Path) -> None:
    from stage_cache import HashMemo
    path = tmp_path / "input.bin"
    path.write_bytes(b"one")
    memo = HashMemo(tmp_path / "memo.json")
    assert memo.sha256(path) == sha256(path)
    path.write_bytes(b"two!")
    assert HashMemo(tmp_path / "memo.json").sha256(path) == sha256(path)



# Execution modes (performance / timeline export) and the shared schedule engine

def completed_run(root: Path) -> tuple[Path, tuple[Path, Path, Path]]:
    """A completed performance run whose stored schedule, IR and NPU results live in stage-cache entries."""
    import shutil
    (root / "fixture").mkdir()
    files = fixture(root / "fixture")
    run = root / "run"
    (run / "replay").mkdir(parents=True)
    entries: list[Json] = []
    for stage, path, name in (("schedule", files[0], "schedule.sqlite"), ("execution-ir", files[1], "execution.sqlite"),
                              ("replay", files[2], "npu-cycle-result.jsonl")):
        entry = root / "cache" / stage
        entry.mkdir(parents=True)
        shutil.copy(path, entry / name)
        entries.append({"stage": stage, "key": stage, "cache_hit": False, "entry": str(entry),
                        "outputs": {name: {"sha256": sha256(entry / name), "bytes": (entry / name).stat().st_size}}})
    write_json(run / "manifest.json", {"status": "PASS", "stage_cache": entries, "workload": {"workload_mode": "X"}})
    write_json(run / "performance.json", {"clock": {"schedule_clock_hz": 1_000_000_000,
                                                    "status": "DIAGNOSTIC_CONFIGURED_TEST_CLOCK"}})
    return run, files


def forbidden(*names: str):  # type: ignore[no-untyped-def]
    import contextlib

    import run_cycle_evaluation as runner
    stack = contextlib.ExitStack()
    for name in names:
        stack.enter_context(patch.object(runner, name, side_effect=AssertionError("must not run: " + name)))
    return stack


PIPELINE = ("run_schedule", "reconstruct_local", "reconstruct", "collect", "build_all", "pmu_contract",
            "cached_cycle_model", "cached_llama_build")


def test_timeline_export_reads_only_the_stored_schedule(tmp_path: Path) -> None:
    import argparse

    import run_cycle_evaluation as runner
    source, (schedule, bundle, results) = completed_run(tmp_path)
    expected = tmp_path / "expected.jsonl"
    write_timeline(expected, timeline_rows(schedule, bundle, results, Axis(1_000_000_000, False)))
    # When: only the timeline is exported from the completed run.
    with forbidden(*PIPELINE):
        code = runner.export_timeline(argparse.Namespace(from_run=source, output=tmp_path / "export", format="jsonl"))
    # Then: the export equals the timeline of that stored schedule and nothing else ran.
    assert code == 0 and (tmp_path / "export/timeline.jsonl").read_bytes() == expected.read_bytes()
    export = json.loads((tmp_path / "export/export.json").read_text())
    assert export["reuse"] == "TIMELINE_EXPORT_FROM_STORED_SCHEDULE" and "scheduling" in export["not_run"]
    assert json.loads((tmp_path / "export/manifest.json").read_text())["markers"] == {
        "MEASUREMENT_DOMAIN": "timeline", "TIMELINE_EXPORT_ONLY": "PASS", "PERFORMANCE_RUN": "NOT_RUN",
        "HARDWARE_METRICS_RUN": "NOT_RUN"}
    # And: the export directory is sealed like every other run directory.
    sums = dict(reversed(line.split("  ", 1)) for line in (tmp_path / "export/SHA256SUMS").read_text().splitlines())
    assert set(sums) == {"export.json", "manifest.json", "timeline-rows.json", "timeline.jsonl"}
    assert sums["timeline.jsonl"] == export["timeline"]["sha256"]
    assert all(sha256(tmp_path / "export" / name) == value for name, value in sums.items())
    # And: a changed or missing stored schedule fails instead of being recomputed.
    stored = tmp_path / "cache/schedule/schedule.sqlite"
    stored.chmod(0o644)
    with sqlite3.connect(stored) as database:
        database.execute("UPDATE results SET body=body WHERE ordinal=0")
        database.execute("CREATE TABLE extra(x)")
    with forbidden(*PIPELINE):
        assert runner.export_timeline(argparse.Namespace(from_run=source, output=tmp_path / "e2", format="jsonl")) == 1
    assert "differs from its stage receipt" in json.loads((tmp_path / "e2/manifest.json").read_text())["stages"][-1]["reason"]
    stored.unlink()
    with forbidden(*PIPELINE):
        assert runner.export_timeline(argparse.Namespace(from_run=source, output=tmp_path / "e3", format="jsonl")) == 1
    assert "is gone" in json.loads((tmp_path / "e3/manifest.json").read_text())["stages"][-1]["reason"]


def test_engine_modes_are_performance_and_timeline_only(capsys: pytest.CaptureFixture[str]) -> None:
    import run_cycle_evaluation as runner
    assert runner.RUN_MODES == ("performance", "timeline")
    for mode in ("metrics", "quality", "perplexity"):
        with patch.object(sys, "argv", ["run_cycle_evaluation.py", "--run", mode]), pytest.raises(SystemExit):
            runner.arguments()
        assert "invalid choice" in capsys.readouterr().err
    # The help names only the two modes and nothing of perplexity.
    with patch.object(sys, "argv", ["run_cycle_evaluation.py", "--help"]), pytest.raises(SystemExit):
        runner.arguments()
    shown = capsys.readouterr().out
    assert "{performance,timeline}" in shown and not re.search(r"\b(perplexity|ppl|quality)\b", shown.lower())
    # And: no perplexity code path is left in the engine.
    assert not [name for name in dir(runner) if re.search(r"perplexity|(^|_)ppl(_|$)|quality", name.lower())]


def test_performance_builds_only_the_collection_builds(tmp_path: Path) -> None:
    import argparse

    import run_cycle_evaluation as runner
    if str(IM2P) not in sys.path:
        sys.path.insert(0, str(IM2P))
    kinds: list[str] = []
    library = tmp_path / "cycle"
    library.mkdir()
    write_json(library / "build-receipt.json", {"library": {"path": str(library / "lib.so"), "sha256": "x"}})

    def llama(cache: Path, kind: str, *rest: object) -> tuple[Path, bool]:
        kinds.append(kind)
        path = tmp_path / kind
        path.mkdir(exist_ok=True)
        for name in ("build-receipt.json", "build-info.json"):
            write_json(path / name, {"kind": kind}) if not (path / name).exists() else None
        return path, True
    args = argparse.Namespace(build_cache=tmp_path / "build-cache", jobs=1, validation_mode="nano-local",
                              local_validation=tmp_path / "receipt.json", precision="a8w8", dim=32)
    for collect_now, expected in ((True, ["potal-host", "fullcpu-host"]), (False, [])):
        kinds.clear()
        run = runner.Run(tmp_path / f"run-{collect_now}", {})
        run.root.mkdir()
        with patch.object(runner, "cached_cycle_model", side_effect=AssertionError("nano-local uses the admitted library")), \
                patch.object(runner, "admitted_cycle_model", return_value=(library, True)), \
                patch.object(runner, "cached_llama_build", side_effect=llama), \
                patch("sim.cycle.local_validation.admit_library", return_value=({}, "NANO_LOCAL_VALIDATED")):
            builds = runner.build_all(args, run, collect=collect_now)
        # Then: only the collection builds (metric sinks off); never a metric kind or an extra target.
        assert kinds == expected and set(builds) == {"cycle-model", *("potal", "fullcpu")[:len(expected)]}


@pytest.mark.parametrize("timeline", ["none", "compact"])
def test_performance_run_writes_a_timeline_only_on_request(tmp_path: Path, timeline: str,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    import argparse

    import run_cycle_evaluation as runner
    model, dataset, receipt, manifest = (tmp_path / name for name in ("m.gguf", "wiki.raw", "receipt.json", "manifest.json"))
    for path in (model, dataset, receipt, manifest):
        path.write_text(path.name)
    replay = tmp_path / "replay"
    replay.mkdir()
    write_json(replay / "schedule-performance.json", {"result": {"ttft": {"npu_cycles": 7}, "tpot": {}},
                                                      "checks": {"status": "PASS"}, "rows": 3})
    write_json(replay / "npu-summary.json", {})
    write_json(replay / "result.json", {"replay_mode": "LOCAL"})
    pending: list[Path | None] = []

    def scheduled(*call: object) -> Path:
        pending.append(call[5])  # type: ignore[arg-type]
        return replay / "schedule.sqlite"

    def exported(*call: object) -> int:
        (call[7] / "timeline.jsonl").write_text("rows")  # type: ignore[operator]
        return 3
    args = argparse.Namespace(
        model=model, prompt_file=dataset, from_run=None, prompt_tokens=256, generate=128, smoke=True,
        validation_mode="nano-local", certificates=None, local_validation=receipt,
        model_manifest=manifest, output=tmp_path / "run", profile="a8w8-d32-hp1", reuse_collection=tmp_path,
        stage_cache=None, keep_raw=True, clock_selection=None, target_host_timing=None, target_interface_cost=None,
        timeline=timeline, precision="a8w8", dim=32)
    reuse: Record = {"model_sha256": "model", "pmu_contract": {"status": "READY"}, "source_run": "collection"}
    builds: Record = {"cycle-model": {"receipt": {"library": {"path": str(tmp_path / "lib.so")}}}}
    with forbidden("export_timeline", "collect", "pmu_contract", "cached_llama_build", "cached_cycle_model"), \
            patch.object(runner, "clean_environment"), patch.object(runner, "host_facts", return_value={}), \
            patch.object(runner, "toolchain", return_value={}), patch.object(runner, "source_state", return_value={}), \
            patch.object(runner, "gguf_identity", return_value={"sha256": "model"}), \
            patch.object(runner, "model_entry", return_value={"artifact": "Q8_HP1"}), \
            patch.object(runner, "reused_collection", return_value=(tmp_path, tmp_path, reuse)), \
            patch.object(runner, "build_all", return_value=builds), \
            patch.object(runner, "reconstruct_local", return_value=replay), \
            patch.object(runner, "collection_identity", return_value={
                "input_tokens_sha256": "i", "generated_tokens_sha256": "g", "host_id": "h"}), \
            patch.object(runner, "timing_validity", return_value={}), \
            patch.object(runner, "schedule", side_effect=scheduled), \
            patch.object(runner, "timeline_stage", side_effect=exported) as timeline_stage, \
            patch.object(runner, "cross_check", return_value={}), \
            patch.object(runner, "publication_readiness", return_value={
                "clock_ready": False, "target_host_ready": False, "target_interface_ready": False}), \
            patch.object(runner, "instrumentation_markers", return_value={}):
        assert runner.performance_run(args) == 0
    root = tmp_path / "run"
    markers = json.loads((root / "manifest.json").read_text())["markers"]
    # Then: stdout is the component/E2E summary of performance.json, and a reused collection is named as reused.
    shown = json.loads(capsys.readouterr().out)
    assert set(shown) == {"output", "markers", "ttft_npu_cycles", "npu_tpot_cycles", "ttft", "tpot", "overlap",
                          "measurement"}
    assert shown["output"] == str(root) and shown["markers"] == markers
    assert shown["ttft"]["npu_service_cycles"] == 7 and shown["ttft"]["scheduled_e2e_cycles"] is None
    # And: the legacy top-level keys are aliases of the structured NPU values; an absent source stays null, not zero.
    assert shown["ttft_npu_cycles"] == 7
    assert shown["npu_tpot_cycles"] is None and shown["tpot"]["npu_service_cycles_mean"] is None
    assert shown["measurement"]["cpu_collection"] == "REUSED_COLLECTION"
    assert shown["measurement"]["publication_ready"] is False and markers["PUBLICATION_E2E_MS_READY"] == "NOT_READY"
    # Then: the run is a performance run only, whatever the timeline switch.
    assert markers["MEASUREMENT_DOMAIN"] == "performance" and markers["HARDWARE_METRICS_RUN"] == "NOT_RUN"
    assert set(markers) >= {"FAST_CYCLE_EVALUATION_RUN", "TIMELINE_LOG_READY", "VALIDATION_MODE"}
    assert not [name for name in markers if "QUALITY" in name or "METRICS_AUTOMATION" in name]
    assert (root / "performance.json").is_file() and not (root / "metrics.json").exists()
    if timeline == "none":
        # And: no timeline sink was given to the scheduling pass and no timeline file or directory exists.
        assert pending == [None] and not timeline_stage.called and not (root / "timeline").exists()
        assert markers["TIMELINE_LOG_READY"] == "NOT_REQUESTED"
        assert not [line for line in (root / "SHA256SUMS").read_text().splitlines() if "timeline" in line.split("  ")[1]]
    else:
        assert pending == [root / "timeline/in-pass-timeline.jsonl.partial"] and timeline_stage.call_count == 1
        assert markers["TIMELINE_LOG_READY"] == "READY" and (root / "timeline/summary.json").is_file()


def test_run_mode_dispatches_to_exactly_one_runner() -> None:
    import run_cycle_evaluation as runner
    for mode, target in (("performance", "performance_run"), ("timeline", "export_timeline")):
        with patch.object(runner, "performance_run", return_value=0) as performance_run, \
                patch.object(runner, "export_timeline", return_value=0) as export_timeline, \
                patch.object(sys, "argv", ["run_cycle_evaluation.py", "--run", mode]):
            assert runner.main() == 0
        called = {"performance_run": performance_run.called, "export_timeline": export_timeline.called}
        assert [name for name, value in called.items() if value] == [target]


def test_schedule_sinks_give_one_result_with_or_without_timeline(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from schedule_engine import ScheduleSinks
    schedule, bundle, results = fixture(tmp_path)
    reference = tmp_path / "reference.jsonl"
    write_timeline(reference, timeline_rows(schedule, bundle, results, Axis(1_000_000_000, False)))
    expected = performance(reference, 2, None, "NANO_LOCAL_VALIDATED", 1_000_000_000)
    outcomes = []
    created: list[tuple[int, int]] = []
    for timeline in (None, tmp_path / "in-pass.jsonl"):
        sinks = ScheduleSinks(results, Axis(1_000_000_000, False), 2, "NANO_LOCAL_VALIDATED", timeline)
        with sqlite3.connect(schedule) as out, sqlite3.connect(bundle) as ir:
            for ordinal, identity, body in out.execute("SELECT ordinal,identity,body FROM results ORDER BY ordinal"):
                kind, node = ir.execute("SELECT kind,body FROM nodes WHERE identity=?", (identity,)).fetchone()
                info = json.loads(node)
                sinks.observe(ordinal, SimpleNamespace(identity=identity, operation=info["operation_id"],
                                                       phase=info["phase"], kind=SimpleNamespace(value=kind)),
                              None, json.loads(body))
        sinks.close()
        outcomes.append(sinks.accumulator.result())
        created.append((sinks.row_objects_created, sinks.written))
    # Then: the timeline switch changes only what is written, and the in-pass rows are the exported timeline.
    assert outcomes[0] == outcomes[1] == expected
    # And: the stored-timeline and in-pass paths attribute the same works to the same categories.
    result = outcomes[0][0]
    assert result["ttft"]["npu_service_breakdown"]["main_gemm_cycles"] == 100
    assert result["ttft"]["npu_service_breakdown"]["status"] == "COMPLETE"
    assert result["tpot"]["npu_service_breakdown_per_token"][0]["residual_gemm_cycles"] == 50
    assert result["npu_service_by_layer"]["tpot_decode_sum"]["blk.0"]["residual_gemm_cycles"] == 50
    assert (tmp_path / "in-pass.jsonl").read_bytes() == reference.read_bytes()
    # And: performance-only creates and writes no timeline row object at all.
    rows = len(reference.read_text().splitlines())
    assert created == [(0, 0), (rows, rows)]


def test_e2e_endpoints_keep_clock_domains_apart_and_counters_beyond_32_bits() -> None:
    from e2e_timeline import PerformanceAccumulator
    from eval_common import record
    big = 5_000_000_000
    accumulator = PerformanceAccumulator(2, None, "NANO_LOCAL_VALIDATED", 1_000_000_000)
    rows: list[Record] = [
        {"kind": "event", "event": "request_start", "token_index": None, "start_cycle": 0, "end_cycle": 0},
        {"kind": "npu", "token_index": 0, "start_cycle": 0, "end_cycle": big, "npu_cycles": big},
        {"kind": "cpu", "token_index": 0, "start_cycle": big, "end_cycle": big + 7, "host_elapsed_ns": big,
         "host_thread_cpu_ns": 3, "cpu_cycles_valid": True, "cpu_cycles": 3 * big},
        {"kind": "event", "event": "token_ready", "token_index": 0, "start_cycle": big + 7, "end_cycle": big + 7},
        {"kind": "npu", "token_index": 1, "start_cycle": big + 7, "end_cycle": 2 * big, "npu_cycles": big - 7},
        {"kind": "event", "event": "token_ready", "token_index": 1, "start_cycle": 2 * big + 1, "end_cycle": 2 * big + 1}]
    for row in rows:
        accumulator.add(row)
    result, checks = accumulator.result()
    e2e, components = record(result["e2e"]), record(result["components"])
    # Then: E2E comes from the scheduled endpoints, the components stay per clock domain, nothing wraps at 2**32.
    assert e2e["ttft"] == big + 7 and e2e["tpot_intervals"] == [big - 6] and e2e["mean_tpot"] == big - 6
    assert record(components["npu"])["npu_service_cycles_sum"] == 2 * big - 7
    assert record(components["cpu"])["cpu_service_cycles_sum"] == 3 * big
    # And: without clock, interface or target-host evidence nothing target-facing is published.
    target = record(e2e["target"])
    assert target["e2e_ms"] is None and target["e2e_cycles"] is None and target["publication_ready"] is False
    assert e2e["ttft_ms"] is None and record(components["npu"])["npu_ms"] is None and checks["status"] == "PASS"


def test_a_new_clock_reschedules_instead_of_rescaling(tmp_path: Path) -> None:
    import run_cycle_evaluation as runner
    from stage_cache import StageCache, digest
    cache = StageCache(tmp_path / "cache")
    bundle, npu = tmp_path / "ir.sqlite", tmp_path / "npu.jsonl"
    bundle.write_bytes(b"ir")
    npu.write_bytes(b"npu")
    keys = {digest(runner.schedule_identity(cache, bundle, npu, axis, 128, "NANO_LOCAL_VALIDATED"))
            for axis in (Axis(1_000_000_000, False), Axis(1_000_000_000, True), Axis(250_000_000, True))}
    assert len(keys) == 3


# Performance stdout: CPU, NPU and overlap-aware scheduled E2E (never CPU cycles + NPU cycles)

HZ = 1_000_000_000
NOT_READY_MARKERS: Record = {"OPERATING_CLOCK_READY": "NOT_READY", "TARGET_CPU_TIMING_READY": "NOT_READY",
                             "TARGET_INTERFACE_COST_READY": "NOT_READY", "PUBLICATION_E2E_MS_READY": "NOT_READY"}


def event(name: str, at: int, token: int | None = None) -> Record:
    return {"kind": "event", "event": name, "token_index": token, "start_cycle": at, "end_cycle": at}


def cpu_row(token: int, start: int, end: int, cycles: int | None = None, thread: int | None = None) -> Record:
    """A CPU service on the schedule axis; its PMU cycles (host clock domain) are deliberately not its duration."""
    return {"kind": "cpu", "token_index": token, "start_cycle": start, "end_cycle": end,
            "host_elapsed_ns": end - start, "host_thread_cpu_ns": thread, "cpu_cycles_valid": cycles is not None,
            "cpu_cycles": cycles}


def npu_row(token: int, start: int, end: int, provenance: str | None = "dense_main", scope: str | None = "stripe",
            layer: str = "blk.0") -> Record:
    return {"kind": "npu", "token_index": token, "start_cycle": start, "end_cycle": end, "npu_cycles": end - start,
            "npu_provenance": provenance, "npu_scope": scope, "layer": layer}


def summarized(rows: list[Record], generated: int, validated_hz: int | None = None, reuse: Record | None = None,
               markers: Record | None = None, publication: Record | None = None) -> tuple[Record, Record]:
    """(stdout summary, performance.json document) of one accumulator pass, assembled as performance_run does."""
    import argparse

    import run_cycle_evaluation as runner
    from e2e_timeline import PerformanceAccumulator
    accumulator = PerformanceAccumulator(generated, validated_hz, "NANO_LOCAL_VALIDATED", validated_hz or HZ)
    for row in rows:
        accumulator.add(row)
    result, checks = accumulator.result()
    document: Record = {
        **result, "timeline_checks": checks,
        "clock": {"schedule_clock_hz": validated_hz or HZ,
                  "status": "VALIDATED_OPERATING_CLOCK" if validated_hz else "DIAGNOSTIC_CONFIGURED_TEST_CLOCK"},
        "publication": publication or {"TARGET_LATENCY_READY": False},
        "execution": runner.execution_record(argparse.Namespace(timeline="none"), reuse, Path("/nonexistent"), [])}
    untouched = json.dumps(document, sort_keys=True)
    summary = runner.performance_summary(Path("/run"), markers or dict(NOT_READY_MARKERS), document)
    assert json.dumps(document, sort_keys=True) == untouched  # the summary only reads performance.json
    # The legacy stdout keys are compatibility aliases of the structured NPU values and of performance.json.
    assert set(summary) >= {"ttft", "tpot", "overlap", "measurement"}
    assert summary["ttft_npu_cycles"] == summary["ttft"]["npu_service_cycles"] == document["ttft"]["npu_cycles"]
    assert summary["ttft"]["npu_service_breakdown"] == document["ttft"]["npu_service_breakdown"]
    assert summary["tpot"]["npu_service_breakdown_mean"] == document["tpot"]["npu_service_breakdown_mean"]
    assert summary["npu_tpot_cycles"] == summary["tpot"]["npu_service_cycles_mean"] == document["tpot"]["npu_tpot_cycles"]
    return summary, document


@pytest.mark.parametrize("cpu, npu, overlap, service_sum", [
    ((0, 10), (10, 30), 0, 30),   # A: no overlap
    ((0, 20), (10, 30), 10, 40),  # B: partial overlap
    ((0, 30), (10, 20), 10, 40),  # C: NPU fully inside the CPU work
])
def test_scheduled_e2e_is_the_endpoint_span_not_the_service_sum(cpu: tuple[int, int], npu: tuple[int, int],
                                                                overlap: int, service_sum: int) -> None:
    summary, _ = summarized([event("request_start", 0), cpu_row(0, *cpu, cycles=7_000), npu_row(0, *npu),
                             event("token_ready", 30, 0)], 1)
    shown = summary["overlap"]
    assert (cpu[1] - cpu[0]) + (npu[1] - npu[0]) == service_sum
    # Then: E2E is request -> token 0 ready on the schedule axis; overlapped work is not counted twice.
    assert summary["ttft"]["scheduled_e2e_cycles"] == 30 == shown["request_span_cycles"]
    assert shown["cpu_npu_overlap_cycles"] == overlap
    assert shown["cpu_busy_union_cycles"] == cpu[1] - cpu[0] and shown["npu_busy_cycles"] == npu[1] - npu[0]
    assert shown["npu_idle_cycles_in_request"] == 30 - (npu[1] - npu[0])
    # And: neither component, nor the PMU count, nor any sum of them is reported as E2E.
    assert summary["ttft"]["cpu_service_cycles"] == 7_000 and summary["ttft"]["npu_service_cycles"] == npu[1] - npu[0]
    assert summary["ttft"]["scheduled_e2e_cycles"] != 7_000 + summary["ttft"]["npu_service_cycles"]
    assert summary["ttft_npu_cycles"] == npu[1] - npu[0]
    # And: the NPU main/residual split is service attribution only; it leaves the scheduled E2E at 30.
    split = summary["ttft"]["npu_service_breakdown"]
    assert split["main_gemm_cycles"] == split["total_cycles"] == npu[1] - npu[0] and split["status"] == "COMPLETE"
    # And: one generated token has no TPOT; its means are null with a reason, never zero.
    assert summary["npu_tpot_cycles"] is None
    assert summary["tpot"]["scheduled_e2e_cycles_mean"] is None and summary["tpot"]["cpu_service_cycles_mean"] is None
    assert summary["tpot"]["cpu_service_cycles_mean_status"].startswith("NOT_APPLICABLE")


DECODE: list[Record] = [
    event("request_start", 0), cpu_row(0, 0, 20, cycles=900, thread=15), npu_row(0, 10, 30), event("token_ready", 30, 0),
    cpu_row(1, 30, 45, cycles=300, thread=10), npu_row(1, 35, 50), event("token_ready", 50, 1),
    cpu_row(2, 50, 75, cycles=500, thread=20), npu_row(2, 60, 90), event("token_ready", 95, 2),
    cpu_row(3, 95, 100, cycles=100, thread=3), npu_row(3, 100, 110), event("token_ready", 110, 3)]


def test_summary_tpot_is_the_token_ready_difference_and_cpu_means_stay_in_their_domain() -> None:
    summary, document = summarized(DECODE, 4)
    ready = document["e2e"]["endpoints"]["token_ready"]
    # Test D: TPOT_i = ready_i - ready_(i-1); the reported mean is (t_last - t_0) / intervals.
    assert ready == [30, 50, 95, 110] and document["e2e"]["tpot_intervals"] == [20, 45, 15]
    assert summary["tpot"]["scheduled_e2e_cycles_mean"] == pytest.approx((110 - 30) / 3)
    assert summary["tpot"]["scheduled_e2e_cycles_mean"] == document["e2e"]["mean_tpot"]
    assert summary["ttft"]["scheduled_e2e_cycles"] == 30 and summary["tpot"]["intervals"] == 3
    # And: NPU and CPU means are the per-interval service means of performance.json, each in its own domain.
    assert summary["tpot"]["npu_service_cycles_mean"] == document["tpot"]["npu_tpot_cycles"] == pytest.approx(55 / 3)
    assert summary["ttft_npu_cycles"] == 20 and summary["npu_tpot_cycles"] == document["tpot"]["npu_tpot_cycles"]
    assert summary["tpot"]["cpu_service_cycles_mean"] == pytest.approx(300) and \
        summary["tpot"]["cpu_service_cycles_mean_status"] == "VALID"
    assert summary["tpot"]["cpu_host_elapsed_ms_mean"] == pytest.approx(45 / 3 / 1e6)
    assert summary["tpot"]["cpu_thread_ms_mean"] == pytest.approx(33 / 3 / 1e6)
    # Test E: PMU cycles pass through unconverted; CPU ms come from host ns, never PMU cycles over the NPU clock.
    assert summary["ttft"]["cpu_service_cycles"] == 900 == document["ttft"]["cpu_measured"]["cpu_service_cycles"]
    assert summary["ttft"]["cpu_host_elapsed_ms"] == pytest.approx(20 / 1e6) != pytest.approx(900 * 1000 / HZ)
    assert summary["ttft"]["cpu_thread_ms"] == pytest.approx(15 / 1e6)
    assert summary["measurement"]["cpu_clock_domain"].startswith("HOST_CPU_PMU")
    assert summary["measurement"]["npu_clock_domain"] == "NPU cycle model"
    numbers = {value for block in ("ttft", "tpot", "overlap") for value in summary[block].values()
               if isinstance(value, (int, float))}
    for cpu_cycles, npu_cycles in ((900, 20), (300, 55 / 3), (1800, 75)):
        assert not [value for value in numbers if value == pytest.approx(cpu_cycles + npu_cycles)]
    assert "never CPU cycles + NPU cycles" in summary["measurement"]["e2e_definition"]
    assert summary["measurement"]["schedule_scope"] == "SYNTHETIC_RECONSTRUCTION"


def test_summary_cpu_means_are_null_when_a_decode_interval_lacks_valid_values() -> None:
    rows = [dict(row) for row in DECODE]
    rows[7].update(cpu_cycles_valid=False, cpu_cycles=None, host_thread_cpu_ns=None)  # token 2 CPU row
    summary, _ = summarized(rows, 4)
    tpot = summary["tpot"]
    # Then: an invalid PMU / thread-time interval is not averaged as zero.
    assert tpot["cpu_service_cycles_mean"] is None and tpot["cpu_thread_ms_mean"] is None
    assert tpot["cpu_service_cycles_mean_status"] == tpot["cpu_thread_ms_mean_status"] == \
        "UNAVAILABLE: 1 of 3 decode intervals lack a valid value"
    # And: host elapsed time, NPU service and the scheduled E2E are unaffected.
    assert tpot["cpu_host_elapsed_ms_mean"] == pytest.approx(45 / 3 / 1e6)
    assert tpot["scheduled_e2e_cycles_mean"] == pytest.approx(80 / 3) and summary["ttft"]["cpu_service_cycles"] == 900


def test_summary_ms_is_diagnostic_without_a_validated_clock_and_publication_stays_closed() -> None:
    # Test F: no clock selection -> the ms value is the schedule-clock conversion, labelled diagnostic.
    summary, document = summarized(DECODE, 4)
    assert document["e2e"]["ttft_ms"] is None and document["e2e"]["target"]["e2e_ms"] is None
    for block, cycles, ms_key in (("ttft", "scheduled_e2e_cycles", "scheduled_e2e_ms"),
                                  ("tpot", "scheduled_e2e_cycles_mean", "scheduled_e2e_ms_mean")):
        assert summary[block]["scheduled_e2e_ms_status"] == "DIAGNOSTIC_CONFIGURED_TEST_CLOCK"
        assert summary[block][ms_key] == pytest.approx(summary[block][cycles] * 1000 / HZ)
    assert summary["measurement"]["schedule_clock_hz"] == HZ and summary["measurement"]["publication_ready"] is False
    assert summary["measurement"]["cpu_collection"] == "FRESH_COLLECTION"
    # And: TARGET_CPU_TIMING_READY=NOT_READY is target admission; the host CPU measurement is still reported.
    assert summary["markers"]["TARGET_CPU_TIMING_READY"] == "NOT_READY" and summary["ttft"]["cpu_service_cycles"] == 900
    assert "target CPU timing admission is separate" in summary["measurement"]["cpu_timing_scope"]
    # Test G: a validated operating clock converts with that clock; the other gates still hold publication closed.
    clocked, document = summarized(DECODE, 4, validated_hz=250_000_000, reuse={"source_run": "old"},
                                   markers={**NOT_READY_MARKERS, "OPERATING_CLOCK_READY": "READY"})
    assert clocked["ttft"]["scheduled_e2e_ms_status"] == clocked["tpot"]["scheduled_e2e_ms_status"] == \
        "VALIDATED_OPERATING_CLOCK"
    assert clocked["ttft"]["scheduled_e2e_ms"] == document["e2e"]["ttft_ms"] == pytest.approx(30 * 1000 / 250_000_000)
    assert clocked["tpot"]["scheduled_e2e_ms_mean"] == document["e2e"]["mean_tpot_ms"]
    assert clocked["measurement"]["publication_ready"] is False
    assert clocked["measurement"]["cpu_collection"] == "REUSED_COLLECTION"
    # And: the marker alone or the gates alone never open publication.
    for markers, gates in (({**NOT_READY_MARKERS, "PUBLICATION_E2E_MS_READY": "READY"}, {"TARGET_LATENCY_READY": False}),
                           (dict(NOT_READY_MARKERS), {"TARGET_LATENCY_READY": True})):
        closed, _ = summarized(DECODE, 4, validated_hz=250_000_000, markers=markers, publication=gates)
        assert closed["measurement"]["publication_ready"] is False


# NPU service attribution: main / residual / other GEMM work (producer provenance only)

def test_npu_category_uses_only_the_producer_provenance_and_scope() -> None:
    from e2e_timeline import npu_category
    assert npu_category("dense_main", "stripe") == npu_category("dense_main", "full") == "MAIN_GEMM"
    assert npu_category("residual", "residual_compact") == "RESIDUAL_GEMM"
    # Missing or inconsistent provenance is UNCLASSIFIED, never main.
    for provenance, scope in ((None, None), ("dense_main", None), ("dense_main", "residual_compact"),
                              ("residual", "stripe"), ("rmd", "residual_compact"), ("", "")):
        assert npu_category(provenance, scope) == "UNCLASSIFIED"


def attribution() -> tuple[Record, Record]:
    """Token 0: main 100 + residual 40 + other 10; token 1: main 80 + residual 20; token 2: main 90."""
    from fractions import Fraction

    from e2e_timeline import PerformanceAccumulator
    accumulator = PerformanceAccumulator(3, None, "NANO_LOCAL_VALIDATED", HZ)
    accumulator.request_start(Fraction(0))
    at = 0
    for token, works in enumerate(((("MAIN_GEMM", 100, "L0"), ("RESIDUAL_GEMM", 40, "L0"), ("OTHER_NPU", 10, "L1")),
                                   (("MAIN_GEMM", 80, "L0"), ("RESIDUAL_GEMM", 20, "L0")),
                                   (("MAIN_GEMM", 90, "L0"),))):
        for category, cycles, layer in works:
            accumulator.npu_interval(token, Fraction(at), Fraction(at + cycles), cycles, category, layer)
            at += cycles
        accumulator.token_ready(token, Fraction(at))
    return accumulator.result()


def test_npu_service_splits_into_main_residual_and_other_exactly() -> None:
    result, checks = attribution()
    ttft = result["ttft"]["npu_service_breakdown"]
    assert checks["status"] == "PASS"
    assert (ttft["main_gemm_cycles"], ttft["residual_gemm_cycles"], ttft["other_npu_cycles"]) == (100, 40, 10)
    assert ttft["total_cycles"] == ttft["classified_total_cycles"] == result["ttft"]["npu_cycles"] == 150
    assert ttft["status"] == "COMPLETE" and ttft["unclassified_npu_cycles"] == 0
    assert ttft["residual_main_cycle_ratio"] == pytest.approx(0.4)
    per_token = result["tpot"]["npu_service_breakdown_per_token"]
    assert [(row["main_gemm_cycles"], row["residual_gemm_cycles"], row["other_npu_cycles"], row["total_cycles"])
            for row in per_token] == [(80, 20, 0, 100), (90, 0, 0, 90)]
    # A residual-free token has real zero residual work, not null.
    assert per_token[1]["residual_gemm_cycles"] == 0 and per_token[1]["residual_main_cycle_ratio"] == 0
    assert [row["total_cycles"] for row in per_token] == result["tpot"]["per_token_npu_cycles"]
    mean = result["tpot"]["npu_service_breakdown_mean"]
    assert (mean["main_gemm_cycles"], mean["residual_gemm_cycles"], mean["other_npu_cycles"],
            mean["total_cycles"]) == (85, 10, 0, 95)
    assert mean["total_cycles"] == result["tpot"]["npu_tpot_cycles"] and mean["intervals"] == 2
    assert mean["status"] == "COMPLETE" and mean["residual_main_cycle_ratio"] == pytest.approx(10 / 85)
    assert mean["main_gemm_cycles"] + mean["residual_gemm_cycles"] + mean["other_npu_cycles"] == mean["total_cycles"]
    # And: the existing totals and endpoints are the ones without attribution.
    assert result["e2e"]["ttft"] == 150 and result["e2e"]["tpot_intervals"] == [100, 90]


def test_npu_service_by_layer_aggregates_the_same_classified_works() -> None:
    result, _ = attribution()
    layers = result["npu_service_by_layer"]
    assert layers["ttft"]["L0"] == {"main_gemm_cycles": 100, "residual_gemm_cycles": 40, "other_npu_cycles": 0,
                                    "unclassified_npu_cycles": 0, "total_cycles": 140,
                                    "residual_main_cycle_ratio": pytest.approx(0.4)}
    assert layers["ttft"]["L1"]["other_npu_cycles"] == 10 and layers["ttft"]["L1"]["residual_main_cycle_ratio"] is None
    assert layers["tpot_decode_sum"] == {"L0": {"main_gemm_cycles": 170, "residual_gemm_cycles": 20,
                                                "other_npu_cycles": 0, "unclassified_npu_cycles": 0,
                                                "total_cycles": 190,
                                                "residual_main_cycle_ratio": pytest.approx(20 / 170)}}
    assert sum(row["total_cycles"] for row in layers["ttft"].values()) == result["ttft"]["npu_cycles"]


def test_unknown_npu_provenance_is_incomplete_and_never_main() -> None:
    rows = [event("request_start", 0), npu_row(0, 0, 60), npu_row(0, 60, 80, provenance=None, scope=None),
            event("token_ready", 80, 0), npu_row(1, 80, 100, provenance="residual", scope="stripe"),
            event("token_ready", 100, 1)]
    summary, document = summarized(rows, 2)
    ttft = document["ttft"]["npu_service_breakdown"]
    assert (ttft["main_gemm_cycles"], ttft["unclassified_npu_cycles"], ttft["classified_total_cycles"]) == (60, 20, 60)
    assert ttft["status"].startswith("INCOMPLETE") and ttft["total_cycles"] == document["ttft"]["npu_cycles"] == 80
    mean = document["tpot"]["npu_service_breakdown_mean"]
    assert mean["main_gemm_cycles"] == mean["residual_gemm_cycles"] == 0 and mean["unclassified_npu_cycles"] == 20
    assert mean["status"].startswith("INCOMPLETE")
    # And: the total NPU service and its legacy aliases stay available.
    assert summary["ttft_npu_cycles"] == 80 and summary["npu_tpot_cycles"] == 20
    assert summary["ttft"]["npu_service_breakdown"]["status"].startswith("INCOMPLETE")


def test_single_token_has_no_tpot_attribution() -> None:
    _, document = summarized([event("request_start", 0), npu_row(0, 0, 10), event("token_ready", 10, 0)], 1)
    assert document["tpot"]["npu_service_breakdown_per_token"] == []
    assert document["tpot"]["npu_service_breakdown_mean"] is None
    assert document["npu_service_by_layer"]["tpot_decode_sum"] == {}
