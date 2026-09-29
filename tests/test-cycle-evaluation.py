# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B -m pytest -q tests/test-cycle-evaluation.py
from __future__ import annotations

import json
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
    PLATFORMS,
    Platform,
    cached,
    llama_plan,
    platform_profile,
    semantic_options,
    with_platform,
)
from certified_reconstruction import publication_row, reconstructed_row
from e2e_timeline import Axis, pack_lanes, performance, timeline_rows, write_timeline
from eval_common import EvaluationError, Json, Record, sha256, write_json
from run_cycle_evaluation import ppl_config, quality_identity

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


def test_quality_build_shares_potal_semantics_but_fullcpu_does_not() -> None:
    potal = llama_plan("potal-host", "a8w8", 32, IM2P, "STRIPE_PIPELINE")
    quality = llama_plan("cycle", "a8w8", 32, IM2P, "STRIPE_PIPELINE", ("llama-perplexity",))
    fullcpu = llama_plan("fullcpu-host", "a8w8", 32, IM2P)
    assert potal.semantic_options_sha256 == quality.semantic_options_sha256
    assert fullcpu.semantic_options_sha256 != potal.semantic_options_sha256
    assert "llama-perplexity" in quality.targets


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

def fixture(root: Path, npu1_accept: int = 130) -> tuple[Path, Path, Path]:
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
                                "thread_cpu_valid": False}}]}
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
    ]
    bundle, schedule, results = root / "ir.sqlite", root / "schedule.sqlite", root / "npu.jsonl"
    with sqlite3.connect(bundle) as ir, sqlite3.connect(schedule) as out:
        ir.execute("CREATE TABLE nodes(identity TEXT PRIMARY KEY, kind TEXT, operation TEXT, body TEXT, pending INTEGER)")
        out.execute("CREATE TABLE results(identity TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, body TEXT)")
        for ordinal, (identity, kind, phase, body) in enumerate(nodes):
            ir.execute("INSERT INTO nodes VALUES(?,?,?,?,0)", (identity, kind, "op", json.dumps(
                {"node_id": identity, "kind": kind, "phase": phase, "operation_id": "op"})))
            out.execute("INSERT INTO results VALUES(?,?,?)", (identity, ordinal, json.dumps({"node_id": identity, **body})))
    results.write_text("".join(json.dumps({"sequence": index, "work_id": index, "layer": "blk.0", "operation": "MUL_MAT",
                                           "modeled": {"total_cycles": cycles}}) + "\n"
                               for index, cycles in ((0, 100), (1, 50))))
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


# Metrics

def test_quality_config_is_deterministic_and_bound_to_the_performance_inputs(tmp_path: Path) -> None:
    assert ppl_config(64) == ppl_config(64) and ppl_config(64) != ppl_config(0)
    model, dataset, other = tmp_path / "m.gguf", tmp_path / "d.raw", tmp_path / "o.raw"
    for path, text in ((model, "model"), (dataset, "text"), (other, "other")):
        path.write_text(text)
    expected_model, expected_dataset = {"sha256": sha256(model)}, {"sha256": sha256(dataset)}
    quality_identity(model, dataset, expected_model, expected_dataset)
    with pytest.raises(EvaluationError, match="quality model differs"):
        quality_identity(other, dataset, expected_model, expected_dataset)
    with pytest.raises(EvaluationError, match="quality dataset differs"):
        quality_identity(model, other, expected_model, expected_dataset)
