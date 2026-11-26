from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from campaign_build import PLATFORMS, Platform, cached, llama_plan, platform_profile, semantic_options, with_platform
from eval_common import EvaluationError, Record, record, sha256, text
from e2e_timeline import Axis, pack_lanes, merged, overlap
from fractions import Fraction
from certified_reconstruction import artifact_reference, bound_artifact, publication_row, reconstructed_row
from target_admission import admit_target_host, admit_target_interface
from cpu_profiling import instrumentation_markers, pmu_contract, timing_validity

KINDS = (("potal-host", "STRIPE_PIPELINE", ()), ("fullcpu-host", "FULL", ()))
PROBE_PASS = {"status": "PASS", "exclude_kernel": 0, "cap_user_rdpmc": 1, "index": 32, "direct_read_delta_cycles": 5}

def test_semantic_options_do_not_depend_on_platform() -> None:
    # Given every supported host and every build kind of the evaluation.
    for kind, matmul, extra in KINDS:
        plan = llama_plan(kind, "a8w8", 32, matmul, extra)
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
        with_platform(llama_plan("potal-host", "a8w8", 32, "STRIPE_PIPELINE").options, forged)


def test_extra_target_build_shares_potal_semantics_but_fullcpu_does_not() -> None:
    potal = llama_plan("potal-host", "a8w8", 32, "STRIPE_PIPELINE")
    extra = llama_plan("potal-host", "a8w8", 32, "STRIPE_PIPELINE", ("llama-perplexity",))
    fullcpu = llama_plan("fullcpu-host", "a8w8", 32)
    # An extra CMake target and the cycle-log instrumentation are not semantic options.
    assert potal.semantic_options_sha256 == extra.semantic_options_sha256
    assert fullcpu.semantic_options_sha256 != potal.semantic_options_sha256
    assert "llama-perplexity" in extra.targets


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


@pytest.mark.parametrize("paranoid, user_access, missing", [("2", "1", ["perf_event_paranoid=1"]),
                                                            ("1", "0", ["perf_user_access=1"]),
                                                            ("2", "0", ["perf_event_paranoid=1", "perf_user_access=1"])])
def test_pmu_preflight_fails_fast_with_the_settings_to_apply(paranoid: str, user_access: str, missing: list[str]) -> None:
    values = {"perf_event_paranoid": paranoid, "perf_user_access": user_access}
    with patch("cpu_profiling.platform_profile", return_value=platform_profile("Linux", "aarch64")), \
         patch.object(Path, "is_file", lambda self: self.name in values), \
         patch.object(Path, "read_text", lambda self, *args, **kwargs: values[self.name] + "\n"):
        with pytest.raises(EvaluationError) as failure:
            pmu_contract()
    assert all("sudo sysctl kernel." + setting in str(failure.value) for setting in missing)


def test_pmu_preflight_records_the_measurement_contract() -> None:
    values = {"perf_event_paranoid": "1", "perf_user_access": "1", "current_clocksource": "arch_sys_counter"}
    with patch("cpu_profiling.platform_profile", return_value=platform_profile("Linux", "aarch64")), \
         patch("cpu_profiling.pmu_probe", return_value=PROBE_PASS), \
         patch("cpu_profiling.cpu_model", return_value="Cortex-A78AE"), \
         patch.object(Path, "is_file", lambda self: self.name in values), \
         patch.object(Path, "read_text", lambda self, *args, **kwargs: values[self.name] + "\n"):
        contract = pmu_contract()
    assert contract["status"] == "READY" and contract["exclude_kernel"] == 0 and contract["cycle_scope"] == "user + kernel"
    assert contract["kernel.perf_event_paranoid"] == 1 and contract["kernel.perf_user_access"] == 1
    assert contract["probe"] == PROBE_PASS and contract["cpu_model"] == "Cortex-A78AE"
    assert contract["clocksource"] == "arch_sys_counter"
    with patch("cpu_profiling.platform_profile", return_value=platform_profile("Darwin", "arm64")):
        assert pmu_contract()["status"] == "NOT_APPLICABLE"


@pytest.mark.parametrize("probe", [{"status": "FAILED", "stage": "perf_event_open", "errno": 13},
                                   {"status": "FAILED", "stage": "direct_read_index", "index": 0}])
def test_pmu_preflight_fails_fast_when_the_collector_probe_fails(probe: dict[str, object]) -> None:
    values = {"perf_event_paranoid": "1", "perf_user_access": "1"}
    with patch("cpu_profiling.platform_profile", return_value=platform_profile("Linux", "aarch64")), \
         patch("cpu_profiling.pmu_probe", return_value=probe), \
         patch.object(Path, "is_file", lambda self: self.name in values), \
         patch.object(Path, "read_text", lambda self, *args, **kwargs: values[self.name] + "\n"):
        with pytest.raises(EvaluationError, match="collector-equivalent PMU probe"):
            pmu_contract()


def test_real_pmu_probe_on_this_host() -> None:
    from cpu_profiling import pmu_probe
    if platform_profile().machine != "aarch64":
        pytest.skip("aarch64 collector probe")
    result = pmu_probe()
    assert result["status"] in ("PASS", "FAILED") and len(str(result["source_sha256"])) == 64


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
    cores = record(summary["cores"])
    assert (cores["migrated"], cores["not_migrated"]) == (1, 2) and record(cores["start_end_matrix"])["1->4"] == 1
    assert cores["migration_ratio"] == pytest.approx(1 / 3)


def test_instrumentation_markers_fail_closed_on_invalid_cpu_timing(tmp_path: Path) -> None:
    good = timing_validity(collection(tmp_path / "a", [timed(start=0, end=0)]), True)
    bad = timing_validity(collection(tmp_path / "b", [timed(valid=False, start=0, end=0)]), True)
    unknown = timing_validity(collection(tmp_path / "c", [timed(start=None, end=0)]), True)
    result = {"ttft": {"npu_cycles": 5, "cpu_measured": {"cpu_service_cycles": 3, "cpu_thread_ns": 1}}, "tpot": {}}
    pmu: Record = {"status": "READY"}
    ready = instrumentation_markers({"pmu": pmu, "potal": good, "fullcpu": good}, result, 1)
    assert ready["CPU_STAGE_CYCLE_LOG_READY"] == ready["CPU_CORE_TIMELINE_READY"] == "READY"
    assert ready["TPOT_CPU_COMPONENT_READY"] == ready["TPOT_NPU_COMPONENT_READY"] == "NOT_APPLICABLE"
    invalid = instrumentation_markers({"pmu": pmu, "potal": good, "fullcpu": bad}, result, 1)
    assert invalid["CPU_STAGE_CYCLE_LOG_READY"] == invalid["TTFT_CPU_COMPONENT_READY"] == "NOT_READY"
    malformed = instrumentation_markers({"pmu": pmu, "potal": good, "fullcpu": unknown}, result, 1)
    assert malformed["CPU_CORE_TIMELINE_READY"] == "NOT_READY"
    missing = instrumentation_markers({"potal": good, "fullcpu": good}, result, 1)
    assert missing["NANO_HOST_PMU_READY"] == missing["CPU_STAGE_CYCLE_LOG_READY"] == "NOT_READY"


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
    def produce(entry: Path) -> None:
        (entry / "out.txt").write_text("ok")
    result = cache.run("stage", {"input": "a"}, ("out.txt",), tmp_path / "run2",
                       produce)
    assert result["cache_hit"] is False and (tmp_path / "run2/out.txt").read_text() == "ok"
    with pytest.raises(EvaluationError, match="did not produce"):
        cache.run("stage", {"input": "c"}, ("out.txt",), tmp_path / "run3", lambda entry: None)


def test_hash_memo_rehashes_when_the_file_changes(tmp_path: Path) -> None:
    from stage_cache import HashMemo
    path = tmp_path / "input.bin"
    path.write_bytes(b"one")
    memo = HashMemo(tmp_path / "memo.json")
    assert memo.sha256(path) == sha256(path)
    path.write_bytes(b"two!")
    assert HashMemo(tmp_path / "memo.json").sha256(path) == sha256(path)


def test_lane_packing_never_merges_concurrent_threads() -> None:
    lanes = pack_lanes({("x", 1): (0, 100), ("x", 2): (10, 20), ("x", 3): (21, 30), ("x", 4): (20, 25)})
    assert lanes[("x", 1)] != lanes[("x", 2)] and lanes[("x", 2)] != lanes[("x", 4)]
    assert lanes[("x", 3)] == lanes[("x", 2)]


# Publication (fail closed)

WORKLOAD: Record = {"model_sha256": "m", "input_tokens_sha256": "i", "generated_tokens_sha256": "g",
                    "profile": "a8w8-d32-hp1"}


def test_reconstruction_keeps_the_target_latency_deficiency_marker() -> None:
    row = reconstructed_row({"target_latency": "NOT_READY_REQUIRES_CERTIFIED_REPLAY_JOIN_CLOCK_AND_SERVICE_PROOF"},
                            {"ttft_ns": 1}, {"schema": "proof"})
    assert row["measurement_kind"] == "VALIDATED_RECONSTRUCTION"
    assert text(row, "target_latency").startswith("NOT_READY")


def test_native_timing_axis_and_interval_arithmetic() -> None:
    spans = [(Fraction(0), Fraction(3)), (Fraction(2), Fraction(5)), (Fraction(7), Fraction(9))]
    assert merged(spans) == [(Fraction(0), Fraction(5)), (Fraction(7), Fraction(9))]
    assert overlap(merged(spans), [(Fraction(1), Fraction(4))]) == 3
    row: Record = {}
    Axis(500_000_000, False).place(row, Fraction(2), Fraction(6))
    assert row["duration_cycles"] == 2 and row["duration_ns"] is None
    Axis(500_000_000, True).place(row, Fraction(2), Fraction(6))
    assert row["start_ns"] == 2 and row["end_ns"] == 6 and row["duration_ns"] == 4


def test_native_target_admission_and_artifact_binding(tmp_path: Path) -> None:
    # Measured target artifacts use real hash-bound evidence, independent of replay engines.
    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(b"target counters")
    reference = artifact_reference(evidence)
    assert bound_artifact(reference) == evidence
    workload: Record = {key: key for key in ("model_sha256", "input_tokens_sha256", "generated_tokens_sha256", "profile")}
    host: Record = {"schema": "potal-target-host-timing", "version": 1,
                    "target": {"host_id": "target", "system": "Linux", "machine": "aarch64", "board": "fpga", "cpu_model": "a53"},
                    "measurement": {"kind": "TARGET_MEASURED", "unit": "cycle", "counter": "PMCCNTR"},
                    "clock": {"cpu_frequency_hz": 1_200_000_000, "provenance": "board clock tree"},
                    "cpu_service_count": 3, "evidence": reference, "workload": workload}
    assert admit_target_host(host, workload, {"target"})["host_id"] == "target"
    with pytest.raises(EvaluationError, match="admitted target host"):
        admit_target_host(host, workload, {"development"})
    with pytest.raises(EvaluationError, match="workload mismatch"):
        admit_target_host(host, {**workload, "model_sha256": "other"}, {"target"})
    interface: Record = {"schema": "potal-target-interface-cost", "version": 1, "status": "MEASURED",
                         "stages": {"input": {"cycles": 10, "provenance": "counter"}},
                         "evidence": reference, "workload": workload}
    assert admit_target_interface(interface, workload, ["input"])["stages"] == ["input"]
    with pytest.raises(EvaluationError, match="uncovered stages"):
        admit_target_interface(interface, workload, ["input", "output"])
    with pytest.raises(EvaluationError, match="UNMODELED"):
        admit_target_interface({**interface, "status": "UNMODELED"}, workload, ["input"])
    with pytest.raises(EvaluationError, match="NOT_READY"):
        publication_row({}, {"TARGET_LATENCY_READY": False})
    assert publication_row({}, {"TARGET_LATENCY_READY": True})["target_latency"] == "TARGET_LATENCY_READY"
    evidence.write_bytes(b"changed")
    with pytest.raises(EvaluationError, match="artifact binding mismatch"):
        bound_artifact(reference)
