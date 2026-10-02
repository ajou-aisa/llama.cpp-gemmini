#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B scripts/eval/run_cycle_evaluation.py --model M.gguf --prompt-file wiki.test.raw \
#   --prompt-tokens 256 --generate 128 --precision a8w8 --dim 32 --certificates certificate-set.json --output DIR
"""Performance runs and timeline exports (one mode per invocation; `run_measurement.py` is the user-facing entry
point).

  --run performance  host detection, cached builds, PoTal/FullCPU collection (or a reused one), NPU replay,
                     join/lifecycle/IR, one SYNTHETIC scheduling pass with TTFT/TPOT and CPU/NPU components,
                     optional timeline. Never runs the activation/residual/SCU metrics.
  --run timeline     timeline export from a completed run's stored schedule; nothing is collected or scheduled.

Activation, residual and SCU metrics are a different domain: `campaign.py activation|residual|scu`.

Nothing here certifies anything: certified inputs (cycle library identity, base/run-aware/transition
certificates) are admitted by the unchanged official stages. NPU wall time exists only with a validated
operating clock; CPU target timing and interface cost stay NOT_READY/UNMODELED until evidence exists.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Final, TypeVar

from campaign_build import (
    REPO,
    cached_cycle_model,
    cached_llama_build,
    cycle_model_argv,
    llama_argv,
    llama_plan,
    platform_profile,
    source_state,
    toolchain,
)
from e2e_timeline import Axis, isolated_phase_table, timeline_rows, worker_scenario, write_timeline
from eval_common import (
    EvaluationError,
    Json,
    Record,
    clean_environment,
    compiled_info,
    integer,
    read_json,
    record,
    require,
    sha256,
    text,
    write_json,
)
from evaluation_host import host_facts
from model_manifest import model_entry
from offline_pipeline import DISK_RESERVE_BYTES
from schedule_engine import ScheduleSinks, run_schedule
from stage_cache import StageCache, source_digest
from target_admission import clock_frequency, publication_readiness

EVAL: Final = Path(__file__).resolve().parent
IM2P: Final = REPO.parent / "IM2P.sim"
CONFIGURED_TEST_CLOCK_HZ: Final = 1_000_000_000  # sim/cycle/interleaved_schedule_pins.py npu_frequency_hz
GENERATED: Final = 128
VALIDATION: Final = {"certified": "CURRENT_CERTIFIED", "nano-local": "NANO_LOCAL_VALIDATED"}
SETTINGS: Final[Record] = {"schema": "potal-evaluation-settings", "version": 2, "seed": 1234, "temperature": 0,
    "top_k": 0, "top_p": 1, "min_p": 0, "repeat_penalty": 1, "repeat_last_n": 0, "grammar": None,
    "eos_stopping": False, "warmup": 0, "sampler_policy": "user-confirmed-greedy-v3", "split": "test",
    "chunk_ids": list(range(10)), "threads": 1, "threads_batch": 1, "batch_size": 256, "ubatch_size": 256,
    "scope": "DEVELOPMENT_EVALUATION_NOT_PAPER_CAMPAIGN"}
# Stage source sets: a change to any of these files invalidates the cached outputs of the stages they implement.
IM2P_STAGE_SOURCES: Final = ("sim/cycle/*.py", "scripts/*.py", "config/**/*.json")
TIMELINE_STAGE_SOURCES: Final = ("e2e_timeline.py", "eval_common.py", "schedule_engine.py")
OFFLINE_STAGES: Final = ("worker-scenario", "replay", "join", "producer-lifecycle", "execution-ir", "schedule", "timeline")
# Code each offline stage runs and the stages whose outputs it reads. A stage identity carries the code digest of the
# stage and of every upstream stage, so a source change misses that stage and all of its downstream stages even when
# an upstream output is reproduced byte for byte.
STAGE_CODE: Final = {"worker-scenario": ("timeline_code",), "replay": ("im2p_code",), "join": ("im2p_code",),
                     "producer-lifecycle": ("im2p_code",), "execution-ir": ("im2p_code",),
                     "schedule": ("im2p_code", "timeline_code"), "timeline": ("timeline_code",)}
STAGE_UPSTREAM: Final = {"worker-scenario": (), "replay": (), "join": ("replay",),
                         "producer-lifecycle": ("worker-scenario", "replay", "join"),
                         "execution-ir": ("replay", "join", "producer-lifecycle"),
                         "schedule": ("replay", "execution-ir"), "timeline": ("replay", "execution-ir", "schedule")}
CPU_POLICY: Final = "HOST_ELAPSED_NS_GANG"
RAW_LARGE: Final = ("native/chunk-0/cycle-log.jsonl", "native/chunk-0/npu-cycle-trace.jsonl",
                    "native/chunk-0/semantic-graph.jsonl", "native/chunk-0/execution-lifecycle.jsonl")
T = TypeVar("T")


def reference(path: Path) -> Record:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def checksums(root: Path, known: dict[Path, str] | None = None) -> None:
    """SHA256SUMS over every file of a finished run directory; `known` holds digests that were already computed."""
    known = known or {}
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text("".join(f"{known.get(path) or sha256(path)}  {path.relative_to(root)}\n"
                                             for path in files))


def workload_mode(args: argparse.Namespace) -> Record:
    """SMOKE_256P1 is a separate connectivity workload; the 256+128 evaluation contract is unchanged."""
    if args.smoke:
        return {"workload_mode": "SMOKE_256P1", "prompt_tokens": 256, "generated_tokens": 1, "full_evaluation": False}
    return {"workload_mode": "E2E_GENERATION_256_128", "prompt_tokens": 256, "generated_tokens": GENERATED,
            "full_evaluation": True}


# Jetson/Linux aarch64 CPU intervals come from the per-thread PMU reader (ggml-gemmini-utils cycle_reader_aarch64):
# PERF_COUNT_HW_CPU_CYCLES with exclude_kernel=0 (user + kernel cycles), read directly from PMCCNTR in user space.
PMU_SYSCTLS: Final = {"perf_event_paranoid": ("<=", 1), "perf_user_access": ("==", 1)}


def pmu_contract() -> Record:
    """Measure-only PMU preflight; never changes a sysctl. Only Linux aarch64 collections depend on it."""
    host = platform_profile()
    if (host.system, host.machine) != ("Linux", "aarch64"):
        return {"status": "NOT_APPLICABLE", "platform": host.record()}
    values: Record = {}
    for name in PMU_SYSCTLS:
        path = Path("/proc/sys/kernel") / name
        values[name] = int(path.read_text().strip()) if path.is_file() else None
    failed = [name for name, (op, limit) in PMU_SYSCTLS.items()
              if not isinstance(values[name], int) or
              not (int(str(values[name])) <= limit if op == "<=" else int(str(values[name])) == limit)]
    require(not failed, "PMU preflight failed (" + ", ".join(f"{name}={values[name]}" for name in failed) +
            "): the Linux aarch64 collector needs per-thread user-space PMU access; set it outside the runner with: " +
            "; ".join(f"sudo sysctl kernel.{name}={limit}" for name, (_, limit) in PMU_SYSCTLS.items() if name in failed))
    probe = pmu_probe()
    require(probe.get("status") == "PASS", "PMU preflight failed: collector-equivalent PMU probe " + json.dumps(probe))
    return {"status": "READY", **{"kernel." + name: value for name, value in values.items()},
            "pmu_source": "linux_perf_cpu_cycles", "event": "PERF_COUNT_HW_CPU_CYCLES", "exclude_kernel": 0,
            "cycle_scope": "user + kernel", "read_path": "direct PMCCNTR user-space read (perf mmap page)",
            "probe": probe, "cpu_architecture": host.machine, "cpu_model": cpu_model(),
            "clocksource": read_text_or_none(Path("/sys/devices/system/clocksource/clocksource0/current_clocksource"))}


def read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cpu_model() -> str | None:
    done = subprocess.run(["lscpu"], capture_output=True, text=True, check=False)
    names = [line.split(":", 1)[1].strip() for line in done.stdout.splitlines() if line.startswith("Model name:")]
    return names[0] if names else None


def pmu_probe() -> Record:
    """Compile and run scripts/eval/pmu_probe.c: the collector's perf attributes and direct PMCCNTR read."""
    source = EVAL / "pmu_probe.c"
    with tempfile.TemporaryDirectory(prefix="pmu-probe-") as directory:
        binary = Path(directory) / "pmu_probe"
        built = subprocess.run(["cc", "-O1", "-o", str(binary), str(source)], capture_output=True, text=True,
                               check=False, timeout=120)
        if built.returncode != 0:
            return {"status": "FAILED", "stage": "compile", "error": built.stderr.strip()[-400:]}
        done = subprocess.run([str(binary)], capture_output=True, text=True, check=False, timeout=30)
    try:
        result = record(json.loads(done.stdout))
    except ValueError:
        return {"status": "FAILED", "stage": "output", "output": done.stdout[-400:]}
    return {**result, "source_sha256": sha256(source)}


def instrumentation_markers(timing: Record, result: Record, generated: int) -> Record:
    """Host PMU instrumentation readiness; never target timing admission. Missing evidence is NOT_READY."""
    def part(value: Json) -> Record:
        return record(value) if isinstance(value, dict) else {}

    def ready(value: bool) -> str:
        return "READY" if value else "NOT_READY"
    pmu_ready = part(timing.get("pmu")).get("status") == "READY"
    logs = [part(timing.get(name)) for name in ("potal", "fullcpu")]
    log_ready = pmu_ready and all(log.get("cycle_log_ready") is True for log in logs)
    core_ready = log_ready and all(part(log.get("cores")).get("with_core_fields") == log.get("applicable") and
                                   part(log.get("cores")).get("core_unknown") == 0 for log in logs)
    ttft = part(result.get("ttft"))
    measured = part(ttft.get("cpu_measured"))
    ttft_cpu = log_ready and measured.get("cpu_service_cycles") is not None and measured.get("cpu_thread_ns") is not None
    per_token = part(result.get("tpot")).get("cpu_measured_per_token")
    tpot = [part(row) for row in per_token] if isinstance(per_token, list) else []
    markers: Record = {
        "NANO_HOST_PMU_READY": ready(pmu_ready), "CPU_STAGE_CYCLE_LOG_READY": ready(log_ready),
        "CPU_CORE_TIMELINE_READY": ready(core_ready), "TTFT_CPU_COMPONENT_READY": ready(ttft_cpu),
        "TTFT_NPU_COMPONENT_READY": ready(ttft.get("npu_cycles") is not None),
        "TPOT_CPU_COMPONENT_READY": "NOT_APPLICABLE" if generated == 1 else
            ready(log_ready and bool(tpot) and all(row.get("cpu_service_cycles") is not None for row in tpot)),
        "TPOT_NPU_COMPONENT_READY": "NOT_APPLICABLE" if generated == 1 else
            ready(part(result.get("tpot")).get("npu_tpot_cycles") is not None)}
    return markers


def timing_validity(collection: Path, pmu_required: bool) -> Record:
    """CPU timing applicability/validity and observed-core migration of one collection (cycle log + application).

    Applicable = a CPU service interval carrying the named timing contract. Non-applicable = NPU dependency
    envelopes, functional emulation and structural/diagnostic rows (cpu_service false or no contract)."""
    counts: dict[str, int] = {"rows": 0, "applicable": 0, "valid": 0, "invalid": 0, "non_applicable": 0}
    reasons: dict[str, int] = {}
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    matrix: dict[str, int] = {}
    cores: dict[str, int] = {"with_core_fields": 0, "core_unknown": 0, "migrated": 0, "not_migrated": 0}
    sources = (collection / "native/chunk-0/cycle-log.jsonl", collection / "native/application-cpu.jsonl")
    for path in sources:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                counts["rows"] += 1
                if '"cpu_work_cycles_valid"' not in line:
                    counts["non_applicable"] += 1
                    continue
                row = record(json.loads(line))
                if row.get("schema") != "potal-application-cpu" and row.get("cpu_service") is not True:
                    counts["non_applicable"] += 1
                    continue
                counts["applicable"] += 1
                if row.get("cpu_work_cycles_valid") is True and row.get("cpu_work_cycles_source") == "linux_perf_cpu_cycles":
                    counts["valid"] += 1
                else:
                    counts["invalid"] += 1
                    reason = str(row.get("cpu_work_cycles_sample_reason") or row.get("cpu_work_cycles_reason"))
                    reasons[reason] = reasons.get(reason, 0) + 1
                if "host_cpu_core_start" not in row:
                    continue
                cores["with_core_fields"] += 1
                start, end = row.get("host_cpu_core_start"), row.get("host_cpu_core_end")
                if not isinstance(start, int) or not isinstance(end, int):
                    cores["core_unknown"] += 1
                    continue
                cores["migrated" if start != end else "not_migrated"] += 1
                starts[str(start)] = starts.get(str(start), 0) + 1
                ends[str(end)] = ends.get(str(end), 0) + 1
                matrix[f"{start}->{end}"] = matrix.get(f"{start}->{end}", 0) + 1
    known = cores["migrated"] + cores["not_migrated"]
    ready = (counts["applicable"] > 0 and counts["invalid"] == 0) if pmu_required else None
    distribution: Record = {"start_core_distribution": dict(sorted(starts.items())),
                            "end_core_distribution": dict(sorted(ends.items())),
                            "start_end_matrix": dict(sorted(matrix.items()))}
    core_summary: Record = {**cores, "migration_ratio": (cores["migrated"] / known) if known else None,
                            **distribution, "affinity": "NOT_ENFORCED (observation of the default Linux scheduler)"}
    summary: Record = {**counts, "invalid_reasons": {key: value for key, value in reasons.items()}, "cpu_timing_source": "linux_perf_cpu_cycles",
            "cpu_cycle_scope": "user+kernel", "cycle_log_ready": ready,
            "cores": core_summary}
    return summary


def call(argv: list[str], log: Path, cwd: Path, timeout: int) -> None:
    with log.open("x", encoding="utf-8") as stream:
        done = subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    require(done.returncode == 0, f"stage command failed ({done.returncode}); see {log}")


class Run:
    """Stage ledger: every stage is timed and recorded, and a failure leaves an explicit manifest."""

    def __init__(self, root: Path, header: Record) -> None:
        self.root, self.header, self.stages = root, header, list[Json]()
        self.markers: Record = {}
        self.cache: list[Json] = []

    def step(self, name: str, action: Callable[[], T]) -> T:
        started = time.monotonic()
        try:
            value = action()
        except Exception as error:
            self.stages.append({"stage": name, "status": "FAILED", "seconds": round(time.monotonic() - started, 3),
                                "reason": str(error)})
            self.write("FAILED")
            raise
        self.stages.append({"stage": name, "status": "PASS", "seconds": round(time.monotonic() - started, 3)})
        self.write("RUNNING")
        return value

    def write(self, status: str) -> None:
        manifest: Record = {**self.header, "status": status, "stages": self.stages, "markers": self.markers,
                            "stage_cache": self.cache}
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def gguf_identity(model: Path) -> Record:
    sys.path.insert(0, str(REPO / "gguf-py"))
    gguf: Any = importlib.import_module("gguf")
    fields: Any = gguf.GGUFReader(model).fields

    def value(key: str) -> Json:
        require(key in fields, "GGUF model lacks " + key)
        field: Any = fields[key]
        part: Any = field.parts[field.data[0]]
        return str(bytes(part).decode()) if field.types[0].name == "STRING" else int(part[0])
    file_type = integer({"file_type": value("general.file_type")}, "file_type")
    return {"sha256": sha256(model), "bytes": model.stat().st_size, "architecture": value("general.architecture"),
            "tokenizer_model": value("tokenizer.ggml.model"), "file_type": file_type,
            "quantization": str(gguf.LlamaFileType(file_type).name)}


def build_all(args: argparse.Namespace, run: Run, collect: bool = True) -> Record:
    """Performance builds only: the cycle model and, for a fresh collection, PoTal and FullCPU (metric flags off)."""
    cache = args.build_cache.resolve()
    builds: Record = {}
    (run.root / "build").mkdir()
    cycle, hit = run.step("build-cycle-model", lambda: cached_cycle_model(cache, IM2P, args.jobs)
                          if args.validation_mode == "certified" else admitted_cycle_model(args.local_validation))
    receipt = read_json(cycle / "build-receipt.json")
    library = record(receipt.get("library"))
    if args.validation_mode == "certified":
        require(library.get("sha256") == args.certificate_set["cycle_library_sha256"],
                "fresh cycle library differs from the certified CURRENT library; certified replay would reject it")
        builds["cycle-model"] = {"path": str(cycle), "cache_hit": hit, "receipt": receipt, "certified_identity_match": True}
    else:
        # The receipt must name these exact library bytes and the current cycle-model/authority sources.
        from sim.cycle.local_validation import admit_library
        _, validation = admit_library(args.local_validation, Path(text(library, "path")))
        require(validation == VALIDATION["nano-local"], "local validation receipt is not a PASS receipt")
        builds["cycle-model"] = {"path": str(cycle), "cache_hit": hit, "receipt": receipt,
                                 "certified_identity_match": False, "local_validation_match": True}
    kinds: list[tuple[str, str, str, tuple[str, ...]]] = ([("potal", "potal-host", "STRIPE_PIPELINE", ()),
                                                           ("fullcpu", "fullcpu-host", "FULL", ())] if collect else [])
    for name, kind, matmul, extra in kinds:
        path, hit = run.step("build-" + name, lambda kind=kind, matmul=matmul, extra=extra: cached_llama_build(
            cache, kind, args.precision, args.dim, IM2P, args.jobs, matmul, extra))
        builds[name] = {"path": str(path), "cache_hit": hit, "receipt": read_json(path / "build-receipt.json"),
                        "build_info": read_json(path / "build-info.json")}
    for name, value in builds.items():
        write_json(run.root / "build" / (name + ".json"), value)
    return builds


def admitted_cycle_model(receipt_path: Path) -> tuple[Path, bool]:
    """nano-local: the cycle-model build the local validation receipt admits. Admission re-checks the library bytes,
    its compiled source closure and the authority sources against the current tree, so an evaluation-code change
    neither rebuilds nor silently swaps the validated library; a changed C++ closure fails instead."""
    from sim.cycle.local_validation import admit_library, read_receipt
    bound = record(read_receipt(receipt_path).get("build_receipt"))
    build_receipt = Path(text(bound, "path"))
    require(sha256(build_receipt) == bound.get("sha256"), "cycle-model build receipt changed after local validation")
    library = Path(text(record(read_json(build_receipt).get("library")), "path"))
    admit_library(receipt_path, library)
    return build_receipt.parent, True


def collect(args: argparse.Namespace, run: Run, builds: Record, settings: Path) -> tuple[Path, Path]:
    raw = run.root / "raw"
    raw.mkdir()
    base = [sys.executable, "-B", str(EVAL / "end_to_end.py")]
    common = ["--im2p", str(IM2P), "--model", str(args.model), "--dataset", str(args.prompt_file),
              "--settings", str(settings), "--repetitions", "1", "--timeout", str(args.timeout)]
    potal, fullcpu = raw / "potal", raw / "fullcpu"
    runner = {name: str(Path(text(record(builds[name]), "path")) / "bin/llama-eval-workload")
              for name in ("potal", "fullcpu")}
    if args.smoke:
        # 256 prompt + 1 generated token PoTal/FullCPU pair (INTEGRATION_ONLY collector, metrics OFF).
        smoke = raw / "smoke"
        run.step("collect-smoke-pair", lambda: call(
            [sys.executable, "-B", str(EVAL / "run_stateful_integration_smoke.py"), "--potal-runner", runner["potal"],
             "--fullcpu-runner", runner["fullcpu"], "--model", str(args.model), "--dataset", str(args.prompt_file),
             "--im2p", str(IM2P), "--output", str(smoke), "--timeout", str(args.timeout)],
            raw / "collect-smoke.log", EVAL, 2 * args.timeout + 120))
        return smoke / "potal/collection", smoke / "fullcpu/collection"
    run.step("collect-potal", lambda: call([*base, "--output", str(potal), "collect", "--runner", runner["potal"],
                                           *common, "--role", "potal"], raw / "collect-potal.log", EVAL,
                                          args.timeout + 60))
    run.step("collect-fullcpu", lambda: call([*base, "--output", str(fullcpu), "collect", "--runner", runner["fullcpu"],
                                             *common, "--role", "fullcpu-cost-only", "--paired-potal", str(potal)],
                                            raw / "collect-fullcpu.log", EVAL, args.timeout + 60))
    return potal / "repetition-00", fullcpu / "repetition-00"


def reconstruct(args: argparse.Namespace, run: Run, potal: Path, fullcpu: Path, cycle_library: Path) -> Path:
    inputs = run.root / "replay-inputs"
    inputs.mkdir()
    mapping, scenario = run.step("worker-scenario", lambda: worker_scenario(potal, fullcpu, "HOST_ELAPSED_NS_GANG"))
    write_json(inputs / "worker-resources.json", {key: value for key, value in mapping.items()})
    write_json(inputs / "cpu-scenario.json", scenario)
    certificates = args.certificate_set
    replay = run.root / "replay"
    argv = [sys.executable, "-B", str(EVAL / "end_to_end.py"), "--output", str(replay), "reconstruct",
            "--im2p", str(IM2P),
            "--full-cpu-log", str(fullcpu / "native/chunk-0/cycle-log.jsonl"),
            "--full-cpu-graph", str(fullcpu / "native/chunk-0/semantic-graph.jsonl"),
            "--full-cpu-provenance", str(fullcpu / "collection-provenance.json"),
            "--potal-log", str(potal / "native/chunk-0/cycle-log.jsonl"),
            "--potal-graph", str(potal / "native/chunk-0/semantic-graph.jsonl"),
            "--potal-provenance", str(potal / "collection-provenance.json"),
            "--npu-trace", str(potal / "native/chunk-0/npu-cycle-trace.jsonl"),
            "--library", str(cycle_library),
            *(f"--{name.replace('_', '-')}={text(record(certificates[name]), 'path')}"
              for name in ("cycle_certificate", "run_aware_certificate", "transition_certificate")),
            *(["--potal-result", str(potal / "result.json")] if (potal / "result.json").is_file() else []),
            "--application", str(potal / "native/application-cpu.jsonl"),
            "--lifecycle-sidecar", str(potal / "native/chunk-0/execution-lifecycle.jsonl"),
            "--worker-resources", str(inputs / "worker-resources.json"),
            "--cpu-policy", "HOST_ELAPSED_NS_GANG", "--sampler-resource", text(scenario, "sampler_resource"),
            "--profile", args.profile, "--timeout", str(args.reconstruct_timeout),
            "--replay-workers", str(args.replay_workers), "--storage-factor", str(args.storage_factor)]
    run.step("reconstruct", lambda: call(argv, run.root / "reconstruct.log", EVAL, args.reconstruct_timeout + 60))
    result = read_json(replay / "result.json")
    require(result.get("replay") == "PASS" and result.get("three_source_join") == "PASS" and
            result.get("execution_ir") == "PASS" and result.get("replay_mode") == "FAST_EVALUATION_PARALLEL",
            "official offline stages did not all pass")
    return replay


class StopAfter(Exception):
    """--stop-after reached: the run ends after this stage with its outputs published (not a failure)."""


def stage_code(name: str) -> Record:
    """Code digests of a stage and of every stage upstream of it."""
    keys: set[str] = set()
    pending = [name]
    while pending:
        stage = pending.pop()
        keys.update(STAGE_CODE[stage])
        pending.extend(STAGE_UPSTREAM[stage])
    sources = {"im2p_code": (IM2P, IM2P_STAGE_SOURCES), "timeline_code": (EVAL, TIMELINE_STAGE_SOURCES)}
    return {key: source_digest(*sources[key]) for key in sorted(keys)}


def cached_stage(args: argparse.Namespace, run: Run, cache: StageCache, name: str, identity: Record,
                 outputs: tuple[str, ...], destination: Path, produce: Callable[[Path], None]) -> Record:
    code = stage_code(name)
    require(all(identity.get(key, digest) == digest for key, digest in code.items()),
            f"stage {name} identity carries a stale code digest")
    identity = {**identity, **code}
    info = run.step(name, lambda: cache.run(name, identity, outputs, destination, produce))
    run.cache.append(info)
    run.write("RUNNING")
    if args.stop_after == name:
        raise StopAfter(name)
    return info


def output_sha(info: Record, name: str) -> Json:
    return record(record(info.get("outputs")).get(name)).get("sha256")


def reconstruct_local(args: argparse.Namespace, run: Run, potal: Path, fullcpu: Path, cycle_library: Path,
                      cache: StageCache) -> Path:
    """NANO_LOCAL_VALIDATED replay, join, lifecycle and streaming IR through the shared IM2P cores.

    The official offline pipeline admits only certificates, so this path calls the IM2P evaluation-only
    entrypoints (sim.cycle.local_replay, sim.cycle.nano_local_execution); nothing here is certified. Every stage is
    content-addressed in the stage cache: identical inputs, parameters and stage sources reuse the published outputs."""
    inputs, replay = run.root / "replay-inputs", run.root / "replay"
    p, f = potal / "native/chunk-0", fullcpu / "native/chunk-0"
    files = {"fullcpu_log": f / "cycle-log.jsonl", "fullcpu_graph": f / "semantic-graph.jsonl",
             "fullcpu_provenance": fullcpu / "collection-provenance.json", "potal_log": p / "cycle-log.jsonl",
             "potal_graph": p / "semantic-graph.jsonl", "potal_provenance": potal / "collection-provenance.json",
             "trace": p / "npu-cycle-trace.jsonl", "sidecar": p / "execution-lifecycle.jsonl",
             "application": potal / "native/application-cpu.jsonl"}
    hashed: Record = run.step("input-identity", lambda: {name: cache.input(path) for name, path in files.items()})
    im2p_code, timeline_code = source_digest(IM2P, IM2P_STAGE_SOURCES), source_digest(EVAL, TIMELINE_STAGE_SOURCES)
    authority: Record = {"library": cache.input(cycle_library), "receipt": cache.input(args.local_validation),
                         "validation_mode": VALIDATION["nano-local"], "im2p_code": im2p_code}
    argv_authority = ["--library", str(cycle_library), "--local-validation", str(args.local_validation)]

    def scenario_stage(entry: Path) -> None:
        mapping, scenario = worker_scenario(potal, fullcpu, CPU_POLICY)
        write_json(entry / "worker-resources.json", {key: value for key, value in mapping.items()})
        write_json(entry / "cpu-scenario.json", scenario)
    scenario_info = cached_stage(args, run, cache, "worker-scenario",
                                 {"potal_log": hashed["potal_log"], "fullcpu_log": hashed["fullcpu_log"],
                                  "application": hashed["application"], "policy": CPU_POLICY,
                                  "timeline_code": timeline_code},
                                 ("worker-resources.json", "cpu-scenario.json"), inputs, scenario_stage)
    scenario = read_json(inputs / "cpu-scenario.json")
    replay.mkdir(exist_ok=True)
    workload_bytes = sum(path.stat().st_size for path in files.values())
    required, available = DISK_RESERVE_BYTES + args.storage_factor * workload_bytes, shutil.disk_usage(replay).free
    if available < required:
        write_json(replay / "failure.json", {"schema": "potal-offline-failure", "version": 1, "boundary": "preflight",
                                             "status": "REJECTED", "workload_bytes": workload_bytes,
                                             "required_free_bytes": required, "available_free_bytes": available})
        require(False, f"offline storage requires {required} free bytes; available {available}")
    npu, dataset, join = replay / "npu-cycle-result.jsonl", replay / "dataset.jsonl.gz", replay / "join-summary.json"
    lifecycle = replay / "execution-lifecycle.json"

    def command(entry: Path, name: str, argv: list[str]) -> None:
        call([sys.executable, "-B", "-m", *argv], entry / (name + ".log"), IM2P, args.reconstruct_timeout)

    tasks = npu_task_cache(args, cache)
    replay_info = cached_stage(
        args, run, cache, "replay", {"trace": hashed["trace"], **authority},
        ("npu-cycle-result.jsonl", "npu-summary.json", "replay.log", "replay-stats.json"), replay,
        lambda entry: command(entry, "replay", [
            "sim.cycle.local_replay", str(files["trace"]), *argv_authority,
            "--output", str(entry / "npu-cycle-result.jsonl"), "--summary", str(entry / "npu-summary.json"),
            "--workers", str(args.replay_workers), "--stats", str(entry / "replay-stats.json"),
            *(["--task-cache", str(tasks)] if tasks is not None else [])]))
    join_info = cached_stage(
        args, run, cache, "join",
        {**{name: hashed[name] for name in ("fullcpu_log", "fullcpu_graph", "fullcpu_provenance", "potal_log",
                                             "potal_graph", "potal_provenance", "trace")},
         "npu_results": output_sha(replay_info, "npu-cycle-result.jsonl"), **authority},
        ("dataset.jsonl.gz", "join-summary.json", "join.log"), replay,
        lambda entry: command(entry, "join", [
            "sim.cycle.nano_local_execution", "join", *argv_authority,
            "--full-cpu-log", str(files["fullcpu_log"]), "--full-cpu-graph", str(files["fullcpu_graph"]),
            "--full-cpu-provenance", str(files["fullcpu_provenance"]),
            "--potal-log", str(files["potal_log"]), "--potal-graph", str(files["potal_graph"]),
            "--potal-provenance", str(files["potal_provenance"]), "--npu-trace", str(files["trace"]),
            "--npu-results", str(npu), "--output", str(entry / "dataset.jsonl.gz"),
            "--summary", str(entry / "join-summary.json")]))
    lifecycle_info = cached_stage(
        args, run, cache, "producer-lifecycle",
        {**{name: hashed[name] for name in ("sidecar", "potal_graph", "potal_provenance", "application")},
         "dataset": output_sha(join_info, "dataset.jsonl.gz"), "join_summary": output_sha(join_info, "join-summary.json"),
         "npu_results": output_sha(replay_info, "npu-cycle-result.jsonl"),
         "worker_resources": output_sha(scenario_info, "worker-resources.json"), "cpu_policy": CPU_POLICY,
         "sampler_resource": text(scenario, "sampler_resource"), **authority},
        ("execution-lifecycle.json", "execution-lifecycle.json.nano-local.json", "producer-lifecycle.log"), replay,
        lambda entry: command(entry, "producer-lifecycle", [
            "sim.cycle.nano_local_execution", "lifecycle", *argv_authority,
            "--sidecar", str(files["sidecar"]), "--semantic-graph", str(files["potal_graph"]),
            "--provenance", str(files["potal_provenance"]), "--application", str(files["application"]),
            "--dataset", str(dataset), "--npu-results", str(npu), "--join-summary", str(join),
            "--worker-resources", str(inputs / "worker-resources.json"), "--cpu-policy", CPU_POLICY,
            "--sampler-resource", text(scenario, "sampler_resource"),
            "--output", str(entry / "execution-lifecycle.json")]))
    cached_stage(
        args, run, cache, "execution-ir",
        {"dataset": output_sha(join_info, "dataset.jsonl.gz"), "join_summary": output_sha(join_info, "join-summary.json"),
         "lifecycle": output_sha(lifecycle_info, "execution-lifecycle.json"),
         "npu_results": output_sha(replay_info, "npu-cycle-result.jsonl"), "application": hashed["application"],
         **authority},
        ("execution.sqlite", "execution.sqlite.nano-local.json", "execution-ir.log"), replay,
        lambda entry: command(entry, "execution-ir", [
            "sim.cycle.nano_local_execution", "adapt", *argv_authority, "--dataset", str(dataset),
            "--lifecycle", str(lifecycle), "--npu-results", str(npu), "--join-summary", str(join),
            "--application", str(files["application"]), "--output", str(entry / "execution.sqlite")]))
    summary = read_json(join)
    require(summary.get("validation_mode") == VALIDATION["nano-local"] and summary.get("publication_certified") is False,
            "nano-local join lacks its validation identity")
    write_json(replay / "result.json", {
        "schema": "potal-offline-evaluation-nano-local", "version": 1, "replay": "PASS", "three_source_join": "PASS",
        "execution_ir": "PASS", "replay_mode": "FAST_EVALUATION_PARALLEL", "execution_ir_format": "SQLITE",
        "validation_mode": VALIDATION["nano-local"], "publication_certified": False,
        "validation_receipt_sha256": summary.get("validation_receipt_sha256"),
        "E2E_RECONSTRUCTION_READY": False, "TARGET_LATENCY_READY": False,
        "storage_preflight": {"workload_bytes": workload_bytes, "required_free_bytes": required,
                              "available_free_bytes": available},
        "scope": "NANO_LOCAL evaluation-only offline dataset and synthetic schedule inputs; never publication"})
    return replay


def collection_identity(args: argparse.Namespace, potal: Path) -> Record:
    """Workload token identity and host of the PoTal collection (E2E result or smoke pair summary)."""
    if not args.smoke:
        result = read_json(potal / "result.json")
        return {"input_tokens_sha256": result.get("input_tokens_sha256"),
                "generated_tokens_sha256": result.get("generated_tokens_sha256"), "host_id": text(result, "host_id")}
    summary = read_json(potal.parent.parent / "collection-summary.json")
    pairing = record(summary.get("pairing"))
    chunks = read_json(potal / "native/workload.json").get("chunks")
    require(isinstance(chunks, list) and len(chunks) == 1, "smoke workload must map one prompt chunk")
    prompt = record(chunks[0] if isinstance(chunks, list) else None).get("input_tokens")
    require(isinstance(prompt, list) and len(prompt) == 256, "smoke prompt must contain 256 native token IDs")
    return {"input_tokens_sha256": sha256_text(json.dumps(prompt, separators=(",", ":"))),
            "generated_tokens_sha256": pairing.get("generated_token_sha256"), "host_id": text(pairing, "host_id")}


def discard(run: Run, paths: list[Path], ledger: list[Json]) -> None:
    for path in paths:
        if path.is_file():
            ledger.append({**reference(path), "status": "REMOVED_AFTER_USE"})
            path.unlink()


def schedule_identity(cache: StageCache, bundle: Path, npu: Path, axis: Axis, generated: int, validation: str) -> Record:
    """Schedule-stage identity: IR, NPU results, clock axis and sink parameters (cached_stage adds the code digests).

    The clock is part of it, so a new operating clock reschedules (overlap and token-ready times can move); an earlier
    schedule is never rescaled into milliseconds."""
    return {"bundle": cache.input(bundle), "npu_results": cache.input(npu), "frequency_hz": axis.frequency_hz,
            "clock_validated": axis.validated, "synthetic": True, "generated": generated, "validation": validation}


def schedule(args: argparse.Namespace, run: Run, replay: Path, axis: Axis, cache: StageCache,
             timeline: Path | None) -> Path:
    """SYNTHETIC schedule of the execution IR in one pass: the stored schedule, the performance accumulator and, when
    `timeline` is given, the timeline rows of that same pass (schedule_engine)."""
    bundle, npu = replay / "execution.sqlite", replay / "npu-cycle-result.jsonl"
    validation = VALIDATION[args.validation_mode]

    def produce(entry: Path) -> None:
        table = entry / "isolated-phase-table.json"
        write_json(table, isolated_phase_table(npu))
        sinks = ScheduleSinks(npu, axis, args.generated, validation, timeline)
        outcome = run_schedule(IM2P, bundle, table, axis.frequency_hz, entry / "schedule.sqlite", sinks)
        result, checks = sinks.accumulator.result()
        write_json(entry / "schedule-performance.json", {"schema": "potal-schedule-performance", "version": 1,
                                                         "result": result, "checks": checks,
                                                         "nodes": outcome["nodes"], "rows": outcome["rows"]})
        (entry / "schedule.log").write_text(json.dumps({**record(outcome["summary"]), "seconds": outcome["seconds"],
                                                        "counters": outcome["counters"]},
                                                       sort_keys=True) + "\n")
    cached_stage(args, run, cache, "schedule", schedule_identity(cache, bundle, npu, axis, args.generated, validation),
                 ("schedule.sqlite", "isolated-phase-table.json", "schedule-performance.json", "schedule.log"),
                 replay, produce)
    return replay / "schedule.sqlite"


def timeline_stage(args: argparse.Namespace, run: Run, cache: StageCache, replay: Path, schedule_path: Path, axis: Axis,
                   pending: Path | None, destination: Path, rows: int) -> int:
    """Timeline of the stored schedule: the rows the scheduling pass just wrote, or (after a schedule-stage cache hit)
    an export from the stored schedule. Both come from `e2e_timeline.node_rows`; nothing is scheduled again."""
    bundle, npu = replay / "execution.sqlite", replay / "npu-cycle-result.jsonl"

    def produce(entry: Path) -> None:
        target = entry / "timeline.jsonl"
        if pending is not None and pending.is_file():
            os.replace(pending, target)
            count = rows
        else:
            count = write_timeline(target, timeline_rows(schedule_path, bundle, npu, axis))
        write_json(entry / "timeline-rows.json", {"rows": count})
    cached_stage(args, run, cache, "timeline",
                 {"schedule": cache.input(schedule_path), "bundle": cache.input(bundle), "npu_results": cache.input(npu),
                  "axis": {"frequency_hz": axis.frequency_hz, "validated": axis.validated}},
                 ("timeline.jsonl", "timeline-rows.json"), destination, produce)
    if pending is not None and pending.exists():
        pending.unlink()  # a timeline-stage cache hit already holds these exact rows
    return integer(read_json(destination / "timeline-rows.json"), "rows")


def timeline_summary(count: int, timeline: Path, clock: Record, checks: Record, consistency: Record | None,
                     workload: Record, validation: Record, mode: str) -> Record:
    return {
        "schema": "im2p-e2e-timeline-summary", "version": 1, "rows": count, "timeline": reference(timeline),
        "axis": {"unit": "npu_cycle", "clock": clock,
                 "ns_fields": "non-null only under a validated operating clock",
                 "values": "integers when exact, else doubles; exact rationals are in the schedule"},
        "scope": f"SYNTHETIC_ONLY schedule: isolated {mode} NPU service, "
                 "development-host CPU durations, interface UNMODELED; a model, never an observation",
        "workload": workload, "validation": validation,
        "lanes": {"cpu": "resource/scheduler_lane is a SYNTHETIC scheduler lane, not a CPU core; "
                         "host_thread_id is the observed host thread; host_cpu_core_start/end are the observed "
                         "Linux cores at the interval endpoints (host_cpu_core only when not migrated); "
                         "target_cpu_core is null (no target mapping)",
                  "npu": "npu:0"},
        "checks": checks, "replay_consistency": consistency,
        "row_fields": ["seq", "row_type", "kind", "source", "resource", "resource_class", "scheduler_lane",
                       "tid", "host_thread_id", "host_cpu_core", "host_cpu_core_start", "host_cpu_core_end",
                       "cpu_migrated", "cpu_cycles", "cpu_cycles_valid", "cpu_cycle_source", "cpu_cycle_scope",
                       "target_cpu_core", "worker_id", "node_id",
                       "work_id", "op", "layer", "phase", "decode_index", "token_index", "start_cycle",
                       "end_cycle", "duration_cycles", "start_ns", "end_ns", "duration_ns", "npu_cycles",
                       "npu_ms", "host_elapsed_ns", "host_thread_cpu_ns", "target_cpu_cycles", "target_cpu_ms",
                       "evidence_id", "event", "timing_source", "source_line"]}


def stored_schedule(source: Path) -> tuple[Record, Record, Axis]:
    """The stored schedule, IR and NPU results of a completed performance run, each checked against its receipt."""
    manifest = read_json(source / "manifest.json")
    require(manifest.get("status") == "PASS", "source run did not complete: " + str(manifest.get("status")))
    rows = manifest.get("stage_cache")
    entries = {text(record(row), "stage"): record(row) for row in (rows if isinstance(rows, list) else [])
               if isinstance(row, dict)}
    inputs: Record = {}
    for name, stage, output in (("schedule", "schedule", "schedule.sqlite"), ("bundle", "execution-ir", "execution.sqlite"),
                                ("npu_results", "replay", "npu-cycle-result.jsonl")):
        require(stage in entries, f"source run has no stored {stage} stage output; a timeline export needs the "
                                  "preserved schedule, IR and NPU results (run --run performance first)")
        entry = entries[stage]
        expected = record(record(entry.get("outputs")).get(output)).get("sha256")
        candidates = (source / "replay" / output, Path(text(entry, "entry")) / output)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        require(path is not None, f"stored {output} of the source run is gone (stage cache {entry.get('entry')})")
        assert path is not None
        require(sha256(path) == expected, f"stored {output} differs from its stage receipt")
        inputs[name] = {"path": str(path), "sha256": expected, "stage_key": entry.get("key")}
    clock = record(read_json(source / "performance.json").get("clock"))
    axis = Axis(integer(clock, "schedule_clock_hz"), clock.get("status") == "VALIDATED_OPERATING_CLOCK")
    return manifest, inputs, axis


def export_timeline(args: argparse.Namespace) -> int:
    """--run timeline: export the timeline of a completed run's stored schedule. It never collects, replays or
    schedules; a run without the preserved schedule fails instead of recomputing it."""
    source_run: Path | None = args.from_run
    output: Path | None = args.output
    require(source_run is not None and output is not None, "--run timeline requires --from-run and --output")
    require(args.format == "jsonl", "only the jsonl timeline format is implemented")
    assert source_run is not None and output is not None
    source = source_run.resolve(strict=True)
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Run(root, {"schema": "im2p-cycle-evaluation-run", "version": 1, "started_utc": stamp, "argv": list(sys.argv),
                     "mode": "timeline-export", "source_run": str(source)})
    run.write("RUNNING")
    try:
        manifest, inputs, axis = run.step("stored-schedule", lambda: stored_schedule(source))
        timeline = root / "timeline.jsonl"
        count = run.step("timeline-export", lambda: write_timeline(timeline, timeline_rows(
            Path(text(record(inputs["schedule"]), "path")), Path(text(record(inputs["bundle"]), "path")),
            Path(text(record(inputs["npu_results"]), "path")), axis)))
        write_json(root / "timeline-rows.json", {"rows": count})
        exported = reference(timeline)
        write_json(root / "export.json", {
            "schema": "potal-timeline-export", "version": 1, "format": "jsonl", "rows": count,
            "timeline": exported, "source_run": str(source), "source_manifest": reference(source / "manifest.json"),
            "source_workload": manifest.get("workload"), "source_validation": manifest.get("validation"),
            "inputs": inputs, "axis": {"frequency_hz": axis.frequency_hz, "validated": axis.validated},
            "code": stage_code("timeline"),
            "not_run": ["collection", "NPU replay", "join/lifecycle/IR", "scheduling", "performance",
                        "metrics (activation/residual/SCU)"],
            "reuse": "TIMELINE_EXPORT_FROM_STORED_SCHEDULE"})
        run.markers.update({"MEASUREMENT_DOMAIN": "timeline", "TIMELINE_EXPORT_ONLY": "PASS", **NOT_RUN_DOMAINS,
                            "PERFORMANCE_RUN": "NOT_RUN"})
        run.write("PASS")
        checksums(root, {timeline: text(exported, "sha256")})
        print(json.dumps({"output": str(root), "rows": count, "source_run": str(source)}, sort_keys=True))
        return 0
    except (EvaluationError, OSError, ValueError) as error:
        if not run.stages or record(run.stages[-1]).get("status") != "FAILED":
            run.stages.append({"stage": "runner", "status": "FAILED", "reason": str(error)})
        run.write("FAILED")
        print(f"timeline export failed: {error}", file=sys.stderr)
        return 1


def npu_task_cache(args: argparse.Namespace, cache: StageCache) -> Path | None:
    """Per-work NPU answer cache: its own identity contract (sim.cycle.npu_task_cache), separate from the stage cache
    and from any golden/evidence directory. `none` disables it."""
    if args.npu_task_cache == "none":
        return None
    return Path(args.npu_task_cache).resolve() if args.npu_task_cache else cache.root / "npu-tasks.sqlite"


def execution_record(args: argparse.Namespace, reuse: Record | None, replay: Path, stages: list[Json]) -> Record:
    """What this run measured and what it reused (fresh CPU collection vs reused, NPU answers, stage outputs)."""
    stats = read_json(replay / "replay-stats.json") if (replay / "replay-stats.json").is_file() else {}
    hits = {text(record(row), "stage"): record(row).get("cache_hit") for row in stages if isinstance(row, dict)}
    return {"mode": "performance", "timeline": args.timeline,
            "cpu_measurement": "REUSED_COLLECTION" if reuse is not None else "FRESH_COLLECTION",
            "collection_source": reuse.get("source_run") if reuse is not None else "this run",
            "npu_timing": {"replay_stage_cache_hit": hits.get("replay"),
                           "task_cache": stats.get("npu_task_cache"), "model_calls": stats.get("model_calls"),
                           "binding_misses": stats.get("binding_misses"),
                           "note": "counters of the replay that produced the replay-stage outputs"},
            "stage_cache_hits": hits,
            "timeline_export": "WRITTEN_FROM_THIS_SCHEDULE" if args.timeline == "compact"
                               else "NOT_WRITTEN (export later with --run timeline --from-run)"}


def reused_collection(args: argparse.Namespace) -> tuple[Path, Path, Record]:
    """Collections of an earlier run (never re-executed): same workload, profile and completed collect stages."""
    source = Path(args.reuse_collection).resolve(strict=True)
    provenance, manifest = read_json(source / "provenance.json"), read_json(source / "manifest.json")
    potal, fullcpu = ((source / "raw/smoke/potal/collection", source / "raw/smoke/fullcpu/collection") if args.smoke
                      else (source / "raw/potal/repetition-00", source / "raw/fullcpu/repetition-00"))
    require(potal.is_dir() and fullcpu.is_dir(), "reused run lacks the PoTal/FullCPU collections: " + str(source))
    require(record(provenance.get("workload")) == workload_mode(args), "reused collection workload differs")
    require(manifest.get("profile") == args.profile, "reused collection profile differs")
    rows = manifest.get("stages")
    stages = {text(record(row), "stage"): record(row).get("status")
              for row in (rows if isinstance(rows, list) else []) if isinstance(row, dict)}
    wanted: list[Json] = ["collect-smoke-pair"] if args.smoke else ["collect-potal", "collect-fullcpu"]
    require(all(stages.get(str(name)) == "PASS" for name in wanted), "reused collection stages did not all pass")
    potal_build = record(read_json(source / "build/potal.json").get("build_info"))
    require(potal_build.get("hp1") is True and potal_build.get("backend") == "IM2P_SIM",
            "reused PoTal collection was not produced by an IM2P_SIM HP1 build")
    lineage: Record = {"source_run": str(source), "provenance": reference(source / "provenance.json"),
                       "source_manifest_status": manifest.get("status"), "collect_stages": wanted,
                       "model_sha256": record(provenance.get("model")).get("sha256"),
                       "sources_at_collection": provenance.get("sources"),
                       "potal_build": record(read_json(source / "build/potal.json")).get("path"),
                       "fullcpu_build": record(read_json(source / "build/fullcpu.json")).get("path")}
    return potal, fullcpu, {**lineage, "pmu_contract": provenance.get("pmu_contract"), "host": provenance.get("host")}


def sha256_text(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def cross_check(result: Record, summary: Record, generated: int = GENERATED) -> Record:
    per_token = record(result.get("tpot")).get("per_token_npu_cycles")
    expected = [integer(record(summary.get("per_decode_token_cycle_sums")), str(index)) for index in range(generated - 1)]
    checks: Record = {"ttft_npu_equals_replay_prefill_sum":
                      record(result.get("ttft")).get("npu_cycles") == integer(summary, "prefill_cycle_sum"),
                      "per_token_npu_equals_replay_decode_sums": per_token == expected}
    require(all(value is True for value in checks.values()), "timeline NPU cycles differ from the replay summary")
    return checks


RUN_MODES: Final = ("performance", "timeline")
# The other measurement domain, which a performance or timeline run never executes (activation/residual/SCU metrics
# live in campaign.py).
NOT_RUN_DOMAINS: Final = {"HARDWARE_METRICS_RUN": "NOT_RUN"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, help="GGUF model (performance)")
    parser.add_argument("--prompt-file", type=Path, help="WikiText-2 raw test text (native recipe)")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--generate", type=int, default=GENERATED)
    parser.add_argument("--precision", choices=("a4w4", "a8w8"), default="a8w8")
    parser.add_argument("--dim", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--certificates", type=Path, default=os.environ.get("IM2P_CERTIFICATE_SET"),
                        help="certificate set JSON (or IM2P_CERTIFICATE_SET); --validation-mode certified")
    parser.add_argument("--validation-mode", choices=tuple(VALIDATION), default="certified",
                        help="certified: reviewed certificate chain (CURRENT_CERTIFIED); nano-local: host-local "
                             "validation receipt (NANO_LOCAL_VALIDATED, never publication)")
    parser.add_argument("--local-validation", type=Path, help="sim.cycle.local_validation v2 receipt (nano-local)")
    parser.add_argument("--model-manifest", type=Path, help="model_manifest.py manifest; required for nano-local")
    parser.add_argument("--smoke", action="store_true",
                        help="SMOKE_256P1 connectivity workload (256 prompt + 1 token); never a full evaluation")
    parser.add_argument("--output", type=Path, help="fresh run directory; default runs/<utc>-<model>-<config>")
    parser.add_argument("--run", default="performance", choices=RUN_MODES,
                        help="performance: TTFT/TPOT and CPU/NPU components (default); timeline: export a timeline "
                             "from a completed run's stored schedule")
    parser.add_argument("--timeline", choices=("none", "compact"), default="none",
                        help="performance: also write timeline.jsonl from the same scheduling pass (compact)")
    parser.add_argument("--from-run", type=Path, help="timeline: completed performance run whose schedule is exported")
    parser.add_argument("--format", choices=("jsonl",), default="jsonl", help="timeline export format")
    parser.add_argument("--npu-task-cache", help="persistent per-work NPU answer cache (default <stage-cache>/"
                                                 "npu-tasks.sqlite; 'none' disables)")
    parser.add_argument("--clock-selection", type=Path, help="validated operating clock; enables NPU ms")
    parser.add_argument("--target-host-timing", type=Path, help="admitted target-host timing (publication gate)")
    parser.add_argument("--target-interface-cost", type=Path, help="admitted interface cost (publication gate)")
    parser.add_argument("--build-cache", type=Path, default=Path("runs/.build-cache"))
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    # One trace pass feeds the pool while it parses; on the 6-core Orin Nano 6 workers ran the same retained
    # documents 1.32-1.45x faster than 4 (bench/workers-bench.json), with identical results.
    parser.add_argument("--replay-workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--reuse-collection", type=Path,
                        help="earlier run directory whose completed PoTal/FullCPU collections are reused, not re-run")
    parser.add_argument("--stage-cache", type=Path,
                        help="content-addressed cache for replay/join/lifecycle/IR/schedule/timeline outputs")
    parser.add_argument("--stop-after", choices=OFFLINE_STAGES, help="end the run after this offline stage")
    parser.add_argument("--storage-factor", type=int, default=3,
                        help="offline preflight free-space multiple of workload bytes (certified CLI default 8)")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--reconstruct-timeout", type=int, default=36000)
    parser.add_argument("--keep-raw", action="store_true", help="keep cycle logs, traces, IR and schedule")
    parser.add_argument("--dry-run", action="store_true", help="print the detected platform and planned commands")
    parser.add_argument("--system", help="dry-run platform override")
    parser.add_argument("--machine", help="dry-run architecture override")
    return parser.parse_args()


def argv_json(argv: list[list[str]]) -> list[Json]:
    return [[part for part in command] for command in argv]


def plan(args: argparse.Namespace) -> Record:
    host = platform_profile(args.system, args.machine)
    output = args.output or Path("runs/DRY-RUN")
    cache = args.build_cache
    builds: Record = {"cycle-model": argv_json(cycle_model_argv(cache / "cycle-model", IM2P, args.jobs, host))}
    kinds: list[tuple[str, str, str, tuple[str, ...]]] = [
        ("potal", "potal-host", "STRIPE_PIPELINE", ()), ("fullcpu", "fullcpu-host", "FULL", ())]
    for name, kind, matmul, extra in kinds:
        build = llama_plan(kind, args.precision, args.dim, IM2P, matmul, extra)
        builds[name] = {"semantic_options_sha256": build.semantic_options_sha256,
                        "argv": argv_json(llama_argv(build, cache / name, args.jobs, host))}
    return {"platform": host.record(), "output": str(output), "profile": args.profile, "builds": builds,
            "build_domains": {"performance": ["cycle-model", "potal", "fullcpu"], "timeline": []},
            "stages": {"performance": ["detect", "build-cycle-model", "build-potal", "build-fullcpu", "collect-potal",
                                       "collect-fullcpu", "worker-scenario", "replay", "join", "producer-lifecycle",
                                       "execution-ir", "schedule (+performance accumulator, +timeline if compact)",
                                       "timeline (--timeline compact only)", "performance", "provenance"],
                       "timeline": ["stored-schedule", "timeline-export"]},
            "schedule_clock": {"hz": CONFIGURED_TEST_CLOCK_HZ, "status": "DIAGNOSTIC_CONFIGURED_TEST_CLOCK"}}


def main() -> int:
    args = arguments()
    args.profile = f"{args.precision}-d{args.dim}-hp1"
    require(args.dry_run or (args.system is None and args.machine is None), "platform overrides are dry-run only")
    if args.dry_run:
        print(json.dumps(plan(args), indent=2, sort_keys=True))
        return 0
    runners: dict[str, Callable[[argparse.Namespace], int]] = {
        "performance": performance_run, "timeline": export_timeline}
    return runners[args.run](args)


def performance_run(args: argparse.Namespace) -> int:
    """--run performance: collection (or a reused one), offline replay/join/lifecycle/IR, one scheduling pass with the
    performance accumulator (+ timeline rows with --timeline compact). Never builds or runs the activation/residual/
    SCU metrics."""
    model_path: Path | None = args.model
    prompt_path: Path | None = args.prompt_file
    require(model_path is not None and prompt_path is not None, "--run performance requires --model and --prompt-file")
    assert model_path is not None and prompt_path is not None
    require(args.from_run is None, "--from-run belongs to --run timeline")
    require(args.prompt_tokens == 256 and args.generate == GENERATED,
            "only the native E2E_GENERATION_256_128 recipe (256 prompt + 128 generated) is implemented; "
            "use --smoke for the separate 256+1 connectivity workload")
    args.generated = 1 if args.smoke else GENERATED
    args.model, args.prompt_file = model_path.resolve(strict=True), prompt_path.resolve(strict=True)
    certificate_set: Record | None = None
    if args.validation_mode == "certified":
        require(args.certificates is not None, "--certificates or IM2P_CERTIFICATE_SET is required")
        require(args.local_validation is None, "--local-validation belongs to --validation-mode nano-local")
        certificate_set = read_json(args.certificates.resolve(strict=True))
        require(certificate_set.get("schema") == "im2p-evaluation-certificate-set" and certificate_set.get("version") == 1,
                "unsupported certificate set")
        for name in ("cycle_certificate", "run_aware_certificate", "transition_certificate"):
            item = record(certificate_set.get(name))
            require(sha256(Path(text(item, "path"))) == text(item, "sha256"), "certificate set binding mismatch: " + name)
    else:
        require(args.certificates is None, "certificates cannot be combined with --validation-mode nano-local")
        receipt: Path | None = args.local_validation
        require(receipt is not None and args.model_manifest is not None,
                "--validation-mode nano-local requires --local-validation and --model-manifest")
        assert receipt is not None
        args.local_validation = receipt.resolve(strict=True)
        if str(IM2P) not in sys.path:
            sys.path.insert(0, str(IM2P))
    args.certificate_set = certificate_set
    clean_environment()
    host = platform_profile()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (args.output or Path("runs") / f"{stamp}-{args.model.stem}-{args.profile}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    validation: Record = {"validation_mode": VALIDATION[args.validation_mode],
                          "validation_receipt": reference(args.local_validation) if args.local_validation else None,
                          "publication_certified": False}
    header: Record = {"schema": "im2p-cycle-evaluation-run", "version": 1, "started_utc": stamp,
                      "argv": list(sys.argv), "profile": args.profile, "platform": host.record(),
                      "validation": validation, "workload": workload_mode(args)}
    run = Run(root, header)
    run.write("RUNNING")
    removed: list[Json] = []
    try:
        model = run.step("model-validation", lambda: gguf_identity(args.model))
        if args.model_manifest is not None:
            model["manifest"] = run.step("model-manifest", lambda: model_entry(args.model_manifest, args.model))
        dataset = {"path": str(args.prompt_file), "sha256": sha256(args.prompt_file),
                   "bytes": args.prompt_file.stat().st_size}
        reuse: Record | None = None
        potal = fullcpu = Path()
        if args.reuse_collection is not None:
            potal, fullcpu, reuse = run.step("reuse-collection", lambda: reused_collection(args))
            require(reuse.get("model_sha256") == model["sha256"], "reused collection used a different model")
            # The collection-time PMU contract is the measurement authority of the reused CPU timing.
            pmu = {**record(reuse.get("pmu_contract")), "inherited_from": reuse.get("source_run")}
        else:
            pmu = run.step("pmu-preflight", pmu_contract)
        write_json(root / "provenance.json", {"host": host_facts(), "platform": host.record(),
                                              "toolchain": toolchain(), "sources": source_state(),
                                              "model": model, "dataset": dataset, "certificates": certificate_set,
                                              "validation": validation, "workload": workload_mode(args),
                                              "pmu_contract": pmu, "collection_reuse": reuse})
        builds = build_all(args, run, collect=reuse is None)
        cycle_library = Path(text(record(record(record(builds["cycle-model"]).get("receipt")).get("library")), "path"))
        cache = StageCache((args.stage_cache or root / "stage-cache").resolve())
        if reuse is None:
            settings = root / "settings.json"
            write_json(settings, SETTINGS)
            potal, fullcpu = collect(args, run, builds, settings)
            potal_info = compiled_info(Path(text(record(builds["potal"]), "path")) / "bin/llama-eval-workload")
            require(potal_info.get("hp1") is True and potal_info.get("backend") == "IM2P_SIM",
                    "PoTal build is not IM2P_SIM HP1")
        replay = (reconstruct(args, run, potal, fullcpu, cycle_library) if args.validation_mode == "certified"
                  else reconstruct_local(args, run, potal, fullcpu, cycle_library, cache))
        identity = collection_identity(args, potal)
        pmu_required = pmu.get("status") == "READY"
        timing: Record = {"pmu": pmu, "potal": timing_validity(potal, pmu_required),
                  "fullcpu": timing_validity(fullcpu, pmu_required)}
        write_json(root / "timing-provenance.json", {"schema": "potal-cpu-timing-provenance", "version": 1, **timing,
                                                     "host": host_facts()})
        if not args.keep_raw and reuse is None:
            discard(run, [directory / name for directory in (potal, fullcpu) for name in RAW_LARGE], removed)
        frequency, validated = CONFIGURED_TEST_CLOCK_HZ, False
        if args.clock_selection is not None:
            frequency = run.step("operating-clock", lambda: clock_frequency(args.clock_selection.resolve(strict=True),
                                                                            args.profile, IM2P))
            validated = True
        axis = Axis(frequency, validated)
        timeline_dir = root / "timeline"
        timeline = timeline_dir / "timeline.jsonl"
        pending: Path | None = None
        if args.timeline == "compact":
            timeline_dir.mkdir()
            pending = timeline_dir / "in-pass-timeline.jsonl.partial"
        # One scheduling pass: stored schedule + performance accumulator (+ timeline rows when requested).
        schedule_path = schedule(args, run, replay, axis, cache, pending)
        computed = read_json(replay / "schedule-performance.json")
        count: int | None = None
        if args.timeline == "compact":
            count = timeline_stage(args, run, cache, replay, schedule_path, axis, pending, timeline_dir,
                                   integer(computed, "rows"))
        result, checks = run.step("performance", lambda: (record(computed.get("result")), record(computed.get("checks"))))
        require(checks["status"] == "PASS", "timeline invariants failed: " + json.dumps(checks))
        summary = read_json(replay / "npu-summary.json")
        consistency = cross_check(result, summary, args.generated)
        clock: Record = {"schedule_clock_hz": frequency,
                         "status": "VALIDATED_OPERATING_CLOCK" if validated else "DIAGNOSTIC_CONFIGURED_TEST_CLOCK",
                         "artifact": reference(args.clock_selection.resolve()) if validated else None,
                         "source": "clock selection artifact" if validated else
                                   "configured test clock of the interleaved schedule pins; placement only"}
        if count is not None:
            write_json(timeline_dir / "summary.json", timeline_summary(
                count, timeline, clock, checks, consistency, workload_mode(args), validation,
                VALIDATION[args.validation_mode]))
        replay_result = read_json(replay / "result.json")
        readiness = publication_readiness(
            {name: {"path": str(path.resolve(strict=True)), "sha256": sha256(path.resolve(strict=True))}
             for name, path in (("clock_selection", args.clock_selection), ("target_host_timing", args.target_host_timing),
                                ("target_interface_cost", args.target_interface_cost)) if path is not None},
            {"model_sha256": model["sha256"], "input_tokens_sha256": identity["input_tokens_sha256"],
             "generated_tokens_sha256": identity["generated_tokens_sha256"],
             "profile": args.profile}, {text(identity, "host_id")}, False, IM2P)
        run.markers.update({
            "OPERATING_CLOCK_READY": "READY" if readiness["clock_ready"] else "NOT_READY",
            "TARGET_CPU_TIMING_READY": "READY" if readiness["target_host_ready"] else "NOT_READY",
            "TARGET_INTERFACE_COST_READY": "READY" if readiness["target_interface_ready"] else "NOT_READY",
            "PUBLICATION_E2E_MS_READY": "NOT_READY"})
        write_json(root / "performance.json", {
            "schema": "im2p-cycle-evaluation-performance", "version": 1, "workload": {
                "model": model, "dataset": dataset, **workload_mode(args),
                "chunk_id": 0, "profile": args.profile,
                "input_tokens_sha256": identity["input_tokens_sha256"],
                "generated_tokens_sha256": identity["generated_tokens_sha256"]},
            "validation": validation,
            **result, "timeline_checks": checks, "replay_consistency": consistency, "clock": clock,
            "npu_cycle_source": {"replay_summary": reference(replay / "npu-summary.json"),
                                 "replay_mode": replay_result.get("replay_mode"),
                                 "validation_scope": summary.get("validation_scope"),
                                 "cycle_library_sha256": summary.get("cycle_library_sha256")},
            "publication": readiness, "offline_pipeline": reference(replay / "result.json"),
            "execution": execution_record(args, reuse, replay, run.cache)})
        if not args.keep_raw:
            discard(run, [replay / "execution.sqlite", schedule_path], removed)
        provenance = read_json(root / "provenance.json")
        provenance.update(removed_raw=removed, stages=run.stages)
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        run.markers.update({"ONE_COMMAND_EVALUATION_READY": "READY",
                            "TIMELINE_LOG_READY": "READY" if count is not None else "NOT_REQUESTED",
                            "FAST_CYCLE_EVALUATION_RUN": "PASS", "MEASUREMENT_DOMAIN": "performance",
                            **NOT_RUN_DOMAINS,
                            "VALIDATION_MODE": VALIDATION[args.validation_mode], "PUBLICATION_CERTIFIED": "false",
                            "WORKLOAD_MODE": text(workload_mode(args), "workload_mode"),
                            "TPOT_READY": "NOT_APPLICABLE" if args.generated == 1 else "NPU_CYCLES_ONLY",
                            **instrumentation_markers(timing, result, args.generated)})
        run.write("PASS")
        checksums(root)
        print(json.dumps({"output": str(root), "markers": run.markers,
                          "ttft_npu_cycles": record(result["ttft"]).get("npu_cycles"),
                          "npu_tpot_cycles": record(result["tpot"]).get("npu_tpot_cycles")}, sort_keys=True))
        return 0
    except StopAfter as stop:
        run.markers.update({"STOPPED_AFTER": str(stop), "ONE_COMMAND_EVALUATION_READY": "NOT_READY",
                            "PUBLICATION_E2E_MS_READY": "NOT_READY"})
        run.write("STOPPED_AFTER_" + str(stop).upper().replace("-", "_"))
        print(json.dumps({"output": str(root), "stopped_after": str(stop), "stage_cache": run.cache}, sort_keys=True))
        return 0
    except (EvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
        if not run.stages or record(run.stages[-1]).get("status") != "FAILED":
            run.stages.append({"stage": "runner", "status": "FAILED", "reason": str(error)})
        run.write("FAILED")
        print(f"cycle evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
