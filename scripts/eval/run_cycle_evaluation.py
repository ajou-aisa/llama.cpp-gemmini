#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B scripts/eval/run_cycle_evaluation.py --model M.gguf --prompt-file wiki.test.raw \
#   --prompt-tokens 256 --generate 128 --precision a8w8 --dim 32 --certificates certificate-set.json --output DIR
"""One-command cycle evaluation: host detection, cached builds, PoTal/FullCPU workload, FAST_EVALUATION replay,
official join/lifecycle/IR, SYNTHETIC schedule timeline, TTFT/TPOT with timing ownership, quality metrics.

Nothing here certifies anything: certified inputs (cycle library identity, base/run-aware/transition
certificates) are admitted by the unchanged official stages. NPU wall time exists only with a validated
operating clock; CPU target timing and interface cost stay NOT_READY/UNMODELED until evidence exists.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
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
from e2e_timeline import Axis, isolated_phase_table, performance, timeline_rows, worker_scenario, write_timeline
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
from target_admission import clock_frequency, publication_readiness

EVAL: Final = Path(__file__).resolve().parent
IM2P: Final = REPO.parent / "IM2P.sim"
CONFIGURED_TEST_CLOCK_HZ: Final = 1_000_000_000  # sim/cycle/interleaved_schedule_pins.py npu_frequency_hz
GENERATED: Final = 128
SETTINGS: Final[Record] = {"schema": "potal-evaluation-settings", "version": 2, "seed": 1234, "temperature": 0,
    "top_k": 0, "top_p": 1, "min_p": 0, "repeat_penalty": 1, "repeat_last_n": 0, "grammar": None,
    "eos_stopping": False, "warmup": 0, "sampler_policy": "user-confirmed-greedy-v3", "split": "test",
    "chunk_ids": list(range(10)), "threads": 1, "threads_batch": 1, "batch_size": 256, "ubatch_size": 256,
    "scope": "DEVELOPMENT_EVALUATION_NOT_PAPER_CAMPAIGN"}
RAW_LARGE: Final = ("native/chunk-0/cycle-log.jsonl", "native/chunk-0/npu-cycle-trace.jsonl",
                    "native/chunk-0/semantic-graph.jsonl", "native/chunk-0/execution-lifecycle.jsonl")
T = TypeVar("T")


def reference(path: Path) -> Record:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def call(argv: list[str], log: Path, cwd: Path, timeout: int) -> None:
    with log.open("x", encoding="utf-8") as stream:
        done = subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    require(done.returncode == 0, f"stage command failed ({done.returncode}); see {log}")


class Run:
    """Stage ledger: every stage is timed and recorded, and a failure leaves an explicit manifest."""

    def __init__(self, root: Path, header: Record) -> None:
        self.root, self.header, self.stages = root, header, list[Json]()
        self.markers: Record = {}

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
        manifest: Record = {**self.header, "status": status, "stages": self.stages, "markers": self.markers}
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


def build_all(args: argparse.Namespace, run: Run, quality: bool) -> Record:
    cache = args.build_cache.resolve()
    builds: Record = {}
    (run.root / "build").mkdir()
    cycle, hit = run.step("build-cycle-model", lambda: cached_cycle_model(cache, IM2P, args.jobs))
    receipt = read_json(cycle / "build-receipt.json")
    library = record(receipt.get("library"))
    require(library.get("sha256") == args.certificate_set["cycle_library_sha256"],
            "fresh cycle library differs from the certified CURRENT library; certified replay would reject it")
    builds["cycle-model"] = {"path": str(cycle), "cache_hit": hit, "receipt": receipt, "certified_identity_match": True}
    kinds: list[tuple[str, str, str, tuple[str, ...]]] = [("potal", "potal-host", "STRIPE_PIPELINE", ()),
                                                          ("fullcpu", "fullcpu-host", "FULL", ())]
    if quality:
        kinds.append(("metrics", "cycle", "STRIPE_PIPELINE", ("llama-perplexity",)))
    for name, kind, matmul, extra in kinds:
        path, hit = run.step("build-" + name, lambda kind=kind, matmul=matmul, extra=extra: cached_llama_build(
            cache, kind, args.precision, args.dim, IM2P, args.jobs, matmul, extra))
        builds[name] = {"path": str(path), "cache_hit": hit, "receipt": read_json(path / "build-receipt.json"),
                        "build_info": read_json(path / "build-info.json")}
    if quality:
        semantic = {name: record(record(builds[name]).get("receipt")).get("semantic_options_sha256")
                    for name in ("potal", "metrics")}
        require(len(set(semantic.values())) == 1, "metrics build semantic configuration differs from PoTal")
    for name, value in builds.items():
        write_json(run.root / "build" / (name + ".json"), value)
    return builds


def collect(args: argparse.Namespace, run: Run, builds: Record, settings: Path) -> tuple[Path, Path]:
    raw = run.root / "raw"
    raw.mkdir()
    base = [sys.executable, "-B", str(EVAL / "end_to_end.py")]
    common = ["--im2p", str(IM2P), "--model", str(args.model), "--dataset", str(args.prompt_file),
              "--settings", str(settings), "--repetitions", "1", "--timeout", str(args.timeout)]
    potal, fullcpu = raw / "potal", raw / "fullcpu"
    runner = {name: str(Path(text(record(builds[name]), "path")) / "bin/llama-eval-workload")
              for name in ("potal", "fullcpu")}
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
            "--potal-result", str(potal / "result.json"),
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


def discard(run: Run, paths: list[Path], ledger: list[Json]) -> None:
    for path in paths:
        if path.is_file():
            ledger.append({**reference(path), "status": "REMOVED_AFTER_USE"})
            path.unlink()


def schedule(args: argparse.Namespace, run: Run, replay: Path, frequency_hz: int) -> Path:
    table = replay / "isolated-phase-table.json"
    write_json(table, isolated_phase_table(replay / "npu-cycle-result.jsonl"))
    output = replay / "schedule.sqlite"
    run.step("schedule", lambda: call([sys.executable, "-B", "-m", "sim.cycle.execution_cli", "schedule",
                                       "--bundle", str(replay / "execution.sqlite"), "--phase-table", str(table),
                                       "--frequency-hz", str(frequency_hz), "--synthetic", "--output", str(output)],
                                      replay / "schedule.log", IM2P, args.reconstruct_timeout))
    return output


def ppl_config(chunks: int) -> Record:
    return {"context": 256, "batch": 256, "threads": 1, "chunks": chunks, "seed": 1234,
            "mask": "second half of each 256-token context (native perplexity_half)"}


def quality_identity(model: Path, dataset: Path, expected_model: Record, expected_dataset: Record) -> None:
    """Quality metrics must use the exact model and text that produced the performance run."""
    require(sha256(model) == expected_model.get("sha256"), "quality model differs from the performance model")
    require(sha256(dataset) == expected_dataset.get("sha256"), "quality dataset differs from the performance dataset")


def perplexity(args: argparse.Namespace, run: Run, builds: Record, model: Record, dataset: Record) -> Record:
    metrics = record(builds["metrics"])
    binary = Path(text(metrics, "path")) / "bin/llama-perplexity"
    quality_identity(args.model, args.prompt_file, model, dataset)
    config = ppl_config(args.ppl_chunks)
    argv = [str(binary), "-m", str(args.model), "-f", str(args.prompt_file), "-c", "256", "-b", "256",
            "-t", "1", "--chunks", str(args.ppl_chunks if args.ppl_chunks > 0 else -1), "-s", "1234"]
    log = run.root / "metrics" / "perplexity.log"
    log.parent.mkdir()
    run.step("quality-perplexity", lambda: call(argv, log, run.root, args.timeout * 4))
    output = log.read_text(errors="replace")
    final = re.findall(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", output)
    chunks = re.findall(r"calculating perplexity over (\d+) chunks", output)
    require(len(final) == 1 and len(chunks) == 1, "perplexity output lacks one final estimate")
    quality_identity(args.model, args.prompt_file, model, dataset)
    return {"name": "perplexity", "value": float(final[0][0]), "uncertainty": float(final[0][1]), "unit": "PPL",
            "workload": {"dataset": dataset, "split": "test", "context_tokens": 256,
                         "chunks_evaluated": int(chunks[0]),
                         "coverage": "FULL_COMPLETE_CHUNKS" if args.ppl_chunks <= 0 else "BOUNDED_CHUNK_SUBSET"},
            "model": model, "runner": {"path": str(binary), "sha256": sha256(binary)},
            "build_receipt": reference(Path(text(metrics, "path")) / "build-receipt.json"),
            "semantic_options_sha256": record(metrics.get("receipt")).get("semantic_options_sha256"),
            "config": config, "config_sha256": sha256_text(json.dumps(config, sort_keys=True)),
            "argv": list(argv), "log": reference(log)}


def sha256_text(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def cross_check(result: Record, summary: Record) -> Record:
    per_token = record(result.get("tpot")).get("per_token_npu_cycles")
    expected = [integer(record(summary.get("per_decode_token_cycle_sums")), str(index)) for index in range(GENERATED - 1)]
    checks: Record = {"ttft_npu_equals_replay_prefill_sum":
                      record(result.get("ttft")).get("npu_cycles") == integer(summary, "prefill_cycle_sum"),
                      "per_token_npu_equals_replay_decode_sums": per_token == expected}
    require(all(value is True for value in checks.values()), "timeline NPU cycles differ from the replay summary")
    return checks


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True, help="WikiText-2 raw test text (native recipe)")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--generate", type=int, default=GENERATED)
    parser.add_argument("--precision", choices=("a4w4", "a8w8"), default="a8w8")
    parser.add_argument("--dim", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--certificates", type=Path, default=os.environ.get("IM2P_CERTIFICATE_SET"),
                        help="certificate set JSON (or IM2P_CERTIFICATE_SET)")
    parser.add_argument("--output", type=Path, help="fresh run directory; default runs/<utc>-<model>-<config>")
    parser.add_argument("--run", default="performance,quality", help="comma list of performance,quality")
    parser.add_argument("--clock-selection", type=Path, help="validated operating clock; enables NPU ms")
    parser.add_argument("--target-host-timing", type=Path, help="admitted target-host timing (publication gate)")
    parser.add_argument("--target-interface-cost", type=Path, help="admitted interface cost (publication gate)")
    parser.add_argument("--build-cache", type=Path, default=Path("runs/.build-cache"))
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--replay-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--storage-factor", type=int, default=3,
                        help="offline preflight free-space multiple of workload bytes (certified CLI default 8)")
    parser.add_argument("--ppl-chunks", type=int, default=64, help="bounded PPL chunks; 0 evaluates all")
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
        ("potal", "potal-host", "STRIPE_PIPELINE", ()), ("fullcpu", "fullcpu-host", "FULL", ()),
        ("metrics", "cycle", "STRIPE_PIPELINE", ("llama-perplexity",))]
    for name, kind, matmul, extra in kinds:
        build = llama_plan(kind, args.precision, args.dim, IM2P, matmul, extra)
        builds[name] = {"semantic_options_sha256": build.semantic_options_sha256,
                        "argv": argv_json(llama_argv(build, cache / name, args.jobs, host))}
    return {"platform": host.record(), "output": str(output), "profile": args.profile, "builds": builds,
            "stages": ["detect", "build", "collect-potal", "collect-fullcpu", "worker-scenario", "reconstruct",
                       "schedule", "timeline", "performance", "quality-perplexity", "provenance"],
            "schedule_clock": {"hz": CONFIGURED_TEST_CLOCK_HZ, "status": "DIAGNOSTIC_CONFIGURED_TEST_CLOCK"}}


def main() -> int:
    args = arguments()
    args.profile = f"{args.precision}-d{args.dim}-hp1"
    require(args.dry_run or (args.system is None and args.machine is None), "platform overrides are dry-run only")
    if args.dry_run:
        print(json.dumps(plan(args), indent=2, sort_keys=True))
        return 0
    parts = set(args.run.split(","))
    require(parts <= {"performance", "quality"} and "performance" in parts, "--run must include performance")
    require(args.prompt_tokens == 256 and args.generate == GENERATED,
            "only the native E2E_GENERATION_256_128 recipe (256 prompt + 128 generated) is implemented")
    require(args.certificates is not None, "--certificates or IM2P_CERTIFICATE_SET is required")
    args.model, args.prompt_file = args.model.resolve(strict=True), args.prompt_file.resolve(strict=True)
    certificate_set = read_json(args.certificates.resolve(strict=True))
    require(certificate_set.get("schema") == "im2p-evaluation-certificate-set" and certificate_set.get("version") == 1,
            "unsupported certificate set")
    for name in ("cycle_certificate", "run_aware_certificate", "transition_certificate"):
        item = record(certificate_set.get(name))
        require(sha256(Path(text(item, "path"))) == text(item, "sha256"), "certificate set binding mismatch: " + name)
    args.certificate_set = certificate_set
    clean_environment()
    host = platform_profile()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (args.output or Path("runs") / f"{stamp}-{args.model.stem}-{args.profile}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    header: Record = {"schema": "im2p-cycle-evaluation-run", "version": 1, "started_utc": stamp,
                      "argv": list(sys.argv), "profile": args.profile, "platform": host.record()}
    run = Run(root, header)
    run.write("RUNNING")
    removed: list[Json] = []
    try:
        model = run.step("model-validation", lambda: gguf_identity(args.model))
        dataset = {"path": str(args.prompt_file), "sha256": sha256(args.prompt_file),
                   "bytes": args.prompt_file.stat().st_size}
        write_json(root / "provenance.json", {"host": host_facts(), "platform": host.record(),
                                              "toolchain": toolchain(), "sources": source_state(),
                                              "model": model, "dataset": dataset, "certificates": certificate_set})
        builds = build_all(args, run, "quality" in parts)
        cycle_library = Path(text(record(record(record(builds["cycle-model"]).get("receipt")).get("library")), "path"))
        settings = root / "settings.json"
        write_json(settings, SETTINGS)
        potal, fullcpu = collect(args, run, builds, settings)
        potal_info = compiled_info(Path(text(record(builds["potal"]), "path")) / "bin/llama-eval-workload")
        require(potal_info.get("hp1") is True and potal_info.get("backend") == "IM2P_SIM", "PoTal build is not IM2P_SIM HP1")
        replay = reconstruct(args, run, potal, fullcpu, cycle_library)
        if not args.keep_raw:
            discard(run, [directory / name for directory in (potal, fullcpu) for name in RAW_LARGE], removed)
        frequency, validated = CONFIGURED_TEST_CLOCK_HZ, False
        if args.clock_selection is not None:
            frequency = run.step("operating-clock", lambda: clock_frequency(args.clock_selection.resolve(strict=True),
                                                                            args.profile, IM2P))
            validated = True
        schedule_path = schedule(args, run, replay, frequency)
        timeline_dir = root / "timeline"
        timeline_dir.mkdir()
        timeline = timeline_dir / "timeline.jsonl"
        count = run.step("timeline", lambda: write_timeline(timeline, timeline_rows(
            schedule_path, replay / "execution.sqlite", replay / "npu-cycle-result.jsonl", Axis(frequency, validated))))
        result, checks = run.step("performance", lambda: performance(timeline, GENERATED, frequency if validated else None))
        require(checks["status"] == "PASS", "timeline invariants failed: " + json.dumps(checks))
        summary = read_json(replay / "npu-summary.json")
        consistency = cross_check(result, summary)
        clock: Record = {"schedule_clock_hz": frequency,
                         "status": "VALIDATED_OPERATING_CLOCK" if validated else "DIAGNOSTIC_CONFIGURED_TEST_CLOCK",
                         "artifact": reference(args.clock_selection.resolve()) if validated else None,
                         "source": "clock selection artifact" if validated else
                                   "configured test clock of the interleaved schedule pins; placement only"}
        write_json(timeline_dir / "summary.json", {
            "schema": "im2p-e2e-timeline-summary", "version": 1, "rows": count, "timeline": reference(timeline),
            "axis": {"unit": "npu_cycle", "clock": clock,
                     "ns_fields": "non-null only under a validated operating clock",
                     "values": "integers when exact, else doubles; exact rationals are in the schedule"},
            "scope": "SYNTHETIC_ONLY schedule: isolated certified NPU service, development-host CPU durations, "
                     "interface UNMODELED; a model, never an observation",
            "lanes": {"cpu": "resource (scheduler lane); tid keeps the observed host thread",
                      "npu": "npu:0"},
            "checks": checks, "replay_consistency": consistency,
            "row_fields": ["seq", "row_type", "kind", "source", "resource", "tid", "worker_id", "node_id",
                           "work_id", "op", "layer", "phase", "decode_index", "token_index", "start_cycle",
                           "end_cycle", "duration_cycles", "start_ns", "end_ns", "duration_ns", "npu_cycles",
                           "npu_ms", "host_elapsed_ns", "host_thread_cpu_ns", "target_cpu_cycles", "target_cpu_ms",
                           "evidence_id", "event", "timing_source", "source_line"]})
        replay_result = read_json(replay / "result.json")
        readiness = publication_readiness(
            {name: {"path": str(path.resolve(strict=True)), "sha256": sha256(path.resolve(strict=True))}
             for name, path in (("clock_selection", args.clock_selection), ("target_host_timing", args.target_host_timing),
                                ("target_interface_cost", args.target_interface_cost)) if path is not None},
            {"model_sha256": model["sha256"], "input_tokens_sha256": read_json(potal / "result.json").get("input_tokens_sha256"),
             "generated_tokens_sha256": read_json(potal / "result.json").get("generated_tokens_sha256"),
             "profile": args.profile}, {text(read_json(potal / "result.json"), "host_id")}, False, IM2P)
        run.markers.update({
            "OPERATING_CLOCK_READY": "READY" if readiness["clock_ready"] else "NOT_READY",
            "TARGET_CPU_TIMING_READY": "READY" if readiness["target_host_ready"] else "NOT_READY",
            "TARGET_INTERFACE_COST_READY": "READY" if readiness["target_interface_ready"] else "NOT_READY",
            "PUBLICATION_E2E_MS_READY": "NOT_READY"})
        write_json(root / "performance.json", {
            "schema": "im2p-cycle-evaluation-performance", "version": 1, "workload": {
                "model": model, "dataset": dataset, "prompt_tokens": 256, "generated_tokens": GENERATED,
                "chunk_id": 0, "profile": args.profile,
                "input_tokens_sha256": read_json(potal / "result.json").get("input_tokens_sha256"),
                "generated_tokens_sha256": read_json(potal / "result.json").get("generated_tokens_sha256")},
            **result, "timeline_checks": checks, "replay_consistency": consistency, "clock": clock,
            "npu_cycle_source": {"replay_summary": reference(replay / "npu-summary.json"),
                                 "replay_mode": replay_result.get("replay_mode"),
                                 "validation_scope": summary.get("validation_scope"),
                                 "cycle_library_sha256": summary.get("cycle_library_sha256")},
            "publication": readiness, "offline_pipeline": reference(replay / "result.json")})
        if "quality" in parts:
            metric = perplexity(args, run, builds, model, dataset)
            write_json(root / "metrics.json", {"schema": "im2p-cycle-evaluation-metrics", "version": 1,
                                               "metrics": [metric], "shared_identity": {
                                                   "model_sha256": model["sha256"],
                                                   "performance_semantic_options_sha256": record(record(
                                                       builds["potal"]).get("receipt")).get("semantic_options_sha256"),
                                                   "metrics_semantic_options_sha256": metric["semantic_options_sha256"]}})
        if not args.keep_raw:
            discard(run, [replay / "execution.sqlite", schedule_path], removed)
        provenance = read_json(root / "provenance.json")
        provenance.update(removed_raw=removed, stages=run.stages)
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        run.markers.update({"ONE_COMMAND_EVALUATION_READY": "READY", "TIMELINE_LOG_READY": "READY",
                            "FAST_CYCLE_EVALUATION_RUN": "PASS",
                            "METRICS_AUTOMATION_RUN": "PASS" if "quality" in parts else "NOT_RUN"})
        run.write("PASS")
        sums = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
        (root / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(root)}\n" for path in sums))
        print(json.dumps({"output": str(root), "markers": run.markers,
                          "ttft_npu_cycles": record(result["ttft"]).get("npu_cycles"),
                          "npu_tpot_cycles": record(result["tpot"]).get("npu_tpot_cycles")}, sort_keys=True))
        return 0
    except (EvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
        if not run.stages or record(run.stages[-1]).get("status") != "FAILED":
            run.stages.append({"stage": "runner", "status": "FAILED", "reason": str(error)})
        run.write("FAILED")
        print(f"cycle evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
