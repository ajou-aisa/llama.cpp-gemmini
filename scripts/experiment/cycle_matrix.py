#!/usr/bin/env python3
"""Run the CPU inference/cycle acceptance matrix (Python >= 3.9, stdlib only).

From the repository root on Nano:
  python3 scripts/experiment/cycle_matrix.py --plan
  python3 scripts/experiment/cycle_matrix.py --base-build build-arm64-cpu

The JSON manifest defines models, exact quantization filename patterns and prompts.
Every run creates EMPTY, run-specific A4/W4, A8/W8 and A16/W16 build directories.
Only settings are read from the base CMake cache; no binaries/objects are reused.
The base build and model files are never modified. This is NOT an NPU/RTL test.
Missing/ambiguous models, unavailable PMU values, incomplete summaries and invalid
analyzer output FAIL the case. Partial selections are explicitly labelled as such.
Each run has fresh logs; no old log can satisfy a new case. CYCLE_MATRIX_OK and
OK.txt are issued ONLY after the complete bundled matrix passes, including fresh
builds and unchanged-source/artifact checks. Copy that final line as the receipt.
Selections get CYCLE_MATRIX_SELECTION_PASSED instead, never the full OK marker.
Exit 0 means the selected cases passed (--plan only checks inputs). Output includes
results.json/CSV, commands, exit codes, build logs, raw logs, summaries and traces.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("prefill", "decode")
DEFAULT_CONFIG = Path(__file__).with_name("cycle-matrix.json")
CASE_GATES = ("build_ok", "inference_ok", "summary_ok", "analyzer_ok", "artifacts_ok")
# Only user configuration is copied, not CMake's generated paths or probe results.
CACHE_KEYS = {
    "GEMMINI_SW_PATH", "BUILD_SHARED_LIBS", "CMAKE_BUILD_TYPE",
    "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_TOOLCHAIN_FILE",
    "GGML_NATIVE", "GGML_OPENMP", "GGML_CPU_ARM_ARCH", "GGML_LLAMAFILE",
    "GGML_BLAS", "GGML_BLAS_VENDOR", "GGML_BACKEND_DL",
}


class CheckError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def json_read(path: Path) -> Any:
    def invalid_constant(value: str) -> None:
        raise CheckError(f"non-finite JSON constant: {value}")
    def unique_keys(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=invalid_constant, object_pairs_hook=unique_keys)


def json_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def absolute(path: str, root: Path = ROOT) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_keys(config: dict) -> set[tuple[str, int, str]]:
    return {(model["name"], bits, family) for model in config["models"]
            for bits in config["bits"] for family in config["families"]}


def source_snapshot() -> dict[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    digest = hashlib.sha256(commit + b"\0" + patch)
    # Untracked build/results directories do not matter; untracked source does.
    extra = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z", "--",
                                     "src", "ggml", "common", "cmake", "scripts", "tools", "tests"], cwd=ROOT)
    for name in sorted(filter(None, extra.split(b"\0"))):
        digest.update(name + b"\0" + file_hash(ROOT / os.fsdecode(name)).encode())
    return {"commit": commit.decode(), "sha256": digest.hexdigest()}


def build_artifacts(build: Path) -> dict[str, str]:
    paths = {build / "bin" / name for name in ("llama-cli", "llama-cycle-summary")}
    paths.add(build / "CMakeCache.txt")
    for pattern in ("lib*.so*", "lib*.dylib*"):
        paths.update((build / "bin").glob(pattern))
    result = {}
    for path in sorted(paths):
        require(path.is_file() and build.resolve() in path.resolve().parents,
                f"missing or nonlocal build artifact: {path}")
        result[str(path.relative_to(build))] = file_hash(path)
    return result


def check_inputs(case: dict) -> None:
    model = Path(case["model_path"])
    stat = model.stat()
    require((stat.st_size, stat.st_mtime_ns) == (case["model_bytes"], case["model_mtime_ns"]),
            f"model changed after preflight: {model}")
    require(file_hash(Path(case["prompt"])) == case["prompt_sha256"], "prompt changed after preflight")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def cache_read(directory: Path) -> dict[str, tuple[str, str]]:
    cache = directory / "CMakeCache.txt"
    require(cache.is_file(), f"missing base/configured cache: {cache}")
    entries = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([^#/:][^:]*):([^=]+)=(.*)$", line)
        if match:
            entries[match[1]] = (match[2], match[3])
    require(entries.get("CMAKE_HOME_DIRECTORY", (None, None))[1] == str(ROOT),
            f"CMake cache belongs to another source tree: {cache}")
    return entries


def input_cases(config: dict, model_root: Path, models: list[str] | None,
                bits: list[int] | None, families: list[str] | None) -> list[dict]:
    require(config.get("version") == 1, "unsupported matrix manifest version")
    definitions = config["models"]
    names = [model["name"] for model in definitions]
    require(len(names) == len(set(names)), "duplicate model names in manifest")
    selected_models = names if models is None else models
    selected_bits = config["bits"] if bits is None else bits
    selected_families = config["families"] if families is None else families
    for selected, known, label in ((selected_models, names, "models"),
                                  (selected_bits, config["bits"], "bits"),
                                  (selected_families, config["families"], "families")):
        require(bool(selected) and len(selected) == len(set(selected)) and set(selected) <= set(known),
                f"unknown, empty or repeated {label}: {selected}")
    cases = []
    for model in definitions:
        if model["name"] not in selected_models:
            continue
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model["name"]) is not None,
                "model names must be safe directory basenames")
        for width in selected_bits:
            require(width in (4, 8, 16), f"unsupported operand width: {width}")
            for family in selected_families:
                require(family in ("0", "H1", "HP1"), f"unsupported family: {family}")
                case = {"name": f"{model['name']}-Q{width}_{family}", "model": model["name"],
                        "bits": width, "family": family, "status": "pending", "errors": []}
                try:
                    candidates = set()
                    for pattern in model["files"]:
                        expanded = pattern.format(bits=width, family=family)
                        require(not Path(expanded).is_absolute() and ".." not in Path(expanded).parts,
                                f"model glob must be relative to --model-root: {expanded}")
                        candidates.update(p.resolve() for p in model_root.glob(expanded) if p.is_file())
                    require(len(candidates) == 1,
                            f"expected one model, found {len(candidates)}: " +
                            ", ".join(map(str, sorted(candidates))))
                    path = candidates.pop()
                    with path.open("rb") as stream:
                        require(stream.read(4) == b"GGUF", f"not a GGUF file: {path}")
                    prompt = absolute(model["prompt"])
                    require(prompt.is_file() and prompt.stat().st_size > 0, f"missing/empty prompt: {prompt}")
                    case.update(model_path=str(path), model_bytes=path.stat().st_size,
                                model_mtime_ns=path.stat().st_mtime_ns, prompt=str(prompt),
                                prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest())
                except (CheckError, OSError) as error:
                    case.update(status="failed", errors=[f"preflight: {error}"])
                cases.append(case)
    return cases


def execute(command: list[str], directory: Path, label: str, env: dict[str, str],
            timeout: int, cwd: Path = ROOT) -> None:
    json_write(directory / f"{label}.command.json", {"argv": command, "cwd": str(cwd)})
    stdout = directory / f"{label}.stdout.txt"
    stderr = directory / f"{label}.stderr.txt"
    started = time.monotonic()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            # Also terminate descendants; never leave an inference/build running.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            # A child can survive after the group leader exits on SIGTERM.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            json_write(directory / f"{label}.result.json", {
                "success": False, "exit_code": process.returncode,
                "error": type(error).__name__, "elapsed_seconds": time.monotonic() - started})
            raise
    json_write(directory / f"{label}.result.json", {
        "success": code == 0, "exit_code": code, "elapsed_seconds": time.monotonic() - started})
    require(code == 0, f"{label}: exit={code}; see {stderr}")


def prepare_build(args: argparse.Namespace, base: dict, width: int, logs: Path) -> Path:
    # logs.parent is the exclusive output/run ID, not the reusable base build.
    build = absolute(args.build_root) / logs.parent.name / f"a{width}-w{width}-detail{args.detail}"
    require(not build.exists(), f"fresh build directory already exists: {build}")
    build.mkdir(parents=True, exist_ok=False)
    required = {
        "GGML_GEMMINI": "ON", "GGML_GEMMINI_OPTION": "CPU",
        "GGML_GEMMINI_EXECUTION_BACKEND": "HARDWARE",
        "GGML_GEMMINI_DEFAULT_RMD_BACKEND": "CPU",
        "GGML_GEMMINI_ACTIVATION_BITS": str(width), "GGML_GEMMINI_WEIGHT_BITS": str(width),
        "LOG_CYCLE": "1", "CYCLE_DETAIL": str(args.detail),
        "LLAMA_BUILD_COMMON": "ON", "LLAMA_BUILD_TOOLS": "ON",
        "LLAMA_BUILD_TESTS": "OFF", "LLAMA_BUILD_SERVER": "OFF", "LLAMA_BUILD_EXAMPLES": "OFF",
        "LLAMA_CURL": "OFF", "GGML_METAL": "OFF", "GGML_CUDA": "OFF", "IM2P_SIM_IMPLEMENTATION": "",
        "CMAKE_C_COMPILER_LAUNCHER": "", "CMAKE_CXX_COMPILER_LAUNCHER": "",
    }
    options = {key: value for key, (kind, value) in base.items()
               if kind in ("BOOL", "STRING", "PATH", "FILEPATH", "UNINITIALIZED")
               and (key in CACHE_KEYS or key.startswith("GGML_GEMMINI_"))}
    options.update(required)
    command = ["cmake", "-S", str(ROOT), "-B", str(build)]
    command += [f"-D{key}={value}" for key, value in sorted(options.items())]
    execute(command, logs, "configure", os.environ.copy(), args.timeout)
    execute(["cmake", "--build", str(build), "--target", "llama-cli", "llama-cycle-summary",
             "-j", str(args.jobs)], logs, "build", os.environ.copy(), args.timeout)
    configured = cache_read(build)
    for key, value in required.items():
        actual = configured.get(key, (None, None))[1]
        # CMake permits ON/1/TRUE spellings for booleans.
        truth = lambda item: str(item).upper() in ("1", "ON", "TRUE", "YES")
        same = truth(actual) == truth(value) if value in ("ON", "OFF") else actual == value
        require(actual is not None and same, f"wrong build configuration {key}: {actual!r}, expected {value!r}")
    for executable in ("llama-cli", "llama-cycle-summary"):
        require(os.access(build / "bin" / executable, os.X_OK), f"missing executable: {build / 'bin' / executable}")
    json_write(logs / "cache.json", {key: value for key, (_, value) in configured.items()})
    json_write(logs / "build.json", {
        "path": str(build), "logs": str(logs), "fresh": True, "bits": width, "detail": args.detail,
        "artifacts": build_artifacts(build)})
    return build


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_summary(summary: dict, tokens: int, runtime: str) -> list[str]:
    require(summary.get("record_type") == "FINAL_INFERENCE_SUMMARY" and summary.get("available") is True,
            f"summary unavailable: {summary.get('reason')}")
    require(summary.get("requests") == summary.get("completed_requests") == 1,
            "summary must describe one completed request")
    require(summary.get("tokens") == tokens, "generated token count differs from the requested count")
    for key in ("cpu_interval_coverage", "latency_boundary_status"):
        require(summary.get(key) == "verified", f"unverified {key}")
    require(summary.get("ttft_samples") == 1 and summary.get("tpot_gaps") == tokens - 1,
            "missing token-boundary samples")
    require(re.search(rf"^Inference performance \(1 request, {tokens} tokens\)$", runtime, re.M) is not None,
            "CLI did not print a successful runtime summary")
    require("Inference performance: n/a" not in runtime, "CLI printed an unavailable runtime summary")
    phases = summary.get("phases", [])
    require(len(phases) == 2 and {p.get("phase") for p in phases} == set(PHASES), "missing/duplicate phases")
    notes = []
    for phase in phases:
        name = phase["phase"]
        require(type(phase.get("operations")) is int and phase["operations"] > 0 and
                phase.get("failed_operations") == 0, f"{name}: missing or failed operations")
        for key in ("elapsed_ns", "cpu_cycles", "thread_cpu_ns"):
            value = phase.get(key)
            require(type(value) is int and value > 0 and phase.get(key + "_reason") is None,
                    f"{name}.{key}: {phase.get(key + '_reason') or value}")
        line = re.search(rf"^  {name}: elapsed=([0-9.]+) ms, CPU cycles=([0-9]+), worker CPU=([0-9.]+) ms, .*failures=0$",
                         runtime, re.M)
        require(line is not None, f"{name}: runtime CPU metrics missing")
        require(int(line[2]) == phase["cpu_cycles"], f"{name}: runtime/replay CPU cycle mismatch")
        for observed, key in ((float(line[1]), "elapsed_ns"), (float(line[3]), "thread_cpu_ns")):
            require(abs(observed - phase[key] / 1e6) <= 0.000501, f"{name}: runtime/replay {key} mismatch")
        if phase.get("cpu_work_wall_ns") is None:
            notes.append(f"{name}.cpu_work_wall: {phase.get('cpu_work_wall_ns_reason')}")
        if phase.get("npu_reason"):
            notes.append(f"{name}.npu: {phase['npu_reason']} (CPU-only matrix)")
    for key, label in (("ttft_ns", "TTFT"), ("tpot_ns", "TPOT")):
        value = summary.get(key)
        require(number(value) and value > 0 and summary.get(key + "_reason") is None, f"invalid {key}")
        match = re.search(rf"{label}=([0-9.]+) ms", runtime)
        require(match is not None and abs(float(match[1]) - value / 1e6) <= 0.000501,
                f"runtime/replay {label} mismatch")
    return notes


def validate_analyzer(report: dict, rows: Path, summary: dict) -> dict[str, int]:
    require(report.get("status") == "ok" and report.get("relationship_errors") == [], "analyzer rejected log relations")
    require(report.get("invalid_cycle_rows") == 0 and report.get("invalid_cycle_reasons") == {}, "invalid cycle samples")
    coverage = report.get("cpu_interval_coverage", {})
    operations = sum(phase["operations"] for phase in summary["phases"])
    require(coverage.get("status") == "verified" and coverage.get("verified_operations") == operations and
            coverage.get("operation_end_records") == operations and
            coverage.get("unverified_sequence_operations") == coverage.get("legacy_operation_ends") == 0,
            "CPU interval cardinality is incomplete")
    require(report.get("interval_rows", 0) > 0, "empty analyzer output")
    phase_rows = Counter()
    total = 0
    with rows.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            total += 1
            if row.get("row_type") == "interval" and row.get("request_id") == 1:
                phase = row.get("phase")
                require(phase in PHASES, "request interval has no valid phase")
                phase_rows[phase] += 1
    require(total == report.get("normalized_rows"), "normalized output count mismatch")
    require(all(phase_rows[p] > 0 for p in PHASES), "normalized intervals missing prefill/decode")
    for view in ("thread", "operator", "task", "stripe"):
        trace = report.get("traces", {}).get(view, {})
        path = Path(trace.get("path", ""))
        require(path.is_file() and path.stat().st_size > 0 and path.parent.resolve() == rows.parent.resolve(),
                f"missing or wrong-destination {view} trace")
        # Existence alone is not evidence: parse each trace separately to bound memory.
        document = json_read(path)
        require(isinstance(document, dict) and document.get("view") == view and
                isinstance(document.get("traceEvents"), list), f"malformed {view} trace")
        placed = 0
        for event in document["traceEvents"]:
            require(isinstance(event, dict), f"malformed {view} trace event")
            if event.get("ph") == "X":
                placed += 1
                require(all(number(event.get(key)) and event[key] >= 0 for key in ("ts", "dur")),
                        f"invalid {view} trace interval")
        require(placed == trace.get("placed_intervals"), f"{view} trace count mismatch")
        del document
    require(report["traces"]["thread"].get("skipped_missing_lane") == 0 and
            report["traces"]["thread"].get("placed_intervals") == report["interval_rows"],
            "host intervals could not all be placed on a thread lane")
    return dict(phase_rows)


def run_case(case: dict, build: Path, config: dict, output: Path, timeout: int) -> None:
    check_inputs(case)
    directory = output / case["name"]
    directory.mkdir()
    case["output"] = str(directory)
    raw = directory / "cycle-log.jsonl"
    generation = config["generation"]
    env = os.environ.copy()
    # Do not inherit another experiment's destination or runtime routing overrides.
    for key in list(env):
        if key.startswith("GGML_GEMMINI_") or key.startswith("GEMMINI_LOG_"):
            env.pop(key)
    # The freshly built backend must not be replaced through loader overrides.
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
                "DYLD_FALLBACK_LIBRARY_PATH", "GGML_BACKEND_PATH"):
        env.pop(key, None)
    env.update(GGML_CPU_CYCLE_LOG="1", GGML_GEMMINI_TELEMETRY_HASH="0",
               OUTPUT_ROOT=str(directory), OUTPUT_DIR=str(directory), EXPERIMENT_DIR=str(directory),
               LOG_DIR=str(directory), GEMMINI_LOG_DIR=str(directory),
               GGML_GEMMINI_CYCLE_DETAIL_LOG=str(directory / "exsia-cycle-detail.jsonl"))
    command = [str(build / "bin/llama-cli"), "-m", case["model_path"], "-f", case["prompt"],
               "-n", str(generation["tokens"]), "-t", str(generation["threads"]),
               "-tb", str(generation["threads"]), "-c", str(generation["context"]),
               "-b", str(generation["batch"]), "--seed", str(generation["seed"]),
               "--temp", "0", "--ignore-eos", "--no-conversation", "--no-display-prompt",
               "--gemmini-cycle-log", str(raw)]
    execute(command, directory, "inference", env, timeout, cwd=directory)
    case["inference_ok"] = True
    require(raw.is_file() and raw.stat().st_size > 0, "no fresh cycle-log.jsonl")
    raw_digest = file_hash(raw)
    summary_command = [str(build / "bin/llama-cycle-summary")]
    execute(summary_command + [str(raw)], directory, "summary", env, timeout)
    execute(summary_command + ["--json", str(raw)], directory, "summary-json", env, timeout)
    summary = json_read(directory / "summary-json.stdout.txt")
    json_write(directory / "summary.json", summary)
    # Run the analyzer even if summary metrics are unavailable, to retain both diagnostics.
    rows = directory / "cycle-log.timeline.jsonl"
    errors = []
    try:
        case["notes"] = validate_summary(summary, generation["tokens"],
                                         (directory / "inference.stderr.txt").read_text(encoding="utf-8", errors="replace"))
        case["summary_ok"] = True
    except CheckError as error:
        errors.append(f"summary: {error}")
    try:
        execute([sys.executable, str(ROOT / "scripts/utils/cycle_timeline.py"), str(raw),
                 "--rows", str(rows), "--trace-dir", str(directory), "--all-views"],
                directory, "analyzer", env, timeout)
        report = json_read(directory / "analyzer.stdout.txt")
        json_write(directory / "analyzer.json", report)
        case["phase_rows"] = validate_analyzer(report, rows, summary)
        case["analyzer_ok"] = True
        case["stripe_coverage"] = report.get("stripe_coverage")
    except CheckError as error:
        errors.append(f"analyzer: {error}")
    case["summary"] = summary
    check_inputs(case)
    require(file_hash(raw) == raw_digest, "raw log changed during summary/analyzer checks")
    case["raw_sha256"] = raw_digest
    require(not errors, "; ".join(errors))
    case["artifacts_ok"] = True


def save_results(output: Path, report: dict) -> None:
    counts = Counter(case["status"] for case in report["cases"])
    report["counts"] = dict(counts)
    report["status"] = ("running" if counts.get("pending") else
                        "ok" if counts.get("ok") == len(report["cases"]) else "failed")
    json_write(output / "results.json", report)
    columns = ("name", "bits", "family", "status", *CASE_GATES, "errors", "output")
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for case in report["cases"]:
            writer.writerow({**case, "errors": "; ".join(case["errors"])})


def finish_run(output: Path, report: dict) -> tuple[str, int]:
    # Never leave an earlier success receipt next to newly failed results.
    marker = output / "OK.txt"
    marker.unlink(missing_ok=True)
    audit_errors = []
    try:
        require(source_snapshot() == report["source"], "source changed during the matrix run")
        widths = {case["bits"] for case in report["cases"]}
        builds = report["builds"]
        require(len(builds) == len(widths) and {build["bits"] for build in builds} == widths,
                "not every selected width has a fresh successful build")
        for build in builds:
            require(build.get("fresh") is True and build["detail"] == report["detail"],
                    "build receipt does not match the run")
            path, logs = Path(build["path"]), Path(build["logs"])
            require(path.parent.name == output.name and logs.parent == output,
                    "build belongs to a different run")
            for step in ("configure", "build"):
                result = json_read(logs / f"{step}.result.json")
                require(result.get("success") is True and result.get("exit_code") == 0,
                        f"unsuccessful {step}: {logs}")
            require(build_artifacts(path) == build["artifacts"], "build artifacts changed after compilation")
    except (CheckError, OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
        audit_errors.append(str(error))
    for case in report["cases"]:
        if case["status"] == "ok":
            missing = [gate for gate in CASE_GATES if case.get(gate) is not True]
            problems = audit_errors + (["missing gates: " + ", ".join(missing)] if missing else [])
            if case["errors"] or problems:
                case["status"] = "failed"
                case["errors"].extend(problems)
    report["audit_errors"] = audit_errors
    total = len(report["cases"])
    expected = matrix_keys(json_read(DEFAULT_CONFIG))
    observed = {(case["model"], case["bits"], case["family"]) for case in report["cases"]}
    full = observed == expected and total == len(expected)
    report["scope"] = "full" if full else "selection"
    save_results(output, report)
    passed = report["counts"].get("ok", 0)
    complete = total > 0 and passed == total and not audit_errors
    if complete and full:
        label = "CYCLE_MATRIX_OK"
    elif complete:
        label = "CYCLE_MATRIX_SELECTION_PASSED"
    else:
        label = "CYCLE_MATRIX_FAILED"
    line = (f"{label} cases={passed}/{total} builds={len(report['builds'])}/{len({c['bits'] for c in report['cases']})} "
            f"fresh=1 detail={report['detail']} tokens={report['config']['generation']['tokens']} "
            f"commit={report['source']['commit'][:12]} source={report['source']['sha256'][:16]} "
            f"run={output.name} report_sha256={file_hash(output / 'results.json')}")
    if complete and full:
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(line + "\n")
    return line, 0 if complete else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--base-build", default="build-arm64-cpu")
    parser.add_argument("--build-root", default=".cache/cycle-matrix-build")
    parser.add_argument("--output-root", default="output/cycle-matrix")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--bits", nargs="+", type=int)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--detail", type=int, choices=(0, 1), default=0)
    parser.add_argument("--jobs", type=positive_int, default=2)
    parser.add_argument("--timeout", type=positive_int, default=1800, help="timeout per subprocess, seconds")
    parser.add_argument("--plan", action="store_true", help="resolve the matrix and check files without building/running")
    args = parser.parse_args()
    config = json_read(absolute(args.config))
    generation = config["generation"]
    for key in ("tokens", "threads", "context", "batch"):
        require(type(generation[key]) is int and generation[key] > 0, f"invalid generation.{key}")
    require(generation["tokens"] >= 2, "at least two output tokens are needed to validate decode and TPOT")
    require(type(generation["seed"]) is int and generation["seed"] >= 0, "invalid seed")
    cases = input_cases(config, absolute(args.model_root), args.models, args.bits, args.families)
    expected_cases = matrix_keys(json_read(DEFAULT_CONFIG))
    full_count = len(expected_cases)
    actual_cases = {(case["model"], case["bits"], case["family"]) for case in cases}
    scope = "full" if actual_cases == expected_cases and len(cases) == full_count else "selection"
    for case in cases:
        print(f"{'MISSING' if case['errors'] else 'READY'} {case['name']} " +
              ("; ".join(case["errors"]) or case["model_path"]), flush=True)
    if args.plan:
        print(json.dumps({"status": "planned", "scope": scope, "cases": len(cases),
                          "input_failures": sum(bool(case["errors"]) for case in cases)}))
        return 2 if any(case["errors"] for case in cases) else 0
    base = cache_read(absolute(args.base_build))
    build_root = absolute(args.build_root)
    require(build_root != absolute(args.base_build) and absolute(args.base_build) not in build_root.parents,
            "matrix build root must not be inside the base build")
    build_root.mkdir(parents=True, exist_ok=True)
    lock = build_root / ".matrix.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()) + "\n")
    except FileExistsError as error:
        raise CheckError(f"another matrix run owns {lock}; check its PID before removing a stale lock") from error
    try:
        output = absolute(args.output_root) / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        output.mkdir(parents=True)
        source = source_snapshot()
        report = {"scope": scope, "configured_cases": full_count, "platform": platform.platform(),
                  "git_commit": source["commit"], "source": source, "detail": args.detail,
                  "base_build": str(absolute(args.base_build)), "build_policy": "fresh_run_directory",
                  "builds": [], "config": config, "cases": cases}
        patch = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
        (output / "source.patch").write_bytes(patch)
        save_results(output, report)
        try:
            for width in dict.fromkeys(case["bits"] for case in cases):
                group = [case for case in cases if case["bits"] == width and not case["errors"]]
                if not group:
                    continue
                logs = output / f"build-a{width}-w{width}"
                logs.mkdir()
                try:
                    print(f"BUILD A{width}/W{width}", flush=True)
                    build = prepare_build(args, base, width, logs)
                    report["builds"].append(json_read(logs / "build.json"))
                    for case in group:
                        case["build_ok"] = True
                except (CheckError, OSError, subprocess.SubprocessError) as error:
                    for case in group:
                        case.update(status="failed", errors=[f"build: {error}"])
                    save_results(output, report)
                    continue
                for case in group:
                    print(f"RUN {case['name']}", flush=True)
                    try:
                        run_case(case, build, config, output, args.timeout)
                        case["status"] = "ok"
                    except (CheckError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                        case.update(status="failed", errors=[str(error)])
                    label = "CASE_PASSED" if case["status"] == "ok" else "CASE_FAILED"
                    print(f"{label} {case['name']} {'; '.join(case['errors'])}", flush=True)
                    save_results(output, report)
        except KeyboardInterrupt:
            for case in cases:
                if case["status"] == "pending":
                    case.update(status="failed", errors=["interrupted before completion"])
            save_results(output, report)
            print(f"CYCLE_MATRIX_INTERRUPTED results={output / 'results.json'}", file=sys.stderr)
            return 130
        line, code = finish_run(output, report)
        print(f"Results: {output / 'results.json'}", flush=True)
        print(line, flush=True)
        return code
    finally:
        lock.unlink()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CheckError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"CYCLE_MATRIX_FAILED error={error}", file=sys.stderr)
        sys.exit(2)
