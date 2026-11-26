"""Native CPU timing and PMU profiling from measured host logs."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Final

from campaign_build import platform_profile
from eval_common import Json, Record, record, require, sha256

EVAL: Final = Path(__file__).resolve().parent
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
