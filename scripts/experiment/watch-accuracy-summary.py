#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: watch-fp-accuracy.ps1 / watch-quantized-accuracy.ps1 [-Once].
"""Read-only ACC tables and current progress for the combined monitor."""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import textwrap
import sys
import time
from typing import Final

monitor = import_module("watch-cuda-ppl")
confidence = import_module("accuracy-confidence")
BASE: Final = Path("/home/mojoi/potal-results")
TASKS: Final = {"piqa": "PIQA", "hellaswag": "HellaSwag", "winogrande": "WinoGrande",
               "arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge",
               "boolq": "BoolQ", "openbookqa": "OpenBookQA"}
VARIANTS: Final = {"potal_top2": "TOP2 FULL", "potal_top3": "TOP3 FULL",
                  "mxint_wa": "MXINT-WA", "nvfp4_public": "NVFP4 public",
                  "nvfp4_llama": "NVFP4 llama", "rtn4": "RTN-WA4", "rtn8": "RTN-WA8",
                  "mxint4": "MXINT-WA4", "mxint8": "MXINT-WA8",
                  "potal_top3_4": "TOP3 FULL4", "potal_top3_8": "TOP3 FULL8"}
NORMALIZED: Final = frozenset(("piqa", "hellaswag", "arc_easy", "arc_challenge", "openbookqa"))
LEGEND: Final = (
    "ACC = raw accuracy; NORM = character-length normalized accuracy.",
    "Values: estimate [low,high] in %; two-sided 95% Wilson CI; N = items.",
    "Numbers = complete cases only; -- = metric not provided.",
    "RUN running | PREP loading | WAIT queued",
    "CANCEL cancelled | FAIL stopped | BAD invalid",
)


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    family: str
    task: str
    variant: str
    rows: int
    state: str
    acc: str
    norm: str

    def cell(self, normalized: bool) -> str:
        return (self.norm if normalized else self.acc) if self.state == "DONE" else self.state


@dataclass(frozen=True, slots=True)
class Summary:
    lines: tuple[str, ...]
    current: tuple[str, ...]


def accuracy_summary(quantized: bool) -> Summary:
    """Render every planned case, deriving completed metrics from its result file."""
    root = BASE / ("quantized-accuracy-4" if quantized else "fp-accuracy-7")
    config = (Path(__file__).with_name("quantized-accuracy-jobs.json")
              if quantized else root / "plan.json")
    jobs = monitor.read(config).get("jobs", [])
    status = monitor.read(root / "status.json")
    cases: list[Case] = []
    for job in jobs:
        case_status = monitor.read(root / job["id"] / "status.json")
        result = monitor.read(root / job["id"] / "result.json")
        state, acc, norm = "WAIT", "--", "--"
        if result:
            metrics = result.get("results", {}).get(job["task"], {})
            samples = result.get("n-samples", {}).get(job["task"], {}).get("effective")
            complete = case_status.get("state") == "complete" and samples == job["rows"]
            if complete and metrics.get("acc,none") is not None:
                try:
                    acc = confidence.format_accuracy(metrics["acc,none"], samples)
                    norm = confidence.format_accuracy(metrics.get("acc_norm,none"), samples)
                    state = "DONE"
                except confidence.ConfidenceError:
                    state, acc, norm = "BAD", "--", "--"
            else:
                state = "BAD"
        elif status.get("state") == "cancelled":
            state = "CANCEL"
        elif status.get("current") == job["id"]:
            state = "FAIL" if status.get("state") == "failed" else (
                "RUN" if case_status.get("state") == "running" else "PREP")
        cases.append(Case(job["id"], job["family"], job["task"], job.get("variant", "FP16"),
                          job["rows"], state, acc, norm))
    label = "Quant ACC" if quantized else "FP16 ACC"
    done = sum(case.state == "DONE" for case in cases)
    heading = f'{label}: {done}/{len(cases)} complete | {str(status.get("state", "not started")).replace("_", " ")}'
    lines = [heading, f'{"Method/task":<24}{"Stat":<5}{"N":>6} '
             f'{"Qwen2-0.5B":^21} {"Llama3.2-1B":^21}']
    groups = tuple(dict.fromkeys((case.variant, case.task) for case in cases))
    for variant, task in groups:
        selected = [next((case for case in cases if case.variant == variant and
                         case.task == task and case.family == family), None)
                    for family in ("qwen", "llama")]
        name = f'{VARIANTS.get(variant, variant)}/{TASKS.get(task, task)}'
        rows = next(case.rows for case in selected if case)
        for normalized in ((False, True) if task in NORMALIZED else (False,)):
            metric = "NORM" if normalized else "ACC"
            row_label = "" if normalized else name
            lines.append(f'{row_label:<24}{metric:<5}{rows:>6} ' + " ".join(
                f'{case.cell(normalized) if case else "NOT PLANNED":>21}' for case in selected))
    current = [heading]
    active = next((case for case in cases if case.id == status.get("current") and case.state != "DONE"), None)
    detail = monitor.read(root / status["current"] / "status.json") if status.get("current") else {}
    if active:
        current.append(f'{active.state}: {active.family.capitalize()} / {VARIANTS.get(active.variant, active.variant)} / {TASKS.get(active.task, active.task)}')
        total = detail.get("total_requests", 0)
        if total:
            count = detail.get("completed_requests", 0)
            elapsed = monitor.duration(detail.get("elapsed_seconds", 0))
            current.append(f'Choices {count:,}/{total:,} ({100 * count / total:.1f}%) | elapsed {elapsed}')
    cpus = sorted(set(detail.get("allowed_cpus", status.get("allowed_cpus", []))))
    cpu_label = ",".join(map(str, cpus)) or "--"
    if len(cpus) > 1 and cpus[-1] - cpus[0] + 1 == len(cpus):
        cpu_label = f"{cpus[0]}-{cpus[-1]}"
    workers = detail.get("threads", status.get("threads", "--"))
    gpu_calls = detail.get("gpu_calls", status.get("gpu_calls", "--"))
    current.extend(textwrap.wrap(
        f"CPU workers {workers} | CPUs {cpu_label} | GPU calls {gpu_calls}", width=76))
    pending = next((case for case in cases if case.state == "WAIT"), None)
    if pending:
        current.append(f'Next {label}: {pending.family.capitalize()} / {VARIANTS.get(pending.variant, pending.variant)} / {TASKS.get(pending.task, pending.task)}')
    if status.get("error"):
        current.extend(textwrap.wrap("ERROR: " + str(status["error"]), width=76))
    if status.get("reason"):
        current.extend(textwrap.wrap("Reason: " + str(status["reason"]), width=76))
    return Summary(tuple(lines), tuple(current))


def main() -> None:
    """Refresh the dedicated read-only view using the same verified result cells."""
    once = "--once" in sys.argv
    quantized = "--quantized" in sys.argv
    try:
        if not once:
            print("Loading accuracy results...", flush=True)
        while True:
            summary = accuracy_summary(quantized)
            print("\n".join((*summary.lines, "", *LEGEND,
                             "FULL = LINEAR + QK + PV", "", *summary.current)))
            print("Refresh: 5s | Scroll up for results; -Once freezes snapshot.")
            print("Ctrl+C closes this monitor only.", flush=True)
            if once:
                return
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitor closed.")


if __name__ == "__main__":
    main()
