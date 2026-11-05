#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "polars", "pydantic", "matplotlib"]
# ///
# How to run: .\scripts\experiment\watch-ties-away.ps1 [-Once]
"""Read-only full experiment report, with current activity at the bottom."""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import json
import re
import sys
import textwrap
import time
from typing import Final

driver = import_module("ties-away-ppl")
monitor = import_module("watch-cuda-ppl")
past = import_module("watch-attention-ppl")
accuracy = import_module("watch-accuracy-summary")
ROOT: Final = driver.ROOT
OUTPUT: Final = Path("/home/mojoi/potal-results/ties-away-full-q4")
CONFIG: Final = Path(__file__).with_name("ties-away-full-q4-jobs.json")
FAMILIES: Final = ("qwen2-0.5B", "llama3.2-1B")


@dataclass(frozen=True, slots=True)
class Cell:
    value: str
    error: str = ""


def final_cell(job: driver.runtime.Job, output: Path) -> Cell:
    """Only display final PPL after checking its original proof and manifest."""
    full = output / job["id"] / "full"
    result = monitor.read(full / "result.json")
    if not result:
        return Cell("--")
    try:
        paths = tuple(full / name for name in ("ppl.log", "result.json", "manifest.json", "exit-status.txt"))
        signature = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
        checked = driver.verified_for_watch(str(paths[0]), json.dumps(job, sort_keys=True),
                                           job["full_chunks"], signature)
        driver.queue.require(all(result.get(key) == value for key, value in checked.items()) and
                             paths[3].read_text().strip() == "0" and
                             monitor.read(paths[2]).get("job") == job, "Stored result changed")
        return Cell(f'{result["ppl"]:.6f}')
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return Cell("INVALID", f'{job["id"]}: {error}')


def active_lines(jobs: list[driver.runtime.Job]) -> list[str]:
    """Keep cancelled archive runs out of the live queue and label partial PPL."""
    state = monitor.read(OUTPUT / "status.json")
    cells = {job["id"]: final_cell(job, OUTPUT) for job in jobs}
    completed = sum(cell.value not in ("--", "INVALID") for cell in cells.values())
    lines = ["PoTal n=4 FULL (QK+PV) | WikiText-2 test",
             f'Queue: {state.get("state", "waiting")} | verified {completed}/{len(jobs)}']
    current = next((job for job in jobs if job["id"] == state.get("job")), None)
    if current and cells[current["id"]].value == "--":
        phase = state.get("phase", "full")
        directory = OUTPUT / current["id"] / phase
        execution = monitor.read(directory / "execution.json")
        running = bool(execution and driver.queue.alive(execution))
        labels = {"checking_inputs": "PREP", "building": "BUILD", "failed": "FAILED"}
        label = labels.get(state.get("state"), "RUN" if running else "STOPPED")
        cells[current["id"]] = Cell(label)
        text = ""
        log = directory / "ppl.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 65536))
                text = stream.read().decode(errors="replace")
        progress = re.findall(r"^\s*(\d+)\s+([\d.eE+-]+)\s+[\d.eE+-]+\s+[\d.eE+-]+\s*$", text, re.M)
        count = int(progress[-1][0]) // 512 + 1 if progress else 0
        if progress and phase == "full" and running:
            cells[current["id"]] = Cell(progress[-1][1] + "*")
        total = 1 if phase == "smoke" else current["full_chunks"]
        elapsed = monitor.duration(monitor.active_seconds(state, execution)) if execution else "--:--:--"
        family = {"qwen2-0.5B": "Qwen2-0.5B", "llama3.2-1B": "Llama3.2-1B"}[current["family"]]
        lines += [f'Current: {family} / TOP{current["anchor"]} FULL [{label}]',
                  f"Chunks {count}/{total} | elapsed {elapsed}"]
        command = monitor.read(directory / "manifest.json").get("command", [])
        threads = command[command.index("--threads") + 1] if "--threads" in command else "--"
        lines.append(f"CPU workers {threads} | CUDA | * = partial PPL")
    if state.get("error"):
        lines += textwrap.wrap("Error: " + str(state["error"]), width=76)
    lines.append(f'{"Anchor":<17}{"Qwen2-0.5B":>15}{"Llama3.2-1B":>15}')
    for anchor in (2, 3):
        values = [cells[next(job["id"] for job in jobs if job["family"] == family and
                             job["anchor"] == anchor)].value for family in FAMILIES]
        lines.append(f'{f"TOP{anchor} FULL":<17}' + "".join(f"{value:>15}" for value in values))
    for cell in cells.values():
        if cell.error:
            lines += textwrap.wrap("INVALID " + cell.error, width=76)
    return lines


def ppl_lines(jobs: list[driver.runtime.Job]) -> list[str]:
    """Keep current policies, cancelled archives and original policies distinct."""
    archived = driver.queue.read(driver.CONFIG)["jobs"]
    current = {job["id"]: job for job in jobs}
    cells = {}
    for job in archived:
        cell = final_cell(current.get(job["id"], job), OUTPUT if job["id"] in current else driver.OUTPUT)
        cells[job["id"]] = cell.value
    header = f'{"Method":<26}' + "".join(f'{name:>12}' for name in ("Qwen n4", "Qwen n8", "Llama n4", "Llama n8"))
    lines = ["LATEST PPL | native RTN; ties-away MXINT / PoTal | WikiText-2 test", header]
    groups = [("Native RTN-WA", "rtn", 0, "none"), ("True MXINT-WA", "rtn-mx", 0, "none"),
              ("PoTal TOP2 FULL", "potal", 2, "both"), ("PoTal TOP3 FULL", "potal", 3, "both")]
    for label, kind, anchor, scope in groups:
        values = []
        for family in FAMILIES:
            for bits in (4, 8):
                job = next(job for job in archived if job["kind"] == kind and job["family"] == family and
                           job["bits"] == bits and (kind != "potal" or
                           (job["anchor"] == anchor and job["attention_scope"] == scope)))
                value = cells[job["id"]]
                values.append("NOT RUN" if value == "--" else value)
        lines.append(f'{label:<26}' + "".join(f'{value:>12}' for value in values))
    verified = sum(value not in ("--", "INVALID") for value in cells.values())
    lines += [f"Latest verified: {verified} full-corpus results; unmeasured slots NOT queued.",
              "Numbers = verified final PPL. Latest PoTal n8: smoke only; NOT queued.",
              "FULL = LINEAR + QK + PV. Archived LINEAR below is no longer queued.",
              "", "LATEST LINEAR HISTORY | ties-away; CANCEL = interrupted", header]
    archive_state = monitor.read(driver.OUTPUT / "status.json")
    for anchor in (2, 3):
        values = []
        for model in ("qwen", "llama"):
            for bits in (4, 8):
                name = f"{model}-potal-top{anchor}-w{bits}a{bits}-linear"
                value = cells[name]
                if value == "--":
                    value = "CANCEL" if name == archive_state.get("job") else "NOT RUN"
                values.append(value)
        lines.append(f'{f"PoTal TOP{anchor} LINEAR":<26}' + "".join(f'{value:>12}' for value in values))
    lines += ["", "EARLIER PPL HISTORY | original policies (PoTal RNE)", *past.historical_lines(),
              "", "EARLIER FULL HISTORY | original PoTal RNE; separate from latest", header]
    previous = past.attention_rows()
    for anchor in (2, 3):
        selected = [next(row for row in previous if row.name == f'{model}-top{anchor}-n{bits}-both')
                    for model in ("qwen", "llama") for bits in (4, 8)]
        values = [row.ppl if row.state == "DONE" else row.state for row in selected]
        lines.append(f'{f"PoTal TOP{anchor} FULL (old)":<26}' + "".join(f'{value:>12}' for value in values))
    lines.append(f'Historical FULL verified: {sum(row.state == "DONE" for row in previous)}/8')
    return lines


def show() -> None:
    """Print all results into scrollback, finishing with the current live activity."""
    if not CONFIG.exists():
        print("FULL n=4 queue is being prepared. Refresh in 5 seconds.")
        return
    jobs = driver.queue.read(CONFIG)["jobs"]
    fp, quant = accuracy.accuracy_summary(False), accuracy.accuracy_summary(True)
    print("\n".join([*ppl_lines(jobs), "", *fp.lines, "", *quant.lines]))
    print("\n".join(accuracy.LEGEND))
    print("\nCURRENT | " + time.strftime("%H:%M:%S") + " | refresh 5s")
    live = active_lines(jobs)
    print("\n".join(live[:next(i for i, line in enumerate(live) if line.startswith("Anchor"))]))
    sensitivity = monitor.read(driver.runtime.OUTPUT / "sensitivity-status.json")
    quant_status = monitor.read(accuracy.BASE / "quantized-accuracy-4/status.json")
    if quant_status.get("state") == "cancelled" and sensitivity.get("state") == "cancelled":
        print("Next: FP16 only; quantized ACC and sensitivity cancelled.")
    else:
        print("Next: Four-task quantized ACC queue; sensitivity remains cancelled.")
    print("\n".join((*fp.current, *quant.current)))
    print("Sensitivity:", sensitivity.get("state", "queued after ACC; not started"))
    print("Scroll up for all results; -Once freezes snapshot.")
    print("Ctrl+C closes this monitor only.")


def main() -> None:
    """Refresh on demand without launching inference or changing result files."""
    once = "--status" in sys.argv or "--once" in sys.argv
    try:
        if not once:
            print("Loading verified PPL results...", flush=True)
        while True:
            show()
            sys.stdout.flush()
            if once:
                return
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitor closed.")


if __name__ == "__main__":
    main()
