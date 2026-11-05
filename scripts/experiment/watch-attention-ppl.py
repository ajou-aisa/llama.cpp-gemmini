#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run: uv run scripts/experiment/watch-attention-ppl.py [--once]
"""Display all eight attention jobs and every verified historical full-corpus PPL."""
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
import math
import re
import sys
import time
from typing import Final, TypedDict

attention = import_module("attention-ppl")
cuda_attention = import_module("cuda-attention-ppl")
legacy = import_module("cuda-ppl-queue")
monitor = import_module("watch-cuda-ppl")
ROOT: Final = Path(__file__).resolve().parents[2]
OUTPUT: Final = Path("/home/mojoi/potal-results/attention-top3-top2-cuda-remaining-7")
OLD: Final = Path("/home/mojoi/potal-results/attention-top3-top2-full-8")
HISTORY: Final = Path("/home/mojoi/potal-results/rtx3070-full-24")
CONFIG: Final = Path(__file__).with_name("attention-ppl-jobs.cuda-remaining.json")
CPU_ID: Final = "qwen-top3-n4-both"


class Job(TypedDict):
    id: str
    bits: int
    full_chunks: int


@dataclass(frozen=True, slots=True)
class Row:
    name: str
    bits: int
    backend: str
    state: str = "WAIT"
    count: str = "-"
    ppl: str = "-"
    elapsed: str = "--:--:--"
    eta: str = "--:--:--"


def snapshot(job: Job, output: Path, cuda: bool) -> Row:
    """Verify completed evidence and retain live progress without changing a queue."""
    directory = output / job["id"]
    full = directory / "full"
    result = monitor.read(full / "result.json")
    verify = cuda_attention.verify if cuda else attention.verify
    backend = "CUDA" if cuda else "CPU"
    if result:
        try:
            checked = verify(full / "ppl.log", job, job["full_chunks"])
            legacy.require(all(result.get(k) == v for k, v in checked.items()), "Result changed")
            legacy.require(monitor.read(full / "manifest.json").get("job") == job, "Job changed")
            legacy.require((full / "exit-status.txt").read_text().strip() == "0", "Run failed")
            return Row(job["id"], job["bits"], backend, "DONE",
                       f'{result["chunks"]}/{job["full_chunks"]}', f'{result["ppl"]:.6f}',
                       monitor.duration(result["seconds"]))
        except (OSError, ValueError, KeyError, RuntimeError):
            return Row(job["id"], job["bits"], backend, "INVALID")
    status = monitor.read(output / "status.json")
    if status.get("job") == job["id"] and status.get("state") == "failed":
        return Row(job["id"], job["bits"], backend, "FAILED")
    if status.get("job") == job["id"] and status.get("state") == "running":
        phase = status["phase"]
        execution = monitor.read(directory / phase / "execution.json")
        running = bool(execution and legacy.alive(execution))
        state = ("SMOKE" if phase == "smoke" else "RUNNING") if running else "STOPPED"
        seconds = monitor.active_seconds(status, execution)
        try:
            text = (directory / phase / "ppl.log").read_text(errors="replace")
        except OSError:
            text = ""
        chunks = re.findall(r"\bPPL_CHUNK ([^\r\n]*)", text)
        entries: list[tuple[float, int]] = []
        for chunk in chunks:
            try:
                fields = dict(field.split("=", 1) for field in chunk.split())
                nll, tokens = float(fields["nll"]), int(fields["scored_tokens"])
                if math.isfinite(nll) and tokens > 0:
                    entries.append((nll, tokens))
            except (ValueError, KeyError):
                continue
        completed, ppl = 0, "-"
        if entries:
            nll = sum(c[0] for c in entries)
            tokens = sum(c[1] for c in entries)
            if tokens > 0 and math.isfinite(nll):
                completed, ppl = len(entries), f"{math.exp(nll/tokens):.6f}*"
        else:
            rows = re.findall(r"^\s*(\d+)\s+([\d.eE+-]+)\s+[\d.eE+-]+\s+[\d.eE+-]+\s*$", text, re.M)
            if rows:
                completed, ppl = int(rows[-1][0])//512+1, rows[-1][1]+"*"
        total = 1 if phase == "smoke" else job["full_chunks"]
        count = f"{completed}/{total}" if completed else "-"
        eta = monitor.duration(seconds/completed*max(0, total-completed)) if completed and running else "--:--:--"
        return Row(job["id"], job["bits"], backend, state, count, ppl, monitor.duration(seconds), eta)
    return Row(job["id"], job["bits"], backend)


def attention_rows() -> list[Row]:
    """Include the retained CPU result in the original eight-job order."""
    original = monitor.read(CONFIG.with_name("attention-ppl-jobs.rtx3070.json"))["jobs"]
    current = next(job for job in original if job["id"] == CPU_ID)
    remaining = monitor.read(CONFIG)["jobs"]
    return [snapshot(current, OLD, False), *(snapshot(job, OUTPUT, True) for job in remaining)]


def render_attention(rows: list[Row]) -> list[str]:
    """Render both backends in one table and count all eight full runs."""
    lines = [f'{"Experiment":<26} {"n":>2} {"QK/PV":<5} {"Status":<9} {"Chunks":>9} {"PPL":>12} {"Elapsed":>9} {"ETA~":>9}']
    lines += [f"{r.name:<26} {r.bits:>2} {r.backend:<5} {r.state:<9} {r.count:>9} {r.ppl:>12} {r.elapsed:>9} {r.eta:>9}"
              for r in rows]
    lines.append(f"Completed attention: {sum(r.state == 'DONE' for r in rows)}/{len(rows)} full runs.")
    return lines


def historical_lines() -> list[str]:
    """Revalidate history when its files change, keeping refreshes cheap during inference."""
    config_path = ROOT / "scripts/experiment/cuda-ppl-jobs.rtx3070.json"
    jobs = monitor.read(config_path)["jobs"]
    directories = [HISTORY / job["id"] / "full" for job in jobs]
    directories += [HISTORY / folder / model for folder in ("nvfp4-public-fp32", "nvfp4-llama-policy")
                    for model in ("qwen", "llama")]
    paths = [config_path, *(directory / name for directory in directories
                           for name in ("result.json", "ppl.log", "manifest.json", "exit-status.txt"))]
    signature: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            signature.append((str(path), -1, -1))
    return list(verified_history(tuple(signature)))


@lru_cache(maxsize=1)
def verified_history(_signature: tuple[tuple[str, int, int], ...]) -> tuple[str, ...]:
    """Load all 28 prior full runs with their original proof verifiers."""
    config = monitor.read(ROOT / "scripts/experiment/cuda-ppl-jobs.rtx3070.json")
    jobs = {job["id"]: job for job in config["jobs"]}
    verified_count = 0

    def cell(name: str) -> str:
        nonlocal verified_count
        job = jobs[name]
        directory = HISTORY / name / "full"
        result = monitor.read(directory / "result.json")
        if not result:
            return "-"
        try:
            checked = legacy.verify(directory / "ppl.log", job, job["full_chunks"])
            legacy.require(all(result.get(k) == v for k, v in checked.items()), "Historical result changed")
            legacy.require(monitor.read(directory / "manifest.json").get("job") == job, "Historical job changed")
            legacy.require((directory / "exit-status.txt").read_text().strip() == "0", "Historical run failed")
            verified_count += 1
            return f'{result["ppl"]:.6f}'
        except (OSError, ValueError, KeyError, RuntimeError):
            return "INVALID"

    def pair(label: str, qwen: str, llama: str) -> str:
        return f"{label:<27} {qwen:^25} {llama:^25}"

    lines = [pair("FP16 W16/A16", cell("qwen-fp16-w16a16"), cell("llama-fp16-w16a16"))]
    for label, stem, tail in (("RTN-W", "rtn", "16"), ("RTN-WA", "rtn", ""),
                              ("MXINT-W + RTN-A", "rtn-mx", ""), ("PoTal TOP2 linear", "potal-top2", ""),
                              ("PoTal TOP3 linear", "potal-top3", "")):
        values = [cell(f"{model}-{stem}-w{bits}a{tail or bits}")
                  for model in ("qwen", "llama") for bits in (4, 8)]
        lines.append(f'{label:<27} ' + " ".join(f"{value:>12}" for value in values))
    lines.append(pair("NVFP4 ref E5M10 (old)", cell("qwen-nvfp4-w4a4"), cell("llama-nvfp4-w4a4")))
    for public, label in ((True, "NVFP4 public W4/A4"), (False, "NVFP4 llama W4/A8")):
        values = []
        folder = "nvfp4-public-fp32" if public else "nvfp4-llama-policy"
        for model in ("Qwen", "Llama"):
            result = monitor.read(HISTORY / folder / model.lower() / "result.json")
            if not result:
                values.append("-")
                continue
            try:
                checked = monitor.verify_upstream_result(HISTORY, model, result["chunks"], public)
                legacy.require(result["ppl"] == checked["ppl"], "NVFP4 result changed")
                verified_count += 1
                values.append(f'{checked["ppl"]:.6f}')
            except (OSError, ValueError, KeyError, RuntimeError):
                values.append("INVALID")
        lines.append(pair(label, values[0], values[1]))
    header = f'{"Method (original policies)":<27} {"Qwen n=4":>12} {"Qwen n=8":>12} {"Llama n=4":>12} {"Llama n=8":>12}'
    return (f"Previous full-corpus PPL: {verified_count}/28 verified.", header, *lines)


def show() -> None:
    """Show final and live results together, without restarting inference."""
    status = monitor.read(OUTPUT / "status.json")
    print("PoTal QK+PV | full WikiText-2 | * = partial PPL, no * = final PPL")
    print("Queue:", status.get("state", "waiting for current CPU run"), status.get("job", ""), status.get("phase", ""))
    if status.get("error"):
        print("ERROR:", status["error"])
    print("\n".join(render_attention(attention_rows())))
    print("\n".join(historical_lines()))
    print("Ctrl+C closes this monitor only.")


def main() -> None:
    """Refresh the read-only screen, including the one-shot CLI mode."""
    once = "--once" in sys.argv
    try:
        while True:
            if not once:
                print("\033[2J\033[H", end="")
            show()
            if once:
                return
            time.sleep(5)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
