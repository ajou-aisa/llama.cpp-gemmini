#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# Run: python3 scripts/experiment/run-metal-ablation.py --after OLD_QUEUE --output NEW_QUEUE
"""Run remaining full PPL cases after the predecessor queue terminates."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
HEADER: Final = "method\tmodel\tbits\tdim\tchunks\tscored_tokens\tppl\tprocess_seconds\texit\n"

if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required; use /opt/homebrew/bin/python3")


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    method: str
    family: str
    bits: int
    dim: int

    @property
    def model(self) -> str:
        family = "gpt2" if self.family == "gpt2" else "llama3.2-1B"
        quant = "HP1" if self.method in ("RTN-WA2", "PoTal") else "0"
        return f"{family}.Q{self.bits}_{quant}.gguf"


def cases() -> tuple[Case, ...]:
    result: list[Case] = []
    for method, prefix, dims in (("RTN-W", "rtnw", (16,)), ("RTN-WA1", "rtnwa1-n", (16,)),
                                 ("RTN-WA2", "rtnwa2-n", (16, 32, 64)), ("PoTal", "potal", (16, 32, 64))):
        for family in ("gpt2", "llama"):
            for bits in (4, 8):
                for dim in dims:
                    suffix = f"-d{dim}" if len(dims) > 1 else ""
                    result.append(Case(f"{family}-{prefix}{bits}{suffix}", method, family, bits, dim))
    return tuple(result)


def predecessor_finished(directory: Path) -> bool:
    """A vanished process alone is not a completion signal."""
    expected = set((directory / "queue.txt").read_text().splitlines())
    states: dict[str, str] = {}
    with (directory / "status.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            states[row["case"]] = row["state"]
    terminal = bool(expected) and all(states.get(name) in ("complete", "failed") for name in expected)
    if not terminal:
        pid = int((directory / "runner.pid").read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError as error:
            raise SystemExit(f"Predecessor interrupted, not completed: {directory}") from error
    return terminal


def result_row(directory: Path, case: Case, *, allow_cpu_attention: bool = False) -> str | None:
    """Reuse only a complete result with matching model, corpus and arithmetic."""
    if not (directory / "result.tsv").is_file():
        return None
    try:
        fields = (directory / "result.tsv").read_text().strip().split("\t")
        proof = json.loads((directory / "metal-cpu-exact-proof.json").read_text())
        command = shlex.split((directory / "command.txt").read_text())
        chunks = 559 if case.family == "gpt2" else 564
        expected = [case.method, case.family, str(case.bits), str(case.dim), str(chunks), str(chunks * 255)]
        if len(fields) != 9 or fields[:6] != expected or fields[-1] != "0" or not math.isfinite(float(fields[6])) or float(fields[6]) <= 0:
            return None
        if any(command[command.index(flag) + 1] != value for flag, value in
               (("--chunks", "-1"), ("--ctx-size", "512"), ("--batch-size", "512"), ("--ubatch-size", "512"),
                ("--cache-type-k", "f16"), ("--cache-type-v", "f16"), ("--seed", "42"))):
            return None
        manifest = {Path(name).name: digest for digest, name in
                    (line.split() for line in (ROOT / "scripts/experiment/default-ppl-models.sha256").read_text().splitlines())}
        hashes = {Path(name).name: digest for digest, name in
                  (line.split(maxsplit=1) for line in (directory / "inputs-binaries.sha256").read_text().splitlines())}
        with (ROOT / "wikitext-2-raw/wiki.test.raw").open("rb") as stream:
            corpus = hashlib.file_digest(stream, "sha256").hexdigest()
        if hashes.get(case.model) != manifest[case.model] or hashes.get("wiki.test.raw") != corpus:
            return None
        layers = 12 if case.family == "gpt2" else 16
        activation = "EXSIA" if case.method in ("RTN-WA2", "PoTal") else "BLOCK"
        compute = "FLOAT" if case.method == "RTN-W" else "INT"
        cpu_attention = (allow_cpu_attention and case.method == "RTN-W" and
                         proof.get("version") == 1 and proof.get("graph") == "original_cpu")
        if proof.get("schema") != "metal-cpu-exact-ppl" or (proof.get("graph") != "cpu_equivalent_metal" and not cpu_attention):
            return None
        if (proof.get("bits"), proof.get("dim"), proof.get("compute"), proof.get("activation")) != (case.bits, case.dim, compute, activation):
            return None
        if (proof.get("version") not in (2, 3) and not cpu_attention) or proof.get("complete") is not True or proof.get("scored_tokens") != chunks * 255:
            return None
        matmuls = (layers * (4 if case.family == "gpt2" else 7) + 1) * chunks
        if proof.get("observed_matmuls") != matmuls or proof.get("verified_matmuls") != matmuls:
            return None
        head_calls = chunks if case.bits == 4 and case.method in ("RTN-W", "RTN-WA1") else 0
        if proof.get("gpu_q6_head_matmuls") != head_calls:
            return None
        if case.method == "RTN-W":
            if proof.get("float_gpu_calls") != matmuls or proof.get("integer_gpu_launches") != 0:
                return None
        elif proof.get("integer_gpu_launches", 0) <= 0 or proof.get("float_gpu_calls") != 0:
            return None
        if proof.get("cpu_q6_head_matmuls") != 0:
            return None
        if not cpu_attention and any(proof.get(key) != 2 * layers * chunks for key in
                                     ("attention_gpu_calls", "observed_attention_matmuls", "verified_attention_matmuls")):
            return None
        full = case.method == "PoTal"
        if (proof.get("residual_gpu_launches", 0) > 0) != full or proof.get("activation_fp16") != (case.method == "RTN-W"):
            return None
        if proof.get("version") == 3 and (proof.get("outlier_selection") != full or proof.get("residual_enabled") != full):
            return None
        if case.method in ("RTN-WA1", "RTN-WA2") and (proof.get("version") != 3 or proof.get("outlier_selection") is not False or proof.get("residual_enabled") is not False):
            return None
        if case.method == "RTN-WA1" and case.bits == 4 and proof.get("q6_head_activation_bits") != 4:
            return None
        if (directory / "exit-status.txt").read_text().strip() != "0":
            return None
        return "\t".join(fields) + "\n"
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        print(f"Rejecting reuse {directory}: {error}", flush=True)
        return None


def main() -> int:
    options = iter(sys.argv[1:])
    after: Path | None = None
    output = ROOT / "output/experiment" / time.strftime("metal-ablation-full-%Y%m%d-%H%M%S")
    dry = False
    for option in options:
        if option == "--dry-run":
            dry = True
        elif option in ("--after", "--output"):
            value = next(options, "")
            if not value:
                raise SystemExit(f"Missing value: {option}")
            if option == "--after":
                after = Path(value).resolve()
            else:
                output = Path(value).resolve()
        else:
            raise SystemExit(f"Unknown option: {option}")
    matrix = cases()
    reused: dict[str, Path] = {}
    reuse_manifest = output / "reuse.tsv"
    if reuse_manifest.exists():
        with reuse_manifest.open() as stream:
            for entry in csv.DictReader(stream, delimiter="\t"):
                case = next((case for case in matrix if case.name == entry["case"]), None)
                if case is None or case.name in reused:
                    raise SystemExit(f"Unknown or duplicate reused case: {entry['case']}")
                source = Path(entry["source"]).resolve()
                if result_row(source, case, allow_cpu_attention=True) is None:
                    raise SystemExit(f"Invalid reused result: {source}")
                reused[case.name] = source
    predecessor_cases = set((after / "queue.txt").read_text().splitlines()) if after else set()
    excluded = {case.name for case in matrix if case.name in predecessor_cases} | reused.keys()
    queued = tuple(case for case in matrix if case.name not in excluded)
    if dry:
        for case in queued:
            print(f"{case.name}\t{case.model}\tselection={int(case.method == 'PoTal')}\tRC={int(case.method == 'PoTal')}")
        print(f"{len(queued)} full-corpus cases; {len(excluded)} existing cases excluded; predecessor={after}; output={output}")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f"Queue already active: {output}") from error
        (output / "runner.pid").write_text(f"{os.getpid()}\n")
        (output / "queue.txt").write_text("".join(case.name + "\n" for case in queued))
        (output / "excluded-cases.txt").write_text("".join(case.name + "\n" for case in matrix if case.name in excluded))
        print(f"Queued {len(queued)} full PPL cases; excluded {len(excluded)} existing cases from execution", flush=True)
        state = output / "state.txt"
        state.write_text("waiting\n" if after else "running\n")
        if after:
            print(f"Waiting for full predecessor queue: {after}", flush=True)
            try:
                while not predecessor_finished(after):
                    time.sleep(30)
            except (SystemExit, OSError, ValueError):
                state.write_text("predecessor-interrupted\n")
                raise
        state.write_text("running\n")
        completed: dict[str, Path] = {}
        status = output / "status.tsv"
        if status.exists():
            with status.open() as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    if row["state"] in ("complete", "reused"):
                        completed[row["case"]] = Path(row["source"])
        else:
            status.write_text("case\tstate\tsource\n")
        results: list[str] = []
        prepared: set[str] = set()
        failed = False
        runner = ROOT / "scripts/experiment/run-metal-quality-ppl.sh"
        for case in matrix:
            candidate = reused.get(case.name, completed.get(case.name, (after / case.name / case.name) if after else output / "absent"))
            row = result_row(candidate, case, allow_cpu_attention=True) if case.name in reused else result_row(candidate, case)
            case_state = "reused"
            if row is None and case.name in excluded:
                case_state = "excluded-unavailable"
            elif row is None:
                with status.open("a") as stream:
                    stream.write(f"{case.name}\trunning\t\n")
                profile = case.name.split("-", 1)[1]
                common = ["rtk", "proxy", "bash", str(runner), "--case", case.name]
                code = 0
                if profile not in prepared:
                    print(f"Preparing {case.name}", flush=True)
                    with (output / f"{profile}-build.log").open("w") as log:
                        code = subprocess.run([*common, "--prepare"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
                    if code == 0:
                        prepared.add(profile)
                attempt = output / case.name / time.strftime("attempt-%Y%m%d-%H%M%S")
                if code == 0:
                    print(f"Running full PPL {case.name}: {attempt}", flush=True)
                    code = subprocess.run([*common, "--context", "512", "--chunks", "-1", "--output", str(attempt)], cwd=ROOT, check=False).returncode
                candidate = attempt / case.name
                row = result_row(candidate, case) if code == 0 else None
                case_state = "complete" if row else "failed"
            with status.open("a") as stream:
                stream.write(f"{case.name}\t{case_state}\t{candidate}\n")
            if row:
                results.append(row)
                print(f"{case_state}: {case.name}", flush=True)
            else:
                failed = True
                if case.name in excluded:
                    print(f"EXCLUDED: {case.name}; no valid full result, no rerun", flush=True)
                else:
                    print(f"FAILED: {case.name}; proceeding to next case", flush=True)
            (output / "results.tsv").write_text(HEADER + "".join(results))
            subprocess.run(["rtk", "proxy", sys.executable, str(ROOT / "scripts/experiment/summarize-metal-quality-ppl.py"), str(output)], check=True)
        state.write_text("failed\n" if failed else "complete\n")
        print(f"Finished {len(results)}/{len(matrix)}: {output / 'results.tsv'}", flush=True)
        return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
