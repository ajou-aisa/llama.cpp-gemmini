#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2,<3", "typer>=0.16,<1"]
# ///
# ─── How to run ───
# 1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run: uv run fill_cycles.py RESULTS_DIR CYCLE_RUNNER
# 3. Or make executable: chmod +x fill_cycles.py && ./fill_cycles.py RESULTS_DIR CYCLE_RUNNER
# ──────────────────
"""Estimate scalar device shapes after host measurements have completed."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final

import typer
from pydantic import BaseModel, ConfigDict, Field

WORKERS: Final = 8
FIELDS: Final = ("key", "device_cycles", "fragments", "loads", "stores", "scales")


class Estimate(BaseModel):
    """Validate the scalar cycle runner's result at its output boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    key: str
    device_cycles: int = Field(gt=0)
    fragments: int
    loads: int
    stores: int
    scales: int


@dataclass(frozen=True, slots=True)
class Job:
    """A batch of unique value-free device geometries."""

    executable: Path
    geometries: Path
    results: Path
    log: Path


def run_batch(job: Job) -> Path:
    """Run one estimator process; its elapsed CPU time is never a host measurement."""
    with job.results.open("w") as output, job.log.open("w") as log:
        _ = subprocess.run(
            ["rtk", "proxy", str(job.executable.resolve()), str(job.geometries)],
            stdout=output,
            stderr=log,
            check=True,
        )
    return job.results


def read_estimates(path: Path) -> list[Estimate]:
    """Read flushed complete lines from a finished or interrupted batch."""
    return [
        Estimate.model_validate(dict(zip(FIELDS, line.split(), strict=True)))
        for line in path.read_text().splitlines(keepends=True)
        if line.endswith("\n")
    ]


def load_cache(
    work: Path, cycle_runner: Path, keys: frozenset[str]
) -> tuple[Estimate, ...]:
    """Pin the runner identity and checkpoint completed scalar estimates."""
    identity = work / "runner.sha256"
    signature = hashlib.sha256(cycle_runner.read_bytes()).hexdigest()
    cache = work / "complete-cache.tsv"
    existing = sorted(work.glob("worker-*.output.tsv"))
    if cache.exists():
        existing.append(cache)
    if existing and (
        not identity.exists() or identity.read_text().strip() != signature
    ):
        message = "Existing cycle cache has a different or missing runner identity"
        raise RuntimeError(message)
    results: dict[str, Estimate] = {}
    for path in existing:
        for parsed in read_estimates(path):
            if parsed.key in keys:
                results[parsed.key] = parsed
    _ = identity.write_text(signature + "\n")
    with cache.open("w") as stream:
        for key, value in sorted(results.items()):
            _ = stream.write(
                f"{key} {value.device_cycles} {value.fragments} {value.loads} {value.stores} {value.scales}\n"
            )
    return tuple(results.values())


def main(result_dir: Path, cycle_runner: Path) -> None:
    """Fill CSV device columns with resumable independent estimator processes."""
    sources = sorted(result_dir.glob("*-a[48]-d*.csv.geometry.tsv"))
    unique: dict[str, str] = {}
    for source in sources:
        for line in source.read_text().splitlines():
            key = line.split(maxsplit=1)[0]
            _ = unique.setdefault(key, line)
    jobs: list[Job] = []
    work = result_dir / "cycle-jobs"
    work.mkdir(exist_ok=True)
    results = {
        value.key: value for value in load_cache(work, cycle_runner, frozenset(unique))
    }
    ordered = sorted((key, line) for key, line in unique.items() if key not in results)
    for worker in range(WORKERS):
        prefix = work / f"worker-{worker}"
        geometries = prefix.with_suffix(".input.tsv")
        _ = geometries.write_text(
            "\n".join(line for _, line in ordered[worker::WORKERS]) + "\n"
        )
        jobs.append(
            Job(
                cycle_runner,
                geometries,
                prefix.with_suffix(".output.tsv"),
                prefix.with_suffix(".log"),
            )
        )
    typer.echo(
        f"Estimating {len(ordered)} pending geometries with {WORKERS} workers; reusing {len(results)}"
    )
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for completed in executor.map(run_batch, jobs):
            for parsed in read_estimates(completed):
                results[parsed.key] = parsed
    if set(results) != set(unique):
        message = "Device estimate coverage mismatch"
        raise RuntimeError(message)
    for path in sorted(result_dir.glob("*-a[48]-d*.csv")):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            names = reader.fieldnames
            rows = list(reader)
        replacement = path.with_suffix(".filled.csv")
        if names is None:
            message = f"Missing CSV header: {path}"
            raise RuntimeError(message)
        with replacement.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            for row in rows:
                estimate = results[row["cycle_key"]]
                row.update(
                    device_cycles=str(estimate.device_cycles),
                    fragments=str(estimate.fragments),
                    loads=str(estimate.loads),
                    stores=str(estimate.stores),
                    scales=str(estimate.scales),
                )
                writer.writerow(row)
        _ = replacement.replace(path)
    typer.echo("PASS: every replay row has a complete device estimate")


if __name__ == "__main__":
    typer.run(main)
