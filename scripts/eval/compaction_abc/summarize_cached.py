#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2,<3", "typer>=0.16,<1"]
# ///
# ─── How to run ───
# 1. Install uv.
# 2. Run: uv run summarize_cached.py RESULTS_DIR
# 3. Or chmod +x summarize_cached.py and invoke directly.
# ──────────────────
"""Combine measured compact copies with the host-buffer reuse scenario."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Literal, assert_never

import typer
from pydantic import BaseModel, ConfigDict

from summarize import reduce_file


class CopyRow(BaseModel):
    """Parse one measured copy from the cached C++ replay."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    case: int
    layer: str
    phase: Literal["prefill", "decode"]
    bits: int
    n: int
    original_k: int
    compact_k: int
    copy_p50_ns: int
    full_cache_bytes: int
    layer_decode_ns: int


def main(result_dir: Path) -> None:
    """Write DIM16 reuse costs, pairing cached copies with the original replay."""
    destination = result_dir / "cached-weight-summary.csv"
    setup = result_dir / "cached-weight-setup.csv"
    with (
        destination.open("w", newline="") as stream,
        setup.open("w", newline="") as startup,
    ):
        writer, startup_writer = csv.writer(stream), csv.writer(startup)
        writer.writerow(
            [
                "model",
                "bits",
                "dim",
                "phase",
                "variant",
                "cases",
                "host_ms",
                "copy_ms",
                "device_cycles",
                "sequential_1ghz_ms",
            ]
        )
        startup_writer.writerow(
            ["model", "bits", "layers", "full_cache_bytes", "layer_decode_sum_ms"]
        )
        for model in ("gpt2", "llama"):
            for bits in (4, 8):
                path = result_dir / f"{model}-a{bits}-cached.csv"
                with path.open(newline="") as source:
                    copies = [
                        CopyRow.model_validate(row) for row in csv.DictReader(source)
                    ]
                rows = reduce_file(result_dir / f"{model}-a{bits}-d16.csv")
                expected = {
                    (r.case, r.phase, r.bits, r.n, r.original_k, r.k)
                    for r in rows
                    if r.variant == "A"
                }
                actual = {
                    (r.case, r.phase, r.bits, r.n, r.original_k, r.compact_k)
                    for r in copies
                }
                if expected != actual or len(copies) != len(expected):
                    message = f"Cached copy coverage mismatch: {path}"
                    raise RuntimeError(message)
                layers = {r.layer: r for r in copies}
                startup_writer.writerow(
                    [
                        model,
                        bits,
                        len(layers),
                        sum(r.full_cache_bytes for r in layers.values()),
                        sum(r.layer_decode_ns for r in layers.values()) / 1e6,
                    ]
                )
                copying: defaultdict[str, int] = defaultdict(int)
                for row in copies:
                    copying[row.phase] += row.copy_p50_ns
                for phase in ("prefill", "decode"):
                    for variant in ("A", "B", "C"):
                        paired = [
                            r for r in rows if r.phase == phase and r.variant == variant
                        ]
                        match variant:
                            case "A":
                                copy_ns, label = copying[phase], "A*"
                            case "B" | "C":
                                copy_ns, label = 0, variant
                            case unreachable:
                                assert_never(unreachable)
                        host_ns = (
                            sum(
                                r.decompose_ns + r.pack_ns + r.restore_ns
                                for r in paired
                            )
                            + copy_ns
                        )
                        cycles = sum(r.device_cycles for r in paired)
                        writer.writerow(
                            [
                                model,
                                bits,
                                16,
                                phase,
                                label,
                                len(paired),
                                host_ns / 1e6,
                                copy_ns / 1e6,
                                cycles,
                                (host_ns + cycles) / 1e6,
                            ]
                        )
    typer.echo(f"Wrote {destination} and {setup}")


if __name__ == "__main__":
    typer.run(main)
