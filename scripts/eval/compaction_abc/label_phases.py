#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.16,<1"]
# ///
# ─── How to run ───
# 1. Install uv.
# 2. Run: uv run label_phases.py RESULTS_DIR
# 3. Or chmod +x label_phases.py and invoke directly.
# ──────────────────
"""Identify the four inference passes by chronological transformer block resets."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import typer

PASSES: Final = 4
HEADER_BYTES: Final = 80
A4_BITS: Final = 4


@dataclass(frozen=True, slots=True)
class Capture:
    """A source-bound position in the sequential four-pass capture."""

    case: int
    layer: str
    block: int
    graph_m: int


def read_capture(path: Path) -> Capture:
    """Parse a residual header and transformer block identity."""
    with path.open("rb") as stream:
        header = stream.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES:
            message = f"Truncated capture header: {path}"
            raise RuntimeError(message)
        name_bytes = int.from_bytes(header[64:72], "little")
        layer = stream.read(name_bytes).decode("utf-8")
    block = re.search(r"\bblk\.(\d+)\.", layer)
    if block is None:
        message = f"Missing transformer block identity: {path}"
        raise RuntimeError(message)
    return Capture(
        int(path.stem),
        layer,
        int(block.group(1)),
        int.from_bytes(header[72:80], "little"),
    )


def identify_passes(captures: list[Capture], directory: Path) -> tuple[int, ...]:
    """Require sequential block order and four complete inference passes."""
    indices: list[int] = []
    previous, inference_pass = 0, 0
    for position, capture in enumerate(captures):
        if capture.case != position:
            message = f"Non-contiguous capture IDs: {directory}"
            raise RuntimeError(message)
        if capture.block < previous:
            if capture.block != 0:
                message = f"Unexpected block decrease: {directory}, case {capture.case}"
                raise RuntimeError(message)
            inference_pass += 1
        if inference_pass > 0 and capture.graph_m != 1:
            message = f"Non-single-row decode: {directory}, case {capture.case}"
            raise RuntimeError(message)
        indices.append(inference_pass)
        previous = capture.block
    if inference_pass + 1 != PASSES or not captures or captures[0].block != 0:
        message = f"Expected one prefill + three decode passes: {directory}"
        raise RuntimeError(message)
    return tuple(indices)


def main(result_dir: Path) -> None:
    """Save pass provenance and replace phase labels in uncached/cached CSVs."""
    for model in ("gpt2", "llama"):
        for bits in (4, 8):
            suffix = "" if bits == A4_BITS else "-a8"
            directory = result_dir / f"{model}{suffix}"
            captures = sorted(
                (read_capture(p) for p in directory.glob("*.rbin")),
                key=lambda r: r.case,
            )
            indices = identify_passes(captures, directory)
            with (directory / "passes.tsv").open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(
                    ["case", "execution_pass", "phase", "block", "graph_m", "layer"]
                )
                for capture in captures:
                    index = indices[capture.case]
                    writer.writerow(
                        [
                            capture.case,
                            index,
                            "prefill" if index == 0 else "decode",
                            capture.block,
                            capture.graph_m,
                            capture.layer,
                        ]
                    )
            for path in sorted(result_dir.glob(f"{model}-a{bits}-*.csv")):
                with path.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    header_names, rows = reader.fieldnames, list(reader)
                if header_names is None:
                    message = f"Missing replay header: {path}"
                    raise RuntimeError(message)
                names = list(header_names)
                if "execution_pass" not in names:
                    names.append("execution_pass")
                replacement = path.with_suffix(".phases.csv")
                with replacement.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=names)
                    writer.writeheader()
                    for row in rows:
                        index = indices[int(row["case"])]
                        row.update(
                            phase="prefill" if index == 0 else "decode",
                            execution_pass=str(index),
                        )
                        writer.writerow(row)
                _ = replacement.replace(path)
            typer.echo(
                f"{model} W{bits}A{bits}: execution passes {dict(Counter(indices))}"
            )


if __name__ == "__main__":
    typer.run(main)
