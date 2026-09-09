#!/usr/bin/env python3
"""Deterministic Markdown and CSV table output."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


class TableOutputError(Exception):
    """Raised when table data violates the renderer contract."""


TableRow = tuple[str, ...]


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def export_csv(path: Path, headers: Sequence[str], rows: Iterable[TableRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def render_markdown(headers: Sequence[str], rows: Sequence[TableRow]) -> str:
    """Render a stable ASCII Markdown table."""
    if not headers:
        raise TableOutputError("at least one header is required")
    width = len(headers)
    if any(len(row) != width for row in rows):
        raise TableOutputError("row width does not match headers")
    header = "| " + " | ".join(_markdown_cell(cell) for cell in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_markdown_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join((header, separator, *body)) + "\n"
