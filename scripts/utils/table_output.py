#!/usr/bin/env python3
"""Deterministic Markdown and SVG table renderers."""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Sequence


class TableOutputError(Exception):
    """Raised when table data violates the renderer contract."""


@dataclass(frozen=True)
class TableRow:
    """One caller-ordered Markdown row."""

    cells: tuple[str, ...]


@dataclass(frozen=True)
class BarValue:
    """One caller-ordered numeric bar with its unit."""

    label: str
    value: float
    unit: str


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_markdown(headers: Sequence[str], rows: Sequence[TableRow]) -> str:
    """Render a stable ASCII Markdown table."""
    if not headers:
        raise TableOutputError("at least one header is required")
    width = len(headers)
    if any(len(row.cells) != width for row in rows):
        raise TableOutputError("row width does not match headers")
    header = "| " + " | ".join(_markdown_cell(cell) for cell in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_markdown_cell(cell) for cell in row.cells) + " |" for row in rows]
    return "\n".join((header, separator, *body)) + "\n"


def render_svg(title: str, bars: Sequence[BarValue]) -> str:
    """Render deterministic, dependency-free horizontal bars as SVG."""
    units = {bar.unit for bar in bars}
    if len(units) > 1:
        raise TableOutputError("bar series must use one unit")
    if any(not math.isfinite(bar.value) or bar.value < 0 for bar in bars):
        raise TableOutputError("bar values must be finite and non-negative")
    maximum = max((bar.value for bar in bars), default=0.0)
    width = 400
    row_height = 24
    height = max(row_height, row_height * len(bars) + 24)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    lines.append(f"<title>{html.escape(title, quote=True)}</title>")
    for index, bar in enumerate(bars):
        y = index * row_height + 20
        bar_width = 0.0 if maximum == 0 else (bar.value / maximum) * 300.0
        label = html.escape(bar.label, quote=True)
        numeric = html.escape(f"{bar.value:g} {bar.unit}", quote=True)
        lines.append(f'<text x="0" y="{y}">{label}</text>')
        lines.append(f'<rect x="90" y="{y - 14}" width="{bar_width:g}" height="12"/>')
        lines.append(f'<text x="395" y="{y}" text-anchor="end">{numeric}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"
