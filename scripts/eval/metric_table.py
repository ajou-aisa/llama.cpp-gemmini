"""Deterministic stdout tables of metric summaries (display only; exact values stay in the summaries and JSON/CSV)."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from eval_common import EvaluationError, Json, Record, integer, read_json, record, text

Column = tuple[str, str, str]  # (header, summary field, format: percent | decimal | integer)
Entry = tuple[str, str, int, Record]  # (model, precision, DIM, summary values)
# Title and paper columns of each metric kind, in display order.
TABLES: Final[dict[str, tuple[str, tuple[Column, ...]]]] = {
    "activation": ("Activation Adaptation", (
        ("Recall", "recall", "percent"), ("Jaccard", "jaccard", "percent"),
        ("Requant.", "requant_ratio", "percent"), ("Residual Frac", "residual_fraction", "percent"))),
    "scu": ("SCU Weight-Scale Alignment", (
        ("Avg ΔW", "avg_delta_w", "decimal"), ("Max ΔW", "max_delta_w", "integer"),
        ("Update Frac", "scu_update_fraction", "percent"))),
    "residual": ("Residual Overhead", (
        ("Retained Row", "retained_row_factor", "percent"), ("Retained K", "retained_k_factor", "percent"),
        ("Logical R/MAC", "logical_ratio", "percent"), ("Padded R/MAC", "padded_ratio", "percent"))),
}
KEYS: Final = ("Model", "Prec", "DIM")
ASCII: Final = str.maketrans({"Δ": "d", "—": "-"})


def cell(value: Json, style: str) -> str:
    """Display text of one value; null (an undefined ratio) is `-`."""
    if value is None:
        return "-"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError("metric value is not a number")
    if style == "percent":
        return f"{value * 100:.2f}%"
    return f"{value:.2f}" if style == "decimal" else str(value)


def render(title: str, header: Sequence[str], rows: Sequence[Sequence[str]], plain: bool = False,
           text_columns: int = 2) -> str:
    """A fixed table, independent of the terminal: the first `text_columns` columns left-aligned, the others
    right-aligned; box drawing, or ASCII-only columns with `plain`."""
    if plain:
        title, header = title.translate(ASCII), [name.translate(ASCII) for name in header]
    widths = [max(len(value) for value in column) for column in zip(header, *rows)]

    def cells(values: Sequence[str]) -> list[str]:
        return [value.ljust(width) if index < text_columns else value.rjust(width)
                for index, (value, width) in enumerate(zip(values, widths))]

    if plain:
        return "\n".join([title, *("  ".join(cells(values)).rstrip() for values in (header, *rows))])

    def rule(left: str, middle: str, right: str) -> str:
        return left + middle.join("─" * (width + 2) for width in widths) + right

    body = ["│ " + " │ ".join(cells(values)) + " │" for values in (header, *rows)]
    return "\n".join([title, rule("┌", "┬", "┐"), body[0], rule("├", "┼", "┤"), *body[1:], rule("└", "┴", "┘")])


def metric_table(kind: str, entries: Sequence[Entry], plain: bool = False, variant: str = "") -> str:
    """One row per (model, precision, DIM) with the paper columns of `kind`, in the given order."""
    title, columns = TABLES[kind]
    rows = [[model, precision, str(dim), *(cell(values.get(field), style) for _, field, style in columns)]
            for model, precision, dim, values in entries]
    return render(f"{title} — {variant}" if variant else title, [*KEYS, *(name for name, _, _ in columns)], rows, plain)


def run_table(kind: str, run: Path) -> str:
    """A finished metric run as a one-row plain table (SCU: its dense population, the paper default)."""
    summary, manifest = read_json(run / "summary.json"), read_json(run / "evaluation_manifest.json")
    values = record(summary["dense"]) if kind == "scu" else summary
    return metric_table(kind, [(text(manifest, "model"), text(manifest, "precision"), integer(manifest, "dim"), values)],
                        plain=True, variant="Dense" if kind == "scu" else "")
