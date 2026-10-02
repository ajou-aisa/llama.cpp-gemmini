"""Signed original-FP, logical-row/BK32 activation metrics."""

from collections.abc import Iterator, Sequence
from math import fsum, isfinite, sqrt
from pathlib import Path
from typing import Final

from evaluation.manifest import Manifest
from evaluation.reducer import metric_rows, summary
from scripts.eval.eval_common import Record, integer, ratio, require

REFERENCE: Final = "signed-row-original-bk32-population-2sigma-v1"
COUNTS: Final = ("valid_positions", "fp_selected", "potal_selected", "intersection", "union",
                "residual_nnz", "eligible_logical_blocks", "unique_actual_requantized_blocks",
                "p3_requantization_events")


def fp_reference(rows: Sequence[Sequence[float]]) -> tuple[tuple[bool, ...], ...]:
    """Select strict signed x > mean + 2 population sigma in each original BK32 block."""
    require(bool(rows) and bool(rows[0]), "empty original FP activation")
    width = len(rows[0])
    result: list[tuple[bool, ...]] = []
    for row in rows:
        require(len(row) == width and all(isfinite(value) for value in row),
                "ragged/nonfinite original FP activation")
        mask: list[bool] = []
        for begin in range(0, width, 32):
            block = row[begin:begin + 32]
            mean = fsum(block) / len(block)
            sigma = sqrt(fsum((value - mean) ** 2 for value in block) / len(block))
            mask.extend(value > mean + 2 * sigma for value in block)
        result.append(tuple(mask))
    return tuple(result)


def observed_counts(rows: Sequence[Sequence[float]], selected: Sequence[Sequence[bool]],
                    residual: Sequence[Sequence[bool]], requant_events: Sequence[tuple[int, int]],
                    manifest: Manifest) -> Record:
    """Produce bound integer counts from original FP and actual logical-coordinate observations."""
    reference = fp_reference(rows)
    m, k = len(rows), len(rows[0])
    for mask in (selected, residual):
        require(len(mask) == m and all(len(row) == k and all(type(value) is bool for value in row)
                for row in mask), "activation mask shape/type mismatch")
    eligible = (k + 31) // 32
    require(all(type(row) is int and type(block) is int and 0 <= row < m and 0 <= block < eligible
                for row, block in requant_events), "requantization outside original logical block")
    fp_count = sum(sum(row) for row in reference)
    selected_count = sum(sum(row) for row in selected)
    intersection = sum(reference[row][column] and selected[row][column]
                       for row in range(m) for column in range(k))
    return {"manifest_sha256": manifest.sha256, "kind": "COUNTS", "m": m, "k": k,
            "valid_positions": m * k, "finite_positions": m * k, "nonfinite_positions": 0,
            "reference_complete": True, "reference_invalid_reason": None,
            "fp_selected": fp_count, "potal_selected": selected_count, "intersection": intersection,
            "union": fp_count + selected_count - intersection,
            "residual_nnz": sum(sum(row) for row in residual), "eligible_logical_blocks": m * eligible,
            "unique_actual_requantized_blocks": len(set(requant_events)),
            "p3_requantization_events": len(requant_events)}


def reduce(path: Path, manifest: Manifest) -> Record:
    return summarize(metric_rows(path, "im2p-activation-quant-metrics", manifest), path, manifest)


def summarize(rows: Iterator[Record], path: Path, manifest: Manifest) -> Record:
    """Reduce already validated observations, including a single layer's partition."""
    header = next(rows)
    require(header.get("definition_status") == "CONFIRMED_BY_USER" and
            header.get("reference_revision") == REFERENCE, "unbound ACT reference definition")
    totals = {key: 0 for key in COUNTS}
    seen: set[int] = set()
    for row in rows:
        require(row.get("kind") == "COUNTS", "unexpected ACT record")
        invocation = integer(row, "invocation_id")
        require(invocation not in seen, "duplicate ACT invocation")
        seen.add(invocation)
        require(row.get("reference_complete") is True and
                integer(row, "nonfinite_positions") == 0 and
                row.get("reference_invalid_reason") is None, "invalid ACT original FP reference")
        count = {name: integer(row, name) for name in COUNTS}
        m, k = integer(row, "m", 1), integer(row, "k", 1)
        require(count["valid_positions"] == integer(row, "finite_positions") == m * k,
                "ACT coordinate population mismatch")
        require(count["eligible_logical_blocks"] == m * ((k + 31) // 32),
                "ACT original BK32 denominator mismatch")
        require(max(count["fp_selected"], count["potal_selected"], count["residual_nnz"],
                    count["union"]) <= m * k, "ACT count exceeds population")
        require(count["intersection"] <= min(count["fp_selected"], count["potal_selected"]) and
                count["union"] == count["fp_selected"] + count["potal_selected"] - count["intersection"],
                "invalid ACT intersection/union")
        require(count["unique_actual_requantized_blocks"] <= count["eligible_logical_blocks"] and
                count["p3_requantization_events"] >= count["unique_actual_requantized_blocks"],
                "invalid ACT requantization coverage")
        for key, value in count.items():
            totals[key] += value
    require(bool(seen), "empty ACT observation stream")
    return {**summary(manifest, path, "activation"), **totals, "reference_revision": REFERENCE,
            "recall": ratio(totals["intersection"], totals["fp_selected"]),
            "jaccard": ratio(totals["intersection"], totals["union"]),
            "fp_fraction": ratio(totals["fp_selected"], totals["valid_positions"]),
            "residual_fraction": ratio(totals["residual_nnz"], totals["valid_positions"]),
            "requant_ratio": ratio(totals["unique_actual_requantized_blocks"], totals["eligible_logical_blocks"])}
