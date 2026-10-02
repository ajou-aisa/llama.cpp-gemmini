"""SCU offset observations; no timing or unavailable pre-quantization inference.

Two native collection modes reduce to one summary: `detailed` streams one SCALE_ALIGNMENT record per
(invocation, work type, stripe, column, original block); `aggregate` validates every such coordinate in the
producer (the checks of `summarize` below, duplicates included) and streams only the integer sums of each
(chunk, layer, work type). Both give the same integer sums, hence the same ratios."""

from collections.abc import Iterator
from itertools import chain
from math import isfinite, ldexp
from pathlib import Path
from typing import Final

from evaluation.manifest import Manifest
from evaluation.reducer import metric_rows, summary
from scripts.eval.eval_common import (
    EvaluationError,
    Record,
    integer,
    ratio,
    records,
    require,
    text,
)

# Producer work type -> summary population; `overall` is their union (the top-level fields).
WORK_TYPES: Final = {"DENSE": "dense", "RESIDUAL": "residual"}
SUMS: Final = ("delta_w_sum", "alignment_count", "updated_partial_sum_count", "total_partial_sum_count")
DETAILED: Final = "im2p-scale-alignment-metrics"
AGGREGATE: Final = "im2p-scale-alignment-aggregate"
MAX_OFFSET: Final = 32767


def _scale(row: Record, field: str) -> float:
    value = row.get(field)
    require(isinstance(value, (int, float)) and not isinstance(value, bool), "invalid scale: " + field)
    if not isinstance(value, (int, float)):
        raise EvaluationError("invalid scale: " + field)
    try:
        result = float(value)
    except OverflowError as error:
        raise EvaluationError("scale exceeds finite numeric domain: " + field) from error
    require(isfinite(result) and result >= 0, "nonfinite/negative scale: " + field)
    return result


def reduce(path: Path, manifest: Manifest) -> Record:
    return summarize(rows(path, manifest), path, manifest)


def rows(path: Path, manifest: Manifest) -> Iterator[Record]:
    """Validated RUN and observations of either collection mode, chosen by the stream's own schema."""
    empty: Record = {}
    if next(records(path), empty).get("schema") != AGGREGATE:
        return metric_rows(path, DETAILED, manifest)
    return _aggregate_rows(path, manifest)


def _aggregate_rows(path: Path, manifest: Manifest) -> Iterator[Record]:
    """The aggregate stream boundary: bound identity, one record per (chunk, layer, work type), coverage."""
    iterator = records(path)
    first = next(iterator)
    require(first.get("kind") == "RUN" and first.get("collection_mode") == "aggregate", "missing aggregate SCU RUN")
    run_id, workload_id = text(first, "run_id"), text(first, "workload_id")
    sequence, observed, alignments, ended = -1, 0, 0, False
    keys: set[tuple[int, str, str]] = set()
    for row in chain((first,), iterator):
        require(not ended, "metric record follows RUN_END")
        require(row.get("schema") == AGGREGATE and integer(row, "version") == 1, "wrong metric stream schema/version")
        require(row.get("run_id") == run_id and row.get("workload_id") == workload_id, "metric stream identity changed")
        require(row.get("manifest_sha256") == manifest.sha256, "metric manifest hash mismatch")
        require(row.get("precision") == manifest.precision and integer(row, "dim") == manifest.dim,
                "missing/mixed metric profile")
        require(integer(row, "sequence") > sequence, "non-increasing metric sequence")
        sequence = integer(row, "sequence")
        if row.get("kind") == "RUN_END":
            require(row.get("success") is True, "metric run failed")
            require(integer(row, "observation_count") == observed and integer(row, "alignment_count") == alignments,
                    "metric observation coverage mismatch")
            require(integer(row, "scale_invocation_count") == integer(row, "invocation_count", 1),
                    "metric invocation coverage mismatch")
            ended = True
        elif row is first:
            yield row
        else:
            key = (integer(row, "chunk_id"), text(row, "layer"), text(row, "work_type"))
            require(key not in keys, "duplicate SCU aggregate")
            keys.add(key)
            observed += 1
            alignments += integer(row, "alignment_count")
            yield row
    require(ended, "missing successful metric RUN_END")


def _population(sums: dict[str, int], offset_max: int) -> Record:
    """Integer sums of one SCU population, then division; null when the population is empty."""
    observed = sums["alignment_count"] > 0
    return {**sums, "avg_delta_w": ratio(sums["delta_w_sum"], sums["alignment_count"]),
            "max_delta_w": offset_max if observed else None,
            "scu_update_fraction": ratio(sums["updated_partial_sum_count"], sums["total_partial_sum_count"])}


def summarize(rows: Iterator[Record], path: Path, manifest: Manifest) -> Record:
    """Reduce already validated observations, including a single layer's partition."""
    header = next(rows)
    domain = text(header, "scale_domain")
    require(domain == "hp1_block_pot_to_channel_anchor", "unsupported/unbound scale domain")
    seen: set[tuple[int, str, int, int, int]] = set()
    previous_invocation = -1
    sums: dict[str, dict[str, int]] = {work_type: dict.fromkeys(SUMS, 0) for work_type in WORK_TYPES}
    maxima: dict[str, int] = dict.fromkeys(WORK_TYPES, 0)
    if header.get("collection_mode") == "aggregate":
        _add_aggregates(rows, sums, maxima)
        rows = iter(())
    for row in rows:
        require(row.get("kind") == "SCALE_ALIGNMENT", "unexpected SCU record")
        work_type = text(row, "work_type")
        require(work_type in WORK_TYPES, "unsupported SCU work type")
        invocation = integer(row, "invocation_id")
        require(invocation >= previous_invocation, "SCU invocation order regressed")
        if invocation != previous_invocation:
            seen.clear()
            previous_invocation = invocation
        key = (integer(row, "invocation_id"), work_type, integer(row, "stripe_id"),
               integer(row, "column"), integer(row, "original_block"))
        require(key not in seen, "duplicate SCU coordinate")
        seen.add(key)
        original, aligned = _scale(row, "original_weight_scale"), _scale(row, "aligned_pot_scale")
        offset = integer(row, "scu_shift_offset")
        require(offset <= 32767, "SCU offset exceeds carrier domain")
        changed, count = integer(row, "updated_partial_sum_count"), integer(row, "total_partial_sum_count", 1)
        require(changed <= count, "SCU updates exceed partial-sum population")
        require(type(row.get("zero_weight")) is bool, "missing SCU zero-weight status")
        if row["zero_weight"]:
            require(original == 0 and offset == 0 and changed == 0, "invalid zero-weight SCU record")
        else:
            try:
                expected = ldexp(aligned, offset)
            except OverflowError as error:
                raise EvaluationError("SCU offset overflows finite scale domain") from error
            require(original > 0 and aligned > 0 and original == expected,
                    "SCU scales disagree with offset")
            require(changed == (count if offset else 0), "SCU update count disagrees with required shift")
        group = sums[work_type]
        group["delta_w_sum"] += offset
        group["alignment_count"] += 1
        group["updated_partial_sum_count"] += changed
        group["total_partial_sum_count"] += count
        maxima[work_type] = max(maxima[work_type], offset)
    require(bool(seen) or sums["DENSE"]["alignment_count"] + sums["RESIDUAL"]["alignment_count"] > 0,
            "empty SCU observation stream")
    overall = _population({name: sum(group[name] for group in sums.values()) for name in SUMS},
                          max(maxima.values()))
    return {**summary(manifest, path, "scale-alignment"), "scale_domain": domain, **overall,
            "avg_delta_w_weighting": "one_per_observed_work_stripe_column_original_block",
            "update_definition": "partial_sums_requiring_nonzero_scu_shift",
            "pre_pot_fp_scale_available": False, "overall": overall,
            **{name: _population(sums[work_type], maxima[work_type]) for work_type, name in WORK_TYPES.items()}}


def _add_aggregates(rows: Iterator[Record], sums: dict[str, dict[str, int]], maxima: dict[str, int]) -> None:
    """Add producer-validated integer sums; reject sums no set of valid coordinates can produce."""
    for row in rows:
        require(row.get("kind") == "AGGREGATE", "unexpected SCU record")
        work_type = text(row, "work_type")
        require(work_type in WORK_TYPES, "unsupported SCU work type")
        value = {name: integer(row, name) for name in SUMS}
        count, offset_max = integer(row, "alignment_count", 1), integer(row, "max_delta_w")
        updated, total = value["updated_partial_sum_count"], value["total_partial_sum_count"]
        require(offset_max <= MAX_OFFSET and offset_max <= value["delta_w_sum"] <= offset_max * count,
                "SCU aggregate offsets inconsistent")
        require(updated <= total and total >= count and integer(row, "zero_weight_count") <= count and
                (offset_max > 0 or updated == 0), "SCU aggregate counts inconsistent")
        for name in SUMS:
            sums[work_type][name] += value[name]
        maxima[work_type] = max(maxima[work_type], offset_max)
