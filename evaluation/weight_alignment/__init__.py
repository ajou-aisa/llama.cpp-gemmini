"""SCU offset observations; no timing or unavailable pre-quantization inference."""

from math import isfinite, ldexp
from pathlib import Path

from evaluation.manifest import Manifest
from evaluation.reducer import metric_rows, summary
from scripts.eval.eval_common import (
    EvaluationError,
    Record,
    integer,
    ratio,
    require,
    text,
)


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
    rows = metric_rows(path, "im2p-scale-alignment-metrics", manifest)
    header = next(rows)
    domain = text(header, "scale_domain")
    require(domain == "hp1_block_pot_to_channel_anchor", "unsupported/unbound scale domain")
    seen: set[tuple[int, str, int, int, int]] = set()
    offset_sum, offset_max, updated, total = 0, 0, 0, 0
    for row in rows:
        require(row.get("kind") == "SCALE_ALIGNMENT", "unexpected SCU record")
        work_type = text(row, "work_type")
        require(work_type in ("DENSE", "RESIDUAL"), "unsupported SCU work type")
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
        offset_sum += offset
        offset_max = max(offset_max, offset)
        updated += changed
        total += count
    require(bool(seen), "empty SCU observation stream")
    return {**summary(manifest, path, "scale-alignment"), "scale_domain": domain,
            "avg_delta_w": ratio(offset_sum, len(seen)), "max_delta_w": offset_max,
            "scu_update_fraction": ratio(updated, total), "delta_w_sum": offset_sum,
            "alignment_count": len(seen), "updated_partial_sum_count": updated,
            "avg_delta_w_weighting": "one_per_observed_work_stripe_column_original_block",
            "total_partial_sum_count": total,
            "update_definition": "partial_sums_requiring_nonzero_scu_shift",
            "pre_pot_fp_scale_available": False}
