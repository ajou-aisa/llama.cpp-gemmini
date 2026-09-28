"""Boundary checks for count conservation, empty residuals and finite SCU carriers."""

import json
from collections.abc import Callable
from pathlib import Path

from evaluation import activation, residual, weight_alignment
from evaluation.manifest import Manifest
from scripts.eval.eval_common import EvaluationError, Record, records


def _write(root: Path, name: str, rows: list[Record]) -> Path:
    path = root / (name + ".jsonl")
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return path


def _reject(action: Callable[[], Record]) -> None:
    try:
        action()
    except EvaluationError:
        return
    raise AssertionError("invalid input accepted")


def edges(root: Path, manifest: Manifest) -> None:
    # Given a subnormal-origin finite scale with carrier 256, when reducing, then keep the valid offset.
    rows = list(records(root / "scale.jsonl"))
    rows[1].update(scu_shift_offset=256, original_weight_scale=2.0 ** 107,
                   aligned_pot_scale=2.0 ** -149, updated_partial_sum_count=16)
    result = weight_alignment.reduce(_write(root, "scale-large-carrier", rows), manifest)
    assert result["avg_delta_w"] == 260 / 3 and result["max_delta_w"] == 256
    assert result["scu_update_fraction"] == 1
    # Given a carrier whose scale cannot remain finite, when reducing, then return a typed rejection.
    rows[1].update(scu_shift_offset=32767, aligned_pot_scale=0.25)
    path = _write(root, "scale-overflow", rows)
    _reject(lambda: weight_alignment.reduce(path, manifest))
    # Given more SCU updates than partial sums, when reducing, then reject impossible counts.
    rows = list(records(root / "scale.jsonl"))
    rows[2]["updated_partial_sum_count"] = 33
    path = _write(root, "scale-impossible-updates", rows)
    _reject(lambda: weight_alignment.reduce(path, manifest))
    # Given a removed scale observation but an intact footer, when reducing, then reject lost coverage.
    rows = list(records(root / "scale.jsonl"))
    del rows[2]
    path = _write(root, "scale-missing-observation", rows)
    _reject(lambda: weight_alignment.reduce(path, manifest))
    # Given a zero-weight observation, when reducing, then exclude it from updates and preserve its sample.
    rows = list(records(root / "scale.jsonl"))
    rows[1].update(original_weight_scale=0.0, aligned_pot_scale=0.0, zero_weight=True)
    result = weight_alignment.reduce(_write(root, "scale-zero-weight", rows), manifest)
    assert result["alignment_count"] == 3 and result["scu_update_fraction"] == 3 / 4
    # Given zero radix limbs, when reducing, then count main capacity and report undefined row retention.
    rows = list(records(root / "residual.jsonl"))
    rows[2].update(radix_limb_count=0, original_rows=0, original_radix_rows=0)
    del rows[3]
    rows[-1]["observation_count"] = 2
    result = residual.reduce(_write(root, "residual-zero", rows), manifest)
    assert result["logical_ratio"] == result["padded_ratio"] == result["retained_k_factor"] == 0
    assert result["retained_row_factor"] is None
    # Given a larger residual-free main stripe, when reducing, then accumulate its full denominator.
    rows = list(records(root / "residual.jsonl"))
    second = {**rows[1], "invocation_id": 1, "m": 4, "row_count": 4, "n": 64, "sequence": 4}
    radix = {**rows[2], "invocation_id": 1, "radix_limb_count": 0, "original_rows": 0,
             "original_radix_rows": 0, "main_original_rows": 4, "sequence": 5}
    rows[-1].update(sequence=6, observation_count=5, invocation_count=2)
    rows = [*rows[:-1], second, radix, rows[-1]]
    result = residual.reduce(_write(root, "residual-unequal-stripes", rows), manifest)
    assert result["logical_ratio"] == 192 / (2048 + 16384) and result["padded_ratio"] == 1 / 2
    assert result["retained_k_factor"] == 4 / 128 and result["retained_row_factor"] == 3 / 4
    # Given nonzero radix metadata but missing compact work, when reducing, then reject incomplete execution.
    rows = list(records(root / "residual.jsonl"))
    del rows[3]
    rows[-1]["observation_count"] = 2
    path = _write(root, "residual-missing-work", rows)
    _reject(lambda: residual.reduce(path, manifest))
    # Given one total-K fragment instead of two original-run fragments, when reducing, then reject padding.
    rows = list(records(root / "residual.jsonl"))
    rows[3]["physical_fragments"] = 1
    path = _write(root, "residual-wrong-padding", rows)
    _reject(lambda: residual.reduce(path, manifest))
    # Given a missing record profile, when reducing, then reject the unbound observation.
    rows = list(records(root / "activation.jsonl"))
    del rows[1]["precision"]
    path = _write(root, "activation-missing-profile", rows)
    _reject(lambda: activation.reduce(path, manifest))
    # Given no selected FP positions, when reducing, then recall remains undefined instead of fabricated.
    rows = list(records(root / "activation.jsonl"))
    for row in rows[1:-1]:
        row.update(fp_selected=0, intersection=0, union=row["potal_selected"])
    result = activation.reduce(_write(root, "activation-empty-reference", rows), manifest)
    assert result["recall"] is None and result["jaccard"] == result["fp_fraction"] == 0
