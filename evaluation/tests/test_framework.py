# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run --no-project --offline python -m evaluation.tests.test_framework
"""Exact independent metric smoke and fail-closed input contracts."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from evaluation import activation, residual, weight_alignment
from evaluation.manifest import Manifest
from evaluation.tests.test_collection import collection
from evaluation.tests.test_edges import edges
from evaluation.tests.test_scale_breakdown import scale_aggregate, scale_breakdown
from scripts.eval.eval_common import (
    EvaluationError,
    Record,
    integer,
    read_json,
    write_json,
)


def _emit(path: Path, manifest: Manifest, schema: str, observations: list[Record],
          extra: Record | None = None) -> None:
    common: Record = {"schema": schema, "version": 1, "run_id": "smoke", "workload_id": "known",
                      "manifest_sha256": manifest.sha256, "precision": manifest.precision, "dim": manifest.dim}
    header: Record = {**common, "kind": "RUN", "sequence": 0, "precision": manifest.precision,
                      "dim": manifest.dim, **(extra or {})}
    result = [header]
    for index, row in enumerate(observations, 1):
        result.append({**common, "sequence": index, "chunk_id": 0, "invocation_id": 0,
                       "layer": "known", **row})
    result.append({**common, "kind": "RUN_END", "sequence": len(result), "success": True,
                   "invocation_count": len({integer(row, "invocation_id") for row in result[1:]}),
                   "observation_count": len(observations), "reference_complete": True})
    with path.open("x", encoding="utf-8") as stream:
        for row in result:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def _counts(k: int, fp: int, selected: int, intersection: int, invocation: int) -> Record:
    return {"kind": "COUNTS", "invocation_id": invocation, "m": 1, "k": k,
            "valid_positions": k, "finite_positions": k, "nonfinite_positions": 0,
            "reference_complete": True, "reference_invalid_reason": None,
            "fp_selected": fp, "potal_selected": selected, "intersection": intersection,
            "union": fp + selected - intersection, "residual_nnz": 1,
            "eligible_logical_blocks": k // 32, "unique_actual_requantized_blocks": 1,
            "p3_requantization_events": 2}


def fixtures(root: Path, dim: int = 64) -> Manifest:
    """Create small auditable producer-shaped inputs without cycle estimates."""
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "evaluation_manifest.json", {
        "model": "synthetic-known-tensors", "dataset": "WikiText-2",
        "tokenizer_sha256": "1" * 64, "chunk_policy": "smoke-only; not a dataset campaign",
        "precision": "A8W8", "dim": dim, "BK": 32, "seed": 1234, "git_sha": "2" * 40})
    manifest = Manifest.load(root / "evaluation_manifest.json")
    _emit(root / "activation.jsonl", manifest, "im2p-activation-quant-metrics",
          [_counts(32, 1, 2, 1, 0), _counts(64, 2, 1, 1, 1)],
          {"definition_status": "CONFIRMED_BY_USER", "reference_revision": activation.REFERENCE})
    main: Record = {"kind": "MAIN_STRIPE", "stripe_id": 0, "row_begin": 0, "row_count": 2,
                    "m": 2, "n": 16, "k": 64, "physical_fragments": 64 // min(dim, 32)}
    radix: Record = {"kind": "RADIX_STRIPE", "stripe_id": 0, "radix_limb_count": 2,
                     "original_rows": 4, "original_radix_rows": 4, "main_original_rows": 2,
                     "original_k": 64}
    compact: Record = {"kind": "COMPACT_WORK", "stripe_id": 0, "m": 3, "n": 16, "k": 4,
                       "original_k": 64, "source_row_begin": 0, "source_row_count": 2,
                       "tile_i_count": 1, "tile_j_count": 1, "tile_k_count": 1,
                       "radix_limb_count": 2, "original_rows": 4, "retained_rows": 3,
                       "zero_limb_pruned_count": 1, "retained_k": 4, "compact_k": 4,
                       "physical_fragments": 2, "runs": [
                           {"original_block_id": 0, "original_k_mask": 3,
                            "compact_k_begin": 0, "compact_k_count": 2},
                           {"original_block_id": 1, "original_k_mask": 5,
                            "compact_k_begin": 2, "compact_k_count": 2}],
                       "row_map": [{"original_lane_id": 0, "source_row": 0},
                                   {"original_lane_id": 0, "source_row": 1},
                                   {"original_lane_id": 1, "source_row": 0}]}
    _emit(root / "residual.jsonl", manifest, "im2p-residual-path-metrics", [main, radix, compact])
    scale: list[Record] = []
    for column, offset, count in ((0, 0, 16), (1, 3, 32), (2, 1, 16)):
        scale.append({"kind": "SCALE_ALIGNMENT", "column": column, "original_block": 0,
                      "work_type": "DENSE", "stripe_id": 0,
                      "original_weight_scale": 0.25 * 2 ** offset, "aligned_pot_scale": 0.25,
                      "scu_shift_offset": offset, "updated_partial_sum_count": count if offset else 0,
                      "total_partial_sum_count": count, "zero_weight": False})
    _emit(root / "scale.jsonl", manifest, "im2p-scale-alignment-metrics", scale,
          {"scale_domain": "hp1_block_pot_to_channel_anchor"})
    return manifest


def smoke(root: Path) -> None:
    # Given original signed rows with independent BK32 blocks and a strict equality boundary.
    values = [[0.0] * 31 + [10.0] + [-100.0] + [0.0] * 31, [1.0] * 64]
    # When selecting the FP reference.
    mask = activation.fp_reference(values)
    # Then negative outliers and constant blocks are excluded; positive top1 remains eligible.
    assert [index for index, selected in enumerate(mask[0]) if selected] == [31]
    assert not any(mask[1])
    assert not any(activation.fp_reference([[0.0, 0.0, 0.0, 0.0, 5.0]])[0])
    manifest = fixtures(root)
    # Given known FP and PoTal masks plus repeated requant events, when counting, then preserve unions.
    selected = [[index in (0, 31) for index in range(64)], [False] * 64]
    observed = activation.observed_counts(values, selected, selected, [(0, 0), (0, 0)], manifest)
    assert (observed["fp_selected"], observed["potal_selected"], observed["intersection"], observed["union"]) == (1, 2, 1, 2)
    assert observed["unique_actual_requantized_blocks"] == 1 and observed["p3_requantization_events"] == 2
    # Given unequal invocation/block sizes; when reducing, then divide accumulated counts.
    act = activation.reduce(root / "activation.jsonl", manifest)
    assert act["recall"] == 2 / 3 and act["jaccard"] == 1 / 2
    assert act["fp_fraction"] == 3 / 96 and act["residual_fraction"] == 2 / 96
    assert act["requant_ratio"] == 2 / 3
    # Given two original BK32 runs each needing a D64 fragment; when reducing, keep both pads.
    res = residual.reduce(root / "residual.jsonl", manifest)
    assert res["logical_ratio"] == 3 / 32 and res["padded_ratio"] == 1
    assert res["retained_row_factor"] == 3 / 4 and res["retained_k_factor"] == 1 / 16
    # Given offsets 0,3,1 and known weighted update counts; when reducing, then exact fractions.
    scale = weight_alignment.reduce(root / "scale.jsonl", manifest)
    assert scale["avg_delta_w"] == 4 / 3 and scale["max_delta_w"] == 3
    assert scale["scu_update_fraction"] == 3 / 4
    for metric, filename in (("activation", "activation_metrics.json"),
                             ("residual", "residual_metrics.json"),
                             ("scale", "scale_alignment_metrics.json")):
        # Given native-shaped input; when invoking the public CLI; then a bound output is emitted.
        result = subprocess.run([sys.executable, "-B", "-m", "evaluation", metric,
            "--manifest", str(root / "evaluation_manifest.json"), "--input", str(root / (metric + ".jsonl")),
            "--output-dir", str(root / "outputs")], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert read_json(root / "outputs" / filename)["manifest_sha256"] == manifest.sha256
    malformed(root, manifest)
    edges(root, manifest)
    scale_breakdown(root, manifest)
    scale_aggregate(root, manifest)
    collection(root)


def malformed(root: Path, manifest: Manifest) -> None:
    source = (root / "activation.jsonl").read_text(encoding="utf-8")
    variants = {
        "truncated": "\n".join(source.splitlines()[:-1]) + "\n",
        "manifest": source.replace(manifest.sha256, "0" * 64),
        "mixed-profile": source.replace('"dim": 64', '"dim": 32'),
        "negative-count": source.replace('"fp_selected": 1', '"fp_selected": -1'),
        "invalid-union": source.replace('"union": 2', '"union": 99'),
        "duplicate-key": source.replace('"m": 1', '"m": 1, "m": 1'),
    }
    for name, value in variants.items():
        # Given a malformed stream; when reducing; then fail before returning metrics.
        path = root / (name + ".jsonl")
        path.write_text(value, encoding="utf-8")
        try:
            activation.reduce(path, manifest)
        except EvaluationError:
            continue
        raise AssertionError("accepted " + name)
    # Given a manifest whose exact bytes changed, when loading, then its identity changes.
    changed = root / "changed-manifest.json"
    changed.write_text((root / "evaluation_manifest.json").read_text() + " ", encoding="utf-8")
    assert Manifest.load(changed).sha256 != manifest.sha256
    # Given a mismatched manifest, when using the CLI, then no metric output is created.
    result = subprocess.run([sys.executable, "-B", "-m", "evaluation", "activation", "--manifest",
        str(changed), "--input", str(root / "activation.jsonl"), "--output-dir", str(root / "rejected")],
        capture_output=True, text=True, check=False)
    assert result.returncode == 2 and not (root / "rejected").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        with tempfile.TemporaryDirectory(prefix="evaluation-framework-") as directory:
            smoke(Path(directory))
    else:
        smoke(args.output.resolve())
    print("PASS: ACT signed BK32, integer aggregation, RES run padding, SCU offsets, SCU dense/residual split, "
          "SCU aggregate parity, manifest/stream rejection, CLI")


if __name__ == "__main__":
    main()
