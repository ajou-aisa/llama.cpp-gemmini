from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation import weight_alignment
from evaluation.manifest import Manifest
from evaluation.tests.test_framework import fixtures
from scripts.eval.eval_common import Record, read_json, record, records, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/eval"))
from campaign_inputs import dataset_input
from campaign_metrics import outputs


def rejected_manifest(path: Path) -> None:
    try:
        Manifest.load(path)
    except ValueError:
        return
    raise AssertionError("invalid campaign manifest admitted")


def check(root: Path) -> None:
    manifest = fixtures(root)
    value = read_json(root / "evaluation_manifest.json")
    value.update({"tokenizer_hash": value["tokenizer_sha256"], "DIM": value["dim"], "build_hash": "3" * 64})
    extended = root / "extended.json"
    write_json(extended, value)
    assert Manifest.load(extended).dim == 64
    value["DIM"] = 16
    write_json(root / "bad-alias.json", value)
    rejected_manifest(root / "bad-alias.json")
    source = list(records(root / "activation.jsonl"))
    source[1]["layer"], source[2]["layer"] = "first", "lm_head"
    raw = root / "layers.jsonl.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as stream:
        for row in source:
            stream.write(json.dumps(row) + "\n")
    output = root / "layers"
    output.mkdir()
    outputs("activation", raw, manifest, output, {"first", "lm_head"}, {0})
    assert read_json(output / "activation_metrics.json")["recall"] == 2 / 3
    layer_values = read_json(output / "layer_activation_metrics.json")["layers"]
    assert isinstance(layer_values, list)
    assert [record(row)["recall"] for row in layer_values] == [1, 0.5]
    original: Record = source[-1]
    source[-1] = {**source[1], "invocation_id": 2, "sequence": 3, "chunk_id": 1}
    source.append({**original, "sequence": 4, "observation_count": 3, "invocation_count": 3})
    missing = root / "missing-head.jsonl"
    with missing.open("x", encoding="utf-8") as stream:
        for row in source:
            stream.write(json.dumps(row) + "\n")
    incomplete = root / "incomplete"
    incomplete.mkdir()
    try:
        outputs("activation", missing, manifest, incomplete, {"first", "lm_head"}, {0, 1})
    except ValueError as error:
        assert "per-chunk" in str(error)
    else:
        raise AssertionError("missing second-chunk lm_head admitted")
    assert not list(incomplete.glob("*.json"))
    residual = root / "residual-output"
    residual.mkdir()
    outputs("residual", root / "residual.jsonl", manifest, residual, {"known"}, {0})
    summary = read_json(residual / "residual_metrics.json")
    assert summary["radix_limb_count"] == 2
    assert record(summary["zero_limb_pruning"])["fraction"] == 1 / 4
    assert record(summary["inactive_k_compaction"])["fraction"] == 15 / 16
    scale_rows = list(records(root / "scale.jsonl"))
    second = [{**row, "invocation_id": 1, "chunk_id": 1, "sequence": index + 4}
              for index, row in enumerate(scale_rows[1:-1])]
    scale_rows = [*scale_rows[:-1], *second,
                  {**scale_rows[-1], "sequence": 7, "invocation_count": 2, "observation_count": 6}]
    scale_path = root / "two-invocations.jsonl"
    with scale_path.open("x", encoding="utf-8") as stream:
        for row in scale_rows:
            stream.write(json.dumps(row) + "\n")
    scale_summary = weight_alignment.reduce(scale_path, manifest)
    assert scale_summary["alignment_count"] == 6 and scale_summary["avg_delta_w"] == 4 / 3
    dataset = root / "data.txt"
    dataset.write_text("fixture\n", encoding="utf-8")
    write_json(root / "bad-dataset.json", {"dataset": "WikiText-2", "split": "test",
               "path": "data.txt", "sha256": "0" * 64})
    try:
        dataset_input(root / "bad-dataset.json", dataset, root / "dataset-receipt.json")
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered dataset admitted")
    assert not (root / "dataset-receipt.json").exists()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="evaluation-campaign-test-") as temporary:
        check(Path(temporary))
    print("PASS: manifest aliases, gzip exact counts, layer weighting, per-chunk lm_head rejection, RES fields, SCU multi-invocation counts, dataset tampering")
