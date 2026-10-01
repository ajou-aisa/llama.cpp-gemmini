from __future__ import annotations

import gzip
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, TextIO

from eval_common import Json, Record, integer, ratio, records, require, text, write_json

from evaluation import activation, residual, weight_alignment
from evaluation.manifest import Manifest
from evaluation.reducer import metric_rows


# (aggregate summary, per-layer summary) file names of each metric kind.
OUTPUT_FILENAMES: Final = {"activation": ("activation_metrics.json", "layer_activation_metrics.json"),
                           "residual": ("residual_metrics.json", "layer_residual_metrics.json"),
                           "scu": ("scale_alignment_metrics.json", "layer_scale_metrics.json")}


def residual_extensions(row: Record, radix_count: int) -> Record:
    return {**row, "radix_limb_count": radix_count, "radix_limb_count_aggregation": "sum_per_main_stripe",
        "zero_limb_pruning": {"pruned_rows": row["zero_limb_pruned_count"],
            "fraction": ratio(integer(row, "zero_limb_pruned_count"), integer(row, "original_radix_rows"))},
        "inactive_k_compaction": {"removed_k": integer(row, "original_k") - integer(row, "compact_k"),
            "fraction": ratio(integer(row, "original_k") - integer(row, "compact_k"), integer(row, "original_k"))}}


def outputs(kind: str, raw: Path, manifest: Manifest, output: Path,
            expected_layers: set[str], expected_chunks: set[int]) -> None:
    reducer = {"activation": activation, "residual": residual, "scu": weight_alignment}[kind]
    schemas = {"activation": "im2p-activation-quant-metrics", "residual": "im2p-residual-path-metrics",
               "scu": "im2p-scale-alignment-metrics"}
    aggregate = reducer.reduce(raw, manifest)
    layers: list[Json] = []
    shapes: list[Json] = []
    radix_count = 0
    radix_by_layer: dict[str, int] = {}
    coverage: dict[int, set[str]] = {}
    with TemporaryDirectory(prefix="campaign-layers-") as directory, ExitStack() as stack:
        files: dict[str, tuple[Path, TextIO]] = {}
        rows = weight_alignment.rows(raw, manifest) if kind == "scu" else metric_rows(raw, schemas[kind], manifest)
        header = next(rows)
        for row in rows:
            layer = text(row, "layer")
            coverage.setdefault(integer(row, "chunk_id"), set()).add(layer)
            if layer not in files:
                path = Path(directory) / (str(len(files)) + ".gz")
                writer = stack.enter_context(gzip.open(path, "wt", encoding="utf-8", compresslevel=1))
                files[layer] = (path, writer)
                writer.write(json.dumps(header) + "\n")
            files[layer][1].write(json.dumps(row) + "\n")
            if row.get("kind") == "RADIX_STRIPE":
                radix_count += integer(row, "radix_limb_count")
                radix_by_layer[layer] = radix_by_layer.get(layer, 0) + integer(row, "radix_limb_count")
            if row.get("kind") in ("MAIN_STRIPE", "RADIX_STRIPE", "COMPACT_WORK"):
                shapes.append({key: value for key, value in row.items() if key not in ("row_map", "runs")})
        require(set(files) == expected_layers,
                "offloaded layer coverage differs: missing=" + str(sorted(expected_layers - files.keys())) +
                " unexpected=" + str(sorted(files.keys() - expected_layers)))
        require(set(coverage) == expected_chunks and all(value == expected_layers for value in coverage.values()),
                "incomplete per-chunk offloaded layer coverage")
        stack.close()
        for layer, (path, _) in sorted(files.items()):
            layer_result = reducer.summarize(records(path), raw, manifest)
            if kind == "residual":
                layer_result = residual_extensions(layer_result, radix_by_layer[layer])
            layers.append({**layer_result, "layer": layer})
    if kind == "residual":
        aggregate = residual_extensions(aggregate, radix_count)
        write_json(output / "compact-shape-summary.json", {**manifest.identity(), "stripes": shapes,
                   "raw_runs_and_row_maps": str(raw), "input_sha256": aggregate["input_sha256"]})
    summary_name, layer_name = OUTPUT_FILENAMES[kind]
    write_json(output / summary_name, aggregate)
    write_json(output / layer_name, {**manifest.identity(), "input_sha256": aggregate["input_sha256"],
               "aggregation": "integer-sum-before-division", "layers": layers})
