#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/eval/measurement_identity.py RUN [RUN ...]
"""Shared identity of finished measurement runs (read only).

  measurement_identity.py RUN [RUN ...]        also: run_measurement.py identity RUN [RUN ...]

For a performance, timeline or metric (activation/residual/SCU) run directory this prints the identity the domains
share (model, model manifest, precision, DIM, block size, dataset) and the build identities the run recorded on its
own. Runs of one model and configuration agree on the shared part; their builds are expected to differ (same model
and configuration, different instrumentation build). Exit status 1 when a shared field differs between the given
runs. Nothing is measured, built or modified.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Final

from campaign_build import COMBINED, METRIC_SINKS
from eval_common import EvaluationError, Json, Record, read_json, require

SHARED: Final = ("model_sha256", "model_manifest_sha256", "model_artifact", "architecture", "quantization",
                 "precision", "dim", "block_size", "dataset_sha256")


def at(value: Json, *keys: str) -> Json:
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def optional(path: Path) -> Record:
    return read_json(path) if path.is_file() else {}


def profile(value: Json) -> tuple[str | None, int | None]:
    """(`a8w8`, 32) from the hardware profile `a8w8-d32-hp1`."""
    match = re.fullmatch(r"(a\d+w\d+)-d(\d+)-hp1", value) if isinstance(value, str) else None
    return (match.group(1), int(match.group(2))) if match else (None, None)


def model_fields(sha256: Json, entry: Json) -> Record:
    """Model part of the shared identity: the model bytes and, when recorded, its frozen model-manifest entry."""
    return {"model_sha256": sha256, "model_manifest_sha256": at(entry, "manifest", "sha256"),
            "model_artifact": at(entry, "artifact"), "architecture": at(entry, "gguf", "general.architecture"),
            "quantization": at(entry, "gguf", "quantization")}


def build_identity(receipt: Json, info: Json) -> Record:
    """What a llama build receipt and its --build-info say about one build; never compared between domains."""
    sinks: Record = {kind: at(info, key) for kind, (_, key) in METRIC_SINKS.items()}
    return {"kind": at(receipt, "kind"), "semantic_options_sha256": at(receipt, "semantic_options_sha256"),
            "binary_sha256": at(receipt, "runner_sha256"), "metric_sinks": sinks}


def performance(run: Path) -> Record:
    workload = at(read_json(run / "performance.json"), "workload")
    reuse = at(optional(run / "provenance.json"), "collection_reuse")
    library = at(optional(run / "build/cycle-model.json"), "receipt", "library", "sha256")
    builds: Record = {"cycle-model": {"kind": "cycle-model", "binary_sha256": library}}
    block: Json = None
    for name in ("potal", "fullcpu"):
        fresh, reused = optional(run / "build" / f"{name}.json"), at(reuse, f"{name}_build")
        if fresh:
            receipt, info = fresh.get("receipt"), fresh.get("build_info")
        elif isinstance(reused, str) and (Path(reused) / "build-receipt.json").is_file():
            receipt, info = read_json(Path(reused) / "build-receipt.json"), optional(Path(reused) / "build-info.json")
        else:
            continue
        builds[name] = {**build_identity(receipt, info), "collection": "reused" if not fresh else "fresh"}
        block = at(info, "block_size") if name == "potal" else block
    precision, dim = profile(at(workload, "profile"))
    model = model_fields(at(workload, "model", "sha256"), at(workload, "model", "manifest"))
    for field in ("architecture", "quantization"):  # the GGUF header of the measured file, else the manifest entry
        model[field] = at(workload, "model", field) or model[field]
    return {"domain": "performance", "measurement": "performance", "builds": builds,
            "shared": {**model, "precision": precision, "dim": dim, "block_size": block,
                       "dataset_sha256": at(workload, "dataset", "sha256")}}


def timeline(run: Path) -> Record:
    source = at(read_json(run / "export.json"), "source_run")
    require(isinstance(source, str) and (Path(source) / "performance.json").is_file(),
            "the source performance run of this timeline export is gone: " + str(source))
    return {**performance(Path(str(source))), "domain": "timeline", "measurement": "timeline", "builds": {},
            "source_run": source}


def metric(run: Path, manifest: Record) -> Record:
    info, kind = at(manifest, "build_info"), str(manifest["kind"])
    runner = manifest.get("runner")
    receipt = optional(Path(runner).parent.parent / "build-receipt.json") if isinstance(runner, str) else {}
    build = {**build_identity(receipt, info), "kind": kind, "binary_sha256": manifest.get("build_hash"),
             "build_receipt_sha256": manifest.get("build_receipt_sha256")}
    return {"domain": "metric", "measurement": kind, "builds": {kind: build},
            "shared": {**model_fields(manifest.get("model_sha256"), manifest.get("model_manifest")),
                       "precision": f"a{at(info, 'activation_bits')}w{at(info, 'weight_bits')}", "dim": at(info, "dim"),
                       "block_size": at(info, "block_size"), "dataset_sha256": manifest.get("dataset_sha256")}}


def run_identity(run: Path) -> Record:
    """Identity of one finished run directory; the domain is recognised by the run's own artifacts."""
    require(run.is_dir(), "not a run directory: " + str(run))
    if (run / "export.json").is_file():
        row = timeline(run)
    elif (run / "performance.json").is_file():
        row = performance(run)
    else:
        manifest = optional(run / "manifest.json")
        require(manifest.get("kind") in (*METRIC_SINKS, COMBINED),
                "not a finished performance, timeline or metric run: " + str(run))
        row = metric(run, manifest)
    return {"run": str(run), **row}


def compare(rows: list[Record]) -> Record:
    """Shared fields that differ between runs (a mismatch), and fields a run did not record (not a mismatch)."""
    differing: Record = {}
    unrecorded: Record = {}
    for field in SHARED:
        values = {str(row["run"]): at(row, "shared", field) for row in rows}
        known = {json.dumps(value) for value in values.values() if value is not None}
        if len(known) > 1:
            differing[field] = {run: value for run, value in values.items()}
        missing: list[Json] = [run for run, value in values.items() if value is None]
        if missing:
            unrecorded[field] = missing
    return {"same_model_and_configuration": not differing, "differing": differing, "unrecorded": unrecorded}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if arguments else 2
    try:
        rows = [run_identity(Path(argument).resolve()) for argument in arguments]
    except (EvaluationError, OSError, ValueError) as error:
        print(f"measurement identity failed: {error}", file=sys.stderr)
        return 2
    runs: list[Json] = [row for row in rows]
    comparison = compare(rows)
    print(json.dumps({"schema": "potal-measurement-identity", "version": 1, "runs": runs, "comparison": comparison},
                     indent=2, sort_keys=True))
    return 0 if comparison["same_model_and_configuration"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
