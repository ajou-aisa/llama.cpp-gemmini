"""Strict native stream boundary shared by ACT, RES and SCU reducers."""

from collections.abc import Iterator
from pathlib import Path

from evaluation.manifest import Manifest
from scripts.eval.eval_common import Record, integer, records, require, sha256, text


def metric_rows(path: Path, schema: str, manifest: Manifest) -> Iterator[Record]:
    """Yield RUN and observations only after checking their bound identities."""
    iterator = records(path)
    empty: Record = {}
    first = next(iterator, empty)
    require(first.get("kind") == "RUN", "missing metric RUN")
    require(first.get("precision") == manifest.precision and integer(first, "dim") == manifest.dim,
            "metric profile differs from manifest")
    run_id, workload_id = text(first, "run_id"), text(first, "workload_id")
    sequence = -1
    invocations: dict[int, tuple[int, str]] = {}
    ended = False
    observed = 0
    for row in _prepend(first, iterator):
        require(not ended, "metric record follows RUN_END")
        require(row.get("schema") == schema and integer(row, "version") == 1,
                "wrong metric stream schema/version")
        require(row.get("run_id") == run_id and row.get("workload_id") == workload_id,
                "metric stream identity changed")
        require(row.get("manifest_sha256") == manifest.sha256, "metric manifest hash mismatch")
        require(row.get("precision") == manifest.precision and integer(row, "dim") == manifest.dim,
                "missing/mixed metric profile")
        current = integer(row, "sequence")
        require(current > sequence, "non-increasing metric sequence")
        sequence = current
        kind = text(row, "kind")
        if kind == "RUN_END":
            require(row.get("success") is True, "metric run failed")
            count = integer(row, "invocation_count", 1)
            require(set(invocations) == set(range(count)), "metric invocation coverage mismatch")
            require(integer(row, "observation_count") == observed, "metric observation coverage mismatch")
            require(row.get("reference_complete", True) is True, "incomplete FP reference")
            ended = True
        elif kind == "RUN":
            require(row is first, "duplicate metric RUN")
            yield row
        else:
            key = integer(row, "invocation_id")
            identity = (integer(row, "chunk_id"), text(row, "layer"))
            require(key not in invocations or invocations[key] == identity,
                    "metric invocation identity changed")
            invocations[key] = identity
            observed += 1
            yield row
    require(ended, "missing successful metric RUN_END")


def _prepend(first: Record, iterator: Iterator[Record]) -> Iterator[Record]:
    yield first
    yield from iterator


def summary(manifest: Manifest, path: Path, metric: str) -> Record:
    return {**manifest.identity(), "schema": "potal-" + metric + "-metrics", "version": 1,
            "input_sha256": sha256(path), "aggregation": "integer-sum-before-division",
            "undefined_ratio_policy": "null"}
