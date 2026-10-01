"""Chunk shards reduced as one stream: the unchanged reducers give the unsharded result."""

import json
from collections.abc import Callable, Iterator
from itertools import chain
from pathlib import Path

from evaluation import activation, residual, weight_alignment
from evaluation.manifest import Manifest
from evaluation.reducer import metric_rows, shard_rows
from scripts.eval.eval_common import EvaluationError, Json, Record, integer, records


def _write(path: Path, header: Record, observations: list[Record], footer: Record) -> Path:
    rows = [{**header, "sequence": 0}, *({**row, "sequence": index} for index, row in enumerate(observations, 1))]
    end: Record = {**footer, "sequence": len(rows), "observation_count": len(observations),
                   "invocation_count": len({integer(row, "invocation_id") for row in observations})}
    path.write_text("".join(json.dumps(row) + "\n" for row in (*rows, end)), encoding="utf-8")
    return path


def _split(root: Path, name: str, chunks: int) -> tuple[Path, list[Path]]:
    """The fixture stream repeated over `chunks` chunks: one unsharded stream, and one stream per chunk whose
    invocation ids restart at 0 as in a separate shard process."""
    source = list(records(root / (name + ".jsonl")))
    header, observations, footer = source[0], source[1:-1], source[-1]
    invocations = 1 + max(integer(row, "invocation_id") for row in observations)

    def copy(chunk: int, shift: int, run: Json) -> list[Record]:
        return [{**row, "chunk_id": chunk, "invocation_id": integer(row, "invocation_id") + shift, "run_id": run}
                for row in observations]

    whole = _write(root / f"{name}-whole.jsonl", header,
                   [row for chunk in range(chunks) for row in copy(chunk, chunk * invocations, header["run_id"])],
                   footer)
    shards = [_write(root / f"{name}-shard-{chunk}.jsonl", {**header, "run_id": f"shard-{chunk}"},
                     copy(chunk, 0, f"shard-{chunk}"), {**footer, "run_id": f"shard-{chunk}"}) for chunk in range(chunks)]
    return whole, shards


def shards(root: Path, manifest: Manifest) -> None:
    streams: dict[str, tuple[Callable[[Path], Iterator[Record]], Callable[..., Record]]] = {
        "activation": (lambda path: metric_rows(path, "im2p-activation-quant-metrics", manifest), activation.summarize),
        "residual": (lambda path: metric_rows(path, "im2p-residual-path-metrics", manifest), residual.summarize),
        "scale": (lambda path: weight_alignment.rows(path, manifest), weight_alignment.summarize)}
    for name, (rows, summarize) in streams.items():
        # Given one stream over three chunks and the same observations as three shard streams.
        whole, parts = _split(root, name, 3)
        # When reducing the shards as one stream, then every field equals the unsharded result (same input path).
        assert summarize(shard_rows(parts, rows), whole, manifest) == summarize(rows(whole), whole, manifest), name
        # And a shard whose RUN header differs from the first one is refused.
        bad = _write(root / f"{name}-bad.jsonl", {**next(records(parts[1])), "metric_revision": "other"},
                     list(records(parts[1]))[1:-1], list(records(parts[1]))[-1])
        try:
            list(shard_rows([parts[0], bad], rows))
        except EvaluationError as error:
            assert str(error) == "metric shard RUN headers differ"
        else:
            raise AssertionError("mismatched shard header accepted: " + name)
    # Without moving invocation ids past the previous shard, the residual reducer sees duplicate stripes.
    _, parts = _split(root, "residual", 2)
    first, second = (metric_rows(path, "im2p-residual-path-metrics", manifest) for path in parts)
    header = next(first)
    next(second)
    try:
        residual.summarize(chain([header], first, second), parts[0], manifest)
    except EvaluationError as error:
        assert str(error) == "duplicate MAIN_STRIPE"
        return
    raise AssertionError("duplicate residual stripes accepted")
