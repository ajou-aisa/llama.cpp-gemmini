"""SCU split by work type: integer sums per DENSE/RESIDUAL population, overall unchanged."""

import json
from pathlib import Path

from evaluation import weight_alignment
from evaluation.manifest import Manifest
from scripts.eval.eval_common import (
    EvaluationError,
    Json,
    Record,
    integer,
    ratio,
    record,
    records,
)

FIELDS = ("avg_delta_w", "max_delta_w", "scu_update_fraction", "delta_w_sum", "alignment_count",
          "updated_partial_sum_count", "total_partial_sum_count")
EMPTY: Record = {"delta_w_sum": 0, "alignment_count": 0, "updated_partial_sum_count": 0,
                 "total_partial_sum_count": 0, "avg_delta_w": None, "max_delta_w": None, "scu_update_fraction": None}


def _scale(work_type: str, column: int, offset: int, count: int, invocation: int = 0, zero: bool = False) -> Record:
    return {"work_type": work_type, "column": column, "invocation_id": invocation, "scu_shift_offset": offset,
            "original_weight_scale": 0.0 if zero else 0.25 * 2 ** offset, "aligned_pot_scale": 0.0 if zero else 0.25,
            "updated_partial_sum_count": count if offset else 0, "total_partial_sum_count": count, "zero_weight": zero}


def _stream(root: Path, name: str, observations: list[Record]) -> Path:
    """The fixture stream's RUN/RUN_END around new observations, renumbered."""
    source = list(records(root / "scale.jsonl"))
    rows: list[Record] = [source[0]]
    for row in observations:
        rows.append({**source[1], **row, "sequence": len(rows)})
    rows.append({**source[-1], "sequence": len(rows), "observation_count": len(observations),
                 "invocation_count": len({integer(row, "invocation_id") for row in observations})})
    path = root / (name + ".jsonl")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _pre_split(rows: list[Record]) -> Record:
    """One SCU population as reduced before the split (the definition the overall fields keep)."""
    offsets = [integer(row, "scu_shift_offset") for row in rows]
    updated = sum(integer(row, "updated_partial_sum_count") for row in rows)
    total = sum(integer(row, "total_partial_sum_count") for row in rows)
    return {"avg_delta_w": ratio(sum(offsets), len(offsets)), "max_delta_w": max([0, *offsets]),
            "scu_update_fraction": ratio(updated, total), "delta_w_sum": sum(offsets),
            "alignment_count": len(offsets), "updated_partial_sum_count": updated, "total_partial_sum_count": total}


def _top(result: Record) -> dict[str, Json]:
    return {key: result[key] for key in FIELDS}


def _measures(population: Json) -> tuple[Json, Json, Json]:
    value = record(population)
    return value["avg_delta_w"], value["max_delta_w"], value["scu_update_fraction"]


def scale_breakdown(root: Path, manifest: Manifest) -> None:
    def reduce(name: str, rows: list[Record]) -> Record:
        return weight_alignment.reduce(_stream(root, name, rows), manifest)

    dense = [_scale("DENSE", 0, 0, 16), _scale("DENSE", 1, 3, 32), _scale("DENSE", 2, 1, 16)]
    # Given DENSE-only observations, when reducing, then dense is the whole population and residual is empty.
    result = reduce("scale-dense-only", dense)
    assert result["dense"] == result["overall"] == _top(result) == _pre_split(dense)
    assert result["residual"] == EMPTY
    # Given the same observations as RESIDUAL work, when reducing, then the populations swap.
    residual = [{**row, "work_type": "RESIDUAL"} for row in dense]
    result = reduce("scale-residual-only", residual)
    assert result["residual"] == result["overall"] == _top(result) == _pre_split(residual)
    assert result["dense"] == EMPTY
    # Given mixed work types sharing one coordinate, when reducing, then both are kept and overall is the union.
    mixed = [*dense, _scale("RESIDUAL", 0, 2, 8), _scale("RESIDUAL", 1, 0, 24)]
    result = reduce("scale-mixed", mixed)
    assert result["dense"] == _pre_split(dense) and result["residual"] == _pre_split(mixed[3:])
    assert result["overall"] == _top(result) == _pre_split(mixed)
    assert _measures(result["overall"]) == (6 / 5, 3, 56 / 96) and _measures(result["residual"]) == (1, 2, 8 / 32)
    # Given unequal populations per invocation, when reducing, then each work type divides its integer sums
    # (a mean of the DENSE invocation fractions 1/1 and 0/3 would be 1/2).
    result = reduce("scale-integer-sums", [
        _scale("DENSE", 0, 1, 1), _scale("RESIDUAL", 0, 4, 2),
        _scale("DENSE", 0, 0, 3, invocation=1), _scale("RESIDUAL", 0, 0, 6, invocation=1),
        _scale("RESIDUAL", 1, 0, 6, invocation=1)])
    assert _measures(result["dense"]) == (1 / 2, 1, 1 / 4) and _measures(result["residual"]) == (4 / 3, 4, 2 / 14)
    # Given a zero-weight RESIDUAL observation, when reducing, then it is sampled and counted but never updated.
    rows = [_scale("DENSE", 0, 2, 4), _scale("RESIDUAL", 0, 0, 12, zero=True), _scale("RESIDUAL", 1, 1, 4)]
    result = reduce("scale-zero-weight-split", rows)
    assert record(result["residual"])["alignment_count"] == 2 and record(result["dense"])["alignment_count"] == 1
    assert _measures(result["residual"]) == (1 / 2, 1, 4 / 16) and _measures(result["dense"]) == (2, 2, 1)
    assert result["overall"] == _pre_split(rows)
    # Given one coordinate twice within one work type, when reducing, then reject the duplicate.
    path = _stream(root, "scale-duplicate", [_scale("DENSE", 0, 0, 4), _scale("DENSE", 0, 1, 4)])
    try:
        weight_alignment.reduce(path, manifest)
    except EvaluationError as error:
        assert str(error) == "duplicate SCU coordinate"
    else:
        raise AssertionError("duplicate SCU coordinate accepted")


def _aggregate_stream(root: Path, name: str, detailed: Path, edit: dict[str, Json] | None = None,
                      extra: list[Record] | None = None) -> Path:
    """What the aggregate producer emits for the observations of `detailed`: integer sums per
    (chunk, layer, work type) between the same RUN/RUN_END identity."""
    source = list(records(detailed))
    run, end, observations = source[0], source[-1], source[1:-1]
    groups: dict[tuple[Json, Json, Json], Record] = {}
    for row in observations:
        key = (row["chunk_id"], row["layer"], row["work_type"])
        group = groups.setdefault(key, {"chunk_id": row["chunk_id"], "layer": row["layer"],
                                        "work_type": row["work_type"], "delta_w_sum": 0, "max_delta_w": 0,
                                        "alignment_count": 0, "updated_partial_sum_count": 0,
                                        "total_partial_sum_count": 0, "zero_weight_count": 0})
        group["delta_w_sum"] = integer(group, "delta_w_sum") + integer(row, "scu_shift_offset")
        group["max_delta_w"] = max(integer(group, "max_delta_w"), integer(row, "scu_shift_offset"))
        for name_ in ("updated_partial_sum_count", "total_partial_sum_count"):
            group[name_] = integer(group, name_) + integer(row, name_)
        group["alignment_count"] = integer(group, "alignment_count") + 1
        group["zero_weight_count"] = integer(group, "zero_weight_count") + int(row["zero_weight"] is True)
    common = {key: run[key] for key in ("version", "run_id", "workload_id", "manifest_sha256", "precision", "dim")}
    rows: list[Record] = [{**run, "schema": weight_alignment.AGGREGATE, "collection_mode": "aggregate"}]
    for group in [*sorted(groups.values(), key=lambda value: json.dumps(value, sort_keys=True)), *(extra or [])]:
        rows.append({**common, "schema": weight_alignment.AGGREGATE, "kind": "AGGREGATE", "sequence": len(rows),
                     **group})
    rows.append({**end, "schema": weight_alignment.AGGREGATE, "sequence": len(rows),
                 "observation_count": len(rows) - 1, "alignment_count": len(observations),
                 "scale_invocation_count": end["invocation_count"], **(edit or {})})
    path = root / (name + ".jsonl")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def scale_aggregate(root: Path, manifest: Manifest) -> None:
    # Given detailed observations over two chunks, two layers, three invocations and both work types (zero weight
    # included), and the aggregate stream of the same observations.
    def at(chunk: int, layer: str, row: Record) -> Record:
        return {**row, "chunk_id": chunk, "layer": layer}
    observations = [at(0, "blk.0", _scale("DENSE", 0, 4, 3)), at(0, "blk.0", _scale("DENSE", 1, 0, 5)),
                    at(0, "blk.0", _scale("RESIDUAL", 0, 2, 7)), at(0, "blk.0", _scale("RESIDUAL", 1, 0, 11, zero=True)),
                    at(0, "blk.1", _scale("DENSE", 0, 7, 2, invocation=1)),
                    at(1, "blk.0", _scale("DENSE", 0, 2, 4, invocation=2)),
                    at(1, "blk.0", _scale("RESIDUAL", 0, 9, 1, invocation=2))]
    detailed = _stream(root, "scale-detailed-split", observations)
    aggregate = _aggregate_stream(root, "scale-aggregate-split", detailed)
    # When reducing both, then every population carries the same integer sums and the same ratios.
    expected, result = weight_alignment.reduce(detailed, manifest), weight_alignment.reduce(aggregate, manifest)
    for population in ("overall", "dense", "residual"):
        assert result[population] == expected[population], population
    assert {key: value for key, value in result.items() if key != "input_sha256"} == \
        {key: value for key, value in expected.items() if key != "input_sha256"}
    assert record(result["overall"])["alignment_count"] == 7 and record(result["residual"])["max_delta_w"] == 9
    # And per layer (the campaign's layer partition), the same.
    for layer in ("blk.0", "blk.1"):
        def partition(path: Path, layer: str = layer) -> Record:
            stream = weight_alignment.rows(path, manifest)
            header = next(stream)
            return weight_alignment.summarize(iter([header, *(row for row in stream if row["layer"] == layer)]),
                                              path, manifest)
        left, right = partition(detailed), partition(aggregate)
        assert all(left[name] == right[name] for name in ("overall", "dense", "residual")), layer
    # Given aggregate streams that no valid producer run can emit, when reducing, then reject them.
    duplicate = list(records(aggregate))[1]
    bad: dict[str, tuple[dict[str, Json], list[Record] | None]] = {"duplicate": ({}, [{key: duplicate[key] for key in duplicate if key not in
                                ("schema", "kind", "sequence", "version", "run_id", "workload_id", "manifest_sha256",
                                 "precision", "dim")}]),
           "lost-alignment": ({"alignment_count": 6}, None),
           "lost-invocation": ({"scale_invocation_count": 2}, None),
           "max-above-sum": ({}, [{"chunk_id": 2, "layer": "blk.9", "work_type": "DENSE", "delta_w_sum": 1,
                                  "max_delta_w": 3, "alignment_count": 1, "updated_partial_sum_count": 1,
                                  "total_partial_sum_count": 1, "zero_weight_count": 0}]),
           "update-without-shift": ({}, [{"chunk_id": 2, "layer": "blk.9", "work_type": "DENSE", "delta_w_sum": 0,
                                         "max_delta_w": 0, "alignment_count": 1, "updated_partial_sum_count": 1,
                                         "total_partial_sum_count": 1, "zero_weight_count": 0}])}
    for name, (edit, extra) in bad.items():
        path = _aggregate_stream(root, "scale-aggregate-" + name, detailed, edit, extra)
        if extra:  # keep RUN_END totals consistent so only the record itself is wrong
            path.unlink()
            path = _aggregate_stream(root, "scale-aggregate-" + name, detailed,
                                     {"alignment_count": len(observations) + integer(extra[0], "alignment_count")}, extra)
        try:
            weight_alignment.reduce(path, manifest)
        except EvaluationError:
            continue
        raise AssertionError("aggregate stream accepted: " + name)
