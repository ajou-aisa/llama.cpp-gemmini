from __future__ import annotations

from contextlib import closing
from fractions import Fraction
from pathlib import Path
import sqlite3

from eval_common import Record, decode, integer, read_json, record, require, text


def rational(value: Fraction) -> Record:
    return {"numerator": value.numerator, "denominator": value.denominator}


def schedule_time(row: Record, field: str = "result_ready_ns") -> Fraction:
    value = record(row.get(field))
    return Fraction(integer(value, "numerator"), integer(value, "denominator", 1))


def prefill_dispatches(lifecycle: Record) -> list[int]:
    require(lifecycle.get("schema") == "im2p-execution-lifecycle" and lifecycle.get("version") == 2 and
            lifecycle.get("source_kind") == "PRODUCER_DECLARED", "current producer lifecycle required")
    application = record(lifecycle.get("application"))
    require(integer(application, "expected_samples") == 128, "incomplete lifecycle application coverage")
    steps = application.get("prefill_steps")
    require(isinstance(steps, list) and bool(steps), "missing source-bound prefill preparation steps")
    if not isinstance(steps, list):
        return []
    dispatches = [integer(record(step), "dispatch_id") for step in steps]
    require(dispatches[0] == 0 and len(set(dispatches)) == len(dispatches) and
            all(integer(record(step), "batch_index") == index for index, step in enumerate(steps)),
            "invalid prefill preparation/dispatch coverage")
    return dispatches


def scheduled_application_result(path: Path, prefill_ids: list[int]) -> Record:
    require(bool(prefill_ids) and prefill_ids[0] == 0 and len(prefill_ids) == len(set(prefill_ids)),
            "explicit prefill dispatch coverage required")
    dispatch_nodes = {f"dispatch:{identity}:begin" for identity in prefill_ids}
    prep_nodes = {f"application:prefill-prep:{identity}" for identity in prefill_ids}
    wanted = {"application:request:begin", *dispatch_nodes, *prep_nodes,
              *(f"application:sample:{index}" for index in range(128))}
    with path.open("rb") as stream:
        sqlite_schedule = stream.read(16) == b"SQLite format 3\x00"
    if sqlite_schedule:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as database:
            manifest = database.execute("SELECT body FROM metadata WHERE key='manifest'").fetchone()
            require(manifest is not None, "missing official schedule manifest")
            header = decode(manifest[0])
            require(header.get("schema") == "im2p-execution-schedule-sqlite" and header.get("version") == 1,
                    "unsupported official SQLite schedule")
            placeholders = ",".join("?" for _ in dispatch_nodes)
            selected = database.execute("SELECT identity,body FROM results WHERE identity='application:request:begin' "
                "OR identity LIKE 'application:sample:%' OR identity LIKE 'application:prefill-prep:%' "
                f"OR identity IN ({placeholders})", tuple(dispatch_nodes)).fetchall()
            rows = [decode(body) for _, body in selected]
            require(all(row.get("node_id") == identity for (identity, _), row in zip(selected, rows)),
                    "SQLite schedule node identity/body mismatch")
    else:
        header = read_json(path)
        require(header.get("schema") == "im2p-execution-schedule" and header.get("version") == 1,
                "unsupported official JSON schedule")
        raw_nodes = header.get("nodes")
        require(isinstance(raw_nodes, list), "missing official schedule nodes")
        rows = [record(raw) for raw in raw_nodes] if isinstance(raw_nodes, list) else []
    require(header.get("scope") == "RECONSTRUCTED" and
            header.get("service_validation_scope") == "CURRENT_CERTIFIED_SEQUENCE",
            "current certified reconstructed schedule required")
    selected_rows: list[Record] = []
    for row in rows:
        identity = row.get("node_id")
        if isinstance(identity, str) and (identity in wanted or identity.startswith("application:sample:") or
                                          identity.startswith("application:prefill-prep:")):
            selected_rows.append(row)
    require(len(selected_rows) == len(wanted), "application schedule endpoint/prep coverage mismatch")
    nodes = {text(row, "node_id"): row for row in selected_rows}
    require(set(nodes) == wanted, "application schedule endpoint/prep identity mismatch")
    request = nodes["application:request:begin"]
    start = schedule_time(request)
    require(schedule_time(request, "accepted_ns") == start, "request-start barrier has duration")
    samples = [schedule_time(nodes[f"application:sample:{index}"]) for index in range(128)]
    for identity in prefill_ids:
        prep = nodes[f"application:prefill-prep:{identity}"]
        dispatch = nodes[f"dispatch:{identity}:begin"]
        require(start <= schedule_time(prep, "accepted_ns") <= schedule_time(prep) <=
                schedule_time(dispatch, "accepted_ns") == schedule_time(dispatch) <= samples[0],
                "invalid prefill preparation interval or dispatch ordering")
    require(all(a <= b for a, b in zip(samples, samples[1:])),
            "non-monotonic reconstructed application endpoints")
    return {"ttft_ns": rational(samples[0] - start),
            "tpot_ns": rational((samples[-1] - samples[0]) / 127),
            "samples": 128, "decode_calls": 127}
