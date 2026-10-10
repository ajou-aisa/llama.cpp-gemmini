#!/usr/bin/env python3
# How to run: python3 -B tests/test-e2e-collection-validation.py --im2p ../IM2P.sim FIXTURE_LOG_DIR
"""scripts/eval/check_e2e_collection.py on real producer output: the exsia-paired-pipeline dispatch fixture (a compact
parent and a paired parent with admitted and CAPACITY_NONFIT stripes) passes, and single mutations of its logs are
rejected with the check that owns them: a duplicate work identity or host-stage declaration, a missing call or host-stage
declaration, canonical CPU overlap, invalid geometry, a missing stripe publication, paired facts inconsistent with
admission, an admitted stripe that owns a residual child, a fallback stripe without its residual child, and a residual
child without its stripe. The run-time checks stay identical; only the logs are changed (sequence numbers renumbered
after a record is added or removed)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/eval"))
from check_e2e_collection import validate_collection  # noqa: E402
from eval_common import EvaluationError  # noqa: E402


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def save(path: Path, records: list[dict]) -> None:
    for sequence, record in enumerate(records):
        if "sequence" in record:
            record["sequence"] = sequence
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))


def first(records: list[dict], **fields) -> int:
    return next(i for i, r in enumerate(records) if all(r.get(k) == v for k, v in fields.items()))


def admitted(paired: dict) -> dict:
    return {**paired, "admitted": True, "route": "PAIR_LOOP_TAIL", "fallback_reason": None}


def fallback(paired: dict) -> dict:
    return {**paired, "admitted": False, "route": "COMPACT_FALLBACK", "fallback_reason": "CAPACITY_NONFIT"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--im2p", type=Path, required=True)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    trace0, cycle0 = load(args.fixture / "npu-cycle-trace.jsonl"), load(args.fixture / "cycle-log.jsonl")
    works = [i for i, r in enumerate(trace0) if r.get("kind") == "NPU_WORK"]
    paired_admitted = next(i for i in works if (trace0[i].get("paired") or {}).get("admitted") is True)
    paired_nonfit = next(i for i in works if (trace0[i].get("paired") or {}).get("admitted") is False)
    # A residual child of the compact parent: a compact stripe may lose it, so moving it leaves only an orphan.
    compact_child = next(i for i in works if trace0[i]["scope"] == "residual_compact" and
                         trace0[i]["operation_id"] != trace0[paired_nonfit]["operation_id"])
    canonical = [i for i, r in enumerate(cycle0) if r.get("duration_role") == "POTAL_HOST"]

    def edit(index: int, **fields):
        return lambda t, c: t.__setitem__(index, {**t[index], **fields})

    def drop(predicate):
        return lambda t, c: t.__delitem__(next(i for i, r in enumerate(t) if predicate(r)))

    def duplicate(predicate):
        return lambda t, c: t.insert(next(i for i, r in enumerate(t) if predicate(r)) + 1,
                                     dict(next(r for r in t if predicate(r))))

    def overlap(t, c):
        r = c[canonical[3]]
        c[canonical[3]] = {**r, "host_end_ns": r["host_end_ns"] + 10**7, "host_elapsed_ns": r["host_elapsed_ns"] + 10**7}

    stripe_call = trace0[paired_admitted]["call_id"]
    cases = [
        ("duplicate work identity", edit(works[2], work_id=trace0[works[1]]["work_id"]), "NPU trace validation failed"),
        ("duplicate host-stage declaration",
         duplicate(lambda r: r.get("kind") == "HOST_STAGE" and r.get("event") == "BEGIN"), "host-stage check failed"),
        ("missing host-stage declaration",
         drop(lambda r: r.get("kind") == "HOST_STAGE" and r.get("event") == "BEGIN" and
              r.get("execution_class") == "POTAL_HOST"), "host-stage check failed"),
        ("missing call of a work",
         drop(lambda r: r.get("kind") == "NPU_CALL" and r.get("call_id") == stripe_call and r.get("stage") == "INVOKE"),
         "NPU trace validation failed"),
        ("canonical CPU overlap on one thread", overlap, "canonical PoTal host intervals overlap on one thread"),
        ("invalid geometry", edit(works[0], m=0), "NPU trace validation failed"),
        ("missing stripe publication",
         drop(lambda r: r.get("kind") == "NPU_CALL" and r.get("call_id") == stripe_call and r.get("stage") == "PUBLISH"),
         "NPU trace validation failed"),
        ("paired route inconsistent with admission",
         edit(paired_admitted, paired={**trace0[paired_admitted]["paired"], "route": "COMPACT_FALLBACK"}),
         "paired route/fallback reason inconsistent with admission"),
        ("admitted stripe owns a residual child",
         edit(paired_nonfit, paired=admitted(trace0[paired_nonfit]["paired"])),
         "admitted PAIR_LOOP_TAIL stripe owns a residual child work"),
        ("fallback stripe without its residual child",
         edit(paired_admitted, paired=fallback(trace0[paired_admitted]["paired"])),
         "CAPACITY_NONFIT fallback stripe without exactly one residual child work"),
        ("residual child without its stripe", edit(compact_child, stripe_id=97), "residual child work without its stripe"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        clean = root / "clean"
        clean.mkdir()
        shutil.copy(args.fixture / "npu-cycle-trace.jsonl", clean)
        shutil.copy(args.fixture / "cycle-log.jsonl", clean)
        result = validate_collection(clean, args.im2p.resolve())
        print("ok unmodified fixture passes:", json.dumps(result["lifecycle"], sort_keys=True))
        for index, (name, mutate, detail) in enumerate(cases):
            native = root / f"case-{index}"
            native.mkdir()
            trace, cycle = [dict(r) for r in trace0], [dict(r) for r in cycle0]
            mutate(trace, cycle)
            save(native / "npu-cycle-trace.jsonl", trace)
            save(native / "cycle-log.jsonl", cycle)
            try:
                validate_collection(native, args.im2p.resolve())
            except EvaluationError as error:
                if detail not in str(error):
                    raise SystemExit(f"FAIL {name}: rejected for another reason: {error}")
                print(f"ok {name}: {error}")
                continue
            raise SystemExit(f"FAIL {name}: accepted")
    print("E2E_COLLECTION_VALIDATION_MUTATIONS_PASS")


if __name__ == "__main__":
    main()
