#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    trace, cycle = map(Path, sys.argv[1:3])
    expected: dict[int, str] = {}
    completed: set[int] = set()
    run_success = False
    for record in rows(trace):
        if record.get("kind") == "RUN_END":
            run_success = record.get("status") == "success"
        if record.get("kind") != "HOST_STAGE" or record.get("execution_class") != "POTAL_HOST":
            continue
        identity = record.get("host_stage_id")
        stage = record.get("stage_name")
        if not isinstance(identity, int) or not isinstance(stage, str):
            raise ValueError("invalid PoTal host-stage identity")
        if record.get("event") == "BEGIN":
            if identity in expected:
                raise ValueError("duplicate PoTal host-stage declaration")
            expected[identity] = stage
        elif record.get("event") == "END" and record.get("status") == "success":
            completed.add(identity)
    measured: dict[int, str] = {}
    for record in rows(cycle):
        if record.get("duration_role") != "POTAL_HOST":
            continue
        identity = record.get("host_stage_id")
        stage = record.get("op")
        if (not isinstance(identity, int) or not isinstance(stage, str) or
                record.get("interval_class") != "CANONICAL_ADDITIVE"):
            raise ValueError("invalid canonical PoTal host measurement")
        if "additive" in record and record["additive"] is not True:
            raise ValueError("canonical PoTal host measurement is non-additive")
        if identity in measured:
            raise ValueError("duplicate canonical PoTal host measurement")
        if identity not in expected or expected[identity] != stage:
            raise ValueError("undeclared or mismatched canonical PoTal host measurement")
        measured[identity] = stage
    if run_success and (set(expected) != completed or expected != measured):
        raise ValueError("PoTal host declaration/completion/measurement mismatch")
    print(f"POTAL_CANONICAL_HOST_INTERVALS_PASS count={len(expected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
