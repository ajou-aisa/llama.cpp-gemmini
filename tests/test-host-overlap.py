#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
# How to run: python3 tests/test-host-overlap.py
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

SCRIPT: Final = Path(__file__).resolve().parents[1] / "scripts/utils/summarize_host_overlap.py"


def main() -> None:
    # Given: integer nanoseconds beyond float precision, nested worker spans,
    # cross-stripe overlap, touching endpoints, and unrelated executions.
    base = 2**60
    records = [
        {"op": "left", "stripe_id": 0, "start": 0, "end": 10},
        {"op": "left", "stripe_id": 1, "start": 4, "end": 12},
        {"op": "left", "stripe_id": 1, "start": 5, "end": 6},
        {"op": "left", "stripe_id": 2, "start": 20, "end": 25},
        {"op": "right", "stripe_id": 3, "start": 8, "end": 22},
        {"op": "right", "stripe_id": 4, "start": 9, "end": 11},
        {"op": "right", "stripe_id": 5, "start": 25, "end": 30},
        {"op": "right-extra", "stripe_id": 6, "start": 40, "end": 45},
        {"op": "left", "start": 0, "end": 10, "execution_id": "other"},
        {"op": "right", "start": 0, "end": 10, "run_id": None},
        {"op": "right", "start": 0, "end": 10, "run_id": -1},
        {"op": "right", "start": 0, "end": 10, "layer": None},
        {"op": "right", "start": 0, "end": 10, "host_valid": False},
        {"op": "left", "start": 0, "end": 10, "run_id": 1},
        {"op": "right", "start": 10, "end": 20, "run_id": 1},
    ]
    encoded = []
    for record in records:
        encoded.append(json.dumps({
            "op": record["op"], "valid": False, "worker_id": 3,
            "run_id": record.get("run_id", 0), "layer": record.get("layer", "blk.2.attn_q"),
            "stripe_id": record.get("stripe_id", 0),
            "host_timing": {
                "execution_id": record.get("execution_id", "first"),
                "clock": "steady_clock", "unit": "nanosecond",
                "start_ns": base + record["start"], "end_ns": base + record["end"],
                "valid": record.get("host_valid", True),
                "start_tid": 42, "end_tid": 42, "thread_id_kind": "linux_tid",
            },
        }))
    encoded += [json.dumps({"op": "left", "start_ns": base, "end_ns": base + 100}),
                json.dumps({"op": "unselected"})]
    with tempfile.TemporaryDirectory(prefix="host-overlap-") as temporary:
        source = Path(temporary) / "main.jsonl"
        detail = Path(temporary) / "detail.jsonl"
        source.write_text("\n".join(encoded[:4]) + "\n", encoding="utf-8")
        detail.write_text("\n".join(encoded[4:]) + "\n", encoding="utf-8")
        command = [sys.executable, str(SCRIPT), str(source), str(detail),
                   "--left-op", "left", "--right-op", "right", "--right-op", "right-extra"]
        # When: both files are analyzed through the public CLI.
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        # Then: union intersection is [8,12) + [20,22) = 6 ns, never 10 ns
        # from summing nested workers; a different execution has no paired side.
        assert result.returncode == 0, result.stderr
        output = json.loads(result.stdout)
        groups = {(item["execution_id"], item["run_id"]): item for item in output["groups"]}
        first = groups[("first", 0)]
        assert (first["left_intervals"], first["right_intervals"]) == (4, 4)
        assert (first["left_union_ns"], first["right_union_ns"], first["overlap_ns"]) == (17, 24, 6)
        assert groups[("first", 1)]["overlap_ns"] == 0
        assert groups[("other", 0)]["overlap_ns"] is None
        assert groups[("other", 0)]["status"] == "missing_side"
        assert output["sources"][1]["skipped"] == {
            "unselected_op": 1, "missing_host_timing": 1,
            "invalid_host_timing": 1, "unknown_identity": 3,
        }

        # Given: a selected valid sidecar uses an incompatible clock.
        source.write_text(encoded[0].replace('"steady_clock"', '"realtime"') + "\n", encoding="utf-8")
        # When: the same CLI reads the incompatible clock.
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        # Then: it rejects the input instead of mixing clock domains.
        assert result.returncode != 0
        assert str(source) + ":1:" in result.stderr
        assert "clock" in result.stderr

        # Given: a JSONL file contains malformed JSON on its second line.
        source.write_text(encoded[0] + "\n{broken\n", encoding="utf-8")
        # When: the same CLI reads the malformed file.
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        # Then: the error identifies the source and line.
        assert result.returncode != 0
        assert str(source) + ":2:" in result.stderr
        assert not result.stdout
    print("host overlap CLI: PASS")


if __name__ == "__main__":
    main()
