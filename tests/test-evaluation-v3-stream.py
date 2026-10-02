#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B tests/test-evaluation-v3-stream.py --help
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Real dedicated metric stream mutation checks; source remains unchanged.")
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.activation.read_bytes()
    rows = [json.loads(line) for line in source.splitlines()]
    assert rows[0]["kind"] == "RUN" and rows[-1]["kind"] == "RUN_END"
    command = Path(__file__).resolve().parents[1] / "scripts/eval/activation_quant_metrics.py"
    duplicate = [*rows[:-1], dict(rows[1]), dict(rows[-1])]
    for index, row in enumerate(duplicate):
        row = dict(row)
        row["sequence"] = index
        duplicate[index] = row
    mutations = {"missing-run-end": rows[:-1], "duplicate-counts": duplicate,
                 "failed-run-end": [*rows[:-1], {**rows[-1], "success": False}]}
    args.output.mkdir(parents=True, exist_ok=False)
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="evaluation-v3-stream-") as temporary:
        root = Path(temporary)
        for name, mutation in mutations.items():
            path, output = root / (name + ".jsonl"), root / name
            path.write_text("".join(json.dumps(row) + "\n" for row in mutation))
            argv = [sys.executable, "-B", str(command), "--reduce", str(path), "--output", str(output)]
            result = subprocess.run(argv, text=True, capture_output=True, timeout=15)
            (args.output / (name + ".log")).write_text(result.stdout + result.stderr)
            assert result.returncode == 1, (name, result.stderr)
            assert not (output / "activation-quant-summary.json").exists(), name
            outcomes.append({"case": name, "exit_code": result.returncode, "summary_published": False,
                             "stderr": result.stderr.strip(), "argv": argv})
    assert args.activation.read_bytes() == source
    (args.output / "result.json").write_text(json.dumps({"status": "PASS", "source": str(args.activation),
        "source_sha256": hashlib.sha256(source).hexdigest(), "cases": outcomes,
        "temporary_artifacts_cleaned": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
