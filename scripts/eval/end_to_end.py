#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B scripts/eval/end_to_end.py --help
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import subprocess
import sys

from application_results import aggregate_results, load_measurement
from eval_common import EvaluationError, require, write_json
from offline_pipeline import add_arguments, reconstruct


def main() -> int:
    parser = argparse.ArgumentParser(description="Metric-OFF native E2E collection and verified offline reconstruction.")
    parser.add_argument("--output", type=Path, help="fresh output directory (required by each command)")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="actual 256+128 host measurements; no warmup")
    collect.add_argument("--runner", type=Path, required=True)
    collect.add_argument("--im2p", type=Path, help="IM2P source root for official FullCPU/PoTal build and source provenance")
    collect.add_argument("--model", type=Path, required=True)
    collect.add_argument("--dataset", type=Path, required=True)
    collect.add_argument("--settings", type=Path, required=True, help="explicit seed, temperature, threads and prompt mapping")
    collect.add_argument("--role", choices=("fullcpu", "fullcpu-cost-only", "potal", "cuda"), required=True)
    collect.add_argument("--paired-potal", type=Path, help="completed PoTal collection root supplying forced CPU cost-only token trajectories")
    collect.add_argument("--repetitions", type=int, default=10, help="1..10 actual runs; fewer than ten is smoke only")
    collect.add_argument("--timeout", type=int, default=600)
    aggregate = sub.add_parser("aggregate", help="median of ten real per-run results, never replay duplicates")
    aggregate.add_argument("--result", type=Path, action="append", required=True)
    offline = sub.add_parser("reconstruct", help="official certified replay, source join, execution IR; no inferred latency")
    add_arguments(offline)
    try:
        args = parser.parse_args()
        require(args.output is not None, "--output is required")
        if args.command == "aggregate":
            result = aggregate_results([load_measurement(path) for path in args.result])
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "result.json", result)
        elif args.command == "reconstruct":
            reconstruct(args)
        else:
            from e2e_run import collect_runs
            collect_runs(args)
        return 0
    except (EvaluationError, OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as error:
        print(f"evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
