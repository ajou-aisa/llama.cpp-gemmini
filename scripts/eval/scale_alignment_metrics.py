#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run --no-project --offline scripts/eval/scale_alignment_metrics.py --help
from metric_run import main

if __name__ == "__main__":
    raise SystemExit(main("scale"))
