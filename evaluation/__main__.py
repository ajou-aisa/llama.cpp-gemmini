# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run --no-project --offline python -m evaluation --help
"""Reduce a native measurement stream against its exact evaluation manifest."""

import argparse
import sys
from pathlib import Path

from evaluation import activation, residual, weight_alignment
from evaluation.manifest import Manifest
from scripts.eval.eval_common import EvaluationError, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", choices=("activation", "residual", "scale"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = Manifest.load(args.manifest)
        reducer, filename = {
            "activation": (activation.reduce, "activation_metrics.json"),
            "residual": (residual.reduce, "residual_metrics.json"),
            "scale": (weight_alignment.reduce, "scale_alignment_metrics.json"),
        }[args.metric]
        result = reducer(args.input, manifest)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / filename, result)
    except (EvaluationError, OSError, ValueError) as error:
        print("evaluation: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
