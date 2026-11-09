#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["lm_eval[api]==0.4.13", "datasets==5.1.0", "huggingface_hub==2.4.0"]
# ///
# How to run:
# Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# Warm task/metric caches: uv run scripts/experiment/check-accuracy-tasks.py
# Offline check: uv run --offline scripts/experiment/check-accuracy-tasks.py --offline
"""Construct real harness requests for every requested task without model inference."""
from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from typing import Final, TypedDict

ROOT: Final = Path(__file__).resolve().parents[2]
OUTPUT: Final = ROOT.parent / "datasets/accuracy"
OFFLINE: Final = "--offline" in sys.argv
os.environ["HF_HOME"] = str(OUTPUT / "hf-cache")
os.environ["HF_DATASETS_CACHE"] = str(OUTPUT / "hf-cache/datasets")
os.environ["HF_HUB_CACHE"] = str(OUTPUT / "hf-cache/hub")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

from lm_eval.tasks import TaskManager


class TaskResult(TypedDict):
    name: str
    evaluation_rows: int
    fewshot: int
    constructed_requests: int
    request_types: list[str]
    first_context_sha256: str


class TaskReadinessError(RuntimeError):
    def __init__(self, task: str, reason: str) -> None:
        self.task, self.reason = task, reason
        super().__init__(f"{task}: {reason}")


def main() -> None:
    """Construct real zero-shot requests for all seven selected FP baseline tasks."""
    tasks = TaskManager().load(["piqa", "hellaswag", "winogrande", "boolq", "arc_easy",
                                "arc_challenge", "openbookqa"])["tasks"]
    if len(tasks) != 7:
        raise TaskReadinessError("task-matrix", f"expected 7 leaf tasks, got {len(tasks)}")
    results: list[TaskResult] = []
    for name, task in sorted(tasks.items()):
        fewshot = 0
        task.set_config("num_fewshot", fewshot)
        task.set_fewshot_seed(1234)
        task.build_all_requests(limit=1, rank=0, world_size=1)
        if not task.instances:
            raise TaskReadinessError(name, "no actual requests constructed")
        results.append(dict(name=name, evaluation_rows=len(task.eval_docs), fewshot=fewshot,
            constructed_requests=len(task.instances),
            request_types=sorted({request.request_type for request in task.instances}),
            first_context_sha256=sha256(str(task.instances[0].args[0]).encode()).hexdigest()))
    expected = {"piqa": 1838, "hellaswag": 10042, "winogrande": 1267, "boolq": 3270,
                "arc_easy": 2376, "arc_challenge": 1172, "openbookqa": 500}
    actual = {row["name"]: row["evaluation_rows"] for row in results}
    if any(actual.get(name) != count for name, count in expected.items()):
        raise TaskReadinessError("task-matrix", "evaluation population differs from downloaded data")
    report = dict(schema="potal-harness-readiness-v1", harness_version=version("lm_eval"),
        offline=OFFLINE, validated_leaf_tasks=len(results), model_inference_verified=False,
        model_scores_measured=False, tasks=results)
    path = OUTPUT / ("harness-offline.json" if OFFLINE else "harness-online.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(results)} actual task/metric/prompt request constructions; model inference not run.")


if __name__ == "__main__":
    main()
