#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["datasets==5.1.0", "huggingface_hub==2.4.0"]
# ///
# How to run:
# Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# Download: uv run scripts/experiment/prepare-accuracy.py
# Offline verification: uv run --offline scripts/experiment/prepare-accuracy.py --check
"""Download the seven selected FP baseline datasets and verify their labeled evaluation splits."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import file_digest
import json
import os
from pathlib import Path
import sys
from typing import Final, Literal, TypedDict, assert_never

ROOT: Final = Path(__file__).resolve().parents[2]
OUTPUT: Final = ROOT.parent / "datasets/accuracy"
CHECK: Final = "--check" in sys.argv
type TaskName = Literal["piqa", "hellaswag", "winogrande", "boolq", "arc_easy", "arc_challenge", "openbookqa"]
os.environ["HF_HOME"] = str(OUTPUT / "hf-cache")
os.environ["HF_DATASETS_CACHE"] = str(OUTPUT / "hf-cache/datasets")
os.environ["HF_HUB_CACHE"] = str(OUTPUT / "hf-cache/hub")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
if CHECK:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

from datasets import DatasetDict, get_dataset_config_names, load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError


@dataclass(frozen=True, slots=True)
class Task:
    name: TaskName
    repo: str
    config: str
    split: str
    rows: int
    files: tuple[str, ...] = ("README.md", "*.parquet")


TASKS: Final = (
    Task("piqa", "baber/piqa", "default", "validation", 1838),
    Task("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "test", 1172),
    Task("openbookqa", "allenai/openbookqa", "main", "test", 500),
    Task("boolq", "aps/super_glue", "boolq", "validation", 3270, ("README.md", "boolq/*.parquet")),
    Task("arc_easy", "allenai/ai2_arc", "ARC-Easy", "test", 2376),
    Task("hellaswag", "Rowan/hellaswag", "default", "validation", 10042),
    Task("winogrande", "allenai/winogrande", "winogrande_xl", "validation", 1267),
)


class Receipt(TypedDict):
    name: str
    repo: str
    revision: str
    config: str
    evaluation_split: str
    evaluation_rows: int
    splits: dict[str, int]
    cached_configs: list[str]
    artifacts: dict[str, str]


class DatasetMismatchError(RuntimeError):
    def __init__(self, task: str, reason: str) -> None:
        self.task, self.reason = task, reason
        super().__init__(f"{task}: {reason}")


def require(condition: bool, task: str, reason: str) -> None:
    if not condition:
        raise DatasetMismatchError(task, reason)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def validate(task: Task, data: DatasetDict) -> None:
    """Check complete labeled evaluation splits and each task's answer domain."""
    split = data[task.split]
    require(len(split) == task.rows, task.name, "wrong evaluation population")
    match task.name:
        case "hellaswag":
            valid = all(str(v) in ("0", "1", "2", "3") for v in split["label"])
            valid &= all(len(v) == 4 for v in split["endings"])
        case "winogrande":
            valid = all(v in ("1", "2") for v in split["answer"])
            valid &= all("_" in v for v in split["sentence"])
        case "piqa":
            valid = all(v in (0, 1) for v in split["label"])
            valid &= all(bool(v) for v in split["sol1"]) and all(bool(v) for v in split["sol2"])
        case "boolq":
            valid = all(v in (0, 1) for v in split["label"])
            valid &= all(bool(v) for v in split["passage"])
        case "arc_easy" | "arc_challenge" | "openbookqa":
            valid = all(answer in choices["label"] and len(choices["text"]) == len(choices["label"])
                        and len(choices["text"]) >= 2
                        for answer, choices in zip(split["answerKey"], split["choices"], strict=True))
        case unexpected:
            assert_never(unexpected)
    require(valid, task.name, "missing/invalid labels or choices")


def check(task: Task) -> Receipt:
    """Verify portable hashes and native cache; --check disables network access."""
    directory = OUTPUT / task.name
    receipt: Receipt = json.loads((directory / "manifest.json").read_text())
    require(receipt["repo"] == task.repo and receipt["config"] == task.config,
            task.name, "dataset identity changed")
    require(all(digest(directory / path) == expected for path, expected in receipt["artifacts"].items()),
            task.name, "downloaded artifact changed")
    validate(task, load_from_disk(str(directory / "dataset")))
    for config in receipt["cached_configs"]:
        data = load_dataset(task.repo, config, revision=receipt["revision"])
        require(bool(data), task.name, "missing native loader cache: " + config)
    print(f"PASS {'offline' if CHECK else 'verified'} {task.name}: {task.rows} labeled {task.split} rows", flush=True)
    return receipt


def download(task: Task) -> Receipt:
    """Retain the public source snapshot, native cache, portable Arrow data, and commit identity."""
    directory = OUTPUT / task.name
    if (directory / "manifest.json").exists():
        return check(task)
    print(f"Downloading {task.name}: {task.repo} [{task.config}]", flush=True)
    snapshot = Path(snapshot_download(task.repo, repo_type="dataset", revision="main",
        allow_patterns=list(task.files), max_workers=3))
    revision = snapshot.name
    data = load_dataset(task.repo, task.config, revision=revision)
    validate(task, data)
    directory.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(str(directory / "dataset"))
    configs = [task.config]
    available = get_dataset_config_names(task.repo, revision=revision)
    require(set(configs) <= set(available), task.name, "missing task configurations")
    receipt: Receipt = dict(name=task.name, repo=task.repo, revision=revision, config=task.config,
        evaluation_split=task.split, evaluation_rows=task.rows,
        splits={name: len(split) for name, split in data.items()}, cached_configs=configs,
        artifacts={str(path.relative_to(directory)): digest(path)
                   for path in sorted((directory / "dataset").rglob("*")) if path.is_file()})
    (directory / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"PASS downloaded {task.name}: {task.rows} labeled evaluation rows; commit {revision}", flush=True)
    return receipt


def main() -> None:
    """Try every requested dataset and retain specific failures for manual recovery."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results: list[Receipt] = []
    failures: dict[str, str] = {}
    for task in TASKS:
        try:
            results.append(check(task) if CHECK else download(task))
        except (OSError, ValueError, HfHubHTTPError, DatasetMismatchError) as error:
            failures[task.name] = str(error)
            print(f"FAILED {task.name}: {error}", flush=True)
    report = dict(schema="potal-accuracy-inputs-v1", mode="offline-check" if CHECK else "download",
        ready=len(results), requested=len(TASKS), model_inference_verified=False,
        datasets=results, failures=failures)
    destination = OUTPUT / ("offline-check.json" if CHECK else "readiness.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    require(not failures, "accuracy", str(failures))


if __name__ == "__main__":
    main()
