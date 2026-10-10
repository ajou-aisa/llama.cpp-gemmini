#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==1.26.4", "tokenizers>=0.19,<1"]
# ///
# How to run: PYTHONPATH=gguf-py:. python -m experiments.potal_a4.run MODEL TOKENS CTX CHUNKS MODE OUT.json
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from .model import Forward, Weights
from .types import MODES, Mode, QuantError


@dataclass(frozen=True, slots=True)
class RunInput:
    model: Path
    tokens: Path
    context: int
    chunks: int
    mode: Mode
    output: Path

    @classmethod
    def parse(cls, args: list[str]) -> RunInput:
        if len(args) != 6:
            raise QuantError("Expected MODEL TOKENS CTX CHUNKS MODE OUT.json")
        mode = next((m for m in MODES if m.name == args[4]), None)
        if mode is None:
            raise QuantError(f"Unknown mode: {args[4]}")
        result = cls(Path(args[0]), Path(args[1]), int(args[2]), int(args[3]), mode, Path(args[5]))
        if result.context < 32 or result.context > 1024 or result.context % 32 or result.chunks < 1 or result.output.exists():
            raise QuantError("Require context 32..1024, positive chunks and a new output file")
        return result


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run(config: RunInput) -> None:
    started = time.perf_counter()
    tokens = np.fromfile(config.tokens, dtype="<i4")
    if len(tokens) < config.context * config.chunks:
        raise QuantError("Not enough token data")
    weights = Weights(config.model)
    model_loaded_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    forward = Forward(weights, config.mode)
    losses: list[float] = []
    correct: list[int] = []
    scored = config.context // 2 - 1
    for chunk in range(config.chunks):
        batch = tokens[chunk * config.context:(chunk + 1) * config.context]
        hidden = forward.run(batch)
        selected = hidden[config.context // 2:]
        logits = forward.logits(selected)[:-1].astype(np.float64)
        labels = batch[config.context // 2 + 1:]
        maximum = logits.max(axis=1)
        nll = np.log(np.exp(logits - maximum[:, None]).sum(axis=1)) + maximum - logits[np.arange(scored), labels]
        if not np.isfinite(nll).all():
            raise QuantError("Nonfinite NLL")
        losses.append(float(nll.sum()))
        correct.append(int(np.count_nonzero(logits.argmax(axis=1) == labels)))
        print(json.dumps({"mode": config.mode.name, "context": config.context, "chunk": chunk,
                          "ppl": float(np.exp(nll.mean())), "seconds": time.perf_counter() - started}), flush=True)
        np.savez_compressed(config.output.with_suffix(f".chunk{chunk}.npz"),
                            nll=nll, top1=logits.argmax(axis=1), hidden=hidden, labels=labels)
    report = {
        "scope": "CPU NumPy policy experiment with materialized int8 compact packets; not native Gemmini execution or timing",
        "mode": asdict(config.mode), "model": str(config.model), "model_sha256": digest(config.model),
        "tokens_sha256": digest(config.tokens), "context": config.context, "chunks": config.chunks,
        "scored_tokens": scored * config.chunks, "nll_sums": losses,
        "ppl": float(np.exp(sum(losses) / (scored * config.chunks))),
        "top1_accuracy": sum(correct) / (scored * config.chunks),
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "loaded_model_peak_rss_bytes": model_loaded_rss,
        "rss_units": "bytes on macOS; total process high-water mark, not native buffer savings",
        "metrics": asdict(forward.metrics),
        "numpy": np.__version__,
        "head_rows_per_chunk": config.context // 2,
        "linear_scope": "Only stored Q4_HP1 tensors; F16 embedding and tied output head preserved",
        "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    }
    config.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": str(config.output), "ppl": report["ppl"]}), flush=True)


if __name__ == "__main__":
    run(RunInput.parse(sys.argv[1:]))
