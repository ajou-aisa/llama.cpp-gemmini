# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# How to run: use the existing gptc Python, without installing dependencies.
# python -B potal-full-inference-compare.py --help
"""Quantify captured full-inference logits without inventing an accuracy tolerance."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
from numpy.typing import NDArray

RELATIVE_FLOOR: Final = 1e-6
RMS_FLOOR: Final = 1e-12


class RequestRecord(TypedDict, total=False):
    """Fields shared by the native runner and independent FP32 reference."""

    prompt_ids: list[int]
    generated_ids: list[int]
    vocab_size: int
    forced_inputs: bool
    forced_token_ids: list[int]


@dataclass(frozen=True, slots=True)
class ComparisonInputError(ValueError):
    """A malformed capture rather than a failed numerical tolerance."""

    path: Path
    field: str

    def __str__(self) -> str:
        return f"invalid {self.field}: {self.path}"


@dataclass(frozen=True, slots=True)
class Capture:
    """One immutable token trajectory and its final-position logit rows."""

    prefix: Path
    prompt: tuple[int, ...]
    selected: tuple[int, ...]
    inputs: tuple[int, ...]
    logits: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Errors:
    max_absolute: float
    rmse: float
    max_relative: float
    mean_relative: float
    normalized_rmse: float


class Arguments(argparse.Namespace):
    """Mutable only while the project-style CLI parser fills its fields."""

    reference: Path
    candidate: Path
    output: Path


def load(prefix: Path) -> Capture:
    """Parse the runner's capture schema and validate binary extents and finiteness."""
    record: RequestRecord = json.loads(Path(f"{prefix}.json").read_text())
    prompt = tuple(record["prompt_ids"])
    selected = tuple(record["generated_ids"])
    vocabulary = record["vocab_size"]
    if vocabulary != 50257 or not prompt or len(selected) < 2:
        raise ComparisonInputError(prefix, "GPT2 capture dimensions")
    if any(type(token) is not int or not 0 <= token < vocabulary for token in prompt + selected):
        raise ComparisonInputError(prefix, "token IDs")
    trajectory = selected
    if record.get("forced_inputs", False):
        trajectory = tuple(record.get("forced_token_ids", []))
        if len(trajectory) < len(selected) - 1:
            raise ComparisonInputError(prefix, "recorded forced trajectory")
    with Path(f"{prefix}.logits.f32").open("rb") as stream:
        values: NDArray[np.float32] = np.fromfile(stream, dtype="<f4")
    if values.size != len(selected) * vocabulary or not bool(np.isfinite(values).all()):
        raise ComparisonInputError(prefix, "finite binary32 logit extent")
    values = values.reshape(len(selected), vocabulary)
    values.setflags(write=False)
    return Capture(prefix, prompt, selected, trajectory, values)


def errors(reference: NDArray[np.float32], candidate: NDArray[np.float32]) -> Errors:
    """Use float64 reductions; denominator floors define metrics, not PASS thresholds."""
    ref = reference.astype(np.float64)
    difference = candidate.astype(np.float64) - ref
    absolute = np.abs(difference)
    relative = absolute / np.maximum(np.abs(ref), RELATIVE_FLOOR)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_rms = float(np.sqrt(np.mean(ref * ref)))
    return Errors(float(np.max(absolute)), rmse, float(np.max(relative)),
                  float(np.mean(relative)), rmse / max(reference_rms, RMS_FLOOR))


def main() -> None:
    """Compare every step and separately aggregate steps with identical histories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(namespace=Arguments())
    reference, candidate = load(arguments.reference), load(arguments.candidate)
    if reference.prompt != candidate.prompt or reference.logits.shape != candidate.logits.shape:
        raise ComparisonInputError(arguments.candidate, "matching prompt and capture shape")
    per_step = []
    aligned = []
    for step in range(len(reference.selected)):
        same_history = reference.inputs[:step] == candidate.inputs[:step]
        aligned.append(same_history)
        per_step.append({
            "step": step, "same_history": same_history,
            "reference_token": reference.selected[step], "candidate_token": candidate.selected[step],
            "top1_policy_agreement": reference.selected[step] == candidate.selected[step],
            "raw_top1_agreement": int(np.argmax(reference.logits[step])) == int(np.argmax(candidate.logits[step])),
            "bitwise_equal": bool(np.array_equal(reference.logits[step].view(np.uint32),
                                                candidate.logits[step].view(np.uint32))),
            **asdict(errors(reference.logits[step], candidate.logits[step])),
        })
    mismatch = next((step for step in range(len(reference.selected))
                     if reference.selected[step] != candidate.selected[step]), None)
    mask = np.asarray(aligned, dtype=np.bool_)
    result = {
        "reference": str(reference.prefix), "candidate": str(candidate.prefix),
        "prompt_tokens": len(reference.prompt), "generated_tokens": len(reference.selected),
        "finite": True, "reference_nan_inf": 0, "candidate_nan_inf": 0,
        "relative_denominator": f"max(abs(reference), {RELATIVE_FLOOR})",
        "normalized_rmse_denominator": f"max(RMS(reference), {RMS_FLOOR})",
        "acceptance_tolerance": None,
        "first_token_mismatch": mismatch,
        "reference_generated_ids": reference.selected, "candidate_generated_ids": candidate.selected,
        "top1_policy_agreement": sum(a == b for a, b in zip(reference.selected, candidate.selected)) / len(reference.selected),
        "bitwise_equal_all_logits": bool(np.array_equal(reference.logits.view(np.uint32),
                                                       candidate.logits.view(np.uint32))),
        "matched_history_steps": int(np.count_nonzero(mask)),
        "matched_history_errors": asdict(errors(reference.logits[mask], candidate.logits[mask])),
        "steps": per_step,
    }
    if mismatch is not None:
        ref_token, candidate_token = reference.selected[mismatch], candidate.selected[mismatch]
        result["first_divergence_logits"] = {
            "reference_at_reference_token": float(reference.logits[mismatch, ref_token]),
            "reference_at_candidate_token": float(reference.logits[mismatch, candidate_token]),
            "candidate_at_reference_token": float(candidate.logits[mismatch, ref_token]),
            "candidate_at_candidate_token": float(candidate.logits[mismatch, candidate_token]),
        }
    text = json.dumps(result, indent=2, allow_nan=False)
    arguments.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
