# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch==2.5.1", "transformers==4.46.1"]
# ///
# How to run: use the existing gptc Python, without installing dependencies.
# python -B potal-full-inference-reference.py --help
"""Independent cached-checkpoint FP32 reference for H3-7A logit capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NewType, TypedDict

import numpy as np
import torch
import transformers
from transformers import GPT2LMHeadModel

TokenId = NewType("TokenId", int)
VOCAB_SIZE: Final = 50257
PARAMETERS: Final = 124439808
EOS: Final = 50256


class PromptRecord(TypedDict):
    """The prompt-ID field emitted by the real llama acceptance runner."""

    prompt_ids: list[int]


@dataclass(frozen=True, slots=True)
class ReferenceInputError(ValueError):
    """An invalid reference fixture, not a numerical acceptance threshold."""

    path: Path
    field: str

    def __str__(self) -> str:
        return f"invalid {self.field}: {self.path}"


class Arguments(argparse.Namespace):
    """Mutable only while the existing project-style CLI parser fills it."""

    checkpoint: Path
    request_json: Path
    output: Path
    generate: int
    force_tokens: Path | None = None


def sha256(path: Path) -> str:
    """Hash source bytes without mutating the cached checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Run prefill and cached autoregressive FP32 decode, then save raw logits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate", type=int, default=8)
    parser.add_argument("--force-tokens", type=Path)
    arguments = parser.parse_args(namespace=Arguments())
    record: PromptRecord = json.loads(arguments.request_json.read_text())
    prompt = tuple(TokenId(token) for token in record["prompt_ids"])
    if not 1 <= len(prompt) <= 256 or not 2 <= arguments.generate <= 32:
        raise ReferenceInputError(arguments.request_json, "workload dimensions")
    if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in prompt):
        raise ReferenceInputError(arguments.request_json, "prompt token IDs")
    forced: tuple[TokenId, ...] = ()
    if arguments.force_tokens is not None:
        forced = tuple(TokenId(int(line)) for line in arguments.force_tokens.read_text().splitlines())
        if len(forced) < arguments.generate - 1 or any(not 0 <= token < VOCAB_SIZE for token in forced):
            raise ReferenceInputError(arguments.force_tokens, "forced trajectory")

    torch.set_num_threads(3)
    model = GPT2LMHeadModel.from_pretrained(
        str(arguments.checkpoint), local_files_only=True
    ).to(device="cpu", dtype=torch.float32).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != PARAMETERS or model.config.vocab_size != VOCAB_SIZE:
        raise ReferenceInputError(arguments.checkpoint, "GPT2 checkpoint identity")
    logits = np.empty((arguments.generate, VOCAB_SIZE), dtype=np.float32)
    generated: list[TokenId] = []
    steps: list[dict[str, int | list[int]]] = []
    inputs = torch.tensor([prompt], dtype=torch.long, device="cpu")
    past = None
    with torch.inference_mode():
        for step in range(arguments.generate):
            result = model(input_ids=inputs, past_key_values=past, use_cache=True, return_dict=True)
            row = result.logits[0, -1].cpu().numpy().copy()
            if row.shape != (VOCAB_SIZE,) or not bool(np.isfinite(row).all()):
                raise ReferenceInputError(arguments.checkpoint, f"finite logits at step {step}")
            logits[step] = row
            policy = row.copy()
            policy[EOS] = -np.inf
            token = TokenId(int(np.argmax(policy)))
            generated.append(token)
            past = result.past_key_values
            cached = int(past[0][0].shape[-2])
            if cached != len(prompt) + step:
                raise ReferenceInputError(arguments.checkpoint, f"KV cache length at step {step}")
            steps.append({"step": step, "kv_tokens": cached, "selected": token,
                          "input_ids": inputs[0].tolist()})
            print(f"H7_FP32_TOKEN step={step} token_id={token}", flush=True)
            next_input = forced[step] if forced and step + 1 < arguments.generate else token
            inputs = torch.tensor([[next_input]], dtype=torch.long, device="cpu")

    summary = {
        "schema": "h3-7a.request.v1",
        "backend": "TRANSFORMERS_CPU_FP32",
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": sha256(arguments.checkpoint / "model.safetensors"),
        "parameters": parameter_count,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "weight_dtype": "float32",
        "kv_dtype": "float32",
        "attention_implementation": model.config._attn_implementation,
        "threads": 3,
        "prompt_ids": prompt,
        "generated_ids": generated,
        "forced_inputs": bool(forced),
        "forced_token_ids": forced,
        "ignore_eos": True,
        "sampling": "greedy_lowest_id_tie",
        "vocab_size": VOCAB_SIZE,
        "steps": steps,
    }
    with Path(f"{arguments.output}.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    with Path(f"{arguments.output}.logits.f32").open("wb") as stream:
        logits.astype("<f4", copy=False).tofile(stream)
    with Path(f"{arguments.output}.tokens").open("w") as stream:
        stream.writelines(f"{token}\n" for token in generated)
    print(f"H7_FP32_REFERENCE_COMPLETE prompt={len(prompt)} generated={len(generated)}", flush=True)


if __name__ == "__main__":
    main()
