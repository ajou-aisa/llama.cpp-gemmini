# Minimal CUDA and native accuracy package

Use a complete checkout of this branch, not individual CUDA files. Build fresh
binaries on the destination machine. Model GGUFs, datasets, CUDA toolkit and
Gemmini software headers are separate inputs.

## Build the backends

Run from the repository root on Linux or CUDA-capable WSL:

```bash
export GEMMINI_SW_PATH=/path/to/RISC-V-DynDNN-gemmini-include
export CUDAARCHS=native
bash scripts/experiment/build-potal-cuda.sh --rtn --jobs 2 --build-only
for bits in 4 8; do
    bash scripts/experiment/build-potal-cuda.sh \
        --bits "$bits" --anchor 3 --attention --jobs 2 --build-only
done
bash scripts/experiment/build-nvfp4-reference.sh \
    "$PWD/build-potal-cuda/nvfp4-fp32" fp32
```

`--build-only` compiles without CUDA tests or model inference. See
`GEMMINI-CUDA-PPL.md` for the existing numerical checks, profile environments
and PPL commands. The WikiText-2 recipe
is unchanged. The NVFP4 helper only builds; FP32 and FP16 reconstruction modes
remain different numerical profiles.

## Build the native accuracy frontends

```bash
bash scripts/experiment/build-fp16-accuracy-cuda.sh
bash scripts/experiment/build-quantized-accuracy.sh
bash scripts/experiment/build-accuracy-linear.sh
```

Defaults use the backend build directories above. For other locations, set
`ACCURACY_RTN_BUILD`, `ACCURACY_POTAL4_BUILD` and `ACCURACY_POTAL8_BUILD`.
Output overrides are `ACCURACY_FP16_OUTPUT_DIR`, `ACCURACY_OUTPUT_DIR` and
`ACCURACY_LINEAR_OUTPUT_DIR`. Compilation does not load a model or run CUDA.

The implemented native interfaces are:

| Executable under `build-potal-cuda/` | Positional arguments |
| --- | --- |
| `accuracy-fp16/native-score-fp16-cuda` | `MODEL THREADS BATCH` |
| `accuracy-quantized/native-score-rtn` | `MODEL THREADS BATCH PROFILE` |
| `accuracy-quantized/native-score-potal4` | `MODEL THREADS BATCH potal` |
| `accuracy-quantized/native-score-potal8` | `MODEL THREADS BATCH potal` |
| `accuracy-linear/native-score-potal4` | `MODEL` |
| `accuracy-linear/native-score-potal8` | `MODEL` |

Apply the matching numerical profile environment and verified checkpoint, as
documented in the CUDA PPL guide and rounding policy. The general quantized frontend
supports `rtn`, `mxint`, `potal` and `nvfp4-public`. Do not infer activation
precision or reconstruction mode from an executable filename alone.

Each process loads one model and accepts consecutive JSON lines on stdin:

```json
{"context":"The capital of France is","continuation":" Paris."}
```

It emits a readiness message followed by request likelihoods, token boundaries
and backend counters. EOF ends the process and emits completion proofs on stderr.
Use the established schedule: six threads, batch 512, context limit 4096,
independent requests, no prefix sharing or mixed-question batching. The LINEAR
frontend fixes its own six-thread/512 schedule. Do not silently truncate inputs.
These frontends calculate likelihoods; they do not themselves generate task
prompts or publish task-level accuracy.

## Prepare the official task inputs

```bash
uv run --python 3.12 scripts/experiment/prepare-accuracy.py
uv run --python 3.12 scripts/experiment/check-accuracy-tasks.py --offline
```

The first command downloads and verifies PIQA, HellaSwag, WinoGrande, ARC-Easy,
ARC-Challenge, BoolQ and OpenBookQA under `../datasets/accuracy`. The second
constructs one real zero-shot request group per task with the pinned harness;
it does not score models. Preserve that harness's task prompts, continuation
boundary handling, `process_results` and aggregation when connecting the native
frontends. This package does not implement the newly requested token/response
cache runner or claim a complete seven-task native evaluation.

The build helpers have been exercised against existing backend builds.
The selected images had 16/20 complete model equivalence pairs at handoff;
the remaining Llama pairs and qualified model performance are still pending.
This is not validation on a second GPU or a new full-dataset accuracy result.
