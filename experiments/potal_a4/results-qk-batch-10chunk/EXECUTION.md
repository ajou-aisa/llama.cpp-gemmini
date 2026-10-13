# Native batching experiment (local Apple M5)

Worktree: `/Users/chan/Projects/AISA/llama.cpp-gemmini/.worktrees/potal-qk-mask`.
The branch is `work/potal-qk-mask-local`, based on Nano commit `63e166d`.
The parent develop checkout is not modified by this experiment.

## Build and operator checks

The existing Release replay build links `build-potal-qk/bin`. Its selected
configuration is A4/W4/DIM32, CPU execution, with Metal/CUDA disabled. The
attention quantization geometry remains `(4,4096,512)`; the product header's
ACC_ROWS1024 is not substituted for that logical geometry.

```sh
rtk proxy cmake --build build-potal-replay -j 4
rtk proxy ctest --test-dir build-potal-replay --output-on-failure
```

CTest: 7/7 passed. The additional native-batch test covers distinct column and
lane shifts, `2048 * 2^21 = 4294967296` without int32 saturation, rejected
overflow/row/region inputs with the public output unchanged, rejected narrowing
of 256 to int8, reordered residual rows, causal/irregular/finite/all-blocked
masks, tails/decode, and PV zero-point cancellation. Both batch variants run
against the same independent reconstructed-operand GEMM checks.

## Initial component comparison

```sh
rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.native_measure batch-components build-potal-replay/libpotal-native-replay.dylib experiments/potal_a4/results-qk-mask-10chunk experiments/potal_a4/batch-components-initial.json
```

The output was moved without changing its contents to
`components-initial.json` in this directory. It compares the same 15 QK and 15
PV captures, one warmup plus five rotated-order measurements per mode. The
sum of per-capture medians is the primary aggregation. Common quantization is
outside timing; mask classification/planning, descriptor construction, int8
packing, validation, native calls, raw dot, wide merge and output publication
are inside timing. There was no concurrent build/model experiment.

All results match the independent NumPy int64 GEMM exactly (valid QK entries,
all PV entries), and float32 publication matches the saved captures. All
masked variants have identical DIM32 fragment counts and raw work regions.

| Component | Dense ms | Existing QK skip ms | Batched full partial ms | Batched direct ms |
| --- | ---: | ---: | ---: | ---: |
| QK, 15 captures | 384.890166 | 404.401916 | 353.782417 | 319.964000 |
| PV, 15 captures | 161.412624 | 162.436877 | 174.198086 | 173.078210 |

QK FFI submissions: 1440 dense, 2880 existing skip, 120 for either batch.
The batches still execute 2880 retained raw regions. PV submissions drop from
1660 to 60, but its timings regress; fewer calls alone are not sufficient.
The full-partial batch initializes 188,612,608 bytes of QK int32 partial arrays
across these captures; direct accumulation removes those full-width partials.
This is cumulative buffer initialization, not peak RAM or memory-bus traffic.

The batch workspace counter sums described buffer sizes (shared aliases may
be counted more than once), descriptors, and native private accumulator/scratch
capacity. It is not process peak memory and is not directly comparable with the
legacy counter, which measures only sparse workspace.

## Paired 10-chunk model replay and final component comparison

Completed successfully: both PPL values are **13.801307254545433**, with exact
packet/scale/valid-QK/softmax/PV/logit/token-NLL equality. All 10 dense records
also match the previous QK experiment's dense records exactly. All five real
mask/cache scenarios pass in both modes. The final component medians are:

| Component | Dense ms | Existing QK skip ms | Batched full partial ms | Batched direct ms |
| --- | ---: | ---: | ---: | ---: |
| QK, 15 captures | 399.586459 | 397.461962 | 368.414042 | 326.162122 |
| PV, 15 captures | 160.280377 | 161.176461 | 171.007750 | 175.863913 |

The final comparison ran after model QA, without concurrent build or model
execution. The full-partial native control includes moving SCU to native code;
its difference from existing skip does not isolate FFI cost alone. QK direct
reduces time 17.94% versus existing skip; PV direct increases time 9.72% versus
dense. See `component-summary.json` for counts, ranges and artifact identities.
The chunk-record field `max_sparse_workspace_bytes` retains its legacy name;
for `batch_direct` it holds the batch buffer-size upper bound described above.

```sh
rtk proxy env PYTHONPATH=gguf-py:. OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=2 /Users/chan/Projects/AISA/llama.cpp-gemmini/.venv-convert311/bin/python -m experiments.potal_a4.native_measure qk-batch build-potal-replay/libpotal-native-replay.dylib /Users/chan/Projects/AISA/llama.cpp-gemmini/models/llama3.2-1B.Q4_HP1.gguf /Users/chan/Projects/AISA/llama.cpp-gemmini/wikitext-2-raw/wiki.test.raw 512 10 experiments/potal_a4/results-qk-batch-10chunk
```

This command requires a new output directory. It compares dense CPU attention
with batched direct QK/PV; only QK skips fully masked tiles. Both use all original
digits, K nibble passes, per-block/per-column exponents, and full PV zero-point
compensation. `chunks.jsonl` records every chunk's exact checks and NLL values;
`report.json` contains the completed run, model/token/library/source hashes, and
the fresh four-mode component comparison. Five additional real graph scenarios
cover tail prefill, prefix, decode, cached suffix, and packed sequences.

The native model still computes its original floating QK/PV before the callback
replaces outputs. Replay wall time is diagnostic, not an integrated backend
speedup. The native batch accepts int8 signed nibble operands and uses int32 raw
dots, int64 accumulators, and checked int128 intermediate scaling. It rejects
int64 overflow; arbitrary-width exact accumulation is not implemented here.
No Gemmini/IM2P cycles, CUDA timing, or full-model speedup is measured.
