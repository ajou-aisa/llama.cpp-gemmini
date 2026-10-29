# H3-8D1: opt-in K-tile Paired Scale reuse

Date: 2026-10-29.

## Result

The explicit DIM32 K-tile route passed native ownership/numerical tests,
actual FPGA raw Main/Residual INT32 goldens, QK/PV and PV-range regression,
and matched Decode/FFN benchmarks. Against H2 in the same new binary and
same existing DIM32 bitstream, steady graph latency decreased **18.996542%**
for Decode and **20.408209%** for FFN.

Default production H2 remains unchanged. The candidate remains opt-in:

```sh
export GGML_POTAL_PAIRED_KTILE=ON
```

This requires DIM32, Paired execution and compensation. Invalid values and
unsupported selection fail before commands; a candidate capacity refusal
cannot silently select Serial fallback. OFF/unset/empty retain the default.
No C11 API, RTL, resident capability, quantization or attention arithmetic
was changed. Changes remain separate and uncommitted after the H3-8
checkpoint; no push occurred.

## What changed, rather than renaming H2

H2 already uses K-window B residency and shares B between Main and Indexed
Residual. D1 keeps that verified schedule, not E4C1's small alternating
microtile schedule. It removes actual redundant P2 Scale Write/Release
commands when the existing Main allocation has exactly the needed identity.

DIM32's Main P1 uses one physical pass per original K32 block. Residual P2
uses one carrier per physical compact fragment. When a window's original
block IDs are consecutive, both expressions select the same carrier at the
same original J column:

```text
effective_base = Main.slot_base
               + (first_original_block - panel.block_begin) * J
effective_generation = actual_Main_generation
```

The executor derives a local mapping after validating the unchanged H2
plan, before command effects. Eligibility requires matching residency/J
coverage, consecutive original block IDs, contained checked slot extents,
Main programming before the alias and Main release after it. A matching
physical pair or equal carrier values alone is insufficient.

Gapped windows use ordinary P2 programming. DIM16 is explicitly unsupported
for this specialization: its Main P1 has two physical passes per K32 block.
No old resident capability was forced onto DIM32.

The original plan remains the logical schedule; the executor explicitly
specializes Scale lowering. Logical generation advancement is retained,
including skipped programs, while aliases read the Main wire generation.
Thus subsequent actual writes retain the original generation sequence.
`scale_reused_windows` and `scale_reused_columns` count the specialization.

Each skipped P2 release becomes a real checked global Fence. It never
releases Main-owned columns. Main releases remain the sole normal frees.
This preserves the original global drain cut, A-region reuse, loop-slot
reuse, pending readbacks and both pairs' reader completion. Main and
Residual accumulation order, ACC epochs, partial/ACC saturation, Index
ordering, B loading, INT64 reconstruction and final F32 merge are unchanged.

Source changes:

- `ggml/src/ggml-potal/matmul/paired-auto.{hpp,cpp}`: opt-in request,
  checked ownership mapping, actual command elimination and Fence boundaries.
- `ggml/src/ggml-potal/matmul/options.hpp`, `ops.cpp`,
  `matmul/stripe.{hpp,cpp}`: selection, propagation and refusal boundaries.
- `ggml/src/ggml-potal/tests/potal-paired-execution-test.cpp`: deterministic
  numerical, stream, identity, generation, failure and option regressions.
- `ggml/src/ggml-potal/tests/potal-paired-smoke.cpp`: independent actual raw
  INT32 golden and same-binary H2/K-tile benchmark route.

No batch API was added: eliminating checked calls removes more work than
merely batching repeated checks. Existing checked Scale APIs, open-time Raw
Trace behavior, liveness and generations remain intact.

## Ownership and failure proof

An alias is an additional reader, not an allocation. It creates no duplicate
C11 live bits and no separate cleanup responsibility. A recoverable failure
uses existing cleanup for actual live Main/nonaliased P2 allocations.
A replacement Fence failure follows the existing `"fence"` fatal path:
no retry, no further issue/release, no result publication, and retained
buffers/issuer ownership. Native fatal tests use process isolation.

Release and Fence share RTL `allDrained`, including final memory/writeback
completion. This is a synchronization proof, not a measurement of hardware
waiting. The executor's `drains` counter counts logical boundaries; C11's
session drain counter counts individual accepted commands. They need not
have equal values.

The legacy cleanup path still has the pre-existing limitation described in
the audit: a cleanup Fence failing after another sticky error has no separate
unknown-Fence latch. D1 adds no retries or recovery around that path.

E4C1 remains test-local and DIM16-only. Its C11/RTL resident capability is
not supported by the current DIM32 image. Consequently no DIM32 E4C1 timing
row is fabricated, and no production port was performed merely to add one.

## Numerical and regression evidence

The selected native suites ultimately pass for DIM16 and DIM32: ten tests
passed initially and the corrected execution regression passed separately.
They cover unchanged default command streams, exact raw Main/Residual and
INT64 reconstruction, consecutive/gapped blocks, equal carriers with
different identities, compact/lane/J/K tails, both pairs, generation wrap,
repeated calls, checked alias Fence placement, recoverable cleanup,
isolated fatal Fence handling, parser propagation and unsupported refusal.
These native results are software-only.

Actual DIM32 FPGA raw golden:

| Input | Route | Main INT32 values | Residual INT32 values | Scale Write = Release |
|---|---|---:|---:|---:|
| Consecutive original blocks | H2 | 1105 exact | 130 exact | 2210 |
| Consecutive original blocks | K-tile | 1105 exact | 130 exact | 1105 |
| Gapped original blocks | H2 | 1105 exact | 130 exact | 1560 |
| Gapped original blocks | K-tile | 1105 exact | 130 exact | 1560 |

The consecutive case reused 18 windows; the gapped case reused zero.
Expected INT32 values use independent wide dot products and explicit
saturation, not production SCU/lowering/emulator arithmetic.

The actual full attention smoke and PV-range-only smoke passed.
PV extension reports 49 jobs, 16 bitplane jobs, 11 corrections, 22 bias
MVINs, 49 LOOP_WS, 71 fences and 28224 balanced Scale writes/releases,
with `exact=1 status=0`. No full-model inference rerun occurred.

## Matched performance

Both routes execute from the same candidate binary, same original DIM32
bitstream, same seed/input generator and Performance settings as D0.
`--bench-ktile` substitutes K-tile for the Conservative comparison route,
retaining rotating/reversed order and the Serial control:

```sh
./bin/potal-paired-smoke --ktile-golden
./bin/potal-paired-smoke --bench-ktile --case decode_1x768x768 --repeat 7
./bin/potal-paired-smoke --bench-ktile --case gpt2_ffn_up_16x3072x768 --repeat 7
```

All values below are guest milliseconds, median / MAD.

| Measurement | H2 | K-tile |
|---|---:|---:|
| Decode steady graph | 17.061 / 0.062 | 13.820 / 0.045 |
| Decode steady pipeline | 16.432 / 0.018 | 13.211 / 0.041 |
| Decode cold graph | 32.406 / 0.028 | 29.152 / 0.081 |
| FFN steady graph | 89.170 / 0.150 | 70.972 / 0.167 |
| FFN steady pipeline | 88.538 / 0.250 | 70.345 / 0.176 |
| FFN cold graph | 150.815 / 0.241 | 132.350 / 0.100 |

The H2 Decode sample reaching 25.207 ms remains in the raw samples.
No outlier was removed. All 84 primary samples are exact; all Scale totals
balance; unexpected Paired fallback is zero. Every graph/pipeline Median
and MAD was independently recomputed.

Serial remains faster in this experiment: steady Decode 10.480 ms and FFN
56.022 ms. D1 is an improvement over H2, not a claim of globally optimal
routing or an overall model speedup. H2 remains default.

| Steady command count | Decode H2 | Decode K-tile | FFN H2 | FFN K-tile |
|---|---:|---:|---:|---:|
| Main LOOP_WS | 12 | 12 | 48 | 48 |
| Residual LOOP_WS / Index Config | 60 | 60 | 288 | 288 |
| Index Write | 1830 | 1830 | 11496 | 11496 |
| Scale Write = Release | 33024 | 19968 | 147456 | 73728 |
| ACC readbacks / tiles | 12 / 48 | 12 / 48 | 48 / 192 | 48 / 192 |
| Explicit fences | 1 | 55 | 1 | 289 |
| Logical drain boundaries | 73 | 73 | 337 | 337 |

Decode removes 13056 writes and 13056 releases, adding 54 Fences.
FFN removes 73728 writes and 73728 releases, adding 288 Fences.
Index and LOOP work do not decrease. B reloads are not optimized here.
The actual unchanged command stream apart from Scale keys and synchronization
replacement is covered by native stream comparisons.

CPU-observed steady Scale API median decreases from 8.182 to 4.905 ms for
Decode, and 36.478 to 18.107 ms for FFN. Explicit Fence API median increases
from 0.001 to 0.037/0.199 ms respectively. Issue API medians remain about
0.590/0.589 ms and 3.405/3.391 ms. These are CPU API intervals, not physical
NPU execution or waiting. Overlapping durations are not summed into graph
wall time. Separate profiled runs also complete, but are not mixed into
the primary timing table.

## Provenance and preserved first failure

Base software commit: `df8482e9fec038dd1f5db48c8b3ab9c1c11c3df9`.
Exact uncommitted source patch: `/tmp/h3-8d1/source.patch`,
SHA256 `d93b7717070e73a8316d2e4b07ffc1cf1be7d76e409a6d2f6551c45964f7856f`.
Paired-smoke binary SHA256:
`ac6d7bd559010fcfc45af31f6de89e6a0ee6da58f10fe332b2b70b2ed59d91bd`.
The unchanged DIM32 bitstream remains
`f114de218a94b61c57b6c730a5534b9c2b973ac7ddb23f70be5f25baf009aa75`.

```text
/tmp/h3-8d1/artifacts-before-run.sha256
/tmp/h3-8d1/deployed.sha256
/tmp/h3-8d1/results.sha256
/tmp/h3-8d1/acceptance.json
/tmp/h3-8d1/extracted/results/
/tmp/h3-8d1/run/uartlog
/tmp/h3-8d1/run/driver.log
```

The isolated guest and driver both exited 0. Existing D0/H3-8C results and
images remain intact. D1 uses one separate rootfs directly; its new runtime
slot does not duplicate that large image.

First native failure: `R fixture` on both DIMs. The new M65 fixture emitted
both endpoints of its last one-row stripe, duplicating residual coordinates
and invalidating packet metadata before execution. The fixture now emits
each endpoint once. Expected values, production code and tolerances were
not relaxed. Initial and corrected logs remain under
`/tmp/h3-8d1-d{16,32}-{native-gates-first,fixture-regression}.log`;
details are `/tmp/h3-8d1-first-failures.md`.

C/C++ LSP lacks generated include configuration; actual native and RISC-V
compiler results establish build acceptance. No new RTL build was needed.
No DMA/PE overlap, DMA-byte reduction, TTFT or TPOT improvement is claimed.
Independent Next-K prefetch remains the next separately gated stage.
