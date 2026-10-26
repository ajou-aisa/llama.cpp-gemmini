# H3-7B5: QK Host-INT64 Limb Reconstruction

Date: 2026-10-26.

## 1. Executive Summary

**A -- Implemented & Hardware Verified, within the declared DIM16 domain.**

The QK implementation now completes independent K-low/high INT32 hardware
jobs at the original block carrier Delta. Host checked INT64 combines
`low + 256*high`, then combines folded Q limbs. Quantization, clipping,
DIM16 partial saturation, final floating scaling and PV semantics are preserved.

The actual DIM16 QK/PV smoke and GPT-2 16+8 inference passed. Raw signed32
low/high outputs match the new contract on the destructive witness.
All 402056 full-inference logits are bitwise identical to H3-7B; FP32 accuracy
therefore did not improve on this request. Matched n=7 kernel medians decreased
by 6.06% and 6.43% in the two measured fixtures; the single cold inference pair
does not establish a reproducible end-to-end speedup.
The supplemental K4097 hardware case also passed three K panels per stationary
limb, with independent initialization and final-only readback.

## 2. Committed H3-7B/E/E3 Checkpoint

Two explicitly scoped checkpoints were committed on the existing `potal` branch:

| SHA | Commit |
|---|---|
| `3e373948d44d0eccbc7c52fd52f65750da703314` | `feat(potal): implement and validate H3-7B attention offload` |
| `dc257bcbb707913edb0af5a1d623970276051528` | `test(potal): add H3-6E paired scheduling analysis and prototypes` |

Their predecessor is the requested H3-7A checkpoint
`dcab70ff8a76348a531c3f89785a29f13a2e8413`.
The Attention commit has 21 paths; the scheduling commit has 13 paths.
The shared backend CMake file was split in the index: Attention targets first,
inventory and opt-in E3 targets second. E3's ACC test depends on the earlier
Attention C11 ownership support. The worktree contents were not reverted.

Staged whitespace, name/status and diff statistics passed before each commit.
Only listed source, test and report paths were staged. `models/gpt2/` remained
untracked; no binary, model, rootfs, build product, Gemmini or FireSim user
change entered either commit. No push or new branch was made.

All B5 changes remain uncommitted. A detached baseline worktree at
`/tmp/h3-7b5-baseline-source` points to `dc257bc`; benchmark-only harness changes
there do not alter the checkpoint production implementation.

Frozen H3-7B binaries and libraries: `/tmp/h3-7b5-baseline/bin`.

| Artifact | SHA256 |
|---|---|
| H3-7B attention smoke | `fb585e3dbb2578e7f5ccf2c2b08e39f85dca29937fc80da78bb1b37121956384` |
| H3-7B full inference | `d227b1590e7d01d9f4b865cd4482543ffc701be1f8004b7a9c49b1b7c28894c3` |
| H3-7B production library | `630fcc334458c7cad69506be1d8588f2bd0253c7becc2d73c7ceb9835a16e513` |
| GPT-2 Q8_HP1 GGUF | `cf23a45e4db23736d7d1f67ab42018803a198cfcb01db36daca69ff56bdbc7d4` |
| GPT-2 FP16 GGUF | `5bd663a1e1d303f9404dd2edd92e4e3ca3e24bcf80d19c655880f0e455c8f4df` |

## 3. Original Shared-ACC Architecture

For each folded Q limb, H3-7B submitted one logical doubled-K job:

```text
A = [Q_limb, Q_limb]
B = [K_low; K_high]
carriers = [Delta, Delta+8]
```

All low DIM16 partials preceded all high partials in one saturating INT32
accumulator. C was present only on the last high K panel. Host reconstructed
only the completed Q-limb INT32 result.

The original choice was deliberate, not a failed implementation of the new
contract. B5 changes the numerical contract between stationary limbs.

## 4. New Separate-ACC / Host-INT64 Architecture

For every active folded Q limb, including dense zero digits within its extent:

```text
low  = completed_INT32_WS(Q[r], K_low,  Delta)
high = completed_INT32_WS(Q[r], K_high, Delta)
C_r  = checked_INT64(low + 256*high)
out  = checked_INT64(out + 256^r*C_r)
```

Each job has original padded K, its own first-panel overwrite, ordered later
panel accumulation, and one final store per I/J output group. A successful
fence covers that store before Host consumption or buffer reuse.

The original doubled-K tile-selection input is retained solely to freeze I/J/K
choices. It is not a doubled-K hardware descriptor. Whole-block panel/divisor
rules remain unchanged. The single original-K A allocation and identical
carriers are reused; B is repacked for high only after low completion.
DMA row strides may include alignment padding beyond logical K.

Signed INT32 results are widened before radix weighting. The existing helper
uses `__int128_t` multiplication by a positive radix power, checks the signed
INT64 range and only then converts. No negative signed shift, wraparound,
silent clipping or later-limb cancellation repairs an intermediate overflow.

## 5. SCU/INT32 Saturation Semantics

Within each stationary limb:

```text
partial = signed DIM16 dot
aligned = Sat32(partial * 2^Delta)
acc     = Sat32(acc + aligned)
```

The zero-carrier sentinel still produces zero. K-high retains Delta but no
longer adds 8. Per-limb K-panel and DIM16 contribution order is unchanged.
Radix weighting between independently completed limbs now occurs in Host INT64.

Q local adaptation, stripe folding, rho14, signed radix decomposition, K's
runtime MXINT/block32 carriers and clipping `[-32768,32639]` are unchanged.
Final output retains the existing double `ldexp` followed by F32 conversion
and finite-result check.

PV still uses unsigned 16-bit P quantization, biased signed low bytes, balanced
upper P limbs, signed8 V and one exact representable INT32 physical ACC bias
addition. Its upper P reconstruction and saturation order are unchanged.
An exact PV bias outside INT32 is still explicitly unsupported.

## 6. Implementation Changes

Production changes are confined to Attention arithmetic and truthful QK
statistics/diagnostics:

- `matmul/attention.cpp`: independent stationary passes and checked two-level
  reconstruction; original tile selection and failure ownership retained.
- `matmul/attention.hpp`: new numerical contract and `qk_limb_jobs`.
- `ggml/include/ggml-potal.h`, `ops.cpp`, `ggml-potal.cpp`: replace the obsolete
  shared-ACC counter with completed independent `attention_qk_limb_jobs`.
- `tests/potal-full-inference.cpp`: emit and validate that new counter.
- `tests/potal-attention-test.cpp` and `tests/support/attention-golden.hpp`:
  independent new-contract integers, separate descriptor checks and failures.
- `tests/potal-attention-smoke.cpp`: actual raw results and command-epoch
  validation through transparent test-only API wrappers.
- `tests/potal-attention-bench.cpp` and backend CMake: opt-in matched benchmark,
  separate profiling and checked reconstruction replay.

Software paths without a full prefix are under `ggml/src/ggml-potal/`.
Function-level pre-edit dataflow and ownership were recorded in
`/tmp/h3-7b5-source-audit.md` before changing production.

`attention-hw.cpp`, regular WS tiling/issuer/C11 APIs, PV bias encoding,
Linear GEMM, Paired planning/execution, RTL, FPGA image, Gemmini and host-init
are outside the implementation changes.

## 7. Independent Integer Reference

`tests/support/attention-golden.hpp` derives Q digits from original integer
codes and uses independent wide arithmetic. `qk_pass` calculates one complete
signed INT32 result at original Delta. Default `evaluate` accepts only the new
contract; explicit legacy mode exists only to demonstrate the old answer.

The K32 witness Q127, K-129, Delta20 has:

```text
K_low=127, K_high=-1
low_ACC  = 2147483647
high_ACC = -2147483648
B5       = -547608330241
H3-7B    = -2147483648
```

The executable K64 witness has active Q limb1:

```text
low_ACC  = 2147483647
high_ACC = 1073872895
B5       = 70927089860352
H3-7B    = 549755813632
B5 F32 at scale 2^-48 = 0.251983642578125
```

The old test's separately summed panels with high Delta+8 are not the new
reference. No test accepts either old or new output as correct.

Native coverage includes signed K/high signs, all five Q limbs, zero and
positive carriers, independent saturation, cancellation, INT64 intermediate
overflow, K1/17/31/32/33/64/65, multiple J tiles, heads/batches, F16/F32 strides,
failed low/high boundaries, fatal high completion and transactional output.
Preserved PV bias/saturation/rejection tests remain active.

An execution-level F32 fixture also creates five folded Q limbs and Delta30
through real preparation. Reconstruction fails with POTAL_ERR_CAPACITY after
eight completed low/high jobs, publishes no output, and leaves no DMA or Scale
entry pending. A later carry limb cannot repair an already-rejected intermediate
overflow. K4097 additionally passes the native multi-panel regression.

Focused native API, Attention API/unit, GGML dispatch, stripe pipeline and paired
execution regressions passed. Initial new-test assumptions were corrected:
DMA-aligned lda can exceed logical K, and a 2^-30 stationary value requires F32
instead of the fixture's default F16. No production check was weakened.

## 8. Hardware Validation

The actual DIM16 smoke exited 0 and emitted `__B5_HARDWARE_INTEGER_GATE_PASS__`
before either full-inference request began.

| QK raw fixture | Completed K-limb jobs | Actual issue calls | Final C-bearing calls | Fences | Result |
|---|---:|---:|---:|---:|---|
| K64, F16, strided, head/batch repeat 2 | 96 | 96 | 96 | 96 | Exact |
| K33, F32, strided | 4 | 4 | 4 | 4 | Exact |
| K65, F32, rows 17, strided, head/batch repeat 2 | 96 | 96 | 96 | 96 | Exact |
| K64 destructive witness | 4 | 4 | 4 | 4 | Exact |
| K4097, F32, strided, three panels per limb | 4 | 12 | 4 | 12 | Exact |

The normal fixtures exercise negative K-high and two or three nonzero Q limbs.
Each raw C is inspected only after the successful runner's completion fence.
Transparent wrappers invoke the real Regular WS API once and check original-K
operand identity, unchanged Delta, first overwrite, later accumulation,
final-only C, generations, Scale counts and fence counts. C-bearing issue counts
are command evidence, not measured bus transfers or individual DMA bursts.

Actual witness at active Q limb 1:

```text
low:  actual 2147483647, expected 2147483647, generation 162
high: actual 1073872895, expected 1073872895, generation 163
reconstructed: 70927089860352
F32: 0.251983642578125
old Shared-ACC F32: 0.001953125
independent CPU FP32: 0.755950927734375
```

Thus the hardware selects the new integer contract, not the old golden.
Within-limb saturation still causes error in this deliberately destructive
fixture; independent limbs do not imply FP32 equality.

Preserved PV K17/K33/K65, carrier24 cancellation and physical negative-bias
saturation checks passed. The ordinary QK/PV fixtures also matched their
independent CPU outputs exactly. This does not generalize to all model logits.

The first smoke's limbs each fit one software K panel. The supplemental K4097
run verifies three panels per limb: first overwrite, two accumulations, and C
only on the third panel. All four completed signed32 results and the final F32
output match the independent reference; FP32 error is zero. Its graph CPU
interval is 41.504 ms, one diagnostic sample, not a performance distribution.
Both the edge script and its simulator exited 0.

Primary evidence: `/tmp/h3-7b5-run/uartlog`.
Initial hardware smoke SHA256:
`a2682335bcb13fc24e0312a50ac8108e29ec9939ccbc7f5e6ebaceed5fc585ba`.
Production library SHA256:
`085550b94546e89d637d76391d0101cc70fe03b7bc5a714ad55024ef972ff187`.
Final smoke, adding the K4097 case and `--multi-panel-only`, SHA256:
`d2cdfca7719fce4165af0f18e8819ac6acfd9ea2b8424a5cfaa1d4e60c67e02d`.
Supplemental evidence: `/tmp/h3-7b5-edge-run/uartlog`.

## 9. GPT-2 Full Inference Results

Fresh H3-7B baseline and B5 each completed one cold captured request:
serial hardware, Attention ON, prompt 16 / generate 8, deterministic greedy
selection, full logits and `--no-warmup`. Both used the preserved
124439808-parameter Q8_HP1 model, F16 KV cache and three CPU threads.

Every one of eight forwards has:

- 73 PoTal MatMuls: 49 HP1 weighted MatMuls, 12 QK and 12 PV.
- Zero CPU Attention MatMul and zero CPU HP1 MatMul.
- 12 CPU Softmax operations.
- A populated collector: 73 MatMul records and 12 positive CPU operation kinds.

Both requests report QK 96, PV 96, Attention completed 192 / failed 0, HP1
completed 392 / failed 0, no pipeline/compensation/GEMM failure, and last status 0.
Issuer Linux CPU1/hart0 was ready. KV token counts progress 16..23 and maximum
positions 15..22. Both captures contain 402056 finite logits.

Generated token IDs, identical to CPU FP32 and archived H3-7B:

```text
5911 290 5911 262 16936 4780 13 383
```

Evidence:
`/tmp/h3-7b5-results/{baseline,new}/attention-16x8.{model.json,request0.json,request0.logits.f32,request0.tokens}`.
The primary guest suite and simulator exited 0. Kernel warmups were separate
kernel calls, not extra full-inference requests.

## 10. Numerical Accuracy Comparison

The existing independent Transformers CPU FP32 capture is reused unchanged.
The existing comparison module validates finite binary32 extent, prompt IDs and
history alignment and uses float64 reductions. NRMSE is
`RMSE / max(RMS(reference), 1e-12)`.

| Comparison, 402056 logits | Max absolute | RMSE | NRMSE | Token agreement |
|---|---:|---:|---:|---:|
| CPU FP32 vs fresh H3-7B | 2.278503418 | 0.562603866 | 0.004445341 | 8/8 |
| CPU FP32 vs B5 | 2.278503418 | 0.562603866 | 0.004445341 | 8/8 |
| Fresh H3-7B vs B5 | 0 | 0 | 0 | 8/8 |
| Archived vs fresh H3-7B | 0 | 0 | 0 | 8/8 |

All eight histories align. There is no token divergence step. Fresh baseline,
archived H3-7B and B5 logits are bitwise equal; they are not bitwise equal to
CPU FP32. The historical error values are reproduced, not used as thresholds.

CPU FP32 error by step, identical for H3-7B and B5:

| Step | Max absolute | RMSE | NRMSE |
|---:|---:|---:|---:|
| 0 | 1.882049561 | 0.644828844 | 0.004264240 |
| 1 | 1.387077332 | 0.440898443 | 0.003907957 |
| 2 | 1.923629761 | 0.681905672 | 0.004838841 |
| 3 | 1.489776611 | 0.491370313 | 0.004255078 |
| 4 | 1.591606140 | 0.306908688 | 0.002534195 |
| 5 | 1.258003235 | 0.234196404 | 0.002348130 |
| 6 | 1.782821655 | 0.880099119 | 0.008419473 |
| 7 | 2.278503418 | 0.540308911 | 0.003501647 |

QK-only ordinary smoke fixtures have zero FP32 error. In the synthetic
destructive witness, absolute error changes from 0.753997803 for the verified
Shared-ACC value to 0.503967285 for B5. That local improvement does not become
a claim of model-level accuracy improvement: none was observed here.

GPT-2 internal QK tensors and hardware saturation incidence were not captured.
Bitwise-equal final logits do not prove that no internal saturation occurred.
Quantization/saturation hardware counters: **Not measured**.

Machine-readable comparison:
`/tmp/h3-7b5-analysis/numerical-comparison.json`.

## 11. Performance and Command Overhead

### Matched primary kernel measurements

Both kernel builds use the same compiler/optimization and diagnostics-OFF
definitions; the only differing definition selects contract-specific benchmark
checks and replay.
Production source comes from the checkpoint or B5 respectively. The benchmark
and independent oracle sources are identical.

Seven rounds alternate baseline/new processes, reversing their order each
round. Each process has one untimed kernel warmup and one timed sample.
The timer encloses production `attention::execute` on the established hart0
owner: preparation, packing, WS jobs, Host reconstruction and output publication.
Owner startup, validation and printing are outside the timer. No per-call
capture or profile timer is enabled in this primary mode.

| QK fixture M/N/K | Baseline median / MAD ms | B5 median / MAD ms | Median difference |
|---|---:|---:|---:|
| Prefill 16/16/64 | 1.419 / 0.014 | 1.333 / 0.009 | -0.086 ms (-6.06%) |
| Decode 1/23/64 | 0.949 / 0.002 | 0.888 / 0.004 | -0.061 ms (-6.43%) |

All 28 samples pass their respective independent integer contract and stable
command-count checks. These measured kernel improvements apply to these fixed
fixtures, not all Attention shapes or full inference.

| Per-sample count | Baseline | B5 |
|---|---:|---:|
| Completed QK jobs/results | 3 shared | 6 independent |
| LOOP_WS | 6 | 6 |
| Reported loop-completion fences | 6 | 6 |
| TLB flushes and corresponding startup fences | 3 | 6 |
| Prefill Scale writes / releases | 192 / 192 | 192 / 192 |
| Decode Scale writes / releases | 276 / 276 | 276 / 276 |

Each separate pass now has its own completed C result. This increases result
stores and startup work without increasing LOOP_WS count in these frozen plans.
The smaller operand allocation and removed duplicate A/carrier packing coexist
with added reconstruction work; this experiment does not causally isolate each
effect's share of the total.

### Separate diagnostics

Optional profile n=7 median / MAD of CPU runner intervals, ms:

| Fixture | Baseline combined K runner sum | B5 low runner sum | B5 high runner sum |
|---|---:|---:|---:|
| Prefill | 0.088 / 0.001 | 0.055 / 0 | 0.050 / 0.001 |
| Decode | 0.099 / 0.001 | 0.059 / 0 | 0.054 / 0 |

Each sum covers sequential calls across three Q limbs within one profile run.
The intervals include checked Scale/issue/fence and embedded readback behavior.
They are not pure FPGA GEMM time. Baseline low/high cannot be separated by this
call boundary. Do not add these profile medians to primary wall measurements.

Checked Host reconstruction replay, n=7, 4096 replays per captured sample:

| Fixture | Baseline median / MAD microseconds | B5 median / MAD microseconds |
|---|---:|---:|
| Prefill | 20.510010 / 0.001221 | 65.823975 / 0.014160 |
| Decode | 1.985352 / 0.000488 | 6.402344 / 0.002686 |

Replay consumes actual completed hardware signed32 C and calls each version's
production checked helper. It includes buffer reset and checksum work and runs
on the calling CPU thread, not the issuer. It is a separate microbenchmark,
not an attribution or subtraction from production node time.

Regular WS embeds final ACC readback in its last LOOP_WS. Isolated readback DMA
latency and actual accelerator execution/wait components are **Not measured**.

### Full inference: one cold capture per version

| CPU-observed interval, ms | Fresh H3-7B | B5 |
|---|---:|---:|
| TTFT | 8064.034 | 8097.401 |
| TPOT | 1934.823429 | 1924.606714 |
| Total 16+8 | 21607.798 | 21569.648 |
| Aggregate Attention node envelopes | 2723.369 | 2703.871 |
| Model loading, outside request interval | 2481.752 | 1397.784 |

These are n=1 observations, not median/MAD estimates. Each process has a cold
application weight cache, but the second model load can benefit from the OS
file cache. The -38.150 ms total difference does not establish a reproducible
full-inference speedup. Historical H3-7B 8037.795/1927.997/21533.775 ms remains
a separate single capture.

Full-inference Attention jobs change 6912 to 10368; QK jobs change 3456 shared to
6912 independent. Attention LOOP_WS remains 10368. PV corrections 1152 and bias
MVIN tiles 4608 are unchanged. HP1 Main/Residual loops remain 9328/7487, and total
Scale writes/releases remain 56977433 each. These are successful execution
statistics, not physical bus-byte or accelerator-cycle measurements.

Raw results: `/tmp/h3-7b5-results/kernel-*.log`, `profile-*.log`,
`reconstruct-*.log` and both inference JSONs.
Aggregates: `/tmp/h3-7b5-analysis/execution-and-performance.json`.

## 12. Error/Ownership Validation

Checked descriptor validation, TLB ordering, Scale generation and fence/release
behavior are retained. Low/high reuse occurs only after the prior runner
finishes successfully. A fatal command retains DMA allocations through
`operands.abandon()`, propagates the original failure and stops later work.

All output remains staged across limbs, stripes, heads and batches. A failed
high pass or reconstruction overflow cannot publish earlier low results.
Host scratch not referenced by DMA may be freed; DMA operands with unknown
completion may not.

Native failure tests are software-only evidence. Actual hardware fault injection
is not claimed.

The final check found 50 protected source/artifact hashes unchanged. The Q/K/P/V
preparation functions and exact PV bias builder are text-identical to checkpoint;
the entire `attention-hw.cpp` is unchanged. Gemmini HEAD/pre-existing diff,
the 67-file PoTal Scala manifest and host-init match their prior fingerprints.
Existing models, old rootfs images and frozen binaries were preserved.

Native and DIM16 builds passed. LSP still lacks the generated target include
configuration. Existing macro-redefinition/type-limit/unused-code warnings
remain in untouched generated/common sources; none was suppressed for B5.
The Chipyard `.conda-env` activation warning did not block the existing
FireSim toolchain.

Primary reproduction script: `/tmp/h3-7b5-guest.sh`.
Additional hardware gate: `/tmp/h3-7b5-edge-guest.sh`.
Kernel CLI:

```text
potal-attention-bench --bench --case prefill --repeat 1
potal-attention-bench --bench --case decode --repeat 1
potal-attention-bench --profile --case all --repeat 7
potal-attention-bench --reconstruct --case all --repeat 7
```

The primary script alternates the first two commands between baseline and B5
over seven rounds. Benchmark targets are opt-in with `POTAL_B5_TESTS=ON`;
`POTAL_B5_SHARED_BASELINE` changes only test expectations/replay against the
checkpoint source, not production routing.

## 13. Remaining Limitations

Independent stationary limbs do not remove DIM16 partial or within-limb INT32
saturation. They define different arithmetic, not exact real-number GEMM.
PV's existing exact-bias INT32 boundary remains. Hardware wait components and
unobserved saturation counters must remain Not measured.

## 14. Recommendation for H3-6E4

**Final result: A -- Implemented & Hardware Verified.** The declared new integer
contract, actual RoCC execution, software failure/ownership behavior, supported
PV regression and real short Attention inference are verified.

Performance is a separate result: both fixed kernel medians improve in the
matched n=7 run, while full inference has only one fresh sample per version.
Model-level FP32 accuracy does not change on this request.

H3-6E4 should use this independent-K-limb contract as its numerical baseline.
Before implementation, specify resident B offsets/strides, ACC epoch selection,
SCU metadata binding, load-only ownership/completion and Scale/Index lifetimes.
Any reordered/pipelined candidate must retain each limb's saturating partial
order, exact integer outputs, safe failed/unknown DMA handling and measured
benefit after accounting for extra commands and context transfers.

No H3-6E4 interface, RTL extension, FPGA rebuild, Paired schedule replacement or
PV numerical change was implemented. The current branch remains `potal` at
`dc257bcbb707913edb0af5a1d623970276051528`; the index is empty. B5 source, tests
and this report remain uncommitted. No push was performed.
