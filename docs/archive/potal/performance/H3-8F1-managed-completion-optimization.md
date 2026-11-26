# H3-8F1: managed completion bookkeeping reuse

Date: 2026-10-29.

## Result

The explicit `POTAL_H2_COMPLETION_REUSE=1` mode skips unchanged H2 host
reconciliation after a successful checked drain. It retains every real
Scale Release, validation, status response, live-column clear, core
outstanding-count clear, and CPU memory fence. Default is off.

Actual DIM32 FPGA numerical gates and seven-repeat benchmarks completed.
Guest result is `H3_8F1_EXIT=0`. Bookkeeping work decreased substantially;
these samples do **not** establish a useful graph-latency improvement.
Keep production H2 and all optimization defaults unchanged.

The enclosing FireSim driver hit the monitor's 30-minute deadline after
the successful guest script and Linux Power down. It did not produce a
driver exit file or `Simulation complete / PASSED` summary. This report
claims the recorded actual FPGA numerical/guest results, not a clean
enclosing-driver PASS. The new rootfs passed read-only filesystem checking
after the driver stopped, and its results were extracted separately.

## Source and invariant

Software base: `b5683e474b6b8708fe90f3d5a150a56322f64081`.
RTL/image: preserved D2 observation configuration at Chipyard
`bf1d98ca8a291a53845648ae475c80e01fd0b83d`.
Gemmini gitlink: `185293e87cbbc2a336410f49a3fd73d01ed8d04e`.
F1 is uncommitted C11, diagnostic output and regression work. No RTL,
quantization, ExSIA, RMD, attention arithmetic, or saturation change was
needed for this experiment. The later H3-8G source changes are not in the
frozen F1 binaries used here.

Responsible path:
`ggml/src/ggml-potal/potal.c`:
`paired_scale_release_core -> paired_drained -> h2_completed`.
The source mechanism and all callers were read before editing; language
references confirm `paired_drained` is the only caller of `h2_completed`.

`h2_completed` still executes `potal_cpu_fence_rw` before considering reuse.
Only its two-panel publication/reader reconciliation and pending-state
stores can be skipped. The first successful reconciliation certifies the
current managed state. This is a software idempotence certificate, not a
replacement for hardware `allDrained`.

| Effect or boundary | Certificate treatment |
|---|---|
| B CONFIG/MVIN or LdB-only staging/admission | Invalidate before raw dispatch, including failed attempts |
| Main/Residual LOOP_WS and preceding configuration | Invalidate before every attempted non-drain command |
| Scale Write; Index Config/Write | Invalidate even when synchronous success clears control-pending |
| ACC/readback commands | Invalidate before CONFIG/MVOUT or other effect |
| New panel generation; retirement | Explicit invalidation even without a raw command |
| Answered failure; partial issue | `paired_fail` invalidates; failed sessions never reuse |
| Cleanup | Existing one-drain policy; failed state cannot publish or reuse |
| Unknown Fence/Release response | Existing sticky error and quarantine; no retry, close or free |
| Successful valid Release/Fence | Retain command, response, CPU fence and core bookkeeping; reuse only unchanged H2 reconciliation |
| Query | Observation only; cannot publish or certify |
| Safe close | Existing pending, live Scale, Index, generation and ownership gates remain |

Release's live-column clear remains outside the cached reconciliation.
Changing that bit does not change B loading, issued/completed readers or
the pending/control state reconciled by `h2_completed`. Existing resident
mode is mutually exclusive and does not use this certificate.
The option is sampled once at H2 begin, accepting exactly `"1"`.
Completion counters exclude any legacy drains preceding managed begin.

## Validation and provenance

Separate artifacts, without overwriting the D2 rootfs or old test output:

```text
/tmp/h3-8fg/d32-host
/tmp/h3-8fg/d32-performance
/tmp/h3-8fg/rootfs-f1.img
/tmp/h3-8fg/staged.sha256
/tmp/h3-8fg/run-f1-validation.sh
/tmp/h3-8fg/f1-run/{uartlog,driver.log,uart-sent.log}
/tmp/h3-8fg/extracted-f1/results
```

Frozen RISC-V `potal-paired-smoke` SHA-256:
`a2670cee5d2434f65e6b0e87700ff0898290641abac3b26c4caad5ec2bfe4f45`.
Frozen `libggml-potal.so` SHA-256:
`d2b3b903d8f05c8ad65e44cc4200b623ae1dde08f36d0057e7bce94cdf3a76db`.
All staged binaries/libraries were checked inside the guest before tests.
The build manifest records the actual RISC-V compiler and DIM32 parameter
header hash, not just a directory-name inference.

Native default and reuse runs passed 12/12 targeted tests, including C11
command/status checks, numerical execution, stale generations, partial
issues, cleanup and fork-isolated quarantine. These are software-only.
The new C11 fixture requires all eleven literal command records and eight
checked managed drain responses with both modes.

An initial overbroad CTest selection also ran two unchanged legacy
microtile tests with DIM16-specific expected shapes on DIM32. They failed:
`potal-paired-schedule-test` and `potal-paired-unit-test`.
The same invocation selected an unbuilt `potal-gguf-load-test`.
The complete failing log remains at `/tmp/h3-8fg/native-default.log`.
No failing test was edited, deleted or described as passing. The accepted
D2 scope is eleven tests; the twelve-test run adds the existing log test.
The editor's default include paths were incomplete. Explicit CMake
compilation passed; an explicit clangd C11 check additionally reported
twelve pre-existing declaration-parameter-name tidy findings.

Actual FPGA gates:

- API smoke passed.
- Nine raw Main/Residual INT32 golden cases passed with reuse off; nine
  passed with reuse on, including signed saturation and I/J/K tails.
- Full attention smoke and PV-range smoke passed. The latter retained
  exact=1/status=0, 49 jobs, 16 bitplane jobs and 28224 balanced Scale
  writes/releases.
- The full-row D2 fixture retained physical cross-pair SRAM/MAC overlap:
  108 cycles off, 130 cycles on. H2/K-tile controls and the two smaller
  fixtures had zero. These are bounded representative-PE witnesses, not
  Decode/FFN overlap measurements or saved cycles.

## Measured work and graph latency

Separate diagnostic-profile runs give these identical per-graph counts
in all four recorded invocations of each condition:

| Workload | Managed calls | Full bookkeeping off/on | Reuses on | CPU fences off/on | CPU raw commands off/on |
|---|---:|---:|---:|---:|---:|
| Decode 1x768x768 | 20023 | 20023 / 61 | 19962 | 20023 / 20023 | 42860 / 42860 |
| FFN 16x3072x768 | 74017 | 74017 / 289 | 73728 | 74017 / 74017 | 163922 / 163922 |

The raw-command counts are CPU submissions, not internally generated
accelerator commands or measured bus transactions. CPU fences are retained
instruction counts derived from successful managed completion calls,
not hardware waiting cycles.

Primary runs unset tracing/profiling and retain the same DIM32 image,
30 MHz host target, EXSIA/RMD settings, seed 20261023, input-generation
order, frozen tiling and cache procedure. Each condition has seven cold
and seven steady samples of H2, K-tile and D2, with rotating/reversed
route order within that condition. Off/on conditions are separate process
runs, not interleaved causal pairs.

Steady graph-compute milliseconds, median (MAD):

| Workload | H2 off/on control | K-tile off/on control | D2 off | D2 + reuse |
|---|---|---|---|---|
| Decode | 18.690 (.074) / 19.148 (.183) | 14.700 (.025) / 15.425 (.012) | 15.896 (.054) | 15.919 (.025) |
| FFN | 98.230 (.885) / 98.795 (.306) | 76.609 (.091) / 76.825 (.267) | 79.102 (.145) | 78.729 (.193) |

Individual samples, cold summaries, min/max, cache counters and CPU stage
intervals remain in
`results/{decode_1x768x768,gpt2_ffn_up_16x3072x768}-reuse-{0,1}/bench.{out,err}`.
Diagnostic counts are separate in the corresponding `profile-*` paths.
Intervals are CPU-observed and overlapping stages are never summed into
graph wall time.

Decode does not improve; FFN's small median decrease is not sufficient for
a performance-adoption claim, especially with shifting controls and the
noninterleaved off/on runs. The experiment establishes less host work,
not that completion bookkeeping exclusively caused D2's regression.
It does not establish DMA, Execute or readback waiting attribution.
No full GPT-2 inference was repeated.

## Subsequent matched-image comparison

On 2026-10-30 the corrected g2 image completed an interleaved five-route
comparison with seven cold and seven steady samples per route. That
separate evidence is in `H3-8FG-performance-comparison.md`; it does not
replace or relabel the frozen F1/D2-image samples above. On g2, D2-to-F1
steady median reductions were 1.52% Decode, 0.66% FFN and 1.58% tail.
These modest results retain all original completion-safety constraints
and do not justify changing production defaults.
