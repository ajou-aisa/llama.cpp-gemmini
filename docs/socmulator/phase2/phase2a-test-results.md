# Phase 2A test results

Status: complete. All requested builds and checks were attempted.
Nine comparable native lanes retain exactly the same 734 registered
PASS/FAIL/NOT RUN outcomes as the post-removal baseline: 649 PASS,
69 pre-existing FAIL, 16 pre-existing NOT RUN. No test disappeared and no
previously passing test newly failed. PB's newly enabled numerical tests
and the extra contract checks are listed separately below.
Evidence root: `build-npu/phase2a-evidence/` (`E` below).
Native Linux/cross-build logs: `build-npu/phase2a-linux/logs/` (`L`).
Host: macOS arm64; Linux: `socm-baseline:ubuntu24`, arm64, GCC 13;
RISC-V: the image's riscv64-linux-gnu toolchain, cross-build only.

## Results

| Lane | Build | Tests actually run | Comparison |
| --- | --- | --- | --- |
| macOS CPU-only | PASS | 56 PASS, 4 FAIL / 60 | identical post-removal set |
| macOS standalone Gemmini API | PASS | 5/5 PASS | includes expanded policy contract checks |
| macOS Gemmini INT8 D16 reference | PASS | 110 PASS, 9 FAIL / 119 | identical post-removal set |
| macOS software A4W4 on INT8 D16 | PASS | 104 PASS, 9 FAIL / 113 | identical post-removal set |
| Linux CPU-only | PASS | 55 PASS, 6 FAIL / 61 | identical post-removal set |
| Linux Gemmini INT8 D16 reference | PASS | 107 PASS, 13 FAIL / 120 | identical post-removal set |
| Linux PB D32 Q8_0 reference | PASS | Q8_0 block, attention, functional model: 3/3 PASS | previously native BLOCKED; newly executed |
| Linux PB D32 MXINT8 reference | PASS | MXINT8 block, attention, functional model: 3/3 PASS | previously native BLOCKED; newly executed |
| Linux PB five-core Q8_0 | PASS | block and attention: 2/2 PASS | zero CPU placement mismatches |
| Linux PoTAL INT8 D16 | PASS | 75 PASS, 9 FAIL, 1 NOT RUN / 85 | identical post-removal set |
| Linux PoTAL INT8 D32 | PASS | 71 PASS, 11 FAIL, 1 NOT RUN / 83 | identical post-removal set |
| Linux PoTAL INT4 D32 | FAIL, pre-existing fixture compile errors | 66 PASS, 8 FAIL, 14 NOT RUN / 88 | identical post-removal set |
| RISC-V Gemmini API | PASS | NOT RUN | no physical target |
| RISC-V Gemmini INT8 D16 | PASS | NOT RUN | no physical target |
| RISC-V PoTAL INT8 D16 | PASS | NOT RUN | no physical target |
| RISC-V PB INT8 D32 | PASS | NOT RUN | no physical target |

The Linux baselines were copied before any rerun to `E/pre/*.ctest.log`.
macOS comparisons use complete `*.restored-ctest.log` from the removal
phase, copied to `E/pre/mac-*.restored-ctest.log`; later config-only logs
contain only one or two tests and are not mistaken for full baselines.
Skipped/disabled tests remain NOT RUN; they are included in registration
counts even where CTest's footer excludes disabled tests.
`E/per-test-comparison.json` records every compared name and its before/after
status, including failures, disabled/skipped tests, additions and disappearances.
All nine comparable lanes have no status or registration changes.
`E/comparison-summary.json` contains lane totals and the empty change lists.

Evidence by lane:

| Result | Logs |
| --- | --- |
| macOS full suites | `E/mac-{mac-cpu-only,gemmini-int8-d16-ref,gemmini-a4w4-int8-d16-ref}.{build,ctest}.log` |
| Final INT8 source refresh and affected tests | `E/mac-int8-scope-final.{build,ctest}.log` |
| API/configuration | `E/policy-{build,tests}.log` |
| Linux native suites | `L/{cpu-only,gemmini-int8-d16-ref,potal-int8-d16,potal-int8-d32,potal-int4-d32}.{configure,build,ctest}.log` |
| Cross-builds | `L/{rv-gemmini-api,rv-gemmini-int8-d16-rocc,rv-potal-int8-d16-rocc,rv-gemmini-pb-d32}.{configure,build}.log` |
| Final builds after restoring unchanged FP hook semantics | `E/{rv-gemmini-int8-d16-rocc,gemmini-int8-d32-pb-reference,gemmini-int8-d32-pb-mxint8-reference}.scope-final-build.log` |
| Matrix exit codes | `E/linux-matrix.log`, `E/riscv-matrix.log` |

## Pre-existing failures

The completed full-suite FAIL results are retained baseline failures, not
silently treated as PASS. Linux CPU retains `test-experiment-output`,
`test-chat`, `test-gemmini-cpu-cycle-validity-contract`,
`test-gemmini-off-build`, `test-gguf`, and `test-eval-callback`.
Linux Gemmini additionally retains `gemmini_weight_formats`,
`gemmini_mxint_weight_routes_activation`,
`gemmini_direct_metrics_source_contract`,
`test-gemmini-exsia-profile-contract`, `test-gemmini-rmd`,
`test-gemmini-profiled-domain-contract`, and `test-gemmini-build-contract`.
The per-name ledger records the differing macOS set, including
`test-backend-ops`.

PoTAL INT4's pre-existing full-build failure is in
`ggml/src/ggml-potal-w4a4/tests/potal-attention-test.cpp`: stale fixtures pass
`vector<int32_t>` where the API requires `vector<int64_t>`, pass stripe
callbacks where `paired_runner` is required, and call absent `run_corrected`.
The correction scope is those test fixtures/callbacks and their numerical
oracles, as a separate follow-up. This phase changes no PoTAL production API.
Baseline diagnostics remain in `build-npu/linux/logs/potal-int4-d32.build.log`;
the final comparison uses the new lane's own build log.
`E/potal-int4-build-comparison.json` confirms all 25 unique compiler errors
are identical before/after, with no new or removed error.
Thirteen pre-existing `w8-only` registrations remain disabled in both INT4
trees; the model-dependent `potal-gguf-load-test` remains skipped. No new
disabling or suppression was introduced.

## PoTAL D32 stripe investigation

The retained `potal-stripe-pipeline-test` fails in both full-suite logs.
The original pre-phase binary, run five times in the original `/src` mount,
fails 5/5 with 26, 24, 26, 27 and 27 assertion failures
(`E/pre/stripe-d32-original-binary.log`).
The new unmodified binary fails 3/5 isolated runs, with failure counts
17, 1, 0, 1, 0 (`E/stripe-d32-post-isolated.log`).
These are actual runs, not an assumed “flaky” classification.

The failing assertions are the `ROLES ... Main/Residual INT32, INT64, F32
and dependencies` and injected-failure `... joined, sticky failure` checks.
A diagnostic-only copy under `E/potal-stripe-diagnostic.cpp`, linked to
the new build's unchanged production objects, logs only after a predicate
fails. Five runs produce 31 failed assertions and exactly 31
`DIAG joined=false tid=...` records, with no numeric/output/order or
role-checkpoint diagnostic (`E/stripe-d32-diagnostic.log`).
Thus these observed failures are the test's `roles_probe::joined()` check
that `/proc/self/task/<tid>` has disappeared, not failed matrix values or
dependency ordering.

Actual joins are present at
`ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:986`,
`ggml/src/ggml-potal/matmul/stripe.cpp:697,1731`, and
`ggml/src/ggml-potal/tests/support/stripe-owner.hpp:55-58`.
The original test also explicitly calls `owner.join()` before checking.
The exact kernel/procfs teardown timing is not independently established.
The evidence establishes a timing-sensitive procfs-presence assertion after
the joins and reproduces the failure before this phase. A more verbose
diagnostic run passed; its success-path logging changed timing and is not
used as proof that the original test passes
(`E/stripe-d32-diagnostic-verbose.log`).

No original PoTAL test, production API, numerical assertion, or test
registration was changed to hide this failure. A test-lifecycle correction
is separate follow-up work. The first attempt used an incorrect container
mount, so `E/pre/stripe-d32-isolated.log` contains NOT RUN/no-tests output;
it is explicitly excluded from all reproduction counts.

## Actual matrix commands

Run from repository root on macOS:

```sh
# Native Linux CPU/Gemmini/PoTAL, including PoTAL INT4:
docker run --rm -v "$PWD:/src" -w /src -e OUT_SUFFIX=/phase2a-linux -e JOBS=6 \
  socm-baseline:ubuntu24 bash build-npu/run-npu-matrix.sh host
# RISC-V API, Gemmini, PoTAL and added PB preset:
docker run --rm -v "$PWD:/src" -w /src -e OUT_SUFFIX=/phase2a-linux -e JOBS=6 \
  socm-baseline:ubuntu24 bash build-npu/run-npu-matrix.sh riscv
# macOS lanes (same existing configurations):
for lane in mac-cpu-only gemmini-int8-d16-ref gemmini-a4w4-int8-d16-ref; do
  cmake --build build-npu/$lane -j 6
  ctest --test-dir build-npu/$lane -j 4 --output-on-failure --timeout 600
done
cmake --build build-npu/gemmini-api -j 8
ctest --test-dir build-npu/gemmini-api --output-on-failure
bash build-npu/phase2a-evidence/check-full-goldens.sh
```

The matrix configures Release/Ninja with LLAMA_CURL=OFF and tests ON,
builds with `-- -k 0`, and runs `ctest -j 8 --output-on-failure --timeout 600`.
RISC-V uses `cmake/toolchains/riscv64-linux.cmake`, GGML_RVV=OFF,
GGML_NATIVE=OFF and GGML_OPENMP=OFF. Exact per-lane parameters are retained
in `build-npu/run-npu-matrix.sh`, presets, and each CMakeCache.txt.
The script no longer deletes an existing build directory.
PB commands, independent oracles and native test corrections: `pb-reference.md`.
The resulting `rv-gemmini-pb-d32/bin/gemmini-pb-e2e-runner` and
`rv-gemmini-int8-d16-rocc/bin/llama-cli` are ELF64 RISC-V executables.

Intermediate build/test attempts remain in the evidence directory:
the decomposition initially exposed missing direct includes, and the first
PB builds exposed an incorrectly scoped transport binding. Both were fixed
before the successful builds. The initial broad PB CTest attempts in
`L/gemmini-int8-d32-pb-*.ctest.log` followed those failed builds and are not
used as acceptance evidence. Final PB acceptance uses all four required
numerical tests, both functional-model runs and the two five-core tests,
with their successful logs named in `pb-reference.md`.

## Golden and contract review

PASS: original 13 shapes x four geometries, full hash and stored words;
additional SDK-only full captures compare every command plus drain marker.
`E/golden-full.log`, `E/golden-sha256.json`, and
`E/check-full-goldens.sh` retain the evidence. No original golden changed.

ACC_READ_FULL_WIDTH and memory-ordering findings are in
`gemmini-legacy-removal.md`. No ISA/readback/fence change was made.

## Unverified boundaries

PB end-to-end runner builds and `--help` passes (`E/pb-e2e-help.log`).
LLM generation: NOT RUN, no model-weight GGUF/prompt-token fixture supplied.
Physical RoCC execution, hardware performance/cycles and DMA coherence:
NOT RUN. macOS PB Linux-affinity runtime and external FP32 DIM8 overlay:
NOT RUN; no port or full reimplementation was attempted.
Full LLM generation is BLOCKED by missing model inputs, rather than a
reported successful runner execution. All required Phase 2A implementation,
contract review and regression work is complete; these unverified boundaries
and the pre-existing PoTAL fixture/lifecycle failures remain explicit.
