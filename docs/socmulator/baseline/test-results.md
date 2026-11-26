# Baseline and Phase 1 test results

Status values: PASS, FAIL, NOT RUN, BLOCKED, NOT APPLICABLE. A failure is
"pre-existing" only when the same test fails in the baseline build of the
same configuration (llama.cpp `64a072a`, worktree
`reference/llama.cpp-baseline-64a072a`). No hardware was available: every
hardware-execution, counter and cycle result is **NOT RUN**.

Hosts: macOS 25.5 arm64 (Apple clang 21, CMake 4.3); Linux = Docker
`ubuntu:24.04` arm64 (GCC 13.3, CMake 3.28); RISC-V = Ubuntu
`riscv64-linux-gnu-gcc` 13.3 cross compiler (not the FireSim toolchain).
Logs: `build-baseline/logs`, `reference/llama.cpp-baseline-64a072a/build-baseline/linux/logs`
(baseline), `build-npu/logs`, `build-npu/linux/logs` (Phase 1); not in git.

## Commands

Baseline (`build-baseline/run-matrix.sh`, run in the baseline worktree):

```sh
cmake -S . -B <dir> -G Ninja -DCMAKE_BUILD_TYPE=Release [-DLLAMA_CURL=OFF on Linux] <config>
cmake --build <dir> -j N [-- -k 0]
ctest -j 8 --output-on-failure --timeout 600
```

| Baseline config | Options |
|---|---|
| cpu-only | (defaults) |
| gemmini-int8-d16-cpu / -ws | `-DGGML_GEMMINI=ON -DGGML_POTAL=OFF -DLLAMA_BUILD_TESTS=ON -DGEMMINI_SW_PATH=<develop fcfcf5ce> -DGGML_GEMMINI_DIM=16 -DGGML_GEMMINI_OPTION=CPU\|WS` |
| potal-api-only-d16 | `-DPOTAL_API_ONLY=ON -DPOTAL_HW_PARAMS=<chipyard potal>/gemmini_potal_params.h -DPOTAL_EXPECTED_DIM=16` |
| potal-int8-d16 / d32 | `-DGGML_POTAL=ON -DGGML_GEMMINI=OFF -DLLAMA_BUILD_TESTS=ON -DPOTAL_HW_PARAMS=<header> -DPOTAL_EXPECTED_DIM=N [-DGGML_GEMMINI_DIM=32]` |
| potal-int4-d32 | as above with `-DGGML_POTAL_W4A4=ON`, the W4A4 header, `-DGGML_GEMMINI_ACTIVATION_BITS=4 -DGGML_GEMMINI_WEIGHT_BITS=4 -DGGML_GEMMINI_EXSIA_STRIPE_ANCHOR=3 -DGGML_GEMMINI_ENABLE_RMD=ON -DGGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=WS -DGGML_POTAL_PAIRED_AUTO=ON` |
| rv-* | `-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 -DCMAKE_C(XX)_COMPILER=riscv64-linux-gnu-gcc/g++ -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF [-DGGML_RVV=OFF]` |

Phase 1 (`build-npu/run-npu-matrix.sh`): the same configurations through
`GGML_NPU` + `NPU_*` (values in configurations.json) and
`NPU_TRANSPORT=REFERENCE` (host) / `ROCC` with
`cmake/toolchains/riscv64-linux.cmake` (RISC-V).

## Build results

| Configuration | Host | Baseline | Phase 1 |
|---|---|---|---|
| CPU-only | macOS | PASS | PASS |
| CPU-only | Linux | not built at baseline (see Gemmini/PoTAL rows: same single pre-existing link failure) | FAIL: only `test-gemmini-cycle-reader-aarch64` link (pre-existing, also in every Linux baseline build) |
| Gemmini INT8 D16 CPU reference | macOS | PASS (with external SDK) | PASS (no SDK) |
| Gemmini INT8 D16 WS on host | macOS | PASS (RoCC macros were silent no-ops) | NOT APPLICABLE: `NPU_TRANSPORT=ROCC` is rejected on non-riscv64 by design |
| Gemmini INT8 D16 | Linux | FAIL: `test-gemmini-cycle-reader-aarch64` link only | same single failure |
| Gemmini standalone API (`NPU_API_ONLY`) | macOS | did not exist | PASS |
| Gemmini INT8 D16 RoCC | RISC-V cross | FAIL: `rocc-software/src/xcustom.h: No such file or directory` (develop SDK has no xcustom.h) + ggml-cpu RVV intrinsics (GCC 13) | PASS with `GGML_RVV=OFF` (full llama.cpp incl. `llama-cli`, ELF riscv64) |
| Gemmini API RoCC | RISC-V cross | did not exist | PASS: `gemmini-api-smoke` ELF; disassembly shows 16 custom-3 instructions, funct7 0-14 with funct3=3 and COUNTER 126 with funct3=7 |
| PoTAL API-only D16 | macOS | PASS | PASS (header generated, sha256 = RTL header) |
| PoTAL API-only D16 | Linux | FAIL: `potal.c:3149 'CLOCK_MONOTONIC' undeclared` (strict C11 on glibc) | not re-run (same source) |
| PoTAL API-only D32 / INT4 D32 | macOS | PASS | PASS |
| PoTAL API-only D64 | macOS | FAIL: `potal-stripe-api-test.cpp:94 static_assert D == 16 \|\| D == 32` | same failure |
| PoTAL INT8 D16 / D32 | Linux | FAIL: `test-gemmini-cycle-reader-aarch64` link only | same single failure |
| PoTAL INT8 D16 | macOS | FAIL: `trace/cpu-sample.hpp` uses `cpu_set_t`/`sched_getcpu` (Linux-only) | not re-run (same source) |
| PoTAL INT4 D32 | Linux | FAIL: `potal-attention-test.cpp` compile + aarch64 link | same two failures (generated header named `gemmini_potal_w4a4_d32_params.h`, sha256 = RTL) |
| PoTAL INT8 D16 RoCC | RISC-V cross | FAIL with RVV (GCC 13 intrinsics); PASS with `GGML_RVV=OFF` | PASS (`NPU_TRANSPORT=ROCC`, toolchain file, `GGML_RVV=OFF`) |

## CTest results

Pass/fail sets are compared name by name (baseline → Phase 1).

| Configuration | Baseline | Phase 1 | Difference |
|---|---|---|---|
| CPU-only, macOS | 52/60 pass | 52/60 pass | identical set |
| Gemmini INT8 D16, macOS | 103/117 pass | 109/123 pass | identical set + 6 new PASS: gemmini-api-test, gemmini-encoding-test, gemmini-transport-test, gemmini-tiling-test, gemmini-functional-test, npu-config-test |
| Gemmini INT8 D16, Linux | 99/118 pass | 105/124 pass | identical set + the same 6 new PASS |
| Gemmini standalone API, macOS | n/a | 5/5 PASS | - |
| PoTAL API-only D16 / D32 / INT4 D32, macOS | 4/4, 3/4, 3/4 | 4/4, 3/4, 3/4 | identical (`potal-api-test` fails at D32: "stream failure isolation requires fork") |
| PoTAL INT8 D16, Linux | 73/87 | 73/87 | identical set |
| PoTAL INT8 D32, Linux | 70/85 | 69/85 | `potal-stripe-pipeline-test` PASS → FAIL; nondeterministic, see below |
| PoTAL INT4 D32, Linux | 64/77 | 64/77 | identical set |

Pre-existing failures (fail in the baseline of the same configuration):
`test-chat` (template files not found from the build dir), `test-gguf`,
`test-backend-ops` (macOS, aborts), `test-gemmini-cpu-cycle-validity-contract`,
`test-build-arm64-log-defaults`, `test-gemmini-off-build`,
`test-im2p-cache-build-scripts` (needs IM2P.sim), `test-potal-target-profile`
(needs `../../gemmini-rocc-tests`), `gemmini_weight_formats`,
`gemmini_mxint_weight_routes_activation`, `test-gemmini-exsia-profile-contract`,
`test-gemmini-build-contract`, `test-gemmini-hp1-contract`,
`test-gemmini-profiled-domain-contract`; Linux additionally
`test-experiment-output`, `test-evaluation-build-options`, `test-eval-callback`,
`gemmini_direct_metrics_source_contract`, `test-gemmini-rmd`,
`test-gemmini-cycle-reader-aarch64` (NOT RUN, link failure), and PoTAL
`potal-log-test`, `potal-index-queue-test`, `potal-hw-timeline-test`,
`potal-paired-schedule-test`, `potal-paired-unit-test` (configuration-dependent).

### potal-stripe-pipeline-test (nondeterministic)

Its `ROLES ... dependencies` checks fail at a different run index each time
and depend on machine load: baseline D16 failed (run 12), baseline D32
passed, Phase 1 D16 failed (run 16), Phase 1 D32 failed (run 6). Repeated
alone on an idle-ish container, the baseline D32 binary passed 6/6 while the
Phase 1 D32 binary passed 3/6 with concurrent cross builds running. The two
builds use identical CMake options and compile definitions (CMakeCache and
build.ninja `-D` sets diffed; only output paths differ) and byte-identical
parameter headers, and Phase 1 changes no PoTAL source. In a final A/B run (one container, the
baseline and Phase 1 binaries alternated five times after a full build),
**both failed 5/5**. The test is load-dependent in both builds; it is
classified as a pre-existing nondeterministic failure, not a Phase 1
regression.

## Golden references

| Target | Comparison | Status |
|---|---|---|
| Gemmini instruction encoding (CONFIG EX/LD/ST, MVIN/MVIN2/MVIN3, MVOUT, PRELOAD, COMPUTE, FLUSH, LOOP_WS ×6, COUNTER) | bitwise vs SDK macro expressions | PASS (gemmini-encoding-test) |
| Gemmini command sequence, 13 shapes × 4 geometries (D16 4×4096/1024, D16 4×2048/512, D32 4×2048/512, D64 4×1024/256) | ordered exact (count + FNV-1a of normalized stream + first 24 commands) vs captures of the verbatim SDK tiler | PASS (gemmini-tiling-test) |
| Gemmini tile selection (`gemmini_set_tile_ws`) | exact | PASS |
| Gemmini command semantics | tiler stream executed by the independent LoopMatmul host model == naive INT32 GEMM; shapes incl. sub-DIM, non-multiple, M=1, K=4096 with ±127/-128 boundaries | PASS (gemmini-functional-test) |
| Gemmini CPU reference (`matmul_cpu`) | bitwise output hash vs SDK capture, 7 cases (all D16 cases without `low_D` and below 2^22 MACs) | PASS |
| Parameter headers | sha256 of generated header == RTL/SDK header for the 5 reference configurations | PASS (configure-time check) |
| Quantized activation, weight packing, scale metadata, INT64 reconstruction, attention, FP32 output | existing test suites (exsia, rmd, potal-*) unchanged | PASS where the baseline passes (identical sets) |
| LLM output (token IDs, logits) | needs a GGUF model and a target | NOT RUN (no model run in this session) |
| Hardware counters, cycles | same configuration on hardware | NOT RUN (no hardware) |

DMA addresses are normalized to buffer+offset; no other field is masked.
