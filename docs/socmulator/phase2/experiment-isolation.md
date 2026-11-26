# Phase 2C experiment isolation

Phase 2C organizes existing implementations by responsibility. It does not
change numerical algorithms, command encodings, scheduling, completion,
buffer ownership, common types, profiling schemas or model formats.
Current source lists and callers supersede the Phase 0 inventories.
Evidence root: `build-npu/phase2c-evidence/`.

| Classification | Current owner and evidence | Disposition |
| --- | --- | --- |
| Production numerics | `ggml-npu-common/{quantization,residual,types,buffer}`; Gemmini and PoTAL backend source lists consume these sources through the common extraction target | Retained. The direct residual executor only changes CUDA header locations. |
| Production hardware execution | Gemmini `api/`, `matmul/tiling.cpp`; each PoTAL `potal.c`, `matmul/{paired-auto,execution,stripe,attention}.cpp` | Retained byte-for-byte. |
| Supported comparison baseline | Gemmini INT8, software A4W4 on INT8, `pb/`, `fp32/` | Kept independent. PB command-functional reference is not generic CPU matmul. FP32 has separate DIM8 presets and its own external overlay. |
| Accuracy evaluation | `ggml-npu-common/evaluation/{cuda-exact,cuda-rtn,gemmini}` | CUDA kernels moved from `ggml-cuda/cpu-exact`; graph adapter moved from Gemmini. Explicit existing evaluation options select them. |
| Historical prototype | Each PoTAL `experiments/{microtile-pipeline,resident-weight,sequential-paired,prefetch-admission}` | Former test-local implementations and benchmarks are built only by their existing opt-in options. |
| Active contract test | Each backend's `tests/`, including E3 prototype/context tests, E4A resident tests, E4B1 contract tests, E4B2 C11/emulator/hardware tests, E4C1 correctness tests | Kept. Tests of a proposed prefetch protocol move with that model; target and CTest identity remain. |
| Historical document | `docs/archive/potal/` | Thirty-six H3 reports moved with identical bytes. Four live backend contract documents remain in their original locations. Negative results and external provenance retained. |

The production PoTAL route is `ops.cpp -> matmul/execution.cpp -> stripe/dense
-> paired-auto.cpp -> potal.c`. In `paired-auto.cpp`, request policy selects
resident, K-tile, next-K, early-LdB, streaming and index-queue paths.
Capability checks and checked session operations gate the hardware calls.
These are production mechanisms despite historical H/E names. Resident B
lifetime, indexed preload, scale generations, Main/Residual ACC scheduling,
LDB_ONLY, STREAM, Index Queue, supported prefetch and managed completion
remain in their existing owners, with opt-in defaults unchanged.

In contrast, E3's `potal-e3-context.c` includes the production `potal.c` in a
test-local replacement translation unit; E4A's resident library also replaces
the API only within its closed harness. E4B1 declares proposed metadata with
no issuer. E4C1's benchmark substitutes `ops.cpp` only for that executable.
E4C2-0 models proposed prefix queries and explicitly denies that they are
current hardware observations. None belongs in the production source list.
The exact old/new paths are in `source-moves.json`; no implementation or
test oracle was deleted as a duplicate.

PB remains `GGML PB -> K-block preparation -> WS tiler -> encoder ->
RoCC or owned FunctionalTransport -> INT32 partial -> ordered FP32 scaling`.
`NPU_MATMUL_ALGORITHM=PARTIAL_SUM` accepts REFERENCE and ROCC. Both D32
presets retain INT8, SPAD 4 x 4096, ACC 2 / 1024 total rows, DMA 64.
The manifest remains unverified against a deployed RTL geometry. The
five-core issuer owns its model and retains it through completion.
No PB implementation, tiler, encoder or transport source changes in 2C.

See [FP32](fp32-baseline.md), [CUDA](cuda-evaluation.md),
[PoTAL](potal-experiments.md), [dependency evidence](phase2c-dependencies.md),
[regression results](phase2c-test-results.md), and
[historical report index](../../archive/potal/README.md).
Earlier audit documents remain snapshots rather than being bulk rewritten.
