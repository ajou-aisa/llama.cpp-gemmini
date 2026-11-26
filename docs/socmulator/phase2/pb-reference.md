# PB command-functional reference

PB is the Gemmini baseline. Its native REFERENCE lane executes the same
WS command lowering as the RoCC lane:

```
GGML PB linear / attention
  -> existing K-block packing (32 elements)
  -> matmul/execution.cpp -> matmul/tiling.cpp
  -> C11 descriptors and api/encode.c
  -> pb::FunctionalTransport (pb/pb-rocc-model.{hpp,cpp})
  -> decoded CONFIG / LOOP_WS commands -> INT32 partial output
  -> existing ordered FP32 post-scaling
```

The model owns operand-reuse caches, configuration, and accumulator state.
It decodes real encoded words, checks complete WS configuration, bounds,
strides, full-width output, reuse IDs, and unfinished accumulation.
Invalid command sequences produce a sticky transport failure; execution
does not turn that into success. It supports PB's no-bias,
non-transposed INT8 WS contract, not every Gemmini operation.

Each synchronous PB execution owns a model. The existing five-core runtime
constructs its model on the issuer thread; that model lives until the issuer
exits and is never selected by another thread through a global accessor.
The existing release/acquire request handoff, buffer lifetimes, flush
placement and final tiler drain remain unchanged. The scalar numerical
reference and standalone recording transport remain separate: the recorder
does not compute, and PB never substitutes a generic CPU matmul.

## Configuration

`NPU_TRANSPORT=ROCC|REFERENCE` selects command delivery.
`NPU_MATMUL_ALGORITHM=BASELINE|PARTIAL_SUM` selects implementation.
An empty algorithm derives from the existing `GGML_GEMMINI_PB` option.
PARTIAL_SUM derives PB=ON and WS execution for both transports.
BASELINE/REFERENCE retains generic CPU-reference execution.
Conflicting explicit old options are rejected.
Generated configuration/manifest JSON records algorithm and execution policy
separately from transport.

Presets:

- `gemmini-int8-d32-pb-reference`
- `gemmini-int8-d32-pb-rocc`

Both inherit the existing Gemmini geometry generator: INT8, D32,
four SPAD banks of 4096 rows, two accumulator banks, 1024 total accumulator
rows, DMA maximum 64 bytes. These are the exact values of the previously
verified PB D32 cross-build, not PoTAL D32's smaller memories.
The manifest correctly marks this Gemmini geometry as lacking a matching
RTL reference header. Software geometry/functional validation does not
certify a deployed hardware configuration.

PB uses COMPUTE_TYPE=INT, 8-bit activation and weight storage,
BLOCK activation quantization, block size 32, and RMD=OFF.
`GGML_GEMMINI_PB_WEIGHT_FORMAT=Q8_0|MXINT8` keeps both numeric baselines.
The tested native runtime is Linux arm64. macOS PB's Linux affinity/thread
runtime was not ported; the separate FP32 overlay was not changed.

## Commands actually used

Host is macOS arm64; native Linux runs inside `socm-baseline:ubuntu24`.
The container mounts the source at `/src`, matching the saved baseline.

```sh
docker run --rm -v "$PWD:/src" -w /src \
  -e OUT_SUFFIX=/phase2a-linux -e JOBS=12 \
  socm-baseline:ubuntu24 bash build-npu/run-npu-matrix.sh pb

# Equivalent preset configurations, performed by that matrix:
cmake --preset gemmini-int8-d32-pb-reference \
  -B build-npu/phase2a-linux/gemmini-int8-d32-pb-reference \
  -DLLAMA_CURL=OFF -DGGML_GEMMINI_PB_WEIGHT_FORMAT=Q8_0
cmake --preset gemmini-int8-d32-pb-reference \
  -B build-npu/phase2a-linux/gemmini-int8-d32-pb-mxint8-reference \
  -DLLAMA_CURL=OFF -DGGML_GEMMINI_PB_WEIGHT_FORMAT=MXINT8
cmake --build build-npu/phase2a-linux/gemmini-int8-d32-pb-reference -j 12
cmake --build build-npu/phase2a-linux/gemmini-int8-d32-pb-mxint8-reference -j 12
ctest --test-dir build-npu/phase2a-linux/gemmini-int8-d32-pb-reference \
  --output-on-failure -R '^gemmini-(pb-|functional-test)'
ctest --test-dir build-npu/phase2a-linux/gemmini-int8-d32-pb-mxint8-reference \
  --output-on-failure -R '^gemmini-(pb-|functional-test)'

# Existing issuer + three helpers + coordinator, on available container CPUs:
GGML_GEMMINI_PB_TEST_FIVE_CORE=0:1,2,3 taskset -c 4 \
  build-npu/phase2a-linux/gemmini-int8-d32-pb-reference/bin/gemmini-pb-q8-0-block-test
GGML_GEMMINI_PB_TEST_FIVE_CORE=0:1,2,3 taskset -c 4 \
  build-npu/phase2a-linux/gemmini-int8-d32-pb-reference/bin/gemmini-pb-q8-0-attention-test
```

## Numerical evidence and discovered oracle defects

PASS: `gemmini-pb-block-test`, `gemmini-pb-attention-test`,
`gemmini-pb-q8-0-block-test`, `gemmini-pb-q8-0-attention-test`.
Both lanes also pass `gemmini-functional-test`, including deterministic
interleaving of two independent model contexts and sticky-failure isolation.
The existing PB block observations now verify actual decoded command deltas:
five configurations, one WS launch and one drain for each fixture K block.
INT32 partials and intermediate/final FP32 results are checked against
independent arithmetic; MXINT8 adds intermediate post-scaling checks.
QK/PV tests run actual GGML graphs, including F16/F32 cache views and softmax.

Two previously unrun oracle defects were exposed and fixed without changing
production quantization:

1. AArch64 defines NEON even without `-march`. The Q8_0 test's local
   `ggml-cpu-quants.c` was selecting vector rounding/accumulation despite
   claiming to be scalar. Its target now explicitly undefines NEON for the
   test oracle only. Production still uses `quantize_row_q8_0_ref`.
2. MXINT8's independent test oracle rounded ties away; current production
   `quantize_mxint` uses `round_nearest_even_i32`. The oracle now implements
   ties-to-even independently with floor/parity, and an explicit positive
   and negative half-integer fixture checks actual production codes.
   Every existing code, partial-result and output assertion remains.

Evidence under `build-npu/phase2a-evidence/`:
`gemmini-int8-d32-pb-reference.verified-{build,tests}.log`,
`pb-mxint8-final-{build,tests}.log`,
`gemmini-pb-q8-0-{block,attention}-test.five-core.log`.
Five-core tests report zero CPU-placement mismatches; both numeric suites pass.
Their timings are host runtime observations, never hardware performance.
An isolated count-omission mutation compiles but fails specifically at
`actual decoded command counts` (`pb-model-count-mutant.{build,test}.log`);
production files were not mutated.

`gemmini-pb-e2e-runner` builds and `--help` passes (`pb-e2e-help.log`).
Full LLM generation is NOT RUN: the repository contains vocabulary GGUFs,
but no model-weight GGUF or prompt-token fixture was supplied.
Physical RoCC execution, DMA coherence, hardware counters and cycle accuracy
are NOT RUN. Cross-build results appear in `phase2a-test-results.md`.
