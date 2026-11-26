# CUDA accuracy evaluation

The explicit evaluation owner is
`ggml/src/ggml-npu-common/evaluation/`. Kernel and C API moves:

| Previous owner | Current owner | Existing target |
| --- | --- | --- |
| `ggml-cuda/cpu-exact/ggml-cuda-{cpu-exact,hp1-scu,potal-attention}.{cu,h}` | `evaluation/cuda-exact/` | `ggml-gemmini-cuda-cpu-exact` |
| `ggml-cuda/cpu-exact/ggml-cuda-rtn.{cu,h}` | `evaluation/cuda-rtn/` | `ggml-cuda-rtn` |
| `ggml-gemmini/ggml-gpu-cpu-exact.hpp` | `evaluation/gemmini/graph.hpp` | Header adapter used only by evaluation-enabled graph consumers |
| CUDA option block in `ggml/CMakeLists.txt` | `evaluation/options.cmake` | Existing public options retained |
| CUDA target block in Gemmini `CMakeLists.txt` | `evaluation/backend.cmake` | Existing static library names retained |

Stock `ggml-cuda/CMakeLists.txt` and its normal kernels are untouched. Its
nonrecursive source glob did not own the moved comparator directory.
`GGML_CUDA` still selects stock llama.cpp CUDA. Neither it nor a runtime
environment variable implicitly enables NPU accuracy interception.
There is no `NPU_TRANSPORT=CUDA`.

`GGML_GEMMINI_CUDA_CPU_EXACT` explicitly selects the Gemmini CPU host graph,
matched A4W4/A8W8 INT preparation, and BLOCK with RMD off or EXSIA. Existing
validation rejects other GPU backends, dynamic registration, non-CPU policy
and incompatible quantization. `GGML_GEMMINI_CUDA_RTN` selects RTN/FP16
comparison with compensation and explicit PoTAL attention off.
`GGML_GEMMINI_CUDA_RTN_ATTENTION` requires CPU-exact, preserving its baseline
attention reuse. Option and environment names are retained for Phase 2D.

Call graph: explicit compile definitions enable the existing graph
interception in `ggml-gemmini.cpp` and support dispatch in `ops.cpp`.
`matmul/dense.cpp` calls raw integer-dot/HP1 comparison APIs; attention
and stationary-cache adapters call the quantized attention C API.
`evaluation/gemmini/graph.hpp` retains RTN attention selection and proof
counters. Buffer teardown retains the existing CUDA weight-cache cleanup.
Perplexity verification and existing CUDA tests use the moved headers.
The NVFP4 reference build script uses the relocated header directory.

The common direct residual executor calls the common evaluation C APIs
directly under `GGML_GEMMINI_CUDA_CPU_EXACT`; it does not include the
Gemmini graph adapter. Its call conditions, event packing, HP1/SCU function
choice, staged INT64 output, overflow rejection and publication are unchanged.
In production configurations those definitions are absent, so CUDA headers,
branches and symbols do not enter the compiler/link closure. Evaluation
targets alone request CUDA language/toolkit, cudart and cuBLAS.

All eight moved kernel/header files retain identical bytes. Quantized
operand semantics, CPU-exact integer results, RTN rounding, HP1/SCU behavior,
attention arithmetic, metrics and model formats are preserved. Source
consumers change include paths only. GPU latency is accuracy-evaluation
latency; it is never an NPU RoCC or physical timing measurement.

Actual GPU numerical execution is BLOCKED: no nvcc/toolkit or GPU execution
environment is available. `build-npu/phase2c-evidence/optional/`
contains `cuda-exact.configure.log` and `cuda-rtn.configure.log`, which reach
the evaluation-owned `enable_language(CUDA)` and fail on missing nvcc.
Existing tests remain registered only when explicitly enabled:
`test-cuda-cpu-exact-graph`, `test-cuda-hp1-scu`,
`test-cuda-hp1-scu-graph`, `test-cuda-potal-attention`,
`test-potal-sensitivity`, `test-potal-sensitivity-invalid`, `test-cuda-rtn`.
No unavailable GPU test is reported PASS. Production evaluation-OFF builds,
target/dependency evidence and retained regressions are recorded separately.
