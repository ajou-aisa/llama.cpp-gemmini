# Phase 2B ownership and rename map

Paths are repository-relative. These are ownership moves, with caller includes
and source lists migrated. No source forwarding header was created. Historical
documents, public GGML backend registration, GGUF identifiers, golden identities,
external trace fields and environment variables retain their existing names.

| Original file | Current owner |
| --- | --- |
| `ggml/src/ggml-gemmini-utils/CMakeLists.txt` | `ggml/src/ggml-npu-common/profiling/CMakeLists.txt` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp` | `ggml/src/ggml-npu-common/profiling/include/npu/cpu_log_context.hpp` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/cpu-timing.h` | `ggml/src/ggml-npu-common/profiling/include/npu/cpu-timing.h` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.h` | `ggml/src/ggml-npu-common/profiling/include/npu/cycle_reader.h` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp` | `ggml/src/ggml-npu-common/profiling/include/npu/cycle_reader.hpp` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp` | `ggml/src/ggml-npu-common/profiling/include/npu/evaluation_metrics.hpp` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/host-timing.hpp` | `ggml/src/ggml-npu-common/profiling/include/npu/host-timing.hpp` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/layer.h` | `ggml/src/ggml-npu-common/profiling/include/npu/layer.h` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/layer.hpp` | `ggml/src/ggml-npu-common/profiling/include/npu/layer.hpp` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/log.h` | `ggml/src/ggml-npu-common/profiling/include/npu/log.h` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp` | `ggml/src/ggml-npu-common/profiling/include/npu/log.hpp` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/optrace.hpp` | `ggml/src/ggml-npu-common/profiling/include/npu/optrace.hpp` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/performance.hpp` | `ggml/src/ggml-npu-common/profiling/include/npu/performance.hpp` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/semantic.h` | `ggml/src/ggml-npu-common/profiling/include/npu/semantic.h` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/semantic.hpp` | `ggml/src/ggml-npu-common/profiling/include/npu/semantic.hpp` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/trace-context.h` | `ggml/src/ggml-npu-common/profiling/include/npu/trace-context.h` |
| `ggml/src/ggml-gemmini-utils/include/gemmini/trace-context.hpp` | `ggml/src/ggml-npu-common/profiling/include/npu/trace-context.hpp` |
| `ggml/src/ggml-gemmini-utils/src/cycle_reader_aarch64.cpp` | `ggml/src/ggml-npu-common/profiling/src/cycle_reader_aarch64.cpp` |
| `ggml/src/ggml-gemmini-utils/src/cycle_reader_internal.h` | `ggml/src/ggml-npu-common/profiling/src/cycle_reader_internal.h` |
| `ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp` | `ggml/src/ggml-npu-common/profiling/src/cycle_reader-capi.cpp` |
| `ggml/src/ggml-gemmini-utils/src/cycle.cpp` | `ggml/src/ggml-npu-common/profiling/src/cycle.cpp` |
| `ggml/src/ggml-gemmini-utils/src/debug.cpp` | `ggml/src/ggml-npu-common/profiling/src/debug.cpp` |
| `ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp` | `ggml/src/ggml-npu-common/profiling/src/evaluation_metrics.cpp` |
| `ggml/src/ggml-gemmini-utils/src/log-capi.cpp` | `ggml/src/ggml-npu-common/profiling/src/log-capi.cpp` |
| `ggml/src/ggml-gemmini-utils/src/optrace.cpp` | `ggml/src/ggml-npu-common/profiling/src/optrace.cpp` |
| `ggml/src/ggml-gemmini-utils/src/performance.cpp` | `ggml/src/ggml-npu-common/profiling/src/performance.cpp` |
| `ggml/src/ggml-gemmini-utils/src/semantic.cpp` | `ggml/src/ggml-npu-common/profiling/src/semantic.cpp` |
| `ggml/src/ggml-gemmini-utils/src/trace-context.cpp` | `ggml/src/ggml-npu-common/profiling/src/trace-context.cpp` |
| `ggml/src/ggml-gemmini-utils/src/trace-metadata.hpp` | `ggml/src/ggml-npu-common/profiling/src/trace-metadata.hpp` |
| `ggml/src/ggml-gemmini/ggml-gemmini-config.hpp` | `ggml/src/ggml-npu-common/config/software.hpp` |
| `ggml/src/ggml-gemmini/ggml-gemmini-evaluation-observer.hpp` | `ggml/src/ggml-npu-common/profiling/evaluation-observer.hpp` |
| `ggml/src/ggml-gemmini/ggml-gemmini-matmul-config.hpp.in` | `ggml/src/ggml-npu-common/config/matmul.hpp.in` |
| `ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp` | `ggml/src/ggml-npu-common/profiling/telemetry.cpp` |
| `ggml/src/ggml-gemmini/ggml-gemmini-telemetry.hpp` | `ggml/src/ggml-npu-common/profiling/telemetry.hpp` |
| `ggml/src/ggml-gemmini/matmul/options.cpp` | `ggml/src/ggml-npu-common/matmul/options.cpp` |
| `ggml/src/ggml-gemmini/matmul/options.hpp` | `ggml/src/ggml-npu-common/matmul/options.hpp` |
| `ggml/src/ggml-gemmini/quants/act/block/block.cpp` | `ggml/src/ggml-npu-common/quantization/activation/block/block.cpp` |
| `ggml/src/ggml-gemmini/quants/act/block/block.hpp` | `ggml/src/ggml-npu-common/quantization/activation/block/block.hpp` |
| `ggml/src/ggml-gemmini/quants/act/block/types.hpp` | `ggml/src/ggml-npu-common/quantization/activation/block/types.hpp` |
| `ggml/src/ggml-gemmini/quants/act/buffer.hpp` | `ggml/src/ggml-npu-common/buffer/activation.hpp` |
| `ggml/src/ggml-gemmini/quants/act/dispatch.cpp` | `ggml/src/ggml-npu-common/quantization/activation/dispatch.cpp` |
| `ggml/src/ggml-gemmini/quants/act/dispatch.hpp` | `ggml/src/ggml-npu-common/quantization/activation/dispatch.hpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/exsia_float.hpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/exsia_float.hpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/exsia_shift.hpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/exsia_shift.hpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/exsia-event.hpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/exsia-profile.cpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/exsia-profile.hpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/exsia-state.hpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/exsia-state.hpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/exsia.cpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/exsia.hpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/folding.cpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/folding.hpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/folding.hpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/local.cpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/local.cpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/local.hpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/local.hpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/strict-profile.hpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/strict-profile.hpp` |
| `ggml/src/ggml-gemmini/quants/act/exsia/types.hpp` | `ggml/src/ggml-npu-common/quantization/activation/exsia/types.hpp` |
| `ggml/src/ggml-gemmini/quants/act/meta.hpp` | `ggml/src/ggml-npu-common/quantization/activation/meta.hpp` |
| `ggml/src/ggml-gemmini/quants/act/preparation-cache.cpp` | `ggml/src/ggml-npu-common/quantization/activation/preparation-cache.cpp` |
| `ggml/src/ggml-gemmini/quants/act/preparation-cache.hpp` | `ggml/src/ggml-npu-common/quantization/activation/preparation-cache.hpp` |
| `ggml/src/ggml-gemmini/quants/act/quantize.cpp` | `ggml/src/ggml-npu-common/quantization/activation/quantize.cpp` |
| `ggml/src/ggml-gemmini/quants/act/quantize.hpp` | `ggml/src/ggml-npu-common/quantization/activation/quantize.hpp` |
| `ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp` | `ggml/src/ggml-npu-common/quantization/activation/stripe/stripe.cpp` |
| `ggml/src/ggml-gemmini/quants/act/stripe/stripe.hpp` | `ggml/src/ggml-npu-common/quantization/activation/stripe/stripe.hpp` |
| `ggml/src/ggml-gemmini/quants/act/stripe/types.hpp` | `ggml/src/ggml-npu-common/quantization/activation/stripe/types.hpp` |
| `ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp` | `ggml/src/ggml-npu-common/quantization/activation/tensor/tensor.cpp` |
| `ggml/src/ggml-gemmini/quants/act/tensor/tensor.hpp` | `ggml/src/ggml-npu-common/quantization/activation/tensor/tensor.hpp` |
| `ggml/src/ggml-gemmini/quants/act/tensor/types.hpp` | `ggml/src/ggml-npu-common/quantization/activation/tensor/types.hpp` |
| `ggml/src/ggml-gemmini/quants/act/token/token.cpp` | `ggml/src/ggml-npu-common/quantization/activation/token/token.cpp` |
| `ggml/src/ggml-gemmini/quants/act/token/token.hpp` | `ggml/src/ggml-npu-common/quantization/activation/token/token.hpp` |
| `ggml/src/ggml-gemmini/quants/act/token/types.hpp` | `ggml/src/ggml-npu-common/quantization/activation/token/types.hpp` |
| `ggml/src/ggml-gemmini/quants/act/types.hpp` | `ggml/src/ggml-npu-common/quantization/activation/types.hpp` |
| `ggml/src/ggml-gemmini/quants/common/dequant.cpp` | `ggml/src/ggml-npu-common/quantization/common/dequant.cpp` |
| `ggml/src/ggml-gemmini/quants/common/dequant.hpp` | `ggml/src/ggml-npu-common/quantization/common/dequant.hpp` |
| `ggml/src/ggml-gemmini/quants/common/fp16_util.cpp` | `ggml/src/ggml-npu-common/quantization/common/fp16_util.cpp` |
| `ggml/src/ggml-gemmini/quants/common/fp16_util.hpp` | `ggml/src/ggml-npu-common/quantization/common/fp16_util.hpp` |
| `ggml/src/ggml-gemmini/quants/common/hp1_scu.hpp` | `ggml/src/ggml-npu-common/quantization/common/hp1_scu.hpp` |
| `ggml/src/ggml-gemmini/quants/common/math.hpp` | `ggml/src/ggml-npu-common/quantization/common/math.hpp` |
| `ggml/src/ggml-gemmini/quants/common/rounding.hpp` | `ggml/src/ggml-npu-common/quantization/common/rounding.hpp` |
| `ggml/src/ggml-gemmini/quants/common/tensor_util.hpp` | `ggml/src/ggml-npu-common/quantization/common/tensor_util.hpp` |
| `ggml/src/ggml-gemmini/quants/common/weight_reader.cpp` | `ggml/src/ggml-npu-common/quantization/common/weight_reader.cpp` |
| `ggml/src/ggml-gemmini/quants/common/weight_reader.hpp` | `ggml/src/ggml-npu-common/quantization/common/weight_reader.hpp` |
| `ggml/src/ggml-gemmini/quants/common/weight_route.hpp` | `ggml/src/ggml-npu-common/quantization/common/weight_route.hpp` |
| `ggml/src/ggml-gemmini/quants/stripe.hpp` | `ggml/src/ggml-npu-common/quantization/stripe.hpp` |
| `ggml/src/ggml-gemmini/residual/direct/direct-builder.hpp` | `ggml/src/ggml-npu-common/residual/direct/direct-builder.hpp` |
| `ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp` | `ggml/src/ggml-npu-common/residual/direct/direct-executor.cpp` |
| `ggml/src/ggml-gemmini/residual/direct/direct-executor.hpp` | `ggml/src/ggml-npu-common/residual/direct/direct-executor.hpp` |
| `ggml/src/ggml-gemmini/residual/direct/direct-profile.hpp` | `ggml/src/ggml-npu-common/residual/direct/direct-profile.hpp` |
| `ggml/src/ggml-gemmini/residual/direct/direct-stage-profile.hpp` | `ggml/src/ggml-npu-common/residual/direct/direct-stage-profile.hpp` |
| `ggml/src/ggml-gemmini/residual/direct/direct-types.hpp` | `ggml/src/ggml-npu-common/residual/direct/direct-types.hpp` |
| `ggml/src/ggml-gemmini/residual/residual-capture.hpp` | `ggml/src/ggml-npu-common/residual/residual-capture.hpp` |
| `ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp` | `ggml/src/ggml-npu-common/residual/rmd/rmd-bitmap-builder.cpp` |
| `ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.hpp` | `ggml/src/ggml-npu-common/residual/rmd/rmd-bitmap-builder.hpp` |
| `ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp` | `ggml/src/ggml-npu-common/residual/rmd/rmd-builder.cpp` |
| `ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp` | `ggml/src/ggml-npu-common/residual/rmd/rmd-builder.hpp` |
| `ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp` | `ggml/src/ggml-npu-common/residual/rmd/rmd-compose.cpp` |
| `ggml/src/ggml-gemmini/residual/rmd/rmd-compose.hpp` | `ggml/src/ggml-npu-common/residual/rmd/rmd-compose.hpp` |
| `ggml/src/ggml-gemmini/residual/rmd/rmd-reference.cpp` | `ggml/src/ggml-npu-common/residual/rmd/rmd-reference.cpp` |
| `ggml/src/ggml-gemmini/residual/rmd/rmd-reference.hpp` | `ggml/src/ggml-npu-common/residual/rmd/rmd-reference.hpp` |
| `ggml/src/ggml-gemmini/residual/rmd/rmd-types.hpp` | `ggml/src/ggml-npu-common/residual/rmd/rmd-types.hpp` |

Additional splits keep the architecture-owned remainder in its original file:

| Original | Shared extraction | Retained responsibility |
| --- | --- | --- |
| `ggml-gemmini-args.h` | `types/matmul.hpp` | transport, physical scales/dataflow and execution flags |
| `ggml-gemmini-geometry.hpp` | `types/geometry.hpp` | original attention quantization geometry policy |
| `residual/rmd/rmd-executor.hpp` | `residual/metrics.hpp` | physical tile assembler and Gemmini WS execution |
| `matmul/types.hpp` | `profiling/matmul-types.hpp` | execution routes, states, jobs and collector snapshots |
| `matmul/detail.hpp` | `profiling/matmul-detail.hpp` | Gemmini execution test instrumentation |

Shared C++ namespaces are `ggml::npu`, including `quants`, `rmd`, `residual`,
`config`, `log`, `cycle`, `evaluation`, `semantic`, `trace` and `optrace`.
Generic geometry names use `Npu*` and `make_npu_geometry`; the common residual
policy enumerator is `RmdBackend::npu_compact`. The generated software-options
header is `npu-matmul-config.hpp`. Existing C timing/logging symbol prefixes
remain to preserve their C interface; no forwarding implementation was added.
Gemmini dump/scale-dump code and its names remain Gemmini-owned.
