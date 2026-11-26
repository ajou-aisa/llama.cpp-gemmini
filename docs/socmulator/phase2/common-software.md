# Common NPU software ownership

Phase 2B moves 99 original files, including 31 C++ translation units, and splits
five mixed headers. The complete file-by-file map is phase2b-rename-map.md.
The already verified phase2a-complete checkpoint is referenced by
build-npu/checkpoints/README.txt; no replacement checkpoint or Git commit was made.

The historical 18 PoTAL-shared Gemmini sources were confirmed in both live
PoTAL source lists before extraction. They are now owned by ggml-npu-common:

| Original responsibility | Current location | Consumers and numerical coverage |
| --- | --- | --- |
| dispatch, quantize, tensor, token, block, stripe, exsia/local/folding/profile | quantization/activation | Gemmini and both PoTAL precisions; ExSIA, dispatch, PB and PoTAL numerical tests |
| weight_reader, dequant; generic math, rounding, FP16 and GGUF views | quantization/common | dense/residual/attention preparation; weight-reader tests and bitwise harness |
| direct executor, builders, bitmap builder, compose and metadata | residual/direct and residual/rmd | backend residual paths; packet/capture/reducer tests, PoTAL residual and golden tests |
| software options | matmul/options | FULL/STRIPE_PIPELINE and residual-policy consumers; option tests |
| telemetry | profiling/telemetry | both backends and profiling tests; schema and interval contracts |

Three additional numerical translation units are shared: activation preparation
cache, FP16 conversion helpers, and the CPU residual reference. Generic activation
buffer ownership is in buffer/activation.hpp. Checked shape/stride arithmetic is
in types/geometry.hpp. No allocator was replaced and no buffer copy was added.
The cache still evaluates the existing Gemmini sensitivity veto at the same
short-circuit point, through an explicit invocation callback.

ExSIA's owner supplies role tracing through HartRoles callbacks. The common
quantizer no longer includes PoTAL timeline.hpp. Existing pinning, worker count,
thread creation/join, publication order, failure propagation and buffer lifetimes
are unchanged. The PoTAL callback implementation retains its existing event IDs,
clock source, thread registration, span boundaries and run/layer identity.

Gemmini retains ISA/encoders/transports, WS tiling, residual physical lowering,
its physical-tile assembler, PB model/issuer, dump profiler and hardware counters.
PoTAL retains checked API/ISA, scale/index programming, paired planner, resident
weights, INT4 DMA ABI, accumulator scheduling and checked hardware reconstruction.
The original Gemmini CPU INT32 reference has Gemmini scaling/activation semantics
and stays architecture-owned. The original attention geometry policy also stays
Gemmini-owned. They were not merged with PoTAL arithmetic.

The optional CUDA residual comparator calls its existing CUDA C APIs directly;
it no longer imports the Gemmini wrapper. Arithmetic and dispatch conditions are
unchanged. Full CUDA and FP32 overlay execution remain outside this phase; source
and include wiring were checked, not represented as executed GPU/FPGA evidence.

Bitwise method: restore phase2a-complete into a scratch tree from HEAD archive,
its saved tracked patch and untracked archive; compile the same independent
serialization harness against that tree and the final common sources. Hash
logical values and metadata fields individually, never object padding/pointers.
Both software widths 4 and 8 match for EXSIA/TENSOR/TOKEN/STRIPE/BLOCK/MXINT,
both packet/direct residual routes, bitmap masks, lane descriptors/limbs and
HP1 weight codes/carrier/column scale. See build-npu/phase2b-evidence/bitwise.cpp,
run-bitwise.sh, bitwise-snapshot-a{4,8}.txt, bitwise-current-a{4,8}.txt and
bitwise-final.log. This is deterministic fixture evidence, not an exhaustive
proof over all possible tensors. Existing independent numerical and command
regressions supply additional coverage; no existing golden was regenerated.

The INT4 attention fixture remains explicitly BLOCKED after the bounded attempt;
its precise 181-site gap ledger and retained consumers are in test-readiness.md.
