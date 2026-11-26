# Common numerical types

The common invocation is npu_matmul_args in ggml-npu-common/types/matmul.hpp.
ggml_gemmini_args_t derives from this value type and owns the Gemmini-only
fields. There is no virtual hierarchy or global mutable transport selector.

| Field group | Common owner and lifetime | Architecture remainder |
| --- | --- | --- |
| Shape | I/J/K; sA/sB/sC/sD and output strides; row offsets | tile selection stays in each backend |
| Buffers | shared quantized A backing; borrowed B, A_fp32, B_fp32, C, D and f_out | DMA/local addresses, resident B and completion |
| Weight metadata | unpacked vectors own storage; GGUF block/scale views borrow it; validation state shared per immutable invocation | packed-nibble DMA stays PoTAL-specific |
| Activation metadata | value-owned variant, scales/exponents, shared immutable residual payloads | architecture supplies stripe sink explicitly |
| Layout | transpose_B, block sizes, stripe metadata and numeric tile factors | transpose_A, repeating_bias, full_C and low_D stay Gemmini-owned |
| Policy | folding anchor, residual route, quantizer and software execution options | OS/WS/CPU, transport and hardware algorithm selection stay separate |
| Observation | layer/model identity and immutable optrace/evaluation context | optional role observation and affinity callbacks supplied by owner |
| Hardware | no hardware API in common type | transport shared_ptr; scale_B/scale_D; act; scale/bert_scale; weightA; tiled_matmul_type |

Original geometry and storage static assertions remain in the Gemmini argument
header beside its physical parameter include. Common npu_element_t is int16
only for software width 16, otherwise int8. INT4 logical values therefore remain
distinct from packed PoTAL DMA bytes. Generated npu_config.h supplies geometry;
quantization block size remains independent. Generator validation and supported
configuration constraints are unchanged. PB D32 retains 4x4096 SPAD and 1024
TOTAL ACC rows; PoTAL D32 retains 4x2048 and 512 TOTAL ACC rows.

The exact weight_format_t : uint8_t enum body is unchanged: 0,1,2, retired 3,
4,5,6,7,8, retired 9,10,11,12,13. No GGUF identifier changed. Evidence:
build-npu/phase2b-evidence/layering-audit.json.

Gemmini invocation copies still copy the owning transport shared_ptr. Numerical
functions borrow the common base; execution retains the complete invocation
through the final drain. PoTAL uses the common argument without Gemmini transport
or dataflow fields. Sessions, paired state, issuer scheduling and buffer-release
rules remain in PoTAL execution.

Stage-2 evidence under build-npu/phase2b-evidence: types-gemmini-final.build.log,
types-gemmini.ctest.log (8/8), types-potal-final.build.log and
types-potal.ctest.log (3/3). common-include-deps.log is actual compiler dependency
evidence with no hardware API or parameter shim. Final matrix and affected
rechecks are recorded separately in phase2b-test-results.md.
