NVFP4 dynamic-max numerical reference on compatible CUDA GPUs

This is NVFP4-format W4/A4 fake quantization followed by cuBLAS GEMM.
The fp32 accuracy profile keeps reconstructed operands in FP32 and uses pedantic
FP32 GEMM. The explicit fp16 compatibility profile rounds operands to FP16 before
FP32 accumulation. Neither uses native Blackwell FP4 instructions or stores a
packed NVFP4 GGUF; do not report either as native NVFP4 throughput.

Reference: NVIDIA Model-Optimizer nvfp4_tensor.py, pinned commit and SHA256 in
reference-identity.json. The archived CPU QA executed official quantize/dequantize
methods unchanged; only package storage and block-reduction integration helpers
were supplied locally. The downloaded source retains its Apache-2.0 license header.
Authoritative sources:
https://github.com/NVIDIA/Model-Optimizer/blob/ff336e72129f5b2bae47fb2e3ff35112f1bdbc8d/modelopt/torch/quantization/qtensor/nvfp4_tensor.py
https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html
https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/

Formula (all arithmetic steps FP32):
  g = max(abs(X)) / (6 * 448)
  raw_b = max(abs(X_b)) / (6 * g); replace exactly-zero raw_b with 1
  s_b = RNE_E4M3FN(clamp(raw_b, 2^-9, 448))
  q_i = RNE_E2M1(X_i / (s_b * g)), saturated to magnitude 6
  Xhat_i = q_i * (s_b * g)
Blocks contain 16 consecutive values along K. E2M1 magnitudes are
0, 0.5, 1, 1.5, 2, 3, 4, 6. Ties select the even encoded ordinal.
The tensor scale g is FP32. Both E4M3 normal and subnormal scales are supported.
Only the fp16 compatibility profile casts Xhat to FP16 after reconstruction.

Scope: every dense linear matrix including lm_head uses quantized weights and
activations. Weight scales use the original FP16 tensor's maximum once, with cached
dequantized GPU weights. Activation scales use the current GEMM input's maximum.
Embedding GET_ROWS and non-matmul operators remain in the original graph precision;
the embedding table stays FP16. QK/VP attention delegates to the existing RTN CUDA
FP16-input / FP32-accumulating path; softmax is F32 and KV cache is F16.

This recipe uses no calibration corpus, learned scales, MSE scale search, random
Hadamard transform, stochastic rounding, or 2D scaling. It is the named dynamic-max
ModelOpt formula, not a reproduction of a calibrated ModelOpt checkpoint or training
recipe. Do not compare its speed to native FP4 hardware or infer full-model quality
from a one-chunk smoke run.

Explicit edge policies: an entirely zero tensor reconstructs to zero using g=1.
The pinned upstream dynamic helper instead produces non-finite reconstruction from
g=0; this extension intentionally differs. NaN/Inf inputs and global-scale underflow
are rejected instead of silently producing a result. Zero blocks inside a nonzero
tensor match the upstream raw_b=1 safety rule.

Build in Linux/WSL with a compatible NVIDIA driver and CUDA toolkit.
Run from the repository root:
  repo=$(pwd)
  out="$repo/build-potal-cuda/nvfp4-fp32"
  bash "$repo/scripts/experiment/build-potal-cuda.sh" --rtn --jobs 2 --build-only
  bash "$repo/scripts/experiment/build-nvfp4-reference.sh" "$out" fp32

The default GPU architecture is native. Set CUDAARCHS for the CMake RTN build
and NVFP4_CUDA_ARCH for nvcc when explicitly targeting another architecture.
CUDACXX and CUDA_HOME select a nondefault toolkit; GEMMINI_SW_PATH selects the
external Gemmini software headers required by the regular backend build.
The output directory contains libnvfp4-reference.so, libggml-nvfp4-preload.so
and build-identity.json with mode, architecture and source/image hashes.
Build success alone does not mean numerical verification passed.

The build script retains fp16 as its compatibility default. Select either mode
explicitly and keep separate output directories. For fp16:
  bash "$repo/scripts/experiment/build-nvfp4-reference.sh" \
    "$repo/build-potal-cuda/nvfp4-fp16" fp16

Use the complete PPL recipe in docs/backend/GEMMINI-CUDA-PPL.md with the regular
RTN executable and original *.fp16.gguf model. Launch that complete command
from "$out", with absolute executable/model/corpus paths, preferably in a
subshell. Attach these environment values to it, not to unrelated commands:
  GGML_CUDA_RTN_A_BITS=4
  GGML_CUDA_NVFP4=1
  LD_PRELOAD=./libggml-nvfp4-preload.so
Quoting an absolute LD_PRELOAD path does not protect spaces from loader splitting.
The fp32 profile has no intermediate FP16 operand rounding and uses no TF32 math.
QK/VP attention retains the common FP16-input/FP32-accumulating path in both modes.

CUDA_RTN_PROOF verifies graph-wide coverage; its block_size=32 is an observer
field, not NVFP4 geometry. NVFP4_PROOF independently declares block_size=16,
E4M3FN/FP32 scales, RNE, reconstruction/GEMM mode, dense calls, weight counts
and the non-native implementation. Require both proofs and actual loader binding
to the selected preload. The expected complete workload has dense calls
(7*layers+1)*chunks and attention calls 2*layers*chunks.

Archived official-source QA and loader reports were kept outside Git. Their
machine-local directories are historical evidence, not portable prerequisites.
The dedicated archived harness was removed after verification; changed source
must be verified again. Full-corpus runs preserve their declared population;
do not infer model quality or throughput from a one-chunk smoke run.
