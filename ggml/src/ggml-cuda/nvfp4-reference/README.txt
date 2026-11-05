NVFP4 dynamic-max numerical reference on RTX 3070 (sm_86)

This is NVFP4-format W4/A4 fake quantization, followed by FP16-input / FP32-accumulating
cuBLAS GEMM. It does not use native Blackwell FP4 instructions, does not store a
packed NVFP4 GGUF, and must not be reported as native NVFP4 throughput.

Reference: NVIDIA Model-Optimizer nvfp4_tensor.py, pinned commit and SHA256 in
reference-identity.json. Official quantize/dequantize methods are executed unchanged
by the CPU QA harness; only package storage and block-reduction integration helpers
are supplied locally. The downloaded source retains its Apache-2.0 license header.
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
The fake-GEMM implementation casts Xhat to FP16 after the FP32 reconstruction.

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

Reproduce in Ubuntu-24.04 WSL (repository root = $repo):
  bash "$repo/scripts/experiment/build-nvfp4-reference.sh"
  /home/mojoi/.cache/nvfp4-reference-venv/bin/python \
    "$repo/scripts/experiment/test-nvfp4-reference.py" --fetch
CPU-only loader/interposition check:
  /home/mojoi/.cache/nvfp4-reference-venv/bin/python \
    "$repo/scripts/experiment/test-nvfp4-reference.py" \
    --check-bindings /home/mojoi/build/potal-optimized-rtn/bin/llama-perplexity
The isolated reference environment requires CPU-only torch; it does not require
NumPy or NVIDIA Python packages. --fetch retrieves the exact pinned URL and verifies
its digest, not the current upstream branch. Offline reruns omit --fetch.

Explicitly scheduled GPU vectors (load the RTN host ABI library first):
  /home/mojoi/.cache/nvfp4-reference-venv/bin/python \
    "$repo/scripts/experiment/test-nvfp4-reference.py" \
    --gpu-library /home/mojoi/build/potal-nvfp4/libggml-nvfp4-preload.so \
    --base-library /home/mojoi/build/potal-optimized-rtn/bin/libggml-gemmini.so

Build outputs and build-identity.json:
  /home/mojoi/build/potal-nvfp4/
QA output / pinned third-party snapshot (outside the Git repository):
  /mnt/c/Users/mojoi/Projects/PoTal/experiment-tools/nvfp4-qa/

Queue interface: use the regular RTN build and original *.fp16.gguf model, with
  GGML_CUDA_RTN_A_BITS=4
  GGML_CUDA_NVFP4=1
  LD_PRELOAD=/home/mojoi/build/potal-nvfp4/libggml-nvfp4-preload.so
The regular CUDA_RTN_PROOF verifies graph-wide GPU coverage; its block_size=32 is a
hardcoded observer field, not NVFP4 geometry. NVFP4_PROOF independently declares
block_size=16, E4M3FN/FP32 scales, rounding, dense calls, weight tensor counts and
non-native fake-GEMM implementation. Both proofs must pass, with dense calls
(7*layers+1)*chunks and attention calls 2*layers*chunks. Loader relocation/binding
evidence must also confirm the preload actually owns compute/counters/clear.
No existing backend code is modified; new files plus opt-in LD_PRELOAD suffice.
