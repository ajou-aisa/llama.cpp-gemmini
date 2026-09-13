# IM2P main-parity: current host worktree

> Historical continuation snapshot. Current working-branch stripe/provider
> implementation and ordinary build instructions are documented in
> [HOST_STRIPE_CONTRACT.md](../../IM2P.sim/docs/HOST_STRIPE_CONTRACT.md).
> Earlier unimplemented/blocked entries below retain their original run scope.

This repository remains on `fpga`, based on
`7a74ac29c9c8fa8fbc5d47cf073b8b9266e3e7a9`, with the user's pre-existing
`build-x86.sh` defaults preserved. No staging/commit/push was performed.

Full main-equivalent FPGA inference is **not yet implemented or verified**.
The existing source02/SCU integration remains historical evidence, not proof that
this checkout now supports board HP1, nonzero accelerator residual, or bounded
large logical GEMM. No original model was converted to H1 or reduced in size.

The common workspace's current evidence is
`.parity/20260911-main-parity-02/evidence/`. The core report is
`../IM2P.sim/docs/FPGA_MAIN_PARITY.md` and the machine-readable state is
`../IM2P.sim/fpga/main-parity.json`.

## Read-only model inventory

The new ordinary CMake target `llama-im2p-model-inventory` uses the native GGUF
reader (`no_alloc=true`) and links `ggml-base`. It does not initialize a backend,
load a device, quantize weights, execute matmul, or run prefill/decode. Its JSON
records stored tensor axes, not runtime invocation shapes or dispatch counts.
It validates tensor extents against the file size and publishes no partial JSON
on a parse/extent failure. NumPy is not required.

The following configuration was actually built with tests still enabled:

```bash
# Execute from this repository. Choose a new writable build directory.
BUILD=/absolute/new-output/metadata-cpu-build
cmake -S . -B "$BUILD" \
  -DGGML_GEMMINI=ON \
  -DGGML_GEMMINI_OPTION=CPU \
  -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE \
  -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$BUILD" --target llama-im2p-model-inventory -j2
python3 tools/im2p-model-inventory/test_inventory.py \
  --binary "$BUILD/bin/llama-im2p-model-inventory"
"$BUILD/bin/llama-im2p-model-inventory" models/gpt2/gpt2.Q8_HP1.gguf
"$BUILD/bin/llama-im2p-model-inventory" models/llama3.2-1B/llama3.2-1B.Q8_HP1.gguf
```

Here CPU+HARDWARE is the existing configuration used to generate the host project;
it is not FPGA execution. The inventory target itself has no Gemmini module or
simulator archive dependency. The native parser tests passed all seven cases.

Actual file metadata:

| Model | Stored tensors | HP1 / F32 | Selected `ne[0],ne[1]` |
|---|---:|---|---|
| GPT-2 | 148 | 49 / 99 | QKV 768,2304; FFN up 768,3072 |
| Llama3.2-1B | 147 | 113 / 34 | attention Q 2048,2048; FFN up/gate 2048,8192 |

These are not full-size invocation or model numerical PASS results. Actual M,
quantized activation stripes, residual packets, output and logits have not yet
been captured. The future invocation differential must retain the original format
and logical axes and compare independent main/current simulator/board-provider RTL.

## Baseline compatibility

A separate detached historical host at
`7a6ed1e6d98f00e8bb5db4ca235b277260c3d684` successfully compiles the unchanged main
`6fef570` frontend contract tests. This is an additional reference, not a change
to this worktree's pin. Later host commit `4cf791e` removed raw cycle fields while
nanosecond fields already existed. The history is not a simple unit rename, and
no synthetic cycle-to-time conversion was applied. The historical frontend tests
use mock simulator APIs and are not RTL numerical evidence.

## Blockers and remaining host integration

The CatDesk command filesystem does not expose the existing BSC/Vivado roots,
OSS CAD runtime share/lib directories, or `/mnt/fpga-build`. Merely exporting PATH
cannot fix their absence from that filesystem. No tool reinstall was attempted.
The core report lists exact paths and attempted fresh build failures.

An independent existing configure defect is retained as
`metadata-configure.log`: `GGML_GEMMINI=OFF, LLAMA_BUILD_TESTS=ON` reaches
`tests/CMakeLists.txt:865` and calls
`ggml_gemmini_apply_profile_compile_definitions`, which is only defined when the
backend is enabled. This was not fixed by disabling tests. The CPU+HARDWARE recipe
above is a separately tested configuration, not a claim that the OFF case works.

Still required: port the complete ordinary FPGA integration into the actual host
source, reconcile main/SCU numerical semantics, connect the bounded provider and
HP1/nonzero residual, validate real runtime assignment/completion and no dot
fallback, and compare real CLI/PPL/model invocation results. This continuation
only added metadata inspection and its tests to this host checkout.

Physical UART/JTAG/CAP/programming/Flash/reset remain unauthorized and unperformed.
A new physical approval package requires fresh numerical/model and 25 MHz route
qualification first.
