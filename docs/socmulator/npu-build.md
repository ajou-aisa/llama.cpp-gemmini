# NPU build configuration and self-contained Gemmini API

## Structure

```
llama.cpp
├── cmake/npu-config.cmake, cmake/npu-validation.cmake   parameter generator + checks
├── cmake/toolchains/riscv64-linux.cmake                 portable RISC-V Linux cross toolchain
├── ggml/src/ggml-npu-common/config/*.in                 npu_config.h/json, gemmini_params.h, PoTAL header templates
├── ggml/src/ggml-npu-common/tests/npu-config-test.cmake configure-time contract test
└── ggml/src/ggml-gemmini/
    ├── api/                 standalone C11 API (no GGML, no SDK)
    │   ├── include/gemmini.h            umbrella
    │   ├── include/gemmini_isa.h        RTL constants (funct7, CONFIG modes, local address, counters)
    │   ├── include/gemmini_types.h      status, gemmini_hw_config_t
    │   ├── include/gemmini_command.h    command, descriptors, encoders, validation
    │   ├── include/gemmini_transport.h  issue/drain, RoCC, recorder
    │   ├── gemmini.c encode.c validate.c counter.c
    │   └── transport/rocc.c transport/recording.c
    ├── matmul/tiling.{hpp,cpp}   WS tiler on the API (gemmini-tiled-api)
    ├── matmul/reference.hpp      CPU reference matmul (verbatim SDK)
    ├── matmul/baseline.{hpp,cpp}, block-quantized.{hpp,cpp}  GGML numerical adapters
    ├── reference/cpu.{hpp,cpp}   FP32 and INT32 CPU kernels
    ├── pb/pb-rocc-model.{hpp,cpp}  owned command-functional transport
    └── tests/                    API, encoding, transport, tiling, functional tests + golden captures
```

Layering (Phase 2A):

```
GGML MUL_MAT → ops.cpp / matmul/dense.cpp / execution.cpp / residual
   → baseline.cpp / block-quantized.cpp → tiled_matmul(explicit transport)
   → matmul/tiling (tile choice, outer loop, LOOP_WS plan)
   → api encode.c (descriptor → rs1/rs2) → transport (RoCC | PB model | test recorder) → drain
```

## Build parameters

| Option | Values | Notes |
|---|---|---|
| `GGML_NPU` | `NONE`, `GEMMINI`, `POTAL` | empty: derived from legacy `GGML_GEMMINI`/`GGML_POTAL` |
| `NPU_PRECISION` | Gemmini `INT8` (`INT16`: REFERENCE only; `FP32`: legacy DIM8 lane); PoTAL `INT8`, `INT4` | one fixed precision per build |
| `NPU_D` | 16, 32, 64 (FP32 lane: 8) | power of two |
| `NPU_SPAD_BANKS`, `NPU_SPAD_BANK_ROWS` | powers of two | rows per bank; PoTAL: 4 banks only |
| `NPU_ACC_BANKS`, `NPU_ACC_ROWS` | powers of two | ACC_ROWS is the **total**; per-bank = ACC_ROWS/ACC_BANKS; PoTAL: 2 banks only |
| `NPU_DMA_MAX_BYTES` | 64 | the RTL header generators fix MAX_BYTES=64 |
| `NPU_INDEX_ENTRIES` | PoTAL, ≥ D | per-buffer K-row window (isa-contract.md); must be empty/0 for Gemmini |
| `NPU_SCALE_ENTRIES` | PoTAL, 256 | 8-bit scale slot |
| `NPU_TRANSPORT` | `ROCC` (riscv64 only), `REFERENCE` | `RPC`, `SIM_DEVICE`: rejected (not implemented) |
| `NPU_MATMUL_ALGORITHM` | Gemmini `BASELINE`, `PARTIAL_SUM` | Empty derives from `GGML_GEMMINI_PB`; PB REFERENCE keeps WS lowering |
| `NPU_API_ONLY` | ON/OFF | configure only the standalone API (and its tests) |
| `NPU_PROFILING` | ON/OFF | recorded in the manifest |

Every enabled NPU needs all of its parameters (no inference); presets
provide the RTL-confirmed sets. Validation rejects malformed numbers,
non-power-of-two geometry, ACC rows not divisible by banks, local-address
overflow (25 row bits), SPAD/ACC halves too small for one double-buffered
tile, unsupported precision/D/topology, conflicting legacy options and
`GEMMINI_SW_PATH`. See npu-config-test.cmake for the exact cases.

Generated in `<build>/generated/`: `npu_config.h`, `npu_config.json`
(machine-readable, for PyTorchSim/gem5), `npu_build_manifest.json`
(geometry, ISA revision, packing ABI, transport, header sha256, RTL
reference + validation status, software commit, compilers, toolchain,
config hash), and the parameter header (`gemmini_params.h` or
`npu/gemmini_potal*_params.h`, named like the RTL output). For the five RTL configurations the generated
header is byte-identical to the RTL-generated one (configure fails if not);
other geometries are marked `unverified` in the manifest.

## Legacy option mapping

| Legacy | New | Behaviour |
|---|---|---|
| `GGML_GEMMINI`, `GGML_POTAL` | `GGML_NPU` | derived; explicit conflicting value is an error |
| `GGML_POTAL_W4A4` | `NPU_PRECISION=INT4` | derived |
| `GGML_GEMMINI_DIM`, `POTAL_EXPECTED_DIM` | `NPU_D` | derived |
| `GGML_GEMMINI_PB` | `NPU_MATMUL_ALGORITHM` BASELINE/PARTIAL_SUM | derived with conflict checks |
| `GGML_GEMMINI_OPTION` CPU/WS | algorithm + transport | BASELINE/REFERENCE uses CPU; PB uses WS with either transport |
| `GGML_GEMMINI_ACTIVATION_BITS/WEIGHT_BITS` | PoTAL: from `NPU_PRECISION`; Gemmini INT16: 16 | Gemmini INT8 keeps 4/8 as the *software* quantization width in int8 elements (A4W4 runs on the INT8 datapath) |
| `POTAL_HW_PARAMS` | generated header | still accepted: values imported into `NPU_*`, conflicts rejected |
| `POTAL_API_ONLY` | `NPU_API_ONLY` | derived |
| `GEMMINI_SW_PATH` | internal API | error, except the `GGML_GEMMINI_FP32` overlay |

## Commands

```sh
# Standalone Gemmini API (any host), then its tests
cmake -S . -B build/gemmini-api -G Ninja -DGGML_NPU=GEMMINI -DNPU_API_ONLY=ON \
  -DNPU_PRECISION=INT8 -DNPU_D=16 -DNPU_SPAD_BANKS=4 -DNPU_SPAD_BANK_ROWS=4096 \
  -DNPU_ACC_BANKS=2 -DNPU_ACC_ROWS=1024 -DNPU_DMA_MAX_BYTES=64 -DNPU_TRANSPORT=REFERENCE
cmake --build build/gemmini-api && ctest --test-dir build/gemmini-api

# Presets (CPU reference host build, or RoCC cross build)
cmake --preset gemmini-int8-d16-reference && cmake --build build-gemmini-int8-d16-reference
RISCV_TOOLCHAIN_ROOT=$RISCV cmake --preset potal-int8-d32-rocc -DGGML_RVV=OFF
cmake --preset potal-int4-d32-reference -DGGML_GEMMINI_EXSIA_STRIPE_ANCHOR=3 -DGGML_GEMMINI_ENABLE_RMD=ON \
  -DGGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=WS -DGGML_POTAL_PAIRED_AUTO=ON
```

`RISCV_TOOLCHAIN_ROOT` (or `RISCV_TOOLCHAIN_PREFIX`, `RISCV_SYSROOT`) selects
the cross compiler; the toolchain file checks that it targets riscv64 Linux
and rejects bare-metal `*-elf-` toolchains. `gemmini-api-smoke` is built for
riscv64 only and is run by hand on the target.

## Transports

* `gemmini_rocc_transport()`: `.insn r CUSTOM_3` per funct7 (immediate),
  xd only for COUNTER, no memory clobber (as xcustom.h); drain = `fence`.
  Non-RISC-V: every call returns `GEMMINI_ERR_NO_HARDWARE`.
* `gemmini_recorder_t`: ordered issue/drain log with configurable response
  and error injection; computes nothing.
* Execution takes an explicit transport. GGML argument copies retain owned
  test contexts; there is no process-global transport setter.
* `pb::FunctionalTransport` decodes actual WS commands into INT32 partials.
  Each synchronous execution or existing runtime issuer owns its model.
  PB presets and numerical evidence: `phase2/pb-reference.md`.
* SDK-derived implementation provenance and BSD texts remain under
  `matmul/`; removal map and contract review: `phase2/gemmini-legacy-removal.md`.
