# Independent Gemmini FP32 baseline

Owner remains `ggml/src/ggml-gemmini/fp32/`. `backend.cmake` selects
`backend.cpp`, `exec.cpp` and the shared buffer implementation instead of the
INT8 backend list, then returns before the internal INT8 API is added.
Native selection adds `rocc-model.cpp`; RISC-V selects overlay RoCC issue.
Target name remains `ggml-gemmini` in this separate build directory.
`NPU_PRECISION=FP32` is a build-time baseline selection, never a runtime
precision mode of Gemmini INT8.

New presets are `gemmini-fp32-d8-reference` and `gemmini-fp32-d8-rocc`.
REFERENCE selects the FP32 command model and WS lowering; it must not derive
the generic INT8 CPU policy. Both use DIM8 and FLOAT, with PB, OpenMP,
residual compensation and A4NKS paths off. CUDA evaluation remains disallowed.

No matching overlay is supplied or verified. Required input is an explicit
`GEMMINI_FP32_SDK` environment path (or `GEMMINI_SW_PATH` cache path) containing
`gemmini.h` and `gemmini_params.h` at its root. Supply every transitive header
that this actual overlay includes, including its own `xcustom.h` and any
counter header if referenced. There are no guessed replacement headers or
hardware sources in this change.

The compile-time contract consumed by the existing implementation is:

- `DIM == 8`, `elem_t` and `acc_t` are four-byte floating types;
  `BANK_NUM`, `BANK_ROWS`, `ACC_ROWS` define model storage.
- `gemmini_flush(0)` and the overlay's
  `tiled_matmul_auto(m,n,k,a,b,nullptr,c,k,n,0,n,1,1,1,
  NO_ACTIVATION,1,0,false,false,false,true,false,0,WS)` interface.
- Native `sdk.hpp` intercepts `ROCC_INSTRUCTION_0_R_R` and the SDK fence.
  `ROCC_SOFTWARE_SRC_XCUSTOM_H_` skips RISC-V assembly. `asm`/`volatile`
  macro interception is restricted to inclusion of this overlay, then undone.
  The overlay must not add unrelated inline assembly under this interception.
  Response-bearing `ROCC_INSTRUCTION` aborts in the existing native model.
- The existing FP32 model consumes CONFIG 0, LOAD2 1, MVIN 2, MVOUT 3,
  COMPUTE 4/5, PRELOAD 6 and FLUSH 7. It validates RNE, unity FP32 scale
  bits, FP32 row strides, accumulator ownership and complete configuration.
  It does not decode the INT8 PB LOOP_WS transport.
- Existing FMA order choices, FP32 load/store FMA semantics, no-fast-math and
  no-contraction flags remain unchanged.

The source proves DIM8, not deployed memory geometry or SDK provenance.
Presets therefore require `FP32_SPAD_BANKS`, `FP32_SPAD_BANK_ROWS`,
`FP32_ACC_BANKS`, `FP32_ACC_ROWS` from the actual overlay; DMA is explicitly
64 and can be overridden with its verified configuration. No INT8 memory
geometry is silently assigned to FP32. The manifest marks the overlay
unverified. A future supplied overlay needs header hashes/revision, geometry
cross-check, exact encoding review, and the existing native and target tests.

`tools/offline.sh`, `check-options.sh`, `rtl-closure.sh` and
`compare-logits.py` no longer embed lab-host installation paths.
Use `FP32_OUTPUT_DIR`; model/replay requires `FP32_MODEL` and
`FP32_PROMPT_IDS`. RTL vector checks require `FP32_RTL_VECTORS` and
`FP32_RTL_VECTORS_LEGACY`. Historical PB comparison explicitly requires
`FP32_PB_BASELINE_SOURCE` and `FP32_PB_BASELINE_SDK`. Current PB regression
uses the self-contained PB preset. Activate any external build environment
before invoking these FP32-only tools.

Verification evidence under `build-npu/phase2c-evidence/optional/`:
`presets.log`, `fp32-reference.configure.log`, `fp32-rocc-linux.configure.log`.
Both selected lanes diagnose the missing overlay explicitly. The initial
macOS RoCC attempt instead reports the missing cross compiler; the existing
Linux image supplies that compiler for the diagnostic check. Shell syntax
is recorded in `optional-summary.log`. Normal production build evidence
and absence of overlay include dependencies are in `phase2c-dependencies.md`.

Native FP32 build, RoCC cross-build, block/attention execution,
`golden_replay_fp32`, and RTL vector execution are BLOCKED by the missing
overlay and vector/model artifacts. Their source targets and numerical
oracles remain intact. No FP32 hardware correctness or performance is claimed.
