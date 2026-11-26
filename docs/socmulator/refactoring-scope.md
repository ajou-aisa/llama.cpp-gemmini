# Phase 0 → Phase 1: refactoring scope

Decisions taken from the Phase 0 audit (architecture-current.md,
dependency-inventory.md, isa-contract.md, hardware-parameters.md,
build-matrix.md, experiment-inventory.md) and what Phase 1 changed.

## Baseline that Phase 1 must preserve

* Software: llama.cpp-gemmini `socmulator-dev` @ `64a072a`, clean tree.
* External SDK actually compiled: RISC-V-DynDNN-gemmini-include `develop`
  `fcfcf5ce` (gemmini.h sha256 `a6d7207d…`). The 205 KB gemmini-aisa copy
  (`185293e8`) is its ancestor and does not compile against this tree.
* RTL: gemmini-aisa `185293e8` (Gemmini), chipyard-aisa `potal` `fab3d262`
  (PoTAL INT8 `potal/`, INT4 `potalw4/`). See baseline/rtl.json.
* Command streams: golden captures of the SDK tiler for 13 matmul shapes ×
  4 geometries (ggml/src/ggml-gemmini/tests/golden). Numeric CPU reference
  hashes for the D16 geometry in the same files.
* Test pass/fail sets per configuration: baseline/test-results.md.

## In scope (done in Phase 1)

| Item | Result |
|---|---|
| External SDK removal for the core Gemmini path | `GEMMINI_SW_PATH`, external gemmini.h/gemmini_params.h/gemmini_counter.h/xcustom.h no longer read; configure rejects `GEMMINI_SW_PATH` outside the FP32/IM2P experiment lanes |
| Standalone Gemmini API | `ggml/src/ggml-gemmini/api` (C11): ISA constants, descriptors, encoders, geometry validation, RoCC and recording transports, counters; builds alone with `GGML_NPU=GEMMINI NPU_API_ONLY=ON` |
| Tiler | `matmul/tiling.{hpp,cpp}`: port of gemmini_set_tile_ws / tiled_matmul_auto tile choice / tiled_matmul checks / tiled_matmul_outer / sp_tiled_matmul_ws on `gemmini_hw_config_t`; command streams equal the SDK's for all captured cases |
| CPU reference | `matmul/reference.hpp`, verbatim SDK `matmul_cpu`; output hashes equal the SDK capture |
| Compatibility layer | `legacy/gemmini_legacy.hpp`: the SDK's GGML adapters (verbatim) and its raw entry points as wrappers; single place for legacy macro names |
| Common parameter generator | `cmake/npu-config.cmake`, `cmake/npu-validation.cmake`, templates in `ggml/src/ggml-npu-common/config`; generates npu_config.h/json, build manifest, gemmini_params.h (GEMMINI) or the PoTAL parameter header (POTAL); reproduces all five RTL reference headers byte for byte |
| PoTAL wiring | `GGML_NPU=POTAL` + `NPU_*` generates the header fed to the unchanged cmake/potal-hw-params.cmake validation; `POTAL_HW_PARAMS` still accepted and cross-checked |
| Build options | `GGML_NPU`, `NPU_*` public; legacy options derived with conflict detection; presets for the five RTL configurations |
| Toolchain | `cmake/toolchains/riscv64-linux.cmake` (RISCV_TOOLCHAIN_ROOT/PREFIX/SYSROOT, Linux-ELF check); `/home/alveo/...` removed from cmake/riscv*.cmake |
| Host behaviour | RoCC issue on a non-RISC-V build returns `GEMMINI_ERR_NO_HARDWARE` (legacy entry points abort) instead of a silent no-op; `NPU_TRANSPORT=ROCC` is rejected at configure unless the target is riscv64 |

## Explicitly unchanged

Activation/weight quantization, ExSIA, stripe folding, RMD/residual,
paired GEMM, indexed preload, resident reuse, attention, INT64
reconstruction, dequantization, GGUF formats, PoTAL ISA/potal.c/potal.h,
paired planner, issuer threads and hart affinity, buffer lifetime, and the
Gemmini command order/fence placement (golden-checked). PoTAL INT8 and INT4
remain separate backends (`ggml-potal`, `ggml-potal-w4a4`).

## Deferred (Phase 2+)

* Move the GGML adapters out of `legacy/gemmini_legacy.hpp` into `matmul/`
  and delete `legacy/`.
* Port the FP32 DIM8 lane (`fp32/`, own SDK overlay) and the IM2P lanes off
  `GEMMINI_SW_PATH`; neither could be built here (no FP32 overlay, no
  IM2P.sim checkout).
* Wire `pb/pb-rocc-model.cpp` for the PB lane: it is now the PB host
  transport (`GEMMINI_ROCC_MODEL`); the PB build itself was not run.
* Merge `ggml-potal-w4a4` into `ggml-potal`.
* RPC / SIM_DEVICE transports, common NPU protocol, PyTorchSim and gem5.
* Legacy build-option removal (GGML_GEMMINI_OPTION, POTAL_HW_PARAMS, …) once
  experiment scripts use `NPU_*` presets.
* Pre-existing failures listed in baseline/test-results.md.
