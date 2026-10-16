# PoTal minimal RoCC API

Standalone C11 API for the PoTal SCU hardware (Gemmini-derived, Sat32 ACC).
It is configured only with `POTAL_API_ONLY=ON` and does not use `gemmini.h`,
`xcustom.h`, `GEMMINI_SW_PATH`, `RISC-V-DynDNN-gemmini-include` or `IM2P.sim`.
The existing Gemmini backend and `llama-cli` builds are unchanged when
`POTAL_API_ONLY` is OFF.

| File | Role |
|---|---|
| `include/potal.h` | Public API: status codes, encoders, scale control, WS GEMM |
| `include/potal_rocc.h` | Unchecked raw custom-3 issue layer (RISC-V only) |
| `potal.c` | Validation, encoding, hardware calls; cross-checks the original header |
| `tests/potal-api-test.c` | Host/target packing and validation test (`potal-api-test`) |
| `tests/potal-api-smoke.c` | DIM16 hardware test (`potal-api-smoke`) |
| `../../../cmake/potal-hw-params.cmake` | Hardware header validation (single implementation) |
| `../../../scripts/potal-hw-params-test.sh` | Configure-level positive/negative header tests |
| `../../../scripts/potal-build-baremetal.sh` | Bare-metal ELF of the smoke test |

## Hardware parameter contract

The RTL-generated header is the source of truth. Configure fails (no fallback,
no default, no correction) unless:

- `POTAL_HW_PARAMS` is an absolute path to a readable file with the
  `GEMMINI_PARAMS_H` guard. Typed relative values (`:FILEPATH`, `:STRING`)
  are rejected; an untyped relative `-D` value is converted by CMake to an
  absolute path against the invocation directory before validation. The
  absolute path and its realpath are recorded in the manifest;
- `POTAL_EXPECTED_DIM` is 16, 32 or 64 and equals the header `DIM`;
- `XCUSTOM_ACC`, `DIM`, `ADDR_LEN`, `BANK_NUM`, `BANK_ROWS`, `ACC_ROWS`,
  `MAX_BYTES` are each defined exactly once as decimal literals;
- `XCUSTOM_ACC == 3`, `ADDR_LEN == 32`, `BANK_NUM`/`BANK_ROWS`/`ACC_ROWS` are
  powers of two, `MAX_BYTES` is a multiple of `DIM`, and half of SPAD/ACC holds
  one A/B tile pair and one C tile;
- `typedef int8_t elem_t;` and `typedef int32_t acc_t;` (A8W8, INT32 ACC).

Configure writes `<build>/generated/`:

- `original-potal-params.h`: byte-exact copy; included only by `potal.c`, which
  `_Static_assert`s every adapter value and the `elem_t`/`acc_t` types against it;
- `potal_hw_params.h`: `POTAL_HW_*` names only (no `DIM`, `elem_t`, ... leak);
- `potal_build_manifest.txt`: header path/realpath/SHA-256, parameters, git HEAD
  of this repo and of the repo holding the header, `POTAL_HW_CONFIG`, toolchain.

The header SHA-256 identifies the parameter header only. It does not identify
the RTL revision or the Sat32 configuration; record those with
`-DPOTAL_HW_CONFIG=<Chipyard config>` and the hardware git HEAD.

Editing the selected header re-runs validation on the next build
(`configure_file` dependency). Build scripts clear `POTAL_*` cache entries on
every configure (`-U 'POTAL_*'`); each build dir has its own generated copy.
Setting `POTAL_HW_PARAMS`/`POTAL_EXPECTED_DIM` without `POTAL_API_ONLY=ON` is a
configure error.

## API summary

All functions return `potal_status`. `0..7` are the hardware codes of
`PoTalScaleCommandStatus`; negative values are software rejections where
nothing was issued.

| Function | Notes |
|---|---|
| `potal_scale_key_encode(slot, gen, col, &rs1)` | slot < 256, gen <= 255, col < DIM |
| `potal_carrier_is_valid(c)` | 0..32767 or `0x80000000` (zero); 0 means x1 |
| `potal_scale_write(slot, gen, col, carrier)` | validates, then SCALE_WRITE (funct 23); returns hw status |
| `potal_scale_release(slot, gen, col)` | SCALE_RELEASE (24); hw holds it until older work drains |
| `potal_fence()` | POTAL_FENCE (25); `POTAL_OK` = older GEMMs done, C stores drained |
| `potal_gemm_ws_check(&g)` | argument -> capacity -> validated-scope checks |
| `potal_gemm_ws_scale_slot(&g, k_block, n_tile, &slot)` | `base + k_block * J + n_tile` |
| `potal_gemm_ws_encode(&g, &program)` | CONFIG_LD A/B, CONFIG_EX, CONFIG_ST, LOOP_WS 9..13, LOOP_WS 8 |
| `potal_gemm_ws_issue(&g)` | issues the program; `POTAL_OK` = issued, not finished |

Rules:

- Write every scale `slot = potal_gemm_ws_scale_slot(...)` for all `DIM`
  columns with the GEMM's generation before issuing.
- After `potal_gemm_ws_issue`, do not read C or modify/free A, B, C until
  `potal_fence()` returns `POTAL_OK`. A second issue before that returns
  `POTAL_ERR_BUSY`. Release scales after the fence.
- Buffers: row-major, strides in elements, base addresses aligned to
  `POTAL_HW_MAX_BYTES`, NULL rejected (DRAM address 0 means "skip" in RTL),
  C must not overlap A/B, extents must stay within the 32-bit LoopMatmul
  DRAM offset.
- Validated execution scope: DIM16, `m == n == 16`, `k` in {16, 32},
  `passes_per_block` dividing the K tile count, `flags == 0`. Other RTL-legal
  shapes (more tiles, padding, DIM32/64) return `POTAL_ERR_UNSUPPORTED`.
  Transpose, bias, resadd and accumulate flags return `POTAL_ERR_UNSUPPORTED`.
- `issue` executes `fence rw,rw` before the commands and `potal_fence` after
  the response. These order CPU accesses; completion comes only from the
  POTAL_FENCE response. Linux runs rely on Gemmini's TLB and coherent DMA as
  the existing backend does; page pinning and explicit cache maintenance are
  not provided.
- Non-RISC-V builds validate and then return `POTAL_ERR_NO_HARDWARE`.
- Not thread-safe; single core.

## Commands

```bash
LLAMA=~/firesim/workspace/potal/software/workspace/llama.cpp-potal
PARAMS=~/firesim/workspace/potal/software/gemmini-rocc-tests/include

# Host: packing/validation test (expects "POTAL_API_TEST PASS")
cd "$LLAMA"
cmake -S . -B build-potal-host -DGGML_GEMMINI=OFF -DPOTAL_API_ONLY=ON \
  -DPOTAL_EXPECTED_DIM=16 -DPOTAL_HW_PARAMS:FILEPATH="$PARAMS/gemmini_potal_params.h"
cmake --build build-potal-host
ctest --test-dir build-potal-host --output-on-failure

# Configure-level header tests; negative cases pass only when configure fails
scripts/potal-hw-params-test.sh   # expects "POTAL_HW_PARAMS_TEST PASS"

# RISC-V Linux cross build (bin/potal-api-test, bin/potal-api-smoke)
BUILD_DIR=build-potal-d16 ./build-riscv.sh -DGGML_GEMMINI=OFF -DPOTAL_API_ONLY=ON \
  -DPOTAL_EXPECTED_DIM=16 -DPOTAL_HW_PARAMS:FILEPATH="$PARAMS/gemmini_potal_params.h" \
  -DPOTAL_HW_CONFIG=PoTalSat32GemminiRocketConfig

# Bare-metal ELF (needs riscv64-unknown-elf-gcc: source chipyard env.sh)
scripts/potal-build-baremetal.sh build-potal-host
```
