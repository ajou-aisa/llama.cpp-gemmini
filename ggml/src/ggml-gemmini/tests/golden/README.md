# Gemmini tiler golden captures

`gemmini-<geometry>.golden` record, for every case in `golden-format.hpp`,
what the legacy SDK tiler (RISC-V-DynDNN-gemmini-include develop
`fcfcf5ce`, gemmini.h sha256 `a6d7207d…`) produced:

* `SET_TILE_WS` - the tiles of `gemmini_set_tile_ws`;
* `WS_COMMANDS <n> HASH <fnv1a>` - command count and hash of the normalized
  RoCC stream of `tiled_matmul_auto(..., WS)`, followed by its first 24
  commands (`CMD f=<funct7> rs1=... rs2=...`, `FENCE` for `gemmini_fence`);
* `CPU_REFERENCE HASH` - FNV-1a of the `matmul_cpu` output (elem_t C).

DRAM operands of MVIN/MVOUT rs1 and LOOP_WS_CONFIG_ADDRS_AB/DC are printed as
`<buffer>+<byte offset>`; every other bit is printed raw.

`gemmini-tiling-test` and `gemmini-functional-test` compare the in-tree
implementation against these files.

## Regenerating (only when the legacy reference itself changes)

```sh
SDK=/path/to/RISC-V-DynDNN-gemmini-include   # develop checkout
W=$(mktemp -d)
python3 extract-legacy-sdk.py "$SDK/gemmini.h" "$W/legacy-extract.inc"
# one gemmini_params.h per geometry: the D16 file is $SDK/gemmini_params.h;
# the others are cmake/npu-config.cmake output for the same NPU_* values.
clang++ -std=c++17 -O2 -I<params dir> -I"$W" -I. legacy-capture.cpp -o "$W/capture"
"$W/capture" > gemmini-d16-4x4096-acc1024.golden
```

The capture replaces only the SDK's host no-op RoCC bridge
(`ROCC_INSTRUCTION_RS1_RS2`, `gemmini_fence`) with a recorder; all tiler code
is the SDK text extracted verbatim.
