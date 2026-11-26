# Gemmini adapter decomposition

Phase 2A inventory, 2026-11-25. The current tree supersedes the
Phase 0 SDK paths. Existing staged deletions and all other pending work are
preserved. Initial read-only Git reports and the five post-removal Linux
CTest logs are saved under `build-npu/phase2a-evidence/pre/`.
The source checkpoint is identified by `build-npu/checkpoints/README.txt`.

## Old function to new implementation map

| Existing implementation | Category | Destination / callers |
| --- | --- | --- |
| `gemmini_block_quantized_matmul`, `gemmini_block_quantized_matmul_auto` | Block-quantized Matmul | `matmul/block-quantized.cpp`; called by dense dispatch and baseline native-Q8 route |
| `tiled_matmul_auto_baseline`, `tiled_matmul_auto_baseline_dense`, `baseline_route_for` and route enums | Baseline Matmul | `matmul/baseline.{cpp,hpp}`; dense dispatch and semantic observer probes in `ops.cpp` |
| `baseline_weight_scale_provider`, `baseline_activation_scale_provider` | Activation-Weight Scale Processing | Private baseline implementation; preserve channel/block contracts and metadata validation |
| `apply_activation_exponent` | Activation-Weight Scale Processing | Reuse existing `quants/common/math.hpp`; same sentinel and base-two scaling |
| `update_q80_r_output_impl`, `dequantize` calls | Dequantization | Existing `quants/common/dequant.hpp`; no second implementation |
| `matmul_cpu_fp`, `matmul_cpu_int32` | CPU Reference | `reference/cpu.{cpp,hpp}`; dense FP fallback and block dot |
| `matmul_cpu`, `scale_and_sat`, `int_sqrt` | CPU Reference | Existing `matmul/reference.hpp`; numeric golden reference retained |
| `gemmini_set_tile_ws` | Tiling | GGML adapter in `matmul/execution.cpp`; pure tile selection remains `matmul/tiling.cpp` |
| `tiled_matmul_total_spad_rows`, `tiled_matmul_total_acc_rows` | Tiling | Call existing `tiling::total_spad_rows` / `total_acc_rows` directly |
| `tiled_matmul`, `tiled_matmul_auto`, `make_matmul` | Hardware Command Issue / Tiling | Execution implementation using descriptors, checked WS tiler, explicit transport |
| `WsObserverState`, `ws_before_loops`, `ws_before_inner`, `ws_after_inner`, `ws_after_fence` | Profiling | Execution implementation; counters use the same transport as their command sequence |
| `scale_t_to_scale_t_bits`, `acc_scale_t_to_acc_scale_t_bits` | Hardware Command Issue | Private execution conversion; encoders remain solely in `api/encode.c` |
| `gemmini_flush`, `gemmini_fence`, `counter_*` | Hardware Command Issue | Direct Phase 1 API calls; PB flush and tiler drain keep existing placement |
| `transport`, `set_transport` | Transport Ownership / Test-only Hooks | Remove process-global accessor and setter; explicit sequence transport and owned test injection |
| `resolve_env_threads`, `resolve_block_matmul_threads` | Block-quantized Matmul | Private block implementation; OpenMP policy unchanged |
| `k_*`, `CONFIG_*`, activation and counter aliases | Hardware Command Issue | Callers use `GEMMINI_*` API constants directly; no alias header remains |
| `GEMMINI_ASSERTIONS`, WS debug/reuse/profiling switches | Tiling / Profiling | Existing mandatory checks and WS switches remain in execution; unused SDK-only switches are dropped |
| `tiled_matmul_type_t` | Execution policy | Existing OS/WS/CPU values remain in `ggml-gemmini-args.h` |
| `tiled_matmul_auto_fp` | Profiling / Test-only Hooks | `matmul/baseline.cpp`; existing non-computing semantic/logging hook preserved. Real FP compute remains `matmul_cpu_fp` in dense dispatch and the separate FP32 backend |
| Uncalled inverse scale converters, `ceil_divide_int`, `resolve_exsia_threads`, `tiled_matmul_fp`, empty `tiled_matmul_outer_fp`, three `tiled_matmul_auto_A_*` wrappers, unused cycle global | Unreachable SDK remnants | Remove after caller audit; do not create replacement no-ops |

`weight_format_t` remains `uint8_t` with values 0, 1, 2, retired 3, 4, 5,
6, 7, 8, retired 9, 10, 11, 12, 13. TENSOR/TOKEN/BLOCK/STRIPE/EXSIA,
HP1, software INT4/INT8, residual reconstruction and GGUF identities are
outside the decomposition's numeric changes.

## Transport ownership decision

The removed atomic global selected a borrowed transport pointer independently
for each command/counter call. Its only setter caller is
`tests/test-gemmini-matmul.cpp`. It permits cross-thread recorder capture and
cannot retain the recorder's storage. The PB model also had process-global
mutable state.

Execution uses explicit transport arguments for synchronous command sequences. GGML
argument copies retain an optional owned transport through `shared_ptr`;
tests use an aliasing pointer whose owner also contains the recorder and
record storage. Production baseline uses the immutable RoCC transport.
Keep an owning snapshot through the final drain and use that same transport
for observer counter calls. A failed transport remains fatal at the existing
void execution boundary, rather than returning successful hardware output.

PB gets a model instance per synchronous execution and one instance owned by
the existing runtime issuer. The model is destroyed on issuer exit; runtime
shutdown joins that issuer before releasing runtime state.
No new issuer scheduling or process-global test override is needed.

## Removed files and dependencies

Removed `legacy/gemmini_legacy.hpp`, `legacy/gemmini_legacy_isa.hpp`,
`legacy/transport.cpp`, and the old `legacy/README.md`.
Both `LICENSE.gemmini` and `LICENSE.gemmini-rocc-tests` retain their exact
bytes under `matmul/`, next to `matmul/README.md` with the SDK commit,
hashes and upstream attribution. Production has no include or link
dependency on `legacy/`.

The C11 API remains independent of GGML. GGML adapters still depend on
generated `gemmini_params.h`, GGML args/weight formats, activation metadata,
existing dequantization utilities, and logging. The separate FP32 DIM8
overlay remains external and unchanged. PoTAL's production API is unchanged.
No quantizer, reconstruction order, weight enum value or GGUF ID was changed.

## Targeted contract review

`ACC_READ_FULL_WIDTH`: develop's `gemmini_params.h` omits the feature macro;
the RTL-side header defines it at line 81. A source search of current
`ggml/`, `tests/`, `cmake/` finds only the PoTAL generated-header template,
with no consumer in Gemmini production or the SDK `gemmini.h`. The actual
contract is LOOP_WS `full_c`, encoded by `api/encode.c`; RTL
`LoopMatmul.scala:550-567` selects ACC-byte DMA offsets and
`read_full_acc_row` from `req.full_c`. `LocalAddr.scala:21-26` carries the
flag separately from row bits. Full INT32 readback therefore remains
unchanged. No macro or hardware/numeric change is needed for this removal.

Memory ordering: `api/transport/rocc.c` retains original no-clobber
inline assembly. Compiler ordering and hardware completion are distinct:
PB's existing empty `"memory"` assembly barriers surround buffer handoff,
and its runtime retains sequential CPU fences and release/acquire request
completion. The tiler retains its final transport drain. A RoCC CPU fence
waits for Rocket's RoCC busy path; Gemmini busy includes queue/unroller,
reservation-station and scratchpad work, and DMA write acknowledgements
(source references in `../isa-contract.md`).
This does not prove physical platform coherence or make a missing compiler
clobber harmless under every optimizer/LTO setting. No miscompilation was
demonstrated, so this refactor adds no per-command fence and changes no
ordering instruction. Hardware visibility and any separately verified
compiler-ordering change remain unverified.

Golden evidence: all original 13 shapes times four geometries pass count,
full-stream FNV hash, and stored-word comparison. Additionally,
`build-npu/phase2a-evidence/check-full-goldens.sh` extracts the verbatim
develop SDK, captures all commands plus drains, first proves its default
output byte-identical to the four existing goldens, then compares every
captured line with the new tiler. `golden-full.log` reports PASS. Existing
goldens were not regenerated.

Decomposition build and 13 affected tests pass:
`decomposition-verified-build.log`, `decomposition-tests.log`.
API/policy tests pass: `policy-build.log`, `policy-tests.log`.
All paths in this paragraph are under `build-npu/phase2a-evidence/`.
Full regression results are recorded separately in `phase2a-test-results.md`.
`source-audit.json` records unchanged weight enum text, preserved staged
diff, golden SHA-256 identities, and a clean `git diff --check HEAD`.
No commit, push, checkout, reset, stash or clean was performed.
