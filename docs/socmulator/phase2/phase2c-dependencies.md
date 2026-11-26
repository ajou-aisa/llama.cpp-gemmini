# Phase 2C dependency evidence

Evidence root: `build-npu/phase2c-evidence/`. `collect-dependencies.sh`
records the actual Ninja target graph and commands, compile database,
compiler-recorded include closure, CTest JSON and generated configuration
for twelve production/API configurations. These are observations from the
configured builds, including valid cached dependencies for unchanged objects,
not an inferred diagram.

| Lane | Evidence directory under `dependencies/` | Registrations |
| --- | --- | --- |
| macOS CPU | `mac-mac-cpu-only` | 60 |
| Gemmini API | `mac-gemmini-api` | 5 |
| PoTAL INT8 D16 API | `mac-potal-int8-d16-api` | 4 |
| PoTAL INT8 D32 API | `mac-potal-int8-d32-api` | 4 |
| PoTAL INT4 D32 API | `mac-potal-int4-d32-api` | 4 |
| macOS Gemmini INT8 | `mac-gemmini-int8-d16-ref` | 119 |
| Linux CPU | `linux-cpu-only` | 61 |
| Linux Gemmini INT8 | `linux-gemmini-int8-d16-ref` | 120 |
| Linux PoTAL INT8 D16 | `linux-potal-int8-d16` | 85 |
| Linux PoTAL INT8 D32 | `linux-potal-int8-d32` | 83 |
| Linux PoTAL INT4 D32 | `linux-potal-int4-d32` | 88 |
| Linux PB Q8_0 | `linux-gemmini-int8-d32-pb-reference` | 126 |

Each directory contains `target-graph.dot`, `target-commands.log`,
`compile_commands.json`, `compiler-includes.log`, `CMakeCache.txt`,
`ctest-registration.json`, and generated files when that lane produces them.
`production-isolation.json` records production source sets and findings.

PASS: all audited production targets contain zero CUDA evaluation source
files, zero evaluation compile definitions, zero experiment implementations,
and zero cudart/cuBLAS/evaluation-library link commands. No normal lane
requires the FP32 overlay. PoTAL compile databases contain zero Gemmini
source files. CPU/API lanes do not build common numerical software.
The actual common ExSIA and direct-residual compiler closures contain no
Gemmini/PoTAL hardware API, transport or external SDK headers. Profiling's
historical event/symbol names are not hardware dependencies.

PB remains a supported comparison baseline. Its generated manifest records
INT8, D32, SPAD 4 x 4096, ACC 2 / 1024 total rows, DMA 64, PARTIAL_SUM,
REFERENCE and WS, with RTL geometry `unverified`. Its RISC-V build remains
separate. The container's source-only mount cannot resolve the worktree's
external Git directory, so its manifest `software_commit` is empty; host
repository-state evidence records the revision. This limitation is retained
from the baseline, not filled with a fabricated commit.

The changed build owners are `CMakePresets.json`, `cmake/npu-config.cmake`
(FP32 dependency and WS selection), FP32 `backend.cmake`, `ggml/CMakeLists.txt`
(evaluation option include), Gemmini `CMakeLists.txt` (evaluation target
include), and the two PoTAL backend CMake files (prototype definitions moved
to experiment fragments). Production source lists otherwise keep their
implementations. The CUDA kernel targets retain their original names,
precision flags and library dependencies under explicit evaluation options.
The common collector itself is unchanged.

`preservation.json` verifies no diff in Gemmini API/PB/tiling/goldens,
common quantization/types/profiling, either PoTAL API or matmul tree,
GGML type declarations or shared format definitions. All eight CUDA
kernel/header files are byte-identical after moving. The common direct
executor and Gemmini graph/numerical adapters only redirect evaluation
includes. Separate experiment compile repairs are documented in
`potal-experiments.md`.

No IM2P/CYCLE_SIM implementation, legacy transport, new NPU transport,
compatibility wrapper or common wire protocol was introduced. Runtime
environment variables cannot activate an evaluation branch absent from the
compiled production target. Stock CUDA remains independently selectable;
its CUDA compiler/link execution is NOT RUN on this host.

FP32 target include/link wiring is inspected in source because its required
overlay is absent. CUDA-enabled host branches compile successfully in all
eight selected adapters with `-fsyntax-only`; commands and logs are in
`check-cuda-host-syntax.sh`, `cuda-host-syntax-summary.log`, and `optional/`.
This proves header/API wiring, not CUDA kernel compilation or GPU numerics.
