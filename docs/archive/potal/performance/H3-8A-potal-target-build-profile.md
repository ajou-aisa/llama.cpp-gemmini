# H3-8A: PoTal FireSim target build profiles

Date: 2026-10-28.

## Scope and provenance

The opt-in `build-potal-firesim.sh` wrapper supplies the H3-7C target defaults
through the existing `build-riscv.sh` option resolver. Generic CPU, Gemmini
and IM2P defaults are not changed. No commit or push was made.

- Software repository: `ajou-aisa/llama.cpp-gemmini`, branch `potal`.
- Starting software HEAD: `fc0bdcdd0dba8ccfef8a54cd2e3c7f39a0be3c9b`.
- RTL repository: `ajou-aisa/chipyard-aisa`, branch `potal`.
- Starting RTL HEAD: `10ff4a10a87662edc7efd5d1a718027c1853550b`.
- These SHAs identify the starting commits, not the uncommitted H3-8 changes.
- H3-7C's historical software HEAD was
  `8fb19aa5b66ace4cc50014581e5b3fe5ad7fc546`; its deployed binaries remain
  separate from these new build directories.

## Invocation and isolation

On this workstation, initialize the existing FireSim environment first:

```bash
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
conda activate /home/alveo/firesim/.conda-env
cd generators/potal/software/workspace/llama.cpp-potal
```

The checked-in `env.sh` currently attempts a missing Chipyard-local
`.conda-env`. The explicit activation above selects the installed FireSim
environment without editing either environment script.

```bash
bash build-potal-firesim.sh functional 16 --dry-run
bash build-potal-firesim.sh functional 32 --dry-run
bash build-potal-firesim.sh performance 16 --dry-run
bash build-potal-firesim.sh performance 32 --dry-run
```

Omit `--dry-run` to configure and build for RISC-V. DIM is mandatory; it is
never inferred as 16 when omitted or invalid. Supported target dimensions
are exactly 16 and 32.

Default directories are `build-potal-firesim-d<DIM>-<profile>`. Existing
DIM16 builds are not reconfigured. `BUILD_DIR` can explicitly select another
directory. The existing resolver still applies command-line, environment,
default-environment and script-default precedence, in that order. Resolved
options are emitted as machine-readable `IM2P_EFFECTIVE_CONFIG`.

## Common defaults

```text
GGML_POTAL=ON
GGML_GEMMINI=OFF
GGML_BACKEND_DL=OFF
GGML_GEMMINI_EXECUTION_BACKEND=HARDWARE
GGML_GEMMINI_OPTION=WS
GGML_GEMMINI_ACTIVATION_QUANT=EXSIA
GGML_GEMMINI_ENABLE_RMD=ON
GGML_GEMMINI_DEFAULT_RMD_BACKEND=WS
GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE
GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY=2
GGML_POTAL_PAIRED_AUTO=ON
GGML_GEMMINI_EXSIA_DEFAULT_MODE=SEQUENTIAL
GGML_GEMMINI_ACTIVATION_BITS=8
GGML_GEMMINI_WEIGHT_BITS=8
GGML_GEMMINI_BLOCK_SIZE=32
CYCLE_SIM=0
CYCLE_DETAIL=0
POTAL_PLANNING_DIAGNOSTICS=OFF
GGML_CPU_CYCLE_LOG=OFF
GGML_GEMMINI_ENABLE_A4NKS_ATTENTION=OFF
GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION=OFF
```

Functional uses `LOG_DEBUG=1, LOG_CYCLE=1`. Performance uses
`LOG_DEBUG=0, LOG_CYCLE=1`. `GGML_CPU_CYCLE_LOG=OFF` matches the actual H3-7C
reference cache; it does not disable canonical request latency events.

`src/CMakeLists.txt` independently defines `GGML_POTAL_ATTENTION=1` for the
llama graph when PoTal is enabled. The unrelated Gemmini A4NKS path stays OFF.
The DIM32 cross-build's `src/CMakeFiles/llama.dir/flags.make` confirms both
`GGML_POTAL_ATTENTION=1` and `LOG_CYCLE=1`.

## DIM and generated-parameter contract

The selected header remains the source of hardware memory geometry.
`cmake/potal-hw-params.cmake` rejects disagreement between its DIM,
`POTAL_EXPECTED_DIM`, `GGML_GEMMINI_DIM` and the wrapper's
`POTAL_TARGET_DIM`. The wrapper appends its target DIM after forwarded
arguments, so overriding both software defaults cannot silently change the
requested target. Missing or unreadable headers fail configuration.
The named quad-Rocket DIM16/DIM32 `POTAL_HW_CONFIG` values are also checked
against the header DIM; their names cannot contradict the selected geometry.
The wrapper also locks `POTAL_TARGET_PARAMS` to the corresponding canonical
RTL-generated header. Configure compares the complete selected and target
header SHA256 values, rejecting a same-DIM header with different memory
geometry. The manifest records both header paths and hashes.

The existing C11 static assertions compare the selected original header
with the generated adapter. Configuration copies the original bytes and
records header hash, software/header repository revisions, target config,
compiler, toolchain and dimensions in `generated/potal_build_manifest.txt`.

| Parameter | DIM16 | DIM32 |
|---|---:|---:|
| Scratchpad banks | 4 | 4 |
| Rows per scratchpad bank | 4096 | 2048 |
| Total accumulator rows | 1024 | 512 |
| Scale slots per column | 256 | 256 |
| Index entries per buffer | 32 | 64 |
| Maximum DMA transaction bytes | 64 | 64 |
| Full INT32 output alignment | 64 bytes | 128 bytes |
| Physical passes per K32 scale block | 2 | 1 |

Headers are under
`../../gemmini-rocc-tests/include/gemmini_potal{,_d32}_params.h`.

```text
DIM16 SHA256:
b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7
DIM32 SHA256:
3580b84f15ed8de9f9d311edea64eaa7c1f9bb308b1552a8f79453ec2c05ea33
```

The DIM32 header was independently regenerated through
`potal.PoTalDim32Header`, which calls the actual RTL configuration's
`generateHeader()`, into `/tmp/h3-8/gemmini_potal_d32_params.h`.
Its bytes and SHA256 match the selected DIM32 source header. No DIM16
header was regenerated or replaced.

## Canonical timing

The timing implementation was not changed. Both profiles retain
`ggml/src/ggml-gemmini-utils/src/performance.cpp` request/token events:

- TTFT: `first_token_ready - request_start`.
- TPOT: average gap between successive generated-token-ready events.
- Request latency: the existing request boundary interval.

These are guest monotonic request measurements when running inside
FireSim, not accelerator cycles, FPGA host wall time, or hardware wait time.
CPU-observed API intervals and overlapping stages must not be added as
independent request-wall contributions.

## Phase A validation before DIM32 production edits

1. All four profile/DIM combinations passed dry-run checks against the
   resolved machine-readable values and selected header DIM.
2. Conflicting software DIM, expected DIM and requested target DIM each
   failed a real CMake API-only configure as required.
3. A fresh DIM16 RISC-V full-backend configure passed in
   `build-potal-h3-8-profile-d16`. Its manifest retained the historical
   DIM16 parameter hash and the H3-7C functional options.
4. The existing, unrecompiled H3-7C native attention API and attention tests
   passed. The seven historical final-result/script hash checks passed.
5. `bash -n build-potal-firesim.sh build-riscv.sh` passed.

Evidence: `/tmp/h3-8/phase-a-configure.log`,
`phase-a-regression.log`, `header-provenance.log`, and monitor transcripts
for the frozen H3-7C binaries and hash validation.
The final four-way software/expected/target/config DIM rejection regression
also passed in `/tmp/h3-8/phase-a-final-profile.log`.
The final canonical profile regression additionally rejected a DIM16 header
with BANK_ROWS changed from 4096 to 8192:
`/tmp/h3-8/profile-canonical-final.log`, exit 0.
All seven historical result/script hashes, the DIM16 bitfile and both
parameter headers still match their original hashes. Model and prompt
bytes are unchanged. Evidence:
`/tmp/h3-8/dim16-baseline-preservation-final.log`, exit 0.

The existing canonical host-timing test passed for both Functional and
Performance profiles. Evidence: `/tmp/h3-8/functional-timing-contract.log`
and `/tmp/h3-8/performance-timing-contract.log`. These tests validate event
accounting, not DIM32 accelerator performance.

The actual Functional DIM32 wrapper completed its full default RISC-V
build, including API/backend tests, hardware smoke binaries, `llama-cli`,
`llama-perplexity` and `llama-quantize`, with exit 0:

```bash
BUILD_DIR=build-potal-h3-8-d32-functional BUILD_JOBS=8 \
  bash build-potal-firesim.sh functional 32
```

Evidence: `/tmp/h3-8/canonical-d32-functional-build-final.log`.
Its generated manifest records equal selected-header and target-header
SHA256 values, both
`3580b84f15ed8de9f9d311edea64eaa7c1f9bb308b1552a8f79453ec2c05ea33`.

Final review found and fixed a wrapper-only precedence error: assigning
profile defaults unconditionally overwrote caller-supplied `*_DEFAULT`
values before the existing resolver saw them. A regression reproduced
capacity 2 instead of the requested default 3. The wrapper now supplies
each profile default only when that default-environment variable is unset
or empty, matching the existing build script convention. Locked target DIM
and canonical target-header arguments remain mandatory.

The expanded profile regression passes for both Functional and Performance:
default-environment capacity 3, public-environment capacity 4, then
command-line capacity 5, in that precedence order. It also reruns all four
clean profile/DIM defaults and geometry mismatch checks. Evidence:
`/tmp/h3-8/profile-default-precedence-before.log`,
`profile-precedence-regression-before.log` (expected exit 1), and
`profile-precedence-final.log` (exit 0). Clean-profile effective values are
unchanged from the successful canonical RISC-V build. Shell syntax passes.
The post-elaboration preservation check also passes all seven historical
hashes, both parameter-header hashes, the DIM16 bitstream hash, executable
metadata for `build-riscv.sh`, and scoped whitespace validation:
`/tmp/h3-8/final-preservation-after-generation.log`.

The unrelated existing `tests/test-gemmini-off-build.cmake` failed because
it expects `build-arm64.sh` to default Gemmini OFF, while that unchanged
script explicitly defaults Gemmini ON. This pre-existing expectation was
not edited to pass. No generic defaults were changed.

Default LSP support is unavailable for CMake and Scala; Bash/YAML language
servers are not installed. Configuration and compiler/test results, rather
than those unavailable diagnostics, are the validation gates.
