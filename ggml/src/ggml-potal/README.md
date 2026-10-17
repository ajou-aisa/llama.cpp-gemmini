# PoTal minimal RoCC API and GGML backend

Standalone C11 API for the PoTal SCU hardware (Gemmini-derived, Sat32 ACC),
and the `POTAL` GGML backend built on it. Neither uses `gemmini.h`,
`xcustom.h`, a Gemmini header path or `IM2P.sim`. `build-riscv.sh` builds this
backend by default (`GGML_GEMMINI=OFF`, `GGML_POTAL=ON`).

| Configuration | Result |
|---|---|
| `POTAL_API_ONLY=ON`, `GGML_GEMMINI=OFF` | only `potal-api` and its tests; ggml/llama are not configured |
| `GGML_POTAL=ON`, `POTAL_API_ONLY=OFF`, `GGML_GEMMINI=OFF` | ggml/llama plus the `ggml-potal` backend and all PoTal tests |
| `GGML_POTAL=OFF`, `POTAL_API_ONLY=OFF` | the existing ggml/llama build (CPU or Gemmini backend), unchanged |
| `GGML_POTAL=ON` with `GGML_GEMMINI=ON` or with `POTAL_API_ONLY=ON` | configure error |

| File | Role |
|---|---|
| `include/potal.h` | Public API: status codes, encoders, scale control, WS GEMM |
| `include/potal_rocc.h` | Unchecked raw custom-3 issue layer (RISC-V only) |
| `potal.c` | Validation, encoding, hardware calls; cross-checks the original header |
| `tests/potal-api-test.c` | Host/target packing and validation test (`potal-api-test`) |
| `tests/potal-api-smoke.c` | DIM16 hardware test (`potal-api-smoke`) |
| `../../include/ggml-potal.h` | GGML backend API: init/reg, reject reasons, counters |
| `ggml-potal.cpp` | Backend, device and registry; `supports_op`, `graph_compute` |
| `ops.{hpp,cpp}` | GGML op validation and dispatch (I8 and Q8_HP1 MUL_MAT), logging |
| `matmul/options.hpp` | Fixed W4-5 policy: TENSOR activation quant, Q8_HP1, compensation OFF, scale window |
| `matmul/dense.{hpp,cpp}` | F32 x Q8_HP1 contract, DMA staging operands, F32 reconstruction, integer golden |
| `matmul/tiling.{hpp,cpp}` | `potal_tiled_matmul_auto/_tiled_matmul/_outer/_sp_tiled_matmul_ws` (gemmini.h layering) |
| `matmul/pack.{hpp,cpp}` | I8 MUL_MAT <-> validated PoTal GEMM contract, A/B packing, C -> F32 |
| `matmul/execution.{hpp,cpp}` | hart-0 issuer thread: TLB flush, then validated GEMMs and tiled jobs |
| `ggml-potal-args.h` | `ggml_potal_args_t`: copy of `ggml_gemmini_args_t` reduced to TENSOR + Q8_HP1 |
| `quants/act/{buffer,meta}.hpp`, `quants/act/quantize.*`, `quants/act/tensor/*` | Activation quantization copied from `ggml-gemmini/quants/act` (TENSOR only, no RMD) |
| `quants/common/{hp1_scu,math,tensor_util}.hpp`, `quants/common/weight_reader.*` | Copied from `ggml-gemmini/quants/common` (Q8_HP1 branches only) |
| `quants/weight/hp1/hp1.{hpp,cpp}` | Q8_HP1 -> B / carriers / channel scales on the copied reader, PoTal row checks |
| `tests/potal-ggml-unit-test.cpp` | Hardware-free backend test (`potal-ggml-unit-test`) |
| `tests/potal-tiling-unit-test.cpp` | Hardware-free W4-5 test (`potal-tiling-unit-test`) |
| `tests/potal-ggml-smoke.cpp` | FireSim GGML MUL_MAT hardware test (`potal-ggml-smoke`) |
| `tests/potal-hp1-smoke.cpp` | FireSim F32 x Q8_HP1 hardware test (`potal-hp1-smoke`) |
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
Setting `POTAL_HW_PARAMS`/`POTAL_EXPECTED_DIM` without `POTAL_API_ONLY=ON` or
`GGML_POTAL=ON` is a configure error. Both modes run the same
`potal_configure_hw_params()` and write the same `<build>/generated/` files.

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
| `potal_tlb_flush()` | FLUSH (7, rs1=0 flush+retry); `POTAL_OK` = issued; `ERR_BUSY` if a GEMM is unfenced |
| `potal_gemm_ws_check(&g)` | argument -> capacity -> validated-scope checks |
| `potal_gemm_ws_scale_slot(&g, k_block, n_tile, &slot)` | `base + k_block * J + n_tile` |
| `potal_gemm_ws_encode(&g, &program)` | CONFIG_LD A/B, CONFIG_EX, CONFIG_ST, LOOP_WS 9..13, LOOP_WS 8 |
| `potal_gemm_ws_issue(&g)` | issues the program; `POTAL_OK` = issued, not finished |

Rules:

- Per process, before the first GEMM: `potal_tlb_flush()` then
  `potal_fence()`, both `POTAL_OK`. The accelerator TLB is not flushed on
  Linux context switches, so a new process could otherwise reuse a stale
  translation (observed: C left at its initial value). FLUSH has no response;
  it travels in the same in-order raw command queue as FENCE and is consumed
  at the unrolled command queue, which FENCE requires to be empty
  (`Controller.scala`), so the fence completing implies the flush was applied.
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
- Not thread-safe. Exactly one thread, pinned to the Linux CPU whose
  `/proc/cpuinfo` `hart` is 0 (the hart the PoTal RoCC is attached to), may
  issue commands. Other CPUs raise an illegal-instruction trap on custom-3.
- One process at a time. Back-to-back processes are covered by the per-process
  TLB flush; concurrent processes sharing the accelerator are not supported
  (scale slots, TLB and in-flight GEMMs are global hardware state).

## Multicore smoke (`potal-multicore-smoke`, RISC-V Linux, DIM16)

One issuer pthread pins itself to the hart-0 CPU (found from
`/proc/cpuinfo`; never assumed) and is the only thread issuing RoCC commands:
TLB flush, fence, then `--iterations` GEMMs (alternating K16/K32, scale slots
from 128). CPU workers each pin themselves to a different other allowed online
CPU and run INT8 quantize + INT32 dot-product work units; checksums are
recomputed serially after join. The main thread never narrows its own
affinity, so workers do not inherit the issuer CPU.

```
--mode=pthread|openmp   worker pool (openmp needs -DPOTAL_ENABLE_OPENMP=ON)
--workers=N             default: OMP_NUM_THREADS (openmp), else all non-hart0 CPUs
--iterations=N          GEMMs, default 4
--work-units=N          units per worker, default 4000
--verbose               worker progress lines
```

Worker count precedence: `--workers` > `OMP_NUM_THREADS` (openmp mode only) >
available CPUs. In openmp mode the team is created with `num_threads(N)` and
must have exactly N threads; each OpenMP thread pins itself explicitly, so
`OMP_PROC_BIND`/`OMP_PLACES`/`GOMP_CPU_AFFINITY` are logged and overridden.
`--mode=openmp` in a build without OpenMP fails (exit 2); it never falls back.

Results (separate):

| Line | Meaning |
|---|---|
| `CPU_MULTICORE_VERIFIED` | every worker: affinity = its own CPU, all periodic `sched_getcpu()` samples on it, not the hart-0 CPU, all CPUs distinct, full progress, checksum = reference |
| `POTAL_GEMM_VERIFIED` | flush/fence OK and every GEMM equals the int64 golden model |
| `CPU_POTAL_OVERLAP_OBSERVED` / `_NOT_OBSERVED` | summed worker progress increased between the return of a verified GEMM issue and the return of its FENCE |

`POTAL_MULTICORE_SMOKE PASS` (exit 0) requires the first two. Overlap is
reported only; if not observed, raise `--work-units` and/or `--iterations`.
Times are guest `CLOCK_MONOTONIC` seconds from process start and
`CLOCK_THREAD_CPUTIME_ID`; they are not FPGA host time and not a speedup
measurement. Do not run the whole program under `taskset -c <one cpu>`: the
workers would have no other CPU and the test fails by design.

OpenMP build: `-DPOTAL_ENABLE_OPENMP=ON` runs `find_package(OpenMP C REQUIRED)`
with the RISC-V toolchain (configure fails if it cannot compile and link) and
links only `potal-multicore-smoke`; it is independent of `GGML_OPENMP` /
`GGML_GEMMINI_ENABLE_OPENMP`. The dynamic build needs `libgomp.so.1` in the
guest rootfs (toolchain sysroot `lib/libgomp.so.1`).

## GGML backend (`GGML_POTAL=ON`)

Device `POTAL` (type ACCEL, statically registered before CPU; `GGML_BACKEND_DL`
is not supported). It executes `GGML_OP_MUL_MAT` in two forms: the validated
I8 GEMM below (W4-4, unchanged) and F32 x Q8_HP1 through the tiled path
(W4-5, next section). The I8 form:

| | GGML | PoTal (`potal.h`) |
|---|---|---|
| weights | `src0` I8, `ne = {K, N}`, row n = K values | `B = src0^T`, int8 `[K][N]` |
| activations | `src1` I8, `ne = {K, M}`, row m = K values | `A = src1`, int8 `[M][K]` |
| result | `dst` F32, `ne = {N, M}`, `dst[m][n] = sum_k src1[m][k] * src0[n][k]` | `C` int32 `[M][N]`, converted to F32 |
| shape | 2-D, `M = N = 16`, `K = 16` or `32`, contiguous, host memory | one DIM16 output tile |
| scales | none in a plain `GGML_OP_MUL_MAT` | every carrier `0` (x1) in one scale block |

With x1 carriers `|C| <= 32 * 128 * 128 < 2^24`, so the INT32 result is the
exact integer dot product and exact in F32. No scale is invented and F32
inputs are never quantized: F32/F16/quantized MUL_MAT, other shapes,
batches, transposed/permuted/strided operands, `op_params[1..]` tags
(attention roles) and every other op (bias ADD, SOFT_MAX, FLASH_ATTN_EXT, ...)
are rejected by `supports_op` with a reason (`ggml_potal_reject_name()`);
`ggml_backend_sched` then places supported ones on the CPU backend. Metadata
view ops (NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE) of host tensors are accepted
and are no-ops. The default buffer type is the CPU (host) buffer type; the
backend copies operands into its own DMA staging buffers.

`graph_compute` first checks every node; a node outside the contract fails
the graph with `GGML_STATUS_FAILED` before any hardware command (no CPU
fallback inside the backend, no abort). A failed node's F32 `dst` is set to
NaN, so stale or partial data is never presented as a result.

Execution of one node (`matmul/execution.cpp`):

```
graph_compute thread: pack src1 -> A[16][K], src0 -> B[K][16]   (pack_us)
  -> process-wide issuer thread (created on the first request):
       once: find the hart-0 CPU in /proc/cpuinfo, pin + verify affinity,
             potal_tlb_flush() -> potal_fence()            [POTAL_ISSUER] [POTAL_TLB]
       copy A/B into static DMA staging buffers (lda 32, ldb 16), C <- sentinel
       potal_scale_write(slot 192, generation 1..255, 16 columns, carrier 0)
       potal_gemm_ws_issue()  (passes_per_block = K/16)   dispatch=HARDWARE
       potal_fence()                                      fence_status
       copy C out, potal_scale_release() x16, check no C element kept the sentinel
graph_compute thread: C int32 -> dst F32                  (unpack_us)
```

- Only the issuer thread issues PoTal commands; the caller's thread and
  affinity are untouched. Requests are serialized (one at a time).
- The staging buffers live for the whole process, use the smoke-test layout
  and are touched only by the issuer while no GEMM is in flight.
- Scale slot 192 is not used by `potal-api-smoke` (0..41) or
  `potal-multicore-smoke` (128..129).
- Software rejections before any command (`potal_gemm_ws_check`) fail only the
  node. Any failure after a command was sent (scale write, issue, fence,
  release, C not written) disables the issuer for the rest of the process:
  buffers and slots of a possibly unfinished GEMM are never reused.
- One process at a time, as for the API. Off RISC-V the request is validated
  by `potal_gemm_ws_issue()` and fails with `POTAL_ERR_NO_HARDWARE`.

Logs (GGML log, INFO unless noted): `[POTAL_BACKEND] registered`,
`device_init`, per node `node=... op=MUL_MAT src0_type=i8 ... M N K`,
`[POTAL_MATMUL] pack_A_bytes/pack_B_bytes`, `scale_base/generation`,
`dispatch=HARDWARE issue_status=POTAL_OK` (printed only after the issue
returned `POTAL_OK`), `fence_status=POTAL_OK`, `scale_release=POTAL_OK`,
`output_type=F32`, `[POTAL_BACKEND] hw_gemm_count`; errors as
`dispatch=FAILED stage=... status=...`; `supports_op` decisions for MUL_MAT at
DEBUG. `ggml_backend_potal_get_stats()` returns per-backend counters (nodes,
hardware requests, issued, fenced, completed, failed, last status/slot/
generation), cumulative pack/stage/scale/issue/fence/unpack times, device-wide
`supports_op` answers by reason and the issuer CPU/tid. Times are guest
`CLOCK_MONOTONIC`, not FPGA cycles or array utilization, and say nothing about
GGML inference speed.

Tests:

| Binary | Runs on | Checks | Final line |
|---|---|---|---|
| `potal-ggml-unit-test` | host and target, no accelerator access | registry/GUID/props, `supports_op` table with reasons and counters, packing against a CPU-backend F32 MUL_MAT golden, refusal before hardware, off-RISC-V `POTAL_ERR_NO_HARDWARE` with NaN dst, scheduler placement POTAL vs CPU | `POTAL_GGML_UNIT_TEST PASS` |
| `potal-ggml-smoke` | FireSim (RISC-V Linux) | `ggml_mul_mat` graphs on the POTAL backend: K16 and K32, 3 patterns each, plus POTAL + CPU scheduler; per GEMM `GGML_BACKEND=POTAL`, `DISPATCH=HARDWARE`, `POTAL_GEMM_ISSUED`, `POTAL_FENCE_PASS` (from backend counters), `GGML_MUL_MAT_NUMERICAL_PASS` (all 256 outputs vs int64 golden); issuer CPU maps to hart 0; main thread affinity unchanged | `POTAL_GGML_SMOKE PASS` |

`potal-ggml-smoke [--case=k16|k32|sched|all] [--verbose]`; `--verbose` adds
GGML DEBUG lines. Exit 0 pass, 1 fail, 2 usage/environment error.

Not implemented in the I8 form: other shapes; those go through the Q8_HP1 form
only. Not implemented in the backend: F16/other quantized MUL_MAT,
Stripe/Paired execution, DIM32/64, dynamic backend loading, concurrent processes.

## Q8_HP1 tiled path (W4-5)

`GGML_OP_MUL_MAT` with `src0` `GGML_TYPE_Q8_HP1` (`ne = {K, N}`, contiguous)
and `src1` F32 (`ne = {K, M}`, contiguous), 2-D, any M and N,
`K % 32 == 0`:

```
src1 F32 --tensor quant (one scale for the whole tensor)--> A int8 [M][K]
src0 HP1 --validate + unpack--> B int8 [K][N], carrier[K/32][N], channel_scale[N]
potal_tiled_matmul_auto(M, N, K, A, B, NULL, C, ...)          on the hart-0 issuer
  per LOOP_WS invocation: SCALE_WRITE every (block, column) carrier
                          -> potal_gemm_ws_issue (validated 16x16x16|32 tile)
                             or potal_loop_ws_issue (everything else)
                          -> POTAL_FENCE -> SCALE_RELEASE
dst[m][n] = float(double(C[m][n]) * activation_scale * channel_scale[n])
```

| gemmini.h | PoTal | Notes |
|---|---|---|
| `tiled_matmul_auto` | `potal_tiled_matmul_auto` | same arguments, plus the carrier operand and the issuer session |
| `tiled_matmul` | `potal_tiled_matmul` | GEMMINI_ASSERTIONS returned as statuses |
| `tiled_matmul_outer` | `potal_tiled_matmul_outer` | same I0/J0/K0, last tile, padding, A/B/C addressing, i0-j0-k0 order; bias/C only with the first/last K chunk |
| `sp_tiled_matmul_ws` | `potal_sp_tiled_matmul_ws` | `ex_accumulate = !no_bias \|\| D == NULL`; adds scale write/fence/release |
| `gemmini_loop_ws` | `potal_gemm_ws_issue` / `potal_loop_ws_issue` | one encoder and one RoCC issue sequence in `potal.c` |

Tile selection is `tiled_matmul_auto`'s WS branch (start from the `db_` sizes,
grow J, I, K while A+B fit half the SPAD and C half the ACC), plus the PoTal
limits: `tile_K` starts and grows in whole scale blocks (`passes_per_block =
32 / DIM`), `tile_I * tile_J <= 64` work IDs, and `(tile_K / passes_per_block)
* tile_J <= 256` scale slots. Where these limits do not bind (and
`passes_per_block = 1`) the choice equals gemmini.h's (`potal-tiling-unit-test`
section A). Supported request subset: WS, no D/bias, no activation, identity
scale factors, no transpose, `full_C`; anything else is
`POTAL_ERR_UNSUPPORTED` before a command. Operand reuse (`a/b_spad_id`) is not
used (as `GEMMINI_DISABLE_WS_REUSE=1`).

Q8_HP1 validation: the copied `wreader::validate` (storage contract, zero
padding, finite `channel_scale`, `m` in `0..32767` or `INT16_MIN` = zero
carrier), plus in `quants/weight/hp1` one bit-identical, non-negative
`channel_scale` per row (a row of 0 only with all blocks zero); any violation
fails the node. The HP1 exponent is applied only by the SCU, never again in
reconstruction, and weights are never dequantized and requantized.

Activation quantization is selected by the existing build option
`GGML_GEMMINI_ACTIVATION_QUANT` (must be `TENSOR`; `build-riscv.sh` defaults to
it) with `GGML_GEMMINI_ACTIVATION_BITS=8`; other values fail configure.
(`quants/act/tensor`, copied from Gemmini with RMD removed): `scale = max_abs /
127` over the finite values of the whole tensor, `lrint` (half to even), clip
`[-128, 127]`; NaN/Inf -> code 0; all-zero tensor -> scale 1, codes 0; no
finite value -> the node fails. src1 must be contiguous.

Error compensation is OFF and absent: no outlier selection, residual
extraction, RMD/direct correction, compensation GEMM or recomposition exists in
`ggml-potal` (`options::compensation` is only reported).

Integer contract (golden `reference_gemm_i32`): for each output, K passes of
DIM elements in order; `partial << carrier` with Sat32 (PoTalSCU), first pass
overwrites, later passes Sat32-add (AccPipe).

Unvalidated on hardware until `potal-hp1-smoke` passes on FireSim: multiple I/J
tiles, padding, K accumulation across invocations (`C == NULL`), tile origins
at DIM-byte offsets. A failure after a command was sent disables the issuer
(fail-stop) and leaks that job's DMA buffers on purpose.

Logging uses the existing `ggml-gemmini-utils` system (no PoTal logger):
`log::debug` JSONL (`LOG_DEBUG=1`, file `debug-log.jsonl` under
`GEMMINI_LOG_DIR`) with `[POTAL]` records per node (backend, node, layer via
`resolve_matmul_layer`, `weight_type=Q8_HP1`, `activation_quant=TENSOR`,
`compensation=OFF`, M/N/K, tile_I/J/K, I0/J0/K0, pads, SPAD/ACC rows used/
available, scale slots, generation, activation scale, channel-scale and
exponent statistics, zero carriers, GEMM issue count, fences, status) and
`cpu_fallback` records for rejected MUL_MATs; `log::cycle` intervals
(`LOG_CYCLE=1`: `potal.act_quant_tensor`, `potal.weight_pack_hp1`,
`potal.tiled_gemm`, `potal.output_reconstruct`, target `rdcycle`). Counters:
`ggml_backend_potal_stats` `hp1_*`, `tiled_*`, `scale_writes/releases`,
`zero_carriers`, `act_clipped`, guest seconds per stage, and
`supports_mul_mat_rejected`. Guest seconds are not FPGA cycles. The CPU
backend does not run Q8_HP1 MUL_MAT, so an HP1 GEMM is never silently computed
on the CPU.

| Binary | Runs on | Checks | Final line |
|---|---|---|---|
| `potal-tiling-unit-test` | host and target (emulator, no accelerator) | tile choice vs gemmini.h, tiled execution on an RTL-semantics emulator == integer golden (multi-tile, K chunks, padding, zero carriers, Sat32), scale release, fail-stop, HP1 validation, tensor quantization, exact F32 reconstruction, GGML contract | `POTAL_TILING_UNIT_TEST PASS` |
| `potal-hp1-smoke` | FireSim (RISC-V Linux) | F32 x Q8_HP1 graphs on the POTAL backend, every element == CPU golden, invocations fenced, scales released; 2-layer chain; generation wrap | `POTAL_HP1_SMOKE PASS` |

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

# RISC-V Linux cross build (bin/potal-api-test, bin/potal-api-smoke,
# bin/potal-multicore-smoke; add -DPOTAL_ENABLE_OPENMP=ON for --mode=openmp)
BUILD_DIR=build-potal-d16 ./build-riscv.sh -DGGML_POTAL=OFF -DPOTAL_API_ONLY=ON \
  -DPOTAL_EXPECTED_DIM=16 -DPOTAL_HW_PARAMS:FILEPATH="$PARAMS/gemmini_potal_params.h" \
  -DPOTAL_HW_CONFIG=PoTalSat32GemminiRocketConfig

# Guest (FireSim Linux): back-to-back processes, each flushes the TLB itself
./potal-api-smoke; ./potal-api-smoke
./potal-multicore-smoke --mode=pthread --workers=3 --iterations=4 --verbose
OMP_NUM_THREADS=3 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE OMP_DISPLAY_ENV=TRUE \
  ./potal-multicore-smoke --mode=openmp --iterations=4 --verbose

# Bare-metal ELF (needs riscv64-unknown-elf-gcc: source chipyard env.sh)
scripts/potal-build-baremetal.sh build-potal-host

# GGML backend, host (expects "POTAL_GGML_UNIT_TEST PASS")
cmake -S . -B build-potal-ggml-host -DGGML_GEMMINI=OFF -DGGML_POTAL=ON -DGGML_BACKEND_DL=OFF \
  -DLLAMA_CURL=OFF -DPOTAL_EXPECTED_DIM=16 -DPOTAL_HW_PARAMS:FILEPATH="$PARAMS/gemmini_potal_params.h"
cmake --build build-potal-ggml-host --target potal-ggml-unit-test potal-tiling-unit-test potal-api-test -j
build-potal-ggml-host/bin/potal-ggml-unit-test
build-potal-ggml-host/bin/potal-tiling-unit-test   # expects "POTAL_TILING_UNIT_TEST PASS"

# GGML backend, RISC-V Linux cross build: the default of build-riscv.sh
# (bin/potal-*-test, bin/potal-*-smoke, bin/libggml*.so, bin/llama-cli, bin/llama-perplexity)
BUILD_DIR=build-potal-ggml-d16 ./build-riscv.sh -DPOTAL_HW_CONFIG=PoTalSat32GemminiRocketConfig

# Guest (FireSim Linux): copy potal-ggml-smoke with libggml.so, libggml-base.so,
# libggml-cpu.so and libggml-potal.so (RPATH is $ORIGIN, or set LD_LIBRARY_PATH)
./potal-ggml-smoke --case=k16 --verbose
./potal-ggml-smoke --case=k32 --verbose
./potal-ggml-smoke --case=sched
./potal-hp1-smoke --quick && ./potal-hp1-smoke      # expects "POTAL_HP1_SMOKE PASS"

# Q8_HP1 model (host): the registered llama-quantize type
cmake -S . -B build-host-quant -DGGML_GEMMINI=OFF -DLLAMA_CURL=OFF && cmake --build build-host-quant --target llama-quantize -j
build-host-quant/bin/llama-quantize <model-f16.gguf> <model-q8_hp1.gguf> Q8_HP1

# End to end (model path and prompt are yours). Logs in $GEMMINI_LOG_DIR.
export GEMMINI_LOG_DIR=$PWD/potal-log
./llama-cli -m <model.Q8_HP1.gguf> -p "<short prompt>" -n 4 -t 1 --no-warmup
./llama-perplexity -m <model.Q8_HP1.gguf> -f <text file> -c 128 --chunks 1
```

Build with `-DLOG_DEBUG=1 -DLOG_CYCLE=1` (`build-riscv.sh` arguments) for the
`[POTAL]` JSONL records and stage cycles. A run counts as PoTal end to end only
if `hp1_completed > 0` and `hp1_failed == 0` in the stats / the debug log shows
`dispatch=HARDWARE` for the model's MUL_MATs.
