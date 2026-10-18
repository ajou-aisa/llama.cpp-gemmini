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
| `matmul/options.hpp` | The configured options (activation algorithm, compensation ON/OFF) for logs and checks; no fixed policy |
| `matmul/dense.{hpp,cpp}` | F32 x Q8_HP1 contract, DMA staging operands, args as Gemmini sets them, route selection and execution, compensation step, integer golden |
| `matmul/tiling.{hpp,cpp}` | `potal_tiled_matmul_auto/_tiled_matmul/_outer/_sp_tiled_matmul_ws` (gemmini.h layering) |
| `matmul/pack.{hpp,cpp}` | I8 MUL_MAT <-> validated PoTal GEMM contract, A/B packing, C -> F32 |
| `matmul/execution.{hpp,cpp}` | hart-0 issuer thread: TLB flush, then validated GEMMs and tiled jobs (TLB flush per job, caller trace context) |
| `quants/weight/hp1/hp1.{hpp,cpp}` | Q8_HP1 -> B / carriers / channel scales on the Gemmini weight reader (HP1 SCU route), PoTal row checks |
| `../ggml-gemmini/{quants,residual,matmul/options.cpp,ggml-gemmini-telemetry.cpp}` | Unmodified Gemmini sources compiled into ggml-potal: `ggml_gemmini_args_t`, the five activation quantizers and their metadata, the weight reader/route, cpu_direct residual compensation, matmul options (see CMakeLists.txt `POTAL_GEMMINI_SHARED_SOURCES`) |
| `tests/potal-ggml-unit-test.cpp` | Hardware-free backend test (`potal-ggml-unit-test`) |
| `tests/potal-tiling-unit-test.cpp` | Hardware-free tiling/route/compensation test (`potal-tiling-unit-test`) |
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
and are no-ops. The default buffer type is the POTAL buffer type
(`ggml-potal-buffer.cpp`): aligned host memory the CPU backend reads directly,
under its own identity so llama.cpp's model loader lists POTAL as an ACCEL
weight buffer candidate (`make_cpu_buft_list()` skips an ACCEL device whose
default is the CPU buffer type). `buffer_from_host_ptr` aliases the loader's
mmap without copying or owning it; each buffer carries a generation that is
part of the weight cache key. A shared GPT-2 `token_embd.weight` is placed
twice: CPU buffer for the input GET_ROWS, POTAL buffer for the output MUL_MAT
(with mmap both alias the same file bytes). The backend copies operands into
its own DMA staging buffers. `potal-gguf-load-test` (tests/) checks this
placement on a real GGUF without computing anything.

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

Logs: the existing Gemmini debug JSONL (`$GEMMINI_LOG_DIR/debug-log.jsonl`,
else `./output/log/debug-log.jsonl`, or `--gemmini-debug-log`), one
`{"layer","msg"}` record per event through `log.hpp`; nothing routine is
printed to stdout/stderr. Layers `potal.backend` (`[POTAL_BACKEND] registered`,
written with the first `device_init`), `potal.issuer`, `potal.tlb`,
`potal.matmul`, `potal.supports_op`, `potal.summary`, and the resolved matmul
layer for HP1 per-node records. Per node `node=... op=MUL_MAT src0_type=i8 ... M N K`,
`[POTAL_MATMUL] pack_A_bytes/pack_B_bytes`, `scale_base/generation`,
`dispatch=HARDWARE issue_status=POTAL_OK` (printed only after the issue
returned `POTAL_OK`), `fence_status=POTAL_OK`, `scale_release=POTAL_OK`,
`output_type=F32`, `[POTAL_BACKEND] hw_gemm_count`; errors as
`dispatch=FAILED stage=... status=...` (the node still fails; with `LOG_DEBUG=0`
failure records go to the GGML error log instead); `supports_op` decisions for
MUL_MAT. `ggml_backend_potal_get_stats()` returns per-backend counters (nodes,
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
Paired Main/Residual execution, DIM32/64, dynamic backend loading, concurrent
processes. The stripe pipeline (STRIPE_PIPELINE) has its own section below.

## Q8_HP1 path: configured activation quantization, routes, compensation

`GGML_OP_MUL_MAT` with `src0` `GGML_TYPE_Q8_HP1` (`ne = {K, N}`, contiguous)
and `src1` F32 (`ne = {K, M}`, contiguous), 2-D, any M and N, `K % 32 == 0`.
Four separate notions: the *activation granularity* (the quantizer's
metadata), the *weight granularity* (HP1: one carrier per 32-wide K block and
column, `block`), the *weight format* (Q8_HP1 payload/scale contract) and the
*hardware tile* (SPAD/ACC capacity, `potal_tile_select`).

```
matmul options          ggml::gemmini::resolve_matmul_options() (mode, rmd backend, stripe capacity)
tile plan               potal_tile_select() -> args.tile_I/J/K, activation_rows_per_stripe
src1 F32 --ggml::gemmini::quants::quantize_activation--> A int8 [M][K] + act::Meta (+ residuals if RMD)
                         GGML_GEMMINI_ACTIVATION_QUANT: EXSIA | TENSOR | TOKEN | BLOCK | STRIPE
src0 HP1 --weight reader (HP1 SCU route)--> B int8 [K][N], carrier[K/32][N], channel_scale[N]
route = activation metadata x block weight
  Tensor/Token/Stripe/ExSIA+block: one potal_tiled_matmul_auto over K,
      dst[m][n] = C[m][n] * row_scale[m] * channel_scale[n]
  Block+block: one potal_tiled_matmul_auto per 32-wide K group,
      dst[m][n] = sum_g C_g[m][n] * scale[m][g] * channel_scale[n]
  per LOOP_WS invocation: SCALE_WRITE every (block, column) carrier -> LOOP_WS
                          -> POTAL_FENCE -> SCALE_RELEASE        (hart-0 issuer)
compensation (GGML_GEMMINI_ENABLE_RMD=ON, cpu_direct): residual::execute_direct_stripe
      + rmd::merge_rmd_correction into dst, as in the Gemmini dense path
```

The quantizers, metadata, weight reader and residual code are the Gemmini
files themselves (no copies): all five algorithms keep their own metadata,
reset and failure handling, and none is replaced by TENSOR. The row scale of
the constant-along-K routes is `act::activation_scales()` (tensor scale, token
scale, stripe scale, ExSIA `2^theta`); BLOCK changes scale along K, so it is
never post-scaled after a whole-K accumulation.

Compensation keeps the Gemmini meaning. OFF (`GGML_GEMMINI_ENABLE_RMD=OFF`) is
the original OFF build: no outlier selection, no residual capture, the scale
from all finite values, no correction work at all. ON captures outlier
residuals in the quantizer and corrects dst on the CPU (`cpu_direct`, the
`GGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU` route) in FULL mode. The WS residual
route (`gemmini_ws_compact`) never uses the Gemmini WS executor
(`rmd-executor.cpp`, gemmini.h/IM2P): in STRIPE_PIPELINE mode it runs as a
PoTal Residual GEMM (`potal_ws`, next section); in FULL mode it is refused
(configure fails for `DEFAULT_RMD_BACKEND=WS` with a FULL default, a FULL node
with `GEMMINI_RMD_BACKEND=WS` at run time fails, a ws_packet payload reaching
FULL compensation fails), never silently dropped.

Tile selection is `tiled_matmul_auto`'s WS branch (start from the `db_` sizes,
grow J, I, K while A+B fit half the SPAD and C half the ACC), plus the PoTal
limits: `tile_K` starts and grows in whole scale blocks (`passes_per_block =
32 / DIM`), `tile_I * tile_J <= 64` work IDs, and `(tile_K / passes_per_block)
* tile_J <= 256` scale slots. Where these limits do not bind (and
`passes_per_block = 1`) the choice equals gemmini.h's (`potal-tiling-unit-test`
section A). Supported request subset: WS, no D/bias, no activation, identity
scale factors, no transpose, `full_C`; anything else is
`POTAL_ERR_UNSUPPORTED` before a command. Operand reuse (`a/b_spad_id`) and the
overlap of consecutive invocations are not used (every invocation is fenced and
releases its scale slots before the next one writes).

Q8_HP1 validation: the Gemmini `wreader::validate` / block readers (storage
contract, zero padding, finite `channel_scale`, `m` in `0..32767` or
`INT16_MIN` = zero carrier), plus one bit-identical, non-negative
`channel_scale` per row (a row of 0 only with all blocks zero). The HP1
exponent is applied only by the SCU, never again in reconstruction, and
weights are never dequantized and requantized.

Integer contract (golden `reference_gemm_i32`): for each output, K passes of
DIM elements in order; `partial << carrier` with Sat32 (PoTalSCU), first pass
overwrites, later passes Sat32-add (AccPipe).

DMA buffers and TLB: each node allocates its own staging buffers; before each
tiled job the issuer issues TLB FLUSH + FENCE (no DMA is in flight then, every
earlier job ended with a FENCE), so a reallocated virtual page never reuses a
stale accelerator translation. A failure after a command was sent disables the
issuer (fail-stop) and leaks that job's DMA buffers on purpose.

Logging uses the existing `ggml-gemmini-utils` system (no PoTal logger, no new
schema): `log::debug` JSONL (`LOG_DEBUG=1`, `debug-log.jsonl` under
`GEMMINI_LOG_DIR`), per HP1 node with the caller's trace context
(`gemmini_trace_operator`, bound on the issuer thread for the node's jobs):

| Record | Proves |
|---|---|
| `[POTAL] config backend=POTAL node= ... weight_type=Q8_HP1 weight_granularity=block activation_quant= compensation= rmd_backend= M= N= K= rows=prefill_shaped\|decode_shaped params_sha256=` | the configuration this node ran with |
| `[POTAL] route=Tensor+block activation_meta=Tensor dispatcher=potal_tiled_matmul_auto ... shift_carrier_count=` | the route chosen from the metadata the quantizer actually produced; HP1 carrier statistics |
| `[POTAL] route= ... jobs= k_groups= tile_I/J/K I0/J0/K0 pads spad/acc rows scale_slots generation= tlb_flushes= loop_ws_issued= fence_ok= scale_writes= scale_releases= status=` | the hardware work actually issued and fenced |
| `[POTAL] weight= weight_cache=HIT\|MISS\|BYPASS\|UNCACHED validation=performed\|skipped pack=performed\|skipped prepare_ms= alloc_ms= lookup_ms= weight_bytes= cache_entries= cache_bytes= cache_budget=` | whether this node validated and unpacked its weight or reused the prepared one |
| `[POTAL] compensation= residual_payloads= corrections_merged= residual_events=` | what the compensation step actually did (OFF: all 0) |
| `[POTAL] node= route= dispatch=HARDWARE fence=POTAL_OK completed=` | completion |
| `[POTAL] supports_op_reject op=MUL_MAT ...` | a `supports_op` answer, not a CPU execution |

## Stripe pipeline (`STRIPE_PIPELINE`, ExSIA, `matmul/stripe.hpp`)

`GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE` (or `GEMMINI_MATMUL_MODE`
at run time) with `GGML_GEMMINI_ACTIVATION_QUANT=EXSIA`. The Main tile plan is
selected once, before ExSIA starts (`potal_tile_select`; its `tile_I * DIM` is
the stripe height). The calling thread quantizes; at every stripe's folding
commit ExSIA calls the stripe sink, which copies the stripe's codes, post-fold
theta and residual packet into one of 1..2 run-owned slots and hands it to the
hart-0 issuer at once, then ExSIA goes on with the next stripe. The issuer
runs, per stripe and in order, the Main GEMM (frozen plan), with a packet the
Residual GEMM (WS residual on PoTal, `potal_ws`), then on its CPU the F32 Main
reconstruction, the radix INT64 reconstruction and the checked merge; only then
the slot is free again. dst is written once, from the run's private output,
after every stripe completed. RMD OFF runs the same pipeline with Main GEMMs
only (the control). Each planned GEMM runs as the issuer's owner step
(`run_planned_stripe_gemm`: preflight, TLB FLUSH + FENCE, then the execution of
the C API `potal_tiled_matmul_stripe()`); a descriptor the preflight refuses
issues no command. The residual merge uses the prepared weight's channel scales
(no per-stripe HP1 revalidation), and every `[POTAL_STRIPE] event=` stage record
carries the run's `event_sequence`, written in that order.

```
producer (graph_compute thread)          consumer (hart-0 issuer)
ExSIA stripe s folding commit -> submit  Main(s) -> Residual(s) -> reconstruct + merge(s) -> slot free
ExSIA stripe s+1 ... (waits if 2 held)   Main(s+1) ...
finish -> commit private output to dst
```

Supported: STRIPE_PIPELINE + EXSIA + RMD ON + WS, and STRIPE_PIPELINE + EXSIA +
RMD OFF; job capacity 1 or 2. Refused (configure or per node, never replaced):
STRIPE_PIPELINE with another activation algorithm or with the CPU residual,
other capacities. Details, numerics, logs and commands:
[docs/potal-stripe-api.md](docs/potal-stripe-api.md).

### Prepared weights (`matmul/weight_cache.hpp`)

A Q8_HP1 weight is validated and unpacked into B / carriers / channel_scale
once per backend instance and reused by every later MUL_MAT of it, so decode
steps do no O(N x K) weight work. Only a weight in a
`GGML_BACKEND_BUFFER_USAGE_WEIGHTS` buffer is cached (llama.cpp sets it on
every model buffer; the model outlives the context and its backends). The key
is buffer, data address (a view's offset included), type, ne and nb. Anything
else (compute buffers, tensors without a buffer, e.g. the smoke tests) is
prepared per call as before (`UNCACHED`). A weight that fails validation is
never cached. `GGML_POTAL_WEIGHT_CACHE_MB` (default 1024, 0 = off) bounds the
cache; a weight that does not fit is prepared per call (`BYPASS`), nothing is
ever evicted while the backend lives. Entries are freed with the backend or by
`ggml_backend_potal_clear_weight_cache()`, which is required before a weight
buffer is mutated or freed while the backend lives (a hit never re-reads the
weight) and must not overlap a graph_compute. After a fatal hardware failure
the entry is abandoned (leaked), never freed or reused. `weight_pack_s` counts
only nodes that prepared their weight; the `[POTAL] summary` adds
`weight_cache_hits/misses/bypasses/uncached/entries/bytes`,
`weight_validations`, `weight_packs`, `weight_pack_bytes`, `weight_alloc_s` and
`weight_lookup_s`. The first preparation reads the validated blocks' `qs`
directly in column tiles instead of one checked reader call per element.
`potal-weight-prep-bench [--gguf model.gguf]` measures both on the host.

`log::cycle` intervals with `LOG_CYCLE=1`: `potal.act_quant`,
`potal.weight_pack_hp1` (preparing nodes only), `potal.route_call` (caller side: submission, issuer
wait and reconstruction, not SCU time), `potal.compensation`, each with the
node's semantic identity (phase, graph occurrence, ordinal) and source role.
Every completed node adds a performance NPU record `POTAL/dense/work_total_cycles`
with `collected=false reason=unavailable_device_counter` (PoTal has no cycle
counter; none is derived from CPU `rdcycle`, LOOP_WS or FENCE counts) and marks
the phase's CPU work wall incomplete (its host stages are not CPU-resource
samples). In the semantic graph, POTAL nodes are `TARGET_NPU`; a GGML_POTAL
build is never a CPU-only (FULL_CPU) build. llama passes the model architecture
to POTAL as to Gemmini, so layers resolve (`blk.N.attn_qkv`, `lm_head`) and
`model_arch=unknown` marks a direct caller without one. At backend free one
`potal.summary` record gives the stats: per-route completions,
prefill/decode-shaped node counts, jobs, LOOP_WS, fences, scale writes/
releases, TLB flushes, zero/shift carriers, compensation counters and
`supports_op_mul_mat_rejected_answers` (answers, not executions). Which backend
the scheduler assigned is printed by ggml itself with `GGML_SCHED_DEBUG=2`.
Guest seconds are not FPGA cycles. The CPU backend does not run Q8_HP1
MUL_MAT, so an HP1 GEMM is never silently computed on the CPU.

| Binary | Runs on | Checks | Final line |
|---|---|---|---|
| `potal-tiling-unit-test` | host and target (emulator, no accelerator) | tile choice vs gemmini.h, tiled execution on an RTL-semantics emulator == integer golden (multi-tile, K chunks, padding, zero carriers, Sat32), scale release, fail-stop, HP1 validation, the configured dispatcher (metadata kind, route, never TENSOR in its place), all five algorithms through their own quantizer and route vs a golden from the raw metadata and vs their own dequantization, compensation as configured, generation wrap, GGML contract | `POTAL_TILING_UNIT_TEST PASS` |
| `potal-tiling-unit-test` (W) | host and target | prepared-weight cache: MISS then HIT, operands == the reader's per-element values, cached dst == per-call dst bitwise for all five algorithms, distinct weights and views, uncached/bypass/budget, invalid weights never cached, mutation only after clear(), concurrent first use, fatal abandon through the backend | (same) |
| `potal-hp1-smoke` | FireSim (RISC-V Linux) | F32 x Q8_HP1 graphs on the POTAL backend (configured route), every element == CPU golden of the same route, invocations fenced, scales released, one TLB flush per job, compensation counters; GPT-2 qkv/lm_head shapes; 2-layer chain; DMA reallocation; generation wrap asserted from the totals | `POTAL_HP1_SMOKE PASS` |
| `potal-stripe-api-test` | host and target (emulator) | C11 caller of `potal_tiled_matmul_stripe` (header, link, layout), frozen plans (tail lowers tile_i only, N tail, K chunks) == integer reference, no tile search while executing, refused descriptors issue nothing, generation wrap, fatal session, shared generations with the legacy API | `POTAL_STRIPE_API_TEST PASS` |
| `potal-residual-unit-test` | host and target (emulator) | PoTal WS residual: run-aware request, K32 padded lowering == compact-run golden (INT32 bit-exact) for run lengths 1/15/16/17/31/32, multi-run, carry lanes, zero/large carriers, saturation, Sat32 cancellation order; INT64 reconstruction and checked merge == golden; failure cases | `POTAL_RESIDUAL_UNIT_TEST PASS` |
| `potal-stripe-pipeline-test` | host and target (emulator thread) | real ExSIA events through the production run/driver: overlap and backpressure gates, semantic capacity, serialized == pipelined == golden, immutable accepted stripes, failure wake-up, no partial commit, repeated runs, producer placement | `POTAL_STRIPE_PIPELINE_TEST PASS` |
| `potal-stripe-smoke` | FireSim (RISC-V Linux) | STRIPE_PIPELINE graphs on the POTAL backend (Main and Residual GEMMs on the hart-0 issuer) == CPU golden from the same ExSIA codes/theta/packets; stripes, packets, merges, fences, TLB flushes; generation wrap | `POTAL_STRIPE_PIPELINE_SMOKE PASS` |

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

# GGML backend, host. The HP1 validation configuration is spelled out; plain
# cmake defaults differ (GGML_GEMMINI_ACTIVATION_QUANT=EXSIA, GGML_GEMMINI_ENABLE_RMD=ON).
POTAL_OPTS=(-DGGML_GEMMINI=OFF -DGGML_POTAL=ON -DPOTAL_API_ONLY=OFF -DGGML_BACKEND_DL=OFF -DLLAMA_CURL=OFF
  -DPOTAL_EXPECTED_DIM=16 -DPOTAL_HW_PARAMS:FILEPATH="$PARAMS/gemmini_potal_params.h"
  -DGGML_GEMMINI_ACTIVATION_QUANT=TENSOR -DGGML_GEMMINI_ENABLE_RMD=OFF -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU
  -DGGML_GEMMINI_ENABLE_OPENMP=OFF -DLOG_DEBUG=1)
cmake -S . -B build-potal-ggml-host "${POTAL_OPTS[@]}"
cmake --build build-potal-ggml-host --target potal-api-test potal-ggml-unit-test potal-tiling-unit-test -j
build-potal-ggml-host/bin/potal-ggml-unit-test     # expects "POTAL_GGML_UNIT_TEST PASS"
build-potal-ggml-host/bin/potal-tiling-unit-test   # expects "POTAL_TILING_UNIT_TEST PASS"
# Other selections build and test the same way (one build directory each), e.g.
#   -DGGML_GEMMINI_ACTIVATION_QUANT=BLOCK            (Block+block route)
#   -DGGML_GEMMINI_ENABLE_RMD=ON                     (compensation ON, cpu_direct)

# Stripe pipeline: its own build directories (commands in docs/potal-stripe-api.md).

# GGML backend, RISC-V Linux cross build (build-riscv.sh defaults to the same
# selection; the options are spelled out anyway)
BUILD_DIR=build-potal-ggml-d16 ./build-riscv.sh "${POTAL_OPTS[@]}" -DLOG_CYCLE=1 \
  -DPOTAL_HW_CONFIG=FireSimQuadRocketSinglePoTalD16Config

# Guest (FireSim Linux). host-init.sh copies the whole software/workspace tree
# to /root/workspace, so the build and the GGUF are at:
B=/root/workspace/llama.cpp-potal/build-potal-ggml-d16/bin
MODEL=/root/workspace/llama.cpp-potal/models/gpt2/gpt2.Q8_HP1.gguf
cd "$B"
./potal-ggml-smoke --case=k16 --verbose; ./potal-ggml-smoke --case=k32 --verbose; ./potal-ggml-smoke --case=sched
./potal-hp1-smoke --quick && ./potal-hp1-smoke      # expects "POTAL_HP1_SMOKE PASS"

# End to end: short prompt, prefill + decode. Exit status kept separately from the logs.
export GEMMINI_LOG_DIR=/root/potal-log-gpt2; mkdir -p "$GEMMINI_LOG_DIR"
GGML_SCHED_DEBUG=2 ./llama-cli -m "$MODEL" -p "The capital of France is" -n 8 -no-cnv --no-warmup \
  --temp 0 -s 1 -t 3 > "$GEMMINI_LOG_DIR/llama-cli.out" 2> "$GEMMINI_LOG_DIR/llama-cli.err"
echo "llama-cli exit=$?"
```

A run counts as PoTal end to end only if `llama-cli` exits 0, the
`potal.summary` record (`[POTAL] summary`) in the debug log shows `hp1_failed=0`, `hp1_completed > 0` with every
completion on `route_tensor_block`, `prefill_shaped > 0` and `decode_shaped >
0`, `residual_payloads=0 compensation_executed=0` (compensation OFF), `fences
== loop_ws_issued`, `scale_writes == scale_releases`, `tlb_flushes ==
tiled_jobs`, and the debug log has `dispatch=HARDWARE` records (including the
lm_head, `result_output`) and no `dispatch=FAILED`.
