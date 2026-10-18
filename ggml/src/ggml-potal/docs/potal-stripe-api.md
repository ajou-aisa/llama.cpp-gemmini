# PoTal stripe pipeline and the planned stripe GEMM API

This document covers the `STRIPE_PIPELINE` execution mode of the GGML PoTal
backend (`matmul/stripe.hpp`), the PoTal WS residual
(`residual/stripe-residual.hpp`) and the C API it executes through
(`potal_tiled_matmul_stripe`, `include/potal.h`, library `potal-tiled-api`).
The behavioral contract follows the IM2P.sim frontend PIPELINE mode
(`docs/HOST_STRIPE_CONTRACT.md`, `docs/NUMERICAL_CONTRACT.md`,
`frontend/src/im2p_gemmini_frontend.cpp`), mapped onto one physical PoTal and
its one issuer thread. Nothing of IM2P.sim is built or linked.

## 1. What runs where

```
calling thread (producer)                hart-0 issuer thread (consumer)
------------------------------------     -----------------------------------------------
potal_tile_select (Main plan, once)
issuer start (bring-up under the
caller's affinity), producer pin
ExSIA (SEQUENTIAL) stripe 0
  folding + residual packet seal
  -> on_ready -> submit(0)  ----------->  Main GEMM(0)      PoTal RoCC (frozen Main plan)
ExSIA stripe 1                            Residual GEMM(0)  PoTal RoCC (packet -> K32 lowering)
  -> submit(1)  ----------------------->  Main F32 reconstruction(0)        CPU
ExSIA stripe 2                            radix INT64 reconstruction(0)      CPU
  -> submit(2): 2 stripes held, waits     checked merge(0)                   CPU
                                          SEMANTIC_DONE(0): slot free  ---> submit(2) accepted
...                                       Main GEMM(1) ...
finish: all stripes semantically done
commit private output -> GGML dst
```

* Producer: the thread that runs the GGML node; ExSIA's internal mode
  (SEQUENTIAL in the reference configuration) is unchanged. After the issuer
  is up it is placed, for the node only, beside the issuer
  (`producer_affinity`): with SEQUENTIAL on one CPU of its own mask other than
  the issuer's; with the OpenMP ExSIA modes on its whole mask without the
  issuer's CPU, because an OpenMP team started inside the window inherits the
  narrowed mask (one CPU would serialize ExSIA's workers for the rest of the
  process). The caller's mask is restored on every return path; a mask
  without another CPU is refused.
* Consumer: the existing hart-0 issuer (`execution.cpp`), the only thread that
  issues PoTal RoCC commands. It runs Main and Residual GEMMs of a stripe one
  after the other, then reconstructs and merges on its CPU. No second issuer,
  no paired Main/Residual GEMM, no new opcode.
* Order on the accelerator: Main(s) -> Residual(s) -> reconstruction/merge(s)
  -> Main(s+1). Quantization of later stripes overlaps all of it.

Not part of this mode: stage worker pools, task stealing, overlapping
Merge(s) with Main(s+1), Main/Residual paired GEMMs, scratchpad/ACC changes.

## 2. Configuration

| Setting | Values |
|---|---|
| `GGML_GEMMINI_DEFAULT_MATMUL_MODE` / `GEMMINI_MATMUL_MODE` | `STRIPE_PIPELINE` (`FULL` keeps the previous path unchanged) |
| `GGML_GEMMINI_ACTIVATION_QUANT` | `EXSIA` (required) |
| `GGML_GEMMINI_ENABLE_RMD` | `ON`: Residual GEMMs (`potal_ws`); `OFF`: Main only (control) |
| `GGML_GEMMINI_DEFAULT_RMD_BACKEND` / `GEMMINI_RMD_BACKEND` | `WS` with RMD ON (enum `gemmini_ws_compact`; executed by PoTal, logged `POTAL/potal_ws`) |
| `GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY` / `GEMMINI_STRIPE_JOB_CAPACITY` | `2` (default) or `1` (serialized control) |
| `GGML_GEMMINI_ENABLE_STRIPE_MATMUL`, `..._STRIPE_PIPELINE` | `ON` |
| `GGML_GEMMINI_EXSIA_DEFAULT_MODE` | `SEQUENTIAL` (OpenMP modes are kept; their events go through the same thread-safe sink) |

Refused, never replaced: STRIPE_PIPELINE with another activation algorithm,
STRIPE_PIPELINE with RMD ON and the CPU residual, a capacity other than 1/2,
FULL with RMD ON and WS. Build defaults are checked by CMake
(`ggml-potal/CMakeLists.txt`); run-time overrides per node
(`options::unsupported_combination`), failing the node (`GGML_STATUS_FAILED`,
dst NaN).

## 3. C API (`include/potal.h`, library `potal-tiled-api`)

```c
typedef struct potal_stripe_session potal_stripe_session; /* opaque */

typedef enum potal_stripe_work_kind {
    POTAL_STRIPE_WORK_MAIN     = 0,
    POTAL_STRIPE_WORK_RESIDUAL = 1,
} potal_stripe_work_kind;

typedef struct potal_stripe_gemm_desc {
    uint64_t run_id, stripe_id, row_begin; /* identity only */
    uint32_t work_kind;                     /* identity only */
    uint32_t m, n, k;
    const int8_t *a; const int8_t *b; int32_t *c;
    size_t lda, ldb, ldc;                   /* element strides */
    const uint32_t *carriers; size_t carrier_stride; /* uint32 elements */
    uint32_t scale_block_k;                 /* 32 here */
    uint32_t tile_i, tile_j, tile_k;        /* frozen plan, DIM tiles */
} potal_stripe_gemm_desc;

typedef struct potal_stripe_gemm_report {
    potal_status status; const char *stage; int fatal;
    uint64_t loop_ws, validated_gemms, fences, scale_writes, scale_releases;
    uint32_t last_generation;
    uint32_t tile_i, tile_j, tile_k, i0, j0, k0, pad_i, pad_j, pad_k;
    double scale_s, issue_s, fence_s;
} potal_stripe_gemm_report;

potal_status potal_stripe_session_create(potal_stripe_session **out);
void         potal_stripe_session_destroy(potal_stripe_session *session);
potal_status potal_tiled_matmul_stripe(potal_stripe_session *session,
                                       const potal_stripe_gemm_desc *desc,
                                       potal_stripe_gemm_report *report);
```

* One call = one logical GEMM, `C = Sat32 sum_p SCU(A_p . B_p, carrier)`, on
  the Gemmini-compatible tiler (`potal_tiled_matmul -> outer -> sp_tiled_matmul_ws
  -> potal_loop_ws_issue / potal_gemm_ws_issue`). Return POTAL_OK = every
  LOOP_WS issued, fenced and its scale entries released (C complete); the
  low-level issue functions keep "POTAL_OK = issued".
* No tile search: `tile_j`/`tile_k` as given, `tile_i` limited to
  `ceil(m / DIM)` (a tail stripe). `potal_tile_select_count()` does not move.
* Refused before the first command (report.loop_ws == 0): NULL pointers,
  `work_kind`, zero sizes, strides, `scale_block_k`, tiles vs shape/capacity
  (SPAD/ACC halves, WorkIdsPerLoop, scale window), A/B DIM-byte and C DIM*4-byte
  alignment, operand extents beyond the 32-bit DRAM offset, C overlapping A/B,
  any invalid carrier. Bias/D, transposes, activations and scale factors
  cannot be expressed.
* `a` points at the stripe's first row; `row_begin` is not added again. `k`
  is the extent handed over (the Residual's padded execution K).
* Session: created without RoCC or threads (`*out = NULL` on failure,
  `POTAL_ERR_CAPACITY` if out of host memory). Not thread-safe; its owner is
  pinned to the hart-0 CPU. The issuer owns exactly one session for the
  process, shared by the I8 GEMM, the FULL tiled jobs and the stripe GEMMs: one
  8-bit generation sequence (1..255, wrapping) and one set of counters. After a
  fatal report the session refuses all work (`stage="session_disabled"`) and
  the operands of that GEMM must never be freed. Destroy only when no call runs.
* TLB: the caller's job. The issuer flushes (FLUSH + FENCE) before every
  planned GEMM, since stripe slots and residual operands may be fresh pages.
* C callers link `potal-tiled-api` (it contains C++); `potal-api` (potal.c)
  stays the C11 low-level API and the bare-metal build is unchanged.

## 4. Stripe runtime (`matmul/stripe.hpp`)

Slot states (every state but FREE holds one unit of capacity):
`FREE -> RESERVED -> READY -> MAIN_RUNNING -> MAIN_DONE -> RESIDUAL_PREPARE ->
RESIDUAL_RUNNING -> RESIDUAL_DONE -> RECONSTRUCT_MERGE -> SEMANTIC_DONE -> FREE`;
without a packet `MAIN_DONE -> RECONSTRUCT_MERGE`.

* `submit` (ExSIA `on_ready`): checks the post-fold theta (finite positive
  `2^theta`), no cpu_direct payload, the packet's identity/shape, the run id
  (bound by the first stripe, 0 is valid), stripe order, row range and the
  frozen stripe height; waits on a condition variable while `capacity` stripes
  are held (mutex released; `queue_wait` record); reserves a slot, copies the
  stripe's A8 codes into the slot's DMA buffer, stores the theta snapshot, the
  packet handle, timings and the trace context; publishes READY; returns
  true. false only for a failed run: a full pipeline never fails a stripe.
* Ownership: the slot copy, not ExSIA's workspace or `args.A` (ExSIA reuses
  its slot, and its failure path zero-fills `args.A`). The consumer never reads
  the producer's arguments, metadata or ExSIA state; the merge uses a separate
  view (`init_merge_args`: weight view + private output).
* Capacity returns after SEMANTIC_DONE only: Main raw completion and Residual
  completion do not release it. Pool slot index, stripe id, slot reuse
  generation and the SCU generation are different numbers in the records.
* `finish`: closes the input, fails on missing stripes, waits for all stripes
  or the first failure. `abort`: sticky failure + close. `poll_completed`
  returns the completion records (stripe, rows, slot, slot generation,
  residual rows/K, nonzero corrections, CPU times of READY, Main start and
  SEMANTIC_DONE) added since the last poll; the driver puts them in the run's
  report.
* Failure: the first error is kept; producer and finish waiters wake, no
  stripe starts, the private output is dropped; ops.cpp keeps its
  GGML_STATUS_FAILED + NaN dst policy. A software/metadata failure (refused
  descriptor, bad packet, overflow, non-finite merge) leaves the accelerator
  drained and the buffers freeable; a fatal one (hardware state unknown) makes
  the issuer disable itself and the run abandon its slot, residual and weight
  buffers. An abort cannot interrupt a running GEMM: POTAL_FENCE blocks in
  hardware until the GEMM drained, and only then does the issuer see it.
* dst: written once, from the private output, after every stripe completed.

## 5. Numerics

Main: `C_main = PoTal(A_stripe, B, carriers)` with the frozen plan;
`main_f32 = float(double(C_main) * double(float(2^theta_s)) * double(channel_scale[n]))`,
the FULL route's formula (`reconstruct_row`), into the stripe's rows of the
private output.

Residual request (IM2P run-aware meaning, rebuilt from the packet):
runs = the packet's original weight blocks in block order, each with its
selected block-local K ascending; rows = every (original lane, source row)
with a nonzero digit in any run, lane-major then row; a run without a digit
of that lane/row contributes zeros. B and carriers are gathered from the
prepared weight (weight cache), never from the GGUF again:
`B_r[32 r + kk][n] = B[block_r * 32 + local_k[kk]][n]`, `car_r[r][n] = carrier[block_r][n]`.

K32 padded lowering (PoTal LOOP_WS chooses one scale per scale block with one
`passes_per_block` per invocation): run r occupies 32 execution K, its c_r
compact K first, then 32 - c_r zero K in A_r and B_r. `K_compact = sum c_r`,
`K_execution = 32 * runs`, `scale_block_k = 32`, `passes_per_block = 2` at
DIM16. No DIM partial mixes runs; zero K and all-zero passes add
SCU(0) = 0. The residual plan is selected once, when the sealed packet becomes
a request, before its first command; the Main plan is never changed for it.
Compact K 31 is one logical GEMM with passes of 16 and 15 (+1 zero) K.
This is a padded representation chosen for PoTal's command set, not IM2P's
physical encoding; `residual_compact_k` and `residual_execution_k` record the
cost difference.

Golden of the Residual GEMM (`tests/support/stripe-golden.hpp`, no padding):
for each (lane, row) and n, runs in block order, fragments of at most DIM of
the compact K: `p = exact dot`, `q = Sat32(HP1_shift(p, carrier))`,
`acc = q`, then `acc = Sat32(acc + q)`. Summing per-run results on the CPU
instead changes saturating cases and is not done.

Reconstruction: `R[row][n] = sum C_r[lane, row][n] * 256^lane` in checked
`__int128`, INT64 range checked at the end (the A8 balanced radix 256, 5
lanes including the carry). C_r already went through the SCU and Sat32; the
HP1 exponent is never applied again. Tagged `BlockScaledInt64Correction`.

Merge: `rmd::detail::merge_rmd_correction_with_metadata` with an ExSIA Meta
holding only this stripe's theta: `delta = float(double(R) * double(channel_scale)
* double(float(2^theta)))`, `out = main_f32 + delta`; NaN/Inf, INT64 overflow or
missing metadata fail the stripe; all cells or none. An empty residual (no
packet) merges nothing and runs no Residual GEMM.

## 6. Records and counters

Debug JSONL (`LOG_DEBUG=1`, `$GEMMINI_LOG_DIR/debug-log.jsonl`), one record
per step, layer = the node's matmul layer, with run id, stripe, pool slot,
slot generation, rows, outstanding stripes, capacity, executor and the
operator identity (`phase=... request_id=... operator_id=...`; the issuer binds
the stripe's trace context):

```
[POTAL_STRIPE] event=quant_begin|folding_commit|queue_wait|submit_accepted|main_begin|main_end|
  residual_prepare_begin|residual_prepare_end|residual_begin|residual_end|
  radix_reconstruct_begin|radix_reconstruct_end|merge_begin|merge_end|semantic_complete|
  capacity_release|quant_end|run_commit|run_failed  run_id= stripe= slot= slot_generation= rows=a..b
  outstanding= capacity= executor=POTAL/potal_ws backend=HARDWARE|EMULATED t_ns= ...
submit_accepted adds: producer_slot= quantization_start_ns= quantization_end_ns= folding_commit_ns=
  (ExSIA's own values; 0 when its profiling is compiled out)
main_end/residual_end add: gemm_rows= N= loop_ws= validated_api= fences= scale_writes= scale_releases=
  generation= tile_i/j/k= i0/j0/k0= k_compact= k_execution= runs= status=POTAL_...
```

Per node: `[POTAL] config ... matmul_mode=STRIPE_PIPELINE residual_executor=POTAL/potal_ws job_capacity=`,
`[POTAL] route=ExSIA+block matmul_mode=STRIPE_PIPELINE ... stripe_count= stripes_accepted=
stripes_semantic_completed= max_outstanding_semantic= producer_capacity_waits= main_logical_gemms=
main_loop_ws= main_fences= residual_nonempty_packets= residual_logical_gemms= residual_loop_ws=
residual_fences= residual_compact_k= residual_execution_k= active_lane_rows= radix_reconstructions=
merges_completed= corrections_nonzero= scale_writes= scale_releases= tlb_flushes= generation= status=`
and the cycle interval `potal.stripe_pipeline` (caller side). The backend
summary at free and `ggml_backend_potal_stats` carry the same counters
(`stripe_pipeline_nodes`, `stripe_pipeline_failed`, ...); `hp1_completed`
still counts nodes. `t_ns` values are CPU steady-clock observations of
submission/completion, not device cycles; the NPU record stays
`unavailable_device_counter`. Logical GEMMs, LOOP_WS invocations, GEMM
fences and TLB flushes are separate counters.

## 7. Tests

| Binary | Where | What |
|---|---|---|
| `potal-stripe-api-test` | host/target, emulator | C11 caller + layout, frozen plans, tails, K chunks, refusals without commands, generation wrap, fatal session, shared generations |
| `potal-residual-unit-test` | host/target, emulator | run-aware request, K32 lowering == compact golden (1/15/16/17/31/32, multi-run, carry lanes, carriers, saturation, cancellation), INT64 reconstruction, merge, failures |
| `potal-stripe-pipeline-test` | host/target, emulator thread | real ExSIA events, overlap/backpressure gates, capacity, serialized == pipelined == golden, immutability, failure wake-up, no partial commit, repeated runs, producer placement |
| `potal-stripe-smoke` | FireSim | backend graphs, Main/Residual on the hart-0 issuer, == CPU golden |

The host tests run the production code with an RTL-semantics emulator in
place of the accelerator (backend=EMULATED); a gate in the emulated runner
makes the overlap order deterministic. They show what the software lets
overlap; they are not an NPU overlap measurement.

## 8. Commands

Host build and tests (new build directory; the existing ones keep their
settings):

```bash
LLAMA=~/firesim/workspace/potal/software/workspace/llama.cpp-potal
PARAMS=~/firesim/workspace/potal/software/gemmini-rocc-tests/include/gemmini_potal_params.h
cd "$LLAMA"
STRIPE_OPTS=(-DGGML_POTAL=ON -DGGML_GEMMINI=OFF -DPOTAL_API_ONLY=OFF -DGGML_BACKEND_DL=OFF
  -DPOTAL_EXPECTED_DIM=16 -DPOTAL_HW_PARAMS:FILEPATH="$PARAMS"
  -DGGML_GEMMINI_ACTIVATION_BITS=8 -DGGML_GEMMINI_WEIGHT_BITS=8 -DGGML_GEMMINI_BLOCK_SIZE=32
  -DGGML_GEMMINI_ACTIVATION_QUANT=EXSIA -DGGML_GEMMINI_ENABLE_RMD=ON -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=WS
  -DGGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE -DGGML_GEMMINI_ENABLE_STRIPE_MATMUL=ON
  -DGGML_GEMMINI_ENABLE_STRIPE_PIPELINE=ON -DGGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY=2
  -DGGML_GEMMINI_EXSIA_DEFAULT_MODE=SEQUENTIAL -DGGML_OPENMP=OFF -DGGML_GEMMINI_ENABLE_OPENMP=OFF
  -DLOG_DEBUG=1 -DLOG_CYCLE=1 -DCYCLE_SIM=0 -DLLAMA_CURL=OFF)
cmake -S . -B build-potal-stripe-host "${STRIPE_OPTS[@]}"
cmake --build build-potal-stripe-host --target potal-stripe-api-test potal-residual-unit-test \
  potal-stripe-pipeline-test -j
for t in potal-stripe-api-test potal-residual-unit-test potal-stripe-pipeline-test; do
  build-potal-stripe-host/bin/$t; echo "$t exit=$?"
done
```

RISC-V cross build (does not run anything):

```bash
BUILD_DIR=build-potal-stripe-d16 ./build-riscv.sh "${STRIPE_OPTS[@]}" \
  -DPOTAL_HW_CONFIG=FireSimQuadRocketSinglePoTalD16Config
```

FireSim guest: see section 9.

## 9. FireSim procedure (user)

Build directories (host side, all under `llama.cpp-potal/`):

| Directory | Selection | Use |
|---|---|---|
| `build-potal-stripe-d16` | EXSIA, RMD ON, WS, STRIPE_PIPELINE, capacity 2, SEQUENTIAL, LOG_DEBUG=1, LOG_CYCLE=1 | stripe tests and smoke, GPT-2 STRIPE_PIPELINE runs |
| `build-potal-full-d16` | build-riscv.sh defaults: TENSOR, RMD OFF, FULL, LOG_DEBUG=1, LOG_CYCLE=1 | I8 / HP1 FULL regression with this code |
| `build-potal-ggml-d16`, `build-riscv` | earlier builds | not used here: do not mix their `bin/` with the two above |

ELF facts of these builds (`riscv64-unknown-linux-gnu-readelf -d`): every
test binary and `llama-cli` has `DT_RPATH=$ORIGIN` (searched before
`LD_LIBRARY_PATH`), so it loads `libggml.so`, `libggml-base.so`,
`libggml-cpu.so`, `libggml-potal.so`, `libggml-gemmini-utils.so` (and
`libllama.so`) from its own `bin/`. From the rootfs it needs `libstdc++.so.6`,
`libgcc_s.so.1`, `libatomic.so.1`, `libm.so.6`, `libpthread.so.0`, `libc.so.6`
(the same set as the earlier `build-potal-ggml-d16`; if `libatomic.so.1` is
missing, take it from `~/firesim/.conda-env/riscv-tools/sysroot/lib/`).
`potal-stripe-api-test` needs only the C/C++ runtime.

1. Deploy as before: `~/firesim/workspace/update_rootfs_potal.sh` (marshal
   build + install of `gemmini-tests-interactive.json`; `host-init.sh` copies
   `software/workspace` to `/root/workspace`), then boot the PoTal D16 FireSim
   target the usual way.
2. Guest, check the libraries each binary resolves:

```bash
S=/root/workspace/llama.cpp-potal/build-potal-stripe-d16/bin
F=/root/workspace/llama.cpp-potal/build-potal-full-d16/bin
MODEL=/root/workspace/llama.cpp-potal/models/gpt2/gpt2.Q8_HP1.gguf
unset LD_LIBRARY_PATH
/lib/ld-linux-riscv64-lp64d.so.1 --list "$S/potal-stripe-smoke"   # every libggml*.so must come from $S
/lib/ld-linux-riscv64-lp64d.so.1 --list "$S/llama-cli"
```

3. I8 and FULL HP1 regression (new code, the earlier validation selection):

```bash
cd "$F"
./potal-api-smoke; ./potal-api-smoke
./potal-ggml-smoke --case=k16 --verbose; ./potal-ggml-smoke --case=k32 --verbose; ./potal-ggml-smoke --case=sched
./potal-hp1-smoke --quick && ./potal-hp1-smoke                 # POTAL_HP1_SMOKE PASS
```

4. Stripe pipeline (hardware-free checks first, then the smoke):

```bash
cd "$S"
export GEMMINI_LOG_DIR=/root/potal-stripe-logs; mkdir -p "$GEMMINI_LOG_DIR"
./potal-stripe-api-test; echo "exit=$?"          # POTAL_STRIPE_API_TEST PASS (emulator)
./potal-residual-unit-test; echo "exit=$?"       # POTAL_RESIDUAL_UNIT_TEST PASS (emulator)
./potal-stripe-pipeline-test; echo "exit=$?"     # POTAL_STRIPE_PIPELINE_TEST PASS (emulator, threads)
./potal-stripe-smoke --quick; echo "exit=$?"     # hardware: quick
./potal-stripe-smoke; echo "exit=$?"             # hardware: full (tail, K chunks, runs/lanes, GPT-2 FFN, repeat, wrap)
```

   PASS needs exit 0 and, in this order of evidence: `POTAL_STRIPE_CASE PASS`
   for every case, `POTAL_RESIDUAL_GEMM_PASS`, `POTAL_RECONSTRUCT_MERGE_PASS`,
   `POTAL_STRIPE_GENERATION_WRAP PASS` (full), `POTAL_STRIPE_PIPELINE_SMOKE PASS`.
   In this build `potal-hp1-smoke` would run the pipeline (the default mode);
   to run it here in FULL mode use
   `GEMMINI_MATMUL_MODE=FULL GEMMINI_RMD_BACKEND=CPU ./potal-hp1-smoke`.

5. GPT-2 Q8_HP1, a prefill long enough for several stripes, then decode. With
   DIM16 the frozen Main plans of GPT-2's MUL_MATs use `tile_I = 5`, i.e.
   80-row stripes once M >= 80: a prompt of more than 160 tokens gives at least
   3 stripes per prefill node (check `stripe_count=` in the log, do not infer
   it from the word count).

```bash
cd "$S"
export GEMMINI_LOG_DIR=/root/potal-stripe-gpt2; mkdir -p "$GEMMINI_LOG_DIR"
cat > /root/prompt-long.txt <<'TXT'
The history of the river city begins in the early medieval period, when a small settlement grew around a ford
and the weekly market that formed beside it. Merchants from the hill villages brought wool, timber and salt,
and boatmen carried grain downstream to the coast. Over the following centuries the town expanded beyond its
first wooden palisade. A stone bridge replaced the ford, a cathedral rose on the northern bank, and the guilds
of weavers, smiths and brewers built halls around the main square. Plague and war interrupted this growth more
than once, yet each time the city recovered, rebuilding its walls and widening its streets. In the age of
steam the river was dredged, warehouses lined the quays, and a railway station connected the old center with
the industrial suburbs to the east. Today visitors walk along the restored embankments, read the inscriptions
on the guild halls, and climb the cathedral tower to see the whole valley, where the river still bends
around the place where the first travelers crossed it
TXT
GGML_SCHED_DEBUG=2 ./llama-cli -m "$MODEL" -f /root/prompt-long.txt -n 8 -no-cnv --no-warmup --temp 0 -s 1 -t 3 \
  > "$GEMMINI_LOG_DIR/llama-cli.out" 2> "$GEMMINI_LOG_DIR/llama-cli.err"
echo "llama-cli exit=$?"
```

6. Check the logs (on the guest or after copying `$GEMMINI_LOG_DIR` out):

```bash
D=$GEMMINI_LOG_DIR/debug-log.jsonl
grep -c '"\[POTAL\] route=ExSIA+block matmul_mode=STRIPE_PIPELINE' "$D"         # pipelined nodes
grep -o 'stripe_count=[0-9]*' "$D" | sort | uniq -c                              # need some >= 2 (prefill)
grep -o 'residual_loop_ws=[0-9]*' "$D" | sort | uniq -c | tail                   # need > 0 somewhere
grep '\[POTAL\] summary' "$D" | grep -o 'stripe_pipeline_failed=[0-9]*\|merges_completed=[0-9]*\|hp1_failed=[0-9]*\|residual_loop_ws=[0-9]*\|stripes_semantic_completed=[0-9]*'
grep -c 'dispatch=FAILED\|event=run_failed' "$D"                               # must be 0
grep 'event=queue_wait' "$D" | head -3                                            # backpressure actually happened
```

   A run counts only with `llama-cli exit=0`, `stripe_pipeline_failed=0`,
   `hp1_failed=0`, `merges_completed > 0`, `residual_loop_ws > 0`, at least
   one node with `stripe_count >= 2`, and no `dispatch=FAILED`/`run_failed`.
   Decode (M=1) nodes are one stripe each: a decode success alone shows no
   inter-stripe overlap; use the prefill nodes' `stripe_count` and their
   `[POTAL_STRIPE]` order (`submit_accepted` of stripe s+1 before
   `semantic_complete` of stripe s) for that. `t_ns` are CPU observations;
   there is no device cycle counter (`unavailable_device_counter`).
