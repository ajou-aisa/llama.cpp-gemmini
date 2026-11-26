# Paired Main/Residual GEMM — software contract

Status: **software model only.** `paired_tiled_matmul_auto()` (`matmul/paired.hpp`) runs the paired
execution on a functional emulator (`PAIRED_EMULATED`). The current PoTal RTL and FireSim bitstream have
no paired execution: a hardware request returns `PAIRED_HARDWARE_UNSUPPORTED` (`POTAL_ERR_UNSUPPORTED`,
stage `backend`) before any work, and never falls back to the serialized GEMMs. The hardware path stays
`STRIPE_PIPELINE` (`matmul/stripe.hpp`: Main GEMM, then Residual GEMM, per stripe). Every byte or
operation count below is a logical, emulated count, not a hardware cycle or FireSim DMA measurement.

This document is the contract the emulator implements and a later RTL implementation must follow.
Section E lists what the current RTL cannot express.

Sources: `matmul/paired.hpp` (types, API), `paired-plan.cpp` (plan, Residual descriptor),
`paired-schedule.cpp` (schedule, static validator), `paired-emulator.cpp` (functional model),
`paired.cpp` (`paired_tiled_matmul_auto`, reconstruction, merges). Tests: `tests/potal-paired-plan-test.cpp`,
`tests/potal-paired-schedule-test.cpp`, `tests/potal-paired-unit-test.cpp`.

## Flow

```text
paired_tiled_matmul_auto(desc)
  backend == hardware                      -> PAIRED_HARDWARE_UNSUPPORTED, nothing run
  validate request (shape, A, prepared weight)
  plan: paired_tile_select | paired_frozen_plan(desc.tile_I/J/K)   t_I, t_J, stripe height frozen
  stripes: desc.stripe_rows == plan.stripe_rows, one input per stripe
  scales: every stripe theta (finite 2^theta), every carrier (potal_carrier_is_valid)
  per stripe: build_paired_residual_request (packet -> runs, rows, limbs, fragments, groups)
  per stripe: build_paired_schedule -> validate_paired_schedule -> peak spill <= desc.spill_capacity_bytes
  per stripe: execute_paired_emulated (op by op)
  per stripe: radix INT64 reconstruction (checked __int128)
  merge (LEGACY_F32 | INT64)
  commit (the output is written only here)
```

Every check above runs before the first emulated op: a refused request reports `executed_ops == 0` and leaves
the caller's output untouched. A failure during execution, reconstruction or merge also leaves it untouched.

## A. Paired descriptor

### Main

`A` = Main A8 codes `[m][lda]` (ExSIA codes after folding), `B` = the prepared Q8_HP1 weight (`hp1_weight`:
`b [k][ldb]`, `carriers [k/32][cstride]`, `scale [n]`). One HP1 scale block = K32 = one carrier per
`(block, column)`; `passes_per_block = 32 / DIM` (2 at DIM16).

### Residual (one stripe)

From the sealed RMD packet, through `residual::decode_packet()` (the same decode `build_request()` uses, so
both paths see the same rows, runs and order):

| Item | Meaning |
|---|---|
| run `r` | one original weight block `block_id` (ascending) and its selected block-local K `local_k` (ascending) |
| compact K | concatenation of every run's `local_k`, run order; `k_compact = sum c_r` |
| lane row `i` | `(lane, source_row)` with a nonzero digit in any run, lane-major then source row |
| limbs | `int8 [rows][k_compact]` balanced radix-256 digits (`rmd::read_packet_digit`) |
| fragment | `count <= DIM` consecutive compact K of **one** run: `ceil(c_r / DIM)` fragments per run |
| group | `<= stripe_rows` consecutive lane rows; `nonzero[f]`: some digit of the group is nonzero in fragment `f` |

Original K of the x-th K of fragment `f`: `block_id * 32 + local_k[local_begin + x]`
(`paired_residual_request::original_k`). There is **no Residual B** and no gathered carrier array: the
Residual path reads the shared prepared weight by original K index and uses the carrier of its block.

Compact vs original K: a Residual DIM partial is the dot product of one fragment, i.e. at most DIM compact K of
one block. It is never zero-expanded to original K positions (that would regroup the DIM partials and change
the Sat32 result). Partials of two blocks are never mixed.

### Carrier selection

Main pass `p` (original K `[16p, 16p+16)` at DIM16) uses `carrier(p * DIM / 32, n)`; Residual fragment `f` uses
`carrier(f.block_id, n)`. Both read the same window scale context (B, `scale_load`).

### Output ACC mapping

| Path | ACC bank | Tile | Row address (from the bank base, DIM-row units) |
|---|---|---|---|
| Main | 0 | stripe rows x J group columns | `(i * j_run + j) * DIM + r` |
| Residual | 1 | one group's rows x J group columns | `(i * j_run + j) * DIM + r` (bank base `ACC_ROWS / 2`) |

`i` row tile, `j` N tile in the J group (`j_run = ceil(cols / DIM)`), `r` row inside the tile. ACC capacity per
bank: `POTAL_HW_ACC_ROWS / 2` = 512 rows = 32 KiB (the half `potal_tile_select` already uses; not halved again).

## B. Scratchpad map

Geometry from `potal_hw_params.h` (`POTAL_HW_BANK_NUM` 4, `POTAL_HW_BANK_ROWS` 4096, DIM 16: 256 KiB). Rows of
DIM int8. Pair `p` = window instance `u % 2`.

| Pair | Bank | Rows | Content |
|---|---|---|---|
| 0 (current/next) | 0 | `[0, 2048)` | Main activation |
| | 0 | `[2048, 4096)` | Residual limbs |
| | 1 | `[w_start, 8192)` | shared weight, rear-placed |
| 1 | 2 | `[8192, 10240)` | Main activation |
| | 2 | `[10240, 12288)` | Residual limbs |
| | 3 | `[w_start, 16384)` | shared weight, rear-placed |

* Main/Residual activation partitions are fixed 1:1; neither borrows the other's free half.
* Activation from the front, weight from the rear: LoopMatmul semantics (`a_addr_start = loop * half`,
  `b_addr_end = (loop + 1) * half`, `b_addr_start = b_addr_end - max_k * max_j * DIM`).
* Main A tile `(i, pass p)`: row `main_a_base + (i * passes + p) * DIM + r` (gemmini `a_addr`).
* Weight tile `(pass p, N tile j)`: rows `w_start + (p * j_run + j) * DIM + [0, DIM)`,
  `w_start = weight_end - passes * j_run * DIM` (gemmini `b_addr`).
* Residual limbs `(group row tile i, window fragment x)`: row `residual_a_base + (i * F_w + x) * DIM + r`,
  `F_w` = fragments of the window; columns `[0, count)` valid, the rest zero.
* **Indexed weight addressing**: original K `k` of the window `[k0, k1)` -> `rel = k - k0` ->
  row `w_start + ((rel / DIM) * j_run + j) * DIM + rel % DIM`. A K outside the resident window or outside the
  fragment's block fails.

Capacity per window (checked by the planner, the validator and the emulator):
`tile_I * tile_K * DIM <= 2048` (Main), group row tiles `* F_w * DIM <= 2048` (Residual; implied by the Main bound
since a group has `<= stripe_rows` rows and `F_w <= tile_K`), `tile_K * tile_J * DIM <= 4096` (weight),
`tile_I * tile_J * DIM <= 512` (each ACC bank), `tile_I * tile_J <= WorkIdsPerLoop`, window blocks x `tile_J`
`<= POTAL_SCALE_ENTRIES`.

Bank switching: per partition `EMPTY -> LOADING -> READY -> ACTIVE -> DRAINED -> EMPTY`.

* `window_begin(u)`: every loaded partition of pair `u % 2` READY -> ACTIVE (the Residual partition stays EMPTY
  in a window without Residual work); the other pair must not be ACTIVE.
* loads of instance `u + 1` go to the other pair while `u` is ACTIVE (prefetch); a load into the ACTIVE pair
  is refused, except a Residual group reload (below).
* Residual group switch inside a window: its partition ACTIVE -> DRAINED -> EMPTY -> LOADING -> READY -> ACTIVE.
* `window_end(u)`: ACTIVE -> DRAINED -> EMPTY; the pair's scale context is released.

### Plan

`paired_tile_select`: gemmini `tiled_matmul_auto` start (`sqrt` of one ACC bank's tiles, K from the Main
partition) and growth order J, I, K (K in whole K32 blocks) with the limits above.
`paired_frozen_plan`: the caller's Main tiles (e.g. the STRIPE_PIPELINE plan that fixed the ExSIA stripe height);
`tile_I` (limited to the I tiles of `m`) and `tile_J` are kept, only the common K window shrinks in K32 steps.
No K32 window fits: `POTAL_ERR_CAPACITY` (never a smaller `t_I`/`t_J`, never a serialized fallback).

Windows: blocks `[w * window_blocks, min(K/32, (w+1) * window_blocks))`, the last may be partial. Instance
`u = j_group * k_windows + w`.

## C. Paired micro-operation

A micro-tile is one stationary weight tile (DIM x DIM) streamed over the row tiles of its output:

| Path | Micro-tile | Rows streamed |
|---|---|---|
| MAIN | one DIM K pass x one N tile | the stripe's I tiles |
| RESIDUAL | one fragment x one N tile | the group's row tiles |

Fields (`paired_op`), the minimum a hardware descriptor needs:

| Field | Meaning |
|---|---|
| `path` | MAIN / RESIDUAL |
| `window`, `j_group`, `pair` | common K window, N tile group (ACC epoch = `j_group + 1`), SP pair |
| `block_id` | original weight block = carrier / scale block |
| `k_begin` / `fragment`, `k_count` | Main original K of the pass; Residual compact fragment and its K count |
| `j`, `cols` | N tile in the group and its valid columns |
| `group`, `rows`, `i_tiles` | Residual group (Main: none), valid rows, row tiles |
| `sp_a`, `sp_a_stride` | SP row of row tile 0, row-tile stride |
| `sp_w` | Main: weight tile row; Residual: the window's weight base (indexed from it) |
| `acc_bank`, `acc_addr` | ACC destination of row tile 0 (row tile stride `j_run * DIM`) |
| `scale_slot` | `(block - window first block) * j_run + j` in the window's scale context |
| `first`, `last` | first contribution (overwrite) / last contribution of this ACC tile |
| `tail` | beyond the `2 * min(Main, Residual)` alternating micro-tiles of its window |

Every micro-tile is preceded by `weight_preload` with the same identity. Main preload: rows
`[sp_w, sp_w + DIM)`. Residual preload (indexed): stationary row `x < count` = SP row of `original_k(f, x)`,
rows `>= count` zero.

### Order inside a window instance

```text
Main stream     for pass p, for N tile j                (K order)
Residual stream for group g in the window's order:
                  [group switch]
                  for fragment f of g with nonzero digits, for N tile j
issue           MAIN, RESIDUAL, MAIN, RESIDUAL, ... while both have work,
                then the remaining path alone (no dummy micro-tile, no zero operation)
                a mode_switch op marks every path change
```

The 1:1 is the issue order, not equal work. Each output's contributions keep their K order (Main passes
ascending; Residual fragments ascending), so the per-output Sat32 order is the serialized one.

A fragment whose digits are all zero for a group is not issued (`skipped_zero_fragments`): its SCU output is 0 and
`Sat32(acc + 0) = acc`, also for a first contribution (the ACC is zero-initialized).

### Numerics per micro-tile

```text
partial[r][c] = sum_{kk < DIM} A_sp[r][kk] * W_stationary[kk][c]       (exact INT32)
q             = SCU(partial, carrier[scale_slot][c])                    (hp1::apply_validated)
acc[r][c]     = first ? q : Sat32(acc[r][c] + q)                        (hp1::accumulate)
```

## D. ACC lifecycle

* Main: ACC bank 0 holds the stripe x J group tile for all windows; `first` on pass 0 of window 0, `last` on the
  final pass; `acc_complete(MAIN)` (MVOUT) after the J group's last window.
* Residual: ACC bank 1 holds one group (`res_group`, epoch, valid, dirty).
  * enter group g: `residual_a_load` (unless it is the window's prefetched first group), then `group_restore`
    if g was saved in this epoch, else `group_init` (zero).
  * leave group g: `group_save` (raw INT32 to the spill backing, checksum) if g has fragments in a later window
    of this J group, else `acc_complete(RESIDUAL, g)` (MVOUT, final). An unmodified group is never saved.
  * the saved partial is restored bit for bit; the SCU is never applied to it again; saving is not
    dequantization.
* Group order: the windows with Residual work alternate direction (`G0 G1 G2`, `G2 G1 G0`, ...). The last group
  of a window is the first of the next and stays in ACC-R: `(G-1)(W-1)` saves instead of `G(W-1)` when every
  group works in every window. A group without work in a window keeps its ACC-R (if resident) or its saved
  partial; nothing is reinitialized or transferred for it.
* Spill backing: one slot per group, reused across J groups; a restore checks group, epoch and checksum. The
  peak spill (validator) must fit `desc.spill_capacity_bytes`, else `POTAL_ERR_CAPACITY` before execution;
  a corrupted slot fails the restore (`POTAL_ERR_BAD_RESPONSE`) and nothing is committed.
* End of a stripe: no ACTIVE pair, every partition EMPTY, no valid ACC tile, no saved slot, every Main and
  Residual tile completed.

### After the GEMMs (CPU)

* Radix INT64: `R[row][n] = sum C_r[(lane, row)][n] * 256^lane`, checked `__int128`, INT64 range
  (`residual::reconstruct_rows`, shared with `reconstruct`). Overflow fails; it is never wrapped or saturated.
* `LEGACY_F32` (default): `reconstruct_row` (Main F32 with `float(2^theta)`) + `merge_prepared_hp1` — bitwise
  the STRIPE_PIPELINE output.
* `INT64`: `Y = check64(C_main + R)`, `out = float(double(Y) * (double(channel_scale[n]) * double(2^theta)))`,
  non-finite values fail. It may differ from LEGACY_F32 (one rounding instead of two); the tests compare each
  mode with its own golden and print the number of differing cells.

### Trace / report

`paired_schedule::ops` is the trace (`paired_op_text` prints it); it reconstructs K-window loads, Main and
Residual micro-tiles, mode switches, group save/restore, weight preloads, K-window switches and ACC completions.
`paired_execution_report` carries the plan (tiles, stripe rows, windows, groups), per-stripe and total counts
(`paired_counts`, `traffic_basis = "emulated"`), `executed_ops`, `execution_backend`, `status/stage/detail`;
`main_int32_match`, `residual_int32_match`, `reconstruction_int64_match` are set by a verifier (the host tests),
never by the executor.

## E. Hardware changes required

Evidence: `generators/potal/src/main/scala/potal/` (read only; no RTL was changed).

| Function | Current RTL | RTL change needed | Evidence |
|---|---|---|---|
| SP 1:1 Main/Residual partition | Partly: two loop halves (`concurrent_loops = 2`), A from the front, B from the rear of a half; one A region per loop | Yes: a second activation base (Residual limbs at `half/2`) inside one loop, explicit partition addressing | `LoopMatmul.scala` (`a_addr_start`, `b_addr_end`, `a_ex_spad_id`, init at 1240) |
| ACC Main/Residual partition | No: the ACC half alternates per loop (`ex_c_addr_start` floorAdd `max_acc_addr / 2`) for double buffering | Yes: ACC bank by path, kept across window instances | `LoopMatmul.scala` (1138-1150) |
| Indexed weight preload | No: LdB moves contiguous DRAM rows, preload reads contiguous `b_addr` rows | Yes: preload of DIM rows by original-K index list from the resident window | `LoopMatmul.scala` (`b_addr`, LdB) |
| Paired scheduling (1:1 micro-tiles, shared resident B) | No: one loop's preload/compute stream; two loops are independent GEMMs, each loading its own B | Yes: issue of two streams over one resident weight, mode switches | `LoopMatmul.scala`, `ExecuteController.scala` |
| Residual ACC save/restore | Not verified: Gemmini has MVOUT `full_C` and ACC loads (LdD), but the PoTal path rejects D (`tiling.cpp` `check_request`) and EX writes to ACC go through `PoTalAccWriteRouter` / `PoTalWriteback` (SCU path) | Yes, at least verification: raw INT32 restore that bypasses the SCU and keeps work tracking consistent | `PoTalAccWriteRouter.scala`, `PoTalWriteback.scala`, `matmul/tiling.cpp` |
| Scale window reuse | Per invocation: SCALE_WRITE -> LOOP_WS -> FENCE -> SCALE_RELEASE, one generation per invocation | Yes: one scale context per K window shared by both paths, released at `window_end` | `matmul/tiling.cpp` (`potal_sp_tiled_matmul_ws`), `PoTalScaleMemory.scala` |
| Next-K prefetch | Partly: loop i+1's loads may run while loop i executes (two loop halves) | Yes: prefetch of a paired window (Main A, shared B, first group's limbs, scale context) and in-window Residual partition reload synchronization | `LoopMatmul.scala` (`loop_being_configured`, LdA/LdB/EX per loop) |
| Paired completion | No: `firstContribution = !ex_accumulate && k == 0`, `lastContribution = k == max_k - 1` per invocation; one work ID per output tile per loop | Yes: first/last contribution per path and group across windows and spills; completion of both paths before FENCE | `LoopMatmul.scala` (934-935), `PoTalWorkTracker.scala`, `PoTalWriteback.scala` |
| ACC bank count | `acc_banks = 2` in `PoTalConfigs.scala`, not in the generated header (`paired_acc_banks` is a constant) | Header export | `PoTalConfigs.scala`, `potal_hw_params.h` |

## Verification (host)

| Target | Content |
|---|---|
| `potal-paired-plan-test` | geometry, plan limits, frozen plan (K window shrink only), descriptor == serialized request, indexed weight == gathered rows, refusals with 0 ops, tampered addresses |
| `potal-paired-schedule-test` | trace of a 2-stripe / 3-group case, alternation and tails, reversal, save/restore, empty group windows, K order, determinism, validator negatives |
| `potal-paired-unit-test` | golden == serialized == paired for Main INT32, Residual INT32, INT64 on the required shapes and edge cases, LEGACY_F32 and INT64 merges, failures without commit, real ExSIA packets |

These tests run without RoCC. Passing them shows that the software contract holds. It does not show that any
hardware implements it.
