# potal_paired_matmul_auto — paired Main/Residual GEMM on the H2 hardware

Status: **H3-0 (tile plan), H3-1 (execution plan + validator), H3-2A (the
checked C11 command API, `include/potal.h` `potal_paired_*`) and H3-2B
(`potal_paired_matmul_auto()` and its `STRIPE_PIPELINE` route) implemented and
host-tested on command-level emulators.** The route is off by default
(`GGML_POTAL_PAIRED_AUTO`); with it off the serialized `STRIPE_PIPELINE` path is
unchanged. No FireSim run yet: `potal-paired-smoke` is the hardware regression.

Hardware contract (read-only evidence, `generators/potal`, branch `potal`):
`LoopMatmul.scala` (LOOP_WS rs1[20] PoTal enable, rs1[36:21] scale base,
rs1[44:37] generation, rs1[60:45] passes per block, rs1[61] paired ACC,
rs1[62] residual, rs1[63] indexed weight, rs1[19:18]/[17:16] A/B bank pair
selector, rs2[31:3] bank-local A offset in DIM-row tiles),
`PoTalScaleCommandIngress.scala` (INDEX_CONFIG funct 27: rs1[31:0] B sp base,
rs1[63:32] J tiles, rs2[31:0] source K rows, rs2[63:32] active rows; the weight
buffer is derived from the base; INDEX_WRITE funct 26: rs1 slot, rs2 source K
row of the resident panel), `ExecuteController.scala` (indexed PRELOAD: sp row =
bBase + ((sourceK / DIM) * jTiles + jTile) * DIM + sourceK % DIM; padded rows of
the last K pass are not looked up), the H2 BareMetalC tests
(`potal_shared_weight_indexed.c`, `potal_shared_weight_multiwindow.c`,
`potal_h2b1_a_cohabit.c`, `potal_h2c_*`, `potal_h2d_cycle_bench.c`).

## Resources and units

| Resource | Pair 0 | Pair 1 | sp rows |
|---|---|---|---|
| Activation bank (Main A, then Residual A) | bank 0 | bank 2 | `[2p*R, 2p*R + R)`, `R = POTAL_HW_BANK_ROWS` |
| Weight bank (resident original B panel) | bank 1 | bank 3 | `[(2p+1)*R, (2p+2)*R)`, B rear-placed |
| Index Table | buffer 0 | buffer 1 | `POTAL_HW_INDEX_ENTRIES` slots |
| LOOP_WS bank pair selector | 1 | 2 | `a/b_ex_spad_id` |

ACC0 = rows `[0, ACC_ROWS/2)` (Main), ACC1 = `[ACC_ROWS/2, ACC_ROWS)` (Residual).

Units (`paired-auto.hpp`): **sp row** (DIM int8), **tile** (DIM rows),
**original_k** (element of the weight's K), **resident_k = original_k -
panel.k_begin** (Index Table entry value; `INDEX_CONFIG sourceKRows` = the
panel's K rows), **compact k** (position in the stripe's compact residual K).
`POTAL_HW_INDEX_ENTRIES` and `POTAL_HW_SCALE_ENTRIES` come from the generated
parameter header (`cmake/potal-hw-params.cmake`), never from a constant.

## Tile plan (H3-0)

Inputs: the Main auto-tiler's plan (`potal_tile_select`, or the frozen
`STRIPE_PIPELINE` plan), the stripe rows, N, K and the stripe's decoded residual
(`residual::decode_packet`: lane rows, runs). Order, as required:

1. `tile_I = min(Main tile_I, I tiles of the stripe)`, `tile_J = Main tile_J`.
   Never changed afterwards; a capacity the I/J tiles alone exceed is a refusal.
2. Residual geometry: `lane rows` (active `(lane, source row)` pairs, each one
   Residual A/C row), `row tiles = ceil(lane rows / DIM)`, fragments (at most
   DIM consecutive compact K of one run, run order), `fragments per window =
   min(INDEX_ENTRIES / DIM, fragments)`.
3. K limits in DIM tiles, floored to whole scale blocks (`passes_per_block`):

   | Limit | Rule |
   |---|---|
   | weight bank | `tile_K * tile_J * DIM <= BANK_ROWS` |
   | activation bank | `tile_I * tile_K * DIM + ResidualARows <= BANK_ROWS`, `ResidualARows = row tiles * fragments per window * DIM` |
   | scale cache | `(tile_K / ppb) * tile_J + fragments per window * tile_J <= SCALE_ENTRIES` (a Main panel and a Residual window resident together) |

   `tile_K_hardware_limit` = the three without any residual,
   `tile_K_residual_limit` = the three with this stripe's residual,
   `tile_K = min(tile_K_main, tile_K_residual_limit, tile_K_hardware_limit)`.
   `tile_K < one block` is refused as `WEIGHT_BANK_CAPACITY`,
   `ACTIVATION_BANK_CAPACITY` or `SCALE_WINDOW_CAPACITY` (first binding one).
4. ACC: `tile_I * tile_J * DIM <= ACC_ROWS / 2` and `tile_I * tile_J <=
   WorkIdsPerLoop` (`MAIN_ACC_CAPACITY`), `row tiles * tile_J * DIM <= ACC_ROWS
   / 2` and `row tiles * tile_J <= WorkIdsPerLoop` (`RESIDUAL_ACC_CAPACITY`).
5. The Main execution plan is rebuilt with the final K through
   `potal_frozen_plan` (chunk counts `K0` = K panels, padding); the Residual A
   offset is `tile_I * tile_K` tiles (LOOP_WS rs2[31:3]) for every panel.
6. K panels: `[p * tile_K * DIM, ...)`, the last may be shorter; a scale block
   never straddles two panels. A panel's residual = the fragments whose block
   lies in it; they are split into Index Windows of at most `fragments per
   window` fragments: `active_rows = (k_tiles - 1) * DIM + last count`, `pad_k
   = k_tiles * DIM - active_rows` (hardware padding of the last pass), filler
   slots for a shorter non-last fragment (valid resident_k, zero A digits). A
   panel without residual has no window (no Indexed LOOP_WS).

Every count is checked with overflow-safe arithmetic. Refusals carry a
diagnostic string with the numbers; the plan fields computed before the refusal
stay filled.

Known limitation (explicit refusal, not a fallback): residual lane rows beyond
`ACC1 / (tile_J * DIM)` row tiles (lane expansion of a tall stripe) return
`RESIDUAL_ACC_CAPACITY`. Splitting the residual rows into groups would need a
second K sweep per group with a weight-only residency load; it is not planned
here.

Zero fragments: when the decoded packet carries its digits, a fragment that is
zero in every lane row is left out of the fragments, windows and footprints
(`residual_zero_fragments`); SCU(0) = 0 and Sat32(acc + 0) = acc, so nothing
changes numerically. A sealed packet never has one.

## Execution plan (H3-1)

`potal_make_execution_plan(tile, decoded packet)` emits one stripe's operations
in issue order; `potal_validate_execution_plan()` re-checks them independently
and the maker returns a plan only when it passes. No hardware bitfield: H3-2
maps each operation onto the validated H2 command forms.

Residency `r = j_group * k_panels + panel` = one instance of the original B
panel of (J group, K panel) in pair `r % 2`, weight generation `r`. Order per
residency (K panels ascending inside a J group, J groups ascending):

```text
PAIR_RETIRE(r-2)           complete deps: MAIN_GEMM(r-2), every RESIDUAL_GEMM(r-2)
SCALE_PROGRAM(Main r)      slots [0, blocks * j_tiles), generation n     <- SCALE_RELEASE(Main r-1)
LOAD_MAIN_A(r)             activation bank tiles [0, tile_I * k_tiles)   <- PAIR_RETIRE(r-2)
LOAD_WEIGHT(r)             weight bank rear [end - k_tiles*j_tiles*DIM, end)
MAIN_GEMM(r)               ACC0 epoch = J group, first on panel 0, last on the last panel
[ACC_COMPLETE(ACC0)]       after the last panel
per Index Window w:
  INDEX_PROGRAM(r, w)      INDEX_CONFIG(bBase = resident B start, jTiles, sourceKRows = panel K rows,
                           activeRows) + INDEX_WRITE(slot, resident_k) x activeRows: one transaction;
                           in-order dep on the previous RESIDUAL_GEMM of the same buffer (hardware-stalled
                           while busy, validated H2-2C); no dep on the other buffer (cross-buffer update)
  SCALE_PROGRAM(Res r,w)   slots [scale_slots_main, + k_tiles * j_tiles), one block per fragment
  LOAD_RESIDUAL_A(r, w)    activation bank at tile_I * tile_K tiles; complete dep on the previous
                           RESIDUAL_GEMM of the pair (same region reused: conservative)
  RESIDUAL_GEMM(r, w)      ACC1 epoch = J group; first / last at the true first / last window over all panels
  [ACC_COMPLETE(ACC1)]     reconstruction_required: raw INT32 lane rows, radix step on the CPU
  SCALE_RELEASE(Res r,w)   complete dep: RESIDUAL_GEMM(r, w)
SCALE_RELEASE(Main r)      complete dep: MAIN_GEMM(r); after the windows, since an accepted
                           SCALE_RELEASE is a drain point (Controller.scala releaseSafe ==
                           fenceDrained): Main(r) and Residual(r, w0) share the stream, the
                           first release drains both
...                        PAIR_RETIRE of the last two residencies at the end
```

In H3-2: `LOAD_MAIN_A + LOAD_WEIGHT + MAIN_GEMM` = one Main LOOP_WS (its LdA /
LdB), `LOAD_RESIDUAL_A + RESIDUAL_GEMM` = one Indexed LOOP_WS (LdA at the
Residual A offset, B DRAM address 0), `ACC_COMPLETE` = the C store of the last
contribution. A dependency with `complete = true` must be *completed* (POTAL_FENCE
or a validated ordering argument) before the op is issued; `complete = false`
means the in-order command stream and the hardware's own hazard handling order
it. Issuing is never taken for completion.

Index entries: slot `s` of a window = the s-th packed digit column of the
window's Residual A (`compact_k`), `resident_k = original_k - panel.k_begin <
panel.k_rows`; a shorter non-last fragment is padded with filler slots (a resident
row, zero digits, `compact_k` none); the last fragment's tail is hardware padding
(`pad_k`), not an entry. An empty panel has no Index transaction and no Indexed
GEMM.

Pair phases: `FREE -> LOADING (first load) -> RESIDENT (Main GEMM) -> RETIRED
(PAIR_RETIRE) -> LOADING ...`; a pair is loaded again only after the retire of
its previous residency, and the retire waits for every Main and Residual
consumer. The ACC1 identity is (J group); it is never reset at a panel or pair
change. Scale windows: Main at slot 0, Residual behind the largest Main window
(the H3-0 budget: one Main panel + one Residual window live), generations 1..N
in program order, release after the consumer completed, no live overlap.

Validator checks: op ids and dependency direction; residency order and pair =
r % 2; pair reuse only after a retire whose completion deps cover every
consumer; sp regions inside their banks, Residual A behind Main A, B
rear-placed; INDEX_CONFIG fields against the resident panel; entries complete,
in slot order, in range, `resident_k = original_k - k_begin`, original K = the
packet's; same-buffer updates after the previous Indexed gemm and never between
a window's program and its gemm; scale windows in the cache, no live overlap,
release only after a completed consumer, generations in sequence; ACC
identities with exactly one first and one last in order and a completion
after the last (ACC1 flagged for reconstruction); work IDs within
WorkIdsPerLoop; every Main panel and residual window exactly once; every
residency retired; residual rows within ACC1.

Not planned here (explicit refusal stays): residual rows beyond ACC1 (row-group
sweeps, weight reloads, spill/restore). Not promised: simultaneous GEMMs or
two live Main scale windows.

## C11 command API (H3-2A, `include/potal.h`, `potal.c`, `potal_rocc.h`)

Pure encoders (host-testable, no hardware): `potal_index_config_check/encode`
(funct 27: rs1 = {jTiles[63:32], bBase[31:0]}, rs2 = {activeRows[63:32],
sourceKRows[31:0]}; bBase and the region inside the pair's weight bank,
activeRows <= `POTAL_HW_INDEX_ENTRIES`), `potal_index_write_check/encode` (funct
26: rs1 = slot < activeRows, rs2 = resident_k < sourceKRows),
`potal_paired_loop_ws_check/encode` (the regular 10-command program with LOOP_WS
rs1 bits 61 paired / 62 residual / 63 indexed, a/b scratchpad selector = pair +
1 at [19:18] / [17:16], rs2[31:3] = the bank-local A offset in tiles; Main loads
the original B, Residual has B DRAM address 0; C store unsupported: results
leave the ACC through the readback), `potal_acc_readback_check/encode`
(CONFIG_ST + one MVOUT per tile: rs2 = {rows[63:48], cols[47:32], 5 << 29 +
bank * ACC_ROWS/2 + (i * j_tiles + j) * DIM}, the H2 `acc_desc`).

Paired session (`potal_paired_session_open/close/query`, `_tlb_flush`,
`_scale_write/_scale_release`, `_index_config/_index_write`, `_loop_ws_issue`,
`_acc_readback_issue`, `_fence`): one owner of the accelerator; the regular
`potal_gemm_ws_issue` / `potal_loop_ws_issue` / `potal_fence` / `potal_tlb_flush`
refuse with `POTAL_ERR_BUSY` while it is open, and it cannot open while a
regular GEMM is unfenced. Several paired LOOP_WS stay outstanding until
`potal_paired_fence()` == `POTAL_OK`. Runtime rules: a Main LOOP_WS needs a pair
with no outstanding work and records its resident B (base, J tiles, K rows) and
Main A footprint; INDEX_CONFIG opens one transaction (refused while another is
unfinished, any pair while the other executes), INDEX_WRITE fills it in slot
order, the window is valid when complete; a Residual LOOP_WS needs the pair's
resident B == its complete window (base, J tiles, K rows, active rows ==
k_tiles*DIM - pad_k) and an A offset behind Main A; a live scale slot is never
rewritten with another generation (`POTAL_ERR_CAPACITY`), release needs the
live generation; a hardware status other than OK makes the session failed
(sticky error and stage, outstanding work kept, fence still drains); close
refuses with outstanding work unless failed, and a failed session with
outstanding work keeps the accelerator owned. On a host without a sink every
issue validates and returns `POTAL_ERR_NO_HARDWARE` without changing state; a
sink (`potal_paired_sink`) records the command stream for tests, it models no
RTL concurrency.

## Executor (H3-2B.1, `potal_paired_matmul_auto`)

`potal_paired_matmul_auto(const potal_paired_request &, potal_paired_session *,
potal_paired_report &)`: request = stripe identity, rows / n / k, the frozen Main
`potal_tile_plan`, Main A8 codes (`a`, `lda`), the prepared `hp1_weight`, the
decoded packet (`residual::decode_packet`, with digits; nullptr: none), Main C and
Residual C (INT32, DIM*4-byte aligned, the readback destinations), TLB flush flag
and the owner's 8-bit generation counter. Steps, all checks before the first
command: request -> `potal_paired_plan_stripe` (H3-0 + H3-1, validated) ->
residual staging -> session idle -> TLB flush -> the plan's ops -> final fence.

| H3-1 op | C11 command(s) |
|---|---|
| LOAD_MAIN_A + LOAD_WEIGHT + MAIN_GEMM | one Main `potal_paired_loop_ws` (A = codes at the panel's K, B = the prepared weight rows of the panel, original `ldb`, columns of the J group; ACC0; accumulate except the first panel) |
| INDEX_PROGRAM | `potal_paired_index_config` + one `potal_paired_index_write(slot, resident_k)` per entry |
| LOAD_RESIDUAL_A + RESIDUAL_GEMM | one Indexed `potal_paired_loop_ws` (A = the window's staged digits, B DRAM 0, `a_offset_tiles` = the plan's, ACC1, passes_per_block 1) |
| SCALE_PROGRAM / SCALE_RELEASE | `potal_paired_scale_write` / `_release` of the window's (slot, column) with the next hardware generation; Main blocks = the panel's scale blocks, Residual blocks = the fragments' original blocks |
| ACC_COMPLETE | `potal_paired_acc_readback_issue` of the epoch's tiles into Main C / Residual C at the J group's columns |
| PAIR_RETIRE, `complete` deps | satisfied by a drain point already passed, else `potal_paired_fence` |

Residual staging (`potal_paired_pack_residual`): one DMA-ready `[lane rows][lda]`
matrix per Index Window; column s = the compact digits of the window's slot s, in
the plan's entry order; filler slots and the `pad_k` tail are zero.

One call executes one stripe: rows must fit one Main I group (`rows <= tile_I * DIM`,
the STRIPE_PIPELINE stripe height); more rows are refused, never truncated. A caller
that already planned (the stripe route) hands its plan in `potal_paired_request::plan`:
it is validated again and executed as is.

Drain points: an accepted SCALE_RELEASE is one (RTL `releaseSafe == fenceDrained`),
so releasing a window drains everything older; explicit fences are issued only for
a `complete` dependency no drain point covered yet, before an ACC epoch is
overwritten while its readback is pending, and at the end (the CPU reads C only
after it). Failure: an error status the hardware answered is followed by
`potal_paired_session_cleanup` (fence + release of the session's live scale
entries): not fatal. A failed fence is never retried: fatal, the staging is
abandoned, the caller keeps every buffer. An exception after the first command
(out of memory while reporting) gets the same settlement without allocating.

### Ordering the stream relies on (RTL, not yet hardware-validated)

A read-only audit of the RTL (8 questions, each confirmed by independent reviewers)
found these orderings guaranteed without a fence; no H2 bare-metal test exercised
them (every H2 test fences between the steps), so `potal-paired-smoke` is their first
hardware run:

| Ordering | Mechanism (generators/potal/src/main/scala/potal) |
|---|---|
| Indexed PRELOAD after the Main LdB of the same pair | ReservationStation gives an indexed PRELOAD the hazard range of the whole resident panel from the INDEX_CONFIG descriptor (`bBase`, `ceil(sourceKRows/DIM) * jTiles * DIM`) and waits for every overlapping MVIN's completion |
| MVOUT after the LOOP_WS writing those ACC rows | RS PRELOAD -> STORE dependency on the ACC rows (CONFIG_EX c_stride 1, CONFIG_ST pool_stride 0), writeback delays PRELOAD completion to the SRAM commit |
| SCALE_RELEASE as a drain | `releaseSafe := allDrained`, the FENCE condition (Controller.scala); every older LOOP_WS, LD/EX/ST, MVOUT ack and SCU writeback |
| Residual A reuse, ACC epoch reuse, Index Table updates | release drains, the 10-command CONFIG preamble (LoopMatmul admits non-loop commands only with no loop configured), H2-2C same-buffer stall |
| LoopMatmul slot reuse | two loop slots: a LOOP_WS two positions later reuses a slot and its work IDs; the per-window releases drain in between (validator rule) |

Obligations the software keeps (checked by the session or the validator): INDEX_CONFIG
and every INDEX_WRITE answered OK before the Indexed LOOP_WS, `bBase` = the LdB rear
placement, `pad_k = k_tiles*DIM - activeRows`, every column of every slot written
before the LOOP_WS that reads it (a missing one would stall the PRELOAD), releases
with `xd`, one final POTAL_FENCE before the CPU reads C.

`GGML_POTAL_PAIRED_FENCES=CONSERVATIVE` (request field `conservative_fences`) adds a
POTAL_FENCE before every Indexed LOOP_WS, ACC readback and scale release, the H2-tested
placement, for an A/B run on FireSim; RTL (default) relies on the table above.

## Stripe route (H3-2B.2)

`pipeline_config::paired` (STRIPE_PIPELINE, compensation ON) makes every stripe:

```text
main_begin -> paired_prepare_begin -> decode_packet once + potal_paired_plan_stripe
  ok:        paired_prepare_end(route=paired) -> paired_begin -> owner paired runner
             (issuer: hart-0 check, own paired session, potal_paired_matmul_auto) ->
             paired_end -> reconstruct_row (Main) -> radix reconstruct_rows -> merge_prepared_hp1
             -> semantic_complete -> capacity_release
  capacity:  paired_prepare_end(route=serial_fallback, refusal) -> the serial route
             (Main GEMM, residual::build_request, Residual GEMM) as before
  other:     the stripe fails (paired_plan); a bad packet fails at residual_prepare
```

No main_end / residual_* events for a paired stripe. The paired runner is part of
`owner_runners` (`execution.hpp`): the issuer runs it on its thread with its scale
generation counter; host tests bind the H2 emulator. A fatal paired report fails
the run fatal (the issuer disables itself, the run's and the weight's buffers are
abandoned).

Option: CMake `GGML_POTAL_PAIRED_AUTO` (default OFF) sets the default; the
environment variable `GGML_POTAL_PAIRED_AUTO=ON|OFF|1|0` overrides it per node (an
invalid value fails the node). `GGML_POTAL_PAIRED_FENCES=RTL|CONSERVATIVE` selects
the fence placement. Counters: `ggml_backend_potal_stats::paired_*` (stripes,
fallback stripes, Main / Indexed LOOP_WS, INDEX_CONFIG / WRITE, readbacks, MVOUT
tiles, fences, drains), scale_s / issue_s / fence_s (both routes), and the `[POTAL]`
node record; the planned-GEMM counters (hw_requests, tiled_*, main_* / residual_*
GEMMs, residual_execution_k) stay the serial route's.

## Verification

`potal-paired-execution-test` (host): command translation counts and order,
staging, Main / Residual INT32, radix INT64 and merged F32 == the serialized
reference bitwise on 9 geometries (panels, windows, J groups, tails, decode, carry
lanes, sparse), pair alternation and generation wrap, failure paths (answered
error -> cleaned, failed fence -> fatal, no hardware), and the stripe pipeline:
real ExSIA through `run_stripe_pipeline`, paired == serial F32 bitwise at
capacity 1 and 2, exactly one paired runner call and no planned GEMM per paired
stripe, paired event order, capacity fallback, missing runner, fatal report.
The emulator (`tests/support/paired-h2-emulator.hpp`) interprets the raw command
stream on the documented H2 semantics; it does not model RTL timing or hazards.

`potal-paired-smoke` (FireSim, RISC-V): real GGML MUL_MAT graphs on the POTAL
backend, each case paired (RTL), paired (CONSERVATIVE) and serial, all == the
stripe-pipeline golden bitwise, exactly-once counters, measured steady-clock compute
time of the three runs after an untimed warm-up (`POTAL_PAIRED_PERF`, single runs).

Known gaps of the host tests: the emulator applies each command atomically, so it
cannot show an ordering hazard (that is the smoke's job); buffer abandonment after a
fatal report is checked through the reports' flags, not through the allocator.

`potal-api-test` (C11, host DIM16 and DIM32 API-only builds): INDEX_CONFIG /
INDEX_WRITE fields against the H2 firmware's encoding, Index capacity per
DIM, slot / resident K rejection, pair 0 / 1 selectors, Main vs Residual flags,
bits 61-63, rs2[31:3] A offset, ACC0 / ACC1 readback == H2 `acc_desc`, padding,
bank overlap refusals, Index transaction non-interleaving, regular API refusals
during a session, session lifecycle, several outstanding LOOP_WS without a
fence, failure states, host NO_HARDWARE, unchanged `potal_gemm_ws_encode` /
`potal_loop_ws_encode` literals.

`potal-paired-auto-test` (host, no RoCC). H3-1 cases X1-X17 and XR: single
panel without / with residual (exact op sequence), several panels, residual in
some panels only, several windows in one panel with same-buffer and
cross-buffer Index dependencies, tails and fillers, pair alternation and
retire, zero fragments, scale lifetimes, I/J/K tails, determinism, 20 mutated
plans refused, every contribution once, real RMD packets against the
`paired.hpp` descriptor. H3-0 cases: geometry and pairs; Main I/J kept and
the final K propagated; the K formula against independently recomputed limits
with the binding limit named per case; occupancy; the Residual A offset; panels
(coverage, block boundaries, original -> resident K, empty panels); Index
Windows (capacity, non-DIM-aligned tails, fillers, no compact K lost); real RMD
packets through `decode_packet` against the paired descriptor of `paired.hpp`;
refusals and determinism.
