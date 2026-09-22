# Run-aware HP1 residual work

One nonempty `StripePacket` is one logical residual request when its output
uses the common HP1 integer correction domain. Independent caller stripes
remain separate requests. `build_run_aware_request` owns the compact arrays
until synchronous `im2p_execute_matmul_planned_runs` returns.

The row map is ordered by original radix lane ID, then source row. A row
survives if any original block contributes to it. Missing lane/row/K cells
inside a run become zero; block-local lane-group order never becomes a global
lane identity. Each original K32 block contributes one ordered run whose mask
is its selected-K union. Compact K starts at zero and has no DIM padding
between runs. Weights are read from original block/local-K coordinates into
compact rows. HP1 carriers are owned run-major `[run][column]`; numerical
`read_scale(run_ordinal, ...)` selects that row, including when original block
IDs skip from 0 to 3.

The shared `gemmini_set_tile_ws()` selects geometry after final M/N/K exists.
Numerical `IM2P_SIM` builds generate `gemmini_params.h` memory macros from the
verified selected archive's hardware-lowering contract; CPU-functional builds
use their resolved profile contract. The header repository stays untouched.
Tile factors are never clamped after selection.
The descriptor, run view, CYCLE_SIM trace, and offline estimate use those exact
tile counts. One logical request may lower to several hardware submissions.
For each output lane, HP1 SCU saturates every run-local hardware fragment;
signed32 accumulation persists across runs. CPU work after the final matrix
callback applies radix place using original lane IDs, then the existing float
reconstruction/merge. CPU never sums independently completed block-local NPU
finals on this route.

The planned-runs provider writes one contiguous `M*N` final matrix. Caller
stages the Correction and metrics. In CYCLE_SIM collection, final publication
is measured as PoTal host work and rolled back if provenance logging fails
after the copy. Functional GEMM time remains excluded. A missing bound
`execute_planned_runs` capability fails before output publication; legacy
K<=32 SCU routes keep their own guard. Production optrace currently rejects
this run-aware route if it cannot record the runs, rather than emitting a
misleading block-local record.

Activation-block `Meta` has distinct floating scales per original K32 block.
A single saturated cross-run signed32 final cannot reconstruct those separate
terms; that explicit route retains historical block-local composition.
The normal compatible HP1 Correction route uses one run-aware request.

Dedicated CPU-functional NPU trace v2 contains original K, runs, row-map
identity, final geometry and dependencies, but no carriers or tensor values.
Offline replay calls `im2p_cycle_estimate_runs` under isolated-work reference
memory assumptions. Historical v1 block-local residual records cannot be
upgraded by grouping them. The 48-case run-aware RTL certificate remains
`FIXTURE_ONLY`; current production and whole-model readiness require fresh
source-bound RTL evidence and a matched FullCPU/PoTal/replay dataset. No
CPU/NPU scheduler, overlap or latency conversion is claimed here.
