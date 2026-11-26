# H3-8D: E4C1 production integration and E4C2 prefetch implementation plan

Date: 2026-10-29.

Status: plan only, prepared after the actual DIM32 Functional/E2E baseline
passed. No work described below is implemented by H3-8.

Paths below are relative to `ggml/src/ggml-potal`, except RTL paths, which
are relative to Chipyard. Historical design references are the existing
`H3-6E4C1-sequential-paired-microtile-integration.md` and
`H3-6E4C2-0-prefetch-admission-completion-feasibility.md`.

## Entry evidence and boundaries

H3-8C records execution `173-130094429000`: 256 prompt tokens, 128 output
tokens, CLI exit 0, 62592 successful joined node executions, all 9344
matmuls on POTAL/TARGET_NPU, no required CPU GEMM fallback, balanced
1047139964 Scale writes/Releases, and byte-identical DIM16 stdout.
The completed logs, manifest and comparison are:

```text
/tmp/h3-8/full-results
/tmp/h3-8/full-results.sha256
/tmp/h3-8/full-acceptance.json
/tmp/h3-8/dim16-dim32-comparison.json
```

Freeze the H3-8A/B/C profiles, source identities, binaries, model, prompt,
DIM16 artifacts and DIM32 bitstream/package before future implementation.
Use new build/result/rootfs/HWDB paths, never overwrite those baselines.
The starting DIM32 software and RTL revisions are
`fc0bdcdd0dba8ccfef8a54cd2e3c7f39a0be3c9b` and
`10ff4a10a87662edc7efd5d1a718027c1853550b`, plus uncommitted H3-8 edits.
Bind future comparisons to full source and artifact hashes, not HEAD alone.

Production H2 remains the default reference. E4C1 integration must first be
explicitly opt-in; E4C2 is a separate later opt-in change. Neither stage
changes HP1 packing, ExSIA adaptation, compensation, quantization, numerical
tolerances, attention arithmetic, or Release/Fence synchronization.
No commit, push or implementation is authorized by this document itself.

## Current-source findings that control the order

| Current evidence | Planning consequence |
|---|---|
| `matmul/execution.cpp:717-756`, `matmul/execution.hpp:124-142` construct `owner_runners.paired` inside the existing issuer callback | Change the owner-side paired runner, not producer threads, owner locking or graph routing. |
| `tests/support/e4c1-sequential.hpp:3-5,67-76` is explicitly test-only and consumes an existing validated plan | Reuse its mechanism through a production-owned implementation; production must not include test support files. |
| `tests/support/e4c1-sequential.cpp:16,495-522` fixes D16, Index32, J9/K28 and the old ACC geometry | It is not a DIM32-ready executor. Do not merely move it into production. |
| `potal.c:657-661,1746-1753` rejects resident open outside fixed DIM16 geometry | DIM32 production H2 acceptance does not establish DIM32 resident-API acceptance. |
| `include/potal.h:642-668` documents one pending transaction, shared checked session, owner-thread use and quarantine | Preserve these contracts during E4C1; do not introduce prefetch there. |
| `generators/potal/src/main/scala/potal/PoTalResidentView.scala:17-33` restricts resident capability to DIM16/SP4096/ACC-bank512/Index32/20-bit partial | The current H3-8 DIM32 image cannot be treated as resident-capable. A DIM32 resident RTL port and separate image are prerequisites. |
| `generators/potal/src/main/scala/potal/Controller.scala:62-63,305` passes that capability gate to counters and loop logic | A software override or fabricated capability cannot enable the extension safely. |
| `CMakeLists.txt:121-128,446-467` registers default-OFF C2-0 and C1 tests | Existing software models remain acceptance inputs, not production or hardware proof. |

The actual DIM32 header hash is
`3580b84f15ed8de9f9d311edea64eaa7c1f9bb308b1552a8f79453ec2c05ea33`.
It defines DIM32, four 2048-row SP banks, 512 total ACC rows, Scale256,
Index64 and MAX_BYTES64. Each ACC bank has eight DIM32 tiles.
K32 has one physical partial. The existing DIM16 resident constants,
six 64 KiB pools, physical addresses, J/K maxima and capability signature
must not be copied as DIM32 assumptions.

## Stage 1: establish a checked DIM32 resident contract

This is a prerequisite to E4C1 integration, not part of E4C2.

1. Audit the current C11 resident descriptor encoders, buffer extents,
   liveness/generation checks and every RTL resident-view address consumer.
   Derive SP pair bases, rear B placement, A/B capacity, ACC banks/epochs,
   Index windows, Scale peaks, output alignment and J/K limits from the
   generated geometry. Do not assume J9/K28 or infer capacities from DIM
   alone. Audit the fixed 16-entry fragment declarations and two-fragment
   K32 assumptions in `include/potal.h:723-748`; DIM32 has one physical
   K32 partial. Keep the current H2 planner; do not clone it.
2. Extend the existing checked `potal_paired_resident_*` contract to the
   supported DIM32 geometry. Keep the same opaque session and issuer owner.
   Retain DIM16 behavior and capability compatibility. Define an explicit
   geometry-bound DIM32 capability contract in C11 and RTL; do not relabel
   the old DIM16 signature or permit an environment override.
3. Prove DIM32 native and indexed resident views, source extents, ACC
   addresses and 128-byte INT32 alignment. Preserve DIM32 signed 21-bit
   partials and its physical reduction/saturation order. A DIM16 numerical
   golden is not a valid substitute for a DIM32 golden.
4. Add focused regression coverage beside the existing resident C11 and
   RTL tests. Require invalid geometry/signature, stale handles, incomplete
   descriptors, wrong bank/epoch, overflow/capacity and alignment refusals
   before effects. Preserve existing DIM16 tests and historical failures.
5. Generate a separately named DIM32 resident-capable FireSim image only
   after native/RTL gates pass. Bind generated parameters, capability,
   driver, bitstream, HWDB, timing and utilization reports to its hashes.
   Run direct checked resident Main and Indexed Residual numerical gates
   before any scheduler experiment.

Stop this stage on the first failed address, numerical, ownership,
capability or timing gate. Preserve its first evidence. The existing H3-8
DIM32 bitstream remains the production H2 baseline, not an extension
compatibility claim.

## Stage 2: integrate sequential E4C1 behind one opt-in route

Promote only the reusable sequential executor mechanism into `matmul`;
leave command models, golden arithmetic and benchmark proxies under
`tests`. Route the selected paired call inside the existing
`issuer_owner` callback. Keep producer preparation, bounded queue,
issuer affinity, packet lowering, reconstruction, merge and weight-cache
ownership unchanged. The legacy paired runner remains selectable.

Consume the supplied validated H2 plan and decoded sealed packet.
Parameterize the executor using Stage 1's supported geometry and checked
limits. Preserve each Main and compact Residual contribution order,
including sparse groups, multiple lanes, tails and zero groups.
No reordering may change partial saturation, ACC saturation or final INT64
reconstruction. Do not add ACC spill/restore, a second planner, raw RoCC
issuing, prefetch or an unchecked test-local resident handle.

Retain the sequential load/compute/readback lifecycle:

- Reserve source and pair identity before any possible load effect.
- Publish B only after the matching fully issued load and successful global
  drain; retain source storage until then.
- Bind every consumer to immutable B/Index/Scale generations and ACC epochs.
  Complete its current checked Release/Fence boundary before subsequent work.
- Retire B only after all declared Main and Residual readers and metadata
  leases finish. Retirement must not erase ACC epochs.
- Publish caller output only after all contributions, readbacks, retirement
  and close succeed.
- Preserve the first sticky failure. Permit only the existing recoverable
  cleanup drain. Unknown completion or failed cleanup retains buffers,
  owner and dependencies in quarantine; do not retry, free or silently
  fall back after effects.

An explicit opt-in request that is unsupported must refuse before effects,
not quietly execute another route. Do not make this the production default
merely because correctness passes.

## Stage 3: prove E4C1 correctness and decide adoption

Use separate DIM16 and DIM32 native builds with the current canonical
header/profile bindings. Enable the existing `POTAL_E4C1_TESTS` target
family and resident API tests. Build/run `potal-e4c1-test`,
`potal-e4b2-test` and the applicable production C11, Residual, H2 planner,
H2 command, Stripe Pipeline, attention and GGML unit tests. Use CTest
`--no-tests=error`; cross-compilation alone is not execution.

Extend existing deterministic tests instead of creating a new framework.
Compare public-packet lowering with decoded/move lowering across metadata,
padded buffers, carriers, dimensions, strides, plans and alignment.
Cover compact/sparse/dense, multiple runs/lanes, Main-only/Residual-only,
K/J/M/N tails, multiple panels, generation wrap/refusal, stale bindings,
partial MVIN errors, issue-before/after-effect errors, cleanup failure and
unknown Fence/Release completion. Use independent wide goldens for raw
Main, Residual, reconstructed INT64 and final output. Keep destructive
saturation witnesses. Async tests subscribe to exact state changes; no
fixed sleeps or timing-luck assertions.

On the new resident-capable DIM32 image, repeat the checked API gates,
the exact seeded `decode_1x768x768` and
`gpt2_ffn_up_16x3072x768` fixtures, then QK/PV and range-extension regression.
Use the existing owner, confirm Linux CPU/Hart0 mapping, and keep producers
off that CPU. Require exact same-DIM numerical agreement, completion,
zero unexpected fallback and balanced Scale lifetimes before the 256/2
smoke and exact 256/128 Functional request.

Correctness is necessary but not a speedup. Historical DIM16 C1 medians
were 23.697/122.927 ms versus legacy K28 Paired 13.189/79.623 ms; the report
explicitly calls C1 slower. The H3-8 DIM32 baseline also has higher TPOT
than historical DIM16. Do not use either result to assert a future benefit.

Compare DIM32 H2 and C1 in the same revision, image, profile and request.
Use separate diagnostics/raw-trace correctness runs and matched
diagnostics-OFF benchmark runs. Preserve raw samples and command counts;
do not mix Functional and Performance profiles. Measure actual work
reduction before considering adoption. CPU command elapsed time is not
hardware waiting, and overlapping intervals cannot be summed into graph
wall time. If C1 remains slower, retain the explicit opt-in route and
report that result; do not bundle unrelated optimizations.

## Stage 4: implement only the bounded E4C2 completion protocol

Start only after the checked DIM32 resident port and sequential C1 route
pass the previous gates. Preserve C1 as the serial reference.

The current `potal_paired_resident_query` returns software bookkeeping;
it does not poll independent hardware completion. Global pending-state
exclusion must not simply be removed. Introduce the smallest separately
checked sealed-prefix completion operation and the necessary independent
load/compute ownership state, preserving the single issuer and global
Release/Fence contract.

Use the restrictions in C2-0 section 5:

1. One issuer exclusively owns the command stream, counter configuration
   and both source pools.
2. Premark destination pair, immutable owner/generation/weight descriptor,
   source range and issue cut before effects. Distinguish partially issued
   descriptors from fully admitted ones.
3. Place the query after every relevant generated command in the ordered
   FIFO, after generator ownership clears. Seal the prefix against new
   load-producing work while query/publication resolves.
4. Explicitly configure the live external LD/EX selector, clear snapshot
   mode, discard the configuration response, then obtain a separate
   full-width live response. Bind the sample to the sealed owner,
   generation, descriptor and issue cut.
5. Publish/reuse a load source only after a qualified matching whole-prefix
   LD-count-zero certificate and required CPU memory ordering, or the
   preserved global drain. Never publish a failed or incomplete load.
6. Qualify compute through the complete EX/ACC prefix, not raw generator
   idle or unscaled Execute completion. Keep readback receipt usability
   behind acknowledged stores and the existing global drain.

This certifies all loads in a prefix, not a selected pair while another
load remains. Received-byte totals, command return, queue acceptance and
status-only responses are not completion certificates. An unrestricted
per-pair/generation ticket or earlier raw-MVIN admission requires a
separate RTL/interface design; it is not included in this bounded stage.

Audit every pending-state producer/consumer and alias, close, cleanup,
Fence, Scale Release, retirement and failure path. Any uncertain pair
poisons the shared issuer and live dependencies; do not recover one half
of an unknown accelerator. Do not bypass a blocked in-order Scale Release.

## Stage 5: prove E4C2 admission, completion and useful overlap

Retain `potal-e4c2-0-test` and its default-OFF
`POTAL_E4C2_0_TESTS` software model as regression evidence. Add checked-C11
and generated-DIM32 RTL tests for full versus partial issuance, stale
samples, snapshot contamination, nonzero unrelated loads, generator/RS
ordering, final SP writes, delayed ACC commit, acknowledged stores,
source poisoning after completion, blocked Release, reader lifetime and
all sticky/unknown cleanup outcomes. Preserve DIM16 behavior.

Re-audit the actual Stage 1 DIM32 image and live-counter path; old DIM16
generated RTL is not proof of its implementation. A further image rebuild
is needed if the bounded protocol cannot be supported by that verified
image. Do not claim that aggregate counters supply per-pair tickets.

Only after these gates, attempt late prefetch into the physically disjoint
inactive B pair after the final selected Residual generator, without
admitting its readers early. Prove B publication, old reader completion,
metadata retirement and continuing ACC epochs before advancing. Repeat
same-DIM numerical and error gates before inference.

Keep CPU admission latency, final SRAM-qualified DMA completion,
ACC-qualified compute completion and global Release/Fence waiting
separate. Actual hardware activity/qualified intervals are required to
claim DMA/compute overlap; logical extents and CPU timestamps alone cannot
prove bus bytes, hardware time or performance benefit. The legal sequence
may only move waiting into admission or Release, and that is a valid
negative result to preserve.

## Deliverables and stop conditions for the future work

Deliver separate resident-DIM32, C1-integration and C2-protocol changes,
each with its own source/artifact hashes, numerical/ownership evidence,
first failure evidence, and explicit software-only versus hardware labels.
Do not merge them into a single opaque optimization.

Finish each stage when its stated gates pass or a precise unsupported
contract is evidenced. Keep default H2 and frozen DIM16/H3-8 artifacts
usable throughout. Do not enable a default route or claim a speedup
without the matched evidence above. This document completes the requested
post-baseline plan only; none of these future builds, tests, API changes,
RTL changes or hardware experiments has been run as part of H3-8D.
