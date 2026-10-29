# H3-8 checkpoint and H3-8D final report

Date: 2026-10-29.

## Outcome

The authorized checkpoints, matched baseline, opt-in K-tile implementation,
checked next-K prefetch, actual FPGA overlap gate and repeated comparisons
are complete. Negative results are retained. H2 remains the default;
K-tile and Prefetch require explicit selection. No push occurred.

| Stage | Evidence-backed result |
|---|---|
| H3-8 checkpoint | Reviewed software and RTL commits; prior inference/artifact evidence preserved |
| D0 matched baseline | DIM32 H2 is slower than DIM16 on both requested microbenchmarks |
| D1 K-tile | Removes real redundant Scale commands; faster than matched H2 |
| D2 Prefetch | Checked ownership and physical overlap pass; slower than matched K-tile |

## Commits and source state

The two authorized checkpoint commits are:

- Software `df8482e9fec038dd1f5db48c8b3ab9c1c11c3df9`:
  `feat(potal): validate DIM32 FireSim full-model inference`.
- RTL `8c839ea348a42398da6822440209955b3a015ff8`:
  `feat(potal): add DIM32 quad-Rocket FireSim target`.

Both were created on `potal`, covering 35 software and eight RTL
configuration/test/report files. The historical E4C1/E4C2 document remains
a plan. D1/D2 implementation and new reports are separate uncommitted work,
not silently included in the checkpoints.

Models, images, bitstreams, caches and unrelated incoming changes were not
committed or reverted. Existing dirty Gemmini/host-init/overlay state remains
outside this work. SBT also updates tracked build-stream files; those are
not source fixes and were excluded from the scoped source whitespace check.
The Chipyard-local conda warning remains; commands explicitly activate the
working `/home/alveo/firesim/.conda-env`.

## Matched comparisons

These are within-experiment guest steady graph medians, milliseconds.
Do not stack gains across different binaries or images.

| Experiment / workload | Baseline | Candidate | Candidate change |
|---|---:|---:|---:|
| D0 Decode, DIM16 H2 -> DIM32 H2 | 15.946 | 16.852 | +5.681676% |
| D0 FFN, DIM16 H2 -> DIM32 H2 | 82.650 | 89.275 | +8.015729% |
| D1 Decode, H2 -> K-tile | 17.061 | 13.820 | -18.996542% |
| D1 FFN, H2 -> K-tile | 89.170 | 70.972 | -20.408209% |
| D2 Decode, K-tile -> Prefetch | 14.353 | 15.368 | +7.071692% |
| D2 FFN, K-tile -> Prefetch | 73.123 | 76.864 | +5.116037% |

D2's same-image H2 controls are 17.908 ms Decode and 92.861 ms FFN.
Prefetch is faster than those H2 controls, but not than D1's K-tile mechanism
in the same experiment. Serial controls in D0/D1 also remain in their
reports; no routing default was changed to conceal that Serial was faster.

D0 has 168 exact primary samples; D1 and D2 each have 84 exact primary
samples. D2 additionally has 36 separate-profile samples. Route order
rotates/reverses; cold and steady samples are separate; no outlier was
removed. Per-phase reports include MAD/ranges, command counts and
independently recomputed summaries.

## D2 physical proof and its limits

In the M63/N129/K544, I2/J4/K4 fixture with 63 active Residual rows,
Prefetch produced **103 target cycles** of physical opposite-pair B SRAM
DMA-write and qualified PE00 MAC intersection: 31 in one direction and 72
in the other. H2/K-tile controls measured zero. The two smaller fixtures
still measured zero. All nine raw numerical cases passed.

The witness proves a bounded hardware overlap mechanism. It does not prove
overlap for every workload, the whole array, or specifically the performance
benchmarks. It is not DMA duration, bus bytes or a CPU timing inference.

Getting this result required prepared-reference issue, removing unnecessary
pre-DMA state scans, publishing immutable B once at idle bind, and keeping
CPU clock queries out of the Residual-to-Prefetch interval. Rocket stalls
CPU FENCE on RoCC busy; the configured VDSO's time-read barriers matter.
Actual checked Fence/Release publication and allDrained semantics remain.

The combined Residual+Prefetch CPU API interval is reported once. CPU API
latencies, architectural counters and logical descriptor extents are not
presented as hardware waiting, accelerator cycles or actual DMA volume.
Overlapping stage intervals are not summed into graph wall time.

## Common acceptance

- Latest native regression: 11/11 DIM16 and 11/11 DIM32, exit 0.
  Numerical/command tests include tails, compact/gapped/multi-lane cases,
  both pairs, generation wrap, Scale identity and default streams.
- Ownership tests cover prepared references, stale/reentrant requests,
  partial current/future issue, one-drain cleanup, first sticky error,
  separate unknown completion, quarantine and no-sink owner release.
  These fault-injection results are software-only.
- D2 actual FPGA: API smoke, nine independent raw Main/Residual INT32
  goldens, physical overlap, full attention and PV-range smoke pass.
  Native reconstruction tests and every measured graph's F32 golden pass.
- Actual PV-range result: exact=1/status=0; 49 jobs, 16 bitplane jobs,
  11 corrections, 22 bias MVINs and 28224 balanced Scale writes/releases.
- All D2 primary/profile samples balance Scale writes/releases and report
  zero unexpected Paired fallback. Explicit unsupported opt-ins refuse
  before execution rather than silently selecting Serial.
- D2 targeted RTL: observation/capability 4/4, DIM32 ingress 1/1 and real
  DIM32 SRAM/writeback/ROB drain 3/3. The pre-existing reduced-cache drain
  fixture still fails 3/3 at its mismatched key encoding; no full-suite
  PASS is claimed.
- Routed timing meets reported constraints: WNS +0.029 ns, WHS +0.010 ns,
  zero failing endpoints, no-clock checks and unconstrained internal
  endpoints. Recurring generated-XDC warnings are documented, not hidden.

The canonical full-model timing path, model/prompt, HP1/ExSIA/RMD arithmetic
and attention math were not retuned. No full-model inference rerun is
claimed for D0/D1/D2; the checkpoint's earlier exact 128-token, 9344-matmul
hardware evidence remains historical preserved evidence.
No unsupported DIM32 E4C1 timing or resident-extension port is fabricated.

## Reports and artifacts

- `H3-8D0-matched-performance-baseline.md`
- `H3-8D1-ktile-paired-implementation.md`
- `H3-8D2-next-k-prefetch-implementation.md`
- Checkpoint inclusion/exclusion ledger: `/tmp/h3-8d-checkpoint.md`
- D0 artifacts: `/tmp/h3-8d0`
- D1 artifacts: `/tmp/h3-8d1`
- D2 accepted results: `/tmp/h3-8d2/extracted-final/results`
- D2 complete accepted run: `/tmp/h3-8d2/run-prepared`
- D2 source/binary manifest: `/tmp/h3-8d2/accepted-artifacts.sha256`
- Prior/final artifact preservation check: `/tmp/h3-8d2/final-preservation.log`
- Detailed failed attempts: `/tmp/h3-8d2/first-failures.md`

The D2 report records exact header, bitstream, package, driver and final
software hashes. The new FPGA image, runtime slot and rootfs are separate
from baseline artifacts. Guest validation and simulator both exited 0;
the guest is shut down.

First failures remain preserved: configuration-variable mistakes, the
native-only smoke target, old drain-fixture encoding, launcher/path errors,
missing test generation pointer, all numerically correct zero-overlap
attempts, and disk-full result copying after successful bit generation.
The last was recovered without resynthesis: the new incomplete duplicate
is retained in temporary `/dev/shm/h3-8d2-incomplete-results`, while the
complete authoritative Vivado build remains on disk. Earlier artifacts
were not removed. Hardware failures blocked subsequent benchmarks until
their gate passed.

## Decision

Retain H2 default and both explicit opt-ins. D1 provides a measured
improvement over H2. D2 provides safe prefetch and a physical overlap
witness, but its added checked/raw-submission work does not improve these
two workloads over K-tile. No further tuning or default promotion is part
of this completed task.
