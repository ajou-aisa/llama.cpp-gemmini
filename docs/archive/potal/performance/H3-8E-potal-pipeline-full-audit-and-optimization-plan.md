# H3-8E: PoTal pipeline audit and next optimization plan

Date: 2026-10-29.

## 1. Conclusion and evidence rules

The D0/D1/D2 implementation, tests and reports are now committed. This audit
changes no production algorithm, RTL, ISA, default, model, image or numerical
contract. No push, new FPGA experiment, full inference run or synthesis occurred.

The present Prefetch is **late next-K prefetch**, not a Main-K0/B1 pipeline.
Software reserves the next pair before the last Residual call, but submits
CONFIG_LD/MVIN only after that call has submitted its LOOP_WS. LoopMatmul then
holds raw commands until its configured generators retire. This is later than
CPU acceptance but does not necessarily wait for physical Execute completion.
The first subsequent global drain also includes the future B load.

The measured regression is real; its exclusive attribution is not established.
Extra raw commands, managed live-Scale checks, and repeated managed completion
work are source facts. Their individual contribution to +1.015/+3.741 ms is
**Not Measured**. DMA admission timestamps, workload-specific MAC intersections,
and next-GEMM advancement are also **Not Measured** for Decode/FFN.

**Preferred next implementation:** reduce redundant *host-side managed
completion work* at already-qualified drain points, retaining every checked
Release command and response. This is a bounded, existing-image experiment.
**Alternative:** explicit prepared independent-B admission integrated with the
loop scheduler, paired with a scoped C11 pending-state contract. It needs RTL
work and may need a capability-gated ISA descriptor; it is not a raw bypass.

Labels used throughout:

- **Fact:** established by the named committed source or recorded result.
- **Inference/proposal:** a consequence or possible implementation, not tested.
- **Not Measured:** existing records cannot answer it.
- Time fields are guest CPU-observed intervals, not pure accelerator time.
  Command-declared bytes are not measured bus traffic.

Path shorthand, all references at the following audited commits:

- `S/`: software repository root; `P/`: `S/ggml/src/ggml-potal/`.
- `G/`: `S/ggml/src/ggml-gemmini/`, shared software helpers, not proof of
  Gemmini-backend delegation.
- `R/`: Chipyard `generators/potal/src/main/scala/potal/`.
- `RK/`: Chipyard `generators/rocket-chip/src/main/scala/rocket/`.
- `L/`: `/tmp/h3-8d2/extracted-final/results/`.

## 2. Commits, inclusion, exclusions and preservation

Both repositories were on `potal`, with the requested roots/remotes and
checkpoint HEADs. There were no preexisting staged changes.

| Repository / content | Commit |
|---|---|
| Software H3-8 checkpoint | `df8482e9fec038dd1f5db48c8b3ab9c1c11c3df9` |
| Software D0 report | `9b0cfc030690132951eda231d0fdf6d131ebf648` |
| Software D1/D2 final implementation and reports | `331ada9c3d8a1b19e16291fe18b7b858012f2081` |
| RTL H3-8 checkpoint | `8c839ea348a42398da6822440209955b3a015ff8` |
| RTL D2 observation/configuration/regression | `bf1d98ca8a291a53845648ae475c80e01fd0b83d` |

D0 is independent and was committed separately. D1 and D2 share executor
methods, opt-in propagation, request/stat structures and numerical fixtures;
D2 depends on D1 Scale lifetime. A single final-state software commit avoids
inventing an unverified intermediate index tree. This coupling is recorded in
the commit message. No D1-only RTL change existed.

### Exact included files

Software D0:

```text
H3-8D0-matched-performance-baseline.md
```

Software D1/D2:

```text
H3-8D1-ktile-paired-implementation.md
H3-8D2-next-k-prefetch-implementation.md
H3-8D-final-report.md
ggml/include/ggml-potal.h
ggml/src/ggml-potal/include/potal.h
ggml/src/ggml-potal/include/potal_rocc.h
ggml/src/ggml-potal/matmul/options.hpp
ggml/src/ggml-potal/matmul/paired-auto.cpp
ggml/src/ggml-potal/matmul/paired-auto.hpp
ggml/src/ggml-potal/matmul/stripe.cpp
ggml/src/ggml-potal/matmul/stripe.hpp
ggml/src/ggml-potal/ops.cpp
ggml/src/ggml-potal/potal.c
ggml/src/ggml-potal/tests/potal-api-test.c
ggml/src/ggml-potal/tests/potal-paired-execution-test.cpp
ggml/src/ggml-potal/tests/potal-paired-smoke.cpp
ggml/src/ggml-potal/tests/support/paired-h2-emulator.hpp
```

RTL:

```text
build.sbt
generators/chipyard/src/main/scala/config/RoCCAcceleratorConfigs.scala
generators/firechip/src/main/scala/TargetConfigs.scala
generators/potal/src/main/scala/potal/Configs.scala
generators/potal/src/main/scala/potal/Controller.scala
generators/potal/src/main/scala/potal/CounterFile.scala
generators/potal/src/main/scala/potal/ExecuteController.scala
generators/potal/src/main/scala/potal/Mesh.scala
generators/potal/src/main/scala/potal/MeshWithDelays.scala
generators/potal/src/main/scala/potal/PE.scala
generators/potal/src/main/scala/potal/PoTalConfigs.scala
generators/potal/src/main/scala/potal/PoTalResidentView.scala
generators/potal/src/main/scala/potal/Tile.scala
generators/potal/src/test/scala/potal/PoTalDrainHandshakeSpec.scala
generators/potal/src/test/scala/potal/PoTalPrefetchOverlapSpec.scala
```

### Exact remaining dirty state and exclusions

Software after implementation commits: untracked `models/gpt2/` only.
This H3-8E report is a new documentation deliverable, not a production edit.

RTL preserves these incoming changes:

```text
 m generators/gemmini
 M generators/potal/software/host-init.sh
 D generators/potal/software/overlay/root/run-test-smoke.sh
 D software/firemarshal/runOutput/.gitignore
 M tools/cde/target/streams/compile/packageBin/_global/streams/inputs
?? generators/potal/software/overlay/root/Makefile
?? generators/potal/software/overlay/root/config.status
?? test_run_dir/
```

Gemmini's gitlink/HEAD remains
`185293e87cbbc2a336410f49a3fd73d01ed8d04e`; only its dirty state remains.
No GGUF, rootfs, bitstream, Vivado/CMake/SBT output, historical log or temporary
result was staged or removed. D2 deployment YAML/JSON under the enclosing
`/home/alveo/firesim/deploy/` are outside the two requested repositories;
their existing paths are preserved, not silently committed to a third repo.
Scala target/configuration sources inside Chipyard are included above.

Every add named exact paths. Cached check/stat/name-status were inspected
before each commit. `/tmp/h3-8e/committed-state.log` records SHAs and status.
The original reports' statements that D1/D2 were uncommitted describe their
2026-10-29 creation state; this section supersedes that source-state claim
without rewriting historical results.

### Validation reused and checked

`/tmp/h3-8e/commit-gates.log` records existing native tests, 11/11 per DIM,
and successful SHA checks for accepted source patch, binaries, rootfs, UART
and reports. Reviewed software and scoped RTL diffs byte-match accepted D2
patch artifacts. Bit/MCS/package/timing manifest checks also pass.
No unchanged expensive hardware validation was rerun.

Historical acceptance remains:

- D0 matched DIM16/DIM32 actual FPGA baseline, 168 exact primary samples.
- D1 FPGA numerical/attention/range gates and 84 exact primary samples.
- D2 native 11/11 per DIM; targeted RTL observation 4/4, DIM32 ingress 1/1,
  DIM32 SRAM/writeback/ROB drain 3/3; FPGA raw goldens 9/9; API,
  attention and PV-range pass; 84 primary and 36 separate-profile samples.
- The old reduced-cache drain fixture still fails 3/3 because of mismatched
  key offsets. No full-suite RTL PASS is claimed.
- Final D2 route timing and the 103-cycle fixture witness are retained.
  Earlier numerical failures and zero-overlap attempts remain preserved.
- Prior full-model inference is historical checkpoint evidence only.
  Neither this audit nor D0/D1/D2 reruns full GPT-2 inference.

## 3. Production frontend and three complete call graphs

### Shared frontend and production H2

```text
GGML backend registry [GGML_USE_POTAL]
  ggml_backend_potal_reg / POTAL device
    supports_op -> check_op -> attention-role gate OR HP1/I8 eligibility
    potal_graph_compute [GGML caller thread]
      preflight entire submitted graph; reject/poison before hardware
      compute_mul_mat
        HP1: mul_mat_hp1
          frozen Main tile selection
          weight_cache::get -> prepare_weight -> hp1::unpack [CPU B/carriers]
          mul_mat_hp1_pipeline -> run_stripe_pipeline
            producer: ExSIA local adaptation, fold, seal RMD packet
                      -> submit_stripe -> owned READY slot
            consumer: issuer_owner task on Linux CPU mapped to Hart0
                      stripe_run::execute
                      -> decode residual once
                      -> potal_paired_build_execution_plan
                      -> owner_runners.paired
                      -> issuer::execute_paired
                      -> run_paired_on_owner
                      -> potal_paired_run_session
                      -> potal_paired_matmul_auto
                         validate supplied plan; stage residual CPU buffers
                         Scale/Index + Main/Residual checked LOOP_WS
                         readbacks + checked completion
                      -> INT64 radix reconstruction -> private F32 merge
            producer finish / issuer join
          publish node output only after successful stream completion
```

Sources: `S/ggml/src/ggml-backend-reg.cpp:195-196`;
`P/ggml-potal.cpp:130-180,244-288,344-374`;
`P/ops.cpp:272-309,430-547,668-707`;
`P/matmul/stripe.cpp:695-840,1054-1140,1174-1280`;
`P/matmul/execution.cpp:498-504,717-782`;
`P/matmul/paired-auto.cpp:2390-2595`.

The scheduler's assignment is distinct from backend execution. Presence of
shared `ggml::gemmini` quantization/logging helpers is not delegation to a
Gemmini hardware backend. Direct unsupported graph submissions are refused,
not secretly executed on CPU.

### K-tile explicit branch

```text
same registry / HP1 weight cache / ExSIA / stripe FIFO / Hart0 issuer
  GGML_POTAL_PAIRED_KTILE=ON (requires Paired, DIM32, compensation)
    same validated H2 tile/window plan and arithmetic
    executor::prepare_scale_reuse
      prove consecutive physical fragments cover Main Scale subrange
      alias Residual Scale to live Main generation where proven
      otherwise retain ordinary Residual allocation
    Main -> Index -> Residual as H2
      aliased scale_program: no duplicate Scale Write
      aliased scale_release: real checked Fence, NOT early free
    same readback -> INT64 reconstruction -> F32 merge -> slot release
```

Sources: `P/ops.cpp:229-257,282-309`;
`P/matmul/stripe.cpp:790,1200-1220`;
`P/matmul/paired-auto.cpp:1768-1793,1993-2069`.
K-tile here means **Scale-window reuse in the H2 schedule**, not a new
microtile scheduler or changed multiplication order.

### Prefetch explicit branch

```text
same frontend / quantization / plan / CPU residual staging
  GGML_POTAL_PAIRED_PREFETCH=ON requires K-tile=ON
    idle potal_paired_h2_begin(original immutable B allocation)
      publish CPU B stores once; bind owner
    declare current panel -> checked Main/Residual readers
    before last Residual Kp:
      declare next Kp+1 on opposite, already-retired pair (no DMA)
    issue last Residual Kp:
      checked LOOP_WS with A CPU publication
      immediately panel_prefetch(next reference)
        CONFIG_LD state0 + ordinary tile MVINs
        no clock query between current LOOP_WS and future B submissions
    checked Fence / Scale Release:
      globally drain current work AND issued future B load
      publish next B; complete issued readers
    retire current identity; retain next identity
    next Main descriptor still has real B pointer
      checked encoder; suppress only encoded B address after publication
    same readback / reconstruction / merge / safe close
```

Sources: `P/matmul/paired-auto.cpp:1795-1850,2074-2110,2274-2344`;
`P/potal.c:1754-1905,1907-1978,1989-2088`.
Default flags remain false; malformed or unsupported explicit opt-ins fail
without fallback. Ordinary H2's capacity fallback is a different, recorded
path (`P/matmul/stripe.cpp:713-766`).

## 4. Preparation, buffers, threads and numerical contracts

| Object/stage | Actual owner and storage | When ready / reusable |
|---|---|---|
| HP1 weights | GGML caller, host-memory Q8_HP1 -> aligned CPU int8 B, K32 carriers, per-output scale | unpack complete; borrowed B immutable until checked close, quarantined on unknown DMA |
| Weight cache | backend mutex; key buffer/generation/data/type/all extents/strides | eligible WEIGHTS buffer only; cache budget insertion, no LRU eviction |
| ExSIA workspace | producer or joined OpenMP workers, CPU integer/exponent/mask arrays | local sealed then ordered fold, packet seal, callback; distinct from consumer slots |
| Main activation | CPU stripe slot, padded aligned int8 A | copied before READY; actual SP load later inside Main LOOP_WS |
| Residual packet | shared immutable sparse K32 runs/lane rows/digits | sealed before callback; decoded once on issuer |
| Paired residual staging | issuer-owned aligned CPU compact DIM fragments | all windows packed before effects; actual SP load later inside each Residual LOOP_WS |
| Main result | ACC0 then aligned CPU INT32 C | DMA readback and qualified completion before CPU use |
| Residual result | ACC1 lane-row epochs then CPU INT32 C | readback before epoch overwrite; checked INT64 radix reconstruction |
| F32 output | private pipeline buffer | merge completes before slot FREE; entire node published after finish/join |

Sources: `P/matmul/weight_cache.cpp:19-29,51-147`;
`P/matmul/dense.cpp:100-164,262-268`;
`P/matmul/stripe.cpp:155-190,394-575,1054-1140`;
`P/matmul/paired-auto.cpp:1533-1614`.

The default cache budget is 1024 MiB. An over-budget entry is bypassed, not
evicted by an LRU policy. On fatal uncertainty, abandon deliberately avoids
freeing potential DMA storage. Warm CPU-prepared B is not resident SP B.

### ExSIA, adaptation, folding and packets

`G/quants/act/exsia/exsia.cpp` performs local adaptation and alignment,
then stripe exponent reduction, folding and residual capture, then seals
the packet and invokes the stripe callback. Relevant seams:
`183-205` packet seal, `1340-1421` local work,
`1470-1493,1553,1637-1703` ordered folding/publication.
The RMD handle is `shared_ptr<const StripePacket>`
(`G/residual/rmd/rmd-types.hpp:186`).

The weighted call segment is `quants::quantize_activation` -> `act::quantize`
-> `exsia::ExSIA::run`.
`G/quants/act/quantize.cpp:21-73`; `G/quants/act/dispatch.cpp`;
`G/quants/act/exsia/exsia.cpp:293-297`.
PoTal geometry supplies activation stripe rows; the dispatcher validates
geometry and tensor shape before invoking the algorithm.
`LocalStage::run_optimized` produces local INT32 values, exponents and masks
(`G/quants/act/exsia/local.cpp`). `StripeFolding::run` constructs
folded Main bytes and residual payload, including rounded/clamped integer
shifts and stripe exponent alignment
(`G/quants/act/exsia/folding.cpp:74-169`).
The active stripe pipeline's capture branch uses `RmdBitmapBuilder`, not
every residual builder present in the repository
(`G/residual/residual-capture.hpp:62-98`).
Monotonic row/K coordinates, balanced signed radix decomposition and masks
are enforced in `G/residual/rmd/rmd-bitmap-builder.cpp:23-49,158-209`.

Sequential mode runs on the node producer. LocalParallel joins local work
before ordered folding/callback. LocalFoldingPipeline uses local-sealed and
post-chain dependencies. ExSIA's two workspaces are not the stripe
consumer's capacity-one/two slots; callback copies are what permit upstream
workspace reuse while the issuer still holds its own slot.

Main is folded signed int8 plus carriers; residual digits represent
quantization/folding differences by K32 block, original row and radix lane.
Near INT32_MAX, lane zero can be nonzero; do not assume residual starts at
lane one. Compact runs cannot be reordered or merged into an ideal wide
dot: physical DIM partial saturation and ascending INT32 ACC saturation are
part of the contract. Reconstruction is checked INT64, not a floating
residual approximation. See `P/residual/stripe-residual.hpp:67-128`,
`P/matmul/paired-auto.cpp:1533-1614`,
`P/matmul/stripe.cpp:1054-1140`.

### Issuer and queue blocking

The single issuer maps Hart0 to a Linux CPU rather than assuming CPU0.
Affinity is set and verified; direct historical API tests used CPU1 because
that guest mapped CPU1 to Hart0. The producer excludes the issuer CPU.
`P/matmul/execution.cpp:115-175,320-351,626-675`.

The submit mutex spans an owner task/stripe stream until join; the FIFO
mutex/CVs protect slot transitions. There is no new per-K-window mutex
handoff in D2. Capacity waits are CPU queue waits, not ACC completion.
`P/matmul/execution.cpp:184-315,717-782`;
`P/matmul/stripe.cpp:394-575`.

## 5. Physical dataflow and residency

The proposed diagram needs one correction: CPU Main/Residual preparation
does **not** mean both activations are already staged in scratchpad.

```text
 CPU producer / node caller                  CPU issuer (Hart0)
 --------------------------                 -------------------
 F32 X -> ExSIA -> folded Main int8 ----------> owned Main CPU A
              \-> sparse RMD packet ----------> decoded compact CPU Ares
 Q8_HP1 W -> unpack/cache --------------------> immutable CPU B/carriers
                                                   |
                       checked Main LOOP_WS        | actual generated MVIN
                                                   v
 Pair p: SP bank 2p = A bank        SP bank 2p+1 = B bank
 +---------------------------+     +---------------------------+
 | Main A: front             |     | rear-packed original B    |
 | Residual A: after reserved|     | one (J group,K panel)     |
 | maximum Main A footprint |     | shared by all its readers |
 +---------------------------+     +---------------------------+
          | Main/Residual A reads          | indexed or dense B reads
          +-------------------------------+
                          v
                 WS preload / mesh / PE
                          |
              SCU: live Scale/generation/carrier
                          |
           +--------------+----------------+
           v                               v
   ACC0 Main, K panels ordered      ACC1 compact lane-row epoch
           |                               |
           +--- MVOUT after dependencies --+
                          |
              CPU INT32 readback + completion
                          |
              checked INT64 radix reconstruction
                          |
                    private F32 merge
```

With physical DIM D, `pair=r%2`, `r=j_group*k_panels+panel`:

```text
A_base       = (2*pair)*BANK_ROWS
B_end        = (2*pair+2)*BANK_ROWS
B_base       = B_end - Ktiles*Jtiles*D
B tile row   = B_base + (k*Jtiles+j)*D
B CPU source = original_B + (k_begin+k*D)*ldb + j_begin+j*D
Ares offset  = Main_I_tiles * maximum_K_tiles (tiles, within A bank)
ACC0 offset  = Main output tile position
ACC1 offset  = compact lane-row output tile position within its bank half
```

Main A is loaded for each Main loop; Residual A is loaded for each indexed
window. Indexed Residual does not reload B: Index resolves original
block/local-K coordinates against the same B residency.
Across K panels or J groups, B is a different residency and is loaded again.
D2 changes *when/how* eligible next-K B loads occur, not this identity.

Sources: `P/matmul/paired-auto.hpp:250-330`;
`P/matmul/paired-auto.cpp:281-524,795-878,2113-2238`;
`R/LoopMatmul.scala:105-172,175-256,1167-1211`;
`R/ReservationStation.scala:250-310`;
`R/ExecuteController.scala:845-963`.

Capacity is unchanged: DIM32 SP4x2048, ACC2x256 rows, Scale256, Index64.
The planner reserves Residual A, bounds B to one bank, and checks both ACC
halves. Index64 holds at most two full DIM32 fragments per window.
K is the minimum of original Main K, weight-bank, Main-plus-Residual-A and
Scale limits. D1/D2 do not change these tile decisions.

## 6. K-window command timeline: logical plan versus actual issue

`P/matmul/paired-auto.cpp:740-1030` builds ordered operations. The executor
does not issue DMA for logical `load_main_a`, `load_weight`,
`load_residual_a`, `acc_init` or `pair_retire` markers; actual work is carried
by loop descriptors and checked completion (`2265-2344`).

```text
CPU preparation, once before execution:
  decode packet -> build/validate plan -> pack every Residual CPU window

For each J group, K panel p ascending:
  retire receipt for old same-pair residency r-2, if any
  program Main Scale columns with live generation
  MAIN LOOP_WS: dense A + B (or published prefetched B), ACC0
    first K panel overwrites ACC0; later panels accumulate
    previous ACC0 readback completes before a new output epoch
  For each compact Residual window:
    if reusing ACC1, finish its prior readback first
    Index Config + ordered Index Writes
    Residual Scale program, unless D1 proves a live Main alias
    INDEXED RESIDUAL LOOP_WS: Ares DMA + resident B lookup, ACC1
    readback this completed logical lane-row result
    Residual Release; alias case executes checked Fence instead
  after final K panel: Main ACC0 readback
  Main Scale Release (all valid columns)
  record current pair retirement
  next panel / next J group
final checked Fence -> safe close -> reconstruction/merge consumers
```

The arrows describe dependency/order, not equal-duration intervals.
Index and Scale command handshakes precede the loop; A SRAM arrival is
inside loop execution. Actual generated load-A/load-B/execute scheduling is
not forced into the simple CPU preparation order.

### D1 Scale reuse

Consecutive physical Residual fragments can alias the matching Main Scale
subrange and generation. Gaps or incompatible windows retain separate
program/release. The proof is more than equal carrier values. At an aliased
release boundary, a checked global Fence preserves drain semantics. Main
columns remain live until their real Release.
`P/matmul/paired-auto.cpp:1768-1793,1993-2069`.

### D2 insertion and its true overlap window

```text
Hart0       prepare B1 reference       issue last Residual K0      issue B1 CONFIG/MVIN
                 |                         |                            |
LoopMatmul       |                 generate K0 commands ----------------| blocked
                 |                                      configured clears
RS/Execute       |                 queued/current K0 work ----------- tail
DMA              |                                               B1 request/response/write
                                                                    possible overlap
Release/Fence    |                                                   global wait for BOTH
software         |                                                   publish B1, retire B0
K1 Main          |                                                               A1 load + compute
```

Prefetch is not issued during Main K0 in the current executor. It follows
the last Residual LOOP_WS. If there are no Residual readers or no next K
panel, this insertion does not run. It does not cross J groups.
`P/matmul/paired-auto.cpp:2074-2110`.

Even successful overlap does not imply next Main starts earlier: its Scale,
Index/ACC and software dependencies remain, and the global drain includes B1.

## 7. Sequential versus Prefetch: same math, different commands

Same plans, Main/Residual loop counts, compact fragment order, Index writes,
Scale programming, readbacks and logical drain boundaries are established
for the measured K-tile/Prefetch comparison. There is no new matmul tiling.
There **is** extra CPU/C11 work and a different B-load command shape.

| Steady count | Decode K-tile / Prefetch | FFN K-tile / Prefetch |
|---|---:|---:|
| Main LOOP_WS | 12 / 12 | 48 / 48 |
| Residual LOOP_WS = Index Config | 60 / 60 | 288 / 288 |
| Index Write | 1830 / 1830 | 11496 / 11496 |
| Scale Write = Release | 19968 / 19968 | 73728 / 73728 |
| Readback commands / tiles | 12/48 / 12/48 | 48/192 / 48/192 |
| Explicit Fence / logical drain boundaries | 55/73 / 55/73 | 289/337 / 289/337 |
| Standalone prefetch panels | 0 / 6 | 0 / 24 |
| Raw B tile MVIN attempts | 0 / 192 | 0 / 768 |
| Main loops with encoded B suppressed | 0 / 6 | 0 / 24 |
| Prefetch command-declared bytes | 0 / 196608 | 0 / 786432 |

Sources: `L/<case>/bench.out`, every steady STATS record;
`P/tests/potal-paired-smoke.cpp:326-347`;
`P/potal.c:1824-1848`.

D2 adds 6+192=198 and 24+768=792 CPU-submitted CONFIG/MVIN commands.
Baseline internal LoopMatmul B generator commands are not included in these
CPU counters. Therefore these are not total accelerator-command deltas.
Ordinary generated loads can group B columns using block stride/repetitions;
D2 submits one DIM tile per raw MVIN. Equal logical B coverage does not prove
equal transaction count, TL packetization, byte lanes, row replay or bus time.
`R/LoopMatmul.scala:175-256,1167-1211`;
`R/LoadController.scala:85-181`; `R/DMA.scala:180-210,224-305`.

## 8. C11, RoCC and hardware blocking boundaries

RoCC funct7 values are fixed in `P/include/potal.h:357-374`:
CONFIG0, MVIN2, MVOUT3, FLUSH7, LOOP_WS8, loop configurations9..13,
Scale Write23, Release24, Fence25, Index Write26, Index Config27,
resident view28, counter126. An ordinary loop program issues checked
configuration words then LOOP_WS; internal generated PRELOAD/COMPUTE
commands are a separate RTL stream.
`P/potal.c:1489-1510`; `P/include/potal_rocc.h`.

| Boundary | Acceptance/completion condition | What it does not establish |
|---|---|---|
| RoCC frontend | raw FIFO has space | LoadController admission |
| LoopConv passthrough | convolution unroller not configured | LoopMatmul ready |
| LoopMatmul raw output | no configured loop in either slot and no pending response | physical EX complete |
| RS allocate | free relevant entry | dependencies cleared |
| RS load issue | valid unissued load, dependency vectors clear, LoadController ready | DMA starts or reaches SRAM |
| LoadController | command FIFO/state, tracker, scratchpad DMA ready | TL response complete |
| DMA | TLB/alignment/transaction slots/TL A-ready then returned beats | all consumers finished |
| SRAM write | selected actual DMA port, mask, bank edge | entire panel published |
| Execute completion | final response plus PoTal writeback obligations | all global work drained |
| Fence/Release | real allDrained plus applicable qualified metadata | benchmark waiting attribution |

Evidence: `R/Controller.scala:222-313,553-614,735-825`;
`R/LoopConv.scala:1212-1255`;
`R/LoopMatmul.scala:868-904,1085-1094,1150-1152,1617-1641`;
`R/ReservationStation.scala:121,385-438,467-545`;
`R/LoadController.scala:42,103-181`;
`R/DMA.scala:156-166,218-305`;
`R/Scratchpad.scala:367-417,515-556`;
`R/ExecuteController.scala:1249-1257`;
`R/PoTalWriteback.scala:295-326`.

### Generator versus physical completion

LoopMatmul marks generator completion from idle after its final command
handshake. It clears `configured` after all relevant generators finish;
it does not wait for zero outstanding execution utilization.
Utilization feedback limits generation separately.
`R/LoopMatmul.scala:477-514,793-800,943-949,1244-1248,1617-1641`.
Thus late raw load overlap with the execution tail is structurally possible.

### RS hazards and shared resources

RS ranges are half-open with wrap semantics and SP/ACC space selectors.
MVIN has RAW/WAR dependencies against overlapping operands. CONFIG_LD is
not given a blanket Execute/store dependency. Indexed PRELOAD conservatively
covers the entire original B residency, not only selected compact rows.
`R/ReservationStation.scala:77-87,250-310,385-396`;
`R/LocalAddr.scala:41-78`.

Opposite pairs are disjoint ranges, so independent raw B loads need not
depend on current-pair A/B. This does not make DMA independent: reader,
TLB, transaction slots, BeatMerger and destination arbitration remain shared.
`R/DMA.scala:84-109,156-305`; `R/XactTracker.scala:67-84`;
`R/BeatMerger.scala:64-103`; `R/Scratchpad.scala:515-556`.
The present records do not show conservative RS range false-dependency
cycles for these workloads. Do not call them the measured bottleneck.

### Scale, SCU, accumulator and completion

Scale identity includes slot/generation/column. Index binds pair, original B
extent, logical K coordinates and active entries. SCU uses the live Scale
contract on partial sums. PoTalWriteback completion is delayed until raw
completion and scaled ACC writes are reconciled.
`R/PoTalScaleCommandIngress.scala`;
`R/PoTalScaleMemory.scala:77-191`;
`R/PoTalSCU.scala:11-62`;
`R/PoTalWriteback.scala:295-326`;
`R/AccumulatorMem.scala`.

The global drain includes raw/unrolled queues, loop generators, RS,
controllers, scratchpad/DMA, and PoTal writeback. Fence and Release cannot
be replaced by a MAC counter, elapsed time, CPU memory fence, or generator
idle. `R/Controller.scala:807-825`.

## 9. Scale Release and managed completion costs

**Fact:** the first K0 release boundary waits for already-issued B1 DMA as
well as K0 work. D2 adds no logical drain boundary versus K-tile, but expands
what that boundary waits for. This can move next-B work into K0's measured
completion interval; it does not by itself prove extra E2E delay.

Each successful per-column Scale Release:

1. Checks session, slot, generation and live column.
2. Issues a real status-returning Release command.
3. Clears only that live column after success.
4. Calls `paired_drained`.
5. In managed H2, calls `h2_completed`: CPU `fence rw,rw`, two-panel
   loading/readability/readers state update, and pending-state clear.

Sources: `P/potal.c:1351-1383,1186-1195,1725-1751`.
H2/K-tile do not execute the managed branch. Prefetch executes it after
19968/73728 releases plus 55/289 explicit Fences per steady graph:
**20023/74017 managed completion invocations**.

These are source/count-derived invocations, not measured CPU instruction
counts or stall cycles. Many later columns may have no new work to publish,
but proving redundancy requires an invariant over every producer of pending
work/control state, not a benchmark-dependent clear removal.

The read-only query itself obtains core status and scans Scale slots;
no query can publish completion. Reader issue checks every required live
Scale generation/column; close/retire scan current state. Exact costs and
cache effects are **Not Measured**.
`P/potal.c:1270-1296,1922-1957,1989-2010,2072-2088`.

## 10. Prefetch latency: measurements, causality and gaps

Preserved D1 comparison:

| Workload | H2 | K-tile | Reduction |
|---|---:|---:|---:|
| Decode | 17.061 ms | 13.820 ms | 18.996542% |
| FFN | 89.170 ms | 70.972 ms | 20.408209% |

D2 same-image comparison:

| Workload | K-tile | Prefetch | Difference of medians | Regression |
|---|---:|---:|---:|---:|
| Decode | 14.353 ms | 15.368 ms | +1.015 ms | 7.071692% |
| FFN | 73.123 ms | 76.864 ms | +3.741 ms | 5.116037% |

Matching phase/iteration before subtracting produces steady median paired
differences **+0.980 ms Decode / +3.722 ms FFN**. These are different
statistics, not corrections to the preserved route medians. Every matched
cold and steady difference is positive. Seven samples and rotating order
are descriptive evidence, not a broad population causal study.
`L/<case>/bench.out` SAMPLE/SUMMARY records;
`P/tests/potal-paired-smoke.cpp:389-425`.

Steady cache hits are one, misses zero, weight pack/allocation zero. The
regression is not explained by accidentally repacking B. CPU/cache/TLB or
DRAM warmness is not controlled by the term "cold": it means software
weight cache cleared, not every hardware cache cold.

| Hypothesis | Established | Not established |
|---|---|---|
| Prefetch starts too late | after final Residual LOOP_WS; raw gate waits generator retirement | exact cycles or recoverable time on Decode/FFN |
| RS false dependency | full indexed-B ranges exist; opposite pairs disjoint | blocking counts or false hazard in measured run |
| Global drain includes B1 | yes, source-qualified allDrained | whether it increases total critical path |
| More CPU command issue | +198/+792 raw CONFIG/MVIN submissions | instruction count, per-command stall or causal ms |
| Managed validation/completion | live checks plus 20023/74017 completion invocations | exclusive share of regression |
| CPU FENCE/clock | Rocket stalls FENCE on RoCC busy; VDSO uses fence r,r | remaining stall-cycle attribution |
| Queue/mutex | existing task/slot synchronization, not new per-K mutex | contention-time difference caused by D2 |
| DMA/PE overlap | 103 cycles in one fixture | Decode/FFN overlap or advanced next-start |

The final code keeps no clock query between current Residual submission and
next B issue. It publishes immutable B at idle bind. Per-loop A publication,
completion fences and other clock reads remain. Native testing cannot
validate Rocket FENCE stall behavior; source and RISC-V disassembly provide
the mechanism. `RK/RocketCore.scala:346-358`;
`P/potal.c:1489-1504,1777-1782`;
`P/matmul/paired-auto.cpp:2074-2110`;
`/tmp/h3-8d2/vdso-disassembly.txt`.

Separate profiled invocations cannot be joined to primary samples.
Within the profile process, route mapping is inferred from deterministic
call order and run IDs; SAMPLE records do not have run_id and profile
records do not have route/iteration. Buffered stdout/stderr is not a
timestamp join. Never add overlapping producer/consumer intervals or
differences of unrelated medians to reconstruct graph time.

Conditional envelopes, **not speedup predictions**:

- Matched Scale API interval increases about 0.798/2.703 ms; eliminating
  all of that difference would be a candidate-isolation scenario, not
  proof that it is redundant CPU work or wholly graph-critical.
- Recorded combined Residual+Prefetch API intervals are at most about
  0.084/0.377 ms in the primary samples. Deleting their entire interval
  could not explain all of the +1.015/+3.741 ms gap, even under the
  unrealistic assumption that all of it is removable and critical.
- Assigning the whole gap to managed completion would require about
  50.692/50.543 ns net graph-critical cost per invocation. This is a
  break-even calculation, not measured instruction latency.
- Assigning the whole gap to added raw commands would require about
  5.126/4.723 microseconds net per command, all other changes neutral.
- Ideal hardware overlap benefit is bounded by eligible B-transfer
  critical time and independent useful compute time. Neither is measured
  for Decode/FFN, so there is no defensible numeric hardware speedup bound.

## 11. What the 103-cycle witness proves

The separate opt-in observation image uses selectors45..51 for masked B0/B1
physical DMA SRAM writes, qualified pair0/pair1 PE00 MAC, the two opposite
pair intersections and their union. CPU CLOCK_MONOTONIC does not generate
these counts. Capability is exact `POBS`, query
`0x504f425300000000`, response `0x504f425301200007`.

Provenance follows actual buffered operands and stationary weights;
preload-only/garbage/padding/flush/empty-mask/zero-fill/same-pair/unsupported
paths cannot count as the required event.
`R/Controller.scala:670-674,842-878`;
`R/ExecuteController.scala:969-979,1019-1033,1125-1128`;
`R/MeshWithDelays.scala:172-191,209-224`;
`R/Mesh.scala:49-64`; `R/PE.scala:150-189`;
`R/CounterFile.scala:119-131,252-308`;
`R/PoTalResidentView.scala:37-60`.

In the M63/N129/K544, I2/J4/K4 fixture, Prefetch counts 31 MAC0/B1-write
cycles plus 72 MAC1/B0-write cycles, union103. H2/K-tile controls are zero;
small fixtures are also zero. All routes have the same aggregate
B-write1408/1312 and MAC5544/5166 counts in that fixture.
`L/observation.out:13-18`.

| Requested observation | Existing evidence |
|---|---|
| Direction/pair | yes, aggregate two directions |
| Specific K window/generation per event | Not Measured |
| First/last DMA request or SRAM activity | Not Measured |
| First/last qualified MAC per panel | Not Measured |
| DMA request-to-first-write latency | Not Measured |
| Next Main start advanced | Not Measured |
| Decode/FFN overlap | Not Measured |
| Bus bytes, occupancy, full-array MAC utilization | Not Measured |

Do not multiply 103 by prefetch counts or interpret it as cycles saved.
The event count spans a fixture run, not an E2E attribution.

## 12. Attention and legacy resident paths are separate

Attention dispatch precedes HP1 checks. QK/PV are role-tagged eligible F32
graphs, not ordinary Q8_HP1 weighted stripes. Role metadata reuse is not
backend delegation. `P/ops.cpp:668-707`; `P/matmul/attention.cpp`;
`S/ggml/include/ggml-gemmini.h:24-42`.

Q uses local adaptation/alignment, not full weighted ExSIA folding and RMD
packet capture. Each Q radix digit completes separate signed INT32 jobs
against K-low and K-high using original carriers:

```text
S_l = checked_INT64(INT32_job(Q_l,K_low) + 256*INT32_job(Q_l,K_high))
Y   = checked_INT64(sum_l 256^l*S_l)
```

Physical DIM partial/ACC saturation is preserved; no ideal wide-dot
replacement or software-reference fallback. Historical doubled-K tile
selection is adapted to original-K descriptors.
`P/matmul/attention.cpp:443-475,493-611`.

PV preserves legacy bias eligibility; bias-domain refusal can use certified
H3-7C bitplanes, retaining independent INT32 outputs, explicit restore/
readback completion and checked wider reconstruction. This path must not
be altered to optimize weighted B prefetch. Its historical PV-range result
is a separate numerical gate, not D2 overlap evidence.
See `P/matmul/attention.cpp` PV runner and
`P/tests/potal-attention-smoke.cpp` range witness.

The concrete path is `execute_hardware` -> `issuer_owner::start` ->
`execute` -> `execute_stripe`, using `runners.planned` and a `run_corrected`
callback. The ordinary planned runner reaches `potal_tiled_matmul_stripe`;
corrected PV uses checked paired/bias machinery separately. It does not call
`runners.paired` or D2's next-B API.
`P/matmul/attention.cpp:702-788`; `P/matmul/attention-hw.cpp:14-203`.
QK/PV eligibility checks role, F32 host-memory tensor layout, dimensions
and forbidden flash mode at `P/matmul/attention.cpp:626-663`.
Graph construction transposes/marks contiguous V for PV
(`S/src/llama-graph.cpp:1256-1279`).

PV `make_bias` at `P/matmul/attention.cpp:301-354` compares per-DIM SCU/ACC
behavior against whole-K host conversion. Nonnegative-V eligibility is
separate, not implied by a call-site claim.
`execute` selects legacy bias or range-certified `pv_bitplanes`
(`P/matmul/attention.cpp:361-440,702-767`).
The bitplane route proves every SCU contribution and ascending ACC prefix
fits INT32 before checked wider recombination; it does not erase saturation
differences by assuming algebraic cancellation.
`P/matmul/attention-hw.cpp:14-203` implements `run_corrected`, including ACC
bias initialization, checked execution/readback and failure cleanup.
All are outside the weighted HP1 Prefetch timing path.

E4C1 is an isolated acceptance/benchmark target, default
`POTAL_E4C1_TESTS=OFF`, with a test adapter replacing backend entry points.
It uses `tests/support/e4c1-sequential.cpp`, not the production paired
executor. `P/CMakeLists.txt:446-500`;
`P/tests/potal-e4c1-backend.cpp:1-13`.
Its DIM16 resident extension uses view funct28 and separate capability
`0x504f544102100007`; DIM32 D2 does not enable it.
`R/PoTalResidentView.scala:17-34`; `P/include/potal.h` resident section.
Do not restore microtile alternation as the production default or claim a
DIM32 E4C1 performance result.

## 13. Candidate comparison and priorities

### C1: idempotent managed completion bookkeeping — preferred

**Proposal:** after a successful real checked drain, avoid repeating host
publication/readers bookkeeping and its CPU memory fence when no relevant
effect has occurred since that certificate. Keep every Release instruction,
column/generation validation, status and live-bit clear. Do not replace
hardware allDrained or drop a Release to make a benchmark faster.

Required proof: enumerate all producers of pending B, program/readback/ACC
updates, control/Index transactions, failure, quarantine and session close;
bind a certificate to those effects, not merely "no GEMM since last call."
New MVIN, Index/Scale control effects, readback, bias, failed attempted
program and unknown completion must invalidate the necessary state.
First failed response remains sticky; no retry or implicit safe-close.
Source seams: `P/potal.c:1186-1195,1351-1383,1725-1751`.

This targets up to 20023/74017 invocations, but the safely redundant subset
and saved instructions/time must be measured. Existing FPGA image: **Yes**,
if only host bookkeeping changes. Scope: C11 + regression + counters.
Risk: lifetime/completion certification, not arithmetic. H2/K-tile kept on
the existing path. It is preferable because existing measurements and
source show substantial repeated work, while a new overlap window is not
needed to falsify the hypothesis.

### C2: prepared Scale/Index validation and encoding

**Proposal:** a checked window descriptor can prove invariant geometry once,
then retain live generation/column checks at the mutation boundary.
Do not cache liveness across Release, reuse or failure. Index reuse is
per-pair generation/shape/source manifest, not equality of a few entries.
D1 already removes proven duplicate Scale programming; D2 already prepares
B before issue. Repeating those changes is not a new optimization.

Scope SW/C11, existing image **Yes** for fewer CPU checks with identical
commands. Bulk hardware Release/Scale ISA is a separate, higher-risk option.
Effects on cycles are unmeasured. Validate complete command identity and
negative stale-generation/partial-write cases.

### A: B1 during Main K0

An earlier software call alone is insufficient. C11 currently refuses
Scale Write, Index setup and further loop issue while future B is pending;
ordinary loop A-publication CPU fences also wait on RoCC busy.
`P/potal.c:1446-1486,1907-1917`.
Prefetching before needed Residual Index/Scale work therefore either fails
or forces a global publication wait, defeating the intended pipeline.

Current ISA permits independent ordinary loads after generator retirement,
not raw bypass while configured. A safe earlier path needs a scoped pending
contract permitting only current-panel consumers, validated load-state
ordering, and proof that the next pair was retired. Source B and current A
must already be published without adding a busy-draining CPU fence between
them. ACC0/ACC1 order and compact/saturation order stay fixed.

Existing image: **not established** for a useful Main-overlap implementation.
Pure software tail-placement can be explored on it, but does not overcome
the raw gate. Benefit: larger possible overlap opportunity, no measured
numeric improvement. Risk: load state, lifetime, Scale/Index and completion.

### B: K-window pipeline with separate activation staging

```text
                 proposed, not present in the committed implementation
time ---------------------------------------------------------------->
CPU B bind     publish all immutable B
Pair0          load B0   Main K0   Residual K0     retire B0
Pair1                   load B1                  Main K1 Residual K1
A staging      publish A0/Ares0         prepare/publish A1/Ares1
ACC0           ordered K0 contribution ---------- ordered K1 contribution
ACC1           window epochs/readback ------------ next window epochs
```

Separate immutable weight preparation from writable activation publication.
Do not confuse front-loaded CPU staging with SP completion. Two pairs alone
do not provide additional ACC epochs or Index capacity. The current global
Release still joins pending future work; removing that join requires a new
completion contract, not a local readiness bit.

Scope: planner/executor + C11; likely admission RTL for early loads.
Existing image: **Unknown** until scoped readiness and CPU-fence placement
are proved. No microtile default. Expected DMA coverage unchanged; next
Main may start earlier only if dependency timestamps confirm it.

### D: scheduler-owned independent B admission — alternative

The narrow seam is LoopMatmul's output arbiter before Controller's unrolled
FIFO (`R/LoopMatmul.scala:934-940,1085-1094,1150-1152`). Existing LdB
generator already arbitrates there. Raw CONFIG_LD state0 injection while
the generator is active can corrupt assumptions about shared load state;
RS address independence alone does not protect configuration ordering.

Prefer a scheduler-owned prepared next-B descriptor with explicit permitted
pair/range and immutable source, arbitrated through the existing load path.
Two options must be distinguished:

- Preserve current ISA and recognize a narrowly ordered, validated
  CONFIG_LD/MVIN sequence: requires RTL proof of state stability, queues,
  no producer interleaving, and capacity. Feasibility is not yet proven.
- Add a capability-gated descriptor/doorbell expressing next-B intent:
  may simplify atomic ownership and scheduler ordering but changes ISA,
  requires new image, refusal on old images, and wider verification.

Neither option may bypass checked ownership, RS hazards, tracker completion
or Release allDrained. The alternative is appropriate only after timestamp
evidence shows a useful independent compute window lost to admission.

### Priority matrix

| Candidate | Evidence / removable work | Scope | Existing image | Correctness risk | Minimum validation | Performance status |
|---|---|---|---|---|---|---|
| C1 first | repeated qualified host completion work | C11 | Yes | stale completion, readback ordering | native faults, RISC-V assembly, FPGA paired A/B | unmeasured; conditional envelopes only |
| C2 next | repeated window encoding/validation | SW/C11 | Yes for identical commands | stale Scale/Index lifetime | exact stream + stale/partial negatives | unmeasured |
| D alternative | raw admission excluded while generators configured | RTL+C11, possibly ISA | No for changed admission | load-state ordering, hazards, ownership | controller/RS/DMA RTL + new FPGA witness | unmeasured |
| A after D evidence | Main computation offers earlier potential window | SW/C11 + possibly D | Unknown | pending-state and CPU-fence interaction | full dependency timeline | unmeasured |
| B later | whole-window pipeline, no microtile default | SW/C11/RTL | Unknown | ACC epoch and publication joins | window-tail saturation + fault matrix + FPGA | unmeasured |

H2/K-tile compatibility is mandatory for all candidates. Extra DMA/PE
intersection alone cannot establish E2E improvement.

## 14. Minimum next implementation and measurement plan

No step below was implemented or executed in this audit.

1. **Freeze the baseline.** Use the three commit SHAs above and accepted D2
   binary/image manifests. Record every new input hash and route selection.
2. **Implement only C1 first.** Write its pending-effect/certificate invariant
   before edits. Enumerate every producer/consumer/failure/cleanup path.
   Preserve exact raw command sequence and every Release response.
3. **Native before hardware.** Run both DIM suites once; add deterministic
   tests sized like existing API tests for each invalidation path, multiple
   Release columns, prepared future B, current/future partial issue, Index,
   ACC bias/readback, generation reuse, stale ref and quarantine.
   Compare exact H2/K-tile streams and independent raw/INT64/F32 results.
4. **Inspect RISC-V output.** Confirm which CPU fences/check loops were
   removed and retained. Compiler-visible instruction changes are not
   dynamic retired-instruction or stall counts.
5. **Same-image A/B.** Run required raw/attention/range gates, then matched
   Decode/FFN with unchanged seed and cache/route order. Separate profiling
   from primary timing; give records stable run/route/iteration/window IDs.
   Do not place VDSO time reads in the Residual-to-Prefetch issue gap.
6. **Falsify the hypothesis.** Measure completion-path invocations,
   certificate reuse and graph medians. If reducing this work does not
   reduce graph latency, retain the negative result; do not blame DMA by
   elimination without hardware timestamps.
7. **Admission diagnosis before D.** Minimal future RTL observation should
   latch, per qualified pair/generation: raw-front accept, generator
   passthrough, RS load issue/dependency stall, DMA request, first/last
   SRAM write, first/last useful MAC, global-drain request/response and next
   Main start. Read latched records after drain, not through polling in the
   critical interval. Existing seven counters cannot answer these questions.
8. **Only then evaluate D/A.** Use disjoint-pair positive and same-pair
   negative RTL tests, CONFIG_LD interleaving, Index/Scale busy, DMA
   backpressure/TLB misses, partial attempts, tails and saturation. New
   hardware requires an explicit separate image/capability. Keep old images.

### Artifact index and final scope

Main evidence:

```text
/tmp/h3-8e/commit-gates.log
/tmp/h3-8e/committed-state.log
/tmp/h3-8e/software.diff
/tmp/h3-8e/rtl.diff
/tmp/h3-8e/parent-evidence.md
/tmp/h3-8e/rtl-audit.md
/tmp/h3-8e/quant-attention-audit.md
/tmp/h3-8e/measurement-audit.md
/tmp/h3-8d2/extracted-final/results/
/tmp/h3-8d2/first-failures.md
/tmp/h3-8d2/accepted-artifacts.sha256
/tmp/h3-8d2/bitstream-verified.sha256
```

The listed working notes retain derivations and matched sample tables.
Some draft function names and source positions required correction against
the actual files; use this report's corrected citations as the audit record.
Its diagrams and decisions are self-contained. Existing rootfs, model, logs and bitstreams were not
modified by H3-8E. The only new project content after committing D0/D1/D2
is this audit report. No optimization or safety weakening was performed.
