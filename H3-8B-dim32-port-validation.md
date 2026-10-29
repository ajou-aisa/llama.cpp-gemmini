# H3-8B: DIM32 production port and validation

Date: 2026-10-28.

Status: local port, all required native gates and targeted RTL gates pass.
Full FireSim target elaboration, Golden Gate host-RTL generation and FPGA
driver build also pass. The isolated Vivado bitstream build completed with
exit 0 and positive routed setup/hold slack. Real-hardware numerical gates
and the exact 256/128 Functional inference subsequently pass; their
completed evidence is recorded in H3-8C, dated 2026-10-29.
Native command-emulator results below are software-only, not hardware PASS.

## Source and generated parameters

Starting software HEAD: `fc0bdcdd0dba8ccfef8a54cd2e3c7f39a0be3c9b`.
Starting RTL HEAD: `10ff4a10a87662edc7efd5d1a718027c1853550b`.
Both repositories are on `potal`. H3-8 changes remain uncommitted.
Existing incoming changes were not reverted or committed.

The actual RTL configuration regenerated
`/tmp/h3-8/gemmini_potal_d32_params.h`. Its bytes match the selected source
header and both native/RISC-V generated adapters:

```text
SHA256=3580b84f15ed8de9f9d311edea64eaa7c1f9bb308b1552a8f79453ec2c05ea33
DIM=32
XCUSTOM_ACC=3
BANK_NUM=4
BANK_ROWS=2048
ACC_ROWS=512
POTAL_SCALE_ENTRIES=256
POTAL_INDEX_ENTRIES=64
MAX_BYTES=64
elem_t=int8_t
acc_t=int32_t
```

`PoTalConfigs.a8w8D32Sat32` derives the rows from the existing 256 KiB
scratchpad and 64 KiB accumulator byte capacities. It retains two physical
ACC banks, each with 256 rows, and two sub-banks. The raw full-DIM partial
is signed 21-bit. The INT8 reduction endpoints are -520192 and 524288;
signed 20-bit would lose the positive endpoint.

## Hardware configuration

`generators/firechip/src/main/scala/TargetConfigs.scala` adds:

- `FireSimPoTalD32RocketConfig`.
- `FireSimQuadRocketSinglePoTalD32Config`.

The new configuration uses the existing
`chipyard.PoTalSat32D32GemminiRocketConfig`. It copies the verified DIM16
composition: three additional Rocket harts over the one-hart base,
`WithMultiRoCCFromBuildRoCC(0)`, the same default FireSim bridges, default
FASED memory model, and `WithFireSimConfigTweaks`.

The inherited clock settings are a single nominal 1000 MHz target domain.
They are not a claim about routed FPGA frequency. The default memory model
is the existing latency-pipe FASED configuration, not a newly introduced
DDR scheduler model.

## Production changes

1. **Dimension admission:** the backend and issuer admit DIM16 or DIM32,
   while retaining DIM64 rejection for the K32 production path.
2. **DMA ownership and alignment:** `matmul/dense.cpp` allocates at
   `max(MAX_BYTES, DIM*sizeof(int32_t))`, hence 128-byte alignment for
   DIM32 output/bias buffers. The issuer's permanent INT32 staging buffer
   uses the same full-row alignment.
3. **Checked C11 API:** the small legacy GEMM admits DIM32 only within its
   unchanged bounded shape contract: M=N=DIM and K=DIM or 2*DIM. General
   planned LOOP_WS, paired commands, Index transactions, liveness,
   generations, Fence/Drain and Release checks remain checked.
4. **Legacy I8 packing:** `matmul/pack.hpp` derives its physical tile from
   the selected header. Keeping TILE16 while admitting a DIM32 issuer
   would accept a smaller graph into a larger hardware staging layout.
   Eligibility, packing tests, device description and hardware smoke now
   agree on the selected DIM.
5. **Attention admission:** the H3-7C PV certificate admits DIM16/DIM32.
   Its existing physical-DIM loops, separate INT32 jobs and checked INT64
   reconstruction remain the production arithmetic.
6. **CMake tests:** testing is enabled before the PoTal backend subdirectory.
   Previously CTest could report success with no backend tests registered.
   Gate commands now use `--no-tests=error`.

No Micro-tile production integration, K-window Prefetch, quantization change,
compensation tolerance change, or scheduling optimization was added.
The separate opt-in resident API remains DIM16-only; it is not the active
production H2 route and was not silently enabled for DIM32.

## Capacity and lowering contracts

K32 remains the quantization/carrier block. The physical pass count is
`32/DIM`: two for DIM16 and one for DIM32. The serialized Residual still
pads each compact run to K32. The active paired Residual packs DIM-wide
fragments without crossing a K32 run.

Scale keys retain slot bits 7:0, generation bits 15:8 and column bits
21:16. DIM32 columns 0..31 fit. Scale allocation, consumer completion and
generation-qualified Release remain unchanged.

Index64 holds two DIM32 fragments per window. Active rows, filler entries
and window tails derive from the physical DIM. Index source coordinates
remain `original_k - panel.k_begin`, not absolute K.

Each pair owns one 2048-row activation bank and one 2048-row weight bank.
The two ACC banks hold Main and Residual separately; each 256-row bank
holds eight DIM32 output tiles. Main K remains bounded by activation,
resident-weight, Scale and ACC/work-ID constraints. Residual window
reservation does not authorize a different Main I/J plan.

Full-width readback uses ACC row `(i*J+j)*DIM`, bank base
`bank*(ACC_ROWS/2)`, and DIM*4-byte output alignment. Bias MVIN and
readback still require successful completion before storage reuse.

## Numerical invariants

The hardware arithmetic remains:

```text
partial = sum_DIM(signed_INT8_A * signed_INT8_B)
scaled  = ZERO ? 0 : Sat32(partial * 2^Delta)
ACC     = Sat32(previous_ACC + scaled)
```

DIM32 must not be pinned to DIM16 saturation results. Combining two
DIM16 reductions into one DIM32 reduction can move cancellation before
SCU saturation.

QK retains independently completed low/high INT32 jobs at the original
Delta, followed by checked `low + 256*high` and Q-radix reconstruction.
The high pass does not use Delta+8.

The independent destructive QK execution witness gives:

| Result | DIM16 | DIM32 |
|---|---:|---:|
| Raw low INT32 | 2147483647 | 2147483647 |
| Raw high INT32 | 1073872895 | 2147483647 |
| Reconstructed INT64 | 70927089860352 | 141287244103424 |
| F32 | 0.251983642578125 | 0.501953125 |

The obsolete shared/high+8 witness remains 549755813632 and is required to
differ. A separate all-negative-high reference witness has obsolete
results INT32_MIN for DIM16 and -1 for DIM32; the independent-pass result
is -547608330241 for both.

The PV range certificate checks every physical SCU interval and every
ascending-K ACC prefix before issuing any plane command. DIM32 binary
partials lie in [-4096,4064], inside signed 21-bit. Delta >=32 still needs
a proven-zero interval. Checked INT64 reconstruction and exact independent
quantized-dot comparison remain enabled.

Mixed-sign saturation fixtures place opposing terms in separate physical
microtiles. The range fixture retains the historical P peak at index 16
for both dimensions. DIM32's conservative interval still fits:
`64*2^24 = 2^30`. Keeping those inputs also preserves the original K32
command fixture rather than silently expanding it to K34.
The hardware fixture still requires 49 jobs, 16 plane jobs and 11 corrected
heads; full-width bias tiles change from 44 to 22.

## Verified local results

Native directory: `build-potal-h3-8-d32-host`.
RISC-V directory: `build-potal-h3-8-d32-functional`.

- Native production backend build: PASS.
- RISC-V production backend and `llama-cli` build: PASS.
- The final canonical Functional DIM32 wrapper builds its complete default
  target set, including all listed smoke binaries, `llama-cli`,
  `llama-perplexity` and `llama-quantize`: exit 0. Evidence:
  `/tmp/h3-8/canonical-d32-functional-build-final.log`.
- DIM32 C11 API: PASS.
- DIM32 attention C11 bias/ownership API: PASS.
- DIM32 Residual numerical unit test: PASS.
- DIM32 attention reference, command execution, QK/PV and PV extension: PASS.
- DIM32 GGML eligibility, I8 packing, scheduler and no-fake-hardware test: PASS.
- Older software paired-model planning fixture: PASS after correcting its
  two-stripe fixture geometry; no hardware execution is implied.
- RISC-V API, multicore, basic GGML and attention hardware smoke binaries:
  built successfully. The direct API, production Paired and both attention
  hardware gates now pass on the real DIM32 FPGA; H3-8C records their exact
  scope. The other built binaries are not claimed as executed.
- Existing generated-header contract: PASS for DIM16/32/64 acceptance,
  malformed inputs, isolation, header changes and expected build refusal.
  The final rerun after target-header binding changes also passes:
  `/tmp/h3-8/hw-params-contract-final.log`.
- All 11 selected production gates pass in both DIM32 and a fresh DIM16
  Functional build: C11, attention API, stripe API, attention, GGML unit,
  tiling, Residual, Stripe Pipeline, older paired-model planning, active
  H2 auto planning, and H2 command execution. The native route, ownership
  and no-fake-hardware assertions remain enabled.
- Six targeted RTL suites pass all 34 tests, including actual DIM32
  Scale/Index ingress, DIM16/DIM32 writeback, configuration, SCU,
  Scale-block reduction and wrap32/sat32 accumulator RMW.
  This does not imply hardware-execution acceptance.
- The actual quad-hart DIM32 FireSim target elaborates, passes Golden Gate
  host-RTL compilation and builds its `FireSim-xilinx_alveo_u280` driver.
  The isolated Vivado bitstream build completed with exit 0. Evidence:
  `/tmp/h3-8/firesim-d32-build-first.log`. These generation/build results
  are not DIM32 hardware execution.

Primary evidence:

```text
/tmp/h3-8/header-provenance.log
/tmp/h3-8/host-d32-build.log
/tmp/h3-8/cross-d32-build.log
/tmp/h3-8/host-d32-api-residual.log
/tmp/h3-8/host-d32-attention-fixed.log
/tmp/h3-8/host-d32-attention-registered.log
/tmp/h3-8/host-d32-paired-plan-fixed.log
/tmp/h3-8/cross-d32-api.log
/tmp/h3-8/cross-d32-attention.log
/tmp/h3-8/cross-d32-ggml-smoke-fixed.log
/tmp/h3-8/hw-params-contract.log
/tmp/h3-8/host-d32-production-pass.log
/tmp/h3-8/host-d16-production-pass.log
/tmp/h3-8/cross-api-main-indexed-final.log
/tmp/h3-8/host-d32-attention-original-inputs.log
/tmp/h3-8/host-d16-attention-original-inputs.log
/tmp/h3-8/cross-d32-attention-original-inputs.log
/tmp/h3-8/rtl-dim32-gates-verilator.log
```

## Preserved failures and limitations

The first RTL attempt failed compilation in two existing stale ingress
harness constructors. Their new Index capacity/geometry arguments were
wired without changing production RTL. First evidence remains
`/tmp/h3-8/rtl-targeted.log`.

The older reduced-cache Scale ingress fixture uses eight slots but emits
the fixed eight-bit production slot encoding. It fails with
INVALID_ENCODING, and also treats current Index opcode26 as unknown.
Those stale assertions were not weakened or disabled. An independent
DIM32 fixture uses the actual 256-slot/64-entry configuration and checks
Scale boundaries, both Index buffers, complete windows, bank bounds and
Fence/Release backpressure. That fixture passed 458 simulated cycles.
The complete six-suite run passed 34 tests, with no failures, cancellations,
ignored tests or pending tests. Evidence:
`/tmp/h3-8/rtl-dim32-gates-verilator.log`.

Initial native fixture failures, including a smaller-stripe-count crash
and invalid K indices beyond K32, remain in
`host-d32-core-initial.log` and `host-d32-paired-initial.log`.
The first Stripe Pipeline run was stopped after its obsolete event gate
waited for a nonexistent stripe; it is not a passing run.

A fresh DIM16 Functional build passed C11, attention API, attention and
GGML unit tests. Its tiling test exposed four existing profile-dependent
assumptions also present in the DIM32 run: two dequantized-output checks
and two fatal-cache expectations. Evidence:
`/tmp/h3-8/host-d16-preservation.log`. These are now resolved in the test
fixtures, without changing production semantics or tolerances:

- BLOCK/EXSIA dequantizers include captured cpu_direct residuals; compare
  their complete output after the captured correction. Tensor/Token/Stripe
  dequantizers omit those direct payloads, so their original Main-only
  comparison remains. Every Main result is independently checked first.
- A Paired no-hardware refusal before any command is recoverable and keeps
  the cached weight. The fixture checks zero issued work and the correct
  cache HIT. Other modes retain their fatal-abandon expectation.

The initial actual-geometry RTL run exhausted its default 1 GiB JVM heap:
`jcmd` recorded 1036091 KiB used out of 1048576 KiB. That run was stopped
and retained in `rtl-dim32-gates.log`; the heap evidence is in
`rtl-heap-diagnostic.log`. The full-cache interpreter remained costly with
an explicit 8 GiB heap. The actual-geometry ingress fixture now uses
installed Verilator 5.008; its cache size, generated RTL and assertions are
unchanged. The small arithmetic fixtures keep their existing backend.

## Ordered real-hardware gate fixture

`potal-api-smoke` now also executes checked Main LOOP_WS, programs every
Index entry with a reversed K mapping, reads ACC0, and checks every raw
Main INT32 against independent INT64 arithmetic. It then executes Indexed
Residual LOOP_WS on that resident B, reads ACC1 and checks every raw
Residual INT32 against an independently indexed dot product.

The fixture uses the actual pair-0 rear B address, two physical K passes,
full-width aligned output, successful explicit drains and balanced tracked
Scale leases. Its Main and Residual markers precede later Paired and
attention gates. It has no software sink and does not use the DIM16-only
resident extension. Both DIM16 and DIM32 versions compile. Actual DIM32
hardware execution is recorded separately in H3-8C, not inferred from
compilation.

Default C/C++ LSP diagnostics lack the generated include configuration.
Actual native/RISC-V compiler results are recorded separately. No physical
DIM32 DMA, completion, route timing or inference result is inferred from
these software checks.
