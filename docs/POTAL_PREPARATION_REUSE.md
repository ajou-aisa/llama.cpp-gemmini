# PoTal preparation and reconstruction reuse

This change targets the host C++ path feeding the FPGA PoTal backend, based on
`potal` commit `272802fa6439a939583b474ae7a91e5912adea4d`. It does not change CUDA
kernels or require an RTL change. Attention changes here apply to the W8
`ggml-potal` strict pipeline; they are not a port to `ggml-potal-w4a4`.
The shared residual packet builder supports both 4-bit and 8-bit digits.

## Changes and conditions

| Change | Enabled condition | Work removed |
| --- | --- | --- |
| QK reconstruction fusion | Strict QK jobs | Dense INT64 output/intermediate buffers and repeated passes over them |
| Shared Attention packing | More than one activation limb | Repeated stationary B/carrier allocation and copying across limbs; duplicate Q A packing for K-low/K-high |
| Output scale reuse | More than one row | Repeated `ldexp` for each output row; one exact power-of-two factor per column |
| Completed packet reuse | Explicit decode opt-in, one row, previous packet exclusively owned by builder | Packet/payload/group allocation; all current contents are still regenerated |
| Stationary packing across jobs | Multi-limb jobs in one strict Attention invocation, same KV head and batch | Repacking B/carriers for each stripe or grouped query head |
| Main conversion during residual merge | Linear stripe with a residual packet | Separate Main output write/read traversal before the checked residual merge |

Single-limb packing retains the original path. PV keeps limb-major
reconstruction; it does not use QK's element-major fusion. One-row PV output
conversion keeps its direct `ldexp` loop. Prefill packet pooling is disabled.
There is no new quantization approximation, anchor change, residual suppression,
GEMM removal, or change to the FP baseline.

## Numerical and FPGA ownership contracts

- QK retains low-pass then high-pass reconstruction for each limb, and increasing
  limb order for each output. Every checked INT64 reconstruction is still checked.
  The hardware GEMM order and INT32/SCU saturation behavior are unchanged.
- Scaling first converts INT64 to binary64 as before. Multiplication is used only
  for exponents in `[-1022, 960]`: the factor is an exactly representable normal
  power of two and any converted INT64 value remains finite in binary64. Other
  exponents use the original `ldexp`. The final binary32 conversion is unchanged.
  Floating-point exception flags are not part of the numerical equivalence claim.
- A job owns its packed stationary buffers until completion. QK pass 1 borrows
  pass 0's read-only activation allocation; every GEMM retains a separate C
  allocation. Shapes, padded layout, carrier stride, descriptors, planning,
  validation, command ordering, and fencing stay on the production path.
- Stationary packing is shared across jobs only inside that invocation. A KV
  head/batch change resets the producer's cache; pending jobs retain their own
  shared owners. This is not a persistent KV cache or cross-token quantization
  cache. Single-limb jobs retain their original path.
- The Linear merge keeps Main as `float(double(C) * activation_scale *
  double(weight_scale))` and the residual delta as `float(double(R) *
  double(weight_scale) * double(float(2^theta)))`, then adds those two floats.
  It does not combine integers before conversion or reassociate the factors.
  Residual radix reconstruction and every overflow check remain. Output and
  nonzero count commit only after every staged cell passes; slots are released
  after the same semantic-completion boundary. Trace span 43 now includes Main
  conversion in the merge traversal for residual stripes.
- Unknown DMA completion still abandons all job allocations, including shared
  stationary allocations. No shared DMA input is overwritten or freed on that
  failure path. Public output is committed only after successful completion.
- Packet reuse is opt-in and checks `shared_ptr::unique()`. A packet retained by
  any consumer is immutable; the builder allocates a different packet instead.
  The Linear stripe slot retains its packet until main/residual execution and
  checked merge complete. The pool reuses capacity, not cached numeric results.
  The benefit depends on the caller retaining the builder and releasing packets.

## Validation

Native Release build: GCC 13.3, DIM32, TOP3, A8/W8, PoTal hardware backend enabled,
paired path enabled. Hardware header SHA-256:
`3580b84f15ed8de9f9d311edea64eaa7c1f9bb308b1552a8f79453ec2c05ea33`.

- `potal-attention-test`: strict pipeline versus sequential execution, bitwise
  output, GEMM/Scale/Fence counts, GQA, decode/prefill, odd dimensions, large and
  subnormal output scales, failure publication, and an injected hardware Fence
  failure with unknown DMA state.
- `potal-stripe-pipeline-test`: real host pipeline with the existing accelerator
  model, including packet ownership and backpressure behavior.
- Follow-up integration reran both tests and `potal-residual-unit-test`. Added
  distinct two-batch/two-head Attention fixtures exercise cache reset and pending
  ownership. Fused merge tests compare with the previous separate production
  conversion/merge for strided outputs, cancellation, INT32 extremes, subnormal
  and large scales, invalid Main stride, and late overflow without partial commit.
- `test-gemmini-rmd-bitmap.cpp`: LOG_CYCLE=0 and 1, 4/8-bit exact packets versus
  the existing builder and independent reconstruction. Includes 60 changing
  decode/prefill/empty shapes and retained packets. LOG_CYCLE=1 also passed
  AddressSanitizer and UndefinedBehaviorSanitizer.
- Extracted old/current product bodies: 24 padded packing fixtures and 144
  reconstruction fixtures, including INT32 extrema, 1/3/5 limbs, INT64 overflow,
  and extreme output exponents. Successful output bits and failure statuses match.

The added high-output PV fixture keeps probabilities in their valid range and
scales V; an initial fixture scaling P was rejected by the unchanged reference
and was corrected before the final test run.

## Measurement scope

Nano3 measurements run on one Cortex-A78AE CPU (CPU5), GCC 13.3 `-O3`, without
fast-math or compiler loop/SLP vectorization. Synthetic inputs cover 1/32 rows,
1/3 limbs, QK K=127 with N=1023/4093, and PV K=1023/4093 with N=65.
Each condition uses nine alternating old/new pairs, with 80 repetitions for one
row and eight for 32 rows. Wall time and user-mode CPU cycles are separate metrics.

The stage replay extracts the product packing/reconstruction bodies. It excludes
the shared planner and preparation validator from **both** packing measurements;
they are retained in the product and exercised by the native pipeline tests.
It does not execute DMA or GEMMs. Earlier completed-packet replay measured roughly
24-39% fewer user cycles for one-row packets with no retained consumer; retained
packets showed no reliable gain, and prefill pooling was not selected.

Representative final-code medians (milliseconds; full conditions, including
unchanged single-limb controls, are in [the CSV](potal-preparation-reuse-nano.csv)):

| CPU stage | Rows / K / N / limbs | Old ms | New ms | Wall-time reduction |
| --- | --- | ---: | ---: | ---: |
| QK reconstruction + scaling | 32 / 127 / 4093 / 3 | 6.388 | 3.848 | 39.8% |
| QK packing | 32 / 127 / 4093 / 3 | 11.536 | 2.352 | 79.6% |
| PV reconstruction + scaling | 32 / 4093 / 65 / 3 | 0.0469 | 0.0342 | 27.1% |
| PV packing | 32 / 4093 / 65 / 3 | 2.900 | 0.874 | 69.8% |
| QK reconstruction, decode | 1 / 127 / 4093 / 3 | 0.199 | 0.147 | 26.1% |

Packing includes allocation/zeroing/copy/free and uses contiguous `copy_n` for
the shared stationary data. The C library can implement that copy efficiently;
its benefit on Rocket is not established by this Arm result. User-mode cycle
reductions in the CSV can differ substantially from wall-time reductions because
allocation also incurs kernel work. Do not add stage percentages together.

Local raw replay data, generator, and source hashes are retained under
`.omo/evidence/potal-preparation-pr-20261116/` in the project workspace; the CSV
above contains final-code paired medians, not FPGA data.

These are CPU-stage measurements, not CUDA timings and not Rocket/FPGA TTFT or
TPOT speedups. Native model tests do not establish actual RTL execution. No new
FireSim/FPGA run, RISC-V target build, or full-model PPL run was performed for this
PR. Target execution remains necessary before reporting end-to-end improvement.

## Additional integration measurement (2026-11-16)

The two follow-up changes were replayed from their final product bodies on Nano3
CPU5, with the same scalar compiler settings and seven alternating pairs. The
baseline here is the already optimized PR implementation at `5f85a6f`, not the
original `potal` implementation. Packing includes first allocation/packing and
destruction, with up to four jobs alive. Jobs use three limbs; QK has K=127,
N=4093 and PV has K=4093, N=65. Shared planning/validation remains excluded from
both replay sides and remains enabled in production.

| Stage / group | Before ms | After ms | Wall reduction | User-cycle reduction |
| --- | ---: | ---: | ---: | ---: |
| QK packing, 32 rows, 28 jobs | 32.900 | 22.263 | 32.3% | 49.2% |
| QK packing, 1 row, 28 jobs | 24.070 | 20.764 | 13.7% | 52.6% |
| PV packing, 32 rows, 28 jobs | 24.201 | 22.025 | 9.0% | 8.9% |
| PV packing, 1 row, 28 jobs | 3.659 | 1.535 | 58.0% | 58.0% |
| Linear Main conversion + merge, 32 x 4096 | 1.3104 | 1.1123 | 15.1% | 15.1% |
| Linear Main conversion + merge, 1 x 4096 | 0.04387 | 0.03712 | 15.4% | 15.3% |

Single-job packing controls were within 0.3% wall time. The Linear measurement
excludes unchanged residual radix reconstruction. Both replay correctness checks
passed on Nano and under local ASan/UBSan; native pipeline tests cover the full
production path. Raw pairs and extracted sources are retained under
`.omo/evidence/potal-followup-integration-20261116/` in the project workspace;
paired medians are in [the follow-up CSV](potal-followup-nano.csv).
These reductions cannot be added to the earlier stage percentages or reported
as end-to-end speedups. The FPGA/Rocket and full-model validation limits above
still apply. PV maximum fusion and general preparation-buffer pooling were not
promoted by this change.
