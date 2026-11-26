# H3-7B: PoTal Attention Offload Implementation

Date: 2026-10-26.

**A -- Implemented & Hardware Verified, within the supported DIM16 domain.**
QK and PV execute on PoTal, including shared-ACC K limbs and a physical INT32
PV bias addition. The actual GPT-2 16+8 request completed with 192 PoTal
Attention nodes and the same eight greedy tokens as the independent CPU
reference. This is not a claim of unrestricted PV support: an exact correction
outside INT32 is explicitly rejected; section 10 describes that remaining
hardware/interface limitation.

H3-7A was committed first:

```text
dcab70ff8a76348a531c3f89785a29f13a2e8413
test(potal): record H3-7A GPT-2 full inference baseline
```

Exactly the four requested H3-7A files were committed. Their report still says
B -- Partial Verification, CPU Attention, and incomplete long three-mode
comparison. No later project commit, push, branch, RTL change, or FPGA rebuild
was performed. H3-7B and H3-6E changes remain working-tree changes.

## 1. Gemmini Attention Reuse

Read the original `ggml/src/ggml-gemmini/matmul/attention.cpp`, its header,
dispatch and CMake integration, plus PoTal dispatch, execution, residual
lowering, and C11 APIs.

| Reused source/mechanism | PoTal treatment |
|---|---|
| `attention.cpp:45` F16/F32 stride-based reads | Ported into checked tensor/head/batch preparation |
| Dynamic stationary MXINT preparation | Same block32 exponent scanning/quantization classes, with runtime K or V |
| `quants/act/exsia/local.hpp` | Directly reused LocalStage, WideQuantizer, ExpScanner and scratch/state types |
| `quants/act/exsia/exsia_shift.hpp` | Directly reused Q stripe-folding shift semantics |
| `quants/common/math.hpp` signed low digit | Directly reused, with widened subtraction before division |
| Attention roles in `ggml-gemmini.h` | Reused `op_params[1]`; no tensor-name/shape heuristic |
| Head/batch repetition and per-column PoT bases | Ported and validated for positive, nonoverlapping strides |

The original `ScuGemm` is **not** called. Its IM2P provider requires DIM32;
its software implementation is not a PoTal hardware path. Original QK merges
independently completed K passes in Host INT64. Original PV adds its column
correction in Host INT64. Both execution choices differ from this request.

PoTal instead uses dense signed radix limbs, DIM16 SCU/ACC ordering, one
shared K-low/high accumulator per Q limb, and one physical final PV correction.
Q higher limbs are not replaced with linear GEMM's sparse-outlier path.

The user clarified PV upper-limb accumulation as **Host INT64 reconstruction**.
Accordingly, P upper signed limbs have separate INT32 jobs; they are not all
shifted into the low-limb ACC. That alternative has different saturation.

## 2. PoTal Backend Integration

New implementation:

```text
ggml/src/ggml-potal/matmul/attention.hpp
ggml/src/ggml-potal/matmul/attention.cpp
ggml/src/ggml-potal/matmul/attention-hw.cpp
```

Integration changes:

```text
ggml/include/ggml-potal.h
ggml/src/ggml-potal/ggml-potal.cpp
ggml/src/ggml-potal/ops.cpp
ggml/src/ggml-potal/ops.hpp
ggml/src/ggml-potal/CMakeLists.txt
src/CMakeLists.txt
src/llama-graph.cpp
```

`attention::check` accepts role-tagged MUL_MAT with F16/F32 operands and F32
output, compatible K/output dimensions, integral head/batch repeats, supported
precision parameters, host-visible memory, and nonoverlapping layouts.
It accepts permutations and row gaps, not just contiguous tensors. It
rejects unsupported types, parameters, shapes, overlap/extent overflow,
and nonhost memory. Runtime preparation rejects nonfinite data and invalid P.

The llama graph explicitly assigns both tagged QK/PV nodes to POTAL only
after `supports_op` accepts both. A requested supported Attention path cannot
silently select CPU. Backend execution errors return failure and poison logical
output elements; they do not invoke a CPU Attention implementation.

Q8_HP1 weighted MatMul remains a separate dispatch. Model loading, HP1 cache,
linear quantization, residual decode reuse, and existing Paired plans are
unchanged. Attention's fixed-ACC PV helper uses checked paired-session
primitives, not the paired-auto Main/Residual schedule or a new weight-reuse
optimization.

QK produces all raw logits. `src[2]` does not cause masked tiles to be skipped
or replaced. Existing CPU `ggml_soft_max_ext` subsequently applies the causal
mask, QK scale, and normalization. Softmax and Flash Attention are not claimed
as PoTal operations.

## 3. QK Implementation

The selected target is signed8/DIM16, with `rho=14`.

Q uses existing ExSIA local adaptation and stripe folding. Every folded
signed32 code is decomposed into signed radix256 limbs. Runtime K preparation
uses one exponent per token/channel block of 32, a per-output-column base,
and nonnegative SCU exponent differences. K codes clip to:

```text
[-32768, 32639]
code = signed_low + 256 * signed_high
```

For each Q limb, the actual operands are:

```text
A = [Q_limb, Q_limb]
B = [K_low; K_high]
carriers = [Delta, Delta+8]
```

Zero carriers retain the zero sentinel. Each original half has independent
zero padding to block32. The frozen K tile divides the original padded K and
contains whole carrier blocks, so no command crosses the low/high boundary.

The existing regular planned WS route traverses all lower K chunks before all
upper chunks for each output group. First contribution overwrites ACC;
subsequent chunks accumulate. C is NULL until the last upper chunk. Regular
RTL advances its physical ACC base only when that final store is requested.
Fences/Scale Release do not erase the partial ACC.

Therefore K-high uses SCU offset +8 **before** it is accumulated into the
same INT32 ACC as K-low. K-high is never reconstructed from a separate
completed K result on the host. Each completed Q-limb result alone enters
checked INT64 reconstruction.

Source: `ggml/src/ggml-potal/matmul/attention.cpp:380`,
`ggml/src/ggml-potal/matmul/tiling.cpp:401`, and
Chipyard `generators/potal/src/main/scala/potal/LoopMatmul.scala:1381`.

## 4. PV Implementation

P has no local adaptation or sparse-outlier selection. For each stripe:

```text
scale exponent = max_exponent(P) - 15
P_code = clip(RNE(P / 2^scale), 0, 65535)
```

The biased low digit and complete upper signed digits are retained. For example:

```text
65535 = 127 + 256*(-1) + 65536*1 + 128
```

Casting upper 255 to one signed8 digit would be wrong. The carry limb is
executed through the regular planned WS Residual job seam.

V is runtime signed8 MXINT with block32 carriers. Before any low-job command,
software prepares the exact already-scaled bias operand:

```text
B[j] = 128 * sum_blocks(2^Delta[block,j] * sum_real_k(V_code[k,j]))
```

Block sums and cross-block cancellation precede the final INT32 range check.
Exponent-indexed binary carry normalization supports cancellation even beyond
INT128 width. Individual shifted terms are not pre-clamped. The prescribed
final bias must fit INT32 for this hardware route.

For the low digit, `run_corrected` retains known physical ACC0 through all
K chunks, fences, then adds B exactly once per output lane using full-width
INT32 MVIN with ACC accumulate set. It fences again, performs full INT32
readback, and fences before CPU consumption. No SCU scale is reapplied to B.

Upper P limbs have their own INT32 jobs. Their radix reconstruction combines
with the already-corrected low result in checked Host INT64, then final PoT
scaling occurs. No result-side Host bias correction substitutes for MVIN.

The tempting all-ones correction GEMM was rejected. At carrier24, one +1 and
one -1 in a K32 block have exact bias zero. Two independently saturated
microtile correction updates instead produce -1. One correction panel is
not one atomic correction addition.

## 5. SCU and INT32 Accumulation Order

For each legal DIM16 raw partial:

```text
t = Sat32(raw_dot * 2^carrier)
acc = first ? t : Sat32(acc + t)
```

The signed20 raw array reduction fits a legal signed8 DIM16 dot:
maximum 262144, minimum -260096. A K32 block has two DIM16 partials.
It is not one raw K32 reduction or one final-only saturation.

QK applies every lower partial, then every upper partial with +8 SCU offset,
into one physical ACC. PV applies every biased-low partial, then:

```text
acc_low_corrected = Sat32(acc_low + B[j])
```

exactly once. Upper P limbs retain their separate-job saturation sequence.
Reassociation is forbidden: `Sat32(Sat32(INT32_MAX+1)-1)` differs from
`Sat32(INT32_MAX+1-1)`.

New checked software interface:

```text
ggml/src/ggml-potal/include/potal.h
ggml/src/ggml-potal/include/potal_rocc.h
ggml/src/ggml-potal/potal.c:600
ggml/src/ggml-potal/potal.c:1242
```

`potal_acc_bias_check/encode` validate ACC half, extents, tails, alignment,
pointer arithmetic and capacity. CONFIG_LD uses full-width identity loading,
stride zero to repeat the same bias row, and MVIN funct7=2/xd=0 addresses set
ACC+accumulate, not full-ACC-read. No new arithmetic opcode was added.

`outstanding_acc_updates` prevents early close/flush/LOOP/readback reuse.
Successful global drains clear it. Partial issue retains ownership, sticky
failure and the original status. Cleanup never retries a failed fence;
unknown completion abandons DMA buffers and prevents accelerator reuse.
Existing trace selection/format and Scale liveness/generation rules remain.

## 6. Limb Reconstruction

Signed balanced radix256 extraction widens before `(value-digit)/256`.
INT32_MAX requires five signed8 digits:

```text
[-1, 0, 0, -128, 1]
```

INT32_MIN is `[0,0,0,-128]`. All needed limbs execute, including carry limbs
and dense Q higher limbs. Only completed stream-limb INT32 results are
reconstructed:

```text
wide = checked_INT64(sum_limb(INT32_result[limb] * 256^limb))
output = finite_F32(ldexp(wide, stream_scale + stationary_base[column]))
```

INT64 overflow or nonfinite final output fails explicitly. Output is staged
across stripes, heads and batches and published only after the whole node
succeeds. Allocation/length errors return capacity failure.

RNE quantization establishes FE_TONEAREST and restores the caller's rounding
mode on success/error. Q folding deliberately retains the existing specified
`shift_q_i32` behavior, whose right shifts round ties away from zero. It is
not silently replaced with RNE or hardware SCU rounding. Attention source
files use `-fno-fast-math`.

## 7. Integer Reference Validation

The production integer reference is a testable helper, never backend fallback.
The independent oracle in
`tests/support/attention-golden.hpp` derives digits from original integer
codes, uses separate wide arithmetic, and calls neither the production
reference nor its SCU/quantizer helpers.

`potal-attention-test` checks:

- Exact folded Q/prepared K/V/P fixtures, F16/F32 and rounding-mode restoration.
- Five Q limbs, unsigned P upper carry, asymmetric K clipping and signed8 V.
- Zero carriers versus shift zero, positive/negative SCU and ACC saturation,
  lower-all/upper-all order, and completed-limb INT64 overflow.
- K tails 1/17/31/32/33/65; strided tensors, repeated heads/batches and output gaps.
- Exact within-block/cross-block bias cancellation, carrier24 counterexamples,
  shifts 200/199 cancellation beyond INT128, INT32_MIN bias, and explicit
  nonrepresentable positive/negative bias rejection.
- Raw unmasked QK with a causal mask supplied, invalid metadata/layout/memory,
  nonfinite/negative P, and no partial output on errors.
- Checked command emulation, balanced Scale entries, actual encoded panel
  boundaries, failure at TLB/Scale/LOOP/fence/release/bias stages, later-job
  failure, safe cleanup and retained unknown ownership.

`potal-attention-api-test` separately checks encodings, row replication,
tail columns, busy guards, pending DMA, sticky failure and failed-fence
non-retry. These Host/sink tests are **software tests**, not hardware timing
or Hardware PASS.

## 8. Host Regression

All nine requested binaries were rebuilt and directly executed successfully:

```text
potal-api-test
potal-trace-test
potal-residual-unit-test
potal-paired-plan-test
potal-paired-schedule-test
potal-paired-unit-test
potal-paired-auto-test
potal-paired-execution-test
potal-stripe-pipeline-test
```

`potal-ggml-unit-test`, `potal-attention-api-test`, and `potal-attention-test`
also passed. The separate H3-6E H2 inventory passed; it is not a new hardware
benchmark. No absent CTest registration was counted as successful testing.

The native `build-potal-paired-host` and existing Release
`build-potal-h3-6a-d16` builds succeeded. The latter built the real Attention
smoke, new unit/API tests, production libraries, llama targets, and acceptance
runner using existing RISC-V GCC/G++ 12.2.0 settings. Diagnostics default/cache
remain OFF.

The editor LSP could not resolve the out-of-tree include paths and reported
missing `ggml.h`/`ggml-cpp.h`; native and cross-compiler builds supplied the
actual compile validation. No diagnostics were suppressed. The pre-existing
generated-header warning redefines `POTAL_SCALE_ENTRIES` as `256` versus the
public header's `256u`; the selected capacity is unchanged.

## 9. FireSim Attention Execution

The existing U280 driver/image, DIM16 geometry, HP1/ExSIA configuration and
generated parameter-header hash were retained:

```text
b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7
```

The issuer reports Linux CPU1, the existing mapped physical hart0. The CPU
FP32 backend is a separate comparison execution, not the hardware issuer.

Real backend graphs completed these independent cases:

| Role | Logical K | Rows | Columns | Head/batch handling | LOOP_WS | Bias MVIN tiles |
|---|---:|---:|---:|---|---:|---:|
| QK | 64 | 4 | 19 | 2 stationary heads, repeat2, batch2, strided F16 | 48 | 0 |
| QK | 33 | 1 | 19 | Strided F32 decode | 4 | 0 |
| PV | 17 | 1 | 19 | Strided F16 decode | 3 | 2 |
| PV | 33 | 4 | 19 | 2 stationary heads, repeat2, batch2, strided F16 | 24 | 16 |
| PV | 65 | 1 | 19 | Strided F16 decode | 3 | 2 |
| PV | 64 | 1 | 1 | Cross-block carrier24 cancellation, strided F32 | 3 | 1 |

Each case had actual PoTal completion, nonzero LOOP_WS, balanced Scale commands,
and bitwise F32 output equal to the independently scaled integer oracle.
These six power-of-two/cancellation fixtures also had zero FP32 output error.
That does not imply zero error for arbitrary inputs or the real model.

The GGML smoke's legacy marker says `hardware integer-domain exactness`; its
condition compares final F32 bytes with scaled integer goldens. It is not a
dump of every intermediate INT32 bit. Host tests establish independent integer
recurrences separately. A direct raw INT32 hardware test additionally fills
ACC through SCU saturation and checks all 256 readback values equal
`INT32_MAX-128` after one negative bias addition.

Actual low/high shared-ACC address/order evidence comes from translated
commands and RTL, not from two separately merged hardware passes.
The smoke returned `POTAL_ATTENTION_SMOKE PASS` before model inference began.
The suite and simulator returned 0. Whole-emulation cycle totals are not
individual Attention-kernel cycles, DMA time, or physical PE utilization.

A final independent hardware run added a destructive QK saturation witness.
Its independently reconstructed integer is 549755813632; separate K-panel
accumulators would give 558345747968. Actual hardware F32 output 0.001953125
matches the required scaled golden and differs from the forbidden
0.00198364258. All four LOOP_WS completed with balanced Scale commands.
The independent FP32 value is 0.755950928, absolute error 0.753997803: saturation
semantics are being tested, not an arbitrary-input FP32 accuracy tolerance.

That run also executed the latest mask and beyond-INT128 cancellation unit
cases, repeated the six integration fixtures, and rechecked the raw INT32
bias saturation test. It ended with `__H37B_EDGE_PASS__`,
`__H37B_EDGE_EXIT__=0`, and simulator exit 0. The two simulator host wall times
were 1938.1 s and 378.1 s, including boot/setup/tests, not kernel times.

## 10. Unsupported Hardware Requirements

No new arithmetic RTL was needed for supported QK or INT32-bias PV. Existing
full-width ACC DMA already implements the required final saturating addition;
the missing piece was checked software ownership/encoding.

**Unrestricted exact-wide PV bias remains unsupported.** If exact B is outside
INT32, the current wrapper rejects before that stripe's work. It does not
clamp B, split the correction, apply it per K microtile, or add it after
readback. For example:

```text
a = -1, exact B = 2147483648
Sat32(a + B) = 2147483647
Sat32(a + Sat32(B)) = 2147483646
```

Thus a pre-clamped operand is not an equivalent implementation. Full support
needs an addressed ordered wide-bias saturating ACC-add interface, with
explicit bank/row/column masks, DMA lifetime, completion visibility and
failure ownership. A signed64 operand needs a defined larger-range bound;
arbitrary carrier-domain sums cannot be assumed to fit INT64.

This capability is **Blocked for the rejected domain**, not Hardware Verified.
There is no hidden Host repair. General GGML type/layout/capacity failures and
checked reconstruction overflow are likewise explicit failures, not CPU PASS.

## 11. Short Full-Inference Validation

Command, after unit/API and independent hardware gates:

```bash
./bin/potal-full-inference \
  --backend potal --route serial --attention --no-warmup \
  --model /root/h3-7a/models/gpt2.Q8_HP1.gguf \
  --prompt-file /root/h3-7a/prompt.txt \
  --prompt-tokens 16 --generate 8 --capture --repeat 1 \
  --output /root/h3-7b/results/attention-16x8
```

The actual model still has 124,439,808 parameters. Model/context metadata
records `attention_offload=true`, `warmup=false`, context 512, batch/ubatch 256,
three CPU threads, F16 KV, and Flash Attention OFF. No 256+32 run was started.

Generated IDs match independent cached-checkpoint CPU FP32, native CPU FP16,
and preserved H3-7A Serial mixed execution:

```text
5911 290 5911 262 16936 4780 13 383
```

Every captured forward has **73 observed MatMul records**: 49 HP1 and 24
Attention on POTAL. Each has CPU Attention MatMul=0, CPU HP1 MatMul=0,
unknown MatMul=0 and 12 CPU Softmax operations. The collector is populated and
validated; these zeros are not inferred from an empty callback.

Across eight forwards:

| Hardware evidence | Count |
|---|---:|
| HP1 completed / failed | 392 / 0 |
| QK / PV nodes | 96 / 96 |
| Attention completed / failed | 192 / 0 |
| Attention signed-limb jobs | 6,912 |
| Higher stream-limb jobs, Q and P combined | 4,608 |
| Attention LOOP_WS | 10,368 |
| QK shared-ACC jobs | 3,456 |
| PV low correction jobs | 1,152 |
| Bias MVIN output tiles | 4,608 |
| Total Scale Write / Release, linear and Attention | 56,977,433 / 56,977,433 |
| Linear Main / Residual LOOP_WS | 9,328 / 7,487 |
| Paired-auto stripes / fallback stripes | 0 / 0 |

Correction tiles cover disjoint output row/column tiles, not repeated
corrections per lane. Attention's paired-session primitive counters are
separate from the linear paired-auto counters.

KV checks passed at every step: logical count 16..23, maximum position 15..22.
The final emitted token is not fed back, hence 23 rather than 24 cached
tokens. Physical K/V extents remain padded to 32 for this workload.
Prefill QK/PV outputs are `[32,16,12,1]` / `[64,16,12,1]`;
last-decode outputs are `[32,1,12,1]` / `[64,1,12,1]`.
Standalone PV K17/33/65 tests cross the physical padding boundaries.

CPU work remains: Softmax 96, NORM 200, GET_ROWS 32, UNARY 96, ADD 784, MUL 200,
CPY 192, CONT 384, RESHAPE 384, VIEW 288, PERMUTE 96 and TRANSPOSE 96.
CPU-backed KV storage or embedding buffers do not mean CPU Attention MatMul.

All 402,056 captured logits are finite; all eight histories align.
No FP32 acceptance tolerance was invented:

| Reference versus new Attention | Max absolute | RMSE | Max relative | Mean relative | Normalized RMSE | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Independent FP32 | 2.278503418 | 0.562603866 | 0.016723889 | 0.003782641 | 0.004445341 | 8/8 |
| Native CPU FP16 | 2.277221680 | 0.555312660 | 0.016662610 | 0.003723647 | 0.004387600 | 8/8 |
| Preserved H3-7A Serial, CPU Attention | 0.639099121 | 0.310263439 | 0.006161403 | 0.002172750 | 0.002451302 | 8/8 |

FP32-vs-new last-prefill error is max 1.882049561/RMSE 0.644828844.
The reference checkpoint is unchanged SHA256
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
Token agreement is not a claim of FP32 bitwise equality, corpus accuracy, or
perplexity equivalence.
Historical H3-7A FP32 RMSE was 0.401450281; the new 0.562603866 is larger.
Neither the unchanged tokens nor the lower maximum error establish an accuracy
improvement.

Single captured cold-request guest CPU-observed timings, milliseconds:

| Measurement | H3-7A Serial, CPU Attention | H3-7B Serial, PoTal Attention |
|---|---:|---:|
| Prefill | 7,699.017 | 8,012.482 |
| TTFT | 7,724.151 | 8,037.795 |
| TPOT, seven decode intervals | 1,602.768 | 1,927.997 |
| Total request | 18,943.530 | 21,533.775 |

The new captured request is slower; no speedup is claimed. These are
single cold diagnostic captures, not a repeated matched performance study.
The new collector is populated while the old E1 collector was incomplete.
Model load 2544.215/context init 261.511 ms are outside request timing.
All 49 HP1 weights were validated/packed once, followed by 343 cache hits.
No steady full-inference sample was manufactured from this cold run.

Attention API wall is 2698.075 ms. Total Scale/issue/fence caller intervals
are 5490.415/505.384/19.176 ms. Linear pipeline wall 13960.644 ms overlaps other
preparation; none of these categories is added as independent graph time
or labeled pure FPGA execution.

## 12. Final Implementation Status

The supported implementation is hardware-executed and short-inference verified.
Integer recurrence tests, final hardware outputs, FP32 error and token
agreement are distinct evidence classes. Unrestricted wide-bias PV, broader
models/layouts, long inference, corpus accuracy, and performance improvement
are not established by this result.

H3-6E remains a separate unchanged-binary experiment and report. It did not
replace Paired scheduling or start D2. Its measured K reduction is B-bank-bound;
Attention correctness is not evidence of a Paired pipeline speedup.

New tests/artifacts:

```text
ggml/src/ggml-potal/tests/potal-attention-test.cpp
ggml/src/ggml-potal/tests/potal-attention-api-test.c
ggml/src/ggml-potal/tests/potal-attention-smoke.cpp
ggml/src/ggml-potal/tests/support/attention-golden.hpp
ggml/src/ggml-potal/tests/support/attention-emulator.hpp
```

Test-only integration also updates `potal-ggml-unit-test.cpp` and
`potal-full-inference.cpp`. H3-7A's Python reference/comparison tools are
unchanged and reused.

Primary evidence:

```text
/tmp/h3-7b-run/uartlog
/tmp/h3-7b-guest.sh
/tmp/h3-7b-results/attention-16x8.model.json
/tmp/h3-7b-results/attention-16x8.request0.json
/tmp/h3-7b-results/attention-16x8.request0.logits.f32
/tmp/h3-7b-results/fp32-vs-attention-16x8.json
/tmp/h3-7b-results/fp16-vs-attention-16x8.json
/tmp/h3-7b-results/linear-only-vs-attention-16x8.json
/tmp/h3-7b-attention-test.log
/tmp/h3-7b-edge-run/uartlog
/tmp/h3-7b-edge-guest.sh
```

Production `libggml-potal.so` SHA256:

```text
630fcc334458c7cad69506be1d8588f2bd0253c7becc2d73c7ceb9835a16e513
```

Final hardware smoke/test SHA256:

```text
potal-attention-smoke:
fb585e3dbb2578e7f5ccf2c2b08e39f85dca29937fc80da78bb1b37121956384
potal-attention-test:
3fd6f9596db3ec9c84f917ed076fdf139f005efb35d9fd8490d34b464ed11a29
```

Both model GGUF hashes, Gemmini submodule HEAD/pre-existing diff fingerprint,
`software/host-init.sh`, and protected Linear/Paired/D1 sources were checked.
The Git index is empty. Models/rootfs/build products were not staged.

Next implementation priority is an independently specified resident-B/
microtile scheduling interface from H3-6E, not an assumed loop interchange.
Unrestricted PV needs the wide-add contract above. Performance or wider
accuracy experiments require a separate scope; this task claims no such result.
