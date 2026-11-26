# H3-7C: Attention PV Correction Range Resolution

Date: 2026-10-27.

Status: **E2E Functional PASS; Full 256/128 Workload PASS.**
The sole gated retry exited 0 with 256 input tokens and 128 generated tokens.
All required GEMMs executed on PoTal with zero CPU or Paired fallback.
Generated prose is poor and repetitive; this is not a model-quality PASS.

## 1. Historical failure reconstruction

Primary evidence is the complete `H3-7-GPT2-paired-attention-e2e-report.md`
and the unchanged files under `/tmp/h3-7-gpt2-e2e/e2e-extract/e2e/`.
No inference was rerun to reconstruct this failure.

- First failed graph node: `node_28`, `MUL_MAT`, PV, layer 0.
- Phase: decode index 0, graph occurrence 0, batch 0.
- QK inputs: K F16 `[64,288,12,1]`, Q F32 `[64,1,12,1]`.
  Per-head QK M/N/K is 1/288/64; QK completed successfully.
- PV inputs: V F16 `[288,64,12,1]`, P F32 `[288,1,12,1]`.
  Per-head PV M/N/K is 1/64/288, with nine K32 blocks.
- The 257 active positions do not replace the declared GEMM K of 288.
- Observed status: `ERR_UNSUPPORTED`, stage
  `attention_prepare_or_reconstruct`, `fatal=0`.

The failed head is **source-derived as head 11**, not directly logged.
The successful PV-node records account for 9216 bias updates, while the
terminal summary records 9260. The difference is 44. A completed corrected
job for M1/N64 contributes four updates, and this shape has one stripe per
head. Thus eleven heads completed before the next head's preparation failed.
Execution visits heads in ascending order. The singleton batch is batch 0.
The corresponding partial work is 33 completed jobs and 33 LOOP_WS commands.

Relevant original records are `debug-log.jsonl:3344-3347` and
`semantic-graph.jsonl:494,523`. The semantic graph identifies `cache_v_l0`,
the preceding softmax, and the failed PV operation. The completion-count
derivation is stronger than assuming the first head failed.

Actual P/V arrays, block exponents, carrier distribution, failing column,
and exact correction were not retained. The numerical reproducer below
matches the shape and rejection class, not those unavailable operand bytes.

## 2. Root cause and arithmetic range

The old path represents P as biased signed radix-256 digits of `Pcode-128`.
Its low ACC receives the exact correction:

```text
S[j] = sum_blocks(2^Delta[block,j] * sum_k(Vcode[k,j]))
B[j] = 128 * S[j]
```

`make_bias()` groups coefficients and normalizes binary carries before its
range check. Cancellation is exact even when individual terms are wider
than INT64 or INT128. It accepts S in `[-16777216,16777215]`.
The largest positive representable bias is consequently 2147483520, not
INT32_MAX; the exact negative endpoint is INT32_MIN.

Historical failure localization is unique in the reviewed path:
`attention.cpp:351` throws `POTAL_ERR_UNSUPPORTED` for this representability
test. The hardware caller supplies the corrected runner, ruling out the
other unsupported branch. Runner failures retain their own stages.

Therefore at least one visited column has S <= -16777217 or S >= 16777216.
This is a source-derived classification, not a measured numerical value.
Finite historical F16 V values imply exponents -24 through 15 and Delta <=39.
For K288, the conservative bound is:

```text
abs(B) <= 128 * 288 * 128 * 2^39
       = 2594073385365405696 < INT64_MAX
```

This distinguishes a valid exact correction outside INT32 from an
intermediate-overflow status. Invalid probabilities or carriers produce
different checks/statuses. The evidence does not establish saturation of
the intended PV result; a wide bias alone does not imply such saturation.
P preparation retains its existing finite `[0,1]` input check and
quantized-code range `[0,65535]`. Historical extrema remain unavailable.

## 3. Evaluated candidates and selection

**Unrestricted balanced radix-256 was rejected.** Algebraic reconstruction
does not preserve saturated arithmetic. For Pcode 128, Vcode 127 and
Delta 24, balanced digits are `[-128,1]`:

```text
exact quantized dot             272730423296
reconstructed saturated limbs  543313362944
decomposition error            270582939648
```

The independent Host regression checks this counterexample. No tolerance
was widened to accept it.

Guarded balanced radix-256 and nonnegative radix-128 can each admit useful
subsets, but neither dominates the other. Splitting a wide correction into
saturating bias commands cannot recover information lost in the original
low ACC. Exponent-banded GEMMs with rebased physical carriers would require
a larger lowering contract.

The selected extension is **one guarded unsigned radix-2 mode**. It has
smaller digit magnitudes and a direct exactness proof, at the cost of up to
16 completed GEMMs. No evidence about the missing historical operands
justifies a more complex multi-encoding selector in this change.

Bit planes are not a universal superset of guarded balanced digits.
Separate planes can lose useful signed-V cancellation between bits. The
new domain is deliberately a certified subset, not every finite PV input.

## 4. Representation, saturation and exactness proof

Use the unchanged unsigned P codes:

```text
Pcode = sum_bit(digit[bit] * 2^bit)
digit[bit] = (Pcode >> bit) & 1
```

Each digit is a legal signed INT8 value, 0 or 1. Original V codes, column
bases, carriers, padded K, tile selection and increasing K order are retained.
Every plane completes a separate planned WS GEMM with a fresh ACC epoch.

Actual DIM16 arithmetic is:

```text
partial = sum_16(signed_INT8_digit * signed_INT8_V)
scaled  = ZERO ? 0 : Sat32(partial * 2^Delta)
ACC     = Sat32(previous_ACC + scaled)
```

SCU saturation precedes ACC saturation. K-panel boundaries do not reset an
accumulating job. The binary partial lies in `[-2048,2032]`, inside the
configured signed-20-bit array output.

Before issuing any command for the candidate stripe, compute V-negative and
V-positive sums for each DIM16 microtile. P bitwise AND/OR determine each
plane's exact minimum and maximum digit. With digit extrema m/M:

```text
L = m * Vpositive + M * Vnegative
U = M * Vpositive + m * Vnegative
```

ZERO carriers give `[0,0]`. For Delta <=31, scale both bounds only after
checking their representability. For Delta >=32, require a proven zero
interval. Every individual SCU interval and every ascending-K prefix
interval must fit INT32. Bounds are never clipped.

Induction proves that neither hardware saturation point loses information.
Each returned signed32 ACC is then the exact plane dot. Checked Host INT64
reconstruction equals the unchanged exact wide quantized dot:

```text
sum_bit(C[bit] * 2^bit)
  = sum_k(Pcode[k] * Vcode[k,j] * 2^Delta[k,j])
```

For at most 16 planes, every reconstruction prefix has magnitude below
`2^31 * (2^16-1) < 2^47`. The existing checked arithmetic remains enabled.
The original F32 conversion/scaling and finite-output validation remain.
Newly admitted cases add zero decomposition or saturation error relative
to the quantized dot; this does not remove existing quantization error.

The certificate is conservative. Rejection means not certified, not proof
that the actual result saturates. Safety-bound calculation is not a CPU GEMM
execution route and supplies no output values.

## 5. Source changes and compatibility boundary

Changed files:

1. `ggml/src/ggml-potal/matmul/attention.cpp`: overflow-only mode selection,
   before-issue certificate, ordinary planned plane execution and checked
   radix-2 reconstruction, reusing the original execution pipeline.
2. `ggml/src/ggml-potal/matmul/attention.hpp`: numerical contract and private
   `pv_bitplane_stripes` / `pv_bitplane_jobs` result counters.
3. `ggml/src/ggml-potal/ops.cpp`: expose both counters in existing success
   and failure diagnostics without changing the exported C statistics ABI.
4. `ggml/src/ggml-potal/tests/support/attention-golden.hpp`: independent
   original-code digits, raw saturating planes and exact wide quantized dot.
5. `ggml/src/ggml-potal/tests/potal-attention-test.cpp`: focused numerical,
   descriptor, compatibility, cleanup and failure regressions.
6. `ggml/src/ggml-potal/tests/potal-attention-smoke.cpp`: isolated
   `--pv-range-only` real-hardware mode; existing default smoke remains.

`pv_bitplanes()` begins at `attention.cpp:361`; execution reuses
`execute_stripe()` at `:481`. `column_bias()` remains strict. Public
preparation, `stream`, `max_limbs`, runner signatures, checked bias command,
QK arithmetic and `attention-hw.cpp` are unchanged.

If every bias fits, all original PV operations and saturation behavior remain.
Only a representability rejection permits the alternative. Other failures
propagate. A previously rejected mixed-column head had no published legacy
result; its new certified mode computes all columns exactly. Earlier normal
heads remain on the original path.

All planes are certified before the stripe's first command. Existing
overwrite/accumulate/store epochs, DMA allocations, fences, Scale releases,
fatal allocation retention and whole-operation staged publication remain.
No unchecked accumulator command or CPU fallback was added.

Incoming CMake and API-smoke changes, H3-6/H3-7 files, verified build trees,
historical reports, source rootfs and bitstream remain preserved. No commit
or push was performed.

## 6. Independent numerical regression evidence

New Host tests passed on their first run:

- Attention unit and command execution: 592 PASS checks, zero FAIL.
- Attention C11 command-emulator API: 33 PASS checks, zero FAIL.
- The preserved native baseline also passed without rebuilding its directory.

Coverage includes normal biased PV, both signs beyond INT32, exact bias
endpoints, cross-block cancellation, large and ZERO carriers, signed-radix
boundaries 127/128/255/256/65535, mixed signs, SCU limits at Delta31/32,
every-prefix ACC refusal, padded K32/K64/K96, tails, tiled M/N, K4097
multiple panels, single-row Decode, invalid metadata/codes/padding and
existing checked INT64 overflow witnesses.

The new oracle obtains digits by division/remainder from original codes,
not production-prepared limbs. It models concrete DIM16 signed products,
SCU and ACC saturation, independent plane completion and reconstruction.
An independent sparse wide-integer sum handles cancellation before bounded
shifts. Accepted plane results must also equal that exact quantized dot.

The deterministic condition reproducer has M1/N64/K288, 12 heads and one
batch. Normalized P uses `P[0]=2^-16`, `P[16]=1-2^-16`. V uses signed 1 at
position 0 and signed 2^-24 at position 32 in head 11; earlier heads use
signed 0.5 at position 32. The new head has bias
`+/- (2^37+8192)` and exact output `+/-2^-16`.
This reproduces the historical shape and guard class, not missing data.

## 7. Existing-path compatibility

Existing QK, saturation witnesses, normal PV bias commands and raw INT32
expectations remain tested. The old-path heads in the new execution fixture
also require bitwise F32 equality. Two former reference-range rejections
were updated to require exact acceptance; strict `column_bias()` rejection
is still asserted.

The historical-condition fixture requires eleven original corrected heads,
one extended head and sixteen completed plane jobs. Descriptor checks retain
P/V operands, carriers and frozen tile choices. New-path raw command C must
match the independent oracle, with zero corrected-runner or bias-MVIN calls.
Later-plane recoverable and fatal faults preserve status, no partial
publication, completed-job counts and DMA/Scale ownership.

The unmodified CLI and libllama binaries are byte-identical to H3-7.
Only the backend and affected test binaries differ; the existing monotonic
request clock and event boundary implementation were not changed.

The RISC-V guest repeated all 592 unit checks and 33 C11 command-emulator
checks successfully. The existing real-hardware Smoke also passed its
independent raw QK passes, original ACC epochs and generations, normal PV,
saturation witnesses, Scale balance and K4097 multi-panel case. These are
distinct software-only and physical-hardware results, not interchangeable.

## 8. Host and FireSim targeted validation

Host build: `build-potal-h3-7c-host`, exit 0.
RISC-V build: `build-potal-h3-7c-d16`; CLI, Attention API, unit test and
hardware smoke targets all built with exit 0.

The compiled graph has `GGML_POTAL_ATTENTION=1`. The backend has Paired-auto,
EXSIA A8/W8, block32, RMD, sequential ExSIA, LOG_CYCLE=1, LOG_DEBUG=1,
CYCLE_DETAIL=0 and CYCLE_SIM=0. Attention compiles with `-fno-fast-math`.
The generated DIM16 parameter hash remains
`b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7`.

Host and RISC-V command-emulator results are software-only evidence.
The RISC-V guest unit/API programs and both real-hardware Smoke modes exited
successfully; the aggregate preflight exit was 0. The simulator then exited
0 after normal guest poweroff. No full-model inference occurred on this boot.

| Real hardware range fixture | Observed |
|---|---:|
| M / N / K / heads | 1 / 64 / 288 / 12 |
| Completed jobs / LOOP_WS | 49 / 49 |
| Selected bit-plane stripes / completed plane jobs | 1 / 16 |
| Original correction jobs / bias MVINs | 11 / 44 |
| Fences | 71 |
| Scale Write / Release | 28224 / 28224 |
| Independent exact F32 output | PASS, all positive and negative columns |
| Status / fatal hardware errors | POTAL_OK / none |

The range case uses the real hart-0 issuer and checked planned GEMMs, not a
software GEMM substitute. Existing Smoke QK/PV nodes independently confirm
backend hardware completion. No CPU fallback or buffer-lifetime failure was
reported. Fence counts are completion evidence, not accelerator timing.

Original targeted files are preserved under `/tmp/h3-7c/preflight-results/`.
`attention.out` and `range.out` are the real-hardware records; guest unit
fault-injection messages belong only to the separate command emulator.
The guest confirmed Linux CPU 1 maps to Hart 0. Its model, prompt, CLI,
libllama and new backend hashes match the staged manifest.

Logs and manifests: `/tmp/h3-7c/{host-*,cross-*,identity-manifest.txt}`.
Pre-existing build warnings remain in unrelated source and the generated
parameter header. The default LSP lacks the compile/include configuration.
An explicit native compilation-database check resolved source parsing, but
clangd 16 exited 3 for seven refactoring-feature self-test failures: four
`ExtractFunction` and three `ExpandDeducedType` failures. It emitted no source
compile diagnostics in that check. These tooling failures are retained in
`/tmp/h3-7c/clangd-attention.log`; native and RISC-V GCC builds both succeeded.
Whitespace and unchanged hardware-runner, checked API and timing-file checks
also passed.

The hardware image is a separate offline copy, expanded to 24 GiB with
7.1 GiB free after staging. New software/results use `/root/h3-7c`.
Historical `/root/h3-7-e2e` and original image artifacts remain unchanged.

## 9. Conditional single GPT-2 retry

**The single gated retry completed: CLI exit 0, 256 input tokens and exactly
128 generated tokens.** Execution ID: `143-14649092000`.
The UART and original guest `exit.status` both contain `H3_7C_E2E_EXIT=0`.
No second request, shortened workload, CPU/Serial switch or retry was used.

The model remains the same Q8_HP1 GGUF, SHA256
`cf23a45e4db23736d7d1f67ab42018803a198cfcb01db36daca69ff56bdbc7d4`.
Its actual parameter count is 124439808, despite the historical "163M" label.
Prompt SHA256 remains
`7e13482ec87666be18c7e5e14e04b14f4d09405f1099cfd68d92696c5469cd78`.
All 256 token IDs printed by this CLI match the preserved H3-7 input IDs;
`initial_matching_cache_tokens=0`, warmup is false, seed is 1 and temperature
is 0. No EOS token was generated; the output budget ended the request.

The software HEAD is `8fb19aa5b66ace4cc50014581e5b3fe5ad7fc546` on `potal`.
The tested six-file source manifest still matches. Incoming CMake/API-smoke
changes, historical reports, model files and verified builds remain intact.
`/tmp/h3-7c/source-only.diff` and `source-files.sha256` preserve this change;
`identity-manifest.txt` records the broader incoming working-tree identity.

| Execution artifact | SHA256 or identity |
|---|---|
| DIM16 parameter header | `b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7` |
| Deployed backend | `1d255b54eae30f94dd5d915c4bd905a43812f9ddaf3b1cd2f7183218f1606cd3` |
| CLI | `c2800894c8cfd91d2d022e6f2969b86f3873c287cc3f4736673c191c0c40ebce` |
| Slot driver | `ebbc201ee188d1553912aafb3835b5637f247eed60a696499dcf74a27b110a0d` |
| Preserved slot firesim.bit | `f4c465a451385cbda44c254fb038fc2ede3a25ea12d5e829c2a501b7d495b3de` |
| Hardware configuration | `FireSimQuadRocketSinglePoTalD16Config`, U280, Quad Rocket, one PoTal |

The slot bitstream identity is inherited from the preserved H3-7 deployment,
not an FPGA readback hash. No RTL, ISA or bitstream was changed or rebuilt.
The existing hardware ISA gates and both H3-7C Attention gates passed.

The actual CMake cache has POTAL ON, Gemmini OFF, EXSIA, A8/W8, block32,
RMD ON, STRIPE_PIPELINE, WS, Paired-auto ON, DIM16, LOG_CYCLE=1, LOG_DEBUG=1,
CYCLE_DETAIL=0 and CYCLE_SIM=0. Attention's compile definition remains enabled.
Runtime logs confirm Paired ON with RTL fences. The semantic run record
confirms Flash Attention false. The issuer is tid147 on Linux CPU1/Hart0,
affinity {1}; both CPU pools request three strict workers on CPUs {0,2,3}.
The configuration's observed CPU affinity "3" describes the logging caller,
not a restriction of the issuer to CPU3.

The executed script, copied back unchanged, is
`/tmp/h3-7c/executed-inference.sh`. Its model invocation is:

```sh
cd /root/h3-7c
export LD_LIBRARY_PATH=/root/h3-7c/bin
export GEMMINI_LOG_DIR=/root/h3-7c/results/e2e
export GEMMINI_MATMUL_MODE=STRIPE_PIPELINE
export GEMMINI_RMD_BACKEND=WS
export GGML_POTAL_PAIRED_AUTO=ON
export GGML_POTAL_PAIRED_FENCES=RTL
GGML_SCHED_DEBUG=2 ./bin/llama-cli \
  -m /root/workspace/llama.cpp-potal/models/gpt2/gpt2.Q8_HP1.gguf \
  -f /root/h3-7c/prompt-256.txt \
  -n 128 -c 512 -b 256 -ub 256 -t 3 -tb 3 \
  --cpu-mask 0xD --cpu-mask-batch 0xD --cpu-strict 1 --cpu-strict-batch 1 \
  --temp 0 -s 1 -no-cnv --no-warmup --no-context-shift \
  --no-display-prompt --simple-io --verbose-prompt \
  > "$GEMMINI_LOG_DIR/llama-cli.out" \
  2> "$GEMMINI_LOG_DIR/llama-cli.err"
```

The script first checks all targeted gates and creates its once-only guard.
It clears profiling/raw-trace overrides, prompt-cache overrides and
`GGML_GEMMINI_NPU_HZ`. The original script retains the full environment setup.

A host agent restart removed launcher PID835025 while the original driver
PID835204 and guest request survived. Recovery restored observation without
relaunching or sending another guest request. The driver's eventual process
termination was observed, but its exit code is **unavailable**, not assumed 0.
Read-only collection began only after driver termination and a second live
image check; the image is now unmounted. The collection command exited 0.

The complete synchronized originals are under `/tmp/h3-7c/final-results/`:
`llama-cli.out`, `llama-cli.err`, `cycle-log.jsonl`, `debug-log.jsonl`,
`semantic-graph.jsonl` and `exit.status`. All original hashes and the executed
script pass `sha256sum -c /tmp/h3-7c/final-sha256.txt`.
Live-observation directories are provisional and are not final-run evidence.
The derived record is `/tmp/h3-7c/final-analysis.json`.

## 10. Timing and execution counters

The existing `FINAL_INFERENCE_SUMMARY` is available and records one completed
request, 128 tokens, verified latency boundaries and verified interval
coverage. Its values match an independent subtraction of the original
request/token-ready timestamps. The same values appear in the CLI's
`Inference performance` output.

| Canonical guest monotonic measurement | Actual value |
|---|---:|
| TTFT | 53400.266 ms |
| TPOT, 127 token gaps | 7177.511755905512 ms/token; CLI rounds to 7177.512 |
| Total request latency | 964945.217 ms |
| Prefill operations elapsed | 53366.037 ms |
| Decode operations elapsed | 909575.396 ms |
| Prefill / Decode failed operations | 0 / 0 |
| Input / requested output / actual output | 256 / 128 / 128 |

These are the existing guest `steady_clock` request boundaries, not host
FireSim wall time or accelerator cycles. Model/context creation and prompt
tokenization precede the request. Warmup is disabled. The final summary's
log replay, printing and cleanup follow request_end and are excluded.
Inline execution diagnostics remain part of the measured software path.
Request time includes gaps between operations; phase times are not a
substitute for its direct boundary measurement.

The canonical diagnostic worker CPU times are 18845.633 ms for Prefill and
75935.666 ms for Decode. Full CPU work wall time is unavailable because
`potal_host_cpu_stage_coverage_incomplete`; the recorded instrumented-stage
unions are only 12023.559 ms and 9471.731 ms, respectively. They are not
complete CPU time or hardware waiting. NPU cycles/time are unavailable
(`unavailable_device_counter`); no configured frequency conversion, inferred
accelerator time or overlapping-stage sum is used. Historical TTFT
53051.478 ms remains one observation, not a statistical regression baseline.

### Actual hardware coverage and cleanup

All 62592 NODE_EXECUTION records join their exact phase/decode/graph/node
metadata, with zero missing joins, zero failed nodes and a successful RUN_END.
There are 128 graph instances. All 9344 executed MUL_MAT nodes have actual
backend POTAL and execution class TARGET_NPU: 73 in Prefill and 9271 in Decode.
Required GEMM CPU fallback count is **0**, not inferred from backend registration.

| Executed matrix role | Successful PoTal nodes |
|---|---:|
| Combined Q/K/V projection | 1536 |
| Attention QK | 1536 |
| Attention PV | 1536 |
| Attention output projection | 1536 |
| FFN up | 1536 |
| FFN down | 1536 |
| lm_head | 128 |

The 6272 dense nodes each have a committed Paired pipeline record, Paired ON,
RTL-fence policy, balanced per-node Scale operations and a successful checked
Fence in their HARDWARE-dispatch record. Per-node sums match the final counters.
The pipeline's committed status is not the public API status spelling.
Its duplicate `paired_fences` log key represents the policy first and the
numeric fence count later; the two meanings were checked separately.

| Paired command/completion counter | Observed |
|---|---:|
| Paired stripes | 6410 |
| Paired fallback / Serial fallback stripes | 0 |
| Paired Main LOOP_WS | 284004 |
| Indexed Residual LOOP_WS | 1669931 |
| Index Config / Write | 1669931 / 31814198 |
| Paired readbacks / readback tiles | 250112 / 2279344 |
| Paired Fence / Drain | 6410 / 1960345 |
| Combined Scale Write / Release | 1052152636 / 1052152636 |

The combined Scale totals include dense 939362876 and Attention 112789760
writes, with equal releases. Attention records 3072 hardware completions,
zero failures, 118368 QK LOOP_WS and 55871 PV LOOP_WS, totaling 174239.
There are 113184 completed independent QK jobs, 55871 PV jobs and 82324
PV bias MVIN updates. Attention uses dedicated planned WS jobs; it is not
claimed to be Paired Indexed Residual execution.

The new path executes in **10 PV nodes, 11 stripes and 176 completed bit-plane
jobs**. It is exercised at Decode indices 0, 32, 64 and 96. The original
failing location, layer0/node_28 at Decode0, now completes with one extended
stripe, 16 plane jobs and 49 total jobs. Original normal heads remain on the
correction path. The detailed locations and source log lines are preserved
in final-analysis.json. There are no failed nodes, fatal hardware errors or
reported lifetime failures. Checked completion and balanced resources are
completion evidence, not measurements of DMA time or hardware overlap.

## 11. Generated text quality

The synchronized `llama-cli.out` is 437 bytes and valid UTF-8 (also entirely
ASCII). All 128 token IDs are retained below and in final-analysis.json.
Indices are contiguous 0..127. EOS ID50256 is absent; the final token is ID257
(" a"), and the 128-token budget ends the text mid-sentence.

Complete original generated text:

```text
.

"It's a ship that's been lost for a long time," he said. "It's a ship that's been lost for a long time. It's a ship that's been lost for a long time. It's a ship that's been lost for a long time. It's a ship that's been lost for a long time. It's a ship that's been lost for a long time. It's a ship that's been lost for a long time. It's a ship that's been lost for a long time. It's a ship that's been lost for a long time. It's a

```

Complete generated token IDs:

```json
[
  13, 198, 198, 1, 1026, 338, 257, 4074, 326, 338, 587, 2626, 329, 257, 890, 640,
  553, 339, 531, 13, 366, 1026, 338, 257, 4074, 326, 338, 587, 2626, 329, 257, 890,
  640, 13, 632, 338, 257, 4074, 326, 338, 587, 2626, 329, 257, 890, 640, 13, 632,
  338, 257, 4074, 326, 338, 587, 2626, 329, 257, 890, 640, 13, 632, 338, 257, 4074,
  326, 338, 587, 2626, 329, 257, 890, 640, 13, 632, 338, 257, 4074, 326, 338, 587,
  2626, 329, 257, 890, 640, 13, 632, 338, 257, 4074, 326, 338, 587, 2626, 329, 257,
  890, 640, 13, 632, 338, 257, 4074, 326, 338, 587, 2626, 329, 257, 890, 640, 13,
  632, 338, 257, 4074, 326, 338, 587, 2626, 329, 257, 890, 640, 13, 632, 338, 257
]
```

The continuation remains related to the prompt's lost-ship story, and its
opening dialogue is readable. It then repeats "It's a ship that's been lost
for a long time." exactly eight times. This is substantial repetition and
poor prose quality, not a fluent-generation quality PASS. No invalid token,
visible corruption or execution-related numerical failure is observed.
The execution evidence does not establish the absence of every possible
model-level numerical difference or validate all logits against FP32.

A separate software-only Host FP16 reference, execution ID
`840289-1793101733129057230`, exited 0 with 128 tokens and
no EOS. It has the same architecture, vocabulary, dimensions, 124439808
parameters, all 256 input IDs, seed1/temp0 and warmup OFF. It registers CPU
only; POTAL and Gemmini were disabled. FP16 model SHA256:
`5bd663a1e1d303f9404dd2edd92e4e3ca3e24bcf80d19c655880f0e455c8f4df`.

That reference starts with the same ID13 period and repeats "It was a ship
that was lost in the storm." eight times, also ending mid-sentence. The
hardware continuation differs after the shared opening period. Repetition
therefore also occurs without PoTal; these observations do not establish
that the new PV path caused the quality problem. This is a qualitative
cross-quantization comparison, not token-level equality, CPU fallback,
hardware validation or an FPGA timing baseline. Its full raw text, IDs and
assessment remain separate under `/tmp/h3-7c/cpu-reference/`.

## 12. Remaining limitations and verdict

**E2E Functional PASS. Full 256/128 Workload PASS.** The actual CLI exits 0,
all 256 historical input IDs match, exactly 128 tokens are generated,
required matrix roles execute on PoTal, Paired Main/Indexed Residual and
QK/PV completions are observed, CPU GEMM fallback and Paired fallback are 0,
Scale operations balance, checked completion succeeds and the original
monotonic TTFT/TPOT are valid. The generated text has no observed
execution-related corruption. Its poor repetition is reported separately;
this verdict does not claim acceptable fluent model quality.

Numerical admission remains conservative. Large carriers or prefix intervals
can remain unsupported even when a final wide dot or F32 value would fit.
Added plane jobs can increase runtime. Newly admitted results are exact
relative to the existing quantization, not an FP32 model-quality guarantee.
The successful same-workload Decode0/node_28 extension resolves the observed
rejection class, but missing historical operands prevent a bytewise replay
of the old failed head's complete numerical state.

This is one first-request functional/performance observation, not a repeated
benchmark or statistical improvement claim. Full CPU stage coverage and
accelerator cycle/time counters remain unavailable. The host driver exit
code was lost with the launcher; guest CLI exit 0, its canonical summary,
normal process termination and offline raw collection are independently
preserved. Slot bitstream identity is not FPGA readback proof. Existing
clangd feature-test failures and unrelated build warnings remain documented
in section8; no source compile failure was hidden or weakened.

No passing tests, reference run or model request were repeated during
continuation. No new timer, tolerance change, CPU GEMM fallback, RTL/ISA or
bitstream modification, Serial benchmark, unrelated optimization, commit or
push was introduced. Historical H3-7 failures and all targeted results
remain intact. This phase ends with the report and preserved evidence.
