# H3-7: GPT-2 Paired Attention End-to-End Validation

Date: 2026-10-27.

## Final assessment

**E2E Functional FAIL. Full 256/128 Workload FAIL.**

The three hardware preflight gates passed. The single final invocation
processed the complete 256-token prompt and generated one token, then exited
with status **1** during the first decode step's PV operation. The output was
`.` with token ID `13`. This was not an EOS termination.

No second full-model invocation was launched to obtain a passing result.
No RTL, bitstream, quantization, numerical tolerance or synchronization
semantics were changed to accommodate this failure.

The preparation record below is retained verbatim. Its statements that the
request is running or awaiting results are historical and are superseded by
this final assessment.

### First E2E failure and source-derived cause

The original final-run debug log records:

```text
[POTAL_ATTENTION] node=node_26 role=QK dispatch=HARDWARE limb_jobs=72 loop_ws=72 bias_mvins=0
[POTAL_ATTENTION] node=node_28 status=ERR_UNSUPPORTED stage=attention_prepare_or_reconstruct fatal=0
```

These are lines 3344-3345 of
`/tmp/h3-7-gpt2-e2e/e2e-extract/e2e/debug-log.jsonl`.
The first line follows the successful first-decode `wqkv-0` hardware dispatch.
The second is the first backend failure in this run. The CLI then reports:

```text
ggml_backend_graph_compute_async: semantic metadata: backend execution failed
graph_compute: ggml_backend_sched_graph_compute_async failed with error -1
llama_decode: failed to decode, ret = -3
main : failed to eval
```

The source path localizes this status/stage to the PV correction-bias
representability guard:

- `ggml/src/ggml-potal/matmul/attention.cpp:398-399` requires the corrected
  runner and computes the PV column bias before issuing that stripe.
- `:351` returns `POTAL_ERR_UNSUPPORTED` unless the exact aligned V sum `S`
  is signed 25-bit, so that the correction `128*S` fits signed INT32.
- `:674-692` always supplies a nonempty corrected hardware runner.
  Therefore the missing-runner alternative does not explain this invocation.
- Runner failures preserve their own stage at `:484-493`. The generic
  `attention_prepare_or_reconstruct` stage is assigned at `:668-669`.
  Together these distinguish this host-side numerical-domain rejection
  from a failed Scale, LOOP_WS, Fence or bias-MVIN hardware command.

This localization is derived from the observed status/stage and the reviewed
control flow. The failing head, column and exact out-of-range bias were not
logged; no numerical value is invented here. The representable sum interval
is `[-16777216, 16777215]`. Earlier heads of the failed PV node may already
have completed work before the failing stripe's guard.

The failure is not the previous Scale contamination: the final run logged
no Scale conflict or issuer disable, completed 50 HP1 matrix nodes, and
reported balanced Scale writes/releases. It is also not evidence of an
incompatible bitstream opcode. QK and PV both executed successfully in
preflight and throughout the completed prefill.

The checked rejection was preserved. Supporting this operand domain requires
separate numerical-contract work; it was not bypassed or implemented during
this experiment.

### Canonical request metrics

The existing `Inference performance` output and `FINAL_INFERENCE_SUMMARY`
are the canonical sources. These numbers describe the **failed partial
request**, not a completed 128-token workload or a statistical benchmark.

| Metric | Observed value |
|---|---:|
| Actual input tokens | 256 |
| Requested output tokens | 128 |
| Actual generated tokens | 1 |
| llama-cli exit status | 1 |
| TTFT | 53051.478 ms |
| TPOT | unavailable: fewer than two generated tokens |
| Total request elapsed | 53382.682 ms |
| Canonical prefill-phase elapsed | 53017.344 ms |
| Canonical decode-phase elapsed | 321.958 ms, failed operation |
| Prefill failed operations | 0 |
| Decode failed operations | 1 |

The prefill phase contains the prompt forward and first sampling operation,
as defined by the existing implementation. Decode elapsed covers only the
failed first decode operation. It is not a TPOT sample.

Raw monotonic event timestamps, in nanoseconds:

```text
request_start    11617946000
first_token     64669424000
last_token      64669424000
request_end     65000628000
```

The summary reports `available=true`, verified latency boundaries, one
request, one token and zero TPOT gaps. Its `completed_requests=1` records a
closed request boundary; it does not mean successful inference.
`H3_7_E2E_SCRIPT_EXIT=0` and simulator PASS likewise do not override
`H3_7_E2E_EXIT=1`.

The timing source is the guest `steady_clock` monotonic timeline. The boot
reported an aclint timer timebase of 1,000,000 Hz. `GGML_GEMMINI_NPU_HZ`
was unset and the summary's `npu_frequency_hz` is null. No cycle-frequency
conversion was applied to TTFT, TPOT or total request elapsed.

Supporting diagnostics, not E2E latency substitutes:

- Prefill worker CPU: 18756.931 ms; decode worker CPU: 56.845 ms.
- CPU cycles: unavailable, reason `not_thread_cpu_counter`.
- CPU work wall: unavailable because PoTal host-stage coverage is incomplete.
  The measured subsets were 11931.674 ms and 0.890 ms.
- Attention and dense NPU device counters and cycle-derived times were
  unavailable. No accelerator frequency, DMA duration or hardware overlap
  was inferred from these CPU-side intervals.

Final summary printing and teardown are outside the request boundary.
Existing inline node/stripe instrumentation remains inside request wall time.
Zero logging overhead is therefore not claimed. No new timer, logging-time
subtraction or overlapping-stage sum was introduced.

### Observed hardware coverage and fallback

All counts below come from the final invocation's completion records and
terminal backend summaries, not from the earlier smoke tests. They are
software-observed checked command/completion counts, not device cycle counters.

| Coverage | Final-run observation |
|---|---:|
| HP1 matrix nodes / successful completions | 50 / 50 |
| Fused Q/K/V projection completions | 13 |
| Attention output projection completions | 12 |
| FFN up / down completions | 12 / 12 |
| lm_head hardware completions | 1 |
| Attention QK successful nodes | 13 |
| Attention PV successful nodes | 12 |
| Attention PV failed nodes | 1 |
| Attention total successful / failed nodes | 25 / 1 |
| Paired projection stripes | 188 |
| Paired-to-Serial fallback stripes | 0 |
| Paired Main LOOP_WS | 6160 |
| Indexed Residual LOOP_WS | 94955 |
| Index Config / Write | 94955 / 2448070 |
| Scale Write / Release | 28990751 / 28990751 |
| Paired Fence / Drain | 188 / 101303 |
| Paired readbacks / readback tiles | 8844 / 164828 |
| TLB flushes | 4613 |
| Attention LOOP_WS | 9609 |
| Completed separate-ACC QK jobs | 3528 |
| Attention bias MVINs | 9260 |

The complete prefill has all 49 required HP1 projection nodes and all
12 QK plus 12 PV nodes completed on PoTal. The first decode adds one QKV
projection and one QK completion before PV fails. The debug log contains
75 successful matrix hardware-dispatch records in total.

Required matrix CPU fallback: **0 observed in the executed prefix**.
This conclusion uses the complete prefill's expected matrix inventory and
actual hardware completion records, plus the explicit Attention assignments.
It does not treat `supports_op_mul_mat_rejected_answers=0` as a CPU execution
counter. Nodes after the failure were not executed and are not counted as
fallback. Softmax, normalization, views and other ordinary CPU operations
remain permitted.

The 188 completed HP1 stripes had no Serial fallback. QK/PV nevertheless use
their dedicated planned-GEMM Attention path, not Paired Indexed Residual.
This failed inference is not labeled fully Paired-only E2E.
The generic Serial-route `main_loop_ws=0` field does not negate the dedicated
`paired_main_loop_ws=6160` observation.

Accepted and semantically completed projection stripes both equal 188.
Scale counts balance, completed projections report successful fences, and
the PV rejection is nonfatal to the issuer (`fatal=0`). No Fence/Drain/Scale
semantics were weakened. The guest subsequently powered down normally;
no further hardware workload was run by this session.

### Actual runtime placement and workload proof

The final configuration records:

- Three generation threads and three batch threads; `separate_batch_pool=false`.
- Strict requested worker masks `[0,2,3]`; main-thread observed affinity `"3"`.
- Issuer TID 143, Linux CPU 1, Hart 0, checked affinity `{1}`
  (`debug-log.jsonl:346`).
- No affinity-setup failure was logged. Individual helper-thread CPU residency
  was not independently sampled during the final request.
- `warmup_enabled=false`, prompt cache disabled, zero matching cached tokens,
  temperature 0, seed 1, batch/ubatch 256, context 512, F16 K/V.
- Actual CLI context output confirms `flash_attn=0`.

The final boot map was CPU 0/Hart 2, CPU 1/Hart 0, CPU 2/Hart 1 and CPU 3/Hart 3.
The shell began with unrestricted mask `f`; no whole-process CPU-1 taskset
restriction was used. Successful issuer work confirms the worker masks did
not prevent the issuer from accessing CPU 1.

The runtime printed all 256 prompt token IDs. They match
`prompt-token-ids.json` element for element, and the configuration records
`initial_input_tokens=256`. A successful prefill and first token-ready event
establish that the prompt was processed, rather than merely tokenized.

The Chipyard checkout's `potal` branch tracks `aisa/potal`.
The requested repository is present as remote `aisa`
(`git@github.com:ajou-aisa/chipyard-aisa.git`); `origin` is the upstream
`https://github.com/ucb-bar/chipyard`. The software origin is
`git@github.com:ajou-aisa/llama.cpp-gemmini.git`.

### Complete output and quality assessment

The unmodified generated output is exactly:

```text
.
```

The complete generated token ID sequence is `[13]`. Token 13 is valid in
the 50257-token vocabulary and is not EOS token 50256. `iconv` validated the
original output as UTF-8 with exit 0.

One punctuation token is insufficient to establish semantic continuation
quality or exclude a later repetition collapse. No corrupt output bytes were
observed. The backend explicitly rejected a PV bias domain and aborted before
a second token was sampled; this is not an EOS-shortened success.
The error path poisons the failed logical output before aborting, so absence
of NaN anywhere in the failed graph is not claimed.

Runtime sampling settings match the deterministic request. No replay or
additional inference was run to test repeatability. No existing CPU reference
for this exact prompt was available. Model quality, quantization quality and
long-form generation quality remain unvalidated by this one-token output.

### Preserved final artifacts and control limitations

Original final-run files were extracted read-only after shutdown:

```text
/tmp/h3-7-gpt2-e2e/e2e-extract/e2e/llama-cli.out
/tmp/h3-7-gpt2-e2e/e2e-extract/e2e/llama-cli.err
/tmp/h3-7-gpt2-e2e/e2e-extract/e2e/debug-log.jsonl
/tmp/h3-7-gpt2-e2e/e2e-extract/e2e/cycle-log.jsonl
/tmp/h3-7-gpt2-e2e/e2e-extract/e2e/semantic-graph.jsonl
/tmp/h3-7-gpt2-e2e/e2e-extract/e2e/exit.status
```

The final raw-file hashes are in
`/tmp/h3-7-gpt2-e2e/failed-run-sha256.txt`. Selected SHA256 values:

| Artifact | SHA256 |
|---|---|
| CLI stdout | `cdb4ee2aea69cc6a83331bbe96dc2caa9a299d21329efb0336fc02a82e1839a8` |
| CLI stderr | `63e8660902e7fe409f78a82330bca21d90093487deb5bd0c1dad93e15dc63361` |
| Debug JSONL | `50867d7bd1b9e73e9680a490c14b52c8ac37ea1229d978ad3197581fa36008d2` |
| Cycle JSONL | `d1d474e64a341499ae6e638f6a66dd061ffa87a2a6fad5e92091dd1886639490` |

The actual executed script was separately extracted as
`/tmp/h3-7-gpt2-e2e/executed-inference.sh`. Its SHA256
`42f32fe4b6f2d000615f12243cabb3d9063cdc25c0dbae1fc63450469c696e76`
matches the saved host script. Its working directory is `/root/h3-7-e2e`;
the complete CLI argument list is preserved in Section 9 below.

Derived copies of canonical records and assessments:
`final-inference-summary.json`, `inference-configuration.json`,
`hardware-coverage.json`, `runtime-token-proof.json` and
`generated-token-ids.json`, all under `/tmp/h3-7-gpt2-e2e/`.
The original logs, not the derived copies, remain authoritative.

An additional controller advanced these shared run artifacts and booted the
final guest while exclusive ownership was unresolved. The ownership question
received no answer. This session did not launch a duplicate or reset that
live guest; it observed and recorded the existing single attempt.

Before launch, host `df` reported the 16 GiB clone at 100% usage. A planned
growth stopped when its mount was no longer present; no growth occurred.
That storage condition is a capture risk, but the retained evidence does not
identify it as the cause of the explicit PV representability rejection.
The original slot rootfs and tested binaries were not modified.

### PASS-criteria disposition and next boundary

| Required condition | Result |
|---|---|
| llama-cli exits successfully | FAIL: exit 1 |
| Exactly 256 input tokens processed | PASS |
| At least one output token | PASS: one token |
| Required matrix hardware execution observed | PASS for completed prefill and executed decode prefix |
| QK and PV both complete on PoTal | Both complete during prefill; first-decode PV fails |
| Paired Main and Indexed Residual observed | PASS |
| No PoTal node execution failure | FAIL: one PV node |
| Valid TTFT and TPOT | TTFT valid; TPOT unavailable with one token |
| Generated continuation quality established | Not established |
| Exactly 128 output tokens | FAIL: one token |

The experiment stops here with a failed functional verdict, not a passing
partial-length benchmark. No Qwen2, Llama-3.2, alternate DIM/quantization,
frequency sweep, Serial comparison, repeated E2E run or optimization campaign
was started. The next authorized investigation should address the PV bias
domain boundary before any new full-model validation is considered.

## Historical preparation record

Sections 1-10 below are the preserved pre-completion record. Their
in-progress and awaiting-results statements are superseded by the final
assessment above.

## 1. Current experiment status

**In progress: all preflight gates passed; exactly one final request has started.**
No E2E Functional PASS or Full 256/128 Workload PASS is declared yet. The
request's exit status, canonical timing, generated tokens, final hardware
coverage and text-quality assessment must come from that existing run.
There is no authorization to rerun the full workload to obtain a passing result.

## 2. Previous failure: confirmed cause and evidence

The API smoke expected the wrong status for a malformed DIM16 Scale key.
In Chipyard's
`generators/potal/src/main/scala/potal/PoTalScaleCommandIngress.scala:99`,
`columnWidth = log2Ceil(dim)`. DIM16 uses four column bits. With 256 Scale
slots and eight generation bits, column 16 sets reserved key bit 20.
The RTL checks encoding before column, address and carrier validity
(`:170-194`, `:421-439`). The correct response is therefore
`POTAL_HW_INVALID_ENCODING = 6`, including when the carrier is also invalid.

The checked API still rejects column 16 before issuing a command:
`ggml/src/ggml-potal/potal.c:140-150`. Its argument validation was not changed.

Original UART evidence:
`/tmp/h3-7-gpt2-e2e/prior-uartlog:378` records
`got=6(HW_INVALID_ENCODING) expected=1(HW_INVALID_COLUMN)`.
The immediately following Paired Quick failure is at lines 389-396.
The failed API process exits before releasing slot 3/generation 9/column 0
and slot 3/generation 7/column 15.

The original rootfs debug log was extracted read-only to
`/tmp/h3-7-gpt2-e2e/prior-debug-log.jsonl`:

- Line 5: issuer initialized on Linux CPU 1, Hart 0, affinity `{1}`.
- Line 6: startup TLB flush and Fence passed.
- Line 13: the first Serial-control warmup Main failed at `scale_write`
  with `POTAL_HW_SCALE_CONFLICT`; no LOOP_WS was issued. It completed
  48 successful Scale writes and 48 cleanup releases.
- Line 14: the issuer disabled itself after that Main failure.
- Line 15: the stripe recorded the fatal failure. Later Paired attempts
  inherited the disabled issuer; they did not establish a Paired ISA failure.

The tiler writes `slot = base + block * J + j`, in block/J/column order
(`ggml/src/ggml-potal/matmul/tiling.cpp:548-590`). For the logged base-zero
I1/J2/K4 warmup, 48 successful writes fill slots 0-2. The next write reaches
slot 3/column 0, matching the API smoke's retained generation-9 entry.
This connects the actual hardware status and failure stage to the aborted
API test, rather than assuming that issuer_ready=0 was the original cause.

Evidence SHA256:

| Artifact | SHA256 |
|---|---|
| Original UART copy | `d6b7a82109437c9e2c5f4fc33393c372675378f1883a8062e2f564e92509ec56` |
| Original debug JSONL | `e964fd1712e5776e57450d21c9d146a4279f48388b05940cb1ea5592b3fe117a` |

## 3. Test-only corrections and preserved work

Changed:

1. `ggml/src/ggml-potal/tests/potal-api-smoke.c`: raw column 16 and raw
   column 16 plus invalid carrier now expect INVALID_ENCODING. The latter
   label now says encoding-before-carrier. Checked-API negative tests remain.
2. Chipyard's `generators/potal/software/gemmini-rocc-tests/bareMetalC/potal_scale_control.c`:
   the same obsolete DIM16 raw-column expectation was corrected.

No production API, quantization, tolerance, Scale/Fence/Drain semantics,
RTL or FPGA image was changed. No project commit or push was performed.

The existing 33-line backend CMake change for E4C1/E4C2-0 was preserved:
SHA256 `ec22288b5a4ff2ac1a62cb7cfb8d26611ea33e7203311817577ddffae1af488a`.
The existing untracked E4C1/E4C2-0 source/report files and model directory
were retained. Existing tested build directories and guest binaries were not
rebuilt or overwritten.

Full tracked software differences at build time:
`/tmp/h3-7-gpt2-e2e/software-working-tree.diff`, SHA256
`bbdb11e0decad396520db05ef4b5d3dea335741c0ec53e0db10cbd2ca127d29c`.
The Chipyard test-only difference is
`/tmp/h3-7-gpt2-e2e/rtl-repository-test-only.diff`, SHA256
`e987336626c96022ca9af0759edc814849a4519c0855cdd4005380e6bf2dd2dd`.

## 4. Hardware recovery and preflight

The original target had already powered off. A separate sparse clone of its
16 GiB rootfs was created at `/tmp/h3-7-gpt2-e2e/run/rootfs.img`.
New software and results use only `/root/h3-7-e2e` inside that clone.
The original slot rootfs, UART and tested binaries were preserved.

The unchanged slot driver was restarted with `+reset-pulse-length0=50`.
Driver initialization programs the ResetPulseBridge remaining-pulse register
(`sim/midas/src/main/cc/bridges/reset_pulse.cc:36-38`). The bridge drives
the FireSim harness reference reset; PoTal Scale validity uses RegInit(false).
This is a target reset, not a claim that a TLB flush clears Scale state.
The corrected API's first slot-3 writes subsequently passed.

All three required gates ran once after recovery:

| Gate | Observed hardware result |
|---|---|
| Corrected API smoke, `taskset -c 1` | PASS; raw negative statuses, K16/K32 golden comparisons, Fence and final Scale release passed |
| `potal-paired-smoke --quick` | PASS; issuer ready on CPU 1; real Paired Main and Indexed Residual commands completed; zero failed nodes |
| `potal-attention-smoke` | PASS; actual QK and PV integer-domain hardware results, raw QK epochs, Scale balance and fences passed |

For the RTL-fence Paired case alone: one stripe, Main LOOP_WS 1,
Indexed Residual LOOP_WS 7, Index Config 7, Index Write 150, readbacks 2,
Fence 1, drains 9, and Scale Write/Release 2112/2112. Golden comparison passed;
Serial fallback stripes were zero.

The complete existing Quick test includes its internal Serial numerical
control and conservative-fence control. Its aggregate counters were:
Paired stripes 3, Paired Main LOOP_WS 3, Indexed Residual LOOP_WS 16,
Index Config 16, Index Write 362, Fence 20, drains 39,
Scale Write/Release 6080/6080, failed nodes 0, issuer_ready=1, issuer_cpu=1.
Its printed internal timing comparison was not used as a Serial benchmark.

Attention hardware smoke covered QK K64/K33/K65/K4097 and PV K17/K33/K65,
plus the K64 PV cancellation/saturation witness. QK low/high signed32 raw
results and overwrite/accumulate/store generations were independently checked.
The K4097 case completed 12 commands, 4 stores and 12 fences across four jobs.
These observed operations establish compatibility for the selected executable's
legacy/Paired/Index/Scale/Fence and PV bias-MVIN paths. No newer resident ISA
operation was required by this experiment.

Unmodified preflight outputs and JSONL files:
`/tmp/h3-7-gpt2-e2e/preflight-results/`.
Preflight UART and driver log:
`/tmp/h3-7-gpt2-e2e/run/{uartlog,driver.log}`.
The guest shut down normally and the simulator reported PASS before collection.
A separate clean boot then started the single final request; no preflight or
full-model benchmark was repeated on that boot.

## 5. Execution identity

| Item | Identity |
|---|---|
| Software branch / HEAD | `potal` / `8fb19aa5b66ace4cc50014581e5b3fe5ad7fc546` |
| Current Chipyard branch / HEAD | `potal` / `1dfbe85aea66e0afc0e11072b983335a475a5b14` |
| Hardware configuration | `FireSimQuadRocketSinglePoTalD16Config`, U280, Quad Rocket, one PoTal, no network |
| RTL parameter header SHA256 | `b4cd8b9efef9f3cd9b598ae3cc84af6478f83e628ed73d94a77fb1c452df1fc7` |
| Slot firesim.bit SHA256 | `f4c465a451385cbda44c254fb038fc2ede3a25ea12d5e829c2a501b7d495b3de` |
| Slot driver SHA256 | `ebbc201ee188d1553912aafb3835b5637f247eed60a696499dcf74a27b110a0d` |
| Bitstream metadata FireSim commit | `5c3458a3f54917ab9502bda4a8b3cc3dd513304d` |
| Selected model SHA256 | `cf23a45e4db23736d7d1f67ab42018803a198cfcb01db36daca69ff56bdbc7d4` |
| Model size / actual parameter count | 159,821,472 bytes / 124,439,808 parameters |
| Model quantization | GPT-2 Q8_HP1 GGUF; this existing artifact is not a 163-million-parameter checkpoint |

The bitstream hashes identify deployed slot artifacts, not an FPGA readback
hash. The metadata names the same deployment quintuplet. Actual ISA gates
passed on the deployed target. A source-to-bitstream readback proof was not
invented from the current checkout SHA.

The source header is
`generators/potal/software/gemmini-rocc-tests/include/gemmini_potal_params.h`.
The generated adapter has DIM16, signed A8W8, four banks, 4096 rows per bank,
1024 ACC rows, 256 Scale slots and 32 Index entries. Its source SHA matches
both the original failing guest and the rebuilt passing executable.

New binary SHA256:

| Artifact | SHA256 |
|---|---|
| llama-cli | `c2800894c8cfd91d2d022e6f2969b86f3873c287cc3f4736673c191c0c40ebce` |
| libllama.so | `1eb02079e639c4cf5361b6006998d9430e7e1183d48ae20616f5c5e30f358bbe` |
| libggml-potal.so | `56165a8e51be5185ce7997f936596f4aa74d9b7bd3967e2b8911953dadd57254` |
| potal-api-smoke | `26304c067abf024cd45aa88d3dc4292eb90c20ea46007b78757ae391975926a7` |
| potal-paired-smoke | `bc2fb0c98f5a7728d6431154c7296472edd60d395ec5516f415a37fdf1a2e439` |
| potal-attention-smoke | `d25d8789dfea7cbdc67dd905334014cdba07595752199994cb7b657b61d65352` |

Authoritative manifest: `/tmp/h3-7-gpt2-e2e/identity-manifest.txt`;
the corresponding guest hashes are in `preflight-results/guest-sha256.txt`.

## 6. Verified build and routing configuration

The selected build is `build-potal-h3-7-e2e-d16`, not one of the older
similarly named directories. Its CMakeCache and generated flags establish:

```text
GGML_POTAL=ON
GGML_GEMMINI=OFF
GGML_GEMMINI_ACTIVATION_QUANT=EXSIA
GGML_GEMMINI_ACTIVATION_BITS=8
GGML_GEMMINI_WEIGHT_BITS=8
GGML_GEMMINI_BLOCK_SIZE=32
GGML_GEMMINI_ENABLE_RMD=ON
GGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE
GGML_GEMMINI_DEFAULT_RMD_BACKEND=WS
GGML_POTAL_PAIRED_AUTO=ON
GGML_POTAL_ATTENTION=1
GGML_GEMMINI_EXSIA_DEFAULT_MODE=SEQUENTIAL
GGML_OPENMP=OFF
GGML_GEMMINI_ENABLE_OPENMP=OFF
LOG_CYCLE=1
LOG_DEBUG=1
CYCLE_DETAIL=0
CYCLE_SIM=0
POTAL_PLANNING_DIAGNOSTICS=OFF
```

All four RISC-V targets built with exit 0 in that separate directory.
Build/configuration logs are `/tmp/h3-7-gpt2-e2e/cross-{configure,build}.log`.
The required environment sourcing exposed an existing stale Chipyard-local
Conda path. Activating the already installed `/home/alveo/firesim/.conda-env`
after sourcing resolved tool access without changing env.sh.

The CLI initializes ACCEL backends before CPU
(`src/llama-context.cpp:137-169`). GPT-2 Q/K/V use one fused projection;
output projection, FFN up/down and lm_head use the HP1 route when eligible.
The graph explicitly tags and assigns QK/PV to POTAL when F16/F32 KV and F32 Q
are eligible (`src/llama-graph.cpp:1252-1315`). Unsupported required Attention
offload asserts rather than silently assigning these tagged nodes to CPU.

QK/PV use the dedicated planned-GEMM Attention implementation. QK completes
independent low/high stationary jobs and checked host reconstruction; PV uses
its checked ACC-bias path. Their use of paired-session ownership APIs does not
make them Paired Indexed Residual GEMMs. That distinction is retained in the
final coverage assessment. Softmax and unrelated CPU operators are permitted.

Flash Attention is disabled by default in the actual CLI help, and its enable
flag is absent. `LLAMA_ARG_FLASH_ATTN` is unset. No KV/op-offload disable flag
is used. The exact final option set passed the actual RISC-V CLI's --help
parser with exit 0; see `preflight-results/validated-command-help.txt` and
the preflight UART's `H3_7_COMMAND_HELP_EXIT=0`.

## 7. Existing timing contract audit

The requested four files were inspected before the first source edit:
`tools/main/main.cpp`, `ggml/src/ggml-gemmini-utils/src/performance.cpp`,
`ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp`, and
`src/llama-context.cpp`.

- Model/context creation precedes tokenization and request start
  (main.cpp:411, :576, :906-915).
- --no-warmup disables the conditional model warmup in common/common.cpp:1007.
- Decode synchronizes before sampling (main.cpp:1081-1084;
  llama-context.cpp:388-389).
- Token ready follows sampling and acceptance (main.cpp:1128-1134).
- Request end precedes final performance printing and cleanup (main.cpp:1401).
- The reducer uses the existing token-boundary formulas; it does not derive
  request latency from Rocket cycle differences or worker/NPU cycle sums.

```text
TTFT = first_token_ready - request_start
TPOT = (last_token_ready - first_token_ready) / (generated_tokens - 1)
```

The timestamp source is std::chrono::steady_clock nanoseconds
(`ggml/src/ggml-gemmini-utils/src/cycle.cpp:75-79`). In FireSim Linux this
is the guest CPU-side monotonic timeline, not elapsed workstation wall time.
Both boots reported an aclint timer timebase of 1,000,000 Hz.

The canonical source is the existing Inference performance output and
FINAL_INFERENCE_SUMMARY. Ordinary llama_perf_context_print values are only
supporting data. Final summary processing and teardown are outside the
request boundary. Existing inline instrumentation/output between token-ready
events remains part of request wall time; no synthetic logging-time subtraction
was added. Per-command raw tracing and profiling were disabled.

GGML_GEMMINI_NPU_HZ is unset. It cannot rescale these monotonic TTFT/TPOT values.
Any separately available NPU cycles require their own configured frequency and
label; driver Host Frequency/FMR/fastest-target-cycle summaries are not NPU
request timing. No new rdcycle-based measurement was introduced.

## 8. Exact workload and tokenization proof

Prompt file: `/tmp/h3-7-gpt2-e2e/prompt-256.txt`.
File SHA256: `7e13482ec87666be18c7e5e14e04b14f4d09405f1099cfd68d92696c5469cd78`.
The CLI -f parser removes one trailing LF (common/arg.cpp:1500-1507).
The effective text SHA256 is
`4f0967784705adea9d7798c4823d8456edfefe9ece007209859686947e652e8a`.

The current vocabulary-only llama-tokenize loaded the selected GGUF and
verified the CLI-equivalent effective text has exactly 256 tokens, including
automatic special-token policy. Zero automatic special tokens were inserted.
The raw tokenizer output is `prompt-tokenization.txt`; complete proof and IDs
are in `prompt-token-ids.json` under the same directory.

```text
When the storm finally passed, the small coastal town woke to a silence that felt unfamiliar. For three days, rain had rattled the windows and waves had broken over the harbor wall. Now sunlight reached the fishing boats, and the streets smelled of salt and wet wood.

Anna opened the door of her father's repair shop and found a battered wooden box on the step. A brass plate on its lid carried the name of a ship that had vanished before she was born. Inside lay a compass, a folded chart, and a letter addressed to the town librarian.

She carried the box across the square, where neighbors were clearing fallen branches and checking on one another. The library had lost several roof tiles, but its tall front windows were intact. Mr. Ellis stood beside the entrance with a broom in his hand.

He recognized the ship's name at once. His expression suddenly changed as he carefully unfolded the old letter, and he asked Anna to close the door. Somewhere beyond the empty reading room, a clock struck nine. He placed the chart on a table and pointed to a narrow inlet north of the harbor. The ink was faded, but a small red mark remained beside the place where a lonely old lighthouse once stood. Then he began to explain
```

Exact input token IDs:

```text
2215 262 6388 3443 3804 11 262 1402 17475 3240 19092 284 257 9550 326 2936 22594 13 1114 1115 1528 11 6290 550 48588 262 9168 290 9813 550 5445 625 262 25451 3355 13 2735 19606 4251 262 12478 16659 11 290 262 6483 37609 286 8268 290 9583 4898 13 198 198 31160 4721 262 3420 286 607 2988 338 9185 6128 290 1043 257 33613 13510 3091 319 262 2239 13 317 20422 7480 319 663 19789 5281 262 1438 286 257 4074 326 550 23717 878 673 373 4642 13 14384 3830 257 31855 11 257 24650 8262 11 290 257 3850 9469 284 262 3240 300 35808 13 198 198 3347 5281 262 3091 1973 262 6616 11 810 12020 547 17304 9292 13737 290 10627 319 530 1194 13 383 5888 550 2626 1811 9753 19867 11 475 663 7331 2166 9168 547 16572 13 1770 13 21299 6204 13970 262 10384 351 257 47085 287 465 1021 13 198 198 1544 8018 262 4074 338 1438 379 1752 13 2399 5408 6451 3421 355 339 7773 34660 262 1468 3850 11 290 339 1965 11735 284 1969 262 3420 13 43808 3675 262 6565 3555 2119 11 257 8801 7425 5193 13 679 4624 262 8262 319 257 3084 290 6235 284 257 7135 287 1616 5093 286 262 25451 13 383 16882 373 24887 11 475 257 1402 2266 1317 6150 13970 262 1295 810 257 21757 1468 46371 1752 6204 13 3244 339 2540 284 4727
```

The selected model and prompt hashes also matched inside the guest. The final
boot independently confirmed CPU 1/Hart 0 and an unrestricted shell mask f.
The earlier multicore smoke sampled workers on CPUs 0, 2 and 3 without migration;
the new Paired/Attention smoke confirmed the issuer on CPU 1.
The final request uses strict generation/batch pool mask 0xD (CPUs 0, 2, 3),
three threads per active pool, and no whole-process taskset restriction.
The issuer finds Hart 0 dynamically, pins itself and checks the resulting
kernel affinity before RoCC issue. Final request placement evidence must also
be checked in its existing configuration/issuer records.

## 9. Single final inference command

The guest ran `sh /root/h3-7-e2e/inference.sh` once. Before llama-cli, the
script verifies saved gate markers and CLI options and atomically creates
`results/e2e/started-once`. Its exact saved source is
`/tmp/h3-7-gpt2-e2e/inference.sh`.

```bash
export LD_LIBRARY_PATH=/root/h3-7-e2e/bin
unset POTAL_PLANNING_PROFILE POTAL_PAIRED_PROFILE POTAL_RAW_TRACE
unset GGML_GEMMINI_NPU_HZ LLAMA_ARG_FLASH_ATTN
export GEMMINI_LOG_DIR=/root/h3-7-e2e/results/e2e
export GEMMINI_MATMUL_MODE=STRIPE_PIPELINE
export GEMMINI_RMD_BACKEND=WS
export GGML_POTAL_PAIRED_AUTO=ON
export GGML_POTAL_PAIRED_FENCES=RTL

GGML_SCHED_DEBUG=2 ./bin/llama-cli \
  -m /root/workspace/llama.cpp-potal/models/gpt2/gpt2.Q8_HP1.gguf \
  -f /root/h3-7-e2e/prompt-256.txt \
  -n 128 -c 512 -b 256 -ub 256 -t 3 -tb 3 \
  --cpu-mask 0xD --cpu-mask-batch 0xD --cpu-strict 1 --cpu-strict-batch 1 \
  --temp 0 -s 1 -no-cnv --no-warmup --no-context-shift \
  --no-display-prompt --simple-io --verbose-prompt \
  > "$GEMMINI_LOG_DIR/llama-cli.out" \
  2> "$GEMMINI_LOG_DIR/llama-cli.err"
```

Warmup, conversation templates, prompt-cache reuse and context shifting are
disabled. EOS is respected; --ignore-eos is not used. The output budget is 128,
not a promise that the model will reach 128 tokens before EOS.

Final boot UART and driver paths are
`/tmp/h3-7-gpt2-e2e/e2e-run/{uartlog,driver.log}`.
While running, raw CLI and JSONL results reside inside the guest at
`/root/h3-7-e2e/results/e2e/`. The rootfs must not be mounted concurrently
with its driver. Host extraction and the final assessment follow clean exit.

## 10. Required final assessment still awaiting the running request

The existing request must supply its actual input/generated token counts,
exit/termination reason, canonical TTFT/TPOT, total request latency, phase times,
generated text/IDs, hardware counters and named matrix dispatch evidence.
No provisional numbers are substituted here.

For GPT-2, the expected required matrix nodes per forward are 49 HP1 projection
MatMuls and 12 QK plus 12 PV nodes. These are expectations, not observed final
counts. Final coverage must separately report actual Paired Main/Residual loops,
Index Config/Write, Scale Write/Release, fences/drains, fallback stripes,
failed nodes, FFN and lm_head dispatch. Support-query rejection counts are not
CPU execution/fallback counts.

The final text assessment must use the unmodified output and generated token
IDs, check UTF-8 and execution-related corruption, and report semantic relevance,
repetition, deterministic sampling and any early EOS. No existing CPU reference
for this newly written prompt has been identified. No Serial FPGA performance
comparison or future-model experiment was started.

