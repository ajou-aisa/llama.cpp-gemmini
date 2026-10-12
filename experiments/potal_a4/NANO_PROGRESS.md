# Nano continuation

This branch is the test workspace `work/potal-attn-im2p-nano`. On 2026-10-12 the
user requested a commit and normal push of the existing work and evidence on
this branch for local continuation. Do not merge into develop/hotfix, discard
existing changes, or start another implementation/experiment during this
transfer. Moving the work to hotfix remains a later task.

At the initial audit, fetched `origin/develop` was
`3b2eaa3cb6f7adc36094b7f78662141825cc4ef4`. The product tree at the pre-change
HEAD `f588ad4bb280147a152c0d23f624526fff64d45a` was identical to that revision:
the 81 differing tracked paths were only the PoTal
experiment directory and handoff document. Nano's local `develop` is stale at
`e5eb928e80382cbc89d8ae1d7953d502ffd3632d` (359 product-file differences); it was
not used as the baseline. The product tree was unchanged at that initial audit.
It now contains the A4 source-zero exclusion patch in `quants/act/exsia/local.cpp`
and its tests; no other product algorithm change has been made in this session.

The accepted policy remains `potal-a4-preserve-v1`: selective linear, all required
Q limbs, original geometry P uint8, K [-128,119], and block/channel V4. Product
attention dispatch and fused limb emission remain incomplete.

## Adopted zero policy and specification audit

The user explicitly adopted excluding source zero from the sigma population
after reviewing the result below. Keep this change in future A4 baselines;
it is no longer waiting for an adoption decision. A nonzero source that rounds
to q=0 remains included. A8 is outside this change.

`results-outlier-zero/` contains the paired native product measurements:

| Scope | Before | After |
| --- | ---: | ---: |
| Sigma population | Includes source zero | Excludes source zero (adopted) |
| Context / chunks | 512 / 10 | 512 / 10 |
| Input / scored tokens | 5,120 / 2,550 | 5,120 / 2,550 |
| PPL | 40.18882947104453 | 40.20531899910632 |

The relative change is +0.0410301%; four chunk NLLs improve and six worsen.
Model/text/binary/source hashes and chunk NLLs are in `inputs.json`, `before.json`,
`after.json` and `comparison.json`. The new population fixture failed before the
patch; the patched full/tail/reference fixture and nearby ExSIA CTest gate passed
5/5. Both PPL builds use A4/W4/DIM32/BK32, CPU main and CPU-direct RMD, FP attention,
batch/ubatch512 and 4 threads. Generated geometry is `(4,4096,1024)` in both, not
the original `(4,4096,512)`. The pair measures this policy change on the existing
product path; it does not prove a4nks, original geometry or RTL parity. CPU elapsed
times are not performance comparisons because compilation overlapped portions of
measurement. A layer/token-level cause analysis of the PPL difference has not
been run; changing U can change selection and scales but does not guarantee an
improvement in PPL.

The product's folding stage may still mark a zero as a D>0 correction candidate.
That is permitted by the supplied corr rule and does not make zero a sigma outlier
or a nonzero upper digit. Source-zero population, top, sigma, correction candidates
and actual upper work must be reported separately.

The re-supplied A-D specification has been mapped to the ten implementation tasks
in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Missing explicit gates now
include geometry, full Q4_HP1 scale decoding, 10-chunk paired comparisons, timing
prediction versus the actual supported opcode, and cleanup after each adopted
increment. The seven-test scalar oracle is partial task 1 evidence; source-zero
handling is partial task 4 evidence. No complete product task is checked off.

The current checkpoint has 48 Q4_HP1 tensors, 2,654,208 blocks, no zero blocks,
and no nonzero block with a nonpositive/non-power-of-two channel_scale (direct
GGUF field scan). Its spec scale can therefore be represented as
`thetaW=m+ilogb(channel_scale)`. Native decoder/full-GEMM comparison remains a gate;
the field scan alone does not establish that integration.

## Existing implementation audit before further edits

The user requested checking existing code before adding implementations.
[The reuse map](IMPLEMENTATION_PLAN.md#existing-implementation-and-reuse-audit)
now traces ExSIA local/folding stages, balanced digits/bitmap packets, run-aware
packing, validated weight readers, matmul output ownership, the raw IM2P provider,
the shared attention graph, geometry and evaluation/cycle records to their
existing owners and tests. Existing `BLOCK` quantization, contiguous F16 handling
and saturating SCU/checked-int64 compose were inspected: their contracts alone
do not implement the requested K/P/V, strided views or exact wide accumulation.

Tasks 1-10 now extend existing source/test owners first. Unconditional new PoTal
executor, attention module, test executable and duplicate contract-document
instructions were removed. A small private extraction remains possible when a
specific missing responsibility and real callers justify it. Recheck callers
and remove replaced unused paths after each adopted increment. This audit changed
planning/handoff documents only; it adds no numerical or runtime validation claim.

## Local environment and inputs

- Linux `nano2`, aarch64, kernel `6.8.12-1021-tegra`, 7.4 GiB RAM.
- Python 3.12.3, NumPy 1.26.4. Isolated `/tmp/potal-nano-venv` adds tokenizers
  0.23.3. `rtk` was not installed, so commands ran directly.
- `models/gpt2-base/gpt2.Q4_HP1.gguf`, downloaded
  `models/gpt2-base/tokenizer.json`, and both WikiText-2 sources match every SHA256
  specified in the handoff. Model/tokenizer locations differ from the original
  model path in the handoff. Tokens and captures are local generated artifacts.
- IM2P source: `5da6a4cf0d1da38211b14cacf409bb1abc8d458a`;
  Gemmini header repository: `fcfcf5ce7c4670bd789365f11810177f97016f81`.
  The installed IM2P checkout is untouched; the compatible isolated source used
  for the completed build is `20b82f058aa8b0e69c48d1d00790d8e19eaa6f66` below.

## Numerical and native preparation evidence

Linux `ru_maxrss` is now normalized from KiB to bytes in both `run.py` and
`native_probe.cpp`. Darwin retains its byte interpretation. Historical results
are untouched.

`tests/potal_a4_reference.py` is an independent Python-integer/Fraction primitive
oracle, not a complete reference forward. Seven tests cover full int32 radix
digits, selective +8, K and P code reconstruction, real-tail/source-zero sigma
populations, strict integer arithmetic, catastrophic cancellation, double
rounding, subnormal, infinity and signed-zero publication. The oracle is registered
in the experimental probe CTest. It does not complete plan task 1's linear/full
masked-attention oracle and model fixture requirements.

`results-nano/verify.log`: 141,076 integer roundtrips.
`results-nano/ctest.log`: 5/5 passed, including bitmap LOG0/LOG1.
`results-nano/native-packets.json`: 30 fresh-process probes, three repetitions per
fixture and representation. Median peak RSS and elapsed seconds:

| Fixture | Clipped retained MiB | Direct retained MiB | Clipped RSS MiB | Direct RSS MiB | Clipped seconds | Direct seconds |
|---|---:|---:|---:|---:|---:|---:|
| narrow | 10.0000 | 2.0000 | 13.1133 | 11.2500 | 0.073139 | 0.063907 |
| sparse | 12.0899 | 4.0843 | 16.2617 | 11.2500 | 0.080605 | 0.071224 |
| dense | 17.0875 | 6.9606 | 22.2422 | 11.6133 | 0.276462 | 0.254331 |
| wide | 17.0875 | 4.8336 | 22.2422 | 11.2500 | 0.295756 | 0.200147 |
| carry | 27.7223 | 19.7223 | 33.7227 | 25.7227 | 0.663819 | 0.644372 |

These are the existing **clipped-main/full-correction versus direct-all**
representation probe, with its scalar bridge and reconstruction scratch still
present. They are neither selective-policy parity nor fused-emitter or Gemmini
timing results. The probe uses stripe32, DIM32, and the generated software header
ACC_ROWS=1024. Hardware schedule parity is not claimed.

The fresh `results-nano-baseline/test512-original_a4nks.json` completed 8 chunks,
2,040 scored tokens: PPL **35.378030265292224**, top1 **0.3475490196078431**,
peak RSS **913,309,696 bytes**. Token SHA256 matches the historical run.
The historical macOS PPL was 35.37806; cross-platform bitwise equality is not
claimed. This is the unchanged original-policy NumPy forward, not optimized
product/native inference or an accuracy-equivalence pass for a new implementation.

## Further multiplication-reduction candidates

The user asked whether the many digit products can be reduced. Reused the existing
`profile_attention.prepare` / `profile_gemm.unpack` on the saved context512 capture
(144 heads), verified its SHA against the existing report, and matched the old
fragment totals before comparing schedules. Results and digit histograms are in
[optimization-opportunities.json](results-nano-profile/optimization-opportunities.json).
This is one-chunk component analysis, not a new PPL comparison or native timing.

The supplied cost formula uses the stripe-wide union of active `(lane,row)` pairs
for every surviving K block. Counting only pairs active in each original BK32
block gives `sum_b ceil(m_b/32)*ceil(N/32)*ceil(|C_b|/32)`, with the same codes,
scales, original block boundaries and column unions:

| Component | Reference schedule | Block-local candidate |
| --- | ---: | ---: |
| Q upper fragments, one K pass | 147,456 | 147,168 |
| P upper fragments | 44,724 | 13,782 |
| Total QK / full main | 6.000 | 5.9921875 |
| Total PV / full main | 1.6066081 | 1.1869303 |

P upper work falls 69.1843% and total PV fragments fall 26.1220% in this analytical
comparison. Nine selected heads / 108 stripe upper sums were recomputed with
block-local packed dots and scatter, after the existing int64 bound check; their
integer sums before float conversion matched the original lane loops exactly.
Main and zero-point terms were unchanged. This does not certify native execution,
the full graph, PPL or latency. Gather/call overhead can offset fragment savings.

**Existing implementation to reuse:** `RmdBitmapBuilder::finish` already creates
per-block lane groups/row IDs, and `rmd-executor.cpp` has a compact-dot path using
those row IDs. Its run-aware path instead builds the global row union. The
candidate is to preserve/use the fitting existing path during A4 attention
integration, not add another compactor or claim the current native product is
69% slower. Nominal spec counters and actual submitted counts must remain
separate if this schedule is adopted; the existing acceptance contract is not
silently changed by this experiment.

The later user decision selects fully masked QK tile skipping as the next
implementation scope; only its investigation is complete (see the final section).
At 512 tokens/DIM32, ordinary causal prefill retains 136 of 256 square tiles
including the diagonal. Sparse sign/add handling of upper digits remains an
experimental candidate, evaluated below. In this capture
97.7855% of nonzero Q lane-2 digits are +/-1; nonzero P lane-2 digits are all +1.
These identities can replace multiplications with additions on a suitable host
or vector path, but current signed-4 GEMM hardware does not automatically gain
cycles from them. Transfer/merge costs and actual RTL cycles need measurement.
For PV mask skipping, adjust the weighted zero-point column-sum domain too;
otherwise dropping main=-8 terms breaks cancellation. Lower precision/limb caps
remain unadopted because they change reconstructed operands.

## IM2P build correction and timing authority

**Subsequent raw-provider admission:** the selected manifest and native link now
pass. The corrected `im2p_raw_probe.cpp` passes all 15 independent integer-dot
checks (K1/K31/K32, row/column tails, three repetitions, normal and unit digits),
and returns integrated RTL cycles. The previous pointer-only public API call
worked for K1 but failed for K31: in `sim/src/simulator/matmul.rs`, absent scale
metadata causes `scale_block_size=K`, whereas the HP1 raw runtime requires 1.
The existing product `Im2pCompactDot` provider path preserves explicit block_size;
the probe now uses that same public provider route with checked weight/output
callbacks. No dummy scale buffers or simulator source changes were added.

For 32x32x32, both ordinary signed-nibble inputs and zero/+1/-1 inputs take
**492 RTL cycles** in each of three repetitions. A change in coefficient values
alone does not reduce Gemmini latency. CPU transfer/packing/merge, concurrent
scheduling and a full native a4nks execution remain separate requirements.
Evidence is in `results-unit-add-llama512/rtl-raw.json`. The earlier construction
history below remains useful; its former pending raw-admission gate is now met.

The first provisioning command incorrectly passed the header repository as
`GEMMINI_ROOT`; the cache fingerprint requires the llama source repository there.
The corrected arguments use separate `GEMMINI_ROOT` and `GEMMINI_PARAMS_ROOT`.

The next attempt placed artifacts inside the llama Git repository. IM2P's
`gemmini_vendor.py` calls `git -C OVERLAY apply PATCH`; inside a parent Git
repository this returned success but **skipped all four patch paths**. A verbose
check reproduced `Skipped patch` for GemminiConfigs, LoadController, LoopMatmul,
and StoreController. The resulting unpatched `inputType.getWidth / 8` caused
division by zero for A4. This was a build-location problem, not proof of missing
Gemmini A4 support. The failed logs are under `build-potal-hp1-diagnostic/`.

The corrected numerical artifact build uses `/tmp/potal-nano-im2p-artifacts`,
outside any Git repository. Patched `dma_maxbytes * 8 / inputType.getWidth` was
inspected in the actual generated overlay. A4/W4/DIM32 RTL elaboration succeeded.
At that stage, manifest, native-link, and raw numerical execution checks were
still required; the later compatible provider-path run above completes them.

That build subsequently failed in the C++ frontend because current IM2P checkout
`5da6a4c` still consumes retired native `q8_h1`/`q4_h1` descriptor fields. Develop
retired those contracts in `8a38141`; its commit message explicitly names the
matching IM2P commit **`20b82f058aa8b0e69c48d1d00790d8e19eaa6f66`**. That commit
exists locally and descends from the installed IM2P source through its merge.
The compatible checkout is isolated at `/tmp/potal-nano-im2p-compatible`, leaving
the original IM2P checkout untouched. Its numerical artifacts are built at
`/tmp/potal-nano-hp1-matched`, again outside all Git roots. The failed intermediate
compiler log is `/tmp/potal-nano-im2p-build.log` and the corrected build log is
`/tmp/potal-nano-hp1-matched-build.log`. Do not add retired types or dummy descriptor
padding to make an incompatible frontend compile.

The compatible build has now completed: release compilation finished in
28m59s and selected `GEMMINI_HP1/a4-w4-d32/current` contains both
`libim2p_gemmini_frontend.a`, `libim2p_sim.a` and `real-lib.json`.
This was build evidence only. The subsequent manifest/native-link admission and
raw-provider probe described above now supply numerical and RTL-cycle evidence.

The native cycle predictor is separately built in `/tmp/potal-nano-cycle-model`.
Its CTest passed 16/16. `results-nano/cycle-scaled-smoke.json` binds the built
library hash and resolved geometry `(4,4096,512)` to a 2x3x32 planner-block request;
it returned 331 logical cycles for the predictor's HP1 scaled-work contract. This
is a diagnostic estimate, **not raw-op0 timing parity or whole-model latency**.

The numerical runtime (`sim/backends/gemmini_hp1/runtime.cpp`) admits op0/BYPASS
only with block_size=1 and K<=32. Its elapsed-cycle authority is integrated RTL.
The separate `sim/cycle/` predictor consumes value-free geometry and explicit
backing-memory timing. `CYCLE_SIM=1` in llama selects CPU-functional numerical
execution and trace collection; it does not link or invoke that predictor online.
Existing scaled RMD/SCU certificates cannot certify a new raw-op0/host-exact merge
path. No frequency or application latency is inferred from these cycle counts.

`im2p_raw_probe.cpp` is a real-runtime gate: it verifies implementation, ABI5,
A4/W4/DIM32/ACC512, runs K1/K31/K32 with row/column tails against an independent
integer dot oracle, and requires reported RTL cycles. It offers no CPU fallback.
Configure the probe project with the verified `IM2P_SELECTED_GENERATION` and
`IM2P_SIM_ROOT` to build it. Passing this gate does not imply product QK/PV dispatch.

## Unit-add candidate: native Llama numerical validation complete

The user's candidate 3 is evaluated without dropping or clipping digits. Upper
lanes >=2 have three CPU implementations: dense multiplication, sparse
multiplication, and the same sparse metadata with +/-1 replaced by add/subtract.
Other coefficients are multiplied exactly. Keeping the sparse-multiply control
separates zero skipping from the effect of the unit specialization itself.

Reused owners: `native_model.cpp` retains llama.cpp's model/tokenizer/RoPE/GQA/
mask/softmax/linear execution and exposes the existing evaluation callback;
`profile_ops.py` supplies the original Q/K/P/V policy; `profile_gemm.gemm` accepts
a dot callable; `native_measure.py attention` drives paired evaluations and
component timings. This is an experimental output-replacement seam, not product
attention dispatch. The original floating QK/PV dots still execute, so replay
wall time is diagnostic and must not be called integrated backend speedup.

The input is a public FP16 conversion of Llama-3.2-1B, repository
`am00r/Llama-3.2-1B-GGUF`, revision `8e3690a604735388daca0ca1d51f6c92e932e698`.
Its 2,479,591,584-byte file matches the Hub's SHA256
`92cce0cbec5e029be0742a1dfff93254da3393ff4e47f5c07c6539de2dc445d8`.
The current-source quantizer produced Q4_HP1 while preserving F16 embedding and
output tensors; provenance is under `models/llama3.2-1B/` and the quantizer log
is `/tmp/potal-llama-quantize.log`.

GPT-2 and Llama context32 smoke runs pass 36 independent dot comparisons and
bitwise packet/scale/attention/logits/token-NLL comparisons. Llama's diagnostic
one-chunk PPL is 43.818175515399254 for both modes; this short-context smoke is
not the requested 10-chunk result. Its nonzero Q lane-2 unit fraction is 92.3894%,
and P lane-2 digits are all +1. These are context32 observations only.

The existing CPU tile OpenMP option reduced Llama smoke replay time from about
132 to 21 seconds, with attention/logit/NLL hashes unchanged. This is a test
runtime configuration, not a gain attributed to the unit-add candidate. The
serial build and prior outlier measurements are preserved. Context512 x10
paired evaluation completed in `results-unit-add-llama512/`; runtime libraries,
configuration and source hashes are recorded in `runtime-inputs.json`.
Both dense and sparse-unit-add PPL are **21.486062983190273**, with 5,120 input
tokens and 2,550 scored tokens. Packets/scales/attention/logits and every token
NLL are bitwise identical. Each mode evaluates 5,120 QK and 5,120 PV heads.
This proves the incremental arithmetic replacement in this native callback
experiment, not original-spec native linear or complete Gemmini integration.
The executed runner is archived as `native_measure.executed.py`; its SHA256
matches the pre-run input record. A later shared-helper extraction also passed
the prior context32 smoke's packet/attention/logit/token-NLL hashes exactly.

On the actual Llama context512 layer0/head0 P/V capture, dropping fully future
DIM32 main products while retaining the all-token zero-point correction gives
max absolute output error **2.91015625**. Restricting the weighted correction to
the exact retained key domain restores bitwise equality (error 0). The retained
domain extends through the end of the query tile, so masked entries inside the
diagonal tile still cancel. This is an analytical component check, not adopted
PV dispatch; see `results-unit-add-llama512/mask-domain-check.json`.

The same check now passes bitwise on all 15 selected P/V heads (layers
0/5/8/11/15, heads 0/16/31). The wrong all-token correction gives maximum
absolute errors from 2.28515625 to 29.828125 across these heads; the matching
retained-domain correction gives zero on each. The unmodified stripe scales,
K/V block exponents and residual digits are used in both calculations.
See `results-unit-add-llama512/mask-sample-check.json`.

For clarity, let H[i,j] mark the key positions whose GEMM tiles are retained,
D_s be the diagonal matrix of original P stripe scales, and Vhat the original
weighted reconstruction of V. With M=lo-8 and R=q-lo, the identity is
`O = D_s ((H*M + H*R) Vhat + 8 H Vhat)`, where `*` is elementwise.
H may include masked positions inside diagonal tiles: M=-8 and R=0 there,
so the compensation must include those same positions. Retaining `8 J Vhat`
instead (J all ones) adds the erroneous term `8 D_s (J-H) Vhat`.

Aligned equal-size causal prefill gives block-lower-triangular P. Strictly
upper blocks contain only masked entries, lower blocks are fully valid, and
diagonal blocks contain both. Each retained QK entry is an independent channel
dot; future entries affect neither softmax numerator nor denominator. For
prefix/decode, sequence identity and absolute query/key positions define the
boundary. In `src/llama-graph.cpp`, the ordinary QK node does not receive the
mask that is supplied to softmax; the existing flash-attention route already
receives it. A future skip implementation must reuse this actual mask meaning,
preserve original query IDs through residual row compaction, and keep original
quantization stripe groups. No skip implementation exists in this snapshot.
The current CPU softmax (`ggml-cpu/ops.cpp`) copies and scales the entire score
row before adding the mask. Merely omitting future QK stores can therefore read
uninitialized values: NaN plus -inf stays NaN. A separate-node implementation
must write safe finite placeholders for skipped scores, or reuse a fused path
that never reads those entries. This is another integration cost beyond fewer
GEMM submissions.

Thirty chunk-0 QK/PV samples also separate work reductions. Among Q upper
lanes >=2 nonzeros, 90.6364% are +/-1. Relative to all unpadded QK scalar dot
products, zero skipping removes 29.7893% and unit specialization replaces
another 3.2738% with adds/subtracts. For PV those shares are 4.1712% and
0.0734%. These are contribution counts, not latency estimates. Future-tile
skipping overlaps some zero elimination; the two savings must not be added.
See `candidate-work-counts.json` and `mask-sample-check.json` in the same result
directory. Full 10-chunk histograms are now in `report.json`: Q lane-2 has
18,819,606 nonzero digits, of which 92.2348853% are +/-1; Q lane-3 has 993 nonzero
digits, all +/-1. P lane-2 has 1,023,054 nonzero digits, all +1. These counts
describe this checkpoint/input/policy, not a guarantee for arbitrary models.

## CUDA component comparison and CPU candidate decision

The user's subsequent CUDA request is tested in the existing `native_measure.py`
owner. `CpuDot` and `prepare_head` are shared with the native accuracy replay;
`CudaDot` reuses cuBLAS `cublasGemmEx/CUBLAS_COMPUTE_32I` and supplies int64 host
partials to the unchanged checked SCU merge. Signed nibble values are stored in
int8 operands. This is not a CUDA PoTal model port or a new quantization policy.
The GGML CUDA tree has no current Q4_HP1/Q8_HP1/EXSIA implementation to substitute
for the existing native linear path automatically.

Hardware: Orin (nvgpu), SM8.7, driver 595.78; CUDA runtime/driver APIs 13020,
cuBLAS version 130401. Shapes use DIM32 zero padding and column-major TN layout.
All 84 independent CPU/CUDA raw-dot comparisons passed, including nibble values,
zeros/units, tails and cached padding reuse. All 30 saved Llama QK/PV heads match
the independent integer GEMM in float64 and the published float32 output exactly.

CPU/CUDA measurements ran together after the full numerical replay and earlier
CUDA functional check had exited. Each saved head/mode has one warmup and five
measurements with rotated mode order. The table sums per-head medians; common
quantization preparation is excluded, while conversion/padding/copies/device
synchronization/original host SCU and PV zero point are included.

| Mode | 15 QK heads | 15 PV heads | Combined |
| --- | ---: | ---: | ---: |
| CPU dense | 830.984 ms | 339.361 ms | 1170.345 ms |
| CPU sparse multiply, upper lanes >=2 | 735.405 ms | 336.155 ms | 1071.560 ms |
| CPU sparse unit add, upper lanes >=2 | 733.686 ms | 333.380 ms | 1067.066 ms |
| CUDA dense per K32 dot | 911.831 ms | 607.669 ms | 1519.500 ms |

Zero skipping plus unit handling reduces the combined component time by 8.82%,
but specializing +/-1 adds only 0.42% over the sparse-multiply control. Median
within-capture repeat ranges are 2.23% and 1.31% for those CPU modes. This does
not establish an additional performance benefit from unit specialization alone.
The existing fixed-shape Gemmini raw kernel also takes 492 cycles for ordinary
and unit digits alike. Keep these as measured candidates, not adopted product
defaults or a claimed integrated speedup.

Per-dot CUDA is **29.83% slower** than dense CPU here (QK +9.73%, PV +79.06%).
For example, layer0/head0 needs 96 GPU calls and 14,352,384 copied bytes for QK,
and 108 calls and 3,594,240 copied bytes for PV. Retained operand/output buffers
are 256,000 bytes each on host and device across tested shapes, excluding CUDA/
cuBLAS internal memory and common host outputs. Copies and synchronization occur
for every small dot, and SCU remains on the CPU. GPU-resident batched products
and combined SCU/quantization are plausible follow-ups, not measured gains.

In the complete dense diagnostic replay, tracked attention preparation/execution
accounts for 24.87% of wall time; raw dot kernel work is 5.31%, and raw-dot calls
including conversion/wrapping are 9.28%. The remainder includes native work,
the original floating attention still executed by the callback seam, and other
untracked overhead. Do not label the whole remainder as measured linear time or
infer production model speedup from these diagnostic fractions.

Use `cuda-component-timing.json` and `component-summary.json` for performance.
The original `report.json` CPU component timings overlapped the first CUDA
functional run and are superseded for timing comparisons; its completed
10-chunk numerical/PPL evidence remains valid. No full-model CUDA PPL or
integrated Gemmini unit-add/causal-skip implementation is claimed.

## QK fully masked tile skip investigation — implementation pending

This is the final investigation state before the 2026-10-12 transfer. The user
selected **QK-only** omission of physical output tiles for which every actual
mask entry is negative infinity. No implementation, new PPL pair or skip timing
has been run. Q/K quantization, original stripe partitions, all required digits,
valid inner products and FP softmax must retain their existing values. PV remains
unchanged; this is not a FlashAttention-2 integration.

1. **Use the actual mask and its graph lifetime.** In `src/llama-graph.cpp`,
   `llm_graph_input_attn_kv_unified::set_input` tests sequence membership and absolute
   query/key positions (around lines 439–447), with additional SWA and padding
   rules. The ordinary QK `ggml_mul_mat` at line 1248 has no mask input; the
   mask reaches `ggml_soft_max_ext` at line 1275. The layer's selected mask must
   become an explicit, invocation-correct dependency before the backend can
   omit work. A cached pointer or an assumed `key_index > query_index` rule is
   insufficient for reused graphs, prefixes, cache layouts or multiple sequences.
   Finite negative bias (including ALiBi) is not an all-masked entry.
2. **Keep original row identities after RMD packing.** Existing
   `ggml/src/ggml-gemmini/residual/rmd/rmd-types.hpp` owns `row_ids` and
   `row_begin`. Map packed rows back through their original query rows before
   looking up the mask. For example, original query positions `[0,100]` cannot
   be treated as compressed positions `[0,1]`: keys 32–63 are valid for query
   100. QK output columns are key tokens; residual `C_b` columns are reduction
   channels, so the same index does not describe both axes.
3. **Quantize before choosing execution tiles.** `profile_ops.py::stripe_rows`
   depends on the original GEMM dimensions, including N. Cropping a future tile
   before quantization can change stripe boundaries and E2. Preserve whole
   original packets/scales and apply the mask decision to execution only. Main,
   all Q residual lanes, and both K nibble passes need the same valid-output
   domain, with the original-row mapping applied to each submitted packet.
4. **Publish safe skipped scores.** The current CPU softmax in
   `ggml/src/ggml-cpu/ops.cpp` copies/scales the score row before adding the mask.
   Skipping a store into uninitialized memory can introduce NaNs even when the
   mask is -inf. A separate QK node must publish finite placeholders such as
   zero for omitted entries. Fully masked rows are a separate existing boundary
   (`assert(sum > 0.0)` around line 4763), not something this optimization may
   silently redefine.
5. **Reuse the existing graph and dispatch owners.** The generic Gemmini matmul
   gate in `ggml/src/ggml-gemmini/ops.cpp` around line 557 restricts stationary
   head/batch dimensions and layout. It does not already admit arbitrary Llama
   attention heads/views. Product a4nks role dispatch is still absent. Do not
   merely relax the generic weight gate or build a duplicate executor and claim
   attention integration.
6. **Adjust the future validation gate, not the policy.** The current callback
   runner hashes all QK entries. A correct skip may replace only masked entries
   with zeros, so its future gate must compare full operand packets/scales and
   every valid QK entry, then the entire P, PV, logits and token NLL. A matching
   10-chunk PPL pair remains required. The completed unit-add pair above does
   not validate tile skipping. `native_model.cpp` currently clears KV and starts
   at position zero for each single-sequence chunk; prefix/decode, multiple
   sequences, GQA, tails, changed masks and graph reuse still need coverage.
7. **Keep work counts distinct from latency.** Ordinary aligned T512/DIM32
   causal prefill retains 136 of 256 square main tiles: 46.875% fewer main tiles.
   Residual row packing requires its own submitted-work count. Mask inspection,
   gathers, calls and score initialization have costs; no skip latency has
   been measured. Single-token decode over valid cached keys does not have this
   half-matrix saving. PV skipping also needs a matching weighted zero-point
   correction domain; the existing component checks establish that algebra,
   not an adopted PV backend. Online softmax would additionally need to preserve
   C4's normalized-P stripe maximum before it could replace the present path.

## Transfer checkpoint: completed and remaining

**Completed and preserved:** adopted A4 source-zero population fix and its
full/tail/reference regression cases; native GPT-2 before/after 10-chunk result;
Linux RSS unit correction; partial independent scalar oracle; compatible real
IM2P build/admission and 15 raw-dot checks; separate cycle-predictor diagnostics;
Llama native-callback unit-add 10-chunk bitwise comparison; CPU/CUDA component
comparison; specification/reuse audit and the QK investigation above. Evidence
and its limitations are in the preceding sections and result directories.

**Remaining:** complete A–D primitive/linear/masked-attention oracle coverage;
native original-geometry selective low-digit emission without the dense scalar
residual bridge; actual Q4_HP1 scale decoding/full-GEMM parity; checked/exact-wide
merge and final rounding; product Q/K/P/V dispatch with views/cache/GQA/decode;
the selected QK-only skip implementation and paired correctness/work/timing
measurement; raw-op0 predictor validation and full-path Gemmini measurement.
Unit-add, block-local packing, PV skipping and GPU-resident fusion are not
adopted product defaults. See `IMPLEMENTATION_PLAN.md` for the existing owners
and per-task gates. No complete product task is newly marked finished here.

The transfer itself only updates records, inventories existing files and
preserves them. Historical result JSON and logs have not been overwritten with
new measurements. Small results/validation logs accompany the source commit;
captures, token dumps and large logs are delivered in the separate archive.
The archive also preserves older local cycle logs under an explicitly historical
category; they are not evidence for this PoTal implementation.

[`results-nano/transfer-20261012/provenance.json`](results-nano/transfer-20261012/provenance.json)
records current file/hash checks, dependency commits, resolved IM2P generation,
artifact hashes and CMake settings. Historical commands are in `commands.json`;
per-file sizes, hashes, categories and Git/archive selection are in
`inventory.json`. Hashes first collected during transfer are labelled as such.
The native outlier PPL pair did not save a separate token-ID hash/dump, and the
older NumPy baseline did not bind a complete interpreter/library closure; these
gaps remain explicit. Model weights, caches and compiled executables/libraries
are excluded. Configuration/manifest snapshots are evidence, not bundled builds.

The transfer archive is
`/home/dyndnn/Projects/llama.cpp-gemmini/output/potal-nano-transfer-20261012.tar.gz`
(278,803,401 bytes; 409 members). SHA256:
`458d8ec4eec37419d9615f3be02de8657fe30fa6555f7e5e3e2c305c608e3cc4`.
Every archived member's size and SHA256 was checked after compression. Details
are in [`archive.json`](results-nano/transfer-20261012/archive.json); its adjacent
`.sha256` sidecar is outside Git alongside the archive.

| Preserved archive category | Files | Uncompressed bytes |
| --- | ---: | ---: |
| Current PoTal measurements, including captures/token dumps | 207 | 92,225,254 |
| Older local measurements, not revalidated | 139 | 2,928,587,549 |
| Historical build/validation logs | 24 | 5,666,852 |
| Historical analysis scripts/results | 3 | 16,606 |
| Failed-build diagnostic metadata | 7 | 47,623 |
| Build configuration metadata | 11 | 202,444 |
| WikiText input text | 1 | 1,290,590 |
| Transfer metadata and copied validation logs | 16 | 287,088 |

The additional archive-root `inventory.json` describes these 408 payload files.
`workspace/` contains repository-relative paths; `supplemental/` contains
historical external evidence. Extract into a temporary directory first, then
restore only the needed `workspace/` paths into the local checkout. The bundle
includes the small measurement results as well as the Git-excluded data.
Models, model caches, compiled `build-*` outputs, IM2P libraries/caches,
`.parity` binaries/GGUF fixtures, generated mock build commands, and raw Codex
session/account files are neither committed nor included in this archive.
Original local files remain in place.

Transfer checks: existing result JSON/JSONL parsed successfully; historical
hash expectations for the checked models, inputs, binaries and selected IM2P
artifacts matched; the source/document diff passed whitespace validation. No
build, test suite, PPL replay or performance experiment was started for transfer.
The experimental `.gitattributes` preserves this snapshot's raw measurement
bytes, including CSV line endings and whitespace in logs/diff evidence. Source
and handoff documents retain the normal whitespace checks.
