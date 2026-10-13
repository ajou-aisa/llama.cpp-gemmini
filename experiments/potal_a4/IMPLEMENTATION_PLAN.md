# PoTal A4NKS: nano / IM2P implementation plan

## Current implementation checkpoint (2026-10-13)

The opt-in native QK/PV backend is now implemented in `matmul/attention.cpp`.
Graph construction marks and explicitly assigns both nodes to Gemmini; generic
shared-weight admission alone did not prevent CPU scheduling. Q/P quantization
reuses LocalStage/WideQuantizer, upper packets reuse RmdBitmapBuilder, and raw
IM2P dots reuse execute_im2p_compact_dot. Existing projection main/residual flow
is retained. QK mask skipping remains in the separate experiment, not this
vanilla backend. Checked int64 accumulation rejects overflow atomically;
arbitrary precision remains unfinished. This supersedes the older plan's blanket
statement that no native attention dispatch exists, not its unfinished hardware
validation. See [run commands](README.md#실제-backend-qkpv-측정) and
[current evidence](NANO_PROGRESS.md) before implementing another owner.

Portable continuation of the existing work plan, with the latest accuracy-preserving policy. Read [the handoff](../../POTAL_NANO_HANDOFF.md) first. Historical experiment snapshot: `hotfix/potal-attn` at `ba762b951ec267aa3abed6bab9e3466431416307`; continuation branch: `work/potal-attn-im2p-nano`. The unchecked product tasks below are still unfinished. [Original algorithm](ORIGINAL_SPEC.md) and [measured attention costs](ATTENTION_PROFILE.md) travel with this branch.

## TL;DR (For humans)

**Who this is for and what changes for them:** PoTal 실험을 수행하는 연구자와 Gemmini 구현자가 같은 수치 기준으로 선형 연산과 attention을 실행하고 결과를 비교할 수 있게 한다.

**What you'll get:** 원문 명세의 양자화 결과를 유지하며 값을 직접 limb로 분해하는 A4 경로, 중간 residual 배열을 없앤 처리 과정, a4nks attention, 준비 시간·메모리·실제 GEMM 수행량 비교, CPU/Gemmini 및 모델 검증 결과.

**Why this approach:** 정수를 한 번 분해해 첫 limb는 main으로, 나머지는 기존 compact packet에 바로 담는다. 별도의 residual 계산·저장·재분해와 전체 int64 확장을 피한다. 넓은 합산은 기존 계획대로 raw dot 결과를 host에서 정확히 누적한다.

**What it will NOT do:** 제품 Metal/CUDA 이식, KV-cache 저장 형식 변경, FPGA/RTL 재설계, 성능 수치의 추정 발표는 포함하지 않는다. 사용자의 후속 CUDA 실험 요청은 기존 측정 도구의 cuBLAS 정수곱 비교로 별도 수행했으며 제품 backend 이식은 아니다.

**Effort:** XL
**Risk:** High - 저장 표현을 바꾸면서 값, scale, 누적 순서가 달라지면 정확도 조건을 어길 수 있다. Packet 호환성, 넓은 누적, head/cache 배치를 함께 검증해야 한다.
**Decisions to sanity-check:** 사용자의 최신 조건은 원문 명세 대비 정확도 저하 금지다. 선형의 selective 정책, Q의 전체 보정, 원래 stripe의 P 8비트 및 K/V 규칙을 유지한다. 이전 direct-all 선형 선택은 원문 값 보존 조건으로 대체한다. Limb 제한과 P 8행 세분화는 채택하지 않은 제약 실험이며 제품 기본값이 아니다. 준비 과정 절감과 GEMM 횟수 변화는 따로 측정한다. 새 attention은 기본 OFF이며 기존 FP cache·softmax·Flash/CPU 우선순위를 따른다.

현재 상태: 사용자가 채택한 **source-zero 제외를 A4·A8·A16 공통 ExSIA 정책으로 적용**했다. 기존 full/tail/reference와 테스트를 수정했으며 지원 폭별 회귀 검사, A4/A8 전후 10-chunk PPL 측정을 완료했다. 최신 결과와 과거 nano A4 한정 비교는 [NANO_PROGRESS.md](NANO_PROGRESS.md)에 구분해 기록했다. 호환 IM2P A4/W4/DIM32 archive의 manifest·링크와 실제 raw 정수곱 15건도 통과했다. 제품 전체 동등성은 미검증이다. 상위 lane의 ±1 덧셈 후보는 기존 native 모델 callback으로 Llama context512 ×10에서 전후 PPL 21.486062983190273 및 모든 packet/logits/NLL의 비트 동일성을 확인했다. 단순 unit 처리의 추가 성능 근거는 부족하다. CUDA 개별 정수곱 비교는 실제 attention 30개에서 정확히 같았으나 CPU보다 29.83% 느렸다. 아래 10개 항목은 일부 준비 작업이 끝난 상태이며 전체 완료로 표시하지 않는다.

---

> TL;DR (machine): XL; preserve the supplied selective-linear and all-corrected-Q values exactly; directly emit balanced limbs without a dense A4 residual plane; preserve original P/K/V policies; require CPU/Gemmini parity and no accuracy regression against the same reference policy.

## Latest user decisions and evidence boundaries

- 2026-10-13 follow-up: QK/PV must execute through the model's Gemmini backend,
  not only standalone capture replay. The opt-in native implementation and
  explicit graph assignment are present. Model runs verify both node counts per
  layer against the independent reference. Local CPU execution does not establish
  IM2P runtime compatibility or NPU speed; the Nano link/run is still required.
- 2026-10-13: Prepare local code for remote Gemmini QK/PV checks, vanilla without
  masking by default. Reuse the raw provider and existing batch loop; keep direct
  accumulation/QK skipping behind `--experiments`. Quantization, int8 packing and
  oracles precede execution. The new replay passes local CPU parity checks but
  awaits a matching real-runtime link/run on Nano. This does not complete product
  stacked RMD/attention dispatch or establish a hardware speedup. See the
  [vanilla-first runbook](README.md#gemmini-qkpv-원격-검사-vanilla-기본값).
- 2026-10-13: Exclude the upper-lane zero/±1 specialization from adoption.
  Preserve the existing activation main/residual generation, stacking, PE
  execution and accumulation. The next optimization scope is **NPU operand
  reuse and fully masked QK tile omission** at the attention caller. Inspect
  existing reuse before adding anything; retain all signed-nibble int8 values,
  original stripe geometry and block/lane scales. PV skipping still requires a
  matching zero-point correction domain. Judge gains by actual NPU submissions,
  transfers and cycles, not CPU scalar timing. The closed experiment is preserved
  in [comparison.json](results-upper-comparison/comparison.json).
- Local native batching/direct accumulation is now tested in the existing CPU
  replay owners. It preserves block/column shifts and signed-nibble int8 raw
  operands, with private checked int64 accumulation. Llama context512 x10 and
  five real mask/cache scenarios are exact; CTest passes 7/7. QK time improves
  17.94% versus unbatched skip, but PV batching regresses 9.72%. Keep QK as the
  implementation candidate and leave PV batching opt-in for the experiment.
  This informs tasks 5/7/9, but does not complete their arbitrary-wide arithmetic,
  real PE submission or product dispatch requirements. See the latest section
  of [NANO_PROGRESS.md](NANO_PROGRESS.md) and its recorded measurement scope.
- Local QK tile skipping is implemented in the existing graph/replay/GEMM owners.
  Actual masks and original residual row IDs select fully masked DIM32 tiles;
  quantization and PV compensation are retained. Llama context512 x10 gives
  exact valid-QK/softmax/logits/token-NLL parity, PPL 13.801307254545433 both ways.
  CPU QK component time increases 10.17% despite lower dot work, so the candidate
  stays opt-in. This is CPU replay evidence; product dispatch and IM2P timing tasks remain
  unchecked. See the local section of [NANO_PROGRESS.md](NANO_PROGRESS.md).
- Source-zero exclusion is adopted for all supported ExSIA widths (A4/A8/A16): sigma population is `~top && source!=0`.
  Keep source-nonzero rounded-to-zero entries. The old zero-inclusive population
  was a policy difference from the supplied contract, not a generally invalid
  statistical method. The 10-chunk result does not revoke the user's decision.
- Proceed in isolated increments, compare their benefits/costs and preserve the
  decision record. Do necessary implementation and verification without repeated
  permission questions. After a decision, remove code made obsolete by that
  decision while retaining paths still used by A8 or other supported profiles.
- Inspect the existing implementation and callers before each edit. Modify the
  owner of the behavior instead of adding a second implementation. The reuse map
  below is a required starting point, not proof that a similarly named helper
  already satisfies the supplied arithmetic.
- New before/after PPL comparisons use **10 chunks**, currently context 512,
  batch/ubatch 512, threads/batch-threads 4, the same GPT-2 Q4_HP1 checkpoint and
  WikiText-2 test prefix: 5,120 input tokens and 2,550 scored tokens. Record actual
  token IDs/hash, scoring positions, arguments, source/diff and linked artifact
  identities. Historical 8-chunk results remain historical; they are not a
  matching 10-chunk baseline. Any required input/profile change needs a new pair.
- Zero-only product comparison: `40.18882947104453 -> 40.20531899910632`
  (+0.0410301%), ExSIA tests 5/5. See [results-outlier-zero](results-outlier-zero/).
  Both used CPU main/RMD, FP attention and generated ACC_ROWS=1024. This isolates
  the zero-policy change, but does not certify original `(4,4096,512)` geometry,
  full a4nks, direct limb emission, simulator execution or hardware speed.
- Exact before/after equality is the gate for a **representation optimization
  of the same quantized policy**. Aligning an old product policy with A-D may
  change its legacy PPL; evaluate that change separately and then establish the
  aligned reference baseline. Do not confuse this accepted zero-policy change
  with permission to drop limbs or alter selective-linear/P/K/V rules.
- The nano snapshot is `work/potal-attn-im2p-nano` at `63e166d`; local QK work
  continues on `work/potal-qk-mask-local` based on that verified commit.
  The 2026-10-12 transfer request
  explicitly authorizes committing/pushing the existing work on that test
  branch, with no develop/hotfix merge or new implementation/experiment.
- The next selected scope is fully masked **QK-only** tile skipping. Its mask,
  original-row mapping, output initialization and validation requirements are
  recorded in [NANO_PROGRESS.md](NANO_PROGRESS.md#qk-fully-masked-tile-skip-investigation--implementation-pending).
  It is not implemented or measured in this snapshot; PV skipping is deferred.

## Specification-to-work audit

The user re-supplied both pseudocode and Python. Their A-D arithmetic is the
authority; known helper boundary limitations are not algorithms to preserve.
Packet v8, a particular accumulator representation, default-OFF graph selectors
and role-hint slots are implementation choices. Tail/view/GQA compatibility and
explicit output rounding are integration requirements in this plan. None of
those labels is evidence that product implementation already exists.

| Requirement from the supplied spec/request | Observable acceptance | Tasks and current evidence |
| --- | --- | --- |
| Constants and primitives | A4/W4 signed operands, BK=DIM32, tau2, NEG, half-even qround and half-away folding with int32 saturation | 1,4; primitive oracle exists, product rounding/large-right-shift work remains |
| Logical stripe geometry | Compare the complete `stripe_rows(I,N,K,32,4)` growth rule at boundary shapes; `(4,4096,512)` and original E2 partitions | 1,3,4,8,9; CPU comparison currently uses ACC1024, so geometry gate is open |
| A.1 exponent/top stage | Highest and second distinct exponents; top only with e2; preliminary q from e_pre | 1,4; existing product stage needs full oracle comparison |
| A.1 sigma population | Exclude source zeros/top/padding, retain source-nonzero q=0; strict one-sided 2-sigma | 4; shared A4/A8/A16 full/tail/reference policy and fixture pass; arithmetic already uses `__int128`; local evidence in `results-source-zero-common` |
| A.1 final adaptation | Recompute E from non-top/non-sigma U if any sigma; otherwise e_pre; final q at E-rho | 1,4; explicit intermediate E/q comparison remains |
| A.2 folding and A.3 selection | Distinct non-NEG E2; record D/u/out/corr separately; uncorrected +8 -> 7, corrected +8 preserved | 1,4; full native original-policy gate remains |
| A residual layout | Lane0 zero in upper packets, every required lane retained, per-lane active rows and per-original-block union columns; DIM padding | 2,4,7; current product still clips main and captures scalar residuals |
| B linear stationary W | Preserve actual signed codes, block/output-column exponents, all GGUF scale factors and source weights | 1,5,8; Q4_HP1 channel_scale mapping must be tested, not omitted |
| B SCU/merge | Shift each original block and lane contribution; exact sum before final float publication | 5,8; native exact wide path remains |
| C1 Q | rho6, correct_all=true on 4-bit PEs, all needed upper digits | 4,9; Python policy exists, product Q route remains |
| C2 K | Token/channel-block exponent, [-128,119], exact low+16*high with both digits in [-8,7] | 6; code-level primitive checks exist, native operand packing remains |
| C3 QK | Both dense K passes and all Q lanes, per-key/per-block scales; newly computed S feeds existing scale/mask/FP softmax | 5,9; product complete-graph gate remains |
| C4 P | Original PV stripe geometry, stripe max, uint8 half-even, main `(q&15)-8`, upper `q-(q&15)`, lane2 iff q>=128 | 1,6; primitive/Python checks exist, native P path remains |
| C5 V | Per-channel token-block 4-bit codes/exponents and weighted colsum once per head/invocation | 6,9; native packing/reuse remains |
| C6 PV | Merge main, both upper lanes and weighted offset before publication; exact masked/all-zero cancellation | 5,9; native execution remains |
| D work counts | Main/upper DIM-padded fragments; global lane-row packing, per-block C_b union; charge K-high main+Q-upper to residual | 5,7; reference counts exist, actual native-submission reconciliation remains |
| D add counts | Weighted colsum and broadcast counts (2*T*d for equal lengths); host-wide work counted separately | 6,7,9; reuse/execution gate remains |
| Supplied verification | Exact reconstruction/counters, GEMM comparison, real weight fixtures, full recomputed-softmax attention | 1,10; supplied close() is relative <1e-12 and captured P bypasses softmax; external references unavailable |
| Gemmini numerical execution | Verify matched artifact/ABI/geometry, link and run raw K1/K31/K32 including tails, then linear/QK/PV fixtures | 5,8,9,10; matched artifact admission and 15 raw-provider dot checks pass; product linear/QK/PV gates remain |
| Timing prediction and measurement | Bind actual submissions to supported Gemmini timing contract; compare with RTL cycles; record host preparation/merge separately | 7,10; raw 32x32x32 is 492 RTL cycles for ordinary and unit digits alike; scaled predictor smoke/16 tests passed, no raw-op0 prediction certificate or integrated unit-add speedup |
| Accuracy and cleanup | Per-increment 10-chunk comparison, exact same-policy optimization gate, adopted-path cleanup and regressions | 10,F2; zero-only comparison complete; upper-unit candidate passes Llama 10-chunk exact replay; native product integration remains |

Before marking any row complete, attach source-bound results from its actual
execution path. Small primitive tests, the Python FP32 forward, the native packet
microbenchmark and the Gemmini RTL runtime prove different parts of this table.

## Existing implementation and reuse audit

The user's latest instruction requires checking what already exists **before**
adding code. The following locations were inspected in the current nano tree;
paths in the first table are relative to `ggml/src/ggml-gemmini/` unless marked
otherwise. Earlier proposed file names are not a requirement to create files.

| Work | Existing owner / entry point | Change there; remaining gap |
| --- | --- | --- |
| A.1 exponent, qround, top/sigma, final E | `quants/act/exsia/local.{hpp,cpp}`: `ExpScanner`, `WideQuantizer`, `SigmaDetector`, `LocalStage` | Extend optimized, tail and reference paths together. Keep the adopted zero exclusion. Do not add another production `StreamQuant` implementation. |
| A.2/A.3 folding and main storage | `quants/act/exsia/folding.{hpp,cpp}`: `StripeFolding::run`; `exsia_shift.hpp`; `exsia.cpp` / `exsia-state.hpp`; `quants/act/buffer.hpp` | Fix shift/rounding and selective low-digit output at these owners; use existing per-slot lifecycle and activation storage. Q needs rho6/all-corrected policy, not another ExSIA pipeline. |
| Balanced digits and compact packets | `residual/rmd/rmd-builder.cpp`: `decompose_balanced_radix`; `rmd-bitmap-builder.{hpp,cpp}`: `RmdBitmapBuilder::emit` / `finish`; `rmd-types.hpp` | Reuse existing int32 digit decomposition and bitmap finalization. The builder also has a fused legacy scalar decomposition loop; accept already decomposed upper digits without invoking it again. Extend the same packet validation/slicing/hash machinery for upper-only semantics. |
| Lane-row and K compaction | `residual/rmd/rmd-run-aware.{hpp,cpp}`: `build_run_aware_geometry` / `build_run_aware_request`; `rmd-executor.cpp` | Reuse original block/lane/row maps and gather machinery. Existing carrier-based execution needs adaptation to raw per-block dots; it is not automatically the exact-wide route. Do not build a parallel compactor. |
| Linear weight decoding | `quants/common/weight_route.{hpp,cpp}`; `weight_reader.{hpp,cpp}`: `read_code_tile_validated`, `read_hp1_carrier_validated`, `read_column_scale_validated` | Use validated native readers for stored codes and all scale factors. No second production GGUF/Q4_HP1 decoder. A Python fixture reader remains an independent test oracle. |
| Raw CPU / IM2P execution and output lifecycle | `matmul/dense.cpp`, `matmul/execution.cpp`; `residual/rmd/rmd-im2p-executor.{hpp,cpp}`: `Im2pCompactDot`, `execute_im2p_compact_dot` | Extend the existing scheduling/staging path and provider with the A4 raw contract. Reuse failure statuses, capability checks and provider ownership; no second simulator wrapper or executor stack. |
| Exact scaled accumulation | `quants/common/hp1_scu.hpp`; `residual/rmd/rmd-compose.cpp`: `compose_rmd_output`; `matmul/dense.cpp` | Existing SCU saturates int32; checked compose uses `__int128` intermediates but requires an int64 result. Neither guarantees arbitrary required exponent spread/cancellation. A private exact accumulator is a real gap; place it beside these owners, extracting a shared helper only when linear/QK/PV callers need it. Preserve the legacy saturation contract. |
| Logical / physical geometry | `ggml-gemmini-geometry.hpp`, `ggml-gemmini-args.h`, existing `gemmini_set_tile_ws` call sites; external `${POTAL_GEMMINI_SW}/gemmini.h` | The tile selector definition lives in the external header, not a missing in-repo tiling file. Reuse geometry metadata and existing configuration plumbing; establish original logical `(4,4096,512)` stripe partitions without duplicating capacity selection in each operand path or changing the sibling repository. |
| Runtime K/P/V and views | `quants/common/tensor_util.hpp`, `fp16_util.{hpp,cpp}`, existing ExSIA primitives/buffer; F16 conversion in `ops.cpp` | Reuse base-address and conversion primitives, then add only missing stride-aware runtime packing and K/P/V policies. Current `activation_data` is F32-oriented and the FP matmul conversion assumes contiguous data. Existing `quants/act/block/block.cpp` uses a different scale/outlier rule and cannot substitute for C2/C4/C5. |
| Attention graph and backend routing | Repository `src/llama-graph.cpp`: `build_attn_mha`; `ops.cpp`: `ggml_backend_gemmini_device_supports_op` and MUL_MAT execution; `ops.hpp` | Extend the existing QK and PV nodes in the shared builder, and classify roles before the shared-weight gate. Reuse the existing scale/mask/FP softmax, cache and head layout; do not create a second attention graph. |
| Configuration and artifact selection | Repository `cmake/ggml-gemmini-{options,config,im2p}.cmake`; `scripts/im2p-host-provision.sh` | Add only the missing default-OFF attention opt-in/role contract using current propagation. Existing activation/RMD selectors do not identify QK/PV roles. Reuse artifact resolution; no second build/profile discovery mechanism. |
| Counts, host time and cycles | `ggml-gemmini-evaluation-observer.hpp`; repository `ggml/src/ggml-gemmini-utils/{include/gemmini/evaluation_metrics.hpp,src/evaluation_metrics.cpp}`; `scripts/utils/{potal_records,cycle_schema}.py`; `scripts/eval/cycle_*.py` | Extend current observers/records with original identities and actual raw submissions. Preserve old schema meaning. Reuse Gemmini cycle plumbing and distinguish unsupported raw prediction from measured RTL cycles; no parallel telemetry system. |

Existing regression owners also take precedence over new executable targets:

| Tasks | Extend existing tests / tools first |
| --- | --- |
| 1 | `tests/potal_a4_reference.py`, `tests/test-potal-a4-reference.py`; existing experiment fixtures/readers. These oracle files now exist: extend them. |
| 2 | `test-gemmini-rmd.cpp`, `test-gemmini-rmd-bitmap.cpp`, `test-gemmini-rmd-packet.cpp`, `test-gemmini-rmd-telemetry.cpp` |
| 3,9 | `test-gemmini-im2p-routing.cpp`, `test-backend-ops.cpp`, existing build/config contract tests; use shared llama graph callbacks for model-level coverage |
| 4 | `test-gemmini-exsia.cpp`, `test-gemmini-act-quant-i32-saturation.cpp`, `test-gemmini-exsia-profile.cpp` |
| 5,8 | `test-gemmini-matmul.cpp`, `test-gemmini-rmd-executor.cpp`, `test-gemmini-rmd-im2p-provider.cpp`, `test-gemmini-srmd-raw-ws.cpp` |
| 6 | Existing ExSIA / quantization and backend-op fixtures; choose the owner after runtime operand integration |
| 7 | `test-gemmini-rmd-run-aware.cpp`, `test-gemmini-evaluation-{observer,metrics}.cpp`, existing cycle tests and `experiments/potal_a4/native_probe.cpp` |
| 10 | `test-gemmini-production-parity.cpp`, `examples/eval-callback/eval-callback.cpp`, `tools/perplexity/perplexity.cpp`, existing model capture scripts |

Before each increment, trace its caller -> owner -> consumer, identify the
existing test and record the failing numerical/dispatch requirement. Prefer an
in-place change or extraction from that owner. A new helper/file/target is
justified only by a specific responsibility or build dependency that the
existing owner cannot cleanly cover; record that reason and its actual callers.
Do not create both the proposed new component and an equivalent old path.
Independent test oracles are intentional validation, not alternate production
implementations. Recheck call sites after integration and remove the replaced
unused code once the increment is adopted, keeping active A8/legacy consumers.
This is an implementation rule, not an additional user-approval step.

The audit establishes reuse locations, not completion of the tasks or runtime
parity. The existing raw probe can admit the matched archive; production provider
tests remain the integration gate, so do not add another standalone raw probe.

## Scope
### Affected user and ideal state
**Affected user:** The PoTal researcher needs reproducible linear/attention results and trustworthy cost counts. The implementer needs exact numerical rules and source locations rather than a new parallel quantization stack.

| Row | Statement | Reason |
| --- | --- | --- |
| IS-1 | Direct limb emission preserves the supplied linear/Q codes, scales and reconstructions; K/P/V retain the supplied a4nks rules. | The latest user constraint forbids accuracy loss and supersedes the earlier all-folded-code linear amendment. |
| IS-2 | Zero, tail, rounding, residual-width and accumulation boundaries are executable contracts. | The supplied Python contains reproduced boundary failures. |
| IS-3 | Supported attention executes both QK and PV on Gemmini with existing FP softmax and masks. | Ordinary CPU fallback is not proof of a4nks execution. |
| IS-4 | Fragment/add reports reconcile with executed work. | The reference counters describe a schedule that its NumPy loops do not implement. |
| IS-5 | All ExSIA widths share the adopted source-zero policy. A8 storage/execution, FP alternatives and unrelated experiments retain their other contracts. | Source-zero population is common to A4/A8/A16; representation changes remain scoped to A4/a4nks. |
| IS-6 | A4 emits upper limbs without a dense residual plane, scalar recapture, or a second decomposition; preparation and GEMM costs are measured separately. | Reducing RMD preparation does not guarantee fewer GEMM fragments. |
| GAP-1 | Source-zero exclusion is implemented/adopted for A4/A8/A16; the product still clips the main digit and uses scalar residual capture. | Zero substep verified; complete tasks 2 and 4. |
| GAP-2 | Legacy scalar residual APIs cannot represent the +2^31 upper-limb sum. | Bypass scalar residual materialization with direct digit packets in task 2; do not widen the shared scalar corridor. |
| GAP-3 | Python caps right shifts, overflows int64 and rejects BK tails. | Close with tasks 1 and 5. |
| GAP-4 | Gemmini shared-weight dispatch rejects attention views/heads. | Close with tasks 3 and 9. |
| GAP-5 | Existing HP1 SCU saturates int32, whereas the requested arithmetic is wide. | Close with tasks 5 and 8; preserve the old SCU route for its original contracts. |
| GAP-6 | Role-specific K/P/V quantization and weighted colsum reuse are absent. | Close with tasks 6 and 9. |
| GAP-7 | Captured P bypasses quantized QK/softmax, and costs are formula-only. | Close with tasks 7, 9 and 10. |
| GAP-8 | Clipping, residual-plane writes and scalar capture precede decomposition; the prior plan would widen that whole corridor. | Replace it with fused direct limb emission and measure the effect in tasks 2,4,7,8,10. |
| GAP-9 | Current CPU PPL builds use ACC_ROWS=1024, while the original logical stripe profile requires 512. | Add a geometry gate and isolated comparison in tasks 1,3,4 before claiming original-policy parity. |
| GAP-10 | Scaled-HP1 cycle-model success does not establish prediction for the raw-op0/host-wide workload. | Bind/correlate supported timing requests in tasks 7 and 10; report unsupported prediction as incomplete. |

### Must have

- User-confirmed target: Gemmini implementation plus CPU-reference verification, continued on nano with IM2P. Product-code base: `develop` at `3b2eaa3cb6f7adc36094b7f78662141825cc4ef4`. Experiment snapshot: `hotfix/potal-attn` at `ba762b951ec267aa3abed6bab9e3466431416307`. Work on `work/potal-attn-im2p-nano`.
- Use the existing continuation branch on nano. Preserve local environments, models, worktrees and unrelated changes. The original handoff predated nano execution; current executed evidence is in NANO_PROGRESS.md.
- Fixed requested profile: PE A4/W4, BK=32, DIM=32, tau=2, distinct-E2 stripe anchor, and stripe geometry `(banks,bank_rows,acc_rows)=(4,4096,512)` from the supplied reference. Assert native stripe partitions match this geometry; a different memory profile is not parity evidence. Q uses rho=6 on four-bit PEs; linear uses rho=2. Do not change the hardware width to implement Q's code precision.
- Latest acceptance constraint: preserve accuracy relative to the supplied selective-linear+a4nks policy. Use algorithm identity `potal-a4-preserve-v1`. Linear first selects the original effective code `v = (out || D>0) ? u : sat(u)`; Q uses `v=u`. Both directly decompose v. This supersedes the earlier `potal-a4-direct-v1` all-folded-code amendment, which remains an unadopted experiment. No limb truncation, extra saturation or smaller P scale groups are part of the accepted implementation.
- Retain existing source weights and decode their stationary codes/exponents. K/V are runtime stationary operands and use the new policies below.
- For Q4_HP1, account for `channel_scale * 2^m` in each stored weight scale. Validate positive power-of-two carriers before mapping to `thetaW=m+ilogb(channel_scale)`; preserve zero-block sentinels. A nonrepresentable/malformed carrier needs an explicit capability result, never silent rounded-log2 conversion or weight requantization.
- Support reduction tails and row/output tails, all-zero blocks, strided F32/F16 KV views, prefill, single-token decode, Tq != Tkv, and GQA. Preserve existing sequence/causal/SWA masks and model attention scale.
- Keep FP KV storage and existing FP softmax. Quantize logical views per invocation; do not create a persistent quantized KV-cache format.
- Reuse balanced digit packets, original lane/source-row IDs, original K-block run identities, geometry selection and observer infrastructure.

### Numerical decisions

1. Finite float32 values define the quantization domain. Convert supported F16 views to F32 before these rules. Reject NaN/Inf in source operands before quantization or destination mutation; do not inherit the reference's NaN-to-INT32_MIN behavior. Final-format overflow is a separate conversion rule below.
2. `eps(0)=NEG`; exclude NEG from distinct anchors; choose anchor 0 for an all-zero stripe. Sigma population uses source `x!=0`, not rounded `q!=0`; padding contributes no population entries. Use strict integer `>` and wide intermediates for the two-sigma test.
   This policy is user-adopted for all supported ExSIA full/tail/reference population paths. Reuse the existing source exponent sentinel to exclude zero; do not add a second population mask or scan. Top, sigma, D>0 correction and nonzero upper support are separate concepts; zero correction candidates need not generate work. Under BK32/rho2-or-6, preliminary magnitudes in U are bounded by 128, so sigma-int64 overflow is not an established defect. Keep the existing native `__int128` arithmetic.
3. qround is half-even independent of ambient process rounding mode; fold right shifts are half-away. For int32 q, right shift 32 returns -1 only for INT32_MIN and zero otherwise; shifts >=33 return zero. Positive shifts saturate before any undefined native shift occurs.
4. For linear folded int32 u, preserve the original correction predicate: `corr=out || D>0`, then `v=corr ? u : sat(u)`. Q always uses `v=u`. Compute balanced radix-16 digits of v once, write d0 to dense main and emit every nonzero d1..d8 directly. Since sat(u) lies in [-8,7], this exactly reproduces the original selective main and zero residual at uncorrected positions. Preserve top/sigma adaptation, E, stripe anchor and scale. Reuse `decompose_balanced_radix(v,4,...)` without materializing a scalar residual or dense residual plane. Never limit required limbs. Keep A8 width-specific quantization and scalar capture, with the shared source-zero population policy from item 2. qround/folding int32 saturation and K/P/V clamps retain their original contracts.
5. Preserve the requested wide sum. Existing op5/SCU-final output has already lost saturation information and must not feed the new exact route. Use A4 raw integer fragments, then scale and merge on the host. Apply column exponent, stationary nibble shift and original residual lane shift after the raw dot. This is physical raw-PE execution plus host-wide merging, not a claim that the existing hardware SCU is wide.
6. Use checked int64 accumulation when a bound proves every shift and addition fits. Otherwise use exact exponent-binned host accumulation, scoped to this dot-product operation: balanced radix-2^32 limbs and checked carry normalization. Apply every block/column exponent, nibble/lane weight and PV zero-point compensation before conversion. The reference GEMM result is the exact scaled sum rounded half-even to F64; publishing a GGML F32 tensor rounds that F64 value half-even to F32. This deliberately specifies exact->F64->F32, not direct exact->F32. Use gradual underflow with subnormals preserved, signed infinity on final-format overflow, the exact value's sign on underflow to zero, and +0 for exact cancellation. Conversion must not depend on ambient rounding or flush-to-zero state. The graph softmax consumes the published F32 S, including the existing graph's scale/mask. Do not add a general bigint dependency or silently fall back to saturated int32/double partial sums. The independent Python oracle uses Python integers. This slow path is required by finite-input overflow/cancellation counterexamples, not a performance feature.
7. K codes are [-128,119], split into signed low/high nibbles with extra shifts 0/4. V uses per-channel token-BK32 A4 blocks. Zero blocks have zero codes and neutral stored exponent.
8. P has one unsigned-eight-bit scale per stripe, main `(q&15)-8`, and lanes 1/2. Compute exponent-weighted `colsumV` once per KV head per invocation, retain its exact integer/exponent representation, reuse it for corresponding GQA query heads, and merge the scaled offset once per output row before either float conversion. Include precisely the same token domain as the dot products so masked P=0 cancels exactly.
9. Preserve the spec fragment model by packing all surviving `(lane,source-row)` pairs together and applying each row's lane weight in host merging. Use the union of active columns per original K block and pad each block's submitted K to DIM; never combine distinct block exponents as though they shared a carrier.
10. Spec add counts describe weighted-colsum and offset-broadcast terms, not all CPU instructions. Record host-wide merging separately; fragment/add ratios are not latency measurements. For one head with unequal lengths, add terms are `Tkv*d + Tq*d`; GQA reuses the KV-head colsum instead of charging it Hq times.

### Direct limb decomposition and RMD reduction

The representation is `folded u -> original-policy effective v -> d0 main + compact upper-limb packet`. Main remains one dense pass; upper limbs retain active-row pruning, original BK32 block boundaries and DIM padding. RMD execution is still needed for those upper limbs. No lossy limb dropping, extra bit-width option, or new approximation threshold is introduced.

| Comparison baseline | Accepted comparison |
| --- | --- |
| Existing clipped main plus fully corrected residual | This is a separate representation microbenchmark, not proof of original-policy accuracy. Removing its plane preserves that baseline's values; keep its space claims separate from model-policy comparisons. |
| Supplied selective lowdigit linear, using identical E/u | Main, upper digits, support, scales and fragment counts must be identical. Uncorrected +8 remains 7 without an upper limb. Scalar residual preparation/storage can be eliminated. |
| Supplied Q with correct_all=true, using identical E/u | Main, upper digits, support and fragment counts are identical. Only preparation/storage are simplified. |

Keep these baselines separate in fixtures and reports. For finite F32 linear inputs under the corrected rho=2 block rules, uncorrected u is in [-8,8]. A reachable example is a 32x32 matrix of F32 1.99: E=0, D=0, no outlier, q=8. The accepted original-policy result remains 1.75 with one main and zero residual fragments at N=32. The experimental direct-all result is 2.0 with one extra upper fragment and must fail original-policy equivalence. Keep this rejection case in the suite.

Use all nine signed-digit positions 0..8. `INT32_MAX` has digits `[-1,0,0,0,0,0,0,-8,1]`; `INT32_MIN` has `[0,0,0,0,0,0,0,-8]`; `0x77777778` needs nine nonzero digits. Never compute `(u-d0)` in int32 or assume eight signed nibble positions suffice. A portable quotient recurrence for an independent checker uses truncating `q=u/16`, `d=u%16`, then adjusts `(d,q)` by `(-16,+1)` if d>7 or `(+16,-1)` if d<-8. The native implementation should reuse the existing helper before considering another decomposition loop.

Packet/storage contract:

- Add only a direct upper-digit entry to `RmdBitmapBuilder`: `reset_upper_a4(...)` without an outlier selection mask, and `emit_upper_a4(row,k,NativeBalancedDigits)` consuming lanes 1..8. Reuse bitmap storage, lane-row maps, K masks and finish/compaction. Skip zero upper limbs; preserve original lane IDs, including gaps.
- Retain v7 as the legacy scalar-residual packet/default. Define v8 as upper-A4 digits with the same digit/block payload layout, bits=4, no lane 0, and `residual_observations_valid=false`. Validate the two meanings explicitly. Do not widen `ResidualEvent`, global residual vectors, residual extrema or legacy capture APIs.
- An A4 stripe slot owns the bitmap builder directly; CPU raw-dot mode and Gemmini use the same digit packet. Preserve capture timing and sealed-packet lifecycle without routing through `TimedResidualCapture::add_residual` or CPU-direct scalar events. `state_.residual` stays empty for this profile; keep its existing use for legacy profiles.
- Slice v8 packets by copying/remapping digit coordinates, never by composing a scalar and calling `add_residual`. Preserve arbitrary row subranges, stripe crossings and each source stripe's exponent association; never merge differing stripe scales as a single carrier. A4 reconstruction combines main plus weighted digits in a transient checked int64 value and verifies final int32 range; it does not allocate a dense wide residual plane. Legacy packet-to-scalar paths reject v8; the scalar-only `compose_balanced_radix` contract stays unchanged.
- Hash v8 as sorted `(global_row,original_k,original_lane,signed_digit)` coordinates under a distinct versioned hash domain. Operand identity additionally includes main codes, stripe scales and algorithm ID. Preserve v7 hashes and old telemetry interpretation. Report upper-digit support/nnz separately; do not relabel folded-u extrema as scalar residual extrema.
- P keeps unsigned low-nibble main and zero point 8. It may feed digits of the bounded scalar `q-(q&15)` into the same upper-digit emitter; it must not substitute balanced digits of q, which would change support and the zero-point contract. K's two passes and V's A4 clipping remain unchanged.

Measure removal of the existing `4*I*K_padded` residual-plane allocation/writes separately from packet/scratch allocations and live peak memory. The prior proposed eight-byte residual plane is avoided, not an already measured baseline. Report decomposition calls, capture/pack time, packet bytes, upper digit nnz, active lane-row pairs, raw submissions, DIM-padded fragments and host-merge time. Establish native CPU/IM2P latency only with actual runs; no percentage improvement is promised by the plan.

### Integration decisions

- Proposed selector: `GGML_GEMMINI_ENABLE_A4NKS_ATTENTION`, default OFF. It requires integer A4/W4, EXSIA, RMD enabled and BK=DIM=32. Enabling it does not mutate Metal/CUDA branches.
- Use explicit MUL_MAT role hints in the currently unused op_params slot 1 through inline accessors in `ggml/include/ggml-gemmini.h`; leave precision slot 0 intact and document the reservation beside MUL_MAT in `ggml.h`. Roles are NONE, QK and PV with fixed recognized marker values. Do not use tensor names or tensor.extra.
- Set both roles only for an eligible complete non-flash attention case. Shared shape/type eligibility must be available before tagging. Eligibility includes ordinary GPT-2/Llama self attention, supported F32/F16 views, valid GQA ratios and compatible output layouts. Unsupported MLA/quantized-KV/layout variants retain the existing FP route with explicit coverage reporting.
- Explicit flash attention and `--no-kv-offload` retain precedence and FP behavior. Do not change the V-cache layout behind the user's flash setting. A4NKS validation runs must disable flash and require both tagged operations to execute on Gemmini.
- Check role hints before the shared-weight contract in capability and execution dispatch. A rejected tagged operation must never enter the linear executor. A generic backend may ignore the hint, but the run must report fallback/mixed coverage and cannot pass the a4nks execution gate.

### Must NOT have (guardrails, anti-slop, scope boundaries)

- No Metal/CUDA port, weight requantization/model-format change, persistent quantized KV cache, new GGML operation, or unrelated graph refactor.
- No edit to sibling IM2P/Chisel/header repositories. Existing raw op0 support is used; extending external CPU-functional raw tracing or RTL is outside this work.
- No reuse of stale DIM16 builds as DIM32 evidence, tensor-name dispatch, Python-generated native answers, synthetic success counters, or enabling hardware SCU saturation in the oracle.
- No replacement of `evaluation/activation/__init__.py`'s existing signed-FP reference definition. New integer-quantizer tests have their own contract.
- No shared residual int64 migration, duplicate A4 scalar residual plane, residual materialization merely for hashing/slicing, or new public selector between direct-all and selective linear. The supplied selective policy is the accepted linear contract; direct-all remains an unadopted experiment.
- No hardware-cycle/FPGA/production-RTL performance claim from host tests or fragment formulas. Existing raw CYCLE_SIM/optrace restrictions remain explicit; unsupported tracing must fail clearly rather than fabricate records.
- The user explicitly requested publishing the experiment snapshot and a separate nano continuation branch. This handoff does not claim to implement the product tasks below; no PR or merge was requested.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD for numerical regressions and contracts; existing C++/CTest harnesses plus a small NumPy/Python-integer reference. Every implementation task includes its tests.
- Planned integration evidence root: `.omo/evidence/potal-a4nks/`. Existing nano evidence remains under `experiments/potal_a4/results-nano*` and `results-outlier-zero`; link it rather than relabeling it as an integration pass. Each new log/JSON records source SHA, dirty-diff digest, build profile, command and exit code; a historical log is not a passing run.
- Historical reference environment: Python 3.11 and NumPy 1.26.4. On nano, locate a compatible existing interpreter and dependencies; use local `gguf-py` and the tokenizer required by `prepare.py`. Set `POTAL_PYTHON`, `POTAL_GEMMINI_SW` and `POTAL_IM2P_ROOT` to verified local absolute paths before using the recipes. Never reuse the Mac build directories or archives.
- Numerical levels: exact integer codes/exponents/masks/lanes/row maps; exact scaled accumulated coefficients including PV compensation; half-even exact->F64->F32 publication; then existing FP softmax. For the same quantized CPU/Gemmini path require identical operands/raw dots and bitwise F32 outputs, including signed zero and overflow fixtures. Cross-library softmax checks use fixed atol=1e-7, rtol=1e-6 and exact masked zeros; tolerance never applies to integer metadata or cost counters.
- Test the reference itself with ties, D=-31/-32/-33/+31/+32, INT32 endpoints, source-nonzero rounded-to-zero, E2 duplicates/NEG, strict sigma equality, K=31/32/33/63/64, P=0..255, K=-128/119, overflow and cancellation. Include `1+2^-24+2^-54`, which rounds to 1.0 through F64->F32 but upward directly to F32; include positive/negative subnormal, underflow-to-zero, exact cancellation and final-format overflow conversion fixtures.
- GPT-2's pasted verification has not been run. `sf_tradeoff.py` and `attn_quant.py` were not found in scoped source searches. Do not require those absent modules for the independent local oracle, and do not claim cross-reference verification until their exact versions/data become available.
- Direct-limb checks: require reconstruction for all int32 endpoints, lane 8 and zero gaps; no v8 lane 0; digit-only slice/hash behavior; zero dense-residual allocation/writes in A4; and no scalar-capture calls or second decomposition. Compare both selective linear and all-corrected Q bitwise to their original-policy references. Any changed effective value, digit, scale or support fails acceptance.
- Accuracy gate: with fixed weights, token IDs, masks and the same numerical execution path, require identical reconstructed operands and downstream logits, per-token NLL and PPL before/after the storage optimization. Use 10 chunks for each new before/after PPL comparison. A similar average PPL, a bootstrap interval containing zero, or a closer individual float reconstruction is not a substitute for equivalence. Native raw dots and publication must separately match the prescribed exact reference; an unexplained rounding/output mismatch blocks acceptance. This baseline is the supplied quantized policy, not lossless equivalence to FP32 weights/activations or to the old zero-inclusive product.
- Historical planning review: Metis returned CLEAR for the earlier direct-all amendment only; it did not approve this accuracy-preserving revision. CPU policy experiments and native representation probes are complete in the task worktree. Product integration and matched simulator/model parity remain unverified.

### Fresh CPU-mode build recipe (run only during execution)

Run from the repository root after implementing the planned configuration and executor. `GGML_GEMMINI_ENABLE_A4NKS_ATTENTION` does not exist at the handoff snapshot. This recipe exercises Gemmini with host CPU raw-dot execution; it is distinct from the external IM2P CPU-functional op5 backend. Use the verified local path variables described above.

```sh
rtk proxy cmake -S . -B build-potal-a4nks-cpu \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_COMMON=ON \
  -DGGML_GEMMINI=ON -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_BACKEND_DL=OFF \
  -DGGML_GEMMINI_OPTION=CPU -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE \
  -DGGML_GEMMINI_ACTIVATION_BITS=4 -DGGML_GEMMINI_WEIGHT_BITS=4 \
  -DGGML_GEMMINI_DIM=32 -DGGML_GEMMINI_BLOCK_SIZE=32 \
  -DGGML_GEMMINI_ACTIVATION_QUANT=EXSIA -DGGML_GEMMINI_ENABLE_RMD=ON \
  -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU -DGGML_GEMMINI_ENABLE_OPENMP=OFF \
  -DGGML_GEMMINI_ENABLE_A4NKS_ATTENTION=ON \
  -DLLAMA_CURL=OFF \
  -DGEMMINI_SW_PATH="${POTAL_GEMMINI_SW}" \
  -DPython3_EXECUTABLE="${POTAL_PYTHON}"
rtk proxy cmake --build build-potal-a4nks-cpu -j 4
rtk proxy ctest --test-dir build-potal-a4nks-cpu --output-on-failure -R 'potal|gemmini.attention|gemmini.exsia|gemmini.rmd|act.quant.i32'
```

New scenarios below belong in the existing test owners listed in the reuse audit. Register their actual case names in the existing CMake setup during implementation; do not create `test-gemmini-potal-a4` or `test-gemmini-attention` just because an older version of this plan named them. Recheck executable/case names before running and record the actual commands; no future scenario is claimed to pass now. The generated-header/logical-stripe geometry gate must pass before this recipe is original-policy evidence: today's generic CPU recipe generates ACC_ROWS=1024. Fix the shared logical geometry without changing an unrelated legacy profile; preserve its anchor across any physical tile subdivision. Also configure a fresh A8 profile with A4NKS OFF and a Gemmini-OFF build for compatibility tests. On nano, `rtk` was unavailable; equivalent direct commands were used and should be recorded as executed.

### Gemmini simulator recipe

Use the same configuration with a separate `build-potal-a4nks-im2p` directory and overrides `GGML_GEMMINI_OPTION=WS`, `GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM`, `GGML_GEMMINI_DEFAULT_RMD_BACKEND=WS`, `IM2P_SIM_ROOT=${POTAL_IM2P_ROOT}`, `IM2P_SIM_IMPLEMENTATION=GEMMINI_HP1`, `CYCLE_SIM=0`. Use the repository's verified archive resolver and record the resolved A4/W4/DIM32 identity. Run the operator/provider parity cases on the actual raw op0 provider. A fake callback, old 48-case certificate or CPU-functional op5 trace does not satisfy this gate. If the matched artifact cannot be built/loaded, report simulator validation blocked and keep that criterion incomplete.

## Execution strategy
### Execution order and isolated comparisons
The user requested one change at a time with comparison and adoption decisions.
The dependency table describes technical independence, not authorization to batch
policy changes or delegate to agents. Three work phases close cohesive contracts.

- Phase 1: tasks 1-3, reference/representation/dispatch contracts.
- Phase 2: tasks 4-7, StreamQuant, exact raw-dot execution, attention operands and cost records.
- Phase 3: tasks 8-10, linear integration, attention integration, then model/regression verification.
- Keep CMake registrations and shared metadata changes under one integrator at each phase. Task 2 finishes direct digit emission, v8 compatibility and slicing before tasks 4/5/6 edit their consumers. Tasks 4 and 6 may research independently, but changes to shared quantization headers are sequenced. Task 10 waits for tasks 8 and 9.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | none | 4,5,6,7 | 2,3 with distinct files |
| 2 | none | 4,5,6,7,8 | 1,3 with coordinated test registration |
| 3 | none | 9,10 | 1,2 with coordinated configuration edits |
| 4 | 1,2 | 8,9 | 5,7; coordinate shared quant headers with 6 |
| 5 | 1,2 | 8,9 | 4,6,7 |
| 6 | 1,2 | 9 | 5,7; coordinate shared quant headers with 4 |
| 7 | 1,2 | 8,9,10 | 4,5,6 with separate observer ownership |
| 8 | 4,5,7 | 10 | 9 except shared dispatcher/observer edits |
| 9 | 3,4,5,6,7 | 10 | 8 except shared dispatcher/observer edits |
| 10 | 8,9 | F1-F4 | none |

## Todos
> Implementation + Test = ONE todo. Never separate.
- [ ] 1. Establish the corrected integer specification and independent oracle
  Current evidence: the independent scalar primitive oracle and seven tests exist. Full StreamQuant/linear/newly-computed-softmax attention and model fixtures are still open; primitive success does not complete this task.
  Recommended task executor category: deep-low - exact arithmetic and fixtures, with an explicit reference contract.
  What to do / Must NOT do: Extend the existing `tests/potal_a4_reference.py` and `tests/test-potal-a4-reference.py` with the supplied algorithm's remaining stages, provenance and corrected boundaries; make selective-linear/all-corrected-Q the native-comparison entry. Reuse existing direct-all and clip-plus-residual experiment functions on identical E/u rather than copying those experiments into another module. Add compact deterministic fixture data only where required. Use Python integer accumulation, real BK tails, finite-input validation, and actual masked softmax from newly computed S. Update `experiments/potal_a4/ORIGINAL_SPEC.md` with contract decisions. Keep the supplied captured-P test identified as a component test. Require original-policy equivalence, but do not claim sf_tradeoff/attn_quant parity without those absent references.
  Closes: GAP-3,GAP-7. Phase 1; blocks 4,5,6,7.
  References: `experiments/potal_a4/ORIGINAL_SPEC.md` (supplied sections A-D and known reference limitations), the numerical decisions above, and the current experiment/self-check sources; `evaluation/activation/__init__.py:18` (different existing metric); `tests/test-gemmini-rmd.cpp:2323` (independent native oracle pattern).
  Acceptance criteria: Codes/exponents/lanes/counts are deterministic; `shift(2^30,-32)==0`; main plus direct digits reconstruct INT32 endpoints and 0x77777778 without a scalar residual input when correction applies. The 1/2^63 two-block example yields the correct rounded result; unsigned P codes 0..255 reconstruct exactly. Q equals the supplied corrected all-code rule; accepted linear X=1.99 at shape 32x32 remains 1.75 with no upper fragment, and the changed direct-all result is rejected. The double-rounding fixture and IEEE boundary fixtures enforce numerical decision 6. Test fixtures include actual linear stationary codes, not only random float weights.
  QA happy: `rtk proxy "${POTAL_PYTHON}" tests/test-potal-a4-reference.py` exercises primitive, linear and complete masked-attention fixtures; evidence `.omo/evidence/potal-a4nks/task-01-reference.log`.
  QA failure: The same suite injects NaN/Inf, negative probabilities, ragged shapes and an invalid anchor; each is rejected with a specific error. Mutation tests inside the suite use the old clipped-right-shift and old int64 accumulator to demonstrate that the new assertions fail; evidence `.omo/evidence/potal-a4nks/task-01-negative.json`.

- [ ] 2. Emit upper A4 digits directly and retain legacy packet compatibility
  Recommended task executor category: deep-low - one representation change with slicing, lifecycle and hash boundaries.
  What to do / Must NOT do: Add the narrow upper-A4 reset/emission methods and v8 contract described above to the existing bitmap builder; share finalization and compaction with the legacy entry. Reuse the int32 balanced-digit helper. Add digit-preserving v8 slicing, checked main+digit reconstruction and coordinate hashing; guard legacy packet-to-scalar and CPU-direct consumers against v8. Keep v7 defaults and scalar APIs, int32 planes and extrema unchanged. Do not introduce a parallel packet stack or a dense int64 residual plane. A4 slot ownership/call-site wiring belongs to task 4; common helpers must be ready here.
  Closes: GAP-2,GAP-8. Phase 1; blocks 4,5,6,7,8.
  References: `ggml/src/ggml-gemmini/residual/rmd/{rmd-types.hpp:39,rmd-types.hpp:152,rmd-builder.cpp:143,rmd-builder.cpp:215,rmd-builder.cpp:770,rmd-builder.cpp:891,rmd-builder.cpp:906,rmd-bitmap-builder.hpp:19,rmd-bitmap-builder.cpp:97,rmd-bitmap-builder.cpp:158,rmd-compose.cpp:112}`; `residual/residual-capture.hpp:93`; `residual/direct/direct-types.hpp:13`; `ggml-gemmini-telemetry.cpp:167,190`; `tests/test-gemmini-rmd.cpp`; `tests/test-gemmini-rmd-telemetry.cpp`; `scripts/utils/potal_records.py`. Shorthand paths are under `ggml/src/ggml-gemmini/`.
  Acceptance criteria: v8 retains original lanes 1..8, row/K maps, tails and zero gaps; round trips with main at both int32 endpoints and 0x77777778. Row slicing across original stripes does not reconstruct a scalar residual. A change to lane, sign or coordinate changes the canonical digit hash. No v8 scalar extrema are reported; v7 hashes/fixtures remain valid. Empty upper support produces no packet. Packet payload still stores signed-byte digits and no dense residual buffer.
  QA happy: Extend the existing RMD/bitmap/packet/telemetry tests with direct decomposition, v8 slice/reconstruction/hash, repeated stripe reuse and mixed empty/nonempty stripes; run `ctest --test-dir build-potal-a4nks-cpu --output-on-failure -R 'gemmini.rmd|act.quant.i32'`. Evidence `.omo/evidence/potal-a4nks/task-02-limbs.json` and `task-02-legacy.log`.
  QA failure: Extend their rejection fixtures for v8 lane0, lane9, out-of-range digit, malformed row/K map, stale slot data, mixed packet semantics and calls into legacy scalar expansion. Failures retain previously published outputs; legacy telemetry decoding remains covered. Evidence `.omo/evidence/potal-a4nks/task-02-rejection.json`.

- [ ] 3. Add explicit attention roles and opt-in configuration
  Recommended task executor category: unspecified-high - a small public configuration and graph/backend contract.
  What to do / Must NOT do: Add the default-OFF selector and prerequisite checks, inline role accessors and slot reservation. Propagate the selector to llama, Gemmini and tests. Add shared eligibility checks and role-first capability classification without enabling unfinished execution. Preserve GGML_PREC_F32, graph-copy behavior, plain MUL_MAT, flash precedence and CPU backend fallback. Do not repurpose tensor.extra or add a GGML op.
  Closes: GAP-4. Phase 1; blocks 9,10.
  References: `cmake/ggml-gemmini-options.cmake:112,242`; `cmake/ggml-gemmini-config.cmake:19`; `src/CMakeLists.txt:38`; `ggml/include/ggml-gemmini.h`; `ggml/include/ggml.h:599`; `ggml/src/ggml.c:2823`; `ggml/src/ggml-backend.cpp:1768`; `src/llama-graph.cpp:1183`; `ggml/src/ggml-gemmini/{ops.cpp:557,ops.cpp:2227,ggml-gemmini.cpp:169}`.
  Acceptance criteria: Role hints survive graph copy and do not change precision. Unsupported tagged nodes cannot reach shared-weight execution. Invalid A8/DIM16/RMD-OFF combinations with A4NKS ON fail configure clearly. A4NKS OFF is the default. Record logical and physical memory profiles separately; the A4 policy's logical geometry must be `(4,4096,512)`. An ACC1024-derived logical stripe cannot silently pass. Compare stripe_rows at I/N/K=1,31,32,33 and capacity-growth boundaries; physical splits preserve the parent stripe anchor.
  QA happy: Extend existing backend-op / Gemmini routing tests with role-copy, precision and capability cases; use current build/config contract tests for selector propagation. Evidence `.omo/evidence/potal-a4nks/task-03-roles.log`.
  QA failure: The same owners cover unknown hint, invalid GQA ratio, unsupported layouts and tags on non-MUL_MAT. CMake contract tests exercise incompatible profiles; evidence `.omo/evidence/potal-a4nks/task-03-config.log`.

- [ ] 4. Fuse A4 StreamQuant folding with direct limb emission
  Current evidence: Shared A4/A8/A16 source-zero exclusion in optimized full/tail/reference paths is implemented and user-adopted. Release CTests pass A4 9/9 with OpenMP ON and OFF, A8 14/14 and A16 9/9; current 10-chunk PPL evidence is in NANO_PROGRESS.md and results-source-zero-common. Preserve the common policy. Geometry, rounding/large shifts, low-digit main and fused packets remain open.
  Recommended task executor category: deep-low - one coupled quantization contract across optimized/reference paths.
  What to do / Must NOT do: Keep role-specific rho (linear=2, Q=6). Fix source-zero exclusion in full, partial and reference block paths; retain top/sigma adaptation, strict integer sigma and distinct E2 behavior. Linear computes v with its original corr/sat rule; Q uses v=u. Feed each effective v once to the digit helper; write d0 to main and emit all required upper digits. Give each A4 pipeline slot the task 2 bitmap builder; skip scalar capture and state_.residual allocation/writes, including CPU mode. Retain the metadata needed for original linear corr and enabled observers. Fix large negative shifts and define half-even qround without changing process-global rounding state. Gate A4 representation changes on the requested policy; retain the shared source-zero population for all widths and do not enable eight-bit PEs for Q.
  Closes: GAP-1,GAP-3,GAP-8. Phase 2; depends 1,2; blocks 8,9.
  References: `ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:29,256,342,528,710`; `exsia_shift.hpp:18`; `folding.cpp:27,68,127,145`; `exsia-state.hpp:291,590`; `exsia.cpp:171,415,456,692,1214,1411,1659,1809`; `ggml-gemmini-config.hpp:128`; `residual/residual-capture.hpp:93`; `tests/test-gemmini-exsia{,-profile}.cpp`; `tests/CMakeLists.txt:667,840`.
  Acceptance criteria: All A4 implementations match the original-policy reference E/q/M/digits and stripe partitions for full/tail blocks. Upper packet lane0 is absent; corrected INT32_MAX reconstructs exactly. Unselected linear u=8 produces main 7 and no upper digit; corrected u=8 is preserved. q in [-8,7] emits only main. Allocation/capture instrumentation proves zero A4 dense-residual bytes and zero scalar residual events, including reused slots. A4 tests compare digit reconstruction/support instead of relying on the removed scalar plane. Fresh A8 zero-heavy regressions use the shared source-zero policy; representation changes must preserve that adopted baseline's codes/exponents/masks.
  QA happy: Extend `test-gemmini-exsia` to compare fixtures for linear and Q policies, serial and available worker modes; retain its existing `source-zero-sigma`, `stripe-geometry` and `non-outlier-clipping` cases. The source-zero case runs at every supported width. Evidence `.omo/evidence/potal-a4nks/task-04-stream.json`.
  QA failure: Extend ExSIA / activation-i32 boundary tests to change the local test thread rounding mode and check invariant qround output, extreme exponent gaps and invalid finite-domain input; evidence `.omo/evidence/potal-a4nks/task-04-boundaries.json`.

- [ ] 5. Execute raw A4 fragments and merge with exact wide arithmetic
  Current evidence: compatible IM2P source `20b82f058aa8b0e69c48d1d00790d8e19eaa6f66` built A4/W4/DIM32 archives under `/tmp/potal-nano-hp1-matched`. The existing `experiments/potal_a4/im2p_raw_probe.cpp` covers K1/K31/K32 and tails. Reuse it for manifest/link/runtime admission before integration; its execution has not yet been certified.
  Recommended task executor category: deep-high - the central hardware/host arithmetic invariant requires coherent implementation.
  What to do / Must NOT do: Reuse `Im2pCompactDot` with BYPASS, no SCU carriers and K<=32 on the matched Gemmini HP1 provider. Provide the same bounded raw-dot contract in Gemmini CPU mode, consuming the same v8 digit packets rather than CPU-direct scalar residuals. Extend `matmul/dense.cpp`, `matmul/execution.cpp` and existing RMD packing/execution at their responsible boundaries. Implement the missing exact accumulator with the checked int64 fast path and exponent-binned slow path defined above, reusing staging and publication ownership. Extract a private shared helper only where the actual linear/QK/PV callers need it; the previously proposed `potal_accumulator.hpp` and `matmul/potal.{hpp,cpp}` are not mandatory files or a separate executor. Keep original block/lane identities through row packing. Validate all results before publication. Never recover a wide value from saturated op5 output. Leave unsupported CYCLE_SIM/optrace rejected and explicit.
  Closes: GAP-3,GAP-5. Phase 2; depends 1,2; blocks 8,9.
  References: `ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp:121,147`; `rmd-im2p-executor.cpp:678,684,744`; `quants/common/hp1_scu.hpp:31`; `docs/development/rmd-run-aware-production.md`; read-only external evidence `${POTAL_IM2P_ROOT}/sim/backends/gemmini_hp1/runtime.cpp:70` and `${POTAL_IM2P_ROOT}/frontend/src/im2p_cpu_functional_compute.cpp:89`.
  Acceptance criteria: Raw K32 A4 dots are exact and int32-safe. Positive/negative exponent spreads and catastrophic cancellation match Python integer coefficients before final conversion, including the 2^63 counterexample. Exact->F64->F32 results match the independent oracle bitwise for double-rounding, overflow, subnormal and signed-zero fixtures. Lane weights apply by original row-map lane, so packing two lanes in one DIM tile preserves both numerical output and physical fragment count. Existing sat32 helpers/tests remain unchanged for legacy routes.
  QA happy: Extend existing matmul/RMD executor/provider tests with exact-wide fixtures and run them on CPU and the matched simulator. Use the existing raw probe for initial archive admission. Evidence `.omo/evidence/potal-a4nks/task-05-cpu.json` and `task-05-im2p.json`.
  QA failure: Extend existing provider fault-injection tests for missing raw capability, invalid K>32, allocation/read/write failure and forbidden traced raw mode; all retain prior output and return the existing explicit status. Wide fixtures include fast-to-slow transition, ties and cancellation. Evidence `.omo/evidence/potal-a4nks/task-05-failure.json`.

- [ ] 6. Implement K, P and V policies with logical-view packing
  Recommended task executor category: unspecified-high - bounded operand conversion and view handling using established arithmetic.
  What to do / Must NOT do: Use existing quantization primitives, activation storage, tensor base-address helpers and F16 conversion; add the missing runtime operand policies at those integration points. Existing generic BLOCK quantization is not the C2/C4/C5 policy. Implement K block rho6 clamp[-128,119] and two signed passes; V token-block/channel rho2 clamp[-8,7]; P stripe-max uint8 with main lo-8 and lanes1/2. P feeds digits of q-lo to task 2's upper-digit emitter without a dense residual plane; do not balanced-decompose q itself or remove its zero point. Decode strides explicitly and stage one head/stripe at a time. Compute exponent-weighted colsumV once per KV head/invocation; cache only within that invocation. Extract an internal operand module only if needed by these callers; `quants/attention.{hpp,cpp}` is a possible destination, not an instruction to duplicate packing/arithmetic.
  Closes: GAP-6. Phase 2; depends 1,2; blocks 9.
  References: User spec C2,C4,C5; `ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:171`; `residual/rmd/rmd-builder.cpp:143`; `src/llama-graph.cpp:1464`; `src/llama-kv-cache.cpp:103`; task 1 reference and fixtures.
  Acceptance criteria: K lo+16hi is exact at -128/119 and both nibbles remain [-8,7]. All P codes 0..255 reconstruct exactly, with lane2 iff q>=128. All-zero P returns zero after compensation. Tail/padding changes do not change real-element statistics or colsum. F16/F32 strided views agree after F32 conversion.
  QA happy: Extend existing quantization/backend-op fixtures with K/P/V endpoints, zero compensation, tails and strided views; evidence `.omo/evidence/potal-a4nks/task-06-operands.json`.
  QA failure: The same fixtures reject nonfinite inputs, negative P, incompatible shapes/types and invalid stride extents before modifying destination buffers; evidence `.omo/evidence/potal-a4nks/task-06-invalid.json`.

- [ ] 7. Measure RMD preparation savings and reconcile actual submitted work
  Additional acceptance from the nano review: report source-zero/nonzero-q0 population, top, sigma, folded correction candidates and nonzero upper support with explicit denominators. Do not call a D>0-marked zero a sigma outlier or count an empty residual as a fragment. For timing, bind geometry, actual submissions/opcode, memory timing assumptions and artifact hashes; distinguish CPU wall time, integrated RTL cycles and predictor estimates. Compare prediction with RTL only for the same supported contract. The existing 331-cycle scaled-HP1 smoke does not predict raw op0; unsupported raw prediction stays explicitly incomplete. Repeat latency/peak-memory measurements without concurrent builds before performance comparisons.
  Recommended task executor category: unspecified-high - observer/schema changes with exact integer accounting.
  What to do / Must NOT do: Reuse evaluation observers and actual fragment submissions. Distinguish linear main/upper limbs, QK low-main/Q-upper limbs, K-hi dense/Q-upper limbs, PV main/upper limbs, weighted colsum and offset-broadcast terms. Record original block IDs, real/padded K, global lane-row count and actual raw submissions. Derive legacy report names `main_frag`, `clip_frag`, `add_ops` under `potal-a4-preserve-v1`; clip_frag labels additional passes. Separately record upper-digit positions/nnz, packet bytes, residual-plane bytes/writes, decomposition/capture calls, preparation time and host-wide merge time. Preserve unavailable scalar statistics and legacy hash/schema interpretation. Compare current clip-plus-residual, the accepted original-policy direct emitter and unadopted direct-all on identical E/u fixture sets; do not infer latency from counts.
  Closes: GAP-7,GAP-8. Phase 2; depends 1,2; blocks 8,9,10.
  References: `ggml/src/ggml-gemmini/ggml-gemmini-evaluation-observer.hpp:13,48`; `residual/rmd/rmd-run-aware.cpp:70`; `rmd-executor.cpp:1667,1685`; `tests/test-gemmini-evaluation-metrics.cpp:186`; `scripts/utils/potal_records.py`; `evaluation/schemas/residual_metrics.schema.json`.
  Acceptance criteria: Two lanes with one active row each submit/count one packed upper-row fragment when DIM32 packing applies. K hi charges another Q main+upper work. Per-head add terms equal Tkv*d+Tq*d, with GQA reuse recorded. With BK=DIM32, 1..32 selected columns in one block remain one fragment. Both linear and Q fragments equal their original-policy references; accepted linear X=1.99 has no upper fragment. Native preparation proves zero dense-residual allocation/writes and no scalar recapture/re-decomposition, while reporting remaining packet/scratch memory. Timing results include repeated-run spread; no speedup or fragment decrease is required.
  QA happy: Extend current run-aware/evaluation tests and `experiments/potal_a4/native_probe.cpp` for costs and limb preparation; include all-in-range, sparse-outlier, dense-upper and endpoint fixtures with allocation/call counters and repeated timing. Evidence `.omo/evidence/potal-a4nks/task-07-costs.json` and `task-07-preparation.json`.
  QA failure: Existing/new evaluation tests inject duplicate/missing fragments, wrong lane maps, unrecorded fallback and double colsum charging; reducers reject mismatches rather than recomputing a plausible total. Evidence `.omo/evidence/potal-a4nks/task-07-invalid.log`.

- [ ] 8. Integrate corrected StreamQuant and wide execution into linear A4
  Recommended task executor category: unspecified-high - reuse the existing linear scheduling lifecycle without changing other profiles.
  What to do / Must NOT do: Route the requested A4/DIM32 ExSIA profile through shared original-policy main/v8-packet/raw-wide execution using the supplied stationary weight codes and original K-block exponents. Resolve the A4 path before legacy cpu_direct/ws scalar payload assumptions, including dequantization and row-sliced sequential execution. Preserve stripe selection, result staging, failure cleanup and main/upper completion rules. Keep non-A4 and unrelated weight routes on their existing implementation. Include real Q4_HP1 weight fixtures; do not re-quantize source weights to make a test pass.
  Closes: GAP-1,GAP-5,GAP-8. Phase 3; depends 4,5,7; blocks 10.
  References: `ggml/src/ggml-gemmini/matmul/dense.cpp:524,1046`; `matmul/execution.cpp:807,1572,1866`; `ops.cpp:966`; `docs/development/rmd-gemm-scope.md`; task 1 oracle and tasks 4/5 helpers.
  Acceptance criteria: Native A4 linear outputs and observed fragments equal the original selective oracle for sparse/dense upper limbs, multiple stripes, extreme shifts and partial output/K tiles. Decode Q4_HP1 channel_scale and m into the full stationary scale, and compare real-checkpoint weights to the native decoder before GEMM tests. Row slices crossing stripe boundaries preserve endpoint digits without scalar expansion. Changed +8 output/counts are regressions, not accepted differences. Existing A8 and unsupported-weight behavior remain covered. Cancellation across main and upper limbs is merged before final output rounding.
  QA happy: Extend `test-gemmini-matmul` native-Q4 and multiblock coverage for the complete original-policy route and run the same fixtures on `build-potal-a4nks-cpu` and `build-potal-a4nks-im2p`; evidence `.omo/evidence/potal-a4nks/task-08-linear-cpu.json` and `task-08-linear-im2p.json`.
  QA failure: Extend existing matmul transaction-atomicity coverage to fail one residual/raw submission after main work and verify no partially published output; run legacy A8 CTest cases in the fresh A8 build. Evidence `.omo/evidence/potal-a4nks/task-08-regression.log`.

- [ ] 9. Connect complete a4nks attention through the real graph
  Recommended task executor category: deep-low - head/cache layouts and graph dispatch must agree with the arithmetic.
  What to do / Must NOT do: Extend `ops.cpp`/`ops.hpp` with role-specific dispatch into tasks 4-6's shared quantization/execution. Start in the existing owner; extract an internal attention helper only if its responsibility warrants it. For QK use Q StreamQuant rho6/direct-all with both K passes; retain the graph's scale/mask/FP softmax; for PV use the unchanged P/V policies plus weighted-colsum offset and v8 upper-digit packets. Handle `Hq/Hkv` head mapping and Tq/Tkv independently. Apply role tags at the QK and PV nodes inside the existing shared `llm_graph_context::build_attn_mha` only after complete-case eligibility. Record actual route coverage, including explicit fallback. Do not force K/V views into the old shared-weight executor or add another attention graph.
  Closes: GAP-4,GAP-6,GAP-7. Phase 3; depends 3,4,5,6,7; blocks 10.
  References: `src/llama-graph.cpp:1183,1252,1281,1288,1393,1464`; `src/llama-kv-cache.cpp:20,103,533`; `src/llama-context.cpp:2147`; `ggml/src/ggml-gemmini/{ops.hpp,ops.cpp:2227,ggml-gemmini.cpp:169,CMakeLists.txt:76}`; `ggml/src/ggml-cpu/ops.cpp:4679`; `tests/test-backend-ops.cpp:1975,2452,3284`.
  Acceptance criteria: Deterministic complete QK->softmax->PV tests consume their newly computed S. Prefill 33 tokens and subsequent single-token decode work with strided F16/F32 KV and GQA. Future-token and wrong-sequence probabilities stay zero, and zero-point cancellation remains exact. Supported cases execute both tagged operations on Gemmini; a fallback cannot pass that assertion.
  QA happy: Extend current backend-op/routing tests and llama graph callback coverage for complete attention and cache/GQA cases, repeated for the matched simulator; evidence `.omo/evidence/potal-a4nks/task-09-graph.json` and `task-09-cache-gqa.json`.
  QA failure: The same owners cover flash ON, no-KV-offload, unsupported MLA/quantized-KV/layout and malformed role contracts; FP fallback is explicit, output remains valid, and a4nks counters cannot claim those nodes. Evidence `.omo/evidence/potal-a4nks/task-09-fallback.json`.

- [ ] 10. Verify real model coverage, numerical parity and compatibility
  Recommended task executor category: unspecified-high - source-bound integration and reproducible model evidence.
  What to do / Must NOT do: Extend existing production parity/model tools with capture/compare using current llama/backend callback patterns. Reuse `experiments/potal_a4/native_model.cpp` and existing capture scripts where their contracts fit; do not assume a new dedicated attention harness. Use the local GPT-2 Q4_HP1 model and capture layers 0/5/11, actual projection weights and Q/K/V/P at fixed prompt/token IDs. Add a Llama GQA smoke case after locating and recording the Llama 3.2 1B Q4_HP1 model; absence is incomplete evidence. Run both Gemmini CPU mode and matched simulator operator fixtures. Require original-policy equivalence of the optimized preparation on identical captures, including +8 behavior, per-layer upper fragments, preparation time and memory. On the same evaluation execution path, compare baseline and optimized logits, per-token NLL and PPL exactly; any degradation or unexplained difference from a representation optimization fails acceptance. Separately report old-product-policy and FP32-versus-A4NKS accuracy so policy changes and existing quantization loss are not attributed to storage optimization. Use the fixed WikiText-2 test prefix for each 10-chunk PPL comparison, with separately labeled compact operator smoke fixtures.
  Closes: GAP-7,GAP-8 and proves IS-1..IS-6. Phase 3; depends 8,9; blocks F1-F4.
  References: `examples/eval-callback/eval-callback.cpp`; `tools/perplexity/perplexity.cpp:492`; `tests/test-backend-ops.cpp:4680`; `experiments/potal_a4/native_model.cpp`; local model `models/gpt2-base/gpt2.Q4_HP1.gguf` and the Llama model after path verification; `tests/CMakeLists.txt`; existing `experiments/potal_a4/{ORIGINAL_SPEC,README,NANO_PROGRESS}.md`.
  Acceptance criteria: Store model/token hashes, resolved dimensions, build/IM2P identity and full operator coverage. Captured integer operands/results match the independent original-policy reference. Run prefill+decode and require the same-path before/after logits, per-token NLL and PPL to agree, not merely remain finite or have similar averages. A8, Gemmini OFF, A4NKS OFF and FP/flash alternatives pass relevant regressions. Missing models/artifacts, unexplained output differences or unexpected fallback are incomplete evidence, never a pass.
  QA happy: Run GPT-2 layer captures and Llama GQA model fixtures through the extended existing model tool after verifying model paths; record its actual CLI. Run `build-potal-a4nks-cpu/bin/llama-perplexity -m models/gpt2-base/gpt2.Q4_HP1.gguf -f wikitext-2-raw/wiki.test.raw -c 512 -b 512 -ub 512 -t 4 -tb 4 --chunks 10 -ngl 0`. Use automatic ACCEL placement and verify actual Gemmini coverage in both runs; do not use GPU-layer flags as proof. Evidence `.omo/evidence/potal-a4nks/task-10-models.json`, `task-10-ppl.log`, `task-10-regressions.log`.
  QA failure: The extended tools reject a missing/mismatched model, stale source/build identity, zero executed a4nks nodes, mixed CPU fallback in a required Gemmini run and an altered integer fixture. Exercise their help/invalid-argument paths using the actual CLI. Evidence `.omo/evidence/potal-a4nks/task-10-evidence-rejection.log`.

## Final verification wave
> Run after all implementation tasks. Each check needs observed evidence from the actual target; report unmet criteria explicitly.
- [ ] F1. Plan compliance audit
  Check every task's evidence against the live source SHA and build identity, all A-D arithmetic requirements, and actual CPU/Gemmini dispatch. Missing simulator evidence keeps this lane incomplete. Record `.omo/evidence/potal-a4nks/f1-compliance.md`.
- [ ] F2. Code quality review
  Review direct digit emission, v7/v8 and canonical hash compatibility, digit-only slicing, checked allocations/strides/shifts, exact accumulator conversion, concurrent stripe ownership and failure atomicity. Reject dense residual re-materialization, unrelated refactors and silent narrowing. Record `.omo/evidence/potal-a4nks/f2-code-review.md`.
  Check new files/functions/options against the reuse audit: each must have a demonstrated missing responsibility and actual callers. Reject a duplicated quantizer, decoder, compactor, provider, graph, metric path or equivalent test runner. Verify that extraction moved ownership instead of keeping two implementations.
  After each adopted increment, remove temporary policy switches, orphan helpers and replaced A4 paths that no supported caller uses. Confirm references/callers and run the affected regressions. Preserve still-required A8/v7 consumers and before/after measurement artifacts; historical evidence is not orphan product code. Do not postpone all cleanup until the final wave.
- [ ] F3. Real manual QA
  Re-run the extended existing tests/model tools' real graph/prefill/decode cases, CLI help/invalid arguments and the fixed model/PPL smoke through the real binaries. Record commands and statuses at `.omo/evidence/potal-a4nks/f3-manual-qa.md`; do not substitute reference-only or grep checks.
- [ ] F4. Ideal-state fidelity
  Map every success row below to observed output, independent oracle data and actual submitted work. Check that the physical raw-PE/host-wide implementation is described honestly and no cycle-speed claim is inferred. Record `.omo/evidence/potal-a4nks/f4-fidelity.md`.

## Commit strategy

The snapshot and nano handoff commits/pushes were explicitly requested. Keep verified implementation increments buildable and separate the oracle, direct limb packet contract, StreamQuant, attention operands/dispatch, execution integration and evaluation evidence. Preserve unrelated files. Do not commit models, token arrays, capture arrays, environments or build outputs; retain compact result records and their provenance.

## Success criteria
> One row per IS row. The plan is complete only when every IS row has a delivering todo and a proving QA scenario; F4 checks the delivered behavior against these rows 1:1, and a shortfall becomes new `- [ ] N.` rows, never a note.
| IS | Delivering todo(s) | Proving QA scenario | Evidence |
| --- | --- | --- | --- |
| IS-1 | 1,4,6,8,9 | stream, operands, linear, complete graph | task-04-stream.json; task-06-operands.json; task-08-linear-*.json; task-09-graph.json |
| IS-2 | 1,2,5 | negative boundaries, direct limb endpoints, wide-gemm cancellation | task-01-negative.json; task-02-limbs.json; task-05-*.json |
| IS-3 | 3,9,10 | roles, cache-gqa, actual model coverage | task-03-roles.log; task-09-cache-gqa.json; task-10-models.json |
| IS-4 | 5,7,9 | packed lane rows, executed fragments, once-per-head colsum | task-05-*.json; task-07-costs.json; task-09-graph.json |
| IS-5 | 2,3,8,9,10 | legacy telemetry, invalid config, A8/FP/flash regressions | task-02-rejection.json; task-03-config.log; task-08-regression.log; task-09-fallback.json; task-10-regressions.log |
| IS-6 | 2,4,7,8,10 | fused emission, no dense residual/capture, digit slicing, preparation and work comparisons | task-02-limbs.json; task-04-stream.json; task-07-preparation.json; task-07-costs.json; task-10-models.json |

All table evidence paths are relative to `.omo/evidence/potal-a4nks/`. Plan preparation itself does not mark these implementation or final-verification tasks complete.
