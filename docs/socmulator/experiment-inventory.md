# NPU experiment and preservation inventory (Phase 0)

Snapshot: inventories refer to the pinned Phase 0 revision, excluding concurrently added untracked Phase 1 API/tiling files. This audit wrote only the three requested Markdown files. Build and hardware outcomes remain UNVERIFIED; baseline results belong to the separate baseline report.

Audit date: 2026-11-25. This is a source-based classification for Phase 1, not authorization to remove code. No builds, model runs, target probes or historical artifact replays were performed. Hardware results claimed by existing reports remain UNVERIFIED for this HEAD unless separately reproduced by the baseline agent. Source references use repository-relative file:line. Categories describe responsibility, not quality or importance; a file may cross categories.

## Classification rule

- CORE: graph integration, integer numerical preparation, planning/execution and output correctness that Phase 1 must preserve; active production source lists are evidence of reachability (ggml/src/ggml-gemmini/CMakeLists.txt:81-120; ggml/src/ggml-potal/CMakeLists.txt:262-319).
- HARDWARE: ISA definitions, checked command encoders, raw issue, completion, DMA ordering, resident/indexed state and target geometry contracts (ggml/src/ggml-potal/include/potal.h:358-381; ggml/src/ggml-potal/include/potal_rocc.h:37-168; ggml/src/ggml-potal/potal.c:847-899).
- REFERENCE: independent numeric or command-level oracles and native device models; they do not prove physical timing or completion (ggml/src/ggml-potal/matmul/dense.cpp:486-499; ggml/src/ggml-potal/CMakeLists.txt:613-631; ggml/src/ggml-gemmini/fp32/sdk.hpp:8-26).
- EXPERIMENT: explicit alternate routes, feasibility prototypes, comparative baselines and target observation fixtures; retain isolation until usage/contract analysis supports migration (ggml/src/ggml-potal/CMakeLists.txt:418-501; ggml/src/ggml-gemmini/CMakeLists.txt:125-208).
- PROFILING: counters, clock domains, telemetry, diagnostic tracing, logging, dump and evidence-analysis tools (ggml/src/ggml-gemmini-utils/CMakeLists.txt:6-16; ggml/src/ggml-potal/CMakeLists.txt:315-335; ggml/src/ggml-potal/trace/hw-window.cpp:62-92).
- LEGACY: compatibility aliases, superseded descriptions or unreferenced wrappers/config artifacts; the label alone does not establish global non-use (cmake/ggml-gemmini-options.cmake:186-193; ggml/src/ggml-gemmini/run_llama.sh:4-31; ggml/src/ggml-gemmini/2nd_run_llama.sh:5-32).

## Must-preserve feature ledger

| Feature | Category / Phase 1 boundary | Confirmed implementation and preservation requirement |
|---|---|---|
| Gemmini baseline execution | CORE + HARDWARE; matmul/execution -> new API | execute_dense retains baseline CHANNEL/TENSOR and HP1 paths; external tiled_matmul_* calls must be made self-contained without changing arithmetic/tiling. ggml/src/ggml-gemmini/matmul/dense.cpp:844-867. External command encoding/fence internals UNVERIFIED because SDK is external (cmake/ggml-gemmini-geometry.cmake:20-28). |
| PoTAL INT8 | CORE + HARDWARE; current checked API/tiler/issuer | Full backend requires A8W8/BK32 and routes FULL or EXSIA pipeline under explicit constraints. ggml/src/ggml-potal/CMakeLists.txt:194-245; ggml/src/ggml-potal/ops.cpp:656-695. Preserve native NO_HARDWARE failures (ggml/src/ggml-potal/potal.c:902-940). |
| PoTAL INT4 | CORE + HARDWARE; precision-specialized packing and API contract | Separate W4 target selection, fixed DIM32/packing identity, signed adjacent nibbles; source is not an unmodified copy. ggml/src/CMakeLists.txt:321-329; ggml/src/ggml-potal-w4a4/CMakeLists.txt:18-47; ggml/src/ggml-potal-w4a4/include/potal-int4.h:8-63. Keep independent W4 goldens and existing W8-only disabled labels until fixtures are ported (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:3-63). |
| Integer activation quantization | CORE; software processing | quantize_activation validates/reset state, calls algorithm dispatcher; EXSIA/TENSOR/TOKEN/BLOCK/STRIPE and matched widths remain meaningful. ggml/src/ggml-gemmini/quants/act/quantize.cpp:21-49; ggml/src/ggml-gemmini/quants/act/dispatch.cpp:150-160; cmake/ggml-gemmini-options.cmake:96-137. |
| Local adaptation | CORE; CPU software processing | LocalStage::run_optimized/full/partial, wide integer quantization and sigma detector; separate reference implementation retained. ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:266-278 and ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:335-511 and ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:540-739. Do not turn this CPU computation into an ISA feature. |
| Stripe folding | CORE; CPU software processing with ordered handoff | StripeFolding computes stripe theta, clips low code and captures higher residual limbs. ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:117-188 and ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:237-261. The stripe-ready event follows folding commit (ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:956-965). |
| Residual correction | CORE; software + matmul/execution | FULL uses CPU direct correction; pipeline can lower residual into PoTAL GEMM, then checked reconstruction/merge. ggml/src/ggml-potal/matmul/dense.cpp:435-474; ggml/src/ggml-potal/residual/stripe-residual.cpp:271-464; ggml/src/ggml-potal/matmul/stripe.cpp:1405-1425. RMD OFF is an ablation, not justification to delete correction (cmake/ggml-gemmini-options.cmake:284-286). |
| Paired GEMM | CORE + HARDWARE + REFERENCE | Explicit pair residency/layout, Main/Residual plan, validator and command executor; keep serial comparator too. ggml/src/ggml-potal/matmul/paired-auto.cpp:106-130 and ggml/src/ggml-potal/matmul/paired-auto.cpp:688-1036 and ggml/src/ggml-potal/matmul/paired-auto.cpp:3670-3742; tests documented at ggml/src/ggml-potal/CMakeLists.txt:613-646. H2/H3 names do not imply dead prototypes. |
| Indexed weight execution | CORE + HARDWARE; api/encode/transport + planning | Checked index config/write encoders and real custom-3 dispatch; plan index_program installs mapped residual indices. ggml/src/ggml-potal/potal.c:398-453 and ggml/src/ggml-potal/potal.c:1190-1191; ggml/src/ggml-potal/matmul/paired-auto.cpp:3515-3555. Preserve index capacity and resident-K coordinate checks. |
| Prepared/indexed weight preload | CORE; buffer/cache lifecycle + HARDWARE index path | Backend precompute preparation skips optional preload with success once compute_started; before compute it calls weights.prepare; cache get reuses prepared weights. ggml/src/ggml-potal/ggml-potal.cpp:313 and ggml/src/ggml-potal/ggml-potal.cpp:551-571; ggml/src/ggml-potal/matmul/weight_cache.cpp:113-173. CPU prepared cache is distinct from index-table preload and hardware B residency; preserve both. |
| Resident B reuse | CORE + HARDWARE; matmul/execution owns lifetime | H2 panel begin/prefetch/loop/retire and stream counterparts are reached from paired executor; resident views/ldb-only/stream have raw response commands. ggml/src/ggml-potal/matmul/paired-auto.cpp:3002-3050 and ggml/src/ggml-potal/matmul/paired-auto.cpp:3348-3405 and ggml/src/ggml-potal/matmul/paired-auto.cpp:3742; ggml/src/ggml-potal/potal.c:1180-1184. |
| Attention | CORE + HARDWARE + REFERENCE | llama sets GGML_POTAL_ATTENTION under GGML_POTAL; QK/PV have CPU prepare, hardware issue, checked reconstruction and strict role mode. src/CMakeLists.txt:46-50; ggml/src/ggml-potal/matmul/attention.cpp:609-746 and ggml/src/ggml-potal/matmul/attention.cpp:1163-1180 and ggml/src/ggml-potal/matmul/attention.cpp:1525-1535. INT4 upper residuals use radix16, not byte-radix code copied blindly (ggml/src/ggml-potal-w4a4/matmul/attention-residual.hpp:14-46). |
| CPU issuer and stripe pipeline | CORE; matmul/execution | Dedicated hart0 issuer, checked map/affinity, producer/fold/consumer/finalizer handoffs, backpressure and join. ggml/src/ggml-potal/matmul/execution.cpp:58-150 and ggml/src/ggml-potal/matmul/execution.cpp:389-445 and ggml/src/ggml-potal/matmul/execution.cpp:475-505; ggml/src/ggml-potal/matmul/stripe.cpp:465-697 and ggml/src/ggml-potal/matmul/stripe.cpp:1676-1731. OpenMP and explicit std::threads are separate modes (ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:995-1032). |
| INT64 reconstruction | CORE; software processing | Residual reconstruct_rows checks wide values before INT64 assignment; attention preserves pass/limb reconstruction order and separates final FP32 conversion. ggml/src/ggml-potal/residual/stripe-residual.cpp:426-464; ggml/src/ggml-potal/matmul/attention.cpp:143-147 and ggml/src/ggml-potal/matmul/attention.cpp:895-955. Do not merge INT32 partial saturation with final INT64 accumulation. |
| Hardware counters / profiling | PROFILING + HARDWARE; profiling delegates raw reads to transport | COUNTER returns uint64 rather than status; paired executor configures/reads counters and observation windows; CPU rdcycle is a different measurement domain. ggml/src/ggml-potal/include/potal_rocc.h:134-138; ggml/src/ggml-potal/matmul/paired-auto.cpp:1878-2047; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:91-112. Preserve hardware-vs-emulator provenance (ggml/src/ggml-potal/CMakeLists.txt:601-603). |

## Separately investigated optional and historical work

| Component / options | Classification | Keep / removal evidence and Phase 1 recommendation |
|---|---|---|
| IM2P adapter, ggml-gemmini-im2p.cpp, im2p/route.cpp; GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM, IM2P_SIM_* | EXPERIMENT + REFERENCE, active build path | KEEP as isolated simulator adapter. Native archive verification, CPU-functional CYCLE_SIM branch and production ops routing have live consumers, so no evidence supports deleting IM2P merely as an old experiment. cmake/ggml-gemmini-im2p.cmake:1-59 and cmake/ggml-gemmini-im2p.cmake:99-161; ggml/src/ggml-gemmini/CMakeLists.txt:271-373; ggml/src/ggml-gemmini/ops.cpp:1921-2026. External archives, simulator sources and hardware contract reproduction UNVERIFIED. |
| CYCLE_SIM vs hardware timing | PROFILING + REFERENCE | KEEP separate producer mode. WS compiles im2p_cpu_functional*.cpp, while generation binds target information; collection is not a physical NPU run. ggml/src/ggml-gemmini/CMakeLists.txt:271-290; cmake/ggml-gemmini-im2p.cmake:15-27. Do not rename this blindly to NPU_TRANSPORT=recording. |
| CUDA CPU-exact, HP1 SCU, attention helpers | REFERENCE + EXPERIMENT | KEEP numerical comparator. Optional libraries and graph interception are active; source/test linkage exists. ggml/src/ggml-gemmini/CMakeLists.txt:140-154; ggml/src/ggml-gemmini/ggml-gemmini.cpp:182-208; ggml/src/ggml-gemmini/matmul/dense.cpp:502-539; tests/CMakeLists.txt:1740-1757. CUDA execution and claimed exactness are UNVERIFIED by this audit. |
| GPU RTN and RTN attention | REFERENCE + EXPERIMENT | KEEP explicit comparator. Distinct RTN path intercepts graph work, links its CUDA library; RTN attention depends on CPU-exact. ggml/src/ggml-gemmini/CMakeLists.txt:125-138; ggml/src/ggml-gemmini/ggml-gemmini.cpp:193-200; ggml/CMakeLists.txt:155-180; tests/CMakeLists.txt:1760-1766. Not production custom-3 execution. |
| fp32/ and GGML_GEMMINI_FP32 | EXPERIMENT + HARDWARE + REFERENCE | KEEP separate FP32 DIM8 WS baseline. It replaces backend source list, includes native command model and block/attention/end-to-end/replay targets. ggml/src/ggml-gemmini/CMakeLists.txt:76-79; ggml/src/ggml-gemmini/fp32/backend.cmake:1-43. Native SDK macro shim intentionally models issue (ggml/src/ggml-gemmini/fp32/sdk.hpp:8-26). CPU replay subproject force-disables accelerator knobs (ggml/src/ggml-gemmini/fp32/cpu/CMakeLists.txt:4-14). Do not conflate with COMPUTE_TYPE=FLOAT. |
| pb/ and GGML_GEMMINI_PB/PB_WEIGHT_FORMAT | EXPERIMENT + HARDWARE + REFERENCE | KEEP partial-sum baseline. Explicit option constraints, active sources, Q8_0/MXINT8 tests and native RoCC model show use. cmake/ggml-gemmini-options.cmake:429-443; ggml/src/ggml-gemmini/CMakeLists.txt:158-208; ggml/src/ggml-gemmini/pb/pb-exec.cpp:288-296. Public arithmetic/attention comparisons depend on this lane; no removal evidence. |
| Production H2/H3 paired code (paired-auto.cpp, potal.c) | CORE + HARDWARE | KEEP. Session/panel/scale/index APIs are called from the backend executor; H2 labels describe protocol ownership and scheduling, not test-only code. ggml/src/ggml-potal/matmul/paired-auto.cpp:3002-3050 and ggml/src/ggml-potal/matmul/paired-auto.cpp:3348-3405; ggml/src/ggml-potal/potal.c:1914-1938. |
| POTAL_E3_TESTS; tests/e3/, potal-e3-context/prototype/kbench/window-diag | EXPERIMENT + REFERENCE + PROFILING | KEEP isolated acceptance/feasibility targets. CMake removes ops.cpp for test backend variants and wraps paired issue functions for diagnostics; no add_test declaration in that 51-line fragment, so ctest absence is not deadness. ggml/src/ggml-potal/CMakeLists.txt:418-421; ggml/src/ggml-potal/tests/e3/CMakeLists.txt:1-50. The feasibility report explicitly says full hardware microtile pipeline was not implemented (H3-6E3-microtile-pipeline-feasibility.md:3-10); do not call it production capability. |
| POTAL_E4A_TESTS; tests/potal-resident.* | EXPERIMENT + HARDWARE contract proof | KEEP isolated resident-B interface and model tests. Production source list does not include test resident library; gated target list does. ggml/src/ggml-potal/CMakeLists.txt:438-464. Report describes bounded end-aligned K28/J2 subset, not arbitrary interior Main addressing (H3-6E4A-resident-b-microtile-interface.md:7-11). Current hardware results UNVERIFIED. |
| POTAL_E4B1_TESTS; tests/potal-e4b1.* | EXPERIMENT + REFERENCE | KEEP software resident contract proof; explicitly built as separate library/test. ggml/src/ggml-potal/CMakeLists.txt:467-477. Report selects generated LOOP_WS extension and rejects raw untagged compute shortcut (H3-6E4B1-general-resident-microtile-contract.md:7-13). No evidence of unused target. |
| POTAL_E4B2_TESTS; e4b2 C11/rocc/hardware fixtures | EXPERIMENT + HARDWARE + REFERENCE | KEEP. Host test versus cross RISC-V executable selection, plus full backend hardware executable, are explicit. ggml/src/ggml-potal/CMakeLists.txt:111-125 and ggml/src/ggml-potal/CMakeLists.txt:406-410. API/resident functions in production potal.c must be evaluated separately from the optional fixtures (ggml/src/ggml-potal/potal.c:663-805). |
| POTAL_E4C1_TESTS; e4c1-test/bench/backend/support | EXPERIMENT + REFERENCE + PROFILING | KEEP isolated sequential paired microtile integration. It substitutes a benchmark ops/backend variant with its own source list and no default production promotion. ggml/src/ggml-potal/CMakeLists.txt:480-500. A historical implementation plan alone is not evidence of current integration (H3-8D-e4c1-e4c2-implementation-plan.md:5-11). |
| POTAL_E4C2_0_TESTS | EXPERIMENT + REFERENCE | KEEP software-only ownership/completion feasibility test; native-only ctest registration. ggml/src/ggml-potal/CMakeLists.txt:128-135. Report says the then-current resident API could not safely execute concurrent experiment and no physical experiment occurred (H3-6E4C2-0-prefetch-admission-completion-feasibility.md:7-10). Preserve this negative contract evidence. |
| K-tile, next-K prefetch, early LdB, stream | CORE alternate execution + EXPERIMENT + HARDWARE | KEEP opt-in policies and raw commands. Pipeline validates incompatible combinations rather than silently replacing routes; reports record both positive numerical and negative performance/overlap outcomes. ggml/src/ggml-potal/matmul/stripe.cpp:1620-1629; ggml/src/ggml-potal/potal.c:2390-2578; H3-8FG-performance-comparison.md:7-13; H3-8H-paired-streaming-implementation.md:3-14. No default promotion follows from static audit. |
| POTAL_B5_TESTS / B5_SHARED_BASELINE | EXPERIMENT + REFERENCE | KEEP isolated QK limb benchmark; optional oracle-selection compile definition has active target consumer. ggml/src/ggml-potal/CMakeLists.txt:423-435. INT64 production reconstruction remains CORE regardless of benchmark gate (ggml/src/ggml-potal/residual/stripe-residual.cpp:426-464). |
| POTAL_ATTN_SWEEP | EXPERIMENT + PROFILING | KEEP attention-only sweep tool; host includes backend sources, RISC-V uses target path. ggml/src/ggml-potal/CMakeLists.txt:514-523. It is an executable, not a registered test. |
| Hardware observation / paired-index probe | PROFILING + HARDWARE | KEEP encoder self-test and independent target probe. Probe target is DIM16/Linux and registered --self-test only when not cross-compiling; physical observation needs separate target execution. ggml/src/ggml-potal/CMakeLists.txt:73-97. Strict hardware timeline tests explicitly label emulation, not measured hardware time (ggml/src/ggml-potal/CMakeLists.txt:601-610). |
| potal-clock-probe, multicore-smoke, scale-path-bench | PROFILING + EXPERIMENT + HARDWARE (where issued) | KEEP. Cross-hart clock probe uses no accelerator; multicore target needs Linux/riscv and optional C OpenMP; scale/Index probes are bounded DIM16/Linux. ggml/src/ggml-potal/CMakeLists.txt:73-98 and ggml/src/ggml-potal/CMakeLists.txt:138-171. Counter clocks/role placement are part of research validity. |
| LOG_DUMP/LOG_DUMP_SCALE, dump/, node dumps | PROFILING + EXPERIMENT | KEEP gated evidence capture, separate from required hardware transport. Compile switches default zero; node dumps require selected directory/patterns. cmake/ggml-gemmini-options.cmake:411-419; ggml/src/ggml-gemmini/CMakeLists.txt:115-119; ggml/src/ggml-potal/ggml-potal.cpp:90-104 and ggml/src/ggml-potal/ggml-potal.cpp:171-211. No proof these are unused. |
| RAW_TRACE, planning/scale/graph diagnostics, timeline writers | PROFILING + EXPERIMENT | KEEP opt-in observation; primary benchmark runners reject diagnostic environments. ggml/src/ggml-potal/potal.c:1307-1309; ggml/src/ggml-potal/matmul/paired-auto.cpp:4112-4133; ggml/src/ggml-potal/tests/potal-full-inference.cpp:382-383; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:886-900. Transport extraction must preserve recording without forcing tracing into hot path. |

## Removal candidates: evidence and limits

No algorithm, NPU precision, attention implementation, baseline or registered test is recommended for deletion in Phase 1. The following are narrow migration/removal candidates. The absence-of-reference search was run before these documents were written, across repository text excluding build* directories and VCS internals; positive source/definition evidence is cited. No in-tree literal caller is not proof that a user never runs a script manually or passes a data path dynamically.

| Candidate | Confirmed evidence | Proposed disposition / unresolved use |
|---|---|---|
| ggml-gemmini/run_llama.sh | Script builds via relative ../../../, then invokes external update_rootfs.sh and launch_sim.sh. ggml/src/ggml-gemmini/run_llama.sh:4-31. Whole-tree literal run_llama.sh search found no caller/reference before audit docs. | LEGACY wrapper removal candidate after lab confirms manual entrypoints. External scripts and user invocation UNVERIFIED. Preserve build-riscv.sh itself. |
| ggml-gemmini/2nd_run_llama.sh | Same relative build/update/launch sequence; labels say update_rootfs_2.sh and launch_sim_2.sh, but executed commands still call unnumbered scripts. ggml/src/ggml-gemmini/2nd_run_llama.sh:17-32. No in-tree literal reference found. | LEGACY duplicate wrapper removal candidate, or correct explicit deployment entrypoint outside API refactor. Do not infer second-device behavior from its name. |
| ggml-gemmini/exsia_config.json | Data contains q defaults and per-architecture/layer overrides (ggml/src/ggml-gemmini/exsia_config.json:1-68). Whole-tree literal filename search returned no reference before audit docs. | LEGACY data removal candidate only for literal wiring; dynamic/external readers UNVERIFIED. Do not delete ExSIA algorithm or anchor-policy implementation. |
| IM2P_DIM | Explicit deprecated alias mapped to GGML_GEMMINI_DIM; still read by CMake/resolver and forwarded to external simulator make. cmake/ggml-gemmini-options.cmake:186-193; scripts/im2p-build-options.py:64-71; scripts/im2p-host-provision.sh:88-93. | LEGACY alias, MERGE into NPU_D after caller migration. NOT confirmed unused. |
| GGML_GEMMINI_ACT_METRICS | Alias enables ACT_QUANT_METRICS; both exported (cmake/ggml-gemmini-options.cmake:15-30; cmake/ggml-gemmini-config.cmake:54-58). | MERGE alias; preserve independent activation/residual/scale metrics. NOT unused. |
| OPTION=OS / FPGA_UART enum values | OS route is rejected; FPGA_UART explicitly deprecated/rejected (cmake/ggml-gemmini-options.cmake:142-154). | LEGACY rejected values, remove from advertised accepted choices when common selector is introduced; keep rejection diagnostics while callers migrate. No execution feature is lost. |
| PRINT_TILE/FULL_C/LOW_D/TRANSPOSE_B/WS_* exports | Compile definitions forwarded to selected SDK; bool validation exists (cmake/ggml-gemmini-config.cmake:22-30; cmake/ggml-gemmini-options.cmake:247-257). | Consumer effect in external SDK UNVERIFIED. Do not call these confirmed-unused or remove them solely because in-tree command macros are absent. |

## Root H3 report inventory

Every root H3-*.md file observed is listed below. All are EXPERIMENT/PROFILING provenance documents; older line references, branch names, measurements and implementation status refer to their historical checkpoint, not automatically this HEAD. Recommendation: KEEP the evidence (an archive/index move can be a later documentation task), especially negative outcomes. The checkpoint index explicitly warns that evidence artifacts are outside the repository under lab /tmp and may be cleaned (H3-8I-25-checkpoint-evidence.md:3-5). No report was used as a substitute for reading current source.

| Report | Classification / source evidence | Keep rationale |
|---|---|---|
| H3-6A-paired-performance-profile.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6A FireSim paired performance profile | Matched cold/steady paired performance profiling; retain controls and measurement boundaries. |
| H3-6B0-scale-path-analysis.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6B0: isolated Scale command overhead | Isolated scale command overhead attribution; preserve negative evidence against a hardware-only explanation. |
| H3-6B1-trace-cache-optimization.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6B1: cache paired trace selection at session open | Session-open trace-selection caching report; provenance for current trace dispatch design. |
| H3-6B2-trace-cold-path-report.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6B2: isolate the paired trace cold path | Outlined trace cold-path optimization report; preserve historical latency comparisons. |
| H3-6B3-checked-dispatch-report.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6B3: checked Paired command dispatch | Rejected checked-dispatch specialization; report explicitly says keep B2, so do not recreate rejected prototype from prose. |
| H3-6B4-remaining-scale-overhead.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6B4 - Remaining Paired Scale Overhead Attribution | Remaining scale overhead attribution; preserve reason planning/validation became next investigation. |
| H3-6C-fallback-decode-reuse-report.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6C - Paired Fallback Decode Reuse | Capacity fallback reuses decoded packet; preserve numerical and no-second-decode rationale. |
| H3-6D1-paired-planning-validation-analysis.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6D1 Paired Planning/Validation analysis | Planning/validation feasibility and ownership analysis; retain report-copy lifetime constraints. |
| H3-6E-paired-dataflow-scheduling-analysis.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6E: Paired Dataflow and Scheduling Analysis | Paired dataflow/scheduling analysis; preserve gap between resident reuse and full desired pipeline. |
| H3-6E3-microtile-pipeline-feasibility.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6E3: Micro-tile Pipeline Feasibility | Feasibility report states complete hardware microtile pipeline not implemented; keep explicit interface limitations. |
| H3-6E3-paired-bottleneck-isolation.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6E3: Paired Bottleneck Isolation | Matched-K paired bottleneck isolation; retain negative same-build comparison. |
| H3-6E4A-resident-b-microtile-interface.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6E4A: Resident-B Micro-tile Interface | Bounded resident-B interface evidence; distinguish constrained suffix execution from arbitrary residency. |
| H3-6E4B1-general-resident-microtile-contract.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6E4B1: General Resident-B Micro-tile Contract | General resident contract selects generated LOOP_WS extension; preserve rejected raw-compute shortcut rationale. |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6E4B2: Resident Micro-tile RTL/ISA Implementation | Resident microtile RTL/ISA implementation report; distinguish software, RTL and physical evidence. |
| H3-6E4C1-sequential-paired-microtile-integration.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6E4C1 - Scale-aware Sequential Paired Micro-tile Integration | Sequential paired microtile integration checkpoint; keep isolated implementation and deployment provenance. |
| H3-6E4C2-0-prefetch-admission-completion-feasibility.md:1-14 | EXPERIMENT / PROFILING provenance: H3-6E4C2-0 - Independent B-Prefetch Admission and Completion Contract | Prefetch ownership/completion feasibility; explicitly no unsafe concurrent experiment was attempted. |
| H3-7-GPT2-paired-attention-e2e-report.md:1-14 | EXPERIMENT / PROFILING provenance: H3-7: GPT-2 Paired Attention End-to-End Validation | Failed full 256/128 attention run; keep failure provenance rather than rewriting history after later fixes. |
| H3-7A-llama-full-inference-baseline.md:1-14 | EXPERIMENT / PROFILING provenance: H3-7A: llama.cpp Full Inference Baseline | Partial full-inference baseline; canceled/uncompleted checks must remain distinguished from PASS. |
| H3-7B-potal-attention-offload-implementation.md:1-14 | EXPERIMENT / PROFILING provenance: H3-7B: PoTal Attention Offload Implementation | PoTAL attention offload implementation with then-bounded PV correction domain. |
| H3-7B5-qk-host-int64-limb-reconstruction.md:1-14 | EXPERIMENT / PROFILING provenance: H3-7B5: QK Host-INT64 Limb Reconstruction | Host INT64 QK limb reconstruction report; retain original partial saturation and scaling requirements. |
| H3-7C-attention-pv-range-resolution-report.md:1-14 | EXPERIMENT / PROFILING provenance: H3-7C: Attention PV Correction Range Resolution | PV range resolution and later full-workload pass claim; do not erase earlier H3-7 failure record. |
| H3-8A-potal-target-build-profile.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8A: PoTal FireSim target build profiles | FireSim target profile wrapper provenance; compile defaults and target header identity context. |
| H3-8B-dim32-port-validation.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8B: DIM32 production port and validation | DIM32 port validation; differentiates native emulation and later real hardware evidence. |
| H3-8C-dim32-firesim-e2e-report.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8C: DIM32 FireSim functional and GPT-2 baseline | DIM32 full-inference baseline; report warns functional comparison is not DIM-only performance experiment. |
| H3-8D-e4c1-e4c2-implementation-plan.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8D: E4C1 production integration and E4C2 prefetch implementation plan | Plan-only document, not proof that planned E4C work was implemented. |
| H3-8D-final-report.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8 checkpoint and H3-8D final report | Checkpoint/final report retaining negative outcomes and opt-in K-tile/prefetch status. |
| H3-8D0-matched-performance-baseline.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8D0: matched DIM16/DIM32 Performance baseline | Matched DIM16/DIM32 performance baseline with negative DIM32 latency result. |
| H3-8D1-ktile-paired-implementation.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8D1: opt-in K-tile Paired Scale reuse | Opt-in K-tile scale reuse implementation/performance report; defaults remained unchanged. |
| H3-8D2-next-k-prefetch-implementation.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8D2: checked next-K prefetch and physical observation | Checked next-K prefetch/physical observation; preserve exact overlap metric and zero-overlap controls. |
| H3-8E-potal-pipeline-full-audit-and-optimization-plan.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8E: PoTal pipeline audit and next optimization plan | Pipeline audit distinguishes late next-K raw prefetch from early scheduler-owned load. |
| H3-8F0-original-gemmini-prefetch-audit.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8F0: pinned Gemmini versus PoTal loop/prefetch architecture | Pinned original-Gemmini/PoTAL architecture audit; preserve external source provenance boundaries. |
| H3-8F1-managed-completion-optimization.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8F1: managed completion bookkeeping reuse | Managed completion bookkeeping reuse; lower work is not itself useful graph speedup. |
| H3-8FG-performance-comparison.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8F/G: matched performance and adoption record | Matched early-prefetch adoption record; small benefit versus F1 did not justify default adoption. |
| H3-8G-gemmini-based-early-prefetch-implementation.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8G: scheduler-owned early LdB prefetch | Scheduler-owned early LdB path and verification boundaries; preserve software versus physical distinction. |
| H3-8H-paired-streaming-implementation.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8H Paired Streaming implementation | Streaming implementation retains negative overlap/performance outcomes; no default promotion. |
| H3-8I-25-checkpoint-evidence.md:1-14 | EXPERIMENT / PROFILING provenance: H3-8I-25 checkpoint: software commits and evidence index | Checkpoint evidence index explicitly says underlying lab /tmp artifacts are not versioned. |

## Option classification ledger

This table assigns every inventoried NPU cache/input option to its responsibility. Defaults, consumers, legal combinations and test registrations are in build-matrix.md; source references here anchor the classification. Recommendations classify organization and preservation, not current hardware availability.

| Option | Category | Disposition / source evidence |
|---|---|---|
| CYCLE_DETAIL | PROFILING | KEEP profiling; cmake/ggml-gemmini-options.cmake:261 |
| CYCLE_SIM | PROFILING + REFERENCE | KEEP separate simulator collection mode; cmake/ggml-gemmini-options.cmake:31 |
| GEMMINI_SW_PATH | HARDWARE + CORE | REMOVE external dependency after api extraction; cmake/ggml-gemmini-geometry.cmake:5 |
| GGML_CPU_CYCLE_LOG | CORE (execution/software policy) | KEEP CPU profiling; ggml/src/ggml-cpu/CMakeLists.txt:1 |
| GGML_GEMMINI | CORE (execution/software policy) | MERGE into GGML_NPU=GEMMINI; ggml/src/CMakeLists.txt:320 |
| GGML_GEMMINI_ACT_METRICS | PROFILING | MERGE alias into ACT_QUANT_METRICS; cmake/ggml-gemmini-options.cmake:16 |
| GGML_GEMMINI_ACT_QUANT_METRICS | PROFILING | KEEP independent profiling metric; cmake/ggml-gemmini-options.cmake:16 |
| GGML_GEMMINI_ACTIVATION_BITS | HARDWARE + CORE | MERGE into NPU_PRECISION; cmake/ggml-gemmini-options.cmake:125 |
| GGML_GEMMINI_ACTIVATION_QUANT | CORE (execution/software policy) | KEEP software algorithm selector; cmake/ggml-gemmini-options.cmake:96 |
| GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE | CORE (execution/software policy) | KEEP runtime policy gate; cmake/ggml-gemmini-options.cmake:309 |
| GGML_GEMMINI_BLOCK_SIZE | CORE (execution/software policy) | KEEP numerical quantization parameter; cmake/ggml-gemmini-options.cmake:118 |
| GGML_GEMMINI_COMPUTE_TYPE | CORE (execution/software policy) | KEEP software/reference algorithm selector; cmake/ggml-gemmini-options.cmake:73 |
| GGML_GEMMINI_CUDA_CPU_EXACT | EXPERIMENT + REFERENCE | KEEP reference/experiment lane; ggml/CMakeLists.txt:155 |
| GGML_GEMMINI_CUDA_RTN | EXPERIMENT + REFERENCE | KEEP reference/experiment lane; ggml/CMakeLists.txt:156 |
| GGML_GEMMINI_CUDA_RTN_ATTENTION | EXPERIMENT + REFERENCE | KEEP reference/experiment lane; ggml/CMakeLists.txt:157 |
| GGML_GEMMINI_DEFAULT_MATMUL_MODE | CORE (execution/software policy) | KEEP invocation policy; cmake/ggml-gemmini-options.cmake:303 |
| GGML_GEMMINI_DEFAULT_RMD_BACKEND | CORE (execution/software policy) | KEEP residual execution policy; cmake/ggml-gemmini-options.cmake:287 |
| GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY | CORE (execution/software policy) | KEEP backpressure policy; cmake/ggml-gemmini-options.cmake:307 |
| GGML_GEMMINI_DEQUANT_FP_TEST | CORE (execution/software policy) | KEEP reference validation; cmake/ggml-gemmini-options.cmake:86 |
| GGML_GEMMINI_DIM | HARDWARE + CORE | MERGE into NPU_D; cmake/ggml-gemmini-options.cmake:195 |
| GGML_GEMMINI_DISABLE_WS_REUSE | CORE (execution/software policy) | KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls; cmake/ggml-gemmini-options.cmake:60 |
| GGML_GEMMINI_ENABLE_A4NKS_ATTENTION | CORE (execution/software policy) | KEEP simulation/reference attention gate; cmake/ggml-gemmini-options.cmake:213 |
| GGML_GEMMINI_ENABLE_OPENMP | CORE (execution/software policy) | KEEP CPU worker implementation switch; cmake/ggml-gemmini-options.cmake:283 |
| GGML_GEMMINI_ENABLE_RMD | CORE (execution/software policy) | KEEP integer residual correction; cmake/ggml-gemmini-options.cmake:286 |
| GGML_GEMMINI_ENABLE_STRIPE_MATMUL | CORE (execution/software policy) | KEEP capability gate; cmake/ggml-gemmini-options.cmake:301 |
| GGML_GEMMINI_ENABLE_STRIPE_PIPELINE | CORE (execution/software policy) | KEEP capability gate; cmake/ggml-gemmini-options.cmake:302 |
| GGML_GEMMINI_EXECUTION_BACKEND | HARDWARE + CORE | MERGE production transport into NPU_TRANSPORT; KEEP IM2P simulator lane; cmake/ggml-gemmini-options.cmake:139 |
| GGML_GEMMINI_EXSIA_DEFAULT_MODE | CORE (execution/software policy) | KEEP CPU staging policy; cmake/ggml-gemmini-options.cmake:384 |
| GGML_GEMMINI_EXSIA_LOCAL_WORKERS | CORE (execution/software policy) | KEEP CPU worker policy; cmake/ggml-gemmini-options.cmake:393 |
| GGML_GEMMINI_EXSIA_PROFILE_SCOPE | PROFILING | KEEP profiling; cmake/ggml-gemmini-options.cmake:267 |
| GGML_GEMMINI_EXSIA_SIGMA | CORE (execution/software policy) | KEEP local adaptation algorithm; cmake/ggml-gemmini-options.cmake:207 |
| GGML_GEMMINI_EXSIA_STRIPE_ANCHOR | CORE (execution/software policy) | KEEP stripe folding algorithm; cmake/ggml-gemmini-options.cmake:208 |
| GGML_GEMMINI_FP32 | EXPERIMENT + REFERENCE | KEEP experiment baseline; cmake/ggml-gemmini-options.cmake:2 |
| GGML_GEMMINI_FULL_C | CORE (execution/software policy) | KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls; cmake/ggml-gemmini-options.cmake:50 |
| GGML_GEMMINI_LOW_D | CORE (execution/software policy) | KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls; cmake/ggml-gemmini-options.cmake:52 |
| GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION | CORE (execution/software policy) | KEEP explicit simulated attention execution gate; cmake/ggml-gemmini-options.cmake:214 |
| GGML_GEMMINI_OPTION | CORE (execution/software policy) | KEEP execution policy; merge hardware selection only after CPU reference is separated; cmake/ggml-gemmini-options.cmake:38 |
| GGML_GEMMINI_PB | EXPERIMENT + REFERENCE | KEEP experiment baseline; cmake/ggml-gemmini-options.cmake:429 |
| GGML_GEMMINI_PB_WEIGHT_FORMAT | EXPERIMENT + REFERENCE | KEEP PB reference format; ggml/src/ggml-gemmini/CMakeLists.txt:166 |
| GGML_GEMMINI_PRINT_TILE | PROFILING | KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls; cmake/ggml-gemmini-options.cmake:36 |
| GGML_GEMMINI_RESIDUAL_METRICS | PROFILING | KEEP independent profiling metric; cmake/ggml-gemmini-options.cmake:16 |
| GGML_GEMMINI_SCALE_METRICS | PROFILING | KEEP independent profiling metric; cmake/ggml-gemmini-options.cmake:16 |
| GGML_GEMMINI_TRANSPOSE_B | CORE (execution/software policy) | KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls; cmake/ggml-gemmini-options.cmake:54 |
| GGML_GEMMINI_WEIGHT_BITS | HARDWARE + CORE | MERGE into NPU_PRECISION; cmake/ggml-gemmini-options.cmake:132 |
| GGML_GEMMINI_WS_DEBUG | PROFILING | KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls; cmake/ggml-gemmini-options.cmake:56 |
| GGML_GEMMINI_WS_LOOP_CYCLE | PROFILING | KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls; cmake/ggml-gemmini-options.cmake:58 |
| GGML_POTAL | CORE (execution/software policy) | MERGE into GGML_NPU=POTAL; ggml/CMakeLists.txt:220 |
| GGML_POTAL_PAIRED_AUTO | CORE (execution/software policy) | KEEP paired execution policy; ggml/src/ggml-potal/CMakeLists.txt:227 |
| GGML_POTAL_W4A4 | HARDWARE + CORE | MERGE into NPU_PRECISION=INT4; ggml/CMakeLists.txt:221 |
| IM2P_DIM | LEGACY alias + REFERENCE | REMOVE candidate alias after caller migration; cmake/ggml-gemmini-options.cmake:186 |
| IM2P_SIM_BUILD_DIR | EXPERIMENT + REFERENCE | KEEP simulator cache isolation; cmake/ggml-gemmini-options.cmake:162 |
| IM2P_SIM_IMPLEMENTATION | EXPERIMENT + REFERENCE | KEEP simulator implementation selector; cmake/ggml-gemmini-options.cmake:164 |
| IM2P_SIM_ROOT | EXPERIMENT + REFERENCE | KEEP simulator input; cmake/ggml-gemmini-options.cmake:161 |
| LOG_CYCLE | PROFILING | KEEP profiling/experiment control; cmake/ggml-gemmini-options.cmake:409 |
| LOG_DEBUG | PROFILING | KEEP profiling/experiment control; cmake/ggml-gemmini-options.cmake:407 |
| LOG_DUMP | PROFILING | KEEP profiling/experiment control; cmake/ggml-gemmini-options.cmake:411 |
| LOG_DUMP_SCALE | PROFILING | KEEP profiling/experiment control; cmake/ggml-gemmini-options.cmake:413 |
| POTAL_API_ONLY | CORE (execution/software policy) | KEEP standalone API build mode; CMakeLists.txt:169 |
| POTAL_ATTN_SWEEP | EXPERIMENT + REFERENCE | KEEP isolated experiment/proof target gate; ggml/src/ggml-potal/CMakeLists.txt:514 |
| POTAL_B5_SHARED_BASELINE | EXPERIMENT + REFERENCE | KEEP isolated experiment/proof target gate; ggml/src/ggml-potal/CMakeLists.txt:425 |
| POTAL_B5_TESTS | EXPERIMENT + REFERENCE | KEEP isolated experiment/proof target gate; ggml/src/ggml-potal/CMakeLists.txt:423 |
| POTAL_E3_TESTS | EXPERIMENT + REFERENCE | KEEP isolated experiment/proof target gate; ggml/src/ggml-potal/CMakeLists.txt:418 |
| POTAL_E4A_TESTS | EXPERIMENT + REFERENCE | KEEP isolated experiment/proof target gate; ggml/src/ggml-potal/CMakeLists.txt:438 |
| POTAL_E4B1_TESTS | EXPERIMENT + REFERENCE | KEEP isolated experiment/proof target gate; ggml/src/ggml-potal/CMakeLists.txt:467 |
| POTAL_E4B2_TESTS | EXPERIMENT + REFERENCE | KEEP isolated experiment/proof target gate; ggml/src/ggml-potal/CMakeLists.txt:111 |
| POTAL_E4C1_TESTS | EXPERIMENT + REFERENCE | KEEP isolated experiment/proof target gate; ggml/src/ggml-potal/CMakeLists.txt:480 |
| POTAL_E4C2_0_TESTS | EXPERIMENT + REFERENCE | KEEP isolated experiment/proof target gate; ggml/src/ggml-potal/CMakeLists.txt:128 |
| POTAL_ENABLE_OPENMP | CORE (execution/software policy) | KEEP separate smoke worker option; ggml/src/ggml-potal/CMakeLists.txt:14 |
| POTAL_EXPECTED_DIM | HARDWARE + CORE | MERGE software expectation into NPU_D; ggml/src/ggml-potal/CMakeLists.txt:12 |
| POTAL_HW_CONFIG | HARDWARE + CORE | KEEP provenance/config assertion; ggml/src/ggml-potal/CMakeLists.txt:13 |
| POTAL_HW_PARAMS | HARDWARE + CORE | MERGE hardware inputs into generated NPU_*; KEEP provenance; ggml/src/ggml-potal/CMakeLists.txt:11 |
| POTAL_PLANNING_DIAGNOSTICS | PROFILING | KEEP profiling; ggml/src/ggml-potal/CMakeLists.txt:228 |
| POTAL_TARGET_DIM | HARDWARE + CORE | KEEP target assertion against NPU_D; cmake/potal-hw-params.cmake:140 |
| POTAL_TARGET_PARAMS | HARDWARE + CORE | KEEP target provenance assertion; cmake/potal-hw-params.cmake:101 |

## Per-file classification ledger

Backend files inherit the category below for all functions unless the mixed responsibilities described above apply. These are proposed organizational labels based on file role, not claims of unused symbols. In particular potal.c must be split across ISA/encode/transport and session execution, and quants/residual code remains software numerical work even when it lives under a hardware-named directory (ggml/src/ggml-potal/potal.c:139-389 and ggml/src/ggml-potal/potal.c:868-940 and ggml/src/ggml-potal/potal.c:1287-1351; ggml/src/ggml-potal/CMakeLists.txt:258-280).

| File | Classification | Rationale / Phase 1 responsibility |
|---|---|---|
| ggml/src/ggml-gemmini-utils/CMakeLists.txt:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/cpu-timing.h:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.h:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/cycle_sim_context.h:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/cycle_sim_log.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/host-timing.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/layer.h:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/layer.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/log.h:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/optrace.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/performance.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/semantic.h:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/semantic.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/trace-context.h:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/include/gemmini/trace-context.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/cycle.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/cycle_reader_aarch64.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/cycle_reader_internal.h:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/cycle_sim_internal.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/cycle_sim_log.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/cycle_sim_records.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/cycle_sim_stages.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/debug.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/log-capi.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/optrace.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/performance.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/semantic.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/trace-context.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini-utils/src/trace-metadata.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/2nd_run_llama.sh:1 | LEGACY (candidate; external use UNVERIFIED) | software processing / configuration contract |
| ggml/src/ggml-gemmini/CMakeLists.txt:1 | CORE build/numerical contract | software processing / configuration contract |
| ggml/src/ggml-gemmini/dump/dump_common.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/dump/dump_common.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/dump/dump_tensor.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/dump/dump_tensor.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/dump/scale/scale_dump_block.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/dump/scale/scale_dump_group_select.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/dump/scale/scale_dump_internal.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/dump/scale/scale_dump_stub.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/dump/scale/scale_dump_types.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/exsia_config.json:1 | LEGACY (candidate; external use UNVERIFIED) | software processing / configuration contract |
| ggml/src/ggml-gemmini/fp32/backend.cmake:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/fp32/backend.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/fp32/cpu/CMakeLists.txt:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/fp32/dump.hpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/fp32/exec.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/fp32/exec.hpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/fp32/rocc-model.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/fp32/sdk.hpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/fp32/tests/attention.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference (test functions) |
| ggml/src/ggml-gemmini/fp32/tests/block.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference (test functions) |
| ggml/src/ggml-gemmini/fp32/tests/golden.hpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference (test functions) |
| ggml/src/ggml-gemmini/fp32/tests/replay.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference (test functions) |
| ggml/src/ggml-gemmini/fp32/tests/rtl-vectors.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference (test functions) |
| ggml/src/ggml-gemmini/fp32/tests/runner.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference (test functions) |
| ggml/src/ggml-gemmini/fp32/tests/test-util.hpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference (test functions) |
| ggml/src/ggml-gemmini/fp32/tools/check-options.sh:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/fp32/tools/offline.sh:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/fp32/tools/rtl-closure.sh:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/ggml-gemmini-args.h:1 | CORE build/numerical contract | software processing / configuration contract |
| ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:1 | CORE execution / GGML integration | software processing: tensor and buffer ownership |
| ggml/src/ggml-gemmini/ggml-gemmini-buffer.hpp:1 | CORE execution / GGML integration | software processing: tensor and buffer ownership |
| ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:1 | CORE build/numerical contract | software processing / configuration contract |
| ggml/src/ggml-gemmini/ggml-gemmini-evaluation-observer.hpp:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-gemmini/ggml-gemmini-geometry.hpp:1 | CORE build/numerical contract | matmul/tiling |
| ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference (IM2P adapter) |
| ggml/src/ggml-gemmini/ggml-gemmini-im2p.hpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference (IM2P adapter) |
| ggml/src/ggml-gemmini/ggml-gemmini-matmul-config.hpp.in:1 | CORE build/numerical contract | software processing / configuration contract |
| ggml/src/ggml-gemmini/ggml-gemmini-matmul.hpp:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/ggml-gemmini-telemetry.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-gemmini/ggml-gemmini.cpp:1 | CORE execution / GGML integration | software processing / GGML backend dispatch |
| ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-gemmini/im2p/route.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference (IM2P adapter) |
| ggml/src/ggml-gemmini/im2p/route.hpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference (IM2P adapter) |
| ggml/src/ggml-gemmini/matmul/attention-stationary.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-gemmini/matmul/attention-stationary.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-gemmini/matmul/attention.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-gemmini/matmul/attention.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-gemmini/matmul/dense.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-gemmini/matmul/dense.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-gemmini/matmul/detail.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-gemmini/matmul/execution.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-gemmini/matmul/execution.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-gemmini/matmul/options.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-gemmini/matmul/options.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-gemmini/matmul/types.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-gemmini/ops.cpp:1 | CORE execution / GGML integration | software processing / GGML backend dispatch |
| ggml/src/ggml-gemmini/ops.hpp:1 | CORE execution / GGML integration | software processing / GGML backend dispatch |
| ggml/src/ggml-gemmini/pb/pb-attention.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/pb/pb-attention.hpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/pb/pb-exec.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/pb/pb-exec.hpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/pb/pb-rocc-model.cpp:1 | EXPERIMENT + REFERENCE (hardware seam where present) | experiment/reference; execution / SDK transport seam |
| ggml/src/ggml-gemmini/potal-sensitivity.cpp:1 | PROFILING | software processing / configuration contract |
| ggml/src/ggml-gemmini/potal-sensitivity.hpp:1 | PROFILING | software processing / configuration contract |
| ggml/src/ggml-gemmini/quants/act/block/block.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/block/block.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/block/types.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/buffer.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/dispatch.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/dispatch.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/exsia-state.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/exsia_float.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/exsia_shift.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/folding.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/local.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/strict-profile.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/exsia/types.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/meta.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/preparation-cache.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/preparation-cache.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/quantize.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/quantize.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/stripe/stripe.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/stripe/types.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/tensor/tensor.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/tensor/types.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/token/token.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/token/token.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/token/types.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/act/types.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/common/dequant.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/common/dequant.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/common/fp16_util.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/common/fp16_util.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/common/hp1_scu.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/common/math.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/common/rounding.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/common/tensor_util.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/common/weight_reader.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/common/weight_reader.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/common/weight_route.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/quants/stripe.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/direct/direct-builder.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/direct/direct-executor.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/direct/direct-profile.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/direct/direct-stage-profile.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/direct/direct-types.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/residual-capture.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-builder.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-compose.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-compose.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-reference.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-reference.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-run-aware.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-run-aware.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/residual/rmd/rmd-types.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-gemmini/run_llama.sh:1 | LEGACY (candidate; external use UNVERIFIED) | software processing / configuration contract |
| ggml/src/ggml-gemmini/tests/gemmini-pb-attention-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-gemmini/tests/gemmini-pb-block-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-gemmini/tests/gemmini-pb-e2e-runner.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-gemmini/tests/gemmini-pb-q8-0-attention-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-gemmini/tests/gemmini-pb-q8-0-block-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-gemmini/tests/pb-five-core-env.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/CMakeLists.txt:1 | CORE build/numerical contract | software processing / configuration contract |
| ggml/src/ggml-potal-w4a4/README.md:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-potal-w4a4/docs/paired-gemm-software-contract.md:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-potal-w4a4/docs/potal-paired-matmul-auto.md:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-potal-w4a4/docs/potal-stripe-api.md:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-potal-w4a4/ggml-potal-buffer.cpp:1 | CORE execution / GGML integration | software processing: tensor and buffer ownership |
| ggml/src/ggml-potal-w4a4/ggml-potal-buffer.hpp:1 | CORE execution / GGML integration | software processing: tensor and buffer ownership |
| ggml/src/ggml-potal-w4a4/ggml-potal.cpp:1 | CORE execution / GGML integration | software processing / GGML backend dispatch |
| ggml/src/ggml-potal-w4a4/include/potal-int4.h:1 | HARDWARE + CORE | api/ISA + api/encode contracts |
| ggml/src/ggml-potal-w4a4/include/potal.h:1 | HARDWARE + CORE | api/ISA + api/encode contracts |
| ggml/src/ggml-potal-w4a4/include/potal_rocc.h:1 | HARDWARE + CORE | api/transport/rocc |
| ggml/src/ggml-potal-w4a4/log.cpp:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-potal-w4a4/log.hpp:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-potal-w4a4/matmul/attention-hw.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/attention-residual.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/attention.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/attention.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/dense.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/dense.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/execution.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/execution.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/options.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/pack.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/pack.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/paired-auto.hpp:1 | CORE execution / GGML integration | matmul/tiling |
| ggml/src/ggml-potal-w4a4/matmul/paired-emulator.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/paired-plan.cpp:1 | CORE execution / GGML integration | matmul/tiling |
| ggml/src/ggml-potal-w4a4/matmul/paired-schedule.cpp:1 | CORE execution / GGML integration | matmul/tiling |
| ggml/src/ggml-potal-w4a4/matmul/paired.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/paired.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/planning-profile.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal-w4a4/matmul/planning-profile.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal-w4a4/matmul/stripe-api.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/stripe-api.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/stripe.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/tiling.cpp:1 | CORE execution / GGML integration | matmul/tiling |
| ggml/src/ggml-potal-w4a4/matmul/tiling.hpp:1 | CORE execution / GGML integration | matmul/tiling |
| ggml/src/ggml-potal-w4a4/matmul/weight_cache.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/matmul/weight_cache.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal-w4a4/ops.cpp:1 | CORE execution / GGML integration | software processing / GGML backend dispatch |
| ggml/src/ggml-potal-w4a4/ops.hpp:1 | CORE execution / GGML integration | software processing / GGML backend dispatch |
| ggml/src/ggml-potal-w4a4/potal.c:1 | HARDWARE + CORE | split: api/ISA, api/encode, api/transport, matmul/execution (see overrides) |
| ggml/src/ggml-potal-w4a4/quants/weight/hp1/hp1.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-potal-w4a4/quants/weight/hp1/hp1.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-potal-w4a4/residual/stripe-residual.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-potal-w4a4/residual/stripe-residual.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/attention-measurement-check.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/e3/CMakeLists.txt:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-api-smoke.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-api-test.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-attention-api-test.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-attention-smoke.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-attention-sweep.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-attention-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-clock-probe.c:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e2e-runner.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e3-backend.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e3-context-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e3-context.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e3-context.h:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e3-kbench.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e3-prototype-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e3-window-diag.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e4b1-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e4b1.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e4b1.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e4b2-c11-test.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e4b2-hardware.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e4b2-rocc.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e4b2-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e4c1-backend.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e4c1-bench.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e4c1-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-e4c2-0-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-full-inference.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-ggml-smoke.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-ggml-unit-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-hp1-smoke.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-hw-timeline-test.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-index-payload-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-index-queue-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-multicore-smoke.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-paired-auto-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-paired-execution-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-paired-index-hw-probe.c:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-paired-inventory.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-paired-plan-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-paired-schedule-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-paired-unit-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-pipeline-bench.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-plan-deps-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-planning-profile-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-resident-model-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-resident-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-resident.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-resident.h:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-residual-merge-bench.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-residual-unit-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-scale-clock.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-scale-path-bench.c:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-scale-window-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-stripe-api-c.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-stripe-api-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-stripe-pipeline-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-stripe-smoke.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-tiling-unit-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-timeline-test.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-trace-test.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-weight-pack-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/potal-weight-prep-bench.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/attention-emulator.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/attention-golden.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/e4b2-emulator.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/e4b2-fixtures.h:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/e4c1-sequential.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/e4c1-sequential.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/gemmini-merge.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/hw-trace-model.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/paired-h2-emulator.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/paired-support.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/potal-emulator.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/stripe-golden.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/stripe-owner.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/tests/support/w4a4-golden.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal-w4a4/trace/cpu-sample.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal-w4a4/trace/hw-window.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal-w4a4/trace/hw-window.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal-w4a4/trace/timeline.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal-w4a4/trace/timeline.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal/CMakeLists.txt:1 | CORE build/numerical contract | software processing / configuration contract |
| ggml/src/ggml-potal/README.md:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-potal/docs/paired-gemm-software-contract.md:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-potal/docs/potal-paired-matmul-auto.md:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-potal/docs/potal-stripe-api.md:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-potal/ggml-potal-buffer.cpp:1 | CORE execution / GGML integration | software processing: tensor and buffer ownership |
| ggml/src/ggml-potal/ggml-potal-buffer.hpp:1 | CORE execution / GGML integration | software processing: tensor and buffer ownership |
| ggml/src/ggml-potal/ggml-potal.cpp:1 | CORE execution / GGML integration | software processing / GGML backend dispatch |
| ggml/src/ggml-potal/include/potal.h:1 | HARDWARE + CORE | api/ISA + api/encode contracts |
| ggml/src/ggml-potal/include/potal_rocc.h:1 | HARDWARE + CORE | api/transport/rocc |
| ggml/src/ggml-potal/log.cpp:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-potal/log.hpp:1 | CORE execution / GGML integration | software processing / configuration contract |
| ggml/src/ggml-potal/matmul/attention-hw.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/attention.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/attention.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/dense.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/dense.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/execution.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/execution.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/options.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/pack.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/pack.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/paired-auto.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/paired-auto.hpp:1 | CORE execution / GGML integration | matmul/tiling |
| ggml/src/ggml-potal/matmul/paired-emulator.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/paired-plan.cpp:1 | CORE execution / GGML integration | matmul/tiling |
| ggml/src/ggml-potal/matmul/paired-schedule.cpp:1 | CORE execution / GGML integration | matmul/tiling |
| ggml/src/ggml-potal/matmul/paired.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/paired.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/planning-profile.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal/matmul/planning-profile.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal/matmul/stripe-api.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/stripe-api.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/stripe.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/stripe.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/tiling.cpp:1 | CORE execution / GGML integration | matmul/tiling |
| ggml/src/ggml-potal/matmul/tiling.hpp:1 | CORE execution / GGML integration | matmul/tiling |
| ggml/src/ggml-potal/matmul/weight_cache.cpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/matmul/weight_cache.hpp:1 | CORE execution / GGML integration | matmul/execution (CPU arithmetic retained) |
| ggml/src/ggml-potal/ops.cpp:1 | CORE execution / GGML integration | software processing / GGML backend dispatch |
| ggml/src/ggml-potal/ops.hpp:1 | CORE execution / GGML integration | software processing / GGML backend dispatch |
| ggml/src/ggml-potal/potal.c:1 | HARDWARE + CORE | split: api/ISA, api/encode, api/transport, matmul/execution (see overrides) |
| ggml/src/ggml-potal/quants/weight/hp1/hp1.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-potal/quants/weight/hp1/hp1.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-potal/residual/stripe-residual.cpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-potal/residual/stripe-residual.hpp:1 | CORE software numerical processing | software processing (SDK residual executor additionally matmul/execution) |
| ggml/src/ggml-potal/tests/e3/CMakeLists.txt:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-api-smoke.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-api-test.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-attention-api-test.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-attention-bench.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-attention-smoke.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-attention-sweep.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-attention-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-clock-probe.c:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e2e-runner.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e3-backend.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e3-context-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e3-context.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e3-context.h:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e3-kbench.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e3-prototype-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e3-window-diag.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e4b1-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e4b1.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e4b1.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e4b2-c11-test.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e4b2-hardware.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e4b2-rocc.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e4b2-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e4c1-backend.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e4c1-bench.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e4c1-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-e4c2-0-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-full-inference.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-ggml-smoke.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-ggml-unit-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-hp1-smoke.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-hw-timeline-test.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-index-queue-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-log-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-multicore-smoke.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-paired-auto-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-paired-execution-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-paired-index-hw-probe.c:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-paired-inventory.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-paired-plan-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-paired-schedule-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-paired-unit-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-pipeline-bench.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-planning-profile-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-resident-model-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-resident-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-resident.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-resident.h:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-residual-merge-bench.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-residual-unit-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-scale-clock.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-scale-path-bench.c:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-scale-window-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-stripe-api-c.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-stripe-api-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-stripe-pipeline-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-stripe-smoke.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-tiling-unit-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-timeline-binary-fixture.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-timeline-test.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-trace-test.c:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-weight-preload-test.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/potal-weight-prep-bench.cpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/attention-emulator.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/attention-golden.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/e4b2-emulator.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/e4b2-fixtures.h:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/e4c1-sequential.cpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/e4c1-sequential.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/gemmini-merge.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/hw-trace-model.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/paired-h2-emulator.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/paired-support.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/potal-emulator.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/stripe-golden.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/support/stripe-owner.hpp:1 | REFERENCE + EXPERIMENT (regression/contract) | experiment/reference (test functions) |
| ggml/src/ggml-potal/tests/timeline-cpu-check.hpp:1 | PROFILING + EXPERIMENT | experiment/reference (test functions) |
| ggml/src/ggml-potal/trace/cpu-sample.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal/trace/hw-window.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal/trace/hw-window.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal/trace/timeline-binary.hpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal/trace/timeline.cpp:1 | PROFILING | profiling |
| ggml/src/ggml-potal/trace/timeline.hpp:1 | PROFILING | profiling |

## Build, regression and analysis files outside backends

The following related files preserve integration checks, target policy and analysis tooling alongside backend code. File labels are recommendations; actual runtime results remain UNVERIFIED.

| File | Category / reason |
|---|---|
| build-arm64-cpu.sh:1 | CORE/EXPERIMENT: explicit build/profile entrypoint; KEEP |
| build-arm64.sh:1 | CORE/EXPERIMENT: explicit build/profile entrypoint; KEEP |
| build-potal-firesim.sh:1 | CORE/EXPERIMENT: explicit build/profile entrypoint; KEEP |
| build-riscv.sh:1 | CORE/EXPERIMENT: explicit build/profile entrypoint; KEEP |
| build-x86.sh:1 | CORE/EXPERIMENT: explicit build/profile entrypoint; KEEP |
| cmake/ggml-gemmini-config.cmake:1 | CORE/HARDWARE: configuration or target contract; KEEP pending generator migration |
| cmake/ggml-gemmini-geometry.cmake:1 | CORE/HARDWARE: configuration or target contract; KEEP pending generator migration |
| cmake/ggml-gemmini-im2p.cmake:1 | CORE/HARDWARE: configuration or target contract; KEEP pending generator migration |
| cmake/ggml-gemmini-options.cmake:1 | CORE/HARDWARE: configuration or target contract; KEEP pending generator migration |
| cmake/potal-hw-params.cmake:1 | CORE/HARDWARE: configuration or target contract; KEEP pending generator migration |
| cmake/riscv-static-release.cmake:1 | CORE/HARDWARE: configuration or target contract; KEEP pending generator migration |
| cmake/riscv-static.cmake:1 | CORE/HARDWARE: configuration or target contract; KEEP pending generator migration |
| cmake/riscv.cmake:1 | CORE/HARDWARE: configuration or target contract; KEEP pending generator migration |
| scripts/cycle-sim-build-info.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/eval/actual_cycle_bundle.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/eval/campaign_cycle.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/eval/cycle_accounting.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/eval/cycle_campaign.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/eval/cycle_campaign_bundle.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/eval/cycle_trace_capture.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/eval/plot_potal_pipeline.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/eval/run_cycle_campaign.sh:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/eval/run_cycle_evaluation.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/eval/run_cycle_trace_capture.sh:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/experiment/build-potal-cuda.sh:1 | EXPERIMENT/REFERENCE: simulator or target workflow tooling; KEEP |
| scripts/experiment/cycle-matrix.json:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/experiment/cycle_matrix.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/im2p-build-options.py:1 | EXPERIMENT/REFERENCE: simulator or target workflow tooling; KEEP |
| scripts/im2p-host-provision.sh:1 | EXPERIMENT/REFERENCE: simulator or target workflow tooling; KEEP |
| scripts/potal-build-baremetal.sh:1 | EXPERIMENT/REFERENCE: simulator or target workflow tooling; KEEP |
| scripts/potal-hw-params-test.sh:1 | EXPERIMENT/REFERENCE: simulator or target workflow tooling; KEEP |
| scripts/potal-timeline/README.md:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/potal-timeline/binary_format.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/potal-timeline/binary_to_csv.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/potal-timeline/breakdown.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/potal-timeline/cpu_parse.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/potal-timeline/overhead_summary.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/potal-timeline/plot_timeline.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/potal-timeline/test_binary_to_csv.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/potal-timeline/test_cpu_parse.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/potal-timeline/test_timeline_parse.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/potal-timeline/timeline_parse.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/utils/cycle_schema.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/utils/cycle_timeline.py:1 | PROFILING: observation/analysis tooling; KEEP |
| scripts/utils/export_potal_csv.py:1 | EXPERIMENT/REFERENCE: simulator or target workflow tooling; KEEP |
| scripts/utils/potal_records.py:1 | EXPERIMENT/REFERENCE: simulator or target workflow tooling; KEEP |
| scripts/utils/potal_rows.py:1 | EXPERIMENT/REFERENCE: simulator or target workflow tooling; KEEP |
| tests/check-cycle-sim-host-stages.py:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/gemmini-parity-output.hpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/gemmini-quant-dispatch-smoke.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/gemmini-quant-dispatch-unsupported.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-cuda-potal-attention.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-cycle-evaluation.py:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-cycle-matrix.py:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-cycle-sim-build-contract.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-cycle-sim-coverage.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-cycle-sim-cpu-smoke.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-cycle-sim-dispatch.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-cycle-sim-reader.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-cycle-timeline.py:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-evaluation-lifecycle.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-act-quant-i32-saturation.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-block-dense.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-block-end-to-end.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-block-quant.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-buffer-artifacts.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-build-contract.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-cpu-cycle-validity-contract.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-cycle-cli.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-cycle-reader-aarch64.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-cycle-sim-dispatch.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-cycle-sim-log.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-direct-metrics-source-contract.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-dump-phase.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-evaluation-metrics.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-evaluation-observer.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-exsia-case.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-exsia-profile-contract.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-exsia-profile.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-exsia-sigma.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-exsia.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-geometry-contract-adversarial.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-geometry-contract.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-host-timing.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-hp1-contract.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-im2p-cycle-overlap.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-im2p-lifecycle.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-im2p-routing.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-log-boundary.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-log-c-boundary.c:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-log-cross-image.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-log.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-matmul-cpu-boundaries.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-matmul-options.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-matmul.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-memory-contract.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-metric-terminal-elision.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-native-dequant-fp.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-off-build.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-optrace-dispatch.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-optrace-transaction.sh:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-optrace.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-production-host-backend.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-production-parity.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-profiled-domain-contract.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-rmd-backend-denylist.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-rmd-backend-gate.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-rmd-bitmap.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-rmd-capture-finish-boundary.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-rmd-capture.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-rmd-executor.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-rmd-im2p-provider.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-rmd-packet.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-rmd-run-aware.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-rmd-telemetry.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-rmd.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-semantic.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-semantic.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-srmd-raw-ws.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-telemetry-clock-gate.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-trace-context.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-weight-formats.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-gemmini-weight-reader-callers.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-ggml-cpu-cycle-sink.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-host-overlap.py:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-im2p-cache-build-scripts.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-potal-csv.py:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-potal-gguf-load.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-potal-sensitivity.cpp:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-potal-target-profile.cmake:1 | REFERENCE: regression/contract fixture; KEEP |
| tests/test-residual-profile.py:1 | REFERENCE: regression/contract fixture; KEEP |

## Phase 1 preservation boundary

Extract pure ISA constants, validators and encoders first; put real RoCC and recording behind one raw command/response boundary. Keep session lifetime, generation ownership, failed-fence retention, tiling, issuer affinity, CPU numerical stages and output commit above that boundary. These boundaries already have distinct source seams, so a self-contained API must not silently absorb or delete the CPU/reference algorithms (ggml/src/ggml-potal/potal.c:139-389 and ggml/src/ggml-potal/potal.c:847-940 and ggml/src/ggml-potal/potal.c:1154-1254; ggml/src/ggml-potal/matmul/execution.cpp:475-505; ggml/src/ggml-potal/matmul/stripe.cpp:1405-1425; ggml/src/ggml-gemmini/matmul/dense.cpp:1222-1290).

Geometry generation must preserve logical precision versus packed byte stride, DIM-dependent partial-sum semantics, scale/index capacities and provenance. Current geometry is duplicated across the RTL adapter, fixed W4 validation and software-only make_potal_geometry constants; Phase 1 should explicitly reconcile these, not merely rename macros (cmake/potal-hw-params.cmake:120-176 and cmake/potal-hw-params.cmake:184-201; ggml/src/ggml-potal-w4a4/CMakeLists.txt:23-38; ggml/src/ggml-gemmini/ggml-gemmini-geometry.hpp:120-150; ggml/src/ggml-potal-w4a4/include/potal-int4.h:8-26). Baseline execution, INT8/INT4 numerics, paired/index/resident reuse, attention, INT64 reconstruction and counter-domain integrity remain must-preserve regardless of how experiment folders are eventually organized.

