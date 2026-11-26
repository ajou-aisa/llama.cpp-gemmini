# NPU build matrix and option inventory (Phase 0)

Snapshot boundary: inventory and line references describe tracked revision 64a072afa5a7a22dec74e94d599ffa4b848c9a20, excluding concurrently added untracked Phase 1 files. Only the three requested audit documents were written by this audit; baseline generation/builds belong to another agent.

Static audit on 2026-11-25 at socmulator-dev / 64a072afa5a7a22dec74e94d599ffa4b848c9a20. No configure, build or test was run by this audit. Defaults below distinguish direct CMake from wrapper policy. A source occurrence proves configuration wiring, not successful compilation or execution. Actual baseline results and external SDK/RTL behavior are UNVERIFIED here. Source references are repository-relative; registrations are static declarations, not ctest output.

## Baseline build results

Measured results (commands, per-configuration build and CTest outcome,
pre-existing failures, and the Phase 1 comparison) are in
[baseline/test-results.md](baseline/test-results.md). The new `GGML_NPU` /
`NPU_*` options are documented in [npu-build.md](npu-build.md).

## Configuration entrypoints and legal combinations

| Contract | Confirmed configuration behavior / recommendation |
|---|---|
| K1: backend ownership | POTAL_API_ONLY returns before ggml/llama and refuses GGML_GEMMINI/GGML_POTAL. Full GGML_POTAL refuses GGML_GEMMINI because both own custom-3; enables testing before backend subdirectories. POTAL hardware/target arguments without either PoTAL mode are errors. CMakeLists.txt:164-189. MERGE backend selection into GGML_NPU while preserving explicit ownership. |
| K2: INT4 selector | GGML_POTAL_W4A4 is consumed only inside GGML_POTAL AND GGML_POTAL_W4A4; W4 alone does not select a backend. API_ONLY always adds INT8 ggml-potal directly. ggml/src/CMakeLists.txt:320-329; CMakeLists.txt:174-179. MERGE precision into NPU_PRECISION, preserve the existing W4 ABI checks. |
| K3: Gemmini route | OPTION accepts CPU/OS/WS syntactically, but OS is rejected; legal route pairs are CPU+HARDWARE, WS+HARDWARE, WS+IM2P_SIM. Widths must match. cmake/ggml-gemmini-options.cmake:139-159. CPU+HARDWARE is a CPU route, not proof of NPU use (ggml/src/ggml-gemmini/matmul/dense.cpp:792-841). |
| K4: Gemmini geometry | Generic integer DIM is 16/32/64; activations/weights 4/8/16; BK 32/64/128. Physical RISC-V non-cycle DIM must match SDK. Host generation rewrites DIM and optionally elem_t. cmake/ggml-gemmini-options.cmake:118-137 and cmake/ggml-gemmini-options.cmake:186-204; cmake/ggml-gemmini-geometry.cmake:25-76. MERGE into generated NPU_PRECISION/NPU_D, avoid silent physical geometry substitution. |
| K5: SDK | GGML_GEMMINI requires explicit GEMMINI_SW_PATH and a root gemmini_params.h; headers may be found at root or include/ for gemmini.h, but parameter lookup specifically uses root. cmake/ggml-gemmini-geometry.cmake:7-28. REMOVE external SDK dependency only after Phase 1 vendors encoder/transport; external macro behavior currently UNVERIFIED. |
| K6: PoTAL header | Absolute POTAL_HW_PARAMS and POTAL_EXPECTED_DIM required; DIM16/32/64 allowed in API, DIM16/32 only in backend. Header checks custom-3, signed i8/i32, 32-bit addresses, power-of-two banks/rows, DMA multiple of DIM, SPAD/ACC capacity, 256 scale slots and index_entries >= DIM. Target header SHA and target DIM optionally checked; matching FireSim config names also constrain DIM. cmake/potal-hw-params.cmake:83-176; ggml/src/ggml-potal/CMakeLists.txt:180-188. MERGE values into common generated NPU_* contract, KEEP provenance and target assertions. |
| K7: PoTAL precision | INT8 requires A8W8 and BK32; INT4 requires A4W4 and exact DIM32, BANK_NUM4/BANK_ROWS2048/ACC_ROWS512/MAX_BYTES64/index64/packing ABI1/SP row16 and named W4 header. ggml/src/ggml-potal/CMakeLists.txt:194-199; ggml/src/ggml-potal-w4a4/CMakeLists.txt:18-39 and ggml/src/ggml-potal-w4a4/CMakeLists.txt:223-229. |
| K8: PoTAL execution | Static registration only; RISC-V requires Linux; IM2P_SIM and CYCLE_SIM forbidden. FULL compensation requires CPU residual. STRIPE_PIPELINE requires EXSIA and WS residual if RMD ON. Capacity defaults only 1 or 2. Runtime overrides are refused per node when incompatible. CMakeLists.txt:197-202; ggml/src/ggml-potal/CMakeLists.txt:183-245. |
| K9: stripe capabilities | Pipeline requires stripe matmul; stripe defaults require enabled capability. Modes FULL/STRIPE_PIPELINE, capacity positive uint32. cmake/ggml-gemmini-options.cmake:300-370. Preserve CPU issuer and backpressure semantics when renaming defaults. |
| K10: ExSIA workers | Internal modes SEQUENTIAL/LOCAL_PARALLEL/LOCAL_FOLDING_PIPELINE; local workers 3 or 4; parallel defaults require Gemmini OpenMP. Backend checks OpenMP task-depend support; PoTAL links CXX OpenMP separately from smoke-test C OpenMP. cmake/ggml-gemmini-options.cmake:383-404; ggml/src/ggml-gemmini/CMakeLists.txt:376-408; ggml/src/ggml-potal/CMakeLists.txt:145-159 and ggml/src/ggml-potal/CMakeLists.txt:361-376. Explicit hart-role std::threads remain a separate path (ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:995-1012). |
| K11: profiling | CYCLE_DETAIL requires LOG_CYCLE; non-OFF EXSIA profile scope requires CYCLE_DETAIL. Metric knobs must be 0/1; ACT_METRICS enables ACT_QUANT_METRICS. cmake/ggml-gemmini-options.cmake:15-30 and cmake/ggml-gemmini-options.cmake:261-281 and cmake/ggml-gemmini-options.cmake:407-426. Keep profiling opt-in and distinct from execution choice. |
| K12: IM2P | Native non-cross IM2P_SIM requires GGML_GEMMINI, valid source root, atomically selected private generation, verified manifest and matched archives. Implementation LEGACY_BSV or GEMMINI_HP1; default depends on CYCLE_SIM. CYCLE_SIM needs frontend sources, matched A4/A8 and BK32, and uses generated target contract; WS compiles CPU-functional frontend sources instead of external RTL archives. cmake/ggml-gemmini-options.cmake:161-184; cmake/ggml-gemmini-im2p.cmake:1-59 and cmake/ggml-gemmini-im2p.cmake:71-161; ggml/src/ggml-gemmini/CMakeLists.txt:271-305. KEEP separate simulation/reference lane; do not equate CYCLE_SIM with real RTL execution. |
| K13: simulated attention | ENABLE_A4NKS_ATTENTION requires Gemmini INT matched A4/A8, DIM32/BK32/sigma2/CYCLE_SIM0. OFFLOAD additionally needs WS+IM2P_SIM+GEMMINI_HP1. This is separate from automatic full-PoTAL llama attention wiring. cmake/ggml-gemmini-options.cmake:213-234; src/CMakeLists.txt:46-50. |
| K14: FP32 | Requires Gemmini, DIM8; PB/PoTAL/A4NKS/OpenMP/cycle/CUDA paths OFF. Uses replacement backend source list and SDK overlay. cmake/ggml-gemmini-options.cmake:2-13 and cmake/ggml-gemmini-options.cmake:198-201; ggml/src/ggml-gemmini/fp32/backend.cmake:1-17. KEEP opt-in experiment baseline, separate from COMPUTE_TYPE=FLOAT CPU reference. |
| K15: PB | INT8/BK32/WS/HARDWARE/BLOCK, RMD OFF, GGML_POTAL OFF. Weight format Q8_0 or MXINT8; native RoCC model added on non-RISC-V. cmake/ggml-gemmini-options.cmake:429-443; ggml/src/ggml-gemmini/CMakeLists.txt:158-208. KEEP opt-in partial-sum baseline. |
| K16: CUDA | RTN attention requires CPU-exact. RTN and CPU-exact exclude normal CUDA/Metal/DL combinations as specified; CPU-exact requires matched A4/A8 INT, CPU option, HARDWARE selector, BLOCK/EXSIA, no cycle, and no block RMD. ggml/CMakeLists.txt:155-180. KEEP reference/experiment comparators; independent of NPU_TRANSPORT. |
| K17: planning | POTAL_PLANNING_DIAGNOSTICS requires GNU C++ Linux linker --wrap; optional profile library and wrapped new/delete accounting. ggml/src/ggml-potal/CMakeLists.txt:228-236 and ggml/src/ggml-potal/CMakeLists.txt:320-335. KEEP profiling. |

## Common parameter generator migration

Proposed GGML_NPU should replace backend ownership choices; NPU_PRECISION should replace matched bit-width selectors and W4 fork selection; NPU_D should replace GGML_GEMMINI_DIM/IM2P_DIM/expected software DIM duplication. Hardware memory values map BANK_NUM -> NPU_SPAD_BANKS, BANK_ROWS -> NPU_SPAD_BANK_ROWS, ACC_ROWS -> NPU_ACC_ROWS, MAX_BYTES -> NPU_DMA_MAX_BYTES, POTAL_INDEX_ENTRIES -> NPU_INDEX_ENTRIES and POTAL_SCALE_ENTRIES -> NPU_SCALE_ENTRIES. These names are Phase 1 proposals; the current adapter exposes POTAL_HW_* (cmake/potal-hw-params.cmake:184-201). A separately named NPU_ACC_BANKS has no independently parsed current header field: derivation/RTL meaning is UNVERIFIED and must be specified before generation (cmake/potal-hw-params.cmake:120-123). NPU_TRANSPORT should select raw RoCC/recording without erasing CPU-reference, simulator or numeric-algorithm selection; PoTAL already separates command sink and real issue (ggml/src/ggml-potal/include/potal.h:584-596; ggml/src/ggml-potal/potal.c:1154-1192).

## Option inventory

Each item includes direct default/definition, static use sites, relevant compile branches/constraints, runtime relationship and recommendation. All references in the use ledger are from the specified build files and read backend/source/test corpus; they include exports and assertions, not only runtime consumers. A missing literal consumer is explicitly UNVERIFIED when the selected external SDK or generated-name translation owns it. Wrapper defaults are inventoried separately below. K1-K17 apply in addition to each item's stated domain; no unlisted combination is claimed tested.

### CYCLE_DETAIL

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:261.
- Behavior / legal combinations: 0/1; requires LOG_CYCLE. Contracts: K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ops.cpp:99; ggml/src/ggml-gemmini/ops.cpp:109; ggml/src/ggml-gemmini/ops.cpp:136; ggml/src/ggml-potal/tests/potal-e4c1-bench.cpp:647; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-potal-w4a4/tests/potal-e4c1-bench.cpp:647; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:24; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:30; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:307; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:325; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:391; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:417; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:605; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:683; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1038; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1227; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1249; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1417; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1432; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:294; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2512; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2566; ggml/src/ggml-gemmini/residual/residual-capture.hpp:8; ggml/src/ggml-gemmini/residual/residual-capture.hpp:115; ggml/src/ggml-gemmini/residual/residual-capture.hpp:118; ggml/src/ggml-gemmini/residual/residual-capture.hpp:138; ggml/src/ggml-gemmini/residual/residual-capture.hpp:177; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp:214; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:17; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:88; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:349; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:394; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:409; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:429; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:469; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:536; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:587; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:609; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:653; ggml/src/ggml-gemmini/residual/direct/direct-profile.hpp:78; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:21; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:612; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:625; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:666; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:35; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:43; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:47; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:126; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:164; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:198; ggml/src/ggml-gemmini/matmul/execution.cpp:198; ggml/src/ggml-gemmini/matmul/execution.cpp:217; ggml/src/ggml-gemmini/matmul/execution.cpp:1948; ggml/src/ggml-gemmini/matmul/types.hpp:213; ggml/src/ggml-gemmini-utils/src/cycle.cpp:335; ggml/src/ggml-gemmini-utils/src/cycle.cpp:356; ggml/src/ggml-gemmini-utils/src/cycle.cpp:423; ggml/src/ggml-gemmini-utils/src/cycle.cpp:442; ggml/src/ggml-gemmini-utils/src/cycle.cpp:515; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1732; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1964; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:122; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:204; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:356; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:365; ggml/src/ggml-gemmini-utils/src/log-capi.cpp:30; ggml/src/ggml-gemmini-utils/src/log-capi.cpp:259; ggml/src/ggml-gemmini-utils/src/log-capi.cpp:278; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:5; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:37; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:51; ggml/src/ggml-gemmini-utils/src/cycle_reader_internal.h:45; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:87; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:111; ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp:60-61; ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp:64
- Runtime override: No runtime enabling beyond compiled support confirmed.
- Recommendation: KEEP profiling (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:261-264; cmake/ggml-gemmini-options.cmake:422-423; cmake/ggml-gemmini-options.cmake:425-426; cmake/ggml-gemmini-config.cmake:65; cmake/ggml-gemmini-config.cmake:87; ggml/src/ggml-potal/CMakeLists.txt:339; scripts/im2p-build-options.py:97-98; scripts/im2p-build-options.py:102-103; ggml/src/ggml-potal-w4a4/CMakeLists.txt:378; ggml/src/ggml-gemmini/CMakeLists.txt:49; ggml/src/ggml-gemmini/CMakeLists.txt:58
- Source/macro consumers: ggml/src/ggml-gemmini/ops.cpp:99; ggml/src/ggml-gemmini/ops.cpp:109; ggml/src/ggml-gemmini/ops.cpp:136; ggml/src/ggml-gemmini/ops.cpp:1229; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:24; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:30; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:307; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:325; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:391; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:417; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:605; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:683; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1038; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1227; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1249; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1417; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1432; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:294; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2512; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2566; ggml/src/ggml-gemmini/residual/residual-capture.hpp:8; ggml/src/ggml-gemmini/residual/residual-capture.hpp:115; ggml/src/ggml-gemmini/residual/residual-capture.hpp:118; ggml/src/ggml-gemmini/residual/residual-capture.hpp:138; ggml/src/ggml-gemmini/residual/residual-capture.hpp:177; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp:214; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:17; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:44; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:88; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:349; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:361; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:370; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:386; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:394; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:409; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:429; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:469; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:536; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:543; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:587; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:609; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:649; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:653; ggml/src/ggml-gemmini/residual/direct/direct-profile.hpp:78; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:21; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:612; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:625; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:666; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:32; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:34-35; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:43-44; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:47-48; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:126; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:164; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:198; ggml/src/ggml-gemmini/matmul/execution.cpp:198; ggml/src/ggml-gemmini/matmul/execution.cpp:217; ggml/src/ggml-gemmini/matmul/execution.cpp:1948; ggml/src/ggml-gemmini/matmul/types.hpp:213; ggml/src/ggml-gemmini-utils/src/cycle.cpp:335; ggml/src/ggml-gemmini-utils/src/cycle.cpp:356; ggml/src/ggml-gemmini-utils/src/cycle.cpp:423; ggml/src/ggml-gemmini-utils/src/cycle.cpp:442; ggml/src/ggml-gemmini-utils/src/cycle.cpp:515; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1732; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1964; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:122; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:204; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:356; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:365; ggml/src/ggml-gemmini-utils/src/log-capi.cpp:30; ggml/src/ggml-gemmini-utils/src/log-capi.cpp:259; ggml/src/ggml-gemmini-utils/src/log-capi.cpp:278; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:5; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:37; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:51; ggml/src/ggml-gemmini-utils/src/cycle_reader_internal.h:45; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:87; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:111; ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp:60-61; ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp:64
- Existing test references: tests/CMakeLists.txt:23; tests/CMakeLists.txt:32; tests/CMakeLists.txt:343; tests/CMakeLists.txt:368; tests/CMakeLists.txt:370; tests/CMakeLists.txt:398; tests/CMakeLists.txt:405; tests/CMakeLists.txt:622; tests/CMakeLists.txt:676; tests/CMakeLists.txt:700; tests/CMakeLists.txt:730; tests/CMakeLists.txt:753; tests/CMakeLists.txt:769; tests/CMakeLists.txt:771; tests/CMakeLists.txt:787; tests/CMakeLists.txt:819; tests/CMakeLists.txt:893; tests/CMakeLists.txt:1081; tests/CMakeLists.txt:1124; tests/CMakeLists.txt:1159; tests/CMakeLists.txt:1178; tests/CMakeLists.txt:1363; tests/CMakeLists.txt:1413; tests/CMakeLists.txt:1516; ggml/src/ggml-potal/tests/potal-e4c1-bench.cpp:647-648; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:98; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-potal-w4a4/tests/potal-e4c1-bench.cpp:647-648; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:98; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:892; tests/test-gemmini-im2p-cycle-overlap.cmake:26; tests/test-gemmini-rmd-capture-finish-boundary.cmake:126; tests/test-potal-target-profile.cmake:28; tests/test-gemmini-matmul-cpu-boundaries.cmake:193; tests/test-gemmini-matmul-cpu-boundaries.cmake:250; tests/test-gemmini-matmul-cpu-boundaries.cmake:262; tests/test-gemmini-matmul-cpu-boundaries.cmake:338

### CYCLE_SIM

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:31.
- Behavior / legal combinations: 0/1; CPU-functional producer lane, not hardware flag. Contracts: K8,K12.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini.cpp:33; ggml/src/ggml-gemmini/ggml-gemmini.cpp:160; ggml/src/ggml-gemmini/ggml-gemmini.cpp:215; ggml/src/ggml-gemmini/ggml-gemmini.cpp:236; ggml/src/ggml-potal/ggml-potal.cpp:365; ggml/src/ggml-gemmini/ops.cpp:30; ggml/src/ggml-gemmini/ops.cpp:1026; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-potal-w4a4/ggml-potal.cpp:365; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:365; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:10; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:253; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:305; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:316; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:322; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:370; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:466; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:491; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:619; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:628; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:653; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:676; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:685; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1044; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1157; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1261; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1285; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1384; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1406; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1507; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1531; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1701; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1854; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2188; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2351; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2479; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2492; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2529; ggml/src/ggml-gemmini/ggml-gemmini-args.h:24; ggml/src/ggml-gemmini/ggml-gemmini-args.h:331; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:14; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:503; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:528; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:617; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:665; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:681; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:696; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:741; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:843; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:907; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:977; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:1008; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:1019; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp:16; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp:143; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp:184; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:16; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:839; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:911; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:921; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:945; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:954; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1008; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1015; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1032; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1062; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1099; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1457; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1499; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1521; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1613; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1626; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1641; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1646; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1769; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1791; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1805; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1832; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1873; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1927; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:2001; ggml/src/ggml-gemmini/im2p/route.cpp:38; ggml/src/ggml-gemmini/quants/act/preparation-cache.cpp:92; ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp:14; ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp:58; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:18; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:331; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:590; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1072; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1145; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1218; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1523; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1578; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1638; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1966; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1997; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:2012; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:19; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:233; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:267; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:305; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:335; ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp:28; ggml/src/ggml-gemmini/matmul/types.hpp:198; ggml/src/ggml-gemmini-utils/src/cycle.cpp:7; ggml/src/ggml-gemmini-utils/src/cycle.cpp:163; ggml/src/ggml-gemmini-utils/src/cycle.cpp:169; ggml/src/ggml-gemmini-utils/src/cycle.cpp:180; ggml/src/ggml-gemmini-utils/src/cycle.cpp:253-254; ggml/src/ggml-gemmini-utils/src/cycle.cpp:347; ggml/src/ggml-gemmini-utils/src/optrace.cpp:166; ggml/src/ggml-gemmini-utils/src/cycle_sim_internal.hpp:3; ggml/src/ggml-gemmini-utils/src/cycle_sim_records.cpp:2; ggml/src/ggml-gemmini-utils/src/cycle_sim_stages.cpp:2; ggml/src/ggml-gemmini-utils/src/cycle_sim_log.cpp:2; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_sim_log.hpp:3-4; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_sim_log.hpp:8; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:7; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:12-13; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:41; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:62; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:77; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:83; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:92
- Runtime override: Build-time; simulator telemetry environment listed below.
- Recommendation: KEEP separate simulator collection mode (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-geometry.cmake:38; cmake/ggml-gemmini-options.cmake:9; cmake/ggml-gemmini-options.cmake:31-34; cmake/ggml-gemmini-options.cmake:169; cmake/ggml-gemmini-options.cmake:226-227; cmake/ggml-gemmini-config.cmake:41; cmake/ggml-gemmini-config.cmake:59; CMakeLists.txt:201-202; ggml/CMakeLists.txt:171; ggml/CMakeLists.txt:175; ggml/src/ggml-gemmini-utils/CMakeLists.txt:3-4; ggml/src/ggml-gemmini-utils/CMakeLists.txt:33; ggml/src/ggml-gemmini-utils/CMakeLists.txt:61; ggml/src/ggml-gemmini-utils/CMakeLists.txt:63; ggml/src/ggml-gemmini-utils/CMakeLists.txt:81; cmake/ggml-gemmini-im2p.cmake:1; cmake/ggml-gemmini-im2p.cmake:7; cmake/ggml-gemmini-im2p.cmake:13; cmake/ggml-gemmini-im2p.cmake:25; cmake/ggml-gemmini-im2p.cmake:37; scripts/im2p-build-options.py:120; scripts/im2p-build-options.py:152; scripts/im2p-build-options.py:184; ggml/src/CMakeLists.txt:4-5; ggml/src/CMakeLists.txt:294-295; ggml/src/ggml-gemmini/CMakeLists.txt:271; src/CMakeLists.txt:39-40; ggml/src/ggml-cpu/CMakeLists.txt:45; ggml/src/ggml-cpu/CMakeLists.txt:49
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini.cpp:33; ggml/src/ggml-gemmini/ggml-gemmini.cpp:160; ggml/src/ggml-gemmini/ggml-gemmini.cpp:215; ggml/src/ggml-gemmini/ggml-gemmini.cpp:236; ggml/src/ggml-potal/ggml-potal.cpp:365; ggml/src/ggml-gemmini/ops.cpp:30; ggml/src/ggml-gemmini/ops.cpp:1026; ggml/src/ggml-potal-w4a4/ggml-potal.cpp:365; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:365; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:10; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:253; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:305; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:316; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:322; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:370; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:466; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:491; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:619; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:628; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:653; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:676; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:685; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1044; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1157; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1261; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1285; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1384; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1406; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1507; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1531; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1701; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1854; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2188; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2351; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2479; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2492; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2529; ggml/src/ggml-gemmini/ggml-gemmini-args.h:24; ggml/src/ggml-gemmini/ggml-gemmini-args.h:331; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:14; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:503; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:528; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:617; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:665; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:681; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:696; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:741; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:843; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:907; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:977; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:1008; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:1019; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp:16; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp:143; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp:184; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:16; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:839; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:911; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:921; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:945; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:954; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1008; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1015; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1032; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1062; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1099; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1457; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1499; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1521; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1613; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1626; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1641; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1646; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1769; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1791; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1805; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1832; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1873; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1927; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:2001; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:2157; ggml/src/ggml-gemmini/im2p/route.cpp:38; ggml/src/ggml-gemmini/quants/act/preparation-cache.cpp:92; ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp:14; ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp:58; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:18; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:331; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:590; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1072; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1145; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1218; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1523; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1578; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1638; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1966; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1997; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:2012; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:19; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:32; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:233; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:267; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:305; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:335; ggml/src/ggml-gemmini/quants/act/exsia/exsia.hpp:28; ggml/src/ggml-gemmini/matmul/types.hpp:198; ggml/src/ggml-gemmini-utils/src/cycle.cpp:7; ggml/src/ggml-gemmini-utils/src/cycle.cpp:163; ggml/src/ggml-gemmini-utils/src/cycle.cpp:169; ggml/src/ggml-gemmini-utils/src/cycle.cpp:180; ggml/src/ggml-gemmini-utils/src/cycle.cpp:253-254; ggml/src/ggml-gemmini-utils/src/cycle.cpp:347; ggml/src/ggml-gemmini-utils/src/optrace.cpp:166; ggml/src/ggml-gemmini-utils/src/cycle_sim_internal.hpp:3; ggml/src/ggml-gemmini-utils/src/cycle_sim_records.cpp:2; ggml/src/ggml-gemmini-utils/src/cycle_sim_stages.cpp:2; ggml/src/ggml-gemmini-utils/src/cycle_sim_log.cpp:2; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_sim_log.hpp:3-4; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_sim_log.hpp:8; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:7; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:12-13; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:41; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:62; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:77; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:83; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:92
- Existing test references: tests/CMakeLists.txt:6; tests/CMakeLists.txt:9; tests/CMakeLists.txt:13; tests/CMakeLists.txt:23; tests/CMakeLists.txt:31; tests/CMakeLists.txt:41; tests/CMakeLists.txt:45; tests/CMakeLists.txt:53; tests/CMakeLists.txt:92; tests/CMakeLists.txt:97; tests/CMakeLists.txt:354; tests/CMakeLists.txt:435; tests/CMakeLists.txt:446; tests/CMakeLists.txt:552; tests/CMakeLists.txt:729; tests/CMakeLists.txt:780; tests/CMakeLists.txt:813; tests/CMakeLists.txt:1077; tests/CMakeLists.txt:1413; tests/CMakeLists.txt:1428; tests/CMakeLists.txt:1441; tests/CMakeLists.txt:1458; tests/CMakeLists.txt:1464; tests/CMakeLists.txt:1607; tests/CMakeLists.txt:1684; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:98; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:98; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:892; tests/test-potal-target-profile.cmake:28

### GEMMINI_SW_PATH

- Direct default / definition: unset; explicit SDK path; cmake/ggml-gemmini-geometry.cmake:5.
- Behavior / legal combinations: Explicit header path, no default; currently required for Gemmini and not required for PoTAL. Current macro consumers outside tree UNVERIFIED. Contracts: K5.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Shell resolver configurable() excludes bare GEMMINI_* environment names; explicit -DGEMMINI_SW_PATH still enters CLI map and CMake arguments (scripts/im2p-build-options.py:21-24 and scripts/im2p-build-options.py:49-77 and scripts/im2p-build-options.py:178-188).
- Recommendation: REMOVE external dependency after api extraction (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-geometry.cmake:5-9; cmake/ggml-gemmini-geometry.cmake:11-13; cmake/ggml-gemmini-geometry.cmake:15; cmake/ggml-gemmini-geometry.cmake:21; cmake/ggml-gemmini-geometry.cmake:25; ggml/src/ggml-gemmini-utils/CMakeLists.txt:89; cmake/ggml-gemmini-im2p.cmake:235-236; scripts/im2p-host-provision.sh:24; scripts/im2p-host-provision.sh:31; ggml/src/ggml-gemmini/CMakeLists.txt:5-6; ggml/src/ggml-gemmini/CMakeLists.txt:8-10; ggml/src/ggml-gemmini/CMakeLists.txt:241-242; ggml/src/ggml-gemmini/CMakeLists.txt:257-259; ggml/src/ggml-gemmini/CMakeLists.txt:300; ggml/src/ggml-gemmini/CMakeLists.txt:360; ggml/src/ggml-gemmini/fp32/backend.cmake:2; ggml/src/ggml-gemmini/fp32/backend.cmake:13
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/CMakeLists.txt:65; tests/CMakeLists.txt:658-659; tests/CMakeLists.txt:906; tests/CMakeLists.txt:964-965; tests/CMakeLists.txt:983-984; tests/CMakeLists.txt:1037-1038; tests/CMakeLists.txt:1050; tests/CMakeLists.txt:1061; tests/CMakeLists.txt:1086-1087; tests/CMakeLists.txt:1107-1108; tests/CMakeLists.txt:1128; tests/CMakeLists.txt:1163; tests/CMakeLists.txt:1182; tests/CMakeLists.txt:1206-1208; tests/CMakeLists.txt:1225-1226; tests/CMakeLists.txt:1253-1254; tests/CMakeLists.txt:1287; tests/CMakeLists.txt:1325-1326; tests/CMakeLists.txt:1375-1376; tests/CMakeLists.txt:1408-1409; tests/CMakeLists.txt:1473; tests/CMakeLists.txt:1565; tests/CMakeLists.txt:1622; tests/test-gemmini-build-contract.cmake:1148

### GGML_CPU_CYCLE_LOG

- Direct default / definition: OFF; ggml/src/ggml-cpu/CMakeLists.txt:1.
- Behavior / legal combinations: OFF direct CMake; wrappers derive from effective LOG_CYCLE unless explicitly overridden. Contracts: K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Build-time, derived in scripts/im2p-build-options.py:78-80.
- Recommendation: KEEP CPU profiling (preserve the behavior and constraints above).
- Build use locations: scripts/im2p-build-options.py:28; scripts/im2p-build-options.py:78-80; ggml/src/ggml-cpu/CMakeLists.txt:1; ggml/src/ggml-cpu/CMakeLists.txt:47; ggml/src/ggml-cpu/CMakeLists.txt:49
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/CMakeLists.txt:34; tests/CMakeLists.txt:93; tests/CMakeLists.txt:441; tests/CMakeLists.txt:1694; tests/test-potal-target-profile.cmake:30

### GGML_GEMMINI

- Direct default / definition: unset (false); no root cache option declaration; ggml/src/CMakeLists.txt:320.
- Behavior / legal combinations: Selects backend through ggml_add_backend(GEMMINI); no ordinary top-level option declaration was found in the audited root/ggml CMake files. Unset evaluates false. The OFF declaration in fp32/cpu is a separate nested CPU replay project. Contracts: K1,K3-K5.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: No runtime override of compiled backend.
- Recommendation: MERGE into GGML_NPU=GEMMINI (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-geometry.cmake:4; cmake/ggml-gemmini-geometry.cmake:7; cmake/ggml-gemmini-geometry.cmake:20; cmake/ggml-gemmini-geometry.cmake:22; cmake/ggml-gemmini-options.cmake:4; cmake/ggml-gemmini-options.cmake:219-220; cmake/ggml-gemmini-config.cmake:77; CMakeLists.txt:171-172; CMakeLists.txt:181-182; ggml/CMakeLists.txt:162; ggml/CMakeLists.txt:168-169; ggml/src/ggml-gemmini-utils/CMakeLists.txt:42; ggml/src/ggml-gemmini-utils/CMakeLists.txt:47; cmake/ggml-gemmini-im2p.cmake:28; cmake/ggml-gemmini-im2p.cmake:55-56; build-x86.sh:11; build-riscv.sh:14; scripts/im2p-build-options.py:32; scripts/im2p-build-options.py:86-87; scripts/im2p-build-options.py:113; scripts/im2p-build-options.py:115; scripts/im2p-build-options.py:117; ggml/src/ggml-gemmini/fp32/cpu/CMakeLists.txt:8
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/CMakeLists.txt:378; tests/test-gemmini-build-contract.cmake:923; tests/test-gemmini-off-build.cmake:7; tests/test-gemmini-off-build.cmake:12; tests/test-gemmini-off-build.cmake:15; tests/test-gemmini-off-build.cmake:17; tests/test-im2p-cache-build-scripts.cmake:38; tests/test-im2p-cache-build-scripts.cmake:91; tests/test-im2p-cache-build-scripts.cmake:120; tests/test-im2p-cache-build-scripts.cmake:200; tests/test-im2p-cache-build-scripts.cmake:225; tests/test-im2p-cache-build-scripts.cmake:360; tests/test-im2p-cache-build-scripts.cmake:404; tests/test-im2p-cache-build-scripts.cmake:477; tests/test-im2p-cache-build-scripts.cmake:547; tests/test-potal-target-profile.cmake:20

### GGML_GEMMINI_ACT_METRICS

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:16.
- Behavior / legal combinations: Validated 0/1 and exported through runtime/utils definitions. ACT_METRICS implies ACT_QUANT_METRICS. Contracts: K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:10-11
- Runtime override: Build-time gate; paths/output controls are independent.
- Recommendation: MERGE alias into ACT_QUANT_METRICS (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:28; cmake/ggml-gemmini-config.cmake:55; cmake/ggml-gemmini-config.cmake:81; scripts/im2p-build-options.py:83; scripts/im2p-build-options.py:85
- Source/macro consumers: ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:10-11
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### GGML_GEMMINI_ACT_QUANT_METRICS

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:16.
- Behavior / legal combinations: Validated 0/1 and exported through runtime/utils definitions. ACT_METRICS implies ACT_QUANT_METRICS. Contracts: K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ops.cpp:286; ggml/src/ggml-gemmini/ops.cpp:1020; ggml/src/ggml-gemmini/ops.cpp:1311; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1381; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1771; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1967; ggml/src/ggml-gemmini/ggml-gemmini-args.h:325; ggml/src/ggml-gemmini/quants/act/preparation-cache.cpp:92; ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp:45; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:456; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:586; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:741; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1726; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:2048; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:302; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:499; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:699; ggml/src/ggml-gemmini/quants/act/exsia/exsia-state.hpp:195; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:204; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:245; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:118; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:231; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:334; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:404; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:416; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:451; ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:9; ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:11; ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:13
- Runtime override: Build-time gate; paths/output controls are independent.
- Recommendation: KEEP independent profiling metric (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:29; cmake/ggml-gemmini-config.cmake:55; cmake/ggml-gemmini-config.cmake:57; cmake/ggml-gemmini-config.cmake:81; cmake/ggml-gemmini-config.cmake:83; ggml/src/ggml-gemmini-utils/CMakeLists.txt:34; scripts/im2p-build-options.py:84-85
- Source/macro consumers: ggml/src/ggml-gemmini/ops.cpp:286; ggml/src/ggml-gemmini/ops.cpp:1020; ggml/src/ggml-gemmini/ops.cpp:1311; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1381; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1771; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1967; ggml/src/ggml-gemmini/ggml-gemmini-args.h:325; ggml/src/ggml-gemmini/quants/act/preparation-cache.cpp:92; ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp:45; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:456; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:586; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:741; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1726; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:2048; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:302; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:499; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:699; ggml/src/ggml-gemmini/quants/act/exsia/exsia-state.hpp:195; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:204; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:245; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:118; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:231; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:241; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:334; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:404; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:416; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:451; ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:9; ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:11; ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:13
- Existing test references: tests/CMakeLists.txt:318; tests/CMakeLists.txt:1042; tests/CMakeLists.txt:1054; tests/CMakeLists.txt:1724

### GGML_GEMMINI_ACTIVATION_BITS

- Direct default / definition: "8"; cmake/ggml-gemmini-options.cmake:125.
- Behavior / legal combinations: Matched width 4/8/16; physical backend restrictions apply. Contracts: K4,K7,K12-K16.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ops.cpp:1508; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:42-43; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:51; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:190; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:197; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:202; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:217; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:219; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1266
- Runtime override: No process env precision override in backend.
- Recommendation: MERGE into NPU_PRECISION (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-geometry.cmake:47; cmake/ggml-gemmini-options.cmake:125-128; cmake/ggml-gemmini-options.cmake:130; cmake/ggml-gemmini-options.cmake:156; cmake/ggml-gemmini-options.cmake:222; cmake/ggml-gemmini-options.cmake:433; cmake/ggml-gemmini-config.cmake:37; cmake/ggml-gemmini-config.cmake:72; ggml/src/ggml-potal/CMakeLists.txt:194-196; ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:3; ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:27; ggml/CMakeLists.txt:173-174; ggml/src/ggml-gemmini-utils/CMakeLists.txt:36; ggml/src/ggml-gemmini-utils/CMakeLists.txt:92; cmake/ggml-gemmini-im2p.cmake:9; cmake/ggml-gemmini-im2p.cmake:20; cmake/ggml-gemmini-im2p.cmake:72; cmake/ggml-gemmini-im2p.cmake:199; cmake/ggml-gemmini-im2p.cmake:257; cmake/ggml-gemmini-im2p.cmake:301; cmake/ggml-gemmini-im2p.cmake:303; cmake/ggml-gemmini-im2p.cmake:309; scripts/im2p-build-options.py:145; ggml/src/ggml-potal-w4a4/CMakeLists.txt:223-225; ggml/src/ggml-gemmini/CMakeLists.txt:283-284; ggml/src/ggml-gemmini/CMakeLists.txt:353
- Source/macro consumers: ggml/src/ggml-gemmini/ops.cpp:81; ggml/src/ggml-gemmini/ops.cpp:216; ggml/src/ggml-gemmini/ops.cpp:219; ggml/src/ggml-gemmini/ops.cpp:501; ggml/src/ggml-gemmini/ops.cpp:1202; ggml/src/ggml-gemmini/ops.cpp:1207; ggml/src/ggml-gemmini/ops.cpp:1258; ggml/src/ggml-gemmini/ops.cpp:1260; ggml/src/ggml-gemmini/ops.cpp:1366; ggml/src/ggml-gemmini/ops.cpp:1371; ggml/src/ggml-gemmini/ops.cpp:1440; ggml/src/ggml-gemmini/ops.cpp:1445; ggml/src/ggml-gemmini/ops.cpp:1508; ggml/src/ggml-gemmini/ops.cpp:2352; ggml/src/ggml-potal/matmul/dense.cpp:305; ggml/src/ggml-potal/matmul/options.hpp:44; ggml/src/ggml-potal-w4a4/matmul/dense.cpp:314; ggml/src/ggml-potal-w4a4/matmul/options.hpp:44; ggml/src/ggml-gemmini/potal-sensitivity.cpp:167; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:42-43; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:51; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:128-130; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:134; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:136; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:141; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:143; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:145; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1189; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1442; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2539; ggml/src/ggml-gemmini/ggml-gemmini-args.h:61; ggml/src/ggml-gemmini/residual/residual-capture.hpp:71; ggml/src/ggml-gemmini/residual/residual-capture.hpp:79; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:406; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:128; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:727; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:39; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:182-183; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:190; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:197; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:202; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:217; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:219; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:673; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:699; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:805; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1266; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1597; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:2015; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:2170; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:2280; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:2324; ggml/src/ggml-gemmini/im2p/route.cpp:184; ggml/src/ggml-gemmini/quants/act/preparation-cache.cpp:108; ggml/src/ggml-gemmini/quants/act/buffer.hpp:136; ggml/src/ggml-gemmini/quants/act/buffer.hpp:141; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:502; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:372; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:395; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:418; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:577; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:601; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:623; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:62; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:86; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:126; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:311; ggml/src/ggml-gemmini/matmul/attention-stationary.cpp:19; ggml/src/ggml-gemmini/matmul/execution.cpp:1710; ggml/src/ggml-gemmini/matmul/attention.cpp:61; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:119
- Existing test references: ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:3; ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:27; tests/CMakeLists.txt:80; tests/CMakeLists.txt:522; tests/CMakeLists.txt:595; tests/CMakeLists.txt:616; tests/CMakeLists.txt:672; tests/CMakeLists.txt:695; tests/CMakeLists.txt:732; tests/CMakeLists.txt:781; tests/CMakeLists.txt:814; tests/CMakeLists.txt:886; tests/CMakeLists.txt:950; tests/CMakeLists.txt:977; tests/CMakeLists.txt:1031; tests/CMakeLists.txt:1043; tests/CMakeLists.txt:1078; tests/CMakeLists.txt:1099; tests/CMakeLists.txt:1120; tests/CMakeLists.txt:1157; tests/CMakeLists.txt:1176; tests/CMakeLists.txt:1239; tests/CMakeLists.txt:1310; tests/CMakeLists.txt:1364; tests/CMakeLists.txt:1396; tests/CMakeLists.txt:1513; tests/CMakeLists.txt:1552; tests/CMakeLists.txt:1615; tests/CMakeLists.txt:1749; ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:798; ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:805; tests/test-gemmini-build-contract.cmake:1155; tests/test-gemmini-build-contract.cmake:1213; tests/test-gemmini-build-contract.cmake:1215; tests/test-im2p-cache-build-scripts.cmake:203; tests/test-im2p-cache-build-scripts.cmake:251; tests/test-im2p-cache-build-scripts.cmake:273; tests/test-im2p-cache-build-scripts.cmake:330; tests/test-potal-target-profile.cmake:27

### GGML_GEMMINI_ACTIVATION_QUANT

- Direct default / definition: "EXSIA"; cmake/ggml-gemmini-options.cmake:96.
- Behavior / legal combinations: EXSIA/TENSOR/TOKEN/BLOCK/STRIPE; MXINT has format-driven block handling. Contracts: K7-K10,K13,K15,K16.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:28-29; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:90; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:92; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:94; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:96; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:98; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:110; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:112; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:114; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:116; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:118
- Runtime override: Build-time algorithm, ExSIA recompute/anchor diagnostics listed below.
- Recommendation: KEEP software algorithm selector (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:96-98; cmake/ggml-gemmini-options.cmake:104; cmake/ggml-gemmini-config.cmake:33; cmake/ggml-gemmini-config.cmake:68; ggml/src/ggml-potal/CMakeLists.txt:192; ggml/src/ggml-potal/CMakeLists.txt:208; ggml/src/ggml-potal/CMakeLists.txt:212; ggml/src/ggml-potal/CMakeLists.txt:550; ggml/CMakeLists.txt:177; ggml/CMakeLists.txt:179; build-riscv.sh:23; ggml/src/ggml-potal-w4a4/CMakeLists.txt:221; ggml/src/ggml-potal-w4a4/CMakeLists.txt:237; ggml/src/ggml-potal-w4a4/CMakeLists.txt:241; ggml/src/ggml-potal-w4a4/CMakeLists.txt:598
- Source/macro consumers: ggml/src/ggml-potal/matmul/stripe.cpp:1594; ggml/src/ggml-potal/matmul/options.hpp:8; ggml/src/ggml-potal/matmul/options.hpp:124; ggml/src/ggml-potal/matmul/dense.hpp:11; ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:1616; ggml/src/ggml-potal-w4a4/matmul/options.hpp:8; ggml/src/ggml-potal-w4a4/matmul/options.hpp:121; ggml/src/ggml-potal-w4a4/matmul/dense.hpp:11; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:28-29; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:90; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:92; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:94; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:96; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:98; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:101; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:110; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:112; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:114; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:116; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:118
- Existing test references: tests/CMakeLists.txt:597; tests/CMakeLists.txt:618; tests/CMakeLists.txt:1312; tests/CMakeLists.txt:1342; tests/CMakeLists.txt:1366; tests/CMakeLists.txt:1554; tests/CMakeLists.txt:1746; tests/CMakeLists.txt:1748; ggml/src/ggml-potal/tests/potal-stripe-smoke.cpp:18; ggml/src/ggml-potal/tests/potal-residual-merge-bench.cpp:122; ggml/src/ggml-potal/tests/potal-paired-execution-test.cpp:4024; ggml/src/ggml-potal/tests/potal-tiling-unit-test.cpp:10; ggml/src/ggml-potal/tests/potal-tiling-unit-test.cpp:783; ggml/src/ggml-potal/tests/potal-stripe-pipeline-test.cpp:51; ggml/src/ggml-potal/tests/potal-stripe-pipeline-test.cpp:1737; ggml/src/ggml-potal-w4a4/tests/potal-stripe-smoke.cpp:18; ggml/src/ggml-potal-w4a4/tests/potal-residual-merge-bench.cpp:122; ggml/src/ggml-potal-w4a4/tests/potal-paired-execution-test.cpp:4024; ggml/src/ggml-potal-w4a4/tests/potal-tiling-unit-test.cpp:10; ggml/src/ggml-potal-w4a4/tests/potal-tiling-unit-test.cpp:699; ggml/src/ggml-potal-w4a4/tests/potal-stripe-pipeline-test.cpp:51; ggml/src/ggml-potal-w4a4/tests/potal-stripe-pipeline-test.cpp:1769; tests/test-gemmini-build-contract.cmake:191; tests/test-im2p-cache-build-scripts.cmake:202; tests/test-potal-target-profile.cmake:22

### GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE

- Direct default / definition: ON; cmake/ggml-gemmini-options.cmake:309.
- Behavior / legal combinations: Controls common GEMMINI_* option resolution; invalid values fail rather than selecting an arbitrary route. Contracts: K9.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: GEMMINI_MATMUL_MODE, GEMMINI_RMD_BACKEND, GEMMINI_STRIPE_JOB_CAPACITY (ggml/src/ggml-gemmini/matmul/options.cpp:19-44).
- Recommendation: KEEP runtime policy gate (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:309; cmake/ggml-gemmini-options.cmake:332; cmake/ggml-gemmini-options.cmake:376; cmake/ggml-gemmini-config.cmake:17; ggml/src/ggml-gemmini-utils/CMakeLists.txt:47
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: ggml/src/ggml-potal/tests/potal-stripe-smoke.cpp:17; ggml/src/ggml-potal-w4a4/tests/potal-stripe-smoke.cpp:17

### GGML_GEMMINI_BLOCK_SIZE

- Direct default / definition: "32"; cmake/ggml-gemmini-options.cmake:118.
- Behavior / legal combinations: 32/64/128 generic, PoTAL and special lanes constrain BK32. This is a quantization block, not NPU_D. Contracts: K4,K7,K12-K16.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:32-33; ggml/src/ggml-gemmini/quants/act/types.hpp:7-8; ggml/src/ggml-gemmini/quants/act/types.hpp:12
- Runtime override: No runtime override confirmed.
- Recommendation: KEEP numerical quantization parameter (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:118-121; cmake/ggml-gemmini-options.cmake:123; cmake/ggml-gemmini-options.cmake:224; cmake/ggml-gemmini-options.cmake:435; cmake/ggml-gemmini-config.cmake:13; cmake/ggml-gemmini-config.cmake:34; ggml/src/ggml-potal/CMakeLists.txt:198-199; ggml/src/ggml-potal/CMakeLists.txt:343; cmake/ggml-gemmini-im2p.cmake:10; cmake/ggml-gemmini-im2p.cmake:114; cmake/ggml-gemmini-im2p.cmake:256; scripts/im2p-build-options.py:148; ggml/src/ggml-potal-w4a4/CMakeLists.txt:227-228; ggml/src/ggml-potal-w4a4/CMakeLists.txt:382; ggml/src/ggml-gemmini/CMakeLists.txt:320
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:32-33; ggml/src/ggml-gemmini/ggml-gemmini-args.h:187; ggml/src/ggml-gemmini/ggml-gemmini-args.h:255; ggml/src/ggml-gemmini/quants/act/types.hpp:7-8; ggml/src/ggml-gemmini/quants/act/types.hpp:12
- Existing test references: tests/CMakeLists.txt:2; tests/test-gemmini-build-contract.cmake:195; tests/test-gemmini-build-contract.cmake:704; tests/test-potal-target-profile.cmake:28

### GGML_GEMMINI_COMPUTE_TYPE

- Direct default / definition: "INT"; cmake/ggml-gemmini-options.cmake:73.
- Behavior / legal combinations: INT=0/FLOAT=1; FLOAT is not the separate GGML_GEMMINI_FP32 WS implementation. Contracts: K3,K13,K15,K16.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ops.cpp:1387; ggml/src/ggml-gemmini/ops.cpp:1508; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:20-21; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:81; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:83
- Runtime override: Compile-time selection.
- Recommendation: KEEP software/reference algorithm selector (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:73-75; cmake/ggml-gemmini-options.cmake:78; cmake/ggml-gemmini-options.cmake:88; cmake/ggml-gemmini-config.cmake:31; cmake/ggml-gemmini-config.cmake:66; ggml/CMakeLists.txt:172; scripts/im2p-build-options.py:150
- Source/macro consumers: ggml/src/ggml-gemmini/ops.cpp:1387; ggml/src/ggml-gemmini/ops.cpp:1508; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:20-21; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:78; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:81; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:83; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:86
- Existing test references: tests/CMakeLists.txt:379; tests/CMakeLists.txt:1747

### GGML_GEMMINI_CUDA_CPU_EXACT

- Direct default / definition: OFF; ggml/CMakeLists.txt:155.
- Behavior / legal combinations: Compile CUDA helper library and link cudart/cublas, with numerical compiler flags; CUDA and device execution UNVERIFIED in this audit. Contracts: K16.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini.cpp:22; ggml/src/ggml-gemmini/ggml-gemmini.cpp:103; ggml/src/ggml-gemmini/ggml-gemmini.cpp:182; ggml/src/ggml-gemmini/ggml-gemmini.cpp:202; ggml/src/ggml-gemmini/ops.cpp:71; ggml/src/ggml-gemmini/ops.cpp:1009; ggml/src/ggml-gemmini/ops.cpp:2292; ggml/src/ggml-gemmini/matmul/dense.cpp:6; ggml/src/ggml-gemmini/matmul/dense.cpp:254; ggml/src/ggml-gemmini/matmul/dense.cpp:261; ggml/src/ggml-gemmini/matmul/dense.cpp:271; ggml/src/ggml-gemmini/matmul/dense.cpp:421; ggml/src/ggml-gemmini/matmul/dense.cpp:459; ggml/src/ggml-gemmini/matmul/dense.cpp:479; ggml/src/ggml-gemmini/matmul/dense.cpp:823; ggml/src/ggml-gemmini/matmul/dense.cpp:837; ggml/src/ggml-gemmini/potal-sensitivity.cpp:17; ggml/src/ggml-gemmini/potal-sensitivity.cpp:147; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:5; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:84; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:3; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:9; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:263; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:304; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:327; ggml/src/ggml-gemmini/matmul/attention-stationary.cpp:158; ggml/src/ggml-gemmini/matmul/attention-stationary.hpp:10; ggml/src/ggml-gemmini/matmul/attention-stationary.hpp:38; ggml/src/ggml-gemmini/matmul/attention.cpp:16; ggml/src/ggml-gemmini/matmul/attention.cpp:376; ggml/src/ggml-gemmini/matmul/attention.cpp:433; ggml/src/ggml-gemmini/matmul/attention.cpp:461; ggml/src/ggml-gemmini/matmul/attention.cpp:563
- Runtime override: GPU helper runtime enablement is distinct from NPU hardware transport.
- Recommendation: KEEP reference/experiment lane (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:10; ggml/CMakeLists.txt:155; ggml/CMakeLists.txt:158-159; ggml/CMakeLists.txt:162; ggml/CMakeLists.txt:167; ggml/src/ggml-gemmini/CMakeLists.txt:140; ggml/src/ggml-gemmini/CMakeLists.txt:154
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini.cpp:22; ggml/src/ggml-gemmini/ggml-gemmini.cpp:103; ggml/src/ggml-gemmini/ggml-gemmini.cpp:182; ggml/src/ggml-gemmini/ggml-gemmini.cpp:202; ggml/src/ggml-gemmini/ops.cpp:71; ggml/src/ggml-gemmini/ops.cpp:1009; ggml/src/ggml-gemmini/ops.cpp:2292; ggml/src/ggml-gemmini/matmul/dense.cpp:6; ggml/src/ggml-gemmini/matmul/dense.cpp:254; ggml/src/ggml-gemmini/matmul/dense.cpp:261; ggml/src/ggml-gemmini/matmul/dense.cpp:271; ggml/src/ggml-gemmini/matmul/dense.cpp:421; ggml/src/ggml-gemmini/matmul/dense.cpp:459; ggml/src/ggml-gemmini/matmul/dense.cpp:479; ggml/src/ggml-gemmini/matmul/dense.cpp:823; ggml/src/ggml-gemmini/matmul/dense.cpp:837; ggml/src/ggml-gemmini/potal-sensitivity.cpp:17; ggml/src/ggml-gemmini/potal-sensitivity.cpp:147; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:5; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:84; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:3; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:60; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:9; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:263; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:304; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:327; ggml/src/ggml-gemmini/matmul/attention-stationary.cpp:158; ggml/src/ggml-gemmini/matmul/attention-stationary.hpp:10; ggml/src/ggml-gemmini/matmul/attention-stationary.hpp:38; ggml/src/ggml-gemmini/matmul/attention.cpp:16; ggml/src/ggml-gemmini/matmul/attention.cpp:376; ggml/src/ggml-gemmini/matmul/attention.cpp:433; ggml/src/ggml-gemmini/matmul/attention.cpp:461; ggml/src/ggml-gemmini/matmul/attention.cpp:563
- Existing test references: tests/CMakeLists.txt:1740

### GGML_GEMMINI_CUDA_RTN

- Direct default / definition: OFF; ggml/CMakeLists.txt:156.
- Behavior / legal combinations: Compile CUDA helper library and link cudart/cublas, with numerical compiler flags; CUDA and device execution UNVERIFIED in this audit. Contracts: K16.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini.cpp:70; ggml/src/ggml-gemmini/ggml-gemmini.cpp:193; ggml/src/ggml-gemmini/ops.cpp:68; ggml/src/ggml-gemmini/ops.cpp:2288; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:8; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:87
- Runtime override: GPU helper runtime enablement is distinct from NPU hardware transport.
- Recommendation: KEEP reference/experiment lane (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:10; ggml/CMakeLists.txt:156; ggml/CMakeLists.txt:161; ggml/src/ggml-gemmini/CMakeLists.txt:125; ggml/src/ggml-gemmini/CMakeLists.txt:134-135
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini.cpp:70; ggml/src/ggml-gemmini/ggml-gemmini.cpp:193; ggml/src/ggml-gemmini/ops.cpp:68; ggml/src/ggml-gemmini/ops.cpp:2288; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:8; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:87
- Existing test references: tests/CMakeLists.txt:1760

### GGML_GEMMINI_CUDA_RTN_ATTENTION

- Direct default / definition: OFF; ggml/CMakeLists.txt:157.
- Behavior / legal combinations: Compile CUDA helper library and link cudart/cublas, with numerical compiler flags; CUDA and device execution UNVERIFIED in this audit. Contracts: K16.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:8; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:87; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:7; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:18; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:27; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:35; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:41; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:47
- Runtime override: GPU helper runtime enablement is distinct from NPU hardware transport.
- Recommendation: KEEP reference/experiment lane (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:10; ggml/CMakeLists.txt:157-158; ggml/src/ggml-gemmini/CMakeLists.txt:125; ggml/src/ggml-gemmini/CMakeLists.txt:137
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:8; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:87; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:7; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:18; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:27; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:35; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:41; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:47
- Existing test references: tests/CMakeLists.txt:1760

### GGML_GEMMINI_DEFAULT_MATMUL_MODE

- Direct default / definition: "FULL"; cmake/ggml-gemmini-options.cmake:303.
- Behavior / legal combinations: FULL or STRIPE_PIPELINE. Contracts: K8,K9.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: GEMMINI_MATMUL_MODE if override enabled.
- Recommendation: KEEP invocation policy (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:303; cmake/ggml-gemmini-options.cmake:305; cmake/ggml-gemmini-options.cmake:343; cmake/ggml-gemmini-options.cmake:347; cmake/ggml-gemmini-options.cmake:352; cmake/ggml-gemmini-options.cmake:356; cmake/ggml-gemmini-config.cmake:10; ggml/src/ggml-potal/CMakeLists.txt:209; ggml/src/ggml-potal/CMakeLists.txt:211; ggml/src/ggml-potal/CMakeLists.txt:221; ggml/src/ggml-gemmini-utils/CMakeLists.txt:93; ggml/src/ggml-potal-w4a4/CMakeLists.txt:238; ggml/src/ggml-potal-w4a4/CMakeLists.txt:240; ggml/src/ggml-potal-w4a4/CMakeLists.txt:250; ggml/src/ggml-potal-w4a4/CMakeLists.txt:261
- Source/macro consumers: ggml/src/ggml-potal/matmul/options.hpp:14; ggml/src/ggml-potal/matmul/stripe.hpp:3; ggml/src/ggml-potal-w4a4/matmul/options.hpp:14; ggml/src/ggml-potal-w4a4/matmul/stripe.hpp:3
- Existing test references: ggml/src/ggml-potal/tests/potal-stripe-smoke.cpp:14; ggml/src/ggml-potal-w4a4/tests/potal-stripe-smoke.cpp:14; tests/test-gemmini-build-contract.cmake:190; tests/test-gemmini-build-contract.cmake:1040; tests/test-potal-target-profile.cmake:24

### GGML_GEMMINI_DEFAULT_RMD_BACKEND

- Direct default / definition: "CPU"; cmake/ggml-gemmini-options.cmake:287.
- Behavior / legal combinations: CPU or WS; WS means PoTAL residual in its pipeline, not linking Gemmini WS executor. Contracts: K8,K9.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: GEMMINI_RMD_BACKEND if override enabled.
- Recommendation: KEEP residual execution policy (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:287; cmake/ggml-gemmini-options.cmake:289-291; cmake/ggml-gemmini-options.cmake:293; cmake/ggml-gemmini-options.cmake:295; cmake/ggml-gemmini-config.cmake:12; ggml/src/ggml-potal/CMakeLists.txt:214; ggml/src/ggml-potal/CMakeLists.txt:216; ggml/src/ggml-potal/CMakeLists.txt:218; ggml/src/ggml-potal/CMakeLists.txt:220; ggml/src/ggml-gemmini-utils/CMakeLists.txt:48; ggml/src/ggml-gemmini-utils/CMakeLists.txt:95; build-riscv.sh:24; ggml/src/ggml-potal-w4a4/CMakeLists.txt:243; ggml/src/ggml-potal-w4a4/CMakeLists.txt:245; ggml/src/ggml-potal-w4a4/CMakeLists.txt:247; ggml/src/ggml-potal-w4a4/CMakeLists.txt:249; ggml/src/ggml-potal-w4a4/CMakeLists.txt:262
- Source/macro consumers: ggml/src/ggml-potal/matmul/options.hpp:12; ggml/src/ggml-potal-w4a4/matmul/options.hpp:12
- Existing test references: tests/test-gemmini-build-contract.cmake:194; tests/test-gemmini-build-contract.cmake:1052; tests/test-gemmini-off-build.cmake:22; tests/test-im2p-cache-build-scripts.cmake:486; tests/test-potal-target-profile.cmake:23

### GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY

- Direct default / definition: "2"; cmake/ggml-gemmini-options.cmake:307.
- Behavior / legal combinations: Positive uint32 generic, only 1/2 in PoTAL default. Contracts: K8,K9.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: GEMMINI_STRIPE_JOB_CAPACITY if override enabled.
- Recommendation: KEEP backpressure policy (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:307; cmake/ggml-gemmini-options.cmake:361; cmake/ggml-gemmini-config.cmake:16; ggml/src/ggml-potal/CMakeLists.txt:243-245; ggml/src/ggml-potal-w4a4/CMakeLists.txt:281-283
- Source/macro consumers: ggml/src/ggml-potal/matmul/options.hpp:16; ggml/src/ggml-potal-w4a4/matmul/options.hpp:16; ggml/src/ggml-gemmini/ggml-gemmini-matmul-config.hpp.in:17
- Existing test references: tests/test-potal-target-profile.cmake:25; tests/test-potal-target-profile.cmake:64; tests/test-potal-target-profile.cmake:78

### GGML_GEMMINI_DEQUANT_FP_TEST

- Direct default / definition: OFF; cmake/ggml-gemmini-options.cmake:86.
- Behavior / legal combinations: Requires COMPUTE_TYPE=FLOAT; activation dequantization validation, not FP32 NPU execution. Contracts: K16.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:24-25
- Runtime override: Build-time only.
- Recommendation: KEEP reference validation (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:86-88; cmake/ggml-gemmini-options.cmake:90; cmake/ggml-gemmini-config.cmake:32; cmake/ggml-gemmini-config.cmake:67; ggml/CMakeLists.txt:178; scripts/im2p-build-options.py:32
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:24-25; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:122
- Existing test references: tests/CMakeLists.txt:380; tests/CMakeLists.txt:564

### GGML_GEMMINI_DIM

- Direct default / definition: "16"; cmake/ggml-gemmini-options.cmake:195.
- Behavior / legal combinations: Direct default 16; FP32 needs 8; INT4 fork fixed 32. No claim that all syntactically accepted dimensions execute every route. Contracts: K4,K6,K7,K14.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini-args.h:52
- Runtime override: IM2P_DIM is build alias, not runtime override.
- Recommendation: MERGE into NPU_D (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-geometry.cmake:39; cmake/ggml-gemmini-geometry.cmake:41; cmake/ggml-gemmini-geometry.cmake:45; cmake/potal-hw-params.cmake:137-138; cmake/potal-hw-params.cmake:226; cmake/ggml-gemmini-options.cmake:186-188; cmake/ggml-gemmini-options.cmake:190-191; cmake/ggml-gemmini-options.cmake:193; cmake/ggml-gemmini-options.cmake:195; cmake/ggml-gemmini-options.cmake:197; cmake/ggml-gemmini-options.cmake:199-200; cmake/ggml-gemmini-options.cmake:202; cmake/ggml-gemmini-options.cmake:204; cmake/ggml-gemmini-options.cmake:223; cmake/ggml-gemmini-config.cmake:7; cmake/ggml-gemmini-config.cmake:39; ggml/src/ggml-potal/CMakeLists.txt:346; ggml/src/ggml-gemmini-utils/CMakeLists.txt:38; ggml/src/ggml-gemmini-utils/CMakeLists.txt:92; cmake/ggml-gemmini-im2p.cmake:20; cmake/ggml-gemmini-im2p.cmake:72; cmake/ggml-gemmini-im2p.cmake:204; cmake/ggml-gemmini-im2p.cmake:329; cmake/ggml-gemmini-im2p.cmake:331; build-x86.sh:18; build-arm64-cpu.sh:11; build-riscv.sh:34; build-arm64.sh:11; scripts/im2p-build-options.py:67-69; scripts/im2p-build-options.py:71; scripts/im2p-build-options.py:147; ggml/src/ggml-potal-w4a4/CMakeLists.txt:385; ggml/src/ggml-gemmini/CMakeLists.txt:282; ggml/src/ggml-gemmini/CMakeLists.txt:286; ggml/src/ggml-gemmini/CMakeLists.txt:352
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini-args.h:52-53; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:120; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:491-493
- Existing test references: tests/CMakeLists.txt:82; tests/CMakeLists.txt:1315; tests/CMakeLists.txt:1348; tests/CMakeLists.txt:1368; tests/CMakeLists.txt:1750; tests/test-gemmini-build-contract.cmake:926; tests/test-gemmini-build-contract.cmake:973; tests/test-gemmini-build-contract.cmake:985; tests/test-gemmini-build-contract.cmake:1159; tests/test-im2p-cache-build-scripts.cmake:251; tests/test-im2p-cache-build-scripts.cmake:272; tests/test-im2p-cache-build-scripts.cmake:330; tests/test-im2p-cache-build-scripts.cmake:360; tests/test-potal-target-profile.cmake:32

### GGML_GEMMINI_DISABLE_WS_REUSE

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:60.
- Behavior / legal combinations: Validated 0/1, translated to SDK macro in cmake/ggml-gemmini-config.cmake:22-30. In-tree source use and exports listed below; external SDK effect UNVERIFIED. Contracts: K3,K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: No direct runtime getenv of this cache option confirmed.
- Recommendation: KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:60-61; cmake/ggml-gemmini-options.cmake:254; cmake/ggml-gemmini-config.cmake:30
- Source/macro consumers: ggml/src/ggml-potal/matmul/tiling.cpp:395; ggml/src/ggml-potal-w4a4/matmul/tiling.cpp:420
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### GGML_GEMMINI_ENABLE_A4NKS_ATTENTION

- Direct default / definition: OFF; cmake/ggml-gemmini-options.cmake:213.
- Behavior / legal combinations: Separate from automatic full-PoTAL attention compile marker; requires the constrained Gemmini INT profile. Contracts: K13.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/matmul/attention-stationary.cpp:15; ggml/src/ggml-gemmini/matmul/attention.cpp:56; src/llama-graph.cpp:10; src/llama-graph.cpp:1200; src/llama-graph.cpp:1271; src/llama-graph.cpp:1278; src/llama-graph.cpp:1351; src/llama-graph.cpp:1354
- Runtime override: Compile-time; scheduler/query policy runtime controls listed below.
- Recommendation: KEEP simulation/reference attention gate (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:5; cmake/ggml-gemmini-options.cmake:213; cmake/ggml-gemmini-options.cmake:215-216; cmake/ggml-gemmini-options.cmake:218; cmake/ggml-gemmini-config.cmake:14; ggml/CMakeLists.txt:163; build-x86.sh:19; build-arm64-cpu.sh:12; build-riscv.sh:35; build-arm64.sh:12; scripts/im2p-build-options.py:33; scripts/im2p-build-options.py:109; scripts/im2p-build-options.py:112; src/CMakeLists.txt:46-47
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini-matmul-config.hpp.in:4; ggml/src/ggml-gemmini/matmul/attention-stationary.cpp:15; ggml/src/ggml-gemmini/matmul/attention.cpp:56; src/llama-graph.cpp:10; src/llama-graph.cpp:1200; src/llama-graph.cpp:1271; src/llama-graph.cpp:1278; src/llama-graph.cpp:1351; src/llama-graph.cpp:1354
- Existing test references: tests/CMakeLists.txt:1518; tests/test-im2p-cache-build-scripts.cmake:249; tests/test-im2p-cache-build-scripts.cmake:259; tests/test-im2p-cache-build-scripts.cmake:272; tests/test-im2p-cache-build-scripts.cmake:327; tests/test-im2p-cache-build-scripts.cmake:341; tests/test-im2p-cache-build-scripts.cmake:478; tests/test-potal-target-profile.cmake:30

### GGML_GEMMINI_ENABLE_OPENMP

- Direct default / definition: OFF; cmake/ggml-gemmini-options.cmake:283.
- Behavior / legal combinations: Requires CXX OpenMP and task depend capability on Gemmini. Contracts: K10.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: OpenMP environment separate from compiled support.
- Recommendation: KEEP CPU worker implementation switch (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:9; cmake/ggml-gemmini-options.cmake:283; cmake/ggml-gemmini-options.cmake:400-401; ggml/src/ggml-potal/CMakeLists.txt:362; ggml/src/ggml-potal/CMakeLists.txt:374; ggml/src/ggml-potal-w4a4/CMakeLists.txt:401; ggml/src/ggml-potal-w4a4/CMakeLists.txt:414; ggml/src/ggml-gemmini/CMakeLists.txt:13-14; ggml/src/ggml-gemmini/CMakeLists.txt:376; ggml/src/ggml-gemmini/CMakeLists.txt:403; ggml/src/ggml-gemmini/CMakeLists.txt:408
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/CMakeLists.txt:703; tests/CMakeLists.txt:796; tests/CMakeLists.txt:829; tests/CMakeLists.txt:896

### GGML_GEMMINI_ENABLE_RMD

- Direct default / definition: ON; cmake/ggml-gemmini-options.cmake:286.
- Behavior / legal combinations: OFF is compensation ablation, not another compensation algorithm. Contracts: K8,K9,K15.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/matmul/dense.cpp:1414; ggml/src/ggml-potal/matmul/dense.cpp:437; ggml/src/ggml-potal-w4a4/matmul/dense.cpp:439; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:38-39; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:807; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:201; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:219; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:241; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:255; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:299; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:333; ggml/src/ggml-gemmini/quants/act/block/block.cpp:56; ggml/src/ggml-gemmini/quants/act/block/block.cpp:67; ggml/src/ggml-gemmini/quants/act/block/block.cpp:75; ggml/src/ggml-gemmini/quants/act/block/block.cpp:117; ggml/src/ggml-gemmini/quants/act/block/block.cpp:164; ggml/src/ggml-gemmini/quants/act/block/block.cpp:178; ggml/src/ggml-gemmini/quants/act/block/block.cpp:187; ggml/src/ggml-gemmini/quants/act/block/block.cpp:281; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:206; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:500; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:163; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:259; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:308; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:344; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:191; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:218; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:231; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:267; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:294; ggml/src/ggml-gemmini/quants/act/token/token.cpp:188; ggml/src/ggml-gemmini/quants/act/token/token.cpp:203; ggml/src/ggml-gemmini/quants/act/token/token.cpp:223; ggml/src/ggml-gemmini/quants/act/token/token.cpp:236; ggml/src/ggml-gemmini/quants/act/token/token.cpp:275; ggml/src/ggml-gemmini/quants/act/token/token.cpp:309; ggml/src/ggml-gemmini/matmul/execution.cpp:243; ggml/src/ggml-gemmini/matmul/execution.cpp:967; ggml/src/ggml-gemmini/matmul/execution.cpp:1431; ggml/src/ggml-gemmini/matmul/execution.cpp:1481; ggml/src/ggml-gemmini/matmul/execution.cpp:2141
- Runtime override: No independent runtime enable switch confirmed; residual backend override is separate.
- Recommendation: KEEP integer residual correction (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:286; cmake/ggml-gemmini-options.cmake:329; cmake/ggml-gemmini-options.cmake:336; cmake/ggml-gemmini-options.cmake:439; cmake/ggml-gemmini-config.cmake:11; cmake/ggml-gemmini-config.cmake:71; ggml/src/ggml-potal/CMakeLists.txt:193; ggml/src/ggml-potal/CMakeLists.txt:214-215; ggml/src/ggml-potal/CMakeLists.txt:218-219; ggml/CMakeLists.txt:163; ggml/CMakeLists.txt:179; ggml/src/ggml-gemmini-utils/CMakeLists.txt:48; ggml/src/ggml-gemmini-utils/CMakeLists.txt:94; build-riscv.sh:23; scripts/im2p-build-options.py:32; ggml/src/ggml-potal-w4a4/CMakeLists.txt:222; ggml/src/ggml-potal-w4a4/CMakeLists.txt:243-244; ggml/src/ggml-potal-w4a4/CMakeLists.txt:247-248; ggml/src/ggml-potal-w4a4/CMakeLists.txt:260
- Source/macro consumers: ggml/src/ggml-gemmini/ops.cpp:508; ggml/src/ggml-gemmini/ops.cpp:1262; ggml/src/ggml-gemmini/ops.cpp:1413; ggml/src/ggml-gemmini/matmul/dense.cpp:1414; ggml/src/ggml-potal/residual/stripe-residual.hpp:3; ggml/src/ggml-potal/matmul/dense.cpp:437; ggml/src/ggml-potal/matmul/options.hpp:10; ggml/src/ggml-potal/matmul/options.hpp:61; ggml/src/ggml-potal/matmul/dense.hpp:12; ggml/src/ggml-potal/matmul/dense.hpp:194; ggml/src/ggml-potal/matmul/dense.hpp:209; ggml/src/ggml-potal-w4a4/residual/stripe-residual.hpp:3; ggml/src/ggml-potal-w4a4/matmul/dense.cpp:439; ggml/src/ggml-potal-w4a4/matmul/options.hpp:10; ggml/src/ggml-potal-w4a4/matmul/options.hpp:61; ggml/src/ggml-potal-w4a4/matmul/dense.hpp:12; ggml/src/ggml-potal-w4a4/matmul/dense.hpp:194; ggml/src/ggml-potal-w4a4/matmul/dense.hpp:209; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:38-39; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:807; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:201; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:219; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:241; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:255; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:299; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:333; ggml/src/ggml-gemmini/quants/act/block/block.cpp:56; ggml/src/ggml-gemmini/quants/act/block/block.cpp:67; ggml/src/ggml-gemmini/quants/act/block/block.cpp:75; ggml/src/ggml-gemmini/quants/act/block/block.cpp:117; ggml/src/ggml-gemmini/quants/act/block/block.cpp:164; ggml/src/ggml-gemmini/quants/act/block/block.cpp:178; ggml/src/ggml-gemmini/quants/act/block/block.cpp:187; ggml/src/ggml-gemmini/quants/act/block/block.cpp:281; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:206; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:225; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:500; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:163; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:239; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:259; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:308; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:344; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:191; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:218; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:231; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:267; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:294; ggml/src/ggml-gemmini/quants/act/token/token.cpp:188; ggml/src/ggml-gemmini/quants/act/token/token.cpp:203; ggml/src/ggml-gemmini/quants/act/token/token.cpp:223; ggml/src/ggml-gemmini/quants/act/token/token.cpp:236; ggml/src/ggml-gemmini/quants/act/token/token.cpp:275; ggml/src/ggml-gemmini/quants/act/token/token.cpp:309; ggml/src/ggml-gemmini/matmul/execution.cpp:243; ggml/src/ggml-gemmini/matmul/execution.cpp:967; ggml/src/ggml-gemmini/matmul/execution.cpp:1431; ggml/src/ggml-gemmini/matmul/execution.cpp:1481; ggml/src/ggml-gemmini/matmul/execution.cpp:2141
- Existing test references: tests/CMakeLists.txt:599; tests/CMakeLists.txt:620; tests/CMakeLists.txt:699; tests/CMakeLists.txt:786; tests/CMakeLists.txt:819; tests/CMakeLists.txt:890; tests/CMakeLists.txt:1043; tests/CMakeLists.txt:1241; tests/CMakeLists.txt:1313; tests/CMakeLists.txt:1331; tests/CMakeLists.txt:1343; tests/CMakeLists.txt:1367; tests/CMakeLists.txt:1464; tests/CMakeLists.txt:1515; tests/CMakeLists.txt:1555; ggml/src/ggml-potal/tests/potal-residual-merge-bench.cpp:122; ggml/src/ggml-potal/tests/potal-paired-execution-test.cpp:4024; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:572; ggml/src/ggml-potal/tests/potal-tiling-unit-test.cpp:17; ggml/src/ggml-potal/tests/potal-tiling-unit-test.cpp:783; ggml/src/ggml-potal/tests/potal-hp1-smoke.cpp:16; ggml/src/ggml-potal-w4a4/tests/potal-residual-merge-bench.cpp:122; ggml/src/ggml-potal-w4a4/tests/potal-paired-execution-test.cpp:4024; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:572; ggml/src/ggml-potal-w4a4/tests/potal-tiling-unit-test.cpp:17; ggml/src/ggml-potal-w4a4/tests/potal-tiling-unit-test.cpp:699; ggml/src/ggml-potal-w4a4/tests/potal-hp1-smoke.cpp:16; tests/test-gemmini-build-contract.cmake:193; tests/test-im2p-cache-build-scripts.cmake:487; tests/test-potal-target-profile.cmake:22

### GGML_GEMMINI_ENABLE_STRIPE_MATMUL

- Direct default / definition: ON; cmake/ggml-gemmini-options.cmake:301.
- Behavior / legal combinations: Required for stripe mode. Contracts: K9.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Runtime mode may request enabled capability.
- Recommendation: KEEP capability gate (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:301; cmake/ggml-gemmini-options.cmake:330; cmake/ggml-gemmini-options.cmake:349-350; cmake/ggml-gemmini-options.cmake:353-354; cmake/ggml-gemmini-options.cmake:374; cmake/ggml-gemmini-config.cmake:8
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/test-gemmini-build-contract.cmake:188

### GGML_GEMMINI_ENABLE_STRIPE_PIPELINE

- Direct default / definition: ON; cmake/ggml-gemmini-options.cmake:302.
- Behavior / legal combinations: Requires stripe matmul capability. Contracts: K9.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Runtime mode may request enabled capability.
- Recommendation: KEEP capability gate (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:302; cmake/ggml-gemmini-options.cmake:331; cmake/ggml-gemmini-options.cmake:349-350; cmake/ggml-gemmini-options.cmake:357-358; cmake/ggml-gemmini-options.cmake:375; cmake/ggml-gemmini-config.cmake:9
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/test-gemmini-build-contract.cmake:189

### GGML_GEMMINI_EXECUTION_BACKEND

- Direct default / definition: "HARDWARE"; cmake/ggml-gemmini-options.cmake:139.
- Behavior / legal combinations: HARDWARE default does not imply hardware work when OPTION=CPU. FPGA_UART explicitly rejected. Contracts: K3,K8,K12.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: No direct runtime transport override confirmed.
- Recommendation: MERGE production transport into NPU_TRANSPORT; KEEP IM2P simulator lane (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:139; cmake/ggml-gemmini-options.cmake:141-142; cmake/ggml-gemmini-options.cmake:145-146; cmake/ggml-gemmini-options.cmake:148; cmake/ggml-gemmini-options.cmake:152; cmake/ggml-gemmini-options.cmake:154; cmake/ggml-gemmini-options.cmake:167; cmake/ggml-gemmini-options.cmake:175; cmake/ggml-gemmini-options.cmake:178; cmake/ggml-gemmini-options.cmake:180; cmake/ggml-gemmini-options.cmake:231; cmake/ggml-gemmini-options.cmake:437; cmake/ggml-gemmini-config.cmake:45; CMakeLists.txt:201-202; ggml/CMakeLists.txt:178; ggml/src/ggml-gemmini-utils/CMakeLists.txt:41; ggml/src/ggml-gemmini-utils/CMakeLists.txt:81; cmake/ggml-gemmini-im2p.cmake:11; cmake/ggml-gemmini-im2p.cmake:48; cmake/ggml-gemmini-im2p.cmake:56; cmake/ggml-gemmini-im2p.cmake:59; cmake/ggml-gemmini-im2p.cmake:62; scripts/im2p-build-options.py:106; scripts/im2p-build-options.py:116; scripts/im2p-build-options.py:118; scripts/im2p-build-options.py:122; scripts/im2p-build-options.py:132; scripts/im2p-build-options.py:184; ggml/src/ggml-gemmini/CMakeLists.txt:306
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/CMakeLists.txt:96; tests/CMakeLists.txt:530; tests/CMakeLists.txt:551; tests/CMakeLists.txt:952; tests/CMakeLists.txt:1011; tests/CMakeLists.txt:1055; tests/CMakeLists.txt:1131; tests/CMakeLists.txt:1304; tests/test-gemmini-build-contract.cmake:197; tests/test-gemmini-build-contract.cmake:925; tests/test-gemmini-build-contract.cmake:972; tests/test-gemmini-build-contract.cmake:1122; tests/test-gemmini-build-contract.cmake:1124; tests/test-gemmini-build-contract.cmake:1200; tests/test-im2p-cache-build-scripts.cmake:39; tests/test-im2p-cache-build-scripts.cmake:92; tests/test-im2p-cache-build-scripts.cmake:121; tests/test-im2p-cache-build-scripts.cmake:201; tests/test-im2p-cache-build-scripts.cmake:216; tests/test-im2p-cache-build-scripts.cmake:219; tests/test-im2p-cache-build-scripts.cmake:250; tests/test-im2p-cache-build-scripts.cmake:274; tests/test-im2p-cache-build-scripts.cmake:329; tests/test-im2p-cache-build-scripts.cmake:450; tests/test-im2p-cache-build-scripts.cmake:481; tests/test-im2p-cache-build-scripts.cmake:548; tests/test-potal-target-profile.cmake:21

### GGML_GEMMINI_EXSIA_DEFAULT_MODE

- Direct default / definition: "SEQUENTIAL"; cmake/ggml-gemmini-options.cmake:384.
- Behavior / legal combinations: Internal ExSIA modes separate from public FULL/STRIPE_PIPELINE. Contracts: K10.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Build default; no same-name runtime getenv consumer confirmed.
- Recommendation: KEEP CPU staging policy (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:384-386; cmake/ggml-gemmini-options.cmake:390
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/test-potal-target-profile.cmake:26

### GGML_GEMMINI_EXSIA_LOCAL_WORKERS

- Direct default / definition: "4"; cmake/ggml-gemmini-options.cmake:393.
- Behavior / legal combinations: 3/4 allowed; OpenMP team derived as workers+1. Contracts: K10.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1340; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1345; ggml/src/ggml-gemmini/quants/act/exsia/types.hpp:22-23
- Runtime override: OMP runtime controls plus explicit hart roles, not geometry.
- Recommendation: KEEP CPU worker policy (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:393-395; cmake/ggml-gemmini-options.cmake:397-398; cmake/ggml-gemmini-options.cmake:404; cmake/ggml-gemmini-config.cmake:20; cmake/ggml-gemmini-config.cmake:70
- Source/macro consumers: ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1340; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1345; ggml/src/ggml-gemmini/quants/act/exsia/types.hpp:22-23; ggml/src/ggml-gemmini/quants/act/exsia/types.hpp:60
- Existing test references: tests/CMakeLists.txt:784; tests/CMakeLists.txt:817; tests/CMakeLists.txt:892; tests/CMakeLists.txt:1243; tests/test-gemmini-build-contract.cmake:192

### GGML_GEMMINI_EXSIA_PROFILE_SCOPE

- Direct default / definition: "OFF"; cmake/ggml-gemmini-options.cmake:267.
- Behavior / legal combinations: OFF/TIMELINE/STAGE ->0/1/2. Contracts: K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: GGML_GEMMINI_CYCLE_DETAIL_LOG sets output path (ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:31-41).
- Recommendation: KEEP profiling (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:267-269; cmake/ggml-gemmini-options.cmake:273; cmake/ggml-gemmini-options.cmake:426; scripts/im2p-build-options.py:99; scripts/im2p-build-options.py:101; scripts/im2p-build-options.py:104-105; ggml/src/ggml-gemmini/CMakeLists.txt:17-19; ggml/src/ggml-gemmini/CMakeLists.txt:27
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### GGML_GEMMINI_EXSIA_SIGMA

- Direct default / definition: "2"; cmake/ggml-gemmini-options.cmake:207.
- Behavior / legal combinations: Positive integer <=2147483647; attention requires 2. Contracts: K13.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:17-18; ggml/src/ggml-gemmini/quants/act/block/block.cpp:18-19; ggml/src/ggml-gemmini/quants/act/exsia/types.hpp:12-13; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:18-19; ggml/src/ggml-gemmini/quants/act/token/token.cpp:18-19
- Runtime override: No direct runtime sigma override confirmed.
- Recommendation: KEEP local adaptation algorithm (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:207; cmake/ggml-gemmini-options.cmake:225; cmake/ggml-gemmini-options.cmake:236-237; cmake/ggml-gemmini-options.cmake:239; cmake/ggml-gemmini-options.cmake:242; cmake/ggml-gemmini-options.cmake:244; cmake/ggml-gemmini-config.cmake:35; ggml/src/ggml-potal/CMakeLists.txt:344; scripts/im2p-build-options.py:149; ggml/src/ggml-potal-w4a4/CMakeLists.txt:258; ggml/src/ggml-potal-w4a4/CMakeLists.txt:383
- Source/macro consumers: ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:17-18; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:123; ggml/src/ggml-gemmini/quants/act/block/block.cpp:18-19; ggml/src/ggml-gemmini/quants/act/block/block.cpp:89; ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp:26; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:418; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:27; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:249; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:814; ggml/src/ggml-gemmini/quants/act/exsia/types.hpp:12-13; ggml/src/ggml-gemmini/quants/act/exsia/types.hpp:69; ggml/src/ggml-gemmini/quants/act/exsia/types.hpp:78; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:18-19; ggml/src/ggml-gemmini/quants/act/tensor/tensor.cpp:119; ggml/src/ggml-gemmini/quants/act/token/token.cpp:18-19; ggml/src/ggml-gemmini/quants/act/token/token.cpp:118
- Existing test references: tests/CMakeLists.txt:598; tests/CMakeLists.txt:619; tests/CMakeLists.txt:674; tests/CMakeLists.txt:697; tests/CMakeLists.txt:783; tests/CMakeLists.txt:816; tests/CMakeLists.txt:888; tests/CMakeLists.txt:1556; ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:799; ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:806

### GGML_GEMMINI_EXSIA_STRIPE_ANCHOR

- Direct default / definition: "2"; cmake/ggml-gemmini-options.cmake:208.
- Behavior / legal combinations: 2 or 3 distinct-exponent anchor. Contracts: K10.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:132; ggml/src/ggml-gemmini/quants/act/exsia/types.hpp:16-17
- Runtime override: Gemmini sensitivity policy/trace controls are separate; see runtime inventory.
- Recommendation: KEEP stripe folding algorithm (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:208-211; cmake/ggml-gemmini-config.cmake:19; cmake/ggml-gemmini-config.cmake:36; ggml/src/ggml-potal/CMakeLists.txt:345; ggml/src/ggml-potal-w4a4/CMakeLists.txt:259; ggml/src/ggml-potal-w4a4/CMakeLists.txt:384
- Source/macro consumers: ggml/src/ggml-potal/matmul/attention.cpp:351; ggml/src/ggml-potal-w4a4/matmul/attention.cpp:360; ggml/src/ggml-gemmini/potal-sensitivity.cpp:26; ggml/src/ggml-gemmini/ggml-gemmini-args.h:150; ggml/src/ggml-gemmini/quants/act/exsia/types.hpp:16-17; ggml/src/ggml-gemmini/quants/act/exsia/types.hpp:19
- Existing test references: tests/CMakeLists.txt:3; ggml/src/ggml-potal/tests/potal-attention-test.cpp:418; ggml/src/ggml-potal/tests/potal-attention-test.cpp:422-423; ggml/src/ggml-potal-w4a4/tests/potal-attention-test.cpp:418; ggml/src/ggml-potal-w4a4/tests/potal-attention-test.cpp:422-423; ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:132-133

### GGML_GEMMINI_FP32

- Direct default / definition: OFF; cmake/ggml-gemmini-options.cmake:2.
- Behavior / legal combinations: Replaces source list, DIM8 SDK requirement; has block/attention tests. Contracts: K14.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:15; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:81; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:98; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:115; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:136; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:154; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:3; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:89; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:100; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:113; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:131; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:147; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:164; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:176; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:183; src/llama-graph.cpp:7; src/llama-graph.cpp:1264; src/llama-graph.cpp:1334
- Runtime override: No runtime toggle.
- Recommendation: KEEP experiment baseline (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:2-3; cmake/ggml-gemmini-options.cmake:7; cmake/ggml-gemmini-options.cmake:11; cmake/ggml-gemmini-options.cmake:13; cmake/ggml-gemmini-options.cmake:198; cmake/ggml-gemmini-options.cmake:200; ggml/src/ggml-gemmini/CMakeLists.txt:76; ggml/src/ggml-gemmini/fp32/backend.cmake:8; ggml/src/ggml-gemmini/fp32/cpu/CMakeLists.txt:9
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:15; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:81; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:98; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:115; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:136; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:154; src/llama-graph.cpp:7; src/llama-graph.cpp:1264; src/llama-graph.cpp:1334
- Existing test references: ggml/src/ggml-gemmini/fp32/tests/runner.cpp:3; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:89; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:100; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:113; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:131; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:147; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:164; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:176; ggml/src/ggml-gemmini/fp32/tests/runner.cpp:183

### GGML_GEMMINI_FULL_C

- Direct default / definition: "1"; cmake/ggml-gemmini-options.cmake:50.
- Behavior / legal combinations: Validated 0/1, translated to SDK macro in cmake/ggml-gemmini-config.cmake:22-30. In-tree source use and exports listed below; external SDK effect UNVERIFIED. Contracts: K3,K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini.cpp:58-59; ggml/src/ggml-gemmini/ops.cpp:56-57
- Runtime override: No direct runtime getenv of this cache option confirmed.
- Recommendation: KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:50-51; cmake/ggml-gemmini-options.cmake:249; cmake/ggml-gemmini-config.cmake:25; ggml/src/ggml-gemmini/CMakeLists.txt:230
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini.cpp:58-59; ggml/src/ggml-gemmini/ops.cpp:56-57; ggml/src/ggml-gemmini/ops.cpp:1332
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### GGML_GEMMINI_LOW_D

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:52.
- Behavior / legal combinations: Validated 0/1, translated to SDK macro in cmake/ggml-gemmini-config.cmake:22-30. In-tree source use and exports listed below; external SDK effect UNVERIFIED. Contracts: K3,K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini.cpp:61-62; ggml/src/ggml-gemmini/ops.cpp:59-60
- Runtime override: No direct runtime getenv of this cache option confirmed.
- Recommendation: KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:52-53; cmake/ggml-gemmini-options.cmake:250; cmake/ggml-gemmini-config.cmake:26; ggml/src/ggml-gemmini/CMakeLists.txt:231
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini.cpp:61-62; ggml/src/ggml-gemmini/ops.cpp:59-60; ggml/src/ggml-gemmini/ops.cpp:1333
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION

- Direct default / definition: OFF; cmake/ggml-gemmini-options.cmake:214.
- Behavior / legal combinations: Requires ENABLE_A4NKS_ATTENTION plus WS+IM2P_SIM+GEMMINI_HP1. Contracts: K13.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/matmul/attention.cpp:330; ggml/src/ggml-gemmini/matmul/attention.cpp:334; ggml/src/ggml-gemmini/matmul/attention.cpp:345; ggml/src/ggml-gemmini/matmul/attention.cpp:408; ggml/src/ggml-gemmini/matmul/attention.cpp:445
- Runtime override: Compile-time only.
- Recommendation: KEEP explicit simulated attention execution gate (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:6; cmake/ggml-gemmini-options.cmake:214-215; cmake/ggml-gemmini-options.cmake:229; cmake/ggml-gemmini-config.cmake:15; build-x86.sh:20; build-arm64-cpu.sh:13; build-riscv.sh:36; build-arm64.sh:13; scripts/im2p-build-options.py:33; scripts/im2p-build-options.py:110
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini-matmul-config.hpp.in:5; ggml/src/ggml-gemmini/matmul/attention.cpp:330; ggml/src/ggml-gemmini/matmul/attention.cpp:334; ggml/src/ggml-gemmini/matmul/attention.cpp:345; ggml/src/ggml-gemmini/matmul/attention.cpp:408; ggml/src/ggml-gemmini/matmul/attention.cpp:445
- Existing test references: tests/test-im2p-cache-build-scripts.cmake:328; tests/test-im2p-cache-build-scripts.cmake:342; tests/test-im2p-cache-build-scripts.cmake:479; tests/test-potal-target-profile.cmake:31

### GGML_GEMMINI_OPTION

- Direct default / definition: "CPU"; cmake/ggml-gemmini-options.cmake:38.
- Behavior / legal combinations: CPU default; OS accepted in advertised strings but always rejected by legal-pair validation. CPU and WS remain meaningful distinct routes. Contracts: K3.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini.cpp:64-65; ggml/src/ggml-gemmini/ops.cpp:62-63
- Runtime override: Runtime matmul options may change invocation/residual policy, not this compiled macro directly.
- Recommendation: KEEP execution policy; merge hardware selection only after CPU reference is separated (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:38-40; cmake/ggml-gemmini-options.cmake:45; cmake/ggml-gemmini-options.cmake:47-49; cmake/ggml-gemmini-options.cmake:230; cmake/ggml-gemmini-options.cmake:440; cmake/ggml-gemmini-config.cmake:24; ggml/CMakeLists.txt:171; ggml/src/ggml-gemmini-utils/CMakeLists.txt:42; scripts/im2p-build-options.py:141; ggml/src/ggml-gemmini/CMakeLists.txt:229; ggml/src/ggml-gemmini/CMakeLists.txt:271
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini.cpp:64-65; ggml/src/ggml-gemmini/ops.cpp:62-63; ggml/src/ggml-gemmini/ops.cpp:501; ggml/src/ggml-gemmini/ops.cpp:534; ggml/src/ggml-gemmini/ops.cpp:545; ggml/src/ggml-gemmini/ops.cpp:1338; ggml/src/ggml-gemmini/ops.cpp:1817; ggml/src/ggml-gemmini/ops.cpp:2338; ggml/src/ggml-gemmini/ops.cpp:2348; ggml/src/ggml-gemmini/ops.cpp:2454
- Existing test references: tests/CMakeLists.txt:379; tests/CMakeLists.txt:552; tests/test-gemmini-build-contract.cmake:754; tests/test-gemmini-build-contract.cmake:924; tests/test-gemmini-build-contract.cmake:1076; tests/test-gemmini-off-build.cmake:21; tests/test-im2p-cache-build-scripts.cmake:40; tests/test-im2p-cache-build-scripts.cmake:93; tests/test-im2p-cache-build-scripts.cmake:122; tests/test-im2p-cache-build-scripts.cmake:250; tests/test-im2p-cache-build-scripts.cmake:274; tests/test-im2p-cache-build-scripts.cmake:329; tests/test-im2p-cache-build-scripts.cmake:480; tests/test-im2p-cache-build-scripts.cmake:549; tests/test-potal-target-profile.cmake:21

### GGML_GEMMINI_PB

- Direct default / definition: OFF; cmake/ggml-gemmini-options.cmake:429.
- Behavior / legal combinations: Partial-sum K-block path; has Q8_0 or MXINT8 block/attention tests. Contracts: K15.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini.cpp:247; ggml/src/ggml-gemmini/ops.cpp:974; ggml/src/ggml-gemmini/ops.cpp:2309; ggml/src/ggml-gemmini/ops.cpp:2320; ggml/src/ggml-gemmini/matmul/dense.cpp:3; ggml/src/ggml-gemmini/matmul/dense.cpp:765; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:12; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:78; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:111; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:150; ggml/src/ggml-gemmini/matmul/attention.cpp:2; ggml/src/ggml-gemmini/matmul/attention.cpp:33; ggml/src/ggml-gemmini/pb/pb-attention.cpp:3; ggml/src/ggml-gemmini/pb/pb-rocc-model.cpp:1; ggml/src/ggml-gemmini/pb/pb-exec.hpp:3; ggml/src/ggml-gemmini/pb/pb-attention.hpp:3; ggml/src/ggml-gemmini/pb/pb-exec.cpp:3; src/llama-graph.cpp:7; src/llama-graph.cpp:1259; src/llama-graph.cpp:1319
- Runtime override: No runtime toggle.
- Recommendation: KEEP experiment baseline (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:4; cmake/ggml-gemmini-options.cmake:429-430; cmake/ggml-gemmini-options.cmake:440; cmake/ggml-gemmini-options.cmake:443; ggml/src/ggml-gemmini/CMakeLists.txt:158; ggml/src/ggml-gemmini/CMakeLists.txt:165
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini.cpp:247; ggml/src/ggml-gemmini/ops.cpp:974; ggml/src/ggml-gemmini/ops.cpp:2309; ggml/src/ggml-gemmini/ops.cpp:2320; ggml/src/ggml-gemmini/matmul/dense.cpp:3; ggml/src/ggml-gemmini/matmul/dense.cpp:765; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:12; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:78; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:111; ggml/src/ggml-gemmini/ggml-gemmini-buffer.cpp:150; ggml/src/ggml-gemmini/matmul/attention.cpp:2; ggml/src/ggml-gemmini/matmul/attention.cpp:33; ggml/src/ggml-gemmini/matmul/attention.cpp:676; ggml/src/ggml-gemmini/pb/pb-attention.cpp:3; ggml/src/ggml-gemmini/pb/pb-rocc-model.cpp:1; ggml/src/ggml-gemmini/pb/pb-exec.hpp:3; ggml/src/ggml-gemmini/pb/pb-attention.hpp:3; ggml/src/ggml-gemmini/pb/pb-exec.cpp:3; src/llama-graph.cpp:7; src/llama-graph.cpp:1259; src/llama-graph.cpp:1284; src/llama-graph.cpp:1319; src/llama-graph.cpp:1376
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### GGML_GEMMINI_PB_WEIGHT_FORMAT

- Direct default / definition: "Q8_0"; ggml/src/ggml-gemmini/CMakeLists.txt:166.
- Behavior / legal combinations: Q8_0/MXINT8 only, default Q8_0 inside PB backend. Contracts: K15.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Build-time only.
- Recommendation: KEEP PB reference format (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-gemmini/CMakeLists.txt:166-167; ggml/src/ggml-gemmini/CMakeLists.txt:169-170; ggml/src/ggml-gemmini/CMakeLists.txt:178
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### GGML_GEMMINI_PRINT_TILE

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:36.
- Behavior / legal combinations: Validated 0/1, translated to SDK macro in cmake/ggml-gemmini-config.cmake:22-30. In-tree source use and exports listed below; external SDK effect UNVERIFIED. Contracts: K3,K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: No direct runtime getenv of this cache option confirmed.
- Recommendation: KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:36-37; cmake/ggml-gemmini-options.cmake:248; cmake/ggml-gemmini-config.cmake:23; ggml/src/ggml-gemmini/CMakeLists.txt:228
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### GGML_GEMMINI_RESIDUAL_METRICS

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:16.
- Behavior / legal combinations: Validated 0/1 and exported through runtime/utils definitions. ACT_METRICS implies ACT_QUANT_METRICS. Contracts: K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ops.cpp:286; ggml/src/ggml-gemmini/ops.cpp:1020; ggml/src/ggml-gemmini/ops.cpp:1311; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1381; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1771; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1967; ggml/src/ggml-gemmini/ggml-gemmini-args.h:325; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1689; ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp:45; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:223; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:456; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:586; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:2048; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:151; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:118; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:231; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:436; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:482; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:503; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:526; ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:16-17
- Runtime override: Build-time gate; paths/output controls are independent.
- Recommendation: KEEP independent profiling metric (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-config.cmake:58; cmake/ggml-gemmini-config.cmake:84; ggml/src/ggml-gemmini-utils/CMakeLists.txt:34
- Source/macro consumers: ggml/src/ggml-gemmini/ops.cpp:286; ggml/src/ggml-gemmini/ops.cpp:1020; ggml/src/ggml-gemmini/ops.cpp:1311; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1381; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1771; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1967; ggml/src/ggml-gemmini/ggml-gemmini-args.h:325; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1689; ggml/src/ggml-gemmini/quants/act/preparation-cache.cpp:93; ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp:45; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:223; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:456; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:586; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:2048; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:151; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:118; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:231; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:243; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:436; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:482; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:503; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:526; ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:16-17
- Existing test references: tests/CMakeLists.txt:319; tests/CMakeLists.txt:1042; tests/CMakeLists.txt:1054; tests/CMakeLists.txt:1725

### GGML_GEMMINI_SCALE_METRICS

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:16.
- Behavior / legal combinations: Validated 0/1 and exported through runtime/utils definitions. ACT_METRICS implies ACT_QUANT_METRICS. Contracts: K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ops.cpp:286; ggml/src/ggml-gemmini/ops.cpp:1020; ggml/src/ggml-gemmini/ops.cpp:1311; ggml/src/ggml-gemmini/ggml-gemmini-evaluation-observer.hpp:12; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1381; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1771; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1967; ggml/src/ggml-gemmini/ggml-gemmini-args.h:325; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1718; ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp:45; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:456; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:586; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:2048; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:156; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:40; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:118; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:231; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:590; ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:19-20
- Runtime override: Build-time gate; paths/output controls are independent.
- Recommendation: KEEP independent profiling metric (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-config.cmake:56; cmake/ggml-gemmini-config.cmake:82; ggml/src/ggml-gemmini-utils/CMakeLists.txt:34
- Source/macro consumers: ggml/src/ggml-gemmini/ops.cpp:286; ggml/src/ggml-gemmini/ops.cpp:1020; ggml/src/ggml-gemmini/ops.cpp:1311; ggml/src/ggml-gemmini/ggml-gemmini-evaluation-observer.hpp:12; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1381; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1771; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1967; ggml/src/ggml-gemmini/ggml-gemmini-args.h:325; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1718; ggml/src/ggml-gemmini/quants/act/preparation-cache.cpp:93; ggml/src/ggml-gemmini/quants/act/exsia/exsia-event.hpp:45; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:456; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:586; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:2048; ggml/src/ggml-gemmini/quants/act/exsia/folding.cpp:156; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:40; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:118; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:231; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:245; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:590; ggml/src/ggml-gemmini-utils/include/gemmini/evaluation_metrics.hpp:19-20
- Existing test references: tests/CMakeLists.txt:317; tests/CMakeLists.txt:1042; tests/CMakeLists.txt:1054; tests/CMakeLists.txt:1723

### GGML_GEMMINI_TRANSPOSE_B

- Direct default / definition: "1"; cmake/ggml-gemmini-options.cmake:54.
- Behavior / legal combinations: Validated 0/1, translated to SDK macro in cmake/ggml-gemmini-config.cmake:22-30. In-tree source use and exports listed below; external SDK effect UNVERIFIED. Contracts: K3,K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini.cpp:55-56; ggml/src/ggml-gemmini/ops.cpp:53-54
- Runtime override: No direct runtime getenv of this cache option confirmed.
- Recommendation: KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:54-55; cmake/ggml-gemmini-options.cmake:251; cmake/ggml-gemmini-config.cmake:27; ggml/src/ggml-gemmini/CMakeLists.txt:232
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini.cpp:55-56; ggml/src/ggml-gemmini/ops.cpp:53-54; ggml/src/ggml-gemmini/ops.cpp:530; ggml/src/ggml-gemmini/ops.cpp:1330; ggml/src/ggml-gemmini/ops.cpp:1636; ggml/src/ggml-gemmini/ops.cpp:2433; ggml/src/ggml-gemmini/ops.cpp:2437; ggml/src/ggml-gemmini/ops.cpp:2510
- Existing test references: tests/test-gemmini-cpu-cycle-validity-contract.cmake:219

### GGML_GEMMINI_WEIGHT_BITS

- Direct default / definition: "8"; cmake/ggml-gemmini-options.cmake:132.
- Behavior / legal combinations: Matched with activation bits; must preserve storage ABI separately from logical width. Contracts: K4,K7,K12-K16.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:46-47; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:55
- Runtime override: No process env precision override in backend.
- Recommendation: MERGE into NPU_PRECISION (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:132-135; cmake/ggml-gemmini-options.cmake:137; cmake/ggml-gemmini-options.cmake:157; cmake/ggml-gemmini-options.cmake:434; cmake/ggml-gemmini-config.cmake:38; cmake/ggml-gemmini-config.cmake:73; ggml/src/ggml-potal/CMakeLists.txt:194-196; ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:27; ggml/CMakeLists.txt:174; ggml/src/ggml-gemmini-utils/CMakeLists.txt:37; cmake/ggml-gemmini-im2p.cmake:72; cmake/ggml-gemmini-im2p.cmake:202; cmake/ggml-gemmini-im2p.cmake:258; cmake/ggml-gemmini-im2p.cmake:305; cmake/ggml-gemmini-im2p.cmake:307; cmake/ggml-gemmini-im2p.cmake:314; scripts/im2p-build-options.py:146; ggml/src/ggml-potal-w4a4/CMakeLists.txt:223-225; ggml/src/ggml-gemmini/CMakeLists.txt:285
- Source/macro consumers: ggml/src/ggml-gemmini/ops.cpp:82; ggml/src/ggml-gemmini/ops.cpp:216; ggml/src/ggml-gemmini/ops.cpp:219; ggml/src/ggml-gemmini/ops.cpp:331; ggml/src/ggml-gemmini/ops.cpp:502; ggml/src/ggml-gemmini/ops.cpp:1203; ggml/src/ggml-gemmini/ops.cpp:1209; ggml/src/ggml-gemmini/ops.cpp:1251; ggml/src/ggml-gemmini/ops.cpp:1261; ggml/src/ggml-potal/matmul/options.hpp:44; ggml/src/ggml-potal-w4a4/matmul/options.hpp:44; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:46-47; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:55; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:131-134; ggml/src/ggml-gemmini/ggml-gemmini-config.hpp:138; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:709; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:839; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1918; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2124; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2540; ggml/src/ggml-gemmini/ggml-gemmini-args.h:376; ggml/src/ggml-gemmini/ggml-gemmini-args.h:378; ggml/src/ggml-gemmini/ggml-gemmini-args.h:404; ggml/src/ggml-gemmini/ggml-gemmini-args.h:406; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:129; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:728; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:806; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1598; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:2016; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:2171; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:2281; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:2325; ggml/src/ggml-gemmini/im2p/route.cpp:185; ggml/src/ggml-gemmini-utils/src/evaluation_metrics.cpp:120
- Existing test references: ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:27; tests/CMakeLists.txt:81; tests/CMakeLists.txt:523; tests/CMakeLists.txt:596; tests/CMakeLists.txt:617; tests/CMakeLists.txt:673; tests/CMakeLists.txt:696; tests/CMakeLists.txt:733; tests/CMakeLists.txt:782; tests/CMakeLists.txt:815; tests/CMakeLists.txt:887; tests/CMakeLists.txt:951; tests/CMakeLists.txt:978; tests/CMakeLists.txt:1032; tests/CMakeLists.txt:1079; tests/CMakeLists.txt:1100; tests/CMakeLists.txt:1121; tests/CMakeLists.txt:1158; tests/CMakeLists.txt:1177; tests/CMakeLists.txt:1240; tests/CMakeLists.txt:1311; tests/CMakeLists.txt:1365; tests/CMakeLists.txt:1397; tests/CMakeLists.txt:1452; tests/CMakeLists.txt:1459; tests/CMakeLists.txt:1514; tests/CMakeLists.txt:1553; tests/CMakeLists.txt:1615; ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:798; ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:805; tests/test-gemmini-build-contract.cmake:196; tests/test-gemmini-build-contract.cmake:1157; tests/test-im2p-cache-build-scripts.cmake:203; tests/test-im2p-cache-build-scripts.cmake:252; tests/test-im2p-cache-build-scripts.cmake:273; tests/test-im2p-cache-build-scripts.cmake:331; tests/test-potal-target-profile.cmake:27

### GGML_GEMMINI_WS_DEBUG

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:56.
- Behavior / legal combinations: Validated 0/1, translated to SDK macro in cmake/ggml-gemmini-config.cmake:22-30. In-tree source use and exports listed below; external SDK effect UNVERIFIED. Contracts: K3,K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: No direct runtime getenv of this cache option confirmed.
- Recommendation: KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:56-57; cmake/ggml-gemmini-options.cmake:252; cmake/ggml-gemmini-config.cmake:28
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### GGML_GEMMINI_WS_LOOP_CYCLE

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:58.
- Behavior / legal combinations: Validated 0/1, translated to SDK macro in cmake/ggml-gemmini-config.cmake:22-30. In-tree source use and exports listed below; external SDK effect UNVERIFIED. Contracts: K3,K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: No direct runtime getenv of this cache option confirmed.
- Recommendation: KEEP until selected SDK consumer is audited; then merge fixed contract/debug controls (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:58-59; cmake/ggml-gemmini-options.cmake:253; cmake/ggml-gemmini-config.cmake:29
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/test-gemmini-build-contract.cmake:90; tests/test-gemmini-build-contract.cmake:855

### GGML_POTAL

- Direct default / definition: OFF; ggml/CMakeLists.txt:220.
- Behavior / legal combinations: Selects full INT8 backend unless W4 selector also ON; enables llama attention marker. Contracts: K1,K6-K8.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: No runtime override of compiled backend.
- Recommendation: MERGE into GGML_NPU=POTAL (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-geometry.cmake:2; cmake/ggml-gemmini-options.cmake:4; cmake/ggml-gemmini-options.cmake:431; cmake/ggml-gemmini-options.cmake:440; ggml/src/ggml-potal/CMakeLists.txt:5; ggml/src/ggml-potal/CMakeLists.txt:8; ggml/src/ggml-potal/CMakeLists.txt:178; ggml/src/ggml-potal/CMakeLists.txt:181; ggml/src/ggml-potal/CMakeLists.txt:184; ggml/src/ggml-potal/CMakeLists.txt:187; ggml/src/ggml-potal/CMakeLists.txt:195; ggml/src/ggml-potal/CMakeLists.txt:199; ggml/src/ggml-potal/CMakeLists.txt:211; ggml/src/ggml-potal/CMakeLists.txt:215; ggml/src/ggml-potal/CMakeLists.txt:219; ggml/src/ggml-potal/CMakeLists.txt:244; CMakeLists.txt:167; CMakeLists.txt:174-175; CMakeLists.txt:180; CMakeLists.txt:182; CMakeLists.txt:188; CMakeLists.txt:197; CMakeLists.txt:202; ggml/CMakeLists.txt:220; build-x86.sh:11; build-x86.sh:50; build-riscv.sh:14; build-riscv.sh:65; scripts/im2p-build-options.py:34; scripts/im2p-build-options.py:90; ggml/src/CMakeLists.txt:321; ggml/src/ggml-potal-w4a4/CMakeLists.txt:5; ggml/src/ggml-potal-w4a4/CMakeLists.txt:8; ggml/src/ggml-potal-w4a4/CMakeLists.txt:207; ggml/src/ggml-potal-w4a4/CMakeLists.txt:210; ggml/src/ggml-potal-w4a4/CMakeLists.txt:213; ggml/src/ggml-potal-w4a4/CMakeLists.txt:216; ggml/src/ggml-potal-w4a4/CMakeLists.txt:228; ggml/src/ggml-potal-w4a4/CMakeLists.txt:240; ggml/src/ggml-potal-w4a4/CMakeLists.txt:244; ggml/src/ggml-potal-w4a4/CMakeLists.txt:248; ggml/src/ggml-potal-w4a4/CMakeLists.txt:282; ggml/src/ggml-gemmini/fp32/cpu/CMakeLists.txt:10; src/CMakeLists.txt:49
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/CMakeLists.txt:1729; ggml/src/ggml-potal/tests/potal-ggml-smoke.cpp:383; ggml/src/ggml-potal/tests/potal-log-test.cpp:250; ggml/src/ggml-potal/tests/potal-ggml-unit-test.cpp:160; ggml/src/ggml-potal-w4a4/tests/potal-ggml-smoke.cpp:383; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:250; ggml/src/ggml-potal-w4a4/tests/potal-ggml-unit-test.cpp:160; tests/test-im2p-cache-build-scripts.cmake:200; tests/test-potal-target-profile.cmake:20

### GGML_POTAL_PAIRED_AUTO

- Direct default / definition: OFF; ggml/src/ggml-potal/CMakeLists.txt:227.
- Behavior / legal combinations: Default OFF; meaningful only STRIPE_PIPELINE with compensation; serial route remains baseline. Contracts: K8,K9.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-potal/matmul/options.hpp:92; ggml/src/ggml-potal-w4a4/matmul/options.hpp:92
- Runtime override: GGML_POTAL_PAIRED_AUTO env overrides process default (ggml/src/ggml-potal/ops.cpp:229-246).
- Recommendation: KEEP paired execution policy (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-potal/CMakeLists.txt:224; ggml/src/ggml-potal/CMakeLists.txt:227; ggml/src/ggml-potal/CMakeLists.txt:237; ggml/src/ggml-potal/CMakeLists.txt:347; ggml/src/ggml-potal/CMakeLists.txt:720; ggml/src/ggml-potal-w4a4/CMakeLists.txt:253; ggml/src/ggml-potal-w4a4/CMakeLists.txt:256; ggml/src/ggml-potal-w4a4/CMakeLists.txt:263; ggml/src/ggml-potal-w4a4/CMakeLists.txt:275; ggml/src/ggml-potal-w4a4/CMakeLists.txt:386; ggml/src/ggml-potal-w4a4/CMakeLists.txt:768
- Source/macro consumers: ggml/src/ggml-potal/ops.cpp:231; ggml/src/ggml-potal/ops.cpp:321; ggml/src/ggml-potal/ops.cpp:334; ggml/src/ggml-potal/matmul/options.hpp:90; ggml/src/ggml-potal/matmul/options.hpp:92-93; ggml/src/ggml-potal/matmul/options.hpp:95; ggml/src/ggml-potal/matmul/stripe.hpp:202; ggml/src/ggml-potal-w4a4/ops.cpp:233; ggml/src/ggml-potal-w4a4/ops.cpp:323; ggml/src/ggml-potal-w4a4/ops.cpp:336; ggml/src/ggml-potal-w4a4/matmul/options.hpp:90; ggml/src/ggml-potal-w4a4/matmul/options.hpp:92-93; ggml/src/ggml-potal-w4a4/matmul/options.hpp:95; ggml/src/ggml-potal-w4a4/matmul/stripe.hpp:202
- Existing test references: ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:1; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:14; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:178; ggml/src/ggml-potal/tests/potal-full-inference.cpp:384; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:1; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:14; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:178; ggml/src/ggml-potal-w4a4/tests/potal-full-inference.cpp:384; tests/test-potal-target-profile.cmake:25

### GGML_POTAL_W4A4

- Direct default / definition: OFF; ggml/CMakeLists.txt:221.
- Behavior / legal combinations: Requires GGML_POTAL selection; exact packed W4 contract remains required. Contracts: K2,K7.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: No runtime precision switch.
- Recommendation: MERGE into NPU_PRECISION=INT4 (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:4; ggml/CMakeLists.txt:221; ggml/src/CMakeLists.txt:321; ggml/src/ggml-potal-w4a4/CMakeLists.txt:224
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### IM2P_DIM

- Direct default / definition: unset; deprecated DIM alias; cmake/ggml-gemmini-options.cmake:186.
- Behavior / legal combinations: Deprecated compatibility alias; conflicts with GGML_GEMMINI_DIM rejected. Has active resolver and validation references, so not confirmed unused. Contracts: K4,K12.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Build-time only.
- Recommendation: REMOVE candidate alias after caller migration (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:186-188; cmake/ggml-gemmini-options.cmake:190-191; cmake/ggml-gemmini-options.cmake:193; build-x86.sh:18; build-arm64-cpu.sh:11; build-arm64.sh:11; scripts/im2p-build-options.py:66-71; scripts/im2p-host-provision.sh:90; ggml/src/ggml-gemmini/CMakeLists.txt:286
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/CMakeLists.txt:82; tests/test-gemmini-build-contract.cmake:909

### IM2P_SIM_BUILD_DIR

- Direct default / definition: ""; cmake/ggml-gemmini-options.cmake:162.
- Behavior / legal combinations: Empty means IM2P_SIM_ROOT/build. Contracts: K12.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Wrapper environment forwarded at build time.
- Recommendation: KEEP simulator cache isolation (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:162; cmake/ggml-gemmini-im2p.cmake:75; cmake/ggml-gemmini-im2p.cmake:79; scripts/im2p-build-options.py:139-140
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/test-im2p-cache-build-scripts.cmake:440

### IM2P_SIM_IMPLEMENTATION

- Direct default / definition: ""; cmake/ggml-gemmini-options.cmake:164.
- Behavior / legal combinations: Empty outside IM2P; defaults GEMMINI_HP1 with cycle else LEGACY_BSV in direct CMake. Contracts: K12.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Wrapper environment forwarded at build time.
- Recommendation: KEEP simulator implementation selector (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:164; cmake/ggml-gemmini-options.cmake:166; cmake/ggml-gemmini-options.cmake:168; cmake/ggml-gemmini-options.cmake:170; cmake/ggml-gemmini-options.cmake:172; cmake/ggml-gemmini-options.cmake:176; cmake/ggml-gemmini-options.cmake:178; cmake/ggml-gemmini-options.cmake:181-182; cmake/ggml-gemmini-options.cmake:184; cmake/ggml-gemmini-options.cmake:232; cmake/ggml-gemmini-config.cmake:47; ggml/src/ggml-gemmini-utils/CMakeLists.txt:82; cmake/ggml-gemmini-im2p.cmake:12; cmake/ggml-gemmini-im2p.cmake:49; cmake/ggml-gemmini-im2p.cmake:82; cmake/ggml-gemmini-im2p.cmake:90; cmake/ggml-gemmini-im2p.cmake:113; cmake/ggml-gemmini-im2p.cmake:126; scripts/im2p-build-options.py:116; scripts/im2p-build-options.py:118; scripts/im2p-build-options.py:125; scripts/im2p-build-options.py:127; scripts/im2p-build-options.py:129; scripts/im2p-build-options.py:131; scripts/im2p-build-options.py:133-134; scripts/im2p-build-options.py:136-138; scripts/im2p-host-provision.sh:87; ggml/src/ggml-gemmini/CMakeLists.txt:319
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/CMakeLists.txt:97; tests/CMakeLists.txt:955; tests/CMakeLists.txt:1398; tests/CMakeLists.txt:1414; tests/CMakeLists.txt:1428; tests/CMakeLists.txt:1451; tests/CMakeLists.txt:1458; tests/CMakeLists.txt:1464; tests/CMakeLists.txt:1488; tests/test-im2p-cache-build-scripts.cmake:142; tests/test-im2p-cache-build-scripts.cmake:321; tests/test-im2p-cache-build-scripts.cmake:594

### IM2P_SIM_ROOT

- Direct default / definition: ""; cmake/ggml-gemmini-options.cmake:161.
- Behavior / legal combinations: Empty direct default; CYCLE_SIM derives sibling ../IM2P.sim. Contracts: K12.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Wrapper environment forwarded at build time.
- Recommendation: KEEP simulator input (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:161; cmake/ggml-gemmini-options.cmake:163; ggml/src/ggml-gemmini-utils/CMakeLists.txt:74; ggml/src/ggml-gemmini-utils/CMakeLists.txt:88; cmake/ggml-gemmini-im2p.cmake:2-3; cmake/ggml-gemmini-im2p.cmake:5-6; cmake/ggml-gemmini-im2p.cmake:19; cmake/ggml-gemmini-im2p.cmake:61-62; cmake/ggml-gemmini-im2p.cmake:64-66; cmake/ggml-gemmini-im2p.cmake:68; cmake/ggml-gemmini-im2p.cmake:76; cmake/ggml-gemmini-im2p.cmake:102; cmake/ggml-gemmini-im2p.cmake:237-238; build-x86.sh:17; scripts/im2p-build-options.py:139; scripts/im2p-host-provision.sh:72; scripts/im2p-host-provision.sh:78; ggml/src/ggml-gemmini/CMakeLists.txt:274-276; ggml/src/ggml-gemmini/CMakeLists.txt:294; ggml/src/ggml-gemmini/CMakeLists.txt:297; ggml/src/ggml-gemmini/CMakeLists.txt:304; ggml/src/ggml-gemmini/CMakeLists.txt:308; ggml/src/ggml-gemmini/CMakeLists.txt:340; ggml/src/ggml-gemmini/CMakeLists.txt:343; ggml/src/ggml-gemmini/CMakeLists.txt:357; ggml/src/ggml-gemmini/CMakeLists.txt:365-366
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/CMakeLists.txt:46; tests/CMakeLists.txt:55; tests/CMakeLists.txt:64-65; tests/CMakeLists.txt:76; tests/CMakeLists.txt:84; tests/CMakeLists.txt:557; tests/CMakeLists.txt:1015; tests/CMakeLists.txt:1061; tests/CMakeLists.txt:1135; tests/CMakeLists.txt:1166; tests/CMakeLists.txt:1183; tests/CMakeLists.txt:1323-1324; tests/CMakeLists.txt:1373-1374; tests/CMakeLists.txt:1405-1407; tests/CMakeLists.txt:1472-1473; tests/CMakeLists.txt:1608; tests/CMakeLists.txt:1620-1621; tests/test-gemmini-build-contract.cmake:921; tests/test-gemmini-build-contract.cmake:1119-1120; tests/test-gemmini-build-contract.cmake:1202; tests/test-im2p-cache-build-scripts.cmake:36; tests/test-im2p-cache-build-scripts.cmake:89; tests/test-im2p-cache-build-scripts.cmake:119; tests/test-im2p-cache-build-scripts.cmake:165; tests/test-im2p-cache-build-scripts.cmake:326; tests/test-im2p-cache-build-scripts.cmake:449; tests/test-im2p-cache-build-scripts.cmake:497; tests/test-im2p-cache-build-scripts.cmake:546

### LOG_CYCLE

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:409.
- Behavior / legal combinations: 0/1; compile-time gate. Dump paths are experiment artifacts, not accelerator transport. Contracts: K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini.cpp:33; ggml/src/ggml-gemmini/ggml-gemmini.cpp:36; ggml/src/ggml-gemmini/ggml-gemmini.cpp:160; ggml/src/ggml-potal/ggml-potal.cpp:365; ggml/src/ggml-gemmini/ops.cpp:30; ggml/src/ggml-gemmini/ops.cpp:33; ggml/src/ggml-gemmini/ops.cpp:94; ggml/src/ggml-gemmini/ops.cpp:997; ggml/src/ggml-gemmini/ops.cpp:1033; ggml/src/ggml-gemmini/ops.cpp:1393; ggml/src/ggml-gemmini/ops.cpp:2231; ggml/src/ggml-gemmini/matmul/dense.cpp:719; ggml/src/ggml-gemmini/matmul/dense.cpp:730; ggml/src/ggml-gemmini/matmul/dense.cpp:775; ggml/src/ggml-gemmini/matmul/dense.cpp:875; ggml/src/ggml-gemmini/matmul/dense.cpp:1422; ggml/src/ggml-gemmini/matmul/dense.cpp:1453; ggml/src/ggml-gemmini/matmul/dense.cpp:1468; ggml/src/ggml-gemmini/matmul/dense.cpp:1472; ggml/src/ggml-gemmini/matmul/dense.cpp:1497; ggml/src/ggml-gemmini/matmul/dense.cpp:1515; ggml/src/ggml-gemmini/matmul/dense.cpp:1539; ggml/src/ggml-gemmini/matmul/dense.cpp:1565; ggml/src/ggml-potal/ops.cpp:64; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:97; ggml/src/ggml-potal/tests/potal-log-test.cpp:278; ggml/src/ggml-potal/tests/potal-log-test.cpp:292; ggml/src/ggml-potal/tests/potal-log-test.cpp:417; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-potal-w4a4/ops.cpp:64; ggml/src/ggml-potal-w4a4/ggml-potal.cpp:365; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:97; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:278; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:292; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:417; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:24; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:30; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:46; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:74; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:632; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:904; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:927; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:953; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1012; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1038; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1077; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1249; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1349; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1368; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1379; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1404; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1417; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1432; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1440; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1451; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:86; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:96; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:109; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:165; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:275; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:287; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:442; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1176; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1200; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1429; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1453; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2501; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2534; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2566; ggml/src/ggml-gemmini/residual/residual-capture.hpp:115; ggml/src/ggml-gemmini/residual/residual-capture.hpp:177; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp:200; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp:209; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp:247; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:725; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:729; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:751; ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:142; ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:169; ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:254; ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:466; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:907; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1279; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:14; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:126; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:217; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:273; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:349; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:380; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:394; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:409; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:429; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:469; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:534; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:572; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:584; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:587; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:602; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:609; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:624; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:653; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:38; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:353; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:608; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:628; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:665; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1028; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1037; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1612; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1768; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1849; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:2045; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:43; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:239; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:271; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:321; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:340; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:26; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:67; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:80; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:91; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:98; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:110; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:198; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:206; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:213; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:286; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:337; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:368; ggml/src/ggml-gemmini/matmul/execution.cpp:809; ggml/src/ggml-gemmini/matmul/execution.cpp:836; ggml/src/ggml-gemmini/matmul/execution.cpp:854; ggml/src/ggml-gemmini/matmul/execution.cpp:996; ggml/src/ggml-gemmini/matmul/execution.cpp:1160; ggml/src/ggml-gemmini/matmul/execution.cpp:1200; ggml/src/ggml-gemmini/matmul/execution.cpp:1221; ggml/src/ggml-gemmini/matmul/execution.cpp:1765; ggml/src/ggml-gemmini/matmul/execution.cpp:1782; ggml/src/ggml-gemmini/matmul/types.hpp:198; ggml/src/ggml-gemmini/matmul/types.hpp:201; ggml/src/ggml-gemmini-utils/src/cycle.cpp:7; ggml/src/ggml-gemmini-utils/src/cycle.cpp:163; ggml/src/ggml-gemmini-utils/src/cycle.cpp:169; ggml/src/ggml-gemmini-utils/src/cycle.cpp:180; ggml/src/ggml-gemmini-utils/src/cycle.cpp:237; ggml/src/ggml-gemmini-utils/src/cycle.cpp:253; ggml/src/ggml-gemmini-utils/src/cycle.cpp:590; ggml/src/ggml-gemmini-utils/src/cycle.cpp:974; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1073; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1085; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1121; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1158; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1186; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1195; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1216; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1240; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1264; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1305; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1374; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1507; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1893; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1912; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1936; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1964; ggml/src/ggml-gemmini-utils/src/performance.cpp:429; ggml/src/ggml-gemmini-utils/src/performance.cpp:477; ggml/src/ggml-gemmini-utils/src/performance.cpp:817; ggml/src/ggml-gemmini-utils/src/performance.cpp:860; ggml/src/ggml-gemmini-utils/src/performance.cpp:882; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:13; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:29; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:45; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:57; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:74; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:119; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:187; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:411; ggml/src/ggml-gemmini-utils/src/log-capi.cpp:207; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:5; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:37; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:56; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:7; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:87; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:111; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:8-9; ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp:57-58
- Runtime override: GEMMINI_LOG_DIR and output controls below are runtime paths, not compilation overrides.
- Recommendation: KEEP profiling/experiment control (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:409-410; cmake/ggml-gemmini-options.cmake:415; cmake/ggml-gemmini-options.cmake:422-423; cmake/ggml-gemmini-config.cmake:61; cmake/ggml-gemmini-config.cmake:86; ggml/src/ggml-gemmini-utils/CMakeLists.txt:121; build-x86.sh:100; build-arm64-cpu.sh:84; build-riscv.sh:121; build-arm64.sh:92; scripts/im2p-build-options.py:79-80; scripts/im2p-build-options.py:97-98; ggml/src/CMakeLists.txt:7-8; ggml/src/CMakeLists.txt:294-295; ggml/src/ggml-gemmini/CMakeLists.txt:50; ggml/src/ggml-gemmini/CMakeLists.txt:59; src/CMakeLists.txt:39-40; ggml/src/ggml-cpu/CMakeLists.txt:46; ggml/src/ggml-cpu/CMakeLists.txt:49
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini.cpp:33; ggml/src/ggml-gemmini/ggml-gemmini.cpp:36; ggml/src/ggml-gemmini/ggml-gemmini.cpp:160; ggml/src/ggml-potal/ggml-potal.cpp:365; ggml/src/ggml-gemmini/ops.cpp:30; ggml/src/ggml-gemmini/ops.cpp:33; ggml/src/ggml-gemmini/ops.cpp:94; ggml/src/ggml-gemmini/ops.cpp:997; ggml/src/ggml-gemmini/ops.cpp:1033; ggml/src/ggml-gemmini/ops.cpp:1393; ggml/src/ggml-gemmini/ops.cpp:2231; ggml/src/ggml-gemmini/matmul/dense.cpp:719; ggml/src/ggml-gemmini/matmul/dense.cpp:730; ggml/src/ggml-gemmini/matmul/dense.cpp:775; ggml/src/ggml-gemmini/matmul/dense.cpp:875; ggml/src/ggml-gemmini/matmul/dense.cpp:1422; ggml/src/ggml-gemmini/matmul/dense.cpp:1453; ggml/src/ggml-gemmini/matmul/dense.cpp:1468; ggml/src/ggml-gemmini/matmul/dense.cpp:1472; ggml/src/ggml-gemmini/matmul/dense.cpp:1497; ggml/src/ggml-gemmini/matmul/dense.cpp:1515; ggml/src/ggml-gemmini/matmul/dense.cpp:1539; ggml/src/ggml-gemmini/matmul/dense.cpp:1565; ggml/src/ggml-potal/ops.cpp:62; ggml/src/ggml-potal/ops.cpp:64; ggml/src/ggml-potal-w4a4/ops.cpp:62; ggml/src/ggml-potal-w4a4/ops.cpp:64; ggml/src/ggml-potal-w4a4/ggml-potal.cpp:365; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:24; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:30; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:46; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:74; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:632; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:904; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:927; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:953; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1012; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1038; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1077; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1249; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1349; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1368; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1379; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1404; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1417; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1432; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1440; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1451; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:86; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:96; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:109; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:165; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:275; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:287; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:442; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1176; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1200; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1429; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:1453; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2501; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2534; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2566; ggml/src/ggml-gemmini/residual/residual-capture.hpp:115; ggml/src/ggml-gemmini/residual/residual-capture.hpp:177; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp:200; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp:209; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.hpp:247; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:725; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:729; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:751; ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:142; ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:169; ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:254; ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:466; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:907; ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:1279; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:14; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:126; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:217; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:273; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:349; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:380; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:394; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:409; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:429; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:469; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:534; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:572; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:584; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:587; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:602; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:609; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:624; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:653; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:38; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:353; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:608; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:628; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:665; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1028; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1037; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1612; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1768; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:1849; ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:2045; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:34; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:43-44; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:239; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:271; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:321; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.hpp:340; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:26; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:67; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:80; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:91; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:98; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:110; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:198; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:206; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:213; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:286; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:337; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:368; ggml/src/ggml-gemmini/matmul/execution.cpp:809; ggml/src/ggml-gemmini/matmul/execution.cpp:836; ggml/src/ggml-gemmini/matmul/execution.cpp:854; ggml/src/ggml-gemmini/matmul/execution.cpp:996; ggml/src/ggml-gemmini/matmul/execution.cpp:1160; ggml/src/ggml-gemmini/matmul/execution.cpp:1200; ggml/src/ggml-gemmini/matmul/execution.cpp:1221; ggml/src/ggml-gemmini/matmul/execution.cpp:1765; ggml/src/ggml-gemmini/matmul/execution.cpp:1782; ggml/src/ggml-gemmini/matmul/types.hpp:198; ggml/src/ggml-gemmini/matmul/types.hpp:201; ggml/src/ggml-gemmini-utils/src/cycle.cpp:7; ggml/src/ggml-gemmini-utils/src/cycle.cpp:163; ggml/src/ggml-gemmini-utils/src/cycle.cpp:169; ggml/src/ggml-gemmini-utils/src/cycle.cpp:180; ggml/src/ggml-gemmini-utils/src/cycle.cpp:237; ggml/src/ggml-gemmini-utils/src/cycle.cpp:253; ggml/src/ggml-gemmini-utils/src/cycle.cpp:590; ggml/src/ggml-gemmini-utils/src/cycle.cpp:974; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1073; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1085; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1121; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1158; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1186; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1195; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1216; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1240; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1264; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1305; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1374; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1507; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1893; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1912; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1936; ggml/src/ggml-gemmini-utils/src/cycle.cpp:1964; ggml/src/ggml-gemmini-utils/src/performance.cpp:429; ggml/src/ggml-gemmini-utils/src/performance.cpp:477; ggml/src/ggml-gemmini-utils/src/performance.cpp:817; ggml/src/ggml-gemmini-utils/src/performance.cpp:860; ggml/src/ggml-gemmini-utils/src/performance.cpp:882; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:13; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:29; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:45; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:57; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:74; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:119; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:187; ggml/src/ggml-gemmini-utils/src/trace-context.cpp:411; ggml/src/ggml-gemmini-utils/src/log-capi.cpp:207; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:5; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:37; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:56; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:7; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:87; ggml/src/ggml-gemmini-utils/include/gemmini/cycle_reader.hpp:111; ggml/src/ggml-gemmini-utils/include/gemmini/cpu_log_context.hpp:8-9; ggml/src/ggml-gemmini-utils/include/gemmini/log.h:3; ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp:3; ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp:57-58
- Existing test references: tests/CMakeLists.txt:6; tests/CMakeLists.txt:9; tests/CMakeLists.txt:13; tests/CMakeLists.txt:23; tests/CMakeLists.txt:31; tests/CMakeLists.txt:33; tests/CMakeLists.txt:45; tests/CMakeLists.txt:92; tests/CMakeLists.txt:262; tests/CMakeLists.txt:342; tests/CMakeLists.txt:367; tests/CMakeLists.txt:369; tests/CMakeLists.txt:384; tests/CMakeLists.txt:397; tests/CMakeLists.txt:404; tests/CMakeLists.txt:435; tests/CMakeLists.txt:441; tests/CMakeLists.txt:446; tests/CMakeLists.txt:458; tests/CMakeLists.txt:466; tests/CMakeLists.txt:621; tests/CMakeLists.txt:676; tests/CMakeLists.txt:701; tests/CMakeLists.txt:731; tests/CMakeLists.txt:753; tests/CMakeLists.txt:769; tests/CMakeLists.txt:771; tests/CMakeLists.txt:788; tests/CMakeLists.txt:819; tests/CMakeLists.txt:894; tests/CMakeLists.txt:949; tests/CMakeLists.txt:994; tests/CMakeLists.txt:1080; tests/CMakeLists.txt:1123; tests/CMakeLists.txt:1139; tests/CMakeLists.txt:1159; tests/CMakeLists.txt:1178; tests/CMakeLists.txt:1341; tests/CMakeLists.txt:1362; tests/CMakeLists.txt:1413; tests/CMakeLists.txt:1516; tests/CMakeLists.txt:1616; tests/CMakeLists.txt:1694; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:97; ggml/src/ggml-potal/tests/potal-log-test.cpp:230; ggml/src/ggml-potal/tests/potal-log-test.cpp:278; ggml/src/ggml-potal/tests/potal-log-test.cpp:292; ggml/src/ggml-potal/tests/potal-log-test.cpp:417; ggml/src/ggml-potal/tests/potal-log-test.cpp:463; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:97; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:230; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:278; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:292; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:417; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:463; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:892; tests/test-gemmini-build-contract.cmake:47-48; tests/test-gemmini-build-contract.cmake:922; tests/test-im2p-cache-build-scripts.cmake:489; tests/test-potal-target-profile.cmake:29; tests/test-gemmini-matmul-cpu-boundaries.cmake:250; tests/test-gemmini-matmul-cpu-boundaries.cmake:295; tests/test-gemmini-telemetry-clock-gate.cmake:28

### LOG_DEBUG

- Direct default / definition: "1"; cmake/ggml-gemmini-options.cmake:407.
- Behavior / legal combinations: 0/1; compile-time gate. Dump paths are experiment artifacts, not accelerator transport. Contracts: K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-potal/ops.cpp:465; ggml/src/ggml-potal/ops.cpp:516; ggml/src/ggml-potal/ops.cpp:565; ggml/src/ggml-potal/ops.cpp:645; ggml/src/ggml-potal/ops.cpp:680; ggml/src/ggml-potal/ops.cpp:713; ggml/src/ggml-potal/ops.cpp:758; ggml/src/ggml-potal/ops.cpp:772; ggml/src/ggml-potal/ops.cpp:835; ggml/src/ggml-potal/log.cpp:29; ggml/src/ggml-potal/log.cpp:44; ggml/src/ggml-potal/matmul/stripe.cpp:356; ggml/src/ggml-potal/tests/potal-e4c1-bench.cpp:647; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:97; ggml/src/ggml-potal/tests/potal-log-test.cpp:232; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-potal/tests/potal-stripe-pipeline-test.cpp:1315; ggml/src/ggml-potal-w4a4/ops.cpp:467; ggml/src/ggml-potal-w4a4/ops.cpp:518; ggml/src/ggml-potal-w4a4/ops.cpp:567; ggml/src/ggml-potal-w4a4/ops.cpp:647; ggml/src/ggml-potal-w4a4/ops.cpp:682; ggml/src/ggml-potal-w4a4/ops.cpp:715; ggml/src/ggml-potal-w4a4/ops.cpp:760; ggml/src/ggml-potal-w4a4/ops.cpp:774; ggml/src/ggml-potal-w4a4/ops.cpp:841; ggml/src/ggml-potal-w4a4/log.cpp:29; ggml/src/ggml-potal-w4a4/log.cpp:44; ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:357; ggml/src/ggml-potal-w4a4/tests/potal-e4c1-bench.cpp:647; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:97; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:232; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-potal-w4a4/tests/potal-stripe-pipeline-test.cpp:1316; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:863; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1396; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:193; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2468; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2534; ggml/src/ggml-gemmini-utils/src/debug.cpp:516; ggml/src/ggml-gemmini-utils/src/debug.cpp:528; ggml/src/ggml-gemmini-utils/src/debug.cpp:548; ggml/src/ggml-gemmini-utils/src/debug.cpp:585; ggml/src/ggml-gemmini-utils/src/debug.cpp:598; ggml/src/ggml-gemmini-utils/src/debug.cpp:622; ggml/src/ggml-gemmini-utils/src/debug.cpp:654; ggml/src/ggml-gemmini-utils/src/debug.cpp:670; ggml/src/ggml-gemmini-utils/src/debug.cpp:688; ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp:54-55
- Runtime override: GEMMINI_LOG_DIR and output controls below are runtime paths, not compilation overrides.
- Recommendation: KEEP profiling/experiment control (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:407-408; cmake/ggml-gemmini-options.cmake:415; cmake/ggml-gemmini-config.cmake:60; cmake/ggml-gemmini-config.cmake:85; ggml/src/ggml-gemmini-utils/CMakeLists.txt:120
- Source/macro consumers: ggml/src/ggml-potal/ops.cpp:465; ggml/src/ggml-potal/ops.cpp:516; ggml/src/ggml-potal/ops.cpp:565; ggml/src/ggml-potal/ops.cpp:645; ggml/src/ggml-potal/ops.cpp:680; ggml/src/ggml-potal/ops.cpp:713; ggml/src/ggml-potal/ops.cpp:758; ggml/src/ggml-potal/ops.cpp:772; ggml/src/ggml-potal/ops.cpp:835; ggml/src/ggml-potal/log.cpp:29; ggml/src/ggml-potal/log.cpp:44; ggml/src/ggml-potal/log.hpp:32; ggml/src/ggml-potal/log.hpp:35; ggml/src/ggml-potal/matmul/stripe.cpp:356; ggml/src/ggml-potal-w4a4/ops.cpp:467; ggml/src/ggml-potal-w4a4/ops.cpp:518; ggml/src/ggml-potal-w4a4/ops.cpp:567; ggml/src/ggml-potal-w4a4/ops.cpp:647; ggml/src/ggml-potal-w4a4/ops.cpp:682; ggml/src/ggml-potal-w4a4/ops.cpp:715; ggml/src/ggml-potal-w4a4/ops.cpp:760; ggml/src/ggml-potal-w4a4/ops.cpp:774; ggml/src/ggml-potal-w4a4/ops.cpp:841; ggml/src/ggml-potal-w4a4/log.cpp:29; ggml/src/ggml-potal-w4a4/log.cpp:44; ggml/src/ggml-potal-w4a4/log.hpp:32; ggml/src/ggml-potal-w4a4/log.hpp:35; ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:357; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:863; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1396; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:193; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2468; ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:2534; ggml/src/ggml-gemmini-utils/src/debug.cpp:516; ggml/src/ggml-gemmini-utils/src/debug.cpp:528; ggml/src/ggml-gemmini-utils/src/debug.cpp:548; ggml/src/ggml-gemmini-utils/src/debug.cpp:585; ggml/src/ggml-gemmini-utils/src/debug.cpp:598; ggml/src/ggml-gemmini-utils/src/debug.cpp:622; ggml/src/ggml-gemmini-utils/src/debug.cpp:654; ggml/src/ggml-gemmini-utils/src/debug.cpp:670; ggml/src/ggml-gemmini-utils/src/debug.cpp:688; ggml/src/ggml-gemmini-utils/include/gemmini/log.h:3; ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp:3; ggml/src/ggml-gemmini-utils/include/gemmini/log.hpp:54-55
- Existing test references: tests/CMakeLists.txt:341; tests/CMakeLists.txt:366; tests/CMakeLists.txt:454; tests/CMakeLists.txt:1122; tests/CMakeLists.txt:1159; tests/CMakeLists.txt:1316; ggml/src/ggml-potal/tests/potal-e4c1-bench.cpp:647-648; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:97; ggml/src/ggml-potal/tests/potal-log-test.cpp:230; ggml/src/ggml-potal/tests/potal-log-test.cpp:232; ggml/src/ggml-potal/tests/potal-log-test.cpp:240; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-potal/tests/potal-stripe-pipeline-test.cpp:39; ggml/src/ggml-potal/tests/potal-stripe-pipeline-test.cpp:1315; ggml/src/ggml-potal/tests/potal-stripe-pipeline-test.cpp:1347; ggml/src/ggml-potal-w4a4/tests/potal-e4c1-bench.cpp:647-648; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:97; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:230; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:232; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:240; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:892; ggml/src/ggml-potal-w4a4/tests/potal-stripe-pipeline-test.cpp:39; ggml/src/ggml-potal-w4a4/tests/potal-stripe-pipeline-test.cpp:1316; ggml/src/ggml-potal-w4a4/tests/potal-stripe-pipeline-test.cpp:1348; tests/test-gemmini-build-contract.cmake:1054; tests/test-im2p-cache-build-scripts.cmake:361; tests/test-im2p-cache-build-scripts.cmake:488; tests/test-potal-target-profile.cmake:29

### LOG_DUMP

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:411.
- Behavior / legal combinations: 0/1; compile-time gate. Dump paths are experiment artifacts, not accelerator transport. Contracts: K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/ggml-gemmini.cpp:51; ggml/src/ggml-gemmini/ggml-gemmini.cpp:132; ggml/src/ggml-gemmini/ggml-gemmini.cpp:211; ggml/src/ggml-gemmini/ggml-gemmini.cpp:258; ggml/src/ggml-gemmini/ops.cpp:49; ggml/src/ggml-gemmini/ops.cpp:1063; ggml/src/ggml-gemmini/dump/dump_tensor.hpp:3-4; ggml/src/ggml-gemmini/dump/dump_tensor.hpp:21; ggml/src/ggml-gemmini/dump/dump_tensor.cpp:5; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.hpp:8-9; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:6-7
- Runtime override: GEMMINI_LOG_DIR and output controls below are runtime paths, not compilation overrides.
- Recommendation: KEEP profiling/experiment control (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:411-412; cmake/ggml-gemmini-options.cmake:415; cmake/ggml-gemmini-config.cmake:62; ggml/src/ggml-gemmini-utils/CMakeLists.txt:122
- Source/macro consumers: ggml/src/ggml-gemmini/ggml-gemmini.cpp:51; ggml/src/ggml-gemmini/ggml-gemmini.cpp:132; ggml/src/ggml-gemmini/ggml-gemmini.cpp:211; ggml/src/ggml-gemmini/ggml-gemmini.cpp:258; ggml/src/ggml-gemmini/ops.cpp:49; ggml/src/ggml-gemmini/ops.cpp:1063; ggml/src/ggml-gemmini/dump/dump_tensor.hpp:3-4; ggml/src/ggml-gemmini/dump/dump_tensor.hpp:21; ggml/src/ggml-gemmini/dump/dump_tensor.cpp:5; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.hpp:8-9; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:6-7
- Existing test references: tests/CMakeLists.txt:378; tests/CMakeLists.txt:384; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:99; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:99; tests/test-gemmini-build-contract.cmake:1055

### LOG_DUMP_SCALE

- Direct default / definition: "0"; cmake/ggml-gemmini-options.cmake:413.
- Behavior / legal combinations: 0/1; compile-time gate. Dump paths are experiment artifacts, not accelerator transport. Contracts: K11.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.hpp:7; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.hpp:9; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.hpp:11; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:5; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:7; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:9; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:14; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:64
- Runtime override: GEMMINI_LOG_DIR and output controls below are runtime paths, not compilation overrides.
- Recommendation: KEEP profiling/experiment control (preserve the behavior and constraints above).
- Build use locations: cmake/ggml-gemmini-options.cmake:413-415; cmake/ggml-gemmini-config.cmake:63; ggml/src/ggml-gemmini-utils/CMakeLists.txt:123
- Source/macro consumers: ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.hpp:7; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.hpp:9; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.hpp:11; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:5; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:7; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:9; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:14; ggml/src/ggml-gemmini/dump/scale/scale_dump_dispatch.cpp:64
- Existing test references: tests/CMakeLists.txt:384; tests/CMakeLists.txt:387; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:99; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:99; tests/test-gemmini-build-contract.cmake:1056

### POTAL_API_ONLY

- Direct default / definition: OFF; CMakeLists.txt:169.
- Behavior / legal combinations: Builds potal-api/potal-tiled-api and tests then root returns; independent of ggml/llama SDK command sources. Contracts: K1,K6.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Build-time only.
- Recommendation: KEEP standalone API build mode (preserve the behavior and constraints above).
- Build use locations: cmake/potal-hw-params.cmake:244; ggml/src/ggml-potal/CMakeLists.txt:3; ggml/src/ggml-potal/CMakeLists.txt:26; ggml/src/ggml-potal/CMakeLists.txt:44; ggml/src/ggml-potal/CMakeLists.txt:174; CMakeLists.txt:165; CMakeLists.txt:169-170; CMakeLists.txt:172; CMakeLists.txt:175; CMakeLists.txt:188; build-riscv.sh:65; scripts/im2p-build-options.py:34; scripts/im2p-build-options.py:90; ggml/src/ggml-potal-w4a4/CMakeLists.txt:3; ggml/src/ggml-potal-w4a4/CMakeLists.txt:55; ggml/src/ggml-potal-w4a4/CMakeLists.txt:73; ggml/src/ggml-potal-w4a4/CMakeLists.txt:203
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### POTAL_ATTN_SWEEP

- Direct default / definition: OFF; ggml/src/ggml-potal/CMakeLists.txt:514.
- Behavior / legal combinations: OFF default. Adds the named prototype/acceptance or benchmark target; no evidence of dead production feature. B5_SHARED_BASELINE is defined only inside B5_TESTS. Some executables are deliberately not registered with ctest. Contracts: K6-K8.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Diagnostic environment remains separate; see experiment inventory.
- Recommendation: KEEP isolated experiment/proof target gate (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-potal/CMakeLists.txt:514-515; ggml/src/ggml-potal-w4a4/CMakeLists.txt:562-563
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### POTAL_B5_SHARED_BASELINE

- Direct default / definition: OFF; ggml/src/ggml-potal/CMakeLists.txt:425.
- Behavior / legal combinations: OFF default. Adds the named prototype/acceptance or benchmark target; no evidence of dead production feature. B5_SHARED_BASELINE is defined only inside B5_TESTS. Some executables are deliberately not registered with ctest. Contracts: K6-K8.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-potal/tests/potal-attention-bench.cpp:32; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:120; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:161; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:32; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:120; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:161
- Runtime override: Diagnostic environment remains separate; see experiment inventory.
- Recommendation: KEEP isolated experiment/proof target gate (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-potal/CMakeLists.txt:425; ggml/src/ggml-potal/CMakeLists.txt:433-434; ggml/src/ggml-potal-w4a4/CMakeLists.txt:473; ggml/src/ggml-potal-w4a4/CMakeLists.txt:481-482
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: ggml/src/ggml-potal/tests/potal-attention-bench.cpp:4; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:32; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:120; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:161; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:4; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:32; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:120; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:161

### POTAL_B5_TESTS

- Direct default / definition: OFF; ggml/src/ggml-potal/CMakeLists.txt:423.
- Behavior / legal combinations: OFF default. Adds the named prototype/acceptance or benchmark target; no evidence of dead production feature. B5_SHARED_BASELINE is defined only inside B5_TESTS. Some executables are deliberately not registered with ctest. Contracts: K6-K8.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Diagnostic environment remains separate; see experiment inventory.
- Recommendation: KEEP isolated experiment/proof target gate (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-potal/CMakeLists.txt:423-424; ggml/src/ggml-potal-w4a4/CMakeLists.txt:471-472
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### POTAL_E3_TESTS

- Direct default / definition: OFF; ggml/src/ggml-potal/CMakeLists.txt:418.
- Behavior / legal combinations: OFF default. Adds the named prototype/acceptance or benchmark target; no evidence of dead production feature. B5_SHARED_BASELINE is defined only inside B5_TESTS. Some executables are deliberately not registered with ctest. Contracts: K6-K8.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Diagnostic environment remains separate; see experiment inventory.
- Recommendation: KEEP isolated experiment/proof target gate (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-potal/CMakeLists.txt:418-419; ggml/src/ggml-potal-w4a4/CMakeLists.txt:466-467
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### POTAL_E4A_TESTS

- Direct default / definition: OFF; ggml/src/ggml-potal/CMakeLists.txt:438.
- Behavior / legal combinations: OFF default. Adds the named prototype/acceptance or benchmark target; no evidence of dead production feature. B5_SHARED_BASELINE is defined only inside B5_TESTS. Some executables are deliberately not registered with ctest. Contracts: K6-K8.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Diagnostic environment remains separate; see experiment inventory.
- Recommendation: KEEP isolated experiment/proof target gate (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-potal/CMakeLists.txt:438-439; ggml/src/ggml-potal-w4a4/CMakeLists.txt:486-487
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### POTAL_E4B1_TESTS

- Direct default / definition: OFF; ggml/src/ggml-potal/CMakeLists.txt:467.
- Behavior / legal combinations: OFF default. Adds the named prototype/acceptance or benchmark target; no evidence of dead production feature. B5_SHARED_BASELINE is defined only inside B5_TESTS. Some executables are deliberately not registered with ctest. Contracts: K6-K8.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Diagnostic environment remains separate; see experiment inventory.
- Recommendation: KEEP isolated experiment/proof target gate (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-potal/CMakeLists.txt:467-468; ggml/src/ggml-potal-w4a4/CMakeLists.txt:515-516
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### POTAL_E4B2_TESTS

- Direct default / definition: OFF; ggml/src/ggml-potal/CMakeLists.txt:111.
- Behavior / legal combinations: OFF default. Adds the named prototype/acceptance or benchmark target; no evidence of dead production feature. B5_SHARED_BASELINE is defined only inside B5_TESTS. Some executables are deliberately not registered with ctest. Contracts: K6-K8.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Diagnostic environment remains separate; see experiment inventory.
- Recommendation: KEEP isolated experiment/proof target gate (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-potal/CMakeLists.txt:111-112; ggml/src/ggml-potal/CMakeLists.txt:406; ggml/src/ggml-potal-w4a4/CMakeLists.txt:140-141; ggml/src/ggml-potal-w4a4/CMakeLists.txt:454
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### POTAL_E4C1_TESTS

- Direct default / definition: OFF; ggml/src/ggml-potal/CMakeLists.txt:480.
- Behavior / legal combinations: OFF default. Adds the named prototype/acceptance or benchmark target; no evidence of dead production feature. B5_SHARED_BASELINE is defined only inside B5_TESTS. Some executables are deliberately not registered with ctest. Contracts: K6-K8.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Diagnostic environment remains separate; see experiment inventory.
- Recommendation: KEEP isolated experiment/proof target gate (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-potal/CMakeLists.txt:480-481; ggml/src/ggml-potal-w4a4/CMakeLists.txt:528-529
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### POTAL_E4C2_0_TESTS

- Direct default / definition: OFF; ggml/src/ggml-potal/CMakeLists.txt:128.
- Behavior / legal combinations: OFF default. Adds the named prototype/acceptance or benchmark target; no evidence of dead production feature. B5_SHARED_BASELINE is defined only inside B5_TESTS. Some executables are deliberately not registered with ctest. Contracts: K6-K8.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Diagnostic environment remains separate; see experiment inventory.
- Recommendation: KEEP isolated experiment/proof target gate (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-potal/CMakeLists.txt:128-129; ggml/src/ggml-potal-w4a4/CMakeLists.txt:157-158
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: No option-specific literal test reference confirmed in the read test sources. Consult the registration matrix below for feature-level tests; coverage of every value is UNVERIFIED.

### POTAL_ENABLE_OPENMP

- Direct default / definition: OFF; ggml/src/ggml-potal/CMakeLists.txt:14.
- Behavior / legal combinations: Only adds C OpenMP worker mode to multicore smoke; distinct from GGML_GEMMINI_ENABLE_OPENMP. Contracts: K10.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: OMP_* controls smoke workers; not NPU geometry.
- Recommendation: KEEP separate smoke worker option (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-potal/CMakeLists.txt:14; ggml/src/ggml-potal/CMakeLists.txt:151; ggml/src/ggml-potal/CMakeLists.txt:158-159; ggml/src/ggml-potal-w4a4/CMakeLists.txt:14; ggml/src/ggml-potal-w4a4/CMakeLists.txt:180; ggml/src/ggml-potal-w4a4/CMakeLists.txt:187-188
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: ggml/src/ggml-potal/tests/potal-multicore-smoke.c:812; ggml/src/ggml-potal-w4a4/tests/potal-multicore-smoke.c:812

### POTAL_EXPECTED_DIM

- Direct default / definition: ""; ggml/src/ggml-potal/CMakeLists.txt:12.
- Behavior / legal combinations: Required empty-default assertion against selected header; must still check target mismatch. Contracts: K6,K7.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Build-time only.
- Recommendation: MERGE software expectation into NPU_D (preserve the behavior and constraints above).
- Build use locations: cmake/potal-hw-params.cmake:9; cmake/potal-hw-params.cmake:93-94; cmake/potal-hw-params.cmake:134-135; cmake/potal-hw-params.cmake:222; ggml/src/ggml-potal/CMakeLists.txt:12; CMakeLists.txt:186; build-x86.sh:12; build-riscv.sh:15; scripts/im2p-build-options.py:89; scripts/im2p-build-options.py:91; ggml/src/ggml-potal-w4a4/CMakeLists.txt:12
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/test-im2p-cache-build-scripts.cmake:201; tests/test-potal-target-profile.cmake:32

### POTAL_HW_CONFIG

- Direct default / definition: ""; ggml/src/ggml-potal/CMakeLists.txt:13.
- Behavior / legal combinations: Empty default; not merely recorded: recognized FireSimQuadRocketSinglePoTalD16/32Config names also validate DIM. Contracts: K6.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Build-time only.
- Recommendation: KEEP provenance/config assertion (preserve the behavior and constraints above).
- Build use locations: cmake/potal-hw-params.cmake:9; cmake/potal-hw-params.cmake:145; cmake/potal-hw-params.cmake:147; cmake/potal-hw-params.cmake:210-211; ggml/src/ggml-potal/CMakeLists.txt:13; ggml/src/ggml-potal-w4a4/CMakeLists.txt:13
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/test-potal-target-profile.cmake:33

### POTAL_HW_PARAMS

- Direct default / definition: ""; ggml/src/ggml-potal/CMakeLists.txt:11.
- Behavior / legal combinations: Absolute external RTL header required, adapter and byte-exact copy generated. Contracts: K6,K7.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Build-time env forwarding via wrappers; no runtime override.
- Recommendation: MERGE hardware inputs into generated NPU_*; KEEP provenance (preserve the behavior and constraints above).
- Build use locations: cmake/potal-hw-params.cmake:9; cmake/potal-hw-params.cmake:23; cmake/potal-hw-params.cmake:26; cmake/potal-hw-params.cmake:29; cmake/potal-hw-params.cmake:48; cmake/potal-hw-params.cmake:84-85; cmake/potal-hw-params.cmake:87-88; cmake/potal-hw-params.cmake:90-91; cmake/potal-hw-params.cmake:97; cmake/potal-hw-params.cmake:219; ggml/src/ggml-potal/CMakeLists.txt:11; ggml/src/ggml-potal/CMakeLists.txt:13; CMakeLists.txt:168; CMakeLists.txt:186; CMakeLists.txt:198; build-x86.sh:12; build-riscv.sh:15; scripts/im2p-build-options.py:89; scripts/im2p-build-options.py:91; ggml/src/ggml-potal-w4a4/CMakeLists.txt:11; ggml/src/ggml-potal-w4a4/CMakeLists.txt:13; ggml/src/ggml-potal-w4a4/CMakeLists.txt:20; ggml/src/ggml-potal-w4a4/CMakeLists.txt:36
- Source/macro consumers: ggml/src/ggml-potal/include/potal.h:7; ggml/src/ggml-potal-w4a4/include/potal.h:7
- Existing test references: tests/test-potal-target-profile.cmake:42

### POTAL_PLANNING_DIAGNOSTICS

- Direct default / definition: OFF; ggml/src/ggml-potal/CMakeLists.txt:228.
- Behavior / legal combinations: GNU/Linux linker wrapping prerequisite. Contracts: K17.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: ggml/src/ggml-potal/matmul/paired-auto.cpp:9; ggml/src/ggml-potal/matmul/paired-auto.cpp:61; ggml/src/ggml-potal/matmul/paired-auto.cpp:153; ggml/src/ggml-potal/matmul/paired-auto.cpp:176; ggml/src/ggml-potal/matmul/paired-auto.cpp:215; ggml/src/ggml-potal/matmul/paired-auto.cpp:239; ggml/src/ggml-potal/matmul/paired-auto.cpp:289; ggml/src/ggml-potal/matmul/paired-auto.cpp:352; ggml/src/ggml-potal/matmul/paired-auto.cpp:445; ggml/src/ggml-potal/matmul/paired-auto.cpp:449; ggml/src/ggml-potal/matmul/paired-auto.cpp:692; ggml/src/ggml-potal/matmul/paired-auto.cpp:726; ggml/src/ggml-potal/matmul/paired-auto.cpp:731; ggml/src/ggml-potal/matmul/paired-auto.cpp:735; ggml/src/ggml-potal/matmul/paired-auto.cpp:743; ggml/src/ggml-potal/matmul/paired-auto.cpp:747; ggml/src/ggml-potal/matmul/paired-auto.cpp:1591; ggml/src/ggml-potal/matmul/paired-auto.cpp:3787; ggml/src/ggml-potal/matmul/paired-auto.cpp:4103; ggml/src/ggml-potal/matmul/paired-auto.cpp:4140; ggml/src/ggml-potal/matmul/paired-auto.cpp:4197; ggml/src/ggml-potal/matmul/paired-auto.cpp:4235; ggml/src/ggml-potal/matmul/paired-auto.cpp:4245; ggml/src/ggml-potal/matmul/paired-auto.cpp:4255; ggml/src/ggml-potal/matmul/paired-auto.cpp:4259; ggml/src/ggml-potal/matmul/paired-auto.cpp:4267; ggml/src/ggml-potal/matmul/paired-auto.cpp:4271; ggml/src/ggml-potal/matmul/paired-auto.cpp:4286; ggml/src/ggml-potal/matmul/planning-profile.hpp:5; ggml/src/ggml-potal/matmul/planning-profile.cpp:3; ggml/src/ggml-potal/tests/potal-planning-profile-test.cpp:5; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:48; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:200; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:211; ggml/src/ggml-potal-w4a4/matmul/planning-profile.hpp:5; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:10; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:62; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:154; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:177; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:216; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:240; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:290; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:353; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:446; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:450; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:693; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:727; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:732; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:736; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:744; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:748; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:1628; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:3846; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4162; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4202; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4270; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4308; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4318; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4328; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4332; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4340; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4344; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4359; ggml/src/ggml-potal-w4a4/matmul/planning-profile.cpp:3; ggml/src/ggml-potal-w4a4/tests/potal-planning-profile-test.cpp:5; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:48; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:200; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:211
- Runtime override: POTAL_PLANNING_PROFILE runtime output/timing control (ggml/src/ggml-potal/matmul/planning-profile.cpp:38).
- Recommendation: KEEP profiling (preserve the behavior and constraints above).
- Build use locations: ggml/src/ggml-potal/CMakeLists.txt:228-229; ggml/src/ggml-potal/CMakeLists.txt:231; ggml/src/ggml-potal/CMakeLists.txt:320; ggml/src/ggml-potal/CMakeLists.txt:335; ggml/src/ggml-potal/CMakeLists.txt:370; ggml/src/ggml-potal/CMakeLists.txt:650; ggml/src/ggml-potal-w4a4/CMakeLists.txt:266-267; ggml/src/ggml-potal-w4a4/CMakeLists.txt:269; ggml/src/ggml-potal-w4a4/CMakeLists.txt:358; ggml/src/ggml-potal-w4a4/CMakeLists.txt:373; ggml/src/ggml-potal-w4a4/CMakeLists.txt:410; ggml/src/ggml-potal-w4a4/CMakeLists.txt:698
- Source/macro consumers: ggml/src/ggml-potal/matmul/paired-auto.cpp:9; ggml/src/ggml-potal/matmul/paired-auto.cpp:61; ggml/src/ggml-potal/matmul/paired-auto.cpp:153; ggml/src/ggml-potal/matmul/paired-auto.cpp:176; ggml/src/ggml-potal/matmul/paired-auto.cpp:215; ggml/src/ggml-potal/matmul/paired-auto.cpp:239; ggml/src/ggml-potal/matmul/paired-auto.cpp:289; ggml/src/ggml-potal/matmul/paired-auto.cpp:352; ggml/src/ggml-potal/matmul/paired-auto.cpp:445; ggml/src/ggml-potal/matmul/paired-auto.cpp:449; ggml/src/ggml-potal/matmul/paired-auto.cpp:692; ggml/src/ggml-potal/matmul/paired-auto.cpp:726; ggml/src/ggml-potal/matmul/paired-auto.cpp:731; ggml/src/ggml-potal/matmul/paired-auto.cpp:735; ggml/src/ggml-potal/matmul/paired-auto.cpp:743; ggml/src/ggml-potal/matmul/paired-auto.cpp:747; ggml/src/ggml-potal/matmul/paired-auto.cpp:1591; ggml/src/ggml-potal/matmul/paired-auto.cpp:3787; ggml/src/ggml-potal/matmul/paired-auto.cpp:4103; ggml/src/ggml-potal/matmul/paired-auto.cpp:4140; ggml/src/ggml-potal/matmul/paired-auto.cpp:4197; ggml/src/ggml-potal/matmul/paired-auto.cpp:4235; ggml/src/ggml-potal/matmul/paired-auto.cpp:4245; ggml/src/ggml-potal/matmul/paired-auto.cpp:4255; ggml/src/ggml-potal/matmul/paired-auto.cpp:4259; ggml/src/ggml-potal/matmul/paired-auto.cpp:4267; ggml/src/ggml-potal/matmul/paired-auto.cpp:4271; ggml/src/ggml-potal/matmul/paired-auto.cpp:4286; ggml/src/ggml-potal/matmul/planning-profile.hpp:5; ggml/src/ggml-potal/matmul/planning-profile.hpp:159; ggml/src/ggml-potal/matmul/planning-profile.cpp:3; ggml/src/ggml-potal/matmul/planning-profile.cpp:328; ggml/src/ggml-potal-w4a4/matmul/planning-profile.hpp:5; ggml/src/ggml-potal-w4a4/matmul/planning-profile.hpp:159; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:10; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:62; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:154; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:177; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:216; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:240; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:290; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:353; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:446; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:450; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:693; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:727; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:732; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:736; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:744; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:748; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:1628; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:3846; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4162; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4202; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4270; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4308; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4318; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4328; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4332; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4340; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4344; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4359; ggml/src/ggml-potal-w4a4/matmul/planning-profile.cpp:3; ggml/src/ggml-potal-w4a4/matmul/planning-profile.cpp:328
- Existing test references: ggml/src/ggml-potal/tests/potal-planning-profile-test.cpp:5; ggml/src/ggml-potal/tests/potal-planning-profile-test.cpp:452; ggml/src/ggml-potal/tests/potal-attention-bench.cpp:100; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:48; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:200; ggml/src/ggml-potal/tests/potal-paired-smoke.cpp:211; ggml/src/ggml-potal-w4a4/tests/potal-planning-profile-test.cpp:5; ggml/src/ggml-potal-w4a4/tests/potal-planning-profile-test.cpp:452; ggml/src/ggml-potal-w4a4/tests/potal-attention-bench.cpp:100; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:48; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:200; ggml/src/ggml-potal-w4a4/tests/potal-paired-smoke.cpp:211; tests/test-potal-target-profile.cmake:29

### POTAL_TARGET_DIM

- Direct default / definition: unset; optional target assertion; cmake/potal-hw-params.cmake:140.
- Behavior / legal combinations: Optional wrapper-supplied fixed target DIM16/32. Contracts: K6.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Build-time only.
- Recommendation: KEEP target assertion against NPU_D (preserve the behavior and constraints above).
- Build use locations: cmake/potal-hw-params.cmake:140-142; cmake/potal-hw-params.cmake:223; CMakeLists.txt:187
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/test-potal-target-profile.cmake:32

### POTAL_TARGET_PARAMS

- Direct default / definition: unset; optional header SHA assertion; cmake/potal-hw-params.cmake:101.
- Behavior / legal combinations: Optional target-header SHA equality; not a replacement runtime header path. Contracts: K6.
- Actually used: YES, static configuration/export or source references below. Compile branches / numeric selection: configured by CMake/generation; no literal preprocessor branch confirmed in the read source corpus.
- Runtime override: Build-time only.
- Recommendation: KEEP target provenance assertion (preserve the behavior and constraints above).
- Build use locations: cmake/potal-hw-params.cmake:101-104; cmake/potal-hw-params.cmake:106; cmake/potal-hw-params.cmake:108; cmake/potal-hw-params.cmake:110; cmake/potal-hw-params.cmake:224; CMakeLists.txt:187
- Source/macro consumers: No direct literal C/C++ consumer confirmed; value is consumed in the build contract above. External/generated downstream effect UNVERIFIED where applicable.
- Existing test references: tests/test-potal-target-profile.cmake:43

## Wrapper defaults and shell-only controls

The resolver precedence is command-line > ordinary environment > NAME_DEFAULT environment > script default; IM2P_DIM is resolved at each precedence level and same-level conflicts are errors (scripts/im2p-build-options.py:42-77). Names with GGML_/IM2P_/LOG_/CYCLE_/CMAKE_/LLAMA_/POTAL_ prefixes are configurable; BUILD_DIR/BUILD_JOBS/IM2P_ARTIFACT_SET/IM2P_CACHE_JOBS/APPLE_SILICON_ARCH/LIBOMP_PREFIX are shell controls, not CMake cache options (scripts/im2p-build-options.py:16-24). IM2P_ARTIFACT_SET allows SELECTED or ALL_MATCHED; job counts must be positive, and ALL_MATCHED defaults cache parallelism to 1 (scripts/im2p-build-options.py:164-174). KEEP these provisioning controls separate from generated NPU geometry. The following is the complete wrapper default-assignment ledger, including upstream platform dependency knobs rather than misclassifying them as new NPU APIs.

| Wrapper default assignment | Evidence |
|---|---|
| export GGML_POTAL_DEFAULT="${GGML_POTAL_DEFAULT:-ON}" | build-potal-firesim.sh:28 |
| export GGML_GEMMINI_DEFAULT="${GGML_GEMMINI_DEFAULT:-OFF}" | build-potal-firesim.sh:29 |
| export GGML_BACKEND_DL_DEFAULT="${GGML_BACKEND_DL_DEFAULT:-OFF}" | build-potal-firesim.sh:30 |
| export GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT="${GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT:-HARDWARE}" | build-potal-firesim.sh:31 |
| export GGML_GEMMINI_OPTION_DEFAULT="${GGML_GEMMINI_OPTION_DEFAULT:-WS}" | build-potal-firesim.sh:32 |
| export GGML_GEMMINI_ACTIVATION_QUANT_DEFAULT="${GGML_GEMMINI_ACTIVATION_QUANT_DEFAULT:-EXSIA}" | build-potal-firesim.sh:33 |
| export GGML_GEMMINI_ENABLE_RMD_DEFAULT="${GGML_GEMMINI_ENABLE_RMD_DEFAULT:-ON}" | build-potal-firesim.sh:34 |
| export GGML_GEMMINI_DEFAULT_RMD_BACKEND_DEFAULT="${GGML_GEMMINI_DEFAULT_RMD_BACKEND_DEFAULT:-WS}" | build-potal-firesim.sh:35 |
| export GGML_GEMMINI_DEFAULT_MATMUL_MODE_DEFAULT="${GGML_GEMMINI_DEFAULT_MATMUL_MODE_DEFAULT:-STRIPE_PIPELINE}" | build-potal-firesim.sh:36 |
| export GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT="${GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT:-2}" | build-potal-firesim.sh:37 |
| export GGML_POTAL_PAIRED_AUTO_DEFAULT="${GGML_POTAL_PAIRED_AUTO_DEFAULT:-ON}" | build-potal-firesim.sh:38 |
| export GGML_GEMMINI_EXSIA_DEFAULT_MODE_DEFAULT="${GGML_GEMMINI_EXSIA_DEFAULT_MODE_DEFAULT:-SEQUENTIAL}" | build-potal-firesim.sh:39 |
| export GGML_GEMMINI_ACTIVATION_BITS_DEFAULT="${GGML_GEMMINI_ACTIVATION_BITS_DEFAULT:-8}" | build-potal-firesim.sh:40 |
| export GGML_GEMMINI_WEIGHT_BITS_DEFAULT="${GGML_GEMMINI_WEIGHT_BITS_DEFAULT:-8}" | build-potal-firesim.sh:41 |
| export GGML_GEMMINI_BLOCK_SIZE_DEFAULT="${GGML_GEMMINI_BLOCK_SIZE_DEFAULT:-32}" | build-potal-firesim.sh:42 |
| export CYCLE_SIM_DEFAULT="${CYCLE_SIM_DEFAULT:-0}" CYCLE_DETAIL_DEFAULT="${CYCLE_DETAIL_DEFAULT:-0}" | build-potal-firesim.sh:43 |
| export POTAL_PLANNING_DIAGNOSTICS_DEFAULT="${POTAL_PLANNING_DIAGNOSTICS_DEFAULT:-OFF}" | build-potal-firesim.sh:44 |
| export LOG_DEBUG_DEFAULT="${LOG_DEBUG_DEFAULT:-$debug}" LOG_CYCLE_DEFAULT="${LOG_CYCLE_DEFAULT:-1}" | build-potal-firesim.sh:45 |
| export GGML_CPU_CYCLE_LOG_DEFAULT="${GGML_CPU_CYCLE_LOG_DEFAULT:-OFF}" | build-potal-firesim.sh:46 |
| export GGML_GEMMINI_ENABLE_A4NKS_ATTENTION_DEFAULT="${GGML_GEMMINI_ENABLE_A4NKS_ATTENTION_DEFAULT:-OFF}" | build-potal-firesim.sh:47 |
| export GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION_DEFAULT="${GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION_DEFAULT:-OFF}" | build-potal-firesim.sh:48 |
| export POTAL_EXPECTED_DIM_DEFAULT="${POTAL_EXPECTED_DIM_DEFAULT:-$dim}" | build-potal-firesim.sh:49 |
| export GGML_GEMMINI_DIM_DEFAULT="${GGML_GEMMINI_DIM_DEFAULT:-$dim}" | build-potal-firesim.sh:50 |
| export POTAL_HW_PARAMS_DEFAULT="${POTAL_HW_PARAMS_DEFAULT:-$SCRIPT_ROOT/../../gemmini-rocc-tests/include/$header}" | build-potal-firesim.sh:51 |
| export POTAL_HW_CONFIG_DEFAULT="${POTAL_HW_CONFIG_DEFAULT:-FireSimQuadRocketSinglePoTalD${dim}Config}" | build-potal-firesim.sh:52 |
| BUILD_DIR=${BUILD_DIR:-${BUILD_DIR_DEFAULT:-build-x86}} | build-x86.sh:27 |
| BUILD_JOBS_DEFAULT=${BUILD_JOBS:-${BUILD_JOBS_DEFAULT:-$(im2p_host_jobs)}} | build-x86.sh:28 |
| IM2P_ARTIFACT_SET_DEFAULT=${IM2P_ARTIFACT_SET_DEFAULT:-SELECTED} | build-x86.sh:29 |
| IM2P_CACHE_JOBS_DEFAULT=${IM2P_CACHE_JOBS_DEFAULT:-} | build-x86.sh:30 |
| GGML_CPU_DEFAULT=${GGML_CPU_DEFAULT:-ON} | build-x86.sh:33 |
| GGML_GEMMINI_DEFAULT=${GGML_GEMMINI_DEFAULT:-OFF} | build-x86.sh:34 |
| GGML_POTAL_DEFAULT=${GGML_POTAL_DEFAULT:-ON} | build-x86.sh:35 |
| GGML_METAL_DEFAULT=${GGML_METAL_DEFAULT:-ON} | build-x86.sh:37 |
| GGML_BLAS_DEFAULT=${GGML_BLAS_DEFAULT:-ON} | build-x86.sh:38 |
| GGML_METAL_DEFAULT=${GGML_METAL_DEFAULT:-OFF} | build-x86.sh:40 |
| GGML_BLAS_DEFAULT=${GGML_BLAS_DEFAULT:-OFF} | build-x86.sh:41 |
| GGML_CUDA_DEFAULT=${GGML_CUDA_DEFAULT:-OFF} | build-x86.sh:43 |
| GGML_VULKAN_DEFAULT=${GGML_VULKAN_DEFAULT:-OFF} | build-x86.sh:44 |
| GGML_OPENCL_DEFAULT=${GGML_OPENCL_DEFAULT:-OFF} | build-x86.sh:45 |
| GGML_RPC_DEFAULT=${GGML_RPC_DEFAULT:-OFF} | build-x86.sh:46 |
| GGML_BACKEND_DL_DEFAULT=${GGML_BACKEND_DL_DEFAULT:-OFF} | build-x86.sh:47 |
| POTAL_HW_PARAMS_DEFAULT=${POTAL_HW_PARAMS_DEFAULT:-"$SCRIPT_ROOT/../../gemmini-rocc-tests/include/gemmini_potal_params.h"} | build-x86.sh:51 |
| POTAL_EXPECTED_DIM_DEFAULT=${POTAL_EXPECTED_DIM_DEFAULT:-16} | build-x86.sh:52 |
| GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT=${GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT:-IM2P_SIM} | build-x86.sh:55 |
| GGML_GEMMINI_OPTION_DEFAULT=${GGML_GEMMINI_OPTION_DEFAULT:-WS} | build-x86.sh:56 |
| GGML_NATIVE_DEFAULT=${GGML_NATIVE_DEFAULT:-ON} | build-x86.sh:57 |
| GGML_OPENMP_DEFAULT=${GGML_OPENMP_DEFAULT:-ON} | build-x86.sh:58 |
| GGML_GEMMINI_ENABLE_OPENMP_DEFAULT=${GGML_GEMMINI_ENABLE_OPENMP_DEFAULT:-ON} | build-x86.sh:59 |
| LLAMA_BUILD_TESTS_DEFAULT=${LLAMA_BUILD_TESTS_DEFAULT:-ON} | build-x86.sh:60 |
| LLAMA_CURL_DEFAULT=${LLAMA_CURL_DEFAULT:-OFF} | build-x86.sh:61 |
| CMAKE_BUILD_TYPE_DEFAULT=${CMAKE_BUILD_TYPE_DEFAULT:-Release} | build-x86.sh:62 |
| BUILD_SHARED_LIBS_DEFAULT=${BUILD_SHARED_LIBS_DEFAULT:-ON} | build-x86.sh:63 |
| GGML_GEMMINI_COMPUTE_TYPE_DEFAULT=${GGML_GEMMINI_COMPUTE_TYPE_DEFAULT:-INT} | build-x86.sh:66 |
| GGML_GEMMINI_DEQUANT_FP_TEST_DEFAULT=${GGML_GEMMINI_DEQUANT_FP_TEST_DEFAULT:-OFF} | build-x86.sh:67 |
| GGML_GEMMINI_ACTIVATION_BITS_DEFAULT=${GGML_GEMMINI_ACTIVATION_BITS_DEFAULT:-8} | build-x86.sh:68 |
| GGML_GEMMINI_WEIGHT_BITS_DEFAULT=${GGML_GEMMINI_WEIGHT_BITS_DEFAULT:-8} | build-x86.sh:69 |
| GGML_GEMMINI_DIM_DEFAULT=${GGML_GEMMINI_DIM_DEFAULT:-16} | build-x86.sh:70 |
| GGML_GEMMINI_BLOCK_SIZE_DEFAULT=${GGML_GEMMINI_BLOCK_SIZE_DEFAULT:-32} | build-x86.sh:71 |
| GGML_GEMMINI_PRINT_TILE_DEFAULT=${GGML_GEMMINI_PRINT_TILE_DEFAULT:-0} | build-x86.sh:72 |
| GGML_GEMMINI_FULL_C_DEFAULT=${GGML_GEMMINI_FULL_C_DEFAULT:-1} | build-x86.sh:73 |
| GGML_GEMMINI_LOW_D_DEFAULT=${GGML_GEMMINI_LOW_D_DEFAULT:-0} | build-x86.sh:74 |
| GGML_GEMMINI_TRANSPOSE_B_DEFAULT=${GGML_GEMMINI_TRANSPOSE_B_DEFAULT:-1} | build-x86.sh:75 |
| GGML_GEMMINI_WS_DEBUG_DEFAULT=${GGML_GEMMINI_WS_DEBUG_DEFAULT:-0} | build-x86.sh:76 |
| GGML_GEMMINI_WS_LOOP_CYCLE_DEFAULT=${GGML_GEMMINI_WS_LOOP_CYCLE_DEFAULT:-0} | build-x86.sh:77 |
| GGML_GEMMINI_DISABLE_WS_REUSE_DEFAULT=${GGML_GEMMINI_DISABLE_WS_REUSE_DEFAULT:-0} | build-x86.sh:78 |
| GGML_GEMMINI_ENABLE_A4NKS_ATTENTION_DEFAULT=${GGML_GEMMINI_ENABLE_A4NKS_ATTENTION_DEFAULT:-OFF} | build-x86.sh:81 |
| GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION_DEFAULT=${GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION_DEFAULT:-OFF} | build-x86.sh:82 |
| GGML_GEMMINI_ENABLE_STRIPE_MATMUL_DEFAULT=${GGML_GEMMINI_ENABLE_STRIPE_MATMUL_DEFAULT:-ON} | build-x86.sh:83 |
| GGML_GEMMINI_ENABLE_STRIPE_PIPELINE_DEFAULT=${GGML_GEMMINI_ENABLE_STRIPE_PIPELINE_DEFAULT:-ON} | build-x86.sh:84 |
| GGML_GEMMINI_DEFAULT_MATMUL_MODE_DEFAULT=${GGML_GEMMINI_DEFAULT_MATMUL_MODE_DEFAULT:-STRIPE_PIPELINE} | build-x86.sh:85 |
| GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT=${GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT:-2} | build-x86.sh:86 |
| GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE_DEFAULT=${GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE_DEFAULT:-ON} | build-x86.sh:87 |
| GGML_GEMMINI_ENABLE_RMD_DEFAULT=${GGML_GEMMINI_ENABLE_RMD_DEFAULT:-OFF} | build-x86.sh:90 |
| GGML_GEMMINI_DEFAULT_RMD_BACKEND_DEFAULT=${GGML_GEMMINI_DEFAULT_RMD_BACKEND_DEFAULT:-CPU} | build-x86.sh:91 |
| GGML_GEMMINI_ACTIVATION_QUANT_DEFAULT=${GGML_GEMMINI_ACTIVATION_QUANT_DEFAULT:-TENSOR} # HP1 validation setting; all five are supported | build-x86.sh:94 |
| GGML_GEMMINI_EXSIA_SIGMA_DEFAULT=${GGML_GEMMINI_EXSIA_SIGMA_DEFAULT:-2} | build-x86.sh:95 |
| GGML_GEMMINI_EXSIA_DEFAULT_MODE_DEFAULT=${GGML_GEMMINI_EXSIA_DEFAULT_MODE_DEFAULT:-SEQUENTIAL} | build-x86.sh:96 |
| GGML_GEMMINI_EXSIA_LOCAL_WORKERS_DEFAULT=${GGML_GEMMINI_EXSIA_LOCAL_WORKERS_DEFAULT:-4} | build-x86.sh:97 |
| GGML_GEMMINI_EXSIA_PROFILE_SCOPE_DEFAULT=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_DEFAULT:-OFF} | build-x86.sh:98 |
| LOG_DEBUG_DEFAULT=${LOG_DEBUG_DEFAULT:-1} | build-x86.sh:101 |
| LOG_CYCLE_DEFAULT=${LOG_CYCLE_DEFAULT:-1} | build-x86.sh:102 |
| LOG_DUMP_DEFAULT=${LOG_DUMP_DEFAULT:-0} | build-x86.sh:103 |
| LOG_DUMP_SCALE_DEFAULT=${LOG_DUMP_SCALE_DEFAULT:-0} | build-x86.sh:104 |
| CYCLE_DETAIL_DEFAULT=${CYCLE_DETAIL_DEFAULT:-0} | build-x86.sh:105 |
| CYCLE_SIM_DEFAULT=${CYCLE_SIM_DEFAULT:-0} | build-x86.sh:106 |
| GGML_CPU_CYCLE_LOG_DEFAULT=${GGML_CPU_CYCLE_LOG_DEFAULT:-$LOG_CYCLE_DEFAULT} | build-x86.sh:107 |
| GGML_GEMMINI_ACT_QUANT_METRICS_DEFAULT=${GGML_GEMMINI_ACT_QUANT_METRICS_DEFAULT:-0} | build-x86.sh:108 |
| GGML_GEMMINI_RESIDUAL_METRICS_DEFAULT=${GGML_GEMMINI_RESIDUAL_METRICS_DEFAULT:-0} | build-x86.sh:109 |
| GGML_GEMMINI_SCALE_METRICS_DEFAULT=${GGML_GEMMINI_SCALE_METRICS_DEFAULT:-0} | build-x86.sh:110 |
| IM2P_SIM_ROOT_DEFAULT=${IM2P_SIM_ROOT_DEFAULT:-"$SCRIPT_ROOT/../IM2P.sim"} | build-x86.sh:113 |
| IM2P_SIM_BUILD_DIR_DEFAULT=${IM2P_SIM_BUILD_DIR_DEFAULT:-} | build-x86.sh:114 |
| IM2P_SIM_IMPLEMENTATION_DEFAULT=${IM2P_SIM_IMPLEMENTATION_DEFAULT:-GEMMINI_HP1} | build-x86.sh:115 |
| BUILD_DIR=${BUILD_DIR:-${BUILD_DIR_DEFAULT:-build-arm64-cpu}} | build-arm64-cpu.sh:20 |
| BUILD_JOBS_DEFAULT=${BUILD_JOBS:-${BUILD_JOBS_DEFAULT:-$(im2p_host_jobs)}} | build-arm64-cpu.sh:21 |
| IM2P_ARTIFACT_SET_DEFAULT=${IM2P_ARTIFACT_SET_DEFAULT:-SELECTED} | build-arm64-cpu.sh:22 |
| IM2P_CACHE_JOBS_DEFAULT=${IM2P_CACHE_JOBS_DEFAULT:-} | build-arm64-cpu.sh:23 |
| APPLE_SILICON_ARCH_DEFAULT=${APPLE_SILICON_ARCH_DEFAULT:-arm64} | build-arm64-cpu.sh:24 |
| LIBOMP_PREFIX_DEFAULT=${LIBOMP_PREFIX_DEFAULT:-} | build-arm64-cpu.sh:25 |
| GGML_GEMMINI_DEFAULT=${GGML_GEMMINI_DEFAULT:-ON} | build-arm64-cpu.sh:28 |
| GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT=${GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT:-HARDWARE} | build-arm64-cpu.sh:29 |
| GGML_GEMMINI_OPTION_DEFAULT=${GGML_GEMMINI_OPTION_DEFAULT:-CPU} | build-arm64-cpu.sh:30 |
| GGML_NATIVE_DEFAULT=${GGML_NATIVE_DEFAULT:-ON} | build-arm64-cpu.sh:31 |
| GGML_OPENMP_DEFAULT=${GGML_OPENMP_DEFAULT:-ON} | build-arm64-cpu.sh:32 |
| GGML_GEMMINI_ENABLE_OPENMP_DEFAULT=${GGML_GEMMINI_ENABLE_OPENMP_DEFAULT:-ON} | build-arm64-cpu.sh:33 |
| GGML_CUDA_DEFAULT=${GGML_CUDA_DEFAULT:-OFF} | build-arm64-cpu.sh:34 |
| GGML_BACKEND_DL_DEFAULT=${GGML_BACKEND_DL_DEFAULT:-ON} | build-arm64-cpu.sh:35 |
| LLAMA_BUILD_TESTS_DEFAULT=${LLAMA_BUILD_TESTS_DEFAULT:-OFF} | build-arm64-cpu.sh:36 |
| LLAMA_CURL_DEFAULT=${LLAMA_CURL_DEFAULT:-OFF} | build-arm64-cpu.sh:37 |
| CMAKE_BUILD_TYPE_DEFAULT=${CMAKE_BUILD_TYPE_DEFAULT:-Release} | build-arm64-cpu.sh:38 |
| BUILD_SHARED_LIBS_DEFAULT=${BUILD_SHARED_LIBS_DEFAULT:-ON} | build-arm64-cpu.sh:39 |
| GGML_METAL_DEFAULT=${GGML_METAL_DEFAULT:-OFF} | build-arm64-cpu.sh:40 |
| GGML_BLAS_DEFAULT=${GGML_BLAS_DEFAULT:-OFF} | build-arm64-cpu.sh:41 |
| LIBOMP_PREFIX_DEFAULT=${LIBOMP_PREFIX:-${LIBOMP_PREFIX_DEFAULT:-$(im2p_host_libomp_prefix)}} | build-arm64-cpu.sh:43 |
| CMAKE_OSX_ARCHITECTURES_DEFAULT=${CMAKE_OSX_ARCHITECTURES_DEFAULT:-"${APPLE_SILICON_ARCH:-$APPLE_SILICON_ARCH_DEFAULT}"} | build-arm64-cpu.sh:44 |
| OpenMP_ROOT_DEFAULT=${OpenMP_ROOT_DEFAULT:-"$LIBOMP_PREFIX_DEFAULT"} | build-arm64-cpu.sh:46 |
| CMAKE_PREFIX_PATH_DEFAULT=${CMAKE_PREFIX_PATH_DEFAULT:-"$LIBOMP_PREFIX_DEFAULT"} | build-arm64-cpu.sh:47 |
| GGML_GEMMINI_COMPUTE_TYPE_DEFAULT=${GGML_GEMMINI_COMPUTE_TYPE_DEFAULT:-INT} | build-arm64-cpu.sh:52 |
| GGML_GEMMINI_DEQUANT_FP_TEST_DEFAULT=${GGML_GEMMINI_DEQUANT_FP_TEST_DEFAULT:-OFF} | build-arm64-cpu.sh:53 |
| GGML_GEMMINI_ACTIVATION_BITS_DEFAULT=${GGML_GEMMINI_ACTIVATION_BITS_DEFAULT:-8} | build-arm64-cpu.sh:54 |
| GGML_GEMMINI_WEIGHT_BITS_DEFAULT=${GGML_GEMMINI_WEIGHT_BITS_DEFAULT:-8} | build-arm64-cpu.sh:55 |
| GGML_GEMMINI_DIM_DEFAULT=${GGML_GEMMINI_DIM_DEFAULT:-64} | build-arm64-cpu.sh:56 |
| GGML_GEMMINI_BLOCK_SIZE_DEFAULT=${GGML_GEMMINI_BLOCK_SIZE_DEFAULT:-32} | build-arm64-cpu.sh:57 |
| GGML_GEMMINI_PRINT_TILE_DEFAULT=${GGML_GEMMINI_PRINT_TILE_DEFAULT:-0} | build-arm64-cpu.sh:58 |
| GGML_GEMMINI_FULL_C_DEFAULT=${GGML_GEMMINI_FULL_C_DEFAULT:-1} | build-arm64-cpu.sh:59 |
| GGML_GEMMINI_LOW_D_DEFAULT=${GGML_GEMMINI_LOW_D_DEFAULT:-0} | build-arm64-cpu.sh:60 |
| GGML_GEMMINI_TRANSPOSE_B_DEFAULT=${GGML_GEMMINI_TRANSPOSE_B_DEFAULT:-1} | build-arm64-cpu.sh:61 |
| GGML_GEMMINI_WS_DEBUG_DEFAULT=${GGML_GEMMINI_WS_DEBUG_DEFAULT:-0} | build-arm64-cpu.sh:62 |
| GGML_GEMMINI_WS_LOOP_CYCLE_DEFAULT=${GGML_GEMMINI_WS_LOOP_CYCLE_DEFAULT:-0} | build-arm64-cpu.sh:63 |
| GGML_GEMMINI_DISABLE_WS_REUSE_DEFAULT=${GGML_GEMMINI_DISABLE_WS_REUSE_DEFAULT:-0} | build-arm64-cpu.sh:64 |
| GGML_GEMMINI_ENABLE_A4NKS_ATTENTION_DEFAULT=${GGML_GEMMINI_ENABLE_A4NKS_ATTENTION_DEFAULT:-OFF} | build-arm64-cpu.sh:67 |
| GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION_DEFAULT=${GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION_DEFAULT:-OFF} | build-arm64-cpu.sh:68 |
| GGML_GEMMINI_ENABLE_STRIPE_MATMUL_DEFAULT=${GGML_GEMMINI_ENABLE_STRIPE_MATMUL_DEFAULT:-ON} | build-arm64-cpu.sh:69 |
| GGML_GEMMINI_ENABLE_STRIPE_PIPELINE_DEFAULT=${GGML_GEMMINI_ENABLE_STRIPE_PIPELINE_DEFAULT:-ON} | build-arm64-cpu.sh:70 |
| GGML_GEMMINI_DEFAULT_MATMUL_MODE_DEFAULT=${GGML_GEMMINI_DEFAULT_MATMUL_MODE_DEFAULT:-STRIPE_PIPELINE} | build-arm64-cpu.sh:71 |
| GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT=${GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT:-2} | build-arm64-cpu.sh:72 |
| GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE_DEFAULT=${GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE_DEFAULT:-OFF} | build-arm64-cpu.sh:73 |
| GGML_GEMMINI_ENABLE_RMD_DEFAULT=${GGML_GEMMINI_ENABLE_RMD_DEFAULT:-ON} | build-arm64-cpu.sh:74 |
| GGML_GEMMINI_DEFAULT_RMD_BACKEND_DEFAULT=${GGML_GEMMINI_DEFAULT_RMD_BACKEND_DEFAULT:-CPU} | build-arm64-cpu.sh:75 |
| GGML_GEMMINI_ACTIVATION_QUANT_DEFAULT=${GGML_GEMMINI_ACTIVATION_QUANT_DEFAULT:-EXSIA} | build-arm64-cpu.sh:78 |
| GGML_GEMMINI_EXSIA_SIGMA_DEFAULT=${GGML_GEMMINI_EXSIA_SIGMA_DEFAULT:-2} | build-arm64-cpu.sh:79 |
| GGML_GEMMINI_EXSIA_DEFAULT_MODE_DEFAULT=${GGML_GEMMINI_EXSIA_DEFAULT_MODE_DEFAULT:-LOCAL_FOLDING_PIPELINE} | build-arm64-cpu.sh:80 |
| GGML_GEMMINI_EXSIA_LOCAL_WORKERS_DEFAULT=${GGML_GEMMINI_EXSIA_LOCAL_WORKERS_DEFAULT:-3} | build-arm64-cpu.sh:81 |
| GGML_GEMMINI_EXSIA_PROFILE_SCOPE_DEFAULT=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_DEFAULT:-OFF} | build-arm64-cpu.sh:82 |
| LOG_DEBUG_DEFAULT=${LOG_DEBUG_DEFAULT:-1} | build-arm64-cpu.sh:85 |
| LOG_CYCLE_DEFAULT=${LOG_CYCLE_DEFAULT:-1} | build-arm64-cpu.sh:86 |
| LOG_DUMP_DEFAULT=${LOG_DUMP_DEFAULT:-0} | build-arm64-cpu.sh:87 |
| LOG_DUMP_SCALE_DEFAULT=${LOG_DUMP_SCALE_DEFAULT:-0} | build-arm64-cpu.sh:88 |
| CYCLE_DETAIL_DEFAULT=${CYCLE_DETAIL_DEFAULT:-1} | build-arm64-cpu.sh:89 |
| CYCLE_SIM_DEFAULT=${CYCLE_SIM_DEFAULT:-0} | build-arm64-cpu.sh:90 |
| GGML_CPU_CYCLE_LOG_DEFAULT=${GGML_CPU_CYCLE_LOG_DEFAULT:-$LOG_CYCLE_DEFAULT} | build-arm64-cpu.sh:91 |
| GGML_GEMMINI_ACT_QUANT_METRICS_DEFAULT=${GGML_GEMMINI_ACT_QUANT_METRICS_DEFAULT:-0} | build-arm64-cpu.sh:92 |
| GGML_GEMMINI_RESIDUAL_METRICS_DEFAULT=${GGML_GEMMINI_RESIDUAL_METRICS_DEFAULT:-0} | build-arm64-cpu.sh:93 |
| GGML_GEMMINI_SCALE_METRICS_DEFAULT=${GGML_GEMMINI_SCALE_METRICS_DEFAULT:-0} | build-arm64-cpu.sh:94 |
| IM2P_SIM_ROOT_DEFAULT=${IM2P_SIM_ROOT_DEFAULT:-"$SCRIPT_ROOT/../IM2P.sim"} | build-arm64-cpu.sh:97 |
| IM2P_SIM_BUILD_DIR_DEFAULT=${IM2P_SIM_BUILD_DIR_DEFAULT:-} | build-arm64-cpu.sh:98 |
| IM2P_SIM_IMPLEMENTATION_DEFAULT=${IM2P_SIM_IMPLEMENTATION_DEFAULT:-GEMMINI_HP1} | build-arm64-cpu.sh:99 |
| target=riscv | build-riscv.sh:40 |
| BUILD_DIR=${BUILD_DIR:-${BUILD_DIR_DEFAULT:-build-$target}} | build-riscv.sh:47 |
| BUILD_JOBS_DEFAULT=${BUILD_JOBS:-${BUILD_JOBS_DEFAULT:-$(im2p_host_jobs)}} | build-riscv.sh:48 |
| IM2P_ARTIFACT_SET_DEFAULT=${IM2P_ARTIFACT_SET_DEFAULT:-SELECTED} | build-riscv.sh:49 |
| IM2P_CACHE_JOBS_DEFAULT=${IM2P_CACHE_JOBS_DEFAULT:-} | build-riscv.sh:50 |
| GGML_CPU_DEFAULT=${GGML_CPU_DEFAULT:-ON} | build-riscv.sh:53 |
| GGML_GEMMINI_DEFAULT=${GGML_GEMMINI_DEFAULT:-OFF} | build-riscv.sh:54 |
| GGML_POTAL_DEFAULT=${GGML_POTAL_DEFAULT:-ON} | build-riscv.sh:55 |
| GGML_METAL_DEFAULT=${GGML_METAL_DEFAULT:-OFF} | build-riscv.sh:56 |
| GGML_CUDA_DEFAULT=${GGML_CUDA_DEFAULT:-OFF} | build-riscv.sh:57 |
| GGML_BLAS_DEFAULT=${GGML_BLAS_DEFAULT:-OFF} | build-riscv.sh:58 |
| GGML_VULKAN_DEFAULT=${GGML_VULKAN_DEFAULT:-OFF} | build-riscv.sh:59 |
| GGML_OPENCL_DEFAULT=${GGML_OPENCL_DEFAULT:-OFF} | build-riscv.sh:60 |
| GGML_RPC_DEFAULT=${GGML_RPC_DEFAULT:-OFF} | build-riscv.sh:61 |
| GGML_BACKEND_DL_DEFAULT=${GGML_BACKEND_DL_DEFAULT:-OFF} | build-riscv.sh:62 |
| POTAL_HW_PARAMS_DEFAULT=${POTAL_HW_PARAMS_DEFAULT:-"$SCRIPT_ROOT/../../gemmini-rocc-tests/include/gemmini_potal_params.h"} | build-riscv.sh:66 |
| POTAL_EXPECTED_DIM_DEFAULT=${POTAL_EXPECTED_DIM_DEFAULT:-16} | build-riscv.sh:67 |
| GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT=${GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT:-HARDWARE} | build-riscv.sh:70 |
| GGML_GEMMINI_OPTION_DEFAULT=${GGML_GEMMINI_OPTION_DEFAULT:-WS} | build-riscv.sh:71 |
| GGML_NATIVE_DEFAULT=${GGML_NATIVE_DEFAULT:-OFF} | build-riscv.sh:72 |
| GGML_OPENMP_DEFAULT=${GGML_OPENMP_DEFAULT:-OFF} | build-riscv.sh:73 |
| GGML_GEMMINI_ENABLE_OPENMP_DEFAULT=${GGML_GEMMINI_ENABLE_OPENMP_DEFAULT:-OFF} | build-riscv.sh:74 |
| LLAMA_BUILD_TESTS_DEFAULT=${LLAMA_BUILD_TESTS_DEFAULT:-ON} | build-riscv.sh:75 |
| LLAMA_CURL_DEFAULT=${LLAMA_CURL_DEFAULT:-OFF} | build-riscv.sh:76 |
| CMAKE_TOOLCHAIN_FILE_DEFAULT=${CMAKE_TOOLCHAIN_FILE_DEFAULT:-"$SCRIPT_ROOT/cmake/$target.cmake"} | build-riscv.sh:77 |
| CMAKE_BUILD_TYPE_DEFAULT=${CMAKE_BUILD_TYPE_DEFAULT:-DEBUG} | build-riscv.sh:79 |
| BUILD_SHARED_LIBS_DEFAULT=${BUILD_SHARED_LIBS_DEFAULT:-OFF} | build-riscv.sh:80 |
| CMAKE_BUILD_TYPE_DEFAULT=${CMAKE_BUILD_TYPE_DEFAULT:-Release} | build-riscv.sh:82 |
| BUILD_SHARED_LIBS_DEFAULT=${BUILD_SHARED_LIBS_DEFAULT:-ON} | build-riscv.sh:83 |
| GGML_GEMMINI_COMPUTE_TYPE_DEFAULT=${GGML_GEMMINI_COMPUTE_TYPE_DEFAULT:-INT} | build-riscv.sh:87 |
| GGML_GEMMINI_DEQUANT_FP_TEST_DEFAULT=${GGML_GEMMINI_DEQUANT_FP_TEST_DEFAULT:-OFF} | build-riscv.sh:88 |
| GGML_GEMMINI_ACTIVATION_BITS_DEFAULT=${GGML_GEMMINI_ACTIVATION_BITS_DEFAULT:-8} | build-riscv.sh:89 |
| GGML_GEMMINI_WEIGHT_BITS_DEFAULT=${GGML_GEMMINI_WEIGHT_BITS_DEFAULT:-8} | build-riscv.sh:90 |
| GGML_GEMMINI_DIM_DEFAULT=${GGML_GEMMINI_DIM_DEFAULT:-16} | build-riscv.sh:91 |
| GGML_GEMMINI_BLOCK_SIZE_DEFAULT=${GGML_GEMMINI_BLOCK_SIZE_DEFAULT:-32} | build-riscv.sh:92 |
| GGML_GEMMINI_PRINT_TILE_DEFAULT=${GGML_GEMMINI_PRINT_TILE_DEFAULT:-0} | build-riscv.sh:93 |
| GGML_GEMMINI_FULL_C_DEFAULT=${GGML_GEMMINI_FULL_C_DEFAULT:-1} | build-riscv.sh:94 |
| GGML_GEMMINI_LOW_D_DEFAULT=${GGML_GEMMINI_LOW_D_DEFAULT:-0} | build-riscv.sh:95 |
| GGML_GEMMINI_TRANSPOSE_B_DEFAULT=${GGML_GEMMINI_TRANSPOSE_B_DEFAULT:-1} | build-riscv.sh:96 |
| GGML_GEMMINI_WS_DEBUG_DEFAULT=${GGML_GEMMINI_WS_DEBUG_DEFAULT:-0} | build-riscv.sh:97 |
| GGML_GEMMINI_WS_LOOP_CYCLE_DEFAULT=${GGML_GEMMINI_WS_LOOP_CYCLE_DEFAULT:-0} | build-riscv.sh:98 |
| GGML_GEMMINI_DISABLE_WS_REUSE_DEFAULT=${GGML_GEMMINI_DISABLE_WS_REUSE_DEFAULT:-0} | build-riscv.sh:99 |
| GGML_GEMMINI_ENABLE_A4NKS_ATTENTION_DEFAULT=${GGML_GEMMINI_ENABLE_A4NKS_ATTENTION_DEFAULT:-OFF} | build-riscv.sh:102 |
| GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION_DEFAULT=${GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION_DEFAULT:-OFF} | build-riscv.sh:103 |
| GGML_GEMMINI_ENABLE_STRIPE_MATMUL_DEFAULT=${GGML_GEMMINI_ENABLE_STRIPE_MATMUL_DEFAULT:-ON} | build-riscv.sh:104 |
| GGML_GEMMINI_ENABLE_STRIPE_PIPELINE_DEFAULT=${GGML_GEMMINI_ENABLE_STRIPE_PIPELINE_DEFAULT:-ON} | build-riscv.sh:105 |
| GGML_GEMMINI_DEFAULT_MATMUL_MODE_DEFAULT=${GGML_GEMMINI_DEFAULT_MATMUL_MODE_DEFAULT:-FULL} | build-riscv.sh:106 |
| GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT=${GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT:-2} | build-riscv.sh:107 |
| GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE_DEFAULT=${GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE_DEFAULT:-ON} | build-riscv.sh:108 |
| GGML_GEMMINI_ENABLE_RMD_DEFAULT=${GGML_GEMMINI_ENABLE_RMD_DEFAULT:-OFF} | build-riscv.sh:111 |
| GGML_GEMMINI_DEFAULT_RMD_BACKEND_DEFAULT=${GGML_GEMMINI_DEFAULT_RMD_BACKEND_DEFAULT:-CPU} | build-riscv.sh:112 |
| GGML_GEMMINI_ACTIVATION_QUANT_DEFAULT=${GGML_GEMMINI_ACTIVATION_QUANT_DEFAULT:-TENSOR} # HP1 validation setting; all five are supported | build-riscv.sh:115 |
| GGML_GEMMINI_EXSIA_SIGMA_DEFAULT=${GGML_GEMMINI_EXSIA_SIGMA_DEFAULT:-2} | build-riscv.sh:116 |
| GGML_GEMMINI_EXSIA_DEFAULT_MODE_DEFAULT=${GGML_GEMMINI_EXSIA_DEFAULT_MODE_DEFAULT:-SEQUENTIAL} | build-riscv.sh:117 |
| GGML_GEMMINI_EXSIA_LOCAL_WORKERS_DEFAULT=${GGML_GEMMINI_EXSIA_LOCAL_WORKERS_DEFAULT:-4} | build-riscv.sh:118 |
| GGML_GEMMINI_EXSIA_PROFILE_SCOPE_DEFAULT=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_DEFAULT:-OFF} | build-riscv.sh:119 |
| LOG_DEBUG_DEFAULT=${LOG_DEBUG_DEFAULT:-1} | build-riscv.sh:122 |
| LOG_CYCLE_DEFAULT=${LOG_CYCLE_DEFAULT:-1} | build-riscv.sh:123 |
| LOG_DUMP_DEFAULT=${LOG_DUMP_DEFAULT:-0} | build-riscv.sh:124 |
| LOG_DUMP_SCALE_DEFAULT=${LOG_DUMP_SCALE_DEFAULT:-0} | build-riscv.sh:125 |
| CYCLE_DETAIL_DEFAULT=${CYCLE_DETAIL_DEFAULT:-0} | build-riscv.sh:126 |
| CYCLE_SIM_DEFAULT=${CYCLE_SIM_DEFAULT:-0} | build-riscv.sh:127 |
| GGML_CPU_CYCLE_LOG_DEFAULT=${GGML_CPU_CYCLE_LOG_DEFAULT:-$LOG_CYCLE_DEFAULT} | build-riscv.sh:128 |
| GGML_GEMMINI_ACT_QUANT_METRICS_DEFAULT=${GGML_GEMMINI_ACT_QUANT_METRICS_DEFAULT:-0} | build-riscv.sh:129 |
| GGML_GEMMINI_RESIDUAL_METRICS_DEFAULT=${GGML_GEMMINI_RESIDUAL_METRICS_DEFAULT:-0} | build-riscv.sh:130 |
| GGML_GEMMINI_SCALE_METRICS_DEFAULT=${GGML_GEMMINI_SCALE_METRICS_DEFAULT:-0} | build-riscv.sh:131 |
| IM2P_SIM_ROOT_DEFAULT=${IM2P_SIM_ROOT_DEFAULT:-"$SCRIPT_ROOT/../IM2P.sim"} | build-riscv.sh:134 |
| IM2P_SIM_BUILD_DIR_DEFAULT=${IM2P_SIM_BUILD_DIR_DEFAULT:-} | build-riscv.sh:135 |
| IM2P_SIM_IMPLEMENTATION_DEFAULT=${IM2P_SIM_IMPLEMENTATION_DEFAULT:-GEMMINI_HP1} | build-riscv.sh:136 |
| BUILD_DIR=${BUILD_DIR:-${BUILD_DIR_DEFAULT:-build-arm64}} | build-arm64.sh:18 |
| BUILD_JOBS_DEFAULT=${BUILD_JOBS:-${BUILD_JOBS_DEFAULT:-$(im2p_host_jobs)}} | build-arm64.sh:19 |
| IM2P_ARTIFACT_SET_DEFAULT=${IM2P_ARTIFACT_SET_DEFAULT:-SELECTED} | build-arm64.sh:20 |
| IM2P_CACHE_JOBS_DEFAULT=${IM2P_CACHE_JOBS_DEFAULT:-} | build-arm64.sh:21 |
| APPLE_SILICON_ARCH_DEFAULT=${APPLE_SILICON_ARCH_DEFAULT:-arm64} | build-arm64.sh:22 |
| LIBOMP_PREFIX_DEFAULT=${LIBOMP_PREFIX_DEFAULT:-} | build-arm64.sh:23 |
| GGML_CPU_DEFAULT=${GGML_CPU_DEFAULT:-ON} | build-arm64.sh:26 |
| GGML_GEMMINI_DEFAULT=${GGML_GEMMINI_DEFAULT:-ON} | build-arm64.sh:27 |
| GGML_METAL_DEFAULT=${GGML_METAL_DEFAULT:-ON} | build-arm64.sh:29 |
| GGML_METAL_DEFAULT=${GGML_METAL_DEFAULT:-OFF} | build-arm64.sh:31 |
| GGML_CUDA_DEFAULT=${GGML_CUDA_DEFAULT:-OFF} | build-arm64.sh:33 |
| GGML_BLAS_DEFAULT=${GGML_BLAS_DEFAULT:-OFF} | build-arm64.sh:34 |
| GGML_VULKAN_DEFAULT=${GGML_VULKAN_DEFAULT:-OFF} | build-arm64.sh:35 |
| GGML_OPENCL_DEFAULT=${GGML_OPENCL_DEFAULT:-OFF} | build-arm64.sh:36 |
| GGML_RPC_DEFAULT=${GGML_RPC_DEFAULT:-OFF} | build-arm64.sh:37 |
| GGML_BACKEND_DL_DEFAULT=${GGML_BACKEND_DL_DEFAULT:-ON} | build-arm64.sh:38 |
| GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT=${GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT:-HARDWARE} | build-arm64.sh:41 |
| GGML_GEMMINI_OPTION_DEFAULT=${GGML_GEMMINI_OPTION_DEFAULT:-CPU} | build-arm64.sh:42 |
| GGML_NATIVE_DEFAULT=${GGML_NATIVE_DEFAULT:-ON} | build-arm64.sh:43 |
| GGML_OPENMP_DEFAULT=${GGML_OPENMP_DEFAULT:-ON} | build-arm64.sh:44 |
| GGML_GEMMINI_ENABLE_OPENMP_DEFAULT=${GGML_GEMMINI_ENABLE_OPENMP_DEFAULT:-ON} | build-arm64.sh:45 |
| LLAMA_BUILD_TESTS_DEFAULT=${LLAMA_BUILD_TESTS_DEFAULT:-OFF} | build-arm64.sh:46 |
| LLAMA_CURL_DEFAULT=${LLAMA_CURL_DEFAULT:-OFF} | build-arm64.sh:47 |
| CMAKE_BUILD_TYPE_DEFAULT=${CMAKE_BUILD_TYPE_DEFAULT:-Release} | build-arm64.sh:48 |
| BUILD_SHARED_LIBS_DEFAULT=${BUILD_SHARED_LIBS_DEFAULT:-ON} | build-arm64.sh:49 |
| LIBOMP_PREFIX_DEFAULT=${LIBOMP_PREFIX:-${LIBOMP_PREFIX_DEFAULT:-$(im2p_host_libomp_prefix)}} | build-arm64.sh:51 |
| CMAKE_OSX_ARCHITECTURES_DEFAULT=${CMAKE_OSX_ARCHITECTURES_DEFAULT:-"${APPLE_SILICON_ARCH:-$APPLE_SILICON_ARCH_DEFAULT}"} | build-arm64.sh:52 |
| OpenMP_ROOT_DEFAULT=${OpenMP_ROOT_DEFAULT:-"$LIBOMP_PREFIX_DEFAULT"} | build-arm64.sh:54 |
| CMAKE_PREFIX_PATH_DEFAULT=${CMAKE_PREFIX_PATH_DEFAULT:-"$LIBOMP_PREFIX_DEFAULT"} | build-arm64.sh:55 |
| GGML_GEMMINI_COMPUTE_TYPE_DEFAULT=${GGML_GEMMINI_COMPUTE_TYPE_DEFAULT:-INT} | build-arm64.sh:60 |
| GGML_GEMMINI_DEQUANT_FP_TEST_DEFAULT=${GGML_GEMMINI_DEQUANT_FP_TEST_DEFAULT:-OFF} | build-arm64.sh:61 |
| GGML_GEMMINI_ACTIVATION_BITS_DEFAULT=${GGML_GEMMINI_ACTIVATION_BITS_DEFAULT:-4} | build-arm64.sh:62 |
| GGML_GEMMINI_WEIGHT_BITS_DEFAULT=${GGML_GEMMINI_WEIGHT_BITS_DEFAULT:-4} | build-arm64.sh:63 |
| GGML_GEMMINI_DIM_DEFAULT=${GGML_GEMMINI_DIM_DEFAULT:-32} | build-arm64.sh:64 |
| GGML_GEMMINI_BLOCK_SIZE_DEFAULT=${GGML_GEMMINI_BLOCK_SIZE_DEFAULT:-32} | build-arm64.sh:65 |
| GGML_GEMMINI_PRINT_TILE_DEFAULT=${GGML_GEMMINI_PRINT_TILE_DEFAULT:-0} | build-arm64.sh:66 |
| GGML_GEMMINI_FULL_C_DEFAULT=${GGML_GEMMINI_FULL_C_DEFAULT:-1} | build-arm64.sh:67 |
| GGML_GEMMINI_LOW_D_DEFAULT=${GGML_GEMMINI_LOW_D_DEFAULT:-0} | build-arm64.sh:68 |
| GGML_GEMMINI_TRANSPOSE_B_DEFAULT=${GGML_GEMMINI_TRANSPOSE_B_DEFAULT:-1} | build-arm64.sh:69 |
| GGML_GEMMINI_WS_DEBUG_DEFAULT=${GGML_GEMMINI_WS_DEBUG_DEFAULT:-0} | build-arm64.sh:70 |
| GGML_GEMMINI_WS_LOOP_CYCLE_DEFAULT=${GGML_GEMMINI_WS_LOOP_CYCLE_DEFAULT:-0} | build-arm64.sh:71 |
| GGML_GEMMINI_DISABLE_WS_REUSE_DEFAULT=${GGML_GEMMINI_DISABLE_WS_REUSE_DEFAULT:-0} | build-arm64.sh:72 |
| GGML_GEMMINI_ENABLE_A4NKS_ATTENTION_DEFAULT=${GGML_GEMMINI_ENABLE_A4NKS_ATTENTION_DEFAULT:-OFF} | build-arm64.sh:75 |
| GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION_DEFAULT=${GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION_DEFAULT:-OFF} | build-arm64.sh:76 |
| GGML_GEMMINI_ENABLE_STRIPE_MATMUL_DEFAULT=${GGML_GEMMINI_ENABLE_STRIPE_MATMUL_DEFAULT:-ON} | build-arm64.sh:77 |
| GGML_GEMMINI_ENABLE_STRIPE_PIPELINE_DEFAULT=${GGML_GEMMINI_ENABLE_STRIPE_PIPELINE_DEFAULT:-ON} | build-arm64.sh:78 |
| GGML_GEMMINI_DEFAULT_MATMUL_MODE_DEFAULT=${GGML_GEMMINI_DEFAULT_MATMUL_MODE_DEFAULT:-STRIPE_PIPELINE} | build-arm64.sh:79 |
| GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT=${GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT:-2} | build-arm64.sh:80 |
| GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE_DEFAULT=${GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE_DEFAULT:-OFF} | build-arm64.sh:81 |
| GGML_GEMMINI_ENABLE_RMD_DEFAULT=${GGML_GEMMINI_ENABLE_RMD_DEFAULT:-ON} | build-arm64.sh:82 |
| GGML_GEMMINI_DEFAULT_RMD_BACKEND_DEFAULT=${GGML_GEMMINI_DEFAULT_RMD_BACKEND_DEFAULT:-CPU} | build-arm64.sh:83 |
| GGML_GEMMINI_ACTIVATION_QUANT_DEFAULT=${GGML_GEMMINI_ACTIVATION_QUANT_DEFAULT:-EXSIA} | build-arm64.sh:86 |
| GGML_GEMMINI_EXSIA_SIGMA_DEFAULT=${GGML_GEMMINI_EXSIA_SIGMA_DEFAULT:-2} | build-arm64.sh:87 |
| GGML_GEMMINI_EXSIA_DEFAULT_MODE_DEFAULT=${GGML_GEMMINI_EXSIA_DEFAULT_MODE_DEFAULT:-LOCAL_FOLDING_PIPELINE} | build-arm64.sh:88 |
| GGML_GEMMINI_EXSIA_LOCAL_WORKERS_DEFAULT=${GGML_GEMMINI_EXSIA_LOCAL_WORKERS_DEFAULT:-3} | build-arm64.sh:89 |
| GGML_GEMMINI_EXSIA_PROFILE_SCOPE_DEFAULT=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_DEFAULT:-OFF} | build-arm64.sh:90 |
| LOG_DEBUG_DEFAULT=${LOG_DEBUG_DEFAULT:-0} | build-arm64.sh:93 |
| LOG_CYCLE_DEFAULT=${LOG_CYCLE_DEFAULT:-0} | build-arm64.sh:94 |
| LOG_DUMP_DEFAULT=${LOG_DUMP_DEFAULT:-0} | build-arm64.sh:95 |
| LOG_DUMP_SCALE_DEFAULT=${LOG_DUMP_SCALE_DEFAULT:-0} | build-arm64.sh:96 |
| CYCLE_DETAIL_DEFAULT=${CYCLE_DETAIL_DEFAULT:-0} | build-arm64.sh:97 |
| CYCLE_SIM_DEFAULT=${CYCLE_SIM_DEFAULT:-0} | build-arm64.sh:98 |
| GGML_CPU_CYCLE_LOG_DEFAULT=${GGML_CPU_CYCLE_LOG_DEFAULT:-$LOG_CYCLE_DEFAULT} | build-arm64.sh:99 |
| GGML_GEMMINI_ACT_QUANT_METRICS_DEFAULT=${GGML_GEMMINI_ACT_QUANT_METRICS_DEFAULT:-0} | build-arm64.sh:100 |
| GGML_GEMMINI_RESIDUAL_METRICS_DEFAULT=${GGML_GEMMINI_RESIDUAL_METRICS_DEFAULT:-0} | build-arm64.sh:101 |
| GGML_GEMMINI_SCALE_METRICS_DEFAULT=${GGML_GEMMINI_SCALE_METRICS_DEFAULT:-0} | build-arm64.sh:102 |
| IM2P_SIM_ROOT_DEFAULT=${IM2P_SIM_ROOT_DEFAULT:-"$SCRIPT_ROOT/../IM2P.sim"} | build-arm64.sh:105 |
| IM2P_SIM_BUILD_DIR_DEFAULT=${IM2P_SIM_BUILD_DIR_DEFAULT:-} | build-arm64.sh:106 |
| IM2P_SIM_IMPLEMENTATION_DEFAULT=${IM2P_SIM_IMPLEMENTATION_DEFAULT:-GEMMINI_HP1} | build-arm64.sh:107 |

The four generic wrappers invoke the common resolver/provision/configure helpers, then build their target lists; this audit did not execute any wrapper. Cache unsets are explicit in im2p_configure_build, including GEMMINI_SW_PATH (scripts/im2p-host-provision.sh:24-32). Provisioning forwards width, DIM, block and implementation to external make targets and uses selected versus all-matched cache lanes (scripts/im2p-host-provision.sh:37-95). RISCV_LIBGOMP is a target-runtime override advertised by the RISC-V wrapper; its actual library presence/ABI is UNVERIFIED here (build-riscv.sh:13). The FireSim profile requires explicit 16/32 DIM, supplies fixed POTAL_TARGET_DIM/PARAMS, and differs functional/performance LOG_DEBUG default (build-potal-firesim.sh:16-24 and build-potal-firesim.sh:27-57).

CMakePresets.json contains generic platform/debug/static/GPU presets, with no lab NPU preset or NPU geometry cache entry. This absence is confirmed by reading the entire 84-line file (CMakePresets.json:1-84). The RISC-V toolchains set Linux/riscv64, explicit compiler/sysroot and cross-find modes; static variants force additional backend/shared/OpenMP policy and linker flags. Their full assignments follow, because those platform knobs constrain NPU combinations even though most are upstream CMake/GGML settings (cmake/riscv.cmake:1-26; cmake/riscv-static.cmake:1-37; cmake/riscv-static-release.cmake:1-40).

| Toolchain assignment | Evidence |
|---|---|
| set( CMAKE_SYSTEM_NAME Linux ) | cmake/riscv-static-release.cmake:1 |
| set( CMAKE_SYSTEM_PROCESSOR riscv64 ) | cmake/riscv-static-release.cmake:2 |
| set( target riscv64 ) | cmake/riscv-static-release.cmake:4 |
| set(CMAKE_C_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc) | cmake/riscv-static-release.cmake:6 |
| set(CMAKE_CXX_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-g++) | cmake/riscv-static-release.cmake:7 |
| set(CMAKE_SYSROOT /home/alveo/firesim/.conda-env/riscv-tools/sysroot) | cmake/riscv-static-release.cmake:9 |
| set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT}) | cmake/riscv-static-release.cmake:10 |
| set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER) | cmake/riscv-static-release.cmake:12 |
| set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY) | cmake/riscv-static-release.cmake:13 |
| set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY) | cmake/riscv-static-release.cmake:14 |
| set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY) | cmake/riscv-static-release.cmake:15 |
| set( CMAKE_C_COMPILER_TARGET   ${target} ) | cmake/riscv-static-release.cmake:17 |
| set( CMAKE_CXX_COMPILER_TARGET ${target} ) | cmake/riscv-static-release.cmake:18 |
| set( warn_c_flags "-Wno-format -Wno-unused-variable -Wno-unused-function" ) | cmake/riscv-static-release.cmake:20 |
| set (GGML_BACKEND_DL OFF ) | cmake/riscv-static-release.cmake:21 |
| set( CMAKE_BUILD_TYPE Release ) | cmake/riscv-static-release.cmake:23 |
| set( CMAKE_C_FLAGS_INIT   "${warn_c_flags}" ) | cmake/riscv-static-release.cmake:24 |
| set( CMAKE_CXX_FLAGS_INIT "${warn_c_flags}" ) | cmake/riscv-static-release.cmake:25 |
| set( LLAMA_CURL OFF ) | cmake/riscv-static-release.cmake:27 |
| set( GGML_NO_MMAP ON ) | cmake/riscv-static-release.cmake:28 |
| set( GGML_OPENMP OFF ) | cmake/riscv-static-release.cmake:29 |
| set( BUILD_SHARED_LIBS OFF ) | cmake/riscv-static-release.cmake:30 |
| set( GGML_CUDA OFF ) | cmake/riscv-static-release.cmake:31 |
| set( GGML_BLAS OFF ) | cmake/riscv-static-release.cmake:32 |
| set( GGML_CUBLAS OFF ) | cmake/riscv-static-release.cmake:33 |
| set( GGML_METAL OFF ) | cmake/riscv-static-release.cmake:34 |
| set( GGML_OPENBLAS OFF ) | cmake/riscv-static-release.cmake:35 |
| set( CMAKE_EXE_LINKER_FLAGS "-static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive" ) | cmake/riscv-static-release.cmake:37 |
| set( CMAKE_CXX_STANDARD_LIBRARIES "-latomic" ) | cmake/riscv-static-release.cmake:39 |
| set( CMAKE_C_STANDARD_LIBRARIES "-latomic" ) | cmake/riscv-static-release.cmake:40 |
| set( CMAKE_SYSTEM_NAME Linux ) | cmake/riscv-static.cmake:1 |
| set( CMAKE_SYSTEM_PROCESSOR riscv64 ) | cmake/riscv-static.cmake:2 |
| set( target riscv64 ) | cmake/riscv-static.cmake:4 |
| set(CMAKE_C_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc) | cmake/riscv-static.cmake:6 |
| set(CMAKE_CXX_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-g++) | cmake/riscv-static.cmake:7 |
| set(CMAKE_SYSROOT /home/alveo/firesim/.conda-env/riscv-tools/sysroot) | cmake/riscv-static.cmake:9 |
| set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT}) | cmake/riscv-static.cmake:10 |
| set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER) | cmake/riscv-static.cmake:12 |
| set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY) | cmake/riscv-static.cmake:13 |
| set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY) | cmake/riscv-static.cmake:14 |
| set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY) | cmake/riscv-static.cmake:15 |
| set( CMAKE_C_COMPILER_TARGET   ${target} ) | cmake/riscv-static.cmake:17 |
| set( CMAKE_CXX_COMPILER_TARGET ${target} ) | cmake/riscv-static.cmake:18 |
| set( warn_c_flags "-Wno-format -Wno-unused-variable -Wno-unused-function" ) | cmake/riscv-static.cmake:20 |
| set (GGML_BACKEND_DL OFF ) | cmake/riscv-static.cmake:21 |
| set( CMAKE_BUILD_TYPE DEBUG ) | cmake/riscv-static.cmake:23 |
| set( CMAKE_C_FLAGS_INIT   "${warn_c_flags}" ) | cmake/riscv-static.cmake:24 |
| set( CMAKE_CXX_FLAGS_INIT "${warn_c_flags}" ) | cmake/riscv-static.cmake:25 |
| set( LLAMA_CURL OFF ) | cmake/riscv-static.cmake:27 |
| set( GGML_NO_MMAP ON ) | cmake/riscv-static.cmake:28 |
| set( GGML_OPENMP OFF ) | cmake/riscv-static.cmake:29 |
| set( BUILD_SHARED_LIBS OFF ) | cmake/riscv-static.cmake:30 |
| set( GGML_CUDA OFF ) | cmake/riscv-static.cmake:31 |
| set( GGML_BLAS OFF ) | cmake/riscv-static.cmake:32 |
| set( GGML_CUBLAS OFF ) | cmake/riscv-static.cmake:33 |
| set( GGML_METAL OFF ) | cmake/riscv-static.cmake:34 |
| set( GGML_OPENBLAS OFF ) | cmake/riscv-static.cmake:35 |
| set( CMAKE_EXE_LINKER_FLAGS "-static" ) | cmake/riscv-static.cmake:37 |
| set( CMAKE_SYSTEM_NAME Linux ) | cmake/riscv.cmake:1 |
| set( CMAKE_SYSTEM_PROCESSOR riscv64 ) | cmake/riscv.cmake:2 |
| set( target riscv64 ) | cmake/riscv.cmake:4 |
| set(CMAKE_C_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc) | cmake/riscv.cmake:6 |
| set(CMAKE_CXX_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-g++) | cmake/riscv.cmake:7 |
| set(CMAKE_SYSROOT /home/alveo/firesim/.conda-env/riscv-tools/sysroot) | cmake/riscv.cmake:9 |
| set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT}) | cmake/riscv.cmake:10 |
| set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER) | cmake/riscv.cmake:12 |
| set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY) | cmake/riscv.cmake:13 |
| set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY) | cmake/riscv.cmake:14 |
| set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY) | cmake/riscv.cmake:15 |
| set( CMAKE_C_COMPILER_TARGET   ${target} ) | cmake/riscv.cmake:17 |
| set( CMAKE_CXX_COMPILER_TARGET ${target} ) | cmake/riscv.cmake:18 |
| set( warn_c_flags "-Wno-format -Wno-unused-variable -Wno-unused-function" ) | cmake/riscv.cmake:20 |
| set( CMAKE_C_FLAGS_INIT   "${warn_c_flags}" ) | cmake/riscv.cmake:22 |
| set( CMAKE_CXX_FLAGS_INIT "${warn_c_flags}" ) | cmake/riscv.cmake:23 |
| set( LLAMA_CURL OFF ) | cmake/riscv.cmake:25 |
| set( GGML_OPENMP OFF ) | cmake/riscv.cmake:26 |

## Derived build fields (not independent user options)

These names are produced by configuration or backend source-list construction; Phase 1 should generate or keep them internal, rather than expose another independent geometry knob. Their defining statements are the evidence; any external generated target.cmake content remains UNVERIFIED (cmake/ggml-gemmini-im2p.cmake:27).

| Derived name | Definition / construction |
|---|---|
| GGML_GEMMINI_GENERATED_CONFIG_DIR | CMakeLists.txt:204 (GGML_GEMMINI_GENERATED_CONFIG_DIR "${_GGML_GEMMINI_BINARY_DIR}/generated") |
| POTAL_PLANNING_WRAP_OPTIONS | ggml/src/ggml-potal/CMakeLists.txt:233 (POTAL_PLANNING_WRAP_OPTIONS "-Wl,--wrap=_Znwm" "-Wl,--wrap=_Znam" "-Wl,--wrap=_ZdlPv" "-Wl,--wrap=_ZdaPv" "-Wl,--wrap=_ZdlPvm" "-Wl,--wrap=_ZdaPvm") |
| POTAL_GEMMINI_PARAMS_DIR | ggml/src/ggml-potal/CMakeLists.txt:252 (POTAL_GEMMINI_PARAMS_DIR "${POTAL_HW_GENERATED_DIR}/gemmini-params") |
| POTAL_GEMMINI_SHARED_SOURCES | ggml/src/ggml-potal/CMakeLists.txt:262 (POTAL_GEMMINI_SHARED_SOURCES ${_GEMMINI_DIR}/quants/act/dispatch.cpp ${_GEMMINI_DIR}/quants/act/quantize.cpp ${_GEMMINI_DIR}/quants/act/block/block.cpp ${_GEMMINI_DIR}/quants/act/exsia/exsia.cpp ${_GEMMINI_DIR}/quants/act/exsia/exsia-profile.cpp ${_GEMMINI_DIR}/quants/act/exsia/local.cpp ${_GEMMINI_DIR}/quants/act/exsia/folding.cpp ${_GEMMINI_DIR}/quants/act/tensor/tensor.cpp ${_GEMMINI_DIR}/quants/act/stripe/stripe.cpp ${_GEMMINI_DIR}/quants/act/token/token.cpp ${_GEMMINI_DIR}/quants/common/dequant.cpp ${_GEMMINI_DIR}/quants/common/weight_reader.cpp ${_GEMMINI_DIR}/residual/direct/direct-executor.cpp ${_GEMMINI_DIR}/residual/rmd/rmd-builder.cpp ${_GEMMINI_DIR}/residual/rmd/rmd-bitmap-builder.cpp ${_GEMMINI_DIR}/residual/rmd/rmd-compose.cpp ${_GEMMINI_DIR}/matmul/options.cpp ${_GEMMINI_DIR}/ggml-gemmini-telemetry.cpp) |
| POTAL_BACKEND_SOURCES | ggml/src/ggml-potal/CMakeLists.txt:283 (POTAL_BACKEND_SOURCES ggml-potal-buffer.cpp ggml-potal-buffer.hpp log.cpp log.hpp ops.cpp ops.hpp matmul/attention.cpp matmul/attention-hw.cpp matmul/attention.hpp matmul/dense.cpp matmul/dense.hpp matmul/execution.cpp matmul/execution.hpp matmul/options.hpp matmul/pack.cpp matmul/pack.hpp matmul/paired.cpp matmul/paired.hpp matmul/paired-auto.cpp matmul/paired-auto.hpp matmul/paired-emulator.cpp matmul/paired-plan.cpp matmul/paired-schedule.cpp matmul/stripe.cpp matmul/stripe.hpp matmul/weight_cache.cpp matmul/weight_cache.hpp quants/weight/hp1/hp1.cpp quants/weight/hp1/hp1.hpp residual/stripe-residual.cpp residual/stripe-residual.hpp trace/hw-window.cpp trace/hw-window.hpp trace/timeline.cpp trace/timeline.hpp ${POTAL_GEMMINI_SHARED_SOURCES}) |
| POTAL_LOG_DEFINITIONS | ggml/src/ggml-potal/CMakeLists.txt:341 (POTAL_LOG_DEFINITIONS ${GGML_GEMMINI_RUNTIME_COMPILE_DEFS} GGML_GEMMINI_BLOCK_SIZE=${_GGML_GEMMINI_BLOCK_SIZE_VALUE} GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA} GGML_GEMMINI_EXSIA_STRIPE_ANCHOR=${GGML_GEMMINI_EXSIA_STRIPE_ANCHOR} GGML_GEMMINI_CONFIGURED_DIM=${POTAL_HW_DIM} GGML_POTAL_PAIRED_AUTO=${_POTAL_PAIRED_AUTO_VALUE} GGML_GEMMINI_EXECUTION_BACKEND_HARDWARE=1 EXSIA_VALIDATION=0 GGML_POTAL_TIMELINE_HOOKS=1 GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE=${GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE}) |
| POTAL_PRIVATE_INCLUDES | ggml/src/ggml-potal/CMakeLists.txt:352 (POTAL_PRIVATE_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/include ${_GEMMINI_DIR} ${GGML_GEMMINI_GENERATED_CONFIG_DIR} ${POTAL_GEMMINI_PARAMS_DIR} ${POTAL_HW_GENERATED_DIR}) |
| POTAL_W4A4_W8_ONLY_TESTS | ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7 (POTAL_W4A4_W8_ONLY_TESTS potal-api-test potal-stripe-api-test potal-attention-test potal-ggml-unit-test potal-log-test potal-tiling-unit-test potal-residual-unit-test potal-stripe-pipeline-test potal-hw-timeline-test potal-paired-plan-test potal-paired-unit-test potal-paired-auto-test potal-paired-execution-test potal-e4b1-test potal-e4b2-test potal-e4c1-test potal-resident-test potal-resident-model-test) |
| GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE | ggml/src/ggml-gemmini/CMakeLists.txt:21 (GGML_GEMMINI_EXSIA_PROFILE_SCOPE_VALUE 0) |
| GGML_GEMMINI_SOURCES | ggml/src/ggml-gemmini/CMakeLists.txt:81 (GGML_GEMMINI_SOURCES ggml-gemmini.cpp potal-sensitivity.cpp ops.cpp ggml-gemmini-buffer.cpp ggml-gemmini-geometry.hpp matmul/dense.cpp matmul/execution.cpp matmul/options.cpp matmul/attention.cpp matmul/attention-stationary.cpp ggml-gemmini-telemetry.cpp quants/act/dispatch.cpp quants/act/quantize.cpp quants/act/preparation-cache.cpp quants/act/block/block.cpp quants/act/exsia/exsia.cpp quants/act/exsia/exsia-profile.cpp quants/act/exsia/local.cpp quants/act/exsia/folding.cpp quants/act/tensor/tensor.cpp quants/act/stripe/stripe.cpp quants/act/token/token.cpp quants/common/fp16_util.cpp quants/common/dequant.cpp quants/common/weight_reader.cpp residual/direct/direct-executor.cpp residual/rmd/rmd-builder.cpp residual/rmd/rmd-bitmap-builder.cpp residual/rmd/rmd-run-aware.cpp residual/rmd/rmd-executor.cpp residual/rmd/rmd-im2p-executor.cpp residual/rmd/rmd-compose.cpp residual/rmd/rmd-reference.cpp dump/dump_common.cpp dump/dump_tensor.cpp dump/scale/scale_dump_block.cpp dump/scale/scale_dump_dispatch.cpp dump/scale/scale_dump_stub.cpp) |
| GGML_GEMMINI_IM2P_ARTIFACT_ID | cmake/ggml-gemmini-im2p.cmake:71 (GGML_GEMMINI_IM2P_ARTIFACT_ID "a${GGML_GEMMINI_ACTIVATION_BITS}-w${GGML_GEMMINI_WEIGHT_BITS}-d${GGML_GEMMINI_DIM}") |
| GGML_GEMMINI_IM2P_MANIFEST | cmake/ggml-gemmini-im2p.cmake:99 (GGML_GEMMINI_IM2P_MANIFEST "${GGML_GEMMINI_IM2P_GENERATION}/real-lib.json") |
| GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE | cmake/ggml-gemmini-im2p.cmake:151 (GGML_GEMMINI_IM2P_FRONTEND_ARCHIVE "${GGML_GEMMINI_IM2P_GENERATION}/libim2p_gemmini_frontend.a") |
| GGML_GEMMINI_IM2P_SIM_ARCHIVE | cmake/ggml-gemmini-im2p.cmake:153 (GGML_GEMMINI_IM2P_SIM_ARCHIVE "${GGML_GEMMINI_IM2P_GENERATION}/libim2p_sim.a") |
| POTAL_HW_GENERATED_DIR | cmake/potal-hw-params.cmake:253 (POTAL_HW_GENERATED_DIR "${out_dir}" PARENT_SCOPE) |
| POTAL_HW_DIM | cmake/potal-hw-params.cmake:254 (POTAL_HW_DIM "${DIM}" PARENT_SCOPE) |
| GGML_GEMMINI_COMPILE_DEFS | cmake/ggml-gemmini-config.cmake:22 (GGML_GEMMINI_COMPILE_DEFS PRINT_TILE=${GGML_GEMMINI_PRINT_TILE} OPTION=${_GGML_GEMMINI_OPTION_UPPER} FULL_C=${GGML_GEMMINI_FULL_C} LOW_D=${GGML_GEMMINI_LOW_D} TRANSPOSE_B=${GGML_GEMMINI_TRANSPOSE_B} GEMMINI_WS_DEBUG=${GGML_GEMMINI_WS_DEBUG} GEMMINI_WS_LOOP_CYCLE=${GGML_GEMMINI_WS_LOOP_CYCLE} GEMMINI_DISABLE_WS_REUSE=${GGML_GEMMINI_DISABLE_WS_REUSE} GGML_GEMMINI_COMPUTE_TYPE=${_GGML_GEMMINI_COMPUTE_TYPE_VALUE} GGML_GEMMINI_DEQUANT_FP_TEST=${_GGML_GEMMINI_DEQUANT_FP_TEST_VALUE} GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE} GGML_GEMMINI_BLOCK_SIZE=${_GGML_GEMMINI_BLOCK_SIZE_VALUE} GGML_GEMMINI_EXSIA_SIGMA=${GGML_GEMMINI_EXSIA_SIGMA} GGML_GEMMINI_EXSIA_STRIPE_ANCHOR=${GGML_GEMMINI_EXSIA_STRIPE_ANCHOR} GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE} GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE} GGML_GEMMINI_CONFIGURED_DIM=${GGML_GEMMINI_DIM}) |
| GGML_GEMMINI_RUNTIME_COMPILE_DEFS | cmake/ggml-gemmini-config.cmake:54 (GGML_GEMMINI_RUNTIME_COMPILE_DEFS GGML_GEMMINI_ACT_METRICS=${GGML_GEMMINI_ACT_QUANT_METRICS} GGML_GEMMINI_SCALE_METRICS=${GGML_GEMMINI_SCALE_METRICS} GGML_GEMMINI_ACT_QUANT_METRICS=${GGML_GEMMINI_ACT_QUANT_METRICS} GGML_GEMMINI_RESIDUAL_METRICS=${GGML_GEMMINI_RESIDUAL_METRICS} CYCLE_SIM=${CYCLE_SIM} LOG_DEBUG=${LOG_DEBUG} LOG_CYCLE=${LOG_CYCLE} LOG_DUMP=${LOG_DUMP} LOG_DUMP_SCALE=${LOG_DUMP_SCALE} GEMMINI_VERIFY_DEQ=1 CYCLE_DETAIL=${CYCLE_DETAIL} GGML_GEMMINI_COMPUTE_TYPE=${_GGML_GEMMINI_COMPUTE_TYPE_VALUE} GGML_GEMMINI_DEQUANT_FP_TEST=${_GGML_GEMMINI_DEQUANT_FP_TEST_VALUE} GGML_GEMMINI_ACTIVATION_QUANT=${_GGML_GEMMINI_ACTIVATION_QUANT_VALUE} GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE=${_GGML_GEMMINI_EXSIA_DEFAULT_MODE_VALUE} GGML_GEMMINI_EXSIA_LOCAL_WORKERS=${GGML_GEMMINI_EXSIA_LOCAL_WORKERS} GGML_GEMMINI_ENABLE_RMD=${_GGML_GEMMINI_ENABLE_RMD_VALUE} GGML_GEMMINI_ACTIVATION_BITS=${_GGML_GEMMINI_ACTIVATION_BITS_VALUE} GGML_GEMMINI_WEIGHT_BITS=${_GGML_GEMMINI_WEIGHT_BITS_VALUE}) |
| GGML_GEMMINI_UTILS_COMPILE_DEFS | cmake/ggml-gemmini-config.cmake:80 (GGML_GEMMINI_UTILS_COMPILE_DEFS GGML_GEMMINI_ACT_METRICS=${GGML_GEMMINI_ACT_QUANT_METRICS} GGML_GEMMINI_SCALE_METRICS=${GGML_GEMMINI_SCALE_METRICS} GGML_GEMMINI_ACT_QUANT_METRICS=${GGML_GEMMINI_ACT_QUANT_METRICS} GGML_GEMMINI_RESIDUAL_METRICS=${GGML_GEMMINI_RESIDUAL_METRICS} LOG_DEBUG=${LOG_DEBUG} LOG_CYCLE=${LOG_CYCLE} CYCLE_DETAIL=${CYCLE_DETAIL}) |

The generated adapter additionally emits POTAL_HW_XCUSTOM, DIM, ADDR_LEN, BANK_NUM, BANK_ROWS, ACC_ROWS, MAX_BYTES, SCALE_ENTRIES, INDEX_ENTRIES and header SHA256 (cmake/potal-hw-params.cmake:184-201). IM2P_CYCLE_SIM_BANK_COUNT/BANK_ROWS/ACCUMULATOR_ROWS/PARTIAL_BITS and CONFIGURE_INPUTS come from the external target generator; they feed rewritten SDK parameters and CPU-functional compile definitions (cmake/ggml-gemmini-im2p.cmake:27-47; ggml/src/ggml-gemmini/CMakeLists.txt:280-290). No independent current NPU_SPAD_* or NPU_ACC_* user options were confirmed; those are Phase 1 names.

## Runtime environment inventory

Runtime getenv consumers below are distinct from wrapper build-time NAME/NAME_DEFAULT forwarding. Same-name GGML_POTAL_PAIRED_AUTO is both build default and process override. Common GEMMINI_MATMUL_MODE/RMD_BACKEND/STRIPE_JOB_CAPACITY are gated by ALLOW_RUNTIME_MATMUL_OVERRIDE and validated, with incompatible PoTAL nodes rejected (ggml/src/ggml-gemmini/matmul/options.cpp:19-44; ggml/src/ggml-potal/CMakeLists.txt:201-207). Direct production getenv literals are listed exhaustively in the read NPU backend corpus; indirect lookup keys are listed afterward. Unset/default semantics for each diagnostic switch must be read at its cited consumer; runtime enabling is not evidence of compiled support or successful hardware execution.

| Runtime environment name | Production readers | Recommended ownership |
|---|---|---|
| GEMMINI_LOG_DIR | ggml/src/ggml-gemmini-utils/src/debug.cpp:111; ggml/src/ggml-gemmini-utils/src/semantic.cpp:107 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| GEMMINI_MATMUL_MODE | ggml/src/ggml-gemmini/matmul/options.cpp:25 | KEEP execution/cache/numerical policy; not geometry override |
| GEMMINI_RMD_BACKEND | ggml/src/ggml-gemmini/matmul/options.cpp:19; ggml/src/ggml-gemmini/matmul/options.cpp:44 | KEEP execution/cache/numerical policy; not geometry override |
| GEMMINI_STRIPE_JOB_CAPACITY | ggml/src/ggml-gemmini/matmul/options.cpp:37 | KEEP execution/cache/numerical policy; not geometry override |
| GGML_GEMMINI_ANCHOR_DEFAULT | ggml/src/ggml-gemmini/potal-sensitivity.cpp:38 | KEEP software/reference configuration |
| GGML_GEMMINI_ANCHOR_POLICY | ggml/src/ggml-gemmini/potal-sensitivity.cpp:39 | KEEP software/reference configuration |
| GGML_GEMMINI_ANCHOR_TRACE | ggml/src/ggml-gemmini/ggml-gemmini.cpp:180; ggml/src/ggml-gemmini/potal-sensitivity.cpp:196 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| GGML_GEMMINI_CYCLE_DETAIL_LOG | ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:31; ggml/src/ggml-gemmini/quants/act/exsia/exsia-profile.cpp:41 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| GGML_GEMMINI_EXSIA_FORCE_RECOMPUTE | ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:350; ggml/src/ggml-gemmini/quants/act/preparation-cache.cpp:99 | KEEP software/reference configuration |
| GGML_GEMMINI_MODEL_ID | ggml/src/ggml-gemmini/ops.cpp:2238 | KEEP software/reference configuration |
| GGML_GEMMINI_QK_SCHEDULE | ggml/src/ggml-gemmini/matmul/attention.cpp:262 | KEEP software/reference configuration |
| GGML_GEMMINI_RESIDUAL_DEEP_PROFILE | ggml/src/ggml-gemmini/residual/direct/direct-profile.hpp:63 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| GGML_GEMMINI_RUNTIME_BUNDLE_ID | ggml/src/ggml-gemmini/ops.cpp:2237 | KEEP software/reference configuration |
| GGML_GEMMINI_SENSITIVITY_LOG | ggml/src/ggml-gemmini/potal-sensitivity.cpp:52 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| GGML_GEMMINI_TELEMETRY_HASH | ggml/src/ggml-gemmini/matmul/types.hpp:270 | KEEP software/reference configuration |
| GGML_POTAL_DUMP_DIR | ggml/src/ggml-potal-w4a4/ggml-potal.cpp:93; ggml/src/ggml-potal/ggml-potal.cpp:93 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| GGML_POTAL_DUMP_GRAPHS | ggml/src/ggml-potal-w4a4/ggml-potal.cpp:99; ggml/src/ggml-potal/ggml-potal.cpp:99 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| GGML_POTAL_DUMP_NODES | ggml/src/ggml-potal-w4a4/ggml-potal.cpp:98; ggml/src/ggml-potal/ggml-potal.cpp:98 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| GGML_POTAL_FAST_DECODE | ggml/src/ggml-potal-w4a4/residual/stripe-residual.cpp:96; ggml/src/ggml-potal/residual/stripe-residual.cpp:95 | KEEP execution/cache/numerical policy; not geometry override |
| GGML_POTAL_HART_MAP | ggml/src/ggml-potal-w4a4/matmul/execution.cpp:87; ggml/src/ggml-potal/matmul/execution.cpp:86 | KEEP CPU issuer/affinity policy |
| GGML_POTAL_HART_ROLES | ggml/src/ggml-potal-w4a4/matmul/execution.cpp:49; ggml/src/ggml-potal/matmul/execution.cpp:48 | KEEP CPU issuer/affinity policy |
| GGML_POTAL_INDEX_QUEUE | ggml/src/ggml-potal-w4a4/matmul/attention.cpp:600; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4177; ggml/src/ggml-potal/matmul/paired-auto.cpp:4118 | KEEP execution/cache/numerical policy; not geometry override |
| GGML_POTAL_PAIRED_AUTO | ggml/src/ggml-potal-w4a4/ops.cpp:233; ggml/src/ggml-potal/ops.cpp:231 | KEEP execution/cache/numerical policy; not geometry override |
| GGML_POTAL_PAIRED_EARLY_PREFETCH | ggml/src/ggml-potal-w4a4/ops.cpp:277; ggml/src/ggml-potal/ops.cpp:275 | KEEP execution/cache/numerical policy; not geometry override |
| GGML_POTAL_PAIRED_FENCES | ggml/src/ggml-potal-w4a4/ops.cpp:213; ggml/src/ggml-potal/ops.cpp:211 | KEEP execution/cache/numerical policy; not geometry override |
| GGML_POTAL_PAIRED_KTILE | ggml/src/ggml-potal-w4a4/ops.cpp:253; ggml/src/ggml-potal/ops.cpp:251 | KEEP execution/cache/numerical policy; not geometry override |
| GGML_POTAL_PAIRED_PREFETCH | ggml/src/ggml-potal-w4a4/ops.cpp:265; ggml/src/ggml-potal/ops.cpp:263 | KEEP execution/cache/numerical policy; not geometry override |
| GGML_POTAL_PAIRED_STREAM | ggml/src/ggml-potal-w4a4/ops.cpp:289; ggml/src/ggml-potal/ops.cpp:287 | KEEP execution/cache/numerical policy; not geometry override |
| GGML_POTAL_PLAN_LIVE_SCAN | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:1593; ggml/src/ggml-potal/matmul/paired-auto.cpp:1556 | KEEP software/reference configuration |
| GGML_POTAL_PLAN_VALIDATED_REUSE | ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:930; ggml/src/ggml-potal/matmul/stripe.cpp:909 | KEEP execution/cache/numerical policy; not geometry override |
| GGML_POTAL_STRICT_HW_TIMELINE | ggml/src/ggml-potal-w4a4/trace/hw-window.cpp:92; ggml/src/ggml-potal/trace/hw-window.cpp:92 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| GGML_POTAL_STRICT_ISSUER | ggml/src/ggml-potal-w4a4/matmul/execution.cpp:174; ggml/src/ggml-potal/matmul/execution.cpp:173 | KEEP CPU issuer/affinity policy |
| GGML_POTAL_WEIGHT_CACHE_MB | ggml/src/ggml-potal-w4a4/matmul/weight_cache.cpp:23; ggml/src/ggml-potal-w4a4/matmul/weight_cache.cpp:49; ggml/src/ggml-potal/matmul/weight_cache.cpp:23; ggml/src/ggml-potal/matmul/weight_cache.cpp:49 | KEEP execution/cache/numerical policy; not geometry override |
| GGML_POTAL_WINDOW_SCALE_ISSUE | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:1604; ggml/src/ggml-potal/matmul/paired-auto.cpp:1567 | KEEP software/reference configuration |
| POTAL_EARLY_TRACE | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:3073; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4265; ggml/src/ggml-potal/matmul/paired-auto.cpp:3015; ggml/src/ggml-potal/matmul/paired-auto.cpp:4192 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_GRAPH_CPU | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4171; ggml/src/ggml-potal/matmul/paired-auto.cpp:4112 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_H2_COMPLETION_REUSE | ggml/src/ggml-potal-w4a4/potal.c:2338; ggml/src/ggml-potal/potal.c:2238 | KEEP execution/cache/numerical policy; not geometry override |
| POTAL_HW_TIMELINE | ggml/src/ggml-potal-w4a4/trace/hw-window.cpp:201; ggml/src/ggml-potal/trace/hw-window.cpp:201 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_HW_TIMELINE_WINDOWS | ggml/src/ggml-potal-w4a4/trace/hw-window.cpp:62; ggml/src/ggml-potal/trace/hw-window.cpp:62 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_INDEX_PROFILE | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:2592; ggml/src/ggml-potal/matmul/paired-auto.cpp:2534 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_INDEX_QUEUE_BUDGET | ggml/src/ggml-potal-w4a4/potal.c:1865; ggml/src/ggml-potal/potal.c:1771 | KEEP execution/cache/numerical policy; not geometry override |
| POTAL_INDEX_QUEUE_FRESH | ggml/src/ggml-potal-w4a4/potal.c:1864; ggml/src/ggml-potal/potal.c:1770 | KEEP execution/cache/numerical policy; not geometry override |
| POTAL_INDEX_QUEUE_USED | ggml/src/ggml-potal-w4a4/potal.c:1868; ggml/src/ggml-potal/potal.c:1774 | KEEP execution/cache/numerical policy; not geometry override |
| POTAL_INDEX_WINDOW_PROFILE | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4185; ggml/src/ggml-potal/matmul/paired-auto.cpp:4123 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_PAIRED_AUTO_JK | ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:1582; ggml/src/ggml-potal/matmul/stripe.cpp:1560 | KEEP execution/cache/numerical policy; not geometry override |
| POTAL_PAIRED_DIAG_INDEX_ENTRIES | ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:795; ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:1574; ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:1599; ggml/src/ggml-potal/matmul/stripe.cpp:794; ggml/src/ggml-potal/matmul/stripe.cpp:1552; ggml/src/ggml-potal/matmul/stripe.cpp:1577 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_PAIRED_J8K8_FFN | ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:816; ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:1588; ggml/src/ggml-potal/matmul/stripe.cpp:815; ggml/src/ggml-potal/matmul/stripe.cpp:1566 | KEEP execution/cache/numerical policy; not geometry override |
| POTAL_PAIRED_PROFILE | ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:788; ggml/src/ggml-potal-w4a4/residual/stripe-residual.cpp:46; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4173; ggml/src/ggml-potal/matmul/stripe.cpp:787; ggml/src/ggml-potal/residual/stripe-residual.cpp:45; ggml/src/ggml-potal/matmul/paired-auto.cpp:4114 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_PAIRED_STAGE_DIAG | ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:1128; ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:1574; ggml/src/ggml-potal/matmul/stripe.cpp:1107; ggml/src/ggml-potal/matmul/stripe.cpp:1552 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_PAIRED_TIMING | ggml/src/ggml-potal-w4a4/log.hpp:22; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4175; ggml/src/ggml-potal/log.hpp:22; ggml/src/ggml-potal/matmul/paired-auto.cpp:4116 | KEEP execution/cache/numerical policy; not geometry override |
| POTAL_PAIRED_WINDOW_DIAG | ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:797; ggml/src/ggml-potal-w4a4/matmul/stripe.cpp:1575; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:3120; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4265; ggml/src/ggml-potal/matmul/stripe.cpp:796; ggml/src/ggml-potal/matmul/stripe.cpp:1553; ggml/src/ggml-potal/matmul/paired-auto.cpp:3062; ggml/src/ggml-potal/matmul/paired-auto.cpp:4192 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_PLANNING_PROFILE | ggml/src/ggml-potal-w4a4/matmul/planning-profile.cpp:38; ggml/src/ggml-potal/matmul/planning-profile.cpp:38 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_QK_POLICY | ggml/src/ggml-potal-w4a4/matmul/attention.cpp:172; ggml/src/ggml-potal/matmul/attention.cpp:165 | KEEP software/reference configuration |
| POTAL_RAW_TRACE | ggml/src/ggml-potal-w4a4/potal.c:1401; ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:4264; ggml/src/ggml-potal/potal.c:1307; ggml/src/ggml-potal/matmul/paired-auto.cpp:4191 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_SCALE_CPU | ggml/src/ggml-potal-w4a4/potal.c:1403; ggml/src/ggml-potal/potal.c:1309 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_STREAM_ADMISSION_PROFILE | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:2681; ggml/src/ggml-potal/matmul/paired-auto.cpp:2623 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_STREAM_COUNTERS | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:2679; ggml/src/ggml-potal/matmul/paired-auto.cpp:2621 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_STREAM_MANIFEST | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:2559; ggml/src/ggml-potal/matmul/paired-auto.cpp:2501 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_STREAM_PREPARE_AUDIT | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:2579; ggml/src/ggml-potal/matmul/paired-auto.cpp:2521 | KEEP execution/cache/numerical policy; not geometry override |
| POTAL_STREAM_PREPARE_INDEXED | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:2336; ggml/src/ggml-potal/matmul/paired-auto.cpp:2287 | KEEP execution/cache/numerical policy; not geometry override |
| POTAL_STREAM_RELEASE_FAST | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:2683; ggml/src/ggml-potal/matmul/paired-auto.cpp:2625 | KEEP execution/cache/numerical policy; not geometry override |
| POTAL_STREAM_WRITE_FAST | ggml/src/ggml-potal-w4a4/matmul/paired-auto.cpp:2685; ggml/src/ggml-potal/matmul/paired-auto.cpp:2627 | KEEP execution/cache/numerical policy; not geometry override |
| POTAL_TIMELINE | ggml/src/ggml-potal-w4a4/trace/timeline.cpp:311; ggml/src/ggml-potal/trace/timeline.cpp:311 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_TIMELINE_CPU | ggml/src/ggml-potal-w4a4/trace/timeline.cpp:321; ggml/src/ggml-potal/trace/timeline.cpp:321 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_TIMELINE_PROFILE | ggml/src/ggml-potal-w4a4/trace/timeline.cpp:318; ggml/src/ggml-potal/trace/timeline.cpp:318 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |
| POTAL_TIMELINE_REUSE | ggml/src/ggml-potal-w4a4/trace/timeline.cpp:323; ggml/src/ggml-potal/trace/timeline.cpp:323 | KEEP profiling/experiment; exclude from primary performance unless explicitly measured |

Indirect getenv keys are passed by name for timeline serialization/profiling, GPU helpers and logging; the literal-key/dispatch definitions are part of their source contracts. They are not additional CMake options (ggml/src/ggml-potal/trace/timeline.cpp:153 and ggml/src/ggml-potal/trace/timeline.cpp:238; ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:1). See the following source-literal ledger for POTAL_TIMELINE keys, including those looked up through helper arguments.

- m += ",model_id=" + encode(env_or("POTAL_TIMELINE_MODEL_ID", "unknown")); (ggml/src/ggml-potal/trace/timeline.cpp:268).
- m += ",build_id=" + encode(env_or("POTAL_TIMELINE_BUILD_ID", build_default)); (ggml/src/ggml-potal/trace/timeline.cpp:269).
- m += ",image_id=" + encode(env_or("POTAL_TIMELINE_IMAGE_ID", "unknown")); (ggml/src/ggml-potal/trace/timeline.cpp:270).
- const char * p = std::getenv("POTAL_TIMELINE"); (ggml/src/ggml-potal/trace/timeline.cpp:311).
- s.events_cap  = parse_count("POTAL_TIMELINE_EVENTS", kDefaultEvents); (ggml/src/ggml-potal/trace/timeline.cpp:316).
- s.anchors_cap = parse_count("POTAL_TIMELINE_ANCHORS", kDefaultAnchors); (ggml/src/ggml-potal/trace/timeline.cpp:317).
- const char * profile = std::getenv("POTAL_TIMELINE_PROFILE"); (ggml/src/ggml-potal/trace/timeline.cpp:318).
- const char * cpu = std::getenv("POTAL_TIMELINE_CPU"); (ggml/src/ggml-potal/trace/timeline.cpp:321).
- const char * reuse = std::getenv("POTAL_TIMELINE_REUSE"); (ggml/src/ggml-potal/trace/timeline.cpp:323).
- const uint32_t cap = r == role::issuer ? parse_count("POTAL_TIMELINE_ISSUER_EVENTS", s.events_cap) : s.events_cap; (ggml/src/ggml-potal/trace/timeline.cpp:513).
- m += ",model_id=" + encode(env_or("POTAL_TIMELINE_MODEL_ID", "unknown")); (ggml/src/ggml-potal-w4a4/trace/timeline.cpp:268).
- m += ",build_id=" + encode(env_or("POTAL_TIMELINE_BUILD_ID", build_default)); (ggml/src/ggml-potal-w4a4/trace/timeline.cpp:269).
- m += ",image_id=" + encode(env_or("POTAL_TIMELINE_IMAGE_ID", "unknown")); (ggml/src/ggml-potal-w4a4/trace/timeline.cpp:270).
- const char * p = std::getenv("POTAL_TIMELINE"); (ggml/src/ggml-potal-w4a4/trace/timeline.cpp:311).
- s.events_cap  = parse_count("POTAL_TIMELINE_EVENTS", kDefaultEvents); (ggml/src/ggml-potal-w4a4/trace/timeline.cpp:316).
- s.anchors_cap = parse_count("POTAL_TIMELINE_ANCHORS", kDefaultAnchors); (ggml/src/ggml-potal-w4a4/trace/timeline.cpp:317).
- const char * profile = std::getenv("POTAL_TIMELINE_PROFILE"); (ggml/src/ggml-potal-w4a4/trace/timeline.cpp:318).
- const char * cpu = std::getenv("POTAL_TIMELINE_CPU"); (ggml/src/ggml-potal-w4a4/trace/timeline.cpp:321).
- const char * reuse = std::getenv("POTAL_TIMELINE_REUSE"); (ggml/src/ggml-potal-w4a4/trace/timeline.cpp:323).
- const uint32_t cap = r == role::issuer ? parse_count("POTAL_TIMELINE_ISSUER_EVENTS", s.events_cap) : s.events_cap; (ggml/src/ggml-potal-w4a4/trace/timeline.cpp:513).
- constexpr const char * enable_env = "GGML_GEMMINI_CUDA_CPU_EXACT"; (ggml/src/ggml-gemmini/ggml-gpu-cpu-exact.hpp:60).

## Absolute server-path hit inventory

Whole-tree text search, including ignored/hidden files, excluded build* directories and VCS internals. Binary GGUF/media files were skipped; binary payload paths are UNVERIFIED and not build/script source. The search engine prefixes files larger than 4 MiB; the sole oversized text file wikitext-2-raw/wiki.train.raw was separately read through EOF and had no requested path matches. There were 144 matching text lines in 32 files before these audit documents were created. These are substring matches: /workspace/ can occur inside a relative or home-relative documented path. No /home/ubuntu/ or /opt/riscv/ hit was found in the searched text. Each positive hit below carries file:line evidence; absence is a bounded scan result, not a claim about external repositories.

| Source hit | Classification | Matched line |
|---|---|---|
| H3-6B0-scale-path-analysis.md:259 | doc-only | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| H3-6B0-scale-path-analysis.md:260 | doc-only | cd /home/alveo/firesim/target-design/chipyard |
| H3-6B0-scale-path-analysis.md:262 | doc-only | cd generators/potal/software/workspace/llama.cpp-potal |
| H3-6B0-scale-path-analysis.md:269 | doc-only | `/home/alveo/firesim/.conda-env` were usable, and both builds exited zero. |
| H3-6B1-trace-cache-optimization.md:355 | doc-only | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| H3-6B1-trace-cache-optimization.md:356 | doc-only | cd /home/alveo/firesim/target-design/chipyard |
| H3-6B1-trace-cache-optimization.md:358 | doc-only | cd generators/potal/software/workspace/llama.cpp-potal |
| H3-6B1-trace-cache-optimization.md:366 | doc-only | configured compiler under `/home/alveo/firesim/.conda-env` worked. Both builds |
| H3-6B2-trace-cold-path-report.md:416 | doc-only | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| H3-6B2-trace-cold-path-report.md:417 | doc-only | cd /home/alveo/firesim/target-design/chipyard |
| H3-6B2-trace-cold-path-report.md:419 | doc-only | cd generators/potal/software/workspace/llama.cpp-potal |
| H3-6B2-trace-cold-path-report.md:426 | doc-only | compiler under `/home/alveo/firesim/.conda-env`. |
| H3-6B3-checked-dispatch-report.md:61 | doc-only | `/home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc`, |
| H3-6B3-checked-dispatch-report.md:419 | doc-only | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| H3-6B3-checked-dispatch-report.md:420 | doc-only | cd /home/alveo/firesim/target-design/chipyard |
| H3-6B3-checked-dispatch-report.md:422 | doc-only | cd generators/potal/software/workspace/llama.cpp-potal |
| H3-6C-fallback-decode-reuse-report.md:175 | doc-only | `/home/alveo/firesim/.conda-env/riscv-tools/bin/`, Release `-O3 -DNDEBUG`, |
| H3-6C-fallback-decode-reuse-report.md:184 | doc-only | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| H3-6C-fallback-decode-reuse-report.md:185 | doc-only | cd /home/alveo/firesim/target-design/chipyard |
| H3-6C-fallback-decode-reuse-report.md:187 | doc-only | cd generators/potal/software/workspace/llama.cpp-potal |
| H3-6D1-paired-planning-validation-analysis.md:423 | doc-only | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| H3-6D1-paired-planning-validation-analysis.md:424 | doc-only | cd /home/alveo/firesim/target-design/chipyard |
| H3-6D1-paired-planning-validation-analysis.md:426 | doc-only | cd generators/potal/software/workspace/llama.cpp-potal |
| H3-6D1-paired-planning-validation-analysis.md:429 | doc-only | The existing DIM16 Release configuration is reused. Compiler is `/home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-g++`, GCC `12.2.0 (g2ee5e430018)`. Relevant flags are `-O3 -DNDEBUG -std=gnu++17 -fPIC -pthread`, the existing warning flags and the same generated DIM16 include/definition configuration. Activation is A8W8 ExSIA, block size 32, compensation ON, hardware backend, no OpenMP. No compiler, hardware parameter or numerical option was changed. |
| H3-6D1-paired-planning-validation-analysis.md:446 | doc-only | One clean copy of the powered-off B4 guest rootfs was used at `/tmp/h3-6d1-run/rootfs.img`. The existing U280 driver and fixed image in `/home/alveo/FIRESIM_RUNS_DIR/sim_slot_0` were used without FPGA/RTL rebuild. The baseline and diagnostic executables/libraries coexist in `/root/h3-6d1/{baseline,diagnostic}`. |
| H3-6E4C1-sequential-paired-microtile-integration.md:8 | doc-only | F = /home/alveo/firesim |
| H3-6E4C1-sequential-paired-microtile-integration.md:11 | doc-only | L = P/software/workspace/llama.cpp-potal |
| H3-6E4C1-sequential-paired-microtile-integration.md:12 | doc-only | S = /home/alveo/h3-e4b2/source |
| H3-6E4C1-sequential-paired-microtile-integration.md:78 | doc-only | `/home/alveo/h3-e4b2/config_build_recipes.yaml`, build alias |
| H3-6E4C2-0-prefetch-admission-completion-feasibility.md:36 | doc-only | F = /home/alveo/firesim |
| H3-6E4C2-0-prefetch-admission-completion-feasibility.md:39 | doc-only | L = P/software/workspace/llama.cpp-potal |
| H3-6E4C2-0-prefetch-admission-completion-feasibility.md:41 | doc-only | S = /home/alveo/h3-e4b2/source |
| H3-6E4C2-0-prefetch-admission-completion-feasibility.md:517 | doc-only |   -DPOTAL_HW_PARAMS=/home/alveo/firesim/target-design/chipyard/generators/potal/software/gemmini-rocc-tests/include/gemmini_potal_params.h \ |
| H3-7-GPT2-paired-attention-e2e-report.md:515 | doc-only | Conda path. Activating the already installed `/home/alveo/firesim/.conda-env` |
| H3-7-GPT2-paired-attention-e2e-report.md:634 | doc-only |   -m /root/workspace/llama.cpp-potal/models/gpt2/gpt2.Q8_HP1.gguf \ |
| H3-7A-llama-full-inference-baseline.md:57 | doc-only | /home/alveo/firesim/target-design/chipyard/generators/potal/software/workspace/llama.cpp-potal |
| H3-7A-llama-full-inference-baseline.md:69 | doc-only | `/home/alveo/firesim/.conda-env/riscv-tools/bin/`. |
| H3-7A-llama-full-inference-baseline.md:86 | doc-only | /home/alveo/FIRESIM_RUNS_DIR/sim_slot_0/FireSim-xilinx_alveo_u280 |
| H3-7A-llama-full-inference-baseline.md:102 | doc-only | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| H3-7A-llama-full-inference-baseline.md:103 | doc-only | cd /home/alveo/firesim/target-design/chipyard |
| H3-7C-attention-pv-range-resolution-report.md:370 | doc-only |   -m /root/workspace/llama.cpp-potal/models/gpt2/gpt2.Q8_HP1.gguf \ |
| H3-8A-potal-target-build-profile.md:25 | doc-only | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| H3-8A-potal-target-build-profile.md:26 | doc-only | cd /home/alveo/firesim/target-design/chipyard |
| H3-8A-potal-target-build-profile.md:28 | doc-only | conda activate /home/alveo/firesim/.conda-env |
| H3-8A-potal-target-build-profile.md:29 | doc-only | cd generators/potal/software/workspace/llama.cpp-potal |
| H3-8C-dim32-firesim-e2e-report.md:35 | doc-only | /home/alveo/firesim/deploy/results-build/2026-10-24--02-24-42-alveo_u280_firesim_potal_quadcore_d16_no_nic/cl_xilinx_alveo_u280-firesim-FireSim-FireSimQuadRocketSinglePoTalD16Config-BaseXilinxAlveoConfig/vivado_proj/firesim.bit |
| H3-8C-dim32-firesim-e2e-report.md:49 | doc-only | /home/alveo/firesim/deploy/config_build_recipes_h3_8_d32.yaml |
| H3-8C-dim32-firesim-e2e-report.md:50 | doc-only | /home/alveo/firesim/deploy/config_build_h3_8_d32.yaml |
| H3-8C-dim32-firesim-e2e-report.md:56 | doc-only | Build destination: `/home/alveo/build_firesim_h3_8_d32`. |
| H3-8C-dim32-firesim-e2e-report.md:66 | doc-only | /home/alveo/firesim/deploy/results-build/2026-10-28--06-40-13-alveo_u280_firesim_potal_quadcore_d32_no_nic/cl_xilinx_alveo_u280-firesim-FireSim-FireSimQuadRocketSinglePoTalD32Config-BaseXilinxAlveoConfig/firesim.tar.gz |
| H3-8C-dim32-firesim-e2e-report.md:83 | doc-only | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| H3-8C-dim32-firesim-e2e-report.md:84 | doc-only | cd /home/alveo/firesim/target-design/chipyard |
| H3-8C-dim32-firesim-e2e-report.md:86 | doc-only | conda activate /home/alveo/firesim/.conda-env |
| H3-8C-dim32-firesim-e2e-report.md:87 | doc-only | cd /home/alveo/firesim |
| H3-8C-dim32-firesim-e2e-report.md:103 | doc-only | /home/alveo/firesim/deploy/config_runtime_h3_8_d32.yaml |
| H3-8C-dim32-firesim-e2e-report.md:104 | doc-only | /home/alveo/firesim/deploy/workloads/h3-8-d32-interactive.json |
| H3-8C-dim32-firesim-e2e-report.md:107 | doc-only | The simulation directory is `/home/alveo/FIRESIM_RUNS_DIR_H3_8_D32`, |
| H3-8C-dim32-firesim-e2e-report.md:276 | doc-only | /home/alveo/firesim/sim/generated-src/xilinx_alveo_u280/xilinx_alveo_u280-firesim-FireSim-FireSimQuadRocketSinglePoTalD32Config-BaseXilinxAlveoConfig/ |
| H3-8C-dim32-firesim-e2e-report.md:497 | doc-only |   -m /root/workspace/llama.cpp-potal/models/gpt2/gpt2.Q8_HP1.gguf \ |
| H3-8D-final-report.md:38 | doc-only | working `/home/alveo/firesim/.conda-env`. |
| H3-8D0-matched-performance-baseline.md:211 | doc-only | separate slots are `/home/alveo/FIRESIM_RUNS_DIR_H3_8D0_D{16,32}`. |
| H3-8D2-next-k-prefetch-implementation.md:250 | doc-only | - Complete Vivado build: `/home/alveo/build_firesim_h3_8d2_d32`. |
| H3-8D2-next-k-prefetch-implementation.md:251 | doc-only | - Runtime slot: `/home/alveo/FIRESIM_RUNS_DIR_H3_8D2_D32/sim_slot_0`. |
| H3-8E-potal-pipeline-full-audit-and-optimization-plan.md:139 | doc-only | `/home/alveo/firesim/deploy/` are outside the two requested repositories; |
| H3-8F0-original-gemmini-prefetch-audit.md:19 | doc-only | Root: `/home/alveo/firesim/target-design/chipyard`. |
| H3-8F0-original-gemmini-prefetch-audit.md:43 | doc-only |   `generators/potal/software/workspace/llama.cpp-potal`; |
| H3-8F0-original-gemmini-prefetch-audit.md:473 | doc-only | the existing `/home/alveo/firesim/.conda-env` afterward restores CMake |
| H3-8FG-performance-comparison.md:25 | doc-only | - Final corrected build: /home/alveo/build_firesim_h3_8g2_d32, 30 MHz host target. |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:83 | doc-only | F = /home/alveo/firesim |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:86 | doc-only | L = P/software/workspace/llama.cpp-potal |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:87 | doc-only | S = /home/alveo/h3-e4b2/source |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:481 | doc-only | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:482 | doc-only | cd /home/alveo/firesim/target-design/chipyard |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:484 | doc-only | source /home/alveo/firesim/env.sh |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:519 | doc-only | cd /home/alveo/h3-e4b2/source/target-design/chipyard |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:542 | doc-only | cd /home/alveo/h3-e4b2/source/target-design/chipyard/sims/verilator |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:544 | doc-only |  CLASSPATH_CACHE=/home/alveo/h3-e4b2/source/target-design/chipyard/.classpath_cache-e4b2 \ |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:547 | doc-only |  CLASSPATH_CACHE=/home/alveo/h3-e4b2/source/target-design/chipyard/.classpath_cache-e4b2 \ |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:556 | doc-only | cd /home/alveo/firesim/target-design/chipyard/generators/potal/software/workspace/llama.cpp-potal |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:557 | doc-only | COMMON=/home/alveo/firesim/target-design/chipyard/generators/potal/software/gemmini-rocc-tests/riscv-tests/benchmarks/common |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:633 | doc-only | cd /home/alveo/firesim/target-design/chipyard/generators/potal/software/workspace/llama.cpp-potal |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:634 | doc-only | PARAMS=/home/alveo/firesim/target-design/chipyard/generators/potal/software/gemmini-rocc-tests/include/gemmini_potal_params.h |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:637 | doc-only |  -DCMAKE_C_COMPILER=/home/alveo/firesim/.conda-env/bin/cc \ |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:638 | doc-only |  -DCMAKE_CXX_COMPILER=/home/alveo/firesim/.conda-env/bin/c++ \ |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:737 | doc-only | /home/alveo/h3-e4b2/config_build.yaml |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:738 | doc-only | /home/alveo/h3-e4b2/config_build_recipes.yaml |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:740 | doc-only | default_build_dir=/home/alveo/h3-e4b2/build_firesim |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:772 | doc-only | cd /home/alveo/h3-e4b2/source/deploy |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:773 | doc-only | ./firesim -b /home/alveo/h3-e4b2/config_build.yaml \ |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:774 | doc-only |  -r /home/alveo/h3-e4b2/config_build_recipes.yaml buildbitstream |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:991 | doc-only | F=/home/alveo/firesim |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:992 | doc-only | S=/home/alveo/h3-e4b2/source |
| H3-6E4B2-resident-microtile-rtl-isa-implementation.md:1014 | doc-only | BOOT=/home/alveo/FIRESIM_RUNS_DIR/sim_slot_0/gemmini-tests-interactive0-gemmini-tests-interactive-bin |
| H3-8G-gemmini-based-early-prefetch-implementation.md:212 | doc-only |   `/home/alveo/firesim/deploy/`. |
| H3-8G-gemmini-based-early-prefetch-implementation.md:213 | doc-only | - New build directory: `/home/alveo/build_firesim_h3_8g_d32`. |
| H3-8G-gemmini-based-early-prefetch-implementation.md:379 | doc-only | The corrected build uses `/home/alveo/build_firesim_h3_8g2_d32` and a |
| H3-8H-paired-streaming-implementation.md:56 | doc-only | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| H3-8H-paired-streaming-implementation.md:57 | doc-only | cd /home/alveo/firesim/target-design/chipyard |
| H3-8H-paired-streaming-implementation.md:59 | doc-only | conda activate /home/alveo/firesim/.conda-env |
| H3-8H-paired-streaming-implementation.md:60 | doc-only | bun=/home/alveo/.omo/agent/bundled-bun/ca66da8b3b88cc87/bun |
| H3-8H-paired-streaming-implementation.md:133 | doc-only | `/home/alveo/firesim/.conda-env`. Chipyard env.sh references a missing local |
| H3-8H-paired-streaming-implementation.md:939 | doc-only | - Isolated FPGA build root: `/home/alveo/build_firesim_h3_8h_d32`. |
| H3-8H-paired-streaming-implementation.md:1225 | doc-only | `/home/alveo/build_firesim_h3_8g2_d32/platforms/xilinx_alveo_u280/cl_xilinx_alveo_u280-firesim-FireSim-FireSimQuadRocketSinglePoTalD32EarlyPrefetchObservationConfig-BaseXilinxAlveoConfig/vivado_proj/firesim.runs/impl_1/design_1_wrapper_methodology_drc_routed.rpt`. |
| cmake/riscv-static-release.cmake:6 | build-critical | set(CMAKE_C_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc) |
| cmake/riscv-static-release.cmake:7 | build-critical | set(CMAKE_CXX_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-g++) |
| cmake/riscv-static-release.cmake:9 | build-critical | set(CMAKE_SYSROOT /home/alveo/firesim/.conda-env/riscv-tools/sysroot) |
| cmake/riscv-static.cmake:6 | build-critical | set(CMAKE_C_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc) |
| cmake/riscv-static.cmake:7 | build-critical | set(CMAKE_CXX_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-g++) |
| cmake/riscv-static.cmake:9 | build-critical | set(CMAKE_SYSROOT /home/alveo/firesim/.conda-env/riscv-tools/sysroot) |
| cmake/riscv.cmake:6 | build-critical | set(CMAKE_C_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc) |
| cmake/riscv.cmake:7 | build-critical | set(CMAKE_CXX_COMPILER /home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-g++) |
| cmake/riscv.cmake:9 | build-critical | set(CMAKE_SYSROOT /home/alveo/firesim/.conda-env/riscv-tools/sysroot) |
| ggml/src/ggml-gemmini/fp32/tools/check-options.sh:2 | script default (fixed lab setup; critical when invoked) | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| ggml/src/ggml-gemmini/fp32/tools/check-options.sh:3 | script default (fixed lab setup; critical when invoked) | cd /home/alveo/firesim/target-design/chipyard |
| ggml/src/ggml-gemmini/fp32/tools/check-options.sh:5 | script default (fixed lab setup; critical when invoked) | conda activate /home/alveo/firesim/.conda-env |
| ggml/src/ggml-gemmini/fp32/tools/check-options.sh:7 | script default (fixed lab setup; critical when invoked) | L=/home/alveo/firesim/target-design/chipyard/generators/potal/software/workspace/llama.cpp-potal |
| ggml/src/ggml-gemmini/fp32/tools/offline.sh:2 | script default (fixed lab setup; critical when invoked) | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| ggml/src/ggml-gemmini/fp32/tools/offline.sh:3 | script default (fixed lab setup; critical when invoked) | cd /home/alveo/firesim/target-design/chipyard |
| ggml/src/ggml-gemmini/fp32/tools/offline.sh:5 | script default (fixed lab setup; critical when invoked) | conda activate /home/alveo/firesim/.conda-env |
| ggml/src/ggml-gemmini/fp32/tools/offline.sh:7 | script default (fixed lab setup; critical when invoked) | L=/home/alveo/firesim/target-design/chipyard/generators/potal/software/workspace/llama.cpp-potal |
| ggml/src/ggml-gemmini/fp32/tools/rtl-closure.sh:2 | script default (fixed lab setup; critical when invoked) | source /home/alveo/miniforge3/etc/profile.d/conda.sh |
| ggml/src/ggml-gemmini/fp32/tools/rtl-closure.sh:3 | script default (fixed lab setup; critical when invoked) | cd /home/alveo/firesim/target-design/chipyard |
| ggml/src/ggml-gemmini/fp32/tools/rtl-closure.sh:5 | script default (fixed lab setup; critical when invoked) | conda activate /home/alveo/firesim/.conda-env |
| ggml/src/ggml-gemmini/fp32/tools/rtl-closure.sh:7 | script default (fixed lab setup; critical when invoked) | L=/home/alveo/firesim/target-design/chipyard/generators/potal/software/workspace/llama.cpp-potal |
| ggml/src/ggml-potal-w4a4/README.md:466 | doc-only | LLAMA=~/firesim/workspace/potal/software/workspace/llama.cpp-potal |
| ggml/src/ggml-potal-w4a4/README.md:467 | doc-only | PARAMS=~/firesim/workspace/potal/software/gemmini-rocc-tests/include |
| ggml/src/ggml-potal-w4a4/README.md:517 | doc-only | B=/root/workspace/llama.cpp-potal/build-potal-ggml-d16/bin |
| ggml/src/ggml-potal-w4a4/README.md:518 | doc-only | MODEL=/root/workspace/llama.cpp-potal/models/gpt2/gpt2.Q8_HP1.gguf |
| ggml/src/ggml-potal-w4a4/docs/potal-stripe-api.md:322 | doc-only | LLAMA=~/firesim/workspace/potal/software/workspace/llama.cpp-potal |
| ggml/src/ggml-potal-w4a4/docs/potal-stripe-api.md:323 | doc-only | PARAMS=~/firesim/workspace/potal/software/gemmini-rocc-tests/include/gemmini_potal_params.h |
| ggml/src/ggml-potal-w4a4/docs/potal-stripe-api.md:371 | doc-only | 1. Deploy as before: `~/firesim/workspace/update_rootfs_potal.sh` (marshal |
| ggml/src/ggml-potal-w4a4/docs/potal-stripe-api.md:378 | doc-only | S=/root/workspace/llama.cpp-potal/build-potal-stripe-d16/bin |
| ggml/src/ggml-potal-w4a4/docs/potal-stripe-api.md:379 | doc-only | F=/root/workspace/llama.cpp-potal/build-potal-full-d16/bin |
| ggml/src/ggml-potal-w4a4/docs/potal-stripe-api.md:380 | doc-only | MODEL=/root/workspace/llama.cpp-potal/models/gpt2/gpt2.Q8_HP1.gguf |
| ggml/src/ggml-potal/README.md:466 | doc-only | LLAMA=~/firesim/workspace/potal/software/workspace/llama.cpp-potal |
| ggml/src/ggml-potal/README.md:467 | doc-only | PARAMS=~/firesim/workspace/potal/software/gemmini-rocc-tests/include |
| ggml/src/ggml-potal/README.md:517 | doc-only | B=/root/workspace/llama.cpp-potal/build-potal-ggml-d16/bin |
| ggml/src/ggml-potal/README.md:518 | doc-only | MODEL=/root/workspace/llama.cpp-potal/models/gpt2/gpt2.Q8_HP1.gguf |
| ggml/src/ggml-potal/docs/potal-stripe-api.md:322 | doc-only | LLAMA=~/firesim/workspace/potal/software/workspace/llama.cpp-potal |
| ggml/src/ggml-potal/docs/potal-stripe-api.md:323 | doc-only | PARAMS=~/firesim/workspace/potal/software/gemmini-rocc-tests/include/gemmini_potal_params.h |
| ggml/src/ggml-potal/docs/potal-stripe-api.md:371 | doc-only | 1. Deploy as before: `~/firesim/workspace/update_rootfs_potal.sh` (marshal |
| ggml/src/ggml-potal/docs/potal-stripe-api.md:378 | doc-only | S=/root/workspace/llama.cpp-potal/build-potal-stripe-d16/bin |
| ggml/src/ggml-potal/docs/potal-stripe-api.md:379 | doc-only | F=/root/workspace/llama.cpp-potal/build-potal-full-d16/bin |
| ggml/src/ggml-potal/docs/potal-stripe-api.md:380 | doc-only | MODEL=/root/workspace/llama.cpp-potal/models/gpt2/gpt2.Q8_HP1.gguf |

## Complete NPU-related test registration matrix

The table includes all explicit add_test sites and Gemmini helper calls in root tests/CMakeLists.txt, both PoTAL backend CMakeLists, INT4 tests/CMakeLists and Gemmini PB/FP32 backend files. Loop-generated names are expanded. Conditions below are the enclosing lexical CMake conditions, not evaluated results. Inherited gates: root tests require LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION (CMakeLists.txt:237-240); Gemmini backend tests require the selected backend and FP32/PB branch; PoTAL backend tests below the API_ONLY return require full GGML_POTAL, static registration, valid precision and geometry (ggml/src/ggml-potal/CMakeLists.txt:174-245). INT4 rows additionally require GGML_POTAL_W4A4 and its fixed DIM32 W4 contract (ggml/src/CMakeLists.txt:321-329; ggml/src/ggml-potal-w4a4/CMakeLists.txt:18-39). POTAL_API_ONLY registers only the pre-return INT8 API tests. An executable target alone is not ctest registration.

| Test name / registration | Conditions (all enclosing gates apply) | Test source / driver |
|---|---|---|
| gemmini-pb-q8-0-block-test (ggml/src/ggml-gemmini/CMakeLists.txt:187) | if(GGML_GEMMINI_PB) at ggml/src/ggml-gemmini/CMakeLists.txt:158; if(LLAMA_BUILD_TESTS) at ggml/src/ggml-gemmini/CMakeLists.txt:176; if(GGML_GEMMINI_PB_WEIGHT_FORMAT STREQUAL "Q8_0") at ggml/src/ggml-gemmini/CMakeLists.txt:178 | target definition ggml/src/ggml-gemmini/CMakeLists.txt:181; ggml/src/ggml-gemmini/tests/gemmini-pb-q8-0-block-test.cpp:1 |
| gemmini-pb-q8-0-attention-test (ggml/src/ggml-gemmini/CMakeLists.txt:189) | if(GGML_GEMMINI_PB) at ggml/src/ggml-gemmini/CMakeLists.txt:158; if(LLAMA_BUILD_TESTS) at ggml/src/ggml-gemmini/CMakeLists.txt:176; if(GGML_GEMMINI_PB_WEIGHT_FORMAT STREQUAL "Q8_0") at ggml/src/ggml-gemmini/CMakeLists.txt:178 | target definition ggml/src/ggml-gemmini/CMakeLists.txt:181; ggml/src/ggml-gemmini/tests/gemmini-pb-q8-0-attention-test.cpp:1 |
| gemmini-pb-block-test (ggml/src/ggml-gemmini/CMakeLists.txt:196) | if(GGML_GEMMINI_PB) at ggml/src/ggml-gemmini/CMakeLists.txt:158; if(LLAMA_BUILD_TESTS) at ggml/src/ggml-gemmini/CMakeLists.txt:176; if(ELSE of (GGML_GEMMINI_PB_WEIGHT_FORMAT STREQUAL "Q8_0")) at ggml/src/ggml-gemmini/CMakeLists.txt:178 | target definition ggml/src/ggml-gemmini/CMakeLists.txt:192; ggml/src/ggml-gemmini/tests/gemmini-pb-block-test.cpp:1 |
| gemmini-pb-attention-test (ggml/src/ggml-gemmini/CMakeLists.txt:202) | if(GGML_GEMMINI_PB) at ggml/src/ggml-gemmini/CMakeLists.txt:158; if(LLAMA_BUILD_TESTS) at ggml/src/ggml-gemmini/CMakeLists.txt:176; if(ELSE of (GGML_GEMMINI_PB_WEIGHT_FORMAT STREQUAL "Q8_0")) at ggml/src/ggml-gemmini/CMakeLists.txt:178 | target definition ggml/src/ggml-gemmini/CMakeLists.txt:198; ggml/src/ggml-gemmini/tests/gemmini-pb-attention-test.cpp:1 |
| gemmini-fp32-block-test (ggml/src/ggml-gemmini/fp32/backend.cmake:27) | if(LLAMA_BUILD_TESTS) at ggml/src/ggml-gemmini/fp32/backend.cmake:20 | target definition ggml/src/ggml-gemmini/fp32/backend.cmake:21 |
| gemmini-fp32-attention-test (ggml/src/ggml-gemmini/fp32/backend.cmake:29) | if(LLAMA_BUILD_TESTS) at ggml/src/ggml-gemmini/fp32/backend.cmake:20 | target definition ggml/src/ggml-gemmini/fp32/backend.cmake:22 |
| potal-api-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:86) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:85 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:82; ggml/src/ggml-potal-w4a4/tests/potal-api-test.c:1 |
| potal-scale-window-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:93) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:92 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:89; ggml/src/ggml-potal-w4a4/tests/potal-scale-window-test.cpp:1 |
| potal-attention-api-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:99) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:98 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:95; ggml/src/ggml-potal-w4a4/tests/potal-attention-api-test.c:1 |
| potal-trace-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:113) | if(POTAL_HW_DIM EQUAL 16 AND CMAKE_SYSTEM_NAME STREQUAL "Linux") at ggml/src/ggml-potal-w4a4/CMakeLists.txt:103; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:112 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:106; ggml/src/ggml-potal-w4a4/tests/potal-trace-test.c:1; ggml/src/ggml-potal-w4a4/potal.c:1 |
| potal-paired-index-encoder-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:125) | if(POTAL_HW_DIM EQUAL 16 AND CMAKE_SYSTEM_NAME STREQUAL "Linux") at ggml/src/ggml-potal-w4a4/CMakeLists.txt:103; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:124 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:120; ggml/src/ggml-potal-w4a4/tests/potal-paired-index-hw-probe.c:1 |
| potal-stripe-api-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:137) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:136 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:132; ggml/src/ggml-potal-w4a4/tests/potal-stripe-api-test.cpp:1; ggml/src/ggml-potal-w4a4/tests/potal-stripe-api-c.c:1 |
| potal-e4b2-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:149) | if(POTAL_E4B2_TESTS) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:141; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:148 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:142; ggml/src/ggml-potal-w4a4/tests/potal-e4b2-test.cpp:1; ggml/src/ggml-potal-w4a4/tests/potal-e4b2-c11-test.c:1 |
| potal-e4c2-0-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:163) | if(POTAL_E4C2_0_TESTS) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:158; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:162 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:159; ggml/src/ggml-potal-w4a4/tests/potal-e4c2-0-test.cpp:1 |
| potal-attention-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:430) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:429 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:425; ggml/src/ggml-potal-w4a4/tests/potal-attention-test.cpp:1 |
| potal-weight-preload-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:437) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:436 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:433; ggml/src/ggml-potal/tests/potal-weight-preload-test.cpp:1 |
| potal-weight-pack-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:450) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:449 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:442; ggml/src/ggml-potal-w4a4/tests/potal-weight-pack-test.cpp:1 |
| potal-plan-deps-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:451) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:449 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:446; ggml/src/ggml-potal-w4a4/tests/potal-plan-deps-test.cpp:1 |
| potal-resident-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:504) | if(POTAL_E4A_TESTS) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:487; if(ELSE of (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:497 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:494; ggml/src/ggml-potal-w4a4/tests/potal-resident-test.cpp:1 |
| potal-resident-model-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:511) | if(POTAL_E4A_TESTS) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:487; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:510 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:506; ggml/src/ggml-potal-w4a4/tests/potal-resident-model-test.cpp:1 |
| potal-e4b1-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:524) | if(POTAL_E4B1_TESTS) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:516; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:523 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:521; ggml/src/ggml-potal-w4a4/tests/potal-e4b1-test.cpp:1 |
| potal-e4c1-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:536) | if(POTAL_E4C1_TESTS) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:529; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:535 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:530; ggml/src/ggml-potal-w4a4/tests/potal-e4c1-test.cpp:1; ggml/src/ggml-potal-w4a4/tests/support/e4c1-sequential.cpp:1 |
| potal-ggml-unit-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:581) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:580 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:577; ggml/src/ggml-potal-w4a4/tests/potal-ggml-unit-test.cpp:1 |
| potal-log-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:592) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:591 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:587; ggml/src/ggml-potal-w4a4/tests/potal-log-test.cpp:1 |
| potal-tiling-unit-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:605) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:604 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:601; ggml/src/ggml-potal-w4a4/tests/potal-tiling-unit-test.cpp:1 |
| potal-timeline-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:616) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:615 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:611; ggml/src/ggml-potal-w4a4/tests/potal-timeline-test.cpp:1; ggml/src/ggml-potal-w4a4/trace/timeline.cpp:1 |
| potal-residual-unit-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:631) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:630 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:624; ggml/src/ggml-potal-w4a4/tests/potal-residual-unit-test.cpp:1 |
| potal-stripe-pipeline-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:646) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:645 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:639; ggml/src/ggml-potal-w4a4/tests/potal-stripe-pipeline-test.cpp:1 |
| potal-hw-timeline-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:658) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:657 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:652; ggml/src/ggml-potal-w4a4/tests/potal-hw-timeline-test.cpp:1 |
| potal-index-queue-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:685) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:684 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:680; ggml/src/ggml-potal-w4a4/tests/potal-index-queue-test.cpp:1 |
| potal-paired-auto-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:694) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:693 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:689; ggml/src/ggml-potal-w4a4/tests/potal-paired-auto-test.cpp:1 |
| potal-paired-execution-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:694) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:693 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:689; ggml/src/ggml-potal-w4a4/tests/potal-paired-execution-test.cpp:1 |
| potal-paired-plan-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:694) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:693 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:689; ggml/src/ggml-potal-w4a4/tests/potal-paired-plan-test.cpp:1 |
| potal-paired-schedule-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:694) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:693 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:689; ggml/src/ggml-potal-w4a4/tests/potal-paired-schedule-test.cpp:1 |
| potal-paired-unit-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:694) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:693 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:689; ggml/src/ggml-potal-w4a4/tests/potal-paired-unit-test.cpp:1 |
| potal-planning-profile-test (ggml/src/ggml-potal-w4a4/CMakeLists.txt:703) | if(POTAL_PLANNING_DIAGNOSTICS) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:698; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/CMakeLists.txt:702 | target definition ggml/src/ggml-potal-w4a4/CMakeLists.txt:699; ggml/src/ggml-potal-w4a4/tests/potal-planning-profile-test.cpp:1 |
| potal-w4a4-golden-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:42) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:41 | target definition ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:37; ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:1 |
| potal-w4a4-qk-transfer-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:44) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:41 | target definition ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:37; ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:1 |
| potal-w4a4-qk-residual-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:48) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:41 | target definition ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:37; ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:1 |
| potal-w4a4-counters-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:52) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:41 | target definition ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:37; ggml/src/ggml-potal-w4a4/tests/potal-w4a4-golden-test.cpp:1 |
| potal-index-payload-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:62) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:57 | target definition ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:58; ggml/src/ggml-potal-w4a4/tests/potal-index-payload-test.cpp:1 |
| potal-api-test (ggml/src/ggml-potal/CMakeLists.txt:57) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:56 | target definition ggml/src/ggml-potal/CMakeLists.txt:53; ggml/src/ggml-potal/tests/potal-api-test.c:1 |
| potal-scale-window-test (ggml/src/ggml-potal/CMakeLists.txt:64) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:63 | target definition ggml/src/ggml-potal/CMakeLists.txt:60; ggml/src/ggml-potal/tests/potal-scale-window-test.cpp:1 |
| potal-attention-api-test (ggml/src/ggml-potal/CMakeLists.txt:70) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:69 | target definition ggml/src/ggml-potal/CMakeLists.txt:66; ggml/src/ggml-potal/tests/potal-attention-api-test.c:1 |
| potal-trace-test (ggml/src/ggml-potal/CMakeLists.txt:84) | if(POTAL_HW_DIM EQUAL 16 AND CMAKE_SYSTEM_NAME STREQUAL "Linux") at ggml/src/ggml-potal/CMakeLists.txt:74; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:83 | target definition ggml/src/ggml-potal/CMakeLists.txt:77; ggml/src/ggml-potal/tests/potal-trace-test.c:1; ggml/src/ggml-potal/potal.c:1 |
| potal-paired-index-encoder-test (ggml/src/ggml-potal/CMakeLists.txt:96) | if(POTAL_HW_DIM EQUAL 16 AND CMAKE_SYSTEM_NAME STREQUAL "Linux") at ggml/src/ggml-potal/CMakeLists.txt:74; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:95 | target definition ggml/src/ggml-potal/CMakeLists.txt:91; ggml/src/ggml-potal/tests/potal-paired-index-hw-probe.c:1 |
| potal-stripe-api-test (ggml/src/ggml-potal/CMakeLists.txt:108) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:107 | target definition ggml/src/ggml-potal/CMakeLists.txt:103; ggml/src/ggml-potal/tests/potal-stripe-api-test.cpp:1; ggml/src/ggml-potal/tests/potal-stripe-api-c.c:1 |
| potal-e4b2-test (ggml/src/ggml-potal/CMakeLists.txt:120) | if(POTAL_E4B2_TESTS) at ggml/src/ggml-potal/CMakeLists.txt:112; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:119 | target definition ggml/src/ggml-potal/CMakeLists.txt:113; ggml/src/ggml-potal/tests/potal-e4b2-test.cpp:1; ggml/src/ggml-potal/tests/potal-e4b2-c11-test.c:1 |
| potal-e4c2-0-test (ggml/src/ggml-potal/CMakeLists.txt:134) | if(POTAL_E4C2_0_TESTS) at ggml/src/ggml-potal/CMakeLists.txt:129; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:133 | target definition ggml/src/ggml-potal/CMakeLists.txt:130; ggml/src/ggml-potal/tests/potal-e4c2-0-test.cpp:1 |
| potal-attention-test (ggml/src/ggml-potal/CMakeLists.txt:390) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:389 | target definition ggml/src/ggml-potal/CMakeLists.txt:385; ggml/src/ggml-potal/tests/potal-attention-test.cpp:1 |
| potal-qk-residual-test (ggml/src/ggml-potal/CMakeLists.txt:391) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:389 | target definition ggml/src/ggml-potal/CMakeLists.txt:385; ggml/src/ggml-potal/tests/potal-attention-test.cpp:1 |
| potal-qk-transfer-test (ggml/src/ggml-potal/CMakeLists.txt:393) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:389 | target definition ggml/src/ggml-potal/CMakeLists.txt:385; ggml/src/ggml-potal/tests/potal-attention-test.cpp:1 |
| potal-weight-preload-test (ggml/src/ggml-potal/CMakeLists.txt:401) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:400 | target definition ggml/src/ggml-potal/CMakeLists.txt:397; ggml/src/ggml-potal/tests/potal-weight-preload-test.cpp:1 |
| potal-resident-test (ggml/src/ggml-potal/CMakeLists.txt:456) | if(POTAL_E4A_TESTS) at ggml/src/ggml-potal/CMakeLists.txt:439; if(ELSE of (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv")) at ggml/src/ggml-potal/CMakeLists.txt:449 | target definition ggml/src/ggml-potal/CMakeLists.txt:446; ggml/src/ggml-potal/tests/potal-resident-test.cpp:1 |
| potal-resident-model-test (ggml/src/ggml-potal/CMakeLists.txt:463) | if(POTAL_E4A_TESTS) at ggml/src/ggml-potal/CMakeLists.txt:439; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:462 | target definition ggml/src/ggml-potal/CMakeLists.txt:458; ggml/src/ggml-potal/tests/potal-resident-model-test.cpp:1 |
| potal-e4b1-test (ggml/src/ggml-potal/CMakeLists.txt:476) | if(POTAL_E4B1_TESTS) at ggml/src/ggml-potal/CMakeLists.txt:468; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:475 | target definition ggml/src/ggml-potal/CMakeLists.txt:473; ggml/src/ggml-potal/tests/potal-e4b1-test.cpp:1 |
| potal-e4c1-test (ggml/src/ggml-potal/CMakeLists.txt:488) | if(POTAL_E4C1_TESTS) at ggml/src/ggml-potal/CMakeLists.txt:481; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:487 | target definition ggml/src/ggml-potal/CMakeLists.txt:482; ggml/src/ggml-potal/tests/potal-e4c1-test.cpp:1; ggml/src/ggml-potal/tests/support/e4c1-sequential.cpp:1 |
| potal-ggml-unit-test (ggml/src/ggml-potal/CMakeLists.txt:533) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:532 | target definition ggml/src/ggml-potal/CMakeLists.txt:529; ggml/src/ggml-potal/tests/potal-ggml-unit-test.cpp:1 |
| potal-log-test (ggml/src/ggml-potal/CMakeLists.txt:544) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:543 | target definition ggml/src/ggml-potal/CMakeLists.txt:539; ggml/src/ggml-potal/tests/potal-log-test.cpp:1 |
| potal-tiling-unit-test (ggml/src/ggml-potal/CMakeLists.txt:557) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:556 | target definition ggml/src/ggml-potal/CMakeLists.txt:553; ggml/src/ggml-potal/tests/potal-tiling-unit-test.cpp:1 |
| potal-timeline-test (ggml/src/ggml-potal/CMakeLists.txt:568) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:567 | target definition ggml/src/ggml-potal/CMakeLists.txt:563; ggml/src/ggml-potal/tests/potal-timeline-test.cpp:1; ggml/src/ggml-potal/trace/timeline.cpp:1 |
| potal-residual-unit-test (ggml/src/ggml-potal/CMakeLists.txt:583) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:582 | target definition ggml/src/ggml-potal/CMakeLists.txt:576; ggml/src/ggml-potal/tests/potal-residual-unit-test.cpp:1 |
| potal-stripe-pipeline-test (ggml/src/ggml-potal/CMakeLists.txt:598) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:597 | target definition ggml/src/ggml-potal/CMakeLists.txt:591; ggml/src/ggml-potal/tests/potal-stripe-pipeline-test.cpp:1 |
| potal-hw-timeline-test (ggml/src/ggml-potal/CMakeLists.txt:610) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:609 | target definition ggml/src/ggml-potal/CMakeLists.txt:604; ggml/src/ggml-potal/tests/potal-hw-timeline-test.cpp:1 |
| potal-index-queue-test (ggml/src/ggml-potal/CMakeLists.txt:637) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:636 | target definition ggml/src/ggml-potal/CMakeLists.txt:632; ggml/src/ggml-potal/tests/potal-index-queue-test.cpp:1 |
| potal-paired-auto-test (ggml/src/ggml-potal/CMakeLists.txt:646) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:645 | target definition ggml/src/ggml-potal/CMakeLists.txt:641; ggml/src/ggml-potal/tests/potal-paired-auto-test.cpp:1 |
| potal-paired-execution-test (ggml/src/ggml-potal/CMakeLists.txt:646) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:645 | target definition ggml/src/ggml-potal/CMakeLists.txt:641; ggml/src/ggml-potal/tests/potal-paired-execution-test.cpp:1 |
| potal-paired-plan-test (ggml/src/ggml-potal/CMakeLists.txt:646) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:645 | target definition ggml/src/ggml-potal/CMakeLists.txt:641; ggml/src/ggml-potal/tests/potal-paired-plan-test.cpp:1 |
| potal-paired-schedule-test (ggml/src/ggml-potal/CMakeLists.txt:646) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:645 | target definition ggml/src/ggml-potal/CMakeLists.txt:641; ggml/src/ggml-potal/tests/potal-paired-schedule-test.cpp:1 |
| potal-paired-unit-test (ggml/src/ggml-potal/CMakeLists.txt:646) | if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:645 | target definition ggml/src/ggml-potal/CMakeLists.txt:641; ggml/src/ggml-potal/tests/potal-paired-unit-test.cpp:1 |
| potal-planning-profile-test (ggml/src/ggml-potal/CMakeLists.txt:655) | if(POTAL_PLANNING_DIAGNOSTICS) at ggml/src/ggml-potal/CMakeLists.txt:650; if(NOT CMAKE_CROSSCOMPILING) at ggml/src/ggml-potal/CMakeLists.txt:654 | target definition ggml/src/ggml-potal/CMakeLists.txt:651; ggml/src/ggml-potal/tests/potal-planning-profile-test.cpp:1 |
| test-cycle-sim-reader (tests/CMakeLists.txt:8) | no additional local gate | target definition tests/CMakeLists.txt:5; tests/test-cycle-sim-reader.cpp:1 |
| test-gemmini-semantic (tests/CMakeLists.txt:15) | if(LOG_CYCLE OR CYCLE_SIM) at tests/CMakeLists.txt:9 | tests/test-gemmini-semantic.cmake:1; target definition tests/CMakeLists.txt:10; tests/test-gemmini-semantic.cpp:1 |
| test-functional-emulation-exclusion (tests/CMakeLists.txt:26) | no additional local gate | target definition tests/CMakeLists.txt:20; tests/test-functional-emulation-exclusion.cpp:1 |
| test-cycle-sim-cpu-smoke (tests/CMakeLists.txt:36) | if(TARGET ggml-cpu AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:27 | tests/test-cycle-sim-cpu-smoke.cmake:1; target definition tests/CMakeLists.txt:28; tests/test-ggml-cpu-cycle-sink.cpp:1 |
| test-gemmini-cycle-sim-log (tests/CMakeLists.txt:48) | if(CYCLE_SIM) at tests/CMakeLists.txt:41 | target definition tests/CMakeLists.txt:43; tests/test-gemmini-cycle-sim-log.cpp:1 |
| test-gemmini-cycle-sim-log-failure (tests/CMakeLists.txt:49) | if(CYCLE_SIM) at tests/CMakeLists.txt:41 | target definition tests/CMakeLists.txt:43; tests/test-gemmini-cycle-sim-log.cpp:1 |
| test-cycle-sim-coverage (tests/CMakeLists.txt:57) | if(CYCLE_SIM) at tests/CMakeLists.txt:41 | target definition tests/CMakeLists.txt:50; tests/test-cycle-sim-coverage.cpp:1 |
| test-gemmini-cycle-sim-dispatch-fail-full (tests/CMakeLists.txt:70) | if(CYCLE_SIM) at tests/CMakeLists.txt:41; if(TARGET ggml-gemmini-cpu-functional) at tests/CMakeLists.txt:58 | tests/test-cycle-sim-dispatch.cmake:1; target definition tests/CMakeLists.txt:59; tests/test-gemmini-cycle-sim-dispatch.cpp:1 |
| test-gemmini-cycle-sim-dispatch-fail-publish-full (tests/CMakeLists.txt:70) | if(CYCLE_SIM) at tests/CMakeLists.txt:41; if(TARGET ggml-gemmini-cpu-functional) at tests/CMakeLists.txt:58 | tests/test-cycle-sim-dispatch.cmake:1; target definition tests/CMakeLists.txt:59; tests/test-gemmini-cycle-sim-dispatch.cpp:1 |
| test-gemmini-cycle-sim-dispatch-fail-publish-residual (tests/CMakeLists.txt:70) | if(CYCLE_SIM) at tests/CMakeLists.txt:41; if(TARGET ggml-gemmini-cpu-functional) at tests/CMakeLists.txt:58 | tests/test-cycle-sim-dispatch.cmake:1; target definition tests/CMakeLists.txt:59; tests/test-gemmini-cycle-sim-dispatch.cpp:1 |
| test-gemmini-cycle-sim-dispatch-fail-residual (tests/CMakeLists.txt:70) | if(CYCLE_SIM) at tests/CMakeLists.txt:41; if(TARGET ggml-gemmini-cpu-functional) at tests/CMakeLists.txt:58 | tests/test-cycle-sim-dispatch.cmake:1; target definition tests/CMakeLists.txt:59; tests/test-gemmini-cycle-sim-dispatch.cpp:1 |
| test-gemmini-cycle-sim-dispatch-full (tests/CMakeLists.txt:70) | if(CYCLE_SIM) at tests/CMakeLists.txt:41; if(TARGET ggml-gemmini-cpu-functional) at tests/CMakeLists.txt:58 | tests/test-cycle-sim-dispatch.cmake:1; target definition tests/CMakeLists.txt:59; tests/test-gemmini-cycle-sim-dispatch.cpp:1 |
| test-gemmini-cycle-sim-dispatch-large-k (tests/CMakeLists.txt:70) | if(CYCLE_SIM) at tests/CMakeLists.txt:41; if(TARGET ggml-gemmini-cpu-functional) at tests/CMakeLists.txt:58 | tests/test-cycle-sim-dispatch.cmake:1; target definition tests/CMakeLists.txt:59; tests/test-gemmini-cycle-sim-dispatch.cpp:1 |
| test-gemmini-cycle-sim-dispatch-pipeline (tests/CMakeLists.txt:70) | if(CYCLE_SIM) at tests/CMakeLists.txt:41; if(TARGET ggml-gemmini-cpu-functional) at tests/CMakeLists.txt:58 | tests/test-cycle-sim-dispatch.cmake:1; target definition tests/CMakeLists.txt:59; tests/test-gemmini-cycle-sim-dispatch.cpp:1 |
| test-gemmini-cycle-sim-dispatch-residual (tests/CMakeLists.txt:70) | if(CYCLE_SIM) at tests/CMakeLists.txt:41; if(TARGET ggml-gemmini-cpu-functional) at tests/CMakeLists.txt:58 | tests/test-cycle-sim-dispatch.cmake:1; target definition tests/CMakeLists.txt:59; tests/test-gemmini-cycle-sim-dispatch.cpp:1 |
| test-gemmini-cycle-sim-dispatch-residual-runs (tests/CMakeLists.txt:70) | if(CYCLE_SIM) at tests/CMakeLists.txt:41; if(TARGET ggml-gemmini-cpu-functional) at tests/CMakeLists.txt:58 | tests/test-cycle-sim-dispatch.cmake:1; target definition tests/CMakeLists.txt:59; tests/test-gemmini-cycle-sim-dispatch.cpp:1 |
| test-cpu-functional-contract (tests/CMakeLists.txt:86) | if(CYCLE_SIM) at tests/CMakeLists.txt:41; if(TARGET ggml-gemmini-cpu-functional) at tests/CMakeLists.txt:58 | target definition tests/CMakeLists.txt:76; ${IM2P_SIM_ROOT}/frontend/tests/cpu_functional_contract.cpp (external source UNVERIFIED) |
| test-cycle-sim-build-contract (tests/CMakeLists.txt:90) | if(TARGET ggml-gemmini AND TARGET ggml-cpu AND LLAMA_BUILD_TOOLS) at tests/CMakeLists.txt:89 | tests/test-cycle-sim-build-contract.cmake:1 |
| test-gemmini-numerical-memory-contract (tests/CMakeLists.txt:98) | if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1" AND NOT CYCLE_SIM) at tests/CMakeLists.txt:96 | tests/test-gemmini-memory-contract.cmake:1 |
| test-experiment-output (tests/CMakeLists.txt:210) | if(NOT WIN32) at tests/CMakeLists.txt:205; if(BASH_EXECUTABLE AND EXISTS "${BASH_EXECUTABLE}") at tests/CMakeLists.txt:209 | tests/test-experiment-output.sh:1 |
| test-gemmini-cycle-cli (tests/CMakeLists.txt:263) | if(LLAMA_BUILD_TOOLS AND LOG_CYCLE) at tests/CMakeLists.txt:262 | tests/test-gemmini-cycle-cli.cmake:1 |
| test-gemmini-cpu-cycle-validity-contract (tests/CMakeLists.txt:269) | no additional local gate | tests/test-gemmini-cpu-cycle-validity-contract.cmake:1 |
| test-build-arm64-log-defaults (tests/CMakeLists.txt:278) | no additional local gate | build-arm64.sh:1; tests/test-build-arm64-log-defaults.cmake:1 |
| test-build-arm64-cpu-detection (tests/CMakeLists.txt:282) | no additional local gate | build-arm64.sh:1; tests/test-build-arm64-cpu-detection.cmake:1 |
| test-gemmini-off-build (tests/CMakeLists.txt:288) | if(UNIX) at tests/CMakeLists.txt:287 | tests/test-gemmini-off-build.cmake:1 |
| test-evaluation-build-options (tests/CMakeLists.txt:293) | no additional local gate | tests/test-evaluation-build-options.cmake:1 |
| test-evaluation-workload (tests/CMakeLists.txt:299) | no additional local gate | target definition tests/CMakeLists.txt:296; tests/test-evaluation-workload.cpp:1 |
| test-evaluation-trace (tests/CMakeLists.txt:305) | no additional local gate | tests/test-evaluation-metric-sink.cmake:1; target definition tests/CMakeLists.txt:300; tests/test-evaluation-trace.cpp:1; tools/eval/evaluation-trace.cpp:1 |
| test-evaluation-lifecycle (tests/CMakeLists.txt:313) | no additional local gate | target definition tests/CMakeLists.txt:309; tests/test-evaluation-lifecycle.cpp:1 |
| test-gemmini-evaluation-metrics (tests/CMakeLists.txt:321) | no additional local gate | tests/test-evaluation-metric-sink.cmake:1; target definition tests/CMakeLists.txt:314; tests/test-gemmini-evaluation-metrics.cpp:1 |
| test-im2p-cache-build-scripts (tests/CMakeLists.txt:325) | no additional local gate | tests/test-im2p-cache-build-scripts.cmake:1 |
| test-potal-target-profile (tests/CMakeLists.txt:330) | no additional local gate | tests/test-potal-target-profile.cmake:1 |
| test-gemmini-log (tests/CMakeLists.txt:347) | no additional local gate | target definition tests/CMakeLists.txt:337; tests/test-gemmini-log.cpp:1 |
| test-llama-log-callback (tests/CMakeLists.txt:349) | no additional local gate | target definition tests/CMakeLists.txt:337; tests/test-gemmini-log.cpp:1 |
| test-gemmini-optrace (tests/CMakeLists.txt:358) | no additional local gate | target definition tests/CMakeLists.txt:351; tests/test-gemmini-optrace.cpp:1 |
| test-gemmini-log-boundary (tests/CMakeLists.txt:376) | no additional local gate | target definition tests/CMakeLists.txt:360; tests/test-gemmini-log-boundary.cpp:1; tests/test-gemmini-log-c-boundary.c:1 |
| test-gemmini-dump-phase (tests/CMakeLists.txt:386) | if(GGML_GEMMINI AND LOG_DUMP AND NOT GGML_BACKEND_DL AND GGML_GEMMINI_OPTION STREQUAL "CPU" AND GGML_GEMMINI_COMPUTE_TYPE STREQUAL "FLOAT" AND NOT GGML_GEMMINI_DEQUANT_FP_TEST) at tests/CMakeLists.txt:378 | target definition tests/CMakeLists.txt:381; tests/test-gemmini-dump-phase.cpp:1 |
| test-gemmini-dump-output (tests/CMakeLists.txt:388) | if(GGML_GEMMINI AND LOG_DUMP AND NOT GGML_BACKEND_DL AND GGML_GEMMINI_OPTION STREQUAL "CPU" AND GGML_GEMMINI_COMPUTE_TYPE STREQUAL "FLOAT" AND NOT GGML_GEMMINI_DEQUANT_FP_TEST) at tests/CMakeLists.txt:378; if(LOG_DUMP_SCALE AND UNIX) at tests/CMakeLists.txt:387 | target definition tests/CMakeLists.txt:381; tests/test-gemmini-dump-phase.cpp:1 |
| test-gemmini-host-timing (tests/CMakeLists.txt:399) | no additional local gate | target definition tests/CMakeLists.txt:391; tests/test-gemmini-host-timing.cpp:1 |
| test-gemmini-trace-context (tests/CMakeLists.txt:406) | no additional local gate | target definition tests/CMakeLists.txt:400; tests/test-gemmini-trace-context.cpp:1 |
| test-evaluation-metric-framework (tests/CMakeLists.txt:412) | if(Python3_Interpreter_FOUND) at tests/CMakeLists.txt:410; if(Python3_VERSION VERSION_GREATER_EQUAL "3.11") at tests/CMakeLists.txt:411 | evaluation/tests/test_framework.py:1 |
| test-host-overlap (tests/CMakeLists.txt:417) | if(Python3_Interpreter_FOUND) at tests/CMakeLists.txt:410 | tests/test-host-overlap.py:1 |
| test-residual-profile (tests/CMakeLists.txt:419) | if(Python3_Interpreter_FOUND) at tests/CMakeLists.txt:410 | tests/test-residual-profile.py:1 |
| test-cycle-timeline (tests/CMakeLists.txt:421) | if(Python3_Interpreter_FOUND) at tests/CMakeLists.txt:410 | tests/test-cycle-timeline.py:1 |
| test-cycle-matrix (tests/CMakeLists.txt:423) | if(Python3_Interpreter_FOUND) at tests/CMakeLists.txt:410 | tests/test-cycle-matrix.py:1 |
| test-gemmini-cycle-reader-aarch64 (tests/CMakeLists.txt:438) | if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64\|arm64)$") at tests/CMakeLists.txt:426 | target definition tests/CMakeLists.txt:428; tests/test-gemmini-cycle-reader-aarch64.cpp:1; ggml/src/ggml-gemmini-utils/src/cycle_reader_aarch64.cpp:1; ggml/src/ggml-gemmini-utils/src/cycle_reader-capi.cpp:1 |
| test-ggml-cpu-cycle-sink (tests/CMakeLists.txt:449) | if(LOG_CYCLE AND GGML_CPU_CYCLE_LOG AND TARGET ggml-cpu AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:441 | target definition tests/CMakeLists.txt:442; tests/test-ggml-cpu-cycle-sink.cpp:1 |
| test-gemmini-log-explicit-debug-before-default (tests/CMakeLists.txt:455) | if(LOG_DEBUG) at tests/CMakeLists.txt:454 | target definition tests/CMakeLists.txt:337; tests/test-gemmini-log.cpp:1 |
| test-gemmini-log-fault-allocation (tests/CMakeLists.txt:460) | if(LOG_CYCLE) at tests/CMakeLists.txt:458 | target definition tests/CMakeLists.txt:337; tests/test-gemmini-log.cpp:1 |
| test-gemmini-log-fault-flush (tests/CMakeLists.txt:460) | if(LOG_CYCLE) at tests/CMakeLists.txt:458 | target definition tests/CMakeLists.txt:337; tests/test-gemmini-log.cpp:1 |
| test-gemmini-log-fault-open (tests/CMakeLists.txt:460) | if(LOG_CYCLE) at tests/CMakeLists.txt:458 | target definition tests/CMakeLists.txt:337; tests/test-gemmini-log.cpp:1 |
| test-gemmini-log-fault-replacement (tests/CMakeLists.txt:460) | if(LOG_CYCLE) at tests/CMakeLists.txt:458 | target definition tests/CMakeLists.txt:337; tests/test-gemmini-log.cpp:1 |
| test-gemmini-log-fault-write (tests/CMakeLists.txt:460) | if(LOG_CYCLE) at tests/CMakeLists.txt:458 | target definition tests/CMakeLists.txt:337; tests/test-gemmini-log.cpp:1 |
| test-gemmini-log-cross-image (tests/CMakeLists.txt:471) | if(UNIX AND BUILD_SHARED_LIBS AND LOG_CYCLE AND TARGET ggml-gemmini) at tests/CMakeLists.txt:466 | target definition tests/CMakeLists.txt:467; tests/test-gemmini-log-cross-image.cpp:1 |
| gemmini_weight_formats (tests/CMakeLists.txt:517) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:516 | tests/test-gemmini-weight-formats.cpp:1 |
| gemmini_mxint_weight_routes_activation (tests/CMakeLists.txt:518) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:516 | target definition tests/CMakeLists.txt:517; tests/test-gemmini-weight-formats.cpp:1 |
| test-gemmini-weight-reader-callers (tests/CMakeLists.txt:526) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-weight-reader-callers.cmake:1 |
| gemmini_native_dequant_fp (tests/CMakeLists.txt:536) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(_GGML_GEMMINI_BUILD_DEQUANT_FP_TARGET) at tests/CMakeLists.txt:534 | tests/test-gemmini-native-dequant-fp.cpp:1 |
| gemmini_deferred_dequant_fp (tests/CMakeLists.txt:566) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(_GGML_GEMMINI_BUILD_DEQUANT_FP_TARGET) at tests/CMakeLists.txt:534; if(GGML_GEMMINI_DEQUANT_FP_TEST) at tests/CMakeLists.txt:564; if(_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "EXSIA") at tests/CMakeLists.txt:565 | target definition tests/CMakeLists.txt:536; tests/test-gemmini-native-dequant-fp.cpp:1 |
| gemmini_dequant_fp (tests/CMakeLists.txt:572) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(_GGML_GEMMINI_BUILD_DEQUANT_FP_TARGET) at tests/CMakeLists.txt:534; if(GGML_GEMMINI_DEQUANT_FP_TEST) at tests/CMakeLists.txt:564; if(ELSE of (_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "EXSIA")) at tests/CMakeLists.txt:565 | target definition tests/CMakeLists.txt:536; tests/test-gemmini-native-dequant-fp.cpp:1 |
| test-gemmini-act-quant-i32-saturation (tests/CMakeLists.txt:605) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:583; tests/test-gemmini-act-quant-i32-saturation.cpp:1; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:1 |
| gemmini_block_metadata (tests/CMakeLists.txt:628) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "BLOCK") at tests/CMakeLists.txt:608 | target definition tests/CMakeLists.txt:609; tests/test-gemmini-block-quant.cpp:1; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:1 |
| gemmini_block_quant (tests/CMakeLists.txt:630) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "BLOCK") at tests/CMakeLists.txt:608 | target definition tests/CMakeLists.txt:609; tests/test-gemmini-block-quant.cpp:1; ggml/src/ggml-gemmini/quants/act/stripe/stripe.cpp:1 |
| gemmini_block_dense (tests/CMakeLists.txt:639) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "BLOCK") at tests/CMakeLists.txt:608 | target definition tests/CMakeLists.txt:633; tests/test-gemmini-block-dense.cpp:1 |
| gemmini_block_end_to_end (tests/CMakeLists.txt:652) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "BLOCK") at tests/CMakeLists.txt:608; if(NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:641 | target definition tests/CMakeLists.txt:642; tests/test-gemmini-block-end-to-end.cpp:1 |
| test-gemmini-exsia-sigma (tests/CMakeLists.txt:682) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:666; tests/test-gemmini-exsia-sigma.cpp:1; ggml/src/ggml-gemmini/quants/act/exsia/local.cpp:1 |
| test-residual-profile-runtime (tests/CMakeLists.txt:754) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(LOG_CYCLE AND CYCLE_DETAIL AND Python3_Interpreter_FOUND) at tests/CMakeLists.txt:753 | tests/test-residual-profile.py:1; target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| gemmini_direct_metrics_source_contract (tests/CMakeLists.txt:758) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-direct-metrics-source-contract.cmake:1 |
| test-gemmini-exsia-workers3 (tests/CMakeLists.txt:838) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(LOG_CYCLE AND CYCLE_DETAIL) at tests/CMakeLists.txt:771 | target definition tests/CMakeLists.txt:773; tests/test-gemmini-exsia-profile.cpp:1 |
| test-gemmini-exsia-workers4 (tests/CMakeLists.txt:838) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(LOG_CYCLE AND CYCLE_DETAIL) at tests/CMakeLists.txt:771 | target definition tests/CMakeLists.txt:773; tests/test-gemmini-exsia-profile.cpp:1 |
| test-gemmini-exsia-workers3-origin-order (tests/CMakeLists.txt:840) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(LOG_CYCLE AND CYCLE_DETAIL) at tests/CMakeLists.txt:771 | target definition tests/CMakeLists.txt:773; tests/test-gemmini-exsia-profile.cpp:1 |
| test-gemmini-exsia-workers4-origin-order (tests/CMakeLists.txt:840) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(LOG_CYCLE AND CYCLE_DETAIL) at tests/CMakeLists.txt:771 | target definition tests/CMakeLists.txt:773; tests/test-gemmini-exsia-profile.cpp:1 |
| test-gemmini-exsia-profile-contract (tests/CMakeLists.txt:845) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-exsia-profile-contract.cmake:1 |
| gemmini_exsia_all (tests/CMakeLists.txt:865) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-exsia-case.cmake:1; target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| gemmini_exsia_baseline (tests/CMakeLists.txt:865) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-exsia-case.cmake:1; target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| gemmini_exsia_direct_executor (tests/CMakeLists.txt:865) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-exsia-case.cmake:1; target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| gemmini_exsia_dispatch (tests/CMakeLists.txt:865) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-exsia-case.cmake:1; target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| gemmini_exsia_hp1_srmd_software_ws (tests/CMakeLists.txt:865) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; A8W8 only; tests/CMakeLists.txt:857-861 | tests/test-gemmini-exsia-case.cmake:1; target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| gemmini_exsia_non_exsia_capture_context (tests/CMakeLists.txt:865) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-exsia-case.cmake:1; target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| gemmini_exsia_q8_srmd_software_ws (tests/CMakeLists.txt:865) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; A8W8 only; tests/CMakeLists.txt:857-861 | tests/test-gemmini-exsia-case.cmake:1; target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| gemmini_exsia_rmd_direct_parity (tests/CMakeLists.txt:865) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; A8W8 only; tests/CMakeLists.txt:857-861 | tests/test-gemmini-exsia-case.cmake:1; target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| gemmini_exsia_rmd_gather (tests/CMakeLists.txt:865) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; A8W8 only; tests/CMakeLists.txt:857-861 | tests/test-gemmini-exsia-case.cmake:1; target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| gemmini_exsia_rmd_routes (tests/CMakeLists.txt:865) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; A8W8 only; tests/CMakeLists.txt:857-861 | tests/test-gemmini-exsia-case.cmake:1; target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| gemmini_exsia_stripe_geometry (tests/CMakeLists.txt:865) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-exsia-case.cmake:1; target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| gemmini_exsia_unknown_case (tests/CMakeLists.txt:875) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:684; tests/test-gemmini-exsia.cpp:1 |
| test-gemmini-exsia-writer-profile-context (tests/CMakeLists.txt:909) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:879; tests/test-gemmini-exsia.cpp:1 |
| gemmini-exsia-output-default (tests/CMakeLists.txt:915) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:879; tests/test-gemmini-exsia.cpp:1 |
| gemmini-exsia-output-dir (tests/CMakeLists.txt:919) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:879; tests/test-gemmini-exsia.cpp:1 |
| gemmini-exsia-output-relative (tests/CMakeLists.txt:923) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:879; tests/test-gemmini-exsia.cpp:1 |
| gemmini-exsia-output-absolute (tests/CMakeLists.txt:927) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:879; tests/test-gemmini-exsia.cpp:1 |
| gemmini-exsia-output-empty (tests/CMakeLists.txt:931) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:879; tests/test-gemmini-exsia.cpp:1 |
| gemmini-exsia-output-invalid-parent (tests/CMakeLists.txt:935) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:879; tests/test-gemmini-exsia.cpp:1 |
| test-gemmini-rmd (tests/CMakeLists.txt:966) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:941 | target definition tests/CMakeLists.txt:942; tests/test-gemmini-rmd.cpp:1; ggml/src/ggml-gemmini/residual/direct/direct-executor.cpp:1; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:1 |
| test-gemmini-rmd-packet (tests/CMakeLists.txt:985) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:969; tests/test-gemmini-rmd-packet.cpp:1; ggml/src/ggml-gemmini/quants/common/weight_reader.cpp:1; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:1 |
| test-gemmini-rmd-bitmap-log0 (tests/CMakeLists.txt:999) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:989; tests/test-gemmini-rmd-bitmap.cpp:1; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:1; ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:1 |
| test-gemmini-rmd-bitmap-log1 (tests/CMakeLists.txt:999) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:989; tests/test-gemmini-rmd-bitmap.cpp:1; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:1; ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:1 |
| test-gemmini-rmd-run-aware (tests/CMakeLists.txt:1039) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:1003; tests/test-gemmini-rmd-run-aware.cpp:1 |
| test-gemmini-evaluation-observer (tests/CMakeLists.txt:1051) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if((GGML_GEMMINI_ACT_QUANT_METRICS OR GGML_GEMMINI_RESIDUAL_METRICS OR GGML_GEMMINI_SCALE_METRICS) AND GGML_GEMMINI_ENABLE_RMD AND GGML_GEMMINI_ACTIVATION_BITS MATCHES "^(4\|8)$") at tests/CMakeLists.txt:1042 | tests/test-evaluation-metric-sink.cmake:1; target definition tests/CMakeLists.txt:1044; tests/test-gemmini-evaluation-observer.cpp:1 |
| test-gemmini-metric-terminal-elision (tests/CMakeLists.txt:1062) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if((GGML_GEMMINI_ACT_QUANT_METRICS OR GGML_GEMMINI_RESIDUAL_METRICS OR GGML_GEMMINI_SCALE_METRICS) AND GGML_GEMMINI_ENABLE_RMD AND GGML_GEMMINI_ACTIVATION_BITS MATCHES "^(4\|8)$") at tests/CMakeLists.txt:1042; if(GGML_GEMMINI_ACT_QUANT_METRICS AND GGML_GEMMINI_RESIDUAL_METRICS AND GGML_GEMMINI_SCALE_METRICS AND GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1054 | target definition tests/CMakeLists.txt:1056; tests/test-gemmini-metric-terminal-elision.cpp:1 |
| test-gemmini-rmd-capture (tests/CMakeLists.txt:1088) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:1068; tests/test-gemmini-rmd-capture.cpp:1; ggml/src/ggml-gemmini/quants/common/weight_reader.cpp:1; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:1; ggml/src/ggml-gemmini/residual/rmd/rmd-bitmap-builder.cpp:1 |
| test-gemmini-rmd-capture-finish-boundary (tests/CMakeLists.txt:1090) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-rmd-capture-finish-boundary.cmake:1 |
| test-gemmini-production-host (tests/CMakeLists.txt:1109) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:1095; tests/test-gemmini-production-host-backend.cpp:1 |
| test-gemmini-rmd-telemetry (tests/CMakeLists.txt:1137) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1113 | target definition tests/CMakeLists.txt:1114; tests/test-gemmini-rmd-telemetry.cpp:1 |
| test-gemmini-rmd-telemetry-typed-metadata (tests/CMakeLists.txt:1140) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1113; if(LOG_CYCLE) at tests/CMakeLists.txt:1139 | target definition tests/CMakeLists.txt:1114; tests/test-gemmini-rmd-telemetry.cpp:1 |
| test-gemmini-aggregate-cycle-driver (tests/CMakeLists.txt:1144) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1113 | target definition tests/CMakeLists.txt:1114; tests/test-gemmini-rmd-telemetry.cpp:1 |
| test-gemmini-rmd-reducer (tests/CMakeLists.txt:1164) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | target definition tests/CMakeLists.txt:1149; tests/test-gemmini-rmd-telemetry.cpp:1; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:1 |
| test-gemmini-im2p-host-cycle (tests/CMakeLists.txt:1190) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(UNIX AND EXISTS "${IM2P_SIM_ROOT}/frontend/include/im2p_gemmini_frontend.hpp") at tests/CMakeLists.txt:1166 | target definition tests/CMakeLists.txt:1167; tests/test-gemmini-im2p-routing.cpp:1; ggml/src/ggml-gemmini/ggml-gemmini-telemetry.cpp:1; ggml/src/ggml-gemmini/residual/rmd/rmd-builder.cpp:1; ggml/src/ggml-gemmini/im2p/route.cpp:1 |
| test-gemmini-telemetry-clock-gate (tests/CMakeLists.txt:1192) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-telemetry-clock-gate.cmake:1 |
| test-gemmini-srmd-raw-ws-cpu (tests/CMakeLists.txt:1230) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GEMMINI_BUILD_SRMD_RAW_WS_TESTS) at tests/CMakeLists.txt:1203 | target definition tests/CMakeLists.txt:1218; tests/test-gemmini-srmd-raw-ws.cpp:1 |
| test-gemmini-matmul-options (tests/CMakeLists.txt:1262) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(_GGML_GEMMINI_ACTIVATION_BITS_VALUE EQUAL 8 AND _GGML_GEMMINI_WEIGHT_BITS_VALUE EQUAL 8) at tests/CMakeLists.txt:1234 | target definition tests/CMakeLists.txt:1236; tests/test-gemmini-matmul-options.cpp:1 |
| test-gemmini-geometry-contract (tests/CMakeLists.txt:1264) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-geometry-contract.cmake:1 |
| test-gemmini-geometry-contract-adversarial (tests/CMakeLists.txt:1268) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480 | tests/test-gemmini-geometry-contract.cmake:1; tests/test-gemmini-geometry-contract-adversarial.cmake:1 |
| test-gemmini-build-contract (tests/CMakeLists.txt:1279) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(EXISTS "${CMAKE_COMMAND}" AND EXISTS "${LLAMA_TEST_SOURCE_DIR}/CMakeLists.txt") at tests/CMakeLists.txt:1275 | tests/test-gemmini-build-contract.cmake:1 |
| test-gemmini-hp1-contract (tests/CMakeLists.txt:1294) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(EXISTS "${CMAKE_COMMAND}" AND EXISTS "${LLAMA_TEST_SOURCE_DIR}/CMakeLists.txt") at tests/CMakeLists.txt:1275; if(Python3_Interpreter_FOUND) at tests/CMakeLists.txt:1293 | tests/test-gemmini-hp1-contract.cmake:1 |
| test-gemmini-im2p-routing (tests/CMakeLists.txt:1327) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304 | target definition tests/CMakeLists.txt:1305; tests/test-gemmini-im2p-routing.cpp:1 |
| test-gemmini-im2p-route-matrix (tests/CMakeLists.txt:1329) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304 | target definition tests/CMakeLists.txt:1305; tests/test-gemmini-im2p-routing.cpp:1 |
| test-gemmini-im2p-routing-multiwidth-matrix (tests/CMakeLists.txt:1332) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "EXSIA" AND GGML_GEMMINI_ENABLE_RMD) at tests/CMakeLists.txt:1331 | target definition tests/CMakeLists.txt:1305; tests/test-gemmini-im2p-routing.cpp:1 |
| test-gemmini-im2p-routing-boundary-rejections (tests/CMakeLists.txt:1334) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(_GGML_GEMMINI_ACTIVATION_QUANT_UPPER STREQUAL "EXSIA" AND GGML_GEMMINI_ENABLE_RMD) at tests/CMakeLists.txt:1331 | target definition tests/CMakeLists.txt:1305; tests/test-gemmini-im2p-routing.cpp:1 |
| test-gemmini-im2p-routing-geometry (tests/CMakeLists.txt:1337) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304 | target definition tests/CMakeLists.txt:1305; tests/test-gemmini-im2p-routing.cpp:1 |
| test-gemmini-im2p-routing-stats-translation (tests/CMakeLists.txt:1339) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304 | target definition tests/CMakeLists.txt:1305; tests/test-gemmini-im2p-routing.cpp:1 |
| test-gemmini-im2p-cycle-overlap (tests/CMakeLists.txt:1344) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(LOG_CYCLE AND GGML_GEMMINI_ACTIVATION_QUANT STREQUAL "EXSIA" AND GGML_GEMMINI_ENABLE_RMD) at tests/CMakeLists.txt:1341 | tests/test-gemmini-im2p-cycle-overlap.cmake:1; target definition tests/CMakeLists.txt:1305; tests/test-gemmini-im2p-routing.cpp:1 |
| test-gemmini-im2p-lifecycle (tests/CMakeLists.txt:1377) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304 | target definition tests/CMakeLists.txt:1352; tests/test-gemmini-im2p-lifecycle.cpp:1 |
| test-gemmini-im2p-lifecycle-stage-failures (tests/CMakeLists.txt:1379) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304 | target definition tests/CMakeLists.txt:1352; tests/test-gemmini-im2p-lifecycle.cpp:1 |
| test-gemmini-im2p-lifecycle-invalid-reuse (tests/CMakeLists.txt:1382) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304 | target definition tests/CMakeLists.txt:1352; tests/test-gemmini-im2p-lifecycle.cpp:1 |
| test-gemmini-im2p-lifecycle-stats-invariants (tests/CMakeLists.txt:1385) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304 | target definition tests/CMakeLists.txt:1352; tests/test-gemmini-im2p-lifecycle.cpp:1 |
| test-gemmini-rmd-im2p-provider-typed-metadata (tests/CMakeLists.txt:1415) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(LOG_CYCLE AND CYCLE_DETAIL AND NOT CYCLE_SIM AND IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1") at tests/CMakeLists.txt:1413 | target definition tests/CMakeLists.txt:1389; tests/test-gemmini-rmd-im2p-provider.cpp:1 |
| test-gemmini-rmd-im2p-provider-typed-metadata-invalid (tests/CMakeLists.txt:1419) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(LOG_CYCLE AND CYCLE_DETAIL AND NOT CYCLE_SIM AND IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1") at tests/CMakeLists.txt:1413 | target definition tests/CMakeLists.txt:1389; tests/test-gemmini-rmd-im2p-provider.cpp:1 |
| test-gemmini-rmd-im2p-provider-dispatch-parity (tests/CMakeLists.txt:1424) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304 | target definition tests/CMakeLists.txt:1389; tests/test-gemmini-rmd-im2p-provider.cpp:1 |
| test-gemmini-rmd-im2p-provider-dispatch-parity-callback-failure (tests/CMakeLists.txt:1429) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1" AND NOT CYCLE_SIM) at tests/CMakeLists.txt:1428 | target definition tests/CMakeLists.txt:1389; tests/test-gemmini-rmd-im2p-provider.cpp:1 |
| test-gemmini-rmd-im2p-provider-production-fixtures (tests/CMakeLists.txt:1432) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1" AND NOT CYCLE_SIM) at tests/CMakeLists.txt:1428 | target definition tests/CMakeLists.txt:1389; tests/test-gemmini-rmd-im2p-provider.cpp:1 |
| test-gemmini-rmd-im2p-provider-production-oracle-mutations (tests/CMakeLists.txt:1436) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1" AND NOT CYCLE_SIM) at tests/CMakeLists.txt:1428 | target definition tests/CMakeLists.txt:1389; tests/test-gemmini-rmd-im2p-provider.cpp:1 |
| test-gemmini-rmd-im2p-provider-run-aware-functional (tests/CMakeLists.txt:1442) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(CYCLE_SIM) at tests/CMakeLists.txt:1441 | target definition tests/CMakeLists.txt:1389; tests/test-gemmini-rmd-im2p-provider.cpp:1 |
| test-gemmini-rmd-im2p-provider-sat32-functional (tests/CMakeLists.txt:1444) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(CYCLE_SIM) at tests/CMakeLists.txt:1441 | target definition tests/CMakeLists.txt:1389; tests/test-gemmini-rmd-im2p-provider.cpp:1 |
| test-gemmini-rmd-im2p-provider (tests/CMakeLists.txt:1447) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(ELSE of (CYCLE_SIM)) at tests/CMakeLists.txt:1441 | target definition tests/CMakeLists.txt:1389; tests/test-gemmini-rmd-im2p-provider.cpp:1 |
| test-gemmini-rmd-im2p-provider-failures (tests/CMakeLists.txt:1449) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(ELSE of (CYCLE_SIM)) at tests/CMakeLists.txt:1441 | target definition tests/CMakeLists.txt:1389; tests/test-gemmini-rmd-im2p-provider.cpp:1 |
| test-gemmini-rmd-im2p-provider-hp1-success (tests/CMakeLists.txt:1453) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(ELSE of (CYCLE_SIM)) at tests/CMakeLists.txt:1441; if(IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1" AND GGML_GEMMINI_WEIGHT_BITS LESS 16) at tests/CMakeLists.txt:1451 | target definition tests/CMakeLists.txt:1389; tests/test-gemmini-rmd-im2p-provider.cpp:1 |
| test-gemmini-rmd-im2p-provider-dense-scu-stripe (tests/CMakeLists.txt:1460) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if((CYCLE_SIM OR IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1") AND GGML_GEMMINI_WEIGHT_BITS LESS 16) at tests/CMakeLists.txt:1458 | target definition tests/CMakeLists.txt:1389; tests/test-gemmini-rmd-im2p-provider.cpp:1 |
| test-gemmini-optrace-dispatch-full (tests/CMakeLists.txt:1475) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1" AND GGML_GEMMINI_ENABLE_RMD AND NOT CYCLE_SIM) at tests/CMakeLists.txt:1464 | target definition tests/CMakeLists.txt:1465; tests/test-gemmini-optrace-dispatch.cpp:1 |
| test-gemmini-optrace-dispatch-pipeline (tests/CMakeLists.txt:1475) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1" AND GGML_GEMMINI_ENABLE_RMD AND NOT CYCLE_SIM) at tests/CMakeLists.txt:1464 | target definition tests/CMakeLists.txt:1465; tests/test-gemmini-optrace-dispatch.cpp:1 |
| test-gemmini-optrace-dispatch-residual (tests/CMakeLists.txt:1475) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1" AND GGML_GEMMINI_ENABLE_RMD AND NOT CYCLE_SIM) at tests/CMakeLists.txt:1464 | target definition tests/CMakeLists.txt:1465; tests/test-gemmini-optrace-dispatch.cpp:1 |
| test-gemmini-optrace-transaction (tests/CMakeLists.txt:1478) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304; if(IM2P_SIM_IMPLEMENTATION STREQUAL "GEMMINI_HP1" AND GGML_GEMMINI_ENABLE_RMD AND NOT CYCLE_SIM) at tests/CMakeLists.txt:1464 | tests/test-gemmini-optrace-transaction.sh:1; target definition tests/CMakeLists.txt:1465; tests/test-gemmini-optrace-dispatch.cpp:1 |
| test-gemmini-rmd-backend-denylist (tests/CMakeLists.txt:1495) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(GGML_GEMMINI_EXECUTION_BACKEND STREQUAL "IM2P_SIM" AND NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1304 | tests/test-gemmini-rmd-backend-denylist.cmake:1; target definition tests/CMakeLists.txt:1483; tests/test-gemmini-rmd-backend-gate.cpp:1 |
| test-gemmini-matmul (tests/CMakeLists.txt:1517) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1502 | target definition tests/CMakeLists.txt:1505; tests/test-gemmini-matmul.cpp:1 |
| test-gemmini-attention (tests/CMakeLists.txt:1519) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1502; if(GGML_GEMMINI_ENABLE_A4NKS_ATTENTION) at tests/CMakeLists.txt:1518 | target definition tests/CMakeLists.txt:1505; tests/test-gemmini-matmul.cpp:1 |
| test-gemmini-attention-mask (tests/CMakeLists.txt:1520) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1502; if(GGML_GEMMINI_ENABLE_A4NKS_ATTENTION) at tests/CMakeLists.txt:1518 | target definition tests/CMakeLists.txt:1505; tests/test-gemmini-matmul.cpp:1 |
| test-gemmini-matmul-finite-output (tests/CMakeLists.txt:1522) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1502 | target definition tests/CMakeLists.txt:1505; tests/test-gemmini-matmul.cpp:1 |
| test-gemmini-buffer-artifacts (tests/CMakeLists.txt:1529) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:480; if(NOT GGML_BACKEND_DL) at tests/CMakeLists.txt:1502 | target definition tests/CMakeLists.txt:1526; tests/test-gemmini-buffer-artifacts.cpp:1 |
| gemmini_quant_dispatch_fp (tests/CMakeLists.txt:1566) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:1545 | target definition tests/CMakeLists.txt:1546; tests/gemmini-quant-dispatch-smoke.cpp:1 |
| gemmini_quant_dispatch_exsia (tests/CMakeLists.txt:1567) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:1545 | target definition tests/CMakeLists.txt:1546; tests/gemmini-quant-dispatch-smoke.cpp:1 |
| gemmini_quant_dispatch_tensor (tests/CMakeLists.txt:1568) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:1545 | target definition tests/CMakeLists.txt:1546; tests/gemmini-quant-dispatch-smoke.cpp:1 |
| gemmini_quant_dispatch_token (tests/CMakeLists.txt:1569) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:1545 | target definition tests/CMakeLists.txt:1546; tests/gemmini-quant-dispatch-smoke.cpp:1 |
| gemmini_quant_dispatch_unsupported (tests/CMakeLists.txt:1570) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:1545 | tests/gemmini-quant-dispatch-unsupported.cmake:1; target definition tests/CMakeLists.txt:1546; tests/gemmini-quant-dispatch-smoke.cpp:1 |
| test-gemmini-matmul-cpu-boundaries (tests/CMakeLists.txt:1577) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:1545 | tests/test-gemmini-matmul-cpu-boundaries.cmake:1 |
| test-gemmini-profiled-domain-contract (tests/CMakeLists.txt:1589) | if(TARGET ggml-gemmini) at tests/CMakeLists.txt:1545 | tests/test-gemmini-profiled-domain-contract.cmake:1 |
| test-gemmini-rmd-executor (tests/CMakeLists.txt:1624) | if(TARGET ggml-gemmini AND NOT CYCLE_SIM AND EXISTS "${IM2P_SIM_ROOT}/sim/include/im2p_sim.h") at tests/CMakeLists.txt:1607 | target definition tests/CMakeLists.txt:1609; tests/test-gemmini-rmd-executor.cpp:1; ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp:1 |
| potal-gguf-load-test (tests/CMakeLists.txt:1733) | if(GGML_POTAL) at tests/CMakeLists.txt:1729; if(NOT CMAKE_CROSSCOMPILING) at tests/CMakeLists.txt:1732 | target definition tests/CMakeLists.txt:1730; tests/test-potal-gguf-load.cpp:1 |
| test-cuda-cpu-exact-graph (tests/CMakeLists.txt:1754) | if(GGML_GEMMINI_CUDA_CPU_EXACT) at tests/CMakeLists.txt:1740 | target definition tests/CMakeLists.txt:1742; tests/test-cuda-cpu-exact-graph.cpp:1 |
| test-cuda-hp1-scu (tests/CMakeLists.txt:1754) | if(GGML_GEMMINI_CUDA_CPU_EXACT) at tests/CMakeLists.txt:1740 | target definition tests/CMakeLists.txt:1742; tests/test-cuda-hp1-scu.cpp:1 |
| test-cuda-potal-attention (tests/CMakeLists.txt:1754) | if(GGML_GEMMINI_CUDA_CPU_EXACT) at tests/CMakeLists.txt:1740 | target definition tests/CMakeLists.txt:1742; tests/test-cuda-potal-attention.cpp:1 |
| test-potal-sensitivity (tests/CMakeLists.txt:1754) | if(GGML_GEMMINI_CUDA_CPU_EXACT) at tests/CMakeLists.txt:1740 | target definition tests/CMakeLists.txt:1742; tests/test-potal-sensitivity.cpp:1 |
| test-potal-sensitivity-invalid (tests/CMakeLists.txt:1756) | if(GGML_GEMMINI_CUDA_CPU_EXACT) at tests/CMakeLists.txt:1740 | tests/test-potal-sensitivity.cpp:1; tests/CMakeLists.txt:1741-1757 |
| test-cuda-hp1-scu-graph (tests/CMakeLists.txt:1757) | if(GGML_GEMMINI_CUDA_CPU_EXACT) at tests/CMakeLists.txt:1740 | tests/test-cuda-cpu-exact-graph.cpp:1; tests/CMakeLists.txt:1741-1757 |
| test-cuda-rtn (tests/CMakeLists.txt:1766) | if(GGML_GEMMINI_CUDA_RTN OR GGML_GEMMINI_CUDA_RTN_ATTENTION) at tests/CMakeLists.txt:1760 | target definition tests/CMakeLists.txt:1761; tests/test-cuda-rtn.cpp:1 |

### INT4 disabled registrations and environment-specific cases

The INT4 fork deliberately DISABLES the W8-only registered tests rather than deleting their targets. These tests cannot be counted as passing W4 coverage; the W4 golden/counters/packing/dependency tests provide separate checks (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:3-32 and ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:35-63). The exact disabled-name list follows; applies only to registered tests when both configured widths are 4.

- potal-api-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-stripe-api-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-attention-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-ggml-unit-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-log-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-tiling-unit-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-residual-unit-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-stripe-pipeline-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-hw-timeline-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-paired-plan-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-paired-unit-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-paired-auto-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-paired-execution-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-e4b1-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-e4b2-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-e4c1-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-resident-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).
- potal-resident-model-test (ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:7-25).

QK residual/transfer tests set POTAL_QK_POLICY; weight preload tests force hart roles and strict issuer OFF (ggml/src/ggml-potal/CMakeLists.txt:389-403; ggml/src/ggml-potal-w4a4/tests/CMakeLists.txt:44-51; ggml/src/ggml-potal-w4a4/CMakeLists.txt:436-439). Root ExSIA named cases have additional A8W8 cases and force OpenMP test settings; these are matrix specialization, not universal width coverage (tests/CMakeLists.txt:855-873). Root registered Gemmini executables inherit CYCLE_SIM and an aggregate build dependency; inspect the per-test definitions for test-local numeric overrides (tests/CMakeLists.txt:1679-1687).

### Backend executables that are not independently registered tests

These target declarations are not additional ctest registrations. Hardware smoke/bench targets generally run explicitly on the target; E3 is an opt-in included CMake fragment and contains executable construction without an add_test registration. Their presence is evidence to preserve test/probe access, not a hardware PASS (ggml/src/ggml-potal/CMakeLists.txt:418-421 and ggml/src/ggml-potal/CMakeLists.txt:685-710; ggml/src/ggml-potal/tests/e3/CMakeLists.txt:1-50).

| Target | Definition/source |
|---|---|
| potal-scale-path-bench | ggml/src/ggml-potal/CMakeLists.txt:86; tests/potal-scale-path-bench.c tests/potal-scale-clock.cpp |
| potal-e4b2-rocc | ggml/src/ggml-potal/CMakeLists.txt:122; tests/potal-e4b2-rocc.c |
| potal-api-smoke | ggml/src/ggml-potal/CMakeLists.txt:141; tests/potal-api-smoke.c |
| potal-multicore-smoke | ggml/src/ggml-potal/CMakeLists.txt:148; tests/potal-multicore-smoke.c |
| potal-clock-probe | ggml/src/ggml-potal/CMakeLists.txt:169; tests/potal-clock-probe.c |
| potal-e4b2-hardware | ggml/src/ggml-potal/CMakeLists.txt:407; tests/potal-e4b2-hardware.cpp ${POTAL_BACKEND_SOURCES} |
| potal-paired-inventory | ggml/src/ggml-potal/CMakeLists.txt:413; tests/potal-paired-inventory.cpp ${POTAL_BACKEND_SOURCES} |
| potal-attention-bench | ggml/src/ggml-potal/CMakeLists.txt:428; tests/potal-attention-bench.cpp ${_b5_sources} |
| potal-e4c1-bench | ggml/src/ggml-potal/CMakeLists.txt:493; tests/potal-e4c1-bench.cpp tests/potal-e4c1-backend.cpp tests/support/e4c1-sequential.cpp ggml-potal.cpp ${_e4c1_sources} |
| potal-full-inference | ggml/src/ggml-potal/CMakeLists.txt:503; tests/potal-full-inference.cpp |
| potal-e2e-runner | ggml/src/ggml-potal/CMakeLists.txt:510; tests/potal-e2e-runner.cpp |
| potal-attention-sweep | ggml/src/ggml-potal/CMakeLists.txt:516; tests/potal-attention-sweep.cpp |
| potal-weight-prep-bench | ggml/src/ggml-potal/CMakeLists.txt:670; tests/potal-weight-prep-bench.cpp ${POTAL_BACKEND_SOURCES} |
| potal-residual-merge-bench | ggml/src/ggml-potal/CMakeLists.txt:679; tests/potal-residual-merge-bench.cpp ${POTAL_BACKEND_SOURCES} |
| potal-attention-smoke | ggml/src/ggml-potal/CMakeLists.txt:687; tests/potal-attention-smoke.cpp ${POTAL_BACKEND_SOURCES} |
| potal-ggml-smoke | ggml/src/ggml-potal/CMakeLists.txt:694; tests/potal-ggml-smoke.cpp |
| potal-hp1-smoke | ggml/src/ggml-potal/CMakeLists.txt:701; tests/potal-hp1-smoke.cpp ${POTAL_BACKEND_SOURCES} |
| potal-stripe-smoke | ggml/src/ggml-potal/CMakeLists.txt:708; tests/potal-stripe-smoke.cpp ${POTAL_BACKEND_SOURCES} |
| potal-pipeline-bench | ggml/src/ggml-potal/CMakeLists.txt:715; tests/potal-pipeline-bench.cpp ${POTAL_BACKEND_SOURCES} |
| potal-paired-smoke | ggml/src/ggml-potal/CMakeLists.txt:722; tests/potal-paired-smoke.cpp ${POTAL_BACKEND_SOURCES} |
| potal-e3-context-test | ggml/src/ggml-potal/tests/e3/CMakeLists.txt:13; ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-context.c ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-context-test.cpp |
| potal-e3-prototype-test | ggml/src/ggml-potal/tests/e3/CMakeLists.txt:29; ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-prototype-test.cpp ${POTAL_BACKEND_SOURCES} |
| potal-e3-kbench | ggml/src/ggml-potal/tests/e3/CMakeLists.txt:34; ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-kbench.cpp ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-backend.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ggml-potal.cpp ${_e3_absolute_sources} |
| potal-e3-window-diag | ggml/src/ggml-potal/tests/e3/CMakeLists.txt:42; ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-window-diag.cpp ${POTAL_BACKEND_SOURCES} |
| potal-scale-path-bench | ggml/src/ggml-potal-w4a4/CMakeLists.txt:115; tests/potal-scale-path-bench.c tests/potal-scale-clock.cpp |
| potal-e4b2-rocc | ggml/src/ggml-potal-w4a4/CMakeLists.txt:151; tests/potal-e4b2-rocc.c |
| potal-api-smoke | ggml/src/ggml-potal-w4a4/CMakeLists.txt:170; tests/potal-api-smoke.c |
| potal-multicore-smoke | ggml/src/ggml-potal-w4a4/CMakeLists.txt:177; tests/potal-multicore-smoke.c |
| potal-clock-probe | ggml/src/ggml-potal-w4a4/CMakeLists.txt:198; tests/potal-clock-probe.c |
| potal-e4b2-hardware | ggml/src/ggml-potal-w4a4/CMakeLists.txt:455; tests/potal-e4b2-hardware.cpp ${POTAL_BACKEND_SOURCES} |
| potal-paired-inventory | ggml/src/ggml-potal-w4a4/CMakeLists.txt:461; tests/potal-paired-inventory.cpp ${POTAL_BACKEND_SOURCES} |
| potal-attention-bench | ggml/src/ggml-potal-w4a4/CMakeLists.txt:476; tests/potal-attention-bench.cpp ${_b5_sources} |
| potal-e4c1-bench | ggml/src/ggml-potal-w4a4/CMakeLists.txt:541; tests/potal-e4c1-bench.cpp tests/potal-e4c1-backend.cpp tests/support/e4c1-sequential.cpp ggml-potal.cpp ${_e4c1_sources} |
| potal-full-inference | ggml/src/ggml-potal-w4a4/CMakeLists.txt:551; tests/potal-full-inference.cpp |
| potal-e2e-runner | ggml/src/ggml-potal-w4a4/CMakeLists.txt:558; tests/potal-e2e-runner.cpp |
| potal-attention-sweep | ggml/src/ggml-potal-w4a4/CMakeLists.txt:564; tests/potal-attention-sweep.cpp |
| potal-weight-prep-bench | ggml/src/ggml-potal-w4a4/CMakeLists.txt:718; tests/potal-weight-prep-bench.cpp ${POTAL_BACKEND_SOURCES} |
| potal-residual-merge-bench | ggml/src/ggml-potal-w4a4/CMakeLists.txt:727; tests/potal-residual-merge-bench.cpp ${POTAL_BACKEND_SOURCES} |
| potal-attention-smoke | ggml/src/ggml-potal-w4a4/CMakeLists.txt:735; tests/potal-attention-smoke.cpp ${POTAL_BACKEND_SOURCES} |
| potal-ggml-smoke | ggml/src/ggml-potal-w4a4/CMakeLists.txt:742; tests/potal-ggml-smoke.cpp |
| potal-hp1-smoke | ggml/src/ggml-potal-w4a4/CMakeLists.txt:749; tests/potal-hp1-smoke.cpp ${POTAL_BACKEND_SOURCES} |
| potal-stripe-smoke | ggml/src/ggml-potal-w4a4/CMakeLists.txt:756; tests/potal-stripe-smoke.cpp ${POTAL_BACKEND_SOURCES} |
| potal-pipeline-bench | ggml/src/ggml-potal-w4a4/CMakeLists.txt:763; tests/potal-pipeline-bench.cpp ${POTAL_BACKEND_SOURCES} |
| potal-paired-smoke | ggml/src/ggml-potal-w4a4/CMakeLists.txt:770; tests/potal-paired-smoke.cpp ${POTAL_BACKEND_SOURCES} |
| potal-e3-context-test | ggml/src/ggml-potal-w4a4/tests/e3/CMakeLists.txt:13; ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-context.c ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-context-test.cpp |
| potal-e3-prototype-test | ggml/src/ggml-potal-w4a4/tests/e3/CMakeLists.txt:29; ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-prototype-test.cpp ${POTAL_BACKEND_SOURCES} |
| potal-e3-kbench | ggml/src/ggml-potal-w4a4/tests/e3/CMakeLists.txt:34; ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-kbench.cpp ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-backend.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ggml-potal.cpp ${_e3_absolute_sources} |
| potal-e3-window-diag | ggml/src/ggml-potal-w4a4/tests/e3/CMakeLists.txt:42; ${CMAKE_CURRENT_SOURCE_DIR}/tests/potal-e3-window-diag.cpp ${POTAL_BACKEND_SOURCES} |
| ${pb_q8_test} | ggml/src/ggml-gemmini/CMakeLists.txt:181; tests/${pb_q8_test}.cpp ../ggml-cpu/ggml-cpu-quants.c |
| gemmini-pb-e2e-runner | ggml/src/ggml-gemmini/CMakeLists.txt:205; tests/gemmini-pb-e2e-runner.cpp |
| gemmini-fp32-e2e-runner | ggml/src/ggml-gemmini/fp32/backend.cmake:31; fp32/tests/runner.cpp |
| golden_replay_fp32 | ggml/src/ggml-gemmini/fp32/backend.cmake:34; fp32/tests/replay.cpp |
| gemmini-fp32-rtl-vector-test | ggml/src/ggml-gemmini/fp32/backend.cmake:39; fp32/tests/rtl-vectors.cpp |

## Test and build limits

All matrix outcomes remain UNVERIFIED in this document. This audit inspected declarations and scripts without configuring/building, running ctest, provisioning IM2P, invoking a target probe, or using build-baseline/. Source evidence distinguishes checked encoders, command emulators, CPU numerical goldens, actual RISC-V issue and independent hardware-observation tools; successful host tests alone cannot validate physical command completion (ggml/src/ggml-potal/potal.c:902-940; ggml/src/ggml-potal/CMakeLists.txt:601-603 and ggml/src/ggml-potal/CMakeLists.txt:685-710).

