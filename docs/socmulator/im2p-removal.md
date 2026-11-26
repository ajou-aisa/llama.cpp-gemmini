# IM2P removal

Date: 2026-11-25. Branch: `socmulator-dev`. Base: `64a072afa5a7a22dec74e94d599ffa4b848c9a20`. Existing uncommitted Phase 1 work preserved. No commit, reset, checkout or stash performed. The 17 removed tracked paths containing the obsolete name were removed from the index so the requested `git ls-files` audit is empty; other changes remain in the working tree.

## Scope and numeric contracts

The simulator backend/provider/frontend, simulator cycle capture/replay/certification, provisioner and build resolver are removed. Native EXSIA, HP1, block/tensor/token/stripe quantization, residual/RMD, reconstruction, dequantization, attention, PB and PoTAL remain. No simulator compatibility alias, renamed resolver, error-only CLI option, fake provider or timing fallback was added. Generic optrace and native profiling remain.

The retained block-quantized implementation in `legacy/gemmini_legacy.hpp` keeps its arithmetic; only obsolete names, simulator arguments/branches and thin SW/WS wrappers were removed. Golden command and CPU-reference fixtures were not changed. The enum body and uint8_t underlying type were compared before/after and are identical: q8_0_unpacked_to_h1=0, q8_h0=1, q8_h2=2, retired 3, q8_hp1=4, q8_hp2=5, q8_channel=6, q8_channel_dense_sidecar=7, q4_h0=8, retired 9, q4_hp1=10, q16_h0=11, q16_h1=12, q16_hp1=13. GGUF type IDs remain unchanged.

## Disputed files: final disposition

RESTORED means functional native/general code remains and is verified; DELETED means the feature/consumer belongs to the removed simulator route. No new native terminal-elision feature was implemented. Old inventory paths are renamed, not a lost metadata feature.

| File | Disposition | Preserved code or exact removed dependency |
| --- | --- | --- |
| `evaluation/tests/test_cycle_campaign.py` | RESTORED | All interval, cross-drain, pairing, SCU-conservation and rejection assertions retained; independent typed NumPy fixture replaces simulator ABI layout import. |
| `evaluation/tests/test_publication_gates.py` | DELETED | Exercises pinned simulator cost/memory inventory and offline_pipeline reconstruction admission. Independent artifact hash rejection retained in tests/test-cycle-evaluation.py alongside direct native target admission tests. |
| `ggml/src/ggml-gemmini-utils/include/gemmini/optrace.hpp` | RESTORED | Generic Session, Work, Context, ScopedContext and RunInfo API retained. |
| `ggml/src/ggml-gemmini-utils/src/optrace.cpp` | RESTORED | Writer, validation, phase/count reconciliation retained. CYCLE_SIM rejection and fixed simulator backend/three-source binding removed; native source/configuration fingerprints generated. |
| `ggml/src/ggml-gemmini/ggml-gemmini-args.h` | RESTORED | optrace_context retained; cycle_sim_context and simulator terminal-only state removed. |
| `ggml/src/ggml-gemmini/ggml-gemmini.cpp` | RESTORED | Generic optrace include retained; simulator backend setup/configuration removed. |
| `ggml/src/ggml-gemmini/ops.cpp` | RESTORED | Native matmul captures current optrace Context; simulator dispatch and terminal-elision guards removed. |
| `ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp` | RESTORED | Native execution scopes propagated optrace Context; simulator dot/run adapters and cycle records removed. |
| `scripts/eval/actual_cycle_bundle.py` | DELETED | Seals actual_trace_certificate.json plus cycle-results.json campaigns; admits im2p-actual-trace-certificate-v1 only. |
| `scripts/eval/campaign_cycle.py` | DELETED | Imports sim.cycle.execution_sequence_provider.StatefulSequenceProvider and stateful_sequence_evidence.EvidenceContext; native_build builds its cycle library. |
| `scripts/eval/certified_reconstruction.py` | RESTORED | Artifact references/hash checks and reconstruction/publication row guards retained. External official verifier, simulator schedule consumer/loader and consumer-source closure removed. |
| `scripts/eval/cycle_accounting.py` | RESTORED | EventAccounting, dependency pairing, interval unions and conservation checks retained. event_codes/event_dtype/for_library adapters to simulator C ABI removed; caller supplies codes/layout. |
| `scripts/eval/cycle_campaign.py` | DELETED | Loads sim.cycle.cycle_trace_certificate and emits im2p-evaluation-cycle-results-v1 from certified replay windows. |
| `scripts/eval/cycle_campaign_bundle.py` | DELETED | Packages cycle_campaign output and im2p-evaluation-cycle-campaign-v1 manifests; no independent native collection command. |
| `scripts/eval/cycle_trace_capture.py` | DELETED | Requires CYCLE_SIM + IM2P_SIM STRIPE_PIPELINE runner and invokes sim.cycle.cycle_trace_certificate build/verify. |
| `scripts/eval/e2e_cost_ownership.py` | DELETED | Pinned stage inventory names removed frontend, execute_rmd_stripe_im2p_run_aware and sim/cycle/control_engine.cpp; memory_scenario/validate_memory import sim.cycle.certificate_contract.TIMING. Generic interval definitions remain in cycle_accounting.py. |
| `scripts/eval/e2e_integration_prep.py` | DELETED | walk/Graph describe the external StatefulSequenceProvider admission stream; CLI imports sim.cycle.cycle_trace_certificate and requires its actual-trace certificate. |
| `scripts/eval/e2e_timeline.py` | RESTORED | Native thread lifetimes/lane packing/worker scenario, rational clock axis, interval union/overlap/statistics and CPU-core helpers retained. Simulator IR/service-table/schedule consumers and timing-authority labels removed. |
| `scripts/eval/offline_pipeline.py` | DELETED | Every reconstruction run invokes sim.cycle.npu_trace/evaluation_replay, reconstruct, execution_lifecycle_cli and execution_cli. Native collect remains in end_to_end.py/e2e_run.py. |
| `scripts/eval/plot_potal_pipeline.py` | RESTORED | Native PoTAL plotting retained. Only seven simulator operation-label branches and two simulator timing-label entries removed. |
| `scripts/eval/run_cycle_campaign.sh` | DELETED | Only execs the removed cycle_campaign.py. |
| `scripts/eval/run_cycle_evaluation.py` | DELETED | Builds simulator cycle library and runs sim.cycle.local_replay/nano_local_execution and schedule_sqlite. Generic PMU/timing preflight extracted intact to cpu_profiling.py; cache helpers stay in stage_cache.py. |
| `scripts/eval/run_cycle_trace_capture.sh` | DELETED | Only execs the removed cycle_trace_capture.py. |
| `scripts/eval/run_stateful_integration_smoke.py` | DELETED | Requires cycle_sim=1 IM2P_SIM runner; imports sim.cycle.collection_native and reconstruct_pairing. Native paired collection remains in e2e_run.py. |
| `scripts/eval/schedule_engine.py` | DELETED | run_schedule imports sim.cycle.scheduler_sqlite.schedule_sqlite; ScheduleSinks consumes that scheduler node/IR stream only. Host lane/interval arithmetic retained in e2e_timeline.py. |
| `scripts/eval/scheduled_endpoints.py` | DELETED | Accepts only im2p-execution-lifecycle and im2p-execution-schedule(-sqlite) plus VerifiedStatefulSchedule receipts from the external verifier. |
| `scripts/eval/target_admission.py` | RESTORED | Hash binding, workload identity, measured target-host and interface-cost validators retained. External evaluation_clock and simulator-owned readiness orchestration removed. |
| `scripts/optrace-build-info.py` | RESTORED | Simulator root/runtime/profile resolver imports removed. Real native source/configuration hashes generated; missing VCS produces empty commit, never a fabricated commit. |
| `tests/test-evaluation-v3-certified-e2e.py` | DELETED | Constructs im2p collection provenance, official replay joins and execution schedules; certified loading invokes sim.cycle.service_certificate. Native observation/endpoint and rational aggregation coverage remains in retained evaluation tests. |
| `tests/test-evaluation-v3-offline.py` | DELETED | Imports sim.tests.cycle.test_execution_cli.fixture_files; every test drives offline_pipeline sim.cycle replay/join/adapt/schedule, including its preflight/error cases. |
| `tests/test-evaluation-v3-stateful.py` | DELETED | Exercises external official verifier/service-certificate CLI and im2p-execution-schedule v2 admission; hash-bound artifact and publication-row helpers retained/tested independently. |
| `tests/test-gemmini-metric-terminal-elision.cpp` | DELETED | Uses im2p_adapter::test_reset/test_counters and the simulator ExsiaFullExecution metrics-only path. Every numerical route under test is that adapter; native backend dependency proof below. |
| `tests/test-gemmini-optrace.cpp` | RESTORED | Generic writer, validation, phase/count and failure assertions retained; native identities supplied. |
| `tests/test-gemmini-production-parity.cpp` | DELETED | Requires im2p_sim_implementation()==gemmini-hp1-integrated-v1, installs im2p_test_set_work_observer, then calls execute_rmd_stripe_im2p and im2p_adapter::run_full/run_stripe_pipeline. CPU calculations are oracles for these simulator calls. |
| `tests/test-gemmini-rmd-backend-gate.cpp` | DELETED | Every expectation calls removed im2p_adapter::gate_route on ExsiaRouteRequest. CPU-direct/compact-WS and hardware_os values are adapter admission cases, not calls to native executors. |
| `tests/test-gemmini-rmd-executor.cpp` | DELETED | Includes only rmd-im2p-executor.hpp; both success/fault and null-provider cases call execute_im2p_compact_dot using Im2pFullExecutor and im2p_matmul_desc_t/im2p_work_stats_extended_t. |
| `tools/eval/test-terminal-lm-head.cmake` | DELETED | Only tests --terminal-lm-head metrics-only/full parser/guard branches for the removed simulator terminal-elision feature. |
| `tools/main/main.cpp` | RESTORED | GEMMINI_OPTRACE_PATH setup, phase scopes, decode identity and finish/error handling retained; simulator cycle-trace controls removed. |
| `tools/model-inventory/CMakeLists.txt` | RESTORED | Independent GGUF metadata target retained as llama-model-inventory, with original ggml-base linkage. |
| `tools/model-inventory/main.cpp` | RESTORED | Read-only GGUF parser, extent checks and JSON escaping retained; executable help name changed only. |
| `tools/model-inventory/test_inventory.py` | RESTORED | All seven original test cases/assertions retained; obsolete temporary-directory prefix renamed. |

## Terminal metrics-only dependency proof

The old `tools/eval/eval.cpp` guard explicitly required `IM2P_SIM + EXSIA + FULL + WS`. Only `im2p_adapter::ExsiaFullExecution::finish` implemented elision: it observed captured packets through `apply_captured_rmd_full(..., observe_only=true)`, which called `rmd::detail::observe_rmd_stripe_im2p`, then recorded Session elision instead of calling the frontend GEMM/reconstruction/commit. The graph test used `im2p_adapter::test_reset/test_counters` and HP1-carrier MXINT weights. The native `execute_rmd_stripe_ws_output` rejects `plan.hp1_carriers`; host native RMD WS is also rejected by ops. Keeping that option would be a dead option or require a new backend/metric feature. Therefore its CLI, collector/session state, implementation and both terminal tests were removed together. Generic Session, metric collection and native full numerical forwards remain.

## HP1 contract deletion proof

At base revision, `tests/test-gemmini-hp1-contract.cmake:1` requires `TEST_IM2P_ROOT/scripts/gemmini_replay_contract.py`. The embedded Python imports `hardware_contract/compatible/contract_digest` from `scripts.gemmini_replay_contract` (line 16), and `BuildFailure/ProfileSelection/Scu/resolve_profile` from `scripts.gemmini_resolve_profile` (line 19). Every one of six profiles calls `hardware_contract(profile, root)` (line 28); its assertions inspect only the returned external contract facts. The 30 rejection cases alter that metadata and call `compatible` (line 51) or `resolve_profile` against `config/gemmini_host_memory_contracts/{profile}.json` and `config/gemmini_hp1_profiles.json` (lines 56, 62-63). The root passed to the entire program is `TEST_IM2P_ROOT` (line 70). No HP1 quantizer, dequantizer, CPU reference, GEMM, residual or reconstruction is executed. General HP1 algorithm tests remain; their pre-existing failures are reported below rather than removed.

## All removed paths and their dependencies

56 old paths are absent; three inventory paths are restored under a native name as described above.

| Old path | Reason |
| --- | --- |
| `cmake/ggml-gemmini-im2p.cmake` | Imported and verified simulator frontend/provider archives, generated simulator geometry, and selected simulator implementation. |
| `docs/FPGA_MAIN_PARITY.md` | Obsolete simulator-only FPGA-main parity procedure; explicitly requested deletion. |
| `evaluation/tests/test_publication_gates.py` | Exercises pinned simulator cost/memory inventory and offline_pipeline reconstruction admission. Independent artifact hash rejection retained in tests/test-cycle-evaluation.py alongside direct native target admission tests. |
| `ggml/src/ggml-gemmini-utils/include/gemmini/cycle_sim_context.h` | Removed cycle_sim Session/context/record/stage implementation; depends on simulator Work geometry and provider lifecycle, not native cycle_reader/performance profiling. |
| `ggml/src/ggml-gemmini-utils/include/gemmini/cycle_sim_log.hpp` | Removed cycle_sim Session/context/record/stage implementation; depends on simulator Work geometry and provider lifecycle, not native cycle_reader/performance profiling. |
| `ggml/src/ggml-gemmini-utils/src/cycle_sim_internal.hpp` | Removed cycle_sim Session/context/record/stage implementation; depends on simulator Work geometry and provider lifecycle, not native cycle_reader/performance profiling. |
| `ggml/src/ggml-gemmini-utils/src/cycle_sim_log.cpp` | Removed cycle_sim Session/context/record/stage implementation; depends on simulator Work geometry and provider lifecycle, not native cycle_reader/performance profiling. |
| `ggml/src/ggml-gemmini-utils/src/cycle_sim_records.cpp` | Removed cycle_sim Session/context/record/stage implementation; depends on simulator Work geometry and provider lifecycle, not native cycle_reader/performance profiling. |
| `ggml/src/ggml-gemmini-utils/src/cycle_sim_stages.cpp` | Removed cycle_sim Session/context/record/stage implementation; depends on simulator Work geometry and provider lifecycle, not native cycle_reader/performance profiling. |
| `ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp` | im2p_adapter run_full/run_stripe_pipeline and ExsiaFullExecution submit/finish call the removed frontend/provider ABI. |
| `ggml/src/ggml-gemmini/ggml-gemmini-im2p.hpp` | Public declarations for that simulator adapter, its lifecycle, gate_route and test hooks. |
| `ggml/src/ggml-gemmini/im2p/route.cpp` | im2p_adapter::gate_route validates BuildIdentity::im2p_sim_ws routes; CPU/WS enum cases are requests to this adapter. |
| `ggml/src/ggml-gemmini/im2p/route.hpp` | Simulator route identity and request/result types; no native MatMul contract is defined here. |
| `ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.cpp` | execute_im2p_compact_dot invokes im2p_execute_matmul_planned/runs provider ABI. |
| `ggml/src/ggml-gemmini/residual/rmd/rmd-im2p-executor.hpp` | Im2pCompactDot/Im2pFullExecutor and im2p descriptor/provider types. |
| `scripts/cycle-sim-build-info.py` | Generates CYCLE_SIM producer contracts from the external simulator runtime/profile manifests. |
| `scripts/eval/actual_cycle_bundle.py` | Seals actual_trace_certificate.json plus cycle-results.json campaigns; admits im2p-actual-trace-certificate-v1 only. |
| `scripts/eval/campaign_cycle.py` | Imports sim.cycle.execution_sequence_provider.StatefulSequenceProvider and stateful_sequence_evidence.EvidenceContext; native_build builds its cycle library. |
| `scripts/eval/cycle_campaign.py` | Loads sim.cycle.cycle_trace_certificate and emits im2p-evaluation-cycle-results-v1 from certified replay windows. |
| `scripts/eval/cycle_campaign_bundle.py` | Packages cycle_campaign output and im2p-evaluation-cycle-campaign-v1 manifests; no independent native collection command. |
| `scripts/eval/cycle_trace_capture.py` | Requires CYCLE_SIM + IM2P_SIM STRIPE_PIPELINE runner and invokes sim.cycle.cycle_trace_certificate build/verify. |
| `scripts/eval/e2e_cost_ownership.py` | Pinned stage inventory names removed frontend, execute_rmd_stripe_im2p_run_aware and sim/cycle/control_engine.cpp; memory_scenario/validate_memory import sim.cycle.certificate_contract.TIMING. Generic interval definitions remain in cycle_accounting.py. |
| `scripts/eval/e2e_integration_prep.py` | walk/Graph describe the external StatefulSequenceProvider admission stream; CLI imports sim.cycle.cycle_trace_certificate and requires its actual-trace certificate. |
| `scripts/eval/offline_pipeline.py` | Every reconstruction run invokes sim.cycle.npu_trace/evaluation_replay, reconstruct, execution_lifecycle_cli and execution_cli. Native collect remains in end_to_end.py/e2e_run.py. |
| `scripts/eval/run_cycle_campaign.sh` | Only execs the removed cycle_campaign.py. |
| `scripts/eval/run_cycle_evaluation.py` | Builds simulator cycle library and runs sim.cycle.local_replay/nano_local_execution and schedule_sqlite. Generic PMU/timing preflight extracted intact to cpu_profiling.py; cache helpers stay in stage_cache.py. |
| `scripts/eval/run_cycle_trace_capture.sh` | Only execs the removed cycle_trace_capture.py. |
| `scripts/eval/run_stateful_integration_smoke.py` | Requires cycle_sim=1 IM2P_SIM runner; imports sim.cycle.collection_native and reconstruct_pairing. Native paired collection remains in e2e_run.py. |
| `scripts/eval/schedule_engine.py` | run_schedule imports sim.cycle.scheduler_sqlite.schedule_sqlite; ScheduleSinks consumes that scheduler node/IR stream only. Host lane/interval arithmetic retained in e2e_timeline.py. |
| `scripts/eval/scheduled_endpoints.py` | Accepts only im2p-execution-lifecycle and im2p-execution-schedule(-sqlite) plus VerifiedStatefulSchedule receipts from the external verifier. |
| `scripts/im2p-build-options.py` | Simulator implementation/artifact resolver. Generic cache/environment/default precedence moved into cmake/npu-config.cmake, without a replacement resolver script. |
| `scripts/im2p-host-provision.sh` | Builds/provisions external simulator libraries and caches only. |
| `tests/check-cycle-sim-host-stages.py` | Checks HOST_STAGE BEGIN/END declarations emitted by cycle_sim_log against host_stage_id/POTAL_HOST canonical intervals; that declaration producer was removed. |
| `tests/test-cycle-sim-build-contract.cmake` | Checks CYCLE_SIM enablement, IM2P_CYCLE_SIM geometry and simulator build manifests/cache artifact selection. |
| `tests/test-cycle-sim-coverage.cpp` | Uses ggml::gemmini::cycle_sim Session and removed cycle_sim_log.hpp to check simulator trace coverage. |
| `tests/test-cycle-sim-dispatch.cmake` | Runs the simulator dispatch executable and checks its NPU HOST_STAGE and im2p.output_reconstruction trace/cycle correspondence. |
| `tests/test-evaluation-v3-certified-e2e.py` | Constructs im2p collection provenance, official replay joins and execution schedules; certified loading invokes sim.cycle.service_certificate. Native observation/endpoint and rational aggregation coverage remains in retained evaluation tests. |
| `tests/test-evaluation-v3-offline.py` | Imports sim.tests.cycle.test_execution_cli.fixture_files; every test drives offline_pipeline sim.cycle replay/join/adapt/schedule, including its preflight/error cases. |
| `tests/test-evaluation-v3-stateful.py` | Exercises external official verifier/service-certificate CLI and im2p-execution-schedule v2 admission; hash-bound artifact and publication-row helpers retained/tested independently. |
| `tests/test-gemmini-cycle-sim-dispatch.cpp` | Includes im2p_cpu_functional.hpp/im2p_cycle_sim.hpp; installs im2p_matmul_desc_t/production_geometry observers. |
| `tests/test-gemmini-cycle-sim-log.cpp` | Constructs cycle_sim Session Work with im2p_production_geometry_v1_t and checks removed trace schemas/lifecycle. |
| `tests/test-gemmini-hp1-contract.cmake` | All six profiles/30 rejection cases call external scripts.gemmini_replay_contract and scripts.gemmini_resolve_profile using TEST_IM2P_ROOT/config manifests; no quantizer, dequantizer or matmul runs. Detailed proof below. |
| `tests/test-gemmini-im2p-cycle-overlap.cmake` | Reads IM2P_EXECUTION_TELEMETRY/IM2P_STRIPE_TELEMETRY and requires im2p_sim backend identities; native overlap tests remain. |
| `tests/test-gemmini-im2p-lifecycle.cpp` | Calls im2p_adapter ExSIA frontend lifecycle/failure hooks and includes im2p_gemmini_frontend.hpp. |
| `tests/test-gemmini-im2p-routing.cpp` | Includes ggml-gemmini-im2p.cpp directly and exercises FrontendWorkerTiming plus simulator telemetry/worker labels. |
| `tests/test-gemmini-metric-terminal-elision.cpp` | Uses im2p_adapter::test_reset/test_counters and the simulator ExsiaFullExecution metrics-only path. Every numerical route under test is that adapter; native backend dependency proof below. |
| `tests/test-gemmini-optrace-dispatch.cpp` | Uses im2p_adapter, rmd-im2p-executor, im2p_gemmini_frontend and im2p_sim; generic test-gemmini-optrace.cpp restored. |
| `tests/test-gemmini-production-parity.cpp` | Requires im2p_sim_implementation()==gemmini-hp1-integrated-v1, installs im2p_test_set_work_observer, then calls execute_rmd_stripe_im2p and im2p_adapter::run_full/run_stripe_pipeline. CPU calculations are oracles for these simulator calls. |
| `tests/test-gemmini-rmd-backend-gate.cpp` | Every expectation calls removed im2p_adapter::gate_route on ExsiaRouteRequest. CPU-direct/compact-WS and hardware_os values are adapter admission cases, not calls to native executors. |
| `tests/test-gemmini-rmd-executor.cpp` | Includes only rmd-im2p-executor.hpp; both success/fault and null-provider cases call execute_im2p_compact_dot using Im2pFullExecutor and im2p_matmul_desc_t/im2p_work_stats_extended_t. |
| `tests/test-gemmini-rmd-im2p-provider.cpp` | Exercises Im2p provider operands, compact runs, injection and ABI execution via im2p_sim.h/im2p_cpu_functional.hpp. |
| `tests/test-im2p-cache-build-scripts.cmake` | Requires TEST_IM2P_ROOT/Makefile, sets IM2P_CACHE_JOBS/IM2P_SIM_ROOT and tests provisioner/resolver artifacts. |
| `tools/eval/test-terminal-lm-head.cmake` | Only tests --terminal-lm-head metrics-only/full parser/guard branches for the removed simulator terminal-elision feature. |
| `tools/im2p-model-inventory/CMakeLists.txt` | RENAMED to tools/model-inventory/CMakeLists.txt and target llama-model-inventory; independent GGUF metadata functionality preserved. |
| `tools/im2p-model-inventory/main.cpp` | RENAMED to tools/model-inventory/main.cpp; only executable help/name changed, gguf_init_from_file(no_alloc=true) metadata/extent/escaping behavior retained. |
| `tools/im2p-model-inventory/test_inventory.py` | RENAMED to tools/model-inventory/test_inventory.py; all seven test cases and numerical/metadata assertions retained; temporary-directory prefix loses obsolete name. |

## Modified tests and fixtures

This working-tree inventory includes preserved Phase 1 edits. Native numerical assertions remain. Simulator branches/provider counters and trace assertions were removed. Generic schema/weight-format names were updated. The real CMake harness checks defaults/environment/CLI precedence and active/disabled SDK rejection; wrapper checks inspect actual configure caches. Retained native plotting/accounting/host validation tests are described above.

```text
evaluation/tests/test_collection.py
evaluation/tests/test_cycle_campaign.py
evaluation/tests/test_framework.py
evaluation/tests/test_shards.py
tests/CMakeLists.txt
tests/gemmini-quant-dispatch-smoke.cpp
tests/potal-csv-fixture.jsonl
tests/test-build-arm64-log-defaults.cmake
tests/test-cycle-evaluation.py
tests/test-cycle-sim-reader.cpp
tests/test-cycle-timeline.py
tests/test-evaluation-build-options.cmake
tests/test-evaluation-lifecycle.cpp
tests/test-evaluation-trace.cpp
tests/test-evaluation-v3-cuda.py
tests/test-evaluation-v3.py
tests/test-functional-emulation-exclusion.cpp
tests/test-gemmini-act-quant-i32-saturation.cpp
tests/test-gemmini-block-dense.cpp
tests/test-gemmini-block-end-to-end.cpp
tests/test-gemmini-build-contract.cmake
tests/test-gemmini-cpu-cycle-validity-contract.cmake
tests/test-gemmini-evaluation-metrics.cpp
tests/test-gemmini-evaluation-observer.cpp
tests/test-gemmini-exsia.cpp
tests/test-gemmini-host-timing.cpp
tests/test-gemmini-matmul-cpu-boundaries.cmake
tests/test-gemmini-matmul-options.cpp
tests/test-gemmini-matmul.cpp
tests/test-gemmini-native-dequant-fp.cpp
tests/test-gemmini-off-build.cmake
tests/test-gemmini-optrace.cpp
tests/test-gemmini-production-host-backend.cpp
tests/test-gemmini-rmd-run-aware.cpp
tests/test-gemmini-rmd-telemetry.cpp
tests/test-gemmini-rmd.cpp
tests/test-gemmini-telemetry-clock-gate.cmake
tests/test-gemmini-trace-context.cpp
tests/test-gemmini-weight-formats.cpp
tests/test-measurement-domains.py
tests/test-metric-sweep.py
tests/test-potal-csv.py
tests/test-potal-target-profile.cmake
tools/eval/CMakeLists.txt
tools/eval/eval.cpp
tools/eval/evaluation-lifecycle.h
tools/eval/evaluation-trace.cpp
tools/eval/evaluation-trace.h
```


## Removed options, definitions and interfaces

- `CYCLE_SIM`, `IM2P_DIM`, `IM2P_SIM_ROOT`, `IM2P_SIM_IMPLEMENTATION` and the `IM2P_SIM*` build/provisioning controls.
- `NPU_TRANSPORT=IM2P_SIM`; supported transports are `ROCC` and `REFERENCE`.
- `GGML_GEMMINI_EXECUTION_BACKEND=IM2P_SIM`; the retained backend value is `HARDWARE`.
- `GGML_GEMMINI_EXECUTION_BACKEND_IM2P_SIM`, `IM2P_SIM_IMPLEMENTATION_LEGACY_BSV`, `IM2P_SIM_IMPLEMENTATION_GEMMINI_HP1` and simulator-specific source/library definitions.
- `LEGACY_BSV` and `GEMMINI_HP1` simulator implementation selection; simulator artifact/cache controls and `IM2P_THREADS`.
- `GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION`, its simulator executor and associated configuration definition. CPU attention quantization remains.
- Simulator-only `--cycle-trace`, `--terminal-lm-head`, metric terminal-elision state/hooks and offline performance/timeline campaign entry points.

## Renamed symbols and schemas

| Old | New |
| --- | --- |
| `tiled_matmul_im2p_impl` | `gemmini_block_quantized_matmul` |
| `tiled_matmul_auto_im2p` | `gemmini_block_quantized_matmul_auto` |
| `im2p_weight_format_t` | `weight_format_t` |
| `resolve_im2p_threads` | `resolve_block_matmul_threads` |
| `has_q8_channel_im2p_contract` | `has_q8_channel_contract` |
| `physical_im2p_impl` | `physical_block_quantized` |
| `physical_auto_im2p` | `physical_block_quantized_auto` |
| `im2p-semantic-graph` | `gemmini-semantic-graph` |
| `im2p-activation-quant-metrics` | `gemmini-activation-quant-metrics` |
| `im2p-residual-path-metrics` | `gemmini-residual-path-metrics` |
| `im2p-scale-alignment-metrics` | `gemmini-scale-alignment-metrics` |
| `im2p-scale-alignment-aggregate` | `gemmini-scale-alignment-aggregate` |

The thin `tiled_matmul_im2p_sw`/`tiled_matmul_im2p_ws` wrappers were deleted; generic auto dispatch calls the retained implementation directly.

## Preserved identifiers and ABI values

No IM2P identifier remains in current executable code. Preserved generic identifiers include `GEMMINI_SW_PATH` for the FP32 overlay, `LOG_CYCLE`, `CYCLE_DETAIL`, and the existing generic clock-test identities `test-cycle-sim-reader` / `test-cycle-sim-cpu-smoke.cmake`; these contain no simulator backend or `CYCLE_SIM` control. HP1/EXSIA and GGUF type identifiers remain.

`weight_format_t` retains: `q8_0_unpacked_to_h1=0`, `q8_h0=1`, `q8_h2=2`, retired 3, `q8_hp1=4`, `q8_hp2=5`, `q8_channel=6`, `q8_channel_dense_sidecar=7`, `q4_h0=8`, retired 9, `q4_hp1=10`, `q16_h0=11`, `q16_h1=12`, `q16_hp1=13`. GGUF type-number definitions were not changed.

## Documentation

- Deleted IM2P-only `docs/FPGA_MAIN_PARITY.md`.
- Updated `docs/POTAL_EVALUATION_V3.md` to internal REFERENCE build commands and supported native collection, preserving the PoTAL ACT/RES definitions, chunk/mask/sampling policy, replay distinctions and hardware-timing caveats. Removed simulator targets/options/resolver/`--im2p` usage.
- Updated `docs/socmulator/npu-build.md`, current `docs/build.md`, PoTAL README/stripe API documentation, `evaluation/README.md` and measurement script documentation/registries.
- Phase 0 audit/baseline documents and H3 history are untouched. `docs/backend/*.md` is unchanged; remaining audit hits there are historical.


## Generic tracing and native profiling

`GEMMINI_OPTRACE_PATH`, Session/Work validation, phase/context propagation, writer/count reconciliation, source-identity generation and `test-gemmini-optrace` remain. Removed simulator dispatch callbacks are not replaced by fabricated native work records. The build-info generator hashes actual source files and CMake configuration without simulator imports. `hardware_contract_sha256` and `runtime_manifest_sha256` now fingerprint the native NPU/configuration dictionaries; they are configuration provenance, not a physical hardware/bitstream certificate or timing proof. `source-identities.json` identifies this explicitly. When a Docker bind mount has no usable VCS metadata, commit is empty and real file hashes remain mandatory; no commit is invented.

`LOG_CYCLE`, `CYCLE_DETAIL`, native cycle_reader/performance helpers, CPU PMU preflight and host timing validation remain. `scripts/eval/cpu_profiling.py` contains the general profiling extracted from the removed simulator campaign. Native full-CPU/CUDA and metric collectors remain; combined metrics run full numerical forwards. The root `.pytest_cache` created during checks was untracked and removed; subsequent tests use `-p no:cacheprovider`.

## Build wrapper defaults and SDK boundary

Generic option precedence lives in `cmake/npu-config.cmake`: explicit cache/CLI > environment > *_DEFAULT environment > CMake default. *_DEFAULT keys are classified before ordinary options. Root build wrappers invoke CMake directly, pass CLI arguments last, and retain BUILD_DIR/JOBS. ARM default has Gemmini OFF with debug/cycle logs OFF; ARM CPU profiling wrapper keeps logs ON. x86 and RISC-V wrappers use their native PoTAL presets with REFERENCE or ROCC. RISC-V static default build-riscv-static remains; nonempty static_args avoids macOS Bash 3 set-u empty-array errors. These defaults and all three log flags are covered by real configure-cache tests.

`GEMMINI_SW_PATH` is rejected outside the explicit FP32 overlay, including disabled-backend configurations: the check precedes the NPU_NONE early return, preventing later generic geometry code from reading an external SDK. Active and disabled environment/default-path rejection tests exercise this boundary. Native matrix caches have no SDK root.

The Linux standalone cycle-reader executable now links `ggml-gemmini-utils`, which provides existing `timeline_now_ns` and `host_thread_id`. Its assertions/source are unchanged. The old Linux executable could not link and was NOT RUN; the linked test now executes.

PB attention exposed an existing declaration/definition mismatch: the PB definition accepted two arguments while the shared header/call site requires an optional third `AttentionCache *`. The definition now accepts the unused third argument; arithmetic is unchanged. The accepted PB build needs BLOCK activation, RMD OFF, ROCC, DIM32 and RVV OFF with this toolchain.

## Verification

Evidence is under `build-npu/phase2-evidence`; final Linux raw configure/build/CTest logs are under `build-npu/linux/logs`. Pre-edit Phase 1 logs and patch were copied to `build-npu/phase2-evidence/before` before any deletion. The canonical comparison logs are `/Users/zerogod/aisa-lab/DynDNN/reference/llama.cpp-baseline-64a072a/build-baseline/linux/logs`.

### macOS final matrix

| macOS configuration | Configure | Build | PASS | FAIL | NOT RUN | SKIP |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `gemmini-int8-d16-ref` | 0 | 0 | 110 | 9 | 0 | 0 |
| `gemmini-api` | 0 | 0 | 5 | 0 | 0 | 0 |
| `mac-cpu-only` | 0 | 0 | 56 | 4 | 0 | 0 |
| `gemmini-a4w4-int8-d16-ref` | 0 | 0 | 104 | 9 | 0 | 0 |

Every requested configuration configured and built successfully. Gemmini INT8 and A4-on-INT8 both PASS `gemmini-functional-test`, `gemmini-tiling-test`, `gemmini-encoding-test` and restored `test-gemmini-optrace`. API tests are `gemmini-api-test`, `gemmini-transport-test`, `gemmini-encoding-test`, `gemmini-tiling-test` and `npu-config-test`, all PASS. Full-suite CTest exit 8 reflects the enumerated existing failures below; it is not reported as a green suite.

### Independent native utilities

- `restoration-python-final.log`: 89 passed, 1 skipped, 8 subtests passed. The skip is the real Linux-aarch64 PMU probe on macOS. Tests disable pytest source-root caching.
- The same log records standalone `evaluation/tests/test_cycle_campaign.py` PASS, including interval unions, cross-drain pairing, SCU conservation, foreign-work/window/pairing rejection.
- `restoration-pyright.log`: explicit Python 3.11 check of restored utility modules, generator and native test: 0 errors, 0 warnings. The editor LSP used Python 3.9 for imported evaluation modules and rejected pre-existing TypeAlias/dataclass slots APIs; the explicit correct-version check is authoritative.
- `inventory-restored.build.log` and `inventory-restored.test.log`: native GGUF metadata target builds and all seven original tests PASS. `plot-restored.help.log`: retained plotting CLI imports and --help succeed (exit 0). No image-quality claim is made.
- Earlier `csv-final.log`: native PoTAL CSV CLI regression PASS with the built `llama-cycle-summary` executable. Changed shell wrappers passed `bash -n`.

### Linux and RISC-V matrix

| Linux configuration | Configure | Build | PASS | FAIL | NOT RUN | SKIP |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `cpu-only` | 0 | 0 | 55 | 6 | 0 | 0 |
| `gemmini-int8-d16-ref` | 0 | 0 | 107 | 13 | 0 | 0 |
| `potal-int8-d16` | 0 | 0 | 75 | 9 | 0 | 1 |
| `potal-int8-d32` | 0 | 0 | 71 | 11 | 0 | 1 |
| `potal-int4-d32` | 0 | 1 | 66 | 8 | 13 | 1 |

The full INT4 test build remains broken in unchanged `ggml/src/ggml-potal-w4a4/tests/potal-attention-test.cpp`: int32-vector arguments do not match int64 accumulator references (lines 229/232), old descriptor callbacks do not match paired_runner (line 583 onward), and removed `run_corrected` is still called (line 806 onward). The same source/API failures and missing test executables are present in both canonical and saved Phase 1 logs. No test was skipped, weakened or deleted to hide this. Its partial CTest output is not a clean regression suite. The actual `ggml-potal`, `llama-cli`, `llama-quantize` and restored metadata executable targets separately build with exit 0 (`potal-int4-core-restored.build.log`).

The renamed metadata executable also builds and passes all seven original tests on completed Linux CPU-only and Gemmini host configurations (`linux-native-tools.console.log`). This incremental native-tool check covers the tool restored while the matrix was already running; the full CTest suites were not repeated.

| RISC-V configuration | Configure | Build | Runtime |
| --- | --- | --- | --- |
| `rv-gemmini-api` | 0 | 0 | NOT RUN |
| `rv-gemmini-int8-d16-rocc` | 0 | 0 | NOT RUN |
| `rv-potal-int8-d16-rocc` | 0 | 0 | NOT RUN |

The one final Docker matrix run completed (`linux-restored.console.log`). Cross executables are not run: no physical RoCC target or compatible emulator is available.

PB final `rv-gemmini-pb-d32` configure/build both exit 0 (`pb-restored.console.log`, `pb-restored.build.log`). The REFERENCE host attempt is rejected because the native NPU configuration derives CPU while PB requires WS. ROCC requires riscv64, so the host model cannot be selected through this Phase 1 configuration contract. The successful cross build uses the gemmini INT8 RoCC preset, `NPU_D=32`, `GGML_GEMMINI_PB=ON`, `GGML_GEMMINI_ACTIVATION_QUANT=BLOCK`, `GGML_GEMMINI_ENABLE_RMD=OFF`, `GGML_RVV=OFF`, OpenMP/shared libraries OFF and tests ON. PB graph/runtime tests remain NOT RUN. The host `gemmini-functional-test` validates its independent LoopMatmul functional/command model; it is not claimed as a physical PB graph execution.

### Name-by-name comparison

No common test changed from PASS in saved Phase 1 to FAIL or NOT RUN in the final host matrices. Generic `test-gemmini-optrace` remains registered and passes. The only removed host registrations are `test-im2p-cache-build-scripts` and `test-evaluation-terminal-lm-head`; Gemmini configurations also lose `test-cycle-sim-build-contract` and `test-gemmini-hp1-contract`, whose external metadata-only dependency is proved above.

The table below lists every name whose status or presence differs from saved Phase 1 or the comparable canonical configuration. Omitted common names kept their previous status. macOS API has no saved pre-edit per-name log; A4 was newly configured for this task, so its comparison is explicitly the earlier Phase 2 run. Linux canonical Gemmini comparison uses `gemmini-int8-d16-cpu.ctest.log`, matching the host CPU/REFERENCE route, not the RoCC WS lane. CPU-only has no comparable canonical log; its saved Phase 1 log is used. UNSAVED means no matching baseline evidence, never PASS by assumption.

| Platform/configuration | Test name | Comparison source | Before | Final |
| --- | --- | --- | --- | --- |
| macOS gemmini-int8-d16-ref | `test-build-arm64-log-defaults` | saved Phase 1 LastTest | FAIL | PASS |
| macOS gemmini-int8-d16-ref | `test-cycle-sim-build-contract` | saved Phase 1 LastTest | PASS | REMOVED |
| macOS gemmini-int8-d16-ref | `test-evaluation-terminal-lm-head` | saved Phase 1 LastTest | PASS | REMOVED |
| macOS gemmini-int8-d16-ref | `test-gemmini-hp1-contract` | saved Phase 1 LastTest | FAIL | REMOVED |
| macOS gemmini-int8-d16-ref | `test-gemmini-off-build` | saved Phase 1 LastTest | FAIL | PASS |
| macOS gemmini-int8-d16-ref | `test-im2p-cache-build-scripts` | saved Phase 1 LastTest | FAIL | REMOVED |
| macOS gemmini-int8-d16-ref | `test-potal-target-profile` | saved Phase 1 LastTest | FAIL | PASS |
| macOS mac-cpu-only | `test-autorelease` | saved Phase 1 LastTest | ABSENT | PASS |
| macOS mac-cpu-only | `test-build-arm64-log-defaults` | saved Phase 1 LastTest | FAIL | PASS |
| macOS mac-cpu-only | `test-evaluation-terminal-lm-head` | saved Phase 1 LastTest | PASS | REMOVED |
| macOS mac-cpu-only | `test-gemmini-off-build` | saved Phase 1 LastTest | FAIL | PASS |
| macOS mac-cpu-only | `test-im2p-cache-build-scripts` | saved Phase 1 LastTest | FAIL | REMOVED |
| macOS mac-cpu-only | `test-model-load-cancel` | saved Phase 1 LastTest | ABSENT | PASS |
| macOS mac-cpu-only | `test-potal-target-profile` | saved Phase 1 LastTest | FAIL | PASS |
| macOS gemmini-a4w4-int8-d16-ref | `test-gemmini-optrace` | earlier Phase 2 run | ABSENT | PASS |
| macOS gemmini-a4w4-int8-d16-ref | `test-potal-target-profile` | earlier Phase 2 run | FAIL | PASS |
| Linux cpu-only | `test-build-arm64-log-defaults` | Phase 1 | FAIL | PASS |
| Linux cpu-only | `test-evaluation-build-options` | Phase 1 | FAIL | PASS |
| Linux cpu-only | `test-evaluation-terminal-lm-head` | Phase 1 | PASS | REMOVED |
| Linux cpu-only | `test-gemmini-cycle-reader-aarch64` | Phase 1 | NOT RUN | PASS |
| Linux cpu-only | `test-im2p-cache-build-scripts` | Phase 1 | FAIL | REMOVED |
| Linux cpu-only | `test-potal-target-profile` | Phase 1 | FAIL | PASS |
| Linux gemmini-int8-d16-ref | `test-build-arm64-log-defaults` | Phase 1 | FAIL | PASS |
| Linux gemmini-int8-d16-ref | `test-cycle-sim-build-contract` | Phase 1 | PASS | REMOVED |
| Linux gemmini-int8-d16-ref | `test-evaluation-build-options` | Phase 1 | FAIL | PASS |
| Linux gemmini-int8-d16-ref | `test-evaluation-terminal-lm-head` | Phase 1 | PASS | REMOVED |
| Linux gemmini-int8-d16-ref | `test-gemmini-cycle-reader-aarch64` | Phase 1 | NOT RUN | PASS |
| Linux gemmini-int8-d16-ref | `test-gemmini-hp1-contract` | Phase 1 | FAIL | REMOVED |
| Linux gemmini-int8-d16-ref | `test-im2p-cache-build-scripts` | Phase 1 | FAIL | REMOVED |
| Linux gemmini-int8-d16-ref | `test-potal-target-profile` | Phase 1 | FAIL | PASS |
| Linux gemmini-int8-d16-ref | `gemmini-api-test` | canonical | ABSENT | PASS |
| Linux gemmini-int8-d16-ref | `gemmini-encoding-test` | canonical | ABSENT | PASS |
| Linux gemmini-int8-d16-ref | `gemmini-functional-test` | canonical | ABSENT | PASS |
| Linux gemmini-int8-d16-ref | `gemmini-tiling-test` | canonical | ABSENT | PASS |
| Linux gemmini-int8-d16-ref | `gemmini-transport-test` | canonical | ABSENT | PASS |
| Linux gemmini-int8-d16-ref | `npu-config-test` | canonical | ABSENT | PASS |
| Linux gemmini-int8-d16-ref | `test-build-arm64-log-defaults` | canonical | FAIL | PASS |
| Linux gemmini-int8-d16-ref | `test-cycle-sim-build-contract` | canonical | PASS | REMOVED |
| Linux gemmini-int8-d16-ref | `test-evaluation-build-options` | canonical | FAIL | PASS |
| Linux gemmini-int8-d16-ref | `test-evaluation-terminal-lm-head` | canonical | PASS | REMOVED |
| Linux gemmini-int8-d16-ref | `test-gemmini-cycle-reader-aarch64` | canonical | NOT RUN | PASS |
| Linux gemmini-int8-d16-ref | `test-gemmini-hp1-contract` | canonical | FAIL | REMOVED |
| Linux gemmini-int8-d16-ref | `test-im2p-cache-build-scripts` | canonical | FAIL | REMOVED |
| Linux gemmini-int8-d16-ref | `test-potal-target-profile` | canonical | FAIL | PASS |
| Linux potal-int8-d16 | `test-build-arm64-log-defaults` | Phase 1 | FAIL | PASS |
| Linux potal-int8-d16 | `test-evaluation-build-options` | Phase 1 | FAIL | PASS |
| Linux potal-int8-d16 | `test-evaluation-terminal-lm-head` | Phase 1 | PASS | REMOVED |
| Linux potal-int8-d16 | `test-gemmini-cycle-reader-aarch64` | Phase 1 | NOT RUN | PASS |
| Linux potal-int8-d16 | `test-im2p-cache-build-scripts` | Phase 1 | FAIL | REMOVED |
| Linux potal-int8-d16 | `test-potal-target-profile` | Phase 1 | FAIL | PASS |
| Linux potal-int8-d16 | `test-build-arm64-log-defaults` | canonical | FAIL | PASS |
| Linux potal-int8-d16 | `test-evaluation-build-options` | canonical | FAIL | PASS |
| Linux potal-int8-d16 | `test-evaluation-terminal-lm-head` | canonical | PASS | REMOVED |
| Linux potal-int8-d16 | `test-gemmini-cycle-reader-aarch64` | canonical | NOT RUN | PASS |
| Linux potal-int8-d16 | `test-im2p-cache-build-scripts` | canonical | FAIL | REMOVED |
| Linux potal-int8-d16 | `test-potal-target-profile` | canonical | FAIL | PASS |
| Linux potal-int8-d32 | `test-build-arm64-log-defaults` | Phase 1 | FAIL | PASS |
| Linux potal-int8-d32 | `test-evaluation-build-options` | Phase 1 | FAIL | PASS |
| Linux potal-int8-d32 | `test-evaluation-terminal-lm-head` | Phase 1 | PASS | REMOVED |
| Linux potal-int8-d32 | `test-gemmini-cycle-reader-aarch64` | Phase 1 | NOT RUN | PASS |
| Linux potal-int8-d32 | `test-im2p-cache-build-scripts` | Phase 1 | FAIL | REMOVED |
| Linux potal-int8-d32 | `test-potal-target-profile` | Phase 1 | FAIL | PASS |
| Linux potal-int8-d32 | `potal-stripe-pipeline-test` | canonical | PASS | FAIL |
| Linux potal-int8-d32 | `test-build-arm64-log-defaults` | canonical | FAIL | PASS |
| Linux potal-int8-d32 | `test-evaluation-build-options` | canonical | FAIL | PASS |
| Linux potal-int8-d32 | `test-evaluation-terminal-lm-head` | canonical | PASS | REMOVED |
| Linux potal-int8-d32 | `test-gemmini-cycle-reader-aarch64` | canonical | NOT RUN | PASS |
| Linux potal-int8-d32 | `test-im2p-cache-build-scripts` | canonical | FAIL | REMOVED |
| Linux potal-int8-d32 | `test-potal-target-profile` | canonical | FAIL | PASS |
| Linux potal-int4-d32 | `test-build-arm64-log-defaults` | Phase 1 | FAIL | PASS |
| Linux potal-int4-d32 | `test-evaluation-build-options` | Phase 1 | FAIL | PASS |
| Linux potal-int4-d32 | `test-evaluation-terminal-lm-head` | Phase 1 | PASS | REMOVED |
| Linux potal-int4-d32 | `test-gemmini-cycle-reader-aarch64` | Phase 1 | NOT RUN | PASS |
| Linux potal-int4-d32 | `test-im2p-cache-build-scripts` | Phase 1 | FAIL | REMOVED |
| Linux potal-int4-d32 | `test-potal-target-profile` | Phase 1 | FAIL | PASS |
| Linux potal-int4-d32 | `test-build-arm64-log-defaults` | canonical | FAIL | PASS |
| Linux potal-int4-d32 | `test-evaluation-build-options` | canonical | FAIL | PASS |
| Linux potal-int4-d32 | `test-evaluation-terminal-lm-head` | canonical | PASS | REMOVED |
| Linux potal-int4-d32 | `test-gemmini-cycle-reader-aarch64` | canonical | NOT RUN | PASS |
| Linux potal-int4-d32 | `test-im2p-cache-build-scripts` | canonical | FAIL | REMOVED |
| Linux potal-int4-d32 | `test-potal-target-profile` | canonical | FAIL | PASS |

The canonical PoTAL D32 `potal-stripe-pipeline-test` is PASS, but saved Phase 1 and final runs are both FAIL; it is the previously observed load-sensitive failure, not a new Phase 2 status change. All retained failing and missing-executable names are enumerated below. Their raw diagnostics remain in the referenced logs; failures outside the removal were not rewritten.

| Platform/configuration | Retained failing or unrun test | Canonical | Saved Phase 1 / earlier A4 run | Final |
| --- | --- | --- | --- | --- |
| Linux cpu-only | `test-chat` | UNSAVED | FAIL | FAIL |
| Linux cpu-only | `test-eval-callback` | UNSAVED | FAIL | FAIL |
| Linux cpu-only | `test-experiment-output` | UNSAVED | FAIL | FAIL |
| Linux cpu-only | `test-gemmini-cpu-cycle-validity-contract` | UNSAVED | FAIL | FAIL |
| Linux cpu-only | `test-gemmini-off-build` | UNSAVED | FAIL | FAIL |
| Linux cpu-only | `test-gguf` | UNSAVED | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `gemmini_direct_metrics_source_contract` | FAIL | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `gemmini_mxint_weight_routes_activation` | FAIL | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `gemmini_weight_formats` | FAIL | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `test-chat` | FAIL | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `test-eval-callback` | FAIL | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `test-experiment-output` | FAIL | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `test-gemmini-build-contract` | FAIL | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `test-gemmini-cpu-cycle-validity-contract` | FAIL | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `test-gemmini-exsia-profile-contract` | FAIL | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `test-gemmini-off-build` | FAIL | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `test-gemmini-profiled-domain-contract` | FAIL | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `test-gemmini-rmd` | FAIL | FAIL | FAIL |
| Linux gemmini-int8-d16-ref | `test-gguf` | FAIL | FAIL | FAIL |
| Linux potal-int4-d32 | `potal-api-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `potal-attention-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `potal-ggml-unit-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `potal-hw-timeline-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `potal-index-queue-test` | FAIL | FAIL | FAIL |
| Linux potal-int4-d32 | `potal-log-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `potal-paired-auto-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `potal-paired-execution-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `potal-paired-plan-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `potal-paired-schedule-test` | FAIL | FAIL | FAIL |
| Linux potal-int4-d32 | `potal-paired-unit-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `potal-residual-unit-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `potal-stripe-api-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `potal-stripe-pipeline-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `potal-tiling-unit-test` | NOT RUN | NOT RUN | NOT RUN |
| Linux potal-int4-d32 | `test-chat` | FAIL | FAIL | FAIL |
| Linux potal-int4-d32 | `test-eval-callback` | FAIL | FAIL | FAIL |
| Linux potal-int4-d32 | `test-experiment-output` | FAIL | FAIL | FAIL |
| Linux potal-int4-d32 | `test-gemmini-cpu-cycle-validity-contract` | FAIL | FAIL | FAIL |
| Linux potal-int4-d32 | `test-gemmini-off-build` | FAIL | FAIL | FAIL |
| Linux potal-int4-d32 | `test-gguf` | FAIL | FAIL | FAIL |
| Linux potal-int8-d16 | `potal-index-queue-test` | FAIL | FAIL | FAIL |
| Linux potal-int8-d16 | `potal-log-test` | FAIL | FAIL | FAIL |
| Linux potal-int8-d16 | `potal-stripe-pipeline-test` | FAIL | FAIL | FAIL |
| Linux potal-int8-d16 | `test-chat` | FAIL | FAIL | FAIL |
| Linux potal-int8-d16 | `test-eval-callback` | FAIL | FAIL | FAIL |
| Linux potal-int8-d16 | `test-experiment-output` | FAIL | FAIL | FAIL |
| Linux potal-int8-d16 | `test-gemmini-cpu-cycle-validity-contract` | FAIL | FAIL | FAIL |
| Linux potal-int8-d16 | `test-gemmini-off-build` | FAIL | FAIL | FAIL |
| Linux potal-int8-d16 | `test-gguf` | FAIL | FAIL | FAIL |
| Linux potal-int8-d32 | `potal-hw-timeline-test` | FAIL | FAIL | FAIL |
| Linux potal-int8-d32 | `potal-log-test` | FAIL | FAIL | FAIL |
| Linux potal-int8-d32 | `potal-paired-schedule-test` | FAIL | FAIL | FAIL |
| Linux potal-int8-d32 | `potal-paired-unit-test` | FAIL | FAIL | FAIL |
| Linux potal-int8-d32 | `potal-stripe-pipeline-test` | PASS | FAIL | FAIL |
| Linux potal-int8-d32 | `test-chat` | FAIL | FAIL | FAIL |
| Linux potal-int8-d32 | `test-eval-callback` | FAIL | FAIL | FAIL |
| Linux potal-int8-d32 | `test-experiment-output` | FAIL | FAIL | FAIL |
| Linux potal-int8-d32 | `test-gemmini-cpu-cycle-validity-contract` | FAIL | FAIL | FAIL |
| Linux potal-int8-d32 | `test-gemmini-off-build` | FAIL | FAIL | FAIL |
| Linux potal-int8-d32 | `test-gguf` | FAIL | FAIL | FAIL |
| macOS gemmini-a4w4-int8-d16-ref | `gemmini_mxint_weight_routes_activation` | UNSAVED | FAIL | FAIL |
| macOS gemmini-a4w4-int8-d16-ref | `gemmini_weight_formats` | UNSAVED | FAIL | FAIL |
| macOS gemmini-a4w4-int8-d16-ref | `test-chat` | UNSAVED | FAIL | FAIL |
| macOS gemmini-a4w4-int8-d16-ref | `test-eval-callback` | UNSAVED | FAIL | FAIL |
| macOS gemmini-a4w4-int8-d16-ref | `test-gemmini-build-contract` | UNSAVED | FAIL | FAIL |
| macOS gemmini-a4w4-int8-d16-ref | `test-gemmini-cpu-cycle-validity-contract` | UNSAVED | FAIL | FAIL |
| macOS gemmini-a4w4-int8-d16-ref | `test-gemmini-exsia-profile-contract` | UNSAVED | FAIL | FAIL |
| macOS gemmini-a4w4-int8-d16-ref | `test-gemmini-profiled-domain-contract` | UNSAVED | FAIL | FAIL |
| macOS gemmini-a4w4-int8-d16-ref | `test-gguf` | UNSAVED | FAIL | FAIL |
| macOS gemmini-int8-d16-ref | `gemmini_mxint_weight_routes_activation` | UNSAVED | FAIL | FAIL |
| macOS gemmini-int8-d16-ref | `gemmini_weight_formats` | UNSAVED | FAIL | FAIL |
| macOS gemmini-int8-d16-ref | `test-backend-ops` | UNSAVED | FAIL | FAIL |
| macOS gemmini-int8-d16-ref | `test-chat` | UNSAVED | FAIL | FAIL |
| macOS gemmini-int8-d16-ref | `test-gemmini-build-contract` | UNSAVED | FAIL | FAIL |
| macOS gemmini-int8-d16-ref | `test-gemmini-cpu-cycle-validity-contract` | UNSAVED | FAIL | FAIL |
| macOS gemmini-int8-d16-ref | `test-gemmini-exsia-profile-contract` | UNSAVED | FAIL | FAIL |
| macOS gemmini-int8-d16-ref | `test-gemmini-profiled-domain-contract` | UNSAVED | FAIL | FAIL |
| macOS gemmini-int8-d16-ref | `test-gguf` | UNSAVED | FAIL | FAIL |
| macOS mac-cpu-only | `test-backend-ops` | UNSAVED | FAIL | FAIL |
| macOS mac-cpu-only | `test-chat` | UNSAVED | FAIL | FAIL |
| macOS mac-cpu-only | `test-gemmini-cpu-cycle-validity-contract` | UNSAVED | FAIL | FAIL |
| macOS mac-cpu-only | `test-gguf` | UNSAVED | FAIL | FAIL |

### Final audits

- Case-insensitive `im2p|cycle_sim|LEGACY_BSV` search across current ggml, cmake, scripts, tests, tools, common and src: zero matches. Required historical/removal documentation is outside this executable-code scope; Phase 0/H3 and `docs/backend/*.md` were not rewritten.
- `git ls-files` contains zero paths matching the obsolete name. Old simulator filenames have no replacement alias or renamed resolver. The generic metadata tool is retained under `tools/model-inventory`.
- All 13 active matrix/PB directories were inspected: each has CMakeCache.txt, build.ninja and compile_commands.json; none contains IM2P, CYCLE_SIM or LEGACY_BSV. No GEMMINI_SW_PATH or IM2P_SIM_ROOT cache entries exist. This covers configure definitions, compiled include/command paths and generated link graph.
- Active and disabled-backend SDK rejection is exercised by passing `test-evaluation-build-options`; its environment and *_DEFAULT inputs fail with the SDK-boundary diagnostic. The explicit FP32 overlay exception remains in the source; no full FP32 hardware execution is claimed.
- `git diff --check HEAD` passes. Report disposition tables cover every disputed and deleted path; all final matrix outcomes and every changed/failing test name are recorded, with no pending result claimed complete.

All host execution here uses internal REFERENCE/native CPU paths. Successful cross compilation, configuration fingerprints and host functional models do not establish physical NPU timing or a physical PB graph result. Pre-existing compile warnings remain visible; they were not suppressed.
