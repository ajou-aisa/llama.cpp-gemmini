# Common profiling

The former ggml-gemmini-utils implementation now belongs to
 ggml-npu-common/profiling (target ggml-npu-profiling). Headers are under
profiling/include/npu; C++ namespaces are ggml::npu. CPU-only llama/GGML users
link this target without numerical common software or either hardware API.
Consumers include GGML CPU/backend code, llama-context, tools/main, backend
execution, ExSIA/residual code, and the existing profiling tests.

Moved functionality includes native CPU cycle/host readers, CPU exclusion and
correlation, semantic/layer IDs, trace context, log output, evaluation metrics,
performance aggregation, generic Session/Work/Context OpTrace, checked trace
validation/count reconciliation and native source/configuration provenance.
Shared matmul CPU samples, intervals, job metrics and RMD telemetry types were
split from Gemmini execution-state headers. Gemmini dump/scale dump and NPU
hardware-counter command issuance remain architecture-owned.

Host wall time, CPU rdcycle/perf samples, NPU counter windows, reference timing
and functional-model observations remain separate domains. No functional model
observation is relabelled as a hardware cycle. The existing C timing/logging
symbol names, environment variables, trace schema names/versions/fields and
Python parser contracts are preserved. Generated optrace provenance includes
npu/optrace.hpp; its real source hashes still cover the whole ggml source tree.
No simulator timestamp protocol or new schema was introduced.

PoTAL's timeline implementation is connected through owner-supplied ExSIA
observation callbacks. Generic software emits the existing semantic boundaries;
PoTAL retains clock sampling, role registration, IDs and serialization. There
is no common-to-PoTAL header or API dependency.

Incremental evidence under build-npu/phase2b-evidence:
profiling-cpu-final.build.log and profiling-cpu.ctest.log (4/4),
profiling-gemmini-final.build.log and profiling-gemmini.ctest.log (3/3), and
profiling-potal-verified.build.log plus profiling-potal.ctest.log. The latter
has a passing stripe pipeline and the same pre-existing hw-timeline bench-shape
assertion: 'selftest passes on the bench shape of the FPGA self-test
(MEASURED-NATIVE)', with 'window composition unknown'. This is not reported as
an all-green PoTAL profiler suite. Final results and failure-reason comparisons
are in phase2b-test-results.md.
