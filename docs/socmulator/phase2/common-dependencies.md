# Common dependency boundaries

Before: each PoTAL backend/test source list compiled 18 Gemmini-owned software
sources directly and added the Gemmini include root plus a generated
gemmini_params.h shim. Profiling also lived under a Gemmini-named target.

After (one selected architecture/precision per configuration):

    ggml-gemmini -----------+
    ggml-potal ------------+--> ggml-npu-common --> ggml-base
    ggml-potal-w4a4 -------+           |
                                      +--> ggml-npu-profiling
    ggml CPU / llama / tools --------------> ggml-npu-profiling

Gemmini execution separately links gemmini-tiled-api / gemmini-api. PoTAL
execution separately links potal-tiled-api / potal-api. Common numerical and
profiling code has no reverse hardware link or include. Configuration templates
remain under common/config and generate architecture-specific headers without
linking their APIs into common software.

The common CMake module groups actual consumers by compile definitions/options,
features and source-specific properties, removes shared .cpp entries from the
consumer, and links a common archive. Same-configuration sources are compiled
once. Existing strict OBJECT compile probes retain their own instrumentation.
Production ggml-npu-common strips TESTING/TEST_OBSERVER/DIRECT_METRICS_TESTING;
backend test builds and other instrumented consumers link separately keyed
common-test archives. No differing metrics layout is linked as the same object.
No runtime multiprecision dispatch was added. Backend public linkage exposes the
common archive because their public internal headers expose common types.

Actual evidence: build-npu/phase2b-evidence/common-targets.dot, common-graph.log,
common-include-deps.log and layering-audit.json. The representative Gemmini
compile_commands audit has 21 production numerical sources, zero test-hook
leaks, zero identical common compile-command duplicates, and no forbidden
hardware dependencies in the compiler's ExSIA dependency closure. These are
observed build artifacts, not only a proposed diagram.

CPU-only configuration creates profiling only. API-only configuration returns
through the existing standalone API path before the GGML common module. Their
build/test evidence is listed in phase2b-test-results.md. Full CPU/NPU target
registrations are retained.

PoTAL no longer creates or searches its Gemmini parameter shim or Gemmini
software include root. No source forwarding headers remain. Obsolete generated
shim files may remain in pre-existing ignored build caches; they have no producer
or include-path consumer and disappear when those caches are removed normally.
Gemmini's remaining argument/geometry/executor/detail headers own real hardware
or execution responsibilities and are not forwarding-only compatibility files.
Architecture-side namespace imports resolve actual common declarations; common
code contains no Gemmini namespace/API/transport dependency.

The original public CMake/environment option names still feed identical values.
NPU_TRANSPORT remains independent of NPU_MATMUL_ALGORITHM: PB REFERENCE consumes
encoded commands. Geometry validation, manifests, single precision, physical
packing and final-drain transport ownership remain unchanged. Legacy public
option cleanup belongs to Phase 2D.
