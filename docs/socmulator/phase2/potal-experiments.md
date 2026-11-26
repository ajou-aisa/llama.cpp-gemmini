# PoTAL production and experiment boundaries

The same isolation policy is applied independently in `ggml-potal` and
`ggml-potal-w4a4`; their numerical or API implementations are not merged.
Root backend CMake source lists still own production. Opt-in prototype
target definitions now live in `experiments/CMakeLists.txt` and
`experiments/prefetch-admission/CMakeLists.txt`.

| Group | Classification by callers and targets | Location and preserved coverage |
| --- | --- | --- |
| Paired GEMM, preload, resident ownership, scale slots/generations, ACC scheduling | A: production; `ops -> execution -> stripe/dense -> paired-auto -> potal.c` | Unchanged production owners and default source lists |
| LDB_ONLY, STREAM, Index Queue, early/next-K prefetch, managed completion | A: production, conditional on existing request policy and capability validation | API/encoder/issuer and regression tests retained; opt-in stays opt-in |
| E3 ACC restore and benchmark backend | C: replacement `potal.c`/`ops.cpp` translation units reachable only under `POTAL_E3_TESTS` | `experiments/microtile-pipeline/`; context/prototype assertions stay in `tests/` |
| E4A resident API | C: closed test-only ownership API, not production resident implementation | `experiments/resident-weight/potal-resident.{c,h}`; resident and model tests stay in `tests/` |
| E4B1 proposed general contract | C implementation with B contract assertions; no hardware issuer | `experiments/resident-weight/potal-e4b1.{cpp,hpp}`; test stays in `tests/` |
| E4B2 checked API and RTL-semantics emulator tests | B: active contract coverage of production `potal-api` | All C11, host, RoCC and hardware test files remain in `tests/`; `POTAL_E4B2_TESTS` remains beside API targets |
| E4C1 alternative sequential executor and benchmark | C: separate scheduler/graph adapter, not default production scheduler | `experiments/sequential-paired/`; `potal-e4c1-test.cpp` and shared emulators remain in `tests/` |
| E4C2-0 proposed prefix/admission model | C: synthetic event world and proposed query interface, retaining negative current-API checks | `experiments/prefetch-admission/potal-e4c2-0-test.cpp`; target/CTest identity unchanged |

Thirty files move across the two trees. `build-npu/phase2c-evidence/source-moves.json`
lists each old and new location. C/C++ changes relocate includes and repair
two experiment-only compilation defects: E3 uses the current
`POTAL_HW_INDEX_ENTRIES` geometry name, and the E4C1 benchmark includes
`paired-auto.hpp` before its fixture's `main` macro can rename the plan's
`main` member. No assertion, independent oracle, arithmetic, command issue
or cleanup logic is removed. No duplicate was deleted. The two precision
copies remain separate.

## Live contract documents retained

Both backends retain `docs/paired-gemm-software-contract.md` and
`docs/potal-paired-matmul-auto.md` at their original locations, with unchanged
contents. An initial archive classification was corrected after tracing
their current consumers:

- `matmul/paired.hpp:5` explicitly references the software contract.
  `paired.cpp:84` still implements `paired_tiled_matmul_auto`; plan, schedule,
  unit, resident-model and E3/E4C1 tests use it as an independent software
  oracle. Its software-only hardware refusal remains a real contract.
- `matmul/paired-auto.hpp:10` explicitly references the automatic paired
  execution document. Its planning, validation, scale/index residency and
  completion contracts describe the live `paired-auto.cpp` implementation,
  reached by the production issuer/stripe route. The backend README also
  links this document beside the current API and planner.

Their older deployment statements and DIM16/A8 examples do not make the
whole documents historical reports. In particular, the software emulator's
hardware refusal must not be confused with the separate supported H2
executor. INT4-specific stale prose is not silently promoted to an INT4
numerical specification. Current source and the precision contracts govern;
updating those older statements is separate from this byte-preserving move.
All four live documents and inbound source/README references are preserved.
`contract-document-review.json` records this decision under the evidence root.
The archive contains only the 36 H3 reports, whose dated measurements,
checkpoint/prototype status and external artifacts establish their historical
report role.

The following validation purposes remain: resident B lifetime and DMA
ownership; scale-generation conflicts; index isolation; descriptor and
invalid-state rejection; failed-fence quarantine; paired output correctness;
sticky failure; INT4 packing and checked INT64 reconstruction; known-bug
reproducers. Shared `tests/support/e4b2-emulator.hpp`, fixtures and production
paired emulators stay available to their contract-test consumers.
Benchmarks and graph-substitution files now have experiment ownership.

Existing options remain `POTAL_E3_TESTS`, `POTAL_E4A_TESTS`,
`POTAL_E4B1_TESTS`, `POTAL_E4B2_TESTS`, `POTAL_E4C1_TESTS`,
`POTAL_E4C2_0_TESTS`. No new public switch or wrapper hierarchy is added.
All default CTest registrations remain; no target is excluded to hide a
failure. E3 context/prototype executables historically had no `add_test`;
run them directly. Commands, exact selected targets and results are in
`build-npu/phase2c-evidence/check-experiments.sh`, `experiment-summary.log`
and `experiments/`.

To reproduce on Linux with the existing image:

```sh
docker run --rm -v "$PWD:/src" -w /src socm-baseline:ubuntu24 \
  bash build-npu/phase2c-evidence/check-experiments.sh
```

That selects all existing experiment options for INT8 D16/D32 and INT4 D32,
builds all host experiment executables including benchmarks, runs the
resident/E4 CTest group, then the two direct E3 acceptance executables.
Physical hardware executables remain RISC-V-only and are NOT RUN here.
Known precision/configuration limitations remain explicit in the result ledger.
Historical measurements and rejected prototypes are indexed in
[the archive](../../archive/potal/README.md), with byte-identical provenance.

## Separate P2B-R INT4 attention fixture task

The fixture is unchanged in 2C. It still assumes INT8 interfaces and remains
BLOCKED. The current INT4 contract is radix-16, nine Q limbs, u8 probability,
CPU INT64 column bias, `paired_runner`, checked INT64 reconstruction and
explicit legacy PV refusal.

Classify each assertion as `COMMON_CONTRACT`, `PRECISION_SPECIFIC`,
`REMOVED_ALGORITHM`, `INVALID_CONFIGURATION`, `OWNERSHIP_AND_COMPLETION`,
or `NUMERICAL_BOUNDARY`; preserve a per-assertion mapping before replacing it.
The ten-step migration before Phase 3 is:

1. Build an independent INT4 preparation oracle.
2. Build an independent division/remainder radix-16 digit oracle.
3. Check raw ACC partials and saturation separately from reconstruction.
4. Cover strided and broadcast F16/F32 inputs.
5. Compare paired and strict supported execution.
6. Prove invalid-input and overflow rejection.
7. Inject failures and prove no partial publication.
8. Assert explicit legacy PV refusal.
9. Migrate stale benchmark, smoke and emulator consumers.
10. Split W8-only and INT4 targets with every original validation purpose mapped.

The earlier 181-assertion gap ledger in `test-readiness.md` remains the
starting evidence. Existing failures and disabled W8-only registrations
are retained rather than silently replaced by easier INT4 cases.
