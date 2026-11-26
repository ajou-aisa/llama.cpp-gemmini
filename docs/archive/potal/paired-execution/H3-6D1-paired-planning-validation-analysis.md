# H3-6D1 Paired Planning/Validation analysis

Date: 2026-10-25. Scope: measurement and feasibility analysis, not a Planning optimization.

Software baseline: `potal`, `26ec82756feedb131dc5acc6e93e16a0b04512ca`, matching live `origin/potal` at task entry. Hardware-source repository: Chipyard `potal`, `c705d8a7090933a15652165ce64eac196a0a5e4a`, matching live `aisa/potal`. The outer FireSim repository remains on `aisa-customizations`, `5c3458a3f54917ab9502bda4a8b3cc3dd513304d`; it has no `aisa/potal` ref. No repository was switched.

The seven H3-6A, B0, B1, B2, B3, B4 and C reports were read before instrumentation. B2's trace cold path and C's decoded-packet fallback reuse are the baseline, not new D1 work. The older B1 Planning measurements (Decode 2.3345 ms and FFN 13.7265 ms) are separate, older diagnostic observations. They are not the current Planning budget.

Source references below are relative to `ggml/src/ggml-potal/`. Line numbers refer to the D1 working tree unless a baseline is named.

## 1. Executive Summary

**Recommend H3-6D2 option B: investigate the unnecessary report deep copies first, on a stripe-private path with explicit lifetime proof.** Preserve the existing public report's ownership, both complete validators, request/session checks and exact commands. Do not begin with a general cache, allocator pool or shared mutable Plan.

The normal route constructs one Plan and validates it twice. It performs four populated assignment sites, containing three tile copies and two residual-provenance copies. The two report assignments account for almost all measured copy work. Their diagnostic inclusive envelopes are **2.255 ms for Decode and 10.153 ms for FFN**, versus **0.891 ms and 4.644 ms** for the second validator. These envelopes include nested allocator and collector costs; they are not release-time savings.

Fresh unprofiled steady Paired compute is **14.754 / 0.025 ms median/MAD** for Decode and **79.069 / 0.125 ms** for FFN. Serial is **9.189 / 0.056 ms** and **52.743 / 0.146 ms**. D1 does not improve these times.

In the active diagnostic run, exclusive Planning work is **6.848 / 0.031 ms** and **31.845 / 0.042 ms**, with measured collector bookkeeping of **3.240 / 0.012 ms** and **12.799 / 0.059 ms** separately. Allocation/deallocation and construction are substantial; Validation is not the whole Planning cost. Counts and logical bytes are stable across repetitions, cold/steady execution and both Paired routes.

The instrumentation is intrusive: active Paired graph time is 53.67% and 42.17% above the original OFF controls. Subtracting the bookkeeping bucket does not recover production component times. Section 7 gives conditional diagnostic Amdahl ceilings, not promised speedups. Actual D2 performance benefit is **Not measured**.

All nine existing Host regressions pass OFF and ON; accounting tests pass natively and on RISC-V. All three guest benchmark/quick modes and the explicit capacity-fallback case pass. Across 252 benchmark samples, results are Golden-exact and command-count fields match all variants. The guest was powered off cleanly; the driver exited 0 with `*** PASSED ***`.

D1 adds only default-OFF instrumentation, its tests and this report. OFF planner instruction output is identical to the committed baseline. No Planning/Scale optimization, validation removal, ownership change, cache, numerical change, RTL rebuild, branch change, commit or push was performed.

## 2. Current Call Graph

The normal supplied-plan path builds once and validates twice per accepted Paired stripe:

```text
ggml-potal.cpp: potal_graph_compute
  preflight supported nodes
  ops.cpp: compute_mul_mat / mul_mat_hp1
    checked HP1 shape, prepared-weight cache, Main tiler, init_args
    mul_mat_hp1_pipeline
      stripe.cpp: run_stripe_pipeline
        stripe_run::begin / submit
          validate event, stripe order and capacity
          retain slot-owned A/C and immutable packet
        stripe_run::execute
          residual::decode_packet                           once per residual stripe
          potal_paired_plan_stripe                          route root
            potal_make_tile_plan
              potal_paired_fragments
              hardware geometry, capacities, panels/windows
              frozen Main plan validation
            potal_make_execution_plan
              copy tile and residual-row provenance
              build ordered operations/dependencies
              potal_validate_execution_plan                first full validation
              move completed plan into routing
          runners.paired
            issuer-affinity check, session open
            potal_paired_matmul_auto                        executor root
              reset report; bind request and hardware geometry
              validate operand pointers, strides and generation
              match supplied plan to request
              potal_validate_execution_plan                second full validation
              report.plan = *req.plan                       deep copy
              report.tile = req.plan->tile                  another deep copy
            end executor root
            pack residual DMA staging
            validate idle session; TLB maintenance
            executor::run: ordered hardware commands
            final fence / drain, readback, error handling
            session close
          radix reconstruction and checked F32 merge
          semantic completion; release slot
          destroy stripe-local report, routing and decoded metadata
```

Entry points and boundaries:

| Boundary / caller | Inputs and state | Frequency on a successful supplied-plan stripe | Failure / observable effect | Serial difference |
|---|---|---:|---|---|
| `ggml-potal.cpp:120`, `ops.cpp:336,595` | Node type, dimensions, strides, HP1 weight contract, backend options | Per graph/node | Reject unsupported node or invalid execution configuration before issue | Shared frontend contract |
| `matmul/stripe.cpp:155,193,217,394` | Run/stripe identity, row span, producer buffers, queue/slot generation | Begin once per run; submit once per stripe | Invalid event/order, capacity, allocation failure; no partial acceptance | Shared pipeline; route-specific preparation follows |
| `residual/stripe-residual.cpp:142` | Sealed packet, prepared weight, radix, blocks/indices/digits | Once per residual packet | Packet/argument/route/overflow/allocation status | Direct Serial calls through `build_request`; Paired decodes before route choice |
| `matmul/paired-auto.cpp:2185` | Main plan, stripe rows, N/K, decoded runs/rows/digits, eight geometry fields | Once per Paired route attempt; a capacity fallback has no executor root | Capacity refusals choose existing Serial fallback; other failures fail stripe | Direct Serial has no Paired route plan |
| `potal_make_tile_plan:281` and fragments at `149` | Main I/J/K, passes, active residual fragments, capacities | Once per route attempt | Invalid input or precise SPAD/ACC/Index/Scale capacity refusal | Serial residual uses K32-padded lowering and its own tiler |
| `potal_make_execution_plan:682` | Tile, same decoded packet, local builder vectors | Once after successful tile planning | Invalid execution plan or allocation failure; output moves only after validation | No Paired operation/dependency graph |
| `potal_validate_execution_plan:1506` | Const plan, decoded packet, geometry; fresh scratch vectors | Twice per normal accepted stripe | Invalid execution plan or allocation failure | Serial descriptor checks remain, but not this Paired validator |
| Owner/session at `2210` onwards; executor at `2245` | Current bindings, physical-hart owner, session/generation, supplied routing plan | Once per accepted stripe | Request/plan/session/issue failure; sticky hardware errors retained | Both routes use the issuer; synchronization differs by existing route |
| After executor root, `2346` onwards | DMA staging, mutable session and executor state | Per execution, op/window/command | Stage/issue/fence/readback/cleanup status; fatal outstanding DMA is abandoned | Not a reusable Plan-validation result |
| `stripe.cpp:868` onwards and `1092` | Returned INT32 rows, packet radix/provenance, theta, destination | Once per completed stripe | Checked overflow/nonfinite/reconstruction failure; no partial semantic commit | Same independent numerical contract |

`stripe.cpp` retains an older “planned again” comment. The actual request sets `pq.plan = &routing`: the executor validates and copies that plan; it does not regenerate it.

C fallback boundary remains `stripe.cpp:978`: an already decoded Paired capacity fallback calls `build_request_from_decoded(std::move(decoded), ...)`. Direct Serial still uses `build_request(packet, ...)`. The lowerer moves runs/row provenance; it continues reading `decoded.digits` while creating padded DMA A. Moving the entire decoded packet earlier would be wrong.

## 3. Planning Cost Breakdown

### Graph controls

Each cell is seven-sample compute **median / MAD in ms**. These are separate processes in baseline, inactive, active order; modes were not interleaved. Routes rotate/reverse within each process.

| Case | Phase | Route | Original OFF | ON, inactive | ON, active |
|---|---|---|---:|---:|---:|
| Decode | cold | serial | 24.006 / 0.052 | 24.377 / 0.034 | 24.584 / 0.304 |
| Decode | cold | paired | 29.545 / 0.042 | 30.594 / 0.015 | 39.103 / 0.013 |
| Decode | cold | paired_conservative | 29.716 / 0.035 | 30.823 / 0.042 | 37.793 / 0.033 |
| Decode | steady | serial | 9.189 / 0.056 | 9.443 / 0.080 | 9.371 / 0.079 |
| Decode | steady | paired | 14.754 / 0.025 | 15.653 / 0.006 | 22.673 / 0.011 |
| Decode | steady | paired_conservative | 14.977 / 0.015 | 15.895 / 0.013 | 22.890 / 0.017 |
| FFN | cold | serial | 115.680 / 0.091 | 116.398 / 0.068 | 116.319 / 0.053 |
| FFN | cold | paired | 142.136 / 0.026 | 144.752 / 0.174 | 174.858 / 0.124 |
| FFN | cold | paired_conservative | 142.887 / 0.095 | 145.678 / 0.122 | 174.785 / 0.107 |
| FFN | steady | serial | 52.743 / 0.146 | 54.016 / 0.064 | 53.842 / 0.047 |
| FFN | steady | paired | 79.069 / 0.125 | 82.242 / 0.051 | 112.412 / 0.030 |
| FFN | steady | paired_conservative | 79.938 / 0.138 | 83.100 / 0.133 | 112.269 / 0.213 |

Inactive Paired differs from OFF by +0.899 ms (+6.09%) for Decode and +3.173 ms (+4.01%) for FFN. Active differs by +7.919 ms (+53.67%) and +33.343 ms (+42.17%). Even Serial, with zero collected Paired roots, shifts by +0.182 ms (+1.98%) and +1.099 ms (+2.08%) in active mode. These are measured net process/control differences, not isolated costs of a particular wrapper or a pure hardware-wait change.

### Exclusive component accounting

These are **active diagnostic, steady Paired RTL** samples. Both cases have one route root and one executor root per graph/stripe. Serial has zero roots. Median/MAD:

| Case | Route root ms | Executor root ms | Combined roots ms | Bookkeeping ms | Planning denominator ms |
|---|---:|---:|---:|---:|---:|
| Decode | 6.892 / 0.016 | 3.201 / 0.009 | 10.088 / 0.015 | 3.240 / 0.012 | 6.848 / 0.031 |
| FFN | 29.687 / 0.021 | 14.961 / 0.024 | 44.640 / 0.038 | 12.799 / 0.059 | 31.845 / 0.042 |

Planning excludes only the explicitly measured `instrumentation` component from the two root totals. It remains diagnostic time. Median Planning/active-graph ratios are 30.22% and 28.33%; they are not production removable fractions.

| Case | Exclusive component | Calls per graph | Median us/call | Total ms median / MAD | Median share of Planning |
|---|---|---:|---:|---:|---:|
| Decode | validation | 9 | 102.444 | 0.922 / 0.004 | 13.39% |
| Decode | copy | 4 | 186.000 | 0.744 / 0.012 | 10.91% |
| Decode | allocation | 1226 | 1.973 | 2.419 / 0.030 | 35.59% |
| Decode | deallocation | 446 | 1.859 | 0.829 / 0.021 | 12.11% |
| Decode | construction | 3 | 635.000 | 1.905 / 0.008 | 27.81% |
| Decode | other | 2 | 9.500 | 0.019 / 0.000 | 0.28% |
| Decode | instrumentation | 5066 | 0.640 | 3.240 / 0.012 | Outside denominator |
| FFN | validation | 9 | 689.000 | 6.201 / 0.007 | 19.46% |
| FFN | copy | 4 | 927.000 | 3.708 / 0.016 | 11.66% |
| FFN | allocation | 4978 | 2.011 | 10.012 / 0.026 | 31.48% |
| FFN | deallocation | 1710 | 1.901 | 3.251 / 0.023 | 10.19% |
| FFN | construction | 3 | 2880.667 | 8.642 / 0.038 | 27.15% |
| FFN | other | 2 | 10.000 | 0.020 / 0.001 | 0.06% |
| FFN | instrumentation | 20114 | 0.636 | 12.799 / 0.059 | Outside denominator |

Calls are scope entries or wrapped allocator calls, not hardware commands. `us/call` is the aggregate time divided by calls; Validation/Construction combine heterogeneous scopes. Instrumentation calls count bookkeeping intervals and inventory scopes. Observed clock records are quantized to 1 us; the `_ns` fields and fractional aggregate averages do not imply nanosecond measurement resolution.

Every individual root satisfies the exact exclusive component sum and exact exclusive site sum. Medians and median shares are computed independently, so their displayed columns need not add exactly. No overlapping inclusive interval is added to this decomposition.

### Function/site attribution

These **inclusive** times contain nested allocation/deallocation and instrumentation. Each listed site runs once per accepted stripe. They must not be added to the component table, or summed across enclosing/child sites.

| Phase / site | Decode ms median / MAD | FFN ms median / MAD | Interpretation |
|---|---:|---:|---|
| route / fragment_build | 0.121 / 0.000 | 0.129 / 0.000 | Fragment construction plus its nested checks/allocations |
| route / tile_build | 0.199 / 0.001 | 0.218 / 0.000 | Includes fragment build and frozen-plan checks |
| route / execution_build | 6.668 / 0.009 | 29.464 / 0.022 | Includes builder copies and first full validation |
| route / execution_validate | 0.878 / 0.003 | 4.563 / 0.012 | First complete validation |
| executor / execution_validate | 0.891 / 0.003 | 4.644 / 0.012 | Second complete validation, candidate A envelope |
| executor / executor_request | 0.004 / 0.000 | 0.004 / 0.000 | Mandatory current bindings |
| executor / supplied_match | 0.002 / 0.000 | 0.003 / 0.000 | Current request/plan association |
| route / builder_tile_copy | 0.023 / 0.000 | 0.024 / 0.000 | Tile assignment |
| route / builder_rows_copy | 0.006 / 0.000 | 0.007 / 0.000 | Provenance assignment |
| executor / report_plan_copy | 2.231 / 0.019 | 10.126 / 0.028 | Populated execution-plan assignment |
| executor / report_tile_copy | 0.024 / 0.000 | 0.027 / 0.000 | Redundant owned tile assignment |

The nine Validation calls are six route scopes (fragment input, fragment rows, tile input, frozen tile, execution input, full validator) plus three executor scopes (request, match, full validator). Individual mixed-loop checks inside construction were not retimed into an invented pure-validation estimate.

The full validator allocates per-residency, Scale-window, ACC and GEMM-order scratch at `1117` onwards. Its Scale-program liveness check scans all Scale windows at `1214`, giving an O(S squared) scan envelope as window count grows. The observed larger FFN graph increases validation faster than the tiny input checks. D1 does not optimize either the scan or scratch storage.

Preparation before the route root, packet decode, residual DMA packing, hardware issue/wait and final Plan teardown are outside these Planning totals. Their new D1 component split is **Not measured**.

## 4. Validation Duplication Analysis

Classification is about a future trusted internal path. D1 removes no check.

| Invariant and existing sites | What can change between sites? | Class | Requirement before any future reuse/removal |
|---|---|---|---|
| Graph/node HP1 dimensions, type and layout; `potal_graph_compute`, `check_mul_mat_hp1`, `init_args` | Another graph/node, externally mutable tensor or buffer | V1 at public boundary; V2 within one prepared run | A prepared immutable descriptor tied to the actual buffer and layout; public rejection behavior retained |
| Packet seal/geometry and prepared-weight association; `decode_packet` and packet readers | A new packet, weight, radix or payload | V1 for untrusted/new input; fixed packet-header repetitions are potential V3 only within one immutable packet | Successful checked decode, immutable packet lifetime and exact weight association; per-digit access/range checks are not blanket duplicates |
| Decoded run compact offsets, local-K order/range, row provenance and digit extents; fragments/tile/execution input checks | Normally no mutation on the internal route-to-executor path | V3 only for the same decoded object and geometry; V1/V2 for standalone public builders | Proof of checked decode and exact packet identity; no unchecked external `decoded_packet` admitted |
| Main tile, stripe rows and N/K; tile builder, frozen-plan validation, supplied-plan match | Tails, alternate Main tile, diagnostic geometry, distinct caller binding | V2 | Match full stripe/tile/geometry identity; equality of rounded tile counts alone is insufficient |
| Derived SPAD partition, ACC work IDs, Index windows and Scale-slot capacities | Hardware geometry or partition may differ | V1 for newly constructed/public plans; V3 for an unchanged checked internal plan | Preserve all capacity refusals; bind proof to all geometry fields and adapter assumptions |
| Ordered op IDs/deps, pair residency/retirement, loop-slot completion, Index entries, Scale lifetime, ACC first/last; first full validator in builder versus second in executor | Normal stripe passes `const routing`; no mutation occurs before the second validator | V3 on that closed internal handoff | A non-forgeable validated internal representation or equivalent ownership boundary; retain the complete public supplied-plan validator |
| Request A/B/C/residual pointers, alignment, leading dimensions, weight geometry and generation; executor request checks | New DMA buffers and each invocation's generation | V1 | Rebind and validate every execution; a valid pure schedule does not validate memory |
| Runtime session idle/failure/open Index transaction, liveness, actual generation, drain/readback state | Every command and failure changes it | V1 | Never cache away; static Plan generations are not actual session generations |
| Reconstruction row mapping, radix accumulation, output extents/overflow/finite values | Digits, theta and returned outputs change each execution | V1 | Keep checked numerical merge/reconstruction, even if metadata is shared |

The first and second complete validators use the same routing plan, same decoded packet and same geometry on the normal internal path. No hardware command runs between them. The executor's independent binding and session checks are different checks, not duplicate validation.

There are important limits to the existing public supplied-plan validator. These are static source findings, not executed exploit/reproducer results:

* `potal_paired_fragment_original_k` at `266` combines the stored fragment's `block_id` with the current decoded run's `local_k`. It does not compare that run's new `block_id` to the fragment. Equal sizes with changed block provenance are not a cache key.
* The validator checks `plan.residual_rows.size()` at `1089`, not full equality to the current decoded `(lane, source_row)` sequence. Changed provenance with equal counts can leave stale reconstruction metadata.
* The Index check at `1349` calls the fragment helper without rebuilding the builder's complete run/local-K bounds proof. Reusing against shorter/different runs is not protected by shape equality.
* `report = potal_paired_report{}` precedes access to `req.plan`. A caller must not point `req.plan` into that same output report.

Consequently, “keep the existing validator and cache by M/N/K” is not a valid reuse design. These pre-existing concerns were not fixed or expanded into unrelated behavior changes in D1.

## 5. Deep Copy and Ownership Analysis

### Copy inventory

| Object / site | Fixed or dynamic contents | Current ownership / logical copy count per normal stripe | Future feasibility |
|---|---|---|---|
| `potal_tile_plan`, geometry, request and op scalar fields | Dimensions, padding, offsets, statuses, generation labels | Value copies; no DMA ownership | Small fixed copies should not drive an ownership redesign |
| `potal_paired_tile_plan` | Fixed geometry, fragment vector, panel vector and each panel's window vector | `b.plan.tile = tile`, nested `report.plan.tile`, and `report.tile`: three logical deep copies | Immutable metadata could be shared only under an explicit owner |
| Decoded residual row provenance | `(uint8_t lane, uint32_t source_row)` | Builder copies into `plan.residual_rows`; report-plan assignment copies it again | Immutable within one accepted packet; identity must include the whole ordered sequence |
| Full execution plan | Tile, rows, ops and per-op deps, Index entries, Scale windows, residencies, transitions, fixed pair summaries/counts | One populated deep copy, `report.plan = *req.plan` | A borrowed/moved internal view may remove it, but public report ownership and lifetime must not silently change |
| Stripe descriptors / packet | Row range, slot generation, immutable shared packet, slot-owned A/C | Request borrows references; packet lifetime spans owner consumption | Not the same as copying DMA data; slot cannot be released on command issue |
| Residual Index/Scale metadata | Ordered index mappings, Scale scheduling windows and dependencies | CPU vectors inside execution plan | Shareable only when complete topology, weight association and geometry match |
| Carrier payload, prepared B, A codes, C and residual staging | Actual DMA data | Not owned by Plan vectors; borrowed or separately allocated | Keep current DMA ownership and fence/drain lifetime |
| Runtime executor/session state | Staging, mapped generations, pending readbacks, `drained_upto`, last op, counts, errors, tracing/diagnostic buffers | Fresh per execution/session | Never share as immutable Plan state |

`execution_builder::add(op)` copies a fixed op whose dependency vector is still empty, then stores/moves it. This is not another copy of a populated dependency graph. Vector growth can move/copy implementation storage; the explicit logical-copy inventory does not claim to count those bytes.

### Exact logical-field formulas

Let `s = sizeof(size_t)`, `F = fragments`, `P = panels`, `W = total panel windows`, `L = residual lane rows`, `O = operations`, `D = dependency edges`, `X = Index entries`, `S = Scale windows`, `R = residencies`, and `Z = pair transitions`.

```text
T = 47s + 5*sizeof(uint32_t) + sizeof(bool)
    + 20F + 12sP + 8sW

E = T + 10s + 2*(6s + sizeof(pair_phase) + 3*sizeof(bool))
    + 5L + (30s + sizeof(op_kind) + 3*sizeof(bool))*O
    + (s + sizeof(bool))*D
    + (5s + sizeof(bool))*X
    + (11s + sizeof(bool))*S
    + 13sR + (3s + 2*sizeof(pair_phase))*Z

four assignment sites together = T + 5L + E + T
```

These count declared logical fields after actual assignments. They exclude struct padding, vector headers, unused capacity, allocator metadata and memory-bus traffic. On this 64-bit target the fixed part of `T` is 397 bytes. `objects` counts logical nested records, not heap objects.

With `G` J groups and `H` residual presence, the current builder's schedule scales as:

```text
R = G*P
Q = G*W                         residual GEMMs
S = R+Q
Z = 3R
O = 6R+5Q+G*(1+H)
X = G*sum(window.active_rows)
```

Thus shape affects cost through actual panel/window/topology counts, not only M*N*K. A new residual pattern can change active fragments/windows and the Main common-K partition. These are source-derived identities, not measured cache traffic.

### Measured copy and allocator inventory

Each assignment runs once per normal stripe, including cold/steady and Conservative. Logical object counts use the definition above.

| Assignment | Decode bytes / objects | FFN bytes / objects |
|---|---:|---:|
| Builder tile | 1609 / 32 | 1837 / 39 |
| Builder residual rows | 5 / 1 | 65 / 13 |
| Report execution plan, including its tile/rows | 150916 / 2266 | 723114 / 11759 |
| Report tile | 1609 / 32 | 1837 / 39 |
| All four sites | **154139 / 2331** | **726853 / 11850** |

The two report assignments copy 152525 and 724951 logical bytes. Their combined inclusive times, calculated per sample before taking the median, are **2.255 / 0.019 ms** and **10.153 / 0.028 ms**. All four sites together take **2.284 / 0.019 ms** and **10.183 / 0.027 ms** inclusively. Exclusive copy-only shares are 10.91% and 11.66% of the defined Planning denominator; inclusive assignment times also contain allocator and diagnostic work.

| Case / root | Ordinary new calls | Requested bytes | Delete calls | Known sized-delete bytes |
|---|---:|---:|---:|---:|
| Decode / route | 780 | 446096 | 390 | 222508 |
| Decode / executor | 446 | 184164 | 56 | 5892 |
| Decode / total | **1226** | **630260** | **446** | **228400** |
| FFN / route | 3152 | 2173060 | 1518 | 1085052 |
| FFN / executor | 1826 | 877052 | 192 | 24244 |
| FFN / total | **4978** | **3050112** | **1710** | **1109296** |

These are observed target-resolved requests, not estimates from vector size. Per-copy-site allocator attribution, unsized freed bytes, peak live memory and final post-execution destruction cost are **Not measured**. Allocation count minus delete count is not a leak measurement: live output Plan/report vectors intentionally leave these roots.

The observed command counters and logical inventory recover the following topology through the stated builder identities. This is a derived structural cross-check of actual measurements, not another runtime exporter or a traffic estimate:

| Case | G | P | W | F | L | O | D | X | S | R | Z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decode | 6 | 2 | 10 | 19 | 1 | 384 | 748 | 978 | 72 | 12 | 36 |
| FFN | 22 | 2 | 12 | 24 | 13 | 1628 | 3278 | 6314 | 308 | 44 | 132 |

For example, G is readbacks/2 because these cases have Main and Residual results; R and Q are measured Paired Main/Residual loops; P=R/G and W=Q/G. T yields F; measured row-copy bytes yield L; logical object counts yield D. Substitution reproduces both measured E byte totals exactly. These two input patterns demonstrate scaling, not a universal regression model for arbitrary residual topology.

### Lifetimes and asynchronous safety

The run owns queued slots; accepted A and metadata stay immutable until consumption completes. A shared immutable packet and the prepared-weight owner's lifetime cover decode, plan, staging, hardware and reconstruction. The synchronous owner call returns only after the relevant completion/error handling. The run joins its issuer work before releasing the enclosing owner.

`routing`, the decoder and the returned report are stripe locals. Their vectors hold CPU metadata, not DMA buffers. A plan's fixed `pairs[]` describes the generated final state; it is not the actual accelerator's live state. Requests borrow operand/weight/residual/plan pointers. Read-only `const` alone does not extend their lifetime.

Actual command issue is not completion. Staging, residual C and slot A/C must survive the existing fence/drain/readback boundaries. On fatal or unknown outstanding DMA, the current code abandons reachable buffers instead of returning them to a pool. Sharing a report view past `stripe_run::execute`, moving a plan while another observer uses it, recycling a slot on issue, or pooling abandoned DMA would create dangling access or reuse hazards.

The roots measure allocator/deallocator calls encountered during planning, including validator scratch and growth/temporary storage. Final stripe-local Plan/report destruction after execution is **Not measured**. D1 does not move destruction into a timer by changing ownership.

## 6. Plan Reuse Feasibility

There is currently no Plan cache. Same shape alone is not reusable identity.

| Factor | Pure Plan identity / immutable-share candidate | Refresh per execution | Mandatory revalidation |
|---|---|---|---|
| M/N/K | Exact stripe rows, N/K, tails; global M determines stripe partition | Actual row origin and destination binding | Request-to-plan match, exact tail geometry |
| Stride/layout | Schedule itself has no operand pointers/strides | A/B/C/carrier bases, leading dimensions, extents | Pointer validity, alignment, stride and weight association |
| Tile/stripe partition | Main I/J/K, passes/block, panels, windows, padding and stripe rows | Run/stripe identifiers | Partition/tail equality; fallback compatibility |
| Hardware capacity | All eight geometry fields and compiled adapter assumptions | Current hardware/session association | SPAD, ACC, Index, Scale and work-ID limits; no bypass of capacity refusal |
| Main/residual partition | Residual presence, lane rows, fragments/windows, reservations and resulting common K | Current packet and actual digits | Complete topology, not residual count alone |
| Residual indices/metadata | Ordered block/local-K runs, compact offsets, lane/source rows, active fragments and fillers | Digit payload and staging contents | Full provenance and extent identity; current public validator is not sufficient as a cross-packet cache guard |
| Quantization/Scale metadata | Immutable prepared-weight metadata under its real owner; Scale schedule is structural | Activation codes, theta, carrier/weight binding and generation | Radix/type/weight contract and all numerical checks; carriers are not owned by the Plan |
| Execution state/order | Immutable op/dependency sequence and plan-local labels | Actual session generation/liveness, Index transaction, readbacks/drain state, counters, sticky errors and trace sequence | All existing execution/session checks and exact command order |

The narrowest plausible future reuse scope is one already checked immutable routing object handed directly to its executor. Broader reuse needs an explicit validated identity and owner, fresh execution state, mutation invalidation and bounded storage. A shape-only global cache is not justified. Sharing CPU metadata does not authorize sharing DMA lifetime or failure state.

## 7. Optimization Candidates and Risk Ranking

The bounds below refer to measured diagnostic work, not guaranteed release-time savings. Candidate budgets overlap. In particular, a validated internal handoff and duplicate-validation removal address the same second validator; metadata sharing and copy elimination address the same copies.

| Candidate | Responsible sites / narrow possible change | Budget that can be bounded | Correctness and memory risk | Required proof/tests before implementation |
|---|---|---|---|---|
| 1. Remove or hoist duplicate validation | Builder validation at `1028` and supplied-plan validation at `2310`; retain the first complete validation and per-execution request/session checks | Second `execution_validate` site only, including its nested scratch allocation if an inclusive bound is used; not both validators or all Validation | Forged/mutated public plan, wrong packet/geometry, lost rejection or error status | Malformed-plan rejection through existing public entry, exact same-object handoff proof, tails/capacity/packet variations, allocation failure, command records and independent Goldens |
| 2. Separate validated internal Plan path | `stripe_run::execute` constructs `routing`; owner passes it to the executor without admitting arbitrary public structs | Same budget as candidate 1, not an additional saving | A boolean “validated” flag on a mutable struct is forgeable; borrowed proof can outlive its owner | Private ownership/type boundary, immutable plan/packet/geometry association, public API tests unchanged, mutated-plan and cross-packet rejection, error/cleanup and lifetime tests |
| 3. Share immutable metadata | Tile fragments/panels/windows, residual provenance, ops/deps and Index/Scale/residency metadata | Measured copy sites, with their allocations counted once; sharing-management cost is Not measured | Reference-count/arena lifetime, mutation through another alias, unbounded retained metadata, confusion with DMA ownership | Exact lifetime through reconstruction, async owner completion, slot recycling, cancellation/fatal paths, no shared runtime generations/state, memory bounds |
| 4. Eliminate unnecessary deep copies | `report.plan = *req.plan` and `report.tile = req.plan->tile`; builder's tile/provenance assignments are separate | Two report-copy site inclusive intervals; nested tile/provenance must not be counted again | Public report currently owns vectors; replacing these with views can dangle or alter observers; aliasing input/output already unsafe | Preserve current public report contract, fresh/reused report tests, no-plan/supplied-plan paths, caller ownership/alias tests, command and numerical equivalence |
| 5. Reuse allocation storage | Builder vectors, per-op dependency growth, validator scratch; final teardown remains outside measured roots | Exclusive allocation + deallocation components only; copies/construction remain required | Cross-run stale vector contents, nondeterministic capacity retention, concurrent reuse, pooled fatal DMA | Deterministic reset, shape growth/shrink/tails, failure injection, concurrency/owner tests, high-water bound and no DMA pooling |
| 6. Construct a smaller Plan | Operation/dependency/Index/transition construction in `potal_make_execution_plan` and fragment/panel derivation | Exclusive `construction` is an envelope, not all safely removable work | Lost ordering/completion deps, wrong fillers, altered ACC/Scale/Index semantics, changing diagnostic/public report content | Independent validator retained, ordered command/argument equality, all schedule/auto/execution tests, saturation and reconstruction Goldens, no weakened rejection |
| 7. Broader Plan cache/reuse | Route planning across stripes/graphs, only after an explicit identity/owner design | Route construction/validation/copy work is a gross envelope; identity checking, lookup, invalidation and retained memory are Not measured | Equal shape with different runs/lanes, tail/window/geometry changes, stale prepared weights, UAF or unbounded cache, mutable session state | Full identity from Section 6, cross-packet negative cases, weight invalidation, tails/capacity/generation wrap, bounded eviction and failure/async lifetime tests |

Candidates 1 and 2 are one potential D2 workstream, not a request to weaken public validation. Candidates 3 and 4 need a separate ownership design before performance work. Candidate 5 does not justify an allocator/pool abstraction without measuring peak storage and teardown. Candidate 6 must preserve the current ordered schedule and all validation semantics. Candidate 7 has the largest identity and lifetime surface.

### Measured envelopes and priority

All bounds are conditional **active diagnostic** envelopes. No production removable fraction has been measured.

| Priority | Candidate | Decode / FFN envelope ms | Risk / change scope | Decision |
|---|---|---:|---|---|
| 1 | 4. Unnecessary report copies | 2.255 / 10.153 | Medium ownership risk; stripe-private executor/result handoff plus compatibility-preserving tests | D2 B; first bounded experiment |
| 2a | 1. Duplicate validation | 0.891 / 4.644 | Medium trust/invalidation risk; second validator only, public checks retained | Next candidate if B is unsuccessful or completed |
| 2b | 2. Validated internal path | Same 0.891 / 4.644, not additive | Medium contract/lifetime change; mechanism required for 2a | Do not introduce an unchecked public entry |
| 3 | 5. Allocation reuse | Gross exclusive alloc+free 3.255 / 13.268 | Medium/high retained-storage/reset risk; safely reusable subset Not measured | First measure scratch/live storage; no pool yet |
| 4 | 3. Immutable metadata sharing | At most the overlapping copy envelope, all copies 2.284 / 10.183 | Higher ownership machinery than the narrow report path; management cost Not measured | Defer broad sharing |
| 5 | 6. Minimal Plan construction | Gross exclusive construction 1.905 / 8.642 | High schedule/diagnostic-contract risk; includes required derivation/checks | No claim that the full envelope is removable |
| 6 | 7. Broader Plan reuse/cache | Route roots 6.892 / 29.687 include instrumentation and overlapping work | Highest identity, invalidation and memory risk; lookup/key/eviction costs Not measured | Not justified before narrower work |

The allocator envelope is larger than the copy envelope but is not a directly removable candidate. Report-copy elimination can remove only its own allocations; adding those savings to all allocator savings would double-count them. Likewise, the route-root envelope is not an extra saving on top of validation/construction/copy.

### Bounded Amdahl scenarios

For each active steady sample, let C be its measured graph compute time and b one selected nonoverlapping site envelope. If a fraction r of that envelope disappeared with no replacement work and no loss of useful overlap:

```text
conditional speedup = C / (C - r*b)
```

The table takes medians of the seven per-sample calculations. It uses active C, not unprofiled C. It never treats all Planning as removable.

| Selected envelope | Case | Median b/C | Remove half: conditional speedup | Remove all: diagnostic ceiling |
|---|---|---:|---:|---:|
| Second validator only | Decode | 3.93% | 1.0200x | 1.0409x |
| Second validator only | FFN | 4.14% | 1.0211x | 1.0431x |
| Two report assignments only | Decode | 9.94% | 1.0523x | 1.1103x |
| Two report assignments only | FFN | 9.04% | 1.0474x | 1.0994x |

The full-removal column is a deliberately favorable ceiling inside the perturbed diagnostic execution, not a forecast for production. Sharing/ownership handling, retained public report work, cache lookup and critical-path overlap can consume the benefit. Subtracting these diagnostic envelopes from 14.754 or 79.069 ms would mix incompatible measurements. Release-time component savings, achieved end-to-end speedup and any reduction in actual hardware waiting remain **Not measured**.

## 8. Measurement Methodology

### Scope and instrumentation

`POTAL_PLANNING_DIAGNOSTICS` is a CMake option, default **OFF**. ON requires GNU C++ on Linux. Only the planner, diagnostic collector and diagnostic smoke/test sources receive the compile definition. OFF has no collector library, allocator wrappers, diagnostic roots/scopes or smoke reset/dump calls.

ON uses a shared `potal-planning-profile` collector. This matters because the smoke executable compiles backend sources and also loads the registered backend DSO. A collector local to each would split TLS/reset/dump state. This is diagnostic linkage only; the production API and numerical/command paths are unchanged.

`POTAL_PLANNING_PROFILE=1` enables roots at runtime. Other values disable collection. The diagnostic binary with this variable unset is the **inactive** control; it still contains diagnostic code/wrappers and is not the production OFF binary.

There are two sequential roots per normal accepted stripe:

* **route**: entry to `potal_paired_plan_stripe`, after packet decode, through tile/execution construction and the builder's first full validation.
* **executor**: entry to `potal_paired_matmul_auto`, including report reset, request validation, supplied-plan match, second full validation and report copies. It stops before residual DMA staging or hardware issue.

The same thread's nested roots do not create a second total. Separate route/executor roots do not overlap. Across the pipeline, producer work may overlap these issuer intervals; adding roots does not prove an identical reduction in graph critical-path time.

Exclusive components partition each root exactly:

```text
root total =
    validation + copy + allocation + deallocation
    + construction + other + instrumentation

Planning denominator in the cost tables =
    root total - instrumentation
```

Allocation/deallocation nested inside a validation/copy/construction scope are charged to their own components, not twice. `other` covers unscoped control, setup/reset and moves inside the roots. `construction` includes metadata derivation and checks interleaved with fragment/operation generation; it is not a measured “validation-free” generator. `tile_frozen` likewise includes the existing helper's metadata derivation.

Sites report both inclusive and exclusive time. Site inclusive time includes nested work and collector bookkeeping. Site exclusive time also contains bookkeeping charged at that site, whereas the component's `instrumentation` bucket separates it. Therefore site tables are attribution/bounds, not another additive cost decomposition. In particular, adding builder, validator and allocator inclusive times would double-count work.

Collector bookkeeping and the post-copy payload inventory have a distinct measured component. Runtime environment gating, initial fixed-array zeroing and global merging lie outside the root clock but inside graph execution. Clock-call perturbation, changed layout/cache behavior and the inactive wrappers cannot all be calibrated away by subtracting one bucket. The active/inactive/OFF graph controls measure the net perturbation; no release-time component is inferred by subtracting them.

GNU link wrappers cover ordinary scalar/array `new`, unsized delete and sized delete:

```text
-Wl,--wrap=_Znwm -Wl,--wrap=_Znam
-Wl,--wrap=_ZdlPv -Wl,--wrap=_ZdaPv
-Wl,--wrap=_ZdlPvm -Wl,--wrap=_ZdaPvm
```

Each calls the real ABI function and preserves throwing allocation behavior. Coverage is only references resolved in that target: `malloc`, aligned overloads and allocations originating inside external DSOs are **Not measured**. Allocation bytes are actual requested bytes, not resident/high-water memory. Delete bytes are the size supplied to sized delete; unsized delete counts have unknown byte extent. Final post-execution Plan teardown and allocator peak live memory are **Not measured**.

The hardware-free accounting test verifies exclusive-sum equality, nesting, inclusive containment, counts/bytes, explicit scope/root finish, stale stack-address protection, disabled behavior, all six ABI wrappers, throwing allocation propagation, threaded aggregation, reset and parsed machine records. It uses no sleeps or timing-positivity assertions.

### Benchmark protocol

The existing `potal-paired-smoke` benchmark is reused unchanged apart from gated collection hooks. `reset()` occurs before its graph timer; `dump()` occurs after the timer. Per-command printing and legacy detailed Paired profiling stay disabled.

Cases are `decode_1x768x768` and `gpt2_ffn_up_16x3072x768`. Seed `20261023`, tensors, Golden, warmups, route order and seven repetitions are identical across OFF baseline, ON/inactive and ON/active processes. Each benchmark does route warmups, seven cold samples per route, another route warmup, then seven steady samples per route. Order rotates/reverses across Serial, Paired RTL and Paired Conservative.

Cold means prepared-weight-cache miss after issuer/route warmup, not a cold FPGA. Steady means one prepared-weight hit. Every graph result is checked bitwise against the independent F32 Golden. Route fallback, failed-node and semantic-completion checks remain enabled. The quick suite and an explicit existing capacity-fallback benchmark also exercise the preserved C boundary.

All reported times are guest `steady_clock` CPU-observed elapsed time. Graph compute includes software and hardware waiting; component timers measure CPU Planning sections. Neither is pure FPGA execution time. Medians and MAD are calculated across the seven actual samples. Component shares use the median of each sample's component/Planning ratio; component medians need not sum to the median of Planning.

Dump records are associated with the immediately following matching `POTAL_BENCH_SAMPLE`. Warmup dumps are not accidentally accumulated into a measured sample. Structural counts/bytes must be identical across repetitions; each phase's component and site exclusive sums must equal its recorded root total. Serial must emit zero Paired roots. Existing command-count fields are compared across all three variants by case, cold/steady phase, route and iteration.

### Environment, commands and artifacts

Every new shell sources the project environment:

```bash
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
cd generators/potal/software/workspace/llama.cpp-potal
```

The existing DIM16 Release configuration is reused. Compiler is `/home/alveo/firesim/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-g++`, GCC `12.2.0 (g2ee5e430018)`. Relevant flags are `-O3 -DNDEBUG -std=gnu++17 -fPIC -pthread`, the existing warning flags and the same generated DIM16 include/definition configuration. Activation is A8W8 ExSIA, block size 32, compensation ON, hardware backend, no OpenMP. No compiler, hardware parameter or numerical option was changed.

```bash
cmake -S . -B build-potal-paired-host -DPOTAL_PLANNING_DIAGNOSTICS=ON
cmake --build build-potal-paired-host -j 4 --target \
  potal-planning-profile-test potal-api-test potal-trace-test \
  potal-residual-unit-test potal-paired-plan-test potal-paired-schedule-test \
  potal-paired-unit-test potal-paired-auto-test potal-paired-execution-test \
  potal-stripe-pipeline-test

cmake -S . -B build-potal-h3-6a-d16 -DPOTAL_PLANNING_DIAGNOSTICS=ON
cmake --build build-potal-h3-6a-d16 -j 4 --target \
  potal-paired-smoke ggml-potal potal-planning-profile-test
```

The original OFF target executable/libraries were copied before rebuilding. Diagnostic artifacts were preserved before restoring both build directories to `-DPOTAL_PLANNING_DIAGNOSTICS=OFF`. The current-source OFF Host regressions and OFF target build were then rerun successfully.

One clean copy of the powered-off B4 guest rootfs was used at `/tmp/h3-6d1-run/rootfs.img`. The existing U280 driver and fixed image in `/home/alveo/FIRESIM_RUNS_DIR/sim_slot_0` were used without FPGA/RTL rebuild. The baseline and diagnostic executables/libraries coexist in `/root/h3-6d1/{baseline,diagnostic}`.

The complete guest command body is `/tmp/h3-6d1/guest-suite.sh`. Its mode loop unsets the legacy trace/profile/probe/window/stage diagnostics, selects each directory with `LD_LIBRARY_PATH`, and runs:

```bash
# In each of baseline, inactive and active modes:
# baseline uses original OFF artifacts; inactive uses ON artifacts, env unset;
# active uses those same ON artifacts with POTAL_PLANNING_PROFILE=1.
./potal-paired-smoke --bench --case decode_1x768x768 --repeat 7
./potal-paired-smoke --bench --case gpt2_ffn_up_16x3072x768 --repeat 7
./potal-paired-smoke --quick

# Active diagnostic mode, existing capacity-fallback case:
./potal-paired-smoke --bench --case dense_fallback_160x320x512 --repeat 1
```

Raw UART is `/tmp/h3-6d1-run/uartlog`. Host artifacts are under `/tmp/h3-6d1/{baseline,diagnostic,host-off,host-on}`; source-audit notes are `/tmp/h3-6d1/source-analysis.json`.

| Artifact | SHA-256 |
|---|---|
| Original OFF target smoke | `c694e07a3349929f6b3568cb3d6cc3867fbbd4323bed619482ca3f3a10c79c43` |
| Diagnostic ON target smoke | `2beb52467b9c0df6e7ea217cac9923bf65c2de63c9a25b6de7ccf899013193e7` |
| Diagnostic collector DSO | `b459120ba71a3600111b76c5d31218ed0c9c1d3817c22514e510cbcca879c71b` |

### OFF code-generation proof

The committed `paired-auto.cpp` and current diagnostics-OFF source were compiled separately with identical existing target definitions, includes and flags. All disassembled instruction sections were compared, using:

```bash
riscv64-unknown-linux-gnu-objdump -d --no-addresses --no-show-raw-insn FILE.o
```

Only the object filename/file-format header was excluded from hashing. Both normalized outputs have SHA-256 `0d5a89ef0c3658933b2fec39d1403395810445dd56deb047418e68d703f70a55`. The comparison emitted `D1_OFF_ZERO_CODE_OVERHEAD_PASS` with exit code 0. This proves identical planner-object instruction output for OFF, not byte identity of the entire executable. The restored OFF smoke/backend also have no `potal-planning-profile` dynamic dependency.

## 9. Regression Test Results

All nine existing Host executables passed on the preserved OFF baseline and the diagnostic ON build with `POTAL_PLANNING_PROFILE=1`. They also passed after rebuilding the current source with diagnostics OFF:

```text
potal-api-test
potal-trace-test
potal-residual-unit-test
potal-paired-plan-test
potal-paired-schedule-test
potal-paired-unit-test
potal-paired-auto-test
potal-paired-execution-test
potal-stripe-pipeline-test
```

These are actual executable runs, not an inference from absent CTest registrations. The execution/pipeline tests retain injected hardware-status failures, cleanup/fatal paths, command encodings/order/counts, generation behavior, independent numerical references and C's one-decode fallback checks.

After the ON build, the existing executable regression command was:

```bash
for test in potal-api-test potal-trace-test potal-residual-unit-test \
  potal-paired-plan-test potal-paired-schedule-test potal-paired-unit-test \
  potal-paired-auto-test potal-paired-execution-test potal-stripe-pipeline-test
do
  POTAL_PLANNING_PROFILE=1 "build-potal-paired-host/bin/$test" || exit 1
done
```

`potal-planning-profile-test` passed natively, through the shared-collector CMake build, and as a real RISC-V executable in the guest (`__H36D1_ACCOUNTING_TEST_PASS__`). The standalone native core also compiled with `-Wall -Wextra -Werror`.

The complete `potal-trace-test --dump` stream was identical between preserved OFF and ON/active (`D1_TRACE_STREAM_IDENTICAL`, exit code 0). `potal.c` itself remains unchanged, SHA-256 `7dec7c2a9cfc0bc12cb3dcdb9abd93302c0dc69e813a812f6d332c87e16850bd`. B2 trace formatting, disabled-trace dispatch, statuses, sticky errors, cleanup and synchronization were not modified.

### Actual FireSim results and equivalence

| Guest mode | Decode, seven cold + seven steady per route | FFN, seven cold + seven steady per route | Existing quick suite |
|---|---|---|---|
| Original OFF baseline | PASS | PASS | PASS |
| Diagnostic ON, runtime inactive | PASS | PASS | PASS |
| Diagnostic ON, runtime active | PASS | PASS | PASS |

The parser checked 252 samples: 3 modes x 2 cases x 2 phases x 3 routes x 7 repeats. Every `ok` is 1. Cold has exactly one prepared-weight miss and steady exactly one hit. All 168 inactive/active-to-baseline comparisons match every emitted command-count field, including fences, drains, fallback and semantic completion. All 36 printed compute summaries match recomputed median/MAD.

All 56 active Paired/Conservative measured graphs have exactly one route and one executor root. The 28 active Serial graphs have zero roots. Every root's exclusive component/site sums match exactly; logical counts/bytes match across cold/steady, repetitions and Conservative.

Canonical counts per graph below are unchanged across all three builds/modes. `loops` means Serial Main/Residual or Paired Main/Residual for the named route. Fence/drain columns are the existing software counters, not a decomposition of hardware wait time.

| Case / route | Loops Main/Residual | Index config/write | Scale write/release | Paired readbacks/tiles | Fences | Paired drains | Fallback/semantic stripes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decode / Serial | 6 / 4 | 0 / 0 | 33024 / 33024 | 0 / 0 | 6 Main + 4 Residual | 0 | 0 / 1 |
| Decode / Paired | 12 / 60 | 60 / 978 | 33024 / 33024 | 12 / 96 | 1 | 73 | 0 / 1 |
| Decode / Conservative | 12 / 60 | 60 / 978 | 33024 / 33024 | 12 / 96 | 145 | 217 | 0 / 1 |
| FFN / Serial | 22 / 22 | 0 / 0 | 147456 / 147456 | 0 / 0 | 22 Main + 22 Residual | 0 | 0 / 1 |
| FFN / Paired | 44 / 264 | 264 / 6314 | 147456 / 147456 | 44 / 384 | 1 | 309 | 0 / 1 |
| FFN / Conservative | 44 / 264 | 264 / 6314 | 147456 / 147456 | 44 / 384 | 617 | 925 | 0 / 1 |

The active existing `dense_fallback_160x320x512 --repeat 1` benchmark also passed all six cold/steady route samples. Each Paired route records two capacity-fallback stripes, two route roots and zero executor roots. Serial records zero Paired roots. This is a correctness/boundary check, not a seven-repeat performance claim. Existing Host regressions separately prove one decode per rejected stripe and public/decoded-lowering numerical equivalence.

Actual hardware equivalence here is Golden-exact output plus complete emitted count-field equivalence and unchanged command-emitting code. It is not a captured byte-for-byte stream of every hardware command's arguments. The complete C11 Trace/Sink dump comparison and command-level emulator regressions supply the additional encoding/status/error coverage.

Both diagnostic and restored-OFF DIM16 smoke/backend builds completed with exit code 0. Existing warnings remain in generated `POTAL_SCALE_ENTRIES`, shared Gemmini `__int128`, an ExSIA missing declaration and Token's unused variable. No warning suppression was added. The pre-existing `.conda-env` activation warning remains; the configured FireSim toolchain path works. LSP lacks the configured generated/include paths for the planner/test, so it reports missing `potal.h` or `matmul/planning-profile.hpp`; actual configured Host and RISC-V compilation passed. The diagnostic collector itself has no LSP errors.

Source-restriction audit passed: both requested `potal` branches and all three repository HEADs remain unchanged; `stripe.cpp`, `residual/stripe-residual.cpp` and `potal.c` have no diff. The pre-existing `models/gpt2/` directory was preserved. No RTL, numerical code, public API, validation, copy/ownership behavior, command ordering, cache or Scale optimization was implemented.

Protected pre-existing changes are unchanged:

| Protected object | Verified identity |
|---|---|
| PoTal `software/host-init.sh` | SHA-256 `e96d846238946925fd4a1c52f9b5e24ea144e15b3d748d15f8420e1c71c70834` |
| Gemmini submodule HEAD | `185293e87cbbc2a336410f49a3fd73d01ed8d04e` |
| Gemmini existing binary diff | SHA-256 `407b6271652f19485acaa919b81e5fb697083ac73a894d7b217de883c9e71b84` |

`git diff --check` passed. No new branch, commit or push was made.

The final guest markers are `__H36D1_SUITE_PASS__` and `__H36D1_SUITE_EXIT__=0`. After `sync; poweroff`, the filesystem remounted read-only, Linux printed `reboot: Power down`, and the simulator printed `*** PASSED ***`. The driver/script exited 0; UART ends with `COMMAND_EXIT_CODE="0"`. No started guest remains running.

## 10. H3-6D2 Recommendation

Choose **B: copies/ownership**, limited first to the stripe-private report-copy path. This is a recommendation, not an implemented optimization or permission to start D2.

The evidence favors that narrow experiment over immediately removing validation: its diagnostic envelope is approximately 2.5 times the second-validator envelope for Decode and 2.2 times for FFN. The duplicated payload and existing stripe-local owner are concrete. A broad allocator/cache design has a larger unmeasured correctness, memory and invalidation surface.

A suitable future D2 acceptance contract would be:

1. Prove that the same immutable `routing` Plan outlives executor issue, drain, result use and reconstruction. Define a private result/view boundary; preserve the existing owned public report behavior.
2. Target only `report.plan = *req.plan` and `report.tile = req.plan->tile` on that private path. Do not remove either full validation, change numerical/command behavior, or introduce a cross-packet cache/pool.
3. Retain the Host malformed-input, Paired, residual, pipeline, trace, injected-error/cleanup and independent numerical tests. Add only a repository-sized lifetime/copy-path regression if the changed behavior could otherwise escape those tests.
4. Use code-generation/count evidence and matched **diagnostics-OFF** seven-repeat Decode/FFN controls to establish actual benefit beyond dispersion. The current active timing envelopes are not a pass threshold or promised saving.
5. Reject an experiment that cannot preserve public ownership/error contracts or show unprofiled benefit. Reassess A as a separate validated-internal-handoff project rather than combining speculative optimizations.

D1 leaves the following limits explicit: intrusive diagnostic timing; target-resolved ordinary allocator coverage only; unmeasured final teardown/peak storage/per-copy allocator split; no achieved speedup measurement; and static pre-existing cross-packet supplied-plan validation/alias hazards. None is silently converted to zero cost or a valid general reuse contract.

Changed files are limited to:

```text
H3-6D1-paired-planning-validation-analysis.md
ggml/src/ggml-potal/CMakeLists.txt
ggml/src/ggml-potal/matmul/paired-auto.cpp
ggml/src/ggml-potal/matmul/planning-profile.hpp
ggml/src/ggml-potal/matmul/planning-profile.cpp
ggml/src/ggml-potal/tests/potal-paired-smoke.cpp
ggml/src/ggml-potal/tests/potal-planning-profile-test.cpp
```

Both working build configurations are restored to diagnostics OFF. Diagnostic artifacts remain separately preserved for reproducing this analysis. Stop at D1 measurement, verification and this report; no D2 change is included.
