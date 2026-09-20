# Cycle Logging and Timeline Analysis

This document describes the cycle instrumentation used by the Gemmini backend, the JSONL records written by the runtime, and the offline timeline workflow implemented by `scripts/utils/cycle_timeline.py`.

The default experiment configuration discussed here is:

```text
LOG_CYCLE=1
CYCLE_DETAIL=0
GGML_CPU_CYCLE_LOG=1
```

The design goal is to preserve small raw intervals during execution and postpone aggregation until offline analysis. The runtime does not pre-assemble operator totals or a hypothetical pipeline. The raw log keeps enough timing and identity information to reconstruct multiple views later.

## 1. Measurement model

Three time domains are intentionally kept separate.

| Domain | Meaning | Used for |
| --- | --- | --- |
| CPU cycle counter | Per-thread CPU work counter | CPU cost of one same-thread interval |
| Host monotonic nanoseconds | Shared `steady_clock` timeline | Ordering, elapsed wall time, overlap between host threads |
| Device/RTL cycles | Accelerator-local counter | NPU/RTL work inside its own clock domain |

The important interpretation rule is:

```text
host ns       = when the interval happened and how intervals overlapped
CPU cycles    = how much CPU work the measured thread consumed
device cycles = accelerator-local work
```

CPU cycles from different threads are not a common absolute timeline. Device cycles are also not directly comparable to CPU cycles. Do not add CPU and device cycle counts together.

### 1.1 Endpoint sampling

With `CYCLE_DETAIL=0`, a CPU endpoint sample contains:

- native CPU cycle sample,
- shared monotonic `ns`,
- OS thread ID,
- trace/context identity.

With `CYCLE_DETAIL=1`, the same endpoint additionally collects thread CPU time and the detailed counter provenance used by the diagnostic records.

The cycle and monotonic clock are read sequentially inside the endpoint sampling path; they are not one physically simultaneous hardware sample. Treat them as paired observations with different purposes rather than using `cycles / ns` as an exact instantaneous CPU frequency measurement.

Both the generic CPU path (`gemmini_cpu_timing_read()`) and the Gemmini matmul path (`read_matmul_cpu_sample()`) retain the monotonic timestamp in compact mode.

## 2. Build options

### `LOG_CYCLE`

Enables the cycle logging infrastructure.

```bash
-DLOG_CYCLE=1
```

If `LOG_CYCLE=0`, cycle logging is disabled.

### `GGML_CPU_CYCLE_LOG`

Enables generic `ggml-cpu` operator instrumentation. This is separate from the Gemmini logger because the CPU backend is built independently.

```bash
-DGGML_CPU_CYCLE_LOG=ON
```

`build-arm64.sh` defaults this option to the value of `LOG_CYCLE`.

### `CYCLE_DETAIL`

Controls expensive diagnostic detail.

```bash
-DCYCLE_DETAIL=0
```

is the normal experiment mode. It keeps raw cycle intervals and the shared ns timeline but avoids the large nested timing/provenance structures and deep profiling outputs.

```bash
-DCYCLE_DETAIL=1
```

enables detailed diagnostics. It requires `LOG_CYCLE=1`.

The ARM64 build script currently defaults to:

```text
LOG_CYCLE=1
GGML_CPU_CYCLE_LOG=1
CYCLE_DETAIL=0
```

unless the corresponding environment variables override them.

## 3. Cycle log location

The logical default cycle path is:

```text
log/cycle-log.jsonl
```

Relative log paths are resolved as follows:

1. If `GEMMINI_LOG_DIR` is set, the path is placed under that directory. A leading `log/` component is stripped.
2. Otherwise the path is placed below `$PWD/output/log/`.

Therefore the physical path depends on the working directory of the inference process.

For example, if `llama-cli` is launched while the current directory is `build-arm64/bin`, the default file is:

```text
build-arm64/bin/output/log/cycle-log.jsonl
```

For experiments where the launch directory may change, setting an explicit directory is clearer:

```bash
export GEMMINI_LOG_DIR="$PWD/output/log"
```

The detail-only ExSIA profile stream uses the logical default:

```text
log/exsia-cycle-detail.jsonl
```

## 4. Compact raw interval format

With `LOG_CYCLE=1,CYCLE_DETAIL=0`, high-frequency interval records are deliberately compact. They do not repeat `schema`, `version`, and `record_type` on every line.

A valid ARM64 CPU row has the following shape:

```json
{
  "op": "cpu.add",
  "kind": "cpu",
  "layer": "blk.0",
  "run_id": 12,
  "node_id": 7,
  "worker_id": 1,
  "start": 128314,
  "end": 129004,
  "delta": 690,
  "ns_start": 382910234001,
  "ns_end": 382910287552,
  "tid": 1742,
  "valid": true,
  "operator_context": {
    "operator_id": 91,
    "graph_id": 80,
    "operator_kind": "ADD",
    "task_id": 104,
    "parent_task_id": 97,
    "segment_id": 110,
    "parent_segment_id": 104,
    "role": "operator",
    "scope": "owner_segment"
  },
  "inference_context": {
    "request_id": 1,
    "operation_id": 2,
    "phase": "decode",
    "included": true
  }
}
```

Not every identity field is present on every row. Missing optional identities are omitted rather than emitted as repeated `null` fields.

### 4.1 `kind`

Compact intervals use three explicit kinds:

| `kind` | Meaning |
| --- | --- |
| `cpu` | Counted same-thread CPU interval; participates in the CPU interval cardinality contract when an inference operation is active |
| `segment` | Raw structural/task/operator span; preserved for hierarchy/timeline analysis but not counted as another canonical CPU sample |
| `cycle` | Checked scalar cycle interval used by Gemmini/matmul lifecycle instrumentation |

The three kinds are not automatically merged.

### 4.2 Cycle fields

`start` and `end` are endpoints in the CPU counter domain. For a valid same-thread interval:

```text
delta = end - start
valid = true
```

For an invalid or unavailable CPU counter interval:

```json
{
  "delta": null,
  "valid": false,
  "reason": "..."
}
```

A valid zero delta is different from an invalid sample. Zero is preserved as a real measured value.

### 4.3 Shared timeline fields

```text
ns_start
ns_end
```

are `steady_clock` timestamps in one host-wide monotonic domain. They are the coordinates used to reconstruct the observed host timeline.

For a normal same-thread interval:

```text
tid = <OS TID>
```

If the endpoints do not belong to the same thread, the compact representation uses `tid_start` and `tid_end` instead. A cross-thread interval cannot publish a valid CPU cycle delta.

### 4.4 Identity fields

The raw records retain identities needed for later assembly:

```text
request_id / operation_id / phase
graph_id / operator_id / operator_kind
task_id / parent_task_id
segment_id / parent_segment_id
matmul_invocation_id
run_id / stripe_id / slot
node_id / worker_id
```

These identities allow the same raw data to be viewed by thread, operator, task, or pipeline stripe without changing the measured endpoints.

## 5. What is measured

### 5.1 Generic ggml CPU operators

The generic CPU backend has a source-level coverage contract for exactly 81 operator labels. Each covered operation has a native start endpoint, the operation body, a native end endpoint, and a checked log publication.

<details>
<summary>81 generic CPU labels</summary>

```text
cpu.dup
cpu.add
cpu.add1
cpu.acc
cpu.sub
cpu.mul
cpu.div
cpu.sqr
cpu.sqrt
cpu.log
cpu.sin
cpu.cos
cpu.sum
cpu.sum_rows
cpu.mean
cpu.argmax
cpu.count_equal
cpu.repeat
cpu.repeat_back
cpu.concat
cpu.silu_back
cpu.norm
cpu.rms_norm
cpu.rms_norm_back
cpu.group_norm
cpu.l2_norm
cpu.mul_mat
cpu.mul_mat_id
cpu.out_prod
cpu.scale
cpu.set
cpu.cpy
cpu.cont
cpu.reshape
cpu.view
cpu.permute
cpu.transpose
cpu.get_rows
cpu.get_rows_back
cpu.diag
cpu.diag_mask_inf
cpu.diag_mask_zero
cpu.softmax
cpu.softmax_back
cpu.rope
cpu.rope_back
cpu.clamp
cpu.conv_transpose_1d
cpu.im2col
cpu.im2col_back
cpu.conv_2d_dw
cpu.conv_transpose_2d
cpu.pool_1d
cpu.pool_2d
cpu.pool_2d_back
cpu.upscale
cpu.pad
cpu.pad_reflect_1d
cpu.arange
cpu.timestep_embedding
cpu.argsort
cpu.leaky_relu
cpu.flash_attn_ext
cpu.flash_attn_back
cpu.ssm_conv
cpu.ssm_scan
cpu.win_part
cpu.win_unpart
cpu.unary
cpu.get_rel_pos
cpu.add_rel_pos
cpu.rwkv_wkv6
cpu.gated_linear_attn
cpu.rwkv_wkv7
cpu.map_custom1
cpu.map_custom2
cpu.map_custom3
cpu.custom
cpu.cross_entropy_loss
cpu.cross_entropy_loss_back
cpu.opt_step_adamw
```

</details>

Worker and operator structural spans are also retained, including graph worker lifetime, operator dispatch, task lifetime, and explicit synchronization spans such as `barrier.wait`.

Structural envelopes are declared by the call site. The logger does not infer that a record is an envelope from the spelling of `op`.

### 5.2 Gemmini outer host stages

The Gemmini host path has a contract for these nine outer CPU intervals:

```text
gemmini.prepare_args
gemmini.select_tile
gemmini.activation_buffer_preparation
gemmini.quantize_activation
gemmini.prepare_dense_i8_weight
gemmini.convert_q4_0_to_q4_h1
gemmini.convert_q8_0_to_q8_h1
gemmini.prepare_weight
gemmini.output_preparation
```

These intervals retain their measured cycle/ns endpoints and matmul/run identity.

### 5.3 IM2P host boundaries

The IM2P integration preserves route-specific host boundaries around real call sites. Core labels covered by the source contract include:

```text
im2p.host_input_preparation
im2p.stripe_input_capture
im2p.stripe_submit_host_call
im2p.frontend_start_host_call
im2p.fence_host_call
im2p.residual_metadata_preparation
im2p.residual_backend_host_call
im2p.residual_simulator_host_call
im2p.output_correction_apply
im2p.post_fence_validation
im2p.output_authorize_host_call
im2p.output_buffer_copy
```

Route-specific helpers can add more explicit labels, for example residual simulator setup or workload observation.

Whether a host interval is eligible for CPU-work wall accounting is also explicit at its call site through `HostIntervalAccounting`. The implementation does not classify CPU work by matching the `op` string.

The external IM2P frontend can report additional same-thread host stages through the frontend timing callback. These records inherit the executing worker context and are published as raw segments.

### 5.4 Matmul/stripe lifecycle

The matmul path retains raw boundaries for the real lifecycle work rather than inventing cross-task cycle subtraction. Examples include:

- Dense facade/backend host call,
- collector input capture,
- worker-side job preparation,
- residual backend execution,
- Merge,
- Finalize,
- output transaction copy,
- worker/task lifetime and synchronization.

The exact boundary contracts are guarded by `test-gemmini-matmul-cpu-boundaries.cmake`.

An outer interval and an inner interval can both be present. This is intentional containment, not automatically a duplicate. Offline analysis must not sum nested intervals unless the intended aggregation explicitly accounts for inclusion.

### 5.5 Residual CPU-direct and RMD stages

CPU-direct residual execution keeps four coarse phases:

```text
rmd.cpu_direct.validation
rmd.cpu_direct.preparation
rmd.cpu_direct.parallel
rmd.cpu_direct.finalization
```

J tiles retain a separate raw interval:

```text
rmd_direct_j_tile_interval
```

with run/stripe/worker/node identity.

The normal RMD execution path also retains stage-level intervals:

```text
rmd.preparation
rmd.weight_gather
rmd.block_scale_metadata
rmd.dot_output_accumulate
rmd.block_scale_apply
rmd.radix_reconstruct_combine
rmd.final_metadata
rmd.final_scale_combine_stage
rmd.output_store
```

With detail profiling enabled, CPU-direct tiles can additionally expose:

```text
rmd.cpu_direct.event_scan
rmd.cpu_direct.weight_dot
rmd.cpu_direct.scale_apply
```

### 5.6 Inference boundaries

`INFERENCE_EVENT` records provide the request-level host timeline:

```text
session_start
session_end
request_start
operation_start
operation_end
token_ready
request_end
configuration
```

These timestamps are used for request latency, prefill/decode boundaries, TTFT/TPOT reconstruction, and CPU interval cardinality checks.

The model/configuration record is emitted as `INFERENCE_CONFIGURATION`. Among other experiment metadata it includes:

```text
cycle_detail
interval_format
cpu_cycle_source
cpu_cycle_unit
timeline_clock
timeline_unit
CPU pool settings
affinity observations
CPU frequency policy observations
model/build identity
```

### 5.7 Device/RTL counters

Accelerator telemetry remains in its own clock domain.

Important record types include:

```text
IM2P_EXECUTION_TELEMETRY
IM2P_STRIPE_TELEMETRY
IM2P_RMD_EXECUTION_TELEMETRY
IM2P_RMD_STRIPE_TELEMETRY
NPU_OPERATOR_SEGMENT
WS_LOOP_TELEMETRY
```

For an IM2P stripe, the raw device interval contains:

```text
run_id
stripe_id
slot
row_begin / row_end
publish_cycle
completion_cycle
latency_cycles
```

The run-level IM2P record carries `rtl_stripes_published` and `rtl_work_total_cycles` plus backend-dependent diagnostic counters.

A device-local `publish_cycle/completion_cycle` pair is not placed on the host ns timeline unless a real host-ns boundary exists. `cycle_timeline.py` therefore keeps these rows in normalized output but deliberately leaves them out of a Chrome trace when no host ns coordinates exist.

## 6. Other JSONL records

The main cycle JSONL can contain records that are useful for configuration, diagnostics, or aggregate reporting but are not themselves raw host spans.

| Record | Purpose |
| --- | --- |
| `INFERENCE_CONFIGURATION` | Build, model, thread-pool, environment, timing-mode configuration |
| `INFERENCE_EVENT` | Request/operation/token host-ns boundaries |
| `MATMUL_CONFIGURATION` | Matmul geometry/backend/configuration for an invocation |
| `RMD_BACKEND_TELEMETRY` | RMD route, geometry, invocation/stage counters |
| `RMD_STRIPE_TELEMETRY` | Per-stripe RMD metrics and host-stage cycle observations |
| `IM2P_EXECUTION_TELEMETRY` | Run-level main accelerator counters |
| `IM2P_STRIPE_TELEMETRY` | Per-stripe device-local cycle interval |
| `IM2P_RMD_EXECUTION_TELEMETRY` | Run-level residual accelerator counters |
| `IM2P_RMD_STRIPE_TELEMETRY` | Per-stripe residual accelerator counters |
| `WS_LOOP_TELEMETRY` | Gemmini WS loop occupancy/work counters |
| `FINAL_INFERENCE_SUMMARY` | End-of-run inference summary |

`cycle_timeline.py` counts every input JSONL record, but only normalizes the record families required for the raw host/device timeline and its cardinality checks. Unsupported diagnostic records remain untouched in the original log.

## 7. What changes with CYCLE_DETAIL=1

`CYCLE_DETAIL=1` is for diagnosis, not the default performance run.

It retains the full schema and may add:

- nested `native_cycles` endpoint/provenance information,
- nested `host_timing`,
- `thread_cpu_timing`,
- CPU worker summaries/resource samples,
- `RESIDUAL_HOST_PROFILE`,
- `PIPELINE_STRIPE_SUMMARY`,
- `QUANTIZATION_STRIPE_TELEMETRY`,
- ExSIA `EXSIA_RUN_SUMMARY`,
- ExSIA `TIMELINE`, `STAGE`, and `EXSIA_WORKLOAD` records,
- hashes and deeper route-specific diagnostics where enabled.

This mode produces more records and performs more clock/profiling work. Use `CYCLE_DETAIL=0` for the normal cycle/timeline experiment unless the extra attribution is specifically required.

## 8. No name-based semantic inference

The logging path does not infer execution meaning by searching the `op` string for words such as `wait`, `copy`, `submit`, or `rmd`.

Two semantics that affect structure/accounting are explicit:

- structural envelope versus ordinary segment is declared by the measurement call site,
- IM2P CPU-work wall accounting is declared by the `HostIntervalAccounting` argument at the call site.

The offline timeline tool also does not classify an operation by substring matching. It uses recorded IDs only.

This matters because names are labels; they are not a reliable source of scheduling semantics.

## 9. cycle_timeline.py

The timeline tool validates the log, writes normalized raw rows, and renders Chrome-trace views without aggregating the measured intervals.

### 9.1 Simplest command

From the repository root:

```bash
python3 scripts/utils/cycle_timeline.py /path/to/cycle-log.jsonl
```

No other option is required for a normal Nano experiment.

By default this creates files next to the input:

```text
cycle-log.timeline.jsonl
cycle-log.timeline.thread.chrome.json
cycle-log.timeline.operator.chrome.json
```

The default validation is strict: an invalid counted CPU interval causes a non-zero validation result.

### 9.2 Normalized raw JSONL

`cycle-log.timeline.jsonl` keeps one normalized output row per supported raw record. It does not combine nested intervals or sum operators.

Useful normalized fields include:

```text
row_type
source_line
kind
op / layer
request_id / operation_id / phase
graph_id / operator_id / operator_kind
task_id / parent_task_id
segment_id / parent_segment_id
matmul_invocation_id
run_id / stripe_id / slot / node_id / worker_id
tid / tid_start / tid_end
ns_start / ns_end / wall_ns
cycle_clock / cycle_start / cycle_end / cycles
cycle_valid / cycle_reason
```

This file is the intended input for future custom assembly/aggregation code.

### 9.3 Timeline views

A view changes the lane grouping only. It does not change the raw timestamps, cycle deltas, or source intervals.

#### Thread view

```bash
python3 scripts/utils/cycle_timeline.py cycle-log.jsonl --view thread
```

Groups spans by OS thread ID. This is the most direct view of actual host execution and overlap.

Output:

```text
cycle-log.timeline.thread.chrome.json
```

#### Operator view

```bash
python3 scripts/utils/cycle_timeline.py cycle-log.jsonl --view operator
```

Groups the same raw host spans by `operator_id`. Multiple worker/task spans belonging to one operator appear on the same logical operator lane.

Output:

```text
cycle-log.timeline.operator.chrome.json
```

No operator is inferred from `op`. A span without `operator_id` is not silently assigned to an operator; it is counted in `skipped_missing_lane` in the command summary.

#### Task view

```bash
python3 scripts/utils/cycle_timeline.py cycle-log.jsonl --view task
```

Groups spans by `task_id`.

Output:

```text
cycle-log.timeline.task.chrome.json
```

#### Stripe view

```bash
python3 scripts/utils/cycle_timeline.py cycle-log.jsonl --view stripe
```

Groups host spans by `run_id + stripe_id + layer`.

Output:

```text
cycle-log.timeline.stripe.chrome.json
```

Device rows that have only device-local cycles and no host ns coordinates remain unplaced.

### 9.4 Multiple views

`--view` is repeatable:

```bash
python3 scripts/utils/cycle_timeline.py cycle-log.jsonl \
    --view thread \
    --view operator \
    --view stripe
```

To generate every supported view:

```bash
python3 scripts/utils/cycle_timeline.py cycle-log.jsonl --all-views
```

This produces:

```text
thread
operator
task
stripe
```

Chrome trace files.

### 9.5 Options

| Option | Default | Meaning |
| --- | --- | --- |
| `input` | required | Source cycle JSONL |
| `--rows PATH` | `<input>.timeline.jsonl` | Override normalized-row output |
| `--trace-dir DIR` | input directory | Directory for generated Chrome traces |
| `--view VIEW` | `thread` + `operator` | Select one or more lane views; repeatable |
| `--all-views` | off | Generate thread/operator/task/stripe views |
| `--included-only` | off | Hide context-free warmup/raw spans from the Chrome trace |
| `--allow-invalid-cycles` | off | Allow logs from hosts without a usable native CPU counter, while still validating ns/structure |

For example, a local macOS log does not provide the Linux/AArch64 perf CPU-cycle source. It can still be inspected with:

```bash
python3 scripts/utils/cycle_timeline.py cycle-log.jsonl --allow-invalid-cycles
```

For a Nano experiment, leave strict validation enabled.

## 10. Hole detection

The timeline command performs validation before reporting the run as clean.

### CPU interval cardinality

During an inference operation, counted CPU intervals receive `cpu_interval_sequence` values. The matching `operation_end` record publishes `cpu_interval_samples`.

The tool verifies that the observed sequence is exactly:

```text
1 ... cpu_interval_samples
```

with no dropped or duplicated counted interval.

Old logs that do not contain the count contract are reported as unverified rather than being called complete.

### IM2P stripe cardinality

`IM2P_EXECUTION_TELEMETRY.rtl_stripes_published` is compared with the observed `IM2P_STRIPE_TELEMETRY.stripe_id` set.

A run that claims N stripes must contain exactly:

```text
0 ... N-1
```

for that run/layer identity.

### Structural identity validation

The tool also rejects structural contradictions such as:

- one `segment_id` reused by different owners,
- one `task_id` crossing operator identity,
- task self-parenting,
- segment self-parenting,
- malformed compact cycle arithmetic,
- shared ns timeline regression,
- valid CPU cycle intervals crossing thread identity.

### Exit status

```text
0 = validated successfully
1 = malformed input/schema/timestamp/counter record
2 = structurally readable log with a detected coverage/cycle/identity hole
```

The command prints a JSON summary containing the coverage state and per-view placement counts.

## 11. Observed timeline versus reconstructed pipeline

The Chrome trace is an observed host timeline:

```text
ns_start/ns_end -> actual host ordering and overlap
```

It should not be confused with a hypothetical optimized schedule.

A future pipeline assembler can use the normalized rows as inputs:

```text
measured CPU stage cost -> cycles
observed placement       -> ns_start/ns_end
dependency structure     -> task/segment/run/stripe identities
accelerator cost         -> device-local cycles
```

and then apply a chosen pipeline/scheduling model.

For example, the same raw data can later be assembled under:

- sequential execution,
- CPU/NPU overlap,
- stripe-pipelined execution,
- a fixed worker-count model,
- alternative residual/Dense overlap assumptions.

The observed ns trace remains the ground truth for what the current implementation actually did. A reconstructed schedule must be labeled as a model/assumption and must not overwrite the raw observations.

## 12. Practical Nano workflow

Build using the normal ARM64 script with compact cycle logging:

```bash
LOG_CYCLE=1 CYCLE_DETAIL=0 GGML_CPU_CYCLE_LOG=1 ./build-arm64.sh
```

Run the experiment. For a reproducible log directory, set:

```bash
export GEMMINI_LOG_DIR=/absolute/path/to/experiment-log
```

After inference:

```bash
python3 scripts/utils/cycle_timeline.py \
    /absolute/path/to/experiment-log/cycle-log.jsonl
```

For all grouping views:

```bash
python3 scripts/utils/cycle_timeline.py \
    /absolute/path/to/experiment-log/cycle-log.jsonl \
    --all-views
```

For an operator-focused trace only:

```bash
python3 scripts/utils/cycle_timeline.py \
    /absolute/path/to/experiment-log/cycle-log.jsonl \
    --view operator
```

The generated `*.chrome.json` files can be loaded by a Chrome-trace-compatible viewer such as Perfetto.

## 13. Invariants to preserve when adding instrumentation

When adding a new interval:

1. Measure the real same-thread operation boundary.
2. Read start and end from the same endpoint API; do not invent cross-thread cycle subtraction.
3. Keep a shared monotonic ns start/end for host timeline placement.
4. Attach real identity from the executing context; do not synthesize run/stripe/operator identity.
5. If the interval is a structural envelope, declare that explicitly through the measurement API.
6. If an IM2P host interval is eligible for CPU-work wall accounting, declare that explicitly at the call site.
7. Do not classify semantics from `op` string substrings.
8. Keep device cycles in their device clock domain.
9. Preserve valid zero values.
10. Add or update a coverage/cardinality contract so a missing interval is detectable.
