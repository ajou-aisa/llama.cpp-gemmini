# PoTal CPU preparation and reconstruction measurements

This measurement path is shared by W4A4 and W8A8. It changes instrumentation,
not quantization, rounding, residual selection, GEMM ordering, or scheduling.
Use a separate instrumented run; clock reads and trace storage have overhead.

## Enable and collect

Keep the existing board-specific SDK, hardware parameters and build options.
For Linear P0/P1/P2/P3 diagnostics, additionally configure
`-DLOG_CYCLE=1 -DCYCLE_DETAIL=1 -DGGML_GEMMINI_EXSIA_PROFILE_SCOPE=STAGE`.
The strict two-Local-worker / one-Folding-worker path now populates its profile
intervals and local-task statistics instead of leaving the OpenMP-only profile
records empty. Its mode is logged as `StrictRoles`.

For CPU-time and attention breakdowns, set these before starting the process:

```sh
export POTAL_TIMELINE=/absolute/output/timeline.csv
export POTAL_TIMELINE_CPU=1
export POTAL_TIMELINE_PROFILE=coarse
export POTAL_TIMELINE_REUSE=1
export GEMMINI_LOG_DIR=/absolute/output
```

Create the output directory first. Preserve the normal five-role CPU mapping.
The runtime does not change affinity on behalf of the application. `coarse`
avoids detailed RoCC issue-window tracing. Command windows use counter-only
sampling when enabled, without CPU clock syscalls. Increase `POTAL_TIMELINE_EVENTS`,
`POTAL_TIMELINE_ISSUER_EVENTS` and `POTAL_TIMELINE_ANCHORS` if the trace reports
overflow. The sidecar reserves additional memory per event slot.

For a 512-token prefill followed by eight single-token decode steps:

```sh
build/bin/potal-e2e-runner --model MODEL.gguf --out /absolute/output \
  --prompt-tokens 512 --gen-tokens 9 --trace-prefill 1 \
  --trace-steps 1,2,3,4,5,6,7,8 --check-logits 1
python3 scripts/potal-timeline/cpu_parse.py \
  /absolute/output/timeline.csv.cpu.csv /absolute/output/cpu-spans.csv
```

`--gen-tokens 9` includes the first token sampled after prefill; TPOT uses the
median of the remaining eight decode steps. Model boundary CPU samples stop
after `llama_synchronize`, before sampling. Thus their denominator is model
evaluation CPU time, not the sampling-inclusive TTFT/TPOT interval. Model
loading is outside that denominator. `--trace-steps` controls coverage; do not
use a partially traced run as a full-run CPU breakdown.

Validate the main timeline with the existing timeline parser as well: the
sidecar alone cannot detect an entire missing pair. Reject dropped events,
ring/anchor overflow, rejected registrations, missing model boundaries, and
failed inference. Preserve both CSVs and the build/run settings together.

## Fields and accounting

`timeline.csv.cpu.csv` records BEGIN/END samples keyed by thread slot and
event sequence, with run/layer/stripe/tile identities. `cpu_parse.py` pairs
them into **inclusive** intervals. It rejects unmatched boundaries and
non-increasing sequences; invalid counters remain blank rather than zero.

| Field | Meaning |
|---|---|
| `wall_ns` | CLOCK_MONOTONIC elapsed time, including waits |
| `thread_ns` | CLOCK_THREAD_CPUTIME_ID CPU time for that thread |
| `process_ns` | getrusage user+system CPU time, only at model boundaries 81/82/83 |
| `cycles`, `instructions` | RV64 rdcycle/rdinstret deltas on a verified fixed CPU |
| `ipc` | Retired instructions / cycles only when both counters are valid |
| `measurement_kind` | Descriptor CPU work, mixed command API, or inclusive parent scope |

Use the sum of **non-overlapping leaf** `thread_ns` intervals as the numerator
and the matching prefill/decode `process_ns` as the denominator for CPU work
fractions. Sum distinct Local workers because that metric measures CPU work.
For elapsed latency, use wall intervals and their union/critical path instead;
parallel worker durations must not be added to TTFT.

| Stage | Scope and limitation |
|---|---|
| 11 | Linear Local task CPU work; sum tasks, not the Local envelope |
| 13 / 14 / 15 / 16 | Linear mask assembly / anchor reduction / Folding / packet sealing |
| 42 / 43 | Linear INT64 reconstruction / dequantization and output merge |
| 110 / 111 | K / V stationary quantization **plus limb creation and packing** |
| 112 | Q input reads and Local adaptation |
| 113 | P input reads, without quantization |
| 114 / 115 | Q folding and digit creation / P quantization and digit creation |
| 116 | Attention output scale preparation |
| 117 | QK reconstruction and dequantization in their existing fused loop |
| 118 / 119 | PV INT64 reconstruction / floating conversion and output stores |
| 120 | Attention plan generation and staging; exclude from pure quantization |

Stages 75/76/77/79 are inclusive attention parents. Do not add them to their
110-120 children. Similarly, the StrictRoles Local envelope includes join and
scheduling time; its wall time is not additive Local CPU work. Linear Folding
includes residual generation. These labels deliberately do not claim that
all preparation is pure quantization or that all finalization is float-only
dequantization. QK's fused loop cannot be split further by timing its boundary
without changing execution.

## FPGA cycles and validity

On RV64, enable the sidecar only when the target OS permits user-mode
`rdcycle` and `rdinstret`. Counters are read only for workers whose registration
verified single-CPU affinity. Endpoint CPU IDs must match the assigned CPU.
Do not change a worker's affinity during a run. These are **raw hart counters**,
not per-thread PMU counters: interrupts and context switches can contribute,
and Rocket WFI behavior makes them unsuitable as an elapsed-time clock.
Keep CLOCK_MONOTONIC and thread CPU time alongside them.

On ARM/x86 the sidecar reports CPU clocks but marks cycle/instruction counters
unavailable. It never substitutes nanoseconds or the ARM generic timer for CPU
cycles. Native ARM PMU measurements require the existing separately validated
PMU path; do not compare unavailable sidecar counts with RV64 counts.

The e2e runner also rejects caller cycle deltas unless both endpoints have
single-CPU affinity, matching CPUs, and non-regressing nonzero counters.
`steps.csv` appends `cycles_valid`; invalid `cycles` is blank, and the summary
reports `NA` (including TPOT if any included decode sample is invalid).
Raw endpoint values remain available for auditing. This prevents subtracting
counters from different harts, as happened in the earlier FPGA log.

Do not assume a 1 GHz target or infer Rocket latency from ARM IPC alone.
For an interval with valid instruction and cycle counts,
`IPC = instructions / cycles`; conversion to time additionally needs the
verified target frequency and counter behavior. Different binaries, stalls,
cache misses, interrupts, and waits prevent a theoretical operator count from
being a measured instruction count.

Instrumentation is off by default. Keep a matching uninstrumented run for
latency and output comparison. Host-emulated tests validate boundaries and
numerical regressions; they do not certify new FPGA performance or a bitstream.

## Issuer counters for the next run

To retain the normal windowed Scale path while collecting issuer counters, use
`POTAL_TIMELINE_CPU=1` and **unset** `POTAL_TIMELINE_PROFILE`. `coarse` omits
command spans; `detail` can select scalar diagnostic command paths instead of
the production windowed path. Preserve the existing issuer affinity and SDK.

| Stages | Measurement |
|---|---|
| 121 / 122 | Main / residual GEMM descriptor construction, before the SDK call |
| 123 / 124 / 125 | Index / readback / Scale descriptor construction |
| 126 | Windowed Scale columns API, including possible RoCC admission stalls |
| 127 | Scalar Index entry loop, including command submission stalls |
| 26-39, 45-49 | Existing command/API and inclusive issuer scopes, now also sampled |

These samples read RV64 `rdcycle` and `rdinstret` only. They add no clock or
scheduler syscall and no CPU/RoCC fence inside the command window. Main,
residual, readback and Scale descriptors end before their SDK call. In the
queue path, existing queue API spans remain mixed; stage 127 is scalar-only.
Native ARM/x86 execution reports these hardware counters unavailable.

CSV and binary exports retain these samples with zero raw monotonic timestamps.
The CPU parser emits **blank** `wall_ns`, `thread_ns` and `process_ns` for them,
not zero elapsed time. It still emits valid RV64 cycles, instructions and IPC.
Use timeline events and anchors for placement, with the normal interpolation
limits; counter-only samples do not supply an independent elapsed-time clock.

Check the exported thread affinity records as well as sample validity. This
low-overhead sampler relies on the fixed CPU verified at registration rather
than calling `sched_getcpu` at each command. Reject traces with affinity changes,
overflow or missing boundaries. Increased event count requires adequate ring
capacity; a small complete run should be checked before a large capture.

Subtract nested counter children before aggregation. Do not add stage 127 to
its individual Index-write children, or descriptors to their issuer parents.
`cpu_descriptor` can support CPU normalization; `command_api_mixed` and
`fence_api_mixed` do **not** separate host instructions from hardware backpressure.
This change does not make all issuer time pure CPU, nor does it isolate pure
NPU execution from admission, DMA and completion waits.
