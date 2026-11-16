# Static weight preparation before inference

Both PoTal W8 and W4 prepare immutable model weights during scheduler reservation,
before the backend's first graph execution. Llama context initialization already
reserves prefill, decode, and prefill graphs; their assigned weights are prepared
once and retained by the existing backend weight cache.

This does not run a dummy inference. The optional `ggml_backend_prepare_node`
registry callback receives only nodes assigned to that backend. PoTal prepares
HP1 Linear weights in host `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` buffers, using the
same checked weight reader and packing path as lazy execution. Activation and
output storage are not required or allocated by the callback, and no accelerator
commands are issued.

Q/K/V projection weights are static and eligible. Attention Q, K, V and P tensors
are runtime data and are not preloaded. Other backends and mutable compute-buffer
weights are unaffected.

## Cache policy

- With `GGML_POTAL_WEIGHT_CACHE_MB` unset, eager preparation may grow the effective
  cache budget beyond the original 1024 MiB default to retain all prepared weights.
- An explicit positive budget remains fixed. If an assigned weight cannot fit,
  reservation fails before allocating or unpacking that new weight. Increase the
  configured budget rather than treating an over-budget transient as prepared.
- `GGML_POTAL_WEIGHT_CACHE_MB=0` disables caching and eager preparation.
- Empty or malformed values retain the existing 1024 MiB parsing fallback without
  enabling automatic growth.
- Direct users that do not reserve a graph retain the existing lazy preparation
  behavior. Lazy lookups never auto-grow the budget.

Preparation is idempotent for the existing buffer/generation/data/layout cache
key. Only successfully validated weights are inserted. Prepared buffers belong
to the backend; the source model must outlive its contexts as before. Existing
cache-clear, immutable-weight, and fatal-abandon contracts still apply.

The first `graph_compute` entry permanently closes eager preparation for that
backend, including when compute fails. Later KV-cache graph reservations cannot
restart preload. Clearing the cache does not reopen it; subsequent execution uses
the existing checked lazy path.

## Timing and counters

Initialization preparation updates live cache entries, bytes and effective budget.
It does not increment inference graph, weight hit/miss, pack/validation, Attention,
or hardware counters. Consequently a context-level stats snapshot before the first
decode distinguishes initialization work from work inside the first graph.

Moving preparation out of the graph timer reduces an initialization-excluding
TTFT boundary, not the total work from program start to first token. Eliminating
repeated over-budget packing can additionally reduce real decode work, at the
cost of retaining more prepared weight memory.

Historical FPGA `weight_pack_s_ns` can estimate the amount moved out of a graph
timer, assuming other costs stay unchanged. It does not predict current-source
FPGA performance, allocator/page-fault effects or a new Attention policy's speed.
Native initialization timings are host measurements, not Rocket target timings.

## Verification

`potal-weight-preload-test` is built for both backends. It exercises assigned-node
selection, repeated prefill/decode reservations, clear/compute lifecycle,
metadata failure and scheduler reset, exact and insufficient cache budgets,
cache opt-out, and eager/lazy packed-operand equality without hardware execution.
The ordinary backend arithmetic and golden tests remain separate.

The same executable accepts `--model /path/to/model.gguf` for a hardware-free
model/context initialization check. It prints cache entries, bytes, budget and
host context-initialization time, then exits without computing an inference graph.
