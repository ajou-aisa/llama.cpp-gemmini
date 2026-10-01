# Independent evaluation metrics

Metrics are the activation (ACT), residual (RES) and SCU measurements. Canonical campaign entry:
`python3 scripts/eval/run_measurement.py metric activation|residual|scu` (see `scripts/eval/README.md`).

ACT, RES and SCU collection/reduction is separate from the cycle provider. This
package imports no cycle model, calls no cycle estimator, and produces no latency,
frequency, PPL, CUDA, Jetson or FPGA measurement. Default inference metric switches
are all OFF. Enable only the requested native collector:

* `GGML_GEMMINI_ACT_METRICS=1`: `activation_metrics.json`
* `GGML_GEMMINI_RESIDUAL_METRICS=1`: `residual_metrics.json`
* `GGML_GEMMINI_SCALE_METRICS=1`: `scale_alignment_metrics.json`

## Manifest and reduction

Create `evaluation_manifest.json` with exactly these fields. The hashes below are
examples, not evidence. Bind the actual tokenizer artifact and checkout; describe
the actual chunk policy. The wrapper also records model/dataset/binary hashes and
the native token/chunk identities in `workload-binding.json`.

```json
{
  "model": "GPT-2 124M",
  "dataset": "WikiText-2",
  "tokenizer_sha256": "<64 lowercase hex characters>",
  "chunk_policy": "METRIC_PREFILL_256:split=test:max_chunks=1:context=256:tail=drop:output=second_half",
  "precision": "A8W8",
  "dim": 64,
  "BK": 32,
  "seed": 1234,
  "git_sha": "<40 lowercase hex characters>"
}
```

The binding is SHA256 of **exact manifest file bytes**, including whitespace. Each
native record and reduced output carries `manifest_sha256`. Editing the manifest
after collection invalidates that binding. The reducer validates required fields,
hash syntax, profile, stream identity, observation/invocation coverage and count
invariants. It does not independently recover tokenizer provenance from a GGUF;
the manifest author must supply the actual tokenizer artifact hash.

From the repository root, using only Python 3.11+ and the standard library:

```sh
uv run --no-project --offline python -m evaluation activation \
  --manifest evaluation_manifest.json --input activation.jsonl --output-dir fresh-results
uv run --no-project --offline python -m evaluation residual \
  --manifest evaluation_manifest.json --input residual.jsonl --output-dir fresh-results
uv run --no-project --offline python -m evaluation scale \
  --manifest evaluation_manifest.json --input scale.jsonl --output-dir fresh-results
```

Existing files are never overwritten. Undefined ratios are JSON `null`. All
fractions use accumulated integer numerators/denominators, never layer averages.
Each output also includes the exact input stream SHA256. JSON Schemas are under
`evaluation/schemas/`; semantic validation additionally checks cross-record
coverage and numerical invariants.

## Native collection

Use an independently enabled metric build of `llama-eval-workload`. A bounded
collection example (not a full WikiText campaign):

```sh
uv run --no-project --offline scripts/eval/activation_quant_metrics.py \
  --runner /path/to/llama-eval-workload --model /path/to/model.gguf \
  --dataset /path/to/wiki.test.raw --split test --max-chunks 1 \
  --evaluation-manifest evaluation_manifest.json --output fresh-act-run
```

Use `residual_path_metrics.py` or `scale_alignment_metrics.py` for the other
collectors, each with its matching build. The wrapper checks compiled precision,
DIM and enabled sink, copies the exact manifest, and passes `--manifest-sha256` to
the native producer. It also passes the manifest seed, verifies the native seed,
and requires the exact chunk policy string shown above with the requested split
and max-chunks value. Existing unbound historical `--reduce` remains available for
old ACT/RES evidence; new collection requires a manifest. Bound reduction is
available with `--reduce INPUT --evaluation-manifest MANIFEST --output FRESH`.
The existing full-inference collector uses the functional IM2P_SIM route; the
standalone observer/reducer smoke requires no cycle library or timing call.

## Definitions

* **ACT:** original signed FP values, each logical row and original BK32 block;
  population mean and variance including every valid position; strict
  `x > mean + 2*sigma`. No absolute value, top1 exclusion or sample variance.
  `observed_counts` accepts original FP, PoTal and residual masks, and actual
  `(logical_row, original_block)` requant events. `requant_ratio` is the union of
  requantized logical blocks divided by eligible original BK32 blocks. Repeated
  events are retained separately as `p3_requantization_events`.
* **RES:** `logical_ratio = sum(compact M*N*K) / sum(main M*N*K)`.
  `retained_row_factor = sum(retained radix rows) / sum(pre-pruning radix rows)`;
  `retained_k_factor = sum(compact K) / sum(original K per main stripe)`.
  Radix-zero stripes remain in main/K denominators; the radix row denominator is
  zero when no radix limb was produced. `RADIX_STRIPE` is required for every main
  stripe, and nonzero radix stripes require completed compact work. Row/lane maps,
  original BK32 masks and compact runs are validated. Physical MAC capacities use
  each recorded fragment's DIM cubed. Both main and compact K preserve original
  BK32/run boundaries before padding; `ceil(total compact K / DIM)` is incorrect.
* **SCU:** `scale_domain=hp1_block_pot_to_channel_anchor`. ΔW is the actual integer
  HP1 shift `m`; original block PoT scale equals channel anchor times `2**m`.
  Average weights each observed `(invocation, work type, stripe, column, original
  block)` once. Partial-sum update fraction weights actual emitted partial-sum
  counts; an update means a required nonzero SCU shift, not a claim that its
  numerical value changed. Zero-weight records have zero offset and zero updates.
  The same aggregation is also reported per producer `work_type`: `dense`, `residual` and their union `overall`,
  each from its own integer sums (`delta_w_sum`, `alignment_count`, `updated_partial_sum_count`,
  `total_partial_sum_count`) before division; an empty work type has null average, maximum and fraction. The
  top-level `avg_delta_w`, `max_delta_w` and `scu_update_fraction` stay the `overall` values.
  The producer can instead aggregate (`llama-eval-workload --scale-mode aggregate`): it validates every coordinate as
  above (duplicates per invocation included) and emits one `AGGREGATE` record per (chunk, layer, work type) with
  `delta_w_sum`, `max_delta_w`, `alignment_count`, `updated_partial_sum_count`, `total_partial_sum_count` and
  `zero_weight_count` (schema `im2p-scale-alignment-aggregate`; `RUN_END` adds the total `alignment_count` and
  `scale_invocation_count`). The reducer reads either stream and returns the same summary.
  The GGUF lacks the pre-PoT FP scale; outputs explicitly declare
  `pre_pot_fp_scale_available=false`. This is not a reconstructed FP-to-PoT error.

## Verification

```sh
uv run --no-project --offline python -B -m evaluation.tests.test_framework
uv run --no-project --offline python -B -m evaluation.tests.test_framework --output FRESH-SMOKE
pyright -p evaluation/pyrightconfig.json
uv run --no-project --offline --with ruff ruff check evaluation scripts/eval/metric_run.py scripts/eval/scale_alignment_metrics.py
```

Smoke fixtures are synthetic known tensors, explicitly labeled in their manifest;
they are architecture checks, not a dataset measurement campaign. Fixtures cover
exact ACT/RES/SCU values, public CLI success/error, malformed counts, mismatched
profiles, manifest tampering and incomplete streams.
