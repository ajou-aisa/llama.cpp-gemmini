# MXINT / PoTal numerical policy

These numerical definitions must remain identical when rebuilding or moving
evaluation to another host. They are not a scheduler or an experiment queue.
See `GEMMINI-CUDA-NATIVE-ACCURACY.md` for frontend builds and
`GEMMINI-CUDA-PPL.md` for the existing PPL profile environments.

## Rounding and storage

MXINT and PoTal use mathematical nearest rounding with half ties away from zero:
`+0.5 -> +1`, `-0.5 -> -1`, `+2.5 -> +3`, `-2.5 -> -3`. This corresponds to
Microsoft microxcaling `round="nearest"`, not `round="even"`. Do not introduce
the F32 double-rounding behavior of a literal `abs(x) + 0.5` at adjacent values.

| Stage | Required policy |
| --- | --- |
| MXINT weights | Original FP16 source, block32, maximum shared exponent, PoT scales, ties away |
| MXINT activations | Block32, maximum shared exponent, PoT scales, ties away; no compensation |
| PoTal initial activation integers | Ties away, independent of caller floating-point rounding mode |
| PoTal folding shifts | Existing ties-away behavior |
| PoTal main/residual | Existing fragment order, ordered saturation, radix compensation and FP32 merge |
| RTN Q4_0/Q8_0 and NVFP4 | Their separate existing format/reference policies; not an MXINT rounding override |

MXINT clipping remains `[-7,7]` for 4 bits and `[-127,127]` for 8 bits.
Do not substitute `[-8,7]` for the current MXINT4 definition. HP1 storage keeps
PoT channel scales in E8M23 and block exponent differences in int16.

The verified experiment weights use `*.MXINT{4,8}_TIES_AWAY.HP1.gguf` and
`.mxint.json` sidecars. Retain their original FP16 identity, numerical metadata
and checkpoint hashes. A filename containing `RNE` is not sufficient evidence
of the actual weight policy. Do not overwrite older models or reinterpret their
results as a new rounding recipe.

## Evaluation geometry

PoTal n=4/n=8 identifies the PE/main-compensation digit width, not the entire
precision of all intermediate tensors. TOP3 is anchor 3. LINEAR uses unquantized
attention; FULL applies its configured policy to all linear layers and QK/PV.
Softmax and normalization retain the existing floating-point path; KV is F16.

Keep the existing context/batch/microbatch, independent-request boundaries,
stripe schedule, seed and ordered likelihood reduction when comparing results.
Do not equate token-length normalization with a task-specific `acc_norm`.
Use the pinned harness task definitions and aggregation.

The existing weight audit compared all four experiment checkpoints with the
pinned Microsoft reference and C decoder, and independently checked
3,459,416,064 weight values. That historical evidence is not a new host/GPU
validation or a complete downstream QA result. Rebuilding on another GPU still
requires the corresponding numerical owners and model checks.
