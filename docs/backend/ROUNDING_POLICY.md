# MXINT and PoTal rounding policy

New builds use RN: round to nearest, with exact half ties to the even integer.
Examples: `+0.5 -> 0`, `-0.5 -> 0`, `+1.5 -> 2`, `-1.5 -> -2`,
`+2.5 -> 2`, and `-2.5 -> -2`. Adjacent representable values are not ties.

The policy applies to registered MXINT4/8 HP1 weight quantizers, host
BLOCK/MXINT activation conversion, PoTal wide activation conversion and
integer folding right shifts, and CUDA MXINT activation conversion.
CPU integer conversion does not change or depend on the caller's rounding mode.
Existing floating-point scale calculations retain their original precision.

Scale selection, block32 grouping, packing, saturation, compensation digit
order and reduction order are unchanged. MXINT uses symmetric clipping
`[-7,7]` or `[-127,127]`. Standard Q4_0/Q8_0 weight formats, FP16 and NVFP4
retain their existing format-specific policies.

`tools/quantize/mxint-hp1.py` defaults to `--rounding even` with the pinned
Microsoft reference. Explicit `--rounding nearest` retains the historical
ties-away conversion. Both modes write their actual policy into the sidecar
and require a new output path.

Changing source does not change existing GGUF bytes. Old ties-away checkpoints,
results and caches remain historical. RN execution with ties-away weights is
a mixed recipe, not a fully RN-quantized model. A full RN model must be
re-quantized from its original floating-point source into a new checkpoint
and identified by its contents, conversion policy and build.

The runtime anchor-policy proof reports `ties_even` for newly built backends;
it does not attest to stored-weight provenance. Rebuild affected libraries
and retain actual executable/library hashes when comparing results.
No accuracy or performance gain follows solely from changing rounding.

See [GEMMINI-EXSIA.md](GEMMINI-EXSIA.md) for folding, outlier and residual limits.
