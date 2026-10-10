#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(rtk proxy dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$root"
runner="$root/scripts/experiment/run-metal-quality-ppl.sh"
build_root="$root/build-metal-llama-full"
output="$root/output/experiment/metal-llama-full-$(rtk proxy date +%Y%m%d-%H%M%S)-$$"
dry_args=()
case "${1:-}" in
    '') [[ $# == 0 ]] || exit 2 ;;
    --dry-run) [[ $# == 1 ]] || exit 2; dry_args=(--dry-run) ;;
    *) printf 'Usage: bash run-metal-llama-full.sh [--dry-run]\n' >&2; exit 2 ;;
esac
export GGML_GEMMINI_METAL_ACTIVATION_FP16=1
cases=(llama-rtnw4 llama-rtnw8
       llama-potal4-d16 llama-potal4-d32 llama-potal4-d64
       llama-potal8-d16 llama-potal8-d32 llama-potal8-d64)
[[ ${#cases[@]} == 8 ]] || exit 2
if (( ${#dry_args[@]} == 0 )); then
    rtk proxy mkdir -p -- "$output"
    printf '%s\n' "$$" > "$output/runner.pid"
    printf '%s\n' "${cases[@]}" > "$output/queue.txt"
    printf 'case\tstate\texit\n' > "$output/status.tsv"
    printf 'method\tmodel\tbits\tdim\tchunks\tscored_tokens\tppl\tprocess_seconds\texit\n' > "$output/results.tsv"
fi
printf 'Llama-3.2-1B FULL PPL: 8 cases, context=512, chunks=-1. Output: %s\n' "$output"
failed=0
for id in "${cases[@]}"; do
    if (( ${#dry_args[@]} )); then
        rtk proxy bash "$runner" --case "$id" --build-root "$build_root" --prepare --dry-run
        rtk proxy bash "$runner" --case "$id" --build-root "$build_root" --context 512 --chunks -1 --dry-run
        continue
    fi
    printf '%s\trunning\t\n' "$id" >> "$output/status.tsv"
    status=0
    printf 'Preparing %s (log: %s/%s-build.log)\n' "$id" "$output" "$id"
    rtk proxy bash "$runner" --case "$id" --build-root "$build_root" --prepare \
        > "$output/$id-build.log" 2>&1 || status=$?
    if (( status == 0 )); then
        rtk proxy bash "$runner" --case "$id" --build-root "$build_root" \
            --context 512 --chunks -1 --output "$output/$id" || status=$?
    fi
    if (( status == 0 )); then
        rtk proxy awk 'NR > 1' "$output/$id/results.tsv" >> "$output/results.tsv"
        printf '%s\tcomplete\t0\n' "$id" >> "$output/status.tsv"
    else
        failed=1
        printf '%s\tfailed\t%s\n' "$id" "$status" >> "$output/status.tsv"
        printf 'Failed %s (exit %s); continuing. Check build and PPL logs.\n' "$id" "$status" >&2
    fi
done
printf 'Finished queue. Results: %s/results.tsv; status: %s/status.tsv\n' "$output" "$output"
exit "$failed"
