#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(rtk proxy dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
build_root="$root/build-metal-rtn-full"
output="$root/output/experiment/metal-rtn-full-$(rtk proxy date +%Y%m%d-%H%M%S)-$$"
prepare_only=0; run_only=0
while (( $# )); do
    case "$1" in
        --output|--build-root)
            [[ $# -ge 2 ]] || { printf 'Missing value: %s\n' "$1" >&2; exit 2; }
            case "$1" in --output) output=$2 ;; --build-root) build_root=$2 ;; esac
            shift 2 ;;
        --prepare-only) prepare_only=1; shift ;;
        --run-only) run_only=1; shift ;;
        --help|-h)
            printf '%s\n' 'Usage: bash run-metal-rtn-full.sh [--output ABS_DIR] [--build-root ABS_DIR] [--prepare-only | --run-only]' \
                'Builds four Metal profiles, then runs all eight RTN cases over the full WikiText-2 test set.' \
                'GPT-2 and Llama-3.2-1B; n=4/8; RTN-W A=FP16, RTN-WA A=n; residual and outlier selection OFF.'
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ $build_root == /* && $output == /* ]] || { printf 'Paths must be absolute.\n' >&2; exit 2; }
cd -- "$root"
export GGML_GEMMINI_METAL_ACTIVATION_FP16=1
[[ $prepare_only == 0 || $run_only == 0 ]] || { printf 'Choose only one preparation mode.\n' >&2; exit 2; }
if (( run_only == 0 )); then
    printf 'Preparing RTN-W4, RTN-W8, RTN-WA4, RTN-WA8. Output: %s\n' "$output"
    rtk proxy bash "$root/scripts/experiment/run-metal-quality-ppl.sh" \
        --rtn-only --prepare --build-root "$build_root"
fi
(( prepare_only == 0 )) || exit 0
rtk proxy mkdir -p -- "$output"
printf '%s\n' "$$" > "$output/runner.pid"
cases=(gpt2-rtnw4 gpt2-rtnw8 gpt2-rtnwa4 gpt2-rtnwa8 llama-rtnw4 llama-rtnw8 llama-rtnwa4 llama-rtnwa8)
printf '%s\n' "${cases[@]}" > "$output/queue.txt"
printf 'case\tstate\texit\n' > "$output/status.tsv"
printf 'method\tmodel\tbits\tdim\tchunks\tscored_tokens\tppl\tprocess_seconds\texit\n' > "$output/results.tsv"
printf 'Starting all eight FULL PPL cases; context=512, chunks=-1. Output: %s\n' "$output"
failed=0
for id in "${cases[@]}"; do
    printf '%s\trunning\t\n' "$id" >> "$output/status.tsv"
    status=0
    rtk proxy bash "$root/scripts/experiment/run-metal-quality-ppl.sh" \
        --case "$id" --build-root "$build_root" --context 512 --chunks -1 --output "$output/$id" || status=$?
    if (( status == 0 )); then
        rtk proxy awk 'NR > 1' "$output/$id/results.tsv" >> "$output/results.tsv"
        printf '%s\tcomplete\t0\n' "$id" >> "$output/status.tsv"
    else
        failed=1
        printf '%s\tfailed\t%s\n' "$id" "$status" >> "$output/status.tsv"
        printf 'Failed %s (exit %s); continuing with the next case.\n' "$id" "$status" >&2
    fi
done
printf 'Finished queue. Results: %s/results.tsv; status: %s/status.tsv\n' "$output" "$output"
exit "$failed"
