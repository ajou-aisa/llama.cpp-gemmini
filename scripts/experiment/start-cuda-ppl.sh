#!/usr/bin/env bash
# Run a selected queue under user systemd, independent of the SSH terminal.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
output="$root/output/cuda-ppl"
config=
forward=("$@")
while (( $# )); do
    case "$1" in
        --output) output=${2:?Missing output directory}; shift 2 ;;
        --config) config=$(realpath -- "${2:?Missing config file}"); shift 2 ;;
        --help) printf 'Usage: %s [--config FILE] [--output DIR] [--case ID] [--build] [--smoke-only]\n' "$0"; exit 0 ;;
        *) shift ;;
    esac
done
mkdir -p "$output"
output=$(cd -- "$output" && pwd)
forward+=(--output "$output")
if [[ -n $config ]]; then forward+=(--config "$config"); fi
unit="potal-cuda-$(printf '%s' "$output" | sha256sum | cut -c 1-12)"
# Keep the user service manager alive after the last SSH session closes.
loginctl enable-linger "$(id -un)"
build_env=()
for name in GEMMINI_SW_PATH CUDACXX CUDAARCHS; do
    if [[ -n ${!name:-} ]]; then build_env+=("--setenv=$name=${!name}"); fi
done
systemd-run --user --collect --unit="$unit" --description='CUDA PoTal/RTN-WA PPL queue' \
    "${build_env[@]}" \
    --property="WorkingDirectory=$root" \
    --property="StandardOutput=append:$output/queue.log" \
    --property="StandardError=append:$output/queue.log" \
    /usr/bin/python3 -u "$root/scripts/experiment/cuda-ppl-queue.py" --run "${forward[@]}"
printf 'Started %s.service; log: %s/queue.log\n' "$unit" "$output"
