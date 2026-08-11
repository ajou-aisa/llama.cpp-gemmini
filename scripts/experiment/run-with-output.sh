#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
    printf 'usage: %s <name> -- <command> [args...]\n' "${0##*/}" >&2
    exit 2
}

[[ $# -ge 3 && $2 == -- && -n $3 ]] || usage
name=$1
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || usage

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/../.." && pwd -P)
initial_cwd=$(pwd -P)
start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_sha=$(git -C "$repo_root" rev-parse HEAD)
git_branch=$(git -C "$repo_root" symbolic-ref --short -q HEAD || printf HEAD)
if [[ -n $(git -C "$repo_root" status --porcelain) ]]; then
    git_dirty=1
else
    git_dirty=0
fi

if [[ -n ${OUTPUT_ROOT:-} ]]; then
    output_root=$OUTPUT_ROOT
else
    output_root="$repo_root/output"
fi
[[ $output_root = /* ]] || output_root="$initial_cwd/$output_root"
mkdir -p -- "$output_root/experiment"
experiment_dir="$output_root/experiment/$name-$(date -u +%Y%m%d-%H%M%S)"
mkdir -- "$experiment_dir"
mkdir -- "$experiment_dir/raw"

log_dir="$experiment_dir/raw"
gemmini_log_dir="$log_dir"
cycle_detail_log="$log_dir/exsia-cycle-detail.jsonl"
printf -v command_escaped '%q ' "${@:3}"
command_escaped=${command_escaped% }

set +e
(
    cd -- "$repo_root" || exit
    export OUTPUT_ROOT="$output_root"
    export EXPERIMENT_DIR="$experiment_dir"
    export OUTPUT_DIR="$experiment_dir"
    export LOG_DIR="$log_dir"
    export GEMMINI_LOG_DIR="$gemmini_log_dir"
    export GGML_GEMMINI_CYCLE_DETAIL_LOG="$cycle_detail_log"
    "${@:3}"
) >"$log_dir/stdout.txt" 2>"$log_dir/stderr.txt"
exit_status=$?
set -e
end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
    printf 'name=%s\n' "$name"
    printf 'start_utc=%s\n' "$start_utc"
    printf 'end_utc=%s\n' "$end_utc"
    printf 'initial_cwd=%s\n' "$initial_cwd"
    printf 'repo_root=%s\n' "$repo_root"
    printf 'command=%s\n' "$command_escaped"
    printf 'output_root=%s\n' "$output_root"
    printf 'experiment_dir=%s\n' "$experiment_dir"
    printf 'output_dir=%s\n' "$experiment_dir"
    printf 'log_dir=%s\n' "$log_dir"
    printf 'gemmini_log_dir=%s\n' "$gemmini_log_dir"
    printf 'ggml_gemmini_cycle_detail_log=%s\n' "$cycle_detail_log"
    printf 'git_sha=%s\n' "$git_sha"
    printf 'git_branch=%s\n' "$git_branch"
    printf 'git_dirty=%s\n' "$git_dirty"
    printf 'exit_status=%s\n' "$exit_status"
} >"$experiment_dir/manifest.txt"

exit "$exit_status"
