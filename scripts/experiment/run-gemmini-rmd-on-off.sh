#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
    cat <<'EOF'
usage: run-gemmini-rmd-on-off.sh [options]

Run compile-time RMD ON and OFF llama-cli binaries with identical inputs,
copy every generated output artifact under /root/output, and create one
tar.gz archive to return for comparison.

Required binaries are produced on the Alveo manager with:

  run-gemmini-rmd-on-off.sh --print-build-commands

Options:
  --print-build-commands
                       Print exact manager-side ON/OFF build commands
  --run-id ID          Safe result identifier (default: UTC timestamp)
  --workspace PATH     Guest repository root
  --model PATH         GGUF model
  --on-binary PATH     llama-cli built with GGML_GEMMINI_ENABLE_RMD=ON
  --off-binary PATH    llama-cli built with GGML_GEMMINI_ENABLE_RMD=OFF
  --output-root PATH   Result root (default: /root/output)
  --prompt TEXT        Prompt used by both variants
  -h, --help           Show this help

The result archive is written to /root/output/<run-id>.tar.gz by default.
Both variants run even if one fails; the script returns the first nonzero
variant status after preserving both result directories and the archive.
EOF
}

build_commands() {
    cat <<'EOF'
# Run both commands from the checked-out repository on the Alveo manager.
env PATH="/home/alveo/firesim/.conda-env/bin:$PATH" \
  GEM_HOME=/home/alveo/firesim/target-design/chipyard/generators/gemmini/software \
  BUILD_DIR=build-riscv-rmd-on \
  LOG_DEBUG=1 LOG_CYCLE=1 CYCLE_DETAIL=1 \
  GGML_GEMMINI_OPTION=WS \
  GGML_GEMMINI_ENABLE_RMD=ON \
  GGML_GEMMINI_DEFAULT_RMD_BACKEND=WS \
  GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE=ON \
  GGML_GEMMINI_EXSIA_PROFILE_SCOPE=OFF \
  ./build-riscv.sh -DLLAMA_BUILD_TESTS=OFF \
  '-DCMAKE_CXX_FLAGS=-DERROR_COMPENSATION=1 -Dgemmini_set_tile_ws=gemmini_set_tile'

env PATH="/home/alveo/firesim/.conda-env/bin:$PATH" \
  GEM_HOME=/home/alveo/firesim/target-design/chipyard/generators/gemmini/software \
  BUILD_DIR=build-riscv-rmd-off \
  LOG_DEBUG=1 LOG_CYCLE=1 CYCLE_DETAIL=1 \
  GGML_GEMMINI_OPTION=WS \
  GGML_GEMMINI_ENABLE_RMD=OFF \
  GGML_GEMMINI_DEFAULT_RMD_BACKEND=WS \
  GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE=ON \
  GGML_GEMMINI_EXSIA_PROFILE_SCOPE=OFF \
  ./build-riscv.sh -DLLAMA_BUILD_TESTS=OFF \
  '-DCMAKE_CXX_FLAGS=-DERROR_COMPENSATION=1 -Dgemmini_set_tile_ws=gemmini_set_tile'
EOF
}

die_usage() {
    printf 'run-gemmini-rmd-on-off: %s\n' "$*" >&2
    usage >&2
    exit 2
}

die() {
    printf 'run-gemmini-rmd-on-off: %s\n' "$*" >&2
    exit 1
}

workspace=/root/workspace/3rd_llama.cpp
model=/root/workspace/llama.cpp/models/gpt2.Q8_0.gguf
on_binary=
off_binary=
output_root=/root/output
run_id=$(date -u +%Y%m%d-%H%M%S)
prompt='The quick brown fox jumps over the lazy dog.'

while (($# > 0)); do
    case $1 in
        --run-id)
            (($# >= 2)) || die_usage 'missing value for --run-id'
            run_id=$2
            shift 2
            ;;
        --workspace)
            (($# >= 2)) || die_usage 'missing value for --workspace'
            workspace=$2
            shift 2
            ;;
        --model)
            (($# >= 2)) || die_usage 'missing value for --model'
            model=$2
            shift 2
            ;;
        --on-binary)
            (($# >= 2)) || die_usage 'missing value for --on-binary'
            on_binary=$2
            shift 2
            ;;
        --off-binary)
            (($# >= 2)) || die_usage 'missing value for --off-binary'
            off_binary=$2
            shift 2
            ;;
        --output-root)
            (($# >= 2)) || die_usage 'missing value for --output-root'
            output_root=$2
            shift 2
            ;;
        --prompt)
            (($# >= 2)) || die_usage 'missing value for --prompt'
            prompt=$2
            shift 2
            ;;
        --print-build-commands)
            build_commands
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die_usage "unknown argument: $1"
            ;;
    esac
done

[[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
    die_usage "unsafe --run-id: $run_id"
[[ $workspace = /* ]] || die_usage '--workspace must be absolute'
[[ $model = /* ]] || die_usage '--model must be absolute'
[[ $output_root = /* ]] || die_usage '--output-root must be absolute'
[[ -d $workspace ]] || die_usage "workspace does not exist: $workspace"
[[ -f $model ]] || die_usage "model does not exist: $model"

on_binary=${on_binary:-"$workspace/build-riscv-rmd-on/bin/llama-cli"}
off_binary=${off_binary:-"$workspace/build-riscv-rmd-off/bin/llama-cli"}
[[ -x $on_binary ]] || die_usage "ON binary is not executable: $on_binary"
[[ -x $off_binary ]] || die_usage "OFF binary is not executable: $off_binary"

on_binary=$(cd -- "$(dirname -- "$on_binary")" && printf '%s/%s\n' "$(pwd -P)" "$(basename -- "$on_binary")")
off_binary=$(cd -- "$(dirname -- "$off_binary")" && printf '%s/%s\n' "$(pwd -P)" "$(basename -- "$off_binary")")
model=$(cd -- "$(dirname -- "$model")" && printf '%s/%s\n' "$(pwd -P)" "$(basename -- "$model")")
workspace=$(cd -- "$workspace" && pwd -P)

cache_for_binary() {
    local binary=$1
    local build_dir
    build_dir=$(cd -- "$(dirname -- "$binary")/.." && pwd -P)
    printf '%s/CMakeCache.txt\n' "$build_dir"
}

require_rmd_contract() {
    local label=$1
    local binary=$2
    local expected=$3
    local cache
    cache=$(cache_for_binary "$binary")
    [[ -f $cache ]] || die_usage "$label build has no CMakeCache.txt: $cache"
    grep -Fqx -- "GGML_GEMMINI_ENABLE_RMD:BOOL=$expected" "$cache" ||
        die_usage "$label binary is not an RMD $expected build: $binary"
}

require_rmd_contract ON "$on_binary" ON
require_rmd_contract OFF "$off_binary" OFF

mkdir -p -- "$output_root"
output_root=$(cd -- "$output_root" && pwd -P)
run_dir="$output_root/$run_id"
archive="$output_root/$run_id.tar.gz"
[[ ! -e $run_dir ]] || die "result directory already exists: $run_dir"
[[ ! -e $archive ]] || die "result archive already exists: $archive"

temp_root=$(mktemp -d "${TMPDIR:-/tmp}/gemmini-rmd-on-off.XXXXXX")
cleanup() {
    rm -rf -- "$temp_root"
}
trap cleanup EXIT

hash_value() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -- "$1" | awk '{print $1}'
    else
        shasum -a 256 -- "$1" | awk '{print $1}'
    fi
}

mkdir -- "$run_dir"
start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
model_hash=$(hash_value "$model")
printf '%s  %s\n' "$model_hash" "$model" >"$run_dir/model.sha256"

run_variant() {
    local variant=$1
    local binary=$2
    local backend=$3
    local variant_dir="$run_dir/$variant"
    local work_dir="$temp_root/$variant"
    local cache
    local rmd_setting
    local status
    local -a command

    cache=$(cache_for_binary "$binary")
    if [[ $variant == on ]]; then
        rmd_setting=ON
    else
        rmd_setting=OFF
    fi
    mkdir -- "$variant_dir" "$work_dir"
    mkdir -- "$work_dir/output" "$variant_dir/output"
    cp -- "$cache" "$variant_dir/CMakeCache.txt"
    printf '%s  %s\n' "$(hash_value "$binary")" "$binary" >"$variant_dir/binary.sha256"

    command=(
        "$binary"
        --no-warmup
        -m "$model"
        -p "$prompt"
        -n 1
        -t 1
        --seed 1234
        --temp 0
    )
    printf -v command_escaped '%q ' "${command[@]}"
    printf '%s\n' "${command_escaped% }" >"$variant_dir/command.txt"
    {
        printf 'RMD_VARIANT=%s\n' "$variant"
        printf 'GGML_GEMMINI_ENABLE_RMD=%s\n' "$rmd_setting"
        printf 'GEMMINI_MATMUL_MODE=STRIPE_PIPELINE\n'
        if [[ -n $backend ]]; then
            printf 'GEMMINI_RMD_BACKEND=%s\n' "$backend"
        else
            printf 'GEMMINI_RMD_BACKEND=<unset>\n'
        fi
    } >"$variant_dir/environment.txt"

    if (
        cd -- "$work_dir" || exit
        export RMD_VARIANT="$variant"
        export GEMMINI_MATMUL_MODE=STRIPE_PIPELINE
        if [[ -n $backend ]]; then
            export GEMMINI_RMD_BACKEND="$backend"
        else
            unset GEMMINI_RMD_BACKEND
        fi
        "${command[@]}"
    ) >"$variant_dir/stdout.txt" 2>"$variant_dir/stderr.txt"; then
        status=0
    else
        status=$?
    fi

    cp -a -- "$work_dir/output/." "$variant_dir/output/"
    printf '%s\n' "$status" >"$variant_dir/exit-status.txt"
    return "$status"
}

if run_variant on "$on_binary" WS; then
    on_status=0
else
    on_status=$?
fi
if run_variant off "$off_binary" ''; then
    off_status=0
else
    off_status=$?
fi

end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_sha=$(git -C "$workspace" rev-parse HEAD 2>/dev/null || printf unknown)
git_branch=$(git -C "$workspace" symbolic-ref --short -q HEAD 2>/dev/null || printf HEAD)
if git -C "$workspace" diff --quiet --ignore-submodules HEAD 2>/dev/null &&
    [[ -z $(git -C "$workspace" ls-files --others --exclude-standard 2>/dev/null) ]]; then
    git_dirty=0
else
    git_dirty=1
fi

{
    printf 'run_id=%s\n' "$run_id"
    printf 'start_utc=%s\n' "$start_utc"
    printf 'end_utc=%s\n' "$end_utc"
    printf 'workspace=%s\n' "$workspace"
    printf 'output_root=%s\n' "$output_root"
    printf 'model=%s\n' "$model"
    printf 'model_sha256=%s\n' "$model_hash"
    printf 'git_sha=%s\n' "$git_sha"
    printf 'git_branch=%s\n' "$git_branch"
    printf 'git_dirty=%s\n' "$git_dirty"
    printf 'on_binary=%s\n' "$on_binary"
    printf 'off_binary=%s\n' "$off_binary"
    printf 'on_exit_status=%s\n' "$on_status"
    printf 'off_exit_status=%s\n' "$off_status"
} >"$run_dir/manifest.txt"

if tar -C "$output_root" -czf "$archive" "$run_id"; then
    archive_status=0
else
    archive_status=$?
fi
if ((archive_status != 0)); then
    printf 'run-gemmini-rmd-on-off: failed to create archive: %s\n' "$archive" >&2
    exit "$archive_status"
fi

printf 'result_dir=%s\n' "$run_dir"
printf 'archive=%s\n' "$archive"
printf 'on_exit_status=%s\n' "$on_status"
printf 'off_exit_status=%s\n' "$off_status"

if ((on_status != 0)); then
    exit "$on_status"
fi
exit "$off_status"
