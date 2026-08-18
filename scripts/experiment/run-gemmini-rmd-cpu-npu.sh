#!/usr/bin/env bash
set -euo pipefail
umask 077

program=${0##*/}
guest_temp_root=

cleanup_guest_temp() {
    [[ -z $guest_temp_root ]] || rm -rf -- "$guest_temp_root"
}

usage() {
    cat <<EOF
usage:
  $program manager [options]
  $program prepare [options]
  $program launch [options]
  $program run [options]

manager: run prepare and launch consecutively with one manager command.
prepare: source the FireSim manager environment, preserve any old build,
         build one RMD-enabled RISC-V binary, bundle libgomp, and update rootfs.
launch:  source the FireSim manager environment and launch the workload.
run:     execute the same guest binary with CPU and Gemmini NPU residual
         backends in ABBA order, preserve every result, and create a tar archive.

prepare options:
  --workspace PATH      repository on the Alveo manager
  --firesim-root PATH   FireSim root (default: /home/alveo/firesim)
  --build-dir NAME      build directory name (default: build-riscv-rmd-cpu-npu)

manager options:
  same options as prepare

launch options:
  --firesim-root PATH   FireSim root (default: /home/alveo/firesim)

run options:
  --workspace PATH      guest repository (default: /root/workspace/3rd_llama.cpp)
  --build-dir NAME      build directory name (default: build-riscv-rmd-cpu-npu)
  --model PATH          GGUF model
  --output-root PATH    result root (default: /root/output)
  --run-id ID           safe result identifier (default: UTC timestamp)
  --expected-bundle-id SHA256
                         bundle ID printed by prepare (required)
  --runs-per-backend N  CPU and NPU runs each (default: 1)
  --prompt TEXT         identical prompt for every run

After prepare, launch FireSim normally. Inside the guest run:

  /root/workspace/3rd_llama.cpp/scripts/experiment/$program run
EOF
}

die_usage() {
    printf '%s: %s\n' "$program" "$*" >&2
    usage >&2
    exit 2
}

die() {
    printf '%s: %s\n' "$program" "$*" >&2
    exit 1
}

hash_value() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -- "$1" | awk '{print $1}'
    else
        shasum -a 256 -- "$1" | awk '{print $1}'
    fi
}

absolute_file() {
    local path=$1
    cd -- "$(dirname -- "$path")"
    printf '%s/%s\n' "$(pwd -P)" "$(basename -- "$path")"
}

absolute_dir() {
    cd -- "$1"
    pwd -P
}

cache_has() {
    local cache=$1
    local key=$2
    local value=$3
    grep -Eq -- "^${key}(:[^=]+)?=${value}$" "$cache"
}

manifest_value() {
    local manifest=$1
    local key=$2
    awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$manifest"
}

manifest_payload_hash() {
    local manifest=$1
    if command -v sha256sum >/dev/null 2>&1; then
        awk '!/^bundle_id=/' "$manifest" | sha256sum | awk '{print $1}'
    else
        awk '!/^bundle_id=/' "$manifest" | shasum -a 256 | awk '{print $1}'
    fi
}

prepare_manager() {
    local script_dir
    local default_workspace
    local workspace
    local firesim_root=/home/alveo/firesim
    local build_name=build-riscv-rmd-cpu-npu
    local build_dir
    local previous_dir
    local manager_setup
    local update_rootfs
    local compiler
    local libgomp
    local gemmini_header
    local cache
    local bundle_id
    local caller_directory

    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
    default_workspace=$(cd -- "$script_dir/../.." && pwd -P)
    workspace=$default_workspace

    while (($# > 0)); do
        case $1 in
            --workspace)
                (($# >= 2)) || die_usage 'missing value for --workspace'
                workspace=$2
                shift 2
                ;;
            --firesim-root)
                (($# >= 2)) || die_usage 'missing value for --firesim-root'
                firesim_root=$2
                shift 2
                ;;
            --build-dir)
                (($# >= 2)) || die_usage 'missing value for --build-dir'
                build_name=$2
                shift 2
                ;;
            *)
                die_usage "unknown prepare option: $1"
                ;;
        esac
    done

    [[ $workspace = /* ]] || die_usage '--workspace must be absolute'
    [[ $firesim_root = /* ]] || die_usage '--firesim-root must be absolute'
    [[ $build_name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
        die_usage 'unsafe --build-dir'
    [[ -x $workspace/build-riscv.sh ]] ||
        die_usage "missing executable build-riscv.sh: $workspace"

    workspace=$(absolute_dir "$workspace")
    firesim_root=$(absolute_dir "$firesim_root")
    manager_setup="$firesim_root/sourceme-manager.sh"
    update_rootfs="$firesim_root/deploy/update_rootfs.sh"
    compiler="$firesim_root/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc"
    [[ -f $manager_setup ]] || die_usage "missing manager setup: $manager_setup"
    [[ -x $update_rootfs ]] || die_usage "missing rootfs updater: $update_rootfs"
    [[ -x $compiler ]] || die_usage "missing RISC-V compiler: $compiler"

    printf 'manager_step=source_firesim\n'
    caller_directory=$PWD
    cd -- "$firesim_root"
    set +u
    # shellcheck disable=SC1090
    source ./sourceme-manager.sh --skip-ssh-setup
    set -u
    cd -- "$caller_directory"
    [[ ${FIRESIM_SOURCED:-0} == 1 ]] || die 'FireSim manager environment was not sourced'

    build_dir="$workspace/$build_name"
    if [[ -e $build_dir ]]; then
        previous_dir="$build_dir.previous-$(date -u +%Y%m%d-%H%M%S)-$$"
        mv -- "$build_dir" "$previous_dir"
        printf 'preserved_build=%s\n' "$previous_dir"
    fi

    printf 'manager_step=build_cpu_npu_bundle\n'
    export PATH="$firesim_root/.conda-env/bin:$PATH"
    export GEM_HOME="$firesim_root/target-design/chipyard/generators/gemmini/software"
    export BUILD_DIR="$build_dir"
    export LOG_DEBUG=1
    export LOG_CYCLE=1
    export CYCLE_DETAIL=1
    export GGML_GEMMINI_OPTION=WS
    export GGML_GEMMINI_ENABLE_RMD=ON
    export GGML_GEMMINI_DEFAULT_RMD_BACKEND=WS
    export GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE=ON
    export GGML_GEMMINI_EXSIA_PROFILE_SCOPE=OFF
    export CXXFLAGS='-DERROR_COMPENSATION=1 -Dgemmini_set_tile_ws=gemmini_set_tile'
    (
        cd -- "$workspace"
        ./build-riscv.sh -DLLAMA_BUILD_TESTS=ON
    )
    cmake --build "$build_dir" --target test-gemmini-exsia -j
    [[ -x $build_dir/bin/test-gemmini-exsia ]] || die 'build did not produce test-gemmini-exsia'

    cache="$build_dir/CMakeCache.txt"
    [[ -x $build_dir/bin/llama-cli ]] || die 'build did not produce llama-cli'
    [[ -f $build_dir/bin/libggml-gemmini.so ]] ||
        die 'build did not produce libggml-gemmini.so'
    [[ -f $cache ]] || die 'build did not produce CMakeCache.txt'
    cache_has "$cache" GGML_GEMMINI_ENABLE_RMD 'ON|1|TRUE' ||
        die 'build cache does not enable RMD'
    cache_has "$cache" GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE 'ON|1|TRUE' ||
        die 'build cache does not allow runtime backend selection'
    cache_has "$cache" GGML_GEMMINI_OPTION WS ||
        die 'build cache does not keep main Gemmini option WS'
    cache_has "$cache" LOG_CYCLE 'ON|1|TRUE' ||
        die 'build cache does not enable cycle logging'
    cache_has "$cache" CYCLE_DETAIL 'ON|1|TRUE' ||
        die 'build cache does not enable cycle detail'

    gemmini_header=
    for candidate in "$GEM_HOME/gemmini.h" "$GEM_HOME/include/gemmini.h" \
        "$GEM_HOME/gemmini-rocc-tests/include/gemmini.h" \
        "$GEM_HOME/software/gemmini.h"; do
        if [[ -f $candidate ]]; then gemmini_header=$candidate; break; fi
    done
    [[ -n $gemmini_header ]] || die 'could not resolve gemmini.h from GEM_HOME candidates'
    libgomp=$("$compiler" -print-file-name=libgomp.so.1)
    [[ $libgomp != libgomp.so.1 && -f $libgomp ]] ||
        die 'RISC-V compiler could not resolve libgomp.so.1'
    cp -L -- "$libgomp" "$build_dir/bin/libgomp.so.1"

    {
        printf 'git_sha=%s\n' "$(git -C "$workspace" rev-parse HEAD 2>/dev/null || printf unknown)"
        printf 'source_head=%s\n' "$(git -C "$workspace" rev-parse HEAD 2>/dev/null || printf unknown)"
        printf 'source_head_tree=%s\n' "$(git -C "$workspace" rev-parse HEAD^{tree} 2>/dev/null || printf unknown)"
        printf 'source_worktree_status=%s\n' "$(git -C "$workspace" status --porcelain=v1 2>/dev/null | tr '\n' ';')"
        printf 'source_worktree_diff_sha256=%s\n' "$(git -C "$workspace" diff --binary HEAD 2>/dev/null | sha256sum | awk '{print $1}')"
        printf 'gemmini_header_path=%s\n' "$gemmini_header"
        printf 'gemmini_header_sha256=%s\n' "$(hash_value "$gemmini_header")"
        printf 'rmd_routes_test_sha256=%s\n' "$(hash_value "$build_dir/bin/test-gemmini-exsia")"
        printf 'binary_sha256=%s\n' "$(hash_value "$build_dir/bin/llama-cli")"
        printf 'gemmini_library_sha256=%s\n' \
            "$(hash_value "$build_dir/bin/libggml-gemmini.so")"
        printf 'libgomp_sha256=%s\n' "$(hash_value "$build_dir/bin/libgomp.so.1")"
        printf 'cmake_cache_sha256=%s\n' "$(hash_value "$cache")"
        printf 'build_dir=%s\n' "$build_dir"
    } >"$build_dir/experiment-build-manifest.txt"
    bundle_id=$(hash_value "$build_dir/experiment-build-manifest.txt")
    printf 'bundle_id=%s\n' "$bundle_id" >>"$build_dir/experiment-build-manifest.txt"

    printf 'manager_step=update_rootfs\n'
    (
        cd -- "$firesim_root/deploy"
        ./update_rootfs.sh
    )

    printf 'manager_status=ready\n'
    printf 'build_dir=%s\n' "$build_dir"
    printf 'bundle_id=%s\n' "$bundle_id"
    printf 'rootfs_updated=1\n'
    printf 'next_manager_command=%s/scripts/experiment/%s launch --firesim-root %s\n' \
        "$workspace" "$program" "$firesim_root"
    printf 'guest_command=/root/workspace/3rd_llama.cpp/scripts/experiment/%s run --expected-bundle-id %s\n' \
        "$program" "$bundle_id"
}

launch_manager() {
    local firesim_root=/home/alveo/firesim
    local manager_setup
    local caller_directory

    while (($# > 0)); do
        case $1 in
            --firesim-root)
                (($# >= 2)) || die_usage 'missing value for --firesim-root'
                firesim_root=$2
                shift 2
                ;;
            *)
                die_usage "unknown launch option: $1"
                ;;
        esac
    done
    [[ $firesim_root = /* ]] || die_usage '--firesim-root must be absolute'
    firesim_root=$(absolute_dir "$firesim_root")
    manager_setup="$firesim_root/sourceme-manager.sh"
    [[ -f $manager_setup ]] || die_usage "missing manager setup: $manager_setup"

    printf 'manager_step=source_firesim\n'
    caller_directory=$PWD
    cd -- "$firesim_root"
    set +u
    # shellcheck disable=SC1090
    source ./sourceme-manager.sh --skip-ssh-setup
    set -u
    cd -- "$caller_directory"
    [[ ${FIRESIM_SOURCED:-0} == 1 ]] || die 'FireSim manager environment was not sourced'
    export PATH="$firesim_root/.conda-env/bin:$PATH"
    command -v firesim >/dev/null 2>&1 || die 'firesim command is unavailable'
    printf 'manager_step=launch_workload\n'
    (
        cd -- "$firesim_root/deploy"
        firesim runworkload
    )
}

run_manager_all() {
    local firesim_root=/home/alveo/firesim
    local index
    local -a prepare_arguments=("$@")

    for ((index = 0; index < ${#prepare_arguments[@]}; ++index)); do
        if [[ ${prepare_arguments[$index]} == --firesim-root ]]; then
            (($((index + 1)) < ${#prepare_arguments[@]})) ||
                die_usage 'missing value for --firesim-root'
            firesim_root=${prepare_arguments[$((index + 1))]}
            break
        fi
    done
    prepare_manager "${prepare_arguments[@]}"
    launch_manager --firesim-root "$firesim_root"
}

run_guest() {
    local workspace=/root/workspace/3rd_llama.cpp
    local build_name=build-riscv-rmd-cpu-npu
    local model=/root/workspace/llama.cpp/models/gpt2.Q8_0.gguf
    local output_root=/root/output
    local run_id
    local expected_bundle_id=
    local runs_per_backend=1
    local prompt='The quick brown fox jumps over the lazy dog.'
    local build_dir
    local binary
    local cache
    local build_manifest
    local run_dir
    local archive
    local temp_root
    local start_utc
    local end_utc
    local model_hash
    local binary_hash
    local library_hash
    local libgomp_hash
    local cache_hash
    local source_head source_head_tree source_worktree_status source_worktree_diff_hash
    local gemmini_header_path gemmini_header_hash routes_test_hash
    local recomputed_bundle_id
    local cpu_count=0
    local npu_count=0
    local first_failure=0
    local comparison_status=valid
    local validation_status=valid
    local artifact_copy_failed=0
    local archive_status
    local label
    local backend
    local run_number
    local sequence_position=0
    local item
    local stdout_reference=
    local proof_reference=
    local route_test_status=0

    run_id=$(date -u +%Y%m%d-%H%M%S)
    while (($# > 0)); do
        case $1 in
            --workspace)
                (($# >= 2)) || die_usage 'missing value for --workspace'
                workspace=$2
                shift 2
                ;;
            --build-dir)
                (($# >= 2)) || die_usage 'missing value for --build-dir'
                build_name=$2
                shift 2
                ;;
            --model)
                (($# >= 2)) || die_usage 'missing value for --model'
                model=$2
                shift 2
                ;;
            --output-root)
                (($# >= 2)) || die_usage 'missing value for --output-root'
                output_root=$2
                shift 2
                ;;
            --run-id)
                (($# >= 2)) || die_usage 'missing value for --run-id'
                run_id=$2
                shift 2
                ;;
            --expected-bundle-id)
                (($# >= 2)) || die_usage 'missing value for --expected-bundle-id'
                expected_bundle_id=$2
                shift 2
                ;;
            --runs-per-backend)
                (($# >= 2)) || die_usage 'missing value for --runs-per-backend'
                runs_per_backend=$2
                shift 2
                ;;
            --prompt)
                (($# >= 2)) || die_usage 'missing value for --prompt'
                prompt=$2
                shift 2
                ;;
            *)
                die_usage "unknown run option: $1"
                ;;
        esac
    done

    [[ $workspace = /* ]] || die_usage '--workspace must be absolute'
    [[ $model = /* ]] || die_usage '--model must be absolute'
    [[ $output_root = /* ]] || die_usage '--output-root must be absolute'
    [[ $build_name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
        die_usage 'unsafe --build-dir'
    [[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
        die_usage 'unsafe --run-id'
    [[ $expected_bundle_id =~ ^[0-9a-f]{64}$ ]] ||
        die_usage '--expected-bundle-id must be the SHA256 printed by prepare'
    [[ $runs_per_backend =~ ^[1-9][0-9]*$ ]] ||
        die_usage '--runs-per-backend must be positive'
    [[ -d $workspace ]] || die_usage "workspace does not exist: $workspace"
    [[ -f $model ]] || die_usage "model does not exist: $model"

    workspace=$(absolute_dir "$workspace")
    model=$(absolute_file "$model")
    build_dir="$workspace/$build_name"
    binary="$build_dir/bin/llama-cli"
    cache="$build_dir/CMakeCache.txt"
    build_manifest="$build_dir/experiment-build-manifest.txt"
    [[ -x $binary ]] || die_usage "missing executable binary: $binary"
    [[ -f $build_dir/bin/libggml-gemmini.so ]] ||
        die_usage "missing project library: $build_dir/bin/libggml-gemmini.so"
    [[ -f $build_dir/bin/libgomp.so.1 ]] ||
        die_usage "missing RISC-V OpenMP runtime: $build_dir/bin/libgomp.so.1"
    [[ -f $cache ]] || die_usage "missing CMake cache: $cache"
    [[ -f $build_manifest ]] ||
        die_usage "missing experiment build manifest: $build_manifest"
    cache_has "$cache" GGML_GEMMINI_ENABLE_RMD 'ON|1|TRUE' ||
        die_usage 'binary is not an RMD-enabled build'
    cache_has "$cache" GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE 'ON|1|TRUE' ||
        die_usage 'binary does not allow CPU/NPU runtime selection'
    cache_has "$cache" GGML_GEMMINI_OPTION WS ||
        die_usage 'binary does not keep the main Gemmini backend WS'
    binary_hash=$(hash_value "$binary")
    library_hash=$(hash_value "$build_dir/bin/libggml-gemmini.so")
    libgomp_hash=$(hash_value "$build_dir/bin/libgomp.so.1")
    cache_hash=$(hash_value "$cache")
    source_head=$(manifest_value "$build_manifest" source_head)
    source_head_tree=$(manifest_value "$build_manifest" source_head_tree)
    source_worktree_status=$(manifest_value "$build_manifest" source_worktree_status)
    source_worktree_diff_hash=$(manifest_value "$build_manifest" source_worktree_diff_sha256)
    gemmini_header_path=$(manifest_value "$build_manifest" gemmini_header_path)
    gemmini_header_hash=$(manifest_value "$build_manifest" gemmini_header_sha256)
    routes_test_hash=$(manifest_value "$build_manifest" rmd_routes_test_sha256)
    [[ -n $source_head && -n $source_head_tree && -n $source_worktree_diff_hash && -n $gemmini_header_path && -n $gemmini_header_hash && -n $routes_test_hash ]] || die 'build manifest lacks full runtime identity'
    [[ $(manifest_value "$build_manifest" bundle_id) == "$expected_bundle_id" ]] ||
        die 'guest rootfs bundle ID does not match prepare output'
    recomputed_bundle_id=$(manifest_payload_hash "$build_manifest")
    [[ $recomputed_bundle_id == "$expected_bundle_id" ]] ||
        die 'guest build manifest payload does not match its bundle ID'
    [[ $(manifest_value "$build_manifest" binary_sha256) == "$binary_hash" ]] ||
        die 'runtime binary does not match the prepared build manifest'
    [[ $(manifest_value "$build_manifest" gemmini_library_sha256) == "$library_hash" ]] ||
        die 'runtime Gemmini library does not match the prepared build manifest'
    [[ $(manifest_value "$build_manifest" libgomp_sha256) == "$libgomp_hash" ]] ||
        die 'runtime libgomp does not match the prepared build manifest'
    [[ $(manifest_value "$build_manifest" cmake_cache_sha256) == "$cache_hash" ]] ||
        die 'runtime CMake cache does not match the prepared build manifest'

    mkdir -p -- "$output_root"
    output_root=$(absolute_dir "$output_root")
    run_dir="$output_root/$run_id"
    archive="$output_root/$run_id.tar.gz"
    [[ ! -e $run_dir ]] || die "result directory already exists: $run_dir"
    [[ ! -e $archive ]] || die "result archive already exists: $archive"
    mkdir -- "$run_dir"
    mkdir -- "$run_dir/cpu" "$run_dir/npu"
    cp -- "$build_manifest" "$run_dir/experiment-build-manifest.txt"
    cp -- "$build_dir/bin/test-gemmini-exsia" "$run_dir/test-gemmini-exsia"
    cp -- "$gemmini_header_path" "$run_dir/gemmini.h"
    mkdir -- "$run_dir/rmd-routes-test"
    route_test_command=("$build_dir/bin/test-gemmini-exsia" --case=rmd-routes)
    printf 'LD_LIBRARY_PATH=%s\n' "$build_dir/bin" >"$run_dir/rmd-routes-test/environment.txt"
    printf -v route_test_command_escaped '%q ' "${route_test_command[@]}"
    printf 'LD_LIBRARY_PATH=%q %s\n' "$build_dir/bin" "${route_test_command_escaped% }" >"$run_dir/rmd-routes-test/command.txt"
    set +e
    (cd -- "$build_dir" && LD_LIBRARY_PATH="$build_dir/bin" "${route_test_command[@]}") \
        >"$run_dir/rmd-routes-test/stdout.txt" 2>"$run_dir/rmd-routes-test/stderr.txt"
    route_test_status=$?
    set -e
    printf '%s\n' "$route_test_status" >"$run_dir/rmd-routes-test/exit-status.txt"
    if ((route_test_status == 0)); then
        grep -Eq '^RMD_STAGE ' "$run_dir/rmd-routes-test/stdout.txt" &&
            grep -F --quiet 'RMD_ORACLE ' "$run_dir/rmd-routes-test/stdout.txt" &&
            grep -F --quiet 'RMD_ORACLE direct' "$run_dir/rmd-routes-test/stdout.txt" &&
            grep -F --quiet 'RMD_ORACLE radix' "$run_dir/rmd-routes-test/stdout.txt" &&
            grep -F --quiet 'RMD_ORACLE packet-scalar' "$run_dir/rmd-routes-test/stdout.txt" &&
            grep -F --quiet 'RMD_ORACLE WS' "$run_dir/rmd-routes-test/stdout.txt" &&
            grep -F --quiet 'PASS: case=rmd-routes' "$run_dir/rmd-routes-test/stdout.txt" ||
            route_test_status=1
    fi
    printf '%s\n' "$route_test_status" >"$run_dir/rmd-routes-test/validated-status.txt"
    : >"$run_dir/run-order.txt"

    temp_root=$(mktemp -d "${TMPDIR:-/tmp}/gemmini-rmd-cpu-npu.XXXXXX")
    guest_temp_root=$temp_root
    trap cleanup_guest_temp EXIT
    start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    model_hash=$(hash_value "$model")
    printf '%s  %s\n' "$model_hash" "$model" >"$run_dir/model.sha256"
    printf '%s  %s\n' "$binary_hash" "$binary" >"$run_dir/binary.sha256"
    : >"$run_dir/libraries.sha256"
    for item in "$build_dir/bin"/libllama.so* "$build_dir/bin"/libggml*.so* \
                "$build_dir/bin"/libgomp.so*; do
        [[ -f $item ]] || continue
        printf '%s  %s\n' "$(hash_value "$item")" "$item" \
            >>"$run_dir/libraries.sha256"
    done

    record_validation_failure() {
        [[ $validation_status != valid ]] || validation_status=$1
    }

    validate_debug_log() {
        local variant=$1
        local debug_log=$2
        local telemetry_records=0
        local pipeline_records=0
        local line

        [[ -f $debug_log ]] || return 1
        while IFS= read -r line; do
            if [[ $line == *'"record_type":"RMD_BACKEND_TELEMETRY"'* ]]; then
                telemetry_records=$((telemetry_records + 1))
                [[ $line == *'"schema":"gemmini.rmd.telemetry"'* &&
                   $line == *'"version":1'* &&
                   $line == *"\"runtime_bundle_id\":\"$library_hash\""* &&
                   $line == *"\"model_id\":\"$model_hash\""* &&
                   $line == *'"source":"environment"'* &&
                   $line == *'"units":"cycles"'* &&
                   $line == *'"work":true'* &&
                   $line == *'"invocation_total":'* &&
                   $line == *'"dispatch":{'* &&
                   $line == *'"timing":{'* &&
                   $line == *'"backend_service":'* &&
                   $line == *'"residual_total":'* &&
                   $line == *'"geometry":{'* &&
                   $line == *'"stripes":[{'* &&
                   $line =~ \"input_hash\":\"[0-9a-f]{16}\" &&
                   $line =~ \"correction_hash\":\"[0-9a-f]{16}\" &&
                   $line =~ \"output_hash\":\"[0-9a-f]{16}\" ]] || return 1
                if [[ $variant == cpu ]]; then
                    [[ $line == *'"backend":"cpu_direct"'* &&
                       $line =~ \"direct_events\":[1-9][0-9]* &&
                       $line =~ \"direct_calls\":[1-9][0-9]* &&
                       $line == *'"packet_calls":0'* &&
                       $line == *'"ws_calls":0'* &&
                       $line == *'"packet_count":0'* &&
                       $line =~ \"correction_nonzero_count\":[1-9][0-9]* &&
                       $line =~ \"residual_total\":[1-9][0-9]* ]] || return 1
                else
                    [[ $line == *'"backend":"gemmini_ws_compact"'* &&
                       $line == *'"direct_events":0'* &&
                       $line == *'"direct_calls":0'* &&
                       $line =~ \"packet_calls\":[1-9][0-9]* &&
                       $line =~ \"ws_calls\":[1-9][0-9]* &&
                       $line =~ \"packet_count\":[1-9][0-9]* &&
                       $line =~ \"correction_nonzero_count\":[1-9][0-9]* &&
                       $line =~ \"residual_total\":[1-9][0-9]* ]] || return 1
                fi
            elif [[ $line == *'"record_type":"PIPELINE_STRIPE_SUMMARY"'* ]]; then
                pipeline_records=$((pipeline_records + 1))
                [[ $line == *'"backend_route":"gemmini_ws"'* &&
                   $line == *'"schedule":"matmul-rmd-overlap"'* ]] || return 1
            fi
        done <"$debug_log"
        ((telemetry_records > 0 && pipeline_records > 0))
    }

    extract_proofs() {
        local debug_log=$1
        local proof_file=$2
        local line
        local token
        local value

        : >"$proof_file"
        while IFS= read -r line; do
            [[ $line == *'"record_type":"RMD_BACKEND_TELEMETRY"'* ]] || continue
            while IFS= read -r token; do
                case $token in
                    *'"stripe_id":'*) value=${token#*'"stripe_id":'}; value=${value%%\}*}; printf 'stripe_id=%s\n' "$value" ;;
                    *'"row_begin":'*) value=${token#*'"row_begin":'}; value=${value%%\}*}; printf 'row_begin=%s\n' "$value" ;;
                    *'"row_end":'*) value=${token#*'"row_end":'}; value=${value%%\}*}; printf 'row_end=%s\n' "$value" ;;
                    *'"input_hash":"'*) value=${token#*'"input_hash":"'}; value=${value%%\"*}; printf 'input_hash=%s\n' "$value" ;;
                    *'"correction_hash":"'*) value=${token#*'"correction_hash":"'}; value=${value%%\"*}; printf 'correction_hash=%s\n' "$value" ;;
                    *'"output_hash":"'*) value=${token#*'"output_hash":"'}; value=${value%%\"*}; printf 'output_hash=%s\n' "$value" ;;
                esac
            done < <(printf '%s\n' "$line" | tr ',' '\n')
        done <"$debug_log" >"$proof_file"
        [[ -s $proof_file ]]
    }

    run_variant() {
        local variant=$1
        local selector=$2
        local number=$3
        local result_dir="$run_dir/$variant/run-$number"
        local work_dir="$temp_root/$variant-$number"
        local debug_log
        local status
        local -a command

        mkdir -- "$result_dir" "$work_dir"
        mkdir -- "$result_dir/output" "$work_dir/output"
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
        printf '%s\n' "${command_escaped% }" >"$result_dir/command.txt"
        {
            printf 'RMD_EXPERIMENT_LABEL=%s\n' "$variant"
            printf 'RMD_EXPERIMENT_RUN=%s\n' "$number"
            printf 'GEMMINI_MATMUL_MODE=STRIPE_PIPELINE\n'
            printf 'GEMMINI_RMD_BACKEND=%s\n' "$selector"
            printf 'GGML_GEMMINI_RUNTIME_BUNDLE_ID=%s\n' "$library_hash"
            printf 'GGML_GEMMINI_MODEL_ID=%s\n' "$model_hash"
            printf 'LD_LIBRARY_PATH=<unset>\n'
            printf 'LD_PRELOAD=<unset>\n'
        } >"$result_dir/environment.txt"

        printf 'starting_backend=%s run=%s\n' "$variant" "$number" >&2
        if (
            cd -- "$work_dir" || exit
            unset LD_LIBRARY_PATH
            unset LD_PRELOAD
            export RMD_EXPERIMENT_LABEL="$variant"
            export RMD_EXPERIMENT_RUN="$number"
            export GEMMINI_MATMUL_MODE=STRIPE_PIPELINE
            export GEMMINI_RMD_BACKEND="$selector"
            export GGML_GEMMINI_RUNTIME_BUNDLE_ID="$library_hash"
            export GGML_GEMMINI_MODEL_ID="$model_hash"
            "${command[@]}"
        ) >"$result_dir/stdout.txt" 2>"$result_dir/stderr.txt"; then
            status=0
        else
            status=$?
        fi
        if ! cp -a -- "$work_dir/output/." "$result_dir/output/"; then
            artifact_copy_failed=1
        fi
        printf '%s\n' "$status" >"$result_dir/exit-status.txt"
        printf 'completed_backend=%s run=%s exit_status=%s\n' \
            "$variant" "$number" "$status" >&2

        if ((status != 0 && first_failure == 0)); then
            first_failure=$status
        fi
        debug_log="$result_dir/output/log/debug-log.jsonl"
        if ((status == 0 && artifact_copy_failed == 0)); then
            if ! validate_debug_log "$variant" "$debug_log"; then
                record_validation_failure "invalid_${variant}_telemetry"
            elif ! extract_proofs "$debug_log" "$result_dir/proofs.txt"; then
                record_validation_failure "invalid_${variant}_proofs"
            fi
        fi

        if [[ -z $stdout_reference ]]; then
            stdout_reference="$result_dir/stdout.txt"
        elif ! cmp -s -- "$stdout_reference" "$result_dir/stdout.txt"; then
            record_validation_failure invalid_output_mismatch
        fi
    }

    if ((route_test_status == 0)); then
    while ((cpu_count < runs_per_backend || npu_count < runs_per_backend)); do
        for label in cpu npu npu cpu; do
            if [[ $label == cpu ]]; then
                ((cpu_count >= runs_per_backend)) && continue
                cpu_count=$((cpu_count + 1))
                backend=CPU
                run_number=$cpu_count
            else
                ((npu_count >= runs_per_backend)) && continue
                npu_count=$((npu_count + 1))
                backend=WS
                run_number=$npu_count
            fi
            sequence_position=$((sequence_position + 1))
            printf 'position=%s backend=%s run=%s selector=%s\n' \
                "$sequence_position" "$label" "$run_number" "$backend" \
                >>"$run_dir/run-order.txt"
            run_variant "$label" "$backend" "$run_number"
        done
    done
    else
        first_failure=$route_test_status
        validation_status=invalid_rmd_routes_test
    fi

    if ((first_failure == 0 && artifact_copy_failed == 0)) &&
        [[ $validation_status == valid ]]; then
        for item in "$run_dir"/cpu/run-*/proofs.txt "$run_dir"/npu/run-*/proofs.txt; do
            if [[ ! -f $item ]]; then
                record_validation_failure invalid_missing_proofs
                break
            elif [[ -z $proof_reference ]]; then
                proof_reference=$item
            elif ! cmp -s -- "$proof_reference" "$item"; then
                record_validation_failure invalid_proof_mismatch
                break
            fi
        done
    fi
    if ((first_failure != 0)); then
        comparison_status=not_evaluated_variant_failure
    elif ((artifact_copy_failed != 0)); then
        comparison_status=invalid_artifact_copy
    else
        comparison_status=$validation_status
    fi

    end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    {
        printf 'run_id=%s\n' "$run_id"
        printf 'start_utc=%s\n' "$start_utc"
        printf 'end_utc=%s\n' "$end_utc"
        printf 'workspace=%s\n' "$workspace"
        printf 'build_dir=%s\n' "$build_dir"
        printf 'model=%s\n' "$model"
        printf 'model_sha256=%s\n' "$model_hash"
        printf 'binary_sha256=%s\n' "$binary_hash"
        printf 'gemmini_library_sha256=%s\n' "$library_hash"
        printf 'libgomp_sha256=%s\n' "$libgomp_hash"
        printf 'cmake_cache_sha256=%s\n' "$cache_hash"
        printf 'expected_bundle_id=%s\n' "$expected_bundle_id"
        printf 'source_head=%s\n' "$source_head"
        printf 'source_head_tree=%s\n' "$source_head_tree"
        printf 'source_worktree_status=%s\n' "$source_worktree_status"
        printf 'source_worktree_diff_sha256=%s\n' "$source_worktree_diff_hash"
        printf 'gemmini_header_path=%s\n' "$gemmini_header_path"
        printf 'gemmini_header_sha256=%s\n' "$gemmini_header_hash"
        printf 'rmd_routes_test_sha256=%s\n' "$routes_test_hash"
        printf 'git_sha=%s\n' "$source_head"
        printf 'cpu_selector=CPU\n'
        printf 'npu_selector=WS\n'
        printf 'main_gemmini_option=WS\n'
        printf 'matmul_mode=STRIPE_PIPELINE\n'
        printf 'schedule=ABBA\n'
        printf 'cpu_runs=%s\n' "$cpu_count"
        printf 'npu_runs=%s\n' "$npu_count"
        printf 'rmd_routes_test_status=%s\n' "$route_test_status"
        printf 'first_failure=%s\n' "$first_failure"
        printf 'artifact_copy_failed=%s\n' "$artifact_copy_failed"
        printf 'validation_status=%s\n' "$validation_status"
        printf 'comparison_status=%s\n' "$comparison_status"
    } >"$run_dir/manifest.txt"

    if tar c -a -f "$archive" -C "$output_root" "$run_id"; then
        archive_status=0
    else
        archive_status=$?
    fi
    if ((archive_status != 0)); then
        rm -f -- "$archive"
        printf '%s: failed to create archive: %s\n' "$program" "$archive" >&2
        exit "$archive_status"
    fi

    printf 'result_dir=%s\n' "$run_dir"
    printf 'archive=%s\n' "$archive"
    printf 'cpu_runs=%s\n' "$cpu_count"
    printf 'npu_runs=%s\n' "$npu_count"
    printf 'rmd_routes_test_status=%s\n' "$route_test_status"
    printf 'comparison_status=%s\n' "$comparison_status"
    if ((first_failure != 0)); then
        exit "$first_failure"
    fi
    [[ $comparison_status == valid ]] || exit 3
}

case ${1:-} in
    manager)
        shift
        run_manager_all "$@"
        ;;
    prepare)
        shift
        prepare_manager "$@"
        ;;
    launch)
        shift
        launch_manager "$@"
        ;;
    run)
        shift
        run_guest "$@"
        ;;
    -h|--help)
        usage
        ;;
    '')
        die_usage 'missing command'
        ;;
    *)
        die_usage "unknown command: $1"
        ;;
esac
