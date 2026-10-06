#!/bin/bash

im2p_host_jobs() {
  if [[ "$(uname -s)" == Darwin ]]; then
    sysctl -n hw.logicalcpu
  elif command -v nproc >/dev/null 2>&1; then
    nproc
  else
    getconf _NPROCESSORS_ONLN
  fi
}

im2p_host_libomp_prefix() {
  local brew_bin prefix
  brew_bin=$(command -v brew || true)
  for prefix in "${brew_bin%/bin/brew}" /opt/homebrew /usr/local; do
    if [[ -n "$prefix" && -d "$prefix/opt/libomp" ]]; then
      printf '%s\n' "$prefix/opt/libomp"
      return
    fi
  done
}

im2p_configure_build() {
  local build_dir=$1
  shift
  cmake -B "$build_dir" -S "$SCRIPT_ROOT" \
    -U 'GGML_*' -U 'IM2P_*' -U 'LLAMA_*' -U 'LOG_*' -U 'CYCLE_*' \
    -U 'BUILD_SHARED_LIBS' -U 'CMAKE_TOOLCHAIN_FILE' \
    -U 'CMAKE_PREFIX_PATH' -U 'OpenMP_ROOT' "$@"
}

# Shared provisioning for native host builds. Callers define the selected
# frontend identity; this helper only chooses how much of the cache to warm.
im2p_provision_host_artifacts() {
  local sim_root=$1
  local gemmini_root=$2
  local default_jobs=$3
  local activation_bits=$4
  local weight_bits=$5
  local dim=$6
  local block_size=$7
  local implementation=${8:-LEGACY_BSV}
  local artifact_set=$IM2P_ARTIFACT_SET_DEFAULT
  local cache_jobs
  local target

  case "$artifact_set" in
    SELECTED)
      target=gemmini-frontend-real-lib
      cache_jobs=$IM2P_CACHE_JOBS_DEFAULT
      ;;
    ALL_MATCHED)
      target=gemmini-frontend-real-lib-all
      cache_jobs=$IM2P_CACHE_JOBS_DEFAULT
      ;;
    *)
      printf '%s\n' \
        "IM2P_ARTIFACT_SET must be SELECTED or ALL_MATCHED, got '$artifact_set'" >&2
      return 2
      ;;
  esac

  if [[ ! "$cache_jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' \
      "IM2P_CACHE_JOBS must be a positive integer, got '$cache_jobs'" >&2
    return 2
  fi
  if [[ -z "$sim_root" ]]; then
    printf '%s\n' 'IM2P_SIM_ROOT is required for IM2P_SIM host provisioning' >&2
    return 2
  fi

  local sim_root_abs
  sim_root_abs="$(cd "$sim_root" && pwd)"
  local make_args=(
    IM2P_CACHE_JOBS="$cache_jobs" \
    BUILD_DIR="${IM2P_SIM_BUILD_DIR_DEFAULT:-$sim_root_abs/build}" \
    GEMMINI_ROOT="$gemmini_root" \
    IM2P_SIM_IMPLEMENTATION="$implementation" \
    IM2P_ACTIVATION_BITS="$activation_bits" \
    IM2P_WEIGHT_BITS="$weight_bits" \
    IM2P_DIM="$dim" \
    GEMMINI_FRONTEND_ACTIVATION_BITS="$activation_bits" \
    GEMMINI_FRONTEND_WEIGHT_BITS="$weight_bits" \
    GEMMINI_FRONTEND_DIM="$dim" \
    GEMMINI_FRONTEND_BLOCK_SIZE="$block_size")
  make -C "$sim_root_abs" -j"$cache_jobs" "${make_args[@]}" "$target"
}

# Resolve once before provisioning and configure. Python emits only shell-quoted
# assignments; no argument is executed as shell code.
im2p_resolve_build_options() {
  local build_dir=$1
  local platform=$2
  shift 2
  local name resolved
  local defaults=()
  while IFS= read -r name; do
    [[ "$name" == *_DEFAULT ]] || continue
    defaults+=("${name%_DEFAULT}=${!name}")
  done < <(compgen -A variable)
  resolved="$(python3 -B "$SCRIPT_ROOT/scripts/im2p-build-options.py" \
    "$build_dir" "$platform" "${defaults[@]}" -- "$@")" || return $?
  eval "$resolved"
}
