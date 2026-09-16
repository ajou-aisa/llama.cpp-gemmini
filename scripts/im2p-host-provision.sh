#!/bin/bash

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
  local artifact_set=${IM2P_ARTIFACT_SET:-SELECTED}
  local cache_jobs
  local target

  case "$artifact_set" in
    SELECTED)
      target=gemmini-frontend-real-lib
      cache_jobs=${IM2P_CACHE_JOBS:-$default_jobs}
      ;;
    ALL_MATCHED)
      target=gemmini-frontend-real-lib-all
      cache_jobs=${IM2P_CACHE_JOBS:-1}
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
  resolved="$(python3 "$SCRIPT_ROOT/scripts/im2p-build-options.py" \
    "$build_dir" "$platform" "${defaults[@]}" -- "$@")" || return $?
  eval "$resolved"
  if [[ "$GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT" == FPGA_UART &&
        -n "${GGML_GEMMINI_FPGA_SIM_MANIFEST_DEFAULT:-}" ]]; then
    printf '%s\n' \
      'GGML_GEMMINI_FPGA_SIM_MANIFEST is invalid for FPGA_UART; physical external executor uses no simulator archive' >&2
    return 2
  fi
  if [[ "$GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT" == FPGA_UART &&
        "$IM2P_BUILD_DRY_RUN" != 1 && "$(uname -s)" == Linux ]]; then
    local machine
    machine="$(uname -m)"
    case "$platform:$machine" in
      build-x86.sh:x86_64|build-arm64.sh:aarch64|build-arm64.sh:arm64|build-arm64-cpu.sh:aarch64|build-arm64-cpu.sh:arm64) ;;
      *)
        printf 'FPGA_UART native script/host mismatch: %s on %s; use the matching script or direct CMake with target artifacts\n' "$platform" "$machine" >&2
        return 2
        ;;
    esac
  fi
}
