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
  make -C "$sim_root_abs" -j"$cache_jobs" \
    IM2P_CACHE_JOBS="$cache_jobs" \
    GEMMINI_ROOT="$gemmini_root" \
    IM2P_ACTIVATION_BITS="$activation_bits" \
    IM2P_WEIGHT_BITS="$weight_bits" \
    IM2P_DIM="$dim" \
    GEMMINI_FRONTEND_ACTIVATION_BITS="$activation_bits" \
    GEMMINI_FRONTEND_WEIGHT_BITS="$weight_bits" \
    GEMMINI_FRONTEND_DIM="$dim" \
    GEMMINI_FRONTEND_BLOCK_SIZE="$block_size" \
    "$target"
}
