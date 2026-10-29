#!/bin/bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 2 || "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf 'Usage: bash %s functional|performance 16|32 [static] [--dry-run] [-DNAME[:TYPE]=value ...]\n' "${0##*/}"
  printf '%s\n' 'PoTal FireSim H3-7C defaults; generic build scripts are unchanged.'
  printf '%s\n' 'DIM is required. Each DIM/profile uses a separate BUILD_DIR by default.'
  printf '%s\n' 'Overrides use the existing command-line > environment > default-environment > script-default resolver.'
  [[ "${1:-}" == --help || "${1:-}" == -h ]] && exit 0
  exit 2
fi
profile=$1
dim=$2
shift 2
case "$profile" in
  functional) debug=1 ;;
  performance) debug=0 ;;
  *) printf 'Unknown PoTal FireSim profile: %s\n' "$profile" >&2; exit 2 ;;
esac
case "$dim" in
  16) header=gemmini_potal_params.h ;;
  32) header=gemmini_potal_d32_params.h ;;
  *) printf 'PoTal FireSim DIM must be 16 or 32, got %s\n' "$dim" >&2; exit 2 ;;
esac

export BUILD_DIR="${BUILD_DIR:-$SCRIPT_ROOT/build-potal-firesim-d$dim-$profile}"
export GGML_POTAL_DEFAULT="${GGML_POTAL_DEFAULT:-ON}"
export GGML_GEMMINI_DEFAULT="${GGML_GEMMINI_DEFAULT:-OFF}"
export GGML_BACKEND_DL_DEFAULT="${GGML_BACKEND_DL_DEFAULT:-OFF}"
export GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT="${GGML_GEMMINI_EXECUTION_BACKEND_DEFAULT:-HARDWARE}"
export GGML_GEMMINI_OPTION_DEFAULT="${GGML_GEMMINI_OPTION_DEFAULT:-WS}"
export GGML_GEMMINI_ACTIVATION_QUANT_DEFAULT="${GGML_GEMMINI_ACTIVATION_QUANT_DEFAULT:-EXSIA}"
export GGML_GEMMINI_ENABLE_RMD_DEFAULT="${GGML_GEMMINI_ENABLE_RMD_DEFAULT:-ON}"
export GGML_GEMMINI_DEFAULT_RMD_BACKEND_DEFAULT="${GGML_GEMMINI_DEFAULT_RMD_BACKEND_DEFAULT:-WS}"
export GGML_GEMMINI_DEFAULT_MATMUL_MODE_DEFAULT="${GGML_GEMMINI_DEFAULT_MATMUL_MODE_DEFAULT:-STRIPE_PIPELINE}"
export GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT="${GGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY_DEFAULT:-2}"
export GGML_POTAL_PAIRED_AUTO_DEFAULT="${GGML_POTAL_PAIRED_AUTO_DEFAULT:-ON}"
export GGML_GEMMINI_EXSIA_DEFAULT_MODE_DEFAULT="${GGML_GEMMINI_EXSIA_DEFAULT_MODE_DEFAULT:-SEQUENTIAL}"
export GGML_GEMMINI_ACTIVATION_BITS_DEFAULT="${GGML_GEMMINI_ACTIVATION_BITS_DEFAULT:-8}"
export GGML_GEMMINI_WEIGHT_BITS_DEFAULT="${GGML_GEMMINI_WEIGHT_BITS_DEFAULT:-8}"
export GGML_GEMMINI_BLOCK_SIZE_DEFAULT="${GGML_GEMMINI_BLOCK_SIZE_DEFAULT:-32}"
export CYCLE_SIM_DEFAULT="${CYCLE_SIM_DEFAULT:-0}" CYCLE_DETAIL_DEFAULT="${CYCLE_DETAIL_DEFAULT:-0}"
export POTAL_PLANNING_DIAGNOSTICS_DEFAULT="${POTAL_PLANNING_DIAGNOSTICS_DEFAULT:-OFF}"
export LOG_DEBUG_DEFAULT="${LOG_DEBUG_DEFAULT:-$debug}" LOG_CYCLE_DEFAULT="${LOG_CYCLE_DEFAULT:-1}"
export GGML_CPU_CYCLE_LOG_DEFAULT="${GGML_CPU_CYCLE_LOG_DEFAULT:-OFF}"
export GGML_GEMMINI_ENABLE_A4NKS_ATTENTION_DEFAULT="${GGML_GEMMINI_ENABLE_A4NKS_ATTENTION_DEFAULT:-OFF}"
export GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION_DEFAULT="${GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION_DEFAULT:-OFF}"
export POTAL_EXPECTED_DIM_DEFAULT="${POTAL_EXPECTED_DIM_DEFAULT:-$dim}"
export GGML_GEMMINI_DIM_DEFAULT="${GGML_GEMMINI_DIM_DEFAULT:-$dim}"
export POTAL_HW_PARAMS_DEFAULT="${POTAL_HW_PARAMS_DEFAULT:-$SCRIPT_ROOT/../../gemmini-rocc-tests/include/$header}"
export POTAL_HW_CONFIG_DEFAULT="${POTAL_HW_CONFIG_DEFAULT:-FireSimQuadRocketSinglePoTalD${dim}Config}"

# The target dimension is a contract, not an overridable geometry default.
# CMake compares it with the selected header and both software DIM settings.
exec bash "$SCRIPT_ROOT/build-riscv.sh" "$@" "-DPOTAL_TARGET_DIM:STRING=$dim" \
  "-DPOTAL_TARGET_PARAMS:FILEPATH=$SCRIPT_ROOT/../../gemmini-rocc-tests/include/$header"
