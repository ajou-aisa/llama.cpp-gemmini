#!/bin/bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_JOBS=${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN)}
static_args=(-DBUILD_SHARED_LIBS=ON)
if [[ "${1:-}" == static ]]; then
  static_args=(-DBUILD_SHARED_LIBS=OFF)
  BUILD_DIR=${BUILD_DIR:-build-riscv-static}
  shift
else
  BUILD_DIR=${BUILD_DIR:-build-riscv}
fi
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  printf 'Usage: %s [static] [--dry-run] [-DNAME[:TYPE]=value ...]\n' "${0##*/}"
  exit 0
fi
dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
configure=(cmake -S "$SCRIPT_ROOT" -B "$BUILD_DIR" -G Ninja --preset "${NPU_PRESET:-potal-int8-d${POTAL_EXPECTED_DIM:-${POTAL_EXPECTED_DIM_DEFAULT:-16}}-rocc}" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLOG_DEBUG="${LOG_DEBUG:-${LOG_DEBUG_DEFAULT:-1}}" -DLOG_CYCLE="${LOG_CYCLE:-${LOG_CYCLE_DEFAULT:-1}}" -DGGML_RVV=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF "${static_args[@]}" "$@")
if (( dry_run )); then
  printf '%q ' "${configure[@]}"
  printf '\n'
  exit 0
fi
"${configure[@]}"
cmake --build "$BUILD_DIR" --target llama-cli llama-perplexity llama-quantize -j"$BUILD_JOBS"
source "$SCRIPT_ROOT/scripts/stage-riscv-runtime.sh"
stage_riscv_openmp_runtime "$BUILD_DIR"
