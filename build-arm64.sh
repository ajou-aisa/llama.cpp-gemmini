#!/bin/bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR=${BUILD_DIR:-build-arm64}
BUILD_JOBS=${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN)}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  printf 'Usage: %s [--dry-run] [-DNAME[:TYPE]=value ...]\n' "${0##*/}"
  exit 0
fi
dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
configure=(cmake -S "$SCRIPT_ROOT" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLOG_DEBUG="${LOG_DEBUG:-${LOG_DEBUG_DEFAULT:-0}}" -DLOG_CYCLE="${LOG_CYCLE:-${LOG_CYCLE_DEFAULT:-0}}" -DGGML_GEMMINI="${GGML_GEMMINI:-OFF}" -DGGML_BACKEND_DL="${GGML_BACKEND_DL:-ON}" -DLLAMA_BUILD_TESTS="${LLAMA_BUILD_TESTS:-OFF}" "$@")
if (( dry_run )); then
  printf '%q ' "${configure[@]}"
  printf '\n'
  exit 0
fi
"${configure[@]}"
cmake --build "$BUILD_DIR" --target llama-cli llama-perplexity llama-quantize -j"$BUILD_JOBS"
