#!/bin/bash
set -euo pipefail

# Build Gemmini in CPU fallback mode on x86; outputs live in build/bin
BUILD_DIR=${BUILD_DIR:-build}
LLAMA_CURL_DEFAULT=${LLAMA_CURL_DEFAULT:-OFF} # disable libcurl requirement on local x86 unless explicitly enabled
CMAKE_BUILD_TYPE_DEFAULT=${CMAKE_BUILD_TYPE_DEFAULT:-Debug} # Debug | Release
GGML_NATIVE_DEFAULT=${GGML_NATIVE_DEFAULT:-OFF} # ON | OFF

cmake -B "$BUILD_DIR" -S . \
  -DGGML_GEMMINI=ON \
  -DGGML_GEMMINI_FORCE_GGML_OUTPUT=OFF \
  -DGGML_GEMMINI_FORCE_GGML_ALL=OFF \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE_DEFAULT}" \
  -DGGML_NATIVE="${GGML_NATIVE_DEFAULT}" \
  -DLLAMA_CURL="${LLAMA_CURL_DEFAULT}" \
  "$@"

cmake --build "$BUILD_DIR" --target llama-cli llama-perplexity -j"$(nproc)"
