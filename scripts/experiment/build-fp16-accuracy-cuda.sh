#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
output=${ACCURACY_FP16_OUTPUT_DIR:-"$root/build-potal-cuda/accuracy-fp16"}
build=${ACCURACY_RTN_BUILD:-"$root/build-potal-cuda/rtn"}
mkdir -p "$output"
c++ -std=c++17 -O3 -Wall -Wextra \
    -I"$root/include" -I"$root/ggml/include" -I"$root/common" \
    -I"$build/generated" "$root/tools/accuracy-fp16-cuda/score.cpp" -L"$build/bin" \
    -Wl,-rpath,"$build/bin" -lllama -lggml-gemmini -lggml -lggml-base -pthread \
    -o "$output/native-score-fp16-cuda"
sha256sum "$root/tools/accuracy-fp16-cuda/score.cpp" \
    "$root/scripts/experiment/build-fp16-accuracy-cuda.sh" \
    "$root/tools/perplexity/perplexity-cuda-rtn.hpp" \
    "$output/native-score-fp16-cuda" "$build/bin/"*.so* \
    > "$output/native-score-fp16-cuda.sha256"
