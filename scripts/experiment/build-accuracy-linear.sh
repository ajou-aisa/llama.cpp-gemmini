#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
output=${ACCURACY_LINEAR_OUTPUT_DIR:-"$root/build-potal-cuda/accuracy-linear"}
gemmini=${GEMMINI_SW_PATH:-"$root/../RISC-V-DynDNN-gemmini-include"}
mkdir -p "$output"
includes=(-I"$root/include" -I"$root/ggml/include" -I"$root/common"
          -I"$root/ggml/src/ggml-gemmini" -I"$root/ggml/src/ggml-gemmini-utils/include"
          -I"$gemmini" -I"$gemmini/include"
          -I"$root/ggml/src/ggml-cuda/cpu-exact")
for bits in 4 8; do
    if [[ $bits == 4 ]]; then
        build=${ACCURACY_POTAL4_BUILD:-"$root/build-potal-cuda/top3-a4-rmdon-attention"}
    else
        build=${ACCURACY_POTAL8_BUILD:-"$root/build-potal-cuda/top3-a8-rmdon-attention"}
    fi
    c++ -std=c++17 -O3 -Wall -Wextra -fopenmp -fno-fast-math \
        -DGGML_GEMMINI_CUDA_CPU_EXACT=1 -DGGML_GEMMINI_CUDA_RTN_ATTENTION=1 \
        -DEVALUATION_EXACT_COMPUTE='"INT"' -DEVALUATION_EXACT_ACTIVATION='"EXSIA"' \
        -DEVALUATION_EXACT_BITS="$bits" -DEVALUATION_EXACT_DIM=32 \
        "${includes[@]}" -I"$build/generated" "$root/tools/accuracy-linear/score.cpp" \
        -L"$build/bin" -Wl,-rpath,"$build/bin" -lllama -lggml-gemmini -lggml -lggml-base -pthread \
        -o "$output/native-score-potal$bits"
    sha256sum "$root/tools/accuracy-linear/score.cpp" "$root/tools/accuracy-fast/ordered-logprobs.hpp" \
        "$root/tools/perplexity/perplexity-cuda-cpu-exact.hpp" \
        "$root/scripts/experiment/build-accuracy-linear.sh" "$output/native-score-potal$bits" \
        "$build/bin/"*.so* > "$output/native-score-potal$bits.sha256"
done
