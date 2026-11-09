#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
output=${ACCURACY_OUTPUT_DIR:-"$root/build-potal-cuda/accuracy-quantized"}
gemmini=${GEMMINI_SW_PATH:-"$root/../RISC-V-DynDNN-gemmini-include"}
mkdir -p "$output"
includes=(-I"$root/include" -I"$root/ggml/include" -I"$root/common"
          -I"$root/ggml/src/ggml-gemmini" -I"$root/ggml/src/ggml-gemmini-utils/include"
          -I"$gemmini" -I"$gemmini/include"
          -I"$root/ggml/src/ggml-cuda/cpu-exact")
for profile in rtn potal4 potal8; do
    definitions=()
    build=${ACCURACY_RTN_BUILD:-"$root/build-potal-cuda/rtn"}
    if [[ "$profile" == potal4 || "$profile" == potal8 ]]; then
        bits=${profile#potal}
        if [[ "$bits" == 4 ]]; then
            build=${ACCURACY_POTAL4_BUILD:-"$root/build-potal-cuda/top3-a4-rmdon-attention"}
        else
            build=${ACCURACY_POTAL8_BUILD:-"$root/build-potal-cuda/top3-a8-rmdon-attention"}
        fi
        definitions=(-DACCURACY_POTAL=1 -DGGML_GEMMINI_CUDA_CPU_EXACT=1
                     -DGGML_GEMMINI_CUDA_RTN_ATTENTION=1
                     -DEVALUATION_EXACT_COMPUTE='"INT"' -DEVALUATION_EXACT_ACTIVATION='"EXSIA"'
                     -DEVALUATION_EXACT_BITS="$bits" -DEVALUATION_EXACT_DIM=32)
    fi
    [[ -f "$build/bin/libggml-gemmini.so" ]] || { echo "Missing backend: $build" >&2; exit 1; }
    c++ -std=c++17 -O3 -Wall -Wextra "${definitions[@]}" "${includes[@]}" \
        -I"$build/generated" "$root/tools/accuracy-quantized/score.cpp" -L"$build/bin" \
        -Wl,-rpath,"$build/bin" -lllama -lggml-gemmini -lggml -lggml-base -pthread \
        -o "$output/native-score-$profile"
    sha256sum "$root/tools/accuracy-quantized/score.cpp" "$output/native-score-$profile" \
        > "$output/native-score-$profile.sha256"
done
