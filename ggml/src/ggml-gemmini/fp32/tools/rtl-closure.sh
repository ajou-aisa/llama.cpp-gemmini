#!/usr/bin/env bash
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
conda activate /home/alveo/firesim/.conda-env
set -euo pipefail
L=/home/alveo/firesim/target-design/chipyard/generators/potal/software/workspace/llama.cpp-potal
F=$L/ggml/src/ggml-gemmini/fp32
O=/media/alveo/LaCie/fp32-5c-d8/sw
export TMPDIR="$O/tmp"
case "$1" in
native)
    bash "$F/tools/offline.sh" native
    cmake --build "$O/native" -j 4 --target gemmini-fp32-rtl-vector-test
    "$O/native/bin/gemmini-fp32-rtl-vector-test" \
        /media/alveo/LaCie/fp32-5c-d8/hw/numeric-v2/vectors \
        /media/alveo/LaCie/fp32-5c-d8/hw/numeric/vectors
    "$O/native/bin/gemmini-fp32-block-test" 0 0 1,2,3,4
    "$O/native/bin/gemmini-fp32-attention-test" 0 0 1,2,3,4
    ;;
model)
    for mode in smoke target replay; do
        bash "$F/tools/offline.sh" "$mode" > "$O/logs/rtl-confirmed-$mode.log" 2>&1
        printf 'RTL_CONFIRMED_%s_RC=0\n' "$mode"
    done
    python "$F/tools/compare-logits.py"
    ;;
pb-baseline)
    # pb-baseline-src is tracked HEAD plus exactly the saved pre-work two-file diff.
    # No FP32 files or tracked FP32 plumbing exist in that scratch source.
    cmake -S "$O/pb-baseline-src" -B "$O/pb-baseline-build" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON \
        -DGGML_GEMMINI=ON -DGGML_GEMMINI_DIM=32 -DGGML_GEMMINI_PB=ON \
        -DGGML_GEMMINI_PB_WEIGHT_FORMAT=Q8_0 -DGGML_GEMMINI_COMPUTE_TYPE=INT \
        -DGGML_GEMMINI_ACTIVATION_BITS=8 -DGGML_GEMMINI_WEIGHT_BITS=8 \
        -DGGML_GEMMINI_ACTIVATION_QUANT=BLOCK -DGGML_GEMMINI_BLOCK_SIZE=32 \
        -DGGML_GEMMINI_OPTION=WS -DGGML_GEMMINI_ENABLE_RMD=OFF \
        -DGGML_POTAL=OFF -DGGML_OPENMP=OFF -DGGML_GEMMINI_ENABLE_OPENMP=OFF \
        -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_CURL=OFF -DGGML_BLAS=OFF -DGGML_BACKEND_DL=OFF \
        -DGEMMINI_SW_PATH=/tmp/h3-8i-25ai/sdk/dim32
    cmake --build "$O/pb-baseline-build" -j 2 --target gemmini-pb-q8-0-block-test
    ;;
*)
    exit 2 ;;
esac
