#!/usr/bin/env bash
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
conda activate /home/alveo/firesim/.conda-env
set -euo pipefail
L=/home/alveo/firesim/target-design/chipyard/generators/potal/software/workspace/llama.cpp-potal
O=/media/alveo/LaCie/fp32-5c-d8/sw
SDK=/tmp/fp32-5c-d8/sdk/fp32d8
mkdir -p "$O/logs"
export TMPDIR="$O/tmp"
mkdir -p "$TMPDIR"
cd "$L"
case "$1" in
native|riscv)
    kind=$1
    extra=()
    if [ "$kind" = riscv ]; then
        extra+=(-DCMAKE_TOOLCHAIN_FILE=/tmp/h3-8i-25p/build/riscv-static-release.cmake -DGGML_RVV=OFF)
    fi
    cmake -S "$L" -B "$O/$kind" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NATIVE=ON -DGGML_GEMMINI=ON \
        -DGGML_GEMMINI_FP32=ON -DGGML_GEMMINI_DIM=8 -DGGML_GEMMINI_PB=OFF \
        -DGGML_GEMMINI_COMPUTE_TYPE=FLOAT -DGGML_GEMMINI_OPTION=WS -DGGML_GEMMINI_ENABLE_RMD=OFF \
        -DGGML_GEMMINI_ENABLE_A4NKS_ATTENTION=OFF \
        -DGGML_POTAL=OFF -DGGML_POTAL_W4A4=OFF -DGGML_OPENMP=OFF -DGGML_GEMMINI_ENABLE_OPENMP=OFF \
        -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
        -DGGML_BLAS=OFF -DGGML_BACKEND_DL=OFF -DGEMMINI_SW_PATH="$SDK" "${extra[@]}"
    cmake --build "$O/$kind" -j 4 --target gemmini-fp32-e2e-runner gemmini-fp32-block-test gemmini-fp32-attention-test golden_replay_fp32
    ;;
cpu)
    cmake -S "$L/ggml/src/ggml-gemmini/fp32/cpu" -B "$O/cpu" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build "$O/cpu" -j 2 --target fp32-cpu-runner
    ;;
pb-regression)
    cmake -S "$L" -B "$O/pb-off" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NATIVE=ON -DGGML_GEMMINI=ON \
        -DGGML_GEMMINI_FP32=OFF -DGGML_GEMMINI_DIM=32 -DGGML_GEMMINI_PB=ON \
        -DGGML_GEMMINI_PB_WEIGHT_FORMAT=Q8_0 -DGGML_GEMMINI_COMPUTE_TYPE=INT \
        -DGGML_GEMMINI_ACTIVATION_BITS=8 -DGGML_GEMMINI_WEIGHT_BITS=8 \
        -DGGML_GEMMINI_ACTIVATION_QUANT=BLOCK -DGGML_GEMMINI_BLOCK_SIZE=32 \
        -DGGML_GEMMINI_OPTION=WS -DGGML_GEMMINI_ENABLE_RMD=OFF \
        -DGGML_POTAL=OFF -DGGML_OPENMP=OFF -DGGML_GEMMINI_ENABLE_OPENMP=OFF \
        -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
        -DGGML_BLAS=OFF -DGGML_BACKEND_DL=OFF -DGEMMINI_SW_PATH=/tmp/h3-8i-25ai/sdk/dim32
    cmake --build "$O/pb-off" -j 4 --target gemmini-pb-q8-0-block-test gemmini-pb-q8-0-attention-test
    "$O/pb-off/bin/gemmini-pb-q8-0-block-test"
    "$O/pb-off/bin/gemmini-pb-q8-0-attention-test"
    ;;
smoke|target|cpu-target)
    mode=$1
    prompt=512; gen=9; exe="$O/native/bin/gemmini-fp32-e2e-runner"
    if [ "$mode" = smoke ]; then prompt=64; gen=8; fi
    if [ "$mode" = cpu-target ]; then exe="$O/cpu/fp32-cpu-runner"; fi
    "$exe" --model "$L/models/qwen2-0.5B/qwen2-0.5B.fp16.gguf" \
        --prompt-file /tmp/fp32-5c-d8/kit/guest/payload/prompt-512.ids \
        --prompt-tokens "$prompt" --gen-tokens "$gen" --ubatch "$prompt" --ctx 2048 \
        --issuer-cpu 0 --lane-cpus 1,2,3,4 --out "$O/$mode" --order "${2:-0}"
    ;;
replay)
    "$O/native/bin/golden_replay_fp32" "$L/models/qwen2-0.5B/qwen2-0.5B.fp16.gguf" "$O/target/dumps"
    ;;
*)
    printf 'unknown mode: %s\n' "$1" >&2; exit 2 ;;
esac
