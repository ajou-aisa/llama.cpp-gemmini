#!/usr/bin/env bash
set -euo pipefail
L=$(cd "$(dirname "$0")/../../../../.." && pwd)
O=${FP32_OUTPUT_DIR:?Set FP32_OUTPUT_DIR to the evaluation output directory}
mkdir -p "$O/logs"
export TMPDIR="$O/tmp"
mkdir -p "$TMPDIR"
cd "$L"
case "$1" in
native|riscv)
    kind=$1
    : "${GEMMINI_FP32_SDK:?Set GEMMINI_FP32_SDK to the FP32 DIM8 overlay}"
    preset=gemmini-fp32-d8-reference
    if [ "$kind" = riscv ]; then preset=gemmini-fp32-d8-rocc; fi
    cmake --preset "$preset" -S "$L" -B "$O/$kind" \
        -DBUILD_SHARED_LIBS=OFF -DGGML_RVV=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DGGML_BLAS=OFF -DGGML_BACKEND_DL=OFF
    cmake --build "$O/$kind" -j 4 --target gemmini-fp32-e2e-runner gemmini-fp32-block-test gemmini-fp32-attention-test golden_replay_fp32
    ;;
cpu)
    cmake -S "$L/ggml/src/ggml-gemmini/fp32/cpu" -B "$O/cpu" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build "$O/cpu" -j 2 --target fp32-cpu-runner
    ;;
pb-regression)
    cmake --preset gemmini-int8-d32-pb-reference -S "$L" -B "$O/pb-off" \
        -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF \
        -DGGML_BLAS=OFF -DGGML_BACKEND_DL=OFF
    cmake --build "$O/pb-off" -j 4 --target gemmini-pb-q8-0-block-test gemmini-pb-q8-0-attention-test
    "$O/pb-off/bin/gemmini-pb-q8-0-block-test"
    "$O/pb-off/bin/gemmini-pb-q8-0-attention-test"
    ;;
smoke|target|cpu-target)
    mode=$1
    prompt=512; gen=9; exe="$O/native/bin/gemmini-fp32-e2e-runner"
    if [ "$mode" = smoke ]; then prompt=64; gen=8; fi
    if [ "$mode" = cpu-target ]; then exe="$O/cpu/fp32-cpu-runner"; fi
    "$exe" --model "${FP32_MODEL:?Set FP32_MODEL to the F16 GGUF model}" \
        --prompt-file "${FP32_PROMPT_IDS:?Set FP32_PROMPT_IDS to the token fixture}" \
        --prompt-tokens "$prompt" --gen-tokens "$gen" --ubatch "$prompt" --ctx 2048 \
        --issuer-cpu 0 --lane-cpus 1,2,3,4 --out "$O/$mode" --order "${2:-0}"
    ;;
replay)
    "$O/native/bin/golden_replay_fp32" "${FP32_MODEL:?Set FP32_MODEL to the F16 GGUF model}" "$O/target/dumps"
    ;;
*)
    printf 'unknown mode: %s\n' "$1" >&2; exit 2 ;;
esac
