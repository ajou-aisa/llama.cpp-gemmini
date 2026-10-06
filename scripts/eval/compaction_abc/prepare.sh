#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
work=${1:-$repo/.omo/compaction-abc}
im2p=${IM2P_CYCLE_ROOT:-$repo/../worktrees/im2p-gemmini-relative}
source="$work/source"
if [[ ! -d $source ]]; then
    rtk proxy mkdir -p "$source"
    rtk proxy rsync -a --exclude='.git' --exclude='.omo' --exclude='.venv' \
        --exclude='.serena' --exclude='.codegraph' --exclude='/build/' --exclude='/build-*/' \
        --exclude='/output/' --exclude='/models/' --exclude='/log/' --exclude='/wikitext-2-raw/' \
        --exclude='/out.txt' --exclude='/err.txt' --exclude='*.log' "$repo/" "$source/"
    rtk proxy patch -d "$source" -p1 -i "$repo/scripts/eval/compaction_abc/capture.patch"
    rtk proxy cp "$repo/scripts/eval/compaction_abc/capture.hpp" \
        "$source/ggml/src/ggml-gemmini/residual/direct/ablation-capture.hpp"
fi
rtk proxy cmake -S "$im2p/sim/cycle" -B "$work/cycle-build" \
    -DCMAKE_BUILD_TYPE=Release -DIM2P_CYCLE_BUILD_TESTS=ON
rtk proxy cmake --build "$work/cycle-build" -j 4
for bits in 4 8; do
    build="$work/capture-build"
    [[ $bits != 8 ]] || build="$work/capture-build-a8"
    rtk proxy cmake -S "$source" -B "$build" \
        -DCMAKE_BUILD_TYPE=Release -DGGML_GEMMINI=ON \
        -DGEMMINI_SW_PATH="$repo/../RISC-V-DynDNN-gemmini-include" \
        -DGGML_GEMMINI_ACTIVATION_BITS="$bits" -DGGML_GEMMINI_WEIGHT_BITS="$bits" \
        -DGGML_GEMMINI_DIM=16 -DGGML_GEMMINI_OPTION=CPU \
        -DGGML_GEMMINI_DEFAULT_MATMUL_MODE=STRIPE_PIPELINE \
        -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU -DGGML_GEMMINI_ENABLE_RMD=ON \
        -DGGML_GEMMINI_EXSIA_DEFAULT_MODE=SEQUENTIAL -DGGML_GEMMINI_DEFAULT_STRIPE_ROWS=16 \
        -DGGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE=OFF \
        -DGGML_GEMMINI_ENABLE_OPENMP=OFF -DGGML_OPENMP=OFF \
        -DGGML_METAL=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF -DGGML_LLAMAFILE=OFF \
        -DLOG_DEBUG=0 -DLOG_CYCLE=0 -DCYCLE_DETAIL=0 -DLOG_DUMP=0 -DLLAMA_BUILD_TESTS=OFF \
        -DABC_SOURCE="$repo/scripts/eval/compaction_abc" \
        -DABC_CYCLE_INCLUDE="$im2p/sim/include" \
        -DABC_CYCLE_LIBRARY="$work/cycle-build/libim2p_cycle_model.a"
    rtk proxy cmake --build "$build" --target llama-cli llama-quantize compaction-abc compaction-cycle compaction-cached -j 4
done
