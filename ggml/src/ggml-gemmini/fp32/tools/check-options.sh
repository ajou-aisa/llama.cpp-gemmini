#!/usr/bin/env bash
source /home/alveo/miniforge3/etc/profile.d/conda.sh
cd /home/alveo/firesim/target-design/chipyard
source env.sh
conda activate /home/alveo/firesim/.conda-env
set -euo pipefail
L=/home/alveo/firesim/target-design/chipyard/generators/potal/software/workspace/llama.cpp-potal
O=/media/alveo/LaCie/fp32-5c-d8/sw
mkdir -p "$O/options"
base=(-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
      -DLLAMA_CURL=OFF -DGGML_OPENMP=OFF -DGGML_GEMMINI_ENABLE_OPENMP=OFF)
for dim in 16 32 64; do
    cmake -S "$L" -B "$O/options/off-$dim" "${base[@]}" \
        -DGGML_GEMMINI=OFF -DGGML_GEMMINI_FP32=OFF -DGGML_GEMMINI_DIM="$dim" \
        > "$O/options/off-$dim.log" 2>&1
    printf 'OPTION off_dim_%s rc=0\n' "$dim"
done
for conflict in GGML_GEMMINI_PB GGML_POTAL GGML_POTAL_W4A4 GGML_POTAL_ATTENTION \
        GGML_GEMMINI_ENABLE_A4NKS_ATTENTION GGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION; do
    rc=0
    cmake -S "$L" -B "$O/options/$conflict" "${base[@]}" \
        -DGGML_GEMMINI=ON -DGGML_GEMMINI_FP32=ON -DGGML_GEMMINI_DIM=8 \
        "-D$conflict=ON" > "$O/options/$conflict.log" 2>&1 || rc=$?
    [ "$rc" -ne 0 ]
    printf 'OPTION conflict_%s rc=%s expected_nonzero=1\n' "$conflict" "$rc"
done
rc=0
cmake -S "$L" -B "$O/options/off-8" "${base[@]}" \
    -DGGML_GEMMINI=OFF -DGGML_GEMMINI_FP32=OFF -DGGML_GEMMINI_DIM=8 \
    > "$O/options/off-8.log" 2>&1 || rc=$?
[ "$rc" -ne 0 ]
printf 'OPTION off_dim_8 rc=%s expected_nonzero=1\n' "$rc"
printf 'FP32_OPTION_GATES PASS cases=10\n'
