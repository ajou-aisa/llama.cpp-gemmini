#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
result=${1:-$repo/output/experiment/compaction-abc-20261004}
for bits in 4 8; do
    build="$repo/.omo/compaction-abc/capture-build"
    suffix=
    if [[ $bits == 8 ]]; then
        build="$repo/.omo/compaction-abc/capture-build-a8"
        suffix=-a8
    fi
    for dim in 16 64; do
        for model in gpt2 llama; do
            rtk proxy env ABC_CPU_ONLY=1 "$build/bin/compaction-abc" "$result/$model$suffix" \
                "$result/$model-a$bits-d$dim.csv" "$dim" 7 \
                2> "$result/$model-a$bits-d$dim-replay.log"
        done
    done
done
rtk proxy uv run "$repo/scripts/eval/compaction_abc/label_phases.py" "$result"
