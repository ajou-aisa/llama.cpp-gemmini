#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
result=${1:-$repo/output/experiment/compaction-abc-20261004}
runner="$repo/.omo/compaction-abc/capture-build/bin/compaction-cached"
for bits in 4 8; do
    suffix=
    [[ $bits == 4 ]] || suffix=-a8
    for model in gpt2 llama; do
        rtk proxy "$runner" "$result/$model$suffix" "$result/$model-a$bits-cached.csv" 7 \
            2> "$result/$model-a$bits-cached.log"
    done
done
rtk proxy uv run "$repo/scripts/eval/compaction_abc/label_phases.py" "$result"
