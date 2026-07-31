#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)

BIN_DIR=${BIN_DIR:-"$ROOT_DIR/build-arm64/bin"}
RUNS=${RUNS:-10}
TOKENS=${TOKENS:-15}
SEED=${SEED:-1}
PROMPT=${PROMPT:-"How's the weather today?"}
OUTPUT_DIR=${OUTPUT_DIR:-"$BIN_DIR/log/wdeq-experiment-$(date '+%Y%m%d-%H%M%S')"}

CLI="$BIN_DIR/llama-cli"
CYCLE_LOG="$BIN_DIR/log/wdeq-cycle-detail.jsonl"
RUNS_CSV="$OUTPUT_DIR/runs.csv"
SUMMARY_CSV="$OUTPUT_DIR/summary.csv"

BASELINES=(
    Q8_H1
    Q8_HP1
)

if [[ ! "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'RUNS must be a positive integer: %s\n' "$RUNS" >&2
    exit 1
fi

if [[ ! -x "$CLI" ]]; then
    printf 'llama-cli not found: %s\n' "$CLI" >&2
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    printf 'jq is required\n' >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$(dirname -- "$CYCLE_LOG")"
printf 'format,run,record_count,total_cycles\n' > "$RUNS_CSV"

for format in "${BASELINES[@]}"; do
    model="$ROOT_DIR/models/gpt2.$format.gguf"
    expected_op="weight.Dequantize $format"

    if [[ ! -f "$model" ]]; then
        printf 'model not found: %s\n' "$model" >&2
        exit 1
    fi

    for ((run = 1; run <= RUNS; ++run)); do
        printf '[%s] run %d/%d\n' "$format" "$run" "$RUNS"
        : > "$CYCLE_LOG"

        (
            cd "$BIN_DIR"
            GEMMINI_LOG_DIR="$BIN_DIR/log" "$CLI" \
                -m "$model" \
                -p "$PROMPT" \
                -n "$TOKENS" \
                --seed "$SEED"
        ) >/dev/null 2>&1

        metrics=$(jq -ers --arg expected "$expected_op" '
            if length == 0 then
                error("empty weight dequantization log")
            elif any(.[]; .op != $expected or
                           (.cycles | type) != "number" or
                           .cycles <= 0 or
                           .end < .start) then
                error("invalid weight dequantization record")
            else
                "\(length),\(map(.cycles) | add)"
            end
        ' "$CYCLE_LOG")

        IFS=, read -r record_count total_cycles <<< "$metrics"
        printf '%s,%d,%s,%s\n' \
            "$format" "$run" "$record_count" "$total_cycles" >> "$RUNS_CSV"
    done
done

printf 'format,runs,mean_total_cycles,min_total_cycles,max_total_cycles\n' > "$SUMMARY_CSV"

for format in "${BASELINES[@]}"; do
    awk -F, -v format="$format" '
        NR > 1 && $1 == format {
            value = $4 + 0
            count++
            sum += value
            if (count == 1 || value < min) min = value
            if (count == 1 || value > max) max = value
        }
        END {
            if (count == 0) exit 1
            printf "%s,%d,%.2f,%.0f,%.0f\n", format, count, sum / count, min, max
        }
    ' "$RUNS_CSV" >> "$SUMMARY_CSV"
done

printf '\nResults: %s\n' "$OUTPUT_DIR"
cat "$SUMMARY_CSV"
