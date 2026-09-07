#!/bin/bash
set -eu
: "${FAKE_RUN_LOG:?}"
: "${FAKE_TELEMETRY_FIXTURE:?}"
: "${GEMMINI_LOG_DIR:?}"
cycle_log=
model=
route=
previous=
for argument in "$@"; do
    printf '%s\t%s\n' "${FAKE_PHASE:-measured}" "$argument" >> "$FAKE_RUN_LOG"
    if [ "$previous" = model ]; then model=$argument; fi
    if [ "$previous" = cycle ]; then cycle_log=$argument; fi
    case $argument in
        -m) previous=model ;;
        --gemmini-cycle-log) previous=cycle ;;
        *) previous= ;;
    esac
done
printf 'workload\t%s\t%s\n' "${FAKE_PHASE:-measured}" "$model" >> "$FAKE_RUN_LOG"
case ${model##*/} in
    gpt2.Q4_0.gguf) route=q8_h1 ;;
    gpt2.Q4_HP1.gguf|gpt2.Q8_HP1.gguf|gpt2.Q16_HP1.gguf) route=q8_hp1 ;;
    gpt2.i8_tensor.gguf) route=tensor_i8 ;;
    gpt2.Q16_0.gguf) route=q8_h0 ;;
    *) exit 8 ;;
esac
if [ "${FAKE_ROUTE_MISMATCH_MODEL:-}" = "$model" ]; then route=q8_h0; fi
if [ "${FAKE_ROUTE_MALFORMED_MODEL:-}" = "$model" ]; then
    printf '{malformed json\n' > "$GEMMINI_LOG_DIR/debug-log.jsonl"
elif [ "${FAKE_ROUTE_SUPPRESS_MODEL:-}" != "$model" ]; then
    printf '%s\n' '{ "layer": "blk.0", "msg": "unrelated message" }' > "$GEMMINI_LOG_DIR/debug-log.jsonl"
    printf '{"layer":"blk.0","msg":"[matmul.route] invocation=stripe-pipeline activation_route=q8 weight_route=%s backend_route=cpu_direct"}\n' "$route" >> "$GEMMINI_LOG_DIR/debug-log.jsonl"
fi
printf 'level=1 worker=0/2 tid=0x1 cpus=0\nlevel=1 worker=1/2 tid=0x2 cpus=1\n' >&2
if [ "${FAKE_HOLD_MODEL:-}" = "$model" ] && [ "${FAKE_PHASE:-}" = measured ]; then
    printf '%s\n' "$$" > "$FAKE_LLAMA_PID_FILE"
    cat "$FAKE_BLOCK_FIFO" & child=$!
    printf '%s\n' "$child" > "$FAKE_DESCENDANT_PID_FILE"
    trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; exit 143' INT TERM
    printf 'ready\n' > "$FAKE_READY_FIFO"
    wait "$child"
fi
if [ "${FAKE_FAIL_MODEL:-}" = "$model" ] && [ "${FAKE_PHASE:-}" = measured ]; then
    exit 9
fi
if [ "${FAKE_MALFORMED_MODEL:-}" = "$model" ]; then
    printf '{bad json\n' > "$cycle_log"
else
    cp "$FAKE_TELEMETRY_FIXTURE" "$cycle_log"
fi
