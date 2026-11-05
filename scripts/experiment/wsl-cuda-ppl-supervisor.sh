#!/usr/bin/env bash
set -euo pipefail
cd -- "${POTAL_QUEUE_ROOT:?Missing queue root}"
output=${POTAL_QUEUE_OUTPUT:?Missing output}
bash scripts/experiment/start-cuda-ppl.sh --config "${POTAL_QUEUE_CONFIG:?Missing config}" --output "$output"
output=$(realpath -- "$output")
unit="potal-cuda-$(printf '%s' "$output" | sha256sum | cut -c 1-12).service"
# shortcut: Windows sleep/reboot suspends/stops WSL; invoke launcher again after reboot.
while systemctl --user is-active --quiet "$unit"; do sleep 5; done
