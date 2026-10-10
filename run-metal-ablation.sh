#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(rtk proxy dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec rtk proxy /usr/bin/caffeinate -i python3 "$root/scripts/experiment/run-metal-ablation.py" "$@"
