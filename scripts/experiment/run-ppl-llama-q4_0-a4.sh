#!/usr/bin/env bash
exec rtk proxy bash "$(rtk proxy dirname -- "${BASH_SOURCE[0]}")/run-metal-quality-ppl.sh" --case llama-rtnwa4 "$@"
