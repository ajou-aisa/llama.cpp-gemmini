#!/bin/bash
set -eu
trap 'exit 0' INT TERM
[[ -z ${FAKE_TEGRASTATS_PID_FILE:-} ]] || printf '%s\n' "$$" > "$FAKE_TEGRASTATS_PID_FILE"
printf 'RAM 1/2MB CPU [ 10%%@729 , 20%%@918 ] GR3D_FREQ 0%%\n'
while :; do
    wait
done
