#!/usr/bin/env bash
set -euo pipefail

binary=$1
output=$(mktemp -d "$2/optrace-transaction.XXXXXX")
for pair in frontend-full:full full:adapter-full compact:compact; do
    control=${pair%:*}
    failure=${pair#*:}
    "$binary" "$control" "$output/$control.jsonl"
    cutoff=$(LC_ALL=C awk '
        /"kind":"parent_end"/ { print bytes; found = 1; exit }
        { bytes += length($0) + 1 }
        END { if (!found) exit 1 }
    ' "$output/$control.jsonl")
    "$binary" "fail-$failure" "$output/fail-$failure.jsonl" "$cutoff"
    head -c "$cutoff" "$output/$control.jsonl" |
        cmp - <(head -c "$cutoff" "$output/fail-$failure.jsonl")
    if grep -q '"kind":"run_end".*"status":"success"' "$output/fail-$failure.jsonl"; then
        exit 1
    fi
done
printf 'OPTRACE_TRANSACTION_PASS evidence=%s\n' "$output"
