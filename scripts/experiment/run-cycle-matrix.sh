#!/bin/bash
set -uo pipefail
umask 077

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "$script_dir/../.." && pwd -P)
OUTPUT_ROOT=${OUTPUT_ROOT:-$repo_root/output}
MODEL_ROOT=${MODEL_ROOT:-$repo_root}
Q4_BUILD_ROOT=${Q4_BUILD_ROOT:-build-arm64-q4}
Q8_BUILD_ROOT=${Q8_BUILD_ROOT:-build-arm64-cpu}
Q16_BUILD_ROOT=${Q16_BUILD_ROOT:-build-arm64-q16}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-/usr/bin/python3}
MATRIX_TEGRastats=${MATRIX_TEGRastats:-tegrastats}
MATRIX_TIMESTAMP=${MATRIX_TIMESTAMP:-$(date -u +%Y%m%d-%H%M%SZ)}
MATRIX_WARMUPS=${MATRIX_WARMUPS:-1}
MATRIX_REPEATS=${MATRIX_REPEATS:-3}
MATRIX_OMP_THREADS=${MATRIX_OMP_THREADS:-6}
MATRIX_CPUS=${MATRIX_CPUS:-0,1,2,3,4,5}
MATRIX_OMP_PLACES=${MATRIX_OMP_PLACES:-'{0},{1},{2},{3},{4},{5}'}
MATRIX_INTERVAL_MS=${MATRIX_INTERVAL_MS:-100}
MATRIX_READY_TIMEOUT=${MATRIX_READY_TIMEOUT:-10}
MATRIX_WORKLOAD_TIMEOUT=${MATRIX_WORKLOAD_TIMEOUT:-3600}
MATRIX_LEAF_OPERATION=${MATRIX_LEAF_OPERATION:-rmd_direct_j_tile_interval}
BUILD_RECORDS=("4|$Q4_BUILD_ROOT" "8|$Q8_BUILD_ROOT" "16|$Q16_BUILD_ROOT")
CASE_RECORDS=(
    'q4-baseline|4|models/gpt2.Q4_0.gguf|baseline|q8_h1'
    'q4-hp1|4|models/gpt2.Q4_HP1.gguf|hp1|q8_hp1'
    'q8-baseline|8|models/gpt2.i8_tensor.gguf|baseline|tensor_i8'
    'q8-hp1|8|models/gpt2.Q8_HP1.gguf|hp1|q8_hp1'
    'q16-baseline|16|models/gpt2.Q16_0.gguf|baseline|q8_h0'
    'q16-hp1|16|models/gpt2.Q16_HP1.gguf|hp1|q8_hp1'
)
UTILITY_NAMES=(capture_cpu_frequency.py cycle_log_to_csv.py render_operation_cycles.py render_worker_cycles.py render_e2e_cycles.py summarize_cycle_matrix.py)

fail() { printf 'run-cycle-matrix: %s\n' "$*" >&2; return 1; }
absolute() { case $1 in /*) printf '%s\n' "$1";; *) printf '%s/%s\n' "$repo_root" "$1";; esac; }
cache_has() { grep -Eq "^$2(:[^=]+)?=$3$" "$1"; }
owned_pid=
cleanup_owned() {
    if [[ -n $owned_pid ]]; then
        kill -TERM "$owned_pid" 2>/dev/null || true
        wait "$owned_pid" 2>/dev/null || true
        owned_pid=
    fi
}
on_signal() { local number=$1; cleanup_owned; exit $((128 + number)); }
trap 'on_signal 2' INT
trap 'on_signal 15' TERM
trap cleanup_owned EXIT

preflight() {
    local name width relative variant route root binary cache utility
    [[ $MATRIX_TIMESTAMP =~ ^[0-9]{8}-[0-9]{6}Z$ ]] || { fail 'invalid MATRIX_TIMESTAMP'; return 1; }
    [[ $MATRIX_WARMUPS =~ ^[0-9]+$ && $MATRIX_REPEATS =~ ^[1-9][0-9]*$ && $MATRIX_OMP_THREADS =~ ^[1-9][0-9]*$ ]] || { fail 'warmups/repeats/threads must be non-negative integers with positive repeats/threads'; return 1; }
    [[ -x $PYTHON_EXECUTABLE ]] || { fail "Python executable is unavailable: $PYTHON_EXECUTABLE"; return 1; }
    [[ -x $MATRIX_TEGRastats ]] || { fail "tegrastats executable is unavailable: $MATRIX_TEGRastats"; return 1; }
    for utility in "${UTILITY_NAMES[@]}"; do [[ -r $repo_root/scripts/utils/$utility ]] || { fail "utility is unavailable: $utility"; return 1; }; done
    for record in "${BUILD_RECORDS[@]}"; do
        IFS='|' read -r width root <<< "$record"; root=$(absolute "$root"); binary=$root/bin/llama-cli; cache=$root/CMakeCache.txt
        [[ -x $binary && -r $cache ]] || { fail "width $width binary/cache is unavailable"; return 1; }
        cache_has "$cache" GGML_GEMMINI_ACTIVATION_BITS "$width" && cache_has "$cache" GGML_GEMMINI_WEIGHT_BITS "$width" && cache_has "$cache" LOG_DEBUG 1 && cache_has "$cache" LOG_CYCLE 1 && cache_has "$cache" CYCLE_DETAIL 1 && cache_has "$cache" GGML_CPU_CYCLE_LOG '(ON|1|TRUE)' && cache_has "$cache" GGML_OPENMP '(ON|1|TRUE)' && cache_has "$cache" GGML_GEMMINI_ENABLE_OPENMP '(ON|1|TRUE)' && cache_has "$cache" GGML_GEMMINI_OPTION CPU && cache_has "$cache" GGML_GEMMINI_EXECUTION_BACKEND HARDWARE && cache_has "$cache" GGML_GEMMINI_EXSIA_PROFILE_SCOPE STAGE || { fail "width $width cache identity/route mismatch"; return 1; }
    done
    for record in "${CASE_RECORDS[@]}"; do IFS='|' read -r name width relative variant route <<< "$record"; [[ -r $MODEL_ROOT/$relative ]] || { fail "model is unavailable: $relative"; return 1; }; done
    mkdir -p "$OUTPUT_ROOT/experiment" || return 1
    [[ -d $OUTPUT_ROOT/experiment && -w $OUTPUT_ROOT/experiment ]] || { fail 'output parent is not writable'; return 1; }
    experiment=$OUTPUT_ROOT/experiment/$MATRIX_TIMESTAMP
    mkdir "$experiment" || { fail "output directory already exists: $experiment"; return 1; }
}

build_for_width() {
    local wanted=$1 width root record
    for record in "${BUILD_RECORDS[@]}"; do IFS='|' read -r width root <<< "$record"; [[ $width = "$wanted" ]] && { printf '%s/bin/llama-cli\n' "$(absolute "$root")"; return; }; done
    return 1
}
write_manifests() {
    local record cell width relative variant route root
    mkdir "$experiment/manifest" "$experiment/runs" "$experiment/warmups" "$experiment/summary"
    printf 'cell,width,model,variant,route\n' > "$experiment/manifest/cases.csv"
    for record in "${CASE_RECORDS[@]}"; do IFS='|' read -r cell width relative variant route <<< "$record"; printf '%s,%s,%s,%s,%s\n' "$cell" "$width" "$relative" "$variant" "$route" >> "$experiment/manifest/cases.csv"; done
    printf 'width,build_root,binary_sha256,cache_sha256\n' > "$experiment/manifest/builds.csv"
    for record in "${BUILD_RECORDS[@]}"; do IFS='|' read -r width root <<< "$record"; root=$(absolute "$root"); printf '%s,%s,%s,%s\n' "$width" "$root" "$(shasum -a 256 "$root/bin/llama-cli" | awk '{print $1}')" "$(shasum -a 256 "$root/CMakeCache.txt" | awk '{print $1}')" >> "$experiment/manifest/builds.csv"; done
    printf 'timestamp=%s\nrepeats=%s\nwarmups=%s\nomp_threads=%s\nomp_places=%s\nomp_proc_bind=TRUE\ncpus=%s\nprofile_scope=STAGE\n' "$MATRIX_TIMESTAMP" "$MATRIX_REPEATS" "$MATRIX_WARMUPS" "$MATRIX_OMP_THREADS" "$MATRIX_OMP_PLACES" "$MATRIX_CPUS" > "$experiment/manifest/config.txt"
}
validate_route() {
    "$PYTHON_EXECUTABLE" - "$1" "$route" "$2" <<'PY'
import json
from pathlib import Path
import re
import sys

source, expected, target = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
observed = []
status = "missing"
try:
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                status = "malformed"
                break
            message = payload.get("msg") if isinstance(payload, dict) else None
            if not isinstance(message, str):
                status = "malformed"
                break
            if not message.startswith("[matmul.route]"):
                continue
            matches = re.findall(r"(?:^|\s)weight_route=([A-Za-z0-9_]+)(?=\s|$)", message)
            if len(matches) != 1:
                status = "malformed"
                break
            observed.append(matches[0])
except (OSError, UnicodeError):
    status = "missing"
else:
    if status != "malformed":
        status = "valid" if observed and set(observed) == {expected} else ("mismatch" if observed else "missing")
target.write_text("expected_route=%s\nobserved_routes=%s\nobservation_count=%d\nvalidation_status=%s\n" % (expected, ",".join(sorted(set(observed))), len(observed), status), encoding="utf-8")
raise SystemExit(0 if status == "valid" else 1)
PY
}
run_warmups() {
    local cell=$1 width=$2 model=$3 binary warmup directory status route_ok
    binary=$(build_for_width "$width")
    for ((warmup=1; warmup<=MATRIX_WARMUPS; warmup++)); do
        printf -v directory '%s/warmups/%s/warmup-%03d' "$experiment" "$cell" "$warmup"; mkdir -p "$directory/debug"
        OMP_NUM_THREADS=$MATRIX_OMP_THREADS OMP_PLACES=$MATRIX_OMP_PLACES OMP_PROC_BIND=TRUE OMP_DISPLAY_AFFINITY=TRUE LOG_CYCLE=1 CYCLE_DETAIL=1 GGML_CPU_CYCLE_LOG=1 GGML_GEMMINI_EXSIA_PROFILE_SCOPE=STAGE GEMMINI_LOG_DIR="$directory/debug" FAKE_PHASE=warmup GGML_GEMMINI_CYCLE_DETAIL_LOG="$directory/cycles.jsonl" "$PYTHON_EXECUTABLE" "$repo_root/scripts/utils/capture_cpu_frequency.py" --tegrastats "$MATRIX_TEGRastats" --interval-ms "$MATRIX_INTERVAL_MS" --expected-cpus "$MATRIX_CPUS" --raw-output "$directory/frequency.txt" --csv-output "$directory/frequency.csv" --affinity-output "$directory/affinity.csv" --ready-timeout "$MATRIX_READY_TIMEOUT" --workload-timeout "$MATRIX_WORKLOAD_TIMEOUT" -- "$binary" -m "$model" "${WORKLOAD_ARGS[@]}" --gemmini-cycle-log "$directory/cycles.jsonl" > "$directory/stdout.txt" 2> "$directory/stderr.txt" & owned_pid=$!
        wait "$owned_pid"; status=$?; owned_pid=
        validate_route "$directory/debug/debug-log.jsonl" "$directory/route-observations.txt"; route_ok=$?
        printf 'exit_status=%s\n' "$status" > "$directory/status.txt"; cat "$directory/route-observations.txt" >> "$directory/status.txt"
        [[ $status -eq 0 && $route_ok -eq 0 ]] || overall_status=1
    done
}
frequency_metrics() {
    awk -F, 'NR>1 && $7=="true" {n++; f+=$6; u+=$5; w+=$5*$6} END {if(n && u) printf "%.12g,%.12g",f/n,w/u; else exit 1}' "$1"
}
write_run_row() {
    local target=$1 status=$2 reason=$3 cycle=${4:-} cycle_unit=${5:-} wall=${6:-} wall_unit=${7:-} means=${8:-} cycle_source=${9:-} wall_source=${10:-} cycle_status=${11:-unknown} mean weighted
    IFS=, read -r mean weighted <<< "$means"
    printf 'cell,width,variant,repeat,required,status,failure_reason,cycle_value,cycle_unit,wall_value,wall_unit,frequency_mean_mhz,frequency_weighted_mhz,affinity_status,affinity_reason,cycle_source,wall_source,cycle_status\n' > "$target"
    printf '%s,%s,%s,%s,true,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$cell" "$width" "$variant" "$repeat" "$status" "$reason" "$cycle" "$cycle_unit" "$wall" "$wall_unit" "$mean" "$weighted" "$([[ $status = success ]] && printf conflict_free || printf unavailable)" "$([[ $status = success ]] && printf '' || printf '%s' "$reason")" "$cycle_source" "$wall_source" "$cycle_status" >> "$target"
}
run_repeat() {
    local directory=$1 binary=$2 model=$3 raw csv tables capture_status=0 route_ok=0 reason= cycle= cycle_unit= wall= wall_unit= means= status=failure cycle_source= wall_source= cycle_status=unknown cycle_reason= trusted_cycle=
    raw=$directory/raw; csv=$directory/csv; tables=$directory/tables; mkdir -p "$raw/debug" "$csv" "$tables"
    printf -v command_text '%q ' "$binary" -m "$model" "${WORKLOAD_ARGS[@]}" --gemmini-cycle-log "$raw/cycles.jsonl"; printf '%s\n' "${command_text% }" > "$directory/command.txt"
    OMP_NUM_THREADS=$MATRIX_OMP_THREADS OMP_PLACES=$MATRIX_OMP_PLACES OMP_PROC_BIND=TRUE OMP_DISPLAY_AFFINITY=TRUE LOG_CYCLE=1 CYCLE_DETAIL=1 GGML_CPU_CYCLE_LOG=1 GGML_GEMMINI_EXSIA_PROFILE_SCOPE=STAGE GEMMINI_LOG_DIR="$raw/debug" GGML_GEMMINI_CYCLE_DETAIL_LOG="$raw/cycles.jsonl" FAKE_PHASE=measured "$PYTHON_EXECUTABLE" "$repo_root/scripts/utils/capture_cpu_frequency.py" --tegrastats "$MATRIX_TEGRastats" --interval-ms "$MATRIX_INTERVAL_MS" --expected-cpus "$MATRIX_CPUS" --raw-output "$raw/frequency.txt" --csv-output "$csv/frequency.csv" --affinity-output "$csv/affinity.csv" --ready-timeout "$MATRIX_READY_TIMEOUT" --workload-timeout "$MATRIX_WORKLOAD_TIMEOUT" -- "$binary" -m "$model" "${WORKLOAD_ARGS[@]}" --gemmini-cycle-log "$raw/cycles.jsonl" > "$raw/stdout.txt" 2> "$raw/stderr.txt" & owned_pid=$!
    wait "$owned_pid"; capture_status=$?; owned_pid=
    validate_route "$raw/debug/debug-log.jsonl" "$directory/route-observations.txt"; route_ok=$?
    if [[ $capture_status -ne 0 ]]; then reason='workload failed';
    elif [[ $route_ok -ne 0 ]]; then reason='route validation failed';
    elif ! "$PYTHON_EXECUTABLE" "$repo_root/scripts/utils/cycle_log_to_csv.py" "$raw/cycles.jsonl" "$csv" >> "$raw/stdout.txt" 2>> "$raw/stderr.txt"; then reason='CSV conversion failed';
    elif ! grep -Eq '"backend"[[:space:]]*:[[:space:]]*"cpu_direct"' "$raw/cycles.jsonl"; then reason='runtime route mismatch';
    elif ! "$PYTHON_EXECUTABLE" "$repo_root/scripts/utils/render_operation_cycles.py" "$raw/cycles.jsonl" "$tables" >> "$raw/stdout.txt" 2>> "$raw/stderr.txt" || ! "$PYTHON_EXECUTABLE" "$repo_root/scripts/utils/render_worker_cycles.py" "$raw/cycles.jsonl" "$tables" --op "$MATRIX_LEAF_OPERATION" >> "$raw/stdout.txt" 2>> "$raw/stderr.txt" || ! "$PYTHON_EXECUTABLE" "$repo_root/scripts/utils/render_e2e_cycles.py" "$raw/cycles.jsonl" "$tables" >> "$raw/stdout.txt" 2>> "$raw/stderr.txt"; then reason='table rendering failed';
    else
        IFS=, read -r cycle_source cycle_unit _ cycle _ _ cycle_status cycle_reason trusted_cycle < <(tail -n 1 "$tables/caller-envelope-cycles.csv")
        IFS=, read -r wall_source wall_unit _ wall _ _ < <(tail -n 1 "$tables/pipeline-envelope-ns.csv")
        if [[ $cycle_status != complete || -z $trusted_cycle ]]; then
            reason='CPU cycle validity is not complete'
            cycle=
        else
            cycle=$trusted_cycle
            means=$(frequency_metrics "$csv/frequency.csv") && status=success || reason='frequency metrics unavailable'
        fi
    fi
    [[ $status = success ]] || overall_status=1
    write_run_row "$directory/summary-input.csv" "$status" "$reason" "$cycle" "$cycle_unit" "$wall" "$wall_unit" "$means" "$cycle_source" "$wall_source" "$cycle_status"
    printf 'status=%s\nexit_status=%s\nfailure_reason=%s\n' "$status" "$capture_status" "$reason" > "$directory/status.txt"; cat "$directory/route-observations.txt" >> "$directory/status.txt"
    printf 'cell=%s\nwidth=%s\nvariant=%s\nroute=%s\nmodel=%s\nmodel_sha256=%s\nbinary=%s\nbinary_sha256=%s\ncache_sha256=%s\n' "$cell" "$width" "$variant" "$route" "$model" "$(shasum -a 256 "$model" | awk '{print $1}')" "$binary" "$(shasum -a 256 "$binary" | awk '{print $1}')" "$(shasum -a 256 "$(dirname "$(dirname "$binary")")/CMakeCache.txt" | awk '{print $1}')" > "$directory/provenance.txt"; cat "$directory/route-observations.txt" >> "$directory/provenance.txt"
    tail -n 1 "$directory/summary-input.csv" >> "$experiment/manifest/repeats.csv"
}

[[ $# -ge 1 && $1 = -- ]] || { printf 'usage: %s -- [llama-cli arguments...]\n' "${0##*/}" >&2; exit 2; }
shift; WORKLOAD_ARGS=("$@")
preflight || exit 1
write_manifests
printf 'cell,width,variant,repeat,required,status,failure_reason,cycle_value,cycle_unit,wall_value,wall_unit,frequency_mean_mhz,frequency_weighted_mhz,affinity_status,affinity_reason,cycle_source,wall_source,cycle_status\n' > "$experiment/manifest/repeats.csv"
overall_status=0
for record in "${CASE_RECORDS[@]}"; do
    IFS='|' read -r cell width relative variant route <<< "$record"; model=$MODEL_ROOT/$relative; binary=$(build_for_width "$width"); mkdir "$experiment/runs/$cell" "$experiment/warmups/$cell"
    run_warmups "$cell" "$width" "$model"
    for ((repeat=1; repeat<=MATRIX_REPEATS; repeat++)); do printf -v run_dir '%s/runs/%s/repeat-%03d' "$experiment" "$cell" "$repeat"; run_repeat "$run_dir" "$binary" "$model"; done
done
"$PYTHON_EXECUTABLE" "$repo_root/scripts/utils/summarize_cycle_matrix.py" --input "$experiment/manifest/repeats.csv" --output "$experiment/summary" --expected-repeats "$MATRIX_REPEATS" || overall_status=1
printf 'exit_status=%s\n' "$overall_status" > "$experiment/status.txt"
exit "$overall_status"
