#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
runner="$repo_root/scripts/experiment/run-with-output.sh"
test_root=$(mktemp -d "${TMPDIR:-/tmp}/gemmini-experiment-output.XXXXXX")
output_root="$test_root/output"
marker="$test_root/command-ran"
mock="$test_root/mock-producer.sh"
fake_date="$test_root/date"
initial_cwd=$(pwd -P)
outside_before=$(if [[ -d $repo_root/output/experiment ]]; then find "$repo_root/output/experiment" -print 2>/dev/null | sort; fi)

cleanup() {
    rm -rf "$test_root"
}
trap cleanup EXIT

fail() {
    printf 'test-experiment-output: %s\n' "$*" >&2
    exit 1
}

assert_file() {
    [[ -f $1 ]] || fail "missing file: $1"
}

assert_contains() {
    grep -F --quiet -- "$2" "$1" || fail "missing $2 in $1"
}

assert_status() {
    local expected=$1
    shift
    set +e
    "$@" >/dev/null 2>&1
    local status=$?
    set -e
    [[ $status -eq $expected ]] || fail "expected status $expected, got $status: $*"
}

assert_mode() {
    local expected=$1
    local path=$2
    local actual
    if stat -f '%Lp' "$path" >/dev/null 2>&1; then
        actual=$(stat -f '%Lp' "$path")
    else
        actual=$(stat -c '%a' "$path")
    fi
    [[ $actual == "$expected" ]] || fail "expected mode $expected, got $actual: $path"
}

printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'printf ran > "$MARKER"' \
    'printf "stdout-stream\\n"' \
    'printf "stderr-stream\\n" >&2' \
    "printf '%s\\n' '{\"route\":\"log\"}' > \"\$GEMMINI_LOG_DIR/producer-log.jsonl\"" \
    "printf '%s\\n' '{\"route\":\"cycle\"}' > \"\$GGML_GEMMINI_CYCLE_DETAIL_LOG\"" \
    '[[ $(pwd -P) == "$REPO_ROOT" ]]' \
    '[[ $OUTPUT_ROOT == "$EXPECTED_OUTPUT_ROOT" ]]' \
    '[[ $EXPERIMENT_DIR == "$OUTPUT_ROOT"/experiment/* ]]' \
    '[[ $OUTPUT_DIR == "$EXPERIMENT_DIR" ]]' \
    '[[ $LOG_DIR == "$EXPERIMENT_DIR/raw" ]]' \
    'exit "${MOCK_STATUS:-0}"' > "$mock"
chmod +x "$mock"
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'if [[ $2 == +%Y%m%d-%H%M%S ]]; then printf "20260811-010203\\n"; else printf "2026-08-11T01:02:03Z\\n"; fi' > "$fake_date"
chmod +x "$fake_date"

mkdir -p "$output_root"
OUTPUT_ROOT="$output_root" MARKER="$marker" EXPECTED_OUTPUT_ROOT="$output_root" REPO_ROOT="$repo_root" \
    "$runner" success -- "$mock" >"$test_root/success-launch.out" 2>"$test_root/success-launch.err" || fail 'success command failed'

success_runs=("$output_root"/experiment/success-*)
[[ ${#success_runs[@]} -eq 1 && -d ${success_runs[0]} ]] || fail 'expected exactly one success run'
success_run=${success_runs[0]}
assert_file "$success_run/raw/stdout.txt"
assert_file "$success_run/raw/stderr.txt"
assert_file "$success_run/raw/producer-log.jsonl"
assert_file "$success_run/raw/exsia-cycle-detail.jsonl"
assert_file "$success_run/manifest.txt"
assert_mode 700 "$success_run"
assert_mode 700 "$success_run/raw"
assert_mode 600 "$success_run/manifest.txt"
assert_mode 600 "$success_run/raw/stdout.txt"
assert_mode 600 "$success_run/raw/stderr.txt"
assert_mode 600 "$success_run/raw/producer-log.jsonl"
assert_mode 600 "$success_run/raw/exsia-cycle-detail.jsonl"
assert_contains "$success_run/raw/stdout.txt" 'stdout-stream'
assert_contains "$success_run/raw/stderr.txt" 'stderr-stream'
assert_contains "$success_run/raw/producer-log.jsonl" '"route":"log"'
assert_contains "$success_run/raw/exsia-cycle-detail.jsonl" '"route":"cycle"'
assert_contains "$success_run/manifest.txt" 'name=success'
assert_contains "$success_run/manifest.txt" 'start_utc='
assert_contains "$success_run/manifest.txt" 'end_utc='
assert_contains "$success_run/manifest.txt" "initial_cwd=$initial_cwd"
assert_contains "$success_run/manifest.txt" "repo_root=$repo_root"
assert_contains "$success_run/manifest.txt" "command=$mock"
assert_contains "$success_run/manifest.txt" "output_root=$output_root"
assert_contains "$success_run/manifest.txt" "experiment_dir=$success_run"
assert_contains "$success_run/manifest.txt" "gemmini_log_dir=$success_run/raw"
assert_contains "$success_run/manifest.txt" "ggml_gemmini_cycle_detail_log=$success_run/raw/exsia-cycle-detail.jsonl"
assert_contains "$success_run/manifest.txt" 'git_sha='
assert_contains "$success_run/manifest.txt" 'git_branch='
assert_contains "$success_run/manifest.txt" 'git_dirty='
assert_contains "$success_run/manifest.txt" 'exit_status=0'
assert_contains "$success_run/manifest.txt" "output_dir=$success_run"
assert_file "$marker"

rm -f "$marker"
set +e
OUTPUT_ROOT="$output_root" MARKER="$marker" EXPECTED_OUTPUT_ROOT="$output_root" REPO_ROOT="$repo_root" MOCK_STATUS=7 \
    "$runner" failure -- "$mock" >"$test_root/failure-launch.out" 2>"$test_root/failure-launch.err"
failure_status=$?
set -e
[[ $failure_status -eq 7 ]] || fail "expected failure status 7, got $failure_status"
failure_runs=("$output_root"/experiment/failure-*)
[[ ${#failure_runs[@]} -eq 1 && -d ${failure_runs[0]} ]] || fail 'expected exactly one failure run'
failure_run=${failure_runs[0]}
assert_file "$failure_run/raw/stdout.txt"
assert_file "$failure_run/raw/stderr.txt"
assert_file "$failure_run/raw/producer-log.jsonl"
assert_file "$failure_run/raw/exsia-cycle-detail.jsonl"
assert_contains "$failure_run/manifest.txt" 'exit_status=7'
assert_file "$marker"

collision_run="$output_root/experiment/collision-20260811-010203"
mkdir "$collision_run"
printf 'keep\n' > "$collision_run/sentinel.txt"
rm -f "$marker"
assert_status 1 env PATH="$test_root:$PATH" OUTPUT_ROOT="$output_root" MARKER="$marker" "$runner" collision -- "$mock"
assert_contains "$collision_run/sentinel.txt" 'keep'
[[ ! -e $marker ]] || fail 'collision ran command'

before_runs=$(find "$output_root/experiment" -mindepth 1 -maxdepth 1 -type d | sort)
for name in '' .hidden ../x /absolute 'with space' 'semi;colon'; do
    rm -f "$marker"
    assert_status 2 env OUTPUT_ROOT="$output_root" MARKER="$marker" "$runner" "$name" -- "$mock"
    [[ ! -e $marker ]] || fail "unsafe name ran command: $name"
done
assert_status 2 env OUTPUT_ROOT="$output_root" MARKER="$marker" "$runner" valid --
assert_status 2 env OUTPUT_ROOT="$output_root" MARKER="$marker" "$runner" valid
after_runs=$(find "$output_root/experiment" -mindepth 1 -maxdepth 1 -type d | sort)
[[ $before_runs == "$after_runs" ]] || fail 'invalid input created a run directory'

outside_after=$(if [[ -d $repo_root/output/experiment ]]; then find "$repo_root/output/experiment" -print 2>/dev/null | sort; fi)
[[ $outside_before == "$outside_after" ]] || fail 'wrapper wrote outside test-owned OUTPUT_ROOT'
actual_raw=$(find "$success_run/raw" -maxdepth 1 -type f -exec basename {} \; | sort)
expected_raw=$(printf '%s\n' exsia-cycle-detail.jsonl producer-log.jsonl stderr.txt stdout.txt)
[[ $actual_raw == "$expected_raw" ]] || fail "unexpected copied artifact: $actual_raw"

printf 'test-experiment-output: PASS\n'
