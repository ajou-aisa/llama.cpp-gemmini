#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
runner="$repo_root/scripts/experiment/run-gemmini-rmd-on-off.sh"
test_root=$(mktemp -d "${TMPDIR:-/tmp}/gemmini-rmd-on-off.XXXXXX")
test_root=$(cd "$test_root" && pwd -P)
workspace="$test_root/workspace"
output_root="$test_root/root/output"
model="$test_root/model.gguf"
on_build="$test_root/build-riscv-rmd-on"
off_build="$test_root/build-riscv-rmd-off"
on_binary="$on_build/bin/llama-cli"
off_binary="$off_build/bin/llama-cli"
marker_root="$test_root/markers"

cleanup() {
    rm -rf "$test_root"
}
trap cleanup EXIT

fail() {
    printf 'test-gemmini-rmd-on-off-run: %s\n' "$*" >&2
    exit 1
}

assert_file() {
    [[ -f $1 ]] || fail "missing file: $1"
}

assert_contains() {
    grep -F --quiet -- "$2" "$1" || fail "missing '$2' in $1"
}

assert_status() {
    local expected=$1
    shift
    set +e
    "$@" >"$test_root/status.stdout" 2>"$test_root/status.stderr"
    local actual=$?
    set -e
    [[ $actual -eq $expected ]] || fail "expected status $expected, got $actual: $*"
}

mkdir -p "$workspace/output" "$output_root" "$marker_root" "$on_build/bin" "$off_build/bin"
printf 'workspace output must survive\n' >"$workspace/output/sentinel.txt"
printf 'model\n' >"$model"
printf 'GGML_GEMMINI_ENABLE_RMD:BOOL=ON\n' >"$on_build/CMakeCache.txt"
printf 'GGML_GEMMINI_ENABLE_RMD:BOOL=OFF\n' >"$off_build/CMakeCache.txt"

cat >"$test_root/mock-llama-cli" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

[[ ${RMD_VARIANT:-} == on || ${RMD_VARIANT:-} == off ]]
[[ ${GEMMINI_MATMUL_MODE:-} == STRIPE_PIPELINE ]]
if [[ $RMD_VARIANT == on ]]; then
    [[ ${GEMMINI_RMD_BACKEND:-} == WS ]]
else
    [[ -z ${GEMMINI_RMD_BACKEND+x} ]]
fi

mkdir -p output
printf '%s\n' "$RMD_VARIANT" >"output/variant.txt"
printf '%s\n' "$*" >"output/args.txt"
printf '%s\n' "$RMD_VARIANT" >"$MARKER_ROOT/$RMD_VARIANT"
printf 'stdout-%s\n' "$RMD_VARIANT"
printf 'stderr-%s\n' "$RMD_VARIANT" >&2

if [[ $RMD_VARIANT == off && ${FAIL_OFF:-0} == 1 ]]; then
    exit 7
fi
EOF
chmod +x "$test_root/mock-llama-cli"
cp "$test_root/mock-llama-cli" "$on_binary"
cp "$test_root/mock-llama-cli" "$off_binary"

assert_status 0 "$runner" --help
assert_contains "$test_root/status.stdout" 'usage:'
assert_status 0 "$runner" --print-build-commands
assert_contains "$test_root/status.stdout" 'BUILD_DIR=build-riscv-rmd-on'
assert_contains "$test_root/status.stdout" 'GGML_GEMMINI_ENABLE_RMD=ON'
assert_contains "$test_root/status.stdout" 'BUILD_DIR=build-riscv-rmd-off'
assert_contains "$test_root/status.stdout" 'GGML_GEMMINI_ENABLE_RMD=OFF'
assert_status 2 "$runner"
assert_status 2 "$runner" \
    --run-id ../unsafe \
    --output-root "$output_root" \
    --workspace "$workspace" \
    --model "$model" \
    --on-binary "$on_binary" \
    --off-binary "$off_binary"
[[ ! -e $test_root/unsafe ]] || fail 'unsafe run id escaped output root'
assert_status 2 "$runner" \
    --run-id mislabeled \
    --output-root "$output_root" \
    --workspace "$workspace" \
    --model "$model" \
    --on-binary "$on_binary" \
    --off-binary "$on_binary"

MARKER_ROOT="$marker_root" "$runner" \
    --run-id happy \
    --output-root "$output_root" \
    --workspace "$workspace" \
    --model "$model" \
    --on-binary "$on_binary" \
    --off-binary "$off_binary"

run_dir="$output_root/happy"
archive="$output_root/happy.tar.gz"
assert_file "$run_dir/manifest.txt"
assert_file "$archive"
assert_contains "$run_dir/manifest.txt" 'on_exit_status=0'
assert_contains "$run_dir/manifest.txt" 'off_exit_status=0'
assert_contains "$workspace/output/sentinel.txt" 'must survive'

for variant in on off; do
    variant_dir="$run_dir/$variant"
    assert_file "$variant_dir/stdout.txt"
    assert_file "$variant_dir/stderr.txt"
    assert_file "$variant_dir/exit-status.txt"
    assert_file "$variant_dir/command.txt"
    assert_file "$variant_dir/environment.txt"
    assert_file "$variant_dir/binary.sha256"
    assert_file "$variant_dir/output/variant.txt"
    assert_file "$variant_dir/output/args.txt"
    assert_contains "$variant_dir/stdout.txt" "stdout-$variant"
    assert_contains "$variant_dir/stderr.txt" "stderr-$variant"
    assert_contains "$variant_dir/output/variant.txt" "$variant"
    assert_contains "$variant_dir/output/args.txt" "$model"
done
assert_contains "$run_dir/on/environment.txt" 'GEMMINI_RMD_BACKEND=WS'
assert_contains "$run_dir/off/environment.txt" 'GEMMINI_RMD_BACKEND=<unset>'
tar -tzf "$archive" | grep -F --quiet 'happy/on/output/variant.txt' ||
    fail 'archive is missing ON output'
tar -tzf "$archive" | grep -F --quiet 'happy/off/output/variant.txt' ||
    fail 'archive is missing OFF output'

rm -f "$marker_root/on" "$marker_root/off"
set +e
FAIL_OFF=1 MARKER_ROOT="$marker_root" "$runner" \
    --run-id failure \
    --output-root "$output_root" \
    --workspace "$workspace" \
    --model "$model" \
    --on-binary "$on_binary" \
    --off-binary "$off_binary"
failure_status=$?
set -e
[[ $failure_status -eq 7 ]] || fail "expected preserved OFF status 7, got $failure_status"
assert_file "$marker_root/on"
assert_file "$marker_root/off"
assert_file "$output_root/failure/on/output/variant.txt"
assert_file "$output_root/failure/off/output/variant.txt"
assert_file "$output_root/failure.tar.gz"
assert_contains "$output_root/failure/manifest.txt" 'on_exit_status=0'
assert_contains "$output_root/failure/manifest.txt" 'off_exit_status=7'

mkdir "$output_root/collision"
assert_status 1 env MARKER_ROOT="$marker_root" "$runner" \
    --run-id collision \
    --output-root "$output_root" \
    --workspace "$workspace" \
    --model "$model" \
    --on-binary "$on_binary" \
    --off-binary "$off_binary"

printf 'test-gemmini-rmd-on-off-run: PASS\n'
