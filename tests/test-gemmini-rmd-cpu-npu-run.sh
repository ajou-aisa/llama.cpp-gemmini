#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
runner="$repo_root/scripts/experiment/run-gemmini-rmd-cpu-npu.sh"
test_root=$(mktemp -d "${TMPDIR:-/tmp}/gemmini-rmd-cpu-npu.XXXXXX")
workspace="$test_root/workspace"
guest_workspace="$test_root/guest/root/workspace/3rd_llama.cpp"
firesim_root="$test_root/firesim"
output_root="$test_root/output"
marker_root="$test_root/markers"
model="$test_root/model.gguf"
runtime_fixture="$test_root/mock-llama-cli"
libgomp_fixture="$test_root/libgomp.so.1"
mock_bin="$test_root/mock-bin"
real_tar=$(command -v tar)
real_cp=$(command -v cp)

cleanup() {
    rm -rf "$test_root"
}
trap cleanup EXIT

fail() {
    printf 'test-gemmini-rmd-cpu-npu-run: %s\n' "$*" >&2
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

mkdir -p "$workspace" "$(dirname "$guest_workspace")" "$firesim_root/deploy" \
    "$firesim_root/.conda-env/riscv-tools/bin" "$firesim_root/.conda-env/bin" \
    "$output_root" "$marker_root" "$mock_bin"
printf 'model\n' >"$model"
printf 'runtime\n' >"$libgomp_fixture"
mkdir -p "$workspace/build-riscv-rmd-cpu-npu"
printf 'preserve\n' >"$workspace/build-riscv-rmd-cpu-npu/sentinel.txt"

cat >"$firesim_root/sourceme-manager.sh" <<'EOF'
#!/usr/bin/env bash
[[ ${1:-} == --skip-ssh-setup ]]
export FIRESIM_SOURCED=1
EOF
chmod +x "$firesim_root/sourceme-manager.sh"

cat >"$firesim_root/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ ${1:-} == -print-file-name=libgomp.so.1 ]]
printf '%s\n' "$LIBGOMP_FIXTURE"
EOF
chmod +x "$firesim_root/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-gcc"

cat >"$firesim_root/.conda-env/bin/firesim" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ ${FIRESIM_SOURCED:-0} == 1 ]]
[[ ${1:-} == runworkload ]]
printf 'launched\n' >"$LAUNCH_MARKER"
EOF
chmod +x "$firesim_root/.conda-env/bin/firesim"

cat >"$firesim_root/deploy/update_rootfs.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ ${FIRESIM_SOURCED:-0} == 1 ]]
mkdir -p "$GUEST_WORKSPACE"
cp -R "$MANAGER_WORKSPACE/." "$GUEST_WORKSPACE/"
printf 'updated\n' >"$ROOTFS_MARKER"
EOF
chmod +x "$firesim_root/deploy/update_rootfs.sh"

cat >"$runtime_fixture" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ -z ${LD_LIBRARY_PATH+x} ]]
[[ -z ${LD_PRELOAD+x} ]]
[[ ${GEMMINI_MATMUL_MODE:-} == STRIPE_PIPELINE ]]
[[ ${GEMMINI_RMD_BACKEND:-} == CPU || ${GEMMINI_RMD_BACKEND:-} == WS ]]
[[ ${RMD_EXPERIMENT_LABEL:-} == cpu || ${RMD_EXPERIMENT_LABEL:-} == npu ]]
[[ ${RMD_EXPERIMENT_RUN:-} =~ ^[1-5]$ ]]

mkdir -p output/log
backend=cpu_direct
route=cpu
schedule=matmul-then-rmd
dispatch='"direct_events":10,"direct_calls":2,"packet_calls":0,"ws_calls":0'
packet_count=0
output_hash=3333333333333333
row_end=1
if [[ $GEMMINI_RMD_BACKEND == WS ]]; then
    backend=gemmini_ws_compact
    route=gemmini_ws
    schedule=matmul-rmd-overlap
    dispatch='"direct_events":0,"direct_calls":0,"packet_calls":2,"ws_calls":4'
    packet_count=2
fi
if [[ $RMD_EXPERIMENT_LABEL == npu && ${CONTAMINATE_NPU:-0} == 1 ]]; then
    backend=cpu_direct
fi
if [[ $RMD_EXPERIMENT_LABEL == npu && ${MISMATCH_NPU_PROOF:-0} == 1 ]]; then
    output_hash=4444444444444444
fi
if [[ $RMD_EXPERIMENT_LABEL == npu && ${MISMATCH_NPU_GEOMETRY:-0} == 1 ]]; then
    row_end=2
fi
schema='"schema":"gemmini.rmd.telemetry",'
if [[ $RMD_EXPERIMENT_LABEL == npu && ${MALFORM_NPU:-0} == 1 ]]; then
    schema=
fi
printf '{%s"version":1,"record_type":"RMD_BACKEND_TELEMETRY","runtime_bundle_id":"%s","model_id":"%s","run_id":"1","backend":"%s","source":"environment","units":"cycles","work":true,"invocation_total":100,"dispatch":{%s},"timing":{"prep":1,"backend_service":2,"merge":1,"residual_total":4,"queue":0,"dense_end":10,"residual_start":11},"geometry":{"packet_count":%s,"active_blocks":2,"compact_k_count":2,"padded_k_count":2,"physical_tile_count":1},"stripes":[{"stripe_id":0,"row_begin":0,"row_end":%s,"stages":{"dense_start":1,"dense_end":2,"residual_start":3,"backend_start":4,"backend_end":5,"merge_start":6,"merge_end":7,"residual_end":8},"input_hash":"1111111111111111","correction_hash":"2222222222222222","output_hash":"%s"}]}\n' \
    "$schema" "$GGML_GEMMINI_RUNTIME_BUNDLE_ID" "$GGML_GEMMINI_MODEL_ID" \
    "$backend" "$dispatch" "$packet_count" "$row_end" "$output_hash" \
    >output/log/debug-log.jsonl
printf '{"record_type":"PIPELINE_STRIPE_SUMMARY","backend_route":"%s","schedule":"%s"}\n' \
    "$route" "$schedule" >>output/log/debug-log.jsonl
printf '%s-%s\n' "$RMD_EXPERIMENT_LABEL" "$RMD_EXPERIMENT_RUN" \
    >>"$MARKER_ROOT/invocations.txt"
printf 'identical generated output\n'
printf 'stderr-%s-%s\n' "$RMD_EXPERIMENT_LABEL" "$RMD_EXPERIMENT_RUN" >&2
if [[ $RMD_EXPERIMENT_LABEL == cpu && $RMD_EXPERIMENT_RUN == 2 &&
      ${FAIL_CPU_RUN:-0} == 1 ]]; then
    exit 7
fi
EOF
chmod +x "$runtime_fixture"

cat >"$workspace/build-riscv.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ ${FIRESIM_SOURCED:-0} == 1 ]]
[[ ${LOG_DEBUG:-} == 1 ]]
[[ ${LOG_CYCLE:-} == 1 ]]
[[ ${CYCLE_DETAIL:-} == 1 ]]
[[ ${GGML_GEMMINI_OPTION:-} == WS ]]
[[ ${GGML_GEMMINI_ENABLE_RMD:-} == ON ]]
[[ ${GGML_GEMMINI_DEFAULT_RMD_BACKEND:-} == WS ]]
[[ ${GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE:-} == ON ]]
[[ ${CXXFLAGS:-} == '-DERROR_COMPENSATION=1 -Dgemmini_set_tile_ws=gemmini_set_tile' ]]
[[ ${1:-} == -DLLAMA_BUILD_TESTS=OFF ]]
mkdir -p "$BUILD_DIR/bin"
cp "$RUNTIME_FIXTURE" "$BUILD_DIR/bin/llama-cli"
printf 'project-library\n' >"$BUILD_DIR/bin/libggml-gemmini.so"
cat >"$BUILD_DIR/CMakeCache.txt" <<'CACHE'
GGML_GEMMINI_ENABLE_RMD:BOOL=ON
GGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE:BOOL=ON
GGML_GEMMINI_OPTION:STRING=WS
LOG_CYCLE:BOOL=ON
CYCLE_DETAIL:BOOL=ON
CACHE
EOF
chmod +x "$workspace/build-riscv.sh"

cat >"$mock_bin/tar" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
for argument in "$@"; do
    [[ $argument != -*z* ]] || exit 64
done
if [[ ${FAIL_TAR:-0} == 1 ]]; then
    previous=
    for argument in "$@"; do
        if [[ $previous == -f ]]; then
            printf 'partial\n' >"$argument"
            break
        fi
        previous=$argument
    done
    exit 9
fi
exec "$REAL_TAR" "$@"
EOF
chmod +x "$mock_bin/tar"

cat >"$mock_bin/cp" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${FAIL_COPY:-0} == 1 && " $* " == *'/output/. '* ]]; then
    exit 8
fi
exec "$REAL_CP" "$@"
EOF
chmod +x "$mock_bin/cp"

assert_status 0 "$runner" --help
assert_contains "$test_root/status.stdout" 'manager'
assert_contains "$test_root/status.stdout" 'prepare'
assert_contains "$test_root/status.stdout" 'run'
assert_status 2 "$runner" unknown

prepare_output="$test_root/prepare.stdout"
ROOTFS_MARKER="$test_root/rootfs-updated" RUNTIME_FIXTURE="$runtime_fixture" \
LIBGOMP_FIXTURE="$libgomp_fixture" MANAGER_WORKSPACE="$workspace" \
GUEST_WORKSPACE="$guest_workspace" LAUNCH_MARKER="$test_root/launched" \
"$runner" manager \
    --workspace "$workspace" \
    --firesim-root "$firesim_root" >"$prepare_output"

assert_file "$test_root/rootfs-updated"
assert_file "$workspace/build-riscv-rmd-cpu-npu/bin/llama-cli"
assert_file "$workspace/build-riscv-rmd-cpu-npu/bin/libggml-gemmini.so"
assert_file "$workspace/build-riscv-rmd-cpu-npu/bin/libgomp.so.1"
assert_file "$workspace/build-riscv-rmd-cpu-npu/experiment-build-manifest.txt"
assert_file "$guest_workspace/build-riscv-rmd-cpu-npu/experiment-build-manifest.txt"
assert_contains "$workspace/build-riscv-rmd-cpu-npu/CMakeCache.txt" \
    'GGML_GEMMINI_ENABLE_RMD:BOOL=ON'
old_builds=("$workspace"/build-riscv-rmd-cpu-npu.previous-*)
[[ ${#old_builds[@]} -eq 1 ]] || fail 'existing build was not preserved exactly once'
assert_contains "${old_builds[0]}/sentinel.txt" 'preserve'
bundle_id=$(awk -F= '$1 == "bundle_id" { print $2 }' "$prepare_output")
[[ $bundle_id =~ ^[0-9a-f]{64}$ ]] || fail 'prepare did not print a bundle ID'
assert_contains "$prepare_output" \
    "guest_command=/root/workspace/3rd_llama.cpp/scripts/experiment/run-gemmini-rmd-cpu-npu.sh run --expected-bundle-id $bundle_id"
assert_file "$test_root/launched"

assert_status 1 "$runner" run \
    --run-id wrong-bundle \
    --workspace "$guest_workspace" \
    --model "$model" \
    --output-root "$output_root" \
    --expected-bundle-id 0000000000000000000000000000000000000000000000000000000000000000
[[ ! -e $output_root/wrong-bundle ]] ||
    fail 'wrong bundle ID validation created a result directory'

cp "$guest_workspace/build-riscv-rmd-cpu-npu/experiment-build-manifest.txt" \
    "$test_root/build-manifest.saved"
printf 'tampered=1\n' \
    >>"$guest_workspace/build-riscv-rmd-cpu-npu/experiment-build-manifest.txt"
assert_status 1 "$runner" run \
    --run-id tampered-manifest \
    --workspace "$guest_workspace" \
    --model "$model" \
    --output-root "$output_root" \
    --expected-bundle-id "$bundle_id"
[[ ! -e $output_root/tampered-manifest ]] ||
    fail 'tampered manifest validation created a result directory'
mv "$test_root/build-manifest.saved" \
    "$guest_workspace/build-riscv-rmd-cpu-npu/experiment-build-manifest.txt"

cp "$guest_workspace/build-riscv-rmd-cpu-npu/bin/libggml-gemmini.so" \
    "$test_root/libggml-gemmini.so.saved"
printf 'mutation\n' >>"$guest_workspace/build-riscv-rmd-cpu-npu/bin/libggml-gemmini.so"
assert_status 1 "$runner" run \
    --run-id stale-bundle \
    --workspace "$guest_workspace" \
    --model "$model" \
    --output-root "$output_root" \
    --expected-bundle-id "$bundle_id"
[[ ! -e $output_root/stale-bundle ]] ||
    fail 'stale bundle validation created a result directory'
mv "$test_root/libggml-gemmini.so.saved" \
    "$guest_workspace/build-riscv-rmd-cpu-npu/bin/libggml-gemmini.so"

LD_LIBRARY_PATH=/contaminated LD_PRELOAD=/contaminated \
REAL_TAR="$real_tar" REAL_CP="$real_cp" PATH="$mock_bin:$PATH" \
MARKER_ROOT="$marker_root" "$runner" run \
    --run-id happy \
    --workspace "$guest_workspace" \
    --model "$model" \
    --output-root "$output_root" \
    --expected-bundle-id "$bundle_id"

run_dir="$output_root/happy"
archive="$output_root/happy.tar.gz"
assert_file "$run_dir/manifest.txt"
assert_file "$run_dir/binary.sha256"
assert_file "$run_dir/libraries.sha256"
assert_file "$run_dir/run-order.txt"
assert_file "$run_dir/experiment-build-manifest.txt"
assert_file "$archive"
assert_contains "$run_dir/manifest.txt" 'comparison_status=valid'
assert_contains "$run_dir/manifest.txt" 'cpu_runs=5'
assert_contains "$run_dir/manifest.txt" 'npu_runs=5'
[[ $(wc -l <"$marker_root/invocations.txt" | tr -d ' ') == 10 ]] ||
    fail 'expected ten ABBA invocations'
assert_contains "$marker_root/invocations.txt" 'cpu-1'
assert_contains "$marker_root/invocations.txt" 'npu-5'
assert_contains "$run_dir/run-order.txt" 'position=1 backend=cpu run=1 selector=CPU'
assert_contains "$run_dir/run-order.txt" 'position=2 backend=npu run=1 selector=WS'
assert_contains "$run_dir/run-order.txt" 'position=3 backend=npu run=2 selector=WS'
assert_contains "$run_dir/run-order.txt" 'position=10 backend=npu run=5 selector=WS'

for backend in cpu npu; do
    for run in 1 2 3 4 5; do
        result="$run_dir/$backend/run-$run"
        assert_file "$result/stdout.txt"
        assert_file "$result/stderr.txt"
        assert_file "$result/environment.txt"
        assert_file "$result/exit-status.txt"
        assert_file "$result/proofs.txt"
        assert_file "$result/output/log/debug-log.jsonl"
        assert_contains "$result/stdout.txt" 'identical generated output'
        assert_contains "$result/environment.txt" 'LD_LIBRARY_PATH=<unset>'
    done
done
assert_contains "$run_dir/cpu/run-1/environment.txt" 'GEMMINI_RMD_BACKEND=CPU'
assert_contains "$run_dir/npu/run-1/environment.txt" 'GEMMINI_RMD_BACKEND=WS'
assert_contains "$run_dir/cpu/run-1/output/log/debug-log.jsonl" '"backend":"cpu_direct"'
assert_contains "$run_dir/npu/run-1/output/log/debug-log.jsonl" '"backend":"gemmini_ws_compact"'
tar -tzf "$archive" | grep -F --quiet 'happy/cpu/run-5/output/log/debug-log.jsonl' ||
    fail 'archive is missing CPU telemetry'
tar -tzf "$archive" | grep -F --quiet 'happy/npu/run-5/output/log/debug-log.jsonl' ||
    fail 'archive is missing NPU telemetry'
tar -tzf "$archive" | grep -F --quiet 'happy/experiment-build-manifest.txt' ||
    fail 'archive is missing the prepared build manifest'

set +e
FAIL_CPU_RUN=1 MARKER_ROOT="$marker_root" "$runner" run \
    --run-id failed-run \
    --workspace "$guest_workspace" \
    --model "$model" \
    --output-root "$output_root" \
    --expected-bundle-id "$bundle_id"
failed_status=$?
set -e
[[ $failed_status -eq 7 ]] || fail "expected preserved CPU status 7, got $failed_status"
assert_file "$output_root/failed-run.tar.gz"
assert_file "$output_root/failed-run/npu/run-5/output/log/debug-log.jsonl"
assert_contains "$output_root/failed-run/manifest.txt" 'first_failure=7'
assert_contains "$output_root/failed-run/manifest.txt" \
    'comparison_status=not_evaluated_variant_failure'

set +e
CONTAMINATE_NPU=1 MARKER_ROOT="$marker_root" "$runner" run \
    --run-id contaminated \
    --workspace "$guest_workspace" \
    --model "$model" \
    --output-root "$output_root" \
    --expected-bundle-id "$bundle_id"
contaminated_status=$?
set -e
[[ $contaminated_status -eq 3 ]] ||
    fail "expected contaminated NPU status 3, got $contaminated_status"
assert_file "$output_root/contaminated.tar.gz"
assert_contains "$output_root/contaminated/manifest.txt" \
    'comparison_status=invalid_npu_telemetry'

set +e
MALFORM_NPU=1 MARKER_ROOT="$marker_root" "$runner" run \
    --run-id malformed \
    --workspace "$guest_workspace" \
    --model "$model" \
    --output-root "$output_root" \
    --expected-bundle-id "$bundle_id"
malformed_status=$?
set -e
[[ $malformed_status -eq 3 ]] ||
    fail "expected malformed NPU status 3, got $malformed_status"
assert_file "$output_root/malformed.tar.gz"
assert_contains "$output_root/malformed/manifest.txt" \
    'comparison_status=invalid_npu_telemetry'

set +e
MISMATCH_NPU_PROOF=1 MARKER_ROOT="$marker_root" "$runner" run \
    --run-id proof-mismatch \
    --workspace "$guest_workspace" \
    --model "$model" \
    --output-root "$output_root" \
    --expected-bundle-id "$bundle_id" \
    --runs-per-backend 1
proof_status=$?
set -e
[[ $proof_status -eq 3 ]] ||
    fail "expected proof mismatch status 3, got $proof_status"
assert_file "$output_root/proof-mismatch.tar.gz"
assert_contains "$output_root/proof-mismatch/manifest.txt" \
    'comparison_status=invalid_proof_mismatch'

set +e
MISMATCH_NPU_GEOMETRY=1 MARKER_ROOT="$marker_root" "$runner" run \
    --run-id geometry-mismatch \
    --workspace "$guest_workspace" \
    --model "$model" \
    --output-root "$output_root" \
    --expected-bundle-id "$bundle_id" \
    --runs-per-backend 1
geometry_status=$?
set -e
[[ $geometry_status -eq 3 ]] ||
    fail "expected geometry mismatch status 3, got $geometry_status"
assert_contains "$output_root/geometry-mismatch/manifest.txt" \
    'comparison_status=invalid_proof_mismatch'

set +e
FAIL_COPY=1 REAL_CP="$real_cp" REAL_TAR="$real_tar" PATH="$mock_bin:$PATH" \
MARKER_ROOT="$marker_root" "$runner" run \
    --run-id copy-failure \
    --workspace "$guest_workspace" \
    --model "$model" \
    --output-root "$output_root" \
    --expected-bundle-id "$bundle_id" \
    --runs-per-backend 1
copy_failure_status=$?
set -e
[[ $copy_failure_status -eq 3 ]] ||
    fail "expected artifact copy status 3, got $copy_failure_status"
assert_file "$output_root/copy-failure.tar.gz"
assert_contains "$output_root/copy-failure/manifest.txt" \
    'comparison_status=invalid_artifact_copy'

set +e
FAIL_TAR=1 REAL_TAR="$real_tar" REAL_CP="$real_cp" PATH="$mock_bin:$PATH" \
MARKER_ROOT="$marker_root" "$runner" run \
    --run-id archive-failure \
    --workspace "$guest_workspace" \
    --model "$model" \
    --output-root "$output_root" \
    --expected-bundle-id "$bundle_id" \
    --runs-per-backend 1
archive_failure_status=$?
set -e
[[ $archive_failure_status -eq 9 ]] ||
    fail "expected tar status 9, got $archive_failure_status"
[[ ! -e $output_root/archive-failure.tar.gz ]] ||
    fail 'partial archive was not removed'
assert_file "$output_root/archive-failure/manifest.txt"

printf 'test-gemmini-rmd-cpu-npu-run: PASS\n'
