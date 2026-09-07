#!/bin/bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
runner=$repo_root/scripts/experiment/run-cycle-matrix.sh
fixtures=$repo_root/tests/fixtures/cycle-experiment
root=$(mktemp -d "${TMPDIR:-/tmp}/cycle-matrix.XXXXXX")
trap 'rm -rf "$root"' EXIT
# Keep the shipped legacy fixture intact; successful runs need explicit CPU validity.
/usr/bin/python3 - "$fixtures/task-7-telemetry.jsonl" "$root/complete-telemetry.jsonl" <<'PY'
import json, pathlib, sys
with pathlib.Path(sys.argv[2]).open("w") as output:
    for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
        record = json.loads(line)
        if record["record_type"] == "RMD_BACKEND_TELEMETRY":
            record.update(invocation_total_valid=True, invocation_total_reason=None,
                          invocation_total_sample_reason=None, invocation_total_count=1,
                          invocation_total_valid_count=1, invocation_total_not_applicable_count=0)
        output.write(json.dumps(record) + "\n")
PY
fail() { printf 'test-cycle-experiment-runner: %s\n' "$*" >&2; exit 1; }
assert_dead() {
    local pid=$1 label=${2:-process} state
    if kill -0 "$pid" 2>/dev/null; then
        state=$(ps -o stat= -p "$pid" 2>/dev/null || true)
        [[ $state = Z* ]] || fail "$label survived: $pid ($state)"
    fi
}
make_inputs() {
    local base=$1 width build model
    mkdir -p "$base/models" "$base/build q4/bin" "$base/build q8/bin" "$base/build q16/bin"
    for model in gpt2.Q4_0.gguf gpt2.Q4_HP1.gguf gpt2.i8_tensor.gguf gpt2.Q8_HP1.gguf gpt2.Q16_0.gguf gpt2.Q16_HP1.gguf; do printf '%s\n' "$model" > "$base/models/$model"; done
    for width in 4 8 16; do
        case $width in 4) build="$base/build q4";; 8) build="$base/build q8";; 16) build="$base/build q16";; esac
        cp "$fixtures/task-7-fake-llama.sh" "$build/bin/llama-cli"; chmod +x "$build/bin/llama-cli"
        printf '%s\n' "GGML_GEMMINI_ACTIVATION_BITS:STRING=$width" "GGML_GEMMINI_WEIGHT_BITS:STRING=$width" 'LOG_DEBUG:STRING=1' 'LOG_CYCLE:STRING=1' 'CYCLE_DETAIL:STRING=1' 'GGML_CPU_CYCLE_LOG:BOOL=ON' 'GGML_OPENMP:BOOL=ON' 'GGML_GEMMINI_ENABLE_OPENMP:BOOL=ON' 'GGML_GEMMINI_OPTION:STRING=CPU' 'GGML_GEMMINI_EXECUTION_BACKEND:STRING=HARDWARE' 'GGML_GEMMINI_EXSIA_PROFILE_SCOPE:STRING=STAGE' > "$build/CMakeCache.txt"
    done
}
invoke() {
    local base=$1 stamp=$2; shift 2
    local launch=(env OUTPUT_ROOT="$base/output with spaces" MODEL_ROOT="$base" Q4_BUILD_ROOT="$base/build q4" Q8_BUILD_ROOT="$base/build q8" Q16_BUILD_ROOT="$base/build q16" MATRIX_TIMESTAMP="$stamp" MATRIX_WARMUPS=1 MATRIX_REPEATS=1 MATRIX_OMP_THREADS=2 MATRIX_CPUS=0,1 MATRIX_OMP_PLACES='{0},{1}' MATRIX_READY_TIMEOUT=2 MATRIX_WORKLOAD_TIMEOUT=4 MATRIX_LEAF_OPERATION=leaf MATRIX_TEGRastats="$fixtures/task-7-fake-tegrastats.sh" PYTHON_EXECUTABLE=/usr/bin/python3 FAKE_RUN_LOG="$base/workloads.tsv" FAKE_TELEMETRY_FIXTURE="$root/complete-telemetry.jsonl" "$@" "$runner" -- --prompt 'argument with spaces' --seed 17)
    if [[ ${INVOKE_EXEC:-0} = 1 ]]; then exec "${launch[@]}"; else "${launch[@]}"; fi
}
happy=$root/happy; make_inputs "$happy"
invoke "$happy" 20260903-010203Z >"$happy/out" 2>"$happy/err" || fail 'happy matrix failed'
experiment="$happy/output with spaces/experiment/20260903-010203Z"
[[ $(find "$experiment/runs" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ') = 6 ]] || fail 'six cells missing'
[[ $(find "$experiment/runs" -type d -name 'repeat-*' | wc -l | tr -d ' ') = 6 ]] || fail 'six repeats missing'
[[ $(grep -c $'^workload\twarmup\t' "$happy/workloads.tsv") = 6 ]] || fail 'warmups missing'
[[ $(grep -c $'^workload\tmeasured\t' "$happy/workloads.tsv") = 6 ]] || fail 'measured runs missing'
for run in "$experiment"/runs/*/repeat-001; do
    for file in raw/stdout.txt raw/stderr.txt raw/cycles.jsonl raw/frequency.txt raw/debug/debug-log.jsonl csv/frequency.csv csv/affinity.csv tables/operation-cycles.csv tables/operation-cycles.md tables/operation-cycles.svg tables/worker-cycles.csv tables/worker-cycles.md tables/worker-cycles.svg tables/caller-envelope-cycles.csv tables/pipeline-envelope-ns.csv provenance.txt status.txt summary-input.csv command.txt route-observations.txt; do [[ -f $run/$file ]] || fail "missing $run/$file"; done
    grep -F 'validation_status=valid' "$run/route-observations.txt" >/dev/null || fail "route was not validated: $run"
done
for file in runs.csv matrix-summary.csv matrix-summary.md matrix-summary.svg failures.csv; do [[ -f $experiment/summary/$file ]] || fail "missing summary/$file"; done
/usr/bin/python3 - "$experiment" <<'PY'
import csv, pathlib, sys, xml.etree.ElementTree as ET
root = pathlib.Path(sys.argv[1])
assert [r["cell"] for r in csv.DictReader((root / "manifest/cases.csv").open())] == ["q4-baseline", "q4-hp1", "q8-baseline", "q8-hp1", "q16-baseline", "q16-hp1"]
runs = list(csv.DictReader((root / "summary/runs.csv").open()))
assert len(runs) == 6
assert all((r["cycle_source"], r["wall_source"], r["cycle_status"]) == ("linux_perf_cpu_cycles", "steady_clock", "complete") for r in runs)
assert all(r["cycle_source"] == "linux_perf_cpu_cycles" for r in csv.DictReader((root / "manifest/repeats.csv").open()))
for path in root.rglob("*.csv"): list(csv.reader(path.open()))
for path in root.rglob("*.svg"): ET.parse(path)
for path in root.rglob("*.md"): assert "|" in path.read_text()
PY
grep -F $'measured\targument with spaces' "$happy/workloads.tsv" >/dev/null || fail 'arguments split'
/usr/bin/python3 - "$experiment" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = {"q4-baseline": "q8_h1", "q4-hp1": "q8_hp1", "q8-baseline": "tensor_i8", "q8-hp1": "q8_hp1", "q16-baseline": "q8_h0", "q16-hp1": "q8_hp1"}
for cell, route in expected.items():
    fields = dict(line.split("=", 1) for line in (root / "runs" / cell / "repeat-001" / "route-observations.txt").read_text().splitlines())
    assert fields == {"expected_route": route, "observed_routes": route, "observation_count": "1", "validation_status": "valid"}
PY
for mode in suppress mismatch malformed; do
    route_failure=$root/route-$mode; make_inputs "$route_failure"
    case $mode in suppress) variable=FAKE_ROUTE_SUPPRESS_MODEL; stamp=20260903-011000Z;; mismatch) variable=FAKE_ROUTE_MISMATCH_MODEL; stamp=20260903-012000Z;; malformed) variable=FAKE_ROUTE_MALFORMED_MODEL; stamp=20260903-013000Z;; esac
    if invoke "$route_failure" "$stamp" "$variable=$route_failure/models/gpt2.Q4_0.gguf" >/dev/null 2>&1; then fail "$mode route evidence succeeded"; fi
    route_experiment=$(find "$route_failure/output with spaces/experiment" -mindepth 1 -maxdepth 1 -type d)
    [[ $(grep -c $'^workload\tmeasured\t' "$route_failure/workloads.tsv") = 6 ]] || fail "$mode route evidence stopped later cells"
    grep -F 'route validation failed' "$route_experiment/summary/failures.csv" >/dev/null || fail "$mode route failure was not summarized"
done
legacy=$root/legacy; make_inputs "$legacy"
if invoke "$legacy" 20260903-014000Z FAKE_TELEMETRY_FIXTURE="$fixtures/task-7-telemetry.jsonl" >/dev/null 2>&1; then fail 'legacy CPU totals were trusted'; fi
grep -F 'CPU cycle validity is not complete' "$legacy/output with spaces/experiment/20260903-014000Z/summary/failures.csv" >/dev/null || fail 'legacy validity reason missing'
for absent in binary cache model; do
    missing=$root/missing-$absent; make_inputs "$missing"
    case $absent in binary) rm "$missing/build q16/bin/llama-cli";; cache) rm "$missing/build q16/CMakeCache.txt";; model) rm "$missing/models/gpt2.Q16_HP1.gguf";; esac
    if invoke "$missing" 20260903-020304Z >/dev/null 2>&1; then fail "missing Q16 $absent succeeded"; fi
    [[ ! -e $missing/workloads.tsv ]] || fail "missing Q16 $absent started workload"
done
failure=$root/failure; make_inputs "$failure"
if invoke "$failure" 20260903-030405Z FAKE_FAIL_MODEL="$failure/models/gpt2.i8_tensor.gguf" >/dev/null 2>&1; then fail 'failed workload succeeded'; fi
failed="$failure/output with spaces/experiment/20260903-030405Z"
[[ $(grep -c $'^workload\tmeasured\t' "$failure/workloads.tsv") = 6 ]] || fail 'later cells skipped'
grep -F 'workload failed' "$failed/summary/failures.csv" >/dev/null || fail 'failure not summarized'
malformed=$root/malformed; make_inputs "$malformed"
if invoke "$malformed" 20260903-040506Z FAKE_MALFORMED_MODEL="$malformed/models/gpt2.Q8_HP1.gguf" >/dev/null 2>&1; then fail 'malformed telemetry succeeded'; fi
[[ $(grep -c $'^workload\tmeasured\t' "$malformed/workloads.tsv") = 6 ]] || fail 'malformed telemetry stopped matrix'
collision=$root/collision; make_inputs "$collision"; mkdir -p "$collision/output with spaces/experiment/20260903-050607Z"; printf keep > "$collision/output with spaces/experiment/20260903-050607Z/keep"
if invoke "$collision" 20260903-050607Z >/dev/null 2>&1; then fail 'collision succeeded'; fi
[[ $(cat "$collision/output with spaces/experiment/20260903-050607Z/keep") = keep && ! -e $collision/workloads.tsv ]] || fail 'collision mutated output'
signal_root=$root/signal; make_inputs "$signal_root"; mkfifo "$signal_root/ready" "$signal_root/block"; exec 8<>"$signal_root/ready"; exec 9<>"$signal_root/block"
INVOKE_EXEC=1 invoke "$signal_root" 20260903-060708Z FAKE_HOLD_MODEL="$signal_root/models/gpt2.Q4_0.gguf" FAKE_READY_FIFO="$signal_root/ready" FAKE_BLOCK_FIFO="$signal_root/block" FAKE_LLAMA_PID_FILE="$signal_root/llama.pid" FAKE_DESCENDANT_PID_FILE="$signal_root/descendant.pid" FAKE_TEGRASTATS_PID_FILE="$signal_root/tegra.pid" >"$signal_root/out" 2>"$signal_root/err" & runner_pid=$!
IFS= read -r -t 10 ready <&8 || fail 'signal fixture readiness timed out'; [[ $ready = ready ]] || fail 'signal fixture not ready'; kill -TERM "$runner_pid"
set +e; wait "$runner_pid"; signal_status=$?; set -e
[[ $signal_status = 143 ]] || fail "SIGTERM status was $signal_status"
assert_dead "$(cat "$signal_root/llama.pid")" llama; assert_dead "$(cat "$signal_root/descendant.pid")" descendant; assert_dead "$(cat "$signal_root/tegra.pid")" tegrastats
[[ -f "$signal_root/output with spaces/experiment/20260903-060708Z/runs/q4-baseline/repeat-001/raw/stdout.txt" ]] || fail 'partial signal evidence missing'
exec 8>&- 9>&-
second=$root/second; make_inputs "$second"; invoke "$second" 20260903-070809Z >/dev/null 2>&1 || fail 'fresh repeated invocation failed'
printf 'test-cycle-experiment-runner: PASS\n'
