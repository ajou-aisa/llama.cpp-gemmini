#!/usr/bin/env bash
set -euo pipefail

usage() {
    rtk proxy cat <<'EOF'
Usage: run-one-metal-ppl.sh <gpt2|llama> --model existing.gguf [options]
  --bits 4|8                   Matched activation/weight bits (default: 4)
  --activation BLOCK|EXSIA     BLOCK requires Qn_0; EXSIA requires Qn_HP1
  --dim 16|32|64              HP1 fragment geometry (default: 16)
  --rmd ON|OFF                Default: BLOCK OFF, EXSIA ON
  --file dataset              Default: wikitext-2-raw/wiki.test.raw
  --context N --chunks N      Default: context 512, all chunks (-1)
  --threads N --jobs N        Default: 4 and 4
  --build-dir directory       Reusable, separate CMake build directory
  --results-dir directory     New result directory; existing paths rejected
  --dry-run                   Validate inputs and print commands; write nothing
Requires macOS, CMake, rtk, Python 3 + numpy, and Gemmini parameter headers.
GEMMINI_SW_PATH and OpenMP_ROOT may select those headers and libomp.
PYTHON selects Python. Otherwise search repo/sibling .venv and python3 for numpy.
Uses existing GGUF bytes. No model conversion or re-quantization occurs.
CPU produces quantized activations; custom Metal performs integer matmul
and soft-FP64 restoration. Successful runs require scheduler/kernel coverage
proof for every observed quantized matmul in this PPL invocation.
EOF
}

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
case "$1" in
    -h|--help) usage; exit 0 ;;
    gpt2|llama) model_family=$1; shift ;;
    *) usage >&2; exit 2 ;;
esac
repo_root=$(cd -- "$(rtk proxy dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
runner_path="$repo_root/scripts/experiment/run-one-metal-ppl.sh"
model=; dataset="$repo_root/wikitext-2-raw/wiki.test.raw"
bits=4; activation=BLOCK; dim=16; rmd=; context=512; chunks=-1
threads=4; jobs=4; build_dir=; results=; dry_run=0
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift; continue ;;
        -h|--help) usage; exit 0 ;;
        --model|--file|--bits|--activation|--dim|--rmd|--context|--chunks|--threads|--jobs|--build-dir|--results-dir)
            [[ $# -ge 2 ]] || { printf 'Missing value: %s\n' "$1" >&2; exit 2; }
            case "$1" in
                --model) model=$2 ;; --file) dataset=$2 ;; --bits) bits=$2 ;;
                --activation) activation=$2 ;; --dim) dim=$2 ;; --rmd) rmd=$2 ;;
                --context) context=$2 ;; --chunks) chunks=$2 ;; --threads) threads=$2 ;;
                --jobs) jobs=$2 ;; --build-dir) build_dir=$2 ;; --results-dir) results=$2 ;;
            esac
            shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ $(rtk proxy uname -s) == Darwin ]] || { printf 'This runner requires macOS Metal.\n' >&2; exit 2; }
[[ $bits =~ ^(4|8)$ && $dim =~ ^(16|32|64)$ && $activation =~ ^(BLOCK|EXSIA)$ ]] || {
    printf 'Supported profiles: A4W4/A8W8, BLOCK/EXSIA, DIM16/32/64.\n' >&2; exit 2;
}
[[ -n $rmd ]] || { rmd=OFF; [[ $activation != EXSIA ]] || rmd=ON; }
[[ $rmd =~ ^(ON|OFF)$ ]] || { printf 'RMD must be ON or OFF.\n' >&2; exit 2; }
for number in "$context" "$threads" "$jobs"; do
    [[ $number =~ ^[1-9][0-9]*$ ]] || { printf 'Context, threads and jobs must be positive integers.\n' >&2; exit 2; }
done
[[ $chunks == -1 || $chunks =~ ^[1-9][0-9]*$ ]] || { printf 'Chunks must be -1 or positive.\n' >&2; exit 2; }
[[ $context -ge 4 ]] || { printf 'PPL context must be at least 4.\n' >&2; exit 2; }
[[ -n $model && -s $model && -s $dataset ]] || { printf 'Provide an existing --model GGUF and nonempty --file dataset.\n' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
python=${PYTHON:-}
if [[ -z $python ]]; then
    for candidate in "$repo_root/.venv/bin/python" "$repo_root/../.venv/bin/python" python3; do
        if rtk proxy "$candidate" -c 'import numpy' >/dev/null 2>&1; then python=$candidate; break; fi
    done
fi
[[ -n $python ]] || { printf 'Set PYTHON to a Python 3 interpreter with numpy installed.\n' >&2; exit 2; }
quant="Q${bits}_0"
[[ $activation != EXSIA ]] || quant="Q${bits}_HP1"
profile="a${bits}w${bits}-${activation}-d${dim}-rmd${rmd}"
build_dir=${build_dir:-$repo_root/.omo/metal-quantized/ppl-build-$profile}
results=${results:-$repo_root/output/experiment/metal-ppl-$model_family-$profile-$(rtk proxy date -u +%Y%m%d-%H%M%S)-$$}
include_dir=${GEMMINI_SW_PATH:-$repo_root/../RISC-V-DynDNN-gemmini-include}
openmp_root=${OpenMP_ROOT:-/opt/homebrew/opt/libomp}
absolute_path() { rtk proxy "$python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1"; }
model=$(absolute_path "$model"); dataset=$(absolute_path "$dataset")
build_dir=$(absolute_path "$build_dir"); results=$(absolute_path "$results")
include_dir=$(absolute_path "$include_dir")
[[ $build_dir != "$repo_root" ]] || { printf 'Build directory must differ from source root.\n' >&2; exit 2; }
[[ ! -e $results ]] || { printf 'Result directory must be new: %s\n' "$results" >&2; exit 2; }
[[ -f $include_dir/gemmini.h && -f $include_dir/gemmini_params.h ]] || {
    printf 'Missing Gemmini parameter headers: %s\n' "$include_dir" >&2; exit 2;
}

model_metadata=$(rtk proxy "$python" - "$repo_root" "$model" "$model_family" "$quant" <<'PY'
import collections
import json
import sys
sys.path.insert(0, sys.argv[1] + '/gguf-py')
try:
    from gguf import GGUFReader, GGMLQuantizationType, LlamaFileType
except ImportError as error:
    raise SystemExit('GGUF admission requires Python 3 + numpy and bundled gguf-py: ' + str(error))
reader = GGUFReader(sys.argv[2])
architecture = reader.get_field('general.architecture')
file_type = reader.get_field('general.file_type')
expected = GGMLQuantizationType[sys.argv[4]]
if architecture is None or architecture.contents() != sys.argv[3]:
    raise SystemExit('GGUF architecture does not match the requested model family')
if file_type is None or file_type.contents() != int(LlamaFileType['MOSTLY_' + sys.argv[4]]):
    raise SystemExit('GGUF file type does not match the requested quantized profile')
counts = collections.Counter(t.tensor_type.name for t in reader.tensors)
allowed = {expected, GGMLQuantizationType.F16, GGMLQuantizationType.F32}
invalid = [(t.name, t.tensor_type.name) for t in reader.tensors if t.tensor_type not in allowed]
if invalid or counts[expected.name] == 0:
    raise SystemExit('GGUF tensor types do not match the requested profile: ' + str(invalid[:8]))
print(json.dumps({'architecture': architecture.contents(), 'file_type': file_type.contents(),
                  'tensor_types': dict(counts), 'expected_quantization': expected.name}, sort_keys=True))
PY
)
cd -- "$repo_root"
configure=(cmake -S "$repo_root" -B "$build_dir"
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF
    -DGGML_NATIVE=ON -DGGML_CPU=ON -DGGML_GEMMINI=OFF -DGGML_CUDA=OFF
    -DGGML_METAL=ON -DGGML_METAL_QUANTIZED=ON -DGGML_METAL_USE_BF16=OFF
    -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_METAL_NDEBUG=OFF -DGGML_BLAS=OFF
    -DGGML_ACCELERATE=ON -DGGML_OPENMP=OFF -DGGML_GEMMINI_ENABLE_OPENMP=ON
    "-DOpenMP_ROOT=$openmp_root" "-DGEMMINI_SW_PATH=$include_dir"
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
    -DGGML_GEMMINI_OPTION=CPU -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE
    -DGGML_GEMMINI_COMPUTE_TYPE=INT -DGGML_GEMMINI_DEQUANT_FP_TEST=OFF
    "-DGGML_GEMMINI_ACTIVATION_BITS=$bits" "-DGGML_GEMMINI_WEIGHT_BITS=$bits"
    "-DGGML_GEMMINI_ACTIVATION_QUANT=$activation" "-DGGML_GEMMINI_DIM=$dim"
    -DGGML_GEMMINI_BLOCK_SIZE=32 -DGGML_GEMMINI_EXSIA_SIGMA=2
    "-DGGML_GEMMINI_ENABLE_RMD=$rmd" -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU
    -DGGML_GEMMINI_DEFAULT_MATMUL_MODE=FULL -DGGML_GEMMINI_ENABLE_STRIPE_MATMUL=OFF
    -DGGML_GEMMINI_ENABLE_STRIPE_PIPELINE=OFF -DGGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE=OFF
    -DGGML_GEMMINI_EXSIA_DEFAULT_MODE=LOCAL_FOLDING_PIPELINE -DGGML_GEMMINI_EXSIA_LOCAL_WORKERS=3
    -DGGML_GEMMINI_EXSIA_PROFILE_SCOPE=OFF -DGGML_GEMMINI_ACT_QUANT_METRICS=0
    -DGGML_GEMMINI_RESIDUAL_METRICS=0 -DLOG_CYCLE=0 -DLOG_DEBUG=0
    -DCYCLE_SIM=0 -DCYCLE_DETAIL=0 -DGGML_CPU_CYCLE_LOG=0 -DLOG_DUMP=0 -DLOG_DUMP_SCALE=0)
build=(cmake --build "$build_dir" --target llama-perplexity -j "$jobs")
ppl=("$build_dir/bin/llama-perplexity" --model "$model" --file "$dataset"
    --ctx-size "$context" --batch-size "$context" --ubatch-size "$context"
    --chunks "$chunks" --threads "$threads" --threads-batch "$threads"
    --gpu-layers 999 --cache-type-k f16 --cache-type-v f16 --seed 42 --no-warmup)
print_command() { printf '%q ' rtk proxy "$@"; printf '\n'; }
printf 'Profile: %s %s\nModel: %s\nResults: %s\n%s\n' "$model_family" "$profile" "$model" "$results" "$model_metadata"
printf 'CPU activation producer; Metal integer matmul and soft-FP64 restoration.\n'
if (( dry_run )); then
    print_command "${configure[@]}"
    print_command "${build[@]}"
    print_command "${ppl[@]}"
    exit 0
fi

rtk proxy mkdir -p -- "$(rtk proxy dirname -- "$results")"
rtk proxy mkdir -- "$results"
manifest="$results/manifest.txt"
trap 'status=$?; printf "exit_status=%s\n" "$status" >> "$manifest"' EXIT
{
    printf 'model_family=%s\nprofile=%s\nactivation_bits=%s\nweight_bits=%s\ndim=%s\nrmd=%s\n' "$model_family" "$profile" "$bits" "$bits" "$dim" "$rmd"
    printf 'model=%s\ndataset=%s\ncontext=%s\nchunks=%s\nthreads=%s\nbuild_dir=%s\n' "$model" "$dataset" "$context" "$chunks" "$threads" "$build_dir"
    printf 'python=%s\n' "$python"
    printf 'producer=CPU\nmatmul=Metal\nrestoration=Metal_soft_FP64\nweight_conversion=none\nseed=42\nwarmup=OFF\nfallback_coverage=required_for_all_observed_quantized_matmuls\n'
    printf 'timing_scope=process_wall_including_model_load_and_CPU_producer\nsource_commit='
    rtk proxy git rev-parse HEAD
    print_command "${configure[@]}"; print_command "${build[@]}"; print_command "${ppl[@]}"
} > "$manifest"
printf '%s\n' "$model_metadata" > "$results/model-metadata.json"
rtk proxy git diff HEAD > "$results/source.diff"
rtk proxy git status --short > "$results/source-status.txt"
rtk proxy git ls-files -z --cached --others --exclude-standard CMakeLists.txt ggml/src/ggml-metal ggml/src/ggml-gemmini ggml/src/ggml-gemmini-utils ggml/CMakeLists.txt tools/perplexity tools/eval/evaluation-metal.hpp |
    rtk proxy xargs -0 rtk proxy shasum -a 256 > "$results/source-files.sha256"
rtk proxy shasum -a 256 "$model" "$dataset" "$include_dir/gemmini.h" "$include_dir/gemmini_params.h" "$runner_path" > "$results/inputs.sha256"
rtk proxy cp -- "$runner_path" "$results/runner.sh"
rtk proxy sw_vers > "$results/host.txt"
rtk proxy system_profiler SPDisplaysDataType >> "$results/host.txt"
rtk proxy xcrun clang --version > "$results/compiler.txt"
export OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE
rtk proxy "${configure[@]}" 2>&1 | rtk proxy tee "$results/configure.log"
rtk proxy "${build[@]}" 2>&1 | rtk proxy tee "$results/build.log"
rtk proxy cp -- "$build_dir/CMakeCache.txt" "$results/CMakeCache.txt"
rtk proxy shasum -a 256 "${ppl[0]}" "$build_dir/bin/"*.dylib > "$results/binary.sha256"
ppl_status=0
rtk proxy /usr/bin/time -p "${ppl[@]}" 2>&1 | rtk proxy tee "$results/ppl.log" || ppl_status=$?
printf 'ppl_exit_status=%s\n' "$ppl_status" >> "$manifest"
rtk proxy "$python" - "$results" "$bits" "$activation" "$dim" "$rmd" "$ppl_status" <<'PY'
import json
import pathlib
import sys
result_dir = pathlib.Path(sys.argv[1])
prefix = 'METAL_QUANTIZED_PROOF '
proof_lines = [line[len(prefix):] for line in (result_dir / 'ppl.log').read_text().splitlines() if line.startswith(prefix)]
if len(proof_lines) != 1:
    raise SystemExit('Expected one METAL_QUANTIZED_PROOF record; see ppl.log')
proof = json.loads(proof_lines[0])
(result_dir / 'metal-quantized-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
def require(condition, message):
    if not condition:
        raise SystemExit('Metal PPL coverage rejected: ' + message)
require(sys.argv[6] == '0', 'perplexity process failed')
integer_fields = ('version', 'activation_bits', 'weight_bits', 'dim', 'ppl_evaluated_tokens',
                  'observed_quantized_matmuls', 'metal_quantized_matmuls', 'prefill_matmuls', 'decode_matmuls',
                  'dense_launches', 'block_calls', 'hp1_calls', 'failed_calls', 'fallback_calls',
                  'merge_launches', 'residual_launches')
require(all(type(proof.get(key)) is int and proof[key] >= 0 for key in integer_fields),
        'missing or invalid integer profile/counter fields')
require(proof.get('schema') == 'metal-quantized-execution' and proof.get('version') == 1 and
        proof.get('source_role') == 'metal_quantized_ppl' and proof.get('producer') == 'cpu' and
        proof.get('backend') == 'Metal', 'wrong execution proof schema or backend')
require(proof.get('complete') is True and proof.get('placement_verified') is True, 'incomplete placement')
require(type(proof.get('ppl_evaluated_tokens')) is int and proof['ppl_evaluated_tokens'] > 0,
        'no PPL tokens evaluated')
require(proof.get('activation_bits') == proof.get('weight_bits') == int(sys.argv[2]) and
        proof.get('activation_mode') == sys.argv[3] and proof.get('dim') == int(sys.argv[4]) and
        proof.get('rmd_enabled') is (sys.argv[5] == 'ON'), 'compiled profile differs from requested profile')
observed = proof.get('observed_quantized_matmuls')
require(type(observed) is int and observed > 0, 'no quantized matmuls observed')
require(observed == proof.get('metal_quantized_matmuls') == proof.get('dense_launches') and
        proof.get('failed_calls') == proof.get('fallback_calls') == 0, 'missing kernels or failed/fallback calls')
require(proof['prefill_matmuls'] + proof['decode_matmuls'] == observed, 'phase counters do not cover all matmuls')
require((proof.get('block_calls'), proof.get('hp1_calls')) ==
        ((observed, 0) if sys.argv[3] == 'BLOCK' else (0, observed)), 'wrong custom kernel family')
require(proof.get('merge_launches') == proof.get('residual_launches') and
        (proof.get('residual_launches') == 0 or (sys.argv[3] == 'EXSIA' and sys.argv[5] == 'ON')),
        'residual execution does not match the profile')
require(proof.get('scheduler_observer') == 'ask_only_with_split_synchronization', 'missing scheduler observer')
placements = proof.get('tensor_placement')
expected_type = 'q' + sys.argv[2] + ('_0' if sys.argv[3] == 'BLOCK' else '_hp1')
require(isinstance(placements, list) and bool(placements), 'missing tensor placement')
require(all(isinstance(row, dict) and row.get('weight_backend') == row.get('output_backend') == 'Metal' and
            row.get('type') == expected_type and type(row.get('calls')) is int and row['calls'] > 0
            for row in placements) and sum(row['calls'] for row in placements) == observed,
        'tensor placement or counts do not cover every observed quantized matmul')
print('Verified Metal quantized PPL coverage: ' + str(observed) + ' matmuls')
PY
rtk proxy shasum -a 256 -c "$results/source-files.sha256" "$results/inputs.sha256" "$results/binary.sha256" > "$results/provenance-check.log"
printf 'fallback_coverage_verified=ALL_OBSERVED_QUANTIZED_MATMULS\n' >> "$manifest"
printf 'Completed. PPL and process time: %s/ppl.log\n' "$results"
