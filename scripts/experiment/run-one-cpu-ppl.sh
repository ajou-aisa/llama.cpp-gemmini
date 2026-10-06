#!/usr/bin/env bash
set -euo pipefail

usage() {
    rtk proxy cat <<'EOF'
Usage: run-one-cpu-ppl.sh <model-condition> [--dry-run] [--smoke]
  <model-condition>: gpt2-rtn4, gpt2-rtn8, gpt2-potal16, gpt2-potal64,
                     llama-rtn4, llama-rtn8, llama-potal16, llama-potal64,
                     gpt2-rtnw4, gpt2-rtnw8, llama-rtnw4, llama-rtnw8
  Default: full WikiText-2 test, context/batch 512, --chunks -1, one condition.
  --dry-run  Print/resolve the build and commands without writing files.
  --smoke    Context 32 and one chunk; NOT full PPL.
Environment:
  Models: verified models/default GGUFs only; model-path overrides forbidden.
  DATASET     default: wikitext-2-raw/wiki.test.raw
  THREADS=4   BUILD_JOBS=4   RESULTS_DIR=<new directory>
  GEMMINI_SW_PATH  default: ../RISC-V-DynDNN-gemmini-include
Embedding/head retain their original GGUF types; KV caches are F16.
No quantization, head replacement, or new model file is performed. RTN4/RTN8 use
Gemmini CPU INT BLOCK32, matched A4/W4 or A8/W8, RMD OFF. Q4_0/Q8_0
codes and their original FP16 block scales are used as stored (H0).
RTN-W (rtnw4/rtnw8, nominal 16/4 or 16/8) uses Gemmini CPU FLOAT,
RMD OFF, and no activation quantization. Actual activation and arithmetic
are FP32 in develop; this is NOT a rounded FP16 activation experiment.
The matched n/n build widths only satisfy the unused integer-path contract.
Weights stay Q4_0/Q8_0 and decode to FP32 for the FLOAT matmul.
PoTal uses Q4_HP1, EXSIA sigma=2, RMD CPU, STRIPE_PIPELINE.
DIM changes host geometry; these CPU runs do not simulate hardware SCUs.
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
case_id=$1
shift
dry_run=0
context=512
chunks=-1
scope=full
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        --smoke) context=32; chunks=1; scope=smoke ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

case "$case_id" in
    gpt2-rtn4|llama-rtn4) model=${case_id%%-*}; method=rtn; bits=4; dim=16 ;;
    gpt2-rtn8|llama-rtn8) model=${case_id%%-*}; method=rtn; bits=8; dim=16 ;;
    gpt2-rtnw4|llama-rtnw4) model=${case_id%%-*}; method=rtn-w; bits=4; dim=16 ;;
    gpt2-rtnw8|llama-rtnw8) model=${case_id%%-*}; method=rtn-w; bits=8; dim=16 ;;
    gpt2-potal16|llama-potal16) model=${case_id%%-*}; method=potal; bits=4; dim=16 ;;
    gpt2-potal64|llama-potal64) model=${case_id%%-*}; method=potal; bits=4; dim=64 ;;
    *) printf 'Unknown condition: %s\n' "$case_id" >&2; usage >&2; exit 2 ;;
esac

compute=INT
nominal_activation_bits=$bits
runtime_activation_bits=$bits
runtime_activation_precision="INT$bits"
weight_decode=native_quantized_matmul
if [[ $method == rtn-w ]]; then
    compute=FLOAT
    nominal_activation_bits=16
    runtime_activation_bits=32
    runtime_activation_precision=F32
    weight_decode=dequantize_to_F32_for_matmul
fi

repo_root=$(cd -- "$(rtk proxy dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd -- "$repo_root"
runner_path="$repo_root/scripts/experiment/run-one-cpu-ppl.sh"
source "$repo_root/scripts/experiment/ppl-original-models.sh"
dataset=${DATASET:-$repo_root/wikitext-2-raw/wiki.test.raw}
threads=${THREADS:-4}
jobs=${BUILD_JOBS:-4}
[[ $threads =~ ^[1-9][0-9]*$ && $jobs =~ ^[1-9][0-9]*$ ]] || {
    printf 'THREADS and BUILD_JOBS must be positive integers.\n' >&2; exit 2;
}
quant="Q${bits}_0"
[[ $method != potal ]] || quant=Q4_HP1
ppl_original_model "$repo_root" "$model" "$quant"
source_model=$original_model
for input in "$source_model" "$dataset"; do
    [[ -s $input ]] || { printf 'Missing/empty input: %s\n' "$input" >&2; exit 2; }
done
results=${RESULTS_DIR:-$repo_root/output/experiment/cpu-ppl-$case_id-$scope-$(rtk proxy date -u +%Y%m%d-%H%M%S)-$$}
[[ $results = /* ]] || results="$repo_root/$results"
[[ ! -e $results ]] || { printf 'RESULTS_DIR must be new: %s\n' "$results" >&2; exit 2; }
source_sha=$(rtk proxy git rev-parse HEAD)
required_source_sha=a42ed663b490f09166585077ed0c0f898427d5d4
required_include_sha=b1bc70d74d1164650d400dd35fa87c3923e35f2f
include_dir=${GEMMINI_SW_PATH:-$repo_root/../RISC-V-DynDNN-gemmini-include}
[[ -d $include_dir ]] || { printf 'Missing Gemmini include directory: %s\n' "$include_dir" >&2; exit 2; }
include_dir=$(cd -- "$include_dir" && pwd -P)
include_header=
for candidate in "$include_dir/gemmini.h" "$include_dir/include/gemmini.h" "$include_dir/gemmini-rocc-tests/include/gemmini.h"; do
    [[ -f $candidate ]] || continue
    include_header=$candidate
    break
done
[[ -n $include_header ]] || { printf 'Missing gemmini.h: %s\n' "$include_dir" >&2; exit 2; }
rtk proxy git merge-base --is-ancestor "$required_source_sha" HEAD || {
    printf 'Source needs merged as-stored Q4_0/Q8_0 support (%s).\n' "$required_source_sha" >&2; exit 2;
}
rtk proxy git -C "$include_dir" merge-base --is-ancestor "$required_include_sha" HEAD || {
    printf 'Gemmini include needs merged H0 BLOCK support (%s).\n' "$required_include_sha" >&2; exit 2;
}
source_backend="$repo_root/ggml/src/ggml-gemmini/ggml-gemmini.cpp"
if ! rtk proxy grep -Fq 'gemmini_q4_0_q8_0_layout_contract' "$source_backend" ||
    rtk proxy grep -Eq 'prepare_q[48]_0_rows_for_q[48]_h1' "$source_backend"; then
    printf 'Working source does not preserve Q4_0/Q8_0 as stored.\n' >&2; exit 2
fi
for marker in 'cpu.Q4_0 tiled matmul' 'cpu.Q8_0 tiled matmul' 'block_scales->scale'; do
    rtk proxy grep -Fq "$marker" "$include_header" || {
        printf 'Gemmini header lacks H0/BLOCK support: %s (%s)\n' "$marker" "$include_header" >&2; exit 2;
    }
done
include_sha=$(rtk proxy git -C "$include_dir" rev-parse HEAD)
include_git_root=$(rtk proxy git -C "$include_dir" rev-parse --show-toplevel)
runner_hash=$(rtk proxy shasum -a 256 "$runner_path")
runner_hash=${runner_hash%% *}
include_headers=("$include_header")
for candidate in "$include_dir/gemmini.h" "$include_dir/gemmini_params.h" "$include_dir/include/gemmini.h" "$include_dir/include/gemmini_params.h"; do
    [[ ! -f $candidate || $candidate == "$include_header" ]] || include_headers+=("$candidate")
done

for name in $(compgen -v); do
    case "$name" in
        GGML_*|GEMMINI_*|IM2P_*|LLAMA_*|LOG_*|CYCLE_*|CMAKE_*|OpenMP_ROOT|BUILD_SHARED_LIBS)
            unset "$name" ;;
    esac
done
export OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE
export BUILD_JOBS="$jobs"
export BUILD_DIR="$results/build"
common=(
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
    "-DGEMMINI_SW_PATH=$include_dir"
    -DGGML_GEMMINI_OPTION=CPU -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE
    "-DGGML_GEMMINI_COMPUTE_TYPE=$compute" -DGGML_GEMMINI_DEQUANT_FP_TEST=OFF
    -DGGML_GEMMINI_BLOCK_SIZE=32 -DGGML_GEMMINI_EXSIA_SIGMA=2
    -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU
    -DGGML_GEMMINI_EXSIA_DEFAULT_MODE=LOCAL_FOLDING_PIPELINE
    -DGGML_GEMMINI_EXSIA_LOCAL_WORKERS=3 -DGGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY=2
    -DGGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE=OFF
    -DGGML_OPENMP=OFF -DGGML_GEMMINI_ENABLE_OPENMP=ON -DGGML_LLAMAFILE=OFF
    -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
    -DLOG_DEBUG=0 -DLOG_CYCLE=0 -DCYCLE_DETAIL=0 -DCYCLE_SIM=0
    -DGGML_CPU_CYCLE_LOG=0 -DLOG_DUMP=0 -DLOG_DUMP_SCALE=0
    -DGGML_GEMMINI_ACT_QUANT_METRICS=0 -DGGML_GEMMINI_RESIDUAL_METRICS=0
    -DGGML_GEMMINI_EXSIA_PROFILE_SCOPE=OFF -DGGML_GEMMINI_PRINT_TILE=0
)
printf 'Results: %s\nCondition: %s, scope=%s, context=%s, chunks=%s\n' "$results" "$case_id" "$scope" "$context" "$chunks"
if (( ! dry_run )); then
    rtk proxy mkdir -p -- "$(rtk proxy dirname -- "$results")"
    rtk proxy mkdir -- "$results"
    printf 'model\tmethod\tbits\tdim\tchunks\tscored_tokens\tppl\tscope\n' > "$results/results.tsv"
    { printf 'source_sha=%s\n' "$source_sha"; rtk proxy shasum -a 256 "$source_model" "$dataset"; } > "$results/inputs.txt"
    rtk proxy git diff HEAD > "$results/source.diff"
    source_diff_hash=$(rtk proxy shasum -a 256 "$results/source.diff")
    source_diff_hash=${source_diff_hash%% *}
    rtk proxy git -C "$include_dir" diff HEAD > "$results/gemmini-include.diff"
    rtk proxy shasum -a 256 "${include_headers[@]}" > "$results/gemmini-headers.sha256"
    rtk proxy cp -- "$runner_path" "$results/runner.sh"
fi

manifest=
trap 'status=$?; if [[ -n $manifest ]]; then printf "exit_status=%s\n" "$status" >> "$manifest"; fi; if (( status != 0 )); then printf "Stopped (exit %s). Logs: %s\n" "$status" "$results" >&2; fi' EXIT
run_logged() {
    local log=$1
    shift
    printf '%q ' rtk proxy "$@" >> "$manifest"
    printf '\n' >> "$manifest"
    rtk proxy "$@" 2>&1 | rtk proxy tee "$log"
}

quant="Q${bits}_0"
activation=BLOCK; rmd=OFF; mode=FULL; pipeline=OFF; reprocess=none; weight_family=H0
gemmini=ON
if [[ $method == potal ]]; then
    quant=Q4_HP1
    activation=EXSIA; rmd=ON; mode=STRIPE_PIPELINE; pipeline=ON; weight_family=HP1
fi
model_file=$source_model
flags=("${common[@]}" "-DGGML_GEMMINI=$gemmini"
    "-DGGML_GEMMINI_ACTIVATION_BITS=$bits" "-DGGML_GEMMINI_WEIGHT_BITS=$bits"
    "-DGGML_GEMMINI_DIM=$dim" "-DGGML_GEMMINI_ACTIVATION_QUANT=$activation"
    "-DGGML_GEMMINI_ENABLE_RMD=$rmd" "-DGGML_GEMMINI_DEFAULT_MATMUL_MODE=$mode"
    "-DGGML_GEMMINI_ENABLE_STRIPE_MATMUL=$pipeline" "-DGGML_GEMMINI_ENABLE_STRIPE_PIPELINE=$pipeline")
evaluate=("$BUILD_DIR/bin/llama-perplexity" -m "$model_file" -f "$dataset"
    --device none -ngl 0 -t "$threads" -tb "$threads" -c "$context" -b "$context" -ub "$context"
    --chunks "$chunks" --ppl-stride 0 --ppl-output-type 1 --no-warmup
    --cache-type-k f16 --cache-type-v f16 --seed 42)
activation_path=$activation
if [[ $method == rtn-w ]]; then
    activation_path=NONE
fi
printf 'activation=%s A/W=%s/%s RMD=%s weight=%s family=%s weight_reprocess=%s\n' "$activation_path" "$nominal_activation_bits" "$bits" "$rmd" "$quant" "$weight_family" "$reprocess"
printf 'compute=%s runtime_activation=%s runtime_activation_bits=%s build_A/W=%s/%s weight_decode=%s\n' "$compute" "$runtime_activation_precision" "$runtime_activation_bits" "$bits" "$bits" "$weight_decode"
printf 'source_sha=%s gemmini_include_sha=%s gemmini_header=%s runner_sha256=%s\n' "$source_sha" "$include_sha" "$include_header" "$runner_hash"
if (( dry_run )); then
    rtk proxy bash "$repo_root/build-arm64-cpu.sh" --dry-run "${flags[@]}"
    printf 'evaluate: '; printf '%q ' "${evaluate[@]}"; printf '\n'
    exit 0
fi
[[ $(rtk proxy git rev-parse HEAD) == "$source_sha" ]] || { printf 'Source commit changed during run.\n' >&2; exit 1; }
manifest="$results/manifest.txt"
printf 'condition=%s\nscope=%s\nsource_sha=%s\nactivation=%s\nactivation_bits=%s\nweight_bits=%s\nweight_storage=%s\nweight_family=%s\nweight_runtime_reprocess=%s\nembedding=as_stored\nhead=as_stored\ncache_k=F16\ncache_v=F16\n' \
    "$case_id" "$scope" "$source_sha" "$activation_path" "$runtime_activation_bits" "$bits" "$quant" "$weight_family" "$reprocess" > "$manifest"
printf 'method=%s\ncompute=%s\nnominal_activation_bits=%s\nruntime_activation_precision=%s\nbuild_activation_bits=%s\nbuild_weight_bits=%s\nbuild_activation_quant=%s\nweight_runtime_decode=%s\n' \
    "$method" "$compute" "$nominal_activation_bits" "$runtime_activation_precision" "$bits" "$bits" "$activation" "$weight_decode" >> "$manifest"
printf 'required_source_sha=%s\nrequired_include_sha=%s\ngemmini_include_path=%s\ngemmini_include_git_root=%s\ngemmini_include_sha=%s\ngemmini_header=%s\nrunner_sha256=%s\n' \
    "$required_source_sha" "$required_include_sha" "$include_dir" "$include_git_root" "$include_sha" "$include_header" "$runner_hash" >> "$manifest"
rtk proxy shasum -a 256 "$results/source.diff" "$results/gemmini-include.diff" "$results/gemmini-headers.sha256" >> "$manifest"
export GEMMINI_LOG_DIR="$results" LOG_DIR="$results" OUTPUT_DIR="$results"
printf '[build] %s\n' "$case_id"
run_logged "$results/build.log" bash "$repo_root/build-arm64-cpu.sh" "${flags[@]}"
rtk proxy cp -- "$BUILD_DIR/CMakeCache.txt" "$results/CMakeCache.txt"
for flag in "${flags[@]}"; do
    setting=${flag#-D}; key=${setting%%=*}; value=${setting#*=}
    rtk proxy grep -Eq "^${key}:[^=]+=${value}$" "$results/CMakeCache.txt" || {
        printf 'Build setting mismatch: %s\n' "$setting" >&2; exit 1;
    }
done
actual_include=$(rtk proxy sed -n 's/^GEMMINI_SW_PATH:[^=]*=//p' "$results/CMakeCache.txt")
[[ -n $actual_include ]] || { printf 'Built CMakeCache lacks GEMMINI_SW_PATH.\n' >&2; exit 1; }
actual_include=$(cd -- "$actual_include" && pwd -P)
[[ $actual_include == "$include_dir" ]] || { printf 'Built Gemmini include path mismatch: %s\n' "$actual_include" >&2; exit 1; }
printf 'built_gemmini_include_path=%s\n' "$actual_include" >> "$manifest"
rtk proxy shasum -a 256 -c "$results/gemmini-headers.sha256" > "$results/gemmini-header-check.log"
[[ $(rtk proxy git rev-parse HEAD) == "$source_sha" ]] || { printf 'Source commit changed during build.\n' >&2; exit 1; }
current_source_diff_hash=$(rtk proxy git diff HEAD | rtk proxy shasum -a 256)
[[ ${current_source_diff_hash%% *} == "$source_diff_hash" ]] || { printf 'Tracked source changes changed during build.\n' >&2; exit 1; }
[[ $(rtk proxy git -C "$include_dir" rev-parse HEAD) == "$include_sha" ]] || { printf 'Gemmini include commit changed during build.\n' >&2; exit 1; }
current_runner_hash=$(rtk proxy shasum -a 256 "$runner_path")
[[ ${current_runner_hash%% *} == "$runner_hash" ]] || { printf 'Runner changed during build.\n' >&2; exit 1; }
printf '[model] existing original %s\n' "$model_file"
rtk proxy shasum -a 256 "$model_file" "$BUILD_DIR/bin/llama-perplexity" "$BUILD_DIR/bin/"libggml* >> "$manifest"
printf '[PPL] %s\n' "$case_id"
run_logged "$results/ppl.log" "${evaluate[@]}"
if [[ $gemmini == ON ]]; then
    rtk proxy grep -q 'GEMMINI model buffer size' "$results/ppl.log" || {
        printf 'Model did not use the GEMMINI CPU backend.\n' >&2; exit 1;
    }
fi
rtk proxy awk -v model="$model" -v method="$method" -v bits="$bits" -v dim="$dim" -v scope="$scope" -v ctx="$context" '
    /calculating perplexity over/ { for (i=1;i<=NF;i++) if ($i=="chunks,") planned=$(i-1) }
    /PPL_CHUNK / { measured++ }
    /PPL_TOTALS / {
        found++
        for (i=1;i<=NF;i++) { split($i,a,"="); values[a[1]]=a[2] }
    }
    END {
        n=values["chunks"]+0; tokens=values["total_scored_tokens"]+0; ppl=values["corpus_ppl"]
        if (found!=1 || n<=0 || n!=planned || n!=measured || tokens!=n*(ctx/2-1) ||
            ppl !~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ || ppl+0<=0) exit 1
        printf "%s\t%s\t%d\t%d\t%d\t%d\t%s\t%s\n", model,method,bits,dim,n,tokens,ppl,scope
    }' "$results/ppl.log" | rtk proxy tee -a "$results/results.tsv"
printf 'exit_status=0\n' >> "$manifest"
manifest=
printf 'Completed: %s/results.tsv\n' "$results"
