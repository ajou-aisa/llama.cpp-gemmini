#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

usage() {
    printf '%s\n' 'Usage: run-metal-quality-ppl.sh [--prepare] [--case ID | --from ID] [--output DIR] [--dry-run] [--context N] [--chunks N] [--build-dir DIR]' \
        'Default: all 32 GPT-2/Llama ablation cases, context 512, all WikiText-2 test chunks, 4 threads.' \
        'IDs: {gpt2,llama}-{rtnw4,rtnw8,rtnwa1-n4,rtnwa1-n8,rtnwa2-n{4,8}-d{16,32,64},potal{4,8}-d{16,32,64}}' \
        '--prepare builds the selected profiles using build-arm64.sh; it does not run PPL.' \
        '--from starts at that case in the default order; an interrupted case restarts from chunk 1.' \
        '--build-dir requires one --case; normal execution uses that build without reconfiguration.' \
        'Only SHA-256 verified models/default GGUFs are allowed; model overrides are forbidden.' \
        'CPU-equivalent Metal dense/attention matmuls and indexed HP1 residual with checked shifts.' \
        'Activation production, dense floating scale restoration and attention softmax remain on CPU.'
}
fail() { printf '%s\n' "$*" >&2; exit 2; }
root=$(cd -- "$(rtk proxy dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd -- "$root"
source "$root/scripts/experiment/ppl-original-models.sh"
selected=all; start_case=; output=; override=; build_root=; context=512; chunks=-1; dry=0; prepare=0; rtn_only=0
while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dry-run) dry=1; shift; continue ;;
        --prepare) prepare=1; shift; continue ;;
        --rtn-only) rtn_only=1; shift; continue ;;
        --case|--from|--output|--build-dir|--build-root|--context|--chunks)
            [[ $# -ge 2 ]] || fail "Missing value: $1"
            case "$1" in
                --case) selected=$2 ;; --from) start_case=$2 ;; --output) output=$2 ;; --build-dir) override=$2 ;;
                --build-root) build_root=$2 ;;
                --context) context=$2 ;; --chunks) chunks=$2 ;;
            esac
            shift 2 ;;
        *) fail "Unknown option: $1" ;;
    esac
done
[[ $context =~ ^[1-9][0-9]*$ && $context -ge 4 ]] || fail 'Context must be an integer >= 4.'
[[ $chunks == -1 || $chunks =~ ^[1-9][0-9]*$ ]] || fail 'Chunks must be -1 or positive.'
[[ -z $override || $selected != all ]] || fail '--build-dir requires one --case.'
[[ -z $start_case || $selected == all ]] || fail '--from and --case cannot be combined.'
[[ $(rtk proxy uname -s) == Darwin ]] || fail 'macOS Metal is required.'
cases=()
for method in rtnw rtnwa1 rtnwa2 potal; do
    [[ $method != potal && $method != rtnwa2 || $rtn_only == 0 ]] || continue
    for family in gpt2 llama; do
        for bits in 4 8; do
            dims=(16); [[ $method != potal && $method != rtnwa2 ]] || dims=(16 32 64)
            for dim in "${dims[@]}"; do
                id="$family-$method$bits"
                [[ $method != rtnwa1 && $method != rtnwa2 ]] || id="$family-$method-n$bits"
                [[ $method != potal && $method != rtnwa2 ]] || id="$id-d$dim"
                [[ $selected != all && $selected != "$id" ]] || cases+=("$id")
            done
        done
    done
done
(( ${#cases[@]} )) || fail "Unknown case: $selected"
if [[ -n $start_case ]]; then
    remaining=(); started=0
    for id in "${cases[@]}"; do
        [[ $id != "$start_case" ]] || started=1
        (( started == 0 )) || remaining+=("$id")
    done
    (( started )) || fail "Unknown starting case: $start_case"
    cases=("${remaining[@]}")
fi
dataset="$root/wikitext-2-raw/wiki.test.raw"
[[ -s $dataset ]] || fail "Missing dataset: $dataset"
profile() {
    id=$1; family=${id%%-*}; suffix=${id#*-}; bits=${suffix: -1}
    dim=16; activation=BLOCK; rmd=OFF; selection=OFF; compute=INT; method=RTN-WA1; stripe=OFF; matmul=FULL; q6_head=0
    if [[ $suffix == potal* ]]; then
        bits=${suffix:5:1}; dim=${suffix##*-d}; activation=EXSIA; rmd=ON; selection=ON; method=PoTal; stripe=ON; matmul=STRIPE_PIPELINE
    elif [[ $suffix == rtnwa2-* ]]; then
        bits=${suffix#rtnwa2-n}; bits=${bits%%-*}; dim=${suffix##*-d}; activation=EXSIA; method=RTN-WA2; stripe=ON; matmul=STRIPE_PIPELINE
    elif [[ $suffix == rtnw[48] ]]; then compute=FLOAT; method=RTN-W; fi
    build="$root/build-metal-ablation/quality-$suffix"
    [[ -z $build_root ]] || build="$build_root/quality-$suffix"
    build=${override:-$build}
    quant="Q${bits}_0"
    [[ $activation != EXSIA ]] || quant="Q${bits}_HP1"
    ppl_original_model "$root" "$family" "$quant"
    model=$original_model
    [[ $quant != Q4_0 ]] || q6_head=1
    activation_fp16=0
    [[ $compute != FLOAT ]] || activation_fp16=1
    [[ $activation_fp16 == 0 || $activation_fp16 == 1 ]] || fail 'FP16 activation flag must be 0 or 1.'
    command=(env OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE GGML_GEMMINI_METAL_CPU_EXACT=1 "GGML_GEMMINI_METAL_CPU_EXACT_Q6_HEAD=$q6_head" "GGML_GEMMINI_METAL_ACTIVATION_FP16=$activation_fp16")
    command+=(OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0)
    command+=("$build/bin/llama-perplexity"
        --model "$model" --file "$dataset" --ctx-size "$context" --batch-size "$context" --ubatch-size "$context"
        --chunks "$chunks" --threads 4 --threads-batch 4 --gpu-layers 0 --cache-type-k f16 --cache-type-v f16 --seed 42 --no-warmup)
}
build_command() {
    build_args=(env "BUILD_DIR=$build" "BUILD_JOBS=${BUILD_JOBS:-4}" bash "$root/build-arm64.sh"
        -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
        -DGGML_GEMMINI=ON -DGGML_GEMMINI_METAL_CPU_EXACT=ON
        -DGGML_METAL=OFF -DGGML_METAL_QUANTIZED=OFF -DGGML_CUDA=OFF -DGGML_BACKEND_DL=OFF
        -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF -DGGML_GEMMINI_ENABLE_OPENMP=ON
        -DGGML_GEMMINI_OPTION=CPU -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE
        "-DGGML_GEMMINI_COMPUTE_TYPE=$compute" -DGGML_GEMMINI_DEQUANT_FP_TEST=OFF
        "-DGGML_GEMMINI_ACTIVATION_QUANT=$activation" "-DGGML_GEMMINI_ACTIVATION_BITS=$bits"
        "-DGGML_GEMMINI_WEIGHT_BITS=$bits" "-DGGML_GEMMINI_DIM=$dim" -DGGML_GEMMINI_BLOCK_SIZE=32
        "-DGGML_GEMMINI_ENABLE_RMD=$rmd" -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU
        "-DGGML_GEMMINI_EXSIA_OUTLIER_SELECTION=$selection"
        "-DGGML_GEMMINI_ENABLE_STRIPE_MATMUL=$stripe" "-DGGML_GEMMINI_ENABLE_STRIPE_PIPELINE=$stripe"
        "-DGGML_GEMMINI_DEFAULT_MATMUL_MODE=$matmul" -DGGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY=2
        -DGGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE=OFF
        -DGGML_GEMMINI_EXSIA_SIGMA=2 -DGGML_GEMMINI_EXSIA_DEFAULT_MODE=LOCAL_FOLDING_PIPELINE
        -DGGML_GEMMINI_EXSIA_LOCAL_WORKERS=3 -DGGML_GEMMINI_EXSIA_PROFILE_SCOPE=OFF
        -DLOG_DEBUG=0 -DLOG_CYCLE=0 -DCYCLE_DETAIL=0 -DCYCLE_SIM=0 -DGGML_CPU_CYCLE_LOG=0
        -DLOG_DUMP=0 -DLOG_DUMP_SCALE=0 -DGGML_GEMMINI_ACT_QUANT_METRICS=0
        -DGGML_GEMMINI_RESIDUAL_METRICS=0 -DGGML_GEMMINI_PRINT_TILE=0 -DLLAMA_CURL=OFF)
}
if (( prepare )); then
    prepared='|'
    for id in "${cases[@]}"; do
        profile "$id"
        [[ $prepared != *"|$build|"* ]] || continue
        prepared="$prepared$build|"
        build_command
        if (( dry )); then printf '%q ' rtk proxy "${build_args[@]}"; printf '\n'
        else rtk proxy "${build_args[@]}"; fi
    done
    exit 0
fi
check_cache() {
    local actual
    actual=$(rtk proxy awk -F= -v key="$1" '$1 ~ "^" key ":" {print $2}' "$build/CMakeCache.txt")
    [[ $actual == "$2" ]] || fail "$id: $1 expected $2, found $actual ($build)"
}
for id in "${cases[@]}"; do
    profile "$id"
    [[ -s $model ]] || fail "$id: missing default model $model."
    if (( dry )) && [[ ! -x $build/bin/llama-perplexity || ! -s $build/CMakeCache.txt ]]; then continue; fi
    [[ -x $build/bin/llama-perplexity && -s $build/CMakeCache.txt && -s $model ]] || fail "$id: missing prepared binary, cache, or model"
    check_cache GGML_METAL OFF; check_cache GGML_GEMMINI ON; check_cache GGML_METAL_QUANTIZED OFF
    check_cache GGML_GEMMINI_METAL_CPU_EXACT ON; check_cache GGML_BACKEND_DL OFF
    check_cache GGML_CUDA OFF; check_cache GGML_BLAS OFF; check_cache GGML_LLAMAFILE OFF
    check_cache GGML_GEMMINI_EXSIA_OUTLIER_SELECTION "$selection"
    check_cache GGML_OPENMP OFF; check_cache CMAKE_BUILD_TYPE Release; check_cache LLAMA_BUILD_TESTS OFF
    for pair in "ACTIVATION_BITS:$bits" "WEIGHT_BITS:$bits" "ACTIVATION_QUANT:$activation" "DIM:$dim" "ENABLE_RMD:$rmd" "COMPUTE_TYPE:$compute" "ENABLE_STRIPE_MATMUL:$stripe" "ENABLE_STRIPE_PIPELINE:$stripe" "DEFAULT_MATMUL_MODE:$matmul" BLOCK_SIZE:32 DEQUANT_FP_TEST:OFF DEFAULT_RMD_BACKEND:CPU EXSIA_SIGMA:2 OPTION:CPU EXECUTION_BACKEND:HARDWARE ENABLE_OPENMP:ON EXSIA_DEFAULT_MODE:LOCAL_FOLDING_PIPELINE ALLOW_RUNTIME_MATMUL_OVERRIDE:OFF EXSIA_LOCAL_WORKERS:3 DEFAULT_STRIPE_JOB_CAPACITY:2; do
        check_cache "GGML_GEMMINI_${pair%%:*}" "${pair#*:}"
    done
done
print_command() { printf '%q ' rtk proxy /usr/bin/time -p "${command[@]}"; printf '\n'; }
if (( dry )); then
    for id in "${cases[@]}"; do profile "$id"; printf '%s\n' "$id"; print_command; done
    exit 0
fi
output=${output:-$root/output/experiment/metal-quality-ppl-$(rtk proxy date -u +%Y%m%d-%H%M%S)-$$}
[[ ! -e $output ]] || fail "Output directory must be new: $output"
rtk proxy mkdir -p -- "$(rtk proxy dirname -- "$output")"
rtk proxy mkdir -- "$output"
printf 'method\tmodel\tbits\tdim\tchunks\tscored_tokens\tppl\tprocess_seconds\texit\n' > "$output/results.tsv"
for id in "${cases[@]}"; do
    profile "$id"; result="$output/$id"
    rtk proxy mkdir -- "$result"
    print_command > "$result/command.txt"
    printf 'original_q6_k_head=%s\nactivation_fp16=%s\nmodel=%s\noutlier_selection=%s\nresidual_compensation=%s\n' "$q6_head" "$activation_fp16" "$model" "$selection" "$rmd" > "$result/head-policy.txt"
    rtk proxy cp -- "$build/CMakeCache.txt" "$result/CMakeCache.txt"
    rtk proxy cp -- "$build/compile_commands.json" "$result/compile_commands.json"
    rtk proxy cp -- "$build/ggml/src/ggml-gemmini/cpu-exact-build.txt" "$result/cpu-exact-build.txt"
    sibling=$(rtk proxy awk -F= '$1 == "gemmini_sw_path" {print substr($0, index($0, "=") + 1)}' "$result/cpu-exact-build.txt")
    [[ -s $sibling/gemmini.h ]] || fail "Missing CPU arithmetic source: $sibling/gemmini.h"
    rtk proxy shasum -a 256 "$sibling/gemmini.h" "$sibling/gemmini_params.h" > "$result/cpu-arithmetic.sha256"
    rtk proxy git rev-parse HEAD > "$result/source-commit.txt"
    rtk proxy git diff HEAD -- CMakeLists.txt ggml tools/perplexity tools/eval/evaluation-metal.hpp > "$result/source.diff"
    rtk proxy git ls-files -z --cached --others --exclude-standard CMakeLists.txt ggml/src/ggml-metal ggml/src/ggml-gemmini ggml/src/ggml-gemmini-utils ggml/src/ggml-backend.cpp ggml/CMakeLists.txt tools/perplexity tools/eval/evaluation-metal.hpp |
        rtk proxy xargs -0 rtk proxy shasum -a 256 > "$result/source.sha256"
    rtk proxy shasum -a 256 "$model" "$dataset" "$root/scripts/experiment/run-metal-quality-ppl.sh" "$root/scripts/experiment/ppl-original-models.sh" "$root/scripts/experiment/default-ppl-models.sha256" "$build/CMakeCache.txt" "$build/bin/llama-perplexity" "$build/bin/"*.dylib "$build/bin/"*.so > "$result/inputs-binaries.sha256"
    printf 'Running %s (log: %s/ppl.log)\n' "$id" "$result"
    status=0
    rtk proxy /usr/bin/time -p "${command[@]}" 2>&1 | rtk proxy tee "$result/ppl.log" || status=$?
    printf '%s\n' "$status" > "$result/exit-status.txt"
    (( status == 0 )) || { printf 'Failed %s: exit %s\n' "$id" "$status" >&2; exit "$status"; }
    rtk proxy python3 - "$result" "$method" "$family" "$bits" "$dim" "$context" "$chunks" "$q6_head" "$activation_fp16" <<'PY' > "$result/result.tsv"
import json, math, pathlib, re, sys
directory, method, model, bits, dim, context, requested, q6_head, activation_fp16 = sys.argv[1:]
bits, dim, context, requested = map(int, (bits, dim, context, requested))
text = (pathlib.Path(directory) / 'ppl.log').read_text()
def require(condition, message):
    if not condition: raise SystemExit('Rejected PPL result: ' + message)
totals = re.findall(r'^PPL_TOTALS (.*)$', text, re.M)
require(len(totals) == 1, 'expected one final PPL_TOTALS')
values = dict(item.split('=', 1) for item in totals[0].split())
n, tokens, ppl = int(values['chunks']), int(values['total_scored_tokens']), float(values['corpus_ppl'])
planned = re.findall(r'calculating perplexity over (\d+) chunks', text)
require(len(planned) == 1 and n == int(planned[0]) and n > 0, 'incomplete chunks')
require(requested == -1 or n == requested, 'requested chunk count not completed')
chunk_records = [dict(item.split('=', 1) for item in row.split()) for row in re.findall(r'\bPPL_CHUNK ([^\r\n]*)', text)]
require([int(row['index']) for row in chunk_records] == list(range(n)), 'chunk index coverage')
require(all(int(row['scored_tokens']) == context - context // 2 - 1 and math.isfinite(float(row['nll'])) for row in chunk_records) and sum(int(row['scored_tokens']) for row in chunk_records) == tokens, 'chunk/token coverage')
require(math.isfinite(ppl) and ppl > 0 and math.isfinite(float(values['total_nll'])), 'nonfinite final totals')
records = re.findall(r'^METAL_CPU_EXACT_PROOF (.*)$', text, re.M)
require(len(records) == 1, 'expected one CPU-exact Metal execution proof')
p = json.loads(records[0])
mode, compute = ('EXSIA' if method in ('PoTal', 'RTN-WA2') else 'BLOCK'), ('FLOAT' if method == 'RTN-W' else 'INT')
integers = ('version', 'bits', 'dim', 'scored_tokens', 'observed_matmuls', 'verified_matmuls', 'float_gpu_calls', 'integer_gpu_launches', 'residual_gpu_launches', 'attention_gpu_calls', 'observed_attention_matmuls', 'verified_attention_matmuls')
require(all(type(p.get(k)) is int and p[k] >= 0 for k in integers), 'invalid proof counters')
require(p.get('schema') == 'metal-cpu-exact-ppl' and p['version'] == 3 and p.get('graph') == 'cpu_equivalent_metal', 'proof schema/graph')
require(p.get('outlier_selection') == (method == 'PoTal') and p.get('residual_enabled') == (method == 'PoTal'), 'ablation switches')
require(p.get('complete') is True and p['scored_tokens'] == tokens, 'incomplete proof/token count')
require(p['bits'] == bits and p['dim'] == dim and p.get('activation') == mode and p.get('compute') == compute, 'proof profile')
require(p['observed_matmuls'] > 0 and p['observed_matmuls'] == p['verified_matmuls'], 'CPU fallback/missing GPU matmul')
require(p['observed_attention_matmuls'] == p['verified_attention_matmuls'] == p['attention_gpu_calls'] == 2 * (12 if model == 'gpt2' else 16) * n, 'missing GPU attention matmul')
require(p['residual_gpu_launches'] > 0 if method == 'PoTal' else p['residual_gpu_launches'] == 0, 'residual GPU route')
if q6_head == '1':
    require(p.get('gpu_q6_head_matmuls') == n and p.get('cpu_q6_head_matmuls') == 0, 'missing Q6_K GPU head')
    require(p.get('q6_head_activation_bits') == (16 if compute == 'FLOAT' else bits), 'Q6_K head activation precision')
else:
    require(p.get('cpu_q6_head_matmuls', 0) == 0, 'unexpected Q6_K CPU head')
require(p.get('activation_fp16', False) == (activation_fp16 == '1'), 'activation FP16 contract')
require((p['float_gpu_calls'] > 0 and p['integer_gpu_launches'] == 0) if compute == 'FLOAT' else (p['integer_gpu_launches'] > 0 and p['float_gpu_calls'] == 0), 'GPU kernel family')
(pathlib.Path(directory) / 'metal-cpu-exact-proof.json').write_text(json.dumps(p, indent=2) + '\n')
elapsed = re.findall(r'^real\s+([0-9.]+)$', text, re.M)
require(len(elapsed) == 1 and math.isfinite(float(elapsed[0])), 'missing process time')
print('\t'.join(map(str, (method, model, bits, dim, n, tokens, values['corpus_ppl'], elapsed[0], 0))))
PY
    rtk proxy shasum -a 256 -c "$result/source.sha256" "$result/inputs-binaries.sha256" "$result/cpu-arithmetic.sha256" > "$result/provenance-check.log"
    rtk proxy cat "$result/result.tsv" >> "$output/results.tsv"
    rtk proxy python3 "$root/scripts/experiment/summarize-metal-quality-ppl.py" "$output"
done
printf 'Completed: %s/results.tsv\n' "$output"
