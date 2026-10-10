#!/usr/bin/env bash
set -euo pipefail

if ! command -v rtk >/dev/null; then
    rtk() { [[ $1 != proxy ]] || shift; "$@"; }
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
manifest="$root/scripts/experiment/cuda-quality-cases.tsv"
action=list; selected=all; output="$root/output/experiment/cuda-quality-ppl"
while (( $# )); do
    case "$1" in
        --list) action=list; shift ;;
        --prepare) action=prepare; shift ;;
        --run) action=run; shift ;;
        --background) action=background; shift ;;
        --case) selected=${2:?Missing case}; shift 2 ;;
        --output) output=${2:?Missing output directory}; shift 2 ;;
        --help) printf 'Usage: bash %s [--list|--prepare|--run|--background] [--case ID] [--output DIR]\n' "$0"; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
fail() { printf 'CUDA queue: %s\n' "$*" >&2; exit 1; }
[[ $output == /* ]] || output="$PWD/$output"
rows=$(rtk proxy awk -F '\t' -v id="$selected" 'NR>1 && (id=="all" || $1==id)' "$manifest")
[[ -n $rows ]] || fail "Unknown case $selected"
if [[ $action == list ]]; then
    printf 'case\tmodel\tmethod\tbits\tdim\n%s\n' "$rows"
    exit 0
fi
[[ $(rtk proxy uname -s) == Linux ]] || fail 'Run preparation and PPL on the CUDA Linux host.'
for tool in cmake nvcc python3 shasum flock; do command -v "$tool" >/dev/null || fail "Missing command: $tool"; done
rtk proxy python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "Python 3.10 or newer is required")'
rtk proxy mkdir -p "$output"
if [[ $action == background ]]; then
    rtk proxy nohup bash "$0" --run --case "$selected" --output "$output" < /dev/null >> "$output/queue.log" 2>&1 &
    printf 'Queue PID=%s log=%s/queue.log\n' "$!" "$output"
    exit 0
fi
exec 9>"$output/.lock"
rtk proxy flock -n 9 || fail "A queue already owns $output"
build_root=${CUDA_PPL_BUILD_ROOT:-$root/build-cuda-quality}
rtk proxy mkdir -p "$build_root"
rtk proxy nvcc "$root/scripts/experiment/cuda-device-info.cu" -o "$build_root/cuda-device-info"
arch=$(rtk proxy "$build_root/cuda-device-info" --arch)
[[ $arch =~ ^[0-9]+$ ]] || fail 'Invalid device architecture'
rtk proxy "$build_root/cuda-device-info" > "$output/device.txt"
rtk proxy nvcc --version > "$output/nvcc.txt"
rtk proxy uname -a > "$output/host.txt"
source "$root/scripts/experiment/ppl-original-models.sh"
dataset="$root/wikitext-2-raw/wiki.test.raw"
[[ -s $dataset ]] || fail "Missing dataset $dataset"
cd "$root"
if [[ -f $output/source.sha256 ]]; then
    rtk proxy shasum -a 256 -c "$output/source.sha256" > "$output/source-check.log"
else
    rtk proxy git ls-files -z --cached --others --exclude-standard CMakeLists.txt cmake ggml src common tools/perplexity scripts/experiment tests |
        rtk proxy xargs -0 shasum -a 256 > "$output/source.sha256"
    rtk proxy git rev-parse HEAD > "$output/source-commit.txt"
    rtk proxy git diff HEAD -- CMakeLists.txt cmake ggml src common tools/perplexity scripts/experiment tests > "$output/source.diff"
fi
prepared='|'
while IFS=$'\t' read -r id family method bits dim; do
    selection=OFF; residual=OFF; stripe=ON; activation=EXSIA; mode=STRIPE_PIPELINE; quant="Q${bits}_HP1"; head=0
    case "$method" in
        RTN-WA1) stripe=OFF; activation=BLOCK; mode=FULL; quant="Q${bits}_0"; [[ $bits != 4 ]] || head=1 ;;
        RTN-WA2) ;;
        PoTal) selection=ON; residual=ON ;;
        *) fail "Invalid method $method" ;;
    esac
    ppl_original_model "$root" "$family" "$quant"
    model=$original_model
    profile="$method-a$bits-d$dim"; build="$build_root/$profile"; result="$output/$id"
    if [[ -s $result/result.tsv ]]; then
        rtk proxy shasum -a 256 -c "$result/inputs.sha256" > "$result/resume-check.log"
        rtk proxy python3 "$root/scripts/experiment/verify-cuda-quality-ppl.py" "$result" > /dev/null
        printf 'Completed, skipping: %s\n' "$id"
        continue
    fi
    if [[ $prepared != *"|$profile|"* ]]; then
        prepared="$prepared$profile|"
        rtk proxy mkdir -p "$build"
        printf 'Preparing %s (SM%s)\n' "$profile" "$arch"
        rtk proxy cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            "-DCMAKE_CUDA_ARCHITECTURES=$arch" -DGGML_NATIVE=ON -DGGML_GEMMINI=ON \
            -DGGML_GEMMINI_CUDA_CPU_EXACT=ON -DGGML_GEMMINI_METAL_CPU_EXACT=OFF \
            -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_METAL_QUANTIZED=OFF -DGGML_BACKEND_DL=OFF \
            -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF -DGGML_GEMMINI_ENABLE_OPENMP=ON \
            -DGGML_GEMMINI_OPTION=CPU -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE \
            -DGGML_GEMMINI_COMPUTE_TYPE=INT -DGGML_GEMMINI_DEQUANT_FP_TEST=OFF \
            "-DGGML_GEMMINI_ACTIVATION_QUANT=$activation" "-DGGML_GEMMINI_ACTIVATION_BITS=$bits" \
            "-DGGML_GEMMINI_WEIGHT_BITS=$bits" "-DGGML_GEMMINI_DIM=$dim" -DGGML_GEMMINI_BLOCK_SIZE=32 \
            "-DGGML_GEMMINI_ENABLE_RMD=$residual" "-DGGML_GEMMINI_EXSIA_OUTLIER_SELECTION=$selection" \
            -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU "-DGGML_GEMMINI_ENABLE_STRIPE_MATMUL=$stripe" \
            "-DGGML_GEMMINI_ENABLE_STRIPE_PIPELINE=$stripe" "-DGGML_GEMMINI_DEFAULT_MATMUL_MODE=$mode" \
            -DGGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY=2 -DGGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE=OFF \
            -DGGML_GEMMINI_EXSIA_SIGMA=2 -DGGML_GEMMINI_EXSIA_DEFAULT_MODE=LOCAL_FOLDING_PIPELINE \
            -DGGML_GEMMINI_EXSIA_LOCAL_WORKERS=3 -DGGML_GEMMINI_EXSIA_PROFILE_SCOPE=OFF \
            -DLOG_DEBUG=0 -DLOG_CYCLE=0 -DCYCLE_DETAIL=0 -DCYCLE_SIM=0 -DGGML_CPU_CYCLE_LOG=0 \
            -DLOG_DUMP=0 -DLOG_DUMP_SCALE=0 -DGGML_GEMMINI_ACT_QUANT_METRICS=0 \
            -DGGML_GEMMINI_RESIDUAL_METRICS=0 -DGGML_GEMMINI_PRINT_TILE=0 -DLLAMA_CURL=OFF \
            -DLLAMA_BUILD_TESTS=ON > "$build/configure.log" 2>&1
        rtk proxy cmake --build "$build" --target llama-perplexity test-cuda-cpu-exact-int test-cuda-cpu-exact-graph \
            -j "${BUILD_JOBS:-4}" > "$build/build.log" 2>&1
        rtk proxy ctest --test-dir "$build" --output-on-failure -R '^test-cuda-cpu-exact-(int|graph)$' > "$build/verification.log" 2>&1
        rtk proxy "$build/bin/test-cuda-cpu-exact-int" --benchmark > "$build/benchmark.log" 2>&1
        if command -v compute-sanitizer >/dev/null; then
            rtk proxy compute-sanitizer --tool memcheck --error-exitcode 1 "$build/bin/test-cuda-cpu-exact-int" > "$build/memcheck.log" 2>&1
        else
            fail 'compute-sanitizer is required before publishing CUDA PPL results.'
        fi
    fi
    [[ $action != prepare ]] || continue
    ppl_verified_models=''
    ppl_original_model "$root" "$family" "$quant"
    if [[ -e $result ]]; then rtk proxy mv "$result" "$result.interrupted-$(rtk proxy date +%Y%m%d-%H%M%S)"; fi
    rtk proxy mkdir -p "$result"
    rtk proxy cp "$output/device.txt" "$output/nvcc.txt" "$output/host.txt" "$result/"
    printf 'case\tmodel\tmethod\tbits\tdim\n%s\t%s\t%s\t%s\t%s\n' "$id" "$family" "$method" "$bits" "$dim" > "$result/case.tsv"
    rtk proxy cp "$build/CMakeCache.txt" "$build/compile_commands.json" "$build/verification.log" "$build/benchmark.log" "$build/memcheck.log" "$build/ggml/src/ggml-gemmini/cpu-exact-build.txt" "$result/"
    rtk proxy shasum -a 256 "$model" "$dataset" "$build/CMakeCache.txt" "$build/bin/llama-perplexity" "$build/bin/"*.so > "$result/inputs.sha256"
    sibling=$(rtk proxy awk -F= '$1=="gemmini_sw_path" {print substr($0,index($0,"=")+1)}' "$result/cpu-exact-build.txt")
    rtk proxy shasum -a 256 "$sibling/gemmini.h" "$sibling/gemmini_params.h" >> "$result/inputs.sha256"
    printf 'Checking CPU/CUDA model equivalence: %s\n' "$id"
    for gpu in 0 1; do
        rtk proxy /usr/bin/time -p env OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 \
            "GGML_GEMMINI_CUDA_CPU_EXACT=$gpu" "GGML_GEMMINI_CUDA_CPU_EXACT_Q6_HEAD=$head" \
            "$build/bin/llama-perplexity" --model "$model" --file "$dataset" \
            --ctx-size 512 --batch-size 512 --ubatch-size 512 --chunks 1 \
            --threads 4 --threads-batch 4 --gpu-layers 0 --cache-type-k f16 --cache-type-v f16 --seed 42 --no-warmup \
            > "$result/smoke-$gpu.log" 2>&1
    done
    rtk proxy python3 "$root/scripts/experiment/verify-cuda-quality-ppl.py" --smoke \
        "$result/smoke-0.log" "$result/smoke-1.log" > "$result/smoke-verification.txt"
    command=(env OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 \
        GGML_GEMMINI_CUDA_CPU_EXACT=1 "GGML_GEMMINI_CUDA_CPU_EXACT_Q6_HEAD=$head" \
        "$build/bin/llama-perplexity" --model "$model" --file "$dataset" \
        --ctx-size 512 --batch-size 512 --ubatch-size 512 --chunks -1 \
        --threads 4 --threads-batch 4 --gpu-layers 0 --cache-type-k f16 --cache-type-v f16 --seed 42 --no-warmup)
    printf '%q ' "${command[@]}" > "$result/command.txt"; printf '\n' >> "$result/command.txt"
    printf 'Running %s log=%s/ppl.log\n' "$id" "$result"
    status=0
    rtk proxy /usr/bin/time -p "${command[@]}" > "$result/ppl.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$result/exit-status.txt"
    (( status == 0 )) || fail "$id exited with $status; see $result/ppl.log"
    rtk proxy shasum -a 256 -c "$output/source.sha256" "$result/inputs.sha256" > "$result/provenance-check.log"
    rtk proxy python3 "$root/scripts/experiment/verify-cuda-quality-ppl.py" "$result" > "$result/result.tsv.pending"
    rtk proxy mv "$result/result.tsv.pending" "$result/result.tsv"
    rtk proxy cat "$result/result.tsv"
    printf 'case\tmethod\tmodel\tbits\tdim\tchunks\ttokens\tppl\tseconds\n' > "$output/results.tsv.pending"
    for completed in "$output/"*/result.tsv; do rtk proxy cat "$completed" >> "$output/results.tsv.pending"; done
    rtk proxy mv "$output/results.tsv.pending" "$output/results.tsv"
done <<< "$rows"
printf 'CUDA queue complete: %s\n' "$output"
