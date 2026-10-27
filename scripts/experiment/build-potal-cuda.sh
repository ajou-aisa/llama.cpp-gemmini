#!/usr/bin/env bash
# Build the existing CUDA PPL paths; no model conversion or inference.
set -euo pipefail
bits=8 anchor=2 jobs=1 rtn=OFF build_dir=
while (( $# )); do
    case "$1" in
        --bits) bits=${2:?Missing bits}; shift 2 ;;
        --anchor) anchor=${2:?Missing anchor}; shift 2 ;;
        --jobs) jobs=${2:?Missing jobs}; shift 2 ;;
        --build-dir) build_dir=${2:?Missing build directory}; shift 2 ;;
        --rtn) rtn=ON; shift ;;
        --help) printf 'Usage: %s [--bits 4|8] [--anchor 2|3] [--rtn] [--jobs N] [--build-dir DIR]\n' "$0"; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ $bits == 4 || $bits == 8 ]] || { printf 'Expected bits 4 or 8\n' >&2; exit 2; }
[[ $anchor == 2 || $anchor == 3 ]] || { printf 'Expected anchor 2 or 3\n' >&2; exit 2; }
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { printf 'Expected positive build jobs\n' >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
profile="top${anchor}-a${bits}-rmdon"
exact=ON attention=ON rmd=ON quant=EXSIA mode=STRIPE_PIPELINE
targets=(llama-perplexity test-cuda-cpu-exact-graph test-cuda-hp1-scu test-cuda-rtn)
if [[ $rtn == ON ]]; then
    profile=rtn
    exact=OFF attention=OFF rmd=OFF quant=BLOCK mode=FULL
    targets=(llama-perplexity test-cuda-rtn)
fi
build_dir=${build_dir:-"$root/build-potal-cuda/$profile"}
mkdir -p "$build_dir"
build_dir=$(cd -- "$build_dir" && pwd)
exec 9>"$build_dir/.build.lock"
flock -n 9 || { printf 'Build already active: %s\n' "$build_dir" >&2; exit 1; }
extra=()
if [[ -n ${GEMMINI_SW_PATH:-} ]]; then extra+=("-DGEMMINI_SW_PATH=$GEMMINI_SW_PATH"); fi
printf 'Configuring %s; logs: %s\n' "$profile" "$build_dir"
cmake -S "$root" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    "-DCMAKE_CUDA_COMPILER=${CUDACXX:-/usr/local/cuda/bin/nvcc}" \
    "-DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS:-87}" "${extra[@]}" \
    -DGGML_NATIVE=ON -DGGML_GEMMINI=ON "-DGGML_GEMMINI_CUDA_CPU_EXACT=$exact" \
    "-DGGML_GEMMINI_CUDA_RTN=$rtn" "-DGGML_GEMMINI_CUDA_RTN_ATTENTION=$attention" \
    -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BACKEND_DL=OFF -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF \
    -DGGML_OPENMP=OFF -DGGML_GEMMINI_ENABLE_OPENMP=ON \
    -DGGML_GEMMINI_OPTION=CPU -DGGML_GEMMINI_EXECUTION_BACKEND=HARDWARE \
    -DGGML_GEMMINI_COMPUTE_TYPE=INT -DGGML_GEMMINI_DEQUANT_FP_TEST=OFF \
    "-DGGML_GEMMINI_ACTIVATION_QUANT=$quant" "-DGGML_GEMMINI_ACTIVATION_BITS=$bits" \
    "-DGGML_GEMMINI_WEIGHT_BITS=$bits" -DGGML_GEMMINI_DIM=32 -DGGML_GEMMINI_BLOCK_SIZE=32 \
    "-DGGML_GEMMINI_ENABLE_RMD=$rmd" -DGGML_GEMMINI_DEFAULT_RMD_BACKEND=CPU \
    -DGGML_GEMMINI_ENABLE_STRIPE_MATMUL=ON -DGGML_GEMMINI_ENABLE_STRIPE_PIPELINE=ON \
    "-DGGML_GEMMINI_DEFAULT_MATMUL_MODE=$mode" -DGGML_GEMMINI_DEFAULT_STRIPE_JOB_CAPACITY=2 \
    -DGGML_GEMMINI_ALLOW_RUNTIME_MATMUL_OVERRIDE=OFF -DGGML_GEMMINI_EXSIA_SIGMA=2 \
    "-DGGML_GEMMINI_EXSIA_STRIPE_ANCHOR=$anchor" -DGGML_GEMMINI_EXSIA_DEFAULT_MODE=LOCAL_FOLDING_PIPELINE \
    -DGGML_GEMMINI_EXSIA_LOCAL_WORKERS=3 -DGGML_GEMMINI_EXSIA_PROFILE_SCOPE=OFF \
    -DGGML_GEMMINI_ENABLE_A4NKS_ATTENTION=OFF -DGGML_GEMMINI_OFFLOAD_A4NKS_ATTENTION=OFF \
    -DLOG_DEBUG=0 -DLOG_CYCLE=0 -DCYCLE_DETAIL=0 -DCYCLE_SIM=0 -DGGML_CPU_CYCLE_LOG=0 \
    -DLOG_DUMP=0 -DLOG_DUMP_SCALE=0 -DGGML_GEMMINI_ACT_QUANT_METRICS=0 \
    -DGGML_GEMMINI_RESIDUAL_METRICS=0 -DGGML_GEMMINI_PRINT_TILE=0 \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON > "$build_dir/configure.log" 2>&1
printf 'Building %s\n' "$profile"
cmake --build "$build_dir" --target "${targets[@]}" -j "$jobs" > "$build_dir/build.log" 2>&1
printf 'Checking CUDA numerical contracts\n'
OMP_NUM_THREADS=4 ctest --test-dir "$build_dir" --output-on-failure \
    -R '^test-cuda-(cpu-exact-graph|hp1-scu(-graph)?|rtn)$' > "$build_dir/verification.log" 2>&1
python3 "$root/tests/test-perplexity-deterministic.py" > "$build_dir/nll-verification.log" 2>&1
printf 'PASS: %s/bin/llama-perplexity\n' "$build_dir"
