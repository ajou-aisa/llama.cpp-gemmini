#!/usr/bin/env bash
# Build only: GPU execution is a separate, explicitly scheduled operation.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mode=${2:-fp16}
out=${1:-"$root/build-potal-cuda/nvfp4-$mode"}
extra=()
case "$mode" in
    fp16) ;;
    fp32)
        extra=(-DNVFP4_DEQUANT_FP32=1) ;;
    *) printf 'Expected fp16 or fp32\n' >&2; exit 2 ;;
esac
cuda=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$out"
source_dir="$root/ggml/src/ggml-cuda/nvfp4-reference"
c++ -std=c++17 -O2 -fno-fast-math -ffp-contract=off -shared -fPIC \
    "$source_dir/nvfp4-reference.cpp" -o "$out/libnvfp4-reference.so"
"${CUDACXX:-$cuda/bin/nvcc}" -std=c++17 -O2 -shared -Xcompiler -fPIC \
    --ftz=false --fmad=false --prec-div=true "-arch=${NVFP4_CUDA_ARCH:-native}" --cudart shared \
    -I"$root/ggml/include" -I"$root/ggml/src/ggml-cuda/cpu-exact" \
    "${extra[@]}" "$source_dir/ggml-nvfp4-preload.cu" -L"$cuda/lib64" -lcublas -ldl \
    -Xlinker -rpath -Xlinker "$cuda/lib64" -o "$out/libggml-nvfp4-preload.so"
python3 - "$root" "$out" "$mode" "${NVFP4_CUDA_ARCH:-native}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
root, out = map(pathlib.Path, sys.argv[1:3])
paths = list((root / 'ggml/src/ggml-cuda/nvfp4-reference').glob('*'))
paths.append(root / 'scripts/experiment/build-nvfp4-reference.sh')
paths += [root / 'tools/perplexity/CMakeLists.txt']
paths += list(out.glob('*.so'))
manifest = dict(schema='nvfp4-reference-build-v1', gpu_arch=sys.argv[4], native_fp4=False,
                reconstruction=sys.argv[3],
                files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()})
(out / 'build-identity.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(out / 'build-identity.json')
PY
