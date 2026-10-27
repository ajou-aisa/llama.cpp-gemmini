#!/usr/bin/env bash
# Build only: GPU execution is a separate, explicitly scheduled operation.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
out=${1:-/home/mojoi/build/potal-nvfp4}
cuda=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$out"
source_dir="$root/ggml/src/ggml-cuda/nvfp4-reference"
c++ -std=c++17 -O2 -fno-fast-math -ffp-contract=off -shared -fPIC \
    "$source_dir/nvfp4-reference.cpp" -o "$out/libnvfp4-reference.so"
"${CUDACXX:-$cuda/bin/nvcc}" -std=c++17 -O2 -shared -Xcompiler -fPIC \
    --ftz=false --fmad=false --prec-div=true -arch=sm_86 --cudart shared \
    -I"$root/ggml/include" -I"$root/ggml/src/ggml-cuda/cpu-exact" \
    "$source_dir/ggml-nvfp4-preload.cu" -L"$cuda/lib64" -lcublas -ldl \
    -Xlinker -rpath -Xlinker "$cuda/lib64" -o "$out/libggml-nvfp4-preload.so"
python3 - "$root" "$out" <<'PY'
import hashlib, json, pathlib, subprocess, sys
root, out = map(pathlib.Path, sys.argv[1:])
paths = list((root / 'ggml/src/ggml-cuda/nvfp4-reference').glob('*'))
paths += [root / 'scripts/experiment/build-nvfp4-reference.sh', root / 'scripts/experiment/test-nvfp4-reference.py']
paths += [root / 'tools/perplexity/CMakeLists.txt']
paths += list(out.glob('*.so'))
manifest = dict(schema='nvfp4-reference-build-v1', gpu_arch='sm_86', native_fp4=False,
                files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()})
(out / 'build-identity.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(out / 'build-identity.json')
PY
