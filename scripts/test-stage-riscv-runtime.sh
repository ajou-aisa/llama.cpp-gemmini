#!/bin/bash
# Uses real cross-compiled ELF fixtures; runs no target code.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
compiler=${1:-riscv64-unknown-linux-gnu-gcc}
compiler=$(command -v "$compiler")
readelf=$("$compiler" -print-prog-name=readelf)
runtime=$("$compiler" -print-file-name=libgomp.so.1)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
build="$work/build with spaces"
mkdir -p "$build/bin" "$build/CMakeFiles/0.0.0" "$build/CMakeFiles/stale" "$work/shims"
printf 'CMAKE_READELF:FILEPATH=%s\n' "$readelf" > "$build/CMakeCache.txt"
printf 'CMAKE_CACHE_%s_VERSION:INTERNAL=0\n' MAJOR MINOR PATCH >> "$build/CMakeCache.txt"
printf 'set(CMAKE_C_COMPILER "%s")\n' "$compiler" > "$build/CMakeFiles/0.0.0/CMakeCCompiler.cmake"
printf 'set(CMAKE_C_COMPILER "/stale/compiler")\n' > "$build/CMakeFiles/stale/CMakeCCompiler.cmake"
cat > "$work/consumer.c" <<'C'
#include <omp.h>
int worker_count(void) { return omp_get_num_threads(); }
C
"$compiler" -shared -fPIC -fopenmp -Wl,-rpath,'$ORIGIN' "$work/consumer.c" -o "$work/consumer.so"
stage() {
  bash -euo pipefail -c 'source "$1/scripts/stage-riscv-runtime.sh"; stage_riscv_openmp_runtime "$2"' _ "$root" "$build"
}
pass() { printf 'PASS: %s\n' "$1"; }
fail_stage() {
  if stage > "$work/log" 2>&1; then
    printf 'FAIL: expected failure: %s\n' "$1" >&2
    exit 1
  fi
  grep -q "$2" "$work/log"
  pass "$1"
}
printf 'not ELF\n' > "$build/bin/notes"
printf 'int main(void) { return 0; }\n' > "$work/static.c"
"$compiler" -static "$work/static.c" -o "$build/bin/static-test"
stage > "$work/log"
grep -q 'staging SKIPPED' "$work/log"
[[ ! -e "$build/bin/libgomp.so.1" ]]
pass 'static executable / non-ELF skipped'
# Exercise a shared library reached through a symlink, with no executable present.
ln -s "$work/consumer.so" "$build/bin/consumer.so"
stage > "$work/log"
cmp "$runtime" "$build/bin/libgomp.so.1"
[[ ! -L "$build/bin/libgomp.so.1" && -r "$build/bin/libgomp.so.1" ]]
pass 'shared ELF symlink detected; target runtime copied'
stage > "$work/log"
cmp "$runtime" "$build/bin/libgomp.so.1"
pass 'repeat staging'
# A stale host library must be replaced, not trusted or followed.
host_runtime=$(gcc -print-file-name=libgomp.so.1)
[[ -f "$host_runtime" ]]
cp -L "$host_runtime" "$build/bin/libgomp.so.1"
stage > "$work/log"
cmp "$runtime" "$build/bin/libgomp.so.1"
pass 'existing wrong ABI replaced'
RISCV_LIBGOMP="$host_runtime" fail_stage 'host runtime rejected' 'incompatible'
cp -L "$runtime" "$work/wrong-abi.so"
# e_flags is at offset 48 in little-endian ELF64: RVC with soft-float ABI.
printf '\001\000\000\000' | dd of="$work/wrong-abi.so" bs=1 seek=48 conv=notrunc status=none
RISCV_LIBGOMP="$work/wrong-abi.so" fail_stage 'RISC-V float ABI mismatch rejected' 'incompatible'
ln -s "$work/missing" "$work/broken"
RISCV_LIBGOMP="$work/broken" fail_stage 'broken symlink rejected' 'incompatible'
RISCV_LIBGOMP="$work/missing" fail_stage 'missing override rejected' 'not found'
# Force compiler discovery to report no runtime, without relying on host layout.
cat > "$work/no-runtime-cc" <<'SH'
#!/bin/bash
case "$1" in
  -print-file-name=*) printf '%s\n' "${1#*=}" ;;
  -print-sysroot) printf '\n' ;;
  -print-search-dirs) printf 'libraries: =\n' ;;
  *) exit 1 ;;
esac
SH
chmod +x "$work/no-runtime-cc"
printf 'CMAKE_C_COMPILER:FILEPATH=%s\nCMAKE_READELF:FILEPATH=%s\n' "$work/no-runtime-cc" "$readelf" > "$build/CMakeCache.txt"
fail_stage 'target runtime absent' 'not found'
printf 'CMAKE_C_COMPILER:FILEPATH=%s\nCMAKE_READELF:FILEPATH=%s\n' "$compiler" "$work/missing-readelf" > "$build/CMakeCache.txt"
fail_stage 'readelf unavailable' 'readelf unavailable'
printf 'CMAKE_C_COMPILER:FILEPATH=%s\nCMAKE_READELF:FILEPATH=%s\n' "$compiler" "$readelf" > "$build/CMakeCache.txt"
rm "$build/bin/libgomp.so.1"
mkdir "$build/bin/libgomp.so.1"
fail_stage 'destination cannot be replaced' 'staging rename failed'
rmdir "$build/bin/libgomp.so.1"
printf '#!/bin/bash\nexit 1\n' > "$work/shims/cp"
chmod +x "$work/shims/cp"
PATH="$work/shims:$PATH" fail_stage 'copy failure' 'copy failed'
[[ ! -e "$build/bin/libgomp.so.1" ]]
[[ -z "$(find "$build/bin" -name '.libgomp.so.1.*' -print)" ]]
pass 'failed staging leaves no temporary file'
printf '\177ELFbroken' > "$build/bin/broken-elf"
fail_stage 'malformed ELF scan fails' 'Error'
rm "$build/bin/broken-elf"
BUILD_DIR="$work/dry-run-build" bash "$root/build-riscv.sh" --dry-run > "$work/log" 2>&1
[[ ! -e "$work/dry-run-build" ]]
BUILD_DIR="$work/help-build" bash "$root/build-riscv.sh" --help > "$work/log"
[[ ! -e "$work/help-build" ]]
pass 'dry-run and help leave no build directory'
