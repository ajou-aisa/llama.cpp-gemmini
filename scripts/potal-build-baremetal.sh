#!/bin/bash
# Build the PoTal API smoke test as a bare-metal ELF for Chipyard simulation.
#
# Usage: scripts/potal-build-baremetal.sh <configured-potal-build-dir> [output-elf]
#
# The build dir must come from a POTAL_API_ONLY=ON configure (host or RISC-V);
# its generated/potal_hw_params.h is the only hardware-parameter input, so the
# header validation stays in cmake/potal-hw-params.cmake.
#
# Test-environment dependency (not an API dependency): riscv-tests benchmark
# startup (crt.S, syscalls.c, test.ld), by default from
#   generators/potal/software/gemmini-rocc-tests/riscv-tests
# Override with POTAL_RISCV_TESTS=<riscv-tests root>, CC_BAREMETAL=<gcc>.
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 || $# -gt 2 || "$1" == -h || "$1" == --help ]]; then
  sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
fi

build_dir="$(cd "$1" && pwd)"
generated="$build_dir/generated"
out="${2:-$build_dir/baremetal/potal-api-smoke-baremetal}"
riscv_tests="${POTAL_RISCV_TESTS:-$SCRIPT_ROOT/../../gemmini-rocc-tests/riscv-tests}"
cc="${CC_BAREMETAL:-riscv64-unknown-elf-gcc}"
common="$riscv_tests/benchmarks/common"
api="$SCRIPT_ROOT/ggml/src/ggml-potal"

for f in "$generated/potal_hw_params.h" "$generated/original-potal-params.h" \
         "$generated/potal_build_manifest.txt"; do
  if [[ ! -f "$f" ]]; then
    printf 'Missing %s; configure %s with POTAL_API_ONLY=ON first\n' "$f" "$build_dir" >&2
    exit 2
  fi
done
for f in "$common/crt.S" "$common/syscalls.c" "$common/test.ld" "$riscv_tests/env/encoding.h"; do
  if [[ ! -f "$f" ]]; then
    printf 'Missing bare-metal startup file %s (set POTAL_RISCV_TESTS)\n' "$f" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "$out")"
# Flags follow gemmini-rocc-tests/bareMetalC/Makefile (CFLAGS_BAREMETAL).
"$cc" \
  -std=gnu11 -O2 -mcmodel=medany -march=rv64gc -Wa,-march=rv64gc \
  -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
  -DPREALLOCATE=1 -DMULTITHREAD=1 -DBAREMETAL=1 \
  -nostdlib -nostartfiles -static -T "$common/test.ld" \
  -I"$api/include" -I"$generated" \
  -I"$riscv_tests" -I"$riscv_tests/env" -I"$common" \
  "$api/tests/potal-api-smoke.c" "$api/potal.c" \
  "$common/syscalls.c" "$common/crt.S" \
  -lm -lgcc \
  -o "$out"

cp "$generated/potal_build_manifest.txt" "$out.manifest.txt"
printf 'Built %s\n' "$out"
printf 'Manifest %s\n' "$out.manifest.txt"
