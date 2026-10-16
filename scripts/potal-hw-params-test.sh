#!/bin/bash
# Configure-level tests for the PoTal hardware parameter contract.
#
# Runs real host CMake configures of this repository with POTAL_API_ONLY=ON.
# Positive cases must configure; negative cases MUST FAIL to configure with the
# expected message (a negative case passes only when configure fails).
#
# Usage: scripts/potal-hw-params-test.sh
# Env:   POTAL_PARAMS_DIR  directory holding gemmini_potal{,_d32,_d64}_params.h
#                          (default ../../gemmini-rocc-tests/include)
#        POTAL_TEST_ROOT   scratch build root (default build-potal-hwparams-test)
# Output: PASS/FAIL per case, logs in $POTAL_TEST_ROOT/<case>.log,
#         final "POTAL_HW_PARAMS_TEST PASS|FAIL"; exit status 0/1.
set -uo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARAMS_DIR="${POTAL_PARAMS_DIR:-$SCRIPT_ROOT/../../gemmini-rocc-tests/include}"
PARAMS_DIR="$(cd "$PARAMS_DIR" && pwd)" || exit 2
ROOT="${POTAL_TEST_ROOT:-$SCRIPT_ROOT/build-potal-hwparams-test}"
D16="$PARAMS_DIR/gemmini_potal_params.h"
D32="$PARAMS_DIR/gemmini_potal_d32_params.h"
D64="$PARAMS_DIR/gemmini_potal_d64_params.h"

for f in "$D16" "$D32" "$D64"; do
  [[ -f "$f" ]] || { printf 'Missing %s (set POTAL_PARAMS_DIR)\n' "$f" >&2; exit 2; }
done

rm -rf "$ROOT"
mkdir -p "$ROOT/headers"
cd "$ROOT" || exit 2 # fixed cwd for relative-path cases
failures=0

configure() { # <case> <cmake args...>
  local name=$1
  shift
  cmake -S "$SCRIPT_ROOT" -B "$ROOT/$name" -DPOTAL_API_ONLY=ON -DGGML_GEMMINI=OFF "$@" \
    > "$ROOT/$name.log" 2>&1
}

pass() { printf 'PASS %s\n' "$1"; }
# CMake wraps long messages; match against the log with whitespace collapsed.
log_has() { tr -s '[:space:]' ' ' < "$1" | grep -Eq "$2"; }
fail() { printf 'FAIL %s: %s (log %s)\n' "$1" "$2" "$ROOT/$1.log"; failures=$((failures + 1)); }

generated_dim() { # <build dir> -> DIM from the generated adapter
  sed -n 's/^#define POTAL_HW_DIM[[:space:]]*\([0-9]*\)$/\1/p' "$1/generated/potal_hw_params.h"
}

manifest_sha() {
  sed -n 's/^potal_hw_params_sha256=//p' "$1/generated/potal_build_manifest.txt"
}

expect_ok() { # <case> <dim> <header>
  local name=$1 dim=$2 header=$3
  if ! configure "$name" "-DPOTAL_EXPECTED_DIM=$dim" "-DPOTAL_HW_PARAMS=$header"; then
    fail "$name" "configure failed"
  elif [[ "$(generated_dim "$ROOT/$name")" != "$dim" ]]; then
    fail "$name" "generated POTAL_HW_DIM != $dim"
  elif ! cmp -s "$ROOT/$name/generated/original-potal-params.h" "$header"; then
    fail "$name" "original-potal-params.h differs from source"
  else
    pass "$name"
  fi
}

expect_fail() { # <case> <regex> <cmake args...>  (configure must fail)
  local name=$1 pattern=$2
  shift 2
  if configure "$name" "$@"; then
    fail "$name" "configure unexpectedly succeeded"
  elif ! log_has "$ROOT/$name.log" "$pattern"; then
    fail "$name" "configure failed without message /$pattern/"
  else
    pass "$name (expected configure failure)"
  fi
}

mutant() { # <name> <sed script> -> path of mutated copy of the DIM16 header
  sed "$2" "$D16" > "$ROOT/headers/$1.h"
  printf '%s\n' "$ROOT/headers/$1.h"
}

# ---- Accepted headers ------------------------------------------------------
expect_ok d16 16 "$D16"
expect_ok d32 32 "$D32"
expect_ok d64 64 "$D64"

# ---- Rejected inputs -------------------------------------------------------
expect_fail missing-params 'POTAL_HW_PARAMS is required' -DPOTAL_EXPECTED_DIM=16
expect_fail missing-file 'not a readable file' -DPOTAL_EXPECTED_DIM=16 \
  "-DPOTAL_HW_PARAMS=$ROOT/headers/does-not-exist.h"
# Typed relative values stay relative and are rejected.
expect_fail relative-path-filepath 'must be an absolute path' -DPOTAL_EXPECTED_DIM=16 \
  -DPOTAL_HW_PARAMS:FILEPATH=gemmini_potal_params.h
expect_fail relative-path-string 'must be an absolute path' -DPOTAL_EXPECTED_DIM=16 \
  -DPOTAL_HW_PARAMS:STRING=gemmini_potal_params.h
# An untyped relative -D value is made absolute by CMake against the invocation
# directory (the test's cwd, where this name does not exist), then rejected.
expect_fail relative-path-untyped "not a readable file: $ROOT/gemmini_potal_params.h" \
  -DPOTAL_EXPECTED_DIM=16 -DPOTAL_HW_PARAMS=gemmini_potal_params.h
expect_fail directory 'not a readable file' -DPOTAL_EXPECTED_DIM=16 "-DPOTAL_HW_PARAMS=$PARAMS_DIR"
expect_fail missing-expected-dim 'POTAL_EXPECTED_DIM must be' "-DPOTAL_HW_PARAMS=$D16"
expect_fail bad-expected-dim 'POTAL_EXPECTED_DIM must be' -DPOTAL_EXPECTED_DIM=8 "-DPOTAL_HW_PARAMS=$D16"
expect_fail dim-mismatch 'DIM mismatch: header reports 16, POTAL_EXPECTED_DIM=32' \
  -DPOTAL_EXPECTED_DIM=32 "-DPOTAL_HW_PARAMS=$D16"
expect_fail not-generated-header 'GEMMINI_PARAMS_H guard missing' -DPOTAL_EXPECTED_DIM=16 \
  "-DPOTAL_HW_PARAMS=$(mutant no-guard '/GEMMINI_PARAMS_H/d')"
expect_fail xcustom 'XCUSTOM_ACC=0' -DPOTAL_EXPECTED_DIM=16 \
  "-DPOTAL_HW_PARAMS=$(mutant xcustom0 's/^#define XCUSTOM_ACC 3$/#define XCUSTOM_ACC 0/')"
expect_fail missing-bank-rows 'does not define required parameter BANK_ROWS' -DPOTAL_EXPECTED_DIM=16 \
  "-DPOTAL_HW_PARAMS=$(mutant no-bank-rows '/^#define BANK_ROWS /d')"
expect_fail missing-max-bytes 'does not define required parameter MAX_BYTES' -DPOTAL_EXPECTED_DIM=16 \
  "-DPOTAL_HW_PARAMS=$(mutant no-max-bytes '/^#define MAX_BYTES /d')"
expect_fail duplicate-dim 'defines DIM 2 times' -DPOTAL_EXPECTED_DIM=16 \
  "-DPOTAL_HW_PARAMS=$(mutant dup-dim 's/^#define DIM 16$/#define DIM 16\n#define DIM 32/')"
expect_fail non-numeric-acc-rows 'ACC_ROWS must be a plain decimal integer' -DPOTAL_EXPECTED_DIM=16 \
  "-DPOTAL_HW_PARAMS=$(mutant expr-acc 's/^#define ACC_ROWS 1024$/#define ACC_ROWS (512*2)/')"
expect_fail unsupported-dim 'supported DIM is 16, 32, or 64' -DPOTAL_EXPECTED_DIM=16 \
  "-DPOTAL_HW_PARAMS=$(mutant dim8 's/^#define DIM 16$/#define DIM 8/')"
expect_fail bank-rows-not-pow2 'BANK_ROWS=3000 must be a positive power of two' -DPOTAL_EXPECTED_DIM=16 \
  "-DPOTAL_HW_PARAMS=$(mutant bank3000 's/^#define BANK_ROWS 4096$/#define BANK_ROWS 3000/')"
expect_fail addr-len 'ADDR_LEN=64' -DPOTAL_EXPECTED_DIM=16 \
  "-DPOTAL_HW_PARAMS=$(mutant addr64 's/^#define ADDR_LEN 32$/#define ADDR_LEN 64/')"
expect_fail precision-elem 'unsupported precision' -DPOTAL_EXPECTED_DIM=16 \
  "-DPOTAL_HW_PARAMS=$(mutant a16 's/^typedef int8_t elem_t;$/typedef int16_t elem_t;/')"
expect_fail precision-acc 'unsupported precision' -DPOTAL_EXPECTED_DIM=16 \
  "-DPOTAL_HW_PARAMS=$(mutant acc64 's/^typedef int32_t acc_t;$/typedef int64_t acc_t;/')"
expect_fail with-gemmini 'POTAL_API_ONLY=ON requires GGML_GEMMINI=OFF' -DGGML_GEMMINI=ON \
  -DPOTAL_EXPECTED_DIM=16 "-DPOTAL_HW_PARAMS=$D16"
if cmake -S "$SCRIPT_ROOT" -B "$ROOT/params-without-api-only" -DPOTAL_EXPECTED_DIM=16 \
     "-DPOTAL_HW_PARAMS=$D16" > "$ROOT/params-without-api-only.log" 2>&1; then
  fail params-without-api-only "configure unexpectedly succeeded"
elif ! log_has "$ROOT/params-without-api-only.log" 'used only with POTAL_API_ONLY=ON'; then
  fail params-without-api-only "configure failed without expected message"
else
  pass "params-without-api-only (expected configure failure)"
fi

# ---- Isolation and reconfigure ---------------------------------------------
# Two build dirs with different headers keep separate generated parameters.
if [[ "$(generated_dim "$ROOT/d16")" == 16 && "$(generated_dim "$ROOT/d32")" == 32 &&
      "$(manifest_sha "$ROOT/d16")" == "$(sha256sum "$D16" | cut -d' ' -f1)" &&
      "$(manifest_sha "$ROOT/d32")" == "$(sha256sum "$D32" | cut -d' ' -f1)" ]]; then
  pass two-build-dirs-isolated
else
  fail two-build-dirs-isolated "generated DIM/SHA mixed between build dirs"
fi

# Switching the header of an existing build dir replaces the generated copy.
if configure d16 -DPOTAL_EXPECTED_DIM=32 "-DPOTAL_HW_PARAMS=$D32" &&
   [[ "$(generated_dim "$ROOT/d16")" == 32 ]] &&
   cmp -s "$ROOT/d16/generated/original-potal-params.h" "$D32"; then
  pass reconfigure-switch-header
else
  fail reconfigure-switch-header "stale generated parameters after switching header"
fi

# Editing the selected header re-runs validation on the next build.
work="$ROOT/headers/edited.h"
cp "$D16" "$work"
if ! configure edit -DPOTAL_EXPECTED_DIM=16 "-DPOTAL_HW_PARAMS=$work" ||
   ! cmake --build "$ROOT/edit" >> "$ROOT/edit.log" 2>&1; then
  fail header-edit "initial configure/build failed"
else
  before="$(manifest_sha "$ROOT/edit")"
  printf '/* benign edit */\n' >> "$work"
  if ! cmake --build "$ROOT/edit" >> "$ROOT/edit.log" 2>&1 ||
     [[ "$(manifest_sha "$ROOT/edit")" == "$before" ]]; then
    fail header-edit "benign edit did not refresh manifest SHA"
  else
    pass header-edit-reconfigures
  fi
  sed -i 's/^#define DIM 16$/#define DIM 32/' "$work"
  if cmake --build "$ROOT/edit" >> "$ROOT/edit.log" 2>&1; then
    fail header-edit-dim "build succeeded after header DIM changed to 32"
  elif ! log_has "$ROOT/edit.log" 'DIM mismatch: header reports 32, POTAL_EXPECTED_DIM=16'; then
    fail header-edit-dim "build failed without DIM mismatch message"
  else
    pass "header-edit-dim (expected build failure)"
  fi
fi

if [[ $failures -ne 0 ]]; then
  printf 'POTAL_HW_PARAMS_TEST FAIL (%d)\n' "$failures"
  exit 1
fi
printf 'POTAL_HW_PARAMS_TEST PASS\n'
