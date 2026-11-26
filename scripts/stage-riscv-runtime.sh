#!/bin/bash
# Post-build packaging only; never executes target code.

riscv_runtime_cache_value() {
  sed -n "s/^$2:[^=]*=//p" "$1/CMakeCache.txt"
}

riscv_runtime_abi() {
  local header flags data
  header=$(LC_ALL=C "$runtime_readelf" -h "$1") || return 1
  [[ "$header" =~ Class:[[:space:]]+ELF64 && "$header" =~ Machine:[[:space:]]+RISC-V ]] || return 1
  flags=$(sed -n 's/.*Flags:[[:space:]]*\(0x[0-9a-fA-F]*\).*/\1/p' <<< "$header")
  data=$(sed -n 's/.*Data:[[:space:]]*//p' <<< "$header")
  [[ -n "$flags" && -n "$data" ]] || return 1
  # RVC may differ; float ABI, RVE and TSO must agree.
  printf '%s:%s\n' "$data" "$((flags & 30))"
}

riscv_runtime_validate() {
  local path=$1 header dynamic abi
  if [[ -f "$path" ]] && abi=$(riscv_runtime_abi "$path") && [[ "$abi" == "$target_abi" ]]; then
    header=$(LC_ALL=C "$runtime_readelf" -h "$path") || return 1
    dynamic=$(LC_ALL=C "$runtime_readelf" -d "$path") || return 1
    if [[ "$header" =~ Type:[[:space:]]+DYN && "$dynamic" =~ \(SONAME\)[^$'\n']*\[libgomp\.so\.1\] ]]; then
      return 0
    fi
  fi
  printf 'ERROR: OpenMP runtime is incompatible with the RISC-V target: %s (missing/broken file, ELF64/RISC-V, ABI or SONAME mismatch)\n' "$path" >&2
  return 1
}

stage_riscv_openmp_runtime() (
  set -euo pipefail
  local build_dir=$1 bin="$1/bin" compiler cxx metadata cmake_version runtime_readelf sysroot
  local artifact magic dynamic abi target_abi='' candidate source='' directory tmp=''
  local -a consumers=() candidates=() search_dirs=()
  compiler=$(riscv_runtime_cache_value "$build_dir" CMAKE_C_COMPILER)
  cxx=$(riscv_runtime_cache_value "$build_dir" CMAKE_CXX_COMPILER)
  # Toolchain files can set compilers as normal variables, absent from the cache.
  if [[ -z "$compiler" ]]; then
    cmake_version="$(riscv_runtime_cache_value "$build_dir" CMAKE_CACHE_MAJOR_VERSION).$(riscv_runtime_cache_value "$build_dir" CMAKE_CACHE_MINOR_VERSION).$(riscv_runtime_cache_value "$build_dir" CMAKE_CACHE_PATCH_VERSION)"
    metadata="$build_dir/CMakeFiles/$cmake_version/CMakeCCompiler.cmake"
    if [[ -f "$metadata" ]]; then
      compiler=$(sed -n 's/^set(CMAKE_C_COMPILER "\([^"]*\)")/\1/p' "$metadata")
    fi
  fi
  compiler=${compiler:-$cxx}
  [[ -n "$compiler" ]] || { echo 'ERROR: No configured target compiler found' >&2; exit 1; }
  runtime_readelf=$(riscv_runtime_cache_value "$build_dir" CMAKE_READELF)
  if [[ -z "$runtime_readelf" ]]; then
    runtime_readelf=$("$compiler" -print-prog-name=readelf) || exit 1
  fi
  if ! command -v "$runtime_readelf" >/dev/null 2>&1; then
    printf 'ERROR: Target readelf unavailable: %s\n' "$runtime_readelf" >&2
    exit 1
  fi
  [[ -d "$bin" ]] || { printf 'ERROR: Build output directory missing: %s\n' "$bin" >&2; exit 1; }
  for artifact in "$bin"/*; do
    [[ -f "$artifact" ]] || continue
    magic=$(od -An -tx1 -N4 "$artifact") || exit 1
    [[ "${magic// /}" == 7f454c46 ]] || continue
    dynamic=$(LC_ALL=C "$runtime_readelf" -d "$artifact") || exit 1
    if [[ "$dynamic" =~ \(NEEDED\)[^$'\n']*\[libgomp\.so\.1\] ]]; then
      printf 'OpenMP dependency DETECTED: %s\n' "$artifact"
      abi=$(riscv_runtime_abi "$artifact") || { printf 'ERROR: Invalid RISC-V consumer: %s\n' "$artifact" >&2; exit 1; }
      [[ -z "$target_abi" || "$abi" == "$target_abi" ]] || { echo 'ERROR: Mixed target ABIs in bin/' >&2; exit 1; }
      target_abi=$abi
      consumers+=("$artifact")
      if [[ ! "$dynamic" =~ \((RPATH|RUNPATH)\)[^$'\n']*\$ORIGIN ]]; then
        printf 'WARNING: %s lacks $ORIGIN RPATH/RUNPATH; guest loader search path must include bin/\n' "$artifact" >&2
      fi
    fi
  done
  if ((${#consumers[@]} == 0)); then
    echo 'Runtime dependency scan PASS; libgomp staging SKIPPED (no DT_NEEDED)'
    exit 0
  fi
  sysroot=$(riscv_runtime_cache_value "$build_dir" CMAKE_SYSROOT)
  sysroot=${sysroot:-$("$compiler" -print-sysroot)}
  printf 'Target compiler: %s\nTarget sysroot: %s\n' "$compiler" "$sysroot"
  if [[ -n "${RISCV_LIBGOMP:-}" ]]; then
    candidates+=("$RISCV_LIBGOMP")
  else
    for candidate in "$compiler" "$cxx"; do
      [[ -n "$candidate" ]] || continue
      for directory in libgomp.so.1 libgomp.so; do
        source=$("$candidate" -print-file-name="$directory") || exit 1
        [[ "$source" == "$directory" ]] || candidates+=("$source")
      done
    done
    source=$("$compiler" -print-search-dirs) || exit 1
    source=$(sed -n 's/^libraries: =\{0,1\}//p' <<< "$source")
    IFS=: read -r -a search_dirs <<< "$source"
    if [[ -n "$sysroot" && "$sysroot" != / ]]; then
      search_dirs+=("$sysroot/lib" "$sysroot/usr/lib" "$sysroot/lib64" "$sysroot/usr/lib64")
    fi
    for directory in "${search_dirs[@]}"; do
      [[ -n "$directory" ]] || continue
      candidates+=("$directory/libgomp.so.1" "$directory/libgomp.so")
    done
  fi
  source=''
  for candidate in "${candidates[@]}"; do
    if [[ -z "${RISCV_LIBGOMP:-}" && ! -e "$candidate" && ! -L "$candidate" ]]; then
      continue
    fi
    if riscv_runtime_validate "$candidate"; then
      source=$(realpath -e "$candidate") || exit 1
      break
    fi
  done
  [[ -n "$source" ]] || { echo 'ERROR: Required compatible target libgomp.so.1 not found; set RISCV_LIBGOMP to a target runtime file' >&2; exit 1; }
  printf 'RISC-V libgomp FOUND; ABI VALID: %s (%s)\n' "$source" "$target_abi"
  # Rename replaces stale files/symlinks without following a destination symlink.
  trap 'if [[ -n "$tmp" ]]; then rm -f -- "$tmp"; fi' EXIT
  tmp=$(mktemp "$bin/.libgomp.so.1.XXXXXX") || exit 1
  cp -L -- "$source" "$tmp" || { echo 'ERROR: libgomp copy failed' >&2; exit 1; }
  chmod 644 "$tmp" || exit 1
  riscv_runtime_validate "$tmp" || exit 1
  mv -fT -- "$tmp" "$bin/libgomp.so.1" || { echo 'ERROR: libgomp staging rename failed' >&2; exit 1; }
  tmp=''
  riscv_runtime_validate "$bin/libgomp.so.1" || exit 1
  printf 'libgomp STAGED: %s -> %s/libgomp.so.1 (ELF64/RISC-V ABI and SONAME verified)\n' "$source" "$bin"
  dynamic=$(LC_ALL=C "$runtime_readelf" -d "$bin/libgomp.so.1") || exit 1
  printf '%s\n' 'libgomp additional DT_NEEDED (must be supplied by guest; not recursively packaged):'
  sed -n 's/.*(NEEDED).*\[\([^]]*\)\].*/  \1/p' <<< "$dynamic"
)
