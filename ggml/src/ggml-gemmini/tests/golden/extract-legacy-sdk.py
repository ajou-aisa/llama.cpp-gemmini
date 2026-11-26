#!/usr/bin/env python3
"""Extract the matmul/tiling/ISA slices of the legacy Gemmini SDK header.

Writes them verbatim, in source order, into one include file for
legacy-capture.cpp. The only substitution is the RoCC issue/fence bridge
(the header's host no-op ROCC_* block and gemmini_fence), which the fixture
defines as a recorder. Usage:

    extract-legacy-sdk.py <gemmini.h> <out.inc>
"""
import hashlib
import re
import sys

# (first line regex, last line regex) of each verbatim region.
REGIONS = [
    (r"^#define k_CONFIG 0$", r"^#define SOFTMAX 4$"),
    (r"^#ifndef GEMMINI_KBLOCK_DEBUG$", r"^#endif$"),
    (r"^#ifndef GEMMINI_WS_DEBUG$", r"^#endif$"),
    (r"^#ifndef GEMMINI_WS_LOOP_CYCLE$", r"^#endif$"),
    (r"^#ifndef GEMMINI_DISABLE_WS_REUSE$", r"^#endif$"),
    (r"^#ifdef HAS_MVIN_SCALE$", r"^#endif$"),
    (r"^static acc_scale_t acc_scale_t_bits_to_acc_scale_t", r"^}$"),
    (r"^static acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits", r"^}$"),
    (r"^// mvin and mvout$", r"^// fence$"),
    (r"^static inline int ceil_divide_int", r"^}$"),
    (r"^#define gemmini_loop_ws\(", r"^  }$"),
    (r"^static void sp_tiled_matmul_os\(", r"^}$"),
    (r"^static void sp_tiled_matmul_ws\(", r"^}$"),
    (r"^static void tiled_matmul_outer\(", r"^}$"),
    (r"^static acc_t int_sqrt\(", r"^}$"),
    (r"^static elem_t scale_and_sat\(", r"^}$"),
    (r"^#ifdef HAS_MVIN_SCALE$", r"^#endif$", "GEMMINI_SCALE"),
    (r"^#ifdef HAS_MVIN_ACC_SCALE$", r"^#endif$", "GEMMINI_ACC_SCALE"),
    (r"^static void matmul_cpu\(", r"^}$"),
    (r"^enum tiled_matmul_type_t", r"^enum tiled_matmul_type_t"),
    (r"^static void tiled_matmul\(", r"^}$"),
    (r"^static size_t tiled_matmul_total_spad_rows", r"^}$"),
    (r"^static size_t tiled_matmul_total_acc_rows", r"^}$"),
    (r"^  static void gemmini_set_tile_ws\(", r"^  }$"),
    (r"^static void tiled_matmul_auto\(", r"^}$"),
]


def find_region(lines, start_re, end_re, must_contain=None, after=0):
    for i in range(after, len(lines)):
        if not re.match(start_re, lines[i]):
            continue
        if start_re == end_re:
            return i, i
        for j in range(i + 1, len(lines)):
            if re.match(end_re, lines[j]):
                if must_contain and must_contain not in "\n".join(lines[i : j + 1]):
                    break
                return i, j
    raise SystemExit(f"region not found: {start_re} .. {end_re} ({must_contain})")


def main():
    src, out = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    lines = data.decode().split("\n")
    picked = []
    for region in REGIONS:
        start_re, end_re = region[0], region[1]
        must = region[2] if len(region) > 2 else None
        i, j = find_region(lines, start_re, end_re, must)
        if must is None and start_re.startswith("^#ifdef HAS_MVIN_SCALE"):
            i, j = find_region(lines, start_re, end_re, "scale_t_bits_to_scale_t")
        picked.append((i, j))
    with open(out, "w") as f:
        f.write(f"// Extracted from {src}\n// sha256 {hashlib.sha256(data).hexdigest()}\n")
        for i, j in sorted(set(picked)):
            f.write(f"// lines {i + 1}-{j + 1}\n")
            f.write("\n".join(lines[i : j + 1]) + "\n")


if __name__ == "__main__":
    main()
