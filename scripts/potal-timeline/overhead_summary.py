#!/usr/bin/env python3
"""Summarize I25Q bench outputs: per-process and pooled wall statistics, ON/OFF overhead, hashes.

Usage: python3 -I overhead_summary.py RESULTS_DIR > overhead.md
Reads bench-off*.out, bench-win*.out, bench-detail.out, bench-queue.out (BENCH / BENCH_WARMUP lines).
Measured reps only (BENCH lines); warm-up reps are listed separately and never pooled.
"""
import glob
import os
import re
import statistics
import sys

LINE = re.compile(r"^(BENCH|BENCH_WARMUP) mode=(\S+) rep=(-?\d+) ok=(\d) status=(-?\d+) wall_ns=(\d+) .*"
                  r"queue_used=(\d+) queue_remaining=(\d+) hash=([0-9a-f]+)")


def load(path):
    rows = []
    for line in open(path, errors="replace"):
        m = LINE.match(line.strip())
        if m:
            rows.append({"kind": m.group(1), "mode": m.group(2), "rep": int(m.group(3)), "ok": m.group(4) == "1",
                         "wall_ms": int(m.group(6)) / 1e6, "queue_used": int(m.group(7)), "hash": m.group(9)})
    return rows


def stats(v):
    v = sorted(v)
    med = statistics.median(v)
    mad = statistics.median(abs(x - med) for x in v)
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    return med, mad, sd, v[0], v[-1]


def main():
    d = sys.argv[1]
    procs = {}
    for p in sorted(glob.glob(os.path.join(d, "bench-*.out"))):
        procs[os.path.basename(p)[6:-4]] = load(p)
    print("| process | mode | n | median ms | MAD ms | sd ms | min | max | warm-up ms | all ok | hashes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    pooled = {}
    for name, rows in procs.items():
        for mode in sorted({r["mode"] for r in rows}):
            meas = [r for r in rows if r["kind"] == "BENCH" and r["mode"] == mode]
            warm = [r for r in rows if r["kind"] == "BENCH_WARMUP" and r["mode"] == mode]
            if not meas:
                continue
            med, mad, sd, lo, hi = stats([r["wall_ms"] for r in meas])
            ok = all(r["ok"] for r in meas + warm)
            hashes = ",".join(sorted({r["hash"] for r in meas + warm}))
            print("| %s | %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %s | %s | %s |"
                  % (name, mode, len(meas), med, mad, sd, lo, hi,
                     "/".join("%.3f" % r["wall_ms"] for r in warm) or "-", ok, hashes))
            key = (re.sub(r"\d+$", "", name), mode)
            pooled.setdefault(key, []).extend(r["wall_ms"] for r in meas)
    print()
    print("| pooled | mode | n | median ms | MAD ms | sd ms |")
    print("|---|---|---|---|---|---|")
    for (name, mode), v in sorted(pooled.items()):
        med, mad, sd, _, _ = stats(v)
        print("| %s | %s | %d | %.3f | %.3f | %.3f |" % (name, mode, len(v), med, mad, sd))
    print()
    for mode in ("roles", "serial"):
        off, win = pooled.get(("off", mode)), pooled.get(("win", mode))
        if off and win:
            mo, mw = statistics.median(off), statistics.median(win)
            print("overhead %s window ON vs OFF: median %.3f vs %.3f ms = %+.2f%%" % (mode, mw, mo, 100 * (mw / mo - 1)))
        det = pooled.get(("detail", mode))
        if off and det:
            mo, md = statistics.median(off), statistics.median(det)
            print("overhead %s detail ON vs OFF: median %.3f vs %.3f ms = %+.2f%%" % (mode, md, mo, 100 * (md / mo - 1)))
    off_r, off_s = pooled.get(("off", "roles")), pooled.get(("off", "serial"))
    if off_r and off_s:
        print("roles vs serial (timeline OFF, pooled medians): %.3f vs %.3f ms" % (statistics.median(off_r), statistics.median(off_s)))
    for i in (1, 2):
        o, w = procs.get("off%d" % i), procs.get("win%d" % i)
        if o and w:
            mo = statistics.median(r["wall_ms"] for r in o if r["kind"] == "BENCH" and r["mode"] == "roles")
            mw = statistics.median(r["wall_ms"] for r in w if r["kind"] == "BENCH" and r["mode"] == "roles")
            print("pair %d roles: win %.3f vs off %.3f ms = %+.2f%%" % (i, mw, mo, 100 * (mw / mo - 1)))


if __name__ == "__main__":
    main()
