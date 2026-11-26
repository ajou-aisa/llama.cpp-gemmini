#!/usr/bin/env python3
"""Draw a timeline_parse.py output directory as an SVG timeline (stdlib only).

Usage: plot_timeline.py DIR [--out FILE] [--run RUN_ID] [--t0-ns N] [--t1-ns N]

Reads DIR/spans.csv (and DIR/run_summary.csv when present), never the raw trace.
One lane per hart, one sub-lane per thread on that hart. x axis: ms from the first
placed span begin in spans.csv. --t0-ns/--t1-ns are absolute CLOCK_MONOTONIC ns as in
spans.csv; --run selects the node.total interval of that run from run_summary.csv
(spans of every thread inside it, since the issuer tags acc.* spans with its own
run id), or spans with that run_id when run_summary.csv is absent.
"""

import argparse
import csv
import os
import sys
from xml.sax.saxutils import escape

PALETTE = {
    1: "#4d4d4d", 10: "#1f77b4", 11: "#aec7e8", 12: "#6baed6", 13: "#3182bd", 14: "#08519c", 15: "#2171b5",
    16: "#9ecae1", 17: "#c6dbef", 20: "#d9d9d9", 21: "#ff7f0e", 22: "#fdd0a2", 23: "#fdae6b", 24: "#fd8d3c",
    25: "#e6550d", 26: "#9467bd", 27: "#c5b0d5", 28: "#bcbddc", 29: "#756bb1", 30: "#2ca02c", 31: "#98df8a",
    32: "#31a354", 33: "#d62728", 34: "#ff9896", 35: "#8c564b", 36: "#e377c2", 37: "#74c476", 38: "#17becf",
    39: "#dadaeb", 40: "#bcbd22", 41: "#dbdb8d", 42: "#e7ba52", 43: "#ad494a", 44: "#843c39", 45: "#a1d99b",
    46: "#c7e9c0", 47: "#637939", 48: "#de2d26", 49: "#f768a1", 50: "#7f7f7f", 51: "#969696", 52: "#6a51a3",
    53: "#9e9ac8", 54: "#feb24c", 55: "#525252",
}
FALLBACK = ["#8dd3c7", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#bc80bd"]
WAIT_COLOR = "#b0b0b0"
LEFT, RIGHT, TOP, PLOT_W, SUB_H, LANE_GAP = 170, 20, 70, 1400, 22, 8


def color(sid):
    return PALETTE.get(sid, FALLBACK[sid % len(FALLBACK)])


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--out")
    ap.add_argument("--run", type=int)
    ap.add_argument("--t0-ns", type=int)
    ap.add_argument("--t1-ns", type=int)
    args = ap.parse_args()
    out = args.out or os.path.join(args.dir, "timeline.svg")
    rows = read_csv(os.path.join(args.dir, "spans.csv"))
    rs_path = os.path.join(args.dir, "run_summary.csv")
    runs = read_csv(rs_path) if os.path.exists(rs_path) else None

    placed, unplaced = [], 0
    for r in rows:
        if r["begin_ns"] == "" or r["end_ns"] == "":
            unplaced += 1
            continue
        placed.append({"thread": int(r["thread"]), "role": r["role"], "hart": int(r["hart"]),
                       "sid": int(r["stage_id"]), "stage": r["stage"], "kind": r["kind"],
                       "run": int(r["run_id"]), "b": int(r["begin_ns"]), "e": int(r["end_ns"]),
                       "depth": int(r.get("depth") or 0), "stripe": r["stripe"]})
    origin = min((s["b"] for s in placed), default=0)
    t0, t1 = args.t0_ns, args.t1_ns
    brackets = runs or []
    if args.run is not None:
        brackets = [x for x in brackets if int(x["run_id"]) == args.run]
        if runs is not None and brackets:
            t0 = int(brackets[0]["begin_ns"]) if t0 is None else t0
            t1 = int(brackets[0]["end_ns"]) if t1 is None else t1
        else:
            placed = [s for s in placed if s["run"] == args.run]
    if t0 is None:
        t0 = min((s["b"] for s in placed), default=origin)
    if t1 is None:
        t1 = max((s["e"] for s in placed), default=t0 + 1)
    if t1 <= t0:
        t1 = t0 + 1
    shown = [s for s in placed if s["e"] >= t0 and s["b"] <= t1]

    lanes = {}
    for s in shown:
        lanes.setdefault(s["hart"], {}).setdefault(s["thread"], s["role"])
    layout, y = [], TOP + 30
    for hart in sorted(lanes):
        subs = sorted(lanes[hart].items())
        layout.append((hart, y, subs))
        y += SUB_H * len(subs) + LANE_GAP
    sub_y = {}
    for hart, y0, subs in layout:
        for i, (th, _) in enumerate(subs):
            sub_y[th] = y0 + i * SUB_H
    plot_bottom = y
    scale = PLOT_W / (t1 - t0)

    def X(t):
        return LEFT + (min(max(t, t0), t1) - t0) * scale

    shown.sort(key=lambda s: (s["thread"], s["kind"] != "wait", s["depth"], s["b"]))
    rects, drawn, merged = [], 0, None

    def flush():
        nonlocal merged
        if merged is not None:
            rects.append(merged)
            merged = None

    for s in shown:
        x0, x1 = X(s["b"]), X(s["e"])
        inset = min(s["depth"], 4) * 3 if s["kind"] == "busy" else 0
        yy = sub_y[s["thread"]] + 2 + inset
        h = max(3, SUB_H - 4 - 2 * inset)
        fill = WAIT_COLOR if s["kind"] == "wait" else color(s["sid"])
        key = (s["thread"], s["kind"], s["sid"], s["depth"])
        drawn += 1
        if merged is not None and merged["key"] == key and x0 - merged["x1"] < 0.5 and x1 - merged["x0"] < 0.5:
            merged["x1"] = max(merged["x1"], x1)
            merged["n"] += 1
            continue
        flush()
        merged = {"key": key, "x0": x0, "x1": x1, "y": yy, "h": h, "fill": fill, "n": 1, "s": s}
    flush()

    stages_seen = sorted({(s["sid"], s["stage"]) for s in shown if s["kind"] == "busy"})
    legend_rows = (len(stages_seen) + 5) // 6 + 1
    height = plot_bottom + 50 + legend_rows * 18
    width = LEFT + PLOT_W + RIGHT
    title = ("PoTal timeline %s: %d spans drawn (%d rects), %d unplaced spans skipped"
             % (os.path.basename(os.path.abspath(args.dir)), drawn, len(rects), unplaced))
    o = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
         'font-family="sans-serif" font-size="11">' % (width, height, width, height),
         "<title>%s</title>" % escape(title),
         '<rect x="0" y="0" width="%d" height="%d" fill="white"/>' % (width, height),
         '<text x="%d" y="18" font-size="14" font-weight="bold">%s</text>' % (LEFT, escape(title))]
    span_ms = (t1 - t0) / 1e6
    for i in range(11):
        t = t0 + (t1 - t0) * i / 10
        x = LEFT + PLOT_W * i / 10
        o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#eeeeee"/>' % (x, TOP + 25, x, plot_bottom))
        o.append('<text x="%.1f" y="%d" text-anchor="middle">%.3f</text>' % (x, plot_bottom + 14, (t - origin) / 1e6))
    o.append('<text x="%d" y="%d" text-anchor="middle">ms from first placed begin (window %.3f ms)</text>'
             % (LEFT + PLOT_W // 2, plot_bottom + 30, span_ms))
    for hart, y0, subs in layout:
        o.append('<rect x="0" y="%d" width="%d" height="%d" fill="#f7f7f7"/>' % (y0, width, SUB_H * len(subs)))
        o.append('<text class="lane-label" x="6" y="%d" font-weight="bold">hart %d</text>' % (y0 + 15, hart))
        for i, (th, role) in enumerate(subs):
            o.append('<text class="thread-label" x="60" y="%d">t%d %s</text>' % (y0 + i * SUB_H + 15, th, escape(role)))
    for r in rects:
        s = r["s"]
        w = max(r["x1"] - r["x0"], 0.2)
        tip = "%s %s t%d %s%s" % (s["kind"], s["stage"], s["thread"],
                                  "stripe %s " % s["stripe"] if s["stripe"] else "",
                                  "(%d merged)" % r["n"] if r["n"] > 1 else "%.3f us" % ((s["e"] - s["b"]) / 1e3))
        o.append('<rect x="%.2f" y="%d" width="%.2f" height="%d" fill="%s"><title>%s</title></rect>'
                 % (r["x0"], r["y"], w, r["h"], r["fill"], escape(tip)))
        if r["n"] == 1 and s["stripe"] and w > 0.003 * PLOT_W:
            o.append('<text x="%.2f" y="%d" font-size="9" text-anchor="middle">%s</text>'
                     % ((r["x0"] + r["x1"]) / 2, r["y"] + r["h"] - 3, escape(s["stripe"])))
    for b in brackets:
        bb, be = int(b["begin_ns"]), int(b["end_ns"])
        if be < t0 or bb > t1:
            continue
        x0, x1, yb = X(bb), X(be), TOP + 18
        o.append('<path class="run-bracket" d="M%.1f %d V%d H%.1f V%d" fill="none" stroke="black" stroke-width="1.5"/>'
                 % (x0, yb + 6, yb, x1, yb + 6))
        o.append('<text x="%.1f" y="%d" text-anchor="middle">run %s node.total %.3f ms, unattributed %.3f ms</text>'
                 % ((x0 + x1) / 2, yb - 4, escape(b["run_id"]), int(b["node_total_ns"]) / 1e6,
                    int(b["unattributed_ns"]) / 1e6))
    ly = plot_bottom + 48
    items = [(None, "wait", WAIT_COLOR)] + [(sid, name, color(sid)) for sid, name in stages_seen]
    for i, (sid, name, c) in enumerate(items):
        x, yy = LEFT + (i % 6) * 235, ly + (i // 6) * 18
        o.append('<rect x="%d" y="%d" width="12" height="12" fill="%s"/>' % (x, yy - 10, c))
        o.append('<text x="%d" y="%d">%s</text>' % (x + 16, yy, escape(name if sid is None else "%d %s" % (sid, name))))
    o.append("</svg>")
    with open(out, "w") as f:
        f.write("\n".join(o) + "\n")
    print("plot_timeline: %s (%d spans drawn as %d rects, %d unplaced skipped, %d harts)"
          % (out, drawn, len(rects), unplaced, len(layout)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
