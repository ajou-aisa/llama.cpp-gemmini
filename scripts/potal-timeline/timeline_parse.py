#!/usr/bin/env python3
"""Validate and analyse a PoTal timeline CSV (format potal_timeline_v1, see README.md).

Usage: timeline_parse.py TRACE.csv [--out DIR] [--strict] [--failed-runs LIST] [--wfi-tol-ns N]

Checks BEGIN/END and WAIT_BEGIN/WAIT_END pairing, HANDOFF_SEND/RECV edge
pairing, per-thread seq monotonicity, event counts, ring/anchor overflow and
migration (cpu_unverified) flags. Converts each thread's t_local to the common
CLOCK_MONOTONIC ns axis by piecewise-linear interpolation between that thread's
own anchors; extrapolation is allowed up to one neighbouring anchor interval and
refused (reported) beyond it. Files with mono_direct stamps (17 columns) use the
direct-stamp segment rule instead (see SegmentConverter).

Writes into DIR (default TRACE.csv.d): spans.csv, stage_totals.csv,
thread_util.csv, stripes.csv, stage_ii.csv, overlap.csv, critical_path.csv,
thread_stage.csv, run_summary.csv, run_thread.csv, parent_gaps.csv, buffers.csv,
status.csv.

Exit status: 0 valid; 1 pairing/sequence/format/accounting error; 2 warnings under
--strict; 3 incomplete trace (dropped events, ring overflow, unregistered records or
rejected threads) without errors.
Cross-thread times (overlap, critical path, handoff latency, run attribution) are
valid only after the Phase C clock probe passes; this tool does not prove that.
"""

import argparse
import bisect
import collections
import csv
import os
import sys
import time

EV_BEGIN, EV_END, EV_WAIT_BEGIN, EV_WAIT_END, EV_SEND, EV_RECV, EV_INSTANT = 1, 2, 3, 4, 5, 6, 7
NO_ID = 0xFFFFFFFF
ENVELOPE_STAGES = {1}  # node.total spans the whole node; excluded from the critical path
INCLUSIVE_PARENTS = {22, 21, 26, 39, 1}  # issuer.stripe, stripe.residual_prepare, exec.setup, exec.run, node.total
STAGE_SLOT_RELEASE = 44
MSG_CAP = 20  # printed messages per category and level; counts stay exact

THREAD_COLS = ["slot", "role", "req_hart", "req_cpu", "hart_id", "linux_cpu_id", "tid", "pin_verified_start",
               "pin_verified_end", "cpu_end", "cpu_unverified", "events", "dropped", "overflow", "anchors",
               "anchor_overflow", "runs", "run_overflow"]
EVENT_COLS = ["t_local", "seq", "slot", "stage", "event_type", "status", "op_type", "phase", "run_id", "layer_id",
              "stripe_id", "tile_id", "buffer_id", "edge_id", "work", "mono_direct"]


def decode(v):
    out, i = [], 0
    while i < len(v):
        if v[i] == "%" and i + 3 <= len(v):
            out.append(chr(int(v[i + 1:i + 3], 16)))
            i += 3
        else:
            out.append(v[i])
            i += 1
    return "".join(out)


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def print_capped(level, msgs):
    shown = collections.Counter()
    for m in msgs:
        cat = m.split(":", 1)[0]
        shown[cat] += 1
        if shown[cat] <= MSG_CAP:
            print("  %s %s" % (level, m))
    for cat, n in sorted(shown.items()):
        if n > MSG_CAP:
            print("  %s ... %d more '%s' messages (total %d)" % (level, n - MSG_CAP, cat, n))


def load(path, rep):
    meta, stages, threads, anchors, runs, events = {}, {}, {}, collections.defaultdict(list), [], []
    end_count = None
    with open(path, newline="") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#END,events="):
                end_count = int(line.split("=", 1)[1])
                continue
            if line.startswith("#"):
                continue
            cols = line.split(",")
            tag = cols[0]
            try:
                if tag == "META":
                    for kv in cols[1:]:
                        k, _, v = kv.partition("=")
                        if k == "env":
                            meta["env"] = dict((decode(a), decode(b)) for a, _, b in
                                               (e.partition("=") for e in v.split(";") if e))
                        else:
                            meta[k] = decode(v)
                elif tag == "S":
                    stages[int(cols[1])] = cols[2]
                elif tag == "T":
                    t = dict(zip(THREAD_COLS, cols[1:]))
                    for k in THREAD_COLS:
                        if k != "role":
                            t[k] = int(t[k])
                    threads[t["slot"]] = t
                elif tag == "A":
                    anchors[int(cols[1])].append({"idx": int(cols[2]), "kind": cols[3], "run_id": int(cols[4]),
                                                  "mono": int(cols[5]), "unc": int(cols[6]),
                                                  "t": int(cols[7]), "cpu": int(cols[8])})
                elif tag == "R":
                    runs.append(dict(zip(["run_id", "slot", "op_type", "phase", "layer_id", "begin_anchor",
                                          "end_anchor"], map(int, cols[1:]))))
                elif tag == "E":
                    if len(cols) not in (16, 17):
                        raise ValueError("expected 16 or 17 columns, got %d" % len(cols))
                    e = dict(zip(EVENT_COLS, map(int, cols[1:])))
                    e.setdefault("mono_direct", 0)
                    events.append(e)
                else:
                    raise ValueError("unknown row tag %r" % tag)
            except (ValueError, IndexError) as exc:
                rep.error("format: line %d: %s" % (lineno, exc))
    if meta.get("format") != "potal_timeline_v1":
        rep.error("format: missing META format=potal_timeline_v1")
    if end_count is None:
        rep.error("format: missing #END trailer (truncated file?)")
    elif end_count != len(events):
        rep.error("format: #END events=%d but %d E rows" % (end_count, len(events)))
    return meta, stages, threads, anchors, runs, events


class Converter:
    """Per-thread piecewise-linear t_local -> CLOCK_MONOTONIC ns map."""

    def __init__(self, slot, anchors, rep):
        pts = []
        for a in sorted(anchors, key=lambda a: (a["t"], a["idx"])):
            if pts and a["t"] == pts[-1][0]:
                continue
            if pts and a["mono"] < pts[-1][1]:
                rep.warn("anchor: slot %d anchor %d MONOTONIC decreases while t_local increases; dropped"
                         % (slot, a["idx"]))
                continue
            pts.append((a["t"], a["mono"]))
        self.t = [p[0] for p in pts]
        self.m = [p[1] for p in pts]
        self.extrapolated = 0
        self.refused = 0
        self.bad_segments = 0

    def _line(self, i, j, t):
        dt = self.t[j] - self.t[i]
        return self.m[i] + ((t - self.t[i]) * (self.m[j] - self.m[i])) // dt

    def __call__(self, t):
        n = len(self.t)
        if n == 0:
            self.refused += 1
            return None
        k = bisect.bisect_left(self.t, t)
        if k < n and self.t[k] == t:
            return self.m[k]
        if n < 2:
            self.refused += 1
            return None
        if 0 < k < n:
            return self._line(k - 1, k, t)
        if k == 0:
            if self.t[0] - t <= self.t[1] - self.t[0]:
                self.extrapolated += 1
                return self._line(0, 1, t)
        else:
            if t - self.t[-1] <= self.t[-1] - self.t[-2]:
                self.extrapolated += 1
                return self._line(n - 2, n - 1, t)
        self.refused += 1
        return None


class SegmentConverter:
    """Direct-stamp conversion for one thread (events in seq order).

    Rocket stops rdcycle while a hart is in WFI, so t_local is only trusted between two
    direct CLOCK_MONOTONIC stamps whose deltas agree (no WFI or migration in between).
    Events in a disagreeing or one-sided segment are left unconverted and counted.
    Direct-stamped events always keep their own stamp, so a span whose both ends are
    direct (role.start) takes its duration from MONOTONIC, never from t_local.
    """

    def __init__(self, evs, tol_ns=2000, tol_rel=2e-4, anchors=()):
        # Reference points: every direct-stamped event plus the thread's anchors (register,
        # run begin/end, thread exit, flush), each a (t_local, MONOTONIC) pair. A non-direct
        # event is placed only between two neighbouring points whose deltas agree.
        self.mono = {}
        self.refused = 0
        self.extrapolated = 0
        self.bad_segments = 0
        pts = {}
        for i, e in enumerate(evs):
            if e["mono_direct"]:
                self.mono[i] = e["mono_direct"]
                pts.setdefault(e["t_local"], e["mono_direct"])
        for a in anchors:
            pts.setdefault(a["t"], a["mono"])
        ts = sorted(pts)
        bad = set()
        for i, e in enumerate(evs):
            if i in self.mono:
                continue
            k = bisect.bisect_left(ts, e["t_local"])
            if k < len(ts) and ts[k] == e["t_local"]:
                self.mono[i] = pts[ts[k]]
                continue
            if k == 0 or k == len(ts):
                self.refused += 1
                continue
            ta, tb = ts[k - 1], ts[k]
            ma, mb = pts[ta], pts[tb]
            dc, dm = tb - ta, mb - ma
            if dc > 0 and dm >= 0 and abs(dm - dc) <= tol_ns + tol_rel * dm:
                self.mono[i] = ma + (e["t_local"] - ta) * dm // dc
            else:
                self.refused += 1
                bad.add(k)
        self.bad_segments = len(bad)

    def at(self, i):
        return self.mono.get(i)


def union(intervals):
    out = []
    for b, e in sorted(intervals):
        if out and b <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([b, e])
    return out


def length(u):
    return sum(e - b for b, e in u)


def intersect_len(a, b):
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def clip(u, lo, hi, starts=None):
    if starts is None:
        starts = [x[0] for x in u]
    k = max(0, bisect.bisect_right(starts, lo) - 1)
    out = []
    while k < len(u) and u[k][0] < hi:
        b, e = max(u[k][0], lo), min(u[k][1], hi)
        if e > b:
            out.append([b, e])
        k += 1
    return out


def subtract(b, e, u):
    out, cur = [], b
    for x0, x1 in u:
        if x0 > cur:
            out.append([cur, x0])
        cur = max(cur, x1)
    if e > cur:
        out.append([cur, e])
    return out


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(v):
    return "" if v is None else v


def build_forest(ss, rep, slot, sname):
    """Nest one thread's spans of one kind. Parent = innermost earlier span containing
    the child; a partial overlap is warned and the child becomes a sibling."""
    ss.sort(key=lambda s: (s["begin"], -s["end"], s["bseq"]))
    stack, overlaps = [], 0
    for s in ss:
        while stack and stack[-1]["end"] <= s["begin"] and not (
                stack[-1]["begin"] <= s["begin"] and s["end"] <= stack[-1]["end"]):
            stack.pop()
        parent = None
        for p in reversed(stack):
            if p["begin"] <= s["begin"] and s["end"] <= p["end"]:
                parent = p
                break
            if p["end"] > s["begin"] and not s.get("overlap"):
                s["overlap"] = True
                overlaps += 1
                rep.warn("overlap-not-nested: slot %d %s %s seq %d [%d,%d] partially overlaps %s seq %d [%d,%d]; "
                         "treated as sibling" % (slot, s["kind"], sname(s["stage"]), s["bseq"], s["begin"], s["end"],
                                                 sname(p["stage"]), p["bseq"], p["begin"], p["end"]))
        s["parent"] = parent
        s["depth"] = parent["depth"] + 1 if parent is not None else 0
        if parent is not None:
            parent["children"].append(s)
        stack.append(s)
    for s in ss:
        cu = union([(c["begin"], c["end"]) for c in s["children"]])
        s["children_ns"] = length(cu)
        s["excl"] = (s["end"] - s["begin"]) - s["children_ns"]
        s["excl_iv"] = subtract(s["begin"], s["end"], cu)
    return overlaps


def main():
    t_start = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--out")
    ap.add_argument("--strict", action="store_true", help="exit 2 on any warning (overflow, migration, refusal)")
    ap.add_argument("--failed-runs", default="",
                    help="comma list of run ids (a-b ranges allowed) known to fail by design; their unpaired edges are warnings")
    ap.add_argument("--wfi-tol-ns", type=int, default=2000,
                    help="direct-segment rdcycle/MONOTONIC agreement tolerance and per-thread mono decrease limit (ns)")
    args = ap.parse_args()
    failed_runs = set()
    for part in filter(None, args.failed_runs.split(",")):
        lo, _, hi = part.partition("-")
        failed_runs.update(range(int(lo), int(hi or lo) + 1))
    rep = Report()
    incomplete = []
    meta, stages, threads, anchors, runs, events = load(args.trace, rep)
    out = args.out or args.trace + ".d"
    os.makedirs(out, exist_ok=True)

    def sname(sid):
        return stages.get(sid, "stage%d" % sid)

    by_slot = collections.defaultdict(list)
    for e in events:
        by_slot[e["slot"]].append(e)
        if e["slot"] not in threads:
            rep.error("format: event on unknown thread slot %d" % e["slot"])

    for slot, t in sorted(threads.items()):
        if t["overflow"]:
            rep.warn("overflow: slot %d (%s) ring full, %d events dropped" % (slot, t["role"], t["dropped"]))
            incomplete.append("slot %d ring overflow" % slot)
        if t["dropped"] > 0:
            incomplete.append("slot %d dropped %d events" % (slot, t["dropped"]))
        if t["anchor_overflow"]:
            rep.warn("overflow: slot %d anchor table full" % slot)
        if t["run_overflow"]:
            rep.warn("overflow: slot %d run table full" % slot)
        if t["cpu_unverified"]:
            rep.warn("migration: slot %d (%s) cpu_unverified (start cpu %d, end cpu %d)"
                     % (slot, t["role"], t["linux_cpu_id"], t["cpu_end"]))
        if t["req_hart"] >= 0 and t["req_hart"] != t["hart_id"]:
            rep.warn("hart-mismatch: slot %d (%s) requested hart %d but runs on hart %d"
                     % (slot, t["role"], t["req_hart"], t["hart_id"]))
        if t["cpu_end"] != t["linux_cpu_id"]:
            rep.warn("cpu-end-mismatch: slot %d (%s) cpu_end %d != linux_cpu_id %d"
                     % (slot, t["role"], t["cpu_end"], t["linux_cpu_id"]))
        if t["req_cpu"] >= 0 and not t["pin_verified_start"]:
            rep.warn("pin-start: slot %d (%s) requested cpu %d but pin_verified_start=0"
                     % (slot, t["role"], t["req_cpu"]))
        if t["req_cpu"] >= 0 and not t["pin_verified_end"]:
            rep.warn("pin-end: slot %d (%s) requested cpu %d but pin_verified_end=0"
                     % (slot, t["role"], t["req_cpu"]))
        bad_anchor = [a for a in anchors.get(slot, []) if a["cpu"] != t["linux_cpu_id"]]
        if bad_anchor:
            rep.warn("anchor-cpu: slot %d (%s) %d anchors sampled on a cpu other than linux_cpu_id %d (first: idx %d cpu %d)"
                     % (slot, t["role"], len(bad_anchor), t["linux_cpu_id"], bad_anchor[0]["idx"], bad_anchor[0]["cpu"]))
        if len(by_slot.get(slot, [])) != t["events"]:
            rep.error("seq: slot %d thread table says %d events, found %d"
                      % (slot, t["events"], len(by_slot.get(slot, []))))
    for k in ("unregistered_records", "rejected_threads"):
        if int(meta.get(k, "0")) > 0:
            rep.warn("meta: %s=%s (events lost outside the thread table)" % (k, meta[k]))
            incomplete.append("%s=%s" % (k, meta[k]))

    direct_mode = any(e["mono_direct"] for e in events)
    conv = {} if direct_mode else {slot: Converter(slot, anchors.get(slot, []), rep) for slot in threads}
    seg = {slot: SegmentConverter(evs, tol_ns=args.wfi_tol_ns, anchors=anchors.get(slot, []))
           for slot, evs in by_slot.items()} if direct_mode else {}
    spans = []
    sends, recvs = collections.defaultdict(list), collections.defaultdict(list)
    unplaced_by_slot = collections.Counter()
    for slot, evs in sorted(by_slot.items()):
        th = threads.get(slot, {"overflow": 0, "role": "?", "hart_id": -1, "linux_cpu_id": -1, "cpu_unverified": 1})
        cv = conv.get(slot) or Converter(slot, [], rep)
        open_ = collections.defaultdict(list)
        prev_mono, small_dec = None, 0
        for i, e in enumerate(evs):
            if e["seq"] != i:
                rep.error("seq: slot %d event %d has seq %d (expected %d)" % (slot, i, e["seq"], i))
            e["mono"] = seg[slot].at(i) if direct_mode else cv(e["t_local"])
            # rdcycle is per hart: an unpinned thread (all events direct-stamped) may migrate,
            # so its order is checked on MONOTONIC (mono-order) instead of t_local.
            if i > 0 and e["t_local"] < evs[i - 1]["t_local"] and not (e["mono_direct"] and evs[i - 1]["mono_direct"]):
                rep.error("seq: slot %d seq %d t_local goes backwards" % (slot, e["seq"]))
            if e["mono"] is None:
                unplaced_by_slot[slot] += 1
            else:
                if prev_mono is not None and e["mono"] < prev_mono:
                    if prev_mono - e["mono"] > args.wfi_tol_ns:
                        rep.error("mono-order: slot %d seq %d placed MONOTONIC decreases by %d ns (> %d)"
                                  % (slot, e["seq"], prev_mono - e["mono"], args.wfi_tol_ns))
                    else:
                        small_dec += 1
                prev_mono = e["mono"]
            et = e["event_type"]
            key = (e["stage"], e["run_id"], e["stripe_id"], e["tile_id"])
            if et in (EV_BEGIN, EV_WAIT_BEGIN):
                open_[(et,) + key].append(e)
            elif et in (EV_END, EV_WAIT_END):
                bt = EV_BEGIN if et == EV_END else EV_WAIT_BEGIN
                stack = open_.get((bt,) + key)
                if not stack:
                    rep.error("pair: slot %d seq %d %s %s without matching begin"
                              % (slot, e["seq"], "END" if et == EV_END else "WAIT_END", sname(e["stage"])))
                    continue
                b = stack.pop()
                spans.append({"slot": slot, "role": th["role"], "hart": th["hart_id"], "cpu": th["linux_cpu_id"],
                              "stage": e["stage"], "stripe": e["stripe_id"], "tile": e["tile_id"],
                              "begin": b["mono"], "end": e["mono"], "status": e["status"],
                              "kind": "busy" if et == EV_END else "wait", "run": e["run_id"],
                              "dur_local": e["t_local"] - b["t_local"],
                              "buffer": e["buffer_id"], "work": e["work"], "unverified": th["cpu_unverified"],
                              "layer": e["layer_id"], "work_begin": b["work"], "work_end": e["work"], "bseq": b["seq"], "eseq": e["seq"],
                              "parent": None, "depth": None, "children": []})
            elif et == EV_SEND:
                sends[e["edge_id"]].append(e)
            elif et == EV_RECV:
                recvs[e["edge_id"]].append(e)
            elif et != EV_INSTANT:
                rep.error("format: slot %d seq %d unknown event_type %d" % (slot, e["seq"], et))
        if small_dec:
            rep.warn("mono-order: slot %d %d placed MONOTONIC decreases within %d ns tolerance"
                     % (slot, small_dec, args.wfi_tol_ns))
        for k, stack in open_.items():
            for b in stack:
                msg = "pair: slot %d seq %d %s %s never closed" % (
                    slot, b["seq"], "BEGIN" if k[0] == EV_BEGIN else "WAIT_BEGIN", sname(k[1]))
                if th["overflow"]:
                    rep.warn(msg + " (ring overflow truncated the thread)")
                else:
                    rep.error(msg)

    overflowed = {s for s, t in threads.items() if t["overflow"]}
    edges = []
    for edge in sorted(set(sends) | set(recvs)):
        s, r = sends.get(edge, []), recvs.get(edge, [])
        if edge == 0:
            rep.error("edge: %d handoff events carry edge_id 0" % (len(s) + len(r)))
            continue
        if len(s) == 1 and len(r) == 1:
            edges.append((edge, s[0], r[0]))
            if s[0]["mono"] is not None and r[0]["mono"] is not None and r[0]["mono"] < s[0]["mono"]:
                msg = "edge: %d RECV precedes SEND on the common axis by %d ns" % (edge, s[0]["mono"] - r[0]["mono"])
                if s[0]["mono_direct"] and r[0]["mono_direct"]:
                    rep.error(msg + " (both direct stamps)")
                else:
                    rep.warn(msg)
            continue
        msg = "edge: %d has %d SEND and %d RECV" % (edge, len(s), len(r))
        if overflowed:
            rep.warn(msg + " (a thread overflowed)")
        elif failed_runs and all(e["run_id"] in failed_runs for e in s + r):
            rep.warn(msg + " (declared failed run)")
        else:
            rep.error(msg)

    refused = sum(c.refused for c in (conv or seg).values())
    extrap = sum(c.extrapolated for c in (conv or seg).values())
    bad_segments = sum(c.bad_segments for c in seg.values())
    for slot, c in sorted(seg.items()):
        if c.bad_segments:
            rep.warn("clock: slot %d has %d direct-stamp segments whose rdcycle and MONOTONIC deltas disagree (WFI/migration); their events are unplaced" % (slot, c.bad_segments))
    for slot, c in sorted((conv or seg).items()):
        if c.refused:
            rep.warn("clock: slot %d %d events beyond one anchor interval (or <2 anchors): not converted"
                     % (slot, c.refused))

    for s in spans:
        if s["begin"] is not None and s["end"] is not None and s["end"] < s["begin"]:
            rep.error("pair: slot %d %s stripe %d ends before it begins" % (s["slot"], sname(s["stage"]), s["stripe"]))
    timed = [s for s in spans if s["begin"] is not None and s["end"] is not None and s["end"] >= s["begin"]]
    timed_ids = set(map(id, timed))
    unplaced_spans = [s for s in spans if id(s) not in timed_ids]

    forest_in = collections.defaultdict(list)
    for s in timed:
        forest_in[(s["slot"], s["kind"])].append(s)
    overlaps_by_slot = collections.Counter()
    for (slot, kind), ss in sorted(forest_in.items()):
        n = build_forest(ss, rep, slot, sname)
        if kind == "busy":
            overlaps_by_slot[slot] += n

    spans.sort(key=lambda s: (s["begin"] or 0, s["slot"], s["bseq"]))
    for i, s in enumerate(spans):
        s["id"] = i
    write_csv(os.path.join(out, "spans.csv"),
              ["thread", "role", "hart", "cpu", "stage", "stripe", "tile", "begin_ns", "end_ns", "status",
               "stage_id", "kind", "run_id", "buffer_id", "work", "cpu_unverified", "dur_local",
               "span_id", "parent_id", "depth", "work_begin", "work_end"],
              [[s["slot"], s["role"], s["hart"], s["cpu"], sname(s["stage"]),
                "" if s["stripe"] == NO_ID else s["stripe"], "" if s["tile"] == NO_ID else s["tile"],
                fmt(s["begin"]), fmt(s["end"]), s["status"], s["stage"], s["kind"], s["run"], s["buffer"],
                s["work"], s["unverified"], s["dur_local"], s["id"],
                "" if s["parent"] is None else s["parent"]["id"], fmt(s["depth"]), s["work_begin"], s["work_end"]]
               for s in spans])

    busy_u, wait_u = {}, {}
    for slot in threads:
        busy_u[slot] = union([(s["begin"], s["end"]) for s in forest_in.get((slot, "busy"), [])])
        wait_u[slot] = union([(s["begin"], s["end"]) for s in forest_in.get((slot, "wait"), [])])
    wait_starts = {slot: [x[0] for x in u] for slot, u in wait_u.items()}

    def wait_inside(s):
        return length(clip(wait_u.get(s["slot"], []), s["begin"], s["end"], wait_starts.get(s["slot"], [])))

    for slot in sorted(threads):
        ss = forest_in.get((slot, "busy"), [])
        if not ss:
            continue
        excl_sum, ul = sum(s["excl"] for s in ss), length(busy_u[slot])
        if abs(excl_sum - ul) > len(ss):
            msg = ("exclusive-sum: slot %d sum(exclusive_ns)=%d but busy union=%d (diff %d ns over %d spans)"
                   % (slot, excl_sum, ul, excl_sum - ul, len(ss)))
            if overlaps_by_slot[slot]:
                rep.warn(msg + "; %d overlap-not-nested spans double count their overlap" % overlaps_by_slot[slot])
            else:
                rep.error(msg)

    per_stage = collections.defaultdict(list)
    for s in timed:
        per_stage[(s["stage"], s["kind"])].append(s)
    unplaced_stage = collections.Counter((s["stage"], s["kind"]) for s in unplaced_spans)
    rows, unions = [], {}
    for (sid, kind) in sorted(set(per_stage) | set(unplaced_stage)):
        ss = per_stage.get((sid, kind), [])
        d = [s["end"] - s["begin"] for s in ss]
        u = union([(s["begin"], s["end"]) for s in ss])
        if kind == "busy" and ss:
            unions[sid] = u
        wi = sum(wait_inside(s) for s in ss) if kind == "busy" else ""
        rows.append([sid, sname(sid), kind, len(d), sum(d), fmt(min(d) if d else None),
                     fmt(sum(d) // len(d) if d else None), fmt(max(d) if d else None), length(u),
                     sum(d), sum(s["excl"] for s in ss), wi, sum(s["work"] for s in ss), unplaced_stage[(sid, kind)]])
    write_csv(os.path.join(out, "stage_totals.csv"),
              ["stage_id", "stage", "kind", "count", "total_ns", "min_ns", "mean_ns", "max_ns", "union_ns",
               "inclusive_ns", "exclusive_ns", "wait_inside_ns", "work_sum", "unplaced"], rows)

    ts = collections.defaultdict(list)
    for s in timed:
        ts[(s["slot"], s["stage"], s["kind"])].append(s)
    write_csv(os.path.join(out, "thread_stage.csv"),
              ["thread", "role", "hart", "stage", "kind", "count", "inclusive_ns", "exclusive_ns", "work_sum"],
              [[slot, threads.get(slot, {}).get("role", "?"), threads.get(slot, {}).get("hart_id", -1), sname(sid),
                kind, len(ss), sum(s["end"] - s["begin"] for s in ss), sum(s["excl"] for s in ss),
                sum(s["work"] for s in ss)] for (slot, sid, kind), ss in sorted(ts.items())])

    rows = []
    for slot, t in sorted(threads.items()):
        ms = [e["mono"] for e in by_slot.get(slot, []) if e["mono"] is not None]
        if not ms:
            rows.append([slot, t["role"], t["hart_id"], t["linux_cpu_id"], "", "", "", "", "", t["cpu_unverified"],
                         unplaced_by_slot[slot]])
            continue
        w0, w1 = min(ms), max(ms)
        busy, wait = busy_u[slot], wait_u[slot]
        busy_only = length(busy) - intersect_len(busy, wait)
        active = length(union([tuple(x) for x in busy + wait]))
        idle = (w1 - w0) - active
        if busy_only + length(wait) + idle != w1 - w0 or idle < 0:
            rep.error("util: slot %d busy %d + wait %d + idle %d != window %d"
                      % (slot, busy_only, length(wait), idle, w1 - w0))
        rows.append([slot, t["role"], t["hart_id"], t["linux_cpu_id"], w1 - w0, busy_only, length(wait),
                     idle, "%.4f" % (busy_only / (w1 - w0)) if w1 > w0 else "", t["cpu_unverified"],
                     unplaced_by_slot[slot]])
    write_csv(os.path.join(out, "thread_util.csv"),
              ["thread", "role", "hart", "cpu", "window_ns", "busy_ns", "wait_ns", "idle_ns", "busy_frac",
               "cpu_unverified", "unplaced_events"], rows)

    work_busy = sorted((s for s in timed if s["kind"] == "busy" and s["stage"] not in ENVELOPE_STAGES),
                       key=lambda s: s["begin"])
    work_begins = [s["begin"] for s in work_busy]
    all_busy_u = union([(s["begin"], s["end"]) for s in work_busy])
    all_wait_u = union([(s["begin"], s["end"]) for s in timed if s["kind"] == "wait"])
    timed_by_begin = sorted(timed, key=lambda s: s["begin"])
    timed_begins = [s["begin"] for s in timed_by_begin]
    run_rows, rt_rows = [], []
    for env in sorted((s for s in timed if s["kind"] == "busy" and s["stage"] in ENVELOPE_STAGES),
                      key=lambda s: (s["run"], s["begin"])):
        B, E = env["begin"], env["end"]
        total = E - B
        att = clip(all_busy_u, B, E)
        wt = clip(all_wait_u, B, E)
        wait_only = length(wt) - intersect_len(wt, att)
        unatt = total - length(att) - wait_only
        clipped, clipped_ns, crit, crit_stage = 0, 0, 0, ""
        for s in work_busy[:bisect.bisect_left(work_begins, E)]:
            if s["end"] <= B:
                continue
            lo, hi = max(s["begin"], B), min(s["end"], E)
            if lo != s["begin"] or hi != s["end"]:
                clipped += 1
                clipped_ns += (s["end"] - s["begin"]) - max(0, hi - lo)
            if hi - lo > crit:
                crit, crit_stage = hi - lo, sname(s["stage"])
        if clipped:
            rep.warn("run-clip: run %d node.total [%d,%d]: %d busy spans cross the envelope, %d ns clipped"
                     % (env["run"], B, E, clipped, clipped_ns))
        run_rows.append([env["run"], env.get("layer", ""), total, length(att), wait_only, unatt, crit, crit_stage,
                         B, E, env["slot"], clipped])
        per_thread = collections.defaultdict(list)
        for s in timed_by_begin[:bisect.bisect_left(timed_begins, E)]:
            if s["end"] > B or (s["begin"] == s["end"] == B):
                per_thread[s["slot"]].append(s)
        for slot, ss in sorted(per_thread.items()):
            bu = clip(union([(s["begin"], s["end"]) for s in ss
                             if s["kind"] == "busy" and s["stage"] not in ENVELOPE_STAGES]), B, E)
            wu = clip(union([(s["begin"], s["end"]) for s in ss if s["kind"] == "wait"]), B, E)
            busy_only = length(bu) - intersect_len(bu, wu)
            idle = total - length(union([tuple(x) for x in bu + wu]))
            excl = collections.Counter()
            for s in ss:
                if s["kind"] == "busy":
                    excl[s["stage"]] += length(clip(s["excl_iv"], B, E))
            rt_rows.append([env["run"], slot, threads.get(slot, {}).get("role", "?"), busy_only, length(wu), idle,
                            ";".join("%s:%d" % (sname(k), v) for k, v in sorted(excl.items()) if v)])
    write_csv(os.path.join(out, "run_summary.csv"),
              ["run_id", "layer_id", "node_total_ns", "attributed_ns", "wait_only_ns", "unattributed_ns",
               "critical_span_ns", "critical_stage", "begin_ns", "end_ns", "envelope_thread", "clipped_spans"],
              run_rows)
    write_csv(os.path.join(out, "run_thread.csv"),
              ["run_id", "thread", "role", "busy_ns", "wait_ns", "idle_ns", "exclusive_by_stage"], rt_rows)

    write_csv(os.path.join(out, "parent_gaps.csv"),
              ["thread", "stage", "stripe", "begin_ns", "inclusive_ns", "children_union_ns", "uncovered_ns",
               "run_id", "span_id"],
              [[s["slot"], sname(s["stage"]), "" if s["stripe"] == NO_ID else s["stripe"], s["begin"],
                s["end"] - s["begin"], s["children_ns"], s["excl"], s["run"], s["id"]]
               for s in sorted(timed, key=lambda s: (s["slot"], s["begin"]))
               if s["kind"] == "busy" and s["stage"] in INCLUSIVE_PARENTS])

    # buffer_id is a ring slot index reused by later stripes of the same run, so one
    # lifetime is (run, buffer, stripe): a span of the SAME stripe after its release is an error.
    buf = collections.defaultdict(list)
    for s in timed:
        if s["buffer"] not in (0, NO_ID):
            buf[(s["run"], s["buffer"], s["stripe"])].append(s)
    rows = []
    for (run, b, stripe), ss in sorted(buf.items()):
        first, last = min(s["begin"] for s in ss), max(s["end"] for s in ss)
        rel = [s["end"] for s in ss if s["kind"] == "busy" and s["stage"] == STAGE_SLOT_RELEASE]
        if rel:
            t_rel = min(rel)
            for s in ss:
                if s["begin"] > t_rel:
                    rep.error("buffer: use after release: run %d buffer %d stripe %d %s on slot %d begins at %d after "
                              "final.slot_release END at %d" % (run, b, stripe, sname(s["stage"]), s["slot"], s["begin"], t_rel))
        rows.append([run, b, "" if stripe == NO_ID else stripe, first, last, last - first,
                     ";".join(sorted({s["role"] for s in ss})), len(ss), fmt(min(rel) if rel else None)])
    write_csv(os.path.join(out, "buffers.csv"),
              ["run_id", "buffer_id", "stripe", "first_ns", "last_ns", "lifetime_ns", "roles", "spans", "release_end_ns"], rows)

    stripe_spans = collections.defaultdict(list)
    for s in timed:
        if s["kind"] == "busy" and s["stripe"] != NO_ID:
            stripe_spans[(s["run"], s["stripe"])].append(s)
    rows, prev = [], {}
    for (run, stripe), ss in sorted(stripe_spans.items(), key=lambda kv: (kv[0][0], min(s["begin"] for s in kv[1]))):
        b, e = min(s["begin"] for s in ss), max(s["end"] for s in ss)
        rows.append([run, stripe, b, e, e - b, fmt(b - prev[run] if run in prev else None), len(ss)])
        prev[run] = b
    write_csv(os.path.join(out, "stripes.csv"),
              ["run_id", "stripe", "begin_ns", "end_ns", "latency_ns", "ii_from_prev_ns", "spans"], rows)

    rows = []
    groups = collections.defaultdict(list)
    for s in timed:
        if s["kind"] == "busy" and s["stripe"] != NO_ID:
            groups[(s["run"], s["stage"])].append(s)
    for (run, sid), ss in sorted(groups.items()):
        firsts = {}
        for s in ss:
            firsts[s["stripe"]] = min(firsts.get(s["stripe"], s["begin"]), s["begin"])
        bs = [firsts[k] for k in sorted(firsts)]
        ii = [b - a for a, b in zip(bs, bs[1:])]
        rows.append([run, sid, sname(sid), len(bs), fmt(min(ii) if ii else None),
                     fmt(sum(ii) // len(ii) if ii else None), fmt(max(ii) if ii else None)])
    write_csv(os.path.join(out, "stage_ii.csv"),
              ["run_id", "stage_id", "stage", "stripes", "ii_min_ns", "ii_mean_ns", "ii_max_ns"], rows)

    rows = []
    sids = sorted(unions)
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            ov = intersect_len(unions[a], unions[b])
            la, lb = length(unions[a]), length(unions[b])
            rows.append([sname(a), sname(b), ov, "%.4f" % (ov / la) if la else "", "%.4f" % (ov / lb) if lb else ""])
    write_csv(os.path.join(out, "overlap.csv"), ["stage_a", "stage_b", "overlap_ns", "frac_of_a", "frac_of_b"], rows)

    path_rows, path_summary = critical_path(timed, edges, threads, sname)
    write_csv(os.path.join(out, "critical_path.csv"),
              ["step", "kind", "thread", "role", "stage", "stripe", "begin_ns", "end_ns", "dur_ns", "edge_id"],
              path_rows)

    placed = sum(1 for e in events if e.get("mono") is not None)
    if rep.errors:
        result, code = "FAIL", 1
    elif incomplete:
        result, code = "INCOMPLETE", 3
    elif args.strict and rep.warnings:
        result, code = "FAIL_STRICT", 2
    else:
        result, code = "PASS", 0
    wall = time.monotonic() - t_start
    write_csv(os.path.join(out, "status.csv"), ["key", "value"], [
        ["result", result], ["exit_code", code], ["errors", len(rep.errors)], ["warnings", len(rep.warnings)],
        ["events", len(events)], ["placed", placed], ["unplaced", len(events) - placed],
        ["dropped", sum(t["dropped"] for t in threads.values())],
        ["unregistered", int(meta.get("unregistered_records", "0"))],
        ["rejected_threads", int(meta.get("rejected_threads", "0"))],
        ["overflow_threads", len(overflowed)], ["bad_segments", bad_segments],
        ["spans", len(spans)], ["unplaced_spans", len(unplaced_spans)],
        ["incomplete_reasons", "; ".join(incomplete)], ["parse_wall_s", "%.3f" % wall]])

    print("timeline_parse: %s" % args.trace)
    print("  meta: model_id=%s build_id=%s image_id=%s clock_local=%s env_switches=%d"
          % (meta.get("model_id"), meta.get("build_id"), meta.get("image_id"), meta.get("clock_local"),
             len(meta.get("env", {}))))
    print("  threads=%d events=%d (placed %d) spans=%d (timed %d) edges_paired=%d runs=%d"
          % (len(threads), len(events), placed, len(spans), len(timed), len(edges), len(runs)))
    print("  overflow_threads=%d migrated_threads=%d clock_extrapolated=%d clock_refused=%d bad_segments=%d"
          % (len(overflowed), sum(1 for t in threads.values() if t["cpu_unverified"]), extrap, refused, bad_segments))
    for r in run_rows:
        print("  run %d: node_total %d ns, attributed %d, wait_only %d, unattributed %d"
              % (r[0], r[2], r[3], r[4], r[5]))
    print("  critical_path: %s" % path_summary)
    print("  cross-thread overlap/critical path valid only after the Phase C clock probe (not checked here)")
    print_capped("WARN ", rep.warnings)
    print_capped("ERROR", rep.errors)
    print("  outputs: %s (%.2f s)" % (out, wall))
    if code == 1:
        print("RESULT: FAIL (%d errors, %d warnings)" % (len(rep.errors), len(rep.warnings)))
    elif code == 3:
        print("RESULT: INCOMPLETE (%s; %d warnings)" % ("; ".join(incomplete), len(rep.warnings)))
    elif code == 2:
        print("RESULT: FAIL under --strict (%d warnings)" % len(rep.warnings))
    else:
        print("RESULT: PASS (%d warnings)" % len(rep.warnings))
    return code


def critical_path(timed, edges, threads, sname):
    """Backward walk from the last-ending span: at each step follow the handoff
    RECV that last unblocked the current span's thread to its sender, else the
    previous span on the same thread, else (no recorded edge, e.g. a join) the
    latest span on any other thread ending before it, labelled 'implicit'.
    An estimate, not a scheduling proof."""
    busy = [s for s in timed if s["kind"] == "busy" and s["stage"] not in ENVELOPE_STAGES]
    if not busy:
        return [], "no timed spans"
    per_thread = collections.defaultdict(list)
    for s in busy:
        per_thread[s["slot"]].append(s)
    ends = {}
    for slot, ss in per_thread.items():
        ss.sort(key=lambda s: s["end"])
        ends[slot] = [s["end"] for s in ss]
    recv_by_thread = collections.defaultdict(list)
    for edge, snd, rcv in edges:
        if snd["mono"] is not None and rcv["mono"] is not None:
            recv_by_thread[rcv["slot"]].append((rcv["mono"], edge, snd))
    for v in recv_by_thread.values():
        v.sort(key=lambda x: x[0])

    def last_ending_before(slot, t):
        k = bisect.bisect_right(ends.get(slot, []), t)
        return per_thread[slot][k - 1] if k > 0 else None

    cur = max(busy, key=lambda s: s["end"])
    chain, seen = [], set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(("span", cur, None))
        prev = last_ending_before(cur["slot"], cur["begin"])
        rv = recv_by_thread.get(cur["slot"], [])
        k = bisect.bisect_right([x[0] for x in rv], cur["begin"])
        hop = rv[k - 1] if k > 0 else None
        if hop is not None and (prev is None or hop[0] >= prev["end"]):
            t_recv, edge, snd = hop
            chain.append(("handoff", {"slot": cur["slot"], "begin": snd["mono"], "end": t_recv, "edge": edge,
                                      "from": snd["slot"]}, edge))
            cur = last_ending_before(snd["slot"], snd["mono"])
        elif prev is not None:
            cur = prev
        else:
            others = [x for x in (last_ending_before(s, cur["begin"]) for s in per_thread if s != cur["slot"])
                      if x is not None]
            nxt = max(others, key=lambda x: x["end"]) if others else None
            if nxt is not None:
                chain.append(("implicit", {"slot": cur["slot"], "begin": nxt["end"], "end": cur["begin"],
                                           "from": nxt["slot"]}, None))
            cur = nxt
    chain.reverse()
    rows, span_ns, hand_ns, implicit_ns = [], 0, 0, 0
    for i, (kind, s, edge) in enumerate(chain):
        role = threads.get(s["slot"], {}).get("role", "?")
        if kind == "span":
            span_ns += s["end"] - s["begin"]
            rows.append([i, "span", s["slot"], role, sname(s["stage"]), "" if s["stripe"] == NO_ID else s["stripe"],
                         s["begin"], s["end"], s["end"] - s["begin"], ""])
        else:
            if kind == "handoff":
                hand_ns += s["end"] - s["begin"]
            else:
                implicit_ns += s["end"] - s["begin"]
            rows.append([i, kind, "%d->%d" % (s["from"], s["slot"]), "", "", "", s["begin"], s["end"],
                         s["end"] - s["begin"], "" if edge is None else edge])
    first = chain[0][1]["begin"]
    last = chain[-1][1]["end"]
    total = last - first
    summary = ("%d steps, %d ns end-to-end: spans %d ns, handoff latency %d ns, implicit waits %d ns, "
               "other gaps %d ns" % (len(chain), total, span_ns, hand_ns, implicit_ns,
                                     total - span_ns - hand_ns - implicit_ns))
    return rows, summary


if __name__ == "__main__":
    sys.exit(main())
