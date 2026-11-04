#!/usr/bin/env python3
"""Per-run pipeline breakdown from timeline_parse.py outputs plus the bench output of the same process.

Usage: python3 -I breakdown.py PARSE_DIR BENCH_OUT [--skip-runs 1] > breakdown.md
Reads spans.csv, run_summary.csv, run_thread.csv. Roles runs are timeline run_ids (caller begin_run);
BENCH/BENCH_WARMUP lines of mode=roles map to them in order. Everything is on the common
CLOCK_MONOTONIC axis; unplaced spans are never used. Accelerator spans are CPU issue/return time.
"""
import argparse
import collections
import csv
import re
import statistics

ISSUER_STAGES = ["stripe.residual_prepare", "prep.decode", "prep.plan", "prep.rc_alloc", "exec.setup",
                 "exec.plan_copy", "exec.pack_residual", "exec.tlb_flush", "exec.run", "acc.scale_window",
                 "acc.release_window", "acc.index_window", "acc.queue_install", "acc.index_scalar", "acc.gemm_op",
                 "acc.main_issue", "acc.residual_issue", "acc.readback_op", "acc.readback", "acc.fence",
                 "issuer.stripe", "role.start"]


def union(iv):
    out = []
    for b, e in sorted(iv):
        if out and b <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([b, e])
    return out


def length(u):
    return sum(e - b for b, e in u)


def inter(a, b):
    i = j = t = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            t += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t


def clip(u, b, e):
    return [[max(x, b), min(y, e)] for x, y in u if min(y, e) > max(x, b)]


def ms(ns):
    return ns / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parse_dir")
    ap.add_argument("bench_out")
    ap.add_argument("--skip-runs", default="1", help="comma list of timeline run_ids to exclude (warm-up)")
    a = ap.parse_args()
    skip = {int(x) for x in a.skip_runs.split(",") if x}
    walls = [int(m.group(1)) for m in (re.match(r"^BENCH(?:_WARMUP)? mode=roles(?:\+queue)? .*wall_ns=(\d+)", l)
                                       for l in open(a.bench_out)) if m]
    runs = {int(r["run_id"]): r for r in csv.DictReader(open(a.parse_dir + "/run_summary.csv"))}
    spans = [s for s in csv.DictReader(open(a.parse_dir + "/spans.csv")) if s["begin_ns"] and s["end_ns"]]
    for s in spans:
        s["b"], s["e"] = int(s["begin_ns"]), int(s["end_ns"])
    rt = collections.defaultdict(dict)
    for r in csv.DictReader(open(a.parse_dir + "/run_thread.csv")):
        rt[int(r["run_id"])][r["role"]] = r

    rows = []
    stage_ex = collections.defaultdict(lambda: collections.defaultdict(list))
    for rid in sorted(runs):
        if rid in skip:
            continue
        r = runs[rid]
        B, E = int(r["begin_ns"]), int(r["end_ns"])
        node = E - B
        wall = walls[rid - 1] if rid - 1 < len(walls) else None
        sp = [s for s in spans if s["e"] > B and s["b"] < E]
        busy = lambda role, stages=None, stripe=None: union(
            [(max(s["b"], B), min(s["e"], E)) for s in sp if s["kind"] == "busy" and s["role"] == role and
             s["stage"] not in ("node.total",) and (stages is None or s["stage"] in stages) and
             (stripe is None or s["stripe"] == str(stripe))])
        exs = busy("exsia_local0") + busy("exsia_local1") + busy("exsia_fold")
        exs = union([tuple(x) for x in exs])
        iss, fin, fold = busy("issuer"), busy("finalizer"), busy("exsia_fold")
        loc = union([tuple(x) for x in busy("exsia_local0") + busy("exsia_local1")])
        st = {}
        for k in (0, 1):
            st[("local", k)] = busy("exsia_local0", {"exsia.local"}, k) + busy("exsia_local1", {"exsia.local"}, k)
            st[("fold", k)] = busy("exsia_fold", None, k)
            st[("issuer", k)] = busy("issuer", {"issuer.stripe"}, k)
            st[("final", k)] = busy("finalizer", None, k)
        span_of = lambda u: (min(x[0] for x in u), max(x[1] for x in u)) if u else (None, None)
        i0, i1 = span_of(st[("issuer", 0)]), span_of(st[("issuer", 1)])
        rows.append({
            "run": rid, "wall": wall, "node": node, "attr": int(r["attributed_ns"]), "wait_only": int(r["wait_only_ns"]),
            "unattr": int(r["unattributed_ns"]),
            "fill": (i0[0] - B) if i0[0] else None, "ii": (i1[0] - i0[0]) if i0[0] and i1[0] else None,
            "drain": (E - i1[1]) if i1[1] else None,
            "ov_fold0_local1": inter(union([tuple(x) for x in st[("fold", 0)]]), union([tuple(x) for x in st[("local", 1)]])),
            "ov_issuer0_fold1": inter(union([tuple(x) for x in st[("issuer", 0)]]), union([tuple(x) for x in st[("fold", 1)]])),
            "ov_final0_issuer1": inter(union([tuple(x) for x in st[("final", 0)]]), union([tuple(x) for x in st[("issuer", 1)]])),
            "ov_exsia_issuer": inter(exs, iss), "ov_local_fold": inter(loc, fold), "ov_issuer_final": inter(iss, fin),
            "any_busy": length(union([tuple(x) for x in exs + iss + fin + busy("caller")])),
        })
        for role, row in rt[rid].items():
            for kv in row["exclusive_by_stage"].split(";"):
                if kv:
                    k, v = kv.rsplit(":", 1)
                    stage_ex[role][k].append(int(v))
        for role in ("caller", "issuer", "finalizer", "exsia_local0", "exsia_local1", "exsia_fold"):
            if role in rt[rid]:
                x = rt[rid][role]
                stage_ex[role]["#busy"].append(int(x["busy_ns"]))
                stage_ex[role]["#wait"].append(int(x["wait_ns"]))
                stage_ex[role]["#idle"].append(int(x["idle_ns"]))

    med = lambda k: statistics.median([r[k] for r in rows if r[k] is not None])
    # roles and roles+queue BENCH lines map to timeline runs in order (non-roles modes are unregistered).
    print("runs used: %s (warm-up excluded: %s); n=%d" % ([r["run"] for r in rows], sorted(skip), len(rows)))
    print()
    print("| run | bench wall ms | node.total ms | outside node ms | attributed ms | wait-only ms | unattributed ms | fill ms | II ms | drain ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %d | %s | %.3f | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            r["run"], "%.3f" % ms(r["wall"]) if r["wall"] else "-", ms(r["node"]),
            "%.3f" % ms(r["wall"] - r["node"]) if r["wall"] else "-", ms(r["attr"]), ms(r["wait_only"]),
            ms(r["unattr"]), ms(r["fill"]), ms(r["ii"]), ms(r["drain"])))
    print("| median | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
        ms(med("wall")), ms(med("node")), ms(statistics.median([r["wall"] - r["node"] for r in rows if r["wall"]])),
        ms(med("attr")), ms(med("wait_only")), ms(med("unattr")), ms(med("fill")), ms(med("ii")), ms(med("drain"))))
    print()
    print("Cross-hart overlap (busy unions on the common axis, median ms over runs):")
    for k, label in (("ov_fold0_local1", "fold stripe0 x local stripe1"), ("ov_issuer0_fold1", "issuer stripe0 x fold stripe1"),
                     ("ov_final0_issuer1", "finalizer stripe0 x issuer stripe1"), ("ov_exsia_issuer", "ExSIA (all) x issuer"),
                     ("ov_local_fold", "local (both) x fold"), ("ov_issuer_final", "issuer x finalizer")):
        print("- %s: %.3f ms (min %.3f, max %.3f)" % (label, ms(med(k)), ms(min(r[k] for r in rows)), ms(max(r[k] for r in rows))))
    print("- union of all busy threads inside node.total: %.3f ms" % ms(med("any_busy")))
    print()
    print("Per-thread busy/wait/idle and exclusive time by stage (median ms per run):")
    for role in ("caller", "exsia_local0", "exsia_local1", "exsia_fold", "issuer", "finalizer"):
        d = stage_ex.get(role)
        if not d:
            continue
        print("\n%s: busy %.3f, wait %.3f, idle %.3f" % (role, ms(statistics.median(d["#busy"])),
                                                       ms(statistics.median(d["#wait"])), ms(statistics.median(d["#idle"]))))
        for k in sorted((k for k in d if not k.startswith("#")), key=lambda k: -statistics.median(d[k])):
            print("  - %s: %.3f" % (k, ms(statistics.median(d[k]))))


if __name__ == "__main__":
    main()
