#!/usr/bin/env python3
"""PoTal pipeline plots built from stored, already-reconstructed artifacts.

Two analyses, both read-only:

  stripes  Stripe-level pipelined stage timeline (LA -> SF -> WS -> RC) for
           GEMM invocations, bound through execution.sqlite / schedule.sqlite /
           npu-cycle-result.jsonl identities.  This is the primary view.

  token    Legacy token-wide semantic stage occupancy and synthetic-lane Gantt
           over timeline.jsonl.  Kept as a regression/debug output only.

Legacy invocation `plot_potal_pipeline.py TIMELINE.jsonl --token K` still maps
to the `token` subcommand.

Stripe binding (no timestamp matching anywhere):

  WS(s)  NPU work with provenance=dense_main, scope=stripe and stripe_id=s.
         Execution node npu:<work_id>; identity proven by
         services.npu[npu:<work_id>].request_sha256 == result.run_view_sha256
         and by the node's operation being the invocation's operation.
  RG(s)  NPU work with provenance=residual, scope=residual_compact|residual
         and stripe_id=s.  Must share the stripe's LA host stage.
  LA(s)  host:<id> for id in WS(s).required_host_stage_ids that are required
         by no other stripe's WS; each must also be an explicit edge parent.
  SETUP  host stages required by more than one stripe's WS.
  RP(s)  host stages required by RG(s) that are not LA(s) / SETUP.
  MC(s)  CPU nodes reachable from {WS(s), RG(s), LA(s), RP(s)} through explicit
         edges whose complete stripe-owner set is {s}.
  TAIL   CPU nodes reachable the same way whose owner set spans >1 stripe.
  UNBOUND CPU nodes of the invocation with no edge path to any stripe root.
  RC(s)  RP(s) u RG(s) u MC(s): elapsed envelope plus service union.
  SF(s)  Not present as a scheduled node in the stored execution IR; reported
         NOT_OBSERVED_IN_CURRENT_COLLECTION and never inferred.

Op/stage names are used only to label nodes after identity binding.
"""

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


# ---------------------------------------------------------------------------
# Legacy token-wide view (regression/debug output)
# ---------------------------------------------------------------------------


def records(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def find_endpoints(path):
    t0 = None
    ready = {}

    for r in records(path):
        if r.get("kind") != "event":
            continue

        if r.get("event") == "request_start":
            t0 = r["start_cycle"]

        elif r.get("event") == "token_ready":
            ready[r["token_index"]] = r["start_cycle"]

    if t0 is None:
        raise RuntimeError("request_start not found")

    return t0, ready


def stage_name(r):
    if r.get("kind") == "npu":
        cat = (
            r.get("npu_category")
            or r.get("npu_provenance")
            or "UNKNOWN"
        )

        if cat == "MAIN_GEMM":
            return "NPU Main GEMM"
        if cat == "RESIDUAL_GEMM":
            return "NPU Residual GEMM"
        return "NPU " + str(cat)

    op = r.get("op") or r.get("source") or "unknown"

    if op == "exsia.local" or op == "exsia.local_group":
        return "ExSIA Local"

    if op in ("exsia.folding", "exsia.stripe_total"):
        return "ExSIA Folding"

    if op == "exsia.stripe_submission":
        return "Stripe Submission"

    if op == "frontend.input_snapshot":
        return "Input Snapshot"

    if op == "im2p.stripe_input_capture":
        return "Stripe Input Capture"

    if op == "im2p.host_input_preparation":
        return "Host Input Prep"

    if op == "im2p.residual_metadata_preparation":
        return "Residual Metadata"

    if op == "im2p.residual_run_preparation":
        return "Residual Run Prep"

    if op == "im2p.residual_radix_recomposition":
        return "Radix Recompose"

    if op == "im2p.output_correction_apply":
        return "Correction Apply"

    if op == "im2p.output_reconstruction":
        return "Output Reconstruction"

    if op in (
        "frontend.output_copy",
        "im2p.output_buffer_copy",
        "im2p.residual_output_publish",
    ):
        return "Output Copy/Publish"

    if op == "cpu.mul_mat":
        return "CPU MatMul"

    if op == "cpu.softmax":
        return "CPU Softmax"

    if op == "sample_accept":
        return "Sampling"

    return "Other CPU"


STAGE_ORDER = [
    "ExSIA Local",
    "ExSIA Folding",
    "Stripe Submission",
    "Input Snapshot",
    "Stripe Input Capture",
    "Host Input Prep",
    "NPU Main GEMM",
    "Residual Metadata",
    "Residual Run Prep",
    "NPU Residual GEMM",
    "Radix Recompose",
    "Correction Apply",
    "Output Reconstruction",
    "Output Copy/Publish",
    "CPU MatMul",
    "CPU Softmax",
    "Sampling",
    "Other CPU",
]


def clipped_interval(r, lo, hi):
    a = r.get("start_cycle")
    b = r.get("end_cycle")

    if a is None or b is None:
        return None

    a = max(a, lo)
    b = min(b, hi)

    if b <= a:
        return None

    return a, b


def collect_window(path, lo, hi):
    result = []

    for r in records(path):
        if r.get("kind") not in ("cpu", "npu"):
            continue

        interval = clipped_interval(r, lo, hi)
        if interval is None:
            continue

        a, b = interval

        result.append({
            "kind": r.get("kind"),
            "stage": stage_name(r),
            "op": r.get("op"),
            "layer": r.get("layer"),
            "run_id": r.get("run_id"),
            "stripe_id": r.get("stripe_id"),
            "worker_id": r.get("worker_id"),
            "scheduler_lane": r.get("scheduler_lane"),
            "host_thread_id": r.get("host_thread_id"),
            "start": a,
            "end": b,
            "duration": b - a,
        })

    return result


def merge_intervals(intervals):
    if not intervals:
        return []

    intervals = sorted(intervals)
    result = [list(intervals[0])]

    for a, b in intervals[1:]:
        if a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])

    return [(a, b) for a, b in result]


def print_stage_summary(rows, lo, hi):
    by_stage = defaultdict(list)
    lanes = defaultdict(set)
    tids = defaultdict(set)
    workers = defaultdict(set)

    for r in rows:
        stage = r["stage"]
        by_stage[stage].append((r["start"], r["end"]))

        if r["scheduler_lane"] is not None:
            lanes[stage].add(r["scheduler_lane"])

        if r["host_thread_id"] is not None:
            tids[stage].add(r["host_thread_id"])

        if r["worker_id"] is not None:
            workers[stage].add(r["worker_id"])

    print("\n=== stage concurrency ===")
    print(
        f"{'stage':28s} {'union(M)':>10s} "
        f"{'window%':>8s} {'lanes':>7s} {'tids':>6s} {'workers':>8s}"
    )

    for stage in STAGE_ORDER:
        if stage not in by_stage:
            continue

        merged = merge_intervals(by_stage[stage])
        total = sum(b - a for a, b in merged)

        print(
            f"{stage:28s} "
            f"{total/1e6:10.3f} "
            f"{100*total/(hi-lo):8.2f} "
            f"{len(lanes[stage]):7d} "
            f"{len(tids[stage]):6d} "
            f"{len(workers[stage]):8d}"
        )


def stage_color_map(stages):
    cmap = plt.get_cmap("tab20")
    return {
        stage: cmap(i % 20)
        for i, stage in enumerate(stages)
    }


def plot_semantic(rows, lo, hi, out, title):
    present = [
        s for s in STAGE_ORDER
        if any(r["stage"] == s for r in rows)
    ]

    colors = stage_color_map(present)

    fig_h = max(4.5, 0.34 * len(present) + 1.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    ymap = {
        stage: len(present) - 1 - i
        for i, stage in enumerate(present)
    }

    for stage in present:
        intervals = merge_intervals([
            (r["start"], r["end"])
            for r in rows
            if r["stage"] == stage
        ])

        y = ymap[stage]

        for a, b in intervals:
            ax.broken_barh(
                [((a-lo)/1e6, (b-a)/1e6)],
                (y-0.35, 0.7),
                facecolors=colors[stage],
            )

    ax.set_yticks(
        [ymap[s] for s in present],
        labels=present,
    )

    ax.set_xlim(0, (hi-lo)/1e6)
    ax.set_xlabel("Schedule-axis cycles (million)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=250)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def lane_key(r):
    if r["kind"] == "npu":
        return "NPU"

    lane = r["scheduler_lane"]
    tid = r["host_thread_id"]

    if lane is not None and tid is not None:
        return f"CPU lane {lane} / tid {tid}"

    if lane is not None:
        return f"CPU lane {lane}"

    if tid is not None:
        return f"CPU tid {tid}"

    return "CPU unknown"


def plot_threads(rows, lo, hi, out, title):
    keys = sorted(
        {lane_key(r) for r in rows},
        key=lambda x: (x == "NPU", x),
    )

    stages = [
        s for s in STAGE_ORDER
        if any(r["stage"] == s for r in rows)
    ]
    colors = stage_color_map(stages)

    fig_h = max(4.5, 0.38 * len(keys) + 1.8)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    ymap = {
        key: len(keys) - 1 - i
        for i, key in enumerate(keys)
    }

    for r in rows:
        key = lane_key(r)
        y = ymap[key]

        ax.broken_barh(
            [(
                (r["start"]-lo)/1e6,
                r["duration"]/1e6
            )],
            (y-0.34, 0.68),
            facecolors=colors[r["stage"]],
        )

    ax.set_yticks(
        [ymap[k] for k in keys],
        labels=keys,
    )

    ax.set_xlim(0, (hi-lo)/1e6)
    ax.set_xlabel("Schedule-axis cycles (million)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    handles = [
        Patch(facecolor=colors[s], label=s)
        for s in stages
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
    )

    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=250)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def dominant_run(rows):
    score = defaultdict(int)
    meta = {}

    for r in rows:
        rid = r["run_id"]
        if rid is None:
            continue

        if r["stage"] == "Residual Run Prep":
            score[rid] += r["duration"]
            meta[rid] = r["layer"]

    if not score:
        return None, None

    rid = max(score, key=score.get)
    return rid, meta.get(rid)


def plot_run(rows, rid, lo, hi, out, layer):
    selected = [
        r for r in rows
        if r["run_id"] == rid
    ]

    if not selected:
        return

    rlo = min(r["start"] for r in selected)
    rhi = max(r["end"] for r in selected)

    plot_semantic(
        selected,
        rlo,
        rhi,
        out,
        f"PoTal pipeline: run {rid}, layer={layer}",
    )


def token_main(args):
    t0, ready = find_endpoints(args.timeline)

    k = args.token

    if k == 0:
        lo, hi = t0, ready[0]
        label = "TTFT"
    else:
        lo, hi = ready[k-1], ready[k]
        label = f"decode token {k}"

    rows = collect_window(args.timeline, lo, hi)

    print("window:", label)
    print("start :", lo)
    print("end   :", hi)
    print("cycles:", hi-lo)

    print_stage_summary(rows, lo, hi)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    plot_semantic(
        rows,
        lo,
        hi,
        Path(str(args.out) + "-semantic"),
        f"PoTal full pipeline — {label}",
    )

    plot_threads(
        rows,
        lo,
        hi,
        Path(str(args.out) + "-threads"),
        f"PoTal CPU/NPU lane allocation — {label}",
    )

    rid, layer = dominant_run(rows)

    print("\ndominant residual-prep run:", rid)
    print("layer:", layer)

    if rid is not None:
        plot_run(
            rows,
            rid,
            lo,
            hi,
            Path(str(args.out) + "-dominant-run"),
            layer,
        )


# ---------------------------------------------------------------------------
# Stripe-level pipeline view
# ---------------------------------------------------------------------------

SF_NOT_OBSERVED = "NOT_OBSERVED_IN_CURRENT_COLLECTION"
DECODE_SINGLE_STRIPE = (
    "decode invocation has one activation stripe; no intra-GEMM stripe "
    "pipeline exists for this invocation"
)
SINGLE_STRIPE_DEGENERATE = (
    "single-stripe decode invocation; intra-GEMM pipeline analysis is "
    "degenerate"
)

# Exact production provenance -> category.  Anything else stays UNCLASSIFIED.
NPU_CATEGORY = {
    ("dense_main", "stripe"): "MAIN_GEMM",
    ("residual", "residual_compact"): "RESIDUAL_GEMM",
    ("residual", "residual"): "RESIDUAL_GEMM",
}

PRIMARY_STAGES = ["LA", "SF", "WS", "RC"]
EXPANDED_STAGES = ["LA", "SF", "WS", "RP", "RG", "MC"]
STAGE_SIDE = {
    "LA": "CPU", "SF": "CPU", "WS": "NPU",
    "RC": "CPU+NPU", "RP": "CPU", "RG": "NPU", "MC": "CPU",
}
STAGE_TITLE = {
    "LA": "Local Adaptation",
    "SF": "Stripe Folding",
    "WS": "dense main WS GEMM",
    "RC": "Residual Compensation",
    "RP": "residual host prep",
    "RG": "residual NPU GEMM",
    "MC": "merge / correction",
}
# Fixed across every figure.
STAGE_COLOR = {
    "LA": "#4C78A8",
    "SF": "#9ECAE9",
    "WS": "#F58518",
    "RC": "#54A24B",
    "RP": "#B279A2",
    "RG": "#E45756",
    "MC": "#72B7B2",
    "SETUP": "#9D9D9D",
    "TAIL": "#6E6E6E",
}
CONSISTENCY_TOLERANCE = 0.10
SATURATION_OCCUPANCY = 0.50
PRIMARY_OF = {"RP": "RC", "RG": "RC", "MC": "RC"}
FAMILY_RE = re.compile(r"^(?P<family>[A-Za-z_]+?)-(?P<layer>\d+)$")
UNNAMED_RE = re.compile(r"^node_\d+$")
HOST_RE = re.compile(r"^host:(\d+)$")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def ro_sqlite(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def chunked(items, n=900):
    items = list(items)
    for i in range(0, len(items), n):
        yield items[i:i + n]


def rational(value):
    return Fraction(value["numerator"], value["denominator"])


def exact(value):
    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else float(value)
    return value


def union_length(intervals):
    return sum(b - a for a, b in merge_intervals(intervals))


def median(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return exact(Fraction(statistics.median(values)))


def compress_host_ids(ids):
    """Exact, lossless node-id list: host:N runs become [first, last] ranges."""
    numbers = []
    other = []
    for ident in ids:
        m = HOST_RE.match(ident)
        if m:
            numbers.append(int(m.group(1)))
        else:
            other.append(ident)
    numbers.sort()
    ranges = []
    for n in numbers:
        if ranges and n == ranges[-1][1] + 1:
            ranges[-1][1] = n
        else:
            ranges.append([n, n])
    out = sorted(other)
    for a, b in ranges:
        out.append(f"host:{a}" if a == b else f"host:{a}..host:{b}")
    return out


class Inputs:
    def __init__(self, export_path):
        self.export_path = export_path.resolve()
        self.export = json.loads(self.export_path.read_text())
        if self.export.get("schema") != "potal-timeline-export":
            raise SystemExit(f"{export_path}: unexpected schema {self.export.get('schema')}")
        inputs = self.export["inputs"]
        self.execution = Path(inputs["bundle"]["path"])
        self.schedule = Path(inputs["schedule"]["path"])
        self.npu_results = Path(inputs["npu_results"]["path"])
        self.expected = {
            "execution": inputs["bundle"]["sha256"],
            "schedule": inputs["schedule"]["sha256"],
            "npu_results": inputs["npu_results"]["sha256"],
        }
        self.frequency_hz = self.export["axis"]["frequency_hz"]
        self.source_run = Path(self.export["source_run"])
        manifest = self.source_run / "manifest.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.exists() else {}

    def files(self):
        return {
            "export": self.export_path,
            "execution": self.execution,
            "schedule": self.schedule,
            "npu_results": self.npu_results,
        }

    def model_label(self):
        argv = self.manifest.get("argv", [])
        model = None
        if "--model" in argv:
            model = Path(argv[argv.index("--model") + 1]).stem
        profile = self.manifest.get("profile")
        return " ".join(x for x in (model, profile) if x) or "unknown model"

    def hash_inputs(self):
        return {name: sha256_file(path) for name, path in self.files().items()}

    def to_cycles(self, ns):
        cycles = ns * self.frequency_hz / 1_000_000_000
        return cycles


class Invocation:
    """One GEMM invocation (execution-IR operation) and its stripe binding."""

    def __init__(self, operation_id, phase, decode_index, producer_operation_id):
        self.operation_id = operation_id
        self.phase = phase
        self.decode_index = decode_index
        self.producer_operation_id = producer_operation_id
        self.works = []
        self.layer_names = set()
        self.problems = []
        self.stripes = []
        self.setup = []
        self.tail = []
        self.unbound = []
        self.functional_emulation = 0
        self.sf_milestones = {}

    @property
    def layer(self):
        if len(self.layer_names) != 1:
            return None
        return next(iter(self.layer_names))

    @property
    def layer_canonical(self):
        layer = self.layer
        return layer is not None and not UNNAMED_RE.match(layer)

    @property
    def family(self):
        layer = self.layer
        if not self.layer_canonical:
            return None
        m = FAMILY_RE.match(layer)
        return m.group("family") if m else layer

    def label(self):
        if self.layer_canonical:
            return f"{self.layer} ({self.operation_id})"
        if self.layer is not None:
            return f"{self.operation_id} [recorded layer {self.layer}, not canonical]"
        return self.operation_id


def load_npu_results(path, phase):
    rows = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("semantic_phase_kind") != phase:
                continue
            category = NPU_CATEGORY.get((r.get("provenance"), r.get("scope")), "UNCLASSIFIED")
            rows.append({
                "work_id": r["work_id"],
                "sequence": r["sequence"],
                "stripe_id": r["stripe_id"],
                "run_id": r["run_id"],
                "host_slot": r["host_slot"],
                "phase_id": r["phase_id"],
                "producer_operation_id": r["operation_id"],
                "parent_id": r["parent_id"],
                "call_id": r["call_id"],
                "provenance": r["provenance"],
                "scope": r["scope"],
                "category": category,
                "required_host_stage_ids": list(r["required_host_stage_ids"]),
                "layer": r["layer"],
                "m": r["m"], "n": r["n"], "k": r["k"],
                "row_begin": r["row_begin"], "row_count": r["row_count"],
                "source_row_begin": r["source_row_begin"],
                "source_row_count": r["source_row_count"],
                "run_view_sha256": r["run_view_sha256"],
                "modeled_total_cycles": r["modeled"]["total_cycles"],
                "decode_index": r["semantic_decode_index"],
            })
    return rows


def bind_invocations(inputs, phase, log):
    exe = ro_sqlite(inputs.execution)
    sched = ro_sqlite(inputs.schedule)

    results = load_npu_results(inputs.npu_results, phase)
    log(f"NPU works in phase {phase}: {len(results)}")

    # Execution node npu:<work_id> -> operation, with request identity check.
    npu_nodes = {f"npu:{r['work_id']}": r for r in results}
    node_op = {}
    for chunk in chunked(npu_nodes):
        q = ",".join("?" * len(chunk))
        for ident, kind, op in exe.execute(
            f"select identity, kind, operation from nodes where identity in ({q})", chunk
        ):
            if kind != "NPU":
                raise AssertionError(f"{ident}: expected NPU node, got {kind}")
            node_op[ident] = op
    request = {}
    for chunk in chunked(npu_nodes):
        q = ",".join("?" * len(chunk))
        for ident, body in exe.execute(
            f"select identity, body from services where identity in ({q})", chunk
        ):
            request[ident] = json.loads(body)["npu"][ident]["request_sha256"]

    invocations = {}
    for ident, r in npu_nodes.items():
        if ident not in node_op:
            raise AssertionError(f"{ident}: NPU result has no execution node")
        if request.get(ident) != r["run_view_sha256"]:
            raise AssertionError(f"{ident}: service request_sha256 does not match NPU result")
        op = node_op[ident]
        inv = invocations.get(op)
        if inv is None:
            inv = invocations[op] = Invocation(op, phase, r["decode_index"], r["producer_operation_id"])
        if inv.producer_operation_id != r["producer_operation_id"]:
            raise AssertionError(f"{op}: works from several producer operations")
        r["node"] = ident
        inv.works.append(r)
        inv.layer_names.add(r["layer"])

    log(f"GEMM invocations with NPU work: {len(invocations)}")

    # Node inventory for the selected operations (one table scan).
    members = defaultdict(dict)
    for ident, kind, op in exe.execute("select identity, kind, operation from nodes"):
        if op in invocations:
            members[op][ident] = kind

    for inv in sorted(invocations.values(), key=lambda x: x.producer_operation_id):
        bind_one(exe, sched, inputs, inv, members[inv.operation_id])

    exe.close()
    sched.close()
    return sorted(invocations.values(), key=lambda x: (x.decode_index or 0, x.producer_operation_id))


def fetch_parents(exe, idents):
    parents = defaultdict(list)
    for chunk in chunked(idents):
        q = ",".join("?" * len(chunk))
        for node, parent, milestone in exe.execute(
            f"select node, parent, milestone from edges where node in ({q})", chunk
        ):
            parents[node].append((parent, milestone))
    return parents


def fetch_stage_labels(exe, idents):
    labels = {}
    for chunk in chunked(idents):
        q = ",".join("?" * len(chunk))
        for ident, body in exe.execute(
            f"select identity, body from services where identity in ({q})", chunk
        ):
            cpu = json.loads(body).get("cpu", {}).get(ident)
            if cpu:
                stages = {w.get("stage") for w in cpu["workers"]}
                labels[ident] = "/".join(sorted(s for s in stages if s)) or None
    return labels


def fetch_schedule(sched, inputs, idents):
    out = {}
    for chunk in chunked(idents):
        q = ",".join("?" * len(chunk))
        for ident, body in sched.execute(
            f"select identity, body from results where identity in ({q})", chunk
        ):
            b = json.loads(body)
            out[ident] = {
                "accepted": inputs.to_cycles(rational(b["accepted_ns"])),
                "result_ready": inputs.to_cycles(rational(b["result_ready_ns"])),
                "resource_ready": inputs.to_cycles(rational(b["resource_ready_ns"])),
                "resources": sorted({w["resource"] for w in b["worker_intervals"]}),
            }
    return out


def bind_one(exe, sched, inputs, inv, nodes):
    parents = fetch_parents(exe, nodes)
    children = defaultdict(list)
    for node, plist in parents.items():
        for parent, milestone in plist:
            if parent in nodes:
                children[parent].append((node, milestone))

    cpu_nodes = [n for n, k in nodes.items() if k == "CPU"]
    inv.functional_emulation = sum(1 for k in nodes.values() if k == "FUNCTIONAL_EMULATION")
    labels = fetch_stage_labels(exe, cpu_nodes)

    by_stripe = defaultdict(dict)
    for w in inv.works:
        if w["category"] == "UNCLASSIFIED":
            inv.problems.append(f"{w['node']}: UNCLASSIFIED provenance {w['provenance']}/{w['scope']}")
            continue
        slot = "WS" if w["category"] == "MAIN_GEMM" else "RG"
        if slot in by_stripe[w["stripe_id"]]:
            raise AssertionError(f"{inv.operation_id}: duplicate {slot} for stripe {w['stripe_id']}")
        by_stripe[w["stripe_id"]][slot] = w

    def required_hosts(work):
        hosts = []
        edge_parents = {p for p, _ in parents.get(work["node"], [])}
        for sid in work["required_host_stage_ids"]:
            host = f"host:{sid}"
            if nodes.get(host) != "CPU":
                raise AssertionError(f"{work['node']}: required {host} is not a CPU node of {inv.operation_id}")
            if host not in edge_parents:
                raise AssertionError(f"{work['node']}: required {host} has no explicit dependency edge")
            hosts.append(host)
        return hosts

    ws_required = {s: required_hosts(v["WS"]) for s, v in by_stripe.items() if "WS" in v}
    use_count = defaultdict(int)
    for hosts in ws_required.values():
        for h in hosts:
            use_count[h] += 1
    setup = {h for h, c in use_count.items() if c > 1}
    if len(ws_required) == 1:
        # A single stripe cannot distinguish shared setup from stripe-local
        # preparation through the WS requirement list; the stripe-local
        # stages are the ones the residual work also requires.
        only = next(iter(by_stripe.values()))
        rg_req = set(required_hosts(only["RG"])) if "RG" in only else set()
        setup = {h for h in next(iter(ws_required.values())) if h not in rg_req}

    owner = {}
    role = {}
    for s, parts in by_stripe.items():
        la = [h for h in ws_required.get(s, []) if h not in setup]
        for h in la:
            role[h] = ("LA", s)
            owner[h] = {s}
        rp = []
        if "RG" in parts:
            rg_req = required_hosts(parts["RG"])
            if "WS" in parts and not set(la) & set(rg_req):
                inv.problems.append(f"stripe {s}: RG shares no LA host stage with WS")
            rp = [h for h in rg_req if h not in la and h not in setup]
            for h in rp:
                if h in role:
                    raise AssertionError(f"{h}: owned by two stage roles")
                role[h] = ("RP", s)
                owner[h] = {s}
            owner[parts["RG"]["node"]] = {s}
        if "WS" in parts:
            owner[parts["WS"]["node"]] = {s}

    npu_role = {}
    for parts in by_stripe.values():
        for k, w in parts.items():
            npu_role[w["node"]] = k

    # Propagate stripe ownership down explicit edges among CPU/NPU nodes.
    roots = set(owner)
    order = topo_order(nodes, parents)
    for node in order:
        if node in roots or nodes[node] != "CPU" or node in setup:
            continue
        seen = set()
        reached = False
        for parent, _ in parents.get(node, []):
            if parent in owner and nodes.get(parent) in ("CPU", "NPU"):
                seen |= owner[parent]
                reached = True
        if reached:
            owner[node] = seen
            if len(seen) == 1:
                role[node] = ("MC", next(iter(seen)))
            else:
                role[node] = ("TAIL", None)

    timing = fetch_schedule(sched, inputs, list(nodes))
    resolver = Resolver(exe, sched, inputs, nodes, parents, timing)

    def interval(node):
        t = timing[node]
        a, b = t["accepted"], t["result_ready"]
        if b < a:
            raise AssertionError(f"{node}: result_ready < accepted")
        return a, b

    for node in cpu_nodes:
        if node in setup:
            inv.setup.append(node)
        elif node not in role:
            inv.unbound.append(node)
        elif role[node][0] == "TAIL":
            inv.tail.append(node)

    stage_nodes = defaultdict(lambda: defaultdict(list))
    for node, (r, s) in role.items():
        if r in ("LA", "RP", "MC"):
            stage_nodes[s][r].append(node)

    # SF evidence: producer lifecycle barriers carry explicit stripe identity
    # but are zero-duration scheduler milestones, not a measured interval.
    prefix = f"owner:{inv.producer_operation_id}:"
    for node in nodes:
        if node.startswith(prefix):
            parts = node.split(":")
            s = int(parts[2])
            transition = ":".join(parts[3:])
            if transition in ("ACTIVATION_ROWS:COMMIT", "RESIDUAL_PAYLOAD:SEAL"):
                inv.sf_milestones.setdefault(s, {})[transition] = exact(timing[node]["result_ready"])
    folding_labels = sorted({l for l in labels.values() if l and "fold" in l})
    inv.folding_labels = folding_labels

    ordered = sorted(by_stripe)
    stripe_ids = [s for s in ordered]
    if len(set(stripe_ids)) != len(stripe_ids):
        raise AssertionError(f"{inv.operation_id}: stripe ids not unique")

    for index, s in enumerate(ordered):
        parts = by_stripe[s]
        rec = {
            "operation_id": inv.operation_id,
            "producer_operation_id": inv.producer_operation_id,
            "layer": inv.layer if inv.layer_canonical else None,
            "recorded_layer": inv.layer,
            "phase": inv.phase,
            "decode_index": inv.decode_index,
            "stripe_index": index,
            "stripe_id": s,
            "run_id": (parts.get("WS") or parts.get("RG"))["run_id"],
            "slot": parts["WS"]["host_slot"] if "WS" in parts else None,
            "stages": {},
            "components": {},
        }

        rec["stages"]["LA"] = cpu_stage(stage_nodes[s]["LA"], interval, labels)
        rec["stages"]["SF"] = {
            "status": SF_NOT_OBSERVED,
            "start_cycle": None,
            "end_cycle": None,
            "elapsed_cycles": None,
            "service_union_cycles": None,
            "lifecycle_milestones": inv.sf_milestones.get(s, {}),
            "note": "only zero-duration owner barriers are scheduled; no folding interval exists in the execution IR",
        }
        rec["stages"]["WS"] = npu_stage(parts.get("WS"), timing, "MAIN_GEMM")
        rec["stages"]["RP"] = cpu_stage(stage_nodes[s]["RP"], interval, labels)
        rec["stages"]["RG"] = npu_stage(parts.get("RG"), timing, "RESIDUAL_GEMM")
        rec["stages"]["MC"] = cpu_stage(stage_nodes[s]["MC"], interval, labels)

        rc_intervals = []
        rc_sources = []
        for part in ("RP", "RG", "MC"):
            st = rec["stages"][part]
            if st["status"] == "OBSERVED":
                rc_intervals.extend(st["_intervals"])
                rc_sources.extend(st["source_nodes"])
        rec["stages"]["RC"] = envelope(rc_intervals, rc_sources, "RP+RG+MC")

        # Why each stage starts when it does: explicit dependency readiness
        # versus waiting for the stage's resource after dependencies resolved.
        for k in ("LA", "WS", "RP", "RG", "MC", "RC"):
            st = rec["stages"][k]
            if st["status"] != "OBSERVED":
                continue
            first = min(st["source_nodes"], key=lambda n: (timing[n]["accepted"], n))
            sw = st["start_wait"] = resolver.start_wait(first, labels)
            svc = sw.get("critical_service_node")
            if svc is not None:
                sw["critical_service_role"] = (
                    role[svc][0] if svc in role else
                    "SETUP" if svc in setup else
                    npu_role.get(svc, "OUTSIDE_STRIPE_BINDING"))
                sw["critical_service_stripes"] = sorted(owner[svc]) if svc in owner else None
            st["resources"] = sorted({r for n in st["source_nodes"] for r in timing[n]["resources"]})

        # Per-label service union inside RC: explains why RC is long.
        by_label = defaultdict(list)
        for part in ("RP", "MC"):
            for node in stage_nodes[s][part]:
                by_label[labels.get(node) or "unlabelled"].append(interval(node))
        if "RG" in parts:
            by_label["NPU RESIDUAL_GEMM"].append(interval(parts["RG"]["node"]))
        rec["components"] = {
            label: {
                "nodes": len(iv),
                "service_union_cycles": exact(union_length(iv)),
                "start_cycle": exact(min(a for a, _ in iv)),
                "end_cycle": exact(max(b for _, b in iv)),
            }
            for label, iv in sorted(by_label.items())
        }

        starts = [st["start_cycle"] for st in rec["stages"].values() if st.get("start_cycle") is not None]
        ends = [st["end_cycle"] for st in rec["stages"].values() if st.get("end_cycle") is not None]
        rec["start_cycle"] = min(starts) if starts else None
        rec["completion_cycle"] = max(ends) if ends else None
        missing = [k for k in ("LA", "WS", "RC") if rec["stages"][k]["status"] != "OBSERVED"]
        rec["completeness_status"] = "COMPLETE_EXCEPT_SF" if not missing else "MISSING_" + "_".join(missing)
        inv.stripes.append(rec)

    inv.setup_stage = envelope([interval(n) for n in inv.setup], inv.setup, "SETUP") if inv.setup else None
    inv.tail_stage = envelope([interval(n) for n in inv.tail], inv.tail, "TAIL") if inv.tail else None
    inv.unbound_summary = summarize_nodes(inv.unbound, interval, labels)
    inv.setup_labels = sorted({labels.get(n) for n in inv.setup if labels.get(n)})
    inv.tail_labels = sorted({labels.get(n) for n in inv.tail if labels.get(n)})


class Resolver:
    """Explicit-edge critical predecessor lookup, crossing operation bounds."""

    TERMINAL = ("CPU", "NPU", "APPLICATION_CPU")
    MILESTONE = {"RESULT_READY": "result_ready", "RESOURCE_READY": "resource_ready", "ACCEPTED": "accepted"}

    def __init__(self, exe, sched, inputs, nodes, parents, timing):
        self.exe, self.sched, self.inputs = exe, sched, inputs
        self.kind = dict(nodes)
        self.parents = dict(parents)
        self.timing = timing

    def _load(self, idents):
        need = [n for n in idents if n not in self.timing]
        if need:
            self.timing.update(fetch_schedule(self.sched, self.inputs, need))
        need = [n for n in idents if n not in self.kind]
        for chunk in chunked(need):
            q = ",".join("?" * len(chunk))
            for ident, kind in self.exe.execute(f"select identity, kind from nodes where identity in ({q})", chunk):
                self.kind[ident] = kind
        need = [n for n in idents if n not in self.parents]
        if need:
            got = fetch_parents(self.exe, need)
            for n in need:
                self.parents[n] = got.get(n, [])

    def ready(self, node):
        plist = self.parents.get(node)
        if plist is None:
            self._load([node])
            plist = self.parents[node]
        if not plist:
            return None, None, None
        self._load([p for p, _ in plist])
        best = max(plist, key=lambda pm: (self.timing[pm[0]][self.MILESTONE[pm[1]]], pm[0]))
        return best[0], best[1], self.timing[best[0]][self.MILESTONE[best[1]]]

    def start_wait(self, node, labels, depth=24):
        start = self.timing[node]["accepted"]
        parent, milestone, ready = self.ready(node)
        out = {
            "node": node,
            "start_cycle": exact(start),
            "dependency_ready_cycle": exact(ready) if ready is not None else None,
            "resource_wait_cycles": exact(start - ready) if ready is not None else None,
            "critical_chain": [],
        }
        cur = node
        for _ in range(depth):
            parent, milestone, ready = self.ready(cur)
            if parent is None:
                break
            kind = self.kind.get(parent)
            out["critical_chain"].append({
                "node": parent, "kind": kind, "milestone": milestone, "ready_cycle": exact(ready),
            })
            if kind in self.TERMINAL:
                if parent not in labels and kind == "CPU":
                    labels.update(fetch_stage_labels(self.exe, [parent]))
                out["critical_service_node"] = parent
                out["critical_service_label"] = labels.get(parent) or kind
                out["critical_milestone"] = milestone
                break
            cur = parent
        return out


def topo_order(nodes, parents):
    indegree = {n: 0 for n in nodes}
    kids = defaultdict(list)
    for node in nodes:
        for parent, _ in parents.get(node, []):
            if parent in nodes:
                indegree[node] += 1
                kids[parent].append(node)
    ready = sorted(n for n, d in indegree.items() if d == 0)
    order = []
    while ready:
        n = ready.pop()
        order.append(n)
        for k in kids[n]:
            indegree[k] -= 1
            if indegree[k] == 0:
                ready.append(k)
    if len(order) != len(nodes):
        raise AssertionError("dependency cycle inside invocation")
    return order


def envelope(intervals, sources, composition):
    if not intervals:
        return {"status": "NOT_BOUND", "start_cycle": None, "end_cycle": None,
                "elapsed_cycles": None, "service_union_cycles": None,
                "source_nodes": [], "composition": composition, "_intervals": []}
    a = min(x for x, _ in intervals)
    b = max(y for _, y in intervals)
    service = union_length(intervals)
    if service > b - a:
        raise AssertionError("service union exceeds elapsed envelope")
    return {
        "status": "OBSERVED",
        "start_cycle": exact(a),
        "end_cycle": exact(b),
        "elapsed_cycles": exact(b - a),
        "service_union_cycles": exact(service),
        "bubble_cycles": exact(b - a - service),
        "source_nodes": list(sources),
        "composition": composition,
        "_intervals": intervals,
    }


def cpu_stage(node_list, interval, labels):
    st = envelope([interval(n) for n in node_list], node_list, "CPU")
    st["labels"] = sorted({labels.get(n) or "unlabelled" for n in node_list})
    return st


def npu_stage(work, timing, expected):
    if work is None:
        return envelope([], [], "NPU")
    if work["category"] != expected:
        raise AssertionError(f"{work['node']}: {work['category']} bound as {expected}")
    t = timing[work["node"]]
    st = envelope([(t["accepted"], t["result_ready"])], [work["node"]], "NPU")
    st.update({
        "npu_category": work["category"],
        "provenance": f"{work['provenance']}/{work['scope']}",
        "work_id": work["work_id"],
        "call_id": work["call_id"],
        "result_ready_cycle": exact(t["result_ready"]),
        "resource_ready_cycle": exact(t["resource_ready"]),
        "modeled_total_cycles": work["modeled_total_cycles"],
        "shape_mnk": [work["m"], work["n"], work["k"]],
        "labels": [f"NPU {work['category']}"],
    })
    return st


def summarize_nodes(node_list, interval, labels):
    by_label = defaultdict(list)
    for n in node_list:
        by_label[labels.get(n) or "unlabelled"].append(interval(n))
    return {
        label: {"nodes": len(iv), "service_union_cycles": exact(union_length(iv))}
        for label, iv in sorted(by_label.items())
    }


def pipeline_metrics(inv, stage_set):
    stripes = inv.stripes
    observed = [k for k in stage_set if all(s["stages"][k]["status"] == "OBSERVED" for s in stripes)]
    start = min(s["start_cycle"] for s in stripes)
    completion = [s["completion_cycle"] for s in stripes]
    makespan = max(completion) - start
    out_ii = [b - a for a, b in zip(completion, completion[1:])]

    stages = {}
    for k in stage_set:
        sts = [s["stages"][k] for s in stripes]
        if k not in observed:
            stages[k] = {"status": sts[0]["status"] if sts else "NOT_BOUND"}
            continue
        elapsed = [st["elapsed_cycles"] for st in sts]
        service = [st["service_union_cycles"] for st in sts]
        bubble = [st["bubble_cycles"] for st in sts]
        starts = [st["start_cycle"] for st in sts]
        start_ii = [b - a for a, b in zip(starts, starts[1:])]
        occupancy = union_length([(st["start_cycle"], st["end_cycle"]) for st in sts])
        stages[k] = {
            "status": "OBSERVED",
            "median_elapsed": median(elapsed),
            "max_elapsed": max(elapsed),
            "total_elapsed": sum(elapsed),
            "median_service": median(service),
            "total_service": sum(service),
            "median_bubble": median(bubble),
            "total_bubble": sum(bubble),
            "start_ii": start_ii,
            "median_start_ii": median(start_ii),
            "occupancy_union_cycles": occupancy,
            "occupancy": occupancy / makespan if makespan else None,
        }

    gaps = {}
    for a, b in zip(observed, observed[1:]):
        g = [s["stages"][b]["start_cycle"] - s["stages"][a]["end_cycle"] for s in stripes]
        skipped = [k for k in stage_set[stage_set.index(a) + 1:stage_set.index(b)]]
        name = f"{a}->{b}" + (f" (unobserved {','.join(skipped)} between)" if skipped else "")
        blockers = defaultdict(int)
        for s, gap in zip(stripes, g):
            if gap > 0:
                sw = s["stages"][b].get("start_wait") or {}
                rel = sw.get("critical_service_stripes")
                where = ("same stripe" if rel == [s["stripe_id"]] else
                         "earlier stripe" if rel and max(rel) < s["stripe_id"] else
                         "other" if rel else "unbound")
                blockers[f"{sw.get('critical_service_role')}|{sw.get('critical_milestone')}|{where}"] += 1
        gaps[name] = {"per_stripe": g, "median": median(g), "positive_gap_blockers": dict(blockers)}

    return {
        "stage_set": stage_set,
        "stripe_count": len(stripes),
        "first_start_cycle": start,
        "last_completion_cycle": max(completion),
        "pipeline_makespan": makespan,
        "output_ii": out_ii,
        "median_output_ii": median(out_ii),
        "stages": stages,
        "gaps": gaps,
        "assessment": assess(stages, median(out_ii), gaps, stage_set),
    }


def assess(stages, out_ii, gaps, stage_set):
    obs = {k: v for k, v in stages.items() if v["status"] == "OBSERVED"}
    if not obs:
        return {"assessment": "INCONCLUSIVE; no observed stages"}

    def best(key):
        return max(obs, key=lambda k: (obs[k][key] if obs[k][key] is not None else -1))

    result = {
        "longest_service_stage": best("median_service"),
        "longest_elapsed_stage": best("median_elapsed"),
        "largest_bubble_stage": (best("median_bubble")
                                 if (obs[best("median_bubble")]["median_bubble"] or 0) > 0 else None),
        "output_ii": out_ii,
        "tolerance": CONSISTENCY_TOLERANCE,
    }
    consistent = []
    closest = None
    if out_ii:
        rel = {}
        for k, v in obs.items():
            if v["median_start_ii"] is not None:
                rel[k] = abs(v["median_start_ii"] - out_ii) / out_ii
        if rel:
            closest = min(rel, key=rel.get)
            result["stage_most_consistent_with_output_ii"] = closest
            result["relative_ii_error"] = {k: round(float(x), 4) for k, x in rel.items()}
            consistent = [k for k, x in rel.items() if x <= CONSISTENCY_TOLERANCE]
    positive = {k: g["median"] for k, g in gaps.items() if g["median"] is not None and g["median"] > 0}
    downstream = None
    if positive:
        gap = max(positive, key=positive.get)
        result["largest_positive_gap"] = gap
        blockers = gaps[gap]["positive_gap_blockers"]
        if len(blockers) == 1:
            role, milestone, where = next(iter(blockers)).split("|")
            stage = PRIMARY_OF.get(role, role) if "RC" in stage_set else role
            downstream = stage
            result["dominant_downstream_wait"] = (
                f"{gap} gap (median {fmt_m(gaps[gap]['median'])}) waits on {role} {milestone} of {where} "
                f"in every positive-gap stripe")
            result["dominant_downstream_wait_stage"] = stage
        else:
            result["dominant_downstream_wait"] = f"{gap} gap has mixed blockers {blockers}"

    missing = [k for k in stage_set if stages.get(k, {}).get("status") != "OBSERVED"]
    caveat = f"; {','.join(missing)} not observed in current collection" if missing else ""
    if out_ii is None:
        result["pipeline_limiting_candidate"] = None
        result["assessment"] = "INCONCLUSIVE; single stripe has no output cadence" + caveat
    elif consistent:
        # Matching cadence alone also fits a stage that merely follows its
        # producer; a limiting stage must also be saturated.
        cand = max(consistent, key=lambda k: obs[k]["occupancy"] or 0)
        if (obs[cand]["occupancy"] or 0) >= SATURATION_OCCUPANCY:
            result["pipeline_limiting_candidate"] = cand
            result["assessment"] = (
                f"{cand} limits the observed steady-state stripe cadence "
                f"(median start-II within {int(CONSISTENCY_TOLERANCE*100)}% of output II, "
                f"occupancy {obs[cand]['occupancy']:.0%})" + caveat
            )
        else:
            result["pipeline_limiting_candidate"] = None
            result["assessment"] = (
                f"INCONCLUSIVE; {','.join(sorted(consistent))} match output II but none is saturated "
                f"(best occupancy {obs[cand]['occupancy']:.0%} < {SATURATION_OCCUPANCY:.0%})" + caveat
            )
    elif downstream is not None:
        result["pipeline_limiting_candidate"] = downstream
        result["assessment"] = (
            f"{downstream} creates the dominant downstream wait ({result['dominant_downstream_wait']}); "
            f"no stage start-II within {int(CONSISTENCY_TOLERANCE*100)}% of output II (closest: {closest})" + caveat
        )
    else:
        result["pipeline_limiting_candidate"] = None
        result["assessment"] = (
            f"INCONCLUSIVE; no stage start-II within {int(CONSISTENCY_TOLERANCE*100)}% "
            f"of output II (closest: {closest})" + caveat
        )
    return result


def fmt_m(v):
    if v is None:
        return "n/a"
    return f"{float(v)/1e6:.3f}M"


def print_bottleneck(inv, metrics, out=sys.stdout):
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"\n=== {inv.label()} | {inv.phase} | {metrics['stripe_count']} stripes | stages {'/'.join(metrics['stage_set'])} ===")
    p(f"pipeline makespan : {fmt_m(metrics['pipeline_makespan'])} cycles")
    p(f"median output II  : {fmt_m(metrics['median_output_ii'])} cycles  {[fmt_m(x) for x in metrics['output_ii']]}")
    p(f"{'stage':6s} {'med elapsed':>12s} {'max elapsed':>12s} {'med service':>12s} "
      f"{'med start-II':>13s} {'occupancy':>10s} {'med bubble':>11s}")
    for k in metrics["stage_set"]:
        st = metrics["stages"][k]
        if st["status"] != "OBSERVED":
            p(f"{k:6s} {st['status']}")
            continue
        p(f"{k:6s} {fmt_m(st['median_elapsed']):>12s} {fmt_m(st['max_elapsed']):>12s} "
          f"{fmt_m(st['median_service']):>12s} {fmt_m(st['median_start_ii']):>13s} "
          f"{st['occupancy']:>10.1%} {fmt_m(st['median_bubble']):>11s}")
    for name, g in metrics["gaps"].items():
        p(f"gap {name}: median {fmt_m(g['median'])} per-stripe {[fmt_m(x) for x in g['per_stripe']]}")
    a = metrics["assessment"]
    p(f"Longest service stage : {a.get('longest_service_stage')}")
    p(f"Longest elapsed stage : {a.get('longest_elapsed_stage')}")
    p(f"Output II             : {fmt_m(a.get('output_ii'))}")
    c = a.get("stage_most_consistent_with_output_ii")
    if c:
        p(f"Most consistent w/ II : {c} (median start-II {fmt_m(metrics['stages'][c]['median_start_ii'])}, "
          f"occupancy {metrics['stages'][c]['occupancy']:.0%})")
    p(f"Largest bubble stage  : {a.get('largest_bubble_stage')}")
    if a.get("largest_positive_gap"):
        p(f"Largest positive gap  : {a['largest_positive_gap']}")
    if a.get("dominant_downstream_wait"):
        p(f"Downstream wait       : {a['dominant_downstream_wait']}")
    p(f"Assessment            : {a['assessment']}")
    p("stage start causes (resource wait = start - latest explicit dependency ready):")
    for s in inv.stripes:
        parts = []
        for k in metrics["stage_set"]:
            sw = s["stages"][k].get("start_wait")
            if not sw:
                continue
            who = sw.get("critical_service_label", "?")
            role = sw.get("critical_service_role")
            stripes = sw.get("critical_service_stripes")
            tag = f"{role}{'' if stripes is None else 's' + ','.join(map(str, stripes))}"
            parts.append(f"{k}: dep<-{who}[{tag} {sw.get('critical_milestone')}] "
                         f"res-wait {fmt_m(sw['resource_wait_cycles'])}")
        p(f"  s{s['stripe_index']}: " + " | ".join(parts))


def plot_pipeline(inv, metrics, stage_set, out, args, model, caption_extra=""):
    stripes = inv.stripes
    n = len(stripes)
    origin = metrics["first_start_cycle"] if args.relative else 0
    scale = 1e6 if args.unit == "million-cycles" else 1.0
    unit = "million cycles" if scale == 1e6 else "cycles"
    x = lambda c: (float(c) - float(origin)) / scale  # noqa: E731

    extra_rows = []
    if inv.setup_stage:
        extra_rows.append(("invocation setup", inv.setup_stage, "SETUP"))
    if inv.tail_stage:
        extra_rows.append(("invocation tail", inv.tail_stage, "TAIL"))
    rows = n + len(extra_rows)

    fig_h = 1.6 + 0.62 * rows * (1.35 if len(stage_set) > 4 else 1.0)
    fig, ax = plt.subplots(figsize=(13, max(4.0, fig_h)))

    lo = min(float(s["start_cycle"]) for s in stripes)
    hi = max(float(s["completion_cycle"]) for s in stripes)
    for _, st, _ in extra_rows:
        lo = min(lo, float(st["start_cycle"]))
        hi = max(hi, float(st["end_cycle"]))
    span = (hi - lo) / scale
    min_label = 0.035 * span

    band = 0.84
    track = band / len(stage_set)
    yticks, ylabels = [], []

    def draw(y_top, k, st):
        y0 = y_top - track * (stage_set.index(k) + 1)
        color = STAGE_COLOR[k]
        a, b = x(st["start_cycle"]), x(st["end_cycle"])
        ax.broken_barh([(a, b - a)], (y0 + 0.01, track - 0.02), facecolors=color, alpha=0.30,
                       edgecolor=color, linewidth=0.6)
        merged = merge_intervals(st["_intervals"])
        ax.broken_barh([(x(p), x(q) - x(p)) for p, q in merged], (y0 + 0.01, track - 0.02),
                       facecolors=color, alpha=0.95)
        if b - a >= min_label:
            ax.text((a + b) / 2, y0 + track / 2, k, ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold")

    for i, s in enumerate(stripes):
        y_center = rows - 1 - i - len([r for r in extra_rows if r[2] == "SETUP"])
        y_top = y_center + band / 2
        yticks.append(y_center)
        ylabels.append(f"s{s['stripe_index']}")
        for k in stage_set:
            st = s["stages"][k]
            if st["status"] == "OBSERVED":
                draw(y_top, k, st)
            elif k == "SF":
                la = s["stages"]["LA"]
                xa = x(la["end_cycle"]) if la["status"] == "OBSERVED" else x(s["start_cycle"])
                y0 = y_top - track * (stage_set.index(k) + 1)
                ax.text(xa, y0 + track / 2, " SF: not observed", ha="left", va="center",
                        fontsize=6, color="#555555", style="italic")
        ax.axvline(x(s["completion_cycle"]), color="#888888", linestyle=":", linewidth=0.7)

    for label, st, key in extra_rows:
        y_center = rows - 1 if key == "SETUP" else 0
        yticks.append(y_center)
        ylabels.append(label)
        a, b = x(st["start_cycle"]), x(st["end_cycle"])
        ax.broken_barh([(a, b - a)], (y_center - 0.15, 0.3), facecolors=STAGE_COLOR[key], alpha=0.30,
                       edgecolor=STAGE_COLOR[key], linewidth=0.6)
        ax.broken_barh([(x(p), x(q) - x(p)) for p, q in merge_intervals(st["_intervals"])],
                       (y_center - 0.15, 0.3), facecolors=STAGE_COLOR[key], alpha=0.95)

    ax.set_yticks(yticks, labels=ylabels)
    ax.set_ylim(-0.6, rows - 0.4)
    pad = 0.02 * span
    ax.set_xlim(x(lo) - pad, x(hi) + pad)
    ax.set_xlabel(f"schedule-axis {unit}" + (" from first stripe start" if args.relative else ""))
    ax.grid(axis="x", alpha=0.25)

    handles = [Patch(facecolor=STAGE_COLOR[k], label=f"{k} [{STAGE_SIDE[k]}] {STAGE_TITLE[k]}"
                     + (" — not observed" if k == "SF" else "")) for k in stage_set]
    if extra_rows:
        handles += [Patch(facecolor=STAGE_COLOR[k], label=k.lower()) for _, _, k in extra_rows]
    handles.append(Patch(facecolor="#BBBBBB", alpha=0.3, label="pale = elapsed envelope"))
    handles.append(Patch(facecolor="#BBBBBB", label="solid = service union"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)

    a = metrics["assessment"]
    view = "expanded RC" if len(stage_set) > 4 else "stripe pipeline"
    ax.set_title(f"{model} · {inv.phase} · {inv.label()} · {view}", fontsize=10, loc="left")
    caption = (
        f"{n} stripes · makespan {fmt_m(metrics['pipeline_makespan'])} cycles · "
        f"median output II {fmt_m(metrics['median_output_ii'])} · "
        f"bottleneck candidate: {a.get('pipeline_limiting_candidate') or 'none'} — {a['assessment']}"
        + caption_extra
    )
    fig.text(0.01, 0.005, caption, fontsize=7, ha="left", va="bottom", wrap=True)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return [str(out.with_suffix(".png")), str(out.with_suffix(".pdf"))]


def public_stage(st):
    out = {k: v for k, v in st.items() if not k.startswith("_")}
    if "source_nodes" in out:
        out["source_node_count"] = len(out["source_nodes"])
        out["source_nodes"] = compress_host_ids(out["source_nodes"])
    return out


def public_stripe(rec):
    out = dict(rec)
    out["stages"] = {k: public_stage(v) for k, v in rec["stages"].items()}
    return out


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def validate(invs, plotted):
    checks = {}

    def check(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            raise AssertionError(f"validation failed: {name}: {detail}")

    multi = [i for i in plotted]
    check("1_stripe_operation_owner", all(s["operation_id"] == i.operation_id for i in multi for s in i.stripes))
    check("2_ws_main_gemm_provenance", all(
        s["stages"]["WS"].get("npu_category") == "MAIN_GEMM" for i in multi for s in i.stripes))
    check("3_rg_residual_gemm_provenance", all(
        s["stages"]["RG"].get("npu_category") == "RESIDUAL_GEMM"
        for i in multi for s in i.stripes if s["stages"]["RG"]["status"] == "OBSERVED"))
    unclassified = [p for i in invs for p in i.problems if "UNCLASSIFIED" in p]
    check("4_no_fallback_npu_classification", True,
          f"UNCLASSIFIED works kept out of stages: {len(unclassified)}")
    check("5_stripe_ids_unique", all(
        len({s["stripe_id"] for s in i.stripes}) == len(i.stripes) for i in invs))
    check("6_source_nodes_retained", all(
        st["status"] != "OBSERVED" or st["source_nodes"]
        for i in invs for s in i.stripes for st in s["stages"].values()))
    check("7_no_negative_interval", all(
        st["status"] != "OBSERVED" or st["end_cycle"] >= st["start_cycle"]
        for i in invs for s in i.stripes for st in s["stages"].values()))
    check("8_service_within_envelope", all(
        st["status"] != "OBSERVED" or st["service_union_cycles"] <= st["elapsed_cycles"]
        for i in invs for s in i.stripes for st in s["stages"].values()))
    check("9_sf_never_inferred", all(
        s["stages"]["SF"]["status"] == SF_NOT_OBSERVED and s["stages"]["SF"]["start_cycle"] is None
        for i in invs for s in i.stripes))
    check("10_single_stripe_not_plotted", all(len(i.stripes) > 1 for i in plotted))
    return checks


def stripes_main(args):
    inputs = Inputs(args.export)
    out_dir = args.out_dir or inputs.export_path.parent / "pipeline"
    out_dir = out_dir.resolve()
    for p in inputs.files().values():
        if p.resolve().parent == out_dir:
            raise SystemExit(f"output directory {out_dir} contains input {p}; refusing")
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    before = None
    if not args.no_input_hash:
        log("hashing inputs (before)...")
        before = inputs.hash_inputs()
        for name, expected in inputs.expected.items():
            if before[name] != expected:
                raise SystemExit(f"{name}: sha256 {before[name]} != export.json {expected}")

    invs = bind_invocations(inputs, args.phase, log)
    model = inputs.model_label()

    multi = [i for i in invs if len(i.stripes) > 1]
    single = [i for i in invs if len(i.stripes) == 1]
    complete = [i for i in multi if all(s["completeness_status"] == "COMPLETE_EXCEPT_SF" for s in i.stripes)]
    incomplete = [i for i in multi if i not in complete]

    if single:
        msg = DECODE_SINGLE_STRIPE if args.phase == "decode" else (
            "invocation has one activation stripe; no intra-GEMM stripe pipeline exists for this invocation")
        log(f"WARNING: {len(single)} single-stripe invocation(s): {msg}")
        if args.phase == "decode":
            log(f"WARNING: {SINGLE_STRIPE_DEGENERATE}")

    metrics = {i.operation_id: pipeline_metrics(i, PRIMARY_STAGES) for i in complete}
    metrics_x = {i.operation_id: pipeline_metrics(i, EXPANDED_STAGES) for i in complete}

    key = {
        "makespan": lambda i: metrics[i.operation_id]["pipeline_makespan"],
        "median-output-ii": lambda i: metrics[i.operation_id]["median_output_ii"] or 0,
    }[args.worst_metric]
    ranked = sorted(complete, key=key, reverse=True)

    selected = []
    if args.operation:
        found = [i for i in invs if i.operation_id == args.operation]
        if not found:
            raise SystemExit(f"operation {args.operation} not found in phase {args.phase}")
        selected += [("operation", i) for i in found]
    if args.layer:
        found = [i for i in invs if i.layer == args.layer]
        if not found:
            raise SystemExit(f"layer {args.layer} not found in phase {args.phase}")
        selected += [("layer", i) for i in found]
    if not (args.operation or args.layer):
        for mode in args.select:
            if not ranked:
                break
            if mode == "worst":
                selected.append(("worst", ranked[0]))
            elif mode == "median":
                by_key = sorted(complete, key=key)
                selected.append(("median", by_key[(len(by_key) - 1) // 2]))
            elif mode == "all":
                selected += [("all", i) for i in complete]

    figures = []
    plotted = []
    for tag, inv in selected:
        if len(inv.stripes) <= 1:
            log(f"WARNING: {inv.operation_id}: "
                + (SINGLE_STRIPE_DEGENERATE if inv.phase == "decode" else DECODE_SINGLE_STRIPE.replace("decode ", "")))
            continue
        if inv.operation_id not in metrics:
            log(f"WARNING: {inv.operation_id}: incomplete stripe binding; not plotted")
            continue
        plotted.append(inv)
        name = tag if tag in ("worst", "median") else safe_name(inv.layer or inv.operation_id)
        print_bottleneck(inv, metrics[inv.operation_id])
        print_bottleneck(inv, metrics_x[inv.operation_id])
        figures += plot_pipeline(inv, metrics[inv.operation_id], PRIMARY_STAGES,
                                 out_dir / f"{args.phase}-{name}-pipeline", args, model)
        if not args.no_expanded:
            figures += plot_pipeline(inv, metrics_x[inv.operation_id], EXPANDED_STAGES,
                                     out_dir / f"{args.phase}-{name}-expanded", args, model)

    checks = validate(invs, plotted)

    # Machine-readable outputs.
    all_stripes = [public_stripe(s) for i in invs for s in i.stripes]
    (out_dir / "pipeline-stripes.json").write_text(json.dumps(all_stripes, indent=1) + "\n")
    write_stripes_csv(out_dir / "pipeline-stripes.csv", invs)
    write_invocations_csv(out_dir / "pipeline-invocations.csv", invs, metrics)
    families = write_families_csv(out_dir / "pipeline-families.csv", complete, metrics)

    after = None
    if not args.no_input_hash:
        log("hashing inputs (after)...")
        after = inputs.hash_inputs()
        if after != before:
            raise AssertionError("input artifacts changed during analysis")
        checks["11_inputs_byte_identical"] = {"pass": True, "detail": "sha256 before == after"}

    summary = {
        "schema": "potal-stripe-pipeline-summary",
        "version": 1,
        "model": model,
        "phase": args.phase,
        "inputs": {name: {"path": str(p), "sha256_before": (before or {}).get(name),
                          "sha256_after": (after or {}).get(name),
                          "sha256_export_json": inputs.expected.get(name)}
                   for name, p in inputs.files().items()},
        "axis": {"frequency_hz": inputs.frequency_hz, "unit": "schedule-axis cycles"},
        "binding": {
            "ws": "npu:<work_id> with provenance dense_main/stripe; services request_sha256 == run_view_sha256",
            "rg": "npu:<work_id> with provenance residual/residual_compact; shares the stripe LA host stage",
            "la": "WS required_host_stage_ids unique to the stripe, each an explicit edge parent",
            "setup": "WS required host stages shared by more than one stripe",
            "rp": "RG required_host_stage_ids minus LA and setup",
            "mc": "CPU nodes reachable through explicit edges from exactly one stripe's roots",
            "tail": "CPU nodes reachable from roots of more than one stripe",
            "unbound": "CPU nodes of the invocation with no edge path to a stripe root (not plotted)",
            "sf": SF_NOT_OBSERVED,
            "sf_evidence": {
                "folding_stage_labels_in_execution_ir": sorted({l for i in invs for l in i.folding_labels}),
                "lifecycle_barriers": "owner:<op>:<stripe>:ACTIVATION_ROWS:COMMIT / RESIDUAL_PAYLOAD:SEAL are "
                                      "zero-duration scheduler milestones; retained per stripe as audit only",
            },
            "timing": "schedule.sqlite accepted_ns .. result_ready_ns (resource_ready kept for NPU)",
        },
        "coverage": {
            "gemm_invocations": len(invs),
            "multi_stripe_invocations": len(multi),
            "single_stripe_invocations": len(single),
            "single_stripe_operations": [i.label() for i in single],
            "complete_multi_stripe_invocations": len(complete),
            "incomplete_multi_stripe_invocations": [i.operation_id for i in incomplete],
            "stripes": sum(len(i.stripes) for i in invs),
            "stripes_in_multi_stripe_invocations": sum(len(i.stripes) for i in multi),
            "binding_problems": [p for i in invs for p in i.problems],
            "unbound_cpu_nodes": sum(len(i.unbound) for i in invs),
            "unbound_by_label": merge_label_summaries(i.unbound_summary for i in invs),
        },
        "selection_metric": args.worst_metric,
        "selected": [
            {"tag": tag, "operation_id": inv.operation_id, "layer": inv.layer,
             "label": inv.label(),
             "primary": strip_internal(metrics.get(inv.operation_id)),
             "expanded": strip_internal(metrics_x.get(inv.operation_id)),
             "invocation_setup": public_stage(inv.setup_stage) if inv.setup_stage else None,
             "invocation_setup_labels": inv.setup_labels,
             "invocation_tail": public_stage(inv.tail_stage) if inv.tail_stage else None,
             "invocation_tail_labels": inv.tail_labels,
             "unbound": inv.unbound_summary}
            for tag, inv in selected if inv in plotted
        ],
        "top_worst": [
            {"rank": n + 1, "operation_id": i.operation_id, "layer": i.layer,
             "pipeline_makespan": metrics[i.operation_id]["pipeline_makespan"],
             "median_output_ii": metrics[i.operation_id]["median_output_ii"],
             "pipeline_limiting_candidate":
                 metrics[i.operation_id]["assessment"].get("pipeline_limiting_candidate")}
            for n, i in enumerate(ranked[:args.top])
        ],
        "families": families,
        "figures": figures,
        "validation": checks,
    }
    (out_dir / "pipeline-summary.json").write_text(json.dumps(summary, indent=1, default=exact) + "\n")
    log(f"\nwrote {out_dir}")
    for f in figures:
        log(f"  {f}")


def strip_internal(m):
    return m


def merge_label_summaries(items):
    total = defaultdict(lambda: {"nodes": 0, "service_union_cycles_sum": 0})
    for summary in items:
        for label, v in summary.items():
            total[label]["nodes"] += v["nodes"]
            total[label]["service_union_cycles_sum"] += v["service_union_cycles"]
    return dict(sorted(total.items()))


def write_stripes_csv(path, invs):
    cols = ["operation_id", "layer", "recorded_layer", "phase", "decode_index", "stripe_index",
            "stripe_id", "run_id", "slot", "start_cycle", "completion_cycle", "completeness_status"]
    stage_cols = ["status", "start_cycle", "end_cycle", "elapsed_cycles", "service_union_cycles", "bubble_cycles"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + [f"{k}_{c}" for k in EXPANDED_STAGES + ["RC"] for c in stage_cols]
                   + ["WS_result_ready_cycle", "WS_resource_ready_cycle", "WS_work_id", "RG_work_id"])
        for i in invs:
            for s in i.stripes:
                row = [s[c] for c in cols]
                for k in EXPANDED_STAGES + ["RC"]:
                    st = s["stages"][k]
                    row += [st.get(c) for c in stage_cols]
                row += [s["stages"]["WS"].get("result_ready_cycle"), s["stages"]["WS"].get("resource_ready_cycle"),
                        s["stages"]["WS"].get("work_id"), s["stages"]["RG"].get("work_id")]
                w.writerow(row)


def dominant(stages, key):
    obs = {k: v for k, v in stages.items() if v["status"] == "OBSERVED" and v.get(key) is not None}
    return max(obs, key=lambda k: obs[k][key]) if obs else None


def write_invocations_csv(path, invs, metrics):
    cols = ["operation_id", "layer", "recorded_layer", "stripe_count", "pipeline_makespan", "median_output_ii"]
    cols += [f"{k}_median_elapsed" for k in PRIMARY_STAGES]
    cols += [f"{k}_median_service" for k in PRIMARY_STAGES]
    cols += ["dominant_elapsed_stage", "dominant_service_stage", "pipeline_limiting_candidate",
             "assessment", "completeness_status"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in invs:
            m = metrics.get(i.operation_id)
            if len(i.stripes) <= 1:
                status = "SINGLE_STRIPE_DEGENERATE"
            elif m is None:
                status = "INCOMPLETE:" + ";".join(sorted({s["completeness_status"] for s in i.stripes}))
            else:
                status = "COMPLETE_EXCEPT_SF"
            row = [i.operation_id, i.layer if i.layer_canonical else None, i.layer, len(i.stripes)]
            if m:
                st = m["stages"]
                row += [m["pipeline_makespan"], m["median_output_ii"]]
                row += [st[k].get("median_elapsed") for k in PRIMARY_STAGES]
                row += [st[k].get("median_service") for k in PRIMARY_STAGES]
                row += [dominant(st, "median_elapsed"), dominant(st, "median_service"),
                        m["assessment"].get("pipeline_limiting_candidate"), m["assessment"]["assessment"]]
            else:
                row += [None] * (2 + 2 * len(PRIMARY_STAGES) + 4)
            row.append(status)
            w.writerow(row)


def write_families_csv(path, complete, metrics):
    agg = defaultdict(lambda: defaultdict(int))
    for i in complete:
        fam = i.family
        if fam is None:
            continue
        a = agg[fam]
        m = metrics[i.operation_id]
        a["invocations"] += 1
        a["stripes"] += m["stripe_count"]
        a["pipeline_makespan_total"] += m["pipeline_makespan"]
        a["output_ii_total"] += sum(m["output_ii"])
        a["output_ii_count"] += len(m["output_ii"])
        for k in ("LA", "WS", "RC"):
            st = m["stages"][k]
            a[f"{k}_elapsed_total"] += st["total_elapsed"]
            a[f"{k}_service_total"] += st["total_service"]
            a[f"{k}_bubble_total"] += st["total_bubble"]
            a[f"{k}_occupancy_union_total"] += st["occupancy_union_cycles"]
        if m["assessment"].get("pipeline_limiting_candidate"):
            a["candidate_" + m["assessment"]["pipeline_limiting_candidate"]] += 1
    excluded = [i.label() for i in complete if i.family is None]
    keys = sorted({k for a in agg.values() for k in a})
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family"] + keys + ["mean_output_ii_from_totals", "RC_occupancy_from_totals"])
        for fam, a in sorted(agg.items()):
            ii = Fraction(a["output_ii_total"], a["output_ii_count"]) if a["output_ii_count"] else None
            occ = Fraction(exact_int(a["RC_occupancy_union_total"]), exact_int(a["pipeline_makespan_total"]))
            w.writerow([fam] + [exact(a[k]) if isinstance(a[k], Fraction) else a[k] for k in keys]
                       + [exact(ii) if ii is not None else None, round(float(occ), 6)])
    return {
        "families": {fam: {k: exact(v) if isinstance(v, Fraction) else v for k, v in a.items()}
                     for fam, a in sorted(agg.items())},
        "excluded_non_canonical_layer": excluded,
        "note": "ratios derived from integer cycle totals, not averaged ratios",
    }


def exact_int(v):
    v = Fraction(v)
    if v.denominator != 1:
        raise AssertionError(f"non-integral cycle total {v}")
    return v.numerator


def main():
    argv = sys.argv[1:]
    if argv and argv[0] not in ("token", "stripes", "-h", "--help") and argv[0].endswith(".jsonl"):
        argv = ["token"] + argv

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    tok = sub.add_parser("token", help="legacy token-wide occupancy (regression/debug)")
    tok.add_argument("timeline", type=Path)
    tok.add_argument("--token", type=int, default=59)
    tok.add_argument("--out", type=Path, default=Path("potal-timeline"))

    st = sub.add_parser("stripes", help="stripe-level pipelined stage timeline")
    st.add_argument("export", type=Path, help="timeline export.json naming the stored inputs")
    st.add_argument("--phase", choices=("prefill", "decode"), default="prefill")
    st.add_argument("--operation", help="exact execution-IR operation id")
    st.add_argument("--layer", help="recorded NPU layer name")
    st.add_argument("--select", action="append", choices=("worst", "median", "all"))
    st.add_argument("--worst-metric", choices=("makespan", "median-output-ii"), default="makespan")
    st.add_argument("--top", type=int, default=10)
    st.add_argument("--unit", choices=("cycles", "million-cycles"), default="million-cycles")
    st.add_argument("--relative", action="store_true", help="x axis relative to first stripe start")
    st.add_argument("--no-expanded", action="store_true")
    st.add_argument("--out-dir", type=Path)
    st.add_argument("--no-input-hash", action="store_true", help="skip before/after sha256 of inputs")

    args = ap.parse_args(argv)
    if args.command == "token":
        token_main(args)
    else:
        if not args.select:
            args.select = ["worst", "median"]
        stripes_main(args)


if __name__ == "__main__":
    main()
