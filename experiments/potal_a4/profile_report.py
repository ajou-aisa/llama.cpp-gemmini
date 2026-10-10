from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


GROUPS = {
    "Q exponent/top + preliminary rounding": ("q_exponent_top", "q_preliminary_round"),
    "Q sigma selection": ("q_sigma_select",),
    "Q final rounding + folding": ("q_final_round", "q_fold_main"),
    "Q/P digit packet preparation": ("q_packet", "p_packet"),
    "K/V/P range + rounding": ("k_range_round_split", "v_range_round_split", "p_range_round_main"),
    "QK main, K low": ("qk_main_dot", "qk_main_merge"),
    "Q upper compensation, both K passes": ("qk_upper_dot", "qk_upper_merge", "qk_hi_upper_dot", "qk_hi_upper_merge"),
    "K high main": ("qk_hi_main_dot", "qk_hi_main_merge"),
    "PV main": ("pv_main_dot", "pv_main_merge"),
    "P upper compensation": ("pv_upper_dot", "pv_upper_merge"),
    "P zero point: colsum + broadcast": ("pv_colsum", "pv_zero_point"),
    "Softmax + accumulator/publication": ("softmax", "qk_allocate", "qk_publish", "pv_allocate", "pv_publish"),
}


def main() -> None:
    root = Path(sys.argv[1])
    lines = ["# Attention limb and CPU component measurements", "",
             "Original selective-linear+a4nks policy; GPT-2, first WikiText-2 test chunk at each context.",
             "Counts cover all 12 layers × 12 heads. Timing covers layers 0/5/11 × heads 0/6/11, one warmup and five measured replays per head.",
             "Times are measured NumPy int64 CPU component work, not Gemmini time or full model latency. QK and PV integer sums are checked against reconstructed float64 operands.",
             "Reference-only FP QK/mask, reconstruction, unpacking, capture, checks and Python driver overhead are excluded from the component denominator.", ""]
    csv_rows = []
    for context in (128, 512, 1024):
        data = json.loads((root / f"context{context}.json").read_text())
        records = data["records"]
        assert len(records) == 144 and data["hidden_bitwise_matches_baseline_chunk0"]
        lines += [f"## Context {context}", "", "| Operand | values | lane 1 nonzeros | lane 2 nonzeros | lane 1 rows | lane 2 rows | upper packet byte utilization |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        aggregates = {}
        details = []
        for role in ("q", "p"):
            values = sum(r[role]["values"] for r in records)
            digits = sum((Counter(r[role]["lane_nonzeros"]) for r in records), Counter())
            rows = sum((Counter(r[role]["lane_active_rows"]) for r in records), Counter())
            stripes = sum(r[role]["stripes"] for r in records)
            presence = sum((Counter(r[role]["lane_stripes"]) for r in records), Counter())
            histogram = sum((Counter(r[role]["nonzero_upper_digits_per_value"]) for r in records), Counter())
            padded = sum(r[role]["upper_bytes"] for r in records)
            main = sum(r[role]["main_fragments"] for r in records)
            upper = sum(r[role]["upper_fragments"] for r in records)
            aggregates[role] = (main, upper)
            total_rows = context * len(records)
            assert set(digits).issubset({"1", "2"})
            assert sum(histogram.values()) == values
            lines.append(f"| {role.upper()} | {values:,} | {100*digits['1']/values:.3f}% | {100*digits['2']/values:.3f}% | "
                         f"{100*rows['1']/total_rows:.2f}% | {100*rows['2']/total_rows:.2f}% | {100*sum(digits.values())/padded:.3f}% |")
            details.append(f"{role.upper()}: {stripes} stripes; lane 1/2 present in {presence['1']}/{presence['2']} stripes. "
                           f"Values with 0/1/2 nonzero upper digits: {100*histogram['0']/values:.3f}% / {100*histogram['1']/values:.3f}% / {100*histogram['2']/values:.3f}%.")
            csv_rows.append([context, role, values, digits["1"], digits["2"], rows["1"], rows["2"], main, upper, padded])
        k_values = sum(r["k_values"] for r in records)
        v_values = sum(r["v_values"] for r in records)
        k_low, k_high = (sum(r["k_pass_nonzeros"][lane] for r in records) for lane in (0, 1))
        v_nonzero = sum(r["v_nonzeros"] for r in records)
        lines += ["", *details,
                  f"K always executes two dense passes; low/high nonzero values: {100*k_low/k_values:.3f}% / {100*k_high/k_values:.3f}%.",
                  f"V executes one pass; nonzero values: {100*v_nonzero/v_values:.3f}%.", ""]
        selected_values = sum(r["selection"]["values"] for r in records)
        top = sum(r["selection"]["top"] for r in records)
        sigma = sum(r["selection"]["sigma"] for r in records)
        qm, qr = aggregates["q"]
        pm, pr = aggregates["p"]
        work = 2 * (qm + qr) + pm + pr
        lines += ["", f"Q top-exponent selection: {100*top/selected_values:.3f}%; sigma selection: {100*sigma/selected_values:.3f}%; combined: {100*(top+sigma)/selected_values:.3f}%.",
                  "P/K/V have no outlier-selection stage in the supplied policy.", "",
                  "| Fragment category (exclusive) | fragments | fraction of all QK+PV |", "|---|---:|---:|"]
        for name, count in (("QK low main", qm), ("Q upper / K low", qr), ("K high main", qm),
                            ("Q upper / K high", qr), ("PV main", pm), ("P upper", pr)):
            lines.append(f"| {name} | {count:,} | {100*count/work:.2f}% |")
        lines += ["", f"QK/main = {2*(qm+qr)/qm:.3f}; PV/main = {(pm+pr)/pm:.3f}; combined / two single-pass mains = {work/(qm+pm):.3f}.",
                  f"Upper-limb compensation alone = {100*(2*qr+pr)/work:.2f}% of fragments. Including the K-high main = {100*(qm+2*qr+pr)/work:.2f}%.", ""]
        timed = [r["timing"] for r in records if "timing" in r]
        assert len(timed) == 9
        assert all(check["bitwise_float64"] for t in timed for check in t["checks"].values())
        included = {key for keys in GROUPS.values() for key in keys}
        assert len(included) == sum(len(keys) for keys in GROUPS.values())
        excluded = set(data["excluded_timing_keys"])
        assert all(set(run) <= included | excluded for t in timed for run in t["timed_runs"])
        batch = np.asarray([[sum(t["timed_runs"][repeat].get(key, 0) for t in timed for key in keys) / 9e6
                             for keys in GROUPS.values()] for repeat in range(5)])
        medians = np.median(batch, axis=0)
        shares = np.median(100 * batch / batch.sum(axis=1)[:, None], axis=0)
        lines += ["| CPU component | median batch mean ms/head | fraction of timed components |",
                  "|---|---:|---:|"]
        for name, milliseconds, share in zip(GROUPS, medians, shares, strict=True):
            lines.append(f"| {name} | {milliseconds:.4f} | {share:.2f}% |")
        total = batch.sum(axis=1)
        selection = batch[:, :2].sum(axis=1)
        preparation = batch[:, :5].sum(axis=1)
        lines += ["", f"Component total mean/head: median {np.median(total):.4f} ms; repeat p10..p90 [{np.percentile(total,10):.4f}, {np.percentile(total,90):.4f}] ms.",
                  f"Inclusive Q selection: {np.median(selection):.4f} ms, {np.median(100*selection/total):.2f}% of components, {np.median(100*selection/preparation):.2f}% of preparation.",
                  f"Packet creation: {np.median(100*batch[:,3]/preparation):.2f}% of preparation.", ""]
    (root / "MEASUREMENTS.md").write_text("\n".join(lines) + "\n")
    with (root / "limbs.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["context", "role", "values", "lane1_nonzeros", "lane2_nonzeros", "lane1_rows", "lane2_rows", "main_fragments", "upper_fragments", "upper_packet_bytes"])
        writer.writerows(csv_rows)
    files = [*root.glob("context*.json"), *Path(__file__).parent.glob("profile_*.py")]
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root / "artifact-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(root / "MEASUREMENTS.md")


if __name__ == "__main__":
    main()
