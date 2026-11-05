#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "polars", "pydantic", "matplotlib"]
# ///
# How to run: imported by sensitivity-experiment.py; tests use test-sensitivity.py
"""Paired local sensitivity, cost budgets, and correlation estimates."""
from collections import defaultdict
from importlib import import_module
import math
from pathlib import Path
import re
from typing import Literal, TypedDict
import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

queue = import_module("cuda-ppl-queue")


class Measurement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    sse: float = Field(ge=0)
    max_abs_error: float = Field(ge=0)
    output_energy: float = Field(ge=0)
    reference_dot: float
    main_fragments: int = Field(ge=0)
    residual_fragments: int = Field(ge=0)
    saturations: int = Field(ge=0)
    seconds: float = Field(ge=0)


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_name: Literal["potal-sensitivity-v1"] = Field(alias="schema")
    key: str
    node: str
    bits: Literal[4, 8]
    call: int = Field(ge=0)
    selected_anchor: Literal[2, 3]
    logical_macs: int = Field(gt=0)
    samples: int = Field(gt=0)
    reference_energy: float = Field(ge=0)
    reference: Literal["double_dot_stored_weight_same_input"]
    top2: Measurement = Field(alias="2")
    top3: Measurement = Field(alias="3")


class Row(TypedDict):
    key: str
    operation: str
    layer: int
    chunks: int
    nmse_top2: float
    nmse_top3: float
    nmse_gain: float
    gain_ci_low: float
    gain_ci_high: float
    macs_top2: float
    macs_top3: float
    extra_macs: float
    extra_macs_percent: float
    residual_fraction_top2: float
    residual_fraction_top3: float
    cosine_top2: float
    cosine_top3: float
    max_abs_error_top2: float
    max_abs_error_top3: float
    gain_per_mmac: float
    seconds_top2: float
    seconds_top3: float


def operation(key: str) -> str:
    """Group exact policy keys by the actual multiplication's role."""
    if key.startswith("qk:"): return "QK"
    if key.startswith("pv:"): return "PV"
    return key.split(".")[-2] if key.startswith("blk.") else "lm_head"


def interval(values: list[float]) -> tuple[float, float]:
    """Paired chunk bootstrap; chunks, not individual tokens, are the resampling unit."""
    random = np.random.default_rng(42)
    samples = random.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def aggregate(path: Path, chunks: int, layers: int) -> list[Row]:
    """Require complete, unique coverage of every layer/operation before ranking."""
    grouped: dict[str, list[Sample]] = defaultdict(list)
    with path.open() as stream:
        for line in stream:
            sample = Sample.model_validate_json(line)
            grouped[sample.key].append(sample)
    queue.require(len(grouped) == 9*layers+1, "Incomplete sensitivity operation coverage")
    rows: list[Row] = []
    for key, samples in sorted(grouped.items()):
        queue.require(sorted(s.call for s in samples) == list(range(chunks)), "Missing/duplicate calibration calls")
        queue.require(all(s.top2.saturations == s.top3.saturations == 0 for s in samples), "SCU saturation")
        queue.require(all(s.top2.main_fragments == s.top3.main_fragments for s in samples), "Main work changed")
        energy = sum(s.reference_energy for s in samples)
        e2 = sum(s.top2.sse for s in samples)/max(energy, 1e-300)
        e3 = sum(s.top3.sse for s in samples)/max(energy, 1e-300)
        gain = e2-e3
        # Each bootstrap chunk contributes its share to the same aggregate energy denominator.
        lo, hi = interval([(s.top2.sse-s.top3.sse)*chunks/max(energy,1e-300) for s in samples])
        def cost(anchor: int) -> float:
            return 32*sum((s.top2 if anchor == 2 else s.top3).main_fragments+
                          (s.top2 if anchor == 2 else s.top3).residual_fragments for s in samples)/chunks
        def residual(anchor: int) -> float:
            return 32*sum((s.top2 if anchor == 2 else s.top3).residual_fragments for s in samples)/chunks
        def cosine(anchor: int) -> float:
            measurements = [s.top2 if anchor == 2 else s.top3 for s in samples]
            denominator = math.sqrt(energy*sum(m.output_energy for m in measurements))
            return sum(m.reference_dot for m in measurements)/denominator if denominator else 1.0
        c2, c3 = cost(2), cost(3)
        queue.require(c2 > 0 and c3 > 0, "No executed CUDA fragments")
        layer_match = re.search(r"^blk\.(\d+)\.|-(\d+)(?:\D.*)?$", key)
        layer = int(next(g for g in layer_match.groups() if g is not None)) if layer_match else -1
        if operation(key) == "PV":
            queue.require(e2 == e3 and c2 == c3, "PV anchor control unexpectedly changed")
        rows.append(dict(key=key, operation=operation(key), layer=layer, chunks=chunks,
                         nmse_top2=e2, nmse_top3=e3, nmse_gain=gain, gain_ci_low=lo, gain_ci_high=hi,
                         macs_top2=c2, macs_top3=c3, extra_macs=c3-c2, extra_macs_percent=100*(c3/c2-1),
                         residual_fraction_top2=residual(2)/c2, residual_fraction_top3=residual(3)/c3,
                         cosine_top2=cosine(2), cosine_top3=cosine(3),
                         max_abs_error_top2=max(s.top2.max_abs_error for s in samples),
                         max_abs_error_top3=max(s.top3.max_abs_error for s in samples),
                         gain_per_mmac=gain/max((c3-c2)/1e6,1e-12),
                         seconds_top2=sum(s.top2.seconds for s in samples)/chunks,
                         seconds_top3=sum(s.top3.seconds for s in samples)/chunks))
    for name in ("attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down","QK","PV"):
        queue.require(sorted(r["layer"] for r in rows if r["operation"] == name) == list(range(layers)),
                      "Missing layer/operation identity: "+name)
    queue.require(sum(r["operation"] == "lm_head" for r in rows) == 1,"Missing head")
    return rows


def correlations(rows: list[Row]) -> list[dict[str, str | int | float | None]]:
    """Report Pearson/Spearman within profiles and operation families; constants are undefined."""
    results: list[dict[str, str | int | float | None]] = []
    for group in ["all", *sorted({r["operation"] for r in rows})]:
        data = rows if group == "all" else [r for r in rows if r["operation"] == group]
        for x in ("extra_macs", "extra_macs_percent"):
            xs = [float(r[x]) for r in data]
            ys = [r["nmse_gain"] for r in data]
            valid = len(data) >= 3 and np.std(xs) > 0 and np.std(ys) > 0
            pearson = float(np.corrcoef(xs,ys)[0,1]) if valid else None
            rx = pl.Series(xs).rank(method="average").to_numpy()
            ry = pl.Series(ys).rank(method="average").to_numpy()
            spearman = float(np.corrcoef(rx,ry)[0,1]) if valid else None
            results.append(dict(group=group, x=x, y="local_nmse_gain", n=len(data),
                                pearson=pearson, spearman=spearman))
    return results


def policies(rows: list[Row], directory: Path) -> dict[str, Path]:
    """Greedy gain/cost budgets are screening candidates, not an optimality claim."""
    directory.mkdir(parents=True, exist_ok=True)
    # Require the paired improvement's confidence bound to be positive.
    beneficial = [r for r in rows if r["gain_ci_low"] > 0 and r["operation"] != "PV"]
    beneficial.sort(key=lambda r: (-r["gain_per_mmac"], r["key"]))
    budget = sum(max(0,r["extra_macs"]) for r in beneficial)
    paths: dict[str, Path] = {}
    for percent in (25, 50, 75, 100):
        remaining = budget*percent/100
        selected: list[str] = []
        for row in beneficial:
            cost = max(0,row["extra_macs"])
            if cost <= remaining:
                selected.append(row["key"]); remaining -= cost
        path = directory / f"mixed-{percent}.tsv"
        path.write_text("".join(f"{key}\t3\n" for key in sorted(selected)))
        paths[f"mixed-{percent}"] = path
    ranked = sorted((r for r in rows if r["operation"] != "PV"), key=lambda r: -r["nmse_gain"])
    indices = [0,1,2,3,len(ranked)//2,len(ranked)-1]
    selected_rows = [ranked[i] for i in indices]
    selected_rows += [max((r for r in rows if r["operation"] == "QK"),key=lambda r:r["nmse_gain"]),
                      next(r for r in rows if r["operation"] == "PV")]
    for index, row in enumerate({r["key"]:r for r in selected_rows}.values()):
        path = directory / f"single-{index:02}.tsv"
        path.write_text(f'{row["key"]}\t3\n')
        paths[f"single-{index:02}"] = path
    return paths


def export(rows: list[Row], directory: Path, title: str) -> None:
    """Persist table, statistical summaries, and a standalone scientific plot."""
    directory.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(directory / "sensitivity.csv")
    queue.write(directory / "correlations.json",correlations(rows))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for group, color in (("linear","#6f7782"),("QK","#1874b5"),("PV","#df8f2d")):
        data = [r for r in rows if (r["operation"] if r["operation"] in ("QK","PV") else "linear") == group]
        axes[0].scatter([r["extra_macs_percent"] for r in data], [r["nmse_gain"] for r in data],
                        label=group,alpha=.65,s=24,color=color)
    axes[0].axhline(0,color="black",lw=.6); axes[0].legend()
    axes[0].set(xlabel="Additional fragment-equivalent MACs (%)",ylabel="NMSE(TOP2) - NMSE(TOP3)",
                title="Local improvement versus compensation cost")
    best = sorted(rows,key=lambda r:r["nmse_gain"],reverse=True)[:12][::-1]
    axes[1].barh([r["key"] for r in best],[r["nmse_gain"] for r in best],color="#1874b5")
    axes[1].set(xlabel="NMSE reduction (same inputs, stored weights)",title="Largest local improvements")
    fig.suptitle(title)
    fig.savefig(directory / "sensitivity.png",dpi=160,bbox_inches="tight"); plt.close(fig)
