#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B scripts/eval/paired_capacity.py --library LIBIM2P_CYCLE_MODEL --out DIR [--runs-root runs/metrics]
"""Paired-microtile capacity analysis (P0) over the offline residual-path metric datasets.

Each MAIN_STRIPE is joined to its COMPACT_WORK on (run_id, chunk_id, invocation_id, stripe_id). Main tiles come from
a port of gemmini_set_tile_ws (RISC-V-DynDNN-gemmini-include/gemmini.h:1870-1950) on compat contract values. A
paired loop is one Main K32 LoopMatmul loop (geometry.cpp:25-47) that also computes the stripe's residual rows for
the active K of that block at Main's J:

  ACC      (I_M + I_R) * J * D <= F * compat_acc_rows / 2      slot of an F-times physical accumulator
  work IDs (I_M + I_R) * J     <= ids_per_slot                 64 today (128 entries), 128 with 256 entries
  SP       I_R * max_b chunks_b * D <= SP/2 - (I_M + J) * max_k * D
  scale    2 * J               <= 128                          residual contexts use their own scale rows

with I_M = ceil(M/D), I_R = ceil(M_R/D), J = min(tile_J, ceil(N/D)), max_k = ceil(min(K, 32)/D),
chunks_b = ceil(popcount(mask_b) / min(D, 32)) for each active original K32 block b. Coverage is reported by
residual stripe count and by residual physical fragments.

Cycles (planner-blocks, sim/cycle/cli.py estimate(), compat hardware): Main and residual compact works are estimated
directly for sampled stripes. The cycle model rejects a dense loop beyond compat capacity, so the paired loop, one
dense loop of R_eq = M + M_R * sum_b(chunks_b / fpb) / blocks rows at Main's J (fpb = max(1, 32/D)), is the linear
fit T(i) = a + b*i of dense estimates over every capacity-feasible row-tile count i, evaluated at R_eq/D; when Main
fills the compat slot (one feasible i), the fit runs at J = 1 and scales Main's measured cycles by T(R_eq/D)/T(I_M). This is an
upper bound on the gain: it drops each residual loop's fixed cost and omits residual PRELOAD/LdR cost. lm_head (the
metrics-only terminal head) is excluded from cycle sampling: its residual exceeds the model's 10M-fragment cap.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Final

REPO: Final = Path(__file__).resolve().parents[2]
FACTORS: Final = (1, 2, 3, 4)
IDS_PER_SLOT: Final = (64, 128)
SCALE_ROWS_PER_SLOT: Final = 128
CYCLE_EXCLUDED_LAYERS: Final = ("lm_head",)
MODELS: Final = {"gpt2": "sweep-paper-gpt2/gpt2", "llama3.2-1B": "sweep-paper-llama/llama3.2-1B"}
PRECISIONS: Final = ("a4w4", "a8w8")
DIMS: Final = (16, 32, 64)
COLUMNS: Final = (
    "shard", "chunk_id", "invocation_id", "stripe_id", "layer", "M", "N", "K", "M_total", "tile_I", "tile_J",
    "tile_K", "I_M", "J", "M_R", "I_R", "compact_k", "original_k", "blocks_total", "blocks_active", "f_run",
    "f_K", "chunk_sum", "max_chunks", "ra_rows_max", "sp_free", "main_acc_rows", "r_acc_rows", "work_ids",
    "scale_rows", "main_fragments", "residual_fragments", "gather_bytes", "residual_w_bytes", "R_eq",
    *(f"groups_free_{factor}x" for factor in FACTORS))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"paired_capacity: {message}")


def ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def select_tiles(I: int, J: int, K: int, dim: int, bank_count: int, bank_rows: int,
                 acc_rows: int) -> tuple[int, int, int]:
    """gemmini_set_tile_ws for a matmul activation (gemmini.h:1870-1950, helpers :1757-1764)."""
    db_mats_in_partition = (bank_count * bank_rows // 2) // 2 // dim
    db_max_tile_i_j = math.isqrt((acc_rows // 2) // dim)
    db_max_tile_k = db_mats_in_partition // db_max_tile_i_j
    i_tiles, j_tiles, k_tiles = ceil_div(I, dim), ceil_div(J, dim), ceil_div(K, dim)
    max_spad_rows, max_acc_rows = bank_count * bank_rows // 2, acc_rows // 2
    tile_i, tile_j = min(i_tiles, db_max_tile_i_j), min(j_tiles, db_max_tile_i_j)
    tile_k = min(k_tiles, db_max_tile_k)

    def spad(i: int, j: int, k: int) -> int:
        return (i * k + k * j) * dim

    while True:
        increased = False
        if spad(tile_i, tile_j + 1, tile_k) <= max_spad_rows and tile_i * (tile_j + 1) * dim <= max_acc_rows \
                and tile_j + 1 <= j_tiles:
            tile_j += 1
            increased = True
        if spad(tile_i + 1, tile_j, tile_k) <= max_spad_rows and (tile_i + 1) * tile_j * dim <= max_acc_rows \
                and tile_i + 1 <= i_tiles:
            tile_i += 1
            increased = True
        if spad(tile_i, tile_j, tile_k + 1) <= max_spad_rows and tile_k + 1 <= k_tiles:
            tile_k += 1
            increased = True
        if not increased:
            return tile_i, tile_j, tile_k


def contract(im2p: Path, profile: str) -> dict[str, int]:
    document = json.loads((im2p / "config/gemmini_host_memory_contracts" / f"{profile}.json").read_text())
    return {key: int(document[key]) for key in ("dim", "activation_bits", "weight_bits", "bank_count",
                                                "bank_rows", "accumulator_rows")}


def stripe_metrics(main: dict[str, Any], work: dict[str, Any], m_total: int, hw: dict[str, int]) -> dict[str, Any]:
    dim = hw["dim"]
    M, N, K = main["m"], main["n"], main["k"]
    tile_i, tile_j, tile_k = select_tiles(m_total, N, K, dim, hw["bank_count"], hw["bank_rows"],
                                          hw["accumulator_rows"])
    i_m, j = ceil_div(M, dim), min(tile_j, ceil_div(N, dim))
    max_k = ceil_div(min(K, 32), dim)
    fpb = max(1, 32 // dim)
    half_acc, sp_half = hw["accumulator_rows"] // 2, hw["bank_count"] * hw["bank_rows"] // 2
    m_r, compact_k = int(work.get("m", 0)), int(work.get("k", 0))
    original_k = int(work.get("original_k", K))
    runs = work.get("runs") or []
    chunks = [ceil_div(bin(int(run["original_k_mask"])).count("1"), min(dim, 32)) for run in runs]
    require(len({run["original_block_id"] for run in runs}) == len(runs), "duplicate original block in runs")
    residual = m_r > 0 and compact_k > 0
    i_r = ceil_div(m_r, dim) if residual else 0
    blocks_total = ceil_div(original_k, 32)
    chunk_sum = sum(chunks) if residual else 0
    main_acc = i_m * j * dim
    weight_bytes = hw["weight_bits"] / 8
    row = {
        "M": M, "N": N, "K": K, "M_total": m_total, "tile_I": tile_i, "tile_J": tile_j, "tile_K": tile_k,
        "I_M": i_m, "J": j, "M_R": m_r, "I_R": i_r, "compact_k": compact_k, "original_k": original_k,
        "blocks_total": blocks_total, "blocks_active": len(runs) if residual else 0,
        "f_run": len(runs) / blocks_total if residual else 0.0,
        "f_K": compact_k / original_k if residual else 0.0,
        "chunk_sum": chunk_sum, "max_chunks": max(chunks, default=0) if residual else 0,
        "ra_rows_max": i_r * max(chunks, default=0) * dim if residual else 0,
        "sp_free": sp_half - (i_m + j) * max_k * dim,
        "main_acc_rows": main_acc, "r_acc_rows": i_r * j * dim, "work_ids": (i_m + i_r) * j,
        "scale_rows": 2 * j if residual else j,
        "main_fragments": int(main["physical_fragments"]),
        "residual_fragments": int(work.get("physical_fragments", 0)) if residual else 0,
        "gather_bytes": compact_k * N * 4 if residual else 0,
        "residual_w_bytes": int(compact_k * N * weight_bytes) if residual else 0,
        "R_eq": M + m_r * chunk_sum / (fpb * blocks_total) if residual else float(M),
    }
    for factor in FACTORS:
        row[f"groups_free_{factor}x"] = (factor * half_acc - main_acc) // (j * dim)
    return row


def acc_fits(row: dict[str, Any], factor: int, half_acc: int, dim: int) -> bool:
    return (row["I_M"] + row["I_R"]) * row["J"] * dim <= factor * half_acc


def parse_shard(job: tuple[str, str, dict[str, int]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    path, shard, hw = job
    mains: dict[tuple[Any, ...], dict[str, Any]] = {}
    works: dict[tuple[Any, ...], dict[str, Any]] = {}
    totals: dict[tuple[Any, ...], int] = defaultdict(int)
    kinds: Counter[str] = Counter()
    with open(path, "rb") as stream:
        for raw in stream:
            line = raw.decode()
            cut = line.find(',"row_map":')
            record = json.loads(line[:cut] + "}" if cut > 0 and line.rstrip().endswith("]}") else line)
            kind = record["kind"]
            kinds[kind] += 1
            if kind not in ("MAIN_STRIPE", "COMPACT_WORK"):
                continue
            key = (record["run_id"], record["chunk_id"], record["invocation_id"], record["stripe_id"])
            target = mains if kind == "MAIN_STRIPE" else works
            require(key not in target, f"duplicate {kind} {key} in {path}")
            if kind == "MAIN_STRIPE":
                record = {name: record[name] for name in ("layer", "row_begin", "row_count", "m", "n", "k",
                                                          "physical_fragments")}
                invocation = key[:3]
                totals[invocation] = max(totals[invocation], record["row_begin"] + record["row_count"])
            else:
                record = {name: record.get(name) for name in ("m", "n", "k", "original_k", "runs",
                                                              "physical_fragments")}
            target[key] = record
    kinds["UNMATCHED_COMPACT_WORK"] = len(works.keys() - mains.keys())
    rows = []
    for key, main in mains.items():
        work = works.get(key)
        if work is None:
            kinds["MAIN_WITHOUT_COMPACT_WORK"] += 1
            work = {}
        row = stripe_metrics(main, work, totals[key[:3]], hw)
        row.update(shard=shard, chunk_id=key[1], invocation_id=key[2], stripe_id=key[3], layer=main["layer"])
        if work.get("runs"):
            row["_runs"] = work["runs"]
        rows.append(row)
    return rows, kinds


def discover(runs_root: Path) -> list[dict[str, Any]]:
    datasets = []
    for model, relative in MODELS.items():
        for precision in PRECISIONS:
            for dim in DIMS:
                base = runs_root / relative / precision / f"d{dim}"
                passed = [log for log in sorted(base.glob("metrics-all*.campaign.log"))
                          if log.read_text().splitlines()[1:2] == ["exit: 0"]]
                require(len(passed) == 1, f"{base}: expected one successful attempt, found {len(passed)}")
                attempt = base / passed[0].name.removesuffix(".campaign.log")
                shards = sorted(attempt.glob("collection/shard-*/residual-path-metrics.jsonl"))
                require(bool(shards), f"{attempt}: no residual-path-metrics shards")
                datasets.append({"model": model, "profile": f"{precision}-d{dim}-hp1", "attempt": str(attempt),
                                 "shards": [str(path) for path in shards]})
    return datasets


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def distribution(values: list[float]) -> dict[str, float | None]:
    return {"p50": percentile(values, 0.5), "p99": percentile(values, 0.99), "max": max(values, default=None)}


def coverage(rows: list[dict[str, Any]], fits) -> dict[str, float | None]:
    residual = [row for row in rows if row["M_R"] > 0 and row["compact_k"] > 0]
    fragments = sum(row["residual_fragments"] for row in residual)
    fit = [row for row in residual if fits(row)]
    return {"stripes": len(fit) / len(residual) if residual else None,
            "fragments": sum(row["residual_fragments"] for row in fit) / fragments if fragments else None}


def summarize(rows: list[dict[str, Any]], hw: dict[str, int]) -> dict[str, Any]:
    dim, half_acc = hw["dim"], hw["accumulator_rows"] // 2
    residual = [row for row in rows if row["M_R"] > 0 and row["compact_k"] > 0]
    sp_ok = lambda row: row["ra_rows_max"] <= row["sp_free"]
    scale_ok = lambda row: row["scale_rows"] <= SCALE_ROWS_PER_SLOT
    histogram = Counter(min(9, int(row["f_run"] * 10)) for row in residual)
    return {
        "stripes": len(rows), "residual_stripes": len(residual),
        "main_fragments": sum(row["main_fragments"] for row in rows),
        "residual_fragments": sum(row["residual_fragments"] for row in residual),
        "acc_coverage": {f"{factor}x": coverage(rows, lambda row, f=factor: acc_fits(row, f, half_acc, dim))
                         for factor in FACTORS},
        "work_id_coverage": {f"{ids}_per_slot": coverage(rows, lambda row, n=ids: row["work_ids"] <= n)
                             for ids in IDS_PER_SLOT},
        "sp_coverage": coverage(rows, sp_ok), "scale_coverage": coverage(rows, scale_ok),
        "all_coverage": {f"{factor}x_{ids}_per_slot": coverage(
            rows, lambda row, f=factor, n=ids: acc_fits(row, f, half_acc, dim) and row["work_ids"] <= n
            and sp_ok(row) and scale_ok(row)) for factor in FACTORS for ids in IDS_PER_SLOT},
        "I_R": distribution([row["I_R"] for row in residual]),
        "work_ids": distribution([row["work_ids"] for row in residual]),
        "ra_rows_max": distribution([row["ra_rows_max"] for row in residual]),
        "sp_free": distribution([row["sp_free"] for row in residual]),
        "f_run": distribution([row["f_run"] for row in residual]),
        "f_K": distribution([row["f_K"] for row in residual]),
        "f_run_histogram": {f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}": histogram[bucket] for bucket in range(10)},
        "gather_bytes": sum(row["gather_bytes"] for row in residual),
        "residual_w_bytes": sum(row["residual_w_bytes"] for row in residual),
        "layers": dict(sorted(Counter(row["layer"].split(".", 2)[-1] for row in rows).items())),
    }


_LIBRARY: Path | None = None


def _init_worker(im2p: str, library: str) -> None:
    global _LIBRARY
    sys.path.insert(0, im2p)
    _LIBRARY = Path(library)


def estimate_cycles(job: tuple[str, dict[str, Any], dict[str, Any] | None]) -> int | str:
    from sim.cycle import cli
    profile, request, compact = job
    document: dict[str, Any] = {"profile": profile, "request": {**request, "submission": "planner-blocks"},
                                "limits": {"max_cycles": 1 << 40, "max_fragments": 10_000_000}}
    if compact is not None:
        document.update(compact)
    assert _LIBRARY is not None
    try:
        return int(cli.estimate(_LIBRARY, document)["result"]["total_cycles"])
    except ValueError as error:
        return f"rejected: {error}"


def feasible_row_tiles(row: dict[str, Any], hw: dict[str, int], j: int) -> list[int]:
    dim = hw["dim"]
    max_k = ceil_div(min(row["K"], 32), dim)
    sp_half, half_acc = hw["bank_count"] * hw["bank_rows"] // 2, hw["accumulator_rows"] // 2
    return [i for i in range(1, 65) if i * j <= 64 and i * j * dim <= half_acc and (i + j) * max_k * dim <= sp_half]


def fit_width(row: dict[str, Any], hw: dict[str, int]) -> int:
    """Main's J when it leaves two feasible dense row tiles; else J = 1, scaled to Main's measured cycles."""
    return row["tile_J"] if len(feasible_row_tiles(row, hw, row["J"])) >= 2 else 1


def linear_fit(points: list[tuple[int, int]]) -> tuple[float, float]:
    n = len(points)
    mean_x, mean_y = sum(x for x, _ in points) / n, sum(y for _, y in points) / n
    variance = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
    return mean_y - slope * mean_x, slope


def sample(rows: list[dict[str, Any]], per_layer: int, seed: str) -> list[dict[str, Any]]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["M_R"] > 0 and row["compact_k"] > 0:
            strata[row["layer"].split(".", 2)[-1]].append(row)
    chosen = []
    for layer, members in sorted(strata.items()):
        if layer in CYCLE_EXCLUDED_LAYERS:
            continue
        count = min(len(members), per_layer)
        chosen += random.Random(f"{seed}/{layer}").sample(members, count)
    return chosen


def cycles(rows: list[dict[str, Any]], hw: dict[str, int], profile: str, per_layer: int,
           pool: ProcessPoolExecutor) -> dict[str, Any]:
    dim = hw["dim"]
    picked = sample(rows, per_layer, profile)
    jobs: dict[tuple[Any, ...], tuple[str, dict[str, Any], dict[str, Any] | None]] = {}
    for row in picked:
        tiles = {"tile_i": row["tile_I"], "tile_j": row["tile_J"], "tile_k": row["tile_K"]}
        jobs[("main", row["M"], row["N"], row["K"], row["tile_I"], row["tile_J"], row["tile_K"])] = (
            profile, {"m": row["M"], "n": row["N"], "k": row["K"], **tiles}, None)
        width = fit_width(row, hw)
        for i in feasible_row_tiles(row, hw, row["J"] if width == row["tile_J"] else width):
            jobs[("dense", i, row["N"], row["K"], width, row["tile_K"])] = (
                profile, {"m": i * dim, "n": row["N"], "k": row["K"], "tile_i": i, "tile_j": width,
                          "tile_k": row["tile_K"]}, None)
        key = ("residual", row["shard"], row["chunk_id"], row["invocation_id"], row["stripe_id"])
        residual_tiles = select_tiles(row["M_R"], row["N"], row["compact_k"], dim, hw["bank_count"],
                                      hw["bank_rows"], hw["accumulator_rows"])
        jobs[key] = (profile, {"m": row["M_R"], "n": row["N"], "k": row["compact_k"],
                               "tile_i": residual_tiles[0], "tile_j": residual_tiles[1],
                               "tile_k": residual_tiles[2]},
                     {"original_k": row["original_k"], "runs": row["_runs"]})
    keys = list(jobs)
    started = time.time()
    results = dict(zip(keys, pool.map(estimate_cycles, [jobs[key] for key in keys])))
    samples, rejected = [], [str(key) + ": " + value for key, value in results.items() if isinstance(value, str)]
    for row in picked:
        main = results[("main", row["M"], row["N"], row["K"], row["tile_I"], row["tile_J"], row["tile_K"])]
        res = results[("residual", row["shard"], row["chunk_id"], row["invocation_id"], row["stripe_id"])]
        width = fit_width(row, hw)
        direct = width == row["tile_J"]
        points = [(i, results[("dense", i, row["N"], row["K"], width, row["tile_K"])])
                  for i in feasible_row_tiles(row, hw, row["J"] if direct else width)]
        points = [(i, value) for i, value in points if isinstance(value, int)]
        if not isinstance(main, int) or not isinstance(res, int) or len(points) < 2:
            continue
        a, b = linear_fit(points)
        fit_main = a + b * ceil_div(row["M"], dim)
        fit_paired = a + b * row["R_eq"] / dim
        paired = fit_paired if direct else main * fit_paired / fit_main
        samples.append({
            "layer": row["layer"].split(".", 2)[-1], "M": row["M"], "N": row["N"], "K": row["K"],
            "M_R": row["M_R"], "R_eq": row["R_eq"], "main_cycles": main, "residual_cycles": res,
            "residual_to_main": res / main, "fit_mode": "direct" if direct else "j1_scaled",
            "fit_points": points, "fit_main_relative_error": fit_main / main - 1 if direct else None,
            "paired_cycles_upper_bound_gain_model": paired, "paired_gain": 1 - paired / (main + res),
        })
    strata = Counter(row["layer"].split(".", 2)[-1] for row in rows if row["M_R"] > 0 and row["compact_k"] > 0)
    by_layer: dict[str, dict[str, float]] = {}
    for layer in sorted({sample_row["layer"] for sample_row in samples}):
        members = [sample_row for sample_row in samples if sample_row["layer"] == layer]
        by_layer[layer] = {key: sum(sample_row[key] for sample_row in members) / len(members)
                           for key in ("main_cycles", "residual_cycles", "paired_cycles_upper_bound_gain_model")}
    weight = lambda layer, key: strata[layer] * by_layer[layer][key]
    total_separate = sum(weight(layer, "main_cycles") + weight(layer, "residual_cycles") for layer in by_layer)
    total_main = sum(weight(layer, "main_cycles") for layer in by_layer)
    total_paired = sum(weight(layer, "paired_cycles_upper_bound_gain_model") for layer in by_layer)
    return {
        "sampled_stripes": len(samples), "estimates": len(keys), "seconds": round(time.time() - started, 1),
        "excluded_layers": list(CYCLE_EXCLUDED_LAYERS),
        "rejected_estimates": rejected,
        "weighted_residual_to_main": (total_separate - total_main) / total_main if total_main else None,
        "weighted_paired_gain_upper_bound": 1 - total_paired / total_separate if total_separate else None,
        "paired_gain": distribution([sample_row["paired_gain"] for sample_row in samples]),
        "residual_to_main": distribution([sample_row["residual_to_main"] for sample_row in samples]),
        "fit_main_relative_error": distribution([abs(sample_row["fit_main_relative_error"]) for sample_row in samples
                                                 if sample_row["fit_main_relative_error"] is not None]),
        "fit_modes": dict(Counter(sample_row["fit_mode"] for sample_row in samples)),
        "by_layer_mean": by_layer, "layer_weights": dict(strata), "samples": samples,
    }


def decide(summaries: list[dict[str, Any]], target: float) -> dict[str, Any]:
    decision: dict[str, Any] = {}
    for profile in sorted({item["profile"] for item in summaries}):
        members = [item for item in summaries if item["profile"] == profile]
        factor = next((f for f in FACTORS if all(
            (item["summary"]["acc_coverage"][f"{f}x"]["fragments"] or 0) >= target for item in members)), None)
        ids = next((n for n in IDS_PER_SLOT if all(
            (item["summary"]["work_id_coverage"][f"{n}_per_slot"]["fragments"] or 0) >= target
            for item in members)), None)
        decision[profile] = {
            "acc_factor": factor, "acc_factor_status": "pending V1 gate 2" if factor == 3 else
            ("none of 1x-4x reaches the target" if factor is None else "selected"),
            "work_entries": None if ids is None else 2 * ids,
            "sp_fits_all_models": all((item["summary"]["sp_coverage"]["fragments"] or 0) >= target for item in members),
            "scale_fits_all_models": all((item["summary"]["scale_coverage"]["fragments"] or 0) >= target
                                         for item in members),
            "per_model": {item["model"]: {
                "acc_coverage_fragments": {key: value["fragments"]
                                           for key, value in item["summary"]["acc_coverage"].items()},
                "work_id_coverage_fragments": {key: value["fragments"]
                                               for key, value in item["summary"]["work_id_coverage"].items()},
                "weighted_residual_to_main": item["cycles"]["weighted_residual_to_main"],
                "weighted_paired_gain_upper_bound": item["cycles"]["weighted_paired_gain_upper_bound"],
                "paired_gain": item["cycles"]["paired_gain"],
            } for item in members},
        }
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired-microtile capacity analysis (P0)")
    parser.add_argument("--runs-root", type=Path, default=REPO / "runs/metrics")
    parser.add_argument("--im2p", type=Path, default=REPO.parent / "IM2P.sim")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--per-layer", type=int, default=6, help="cycle-model samples per layer kind and dataset")
    parser.add_argument("--target", type=float, default=0.95, help="residual-fragment coverage target")
    arguments = parser.parse_args()
    out = arguments.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    datasets = discover(arguments.runs_root.resolve())
    summaries = []
    with ProcessPoolExecutor(arguments.workers) as parse_pool, ProcessPoolExecutor(
            arguments.workers, initializer=_init_worker,
            initargs=(str(arguments.im2p.resolve()), str(arguments.library.resolve()))) as cycle_pool:
        for dataset in datasets:
            started = time.time()
            hw = contract(arguments.im2p.resolve(), dataset["profile"])
            jobs = [(path, Path(path).parent.name, hw) for path in dataset["shards"]]
            rows: list[dict[str, Any]] = []
            kinds: Counter[str] = Counter()
            for shard_rows, shard_kinds in parse_pool.map(parse_shard, jobs):
                rows += shard_rows
                kinds += shard_kinds
            name = f"{dataset['model']}-{dataset['profile']}"
            with gzip.open(out / f"{name}.csv.gz", "wt", newline="") as stream:
                writer = csv.DictWriter(stream, COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            summary = summarize(rows, hw)
            cycle_report = cycles(rows, hw, dataset["profile"], arguments.per_layer, cycle_pool)
            document = {"schema": "im2p-paired-capacity-p0", "version": 1, **dataset,
                        "shard_sha256": {Path(path).parent.name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                         for path in dataset["shards"]},
                        "contract": hw, "record_counts": dict(kinds), "summary": summary, "cycles": cycle_report,
                        "parse_seconds": round(time.time() - started, 1)}
            (out / f"{name}.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
            summaries.append({"model": dataset["model"], "profile": dataset["profile"], "summary": summary,
                              "cycles": cycle_report})
            print(f"{name}: {summary['stripes']} stripes, {summary['residual_stripes']} residual, "
                  f"acc 2x={summary['acc_coverage']['2x']['fragments']}, "
                  f"gain<={cycle_report['weighted_paired_gain_upper_bound']}", flush=True)
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "profile", "stripes", "residual_stripes", "residual_fragments",
                         *(f"acc_{f}x_fragments" for f in FACTORS), *(f"acc_{f}x_stripes" for f in FACTORS),
                         *(f"ids_{n}_fragments" for n in IDS_PER_SLOT), "sp_fragments", "scale_fragments",
                         "I_R_p50", "I_R_p99", "I_R_max", "work_ids_p50", "work_ids_p99", "work_ids_max",
                         "ra_rows_p50", "ra_rows_p99", "ra_rows_max", "f_run_p50", "gather_bytes",
                         "residual_w_bytes", "residual_to_main", "paired_gain_upper_bound"])
        for item in summaries:
            s, c = item["summary"], item["cycles"]
            writer.writerow([item["model"], item["profile"], s["stripes"], s["residual_stripes"],
                             s["residual_fragments"],
                             *(s["acc_coverage"][f"{f}x"]["fragments"] for f in FACTORS),
                             *(s["acc_coverage"][f"{f}x"]["stripes"] for f in FACTORS),
                             *(s["work_id_coverage"][f"{n}_per_slot"]["fragments"] for n in IDS_PER_SLOT),
                             s["sp_coverage"]["fragments"], s["scale_coverage"]["fragments"],
                             *(s["I_R"][q] for q in ("p50", "p99", "max")),
                             *(s["work_ids"][q] for q in ("p50", "p99", "max")),
                             *(s["ra_rows_max"][q] for q in ("p50", "p99", "max")), s["f_run"]["p50"],
                             s["gather_bytes"], s["residual_w_bytes"], c["weighted_residual_to_main"],
                             c["weighted_paired_gain_upper_bound"]])
    decision = {"schema": "im2p-paired-capacity-decision", "version": 1, "target_fragment_coverage": arguments.target,
                "definitions": __doc__, "profiles": decide(summaries, arguments.target)}
    (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({profile: {key: value[key] for key in ("acc_factor", "acc_factor_status", "work_entries")}
                      for profile, value in decision["profiles"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
