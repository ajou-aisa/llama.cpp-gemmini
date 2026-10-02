#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B scripts/eval/cycle_campaign_bundle.py init|finalize ROOT
"""Create and seal an evaluation-cycle-campaign-<timestamp> bundle from on-disk evidence only."""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from campaign_build import snapshot
from campaign_inputs import checksums
from eval_common import (
    EvaluationError,
    Record,
    read_json,
    record,
    require,
    sha256,
    write_json,
)

SECTIONS: Final = ("baseline", "trace", "certificate", "cycle", "scu", "manifest", "regression")
MODELS: Final = ("GPT-2-124M", "Llama-3.2-1B")
SMOKE: Final = "GPT-2-124M/A8W8/DIM16"
EXCLUDED: Final = ["cycle != latency(ms)", "TTFT/TPOT", "Jetson", "CUDA comparison", "FPGA timing",
                   "post-route frequency", "PPL"]


def configurations() -> list[str]:
    return [f"{model}/{precision}/DIM{dim}" for model in MODELS for precision in ("A8W8", "A4W4")
            for dim in (16, 32, 64)]


def init(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for name in SECTIONS:
        if name != "baseline":
            (root / name).mkdir()
    snapshot(root / "baseline")
    write_json(root / "campaign.json", {"schema": "im2p-evaluation-cycle-campaign-v1",
               "created_at": datetime.now(timezone.utc).isoformat(),
               "configurations": [name for name in configurations()], "smoke": SMOKE})


def optional(path: Path) -> Record | None:
    return read_json(path) if path.is_file() else None


def configuration(root: Path, name: str) -> Record:
    capture = optional(root / "trace/evaluation-cycle" / name / "capture-status.json")
    verified = optional(root / "certificate" / name / "verification.json")
    rejected = optional(root / "certificate" / name / "rejection.json")
    cycle_dir = root / "cycle" / name
    results = optional(cycle_dir / "cycle-results.json")
    failure = optional(cycle_dir / "failure.json")
    manifest = root / "manifest" / name / "evaluation_manifest.json"
    row: Record = {"configuration": name, "capture": "PASS" if capture else "NOT_RUN",
                   "certificate": "VERIFIED" if verified else "REJECTED" if rejected else "NOT_RUN",
                   "cycle": "COMPLETE" if results else "FAILED" if failure else "NOT_RUN",
                   "manifest_sha256": sha256(manifest) if manifest.is_file() else None}
    if rejected:
        row["certificate_rejection"] = rejected
    if capture:
        row["trace_sha256"] = capture["trace_sha256"]
        row["trace_content_sha256"] = capture["trace_content_sha256"]
    if results:
        scu = cycle_dir / "scu-cycle-accounting.json"
        target = root / "scu" / name
        target.mkdir(parents=True, exist_ok=True)
        if not (target / scu.name).exists():
            shutil.copyfile(scu, target / scu.name)
        require(sha256(target / scu.name) == sha256(scu), "SCU accounting copy differs: " + name)
        require(results.get("manifest_sha256") == row["manifest_sha256"], "cycle/manifest binding differs: " + name)
        row.update({"cycle_results": {"path": str(cycle_dir / "cycle-results.json"),
                                      "sha256": sha256(cycle_dir / "cycle-results.json")},
                    "certificate_sha256": results["certificate_sha256"], "work_count": results["work_count"],
                    "window_parity": results["window_parity"], "totals": results["totals"], "scu": results["scu"],
                    "state_domain_revision": results["state_domain_revision"]})
    elif failure:
        row["cycle_failure"] = failure
    return row


def metric_links(root: Path) -> list[Record]:
    links: list[Record] = []
    for manifest in sorted((root / "regression/metric-smoke").glob("*/*/*/*/evaluation_manifest.json")):
        kind, *parts = manifest.parent.relative_to(root / "regression/metric-smoke").parts
        name = "/".join(parts)
        shared = root / "manifest" / name / "evaluation_manifest.json"
        links.append({"kind": kind, "configuration": name, "manifest_sha256": sha256(manifest),
                      "same_manifest_as_cycle": shared.is_file() and sha256(shared) == sha256(manifest)})
    return links


def finalize(root: Path) -> None:
    require(not (root / "final.json").exists() and not (root / "SHA256SUMS").exists(), "bundle already sealed")
    rows = [configuration(root, name) for name in configurations()]
    regression = read_json(root / "regression/results.json")
    checks = record(regression["checks"])
    regression_pass = all(record(value).get("exit_code") == 0 for value in checks.values())
    smoke = next(row for row in rows if row["configuration"] == SMOKE)
    links = metric_links(root)
    smoke_links = {str(link["kind"]) for link in links if link["configuration"] == SMOKE and link["same_manifest_as_cycle"]}
    captured = [row for row in rows if row["capture"] == "PASS"]
    trace_ready = smoke["capture"] == "PASS" and smoke["certificate"] == "VERIFIED"
    campaign_ready = (trace_ready and smoke["cycle"] == "COMPLETE" and smoke.get("window_parity") == "EXACT"
                      and regression_pass and smoke_links >= {"activation", "residual", "scu"})
    scu = record(smoke.get("scu") or {})
    scu_ready = campaign_ready and all(isinstance(scu.get(key), int) for key in (
        "scu_active_cycles", "scu_idle_cycles", "scale_request_count", "scale_response_count", "scale_release_count"))
    document: Record = {
        "schema": "im2p-evaluation-cycle-campaign-final-v1", "sealed_at": datetime.now(timezone.utc).isoformat(),
        "EVALUATION_CYCLE_TRACE_CAPTURE_READY": "READY" if trace_ready else "NOT_READY",
        "EVALUATION_CYCLE_CAMPAIGN_READY": "READY" if campaign_ready else "NOT_READY",
        "SCU_CYCLE_ACCOUNTING_READY": "READY" if scu_ready else "NOT_READY",
        "E2E_RECONSTRUCTION_READY": "NOT_READY", "PAPER_CAMPAIGN_COMPLETE": "NOT_RUN",
        "configurations": [row for row in rows],
        "coverage": {"configurations": len(rows), "captured": len(captured),
                     "certified": sum(row["certificate"] == "VERIFIED" for row in rows),
                     "cycle_complete": sum(row["cycle"] == "COMPLETE" for row in rows)},
        "metric_links": [link for link in links], "regression": regression,
        "baseline": read_json(root / "baseline/repositories.json"),
        "excluded_claims": [claim for claim in EXCLUDED], "cycle_count_is_not_latency_ms": True}
    write_json(root / "final.json", document)
    checksums(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "finalize"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        init(args.root.resolve()) if args.command == "init" else finalize(args.root.resolve(strict=True))
        print(args.root)
        return 0
    except (EvaluationError, OSError, ValueError, KeyError, StopIteration) as error:
        print(f"campaign bundle failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
