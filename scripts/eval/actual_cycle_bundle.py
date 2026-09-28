#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B scripts/eval/actual_cycle_bundle.py init ROOT
"""Create an actual-cycle-campaign-<timestamp> bundle for actual-inference trace certification."""
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

SECTIONS: Final = ("traces", "certificates", "cycles", "scu", "manifests", "regression")
FIRST_TARGET: Final = "gpt2/a8w8/dim32"
EXCLUDED: Final = ("cycle != latency(ms)", "TTFT/TPOT", "Jetson", "CUDA comparison", "FPGA programming",
                   "synthesis", "post-route frequency", "PPL")
ORDER: Final = ("gpt2/a8w8/dim16", "gpt2/a8w8/dim32", "gpt2/a8w8/dim64",
                "gpt2/a4w4/dim16", "gpt2/a4w4/dim32", "gpt2/a4w4/dim64")


def init(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for name in SECTIONS:
        (root / name).mkdir()
    snapshot(root / "baseline")
    write_json(root / "campaign.json", {"schema": "im2p-actual-cycle-campaign-v1",
               "created_at": datetime.now(timezone.utc).isoformat(), "order": [name for name in ORDER],
               "first_target": "gpt2/a8w8/dim32", "recipe": "evaluation"})


def optional(path: Path) -> Record | None:
    return read_json(path) if path.is_file() else None


def configuration(root: Path, name: str) -> Record:
    status = optional(root / "traces" / name / "producer/capture-status.json")
    verified = optional(root / "certificates" / name / "verification.json")
    rejected = optional(root / "certificates" / name / "rejection.json")
    actual = optional(root / "certificates" / name / "actual_trace_certificate.json")
    cycle_dir = root / "cycles" / name
    results = optional(cycle_dir / "cycle-results.json")
    row: Record = {"configuration": name, "capture": "PASS" if status else "NOT_RUN",
                   "certificate": "VERIFIED" if verified else "REJECTED" if rejected else
                   "PENDING_INDEPENDENT_CERTIFICATION" if status else "NOT_RUN",
                   "actual_trace_certificate": None if actual is None else {
                       "path": str(root / "certificates" / name / "actual_trace_certificate.json"),
                       "sha256": sha256(root / "certificates" / name / "actual_trace_certificate.json"),
                       "state_domain_revision": actual["state_domain_revision"],
                       "synthetic_corpus": actual["synthetic_corpus"]},
                   "cycle": "COMPLETE" if results else "FAILED" if (cycle_dir / "failure.json").is_file() else "NOT_RUN"}
    if status:
        row.update({key: status[key] for key in ("trace_sha256", "trace_content_sha256", "cycle_manifest_sha256",
                                                 "recipe_sha256", "manifest_sha256")})
    if results:
        target = root / "scu" / name
        target.mkdir(parents=True, exist_ok=True)
        if not (target / "scu-cycle-accounting.json").exists():
            shutil.copyfile(cycle_dir / "scu-cycle-accounting.json", target / "scu-cycle-accounting.json")
        require(sha256(target / "scu-cycle-accounting.json") == sha256(cycle_dir / "scu-cycle-accounting.json"),
                "SCU accounting copy differs: " + name)
        require(results.get("manifest_sha256") == row.get("manifest_sha256"), "cycle/manifest binding differs: " + name)
        row.update({"cycle_results": {"path": str(cycle_dir / "cycle-results.json"),
                                      "sha256": sha256(cycle_dir / "cycle-results.json")},
                    "certificate_sha256": results["certificate_sha256"], "work_count": results["work_count"],
                    "final_resource_cursor": results["final_resource_cursor"],
                    "window_parity": results["window_parity"], "totals": results["totals"], "scu": results["scu"],
                    "peaks": results["peaks"], "state_domain_revision": results["state_domain_revision"],
                    "certificate_parent_schema": results["certificate_parent_schema"],
                    "actual_inference_parent": results["actual_inference_parent"],
                    "cycle_manifest": results["cycle_manifest"]})
    return row


def evaluated(row: Record) -> bool:
    scu = row.get("scu")
    return (row["cycle"] == "COMPLETE" and row.get("window_parity") == "EXACT" and
            row.get("actual_inference_parent") is True and row.get("cycle_manifest") is not None and
            isinstance(scu, dict) and all(isinstance(scu.get(key), int) for key in (
                "scu_active_cycles", "scu_idle_cycles", "scale_request_count", "scale_response_count",
                "scale_release_count")))


def finalize(root: Path) -> None:
    require(not (root / "final.json").exists() and not (root / "SHA256SUMS").exists(), "bundle already sealed")
    rows = [configuration(root, name) for name in ORDER]
    regression = read_json(root / "regression/results.json")
    checks = record(regression["checks"])
    regression_pass = all(record(value).get("exit_code") == 0 for value in checks.values())
    target = next(row for row in rows if row["configuration"] == FIRST_TARGET)
    actual = target.get("actual_trace_certificate")
    certified = (isinstance(actual, dict) and actual.get("synthetic_corpus") is False and
                 target["certificate"] == "VERIFIED" and target.get("certificate_parent_schema") ==
                 "im2p-actual-trace-certificate-v1" and regression_pass)
    document: Record = {
        "schema": "im2p-actual-cycle-campaign-final-v1", "sealed_at": datetime.now(timezone.utc).isoformat(),
        "ACTUAL_TRACE_CERTIFICATION_CAMPAIGN_READY": "READY" if certified else "NOT_READY",
        "GPT2_CYCLE_EVALUATION_READY": "READY" if certified and evaluated(target) else "NOT_READY",
        "E2E_RECONSTRUCTION_READY": "NOT_READY", "PAPER_CAMPAIGN_COMPLETE": "NOT_RUN",
        "first_target": FIRST_TARGET, "configurations": [row for row in rows],
        "evaluated_configurations": [str(row["configuration"]) for row in rows if evaluated(row)],
        "regression": regression, "baseline": read_json(root / "baseline/repositories.json"),
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
    except (EvaluationError, OSError, ValueError) as error:
        print(f"actual campaign bundle failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
