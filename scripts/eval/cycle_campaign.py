#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
# How to run: scripts/eval/run_cycle_campaign.sh --trace TRACE --certificate CERTIFICATE --output DIRECTORY
"""Verify a cycle trace certificate, admit it into one stateful session and account cycles; not latency."""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Final

from campaign_build import REPO
from campaign_inputs import checksums
from eval_common import (
    EvaluationError,
    Json,
    Record,
    decode,
    integer,
    read_json,
    record,
    records,
    require,
    sha256,
    text,
    write_json,
)

TOTAL_KEYS: Final = ("dense_cycles", "residual_cycles", "scu_cycles", "load_cycles", "store_cycles",
                     "scale_cycles", "scu_active_cycles", "scu_idle_cycles", "scu_window_cycles",
                     "scale_request_count", "scale_response_count", "scale_lane_count", "scale_release_count",
                     "service_cycles", "resource_ready_cycles", "total_cycles")
EVALUATION_PARENT_SCHEMAS: Final = ("stateful-full374-replay-v1", "im2p-actual-trace-certificate-v1")
WINDOW_KEYS: Final = ("offered_cycle", "accepted_cycle", "result_ready_cycle", "final_scale_release_cycle",
                      "resource_ready_cycle")
COUNTER_PAIRS: Final = (("scale_request_count", "scale_request_count"),
                        ("scale_response_count", "scale_response_count"),
                        ("load_request_event_count", "load_request_count"),
                        ("load_response_event_count", "load_response_count"),
                        ("store_request_event_count", "store_request_count"),
                        ("store_completion_event_count", "store_response_count"))


def trace_works(trace: Path) -> list[Record]:
    rows: list[Record] = []
    with (gzip.open(trace, "rt", encoding="utf-8") if trace.suffix == ".gz" else trace.open(encoding="utf-8")) as stream:
        for line in stream:
            if '"NPU_WORK"' in line:
                row = decode(line)
                if row.get("kind") == "NPU_WORK":
                    rows.append({key: row[key] for key in ("work_id", "layer", "provenance", "phase_id", "operation_id")})
    return rows


def reference_windows(path: Path) -> dict[int, Record]:
    return {integer(row, "work_id"): {key: row[key] for key in WINDOW_KEYS} for row in records(path)}


def replay_windows(report: Path) -> dict[int, Record]:
    document = read_json(report)
    bound = record(document["records"])
    path = Path(text(bound, "path"))
    require(document.get("status") == "PASS" and sha256(path) == bound.get("sha256"),
            "native replay reference is not a complete unchanged replay")
    return {integer(row, "work_id"): {key: record(row["window"])[key] for key in WINDOW_KEYS} for row in records(path)}


def cycle_manifest(trace: Path, document: Record, evaluation: bool) -> Record | None:
    path = trace.parent / "cycle_manifest.json"
    if not path.is_file():
        require(not evaluation, "--evaluation requires cycle_manifest.json beside the trace")
        return None
    value = read_json(path)
    require(value.get("trace_sha256") == sha256(trace) and value.get("producer_sha256") == document["producer_sha256"]
            and all(value.get(key) == document[key] for key in ("model", "precision", "dim", "BK")),
            "cycle manifest differs from trace or certificate")
    return {"path": str(path), "sha256": sha256(path)}


def run(args: argparse.Namespace) -> Path:
    im2p = args.im2p.resolve(strict=True)
    sys.path.insert(0, str(im2p))
    from campaign_cycle import COUNTERS, CampaignProvider
    from cycle_accounting import DEFINITIONS, EventAccounting
    from sim.cycle.cycle_trace_certificate import context_from, verify_certificate

    trace, certificate = args.trace.resolve(strict=True), args.certificate.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    document = verify_certificate(certificate)
    bound = record(document["trace"])
    require(bound.get("path") == str(trace) and bound.get("sha256") == sha256(trace),
            "--trace is not the trace bound by --certificate")
    manifest_path = Path(text(record(document["evaluation_manifest"]), "path"))
    manifest_sha = sha256(manifest_path)
    manifest = read_json(manifest_path)
    require(manifest_sha == record(document["evaluation_manifest"]).get("sha256") and
            all(manifest.get(key) == document[key] for key in ("model", "precision", "dim", "BK")),
            "certificate-bound evaluation manifest changed")
    certificate_sha = sha256(certificate)
    parent_schema = record(document["parent_certificate"]).get("schema")
    require(not args.evaluation or parent_schema in EVALUATION_PARENT_SCHEMAS,
            "synthetic producer certificates are regression-only; evaluation cycles need an actual-inference parent")
    bound_cycle_manifest = cycle_manifest(trace, document, args.evaluation)
    write_json(output / "certificate-verification.json", {
        "status": "VERIFIED", "certificate": {"path": str(certificate), "sha256": certificate_sha},
        "state_domain_revision": document["state_domain_revision"], "equivalence": document["equivalence"],
        "parent_certificate": document["parent_certificate"], "manifest_sha256": manifest_sha,
        "cycle_manifest": bound_cycle_manifest, "actual_inference_parent": parent_schema in EVALUATION_PARENT_SCHEMAS})
    context = context_from(record(document["evidence_context"]))
    library = context.shared_library
    identity: Record = {"model": document["model"], "precision": document["precision"], "dim": document["dim"],
                        "BK": document["BK"], "trace_sha256": sha256(trace), "certificate_sha256": certificate_sha,
                        "manifest_sha256": manifest_sha, "unit": "cycles", "cycle_count_is_not_latency_ms": True}
    metadata = trace_works(trace)
    require(args.reference_per_work is None or args.reference_replay is None, "choose one window reference")
    reference_path = args.reference_replay or args.reference_per_work
    reference = (None if reference_path is None else replay_windows(reference_path.resolve(strict=True))
                 if args.reference_replay is not None else reference_windows(reference_path.resolve(strict=True)))
    accounting = EventAccounting.for_library(library)
    totals: Record = {key: 0 for key in TOTAL_KEYS}
    works: list[Record] = []
    mismatches: list[Json] = []
    with CampaignProvider(library, trace, certificate, context, event_sink=accounting.sink) as provider:
        admission = provider.admission
        require(admission.production_admitted and len(provider.requests) == len(metadata) and
                admission.profile == document["profile"], "complete production trace admission required")
        require([integer(row, "work_id") for row in metadata] == [item[0] for item in admission.work_bindings],
                "trace metadata order differs from admitted work order")
        with (output / "per-work-cycle.jsonl").open("x", encoding="utf-8") as stream:
            for request, meta in zip(provider.requests, metadata, strict=True):
                before = provider.counters()
                window = provider.execute(request, provider.native_cursor)
                after = provider.counters()
                work_id = integer(meta, "work_id")
                events = accounting.finish(work_id, window.accepted_cycle, window.resource_ready_cycle)
                delta: Record = {key: integer(after, key) - integer(before, key) for key in COUNTERS}
                for event_key, counter_key in COUNTER_PAIRS:
                    require(events[event_key] == delta[counter_key],
                            f"work {work_id}: {event_key} differs from native {counter_key}")
                service = window.result_ready_cycle - window.accepted_cycle
                row: Record = {
                    "work_id": work_id, **{key: meta[key] for key in ("layer", "provenance", "phase_id", "operation_id")},
                    "offered_cycle": window.offered_cycle, "accepted_cycle": window.accepted_cycle,
                    "result_ready_cycle": window.result_ready_cycle,
                    "final_scale_release_cycle": window.final_scale_release_cycle,
                    "resource_ready_cycle": window.resource_ready_cycle, "total_cycles": service,
                    "service_cycles": service, "resource_ready_cycles": window.resource_ready_cycle - window.offered_cycle,
                    "tag_peak": window.max_tag_occupancy, "row_peak": window.max_row_occupancy,
                    "dense_cycles": service if meta["provenance"] == "dense_main" else 0,
                    "residual_cycles": service if meta["provenance"] == "residual" else 0,
                    **events, "scu_cycles": events["scu_active_cycles"], "native_counters": delta}
                if reference is not None and reference.get(work_id) != {key: row[key] for key in WINDOW_KEYS}:
                    mismatches.append(work_id)
                for key in TOTAL_KEYS:
                    totals[key] = integer(totals, key) + integer(row, key)
                works.append(row)
                stream.write(json.dumps({**identity, **row}, sort_keys=True) + "\n")
        provider.verify_complete()
        final_cursor = provider.native_cursor
        state_revision = admission.scoped.state_domain_revision
    require(totals["resource_ready_cycles"] == final_cursor, "resource cycle conservation failed")
    require(integer(totals, "dense_cycles") + integer(totals, "residual_cycles") == totals["service_cycles"],
            "dense/residual cycle accounting does not conserve service cycles")
    require(integer(totals, "scu_active_cycles") + integer(totals, "scu_idle_cycles") == totals["scu_window_cycles"],
            "SCU active/idle accounting does not conserve its windows")
    peaks: Record = {"tag_peak": max(integer(row, "tag_peak") for row in works),
                     "row_peak": max(integer(row, "row_peak") for row in works)}
    parity: Record = {"reference": None if reference_path is None else
                      {"path": str(reference_path.resolve()), "sha256": sha256(reference_path.resolve()),
                       "kind": "native_replay_records" if args.reference_replay is not None else "per_work_cycle"},
                      "compared_fields": [key for key in WINDOW_KEYS], "works": len(works),
                      "mismatched_work_ids": mismatches,
                      "status": "NOT_RUN" if reference is None else "EXACT" if not mismatches and
                      set(reference) == {integer(row, "work_id") for row in works} else "MISMATCH"}
    write_json(output / "window-parity.json", {**identity, **parity})
    require(parity["status"] != "MISMATCH", "event-observed windows differ from reference replay")
    scu: Record = {**identity, **{key: totals[key] for key in (
        "scu_active_cycles", "scu_idle_cycles", "scu_window_cycles", "scale_request_count",
        "scale_response_count", "scale_lane_count", "scale_release_count", "scale_cycles")},
        "definitions": {key: DEFINITIONS[key] for key in ("scu_active_cycles", "scu_idle_cycles", "scale_cycles")},
        "source_events": [name for name in ("ScaleRequest", "ScaleResponse", "ScaleLane", "ScaleRelease")],
        "scope": "SCU execution timing from native events; independent of SCU scale-alignment metrics",
        "native_event_count": accounting.total_events}
    write_json(output / "scu-cycle-accounting.json", scu)
    results: Record = {
        **identity, "schema": "im2p-evaluation-cycle-results-v1", "profile": document["profile"],
        "state_domain_revision": state_revision, "production_admitted": True, "complete": True,
        "work_count": len(works), "final_resource_cursor": final_cursor,
        "cycle_library": {"path": str(library), "sha256": sha256(library)},
        "workload_policy": "CERTIFIED_CYCLE_PREFILL_256_PLUS_1 (see producer-manifest.json)",
        "cycle_manifest": bound_cycle_manifest, "certificate_parent_schema": parent_schema,
        "actual_inference_parent": parent_schema in EVALUATION_PARENT_SCHEMAS,
        "totals": {key: totals[key] for key in ("dense_cycles", "residual_cycles", "scu_cycles", "load_cycles",
                                                "store_cycles", "scale_cycles", "service_cycles",
                                                "resource_ready_cycles")},
        "scu": {key: totals[key] for key in ("scu_active_cycles", "scu_idle_cycles", "scale_request_count",
                                             "scale_response_count", "scale_release_count")},
        "peaks": peaks, "peak_scope": "native generation-cumulative prefix peak after each work",
        "definitions": {key: value for key, value in DEFINITIONS.items()}, "window_parity": parity["status"],
        "works": [row for row in works], "E2E_RECONSTRUCTION_READY": "NOT_READY", "PAPER_CAMPAIGN_COMPLETE": "NOT_RUN"}
    write_json(output / "cycle-results.json", results)
    checksums(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-per-work", type=Path,
                        help="earlier per-work-cycle.jsonl whose windows must match exactly")
    parser.add_argument("--reference-replay", type=Path,
                        help="independent native replay report.json whose per-work windows must match exactly")
    parser.add_argument("--evaluation", action="store_true",
                        help="paper evaluation: actual-inference parent and matching cycle_manifest.json required")
    parser.add_argument("--im2p", type=Path, default=REPO.parent / "IM2P.sim")
    args = parser.parse_args()
    try:
        print(run(args))
        return 0
    except (EvaluationError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"cycle campaign failed: {error}", file=sys.stderr)
        if args.output.is_dir() and not (args.output / "cycle-results.json").exists():
            write_json(args.output / "failure.json", {"error": str(error),
                       "EVALUATION_CYCLE_CAMPAIGN": "NOT_READY", "cycle_results_published": False})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
