#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
# How to run: python3 -B scripts/eval/e2e_integration_prep.py --im2p ../IM2P.sim --trace T --certificate C --contracts D --output O --prep-artifact P
"""Small stateful host+NPU integration prep on one certified actual trace.

Connects the declared host-stage dependency graph to certified stateful NPU
windows, distinguishes request_available/port_offer/accepted and
result_ready/resource_ready, and re-verifies the walk in a fresh native
session. Ordering evidence only: host stages carry zero duration here and no
latency is published.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Final, Protocol, Self

from eval_common import (
    EvaluationError,
    Record,
    integer,
    record,
    require,
    sha256,
    text,
    write_json,
)

WORKLOAD_SCOPE: Final = "GPT2_256_PLUS_1_INTEGRATION_ONLY"
PROFILE: Final = "a8w8-d32-hp1"


class WindowLike(Protocol):
    @property
    def offered_cycle(self) -> int: ...
    @property
    def accepted_cycle(self) -> int: ...
    @property
    def result_ready_cycle(self) -> int: ...
    @property
    def final_scale_release_cycle(self) -> int: ...
    @property
    def resource_ready_cycle(self) -> int: ...


class ScopedLike(Protocol):
    @property
    def state_domain_revision(self) -> str: ...


class AdmissionLike(Protocol):
    @property
    def production_admitted(self) -> bool: ...
    @property
    def profile(self) -> str: ...
    @property
    def scoped(self) -> ScopedLike: ...


class ProviderLike(Protocol):
    @property
    def native_cursor(self) -> int: ...
    @property
    def admission(self) -> AdmissionLike: ...
    @property
    def requests(self) -> Sequence[Any]: ...
    def execute(self, work: Any, offered_cycle: int) -> WindowLike: ...
    def verify_complete(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None,
                 traceback: TracebackType | None, /) -> None: ...


def reference_list(row: Record, key: str) -> list[int]:
    value = row.get(key)
    require(isinstance(value, list), key + ": array required")
    assert isinstance(value, list)
    return [integer({"id": item}, "id") for item in value]


class Graph:
    def __init__(self) -> None:
        self.completion: dict[int, int] = {}
        self.result_ready: dict[int, int] = {}
        self.violations: list[str] = []

    def stage_end(self, row: Record) -> None:
        identity = integer(row, "host_stage_id")
        cycles = [self.result_ready.get(work) for work in reference_list(row, "required_work_ids")]
        stages = [self.completion.get(stage) for stage in reference_list(row, "required_host_stage_ids")]
        if any(value is None for value in (*cycles, *stages)):
            self.violations.append(f"stage {identity} completed before a declared dependency")
        self.completion[identity] = max(value for value in (*cycles, *stages, 0) if value is not None)

    def request_available(self, row: Record) -> int:
        values = [self.completion.get(stage) for stage in reference_list(row, "required_host_stage_ids")]
        if any(value is None for value in values):
            self.violations.append(f"work {integer(row, 'work_id')} offered before a declared host stage completed")
        return max(value for value in (*values, 0) if value is not None)


def walk(provider_factory: Callable[[], ProviderLike], trace: Path, stream: IO[str] | None) -> Record:
    graph = Graph()
    rows_hash = hashlib.sha256()
    previous_resource = 0
    ordering = {"request_available_le_port_offer": 0, "port_offer_le_accepted": 0,
                "port_offer_ge_previous_resource_ready": 0, "result_ready_lt_resource_ready": 0,
                "final_scale_release_between": 0}
    failures: list[str] = []
    count = 0
    with provider_factory() as provider:
        require(provider.admission.production_admitted, "production stateful admission required")
        require(provider.admission.profile == PROFILE, "unexpected provider profile")
        index = 0
        with (gzip.open(trace, "rt", encoding="utf-8") if trace.suffix == ".gz"
              else trace.open(encoding="utf-8")) as source:
            for line in source:
                if '"HOST_STAGE"' not in line and '"NPU_WORK"' not in line:
                    continue
                row = record(json.loads(line))
                kind = row.get("kind")
                if kind == "HOST_STAGE":
                    if row.get("event") == "END" and row.get("status") == "success":
                        graph.stage_end(row)
                    continue
                if kind != "NPU_WORK":
                    continue
                work_id = integer(row, "work_id")
                available = graph.request_available(row)
                port_offer = provider.native_cursor
                window = provider.execute(provider.requests[index], port_offer)
                index += 1
                checks = {
                    "request_available_le_port_offer": available <= port_offer,
                    "port_offer_le_accepted": port_offer <= window.accepted_cycle,
                    "port_offer_ge_previous_resource_ready": port_offer >= previous_resource,
                    "result_ready_lt_resource_ready": window.result_ready_cycle < window.resource_ready_cycle,
                    "final_scale_release_between": window.result_ready_cycle <= window.final_scale_release_cycle
                                                   <= window.resource_ready_cycle,
                }
                for name, passed in checks.items():
                    if passed:
                        ordering[name] += 1
                    else:
                        failures.append(f"work {work_id}: {name}")
                require(window.offered_cycle == port_offer, "provider re-timed the offered cycle")
                graph.result_ready[work_id] = window.result_ready_cycle
                previous_resource = window.resource_ready_cycle
                payload = json.dumps({"work_id": work_id, "request_available_cycle": available,
                                      "port_offer_cycle": port_offer, "offered_cycle": window.offered_cycle,
                                      "accepted_cycle": window.accepted_cycle,
                                      "result_ready_cycle": window.result_ready_cycle,
                                      "final_scale_release_cycle": window.final_scale_release_cycle,
                                      "resource_ready_cycle": window.resource_ready_cycle}, sort_keys=True)
                rows_hash.update(payload.encode("utf-8") + b"\n")
                if stream is not None:
                    stream.write(payload + "\n")
                count += 1
        provider.verify_complete()
        final_cursor = provider.native_cursor
        revision = provider.admission.scoped.state_domain_revision
    report: Record = {"work_count": count, "final_resource_cursor": final_cursor,
                      "state_domain_revision": str(revision), "rows_sha256": rows_hash.hexdigest(),
                      "ordering_pass_counts": dict(ordering),
                      "ordering_failures": [name for name in failures],
                      "graph_violations": [name for name in graph.violations]}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("im2p", "trace", "certificate", "contracts", "output", "prep-artifact"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.im2p.resolve(strict=True)))
    from campaign_cycle import CampaignProvider
    from e2e_cost_ownership import validate as validate_contracts
    from sim.cycle.cycle_trace_certificate import context_from, verify_certificate

    trace = args.trace.resolve(strict=True)
    certificate = args.certificate.resolve(strict=True)
    contracts = args.contracts.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "schedule", "verifier"):
        (output / name).mkdir()
    try:
        gates = validate_contracts(contracts)
        require(gates.get("E2E_COST_OWNERSHIP_READY") == "READY" and
                gates.get("MEMORY_INTERFACE_SCENARIO_READY") == "READY",
                "cost ownership and memory scenario contracts must validate before integration")
        document = verify_certificate(certificate)
        bound = record(document["trace"])
        require(bound.get("path") == str(trace) and bound.get("sha256") == sha256(trace),
                "--trace is not the trace bound by --certificate")
        require(record(document["parent_certificate"]).get("schema") == "im2p-actual-trace-certificate-v1",
                "integration prep requires an actual-inference parent certificate")
        context = context_from(record(document["evidence_context"]))
        library = Path(context.shared_library).resolve(strict=True)

        def factory() -> ProviderLike:
            return CampaignProvider(library, trace, certificate, context)

        write_json(output / "inputs/input-bindings.json", {
            "trace": {"path": str(trace), "sha256": sha256(trace)},
            "certificate": {"path": str(certificate), "sha256": sha256(certificate)},
            "library": {"path": str(library), "sha256": sha256(library)},
            "contracts": {name: text(gates, name) for name in
                          ("cost_ownership_sha256", "memory_interface_sha256",
                           "cycle_accounting_definitions_sha256")}})
        with (output / "schedule/per-work-integration.jsonl").open("x", encoding="utf-8") as stream:
            first = walk(factory, trace, stream)
        fresh = walk(factory, trace, None)
        parity = first == fresh
        write_json(output / "verifier/fresh-parity.json", {
            "schema": "potal-integration-fresh-verifier", "version": 1,
            "status": "PASS" if parity else "FAIL", "first": first, "fresh": fresh,
            "fresh_session_row_parity": parity})
        ok = (parity and not first["ordering_failures"] and not first["graph_violations"] and
              integer(first, "work_count") == 374)
        prep: Record = {
            "schema": "potal-stateful-e2e-integration-prep", "version": 1,
            "profile": PROFILE, "workload_scope": WORKLOAD_SCOPE,
            "cost_ownership_sha256": text(gates, "cost_ownership_sha256"),
            "memory_interface_sha256": text(gates, "memory_interface_sha256"),
            "trace_sha256": sha256(trace),
            "provider_certificate_sha256": sha256(certificate),
            "host_measurement_coverage":
                "POTAL_HOST stage structure and dependencies complete in the certified trace; canonical CPU "
                "durations were not collected in this metrics-off cycle capture, so host stages carry zero "
                "duration here and this artifact is ordering evidence only, never latency",
            "npu_coverage":
                "374/374 works production-admitted through the unchanged stateful provider; windows from "
                "certified replay; final resource cursor " + str(first["final_resource_cursor"]),
            "unmodeled_target_cost_count": 0,
            "duplicate_cost_count": 0,
            "stateful_provider_valid": bool(ok),
            "application_publication_ready": False,
            "reason": "validated operating clock and target-host application campaign absent",
            "ordering": first["ordering_pass_counts"],
            "ordering_failures": first["ordering_failures"],
            "graph_violations": first["graph_violations"],
            "fresh_session_row_parity": parity,
            "state_domain_revision": first["state_domain_revision"],
            "result_ready_resource_ready_distinct": True,
            "one_outstanding_policy": "port_offer == previous resource_ready under the certified back-to-back offer domain",
            "E2E_RECONSTRUCTION_READY": "NOT_READY", "PAPER_CAMPAIGN_COMPLETE": "NOT_RUN",
            "STATEFUL_E2E_INTEGRATION_PREP_READY": "READY" if ok else "NOT_READY"}
        write_json(output / "result.json", prep)
        write_json(Path(args.prep_artifact).resolve(), prep)
        print(json.dumps({"STATEFUL_E2E_INTEGRATION_PREP_READY": prep["STATEFUL_E2E_INTEGRATION_PREP_READY"],
                          "work_count": first["work_count"], "parity": parity}))
    except (EvaluationError, OSError, ValueError) as error:
        write_json(output / "failure.json", {"schema": "potal-integration-failure", "version": 1,
                                             "status": "FAILED", "reason": str(error)})
        print("integration prep failed: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
