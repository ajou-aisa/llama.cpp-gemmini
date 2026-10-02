"""Fail-closed publication admission for target E2E latency.

A validated reconstruction proves that a schedule follows from certified inputs; it is not target latency.
Publication additionally needs a validated operating clock, admitted target-host timing measured on the host
whose CPU services the reconstruction used, admitted target-interface cost for every declared UNMODELED
interface stage, and one workload identity shared by all of them. Missing or invalid evidence leaves the
publication NOT_READY with explicit codes; development-host timing and zero cost are never substituted.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from e2e_cost_ownership import contract, unmodeled_required
from eval_common import Json, Record, integer, read_json, record, require, sha256, text

TARGET_HOST_SCHEMA: Final = "potal-target-host-timing"
TARGET_INTERFACE_SCHEMA: Final = "potal-target-interface-cost"
WORKLOAD_KEYS: Final = ("model_sha256", "input_tokens_sha256", "generated_tokens_sha256", "profile")
GATES: Final = ("clock_ready", "target_host_ready", "target_interface_ready", "workload_identity_ready",
                "reconstruction_ready")
DEVELOPMENT_SYSTEMS: Final = frozenset({"Darwin"})


def bound(value: Json, name: str) -> Path:
    reference = record(value)
    path = Path(text(reference, "path"))
    require(path.is_absolute() and path.is_file() and sha256(path) == text(reference, "sha256"),
            name + " artifact binding mismatch")
    return path


def workload_mismatch(document: Record, workload: Record) -> list[str]:
    binding = record(document.get("workload"))
    return [key for key in WORKLOAD_KEYS if binding.get(key) is None or binding.get(key) != workload.get(key)]


def admit_target_host(document: Record, workload: Record, collection_hosts: set[str]) -> Record:
    """Target timing must come from a non-development host that also produced the reconstructed CPU services."""
    require(document.get("schema") == TARGET_HOST_SCHEMA and document.get("version") == 1,
            "invalid target-host artifact: unsupported schema")
    target = record(document.get("target"))
    host_id = text(target, "host_id")
    require(all(bool(text(target, key)) for key in ("system", "machine", "board", "cpu_model")),
            "invalid target-host artifact: target description incomplete")
    require(target.get("system") not in DEVELOPMENT_SYSTEMS,
            "invalid target-host artifact: development-host timing is not target timing")
    measurement = record(document.get("measurement"))
    require(measurement.get("kind") == "TARGET_MEASURED" and measurement.get("unit") in ("cycle", "ns") and
            bool(text(measurement, "counter")), "invalid target-host artifact: target measurement required")
    clock = record(document.get("clock"))
    require(integer(clock, "cpu_frequency_hz", 1) > 0 and bool(text(clock, "provenance")),
            "invalid target-host artifact: target CPU clock provenance required")
    require(integer(document, "cpu_service_count", 1) > 0, "invalid target-host artifact: no CPU services")
    bound(document.get("evidence"), "target-host evidence")
    mismatch = workload_mismatch(document, workload)
    require(not mismatch, "target-host workload mismatch: " + ",".join(mismatch))
    require(collection_hosts == {host_id},
            "invalid target-host artifact: reconstructed CPU services were not collected on the admitted target host")
    return {"host_id": host_id, "board": target["board"], "cpu_frequency_hz": clock["cpu_frequency_hz"]}


def admit_target_interface(document: Record, workload: Record, stages: list[str]) -> Record:
    """Every declared UNMODELED interface stage needs a positive modeled or measured cost with provenance."""
    require(document.get("schema") == TARGET_INTERFACE_SCHEMA and document.get("version") == 1,
            "invalid target-interface artifact: unsupported schema")
    status = document.get("status")
    require(status != "UNMODELED", "target interface UNMODELED")
    require(status in ("MODELED", "MEASURED"), "invalid target-interface artifact: status")
    rows = record(document.get("stages"))
    missing = [stage for stage in stages if stage not in rows]
    require(not missing, "invalid target-interface artifact: uncovered stages " + ",".join(missing))
    for stage in stages:
        row = record(rows[stage])
        require(integer(row, "cycles", 1) > 0 and bool(text(row, "provenance")),
                "invalid target-interface artifact: stage cost must be positive with provenance: " + stage)
    bound(document.get("evidence"), "target-interface evidence")
    mismatch = workload_mismatch(document, workload)
    require(not mismatch, "target-interface workload mismatch: " + ",".join(mismatch))
    return {"status": status, "stages": list(stages)}


def clock_frequency(path: Path, profile: str, im2p: Path) -> int:
    if str(im2p) not in sys.path:
        sys.path.insert(0, str(im2p))
    from scripts.evaluation_clock import load_selection
    return load_selection(path, profile).frequency_hz


def publication_readiness(inputs: Record, workload: Record, collection_hosts: set[str],
                          reconstruction_ready: bool, im2p: Path) -> Record:
    """Evaluate every publication gate; never raises for missing or invalid evidence, only reports it."""
    stages = unmodeled_required(contract())
    codes: list[Json] = []
    gates = {name: False for name in GATES}
    frequency: int | None = None
    if inputs.get("clock_selection") is None:
        codes.append("NOT_READY_MISSING_OPERATING_CLOCK")
    else:
        try:
            frequency = clock_frequency(bound(inputs.get("clock_selection"), "clock"), text(workload, "profile"), im2p)
            gates["clock_ready"] = True
        except (ValueError, OSError, KeyError) as error:
            codes.append("NOT_READY_INVALID_OPERATING_CLOCK: " + str(error))
    covered: list[str] = []
    for gate, name, missing_code in (("target_host_ready", "target_host_timing", "NOT_READY_MISSING_TARGET_HOST_ADMISSION"),
                                     ("target_interface_ready", "target_interface_cost",
                                      "NOT_READY_MISSING_TARGET_INTERFACE_COST")):
        if inputs.get(name) is None:
            codes.append(missing_code)
            continue
        try:
            document = read_json(bound(inputs.get(name), name))
            if gate == "target_host_ready":
                admit_target_host(document, workload, collection_hosts)
            else:
                admit_target_interface(document, workload, stages)
                covered = list(stages)
            gates[gate] = True
        except (ValueError, OSError) as error:
            reason = str(error)
            codes.append(("NOT_READY_TARGET_INTERFACE_UNMODELED" if "UNMODELED" in reason else
                          "NOT_READY_WORKLOAD_IDENTITY_MISMATCH" if "workload mismatch" in reason else
                          "NOT_READY_INVALID_" + name.upper()) + ": " + reason)
    gates["workload_identity_ready"] = (all(workload.get(key) for key in WORKLOAD_KEYS) and
                                        not any(str(code).startswith("NOT_READY_WORKLOAD_IDENTITY_MISMATCH")
                                                for code in codes))
    if not gates["workload_identity_ready"] and not any(str(code).startswith("NOT_READY_WORKLOAD") for code in codes):
        codes.append("NOT_READY_WORKLOAD_IDENTITY_INCOMPLETE")
    gates["reconstruction_ready"] = reconstruction_ready
    if not reconstruction_ready:
        codes.append("NOT_READY_RECONSTRUCTION_NOT_VALIDATED")
    uncovered: list[Json] = [stage for stage in stages if stage not in covered]
    result: Record = {"schema": "potal-e2e-publication-readiness", "version": 1, **gates, "codes": codes,
                      "operating_clock_hz": frequency, "unmodeled_target_cost_count": len(uncovered),
                      "unmodeled_target_cost_stage_ids": uncovered,
                      "VALIDATED_RECONSTRUCTION": reconstruction_ready,
                      "TARGET_LATENCY_READY": all(gates.values())}
    return result
