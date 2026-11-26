"""Admission of hash-bound target host and interface measurements."""
from __future__ import annotations

from pathlib import Path
from typing import Final

from eval_common import Json, Record, integer, record, require, sha256, text

TARGET_HOST_SCHEMA: Final = "potal-target-host-timing"
TARGET_INTERFACE_SCHEMA: Final = "potal-target-interface-cost"
WORKLOAD_KEYS: Final = ("model_sha256", "input_tokens_sha256", "generated_tokens_sha256", "profile")
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
