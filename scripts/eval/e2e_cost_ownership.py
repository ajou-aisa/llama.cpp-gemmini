#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -B scripts/eval/e2e_cost_ownership.py build OUTPUT_DIR | validate OUTPUT_DIR
"""Machine-readable PoTal E2E cost-ownership, memory-interface and cycle-accounting contracts.

Every target-latency stage carries exactly one cost authority. Classes:
HOST_MEASURED, NPU_MODELED, EXCLUDED_WITH_REASON, UNMODELED, DIAGNOSTIC_ONLY.
NPU-internal loads/stores/scale traffic stay NPU_MODELED and are never re-added
as host transfers; blocking NPU-wait envelopes are never additive host work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

from eval_common import (
    EvaluationError,
    Record,
    read_json,
    record,
    require,
    sha256,
    text,
    write_json,
)

LLAMA: Final = Path(__file__).resolve().parents[2]
IM2P: Final = LLAMA.parent / "IM2P.sim"
SCENARIO: Final = "gpt2-a8w8-dim32-stripe-pipeline-256p1-greedy-seed1234"
CLASSES: Final = ("HOST_MEASURED", "NPU_MODELED", "EXCLUDED_WITH_REASON", "UNMODELED", "DIAGNOSTIC_ONLY")
ADDITIVE: Final = ("HOST_MEASURED", "NPU_MODELED")
NPU_TRANSFER_IDS: Final = ("npu.backing_loads", "npu.backing_stores", "npu.scale_fetch")
WAIT_ENVELOPES: Final = ("im2p.frontend_start_host_call", "im2p.fence_host_call", "im2p.stripe_submit_host_call",
                         "im2p.residual_simulator_host_call", "dense_backend_host_call", "pipeline_drain_and_join",
                         "exsia.stripe_ready_handoff", "exsia.stripe_total", "exsia.run_total",
                         "frontend.producer_capacity_wait")
TIMING_FIELDS: Final = ("backing_read_delay", "even_read_id_delay", "scale_read_extra_delay",
                        "backing_write_delay", "read_ready_period", "backing_cycle_offset")
DEFINITION_REVISIONS: Final = {
    "load_cycles": "LOAD_BACKING_READ_TRANSACTION_UNION_V1",
    "store_cycles": "STORE_BACKING_WRITE_TRANSACTION_UNION_V1",
    "scale_cycles": "SCALE_FETCH_TRANSACTION_UNION_V1",
    "scu_active_cycles": "SCALE_PATH_FETCH_LANE_RELEASE_UNION_V1",
    "scu_idle_cycles": "WINDOW_MINUS_SCALE_PATH_ACTIVITY_V1",
    "scu_cycles": "SCALE_PATH_FETCH_LANE_RELEASE_UNION_V1",
    "dense_cycles": "SERVICE_WINDOW_BY_PROVENANCE_V1",
    "residual_cycles": "SERVICE_WINDOW_BY_PROVENANCE_V1",
}
MISNOMERS: Final = {
    "load_cycles": "NOT all load-engine latency; only backing read transaction interval union",
    "scu_active_cycles": "NOT SCU ALU utilization; scale-path fetch/lane/release activity union",
    "scu_cycles": "alias of scu_active_cycles kept for backward compatibility",
}


def stage_rows(document: Record) -> list[Record]:
    value = document.get("stages")
    require(isinstance(value, list), "stage array required")
    assert isinstance(value, list)
    return [record(item) for item in value]


def string_rows(document: Record, key: str) -> list[str]:
    value = document.get(key)
    require(isinstance(value, list), key + ": array required")
    assert isinstance(value, list)
    rows: list[str] = []
    for item in value:
        require(isinstance(item, str), key + ": string entries required")
        assert isinstance(item, str)
        rows.append(item)
    return rows


def stage(stage_id: str, name: str, producer: str, location: str, execution: str, resource: str,
          measurement_source: str, unit: str, included: bool, overlap: str, boundary: str,
          reason: str | None, source_file: str | None) -> Record:
    return {"stage_id": stage_id, "name": name, "producer": producer, "source_location": location,
            "execution_class": execution, "resource": resource, "measurement_source": measurement_source,
            "measurement_unit": unit, "included_in_target_latency": included, "overlap_policy": overlap,
            "dependency_boundary": boundary, "reason": reason, "source_file": source_file}


def host(stage_id: str, name: str, location: str, boundary: str, source_file: str,
         producer: str = "llama.cpp-gemmini") -> Record:
    return stage(stage_id, name, producer, location, "HOST_MEASURED", "cpu",
                 "gemmini_cycle_record_v2 CANONICAL_ADDITIVE interval bound to the declared POTAL_HOST stage",
                 "host_cpu_interval", True, "SERIALIZED_WITH_DECLARED_DEPENDENCIES", boundary, None, source_file)


def npu(stage_id: str, name: str, measurement: str, boundary: str) -> Record:
    return stage(stage_id, name, "IM2P.sim", "sim/cycle/control_engine.cpp:Engine", "NPU_MODELED", "npu",
                 measurement, "npu_cycles", True, "INSIDE_ACCEPTED_TO_RESOURCE_READY_WINDOW", boundary,
                 None, "sim/cycle/control_engine.cpp")


def wait(stage_id: str, name: str, location: str, source_file: str,
         producer: str = "llama.cpp-gemmini") -> Record:
    return stage(stage_id, name, producer, location, "EXCLUDED_WITH_REASON", "none",
                 "cpu_service_exclusion npu_dependency_envelope (interval class WAIT)", "host_cpu_interval",
                 False, "NON_ADDITIVE_BLOCKING_INTERVAL",
                 "wall envelope containing NPU wait or another worker's emulation",
                 "blocking wall interval; NPU service inside it is owned by NPU_MODELED windows", source_file)


def stages() -> list[Record]:
    rows: list[Record] = [
        stage("weight.model_load", "one-time GGUF model load and Q8_0/HP1 native weight binding",
              "llama.cpp-gemmini", "ggml/src/ggml-gemmini/ggml-gemmini.cpp:weight buffer preparation",
              "EXCLUDED_WITH_REASON", "none", "outside traced RUN; no per-inference stage declared",
              "not_applicable", False, "ONE_TIME_SETUP", "before REQUEST_START",
              "evaluation latency is per-inference; model load is one-time setup by definition",
              "ggml/src/ggml-gemmini/ggml-gemmini.cpp"),
        host("activation.exsia_local", "per-stripe activation outlier selection, quantization, packing, theta fold",
             "ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp:ExSIA::run",
             "activation rows committed before RESIDUAL_PAYLOAD seal",
             "ggml/src/ggml-gemmini/quants/act/exsia/exsia.cpp"),
        host("input.host_input_preparation", "per-operation host input preparation",
             "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:HostCpuInterval",
             "before frontend start", "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp"),
        host("input.stripe_input_capture", "per-stripe input capture into submission metadata",
             "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:HostCpuInterval",
             "before stripe submit", "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp"),
        host("weight.input_snapshot", "per-operation runtime copy of weight bytes, radix codes and scales",
             "frontend/src/im2p_gemmini_frontend.cpp:retain_provider_operands",
             "before first stripe dequeue of the operation",
             "frontend/src/im2p_gemmini_frontend.cpp", "IM2P.sim"),
        host("residual.metadata_preparation", "per-stripe residual metadata preparation",
             "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:HostCpuInterval",
             "before residual run preparation", "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp"),
        host("residual.run_preparation", "balanced-radix decomposition, zero-limb pruning, row-map and ordered-run construction",
             "ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:execute_rmd_stripe_im2p_run_aware",
             "residual payload sealed before FRONTEND_QUEUE enqueue",
             "ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp"),
        npu("npu.dense_service", "dense GEMM service window",
            "dense_cycles = result_ready - accepted for dense_main works", "result_ready_cycle"),
        npu("npu.residual_service", "residual GEMM service window",
            "residual_cycles = result_ready - accepted for residual works", "result_ready_cycle"),
        npu("npu.backing_loads", "A/B scratchpad loads from reference backing",
            "load_cycles = union of [ReadRequest, matching ReadResponse] (cycle_accounting.py)",
            "inside service window"),
        npu("npu.backing_stores", "accumulator stores to reference backing",
            "store_cycles = union of [WriteRequest, next WriteCompletion] (cycle_accounting.py)",
            "store completion precedes resource_ready"),
        npu("npu.scale_fetch", "scale fetch, lane and release path",
            "scale_cycles and scu_active_cycles unions from scale events (cycle_accounting.py)",
            "final_scale_release_cycle <= resource_ready_cycle"),
        host("reconstruct.output_reconstruction", "integer results scaled into staged float output",
             "frontend/src/im2p_gemmini_frontend.cpp:provider_write_output",
             "after work result_ready", "frontend/src/im2p_gemmini_frontend.cpp", "IM2P.sim"),
        host("reconstruct.radix_recomposition", "CPU radix recomposition of compact residual integers",
             "ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:execute_rmd_stripe_im2p_run_aware",
             "after residual work result_ready", "ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp"),
        host("reconstruct.residual_output_publish", "residual correction publish into staged output",
             "ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp:execute_rmd_stripe_im2p_run_aware",
             "before merge", "ggml/src/ggml-gemmini/residual/rmd/rmd-executor.cpp"),
        host("reconstruct.output_correction_apply", "main/residual merge into final float output",
             "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:HostCpuInterval",
             "after dense and residual results of the stripe", "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp"),
        host("reconstruct.output_copy", "staged output copy to destination buffer",
             "frontend/src/im2p_gemmini_frontend.cpp:commit_output",
             "after all stripe merges of the operation", "frontend/src/im2p_gemmini_frontend.cpp", "IM2P.sim"),
        host("output.buffer_copy", "backend tensor copy of committed output",
             "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:copy_staged_output",
             "after output authorize", "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp"),
        host("output.post_fence_validation", "per-operation post-fence validation",
             "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:HostCpuInterval",
             "after fence", "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp"),
        stage("emulation.materialize", "CPU-functional operand materialization for the value-producing emulator",
              "IM2P.sim", "frontend/src/im2p_cpu_functional_compute.cpp:prepare", "EXCLUDED_WITH_REASON",
              "none", "FUNCTIONAL_EMULATION interval class; capture_cpu_exclusion scope", "host_cpu_interval",
              False, "VALUE_PROGRESSION_ONLY", "not on target",
              "CPU-functional NPU emulation exists only to progress values; target executes this on the NPU",
              "frontend/src/im2p_cpu_functional_compute.cpp"),
        stage("emulation.matmul", "CPU-functional NPU GEMM emulation",
              "IM2P.sim", "frontend/src/im2p_cpu_functional_compute.cpp:execute", "EXCLUDED_WITH_REASON",
              "none", "FUNCTIONAL_EMULATION interval class; capture_cpu_exclusion scope", "host_cpu_interval",
              False, "VALUE_PROGRESSION_ONLY", "not on target",
              "CPU-functional NPU emulation; the same work's target cost is npu.dense_service/npu.residual_service",
              "frontend/src/im2p_cpu_functional_compute.cpp"),
        stage("interface.input_transport", "physical host-to-NPU activation/weight transport of the target platform",
              "IM2P.sim", "frontend lifecycle target_npu_slot_domain=UNDECLARED", "UNMODELED",
              "memory-interface", "no source declares the transport", "unknown", False,
              "UNKNOWN", "between host buffers and reference backing",
              "target buffer visibility undeclared in source; do not assume shared memory, zero-copy or PCIe", None),
        stage("interface.output_transport", "physical NPU-to-host result transport of the target platform",
              "IM2P.sim", "frontend lifecycle target_npu_slot_domain=UNDECLARED", "UNMODELED",
              "memory-interface", "no source declares the transport", "unknown", False,
              "UNKNOWN", "between reference backing and host-visible buffers",
              "target buffer visibility undeclared in source; fail-closed until declared and measured", None),
        wait("wait.frontend_start", "frontend start envelope",
             "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:HostCpuInterval",
             "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp"),
        wait("wait.stripe_submit", "stripe submit envelope (includes producer capacity wait)",
             "frontend/src/im2p_gemmini_frontend.cpp:submit_stripe_planned",
             "frontend/src/im2p_gemmini_frontend.cpp", "IM2P.sim"),
        wait("wait.fence", "fence/drain envelope until all completions",
             "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp:HostCpuInterval",
             "ggml/src/ggml-gemmini/ggml-gemmini-im2p.cpp"),
        stage("application.sampling", "llama sampling and application endpoints",
              "llama.cpp-gemmini", "producer/native/application.jsonl endpoints", "DIAGNOSTIC_ONLY",
              "cpu", "potal-application-endpoints t0/sample_accept", "ns", False, "OUTSIDE_NPU_GRAPH",
              "after lm_head result visibility",
              "application endpoint timeline; joins reconstruction later through the application gate", None),
    ]
    return rows


def contract() -> Record:
    rows = stages()
    for row in rows:
        name = row["source_file"]
        if isinstance(name, str):
            root = LLAMA if row["producer"] == "llama.cpp-gemmini" else IM2P
            row["source_sha256"] = sha256((root / name).resolve(strict=True))
        else:
            row["source_sha256"] = None
    document: Record = {"schema": "potal-e2e-cost-ownership", "version": 1, "scenario": SCENARIO,
                        "classes": list(CLASSES), "wait_envelopes": list(WAIT_ENVELOPES),
                        "npu_transfer_stage_ids": list(NPU_TRANSFER_IDS),
                        "stages": [dict(row) for row in rows]}
    return document


def memory_scenario() -> Record:
    from sim.cycle.certificate_contract import TIMING

    reference: Record = {name: TIMING[name] for name in TIMING_FIELDS}
    document: Record = {"schema": "potal-memory-interface-scenario", "version": 1, "scenario": SCENARIO,
            "npu_memory_model": "REFERENCE_MEMORY", "reference_memory": reference,
            "host_npu_buffer_visibility":
                "UNDECLARED_IN_SOURCE: execution lifecycle declares target_npu_slot_domain=UNDECLARED and "
                "producer_ownership_source=CPU_FUNCTIONAL; no source binds a physical shared buffer",
            "input_transfer_contract":
                "NPU-visible activation/weight bytes are produced by HOST_MEASURED stages "
                "(activation.exsia_local, weight.input_snapshot); physical transport into backing is UNMODELED",
            "output_transfer_contract":
                "NPU results end at reference-backing WriteCompletion; host visibility is produced by "
                "HOST_MEASURED reconstruction stages; physical transport out of backing is UNMODELED",
            "shared_bandwidth_model":
                "NONE: no host/NPU shared-bandwidth arbitration is modeled; the scale path's fixed-priority "
                "one-entry backing queue is NPU-internal and already inside NPU_MODELED windows",
            "actual_dram_modeled": False, "actual_dram_measured": False,
            "excluded_or_unmodeled": [
                "actual DRAM dynamics (row activation, refresh, bank conflicts, bus contention)",
                "physical host<->NPU transport (interface.input_transport, interface.output_transport)",
                "host cache hierarchy effects beyond measured CPU intervals"]}
    return document


def accounting_definitions() -> Record:
    from cycle_accounting import DEFINITIONS

    document: Record = {"schema": "potal-cycle-accounting-definitions", "version": 1,
                        "definitions": {key: str(value) for key, value in DEFINITIONS.items()},
                        "definition_revision": dict(DEFINITION_REVISIONS),
                        "scu_activity_definition": "SCALE_PATH_FETCH_LANE_RELEASE_UNION_V1",
                        "explicit_misnomers_rejected": dict(MISNOMERS),
                        "field_names_frozen_for_backward_compatibility": True}
    return document


def validate_contract(document: Record) -> None:
    require(document.get("schema") == "potal-e2e-cost-ownership" and document.get("version") == 1,
            "unknown cost ownership schema")
    rows = stage_rows(document)
    require(bool(rows), "empty cost ownership contract")
    seen: dict[str, str] = {}
    for row in rows:
        identity, execution = text(row, "stage_id"), text(row, "execution_class")
        require(execution in CLASSES, "unknown execution class: " + identity)
        require(identity not in seen, "duplicate cost authority: " + identity)
        seen[identity] = execution
        included = row.get("included_in_target_latency")
        require(isinstance(included, bool), "included_in_target_latency must be boolean: " + identity)
        require(bool(text(row, "source_location")), "unknown source location: " + identity)
        if execution in ("EXCLUDED_WITH_REASON", "UNMODELED", "DIAGNOSTIC_ONLY"):
            require(bool(row.get("reason")), "exclusion reason required: " + identity)
            require(included is False, "non-authoritative stage cannot enter target latency: " + identity)
        if execution == "HOST_MEASURED":
            require("CANONICAL_ADDITIVE" in text(row, "measurement_source"),
                    "host stage without canonical additive measurement: " + identity)
            require("npu_dependency_envelope" not in text(row, "measurement_source"),
                    "blocking wait counted as active CPU: " + identity)
        if execution in ADDITIVE:
            require(included is True, "additive stage excluded from target latency: " + identity)
        name = row.get("source_file")
        if isinstance(name, str):
            root = LLAMA if text(row, "producer") == "llama.cpp-gemmini" else IM2P
            path = root / name
            require(path.is_file(), "unknown source location: " + identity)
            require(sha256(path) == row.get("source_sha256"), "source hash mutation: " + identity)
        else:
            require(row.get("source_sha256") is None, "source hash without source file: " + identity)
    for identity in NPU_TRANSFER_IDS:
        require(seen.get(identity) == "NPU_MODELED", "missing NPU transfer authority: " + identity)
    host_transfer = [identity for identity, execution in seen.items()
                     if execution == "HOST_MEASURED" and identity.startswith(("npu.", "interface."))]
    require(not host_transfer, "NPU/interface transfer duplicated as host transfer: " + ",".join(host_transfer))
    for base in ("emulation.matmul", "emulation.materialize"):
        require(seen.get(base) == "EXCLUDED_WITH_REASON", "functional emulation must stay excluded: " + base)
    for row in rows:
        if text(row, "execution_class") == "UNMODELED":
            require(row.get("included_in_target_latency") is False,
                    "unmodeled stage cannot be additive: " + text(row, "stage_id"))
    envelope = [name for name in WAIT_ENVELOPES if name in seen]
    require(not envelope, "wait envelopes are interval classes, not stages: " + ",".join(envelope))


def unmodeled_required(document: Record) -> list[str]:
    return [text(row, "stage_id") for row in stage_rows(document)
            if row.get("execution_class") == "UNMODELED"]


def validate_memory_structure(document: Record) -> None:
    require(document.get("schema") == "potal-memory-interface-scenario" and document.get("version") == 1,
            "unknown memory scenario schema")
    require(document.get("npu_memory_model") == "REFERENCE_MEMORY", "unsupported NPU memory model")
    reference = record(document.get("reference_memory"))
    require(tuple(sorted(reference)) == tuple(sorted(TIMING_FIELDS)) and
            all(isinstance(reference[name], int) for name in TIMING_FIELDS),
            "reference memory timing fields incomplete")
    require(document.get("actual_dram_modeled") is False and document.get("actual_dram_measured") is False,
            "actual DRAM must stay unmodeled and unmeasured in this scenario")
    for name in ("host_npu_buffer_visibility", "input_transfer_contract", "output_transfer_contract",
                 "shared_bandwidth_model"):
        value = text(document, name)
        require(bool(value), "empty memory interface field: " + name)
        lowered = value.lower()
        for assumption in ("pcie", "zero-copy", "dma-coherent"):
            require(assumption not in lowered, "unsupported interface assumption in " + name)
        require("shared memory" not in lowered or "no source" in lowered or "NONE" in value,
                "unsupported shared-memory assumption in " + name)
    listed = string_rows(document, "excluded_or_unmodeled")
    require(any("DRAM" in item for item in listed), "DRAM exclusion must be listed")


def validate_memory(document: Record) -> None:
    from sim.cycle.certificate_contract import TIMING

    validate_memory_structure(document)
    reference = record(document.get("reference_memory"))
    require(reference == {name: TIMING[name] for name in TIMING_FIELDS},
            "reference memory timing differs from the pinned certificate contract")


def validate_definitions(document: Record) -> None:
    from cycle_accounting import DEFINITIONS

    require(document.get("schema") == "potal-cycle-accounting-definitions" and document.get("version") == 1,
            "unknown accounting definitions schema")
    require(record(document.get("definitions")) == {key: str(value) for key, value in DEFINITIONS.items()},
            "accounting definitions differ from cycle_accounting.DEFINITIONS")
    require(record(document.get("definition_revision")) == dict(DEFINITION_REVISIONS),
            "definition revisions changed")
    require(document.get("scu_activity_definition") == "SCALE_PATH_FETCH_LANE_RELEASE_UNION_V1",
            "scu activity definition changed")


def build(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    documents = {"cost-ownership-contract.json": contract(),
                 "memory-interface-scenario.json": memory_scenario(),
                 "cycle-accounting-definitions.json": accounting_definitions()}
    for name, document in documents.items():
        write_json(directory / name, document)
    validate(directory)


def validate(directory: Path) -> Record:
    ownership = read_json(directory / "cost-ownership-contract.json")
    validate_contract(ownership)
    memory = read_json(directory / "memory-interface-scenario.json")
    validate_memory(memory)
    definitions = read_json(directory / "cycle-accounting-definitions.json")
    validate_definitions(definitions)
    additive_unmodeled = [text(row, "stage_id") for row in stage_rows(ownership)
                          if row.get("execution_class") == "UNMODELED" and
                          row.get("included_in_target_latency") is True]
    ready: Record = {"E2E_COST_OWNERSHIP_READY": "NOT_READY" if additive_unmodeled else "READY",
                     "MEMORY_INTERFACE_SCENARIO_READY": "READY",
                     "unmodeled_declared_stage_ids": [name for name in unmodeled_required(ownership)],
                     "cost_ownership_sha256": sha256(directory / "cost-ownership-contract.json"),
                     "memory_interface_sha256": sha256(directory / "memory-interface-scenario.json"),
                     "cycle_accounting_definitions_sha256":
                         sha256(directory / "cycle-accounting-definitions.json")}
    print(str(ready))
    return ready


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command == "build":
            build(arguments.directory.resolve())
        else:
            validate(arguments.directory.resolve(strict=True))
    except (EvaluationError, OSError, ValueError) as error:
        print("cost ownership failed: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(IM2P))
    raise SystemExit(main())
