from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/eval")]
import offline_pipeline
from certified_reconstruction import (
    bound_artifact,
    service_arguments,
    stateful_publication_gate_codes,
)
from e2e_cost_ownership import contract, validate_contract, validate_memory_structure
from eval_common import Json, Record, record, write_json


def rejected(document: Record, reason: str) -> None:
    try:
        validate_contract(document)
    except ValueError as error:
        assert reason in str(error), error
    else:
        raise AssertionError("invalid cost ownership contract admitted: " + reason)


def mutate(document: Record, identity: str, **changes: Json) -> Record:
    clone = copy.deepcopy(document)
    stages = clone["stages"]
    assert isinstance(stages, list)
    for row in stages:
        bound = record(row)
        if bound.get("stage_id") == identity:
            bound.update(changes)
            return clone
    raise AssertionError("unknown fixture stage: " + identity)


def offline_arguments(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {"im2p": ROOT.parent / "IM2P.sim", "timeout": 600,
                                 "stateful_sequence_certificate": None, "stateful_evidence_root": None,
                                 "stateful_diagnostic": False, "diagnostic_phase_table": None,
                                 "diagnostic_frequency_hz": None, "clock_selection": None,
                                 "lifecycle_sidecar": None, "worker_resources": None,
                                 "cpu_policy": None, "sampler_resource": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def offline_rejected(reason: str, **overrides: object) -> None:
    with TemporaryDirectory(prefix="offline-gate-") as temporary:
        try:
            offline_pipeline.reconstruct(offline_arguments(output=Path(temporary) / "out", **overrides))
        except ValueError as error:
            assert reason in str(error), error
        else:
            raise AssertionError("offline stateful publication admitted: " + reason)


def check() -> None:
    document = contract()
    validate_contract(document)
    # Given: a valid contract, each forbidden mutation must fail closed.
    duplicated = copy.deepcopy(document)
    stages = duplicated["stages"]
    assert isinstance(stages, list)
    stages.append(copy.deepcopy(stages[1]))
    rejected(duplicated, "duplicate cost authority")
    missing = copy.deepcopy(document)
    stages = missing["stages"]
    assert isinstance(stages, list)
    missing["stages"] = [row for row in stages if record(row).get("stage_id") != "npu.backing_loads"]
    rejected(missing, "missing NPU transfer authority")
    rejected(mutate(document, "interface.input_transport", included_in_target_latency=True),
             "non-authoritative stage cannot enter target latency")
    rejected(mutate(document, "emulation.matmul", execution_class="NPU_MODELED",
                    included_in_target_latency=True, reason=None),
             "functional emulation must stay excluded")
    rebadged = copy.deepcopy(document)
    stages = rebadged["stages"]
    assert isinstance(stages, list)
    stages.append({"stage_id": "npu.backing_loads_as_host_transfer", "name": "re-added NPU DMA",
                   "producer": "llama.cpp-gemmini", "source_location": "x:y",
                   "execution_class": "HOST_MEASURED", "resource": "cpu",
                   "measurement_source": "gemmini_cycle_record_v2 CANONICAL_ADDITIVE interval",
                   "measurement_unit": "host_cpu_interval", "included_in_target_latency": True,
                   "overlap_policy": "SERIALIZED_WITH_DECLARED_DEPENDENCIES",
                   "dependency_boundary": "x", "reason": None, "source_file": None,
                   "source_sha256": None})
    rejected(rebadged, "duplicated as host transfer")
    rejected(mutate(document, "activation.exsia_local",
                    measurement_source="cpu_service_exclusion npu_dependency_envelope CANONICAL_ADDITIVE"),
             "blocking wait counted as active CPU")
    rejected(mutate(document, "activation.exsia_local", source_sha256="0" * 64), "source hash mutation")
    rejected(mutate(document, "activation.exsia_local", source_file="ggml/does-not-exist.cpp"),
             "unknown source location")
    wait_stage = mutate(document, "wait.fence", stage_id="im2p.fence_host_call")
    rejected(wait_stage, "wait envelopes are interval classes")
    # Given: memory scenario structural invariants.
    for name, changes in (("dram", {"actual_dram_modeled": True}),
                          ("assumption", {"input_transfer_contract": "zero-copy PCIe window"})):
        scenario: Record = {"schema": "potal-memory-interface-scenario", "version": 1,
                            "npu_memory_model": "REFERENCE_MEMORY",
                            "reference_memory": {"backing_read_delay": 3, "even_read_id_delay": 13,
                                                 "scale_read_extra_delay": 17, "backing_write_delay": 11,
                                                 "read_ready_period": 5, "backing_cycle_offset": 5},
                            "host_npu_buffer_visibility": "UNDECLARED_IN_SOURCE",
                            "input_transfer_contract": "UNMODELED transport",
                            "output_transfer_contract": "UNMODELED transport",
                            "shared_bandwidth_model": "NONE",
                            "actual_dram_modeled": False, "actual_dram_measured": False,
                            "excluded_or_unmodeled": ["actual DRAM dynamics"]}
        scenario.update(changes)
        try:
            validate_memory_structure(scenario)
        except ValueError:
            continue
        raise AssertionError("invalid memory scenario admitted: " + name)
    # When: no artifacts are bound, the ordered gates fail closed with explicit codes.
    codes = stateful_publication_gate_codes({})
    assert codes[:2] == ["NOT_READY_MISSING_COST_OWNERSHIP_CONTRACT",
                         "NOT_READY_MISSING_MEMORY_INTERFACE_SCENARIO"]
    assert "NOT_READY_MISSING_PROVIDER_CERTIFICATE" in codes
    assert codes[-2:] == ["NOT_READY_MISSING_TARGET_HOST_ADMISSION", "NOT_READY_MISSING_OPERATING_CLOCK"]
    with TemporaryDirectory(prefix="publication-gates-") as temporary:
        artifact = Path(temporary) / "artifact.json"
        write_json(artifact, {"value": 1})
        try:
            bound_artifact({"path": str(artifact), "sha256": "0" * 64})
        except ValueError as error:
            assert "artifact binding mismatch" in str(error)
        else:
            raise AssertionError("wrong artifact hash admitted")
    try:
        service_arguments({"stateful_sequence_certificate": {"path": "a", "sha256": "b"},
                           "service_certificate": {"path": "c", "sha256": "d"}})
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("conflicting provider certificates admitted")
    # Then: the offline entry blocks stateful publication with explicit gate codes.
    offline_rejected("required together", stateful_sequence_certificate=Path("certificate.json"))
    offline_rejected("NOT_READY_MISSING_TARGET_HOST_ADMISSION",
                     stateful_sequence_certificate=Path("certificate.json"),
                     stateful_evidence_root=Path("evidence"))
    offline_rejected("NOT_READY_MISSING_OPERATING_CLOCK",
                     stateful_sequence_certificate=Path("certificate.json"),
                     stateful_evidence_root=Path("evidence"))
    offline_rejected("configured test clock only",
                     stateful_diagnostic=True)
    offline_rejected("synthetic phase table",
                     stateful_sequence_certificate=Path("certificate.json"),
                     stateful_evidence_root=Path("evidence"),
                     diagnostic_phase_table=Path("table.json"))


if __name__ == "__main__":
    check()
    print("PASS: duplicate/missing/unmodeled/emulation/DMA-rebadge/wait/hash/location contract rejections, "
          "memory scenario invariants, ordered fail-closed publication gates")
