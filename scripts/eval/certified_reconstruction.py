from __future__ import annotations

import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

from application_results import load_potal_collection
from eval_common import (
    Json,
    Record,
    decode,
    integer,
    read_json,
    record,
    require,
    sha256,
    text,
)
from scheduled_endpoints import (
    VerifiedStatefulSchedule,
    prefill_dispatches,
    scheduled_application_result,
)


def artifact_reference(path: Path) -> Record:
    source = path.resolve(strict=True)
    return {"path": str(source), "sha256": sha256(source)}


def bound_artifact(value: Json) -> Path:
    reference = record(value)
    path = Path(text(reference, "path"))
    require(path.is_absolute() and sha256(path.resolve(strict=True)) == text(reference, "sha256"),
            "reconstruction artifact binding mismatch: " + str(path))
    return path.resolve()


def consumer_sources() -> Record:
    return {name: sha256(Path(__file__).with_name(name)) for name in
            ("offline_pipeline.py", "certified_reconstruction.py", "scheduled_endpoints.py",
             "application_results.py", "end_to_end.py", "eval_common.py")}


def service_arguments(inputs: Record) -> list[str]:
    stateful = "stateful_sequence_certificate" in inputs
    require(not stateful or "service_certificate" not in inputs, "mutually exclusive service certificates required")
    names = ("library", "npu_trace", "timing", "cycle_certificate", "run_aware_certificate",
             "stateful_sequence_certificate" if stateful else "service_certificate")
    arguments: list[str] = []
    for name in names:
        arguments.extend(("--cycle-library" if name == "library" else "--" + name.replace("_", "-"),
                          str(bound_artifact(inputs.get(name)))))
    scenario = record(inputs.get("scenario"))
    for name in ("initial_scratchpad_half", "initial_accumulator_half"):
        arguments.extend(("--" + name.replace("_", "-"), str(integer(scenario, name))))
    arguments.extend(("--profile", text(scenario, "profile")))
    if stateful:
        root = Path(text(record(inputs.get("stateful_evidence_root")), "path")).resolve(strict=True)
        require(root.is_dir(), "stateful evidence root must be a directory")
        arguments.extend(("--stateful-evidence-root", str(root)))
    if scenario.get("stateful_diagnostic") is True:
        require(stateful and "clock_selection" not in inputs, "stateful diagnostic requires configured test clock only")
        arguments.extend(("--stateful-diagnostic", "--frequency-hz", str(integer(scenario, "diagnostic_frequency_hz", 1))))
    else:
        arguments.extend(("--clock-selection", str(bound_artifact(inputs.get("clock_selection")))))
    return arguments


def verify_official_schedule(files: dict[str, Path], inputs: Record, im2p: Path) -> VerifiedStatefulSchedule | None:
    stateful = "stateful_sequence_certificate" in inputs
    closure = consumer_sources()
    require(not stateful or inputs.get("consumer_sources") == closure, "reconstruction consumer source binding mismatch")
    command = [sys.executable, "-B", "-m", "sim.cycle.execution_cli", "verify-schedule",
               "--schedule", str(files["schedule"]), "--bundle", str(files["bundle"]), *service_arguments(inputs)]
    verified = subprocess.run(command, cwd=im2p, capture_output=True, text=True, timeout=600, check=False)
    require(verified.returncode == 0, "official schedule verification failed: " + verified.stderr.strip())
    result = decode(verified.stdout)
    require(result.get("status") == "PASS", "official schedule verifier did not pass")
    if not stateful:
        return None
    require(result.get("schema") == "im2p-execution-schedule-verification" and result.get("version") == 2 and
            result.get("schedule_sha256") == sha256(files["schedule"]) and consumer_sources() == closure,
            "stateful verifier schedule/consumer source binding mismatch")
    binding = record(result.get("service_binding"))
    scenario = record(inputs.get("scenario"))
    if scenario.get("stateful_diagnostic") is True:
        require(binding.get("scope") == "DIAGNOSTIC_STATEFUL_SEQUENCE" and binding.get("production_admitted") is False,
                "diagnostic stateful verifier scope mismatch")
        return None
    require(binding.get("scope") == "CURRENT_STATEFUL_SEQUENCE" and binding.get("production_admitted") is True and
            binding.get("validation_scope") == "STATEFUL_SEQUENCE_PRODUCTION" and
            binding.get("profile") == scenario.get("profile") and
            binding.get("trace_sha256") == sha256(bound_artifact(inputs.get("npu_trace"))) and
            record(binding.get("scoped")).get("certificate_sha256") == sha256(bound_artifact(inputs.get("stateful_sequence_certificate"))) and
            record(binding.get("source_identity")).get("library_sha256") == sha256(bound_artifact(inputs.get("library"))) and
            binding.get("npu_frequency_hz") == integer(read_json(bound_artifact(inputs.get("clock_selection"))), "selected_frequency_hz", 1),
            "production stateful verifier admission/trace/clock/source binding mismatch")
    works = binding.get("work_bindings")
    require(isinstance(works, list) and len(works) == integer(result, "scheduled_npu_work_count", 1),
            "stateful verifier work coverage mismatch")
    return VerifiedStatefulSchedule(sha256(files["schedule"]), binding)


def reconstructed_row(source: Record, result: Record, proof: Record) -> Record:
    row = dict(source)
    row.pop("target_latency", None)
    row.update(result)
    row.update(measurement_kind="VALIDATED_RECONSTRUCTION", scope="OFFICIAL_RECONSTRUCTED_APPLICATION",
               reconstruction=proof)
    return row


def load_reconstructed_measurement(row: Record) -> Record:
    proof = record(row.get("reconstruction"))
    require(proof.get("schema") == "potal-e2e-reconstruction-proof" and proof.get("version") == 1,
            "unsupported PoTal reconstruction proof")
    files = {name: bound_artifact(proof.get(name)) for name in
             ("input_bindings", "schedule", "bundle", "join_summary", "lifecycle", "npu_results")}
    inputs = read_json(files["input_bindings"])
    stateful = "stateful_sequence_certificate" in inputs
    required = ("potal_result", "application", "potal_provenance", "clock_selection",
                "stateful_sequence_certificate" if stateful else "service_certificate",
                "library", "npu_trace", "timing", "cycle_certificate", "run_aware_certificate")
    sources = {name: bound_artifact(inputs.get(name)) for name in required}
    for name, value in inputs.items():
        if name not in (*required, "scenario", "im2p", "stateful_evidence_root", "consumer_sources"):
            bound_artifact(value)
    origin = record(inputs.get("im2p"))
    im2p = Path(__file__).resolve().parents[3] / "IM2P.sim"
    require(Path(text(origin, "path")).resolve(strict=True) == im2p.resolve(strict=True),
            "recorded IM2P source differs from local official verifier")
    require((im2p / "sim/cycle/execution_cli.py").is_file(), "official IM2P schedule verifier missing")
    lifecycle = read_json(files["lifecycle"])
    producer = record(lifecycle.get("producer_binding"))
    require(lifecycle.get("source_kind") == "PRODUCER_DECLARED" and
            producer.get("provenance_sha256") == sha256(sources["potal_provenance"]) and
            producer.get("join_summary_sha256") == sha256(files["join_summary"]) and
            record(lifecycle.get("application")).get("sha256") == sha256(sources["application"]) and
            lifecycle.get("npu_results_sha256") == sha256(files["npu_results"]) and
            record(record(read_json(files["join_summary"]).get("source_artifacts")).get("npu_results")).get("sha256") ==
                sha256(files["npu_results"]), "producer lifecycle/join/source binding mismatch")
    with files["bundle"].open("rb") as stream:
        sqlite_bundle = stream.read(16) == b"SQLite format 3\x00"
    if sqlite_bundle:
        with closing(sqlite3.connect(files["bundle"].as_uri() + "?mode=ro", uri=True)) as database:
            stored = database.execute("SELECT body FROM metadata WHERE key='manifest'").fetchone()
            require(stored is not None, "missing execution IR manifest")
            bundle = decode(stored[0])
    else:
        bundle = read_json(files["bundle"])
    require(bundle.get("lifecycle_sha256") == sha256(files["lifecycle"]),
            "execution IR/lifecycle binding mismatch")
    verified = verify_official_schedule(files, inputs, im2p)
    source = load_potal_collection(sources["potal_result"], sources["application"],
                                   sources["potal_provenance"], files["join_summary"])
    require(not stateful, "stateful publication NOT_READY: validated target-host/application admission is unavailable; "
                         "host observations and matching host_id do not prove target-host latency")
    require(row == reconstructed_row(source, scheduled_application_result(files["schedule"],
                                                                          prefill_dispatches(lifecycle), verified), proof),
            "reconstructed result differs from bound official schedule/source")
    require(all(bound_artifact(proof.get(name)) == path for name, path in files.items()) and
            all(bound_artifact(inputs.get(name)) == path for name, path in sources.items()),
            "reconstruction source changed during verification")
    return row
