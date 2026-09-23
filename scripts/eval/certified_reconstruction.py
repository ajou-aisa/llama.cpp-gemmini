from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys

from application_results import load_potal_collection
from eval_common import Json, Record, decode, integer, read_json, record, require, sha256, text
from scheduled_endpoints import prefill_dispatches, scheduled_application_result


def artifact_reference(path: Path) -> Record:
    source = path.resolve(strict=True)
    return {"path": str(source), "sha256": sha256(source)}


def bound_artifact(value: Json) -> Path:
    reference = record(value)
    path = Path(text(reference, "path"))
    require(path.is_absolute() and sha256(path.resolve(strict=True)) == text(reference, "sha256"),
            "reconstruction artifact binding mismatch: " + str(path))
    return path.resolve()


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
    required = ("potal_result", "application", "potal_provenance", "clock_selection", "service_certificate",
                "library", "npu_trace", "timing", "cycle_certificate", "run_aware_certificate")
    sources = {name: bound_artifact(inputs.get(name)) for name in required}
    for name, value in inputs.items():
        if name not in (*required, "scenario", "im2p"):
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
    scenario = record(inputs.get("scenario"))
    command = [sys.executable, "-B", "-m", "sim.cycle.execution_cli", "verify-schedule",
               "--schedule", str(files["schedule"]), "--bundle", str(files["bundle"]),
               "--cycle-library", str(sources["library"]), "--npu-trace", str(sources["npu_trace"]),
               "--timing", str(sources["timing"]),
               "--initial-scratchpad-half", str(integer(scenario, "initial_scratchpad_half")),
               "--initial-accumulator-half", str(integer(scenario, "initial_accumulator_half")),
               "--service-certificate", str(sources["service_certificate"]),
               "--cycle-certificate", str(sources["cycle_certificate"]),
               "--run-aware-certificate", str(sources["run_aware_certificate"]),
               "--clock-selection", str(sources["clock_selection"]),
               "--profile", text(scenario, "profile")]
    verified = subprocess.run(command, cwd=im2p, capture_output=True, text=True, timeout=600, check=False)
    require(verified.returncode == 0, "official schedule verification failed: " + verified.stderr.strip())
    require(decode(verified.stdout).get("status") == "PASS", "official schedule verifier did not pass")
    source = load_potal_collection(sources["potal_result"], sources["application"],
                                   sources["potal_provenance"], files["join_summary"])
    require(row == reconstructed_row(source, scheduled_application_result(files["schedule"],
                                                                          prefill_dispatches(lifecycle)), proof),
            "reconstructed result differs from bound official schedule/source")
    require(all(bound_artifact(proof.get(name)) == path for name, path in files.items()) and
            all(bound_artifact(inputs.get(name)) == path for name, path in sources.items()),
            "reconstruction source changed during verification")
    return row
