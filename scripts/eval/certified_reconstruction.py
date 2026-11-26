"""Hash-bound artifacts and publication rows shared by evaluation consumers."""
from __future__ import annotations

from pathlib import Path
from eval_common import Json, Record, record, require, sha256, text


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
    """A validated reconstruction keeps the collection's target_latency deficiency marker; see publication_row."""
    row = dict(source)
    row.update(result)
    row.update(measurement_kind="VALIDATED_RECONSTRUCTION", scope="OFFICIAL_RECONSTRUCTED_APPLICATION",
               reconstruction=proof)
    return row


def publication_workload(source: Record, inputs: Record) -> Record:
    return {"model_sha256": source.get("model_sha256"), "input_tokens_sha256": source.get("input_tokens_sha256"),
            "generated_tokens_sha256": source.get("generated_tokens_sha256"),
            "profile": record(inputs.get("scenario")).get("profile")}


def publication_row(row: Record, readiness: Record) -> Record:
    """Target latency is published only when every gate passed; the reconstruction alone never suffices."""
    require(readiness.get("TARGET_LATENCY_READY") is True,
            "service publication NOT_READY: " + "; ".join(str(code) for code in list_value(readiness.get("codes"))))
    published = dict(row)
    published.update(target_latency="TARGET_LATENCY_READY", publication=readiness)
    return published


def list_value(value: Json) -> list[Json]:
    return value if isinstance(value, list) else []
