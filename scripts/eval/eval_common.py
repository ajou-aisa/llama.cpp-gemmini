from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import TypeAlias

Json: TypeAlias = str | int | float | bool | None | list["Json"] | dict[str, "Json"]
Record: TypeAlias = dict[str, Json]


@dataclass(frozen=True, slots=True)
class EvaluationError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise EvaluationError(reason)


def record(value: Json) -> Record:
    require(isinstance(value, dict), "expected JSON object")
    if not isinstance(value, dict):
        raise EvaluationError("expected JSON object")
    return value


def integer(row: Record, key: str, minimum: int = 0) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise EvaluationError("invalid integer: " + key)
    return value


def text(row: Record, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise EvaluationError("missing text: " + key)
    return value


def unique_pairs(pairs: list[tuple[str, Json]]) -> Record:
    result: Record = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def decode(value: str) -> Record:
    return record(json.loads(value, object_pairs_hook=unique_pairs))


def read_json(path: Path) -> Record:
    return decode(path.read_text(encoding="utf-8"))


def records(path: Path) -> Iterator[Record]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"empty JSONL record at line {number}")
            yield decode(line)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Json) -> None:
    require(not path.exists(), "refusing existing output: " + str(path))
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.link(temporary, path)
    temporary.unlink()


def run(command: list[str], output: Path, timeout: int) -> None:
    require(timeout > 0, "timeout must be positive")
    with (output / "process.log").open("x", encoding="utf-8") as log:
        try:
            completed = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            write_json(output / "command.json", {"argv": list(command), "cwd": str(output),
                                                "exit_code": None, "timeout_seconds": timeout, "status": "TIMEOUT"})
            raise
    write_json(output / "command.json", {"argv": list(command), "cwd": str(output),
                                        "exit_code": completed.returncode, "timeout_seconds": timeout})
    require(completed.returncode == 0, "native collection failed; see process.log")


def artifact_snapshot(binary: Path) -> Record:
    files = {binary.resolve(), *(path.resolve() for path in binary.parent.iterdir()
              if path.is_file() and (".so" in path.name or ".dylib" in path.name))}
    return {str(path): sha256(path) for path in sorted(files)}


def compiled_info(binary: Path) -> Record:
    completed = subprocess.run([str(binary), "--build-info"], capture_output=True,
                               text=True, timeout=15, check=False)
    require(completed.returncode == 0, "native --build-info failed")
    info = decode(completed.stdout)
    require(info.get("schema") == "potal-evaluation-build" and info.get("version") == 1,
            "unsupported native build-info schema")
    return info


def validate_recipe(info: Record, recipe: str) -> None:
    expected = {"activation": (1, 0), "residual": (0, 1), "e2e": (0, 0)}
    require(recipe in expected, "unsupported evaluation recipe")
    actual = tuple(integer(info, name) for name in
                   ("activation_metrics", "residual_metrics"))
    require(actual == expected[recipe], f"script/build metric flag mismatch for {recipe}")
    require(integer(info, "cycle_sim") in (0, 1), "invalid CYCLE_SIM build value")
    if recipe != "e2e":
        require(integer(info, "cycle_sim") == 1 and info.get("backend") == "IM2P_SIM" and info.get("hp1") is True,
                "metric recipe requires explicit CPU-functional CYCLE_SIM=1 HP1 artifact; device/RTL collection is unsupported")
        require(integer(info, "gemmini") == 1 and info.get("gemmini_option") == "WS",
                "metric recipe requires actual offloaded Gemmini WS route")
        require(integer(info, "activation_bits") == integer(info, "weight_bits") and
                integer(info, "activation_bits") in (4, 8) and integer(info, "dim") in (16, 32, 64),
                "unsupported metric hardware profile")
        require(info.get("activation_mode") == "EXSIA" and integer(info, "block_size") == 32,
                "metric hooks require production EXSIA block32 route")
    if recipe == "residual":
        require(info.get("hp1") is True and integer(info, "rmd_enabled") == 1 and
                info.get("rmd_backend") == "WS", "RES recipe requires enabled run-aware HP1 WS residual route")


def clean_environment() -> None:
    forbidden = ("LLAMA_ARG_", "DYLD_", "LD_", "GGML_", "EXSIA_", "OMP_", "GOMP_", "KMP_", "OPENBLAS_", "VECLIB_")
    require(not any(key.startswith(forbidden) or (key.startswith("GEMMINI_") and key != "GEMMINI_LOG_DIR")
                    for key in os.environ),
            "unbound model/loader/backend environment override")


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
