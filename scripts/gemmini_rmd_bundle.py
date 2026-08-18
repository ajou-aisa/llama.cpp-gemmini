"""Typed immutable Gemmini RMD runtime bundle operations."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Final, NewType, TypedDict, cast, override

Sha256 = NewType("Sha256", str)
SCHEMA: Final = "aisa.gemmini-rmd-runtime-bundle"
VERSION: Final = 1
PROJECT_PREFIXES: Final = ("libllama", "libggml")


class Code(IntEnum):
    OK = 0
    USAGE = 2
    IO_ERROR = 3
    WRONG_BINARY = 10
    WRONG_MODE = 11
    MISSING_PROJECT_LIBRARY = 12
    RESOLUTION_ERROR = 13
    HASH_MISMATCH = 20
    PROVENANCE_MISMATCH = 21
    SCHEMA_MISMATCH = 22


@dataclass(frozen=True, slots=True)
class BundleError(Exception):
    code: Code
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.code.name}: {self.detail}"


class FileRecord(TypedDict):
    path: str
    sha256: str
    mode: int


class InputRecord(TypedDict):
    path: str
    sha256: str


class DynamicRecord(TypedDict):
    path: str
    needed: list[str]
    rpath: list[str]


class Manifest(TypedDict):
    schema: str
    schema_version: int
    profile: str
    source_sha256: str
    working_state_sha256: str
    build_flags: list[str]
    binary_version: str
    compiler: InputRecord
    compiler_version: str
    dependencies: list[InputRecord]
    configs: list[InputRecord]
    gemmini_headers: list[InputRecord]
    binary: FileRecord
    project_libraries: list[FileRecord]
    dynamic: list[DynamicRecord]
    model: InputRecord
    guest_model_path: str
    system_inputs: list[InputRecord]
    canonical_bundle_sha256: str


@dataclass(frozen=True, slots=True)
class ManifestRequest:
    bundle: Path
    binary: Path
    libraries: tuple[Path, ...]
    profile: str
    source_sha256: str
    working_state_sha256: str
    build_flags: tuple[str, ...]
    binary_version: str
    compiler: Path
    compiler_flags: tuple[str, ...]
    dependencies: tuple[Path, ...]
    configs: tuple[Path, ...]
    model: Path
    guest_model_path: str
    system_inputs: tuple[Path, ...]
    readelf: str


def digest(path: Path) -> Sha256:
    value = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                value.update(chunk)
    except OSError as error:
        raise BundleError(Code.IO_ERROR, f"cannot hash {path}: {error}") from error
    return Sha256(value.hexdigest())


def input_record(path: Path) -> InputRecord:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": digest(resolved)}


def file_record(root: Path, path: Path) -> FileRecord:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest(path),
        "mode": stat.S_IMODE(path.stat().st_mode),
    }


def run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise BundleError(
            Code.IO_ERROR, f"command failed: {' '.join(command)}: {error}"
        ) from error
    return result.stdout.strip()


def inspect_dynamic(readelf: str, root: Path, path: Path) -> DynamicRecord:
    output = run([readelf, "-d", str(path)])
    needed: list[str] = re.findall(r"\(NEEDED\).*?\[([^]]+)]", output)
    raw: list[str] = re.findall(r"\((?:RUNPATH|RPATH)\).*?\[([^]]*)]", output)
    paths = [entry for group in raw for entry in group.split(":") if entry]
    if any(name.startswith("/") for name in needed):
        raise BundleError(Code.RESOLUTION_ERROR, f"absolute NEEDED in {path}")
    if any(not entry.startswith("$ORIGIN") for entry in paths):
        raise BundleError(
            Code.RESOLUTION_ERROR, f"non-$ORIGIN runtime path in {path}: {paths}"
        )
    return {"path": path.relative_to(root).as_posix(), "needed": needed, "rpath": paths}


def resolved_headers(compiler: Path, flags: tuple[str, ...]) -> list[InputRecord]:
    output = run(
        [
            str(compiler),
            *flags,
            "-M",
            "-E",
            "-x",
            "c++",
            "-include",
            "gemmini.h",
            "/dev/null",
        ]
    )
    words = output.replace("\\\n", " ").partition(":")[2].split()
    headers = sorted(
        {
            Path(word).resolve()
            for word in words
            if "gemmini" in word.lower() and Path(word).is_file()
        }
    )
    if not headers:
        raise BundleError(
            Code.PROVENANCE_MISMATCH, "compiler resolved no Gemmini headers"
        )
    return [input_record(path) for path in headers]


def canonical_hash(manifest: Manifest) -> Sha256:
    material = manifest.copy()
    material["canonical_bundle_sha256"] = ""
    return Sha256(
        hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def create(request: ManifestRequest) -> Path:
    if (
        request.binary.name != "llama-cli"
        or request.binary.read_bytes()[:4] != b"\x7fELF"
    ):
        raise BundleError(
            Code.WRONG_BINARY, f"expected ELF llama-cli, got {request.binary}"
        )
    if request.bundle.exists():
        raise BundleError(Code.IO_ERROR, f"bundle already exists: {request.bundle}")
    request.bundle.mkdir(parents=True, mode=0o755)
    bin_dir = request.bundle / "bin"
    bin_dir.mkdir(mode=0o755)
    sources = (request.binary, *request.libraries)
    copied = [bin_dir / path.name for path in sources]
    for source, destination in zip(sources, copied, strict=True):
        _ = shutil.copy2(source, destination)
        destination.chmod(0o555)
    dynamic = [
        inspect_dynamic(request.readelf, request.bundle, path) for path in copied
    ]
    supplied = {path.name for path in request.libraries}
    required = {
        name
        for item in dynamic
        for name in item["needed"]
        if name.startswith(PROJECT_PREFIXES)
    }
    missing = required - supplied
    if missing:
        raise BundleError(Code.MISSING_PROJECT_LIBRARY, ", ".join(sorted(missing)))
    if required and not dynamic[0]["rpath"]:
        raise BundleError(
            Code.RESOLUTION_ERROR, "llama-cli has no $ORIGIN RPATH/RUNPATH"
        )
    compiler_version = run([str(request.compiler), "--version"]).splitlines()[0]
    manifest: Manifest = {
        "schema": SCHEMA,
        "schema_version": VERSION,
        "profile": request.profile,
        "source_sha256": request.source_sha256,
        "working_state_sha256": request.working_state_sha256,
        "build_flags": list(request.build_flags),
        "binary_version": request.binary_version,
        "compiler": input_record(request.compiler),
        "compiler_version": compiler_version,
        "dependencies": [input_record(path) for path in request.dependencies],
        "configs": [input_record(path) for path in request.configs],
        "gemmini_headers": resolved_headers(request.compiler, request.compiler_flags),
        "binary": file_record(request.bundle, copied[0]),
        "project_libraries": [file_record(request.bundle, path) for path in copied[1:]],
        "dynamic": dynamic,
        "model": input_record(request.model),
        "guest_model_path": request.guest_model_path,
        "system_inputs": [input_record(path) for path in request.system_inputs],
        "canonical_bundle_sha256": "",
    }
    manifest["canonical_bundle_sha256"] = canonical_hash(manifest)
    path = request.bundle / "manifest.json"
    _ = path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    path.chmod(0o444)
    bin_dir.chmod(0o555)
    request.bundle.chmod(0o555)
    return path


def load(path: Path) -> Manifest:
    try:
        decoded = cast(object, json.loads(path.read_text()))
        if not isinstance(decoded, dict):
            raise TypeError("manifest root is not an object")
        return cast(Manifest, cast(object, decoded))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise BundleError(
            Code.SCHEMA_MISMATCH, f"invalid manifest {path}: {error}"
        ) from error


def verify(bundle: Path, readelf: str, expected: Manifest | None = None) -> Manifest:
    manifest = load(bundle / "manifest.json")
    if manifest["schema"] != SCHEMA or manifest["schema_version"] != VERSION:
        raise BundleError(Code.SCHEMA_MISMATCH, "unsupported schema")
    if canonical_hash(manifest) != manifest["canonical_bundle_sha256"]:
        raise BundleError(Code.HASH_MISMATCH, "canonical bundle hash changed")
    records = [manifest["binary"], *manifest["project_libraries"]]
    for record in records:
        path = bundle / record["path"]
        if not path.is_file():
            raise BundleError(Code.MISSING_PROJECT_LIBRARY, record["path"])
        if digest(path) != record["sha256"]:
            raise BundleError(Code.HASH_MISMATCH, record["path"])
        if (
            stat.S_IMODE(path.stat().st_mode) != record["mode"]
            or record["mode"] != 0o555
        ):
            raise BundleError(Code.WRONG_MODE, record["path"])
    if (
        stat.S_IMODE(bundle.stat().st_mode) != 0o555
        or stat.S_IMODE((bundle / "manifest.json").stat().st_mode) != 0o444
    ):
        raise BundleError(Code.WRONG_MODE, str(bundle))
    actual = [
        inspect_dynamic(readelf, bundle, bundle / record["path"]) for record in records
    ]
    if actual != manifest["dynamic"]:
        raise BundleError(Code.PROVENANCE_MISMATCH, "dynamic-link evidence changed")
    inputs = [
        manifest["compiler"],
        *manifest["dependencies"],
        *manifest["configs"],
        *manifest["gemmini_headers"],
        manifest["model"],
        *manifest["system_inputs"],
    ]
    for item in inputs:
        if digest(Path(item["path"])) != item["sha256"]:
            raise BundleError(Code.PROVENANCE_MISMATCH, item["path"])
    if expected is not None and canonical_hash(expected) != canonical_hash(manifest):
        raise BundleError(
            Code.PROVENANCE_MISMATCH, "manifest differs from expected provenance"
        )
    return manifest
