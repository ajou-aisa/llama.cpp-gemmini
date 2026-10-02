"""Content-addressed cache for the offline evaluation stages (replay, join, lifecycle, IR, schedule, timeline).

A stage entry is keyed by the canonical digest of its identity: the SHA-256 of every input artifact, the stage
parameters and the digest of the source that implements the stage. An entry is published only after the stage
completed and its outputs were hashed; a reused entry must still hold exactly those bytes. Nothing is ever reused
across a different input, parameter or source identity, and a stale or partial entry is never served.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from eval_common import Json, Record, read_json, record, require, sha256, write_json

SCHEMA: Final = "potal-offline-stage-cache-entry"
RECEIPT: Final = "stage-receipt.json"


def digest(value: Json) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_digest(root: Path, patterns: Iterable[str]) -> str:
    """Digest of the source files that implement a stage (relative path -> SHA-256)."""
    files = sorted({path for pattern in patterns for path in root.glob(pattern)
                    if path.is_file() and "__pycache__" not in path.parts})
    require(bool(files), "stage source set is empty: " + str(root))
    return digest({str(path.relative_to(root)): sha256(path) for path in files})


class HashMemo:
    """SHA-256 of large immutable inputs, reused only while (device, inode, size, mtime, ctime) are unchanged."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, Json] = {}
        if path.is_file():
            self.rows = dict(record(json.loads(path.read_text())))

    def sha256(self, path: Path) -> str:
        resolved = path.resolve(strict=True)
        state = resolved.stat()
        identity: list[Json] = [state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns]
        cached = self.rows.get(str(resolved))
        if isinstance(cached, dict) and cached.get("stat") == identity and isinstance(cached.get("sha256"), str):
            return str(cached["sha256"])
        value = sha256(resolved)
        require(_stat_identity(resolved) == identity, "input changed while hashing: " + str(resolved))
        self.rows[str(resolved)] = {"stat": identity, "sha256": value}
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.rows, sort_keys=True))
        os.replace(temporary, self.path)
        return value


def _stat_identity(path: Path) -> list[Json]:
    state = path.stat()
    return [state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns]


def _read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


class StageCache:
    """One directory per (stage, identity); outputs are hard-linked (read-only) into each run that uses them."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.memo = HashMemo(root / "hash-memo.json")

    def input(self, path: Path) -> Record:
        return {"sha256": self.memo.sha256(path), "bytes": path.stat().st_size}

    def run(self, name: str, identity: Record, outputs: tuple[str, ...], destination: Path,
            produce: Callable[[Path], None]) -> Record:
        key = digest({"stage": name, "identity": identity})
        entry = self.root / name / key[:32]
        pending = entry.with_name(entry.name + ".pending")
        receipt_path = entry / RECEIPT
        hit = receipt_path.is_file() and not pending.exists()
        if hit:
            receipt = read_json(receipt_path)
            require(receipt.get("schema") == SCHEMA and receipt.get("key") == key and receipt.get("identity") == identity,
                    "stage cache identity collision: " + str(entry))
            for name_, expected in record(receipt.get("outputs")).items():
                row = record(expected)
                path = entry / name_
                require(path.is_file() and path.stat().st_size == row.get("bytes") and sha256(path) == row.get("sha256"),
                        "stage cache entry changed after publication: " + str(path))
        else:
            if entry.exists():
                require(pending.exists(), "unowned stage cache entry: " + str(entry))
                shutil.rmtree(entry)
            entry.parent.mkdir(parents=True, exist_ok=True)
            pending.write_text(key + "\n")
            entry.mkdir()
            started = time.monotonic()
            produce(entry)
            seconds = round(time.monotonic() - started, 3)
            files: Record = {}
            for output in outputs:
                path = entry / output
                require(path.is_file(), f"stage {name} did not produce {output}")
                files[output] = {"sha256": sha256(path), "bytes": path.stat().st_size}
                _read_only(path)
            receipt = {"schema": SCHEMA, "version": 1, "stage": name, "key": key, "identity": identity,
                       "outputs": files, "seconds": seconds,
                       "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}
            write_json(receipt_path, receipt)
            _read_only(receipt_path)
            pending.unlink()
        destination.mkdir(parents=True, exist_ok=True)
        for output in (*outputs, RECEIPT):
            target = destination / (output if output != RECEIPT else f"{name}.{RECEIPT}")
            require(not target.exists(), "stage output already present in the run: " + str(target))
            os.link(entry / output, target)
        return {"stage": name, "key": key, "cache_hit": hit, "entry": str(entry),
                "produced_seconds": receipt.get("seconds"), "outputs": receipt.get("outputs")}
