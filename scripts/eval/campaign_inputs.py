from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import BinaryIO, Final

from eval_common import (
    EvaluationError,
    Record,
    read_json,
    require,
    sha256,
    text,
    write_json,
)

WIDTHS: Final = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_exact(stream: BinaryIO, count: int) -> bytes:
    value = stream.read(count)
    require(len(value) == count, "truncated GGUF metadata")
    return value


def string(stream: BinaryIO) -> bytes:
    size = int.from_bytes(read_exact(stream, 8), "little")
    require(size <= 64 * 1024 * 1024, "oversized GGUF metadata string")
    return read_exact(stream, size)


def skip_value(stream: BinaryIO, kind: int) -> None:
    if kind in WIDTHS:
        read_exact(stream, WIDTHS[kind])
    elif kind == 8:
        string(stream)
    elif kind == 9:
        element, count = struct.unpack("<IQ", read_exact(stream, 12))
        require(element != 9 and count <= 100_000_000, "unsupported GGUF array")
        if element in WIDTHS:
            read_exact(stream, WIDTHS[element] * count)
        else:
            require(element == 8, "unsupported GGUF array element")
            for _ in range(count):
                string(stream)
    else:
        raise EvaluationError("unsupported GGUF metadata type")


def model_metadata(path: Path) -> tuple[str, str, int, int]:
    """Hash sorted exact tokenizer.* GGUF key/type/value bytes, excluding tensor weights."""
    tokenizer: dict[str, bytes] = {}
    fields: dict[str, bytes] = {}
    with path.open("rb") as stream:
        require(read_exact(stream, 4) == b"GGUF", "model is not little-endian GGUF")
        version, _, count = struct.unpack("<IQQ", read_exact(stream, 20))
        require(version in (2, 3) and count < 100_000, "unsupported GGUF header")
        for _ in range(count):
            start = stream.tell()
            name = string(stream).decode("utf-8")
            kind = int.from_bytes(read_exact(stream, 4), "little")
            value_start = stream.tell()
            skip_value(stream, kind)
            end = stream.tell()
            if name.startswith("tokenizer."):
                require(name not in tokenizer, "duplicate GGUF tokenizer key")
                stream.seek(start)
                tokenizer[name] = read_exact(stream, end - start)
            elif name in ("general.architecture", "general.file_type") or name.endswith(
                    (".block_count", ".embedding_length", ".feed_forward_length", ".attention.head_count")):
                stream.seek(value_start)
                fields[name] = read_exact(stream, end - value_start)
            stream.seek(end)
    require("tokenizer.ggml.tokens" in tokenizer, "missing native tokenizer metadata")
    raw_arch = fields.get("general.architecture", b"")
    require(len(raw_arch) >= 8, "missing model architecture")
    architecture = raw_arch[8:].decode("utf-8")
    require(architecture in ("gpt2", "llama"), "unsupported campaign model architecture")
    block_count = int.from_bytes(fields.get(architecture + ".block_count", b""), "little")
    file_type = int.from_bytes(fields.get("general.file_type", b""), "little")
    require(block_count > 0, "missing model block count")
    dimensions = tuple(int.from_bytes(fields.get(architecture + suffix, b""), "little")
                       for suffix in (".embedding_length", ".feed_forward_length", ".attention.head_count"))
    require(dimensions == ((768, 3072, 12) if architecture == "gpt2" else (2048, 8192, 32)),
            "model width/feedforward/head dimensions do not match requested campaign model")
    return hashlib.sha256(b"".join(tokenizer[key] for key in sorted(tokenizer))).hexdigest(), architecture, block_count, file_type


def dataset_input(manifest_path: Path | None, default: Path, output: Path) -> Path:
    if manifest_path is None:
        path = default.resolve(strict=True)
        value: Record = {"dataset": "WikiText-2", "split": "test", "path": str(path),
                         "sha256": sha256(path)}
    else:
        source = manifest_path.resolve(strict=True)
        value = read_json(source)
        path = (source.parent / text(value, "path")).resolve(strict=True)
        require(value.get("dataset") == "WikiText-2" and value.get("split") == "test",
                "campaign requires WikiText-2 test manifest")
        require(value.get("sha256") == sha256(path), "dataset manifest hash mismatch")
    write_json(output, value)
    return path


def checksums(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    with (root / "SHA256SUMS").open("x", encoding="utf-8") as stream:
        for path in files:
            stream.write(sha256(path) + "  " + str(path.relative_to(root)) + "\n")
